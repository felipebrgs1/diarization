import soundfile as sf
import torch
import numpy as np
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as transformers_pipeline

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load audio using soundfile to avoid torchcodec/FFmpeg dependency
waveform, sample_rate = sf.read("audio.wav", dtype='float32')
# Convert to torch tensor and add channel dimension if needed
waveform = torch.from_numpy(waveform).float()
if waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)  # Add channel dimension for mono audio
elif waveform.ndim == 2:
    waveform = waveform.T  # Transpose to (channel, time) format

# Initialize diarization pipeline and move to GPU
print("Loading diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=True  # Uses the token from `huggingface-cli login`
)
pipeline.to(device)

# Configure pipeline parameters for better speaker separation
# These parameters make the system more sensitive to speaker changes
from pyannote.audio.pipelines.utils import PipelineModel

# Adjust segmentation parameters to be more sensitive
if hasattr(pipeline, 'segmentation'):
    # Make the model more sensitive to short speech segments
    pipeline.segmentation.min_duration_on = 0.0
    pipeline.segmentation.min_duration_off = 0.0

# Adjust clustering threshold for better speaker separation
if hasattr(pipeline, 'klustering'):
    # Lower threshold can help separate speakers better (but may create more false speakers)
    # Default is usually around 0.7, we'll keep it conservative
    pass

# Run diarization with pre-loaded audio
print("Running speaker diarization...")
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# Load Whisper large-v3-turbo model for transcription
print("Loading Whisper large-v3-turbo model...")
model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

whisper_pipeline = transformers_pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=device,
    return_timestamps=True
)

# Transcribe full audio
print("Transcribing audio...")
result = whisper_pipeline("audio.wav", generate_kwargs={"language": "portuguese"})

# Create a list of segments with speaker labels
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # Find transcription segments that overlap with this diarization segment
    segment_text = []
    if "chunks" in result:
        for chunk in result["chunks"]:
            seg_start = chunk["timestamp"][0]
            seg_end = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else turn.end

            # Calculate overlap percentage - only include if significant overlap
            overlap_start = max(seg_start, turn.start)
            overlap_end = min(seg_end, turn.end)
            overlap_duration = max(0, overlap_end - overlap_start)
            chunk_duration = seg_end - seg_start

            # Only include chunk if at least 50% of it overlaps with this speaker turn
            if chunk_duration > 0 and (overlap_duration / chunk_duration) >= 0.5:
                segment_text.append(chunk["text"].strip())

    if segment_text:
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "text": " ".join(segment_text)
        })

# Post-process: Split segments that might contain speaker changes based on audio energy
def detect_speaker_changes_in_segment(seg, waveform, sample_rate, window_size=0.1):
    """Detect potential speaker changes within a segment based on energy patterns."""
    start_sample = int(seg["start"] * sample_rate)
    end_sample = int(seg["end"] * sample_rate)
    segment_audio = waveform[0, start_sample:end_sample].numpy()

    if len(segment_audio) == 0:
        return []

    # Calculate RMS energy in small windows
    window_samples = int(window_size * sample_rate)
    energies = []
    for i in range(0, len(segment_audio), window_samples):
        window = segment_audio[i:i + window_samples]
        if len(window) > 0:
            energy = np.sqrt(np.mean(window ** 2))
            energies.append(energy)

    if len(energies) < 3:
        return []

    # Find significant energy changes (potential speaker changes)
    split_points = []
    energies = np.array(energies)
    for i in range(1, len(energies) - 1):
        # Calculate relative change
        if energies[i-1] > 0:
            relative_change = abs(energies[i] - energies[i-1]) / energies[i-1]
            # If energy changes by more than 40% and stays different, mark as potential split
            if relative_change > 0.4:
                time_offset = (i * window_samples) / sample_rate
                split_points.append(seg["start"] + time_offset)

    return split_points

# Apply speaker change detection and refine segments
refined_segments = []
for seg in segments:
    split_points = detect_speaker_changes_in_segment(seg, waveform, sample_rate)

    if not split_points:
        refined_segments.append(seg)
    else:
        # Split segment at detected points
        prev_time = seg["start"]
        words = seg["text"].split()
        words_per_split = max(1, len(words) // (len(split_points) + 1))

        for i, split_time in enumerate(split_points):
            if split_time > prev_time and split_time < seg["end"]:
                text_portion = " ".join(words[i*words_per_split:(i+1)*words_per_split])
                if text_portion.strip():
                    refined_segments.append({
                        "start": prev_time,
                        "end": split_time,
                        "speaker": seg["speaker"],
                        "text": text_portion
                    })
                prev_time = split_time

        # Add remaining portion
        remaining_text = " ".join(words[len(split_points)*words_per_split:])
        if remaining_text.strip():
            refined_segments.append({
                "start": prev_time,
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text": remaining_text
            })

# Merge consecutive segments from the same speaker with stricter criteria
merged_segments = []
for seg in refined_segments:
    if merged_segments and merged_segments[-1]["speaker"] == seg["speaker"] and \
       (seg["start"] - merged_segments[-1]["end"]) < 0.3:  # Less than 0.3 second gap (stricter)
        # Merge with previous segment
        merged_segments[-1]["end"] = seg["end"]
        merged_segments[-1]["text"] += " " + seg["text"]
    else:
        merged_segments.append(seg)

# Calculate duration from audio
duration = len(waveform[0]) / sample_rate

# Generate markdown output
print("\nGenerating transcript.md...")
with open("transcript.md", "w", encoding="utf-8") as f:
    f.write("# Transcrição de Áudio\n\n")
    f.write("## Informações\n\n")
    f.write(f"- **Arquivo:** audio.wav\n")
    f.write(f"- **Duração:** {duration:.1f}s\n")
    f.write(f"- **Idioma:** {result.get('language', 'pt')}\n\n")
    f.write("## Transcrição\n\n")

    for seg in merged_segments:
        minutes = int(seg["start"] // 60)
        seconds = int(seg["start"] % 60)
        f.write(f"**[{minutes:02d}:{seconds:02d}] {seg['speaker']}:**  \n")
        f.write(f"{seg['text']}\n\n")

print("Transcription saved to transcript.md")
