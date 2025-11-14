import soundfile as sf
import torch
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

            # Check if segments overlap
            if (seg_start <= turn.end and seg_end >= turn.start):
                segment_text.append(chunk["text"].strip())

    if segment_text:
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "text": " ".join(segment_text)
        })

# Merge consecutive segments from the same speaker
merged_segments = []
for seg in segments:
    if merged_segments and merged_segments[-1]["speaker"] == seg["speaker"] and \
       (seg["start"] - merged_segments[-1]["end"]) < 1.0:  # Less than 1 second gap
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
