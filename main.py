import soundfile as sf
import torch
import numpy as np
import os
import shutil
from pathlib import Path
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as transformers_pipeline
from datetime import datetime

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define directories
AUDIO_DIR = Path("audio")
TRANSCRIPTION_DIR = Path("transcription")
PROCESSED_DIR = Path("processed")

# Supported audio formats
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}

def load_audio(audio_path):
    """Load audio file and convert to torch tensor."""
    waveform, sample_rate = sf.read(audio_path, dtype='float32')
    # Convert to torch tensor and add channel dimension if needed
    waveform = torch.from_numpy(waveform).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension for mono audio
    elif waveform.ndim == 2:
        waveform = waveform.T  # Transpose to (channel, time) format
    return waveform, sample_rate

# Initialize diarization pipeline and move to GPU (load once)
print("Loading diarization pipeline...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=True  # Uses the token from `huggingface-cli login`
)
diarization_pipeline.to(device)

# Configure pipeline parameters for better speaker separation
from pyannote.audio.pipelines.utils import PipelineModel

# Adjust segmentation parameters to be more sensitive
if hasattr(diarization_pipeline, 'segmentation'):
    diarization_pipeline.segmentation.min_duration_on = 0.0
    diarization_pipeline.segmentation.min_duration_off = 0.0

# Load Whisper large-v3-turbo model for transcription (load once)
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

def process_audio_file(audio_path):
    """Process a single audio file and return the transcript."""
    print(f"\n{'='*60}")
    print(f"Processing: {audio_path.name}")
    print(f"{'='*60}")

    try:
        # Load audio
        print("Loading audio...")
        waveform, sample_rate = load_audio(audio_path)

        # Run diarization
        print("Running speaker diarization...")
        diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # Transcribe full audio
        print("Transcribing audio...")
        result = whisper_pipeline(str(audio_path), generate_kwargs={"language": "portuguese"})

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

        # Apply speaker change detection and refine segments
        print("Refining speaker segments...")
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
        print("Merging segments...")
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

        # Generate output filename
        output_filename = audio_path.stem + ".md"
        output_path = TRANSCRIPTION_DIR / output_filename

        # Generate markdown output
        print(f"Generating {output_filename}...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Transcrição de Áudio\n\n")
            f.write("## Informações\n\n")
            f.write(f"- **Arquivo:** {audio_path.name}\n")
            f.write(f"- **Duração:** {duration:.1f}s\n")
            f.write(f"- **Idioma:** {result.get('language', 'pt')}\n\n")
            f.write("## Transcrição\n\n")

            for seg in merged_segments:
                minutes = int(seg["start"] // 60)
                seconds = int(seg["start"] % 60)
                f.write(f"**[{minutes:02d}:{seconds:02d}] {seg['speaker']}:**  \n")
                f.write(f"{seg['text']}\n\n")

        print(f"✓ Transcription saved to {output_path}")

        # Move processed audio to processed directory
        processed_path = PROCESSED_DIR / audio_path.name
        shutil.move(str(audio_path), str(processed_path))
        print(f"✓ Audio moved to {processed_path}")

        return True

    except Exception as e:
        print(f"✗ Error processing {audio_path.name}: {str(e)}")
        return False

# Main execution
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Audio Transcription & Diarization System")
    print(f"{'='*60}\n")

    # Find all audio files in the audio directory
    audio_files = []
    for ext in SUPPORTED_FORMATS:
        audio_files.extend(AUDIO_DIR.glob(f"*{ext}"))

    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}/")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        exit(1)

    print(f"Found {len(audio_files)} audio file(s) to process\n")

    # Process each audio file
    successful = 0
    failed = 0
    start_time = datetime.now()

    for audio_file in audio_files:
        if process_audio_file(audio_file):
            successful += 1
        else:
            failed += 1

    # Summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")
