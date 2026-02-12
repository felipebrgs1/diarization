import os
import shutil
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as transformers_pipeline

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define directories
AUDIO_DIR = Path("audio")
TRANSCRIPTION_DIR = Path("transcription")
PROCESSED_DIR = Path("processed")

# Supported audio formats
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
DIARIZATION_MODEL_ID = os.getenv("DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-community-1")
FALLBACK_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
DEFAULT_NUM_SPEAKERS = int(os.getenv("NUM_SPEAKERS", "2"))
MERGE_MAX_GAP_SECONDS = 0.4

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

def _load_pipeline_from_pretrained(model_id, hf_token):
    """Load pyannote pipeline handling auth kwarg differences across versions."""
    auth_attempts = []
    if hf_token:
        auth_attempts = [
            {"use_auth_token": hf_token},
            {"token": hf_token},
        ]
    else:
        auth_attempts = [
            {"use_auth_token": True},
            {},
        ]

    last_signature_error = None
    for kwargs in auth_attempts:
        try:
            return Pipeline.from_pretrained(model_id, **kwargs)
        except TypeError as exc:
            error_text = str(exc)
            if "unexpected keyword argument 'token'" in error_text or \
               "unexpected keyword argument 'use_auth_token'" in error_text:
                last_signature_error = exc
                continue
            raise

    if last_signature_error:
        raise last_signature_error

    raise RuntimeError(f"Unable to load pipeline {model_id}")


def load_diarization_pipeline():
    """Load diarization pipeline with safe fallback for legacy pyannote versions."""
    print(f"Loading diarization pipeline ({DIARIZATION_MODEL_ID})...")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    try:
        pipeline = _load_pipeline_from_pretrained(DIARIZATION_MODEL_ID, hf_token)
    except TypeError as exc:
        # community-1 can be incompatible with pyannote.audio 3.x.
        if (
            "unexpected keyword argument 'plda'" in str(exc)
            and DIARIZATION_MODEL_ID != FALLBACK_DIARIZATION_MODEL_ID
        ):
            print(
                f"Model '{DIARIZATION_MODEL_ID}' is incompatible with installed pyannote.audio. "
                f"Falling back to '{FALLBACK_DIARIZATION_MODEL_ID}'."
            )
            pipeline = _load_pipeline_from_pretrained(FALLBACK_DIARIZATION_MODEL_ID, hf_token)
        else:
            raise

    if pipeline is None:
        raise RuntimeError(
            "Failed to load diarization pipeline. "
            "For Docker runs, pass a real Hugging Face token via HF_TOKEN and "
            "accept the model terms at https://hf.co/pyannote/speaker-diarization-community-1"
        )

    pipeline.to(device)
    return pipeline


# Initialize diarization pipeline and move to GPU (load once)
diarization_pipeline = load_diarization_pipeline()

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

def get_diarization_annotation(diarization_output):
    """Extract best available diarization annotation from pipeline output."""
    if hasattr(diarization_output, "exclusive_speaker_diarization"):
        return diarization_output.exclusive_speaker_diarization
    if hasattr(diarization_output, "speaker_diarization"):
        return diarization_output.speaker_diarization
    return diarization_output


def build_speaker_turns(diarization_annotation):
    """Convert diarization annotation to a sorted list of speaker turns."""
    turns = []
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })
    turns.sort(key=lambda x: x["start"])
    return turns


def segment_overlap(start_a, end_a, start_b, end_b):
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_speaker(chunk_start, chunk_end, speaker_turns):
    """Assign speaker by maximum overlap; fallback to nearest turn."""
    best_speaker = None
    best_overlap = 0.0

    for turn in speaker_turns:
        overlap = segment_overlap(chunk_start, chunk_end, turn["start"], turn["end"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    if best_speaker is not None:
        return best_speaker

    if not speaker_turns:
        return "SPEAKER_00"

    mid = (chunk_start + chunk_end) / 2
    nearest = min(
        speaker_turns,
        key=lambda turn: 0.0
        if turn["start"] <= mid <= turn["end"]
        else min(abs(turn["start"] - mid), abs(turn["end"] - mid)),
    )
    return nearest["speaker"]


def build_segments(chunks, speaker_turns):
    """Create speaker-attributed text segments from ASR chunks."""
    segments = []
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "").strip()
        timestamp = chunk.get("timestamp")

        if not chunk_text or not timestamp or timestamp[0] is None:
            continue

        chunk_start = float(timestamp[0])
        chunk_end = timestamp[1]

        if chunk_end is None:
            next_start = None
            if idx + 1 < len(chunks):
                next_ts = chunks[idx + 1].get("timestamp")
                if next_ts:
                    next_start = next_ts[0]
            chunk_end = float(next_start) if next_start is not None else chunk_start + 0.5
        else:
            chunk_end = float(chunk_end)

        if chunk_end <= chunk_start:
            chunk_end = chunk_start + 0.01

        speaker = assign_speaker(chunk_start, chunk_end, speaker_turns)
        segments.append({
            "start": chunk_start,
            "end": chunk_end,
            "speaker": speaker,
            "text": chunk_text,
        })

    return segments


def merge_consecutive_segments(segments, max_gap_seconds=MERGE_MAX_GAP_SECONDS):
    """Merge adjacent segments of same speaker when time gap is short."""
    merged = []
    for seg in segments:
        if (
            merged
            and merged[-1]["speaker"] == seg["speaker"]
            and (seg["start"] - merged[-1]["end"]) <= max_gap_seconds
        ):
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            merged[-1]["text"] = f"{merged[-1]['text']} {seg['text']}".strip()
        else:
            merged.append(seg.copy())
    return merged

def process_audio_file(audio_path):
    """Process a single audio file and return the transcript."""
    print(f"\n{'='*60}")
    print(f"Processing: {audio_path.name}")
    print(f"{'='*60}")

    try:
        # Load audio
        print("Loading audio...")
        waveform, sample_rate = load_audio(audio_path)
        duration = len(waveform[0]) / sample_rate

        # Run diarization
        print("Running speaker diarization...")
        diarization_output = diarization_pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=DEFAULT_NUM_SPEAKERS,
        )
        diarization_annotation = get_diarization_annotation(diarization_output)
        speaker_turns = build_speaker_turns(diarization_annotation)

        # Transcribe full audio
        print("Transcribing audio...")
        result = whisper_pipeline(str(audio_path), generate_kwargs={"language": "portuguese"})

        # Build speaker-attributed segments directly from ASR chunks
        print("Aligning transcription with diarization...")
        chunks = result.get("chunks", [])
        segments = build_segments(chunks, speaker_turns)

        if not segments and result.get("text", "").strip():
            fallback_speaker = speaker_turns[0]["speaker"] if speaker_turns else "SPEAKER_00"
            segments = [{
                "start": 0.0,
                "end": duration,
                "speaker": fallback_speaker,
                "text": result["text"].strip(),
            }]

        # Merge consecutive segments from the same speaker with conservative gap
        print("Merging segments...")
        merged_segments = merge_consecutive_segments(segments)

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

    TRANSCRIPTION_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
