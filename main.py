import os
import shutil
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline as transformers_pipeline,
)

# Mitigate CUDA memory fragmentation on long runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Enforce GPU-only execution.
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this project. "
        "No compatible CUDA device was detected."
    )

device = torch.device("cuda")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

# Define directories
AUDIO_DIR = Path("audio")
TRANSCRIPTION_DIR = Path("transcription")
PROCESSED_DIR = Path("processed")
NOTES_DIR = Path("notes")
NOTE_PROMPT_FILE = Path(os.getenv("NOTE_PROMPT_FILE", "prompt_nota_atendimento.txt"))
RATING_INPUT_DIR = Path(os.getenv("RATING_INPUT_DIR", "transcription"))

# Supported audio formats
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
SUPPORTED_TEXT_FORMATS = {'.txt', '.md'}
DIARIZATION_MODEL_ID = os.getenv("DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-community-1")
FALLBACK_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
DEFAULT_NUM_SPEAKERS = int(os.getenv("NUM_SPEAKERS", "2"))
MERGE_MAX_GAP_SECONDS = 0.4
RATING_MODEL_ID = os.getenv("RATING_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
RATING_MAX_NEW_TOKENS = int(os.getenv("RATING_MAX_NEW_TOKENS", "700"))
RATING_QUANTIZATION = os.getenv("RATING_QUANTIZATION", "4bit").strip().lower()
RUN_STAGE = os.getenv("RUN_STAGE", "audio").strip().lower()


def configure_torch_checkpoint_loading():
    """Compatibility mode for trusted legacy checkpoints with torch>=2.6."""
    # pyannote legacy checkpoints may require pickle objects blocked by
    # torch>=2.6 default (weights_only=True). We only use trusted HF models here.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion

        add_safe_globals([TorchVersion])
    except Exception:
        # Fallback via env var above is still effective when safe globals API
        # changes across torch versions.
        pass


configure_torch_checkpoint_loading()


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
            "Pass a real Hugging Face token via HF_TOKEN and "
            "accept the model terms at https://hf.co/pyannote/speaker-diarization-community-1"
        )

    pipeline.to(device)
    return pipeline


diarization_pipeline = None
whisper_pipeline = None
rating_pipeline = None


def ensure_audio_models_loaded():
    """Load diarization + ASR models once, only when audio stage is executed."""
    global diarization_pipeline
    global whisper_pipeline

    if diarization_pipeline is None:
        diarization_pipeline = load_diarization_pipeline()

    if whisper_pipeline is None:
        # Load Whisper large-v3-turbo model for transcription (load once)
        print("Loading Whisper large-v3-turbo model...")
        model_id = "openai/whisper-large-v3-turbo"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
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
            torch_dtype=torch.float16,
            device=device,
            return_timestamps=True
        )


def ensure_rating_model_loaded():
    """Load local text-generation model for atendimento rating."""
    global rating_pipeline

    if rating_pipeline is not None:
        return

    if RATING_QUANTIZATION not in {"4bit", "none"}:
        raise ValueError("RATING_QUANTIZATION must be '4bit' or 'none'.")

    print(f"Loading rating model ({RATING_MODEL_ID})...")
    print(f"Rating quantization mode: {RATING_QUANTIZATION}")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(RATING_MODEL_ID, token=hf_token)
    torch.cuda.empty_cache()

    model_kwargs = {
        "token": hf_token,
        "low_cpu_mem_usage": True,
    }
    pipeline_kwargs = {}

    if RATING_QUANTIZATION == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16
        pipeline_kwargs["device"] = device

    model = AutoModelForCausalLM.from_pretrained(
        RATING_MODEL_ID,
        **model_kwargs,
    )
    if RATING_QUANTIZATION != "4bit":
        model.to(device)

    rating_pipeline = transformers_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs,
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


def load_note_prompt_template():
    """Load note prompt rules from NOTE_PROMPT_FILE, with sensible defaults."""
    if NOTE_PROMPT_FILE.exists():
        return NOTE_PROMPT_FILE.read_text(encoding="utf-8").strip()

    return (
        "Você é um avaliador de qualidade de atendimento.\n"
        "Analise a conversa e gere uma nota de 0 a 10.\n"
        "Considere clareza, empatia, solução do problema e objetividade.\n"
        "Responda em português com:\n"
        "1) Nota final\n"
        "2) Principais pontos positivos\n"
        "3) Principais pontos de melhoria\n"
        "4) Resumo final em 3 linhas."
    )


def call_rating_model(prompt):
    """Generate atendimento rating text using local transformers model."""
    ensure_rating_model_loaded()

    outputs = rating_pipeline(
        prompt,
        max_new_tokens=RATING_MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
    )
    if not outputs:
        raise RuntimeError("Modelo de rating retornou saída vazia.")

    generated = outputs[0].get("generated_text", "")
    if isinstance(generated, list):
        # Chat pipeline variants can return a message list.
        generated = generated[-1].get("content", "") if generated else ""

    generated = str(generated).strip()
    if not generated:
        raise RuntimeError("Modelo de rating retornou texto vazio.")
    return generated


def build_note_prompt(text_content, rules_prompt):
    """Build final prompt joining rules + transcript text."""
    return (
        f"{rules_prompt}\n\n"
        "### Texto do atendimento\n"
        f"{text_content}\n\n"
        "### Saída esperada\n"
        "Aplique estritamente as regras e retorne somente a avaliação."
    )


def generate_note_from_processed_text(text_path, rules_prompt):
    """Generate atendimento note for one processed text file."""
    print(f"Generating atendimento note: {text_path.name}")
    content = text_path.read_text(encoding="utf-8").strip()
    if not content:
        print(f"- Skipping {text_path.name}: empty content")
        return False

    prompt = build_note_prompt(content, rules_prompt)
    note = call_rating_model(prompt)

    output_path = NOTES_DIR / f"{text_path.stem}.nota.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Nota de Atendimento\n\n")
        f.write(f"- **Origem:** {text_path.name}\n")
        f.write(f"- **Modelo:** {RATING_MODEL_ID}\n\n")
        f.write(note)
        f.write("\n")

    print(f"✓ Atendimento note saved to {output_path}")
    return True


def process_rating_texts():
    """Apply atendimento note prompt to text files in rating input directory."""
    text_files = sorted(
        [
            file_path for file_path in RATING_INPUT_DIR.glob("*")
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_TEXT_FORMATS
        ]
    )

    if not text_files:
        print(f"No text files found in {RATING_INPUT_DIR}/ for note generation")
        return 0, 0

    rules_prompt = load_note_prompt_template()
    success = 0
    failure = 0

    for text_file in text_files:
        try:
            if generate_note_from_processed_text(text_file, rules_prompt):
                success += 1
            else:
                failure += 1
        except Exception as exc:
            print(f"✗ Error generating note for {text_file.name}: {exc}")
            failure += 1

    return success, failure


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
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    if RUN_STAGE not in {"audio", "rating", "all"}:
        print("Invalid RUN_STAGE. Use: audio, rating, or all.")
        exit(1)

    run_audio_stage = RUN_STAGE in {"audio", "all"}
    run_rating_stage = RUN_STAGE in {"rating", "all"}

    audio_files = []
    if run_audio_stage:
        ensure_audio_models_loaded()

        # Find all audio files in the audio directory
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(AUDIO_DIR.glob(f"*{ext}"))

        if not audio_files:
            print(f"No audio files found in {AUDIO_DIR}/")
            print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
            exit(1)

        print(f"Found {len(audio_files)} audio file(s) to process\n")
    else:
        print("Transcription stage skipped (RUN_STAGE=rating).\n")

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
    if run_rating_stage:
        print(f"{'='*60}")
        print(f"Atendimento Note Stage ({RATING_INPUT_DIR}/ text files)")
        note_success, note_failed = process_rating_texts()
        print(f"Notes generated: {note_success}")
        print(f"Note failures: {note_failed}")
    else:
        print(f"{'='*60}")
        print("Atendimento Note Stage skipped (RUN_STAGE=audio)")
    print(f"{'='*60}\n")
