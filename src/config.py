import os
import torch
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No compatible CUDA device was detected.")

DEVICE = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

TORCH_DTYPE = torch.float16

BASE_DIR = Path("audio")
AUDIO_DIR = BASE_DIR / "audio"
TRANSCRIPTION_DIR = BASE_DIR / "transcription"
PROCESSED_DIR = BASE_DIR / "processed"
NOTES_DIR = BASE_DIR / "notes"
NOTE_PROMPT_FILE = Path("src/prompts/atendimento_rating.md")
RATING_INPUT_DIR = TRANSCRIPTION_DIR

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
SUPPORTED_TEXT_FORMATS = {".txt", ".md"}

WHISPER_MODEL_ID = "openai/whisper-large-v3"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
FALLBACK_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
MERGE_MAX_GAP_SECONDS = 0.4

RATING_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
RATING_MAX_NEW_TOKENS = 2048
RATING_QUANTIZATION = "4bit"
