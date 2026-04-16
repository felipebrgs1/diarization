import os
import torch
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No compatible CUDA device was detected.")

DEVICE = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

TORCH_DTYPE = torch.float16

AUDIO_DIR = Path("audio")
TRANSCRIPTION_DIR = Path("transcription")
PROCESSED_DIR = Path("processed")
NOTES_DIR = Path("notes")
NOTE_PROMPT_FILE = Path("src/prompts/atendimento_rating.md")
RATING_INPUT_DIR = Path("transcription")

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
SUPPORTED_TEXT_FORMATS = {".txt", ".md"}

WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
FALLBACK_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
DEFAULT_NUM_SPEAKERS = 2
MERGE_MAX_GAP_SECONDS = 0.4

RATING_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
RATING_MAX_NEW_TOKENS = 700
RATING_QUANTIZATION = "4bit"
