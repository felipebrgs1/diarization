import os
import torch
import whisperx
from dotenv import load_dotenv
from src.config import (
    DEVICE,
    WHISPER_MODEL_ID,
    DIARIZATION_MODEL_ID,
    RATING_MODEL_ID,
    RATING_QUANTIZATION,
    TORCH_DTYPE,
    ALIGN_MODEL_LANGUAGE_CODE,
)

# Load environment variables from .env
load_dotenv()

# Use token found in ~/.cache/huggingface/token or environment
HF_TOKEN = os.getenv("HF_TOKEN")

def configure_torch_checkpoint_loading():
    """Compatibility mode for trusted legacy checkpoints with torch>=2.6."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion

        import warnings
        warnings.filterwarnings(
            "ignore",
            message="std\\(\\): degrees of freedom is <= 0",
            category=UserWarning,
            module="pyannote.audio.models.blocks.pooling",
        )
        add_safe_globals([TorchVersion])
    except Exception:
        pass


configure_torch_checkpoint_loading()


def load_diarization_pipeline():
    """Load WhisperX DiarizationPipeline (wraps pyannote-audio)."""
    from whisperx.diarize import DiarizationPipeline
    print(f"Loading diarization pipeline ({DIARIZATION_MODEL_ID})...")
    return DiarizationPipeline(
        model_name=DIARIZATION_MODEL_ID, 
        token=HF_TOKEN, 
        device=DEVICE.type
    )


def load_whisper_pipeline(language=ALIGN_MODEL_LANGUAGE_CODE):
    """Load WhisperX model for transcription."""
    print(f"Loading WhisperX ({WHISPER_MODEL_ID}) for language '{language}'...")
    return whisperx.load_model(
        WHISPER_MODEL_ID, 
        device=DEVICE.type, 
        compute_type="int8",
        language=language
    )


def load_align_model(language_code):
    """Load WhisperX alignment model."""
    print(f"Loading alignment model for '{language_code}'...")
    return whisperx.load_align_model(language_code=language_code, device=DEVICE.type)


def load_rating_pipeline():
    """Load local text-generation model for atendimento rating."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline as transformers_pipeline,
    )
    
    if RATING_QUANTIZATION not in {"4bit", "none"}:
        raise ValueError("RATING_QUANTIZATION must be '4bit' or 'none'.")

    print(f"Loading rating model ({RATING_MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(RATING_MODEL_ID)
    torch.cuda.empty_cache()

    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
    pipeline_kwargs = {}

    # Bitsandbytes quantization requires CUDA
    actual_quantization = RATING_QUANTIZATION
    if DEVICE.type == "cpu" and actual_quantization == "4bit":
        print(
            "Warning: 4bit quantization is only supported on GPU. Using no quantization for CPU."
        )
        actual_quantization = "none"

    if actual_quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = TORCH_DTYPE
        pipeline_kwargs["device"] = DEVICE

    model = AutoModelForCausalLM.from_pretrained(RATING_MODEL_ID, **model_kwargs)
    if actual_quantization != "4bit":
        model.to(DEVICE)

    return transformers_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs,
    )
