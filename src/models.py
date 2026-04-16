import os
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
from src.config import (
    DEVICE,
    TORCH_DTYPE,
    WHISPER_MODEL_ID,
    DIARIZATION_MODEL_ID,
    FALLBACK_DIARIZATION_MODEL_ID,
    RATING_MODEL_ID,
    RATING_QUANTIZATION,
)


def configure_torch_checkpoint_loading():
    """Compatibility mode for trusted legacy checkpoints with torch>=2.6."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion

        add_safe_globals([TorchVersion])
    except Exception:
        pass


configure_torch_checkpoint_loading()


def load_diarization_pipeline():
    """Load diarization pipeline with safe fallback."""
    print(f"Loading diarization pipeline ({DIARIZATION_MODEL_ID})...")

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL_ID)
    except TypeError as exc:
        if (
            "unexpected keyword argument 'plda'" in str(exc)
            and DIARIZATION_MODEL_ID != FALLBACK_DIARIZATION_MODEL_ID
        ):
            print(
                f"Model '{DIARIZATION_MODEL_ID}' is incompatible. Falling back to '{FALLBACK_DIARIZATION_MODEL_ID}'."
            )
            pipeline = Pipeline.from_pretrained(FALLBACK_DIARIZATION_MODEL_ID)
        else:
            raise

    if pipeline is None:
        raise RuntimeError("Failed to load diarization pipeline.")

    pipeline.to(DEVICE)
    return pipeline


def load_whisper_pipeline():
    """Load Whisper large-v3-turbo pipeline for transcription."""
    print(f"Loading Whisper ({WHISPER_MODEL_ID})...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

    return transformers_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
        return_timestamps=True,
    )


def load_rating_pipeline():
    """Load local text-generation model for atendimento rating."""
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
