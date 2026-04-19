import torch
from transformers import pipeline
import re

_refinement_pipeline = None

def get_refinement_pipeline():
    global _refinement_pipeline
    if _refinement_pipeline is None:
        print("Loading refinement model (wav2vec2-large-xlsr-53-portuguese)...")
        _refinement_pipeline = pipeline(
            "automatic-speech-recognition",
            model="facebook/wav2vec2-large-xlsr-53-portuguese",
            device=0 if torch.cuda.is_available() else -1
        )
    return _refinement_pipeline

def refine_numbers_in_segment(audio_segment, current_text):
    """
    Second pass to refine numbers/CPF.
    Uses wav2vec2 which is often better at literal digit recognition in PT-BR.
    """
    # Simple heuristic: if text has digits or words that sound like digits
    digit_keywords = ["cpf", "número", "documento", "telefone", "zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove"]
    
    should_refine = any(kw in current_text.lower() for kw in digit_keywords) or re.search(r'\d', current_text)
    
    if not should_refine:
        return current_text

    pipe = get_refinement_pipeline()
    # wav2vec2 expects 16kHz
    result = pipe(audio_segment)
    refined_text = result["text"].lower()
    
    # Simple logic: if refined text has more digits/number-like patterns, consider it.
    # For now, let's just return it to see the difference in logs or use it to merge.
    # In a real scenario, we'd use this to fix specific parts.
    return refined_text
