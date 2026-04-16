import os
from src.config import (
    NOTE_PROMPT_FILE, 
    NOTES_DIR, 
    RATING_MODEL_ID, 
    RATING_MAX_NEW_TOKENS,
    RATING_INPUT_DIR,
    SUPPORTED_TEXT_FORMATS
)
from src.models import load_rating_pipeline

# Lazy loading of global pipeline
_rating_pipeline = None

def get_rating_pipeline():
    global _rating_pipeline
    if _rating_pipeline is None:
        _rating_pipeline = load_rating_pipeline()
    return _rating_pipeline

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
    pipeline = get_rating_pipeline()

    outputs = pipeline(
        prompt,
        max_new_tokens=RATING_MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
    )
    if not outputs:
        raise RuntimeError("Modelo de rating retornou saída vazia.")

    generated = outputs[0].get("generated_text", "")
    if isinstance(generated, list):
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
