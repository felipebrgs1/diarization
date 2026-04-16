import shutil
from datetime import datetime
from pathlib import Path

from src.config import (
    BASE_DIR,
    AUDIO_DIR,
    TRANSCRIPTION_DIR,
    PROCESSED_DIR,
    NOTES_DIR,
    SUPPORTED_FORMATS,
    MERGE_MAX_GAP_SECONDS,
    RATING_INPUT_DIR,
    RATING_MODEL_ID,
)
from src.audio import load_audio
from src.models import load_diarization_pipeline, load_whisper_pipeline
from src.diarization import get_diarization_annotation, build_speaker_turns
from src.transcription import build_segments, merge_consecutive_segments
from src.rating import process_rating_texts

_diarization_pipeline = None
_whisper_pipeline = None


def get_pipelines():
    global _diarization_pipeline, _whisper_pipeline
    if _diarization_pipeline is None:
        _diarization_pipeline = load_diarization_pipeline()
    if _whisper_pipeline is None:
        _whisper_pipeline = load_whisper_pipeline()
    return _diarization_pipeline, _whisper_pipeline


def fmt_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def process_audio_file(audio_path: Path) -> bool:
    print(f"→ {audio_path.name}")

    try:
        diar_pipe, whisper_pipe = get_pipelines()
        waveform, sample_rate = load_audio(audio_path)
        duration = len(waveform[0]) / sample_rate

        diarization = diar_pipe(
            {"waveform": waveform, "sample_rate": sample_rate},
            min_speakers=1,
            max_speakers=8,
        )
        speaker_turns = build_speaker_turns(get_diarization_annotation(diarization))

        result = whisper_pipe(
            str(audio_path),
            generate_kwargs={
                "language": "pt",
                "task": "transcribe",
                "num_beams": 5,
                "temperature": 0.0,
            },
        )
        segments = build_segments(result.get("chunks", []), speaker_turns)

        if not segments and result.get("text", "").strip():
            fallback = speaker_turns[0]["speaker"] if speaker_turns else "SPEAKER_00"
            segments = [
                {
                    "start": 0.0,
                    "end": duration,
                    "speaker": fallback,
                    "text": result["text"].strip(),
                }
            ]

        merged = merge_consecutive_segments(
            segments, max_gap_seconds=MERGE_MAX_GAP_SECONDS
        )

        out_path = TRANSCRIPTION_DIR / (audio_path.stem + ".md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Transcrição\n\n")
            f.write(f"- **Arquivo:** {audio_path.name}\n")
            f.write(f"- **Duração:** {duration:.1f}s\n\n")
            for seg in merged:
                f.write(
                    f"**[{fmt_time(seg['start'])}] {seg['speaker']}:**  \n{seg['text']}\n\n"
                )

        shutil.move(str(audio_path), str(PROCESSED_DIR / audio_path.name))
        print(f"   ok: {out_path.name}")
        return True

    except Exception as e:
        print(f"   err: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rating", action="store_true", help="habilita geração de notas de atendimento"
    )
    parser.add_argument(
        "--rating-only",
        action="store_true",
        help="gera apenas notas de textos existentes",
    )
    args = parser.parse_args()

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTION_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    run_audio = not args.rating_only
    run_rating = args.rating or args.rating_only

    audio_files = []
    if run_audio:
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(AUDIO_DIR.glob(f"*{ext}"))
        if not audio_files and not args.rating:
            print("Nenhum áudio encontrado em audio/")
            raise SystemExit(1)

    start = datetime.now()
    success = failed = 0

    for audio_file in audio_files:
        if process_audio_file(audio_file):
            success += 1
        else:
            failed += 1

    elapsed = (datetime.now() - start).total_seconds()

    if run_audio and audio_files:
        print(f"\nDone: {success} ok, {failed} fail, {elapsed:.1f}s")

    if run_rating:
        print(f"\nRating ({RATING_INPUT_DIR}) — {RATING_MODEL_ID}")
        note_ok, note_fail = process_rating_texts()
        print(f"Done: {note_ok} ok, {note_fail} fail")
