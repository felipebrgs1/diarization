import shutil
import torch
from datetime import datetime
from pathlib import Path
import whisperx
from src.config import (
    DEVICE,
    BASE_DIR,
    AUDIO_DIR,
    TRANSCRIPTION_DIR,
    PROCESSED_DIR,
    NOTES_DIR,
    SUPPORTED_FORMATS,
    MERGE_MAX_GAP_SECONDS,
    RATING_INPUT_DIR,
    RATING_MODEL_ID,
    WHISPER_BATCH_SIZE,
    PREPROCESSED_DIR,
    ALIGN_MODEL_LANGUAGE_CODE,
)
from src.models import (
    load_diarization_pipeline,
    load_whisper_pipeline,
    load_align_model,
)
from src.transcription import merge_consecutive_segments
from src.rating import process_rating_texts
from src.audio_processing import preprocess_audio
from src.config import SPEAKER_MAPPING
import json
from src.refinement import refine_numbers_in_segment
import multiprocessing

# Models will be loaded and unloaded locally within stages to ensure memory release
_align_model_cache = {}


def get_align_model(language_code):
    global _align_model_cache
    if language_code not in _align_model_cache:
        _align_model_cache[language_code] = load_align_model(language_code)
    return _align_model_cache[language_code]


def fmt_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def fmt_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"


def transcribe_audio_file(audio_path: Path, whisper_pipe) -> bool:
    """Stage 1: Transcription and Alignment"""
    print(f"→ Transcrevendo {audio_path.name}")
    try:
        audio = whisperx.load_audio(str(audio_path))
        
        print(f"   [1/2] Transcrevendo ({ALIGN_MODEL_LANGUAGE_CODE}) (WhisperX)...")
        result = whisper_pipe.transcribe(
            audio, 
            batch_size=WHISPER_BATCH_SIZE,
            language=ALIGN_MODEL_LANGUAGE_CODE,
            chunk_size=10
        )
        torch.cuda.empty_cache()

        print(f"   [2/2] Alinhando transcrição ({result['language']})...")
        model_a, metadata = get_align_model(result["language"])
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE.type,
            return_char_alignments=False,
        )
        torch.cuda.empty_cache()

        # Save intermediate result
        temp_json = audio_path.with_suffix(".whisper.json")
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"   ✗ erro na transcrição de {audio_path.name}: {e}")
        return False


def diarize_and_finalize(audio_path: Path, diar_pipe) -> bool:
    """Stage 2: Diarization and Export"""
    print(f"→ Diarizando {audio_path.name}")
    try:
        temp_json = audio_path.with_suffix(".whisper.json")
        if not temp_json.exists():
            print(f"   ✗ erro: transcrição intermediária não encontrada para {audio_path.name}")
            return False
            
        with open(temp_json, "r", encoding="utf-8") as f:
            result = json.load(f)

        audio = whisperx.load_audio(str(audio_path))
        duration = len(audio) / 16000 

        print("   [1/3] Identificando falantes (Diarização 4.0)...")
        diarize_segments = diar_pipe(audio, min_speakers=2, max_speakers=2)
        torch.cuda.empty_cache()

        print("   [2/3] Atribuindo falantes...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        print("   [3/3] Refinando e Exportando...")
        # Refinamento opcional
        for seg in result["segments"]:
            if 0.5 < (seg["end"] - seg["start"]) < 15.0:
                start_sample = int(seg["start"] * 16000)
                end_sample = int(seg["end"] * 16000)
                audio_segment = audio[start_sample:end_sample]
                refine_numbers_in_segment(audio_segment, seg["text"])

        # Heurística de falantes
        found_speakers = []
        for seg in result["segments"]:
            spk = seg.get("speaker")
            if spk and spk not in found_speakers:
                found_speakers.append(spk)
        
        mapping = {}
        if len(found_speakers) >= 1:
            mapping[found_speakers[0]] = SPEAKER_MAPPING.get("SPEAKER_00", "Atendente")
        if len(found_speakers) >= 2:
            mapping[found_speakers[1]] = SPEAKER_MAPPING.get("SPEAKER_01", "Cliente")
        
        import numpy as np
        segments = []
        for seg in result["segments"]:
            raw_speaker = seg.get("speaker", "SPEAKER_00")
            
            # Filtro de silêncio
            start_sample = int(seg["start"] * 16000)
            end_sample = int(seg["end"] * 16000)
            audio_chunk = audio[start_sample:end_sample]
            
            if len(audio_chunk) > 0:
                rms = np.sqrt(np.mean(audio_chunk**2))
                db = 20 * np.log10(rms) if rms > 0 else -100
                if db < -40:
                    continue

            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": mapping.get(raw_speaker, raw_speaker),
                "text": seg["text"].strip(),
            })

        merged = merge_consecutive_segments(segments, max_gap_seconds=MERGE_MAX_GAP_SECONDS)

        # Export Files
        stem = audio_path.stem
        
        # MD
        out_md = TRANSCRIPTION_DIR / (stem + ".md")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(f"# Transcrição\n\n- **Arquivo:** {stem}\n- **Duração:** {duration:.1f}s\n\n")
            for seg in merged:
                f.write(f"**[{fmt_time(seg['start'])}] {seg['speaker']}:**  \n{seg['text']}\n\n")

        # SRT
        out_srt = TRANSCRIPTION_DIR / (stem + ".srt")
        with open(out_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(merged, 1):
                f.write(f"{i}\n{fmt_srt_time(seg['start'])} --> {fmt_srt_time(seg['end'])}\n[{seg['speaker']}] {seg['text']}\n\n")

        # JSON
        out_json = TRANSCRIPTION_DIR / (stem + ".json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"file": stem, "duration": duration, "segments": merged}, f, indent=2, ensure_ascii=False)

        # Cleanup intermediate whisper json
        if temp_json.exists():
            temp_json.unlink()
            
        print(f"   ✓ ok: {stem}")
        return True
    except Exception as e:
        print(f"   ✗ erro na diarização de {audio_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_transcription_stage(preprocessed_files):
    """Worker function for Stage 2"""
    print(f"\nEtapa 2: Transcrevendo {len(preprocessed_files)} arquivos...")
    whisper_model = load_whisper_pipeline()
    for wav_file in preprocessed_files:
        transcribe_audio_file(wav_file, whisper_model)
    # Process exits, freeing all VRAM


def run_diarization_stage(transcribed_files):
    """Worker function for Stage 3"""
    print(f"\nEtapa 3: Diarizando {len(transcribed_files)} arquivos...")
    diar_model = load_diarization_pipeline()
    for wav_file in transcribed_files:
        if diarize_and_finalize(wav_file, diar_model):
            # Move original file to processed
            original_file = None
            for ext in SUPPORTED_FORMATS:
                candidate = AUDIO_DIR / (wav_file.stem + ext)
                if candidate.exists():
                    original_file = candidate
                    break
            
            if original_file and original_file.exists():
                shutil.move(str(original_file), str(PROCESSED_DIR / original_file.name))
            
            if wav_file.exists():
                wav_file.unlink()
    # Process exits, freeing all VRAM


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
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    run_audio = not args.rating_only
    run_rating = args.rating or args.rating_only

    audio_files = []
    if run_audio:
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(AUDIO_DIR.glob(f"*{ext}"))
        if not audio_files and not args.rating:
            print("Nenhum áudio encontrado em audio/audio/")
            raise SystemExit(1)

    start = datetime.now()
    success = failed = 0

    if run_audio:
        # Etapa 1: Preprocess (Fast, CPU bound, no need for subprocess)
        print(f"\nEtapa 1: Pré-processando {len(audio_files)} arquivos...")
        preprocessed_files = []
        for audio_file in audio_files:
            target_wav = PREPROCESSED_DIR / (audio_file.stem + ".wav")
            if target_wav.exists():
                print(f"   - {audio_file.name} já pré-processado")
                preprocessed_files.append(target_wav)
                continue
            
            print(f"   → {audio_file.name}...")
            if preprocess_audio(audio_file, target_wav):
                preprocessed_files.append(target_wav)
            else:
                print(f"   ✗ erro no pré-processamento de {audio_file.name}")

        if not preprocessed_files:
            print("Nenhum arquivo pré-processado com sucesso.")
            raise SystemExit(1)

        # Etapa 2: Transcription (Subprocess for VRAM)
        ctx = multiprocessing.get_context("spawn")
        p2 = ctx.Process(target=run_transcription_stage, args=(preprocessed_files,))
        p2.start()
        p2.join()
        
        # Verify which ones have .whisper.json now
        transcribed_files = [f for f in preprocessed_files if f.with_suffix(".whisper.json").exists()]
        
        if not transcribed_files:
            print("Nenhuma transcrição concluída.")
        else:
            # Etapa 3: Diarization (Subprocess for VRAM)
            p3 = ctx.Process(target=run_diarization_stage, args=(transcribed_files,))
            p3.start()
            p3.join()
            success = len([f for f in transcribed_files if not f.exists()]) # Cleanup means success
            failed = len(transcribed_files) - success

    elapsed = (datetime.now() - start).total_seconds()

    if run_audio and audio_files:
        print(
            f"\nConcluído: {success} ok, {failed} falhas, tempo total: {elapsed:.1f}s"
        )

    if run_rating:
        print(f"\nGerando Notas ({RATING_INPUT_DIR}) — {RATING_MODEL_ID}")
        note_ok, note_fail = process_rating_texts()
        print(f"Concluído: {note_ok} ok, {note_fail} falhas")
