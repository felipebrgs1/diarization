# Agents Overview (Caveman Edition)

Project use local AI agents process voice. Max quality. Local VRAM optimize.

## 1. Preprocessor Agent
- **Source:** `src/audio_processing.py`
- **Role:** Prepare audio.
- **Action:** Convert to 16kHz mono WAV. Clean silences. Save `preprocessed/`.

## 2. Transcription & Alignment Agent (WhisperX)
- **Source:** `src/transcription.py` + `whisperx`
- **Model:** Whisper (Faster-Whisper) + Wav2Vec2 Alignment.
- **Role:** Voice to text + precise timing.
- **Action:** 
  - Subprocess spawn (VRAM isolation).
  - Transcribe chunks. 
  - Word-level alignment. 
  - Save `.whisper.json` cache.

## 3. Diarization Agent
- **Source:** `src/diarization.py` + `pyannote`
- **Model:** Pyannote Audio 4.0 (community-1).
- **Role:** Speaker identification.
- **Action:** 
  - Subprocess spawn.
  - Detect speaker turns (min 2, max 2).
  - Assign word speakers using WhisperX metadata.

## 4. Refinement Agent
- **Source:** `src/refinement.py`
- **Role:** Data precision.
- **Action:** Search numbers/CPFs in segments. Fix Whisper hallucination on digits.

## 5. Rating Agent (LLM)
- **Source:** `src/rating.py`
- **Model:** Configurable local LLM (DeepSeek / Llama).
- **Role:** QA evaluator.
- **Action:** 
  - Load prompt `src/prompts/`.
  - Analyze text for empathy/solution.
  - Score 0-10. Save `notes/*.nota.md`.

## Pipeline Flow (Isolated Stages)
1. **Prepare:** `audio/` → `preprocess/` (CPU).
2. **Transcribe:** `preprocess/*.wav` → `.whisper.json` (GPU Stage 1).
3. **Diarize:** `.whisper.json` + `wav` → `transcription/*.md` (GPU Stage 2).
4. **Export:** `.md`, `.srt`, `.json`. Move original to `processed/`.
5. **Rate:** (Optional) `transcription/*.md` → `notes/*.md`.

## Tech Stack
- **Diarization:** Pyannote 3.1+ / 4.0.
- **Transcription:** WhisperX (Faster-Whisper).
- **Refinement:** Wav2Vec2 (local).
- **Orchestration:** Python + Multiprocessing (VRAM management).
