# Agents Overview

System use local AI pipelines to process voice files and evaluate service quality.

## 1. Diarization Agent
- **Source:** `src/diarization.py`
- **Model:** Pyannote Audio.
- **Role:** Identify speaker turns in raw audio.
- **Action:** 
  - Detect waveform segments.
  - Map timestamps to Speaker IDs (SPEAKER_00, SPEAKER_01, etc.).
  - Build speaker turn metadata for transcription alignment.

## 2. Transcription Agent
- **Source:** `src/transcription.py`
- **Model:** OpenAI Whisper (Local via Transformers).
- **Role:** Convert speech to text.
- **Action:**
  - Process audio chunks defined by speaker turns.
  - Generate timestamped text.
  - Merge consecutive segments from same speaker for readability.
  - Save output as Markdown in `transcription/`.

## 3. Rating Agent
- **Source:** `src/rating.py`
- **Model:** Local LLM (configurable in `src/config.py`).
- **Role:** Quality Assurance evaluator.
- **Action:**
  - Load prompt template from `src/prompts/`.
  - Analyze transcribed text for clarity, empathy, and solution efficacy.
  - Output final score (0-10) and improvement points.
  - Save notes in `notes/`.

## Pipeline Flow
1. Audio in `audio/`.
2. `Diarization` + `Transcription` → `transcription/*.md`.
3. `Rating` → `notes/*.nota.md`.
