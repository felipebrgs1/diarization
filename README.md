# Transcriber

Transcrição + diarização de áudio local com `pyannote.audio` e `Whisper large-v3-turbo`. Geração de notas de atendimento opcional com LLM local.

## Requisitos

- `uv` instalado
- Python 3.13
- GPU NVIDIA com CUDA funcional (obrigatório)
- Aceite os termos do modelo `pyannote/speaker-diarization-community-1` no Hugging Face

## Instalação

```bash
make install
make login      # autentica no Hugging Face (salva token localmente)
```

## Uso

Coloque áudios em `audio/` e rode:

```bash
make start           # só transcrição/diarização
make start:rating    # transcrição + notas de atendimento
make rating          # só notas a partir de textos em transcription/
```

## Estrutura

- `audio/` — áudios de entrada
- `transcription/` — transcrições geradas
- `processed/` — áudios movidos após processamento
- `notes/` — notas de atendimento geradas

## Saída

Cada áudio gera um arquivo `.md` em `transcription/` com timestamps e falantes (`SPEAKER_00`, `SPEAKER_01`, ...).
