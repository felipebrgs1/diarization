ENV_FILE ?= .env

ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
endif

UV ?= uv
NUM_SPEAKERS ?= 2
HF_TOKEN ?=
DIARIZATION_MODEL_ID ?= pyannote/speaker-diarization-community-1
NOTE_PROMPT_FILE ?= prompt_nota_atendimento.txt
RATING_INPUT_DIR ?= transcription
RATING_MODEL_ID ?= Qwen/Qwen3-4B-Instruct-2507
RATING_MAX_NEW_TOKENS ?= 700
RATING_QUANTIZATION ?= 4bit

AUDIO_DIR := $(CURDIR)/audio
TRANSCRIPTION_DIR := $(CURDIR)/transcription
PROCESSED_DIR := $(CURDIR)/processed
NOTES_DIR := $(CURDIR)/notes

.PHONY: help install start start\:rating prepare-dirs ensure-env check-uv check-env check-token
.NOTPARALLEL:
.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  make install                 Baixa dependências e prepara o ambiente com uv"
	@echo "  make start                   Processa áudio (transcrição + diarização)"
	@echo "  make start:rating            Gera nota de atendimento a partir de textos em transcription/"
	@echo ""
	@echo "Optional variables:"
	@echo "  NUM_SPEAKERS=2"
	@echo "  ENV_FILE=.env"
	@echo "  NOTE_PROMPT_FILE=prompt_nota_atendimento.txt"
	@echo "  RATING_INPUT_DIR=transcription"
	@echo "  RATING_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507"
	@echo "  RATING_MAX_NEW_TOKENS=700"
	@echo "  RATING_QUANTIZATION=4bit"
	@echo ""
	@echo "Expected .env keys:"
	@echo "  HF_TOKEN=hf_..."
	@echo "  NUM_SPEAKERS=2"
	@echo "  DIARIZATION_MODEL_ID=pyannote/speaker-diarization-community-1"
	@echo "  NOTE_PROMPT_FILE=prompt_nota_atendimento.txt"
	@echo "  RATING_INPUT_DIR=transcription"
	@echo "  RATING_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507"
	@echo "  RATING_MAX_NEW_TOKENS=700"
	@echo "  RATING_QUANTIZATION=4bit"

prepare-dirs:
	@mkdir -p $(AUDIO_DIR) $(TRANSCRIPTION_DIR) $(PROCESSED_DIR) $(NOTES_DIR)

ensure-env:
	@if [ ! -f "$(ENV_FILE)" ] && [ -f ".env.example" ]; then \
		cp .env.example $(ENV_FILE); \
		echo "Arquivo $(ENV_FILE) criado a partir de .env.example."; \
		echo "Edite $(ENV_FILE) e preencha HF_TOKEN antes de executar make start."; \
	fi

check-uv:
	@if ! command -v $(UV) >/dev/null 2>&1; then \
		echo "uv não encontrado."; \
		echo "Instale em: https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	fi

check-env:
	@if [ ! -f "$(ENV_FILE)" ]; then \
		echo "Arquivo $(ENV_FILE) não encontrado."; \
		echo "Crie com: cp .env.example .env"; \
		exit 1; \
	fi

check-token:
	@if [ -z "$(HF_TOKEN)" ] || [ "$(HF_TOKEN)" = "seu_token" ] || [ "$(HF_TOKEN)" = "hf_xxxxx" ] || [ "$(HF_TOKEN)" = "hf_coloque_seu_token_aqui" ]; then \
		echo "HF_TOKEN ausente ou placeholder."; \
		echo "Edite $(ENV_FILE) e defina HF_TOKEN=hf_..."; \
		exit 1; \
	fi
	@if printf '%s' "$(HF_TOKEN)" | grep -q '"'; then \
		echo "HF_TOKEN no $(ENV_FILE) não deve ter aspas."; \
		echo "Use: HF_TOKEN=hf_..."; \
		exit 1; \
	fi

install: check-uv ensure-env prepare-dirs
	@$(UV) sync
	@echo "Ambiente pronto. Use make start para iniciar."

start: check-uv check-env check-token prepare-dirs
	@HF_TOKEN="$(HF_TOKEN)" \
	NUM_SPEAKERS="$(NUM_SPEAKERS)" \
	DIARIZATION_MODEL_ID="$(DIARIZATION_MODEL_ID)" \
	NOTE_PROMPT_FILE="$(NOTE_PROMPT_FILE)" \
	RATING_MODEL_ID="$(RATING_MODEL_ID)" \
	RATING_MAX_NEW_TOKENS="$(RATING_MAX_NEW_TOKENS)" \
	RATING_QUANTIZATION="$(RATING_QUANTIZATION)" \
	RUN_STAGE="audio" \
	$(UV) run main.py

start\:rating: check-uv check-env prepare-dirs
	@HF_TOKEN="$(HF_TOKEN)" \
	NOTE_PROMPT_FILE="$(NOTE_PROMPT_FILE)" \
	RATING_INPUT_DIR="$(RATING_INPUT_DIR)" \
	RATING_MODEL_ID="$(RATING_MODEL_ID)" \
	RATING_MAX_NEW_TOKENS="$(RATING_MAX_NEW_TOKENS)" \
	RATING_QUANTIZATION="$(RATING_QUANTIZATION)" \
	RUN_STAGE="rating" \
	$(UV) run main.py
