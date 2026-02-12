ENV_FILE ?= .env

ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
endif

UV ?= uv
NUM_SPEAKERS ?= 2
HF_TOKEN ?=
DIARIZATION_MODEL_ID ?= pyannote/speaker-diarization-community-1

AUDIO_DIR := $(CURDIR)/audio
TRANSCRIPTION_DIR := $(CURDIR)/transcription
PROCESSED_DIR := $(CURDIR)/processed

.PHONY: help install start prepare-dirs ensure-env check-uv check-env check-token
.NOTPARALLEL:
.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  make install                 Baixa dependências e prepara o ambiente com uv"
	@echo "  make start                   Inicia o processamento com uv"
	@echo ""
	@echo "Optional variables:"
	@echo "  NUM_SPEAKERS=2"
	@echo "  ENV_FILE=.env"
	@echo ""
	@echo "Expected .env keys:"
	@echo "  HF_TOKEN=hf_..."
	@echo "  NUM_SPEAKERS=2"
	@echo "  DIARIZATION_MODEL_ID=pyannote/speaker-diarization-community-1"

prepare-dirs:
	@mkdir -p $(AUDIO_DIR) $(TRANSCRIPTION_DIR) $(PROCESSED_DIR)

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
	$(UV) run main.py
