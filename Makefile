ENV_FILE ?= .env

ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
endif

IMAGE_NAME ?= diarization
IMAGE_TAG ?= latest
CONTAINER_NAME ?= diarization-run
NUM_SPEAKERS ?= 2
HF_TOKEN ?=

AUDIO_DIR := $(CURDIR)/audio
TRANSCRIPTION_DIR := $(CURDIR)/transcription
PROCESSED_DIR := $(CURDIR)/processed

DOCKER_RUN_BASE := docker run --rm -it \
	--name $(CONTAINER_NAME) \
	--env-file $(ENV_FILE) \
	-e NUM_SPEAKERS=$(NUM_SPEAKERS) \
	-v $(AUDIO_DIR):/app/audio \
	-v $(TRANSCRIPTION_DIR):/app/transcription \
	-v $(PROCESSED_DIR):/app/processed

.PHONY: help build run run-gpu shell prepare-dirs doctor-gpu check-env check-token check-gpu
.NOTPARALLEL:

help:
	@echo "Targets:"
	@echo "  make build                   Build Docker image"
	@echo "  make run                     Run diarization in Docker (CPU)"
	@echo "  make run-gpu                 Run diarization in Docker (GPU)"
	@echo "  make doctor-gpu              Check Docker GPU runtime availability"
	@echo "  make shell                   Open interactive shell in container"
	@echo ""
	@echo "Optional variables:"
	@echo "  IMAGE_NAME=diarization IMAGE_TAG=latest NUM_SPEAKERS=2"
	@echo "  ENV_FILE=.env"
	@echo ""
	@echo "Expected .env keys:"
	@echo "  HF_TOKEN=hf_..."
	@echo "  NUM_SPEAKERS=2"
	@echo "  DIARIZATION_MODEL_ID=pyannote/speaker-diarization-community-1"

prepare-dirs:
	@mkdir -p $(AUDIO_DIR) $(TRANSCRIPTION_DIR) $(PROCESSED_DIR)

check-env:
	@if [ ! -f "$(ENV_FILE)" ]; then \
		echo "Arquivo $(ENV_FILE) não encontrado."; \
		echo "Crie com: cp .env.example .env"; \
		exit 1; \
	fi

check-token:
	@if [ -z "$(HF_TOKEN)" ] || [ "$(HF_TOKEN)" = "seu_token" ] || [ "$(HF_TOKEN)" = "hf_xxxxx" ]; then \
		echo "HF_TOKEN ausente ou placeholder."; \
		echo "Edite $(ENV_FILE) e defina HF_TOKEN=hf_..."; \
		exit 1; \
	fi
	@if printf '%s' "$(HF_TOKEN)" | grep -q '"'; then \
		echo "HF_TOKEN no $(ENV_FILE) não deve ter aspas."; \
		echo "Use: HF_TOKEN=hf_..."; \
		exit 1; \
	fi

check-gpu:
	@if ! docker info --format '{{json .Runtimes}}' | grep -q nvidia; then \
		echo "NVIDIA runtime not found in Docker."; \
		echo "Install nvidia-container-toolkit and configure Docker runtime."; \
		echo "For now, use: make run"; \
		exit 1; \
	fi

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

run: check-env check-token prepare-dirs build
	@$(DOCKER_RUN_BASE) $(IMAGE_NAME):$(IMAGE_TAG)

run-gpu: check-env check-token check-gpu prepare-dirs build
	@$(DOCKER_RUN_BASE) --gpus all $(IMAGE_NAME):$(IMAGE_TAG)

shell: check-env prepare-dirs build
	@$(DOCKER_RUN_BASE) --entrypoint /bin/bash $(IMAGE_NAME):$(IMAGE_TAG)

doctor-gpu:
	@echo "Docker runtimes:"
	@docker info --format '{{json .Runtimes}}'
	@echo "NVIDIA driver (host):"
	@nvidia-smi -L
