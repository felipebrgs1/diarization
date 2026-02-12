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
	-e NUM_SPEAKERS=$(NUM_SPEAKERS) \
	-e HF_TOKEN=$(HF_TOKEN) \
	-v $(AUDIO_DIR):/app/audio \
	-v $(TRANSCRIPTION_DIR):/app/transcription \
	-v $(PROCESSED_DIR):/app/processed

.PHONY: help build run run-gpu shell prepare-dirs

help:
	@echo "Targets:"
	@echo "  make build                   Build Docker image"
	@echo "  make run HF_TOKEN=...        Run diarization in Docker (CPU)"
	@echo "  make run-gpu HF_TOKEN=...    Run diarization in Docker (GPU)"
	@echo "  make shell HF_TOKEN=...      Open interactive shell in container"
	@echo ""
	@echo "Optional variables:"
	@echo "  IMAGE_NAME=diarization IMAGE_TAG=latest NUM_SPEAKERS=2"

prepare-dirs:
	@mkdir -p $(AUDIO_DIR) $(TRANSCRIPTION_DIR) $(PROCESSED_DIR)

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

run: prepare-dirs build
	$(DOCKER_RUN_BASE) $(IMAGE_NAME):$(IMAGE_TAG)

run-gpu: prepare-dirs build
	$(DOCKER_RUN_BASE) --gpus all $(IMAGE_NAME):$(IMAGE_TAG)

shell: prepare-dirs build
	$(DOCKER_RUN_BASE) --entrypoint /bin/bash $(IMAGE_NAME):$(IMAGE_TAG)
