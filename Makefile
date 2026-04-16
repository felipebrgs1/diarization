UV ?= uv

.PHONY: help install start start\:rating rating login

help:
	@echo "Uso:"
	@echo "  make install          instala dependências"
	@echo "  make login            autentica no Hugging Face"
	@echo "  make start            transcreve áudios em audio/audio/ (sem rating)"
	@echo "  make start:rating     transcreve + gera notas"
	@echo "  make rating           só notas a partir de audio/transcription/"

install:
	@$(UV) sync

login:
	@$(UV) run huggingface-cli login

start: install
	@$(UV) run main.py

start\:rating: install
	@$(UV) run main.py --rating

rating: install
	@$(UV) run main.py --rating-only
