UV ?= uv

.PHONY: help install start start\:rating rating login

help:
	@echo "Uso:"
	@echo "  make install          instala dependências"
	@echo "  make login            autentica no Hugging Face"
	@echo "  make start            transcreve áudios (sem rating)"
	@echo "  make start:rating     transcreve áudios + gera notas"
	@echo "  make rating           gera notas de textos em transcription/"

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
