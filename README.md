# TranscriberVoz 🎙️

Sistema avançado de transcrição e diarização (identificação de falantes) otimizado para GPUs com VRAM limitada (ex: 6GB).

## ✨ Funcionalidades

- **Transcrição de Alta Precisão**: Utiliza WhisperX (`large-v3`) para transcrições rápidas e precisas.
- **Diarização 4.0**: Identificação de falantes usando o modelo `pyannote/speaker-diarization-community-1`.
- **Alinhamento a nível de palavra**: Sincronização precisa entre áudio e texto.
- **Fluxo de Trabalho em Etapas**:
  1. **Pré-processamento**: Converte áudios para formato padrão (16kHz WAV).
  2. **Transcrição**: Gera o texto base e alinhamento.
  3. **Diarização**: Identifica quem falou cada trecho.
- **Otimização de Memória**: Isolamento total via subprocessos para garantir que a VRAM seja liberada entre as fases de processamento, permitindo rodar modelos pesados em placas como a RTX 3060 Laptop (6GB).
- **Exportação Multi-formato**: Gera resultados em Markdown (`.md`), SRT (`.srt`) e JSON (`.json`).

## 🚀 Como Iniciar

### Pré-requisitos

1. **Python 3.12+** e **uv** instalados.
2. **GPU NVIDIA** com drivers CUDA atualizados.
3. **Token do Hugging Face**: Necessário para baixar os modelos da Pyannote.
   - Crie um token em: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Aceite os termos de uso dos modelos `pyannote/speaker-diarization-3.1` e `pyannote/segmentation-3.0` no Hugging Face.

### Instalação

1. Clone o repositório.
2. Configure o ambiente:
   ```bash
   cp .env.example .env
   # Edite o .env e insira seu HF_TOKEN
   ```
3. Instale as dependências:
   ```bash
   make install
   ```

### Uso

Coloque seus arquivos de áudio na pasta `audio/audio/` e execute:

```bash
make start
```

Os resultados serão salvos em `audio/transcription/`.

## 🛠️ Comandos Disponíveis (Makefile)

- `make install`: Sincroniza dependências via `uv`.
- `make login`: Autentica no Hugging Face CLI.
- `make start`: Inicia o processo completo de transcrição e diarização.
- `make start:rating`: Transcreve e gera notas de avaliação do atendimento (LLM).
- `make rating`: Gera apenas as notas para transcrições já existentes.

## 📁 Estrutura de Pastas

- `audio/audio/`: Arquivos de áudio originais.
- `audio/preprocessed/`: Áudios convertidos e arquivos intermediários.
- `audio/transcription/`: Resultados finais (MD, SRT, JSON).
- `audio/processed/`: Arquivos originais movidos após conclusão.
- `src/`: Código fonte do sistema.

## ⚙️ Configuração

As configurações principais (modelos, mapeamento de falantes, etc) podem ser ajustadas em `src/config.py`.

---
Desenvolvido para máxima eficiência e qualidade em hardware local.
