# Sistema de Transcrição e Diarização de Áudio

Sistema automatizado para transcrever áudios e identificar falantes usando IA.

## Funcionalidades

- **Transcrição de alta qualidade**: Usa Whisper large-v3-turbo para transcrição precisa em português
- **Diarização de falantes**: Usa `pyannote/speaker-diarization-community-1`
- **Alinhamento robusto**: Usa diarização exclusiva para reduzir troca incorreta de falante
- **Processamento em lote**: Processa múltiplos arquivos de áudio automaticamente
- **Organização automática**: Move áudios processados para pasta separada
- **Suporte a múltiplos formatos**: WAV, MP3, FLAC, M4A, OGG

## Melhorias Implementadas

### Precisão de Diarização
- **Diarização exclusiva**: Prioriza `exclusive_speaker_diarization` quando disponível
- **Atribuição por maior sobreposição**: Cada chunk do Whisper é associado ao falante com maior interseção temporal
- **Fallback resiliente**: Quando não há sobreposição, usa o turno mais próximo no tempo
- **Merge conservador**: Junta segmentos adjacentes do mesmo falante com gap curto

### Fluxo de Processamento
```
audio/arquivo.wav  →  [Processamento]  →  transcription/arquivo.md
                                      ↘  processed/arquivo.wav
```
O sistema automaticamente move áudios processados para manter a pasta `audio/` organizada.

## Estrutura do Projeto

```
diarization/
├── audio/              # Coloque seus arquivos de áudio aqui
│   └── .gitkeep
├── transcription/      # Transcrições geradas aparecem aqui
│   └── .gitkeep
├── processed/          # Áudios já processados são movidos para cá
│   └── .gitkeep
├── main.py            # Script principal
├── README.md          # Esta documentação
└── .gitignore         # Ignora arquivos de áudio/transcrição no git
```

## Requisitos

- Python 3.11+
- CUDA (opcional, mas recomendado para melhor performance)
- Token do Hugging Face (para pyannote.audio)

## Instalação

1. Clone o repositório
2. Configure o token do Hugging Face:
   ```bash
   huggingface-cli login
   ```
   Aceite os termos de uso de `pyannote/speaker-diarization-community-1` no Hugging Face.
3. As dependências são gerenciadas automaticamente pelo uv

## Uso

1. Coloque seus arquivos de áudio na pasta `audio/`
2. Execute o script:
   ```bash
   uv run main.py
   ```
   Opcional para chamadas telefônicas:
   ```bash
   NUM_SPEAKERS=2 uv run main.py
   ```
3. As transcrições serão geradas na pasta `transcription/`
4. Os áudios processados serão automaticamente movidos para `processed/`

## Uso com Docker (sem Python local)

Requisito: Docker instalado na máquina.

1. Faça login no Hugging Face na máquina host **ou** passe o token via variável:
   ```bash
   export HF_TOKEN=seu_token_aqui
   ```
2. Faça o build da imagem:
   ```bash
   make build
   ```
3. Rode o processamento em container (CPU):
   ```bash
   make run HF_TOKEN=$HF_TOKEN NUM_SPEAKERS=2
   ```
4. Para usar GPU NVIDIA:
   ```bash
   make run-gpu HF_TOKEN=$HF_TOKEN NUM_SPEAKERS=2
   ```

Pastas `audio/`, `transcription/` e `processed/` são montadas como volume, então os arquivos continuam no host.

### Formatos Suportados
- `.wav`
- `.mp3`
- `.flac`
- `.m4a`
- `.ogg`

## Saída

Cada arquivo de áudio gera:
- **Transcrição markdown** em `transcription/` com:
  - Nome do arquivo original
  - Duração total
  - Idioma detectado
  - Transcrição com timestamps
  - Identificação de falantes (SPEAKER_00, SPEAKER_01, etc.)
- **Áudio original** movido para `processed/` (mantém o mesmo nome)

### Exemplo de Saída

```markdown
# Transcrição de Áudio

## Informações

- **Arquivo:** audio.wav
- **Duração:** 330.8s
- **Idioma:** pt

## Transcrição

**[00:02] SPEAKER_01:**
Olá, boa tarde. Eu consigo falar com Lisette?

**[00:05] SPEAKER_00:**
Sim, é ela.
```

## Tecnologias

- **pyannote.audio**: Diarização de falantes (`speaker-diarization-community-1`)
- **OpenAI Whisper**: Transcrição de áudio (large-v3-turbo)
- **PyTorch**: Framework de deep learning
- **soundfile**: Processamento de áudio

## Performance

- GPU: Recomendado para processamento rápido
- CPU: Funciona mas é significativamente mais lento
- Memória: ~4GB VRAM para GPU, ~8GB RAM para CPU

## Solução de Problemas

### "No audio files found"
- Certifique-se de que há arquivos de áudio na pasta `audio/`
- Verifique se os arquivos têm extensões suportadas

### Erros de autenticação
- Execute `huggingface-cli login` e forneça seu token
- Aceite os termos de uso do pyannote.audio no Hugging Face

### Diarização imprecisa
- Para telefonia, mantenha `NUM_SPEAKERS=2` quando souber que são dois lados da ligação
- Para áudios com muito ruído, considere pré-processamento
- Áudios com mais de 2-3 falantes podem ter precisão reduzida

## Licença

Este projeto usa modelos que requerem aceitar os termos de uso no Hugging Face.
