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

- `uv` instalado
- Python 3.13 (o projeto fixa `>=3.13,<3.14`)
- CUDA (opcional, mas recomendado para melhor performance)
- Token do Hugging Face (para pyannote.audio)

## Instalação

1. Clone o repositório
2. Instale as dependências com:
   ```bash
   make install
   ```
   Isso usa `uv sync`, cria as pastas de trabalho e gera `.env` a partir de `.env.example` (se necessário).
3. Edite o arquivo `.env` e preencha:
   ```bash
   HF_TOKEN=hf_xxx
   ```
4. Aceite os termos de uso de `pyannote/speaker-diarization-community-1` no Hugging Face.

## Uso

1. Coloque seus arquivos de áudio na pasta `audio/`
2. Inicie o processamento:
   ```bash
   make start
   ```
   Opcional para chamadas telefônicas:
   ```bash
   NUM_SPEAKERS=2 make start
   ```
3. As transcrições serão geradas na pasta `transcription/`
4. Os áudios processados serão automaticamente movidos para `processed/`

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
- Confirme `HF_TOKEN` preenchido no arquivo `.env`
- Aceite os termos de uso do pyannote.audio no Hugging Face

### Erro `unexpected keyword argument 'plda'`
- Causa: `pyannote/speaker-diarization-community-1` incompatível com `pyannote-audio==3.x`
- Comportamento atual do projeto: fallback automático para `pyannote/speaker-diarization-3.1`
- Para fixar manualmente no `.env`:
  ```bash
  DIARIZATION_MODEL_ID=pyannote/speaker-diarization-3.1
  ```

### GPU não detectada
- Verifique se `nvidia-smi` funciona no host
- Confirme drivers/CUDA instalados corretamente
- O sistema roda em CPU automaticamente, mas com menor performance

### Diarização imprecisa
- Para telefonia, mantenha `NUM_SPEAKERS=2` quando souber que são dois lados da ligação
- Para áudios com muito ruído, considere pré-processamento
- Áudios com mais de 2-3 falantes podem ter precisão reduzida

## Licença

Este projeto usa modelos que requerem aceitar os termos de uso no Hugging Face.
