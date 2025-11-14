# Sistema de Transcrição e Diarização de Áudio

Sistema automatizado para transcrever áudios e identificar falantes usando IA.

## Funcionalidades

- **Transcrição de alta qualidade**: Usa Whisper large-v3-turbo para transcrição precisa em português
- **Diarização de falantes**: Identifica e separa diferentes falantes automaticamente
- **Detecção precisa de mudanças**: Algoritmo avançado que detecta quando falantes alternam rapidamente
- **Processamento em lote**: Processa múltiplos arquivos de áudio automaticamente
- **Suporte a múltiplos formatos**: WAV, MP3, FLAC, M4A, OGG

## Melhorias Implementadas

### Precisão de Diarização
- **Sobreposição inteligente**: Atribui texto a falantes apenas quando há 50%+ de sobreposição temporal
- **Merge conservador**: Junta segmentos do mesmo falante apenas se estiverem a menos de 0.3s de distância
- **Detecção por energia**: Analisa mudanças na energia do áudio para identificar trocas de falante
- **Segmentação sensível**: Configurado para detectar falas breves e interjeições curtas

## Estrutura do Projeto

```
diarization/
├── audio/              # Coloque seus arquivos de áudio aqui
│   └── .gitkeep
├── transcription/      # Transcrições geradas aparecem aqui
│   └── .gitkeep
├── main.py            # Script principal
├── README.md          # Esta documentação
└── .gitignore         # Ignora arquivos de áudio/transcrição no git
```

## Requisitos

- Python 3.8+
- CUDA (opcional, mas recomendado para melhor performance)
- Token do Hugging Face (para pyannote.audio)

## Instalação

1. Clone o repositório
2. Configure o token do Hugging Face:
   ```bash
   huggingface-cli login
   ```
3. As dependências são gerenciadas automaticamente pelo uv

## Uso

1. Coloque seus arquivos de áudio na pasta `audio/`
2. Execute o script:
   ```bash
   uv run main.py
   ```
3. As transcrições serão geradas na pasta `transcription/`

### Formatos Suportados
- `.wav`
- `.mp3`
- `.flac`
- `.m4a`
- `.ogg`

## Saída

Cada arquivo de áudio gera um arquivo markdown com:
- Nome do arquivo original
- Duração total
- Idioma detectado
- Transcrição com timestamps
- Identificação de falantes (SPEAKER_00, SPEAKER_01, etc.)

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

- **pyannote.audio**: Diarização de falantes (speaker-diarization-3.1)
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
- Os parâmetros já foram otimizados para melhor precisão
- Para áudios com muito ruído, considere pré-processamento
- Áudios com mais de 2-3 falantes podem ter precisão reduzida

## Licença

Este projeto usa modelos que requerem aceitar os termos de uso no Hugging Face.
