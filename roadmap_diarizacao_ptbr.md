# Roadmap — Diarização PT-BR Telefônica com WhisperX + pyannote 4

> Objetivo: reduzir DER para <20% e WER para <15% em ligações telefônicas em português, rodando 100% local.

## Visão Geral
Pipeline escolhido: **ffmpeg pré-processamento → WhisperX (large-v3, VAD pyannote) → pyannote.audio 4 → pós-processamento**. Áudios são sempre 8 kHz, banda estreita, 2 falantes (atendente/cliente).

## Fase 0 — Baseline
- [x] Rodar pipeline padrão sem ajustes
- [x] Medir: DER, WER, tempo real, nº falantes detectados
- **Critério de aceite:** baseline documentado em CSV

## Fase 1 — Pré-processamento de Áudio Telefônico
- [x] Reamostrar para 16 kHz mono
- [x] Filtro banda: highpass 200 Hz, lowpass 3400 Hz
- [x] Denoise: afftdn (ffmpeg) ou RNNoise para ruído não estacionário
- [x] Normalização: loudnorm I=-16 TP=-1.5
- **Comando referência:**
```bash
ffmpeg -i in.mp3 -ar 16000 -ac 1 -af "highpass=f=200,lowpass=f=3400,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11" out.wav
```
- **Critério:** WER reduz ≥10%

## Fase 2 — Configuração WhisperX para PT-BR
- [x] Parâmetros: --language pt --vad_method pyannote --diarize
- [x] Fixar speakers: --min_speakers 2 --max_speakers 2
- [x] Chunk: --chunk_size 10
- [x] Prompt inicial: "ligação telefônica, atendimento ao cliente, português do Brasil"
- [x] Validar alinhamento com WAV2VEC2_ASR_LARGE_960H
- **Critério:** redução de alucinações e timestamps <200ms de erro

## Fase 3 — Otimização pyannote.audio 4
- [x] Usar pipeline speaker-diarization-3.1 com HF token
- [x] Ajustes: clustering.threshold = 0.65, min_cluster_size = 12, segmentation.min_duration_off = 0.3
- [x] Forçar num_speakers=2

## Fase 4 — Adaptação de Domínio (opcional, alto impacto)
- [ ] Fine-tune embedding ECAPA-TDNN no pyannote 4


## Fase 5 — Pós-processamento e Regras de Negócio
- [x] Merge turnos <0.25s do mesmo falante
- [x] Remover segmentos com energia < -40dB
- [x] Mapear speaker_0 → Atendente, speaker_1 → Cliente (heurística: quem fala primeiro)
- [x] Exportar: TXT, SRT, JSON com timestamps
- [x] Correção ASR: segunda passada com wav2vec2-large-xlsr-53-portuguese para números/CPF
- **Critério:** saída legível sem falantes fantasmas


## Métricas de Acompanhamento
| Métrica | Baseline | Meta |
| --- | --- | --- |
| DER | medir | <20% |
| WER pt-BR | medir | <15% |
| Falantes corretos | medir | ≥95% |
| RTF GPU | medir | <0.3 |

## Riscos
- Áudio 8kHz muito degradado → considerar treino específico 8k
- Sobreposição de fala → pyannote 4 detecta overlap, mas precisão cai
- VRAM insuficiente → usar quantização faster-whisper int8

## Checklist para Agente
1. Validar cada fase antes de avançar
2. Salvar artefatos: wav limpo, json diarização, txt final
3. Comparar métricas fase a fase
4. Registrar comandos exatos usados
5. Se DER >25% após Fase 3, priorizar Fase 4
