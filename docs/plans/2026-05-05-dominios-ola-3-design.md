---
title: "Diseño — Sección Dominios, Ola 3 (Audio / Voz)"
date: 2026-05-05
status: aprobado
autor: Roberto Araneda
---

# Diseño — Ola 3: Audio / Voz

## Contexto

Ola 1 (mergeada 2026-05-03) entregó la infraestructura + Texto/NLP + Visión. Ola 2 (mergeada 2026-05-05) sumó Multimodal. Esta Ola 3 agrega **Audio / Voz** como cuarto dominio completo. La infraestructura (shortcodes, CSS, menú, landing) ya existe; solo se reemplaza el stub de `dominios/audio/_index.md` por una página completa.

**Decisión de alcance:** se separa Video a la Ola 4 (decisión del usuario para mantener PRs manejables). Robótica/RL y Datos estructurados pasan a Ola 5.

## Material existente

**Sin material específico** de audio en `fundamentos/` ni `papers/`. Se enlaza material adyacente (LSTM, atención, Transformer, CNN) cuando es predecesor relevante. **Todos los hitos serán `minimal`** con descripción inline (qué hizo + por qué importó).

## Decisiones aprobadas

1. Una página: `dominios/audio/_index.md`, patrón idéntico al de Multimodal/Texto/Visión.
2. Cinco eras: acústica clásica → deep speech híbrido → end-to-end con atención → self-supervised → foundation models.
3. Aproximadamente 17 hitos distribuidos 3+3+4+3+4.
4. Todos los hitos `minimal` (sin material existente). Algunos `covered` opcionales si vinculamos al fundamento adyacente más cercano (LSTM/atención/Transformer) cuando el hito lo justifica claramente.
5. Implementación en 4 tasks (igual que Multimodal — la infraestructura ya existe).
6. Branch: `feat/dominios-ola-3` (creada desde `main` post-merge de Ola 2).

## Estructura de la página

`site/content/dominios/audio/_index.md` reemplaza el stub. Mismo molde que Multimodal:

1. Front matter (`title: "Audio / Voz"`, `weight: 3`, `sidebar.open: true`).
2. `# Audio / Voz` (H1).
3. `## El problema central` — 1-2 párrafos.
4. `## Línea de tiempo` con 5 eras y ~17 hitos.
5. 5 subsecciones `## Era N — <nombre> (<rango>)` con Problema heredado / Idea clave / Qué la destronó (eras 1-4) o Qué viene (era 5).
6. `## Estado del arte hoy` (callout).
7. `## Casos de uso reales`.
8. `## Qué viene`.
9. `## Recursos relacionados`.
10. Pie con fecha.

## El problema central — esbozo

Dos párrafos:

1. **El audio es una señal continua de alta tasa de muestreo** (16-48 kHz típicamente). 1 segundo de habla = 16,000-48,000 muestras. Modelarlo directamente en el dominio de muestras crudas era infactible hasta los 2010s; la técnica clásica era proyectarlo a un **espectrograma** (representación tiempo-frecuencia) que reduce la dimensionalidad y exhibe estructura más amable para modelos.

2. **Dos sub-problemas con tensiones opuestas:**
   - **ASR / comprensión** (signal → texto): exige robustez a ruido, acentos, dispersión hablante, y eficiencia para tiempo real.
   - **TTS / generación** (texto → audio): exige naturalidad prosódica, expresividad, control de timbre y latencia baja para conversación.
   
   Y una tensión transversal: **espectrograma vs raw waveform**. El primero es eficiente pero descarta información; el segundo es fiel pero costoso. La era moderna (Whisper, AudioLM) navega ambos según la tarea.

## Línea de tiempo — eras y hitos

### Era 1 — Acústica clásica (1980s-2010)

| Hito | Año | Status |
|---|---|---|
| MFCC (Davis & Mermelstein) | 1980 | `minimal` |
| HMM-GMM (modelos ocultos de Markov + mezcla gaussiana) | 1980s-1990s | `minimal` |
| n-gramas acústicos + decoders WFST | 1990s-2000s | `minimal` |

### Era 2 — Deep speech híbrido (2011-2014)

| Hito | Año | Status |
|---|---|---|
| DNN-HMM (Hinton, Mohamed, Dahl) | 2011 | `minimal` |
| Kaldi toolkit (Povey) | 2011 | `minimal` |
| DeepSpeech 1 (Hannun / Baidu) | 2014 | `minimal` |

### Era 3 — End-to-end con atención (2015-2018)

| Hito | Año | Status |
|---|---|---|
| CTC loss (Graves, 2006/2014) | 2014 | `covered` → `/fundamentos/lstm-gru` |
| Listen, Attend and Spell (Chan / Google) | 2015 | `covered` → `/fundamentos/mecanismo-atencion` |
| RNN-Transducer (Graves) | 2012/2017 | `minimal` |
| DeepSpeech 2 (Amodei / Baidu) | 2015 | `minimal` |

### Era 4 — Self-supervised (2019-2021)

| Hito | Año | Status |
|---|---|---|
| wav2vec (Schneider / FAIR) | 2019 | `minimal` |
| wav2vec 2.0 (Baevski / FAIR) | 2020 | `minimal` |
| HuBERT (Hsu / FAIR) | 2021 | `minimal` |

### Era 5 — Foundation models (2022-presente)

| Hito | Año | Status |
|---|---|---|
| Whisper (Radford / OpenAI) | 2022 | `minimal` |
| AudioLM (Google) | 2022 | `minimal` |
| VALL-E (Microsoft) | 2023 | `minimal` |
| MusicLM / MusicGen / Suno / Udio | 2023-2024 | `minimal` |
| ElevenLabs / Sesame (TTS de producción) | 2023-2025 | `minimal` |

**Total: 18 hitos** (3+3+4+3+5).

## Estado del arte hoy — esbozo

Callout con 5-6 modelos punteros 2024-2025:

- **Whisper v3** — OpenAI. ASR multilingüe robusto, ya estándar industrial.
- **GPT-4o / Gemini 2.5 audio nativo** — frontier LLMs con entrada/salida de voz directa, no por pipeline ASR→LLM→TTS.
- **AudioLM 2 / MusicGen** — generación de música y ambiente.
- **VALL-E 2 / NaturalSpeech 3** — TTS zero-shot con clonación de voz desde 3 segundos de audio.
- **ElevenLabs Multilingual v3** — TTS de producción con expresividad y control fino.
- **Suno v4 / Udio v2** — música generada por texto, calidad de producción.
- **Sesame** — TTS conversacional con prosodia natural en tiempo real.

## Casos de uso reales

- **ASR en producción**: Zoom Live Transcript, Otter.ai, Apple Dictation, Google Recorder.
- **Asistentes de voz**: Alexa, Siri, Google Assistant — y la nueva generación de voz directa (ChatGPT Voice, Gemini Live).
- **Subtítulos automáticos**: YouTube, Twitch, plataformas de streaming.
- **Audiolibros y TTS comercial**: Audible AI Narration, Google Play Books.
- **Música generativa**: Suno, Udio para creadores; Stable Audio para producción.
- **Clonación de voz**: ElevenLabs para localización de contenido, accesibilidad, dubs.
- **Análisis de llamadas**: centros de contacto, compliance, análisis de sentiment.

## Qué viene

- Modelos de audio nativos en frontier LLMs (no pipeline). Latencia conversacional <300ms.
- Música generativa con control fino (estems separables, edición por prompt).
- Detección de deepfakes de voz como contramedida.
- Audio espacial generativo (3D, binaural) para AR/VR.
- Foundation models multilingües que cubren idiomas low-resource.

## Plan de implementación (4 tasks)

| Task | Entregable |
|---|---|
| 1 | Front matter + problema central + timeline (5 eras + 18 hitos) |
| 2 | 5 era subsections |
| 3 | SOTA + casos + qué viene + recursos |
| 4 | Build limpio + push + PR |

Sin tocar shortcodes, CSS, menú, ni stats.

## Convenciones (heredadas)

- Español con tildes correctas.
- Tono pedagógico-narrativo.
- 800-1500 palabras totales.
- Sin Co-Authored-By en commits.
- `weight: 3` para que aparezca tras Visión y antes de Video.

## Riesgos

| Riesgo | Mitigación |
|---|---|
| `covered` para CTC y Listen-Attend-Spell — los fundamentos linkeados (`lstm-gru`, `mecanismo-atencion`) cubren el concepto general pero no la aplicación específica a audio | Aceptable: linkeamos al concepto fundamental, no al hito puntual. La intro de la era explica la conexión. Alternativa: downgradear ambos a `minimal`. |
| Datos puntuales (años, autores) sin verificar | Code reviewer subagent debe verificar como en Texto/Visión |
| Densidad de modelos en Era 5 podría exceder lo razonable | Mantener "5 hitos máximo por era"; agrupar familias (Suno/Udio) en un solo hito |
