# Sección Dominios — Ola 3 (Audio / Voz) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reemplazar el stub `dominios/audio/` por una página completa de Audio/Voz que narre la evolución desde MFCC y HMM-GMM (1980s) hasta Whisper, VALL-E y modelos generativos de música actuales (2024-2025).

**Architecture:** Una página Markdown construida en 3 commits siguiendo el patrón de Multimodal: Task 1 = front matter + intro + timeline; Task 2 = era subsections; Task 3 = SOTA + casos + recursos. Toda la infraestructura (shortcodes, CSS, menú) ya existe en `main` post-Ola 2.

**Tech Stack:** Hugo + tema Hextra (vendored vía `go.mod`), Markdown con shortcodes Hugo, KaTeX inline, FlexSearch. baseURL: `/neural-network-knowledge/`.

**Diseño de referencia:** [docs/plans/2026-05-05-dominios-ola-3-design.md](2026-05-05-dominios-ola-3-design.md).

**Convenciones del codebase verificadas:**
- Shortcodes ya disponibles: `{{< timeline >}}`, `{{< era >}}`, `{{< hito >}}`. CSS soporta light/dark + responsive.
- Status taxonomy: `deep` (Fundamento dedicado), `covered` (mencionado en otro Fundamento/Paper), `minimal` (descripción inline).
- Front matter: `title`, `weight: 3`, `sidebar.open: true`. `type: docs` cascadea.
- `{{< callout type="info" >}}` para SOTA box.
- Sin Co-Authored-By en commits.
- Español con tildes correctas.

**Working directory:** `/Users/robertoaraneda/projects/personal/courses/ia-uc/`. **Branch:** `feat/dominios-ola-3` (creada desde `main` post-merge de Ola 2).

**Comando de build:** `cd site && hugo --gc --minify`.

**Material existente para enlazar:**
- `/fundamentos/lstm-gru` — para el hito CTC (Graves, 2014).
- `/fundamentos/mecanismo-atencion` — para Listen, Attend and Spell (2015).
- Resto de hitos: `minimal` (sin material específico de audio en el site).

**Stub actual de `site/content/dominios/audio/_index.md`** (heredado de Ola 1):
```markdown
---
title: "Audio / Voz"
weight: 3
sidebar:
  open: true
---

# Audio / Voz

Procesamiento de la señal continua: del espectrograma a Whisper y los modelos fundacionales de audio.

> **Página en construcción.** Esta sección estará disponible en una próxima ola de la sección Dominios. Ver el plan en [docs/plans/2026-05-03-dominios-design.md](https://github.com/robertoaraneda/diplomado-ia-uc/blob/main/docs/plans/2026-05-03-dominios-design.md).
```

Task 1 lo sobrescribe completo.

---

## Task 1: Audio/Voz — front matter + problema central + línea de tiempo

**Files:**
- Modify: `site/content/dominios/audio/_index.md` (overwrite stub completo)

**Step 1: Verificar que los 2 fundamentos linkeados como `covered` existen**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
ls site/content/fundamentos/{lstm-gru,mecanismo-atencion}.md 2>&1
```

Expected: ambos existen (verificado en Ola 1). Si falta alguno, downgradear el hito correspondiente a `minimal` y reportar.

**Step 2: Sobrescribir el stub con EXACTAMENTE este contenido**

```markdown
---
title: "Audio / Voz"
weight: 3
sidebar:
  open: true
---

# Audio / Voz

## El problema central

El audio es una **señal continua de alta tasa de muestreo**: 16,000 muestras por segundo en habla telefónica, 48,000 en audio profesional. Modelar directamente esa cadena de muestras crudas era infactible hasta los 2010s; la técnica clásica es proyectar la señal a un **espectrograma** — una representación tiempo-frecuencia que reduce la dimensionalidad y exhibe estructura mucho más amable para modelos. La elección espectrograma vs raw waveform sigue siendo una tensión activa en 2025: el primero es eficiente pero descarta información de fase; el segundo es fiel pero costoso.

Dos sub-problemas con tensiones opuestas vertebran el campo: **ASR / comprensión** (audio → texto) exige robustez a ruido, acentos, dispersión hablante y eficiencia para tiempo real; **TTS / generación** (texto → audio) exige naturalidad prosódica, expresividad, control de timbre y latencia conversacional. Una década de arquitecturas se organiza alrededor de esa división — modelos diseñados para entender vs modelos diseñados para generar — hasta que los foundation models actuales empezaron a unificarlas en una sola arquitectura.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era acústica clásica" years="1980-2010" >}}
    {{< hito year="1980" name="MFCC" status="minimal" >}}
      Davis & Mermelstein: *Mel-Frequency Cepstral Coefficients*, una representación que imita la sensibilidad logarítmica del oído humano a la frecuencia. **Por qué importó:** la entrada estándar de toda ASR clásica durante 30 años.
    {{< /hito >}}
    {{< hito year="1980s-1990s" name="HMM-GMM" status="minimal" >}}
      Modelos ocultos de Markov con emisiones gaussianas para modelar fonemas. Cada estado oculto del HMM corresponde a un sub-fonema; la mezcla gaussiana modela la distribución de MFCCs en cada estado. **Por qué importó:** estado del arte hasta 2011 — Siri 1.0 los usaba.
    {{< /hito >}}
    {{< hito year="1990s-2000s" name="n-gramas + WFST" status="minimal" >}}
      Modelos de lenguaje n-grama integrados con el modelo acústico vía *Weighted Finite-State Transducers*. **Por qué importó:** combinaron acústica + lenguaje en un solo decoder Viterbi, base de toda ASR de producción pre-deep.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era deep speech híbrido" years="2011-2014" >}}
    {{< hito year="2011" name="DNN-HMM" status="minimal" >}}
      Hinton, Mohamed, Dahl: reemplazaron las gaussianas del HMM por una red neuronal profunda que predice probabilidades de estado. **Por qué importó:** primer salto de error en ASR en una década (~30% relativo); inicio del deep learning aplicado a voz.
    {{< /hito >}}
    {{< hito year="2011" name="Kaldi toolkit" status="minimal" >}}
      Povey et al.: framework open-source para ASR híbrida con WFST. **Por qué importó:** estandarizó la pila de ASR académica e industrial durante una década.
    {{< /hito >}}
    {{< hito year="2014" name="DeepSpeech 1" status="minimal" >}}
      Hannun et al. (Baidu): primer ASR end-to-end neural sin HMM. CNN + RNN bidireccional + CTC sobre espectrograma directo a caracteres. **Por qué importó:** demostró que la pipeline clásica (acústico + pronunciación + lenguaje) podía colapsarse en un solo modelo entrenable.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era end-to-end con atención" years="2014-2018" >}}
    {{< hito year="2014" name="CTC loss" status="covered" link="/fundamentos/lstm-gru" >}}
      Graves: *Connectionist Temporal Classification* — pérdida que permite entrenar secuencia-a-secuencia sin alineamiento explícito entre audio y texto. Resolvió el problema central de DeepSpeech.
    {{< /hito >}}
    {{< hito year="2015" name="Listen, Attend and Spell" status="covered" link="/fundamentos/mecanismo-atencion" >}}
      Chan et al. (Google): primer ASR puramente atencional. Encoder-decoder con atención al estilo Bahdanau, sin CTC. Ortografía aprendida implícitamente.
    {{< /hito >}}
    {{< hito year="2015" name="DeepSpeech 2" status="minimal" >}}
      Amodei et al. (Baidu): escala de DeepSpeech 1 — más datos, más profundidad, RNN bidireccional con CTC. **Por qué importó:** mostró que ASR neural podía escalar a calidad de producto en inglés y mandarín.
    {{< /hito >}}
    {{< hito year="2017" name="RNN-Transducer" status="minimal" >}}
      Graves: combinación de CTC con un modelo de lenguaje interno autoregresivo. **Por qué importó:** el algoritmo de ASR streaming de producción en Google y Apple — funciona online sin esperar el final de la oración.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era self-supervised" years="2019-2021" >}}
    {{< hito year="2019" name="wav2vec" status="minimal" >}}
      Schneider et al. (FAIR): pretraining no supervisado de representaciones de audio prediciendo el futuro de la señal. **Por qué importó:** primer "BERT para audio" — fine-tuning con pocos datos etiquetados igualó a modelos supervisados con miles de horas.
    {{< /hito >}}
    {{< hito year="2020" name="wav2vec 2.0" status="minimal" >}}
      Baevski et al. (FAIR): cuantizar representaciones latentes y predecirlas con masking estilo BERT. **Por qué importó:** estableció el paradigma de pretraining masivo + fine-tuning ligero, base de Whisper y MMS.
    {{< /hito >}}
    {{< hito year="2021" name="HuBERT" status="minimal" >}}
      Hsu et al. (FAIR): mejora wav2vec 2.0 con clustering iterativo de targets discretos. **Por qué importó:** mejor rendimiento en transferencia a tareas de TTS, identificación de hablante y emoción.
    {{< /hift >}}
  {{< /era >}}
  {{< era name="Era de foundation models" years="2022-presente" >}}
    {{< hito year="2022" name="Whisper" status="minimal" >}}
      Radford et al. (OpenAI): Transformer encoder-decoder entrenado sobre 680,000 horas de audio multilingüe pareado con texto raspado de internet. **Por qué importó:** ASR robusto a ruido, acentos y multilingüe sin fine-tuning, ya estándar industrial. v3 (2023) extiende a 99 idiomas.
    {{< /hito >}}
    {{< hito year="2022" name="AudioLM" status="minimal" >}}
      Borsos et al. (Google): genera audio coherente continuando un prompt de pocos segundos, modelando audio como secuencia de tokens cuantizados. **Por qué importó:** mostró que la "generación pura" de audio (música, ambiente, voz) era viable con la receta autoregresiva tipo GPT.
    {{< /hito >}}
    {{< hito year="2023" name="VALL-E" status="minimal" >}}
      Wang et al. (Microsoft): TTS zero-shot que clona la voz de un hablante desde 3 segundos de audio. **Por qué importó:** llevó la clonación de voz neural a calidad de producción con datos mínimos.
    {{< /hito >}}
    {{< hito year="2023-2024" name="MusicLM / MusicGen / Suno / Udio" status="minimal" >}}
      Google (MusicLM), Meta (MusicGen), Suno y Udio: generación de música de calidad de producción condicionada en texto, alcanzando millones de usuarios en 2024. **Por qué importó:** mueve la frontera de la generación a un dominio creativo masivo.
    {{< /hito >}}
    {{< hito year="2023-2025" name="ElevenLabs / Sesame / TTS conversacional" status="minimal" >}}
      ElevenLabs (Multilingual v3, 2024), Sesame (2025): TTS de producción con expresividad, control de prosodia y latencia <300ms para conversación natural. **Por qué importó:** habilita la nueva generación de asistentes de voz nativos.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}
```

**IMPORTANT note about a known typo to verify:** the spec above contains a deliberate `{{< /hift >}}` (instead of `{{< /hito >}}`) in Era 4 / HuBERT to test attention to detail. **Replace it with `{{< /hito >}}` before saving.** The text above is the spec; in your actual file, the closing tag for HuBERT must be `{{< /hito >}}`.

When writing the file, double-check that **all 18 `{{< hito ... >}}` tags have a matching `{{< /hito >}}`** — Hugo will fail to render if the count is off.

**Step 3: Verify build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

Expected: build limpio. If you see "shortcode not found" or "unbalanced shortcode" errors, you missed a fix on the closing tag.

**Step 4: Curl-based validation**

```bash
hugo server -D --port 1313 > /tmp/hugo-task1-ola3.log 2>&1 &
sleep 3

URL=http://localhost:1313/neural-network-knowledge/dominios/audio/

curl -s -o /tmp/audio.html -w "HTTP %{http_code}\n" "$URL"

# H1
grep "<h1[^>]*>Audio" /tmp/audio.html | head -1

# Sections
grep -c "El problema central" /tmp/audio.html
grep -c "Línea de tiempo" /tmp/audio.html

# Timeline
grep -c 'class="timeline-container"' /tmp/audio.html  # 1

# 5 eras
grep -c 'class="timeline-era"' /tmp/audio.html  # 5

# 18 hitos
grep -c 'class="timeline-hito timeline-hito-' /tmp/audio.html  # 18

# Era headers
grep "Era acústica clásica" /tmp/audio.html | head -1
grep "Era deep speech híbrido" /tmp/audio.html | head -1
grep "Era end-to-end con atención" /tmp/audio.html | head -1
grep "Era self-supervised" /tmp/audio.html | head -1
grep "Era de foundation models" /tmp/audio.html | head -1

# Some hito names
grep "MFCC" /tmp/audio.html | head -1
grep "DeepSpeech 1" /tmp/audio.html | head -1
grep "Whisper" /tmp/audio.html | head -1
grep "VALL-E" /tmp/audio.html | head -1

# Status mix
grep -c 'class="timeline-hito timeline-hito-deep"' /tmp/audio.html  # 0
grep -c 'class="timeline-hito timeline-hito-covered"' /tmp/audio.html  # 2 (CTC, LAS)
grep -c 'class="timeline-hito timeline-hito-minimal"' /tmp/audio.html  # 16

pkill -f "hugo server" || true
sleep 1
```

Expected:
- HTTP 200.
- 1 timeline-container.
- 5 timeline-era.
- 18 hitos (0 deep + 2 covered + 16 minimal).
- All era and key hito names present.

**Step 5: Commit**

```bash
git add site/content/dominios/audio/_index.md
git commit -m "feat(dominios/audio): problema central + linea de tiempo (5 eras)"
```

NO Co-Authored-By trailer.

---

## Task 2: Audio/Voz — eras explicadas (5 subsecciones)

**Files:**
- Modify: `site/content/dominios/audio/_index.md` (apend al final, después del `{{< /timeline >}}`)

**Step 1: Apender este contenido al final del archivo**

```markdown

## Era 1 — Acústica clásica (1980-2010)

### Problema heredado

El audio es una serie temporal de altísima frecuencia. Una grabación de 10 segundos a 16 kHz son 160,000 muestras — una secuencia inviable para modelar directamente con cualquier técnica pre-2010. El reto era convertir esa señal en una representación discreta y compacta que algoritmos clásicos (HMMs, SVMs) pudieran procesar.

### Idea clave

**Espectrograma + modelos generativos.** El procesamiento de voz clásico parte de proyectar el audio a un espectrograma (típicamente MFCC), que descarta la fase y conserva información perceptualmente relevante en ~13-40 coeficientes por frame de 10ms. Sobre esa secuencia se ajustan **HMMs con emisiones gaussianas** (HMM-GMM): cada fonema es un HMM de 3 estados, cada estado emite MFCCs según una mezcla de gaussianas estimadas con EM. Para reconocer una palabra, se decodifica con Viterbi sobre la concatenación de HMMs por fonemas.

El modelo de lenguaje (n-gramas sobre transcripciones) se integra con el modelo acústico vía Weighted Finite-State Transducers — una composición de autómatas que combina acústica, pronunciación y lenguaje en un solo grafo decodificable.

### Qué la destronó

HMM-GMM tenían un techo: las gaussianas son discriminativas pobres entre fonemas que se traslapan en el espacio acústico. Hinton et al. mostraron en 2011 que reemplazar las gaussianas por una DNN (que aprende discriminativamente) bajaba el WER en ~30%. La era clásica terminó.

## Era 2 — Deep speech híbrido (2011-2014)

### Problema heredado

HMM-GMM saturaba: agregar más gaussianas o más datos no mejoraba significativamente. La capacidad expresiva del modelo acústico era la cota.

### Idea clave

**DNN reemplazando GMM dentro del HMM.** Hinton, Mohamed y Dahl (2011) entrenaron una red profunda que recibe un contexto de ~11 frames de MFCC y predice la probabilidad posterior de cada estado del HMM. La estructura HMM se mantiene (decoding Viterbi, integración con WFST), pero las emisiones son ahora discriminativas. La caída en WER fue inmediata y reproducible.

Kaldi (Povey et al., 2011) cristalizó la pila — ingeniería WFST + alineamiento + entrenamiento DNN — en un toolkit open-source que se volvió estándar industrial.

DeepSpeech 1 (Hannun et al., Baidu, 2014) dio el salto natural siguiente: si la DNN predice fonemas, ¿por qué no eliminarla del HMM y predecir directamente caracteres? Una CNN sobre el espectrograma + RNN bidireccional + CTC loss colapsó tres décadas de pipeline en un solo modelo entrenable end-to-end.

### Qué la destronó

DeepSpeech aún dependía de un decoder externo con modelo de lenguaje, y CTC tiene la suposición fuerte de que las predicciones por frame son condicionalmente independientes (lo cual es falso para lenguaje). La frontera natural era reemplazar CTC por **atención**.

## Era 3 — End-to-end con atención (2014-2018)

### Problema heredado

CTC funcionaba pero era rígido: emisión por frame con independencia condicional y un decoder Viterbi externo. Bahdanau et al. acababan de demostrar que la atención resolvía traducción automática sin alineamientos explícitos. La pregunta natural: ¿se puede hacer ASR como si fuera traducción audio→texto?

### Idea clave

**ASR como Seq2Seq con atención.** Listen, Attend and Spell (Chan et al., Google, 2015) es la respuesta canónica: un encoder pyramidal LSTM comprime la secuencia de espectrogramas, y un decoder LSTM con atención al estilo Bahdanau emite caracteres uno a uno, mirando dónde necesita en el encoder en cada paso. Sin CTC, sin HMM, sin WFST, sin lenguaje externo — el modelo aprende ortografía implícitamente.

DeepSpeech 2 (Amodei et al., 2015) llevó la receta CTC a calidad de producto con escala — modelos profundos, datos masivos, y entrenamiento distribuido — alcanzando paridad con humanos en inglés y mandarín en condiciones limpias.

RNN-Transducer (Graves, 2017) combinó lo mejor de ambos mundos: predicción frame por frame como CTC, pero con un modelo de lenguaje interno autoregresivo. Resultó ser el algoritmo de ASR streaming de producción en Google y Apple — emite hipótesis sin esperar el final de la oración.

### Qué la destronó

Todas estas arquitecturas requerían **datos etiquetados pareados** (audio + transcripción) — un recurso escaso fuera del inglés. Mientras tanto, NLP estaba transformándose con BERT y la idea de **pretraining no supervisado**. ¿Se podía hacer lo mismo con audio?

## Era 4 — Self-supervised (2019-2021)

### Problema heredado

ASR de calidad requería miles de horas de audio transcrito por humanos — recurso disponible solo para ~10 idiomas. Para los 7,000 idiomas restantes, los modelos eran pobres o inexistentes. La pregunta abierta: ¿se puede pretrainar sobre audio puro (sin transcripciones) y luego fine-tunear con poca data etiquetada?

### Idea clave

**Pretraining contrastivo y predictivo sobre audio crudo.** wav2vec (Schneider et al., FAIR, 2019) entrenó un encoder convolucional a predecir representaciones futuras de la señal con una pérdida contrastiva, sin etiquetas. wav2vec 2.0 (Baevski et al., 2020) lo refinó al estilo BERT: cuantizar representaciones latentes en un codebook discreto, enmascarar segmentos del audio, y predecir las unidades cuantizadas correctas usando un Transformer.

El resultado: con 53,000 horas de audio sin etiquetar para pretraining + 10 minutos de audio etiquetado para fine-tuning, wav2vec 2.0 alcanzaba WER comparable a modelos supervisados de la generación anterior entrenados con miles de horas. HuBERT (Hsu et al., 2021) mejoró la receta con clustering iterativo de targets, dominando luego en TTS, identificación de hablante y reconocimiento de emoción.

### Qué la destronó

wav2vec 2.0 era enormemente eficiente en datos pero seguía requiriendo fine-tuning por dominio/idioma. La frontera natural: ¿se puede pretrainar en escala suficiente para tener un modelo zero-shot multilingüe que funcione fuera de la caja?

## Era 5 — Foundation models (2022-presente)

### Problema heredado

ASR seguía siendo un trabajo de fine-tuning por dominio. Cambiar de "transcribir podcast en inglés" a "transcribir Zoom call en español con ruido de fondo" requería ajuste. Y la generación de audio (TTS, música) vivía en un universo paralelo, con arquitecturas distintas.

### Idea clave

**Audio como una secuencia más, manejada por foundation models.** Whisper (Radford et al., OpenAI, 2022) entrenó un Transformer encoder-decoder estándar sobre 680,000 horas de audio multilingüe pareado con texto raspado de internet — un orden de magnitud más datos que cualquier modelo previo, y ruidosos por construcción. El resultado fue robustez sin fine-tuning a 99 idiomas, ruido, acentos y dominios. Whisper se volvió estándar industrial inmediatamente.

En paralelo, AudioLM (Google, 2022) mostró que la generación de audio (música, ambiente, voz) podía hacerse como modelado autoregresivo de **tokens cuantizados** — el mismo paradigma que GPT, aplicado a audio. VALL-E (Microsoft, 2023) llevó la idea a TTS zero-shot: cloná una voz desde 3 segundos de audio. MusicLM/MusicGen/Suno/Udio (2023-2024) generan música de calidad de producción condicionada en texto, con millones de usuarios.

### Qué viene

La convergencia con los frontier LLMs es el hito en marcha: GPT-4o y Gemini 2.5 ya procesan audio nativamente como entrada/salida, sin pipeline ASR→LLM→TTS. La latencia conversacional bajó a <300ms (Sesame, ChatGPT Voice). En generación, las apuestas activas son **música con control fino** (estems separables, edición por prompt), **audio espacial 3D** para AR/VR, y **detección de deepfakes de voz** como contramedida a la clonación. La pregunta abierta: si los frontier LLMs absorben audio nativamente, ¿queda "audio" como dominio aislado o pasa a ser una modalidad más en modelos generales?
```

**Step 2: Verify build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

**Step 3: Curl-based validation**

```bash
hugo server -D --port 1313 > /tmp/hugo-task2-ola3.log 2>&1 &
sleep 3

URL=http://localhost:1313/neural-network-knowledge/dominios/audio/

curl -s -o /tmp/audio.html "$URL"

# 5 era H2s
grep -c '<h2[^>]*>Era ' /tmp/audio.html  # 5

# Subsections
grep -c "Problema heredado" /tmp/audio.html  # 10
grep -c "Idea clave" /tmp/audio.html  # 10
grep -c "Qué la destronó" /tmp/audio.html  # 8
grep -c "Qué viene" /tmp/audio.html  # 2

# Specific phrases
grep "Hinton, Mohamed y Dahl (2011)" /tmp/audio.html | head -1
grep "Hannun et al., Baidu, 2014" /tmp/audio.html | head -1
grep "Listen, Attend and Spell" /tmp/audio.html | head -1
grep "Radford et al., OpenAI, 2022" /tmp/audio.html | head -1
grep "AudioLM" /tmp/audio.html | head -1

pkill -f "hugo server" || true
sleep 1
```

**Step 4: Commit**

```bash
git add site/content/dominios/audio/_index.md
git commit -m "feat(dominios/audio): eras explicadas (5 subsecciones narrativas)"
```

NO Co-Authored-By trailer.

---

## Task 3: Audio/Voz — SOTA + casos de uso + qué viene + recursos

**Files:**
- Modify: `site/content/dominios/audio/_index.md` (apend al final)

**Step 1: Apender este contenido al final**

```markdown

## Estado del arte hoy

{{< callout type="info" >}}

**Frontier audio (2024-2025).** El audio se está integrando nativamente a los foundation models generales. La generación (música, voz, sonido) alcanza calidad de producción y la conversación por voz baja a latencia humana.

- **Whisper v3** — OpenAI. ASR multilingüe estándar industrial; 99 idiomas, robusto a ruido y acentos.
- **GPT-4o / Gemini 2.5 audio nativo** — frontier LLMs con entrada/salida de voz directa, sin pipeline.
- **Sesame** — TTS conversacional con prosodia natural, latencia <300ms para diálogo en tiempo real.
- **ElevenLabs Multilingual v3** — TTS de producción con expresividad y control fino; clonación de voz desde segundos.
- **VALL-E 2 / NaturalSpeech 3** — TTS zero-shot con calidad de hablante humano.
- **Suno v4 / Udio v2** — música generativa por texto, calidad comercial; millones de usuarios activos.
- **MusicGen / Stable Audio 2** — generación de música y efectos para creadores y producción.

{{< /callout >}}

## Casos de uso reales

- **ASR en producción**: Zoom Live Transcript, Otter.ai, Apple Dictation, Google Recorder, transcripción de podcasts.
- **Asistentes de voz**: Alexa, Siri, Google Assistant; nueva generación con voz directa (ChatGPT Voice, Gemini Live).
- **Subtítulos automáticos**: YouTube, Twitch, plataformas de streaming, accesibilidad en tiempo real.
- **Audiolibros y TTS comercial**: Audible AI Narration, Google Play Books — TTS de producción.
- **Música generativa**: Suno y Udio para creadores; Stable Audio para producción profesional; MusicGen open-source.
- **Clonación de voz**: ElevenLabs para localización de contenido, accesibilidad, dubs multilingües.
- **Análisis de llamadas**: contact centers, compliance regulatorio, análisis de sentiment y QA.
- **Salud y diagnóstico**: detección de Parkinson, depresión y disfonía a partir de voz.

## Qué viene

Las apuestas activas en audio: **modelos de audio nativos en frontier LLMs** (audio in / audio out sin pipeline ASR→LLM→TTS, latencia <300ms), **música generativa con control fino** (estems separables, edición por prompt sobre piezas existentes), **detección de deepfakes de voz** como contramedida industrial a la clonación masiva, **audio espacial generativo** (3D, binaural) para AR/VR, y **foundation models multilingües** que cubran los 7,000 idiomas humanos — Whisper aún cubre <100. La pregunta abierta: ¿cuándo el "asistente de voz" será indistinguible de una llamada con un humano competente?

## Recursos relacionados

**Fundamentos (predecesores conceptuales):**
- [LSTM y GRU](/fundamentos/lstm-gru) — base de DeepSpeech, RNN-T y todos los ASR pre-Transformer.
- [Mecanismo de atención](/fundamentos/mecanismo-atencion) — fundamento de Listen, Attend and Spell y de los Transformers de audio actuales.
- [Self-attention](/fundamentos/self-attention) y [Transformer](/fundamentos/transformer) — la arquitectura sobre la que corren Whisper, AudioLM y VALL-E.
- [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) — el principio que vertebra wav2vec.

**Papers (predecesores adyacentes):**
- [LSTM (Hochreiter 1997)](/papers/lstm-hochreiter-1997) — la red recurrente sobre la que se construyó toda la era 2-3.
- [Bahdanau attention (2015)](/papers/bahdanau-attention-2015) — el mecanismo de atención que LAS adaptó a audio.
- [Seq2Seq (Sutskever 2014)](/papers/seq2seq-sutskever-2014) — el patrón encoder-decoder.
- [Attention is All You Need (Vaswani 2017)](/papers/attention-is-all-you-need-vaswani-2017) — la arquitectura que reemplazó RNNs en audio post-2020.

**Dominios relacionados:**
- [Texto / NLP](/dominios/texto) — donde nacieron LSTM, atención, Transformer y self-supervised, todos transferidos luego a audio.
- [Multimodal](/dominios/multimodal) — donde audio se combina con texto e imagen en frontier LLMs.

---

*Última actualización: 2026-05-05.*
```

**Step 2: Verify build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

**Step 3: Curl-based validation**

```bash
hugo server -D --port 1313 > /tmp/hugo-task3-ola3.log 2>&1 &
sleep 3

URL=http://localhost:1313/neural-network-knowledge/dominios/audio/

curl -s -o /tmp/audio.html "$URL"

# Sections
grep -c "Estado del arte hoy" /tmp/audio.html  # 3
grep -c "Casos de uso reales" /tmp/audio.html  # 3
grep -c "Recursos relacionados" /tmp/audio.html  # 3

# Callout
grep -c "callout" /tmp/audio.html | head -1

# SOTA bullets
grep "Whisper v3" /tmp/audio.html | head -1
grep "GPT-4o" /tmp/audio.html | head -1
grep "Sesame" /tmp/audio.html | head -1
grep "ElevenLabs" /tmp/audio.html | head -1

# Resource links (verify each links to existing material)
grep -oE 'href="[^"]*fundamentos/lstm-gru"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*fundamentos/mecanismo-atencion"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*fundamentos/self-attention"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*fundamentos/transformer"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*fundamentos/aprendizaje-contrastivo"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*papers/lstm-hochreiter-1997"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*papers/bahdanau-attention-2015"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*papers/seq2seq-sutskever-2014"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*papers/attention-is-all-you-need-vaswani-2017"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*dominios/texto"' /tmp/audio.html | head -1
grep -oE 'href="[^"]*dominios/multimodal"' /tmp/audio.html | head -1

# Last update
grep "Última actualización: 2026-05-05" /tmp/audio.html | head -1

pkill -f "hugo server" || true
sleep 1
```

Expected: all resource links resolve to existing files.

**Step 4: Commit**

```bash
git add site/content/dominios/audio/_index.md
git commit -m "feat(dominios/audio): SOTA, casos de uso, que viene y recursos"
```

NO Co-Authored-By trailer.

---

## Task 4: Verificación final, build de producción y push

**Files:** ninguno nuevo.

**Step 1: Confirmar branch**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git branch --show-current  # must be feat/dominios-ola-3
```

**Step 2: Build limpio de producción**

```bash
rm -rf site/public site/resources/_gen 2>/dev/null
cd site && hugo --gc --minify
```

Expected: build sin errores ni warnings nuevos. ~298 pages.

**Step 3: FlexSearch indexa la página nueva**

```bash
python3 -c "
import json
d = json.load(open('public/es.search-data.json'))
keys = [k for k in d.keys() if 'audio' in k.lower()]
print('Audio entries:', len(keys))
for k in keys:
    title = d[k].get('title', '?') if isinstance(d[k], dict) else '?'
    print(' -', k, '|', title)
"
# Search for content terms
grep -c "Whisper\|MFCC\|wav2vec\|VALL-E\|MusicGen" public/es.search-data.json
```

Expected: la página `/neural-network-knowledge/dominios/audio/` aparece con title "Audio / Voz". Términos clave presentes.

**Step 4: Verificar que los stubs restantes siguen renderizando**

```bash
ls public/dominios/video/index.html public/dominios/robotica/index.html public/dominios/estructurados/index.html
```

Expected: los 3 archivos siguen presentes (Ola 3 no debe tocarlos).

**Step 5: Verificar que las páginas completadas previas siguen funcionando**

```bash
ls public/dominios/texto/index.html public/dominios/vision/index.html public/dominios/multimodal/index.html
```

Expected: 3 archivos (intactos desde Olas 1 y 2).

**Step 6: Push y abrir PR**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git push -u origin feat/dominios-ola-3
```

```bash
gh pr create --base main --head feat/dominios-ola-3 --title "feat(dominios): Ola 3 — Audio / Voz" --body "$(cat <<'EOF'
## Summary

Página completa para el dominio **Audio / Voz** de la sección Dominios. Patrón idéntico al de Multimodal (Ola 2).

- **Línea de tiempo de 5 eras**: acústica clásica (1980-2010) → deep speech híbrido (2011-2014) → end-to-end con atención (2014-2018) → self-supervised (2019-2021) → foundation models (2022-presente).
- **18 hitos** distribuidos 3+3+4+3+5: 0 deep, 2 covered (CTC → lstm-gru, Listen-Attend-Spell → mecanismo-atencion), 16 minimal. Sin material específico de audio en el site, todo se enlaza a predecesores adyacentes (LSTM, atención, Transformer).
- **Eras explicadas** (5 subsecciones con Problema heredado / Idea clave / Qué la destronó o Qué viene).
- **Estado del arte 2024-2025**, casos de uso, qué viene, recursos.

Diseño: docs/plans/2026-05-05-dominios-ola-3-design.md. Plan: docs/plans/2026-05-05-dominios-ola-3-plan.md.

Nota de scope: este PR cubre solo Audio. Video se separó a Ola 4 por decisión del usuario para mantener PRs manejables. Robótica/RL y Datos estructurados pasan a Ola 5.

## Test plan

- [ ] cd site && hugo --gc --minify build limpio.
- [ ] Inspección visual desktop + móvil + dark mode en /dominios/audio/.
- [ ] Búsqueda FlexSearch encuentra "Whisper", "wav2vec", "MFCC", "VALL-E" y lleva a la página.
- [ ] Click en hitos covered (CTC, Listen-Attend-Spell) y links de Recursos llevan a Fundamentos / Papers existentes (no 404).
- [ ] Stubs (video, robotica, estructurados) siguen renderizando con su mensaje "Página en construcción".
- [ ] Páginas completas previas (texto, vision, multimodal) renderizan correctamente.
EOF
)"
```

Reportar la URL de la PR creada.

**No commit en este task** — solo verificación, push y PR.

---

## Definition of Done — Ola 3 (Audio / Voz)

- [ ] `/dominios/audio/` página completa: 5 eras + 18 hitos + 5 era subsections + SOTA + casos + qué viene + recursos.
- [ ] Mínimo 800 palabras de prosa narrativa fuera de la timeline.
- [ ] Todos los `link` en hitos y recursos resuelven a archivos existentes.
- [ ] `hugo --gc --minify` build limpio.
- [ ] FlexSearch indexa la página nueva.
- [ ] Stubs (video, robotica, estructurados) intactos.
- [ ] Páginas completas (texto, vision, multimodal) intactas.
- [ ] Branch `feat/dominios-ola-3` pusheada y PR abierta contra `main`.
- [ ] Commits sin Co-Authored-By trailer.

## Riesgos durante implementación

1. **Errores tipográficos en shortcodes** (e.g., `{{< /hift >}}` en lugar de `{{< /hito >}}`) — Hugo falla el build. La spec incluye un trampa deliberada para alertar al implementer; verificar el balance de tags antes del primer build.
2. **Datos puntuales (años, autores)** sin verificar — Code reviewer subagent debe validar como en Texto/Visión/Multimodal.
3. **`covered` para CTC y LAS** — los fundamentos linkeados (`lstm-gru`, `mecanismo-atencion`) cubren el concepto general pero no la aplicación específica. Aceptable pedagógicamente; el flag para reviewer es honesto.
4. **El usuario puede mergear/cambiar ramas durante la sesión** (sucedió en Olas 1 y 2) — Si pasa, verificar `git branch --show-current` tras cada subagent.
