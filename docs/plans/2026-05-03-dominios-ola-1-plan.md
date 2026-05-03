# Sección Dominios — Ola 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Construir la nueva sección "Dominios" del site con su infraestructura (shortcodes + CSS + menú + landing + 7 stubs) y publicar dos dominios completos (Texto/NLP y Visión) con línea de tiempo evolutiva, eras narrativas, estado del arte y casos de uso.

**Architecture:** Tres shortcodes Hugo nuevos (`timeline`, `era`, `hito`) que renderizan HTML/CSS puro sin JavaScript. CSS añadido a `site/assets/css/custom.css` (ya cargado por Hextra automáticamente). Cada página de dominio sigue una plantilla fija de 7 secciones. Las cards de hitos tienen 3 estados (`deep`, `covered`, `minimal`) según material existente. La sección se inserta en el menú con `weight: 15` entre Fundamentos y Clases.

**Tech Stack:** Hugo extended ≥0.112, tema Hextra (vendored vía `go.mod`), CSS con variables del tema para soporte automático de dark mode, Markdown con shortcodes Hugo.

**Diseño de referencia:** [docs/plans/2026-05-03-dominios-design.md](2026-05-03-dominios-design.md).

**Convenciones del codebase verificadas:**
- Hextra carga `assets/css/custom.css` desde `_vendor/github.com/imfing/hextra/layouts/_partials/head.html:32`. Solo hay que agregar al final del archivo existente.
- Variable de color UC ya definida en `:root`: `--primary-hue: 209deg; --primary-saturation: 100%; --primary-lightness: 31%`.
- Modo dark se activa con selector `:is(.dark) .selector`.
- Cards de Hextra: `{{< cards >}}{{< card link icon title subtitle >}}{{< /cards >}}`.
- Iconos de Hextra usados: `academic-cap`, `book-open`, `beaker`, `document-text`, `photograph`, `variable`, `code`, `adjustments`, `eye`.
- Convención de tildes: el contenido nuevo va con tildes correctas (decisión aprobada en diseño). El contenido existente sin tildes no se toca.

**Working directory:** `/Users/robertoaraneda/projects/personal/courses/ia-uc/`. Branch actual: `feat/clase-14-camino-3-interpretabilidad`. Decidir antes de empezar si esta Ola va en una rama nueva (recomendado: `feat/dominios-ola-1`) o continúa en la actual.

**Comando de build local recurrente:** `cd site && hugo server -D` para dev server con live reload, o `cd site && hugo --gc --minify` para validar build de producción.

---

## Task 1: Crear rama de trabajo y estructura de directorios stub

**Files:**
- Create: `site/content/dominios/_index.md`
- Create: `site/content/dominios/texto/_index.md` (stub)
- Create: `site/content/dominios/vision/_index.md` (stub)
- Create: `site/content/dominios/audio/_index.md` (stub)
- Create: `site/content/dominios/video/_index.md` (stub)
- Create: `site/content/dominios/multimodal/_index.md` (stub)
- Create: `site/content/dominios/robotica/_index.md` (stub)
- Create: `site/content/dominios/estructurados/_index.md` (stub)

**Step 1: Crear rama de trabajo**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git checkout -b feat/dominios-ola-1
```

Expected: rama creada, working tree limpio.

**Step 2: Crear landing `dominios/_index.md`**

Contenido inicial (más adelante se expande con grid de cards en Task 13):

```markdown
---
title: "Dominios"
type: docs
weight: 15
sidebar:
  open: true
---

# Dominios

Una vista transversal de cómo cada modalidad de datos forzó la evolución de las arquitecturas neuronales. Para cada dominio recorremos su línea de tiempo: qué problema resolvía cada era, qué idea la definió, qué modelos la encarnaron y qué la destronó.

A diferencia de **Fundamentos** (organizado por concepto teórico) y **Clases** (organizado cronológicamente por sesión del diplomado), aquí el eje es el **dato** — texto, imágenes, audio, video, multimodal, decisiones secuenciales y datos estructurados — y la pregunta guía es: *¿por qué la arquitectura terminó siendo la que es?*

{{< cards >}}
  {{< card link="texto" title="Texto / NLP" subtitle="De n-gramas a LLMs: la era del lenguaje" icon="document-text" >}}
  {{< card link="vision" title="Visión" subtitle="De LeNet a ViT: la jerarquía visual" icon="photograph" >}}
  {{< card link="audio" title="Audio / Voz" subtitle="De MFCC a Whisper: la señal continua" icon="academic-cap" >}}
  {{< card link="video" title="Video" subtitle="De two-stream a Sora: tiempo + espacio" icon="academic-cap" >}}
  {{< card link="multimodal" title="Multimodal" subtitle="De CLIP a VLMs: los puentes entre modalidades" icon="academic-cap" >}}
  {{< card link="robotica" title="Robótica / RL" subtitle="De Q-learning a robot foundation models" icon="academic-cap" >}}
  {{< card link="estructurados" title="Datos estructurados" subtitle="Tabular, series temporales y grafos" icon="academic-cap" >}}
{{< /cards >}}
```

**Step 3: Crear cada uno de los 7 stubs**

Para cada dominio (texto, vision, audio, video, multimodal, robotica, estructurados), crear `site/content/dominios/<dominio>/_index.md` con el formato:

```markdown
---
title: "<Nombre del dominio>"
weight: <N>
sidebar:
  open: true
---

# <Nombre del dominio>

<Una línea introductoria del dominio.>

> **Página en construcción.** Esta sección estará disponible en una próxima ola de la sección Dominios. Ver el plan en [docs/plans/2026-05-03-dominios-design.md](https://github.com/robertoaraneda/diplomado-ia-uc/blob/main/docs/plans/2026-05-03-dominios-design.md).
```

Pesos asignados (orden de aparición en el sidebar):
- texto: 1
- vision: 2
- audio: 3
- video: 4
- multimodal: 5
- robotica: 6
- estructurados: 7

Frases introductorias sugeridas por dominio:
- **Texto:** "Cómo las redes neuronales aprendieron a leer y escribir, desde n-gramas hasta los LLMs actuales."
- **Visión:** "De convoluciones inspiradas en V1 a Vision Transformers: la jerarquía visual aprendida."
- **Audio:** "Procesamiento de la señal continua: del espectrograma a Whisper y los modelos fundacionales de audio."
- **Video:** "El reto de combinar espacio y tiempo: two-stream, 3D-CNN, video transformers y generación temporal coherente."
- **Multimodal:** "Puentes entre modalidades: image captioning, CLIP, modelos visión-lenguaje y generación texto-imagen."
- **Robótica/RL:** "Decisiones secuenciales bajo recompensa: Q-learning, AlphaGo, RLHF y robot foundation models."
- **Estructurados:** "Tabular, series temporales y grafos: cuándo deep learning gana y cuándo XGBoost sigue mandando."

**Step 4: Verificar build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

Expected: build exitoso, sin warnings sobre los nuevos archivos. La sección "Dominios" no aparece en el menú principal todavía (eso es Task 2). Verificar que las URLs `/dominios/`, `/dominios/texto/`, etc. existen en `public/`.

**Step 5: Commit**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git add site/content/dominios/
git commit -m "feat(dominios): estructura inicial — landing + 7 stubs"
```

---

## Task 2: Agregar entrada "Dominios" al menú principal

**Files:**
- Modify: `site/hugo.yaml` (sección `menu.main`)

**Step 1: Editar `hugo.yaml`**

Insertar entre la entrada de Fundamentos (`weight: 10`) y Clases (`weight: 20`):

```yaml
    - name: Dominios
      pageRef: /dominios
      weight: 15
```

**Step 2: Verificar build y navegación**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo server -D
```

Abrir `http://localhost:1313` y verificar:
- "Dominios" aparece en el menú principal entre "Fundamentos" y "Clases".
- Click en "Dominios" lleva a la landing.
- En la landing, las 7 cards renderizan correctamente.
- Click en cada card lleva al stub correspondiente.

Detener el server con Ctrl+C cuando termine la verificación.

**Step 3: Commit**

```bash
git add site/hugo.yaml
git commit -m "feat(dominios): agregar entrada al menu principal (weight 15)"
```

---

## Task 3: Agregar CSS de timeline a `custom.css`

**Files:**
- Modify: `site/assets/css/custom.css` (añadir al final)

**Step 1: Añadir bloque CSS al final del archivo**

```css

/* ============================================================ */
/* Seccion Dominios — Linea de tiempo evolutiva                */
/* ============================================================ */

.timeline-container {
  position: relative;
  margin: 2rem 0 3rem;
  padding-left: 2rem;
  border-left: 2px solid rgba(0, 90, 156, 0.25);
}

:is(.dark) .timeline-container {
  border-left-color: rgba(96, 165, 250, 0.35);
}

.timeline-era {
  position: relative;
  margin-bottom: 2.5rem;
}

.timeline-era::before {
  content: "";
  position: absolute;
  left: -2.5rem;
  top: 0.4rem;
  width: 0.85rem;
  height: 0.85rem;
  border-radius: 50%;
  background: #005a9c;
  border: 3px solid var(--hextra-content-bg, #fff);
  box-shadow: 0 0 0 2px rgba(0, 90, 156, 0.25);
}

:is(.dark) .timeline-era::before {
  background: #60a5fa;
  border-color: #1f2937;
  box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.35);
}

.timeline-era-header {
  margin-bottom: 0.9rem;
}

.timeline-era-years {
  font-size: 0.8rem;
  font-weight: 600;
  color: #005a9c;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

:is(.dark) .timeline-era-years {
  color: #60a5fa;
}

.timeline-era-name {
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 0.15rem;
}

.timeline-hitos-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.85rem;
}

@media (max-width: 768px) {
  .timeline-hitos-grid {
    grid-template-columns: 1fr;
  }
}

.timeline-hito {
  border-radius: 0.5rem;
  padding: 0.85rem 1rem;
  transition: box-shadow 0.2s, transform 0.2s;
  background: rgba(0, 90, 156, 0.02);
}

:is(.dark) .timeline-hito {
  background: rgba(96, 165, 250, 0.05);
}

.timeline-hito-deep {
  border: 1px solid rgba(0, 90, 156, 0.4);
  cursor: pointer;
}

.timeline-hito-deep:hover {
  box-shadow: 0 4px 12px rgba(0, 90, 156, 0.18);
  transform: translateY(-1px);
}

.timeline-hito-covered {
  border: 1px solid rgba(0, 90, 156, 0.2);
  cursor: pointer;
}

.timeline-hito-covered:hover {
  box-shadow: 0 2px 8px rgba(0, 90, 156, 0.1);
}

.timeline-hito-minimal {
  border: 1px dashed rgba(0, 90, 156, 0.3);
  background: rgba(0, 90, 156, 0.04);
}

:is(.dark) .timeline-hito-deep,
:is(.dark) .timeline-hito-covered {
  border-color: rgba(96, 165, 250, 0.35);
}

:is(.dark) .timeline-hito-minimal {
  border-color: rgba(96, 165, 250, 0.4);
  background: rgba(96, 165, 250, 0.06);
}

.timeline-hito-year {
  font-size: 0.75rem;
  font-weight: 700;
  color: #005a9c;
  letter-spacing: 0.04em;
}

:is(.dark) .timeline-hito-year {
  color: #60a5fa;
}

.timeline-hito-name {
  font-size: 0.95rem;
  font-weight: 600;
  margin: 0.15rem 0 0.35rem;
}

.timeline-hito-body {
  font-size: 0.85rem;
  line-height: 1.45;
  color: var(--hextra-content-secondary, #4b5563);
}

:is(.dark) .timeline-hito-body {
  color: #d1d5db;
}

.timeline-hito a,
a.timeline-hito-deep,
a.timeline-hito-covered {
  text-decoration: none;
  color: inherit;
  display: block;
}

.timeline-hito-readmore {
  font-size: 0.75rem;
  font-weight: 600;
  color: #005a9c;
  margin-top: 0.4rem;
  display: inline-block;
}

:is(.dark) .timeline-hito-readmore {
  color: #60a5fa;
}
```

**Step 2: Verificar build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

Expected: build sin errores. CSS aún no se ve aplicado (los shortcodes vienen en Tasks 4-6).

**Step 3: Commit**

```bash
git add site/assets/css/custom.css
git commit -m "feat(dominios): css para timeline (eras + hitos + dark mode)"
```

---

## Task 4: Crear shortcode `era.html`

**Files:**
- Create: `site/layouts/shortcodes/era.html`

**Step 1: Escribir el shortcode**

```go-html-template
{{- $name := .Get "name" -}}
{{- $years := .Get "years" -}}
<div class="timeline-era">
  <div class="timeline-era-header">
    <div class="timeline-era-years">{{ $years }}</div>
    <div class="timeline-era-name">{{ $name }}</div>
  </div>
  <div class="timeline-hitos-grid">
    {{- .Inner -}}
  </div>
</div>
```

**Step 2: Verificar build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

Expected: build sin warnings. El shortcode existe pero aún no se usa en ninguna página.

**Step 3: Commit**

```bash
git add site/layouts/shortcodes/era.html
git commit -m "feat(dominios): shortcode era para encabezado de era en timeline"
```

---

## Task 5: Crear shortcode `hito.html`

**Files:**
- Create: `site/layouts/shortcodes/hito.html`

**Step 1: Escribir el shortcode con las 3 ramas de status**

```go-html-template
{{- $year := .Get "year" -}}
{{- $name := .Get "name" -}}
{{- $status := .Get "status" | default "minimal" -}}
{{- $link := .Get "link" | default "" -}}
{{- $body := .Inner | markdownify -}}
{{- if and (ne $status "minimal") (ne $link "") -}}
<a href="{{ $link | relURL }}" class="timeline-hito timeline-hito-{{ $status }}">
  <div class="timeline-hito-year">{{ $year }}</div>
  <div class="timeline-hito-name">{{ $name }}</div>
  <div class="timeline-hito-body">{{ $body }}</div>
  {{- if eq $status "deep" -}}
  <span class="timeline-hito-readmore">Leer más →</span>
  {{- end -}}
</a>
{{- else -}}
<div class="timeline-hito timeline-hito-{{ $status }}">
  <div class="timeline-hito-year">{{ $year }}</div>
  <div class="timeline-hito-name">{{ $name }}</div>
  <div class="timeline-hito-body">{{ $body }}</div>
</div>
{{- end -}}
```

**Step 2: Verificar build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

Expected: build sin warnings.

**Step 3: Commit**

```bash
git add site/layouts/shortcodes/hito.html
git commit -m "feat(dominios): shortcode hito (3 estados: deep/covered/minimal)"
```

---

## Task 6: Crear shortcode `timeline.html` y validar visualmente

**Files:**
- Create: `site/layouts/shortcodes/timeline.html`
- Create temporally: `site/content/dominios/_test-timeline.md` (página de prueba; se borra al final del task)

**Step 1: Escribir el shortcode wrapper**

```go-html-template
<div class="timeline-container">
  {{- .Inner -}}
</div>
```

**Step 2: Crear página de prueba con contenido representativo de los 3 estados**

Contenido de `site/content/dominios/_test-timeline.md`:

```markdown
---
title: "Test Timeline"
weight: 999
draft: true
---

# Test Timeline

{{< timeline >}}
  {{< era name="Era de prueba A" years="2010-2015" >}}
    {{< hito year="2012" name="Hito DEEP" status="deep" link="/fundamentos/transformer" >}}
      Este hito tiene fundamento dedicado y debe verse con border sólido y "Leer más".
    {{< /hito >}}
    {{< hito year="2014" name="Hito COVERED" status="covered" link="/fundamentos/redes-recurrentes" >}}
      Este hito enlaza a un fundamento relacionado, border más suave.
    {{< /hito >}}
    {{< hito year="2015" name="Hito MINIMAL" status="minimal" >}}
      Este hito no enlaza a nada. **Por qué importó:** validar el estilo punteado.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de prueba B" years="2016-2020" >}}
    {{< hito year="2017" name="Otro DEEP" status="deep" link="/fundamentos/mecanismo-atencion" >}}
      Segundo hito clickeable para verificar grid de 3 columnas.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}
```

Nota: `draft: true` evita que se publique. Para verlo localmente usar `hugo server -D`.

**Step 3: Validación visual con dev server**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo server -D
```

Abrir `http://localhost:1313/dominios/_test-timeline/` y validar:
- ✅ Línea vertical azul con punto por era visible.
- ✅ Hito DEEP: border sólido, hover destacado, "Leer más →" al pie. Click lleva a `/fundamentos/transformer`.
- ✅ Hito COVERED: border más suave, hover sin "Leer más". Click lleva a `/fundamentos/redes-recurrentes`.
- ✅ Hito MINIMAL: border punteado, no clickeable, fondo levemente distinto.
- ✅ Grid de 3 columnas en desktop.
- ✅ Cambiar a viewport móvil (DevTools): grid colapsa a 1 columna.
- ✅ Toggle dark mode (en menú): colores cambian correctamente, todo legible.

Si algo no se ve bien, ajustar CSS de Task 3 antes de continuar (otro commit con `fix(dominios): ...`).

**Step 4: Eliminar página de prueba**

```bash
rm site/content/dominios/_test-timeline.md
```

**Step 5: Commit**

```bash
git add site/layouts/shortcodes/timeline.html
git commit -m "feat(dominios): shortcode timeline + validacion visual"
```

---

## Task 7: Página Texto/NLP — front matter, problema central y timeline

**Files:**
- Modify: `site/content/dominios/texto/_index.md` (reemplaza el stub completo)

**Step 1: Reescribir el stub con la estructura completa hasta la timeline**

Material existente del site para enlazar (verificar rutas exactas con `ls site/content/fundamentos/` antes de escribir):
- `/fundamentos/embeddings-distribuidos` → para Bengio NNLM y word2vec.
- `/fundamentos/redes-recurrentes` → para RNN.
- `/fundamentos/lstm-gru` → para LSTM/GRU.
- `/fundamentos/seq2seq` → para Seq2Seq.
- `/fundamentos/mecanismo-atencion` → para Bahdanau attention.
- `/fundamentos/self-attention` → para self-attention.
- `/fundamentos/transformer` → para Transformer.
- `/fundamentos/positional-encoding` → para positional encoding.
- `/fundamentos/bert` o `/fundamentos/pretraining-bert` → para BERT.
- `/fundamentos/sft` y `/fundamentos/dpo` → para alineamiento.
- `/fundamentos/bpe` → para tokenización.
- `/fundamentos/foundation-models` → para LLMs.

Front matter + secciones 1-3 (problema central + timeline). El contenido de las eras (sección 4) viene en Task 8.

```markdown
---
title: "Texto / NLP"
weight: 1
sidebar:
  open: true
---

# Texto / NLP

## El problema central

El lenguaje natural es **discreto, composicional y ambiguo**. A diferencia de una imagen — donde los píxeles vecinos están altamente correlacionados — en texto la unidad mínima (la palabra o el token) no tiene una métrica natural de "cercanía": *gato* y *perro* son tokens completamente distintos a nivel de símbolo, aunque semánticamente parecidos. Las dependencias importantes pueden estar a una palabra de distancia o a 500 palabras, y el significado de una palabra cambia con el contexto.

Esto fuerza tres decisiones arquitectónicas que vertebran toda la historia del NLP neuronal: (1) cómo representar palabras como vectores densos que capturen similitud semántica, (2) cómo modelar dependencias de largo alcance entre tokens, y (3) cómo entrenar a escala con texto sin etiquetar — porque el texto etiquetado nunca alcanza, pero texto crudo hay infinito.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era pre-neural" years="1948-2010" >}}
    {{< hito year="1948" name="n-gramas (Shannon)" status="minimal" >}}
      Modelos de lenguaje basados en frecuencias de secuencias cortas. **Por qué importó:** estableció el problema de predecir la siguiente palabra y la métrica de perplexity.
    {{< /hito >}}
    {{< hito year="2003" name="Bengio NNLM" status="covered" link="/fundamentos/embeddings-distribuidos" >}}
      Primera red neuronal para modelar lenguaje; introduce embeddings densos aprendidos.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de embeddings" years="2013-2017" >}}
    {{< hito year="2013" name="word2vec" status="covered" link="/fundamentos/embeddings-distribuidos" >}}
      Skip-gram y CBOW: embeddings entrenables a escala que capturan analogías ("rey - hombre + mujer ≈ reina").
    {{< /hito >}}
    {{< hito year="2014" name="GloVe" status="minimal" >}}
      Embeddings basados en factorización de la matriz de coocurrencias global. **Por qué importó:** alternativa a word2vec con mejor uso de estadísticas globales del corpus.
    {{< /hito >}}
    {{< hito year="2014" name="FastText" status="minimal" >}}
      Embeddings que descomponen palabras en n-gramas de caracteres. **Por qué importó:** maneja palabras fuera de vocabulario y morfología rica.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era recurrente y seq2seq" years="2014-2016" >}}
    {{< hito year="2014" name="LSTM aplicado a NLP" status="deep" link="/fundamentos/lstm-gru" >}}
      Redes con memoria de largo plazo capaces de modelar dependencias entre tokens distantes.
    {{< /hito >}}
    {{< hito year="2014" name="Seq2Seq (Sutskever)" status="deep" link="/fundamentos/seq2seq" >}}
      Encoder-decoder con LSTMs: el primer modelo que traducía oraciones completas extremo a extremo.
    {{< /hito >}}
    {{< hito year="2015" name="Bahdanau attention" status="deep" link="/fundamentos/mecanismo-atencion" >}}
      Atención sobre el encoder: rompe el cuello de botella del vector de contexto fijo y permite oraciones largas.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de la atención" years="2017-2020" >}}
    {{< hito year="2017" name="Transformer" status="deep" link="/fundamentos/transformer" >}}
      *Attention is all you need*: self-attention pura, sin recurrencias. Paralelismo masivo en training.
    {{< /hito >}}
    {{< hito year="2018" name="BERT" status="deep" link="/fundamentos/bert" >}}
      Pretraining bidireccional con MLM: el primer modelo que volvió obsoleto entrenar desde cero para cada tarea.
    {{< /hito >}}
    {{< hito year="2018-2019" name="GPT-1 / GPT-2" status="minimal" >}}
      Decoder-only autoregresivo entrenado en texto crudo. **Por qué importó:** mostró que la generación de texto coherente emerge solo con escala y next-token prediction.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de los LLMs" years="2020-presente" >}}
    {{< hito year="2020" name="GPT-3" status="minimal" >}}
      175B parámetros, few-shot in-context learning. **Por qué importó:** la escala desbloqueó capacidades cualitativamente nuevas (razonamiento, programación) sin fine-tuning.
    {{< /hito >}}
    {{< hito year="2022" name="InstructGPT / SFT + RLHF" status="deep" link="/fundamentos/sft" >}}
      Alineamiento por feedback humano: convierte un modelo de lenguaje en un asistente útil y seguro.
    {{< /hito >}}
    {{< hito year="2023" name="DPO" status="deep" link="/fundamentos/dpo" >}}
      Direct Preference Optimization: alineamiento sin RL, equivalente teórico a RLHF pero más simple y estable.
    {{< /hito >}}
    {{< hito year="2023-2025" name="LLMs frontier" status="covered" link="/fundamentos/foundation-models" >}}
      GPT-4/5, Claude, Gemini, LLaMA: razonamiento extendido, herramientas, contexto largo, multimodalidad.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}
```

**Step 2: Verificar visualmente**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo server -D
```

Abrir `http://localhost:1313/dominios/texto/`. Validar:
- ✅ Renderiza la timeline completa con 5 eras y todos los hitos.
- ✅ Cada estado se ve correctamente.
- ✅ Click en hitos `deep`/`covered` lleva a la página correcta de Fundamentos.

**Step 3: Commit**

```bash
git add site/content/dominios/texto/_index.md
git commit -m "feat(dominios/texto): problema central + linea de tiempo (5 eras)"
```

---

## Task 8: Página Texto/NLP — eras explicadas

**Files:**
- Modify: `site/content/dominios/texto/_index.md` (añadir secciones de eras al final)

**Step 1: Apender al archivo las 5 subsecciones de eras**

Después del bloque `{{< /timeline >}}`, añadir:

```markdown

## Era 1 — Pre-neural (1948-2010)

### Problema heredado

Antes de las redes neuronales, modelar lenguaje era un problema de conteo. Shannon (1948) había mostrado que el lenguaje tiene estructura estadística predecible, y los modelos de n-gramas estimaban directamente $P(w_n \mid w_{n-k}, ..., w_{n-1})$ contando frecuencias en corpus. Funcionaban razonablemente para tareas locales (autocompletado, ASR) pero sufrían **dispersidad** — la mayoría de las secuencias de 4 palabras nunca aparecen en el corpus — y no capturaban similitud semántica: para un n-grama, *gato* y *felino* son tokens completamente distintos.

### Idea clave

Usar redes neuronales para **representar palabras como vectores densos en un espacio continuo** donde la cercanía geométrica captura cercanía semántica. El paper de Bengio et al. (2003) introdujo este principio en su *Neural Probabilistic Language Model*: una red feedforward aprendía simultáneamente los embeddings de las palabras y un modelo de lenguaje sobre ellas.

### Qué la destronó

La era pre-neural no terminó por una arquitectura mejor sino por un cambio de **disponibilidad de cómputo**. Cuando entrenar embeddings sobre miles de millones de palabras se volvió viable, la representación distribuida ganó por knockout.

## Era 2 — Embeddings distribuidos (2013-2017)

### Problema heredado

Bengio había probado que los embeddings funcionaban, pero su modelo era costoso: la red feedforward predecía la palabra siguiente con una softmax sobre el vocabulario completo. Para corpus realistas (miles de millones de tokens) era inviable.

### Idea clave

**Aprender embeddings sin modelar la distribución completa.** word2vec (Mikolov, 2013) reformuló el problema: en lugar de predecir la palabra siguiente, entrena con dos tareas mucho más baratas — predecir contexto desde una palabra (skip-gram) o palabra desde contexto (CBOW) — usando **negative sampling** para esquivar la softmax cara. El resultado: embeddings entrenables sobre billones de palabras en horas.

### Qué la destronó

Los embeddings de word2vec son **estáticos**: la palabra *banco* tiene un único vector, independiente de si es entidad financiera o asiento. Para tareas que requieren resolver ambigüedad por contexto — y eso es prácticamente todo NLP serio — esto es una limitación dura. Las RNN y luego los Transformers prometían **embeddings contextuales** que resuelven el sentido en función de la oración completa.

## Era 3 — Recurrente y seq2seq (2014-2016)

### Problema heredado

Las redes feedforward y los embeddings estáticos tratan tokens en aislamiento. El lenguaje es secuencial: *"el gato persigue al ratón"* y *"el ratón persigue al gato"* tienen los mismos tokens y embeddings promedio idénticos, pero significan cosas opuestas.

### Idea clave

Procesar el texto **token por token** en orden, manteniendo un **estado oculto** que se actualiza en cada paso y resume todo lo visto hasta ahora. Las RNN clásicas tenían el problema del gradiente que se desvanece sobre secuencias largas; **LSTM y GRU** resolvieron esto con compuertas que aprenden cuándo retener y cuándo olvidar. Sutskever et al. (2014) llevaron la idea a su punto natural con **Seq2Seq**: un encoder LSTM resume la oración fuente en un vector y un decoder LSTM genera la traducción token a token.

Bahdanau et al. (2015) agregaron **atención** sobre el encoder: en cada paso del decoder, el modelo aprende dónde mirar en la fuente, eliminando el cuello de botella del vector único de contexto.

### Qué la destronó

Las RNN son **secuencialmente irreductibles**: hay que procesar el token $t$ antes de procesar el $t+1$. Esto las hace lentas en GPUs modernas, que están hechas para paralelismo masivo. Y la atención de Bahdanau era un parche sobre la recurrencia. La pregunta natural era: *¿y si quitamos la recurrencia y dejamos solo la atención?*

## Era 4 — Atención pura y pretraining (2017-2020)

### Problema heredado

Las RNN no paralelizaban y la atención existía solo como complemento. El campo necesitaba una arquitectura que pudiera aprovechar GPU y TPU al máximo, y que escalara a contextos cada vez más largos.

### Idea clave

**Self-attention en lugar de recurrencia.** El Transformer (Vaswani et al., 2017) reemplaza por completo las RNN: cada token atiende a todos los demás en una sola operación matricial paralelizable. La información posicional, que la recurrencia provee implícitamente, se inyecta vía positional encoding.

El segundo gran salto fue desacoplar **arquitectura** de **régimen de entrenamiento**. BERT (Devlin et al., 2018) propuso entrenar un Transformer encoder en **Masked Language Modeling** sobre Wikipedia + BookCorpus, y luego fine-tunear sobre tareas específicas. El resultado: un único modelo pre-entrenado destronaba a soluciones especializadas en docenas de benchmarks.

GPT-1 y GPT-2 exploraron la versión decoder-only del mismo principio, entrenada en next-token prediction sobre texto crudo masivo.

### Qué la destronó

BERT y los modelos encoder-only sobreviven en producción para clasificación, búsqueda y embedding. Pero la dirección que terminó dominando fue la **decoder-only autoregresiva escalada**: el camino GPT.

## Era 5 — LLMs y alineamiento (2020-presente)

### Problema heredado

GPT-2 había mostrado que un decoder Transformer entrenado en next-token prediction generaba texto sorprendentemente coherente. Pero seguía siendo un *modelo de lenguaje* — no un asistente. Y nadie había probado qué pasaba al escalarlo cien veces más.

### Idea clave

Tres ideas que se combinaron:

1. **Escala bruta.** GPT-3 (Brown et al., 2020) llevó el tamaño a 175B parámetros y entrenó sobre cientos de miles de millones de tokens. Capacidades cualitativamente nuevas — razonamiento, programación, traducción — emergieron sin entrenamiento específico, vía *in-context learning*.
2. **Alineamiento por feedback humano.** InstructGPT (2022) mostró que un GPT-3 fine-tuneado primero con SFT (datos de demostración) y luego con RLHF (modelo de recompensa entrenado sobre preferencias humanas) se vuelve dramáticamente más útil y seguro. Es lo que separó a GPT-3 de ChatGPT.
3. **Alineamiento sin RL.** DPO (Rafailov et al., 2023) demostró que el objetivo de RLHF se puede reescribir como una pérdida supervisada directa sobre pares de preferencias, eliminando la necesidad del modelo de recompensa y de PPO.

### Qué viene

Esta es la era actual. Las direcciones activas — razonamiento extendido (chain-of-thought, o1, agentes), contexto largo, multimodalidad nativa, herramientas, modelos pequeños competitivos — se desarrollan en paralelo, sin un sucesor claro todavía.
```

**Step 2: Verificar visualmente**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo server -D
```

Validar `/dominios/texto/`:
- ✅ Las 5 secciones `## Era N — ...` renderizan correctamente.
- ✅ El sidebar muestra las eras como sub-entradas.
- ✅ Las matemáticas (`$P(w_n | ...)$`) renderizan vía KaTeX (Hextra ya tiene el passthrough configurado en `hugo.yaml`).

**Step 3: Commit**

```bash
git add site/content/dominios/texto/_index.md
git commit -m "feat(dominios/texto): eras explicadas (5 subsecciones narrativas)"
```

---

## Task 9: Página Texto/NLP — SOTA, casos de uso, qué viene, recursos

**Files:**
- Modify: `site/content/dominios/texto/_index.md` (añadir últimas 3 secciones)

**Step 1: Añadir al final del archivo**

```markdown

## Estado del arte hoy

{{< callout type="info" >}}

**Frontier LLMs (2024-2025).** Los modelos punteros combinan escala (∼1T parámetros), entrenamiento sobre billones de tokens curados, RLHF/DPO y técnicas de razonamiento extendido.

- **GPT-5** — OpenAI. Razonamiento por defecto, contexto extendido, capacidades multimodales nativas (texto + imagen + audio).
- **Claude Opus 4.7** — Anthropic. Contexto de 1M tokens, foco en razonamiento sostenido y uso de herramientas en tareas largas.
- **Gemini 2.5** — Google DeepMind. Multimodal nativo desde el pretraining, integración profunda con búsqueda y herramientas.
- **LLaMA 4** — Meta. Open weights, pesos abiertos competitivos a frontera cerrada en muchos benchmarks.
- **DeepSeek-R1** — DeepSeek. Modelo de razonamiento abierto entrenado con RL puro sobre cadenas de pensamiento.

{{< /callout >}}

## Casos de uso reales

- **Asistentes conversacionales** (ChatGPT, Claude, Gemini): productividad general, redacción, programación.
- **Búsqueda semántica y RAG**: Google AI Overviews, Perplexity, asistentes corporativos sobre documentación interna.
- **Generación de código**: GitHub Copilot, Cursor, Claude Code — completación, refactor y agentes que ejecutan tareas extremo a extremo.
- **Traducción automática**: DeepL, Google Translate (modelos NMT actuales son Transformers descendientes directos de Seq2Seq + atención).
- **Extracción de información estructurada**: del texto libre a JSON conforme a esquema — facturas, historias clínicas, contratos.
- **Moderación y clasificación a escala**: filtros de spam, detección de toxicidad, triaje de tickets de soporte.

## Qué viene

Las apuestas activas hoy — sin un ganador claro — incluyen: **razonamiento explícito** (modelos como o-series y R1 que producen chain-of-thought antes de responder), **agentes** (LLMs que ejecutan secuencias largas de acciones con herramientas), **contextos extra-largos** (millones de tokens, memoria persistente entre conversaciones), **modelos pequeños competitivos** (eficiencia por destilación e instrucción cuidada), y **arquitecturas más allá del Transformer** (Mamba, RWKV, mezclas de expertos a gran escala). Cuál de estas líneas marca el siguiente salto cualitativo es la pregunta abierta de 2025.

## Recursos relacionados

**Fundamentos:**
- [Embeddings distribuidos](/fundamentos/embeddings-distribuidos) — word2vec, GloVe, embeddings contextuales.
- [Redes recurrentes](/fundamentos/redes-recurrentes) y [LSTM/GRU](/fundamentos/lstm-gru).
- [Seq2Seq](/fundamentos/seq2seq) y [mecanismo de atención](/fundamentos/mecanismo-atencion).
- [Self-attention](/fundamentos/self-attention) y [Transformer](/fundamentos/transformer).
- [Positional encoding](/fundamentos/positional-encoding).
- [BPE — tokenización](/fundamentos/bpe).
- [BERT](/fundamentos/bert) y [pretraining BERT](/fundamentos/pretraining-bert).
- [SFT](/fundamentos/sft), [DPO](/fundamentos/dpo) y [KL implícito](/fundamentos/kl-implicito).
- [Foundation models](/fundamentos/foundation-models).

**Papers:**
- [Attention is All You Need (Vaswani 2017)](/papers/attention-is-all-you-need-vaswani-2017).
- [BERT (Devlin 2018)](/papers/bert-devlin-2018).
- [Seq2Seq (Sutskever 2014)](/papers/seq2seq-sutskever-2014).
- [Bahdanau attention (2015)](/papers/bahdanau-attention-2015).
- [LSTM (Hochreiter 1997)](/papers/lstm-hochreiter-1997).
- [GRU (Cho 2014)](/papers/gru-cho-2014).

**Clases del diplomado:**
- Clase 13 — RNNs, seq2seq y atención.
- Clase 14 — Transformer, GPT, BERT, alineamiento.

---

*Última actualización: 2026-05-03.*
```

**Step 2: Verificar visualmente**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo server -D
```

Validar `/dominios/texto/`:
- ✅ Callout "Estado del arte" se ve con estilo destacado de Hextra.
- ✅ Listas de casos de uso y recursos legibles.
- ✅ Todos los enlaces internos funcionan (click cada uno y verificar destino).
- ✅ Pie con fecha de actualización visible.

**Step 3: Commit**

```bash
git add site/content/dominios/texto/_index.md
git commit -m "feat(dominios/texto): SOTA, casos de uso, que viene y recursos"
```

---

## Task 10: Página Visión — front matter, problema central y timeline

**Files:**
- Modify: `site/content/dominios/vision/_index.md` (reemplaza el stub)

**Step 1: Verificar rutas de fundamentos disponibles para Visión**

```bash
ls /Users/robertoaraneda/projects/personal/courses/ia-uc/site/content/fundamentos/ | grep -E "convolucional|vision|transformer"
ls /Users/robertoaraneda/projects/personal/courses/ia-uc/site/content/papers/ | grep -E "alex|vgg|resnet|vit|googlenet"
```

Material disponible:
- `/fundamentos/redes-convolucionales` → CNN.
- `/fundamentos/vision-transformer` → ViT.
- `/fundamentos/transfer-learning`, `/fundamentos/data-augmentation`, `/fundamentos/regularizacion` → técnicas auxiliares (no son hitos pero útiles en recursos).
- `/papers/alexnet-krizhevsky-2012`.
- `/papers/vggnet-simonyan-2014`.
- `/papers/googlenet-szegedy-2014`.
- `/papers/resnet-he-2015`.
- `/papers/vit-dosovitskiy-2021`.

**Step 2: Reescribir el stub con front matter, problema central y timeline**

```markdown
---
title: "Visión"
weight: 2
sidebar:
  open: true
---

# Visión

## El problema central

Una imagen es una grilla de píxeles. Esa grilla tiene **estructura espacial fuerte**: un píxel se parece mucho a sus vecinos, y los objetos relevantes son combinaciones jerárquicas de patrones locales — bordes que forman texturas, texturas que forman partes, partes que forman objetos. Una arquitectura para visión gana o pierde según cuán bien aproveche esa estructura.

Tres tensiones recorren toda la historia: (1) cómo construir **invariancia a traslación, escala y deformaciones** sin perder discriminabilidad, (2) cómo entrenar redes **profundas** sin que el gradiente colapse, y (3) cómo combinar el **sesgo inductivo de localidad** (CNNs) con el **alcance global** (atención) — la pregunta que el Transformer Visual terminó respondiendo en 2020.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era pre-neural" years="1959-2010" >}}
    {{< hito year="1959" name="Hubel y Wiesel" status="minimal" >}}
      Descubrimiento de células simples y complejas en V1 del gato. **Por qué importó:** inspiró el campo receptivo local, base de toda CNN posterior.
    {{< /hito >}}
    {{< hito year="1980" name="Neocognitron (Fukushima)" status="minimal" >}}
      Arquitectura jerárquica con capas alternadas de detección y agrupamiento. **Por qué importó:** prototipo conceptual de la convolución y el pooling.
    {{< /hito >}}
    {{< hito year="1998" name="LeNet-5" status="minimal" >}}
      LeCun: CNN entrenable con backprop para dígitos manuscritos (MNIST). **Por qué importó:** demostró que las CNNs eran prácticas, pero fueron ignoradas por el campo durante una década por falta de cómputo.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era CNN" years="2012-2014" >}}
    {{< hito year="2012" name="AlexNet" status="covered" link="/papers/alexnet-krizhevsky-2012" >}}
      Ganó ImageNet 2012 por margen abrumador con GPUs, ReLU y dropout. El paper que reinició el deep learning moderno.
    {{< /hito >}}
    {{< hito year="2014" name="VGGNet" status="covered" link="/papers/vggnet-simonyan-2014" >}}
      Profundidad uniforme (3x3, stride 1) hasta 19 capas. Demostró que más profundidad = mejor representación.
    {{< /hito >}}
    {{< hito year="2014" name="GoogLeNet / Inception" status="covered" link="/papers/googlenet-szegedy-2014" >}}
      Módulos Inception con convoluciones a múltiples escalas. Más profundo y más eficiente que VGG.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era residual" years="2015-2017" >}}
    {{< hito year="2015" name="ResNet" status="covered" link="/papers/resnet-he-2015" >}}
      Conexiones residuales (skip connections) que permitieron entrenar redes de 152+ capas sin que el gradiente colapsara. Cambió permanentemente cómo se diseñan redes profundas.
    {{< /hito >}}
    {{< hito year="2016" name="DenseNet" status="minimal" >}}
      Conexiones densas: cada capa recibe la concatenación de todas las anteriores. **Por qué importó:** reutilización máxima de features con menos parámetros que ResNet.
    {{< /hito >}}
    {{< hito year="2017" name="MobileNet" status="minimal" >}}
      Convoluciones separables en profundidad para móviles. **Por qué importó:** primera familia diseñada para inferencia eficiente en dispositivos.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de detección y segmentación" years="2014-2018" >}}
    {{< hito year="2014" name="R-CNN" status="minimal" >}}
      Region proposals + CNN para clasificar cada región. **Por qué importó:** primera arquitectura de detección extremo a extremo basada en CNNs.
    {{< /hito >}}
    {{< hito year="2015" name="Faster R-CNN" status="minimal" >}}
      Region Proposal Network integrado dentro de la CNN. **Por qué importó:** detección viable en tiempo casi real.
    {{< /hito >}}
    {{< hito year="2016" name="YOLO" status="minimal" >}}
      Detección como única regresión sobre toda la imagen. **Por qué importó:** detección a 60+ FPS, abrió la puerta a robótica y video.
    {{< /hito >}}
    {{< hito year="2015" name="U-Net" status="minimal" >}}
      Encoder-decoder con skip connections para segmentación médica. **Por qué importó:** sigue siendo el caballo de batalla de segmentación biomédica.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era Transformer" years="2020-presente" >}}
    {{< hito year="2020" name="Vision Transformer (ViT)" status="deep" link="/fundamentos/vision-transformer" >}}
      Aplica un Transformer puro sobre parches de la imagen. Con suficiente data y escala, supera a CNNs sin sesgos inductivos visuales explícitos.
    {{< /hito >}}
    {{< hito year="2021" name="Swin Transformer" status="minimal" >}}
      Transformer jerárquico con ventanas locales que recuperan parte del sesgo inductivo de las CNNs. **Por qué importó:** ViT eficiente para tareas densas (detección, segmentación).
    {{< /hito >}}
    {{< hito year="2021" name="CLIP" status="minimal" >}}
      ViT entrenado por contraste con texto en pares imagen-caption. **Por qué importó:** puente con el dominio multimodal; visión cero-shot por texto.
    {{< /hito >}}
    {{< hito year="2023" name="SAM (Segment Anything)" status="minimal" >}}
      Foundation model para segmentación con prompts. **Por qué importó:** segmentación zero-shot sobre cualquier imagen.
    {{< /hito >}}
    {{< hito year="2024-2025" name="Modelos generativos (Diffusion / Sora)" status="minimal" >}}
      Stable Diffusion 3, Imagen, DALL·E 3, Sora. **Por qué importó:** generación fotorrealista por texto y video como aplicaciones masivas.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}
```

**Step 3: Verificar visualmente y commit**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo server -D
```

Validar `/dominios/vision/` igual que en Task 7. Luego:

```bash
git add site/content/dominios/vision/_index.md
git commit -m "feat(dominios/vision): problema central + linea de tiempo (5 eras)"
```

---

## Task 11: Página Visión — eras explicadas

**Files:**
- Modify: `site/content/dominios/vision/_index.md`

**Step 1: Apender al archivo las 5 subsecciones de eras**

Después del `{{< /timeline >}}`, añadir:

```markdown

## Era 1 — Pre-neural (1959-2010)

### Problema heredado

Antes de las redes profundas, la visión por computador era principalmente **ingeniería de features**: SIFT, HOG, SURF y descriptores hechos a mano que codificaban robustez a iluminación, rotación y escala. Sobre esos features se entrenaban SVMs o random forests. Los resultados eran respetables pero saturaban: cada nueva tarea requería diseñar features específicas, y los benchmarks como ImageNet se resistían a superar el ~25% de error en top-5.

### Idea clave

**Aprender la jerarquía de features en lugar de diseñarla.** Las CNNs llevaban décadas de existencia conceptual — Hubel y Wiesel habían descrito el campo receptivo local en V1, Fukushima había construido el Neocognitron, y LeCun había mostrado con LeNet-5 que una CNN entrenada con backpropagation funcionaba sobre dígitos. Pero el cómputo y los datos disponibles no alcanzaban para imágenes naturales.

### Qué la destronó

Tres factores convergieron: GPUs entrenables (CUDA, 2007), ImageNet (2009) con 1.2M imágenes etiquetadas, y un equipo de Toronto dispuesto a entrenar una CNN profunda sobre ese dataset. El resultado fue AlexNet.

## Era 2 — CNNs (2012-2014)

### Problema heredado

LeNet-5 funcionaba para dígitos pero no para imágenes naturales: 6 capas no alcanzaban para representar la complejidad visual del mundo real. La pregunta era si las CNNs podían escalar.

### Idea clave

**Escalar CNNs en datos, profundidad y cómputo.** AlexNet (Krizhevsky, 2012) puso 8 capas, 60M parámetros, dos GPUs, ReLU en lugar de tanh, dropout para regularizar y data augmentation. Ganó ImageNet 2012 con 15.3% top-5 error contra 26.2% del segundo lugar. Ese delta convenció al campo de que el deep learning no era ruido.

VGG (2014) llevó la idea al extremo de la simplicidad: solo 3x3 conv stride 1 + 2x2 max-pool, hasta 19 capas. Demostró que la profundidad uniforme era una receta robusta. GoogLeNet (2014) tomó la dirección opuesta — módulos Inception con convoluciones a múltiples escalas en paralelo — y mostró que se podía ser más profundo y más eficiente simultáneamente.

### Qué la destronó

Al intentar pasar de 20 a 30, 50, 100 capas, los gradientes colapsaban. La red simplemente no entrenaba mejor con más profundidad — empeoraba.

## Era 3 — Conexiones residuales (2015-2017)

### Problema heredado

El problema de degradación en redes profundas: agregar capas a una red ya buena la empeoraba, incluso al ignorar overfitting. Era un problema de **optimización**, no de capacidad.

### Idea clave

**Skip connections.** ResNet (He et al., 2015) propuso que cada bloque aprendiera una *función residual* $F(x) = H(x) - x$ en lugar de la transformación completa $H(x)$. La identidad $x$ se sumaba directamente a la salida del bloque. Esto convertía el camino del gradiente en una autopista: si el bloque no aprendía nada útil, no estorbaba. Permitió entrenar redes de 152 capas, ganar ImageNet 2015 y volverse el bloque básico de prácticamente toda arquitectura profunda posterior — incluyendo el Transformer.

DenseNet generalizó la idea con conexiones densas, MobileNet la adaptó a inferencia móvil con convoluciones separables. La era residual no fue un nuevo paradigma sino la consolidación del paradigma CNN.

### Qué la destronó

Las CNNs eran imbatibles en clasificación de imágenes naturales, pero tenían un sesgo fuerte: la convolución asume que las features útiles son locales. Eso es cierto en muchas escalas, pero no en todas. Para tareas que requieren razonamiento global o relaciones de largo alcance, la atención prometía algo mejor.

## Era 4 — Detección y segmentación (2014-2018)

### Problema heredado

La clasificación de imágenes responde *qué hay en esta imagen*. Pero las aplicaciones reales — conducción autónoma, robótica, imagenología médica — requieren saber **dónde** está cada cosa y a veces **delinearla a nivel de píxel**.

### Idea clave

Esta era es paralela a las eras CNN y residual; se desarrolló sobre las mismas backbones (AlexNet, VGG, ResNet). Las ideas centrales son:

- **R-CNN family** (R-CNN, Fast, Faster R-CNN, Mask R-CNN): generar propuestas de regiones y clasificar cada una.
- **Single-stage** (YOLO, SSD, RetinaNet): tratar la detección como una única regresión densa sobre la imagen.
- **Encoder-decoder con skip connections** (U-Net, FPN): para segmentación, donde cada píxel necesita una predicción.

### Qué la destronó

Estas arquitecturas siguen vigentes, pero el ecosistema migró progresivamente a backbones Transformer. Y la llegada de SAM (2023) cambió la conversación: ya no se entrena un modelo de segmentación por dataset, sino que se prompt-tunea un foundation model.

## Era 5 — Vision Transformer y foundation models (2020-presente)

### Problema heredado

Las CNNs habían dominado por un sesgo inductivo: localidad y equivariancia a traslación. Pero ese sesgo es también una limitación: la información de un lado de la imagen tarda muchas capas en interactuar con el otro lado. La pregunta abierta de finales de 2010s era si una arquitectura sin sesgo inductivo visual explícito — atención pura sobre parches — podía funcionar dada suficiente data.

### Idea clave

**Tratar una imagen como una secuencia de parches.** ViT (Dosovitskiy et al., 2020) divide la imagen en parches 16x16, los proyecta a tokens y los pasa por un Transformer encoder estándar. Sin convoluciones, sin pooling, sin sesgo de localidad. Con datasets enormes (JFT-300M) y entrenamiento suficiente, ViT supera a CNNs comparables. La interpretación: el sesgo inductivo de las CNNs es una ayuda con poca data y un techo con mucha data.

Swin Transformer (2021) combinó lo mejor de ambos mundos: jerarquía y atención local en ventanas. CLIP (2021) entrenó ViT por contraste con texto, abriendo la era multimodal. SAM (2023) volvió la segmentación un problema zero-shot. Y los modelos generativos (Stable Diffusion 3, Sora) llevaron la generación de imágenes y video a producción masiva.

### Qué viene

La frontera actual son los **modelos visuales fundacionales** entrenados a la escala de los LLMs, capaces de razonar sobre imágenes complejas (GPT-4V, Gemini, Claude con visión), y los **modelos generativos de video** que requieren coherencia temporal — la próxima gran prueba.
```

**Step 2: Verificar visualmente y commit**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo server -D
```

Validar `/dominios/vision/` ahora con las 5 secciones de eras. Luego:

```bash
git add site/content/dominios/vision/_index.md
git commit -m "feat(dominios/vision): eras explicadas (5 subsecciones narrativas)"
```

---

## Task 12: Página Visión — SOTA, casos de uso, qué viene, recursos

**Files:**
- Modify: `site/content/dominios/vision/_index.md`

**Step 1: Añadir al final**

```markdown

## Estado del arte hoy

{{< callout type="info" >}}

**Visión como sub-modalidad de los foundation models (2024-2025).** La visión moderna no compite por sí sola; aparece como capacidad nativa de modelos generales.

- **GPT-4V / GPT-5 Vision** — análisis de imágenes complejas, gráficos, diagramas, documentos.
- **Claude Vision** — Anthropic. Foco en comprensión profunda de imágenes técnicas y diagramas.
- **Gemini 2.5 Pro Vision** — multimodal nativo desde el pretraining.
- **SAM 2** — Meta. Segmentación zero-shot extendida a video.
- **Stable Diffusion 3 / Imagen 3** — generación texto-imagen producción.
- **Sora** — OpenAI. Generación de video con coherencia temporal extendida.
- **DINOv2** — Meta. Foundation model self-supervised para visión, base de muchas pipelines downstream.

{{< /callout >}}

## Casos de uso reales

- **Conducción autónoma**: Tesla FSD, Waymo, Cruise — detección, segmentación, predicción de trayectorias en tiempo real.
- **Diagnóstico médico**: dermatología, radiología, oftalmología — desde Inception en retinopatía diabética (2016) hasta foundation models médicos actuales.
- **Búsqueda visual**: Google Lens, Pinterest Lens, Amazon — search por imagen + texto combinado (CLIP-style).
- **Moderación de contenido**: detección de NSFW, copyright, deepfakes a escala en redes sociales.
- **Robótica industrial**: pick-and-place, control de calidad, inspección visual.
- **Generación creativa**: Midjourney, Stable Diffusion, Adobe Firefly — del concept art al diseño de producto.
- **OCR y comprensión de documentos**: facturas, formularios, contratos — donde visión + texto se cruzan.

## Qué viene

Las apuestas activas en visión hoy: **modelos generativos de video con coherencia temporal extendida** (más allá de Sora), **agentes visuales** (modelos que actúan sobre interfaces gráficas como humanos), **3D nativo** (Gaussian splatting, NeRFs como paso previo a foundation models 3D), **visión embebida** (modelos eficientes para edge), y **vision-language-action** para robótica (RT-2, π0). La integración cada vez más profunda con LLMs hace probable que el campo de "visión por computador" como disciplina aislada se diluya en favor de modelos multimodales generales.

## Recursos relacionados

**Fundamentos:**
- [Redes convolucionales](/fundamentos/redes-convolucionales).
- [Vision Transformer](/fundamentos/vision-transformer).
- [Transfer learning](/fundamentos/transfer-learning).
- [Data augmentation](/fundamentos/data-augmentation).
- [Regularización](/fundamentos/regularizacion).
- [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo).

**Papers:**
- [AlexNet (Krizhevsky 2012)](/papers/alexnet-krizhevsky-2012).
- [VGGNet (Simonyan 2014)](/papers/vggnet-simonyan-2014).
- [GoogLeNet (Szegedy 2014)](/papers/googlenet-szegedy-2014).
- [ResNet (He 2015)](/papers/resnet-he-2015).
- [ViT (Dosovitskiy 2021)](/papers/vit-dosovitskiy-2021).
- [Batch Normalization (Ioffe 2015)](/papers/batch-norm-ioffe-2015).
- [Dropout (Srivastava 2014)](/papers/dropout-srivastava-2014).
- [Mixup (Zhang 2017)](/papers/mixup-zhang-2017).
- [Transferable features (Yosinski 2014)](/papers/transferable-features-yosinski-2014).

**Clases del diplomado:**
- Clases sobre CNNs, ResNets y backbones modernas.
- Clases sobre ViT y atención visual.

---

*Última actualización: 2026-05-03.*
```

**Step 2: Verificar y commit**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo server -D
```

Validar `/dominios/vision/` página completa.

```bash
git add site/content/dominios/vision/_index.md
git commit -m "feat(dominios/vision): SOTA, casos de uso, que viene y recursos"
```

---

## Task 13: Refrescar landing y stats

**Files:**
- Modify: `site/content/dominios/_index.md` (la landing — los iconos pueden necesitar ajuste tras ver los stubs)
- Modify: `site/content/_index.md` (home — actualizar bloque de stats)

**Step 1: Validar iconos del landing**

Probar el landing en navegador con `hugo server -D` y `http://localhost:1313/dominios/`. Si algún icono se ve mal o repetido, sustituir por otro de la lista oficial de Hextra (`heroicons` v1). Sugerencias alternativas: `chip`, `cube`, `cube-transparent`, `microphone`, `film`, `puzzle`, `globe-alt`, `cog`, `sparkles`.

Si todos los iconos lucen aceptables, este step no genera cambios.

**Step 2: Actualizar stats en home**

Editar `site/content/_index.md`. En el bloque `<div class="stats-grid">`, agregar (o ajustar el grid de 4 a 5 elementos):

```html
<div class="stat-item">
<span class="stat-number">7</span>
<span class="stat-label">Dominios</span>
</div>
```

Posición sugerida: después de "Fundamentos" o como primer elemento (queda a criterio visual). Verificar que el CSS del grid soporta 5 elementos sin romperse — si Hextra usa `grid-template-columns: repeat(4, 1fr)`, ajustar a `repeat(5, 1fr)` en el CSS de `.stats-grid` dentro de `assets/css/custom.css` (buscar la regla existente).

**Step 3: Verificar visualmente**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo server -D
```

Validar:
- ✅ Home muestra 5 stats incluyendo "7 Dominios".
- ✅ Landing `/dominios/` con grid de 7 cards bien alineado, sin iconos rotos.
- ✅ Click en cada card abre la página correspondiente (texto y vision con contenido completo, los otros 5 con stub).

**Step 4: Commit**

```bash
git add site/content/dominios/_index.md site/content/_index.md site/assets/css/custom.css
git commit -m "feat(dominios): refrescar landing + stat 7 Dominios en home"
```

---

## Task 14: Verificación final, build de producción y push

**Files:** ninguno nuevo.

**Step 1: Build limpio de producción**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

Expected: build sin errores ni warnings. Anotar tiempo de build.

**Step 2: Verificación FlexSearch**

```bash
hugo server
```

Abrir `http://localhost:1313/`, abrir el buscador (Cmd/Ctrl+K si Hextra lo tiene atajado, o el icono de búsqueda) y probar:
- Buscar "word2vec" → debe aparecer la página `/dominios/texto/`.
- Buscar "Vision Transformer" → debe aparecer `/dominios/vision/`.
- Buscar "ResNet" → debe aparecer la era residual de Visión + el paper existente.

Si algo no aparece, revisar que `enableRobotsTXT` y FlexSearch en `hugo.yaml` (`params.search.flexsearch.index: content`) cubran las nuevas páginas.

**Step 3: Inspección visual completa**

Con dev server corriendo, abrir cada URL y verificar en desktop + móvil + dark mode:

- `/` (home) — stats actualizados.
- `/dominios/` — grid de 7 cards, todos navegables.
- `/dominios/texto/` — timeline + 5 eras + SOTA + casos de uso + recursos. Todos los enlaces internos funcionan.
- `/dominios/vision/` — idem.
- `/dominios/audio/`, `/dominios/video/`, `/dominios/multimodal/`, `/dominios/robotica/`, `/dominios/estructurados/` — stubs renderizan correctamente.

Toggle dark mode y repetir verificación visual rápida sobre las dos páginas completas.

**Step 4: Verificar enlaces no rotos**

Si hay `htmltest` o `lychee` instalado:

```bash
cd site && hugo --gc --minify
htmltest public/    # o: lychee --offline public/
```

Si no hay tooling, hacer pase manual sobre las dos páginas grandes (ya cubierto en Step 3).

**Step 5: Push y abrir PR**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git push -u origin feat/dominios-ola-1
```

Abrir PR contra `main` con título: `feat(dominios): Ola 1 — infraestructura + Texto + Visión` y cuerpo que enlace al diseño y describa qué incluye y qué queda para olas siguientes.

**No commit en este task** — solo verificación y push.

---

## Definition of Done — Ola 1

- [ ] Sección "Dominios" aparece en el menú principal entre Fundamentos y Clases.
- [ ] Landing `/dominios/` con grid de 7 cards navegables.
- [ ] 5 stubs `/dominios/<dominio>/` renderizan correctamente con mensaje "Página en construcción".
- [ ] Tres shortcodes (`timeline`, `era`, `hito`) implementados, documentados y probados visualmente con los 3 estados (`deep`, `covered`, `minimal`).
- [ ] CSS de timeline en `custom.css` con soporte de dark mode y responsive.
- [ ] `/dominios/texto/` página completa: 5 eras + SOTA + casos + qué viene + recursos. Mínimo 800 palabras de prosa narrativa fuera de la timeline.
- [ ] `/dominios/vision/` página completa: 5 eras + SOTA + casos + qué viene + recursos. Mínimo 800 palabras de prosa narrativa.
- [ ] Stats en home actualizados con "7 Dominios".
- [ ] `hugo --gc --minify` build limpio sin warnings.
- [ ] FlexSearch indexa las páginas nuevas.
- [ ] Verificación visual en desktop + móvil + dark mode aprobada.
- [ ] Branch `feat/dominios-ola-1` pusheada y PR abierta contra `main`.

## Riesgos durante implementación

1. **Hextra no expone variable CSS exacta para `--hextra-content-bg`.** Mitigación: si las cards de stub se ven mal en dark mode, reemplazar por colores hex con `:is(.dark)`.
2. **Iconos de Hextra repetidos en la landing.** Mitigación: el catálogo es chico; se puede repetir con icono distinto entre dominios cercanos. No es deal-breaker.
3. **Una de las rutas de fundamento referenciada no existe.** Mitigación: si un enlace 404, cambiar el `status="deep"` a `status="minimal"` y dejar el contenido inline.
4. **El passthrough de `$...$` en hugo.yaml no captura una ecuación.** Mitigación: usar `\(...\)` como delimitador inline alternativo (ya está habilitado en `hugo.yaml`).
5. **El dev server con `-D` no purga páginas eliminadas.** Mitigación: matar y reiniciar el server tras eliminar el `_test-timeline.md` de Task 6.
