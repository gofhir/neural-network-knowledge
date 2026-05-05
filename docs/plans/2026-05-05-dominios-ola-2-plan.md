# Sección Dominios — Ola 2 (Multimodal) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reemplazar el stub `dominios/multimodal/` con una página completa de dominio Multimodal que reaproveche el material existente (CLIP, show-and-tell, show-attend-tell, bottom-up-attention, relation-networks) y narre la evolución desde captioning temprano hasta los VLMs y modelos generativos actuales.

**Architecture:** Una sola página Markdown (`site/content/dominios/multimodal/_index.md`) construida incrementalmente en 3 commits siguiendo el molde de Texto/Visión: Task 1 = front matter + intro + timeline; Task 2 = era subsections; Task 3 = SOTA + casos + recursos. La infraestructura (shortcodes `timeline`/`era`/`hito`, CSS, menú principal, landing) ya existe y se mergeó en `main` con la Ola 1.

**Tech Stack:** Hugo + tema Hextra (vendored vía `go.mod`), Markdown con shortcodes Hugo, KaTeX para fórmulas inline, FlexSearch para búsqueda. baseURL del site: `/neural-network-knowledge/`.

**Diseño de referencia:** [docs/plans/2026-05-05-dominios-ola-2-design.md](2026-05-05-dominios-ola-2-design.md).

**Convenciones del codebase verificadas:**
- Shortcodes ya disponibles: `{{< timeline >}}`, `{{< era name="..." years="..." >}}`, `{{< hito year="..." name="..." status="..." link="..." >}}`. CSS soporta light/dark + responsive.
- Status taxonomy: `deep` (Fundamento dedicado), `covered` (mencionado en otro Fundamento/Paper), `minimal` (descripción inline 2-3 líneas, sin enlace).
- Front matter para páginas de dominio: `title`, `weight` (multimodal=5), `sidebar.open: true`. `type: docs` cascadea desde `dominios/_index.md`.
- Hextra `{{< callout type="info" >}}` para SOTA box.
- Sin Co-Authored-By en commits (preferencia del usuario).
- Español con tildes correctas en contenido nuevo.

**Working directory:** `/Users/robertoaraneda/projects/personal/courses/ia-uc/`. **Branch:** `feat/dominios-ola-2` (ya creada desde `main` en la sesión actual; verificar con `git branch --show-current`).

**Comando de build local recurrente:** `cd site && hugo --gc --minify` para validación de producción; `hugo server` para dev local.

**Estado actual de `site/content/dominios/multimodal/_index.md`** (stub heredado de Ola 1):
```markdown
---
title: "Multimodal"
weight: 5
sidebar:
  open: true
---

# Multimodal

Puentes entre modalidades: image captioning, CLIP, modelos visión-lenguaje y generación texto-imagen.

> **Página en construcción.** Esta sección estará disponible en una próxima ola de la sección Dominios. Ver el plan en [docs/plans/2026-05-03-dominios-design.md](https://github.com/robertoaraneda/diplomado-ia-uc/blob/main/docs/plans/2026-05-03-dominios-design.md).
```

Task 1 sobrescribe completamente este stub.

---

## Task 1: Front matter + problema central + línea de tiempo

**Files:**
- Modify: `site/content/dominios/multimodal/_index.md` (sobrescribir el stub completo)

**Step 1: Verificar fundamentos y papers referenciados**

Antes de escribir, validar que cada `link` apunta a un archivo existente. Si alguno falta, downgradear ese hito a `status="minimal"` y eliminar `link`.

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
# Fundamentos
ls site/content/fundamentos/{aprendizaje-contrastivo,foundation-models}.md 2>&1
# Papers
ls site/content/papers/{show-and-tell-vinyals-2015,show-attend-tell-xu-2015,bottom-up-attention-anderson-2018,relation-networks-santoro-2017,clip-radford-2021}.md 2>&1
```

Expected: los 7 archivos existen (verificado en exploración del diseño). Si alguno faltara, reportar y downgradear.

**Step 2: Sobrescribir el stub con el siguiente contenido EXACTO**

```markdown
---
title: "Multimodal"
weight: 5
sidebar:
  open: true
---

# Multimodal

## El problema central

Las modalidades (texto, imagen, audio, video) viven en **espacios incompatibles**. El texto es discreto y composicional — un vocabulario finito de tokens combinables por reglas sintácticas. Una imagen es una grilla continua de píxeles con estructura espacial. Un audio es una señal temporal continua. Hacerlas dialogar requiere proyectarlas a un espacio común o conectarlas con atención cruzada — y la elección entre ambas opciones vertebra una década de arquitecturas.

Tres tensiones definen el campo: (1) cómo **alinear modalidades sin pares anotados** — los datasets curados (COCO, Flickr30k) son pequeños y caros, mientras que pares ruidosos imagen-caption raspados de la web son virtualmente infinitos; (2) cómo dejar que un LLM **"vea" sin destruir su capacidad lingüística** — entrenamiento conjunto vs adaptadores ligeros que conectan un encoder visual congelado al LLM; (3) cómo **generar coherencia cross-modal** — un modelo que produce video debe respetar física, identidad y composición a través del tiempo, no solo sintaxis local.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era de captioning temprano" years="2014-2016" >}}
    {{< hito year="2015" name="Show and Tell" status="covered" link="/papers/show-and-tell-vinyals-2015" >}}
      Vinyals et al.: CNN encoder + LSTM decoder para generar descripciones de imágenes extremo a extremo. Primer captioner neural competente.
    {{< /hito >}}
    {{< hito year="2015" name="Show, Attend and Tell" status="covered" link="/papers/show-attend-tell-xu-2015" >}}
      Xu et al.: añade atención sobre regiones de la imagen en cada paso del decoder. El modelo "mira" donde necesita para generar la siguiente palabra.
    {{< /hito >}}
    {{< hito year="2015" name="VQA dataset" status="minimal" >}}
      Antol et al.: Visual Question Answering como benchmark — *"¿De qué color es el casco?"* sobre una imagen. **Por qué importó:** forzó al campo más allá del captioning hacia razonamiento sobre imagen + texto.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de atención visual estructurada" years="2017-2018" >}}
    {{< hito year="2017" name="Relation Networks" status="covered" link="/papers/relation-networks-santoro-2017" >}}
      Santoro et al.: módulo que razona sobre **pares de objetos** detectados, no sobre features uniformes de la imagen. Resolvió tareas de razonamiento visual donde CNNs puras fallaban.
    {{< /hito >}}
    {{< hito year="2018" name="Bottom-Up and Top-Down Attention" status="covered" link="/papers/bottom-up-attention-anderson-2018" >}}
      Anderson et al.: la atención visual opera sobre regiones detectadas con Faster R-CNN, no sobre la grilla densa. Estado del arte en captioning y VQA durante 2018-2019.
    {{< /hito >}}
    {{< hito year="2018-2019" name="GQA / VCR" status="minimal" >}}
      Hudson & Manning (GQA), Zellers et al. (VCR): benchmarks de razonamiento visual con preguntas composicionales y de sentido común. **Por qué importó:** revelaron lo lejos que estaban los modelos de razonar realmente.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de pretraining multimodal" years="2019-2020" >}}
    {{< hito year="2019" name="ViLBERT" status="minimal" >}}
      Lu et al.: dos torres BERT (texto + imagen) con cross-attention entre ellas, preentrenadas con MLM extendido. **Por qué importó:** primer pretraining multimodal transferible.
    {{< /hito >}}
    {{< hito year="2019" name="LXMERT" status="minimal" >}}
      Tan & Bansal: variante con tres encoders (texto, visión, cross-modal) y cinco tareas de pretraining. **Por qué importó:** fijó el patrón "encoder por modalidad + cross-attention".
    {{< /hito >}}
    {{< hito year="2020" name="UNITER / OSCAR" status="minimal" >}}
      Chen et al. (UNITER), Li et al. (OSCAR): unifican BERT + visión en un solo Transformer y agregan etiquetas de objetos como "ancla" entre texto e imagen. **Por qué importó:** simplificaron la arquitectura sin perder rendimiento.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era contrastiva y zero-shot" years="2021-2022" >}}
    {{< hito year="2021" name="CLIP" status="deep" link="/fundamentos/aprendizaje-contrastivo" >}}
      Radford et al. (OpenAI): un encoder de texto y uno de imagen entrenados contrastivamente sobre 400M pares imagen-caption raspados de la web. Clasificación zero-shot competitiva con modelos supervisados.
    {{< /hito >}}
    {{< hito year="2021" name="ALIGN" status="minimal" >}}
      Jia et al. (Google): replica CLIP a 1.8B pares ruidosos, demostrando que la calidad del scraping importa menos que el volumen. **Por qué importó:** validó el contraste a escala como receta robusta.
    {{< /hito >}}
    {{< hito year="2022" name="BLIP" status="minimal" >}}
      Li et al. (Salesforce): unifica entendimiento (CLIP-style) y generación (captioning) en un mismo modelo, con pretraining sobre datos auto-filtrados. **Por qué importó:** primer paso hacia VLMs unificados.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de VLMs y generación" years="2022-presente" >}}
    {{< hito year="2022" name="DALL·E 2 / Stable Diffusion / Imagen" status="minimal" >}}
      Ramesh et al. (DALL·E 2), Rombach et al. (Stable Diffusion), Saharia et al. (Imagen): generación texto-imagen de alta calidad con diffusion condicionado en CLIP/T5 text encoders. **Por qué importó:** primer caso en que generación multimodal se vuelve producto masivo.
    {{< /hito >}}
    {{< hito year="2022" name="Flamingo" status="minimal" >}}
      Alayrac et al. (DeepMind): few-shot VLM que intercala visión y texto en una secuencia, con cross-attention adaptado sobre un LLM congelado. **Por qué importó:** fijó el patrón "LLM congelado + adaptador visual".
    {{< /hito >}}
    {{< hito year="2023" name="LLaVA" status="minimal" >}}
      Liu et al.: VLM open-source con un proyector lineal entre encoder visual (CLIP) y LLM (LLaMA). Entrenamiento simple sobre pares instrucción-imagen sintetizados con GPT-4. **Por qué importó:** democratizó VLMs competentes.
    {{< /hito >}}
    {{< hito year="2023-2025" name="GPT-4V / Gemini / Claude Vision" status="covered" link="/fundamentos/foundation-models" >}}
      OpenAI, Google DeepMind, Anthropic: visión integrada nativamente al razonamiento de los frontier LLMs. Análisis de gráficos, documentos, código en pantalla, escenas complejas.
    {{< /hito >}}
    {{< hito year="2024-2025" name="Sora / Veo" status="minimal" >}}
      OpenAI (Sora), Google DeepMind (Veo): generación de video con coherencia temporal extendida (10-60 segundos), física aproximada y control fino por prompt. **Por qué importó:** mueve la frontera de imagen estática a video como producto masivo.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}
```

**Step 3: Verify build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

Expected: build limpio, 282+ pages, solo el warning preexistente del shortcode `tabs`.

**Step 4: Curl-based validation**

```bash
hugo server -D --port 1313 > /tmp/hugo-task1-ola2.log 2>&1 &
sleep 3

URL=http://localhost:1313/neural-network-knowledge/dominios/multimodal/

curl -s -o /tmp/multimodal.html -w "HTTP %{http_code}\n" "$URL"

# H1
grep "<h1[^>]*>Multimodal" /tmp/multimodal.html | head -1

# Sections
grep -c "El problema central" /tmp/multimodal.html
grep -c "Línea de tiempo" /tmp/multimodal.html

# Timeline
grep -c 'class="timeline-container"' /tmp/multimodal.html

# 5 eras
grep -c 'class="timeline-era"' /tmp/multimodal.html

# 17 hitos (3+3+3+3+5)
grep -c 'class="timeline-hito timeline-hito-' /tmp/multimodal.html

# Era headers
grep "Era de captioning temprano" /tmp/multimodal.html | head -1
grep "Era de atención visual estructurada" /tmp/multimodal.html | head -1
grep "Era de pretraining multimodal" /tmp/multimodal.html | head -1
grep "Era contrastiva y zero-shot" /tmp/multimodal.html | head -1
grep "Era de VLMs y generación" /tmp/multimodal.html | head -1

# Some hito names
grep "Show and Tell" /tmp/multimodal.html | head -1
grep "CLIP" /tmp/multimodal.html | head -1
grep "Sora" /tmp/multimodal.html | head -1
grep "DALL·E" /tmp/multimodal.html | head -1

# Status mix
grep -c 'class="timeline-hito timeline-hito-deep"' /tmp/multimodal.html  # 1 (CLIP)
grep -c 'class="timeline-hito timeline-hito-covered"' /tmp/multimodal.html  # 5 (show-and-tell, show-attend-tell, relation-networks, bottom-up, GPT-4V/Gemini)
grep -c 'class="timeline-hito timeline-hito-minimal"' /tmp/multimodal.html  # 11

pkill -f "hugo server" || true
sleep 1
```

Expected:
- HTTP 200.
- 1 `timeline-container`.
- 5 `timeline-era`.
- 17 hitos (1 deep + 5 covered + 11 minimal).
- All 5 era names and key hito names present.

**Step 5: Commit**

```bash
git add site/content/dominios/multimodal/_index.md
git commit -m "feat(dominios/multimodal): problema central + linea de tiempo (5 eras)"
```

NO Co-Authored-By trailer.

---

## Task 2: Eras explicadas (5 subsecciones)

**Files:**
- Modify: `site/content/dominios/multimodal/_index.md` (apend al final, después del `{{< /timeline >}}`)

**Step 1: Apender el siguiente contenido al final del archivo**

```markdown

## Era 1 — Captioning temprano (2014-2016)

### Problema heredado

A inicios de los 2010s la visión por computador y el procesamiento de lenguaje vivían en silos: las CNNs clasificaban imágenes en categorías cerradas, los RNNs traducían texto. Conectar ambas modalidades era un problema abierto. La **descripción automática de imágenes** (image captioning) se volvió el primer terreno de prueba: dada una foto, generar una oración que la describa.

### Idea clave

**Combinar el encoder de visión con el decoder de lenguaje.** Show and Tell (Vinyals et al., 2015) tomó una CNN preentrenada en ImageNet, extrajo el vector de la última capa antes de la clasificación y lo pasó como estado inicial a una LSTM que generaba la descripción palabra por palabra. La arquitectura era una traducción directa de Seq2Seq: en lugar de "encoder de oración fuente", "encoder de imagen".

Show, Attend and Tell (Xu et al., 2015) agregó **atención sobre regiones** de la imagen: en cada paso del decoder el modelo aprende a mirar dónde necesita. Las visualizaciones del paper — el modelo mirando una pelota mientras genera la palabra "ball" — se volvieron icónicas.

### Qué la destronó

La atención uniforme sobre la grilla de features convolucionales no respetaba la estructura del mundo: los objetos no son píxeles aislados sino regiones cohesivas. La idea natural era operar sobre **objetos detectados** en lugar de la grilla densa.

## Era 2 — Atención visual estructurada (2017-2018)

### Problema heredado

Show, Attend and Tell atendía a una grilla 14x14 de features convolucionales — geometría arbitraria que no se alinea con los objetos reales en la escena. Las preguntas de VQA como *"¿de qué color es la pelota a la izquierda del hombre?"* exigen razonar sobre objetos, sus atributos y sus relaciones. Una grilla uniforme no provee esa estructura.

### Idea clave

**Atender a regiones detectadas, no a píxeles uniformes.** Anderson et al. (2018) propusieron usar Faster R-CNN para detectar 36 regiones por imagen, y atender sobre esas regiones desde el decoder de lenguaje. Las regiones traen consigo features semánticamente coherentes (objetos, partes), y la atención sobre ellas es interpretable.

En paralelo, Santoro et al. (2017) introdujeron **Relation Networks**: un módulo que razona explícitamente sobre **pares** de objetos. Para cada par $(o_i, o_j)$ se calcula una función de relación, y las relaciones se agregan para responder la pregunta. Resolvió el dataset CLEVR — un benchmark de razonamiento visual sintético — donde CNNs puras fallaban.

### Qué la destronó

Estas arquitecturas eran ad hoc para captioning/VQA. Cada nueva tarea requería diseñar un módulo nuevo. Faltaba un **paradigma de pretraining transferible** — el equivalente a BERT para multimodal.

## Era 3 — Pretraining multimodal (2019-2020)

### Problema heredado

BERT (2018) había mostrado que pretraining masivo + fine-tuning destronaba a soluciones especializadas en NLP. La pregunta natural: ¿se puede aplicar la misma receta a texto + imagen?

### Idea clave

**BERT-style para texto e imagen.** ViLBERT (Lu et al., 2019) propuso dos torres BERT — una para texto, otra para imagen como secuencia de regiones detectadas — con **cross-attention** entre ellas en cada capa. El pretraining usa Masked Language Modeling extendido: enmascarar tokens de texto, predecirlos con contexto visual; enmascarar regiones, predecirlas con contexto textual.

LXMERT (Tan & Bansal, 2019) fijó el patrón con tres encoders (texto, visión, cross-modal) y cinco tareas de pretraining sobre Conceptual Captions. UNITER y OSCAR (2020) lo simplificaron a un solo Transformer unificado, agregando **etiquetas de objetos** como anclas semánticas que aproximan textos e imágenes en el mismo vocabulario.

### Qué la destronó

Estos modelos dependían de **detectores Faster R-CNN preentrenados** — caros, lentos y un cuello de botella conceptual. Y los datasets curados (Conceptual Captions, COCO) eran pequeños comparados con el texto disponible para pretraining de NLP. CLIP demostró que se podía evitar ambos problemas con datos web ruidosos a escala.

## Era 4 — Contrastiva y zero-shot (2021-2022)

### Problema heredado

Los modelos de pretraining multimodal eran lentos (cada imagen requería detección + atención cruzada profunda) y limitados a las tareas vistas en pretraining. Cambiar el dominio de aplicación requería fine-tuning. La pregunta abierta: ¿hay una arquitectura más simple que aprenda representaciones genéricamente útiles?

### Idea clave

**Aprendizaje contrastivo a escala web.** CLIP (Radford et al., 2021) entrenó un encoder de texto (Transformer) y uno de imagen (ViT o ResNet) sobre 400 millones de pares imagen-caption raspados de internet. La pérdida es contrastiva: dado un batch de N pares, el modelo debe identificar cuál caption corresponde a cuál imagen entre las $N \times N$ combinaciones posibles.

El resultado son dos encoders cuyos embeddings viven en un **espacio compartido**. Para clasificar zero-shot una imagen entre clases arbitrarias, basta describir cada clase como texto ("una foto de un akita"), embeber esos textos, embeber la imagen, y elegir la clase con mayor similitud coseno. CLIP iguala o supera a modelos supervisados especializados en docenas de benchmarks sin haber visto un solo ejemplo de las tareas.

ALIGN (2021, Google) replicó la idea a 1.8B pares ruidosos. BLIP (2022) unificó entendimiento y generación. El paradigma se volvió la base de toda la generación texto-imagen posterior — Stable Diffusion, DALL·E 2 e Imagen condicionan sobre el text encoder de CLIP.

### Qué la destronó

CLIP entendía pero no generaba, y no razonaba. Su embedding compartido es útil pero los detalles finos (composición, conteo, posiciones relativas) escapan a la similitud coseno. La frontera se movió a integrar visión **dentro** de los LLMs y a generación cross-modal de alta fidelidad.

## Era 5 — VLMs y generación (2022-presente)

### Problema heredado

CLIP había abierto dos preguntas: (1) cómo darle visión a un LLM general sin reentrenarlo desde cero, y (2) cómo invertir el proceso — generar imagen y video desde texto, no solo entenderlos.

### Idea clave

Dos líneas convergentes:

1. **Modelos visión-lenguaje (VLMs).** Flamingo (DeepMind, 2022) introdujo el patrón "**LLM congelado + adaptador visual**": un encoder de imagen (ViT preentrenado) produce tokens visuales que se inyectan en el LLM vía cross-attention adaptado, sin tocar los pesos del LLM. LLaVA (2023) lo simplificó a un proyector lineal entre CLIP-ViT y LLaMA, entrenado con instrucciones sintéticas generadas por GPT-4. GPT-4V, Gemini y Claude llevaron la idea a producción con visión nativa desde el pretraining.

2. **Generación texto-a-imagen/video.** Stable Diffusion, DALL·E 2 e Imagen (2022) usan **modelos de difusión** condicionados en el text encoder de CLIP o T5. La difusión aprende a denoise iterativamente desde ruido gaussiano hasta la imagen final, guiada por el embedding de texto en cada paso. Sora (2024) y Veo (2024-2025) extendieron la receta a video con coherencia temporal de 10-60 segundos.

### Qué viene

El campo está convergiendo hacia **modelos any-to-any** que aceptan y producen cualquier combinación de modalidades (texto, imagen, audio, video, acción). Las direcciones activas: agentes visuales que actúan sobre interfaces gráficas, generación de video con física correcta, integración con robótica vía Vision-Language-Action (RT-2, π0, OpenVLA — pendientes de la Ola 4 de Dominios), y modelos pequeños multimodales competentes (eficiencia por destilación). La pregunta abierta: si los frontier LLMs se vuelven multimodales nativos por defecto, ¿queda "multimodal" como disciplina aislada o se diluye en el modelado general de secuencias?
```

**Step 2: Verify build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

**Step 3: Curl-based validation**

```bash
hugo server -D --port 1313 > /tmp/hugo-task2-ola2.log 2>&1 &
sleep 3

URL=http://localhost:1313/neural-network-knowledge/dominios/multimodal/

curl -s -o /tmp/multimodal.html "$URL"

# 5 era H2s
grep -c '<h2[^>]*>Era ' /tmp/multimodal.html  # should be 5

# Subsections (each H3 generates 2 anchor refs in HTML)
grep -c "Problema heredado" /tmp/multimodal.html  # 10 expected
grep -c "Idea clave" /tmp/multimodal.html  # 10
grep -c "Qué la destronó" /tmp/multimodal.html  # 8 (eras 1-4)
grep -c "Qué viene" /tmp/multimodal.html  # 2 (era 5)

# KaTeX inline math from Era 2
grep -E 'o_i|o_j' /tmp/multimodal.html | head -1
grep -E 'N \\times N|N\\\\times N|N \\\\times N' /tmp/multimodal.html | head -1

# Specific phrases
grep "Vinyals et al., 2015" /tmp/multimodal.html | head -1
grep "Anderson et al. (2018)" /tmp/multimodal.html | head -1
grep "Radford et al., 2021" /tmp/multimodal.html | head -1
grep "Flamingo (DeepMind, 2022)" /tmp/multimodal.html | head -1
grep "LLM congelado" /tmp/multimodal.html | head -1

pkill -f "hugo server" || true
sleep 1
```

**Step 4: Commit**

```bash
git add site/content/dominios/multimodal/_index.md
git commit -m "feat(dominios/multimodal): eras explicadas (5 subsecciones narrativas)"
```

NO Co-Authored-By trailer.

---

## Task 3: SOTA + casos de uso + qué viene + recursos

**Files:**
- Modify: `site/content/dominios/multimodal/_index.md` (apend al final)

**Step 1: Verificar fundamentos y papers para los recursos**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
ls site/content/fundamentos/{aprendizaje-contrastivo,foundation-models,mecanismo-atencion,self-attention,vision-transformer}.md 2>&1
ls site/content/papers/{show-and-tell-vinyals-2015,show-attend-tell-xu-2015,bottom-up-attention-anderson-2018,relation-networks-santoro-2017,clip-radford-2021,bahdanau-attention-2015}.md 2>&1
```

Si alguno falta, eliminar el bullet correspondiente y reportar.

**Step 2: Apender al final del archivo**

```markdown

## Estado del arte hoy

{{< callout type="info" >}}

**Frontier multimodal (2024-2025).** La visión y el lenguaje ya no compiten por arquitectura; conviven dentro de los frontier LLMs como capacidad nativa. La generación cross-modal alcanza calidad de producto.

- **GPT-5 (Vision)** — OpenAI. Multimodal nativo desde el pretraining; razona sobre imagen, audio y texto en una sola sesión.
- **Gemini 2.5 Pro Vision** — Google DeepMind. Razonamiento sobre imágenes y video largo, integración con búsqueda y herramientas.
- **Claude Opus 4.7** — Anthropic. Visión integrada al razonamiento extendido; foco en imágenes técnicas y diagramas.
- **LLaMA 4 Multimodal** — Meta. Open weights con visión nativa; competitivo con frontera cerrada en muchos benchmarks.
- **Sora 2 / Veo 2** — OpenAI / Google DeepMind. Generación de video con coherencia temporal extendida (60s+) y control fino por prompt.
- **Stable Diffusion 3 / Imagen 3** — Stability AI / Google. Generación texto-imagen producción.
- **DINOv2** — Meta. Foundation model self-supervised para visión, base de muchas pipelines downstream multimodales.

{{< /callout >}}

## Casos de uso reales

- **Búsqueda visual con texto**: Google Lens, Pinterest Lens, Amazon — search por imagen + descripción combinada (CLIP-style retrieval).
- **Asistentes conversacionales con visión**: ChatGPT, Claude, Gemini — leer documentos, analizar gráficos, depurar capturas de pantalla, describir escenas.
- **Generación creativa**: Midjourney, DALL·E, Adobe Firefly — concept art, marketing, diseño de producto, ilustración editorial.
- **Generación de video**: Sora, Veo — previsualización, marketing, contenido corto en redes.
- **Accesibilidad**: descripción automática de imágenes para personas con discapacidad visual; transcripción enriquecida con contexto visual.
- **Análisis de documentos visión + texto**: facturas, historias clínicas, papers científicos con figuras, contratos con sellos y firmas — donde el OCR puro se queda corto.
- **Robótica vision-language-action**: RT-2, π0, OpenVLA — el robot recibe instrucciones en lenguaje natural y ve su entorno (cubierto en profundidad en la Ola 4 de Dominios, dominio Robótica/RL).

## Qué viene

Las apuestas activas en multimodal hoy: **modelos any-to-any** (entrada y salida arbitraria entre texto, imagen, audio, video, acción), **generación de video con física correcta** (más allá de Sora — coherencia de identidad, conservación de masa, causalidad), **agentes visuales** que actúan sobre interfaces gráficas como un humano (clicks, scrolls, formularios), **modelos pequeños multimodales** competentes vía destilación, e **integración profunda con herramientas** (visión + búsqueda web + ejecución de código). La pregunta abierta de 2025: si los frontier LLMs se vuelven multimodales por defecto, ¿queda "multimodal" como disciplina aislada o se diluye en el modelado general de secuencias?

## Recursos relacionados

**Fundamentos:**
- [Aprendizaje contrastivo (CLIP)](/fundamentos/aprendizaje-contrastivo) — el puente entre visión y lenguaje.
- [Foundation models](/fundamentos/foundation-models) — el contexto general de los VLMs frontier.
- [Mecanismo de atención](/fundamentos/mecanismo-atencion) — la base de show-attend-tell y todas las arquitecturas posteriores.
- [Self-attention](/fundamentos/self-attention) y [Vision Transformer](/fundamentos/vision-transformer) — las piezas que CLIP combinó.

**Papers:**
- [Show and Tell (Vinyals 2015)](/papers/show-and-tell-vinyals-2015).
- [Show, Attend and Tell (Xu 2015)](/papers/show-attend-tell-xu-2015).
- [Relation Networks (Santoro 2017)](/papers/relation-networks-santoro-2017).
- [Bottom-Up Attention (Anderson 2018)](/papers/bottom-up-attention-anderson-2018).
- [CLIP (Radford 2021)](/papers/clip-radford-2021).
- [Bahdanau attention (2015)](/papers/bahdanau-attention-2015) — antecedente de show-attend-tell.

**Dominios relacionados:**
- [Texto / NLP](/dominios/texto) — la mitad lingüística de los VLMs.
- [Visión](/dominios/vision) — la mitad visual.

---

*Última actualización: 2026-05-05.*
```

**Step 3: Verify build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

**Step 4: Curl-based validation**

```bash
hugo server -D --port 1313 > /tmp/hugo-task3-ola2.log 2>&1 &
sleep 3

URL=http://localhost:1313/neural-network-knowledge/dominios/multimodal/

curl -s -o /tmp/multimodal.html "$URL"

# Sections
grep -c "Estado del arte hoy" /tmp/multimodal.html  # 3 (heading + TOC + anchor)
grep -c "Casos de uso reales" /tmp/multimodal.html  # 3
grep -c "Recursos relacionados" /tmp/multimodal.html  # 3

# Callout
grep -c "callout" /tmp/multimodal.html | head -1

# SOTA bullets
grep "GPT-5" /tmp/multimodal.html | head -1
grep "Gemini 2.5" /tmp/multimodal.html | head -1
grep "Sora 2" /tmp/multimodal.html | head -1
grep "DINOv2" /tmp/multimodal.html | head -1

# Resource links
grep -oE 'href="[^"]*fundamentos/aprendizaje-contrastivo"' /tmp/multimodal.html | head -1
grep -oE 'href="[^"]*papers/clip-radford-2021"' /tmp/multimodal.html | head -1
grep -oE 'href="[^"]*dominios/texto"' /tmp/multimodal.html | head -1
grep -oE 'href="[^"]*dominios/vision"' /tmp/multimodal.html | head -1

# Last update
grep "Última actualización: 2026-05-05" /tmp/multimodal.html | head -1

pkill -f "hugo server" || true
sleep 1
```

**Step 5: Commit**

```bash
git add site/content/dominios/multimodal/_index.md
git commit -m "feat(dominios/multimodal): SOTA, casos de uso, que viene y recursos"
```

NO Co-Authored-By trailer.

---

## Task 4: Verificación final, build de producción y push

**Files:** ninguno nuevo.

**Step 1: Build limpio de producción**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
rm -rf site/public site/resources/_gen 2>/dev/null
cd site && hugo --gc --minify
```

Expected: build sin errores ni warnings nuevos. 282+ pages.

**Step 2: FlexSearch indexa la página nueva**

```bash
python3 -c "
import json
d = json.load(open('public/es.search-data.json'))
keys = [k for k in d.keys() if 'multimodal' in k.lower()]
print('Multimodal entries:', len(keys))
for k in keys:
    print(' -', k, '|', d[k].get('title', '?') if isinstance(d[k], dict) else '?')
" 2>&1
# Search for content terms
grep -c "CLIP\|Flamingo\|DALL·E\|Stable Diffusion" public/es.search-data.json
```

Expected: la página `/neural-network-knowledge/dominios/multimodal/` aparece en el índice. Términos clave presentes.

**Step 3: Verificar que los stubs siguen renderizando**

Audio, video, robotica, estructurados deben seguir renderizando correctamente como stubs (la Ola 2 no debe tocarlos):

```bash
ls public/dominios/audio/index.html public/dominios/video/index.html public/dominios/robotica/index.html public/dominios/estructurados/index.html
```

Expected: los 4 archivos existen.

**Step 4: Inspección de contenido (curl)**

```bash
hugo server > /tmp/hugo-final-ola2.log 2>&1 &
sleep 4

URL=http://localhost:1313/neural-network-knowledge/dominios/multimodal/
curl -s -o /tmp/multimodal-final.html "$URL"

# Word count rough estimate (counts words; expect 800-1500)
python3 -c "
import re
html = open('/tmp/multimodal-final.html').read()
# Strip tags crudely
text = re.sub(r'<[^>]+>', ' ', html)
text = re.sub(r'\s+', ' ', text).strip()
words = text.split()
print('Approx word count:', len(words))
"

pkill -f "hugo server" || true
sleep 1
```

Note: this is an approximate count including some HTML tag content noise; expect 1500-3000 because the count includes navigation chrome. Just verify the page is non-empty.

**Step 5: Push y abrir PR**

```bash
git push -u origin feat/dominios-ola-2
```

Expected: push exitoso.

```bash
gh pr create --base main --head feat/dominios-ola-2 --title "feat(dominios): Ola 2 — Multimodal" --body "$(cat <<'EOF'
## Summary

Página completa para el dominio **Multimodal** de la sección Dominios. Patrón idéntico al de Texto/Visión (Ola 1).

- **Línea de tiempo de 5 eras**: captioning temprano (2014-2016) → atención visual estructurada (2017-2018) → pretraining multimodal (2019-2020) → contrastivo y zero-shot (2021-2022) → VLMs y generación (2022-presente).
- **17 hitos** distribuidos 3+3+3+3+5: 1 `deep` (CLIP → aprendizaje-contrastivo), 5 `covered` (show-and-tell, show-attend-tell, relation-networks, bottom-up-attention, GPT-4V/Gemini → foundation-models), 11 `minimal`.
- **Eras explicadas** (5 subsecciones con Problema heredado / Idea clave / Qué la destronó o Qué viene).
- **Estado del arte 2024-2025**, casos de uso, qué viene, recursos.

Reaprovecha 5 fundamentos y 6 papers existentes; sin nuevas dependencias de infraestructura.

Diseño: [`docs/plans/2026-05-05-dominios-ola-2-design.md`](docs/plans/2026-05-05-dominios-ola-2-design.md). Plan: [`docs/plans/2026-05-05-dominios-ola-2-plan.md`](docs/plans/2026-05-05-dominios-ola-2-plan.md).

## Test plan

- [ ] `cd site && hugo --gc --minify` build limpio.
- [ ] Inspección visual desktop + móvil + dark mode en `/dominios/multimodal/`.
- [ ] Búsqueda FlexSearch encuentra "CLIP", "Flamingo", "DALL·E", "Sora" y lleva a la página nueva.
- [ ] Click en hitos `deep` (CLIP) y `covered` (show-and-tell, show-attend-tell, etc.) lleva a Fundamentos / Papers existentes (no 404).
- [ ] Stubs (audio, video, robotica, estructurados) siguen renderizando correctamente con su mensaje "Página en construcción".
EOF
)"
```

Reportar la URL de la PR creada.

**No commit en este task** — solo verificación, push y PR.

---

## Definition of Done — Ola 2

- [ ] `/dominios/multimodal/` página completa: 5 eras + 17 hitos en timeline + 5 era subsections + SOTA + casos + qué viene + recursos.
- [ ] Mínimo 800 palabras de prosa narrativa fuera de la timeline.
- [ ] Todos los `link` en hitos resuelven a archivos existentes.
- [ ] `hugo --gc --minify` build limpio (282+ pages, sin warnings nuevos).
- [ ] FlexSearch indexa la página nueva.
- [ ] Stubs de Olas 3-4 (audio, video, robotica, estructurados) intactos.
- [ ] Branch `feat/dominios-ola-2` pusheada y PR abierta contra `main`.
- [ ] Commits sin Co-Authored-By trailer.

## Riesgos durante implementación

1. **Algún paper referenciado falta** — Verificar con `ls` antes de Task 1; downgradear a `minimal` si falta y reportar.
2. **El fundamento `aprendizaje-contrastivo.md` no cubre lo suficiente para CLIP `deep`** — Verificado en exploración: el archivo trata CLIP en profundidad. OK.
3. **Solapamiento con Era 5 de Visión (CLIP, Sora, Stable Diffusion)** — Intencional. Cada página los enmarca desde su ángulo. No es un bug.
4. **KaTeX `$o_i$`, `$o_j$`, `$N \times N$` no renderizan** — Verificar en curl checks de Task 2; ajustar escapado si hay problema.
5. **El usuario puede mergear ramas paralelas durante la sesión** (sucedió en Ola 1) — Si pasa: verificar estado tras cada subagent invocation con `git branch --show-current` y `git log --oneline -3`. Cherry-pick si necesario.
