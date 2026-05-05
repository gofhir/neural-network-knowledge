---
title: "Diseño — Sección Dominios, Ola 2 (Multimodal)"
date: 2026-05-05
status: aprobado
autor: Roberto Araneda
---

# Diseño — Ola 2: Multimodal

## Contexto

La Ola 1 (mergeada en `main` el 2026-05-03) entregó la infraestructura de la sección **Dominios** y dos páginas completas: Texto/NLP y Visión. Esta Ola 2 agrega un solo dominio — **Multimodal** — siguiendo el patrón establecido. La infraestructura (shortcodes `timeline`/`era`/`hito`, CSS, menú, landing) ya existe; solo se reemplaza el stub de `dominios/multimodal/_index.md` por una página completa.

Material existente que reaprovechar:

- **Fundamentos:** `aprendizaje-contrastivo.md` (cubre CLIP en profundidad), `foundation-models.md`, `mecanismo-atencion.md`, `self-attention.md`, `vision-transformer.md`.
- **Papers:** `show-and-tell-vinyals-2015.md`, `show-attend-tell-xu-2015.md`, `bottom-up-attention-anderson-2018.md`, `relation-networks-santoro-2017.md`, `clip-radford-2021.md`, `bahdanau-attention-2015.md`.

## Decisiones aprobadas

1. **Una sola página de dominio** (`dominios/multimodal/`), patrón idéntico al de Texto/Visión.
2. **Cinco eras** marcando saltos cualitativos: captioning temprano → atención estructurada → pretraining multimodal → contrastivo/zero-shot → VLMs y generación.
3. **17 hitos** distribuidos 3+3+3+3+5 a través de las cinco eras.
4. **Estados de hito** según material existente: 2 `deep` (CLIP, GPT-4V/Gemini), 4 `covered` (show-and-tell, show-attend-tell, bottom-up-attention, relation-networks), 11 `minimal`.
5. **Implementación en 4 tasks** (no 14 como Ola 1) porque la infraestructura ya existe.
6. **Branch nuevo:** `feat/dominios-ola-2` (creada desde `main` post-merge de Ola 1).
7. **Convenciones heredadas:** español con tildes correctas, sin Co-Authored-By en commits, tono pedagógico-narrativo, 800-1500 palabras totales.

## Estructura de la página

`site/content/dominios/multimodal/_index.md` sigue exactamente el molde de Texto y Visión:

1. Front matter: `title: "Multimodal"`, `weight: 5`, `sidebar.open: true`.
2. `# Multimodal` (H1).
3. `## El problema central` — 1-2 párrafos.
4. `## Línea de tiempo` — `{{< timeline >}}` con 5 eras y 17 hitos.
5. 5 subsecciones `## Era N — <nombre> (<rango>)` con Problema heredado / Idea clave / Qué la destronó (eras 1-4) o Qué viene (era 5).
6. `## Estado del arte hoy` — `{{< callout type="info" >}}` con SOTA 2024-2025.
7. `## Casos de uso reales` — lista de 6-7 ejemplos.
8. `## Qué viene` — 1 párrafo.
9. `## Recursos relacionados` — Fundamentos / Papers / Clases.
10. Pie con fecha de última actualización.

## El problema central — esbozo

Dos párrafos:

1. Las modalidades (texto, imagen, audio) viven en espacios incompatibles — texto es discreto y composicional, imagen es continua y espacial. Hacerlas dialogar requiere proyectarlas a un espacio común o conectarlas con atención cruzada.
2. Tres tensiones definen el campo: alineación sin pares anotados (contrastivo a escala), dejar que un LLM "vea" sin destruir su capacidad lingüística (adaptadores ligeros vs entrenamiento conjunto), generación coherente cross-modal (texto-imagen-video).

## Línea de tiempo — eras y hitos

### Era 1 — Captioning temprano (2014-2016)

| Hito | Año | Status | Link |
|---|---|---|---|
| Show and Tell (Vinyals) | 2015 | `covered` | `/papers/show-and-tell-vinyals-2015` |
| Show, Attend and Tell (Xu) | 2015 | `covered` | `/papers/show-attend-tell-xu-2015` |
| VQA dataset (Antol) | 2015 | `minimal` | — |

**Idea clave:** combinar CNN (encoder visual) + RNN (decoder lingüístico) para describir imágenes. Show, Attend and Tell agrega atención sobre regiones de la imagen.

**Qué la destronó:** atender uniformemente sobre la imagen no captura objetos; el siguiente paso fue atender a regiones detectadas (bottom-up attention).

### Era 2 — Atención visual estructurada (2017-2018)

| Hito | Año | Status | Link |
|---|---|---|---|
| Relation Networks (Santoro) | 2017 | `covered` | `/papers/relation-networks-santoro-2017` |
| Bottom-Up and Top-Down Attention (Anderson) | 2018 | `covered` | `/papers/bottom-up-attention-anderson-2018` |
| GQA / VCR datasets (razonamiento visual) | 2018-2019 | `minimal` | — |

**Idea clave:** la atención debe operar sobre objetos detectados (Faster R-CNN) en lugar de píxeles uniformes. Relation Networks agrega razonamiento sobre pares de objetos.

**Qué la destronó:** las arquitecturas eran ad hoc para captioning/VQA; faltaba un paradigma de pretraining transferible.

### Era 3 — Pretraining multimodal (2019-2020)

| Hito | Año | Status | Link |
|---|---|---|---|
| ViLBERT (Lu) | 2019 | `minimal` | — |
| LXMERT (Tan & Bansal) | 2019 | `minimal` | — |
| UNITER / OSCAR | 2020 | `minimal` | — |

**Idea clave:** aplicar el receta de BERT (Masked Language Modeling) al texto+imagen. Dos torres con cross-attention, pretrain sobre Conceptual Captions / COCO, fine-tune para tareas específicas.

**Qué la destronó:** dependían de detectores Faster R-CNN preentrenados (caros, frágiles) y de pares imagen-caption curados. CLIP mostró que el contraste a escala web era más simple y más potente.

### Era 4 — Contrastivo y zero-shot (2021)

| Hito | Año | Status | Link |
|---|---|---|---|
| CLIP (Radford / OpenAI) | 2021 | `deep` | `/fundamentos/aprendizaje-contrastivo` |
| ALIGN (Jia / Google) | 2021 | `minimal` | — |
| BLIP (Li / Salesforce) | 2022 | `minimal` | — |

**Idea clave:** entrenar un encoder de texto y uno de imagen contrastivamente sobre 400M pares imagen-caption raspados de la web. El embedding compartido permite clasificación zero-shot ("una foto de un akita").

**Qué la destronó:** CLIP entendía pero no generaba, y no razonaba. La frontera se movió a integrar visión dentro de LLMs y a generación texto-imagen masiva.

### Era 5 — VLMs y generación (2022-presente)

| Hito | Año | Status | Link |
|---|---|---|---|
| DALL·E 2 / Stable Diffusion / Imagen | 2022 | `minimal` | — |
| Flamingo (DeepMind) | 2022 | `minimal` | — |
| LLaVA (Liu) | 2023 | `minimal` | — |
| GPT-4V / Gemini / Claude Vision | 2023-2025 | `covered` | `/fundamentos/foundation-models` |
| Sora / Veo (video generativo) | 2024-2025 | `minimal` | — |

**Idea clave:** dos líneas convergentes — modelos visión-lenguaje (un LLM al que se le inyectan tokens visuales vía un proyector entrenado) y modelos generativos texto-a-imagen/video (Diffusion condicionado en CLIP text encoder o T5).

**Qué viene:** modelos any-to-any (entrada y salida arbitraria de modalidades), agentes visuales que actúan sobre interfaces, generación de video con física correcta, integración con robótica (Vision-Language-Action).

## Estado del arte hoy — esbozo

`{{< callout type="info" >}}` con 5-6 modelos punteros 2024-2025:

- **GPT-5 (Vision)** — OpenAI. Multimodal nativo desde el pretraining.
- **Gemini 2.5 Pro Vision** — Google DeepMind. Razonamiento sobre imágenes y video.
- **Claude Opus 4.7** — Anthropic. Visión integrada al razonamiento extendido.
- **LLaMA 4 Multimodal** — Meta. Open weights con visión nativa.
- **Sora 2 / Veo 2** — generación de video con coherencia temporal.
- **Stable Diffusion 3 / Imagen 3** — generación texto-imagen producción.

## Casos de uso reales

- Búsqueda visual con texto (Google Lens, Pinterest, Amazon — search por imagen + descripción combinada).
- Asistentes con visión (subir capturas, leer documentos, analizar gráficos — ChatGPT, Claude, Gemini).
- Generación creativa (Midjourney, DALL·E, Adobe Firefly — concept art, marketing, producto).
- Generación de video (Sora, Veo — previsualización, marketing, contenido).
- Accesibilidad (descripción automática de imágenes para personas con discapacidad visual).
- Análisis de documentos visión+texto (facturas, historias clínicas, papers científicos con figuras).
- Robótica vision-language-action (RT-2, π0, OpenVLA — pendiente de la Ola 4).

## Plan de implementación (4 tasks)

| Task | Entregable | Archivos |
|---|---|---|
| 1 | Front matter + problema central + timeline | `site/content/dominios/multimodal/_index.md` (reescrito desde stub) |
| 2 | 5 era subsections | mismo archivo (apend) |
| 3 | SOTA + casos + qué viene + recursos | mismo archivo (apend) |
| 4 | Build limpio + push + PR | — |

Sin tocar shortcodes, CSS, menú ni stats home.

## Pruebas y verificación

- `cd site && hugo --gc --minify` build limpio.
- Curl-based assertions sobre `/dominios/multimodal/`: timeline-container presente, 5 eras, 17 hitos, todas las eras y SOTA renderizadas.
- FlexSearch indexa la página nueva (verificable buscando "CLIP", "Flamingo", "DALL·E").
- Inspección visual queda al usuario antes del merge.

## Riesgos identificados

| Riesgo | Mitigación |
|---|---|
| Algún paper referenciado no existe en `papers/` | Verificar con `ls` en Step 1 de Task 1; downgrade a `minimal` si falta |
| El fundamento `aprendizaje-contrastivo.md` no cubre lo suficiente para justificar `deep` en CLIP | Verificado en exploración: el archivo cubre CLIP en profundidad. OK. |
| Solapamiento con la Era 5 de Visión (CLIP, Sora, Stable Diffusion) | Intencional — son hitos *multimodales* que también aparecen en Visión. Cada página los enmarca desde su ángulo (Visión: como sucesor de CNN/ViT; Multimodal: como puente entre modalidades) |
| Foundation models linkea ambiguamente | Aceptable — `foundation-models.md` cubre conceptualmente VLMs |

## Próximos pasos

1. Commit de este documento.
2. Generar plan de implementación detallado en `docs/plans/2026-05-05-dominios-ola-2-plan.md` vía la skill `superpowers:writing-plans`.
3. Ejecutar las 4 tasks vía Subagent-Driven Development en esta sesión.
