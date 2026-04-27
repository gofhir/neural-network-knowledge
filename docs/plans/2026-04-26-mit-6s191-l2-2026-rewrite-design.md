---
title: MIT 6.S191 (2026) Lecture 2 — Reescritura desde PDF verificado: Diseño
date: 2026-04-26
status: Diseño aprobado, listo para plan de implementación
supersedes: docs/plans/2026-04-26-mit-6s191-l2-2026-video-design.md
brainstorm-origen: conversación 2026-04-26 (post-extracción del PDF)
---

# MIT 6.S191 (2026) Lecture 2 — Reescritura desde PDF verificado

## Contexto y razón de existir

Este documento **reemplaza** el diseño anterior (`2026-04-26-mit-6s191-l2-2026-video-design.md`) y los archivos generados por su plan de implementación.

**Por qué hubo que reescribir todo:**

El diseño original asumió un particionado del PDF en bloques temáticos (slides 1-28 RNN intro/BPTT/vanishing, 29-55 LSTM/GRU/attention motivation/Q-K-V, 56-83 multi-head/Transformer/positional/encoder-decoder/aplicaciones). Estos rangos fueron derivados de **expectativas basadas en el lecture 2020 del mismo curso**, no de leer el PDF 2026.

Cuando los subagentes intentaron leer el PDF, fallaron silenciosamente con el error `"image dimension limit (2000px)"` — el PDF de PyFPDF tiene páginas de 3628×2040 pts y al rasterizarlas excede el límite de la API de visión. Sin acceso visual real a las slides, los subagentes generaron contenido derivado de su conocimiento general del campo (RNN→LSTM→Attention→Transformer es el arco canónico de un curso de DL), produciendo notas/profundización/glosario que **describen un lecture que no es el del 2026**.

Hallazgos críticos del PDF real (verificado leyendo las 83 JPEGs embebidas a su resolución nativa de 1280×720):

- **LSTMs/GRUs**: 1 sola slide (slide 54) sin internals, solo conceptual ("gates exist as a solution"). El lecture 2020 dedicaba ~10 slides a esto.
- **Self-attention**: empieza en slide 65 (no en 29 como asumía el diseño anterior). El build técnico de los 4 pasos del attention head ocupa slides 70-78.
- **Slide 30 SÍ tiene los 4 criterios de diseño explícitos** (variable-length, long-term, order, share parameters). Las notas anteriores afirmaban que esa slide no existía.
- **Slide 40 es duplicado exacto de slide 30** (recap intencional al final de la sección de motivación). No documentado.
- **Slides 32-34, 38-39, 56-58 cargan footers de autores originales** (`H. Suresh, 6.S191 2018.`, `Mozer Complex Systems 1989.`, `Huawei.`, `Socher+, EMNLP 2013.`, `Vaswani+, NeurIPS 2017.`). Indica reciclaje/atribución de slides al lecture 2018 de Harini Suresh, papers fundacionales, etc. — perdido en versiones anteriores.
- **Slide 36 es la única slide sobre embeddings** (vocabulary → indexing → one-hot vs learned, con scatter 2D run/walk/dog/cat / day/sun/happy/sad). Tergiversado en notas previas.
- **Slides 60-64 son la transición pedagógica clave** RNN→Self-Attention (limitations → desired capabilities → "feed everything fails" → "identify and attend"). No reconocida como bloque coherente.
- **Slide 80 cita 5 papers seminales modernos** (BERT/GPT/AlphaFold/ESM/ViT) que no estaban en notas anteriores.

**Recursos disponibles para la reescritura:**

- `docs/handoffs/mit-l2-2026-slides-verified.md` (~30 KB) — log slide-por-slide del contenido **real** del PDF, generado en sesión 2026-04-26 leyendo las 83 JPEGs directamente. **Fuente única de verdad** para esta reescritura.
- `docs/handoffs/mit-l2-2026-pdf-extract/slides/slide-NNN-MMM.jpg` — 83 JPEGs nativas para verificación puntual durante el escritura.
- `site/static/videos/mit-6s191-l2-2026/slides.pdf` — PDF original sin tocar.
- `site/content/videos/mit-6s191-rnn/{notas,profundizacion,glosario}.md` — referencia del estilo y nivel del video 2020 (mismo curso, lecture 2 versión 2020).

## Decisiones del brainstorming

| # | Pregunta | Decisión | Razón |
|---|---|---|---|
| 1 | ¿Qué hacer con los 4 archivos publicados? | **A**: Tirar y reescribir desde cero | Las alucinaciones son sutiles y contaminarían cualquier intento de auditoría parcial. Reescritura limpia es más segura. |
| 2 | Estructura final | **A**: 4 archivos (`_index` + `notas` consolidado + `profundizacion` + `glosario`) | Réplica del patrón del video 2020 ya publicado. Coherencia visual del sitio. Búsqueda Cmd+F unificada. |
| 3 | Diagramas Mermaid | **C**: Solo en `profundizacion.md` | `notas.md` mantiene patrón "texto + fórmulas LaTeX" del video 2020. Profundización es donde un diagrama vale para attention head, BPTT flow. |
| 4 | Alcance de papers en profundización | **B**: Citados literalmente + foundationals canónicos | El lecture omite Hochreiter 1997, Bengio 1994, Pascanu 2013, Sutskever 2014, Bahdanau 2015 — pero el alumno los necesita para entender de dónde vino el campo y por qué los gates resuelven vanishing gradients. |
| 5 | Estrategia DRY vs auto-suficiencia | **C**: Híbrido alineado con el video | Profundización cubre todo lo que el lecture toca, expandido lo necesario para entenderlo. Lo que el lecture NO deriva pero sí motiva (BPTT mecánica, scaled dot-product paso a paso, multi-head) se deriva al detalle aquí. LSTM/GRU se explica conceptualmente con cross-link a `/fundamentos/lstm-gru` para internals. |

**Convenciones de estilo (heredadas del video 2020):**

- Español **sin acentos** (`Motivacion`, `tematico`, `informacion`, `tamano`) — convención del sitio.
- Términos técnicos **en inglés** cuando son nombres propios o términos universalmente usados (e.g., `attention head`, `softmax`, `vanishing gradient`).
- Citas a slides en **cursiva** con rangos: `*(slides X-Y)*`.
- Fórmulas LaTeX con `$$ ... $$` (display) y `$ ... $` (inline). Hugo math=true en frontmatter.

## Sección 1 — Arquitectura de archivos

### Output final (commiteable a `main`)

```
site/content/videos/mit-6s191-l2-2026/
├── _index.md           # ~3-4 KB, ~40 lineas. Landing: embed YouTube, encuadre comparativo vs 2020, cross-links Hugo cards
├── notas.md            # ~40-50 KB, ~600-800 lineas. Recorrido tematico unico, 12 secciones, citas (slides X-Y) en cursiva
├── profundizacion.md   # ~25-35 KB, ~400-500 lineas. Derivaciones formales + 5 Mermaid + 14 papers
└── glosario.md         # ~12-15 KB, ~80-100 entradas. Autosuficiente, alfabetico
```

`site/static/videos/mit-6s191-l2-2026/slides.pdf` ya existe — no tocar.

### Borradores intermedios (en `docs/handoffs/`)

```
docs/handoffs/
├── mit-l2-2026-slides-verified.md     # YA EXISTE - fuente unica de verdad
├── mit-l2-2026-notas-NEW.md           # output del subagente Notas
├── mit-l2-2026-profundizacion-NEW.md  # output del subagente Profundizacion
└── mit-l2-2026-glosario-NEW.md        # output del subagente Glosario
```

Los archivos viejos (`mit-l2-2026-notas-parte-1/2/3.md`, `mit-l2-2026-profundizacion.md`, `mit-l2-2026-glosario.md`, `2026-04-26-mit-6s191-rnn-handoff.md`) **no se eliminan** — quedan como evidencia de las alucinaciones para futuro debugging.

## Sección 2 — Contenido de cada archivo

### `_index.md`

Frontmatter Hugo (`title`, `weight: 5`, `sidebar: open: true`) + 6 secciones cortas:

1. **Embed YouTube** del video — `https://www.youtube.com/watch?v=d02VkQ9MP44`
2. **Atribución MIT** — texto canónico: "Material original MIT 6.S191 (introtodeeplearning.com), Ava Amini, 5 de enero de 2026, distribuido bajo licencia del curso. Las notas en español son una elaboración independiente."
3. **Encuadre comparativo vs video 2020** — 1-2 párrafos: "El lecture 2026 mantiene el armazón del 2020 (RNN → BPTT → vanishing gradients → aplicaciones) pero pivotea en slide 60 hacia self-attention y Transformers, terminando con BERT/GPT/AlphaFold/ViT. La sección de LSTM/GRU del 2020 se colapsa en 1 sola slide (slide 54) — el énfasis se desplaza de gating a attention."
4. **Tabla con la macro-estructura del lecture** — los 12 bloques temáticos con rangos de slides
5. **Cross-links** — Hugo `{{< cards >}}` a clase-11, clase-13, fundamentos-rnn, fundamentos-lstm-gru, fundamentos-mecanismo-atencion, video 2020
6. **Acceso al PDF** — link a `/videos/mit-6s191-l2-2026/slides.pdf`

### `notas.md`

Frontmatter (`title: "Notas - MIT 6.S191 (2026) Deep Sequence Modeling"`, `weight: 10`, `math: true`) + intro corta + recorrido temático en **12 secciones**:

| # | Titulo seccion | Rango slides |
|---|---|---|
| 1 | Motivacion: por que modelar secuencias | 1-8 |
| 2 | Construccion del RNN: del perceptron a la recurrencia | 9-23 |
| 3 | Computational graph y codigo (TF/PyTorch) | 24-29 |
| 4 | Criterios de diseno + predict-next-word + embeddings | 30-40 |
| 5 | Backpropagation Through Time mecanico | 41-44 |
| 6 | Vanishing/exploding gradients y long-term dependencies | 45-53 |
| 7 | Gating como solucion (LSTM/GRU teaser) | 54 |
| 8 | Aplicaciones de RNN: musica y sentiment | 55-58 |
| 9 | Limitaciones de RNN y pivot a attention | 59-64 |
| 10 | Self-attention conceptual: la analogia YouTube/search | 65-69 |
| 11 | Self-attention tecnico: los 4 pasos hasta el head completo | 70-78 |
| 12 | Multi-head, aplicaciones modernas y cierre | 79-83 |

Cada sección: texto descriptivo + fórmulas LaTeX + observaciones de detalle (footers de Suresh, citas a Vaswani, etc.). **Sin diagramas** (van en profundización).

### `profundizacion.md`

Frontmatter + intro + 8 secciones:

| # | Titulo seccion | Mermaid | Papers principales |
|---|---|---|---|
| 1 | BPTT al detalle | RNN unrolled backward flow | Mozer 1989 |
| 2 | Vanishing/exploding gradients: analisis spectral | Comparativa flow exploding vs vanishing | Bengio 1994, Pascanu 2013 |
| 3 | LSTM y GRU (extension conceptual) | LSTM cell con 4 gates | Hochreiter & Schmidhuber 1997, Cho et al. 2014 |
| 4 | De seq2seq a attention (genealogia) | — | Sutskever 2014, Bahdanau 2015, Luong 2015 |
| 5 | Scaled dot-product attention derivacion completa | Attention head completo | Vaswani 2017 |
| 6 | Multi-head attention mecanica | Multi-head con concat | Vaswani 2017 |
| 7 | Position encoding | — | Vaswani 2017, mencion de RoPE |
| 8 | Aplicaciones modernas | — | Devlin 2019, Brown 2020, Jumper 2021, Lin 2023, Dosovitskiy 2020 |

Cross-links a `/fundamentos/lstm-gru`, `/fundamentos/mecanismo-atencion`, `/papers/lstm-hochreiter-1997`, etc. donde corresponda — para no duplicar contenido que ya vive en otra parte del sitio.

5 diagramas Mermaid en total, todos build-time SVG (sin JS cliente).

### `glosario.md`

Frontmatter + intro + entradas alfabéticas. Autosuficiente: cubre todo lo que aparece en el video sin requerir abrir otra página. Para términos con derivación larga, entrada corta + cross-link a `profundizacion.md#seccion`.

Categorías cubiertas (~80-100 entradas):

- **Conceptos básicos:** secuencia, recurrencia, hidden state, time step, recurrence relation
- **RNN:** vanilla RNN, BPTT, forward/backward pass, $W_{xh}/W_{hh}/W_{hy}$, unrolling, computational graph
- **Problemas:** vanishing gradient, exploding gradient, gradient clipping, long-term dependencies
- **Gating:** gate, sigmoid, pointwise multiplication, LSTM, GRU, gated cell
- **Encoding:** token, vocabulary, indexing, one-hot, embedding, learned embedding, position encoding
- **Attention:** query, key, value, attention score, attention mask, attention weight, softmax, scaled dot-product, self-attention, multi-head attention, attention head
- **Arquitecturas:** Transformer, BERT, GPT, ViT, AlphaFold/ESM, encoder, decoder
- **Tareas:** sentiment classification, music generation, machine translation, predict next word, image captioning
- **Métricas/papers:** cross-entropy, softmax cross-entropy

## Sección 3 — Proceso de ejecución

### Fase 1 — Generación de borradores en paralelo

Despacho de **3 subagentes general-purpose en paralelo** (un solo mensaje con 3 tool calls):

| Subagente | Output | Input clave | Tokens out estimados |
|---|---|---|---|
| Notas | `docs/handoffs/mit-l2-2026-notas-NEW.md` | log verified completo + plantilla 12 secciones + estilo del video 2020 | ~12K |
| Profundización | `docs/handoffs/mit-l2-2026-profundizacion-NEW.md` | log verified + 8 secciones + 14 papers + instrucciones de Mermaid | ~10K |
| Glosario | `docs/handoffs/mit-l2-2026-glosario-NEW.md` | log verified + lista de ~90 términos | ~5K |

Cada subagente recibe:

- Instrucción explícita: **NO INVENTAR**. Si un concepto no está en el log y no es derivable trivialmente del log, debe declararlo.
- El log verified completo como contexto.
- Convenciones de estilo (sin acentos, términos en inglés, citas slides en cursiva).
- Ejemplo de salida tomado del video 2020 (notas/profundización/glosario respectivamente).

`_index.md` lo escribe el main thread (es corto, requiere coherencia con la decisión final de los otros 3).

### Fase 2 — Revisión y consolidación en main thread

1. Leer los 3 outputs de subagentes.
2. Validar cada uno contra el log verified — buscar afirmaciones que no estén soportadas por el log.
3. Aplicar correcciones si hay discrepancias.
4. Escribir `_index.md` con encuadre comparativo + cards Hugo.
5. Mover los 4 archivos finales a `site/content/videos/mit-6s191-l2-2026/`, **sobreescribiendo** los archivos publicados existentes.

### Fase 3 — Validación pre-commit

1. **Build de Hugo** — `cd site && hugo --gc --minify` para confirmar compilación sin errores.
2. **Mermaid render** — verificar que los 5 diagramas en profundización rendericen vía `mermaid-build-time` skill.
3. **Cross-links** — Hugo reporta broken links automáticamente; verificar 0 broken links nuevos.
4. **Hugo server** — `hugo server` y abrir `http://localhost:1313/videos/mit-6s191-l2-2026/` para revisión visual de los 4 archivos renderizados.
5. **Atribución MIT** — verificar texto canónico literal en `_index.md`.

### Fase 4 — Commit strategy

**Commit único** sobre la rama actual `feat/mit-6s191-l2-2026-video`:

```
fix: rewrite MIT 6.S191 (2026) lecture 2 content from verified PDF source

Previous content was generated from hallucinated PDF analysis (slides 29-55
were claimed to cover LSTMs/GRUs and attention, but actual coverage is
LSTMs in 1 slide and attention starts at slide 65). Rewrites notas.md,
profundizacion.md, glosario.md and _index.md from a verified slide-by-slide
log (docs/handoffs/mit-l2-2026-slides-verified.md) extracted directly from
the 83 embedded JPEGs of the official PDF.
```

- **Sin** `Co-Authored-By` (preferencia documentada del usuario).
- **Incluye:** `site/content/videos/mit-6s191-l2-2026/{_index,notas,profundizacion,glosario}.md`.
- **No incluye:** los borradores `*-NEW.md` (se eliminan después de mover); el log `slides-verified.md` se commitea por separado si se decide conservarlo en `main` (opcional).

## Out of scope explícito

- **No tocar** `_index.md` de clase-11 ni clase-13 — ya apuntan a este video correctamente.
- **No tocar** `/fundamentos/lstm-gru`, `/fundamentos/mecanismo-atencion`, `/fundamentos/redes-recurrentes`, `/fundamentos/backpropagation-through-time`, `/fundamentos/seq2seq` — ya existen y no son responsabilidad de este flujo.
- **No re-extraer JPEGs** — ya están en `docs/handoffs/mit-l2-2026-pdf-extract/slides/` (83 imágenes a 1280×720).
- **No commitear** `docs/handoffs/mit-l2-2026-pdf-extract/` — es working state, no canon del proyecto.
- **No agregar** lab paralelo de Transformers — iniciativa separada que ya estaba out-of-scope.
- **No actualizar** los `_index.md` de cross-link en clase-11/clase-13 — la infra bidireccional ya estaba correctamente apuntando antes de la primera generación, no necesita cambios.
- **No actualizar** `site/content/laboratorios/_index.md` ni `site/content/fundamentos/representacion-datos.md` (cambios de otro flujo).

## Estimaciones

- Fase 1 (subagentes paralelo): ~5-8 min reloj, ~30-40K tokens out total
- Fase 2 (revisión + `_index.md`): ~3-5 min reloj, ~5-8K tokens
- Fase 3 (validación Hugo): ~2-3 min reloj, ~2K tokens
- Fase 4 (commit): ~1 min, ~1K tokens

Total: ~12-17 min de reloj, ~40-50K tokens.

## Próximos pasos

1. Invocar `superpowers:writing-plans` para producir el plan de implementación detallado en `docs/plans/2026-04-26-mit-6s191-l2-2026-rewrite.md`.
2. Ejecutar el plan (despacho paralelo + consolidación + validación + commit).
