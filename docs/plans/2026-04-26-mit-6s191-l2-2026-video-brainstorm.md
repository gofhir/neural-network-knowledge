# MIT 6.S191 (2026) Lecture 2 — Video Lecture: Brainstorm en progreso

**Fecha:** 2026-04-26
**Estado:** Brainstorm pausado — pendiente respuesta a Pregunta 2.

## Contexto

Replicar la estrategia ya usada para el video `mit-6s191-rnn` (2020) en un video MIT más reciente:

- **Video:** https://www.youtube.com/watch?v=d02VkQ9MP44
- **Título YouTube:** "MIT 6.S191: Recurrent Neural Networks, Transformers, and Attention"
- **Curso oficial:** Deep Sequence Modeling — MIT 6.S191 (2026), Lecture 2
- **Sitio del curso:** https://introtodeeplearning.com/
- **PDF oficial:** https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf (83 pp., ~4.2 MB)
- **PDF cacheado localmente:** `/Users/robertoaraneda/.claude/projects/-Users-robertoaraneda-projects-personal-courses-ia-uc/423b60ab-c456-45e4-816b-e86a00e60d0e/tool-results/webfetch-1777245618017-eodwvp.pdf`

## Estrategia de referencia

Reusar el patrón de `docs/plans/2026-04-26-mit-6s191-rnn-video-design.md` y `docs/plans/2026-04-26-mit-6s191-rnn-video.md`:

- Sección top-level `site/content/videos/<slug>/` con `_index.md` + `notas.md` + `profundizacion.md` + `glosario.md`.
- PDF en `site/static/videos/<slug>/slides.pdf`.
- Embed YouTube + atribución MIT en `_index.md`.
- Multi-agent dispatch (3 agentes Explore + 1 general-purpose) para extraer notas/profundización/glosario en paralelo, consolidación secuencial.
- Idioma español, términos técnicos en inglés cuando corresponda.
- Cross-link con clases del curso UC en `profundizacion.md` (no duplicar derivaciones).

## Decisiones tomadas

### Pregunta 1 — Coexistencia con video 2020 → **A**

Crear nueva carpeta separada en paralelo a `videos/mit-6s191-rnn/` (que cubre el L2 de 2020, solo RNN/LSTM/GRU). El video nuevo extiende a Transformers y Attention, complementa naturalmente con `clases/clase-13` (Seq2Seq + Attention) que ya existe. Son recursos complementarios, no sustitutos.

## Decisiones pendientes

### Pregunta 2 — Slug de carpeta (PENDIENTE)

Opciones propuestas:

- **A)** `videos/mit-6s191-l2-2026/` — explícito sobre año y número de lecture, neutral al tema. **Recomendado.** Escala a futuros L3/L2-otro-año con patrón uniforme `mit-6s191-l<N>-<año>`.
- **B)** `videos/mit-6s191-sequence-modeling/` — semántico (título oficial), agnóstico al año.
- **C)** `videos/mit-6s191-rnn-transformers/` — descriptivo de los temas (RNN + Transformers + Attention).

### Preguntas siguientes (aún no formuladas)

- Particionado del PDF entre agentes Explore (3 agentes × ~28 pp. cada uno, basado en estructura temática del PDF de 83 slides).
- Cross-links explícitos con `clases/clase-11` (RNN) **y** `clases/clase-13` (Seq2Seq + Attention) en `profundizacion.md`.
- Si profundización debe cubrir papers seminales adicionales: Vaswani 2017 (Attention is All You Need), Bahdanau 2015 (NMT alignment), Sutskever 2014 (seq2seq).
- Alcance del glosario: ampliar a términos de Transformers (self-attention, multi-head, positional encoding, query/key/value, etc.).
- Resumen en `_index.md`: ¿enfatizar la diferencia con el video 2020?

## Próximos pasos al retomar

1. Confirmar respuesta a Pregunta 2 (slug).
2. Continuar preguntas 3–5 (partición de slides, papers a cubrir, glosario).
3. Presentar diseño final en secciones.
4. Escribir design doc `docs/plans/2026-04-26-mit-6s191-l2-2026-video-design.md`.
5. Invocar `superpowers:writing-plans` para plan de implementación.
