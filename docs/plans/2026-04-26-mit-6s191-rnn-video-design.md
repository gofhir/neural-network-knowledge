# MIT 6.S191 (2020) — Video Lecture: RNNs — Design

**Fecha:** 2026-04-26
**Estado:** Aprobado, listo para implementación

## Objetivo

Agregar al sitio Hugo (`site/`) el contenido completo de la clase **MIT 6.S191 (2020) Lecture 2: Deep Sequence Modeling / Recurrent Neural Networks** (Ava Soleimany), como primer recurso de una nueva sección de video lectures que crecerá con más videos en el futuro.

## Fuentes

- **Video:** https://www.youtube.com/watch?v=SEnXr6v2ifU
- **Slides oficiales (93 pp.):** http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf
- **Sitio del curso:** http://introtodeeplearning.com/2020/
- **GitHub MIT:** https://github.com/aamini/introtodeeplearning

## Decisiones de diseño

1. **Ubicación:** nueva sección top-level `content/videos/` (escalable a más videos).
2. **Estructura por video:** carpeta con sub-páginas, paridad con clases del curso (`_index` + `notas` + `profundizacion` + `glosario`).
3. **Idioma:** español, términos técnicos en inglés cuando corresponda.
4. **Fuente de contenido:** PDF oficial del MIT como ground truth + investigación complementaria (no transcripción literal del video).
5. **PDF:** copia local en `static/videos/mit-6s191-rnn/slides.pdf` + atribución/link al original del MIT.
6. **YouTube:** link directo + embed en `_index.md`, referenciado en `notas.md`.

## Arquitectura

```
site/content/videos/
├── _index.md                       # landing de la seccion Videos
└── mit-6s191-rnn/
    ├── _index.md                   # overview, embed YouTube, link slides, tarjetas a sub-paginas
    ├── notas.md                    # recorrido tematico de las 93 diapositivas
    ├── profundizacion.md           # papers citados, derivaciones, conexion con clase 11 del curso UC
    └── glosario.md                 # terminos clave en es/en

site/static/videos/mit-6s191-rnn/
└── slides.pdf                      # PDF oficial copiado del MIT
```

## Plan multi-agente

Despachar 4 agentes en paralelo + consolidacion secuencial:

| Agente | Tipo | Input | Output |
|---|---|---|---|
| Agent-A | Explore | PDF pp. 1-30 (motivacion, RNN, BPTT) | seccion 1 de `notas.md` |
| Agent-B | Explore | PDF pp. 31-60 (vanishing gradient, LSTM, GRU) | seccion 2 de `notas.md` |
| Agent-C | Explore | PDF pp. 61-93 (aplicaciones, attention, demos) | seccion 3 de `notas.md` |
| Agent-D | general-purpose | papers citados + research + comparacion con clase 11 | `profundizacion.md` + `glosario.md` |

Cada agente recibe el mismo prompt-skeleton derivado del estilo de `site/content/clases/clase-11/teoria.md` (frontmatter Hugo, encabezados, math LaTeX, mermaid).

**Consolidacion (manual):** ensamblar `notas.md`, escribir `_index.md` con embed YouTube, copiar PDF, verificar con `hugo build`, commit.

## Riesgos y mitigaciones

- **Inconsistencia de estilo** entre agentes → prompt-skeleton compartido + revision manual al consolidar.
- **Solapamiento con clase 11** → Agent-D enlaza explicitamente a `/clases/clase-11/` en vez de duplicar derivaciones.
- **PDF muy denso (93 slides)** → agentes sintetizan en secciones tematicas, no transcripcion 1:1.
- **YouTube embed** → usar shortcode Hugo `{{< youtube SEnXr6v2ifU >}}` (built-in).

## Atribucion

Cada pagina lleva al pie:

> Material adaptado de **MIT 6.S191 (2020) Lecture 2: Deep Sequence Modeling**, Alexander Amini & Ava Soleimany. [Video](https://www.youtube.com/watch?v=SEnXr6v2ifU) · [Slides oficiales](http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf) · [Sitio del curso](http://introtodeeplearning.com/2020/). Notas en espanol con investigacion complementaria.

## Criterios de exito

- `hugo build` pasa sin errores.
- Pagina `/videos/mit-6s191-rnn/` accesible con embed YouTube funcional.
- PDF descargable desde `/videos/mit-6s191-rnn/slides.pdf`.
- `notas.md`, `profundizacion.md`, `glosario.md` con contenido sustantivo (>2000 palabras combinadas).
- Atribucion al MIT visible en cada pagina.
- Sin duplicar derivaciones que ya viven en clase 11; en su lugar, links cruzados.
