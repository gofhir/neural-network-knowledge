# Handoff — MIT 6.S191 (2020) RNN Video Lecture

**Fecha:** 2026-04-26
**Estado:** Extraccion completada, falta consolidacion en sitio Hugo y commit final

## Contexto

Tarea: agregar al sitio Hugo (`site/`) el contenido completo de la clase MIT 6.S191 (2020) Lecture 2 "Deep Sequence Modeling / Recurrent Neural Networks" (Ava Soleimany), como primer recurso de una nueva seccion top-level `videos/`.

- Video YouTube: https://www.youtube.com/watch?v=SEnXr6v2ifU
- Slides oficiales: http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf

## Documentos de referencia

- **Design doc**: `docs/plans/2026-04-26-mit-6s191-rnn-video-design.md`
- **Plan de implementacion**: `docs/plans/2026-04-26-mit-6s191-rnn-video.md`

Ambos comiteados a `main` (commits `92f5917` y `631a626`).

## Lo que YA esta hecho

### 1. Estructura de directorios

```
site/content/videos/mit-6s191-rnn/      # creado, vacio
site/static/videos/mit-6s191-rnn/
  slides.pdf                             # PDF oficial copiado (9 MB, 93 pp.)
```

### 2. Contenido extraido por agentes (5 archivos)

Persistidos en `docs/handoffs/`:

| Archivo | Palabras | Cubre |
|---|---|---|
| `mit-rnn-notas-parte-1.md` | 1861 | Slides 1-30: motivacion, RNN architecture, configuraciones, BPTT setup |
| `mit-rnn-notas-parte-2.md` | 1378 | Slides 31-60: vanishing/exploding, gradient clipping, LSTM equations |
| `mit-rnn-notas-parte-3.md` | 1906 | Slides 61-93: GRU, aplicaciones (musica, sentiment, traduccion), encoder bottleneck, attention, image captioning, sintesis |
| `mit-rnn-profundizacion.md` | 1938 | Contexto MIT 6.S191, papers citados (LSTM/GRU/Pascanu/Sutskever/Bahdanau/Vinyals), comparacion con Clase 11 UC, conceptos no cubiertos (Transformers/Mamba), recursos |
| `mit-rnn-glosario.md` | 1246 | 25-35 terminos clave en formato `**Termino (English)** - definicion en espanol` |

Total: ~8329 palabras de contenido listo para consolidar.

**Nota sobre overlaps**: parts 2 y 3 se solapan en LSTM (parte 2 tiene la derivacion completa, parte 3 tiene aplicaciones). Hay que consolidar inteligentemente, no concatenar.

## Lo que FALTA hacer

### Tarea A: Consolidar `notas.md`

Crear `site/content/videos/mit-6s191-rnn/notas.md` ensamblando los 3 partes con flujo:

1. Motivacion (P1 §1)
2. Limitaciones de enfoques ingenuos (P1 §2)
3. Criterios de diseno (P1 §3)
4. Arquitectura RNN (P1 §4)
5. Configuraciones one/many (P1 §5)
6. Generacion texto caracter (P1 §6)
7. Grafo desplegado (P1 §7 + P2 §1, dedup)
8. BPTT (P1 §8 + P2 §2, dedup)
9. Vanishing/exploding gradients (P2 §3 + P3 §1.1)
10. Soluciones parciales (P3 §1.2)
11. Gradient clipping (P2 §4)
12. Por que fallan RNNs en secuencias largas (P2 §5)
13. Motivacion gating (P2 §6)
14. LSTM arquitectura completa (P2 §7-9 + P3 §2, eligiendo derivacion mas clara)
15. GRU (P2 §10)
16. Aplicacion: generacion de musica (P3 §3.1)
17. Aplicacion: sentiment classification (P3 §3.2)
18. Aplicacion: traduccion automatica seq2seq (P3 §3.3)
19. Cuello de botella encoder (P3 §4)
20. Mecanismo de atencion (P3 §5)
21. Arquitectura completa con BiLSTM + attention (P3 §6)
22. Aplicaciones de attention: image captioning, summarization, bottom-up (P3 §7)
23. Sintesis y evolucion (P3 §8)
24. Conexion con Transformers (P3 §9)

**Frontmatter Hugo a anteponer:**

```yaml
---
title: "Notas - MIT 6.S191 (2020) RNNs"
weight: 10
math: true
---
```

**Header introductorio** (despues del frontmatter):

```markdown
> Recorrido tematico de las 93 diapositivas del lecture, organizado por contenido (no por slide). Citas a slides especificas en *cursiva*.

**Video original:** [YouTube](https://www.youtube.com/watch?v=SEnXr6v2ifU)
**Slides:** [PDF local](/videos/mit-6s191-rnn/slides.pdf) - [Original MIT](http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf)

---
```

**Footer atribucion** (al final):

```markdown
---

> Material adaptado de **MIT 6.S191 (2020) Lecture 2: Deep Sequence Modeling**, Alexander Amini & Ava Soleimany. [Video](https://www.youtube.com/watch?v=SEnXr6v2ifU) - [Slides oficiales](http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf) - [Sitio del curso](http://introtodeeplearning.com/2020/). Notas en espanol con investigacion complementaria. Sin afiliacion oficial con MIT.
```

### Tarea B: Consolidar `profundizacion.md`

Tomar `docs/handoffs/mit-rnn-profundizacion.md` y simplemente agregar:

- Frontmatter:
  ```yaml
  ---
  title: "Profundizacion - MIT 6.S191 RNNs"
  weight: 20
  math: true
  ---
  ```
- Footer atribucion (mismo de arriba).

El primer `# Profundizacion: ...` heading dentro del archivo se puede eliminar (queda como title del frontmatter).

### Tarea C: Consolidar `glosario.md`

Tomar `docs/handoffs/mit-rnn-glosario.md` y agregar:

- Frontmatter:
  ```yaml
  ---
  title: "Glosario - MIT 6.S191 RNNs"
  weight: 30
  ---
  ```
- Footer atribucion.

### Tarea D: Crear `videos/_index.md` (landing de seccion)

```markdown
---
title: "Videos"
weight: 90
sidebar:
  open: true
---

Charlas y video-lectures de referencia, con notas en espanol e investigacion complementaria.

{{< cards >}}
  {{< card link="mit-6s191-rnn" title="MIT 6.S191 (2020): RNNs" subtitle="Ava Soleimany - Deep Sequence Modeling" icon="film" >}}
{{< /cards >}}
```

### Tarea E: Crear `videos/mit-6s191-rnn/_index.md` (landing del video)

```markdown
---
title: "MIT 6.S191 (2020): Deep Sequence Modeling"
weight: 10
sidebar:
  open: true
---

**Curso:** MIT 6.S191 - Introduction to Deep Learning (2020)
**Instructora:** Ava Soleimany
**Lecture:** 2 - Deep Sequence Modeling / Recurrent Neural Networks
**Duracion:** ~45 min
**Video:** [YouTube](https://www.youtube.com/watch?v=SEnXr6v2ifU)
**Slides oficiales:** [PDF local (9 MB)](/videos/mit-6s191-rnn/slides.pdf) - [Original MIT](http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf)

{{< youtube SEnXr6v2ifU >}}

## Resumen

El segundo lecture del curso MIT 6.S191 (edicion 2020) cubre las **redes neuronales recurrentes** desde la motivacion hasta el mecanismo de atencion. Ava Soleimany construye el material en tres bloques: primero, por que las arquitecturas feed-forward fallan con datos secuenciales y como una RNN soluciona los cuatro requisitos clave (longitud variable, dependencias largas, orden, parameter sharing). Segundo, el problema del vanishing/exploding gradient que motiva las arquitecturas con compuertas (LSTM, GRU). Tercero, aplicaciones reales -- generacion de musica, clasificacion de sentimiento, traduccion automatica con encoder-decoder -- y la introduccion a attention como solucion al cuello de botella del context vector.

Para nuestro curso UC, este lecture **complementa** las clases 11 (RNNs, Alvaro Soto) y 13 (Seq2Seq y Attention): aporta intuiciones visuales, demos en codigo y un ritmo pedagogico distinto al enfoque mas formal del curso.

{{< cards >}}
  {{< card link="notas" title="Notas" subtitle="Recorrido tematico de las 93 diapositivas" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Papers, comparacion con clase 11, conceptos no cubiertos" icon="beaker" >}}
  {{< card link="glosario" title="Glosario" subtitle="25-35 terminos clave en espanol e ingles" icon="book-open" >}}
  {{< card link="/clases/clase-11" title="Clase 11 (curso UC)" subtitle="Material complementario de Alvaro Soto" icon="academic-cap" >}}
{{< /cards >}}

---

> Material adaptado de **MIT 6.S191 (2020) Lecture 2: Deep Sequence Modeling**, Alexander Amini & Ava Soleimany.
> [Video](https://www.youtube.com/watch?v=SEnXr6v2ifU) - [Slides oficiales](http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf) - [Sitio del curso](http://introtodeeplearning.com/2020/).
> Notas en espanol con investigacion complementaria. Sin afiliacion oficial con MIT.
```

### Tarea F: Verificar Hugo

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify 2>&1 | tail -30
```

Si falla:
- Frontmatter mal formado
- Shortcode `{{< youtube >}}` no soportado por theme — alternativa: HTML iframe directo `<iframe src="https://www.youtube.com/embed/SEnXr6v2ifU" ...></iframe>`
- Links rotos a `/papers/<slug>` (verificar con `ls site/content/papers/`)
- Mermaid diagrams mal formados — el theme debe estar configurado con `markup.goldmark.renderer.unsafe = true` o usar el shortcode mermaid.

### Tarea G: Commit final

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git add site/content/videos site/static/videos docs/handoffs
git commit -m "feat: add MIT 6.S191 (2020) RNN video lecture content"
```

## Tasks tracker (estado al cierre)

```
#1 [completed]    Write design doc
#2 [completed]    Create implementation plan via writing-plans skill
#3 [completed]    Dispatch parallel agents for extraction
#4 [in_progress]  Consolidate agent outputs into Hugo content   <-- AQUI
#5 [pending]      Verify with hugo build and commit
```

## Notas para el agente que retome

1. Los 5 archivos de extraccion estan en **`docs/handoffs/mit-rnn-*.md`** (persistidos), no solo en `/tmp` (que se borra).
2. Hay un overlap LSTM entre partes 2 y 3 — no copiar ambos, elegir uno (parte 2 es mas detallada).
3. Mantener acentos consistentes con `site/content/clases/clase-11/teoria.md` (que evita acentos en muchas palabras: "esta", "tambien", "matematicas"). Las extracciones de los agentes mezclan estilos -- normalizar al consolidar.
4. Verificar que el shortcode `{{< youtube >}}` funcione con el theme actual antes de comitear. Si no, usar iframe.
5. Verificar que los cross-links a `/clases/clase-11/profundizacion#parte-ii-...` resuelvan al ID de heading correcto -- los anchors auto-generados de Hugo dependen de la version.
6. Considerar agregar `weight: 90` o lo que corresponda en `videos/_index.md` para posicionarlo donde quieras en el sidebar.

## Comandos rapidos para retomar

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc

# ver estado
git status
git log --oneline -10
ls docs/handoffs/

# ver lo que ya esta en el sitio
ls site/content/videos/
ls site/static/videos/mit-6s191-rnn/

# leer las extracciones
cat docs/handoffs/mit-rnn-notas-parte-1.md
# ... etc
```
