# Lab 13 — Tracking colaborativo (Colab + local)

**Fecha inicio:** 2026-04-27
**Notebooks origen:**
- `clase_13/material/Laboratorio/Practico_clase_13_parte_1.ipynb`
- `clase_13/material/Laboratorio/Practico_clase_13_parte_2.ipynb`
- `clase_13/material/Laboratorio/Practico_clase_13_parte_3.ipynb`

**Notebooks resolucion (a generar en Colab):**
- `clase_13/material/Laboratorio/Practico_13_parte_1_RAE.ipynb`
- `clase_13/material/Laboratorio/Practico_13_parte_2_RAE.ipynb`
- `clase_13/material/Laboratorio/Practico_13_parte_3_RAE.ipynb`

## Modalidad

- Roberto ejecuta en Colab y entrega outputs/screenshots
- Claude integra outputs en los RAE notebooks + site content `site/content/laboratorios/lab-13/`

## Estructura del lab

- **Parte 1 — Seq2Seq basico** (encoder + decoder sin attention) — translation
- **Parte 2 — Seq2Seq + Attention** (Bahdanau additive + visualizacion del attention map)
- **Parte 3 — Teacher Forcing** + Actividades 1.1 y 1.2

## Checklist de avance

### Parte 1

- [ ] Setup + dataset cargado
- [ ] Encoder + Decoder construidos y verificados
- [ ] Entrenamiento (n_epochs por confirmar) — outputs: loss curve + ejemplos de traduccion
- [ ] Evaluacion: metrica usada en el notebook (BLEU u otra)

### Parte 2

- [ ] Attention module implementado y verificado
- [ ] Entrenamiento con attention — outputs (loss, ejemplos de traduccion)
- [ ] Visualizacion de attention heatmap — outputs (PNGs)
- [ ] Comparacion contra parte 1 (calidad cualitativa)

### Parte 3

- [ ] Teacher forcing implementado
- [ ] Comparacion entrenamiento con/sin teacher forcing
- [ ] **Actividad 1.1** resuelta
- [ ] **Actividad 1.2** resuelta

### Site content

- [x] `site/content/laboratorios/lab-13/_index.md` (Fase 1 — scaffolding)
- [x] `site/content/laboratorios/lab-13/seq2seq-basico.md` (Fase 1 — conceptual + placeholders)
- [x] `site/content/laboratorios/lab-13/seq2seq-attention.md` (Fase 1 — conceptual + placeholders)
- [x] `site/content/laboratorios/lab-13/teacher-forcing.md` (Fase 1 — conceptual + placeholders)
- [x] `site/content/laboratorios/lab-13/ejercicios.md` (Fase 1 — enunciados)
- [x] `site/content/laboratorios/lab-13/resolucion.md` (Fase 1 — esqueleto)
- [x] +card en `site/content/laboratorios/_index.md` (Fase 1)
- [ ] Reemplazar placeholders `[outputs pendientes]` con resultados reales (Fase 2, por parte)

### Static assets

- [ ] Curvas loss/acc parte 1 — `loss-base.png`, etc.
- [ ] Curvas loss/acc parte 2 — `loss-attn.png`, etc.
- [ ] Attention heatmap(s) — `attn-heatmap-*.png`
- [ ] Curvas comparativas teacher forcing — `loss-tf.png`, etc.
- [ ] Renders HTML + .ipynb descargables (3 c/u)

## Decisiones (a confirmar con Roberto durante Fase 2)

- **Par de idiomas del dataset**: por confirmar al ejecutar parte 1.
- **Metrica de evaluacion**: BLEU u otra que use el notebook.
- **n_epochs por defecto**: mantener lo que diga el notebook salvo que necesitemos ajustar por tiempo de Colab.
- **Tamano del dataset**: por confirmar — el notebook puede usar un subset por velocidad.

## Hallazgos / Insights

[Vacio al inicio. Se completa durante ejecucion de cada parte.]

## Resultados

[Vacio al inicio. Se completa durante ejecucion de cada parte. Replicar el formato de tablas por epoch del lab-12-tracking.md.]
