---
title: "Lab 13 - Seq2Seq y Mecanismos de Atencion"
weight: 130
sidebar:
  open: true
---

**Profesor:** Gabriel Sepulveda
**Fecha:** Abril 2026
**Notebooks origen:** `clase_13/material/Laboratorio/Practico_clase_13_parte_{1,2,3}.ipynb`

## Encuadre

Laboratorio dividido en **tres notebooks** que construyen progresivamente un modelo de traduccion automatica usando arquitecturas Seq2Seq con RNNs:

- **Parte 1 — Seq2Seq basico**: encoder y decoder LSTM sin attention. El context vector unico es el ultimo hidden state del encoder.
- **Parte 2 — Seq2Seq con Attention**: agrega un attention module (Bahdanau additive) y visualiza el attention heatmap sobre los tokens fuente.
- **Parte 3 — Teacher Forcing**: introduce teacher forcing para estabilizar el entrenamiento, mas dos actividades evaluadas (1.1 y 1.2).

Para la teoria detras de cada arquitectura ver la [clase 13](/clases/clase-13/).

## Recursos del lab

{{< cards >}}
  {{< card link="seq2seq-basico" title="Parte 1 - Seq2Seq basico" subtitle="Encoder-decoder sin attention" icon="academic-cap" >}}
  {{< card link="seq2seq-attention" title="Parte 2 - Seq2Seq con Attention" subtitle="Bahdanau attention + visualizacion" icon="academic-cap" >}}
  {{< card link="teacher-forcing" title="Parte 3 - Teacher Forcing" subtitle="Estabilizacion del entrenamiento + actividades" icon="academic-cap" >}}
  {{< card link="ejercicios" title="Ejercicios" subtitle="Enunciados Actividades 1.1 y 1.2" icon="document-text" >}}
  {{< card link="resolucion" title="Resolucion" subtitle="Respuestas a las actividades + insights" icon="check-circle" >}}
{{< /cards >}}

## Notebooks (Colab + descarga)

{{< cards >}}
  {{< card link="/notebooks/lab13-parte-1.ipynb" title="Notebook Parte 1" subtitle="Seq2Seq basico (.ipynb descargable)" icon="document" >}}
  {{< card link="/notebooks/lab13-parte-2.ipynb" title="Notebook Parte 2" subtitle="Seq2Seq + Attention (.ipynb descargable)" icon="document" >}}
  {{< card link="/notebooks/lab13-parte-3.ipynb" title="Notebook Parte 3" subtitle="Teacher Forcing + actividades (.ipynb descargable)" icon="document" >}}
{{< /cards >}}

## Renders HTML

{{< cards >}}
  {{< card link="/notebooks-html/lab13-parte-1.html" title="Render HTML Parte 1" subtitle="Notebook ejecutado renderizado" icon="document-text" >}}
  {{< card link="/notebooks-html/lab13-parte-2.html" title="Render HTML Parte 2" subtitle="Notebook ejecutado renderizado" icon="document-text" >}}
  {{< card link="/notebooks-html/lab13-parte-3.html" title="Render HTML Parte 3" subtitle="Notebook ejecutado renderizado" icon="document-text" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-13" title="Clase 13 - Teoria" subtitle="Recorrido tematico del lecture de Gabriel Sepulveda" icon="academic-cap" >}}
  {{< card link="/fundamentos/seq2seq" title="Fundamento: Seq2Seq" subtitle="Encoder-decoder, teacher forcing, beam search" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Fundamento: Attention" subtitle="Bahdanau/Luong/scaled dot-product, soft vs hard" icon="book-open" >}}
{{< /cards >}}

---

> **Estado actual:** Fase 1 (scaffolding). Las paginas conceptuales estan completas; los outputs reales (curvas, attention heatmaps, ejemplos de traduccion) se integran en Fase 2 segun Roberto ejecuta cada notebook en Colab.
