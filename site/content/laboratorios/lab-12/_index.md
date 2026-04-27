---
title: "Lab 12 - Data Augmentation, Transfer Learning y Finetuning"
weight: 30
sidebar:
  open: true
---

**Clase:** Clase 12 — Data Augmentation y Transfer Learning
**Profesor:** Felipe Del Río
**Fecha:** 2026-04-16

Este laboratorio combina dos técnicas fundamentales para entrenar modelos de deep learning con datos limitados: **data augmentation** (sintetizar más ejemplos a partir de los existentes) y **transfer learning / finetuning** (reutilizar modelos preentrenados). Se aplica en dos dominios distintos: clasificación de imágenes (Oxford 102 Flowers + ResNet18) y clasificación multi-label de texto (Jigsaw Toxic Comments + BERT).

{{< cards >}}
  {{< card link="data-augmentation" title="Data Augmentation" subtitle="Transforms en imagenes, custom transforms y modelo aug vs base" icon="photograph" >}}
  {{< card link="transfer-learning" title="Transfer Learning" subtitle="Finetuning ResNet18 desde ImageNet sobre flowers" icon="refresh" >}}
  {{< card link="bert-finetuning" title="BERT Finetuning" subtitle="Jigsaw multi-label, BERT desde cero vs preentrenado" icon="chip" >}}
  {{< card link="ejercicios" title="Ejercicios" subtitle="3 actividades practicas" icon="pencil" >}}
  {{< card link="resolucion" title="Resolución" subtitle="Respuestas de los 3 ejercicios" icon="check-circle" >}}
{{< /cards >}}

## Notebook completo (Colab final)

{{< notebook-viewer
    src="/notebooks-html/lab12.html"
    download="/notebooks/lab12.ipynb"
    title="Laboratorio 12 - Data Augmentation, Transfer Learning y Finetuning (Resolución)" >}}
