---
title: "Lab 11 - Redes Recurrentes (RNNs)"
weight: 25
sidebar:
  open: true
---

**Clase:** Clase 11 — Redes Recurrentes (RNNs)
**Profesores:** Alain Raymond, Carlos Aspillaga
**Fecha:** 2026-04-15

Este laboratorio entrena tres arquitecturas recurrentes — RNN vanilla (Elman), LSTM y BiLSTM — para clasificar la nacionalidad de un apellido a partir de sus caracteres. Trabaja a nivel de carácter, con one-hot encoding, batch=1 y SGD hecho a mano.

{{< cards >}}
  {{< card link="dataset-nombres" title="Dataset y Tensores" subtitle="18 idiomas, normalizacion Unicode, one-hot por caracter" icon="database" >}}
  {{< card link="modelos" title="Modelos" subtitle="RNN vanilla (Elman), LSTM y BiLSTM" icon="cube" >}}
  {{< card link="entrenamiento" title="Entrenamiento" subtitle="Loop, batch=1, lr=0.005 y matriz de confusion" icon="play" >}}
  {{< card link="ejercicios" title="Ejercicios" subtitle="2 actividades practicas" icon="pencil" >}}
  {{< card link="resolucion" title="Resolución" subtitle="Respuestas de las 2 actividades" icon="check-circle" >}}
{{< /cards >}}

## Notebook completo (Colab final)

{{< notebook-viewer
    src="/notebooks-html/lab11.html"
    download="/notebooks/lab11.ipynb"
    title="Laboratorio 11 - Redes Recurrentes (Resolución)" >}}
