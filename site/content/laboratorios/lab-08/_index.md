---
title: "Lab 08 - Entrenamiento Avanzado"
weight: 30
sidebar:
  open: true
---

**Clase:** Clase 08 — Funciones de Perdida, Regularizacion y Tareas Auxiliares
**Profesor:** Carlos Aspillaga

Este laboratorio cubre tres pilares del entrenamiento avanzado de redes neuronales: como elegir la funcion de perdida adecuada para cada problema (MSE vs Cross-Entropy), como prevenir el overfitting con tecnicas de regularizacion (L1, L2, Dropout), y como mejorar las representaciones internas del modelo mediante tareas auxiliares (multi-task learning).

{{< concept-alert type="clave" >}}
Al terminar este lab seras capaz de seleccionar la funcion de perdida correcta segun el tipo de tarea, aplicar regularizacion para controlar el overfitting, y disenar arquitecturas multi-tarea con CombinedLoss.
{{< /concept-alert >}}

{{< cards >}}
  {{< card link="funciones-perdida" title="Funciones de Perdida" subtitle="MSE, Cross-Entropy, Softmax y cuando usar cada una" icon="calculator" >}}
  {{< card link="regularizacion" title="Regularizacion" subtitle="L2, L1, Dropout y el tradeoff bias-varianza" icon="shield-check" >}}
  {{< card link="tareas-auxiliares" title="Tareas Auxiliares" subtitle="Multi-task learning, CombinedLoss y CelebA" icon="puzzle" >}}
  {{< card link="ejercicios" title="Ejercicios" subtitle="3 experimentos con actividades practicas" icon="pencil" >}}
  {{< card link="resolucion" title="Resolucion" subtitle="Notebook con las resoluciones" icon="check-circle" >}}
{{< /cards >}}

## Notebook completo

{{< notebook-viewer
    src="/notebooks-html/lab08.html"
    download="/notebooks/lab08.ipynb"
    title="Laboratorio 8 - Entrenamiento Avanzado" >}}
