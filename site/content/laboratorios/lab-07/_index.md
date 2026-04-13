---
title: "Lab 07 - PyTorch"
weight: 20
sidebar:
  open: true
---

**Clase:** Clase 07 --- Tecnicas de Entrenamiento

Este laboratorio introduce PyTorch como framework de deep learning. Se cubren las estructuras de datos fundamentales (tensores), la construccion de modelos con `nn.Module`, el manejo de datos con `Dataset` y `DataLoader`, y el pipeline completo de entrenamiento incluyendo funciones de perdida, optimizadores y evaluacion.

{{< concept-alert type="clave" >}}
Al terminar este lab seras capaz de definir una arquitectura de red neuronal en PyTorch, cargar datos con DataLoaders, implementar un loop de entrenamiento completo y evaluar el rendimiento del modelo en train y test.
{{< /concept-alert >}}

{{< cards >}}
  {{< card link="tensores" title="Tensores" subtitle="Estructura N-dimensional, operaciones y dispositivos" icon="cube" >}}
  {{< card link="modulos-capas" title="Modulos y Capas" subtitle="nn.Module, nn.Linear, nn.Sequential y modelos custom" icon="template" >}}
  {{< card link="entrenamiento" title="Entrenamiento" subtitle="Loss, optimizadores, backward y training loop" icon="lightning-bolt" >}}
  {{< card link="dataloaders" title="DataLoaders" subtitle="Dataset, DataLoader, transforms y datos custom" icon="database" >}}
  {{< card link="ejercicios" title="Ejercicios" subtitle="4 actividades con Dropout, BatchNorm y fine-tuning" icon="pencil" >}}
  {{< card link="resolucion" title="Resolucion" subtitle="Notebook con resoluciones completas" icon="check-circle" >}}
{{< /cards >}}

## Notebook completo

{{< notebook-viewer
    src="/notebooks-html/lab07.html"
    download="/notebooks/lab07.ipynb"
    title="Laboratorio 7 - PyTorch" >}}
