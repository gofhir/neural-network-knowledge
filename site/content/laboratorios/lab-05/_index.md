---
title: "Lab 05 - AlexNet y CNNs"
weight: 10
sidebar:
  open: true
---

**Clase:** Clase 05 — Redes Convolucionales  
**Profesor:** Alain Raymond  
**Fecha:** 2026-03-30

Este laboratorio introduce las redes neuronales convolucionales (CNNs) como alternativa a los MLPs para el procesamiento de imagenes. Se construye la arquitectura AlexNet paso a paso en PyTorch, se explican las operaciones de convolucion y pooling con sus formulas de dimensiones, y se aplica el modelo a la clasificacion de 102 tipos de flores.

{{< concept-alert type="clave" >}}
Al terminar este lab seras capaz de implementar una CNN completa en PyTorch, calcular las dimensiones de salida de cada capa, y entender por que las convoluciones son superiores a las capas lineales para imagenes.
{{< /concept-alert >}}

{{< cards >}}
  {{< card link="capas-convolucionales" title="Capas Convolucionales" subtitle="Convolucion, filtros, stride, padding y eficiencia vs MLPs" icon="template" >}}
  {{< card link="alexnet" title="AlexNet" subtitle="Arquitectura completa, ReLU, Dropout y transfer learning" icon="chip" >}}
  {{< card link="operaciones-pooling" title="Pooling y Dimensiones" subtitle="Max Pooling, Average Pooling y flujo de dimensiones" icon="view-grid" >}}
  {{< card link="ejercicios" title="Ejercicios" subtitle="3 actividades de modificacion de arquitectura" icon="pencil" >}}
  {{< card link="resolucion" title="Resolucion" subtitle="Notebook con soluciones completas" icon="check-circle" >}}
{{< /cards >}}

## Notebook completo

{{< notebook-viewer
    src="/notebooks-html/lab05.html"
    download="/notebooks/lab05.ipynb"
    title="Laboratorio 5 - Redes Convolucionales" >}}
