---
title: "Lab 09 - Visualizacion e Interpretabilidad en CNNs"
weight: 10
sidebar:
  open: true
---

**Clase:** Clase 09 — CNNs en Profundidad  
**Profesor:** Miguel Fadic  
**Fecha:** 2026-04-07

Este laboratorio explora dos tecnicas para entender que aprende una red neuronal convolucional:

1. **Feature Visualization** — que inputs maximizan la respuesta de una capa o neurona
2. **Attribution** — que partes del input son responsables de la clasificacion

{{< concept-alert type="clave" >}}
Al terminar este lab seras capaz de navegar las capas de cualquier arquitectura CNN, visualizar sus representaciones internas y usar Extremal Perturbation para auditar decisiones del modelo.
{{< /concept-alert >}}

{{< cards >}}
  {{< card link="librerias" title="Librerias" subtitle="torch-lucent, torchray, PIL y matplotlib" icon="cube" >}}
  {{< card link="capas" title="Acceso a Capas" subtitle="get_model_layers y mapeo por arquitectura" icon="adjustments" >}}
  {{< card link="feature-viz" title="Feature Visualization" subtitle="Algoritmo, channel viz, label viz" icon="photograph" >}}
  {{< card link="flores-overfitting" title="Flores y Overfitting" subtitle="MiAlexNet, fine-tuning y comparacion visual" icon="sparkles" >}}
  {{< card link="attribution" title="Attribution" subtitle="Extremal Perturbation paso a paso" icon="location-marker" >}}
  {{< card link="ejercicios" title="Ejercicios" subtitle="5 actividades + glosario" icon="pencil" >}}
  {{< card link="resolucion" title="Resolución" subtitle="Resultados con ResNet50" icon="check-circle" >}}
{{< /cards >}}

## Notebook completo (Colab final)

{{< notebook-viewer
    src="/notebooks-html/lab09.html"
    download="/notebooks/lab09.ipynb"
    title="Laboratorio 9 - Visualización e Interpretabilidad en CNNs (Resolución)" >}}
