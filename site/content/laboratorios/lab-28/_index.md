---
title: "Lab 28 - Aprendizaje Autosupervisado: UDA (Unsupervised Data Augmentation)"
weight: 280
sidebar:
  open: true
---

**Profesor:** Sebastián Amenábar
**Fecha:** Junio 2026
**Notebook origen:** `clase_28/material/Laboratorio/Practico_Autosupervision_UDA_v18.ipynb`
**Notebook ejecutado:** [lab28.ipynb](/notebooks/lab28.ipynb) · [HTML](/notebooks-html/lab28.html)

## Encuadre

La contraparte práctica de la [clase 28](/clases/clase-28): implementar **UDA** ([Xie et al. 2019](/papers/uda-xie-2019)) para clasificar sentimiento en reseñas **IMDB** con **muy pocas etiquetas** (20), apoyándose en decenas de miles de reseñas **sin etiqueta**.

UDA es **aprendizaje semi-supervisado por consistencia**: aprovecha la asimetría de que las etiquetas son caras pero los datos crudos son baratos. La idea central es que hay transformaciones que **no cambian la etiqueta** de un dato (cambiar una palabra por un sinónimo no vuelve negativa una reseña positiva), así que se puede **forzar** al modelo a predecir lo mismo para un dato y su versión aumentada — **sin conocer su etiqueta real**.

| Pieza | Implementación en el lab |
|---|---|
| Modelo | `bert-base-cased` + cabeza de clasificación (`BertForSequenceClassification`) |
| Aumentación | **back-translation** (EN→FR→EN) con MarianMT — pre-computada para 70k textos |
| Rama supervisada | cross-entropy sobre 20 etiquetas, enmascarada por TSA |
| Rama de consistencia | KL-divergencia entre `P(y|original)` (con `stop_gradient`) y `P(y|aumentado)` |
| TSA | Training Signal Annealing: enmascara ejemplos supervisados demasiado confiados; el umbral crece de 0.5 a 1.0 |
| Balance | `unsup_ratio=3` (3 no-supervisados por cada supervisado en el batch) |

## Resultados consolidados

Tres regímenes evaluados en test (25k reseñas), con los checkpoints oficiales:

| Régimen | Etiquetas | Datos no etiq. | **Test Acc** | **Test Loss** |
|---|---|---|---|---|
| Full (supervisado) | 20.000 | — | **87.65%** | 0.4300 |
| Low (supervisado) | 20 | — | **60.58%** | 2.1453 |
| **UDA (semi-sup.)** | **20** | **~65.000** | **85.06%** | **0.3443** |

### Las lecciones del lab

1. **UDA recupera el ~91% de la brecha.** Con las mismas 20 etiquetas que dieron 60.58%, UDA llega a 85.06% — a solo 2.6 puntos del modelo entrenado con **1000× más etiquetas**. La brecha full−low (27.1 pts) se cierra en 24.5 pts usando solo datos no etiquetados.
2. **UDA está mejor calibrado que el full** (test loss 0.34 < 0.43): la regularización por consistencia + TSA produce probabilidades más honestas, no solo más accuracy. El régimen low, en cambio, es sobreconfianza catastrófica (loss 2.15).
3. **Las tres curvas de entrenamiento tienen firmas distintas:** el full salta de inmediato (50→84% en 250 pasos), el low se estanca con la loss explotando (overfitting a 20 ejemplos), y UDA arranca lento → mesetea → despega tardíamente **sin explotar** (efecto de TSA).
4. **El back-translation no siempre preserva la etiqueta:** a temperatura alta descarrila (`plot→site`, gramática rota); UDA es robusto al ruido de aumentación (por la ley de grandes números + confidence masking), no inmune.

## Bloques del lab

{{< cards >}}
  {{< card link="01-consistencia-y-uda" title="Pérdida de consistencia y el método UDA" subtitle="Las dos ramas, KL-divergencia, el stop_gradient y por qué evita el colapso trivial, la loss combinada" icon="adjustments" >}}
  {{< card link="02-datos-y-back-translation" title="Datos, back-translation y dataloaders" subtitle="IMDB semi-supervisado, back-translation EN→FR→EN, filtro de calidad, los 5 datasets, unsup_ratio" icon="document-text" >}}
  {{< card link="03-tres-regimenes-y-analisis" title="Los tres regímenes, TSA y análisis" subtitle="Full/Low/UDA, TSA como freno anti-overfitting, las tres curvas medidas, la tesis del 91% y la calibración" icon="trending-down" >}}
  {{< card link="04-actividades" title="Actividades resueltas (5 de 6)" subtitle="Por qué cambia BT, qué aumentaciones preservan la etiqueta, filtrar ruido, back-translation × temperatura, contrastivo vs UDA" icon="academic-cap" >}}
{{< /cards >}}

## Papers y fundamentos relacionados

{{< cards >}}
  {{< card link="/papers/uda-xie-2019" title="UDA (2019)" subtitle="Xie et al. — consistency training con aumentaciones de calidad, TSA, el paper de este lab" icon="document-text" >}}
  {{< card link="/papers/simclr-chen-2020" title="SimCLR (2020)" subtitle="Chen et al. — aprendizaje contrastivo, el primo autosupervisado (Actividad 6)" icon="document-text" >}}
  {{< card link="/fundamentos/aprendizaje-semi-supervisado" title="Aprendizaje Semi-supervisado" subtitle="Pocas etiquetas + muchos datos no etiquetados: consistencia, pseudo-labeling" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-autosupervisado" title="Aprendizaje Autosupervisado" subtitle="Aprender sin etiquetas mediante tareas pretexto" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-28" title="Clase 28 - Teoría" subtitle="SSL: tareas pretexto, contrastivo (SimCLR/MoCo), MAE, UDA" icon="academic-cap" >}}
  {{< card link="/clases/clase-28/profundizacion" title="Profundización" subtitle="Math del SSL: InfoNCE, consistencia, sharpening" icon="beaker" >}}
  {{< card link="/fundamentos/aprendizaje-contrastivo" title="Aprendizaje Contrastivo" subtitle="SimCLR/MoCo: invarianza a aumentaciones en el espacio de embeddings" icon="book-open" >}}
  {{< card link="/laboratorios/lab-27" title="Lab 27 - Redes Neuronales de Grafos (anterior)" subtitle="GNN con PyTorch Geometric" icon="arrow-left" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Recorrido celda a celda del notebook (142 celdas) + análisis de los tres regímenes con curvas de entrenamiento propias. Las 5 actividades resueltas (se saltó la Act 5 de entrenar UDA con 20k por costo de cómputo), con la Actividad 4 (back-translation × temperatura) verificada corriendo. Notebook ejecutado en Colab (GPU) con las tres curvas de validación y el schedule de TSA embebidos.
