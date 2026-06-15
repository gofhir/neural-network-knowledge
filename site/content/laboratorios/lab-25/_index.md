---
title: "Lab 25 - Recomendación multimodal con imágenes y texto"
weight: 250
sidebar:
  open: true
---

**Profesores:** Julio Hurtado & Felipe del Río
**Fecha:** Junio 2026
**Notebook origen:** `clase_25/material/Laboratorio/Laboratorio 25 - Recomendación usando imágenes y texto.ipynb` (81 celdas)
**Notebook ejecutado:** [lab25.ipynb](/notebooks/lab25.ipynb) · [HTML](/notebooks-html/lab25.html)

## Encuadre

"Repaso aplicado" que junta todo el curso —visión (AlexNet), NLP (BERT) y recomendación— en un **sistema de recomendación content-based multimodal** estilo Pinterest: recomendar pares imagen+comentario a un usuario según sus interacciones previas. Es la contraparte práctica de la [clase 25](/clases/clase-25).

El hilo conductor es un **proxy task**: el problema real (recomendar) no tiene etiquetas directas, así que se entrena un modelo a **clasificar a qué usuario pertenece cada par imagen-texto** (que sí tiene etiquetas). Al aprender a clasificar usuarios, el modelo construye un espacio de representaciones donde el contenido de un mismo usuario queda agrupado; luego se **descarta el clasificador** (`features=True`) y se usa el descriptor intermedio para **recomendar por vecino más cercano**. Es [metric learning](/fundamentos/triplet-loss) vía tarea pretexto, y en el fondo un [two-tower](/fundamentos/two-tower-retrieval).

| Pieza | Implementación en el lab |
|---|---|
| Representación de imagen | Descriptores **fc7 de AlexNet** (4096-d) pre-computados (ImageNet, por copyright no hay píxeles) |
| Representación de texto | **BERT** (`bert-base-uncased`) sobre el comentario |
| Fusión | Concatenación → capa densa → descriptor de **32-d** |
| Tarea de entrenamiento | Clasificación de usuario (CrossEntropy) — el **proxy task** |
| Recomendación | Distancia coseno mínima a los ítems del usuario + top-k |
| Evaluación | Precision@k, Recall@k, **nDCG** |

## Resultados consolidados

### Multimodal vs baseline solo-imagen (val acc, 5 épocas, 10 usuarios)

| Modelo | Val Acc final | Comportamiento |
|---|---|---|
| **Multimodal (imagen + texto)** | **71.48%** | Sigue subiendo, sin overfitting |
| **Baseline (solo imagen)** | **56.84%** | Satura en ~58%, empieza a sobreajustar en la época 5 |

→ El texto aporta **~15 puntos** sobre el azar (10%) → confirma que el comentario carga señal de gusto que la imagen sola no captura.

### Evaluación de la recomendación: dos bugs de métrica descubiertos y corregidos

| Métrica (k=400) | Con bugs | Corregido |
|---|---|---|
| nDCG | 0.022 | **0.857** |
| Precision | 0.1725 (1 usuario) | **0.227** (promedio 10) |
| Recall | 0.69 (1 usuario) | **0.908** (promedio 10) |

→ El nDCG saltó **~40×** al corregir dos bugs (pasar distancias donde se esperaba similitud; no promediar sobre usuarios). **El recomendador siempre funcionó bien** —P@400 es 2.3× el azar, R@400 captura el 91% de los relevantes—; el problema estaba en la *medición*, no en el modelo. Medir bien es tan importante como modelar bien.

## Bloques del lab

{{< cards >}}
  {{< card link="planteamiento-y-datos" title="Planteamiento del problema y datos" subtitle="Content-based multimodal, proxy task, dataset Pinterest de descriptores, Actividad 1" icon="academic-cap" >}}
  {{< card link="dataset-y-modelo" title="Dataset multimodal y arquitectura" subtitle="ContentRecommender, AlexNet fc7 + BERT, ModelClass (multimodal) vs baseline, features=True" icon="academic-cap" >}}
  {{< card link="entrenamiento" title="Entrenamiento: multimodal vs baseline" subtitle="CrossEntropy/Adam, resultados 71.5% vs 56.8%, val>train por dropout, Actividad 2" icon="academic-cap" >}}
  {{< card link="recomendacion" title="Recomendación por similitud" subtitle="Descriptores 32-d, scoring por distancia mínima, top-k con argpartition" icon="academic-cap" >}}
  {{< card link="evaluacion-y-metricas" title="Evaluación: nDCG y dos bugs de métrica" subtitle="Precision/Recall/nDCG, los bugs (0.02→0.86), el valle en k=100, Actividad 3" icon="academic-cap" >}}
{{< /cards >}}

## Papers relacionados

{{< cards >}}
  {{< card link="/papers/pinterest-dataset-2017" title="Pinterest Dataset (2017)" subtitle="El dataset del lab: pins (imagen + comentario), features CNN 4096-d" icon="document-text" >}}
  {{< card link="/papers/vbpr-he-2016" title="VBPR (2016)" subtitle="He & McAuley — recomendación visual con features CNN, la base conceptual del lab" icon="document-text" >}}
  {{< card link="/papers/alexnet-krizhevsky-2012" title="AlexNet (2012)" subtitle="Krizhevsky et al. — la CNN de donde salen los descriptores fc7 de 4096-d" icon="document-text" >}}
  {{< card link="/papers/bpr-rendle-2009" title="BPR (2009)" subtitle="Rendle et al. — pérdida de ranking, mejora propuesta en la Actividad 2" icon="document-text" >}}
  {{< card link="/papers/two-tower-yi-2019" title="Two-Tower (2019)" subtitle="Yi et al. — el esquema escalable hacia el que apunta el proxy task" icon="document-text" >}}
  {{< card link="/papers/ndcg-jarvelin-2002" title="DCG / nDCG (2002)" subtitle="Järvelin & Kekäläinen — la métrica de ranking del lab" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/recommender-systems" title="Sistemas de Recomendación" subtitle="Content-based vs collaborative, cold start, recomendación multimodal" icon="book-open" >}}
  {{< card link="/fundamentos/ranking-metrics" title="Métricas de Ranking" subtitle="Precision@k, Recall@k, MAP, MRR, DCG/nDCG" icon="book-open" >}}
  {{< card link="/fundamentos/two-tower-retrieval" title="Two-Tower Retrieval" subtitle="Dual encoder, retrieval por vecinos, candidate generation" icon="book-open" >}}
  {{< card link="/fundamentos/triplet-loss" title="Triplet Loss y Metric Learning" subtitle="El espacio donde ítems co-preferidos quedan cerca" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer Learning" subtitle="CNN pre-entrenada como extractor de features (los descriptores)" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-25" title="Clase 25 - Teoría" subtitle="Framework problem/data/model, two-tower, métricas de ranking" icon="academic-cap" >}}
  {{< card link="/clases/clase-25/profundizacion" title="Profundización" subtitle="Formalización de r_ij, metric learning, derivación de nDCG" icon="beaker" >}}
  {{< card link="/dominios/recomendacion" title="Dominio: Recomendación" subtitle="De GroupLens 1994 a la recomendación generativa con LLMs" icon="globe-alt" >}}
  {{< card link="/laboratorios/lab-24" title="Lab 24 - Question Answering (anterior)" subtitle="QA extractivo vs generativo en español" icon="academic-cap" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las 81 celdas del notebook con 5 páginas temáticas. Incluye los resultados medidos (multimodal 71.5% vs baseline 56.8%), el descubrimiento y corrección de **dos bugs de evaluación** (nDCG de 0.02 a 0.86), y gotchas reales (versiones de `transformers`/`torch` de 2020 incompatibles con el Colab actual, `gdown --id` deprecado, `device` siempre en CPU por un typo). Notebook ejecutado en Colab.
