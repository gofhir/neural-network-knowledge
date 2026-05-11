---
title: "Lab 14 - Transformers e Interpretabilidad + CLIP"
weight: 140
sidebar:
  open: true
---

**Profesor:** Gabriel Sepulveda
**Fecha:** Mayo 2026
**Notebooks origen:** `clase_14/material/Laboratorio/Laboratorio 14 - Transformers - Parte {1,2}.ipynb`

## Encuadre

Laboratorio dividido en **dos notebooks** que tocan tematicas distintas pero complementarias dentro del universo Transformer:

- **Parte 1 — Inspeccionando atenciones**: usar BETO (BERT pre-entrenado en espanol) fine-tuned para NER, y abrir el capo con `bertviz` para ver como distribuye atencion token-a-token, capa-a-capa, cabeza-a-cabeza. Cierra con Actividad 1 (comparacion entre versiones de BERT) y Actividad 2 (preguntas conceptuales sobre el decoder).
- **Parte 2 — CLIP zero-shot**: cambia totalmente de tema. Usa **CLIP** (Contrastive Language-Image Pre-training de OpenAI 2021) — un modelo multimodal que vive en un espacio compartido entre texto e imagen — para clasificacion zero-shot sobre Food101 (84% Top-1) y Stanford Cars (58% Top-1). Cierra con Actividad 3 (prompt engineering del template) y Actividad 4 (matriz de similitud con imagenes propias).

Para la teoria detras de la arquitectura Transformer ver la [clase 14](/clases/clase-14/).

## Resultados consolidados

| Experimento | Metrica | Resultado |
| --- | --- | --- |
| BETO NER (Parte 1) | Visualizacion 144 cabezas | Patron sink-CLS → no-op-SEP → diversidad final emerge sin supervision |
| mBERT vs bert-uncased (Act. 1) | Cabezas sintacticas | mBERT muestra Alexis → scored (sujeto-verbo); bert-uncased se va a [SEP] |
| Food101 baseline (Parte 2) | Top-1 / Top-5 | **84.01% / 97.31%** |
| Food101 + Q1 `"A photo of {}."` (Act. 3) | Top-1 / Top-5 | 78.41% / 94.93% (−5.6/−2.4) |
| Food101 + Q2 `"A close-up photo of a plate of {}, a popular dish."` (Act. 3) | Top-1 / Top-5 | 82.49% / 96.88% (−1.5/−0.4) |
| Stanford Cars (Parte 2) | Top-1 / Top-5 | **57.93% / 89.64%** |
| 5 imagenes ImageNet (Act. 4) | Matriz 5×5 | Diagonal correcta — separa clases distintas |

## Recursos del lab — Parte 1 (BETO + bertviz)

{{< cards >}}
  {{< card link="tokenizacion-y-ner" title="Tokenizacion + NER con BETO" subtitle="WordPiece, [CLS]/[SEP], displacy" icon="academic-cap" >}}
  {{< card link="visualizacion-atenciones" title="Visualizacion de atenciones" subtitle="head_view y model_view de bertviz" icon="academic-cap" >}}
  {{< card link="neuron-view-y-modelos" title="Neuron View + Actividad 1" subtitle="Q/K dim por dim, bert-uncased vs mBERT" icon="academic-cap" >}}
  {{< card link="decoder-cross-attention" title="Decoder cross-attention" subtitle="Actividad 2 - preguntas teoricas" icon="academic-cap" >}}
{{< /cards >}}

## Recursos del lab — Parte 2 (CLIP zero-shot)

{{< cards >}}
  {{< card link="clip-setup-y-zero-shot" title="CLIP - setup y zero-shot" subtitle="ViT-B/32 + Food101 caso individual" icon="academic-cap" >}}
  {{< card link="food101-evaluacion-y-templates" title="Food101 + Actividad 3" subtitle="84% Top-1 + prompt engineering" icon="academic-cap" >}}
  {{< card link="stanford-cars-limites" title="Stanford Cars + Actividad 4" subtitle="58% Top-1 + los limites de zero-shot" icon="academic-cap" >}}
{{< /cards >}}

## Actividades y resolucion

{{< cards >}}
  {{< card link="ejercicios" title="Ejercicios" subtitle="Enunciados Actividades 1, 2, 3 y 4" icon="document-text" >}}
  {{< card link="resolucion" title="Resolucion" subtitle="Respuestas razonadas + insights consolidados" icon="check-circle" >}}
{{< /cards >}}

## Notebooks (Colab + descarga)

{{< cards >}}
  {{< card link="/notebooks/lab14-parte-1.ipynb" title="Notebook Parte 1" subtitle="Inspeccionando atenciones (.ipynb descargable)" icon="document" >}}
  {{< card link="/notebooks/lab14-parte-2.ipynb" title="Notebook Parte 2" subtitle="CLIP zero-shot (.ipynb descargable)" icon="document" >}}
{{< /cards >}}

## Renders HTML

{{< cards >}}
  {{< card link="/notebooks-html/lab14-parte-1.html" title="Render HTML Parte 1" subtitle="Notebook ejecutado renderizado" icon="document-text" >}}
  {{< card link="/notebooks-html/lab14-parte-2.html" title="Render HTML Parte 2" subtitle="Notebook ejecutado renderizado" icon="document-text" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-14" title="Clase 14 - Teoria" subtitle="Recorrido tematico del lecture de Gabriel Sepulveda" icon="academic-cap" >}}
  {{< card link="/fundamentos/transformers" title="Fundamento: Transformers" subtitle="Self-attention, multi-head, positional encoding" icon="book-open" >}}
  {{< card link="/fundamentos/bert" title="Fundamento: BERT" subtitle="MLM, NSP, fine-tuning para tareas downstream" icon="book-open" >}}
{{< /cards >}}

---

> **Estado actual:** Lab completo. Las dos partes estan ejecutadas en Colab y sus resultados (visualizaciones, accuracies, matrices) integrados en las paginas tematicas con screenshots reales. Las 11 preguntas conceptuales de las 4 actividades estan respondidas en [resolucion](resolucion).
