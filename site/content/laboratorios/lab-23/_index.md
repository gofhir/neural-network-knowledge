---
title: "Lab 23 - VQA e Image Captioning con BLIP"
weight: 230
sidebar:
  open: true
---

**Profesora:** Bianca Del Solar Medrano
**Fecha:** Junio 2026
**Notebook origen:** `clase_23/material/Laboratorio/Lab23_VQA_ImageCaptioning_v3.ipynb` (50 celdas)
**Notebook ejecutado:** [lab23.ipynb](/notebooks/lab23.ipynb) · [HTML](/notebooks-html/lab23.html)

## Encuadre

Laboratorio práctico sobre **BLIP** ([Li et al. 2022](/papers/blip-li-2022)), el primer **Vision-Language Model unificado** del curso. Donde la [clase teórica](/clases/clase-23) ancló VQA en **Pythia** —clasificación sobre un vocabulario cerrado de ~3000 respuestas con detección de regiones, atención top-down y fusión Hadamard—, el lab usa BLIP para hacer VQA y captioning como **generación de texto libre**. Ese contraste es el hilo conductor de todo el recorrido:

| | Pythia (clase teórica) | BLIP (este lab) |
|---|---|---|
| Paradigma VQA | **Clasificación** | **Generación** |
| Respuestas | Vocabulario cerrado (~3000) | Vocabulario abierto |
| Capa de salida | `sigmoid` multi-etiqueta + BCE | `generate()` autoregresivo |
| Encoder visual | Mask R-CNN + ResNet-101 (**regiones**) | ViT-B/16 (**parches**) |
| Fusión | Producto de Hadamard | **Cross-attention** texto→imagen |
| Arquitectura | Pipeline modular específico | **MED** unificado (3 modos) |

El lab usa dos checkpoints de HuggingFace: `Salesforce/blip-vqa-base` (`BlipForQuestionAnswering`) y `Salesforce/blip-image-captioning-base` (`BlipForConditionalGeneration`).

## Recorrido

1. **Arquitectura: BLIP y el MED** (celdas 0-6): qué es BLIP, el Multimodal mixture of Encoder-Decoder con sus 3 modos, CapFilt, y la carga del modelo VQA en HuggingFace.
2. **VQA como generación** (celdas 7-24): inferencia con `model.generate`, conocimiento del mundo ("olives"), el límite estructural del conteo (jirafas) y un bug del notebook.
3. **Modos de fallo** (celdas 25-29): una taxonomía de los 4 errores de VQA-generación — espacial, vago, granularidad, alucinación OOD.
4. **Image Captioning con BLIP** (celdas 30-34): captioning incondicional, la corrección del error del material (Q-Former es de BLIP-2), y la alucinación dependiente de la tarea.
5. **Decodificación, BLEU y robustez** (transversal): el gotcha real de carga de imágenes en Colab, y los conceptos de decoding/BLEU que el lab promete pero no ejecuta.
6. **Actividad resuelta** (celdas 35-49): las 7 preguntas de opción múltiple con justificación.

## Resultados consolidados

### VQA (blip-vqa-base, greedy)

| Imagen | Pregunta | Respuesta | Veredicto |
|---|---|---|---|
| Mujer + perro + playa | What is this? | `dog and beach` | ✅ Correcta pero parcial |
| Mujer + perro + playa | is there a girl? | `yes` | ✅ |
| Ensalada (PnP-VQA) | What is the black objects on the salad called? | `olives` | ✅ Requiere conocimiento del mundo |
| Jirafas (COCO) | How many giraffes are there? | `1` | ❌ Falla de conteo |
| Perro + silla | Is the dog in front of the chair? | `yes` | ❌ Falla espacial |
| Ornitorrinco | What kind of animal is this? | `monkey` | ❌ Alucinación OOD |

### Captioning (blip-image-captioning-base, `max_length=20`)

| Imagen | Caption generado | Veredicto |
|---|---|---|
| Perro | `a white dog sitting in the grass` | ✅ Preciso |
| Ornitorrinco | `a baby bird is held in a box` | ❌ Alucinación (animal + contexto) |
| Grupo de jóvenes | `group of young people standing in front of white brick wall` | ✅ Bueno, evita el conteo |

→ La **misma** imagen del ornitorrinco alucina distinto según la tarea: `monkey` en VQA (una palabra-clase), `a baby bird is held in a box` en captioning (una escena entera fabricada por exposure bias). Misma causa raíz: entrada **fuera de distribución** + obligación de generar una salida confiada.

### Actividad

| # | Tema | Respuesta |
|---|------|-----------|
| 1 | Qué es VQA | **b** |
| 2 | VQAv2 anti-sesgo | **b** |
| 3 | Problemas de Pythia | **b** |
| 4 | Más variado que greedy | **a** (Beam Search) |
| 5 | Caption ornitorrinco | **a** |
| 6 | Qué mide BLEU | **b** |
| 7 | Error perro/silla | **b** |

## Bloques del lab

{{< cards >}}
  {{< card link="arquitectura-blip" title="Arquitectura: BLIP y el MED" subtitle="BLIP, el Multimodal mixture of Encoder-Decoder (3 modos), CapFilt, carga del modelo VQA" icon="academic-cap" >}}
  {{< card link="vqa-generacion" title="VQA como generación" subtitle="model.generate, pipeline del processor, olives, el límite del conteo y el bug de la celda 24" icon="academic-cap" >}}
  {{< card link="modos-de-fallo" title="Modos de fallo: una taxonomía" subtitle="Espacial, vago, granularidad, alucinación OOD — y por qué alucinan los VLMs" icon="academic-cap" >}}
  {{< card link="image-captioning-blip" title="Image Captioning con BLIP" subtitle="Captioning incondicional, corrección Q-Former vs MED, alucinación dependiente de tarea" icon="academic-cap" >}}
  {{< card link="decoding-y-robustez" title="Decodificación, BLEU y robustez en Colab" subtitle="Gotcha de carga de imágenes (404 en Colab), greedy/beam/nucleus, BLEU, captioning condicional" icon="adjustments" >}}
  {{< card link="actividad" title="Actividad resuelta (7 preguntas)" subtitle="Las 7 preguntas de opción múltiple con justificación de cada alternativa" icon="academic-cap" >}}
{{< /cards >}}

## Papers de este lab

{{< cards >}}
  {{< card link="/papers/blip-li-2022" title="BLIP (2022)" subtitle="Li et al. — el modelo del lab: MED unificado + CapFilt, VQA y captioning como generación" icon="document-text" >}}
  {{< card link="/papers/pythia-jiang-2018" title="Pythia v0.1 (2018)" subtitle="Jiang et al. — el contraste: VQA como clasificación con regiones y Hadamard" icon="document-text" >}}
  {{< card link="/papers/bleu-papineni-2002" title="BLEU (2002)" subtitle="Papineni et al. — la métrica de evaluación de captions" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/vision-language-models" title="Fundamento: Vision-Language Models" subtitle="ViT, cross-attention, contrastivo vs generativo, alucinación en VLMs" icon="book-open" >}}
  {{< card link="/fundamentos/visual-question-answering" title="Fundamento: Visual Question Answering" subtitle="La tarea, datasets, language priors, métrica de consenso" icon="book-open" >}}
  {{< card link="/fundamentos/image-captioning" title="Fundamento: Image Captioning" subtitle="Encoder-decoder, atención, decoding, métricas" icon="book-open" >}}
  {{< card link="/fundamentos/bleu-metric" title="Fundamento: BLEU" subtitle="Modified n-gram precision, brevity penalty" icon="book-open" >}}
  {{< card link="/fundamentos/decoding-strategies" title="Fundamento: Decoding Strategies" subtitle="Greedy, beam search, top-p/nucleus, temperatura" icon="adjustments" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-23" title="Clase 23 - Teoría" subtitle="VQA con Pythia, VQAv2, problemas estructurales, captioning, BLEU" icon="academic-cap" >}}
  {{< card link="/clases/clase-23/profundizacion" title="Profundización" subtitle="Top-down attention, fusión bilineal, beam search, BLEU paso a paso" icon="beaker" >}}
  {{< card link="/dominios/multimodal" title="Dominio: Multimodal" subtitle="De captioning temprano (2014) a los VLMs frontier" icon="globe-alt" >}}
  {{< card link="/laboratorios/lab-22" title="Lab 22 - Summarization (anterior)" subtitle="BertSum extractivo + T5 abstractivo, decoding, ROUGE" icon="academic-cap" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las 50 celdas del notebook con 6 páginas temáticas. Incluye un gotcha real resuelto (CDN devolviendo 404 a las IPs de Colab → patrón `load_image` robusto + URLs de Wikimedia), la corrección de un error del material (Q-Former es de BLIP-2, no de BLIP-1), un bug del notebook (`inputs4`/`inputs5` en la celda 24), y el análisis profundo de la alucinación dependiente de la tarea. Notebook ejecutado en Colab.
