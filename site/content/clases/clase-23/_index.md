---
title: "Clase 23 - VQA e Image Captioning"
weight: 230
sidebar:
  open: true
---

**Profesora:** Bianca Del Solar Medrano
**Fecha:** 2026-06-01

Séptima clase del bloque de visión avanzada y aplicaciones, y la primera enteramente **multimodal** (visión + lenguaje). Recorre dos tareas que obligan a un modelo a razonar simultáneamente sobre una imagen y un texto: **Visual Question Answering (VQA)** — responder en lenguaje natural una pregunta sobre una imagen — e **Image Captioning** — generar una descripción de una imagen. La clase ancla VQA en el modelo **Pythia** (ganador del VQA Challenge 2018, heredero de la atención Bottom-Up/Top-Down) sobre el dataset balanceado **VQAv2**, y expone los tres problemas estructurales de los modelos VQA (language priors, falta de composicionalidad, respuestas limitadas). La segunda mitad cubre el captioning encoder-decoder, las estrategias de decodificación (**Greedy** vs **Beam Search**) y la métrica **BLEU**.

La clase integra piezas de todo el curso: la [Clase 09 (CNN)](/clases/clase-09) como encoder visual, la [Clase 13 (RNN)](/clases/clase-13) y la GRU como encoder de la pregunta, la [Clase 15 (Mecanismo de atención)](/clases/clase-15) como base de la top-down attention, el [GloVe de la Clase 18](/clases/clase-18) para embeber las preguntas, y las [decoding strategies de la Clase 22](/clases/clase-22) para generar captions.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 29 diapositivas: VQA, dataset VQAv2, modelo Pythia, problemas, Image Captioning, greedy/beam search, BLEU" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math detallada: top-down attention, fusion multimodal, fusion bilineal (MCB/MUTAN), beam search, BLEU paso a paso" icon="beaker" >}}
  {{< card link="/laboratorios/lab-23" title="Laboratorio 23" subtitle="VQA e Image Captioning con BLIP (Salesforce) en HuggingFace" icon="academic-cap" >}}
  {{< card link="/clases/clase-22" title="Clase anterior: Summarization" subtitle="BERTSum, T5, decoding, ROUGE" icon="arrow-left" >}}
  {{< card link="/clases/clase-15" title="Base: Mecanismo de atencion" subtitle="La base de la top-down attention" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/visual-question-answering" title="Visual Question Answering" subtitle="La tarea, datasets, language priors, arquitecturas, metrica de consenso" icon="book-open" >}}
  {{< card link="/fundamentos/image-captioning" title="Image Captioning" subtitle="Encoder-decoder, atencion, decoding, metricas, era VLM" icon="book-open" >}}
  {{< card link="/fundamentos/bleu-metric" title="BLEU Metric" subtitle="Modified n-gram precision, brevity penalty, BLEU en captioning" icon="book-open" >}}
  {{< card link="/fundamentos/decoding-strategies" title="Decoding Strategies" subtitle="Greedy, beam search, top-k/p sampling" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Mecanismo de Atencion" subtitle="Base de la top-down attention de Pythia" icon="book-open" >}}
  {{< card link="/fundamentos/redes-recurrentes" title="Redes Recurrentes (GRU)" subtitle="Encoder de la pregunta en Pythia" icon="book-open" >}}
  {{< card link="/fundamentos/glove" title="GloVe" subtitle="Embeddings de las palabras de la pregunta" icon="book-open" >}}
  {{< card link="/fundamentos/deteccion-de-objetos" title="Deteccion de Objetos" subtitle="Faster/Mask R-CNN como proposer de regiones" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/vqa-antol-2015" title="VQA (2015)" subtitle="Antol et al. -- el paper fundacional de Visual Question Answering" icon="document-text" >}}
  {{< card link="/papers/vqav2-goyal-2017" title="VQAv2 (2017)" subtitle="Goyal et al. -- dataset balanceado, las slides 7-8 de la clase" icon="document-text" >}}
  {{< card link="/papers/pythia-jiang-2018" title="Pythia v0.1 (2018)" subtitle="Jiang et al. -- el modelo central de la clase, ganador VQA Challenge 2018" icon="document-text" >}}
  {{< card link="/papers/bleu-papineni-2002" title="BLEU (2002)" subtitle="Papineni et al. -- la metrica de evaluacion de captions" icon="document-text" >}}
{{< /cards >}}

## Papers canonicos (complementarios)

{{< cards >}}
  {{< card link="/papers/bottom-up-attention-anderson-2018" title="Bottom-Up/Top-Down (2018)" subtitle="Anderson et al. -- la base de Pythia, atencion sobre regiones" icon="document-text" >}}
  {{< card link="/papers/stacked-attention-yang-2016" title="Stacked Attention (2016)" subtitle="Yang et al. -- atencion visual multi-hop, antecedente de Pythia" icon="document-text" >}}
  {{< card link="/papers/mcb-fukui-2016" title="MCB (2016)" subtitle="Fukui et al. -- fusion bilineal compacta, VQA Challenge 2016" icon="document-text" >}}
  {{< card link="/papers/mutan-ben-younes-2017" title="MUTAN (2017)" subtitle="Ben-younes et al. -- fusion bilineal por descomposicion de Tucker" icon="document-text" >}}
  {{< card link="/papers/show-and-tell-vinyals-2015" title="Show and Tell (2015)" subtitle="Vinyals et al. -- captioning encoder-decoder CNN+LSTM" icon="document-text" >}}
  {{< card link="/papers/show-attend-tell-xu-2015" title="Show, Attend and Tell (2015)" subtitle="Xu et al. -- atencion visual en captioning" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/multimodal" title="Dominio: Multimodal" subtitle="Linea de tiempo: de captioning temprano (2014) a los VLMs frontier" icon="globe-alt" >}}
{{< /cards >}}
