---
title: "Clase 22 - Summarization"
weight: 220
sidebar:
  open: true
---

**Profesor:** Felipe del Río R.
**Fecha:** 2026-05-25

Sexta clase del bloque de visión avanzada y aplicaciones. Recorre el problema de **Text Summarization** — generar un resumen $y$ desde un texto fuente $x$ con $|y| < |x|$ preservando la información importante. La clase organiza el campo en 11 secciones: definición de la tarea, datasets canónicos, los dos paradigmas (**Extractive** vía BERTSum y **Abstractive** vía T5), text generation y decoding strategies (greedy, beam search, top-p, temperature), las métricas ROUGE family (R-1, R-2, R-L), y un cierre con **Prompt Engineering** para summarization con LLMs instruction-tuned.

La clase complementa la [Clase 14 (Transformers)](/clases/clase-14) con la arquitectura encoder-decoder, la [Clase 15 (Mecanismo de atención)](/clases/clase-15) con la cross-attention del decoder, la [Clase 16 (Introducción a NLP)](/clases/clase-16) con la motivación de NLP clásico, y la [Clase 20 (BERT/GPT/ChatGPT)](/clases/clase-20) con los modelos pretrained que potencian todo el pipeline de summarization moderno.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 67 diapositivas: task, datasets, BERTSum, T5, decoding, ROUGE, prompt engineering" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math detallada: T5 span-corruption, BERTSum oracle ROUGE, beam search, nucleus sampling, ROUGE family completa" icon="beaker" >}}
  {{< card link="/clases/clase-23" title="Clase siguiente: VQA e Image Captioning" subtitle="Pythia, VQAv2, beam search, BLEU" icon="arrow-right" >}}
  {{< card link="/clases/clase-21" title="Clase anterior: Scene Text Recognition" subtitle="ABCNet, curvas Bezier, BezierAlign" icon="arrow-left" >}}
  {{< card link="/clases/clase-20" title="Base: ELMo, BERT, GPT, ChatGPT" subtitle="Pretrained models que potencian summarization" icon="academic-cap" >}}
  {{< card link="/clases/clase-14" title="Base: Transformers" subtitle="Encoder-decoder, self-attention, cross-attention" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/text-summarization" title="Text Summarization" subtitle="Pipeline extractive/abstractive, datasets, metricas, evolucion historica" icon="book-open" >}}
  {{< card link="/fundamentos/t5-encoder-decoder" title="T5 y Encoder-Decoder" subtitle="Text-to-text framework, span-corruption, multi-task fine-tuning" icon="book-open" >}}
  {{< card link="/fundamentos/decoding-strategies" title="Decoding Strategies" subtitle="Greedy, beam search, top-k/p sampling, temperature" icon="book-open" >}}
  {{< card link="/fundamentos/rouge-metric" title="ROUGE Metric" subtitle="ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-W, ROUGE-S con math y ejemplos" icon="book-open" >}}
  {{< card link="/fundamentos/bert" title="BERT (Encoder-only)" subtitle="Base del Extractive Model (BERTSum)" icon="book-open" >}}
  {{< card link="/fundamentos/transformer" title="Transformer" subtitle="Arquitectura base de T5, BART, PEGASUS" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Mecanismo de Atencion" subtitle="Self-attention y cross-attention" icon="book-open" >}}
  {{< card link="/fundamentos/in-context-learning" title="In-Context Learning" subtitle="Prompt engineering con LLMs instruction-tuned" icon="book-open" >}}
  {{< card link="/fundamentos/pretraining-bert" title="Pre-training BERT" subtitle="MLM, NSP, paradigma pretrain+finetune" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/t5-raffel-2020" title="T5 (2020)" subtitle="Raffel et al. -- Text-to-Text Transfer Transformer, paper estrella abstractive" icon="document-text" >}}
  {{< card link="/papers/bertsum-liu-2019" title="BERTSum (2019)" subtitle="Yang Liu -- Fine-tune BERT for Extractive Summarization" icon="document-text" >}}
  {{< card link="/papers/nucleus-sampling-holtzman-2020" title="Nucleus Sampling (2020)" subtitle="Holtzman et al. -- The Curious Case of Neural Text Degeneration" icon="document-text" >}}
  {{< card link="/papers/rouge-lin-2004" title="ROUGE (2004)" subtitle="Lin -- la metrica de facto del campo summarization" icon="document-text" >}}
{{< /cards >}}

## Papers canonicos (complementarios)

{{< cards >}}
  {{< card link="/papers/bart-lewis-2020" title="BART (2020)" subtitle="Lewis et al. -- denoising autoencoder seq2seq, default HuggingFace summarization" icon="document-text" >}}
  {{< card link="/papers/pegasus-zhang-2020" title="PEGASUS (2020)" subtitle="Zhang et al. -- gap-sentence generation, SOTA en 12 benchmarks" icon="document-text" >}}
  {{< card link="/papers/xsum-narayan-2018" title="XSum (2018)" subtitle="Narayan et al. -- dataset extreme summarization BBC one-sentence" icon="document-text" >}}
  {{< card link="/papers/pointer-generator-see-2017" title="Pointer-Generator (2017)" subtitle="See et al. -- copy mechanism, first deep abstractive" icon="document-text" >}}
{{< /cards >}}

## Recursos del laboratorio

{{< cards >}}
  {{< card link="/laboratorios/lab-22" title="Laboratorio 22" subtitle="Practico de Summarization con notebooks Parte 1 + Parte 2" icon="academic-cap" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/texto" title="Dominio: Texto / NLP" subtitle="Linea de tiempo completa: de Shannon 1948 a frontier LLMs 2025" icon="globe-alt" >}}
{{< /cards >}}
