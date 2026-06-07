---
title: "Clase 24 - Question Answering"
weight: 240
sidebar:
  open: true
---

**Profesor:** Vladimir Araujo (Senior AI Researcher @ Sailplane AI)
**Fecha:** 2026-06-07

Clase dedicada a los **modelos de Question Answering (QA)** — sistemas que responden preguntas formuladas en lenguaje natural. La clase recorre QA como una de las tareas más antiguas del NLP (del Turing Test 1950 a los sistemas de tarjetas perforadas de los años 60), las cuatro grandes áreas del campo (**Information Retrieval QA**, **Reading Comprehension**, **Semantic Parsing** y **Visual QA**), y profundiza en **Machine Reading Comprehension (MRC)** con su evolución de arquitecturas: del **Stanford Attentive Reader** (atención unidireccional sobre el pasaje) a **BiDAF** (atención bidireccional sin resumen prematuro), de ahí a los modelos **Transformer-based** (BERT para span extraction) y a la **Generative QA** (GPT, BART, T5). Cierra con las métricas del campo: Accuracy, Mean Reciprocal Rank, Exact Match, F1 a nivel de token y BLEU.

La clase se apoya en la [Clase 14 (Transformers)](/clases/clase-14) y la [Clase 15 (Mecanismo de atención)](/clases/clase-15) para la maquinaria de atención, en la [Clase 20 (BERT/GPT/ChatGPT)](/clases/clase-20) para los modelos preentrenados que dominan QA hoy, y en la [Clase 22 (Summarization)](/clases/clase-22) para los modelos encoder-decoder (BART, T5) que también potencian la QA generativa. El área de **Visual QA** conecta con el [fundamento de VQA](/fundamentos/visual-question-answering).

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 43 diapositivas: historia, areas de QA, IR-based factoid, MRC, Stanford Attentive Reader, BiDAF, BERT, Generative QA, metricas" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math detallada: red neuronal generica para MRC, bilinear attention, BiDAF attention flow, span prediction de BERT, EM/F1 token-level, MRR, in-batch negatives de DPR" icon="beaker" >}}
  {{< card link="/clases/clase-22" title="Clase anterior: Summarization" subtitle="Encoder-decoder, BART, T5, decoding strategies" icon="arrow-left" >}}
  {{< card link="/clases/clase-20" title="Base: ELMo, BERT, GPT, ChatGPT" subtitle="Modelos preentrenados que dominan QA" icon="academic-cap" >}}
  {{< card link="/clases/clase-14" title="Base: Transformers" subtitle="Self-attention, encoder-decoder" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/question-answering" title="Question Answering" subtitle="Taxonomia, las 4 areas, pipeline IR-based factoid, datasets, evolucion historica" icon="book-open" >}}
  {{< card link="/fundamentos/machine-reading-comprehension" title="Machine Reading Comprehension" subtitle="P+Q->A, red generica MRC, attentive readers, BiDAF, span prediction" icon="book-open" >}}
  {{< card link="/fundamentos/qa-evaluation-metrics" title="Metricas de QA" subtitle="Exact Match, token-level F1, MRR, BLEU, accuracy, abstencion" icon="book-open" >}}
  {{< card link="/fundamentos/dense-retrieval" title="Dense Retrieval y Open-Domain QA" subtitle="Bi-encoder, DPR, MIPS/FAISS, retriever-reader, RAG" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Mecanismo de Atencion" subtitle="Base de los attentive readers y BiDAF" icon="book-open" >}}
  {{< card link="/fundamentos/bert" title="BERT (Encoder-only)" subtitle="Base de la QA extractiva moderna" icon="book-open" >}}
  {{< card link="/fundamentos/t5-encoder-decoder" title="T5 y Encoder-Decoder" subtitle="Base de la QA generativa (BART, T5)" icon="book-open" >}}
  {{< card link="/fundamentos/visual-question-answering" title="Visual Question Answering" subtitle="El area de Visual QA mencionada en la clase" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/cnn-dailymail-hermann-2015" title="CNN/Daily Mail (2015)" subtitle="Hermann et al. -- Teaching Machines to Read and Comprehend, dataset cloze + Attentive Reader" icon="document-text" >}}
  {{< card link="/papers/stanford-attentive-reader-chen-2016" title="Stanford Attentive Reader (2016)" subtitle="Chen et al. -- modelo bilinear + critica del dataset CNN/DM" icon="document-text" >}}
  {{< card link="/papers/squad-rajpurkar-2016" title="SQuAD (2016)" subtitle="Rajpurkar et al. -- 100k+ preguntas, extractive QA span-based, EM/F1" icon="document-text" >}}
  {{< card link="/papers/bidaf-seo-2017" title="BiDAF (2017)" subtitle="Seo et al. -- Bidirectional Attention Flow para machine comprehension" icon="document-text" >}}
{{< /cards >}}

## Papers canonicos (complementarios)

{{< cards >}}
  {{< card link="/papers/squad2-rajpurkar-2018" title="SQuAD 2.0 (2018)" subtitle="Rajpurkar et al. -- unanswerable questions, abstencion, Know What You Don't Know" icon="document-text" >}}
  {{< card link="/papers/ms-marco-nguyen-2016" title="MS MARCO (2016)" subtitle="Nguyen et al. -- preguntas reales de Bing, respuestas generadas, QA generativo" icon="document-text" >}}
  {{< card link="/papers/dpr-karpukhin-2020" title="DPR (2020)" subtitle="Karpukhin et al. -- Dense Passage Retrieval, bi-encoder para open-domain QA" icon="document-text" >}}
  {{< card link="/papers/babi-weston-2015" title="bAbI (2015)" subtitle="Weston et al. -- 20 toy tasks, QA sintetico AI-complete" icon="document-text" >}}
  {{< card link="/papers/childrens-book-test-hill-2016" title="Children's Book Test (2016)" subtitle="Hill et al. -- cloze por tipo de palabra, Goldilocks Principle" icon="document-text" >}}
  {{< card link="/papers/lambada-paperno-2016" title="LAMBADA (2016)" subtitle="Paperno et al. -- prediccion de palabra con contexto amplio de discurso" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/texto" title="Dominio: Texto / NLP" subtitle="Linea de tiempo completa: de Shannon 1948 a frontier LLMs 2025" icon="globe-alt" >}}
{{< /cards >}}
