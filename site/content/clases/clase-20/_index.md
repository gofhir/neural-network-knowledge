---
title: "Clase 20 - ELMo, BERT, GPT, ChatGPT"
weight: 200
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga
**Fecha:** 2026-05-17

Cuarta clase del bloque de NLP. Recorre la familia de modelos pre-entrenados que dominaron la era post-Transformer: **ELMo** (Peters 2018) como puente entre embeddings estáticos y contextuales, **BERT** (Devlin 2018) como encoder bidireccional con MLM + NSP, la trayectoria **GPT-1 → GPT-2 → GPT-3** (Radford y Brown, 2018-2020) que validó el scaling decoder-only y el in-context learning, y **InstructGPT/ChatGPT** (Ouyang 2022) que introdujo RLHF como pieza de alignment. Cierra con el ecosistema de herramientas (Hugging Face Transformers, OpenAI API) y conecta al laboratorio práctico.

La clase se complementa con la [Clase 14 (Transformers)](/clases/clase-14) — que cubre la mecánica del encoder/decoder y atención — y con la [Clase 16 (Introducción a NLP)](/clases/clase-16) que entrega el contexto previo de embeddings clásicos, tokenización y representación de texto.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 64 diapositivas: ELMo, BERT, GPT-1/2/3, ChatGPT" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math detallado de biLM, MLM, atencion causal vs bidireccional, RLHF y Bradley-Terry" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="ELMo mini, BERT mini, GPT mini en PyTorch + TensorFlow + JAX, fine-tuning BETO, RLHF toy" icon="code" >}}
  {{< card link="/clases/clase-16" title="Clase anterior: Introduccion a NLP" subtitle="Tokenizacion, BoW, sentiment, Zipf, Heaps" icon="arrow-left" >}}
  {{< card link="/clases/clase-21" title="Clase siguiente: Scene Text Recognition" subtitle="STR pipeline, datasets, ABCNet con curvas Bezier" icon="arrow-right" >}}
  {{< card link="/clases/clase-14" title="Base: Transformers" subtitle="Self-attention, encoder, decoder, positional encoding" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/embeddings-contextualizados" title="Embeddings Contextualizados" subtitle="De word2vec a ELMo a BERT: el salto a representaciones dependientes del contexto" icon="book-open" >}}
  {{< card link="/fundamentos/bert" title="BERT (Encoder-only)" subtitle="Bidireccionalidad profunda, MLM, NSP, fine-tuning" icon="book-open" >}}
  {{< card link="/fundamentos/pretraining-bert" title="Pre-training BERT" subtitle="El paradigma pretrain + finetune como dominante en NLP" icon="book-open" >}}
  {{< card link="/fundamentos/gpt-family" title="Familia GPT (Decoder-only)" subtitle="GPT-1 a GPT-4: arquitectura, scaling laws, capacidades emergentes" icon="book-open" >}}
  {{< card link="/fundamentos/in-context-learning" title="In-Context Learning" subtitle="Zero-shot, one-shot y few-shot prompting sin gradientes" icon="book-open" >}}
  {{< card link="/fundamentos/rlhf" title="RLHF (Alignment)" subtitle="SFT + Reward Model + PPO: el pipeline detras de ChatGPT" icon="book-open" >}}
  {{< card link="/fundamentos/bpe" title="BPE / WordPiece (subword)" subtitle="Tokenizacion de BERT y GPT: algoritmos y trade-offs" icon="book-open" >}}
  {{< card link="/fundamentos/sft" title="Supervised Fine-Tuning" subtitle="El primer paso del alignment LLM" icon="book-open" >}}
  {{< card link="/fundamentos/dpo" title="DPO" subtitle="Alternativa a RLHF sin reward model explicito" icon="book-open" >}}
  {{< card link="/fundamentos/foundation-models" title="Foundation Models" subtitle="Emergencia y homogeneizacion en la era post-2020" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/elmo-peters-2018" title="ELMo (2018)" subtitle="Peters et al. -- Deep contextualized word representations, NAACL Best Paper" icon="document-text" >}}
  {{< card link="/papers/bert-devlin-2018" title="BERT (2018)" subtitle="Devlin et al. -- Pre-training bidireccional con MLM + NSP" icon="document-text" >}}
  {{< card link="/papers/gpt-1-radford-2018" title="GPT-1 (2018)" subtitle="Radford et al. -- Generative Pre-Training, decoder-only" icon="document-text" >}}
  {{< card link="/papers/gpt-2-radford-2019" title="GPT-2 (2019)" subtitle="Radford et al. -- Language Models are Unsupervised Multitask Learners" icon="document-text" >}}
  {{< card link="/papers/gpt-3-brown-2020" title="GPT-3 (2020)" subtitle="Brown et al. -- Language Models are Few-Shot Learners, 175B params" icon="document-text" >}}
  {{< card link="/papers/instructgpt-ouyang-2022" title="InstructGPT (2022)" subtitle="Ouyang et al. -- RLHF formalizado: precursor directo de ChatGPT" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/texto" title="Dominio: Texto / NLP" subtitle="Linea de tiempo completa: de Shannon 1948 a frontier LLMs 2025" icon="globe-alt" >}}
{{< /cards >}}
