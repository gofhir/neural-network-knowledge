---
title: Fundamentos
type: docs
weight: 10
sidebar:
  open: true
---

Teoría de deep learning organizada por tema, en orden lógico de aprendizaje. El contenido está curado y complementado con investigación adicional más allá de lo cubierto en las clases.

A diferencia de **[Clases](/clases)** (organizado cronológicamente por sesión) y **[Dominios](/dominios)** (organizado por modalidad de datos), aquí el eje es el **concepto teórico** — una referencia transversal reutilizable que se cita desde múltiples clases.

## Matemática y entrenamiento básico

{{< cards >}}
  {{< card link="historia-matematica" title="Historia matemática" subtitle="Linaje desde McCulloch-Pitts hasta backprop y attention" icon="book-open" >}}
  {{< card link="arquitectura-redes" title="Arquitectura de Redes Neuronales" subtitle="Capas, dimensiones, conexiones, profundidad" icon="variable" >}}
  {{< card link="backpropagation" title="Backpropagation" subtitle="Regla de la cadena en grafos de cómputo" icon="adjustments" >}}
  {{< card link="funciones-perdida" title="Funciones de Pérdida" subtitle="Cross-entropy, MSE, hinge, contrastivas" icon="adjustments" >}}
  {{< card link="optimizadores" title="Optimizadores" subtitle="SGD, momentum, RMSProp, Adam, AdamW" icon="trending-down" >}}
  {{< card link="learning-rate" title="Learning Rate" subtitle="Schedulers, warmup, cyclic, cosine, OneCycle" icon="trending-down" >}}
  {{< card link="regularizacion" title="Regularización" subtitle="L1/L2, dropout, batch norm, early stopping" icon="adjustments" >}}
{{< /cards >}}

## Representación y embeddings

{{< cards >}}
  {{< card link="representacion-datos" title="Representación de Datos" subtitle="One-hot, dense, sparse, espacios vectoriales" icon="document-text" >}}
  {{< card link="bag-of-words" title="Bag of Words" subtitle="Representación vectorial clásica, n-grams, TF-IDF" icon="document-text" >}}
  {{< card link="tokenizacion-clasica" title="Tokenización clásica" subtitle="Stemming, lemmatización, stopwords, Punkt" icon="document-text" >}}
  {{< card link="bpe" title="BPE (Byte Pair Encoding)" subtitle="Subword tokenization, WordPiece, SentencePiece" icon="document-text" >}}
  {{< card link="word2vec" title="Word2Vec" subtitle="Skip-gram, CBOW, negative sampling" icon="document-text" >}}
  {{< card link="glove" title="GloVe" subtitle="Factorización de matriz de co-ocurrencias" icon="document-text" >}}
  {{< card link="embeddings-distribuidos" title="Embeddings distribuidos" subtitle="Geometría semántica, analogías, distance metrics" icon="document-text" >}}
  {{< card link="embeddings-contextualizados" title="Embeddings contextualizados" subtitle="ELMo, BERT, contextual vs static embeddings" icon="document-text" >}}
  {{< card link="positional-encoding" title="Positional Encoding" subtitle="Sinusoidal, aprendible, RoPE, ALiBi" icon="variable" >}}
{{< /cards >}}

## Arquitecturas

{{< cards >}}
  {{< card link="redes-convolucionales" title="Redes Convolucionales (CNN)" subtitle="Convolución, pooling, AlexNet, VGG, ResNet, Inception" icon="photograph" >}}
  {{< card link="redes-recurrentes" title="Redes Recurrentes (RNN)" subtitle="RNN vanilla, vanishing gradients, BPTT" icon="refresh" >}}
  {{< card link="lstm-gru" title="LSTM y GRU" subtitle="Gates, cell state, forget/input/output gates" icon="refresh" >}}
  {{< card link="backpropagation-through-time" title="Backpropagation Through Time" subtitle="Unrolling, truncated BPTT, gradient flow en RNNs" icon="refresh" >}}
  {{< card link="seq2seq" title="Sequence to Sequence" subtitle="Encoder-decoder, teacher forcing, NMT" icon="sparkles" >}}
  {{< card link="mecanismo-atencion" title="Mecanismo de Atención" subtitle="Bahdanau attention, soft/hard, alineación" icon="sparkles" >}}
  {{< card link="self-attention" title="Self-Attention" subtitle="Q/K/V, multi-head, masked attention" icon="sparkles" >}}
  {{< card link="transformer" title="Arquitectura Transformer" subtitle="Encoder-decoder con self-attention, layer norm, residual" icon="cube-transparent" >}}
  {{< card link="vision-transformer" title="Vision Transformer (ViT)" subtitle="Patches como tokens, transformer aplicado a visión" icon="photograph" >}}
{{< /cards >}}

## Modelos y familias

{{< cards >}}
  {{< card link="modelos-de-lenguaje" title="Modelos de Lenguaje" subtitle="NPLM, RNN-LM, perplexity, scaling laws" icon="document-text" >}}
  {{< card link="skip-thought" title="Skip-Thought y Sentence Embeddings" subtitle="Vectores de oración, transfer learning para NLP" icon="document-text" >}}
  {{< card link="bert" title="BERT" subtitle="Bidirectional encoder, masked LM, NSP" icon="document-text" >}}
  {{< card link="pretraining-bert" title="Pre-training BERT" subtitle="MLM objective, WordPiece, NSP/SOP" icon="document-text" >}}
  {{< card link="gpt-family" title="Familia GPT (decoder-only)" subtitle="GPT-1/2/3/4, autoregressive LM, scaling" icon="document-text" >}}
  {{< card link="foundation-models" title="Foundation Models" subtitle="Pretraining masivo, transfer, capacidades emergentes" icon="academic-cap" >}}
{{< /cards >}}

## Entrenamiento, fine-tuning y alineamiento

{{< cards >}}
  {{< card link="transfer-learning" title="Transfer Learning y Fine-Tuning" subtitle="Pretraining + fine-tuning, freezing, LoRA" icon="chip" >}}
  {{< card link="data-augmentation" title="Data Augmentation" subtitle="Augmentations clásicas, mixup, cutmix, autoaugment" icon="chip" >}}
  {{< card link="tareas-auxiliares" title="Aprendizaje Multitarea y Tareas Auxiliares" subtitle="Multi-task learning, gradient surgery, weight sharing" icon="chip" >}}
  {{< card link="loss-masking" title="Loss Masking en SFT" subtitle="Pad tokens, instruction tuning, attention masks" icon="adjustments" >}}
  {{< card link="sft" title="Supervised Fine-Tuning (SFT)" subtitle="Instruction tuning, prompt format, datasets de calidad" icon="adjustments" >}}
  {{< card link="rlhf" title="RLHF" subtitle="Reward model, PPO, human preference data" icon="adjustments" >}}
  {{< card link="dpo" title="Direct Preference Optimization (DPO)" subtitle="Alternativa a RLHF, sin reward model explícito" icon="adjustments" >}}
  {{< card link="bradley-terry" title="Modelo de Bradley-Terry" subtitle="Modelo probabilístico de preferencias pareadas" icon="variable" >}}
  {{< card link="kl-implicito" title="KL Implícito (en DPO)" subtitle="Derivación matemática del término KL en DPO" icon="variable" >}}
  {{< card link="in-context-learning" title="In-Context Learning" subtitle="Few-shot, zero-shot, chain-of-thought" icon="academic-cap" >}}
{{< /cards >}}

## Visión y tareas específicas

{{< cards >}}
  {{< card link="deteccion-de-objetos" title="Detección de Objetos" subtitle="IoU, NMS, anchors, RPN, RoIAlign, FPN, mAP" icon="cube-transparent" >}}
  {{< card link="pose-estimation" title="Pose Estimation 2D" subtitle="Bottom-up vs top-down, heatmaps, OKS/PCK" icon="user-group" >}}
  {{< card link="dense-correspondence" title="Dense Correspondence y UV Mapping" subtitle="DensePose, SMPL, mapping cuerpo→superficie" icon="user-group" >}}
  {{< card link="sentiment-analysis" title="Sentiment Analysis" subtitle="Rule-based vs neural, VADER, translate-then-analyze" icon="document-text" >}}
  {{< card link="aprendizaje-contrastivo" title="Aprendizaje Contrastivo (CLIP)" subtitle="InfoNCE, dual encoders, alineación cross-modal" icon="academic-cap" >}}
  {{< card link="triplet-loss" title="Triplet Loss y Metric Learning" subtitle="Anchor/positive/negative, FaceNet, hard mining" icon="academic-cap" >}}
{{< /cards >}}

## Interpretabilidad

{{< cards >}}
  {{< card link="interpretabilidad" title="Interpretabilidad de Redes Neuronales" subtitle="Feature visualization, attribution, saliency maps" icon="eye" >}}
  {{< card link="interpretabilidad-mecanicista" title="Interpretabilidad mecanicista" subtitle="Circuits, induction heads, sparse autoencoders" icon="eye" >}}
{{< /cards >}}

## Producción y MLOps

{{< cards >}}
  {{< card link="mlops" title="MLOps" subtitle="9 principios, 9 componentes, drift, retraining" icon="cube-transparent" >}}
  {{< card link="model-serving" title="Model Serving" subtitle="BentoML, Triton, TorchServe, ONNX, batching, cuantización" icon="cube-transparent" >}}
  {{< card link="cloud-computing" title="Cloud Computing" subtitle="IaaS/PaaS/FaaS, GCP/AWS/Azure, spot instances, pricing" icon="cube-transparent" >}}
  {{< card link="docker-containers" title="Docker y Containers" subtitle="Dockerfile, layers, registries, NVIDIA Container Toolkit" icon="cube-transparent" >}}
  {{< card link="gpu-hardware-ml" title="GPU Hardware para ML" subtitle="CUDA, Tensor cores, FP16/BF16/FP8, generaciones Pascal-Blackwell" icon="chip" >}}
{{< /cards >}}
