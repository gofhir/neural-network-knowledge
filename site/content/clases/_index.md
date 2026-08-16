---
title: Clases
type: docs
weight: 20
sidebar:
  open: true
---

Apuntes y análisis de cada sesión del diplomado, en orden cronológico. Cada clase tiene su recorrido teórico (`teoria`), profundización matemática (`profundizacion`) y enlaces al laboratorio asociado.

## Bloque 1 — Fundamentos de Deep Learning (clases 5-10)

{{< cards >}}
  {{< card link="clase-05" title="Clase 05 - Redes Convolucionales" subtitle="LeNet, AlexNet, convoluciones, pooling y la jerarquía visual" icon="photograph" >}}
  {{< card link="clase-06" title="Clase 06 - Práctica" subtitle="Grafos de cómputo, activaciones (ReLU, sigmoid, tanh), inicialización Xavier/He" icon="variable" >}}
  {{< card link="clase-07" title="Clase 07 - Técnicas de Entrenamiento" subtitle="Forward/backward propagation, SGD, mini-batch, BatchNorm" icon="adjustments" >}}
  {{< card link="clase-08" title="Clase 08 - Funciones de Pérdida y Regularización" subtitle="Cross-entropy, MSE, L1/L2, dropout, early stopping" icon="adjustments" >}}
  {{< card link="clase-09" title="Clase 09 - CNNs en Profundidad" subtitle="VGG, Inception, ResNet, visualización e interpretabilidad" icon="eye" >}}
  {{< card link="clase-10" title="Clase 10 - Optimización y Learning Rate" subtitle="GD, SGD, momentum, Adam, schedulers, warmup" icon="trending-down" >}}
{{< /cards >}}

## Bloque 2 — Secuencias y atención (clases 11-14)

{{< cards >}}
  {{< card link="clase-11" title="Clase 11 - Redes Recurrentes (RNNs)" subtitle="RNN vanilla, BPTT, vanishing/exploding gradients, LSTM y GRU" icon="refresh" >}}
  {{< card link="clase-12" title="Clase 12 - Data Augmentation y Transfer Learning" subtitle="Augmentations clásicas, fine-tuning, feature extraction, BERT transfer" icon="chip" >}}
  {{< card link="clase-13" title="Clase 13 - Seq2Seq y Attention" subtitle="Encoder-decoder, Bahdanau attention, teacher forcing, NMT" icon="sparkles" >}}
  {{< card link="clase-14" title="Clase 14 - Transformers" subtitle="Self-attention, multi-head, positional encoding, BETO, interpretabilidad" icon="sparkles" >}}
{{< /cards >}}

## Bloque 3 — Visión avanzada (clases 15, 17)

{{< cards >}}
  {{< card link="clase-15" title="Clase 15 - Reconocimiento de Objetos" subtitle="R-CNN, Fast/Faster R-CNN, YOLO, FPN, Mask R-CNN, métricas mAP/IoU" icon="cube-transparent" >}}
  {{< card link="clase-17" title="Clase 17 - Pose Recognition" subtitle="OpenPose, PifPaf, BlazePose, dense pose, FaceNet, SMPL" icon="user-group" >}}
{{< /cards >}}

## Bloque 4 — NLP avanzado (clases 16, 18, 20)

{{< cards >}}
  {{< card link="clase-16" title="Clase 16 - Introducción a NLP" subtitle="Zipf, BoW, n-grams, TF-IDF, NLTK, spaCy, técnicas clásicas" icon="document-text" >}}
  {{< card link="clase-18" title="Clase 18 - Word2Vec, GloVe y Skip-Thought" subtitle="Embeddings distribuidos, modelos de lenguaje, regularidades lingüísticas" icon="document-text" >}}
  {{< card link="clase-20" title="Clase 20 - ELMo, BERT, GPT, ChatGPT" subtitle="Embeddings contextualizados, pretraining, fine-tuning, RLHF, in-context learning" icon="document-text" >}}
{{< /cards >}}

## Bloque 5 — Producción y MLOps (clase 19)

{{< cards >}}
  {{< card link="clase-19" title="Clase 19 - Entrenamiento, Deployment y MLOps" subtitle="GPUs, cloud, Docker, Vertex AI, serving, drift, retraining, BentoML" icon="cube-transparent" >}}
{{< /cards >}}

## Bloque 6 — Aplicaciones multimodales (clases 21-25)

{{< cards >}}
  {{< card link="clase-21" title="Clase 21 - Scene Text Recognition" subtitle="Detección y lectura de texto en imágenes: ABCNet, curvas de Bézier, CTC loss" icon="document-text" >}}
  {{< card link="clase-22" title="Clase 22 - Summarization" subtitle="Resumen extractivo y abstractivo: BertSum, T5, ROUGE, estrategias de decodificación" icon="document-text" >}}
  {{< card link="clase-23" title="Clase 23 - VQA e Image Captioning" subtitle="Preguntas sobre imágenes y generación de descripciones: Pythia, VQAv2, MCB, BLEU" icon="photograph" >}}
  {{< card link="clase-24" title="Clase 24 - Question Answering" subtitle="Comprensión lectora y QA: SQuAD, BiDAF, dense retrieval, métricas EM/F1" icon="document-text" >}}
  {{< card link="clase-25" title="Clase 25 - Recomendación con Imágenes y Texto" subtitle="Sistemas de recomendación multimodales: MF, BPR, two-tower, nDCG" icon="user-group" >}}
{{< /cards >}}

## Bloque 7 — Paradigmas de aprendizaje (clases 26-32)

{{< cards >}}
  {{< card link="clase-26" title="Clase 26 - Meta-aprendizaje" subtitle="Aprender a aprender: MAML, Prototypical Networks, few-shot, metric learning" icon="sparkles" >}}
  {{< card link="clase-27" title="Clase 27 - Redes Neuronales de Grafos" subtitle="GCN, GraphSAGE, GAT, message passing, expresividad" icon="cube-transparent" >}}
  {{< card link="clase-28" title="Clase 28 - Aprendizaje Autosupervisado" subtitle="Tareas pretexto, SimCLR, MoCo, MAE, UDA, aprendizaje contrastivo" icon="sparkles" >}}
  {{< card link="clase-29" title="Clase 29 - Modelos Generativos en Visión" subtitle="VAE, GAN, modelos de difusión, Stable Diffusion" icon="photograph" >}}
  {{< card link="clase-30" title="Clase 30 - Modelos con memoria externa" subtitle="Memory Networks, KV-MemNN, Neural Turing Machines, DNC" icon="cube-transparent" >}}
  {{< card link="clase-31" title="Clase 31 - Aprendizaje Reforzado" subtitle="Q-Learning, DQN, A3C, PPO, AlphaGo" icon="variable" >}}
  {{< card link="clase-32" title="Clase 32 - Olvido Catastrófico y Aprendizaje Continuo" subtitle="EWC, LwF, rehearsal, escenarios de continual learning" icon="refresh" >}}
{{< /cards >}}

## Bloque 8 — Decisión, razonamiento, audio y video (clases 33-39)

{{< cards >}}
  {{< card link="clase-33" title="Clase 33 - Aprendizaje por Imitación e IRL" subtitle="Behavioral cloning, DAgger, aprendizaje reforzado inverso" icon="variable" >}}
  {{< card link="clase-34" title="Clase 34 - Razonamiento" subtitle="Chain-of-thought, self-consistency, test-time compute, DeepSeek-R1" icon="sparkles" >}}
  {{< card link="clase-35" title="Clase 35 - Introducción al Análisis de Audio" subtitle="Fourier, FFT, muestreo, STFT, MFCC y escala Mel" icon="adjustments" >}}
  {{< card link="clase-36" title="Clase 36 - Introducción al Análisis de Video" subtitle="Tracking, reconocimiento de acciones, flujo óptico, datasets, enfoques de deep learning" icon="eye" >}}
  {{< card link="clase-37" title="Clase 37 - Datasets y Herramientas para Audio" subtitle="El ciclo de vida del dato de audio: formatos, transforms, augmentation, datasets" icon="adjustments" >}}
  {{< card link="clase-38" title="Clase 38 - CNN para reconocimiento en video" subtitle="Modelos pre-entrenados: temporal pooling, RNN, two-stream, C3D e I3D; el inflado de pesos 2D a 3D" icon="eye" >}}
  {{< card link="clase-39" title="Clase 39 - Modelos de Deep Learning para Audio" subtitle="CNN+RNN+MLP sobre log-mel, onda cruda y convoluciones dilatadas, y la auditoría de los Transformers en audio" icon="adjustments" >}}
  {{< card link="clase-40" title="Clase 40 - Analítica de Videos: Reconocimiento de acciones" subtitle="La ruta de la eficiencia: TSN y el muestreo por segmentos, TSM y el desplazamiento temporal a costo cero, y por qué Kinetics y Something-Something miden cosas distintas" icon="eye" >}}
{{< /cards >}}
