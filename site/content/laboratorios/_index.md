---
title: "Laboratorios"
weight: 60
sidebar:
  open: true
---

Exploracion en profundidad de los laboratorios del diplomado: librerias, conceptos, codigo paso a paso y ejercicios.

{{< cards >}}
  {{< card link="lab-05" title="Lab 05 - AlexNet y CNNs" subtitle="Clasificacion de imagenes con redes convolucionales" icon="photograph" >}}
  {{< card link="lab-06" title="Lab 06 - Grafos de Computo, Activaciones e Inicializacion" subtitle="Forward/backward, funciones de activacion y Xavier/He" icon="variable" >}}
  {{< card link="lab-07" title="Lab 07 - PyTorch" subtitle="Tensores, modulos, entrenamiento y DataLoaders" icon="code" >}}
  {{< card link="lab-08" title="Lab 08 - Entrenamiento Avanzado" subtitle="Funciones de perdida, regularizacion y tareas auxiliares" icon="adjustments" >}}
  {{< card link="lab-09" title="Lab 09 - Visualizacion e Interpretabilidad" subtitle="Feature Visualization y Attribution en CNNs" icon="eye" >}}
  {{< card link="lab-10" title="Lab 10 - Optimizacion y Learning Rate" subtitle="GD, SGD, Adam, schedulers en CIFAR10" icon="trending-down" >}}
  {{< card link="lab-11" title="Lab 11 - Redes Recurrentes (RNNs)" subtitle="RNN vanilla, LSTM y BiLSTM para clasificar nacionalidades" icon="refresh" >}}
  {{< card link="lab-12" title="Lab 12 - Data Augmentation, Transfer Learning y Finetuning" subtitle="ResNet18 sobre flowers y BERT sobre Jigsaw toxic comments" icon="chip" >}}
  {{< card link="lab-13" title="Lab 13 - Seq2Seq y Mecanismos de Atencion" subtitle="Translation con encoder-decoder, Bahdanau attention y teacher forcing" icon="sparkles" >}}
  {{< card link="lab-14" title="Lab 14 - Transformers e Interpretabilidad + CLIP" subtitle="BETO + bertviz (Parte 1) y CLIP zero-shot sobre Food101/Cars (Parte 2)" icon="eye" >}}
  {{< card link="lab-15" title="Lab 15 - Faster R-CNN: Inferencia COCO + Fine-tuning Raccoon" subtitle="Detector pre-entrenado en COCO 2017 y fine-tuning para detectar mapaches" icon="cube-transparent" >}}
  {{< card link="lab-16" title="Lab 16 - Introducción a NLP: NLTK + spaCy + NLLB + VADER + Bag of Words" subtitle="Pipeline clásico: tokenización, normalización, estadísticas (Zipf/Heaps), spaCy, BoW + LogReg, VADER, traducción con NLLB-200" icon="document-text" >}}
  {{< card link="lab-17" title="Lab 17 - Pose Recognition: comparación de modelos + clasificación de acciones" subtitle="OpenPifPaf vs OpenPose como feature extractors + MLP para clasificar acciones de Stanford 40" icon="user-group" >}}
  {{< card link="lab-18" title="Lab 18 - Word Embeddings: analogías, doesnt_match, PCA y sentiment analysis" subtitle="Google News Word2Vec sobre Sentiment140 — 3CosMul, polisemia, suma vs promedio" icon="academic-cap" >}}
  {{< card link="lab-19" title="Lab 19 - Entrenamiento, Deployment y MLOps con BentoML" subtitle="Caso Space Z + servidor BentoML + benchmark latencia/concurrencia + compresión JPEG (throughput 1.2 req/s saturado por JSON)" icon="cube-transparent" >}}
  {{< card link="lab-21" title="Lab 21 - Scene Text Recognition: ABCNet end-to-end" subtitle="Disección de la salida (beziers + recs + charset) + OCR de marcas en alemán (transfer zero-shot) + minería geoespacial sobre Street View" icon="photograph" >}}
  {{< card link="lab-22" title="Lab 22 - Summarization: Extractivo (BertSum) y Abstractivo (T5)" subtitle="Selección de oraciones vs generación, trigram blocking, decodificación, ROUGE real sobre CNN/DailyMail" icon="document-text" >}}
  {{< card link="lab-23" title="Lab 23 - VQA e Image Captioning con BLIP" subtitle="Vision-language: preguntas visuales y descripción de imágenes con BLIP" icon="photograph" >}}
  {{< card link="lab-24" title="Lab 24 - Question Answering: Extractivo (BERT) y Generativo (T5/BART)" subtitle="QA extractivo vs generativo en español, BETO/SQuAD-es vs T5/SQAC" icon="document-text" >}}
  {{< card link="lab-25" title="Lab 25 - Recomendación multimodal con imágenes y texto" subtitle="Content-based multimodal estilo Pinterest: AlexNet fc7 + BERT, proxy task, nDCG" icon="academic-cap" >}}
  {{< card link="lab-26" title="Lab 26 - Meta-aprendizaje: MAML y Prototypical Networks" subtitle="Optimization-based vs metric-based sobre Omniglot y Mini-ImageNet, ejes WAYS/SHOTS, las 7 actividades" icon="academic-cap" >}}
  {{< card link="lab-27" title="Lab 27 - Redes Neuronales de Grafos con PyTorch Geometric" subtitle="Clasificación de nodos (Cora) y de grafos (MUTAG): MLP vs GCN, agregación sum vs mean, las 6 actividades" icon="variable" >}}
  {{< card link="lab-28" title="Lab 28 - Aprendizaje Autosupervisado: UDA" subtitle="Semi-supervisión por consistencia sobre IMDB con 20 etiquetas: back-translation, TSA, los tres regímenes y las actividades" icon="adjustments" >}}
  {{< card link="lab-29" title="Lab 29 - Modelos Generativos en Visión: Stable Diffusion" subtitle="Manipular SDXL/SD1.5 con diffusers: pasos, schedulers, guidance, Img2Img, Inpainting, ControlNet + cuestionario (trilemma, difusión latente)" icon="photograph" >}}
  {{< card link="lab-30" title="Lab 30 - Modelos con memoria externa: Key-Value Memory Networks" subtitle="KV-MemNN sobre WikiMovies QA: KB key/value desde texto, blocking, 2 hops de atención, 5 experimentos propios y las 4 actividades" icon="cube-transparent" >}}
  {{< card link="lab-31" title="Lab 31 - Aprendizaje Reforzado: DQN sobre CartPole" subtitle="Deep Q-Network desde cero con experience replay y ε-greedy: resuelve CartPole en 85 episodios, 4 ablations propias (velocidades/replay/target/hiperparámetros) y las 2 preguntas de la tarea" icon="variable" >}}
  {{< card link="lab-32" title="Lab 32 - Aprendizaje Incremental y Olvido Catastrófico" subtitle="Permuted MNIST: mide el olvido (T0 de 96% a 34%) y compara Naive vs Rehearsal (70%) vs EWC (Fisher). Curva de λ, trade-off buffer↔memoria y las 4 actividades. El último lab del curso" icon="trending-down" >}}
  {{< card link="lab-33" title="Lab 33 - Aprendizaje por Imitación y DAGGER" subtitle="Imitar un experto DQN en Breakout (Atari): Behaviour Cloning puro fracasa (score 0), DAGGER lo rescata (0→5), experto=10. Covariate shift, la diferencia de una línea BC vs DAGGER y las 5 preguntas de la tarea" icon="variable" >}}
  {{< card link="lab-34" title="Lab 34 - Razonamiento: tool use, LoRA y optimización de prompt" subtitle="Tres palancas para mejorar un LLM (herramientas, fine-tuning LoRA, GEPA) sobre traducción a Lakota. El modelo base se paraliza (0 tool calls), LoRA lo rescata (0→7), + multimodal Qwen3-VL con boleta→JSON y el error del separador de miles" icon="sparkles" >}}
  {{< card link="lab-35" title="Lab 35 - Introducción al Análisis de Audio" subtitle="Fundamentos DSP: FFT, series de Fourier (cuadrada/triangular/sierra), STFT y espectrogramas, MFCC. El hola-mundo del audio y de dónde salen los tensores para Whisper/wav2vec2. Abre el módulo de Audio" icon="variable" >}}
  {{< card link="lab-36" title="Lab 36 - Introducción al Análisis de Video" subtitle="Clasificación de acciones en UCF11: ResNet-34 + average temporal pooling (bag of frames). El pooling pierde el orden temporal; hallazgo contraintuitivo: 4 frames (85.9%) ≥ 8 frames (84.6%) en la mitad del tiempo, probando que el modelo ignora el tiempo" icon="variable" >}}
{{< /cards >}}
