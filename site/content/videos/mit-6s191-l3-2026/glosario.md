---
title: "Glosario — MIT 6.S191 (2026) L3"
weight: 30
sidebar:
  open: true
---

> Términos clave del lecture **MIT 6.S191 (2026) L3 — Deep Computer Vision**. Cada entrada es bilingüe (español + inglés estándar), agrupada por tema. Para definiciones más extensas o derivaciones formales, ver [`profundizacion.md`](/videos/mit-6s191-l3-2026/profundizacion/) o el material UC enlazado.

---

## Convolución y arquitectura básica

**Convolución (convolution)** — Operación que desliza un filtro pequeño sobre un input multidimensional, calculando producto elemento-a-elemento más suma en cada posición. En CNNs, los pesos del filtro se aprenden por gradiente.

**Filtro / kernel (filter, kernel)** — Matriz de pesos $k \times k$ que la convolución aplica en cada parche del input. Una capa convolucional típicamente tiene decenas o cientos de filtros distintos.

**Feature map (feature map, activation map)** — Salida 2D de aplicar un filtro a todas las posiciones del input. Una capa con $d$ filtros produce $d$ feature maps apilados.

**Stride (stride)** — Cuántos píxeles avanza el filtro entre aplicaciones sucesivas. Stride 1 = denso, stride 2 = downsample por 2 en cada dimensión.

**Padding (padding)** — Añadir píxeles (típicamente ceros) al borde del input para controlar el tamaño de salida. *Same padding* preserva las dimensiones; *valid padding* no añade nada.

**Receptive field (receptive field)** — Región del input original que afecta a un nodo dado en una capa profunda. Crece linealmente al apilar convs $3 \times 3$ con stride 1; exponencialmente con stride > 1 o convs dilatadas.

**Weight sharing (weight sharing, parameter sharing)** — Reusar el mismo filtro en todas las posiciones espaciales del input. Reduce drásticamente el número de parámetros e induce invarianza a traslación.

**Channel / depth (channel, depth)** — Número de feature maps en un volumen. Una imagen RGB tiene 3 canales; una capa con 64 filtros produce un volumen de salida con depth 64.

**Capa convolucional (convolutional layer)** — Capa de una red que aplica un conjunto de filtros aprendibles + bias + activación no-lineal. En código: `tf.keras.layers.Conv2D`, `torch.nn.Conv2d`.

**1×1 convolution (pointwise convolution)** — Convolución con kernel $1 \times 1$. No agrega información espacial pero proyecta entre canales — usada para reducir/expandir profundidad sin costo computacional.

---

## Pooling y normalización

**Max pooling (max pooling)** — Downsampling que reemplaza cada parche $k \times k$ del feature map por su valor máximo. La opción canónica es $2 \times 2$ con stride 2.

**Average pooling (average pooling)** — Reemplaza cada parche por el promedio. Más suave que max pool; usado a veces en arquitecturas modernas.

**Global average pooling (global average pooling, GAP)** — Promedio sobre todo el feature map; convierte $h \times w \times d$ en $d$ escalares. Reemplaza las capas FC al final del backbone en arquitecturas modernas (Inception, ResNet).

**Batch normalization (batch normalization, BN)** — Normaliza activaciones por mini-batch a media 0 / varianza 1, luego aplica scale-shift aprendibles. Estabiliza el entrenamiento y permite learning rates más altos. Ver `profundizacion.md` §2.6.

**Layer normalization (layer normalization, LN)** — Normaliza por feature en vez de por batch. Dominante en transformers; útil cuando el batch es chico.

**ReLU (Rectified Linear Unit)** — Activación no-lineal $g(z) = \max(0, z)$. Estándar en CNNs por su simplicidad y resistencia a vanishing gradient.

---

## Arquitecturas clásicas

**LeNet-5 (LeCun et al. 1998)** — Primera CNN industrialmente útil; reconocimiento de dígitos. Establece el patrón `Conv → Pool → Conv → Pool → FC`.

**AlexNet (Krizhevsky et al. 2012)** — La CNN que ganó ImageNet 2012 reduciendo el error top-5 al 15.3%. ReLU + dropout + GPU en paralelo + augmentation agresiva.

**VGG (Simonyan & Zisserman 2014)** — Arquitectura uniforme de 16/19 capas con solo filtros $3 \times 3$. Insight: dos $3 \times 3$ apiladas tienen el mismo receptive field que una $5 \times 5$ con menos parámetros.

**Inception / GoogLeNet (Szegedy et al. 2015)** — Módulos que aplican varias convs con kernels distintos en paralelo y concatenan las salidas. Usa $1 \times 1$ convs como bottleneck.

**ResNet (He et al. 2015)** — Bloques residuales con skip connections de identidad. Permite entrenar redes de 50, 101, 152 capas evitando degradation. Ver `profundizacion.md` §2.5.

**Skip connection / residual connection (skip / residual connection)** — Conexión que suma la entrada de un bloque directamente a su salida: $y = \mathcal{F}(x) + x$. Garantiza flujo de gradiente y permite redes muy profundas.

**Backbone (backbone)** — La parte convolucional de una red, sin la cabecera específica de tarea. Reutilizable: el mismo backbone preentrenado en ImageNet se usa para clasificación, detección, segmentación, etc.

---

## Detección de objetos

**Bounding box (bounding box, bbox)** — Caja rectangular que encuadra un objeto, expresada como $(x, y, w, h)$ o $(x_1, y_1, x_2, y_2)$.

**IoU (Intersection over Union)** — Métrica de overlap entre dos bounding boxes: $|A \cap B| / |A \cup B|$. Vale 0 si no se tocan, 1 si son idénticas.

**NMS (Non-Maximum Suppression)** — Algoritmo que depura múltiples cajas predichas para el mismo objeto, quedándose con la de mayor score y eliminando las que tienen IoU > umbral con ella.

**Anchor box (anchor box, prior)** — Caja prefabricada con shape predefinido. Detectores como Faster R-CNN/YOLO predicen offsets respecto a anchors en vez de coordenadas absolutas.

**Region proposal (region proposal)** — Caja candidata que podría contener un objeto. Selectiva search en R-CNN, aprendida en Faster R-CNN (RPN).

**RPN (Region Proposal Network)** — Sub-red de Faster R-CNN que aprende a proponer regiones candidatas, eliminando la necesidad de selective search.

**R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN)** — Linaje de detectores de dos etapas. Cada versión mejora velocidad o capacidad. Ver `profundizacion.md` §2.7.

**YOLO (You Only Look Once)** — Detector single-shot que predice clases + bboxes en una sola pasada de CNN sobre la imagen. Real-time (>30 FPS).

**mAP (mean Average Precision)** — Métrica estándar de detección. Promedia precisión sobre múltiples umbrales IoU y clases. En COCO se evalúa sobre IoU $\in \{0.5, 0.55, \dots, 0.95\}$.

---

## Segmentación

**Segmentación semántica (semantic segmentation)** — Asigna una etiqueta de clase a cada píxel de la imagen. No distingue entre instancias de la misma clase.

**Segmentación de instancia (instance segmentation)** — Asigna etiqueta de clase + ID de instancia a cada píxel. Mask R-CNN es el estándar.

**FCN (Fully Convolutional Network)** — Red sin capas FC; encoder convolucional + decoder convolucional. Permite output 2D por píxel. Long et al. 2015.

**Transposed convolution (transposed convolution, deconvolution)** — Operación de upsampling que es la transpuesta matricial de la convolución. No es la inversa — el nombre "deconvolución" es engañoso. Ver `profundizacion.md` §3.2.

**U-Net (Ronneberger et al. 2015)** — Encoder-decoder con skip connections densas que copian features del encoder al decoder. Estándar en segmentación médica y backbone de modelos de difusión.

**Encoder-decoder (encoder-decoder)** — Patrón arquitectural: encoder reduce resolución comprimiendo información, decoder la recupera produciendo output 2D.

---

## Aplicaciones

**Clasificación (classification)** — Asignar una clase (o distribución de probabilidad sobre clases) a una imagen completa.

**Regresión visual (visual regression)** — Predecir valores continuos a partir de una imagen (e.g., posición, ángulo, edad).

**End-to-end learning (end-to-end learning)** — Entrenar un único modelo que mapea inputs crudos directamente a outputs finales, sin etapas intermedias hechas a mano. Ejemplo: cámara → comando de control en conducción autónoma.

**Control probabilístico (probabilistic control)** — En vez de predecir un comando determinista, predecir parámetros de una distribución (e.g., $\mu, \sigma$). Permite expresar incertidumbre.

**Cabecera (head, task head)** — Capas finales de la red específicas a la tarea. El backbone se reutiliza; la cabecera cambia entre clasificación, detección, segmentación.

---

## Entrenamiento y transferencia

**Softmax (softmax)** — Función que convierte logits en una distribución de probabilidad: $\text{softmax}(y_i) = e^{y_i} / \sum_j e^{y_j}$. Estándar al final de un clasificador.

**Cross-entropy loss (cross-entropy loss)** — Pérdida estándar para clasificación: $\mathcal{L} = -\sum_i y_i \log \hat{y}_i$ con $y$ one-hot y $\hat{y}$ softmax.

**Transfer learning (transfer learning)** — Reutilizar pesos preentrenados (típicamente en ImageNet) para una tarea nueva. Ver [`fundamentos/transfer-learning`](/fundamentos/transfer-learning/).

**Feature extraction (feature extraction)** — Estrategia de transfer learning donde se congela el backbone y solo se entrena la cabecera nueva.

**Fine-tuning (fine-tuning)** — Estrategia donde se descongela todo o parte del backbone y se entrena con learning rate bajo.

**Freezing (freezing)** — Marcar parámetros como no-entrenables durante el fine-tuning para preservar el preentrenamiento.

**Data augmentation (data augmentation)** — Aplicar transformaciones aleatorias (crop, flip, rotación, color jitter, Mixup, CutMix) a las imágenes de entrenamiento para regularizar y aumentar diversidad efectiva. Ver [`fundamentos/data-augmentation`](/fundamentos/data-augmentation/).

**ImageNet pretraining (ImageNet pretraining)** — Preentrenar un backbone en ImageNet (1000 clases, 1.2M imágenes) y luego transferir. Receta dominante 2012-2020; reemplazada parcialmente por self-supervised pretraining.

---

## Interpretabilidad

**Feature visualization (feature visualization)** — Generar imágenes que maximicen la activación de un filtro específico para entender qué detecta. Por gradient ascent sobre el input. Ver [`fundamentos/interpretabilidad`](/fundamentos/interpretabilidad/).

**Saliency map (saliency map)** — Mapa que muestra qué píxeles del input fueron más relevantes para una predicción dada. Vanilla saliency = $|\partial y_c / \partial x|$.

**Grad-CAM (Gradient-weighted Class Activation Mapping)** — Heatmap de localización clase-específica calculado a partir de los gradientes en la última capa convolucional. Ver `profundizacion.md` §2.11.

**Attribution (attribution)** — Familia de métodos que asignan importancia a cada píxel/feature respecto a una predicción. Incluye saliency, integrated gradients, Grad-CAM, occlusion sensitivity.

---

## Atribución

> Material adaptado de **MIT 6.S191 (2026) Lecture 3: Deep Computer Vision**, Alexander Amini, 6 de enero de 2026.
> [Video](https://www.youtube.com/watch?v=pqIcoskUuWs) — [Slides oficiales](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L3.pdf) — [Sitio del curso](https://introtodeeplearning.com/).
> Notas en español como elaboración independiente. Sin afiliación oficial con MIT.
