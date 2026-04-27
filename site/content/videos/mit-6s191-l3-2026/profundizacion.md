---
title: "Profundización — MIT 6.S191 (2026) L3"
weight: 20
math: true
sidebar:
  open: true
---

> Investigación complementaria al lecture **MIT 6.S191 (2026) L3 — Deep Computer Vision** (Alexander Amini). Reúne los papers seminales del canon de CNNs, derivaciones que la clase no formaliza, y conceptos que MIT no cubre pero vale la pena conocer.

---

## 1. Contexto

**MIT 6.S191 — Introduction to Deep Learning** es un curso intensivo de una semana dictado en enero del MIT desde 2017, abierto al público en línea. Su lecture 3 ("Deep Computer Vision") es la introducción canónica a CNNs aplicadas a visión: motivación desde primeros principios, convolución como inducción de bias, y un panorama de aplicaciones (clasificación, detección, segmentación, control).

La edición 2026 que aquí glosamos fue dictada por **Alexander Amini** (PhD MIT, fundador de Themis AI) el 6 de enero de 2026. La estructura es muy similar a la edición 2020 (con Ava Soleimany), pero con ejemplos actualizados — McKinney 2020 sobre cribado de cáncer de mama, Amini ICRA 2019 sobre conducción end-to-end con incertidumbre.

El recorrido del lecture se cubre temáticamente en [`notas`](/videos/mit-6s191-l3-2026/notas/). Esta profundización complementa con (a) los papers seminales que la clase nombra al pasar, (b) derivaciones que se asumen, y (c) la frontera moderna que la clase no incluye.

---

## 2. Los papers seminales del canon CNN

### 2.1 LeNet-5 (LeCun et al., 1998)

[*Gradient-Based Learning Applied to Document Recognition*](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf), Proceedings of the IEEE.

- Primer ejemplo industrialmente útil de una CNN: reconocimiento de dígitos manuscritos para procesar cheques bancarios, desplegado en NCR y Lockheed-Martin Cheque a fines de los 90.
- Arquitectura: $C_1 \to S_2 \to C_3 \to S_4 \to C_5 \to F_6 \to \text{output}$ — 5 capas convolucionales/subsampling + 2 fully-connected.
- Establece el patrón `Conv → Pool → Conv → Pool → FC` que el lecture sigue tratando como canónico 28 años después.
- Por qué importa: demuestra que weight sharing + locality + gradiente bastan para ganar a feature engineering en una tarea estructurada.

### 2.2 AlexNet (Krizhevsky, Sutskever, Hinton, 2012)

[*ImageNet Classification with Deep Convolutional Neural Networks*](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), NeurIPS 2012.

- El "Big Bang" del deep learning moderno. Reduce el error top-5 en ImageNet del 26% (sistemas previos) al **15.3%**.
- Tres innovaciones técnicas: ReLU como activación (en vez de tanh/sigmoid, evita saturación), dropout (regularización), y entrenamiento en 2 GPUs en paralelo (la red se parte por canales).
- Arquitectura: 5 conv + 3 FC, ~60M parámetros, entrenado por ~6 días en 2 × GTX 580.
- Por qué importa: demostró empíricamente que profundidad + GPU + dataset grande resuelve visión. Su data augmentation (random crop, horizontal flip, PCA color jitter) es la receta que el material UC cubre con detalle en [`fundamentos/data-augmentation`](/fundamentos/data-augmentation/).

### 2.3 VGG (Simonyan & Zisserman, 2014)

[*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556), arXiv:1409.1556.

- Apuesta deliberada por la **profundidad uniforme** con filtros pequeños: solo $3 \times 3$ convs, apilados 16 ó 19 capas.
- Insight central: dos convs $3 \times 3$ apiladas tienen el mismo receptive field que una $5 \times 5$, pero con menos parámetros y más no-linealidad. La derivación detallada está en [`clases/clase-09/profundizacion`](/clases/clase-09/profundizacion/).
- VGG-16 sigue siendo benchmark didáctico — su simplicidad arquitectural lo hace ideal para enseñar.

### 2.4 GoogLeNet / Inception (Szegedy et al., 2015)

[*Going Deeper with Convolutions*](https://arxiv.org/abs/1409.4842), CVPR 2015.

- Introduce el **módulo Inception**: una capa que aplica varias convs en paralelo con kernels distintos ($1 \times 1$, $3 \times 3$, $5 \times 5$) y concatena las salidas.
- Innovación de eficiencia: **convoluciones $1 \times 1$** como reducción de dimensionalidad antes de las convs caras. La derivación de ahorro de parámetros está en [`clases/clase-09/profundizacion`](/clases/clase-09/profundizacion/).
- Reemplaza la pila de capas FC al final por **global average pooling**, eliminando la mayor parte de los parámetros (FC en VGG = 89% del total).
- Inception v2/v3 (Szegedy 2015b) factoriza $5 \times 5$ en dos $3 \times 3$ y $7 \times 7$ en dos $3 \times 1$ y $1 \times 3$, ganando eficiencia adicional.

### 2.5 ResNet (He, Zhang, Ren, Sun, 2015/2016)

[*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385), CVPR 2016.
[*Identity Mappings in Deep Residual Networks*](https://arxiv.org/abs/1603.05027), ECCV 2016.

- Resuelve el "degradation problem": redes muy profundas entrenan **peor** que redes menos profundas (no por overfitting, sino por dificultad de optimización).
- La cura: bloques residuales con conexiones de identidad que se saltan capas. La función aprendida es $\mathcal{F}(x) + x$ en vez de $\mathcal{H}(x)$ — más fácil de aprender una perturbación pequeña que la transformación completa.
- Derivación del flujo de gradiente:

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \left(1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i)\right)
$$

  El "+1" garantiza que el gradiente nunca se anula, eliminando el vanishing gradient en redes profundas.
- Permite entrenar 50, 101, y hasta 152 capas. Sigue siendo el backbone más usado en producción a 2026, junto con sus descendientes.

### 2.6 Batch Normalization (Ioffe & Szegedy, 2015)

[*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*](https://arxiv.org/abs/1502.03167), ICML 2015.

- Normaliza las activaciones por mini-batch a media 0, varianza 1, luego aplica scale-shift aprendibles $\gamma, \beta$:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta
$$

- Efecto: estabiliza el entrenamiento, permite learning rates más agresivos, reduce sensibilidad a la inicialización.
- Aunque la motivación original ("internal covariate shift") fue cuestionada (Santurkar 2018 argumenta que funciona porque suaviza el landscape de la pérdida), la efectividad empírica nadie la disputa.
- Variantes: **layer normalization** (Ba 2016, normaliza por feature en vez de por batch — útil cuando el batch es chico, dominante en transformers), **group normalization** (Wu 2018), **instance normalization** (Ulyanov 2016).

### 2.7 R-CNN family (Girshick et al., 2014–2017)

- [*Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation*](https://arxiv.org/abs/1311.2524) — R-CNN, Girshick CVPR 2014.
- [*Fast R-CNN*](https://arxiv.org/abs/1504.08083) — Girshick ICCV 2015.
- [*Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*](https://arxiv.org/abs/1506.01497) — Ren et al. NeurIPS 2015.
- [*Mask R-CNN*](https://arxiv.org/abs/1703.06870) — He et al. ICCV 2017.

Evolución incremental:

| Versión | Region proposals | CNN passes | Output |
| --- | --- | --- | --- |
| **R-CNN** (2014) | Selective search (no aprendido) | ~2000 por imagen | Boxes + clases |
| **Fast R-CNN** (2015) | Selective search | 1 por imagen + ROI pooling | Boxes + clases |
| **Faster R-CNN** (2015) | RPN aprendido | 1 por imagen | Boxes + clases |
| **Mask R-CNN** (2017) | RPN aprendido | 1 + ROIAlign | Boxes + clases + máscara |

Mask R-CNN es importante porque unifica detección + segmentación de instancia: predice una máscara binaria por bounding box, separando objetos de la misma clase (algo que la segmentación semántica del lecture no logra).

### 2.8 YOLO (Redmon et al., 2016) y SSD (Liu et al., 2016)

- [*You Only Look Once: Unified, Real-Time Object Detection*](https://arxiv.org/abs/1506.02640) — Redmon CVPR 2016.
- [*SSD: Single Shot MultiBox Detector*](https://arxiv.org/abs/1512.02325) — Liu ECCV 2016.

Detectores **single-shot**: una sola pasada de CNN produce una grid sobre la imagen, y por cada celda de la grid predice $(c, x, y, w, h, p)$ — clase, bounding box, confianza. Eliminan completamente la etapa de propuesta.

Trade-off: más rápidos que Faster R-CNN (real-time, >30 FPS), peor mAP en objetos pequeños — la grid limita la resolución de detección. Las versiones modernas (YOLOv5/v8/v11) cierran gran parte de esa brecha y son el estándar de facto en producción a 2026.

### 2.9 Fully Convolutional Networks (Long et al., 2015)

[*Fully Convolutional Networks for Semantic Segmentation*](https://arxiv.org/abs/1411.4038), CVPR 2015.

- Reemplaza las FC al final del backbone por más capas convolucionales, manteniendo la salida 2D.
- Para recuperar la resolución espacial perdida por pooling, introduce **transposed convolution** (también llamada *fractionally-strided convolution*).
- Skip connections del encoder al decoder ya aparecen aquí en forma rudimentaria; U-Net las llevará a su forma canónica.

### 2.10 U-Net (Ronneberger et al., 2015)

[*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597), MICCAI 2015.

- Encoder-decoder simétrico con **skip connections** densas: la resolución de cada nivel del decoder recibe los feature maps de la misma resolución del encoder, concatenados.
- Diseñada originalmente para segmentación de microscopía electrónica con datasets pequeños (~30 imágenes); ganó el ISBI 2015 challenge sin pretrain.
- Es el backbone arquitectural de muchos modelos de difusión modernos (Stable Diffusion usa una U-Net como denoiser).

### 2.11 Grad-CAM (Selvaraju et al., 2017)

[*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*](https://arxiv.org/abs/1610.02391), ICCV 2017.

- Para una predicción de clase $c$, calcula los gradientes del score $y^c$ con respecto a los feature maps $A^k$ de la última capa convolucional, y los pondera para producir un heatmap:

$$
\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{ij}^k}, \qquad L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left( \sum_k \alpha_k^c A^k \right)
$$

- Resultado: una localización gruesa (resolución del último feature map) de las regiones de la imagen que sustentan la predicción.
- Es el método de attribution más usado en producción por su simplicidad: no requiere modificar el modelo ni reentrenar.
- Cobertura más amplia (saliency maps, occlusion, integrated gradients, Lucent) en [`fundamentos/interpretabilidad`](/fundamentos/interpretabilidad/).

---

## 3. Derivaciones complementarias

### 3.1 Crecimiento del receptive field

Para una pila de $L$ capas convolucionales con kernel $k_i$ y stride $s_i$, el receptive field $r_L$ en la capa $L$ vale:

$$
r_L = 1 + \sum_{l=1}^{L} (k_l - 1) \prod_{i=1}^{l-1} s_i
$$

Con todas las capas $k = 3, s = 1$: $r_L = 1 + L \cdot 2 = 2L + 1$ (crecimiento lineal). Con stride 2 alternado: crecimiento exponencial.

UC formaliza este cálculo en [`clases/clase-09/profundizacion`](/clases/clase-09/profundizacion/) — ahí se recorre VGG-16 capa por capa.

### 3.2 Transposed convolution: por qué no es "deconvolución"

Una convolución estándar puede expresarse como multiplicación matricial $y = Cx$, donde $C$ es una matriz Toeplitz que codifica el filtro y la geometría (stride, padding). La **transposed convolution** aplica $C^\top$, no $C^{-1}$. Por eso "deconvolución" es un nombre engañoso: $C^\top \neq C^{-1}$ en general.

Operativamente, $C^\top$ upsamplea: dado un input $h \times w$, produce un output más grande insertando ceros entre las posiciones del input y aplicando una convolución estándar. La equivalencia con la matriz transpuesta garantiza que los gradientes de la convolución directa se calculan exactamente con la transpuesta — por eso se llama así.

Para upsampling sin artifacts checkerboard, la práctica moderna prefiere `nearest-upsample → conv` en vez de transposed conv (Odena 2016, "Deconvolution and Checkerboard Artifacts").

### 3.3 IoU, NMS y asignación de anchors

**IoU (Intersection over Union)** entre dos bounding boxes $A$ y $B$:

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

Es el criterio canónico de overlap: 0 = no se tocan, 1 = idénticas. mAP en COCO se evalúa promediando precisión sobre umbrales IoU $\in \{0.5, 0.55, \dots, 0.95\}$.

**NMS (Non-Maximum Suppression):** un detector típicamente emite cientos de cajas para el mismo objeto. NMS las depura:

```
ordenar cajas por score descendente
mientras hay cajas:
    tomar la de mayor score → output
    eliminar las demás con IoU > umbral (e.g., 0.5) respecto a la elegida
```

**Asignación de anchors:** durante entrenamiento, cada ground-truth box se asigna al anchor con mayor IoU; ese anchor recibe la pérdida de regresión y clasificación positiva. Anchors con IoU bajo (< 0.3) son negativos; los intermedios se ignoran (no contribuyen al loss). Este balance evita que la masa de anchors negativos abrume el gradiente.

---

## 4. Diferencias con el material UC y con MIT 2020

### 4.1 vs. clase 09 del curso UC

| Tema | MIT 2026 (este lecture) | UC clase-09 |
| --- | --- | --- |
| Motivación de visión, "qué ven los computadores" | Profundo (slides 1-22) | Breve |
| Convolución paso a paso | Animación detallada (slides 27-40) | Formal con dimensión de salida |
| Arquitecturas específicas (VGG, Inception, ResNet) | Mención de pasada | **Profundo** — derivaciones, código, tradeoffs |
| Receptive field | Conceptual | Cálculo explícito por capa |
| Filtros 1×1 / bottleneck | No | Sí |
| Detección y segmentación | **Profundo** — R-CNN family, FCN, U-Net | No |
| Conducción autónoma end-to-end | **Profundo** — caso real con incertidumbre | No |
| Interpretabilidad (Feature Visualization, Saliency) | Implícita en jerarquía aprendida | **Profundo** en [`fundamentos/interpretabilidad`](/fundamentos/interpretabilidad/) |
| Transfer learning | Mencionado en "una arquitectura, muchas apps" | **Profundo** en [`fundamentos/transfer-learning`](/fundamentos/transfer-learning/) |
| Data augmentation | No | **Profundo** en [`fundamentos/data-augmentation`](/fundamentos/data-augmentation/) |

**Recomendación de lectura combinada:** ver el video MIT 2026 para motivación + aplicaciones, leer clase-09 para arquitecturas específicas, y los tres `fundamentos/*` para herramientas (transfer, augmentation, interpretabilidad).

### 4.2 vs. MIT 2020 (lecture L4 de la edición 2020)

La edición 2020 del mismo curso (con Ava Soleimany) cubría CNNs como L4. Diferencias respecto a la 2026:

- **Profundidad de aplicaciones:** el lecture 2026 dedica más slides (~20) a detección/segmentación/control end-to-end. La 2020 cubría detección con menos detalle y agregaba más matemática de pooling.
- **Casos médicos:** ambas usan el ejemplo de cáncer de mama, pero la 2026 cita McKinney 2020 (publicado en *Nature*), que en 2020 acababa de salir.
- **Conducción autónoma:** la 2026 incluye el framework end-to-end probabilístico de Amini ICRA 2019, ausente en la 2020.

---

## 5. Conceptos NO cubiertos por MIT 2026 que vale la pena conocer

### 5.1 Vision Transformers (ViT) — Dosovitskiy et al., 2021

[*An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929), ICLR 2021.

- Aplica el Transformer (MIT L2 2026) directamente a imágenes: divide la imagen en patches $16 \times 16$, los aplana en tokens, agrega embedding posicional, y pasa por capas de self-attention.
- Sin convoluciones. Logra rivalizar y superar a CNNs en ImageNet cuando se entrena en datasets gigantes (JFT-300M).
- Implicación: la inducción arquitectural de la convolución no es estrictamente necesaria si se compensa con datos.

### 5.2 ConvNeXt — Liu et al., 2022

[*A ConvNet for the 2020s*](https://arxiv.org/abs/2201.03545), CVPR 2022.

- Modernización quirúrgica de ResNet incorporando lecciones de ViT: kernels más grandes ($7 \times 7$ depthwise), GELU en vez de ReLU, layer norm en vez de batch norm, fewer activaciones, etc.
- Recupera y supera a ViT en ImageNet sin abandonar la convolución. Argumenta que la arquitectura es menos importante que el régimen de entrenamiento moderno.

### 5.3 EfficientNet — Tan & Le, 2019

[*EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*](https://arxiv.org/abs/1905.11946), ICML 2019.

- Propone **compound scaling**: en vez de escalar profundidad, ancho o resolución por separado, escalarlos juntos con un único hiperparámetro $\phi$:

$$
d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi, \quad \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
$$

- Usa búsqueda neural-arquitectural (NAS) para encontrar la base B0, y aplica el scaling para B1...B7.
- Es el estándar para producción cuando el compute es restringido (mobile, edge).

### 5.4 Pretraining auto-supervisado

CNNs preentrenadas con etiquetas (ImageNet supervisado) son la receta clásica que el lecture asume. La frontera moderna entrena sin etiquetas:

- **SimCLR** (Chen et al. 2020) — contrastive learning con augmentations: dos vistas de la misma imagen deben estar cerca, dos vistas de imágenes distintas deben estar lejos en el embedding.
- **MAE** (He et al. 2021) — masked autoencoder: enmascara 75% de los patches de una imagen, entrena un ViT a reconstruirlos.
- **DINO** (Caron et al. 2021) — self-distillation con ViT, produce features útiles para segmentación sin etiquetas.

Todas producen backbones que, fine-tuneados, igualan o superan al pretrain supervisado en datasets pequeños — democratizando la transferencia.

### 5.5 Modelos de difusión para visión

El lecture cierra apuntando a "control y segmentación", pero no a generación. La generación moderna (Stable Diffusion, DALL-E, Midjourney) usa modelos de difusión donde el denoiser es una **U-Net** convolucional (a veces con cross-attention al texto). El L4 del mismo curso 2026 ("Deep Generative Modeling") cubre este territorio.

---

## Atribución

> Material adaptado de **MIT 6.S191 (2026) Lecture 3: Deep Computer Vision**, Alexander Amini, 6 de enero de 2026.
> [Video](https://www.youtube.com/watch?v=pqIcoskUuWs) — [Slides oficiales](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L3.pdf) — [Sitio del curso](https://introtodeeplearning.com/).
> Notas en español como elaboración independiente. Sin afiliación oficial con MIT.
