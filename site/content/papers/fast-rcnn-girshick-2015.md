---
title: "Fast R-CNN"
weight: 104
math: true
---

{{< paper-card
    title="Fast R-CNN"
    authors="Girshick"
    year="2015"
    venue="ICCV 2015"
    pdf="/papers/fast-rcnn-girshick-2015.pdf"
    arxiv="1504.08083" >}}
Bisagra entre el R-CNN multi-stage de 2014 y el Faster R-CNN end-to-end de finales de 2015. Introduce **RoI Pooling** (extracción de features de tamaño fijo desde un feature map compartido), **multi-task loss** que combina softmax de clasificación y **Smooth L1** para regresión de bounding box, y entrenamiento joint con dos sibling heads. Logra **213× speedup en inferencia** (con truncated SVD), **9× en training**, y mAP superior a R-CNN y SPP-Net en VOC07.
{{< /paper-card >}}

---

## El problema

R-CNN (Girshick et al., CVPR 2014) había dado el primer gran salto en detección con deep learning (de ~35% a 53.7% mAP en VOC07), pero a un costo operativo brutal:

- **Pipeline multi-stage**: CNN fine-tuning → SVMs one-vs-rest por clase → bbox regressors lineales. Tres etapas desconectadas, con features escritas a disco entre cada una.
- **Storage**: las activaciones fc7 de los ~2000 proposals por imagen para todo VOC07 trainval ocupaban **cientos de GB** en disco.
- **Inferencia lenta**: cada uno de los 2000 region proposals pasa por un forward pass completo de la CNN. Con VGG-16, **47 segundos por imagen**.
- **Training largo**: ~84 horas con VGG-16.

SPP-Net (He et al., ECCV 2014) había mitigado parcialmente el costo de inferencia computando el feature map una sola vez por imagen y usando spatial pyramid pooling sobre cada RoI. Pero seguía siendo multi-stage, y crucialmente **no podía propagar gradientes a las capas convolucionales** durante fine-tuning bajo su sampling RoI-centric. Eso le costaba puntos de mAP en redes profundas como VGG-16.

Fast R-CNN resuelve los tres problemas simultáneamente: pipeline end-to-end (excepto los proposals externos), inferencia 100× más rápida, y fine-tuning real de las convs.

## Ideas principales

- **Backbone compartido**: la imagen completa pasa una sola vez por las capas conv del backbone (CaffeNet, VGG_M, o VGG-16). El feature map resultante se reutiliza para todas las RoIs.
- **RoI Pooling**: por cada RoI, se proyecta el rectángulo al espacio del feature map (dividiendo por el stride, ~16 para VGG-16) y se aplica max-pool sobre una grilla $H \times W$ (típicamente $7 \times 7$). Output: tensor de tamaño fijo $C \times H \times W$ por RoI, independiente del tamaño original del rectángulo.
- **Sibling heads**: dos cabezas paralelas comparten el backbone y las FCs. Una predice softmax sobre $K+1$ clases (objeto + background). La otra produce $4K$ deltas de bbox (4 por clase).
- **Multi-task loss**: $L = L_{cls} + \lambda \, [u \geq 1] \, L_{loc}$ con $\lambda = 1$. La regresión sólo se penaliza para RoIs de foreground (Iverson bracket $[u \geq 1]$).
- **Smooth L1** para regresión, robusto a outliers (saturación lineal fuera de $|x| < 1$).
- **Image-centric sampling**: mini-batch de 2 imágenes × 64 RoIs = 128 RoIs por batch. Amortiza el forward pass del backbone entre las 64 RoIs de cada imagen → ~64× speedup vs el sampling RoI-centric de SPP-Net.
- **End-to-end joint training** con proposals externos (Selective Search). Todas las capas se actualizan: convs, FCs, y ambas cabezas.

## Arquitectura

Pipeline en cuatro pasos:

1. **Backbone convolucional**: imagen completa $\to$ feature map $C \times H' \times W'$ (e.g., VGG-16 con stride 16: $H' = H_{img}/16$). La última max-pool del backbone se **reemplaza** por la RoI Pooling layer.

2. **RoI Pooling**: para cada RoI $(r, c, h, w)$ proyectada al feature map, se divide en grilla $H \times W$ (e.g., $7 \times 7$ para VGG-16) y se aplica max-pool por sub-ventana.
   $$
   y_{rj} = \max_{i' \in \mathcal{R}(r,j)} x_{i'}
   $$
   Backward acumula gradientes en los índices argmax (igual que max-pool estándar, generalizado a múltiples regiones que comparten el feature map).

3. **FCs compartidas**: el output del RoI pool entra a fc6 → ReLU → fc7 → ReLU. Vector de 4096-d por RoI.

4. **Sibling heads**:
   - **Cls head**: FC + softmax sobre $K+1$ clases.
   - **Bbox reg head**: FC con $4K$ outputs (un regresor de 4 coords por clase).

Backbone y FCs son compartidos entre todas las RoIs. Sólo las cabezas finales son per-RoI.

## Multi-task loss en detalle

$$
L(p, u, t^u, v) = L_{cls}(p, u) + \lambda \, [u \geq 1] \, L_{loc}(t^u, v)
$$

con:

- $L_{cls}(p, u) = -\log p_u$ (log-loss estándar).
- $L_{loc}(t^u, v) = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t^u_i - v_i)$.

donde Smooth L1 es:

$$
\text{smooth}_{L_1}(x) = \begin{cases} 0.5 x^2 & \text{si } |x| < 1 \\ |x| - 0.5 & \text{en otro caso} \end{cases}
$$

**¿Por qué Smooth L1 y no L2?** L2 produce gradientes que escalan linealmente con el error: un outlier con $|t-v| = 10$ contribuye con gradiente 10 y puede dominar el batch, exigiendo gradient clipping o lr cuidadoso. Smooth L1 satura el gradiente en $\pm 1$ fuera del rango $|x| < 1$ — equivalente a **Huber loss con $\delta = 1$**. R-CNN y SPP-Net usaban L2 y requerían tuning fino del lr; Fast R-CNN entrena estable sin ese cuidado.

Los targets $t$ y $v$ se parametrizan como **scale-invariant translations** y **log-space size shifts** relativos al proposal:

$$
t_x = (G_x - P_x) / P_w, \quad t_w = \log(G_w / P_w)
$$

(análogo para $t_y$, $t_h$), normalizados a media cero y varianza unitaria.

## Training details

**Mini-batch sampling (image-centric)**: $N = 2$ imágenes, $R/N = 64$ RoIs por imagen, total $R = 128$. Dos forward passes del backbone por batch. Las 64 RoIs de cada imagen comparten el feature map.

**Selección de RoIs por imagen**:

- 25% **foreground**: IoU $\geq 0.5$ con alguna GT. Etiqueta = clase del GT.
- 75% **background**: IoU $\in [0.1, 0.5)$. Etiqueta $u = 0$.

El umbral inferior de 0.1 actúa como **hard negative mining implícito**: descarta backgrounds triviales (IoU < 0.1) y se queda con los confundibles. No requiere mining explícito al estilo R-CNN.

**Data augmentation**: solo horizontal flipping (p = 0.5).

**SGD**: lr inicial $10^{-3}$, drop a $10^{-4}$ después de 30k iters (VOC07/12). Momentum 0.9, weight decay $5 \times 10^{-4}$. Per-layer lr 1 para pesos, 2 para biases.

## Truncated SVD para acelerar las FC

En detección, con 2000 RoIs pasando por las FCs por imagen, **casi la mitad del forward time se gasta en fc6 + fc7** (no en las convs como en classification). Truncated SVD descompone $W \in \mathbb{R}^{u \times v}$ como $W \approx U \Sigma_t V^T$ con $t$ componentes, reemplazando la capa única por dos capas sin no-linealidad entre ellas.

**Setup VGG-16**: fc6 ($25088 \times 4096$) $\to$ $t = 1024$; fc7 ($4096 \times 4096$) $\to$ $t = 256$. Resultado: **30% de reducción en tiempo total** (320 ms → 223 ms), con pérdida de **solo 0.3 puntos de mAP** (66.9 → 66.6), sin re-fine-tuning. Es el origen de la idea "comprime las FC con low-rank" que se vuelve estándar en model compression.

## Resultados

### PASCAL VOC

| Método | Train | mAP VOC07 |
| --- | --- | --- |
| SPP-Net BB (VGG16) | 07 \ diff | 63.1 |
| R-CNN BB (VGG16) | 07 | 66.0 |
| **Fast R-CNN (VGG16)** | 07 | **66.9** |
| Fast R-CNN (VGG16) | 07 \ diff | 68.1 |
| Fast R-CNN (VGG16) | 07+12 | **70.0** |

- VOC10: 66.1 mAP (12), 68.8 (07++12).
- VOC12: 65.7 mAP (12), 68.4 (07++12). Top del leaderboard.

### Tiempos (VGG-16 "L")

| | Fast R-CNN L | Fast R-CNN L + SVD | R-CNN L | SPP-Net L |
| --- | --- | --- | --- | --- |
| Train time | **9.5 h** | — | 84 h | 25.5 h |
| Test (s/img) | 0.32 | **0.22** | 47.0 | 2.3 |
| Speedup vs R-CNN | 146× | **213×** | 1× | 20× |
| mAP VOC07 | **66.9** | 66.6 | 66.0 | 63.1 |

Tres logros simultáneos: **9× más rápido en training**, **146× (213× con SVD) en test**, y **mAP más alto** que ambos competidores.

## Insights del paper

### Softmax joint-trained iguala o supera a SVMs

R-CNN entrenaba SVMs one-vs-rest post-hoc porque empíricamente vencían al softmax en su régimen multi-stage. Fast R-CNN reabre la pregunta:

| Configuración | S | M | L |
| --- | --- | --- | --- |
| Fast R-CNN + SVMs post-hoc | 56.3 | 58.7 | 66.8 |
| **Fast R-CNN softmax joint** | **57.1** | **59.2** | **66.9** |

Softmax joint-trained iguala o supera a los SVMs (+0.1 a +0.8 mAP). El "boost" de R-CNN venía del **entrenamiento fragmentado**, no de una virtud intrínseca de los SVMs. Adicionalmente, softmax introduce **competencia entre clases** vía normalización.

### Multi-task helps both tasks

| Configuración | L (VGG-16) |
| --- | --- |
| Sólo $L_{cls}$ (sin bbox reg) | 62.6 |
| Multi-task train, bbox reg disabled at test | 63.4 |
| Stage-wise (cls primero, bbox reg después con backbone frozen) | 64.0 |
| **Multi-task full** | **66.9** |

Dos observaciones canónicas:

1. Entrenar multi-task **mejora la clasificación pura** (+0.8 mAP entre fila 1 y 2): el gradiente de bbox reg actúa como **regularizador** del backbone compartido.
2. Multi-task supera a stage-wise por +1.5 a +3 mAP.

Argumento clásico a favor del multi-task learning con shared representations (Caruana 1997).

### Image-centric sampling es lo que faculta fine-tunear las convs

Con RoI-centric (SPP-Net), 128 RoIs vienen de 128 imágenes distintas $\to$ 128 forward passes parciales por mini-batch. Image-centric con $N=2$, $R/N=64$ $\to$ sólo 2 forward passes. El speedup empírico es ~64×, y es lo que permite que VGG-16 propague gradientes hacia capas profundas (≥ conv3_1).

| Layers fine-tuned (VGG-16) | mAP |
| --- | --- |
| ≥ fc6 (emula SPP-Net) | 61.4 |
| ≥ conv3_1 (default Fast R-CNN) | **66.9** |
| ≥ conv2_1 | 67.2 |

Congelar las convs como hacía SPP-Net **cuesta 5.5 puntos** en VGG-16.

### Sparse proposals son mejores que dense

Reemplazar Selective Search por 45k boxes uniformes densos baja mAP de **66.9% a 52.9%**. Los proposals sparse actúan como un **cascade Viola-Jones-style** que filtra negativos triviales antes del classifier. Y **Average Recall no correlaciona con mAP** al variar el número de proposals: hay que medir mAP directo.

## Limitaciones reconocibles

- **Selective Search externo**: ~2 s/imagen en CPU domina el tiempo total. Aunque la CNN procesa en 0.3 s, el sistema end-to-end no llega a real-time. **Faster R-CNN** lo resuelve con la RPN.
- **Cuantización en RoI Pooling**: las coordenadas del RoI se redondean dos veces (al proyectar al feature map y al dividir en grilla $H \times W$). Para stride 16 de VGG-16, hasta ~16 px de misalignment. Tolerable para clasificación, **desastroso para masks precisas** $\to$ **RoIAlign** (Mask R-CNN).
- **Single-scale** falla con objetos extremos en tamaño $\to$ **FPN** (Lin et al. 2017).
- **No es end-to-end real desde la imagen**: el módulo de proposals no se aprende.
- **Cada RoI clasificada independientemente**: sin contexto inter-objeto ni oclusión modelada.

## Legado

Fast R-CNN cristaliza el patrón arquitectónico **"backbone compartido + per-RoI feature + multi-task sibling heads"** que sobrevive ~10 años y se adapta a cada problema de visión por instancia.

### Familia R-CNN directa

- **Faster R-CNN** (Ren et al., NeurIPS 2015): integra una **RPN** que comparte el backbone. Elimina Selective Search. 5 fps end-to-end.
- **Mask R-CNN** (He et al., ICCV 2017): añade mask head paralela. Reemplaza RoI Pooling por **RoIAlign**.
- **R-FCN**, **Cascade R-CNN**, **Libra R-CNN**, **Grid R-CNN**: variantes que refinan el patrón.

### RoI Pool / RoIAlign como primitiva universal

- **RoIAlign** (Mask R-CNN): bilinear interpolation sin redondeo.
- **PrRoI Pool** (Jiang et al. ECCV 2018): integración continua.
- **Deformable RoI Pooling** (Dai et al. ICCV 2017): offsets aprendibles por sub-ventana.
- **BezierAlign** (ABCNet, Liu et al. CVPR 2020): generaliza RoIAlign a regiones curvas parametrizadas por curvas de Bézier. **Base directa del scene text spotter de la clase 21**.
- **RotatedRoIAlign**: para detección rotada (DOTA, aerial).

### Multi-task con sibling heads

Patrón universal en visión moderna: detección (cls + bbox), instance seg (cls + bbox + mask), pose (cls + bbox + keypoints), **text spotting (cls + bbox + recognition)**, DensePose (cls + bbox + UV), scene graphs.

### Smooth L1 / Huber

Estándar en regresión de bboxes durante toda la era R-CNN/YOLO/SSD/RetinaNet, hasta que IoU loss, GIoU, DIoU, CIoU (Zheng et al. 2020) toman terreno por mejor alineación con la métrica de evaluación.

### Truncated SVD

Origen del paradigma de comprimir FCs con low-rank: aparece después en knowledge distillation, pruning estructurado, hardware-aware NAS.

## Conexión con la clase 21 (Scene Text Spotting)

La clase 21 aborda **Scene Text Recognition (STR)** y **end-to-end text spotting**, donde Fast R-CNN es el antecesor genealógico de prácticamente todos los spotters de la década 2015-2024:

### ABCNet (Liu et al., CVPR 2020)

ABCNet ("Real-time Scene Text Spotting with Adaptive Bezier-Curve Network") es la referencia central de la clase. Genera proposals como **curvas de Bézier de orden 3** (8 puntos de control) para representar texto curvo, y usa **BezierAlign** — una generalización directa de RoIAlign, que a su vez es Fast R-CNN's RoI Pooling con bilinear interpolation. La cabeza de reconocimiento es un CRNN sobre los features bezier-aligned.

El paper ABCNet referencia explícitamente la familia: "Following the success of two-stage detectors like Fast/Faster R-CNN and Mask R-CNN, recent text spotters adopt the share-backbone + RoI-extract + per-RoI heads paradigm." **Fast R-CNN es el tronco del árbol genealógico de ABCNet**.

### Mask TextSpotter (Lyu et al., ECCV 2018; PAMI 2021)

Construido directamente sobre **Mask R-CNN** (que es Fast R-CNN + RPN + mask head + RoIAlign). Detecta texto con masks y reconoce con una cabeza adicional de secuencia carácter por carácter.

### CharNet, FOTS, EAST, PAN, CRAFT, DBNet

Toda la línea comparte ADN con Fast R-CNN: backbone CNN compartido, pooling/alignment operator que extrae features de regiones de tamaño/forma variable a tensores fijos, múltiples heads sibling (detección + reconocimiento), multi-task loss con balancing weights, Smooth L1 en regresión de geometría.

### Por qué importa el patrón

Para el practitioner de visión moderna, leer Fast R-CNN entrega cuatro cosas:

1. **El vocabulario**: RoI, sibling heads, multi-task loss, image-centric sampling, Smooth L1.
2. **Las trade-offs**: shared computation vs per-region, sparse vs dense proposals, single-stage vs multi-stage training.
3. **La intuición**: por qué softmax joint-trained gana a SVMs, por qué multi-task ayuda a ambas tareas, por qué fine-tunear convs profundas importa.
4. **La genealogía**: cuando un paper de 2024 sobre transformers para detección asume Fast R-CNN, hay que conocerlo para entender qué problema resuelven.

Los detectores modernos basados en Transformers (DETR, Deformable DETR, DINO, MaskFormer) reemplazan las RoIs explícitas por queries aprendibles, pero conservan **la idea**: shared encoder + multiple parallel heads with multi-task loss. Es la idea, no la implementación, lo que perdura.

## Notas y enlaces

### Fundamentos

- [Detección de objetos](/fundamentos/deteccion-de-objetos): el problema general que Fast R-CNN ataca.
- [Scene text recognition](/fundamentos/scene-text-recognition): el dominio de la clase 21, descendiente directo.
- [Funciones de pérdida](/fundamentos/funciones-perdida): cross-entropy, Smooth L1, multi-task balancing.

### Papers relacionados

- [Faster R-CNN (Ren et al., 2015)](/papers/faster-rcnn-ren-2015): elimina Selective Search con la RPN.
- [Mask R-CNN (He et al., 2017)](/papers/mask-rcnn-he-2017): añade mask head y reemplaza RoI Pool por RoIAlign.
- [ABCNet (Liu et al., 2020)](/papers/abcnet-liu-2020): scene text spotting con BezierAlign, descendiente directo.
- [FPN (Lin et al., 2017)](/papers/fpn-lin-2017): resuelve el problema multi-escala que Fast R-CNN no abordó.

### Clase

- [Clase 21 — Scene Text Spotting](/clases/clase-21): dominio donde el patrón Fast R-CNN se sigue usando en 2024+.
