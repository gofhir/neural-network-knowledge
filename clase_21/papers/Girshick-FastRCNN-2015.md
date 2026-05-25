---
title: "Fast R-CNN"
authors: ["Ross Girshick"]
year: 2015
venue: "ICCV 2015"
slug: "fast-rcnn-girshick-2015"
arxiv: "1504.08083"
tags: ["object-detection", "rcnn-family", "roi-pooling", "multi-task-learning", "vgg16", "selective-search"]
---

# Fast R-CNN — Girshick, ICCV 2015

## Resumen ejecutivo

"Fast R-CNN" es el segundo paper de la saga R-CNN de Ross Girshick y resuelve los dos talones de Aquiles de su trabajo previo: el pipeline multi-stage de R-CNN (2014) y la inferencia glacial (~47 segundos por imagen con VGG-16). La idea central, deceptivamente simple, es procesar la imagen completa una sola vez con la CNN, obtener un mapa de features compartido, y para cada uno de los ~2000 region proposals extraer un vector de tamaño fijo mediante una operación llamada **RoI Pooling**. Sobre ese vector, dos cabezas hermanas (sibling heads) producen simultáneamente la clasificación softmax sobre $C+1$ clases y la regresión de bounding box, entrenadas conjuntamente con una **multi-task loss**.

Los números son contundentes: **9× speed-up en entrenamiento** (de 84h a 9.5h con VGG-16), **146× en inferencia** (213× con truncated SVD), **mAP 66.9% en VOC07** (vs 66.0% de R-CNN, 63.1% de SPP-Net), eliminación total del caché de features en disco (cientos de GB), y por primera vez la posibilidad de propagar gradientes a través de todas las capas convolucionales incluso usando una red profunda como VGG-16. Detrás de estos resultados hay decisiones de diseño que se vuelven canon: smooth L1 para regresión de coordenadas, image-centric sampling para amortizar el forward pass, hard negative mining implícito por threshold de IoU $\in [0.1, 0.5)$, y la observación contraintuitiva de que softmax joint-trained **iguala o supera** a SVMs entrenados post-hoc.

Fast R-CNN no elimina, sin embargo, la dependencia de un módulo externo de proposals (Selective Search, que toma ~2s por imagen y domina el tiempo real). Esa última pieza la resolvería Faster R-CNN (Ren et al., 2015) unos meses después con el Region Proposal Network (RPN). Aun así, Fast R-CNN dejó establecido el patrón arquitectónico **"backbone compartido + per-RoI heads"** que sigue siendo la columna vertebral de los detectores modernos, los segmentadores de instancia (Mask R-CNN), y los spotters de texto end-to-end (Mask TextSpotter, ABCNet, TextSpotting Transformers). Para la clase 21 del curso, donde el foco es Scene Text Recognition y spotting, Fast R-CNN es el antecesor directo cuya estructura "share + crop + classify+regress" se replica en prácticamente toda la familia de spotters.

## Contexto: la era anterior

### Sliding window con features artesanales

Antes de la explosión de las CNNs profundas (2012-), la detección de objetos era dominio de pipelines artesanales. El paradigma canónico era el **sliding window detector**: barrer la imagen con una ventana a múltiples escalas y posiciones, extraer un descriptor en cada ventana, y aplicar un clasificador lineal o SVM. Los descriptores estrella fueron **HOG (Histogram of Oriented Gradients)** de Dalal y Triggs (CVPR 2005), que codifica la distribución local de gradientes en celdas y normaliza por bloques, y **SIFT** de Lowe (1999/2004) para keypoints.

El refinamiento más sofisticado de este paradigma fue **Deformable Part Models (DPM)** de Felzenszwalb, Girshick, McAllester y Ramanan (TPAMI 2010). DPM modela un objeto como un root filter HOG más part filters con deformaciones permitidas, entrenado con latent SVMs. Durante años fue state-of-the-art en PASCAL VOC con mAPs en los 30s. La irónica observación es que el propio Girshick fue uno de los autores de DPM antes de pivotar a deep learning.

### R-CNN (Girshick et al., CVPR 2014): el primer salto profundo

R-CNN ("Regions with CNN features") rompió la escena con un salto enorme en VOC07 (de ~35% a 53.7% mAP). El pipeline:

1. **Region proposals**: Selective Search (Uijlings et al., 2013) genera ~2000 candidatos por imagen agrupando superpixels jerárquicamente. Cada proposal se "warpea" (rescala) a 227×227.
2. **CNN feature extraction**: cada uno de los 2000 crops pasa por una CNN (AlexNet) fine-tuneada, produciendo un vector de 4096-d en la capa fc7.
3. **SVMs por clase**: se entrena un SVM one-vs-rest por clase sobre esos vectores 4096-d, con hard negative mining.
4. **Bounding-box regression**: una tercera etapa entrena regresores lineales de 4 coords (parametrización scale-invariant) sobre las mismas features fc7.

Los problemas eran tres y se vuelven la motivación explícita de Fast R-CNN:

- **Multi-stage hell**: el entrenamiento es CNN → SVM → bbox regressor, con features escritas a disco entre etapas. Pipeline disjoint, no end-to-end, ~84h de entrenamiento.
- **Storage**: las features de fc7 para todos los proposals de VOC07 trainval ocupan **cientos de GB** en disco.
- **Inferencia lenta**: cada uno de los 2000 proposals requiere un forward pass completo de la CNN. Con VGG-16: **47 segundos por imagen**.

### SPP-Net (He et al., ECCV 2014): el primer paso hacia compartir cómputo

Kaiming He y colaboradores propusieron **Spatial Pyramid Pooling Net (SPP-Net)**, que introdujo la idea crítica de **computar el feature map una sola vez por imagen** y luego, para cada RoI, hacer pooling de la región correspondiente del feature map a un vector de tamaño fijo. Específicamente, SPP usa una pirámide de niveles (e.g., 1×1, 2×2, 3×3, 6×6) concatenados.

Esto aceleró R-CNN entre 10× y 100× en test-time, y 3× en training. Pero SPP-Net heredó dos defectos:

1. **Sigue siendo multi-stage**: fine-tune CNN → entrenar SVMs → entrenar bbox regressors.
2. **No puede actualizar las conv layers debajo del SPP** durante fine-tuning porque la propagación del gradiente a través de SPP es **ineficiente** cuando cada RoI viene de una imagen distinta (que es la estrategia que usaban). El receptive field de un RoI suele cubrir la imagen completa, así que cada RoI sample exige un forward pass entero.

Esta limitación —conv layers congeladas— era especialmente dolorosa con VGG-16, donde el fine-tuning de las capas profundas es lo que da los puntos de mAP.

## Arquitectura Fast R-CNN

La Figura 1 del paper describe el pipeline. La arquitectura toma como input:

- Una **imagen completa** (no crops).
- Una lista de **~2000 RoIs** (rectángulos $(r, c, h, w)$ producidos externamente por Selective Search).

Y procesa así:

1. **Backbone convolucional**: la imagen pasa por todas las capas conv + max-pool de una CNN pre-entrenada (CaffeNet "S", VGG_CNN_M_1024 "M", o VGG-16 "L"). La última max-pool se **reemplaza** por una RoI Pooling layer. El output es un feature map único $C \times H' \times W'$ para la imagen entera (e.g., para VGG-16 con stride 16, $H' = H_{img}/16$).

2. **RoI Pooling**: por cada RoI, se proyecta el rectángulo al espacio del feature map (dividiendo por el stride efectivo, ~16 para VGG-16), y se aplica RoI max-pooling para obtener un tensor de tamaño fijo $C \times H \times W$, típicamente $7 \times 7$ para VGG-16 (de modo que sea compatible con la primera FC del backbone, fc6, que espera $7 \times 7 \times 512$).

3. **Fully connected compartidas**: el output del RoI pool entra en una secuencia de FCs (fc6 → ReLU → fc7 → ReLU), produciendo un feature vector de 4096-d por RoI.

4. **Two sibling heads**:
   - **Classification head**: FC → softmax sobre $K+1$ clases ($K$ object classes + 1 background).
   - **Bbox regression head**: FC con $4K$ outputs (4 deltas por clase).

Los pesos del backbone y las FCs son compartidos entre todas las RoIs. Solo las cabezas finales son per-RoI.

## RoI Pooling (matemática del operador)

RoI Pooling es la pieza original del paper. Dado un RoI $(r, c, h, w)$ en coordenadas del feature map (post-projection), divide el rectángulo en una grilla $H \times W$ (e.g., $7 \times 7$) de sub-ventanas de tamaño aproximado $h/H \times w/W$, y aplica max-pool dentro de cada sub-ventana. El output es un tensor $C \times H \times W$ con $H, W$ fijos.

Formalmente, sea $x_i \in \mathbb{R}$ la $i$-ésima activación de input al layer, y $y_{rj}$ el $j$-ésimo output de la $r$-ésima RoI. Entonces:

$$
y_{rj} = x_{i^*(r,j)}, \quad i^*(r,j) = \arg\max_{i' \in \mathcal{R}(r,j)} x_{i'}
$$

donde $\mathcal{R}(r,j)$ es el conjunto de índices que cubren la sub-ventana correspondiente al output $j$ de la RoI $r$.

**Backward**: el gradiente del loss respecto a $x_i$ se acumula sólo en los índices argmax:

$$
\frac{\partial L}{\partial x_i} = \sum_r \sum_j [i = i^*(r,j)] \, \frac{\partial L}{\partial y_{rj}}
$$

Es decir, un input $x_i$ puede ser argmax para múltiples outputs (de RoIs distintas o sub-ventanas distintas) y todas sus contribuciones se suman. Esto es idéntico al backward de max-pool estándar, generalizado a múltiples regiones que comparten el feature map.

**Notas técnicas importantes**:

- RoI Pooling es el **caso especial de un solo nivel** del Spatial Pyramid Pooling de SPP-Net.
- Las coordenadas del RoI se **cuantizan** dos veces: primero al proyectar al espacio del feature map (dividir por stride, redondear), y segundo al dividir el RoI en la grilla $H \times W$ (las sub-ventanas tienen tamaño aproximado $h/H \times w/W$ con redondeo). Esta doble cuantización introduce **misalignment** entre el RoI nominal y el RoI efectivo, que es precisamente el problema que **RoIAlign** (Mask R-CNN, He et al. 2017) resuelve usando interpolación bilineal en lugar de redondeo.
- Permite **inputs de tamaño variable** (RoIs grandes o chicas) con output fijo. Esto es lo que hace que la red funcione con proposals de cualquier escala/aspect ratio sin warping previo.

## Multi-task loss

La función de pérdida es una de las contribuciones conceptuales del paper:

$$
L(p, u, t^u, v) = L_{\text{cls}}(p, u) + \lambda \, [u \geq 1] \, L_{\text{loc}}(t^u, v)
$$

donde:

- $p = (p_0, \dots, p_K)$ es la distribución softmax sobre $K+1$ clases (índice 0 = background).
- $u \in \{0, 1, \dots, K\}$ es la clase ground-truth (0 = background).
- $t^u = (t^u_x, t^u_y, t^u_w, t^u_h)$ son los offsets predichos por la cabeza de regresión para la clase $u$.
- $v = (v_x, v_y, v_w, v_h)$ son los targets de regresión ground-truth.
- $\lambda$ balancea las dos pérdidas. En todos los experimentos $\lambda = 1$.
- $[u \geq 1]$ es la función indicadora de Iverson: vale 1 si $u \geq 1$ (foreground) y 0 si $u = 0$ (background). Esto desactiva la pérdida de regresión para RoIs de fondo, donde no hay ground-truth bbox.

### Classification loss

$L_{\text{cls}}(p, u) = -\log p_u$ es el log-loss estándar (negative log-likelihood) sobre la clase verdadera.

### Localization loss: Smooth L1

$$
L_{\text{loc}}(t^u, v) = \sum_{i \in \{x, y, w, h\}} \text{smooth}_{L_1}(t^u_i - v_i)
$$

con

$$
\text{smooth}_{L_1}(x) = \begin{cases} 0.5 x^2 & \text{si } |x| < 1 \\ |x| - 0.5 & \text{en otro caso} \end{cases}
$$

**¿Por qué smooth L1 y no L2?** Cuando los targets son outliers (RoIs muy desalineados con su GT), la pérdida L2 produce gradientes que escalan linealmente con el error y pueden explotar, exigiendo un tuning fino del learning rate. Smooth L1 satura linealmente en $|x| - 0.5$ fuera del rango $|x| < 1$, lo que la vuelve **robusta a outliers**: el gradiente es acotado en $\pm 1$. R-CNN y SPP-Net usaban L2 y requerían cuidado con el lr.

Esta función es funcionalmente la **Huber loss** con $\delta = 1$, y se vuelve estándar en toda la familia R-CNN y sus descendientes (Faster R-CNN, Mask R-CNN, YOLO v2/v3, RetinaNet).

### Parametrización de los targets

Siguiendo R-CNN, $t$ y $v$ se parametrizan como **scale-invariant translations** y **log-space height/width shifts** relativos al proposal $P = (P_x, P_y, P_w, P_h)$:

$$
t_x = (G_x - P_x) / P_w, \quad t_y = (G_y - P_y) / P_h
$$
$$
t_w = \log(G_w / P_w), \quad t_h = \log(G_h / P_h)
$$

donde $G$ es el bbox ground-truth. Los targets $v_i$ se normalizan a media cero y varianza unitaria sobre el dataset, lo que estabiliza el entrenamiento.

## Training details

### Mini-batch sampling: image-centric vs RoI-centric

Este es uno de los trucos clave del paper para el speed-up. SPP-Net y R-CNN samplean 128 RoIs de 128 imágenes distintas (RoI-centric), lo que obliga a 128 forward passes parciales por mini-batch. Fast R-CNN propone **image-centric sampling**: sólo $N = 2$ imágenes por mini-batch, con $R/N = 64$ RoIs por imagen, totalizando $R = 128$ RoIs por mini-batch.

Esto significa **dos forward passes** del backbone por mini-batch (uno por imagen), y todas las 64 RoIs de cada imagen comparten el feature map. El speed-up resultante es aproximadamente **64×** sobre el sampling de SPP-Net/R-CNN.

**¿No corrobora esto el problema de correlación entre RoIs de la misma imagen?** En teoría, RoIs correlacionados podrían slowar la convergencia del SGD. En la práctica, Girshick reporta que esto no es un problema: Fast R-CNN converge con **menos iteraciones que R-CNN**, no más.

### Selección de RoIs: foreground/background ratio e implicit hard mining

De los 64 RoIs por imagen:

- **25% son foreground**: RoIs con $\text{IoU}(\text{RoI}, \text{GT}) \geq 0.5$. Se asignan a la clase del GT con mayor overlap.
- **75% son background**: RoIs con $\text{IoU} \in [0.1, 0.5)$. Se etiquetan como $u = 0$.

El threshold inferior de 0.1 **no es accidental**: actúa como un **heuristic for hard negative mining**. Las RoIs con IoU < 0.1 (background "fácil") se descartan porque saturan el classifier sin proveer señal útil. Las RoIs en $[0.1, 0.5)$ son los "hard negatives" naturales que confunden al classifier.

Es importante notar que Fast R-CNN **no hace hard negative mining explícito** como hace R-CNN con sus SVMs. El sampling con threshold $[0.1, 0.5)$ es un proxy más simple y suficiente.

### Data augmentation

**Horizontal flipping** con probabilidad 0.5. Nada más. El paper enfatiza que no se usa cropping, color jittering, ni nada extra.

### Hyperparámetros del SGD

- FC layers de las sibling heads inicializadas con Gaussianas $\mathcal{N}(0, 0.01^2)$ (cls) y $\mathcal{N}(0, 0.001^2)$ (loc). Biases en 0.
- Per-layer lr: 1 para pesos, 2 para biases. Global lr: $10^{-3}$.
- VOC07/12 trainval: 30k iteraciones a lr $10^{-3}$, luego 10k a lr $10^{-4}$.
- Datasets más grandes: 60k-100k iteraciones con drop cada 30k-40k.
- Momentum 0.9, weight decay $5 \times 10^{-4}$.

## Optimizaciones

### Truncated SVD para acelerar las FC

En classification image-wide, las FCs son baratas (el grueso del cómputo está en las convs). Pero en detección, con 2000 RoIs pasando por las FCs, **casi la mitad del forward time se gasta en fc6 y fc7**. La Figura 2 del paper muestra que pre-SVD, fc6 toma 38.7% del tiempo (122 ms) y fc7 toma 6.2% (20 ms) del total de 320 ms.

Truncated SVD descompone la matriz de pesos $W \in \mathbb{R}^{u \times v}$ como:

$$
W \approx U \Sigma_t V^T
$$

donde se retienen las $t$ componentes principales: $U \in \mathbb{R}^{u \times t}$, $\Sigma_t \in \mathbb{R}^{t \times t}$, $V \in \mathbb{R}^{v \times t}$. La capa única $W$ se reemplaza por **dos capas sin no-linealidad entre ellas**: primero $\Sigma_t V^T$ (sin bias), luego $U$ (con el bias original). El número de parámetros baja de $uv$ a $t(u+v)$.

**Setup en VGG-16**: fc6 es $25088 \times 4096$; se retiene $t = 1024$. fc7 es $4096 \times 4096$; se retiene $t = 256$. Resultado: tiempo total baja de 320 ms a 223 ms (**30% reducción**) con pérdida de mAP de solo 0.3 puntos (66.9 → 66.6). Sin re-fine-tuning.

### Multi-scale training/testing

Dos enfoques discutidos:

- **Brute-force single-scale**: una sola escala $s = 600$ (length del lado más corto), capped a 1000 en el lado más largo.
- **Image pyramid**: 5 escalas $s \in \{480, 576, 688, 864, 1200\}$ como SPP-Net, sampleadas aleatoriamente en training y usadas todas en test (asignando cada RoI a la escala más cercana a $224^2$ píxeles).

La Tabla 7 muestra que multi-scale aporta solo **+1.3 mAP** (en modelo S) o **+1.5 mAP** (modelo M) a costa de 3-4× tiempo. Conclusión del paper: **las CNNs profundas aprenden invariancia de escala directamente**. Single-scale es el sweet spot. VGG-16 ni siquiera puede correr multi-scale por memoria GPU.

## Resultados

### PASCAL VOC 2007 (Tabla 1)

| Método | Train set | mAP |
|---|---|---|
| SPP-Net BB (VGG16) | 07 \ diff | 63.1 |
| R-CNN BB (VGG16) | 07 | 66.0 |
| **Fast R-CNN (VGG16)** | 07 | **66.9** |
| **Fast R-CNN (VGG16)** | 07 \ diff | **68.1** |
| **Fast R-CNN (VGG16)** | 07+12 | **70.0** |

### PASCAL VOC 2010, 2012 (Tablas 2, 3)

- **VOC10**: 66.1 mAP (12 trainval), 68.8 mAP (07++12). Supera a SegDeepM (67.2) con extra data.
- **VOC12**: 65.7 mAP (12), 68.4 mAP (07++12). Top del leaderboard al momento.

### Training y test time (Tabla 4)

| | Fast R-CNN S | Fast R-CNN M | Fast R-CNN L | R-CNN L | SPP-Net L |
|---|---|---|---|---|---|
| Train time (h) | 1.2 | 2.0 | **9.5** | 84 | 25.5 |
| Test rate (s/im) | 0.10 | 0.15 | **0.32** | 47.0 | 2.3 |
| Test + SVD (s/im) | 0.06 | 0.08 | **0.22** | — | — |
| Test speedup | 98× | 80× | **146×** | 1× | 20× |
| Speedup + SVD | 169× | 150× | **213×** | — | — |
| mAP VOC07 | 57.1 | 59.2 | **66.9** | 66.0 | 63.1 |

Tres logros simultáneos: **9× más rápido en training** que R-CNN, **146× más rápido en test** (213× con SVD), **mAP más alto**.

### MS COCO preliminar

Aunque el paper es 2015 y COCO recién emergía, Girshick reporta una baseline: 35.9 PASCAL-style mAP y 19.7 COCO-style AP. Esta cifra se vuelve baseline obligatoria para el lineage que viene.

## Insights del paper

### SVMs no son mejores que softmax (Sección 5.4)

R-CNN entrenaba SVMs one-vs-rest post-hoc sobre features fc7 porque empíricamente vencían al softmax cuando éste se aprendía conjuntamente. Fast R-CNN reabre la pregunta en su nuevo régimen joint training:

| Método | S | M | L |
|---|---|---|---|
| R-CNN BB (SVM) | 58.5 | 60.2 | 66.0 |
| Fast R-CNN (SVM post-hoc) | 56.3 | 58.7 | 66.8 |
| **Fast R-CNN (softmax joint)** | **57.1** | **59.2** | **66.9** |

Softmax joint-trained **iguala o supera** a SVMs. Diferencias de +0.1 a +0.8 mAP. Esto es un hallazgo conceptual fuerte: el multi-stage SVM "boost" de R-CNN era un artefacto del entrenamiento fragmentado, no de una virtud intrínseca del SVM. Adicionalmente, softmax introduce **competencia entre clases** al normalizar.

### Multi-task helps both tasks (Sección 5.1)

La Tabla 6 desagrega la contribución del multi-task:

| Configuración | S | M | L |
|---|---|---|---|
| Solo $L_{cls}$ (sin bbox reg) | 52.2 | 54.7 | 62.6 |
| Multi-task train, **bbox reg disabled at test** | 53.3 | 55.5 | 63.4 |
| Stage-wise (cls primero, luego bbox reg frozen) | 54.6 | 56.6 | 64.0 |
| **Multi-task full** | **57.1** | **59.2** | **66.9** |

Tres observaciones:
1. Entrenar multi-task **mejora la clasificación pura** (+0.8 a +1.1 mAP, comparando columnas 1 y 2). El gradiente del bbox loss actúa como **regularizador** en el shared backbone.
2. Multi-task supera consistentemente a stage-wise (+1.5 a +3 mAP).
3. La regresión sola sin clasificación joint también es peor.

Este es un argumento canónico a favor del multi-task learning con shared representations (Caruana 1997).

### More RoIs no siempre ayuda (Sección 5.5)

La Figura 3 sweepa de 1k a 10k proposals de Selective Search. mAP **sube y luego cae** ligeramente. Saturating con más proposals **no es free**: introduce más falsos positivos que confunden al classifier.

Más fuerte aún: cuando reemplazas Selective Search por proposals densos uniformes (45k boxes/imagen), mAP **cae de 66.9% a 52.9%** con softmax (49.3% con SVMs). Los sparse proposals actúan como un **cascade** (Viola-Jones-style) que filtra negativos triviales antes del classifier. Esto justifica empíricamente por qué Selective Search era valioso a pesar de su lentitud.

**Average Recall** (AR), métrica común para evaluar proposals, **no correlaciona con mAP** cuando varías el número de proposals. Hay que evaluar mAP directo.

### Fine-tune conv layers es crítico para very deep nets (Sección 4.5)

| Layers fine-tuned (VGG-16) | mAP |
|---|---|
| ≥ fc6 (emula SPP-Net) | 61.4 |
| ≥ conv3_1 | **66.9** |
| ≥ conv2_1 | 67.2 |
| SPP-Net L (5 scales, ≥ fc6) | 63.1 |

Congelar las convs (como hacía SPP-Net) **cuesta 5.5 puntos de mAP** en VGG-16. Fast R-CNN puede fine-tunear las convs porque su sampling image-centric hace que el backward pass por RoI Pool sea computacionalmente viable. Bajar más allá de conv3_1 da rendimientos decrecientes y aumenta el costo (conv2_1 → +30% train time; conv1_1 → out-of-memory).

## Limitaciones reconocibles

1. **Dependencia de Selective Search**: aún requiere un módulo externo de proposals que toma ~2s/imagen en CPU. Eso significa que aunque la CNN procesa en 0.3s, el sistema end-to-end es dominated por SS. **Faster R-CNN** lo resuelve.

2. **Cuantización en RoI Pooling**: las coordenadas del RoI se redondean al proyectar al feature map y al dividir en grilla. Esto introduce misalignment de hasta ~16 píxeles en el espacio de la imagen (con stride 16 de VGG-16). Para clasificación es tolerable; para **segmentación de instancia con mask precisa** es desastroso. **RoIAlign** (Mask R-CNN) lo resuelve.

3. **No es end-to-end real desde la imagen**: aún hay un paso pre-CNN no-aprendido (SS).

4. **Single-scale tiene limitaciones para objetos muy pequeños o muy grandes**: la "brute force" funciona en VOC07/12 pero falla con objetos extremos. Esto motiva los **Feature Pyramid Networks** (Lin et al. 2017).

5. **Memoria GPU**: VGG-16 + 128 RoIs apenas cabe en una K40. Limita los experimentos a single-scale y a fine-tunear desde conv3_1 hacia arriba.

6. **No maneja oclusiones, contexto global, ni relaciones inter-objeto**. Cada RoI se clasifica independientemente.

## Legado

Fast R-CNN es uno de los papers más influyentes de detección de la década 2014-2024. Su impacto se manifiesta en varias direcciones:

### Familia R-CNN directa

- **Faster R-CNN** (Ren, He, Girshick, Sun; NeurIPS 2015, pocos meses después): integra un **Region Proposal Network (RPN)** que comparte el backbone con la cabeza de detección. Elimina Selective Search. mAP 73.2 VOC07, inferencia a 5 fps. Verdaderamente end-to-end.
- **Mask R-CNN** (He, Gkioxari, Dollár, Girshick; ICCV 2017): añade una cabeza paralela de segmentación de instancia. Reemplaza RoI Pooling por **RoIAlign** (bilinear interpolation sin cuantización). Estado del arte en COCO instance segmentation y keypoint detection.
- **R-FCN** (Dai et al., NeurIPS 2016): introduce position-sensitive score maps para eliminar las FCs caras, hace que la cabeza per-RoI sea casi gratuita.
- **Cascade R-CNN** (Cai & Vasconcelos, CVPR 2018): cascade de cabezas con IoU thresholds crecientes (0.5 → 0.6 → 0.7) para refinar progresivamente.
- **Libra R-CNN, Grid R-CNN, Double-Head R-CNN** y otras variantes.

### RoI Pooling como primitiva

El operador RoI Pool (y su sucesor RoIAlign) se vuelve una **primitiva universal** en computer vision. Sus generalizaciones:

- **RoIAlign** (Mask R-CNN 2017): bilinear interpolation en lugar de redondeo. Crítico para segmentación.
- **PrRoI Pool / Precise RoI Pooling** (Jiang et al. ECCV 2018): integración continua sin sampling.
- **Deformable RoI Pooling** (Dai et al. ICCV 2017): añade offsets aprendibles a las sub-ventanas.
- **BezierAlign** (ABCNet, Liu et al. CVPR 2020): generaliza RoIAlign a regiones curvas parametrizadas por Bézier curves. Es la base del scene text spotter de la clase 21.
- **TextAlign** (TextSpotter / Mask TextSpotter): RoIAlign sobre regiones de texto inclinado.
- **RotatedRoIAlign**: para detección de objetos rotados (DOTA, aerial imagery).

### Multi-task con sibling heads

El patrón "shared backbone → per-RoI feature → multiple sibling heads" es **universal** en visión moderna:

- Detección: cls head + bbox reg head.
- Instance segmentation: cls + bbox + mask head (Mask R-CNN).
- Pose estimation: cls + bbox + keypoints head.
- Text spotting: cls + bbox + recognition head (CRNN-style o Transformer).
- DensePose: cls + bbox + UV coordinates head.

### Smooth L1 / Huber loss

Smooth L1 se vuelve estándar en regresión de bounding boxes en **toda la familia** R-CNN, YOLO v2/v3, SSD, RetinaNet, etc. Sólo recientemente competidores como **IoU loss**, **GIoU**, **DIoU**, **CIoU** (Zheng et al. 2020) han tomado terreno por su mejor alineación con la métrica de evaluación.

### Truncated SVD para inferencia

La idea de comprimir FCs con SVD low-rank se vuelve estándar en model compression. Aparece después en knowledge distillation, pruning estructurado, y en hardware-aware NAS.

## Conexión con la clase 21 (Scene Text Spotting)

La clase 21 del curso aborda el problema de **Scene Text Recognition (STR)** y **end-to-end text spotting**, donde Fast R-CNN es un antecedente directo y necesario.

### ABCNet (Liu et al., CVPR 2020)

ABCNet ("Real-time Scene Text Spotting with Adaptive Bezier-Curve Network") es la referencia principal de la clase. ABCNet genera proposals como **Bézier curves** de orden 3 (8 puntos de control) para representar texto curvo, y luego usa **BezierAlign** —una generalización directa de RoIAlign, que a su vez es una generalización de RoI Pooling— para extraer features de las regiones de texto. La cabeza de reconocimiento es un CRNN sobre los features bezier-aligned.

El paper ABCNet referencia explícitamente la familia R-CNN: "Following the success of two-stage detectors like Fast/Faster R-CNN and Mask R-CNN, recent text spotters adopt the share-backbone + RoI-extract + per-RoI heads paradigm." Fast R-CNN es **la genealogía** de ABCNet.

### Mask TextSpotter (Lyu et al., ECCV 2018; PAMI 2021)

Construido directamente sobre **Mask R-CNN**, que a su vez es Fast R-CNN + RPN + mask head + RoIAlign. Detecta texto con masks (no boxes) y reconoce con una cabeza adicional de secuencia carácter por carácter.

### TextBoxes / TextBoxes++ (Liao et al., AAAI 2017, TIP 2018)

Variante de SSD con anchors largos especializados en texto. Aunque SSD es single-stage y no two-stage, comparte el patrón de "shared backbone + per-anchor sibling heads (cls + bbox reg)" que Fast R-CNN cristalizó.

### CharNet, FOTS, EAST, PAN, CRAFT, DRRG, DBNet

Toda la línea de scene text detectors/spotters comparte ADN con Fast R-CNN:

- Backbone CNN compartido.
- Pooling/alignment operator que extrae features de regiones de tamaño/forma variable a tensores fijos.
- Múltiples heads sibling (detección + reconocimiento).
- Multi-task loss con balancing weights.
- Smooth L1 en regresión de geometría.

### Por qué importa el patrón

Para Roberto y para el contexto del curso, el insight central es que **Fast R-CNN definió un patrón arquitectónico que sobrevive ~10 años** y se adapta a cada nuevo problema: detección genérica, segmentación, pose, texto, scene graphs. Los detectores modernos basados en Transformers (DETR, Deformable DETR, DINO, MaskFormer) reemplazan las RoIs explícitas por queries aprendibles, pero conservan la idea de "shared encoder + multiple parallel heads with multi-task loss". Es la idea, no la implementación específica, lo que perdura.

Para un practitioner de visión moderna en 2026, leer Fast R-CNN es indispensable porque te entrega:

1. **El vocabulario**: RoI, sibling heads, multi-task loss, image-centric sampling, smooth L1.
2. **Las trade-offs**: shared computation vs per-region computation, sparse vs dense proposals, single-stage vs multi-stage training.
3. **La intuición**: por qué softmax joint-trained gana a SVMs, por qué multi-task ayuda a ambas tareas, por qué fine-tunear convs profundas importa.
4. **La genealogía**: cuando lees un paper de 2024 sobre transformers para detección, los autores asumen que conoces Fast R-CNN. Sin esa base, no entiendes qué problema están resolviendo.

Fast R-CNN no es el detector que usarías hoy en producción (usarías YOLO v8, DINO, o un foundation model). Pero es el paper donde se cristalizan las ideas que todos esos métodos modernos siguen usando, ya sea explícita o implícitamente. En el árbol genealógico de los detectores, Fast R-CNN es el tronco donde se bifurcan casi todas las ramas relevantes para Scene Text y más allá.

## Apéndice: notas técnicas adicionales

### Por qué image-centric sampling permite el speed-up: análisis de complejidad

Un detalle que el paper menciona pero merece desarrollo es la matemática del speed-up por image-centric sampling. Sea $T_{\text{fwd}}(I)$ el tiempo de forward pass del backbone para una imagen $I$, y $T_{\text{RoI}}$ el tiempo de procesar una RoI (RoI pool + FCs + heads). Bajo RoI-centric sampling (R-CNN/SPP-Net), un mini-batch de $R = 128$ RoIs muestreadas de $R$ imágenes distintas cuesta:

$$
T_{\text{RoI-centric}} = R \cdot T_{\text{fwd}}(I) + R \cdot T_{\text{RoI}}
$$

Con VGG-16, $T_{\text{fwd}} \approx 140$ ms domina sobre $T_{\text{RoI}} \approx 1$ ms, así que $T_{\text{RoI-centric}} \approx 128 \times 140 = 17.9$ s por mini-batch.

Bajo image-centric con $N = 2$ imágenes y $R/N = 64$ RoIs por imagen:

$$
T_{\text{image-centric}} = N \cdot T_{\text{fwd}}(I) + R \cdot T_{\text{RoI}} \approx 2 \times 140 + 128 \times 1 = 408 \text{ ms}
$$

Ratio: $17900 / 408 \approx 44\times$. El paper reporta ~64× empíricamente, que coincide en orden de magnitud (la diferencia viene del overhead de RoI pool y memoria). Este es exactamente el speed-up que faculta el fine-tuning de las conv layers en VGG-16: sin él, propagar gradientes a través del backbone sería computacionalmente prohibitivo.

### Detalles de la proyección RoI → feature map

La proyección de coordenadas de imagen a coordenadas de feature map implica dividir por el stride efectivo del backbone. Para VGG-16, las 4 max-pool layers de stride 2 dan un stride total de $2^4 = 16$. Un RoI $(x_1, y_1, x_2, y_2)$ en píxeles de imagen se proyecta a:

$$
(x_1', y_1', x_2', y_2') = (\lfloor x_1 / 16 \rfloor, \lfloor y_1 / 16 \rfloor, \lceil x_2 / 16 \rceil, \lceil y_2 / 16 \rceil)
$$

El uso de floor para el corner top-left y ceil para el bottom-right inflama ligeramente la región, lo que es preferible a perderla. Esta cuantización en stride 16 introduce un error de hasta 15 píxeles por lado, lo que para objetos pequeños es significativo. He et al. en Mask R-CNN documentan que para masks de 28×28 sobre objetos pequeños, esta cuantización degrada AP por varios puntos. RoIAlign elimina ambos roundings (proyección y división en grilla) usando aritmética en punto flotante y bilinear interpolation.

### Comparación cuantitativa Smooth L1 vs L2

Considera un outlier con $|t - v| = 10$. La pérdida L2 acumula $0.5 \times 100 = 50$ con gradiente $10$ (que multiplica al lr). La pérdida Smooth L1 acumula $10 - 0.5 = 9.5$ con gradiente saturado en $1$ (signo). Para 128 RoIs en un mini-batch, si una sola es un outlier, en L2 esa RoI domina el gradiente; en Smooth L1, contribuye igual que las demás. Esto es por qué Smooth L1 **no requiere gradient clipping** mientras que L2 sí.

### El rol de los $K$ regresores específicos por clase

Una decisión sutil del paper: la cabeza de regresión produce $4K$ outputs (no $4$). Es decir, **un regresor de 4 coords por cada clase**. En training sólo se penaliza el regresor de la clase ground-truth. En test, se usa el regresor de la clase con mayor score.

¿Por qué class-specific? Porque los offsets óptimos dependen de la geometría típica de cada clase: un bicicleta tiene aspect ratio horizontal, una persona vertical, un avión muy alongated. Compartir un solo regresor entre todas las clases promediaría estos patterns y degradaría la precisión. El costo en parámetros es modesto ($4K \times 4096$ adicionales para la última FC, ~330k params para $K = 20$ en VOC).

### Comparación con OverFeat

OverFeat (Sermanet et al., ICLR 2014) anticipó algunas ideas de Fast R-CNN: forward pass único sobre la imagen y reuso de features. Pero OverFeat usaba sliding window denso (no proposals), entrenamiento por etapas (classification → regression), y no tenía multi-task loss joint. Fast R-CNN explícitamente compara y supera a OverFeat en VOC.

### NMS en post-processing

El paper menciona brevemente que en inferencia se aplica **non-maximum suppression (NMS)** por clase con los settings de R-CNN: IoU threshold 0.3 para suprimir. NMS no es aprendido y es un cuello de botella tradicional. Variantes posteriores como **Soft-NMS** (Bodla et al., ICCV 2017) y **Learnable NMS** (Hosang et al., CVPR 2017) intentan mejorarlo. En arquitecturas modernas tipo DETR, el matching bipartito con Hungarian algorithm elimina la necesidad de NMS explícito.

### Lectura comparada con la era pre-deep

Para apreciar la magnitud del salto, vale comparar con DPM y el paradigma pre-2012:

| Aspecto | DPM (2010) | R-CNN (2014) | Fast R-CNN (2015) |
|---|---|---|---|
| Features | HOG artesanal | CNN AlexNet | CNN VGG-16 |
| Localización | Sliding window denso | Selective Search proposals | Selective Search + RoI Pool |
| Classifier | Latent SVM | SVMs post-hoc | Softmax joint |
| Bbox refinement | Implicit en parts | Regresor lineal stage 3 | Smooth L1 head joint |
| Inferencia | ~2-5s | 47s | 0.3s |
| mAP VOC07 | ~33% | 66.0% | 66.9% |
| Training | Latent SVM (días) | 84h | 9.5h |

En cinco años (2010-2015) el mAP en VOC07 dobla y la inferencia es 7× más rápida que DPM. Es el salto generacional que demuestra el poder del deep learning bien orquestado.

### Reproducibilidad y código abierto

Una contribución no técnica pero importante: Fast R-CNN se publicó con **código en Python y C++ (Caffe)** bajo MIT License en `https://github.com/rbgirshick/fast-rcnn`. Esto aceleró la adopción por la comunidad y permitió que Faster R-CNN, Mask R-CNN, y muchos otros se construyeran encima sin reimplementar desde cero. La cultura de open-source en visión moderna se cimenta en parte en estos releases de Girshick y colaboradores.
