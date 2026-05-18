# PifPaf: Composite Fields for Human Pose Estimation

**Autores:** Sven Kreiss, Lorenzo Bertoni, Alexandre Alahi (EPFL VITA lab, Lausanne)
**Año:** 2019 (CVPR)
**arXiv:** 1903.06593
**Código:** https://github.com/vita-epfl/openpifpaf (Apache-2.0, mantenido como `openpifpaf`)

---

## 1. Contexto histórico

En 2019 la **estimación de pose multi-persona** estaba dominada por dos familias:

- **Top-down** — Mask R-CNN (He et al., 2017), CPN (Chen et al., 2018), MSRA Simple Baseline (Xiao et al., 2018). Detectan personas primero, luego pose dentro de cada bbox. SOTA en alta resolución pero **fallan en oclusión y resoluciones bajas**.
- **Bottom-up** — OpenPose / Part Affinity Fields (Cao et al., 2017), DeepCut (Pishchulin et al., 2016), Associative Embedding (Newell et al., 2017), PersonLab (Papandreou et al., 2018). Detectan partes del cuerpo y las agregan. **No dependen del person detector** y manejan oclusión mejor, pero suelen ser más lentos en alta resolución.

**El nicho que PifPaf ataca**: pose estimation en **resoluciones bajas (30-90 px de altura)** y **escenas urbanas multitudinarias** — el escenario de **coches autónomos y robots sociales**. Los autores son del laboratorio VITA de EPFL, que se enfoca en movilidad urbana inteligente.

OpenPose y Mask R-CNN, sus baselines, **fallan dramáticamente** a 321 px de lado mayor de imagen (que emula el detalle de un crop 4K de la cámara de un auto). En particular, OpenPose pierde recall porque sus PAFs son discretos (anclados a píxeles enteros del feature map), y Mask R-CNN se confunde cuando las bboxes de peatones se intersectan.

## 2. Contribución central

Tres aportes:

1. **Part Intensity Field (PIF)** — *campo escalar + vectorial + escala* por keypoint. Genera mapas de confianza de **alta resolución** vía una **fusión Gaussian-aware** de la salida de la red, **superando la cuantización del feature map**.

2. **Part Association Field (PAF)** — generalización de los PAF de OpenPose. En vez de codificar la asociación como un *campo vectorial discreto anclado al feature map*, PifPaf codifica el origen como un **vector flotante** (mid-range offsets) + dos vectores a los keypoints conectados + spreads $b$ de la distribución Laplace de la regresión. Esto permite asociaciones de larga distancia con precisión sub-píxel.

3. **Laplace loss** para la regresión vectorial — alternativa al $L_1$ vainilla y al SmoothL1. Aprende la incertidumbre de cada predicción y mejora consistentemente el AP en COCO.

Resultado headline: en COCO keypoint task **a baja resolución (321 px lado mayor)**:

| Método | AP | AP₅₀ | AP^M | AP^L | AR |
|---|---|---|---|---|---|
| Mask R-CNN (re-trained) | 41.6 | 68.1 | 28.2 | 59.8 | 49.0 |
| OpenPose | 37.6 | 62.5 | 25.0 | 55.3 | 43.9 |
| **PifPaf** | **50.0** | **73.5** | **35.9** | **69.7** | **55.0** |

Es decir: +8 AP sobre el siguiente mejor método **bottom-up** y +12 AP sobre OpenPose en el régimen donde más importa para self-driving.

A alta resolución (COCO normal), PifPaf empata con el SOTA (PersonLab) y es **9.5% más rápido** en runtime.

## 3. Arquitectura

### 3.1 Backbone y decoder

- **Encoder**: ResNet50/101/152 sin la última pool y modificado para preservar resolución. Reemplazan la convolución inicial 7×7 stride 2 + maxpool por una convolución sub-pixel (Shi et al., 2016) que **duplica la resolución espacial** del feature map de salida.
- **Cabezas**: dos ramas paralelas, ambas **convolución 1×1** sobre los features del encoder.
  - **PIF head**: produce $17 \times 5$ canales (un PIF por keypoint COCO).
  - **PAF head**: produce $19 \times 7$ canales (un PAF por cada una de las 19 conexiones esqueléticas estándar de COCO).

### 3.2 Notación de campos

Un *field* es $\mathbf{f}^{ij}$ sobre la grilla del feature map $(i, j) \in \mathbb{Z}_+^2$. Para un PIF en la posición $(i, j)$ y keypoint $k$:

$$
\mathbf{p}_k^{ij} = \{p_c^{ij},\ p_x^{ij},\ p_y^{ij},\ p_b^{ij},\ p_\sigma^{ij}\}
$$

donde:
- $p_c$ — confianza escalar (sigmoid).
- $(p_x, p_y)$ — vector hacia la posición precisa del keypoint (regresión sub-pixel).
- $p_b$ — spread (escala) de la Laplace para el componente regressivo.
- $p_\sigma$ — escala del keypoint en píxeles.

### 3.3 Mapa de confianza fusionado (Ecuación 1)

La predicción cruda de $p_c$ está cuantizada a la grilla del feature map. Para localizar con precisión sub-pixel, **fusionan** confianza y posición regresada:

$$
f(x, y) = \sum_{ij} p_c^{ij} \cdot \mathcal{N}(x, y \mid p_x^{ij}, p_y^{ij}, p_\sigma^{ij})
$$

donde $\mathcal{N}$ es una Gaussiana no normalizada con desviación $p_\sigma$. Esto convierte el feature map escaso en un mapa continuo y suave, dando localización sub-pixel a partir de un feature map de baja resolución.

### 3.4 PAF — composite vectorial (Sección 3.3)

Para cada una de las 19 conexiones esqueléticas $(k_1, k_2)$ y posición $(i, j)$:

$$
\mathbf{a}_{(k_1,k_2)}^{ij} = \{a_c^{ij},\ a_{x_1}^{ij}, a_{y_1}^{ij},\ a_{x_2}^{ij}, a_{y_2}^{ij},\ a_{b_1}^{ij},\ a_{b_2}^{ij}\}
$$

- $a_c$ — confianza de la conexión.
- $(a_{x_1}, a_{y_1})$ — vector flotante al primer keypoint.
- $(a_{x_2}, a_{y_2})$ — vector flotante al segundo keypoint.
- $(a_{b_1}, a_{b_2})$ — spreads Laplace.

**Diferencia clave vs. OpenPose**: el origen del vector PAF en PifPaf es flotante (mid-range offset learned), no anclado al centro de la celda del feature map. Esto resuelve dos personas adyacentes sin colisión de las anotaciones.

Durante entrenamiento, el ground-truth para PAF se construye así: para cada celda $(i, j)$ en una pose ground-truth, **el primer vector debe apuntar al keypoint más cercano** de la conexión (tipo $k_1$ o $k_2$ — la red infiere cuál); **el segundo vector** apunta al otro keypoint de la conexión, **incluso si está lejos**.

### 3.5 Adaptive Regression Loss — Laplace loss (Ecuación 2)

Para los componentes vectoriales:

$$
L = \frac{|x - \mu|}{b} + \log(2b)
$$

donde $x$ es la predicción, $\mu$ el target y $b > 0$ el spread predicho. Esta es la **log-verosimilitud negativa** de una distribución Laplace con escala aprendida. Equivale a $L_1$ ponderado dinámicamente por la confianza del modelo en cada predicción.

**Por qué importa**: en COCO, un error de 5 px es minor en un cuerpo grande (480 px de altura) y catastrófico en uno pequeño (50 px). El modelo aprende a relajar $b$ para predicciones en cuerpos grandes y a apretarla para cuerpos pequeños — equivalente a *learned uncertainty*.

Ablation (Tabla 3 del paper):

| Loss | AP | AP^M | AP^L |
|---|---|---|---|
| vanilla $L_1$ | 41.7 | 26.5 | 62.5 |
| SmoothL1, $r=0.2\sqrt{A_i}\sigma_k$ | 42.0 | 26.9 | 62.6 |
| **Laplace** | 45.1 | 31.4 | 64.0 |
| **Laplace (b en decoder)** | **45.5** | **31.4** | **64.9** |

Laplace +3.5 AP sobre vanilla L1. Y usar $b$ también en el decoder (greedy decoding) suma otro +0.4.

### 3.6 Greedy decoding (Sección 3.5)

Convierte los campos PIF + PAF en poses concretas:

1. **Seed**: encuentra el máximo del mapa de confianza fusionado $f(x, y)$ — el keypoint con mayor confianza.
2. **Crecimiento greedy**: desde el seed, sigue los PAF para encontrar el keypoint conectado. Para una posición $\vec{x}$ y un candidato a conexión $\mathbf{a}$, computa score:

$$
s(\mathbf{a}, \vec{x}) = a_c \cdot \exp\!\left(-\frac{\|\vec{x} - \vec{a}_1\|_2}{b_1}\right) \cdot f_2(a_{x_2}, a_{y_2})
$$

donde el primer factor es la confianza, el segundo penaliza desalineación con la pose actual ($\vec{a}_1$ es la posición esperada del primer keypoint), y el tercer factor $f_2$ es la confianza fusionada en el destino.

3. Toma la mejor conexión, agrega el nuevo keypoint y **repite**. Una vez agregada una conexión, **no se revisa**.
4. Aplica **reverse matching**: verifica que desde el nuevo keypoint el PAF también apunte hacia el origen — descarta connections inconsistentes.
5. **Non-Maximum Suppression dinámico** en keypoints: el radio de NMS se ajusta dinámicamente según $p_\sigma$ — keypoints en cuerpos grandes tienen radios más amplios que en cuerpos pequeños.

### 3.7 Velocidad

En GPU GTX 1080Ti, ResNet101, imagen completa de COCO val:
- Tiempo total: **240 ms** por imagen.
- Tiempo de decoding: **175 ms**.
- Comparado con PersonLab: **9.5% mejor AP, 32% más rápido**.

## 4. Ejemplos de código (PyTorch / TF / JAX)

### 4.1 Cabezas PIF + PAF (PyTorch)

```python
import torch
import torch.nn as nn


class CompositeHead(nn.Module):
    """Single 1x1 conv producing C channels per location.

    For PIF (per-keypoint): C = num_keypoints * 5
    For PAF (per-connection): C = num_connections * 7
    """

    def __init__(self, in_channels: int, num_fields: int,
                 components_per_field: int):
        super().__init__()
        self.head = nn.Conv2d(in_channels,
                              num_fields * components_per_field,
                              kernel_size=1)
        self.num_fields = num_fields
        self.cpf = components_per_field

    def forward(self, x):
        out = self.head(x)
        b, _, h, w = out.shape
        return out.view(b, self.num_fields, self.cpf, h, w)


class PifPafModel(nn.Module):
    def __init__(self, backbone, in_channels=2048,
                 num_keypoints=17, num_connections=19):
        super().__init__()
        self.backbone = backbone
        self.pif = CompositeHead(in_channels, num_keypoints, 5)
        self.paf = CompositeHead(in_channels, num_connections, 7)

    def forward(self, image):
        f = self.backbone(image)
        pif = self.pif(f)
        paf = self.paf(f)
        return pif, paf  # (B, 17, 5, H/8, W/8), (B, 19, 7, H/8, W/8)
```

### 4.2 Laplace loss (PyTorch)

```python
def laplace_loss(x_pred: torch.Tensor, x_gt: torch.Tensor,
                 b_pred: torch.Tensor, eps: float = 1e-3):
    """
    x_pred, x_gt : (N, 2)   regresion de offset
    b_pred       : (N, 1)   spread, debe ser >0
    L = |x_pred - x_gt|_1 / b + log(2*b)
    """
    b = torch.clamp(b_pred, min=eps)
    abs_err = torch.abs(x_pred - x_gt).sum(dim=-1, keepdim=True)
    return (abs_err / b + torch.log(2 * b)).mean()


def pif_loss(pif_pred, pif_target, mask):
    """
    pif_pred  : (B, K, 5, H, W) con [c, x, y, b, sigma]
    pif_target: (B, K, 5, H, W)  ground-truth construido por el target builder
    mask      : (B, K, H, W) bool donde aplicar la regresion
    """
    p_c = pif_pred[:, :, 0]            # (B, K, H, W)
    t_c = pif_target[:, :, 0]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(p_c, t_c,
                                                                reduction="mean")
    if mask.sum() == 0:
        return bce
    # extrae solo posiciones validas
    p_xy = pif_pred[:, :, 1:3].permute(0, 1, 3, 4, 2)[mask]      # (N, 2)
    t_xy = pif_target[:, :, 1:3].permute(0, 1, 3, 4, 2)[mask]
    p_b  = pif_pred[:, :, 3:4].permute(0, 1, 3, 4, 2)[mask]      # (N, 1)
    reg = laplace_loss(p_xy, t_xy, p_b)
    # tambien una regresion L1 sobre sigma
    p_s  = pif_pred[:, :, 4][mask]
    t_s  = pif_target[:, :, 4][mask]
    scale = torch.nn.functional.smooth_l1_loss(p_s, t_s)
    return bce + reg + scale
```

### 4.3 Fusión de confianza Gaussiana (NumPy / cualquier framework)

```python
import numpy as np


def fused_confidence_map(pif: np.ndarray, stride: int,
                         out_h: int, out_w: int) -> np.ndarray:
    """
    pif : (K, 5, H_low, W_low) con canales [c, dx, dy, b, sigma]
    stride : factor de upsampling (e.g. 8)
    returns: (K, out_h, out_w) mapa fusionado f(x, y)
    """
    K, _, h_lo, w_lo = pif.shape
    f = np.zeros((K, out_h, out_w), dtype=np.float32)
    xs = np.arange(out_w)[None, :]
    ys = np.arange(out_h)[:, None]

    for k in range(K):
        for j in range(h_lo):
            for i in range(w_lo):
                c = pif[k, 0, j, i]
                if c < 0.1:
                    continue
                # posicion absoluta predicha
                px = i * stride + pif[k, 1, j, i]
                py = j * stride + pif[k, 2, j, i]
                sigma = max(pif[k, 4, j, i], 1.0)
                gauss = np.exp(-((xs - px) ** 2 + (ys - py) ** 2)
                               / (2 * sigma ** 2))
                f[k] += c * gauss
    return f
```

### 4.4 Cabezas PIF + PAF (TensorFlow 2 / Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, Model


def composite_head(num_fields, components_per_field, name):
    return layers.Conv2D(num_fields * components_per_field, 1, name=name)


def build_pifpaf(backbone: Model, num_keypoints=17, num_connections=19):
    image_in = backbone.input
    feats = backbone.output  # (B, H/8, W/8, C)
    pif = composite_head(num_keypoints, 5, "pif")(feats)
    paf = composite_head(num_connections, 7, "paf")(feats)
    return Model(image_in, [pif, paf], name="pifpaf")


def laplace_loss_tf(x_pred, x_gt, b_pred, eps=1e-3):
    b = tf.maximum(b_pred, eps)
    abs_err = tf.reduce_sum(tf.abs(x_pred - x_gt), axis=-1, keepdims=True)
    return tf.reduce_mean(abs_err / b + tf.math.log(2.0 * b))
```

### 4.5 Cabezas PIF + PAF (JAX / Flax)

```python
import jax.numpy as jnp
import flax.linen as nn


class CompositeHead(nn.Module):
    num_fields: int
    components_per_field: int

    @nn.compact
    def __call__(self, x):
        out = nn.Conv(self.num_fields * self.components_per_field,
                      (1, 1))(x)
        b, h, w, _ = out.shape
        return out.reshape(b, h, w, self.num_fields, self.components_per_field)


class PifPaf(nn.Module):
    backbone: nn.Module
    num_keypoints: int = 17
    num_connections: int = 19

    @nn.compact
    def __call__(self, image):
        feats = self.backbone(image)
        pif = CompositeHead(self.num_keypoints, 5)(feats)
        paf = CompositeHead(self.num_connections, 7)(feats)
        return pif, paf


def laplace_loss_jax(x_pred, x_gt, b_pred, eps=1e-3):
    b = jnp.maximum(b_pred, eps)
    abs_err = jnp.sum(jnp.abs(x_pred - x_gt), axis=-1, keepdims=True)
    return jnp.mean(abs_err / b + jnp.log(2.0 * b))
```

### 4.6 Greedy decoder pseudocódigo (framework-independiente)

```python
def greedy_decode(pif, paf, fused_conf, connections,
                  conf_threshold=0.1):
    """
    pif        : (K, 5, H, W)
    paf        : (C, 7, H, W) con C = num_connections
    fused_conf : (K, H_full, W_full)  mapa fusionado precomputado
    connections: lista de tuplas (k1, k2) que definen el esqueleto
    """
    poses = []
    # 1. Seeds: top-confidence keypoints sin asignar
    seed_queue = build_seed_queue(fused_conf)

    while seed_queue:
        k, x, y, conf = seed_queue.pop_max()
        if conf < conf_threshold or already_used(x, y, k):
            continue
        pose = {k: (x, y, conf)}

        # 2. BFS por el grafo del esqueleto
        frontier = [k]
        while frontier:
            kc = frontier.pop()
            for (k1, k2) in connections_touching(kc, connections):
                kother = k2 if kc == k1 else k1
                if kother in pose:
                    continue
                # busca el mejor PAF que conecte con kc en (pose[kc])
                xc, yc, _ = pose[kc]
                best = find_best_paf(paf, kc, kother, xc, yc, fused_conf)
                if best is None:
                    continue
                # reverse-match check
                if not reverse_consistent(paf, kother, kc, best):
                    continue
                pose[kother] = best
                frontier.append(kother)

        if len(pose) >= 3:
            poses.append(pose)
            mark_used(pose)

    poses = dynamic_nms(poses, pif)  # radio NMS = funcion de sigma
    return poses
```

## 5. Experimentos clave

### 5.1 Low-resolution COCO (321 px lado mayor)

| Método | AP | AP₅₀ | AP^M | AP^L | AR |
|---|---|---|---|---|---|
| Mask R-CNN* | 41.6 | 68.1 | 28.2 | 59.8 | 49.0 |
| OpenPose | 37.6 | 62.5 | 25.0 | 55.3 | 43.9 |
| **PifPaf** | **50.0** | 73.5 | 35.9 | 69.7 | 55.0 |

\* Mask R-CNN re-entrenado para baja resolución.

### 5.2 High-resolution COCO test-dev

| Método | AP | AP^M | AP^L |
|---|---|---|---|
| Mask R-CNN | 63.1 | 58.0 | 70.4 |
| OpenPose | 61.8 | 57.1 | 68.2 |
| PersonLab (single-scale) | 66.5 | 62.4 | 72.3 |
| **PifPaf (single-scale)** | 66.7 | 62.4 | 72.9 |

A alta resolución empata con PersonLab y supera a OpenPose y Mask R-CNN.

### 5.3 Market-1501 cross-domain Re-ID

Sin re-entrenar, sobre crops 64×128 del benchmark Market-1501:
- Mask R-CNN: 43% de poses correctas (forzado a una pose por bbox).
- **PifPaf: 96%**.

Muestra que el método bottom-up generaliza mucho mejor a dominios cross.

### 5.4 nuScenes (qualitative)

En self-driving real (dataset nuScenes), PifPaf detecta peatones gesticulando que OpenPose y Mask R-CNN omiten — figuras 6-7 del paper documentan esto.

## 6. Limitaciones reconocidas y sutilezas

1. **No optimizado para alta resolución**: el paper se enfoca en baja resolución; en alta-res es solo *a la par* con PersonLab — no significativamente mejor.
2. **Decoder secuencial** (greedy + reverse-match): difícil de batchear en GPU, dominante en el tiempo total.
3. **NMS dinámico depende de $p_\sigma$ aprendido**: si $\sigma$ es ruidoso (poses muy pequeñas), el NMS puede fusionar dos poses cercanas.
4. **Sólo 17 keypoints COCO** — mismo límite topológico que el resto. La crítica del profesor a la elección arbitraria de keypoints sigue aplicando.
5. **Suposición de skeleton fijo**: las 19 conexiones son hardcoded. No se aprende la topología.
6. **Sin temporal**: por frame; tracking temporal requiere postprocesamiento separado (e.g., trajectories).

## 7. Impacto y legado

- **openpifpaf** (https://github.com/vita-epfl/openpifpaf) ha sido mantenido continuamente desde 2019, agregando soporte para animales, manos, faciales, vehículos y tracking. Es uno de los frameworks bottom-up más usados en producción para escenarios constrained.
- La idea de **Laplace loss para regresión vectorial** se ha generalizado a detección de objetos (DETR-uncertainty), object pose 6D y key-value retrieval. El paper estableció la práctica de "predecir tu propia incertidumbre como parte del loss".
- **Composite Fields** (PIF + PAF unificados) — concepto que aparece en MoveNet (Google MediaPipe Pose), Lightweight OpenPose, BlazePose.
- La pipeline de **autonomous-driving + pose** se popularizó: hoy es estándar incluir pose de peatones (gesture-aware) en stacks de auto-conducción.
- **HigherHRNet** (Cheng et al., CVPR 2020) cita PifPaf como baseline bottom-up; **DEKR** (Geng et al., CVPR 2021) y **CID** (Wang et al., CVPR 2022) continúan la línea de centro-vectores.

## 8. Conexión con la clase 17

PifPaf es el ejemplo **canónico de aproximación bottom-up** que el profesor presenta en los slides 37-46 del PDF. Su importancia:

- Demuestra que **detectar partes y luego ensamblar** funciona — y supera al top-down precisamente donde top-down sufre (oclusiones, bboxes intersectados, baja resolución).
- Concretiza las dos componentes que el profesor menciona: **PIF (intensity) + PAF (affinity)**.
- Es el ejemplo que justifica la sección **"¿Por qué Bottom-up > Top-down?"** — el caso del baseball con oclusión y bbox intersectada (slide 48) es exactamente el caso de uso donde PifPaf brilla.
- A la inversa, su debilidad (baja resolución) abre el camino hacia ViTPose, que recupera la precisión top-down con backbones modernos.

Cross-links:
- [[fundamentos/pose-estimation.md]] — bottom-up vs. top-down formal.
- [[papers/DensePose-Guler-2018.md]] — la alternativa top-down de la misma clase.
- [[papers/ViTPose-Xu-2022.md]] — el siguiente paradigma.
- [[clases/clase-17/teoria.md#pifpaf]] — sección de la clase.

## 9. Enlaces

- Paper: https://arxiv.org/abs/1903.06593
- Código: https://github.com/vita-epfl/openpifpaf
- Demo: https://openpifpaf.github.io/
