# DensePose: Dense Human Pose Estimation In The Wild

**Autores:** Rıza Alp Güler (INRIA-CentraleSupélec), Natalia Neverova (Facebook AI Research), Iasonas Kokkinos (Facebook AI Research)
**Año:** 2018 (CVPR)
**arXiv:** 1802.00434
**Project page:** http://densepose.org

---

## 1. Contexto histórico

Cuando se publicó DensePose en febrero de 2018, la **estimación de pose humana** estaba dominada por aproximaciones basadas en **keypoints discretos**: detectar 14-17 puntos articulados (nariz, ojos, hombros, codos, muñecas, caderas, rodillas, tobillos). Los benchmarks principales eran MPII y COCO Keypoint Detection, y las arquitecturas dominantes incluían DeepPose (Toshev y Szegedy, 2014), Convolutional Pose Machines (Wei et al., 2016), Stacked Hourglass (Newell et al., 2016) y Mask R-CNN (He et al., 2017) con su rama de keypoints.

Esta representación, aunque útil, es **fundamentalmente incompleta**. El cuerpo humano es una superficie continua de aproximadamente 7000 vértices cuando se discretiza con el modelo SMPL; representarlo con 17 puntos pierde toda la información sobre orientación, deformación local, contacto con objetos y forma de las partes intermedias entre articulaciones.

El campo de **dense correspondence** (correspondencia densa) había explorado:

- **DenseReg** (Güler et al., CVPR 2017) — la pieza inmediatamente anterior del mismo grupo. Establecía correspondencia densa para caras, con la cara como variedad 2D parametrizable.
- **Vitruvian Manifold** (Taylor et al., 2012) — requería sensor de profundidad.
- **Metric Regression Forests** (Pons-Moll et al., 2013).
- **Unite the People (UP)** (Lassner et al., 2017) — ajustaba el modelo SMPL semi-automáticamente a 8515 imágenes, pero los ajustes fallaban con oclusiones y poses extremas.
- **SURREAL** (Varol et al., 2017) — datos sintéticos renderizados con SMPL desde Mocap CMU. Sufre de *covariate shift* entre imágenes sintéticas y reales.

**El gap que llena DensePose**: un dataset masivo (~50K personas, 5M correspondencias manualmente anotadas) en imágenes **reales** de COCO, combinado con una arquitectura que entrega correspondencia densa imagen↔superficie del cuerpo a 20-26 fps en imágenes de 240×320 sobre una GTX 1080.

## 2. Contribución central

Tres contribuciones clave:

1. **COCO-DensePose dataset**: 50K personas en COCO con correspondencia manual imagen↔superficie SMPL. Esto se logra mediante una pipeline de anotación en 2 etapas (segmentación de partes + correspondencia de puntos) que recolecta ~100-150 puntos por persona a velocidad razonable.

2. **DensePose-RCNN**: arquitectura tipo Mask R-CNN extendida con una rama que predice, para cada píxel del foreground humano:
   - Una **etiqueta discreta de parte del cuerpo** $c \in \{0, 1, ..., 24\}$ (24 partes + background).
   - Coordenadas continuas $(U, V) \in [0,1]^2$ dentro de la parametrización local de cada parte.

3. **Distillation-based ground-truth interpolation**: como el ground-truth manual es disperso (sparse) — solo ~100 puntos por persona — entrenan primero una "teacher network" totalmente convolucional con esos puntos, y luego usan sus predicciones como **señal densa** para entrenar el modelo final, restringido a la región de foreground.

## 3. Arquitectura y método

### 3.1 Parametrización del cuerpo: el "UV unwrapping"

El cuerpo humano se divide en **24 partes** semánticas (cabeza, torso frontal/dorsal, brazos sup/inf izq/der, etc.). Cada parte se "desenvuelve" (unwrap) a un plano 2D mediante:

- **Cabeza, manos, pies**: usan los UV fields ya provistos por el modelo SMPL.
- **Resto del cuerpo**: aplican **multidimensional scaling (MDS)** a las distancias geodésicas del mesh para obtener una parametrización 2D isomorfa al plano (cada extremidad se parte en mitades superior/inferior y frontal/dorsal para que sea aproximadamente plana).

Resultado: cada punto de superficie del cuerpo SMPL queda etiquetado con $(c, u, v)$ donde $c$ es la parte y $(u, v) \in [0,1]^2$ es la coordenada local intra-parte.

### 3.2 Pipeline de anotación (Sección 2.1)

Para cada persona en COCO:

- **Task 1 — Segmentación de partes**: el anotador delinea las 14 regiones semánticas (cabeza, torso, etc.) sobre la imagen.
- **Task 2 — Correspondencia**: se sortean ~14 puntos por parte vía k-means sobre la región segmentada. Para cada punto, se le muestran al anotador **6 vistas pre-renderizadas** de la parte del cuerpo (cabeza, mano, etc.) desde el modelo SMPL y se le pide hacer clic en el punto correspondiente.

Esta indirección clave evita pedir al anotador "rotar una superficie 3D para cazar vértices" — en su lugar elige entre vistas pre-renderizadas en 2D. Se anota detrás de la ropa (asumiendo el cuerpo desnudo subyacente).

**Volumen**: 50K personas × ~100-150 puntos = ~5M correspondencias.

### 3.3 Métricas: GPS y AUC

**Geodesic Point Similarity (GPS)** — inspirada en OKS de COCO Keypoints:

$$
\text{GPS}_j = \frac{1}{|P_j|} \sum_{p \in P_j} \exp\!\left(\frac{-g(i_p, \hat{i}_p)^2}{2\kappa^2}\right)
$$

donde $P_j$ son los puntos ground-truth en la instancia $j$, $i_p$ es el vértice estimado y $\hat{i}_p$ el real, $g(\cdot, \cdot)$ es la distancia geodésica sobre la superficie SMPL, y $\kappa = 0.255$ — calibrado para que un error geodésico de ~30 cm (medio segmento de cuerpo) corresponda a $\text{GPS} = 0.5$.

Luego siguen el protocolo COCO: calculan AP y AR a thresholds GPS de 0.5 a 0.95, en pasos de 0.05.

**Métricas pointwise**: AUC$_{10}$ y AUC$_{30}$ — área bajo la curva del Ratio of Correct Points (RCP) hasta 10cm y 30cm.

### 3.4 Arquitectura DensePose-RCNN

Es **Mask R-CNN** con una rama adicional para DensePose. Concretamente (Figuras 7 y 8 del paper):

1. **Backbone**: ResNet-50-FPN o ResNet-101-FPN.
2. **Region Proposal Network (RPN)**: igual que Faster R-CNN.
3. **RoIAlign**: features alineados sub-pixel para cada bbox propuesto.
4. **Cabezas de tareas** (en paralelo):
   - **Clasificación de objeto** (softmax cross-entropy).
   - **Refinamiento de bbox** (smooth L1).
   - **Máscara de instancia** (Mask R-CNN head).
   - **Keypoints** (Mask R-CNN head, 17 puntos COCO).
   - **DensePose head**: stack de 8 capas Conv 3×3 + ReLU con 512 canales, sobre los features RoIAlign. Produce dos outputs:
     - **Logits de parte**: $H \times W \times 25$ — clasificación pixel-wise en background + 24 partes (cross-entropy).
     - **Regresión UV**: $H \times W \times 48$ — coordenadas (u, v) para cada una de las 24 partes (24 × 2 mapas, smooth L1).

### 3.5 Pérdidas

Para cada píxel $i$ etiquetado con parte $c^*$:

$$
c^* = \arg\max_c P(c \mid i), \quad [U, V] = R^{c^*}(i)
$$

- **Pérdida de clasificación**: $\mathcal{L}_{\text{cls}} = -\log P(c^* \mid i)$ (cross-entropy en 25 clases).
- **Pérdida de regresión UV**: smooth L1 sobre las predicciones $R^{c^*}(u, v)$ — **solo se computa si la predicción de parte es correcta**, lo cual evita gradientes ruidosos cuando el modelo aún no sabe qué parte es.

### 3.6 Cascading (Sección 3.3)

Inspirados en refinamiento iterativo, hacen una **segunda pasada** que recibe como entrada los features RoIAlign + los outputs de las ramas auxiliares (máscara, keypoints). Esto inyecta contexto multi-tarea. Sumado, el modelo final usa información de keypoints/máscaras para mejorar la predicción DensePose.

### 3.7 Distillation-based ground-truth interpolation (Sección 3.4)

Solo ~100-150 píxeles por persona están anotados. Si solo se computa pérdida en esos píxeles, el modelo está sub-supervisado. Solución:

1. Entrenar una **teacher network** (FCN, Deeplab-style) con la supervisión escasa.
2. Inferir las predicciones de la teacher sobre toda la región de foreground humano (no solo los puntos anotados).
3. Usar esas predicciones inferidas como **supervisión densa** para entrenar la red region-based final.
4. Solo se conservan los píxeles dentro de la máscara humana — para evitar entrenar contra ruido del fondo.

Esto se denomina **inpainting** del signal de supervisión. La ganancia: AUC$_{10}$ pasa de 0.253 (FCN bare) a 0.381 (DP-RCNN distillation) en multi-person.

## 4. Ejemplos de código (PyTorch / TF / JAX)

A continuación, **componentes clave** simplificados — la cabeza DensePose, las pérdidas y el cómputo de GPS — en los tres frameworks. El código completo del paper está en Detectron, https://github.com/facebookresearch/Detectron .

### 4.1 Cabeza DensePose (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensePoseHead(nn.Module):
    """8x Conv3x3+ReLU -> {part logits (25 ch), UV regression (24*2 ch)}."""

    def __init__(self, in_channels: int = 256, hidden: int = 512,
                 num_parts: int = 24):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(8):
            layers += [nn.Conv2d(c, hidden, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True)]
            c = hidden
        self.feature = nn.Sequential(*layers)
        self.cls_head = nn.Conv2d(hidden, num_parts + 1, kernel_size=1)
        self.uv_head = nn.Conv2d(hidden, num_parts * 2, kernel_size=1)
        self.num_parts = num_parts

    def forward(self, roi_features: torch.Tensor):
        f = self.feature(roi_features)
        part_logits = self.cls_head(f)
        uv = self.uv_head(f)
        b, _, h, w = uv.shape
        uv = uv.view(b, self.num_parts, 2, h, w)
        return part_logits, uv


def densepose_loss(part_logits, uv_pred, part_gt, uv_gt, fg_mask,
                   lambda_uv: float = 1.0):
    """
    part_logits : (B, 25, H, W)   logits incluyendo background = 0
    uv_pred     : (B, 24, 2, H, W)
    part_gt     : (B, H, W)       int64, valores 0..24, ignore=-1
    uv_gt       : (B, 2, H, W)    float [0,1], valido solo donde part_gt>0
    fg_mask     : (B, H, W)       bool, foreground humano
    """
    valid = (part_gt >= 0) & fg_mask
    cls_loss = F.cross_entropy(
        part_logits.permute(0, 2, 3, 1)[valid],
        part_gt[valid], reduction="mean")

    pos = valid & (part_gt > 0)
    if pos.any():
        b_idx, h_idx, w_idx = pos.nonzero(as_tuple=True)
        part_idx = part_gt[b_idx, h_idx, w_idx] - 1  # 0..23
        uv_p = uv_pred[b_idx, part_idx, :, h_idx, w_idx]  # (N, 2)
        uv_t = uv_gt[b_idx, :, h_idx, w_idx]              # (N, 2)
        uv_loss = F.smooth_l1_loss(uv_p, uv_t)
    else:
        uv_loss = uv_pred.new_tensor(0.0)
    return cls_loss + lambda_uv * uv_loss
```

Detalle clave: la regresión UV solo aplica donde la parte ground-truth no es background (`pos = valid & (part_gt > 0)`), y se indexa el canal de la parte correcta — no se penaliza la predicción UV de partes incorrectas.

### 4.2 Cabeza DensePose (TensorFlow 2 / Keras)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_densepose_head(in_channels: int = 256, hidden: int = 512,
                         num_parts: int = 24) -> keras.Model:
    inputs = keras.Input(shape=(None, None, in_channels))
    x = inputs
    for _ in range(8):
        x = layers.Conv2D(hidden, 3, padding="same", activation="relu")(x)
    part_logits = layers.Conv2D(num_parts + 1, 1, name="part_logits")(x)
    uv_flat = layers.Conv2D(num_parts * 2, 1, name="uv")(x)
    return keras.Model(inputs, [part_logits, uv_flat])


def densepose_loss_tf(part_logits, uv_flat, part_gt, uv_gt, fg_mask,
                      num_parts: int = 24, lambda_uv: float = 1.0):
    """
    part_logits : (B, H, W, 25)
    uv_flat     : (B, H, W, 48)  reorganizado a (B, H, W, 24, 2)
    part_gt     : (B, H, W)      int32 con ignore = -1
    uv_gt       : (B, H, W, 2)
    fg_mask     : (B, H, W)      bool
    """
    valid = tf.logical_and(part_gt >= 0, fg_mask)
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.maximum(part_gt, 0),
        logits=part_logits,
    )
    cls_loss = tf.reduce_mean(tf.boolean_mask(cls_loss, valid))

    pos = tf.logical_and(valid, part_gt > 0)
    uv = tf.reshape(uv_flat, tf.concat([tf.shape(uv_flat)[:3],
                                        [num_parts, 2]], axis=0))
    part_idx = tf.maximum(part_gt - 1, 0)
    uv_pred_sel = tf.gather(uv, part_idx, batch_dims=3)  # (B, H, W, 2)

    diff = uv_pred_sel - uv_gt
    abs_d = tf.abs(diff)
    smooth = tf.where(abs_d < 1.0, 0.5 * diff * diff, abs_d - 0.5)
    uv_loss = tf.reduce_mean(tf.boolean_mask(tf.reduce_sum(smooth, axis=-1),
                                             pos))
    return cls_loss + lambda_uv * uv_loss
```

### 4.3 Cabeza DensePose (JAX / Flax)

```python
import jax
import jax.numpy as jnp
import flax.linen as nn


class DensePoseHead(nn.Module):
    hidden: int = 512
    num_parts: int = 24

    @nn.compact
    def __call__(self, x):
        for _ in range(8):
            x = nn.Conv(self.hidden, (3, 3), padding="SAME")(x)
            x = nn.relu(x)
        part_logits = nn.Conv(self.num_parts + 1, (1, 1))(x)
        uv = nn.Conv(self.num_parts * 2, (1, 1))(x)
        b, h, w, _ = uv.shape
        uv = uv.reshape(b, h, w, self.num_parts, 2)
        return part_logits, uv


def smooth_l1(diff):
    abs_d = jnp.abs(diff)
    return jnp.where(abs_d < 1.0, 0.5 * diff * diff, abs_d - 0.5)


def densepose_loss_jax(part_logits, uv_pred, part_gt, uv_gt, fg_mask,
                       lambda_uv: float = 1.0):
    valid = (part_gt >= 0) & fg_mask
    one_hot = jax.nn.one_hot(jnp.maximum(part_gt, 0), part_logits.shape[-1])
    logp = jax.nn.log_softmax(part_logits, axis=-1)
    pix_ce = -jnp.sum(one_hot * logp, axis=-1)
    cls_loss = jnp.sum(pix_ce * valid) / jnp.maximum(valid.sum(), 1)

    pos = valid & (part_gt > 0)
    part_idx = jnp.maximum(part_gt - 1, 0)        # (B, H, W)
    uv_sel = jnp.take_along_axis(
        uv_pred, part_idx[..., None, None], axis=-2)[..., 0, :]  # (B, H, W, 2)
    uv_loss = jnp.sum(smooth_l1(uv_sel - uv_gt) * pos[..., None]) / \
              jnp.maximum(pos.sum(), 1)
    return cls_loss + lambda_uv * uv_loss
```

### 4.4 Cómputo de GPS (compartido — NumPy)

```python
import numpy as np


def geodesic_point_similarity(pred_vertex_ids, gt_vertex_ids,
                              geodesic_table, kappa: float = 0.255):
    """
    pred_vertex_ids  : (P,) vertices SMPL predichos para cada punto
    gt_vertex_ids    : (P,)
    geodesic_table   : (N_verts, N_verts) tabla precomputada de distancias
                       geodesicas sobre el mesh SMPL.
    Retorna escalar GPS.
    """
    dists = geodesic_table[pred_vertex_ids, gt_vertex_ids]
    return float(np.exp(-(dists ** 2) / (2 * kappa ** 2)).mean())
```

## 5. Experimentos clave (Sección 4 del paper)

### 5.1 Single-person dense pose

Comparación contra **SMPLify** (Bogo et al., 2016) y **UP-SMPLify-91**. Métricas en imágenes full-body sin oclusión vs. todas las imágenes:

| Método | AUC₁₀ (full) | AUC₃₀ (full) | AUC₁₀ (all) | AUC₃₀ (all) |
|---|---|---|---|---|
| UP-SMPLify-91 | 0.155 | 0.306 | — | — |
| SMPLify-14 | 0.226 | 0.416 | 0.099 | 0.19 |
| **DensePose (FCN)** | **0.429** | **0.630** | **0.378** | **0.614** |
| Human (upper bound) | 0.563 | 0.835 | 0.563 | 0.835 |

DensePose entrega aproximadamente **2× la precisión** de SMPLify-14, y es 100-1000× más rápido (0.04s vs 60-200s por imagen).

### 5.2 Multi-person dense pose (Tabla 1)

Sobre `COCO minival`, con ResNet-50-FPN:

| Método | AP | AP₅₀ | AP₇₅ | AR |
|---|---|---|---|---|
| DensePose (ResNet-50) | 51.0 | 83.5 | 54.2 | 60.1 |
| DensePose (ResNet-101) | 51.8 | 83.7 | 56.3 | 61.1 |
| + masks (multi-task) | 51.9 | 85.5 | 54.7 | 61.1 |
| + keypoints (multi-task) | 52.8 | 85.6 | 56.2 | 62.6 |
| + cascading + keypoints | **55.8** | 87.5 | 61.2 | **63.9** |

Conclusión: **multi-task learning con keypoints** y **cascading** aportan +5 AP combinados.

### 5.3 Inpainting / distillation

Sobre la curva multi-person:

| Sistema | AUC₁₀ | AUC₃₀ | IoU mask |
|---|---|---|---|
| DP-FCN (raw) | 0.253 | 0.418 | 0.66 |
| DP-RCNN (points only) | 0.315 | 0.567 | 0.75 |
| **DP-RCNN (distillation)** | 0.381 | 0.645 | 0.79 |
| DP-RCNN (cascade) | 0.390 | 0.664 | 0.81 |
| DP* (privileged, upper bound) | 0.417 | 0.683 | — |
| Human | 0.563 | 0.835 | — |

Distillation entrega +5-7 puntos AUC consistentes.

## 6. Limitaciones reconocidas y sutilezas

1. **Brecha con humano**: aun el mejor sistema (cascading) llega a AUC₁₀ = 0.390 vs. 0.563 humano. La superficie del cuerpo es regresable, no perfecta.
2. **Anotación detrás de la ropa**: el ground-truth asume el cuerpo desnudo subyacente. Funciona razonable con faldas, vestidos y abrigos (las figuras 13 lo muestran), pero formalmente es una *idealización*.
3. **Categorías de cuerpo**: el modelo SMPL base usado tiene una sola topología — no hay variantes para personas con discapacidad, amputaciones u otras particularidades anatómicas.
4. **Sólo un cuerpo a la vez por bbox**: heredan las limitaciones top-down de Mask R-CNN. Cuando dos cuerpos están muy juntos y se intersectan las bboxes propuestas, la rama DensePose puede confundirse — esto es exactamente el punto que critica PifPaf.
5. **Costo de anotación**: 5M correspondencias manuales. No escalable a otras categorías sin una pipeline similar.
6. **Inferencia para "in-the-wild"** sigue dependiendo de un person detector — herencia top-down.

## 7. Impacto y legado

DensePose definió un estándar de "qué significa estimar pose humana densamente" y abrió múltiples ramas:

- **Try-on virtual**: la representación UV permite mapear texturas de ropa al cuerpo (la primera demo del paper).
- **DensePose-Track** (Neverova et al., 2019): extensión temporal para vídeo, con tracking de correspondencias.
- **Continuous Surface Embeddings** (Neverova et al., 2020): generaliza DensePose a animales (Density-CSE).
- **HMR** (Kanazawa et al., 2018) y **VIBE** (Kocabas et al., 2020): pose 3D end-to-end fitting SMPL, usan supervisión DensePose como señal auxiliar.
- **Adam / Frankenstein model** (Joo et al., 2018) — Total Capture: extiende el espíritu de DensePose a cara + manos + cuerpo.
- **Detectron2 DensePose**: el código oficial mantenido por FAIR (https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) continúa siendo el baseline citado.

En 2025 el legado más visible es que la representación $(c, U, V)$ + **SMPL** es la *lingua franca* de cuerpo humano 3D en visión por computador, mientras los métodos modernos (ViTPose, ARTrack, 4DHumans) la integran como cabezal opcional o como supervisión auxiliar.

## 8. Conexión con la clase 17

DensePose es el ejemplo **canónico de aproximación top-down** y de **dense correspondence** en pose recognition (slides 26-36 del PDF de Tomás Vergara). Su importancia para el curso radica en:

- Muestra que **podemos ir más allá de los 17 keypoints** — el cuerpo es una superficie, no un grafo de puntos.
- Es el primer ejemplo donde el modelo **SMPL** se usa como ground-truth en imágenes reales — establece la conexión "imagen real → modelo paramétrico 3D".
- La crítica del profesor a la elección arbitraria de 17 keypoints se resuelve aquí: la parametrización (c, U, V) es **completa** sobre la superficie del cuerpo.
- Su debilidad bajo oclusión (bboxes intersectados) motiva la siguiente sección de la clase: **PifPaf** y los métodos bottom-up.

Cross-links:
- [[fundamentos/pose-estimation.md]] — keypoints vs. dense correspondence.
- [[fundamentos/dense-correspondence.md]] — UV mapping, MDS, parametrización de superficies.
- [[papers/SMPL-Loper-2015.md]] — modelo paramétrico subyacente.
- [[papers/PifPaf-Kreiss-2019.md]] — la alternativa bottom-up.
- [[clases/clase-17/teoria.md#densepose]] — sección de la clase.

## 9. Enlaces

- Paper: https://arxiv.org/abs/1802.00434
- Project page: http://densepose.org
- Código oficial (Detectron2): https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose
- Dataset: COCO-DensePose, descargable desde el project page.
