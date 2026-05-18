# FaceNet: A Unified Embedding for Face Recognition and Clustering

**Autores:** Florian Schroff, Dmitry Kalenichenko, James Philbin (Google Inc.)
**Año:** 2015 (CVPR)
**arXiv:** 1503.03832

---

## 1. Contexto histórico

A inicios de 2015, el panorama de **face recognition** estaba dominado por arquitecturas profundas con cabezales de **clasificación softmax** sobre miles de identidades:

- **DeepFace** (Taigman et al., Facebook AI, 2014) — alineación 3D explícita + CNN entrenado como clasificación softmax sobre 4030 identidades, luego usa la penúltima capa como embedding. 97.35% en LFW.
- **DeepID / DeepID2 / DeepID2+** (Sun et al., 2014-2015) — ensembles de 25-50 CNNs entrenadas sobre patches faciales. Combinan classification + verification loss. Hasta 99.47% en LFW.
- **WebFace + JointBayesian** (Yi et al., 2014) — embeddings con PCA + Bayes para la similaridad.

**El problema con softmax + bottleneck**:

1. **Indirecto**: el modelo aprende a clasificar identidades vistas en entrenamiento; el embedding es un *subproducto*, no el objetivo.
2. **No generaliza bien a identidades nuevas** — la suposición es que el bottleneck "aprende rasgos genéricos".
3. **Embeddings de alta dimensión** (~1000-4000), caros para indexar a escala.
4. **Requiere alineamiento 2D/3D** preprocesado (DeepFace usa rotación + warp).

**Contribución radical de FaceNet**: entrenar **directamente la métrica de similaridad** en el espacio de embedding, vía **triplet loss**, produciendo un embedding compacto de **128-D** que vive en la esfera unitaria ($L_2$-normalizado), tal que:

$$
\|f(x^a_i) - f(x^p_i)\|_2^2 + \alpha < \|f(x^a_i) - f(x^n_i)\|_2^2
$$

para anchor $x^a$, positive $x^p$ (misma identidad) y negative $x^n$ (otra identidad). Resultado: **99.63% en LFW** — un **30% de reducción** del error sobre el SOTA anterior (DeepID2+, 99.47%).

## 2. Contribución central

Tres aportes clave:

1. **Triplet loss directa** para entrenamiento end-to-end del embedding, evitando el bottleneck softmax y permitiendo embeddings compactos (128-D, **96 bytes** si se cuantizan a uint8).

2. **Online semi-hard negative mining**: en lugar de samplear triplets aleatoriamente (mayoría triviales) o solo hard (causa colapso), seleccionan **semi-hard negatives** — negatives que están más lejos del anchor que el positive, pero dentro del margen:

   $$
   \|f(x^a_i) - f(x^p_i)\|_2^2 < \|f(x^a_i) - f(x^n_i)\|_2^2 < \|f(x^a_i) - f(x^p_i)\|_2^2 + \alpha
   $$

3. **Mini-batches grandes con sampling estructurado**: ~40 caras por identidad por mini-batch, con tamaño total ~1800 ejemplares. La elección de hard positives es **all anchor-positive pairs** dentro del batch (más estable que cherry-picking).

## 3. Arquitectura y método

### 3.1 Estructura general (Figura 2)

```
[Batch de imagenes]
         |
[Deep CNN backbone]  -> features
         |
[L2 normalization]   -> embedding en S^{127} subset R^{128}
         |
[Triplet Loss]
```

El backbone es una **CNN profunda**. El paper experimenta con dos:

- **NN1** — Zeiler & Fergus estilo, 22 capas, 140M params, 1.6B FLOPS. Tabla 1 del paper.
- **NN2** — Inception (GoogLeNet) estilo, 7.5M params, 1.6B FLOPS (Tabla 2). 20× menos parámetros que NN1.

La novedad arquitectural: agregan **$1 \times 1 \times d$ convolutions** entre las layers estándar (Network-in-Network style, Lin et al.) para reducir parámetros.

El paso final es **$L_2$ normalization** que pone $\|f(x)\|_2 = 1$ — todas las embeddings viven en la hiperesfera unitaria.

### 3.2 Triplet Loss (Ecuación 3)

$$
\mathcal{L} = \sum_{i=1}^N \left[ \|f(x^a_i) - f(x^p_i)\|_2^2 - \|f(x^a_i) - f(x^n_i)\|_2^2 + \alpha \right]_+
$$

donde $[\cdot]_+$ es el operador hinge ($\max(0, \cdot)$).

**Intuición**: el loss pena solo a los triplets que **violan el margen** $\alpha$. Si el negative ya está suficientemente lejos, el gradiente es cero — el modelo no malgasta capacidad refinando triplets fáciles.

Margen usado en el paper: $\alpha = 0.2$.

### 3.3 Triplet selection (Sección 3.2)

El número total de triplets es $\binom{N}{2} \times (M - N)$ donde $N$ es número de identidades y $M$ ejemplos por identidad — astronómico. Hay que samplear inteligentemente.

**Estrategia ideal**:

$$
x^p_i = \arg\max_{x^p} \|f(x^a_i) - f(x^p)\|_2^2 \quad \text{(hardest positive)}
$$

$$
x^n_i = \arg\min_{x^n} \|f(x^a_i) - f(x^n)\|_2^2 \quad \text{(hardest negative)}
$$

Pero esto es infactible global y, **incluso si fuera factible, lleva a colapso**: con noisy labels y outliers, hardest negatives dominan el entrenamiento.

**Solución de FaceNet — online semi-hard mining**:

1. Construir mini-batch con ~40 muestras por identidad, ~45 identidades = 1800 ejemplos.
2. Computar embeddings de todo el batch.
3. Para cada anchor $x^a$:
   - Usar **todos los positives** del batch como pareja (no solo el hardest — más estable).
   - Para el negative, samplear uno que sea **semi-hard**: $\|f(x^a) - f(x^n)\|^2 < \|f(x^a) - f(x^p)\|^2 + \alpha$ pero $\|f(x^a) - f(x^n)\|^2 > \|f(x^a) - f(x^p)\|^2$. Es decir: el negative que viola el margen pero no está más cerca del anchor que el positive.

Esto previene colapso y acelera convergencia.

### 3.4 Datos y entrenamiento

- **Dataset interno**: 100M-200M caras de 8M identidades. Resolución 96×96 a 224×224.
- **Tiempo de entrenamiento**: ~1000-2000 horas en cluster CPU (DistBelief).
- **Optimizer**: SGD con backprop estándar + AdaGrad. Learning rate inicial 0.05, decay manual.
- **Margen**: $\alpha = 0.2$ fijo.
- **Embedding dimensionality**: 128. Comparable a 64, 256, 512 — 128 fue el sweet spot.

### 3.5 Cuantización a 128 bytes

Aunque el embedding es float32 (4 × 128 = 512 bytes por cara), los autores **cuantizan a 128 bytes** (uint8 por componente) sin pérdida medible. Esto hace prácticas las bases de datos de caras a escala — 1B caras × 128 bytes = 128 GB, manejable.

### 3.6 Tareas downstream

Con el embedding entrenado, las aplicaciones se reducen a operaciones de álgebra lineal:

- **Verification** ("¿es la misma persona?"): threshold sobre $\|f(x_1) - f(x_2)\|_2^2$.
- **Recognition** ("¿quién es?"): k-NN sobre la base de embeddings conocidos.
- **Clustering** ("agrupa por persona"): k-means o agglomerative clustering sobre los embeddings.

No hace falta re-entrenar para nuevas identidades.

## 4. Ejemplos de código (PyTorch / TF / JAX)

### 4.1 Triplet loss + semi-hard mining (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance_squared(x: torch.Tensor) -> torch.Tensor:
    """Returns (N, N) matriz de distancias L2 al cuadrado."""
    dot = x @ x.t()
    sq = torch.diag(dot)
    d2 = sq.unsqueeze(0) + sq.unsqueeze(1) - 2 * dot
    return torch.clamp(d2, min=0.0)


def batch_semi_hard_triplet_loss(embeddings: torch.Tensor,
                                  labels: torch.Tensor,
                                  margin: float = 0.2,
                                  eps: float = 1e-12) -> torch.Tensor:
    """
    embeddings: (N, D) ya L2-normalizadas
    labels    : (N,) ids enteros
    """
    n = embeddings.shape[0]
    d2 = pairwise_distance_squared(embeddings)  # (N, N)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff = ~same
    diag_mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)

    # Para cada anchor, sumamos sobre TODOS los positives validos
    pos_mask = same & diag_mask
    losses = []
    for i in range(n):
        pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue
        for p in pos_idx.tolist():
            d_ap = d2[i, p]
            # negatives semi-hard:  d_ap < d_an < d_ap + margin
            valid_neg = diff[i] & (d2[i] > d_ap) & (d2[i] < d_ap + margin)
            if valid_neg.any():
                d_an = d2[i, valid_neg].min()
            else:
                # fallback: negative mas duro disponible
                d_an = d2[i, diff[i]].min() if diff[i].any() else d_ap.detach()
            losses.append(F.relu(d_ap - d_an + margin))
    if not losses:
        return embeddings.new_tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


class TripletNetwork(nn.Module):
    """Wrapper minimal que muestra el patron de embedding + L2 norm."""

    def __init__(self, backbone: nn.Module, embedding_dim: int = 128):
        super().__init__()
        self.backbone = backbone        # ej: resnet50 sin la capa final
        self.fc = nn.Linear(backbone.feature_dim, embedding_dim)

    def forward(self, x):
        h = self.backbone(x)
        h = self.fc(h)
        return F.normalize(h, p=2, dim=1)        # vive en S^{d-1}
```

### 4.2 Training loop (PyTorch)

```python
def train_facenet(model, dataloader, num_epochs=100, lr=0.05, margin=0.2):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # batch construido con sampler que garantiza P identidades x K muestras
            embeddings = model(images)
            loss = batch_semi_hard_triplet_loss(embeddings, labels,
                                                margin=margin)
            opt.zero_grad(); loss.backward(); opt.step()


class PKSampler:
    """Sampler que devuelve batches de P identidades x K imagenes."""
    def __init__(self, labels, P=45, K=40):
        self.P, self.K = P, K
        self.idx_by_label = {}
        for i, lbl in enumerate(labels):
            self.idx_by_label.setdefault(int(lbl), []).append(i)
        self.classes = list(self.idx_by_label.keys())

    def __iter__(self):
        while True:
            picked = random.sample(self.classes, self.P)
            batch = []
            for c in picked:
                pool = self.idx_by_label[c]
                batch.extend(random.choices(pool, k=self.K))
            yield batch
```

### 4.3 Triplet loss en TensorFlow 2

```python
import tensorflow as tf
import tensorflow_addons as tfa  # tiene una version oficial


def pairwise_distance_squared_tf(x):
    dot = tf.matmul(x, x, transpose_b=True)
    sq = tf.linalg.diag_part(dot)
    return tf.maximum(sq[None, :] + sq[:, None] - 2 * dot, 0.0)


def semi_hard_triplet_loss(labels, embeddings, margin=0.2):
    """Implementacion oficial similar a tfa.losses.TripletSemiHardLoss."""
    pdist = pairwise_distance_squared_tf(embeddings)
    same = tf.equal(labels[:, None], labels[None, :])
    diff = tf.logical_not(same)

    # Para cada anchor-positive (i, j), encuentra el mejor negative semi-hard
    # negative valido: d(a,n) > d(a,p) y d(a,n) - d(a,p) < margin
    n = tf.shape(labels)[0]
    diag_mask = tf.logical_not(tf.eye(n, dtype=tf.bool))
    pos_mask = same & diag_mask  # (N, N) parejas (a, p) validas

    # broadcast: d_ap [N, N] anchor-positive distance
    d_ap = pdist
    # for each (i, j) con pos_mask, compute candidate semi-hard negatives
    # truco: replicar a tres dimensiones
    d_an = pdist[:, None, :]                                 # (N, 1, N)
    d_ap_e = pdist[:, :, None]                                # (N, N, 1)
    semi_hard = diff[:, None, :] & (d_an > d_ap_e) & (d_an < d_ap_e + margin)
    d_an_masked = tf.where(semi_hard, d_an, tf.float32.max)
    best_d_an = tf.reduce_min(d_an_masked, axis=-1)           # (N, N)

    losses = tf.maximum(d_ap - best_d_an + margin, 0.0)
    losses = tf.boolean_mask(losses, pos_mask)
    return tf.reduce_mean(losses)
```

(En la práctica, TF/Keras ofrecen `tensorflow_addons.losses.TripletSemiHardLoss` que implementa esto correctamente y batched.)

### 4.4 Triplet loss en JAX

```python
import jax
import jax.numpy as jnp


def pairwise_distance_squared_jax(x):
    dot = x @ x.T
    sq = jnp.diag(dot)
    return jnp.maximum(sq[None, :] + sq[:, None] - 2 * dot, 0.0)


def semi_hard_triplet_loss_jax(embeddings, labels, margin=0.2):
    pdist = pairwise_distance_squared_jax(embeddings)
    n = labels.shape[0]
    same = labels[:, None] == labels[None, :]
    diff = ~same
    diag = ~jnp.eye(n, dtype=jnp.bool_)
    pos_mask = same & diag

    d_an = pdist[:, None, :]                # (N, 1, N) -> a fijado, n recorre
    d_ap_e = pdist[:, :, None]               # (N, N, 1) -> a, p fijados
    semi_hard = (diff[:, None, :]
                 & (d_an > d_ap_e)
                 & (d_an < d_ap_e + margin))
    very_large = jnp.finfo(jnp.float32).max
    d_an_masked = jnp.where(semi_hard, d_an, very_large)
    best_d_an = jnp.min(d_an_masked, axis=-1)            # (N, N)

    losses = jnp.maximum(pdist - best_d_an + margin, 0.0)
    losses = jnp.where(pos_mask, losses, 0.0)
    n_pos = jnp.sum(pos_mask)
    return jnp.sum(losses) / jnp.maximum(n_pos, 1)
```

### 4.5 Inferencia: verification + clustering

```python
def verify(model, img1: torch.Tensor, img2: torch.Tensor,
           threshold: float = 1.242) -> bool:
    """1.242 es el threshold reportado en LFW por Schroff et al."""
    with torch.no_grad():
        e1 = model(img1.unsqueeze(0))
        e2 = model(img2.unsqueeze(0))
        d2 = ((e1 - e2) ** 2).sum().item()
    return d2 < threshold


def build_index(model, dataset) -> tuple[torch.Tensor, list[int]]:
    """Devuelve (embeddings, labels) para hacer k-NN."""
    embs, labels = [], []
    with torch.no_grad():
        for img, lbl in dataset:
            embs.append(model(img.unsqueeze(0)).cpu())
            labels.append(lbl)
    return torch.cat(embs, dim=0), labels


def cluster_faces(embeddings: torch.Tensor, threshold: float = 0.6):
    """Agglomerative clustering vanilla con threshold de distancia."""
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(embeddings.numpy(), method="average", metric="euclidean")
    return fcluster(Z, t=threshold, criterion="distance")
```

## 5. Experimentos clave

### 5.1 LFW (Labeled Faces in the Wild)

Setup estándar *unrestricted, labeled outside data*: 6000 pares, 10-fold.

| Sistema | Acc % |
|---|---|
| Eigenfaces (PCA, baseline 2007) | ~60 |
| DeepFace (Taigman 2014) | 97.35 |
| DeepID2+ (Sun 2015) | 99.47 |
| **FaceNet (fixed crop)** | 98.87 ± 0.15 |
| **FaceNet + alignment** | **99.63 ± 0.09** |

Reducción de error de **30%** sobre DeepID2+.

### 5.2 YouTube Faces DB

Average pairwise similarity sobre 100 frames por video:

| Sistema | Acc % |
|---|---|
| DeepFace (Taigman) | 91.4 |
| DeepID2+ | 93.2 |
| **FaceNet** | **95.12 ± 0.39** |

### 5.3 Robustez a calidad de imagen (Tabla 4)

| JPEG quality | val rate |
|---|---|
| 10 | 67.3% |
| 20 | 81.4% |
| 30 | 83.9% |
| 50 | 85.5% |
| 90 | 86.5% |

| Pixels | val rate |
|---|---|
| 1,600 (40×40) | 37.8% |
| 6,400 (80×80) | 79.5% |
| 14,400 (120×120) | 84.5% |
| 65,536 (256×256) | 86.4% |

Conclusión: robusto a JPEG ≥ Q20 y a downscale hasta 80×80 px.

### 5.4 Dimensionalidad del embedding (Tabla 5)

| Dim | val rate |
|---|---|
| 64 | 86.8 ± 1.7 |
| **128** | **87.9 ± 1.9** |
| 256 | 87.7 ± 1.9 |
| 512 | 85.6 ± 2.0 |

128 es óptimo — 64 funciona casi igual, lo cual es enorme para deployment móvil.

### 5.5 Cantidad de datos (Tabla 6)

| Imágenes | val rate |
|---|---|
| 2.6M | 76.3% |
| 26M | 85.1% |
| 52M | 85.1% |
| 260M | 86.2% |

Beneficio marginal decreciente después de ~26M. **Diminishing returns** clásico.

## 6. Limitaciones reconocidas

1. **Datos privados**: el modelo se entrena en 200M caras de Google, no público. Replicar exactamente sus números requiere construir un dataset comparable (lo que hicieron OpenFace, FaceNet-PyTorch, ArcFace, etc.).
2. **Sesgo en el dataset**: el paper no analiza performance por demografía. Estudios posteriores (Buolamwini y Gebru, 2018 — *Gender Shades*) mostraron que estos sistemas tienen sesgos sistemáticos hacia caras de piel clara y masculinas.
3. **Triplet selection es costoso**: requiere construir batches estructurados (P identidades × K muestras). Diferente al training estándar.
4. **Sensible al margen**: $\alpha = 0.2$ no es siempre óptimo. Variaciones: **angular triplet loss** (SphereFace), **ArcFace** (additive angular margin), **CosFace** — usan logaritmos angulares en vez de Euclidean, dan mejores resultados.
5. **Solo cara**: el método es agnóstico a contenido — la cabeza-cuerpo entera o partes diferentes funcionarían igual. Pero el entrenamiento siempre fue cara-only.

## 7. Impacto y legado

FaceNet fue **el paper más citado en face recognition** del último decenio.

- **Triplet loss** se generalizó a:
  - **Person Re-ID** (Hermans et al., 2017 — *In Defense of the Triplet Loss for Person Re-Identification*).
  - **Image retrieval** (DeepRanking, FaceNet for SKUs en e-commerce).
  - **NLP**: Sentence-BERT (Reimers y Gurevych, 2019).
  - **Self-supervised contrastive learning** vía SimCLR, MoCo, BYOL — todos descendientes conceptuales.
- **Sucesores en face recognition**:
  - **CenterLoss** (Wen et al., 2016).
  - **SphereFace / A-Softmax** (Liu et al., 2017).
  - **ArcFace** (Deng et al., 2019) — actualmente SOTA en LFW (99.83%) y MegaFace (98%+).
  - **CosFace** (Wang et al., 2018).
- **Implementaciones públicas notables**:
  - **OpenFace** (Amos et al., 2016): replicación open-source de FaceNet, ampliamente usada.
  - **face_recognition** (de Geitgey, 2017): wrapper Pythonic sobre dlib + FaceNet.
  - **InsightFace** (https://github.com/deepinsight/insightface): incluye ArcFace, retinaface, deepfake detection.
- **Aplicaciones industriales**: Google Photos (cluster por persona), Facebook tagging, Apple Face ID (variante propia con anti-spoofing), MS Azure Face API, Amazon Rekognition.

## 8. Conexión con la clase 17

FaceNet aparece en los slides 55-56 del PDF de Clase 17 (sección "Facial recognition") como **el ejemplo canónico de triplet network**. Su rol pedagógico:

- Conecta pose recognition con **otra técnica de la vida real** (reconocimiento facial) — el profesor lo menciona como "common to mix pose recognition with other techniques" en slide 53.
- Introduce el **triplet ranking loss** $L(I_1, I_2, I_3) = \max\{0, m - |f(I_1) - f(I_3)| + |f(I_1) - f(I_2)|\}$ que aparece literalmente en slide 55.
- Es el ejemplo más visual de *metric learning* — útil para que los estudiantes vean que pose y embeddings son técnicas compatibles que comparten infraestructura (CNNs, transformers, loss landscapes).
- Conexión ética implícita: igual que pose recognition, face recognition tiene implicaciones serias en vigilancia, militar y sesgos — el profesor enfatiza esto en slides 57-58.

Cross-links:
- [[fundamentos/triplet-loss.md]] — la familia completa de losses contrastivas.
- [[fundamentos/metric-learning.md]] — el campo conceptual.
- [[clases/clase-17/teoria.md#facial-recognition]] — sección de la clase.
- Conexión con clases tempranas que cubrieron contrastive learning, embeddings y CNNs.

## 9. Enlaces

- Paper: https://arxiv.org/abs/1503.03832
- OpenFace (Carnegie Mellon): https://cmusatyalab.github.io/openface/
- face_recognition (Geitgey, dlib): https://github.com/ageitgey/face_recognition
- InsightFace: https://github.com/deepinsight/insightface
