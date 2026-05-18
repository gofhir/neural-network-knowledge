# ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation

**Autores:** Yufei Xu, Jing Zhang, Qiming Zhang, Dacheng Tao (University of Sydney, JD Explore Academy)
**Año:** 2022 (NeurIPS)
**arXiv:** 2204.12484
**Código:** https://github.com/ViTAE-Transformer/ViTPose

---

## 1. Contexto histórico

En 2022, después de ViT (Dosovitskiy et al., ICLR 2021) y MAE (He et al., CVPR 2022), los **Vision Transformers** habían demostrado SOTA en clasificación (ImageNet), detección (DETR, ViTDet) y segmentación. Pero en **pose estimation** la situación era distinta:

- **HRNet** (Sun et al., CVPR 2019) y su familia de redes multi-resolución dominaban con AP ~76-77 en COCO Keypoint.
- **HRFormer** (Yuan et al., NeurIPS 2021) — primera intrusión transformer importante, pero mantiene la idea multi-resolución de HRNet y agrega complejidad: ramas paralelas a distintas resoluciones, fusiones cuidadosas.
- **TokenPose** (Li et al., ICCV 2021), **TransPose** (Yang et al., ICCV 2021), **PRTR** (Li et al., CVPR 2021) — todos **usan CNN como backbone + transformer como decoder o refiner**. Ninguno explora "ViT puro como backbone".

**La pregunta que Xu et al. plantean**: ¿qué tan bien funciona un ViT *plain* (sin trucos arquitecturales) como backbone para pose estimation, con un decoder simple? La respuesta sorprendente del paper: **80.9 AP en COCO test-dev**, nuevo SOTA, sin elaborar nada.

Este resultado es interesante porque **rompe la hipótesis** de que pose estimation necesita feature maps multi-resolución (la justificación arquitectural de HRNet). Demuestra que con suficiente capacidad y pre-training adecuado (MAE), un transformer plano + decoder de 2 deconvoluciones es suficiente.

## 2. Contribución central

Cuatro propiedades enunciadas como objetivos del paper:

1. **Simplicidad estructural**: ViT plain (no jerárquico) como encoder + decoder lightweight (2 deconvs o incluso 1 bilinear + 1 conv).
2. **Escalabilidad**: modelos de 86M (ViT-B) hasta 1B parámetros (ViTAE-G) cubriendo todo el frente de Pareto throughput-vs-AP.
3. **Flexibilidad de entrenamiento**: tolerante a distintos pretraining datasets (ImageNet, COCO, AI Challenger), resoluciones de entrada, tipos de atención (full, window, shifted-window, pooling), finetuning parcial (FFN-only o MHSA-only) y multi-dataset.
4. **Transferibilidad**: knowledge distillation eficiente vía **token de conocimiento** aprendible (no solo via MSE de outputs).

Resultados clave:

| Modelo | Backbone | Params | AP val | AP test-dev |
|---|---|---|---|---|
| SimpleBaseline | ResNet-152 | 60M | 73.5 | — |
| HRNet-W48 | HRNet | 64M | 75.1 | 76.3 |
| TokenPose-L/D24 | HRNet-W48 | 28M | 75.8 | — |
| HRFormer-B | HRFormer | 43M | 75.6 | 77.2 |
| **ViTPose-B** | ViT-B | 86M | 75.8 | — |
| **ViTPose-L** | ViT-L | 307M | 78.3 | — |
| **ViTPose-H** | ViT-H | 632M | 79.1 | 79.5 |
| **ViTPose-G** | ViTAE-G | 1024M | **80.9** | **80.9** |

ViTPose-G a 1B parámetros entrega +4 AP sobre HRNet, con throughput competitivo gracias a paralelismo nativo de transformers.

## 3. Arquitectura

### 3.1 ViTPose framework (Figura 2 del paper)

```
[Input image]
      |
  Patch Embedding (16x16, stride 16)
      |
  [Tokens: H/16 x W/16 x C]
      |
  L x Transformer Block (MHSA + FFN, LayerNorm pre)
      |
  [F_out: H/16 x W/16 x C]
      |
  Lightweight Decoder
      |
  [Heatmaps: H/4 x W/4 x N_keypoints]
```

**Entrada**: instancia de persona crop a 256×192 (siguiendo SimpleBaseline). Esto significa que ViTPose es **top-down** (un detector externo provee bboxes).

### 3.2 Encoder ViT (Ecuación 1)

Estándar — bloques iguales a ViT:

$$
F'_{i+1} = F_i + \text{MHSA}(\text{LN}(F_i)) \\
F_{i+1} = F'_{i+1} + \text{FFN}(\text{LN}(F'_{i+1}))
$$

Variantes (Tabla 1 del paper):

| Modelo | Backbone | Patch | Layers | Dim | Heads | Params |
|---|---|---|---|---|---|---|
| ViTPose-B | ViT-B | 16 | 12 | 768 | 12 | 86M |
| ViTPose-L | ViT-L | 16 | 24 | 1024 | 16 | 307M |
| ViTPose-H | ViT-H | 14→16 | 32 | 1280 | 16 | 632M |
| ViTPose-G | ViTAE-G | 14→16 | 40 | 1408 | 16 | 1024M |

Para ViT-H y ViTAE-G, que pre-entrenan a patch 14×14, hacen **zero-padding del patch embedding** para usar patch 16×16 al finetunearlos en pose, así matchean ViT-B/L.

### 3.3 Decoder — dos variantes (Sección 3.1)

**Classic decoder** (Ecuación 2): igual a SimpleBaseline.

$$
K = \text{Conv}_{1 \times 1}(\text{Deconv}(\text{Deconv}(F_\text{out})))
$$

- 2 capas deconv (stride 2 cada una) — upsamplea $H/16 \to H/4$.
- BatchNorm + ReLU entre ellas.
- Conv 1×1 final produce $N_k$ heatmaps (17 para COCO).

**Simple decoder** (Ecuación 3):

$$
K = \text{Conv}_{3 \times 3}(\text{Bilinear}_{4 \times}(\text{ReLU}(F_\text{out})))
$$

- Bilinear upsample 4×.
- ReLU.
- Conv 3×3 final.

**Resultado sorpresa** (Tabla 2): con ViT-B/L/H, los decoders simple y classic dan AP casi idéntico (75.8 vs 75.5 para ViTPose-B). Pero con ResNet-50 el simple decoder pierde 18 AP. La interpretación: **el ViT pretrained ya hace todo el trabajo representacional**; el decoder es un detalle.

### 3.4 Pretraining flexibility (Sección 3.3)

ViTPose usa **MAE (Masked Autoencoder)** para pre-entrenar el backbone:

1. Tomar imágenes.
2. Maskear el 75% de los patches.
3. Entrenar el encoder + decoder a reconstruir los patches enmascarados (reconstruction loss MSE).
4. Tirar el decoder MAE, usar el encoder como inicialización para ViTPose.

**Descubrimiento clave**: pre-entrenar MAE **sobre las propias imágenes de pose** (COCO + AI Challenger, 500K imágenes) entrega 75.8 AP — *idéntico* a pre-entrenar sobre ImageNet-1K (1M imágenes). Es decir, **la cantidad de datos de pretraining no es el cuello de botella** — la auto-supervisión MAE basta sobre datos de dominio.

### 3.5 Resolución y atención (Sección 3.3)

- **Resolución mayor**: aumentar input de 256×192 a 384×288 da +2 AP. ViTPose escala bien a inputs grandes simplemente alargando la secuencia de tokens.
- **Atención eficiente**: para inputs grandes, full-attention sufre por la cuadrática en tokens. Probaron:
  - **Window attention** (atención local): -10 AP — demasiado restrictiva.
  - **Shifted window** (Swin-style): +10 AP recuperado.
  - **Pooling window**: pool por ventana → atención cruzada. Complementaria al shift.
  - **Shift + pool**: 22.9G memoria vs 28.6G de full attention con mismo AP (76.8 vs 76.9).

### 3.6 Multi-task / multi-dataset (Sección 3.3)

ViTPose comparte el backbone entre múltiples datasets (COCO, AI Challenger, MPII) y usa **decoders independientes por dataset**. Cada iteración samplea un dataset, pasa el batch por el backbone compartido y por el decoder del dataset elegido. Esto cuesta ~0 extra parameters (decoders son lightweight) y entrega +1.3 AP en COCO val.

### 3.7 Token-based knowledge distillation (Sección 3.4)

Innovación: en vez de solo distillation de outputs:

$$
L^{od}_{t \to s} = \text{MSE}(K_s, K_t)
$$

Proponen **knowledge token** — un token aprendible $t$ que se concatena con los visual tokens del **teacher** durante una fase de pre-distillation:

$$
t^* = \arg\min_t \text{MSE}(T(\{t; X\}), K_{gt})
$$

donde $T$ es el teacher (ViTPose-L congelado) y $X$ son los tokens de imagen. Una vez aprendido $t^*$, se concatena con los tokens del **student** durante finetuning para inyectar conocimiento estructural del teacher:

$$
L^{td}_{t \to s} = \text{MSE}(S(\{t^*; X\}), K_{gt})
$$

o combinado con output distillation:

$$
L^{tod}_{t \to s} = \text{MSE}(S(\{t^*; X\}), K_t) + \text{MSE}(S(\{t^*; X\}), K_{gt})
$$

Resultado: token + output distillation = ViTPose-B mejora de 75.8 → 76.6 AP, transferido desde ViTPose-L.

## 4. Ejemplos de código (PyTorch / TF / JAX)

### 4.1 ViTPose end-to-end (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                         # (B, D, H/p, W/p)
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)         # (B, N, D)
        return x, (h, w)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ViTPoseClassicDecoder(nn.Module):
    def __init__(self, dim=768, num_keypoints=17):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(dim, 256, kernel_size=4,
                                          stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256); self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4,
                                          stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256); self.relu2 = nn.ReLU(inplace=True)
        self.pred = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x_2d):
        x = self.relu1(self.bn1(self.deconv1(x_2d)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        return self.pred(x)


class ViTPoseSimpleDecoder(nn.Module):
    def __init__(self, dim=768, num_keypoints=17):
        super().__init__()
        self.pred = nn.Conv2d(dim, num_keypoints, kernel_size=3, padding=1)

    def forward(self, x_2d):
        x = F.relu(x_2d)
        x = F.interpolate(x, scale_factor=4, mode="bilinear",
                          align_corners=False)
        return self.pred(x)


class ViTPose(nn.Module):
    def __init__(self, depth=12, dim=768, num_heads=12,
                 patch_size=16, num_keypoints=17, simple=False):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, 3, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, dim))  # 16x12 tokens
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.decoder = (ViTPoseSimpleDecoder(dim, num_keypoints)
                        if simple else ViTPoseClassicDecoder(dim, num_keypoints))

    def forward(self, image):
        x, (h, w) = self.patch_embed(image)
        x = x + self.pos_embed[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        b, n, d = x.shape
        x = x.transpose(1, 2).reshape(b, d, h, w)
        return self.decoder(x)
```

### 4.2 Heatmap loss + decoding (PyTorch)

```python
def gaussian_target(joint_xy, hm_h, hm_w, sigma=2.0):
    """Genera heatmap Gaussiano para una articulacion."""
    x, y = joint_xy
    yy, xx = torch.meshgrid(torch.arange(hm_h), torch.arange(hm_w),
                             indexing="ij")
    return torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def heatmap_loss(pred, target, weight=None):
    """MSE pixel-wise, opcionalmente ponderada por keypoint."""
    if weight is None:
        return F.mse_loss(pred, target)
    diff = (pred - target) ** 2
    return (diff.mean(dim=(2, 3)) * weight).mean()


def decode_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """heatmaps: (B, K, H, W) -> (B, K, 2) coords (x, y) con sub-pixel."""
    b, k, h, w = heatmaps.shape
    flat = heatmaps.view(b, k, -1)
    idx = flat.argmax(dim=-1)
    x = (idx % w).float()
    y = (idx // w).float()
    # Refinamiento sub-pixel via gradiente (Sun et al., DARK, 2020):
    # se omite por brevedad. ViTPose usa UDP en sus tablas SOTA.
    return torch.stack([x, y], dim=-1)
```

### 4.3 ViTPose en TensorFlow 2

```python
import tensorflow as tf
from tensorflow.keras import layers, Model


def transformer_block(dim, num_heads):
    def block(x):
        h = layers.LayerNormalization()(x)
        a = layers.MultiHeadAttention(num_heads=num_heads,
                                      key_dim=dim // num_heads)(h, h)
        x = x + a
        h = layers.LayerNormalization()(x)
        m = layers.Dense(dim * 4, activation="gelu")(h)
        m = layers.Dense(dim)(m)
        return x + m
    return block


def build_vitpose(image_size=(256, 192), patch=16, dim=768, depth=12,
                  num_heads=12, num_keypoints=17, simple_decoder=False):
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = layers.Conv2D(dim, patch, strides=patch)(inputs)   # patch embed
    b = tf.shape(x)[0]
    h, w = image_size[0] // patch, image_size[1] // patch
    x = layers.Reshape((h * w, dim))(x)
    pos = tf.Variable(tf.zeros([1, h * w, dim]), trainable=True)
    x = x + pos
    for _ in range(depth):
        x = transformer_block(dim, num_heads)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Reshape((h, w, dim))(x)

    if simple_decoder:
        x = layers.ReLU()(x)
        x = layers.UpSampling2D(size=4, interpolation="bilinear")(x)
        out = layers.Conv2D(num_keypoints, 3, padding="same")(x)
    else:
        x = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        out = layers.Conv2D(num_keypoints, 1)(x)

    return Model(inputs, out, name="vitpose")
```

### 4.4 ViTPose en JAX / Flax

```python
import jax.numpy as jnp
import flax.linen as nn


class TransformerBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        a = nn.SelfAttention(num_heads=self.num_heads,
                              qkv_features=self.dim)(h)
        x = x + a
        h = nn.LayerNorm()(x)
        m = nn.Dense(int(self.dim * self.mlp_ratio))(h)
        m = nn.gelu(m)
        m = nn.Dense(self.dim)(m)
        return x + m


class ViTPose(nn.Module):
    patch: int = 16
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    num_keypoints: int = 17
    simple_decoder: bool = False

    @nn.compact
    def __call__(self, image):
        b, H, W, C = image.shape
        x = nn.Conv(self.dim, (self.patch, self.patch),
                    strides=(self.patch, self.patch))(image)
        h, w = H // self.patch, W // self.patch
        x = x.reshape(b, h * w, self.dim)
        pos = self.param("pos_embed", nn.initializers.zeros,
                          (1, h * w, self.dim))
        x = x + pos
        for _ in range(self.depth):
            x = TransformerBlock(self.dim, self.num_heads)(x)
        x = nn.LayerNorm()(x)
        x = x.reshape(b, h, w, self.dim)
        if self.simple_decoder:
            x = nn.relu(x)
            x = jax.image.resize(x, (b, h * 4, w * 4, self.dim),
                                  method="bilinear")
            x = nn.Conv(self.num_keypoints, (3, 3), padding="SAME")(x)
        else:
            x = nn.ConvTranspose(256, (4, 4), strides=(2, 2),
                                  padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=False)(x); x = nn.relu(x)
            x = nn.ConvTranspose(256, (4, 4), strides=(2, 2),
                                  padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=False)(x); x = nn.relu(x)
            x = nn.Conv(self.num_keypoints, (1, 1))(x)
        return x
```

### 4.5 Knowledge token distillation (PyTorch, conceptual)

```python
class ViTPoseWithKnowledgeToken(nn.Module):
    def __init__(self, vitpose: ViTPose, learn_token: bool = True):
        super().__init__()
        self.vp = vitpose
        self.knowledge_token = nn.Parameter(torch.zeros(1, 1, vitpose.dim))
        if not learn_token:
            self.knowledge_token.requires_grad_(False)

    def forward(self, image):
        x, (h, w) = self.vp.patch_embed(image)
        x = x + self.vp.pos_embed[:, :x.shape[1]]
        # concatena knowledge token al inicio de la secuencia
        kt = self.knowledge_token.expand(x.shape[0], -1, -1)
        x = torch.cat([kt, x], dim=1)
        for blk in self.vp.blocks:
            x = blk(x)
        x = self.vp.norm(x)
        # descarta el knowledge token antes del decoder
        x = x[:, 1:, :]
        b, n, d = x.shape
        x = x.transpose(1, 2).reshape(b, d, h, w)
        return self.vp.decoder(x)


# Fase 1: aprender token con teacher congelado
def learn_token(teacher_model, dataloader, num_epochs=10):
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    teacher_with_token = ViTPoseWithKnowledgeToken(teacher_model)
    # solo el token tiene grad
    for n, p in teacher_with_token.named_parameters():
        p.requires_grad_(n == "knowledge_token")
    opt = torch.optim.AdamW([teacher_with_token.knowledge_token], lr=1e-3)
    for _ in range(num_epochs):
        for img, hm_gt in dataloader:
            hm_pred = teacher_with_token(img)
            loss = F.mse_loss(hm_pred, hm_gt)
            opt.zero_grad(); loss.backward(); opt.step()
    return teacher_with_token.knowledge_token.detach()


# Fase 2: finetune student con token aprendido inyectado
def finetune_student(student_model, learned_token, dataloader,
                      teacher_model, num_epochs=210, lambda_kd=1.0):
    student_with_token = ViTPoseWithKnowledgeToken(student_model)
    student_with_token.knowledge_token.data = learned_token.clone()
    student_with_token.knowledge_token.requires_grad_(False)
    opt = torch.optim.AdamW(student_with_token.parameters(), lr=5e-4)
    teacher_with_token = ViTPoseWithKnowledgeToken(teacher_model)
    teacher_with_token.knowledge_token.data = learned_token.clone()
    teacher_with_token.eval()
    for _ in range(num_epochs):
        for img, hm_gt in dataloader:
            hm_s = student_with_token(img)
            with torch.no_grad():
                hm_t = teacher_with_token(img)
            loss = F.mse_loss(hm_s, hm_gt) + lambda_kd * F.mse_loss(hm_s, hm_t)
            opt.zero_grad(); loss.backward(); opt.step()
```

## 5. Experimentos clave

### 5.1 Estructura — simple vs classic decoder (Tabla 2)

| Backbone | Decoder | AP | AP₅₀ |
|---|---|---|---|
| ResNet-50 | Classic | 71.8 | 89.8 |
| ResNet-50 | Simple | 53.1 (-18.7) | 86.9 |
| ResNet-152 | Classic | 73.5 | 90.5 |
| ResNet-152 | Simple | 55.3 (-18.2) | 87.9 |
| ViTPose-B | Classic | 75.8 | 90.7 |
| **ViTPose-B** | **Simple** | **75.5** (-0.3) | 90.6 |
| ViTPose-L | Classic | 78.3 | 91.4 |
| **ViTPose-L** | **Simple** | **78.2** (-0.1) | 91.4 |

ResNet pierde 18 AP con decoder simple. **ViTPose pierde <0.3 AP** — el ViT *carga* todo el peso representacional.

### 5.2 Influencia del pretraining (Tabla 3)

| Pretrain | Volumen | AP val |
|---|---|---|
| ImageNet-1K | 1M | 75.8 |
| COCO (cropping) | 150K | 74.5 (-1.3) |
| **COCO + AIC** | 500K | **75.8** (=) |
| COCO + AIC (no crop) | 300K | 75.8 (=) |

Conclusión: **el dato de pretraining puede ser del propio dominio**, sin etiquetas, y con MAE basta.

### 5.3 Pareto throughput-AP (Tabla 9)

| Modelo | Backbone | Params | Speed (fps) | AP val |
|---|---|---|---|---|
| SimpleBaseline | ResNet-152 | 60M | 829 | 73.5 |
| HRNet-W48 | HRNet | 64M | 309 | 76.3 |
| HRFormer-B | HRFormer | 43M | 158 | 75.6 |
| **ViTPose-B** | ViT-B | 86M | **944** | **75.8** |
| ViTPose-L | ViT-L | 307M | 411 | 78.3 |
| ViTPose-H | ViT-H | 632M | 241 | 79.1 |

ViTPose-B es **3× más rápido** que HRNet-W48 con AP comparable. ViTPose-L supera a HRNet en AP con throughput similar.

### 5.4 Multi-dataset training (Tabla 7)

| Datasets | AP val |
|---|---|
| COCO | 75.8 |
| COCO + AIC | 77.0 (+1.2) |
| COCO + AIC + MPII | 77.1 (+1.3) |

Multi-dataset escala marginalmente bien — pose es generalizable cross-dataset cuando el backbone es lo suficientemente grande.

### 5.5 Finetuning parcial (Tabla 6)

| MHSA | FFN | AP val |
|---|---|---|
| ✓ | ✓ | 75.8 |
| frozen | ✓ | 75.1 (-0.7) |
| ✓ | frozen | 72.8 (-3.0) |

**Hallazgo**: el módulo **FFN es más específico de tarea** que MHSA. Frenarlo cuesta más AP. MHSA es más "task-agnostic" (modela relaciones token-token, transferible).

## 6. Limitaciones reconocidas

1. **Top-down only**: ViTPose requiere un person detector. No es bottom-up; hereda los problemas de top-down con bboxes intersectados.
2. **Sin decoders elaborados**: el paper deliberadamente no explora FPN, dilated attention, skip connections. Mejoras posibles aparecen como future work.
3. **Solo human pose (2D)**: no animal pose, no facial keypoints, no 3D — aunque la sección 5 menciona como future work la extensión.
4. **Costo de pre-entrenamiento**: MAE en COCO+AIC requiere ~500K imágenes × cientos de epochs. ViTPose-G es entrenado sobre cluster A100 ×8.
5. **No address explícito de occlusion**: el paper no estudia oclusiones específicamente — aunque el AP mejora versus baselines, no hay ablation dedicado.

## 7. Impacto y legado

- **SOTA reset** en COCO Keypoint test-dev — 80.9 AP destronó la era de las arquitecturas multi-resolution (HRNet/HRFormer).
- **Confirma "scale > arquitectura"**: la hipótesis de transformers (más datos + más params + auto-supervisión = SOTA) gana también en pose.
- **mmpose** (https://github.com/open-mmlab/mmpose) integró ViTPose como uno de sus baselines de referencia.
- Inspiró:
  - **ViTPose+** (2023) — versión multi-dataset/multi-task con pose 3D y animales.
  - **DWPose** (Yang et al., 2023) — distillation de ViTPose-X para entornos low-resource.
  - **EdgePose** y **MobileNetV3-Pose** — distillation para mobile via knowledge token (idea ampliada del paper).
- Concretiza la afirmación del profesor Vergara: *"el SOTA en object detection se hace con Vision Transformers, no CNNs. Pero los principios son los mismos."* — y ViTPose lo demuestra.

## 8. Conexión con la clase 17

ViTPose es el **"disclaimer" final** que el profesor introduce en los slides 51-52 del PDF de Clase 17. Su rol en el curso:

- Cierra el arco **CNN → Transformer** que también vimos en Clase 14 (Transformers) y en cualquier discusión de ViT.
- Demuestra que las técnicas vistas (top-down, keypoints, heatmaps) **siguen siendo conceptualmente válidas** — solo cambia el backbone.
- Justifica la importancia de **MAE y self-supervised pretraining**, que apareció en Clase 14 como parte del programa de Transformers.
- El paper conecta directamente con el "transfer learning" de la clase: el knowledge token + MAE pretraining son ejemplos de cómo escalar pose sin etiquetas adicionales.

Cross-links:
- [[fundamentos/pose-estimation.md]] — el lugar de ViT en el espectro de arquitecturas.
- [[fundamentos/vision-transformers.md]] — backbone subyacente (si existe).
- [[papers/DensePose-Guler-2018.md]] — la era CNN top-down anterior.
- [[clases/clase-14/...]] — la familia transformers en general.

## 9. Enlaces

- Paper: https://arxiv.org/abs/2204.12484
- Código: https://github.com/ViTAE-Transformer/ViTPose
- mmpose docs: https://mmpose.readthedocs.io/en/latest/model_zoo/topdown_heatmap.html#vitpose-coco
