# SMPL: A Skinned Multi-Person Linear Model

**Autores:** Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, Michael J. Black (Max Planck Institute for Intelligent Systems, Tübingen)
**Año:** 2015 (SIGGRAPH Asia)
**Project page:** https://smpl.is.tue.mpg.de/

---

## 1. Contexto histórico

A inicios de 2015, modelar el cuerpo humano en gráficos por computador y visión era un terreno fragmentado:

- **Industria de la animación** usaba *Linear Blend Skinning (LBS)* — rigging manual con corrección artesanal por artistas para evitar los artefactos "taffy" (estiramiento) y "candy-wrapper" (rotación). Cada personaje requería días-semanas de trabajo de un rigger experto.
- **Comunidad académica** había desarrollado modelos estadísticos del cuerpo:
  - **SCAPE** (Anguelov et al., SIGGRAPH 2005) — basado en deformaciones de triángulos, realista pero **incompatible con engines de animación estándar** (Maya, Blender, Unity, Unreal) que esperan LBS vertex-based.
  - **BlendSCAPE** (Hirshberg et al., 2012) — variante de SCAPE.
  - **Hasler et al.** (2010) — abstracción "bones" para controlar shape, sin blend shapes aprendidas.

**El gap que SMPL llena**: un modelo paramétrico que combina:
1. **Realismo**: aprendido de ~4000 scans 3D de humanos.
2. **Compatibilidad**: usa LBS estándar — funciona out-of-the-box en Maya, Blender, Unreal, Unity.
3. **Velocidad**: render en tiempo real (>50fps).
4. **Vertex-based**: cada cuerpo es una malla de **6890 vértices** con la misma topología (no triangle-based como SCAPE).

Es el equivalente, en el dominio del cuerpo humano, de lo que **GLoVe/word2vec** fueron para palabras: una *representación distribuida* del espacio de cuerpos humanos.

## 2. Contribución central

SMPL define una **función paramétrica** $M(\vec\beta, \vec\theta; \Phi)$ que mapea:

- $\vec\beta \in \mathbb{R}^{|\beta|}$ — **shape parameters** (típicamente 10 dimensiones, las primeras componentes principales de variación de forma del cuerpo).
- $\vec\theta \in \mathbb{R}^{72}$ — **pose parameters** (3 ángulos axis-angle por cada uno de los 23 joints + 3 para la orientación global = 72).

a una malla de $6890 \times 3 = 20670$ coordenadas:

$$
M : \mathbb{R}^{|\beta| \times |\theta|} \mapsto \mathbb{R}^{3N}, \quad N = 6890
$$

Las contribuciones técnicas concretas:

1. **Shape blend shapes** $B_S(\vec\beta)$: deformación lineal de la malla rest-template como combinación lineal de las componentes principales (PCA) de la base de datos de scans (CAESAR dataset, ~2000 scans/género).

2. **Pose-dependent blend shapes** $B_P(\vec\theta)$: corrección lineal de la malla que **depende de los elementos de las matrices de rotación de los joints** — no de los ángulos directos. Esto es clave: ya que los elementos de $R(\vec\theta)$ están acotados en $[-1, 1]$, las deformaciones generalizan mejor que si se condicionaran en ángulos directos.

3. **Joint regressor** $\mathcal{J}$: matriz aprendida que predice las posiciones 3D de los 23 joints **como combinación lineal esparsa de los vértices superficiales rest-pose**. Esto es genial porque adaptarlo a un nuevo cuerpo solo requiere computar $\mathcal{J}(\vec\beta)$ una vez.

4. **Blend weights** $\mathcal{W}$: matriz $N \times K = 6890 \times 24$ que pondera cuánto cada bone afecta cada vértice durante el LBS.

5. **Training data y procedimiento**: entrenan desde 4000 scans alineados a un mesh template común — multi-pose (1786 registrations, 40 sujetos, ~50 poses cada uno) y multi-shape (CAESAR, ~3700 sujetos en T-pose).

## 3. Modelo matemático

### 3.1 Ecuaciones principales

**Joint locations** dependen solo de la forma:

$$
J(\vec\beta; \mathcal{J}, \bar{T}, \mathcal{S}) = \mathcal{J}(\bar{T} + B_S(\vec\beta; \mathcal{S}))
$$

donde $\bar{T} \in \mathbb{R}^{3N}$ es la malla template promedio.

**Shape blend shapes**:

$$
B_S(\vec\beta; \mathcal{S}) = \sum_{n=1}^{|\beta|} \beta_n S_n
$$

con $S_n \in \mathbb{R}^{3N}$ las componentes principales de variación de shape (aprendidas vía PCA en el dataset multi-shape).

**Pose blend shapes** (Ecuación 9 del paper):

$$
B_P(\vec\theta; \mathcal{P}) = \sum_{n=1}^{9K}\bigl(R_n(\vec\theta) - R_n(\vec\theta^*)\bigr) P_n
$$

donde:
- $R_n(\vec\theta)$ es el $n$-ésimo elemento del **stack de matrices de rotación** $R(\vec\theta) \in \mathbb{R}^{9K}$ ($K = 23$ joints × 9 elementos por matriz $3 \times 3$).
- $\vec\theta^*$ es la pose rest (todos los ángulos a 0).
- $P_n \in \mathbb{R}^{3N}$ son las pose blend shapes aprendidas.
- La sustracción $R_n(\vec\theta^*)$ garantiza que en pose rest $B_P = 0$.

**Función final** (Ecuación 11):

$$
M(\vec\beta, \vec\theta; \Phi) = W\bigl(T_P(\vec\beta, \vec\theta; \bar{T}, \mathcal{S}, \mathcal{P}),\ J(\vec\beta; \mathcal{J}, \bar{T}, \mathcal{S}),\ \vec\theta,\ \mathcal{W}\bigr)
$$

donde:

$$
T_P(\vec\beta, \vec\theta; \cdot) = \bar{T} + B_S(\vec\beta) + B_P(\vec\theta)
$$

es la malla rest deformada por shape y pose blend shapes, **antes** del skinning. Luego $W(\cdot)$ aplica linear blend skinning estándar (o dual-quaternion en la variante SMPL-DQBS).

### 3.2 Linear Blend Skinning estándar

$$
\bar t'_i = \sum_{k=1}^K w_{k,i}\, G'_k(\vec\theta, J)\, \bar t_i
$$

donde:
- $\bar t_i \in \mathbb{R}^3$ es el vértice rest.
- $w_{k,i}$ es el blend weight del bone $k$ sobre el vértice $i$.
- $G'_k$ es la transformación rígida acumulada del bone $k$ en el kinematic tree.

La transformación acumulada:

$$
G'_k(\vec\theta, J) = G_k(\vec\theta, J)\, G_k(\vec\theta^*, J)^{-1}
$$

$$
G_k(\vec\theta, J) = \prod_{j \in A(k)} \begin{bmatrix} \exp(\vec\omega_j) & j_j \\ 0 & 1 \end{bmatrix}
$$

donde $A(k)$ son los ancestros de $k$ en el árbol cinemático, $\exp(\vec\omega_j)$ es la matriz de rotación del joint $j$ vía la fórmula de **Rodrigues**:

$$
\exp(\vec\omega) = I + \hat\omega \sin(\|\vec\omega\|) + \hat\omega^2 \cos(\|\vec\omega\|)
$$

con $\hat\omega$ el operador skew-symmetric.

### 3.3 Entrenamiento — energía de pose (Sección 4.1)

Optimizan los parámetros del modelo $\{\mathcal{J}, \mathcal{W}, \mathcal{P}\}$ minimizando:

$$
E_*(\hat T^P, \hat J^P, \Theta, \mathcal{W}, \mathcal{P}) = E_D + \lambda_Y E_Y + \lambda_J E_J + \lambda_P E_P + E_W
$$

con:

- $E_D$ — data term: distancia euclidiana al cuadrado entre vértices registrados y vértices SMPL.
- $E_Y$ — simetría (peso 100): penaliza asimetría izquierda-derecha en joints y templates.
- $E_J$ — joint anchor: penaliza desviación de joint regressor de la inicialización segmentada manualmente.
- $E_P$ — Frobenius regularización de $\mathcal{P}$.
- $E_W$ — Frobenius regularización de blend weights respecto de los iniciales.

### 3.4 Entrenamiento — shape (Sección 4.2)

1. Pose-normalizar todos los scans CAESAR a la pose rest usando el modelo learned-so-far.
2. Hacer PCA sobre las mallas rest pose-normalizadas → obtener $\bar T$ (mean shape) y $\mathcal{S}$ (PCA basis).
3. Re-entrenar el joint regressor con los nuevos shapes.

Iteran multi-pose y multi-shape hasta convergencia.

## 4. Ejemplos de código (PyTorch / TF / JAX)

### 4.1 SMPL en PyTorch (forward pass)

```python
import torch
import torch.nn as nn


class SMPL(nn.Module):
    """
    Implementacion minima del forward de SMPL.

    Parametros aprendidos cargados desde el .pkl/.npz oficial:
    - template     : (N, 3)        malla rest = T_bar
    - shapedirs    : (N, 3, B)     componentes PCA de shape (B=10)
    - posedirs     : (N, 3, 9K)    pose blend shapes
    - J_regressor  : (K+1, N)      regressor de joints (K+1=24)
    - weights      : (N, K+1)      blend weights LBS
    - kintree      : (K+1,)        padre de cada joint en el arbol
    """

    def __init__(self, smpl_dict: dict):
        super().__init__()
        for k in ("template", "shapedirs", "posedirs", "J_regressor",
                  "weights"):
            self.register_buffer(k, torch.tensor(smpl_dict[k]).float())
        self.register_buffer("kintree", torch.tensor(smpl_dict["kintree"]).long())
        self.K = self.weights.shape[1] - 1   # 23 joints + 1 root = 24

    @staticmethod
    def rodrigues(axis_angle: torch.Tensor) -> torch.Tensor:
        """(..., 3) -> (..., 3, 3)"""
        theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
        k = axis_angle / theta
        kx, ky, kz = k.unbind(-1)
        zero = torch.zeros_like(kx)
        K = torch.stack([torch.stack([zero, -kz, ky], -1),
                          torch.stack([kz, zero, -kx], -1),
                          torch.stack([-ky, kx, zero], -1)], -2)
        cos = torch.cos(theta).unsqueeze(-1)
        sin = torch.sin(theta).unsqueeze(-1)
        I = torch.eye(3, device=axis_angle.device).expand_as(K)
        return I + sin * K + (1 - cos) * (K @ K)

    def forward(self, betas: torch.Tensor, pose: torch.Tensor):
        """
        betas: (B, |beta|)
        pose : (B, (K+1)*3)  axis-angle por cada joint
        """
        b = betas.shape[0]
        # 1. Shape blend shapes
        v_shaped = self.template + torch.einsum(
            "vmb,bb->vbm", self.shapedirs, betas.t()).permute(2, 0, 1)  # (B, N, 3)
        # joints rest
        J_rest = torch.einsum("kn,bnm->bkm", self.J_regressor, v_shaped)  # (B, K+1, 3)

        # 2. Rotation matrices via Rodrigues
        pose = pose.view(b, self.K + 1, 3)
        R = self.rodrigues(pose)                  # (B, K+1, 3, 3)

        # 3. Pose blend shapes: usa (R - I) flattened
        I3 = torch.eye(3, device=pose.device)
        pose_feat = (R[:, 1:] - I3).reshape(b, -1)  # (B, K*9) - excluye root
        v_posed = v_shaped + torch.einsum(
            "vmp,bp->bvm", self.posedirs, pose_feat)

        # 4. Acumulacion de transformaciones en el kinematic tree
        T = self._chain(R, J_rest)                 # (B, K+1, 4, 4)

        # 5. LBS
        W = self.weights                            # (N, K+1)
        T_n = torch.einsum("nk,bkij->bnij", W, T)   # (B, N, 4, 4)
        v_homog = torch.cat([v_posed,
                              torch.ones_like(v_posed[..., :1])], -1)
        v_world = torch.einsum("bnij,bnj->bni", T_n, v_homog)[..., :3]
        return v_world, J_rest

    def _chain(self, R: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Construye las transformaciones acumuladas G'_k."""
        b, K1 = R.shape[:2]
        rel_J = J.clone()
        rel_J[:, 1:] -= J[:, self.kintree[1:]]
        T_local = torch.cat([R, rel_J.unsqueeze(-1)], dim=-1)
        bot = torch.tensor([0, 0, 0, 1.0], device=R.device).expand(b, K1, 1, 4)
        T_local = torch.cat([T_local, bot], dim=-2)         # (B, K+1, 4, 4)
        T_global = [T_local[:, 0]]
        for k in range(1, K1):
            T_global.append(T_global[self.kintree[k]] @ T_local[:, k])
        T_global = torch.stack(T_global, dim=1)             # (B, K+1, 4, 4)
        # Restamos la pose rest para evitar offset doble
        rest_offset = torch.cat([
            torch.zeros(b, K1, 3, 3, device=R.device),
            J.unsqueeze(-1)], dim=-1)
        rest_offset = torch.cat([
            rest_offset,
            torch.zeros(b, K1, 1, 4, device=R.device)], dim=-2)
        return T_global - (T_global @ rest_offset)
```

(Versión completa y oficial: `smplx` package en https://github.com/vchoutas/smplx — soporta SMPL, SMPL-H, SMPL-X, MANO, FLAME.)

### 4.2 SMPL en TensorFlow 2

```python
import tensorflow as tf
import numpy as np


def rodrigues_tf(axis_angle):
    theta = tf.norm(axis_angle, axis=-1, keepdims=True)
    theta = tf.maximum(theta, 1e-8)
    k = axis_angle / theta
    kx, ky, kz = tf.unstack(k, axis=-1)
    zero = tf.zeros_like(kx)
    K = tf.stack([
        tf.stack([zero, -kz, ky], axis=-1),
        tf.stack([kz, zero, -kx], axis=-1),
        tf.stack([-ky, kx, zero], axis=-1),
    ], axis=-2)
    I = tf.eye(3, batch_shape=tf.shape(K)[:-2])
    cos = tf.expand_dims(tf.cos(theta), -1)
    sin = tf.expand_dims(tf.sin(theta), -1)
    return I + sin * K + (1.0 - cos) * tf.matmul(K, K)


class SMPL_TF(tf.keras.Model):
    def __init__(self, smpl_dict):
        super().__init__()
        self.template    = tf.constant(smpl_dict["template"], tf.float32)
        self.shapedirs   = tf.constant(smpl_dict["shapedirs"], tf.float32)
        self.posedirs    = tf.constant(smpl_dict["posedirs"], tf.float32)
        self.J_regressor = tf.constant(smpl_dict["J_regressor"], tf.float32)
        self.weights     = tf.constant(smpl_dict["weights"], tf.float32)
        self.kintree     = tf.constant(smpl_dict["kintree"], tf.int32)
        self.K = self.weights.shape[1] - 1

    def call(self, betas, pose):
        b = tf.shape(betas)[0]
        v_shaped = self.template + tf.einsum("vmb,nb->nvm",
                                              self.shapedirs, betas)
        J_rest = tf.einsum("kn,bnm->bkm", self.J_regressor, v_shaped)
        pose = tf.reshape(pose, (b, self.K + 1, 3))
        R = rodrigues_tf(pose)
        pose_feat = tf.reshape(R[:, 1:] - tf.eye(3),
                                (b, self.K * 9))
        v_posed = v_shaped + tf.einsum("vmp,bp->bvm",
                                        self.posedirs, pose_feat)
        # ... LBS chain similar al de PyTorch (omitido por brevedad)
        return v_posed
```

### 4.3 SMPL en JAX

```python
import jax
import jax.numpy as jnp


def rodrigues_jax(axis_angle):
    theta = jnp.linalg.norm(axis_angle, axis=-1, keepdims=True)
    theta = jnp.maximum(theta, 1e-8)
    k = axis_angle / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = jnp.zeros_like(kx)
    K = jnp.stack([
        jnp.stack([zero, -kz, ky], -1),
        jnp.stack([kz, zero, -kx], -1),
        jnp.stack([-ky, kx, zero], -1),
    ], -2)
    I = jnp.broadcast_to(jnp.eye(3), K.shape)
    cos = jnp.expand_dims(jnp.cos(theta), -1)
    sin = jnp.expand_dims(jnp.sin(theta), -1)
    return I + sin * K + (1 - cos) * (K @ K)


def smpl_forward(betas, pose, template, shapedirs, posedirs,
                  J_regressor, weights, kintree):
    b = betas.shape[0]
    v_shaped = template + jnp.einsum("vmb,nb->nvm", shapedirs, betas)
    J_rest = jnp.einsum("kn,bnm->bkm", J_regressor, v_shaped)
    pose = pose.reshape(b, -1, 3)
    R = rodrigues_jax(pose)
    pose_feat = (R[:, 1:] - jnp.eye(3)).reshape(b, -1)
    v_posed = v_shaped + jnp.einsum("vmp,bp->bvm", posedirs, pose_feat)
    # LBS chain via scan sobre el kinematic tree (omitida)
    return v_posed, J_rest
```

### 4.4 Fitting SMPL a observaciones — pseudo-código (Bogo et al. 2016, SMPLify)

```python
def fit_smpl_to_keypoints(target_keypoints_2d, camera_intrinsics,
                           smpl_model, num_iter=200, lr=1e-2):
    """
    Optimiza (betas, pose, translation) tal que los joints proyectados
    coinciden con los keypoints 2D observados (e.g. detectados por OpenPose).
    """
    betas = torch.zeros(1, 10, requires_grad=True)
    pose = torch.zeros(1, 72, requires_grad=True)
    trans = torch.zeros(1, 3, requires_grad=True)
    opt = torch.optim.LBFGS([betas, pose, trans], lr=lr)

    def closure():
        opt.zero_grad()
        verts, joints_3d = smpl_model(betas, pose)
        joints_3d = joints_3d + trans.unsqueeze(1)
        joints_2d = project(joints_3d, camera_intrinsics)   # (1, K, 2)
        # data term
        data_loss = ((joints_2d - target_keypoints_2d) ** 2).mean()
        # priors: shape, pose, GMM pose prior
        shape_prior = (betas ** 2).sum()
        pose_prior = pose_prior_gmm(pose)   # learned GMM sobre poses validas
        loss = data_loss + 0.01 * shape_prior + 0.1 * pose_prior
        loss.backward()
        return loss

    for _ in range(num_iter):
        opt.step(closure)
    return betas, pose, trans
```

## 5. Experimentos clave (Sección 5)

### 5.1 Comparación con BlendSCAPE (Tabla 11-12 del paper, Figuras 11-12)

| Modelo | Error mean abs (mm) — generalization | Error mean abs (mm) — pose generalization |
|---|---|---|
| LBS (vanilla) | ~7 (a 100 shape coefs) | ~7.5 |
| BlendSCAPE | ~3.5 | ~5 |
| **SMPL-LBS** | **~3.2** | **~4.5** |
| **SMPL-DQBS** | **~3.2** | **~4.0** |
| SMPL-LBS-Sparse | ~3.5 | ~5.5 |

SMPL (LBS o DQBS) supera consistentemente a BlendSCAPE en model generalization Y pose generalization, **a pesar de usar el método "más simple" (LBS)** — la clave son las pose blend shapes aprendidas.

### 5.2 Run-time performance (Sección 5.4)

- Single CPU core: ~5 ms por sample.
- BlendSCAPE: ~25 ms.
- En GPU con renderer integrado: tiempo real en Unity 5 game engine (slide 1 del paper).

## 6. Limitaciones reconocidas

1. **No modela cara y manos**: SMPL tiene un kinematic tree de 23 joints + root, sin articulación facial ni de dedos. Versiones posteriores:
   - **SMPL-H** (2017): manos articuladas.
   - **FLAME** (2017): cara articulada.
   - **SMPL-X** (Pavlakos et al., 2019): combinación completa cara + manos + cuerpo.
2. **Suposición de body topology única**: 6890 vértices fijos. No modela cuerpos con discapacidad (amputaciones, malformaciones).
3. **No incluye ropa**: el cuerpo SMPL es "desnudo". Para vestir requiere métodos adicionales (TailorNet, BCNet, ICON).
4. **Identity-shape lineal**: PCA es lineal — efectos no lineales (e.g., obesidad extrema con folds) no se capturan bien. **STAR** (Osman et al., 2020) corrige esto.
5. **Pose blend shapes lineales en R**: aunque mejor que en ángulos, sigue siendo aproximación local. Para poses extremas (yoga, contorsionismo), las correcciones pueden ser imperfectas.
6. **Data restringido**: scans CAESAR son sujetos en T-pose o A-pose en condiciones controladas. Diversidad demográfica limitada.

## 7. Impacto y legado

SMPL es **el modelo paramétrico de cuerpo humano más usado en la última década**. Casi cualquier paper de body shape/pose recovery posterior a 2015 lo cita o usa:

### Sucesores directos
- **SMPL-H** (Romero et al., 2017) — manos.
- **FLAME** (Li et al., 2017) — cara.
- **SMPL-X** (Pavlakos et al., 2019) — todo el cuerpo.
- **STAR** (Osman et al., ECCV 2020) — corrige no-linealidades.
- **SUPR** (Osman et al., ECCV 2022) — body part articulado (incluye cara, manos, dedos).
- **GHUM/GHUML** (Xu et al., CVPR 2020) — alternativa de Google con generative model VAE.

### Inferencia desde imagen
- **SMPLify** (Bogo et al., ECCV 2016) — fit clásico iterativo SMPL a 2D keypoints.
- **HMR** (Kanazawa et al., CVPR 2018) — regresión end-to-end de $(\vec\beta, \vec\theta)$ desde imagen.
- **VIBE** (Kocabas et al., CVPR 2020) — temporal HMR con GAN.
- **PIFu/PIFuHD** (Saito et al.) — surface fitting + SMPL prior.
- **PROX** (Hassan et al., 2019) — SMPL + interacción con escenas 3D.
- **PARE** (Kocabas et al., ICCV 2021) — SMPL fitting con part-attention.
- **4DHumans** (Goel et al., 2023) — temporal SMPL con transformer.

### Datasets que dependen de SMPL
- **DensePose-COCO**: ground-truth UV definido sobre la superficie SMPL.
- **AMASS** (Mahmood et al., 2019) — unifica datasets mocap en formato SMPL.
- **SURREAL** (Varol et al., 2017) — humanos sintéticos renderizados desde SMPL.
- **AGORA** (Patel et al., 2021) — escenas sintéticas con SMPL + ropa.

### En la industria
- **Apple, Meta, Adobe, Disney**: usan SMPL (o variantes propietarias inspiradas en SMPL) en pipelines de motion capture y avatares.
- **Unity Mecanim, Unreal MetaHuman**: framework de avatares con conceptos derivados de SMPL.
- **VR/AR**: cualquier pipeline de body tracking moderno (Quest, Vision Pro) usa modelos paramétricos en la familia SMPL.

## 8. Conexión con la clase 17

SMPL aparece explícitamente en el slide 35 del PDF: *"With the points correspondence they can generate the U and V images using the Skinned Multi-Person Linear (SMPL) model."*

Su rol pedagógico:

- Es el **modelo paramétrico subyacente a DensePose** — sin SMPL no hay U/V coordinates (las parametrizaciones UV provienen del unwrapping de la superficie SMPL).
- Demuestra el poder de los **modelos generativos paramétricos** del cuerpo: no es solo una nube de keypoints, es una superficie continua que se puede animar.
- Conecta visión por computador (DensePose, HMR) con **gráficos por computador** (Maya/Blender, animación de juegos).
- Establece el lenguaje $(\vec\beta, \vec\theta)$ que aparece en cualquier paper moderno de body pose estimation.

Cross-links:
- [[fundamentos/dense-correspondence.md]] — UV mapping definido sobre superficie SMPL.
- [[fundamentos/body-models.md]] — familia SMPL/SMPL-X/STAR.
- [[papers/DensePose-Guler-2018.md]] — su uso directo como ground-truth.
- [[clases/clase-17/teoria.md#densepose]] — sección de la clase.

## 9. Enlaces

- Project page: https://smpl.is.tue.mpg.de/
- PDF directo: https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf
- Código Python oficial (`smplx`): https://github.com/vchoutas/smplx
- Página de descarga (académica): https://smpl.is.tue.mpg.de/ (requiere registro académico para licencia)
- AMASS dataset: https://amass.is.tue.mpg.de/
- HMR codebase: https://github.com/akanazawa/hmr (referencia para fitting SMPL a imágenes)
