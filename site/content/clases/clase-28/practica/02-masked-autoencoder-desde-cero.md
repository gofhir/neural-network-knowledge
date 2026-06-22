---
title: "Masked Autoencoder desde cero"
weight: 2
math: true
---

El [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) tiene una de sus expresiones mas limpias en el **Masked Autoencoder** (MAE) de [He et al., 2022](/papers/mae-he-2022): esconde la mayor parte de una imagen, pide reconstruir lo que falta, y al hacerlo fuerza a la red a aprender una representacion semantica del contenido visual. No hay etiquetas humanas: la "respuesta correcta" son los propios pixeles que ocultamos. Es la version visual de la idea que BERT habia consagrado en texto —enmascarar tokens y predecirlos— y que la [Clase 28](/clases/clase-28) coloca en el corazon de la familia de *pretext tasks* de prediccion.

El MAE original opera sobre un [Vision Transformer](/papers/vit-dosovitskiy-2021): parte la imagen en parches, enmascara el 75% de ellos, pasa **solo los visibles** por un encoder Transformer pesado, y reconstruye los ocultos con un decoder ligero. En este camino vamos a construir ese **nucleo** desde cero —patchify, mascara aleatoria, encoder asimetrico, mask token, decoder, perdida MSE sobre los ocultos— sobre imagenes pequeñas (MNIST $28\times28$ o parches sinteticos), reemplazando el Transformer por capas lineales/MLP para que la mecanica quede transparente. La sustancia conceptual —**que** se enmascara, **que** ve el encoder, **donde** se mide la perdida— es identica a la del paper.

Lo implementamos en **tres frameworks** —PyTorch, TensorFlow/Keras y JAX— porque el algoritmo (parchear, muestrear una mascara, embeber visibles, rellenar con mask tokens, decodificar, MSE sobre ocultos) es el mismo en los tres, pero cada idioma expresa distinto el manejo de indices y de aleatoriedad. En JAX, ademas, el masking nos obliga a hacer explicito algo que en PyTorch y TF queda escondido: la generacion de numeros aleatorios como funcion pura de una `PRNGKey`.

---

## 1. La idea en cuatro piezas

Antes de tocar codigo, fijemos las cuatro decisiones de diseño que **son** el MAE, todas heredadas directamente del paper de He et al.

1. **Patchify.** La imagen $x \in \mathbb{R}^{H\times W\times C}$ se parte en una grilla de parches no solapados de lado $p$. Para MNIST $28\times28$ con $p=7$ obtenemos $\frac{28}{7}\cdot\frac{28}{7} = 4\cdot4 = 16$ parches, cada uno de $7\times7\times1 = 49$ valores. La imagen pasa de ser una grilla 2D a una **secuencia de $N=16$ vectores** de dimension $49$. Esta es exactamente la tokenizacion del [ViT](/papers/vit-dosovitskiy-2021).

2. **Mascara aleatoria de razon alta.** Muestreamos al azar un subconjunto de parches para **ocultar**. La razon de enmascaramiento del MAE es $\rho = 0.75$ —tres de cada cuatro parches desaparecen. Con $N=16$ parches, ocultamos $\lfloor 0.75\cdot 16\rfloor = 12$ y dejamos $16-12 = 4$ parches **visibles**. El muestreo es uniforme sin reemplazo (random shuffle + corte), no estructurado.

3. **Encoder asimetrico sobre los visibles.** El encoder $f_\phi$ procesa **solo** los parches visibles —en nuestro ejemplo, $4$ de $16$. Esta es la idea de eficiencia central del MAE: el encoder, que es la parte cara (en el paper, un ViT-Large), nunca ve los parches ocultos ni un mask token. Si el encoder cuesta proporcional a la longitud de secuencia, procesar $25\%$ de los parches cuesta $\approx 25\%$ del computo (en un Transformer la atencion es cuadratica, asi que el ahorro es aun mayor).

4. **Decoder ligero + MSE solo sobre los ocultos.** El decoder $g_\theta$ recibe la secuencia **completa** reconstruida posicionalmente: los embeddings de los visibles (salida del encoder) intercalados con un **mask token** aprendido $m$ repetido en cada posicion oculta. Predice los pixeles de **todos** los parches, pero la perdida solo se computa sobre los **ocultos**:

$$
\mathcal{L}(\phi,\theta) = \frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}} \big\lVert \hat{x}_i - x_i \big\rVert_2^2, \tag{1}
$$

donde $\mathcal{M}$ es el conjunto de indices ocultos, $x_i$ el parche original (vector de $p^2 C$ pixeles) y $\hat{x}_i$ su reconstruccion. El decoder es deliberadamente mas chico que el encoder: solo existe durante el pre-entrenamiento y se descarta despues.

{{< concept-alert type="clave" >}}
El MAE es **asimetrico** en dos sentidos a la vez: (a) el encoder ve solo una fraccion pequeña de la entrada (los visibles) mientras el decoder ve todo; (b) el encoder es pesado y el decoder ligero. La perdida vive **solo en los parches que el encoder nunca vio**, lo que convierte la reconstruccion en una verdadera tarea predictiva, no en una copia.
{{< /concept-alert >}}

### 1.1 Verificacion de dimensiones (hazla a mano una vez)

Vale la pena escribir la tabla de shapes para MNIST $28\times28$, $p=7$, $\rho=0.75$, batch $B$. Si tu codigo no produce estos shapes, hay un bug.

| Tensor | Shape | Comentario |
|---|---|---|
| Imagen $x$ | $(B, 28, 28, 1)$ | entrada (channels-last) |
| Parches | $(B, 16, 49)$ | $N=16$ parches de $p^2C = 49$ |
| Indices visibles | $(B, 4)$ | $N_{\text{vis}} = N(1-\rho) = 4$ |
| Indices ocultos | $(B, 12)$ | $N_{\text{mask}} = \lfloor\rho N\rfloor = 12$ |
| Parches visibles | $(B, 4, 49)$ | lo unico que entra al encoder |
| Embeddings visibles | $(B, 4, D)$ | salida del encoder, dim latente $D$ |
| Secuencia decoder | $(B, 16, D)$ | visibles + 12 mask tokens, re-ordenados |
| Reconstruccion $\hat{x}$ | $(B, 16, 49)$ | el decoder predice **todos** los parches |
| Perdida | escalar | MSE solo sobre los 12 ocultos |

El paso sutil es el **re-ordenamiento**: el encoder devuelve los visibles en el orden barajado; antes de decodificar hay que volver a colocar cada parche en su posicion original de la grilla (un *unshuffle*) e insertar los mask tokens en los huecos. Si te equivocas en ese reordenamiento, la perdida baja igual (el decoder aprende un promedio) pero las reconstrucciones salen espacialmente revueltas. Lo verificamos en cada framework.

### 1.2 Configuracion comun

Para que las tres implementaciones sean comparables, fijamos:

| Hiperparametro | Valor | Que controla |
|---|---:|---|
| Imagen $H\times W\times C$ | $28\times28\times1$ | MNIST en escala de grises |
| Lado de parche $p$ | 7 | da $N=16$ parches |
| Numero de parches $N$ | 16 | longitud de secuencia |
| Dim por parche $p^2 C$ | 49 | tamaño del vector de pixeles |
| Razon de mascara $\rho$ | 0.75 | fraccion oculta |
| Visibles $N_{\text{vis}}$ | 4 | lo que ve el encoder |
| Dim latente $D$ | 64 | ancho del encoder |
| Dim latente decoder $D_{\text{dec}}$ | 32 | decoder mas angosto (ligero) |

---

## 2. Seccion 1: PyTorch

PyTorch hace natural el manejo de indices con `torch.gather` y `torch.randperm`, que es justo lo que necesita el shuffle/unshuffle del masking.

### 2.1 Imports y configuracion

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Configuracion (ver tabla 1.2)
IMG, P, C = 28, 7, 1
N = (IMG // P) ** 2          # 16 parches
PATCH_DIM = P * P * C        # 49 valores por parche
MASK_RATIO = 0.75
N_VIS = int(N * (1 - MASK_RATIO))  # 4 visibles
D, D_DEC = 64, 32            # dim latente encoder / decoder
```

### 2.2 Patchify y un-patchify

Partir la imagen en parches es un reordenamiento de pixeles, no una convolucion. La forma mas transparente es reorganizar con `reshape` + `permute`.

```python
def patchify(x, p=P):
    """(B, C, H, W) -> (B, N, p*p*C). Parches no solapados, fila por fila."""
    B, C, H, W = x.shape
    nh, nw = H // p, W // p               # 4, 4
    # Desplegamos la grilla de parches:
    x = x.reshape(B, C, nh, p, nw, p)     # (B, C, 4, 7, 4, 7)
    x = x.permute(0, 2, 4, 1, 3, 5)       # (B, nh, nw, C, p, p)
    x = x.reshape(B, nh * nw, C * p * p)  # (B, 16, 49)
    return x

def unpatchify(patches, p=P, c=C):
    """(B, N, p*p*C) -> (B, C, H, W). Inversa exacta de patchify."""
    B, N, _ = patches.shape
    nh = nw = int(N ** 0.5)               # 4, 4
    x = patches.reshape(B, nh, nw, c, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)       # (B, C, nh, p, nw, p)
    x = x.reshape(B, c, nh * p, nw * p)   # (B, 1, 28, 28)
    return x
```

Conviene verificar que `unpatchify(patchify(x)) == x` antes de seguir: es el invariante que garantiza que no perdimos ni mezclamos pixeles. (`torch.allclose(unpatchify(patchify(x)), x)` debe dar `True`.)

### 2.3 La mascara aleatoria (shuffle / unshuffle)

El truco canonico del MAE para el masking, copiado del repo oficial: generamos **ruido uniforme** por parche, lo ordenamos (`argsort`), y los primeros `N_VIS` indices del orden son los visibles. Guardamos la permutacion inversa para poder restaurar el orden mas tarde.

```python
def random_masking(patches, n_vis=N_VIS):
    """
    patches: (B, N, PATCH_DIM)
    Devuelve:
      vis_patches  (B, n_vis, PATCH_DIM)  parches visibles, en orden barajado
      ids_restore  (B, N)                 permutacion para deshacer el shuffle
      mask         (B, N)                  1 = oculto, 0 = visible (orden original)
    """
    B, n, _ = patches.shape
    noise = torch.rand(B, n)                       # ruido U(0,1) por parche
    ids_shuffle = torch.argsort(noise, dim=1)      # orden creciente de ruido
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # permutacion inversa

    ids_keep = ids_shuffle[:, :n_vis]              # (B, n_vis) los visibles
    vis_patches = torch.gather(
        patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, patches.size(-1)))

    # mask en orden original: 0 para los primeros n_vis del shuffle, 1 el resto
    mask = torch.ones(B, n)
    mask[:, :n_vis] = 0
    mask = torch.gather(mask, 1, ids_restore)      # reordenar al orden original
    return vis_patches, ids_restore, mask
```

`ids_restore` es la pieza clave: `argsort(argsort(noise))` da la permutacion que **deshace** el barajado. La usaremos en el decoder para volver a colocar cada parche en su lugar. El `mask` (1 = oculto) lo necesitamos para computar la perdida solo sobre los ocultos.

### 2.4 Encoder asimetrico, mask token y decoder

```python
class MAE(nn.Module):
    def __init__(self, patch_dim=PATCH_DIM, d=D, d_dec=D_DEC, n=N):
        super().__init__()
        # --- Encoder: solo procesa parches VISIBLES ---
        self.patch_embed = nn.Linear(patch_dim, d)          # parche -> latente
        self.enc_pos = nn.Parameter(torch.randn(1, n, d) * 0.02)  # pos. encoder
        self.encoder = nn.Sequential(                       # MLP ligero (stand-in del ViT)
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

        # --- Decoder: ve TODA la secuencia ---
        self.enc_to_dec = nn.Linear(d, d_dec)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))  # token aprendido
        self.dec_pos = nn.Parameter(torch.randn(1, n, d_dec) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(d_dec, d_dec), nn.GELU())
        self.pred = nn.Linear(d_dec, patch_dim)             # reconstruye pixeles
        nn.init.normal_(self.mask_token, std=0.02)

    def forward_encoder(self, patches):
        vis, ids_restore, mask = random_masking(patches)    # (B, n_vis, PATCH_DIM)
        n_vis = vis.size(1)
        # Embeber visibles y sumarles SU posicion (las primeras n_vis del shuffle).
        # Para simplicidad didactica sumamos pos. tras el unshuffle (seccion 2.5).
        z = self.patch_embed(vis)                           # (B, n_vis, D)
        z = self.encoder(z)                                 # (B, n_vis, D)
        return z, ids_restore, mask

    def forward_decoder(self, z, ids_restore):
        B, n_vis, _ = z.shape
        z = self.enc_to_dec(z)                              # (B, n_vis, D_dec)
        n = ids_restore.size(1)
        # Rellenar con mask tokens hasta completar N, luego DESHACER el shuffle:
        mask_tokens = self.mask_token.expand(B, n - n_vis, -1)
        x = torch.cat([z, mask_tokens], dim=1)              # (B, N, D_dec) orden barajado
        x = torch.gather(                                   # unshuffle al orden original
            x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        x = x + self.dec_pos                                # pos. del decoder (orden original)
        x = self.decoder(x)
        return self.pred(x)                                 # (B, N, PATCH_DIM)

    def forward(self, imgs):
        patches = patchify(imgs)                            # (B, N, PATCH_DIM)
        z, ids_restore, mask = self.forward_encoder(patches)
        pred = self.forward_decoder(z, ids_restore)         # (B, N, PATCH_DIM)
        return pred, patches, mask
```

El punto central esta en `forward_decoder`: concatenamos los `n_vis` embeddings visibles con `n - n_vis` copias del **mask token**, y luego `gather` con `ids_restore` devuelve cada elemento a su posicion original en la grilla. Recien ahi sumamos la codificacion posicional del decoder (que esta en orden original) y predecimos.

### 2.5 La perdida MSE solo sobre los ocultos

```python
def mae_loss(pred, patches, mask):
    """MSE por parche, promediada SOLO sobre los parches ocultos (mask==1)."""
    loss = (pred - patches) ** 2          # (B, N, PATCH_DIM)
    loss = loss.mean(dim=-1)              # (B, N) error medio por parche
    # Promedio ponderado por la mascara: numerador / numero de ocultos
    loss = (loss * mask).sum() / mask.sum()
    return loss
```

El `(loss * mask).sum() / mask.sum()` es la Ecuacion (1): suma el error de los parches con `mask==1` y divide por cuantos hay. Los parches visibles **no** contribuyen al gradiente. Esto es deliberado: si penalizaramos tambien los visibles, la tarea se volveria parcialmente trivial (copiar lo que ya viste).

### 2.6 Mini loop de entrenamiento

```python
model = MAE()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Datos de juguete: un batch de "imagenes" aleatorias (sustituir por MNIST real)
imgs = torch.rand(64, C, IMG, IMG)

model.train()
for step in range(500):
    pred, patches, mask = model(imgs)
    loss = mae_loss(pred, patches, mask)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        # Sanidad: el error sobre visibles deberia ser >= el de ocultos NO es objetivo
        print(f"step {step:4d}  loss(ocultos)={loss.item():.4f}")
```

Con MNIST real veras la perdida caer de forma sostenida; con ruido puro se estanca (no hay estructura que predecir, que es justamente el punto: el MAE solo puede aprender si la señal **tiene** estructura espacial explotable).

---

## 3. Seccion 2: TensorFlow / Keras

El equivalente en TF 2.x usa `tf.argsort` para el shuffle y `tf.gather` con `batch_dims=1` para indexar por batch.

### 3.1 Patchify y masking

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

IMG, P, C = 28, 7, 1
N = (IMG // P) ** 2          # 16
PATCH_DIM = P * P * C        # 49
MASK_RATIO = 0.75
N_VIS = int(N * (1 - MASK_RATIO))   # 4
D, D_DEC = 64, 32


def patchify(x, p=P):
    """(B, H, W, C) -> (B, N, p*p*C). Keras usa channels-last."""
    B = tf.shape(x)[0]
    # tf.image.extract_patches hace el trabajo en una llamada:
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, p, p, 1], strides=[1, p, p, 1],
        rates=[1, 1, 1, 1], padding="VALID")            # (B, nh, nw, p*p*C)
    return tf.reshape(patches, (B, N, P * P * C))       # (B, 16, 49)


def random_masking(patches, n_vis=N_VIS):
    B = tf.shape(patches)[0]
    n = tf.shape(patches)[1]
    noise = tf.random.uniform((B, N))                   # ruido por parche
    ids_shuffle = tf.argsort(noise, axis=1)             # orden por ruido
    ids_restore = tf.argsort(ids_shuffle, axis=1)       # inversa
    ids_keep = ids_shuffle[:, :n_vis]                   # (B, n_vis)
    vis = tf.gather(patches, ids_keep, batch_dims=1)    # (B, n_vis, PATCH_DIM)

    # mask: 0 visibles, 1 ocultos, en orden ORIGINAL
    mask = tf.concat([tf.zeros((B, n_vis)),
                      tf.ones((B, N - n_vis))], axis=1)
    mask = tf.gather(mask, ids_restore, batch_dims=1)   # reordenar a original
    return vis, ids_restore, mask
```

`tf.image.extract_patches` ya devuelve los parches aplanados fila por fila, equivalente al `reshape`/`permute` de PyTorch. El `batch_dims=1` de `tf.gather` indexa cada fila del batch con su propia lista de indices —el analogo de `torch.gather`.

### 3.2 El modelo MAE

```python
class MAE(keras.Model):
    def __init__(self, patch_dim=PATCH_DIM, d=D, d_dec=D_DEC, n=N, **kw):
        super().__init__(**kw)
        self.patch_embed = layers.Dense(d)
        self.encoder = keras.Sequential(
            [layers.Dense(d), layers.Activation("gelu"), layers.Dense(d)])
        self.enc_to_dec = layers.Dense(d_dec)
        # mask token y posicionales como pesos entrenables:
        self.mask_token = self.add_weight(
            shape=(1, 1, d_dec), initializer=keras.initializers.RandomNormal(0., 0.02),
            trainable=True, name="mask_token")
        self.dec_pos = self.add_weight(
            shape=(1, n, d_dec), initializer=keras.initializers.RandomNormal(0., 0.02),
            trainable=True, name="dec_pos")
        self.decoder = keras.Sequential([layers.Dense(d_dec), layers.Activation("gelu")])
        self.pred = layers.Dense(patch_dim)

    def call(self, imgs, training=False):
        patches = patchify(imgs)                            # (B, N, PATCH_DIM)
        vis, ids_restore, mask = random_masking(patches)

        # --- Encoder solo sobre visibles ---
        z = self.encoder(self.patch_embed(vis))             # (B, n_vis, D)

        # --- Decoder ve todo ---
        B = tf.shape(z)[0]
        n_vis = tf.shape(z)[1]
        z = self.enc_to_dec(z)                              # (B, n_vis, D_dec)
        mask_tokens = tf.tile(self.mask_token, [B, N - n_vis, 1])
        x = tf.concat([z, mask_tokens], axis=1)             # (B, N, D_dec) barajado
        x = tf.gather(x, ids_restore, batch_dims=1)         # unshuffle a original
        x = x + self.dec_pos
        x = self.decoder(x)
        pred = self.pred(x)                                 # (B, N, PATCH_DIM)
        return pred, patches, mask
```

### 3.3 Perdida y loop con `GradientTape`

```python
def mae_loss(pred, patches, mask):
    loss = tf.reduce_mean(tf.square(pred - patches), axis=-1)  # (B, N)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)


model = MAE()
opt = keras.optimizers.AdamW(learning_rate=1e-3)
imgs = tf.random.uniform((64, IMG, IMG, C))

@tf.function
def train_step(imgs):
    with tf.GradientTape() as tape:
        pred, patches, mask = model(imgs, training=True)
        loss = mae_loss(pred, patches, mask)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for step in range(500):
    loss = train_step(imgs)
    if step % 100 == 0:
        print(f"step {step:4d}  loss(ocultos)={float(loss):.4f}")
```

Misma estructura que PyTorch; lo unico distinto es el andamiaje (`GradientTape` en vez de `backward`, `add_weight` para el mask token, channels-last desde el principio).

---

## 4. Seccion 3: JAX

En JAX el modelo es una **funcion pura**: recibe sus parametros como argumento y no muta estado. Lo mas instructivo aqui es el masking: la aleatoriedad **no** viene de un generador global escondido (como `torch.rand` o `tf.random`), sino de una `PRNGKey` explicita que pasamos como argumento. Esto hace el muestreo de la mascara **reproducible y trazable** —dos llamadas con la misma key dan exactamente la misma mascara.

### 4.1 Imports, patchify y masking puro

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

IMG, P, C = 28, 7, 1
N = (IMG // P) ** 2          # 16
PATCH_DIM = P * P * C        # 49
MASK_RATIO = 0.75
N_VIS = int(N * (1 - MASK_RATIO))   # 4
D, D_DEC = 64, 32


def patchify(x, p=P):
    """(B, H, W, C) -> (B, N, p*p*C), channels-last."""
    B, H, W, c = x.shape
    nh, nw = H // p, W // p
    x = x.reshape(B, nh, p, nw, p, c)        # despliega la grilla
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)) # (B, nh, nw, p, p, c)
    return x.reshape(B, nh * nw, p * p * c)  # (B, 16, 49)


def random_masking(patches, key, n_vis=N_VIS):
    """Mascara aleatoria DETERMINISTA dada la key (funcion pura)."""
    B, n, _ = patches.shape
    noise = random.uniform(key, (B, n))              # la key gobierna el muestreo
    ids_shuffle = jnp.argsort(noise, axis=1)
    ids_restore = jnp.argsort(ids_shuffle, axis=1)
    ids_keep = ids_shuffle[:, :n_vis]

    # gather por batch con take_along_axis:
    vis = jnp.take_along_axis(
        patches, ids_keep[:, :, None], axis=1)        # (B, n_vis, PATCH_DIM)
    mask = jnp.concatenate(
        [jnp.zeros((B, n_vis)), jnp.ones((B, n - n_vis))], axis=1)
    mask = jnp.take_along_axis(mask, ids_restore, axis=1)
    return vis, ids_restore, mask
```

El `random.uniform(key, ...)` es el corazon del estilo JAX: la misma `key` produce siempre el mismo `noise`, luego la misma permutacion, luego la misma mascara. Para variar la mascara entre pasos, hay que **dividir** la key (`random.split`) y pasar una key fresca en cada llamada —lo hacemos en el loop. `jnp.take_along_axis` es el equivalente de `torch.gather` / `tf.gather(batch_dims=1)`.

### 4.2 Inicializacion de parametros y forward puro

Sin clases ni capas con estado: definimos un dict de parametros y una funcion `apply` que los consume.

```python
def init_params(key):
    """Inicializa todos los pesos del MAE en un pytree (dict anidado)."""
    keys = random.split(key, 8)
    glorot = jax.nn.initializers.glorot_uniform()
    def lin(k, fin, fout):                         # capa lineal: W (fin,fout), b (fout,)
        return {"w": glorot(k, (fin, fout)), "b": jnp.zeros(fout)}
    return {
        "patch_embed": lin(keys[0], PATCH_DIM, D),
        "enc1": lin(keys[1], D, D), "enc2": lin(keys[2], D, D),
        "enc_to_dec": lin(keys[3], D, D_DEC),
        "dec1": lin(keys[4], D_DEC, D_DEC),
        "pred": lin(keys[5], D_DEC, PATCH_DIM),
        "mask_token": random.normal(keys[6], (1, 1, D_DEC)) * 0.02,
        "dec_pos": random.normal(keys[7], (1, N, D_DEC)) * 0.02,
    }


def dense(p, x):                                   # x @ W + b
    return x @ p["w"] + p["b"]


def apply_mae(params, imgs, key):
    """Forward puro: (params, imagenes, key del masking) -> (pred, patches, mask)."""
    patches = patchify(imgs)                                  # (B, N, PATCH_DIM)
    vis, ids_restore, mask = random_masking(patches, key)    # (B, n_vis, PATCH_DIM)

    # --- Encoder solo sobre visibles ---
    z = dense(params["patch_embed"], vis)                    # (B, n_vis, D)
    z = dense(params["enc2"], jax.nn.gelu(dense(params["enc1"], z)))

    # --- Decoder ve la secuencia completa ---
    B, n_vis, _ = z.shape
    z = dense(params["enc_to_dec"], z)                       # (B, n_vis, D_dec)
    mask_tokens = jnp.broadcast_to(
        params["mask_token"], (B, N - n_vis, D_DEC))
    x = jnp.concatenate([z, mask_tokens], axis=1)            # (B, N, D_dec) barajado
    # unshuffle: take_along_axis necesita los indices expandidos a la dim de features
    idx = jnp.broadcast_to(ids_restore[:, :, None], (B, N, D_DEC))
    x = jnp.take_along_axis(x, idx, axis=1)                  # a orden original
    x = x + params["dec_pos"]
    x = jax.nn.gelu(dense(params["dec1"], x))
    pred = dense(params["pred"], x)                          # (B, N, PATCH_DIM)
    return pred, patches, mask
```

### 4.3 Perdida, gradiente y loop

```python
def mae_loss(params, imgs, key):
    pred, patches, mask = apply_mae(params, imgs, key)
    err = jnp.mean((pred - patches) ** 2, axis=-1)           # (B, N)
    return jnp.sum(err * mask) / jnp.sum(mask)               # MSE solo ocultos


key = random.PRNGKey(42)
key, init_key = random.split(key)
params = init_params(init_key)
opt = optax.adamw(learning_rate=1e-3)
opt_state = opt.init(params)

imgs = random.uniform(random.PRNGKey(0), (64, IMG, IMG, C))

@jax.jit
def train_step(params, opt_state, key, imgs):
    loss, grads = jax.value_and_grad(mae_loss)(params, imgs, key)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for step in range(500):
    key, subkey = random.split(key)                          # key fresca -> mascara nueva
    params, opt_state, loss = train_step(params, opt_state, subkey, imgs)
    if step % 100 == 0:
        print(f"step {step:4d}  loss(ocultos)={float(loss):.4f}")
```

Lo que hay que internalizar del patron JAX:

- **El masking es funcion pura de la key.** Pasamos `subkey` distinta en cada paso para que la mascara cambie; con la misma key, la mascara seria identica. Es el control explicito de la aleatoriedad que JAX exige y que aqui resulta didactico: el masking deja de ser "magia global".
- **Los parametros son un pytree.** `mask_token` y `dec_pos` viven en el mismo dict que las capas lineales; `jax.value_and_grad` diferencia respecto a todo el arbol de una vez.
- **`@jax.jit`** compila el paso completo a XLA; la primera llamada compila, las siguientes vuelan.

---

## 5. Comparacion lado a lado

Las tres implementaciones son **isomorfas**: mismo patchify, misma mascara via argsort de ruido, mismo encoder asimetrico, mismo mask token + unshuffle, misma MSE sobre ocultos. Cambia el idioma.

| Concepto | PyTorch | TensorFlow/Keras | JAX |
|---|---|---|---|
| Patchify | `reshape` + `permute` | `tf.image.extract_patches` | `reshape` + `transpose` |
| Ruido para mascara | `torch.rand` | `tf.random.uniform` | `random.uniform(key, ...)` |
| Permutacion / inversa | `torch.argsort` x2 | `tf.argsort` x2 | `jnp.argsort` x2 |
| Gather por batch | `torch.gather` | `tf.gather(batch_dims=1)` | `jnp.take_along_axis` |
| Mask token | `nn.Parameter` | `add_weight` | entrada del pytree |
| Aleatoriedad | global (seed) | global (seed) | **explicita** (`PRNGKey`) |
| Diferenciacion | `loss.backward()` | `tf.GradientTape` | `jax.value_and_grad` |
| Layout imagen | channels-first | channels-last | channels-last |

La leccion: el MAE es corto —parchear, enmascarar, embeber visibles, rellenar con mask tokens, decodificar, MSE sobre ocultos— asi que portarlo es casi mecanico. El unico punto donde los frameworks divergen de verdad es el **manejo de la aleatoriedad del masking**: implicito en PyTorch/TF, explicito y trazable en JAX.

---

## 6. Por que funciona: las tres intuiciones que cierran el MAE

### 6.1 La mascara alta hace la tarea no trivial

¿Por que $75\%$ y no $15\%$ como en BERT? Porque las imagenes son **espacialmente muy redundantes**: un pixel se puede adivinar casi siempre desde sus vecinos inmediatos. Si solo ocultaramos $15\%$ de los parches, el modelo reconstruiria interpolando localmente —copiando el contexto adyacente— sin necesidad de entender **que** es el objeto. Eso es una tarea de bajo nivel que no produce representaciones semanticas utiles.

Al ocultar el $75\%$, eliminamos esa salida facil: ya no quedan vecinos cercanos de los cuales copiar. Para reconstruir un parche oculto, el modelo se ve **forzado a inferir desde un contexto global y disperso** —tiene que "saber" que hay un trazo de un 7, o el lazo de un 4, para completar la region faltante de forma coherente. La razon de mascara alta es lo que convierte una tarea de interpolacion (trivial) en una tarea de comprension (la que produce buenas representaciones). El paper de He et al. lo verifica empiricamente: la accuracy de fine-tuning sube hasta un optimo cerca de $\rho = 0.75$ y solo cae con razones extremas.

Hay un contraste fino con el texto: BERT enmascara solo $\sim15\%$ de los tokens porque el lenguaje es mucho **menos redundante** que la imagen —cada palabra carga informacion densa. Las imagenes toleran (y exigen) razones de mascara mucho mas altas precisamente por su redundancia. Mismo principio, distinto punto de operacion segun la densidad informativa del medio.

### 6.2 El ahorro de computo del encoder asimetrico

La segunda gran idea del MAE es de **eficiencia**, y es la razon por la que el metodo escala a modelos enormes. El encoder procesa **solo los parches visibles** —en nuestro ejemplo, $4$ de $16$, un $25\%$ de la secuencia. Nunca ve los parches ocultos ni un mask token.

El impacto es doble:

- **Costo lineal:** una capa que cuesta proporcional a la longitud de secuencia procesa $25\%$ de los tokens, asi que cuesta $\approx 25\%$.
- **Costo cuadratico (Transformers):** la auto-atencion del ViT real es $O(L^2)$ en la longitud $L$. Procesar $L/4$ tokens cuesta $\approx (1/4)^2 = 1/16$ de la atencion. Combinado, el MAE entrena el encoder con una fraccion pequeña del computo de procesar la imagen completa.

El decoder, que si ve la secuencia completa ($N=16$ posiciones), se mantiene **deliberadamente ligero** (mas angosto y con menos capas: en nuestro codigo $D_{\text{dec}}=32$ vs $D=64$). Y crucialmente, **se descarta despues del pre-entrenamiento**: para la tarea final (clasificacion, deteccion) solo conservamos el encoder. Toda la inversion de computo va donde importa —el encoder que produce la representacion transferible— y el masking alto la abarata. Esa asimetria es lo que permitio a He et al. pre-entrenar ViT-Huge de forma practica.

### 6.3 La conexion con BERT MLM

El MAE es, conceptualmente, **BERT para imagenes**. El *Masked Language Modeling* de BERT enmascara $\sim15\%$ de los tokens de una oracion y entrena el modelo a predecirlos desde el contexto bidireccional. El MAE enmascara parches de una imagen y los predice desde los parches visibles. La estructura es identica:

| | BERT (MLM) | MAE |
|---|---|---|
| Unidad | token (palabra) | parche de imagen |
| Que se oculta | $\sim15\%$ de tokens | $\sim75\%$ de parches |
| Objetivo | predecir el token oculto | reconstruir pixeles ocultos |
| Que ve el modelo | secuencia con `[MASK]` | solo visibles (encoder) |
| Donde mide la perdida | solo tokens enmascarados | solo parches ocultos |
| Resultado | representacion transferible | representacion transferible |

Las dos diferencias de fondo explican el resto del diseño del MAE. **Primera, la redundancia** (seccion 6.1): texto $15\%$, imagen $75\%$. **Segunda, el grano del objetivo:** BERT predice sobre un vocabulario discreto (clasificacion sobre miles de palabras), mientras el MAE predice **valores continuos de pixeles** con MSE (regresion). Esa diferencia es la que vuelve util el decoder de reconstruccion: en texto la "cabeza" de prediccion es trivial (un softmax sobre el vocabulario), pero reconstruir pixeles requiere un pequeño decoder que mapee del espacio latente de vuelta al espacio de pixeles.

El linaje es directo: el MAE toma el *masked autoencoding* que BERT consagro en NLP, lo adapta a la naturaleza continua y redundante de la imagen (mascara alta + decoder de pixeles + encoder asimetrico para abaratar), y lo monta sobre el [ViT](/papers/vit-dosovitskiy-2021) que ya habia traido los Transformers a vision. Es la sintesis de ambos mundos —y la pieza que faltaba para que la autosupervision por reconstruccion finalmente funcionara a escala en vision.

---

**Ver tambien:** [Clase 28 - Aprendizaje Autosupervisado](/clases/clase-28) · [Fundamento aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) · [Paper MAE (He et al. 2022)](/papers/mae-he-2022) · [Paper ViT (Dosovitskiy et al. 2021)](/papers/vit-dosovitskiy-2021) · [Teoria de la clase](../teoria).
