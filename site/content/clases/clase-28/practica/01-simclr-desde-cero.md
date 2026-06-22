---
title: "SimCLR / NT-Xent desde cero"
weight: 1
math: true
---

SimCLR (Chen et al., ICML 2020) demostro algo casi provocador: para aprender buenas representaciones de imagen **sin etiquetas** no hacen falta arquitecturas exoticas, *memory banks* ni mecanismos sofisticados. Basta combinar bien cuatro piezas conocidas —augmentaciones compuestas, un encoder, una cabeza de proyeccion y una perdida contrastiva con temperatura— y tener un **batch grande**. El corazon de todo, lo unico verdaderamente nuevo en la receta, es la perdida **NT-Xent** (*Normalized Temperature-scaled Cross-Entropy*). En este capitulo la construimos desde cero, paso a paso, en los tres frameworks, sobre un *toy dataset* para que cada dimension y cada mascara queden a la vista.

La fundamentacion conceptual del paradigma esta en el [fundamento de aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo), y el analisis del paper original en [SimCLR (Chen et al. 2020)](/papers/simclr-chen-2020). Aqui nos concentramos en el codigo correcto y verificable: queremos que cuando termines puedas escribir `nt_xent_loss(z1, z2, tau)` de memoria y entiendas por que cada linea esta donde esta.

---

## 1. La idea en una frase y media

El pipeline de SimCLR cabe en cuatro pasos:

1. Toma un *minibatch* de $N$ imagenes.
2. A cada imagen aplicale **dos augmentaciones aleatorias distintas** (recorte + jitter de color, por ejemplo). Ahora tienes $2N$ **vistas**.
3. Pasa las $2N$ vistas por un encoder $f$ (una ResNet) y luego por una cabeza de proyeccion $g$ (un MLP), obteniendo $2N$ embeddings $z \in \mathbb{R}^D$.
4. Entrena para que las **dos vistas de la misma imagen** (el par positivo) queden cercanas en el espacio de $z$, y lejos de **todas las demas vistas del batch** (los negativos).

La pieza 4 es NT-Xent. Para un par positivo $(i, j)$ —las dos vistas de una misma imagen— la perdida sobre la vista $i$ es:

$$
\ell_{i,j} = -\log \frac{\exp\big(\text{sim}(z_i, z_j)/\tau\big)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp\big(\text{sim}(z_i, z_k)/\tau\big)}. \tag{1}
$$

Tres cosas que leer en esta ecuacion antes de tocar codigo:

- **$\text{sim}(u, v)$ es la similitud coseno**, $\dfrac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}$. En la practica L2-normalizamos los $z$ primero, y entonces la similitud se reduce al producto punto $z_i^\top z_j$.
- **El denominador suma sobre todas las vistas menos la propia** (la condicion $\mathbb{1}_{[k \neq i]}$). El unico termino "bueno" en el numerador es el positivo $j$; todo lo demas en el denominador son negativos que queremos empujar abajo. Esto es, literalmente, una **cross-entropy de $(2N{-}1)$ clases** donde la "clase correcta" es el indice del positivo.
- **$\tau$ es la temperatura** ($\tau > 0$, tipicamente $0.1$–$0.5$). Divide las similitudes antes del softmax. La perdida total es el promedio de $\ell_{i,j}$ sobre **los $2N$** terminos (cada vista actua como ancla una vez).

{{< concept-alert type="clave" >}}
NT-Xent **no es una perdida nueva y rara**: es exactamente la cross-entropy softmax de toda la vida, aplicada sobre una matriz de similitudes $2N \times 2N$ donde la etiqueta de cada fila es "donde esta mi par positivo". Si entiendes `F.cross_entropy`, ya entiendes el 90% de SimCLR.
{{< /concept-alert >}}

### 1.1 El truco de armar la cross-entropy

La clave de implementacion —y la fuente de casi todos los bugs— es como construir las etiquetas. Concatenamos las dos ramas en un solo tensor $z = [z_1; z_2]$ de shape $(2N, D)$:

- Las filas $0 \dots N{-}1$ son las vistas de la rama 1.
- Las filas $N \dots 2N{-}1$ son las vistas de la rama 2.
- El positivo de la fila $i$ (vista 1 de la imagen $i$) es la fila $i+N$ (vista 2 de la misma imagen).
- Y simetricamente, el positivo de la fila $i+N$ es la fila $i$.

Por lo tanto el vector de etiquetas es:

$$
\text{target} = [\,N, N{+}1, \dots, 2N{-}1,\ 0, 1, \dots, N{-}1\,].
$$

La diagonal de la matriz de similitud (cada fila consigo misma, $\text{sim}(z_i, z_i) = 1$) hay que **enmascararla a $-\infty$**: si la dejaramos, seria el logit mas alto y el softmax la elegiria siempre. La excluimos poniendo $-\infty$ (o un numero muy negativo) en la diagonal, que tras `exp` se vuelve $0$ —exactamente la condicion $\mathbb{1}_{[k \neq i]}$ de la Ecuacion (1).

### 1.2 Verificacion de dimensiones

Fijemos los shapes de una vez, porque son el mejor chequeo de sanidad:

| Objeto | Shape | Comentario |
|---|---|---|
| `z1`, `z2` | $(N, D)$ | embeddings de cada rama |
| `z` concatenado | $(2N, D)$ | las $2N$ vistas |
| matriz de similitud `sim` | $(2N, 2N)$ | todos contra todos |
| mascara diagonal | $(2N, 2N)$ | $-\infty$ en `sim[i,i]` |
| `target` | $(2N,)$ | indice del positivo de cada fila |
| `loss` | escalar | promedio de las $2N$ cross-entropies |

Con $N = 4$ imagenes, $D = 8$: `sim` es $8 \times 8$, `target` es `[4,5,6,7,0,1,2,3]`. Tenerlo claro a mano evita el 100% de los errores de indexado.

---

## 2. Seccion 1: PyTorch

PyTorch es el framework dominante en la investigacion autosupervisada; los repos oficiales de SimCLR, MoCo y SimSiam estan aqui. Su estilo *define-by-run* hace que la matriz de similitud se sienta tan natural como escribir la ecuacion.

### 2.1 Imports y configuracion

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

N = 8        # imagenes por minibatch  -> 2N = 16 vistas
D = 64       # dimension del embedding de proyeccion
TAU = 0.5    # temperatura
```

### 2.2 El nucleo: NT-Xent

Esta es la funcion completa. Recibe los dos lotes de embeddings ya proyectados (uno por augmentacion) y devuelve la perdida escalar.

```python
def nt_xent_loss(z1, z2, tau=0.5):
    """NT-Xent (SimCLR) sobre un batch de N pares.
    z1, z2: (N, D)  embeddings de las dos augmentaciones de las MISMAS N imagenes,
            alineados por fila: z1[i] y z2[i] son las dos vistas de la imagen i.
    Devuelve: perdida escalar (promedio sobre las 2N anclas).
    """
    N = z1.shape[0]
    device = z1.device

    # 1) Apilar las dos ramas en 2N vistas y L2-normalizar
    z = torch.cat([z1, z2], dim=0)        # (2N, D)
    z = F.normalize(z, dim=1)             # cada fila con norma 1 -> producto punto = coseno

    # 2) Matriz de similitud coseno escalada por temperatura
    sim = (z @ z.t()) / tau              # (2N, 2N): sim[i, k] = <z_i, z_k> / tau

    # 3) Enmascarar la diagonal (una vista NO es negativo ni positivo de si misma)
    mask = torch.eye(2 * N, dtype=torch.bool, device=device)
    sim.masked_fill_(mask, float("-inf"))  # exp(-inf) = 0 -> se excluye del softmax

    # 4) Etiquetas: el positivo de la fila i (i<N) es i+N, y el de i+N es i
    targets = torch.cat([torch.arange(N, 2 * N),
                         torch.arange(0, N)]).to(device)   # (2N,)

    # 5) Cross-entropy 2N-vias: trata cada fila como un problema de clasificacion
    #    donde la "clase correcta" es el indice del positivo.
    loss = F.cross_entropy(sim, targets)
    return loss
```

Cinco pasos, y cada uno mapea a una parte de la Ecuacion (1):

- **`torch.cat` + `F.normalize`** arma las $2N$ vistas y las pone en la hiperesfera unitaria. Tras normalizar, `z @ z.t()` es **directamente** la matriz de cosenos: no hace falta dividir por normas porque ya valen 1.
- **`(z @ z.t()) / tau`** es la matriz $(2N, 2N)$ de similitudes escaladas. La fila $i$ contiene $\text{sim}(z_i, z_k)/\tau$ para todo $k$ —exactamente los argumentos de los `exp` de la Ecuacion (1).
- **`masked_fill_(mask, -inf)`** implementa la condicion $\mathbb{1}_{[k \neq i]}$: pone $-\infty$ en la diagonal, que tras el softmax interno de la cross-entropy aporta peso $0$.
- **`targets`** es el vector `[N..2N-1, 0..N-1]` de la seccion 1.1: para cada fila, el indice de su par positivo.
- **`F.cross_entropy(sim, targets)`** hace el resto: aplica `log_softmax` sobre cada fila (numerador = logit del positivo, denominador = suma de todos los no enmascarados) y promedia el $-\log$ sobre las $2N$ filas. Es la Ecuacion (1) promediada, en una linea.

{{< concept-alert type="advertencia" >}}
**No apliques softmax dos veces.** `F.cross_entropy` ya hace `log_softmax` internamente, asi que le pasamos los **logits crudos** `sim` (las similitudes escaladas), nunca un softmax pre-aplicado. Es el bug numero uno al portar esta perdida entre frameworks.
{{< /concept-alert >}}

### 2.3 Verificacion rapida de sanidad

Antes del entrenamiento, conviene comprobar que la perdida se comporta como debe: baja cuando los positivos estan cerca, alta cuando todo es aleatorio.

```python
# Caso A: las dos vistas son casi identicas -> par positivo "facil" -> loss baja
z1 = torch.randn(N, D)
z2 = z1 + 0.01 * torch.randn(N, D)
print("loss (positivos cercanos):", nt_xent_loss(z1, z2, TAU).item())

# Caso B: la segunda vista es ruido independiente -> nada que aprender -> loss alta
z2_rand = torch.randn(N, D)
print("loss (aleatorio):", nt_xent_loss(z1, z2_rand, TAU).item())

# Cota teorica de referencia: con logits ~uniformes, loss ~ log(2N - 1)
import math
print("cota log(2N-1):", math.log(2 * N - 1))
```

El caso A debe dar una perdida claramente menor que el caso B, y el caso aleatorio debe rondar $\log(2N{-}1)$ —el valor de una cross-entropy de $2N{-}1$ clases equiprobables. Si tu funcion no respeta esto, hay un bug en la mascara o en los `targets`.

### 2.4 Encoder pequeno + cabeza de proyeccion + loop de entrenamiento

Ahora el pipeline completo sobre imagenes diminutas (estilo MNIST/CIFAR reducido). El encoder es una CNN minima; la **cabeza de proyeccion** $g$ es un MLP que mapea la representacion al espacio donde se aplica la perdida. Detalle clave de SimCLR: la perdida se aplica sobre $g(h)$, pero para *downstream* se usa $h$ (la salida del encoder, antes de la proyeccion).

```python
class TinyEncoder(nn.Module):
    """Encoder f: imagen pequena -> representacion h."""
    def __init__(self, in_ch=1, rep_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28->14
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(), nn.MaxPool2d(2),  # 14->7
            nn.AdaptiveAvgPool2d(1),                                          # -> (64,1,1)
        )
        self.fc = nn.Linear(64, rep_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)   # (B, 64)
        return self.fc(h)             # (B, rep_dim)  <- representacion h


class ProjectionHead(nn.Module):
    """Cabeza g: h -> z (MLP de 2 capas). La perdida vive en el espacio de z."""
    def __init__(self, rep_dim=128, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim, rep_dim), nn.ReLU(),
            nn.Linear(rep_dim, proj_dim),
        )

    def forward(self, h):
        return self.net(h)            # (B, proj_dim)


def augment(x):
    """Augmentacion de juguete: ruido + flip horizontal aleatorio.
    En SimCLR real seria RandomResizedCrop + ColorJitter + GaussianBlur.
    """
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-1])          # flip horizontal
    return x + 0.1 * torch.randn_like(x)      # jitter


encoder = TinyEncoder(in_ch=1, rep_dim=128)
head = ProjectionHead(rep_dim=128, proj_dim=64)
params = list(encoder.parameters()) + list(head.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

encoder.train(); head.train()
for step in range(500):
    # batch de N imagenes pequenas (aqui sinteticas; reemplaza por tu DataLoader)
    images = torch.randn(N, 1, 28, 28)

    # DOS augmentaciones independientes de las MISMAS imagenes -> 2 vistas alineadas
    v1, v2 = augment(images), augment(images)
    z1 = head(encoder(v1))   # (N, 64)
    z2 = head(encoder(v2))   # (N, 64)

    loss = nt_xent_loss(z1, z2, TAU)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step:4d}  loss={loss.item():.4f}")
```

Notese que `v1` y `v2` salen de las **mismas** `images` con augmentaciones distintas: por eso `z1[i]` y `z2[i]` son el par positivo y la alineacion por fila que asume `nt_xent_loss` se cumple. Con datos sinteticos la perdida no convergera a algo significativo (no hay estructura que aprender), pero el loop es identico al real: cambia `images = torch.randn(...)` por tu `DataLoader` de MNIST/CIFAR y `augment` por las transformaciones de `torchvision` y tendras SimCLR funcional.

---

## 3. Seccion 2: TensorFlow / Keras

TensorFlow 2.x expresa lo mismo con operaciones `tf.*` y `tf.GradientTape` para los gradientes. La logica de indexado es identica; solo cambia el idioma de los tensores.

### 3.1 El nucleo: NT-Xent con `tf`

```python
import tensorflow as tf

def nt_xent_loss(z1, z2, tau=0.5):
    """NT-Xent en TensorFlow. z1, z2: (N, D) alineados por fila."""
    N = tf.shape(z1)[0]
    two_n = 2 * N

    # 1) Apilar y L2-normalizar
    z = tf.concat([z1, z2], axis=0)              # (2N, D)
    z = tf.math.l2_normalize(z, axis=1)

    # 2) Matriz de similitud coseno / temperatura
    sim = tf.matmul(z, z, transpose_b=True) / tau  # (2N, 2N)

    # 3) Enmascarar la diagonal con un valor muy negativo (~ -inf)
    mask = tf.eye(two_n) * -1e9                   # -1e9 en la diagonal, 0 fuera
    sim = sim + mask                              # exp(-1e9) ~ 0

    # 4) Etiquetas: positivo de i es i+N, y de i+N es i
    targets = tf.concat([tf.range(N, two_n),
                         tf.range(0, N)], axis=0)  # (2N,)

    # 5) Cross-entropy con from_logits=True (Keras aplica el log_softmax)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        targets, sim, from_logits=True)           # (2N,)
    return tf.reduce_mean(loss)
```

Mapeo de operaciones respecto a PyTorch:

- `torch.cat` → `tf.concat`
- `F.normalize` → `tf.math.l2_normalize`
- `z @ z.t()` → `tf.matmul(z, z, transpose_b=True)`
- `masked_fill_(eye, -inf)` → sumar `tf.eye(2N) * -1e9` (TF no tiene `-inf` comodo en `matmul`, asi que usamos un numero enorme negativo; `exp(-1e9)` es indistinguible de 0)
- `F.cross_entropy(logits, targets)` → `sparse_categorical_crossentropy(targets, logits, from_logits=True)`

El `from_logits=True` es **obligatorio**: igual que en PyTorch, le pasamos las similitudes crudas y dejamos que Keras haga el `log_softmax` por dentro. Pasar probabilidades ya normalizadas seria el bug del doble softmax.

### 3.2 Encoder + cabeza + paso de entrenamiento

```python
from tensorflow import keras
from tensorflow.keras import layers

def build_encoder(rep_dim=128):
    # OJO: Keras usa channels-last (B, H, W, C)
    return keras.Sequential([
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPool2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPool2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(rep_dim),                # representacion h
    ])

def build_head(rep_dim=128, proj_dim=64):
    return keras.Sequential([
        layers.Dense(rep_dim, activation="relu"),
        layers.Dense(proj_dim),               # proyeccion z
    ])

encoder = build_encoder()
head = build_head()
optimizer = keras.optimizers.Adam(1e-3)

def augment(x):
    x = tf.image.random_flip_left_right(x)
    return x + 0.1 * tf.random.normal(tf.shape(x))

@tf.function
def train_step(images):
    v1, v2 = augment(images), augment(images)
    with tf.GradientTape() as tape:
        z1 = head(encoder(v1, training=True), training=True)
        z2 = head(encoder(v2, training=True), training=True)
        loss = nt_xent_loss(z1, z2, 0.5)
    variables = encoder.trainable_variables + head.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss

for step in range(500):
    images = tf.random.normal((8, 28, 28, 1))   # (B, H, W, C) channels-last
    loss = train_step(images)
    if step % 100 == 0:
        print(f"step {step:4d}  loss={float(loss):.4f}")
```

El `@tf.function` traza el grafo una vez y lo reutiliza (equivalente a `torch.compile` o `jax.jit`). `tf.GradientTape` es el analogo explicito de `loss.backward()`. Recuerda el cambio de layout: Keras espera `(B, H, W, C)`, no `(B, C, H, W)`.

---

## 4. Seccion 3: JAX

JAX trabaja con **funciones puras** y transformaciones (`jit`, `grad`, `vmap`). La perdida NT-Xent es naturalmente pura —recibe `z1, z2, tau` y devuelve un escalar, sin estado— asi que encaja perfecto. Usamos `jax.nn.log_softmax` explicito para que la cross-entropy quede transparente.

### 4.1 El nucleo: NT-Xent puro con `jax.numpy`

```python
import jax
import jax.numpy as jnp

def nt_xent_loss(z1, z2, tau=0.5):
    """NT-Xent en JAX, funcion pura. z1, z2: (N, D) alineados por fila."""
    N = z1.shape[0]
    two_n = 2 * N

    # 1) Apilar y L2-normalizar
    z = jnp.concatenate([z1, z2], axis=0)                 # (2N, D)
    z = z / jnp.linalg.norm(z, axis=1, keepdims=True)     # filas con norma 1

    # 2) Similitud coseno / temperatura
    sim = (z @ z.T) / tau                                 # (2N, 2N)

    # 3) Enmascarar la diagonal (valor muy negativo -> peso 0 tras softmax)
    sim = sim - jnp.eye(two_n) * 1e9

    # 4) Etiquetas: positivo de i es i+N; de i+N es i
    targets = jnp.concatenate([jnp.arange(N, two_n),
                               jnp.arange(0, N)])          # (2N,)

    # 5) Cross-entropy a mano: log_softmax por fila + recoger el log-prob del positivo
    log_p = jax.nn.log_softmax(sim, axis=1)               # (2N, 2N)
    one_hot = jax.nn.one_hot(targets, two_n)              # (2N, 2N)
    loss = -jnp.mean(jnp.sum(one_hot * log_p, axis=1))    # promedio sobre las 2N anclas
    return loss
```

El detalle JAX: como no hay un `F.cross_entropy` con estado, construimos la cross-entropy **a mano**. `jax.nn.log_softmax(sim, axis=1)` normaliza cada fila (la diagonal en $-\infty$ aporta $-\infty$ al log-prob, es decir peso $0$); luego `one_hot * log_p` selecciona el log-prob del positivo de cada fila y `-mean(sum(...))` promedia. Es exactamente la Ecuacion (1).

Una forma idiomatica equivalente, recogiendo el log-prob del positivo sin construir el `one_hot`:

```python
    log_p = jax.nn.log_softmax(sim, axis=1)
    pos_log_p = log_p[jnp.arange(two_n), targets]   # log-prob del positivo de cada fila
    loss = -jnp.mean(pos_log_p)
```

Ambas dan el mismo resultado; la version con `one_hot` deja mas explicito que es una cross-entropy estandar.

### 4.2 Paso de entrenamiento con `jax.value_and_grad`

Aqui el modelo es una funcion pura cuyos parametros se pasan explicitamente. Usamos un encoder + cabeza minimos como diccionario de parametros para mantener el ejemplo autocontenido.

```python
import optax

key = jax.random.PRNGKey(42)

def init_mlp(key, sizes):
    """Inicializa una pila de capas densas como lista de (W, b)."""
    params = []
    for i in range(len(sizes) - 1):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (sizes[i], sizes[i + 1])) * 0.05
        b = jnp.zeros(sizes[i + 1])
        params.append((W, b))
    return params

def forward(params, x):
    """Encoder + cabeza colapsados en un MLP (toy). x: (B, in) -> z: (B, proj)."""
    *hidden, last = params
    for W, b in hidden:
        x = jax.nn.relu(x @ W + b)
    Wl, bl = last
    return x @ Wl + bl                       # proyeccion z (sin activacion)

# x_in: features sinteticos planos (B, 784) simulando imagenes 28x28 aplanadas
params = init_mlp(key, sizes=[784, 256, 128, 64])
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

def augment(key, x):
    return x + 0.1 * jax.random.normal(key, x.shape)   # jitter (toy)

@jax.jit
def train_step(params, opt_state, key, images):
    k1, k2 = jax.random.split(key)
    def loss_fn(p):
        z1 = forward(p, augment(k1, images))
        z2 = forward(p, augment(k2, images))
        return nt_xent_loss(z1, z2, 0.5)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for step in range(500):
    key, sk, dk = jax.random.split(key, 3)
    images = jax.random.normal(dk, (8, 784))   # batch de N=8 features
    params, opt_state, loss = train_step(params, opt_state, sk, images)
    if step % 100 == 0:
        print(f"step {step:4d}  loss={float(loss):.4f}")
```

Lo que hay que internalizar del patron JAX:

- **`forward(params, x)`** recibe los parametros explicitamente: el modelo es una funcion pura, no un objeto con estado.
- **`jax.value_and_grad(loss_fn)(params)`** devuelve `(loss, grads)` de una vez, diferenciando respecto al primer argumento.
- **`@jax.jit`** compila todo el paso a XLA: la primera llamada compila (lenta), las siguientes vuelan.
- Las **PRNGKeys** se dividen (`split`) explicitamente para que cada augmentacion use aleatoriedad fresca y reproducible. Dos keys distintas (`k1`, `k2`) garantizan que las dos vistas sean augmentaciones independientes.

---

## 5. Comparacion lado a lado de los tres frameworks

Las tres implementaciones son **isomorfas matematicamente**: misma matriz de similitud $2N \times 2N$, misma mascara diagonal, mismas etiquetas `[N..2N-1, 0..N-1]`, misma cross-entropy con temperatura. Cambia el idioma.

| Concepto | PyTorch | TensorFlow/Keras | JAX |
|---|---|---|---|
| Apilar vistas | `torch.cat([z1,z2], 0)` | `tf.concat([z1,z2], 0)` | `jnp.concatenate([z1,z2], 0)` |
| L2-normalizar | `F.normalize(z, dim=1)` | `tf.math.l2_normalize(z, 1)` | `z / jnp.linalg.norm(z,1,keepdims=True)` |
| Matriz similitud | `z @ z.t()` | `tf.matmul(z, z, transpose_b=True)` | `z @ z.T` |
| Mascara diagonal | `masked_fill_(eye, -inf)` | `+ tf.eye(2N) * -1e9` | `- jnp.eye(2N) * 1e9` |
| Etiquetas | `torch.arange` + `cat` | `tf.range` + `tf.concat` | `jnp.arange` + `concatenate` |
| Cross-entropy | `F.cross_entropy(sim, t)` | `sparse_categorical_crossentropy(t, sim, from_logits=True)` | `log_softmax` + `one_hot` manual |
| Diferenciacion | `loss.backward()` | `tf.GradientTape` | `jax.value_and_grad` |
| Compilacion JIT | `torch.compile` | `@tf.function` | `@jax.jit` |
| Layout de imagen | channels-first `(B,C,H,W)` | channels-last `(B,H,W,C)` | el que definas |

La leccion: NT-Xent es tan corta (apilar, normalizar, similitud, enmascarar, cross-entropy) que portarla es casi mecanico. Lo unico que hay que cuidar en los tres es **(a)** no aplicar softmax dos veces y **(b)** que las etiquetas apunten al positivo correcto.

---

## 6. Gotchas

Errores que rompen el codigo o lo degradan en silencio:

1. **Doble softmax.** `F.cross_entropy` y `sparse_categorical_crossentropy(from_logits=True)` ya aplican `log_softmax`. Pasales las similitudes **crudas** (`sim`), nunca un softmax pre-aplicado. En JAX, `log_softmax` se aplica una sola vez. Aplicar softmax dos veces aplana los gradientes y el modelo no aprende.
2. **Olvidar la mascara diagonal.** Si no enmascaras `sim[i,i]`, la similitud de una vista consigo misma (que vale $1/\tau$ tras normalizar) domina el softmax y el positivo real nunca gana. El loss baja a casi cero pero las representaciones son basura.
3. **Etiquetas mal alineadas.** `z1[i]` y `z2[i]` **deben** ser las dos augmentaciones de la **misma** imagen. Si tu `DataLoader` baraja `v1` y `v2` de forma independiente, los pares positivos quedan rotos y `targets` apunta a la imagen equivocada. Verifica que ambas vistas salen del mismo `images` con la misma indexacion.
4. **No L2-normalizar.** Si te saltas la normalizacion, `z @ z.t()` ya no es coseno sino producto punto crudo, y la magnitud de los embeddings (no solo su direccion) entra en la perdida. SimCLR asume vectores unitarios; sin normalizar, $\tau$ pierde su interpretacion.
5. **Temperatura mal elegida.** $\tau$ demasiado alta ($\to 1$) aplana las similitudes y casi no discrimina positivos de negativos; demasiado baja ($\to 0$) hace el softmax casi un argmax y los gradientes se concentran en el negativo mas duro, volviendo el entrenamiento inestable. El paper usa $\tau \approx 0.1$–$0.5$ (lo discutimos abajo).
6. **`-1e9` vs `-inf` (TF/JAX).** En TF y JAX usamos `-1e9` en vez de `-inf` para la mascara, porque `-inf` puede propagar `NaN` en algunas operaciones. `exp(-1e9)` es indistinguible de 0 en `float32`, asi que el efecto es el mismo y es numericamente mas seguro.

---

## 7. Por que batch grande, el rol de $\tau$, y la diferencia con MoCo

Tres ideas para cerrar, que conectan el codigo con el diseno de SimCLR y su gran rival.

### 7.1 Por que un batch grande ayuda

En NT-Xent, **los negativos de cada ancla son las otras $2N{-}2$ vistas del mismo batch**. No hay un banco de negativos separado: el batch *es* el conjunto de negativos. Por lo tanto, cuantas mas imagenes por batch, mas negativos por ancla, y mas dura (e informativa) es la tarea contrastiva —el modelo debe separar el positivo de muchos mas distractores simultaneamente. El paper lo muestra empiricamente: SimCLR mejora monotonicamente al subir el batch hasta **4096–8192**, donde cada ancla compite contra ~16000 negativos. Es el mismo principio que el "muestreo de negativos dificiles" del aprendizaje contrastivo en general, y el mismo fenomeno que en few-shot hace que entrenar con mas *way* mejore las representaciones.

El costo es brutal: batches de 4096+ requieren TPUs o muchas GPUs con `BatchNorm` global sincronizado (otro detalle de SimCLR, para que la estadistica de normalizacion no filtre informacion del positivo entre dispositivos). Esta dependencia del hardware es, justamente, lo que MoCo viene a resolver.

### 7.2 El rol de la temperatura $\tau$

$\tau$ controla **cuan concentrada** queda la distribucion del softmax sobre las similitudes:

- **$\tau$ grande** (cerca de 1): el softmax es suave, las similitudes se aplanan, el modelo apenas distingue el positivo de los negativos. Gradientes debiles, aprendizaje lento.
- **$\tau$ pequena** (cerca de 0): el softmax se vuelve casi un argmax. El gradiente se concentra en el **negativo mas duro** (el mas parecido al ancla). Esto puede acelerar pero tambien desestabilizar, y penaliza en exceso negativos que quizas son semanticamente validos.

El sweet spot empirico de SimCLR es $\tau \approx 0.1$–$0.5$ (usan $0.5$ con normalizacion). Conceptualmente, $\tau$ ajusta el **margen efectivo** entre positivos y negativos duros: una temperatura baja exige que el positivo sea *mucho* mas similar que cualquier negativo, lo que produce embeddings mas uniformemente distribuidos en la hiperesfera. Es un hiperparametro de primer orden, no un detalle: cambiarlo mueve la accuracy varios puntos.

### 7.3 La diferencia con MoCo: la cola

[MoCo (He et al. 2019)](/papers/moco-he-2019) ataca exactamente la limitacion de la seccion 7.1 —que el numero de negativos esta atado al *batch size*— con dos ideas:

- **Una cola FIFO de negativos.** En vez de usar solo el batch actual, MoCo mantiene una **cola** (un *dictionary*) de miles de embeddings de batches **anteriores** como negativos. Asi desacopla el numero de negativos del *batch size*: puedes tener un diccionario de 65536 negativos con un batch de 256. Donde SimCLR necesita un batch gigante para ver muchos negativos, MoCo los *recicla* de pasos previos.
- **Un *momentum encoder*.** Como los embeddings de la cola vienen de pasos viejos, sus pesos quedarian desactualizados. MoCo los genera con un encoder "lento", actualizado por **media movil exponencial** de los pesos del encoder principal ($\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q$, con $m \approx 0.999$). Esto mantiene la cola **consistente** pese a venir de momentos distintos del entrenamiento.

El contraste de diseno es nitido:

| | SimCLR | MoCo |
|---|---|---|
| Fuente de negativos | el propio batch ($2N{-}2$ vistas) | cola FIFO de batches previos (miles) |
| Como escala los negativos | subiendo el *batch size* (caro) | cola grande con batch pequeno (barato) |
| Encoder de los negativos | el mismo (un solo encoder) | *momentum encoder* (media movil) |
| Cuello de botella | memoria/hardware del batch | gestion de la cola y el momentum |

En la perdida, ambos usan la misma idea de InfoNCE/NT-Xent; lo que cambia es **de donde salen los negativos**. SimCLR los toma del presente (batch grande), MoCo del pasado (cola con momentum). Las versiones posteriores convergen: [MoCo v2](/papers/moco-v2-chen-2020) adopta la cabeza de proyeccion MLP y las augmentaciones fuertes de SimCLR sobre la maquinaria de cola de MoCo, obteniendo lo mejor de ambos con batches modestos.

---

**Ver tambien:** [Clase 28](/clases/clase-28) · [Fundamento de aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) · [Paper SimCLR (Chen et al. 2020)](/papers/simclr-chen-2020) · [Paper MoCo (He et al. 2019)](/papers/moco-he-2019) · [Teoria de la clase](../teoria).
