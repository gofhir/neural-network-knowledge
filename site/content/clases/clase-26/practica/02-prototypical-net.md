---
title: "02 - Prototypical Networks desde 0"
weight: 32
math: true
---

En el camino 01 construimos el motor que mueve todo el meta-aprendizaje few-shot: el **sampler N-way K-shot**, esa funcion que toma un dataset con muchas clases y arma *episodios* —elige $N$ clases al azar, toma $K$ ejemplos de soporte y $Q$ de consulta por clase— para que cada paso de entrenamiento parezca una mini-tarea de test. Ese sampler es la materia prima. Pero un sampler por si solo no clasifica nada: necesita un modelo que mire un episodio y decida.

En este capitulo construimos ese modelo: **Prototypical Networks** (Snell, Swersky y Zemel, NeurIPS 2017), el metodo *metric-based* mas didactico que existe. La idea es tan simple que cabe en una frase: aprende un espacio de embedding donde cada clase se resuma en su centroide, y clasifica por cercania a ese centroide. No hay capa de clasificacion con pesos por clase, no hay reentrenamiento sobre las clases nuevas, no hay LSTM meta-aprendiz. Solo un encoder, una media, y una distancia.

Lo vamos a implementar en **tres frameworks** —PyTorch, TensorFlow/Keras y JAX/Flax— porque el nucleo del algoritmo (computar prototipos, distancias euclidianas, cross-entropy episodica) es el mismo en los tres, pero cada idioma lo expresa distinto. Cuando termines vas a poder leer cualquier repo de few-shot sin importar el framework, y vas a entender por que el experimento estrella del paper —euclidiana vs coseno— es un cambio de una linea con una explicacion teorica profunda detras.

---

## 1. La idea en una ecuacion

Antes de tocar codigo, fijemos las dos ecuaciones que son **todo** el metodo.

Tenemos un encoder $f_\phi : \mathbb{R}^D \to \mathbb{R}^M$ (una CNN con parametros $\phi$) que mapea cada input a un vector de $M$ dimensiones. Dado el conjunto de soporte de la clase $k$, llamado $S_k$, el **prototipo** $c_k$ es la media de los embeddings de soporte de esa clase:

$$
c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i). \tag{1}
$$

Para clasificar una consulta $x$, aplicamos un *softmax* sobre las **distancias negativas** a los prototipos:

$$
p_\phi(y = k \mid x) = \frac{\exp\big(-d(f_\phi(x), c_k)\big)}{\sum_{k'} \exp\big(-d(f_\phi(x), c_{k'})\big)}. \tag{2}
$$

Y entrenamos minimizando la log-verosimilitud negativa de la clase correcta, $J(\phi) = -\log p_\phi(y = k \mid x)$, acumulada sobre todas las consultas del episodio.

Eso es todo. Es elegante por tres razones que vale la pena internalizar antes de programar:

1. **El clasificador no tiene parametros propios.** El "clasificador" de un episodio es el conjunto de prototipos $\{c_k\}$, y los prototipos son funcion cerrada del soporte (Ecuacion 1). Para clasificar una clase nueva no entrenamos nada: calculamos una media. Toda la capacidad de aprendizaje vive en $f_\phi$, que se entrena sobre las clases base abundantes durante el meta-entrenamiento. El few-shot deja de ser un problema de optimizacion con datos escasos y pasa a ser **recuperacion geometrica** en un espacio bien construido.

2. **El promedio reduce ruido.** Con $|S_k|$ ejemplos, el centroide estima la media de la clase con varianza que decae como $1/|S_k|$. Por eso 5-shot siempre supera a 1-shot: el prototipo de 5-shot es una estimacion mucho mas estable del "centro" de la clase.

3. **La distancia tiene justificacion teorica.** Usar distancia **euclidiana cuadrada** (no coseno) no es un truco: bajo divergencias de Bregman, el prototipo-media es el representante optimo del cluster, y el clasificador resulta equivalente a estimacion de densidad por mezcla de gaussianas esfericas. Lo desarrollamos en la seccion 6.

El doble rol de $f_\phi$ es sutil pero clave: la **misma** red embebe los soportes (para formar $c_k$) y las consultas. Cuando minimizamos $\|f_\phi(x) - c_k\|^2$, el gradiente empuja simultaneamente la consulta hacia su centroide y el centroide (via los soportes) hacia la consulta. No hay `stop_gradient` sobre los prototipos —y eso es deliberado, lo veremos en los gotchas.

### 1.1 Hiperparametros del episodio

Reusamos la nomenclatura del camino 01. Un episodio se define por:

| Simbolo | Nombre | Valor tipico | Que controla |
|---|---|---:|---|
| $N$ ($N_C$) | way | 5, 20, 30 | clases por episodio |
| $K$ ($N_S$) | shot | 1 o 5 | soportes por clase |
| $Q$ ($N_Q$) | query | 15 | consultas por clase |

El sampler del camino 01 nos entrega, por episodio, dos tensores: `support` de shape `(N*K, C, H, W)` con sus etiquetas `(N*K,)`, y `query` de shape `(N*Q, C, H, W)` con sus etiquetas `(N*Q,)`. Las etiquetas estan **remapeadas** a $\{0, \dots, N-1\}$ dentro del episodio (no son los IDs globales de clase). Ese detalle del remapeo es importante: el softmax de la Ecuacion (2) normaliza sobre las $N$ clases del episodio, no sobre todas las clases del dataset.

### 1.2 Por que el entrenamiento es episodico

Vale la pena detenerse en por que entrenamos por episodios y no con batches ordinarios. La regla de oro del few-shot, heredada de Matching Networks, es **"las condiciones de test y de entrenamiento deben coincidir"**. Si en test vamos a resolver tareas 5-way 1-shot —clasificar entre 5 clases nuevas con un ejemplo cada una— entonces cada paso de entrenamiento debe *parecerse* a esa tarea: tomar 5 clases (de las clases base, abundantes), un ejemplo de soporte por clase, unos cuantos de consulta, y aprender a clasificar las consultas contra los prototipos de los soportes.

La consecuencia es que la red **nunca ve una capa de clasificacion fija**. En un clasificador estandar de ImageNet hay 1000 neuronas de salida, una por clase, con pesos que se memorizan. Aqui no: la "capa de clasificacion" se reconstruye en cada episodio a partir de los datos (los prototipos), y las clases cambian de un episodio al siguiente. Esa aleatoriedad —clases distintas combinadas cada vez— es lo que fuerza la generalizacion. El espacio de embedding debe servir para particiones de clase que **nunca se vieron juntas** durante el entrenamiento. Es la diferencia entre "aprender a clasificar estas clases" y "aprender a construir un espacio donde cualquier conjunto de clases sea facil de separar por centroides".

Por eso ProtoNets es un metodo **no parametrico** en el sentido few-shot: el clasificador para una tarea nueva no tiene parametros entrenados sobre esa tarea; se construye sobre la marcha. Su "memoria" son los ejemplos (resumidos en centroides), igual que k-NN o los kernel methods, no un vector de pesos por clase. Esto lo opone a los metodos *optimization-based* como MAML (camino 03), que si adaptan parametros por tarea.

---

## 2. El encoder conv-4

Snell et al. usan un encoder que se volvio el *backbone canonico* del few-shot, conocido como **conv-4**: cuatro bloques identicos, cada uno con

$$
\text{conv } 3\times3 \text{ (64 filtros)} \;\to\; \text{BatchNorm} \;\to\; \text{ReLU} \;\to\; \text{maxpool } 2\times2.
$$

Cada bloque divide a la mitad el lado espacial. La tabla de shapes para una imagen Omniglot $28 \times 28$ en escala de grises y para una miniImageNet $84 \times 84$ a color:

| Etapa | Omniglot $(1, 28, 28)$ | miniImageNet $(3, 84, 84)$ |
|---|---|---|
| Entrada | $(1, 28, 28)$ | $(3, 84, 84)$ |
| Bloque 1 | $(64, 14, 14)$ | $(64, 42, 42)$ |
| Bloque 2 | $(64, 7, 7)$ | $(64, 21, 21)$ |
| Bloque 3 | $(64, 3, 3)$ | $(64, 10, 10)$ |
| Bloque 4 | $(64, 1, 1)$ | $(64, 5, 5)$ |
| Flatten | $\mathbf{64}$ | $\mathbf{1600}$ |

Ese flatten final es el embedding $f_\phi(x)$: dimension $M = 64$ para Omniglot, $M = 1600$ para miniImageNet. Es el mismo encoder para soporte y consulta (no se desacoplan, a diferencia de Matching Networks). El batch norm es la unica regularizacion; no hay dropout ni weight decay (salvo en la variante zero-shot).

Para que el codigo de las tres secciones sea directamente comparable, fijamos la configuracion en una tabla:

| Hiperparametro | Valor |
|---|---:|
| Bloques conv | 4 |
| Filtros por bloque | 64 |
| Kernel | $3 \times 3$, padding 1 |
| Pool | maxpool $2 \times 2$ |
| Canales de entrada | 1 (Omniglot) / 3 (miniImageNet) |
| $M$ (dim embedding) | 64 / 1600 |
| Distancia | euclidiana cuadrada |
| Optimizador | Adam, lr $10^{-3}$, mitad cada 2000 episodios |

---

## 3. Seccion 1: PyTorch

PyTorch es el framework dominante en research few-shot. Casi todos los repos de ProtoNets, MAML y compania estan en PyTorch. Su filosofia *define-by-run* hace que el episodio se sienta natural: cada `forward` construye el grafo sobre la marcha.

### 3.1 Imports y configuracion

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Configuracion del episodio (la entrega el sampler del camino 01)
N_WAY = 5
K_SHOT = 5
Q_QUERY = 15
IN_CHANNELS = 1   # Omniglot escala de grises; 3 para miniImageNet
EMBED_DIM = 64    # 64 para Omniglot 28x28; 1600 para miniImageNet 84x84
```

### 3.2 El encoder conv-4

```python
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hidden),
            conv_block(hidden, hidden),
            conv_block(hidden, hidden),
            conv_block(hidden, hidden),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        z = self.encoder(x)          # (B, 64, H', W')
        return z.flatten(start_dim=1)  # (B, M)
```

El `flatten(start_dim=1)` colapsa los ejes espaciales y de canal en un solo vector por imagen. Para Omniglot $28 \times 28$ el resultado es $(B, 64)$; para miniImageNet $84 \times 84$ es $(B, 1600)$. No hay capa lineal final: el embedding **es** el flatten.

### 3.3 El nucleo: prototipos, distancias y loss

Aqui esta el corazon del metodo. Recibimos del sampler los embeddings ya calculados (o las imagenes y las pasamos por el encoder) y producimos los logits.

```python
def euclidean_dist(x, y):
    """Distancia euclidiana cuadrada entre dos conjuntos de vectores.
    x: (n, M) consultas;  y: (m, M) prototipos.
    Devuelve: (n, m) con d[i, j] = ||x_i - y_j||^2.
    """
    n, m = x.size(0), y.size(0)
    x = x.unsqueeze(1).expand(n, m, -1)   # (n, m, M)
    y = y.unsqueeze(0).expand(n, m, -1)   # (n, m, M)
    return ((x - y) ** 2).sum(dim=2)      # (n, m)


def proto_loss(encoder, support, query, n_way, k_shot, q_query):
    """
    support: (n_way * k_shot, C, H, W), ordenado por clase
    query:   (n_way * q_query, C, H, W), ordenado por clase
    Etiquetas implicitas: las primeras k_shot filas son clase 0, etc.
    """
    # 1) Embeber soporte y consulta con el MISMO encoder
    z_support = encoder(support)   # (n_way * k_shot, M)
    z_query = encoder(query)       # (n_way * q_query, M)

    # 2) Prototipos = media por clase (Ecuacion 1)
    z_support = z_support.view(n_way, k_shot, -1)  # (n_way, k_shot, M)
    prototypes = z_support.mean(dim=1)             # (n_way, M)

    # 3) Distancias euclidianas consulta -> prototipos
    dists = euclidean_dist(z_query, prototypes)    # (n_way * q_query, n_way)

    # 4) Logits = -distancias; softmax + cross-entropy (Ecuacion 2)
    log_p = F.log_softmax(-dists, dim=1)           # (n_way * q_query, n_way)

    # Etiquetas verdaderas: 0..n_way-1, cada una repetida q_query veces
    target = torch.arange(n_way).repeat_interleave(q_query)
    loss = F.nll_loss(log_p, target)

    acc = (log_p.argmax(dim=1) == target).float().mean()
    return loss, acc
```

Cuatro pasos, cuatro lineas de matematica:

- **`view(n_way, k_shot, -1).mean(dim=1)`** es la Ecuacion (1): reordena el soporte en una grilla `(clase, shot, dim)` y promedia sobre el eje del shot. Esto asume que el sampler entrega el soporte **ordenado por clase** —las primeras `k_shot` filas son la clase 0, las siguientes la clase 1, etc.
- **`euclidean_dist`** construye la matriz $(n_q, n_w)$ por broadcasting: expandimos consultas y prototipos a un tensor comun $(n_q, n_w, M)$ y reducimos sobre la dimension del embedding. Es el equivalente manual de `torch.cdist(z_query, prototypes) ** 2`.
- **`log_softmax(-dists)`** es la Ecuacion (2) en forma logaritmica: distancia negativa, normalizada sobre clases.
- **`nll_loss`** completa la cross-entropy. El `target` se construye con `arange(n_way).repeat_interleave(q_query)` porque las consultas tambien vienen ordenadas por clase.

Una alternativa de una sola linea para las distancias: `torch.cdist(z_query, prototypes).pow(2)`. `cdist` calcula la distancia euclidiana (no cuadrada), asi que hay que elevar al cuadrado. La version manual con broadcasting es mas explicita y deja claro de donde sale el shape; en produccion conviene `cdist` por eficiencia.

### 3.4 El loop de meta-entrenamiento episodico

Este es el loop completo. Asumimos un `episode_sampler` del camino 01 que produce `(support, query)` por episodio.

```python
encoder = ConvEncoder(in_channels=IN_CHANNELS)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

encoder.train()
for episode in range(20000):
    # El sampler del camino 01 entrega un episodio N-way K-shot
    support, query = episode_sampler.sample(N_WAY, K_SHOT, Q_QUERY)

    loss, acc = proto_loss(encoder, support, query,
                           N_WAY, K_SHOT, Q_QUERY)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if episode % 1000 == 0:
        print(f"ep {episode:5d}  loss={loss.item():.4f}  acc={acc.item():.3f}")
```

Notese que **no hay fase de meta-optimizacion separada** ni particion de parametros: el gradiente fluye end-to-end a traves del encoder, tanto por la rama de la consulta como por la rama del soporte (que forma los prototipos). Es SGD ordinario. El `StepLR` reduce el lr a la mitad cada 2000 episodios, como en el paper.

Un detalle pedagogico del paper: aunque en test hagamos 5-way, conviene **entrenar con mas way** (20 o 30). La mayor dificultad fuerza decisiones mas finas en el espacio de embedding. Solo hay que cambiar `N_WAY` en la llamada al sampler; el resto del codigo es identico.

### 3.5 El truco del "way": entrenar mas dificil de lo que se testea

Este es uno de los hallazgos mas contraintuitivos y mas citados del paper, y se implementa cambiando un solo numero. Si en test vas a hacer 5-way, **no** entrenes con episodios 5-way: entrena con 20-way o 30-way. Los datos del paper en miniImageNet (5-way 1-shot **en test**, variando el way **en entrenamiento**):

| Way de entrenamiento | Accuracy test 5-way 1-shot |
|---:|---:|
| 5 | 46.14% |
| 10 | 48.27% |
| 15 | 48.60% |
| 20 | 48.57% |
| 30 | **49.42%** |

Son ~3.3 puntos gratis, solo cambiando la composicion de los episodios. La intuicion: el softmax normaliza sobre las clases del episodio. Con 5-way, el modelo solo necesita separar la clase correcta de otras 4; con 30-way, debe separarla de 29 distractores **simultaneamente**. El termino log-sum-exp de la perdida penaliza la cercania a *cualquier* prototipo ajeno, asi que mas clases por episodio significan mas restricciones de margen por paso de gradiente. El espacio resultante tiene que ser globalmente mas fino. Es el mismo principio que el muestreo de negativos dificiles en aprendizaje contrastivo: mas negativos por consulta endurece la tarea y mejora la representacion.

Hay una asimetria importante con el **shot**: a diferencia del way, lo mejor es **igualar el shot de entrenamiento y de test**. Si vas a testear 5-shot, entrena 5-shot (entrenar 1-shot para testear 5-shot rinde peor). Y hay un punto de retornos decrecientes: para 5-shot, subir el way mas alla de ~20 empieza a degradar, porque la tarea se vuelve tan dificil que la estimacion del prototipo ya no alcanza. Resumen operativo: **sube el way, fija el shot**.

---

## 4. Seccion 2: TensorFlow / Keras

TensorFlow 2.x con Keras ofrece la misma flexibilidad pero con otro estilo: capas via subclassing de `keras.Model`, trazado con `@tf.function`, y `tf.GradientTape` para los gradientes. La inferencia es first-class para deployment.

### 4.1 Imports y encoder

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

N_WAY, K_SHOT, Q_QUERY = 5, 5, 15
IN_CHANNELS, EMBED_DIM = 1, 64


def conv_block(out_ch):
    return keras.Sequential([
        layers.Conv2D(out_ch, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D(2),
    ])


class ConvEncoder(keras.Model):
    def __init__(self, hidden=64, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [conv_block(hidden) for _ in range(4)]
        self.flatten = layers.Flatten()

    def call(self, x, training=False):
        # x: (B, H, W, C)  -- OJO: Keras usa channels-last
        for block in self.blocks:
            x = block(x, training=training)
        return self.flatten(x)  # (B, M)
```

La diferencia de convencion mas importante: **Keras usa channels-last** `(B, H, W, C)`, mientras PyTorch usa channels-first `(B, C, H, W)`. El sampler del camino 01 debe entregar los tensores en el layout correcto para cada framework, o transponerlos. El `padding="same"` de Keras equivale al `padding=1` con kernel 3 de PyTorch.

### 4.2 El nucleo: prototipos, distancias y loss

```python
def euclidean_dist(x, y):
    """x: (n, M) consultas; y: (m, M) prototipos -> (n, m) cuadrada."""
    x = tf.expand_dims(x, 1)   # (n, 1, M)
    y = tf.expand_dims(y, 0)   # (1, m, M)
    return tf.reduce_sum(tf.square(x - y), axis=2)  # (n, m)


def proto_loss(encoder, support, query, n_way, k_shot, q_query, training=True):
    z_support = encoder(support, training=training)  # (n_way*k_shot, M)
    z_query = encoder(query, training=training)      # (n_way*q_query, M)

    # Prototipos = media por clase (Ecuacion 1)
    z_support = tf.reshape(z_support, (n_way, k_shot, -1))
    prototypes = tf.reduce_mean(z_support, axis=1)   # (n_way, M)

    # Distancias y log-softmax sobre -dist (Ecuacion 2)
    dists = euclidean_dist(z_query, prototypes)      # (n_way*q_query, n_way)
    log_p = tf.nn.log_softmax(-dists, axis=1)

    target = tf.repeat(tf.range(n_way), q_query)     # (n_way*q_query,)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, log_p, from_logits=False
        )
    )
    # OJO: log_p ya son log-probabilidades, no logits crudos.
    acc = tf.reduce_mean(
        tf.cast(tf.argmax(log_p, axis=1, output_type=tf.int32) == target,
                tf.float32)
    )
    return loss, acc
```

Mapeo de operaciones respecto a PyTorch:

- `tensor.view(...)` -> `tf.reshape(...)`
- `.mean(dim=1)` -> `tf.reduce_mean(..., axis=1)`
- `unsqueeze` -> `tf.expand_dims`
- `repeat_interleave` -> `tf.repeat`

Un cuidado con la cross-entropy: como ya aplicamos `log_softmax`, le pasamos las log-probabilidades a `sparse_categorical_crossentropy` con `from_logits=False`. La funcion espera probabilidades, no logits, asi que internamente hace `-log(p)`; al pasarle `exp(log_p)`... no, cuidado: lo correcto y mas limpio es pasar los **logits crudos** `-dists` con `from_logits=True` y dejar que Keras haga el log-softmax internamente. La version equivalente sin doble softmax:

```python
    dists = euclidean_dist(z_query, prototypes)
    logits = -dists
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, logits, from_logits=True   # Keras aplica log_softmax
        )
    )
```

Esta segunda forma es la recomendada: una sola normalizacion, sin riesgo de aplicar softmax dos veces. Es el gotcha mas comun al portar el codigo entre frameworks.

### 4.3 El paso de update con `tf.GradientTape`

```python
encoder = ConvEncoder(hidden=64)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(support, query):
    with tf.GradientTape() as tape:
        loss, acc = proto_loss(encoder, support, query,
                               N_WAY, K_SHOT, Q_QUERY, training=True)
    grads = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
    return loss, acc

for episode in range(20000):
    support, query = episode_sampler.sample(N_WAY, K_SHOT, Q_QUERY)
    loss, acc = train_step(support, query)
    if episode % 1000 == 0:
        print(f"ep {episode:5d}  loss={float(loss):.4f}  acc={float(acc):.3f}")
```

El `@tf.function` traza el grafo la primera vez y reutiliza la traza —equivalente a `jax.jit` o `torch.compile`. `tf.GradientTape` es el equivalente explicito de `loss.backward()`: solo se trackean las operaciones dentro del `with`. El resto del loop es identico a PyTorch en estructura.

---

## 5. Seccion 3: JAX + Flax

JAX adopta **funciones puras + transformaciones**. No hay estado mutable: el modelo es una funcion que recibe sus parametros como argumento. A cambio obtienes `jit` (compilacion XLA), `grad` (autodiff), `vmap` (vectorizacion automatica) y `pmap` (paralelismo). **Flax** pone una API tipo Keras encima.

### 5.1 Imports y encoder en Flax

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

N_WAY, K_SHOT, Q_QUERY = 5, 5, 15
IN_CHANNELS, EMBED_DIM = 1, 64


class ConvEncoder(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, x, train=True):
        # x: (B, H, W, C)  -- Flax tambien usa channels-last
        for _ in range(4):
            x = nn.Conv(self.hidden, (3, 3), padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
        return x.reshape((x.shape[0], -1))  # flatten -> (B, M)
```

Notas Flax:

- **`@nn.compact`** permite definir sub-modulos dentro de `__call__`.
- **`use_running_average=not train`** es el control del BatchNorm: durante entrenamiento usa estadisticas del batch; en eval usa las acumuladas. En Flax el batch norm mantiene un estado mutable separado de los params (la coleccion `batch_stats`), que hay que propagar explicitamente —lo manejamos abajo.
- Como TF, Flax usa **channels-last** `(B, H, W, C)`.

### 5.2 El nucleo: prototipos, distancias y loss (con `vmap`)

```python
def euclidean_dist(x, y):
    """x: (n, M); y: (m, M) -> (n, m) cuadrada, via vmap."""
    # Para un solo vector de consulta xi contra todos los prototipos:
    dist_one = lambda xi: jnp.sum((y - xi) ** 2, axis=1)  # (m,)
    return jax.vmap(dist_one)(x)                           # (n, m)


def proto_loss(params, batch_stats, apply_fn, support, query,
               n_way, k_shot, q_query, train=True):
    variables = {"params": params, "batch_stats": batch_stats}
    if train:
        z_support, mut = apply_fn(variables, support, train=True,
                                  mutable=["batch_stats"])
        z_query, mut2 = apply_fn({"params": params,
                                  "batch_stats": mut["batch_stats"]},
                                 query, train=True, mutable=["batch_stats"])
        new_batch_stats = mut2["batch_stats"]
    else:
        z_support = apply_fn(variables, support, train=False)
        z_query = apply_fn(variables, query, train=False)
        new_batch_stats = batch_stats

    # Prototipos = media por clase (Ecuacion 1)
    z_support = z_support.reshape(n_way, k_shot, -1)
    prototypes = jnp.mean(z_support, axis=1)          # (n_way, M)

    # Distancias y cross-entropy (Ecuacion 2)
    dists = euclidean_dist(z_query, prototypes)       # (n_way*q_query, n_way)
    log_p = jax.nn.log_softmax(-dists, axis=1)

    target = jnp.repeat(jnp.arange(n_way), q_query)
    one_hot = jax.nn.one_hot(target, n_way)
    loss = -jnp.mean(jnp.sum(one_hot * log_p, axis=1))

    acc = jnp.mean(jnp.argmax(log_p, axis=1) == target)
    return loss, (acc, new_batch_stats)
```

El detalle JAX: la cross-entropy se construye **a mano** con `one_hot` y `log_softmax`, porque no hay un `nll_loss` con estado. `optax.softmax_cross_entropy(logits=-dists, labels=one_hot)` haria lo mismo en una linea —es la version idiomatica:

```python
    loss = jnp.mean(optax.softmax_cross_entropy(logits=-dists,
                                                labels=one_hot))
```

El uso de `jax.vmap` en `euclidean_dist` es ilustrativo: escribimos la distancia para **una** consulta contra todos los prototipos, y `vmap` la vectoriza sobre el batch de consultas automaticamente, sin escribir el broadcasting a mano. Es el idioma JAX por excelencia. (Tambien podriamos haber hecho el broadcasting explicito como en PyTorch/TF; `vmap` es mas legible.)

### 5.3 El paso de update con `optax`

```python
encoder = ConvEncoder(hidden=64)
key = jax.random.PRNGKey(42)

# Inicializacion explicita: params + batch_stats
dummy = jnp.zeros((1, 28, 28, IN_CHANNELS))
variables = encoder.init(key, dummy, train=True)
params, batch_stats = variables["params"], variables["batch_stats"]

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, batch_stats, opt_state, support, query):
    def loss_wrapper(p):
        loss, (acc, new_bs) = proto_loss(
            p, batch_stats, encoder.apply, support, query,
            N_WAY, K_SHOT, Q_QUERY, train=True)
        return loss, (acc, new_bs)

    (loss, (acc, new_bs)), grads = jax.value_and_grad(
        loss_wrapper, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, new_bs, opt_state, loss, acc


for episode in range(20000):
    support, query = episode_sampler.sample(N_WAY, K_SHOT, Q_QUERY)
    params, batch_stats, opt_state, loss, acc = train_step(
        params, batch_stats, opt_state, support, query)
    if episode % 1000 == 0:
        print(f"ep {episode:5d}  loss={float(loss):.4f}  acc={float(acc):.3f}")
```

Lo que hay que internalizar del patron JAX:

- **`encoder.apply(variables, x)`** es el `forward`, pero recibe los params explicitamente. El modelo es una funcion pura.
- **`jax.value_and_grad(..., has_aux=True)`** devuelve `(loss, aux), grads` de una vez. El `has_aux=True` permite arrastrar el `acc` y el nuevo `batch_stats` sin que entren en la diferenciacion.
- **`batch_stats`** (estado del BatchNorm) se pasa y se devuelve explicitamente, igual que `params` y `opt_state`. Nada es mutable.
- **`@jax.jit`** compila todo el paso a XLA. La primera llamada compila (lenta); las siguientes vuelan.

---

## 6. El experimento clave: euclidiana vs coseno

Aqui esta el resultado mas citado del paper, y el cambio de **una linea** que lo demuestra. Reemplazamos la distancia euclidiana por similitud coseno (negativa, para que "mas similar" = "logit mas alto"):

```python
# PyTorch
def cosine_logits(z_query, prototypes):
    zq = F.normalize(z_query, dim=1)      # (n_q, M) unitarios
    cp = F.normalize(prototypes, dim=1)   # (n_w, M) unitarios
    return zq @ cp.t()                    # (n_q, n_w) cosenos en [-1, 1]
# logits = cosine_logits(...)   en vez de  logits = -euclidean_dist(...)
```

En miniImageNet 5-way 5-shot, el paper reporta una brecha enorme: coseno **51.48%** vs euclidiana **68.20%** —unos **17 puntos**. ¿Por que la euclidiana gana de forma tan contundente?

La respuesta es teorica y se llama **divergencia de Bregman**. Una divergencia de Bregman $d_\varphi(z, z') = \varphi(z) - \varphi(z') - (z - z')^\top \nabla\varphi(z')$ tiene una propiedad notable (Banerjee et al. 2005): para *cualquier* Bregman, el representante que minimiza la distancia total a un cluster de puntos **es la media** de esos puntos. La distancia euclidiana cuadrada es de Bregman (con $\varphi(z) = \|z\|^2$). El coseno **no lo es**.

Esto produce una **incoherencia interna** cuando usas coseno: construyes el prototipo como media aritmetica (Ecuacion 1), pero mides con coseno, que solo mira la *direccion* y normaliza a la hiperesfera unitaria. La media de varios vectores unitarios tiene norma menor que 1 y no es el minimizador de la distancia coseno total (ese seria la direccion media renormalizada, otra cantidad). Construyes con un criterio y mides con otro. Con euclidiana cuadrada la media **es** el minimizador, y todo encaja: clustering, distancia y densidad gaussiana esferica.

Hay mas: bajo euclidiana cuadrada, el modelo es exactamente equivalente a **estimacion de densidad por mezcla de gaussianas esfericas** equiponderadas, una por clase. Expandiendo $-\|f_\phi(x) - c_k\|^2 = -\|f_\phi(x)\|^2 + 2 c_k^\top f_\phi(x) - \|c_k\|^2$, el primer termino es constante en $k$ y se cancela en el softmax, dejando un **clasificador lineal** con $w_k = 2c_k$ y $b_k = -\|c_k\|^2$. Toda la no linealidad necesaria vive en $f_\phi$. El coseno rompe esa interpretacion probabilistica y por eso funciona peor —especialmente en ProtoNets, donde el prototipo *es* una media, mas que en Matching Networks, que compara contra puntos individuales.

Moraleja practica: usa **euclidiana cuadrada** y, si normalizas, hazlo con cuidado. Algunos repos escalan los logits por un factor de temperatura $-d/\tau$; es legitimo, pero no cambia la conclusion Bregman.

### 6.1 El clasificador lineal escondido, con derivacion

Vale la pena hacer explicita la derivacion del clasificador lineal, porque ilumina por que la euclidiana es "natural" y de paso explica una optimizacion de codigo. Partimos del exponente de la Ecuacion (2) con $d(z, z') = \|z - z'\|^2$:

$$
-\|f_\phi(x) - c_k\|^2 = -f_\phi(x)^\top f_\phi(x) + 2\,c_k^\top f_\phi(x) - c_k^\top c_k. \tag{7}
$$

El primer termino, $-f_\phi(x)^\top f_\phi(x)$, **no depende de la clase $k$**: es identico en el numerador y en cada termino del denominador del softmax, asi que se cancela. Quedan los dos terminos que si dependen de $k$:

$$
2\,c_k^\top f_\phi(x) - c_k^\top c_k = w_k^\top f_\phi(x) + b_k, \quad w_k = 2 c_k,\ b_k = -\|c_k\|^2. \tag{8}
$$

Es decir: ProtoNets con euclidiana es, en el espacio de embedding, **un clasificador lineal** con pesos $w_k = 2 c_k$ y sesgo $b_k = -\|c_k\|^2$. ¿No es esto una limitacion ("solo un clasificador lineal")? No, y por la misma razon que AlexNet o un ResNet no son "solo" su capa lineal final: **toda la no linealidad vive en $f_\phi$**. La diferencia con un clasificador estandar es que los pesos $w_k$ no se aprenden por gradiente: se calculan como $2 c_k$, y $c_k$ es la media del soporte. La capa de clasificacion es funcion cerrada de los datos.

Esta forma sugiere una optimizacion: en vez de computar la matriz de distancias completa $(n_q, n_w, M)$, podemos calcular directamente los logits como `2 * z_query @ prototypes.T - (prototypes ** 2).sum(1)`. Es matematicamente identico (salvo el termino constante que se cancela) y evita el tensor 3D del broadcasting. En PyTorch:

```python
# Equivalente a -euclidean_dist, sin el tensor 3D intermedio
def proto_logits_linear(z_query, prototypes):
    # logits[i, k] = 2 c_k . z_i - ||c_k||^2  (omitimos -||z_i||^2, constante en k)
    return 2 * z_query @ prototypes.t() - (prototypes ** 2).sum(dim=1)
```

Ambas formas dan el mismo softmax. La version lineal es la que usan internamente algunos repos optimizados.

---

### 6.2 Bonus: la variante zero-shot en una idea

El mismo modelo se extiende a **zero-shot** (cero ejemplos de la clase nueva) con un cambio minimo y elegante. En zero-shot no hay conjunto de soporte; cada clase trae un **vector de meta-datos** $v_k$ (atributos que la describen: color, forma, en el caso de aves; o codificaciones y descriptores fenotipicos en un escenario clinico FHIR). En vez de promediar soportes, definimos el prototipo como un embedding de esos meta-datos via una **segunda funcion de embedding** $g_\vartheta$:

$$
c_k = g_\vartheta(v_k).
$$

El resto del modelo es identico: softmax sobre distancias del embedding de la imagen $f_\phi(x)$ a los prototipos $c_k$. Como imagen y meta-dato vienen de dominios distintos, hay dos encoders separados ($f_\phi$ para imagenes, $g_\vartheta$ para atributos) que mapean a un **espacio compartido**. Empiricamente ayuda fijar el embedding del prototipo $g$ a **norma unitaria** (no asi el de la consulta), para alinear las escalas entre dominios. En CUB-200 zero-shot 50-way, ProtoNets logra 54.6%, estado del arte por margen. En codigo, el unico cambio es reemplazar el `mean(dim=1)` sobre soportes por una pasada `g_theta(metadata)`.

---

## 7. Evaluacion: accuracy con intervalo de confianza

En test muestreamos muchos episodios (600 para miniImageNet, 1000 para Omniglot, segun el paper), medimos accuracy por episodio y reportamos la media con un **intervalo de confianza al 95%**. El IC es esencial: un solo numero de accuracy sin IC no es comparable.

```python
import numpy as np

@torch.no_grad()
def evaluate(encoder, sampler, n_way, k_shot, q_query, n_episodes=600):
    encoder.eval()
    accs = []
    for _ in range(n_episodes):
        support, query = sampler.sample(n_way, k_shot, q_query)
        _, acc = proto_loss(encoder, support, query, n_way, k_shot, q_query)
        accs.append(acc.item())
    accs = np.array(accs)
    mean = accs.mean()
    # IC 95%: 1.96 * desviacion estandar / sqrt(n)
    ci95 = 1.96 * accs.std(ddof=1) / np.sqrt(n_episodes)
    return mean, ci95


# 5-way 1-shot y 5-way 5-shot
m1, c1 = evaluate(encoder, test_sampler, n_way=5, k_shot=1, q_query=15)
m5, c5 = evaluate(encoder, test_sampler, n_way=5, k_shot=5, q_query=15)
print(f"5-way 1-shot: {100*m1:.2f} +/- {100*c1:.2f}%")
print(f"5-way 5-shot: {100*m5:.2f} +/- {100*c5:.2f}%")
```

Numeros de referencia que deberias reproducir con este codigo (miniImageNet, encoder conv-4, euclidiana):

| Configuracion | ProtoNets (paper) |
|---|---:|
| miniImageNet 5-way 1-shot | $49.42 \pm 0.78\%$ |
| miniImageNet 5-way 5-shot | $68.20 \pm 0.66\%$ |
| Omniglot 20-way 1-shot | $96.0\%$ |
| Omniglot 20-way 5-shot | $98.9\%$ |

El IC se calcula como $1.96 \cdot \sigma / \sqrt{n}$ donde $\sigma$ es la desviacion estandar muestral de las accuracies por episodio. Para evaluar, recuerda igualar el **shot** de test al de entrenamiento (en el paper, 5-shot test se entrena con 5-shot); el **way**, en cambio, puede ser mayor en entrenamiento.

---

## 8. Comparacion lado a lado de los tres frameworks

Las tres implementaciones son **isomorfas matematicamente**: mismo encoder conv-4, misma media para los prototipos, misma euclidiana cuadrada, misma cross-entropy episodica. Lo que cambia es el idioma. Esta tabla resume el diccionario de traduccion para el nucleo de ProtoNets:

| Concepto | PyTorch | TensorFlow/Keras | JAX + Flax |
|---|---|---|---|
| Definicion de modulo | `class M(nn.Module)` + `forward` | `class M(keras.Model)` + `call` | `class M(nn.Module)` + `__call__` con `@nn.compact` |
| Estado interno | mutable en `self.W` | mutable en `self.dense` | inmutable, params + `batch_stats` externos |
| Forward | `encoder(x)` | `encoder(x, training=True)` | `encoder.apply(variables, x, train=True)` |
| Media por clase | `z.view(N,K,-1).mean(1)` | `tf.reduce_mean(reshape, axis=1)` | `jnp.mean(reshape, axis=1)` |
| Matriz de distancias | broadcasting o `torch.cdist` | `expand_dims` + `reduce_sum` | `jax.vmap` o broadcasting |
| Cross-entropy | `F.nll_loss(log_softmax, t)` | `sparse_categorical_crossentropy(from_logits=True)` | `optax.softmax_cross_entropy` o manual |
| Diferenciacion | `loss.backward()` | `tf.GradientTape` | `jax.value_and_grad` |
| Modo train/eval | `model.train()`/`.eval()` | argumento `training=` | argumento `train=` + `mutable` |
| Estado BatchNorm | implicito en buffers | implicito | **explicito** en `batch_stats` |
| Compilacion JIT | `torch.compile` | `@tf.function` | `@jax.jit` |
| Optimizador | `torch.optim.Adam` | `keras.optimizers.Adam` | `optax.adam` |
| Layout de imagen | channels-first `(B,C,H,W)` | channels-last `(B,H,W,C)` | channels-last `(B,H,W,C)` |

La leccion practica: el algoritmo ProtoNets es tan corto (cuatro operaciones: embeber, promediar, distancias, cross-entropy) que portarlo entre frameworks es casi mecanico. Lo que realmente cambia es el *andamiaje* —como se manejan estado, gradientes y compilacion— no la matematica.

### 8.1 Cual usar y cuando

- **PyTorch**: research y prototipado. La mayoria de los repos de few-shot (incluido el codigo original de ProtoNets) estan aqui. Si exploras una idea nueva y quieres ver gradientes y prints, es el camino.
- **TensorFlow/Keras**: produccion. TF Serving, TFLite para edge, TF.js. Si el modelo va a vivir fuera de Python, es el camino.
- **JAX + Flax**: escala masiva en TPUs y vectorizacion agresiva. `vmap` brilla precisamente en few-shot, donde a veces quieres procesar varios episodios en paralelo: `jax.vmap` sobre el eje de episodio te da batch de episodios casi gratis.

---

## 9. Gotchas por framework

Cosas que rompen el codigo si las pasas por alto, ordenadas por framework.

**Transversales (los tres):**

1. **Orden del soporte.** El `view/reshape` a `(n_way, k_shot, M)` asume que el soporte viene **ordenado por clase**. Si el sampler entrega filas mezcladas, el promedio computa centroides de clases revueltas y todo falla en silencio (el loss baja, pero la accuracy de test es basura). Verifica el orden que entrega tu sampler del camino 01.
2. **No hay `stop_gradient` en los prototipos.** Es tentador pensar que los prototipos son "objetivos fijos" y cortarles el gradiente. **No lo hagas.** El gradiente debe fluir por los soportes (que forman los prototipos) tanto como por las consultas. Cortar el gradiente del prototipo degrada el entrenamiento porque la red ya no aprende a colocar bien los soportes. El paper entrena end-to-end sin stop-gradient.
3. **Doble softmax.** Si tu funcion ya hace `log_softmax(-dists)`, no le pases el resultado a una cross-entropy que vuelve a normalizar. Pasa **logits crudos** (`-dists`) a la cross-entropy con `from_logits=True`, o usa `nll_loss` sobre log-probabilidades. Aplicar softmax dos veces aplana los gradientes y el modelo no aprende.

**PyTorch:**

4. **Broadcasting de `euclidean_dist`.** El `unsqueeze` + `expand` crea un tensor $(n, m, M)$ que puede ser grande. Para `n_way * q_query = 75` consultas, 5 prototipos y $M = 1600$, son $75 \cdot 5 \cdot 1600 = 600\text{K}$ floats —manejable. Pero para encoders grandes conviene `torch.cdist(z_query, prototypes).pow(2)`, que es mas eficiente en memoria.

**TensorFlow:**

5. **Channels-last.** Keras espera `(B, H, W, C)`. Si tu sampler entrega tensores estilo PyTorch `(B, C, H, W)`, transpone con `tf.transpose(x, [0, 2, 3, 1])` antes del encoder.
6. **`@tf.function` y shapes dinamicos.** Si el `n_way` cambia entre episodios, `@tf.function` re-traza (retracing) cada vez, lo que es lento. Mantenlo fijo dentro de una fase de entrenamiento.

**JAX:**

7. **`vmap` en distancias.** `jax.vmap(dist_one)(x)` vectoriza sobre el primer eje de `x`. Si confundes el eje, obtienes una matriz transpuesta silenciosamente. Verifica el shape resultante `(n_q, n_w)`.
8. **`batch_stats` del BatchNorm.** Es el error mas comun en Flax: olvidar propagar `mutable=["batch_stats"]` en train y `use_running_average` en eval. Si no lo haces, el BatchNorm usa estadisticas equivocadas y la accuracy de test colapsa.
9. **PRNGKey.** Si usaras dropout (no es el caso aqui, conv-4 no lo tiene), cada `apply` necesitaria su `rngs={"dropout": key}` con una key fresca. ProtoNets conv-4 no usa dropout, asi que esto no aplica, pero tenlo presente al portar otros encoders.

---

## 10. Limitaciones y conexion con Matching Networks

ProtoNets es deliberadamente simple, y esa simplicidad tiene costos que conviene conocer:

1. **Clusters unimodales.** Un prototipo por clase asume que cada clase es un unico *blob* compacto (gaussiana esferica). Clases multimodales —"perro" con razas muy distintas, o una patologia con presentaciones heterogeneas— violan ese supuesto. *Infinite Mixture Prototypes* aborda esto con varios prototipos por clase.
2. **Embedding fijo, no adaptado por tarea.** El encoder $f_\phi$ queda congelado tras el meta-entrenamiento. El mismo espacio debe servir para todas las tareas de test. Si una tarea nueva requiere atender dimensiones que el espacio ignora, ProtoNets no se reajusta. TADAM y los metodos de modulacion condicionada a la tarea atacan esto.
3. **Shift de dominio.** Si las clases de test vienen de una distribucion muy distinta (p. ej. entrenar en imagenes naturales, testear en imagenes medicas de otro centro), el embedding fijo puede no transferir. Es la regla, no la excepcion, en salud.

**La conexion con Matching Networks** (camino 05) es directa y muy instructiva:

- **En 1-shot son el mismo modelo.** Con un solo soporte por clase, $c_k = f_\phi(x_k)$: el prototipo *es* ese unico ejemplo. El softmax sobre distancias a prototipos colapsa al vecino mas cercano ponderado de Matching Networks. Por eso en las tablas del paper aparecen fusionados en 1-shot.
- **En K-shot ($K>1$) divergen.** Matching Networks aplica atencion sobre **todos** los puntos de soporte individuales (vecino mas cercano ponderado sobre $N \cdot K$ puntos). ProtoNets primero **promedia** los soportes de cada clase en un prototipo y luego compara contra los $N$ prototipos. Resultado: ProtoNets necesita guardar solo $N$ prototipos en inferencia (costo independiente del tamano del soporte), mientras Matching Networks debe conservar todos los puntos.
- **Lo que ProtoNets descarta.** Matching Networks anade *full context embeddings* (un LSTM bidireccional que hace que el embedding de cada punto dependa del resto del episodio) y la opcion de desacoplar los encoders de soporte y consulta. ProtoNets prescinde de ambos: mismo encoder, sin LSTM, apostando a que con datos escasos un sesgo inductivo simple basta. Y el paper le da la razon empiricamente.

En el camino 05 implementaremos Matching Networks y veremos en codigo esa transicion de "promediar" (ProtoNets) a "atender sobre todos los puntos" (Matching). En el camino 03 (siguiente) iremos al otro gran paradigma del few-shot —el *optimization-based*— con MAML, que en vez de aprender una metrica aprende una **inicializacion** que se adapta rapido a cada tarea con unos pocos pasos de gradiente. Metric-based vs optimization-based es el eje organizador de toda la clase.

---

**Ver tambien:** [Camino 01 - Episodios N-way K-shot](/clases/clase-26/practica/01-episodios-nway-kshot) · [Camino 03 - MAML](/clases/clase-26/practica/03-maml) · [Fundamento metric learning](/fundamentos/metric-learning) · [Fundamento meta-aprendizaje](/fundamentos/meta-aprendizaje) · [Paper Prototypical Networks (Snell et al. 2017)](/papers/prototypical-networks-snell-2017) · [Teoria de la clase](../teoria).
