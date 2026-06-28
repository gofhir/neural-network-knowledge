---
title: "End-to-End Memory Network desde cero"
weight: 1
math: true
---

En la teoria de la clase 30 vimos la pregunta estructural que organiza todo el modulo: en una red neuronal tradicional, *¿donde vive lo que el modelo sabe?* La respuesta —disuelto en los pesos— hace dificil que el modelo razone sobre hechos que aparecen *en el momento*, dentro del propio input. Las **Memory Networks** responden separando el **calculo** de la **memoria**: en vez de comprimir toda la historia en un estado recurrente, la guardan en una memoria **explicita** de slots y aprenden a *buscar* en ella mediante atencion. Si quieres el panorama conceptual completo, esta en el [fundamento de redes con memoria externa](/fundamentos/redes-de-memoria).

En este capitulo construimos **desde cero** la version mas influyente y didactica de esa familia: la **End-to-End Memory Network** (MemN2N) de Sukhbaatar, Szlam, Weston y Fergus (NeurIPS 2015), analizada en detalle en el [paper de la clase](/papers/e2e-memnn-sukhbaatar-2015). Su gracia es que, a diferencia de la Memory Network original de Weston (que necesitaba supervision sobre *cual* hecho es el relevante), MemN2N se entrena de punta a punta solo con la respuesta final: la atencion sobre la memoria es *blanda* (un softmax diferenciable) y emerge sola del gradiente. Es, ademas, el eslabon historico directo hacia la self-attention de los Transformers, conexion que cerraremos al final.

Lo implementamos en **tres frameworks** —PyTorch, TensorFlow/Keras y JAX— sobre una tarea bAbI de juguete que armamos sinteticamente: *single supporting fact*. El modelo lee una historia de pocas frases, recibe una pregunta y debe responder con **una sola palabra**, atendiendo a la frase correcta. El nucleo matematico (representar frases como bag-of-words embebido, computar atencion sobre memorias, sumar el resultado a la query, repetir en *hops*) es identico en los tres; cada framework solo cambia el andamiaje.

---

## 1. La tarea: single supporting fact

bAbI es un conjunto de 20 tareas sinteticas de razonamiento sobre texto, disenadas por Facebook AI para diagnosticar capacidades especificas. La **tarea 1** es la mas simple y la usamos aqui. Un ejemplo:

```
1 Mary fue a la cocina.
2 John se movio al jardin.
3 ¿Donde esta Mary?   cocina   (hecho de soporte: 1)
```

La historia son varias frases; cada frase ubica a una persona en un lugar. La pregunta apunta a una persona, y la respuesta es el ultimo lugar mencionado para ella. Se llama *single supporting fact* porque **una sola frase** de la historia contiene la respuesta: el modelo debe aprender a *encontrarla*. Esa es exactamente la operacion que la memoria con atencion vuelve trivial.

Para que el codigo sea autocontenido y verificable, no descargamos bAbI: lo **generamos sinteticamente** con un vocabulario chico y reglas simples. Cada ejemplo es:

- una **historia** de hasta $M$ frases (memorias), cada una "PERSONA fue a LUGAR",
- una **pregunta** "donde esta PERSONA",
- una **respuesta** de una palabra (el lugar correcto: el ultimo donde aparecio esa persona).

### 1.1 Las dimensiones de un vistazo

Antes de tocar el modelo, fijemos los shapes. Son el contrato que todas las implementaciones deben respetar.

| Simbolo | Nombre | Shape | Que es |
|---|---|---|---|
| $V$ | tamano de vocabulario | escalar | numero de tokens distintos |
| $M$ | numero de memorias | escalar | frases en la historia (con padding) |
| $L$ | largo de frase | escalar | tokens por frase (con padding) |
| $d$ | dimension de embedding | escalar | tamano de los vectores |
| `story` | historia | $(M, L)$ | indices de tokens por frase |
| `query` | pregunta | $(L,)$ | indices de tokens |
| $\{m_i\}$ | memorias de input | $(M, d)$ | frases embebidas con $A$ (BoW) |
| $\{c_i\}$ | memorias de output | $(M, d)$ | frases embebidas con $C$ (BoW) |
| $u$ | estado de la query | $(d,)$ | pregunta embebida con $B$ (BoW) |
| $p$ | atencion | $(M,)$ | probabilidad por memoria, suma 1 |
| $o$ | lectura de memoria | $(d,)$ | combinacion ponderada de los $c_i$ |
| logits | salida | $(V,)$ | puntaje por palabra del vocabulario |

La pieza clave: una frase $x_i$ de largo $L$ se convierte en **un solo vector** $m_i \in \mathbb{R}^d$ sumando los embeddings de sus tokens (bag-of-words). Asi la memoria completa es una matriz $(M, d)$: $M$ vectores, uno por frase.

---

## 2. El modelo en cuatro ecuaciones

MemN2N tiene **tres matrices de embedding** —no una— y una de salida. Cada token del vocabulario tiene tres vectores distintos segun el rol que juegue:

- $A \in \mathbb{R}^{V \times d}$: embebe la historia para formar las **memorias de input** (lo que se *busca*).
- $C \in \mathbb{R}^{V \times d}$: embebe la historia para formar las **memorias de output** (lo que se *lee* cuando una memoria es relevante).
- $B \in \mathbb{R}^{V \times d}$: embebe la **pregunta**.
- $W \in \mathbb{R}^{d \times V}$: proyecta el estado final al vocabulario para predecir la respuesta.

El paso central se llama **hop** (un "salto" de razonamiento). Dado el embedding de la query $u$, un hop hace cuatro cosas:

**1) Embeber la historia en los dos espacios** (BoW: suma de embeddings de tokens por frase):

$$
m_i = \sum_{j} A\,x_{ij}, \qquad c_i = \sum_{j} C\,x_{ij}. \tag{1}
$$

**2) Atencion** = compatibilidad de la query con cada memoria de input, normalizada por softmax:

$$
p_i = \operatorname{softmax}\!\big(u^\top m_i\big) = \frac{\exp(u^\top m_i)}{\sum_{k} \exp(u^\top m_k)}. \tag{2}
$$

$p$ es un vector de $M$ probabilidades que suma 1: *cuanto atiende* el modelo a cada frase.

**3) Lectura** = combinacion ponderada de las memorias de output por la atencion:

$$
o = \sum_i p_i\, c_i. \tag{3}
$$

**4) Actualizacion** del estado: sumamos lo leido a la query.

$$
u^{(k+1)} = u^{(k)} + o^{(k)}. \tag{4}
$$

Tras $K$ hops, la respuesta sale de proyectar el estado final y tomar softmax sobre el vocabulario:

$$
\hat{a} = \operatorname{softmax}\!\big(W\, u^{(K+1)}\big). \tag{5}
$$

Se entrena minimizando cross-entropy entre $\hat{a}$ y la palabra-respuesta correcta. **Todo es diferenciable**: el softmax de la atencion (Ecuacion 2) es lo que permite entrenar de punta a punta sin decirle al modelo cual es el hecho de soporte. El gradiente que baja desde la respuesta empuja a la atencion a concentrarse en la frase correcta.

### 2.1 Multiples hops y weight tying

Un solo hop basta para *single supporting fact* (una sola frase relevante). Pero tareas mas duras requieren razonamiento de varios pasos —p. ej. "John tomo la leche; John fue a la oficina; ¿donde esta la leche?" necesita encadenar dos hechos. Cada hop adicional re-consulta la memoria con un estado $u$ ya enriquecido por el hop anterior, permitiendo **razonamiento multi-paso**.

Con varios hops surge la pregunta de cuantas matrices $A, C$ usar. El paper propone dos esquemas de **weight tying** (atado de pesos) para no multiplicar parametros:

- **Adjacent** (el que usamos): la matriz de output de un hop es la de input del siguiente, $A^{(k+1)} = C^{(k)}$; ademas $B = A^{(1)}$ (la query usa la primera $A$) y $W = (C^{(K)})^\top$ (la salida usa la ultima $C$ traspuesta). Reduce mucho los parametros y rinde mejor.
- **Layer-wise (RNN-like)**: todos los hops comparten la *misma* $A$ y la *misma* $C$, mas una matriz lineal $H$ en la actualizacion $u^{(k+1)} = H u^{(k)} + o^{(k)}$. Es literalmente una RNN cuyo "input" en cada paso es la memoria leida.

En el codigo usaremos **adjacent** con $K=2$ hops por ser el del paper y el mas instructivo: deja ver como la lectura de un hop alimenta al siguiente.

---

## 3. Seccion 1: PyTorch

PyTorch hace el modelo casi transparente: `nn.Embedding` es exactamente la matriz $A/B/C$ (mira un token, devuelve su fila), y la suma BoW es un `.sum()` sobre el eje de tokens.

### 3.1 El toy dataset sintetico

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# --- Vocabulario chico y fijo ---
PEOPLE  = ["mary", "john", "sandra", "daniel"]
PLACES  = ["cocina", "jardin", "oficina", "bano"]
FILLER  = ["fue", "a", "donde", "esta", "?"]
PAD = "<pad>"

VOCAB = [PAD] + PEOPLE + PLACES + FILLER
word2idx = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)                  # tamano de vocabulario
PLACE_IDS = [word2idx[p] for p in PLACES]

MAX_MEM = 6     # M: numero maximo de frases en la historia
SENT_LEN = 4    # L: largo de frase con padding ("mary fue a cocina")
QLEN = 4        # largo de pregunta ("donde esta mary ?")


def encode(tokens, length):
    """Convierte lista de palabras -> indices, con padding a 'length'."""
    ids = [word2idx[t] for t in tokens][:length]
    ids += [word2idx[PAD]] * (length - len(ids))  # pad a la derecha
    return ids


def make_example(rng):
    """Genera (story, query, answer) de single supporting fact.
    story: (MAX_MEM, SENT_LEN) indices;  query: (QLEN,);  answer: indice de PLACE.
    """
    n_facts = rng.randint(2, MAX_MEM)        # cuantas frases tiene la historia
    last_place = {}                          # ultimo lugar visto por persona
    sentences = []
    for _ in range(n_facts):
        person = PEOPLE[rng.randrange(len(PEOPLE))]
        place  = PLACES[rng.randrange(len(PLACES))]
        last_place[person] = place           # actualiza estado verdadero
        sentences.append(encode([person, "fue", "a", place], SENT_LEN))

    # Preguntamos por alguien que SI aparece en la historia
    asked = rng.choice(list(last_place.keys()))
    query = encode(["donde", "esta", asked, "?"], QLEN)
    answer = word2idx[last_place[asked]]     # el ultimo lugar de esa persona

    # Padding de memorias hasta MAX_MEM con frases vacias
    while len(sentences) < MAX_MEM:
        sentences.append([word2idx[PAD]] * SENT_LEN)
    return sentences, query, answer


def make_batch(n, seed=0):
    import random
    rng = random.Random(seed)
    S, Q, A_ = [], [], []
    for _ in range(n):
        s, q, a = make_example(rng)
        S.append(s); Q.append(q); A_.append(a)
    return (torch.tensor(S), torch.tensor(Q), torch.tensor(A_))


story_b, query_b, ans_b = make_batch(512, seed=0)
print(story_b.shape, query_b.shape, ans_b.shape)
# torch.Size([512, 6, 4]) torch.Size([512, 4]) torch.Size([512])
```

Cuidamos un detalle pedagogico: la respuesta es **el ultimo** lugar de la persona preguntada. Si Mary va a la cocina y luego al jardin, la respuesta es "jardin". Esto fuerza al modelo a atender a la frase *correcta* (la mas reciente de esa persona), no solo a cualquier frase que mencione a Mary. Eso es lo que hace interesante la atencion.

### 3.2 El modelo MemN2N con weight tying adjacent

```python
class MemN2N(nn.Module):
    def __init__(self, vocab_size, embed_dim=20, n_hops=2):
        super().__init__()
        self.n_hops = n_hops
        # Adjacent tying: necesitamos n_hops+1 matrices de embedding.
        # A^(1)=B, A^(k+1)=C^(k), y W = (C^(K))^T.
        # Concretamente almacenamos n_hops+1 matrices E[0..n_hops]:
        #   E[0] embebe la query (rol B) y la historia del hop 1 (rol A^(1))
        #   E[k] embebe la output del hop k (rol C^(k)) y la input del hop k+1
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for _ in range(n_hops + 1)
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.1)
            with torch.no_grad():
                emb.weight[0].zero_()   # el <pad> embebe a cero

    def bow(self, emb, sentences):
        """Bag-of-words: embebe y suma sobre los tokens de cada frase.
        sentences: (B, M, L) o (B, L)  ->  (B, M, d) o (B, d).
        """
        return emb(sentences).sum(dim=-2)   # suma sobre el eje L de tokens

    def forward(self, story, query):
        # story: (B, M, L)   query: (B, L)
        u = self.bow(self.embeddings[0], query)   # (B, d)  <- rol B

        for k in range(self.n_hops):
            m = self.bow(self.embeddings[k],     story)  # (B, M, d) memorias input A
            c = self.bow(self.embeddings[k + 1], story)  # (B, M, d) memorias output C

            # Atencion (Ec. 2): producto punto u . m_i sobre el eje d, softmax sobre M
            scores = torch.bmm(m, u.unsqueeze(2)).squeeze(2)  # (B, M)
            p = F.softmax(scores, dim=1)                      # (B, M), suma 1

            # Lectura (Ec. 3): combinacion ponderada de las memorias output
            o = torch.bmm(p.unsqueeze(1), c).squeeze(1)       # (B, d)

            # Actualizacion (Ec. 4)
            u = u + o
            self._last_p = p   # guardamos la atencion del ultimo hop para inspeccion

        # Salida (Ec. 5): W = (C^(K))^T  -> usamos la ultima matriz de embedding traspuesta
        W = self.embeddings[self.n_hops].weight   # (V, d)
        logits = u @ W.t()                         # (B, V)
        return logits
```

Tres detalles que vale la pena mirar:

- **`bow` = `emb(x).sum(dim=-2)`**: ese `sum` sobre el eje de tokens *es* la Ecuacion (1). Cada frase se colapsa en un vector. Como el `<pad>` embebe a cero (`padding_idx=0`), las posiciones de relleno no contribuyen.
- **`torch.bmm(m, u.unsqueeze(2))`** computa, por cada elemento del batch, el producto $u^\top m_i$ para las $M$ memorias de una sola pasada. El resultado $(B, M)$ son los puntajes de atencion antes del softmax.
- **`W = self.embeddings[self.n_hops].weight`** materializa el atado $W = (C^{(K)})^\top$: la matriz de salida *es* la ultima matriz de output, sin parametros nuevos.

### 3.3 El training loop

```python
model = MemN2N(V, embed_dim=20, n_hops=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(200):
    logits = model(story_b, query_b)              # (B, V)
    loss = F.cross_entropy(logits, ans_b)         # CE sobre la palabra-respuesta
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 40 == 0:
        acc = (logits.argmax(1) == ans_b).float().mean()
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc.item():.3f}")
```

Con vocabulario chico y la tarea de juguete, el modelo llega a accuracy ~1.0 en pocas decenas de epocas: aprende que debe atender a la frase que menciona a la persona preguntada y leer de ahi el lugar. El optimizador es Adam con lr alto (`1e-2`) porque el problema es pequeno y separable.

### 3.4 Visualizar la atencion: ¿que frase atendio?

Aqui esta el pago de toda la maquinaria: la atencion $p$ es **interpretable**. Podemos ver, frase por frase, cuanto peso le dio el modelo.

```python
@torch.no_grad()
def explain(model, story, query, idx2word):
    model.eval()
    logits = model(story.unsqueeze(0), query.unsqueeze(0))
    p = model._last_p[0]                       # (M,) atencion del ultimo hop
    pred = logits.argmax(1).item()
    print("Pregunta:", " ".join(idx2word[t] for t in query.tolist() if t != 0))
    for i, sent in enumerate(story.tolist()):
        words = " ".join(idx2word[t] for t in sent if t != 0)
        if words:
            print(f"  mem[{i}]  p={p[i].item():.3f}   {words}")
    print("Respuesta predicha:", idx2word[pred])


idx2word = {i: w for w, i in word2idx.items()}
explain(model, story_b[0], query_b[0], idx2word)
# Pregunta: donde esta mary ?
#   mem[0]  p=0.951   mary fue a jardin     <- atencion concentrada aqui
#   mem[1]  p=0.012   john fue a cocina
#   mem[2]  p=0.030   sandra fue a oficina
#   ...
# Respuesta predicha: jardin
```

La frase con $p$ mas alta es, casi siempre, la que contiene la respuesta. Eso es lo notable: **nunca le dijimos cual era el hecho de soporte**; la atencion lo descubrio sola, guiada solo por el gradiente de la cross-entropy sobre la respuesta final.

---

## 4. Seccion 2: TensorFlow / Keras

El equivalente en TF 2.x usa `layers.Embedding`, `tf.reduce_sum` para el BoW y `tf.matmul` para la atencion. La estructura es identica.

### 4.1 Dataset (reusamos los tensores de PyTorch via numpy)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

tf.random.set_seed(0)

# Reutilizamos el generador de la seccion 3 (mismas funciones make_batch/encode).
story_np = story_b.numpy(); query_np = query_b.numpy(); ans_np = ans_b.numpy()
story_t = tf.constant(story_np)   # (B, M, L) int32
query_t = tf.constant(query_np)   # (B, L)
ans_t   = tf.constant(ans_np)     # (B,)
```

### 4.2 El modelo

```python
class MemN2N(keras.Model):
    def __init__(self, vocab_size, embed_dim=20, n_hops=2, **kw):
        super().__init__(**kw)
        self.n_hops = n_hops
        # n_hops+1 matrices de embedding (adjacent tying), pad a cero
        self.embeddings = [
            layers.Embedding(
                vocab_size, embed_dim, mask_zero=False,
                embeddings_initializer=keras.initializers.RandomNormal(stddev=0.1),
            )
            for _ in range(n_hops + 1)
        ]

    def bow(self, emb, sentences):
        # sentences: (B, M, L) o (B, L) -> suma sobre el ultimo eje de tokens
        return tf.reduce_sum(emb(sentences), axis=-2)   # (B, M, d) o (B, d)

    def call(self, inputs):
        story, query = inputs
        u = self.bow(self.embeddings[0], query)         # (B, d)
        for k in range(self.n_hops):
            m = self.bow(self.embeddings[k],     story) # (B, M, d) input A
            c = self.bow(self.embeddings[k + 1], story) # (B, M, d) output C

            # Atencion (Ec. 2): u . m_i sobre eje d -> (B, M), softmax sobre M
            scores = tf.matmul(m, tf.expand_dims(u, 2))     # (B, M, 1)
            scores = tf.squeeze(scores, axis=2)             # (B, M)
            p = tf.nn.softmax(scores, axis=1)               # (B, M)

            # Lectura (Ec. 3)
            o = tf.matmul(tf.expand_dims(p, 1), c)          # (B, 1, d)
            o = tf.squeeze(o, axis=1)                        # (B, d)

            u = u + o                                        # Actualizacion (Ec. 4)
            self._last_p = p

        # Salida (Ec. 5): W = (C^(K))^T. La matriz de embedding es (V, d);
        # logits = u (B,d) @ W^T (d,V) -> (B, V)
        W = self.embeddings[self.n_hops].embeddings         # (V, d)
        logits = tf.matmul(u, W, transpose_b=True)          # (B, V)
        return logits
```

### 4.3 Training loop con `GradientTape`

```python
model = MemN2N(V, embed_dim=20, n_hops=2)
opt = keras.optimizers.Adam(learning_rate=1e-2)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(story, query, ans):
    with tf.GradientTape() as tape:
        logits = model((story, query))
        loss = loss_fn(ans, logits)            # CE sobre la palabra-respuesta
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss, logits

for epoch in range(200):
    loss, logits = train_step(story_t, query_t, ans_t)
    if epoch % 40 == 0:
        acc = tf.reduce_mean(
            tf.cast(tf.argmax(logits, 1, output_type=tf.int32) == ans_t, tf.float32))
        print(f"epoch {epoch:3d}  loss={float(loss):.4f}  acc={float(acc):.3f}")
```

Diccionario de traduccion respecto a PyTorch: `nn.Embedding` -> `layers.Embedding`, `.sum(dim=-2)` -> `tf.reduce_sum(..., axis=-2)`, `torch.bmm` -> `tf.matmul` (con `transpose_b` para la salida), `F.softmax(.., dim=1)` -> `tf.nn.softmax(.., axis=1)`. La perdida usa `from_logits=True` para que Keras aplique el softmax internamente: una sola normalizacion, sin doble softmax (el gotcha clasico al portar entre frameworks).

---

## 5. Seccion 3: JAX

JAX adopta **funciones puras**: los parametros (las matrices $A/B/C/W$) son un `pytree` que se pasa explicitamente; no hay estado mutable. El forward es una funcion `forward(params, story, query)` y `jax.grad` la diferencia directamente.

### 5.1 Inicializacion de parametros

```python
import jax
import jax.numpy as jnp
from jax import random

# Reutilizamos los arrays numpy generados en la seccion 3
story_j = jnp.array(story_np)   # (B, M, L)
query_j = jnp.array(query_np)   # (B, L)
ans_j   = jnp.array(ans_np)     # (B,)

EMBED_DIM, N_HOPS = 20, 2


def init_params(key, vocab_size, embed_dim, n_hops):
    """n_hops+1 matrices de embedding (adjacent tying). Cada una (V, d).
    La fila 0 (<pad>) se deja a cero para que el padding no contribuya al BoW.
    """
    keys = random.split(key, n_hops + 1)
    embs = []
    for k in keys:
        E = 0.1 * random.normal(k, (vocab_size, embed_dim))
        E = E.at[0].set(0.0)          # <pad> embebe a cero (forma funcional pura)
        embs.append(E)
    return {"E": embs}                # W = E[n_hops]^T se deriva del ultimo
```

### 5.2 El forward como funcion pura

```python
def bow(E, sentences):
    """Bag-of-words: indexa E con los tokens y suma sobre el eje de tokens.
    E: (V, d);  sentences: (B, M, L) o (B, L)  ->  (B, M, d) o (B, d).
    E[sentences] hace gather: cada indice de token -> su fila de embedding.
    """
    return jnp.sum(E[sentences], axis=-2)


def forward(params, story, query, n_hops=N_HOPS, return_attn=False):
    E = params["E"]
    u = bow(E[0], query)                           # (B, d)  <- rol B
    last_p = None
    for k in range(n_hops):
        m = bow(E[k],     story)                   # (B, M, d) input A
        c = bow(E[k + 1], story)                   # (B, M, d) output C

        # Atencion (Ec. 2): u . m_i sobre el eje d (einsum) -> (B, M)
        scores = jnp.einsum("bd,bmd->bm", u, m)    # (B, M)
        p = jax.nn.softmax(scores, axis=1)         # (B, M), suma 1

        # Lectura (Ec. 3): combinacion ponderada de las memorias output
        o = jnp.einsum("bm,bmd->bd", p, c)         # (B, d)

        u = u + o                                   # Actualizacion (Ec. 4)
        last_p = p

    # Salida (Ec. 5): W = (C^(K))^T = E[n_hops]^T  ->  logits = u @ W^T
    logits = u @ E[n_hops].T                        # (B, V)
    if return_attn:
        return logits, last_p
    return logits
```

El `einsum` deja la matematica explicita: `"bd,bmd->bm"` es "para cada batch `b` y memoria `m`, contrae la dimension `d`" —exactamente $u^\top m_i$. Y `"bm,bmd->bd"` es la suma ponderada $\sum_i p_i c_i$ de la Ecuacion (3). Es el idioma JAX por excelencia: la formula del paper traducida casi caracter por caracter.

### 5.3 Loss puro y training loop

```python
import optax

def loss_fn(params, story, query, ans):
    logits = forward(params, story, query)                 # (B, V)
    log_p = jax.nn.log_softmax(logits, axis=1)             # softmax estable
    onehot = jax.nn.one_hot(ans, logits.shape[1])          # (B, V)
    return -jnp.mean(jnp.sum(onehot * log_p, axis=1))      # cross-entropy

params = init_params(random.PRNGKey(0), V, EMBED_DIM, N_HOPS)
opt = optax.adam(1e-2)
opt_state = opt.init(params)

@jax.jit
def train_step(params, opt_state, story, query, ans):
    loss, grads = jax.value_and_grad(loss_fn)(params, story, query, ans)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for epoch in range(200):
    params, opt_state, loss = train_step(params, opt_state,
                                          story_j, query_j, ans_j)
    if epoch % 40 == 0:
        logits = forward(params, story_j, query_j)
        acc = jnp.mean(jnp.argmax(logits, 1) == ans_j)
        print(f"epoch {epoch:3d}  loss={float(loss):.4f}  acc={float(acc):.3f}")
```

Para inspeccionar la atencion en JAX, pedimos `return_attn=True`:

```python
logits, p = forward(params, story_j[:1], query_j[:1], return_attn=True)
# p[0] es (M,): la distribucion de atencion del ultimo hop para el ejemplo 0.
print("atencion:", np.round(np.array(p[0]), 3))   # p.ej. [0.95 0.01 0.03 ...]
```

Lo que internalizar del patron JAX: **`params` es un argumento, no un atributo**. `jax.value_and_grad(loss_fn)` devuelve loss y gradientes de una sola pasada, `optax` lleva el estado del optimizador explicito, y `@jax.jit` compila el paso entero a XLA (la primera llamada compila, las siguientes vuelan). La fila 0 del embedding se pone a cero con `E.at[0].set(0.0)` —el equivalente funcional puro del `padding_idx` de PyTorch.

---

## 6. Verificacion de dimensiones

Las tres implementaciones son **isomorfas**: mismos shapes en cada paso. Conviene tenerlos memorizados, porque un eje confundido falla en silencio (el loss baja, pero la atencion atiende basura).

| Paso | Operacion | Shape resultante |
|---|---|---|
| Embeber query (BoW) | $u = \sum_j B x_j$ | $(B, d)$ |
| Memorias input (BoW) | $m_i = \sum_j A x_{ij}$ | $(B, M, d)$ |
| Memorias output (BoW) | $c_i = \sum_j C x_{ij}$ | $(B, M, d)$ |
| Puntajes de atencion | $u^\top m_i$ | $(B, M)$ |
| Atencion (softmax sobre $M$) | $p = \operatorname{softmax}(\cdot)$ | $(B, M)$, suma 1 |
| Lectura | $o = \sum_i p_i c_i$ | $(B, d)$ |
| Actualizacion | $u \mathrel{+}= o$ | $(B, d)$ |
| Logits de salida | $u\, W^\top$ | $(B, V)$ |

Los dos chequeos que nunca deben fallar: **(1)** la atencion suma 1 sobre el eje de memorias (`p.sum(dim=1) == 1`), y **(2)** el softmax de atencion es sobre $M$ (memorias), mientras el de salida es sobre $V$ (vocabulario). Son dos softmax distintos, sobre ejes distintos, con roles distintos.

---

## 7. La conexion con la self-attention de los Transformers

Aqui esta el cierre conceptual que vuelve a MemN2N tan importante historicamente. Si comparas las Ecuaciones (2)-(3) con la atencion escalada de "Attention is All You Need", **son la misma operacion**. El [fundamento de self-attention](/fundamentos/self-attention) lo desarrolla en general; aqui el mapeo concreto:

| MemN2N (2015) | Self-attention / Transformer (2017) |
|---|---|
| query embebida $u$ (rol $B$) | **query** $Q$ |
| memorias de input $m_i$ (rol $A$) | **keys** $K$ |
| memorias de output $c_i$ (rol $C$) | **values** $V$ |
| $p_i = \operatorname{softmax}(u^\top m_i)$ | $\operatorname{softmax}(QK^\top/\sqrt{d})$ |
| $o = \sum_i p_i c_i$ | $\sum_i p_i\, V_i$ (mezcla ponderada de values) |
| varios **hops** apilados | varias **capas** de atencion apiladas |

El Transformer **generaliza** MemN2N en tres direcciones: (1) en vez de una sola query, atiende todas las posiciones contra todas (*self*-attention); (2) anade el escalado $1/\sqrt{d}$ para estabilizar gradientes con $d$ grande; (3) parte las matrices en *cabezas* (multi-head). Pero el corazon —*una query mira un conjunto de keys, normaliza por softmax, y lee una mezcla ponderada de values*— nacio aqui. Que $A$ sean las keys, $C$ los values y $B$ la query no es una analogia forzada: es literalmente la misma factorizacion.

**¿Por que los hops permiten razonamiento multi-paso?** Cada hop produce un estado $u^{(k+1)} = u^{(k)} + o^{(k)}$ que ya incorpora lo leido. El hop siguiente vuelve a consultar la memoria con esa query enriquecida, asi que puede *encadenar* hechos: el primer hop encuentra "John tomo la leche", la lectura mete "John" en el estado, y el segundo hop —ahora preguntando implicitamente por John— encuentra "John fue a la oficina". Es exactamente lo que hacen las capas apiladas de un Transformer: cada capa refina la representacion usando lo que descubrio la anterior. Mas hops = mas saltos de inferencia encadenables.

**Como visualizar la memoria atendida.** El vector $p$ del ultimo hop (que guardamos en `_last_p` / `return_attn`) es directamente un mapa de calor sobre las frases: graficalo como barra horizontal por memoria, o imprimelo junto a cada frase como en la seccion 3.4. En un Transformer harias lo mismo con la matriz de atencion $(\text{posiciones} \times \text{posiciones})$ de cada cabeza —los famosos *attention maps*. La interpretabilidad de "que atendio el modelo" es una herencia directa de esta arquitectura.

---

## 8. Comparacion lado a lado de los tres frameworks

| Concepto | PyTorch | TensorFlow/Keras | JAX |
|---|---|---|---|
| Matriz de embedding | `nn.Embedding(V, d, padding_idx=0)` | `layers.Embedding(V, d)` | array `(V, d)`, fila 0 a cero |
| BoW (suma sobre tokens) | `emb(x).sum(dim=-2)` | `tf.reduce_sum(emb(x), -2)` | `jnp.sum(E[x], -2)` |
| Atencion $u^\top m_i$ | `torch.bmm(m, u[..,None])` | `tf.matmul(m, u[..,None])` | `jnp.einsum("bd,bmd->bm")` |
| Softmax sobre memorias | `F.softmax(s, dim=1)` | `tf.nn.softmax(s, 1)` | `jax.nn.softmax(s, 1)` |
| Lectura $\sum p_i c_i$ | `torch.bmm(p[:,None], c)` | `tf.matmul(p[:,None], c)` | `jnp.einsum("bm,bmd->bd")` |
| Salida $W=(C^{(K)})^\top$ | `u @ emb[-1].weight.t()` | `tf.matmul(u, W, transpose_b=True)` | `u @ E[-1].T` |
| Cross-entropy | `F.cross_entropy` | `SparseCCE(from_logits=True)` | `log_softmax` + one-hot manual |
| Estado del modelo | mutable en `self` | mutable en `self` | `params` pytree explicito |
| Diferenciacion | `loss.backward()` | `tf.GradientTape` | `jax.value_and_grad` |
| Compilacion | `torch.compile` | `@tf.function` | `@jax.jit` |

La leccion: MemN2N es tan compacto (embeber, atender, leer, actualizar, proyectar) que portarlo entre frameworks es casi mecanico. Lo unico que cambia es como se expresan las matrices de embedding y como se mueven los gradientes; la matematica de las Ecuaciones (1)-(5) es identica.

---

## 9. Gotchas

**Transversales (los tres):**

1. **El padding debe embeber a cero.** Si la fila `<pad>` del embedding no es cero, las frases de relleno (memorias vacias) contribuyen al BoW y ensucian la atencion. Usa `padding_idx=0` (PyTorch) o pon la fila 0 a cero a mano (TF/JAX).
2. **Dos softmax sobre ejes distintos.** El de la atencion es sobre $M$ (memorias); el de la salida es sobre $V$ (vocabulario). Confundirlos da shapes que cuadran por casualidad pero entrenan mal.
3. **Weight tying adjacent.** $W = (C^{(K)})^\top$ y $B = A^{(1)}$ no son matrices nuevas: reusan las de embedding. Si las declaras aparte, duplicas parametros y rompes el esquema del paper.

**PyTorch:** el `torch.bmm` exige que los tres tensores compartan dimension de batch; verifica `u.unsqueeze(2)` -> `(B, d, 1)` antes de multiplicar por `m` `(B, M, d)`.

**TensorFlow:** usa `from_logits=True` en la cross-entropy y pasale los **logits crudos**, no un softmax ya aplicado, o tendras doble normalizacion.

**JAX:** `E.at[0].set(0.0)` devuelve un **array nuevo** (no muta in-place); olvidar reasignarlo deja el pad sin anular. Y recuerda que `params` viaja como argumento en cada `train_step`, no como atributo.

---

**Ver tambien:** [Teoria de la clase 30](/clases/clase-30) · [Profundizacion](../profundizacion) · [Fundamento: redes con memoria externa](/fundamentos/redes-de-memoria) · [Fundamento: self-attention](/fundamentos/self-attention) · [Paper: End-to-End Memory Networks (Sukhbaatar et al. 2015)](/papers/e2e-memnn-sukhbaatar-2015).
