---
title: "03 - Decoder causal mini"
weight: 33
math: true
---

Despues de haber construido un encoder bidireccional con MLM en el [Camino 02](/clases/clase-20/practica/02-mlm-encoder-mini), volvemos a montar el mismo Transformer pero cambiandole una sola cosa: la mascara. En lugar de dejar que cada token vea a sus vecinos en ambas direcciones, lo forzamos a mirar solo el pasado. Esa restriccion convierte el encoder en un **decoder causal**, y el objetivo de entrenamiento deja de ser "reconstruir" para pasar a ser "predecir el siguiente token". Es el cambio que separa BERT de GPT — y, conceptualmente, casi todo lo demas se queda igual.

Este camino replica un mini-GPT en los tres frameworks principales: **PyTorch**, **TensorFlow/Keras** y **JAX + Flax**. La idea no es escribir tres modelos distintos, sino el mismo modelo escrito tres veces para que veas con claridad como cada framework expresa los mismos conceptos: la mascara triangular, el pre-LayerNorm, el sampling autoregresivo, el training step. Si dominaste el [escalon 08 de la Clase 14](/clases/clase-14/practica/08-mini-gpt), aqui solo cambian dos cosas: la arquitectura es un poquito mas chica para que el triple-framework no se haga pesado, y se introducen las dos rutas alternativas (TF y JAX).

---

## 1. Especificacion del modelo

Antes de meternos en codigo, fijemos el diseno comun a los tres frameworks:

| Hiperparametro     | Valor | Justificacion                                         |
|--------------------|-------|--------------------------------------------------------|
| `n_layers`         | 2     | Suficiente para ver patrones, barato de entrenar       |
| `d_model`          | 64    | Dimension de embeddings y residual stream              |
| `n_heads`          | 4     | $d_k = d_v = 16$ por cabeza                            |
| `d_ff`             | 256   | Expansion $4 \times d_{model}$ en el FFN                |
| `vocab_size`       | 1000  | BPE pequeno entrenado sobre el toy corpus              |
| `block_size`       | 64    | Ventana de contexto                                    |
| Normalizacion      | Pre-LN | LayerNorm antes de cada subcapa (no post-LN como BERT) |
| Activacion FFN     | GELU  | Estandar GPT-2/3                                       |
| Embedding posicion | Aprendido | Igual que GPT-2; mas adelante mostramos como pasar a RoPE |
| Tied embeddings    | Si    | La cabeza de salida comparte pesos con `token_emb`     |
| Objetivo           | Next-token cross-entropy (causal LM) | $\mathcal{L} = -\sum_t \log p(x_{t+1} \mid x_{\le t})$ |

El parametro total ronda los **180 K**: dos ordenes de magnitud por debajo de GPT-2 small (124 M) y suficientes para mostrar un texto generado con coherencia local en menos de un minuto de entrenamiento.

### 1.1 La diferencia con el Camino 02 en una linea

Lo unico que cambia entre el encoder bidireccional del Camino 02 y este decoder causal es la mascara que se aplica a la matriz de atencion antes del softmax:

$$
\text{Encoder MLM:}\quad M_{ij} = 0 \ \forall i, j \qquad
\text{Decoder causal:}\quad M_{ij} = \begin{cases} 0 & \text{si } j \le i \\ -\infty & \text{si } j > i \end{cases}
$$

Esa $M$ se suma a los scores $QK^\top / \sqrt{d_k}$ antes del softmax. Todo lo demas — embeddings, multi-head, FFN, LayerNorm, residuales, optimizer, dataset — es identico. La causal mask cuesta tres lineas y define toda la familia de los LLMs generativos.

{{< concept-alert type="clave" >}}
La mascara causal es el unico cambio arquitectonico entre el Camino 02 (encoder/BERT) y el Camino 03 (decoder/GPT). Esa restriccion sobre que tokens puede ver cada posicion es lo que habilita la **generacion autoregresiva**: dado un prefijo, predecir un token a la vez y concatenar. Sin esa restriccion, el modelo veria el futuro durante el entrenamiento y aprenderia a copiarlo en lugar de predecirlo.
{{< /concept-alert >}}

### 1.2 Objetivo de entrenamiento formal

El decoder causal modela la distribucion conjunta de una secuencia $x = (x_1, \dots, x_T)$ factorizada por la regla de la cadena:

$$
p_\theta(x_1, \dots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_1, \dots, x_{t-1}).
$$

El loss por secuencia es la negative log-likelihood promedio:

$$
\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
= \frac{1}{T} \sum_{t=1}^{T} \text{CE}\big(\text{logits}_t,\ x_t\big).
$$

En la practica: hacemos forward pass sobre una ventana de $T$ tokens, los logits en la posicion $t$ predicen el token $t+1$, y aplicamos cross-entropy sobre todas las $T-1$ predicciones de cada secuencia.

### 1.3 El toy corpus

Para que el codigo sea reproducible y rapido en CPU, usamos un corpus chico de citas en espanol (~250 KB, ~50 K tokens BPE). El tokenizer BPE se entrena una vez sobre ese corpus con `vocab_size = 1000` usando `tokenizers` de Hugging Face. Cualquier corpus de texto plano funciona: Quijote completo, articulos de Wikipedia, reportes clinicos anonimizados.

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=1000, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("bpe-1000.json")

VOCAB_SIZE = tokenizer.get_vocab_size()  # 1000
```

El mismo `tokenizer` y las mismas listas de `input_ids` se reutilizan en los tres frameworks. Solo cambia como se convierten esos enteros en tensores: `torch.tensor`, `tf.constant`, `jnp.array`.

---

## 2. Implementacion en PyTorch

Empezamos por PyTorch porque es el framework donde la pedagogia es mas directa: `nn.Module`, `forward`, `loss.backward()`, `optimizer.step()`. Todo es imperativo y los shapes se pueden inspeccionar con `print` en cualquier punto.

### 2.1 Causal self-attention

La mascara se precomputa una vez al construir el modulo y se aplica con `masked_fill_` antes del softmax:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

        causal_mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.W_qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_out(out)
```

Detalles clave:

- `torch.triu(..., diagonal=1)` produce una matriz `True` **estrictamente arriba** de la diagonal — exactamente las posiciones del futuro.
- `masked_fill_` reemplaza esos lugares con $-\infty$ **antes** del softmax. Asi $e^{-\infty} = 0$ y el peso de las posiciones futuras se anula.
- El QKV se calcula con una sola proyeccion `W_qkv: d_model -> 3*d_model` y luego se parte con `unbind`. Es un truco estandar en GPT-2 y nanoGPT: una multiplicacion de matriz mas grande es mas eficiente en GPU que tres mas chicas.
- `register_buffer` registra el tensor de mascara como parte del modulo (`.to(device)` lo mueve) pero no como parametro aprendible. La mascara es estructura, no aprendizaje.

### 2.2 Bloque Transformer con Pre-LN

La diferencia entre **Pre-LN** y **Post-LN** parece menor pero es la razon por la que los GPT-2/3 entrenan estables sin warmup agresivo:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

Pre-LN: el LayerNorm se aplica **antes** de cada subcapa, y la conexion residual va del input directo al output. Post-LN (BERT original) hace al reves: subcapa primero, residual, y LayerNorm despues. Pre-LN tiene una propiedad importante — el gradiente que llega al embedding inicial no pasa por ningun LayerNorm en el camino directo, lo cual estabiliza el entrenamiento de redes profundas y permite quitar el warmup de learning rate sin que el entrenamiento explote.

### 2.3 MiniGPT completo

Ensamblando todo, con embeddings de token + posicion, $N$ bloques y la cabeza tied:

```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, n_heads=4, n_layers=2,
                 d_ff=256, block_size=64):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        self.token_emb.weight.data.normal_(0.0, 0.02)
        self.pos_emb.weight.data.normal_(0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        h = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = h @ self.token_emb.weight.T

        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        return logits, loss
```

Dos detalles importantes:

- **Tied embeddings**: en lugar de aprender una matriz separada `nn.Linear(d_model, vocab_size)` para la cabeza de salida, reutilizamos la transpuesta de `token_emb.weight`. Esto ahorra `vocab_size * d_model = 64 000` parametros (mas del 30% del total) y, ademas, mejora ligeramente la perplexity en modelos chicos. Lo usan GPT-2, GPT-3 y todos los LLMs modernos.
- El loss se calcula solo si vienen `targets`. Eso permite usar el mismo `forward` para entrenamiento (con targets) y para generacion (sin targets).

### 2.4 Pre-training loop autoregresivo

El truco para autoregresivo es trivial pero hay que tenerlo claro: dado un batch `input_ids` de shape `(B, T)`, los targets son `input_ids` desplazado un token a la izquierda:

```python
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for step in range(2000):
    x, y = get_batch(train_ids, block_size=64, batch_size=32, device=device)
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step:5d} | loss {loss.item():.4f}")
```

Notar que **no hay que recortar manualmente** los logits para calcular el loss. Cada posicion del input predice el siguiente token, y `targets[i] = input_ids[i+1]`. El `cross_entropy` sobre los logits aplanados castiga directamente a las predicciones que se equivocan.

`clip_grad_norm_` es un detalle de ingenieria importante: las primeras iteraciones tienden a tener gradientes grandes que pueden hacer divergir el modelo. Recortar la norma global a 1.0 estabiliza el entrenamiento sin afectar las direcciones.

### 2.5 Generacion autoregresiva con temperatura y top-k

```python
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    idx = prompt_ids.clone()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, k=top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        if temperature == 0.0:
            idx_next = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Tres comportamientos en una funcion:

- `temperature=0.0`: argmax puro. Determinista, repetitivo, a menudo entra en loops.
- `temperature` baja (0.3-0.7): distribucion afilada pero con algo de variedad. Lo que usaria un asistente "tono profesional".
- `temperature` alta (1.5+): distribucion casi uniforme, texto creativo / caotico.
- `top_k`: filtra los logits a los $k$ mas probables antes del softmax. Evita que probabilidades bajas pero no nulas produzcan tokens raros. Es la version primitiva del **nucleus / top-p sampling** usado en produccion.

Formalmente, la temperatura escala los logits antes del softmax:

$$
p_\tau(x_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)},
$$

donde $\tau \to 0^+$ converge al argmax y $\tau \to \infty$ a la uniforme.

---

## 3. Implementacion en TensorFlow / Keras

TensorFlow tiene una capa nativa `tf.keras.layers.MultiHeadAttention` que acepta `use_causal_mask=True` y se encarga de construir y aplicar la mascara internamente. Eso ahorra codigo pero esconde la mecanica: vale la pena ver una version "honesta" donde llamamos a la capa y entendemos que esta haciendo.

### 3.1 Bloque Transformer con Pre-LN

```python
import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            use_bias=False,
        )
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dense(d_model),
        ])

    def call(self, x, training=False):
        h = self.ln1(x)
        attn_out = self.attn(query=h, value=h, key=h, use_causal_mask=True, training=training)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x), training=training)
        return x
```

Diferencias frente a la version PyTorch:

- En Keras la dimension por cabeza se pasa como `key_dim`, no se infiere de `d_model / n_heads`. Hay que calcularlo a mano.
- La mascara causal **no se construye explicitamente**. Se delega al flag `use_causal_mask=True`, que internamente genera una mascara con `tf.linalg.band_part` y la aplica antes del softmax.
- El metodo se llama `call`, no `forward`. Y recibe `training` para que dropout (si lo agregaramos) se comporte bien.
- La normalizacion `LayerNormalization` usa `epsilon=1e-5` por default; lo dejamos explicito para que matchee PyTorch.

### 3.2 MiniGPT en Keras subclassing

```python
class MiniGPT(tf.keras.Model):
    def __init__(self, vocab_size=1000, d_model=64, n_heads=4,
                 n_layers=2, d_ff=256, block_size=64, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(block_size, d_model)
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, name=f"block_{i}")
            for i in range(n_layers)
        ]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)

    def call(self, idx, training=False):
        T = tf.shape(idx)[1]
        positions = tf.range(T)
        h = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            h = block(h, training=training)
        h = self.ln_f(h)
        logits = tf.matmul(h, self.token_emb.embeddings, transpose_b=True)
        return logits
```

El tied embedding en TF se hace con `tf.matmul(h, self.token_emb.embeddings, transpose_b=True)`. `self.token_emb.embeddings` accede al kernel de la capa Embedding (matriz `(vocab_size, d_model)`), y `transpose_b=True` lo transpone para la multiplicacion final.

### 3.3 Training step custom con GradientTape

Podriamos usar `model.compile(...)` + `model.fit(...)`, pero para ver bien el control de gradientes hacemos el step a mano:

```python
optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=0.1, clipnorm=1.0)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

model = MiniGPT()
_ = model(tf.zeros((1, 64), dtype=tf.int32))  # build

for step in range(2000):
    x, y = get_batch_tf(train_ids, block_size=64, batch_size=32)
    loss = train_step(model, x, y)
    if step % 200 == 0:
        tf.print("step", step, "loss", loss)
```

`@tf.function` compila el training step a un grafo XLA. Esto es lo que da a TF la performance comparable a JAX en cargas similares. La primera ejecucion toma un par de segundos extra (compilacion), las siguientes son rapidisimas.

`SparseCategoricalCrossentropy(from_logits=True)` espera **etiquetas enteras** (no one-hot) y **logits** (no probabilidades), igual que `F.cross_entropy` en PyTorch. `from_logits=True` aplica el softmax internamente con estabilidad numerica.

### 3.4 Generacion en TF: tf.random.categorical

```python
def generate_tf(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None):
    idx = tf.identity(prompt_ids)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond, training=False)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = tf.math.top_k(logits, k=top_k)
            min_top_k = v[:, -1:]
            logits = tf.where(logits < min_top_k, tf.fill(tf.shape(logits), -1e9), logits)

        if temperature == 0.0:
            idx_next = tf.argmax(logits, axis=-1, output_type=tf.int32)[:, None]
        else:
            idx_next = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)

        idx = tf.concat([idx, idx_next], axis=1)
    return idx
```

La diferencia clave con PyTorch: en TF la funcion para samplear de logits es `tf.random.categorical`, que **toma logits directamente** y aplica el softmax internamente. PyTorch `torch.multinomial` espera **probabilidades** (necesitas aplicar `softmax` antes). Es facil meter la pata cuando se hace switching entre frameworks.

Para top-k, en lugar de `masked_fill_` usamos `tf.where` con un valor muy negativo (`-1e9`) que en la practica cumple el rol de $-\infty$ tras el softmax.

---

## 4. Implementacion en JAX + Flax

JAX es funcionalmente distinto: los modulos `nn.Module` de Flax no llevan estado mutable; los parametros se pasan explicitamente al forward via un PyTree, y los gradientes se calculan con `jax.grad`. La compensacion: una vez que entendes el flujo, `jax.jit` te da compilacion XLA gratis y JAX paraleliza por defecto en TPUs.

### 4.1 Bloque Transformer con Pre-LN en Flax

Flax tiene `nn.MultiHeadDotProductAttention` y un helper `nn.make_causal_mask` que produce la mascara con shape adecuada para broadcasting con los scores:

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x, mask):
        h = nn.LayerNorm(epsilon=1e-5)(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            use_bias=False,
        )(h, h, mask=mask)
        x = x + attn

        h2 = nn.LayerNorm(epsilon=1e-5)(x)
        ff = nn.Dense(self.d_ff)(h2)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.d_model)(ff)
        return x + ff
```

`@nn.compact` permite definir las subcapas inline dentro de `__call__`. Es el estilo idiomatico de Flax y se siente raro al principio si vienes de PyTorch: los submodulos se "registran" en el momento que se llaman por primera vez, no en un `__init__`.

`nn.MultiHeadDotProductAttention` toma `query` y `key` (y opcionalmente `value`; si no se pasa, usa `key`). El `mask` se suma a los scores antes del softmax. Tiene un argumento `decode=False` que dejamos por default — `decode=True` activaria el KV-cache para inferencia, lo discutimos en la seccion 7.

### 4.2 MiniGPT en Flax

```python
class MiniGPT(nn.Module):
    vocab_size: int = 1000
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    block_size: int = 64

    @nn.compact
    def __call__(self, idx):
        B, T = idx.shape

        token_emb = nn.Embed(self.vocab_size, self.d_model, name="token_emb")
        pos_emb = nn.Embed(self.block_size, self.d_model, name="pos_emb")

        positions = jnp.arange(T)
        h = token_emb(idx) + pos_emb(positions)

        causal_mask = nn.make_causal_mask(idx, dtype=jnp.bool_)

        for _ in range(self.n_layers):
            h = TransformerBlock(self.d_model, self.n_heads, self.d_ff)(h, causal_mask)

        h = nn.LayerNorm(epsilon=1e-5)(h)
        logits = h @ token_emb.embedding.T
        return logits
```

`nn.make_causal_mask(idx)` produce un tensor de shape `(B, 1, T, T)` con `True` en el triangulo inferior (incluyendo diagonal) y `False` arriba — exactamente lo que necesita Flax: los `False` se enmascaran. La cabeza tied accede a `token_emb.embedding` (matriz `(vocab_size, d_model)`) y la transpone.

### 4.3 Training step con jax.jit y optax

```python
def loss_fn(params, idx, targets):
    logits = model.apply({"params": params}, idx)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
    loss = -(one_hot * log_probs).sum(axis=-1).mean()
    return loss

@jax.jit
def train_step(state, idx, targets):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, idx, targets)
    state = state.apply_gradients(grads=grads)
    return state, loss

model = MiniGPT()
key = jax.random.PRNGKey(0)
dummy = jnp.zeros((1, 64), dtype=jnp.int32)
params = model.init(key, dummy)["params"]

tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=3e-4, weight_decay=0.1),
)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

for step in range(2000):
    x, y = get_batch_jax(train_ids, block_size=64, batch_size=32, key=jax.random.PRNGKey(step))
    state, loss = train_step(state, x, y)
    if step % 200 == 0:
        print(f"step {step:5d} | loss {float(loss):.4f}")
```

Observaciones:

- **Los parametros estan separados del modelo**. `model` es solo una descripcion estructural; `params` es el dict de pesos que va por separado a cada llamada `apply`. Eso permite que `jax.grad` sepa exactamente cuales son las variables a diferenciar.
- **`jax.jit`** compila el grafo de XLA una sola vez. El primer step es lento (compilacion); los siguientes son ordenes de magnitud mas rapidos.
- **`optax.chain`** compone transformaciones del gradiente. Aqui aplicamos clip global por norma 1.0 antes del paso de AdamW. Es la version JAX de `clip_grad_norm_` + `AdamW(weight_decay=0.1)`.
- **`jax.value_and_grad`** devuelve el loss y los gradientes en una sola pasada — equivalente a `loss.backward()` en PyTorch pero funcional: no muta nada, devuelve un PyTree de gradientes con la misma estructura que `params`.

### 4.4 Generacion en JAX con jax.random.categorical

```python
def generate_jax(model, params, prompt_ids, max_new_tokens, key,
                 temperature=1.0, top_k=None, block_size=64):
    idx = prompt_ids
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model.apply({"params": params}, idx_cond)
        logits = logits[:, -1, :] / jnp.maximum(temperature, 1e-8)

        if top_k is not None:
            v = jax.lax.top_k(logits, top_k)[0]
            min_top_k = v[:, -1:]
            logits = jnp.where(logits < min_top_k, -1e9, logits)

        if temperature == 0.0:
            idx_next = jnp.argmax(logits, axis=-1, keepdims=True)
        else:
            key, subkey = jax.random.split(key)
            idx_next = jax.random.categorical(subkey, logits, axis=-1)[:, None]

        idx = jnp.concatenate([idx, idx_next], axis=1)
    return idx
```

Diferencias notables:

- JAX es **funcional con RNG explicito**. No hay un estado global de aleatoriedad. Cada llamada que necesita aleatoriedad recibe una `key` derivada del split. Eso parece pesado pero es lo que permite reproducibilidad bit-a-bit, fork-join determinista y jit sin problemas.
- `jax.random.categorical` toma logits directamente y la dimension a samplear via `axis`. Mas parecido a `tf.random.categorical` que a `torch.multinomial`.
- `jax.lax.top_k` es la version baja que se compila bien dentro de `jit`. La interfaz mas amigable es `jnp.argpartition` pero tiende a ser mas lenta sin jit.

---

## 5. Comparacion lado a lado de los tres frameworks

Para que el contraste quede claro, esta tabla resume como cada framework expresa los conceptos clave del decoder causal:

| Concepto              | PyTorch                                              | TensorFlow / Keras                                        | JAX + Flax                                                |
|-----------------------|------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| Definicion de modulo  | `class MiniGPT(nn.Module)` con `__init__` + `forward` | `class MiniGPT(tf.keras.Model)` con `__init__` + `call`     | `class MiniGPT(nn.Module)` con `@nn.compact __call__`        |
| Causal mask           | `torch.triu(..., diagonal=1)` + `masked_fill_(-inf)`  | flag `use_causal_mask=True` en `MultiHeadAttention`         | `nn.make_causal_mask(idx)` pasado al modulo de atencion     |
| Multi-head attention  | A mano con `W_qkv` + `softmax` + matmul              | Capa `tf.keras.layers.MultiHeadAttention`                   | `nn.MultiHeadDotProductAttention(num_heads=...)`            |
| Pre-LN                | `x = x + attn(ln1(x))`                                | `x = x + attn(ln1(x), use_causal_mask=True)`                | `x = x + attn(ln1(x), mask=causal_mask)`                    |
| Tied embedding head   | `logits = h @ self.token_emb.weight.T`                | `tf.matmul(h, self.token_emb.embeddings, transpose_b=True)` | `logits = h @ token_emb.embedding.T`                        |
| Estado del modelo     | Mutable en `nn.Module` (pesos viven dentro)          | Mutable en `tf.keras.Model`                                 | **Inmutable**: `params` separado del `model`                |
| Backward / gradientes | `loss.backward()` (automatico via autograd)          | `tape.gradient(loss, vars)` con `tf.GradientTape`           | `jax.grad(loss_fn)(params)` puramente funcional             |
| Optimizer step        | `optimizer.step()`                                    | `optimizer.apply_gradients(zip(grads, vars))`               | `state = state.apply_gradients(grads=grads)`                |
| Sampling de logits    | `torch.multinomial(softmax(logits))`                  | `tf.random.categorical(logits)` (toma logits directos)      | `jax.random.categorical(key, logits)` (con RNG explicito)    |
| Aleatoriedad          | Global (`torch.manual_seed`)                          | Global (`tf.random.set_seed`)                                | **Explicita**: `key` se splittea y se pasa                  |
| Compilacion           | `torch.compile(model)` (PyTorch 2.x, opcional)        | `@tf.function` (recomendado)                                | `@jax.jit` (esencial para performance)                      |
| KV-caching            | Manual: guardar `k_cache`, `v_cache` por capa         | Manual o usar `tf.keras.layers.CachedMultiHeadAttention`    | `decode=True` + variables `"cache"` en `apply`              |

Si tu objetivo es **aprender** los conceptos, PyTorch es lo mas pedagogico. Si tu objetivo es **deploy a produccion** con servir-modelos-en-TPU o quieres composicion funcional fuerte, JAX gana. Si vives en un stack con TFX, TF Serving o Vertex AI, TensorFlow tiene el ecosistema mas completo. Los tres entrenan el mismo modelo con la misma loss y, salvo diferencias de seed, llegan a la misma perplexity.

---

## 6. Demo: sampling con distintas temperaturas

Entrenamos el modelo en PyTorch durante 200 epochs (o ~2000 steps con batch 32) sobre el toy corpus y generamos con cuatro temperaturas distintas a partir del mismo prompt `"La inteligencia"`:

| Temperatura | Comportamiento esperado                  | Output tipico (ejemplo)                                          |
|-------------|-------------------------------------------|------------------------------------------------------------------|
| 0.0         | Argmax, determinista, tiende a loops      | `La inteligencia es la la la la la la la la la la la...`         |
| 0.3         | Conservador, frases coherentes pero cortas | `La inteligencia es una de las mas grandes virtudes del hombre.` |
| 0.7         | Balanceado, variedad razonable             | `La inteligencia es el arte de conocer lo que verdaderamente queremos.` |
| 1.5         | Caotico, vocabulario raro                  | `La inteligencia caballos sueno duro brillante mientras pensamos volver atras junto.` |

(Los outputs son ilustrativos; el corpus de juguete no produce frases perfectas. El patron — determinismo y loops a $\tau = 0$, coherencia local a $\tau \approx 0.7$, ruido a $\tau$ alto — si se reproduce consistentemente.)

Tres lecciones de este experimento:

1. **Temperatura 0 no es lo que quieres**. La intuicion ingenua dice "argmax = la respuesta mas probable", pero el argmax repetido entra en loops porque una vez que el modelo predice "la" con alta probabilidad, el siguiente argmax sigue siendo "la" (o algo con la misma palabra). Necesitas algo de entropia para que el modelo se "destranque".

2. **El sweet spot esta entre 0.5 y 0.9** para texto generativo. Por debajo se vuelve robotico; por encima empieza a alucinar palabras raras. Casi todos los LLMs sirven con `temperature=0.7` por default.

3. **El top-k y el top-p son ortogonales a la temperatura**. Filtran las opciones malas antes de samplear, asi puedes subir un poco la temperatura sin riesgo de sacar tokens muy improbables.

---

## 7. Variantes modernas (mencion, sin implementacion)

El mini-GPT que acabas de construir es esencialmente **GPT-2 (2019) en miniatura**. Los LLMs modernos (LLaMA 3, Mistral, Claude, GPT-4) comparten la columna vertebral pero introdujeron una serie de mejoras incrementales que vale la pena nombrar:

| Tema                  | GPT-2 / este mini       | Variante moderna                          | Por que                                                       |
|-----------------------|--------------------------|--------------------------------------------|----------------------------------------------------------------|
| Embedding posicional  | Aprendido (tabla `pos_emb`) | **RoPE** (Rotary Positional Embeddings)   | Generaliza a longitudes mayores que la vista; mejor extrapolation |
| Normalizacion         | LayerNorm                 | **RMSNorm**                                | Mas barato (no centra por media); igualmente estable           |
| Activacion FFN        | GELU                      | **SwiGLU** ($\text{Swish}(xW_1) \odot xW_2$)| Mas expresivo; aporta 0.5-1% en perplexity                     |
| Multi-head attention  | MHA pura (Q,K,V independientes) | **GQA / MQA** (grouped/multi-query)   | Reduce drasticamente el tamano del KV-cache en inferencia       |
| Kernel de attention   | Softmax estandar O($T^2$ memoria) | **Flash Attention**                  | Reescribe el calculo de attention para evitar el array $T \times T$ intermedio: 2-4x mas rapido y mucho menos VRAM |
| Inferencia            | Re-procesar todo el prefijo en cada paso | **KV-caching**                     | Guardar $K, V$ por capa y re-usar; baja inferencia de O($T^2 N$) a O($T N$) |
| Training stability    | Pre-LN basico             | **Pre-LN + QK-norm**                       | Normaliza Q y K antes del producto interno; previene saturacion del softmax |
| Atencion eficiente    | Cuadratica en $T$         | **Sliding window**, **MoE attention**, **State Space (Mamba)** | Lineal o sub-cuadratica para contextos largos |

El detalle conceptual de cada una esta cubierto en el [fundamento gpt-family](/fundamentos/gpt-family), donde mapeamos cada modelo de la familia GPT (1, 2, 3, 3.5, 4) a las decisiones arquitectonicas que lo definen. La idea importante es: **ninguna de estas mejoras cambia la naturaleza autoregresiva del modelo**. Todas son optimizaciones sobre la misma arquitectura decoder-only con causal mask. Si entendiste este mini-GPT, entendiste la columna vertebral; las variantes modernas son detalles de ingenieria que se montan encima.

### 7.1 KV-caching en una linea

Como inferencia con KV-cache es la diferencia entre 100 ms y 30 segundos por respuesta, vale la pena entenderla aunque no la implementemos completa. La idea: cuando generas el token $T+1$, los $K$ y $V$ de los tokens $1, \dots, T$ ya se calcularon en pasos anteriores y **no cambian**. En lugar de re-calcularlos cada vez, los guardas en un cache por capa y solo calculas el $K, V$ nuevo del token recien generado.

En Flax esto se activa con `nn.MultiHeadDotProductAttention(decode=True)` y una variable especial `"cache"` en la llamada `apply`. En PyTorch hay que armarlo a mano: cada capa de attention recibe `kv_cache` opcional con `k_cache, v_cache` y los concatena con el nuevo $K, V$.

---

## 8. Limitaciones del mini

Antes de pasar al siguiente camino, vale la pena ser honestos sobre lo que este modelo **no** puede hacer:

- **Dos capas son insuficientes para coherencia de varios parrafos**. Vas a ver oraciones razonables y frases cortas con sintaxis correcta, pero el modelo no mantiene un tema a lo largo de un parrafo. GPT-2 small (12 capas) es el minimo para coherencia parrafo a parrafo; GPT-3 (96 capas) lo es para coherencia entre parrafos.

- **Vocab 1000 limita severamente el lexico**. Con BPE de 1000 tokens, palabras raras se parten en subwords cortos y el modelo tiene que "componer" la palabra desde piezas. Para texto en espanol con vocabulario tecnico (medico, legal, cientifico) querras minimo 16 K-32 K tokens.

- **Context window de 64 tokens es minusculo**. Eso es ~10-15 palabras de espanol. Cualquier dependencia mas larga que eso es invisible para el modelo. GPT-2 small tenia 1024; GPT-4 Turbo tiene 128 K; Claude 3 Opus 200 K.

- **No hay KV-caching eficiente**. Cada token generado re-procesa toda la ventana. En un modelo grande esto es prohibitivo en latencia.

- **No hay checkpoint de pre-entrenamiento real**. Estamos pre-entrenando desde scratch en un corpus de 250 KB. Para tareas reales nunca harias esto: cargarias un modelo base ya pre-entrenado (GPT-2, Llama, etc.) y harias fine-tuning. Ese es exactamente el siguiente paso obvio, y es lo que cubre el [Camino 04: Fine-tuning BETO](/clases/clase-20/practica/04-fine-tuning-beto): partir de un encoder ya pre-entrenado y adaptarlo a una tarea clinica.

- **No hay alineacion (RLHF / DPO)**. El modelo predice texto plausible segun el corpus de entrenamiento; no entiende instrucciones, no rechaza pedidos malos, no sigue formatos. Para eso esta el [Camino 05: RLHF toy](/clases/clase-20/practica/05-rlhf-toy).

El mini-GPT es **completo conceptualmente** y **insuficiente practicamente**. Esa es exactamente la posicion pedagogica que queremos: ves la arquitectura entera, sabes que la escala (parametros + datos + computo) es lo que la hace util, y ahora puedes leer cualquier paper de LLM moderno y mapearlo a esta base.

{{< concept-alert type="recordar" >}}
La arquitectura del mini-GPT que acabas de construir es **identica en estructura** a GPT-2, GPT-3 y la familia LLaMA. Las diferencias son: numero de capas (2 vs 12 vs 96 vs 32+), dimension (64 vs 768 vs 12288 vs 4096+), variantes modernas de embedding/norma/activacion, escala de datos (1 MB vs 40 GB vs 570 GB vs 15 T tokens) y costo de entrenamiento ($0.01 vs $1k vs $5M vs $100M+). Pero el algoritmo es el mismo: token embedding + positional + N bloques Transformer con causal mask + LN final + head a vocab + next-token cross-entropy + AdamW + sampling autoregresivo.
{{< /concept-alert >}}

---

## 9. Como seguir

Si terminaste este camino, el flujo natural es:

1. **Re-entrenar con un corpus mas grande**: descarga Don Quijote completo (~2 MB plano), entrena 5000 steps en lugar de 2000, sube `n_layers` a 4 y mira como mejora la coherencia.

2. **Cargar un modelo pre-entrenado de Hugging Face**: en lugar de entrenar desde scratch, descarga GPT-2 small (`gpt2` en espanol o `flax-community/gpt-2-spanish`) y haz forward + generacion. Comparalo con tu mini para apreciar la diferencia que hace la escala.

3. **Implementar RoPE y RMSNorm**: en el mismo codigo base, reemplaza `pos_emb` por una funcion que rota Q y K segun la posicion, y `LayerNorm` por `RMSNorm`. Vas a tener un mini-LLaMA. Aprox 50 lineas de codigo.

4. **Pasar a fine-tuning**: pasa al [Camino 04](/clases/clase-20/practica/04-fine-tuning-beto) donde tomamos BETO (un BERT en espanol) y lo adaptamos a clasificacion de informes radiologicos. La estructura es identica al MLM del Camino 02 pero con cabeza de clasificacion en lugar de cabeza de vocab.

5. **Cerrar con alineacion**: el [Camino 05](/clases/clase-20/practica/05-rlhf-toy) toma un decoder ya pre-entrenado y le aplica el stack SFT + Reward Model + PPO para convertirlo en un asistente. Ese es el ultimo eslabon del viaje de la Clase 20.

---

## 10. Cross-links

- [Camino 01 - ELMo mini](/clases/clase-20/practica/01-elmo-mini): char-CNN + BiLSTM como antecesor inmediato del Transformer encoder.
- [Camino 02 - MLM encoder mini](/clases/clase-20/practica/02-mlm-encoder-mini): mismo Transformer pero bidireccional con masked LM (BERT-style).
- [Camino 04 - Fine-tuning BETO clinico](/clases/clase-20/practica/04-fine-tuning-beto): aplicacion real de encoder pre-entrenado a tarea clinica.
- [Camino 05 - RLHF toy](/clases/clase-20/practica/05-rlhf-toy): SFT + Reward Model + PPO sobre un decoder base.
- [Clase 14 - Escalon 08: Mini-GPT](/clases/clase-14/practica/08-mini-gpt): version solo PyTorch a nivel caracter (Shakespeare) — la base pedagogica de este camino.
- [Paper GPT-1 (Radford 2018)](/papers/gpt-1-radford-2018): el paper que introdujo el decoder-only Transformer pre-entrenado.
- [Paper GPT-2 (Radford 2019)](/papers/gpt-2-radford-2019): scaling-up del decoder + zero-shot capabilities + el paper que popularizo la generacion autoregresiva.
- [Fundamento: GPT family](/fundamentos/gpt-family): mapeo decision-por-decision de GPT-1 a GPT-4 (RoPE, RMSNorm, SwiGLU, GQA, Flash Attention, KV-cache, MoE).

---

**Ver tambien:** [Hub de practica - Clase 20](/clases/clase-20/practica) · [Teoria - Clase 20](/clases/clase-20/teoria) · [Profundizacion - Clase 20](/clases/clase-20/profundizacion).
