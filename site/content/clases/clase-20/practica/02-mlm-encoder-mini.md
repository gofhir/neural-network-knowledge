---
title: "02 - MLM encoder mini"
weight: 32
math: true
---

En el camino 01 construimos ELMo: un biLM con dos LSTM (forward y backward) concatenadas. La idea de "embedding contextual" estaba clara, pero el truco de concatenar dos modelos unidireccionales tenia un costo. Cada LSTM solo veia la mitad del contexto durante el entrenamiento. La concatenacion final es bidireccional, pero el aprendizaje de cada direccion es estrictamente causal.

BERT (Devlin et al., 2018) propuso una alternativa elegante: si la attention puede ver todos los tokens al mismo tiempo, **¿para que entrenar dos modelos?** La respuesta fue **Masked Language Modeling (MLM)**: enmascarar tokens aleatorios del input y pedirle al modelo que los reconstruya usando contexto **a ambos lados**. Esto convirtio al encoder del Transformer en un entrenador bidireccional natural.

En este capitulo implementamos el nucleo de BERT en miniatura — un **encoder Transformer + cabeza MLM** — pero lo hacemos en **tres frameworks**: PyTorch, TensorFlow/Keras y JAX/Flax. La idea no es comparar resultados sino **comparar idiomas**: ver las mismas matematicas escritas en tres dialectos distintos. Cuando termines, vas a poder leer codigo de research papers sin importar el framework en el que esten.

---

## 1. El nucleo de BERT en mini

Antes de tocar codigo, fijemos la arquitectura. Vamos a construir un encoder muy chico pero estructuralmente identico a BERT:

| Hiperparametro | Valor mini | BERT-base |
|---|---:|---:|
| Numero de capas | 2 | 12 |
| Dimension del modelo $d_{model}$ | 64 | 768 |
| Numero de cabezas $h$ | 4 | 12 |
| Dimension FFN $d_{ff}$ | 256 | 3072 |
| Vocab size | 1000 | 30522 |
| Longitud de secuencia | 32 | 512 |
| Parametros aproximados | $\sim$ 200 K | 110 M |

Tres componentes principales:

1. **Embedding layer**: token embedding + positional embedding (aprendido, no sinusoidal — es lo que hace BERT real).
2. **Encoder stack**: dos bloques Transformer en pre-norm, bidireccionales (sin causal mask).
3. **MLM head**: una linear $d_{model} \to V$ que predice el token original en cada posicion enmascarada.

### 1.1 El objetivo MLM

La perdida MLM es simple en concepto. Dado un input $x = (x_1, \ldots, x_T)$, elegimos un subconjunto $\mathcal{M} \subset \{1, \ldots, T\}$ de posiciones a enmascarar (tipicamente 15 %). Construimos $\tilde{x}$ donde los tokens en $\mathcal{M}$ son reemplazados segun la regla 80/10/10 (la veremos en detalle abajo). El modelo recibe $\tilde{x}$ y produce logits $z \in \mathbb{R}^{T \times V}$. La perdida es cross-entropy **solo sobre las posiciones enmascaradas**:

$$
\mathcal{L}_{\text{MLM}} = - \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p(x_i \mid \tilde{x}) = - \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log \text{softmax}(z_i)_{x_i}
$$

Las posiciones no enmascaradas **no contribuyen al loss**. Esto es importante: el modelo no esta haciendo language modeling autoregresivo. Esta haciendo "fill in the blanks" condicionado en todo lo demas.

### 1.2 La regla 80/10/10

El paper original de BERT no reemplaza los tokens elegidos siempre por `[MASK]`. Hace algo mas sutil:

- **80 %** de las posiciones elegidas: reemplazo por el token `[MASK]`.
- **10 %**: reemplazo por un token aleatorio del vocabulario.
- **10 %**: dejar el token original sin cambio.

¿Por que? El token `[MASK]` no aparece en fine-tuning ni en inferencia. Si el modelo aprendiera a "esperar `[MASK]`" para producir output, el fine-tuning seria una catastrofe. Mezclar reemplazos aleatorios y dejar tokens originales fuerza al modelo a producir representaciones utiles para **todo token**, no solo para los enmascarados. Roberta (Liu et al., 2019) confirmo empiricamente que esta regla aporta — aunque NSP no.

### 1.3 La attention bidireccional

La attention multi-cabeza es la misma que vimos en clase 14. La unica diferencia con un decoder causal es **que no aplicamos causal mask**. Cada token puede mirar a todos los demas tokens del input — pasado y futuro. La formula es:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

sin la mascara triangular superior que pondria $-\infty$ en las posiciones futuras. Esa es **la unica diferencia matematica** entre un encoder y un decoder. Todo lo demas — FFN, residuales, LayerNorm, pre-norm — es identico.

---

## 2. Seccion 1: PyTorch

PyTorch es el framework dominante en research. Su filosofia es **define-by-run**: cada `forward` construye el grafo computacional sobre la marcha. La sintaxis se siente como NumPy con autograd.

### 2.1 Imports y configuracion

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Hiperparametros del mini-encoder
VOCAB_SIZE = 1000
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
SEQ_LEN = 32
DROPOUT = 0.1

# Tokens especiales (convencion BERT)
PAD_ID = 0
MASK_ID = 1  # el [MASK] token
CLS_ID = 2
SEP_ID = 3
```

Los IDs reservados al principio del vocabulario son convencion BERT. El resto del vocab (IDs 4 a 999) son tokens "reales" de BPE.

### 2.2 Multi-head self-attention (sin mascara causal)

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, T, d_model)
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # (B, n_heads, T, d_k)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (B, n_heads, T, T)

        if attn_mask is not None:
            # attn_mask: (B, 1, 1, T) con True en posiciones de padding
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V  # (B, n_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)
```

Notese que `attn_mask` aqui es **solo para padding**, no es causal. Si tu input tiene 10 tokens reales y 22 de padding, no quieres que la attention preste atencion al padding. Pero los 10 tokens reales si pueden verse entre si en cualquier direccion.

### 2.3 Bloque encoder

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), attn_mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
```

Usamos **GELU** (Gaussian Error Linear Unit) en lugar de ReLU, que es el estandar en BERT y todos los Transformers modernos. La idea es la misma: una no-linealidad entre las dos linears. GELU es mas suave que ReLU y empiricamente da resultados ligeramente mejores.

### 2.4 MLM encoder completo

```python
class MLMEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # Cabeza MLM: linear hacia vocab
        # Convencion BERT: tied weights con token_emb (las saltamos por simplicidad)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # input_ids: (B, T)
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # Padding mask: (B, 1, 1, T)
        pad_mask = (input_ids == PAD_ID).unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            x = layer(x, attn_mask=pad_mask)

        x = self.ln_final(x)
        logits = self.mlm_head(x)  # (B, T, V)
        return logits
```

Tres detalles que vale la pena resaltar:

- **`padding_idx=PAD_ID`** en `nn.Embedding`: hace que el embedding del token 0 sea constante a cero y no se actualice por gradiente. Es la convencion para tokens de padding.
- **Positional embedding aprendido**: usamos `nn.Embedding(seq_len, d_model)` en lugar de las sinusoides de Vaswani. Esto es lo que hace BERT real. Para secuencias cortas funciona perfectamente; el problema viene si quieres generalizar a longitudes mayores que las vistas en entrenamiento (BERT no puede).
- **No hay tied weights**: en BERT real, la matriz de la cabeza MLM comparte pesos con la matriz del token embedding (output proj = embedding transpuesto). Lo saltamos para mantener el codigo claro.

### 2.5 Funcion de masking 80/10/10

```python
def apply_mlm_mask(input_ids, mask_token_id, vocab_size, mlm_prob=0.15,
                   pad_id=PAD_ID):
    """
    Aplica la regla 80/10/10 de BERT.
    Devuelve (masked_input_ids, labels) donde labels tiene -100 en posiciones
    no enmascaradas (que ignore_index del cross_entropy va a saltar).
    """
    labels = input_ids.clone()

    # Generar mascara probabilistica
    probs = torch.full(input_ids.shape, mlm_prob, device=input_ids.device)
    # No enmascarar padding ni tokens especiales (PAD, CLS, SEP)
    special_mask = (input_ids == PAD_ID) | (input_ids == CLS_ID) | (input_ids == SEP_ID)
    probs.masked_fill_(special_mask, 0.0)

    masked_indices = torch.bernoulli(probs).bool()
    labels[~masked_indices] = -100  # ignore_index

    masked_input = input_ids.clone()

    # 80%: reemplazar por [MASK]
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8,
                                              device=input_ids.device)).bool() & masked_indices
    masked_input[replace_mask] = mask_token_id

    # 10%: reemplazar por token aleatorio (50% de los restantes 20%)
    random_mask = torch.bernoulli(torch.full(input_ids.shape, 0.5,
                                             device=input_ids.device)).bool() & masked_indices & ~replace_mask
    random_tokens = torch.randint(0, vocab_size, input_ids.shape,
                                  device=input_ids.device)
    masked_input[random_mask] = random_tokens[random_mask]

    # 10% restante: dejar el token original (no hace falta accion)
    return masked_input, labels
```

El `ignore_index = -100` es la convencion de PyTorch: `F.cross_entropy(..., ignore_index=-100)` salta las posiciones marcadas con $-100$ al calcular el loss. Asi, la perdida solo se calcula sobre las posiciones enmascaradas, como pide la definicion de MLM.

### 2.6 Loop de entrenamiento mini

```python
# Toy corpus: 64 secuencias de longitud SEQ_LEN con tokens random
torch.manual_seed(0)
toy_corpus = torch.randint(4, VOCAB_SIZE, (64, SEQ_LEN))
# Forzamos CLS al inicio y SEP al final
toy_corpus[:, 0] = CLS_ID
toy_corpus[:, -1] = SEP_ID

model = MLMEncoder(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF,
                   SEQ_LEN, DROPOUT)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
for step in range(200):
    # Muestrear un batch
    idx = torch.randint(0, toy_corpus.size(0), (16,))
    batch = toy_corpus[idx]

    masked_input, labels = apply_mlm_mask(batch, MASK_ID, VOCAB_SIZE)
    logits = model(masked_input)  # (B, T, V)

    loss = F.cross_entropy(
        logits.view(-1, VOCAB_SIZE),
        labels.view(-1),
        ignore_index=-100,
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 20 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}")
```

Sobre un corpus random de 64 secuencias, el loss MLM baja de $\sim \ln(1000) \approx 6.9$ (perdida de un modelo uniforme) hacia $\sim 4$ en 200 pasos. Es overfitting al corpus de juguete — que es exactamente lo que queremos ver para validar que el codigo aprende. En un corpus real con billones de tokens, el loss baja hacia $\sim 1.5$ y se queda ahi.

### 2.7 Evaluacion: predecir un token enmascarado

```python
model.eval()
with torch.no_grad():
    test_seq = toy_corpus[0].clone()
    # Enmascarar la posicion 15
    target_pos = 15
    target_token = test_seq[target_pos].item()
    test_seq[target_pos] = MASK_ID

    logits = model(test_seq.unsqueeze(0))  # (1, T, V)
    probs = F.softmax(logits[0, target_pos], dim=-1)
    top5 = torch.topk(probs, 5)

    print(f"Token original: {target_token}")
    print(f"Top-5 predicciones: {top5.indices.tolist()}")
    print(f"Top-5 probabilidades: {top5.values.tolist()}")
```

Para corpus de juguete, esperamos ver el token original en el top-5. Para corpus real, esto se mide con metricas como **top-k accuracy** sobre un dev set retenido.

---

## 3. Seccion 2: TensorFlow / Keras

TensorFlow 2.x con Keras 3 ofrece la misma flexibilidad que PyTorch pero con un estilo distinto. Las capas se componen via `tf.keras.Model` subclassing, las operaciones se trazan automaticamente con `@tf.function` cuando hace falta optimizar, y la inferencia es first-class para deployment (TF Serving, TFLite, TF.js).

### 3.1 Imports y configuracion

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

VOCAB_SIZE = 1000
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
SEQ_LEN = 32
DROPOUT = 0.1

PAD_ID = 0
MASK_ID = 1
CLS_ID = 2
SEP_ID = 3
```

### 3.2 Bloque encoder con Keras

Keras tiene `layers.MultiHeadAttention` builtin, asi que no necesitamos reimplementarla:

```python
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=dropout,
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, attention_mask=None, training=False):
        # Pre-norm: normalizar antes del sub-bloque
        normed = self.ln1(x)
        attn_out = self.attn(
            query=normed, key=normed, value=normed,
            attention_mask=attention_mask,
            training=training,
        )
        x = x + self.dropout1(attn_out, training=training)

        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout2(ffn_out, training=training)
        return x
```

Notar que `layers.MultiHeadAttention` espera **una mascara de atencion en formato `attention_mask`**, no como en PyTorch. La convencion de Keras: `1` significa "atender", `0` significa "ignorar". Es la convencion **invertida** respecto a PyTorch.

### 3.3 MLM encoder completo

```python
class MLMEncoder(keras.Model):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 seq_len, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(vocab_size, d_model, mask_zero=False)
        self.pos_emb = layers.Embedding(seq_len, d_model)
        self.dropout = layers.Dropout(dropout)

        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        self.ln_final = layers.LayerNormalization(epsilon=1e-6)
        self.mlm_head = layers.Dense(vocab_size)

        self.seq_len = seq_len

    def call(self, input_ids, training=False):
        # input_ids: (B, T)
        positions = tf.range(self.seq_len)[tf.newaxis, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x, training=training)

        # Padding mask: (B, T) -> 1 donde NO es padding
        attention_mask = tf.cast(input_ids != PAD_ID, tf.int32)
        # Keras espera mascara 2D que ella misma broadcast a (B, T, T)

        for layer in self.encoder_layers:
            x = layer(x, attention_mask=attention_mask, training=training)

        x = self.ln_final(x)
        return self.mlm_head(x)  # (B, T, V)
```

Diferencias respecto a PyTorch:

- **`training`** es un argumento explicito que se propaga por todas las capas. Es necesario para que dropout y batchnorm sepan en que modo estan.
- **`mask_zero=False`** en el embedding: Keras tiene un mecanismo de "mascaras automaticas" que propaga mascaras por las capas, pero aqui lo manejamos manualmente para tener control sobre el padding mask.
- **`call`** es el equivalente de `forward` en PyTorch. La convencion `__call__` -> `call` permite a Keras inyectar logica adicional (tracking de pesos, mascaras automaticas).

### 3.4 MLM masking en TensorFlow

```python
def apply_mlm_mask(input_ids, mask_token_id, vocab_size, mlm_prob=0.15):
    """Version TF de apply_mlm_mask. input_ids: (B, T) int32."""
    input_ids = tf.cast(input_ids, tf.int32)
    shape = tf.shape(input_ids)

    # No enmascarar padding ni especiales
    special_mask = (
        (input_ids == PAD_ID) |
        (input_ids == CLS_ID) |
        (input_ids == SEP_ID)
    )

    probs = tf.where(special_mask, 0.0, mlm_prob)
    mask_indicator = tf.random.uniform(shape) < probs  # (B, T) bool

    # labels: -100 donde no enmascaramos
    labels = tf.where(mask_indicator, input_ids, tf.fill(shape, -100))

    masked_input = input_ids

    # 80% -> [MASK]
    replace_with_mask = (tf.random.uniform(shape) < 0.8) & mask_indicator
    masked_input = tf.where(replace_with_mask, mask_token_id, masked_input)

    # 10% -> random (50% de los que no fueron MASK)
    replace_with_random = (
        (tf.random.uniform(shape) < 0.5) &
        mask_indicator &
        ~replace_with_mask
    )
    random_tokens = tf.random.uniform(shape, minval=4, maxval=vocab_size,
                                      dtype=tf.int32)
    masked_input = tf.where(replace_with_random, random_tokens, masked_input)

    return masked_input, labels
```

`tf.where(cond, a, b)` es el equivalente a `np.where`: elige $a$ donde `cond` es True y $b$ donde es False. Es el patron idiomatico para "modificar tensores condicionalmente" en TF, porque no puedes hacer indexing in-place como en PyTorch.

### 3.5 Loop de entrenamiento con `tf.GradientTape`

```python
# Toy corpus
tf.random.set_seed(0)
toy_corpus = tf.random.uniform((64, SEQ_LEN), minval=4, maxval=VOCAB_SIZE,
                                dtype=tf.int32)
# Forzar CLS y SEP
cls_col = tf.fill((64, 1), CLS_ID)
sep_col = tf.fill((64, 1), SEP_ID)
toy_corpus = tf.concat([cls_col, toy_corpus[:, 1:-1], sep_col], axis=1)

model = MLMEncoder(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF,
                   SEQ_LEN, DROPOUT)
optimizer = keras.optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0)

# Definimos el loss con ignore_class=-100
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    ignore_class=-100,
    reduction="sum_over_batch_size",
)

@tf.function
def train_step(batch):
    masked_input, labels = apply_mlm_mask(batch, MASK_ID, VOCAB_SIZE)
    with tf.GradientTape() as tape:
        logits = model(masked_input, training=True)  # (B, T, V)
        loss = loss_fn(labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for step in range(200):
    idx = tf.random.uniform((16,), 0, 64, dtype=tf.int32)
    batch = tf.gather(toy_corpus, idx)
    loss = train_step(batch)
    if step % 20 == 0:
        print(f"step {step:3d}  loss={loss.numpy():.4f}")
```

Tres detalles del estilo TF:

- **`@tf.function`** traza el grafo computacional la primera vez que se llama y reutiliza esa traza despues. Es el equivalente a `torch.compile` o `jax.jit`. Acelera mucho los pasos siguientes.
- **`tf.GradientTape`** es el equivalente del `loss.backward()` de PyTorch, pero explicito: solo se trackean las operaciones dentro del `with`. Permite tape diferenciacion selectiva.
- **`ignore_class=-100`** en `SparseCategoricalCrossentropy` es el equivalente de `ignore_index=-100` en PyTorch. Funciona identico.

Alternativamente, podrias usar `model.compile()` + `model.fit()` pasando un dataset que ya aplique el masking. Pero para entender que pasa por dentro, el `tf.GradientTape` manual es mejor pedagogicamente.

---

## 4. Seccion 3: JAX + Flax

JAX adopta una filosofia muy distinta: **funciones puras + transformaciones**. No hay objetos con estado interno mutable. Los modelos son funciones que toman parametros explicitos como argumento y devuelven outputs. Esto se siente raro al principio si vienes de PyTorch, pero permite optimizaciones agresivas: `jax.jit` compila a XLA, `jax.grad` da gradientes automaticos, `jax.vmap` vectoriza sobre batch, `jax.pmap` paraleliza sobre dispositivos.

**Flax** es la libreria que pone una API tipo Keras encima de JAX. Define modelos como `nn.Module`, pero los parametros se inicializan externamente y se pasan explicitamente a cada llamada.

### 4.1 Imports

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Hiperparametros (mismos que antes)
VOCAB_SIZE = 1000
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
SEQ_LEN = 32
DROPOUT = 0.1

PAD_ID, MASK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3
```

### 4.2 Bloque encoder en Flax

```python
class TransformerEncoderLayer(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, attention_mask=None, deterministic=False):
        # Pre-norm + self-attention
        normed = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(normed, normed, mask=attention_mask)
        x = x + nn.Dropout(self.dropout, deterministic=deterministic)(attn_out)

        # Pre-norm + FFN
        normed = nn.LayerNorm()(x)
        h = nn.Dense(self.d_ff)(normed)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        x = x + nn.Dropout(self.dropout, deterministic=deterministic)(h)
        return x
```

Diferencias importantes con PyTorch/TF:

- **Decorador `@nn.compact`** permite definir sub-modulos dentro de `__call__` (en lugar de tener que declararlos en `setup`). Es opcional pero compacto.
- **Atributos de clase** (`d_model: int`, etc.) son los hiperparametros del modulo. Flax los trata como dataclass fields.
- **`deterministic`** es el equivalente al `training=False` de Keras. Cuando es True, dropout se desactiva. La razon de llamarlo asi en lugar de "training" es que JAX requiere reproducibilidad y "deterministic" describe mejor el comportamiento.

### 4.3 MLM encoder completo

```python
class MLMEncoder(nn.Module):
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    seq_len: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, input_ids, deterministic=False):
        # input_ids: (B, T) int32
        positions = jnp.arange(self.seq_len)[None, :]

        tok_emb = nn.Embed(self.vocab_size, self.d_model)(input_ids)
        pos_emb = nn.Embed(self.seq_len, self.d_model)(positions)
        x = tok_emb + pos_emb
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)

        # Padding mask: (B, 1, 1, T) con True donde SI atender
        pad_mask = (input_ids != PAD_ID)[:, None, None, :]

        for _ in range(self.n_layers):
            x = TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
            )(x, attention_mask=pad_mask, deterministic=deterministic)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits
```

La convencion de mascara en Flax sigue la de TF: `True` donde si atender, `False` donde ignorar. La attention internamente convierte a aditivos $-\infty$.

### 4.4 Inicializacion y manejo de estado

Aqui esta el cambio mas grande respecto a PyTorch. En JAX **no hay `model.parameters()`** porque el modelo no tiene estado interno. Los parametros se calculan llamando a `model.init(rng, dummy_input)`:

```python
model = MLMEncoder(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, SEQ_LEN, DROPOUT)

key = jax.random.PRNGKey(42)
init_key, dropout_key = jax.random.split(key)

dummy_input = jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)
params = model.init(
    {"params": init_key, "dropout": dropout_key},
    dummy_input,
    deterministic=True,
)["params"]

print(f"Numero de parametros: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")
```

`params` es un **PyTree** anidado: un dict de dicts de arrays. Cada clave corresponde a un sub-modulo. Esa estructura es lo que JAX recorre con sus transformaciones (`jax.grad`, `jax.tree_util.tree_map`, etc.).

### 4.5 MLM masking JIT-eable

```python
def apply_mlm_mask(rng, input_ids, mask_token_id, vocab_size, mlm_prob=0.15):
    """Version JAX. rng es un PRNGKey. input_ids: (B, T) int32."""
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    special_mask = (
        (input_ids == PAD_ID) |
        (input_ids == CLS_ID) |
        (input_ids == SEP_ID)
    )

    # Probabilidad de enmascarar por posicion
    probs = jnp.where(special_mask, 0.0, mlm_prob)
    mask_indicator = jax.random.uniform(k1, input_ids.shape) < probs

    labels = jnp.where(mask_indicator, input_ids, -100)

    # 80% -> [MASK]
    replace_with_mask = (jax.random.uniform(k2, input_ids.shape) < 0.8) & mask_indicator
    masked_input = jnp.where(replace_with_mask, mask_token_id, input_ids)

    # 10% -> random
    replace_with_random = (
        (jax.random.uniform(k3, input_ids.shape) < 0.5) &
        mask_indicator &
        ~replace_with_mask
    )
    random_tokens = jax.random.randint(k4, input_ids.shape, 4, vocab_size)
    masked_input = jnp.where(replace_with_random, random_tokens, masked_input)

    return masked_input, labels
```

Notar el manejo explicito de PRNGKey: cada operacion aleatoria recibe una key derivada con `jax.random.split`. JAX no tiene estado global de RNG — todas las aleatoreidades son determministicas dado un key. Eso permite que las funciones sean **puras** y JIT-eables.

### 4.6 Loop de entrenamiento con `optax`

```python
# Toy corpus
data_key = jax.random.PRNGKey(0)
toy_corpus = jax.random.randint(data_key, (64, SEQ_LEN), 4, VOCAB_SIZE)
# Forzar CLS y SEP
toy_corpus = toy_corpus.at[:, 0].set(CLS_ID).at[:, -1].set(SEP_ID)

# Optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=1e-3),
)
opt_state = optimizer.init(params)


def loss_fn(params, masked_input, labels, dropout_key):
    logits = model.apply(
        {"params": params},
        masked_input,
        deterministic=False,
        rngs={"dropout": dropout_key},
    )
    # Cross entropy manual con label mask
    one_hot = jax.nn.one_hot(jnp.maximum(labels, 0), VOCAB_SIZE)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    per_token = -jnp.sum(one_hot * log_probs, axis=-1)  # (B, T)
    label_mask = (labels != -100).astype(jnp.float32)
    loss = jnp.sum(per_token * label_mask) / jnp.maximum(jnp.sum(label_mask), 1.0)
    return loss


@jax.jit
def train_step(params, opt_state, batch, rng):
    mask_key, dropout_key = jax.random.split(rng)
    masked_input, labels = apply_mlm_mask(mask_key, batch, MASK_ID, VOCAB_SIZE)
    loss, grads = jax.value_and_grad(loss_fn)(params, masked_input, labels, dropout_key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


rng = jax.random.PRNGKey(1)
for step in range(200):
    rng, sample_key, step_key = jax.random.split(rng, 3)
    idx = jax.random.randint(sample_key, (16,), 0, 64)
    batch = toy_corpus[idx]
    params, opt_state, loss = train_step(params, opt_state, batch, step_key)
    if step % 20 == 0:
        print(f"step {step:3d}  loss={float(loss):.4f}")
```

Lo que tienes que internalizar de este patron JAX:

- **`model.apply`** es como `forward`, pero recibe los params como primer argumento. El modelo es una funcion pura.
- **`jax.value_and_grad`** transforma `loss_fn` en una funcion que devuelve $(loss, grads)$ a la vez. Es el corazon de la diferenciacion automatica.
- **`jax.jit`** compila la funcion completa a XLA. La primera llamada es lenta (compilacion); las siguientes son rapidisimas.
- **El estado del optimizador (`opt_state`)** se pasa explicitamente, igual que `params`. Nada es mutable.
- **Cross entropy manual** porque `optax.softmax_cross_entropy` no soporta nativamente `ignore_index`. Lo construimos a mano con un `label_mask` que es 0 donde no debemos sumar.

Esto se siente verboso al principio, pero a cambio obtienes pureza, paralelizacion gratuita, y compilacion automatica. JAX es el framework dominante en Google Research y DeepMind por eso.

---

## 5. Comparacion lado a lado

Las tres implementaciones son **isomorfas matematicamente**: misma arquitectura, mismos hiperparametros, mismo MLM, misma regla 80/10/10. Lo que cambia es el idioma.

| Concepto | PyTorch | TensorFlow/Keras | JAX + Flax |
|---|---|---|---|
| Definicion de modulo | `class M(nn.Module)` + `forward` | `class M(layers.Layer)` + `call` | `class M(nn.Module)` + `__call__` con `@nn.compact` |
| Estado interno | mutable, en `self.W = nn.Linear(...)` | mutable, en `self.dense = layers.Dense(...)` | inmutable, params externos como PyTree |
| Inicializacion | automatica al construir | automatica al primer `call` | explicita con `model.init(rng, dummy)` |
| Forward call | `model(x)` | `model(x, training=True)` | `model.apply(params, x, deterministic=False, rngs=...)` |
| Diferenciacion | `loss.backward()` | `tf.GradientTape` | `jax.grad` o `jax.value_and_grad` |
| Modo train/eval | `model.train()` / `model.eval()` | argumento `training=` por capa | argumento `deterministic=` por capa |
| RNG | global (`torch.manual_seed`) | global (`tf.random.set_seed`) | explicito (`jax.random.PRNGKey` + `split`) |
| Compilacion JIT | `torch.compile(model)` | `@tf.function` | `@jax.jit` |
| Optimizadores | `torch.optim` | `keras.optimizers` | `optax` |
| Convencion mascara attention | `True` donde **ignorar** | `1` donde **atender** | `True` donde **atender** |
| Loss con ignore index | `F.cross_entropy(..., ignore_index=-100)` | `SparseCategoricalCrossentropy(ignore_class=-100)` | manual con `label_mask` y reduce |
| Manejo de batch | natural via `nn.Module` | natural via `keras.Model` | mismo, pero `jax.vmap` lo hace explicito si lo necesitas |

### 5.1 Cual usar y cuando

- **PyTorch**: research, papers, prototipado rapido. Hugging Face `transformers`, `datasets`, `peft`, `trl` son nativos PyTorch (con bindings TF/JAX que suelen ir un paso atras). Si estas implementando una idea nueva y quieres ver gradientes, prints, debugger funcionando — PyTorch es el camino.
- **TensorFlow/Keras**: produccion. TF Serving para servir modelos en cluster, TFLite para mobile/edge, TF.js para browser, TFX para pipelines de ML. Si tu modelo va a vivir en infrastructure productiva fuera de Python — TF es el camino.
- **JAX + Flax**: escala masiva. TPUs, paralelismo de datos y de modelo, optimizaciones XLA automaticas. PaLM, Gemini, Stable Diffusion 3 estan escritos en JAX. Si estas escalando a billions of parameters en clusters grandes — JAX es el camino.

En la practica, **muchos labs hacen prototipado en PyTorch, escalan a JAX si necesitan TPUs, y exportan a TF para deployment** mediante ONNX o conversiones nativas. Saber leer los tres te abre todas las puertas.

---

## 6. Conexion con BERT real

Lo que acabamos de construir tiene la **misma estructura** que BERT-base. Las diferencias son:

| Atributo | Nuestro mini | BERT-base | BERT-large |
|---|---:|---:|---:|
| Capas | 2 | 12 | 24 |
| $d_{model}$ | 64 | 768 | 1024 |
| Cabezas | 4 | 12 | 16 |
| $d_{ff}$ | 256 | 3072 | 4096 |
| Vocab | 1000 | 30 522 | 30 522 |
| Seq len | 32 | 512 | 512 |
| Parametros | $\sim$ 200 K | 110 M | 340 M |
| Pretraining data | corpus toy | BookCorpus + Wikipedia (3.3 B tokens) | mismo |
| Pretraining compute | $\sim$ 1 minuto en CPU | 4 days en 16 TPUs | 4 days en 64 TPUs |
| Pretraining objective | MLM | MLM + NSP | MLM + NSP |

### 6.1 Por que no implementamos NSP

BERT original entrena con dos objetivos simultaneos: MLM y **Next Sentence Prediction** (predecir si dos oraciones son consecutivas en el corpus). RoBERTa (Liu et al., 2019) mostro empiricamente que **NSP no aporta**. Quitarlo y entrenar solo con MLM, con corpus mas grande y batches mayores, mejora downstream performance.

Por eso lo saltamos legitimamente. Nuestro mini-encoder es mas cercano a RoBERTa que a BERT en ese sentido. ALBERT, DistilBERT, DeBERTa siguieron el mismo camino: solo MLM (o variantes como SBO).

### 6.2 Usar BERT pre-entrenado en una linea

En la practica, casi nadie pre-entrena BERT desde cero. La idea es **cargar un checkpoint pre-entrenado** y fine-tunearlo a una tarea downstream:

```python
from transformers import BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
print(tokenizer.decode(predictions[0]))
```

Tres lineas. Eso es lo que hace 99 % del codigo de produccion que usa BERT hoy. Toda la sofisticacion de la libreria `transformers` esta en abstraer lo que acabamos de implementar para que solo tengas que pensar en tu downstream task.

Para espanol clinico (lo que verifica el camino 04), el equivalente es BETO (Canete et al., 2020), pre-entrenado en un corpus de espanol grande. La API es identica: `from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")` y listo.

---

## 7. Limitaciones del mini

Hay varias decisiones de simplificacion que hicimos. Vale la pena enumerarlas para que sepas que faltaria para llevarlo a produccion:

1. **Sin NSP**: aceptable. La literatura post-2019 muestra que NSP no aporta. Lo saltamos legitimamente.
2. **Positional embedding aprendido con seq_len mini**: BERT real usa la misma estrategia, pero con seq_len 512. Si extendieras nuestro modelo a secuencias mas largas, tendrias que reinicializar la matriz de pos_emb. Modelos modernos (LLaMA, RoPE, ALiBi) resuelven esto con encodings relativos.
3. **Sin warmup**: BERT real arranca con learning rate 0 y sube linealmente durante 10 000 pasos antes de empezar a bajar. Sin warmup, los primeros gradientes del attention son muy ruidosos y pueden destabilizar. En nuestro caso, con un corpus de juguete, no se nota.
4. **Sin weight decay diferenciado**: BERT real aplica weight decay solo a las matrices, no a los biases ni a LayerNorm. Lo saltamos.
5. **Sin tied weights**: la cabeza MLM en BERT real comparte pesos con el token embedding (es la transposicion). Eso reduce parametros y es matematicamente elegante. Lo saltamos por claridad.
6. **Corpus de juguete**: random tokens no tienen estructura semantica. El modelo "aprende" a memorizar, no a entender. Sirve para validar codigo, no para producir representaciones utiles.

Si quieres llevar este encoder a un dataset real (Wikipedia espanol por ejemplo), las siguientes piezas son: corpus tokenizado a BPE/WordPiece, dataloader que streaming desde disco, training loop multi-GPU con `accelerate` o `jax.pmap`, checkpointing y logging.

---

## 8. Pausa de verificacion

Antes de pasar al siguiente camino, asegurate de poder responder estas preguntas con tus propias palabras.

1. **¿Por que MLM permite entrenamiento bidireccional "honesto" y ELMo no?**
   ELMo concatena dos modelos causales (forward + backward). Cada uno solo ve una direccion durante el entrenamiento. MLM enmascara tokens en una secuencia bidireccional unica y pide reconstruirlos viendo **ambos lados al mismo tiempo**. Es una sola red, no dos.

2. **¿Por que la regla 80/10/10 en lugar de "siempre reemplazar por [MASK]"?**
   Porque `[MASK]` no existe en fine-tuning ni inferencia. Si el modelo solo aprendiera a producir output cuando ve `[MASK]`, downstream tasks fallarian. Mezclar reemplazos aleatorios y tokens originales fuerza al modelo a producir representaciones utiles **para todo token**.

3. **¿Cual es la unica diferencia matematica entre encoder y decoder Transformer?**
   La mascara causal en la self-attention. El encoder no tiene mascara causal — cada token mira a todos los demas. El decoder si la tiene — cada token solo mira a tokens anteriores. Todo lo demas es identico.

4. **¿Que cambia entre PyTorch, TF y JAX en el manejo de estado?**
   PyTorch: estado mutable dentro del `nn.Module`. TF/Keras: igual. JAX/Flax: **estado externo**. Los parametros se inicializan con `model.init` y se pasan explicitamente a cada `apply`. El modelo es una funcion pura. Esto habilita `jit`, `grad`, `vmap`, `pmap` automaticamente.

5. **¿Por que pre-norm en lugar de post-norm?**
   Pre-norm es mas estable: la LayerNorm normaliza el input al sub-bloque, pero la suma residual queda intacta. Esto permite entrenar a profundidades altas con learning rate constante. Post-norm (Vaswani 2017) requiere warmup obligatorio y se vuelve inestable mas alla de $\sim$ 24 capas.

6. **¿Cuanto parametros tendria un BERT-base implementado en cualquiera de los tres frameworks?**
   Los mismos 110 M. El framework no cambia el modelo. Solo cambia el codigo que lo define.

---

## 9. Codigo y siguiente camino

Los tres scripts completos viven en `clase_20/practica/02_mlm_encoder_pytorch.py`, `02_mlm_encoder_tf.py` y `02_mlm_encoder_jax.py`. Corren los tres en CPU en menos de 2 minutos cada uno.

Siguiente camino: [03 - Decoder causal mini](/clases/clase-20/practica/03-causal-decoder-mini) — la otra mitad del mundo. Reemplazamos MLM por causal LM y agregamos la mascara triangular a la attention. Ese cambio chico de implementacion es lo que separa BERT de GPT, y separa "modelo que entiende" de "modelo que genera". Como aqui, lo hacemos en los tres frameworks.

---

**Ver tambien:** [Camino 01 - ELMo mini](/clases/clase-20/practica/01-elmo-mini) · [Camino 03 - Decoder causal mini](/clases/clase-20/practica/03-causal-decoder-mini) · [Paper BERT (Devlin et al. 2018)](/papers/bert-devlin-2018) · [Fundamento BERT](/fundamentos/bert) · [Hub de practica](..).
