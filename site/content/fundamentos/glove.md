---
title: "GloVe"
weight: 281
math: true
---

**GloVe** (Global Vectors for Word Representation, Pennington-Socher-Manning 2014) es el complemento natural de [Word2Vec](/fundamentos/word2vec): mientras W2V usa **ventanas locales** y aproxima el LM, GloVe entrena directamente sobre la **matriz de co-ocurrencia global** del corpus con una loss cuadratica ponderada. **Unifica** las dos tradiciones de word embeddings -- count-based (LSA, PPMI) y predict-based (Word2Vec).

---

## 1. Idea central -- ratios de co-ocurrencia

La intuicion fundamental: las **probabilidades absolutas** $P(k \mid \text{ice})$ son ruidosas, pero los **ratios** $P(k \mid \text{ice}) / P(k \mid \text{steam})$ distinguen palabras relevantes de irrelevantes.

| $k$ | $P(k \mid \text{ice})$ | $P(k \mid \text{steam})$ | Ratio |
|---|---|---|---|
| solid | $1.9 \times 10^{-4}$ | $2.2 \times 10^{-5}$ | **8.9** |
| gas | $6.6 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | **0.085** |
| water | $3.0 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | 1.36 |
| fashion | $1.7 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | 0.96 |

- "solid" (relacionado con ice): ratio grande.
- "gas" (relacionado con steam): ratio chico.
- "water" o "fashion" (irrelevantes): ratio ~1.

**Conclusion**: el modelo debe aprender a **predecir el ratio**, no la probabilidad absoluta.

---

## 2. Derivacion de la loss

Sea $X \in \mathbb{N}^{|V| \times |V|}$ la matriz de co-ocurrencia, $X_i = \sum_k X_{ik}$, $P_{ij} = X_{ij}/X_i$.

Pennington postula $F(\mathbf{w}_i, \mathbf{w}_j, \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$ y deriva en 5 pasos:

1. **Diferencia vectorial**: $F((\mathbf{w}_i - \mathbf{w}_j), \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$.
2. **Producto punto** para escalar: $F((\mathbf{w}_i - \mathbf{w}_j)^T \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$.
3. **Simetria** (homomorfismo $(\mathbb{R}, +) \to (\mathbb{R}_{>0}, \times)$): unica solucion $F = \exp$.
4. Sustituyendo y tomando log: $\mathbf{w}_i^T \tilde{\mathbf{w}}_k = \log X_{ik} - \log X_i$.
5. Absorber $\log X_i$ en bias y agregar $\tilde{b}_k$ por simetria.

Resultado:

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_k + b_i + \tilde{b}_k = \log X_{ik}$$

### Loss final

$$\boxed{\mathcal{J} = \sum_{i,j=1}^{V} f(X_{ij}) \left( \mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2}$$

Least squares ponderada con factor $f(X_{ij})$.

---

## 3. La funcion de peso $f$

$$f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{si } x < x_{\max} \\ 1 & \text{si } x \ge x_{\max} \end{cases}$$

Con $x_{\max} = 100$ y $\alpha = 3/4$ (mismo exponente que en negative sampling de W2V).

**Tres desiderata**:
1. $f(0) = 0$ -- los ceros no contribuyen.
2. $f$ no-decreciente -- pares raros no dominan.
3. $f$ acotada -- pares frecuentes tampoco dominan.

---

## 4. Embedding final

$$\mathbf{w}_{\text{final}} = \mathbf{w} + \tilde{\mathbf{w}}$$

Se promedian las dos matrices (palabra + contexto) por simetria.

---

## 5. GloVe vs Word2Vec

| Aspecto | Word2Vec (SGNS) | GloVe |
|---|---|---|
| **Naturaleza** | Prediccion local | Factorizacion global |
| **Datos consumidos** | Streaming de ventanas | Matriz $X$ pre-computada |
| **Loss** | Log binary classification | Squared error ponderado |
| **Memoria** | $O(\|V\| d)$ | $O(\|X\|_{nnz})$ -- puede ser TB |
| **Embedding final** | $\mathbf{v}_w$ (input solo) | $\mathbf{w} + \tilde{\mathbf{w}}$ |
| **Hiperparametros clave** | ventana, $K$ negativos, subsampling | $x_{\max}, \alpha$, ventana |
| **Calidad** | Similar | Similar |

[Levy & Goldberg 2014](/papers/sgns-implicit-mf-levy-goldberg-2014) demuestran que **SGNS factoriza PMI shifted implicitamente** -- conectando ambos paradigmas.

---

## 6. Embeddings preentrenados publicos

Disponibles en https://nlp.stanford.edu/projects/glove/:

| Nombre | Corpus | Vocab | Dim | Tamano |
|---|---|---|---|---|
| `glove.6B` | Wikipedia + Gigaword 5 (6B) | 400k | 50/100/200/300 | 822 MB |
| `glove.42B.300d` | Common Crawl uncased (42B) | 1.9M | 300 | 1.75 GB |
| `glove.840B.300d` | Common Crawl cased (840B) | 2.2M | 300 | 2.03 GB |
| `glove.twitter.27B` | Twitter (27B) | 1.2M | 25/50/100/200 | 1.42 GB |

```python
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-300")
print(glove.most_similar("computer"))
print(glove.most_similar(positive=["king", "woman"], negative=["man"]))
```

---

## 7. Implementacion: GloVe loss

### PyTorch

```python
import torch
import torch.nn as nn

class GloVe(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, x_max: float = 100.0, alpha: float = 0.75):
        super().__init__()
        self.w = nn.Embedding(vocab_size, emb_dim)
        self.w_tilde = nn.Embedding(vocab_size, emb_dim)
        self.b = nn.Embedding(vocab_size, 1)
        self.b_tilde = nn.Embedding(vocab_size, 1)
        self.x_max = x_max
        self.alpha = alpha

    def f_weight(self, x):
        return torch.where(x < self.x_max, (x / self.x_max) ** self.alpha, torch.ones_like(x))

    def forward(self, i_idx, j_idx, x_ij):
        w_i = self.w(i_idx)
        w_j = self.w_tilde(j_idx)
        b_i = self.b(i_idx).squeeze(-1)
        b_j = self.b_tilde(j_idx).squeeze(-1)
        dot = (w_i * w_j).sum(dim=-1)
        diff = dot + b_i + b_j - torch.log(x_ij)
        return (self.f_weight(x_ij) * diff.pow(2)).mean()

    def get_embeddings(self):
        return self.w.weight + self.w_tilde.weight  # promedio simetrico
```

### TensorFlow

```python
import tensorflow as tf
from tensorflow.keras import layers

class GloVe(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, x_max=100.0, alpha=0.75):
        super().__init__()
        self.w = layers.Embedding(vocab_size, emb_dim)
        self.w_tilde = layers.Embedding(vocab_size, emb_dim)
        self.b = layers.Embedding(vocab_size, 1)
        self.b_tilde = layers.Embedding(vocab_size, 1)
        self.x_max, self.alpha = x_max, alpha

    def call(self, inputs):
        i_idx, j_idx, x_ij = inputs
        w_i = self.w(i_idx)
        w_j = self.w_tilde(j_idx)
        b_i = tf.squeeze(self.b(i_idx), -1)
        b_j = tf.squeeze(self.b_tilde(j_idx), -1)
        dot = tf.reduce_sum(w_i * w_j, axis=-1)
        diff = dot + b_i + b_j - tf.math.log(x_ij)
        weight = tf.where(x_ij < self.x_max, (x_ij / self.x_max) ** self.alpha, tf.ones_like(x_ij))
        return tf.reduce_mean(weight * tf.square(diff))
```

### JAX (Flax)

```python
import flax.linen as nn
import jax.numpy as jnp

class GloVe(nn.Module):
    vocab_size: int
    emb_dim: int
    x_max: float = 100.0
    alpha: float = 0.75

    def setup(self):
        self.w = nn.Embed(self.vocab_size, self.emb_dim)
        self.w_tilde = nn.Embed(self.vocab_size, self.emb_dim)
        self.b = nn.Embed(self.vocab_size, 1)
        self.b_tilde = nn.Embed(self.vocab_size, 1)

    def __call__(self, i_idx, j_idx, x_ij):
        w_i = self.w(i_idx)
        w_j = self.w_tilde(j_idx)
        b_i = jnp.squeeze(self.b(i_idx), -1)
        b_j = jnp.squeeze(self.b_tilde(j_idx), -1)
        dot = jnp.sum(w_i * w_j, axis=-1)
        diff = dot + b_i + b_j - jnp.log(x_ij)
        weight = jnp.where(x_ij < self.x_max, (x_ij / self.x_max) ** self.alpha, 1.0)
        return jnp.mean(weight * diff ** 2)
```

---

## 8. Construccion de la matriz de co-ocurrencia

```python
from collections import Counter

def build_cooc(corpus_tokens, vocab, window=5):
    """Construir matriz de co-ocurrencia con ventana simetrica y peso 1/d."""
    word2idx = {w: i for i, w in enumerate(vocab)}
    cooc = Counter()
    for i, w in enumerate(corpus_tokens):
        if w not in word2idx: continue
        for j in range(max(0, i - window), min(len(corpus_tokens), i + window + 1)):
            if j == i: continue
            c = corpus_tokens[j]
            if c not in word2idx: continue
            d = abs(i - j)
            cooc[(word2idx[w], word2idx[c])] += 1.0 / d   # peso decrece con distancia
    return cooc
```

El peso $1/d$ pondera mas contextos cercanos. Es una de las ideas implicitas de GloVe que mejoran calidad.

---

## 9. Limitaciones

1. **Memoria**: para Common Crawl, la matriz $X$ ocupa TB. Stanford soluciona con streaming en C.
2. **OOV**: igual que W2V, palabras no en vocab no tienen embedding.
3. **Sin subwords**: morfologia no se captura.
4. **Ventana fija**: dependencias largas no.
5. **Embedding no contextual**: un unico vector por palabra.

---

## 10. Cuando usar GloVe vs Word2Vec

| Si tu prioridad es... | Usa |
|---|---|
| Embeddings preentrenados de alta calidad | **GloVe.840B.300d** |
| Entrenar embeddings en corpus pequeno | **Skip-gram** (mas robusto en pocos datos) |
| Velocidad de inferencia | Ambos son equivalentes (lookup table) |
| Analogias semanticas | GloVe (gana en general) |
| Analogias sintacticas | SGNS (mejor en palabras frecuentes) |
| Word similarity | Empate |
| Investigacion sobre embeddings | Ambos como baseline |

---

## Referencias

- [Pennington 2014 - GloVe paper](/papers/glove-pennington-2014).
- [Levy & Goldberg 2014 - SGNS as Implicit MF](/papers/sgns-implicit-mf-levy-goldberg-2014) -- conexion teorica con W2V.

## Fundamentos relacionados

- [Word2Vec](/fundamentos/word2vec), [Modelos de lenguaje](/fundamentos/modelos-de-lenguaje), [Embeddings distribuidos](/fundamentos/embeddings-distribuidos), [Bag of Words](/fundamentos/bag-of-words).

## Clases relacionadas

- [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
