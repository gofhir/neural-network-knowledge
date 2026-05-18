---
title: "Word2Vec"
weight: 280
math: true
---

**Word2Vec** (Mikolov et al. 2013) es el modelo que **democratizo los word embeddings**. Dos arquitecturas drasticamente simplificadas -- CBoW y Skip-gram -- que aprenden vectores densos de palabras desde corpora masivos en horas, no semanas. Sus innovaciones (negative sampling, subsampling, phrase detection) definen el paradigma de embeddings hasta el dia de hoy.

Este fundamento condensa los dos papers ([Efficient Estimation](/papers/word2vec-efficient-mikolov-2013), [Distributed Representations](/papers/word2vec-distributed-mikolov-2013)) en una referencia operativa.

---

## 1. Las dos arquitecturas

### 1.1 Continuous Bag-of-Words (CBoW)

**Tarea**: predecir la palabra central $w_t$ dado el contexto $\{w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c}\}$.

```mermaid
graph LR
    W1[w_{t-2}] --> E1[Embedding]
    W2[w_{t-1}] --> E2[Embedding]
    W3[w_{t+1}] --> E3[Embedding]
    W4[w_{t+2}] --> E4[Embedding]
    E1 --> SUM[Suma/Promedio]
    E2 --> SUM
    E3 --> SUM
    E4 --> SUM
    SUM --> SOFTMAX[Softmax sobre V]
    SOFTMAX --> Wt[w_t]
    
    style SUM fill:#fbbf24,color:#000
```

$$\mathbf{h} = \frac{1}{2c} \sum_{j \neq 0} \mathbf{v}_{w_{t+j}}$$

$$P(w_t = k \mid \text{ctx}) = \frac{\exp(\mathbf{v}'_k \cdot \mathbf{h})}{\sum_i \exp(\mathbf{v}'_i \cdot \mathbf{h})}$$

**No usa orden** (es "bag" -- invariante a permutacion del contexto).

### 1.2 Continuous Skip-gram

**Tarea inversa**: dado $w_t$, predecir cada palabra del contexto.

```mermaid
graph LR
    Wt[w_t] --> E[Embedding]
    E --> P1[P(w_{t-2}|w_t)]
    E --> P2[P(w_{t-1}|w_t)]
    E --> P3[P(w_{t+1}|w_t)]
    E --> P4[P(w_{t+2}|w_t)]
    
    style E fill:#fbbf24,color:#000
```

$$P(w_{t+j} = k \mid w_t) = \frac{\exp(\mathbf{v}'_k \cdot \mathbf{v}_{w_t})}{\sum_i \exp(\mathbf{v}'_i \cdot \mathbf{v}_{w_t})}$$

$$\mathcal{L}_{\text{SG}} = \frac{1}{T} \sum_t \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} \mid w_t)$$

### 1.3 Cuando usar cada uno

| | CBoW | Skip-gram |
|---|---|---|
| Velocidad | Rapida | ~5x mas lenta |
| Calidad sintactica | Mejor | Peor |
| Calidad semantica | Peor | Mejor |
| Palabras raras | Peor (la suma diluye) | Mejor |
| Datasets pequenos | OK | Recomendado |
| Datasets grandes | OK | **Estandar** |

Post-2013, Skip-gram con negative sampling (SGNS) dominio la era.

---

## 2. Dos matrices de embeddings

Cada palabra $w$ tiene **dos vectores distintos**:

- $\mathbf{v}_w \in \mathbb{R}^d$: **input vector** (cuando $w$ es palabra central).
- $\mathbf{v}'_w \in \mathbb{R}^d$: **output vector** (cuando $w$ es palabra de contexto).

Tras entrenar, $\mathbf{v}_w$ se exporta como "el embedding"; $\mathbf{v}'_w$ se descarta o se promedia con $\mathbf{v}_w$ (la practica varia).

{{< concept-alert type="atencion" >}}
Esta **dos matrices distintas** es esencial. Si forzaramos $\mathbf{v} = \mathbf{v}'$ (i.e., $\mathbf{W}^\top \mathbf{W} = \text{PMI}$), la matriz PMI tendria que ser positive semi-definite -- lo cual **no es cierto** para corpora reales (Levy & Goldberg 2014).
{{< /concept-alert >}}

---

## 3. Negative Sampling -- la innovacion practica clave

El softmax exacto sobre $\|V\| = 10^6$ es $O(\|V\| \cdot d)$ por ejemplo. **Negative sampling** lo reemplaza por clasificacion binaria:

$$\mathcal{L}_{\text{SGNS}} = \log \sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n} \left[ \log \sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I}) \right]$$

- Primer termino: par real -> producto punto **alto**.
- Segundo termino: $k$ negativos -> producto punto **bajo**.

**Costo**: $O((k+1) \cdot d)$, independiente de $\|V\|$. Speedup ~$10^5$.

### El exponente $3/4$

$P_n(w) \propto U(w)^{3/4}$ donde $U(w)$ es la frecuencia unigrama. Sin justificacion teorica pero empiricamente robusto.

### Numero de negativos $k$

- Datasets pequenos (1B palabras): $k = 5$-$20$.
- Datasets grandes (>10B palabras): $k = 2$-$5$.

---

## 4. Subsampling de palabras frecuentes

Cada ocurrencia de $w_i$ se **descarta** con probabilidad:

$$P_{\text{discard}}(w_i) = 1 - \sqrt{t / f(w_i)}$$

con $t \approx 10^{-5}$. Palabras como "the" ($f = 0.07$) se descartan **98.8%** de las veces.

**Beneficios**:
- 2-10x speedup.
- Mejor calidad de embeddings de palabras raras (no diluidas por co-ocurrencias triviales).

---

## 5. Hierarchical Softmax con arbol Huffman

Alternativa exacta al softmax: organizar $\|V\|$ palabras en un **arbol binario** (Huffman para palabras frecuentes con caminos cortos):

$$P(w \mid w_I) = \prod_{j=1}^{L(w)-1} \sigma\left( [\![n(w, j+1) = \text{ch}(n(w,j))]\!] \cdot \mathbf{v}'_{n(w,j)} \cdot \mathbf{v}_{w_I} \right)$$

**Costo**: $O(\log V \cdot d)$, exacto (suma a 1).

**HS vs NEG**:
- NEG gana en velocidad y palabras frecuentes / sintaxis.
- HS gana en palabras raras (cada palabra tiene camino propio).

Para analogias **de palabras**: NEG-15 con subsampling es el mejor.
Para analogias **de frases**: HS con subsampling es el mejor.

---

## 6. Phrase embeddings

Detectar frases idiomaticas frecuentes via score bigrama:

$$\text{score}(w_i, w_j) = \frac{\text{count}(w_i w_j) - \delta}{\text{count}(w_i) \cdot \text{count}(w_j)}$$

Bigramas por encima de threshold se reemplazan por tokens unicos (`New_York`, `Air_Canada`). 2-4 pasadas con threshold decreciente para formar frases mas largas.

Permite analogias como:
- `vec("Montreal Canadiens") - vec("Montreal") + vec("Toronto") ≈ vec("Toronto Maple Leafs")`
- `vec("Air Canada") - vec("Canada") + vec("France") ≈ vec("Air France")`

---

## 7. Composicionalidad aditiva

Observacion sorprendente: `vec(Russia) + vec(river)` ~ `vec(Volga River)`. Explicacion (Mikolov 2013, seccion 5):

Si $\mathbf{v}_w \cdot \mathbf{v}'_c \approx \log P(c \mid w)$, entonces:

$$(\mathbf{v}_{w_1} + \mathbf{v}_{w_2}) \cdot \mathbf{v}'_c \approx \log [P(c \mid w_1) \cdot P(c \mid w_2)]$$

Las palabras que aparecen en contextos **comunes a ambos $w_1, w_2$** rankean alto -- es el **"AND" semantico**.

Ejemplos famosos (Tabla 5 del paper):

| Suma | Vecinos top |
|---|---|
| Czech + currency | koruna |
| Vietnam + capital | Hanoi |
| German + airlines | Lufthansa |
| Russian + river | Volga River |

---

## 8. Analogias y composicionalidad lineal

`vec(king) - vec(man) + vec(woman) ≈ vec(queen)`.

Justificacion matematica: ver [Allen & Hospedales 2019](/papers/analogies-explained-allen-hospedales-2019) -- bajo la hipotesis distribucional + factorizacion PMI ([Levy & Goldberg](/papers/sgns-implicit-mf-levy-goldberg-2014)), las analogias corresponden a **word transformations con parametros compartidos**, lo que implica relaciones lineales entre los embeddings con terminos de error explicitos.

---

## 9. Hiperparametros y embeddings preentrenados

### Configuracion estandar de Mikolov

| Hiperparametro | Valor recomendado |
|---|---|
| Dimension $d$ | 100, 200, 300 |
| Ventana $c$ | 5 (CBoW), 10 (Skip-gram) |
| Negativos $k$ | 5 (grande) - 15 (pequeno) |
| Subsampling $t$ | $10^{-5}$ |
| Min count | 5 |
| Epochs | 5 |
| Learning rate inicial | 0.025 (SG), 0.05 (CBoW) |
| Optimizer | SGD con decay lineal |

### Embeddings preentrenados publicos

| Nombre | Corpus | Vocab | Dim | Tamano |
|---|---|---|---|---|
| `GoogleNews-vectors-negative300.bin` | Google News 100B | 3M | 300 | 1.6 GB |

Descargar via [Gensim](https://radimrehurek.com/gensim/):

```python
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
print(model.most_similar("computer"))
print(model.similarity("king", "queen"))
print(model.most_similar(positive=["king", "woman"], negative=["man"]))
```

---

## 10. Implementacion: Skip-gram con NS

### PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, emb_dim)
        self.out_emb = nn.Embedding(vocab_size, emb_dim)
        nn.init.uniform_(self.in_emb.weight, -0.5/emb_dim, 0.5/emb_dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, context, negatives):
        v_c = self.in_emb(center)          # [B, D]
        u_p = self.out_emb(context)        # [B, D]
        u_n = self.out_emb(negatives)      # [B, K, D]

        pos_score = (u_p * v_c).sum(dim=-1)
        neg_score = torch.bmm(u_n, v_c.unsqueeze(-1)).squeeze(-1)

        pos_loss = F.logsigmoid(pos_score)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)
        return -(pos_loss + neg_loss).mean()
```

### TensorFlow

```python
import tensorflow as tf
from tensorflow.keras import layers

class SkipGramNS(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.in_emb = layers.Embedding(vocab_size, emb_dim)
        self.out_emb = layers.Embedding(vocab_size, emb_dim)

    def call(self, inputs):
        center, context, negatives = inputs
        v_c = self.in_emb(center)
        u_p = self.out_emb(context)
        u_n = self.out_emb(negatives)
        pos = tf.reduce_sum(u_p * v_c, axis=-1)
        neg = tf.einsum("bkd,bd->bk", u_n, v_c)
        return -tf.reduce_mean(
            tf.math.log_sigmoid(pos) + tf.reduce_sum(tf.math.log_sigmoid(-neg), axis=-1)
        )
```

### JAX (Flax)

```python
import flax.linen as nn
import jax.numpy as jnp
import jax

class SkipGramNS(nn.Module):
    vocab_size: int
    emb_dim: int

    def setup(self):
        self.in_emb = nn.Embed(self.vocab_size, self.emb_dim)
        self.out_emb = nn.Embed(self.vocab_size, self.emb_dim)

    def __call__(self, center, context, negatives):
        v_c = self.in_emb(center)
        u_p = self.out_emb(context)
        u_n = self.out_emb(negatives)
        pos = jnp.sum(u_p * v_c, axis=-1)
        neg = jnp.einsum("bkd,bd->bk", u_n, v_c)
        return -jnp.mean(
            jax.nn.log_sigmoid(pos) + jnp.sum(jax.nn.log_sigmoid(-neg), axis=-1)
        )
```

---

## 11. Limitaciones y legado

### Limitaciones de Word2Vec

1. **No es un LM**: solo embeddings.
2. **Sin orden**: CBoW promedia.
3. **Sin subwords**: morfologia no se captura (FastText 2016 lo resuelve).
4. **Embeddings no contextuales**: un unico vector por palabra (BERT 2018 lo resuelve).
5. **Polisemia ignorada**: "apple" = un solo vector.
6. **Sesgos sociales**: Bolukbasi 2016 documenta sesgos de genero.

### Sucesores

| Ano | Modelo | Innovacion |
|---|---|---|
| 2014 | [GloVe](/fundamentos/glove) | Factorizacion explicita de log-co-ocurrencia |
| 2015 | [Skip-Thought](/fundamentos/skip-thought) | Sentence embeddings |
| 2016 | FastText | Subword embeddings |
| 2018 | ELMo | Embeddings contextuales con biLSTM |
| 2018 | [BERT](/fundamentos/bert) | Embeddings contextuales con Transformer |

---

## Referencias

- [Word2Vec Efficient Estimation 2013](/papers/word2vec-efficient-mikolov-2013) -- CBoW + Skip-gram.
- [Word2Vec Distributed Representations 2013](/papers/word2vec-distributed-mikolov-2013) -- Negative sampling, subsampling, phrases.
- [Levy & Goldberg 2014 - SGNS as Implicit MF](/papers/sgns-implicit-mf-levy-goldberg-2014) -- analisis teorico.
- [Allen & Hospedales 2019 - Analogies Explained](/papers/analogies-explained-allen-hospedales-2019) -- por que las analogias funcionan.

## Fundamentos relacionados

- [Modelos de lenguaje](/fundamentos/modelos-de-lenguaje), [Embeddings distribuidos](/fundamentos/embeddings-distribuidos), [GloVe](/fundamentos/glove), [Skip-Thought](/fundamentos/skip-thought), [BERT](/fundamentos/bert).

## Clases relacionadas

- [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
