---
title: "Profundizacion - LM, n-gramas, NPLM, W2V, GloVe, Skip-Thought"
weight: 20
math: true
---

> Este documento profundiza los fundamentos matematicos detras de la Clase 18.
> Cubre suavizado n-gram (Laplace, Kneser-Ney), perplejidad como metrica,
> la arquitectura NPLM completa (Bengio 2003), negative sampling y hierarchical softmax
> de Word2Vec, derivacion completa de la loss de GloVe, y conditional GRU de Skip-Thought.

---

# Parte I: Modelos de Lenguaje n-gram en profundidad

## 1. MLE y el problema de sparsity

Estimador por maxima verosimilitud:

$$P_{\text{MLE}}(w_t \mid w_{t-N+1:t-1}) = \frac{\text{count}(w_{t-N+1:t})}{\text{count}(w_{t-N+1:t-1})}$$

**Problema critico** (no mencionado en slides): la mayoria de n-gramas posibles **nunca se observan**. Para vocab $\|V\| = 50k$ y $n = 3$ hay $1.25 \times 10^{14}$ trigramas posibles -- ningun corpus humano contiene mas que una fraccion infima. Un solo $P = 0$ rompe la regla de la cadena (anula toda la oracion).

## 2. Tecnicas de suavizado

### 2.1 Laplace (add-one)

$$P_{\text{Lap}}(w_t \mid w_{t-N+1:t-1}) = \frac{\text{count}(w_{t-N+1:t}) + 1}{\text{count}(w_{t-N+1:t-1}) + |V|}$$

Garantiza $P > 0$ para todos los n-gramas. **Limitacion**: suaviza demasiado para vocabularios grandes -- la masa total se redistribuye uniformemente entre todos los posibles n-gramas.

### 2.2 Add-k smoothing

Reemplaza "+1" por "+$k$" con $k < 1$ optimizado en validacion. Mas flexible pero requiere tuning.

### 2.3 Katz backoff (Katz 1987)

Si un trigrama no se vio, retroceder a bigrama. Si bigrama no se vio, retroceder a unigrama. Con factores de descuento para conservar la masa total.

### 2.4 Kneser-Ney modificado -- el SOTA estadistico

Chen & Goodman 1998. Idea clave: usar **continuation probability** $P_{\text{cont}}(w)$ en vez de frecuencia bruta.

$$P_{\text{KN}}(w \mid h) = \frac{\max(\text{count}(h, w) - d, 0)}{\sum_{w'} \text{count}(h, w')} + \lambda(h) \cdot P_{\text{cont}}(w)$$

Donde:

$$P_{\text{cont}}(w) = \frac{|\{h' : \text{count}(h', w) > 0\}|}{|\{(h', w') : \text{count}(h', w') > 0\}|}$$

es la fraccion de bigramas distintos terminados en $w$.

**Intuicion**: una palabra puede ser frecuente (como "Francisco") pero solo en contextos restringidos (despues de "San"). $P_{\text{cont}}$ corrige esto -- mide diversidad de contextos, no frecuencia absoluta.

Kneser-Ney modificado (con tres descuentos $d_1, d_2, d_{3+}$ segun el conteo) fue el **SOTA de LMs estadisticos durante una decada**, hasta NPLM y RNN-LM.

## 3. Perplejidad: la metrica fundamental

**No aparece en las slides** pero es esencial. Sobre un conjunto de test de $T$ tokens:

$$\text{PPL}(w_{1:T}) = P(w_{1:T})^{-1/T} = \exp\left( -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_{1:t-1}) \right)$$

Interpretacion: PPL es la **media geometrica inversa** de las probabilidades asignadas -- equivalente al exponencial de la **cross-entropy promedio**. Un LM con PPL $= k$ "duda entre $k$ palabras igualmente probables" en cada paso.

| Modelo | PPL en WikiText-103 |
|---|---|
| 5-gram Kneser-Ney | ~80 |
| LSTM-LM | ~50 |
| Transformer-XL | ~18 |
| GPT-3 (zero-shot) | ~10-15 |

PPL conecta con **cross-entropy loss**: minimizar CE durante entrenamiento minimiza log-PPL.

---

# Parte II: Neural Probabilistic LM (Bengio 2003)

El slide 21 reproduce el diagrama del [Bengio NPLM](/papers/nplm-bengio-2003) -- vale la pena derivarlo formalmente.

## 4. Arquitectura completa

Para predecir $w_t$ dado el contexto $w_{t-n+1:t-1}$ (ventana fija de $n - 1$ palabras):

### Paso 1: lookup de embeddings

Matriz $C \in \mathbb{R}^{|V| \times m}$. Cada palabra del contexto se mapea a su fila correspondiente:

$$\mathbf{x} = [C(w_{t-n+1}); C(w_{t-n+2}); \ldots; C(w_{t-1})] \in \mathbb{R}^{(n-1)m}$$

Concatenacion en orden.

### Paso 2: MLP

$$\mathbf{h} = \tanh(d + H \mathbf{x})$$

con $H \in \mathbb{R}^{h \times (n-1)m}$, $d \in \mathbb{R}^h$, donde $h$ es la dimension hidden (50-100 en el paper original).

### Paso 3: output

$$\mathbf{y} = b + W\mathbf{x} + U\mathbf{h}$$

con $U \in \mathbb{R}^{|V| \times h}$, $W \in \mathbb{R}^{|V| \times (n-1)m}$ (skip connections opcionales), $b \in \mathbb{R}^{|V|}$.

$$\hat{P}(w_t = i \mid w_{<t}) = \frac{e^{y_i}}{\sum_j e^{y_j}}$$

### Numero de parametros

$$|V|(1 + nm + h) + h(1 + (n-1)m)$$

Para $|V| = 17.964$, $n = 6$, $m = 100$, $h = 60$: ~12M parametros.

## 5. Generalizacion via geometria

**Insight clave de Bengio**: la red neuronal es una **funcion suave** de los embeddings. Si dos palabras tienen vectores cercanos, sus predicciones son cercanas. Por lo tanto, una sola observacion de "The cat is walking in the bedroom" eleva la probabilidad de:

- "A dog was walking in the bedroom" (cat -> dog cercanos).
- "A dog is running in the room" (multiple sustituciones).
- ... un numero **exponencial** de oraciones vecinas en el espacio de embeddings.

Resuelve la **curse of dimensionality** de n-gramas: el modelo generaliza a combinaciones nunca vistas via similitud semantica.

## 6. Mixture con n-gram

Bengio reporta que combinar NPLM con interpolated trigram mediante interpolacion lineal ($\alpha = 0.5$) **mejora** la perplejidad. Los dos modelos capturan informacion complementaria:

- NN: generaliza via similitud.
- Trigram: captura n-gramas frecuentes especificos exactamente.

---

# Parte III: Word2Vec en profundidad

## 7. Negative Sampling -- derivacion

El [paper companion de Word2Vec](/papers/word2vec-distributed-mikolov-2013) introduce **negative sampling** como alternativa al softmax exacto.

Para un par observado $(w_I, w_O)$, en lugar de:

$$P(w_O \mid w_I) = \frac{\exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})}{\sum_w \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_I})}$$

(costoso por el denominador), modelar **clasificacion binaria**:

$$\mathcal{L}_{\text{SGNS}} = \log \sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I}) \right]$$

**Interpretacion**:
- Primer termino: el par real debe tener producto punto **alto** ($\sigma \to 1$).
- Segundo termino: cada uno de los $k$ negativos debe tener producto punto **bajo** ($\sigma(-\cdot) \to 1$, i.e., $\sigma \to 0$).

**Costo**: $O((k+1) \cdot d)$ -- independiente de $\|V\|$. Speedup $\sim 10^5$ vs softmax exacto.

### Distribucion de ruido $P_n$

Empiricamente, $P_n(w) \propto U(w)^{3/4}$ donde $U$ es la frecuencia unigrama. El exponente $3/4$ comprime la distribucion -- palabras frecuentes pierden masa relativa, raras ganan.

### Numero de negativos $k$

- Datasets pequenos: $k = 5$-$20$.
- Datasets grandes: $k = 2$-$5$.

## 8. Hierarchical Softmax con Huffman

Alternativa exacta: organizar palabras en arbol binario donde palabras frecuentes tienen caminos cortos.

$$P(w \mid w_I) = \prod_{j=1}^{L(w)-1} \sigma\left( [\![n(w, j+1) = \text{ch}(n(w,j))]\!] \cdot \mathbf{v}'_{n(w,j)} \cdot \mathbf{v}_{w_I} \right)$$

- $n(w, j)$: $j$-esimo nodo en el camino raiz -> $w$.
- $\text{ch}(n)$: hijo predeterminado de $n$.
- $[\![\cdot]\!]$: indicator $\{1, -1\}$.
- $L(w)$: profundidad del camino.

**Costo**: $O(\log V \cdot d)$. Una representacion por palabra + una por nodo interno.

## 9. Subsampling de palabras frecuentes

Cada ocurrencia de $w_i$ se descarta con probabilidad:

$$P_{\text{discard}}(w_i) = 1 - \sqrt{t / f(w_i)}, \quad t \approx 10^{-5}$$

Palabras como "the" ($f = 0.07$): $P_{\text{discard}} = 1 - \sqrt{10^{-5}/0.07} \approx 0.988$ -- se descartan el 98.8%.

**Beneficios**: 2-10x speedup + mejora en embeddings de palabras raras.

## 10. SGNS factoriza PMI shifted

Resultado clave de [Levy & Goldberg 2014](/papers/sgns-implicit-mf-levy-goldberg-2014):

$$\mathbf{w} \cdot \mathbf{c} \approx \text{PMI}(w, c) - \log k$$

donde $\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w) P(c)}$. Esto conecta Word2Vec (predict-based) con LSA / PPMI (count-based) -- son la **misma factorizacion implicita**.

Implicacion: SGNS = factorizacion ponderada de la matriz PMI. El "pesado" (proporcional a $\#(w, c)$) explica por que SGNS gana sobre SVD-uniforme en analogias.

---

# Parte IV: Derivacion de la loss de GloVe

## 11. Los 5 pasos del paper

Sea $X_{ij}$ = numero de veces que $j$ aparece en contexto de $i$. $P_{ij} = X_{ij}/X_i$ con $X_i = \sum_k X_{ik}$.

**Observacion inicial**: las razones $P_{ik}/P_{jk}$ encodifican mejor las relaciones semanticas que probabilidades absolutas (Tabla 1 del paper, comparando ice/steam con solid, gas, water, fashion).

### Paso 1 -- diferencia vectorial

$$F(\mathbf{w}_i, \mathbf{w}_j, \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$$

Restringimos $F$ a depender solo de la diferencia: $F((\mathbf{w}_i - \mathbf{w}_j), \tilde{\mathbf{w}}_k)$.

### Paso 2 -- producto punto

Ambos lados deben ser escalares:

$$F((\mathbf{w}_i - \mathbf{w}_j)^T \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$$

### Paso 3 -- simetria palabra <-> contexto

Imponer simetria al intercambiar $\mathbf{w} \leftrightarrow \tilde{\mathbf{w}}$, $X \leftrightarrow X^T$. Esto fuerza $F$ a ser un **homomorfismo entre grupos**:

$$F((\mathbf{w}_i - \mathbf{w}_j)^T \tilde{\mathbf{w}}_k) = \frac{F(\mathbf{w}_i^T \tilde{\mathbf{w}}_k)}{F(\mathbf{w}_j^T \tilde{\mathbf{w}}_k)}$$

$F: (\mathbb{R}, +) \to (\mathbb{R}_{>0}, \times)$ con esta propiedad -> $F = \exp$.

### Paso 4 -- absorber $\log X_i$

Sustituyendo $F = \exp$:

$$\exp(\mathbf{w}_i^T \tilde{\mathbf{w}}_k) = P_{ik} = X_{ik}/X_i$$

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_k = \log X_{ik} - \log X_i$$

$\log X_i$ depende solo de $i$ -> absorberlo en bias $b_i$. Agregar $\tilde{b}_k$ para mantener simetria:

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_k + b_i + \tilde{b}_k = \log X_{ik}$$

### Paso 5 -- least squares ponderado

$$\boxed{\mathcal{J} = \sum_{i,j=1}^{V} f(X_{ij}) \left( \mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2}$$

con $f$ funcion de peso para manejar el rango enorme de $X_{ij}$ (de 0 a $10^7$).

## 12. La funcion de peso $f$

$$f(x) = \begin{cases} (x / x_{\max})^\alpha & x < x_{\max} \\ 1 & x \ge x_{\max} \end{cases}$$

con $x_{\max} = 100$, $\alpha = 3/4$.

**Desiderata**:
1. $f(0) = 0$ -- los ceros no contribuyen (y son ~75-95% de las entradas).
2. $f$ no-decreciente -- pares raros no dominan.
3. $f$ acotada -- pares muy frecuentes (con "the") no dominan.

El exponente $3/4$ es **el mismo** que en negative sampling de Word2Vec. Constante "magica" robusta de las estadisticas Zipf.

## 13. Embedding final

$$\mathbf{w}_{\text{final}} = \mathbf{w} + \tilde{\mathbf{w}}$$

Promedio simetrico de las dos matrices. Mejora consistentemente en tareas downstream.

---

# Parte V: Skip-Thought y Conditional GRU

## 14. Encoder GRU estandar

Para la oracion $s_i = (w_i^1, \ldots, w_i^N)$:

$$\mathbf{r}^t = \sigma(\mathbf{W}_r \mathbf{x}^t + \mathbf{U}_r \mathbf{h}^{t-1})$$

$$\mathbf{z}^t = \sigma(\mathbf{W}_z \mathbf{x}^t + \mathbf{U}_z \mathbf{h}^{t-1})$$

$$\bar{\mathbf{h}}^t = \tanh(\mathbf{W} \mathbf{x}^t + \mathbf{U}(\mathbf{r}^t \odot \mathbf{h}^{t-1}))$$

$$\mathbf{h}^t = (1 - \mathbf{z}^t) \odot \mathbf{h}^{t-1} + \mathbf{z}^t \odot \bar{\mathbf{h}}^t$$

El sentence embedding es $\mathbf{h}_i = \mathbf{h}^N$ (estado final).

## 15. Conditional GRU decoder

**Innovacion** del paper: inyectar $\mathbf{h}_i$ en **cada gate** del decoder via matrices nuevas $\mathbf{C}_r, \mathbf{C}_z, \mathbf{C}$:

$$\mathbf{r}^t = \sigma(\mathbf{W}_r^d \mathbf{x}^{t-1} + \mathbf{U}_r^d \mathbf{h}^{t-1} + \mathbf{C}_r \mathbf{h}_i)$$

$$\mathbf{z}^t = \sigma(\mathbf{W}_z^d \mathbf{x}^{t-1} + \mathbf{U}_z^d \mathbf{h}^{t-1} + \mathbf{C}_z \mathbf{h}_i)$$

$$\bar{\mathbf{h}}^t = \tanh(\mathbf{W}^d \mathbf{x}^{t-1} + \mathbf{U}^d (\mathbf{r}^t \odot \mathbf{h}^{t-1}) + \mathbf{C} \mathbf{h}_i)$$

$$\mathbf{h}_{i+1}^t = (1 - \mathbf{z}^t) \odot \mathbf{h}^{t-1} + \mathbf{z}^t \odot \bar{\mathbf{h}}^t$$

Dos decoders separados (para $s_{i+1}$ y $s_{i-1}$) con parametros distintos. Solo comparten la matriz de vocabulario.

## 16. Vocabulary expansion

Resolver el problema de OOV en test time: aprender una regresion lineal:

$$\mathbf{W}_{\text{exp}} = \arg\min_W \sum_{w \in V_{\text{rnn}} \cap V_{\text{w2v}}} \| W \cdot \text{w2v}(w) - \text{rnn}(w) \|^2$$

Para palabras nuevas en W2V pero no en RNN: $\text{rnn}'(w) = W_{\text{exp}} \cdot \text{w2v}(w)$.

Vocabulario efectivo: de 20k a **930.911**.

## 17. Por que importa Skip-Thought hoy

Skip-Thought establece el **patron** que dominaria sentence embeddings durante 5 anos:

1. Encoder universal aprendido sin supervision (precede InferSent, USE, SBERT).
2. Autosupervision a nivel de oracion (precede NSP de BERT, SOP de ALBERT, contrastive de SimCSE).

---

# Parte VI: Allen & Hospedales -- por que las analogias funcionan

## 18. Paraphrase y word transformation

**Definicion (paraphrase)**: $w_*$ parafrasea $\mathcal{W}$ si $p(c_j \mid w_*) \approx p(c_j \mid \mathcal{W})$ para todo contexto $c_j$.

**Theorem 1** (Allen & Hospedales 2019):

$$\mathbf{w}_* = \mathbf{w}_{\mathcal{W}} + \mathbf{C}^\dagger (\boldsymbol{\rho}^{\mathcal{W}, w_*} + \boldsymbol{\sigma}^{\mathcal{W}} - \tau^{\mathcal{W}} \mathbf{1})$$

donde:
- $\mathbf{w}_{\mathcal{W}} = \sum_{w_i \in \mathcal{W}} \mathbf{w}_i$ (suma de embeddings).
- $\boldsymbol{\rho}^{\mathcal{W}, w_*}$ = **paraphrase error**.
- $\boldsymbol{\sigma}^{\mathcal{W}}, \tau^{\mathcal{W}}$ = **dependence errors** dentro de $\mathcal{W}$.
- $\mathbf{C}^\dagger$ = pseudo-inversa de Moore-Penrose.

**Corolario**: $\mathbf{w}_* \approx \mathbf{w}_{\mathcal{W}}$ si $w_*$ parafrasea $\mathcal{W}$ y las palabras de $\mathcal{W}$ son materialmente independientes.

## 19. De parafrasis a analogias

Una **analogia $a:a^* :: b:b^*$ se cumple** si los mismos parametros $\mathcal{W}^+, \mathcal{W}^-$ transforman ambos pares. Por ejemplo, `man:king :: woman:queen` se cumple con $\mathcal{W}^+ = \{\text{royal}\}$, $\mathcal{W}^- = \emptyset$.

**Resultado central** (derivado del Theorem 1):

$$\boxed{\mathbf{w}_{b^*} \approx \mathbf{w}_{a^*} - \mathbf{w}_a + \mathbf{w}_b}$$

con terminos de error explicitos. Esta es la **primera prueba matematica rigurosa** de la relacion `king - man + woman ≈ queen`.

---

# Parte VII: Implementacion -- los 3 frameworks

## 20. Skip-gram con NS

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
        v_c = self.in_emb(center)
        u_p = self.out_emb(context)
        u_n = self.out_emb(negatives)
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
import jax
import jax.numpy as jnp

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

## 21. GloVe loss en los 3 frameworks

Ver implementaciones completas en el [fundamento GloVe](/fundamentos/glove#7-implementacion-glove-loss).

---

## 22. Referencias

- Bengio et al. 2003. [A Neural Probabilistic Language Model](/papers/nplm-bengio-2003).
- Mikolov et al. 2010. [Recurrent Neural Network Based Language Model](/papers/rnn-lm-mikolov-2010).
- Mikolov et al. 2013a. [Efficient Estimation of Word Representations in Vector Space](/papers/word2vec-efficient-mikolov-2013).
- Mikolov et al. 2013b. [Distributed Representations of Words and Phrases and their Compositionality](/papers/word2vec-distributed-mikolov-2013).
- Pennington, Socher, Manning 2014. [GloVe: Global Vectors for Word Representation](/papers/glove-pennington-2014).
- Levy, Goldberg 2014. [Neural Word Embedding as Implicit Matrix Factorization](/papers/sgns-implicit-mf-levy-goldberg-2014).
- Kiros et al. 2015. [Skip-Thought Vectors](/papers/skip-thought-kiros-2015).
- Allen, Hospedales 2019. [Analogies Explained: Towards Understanding Word Embeddings](/papers/analogies-explained-allen-hospedales-2019).
- Chen & Goodman 1998. *An Empirical Study of Smoothing Techniques for Language Modeling*.
- Jurafsky & Martin. *Speech and Language Processing*, capitulos 3, 6, 7. https://web.stanford.edu/~jurafsky/slp3/
