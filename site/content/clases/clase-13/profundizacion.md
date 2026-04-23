---
title: "Profundizacion - Attention Math y Camino al Transformer"
weight: 20
math: true
---

> Este documento profundiza en los fundamentos matematicos detras de la Clase 13.
> Cubre la derivacion completa de Seq2Seq con maximizacion de log-verosimilitud,
> el mecanismo de atencion aditivo de Bahdanau y sus variantes,
> el analisis de soft vs hard attention, el coverage mechanism de pointer-generator,
> y la evolucion conceptual hacia el Transformer.

---

# Parte I: Seq2Seq Formal

---

## 1. Formulacion Probabilistica

Dado un par (fuente, objetivo) $(x, y)$ con $x = (x_1, \ldots, x_{T_x})$ y $y = (y_1, \ldots, y_{T_y})$, buscamos maximizar la probabilidad condicional:

$$P(y \mid x; \theta) = \prod_{t=1}^{T_y} P(y_t \mid y_{<t}, x; \theta)$$

que descompone por la regla de la cadena. Para cada paso:

$$P(y_t \mid y_{<t}, x) = g(y_{t-1}, s_t, c)$$

donde:
- $s_t$ es el estado del decoder en el paso $t$.
- $c$ es el context vector (resumen del input).
- $g$ es tipicamente un softmax sobre vocabulario: $P(y_t = w) = \text{softmax}(W_o s_t + b_o)$.

### 1.1 Objetivo de entrenamiento

Maximizar log-verosimilitud sobre el training set $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \log P(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T_y^{(i)}} \log P(y_t^{(i)} \mid y_{<t}^{(i)}, x^{(i)}; \theta)$$

Loss per step: **cross-entropy**. Implementado como NLL del token correcto.

### 1.2 Teacher forcing vs. free-running

Durante training:
- **Teacher forcing**: alimentar $y_{t-1}^*$ (ground truth) como input al decoder en el paso $t$.
- **Free-running**: alimentar $\hat{y}_{t-1}$ (la prediccion del modelo).

Teacher forcing **acelera convergencia** y permite paralelizar el entrenamiento (todos los $y_t$ se calculan en una sola pasada con shift-right mask). Pero crea **exposure bias**: en inferencia, el modelo nunca vio sus propias predicciones.

### 1.3 Scheduled Sampling (Bengio et al. 2015)

Mitiga exposure bias mezclando teacher forcing y free-running:

- Con probabilidad $\epsilon$: usar ground truth.
- Con probabilidad $1 - \epsilon$: usar la prediccion del modelo.

$\epsilon$ decae de 1 a 0 a lo largo del training. El modelo se va acostumbrando a sus errores.

---

## 2. Inferencia: Beam Search

En inferencia queremos $\hat{y} = \arg\max_y P(y \mid x)$, lo cual es **intratable** (explorar todo el espacio $|V|^{T_y}$). Aproximaciones:

### 2.1 Greedy decoding

$$\hat{y}_t = \arg\max_{w \in V} P(w \mid \hat{y}_{<t}, x)$$

Rapido pero **suboptimo**: elecciones locales no garantizan optimo global.

### 2.2 Beam search

Mantener las $B$ mejores hipothesis (beams). En cada paso:

1. Para cada hipothesis $h$ actual, expandir con cada $w \in V$: nuevo score = score($h$) + $\log P(w \mid h, x)$.
2. Seleccionar las $B$ con mayor score acumulado.
3. Hipothesis que terminan en `<EOS>` se guardan y se saca de beam activo.

Con $B = 1$: reduce a greedy. Con $B = V^{T_y}$: exhaustive. En la practica $B = 4$-$20$.

### 2.3 Length normalization

Score acumulado es log-prob sumado → sesgo a secuencias cortas. Normalizar:

$$\text{score}(y) = \frac{\log P(y \mid x)}{L(y)^\alpha}$$

con $\alpha \in [0.6, 0.8]$. Parametro ajustable.

### 2.4 Coverage penalty (Wu et al. 2016, Google GNMT)

Penalizar si el modelo no atiende a todas las posiciones de input:

$$\text{cp}(x, y) = \beta \sum_{i=1}^{T_x} \log \min\left(\sum_{t=1}^{T_y} \alpha_{t,i}, 1\right)$$

Score final: $\text{logprob}/L^\alpha + \text{cp}$.

---

# Parte II: Atencion Aditiva (Bahdanau)

---

## 3. Derivacion Completa

### 3.1 Encoder bidireccional

BiLSTM produce para cada posicion $j$:

$$\overrightarrow{h_j} = \text{LSTM}_{\rightarrow}(\overrightarrow{h_{j-1}}, x_j)$$
$$\overleftarrow{h_j} = \text{LSTM}_{\leftarrow}(\overleftarrow{h_{j+1}}, x_j)$$
$$h_j = [\overrightarrow{h_j}; \overleftarrow{h_j}] \in \mathbb{R}^{2n}$$

donde $n$ es el hidden size del LSTM. Los $h_j$ son las **anotaciones**.

### 3.2 Alignment model

Dado el estado del decoder $s_{i-1}$, calcular el **score** de alineamiento con cada anotacion:

$$e_{ij} = a(s_{i-1}, h_j) = V_a^T \tanh(W_a s_{i-1} + U_a h_j)$$

con parametros:
- $V_a \in \mathbb{R}^{n}$
- $W_a \in \mathbb{R}^{n \times n}$
- $U_a \in \mathbb{R}^{n \times 2n}$

Es una MLP de una capa oculta de $n$ unidades con activacion $\tanh$ y salida escalar.

### 3.3 Softmax normalization

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

Ahora $\sum_j \alpha_{ij} = 1$ y $\alpha_{ij} \geq 0$. Es una distribucion de probabilidad sobre posiciones fuente.

### 3.4 Context vector

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

El **valor esperado** de la anotacion bajo la distribucion $\alpha_{i \cdot}$. Puede verse como una **expected annotation**.

### 3.5 Decoder

$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

donde $f$ es una GRU o LSTM. El decoder ahora recibe tres inputs: estado previo, token previo y context adaptativo.

$$P(y_i \mid y_{<i}, x) = g(y_{i-1}, s_i, c_i)$$

con $g$ una capa de salida (maxout + softmax en el paper original, Dense + softmax en implementaciones modernas).

---

## 4. Por que Funciona

### 4.1 Eliminacion del cuello de botella

En Seq2Seq estandar:
$$P(y \mid x) = \prod_t P(y_t \mid y_{<t}, c) \quad \text{con } c \text{ fijo}$$

En atencion:
$$P(y \mid x) = \prod_t P(y_t \mid y_{<t}, c_t) \quad \text{con } c_t \text{ adaptativo}$$

El modelo puede **recuperar informacion** de posiciones especificas del encoder sin depender de que sobrevivan en un vector fijo.

### 4.2 Gradientes no atenuados

El gradient path entre $y_t$ y $x_j$ ahora pasa por $\alpha_{tj} \cdot h_j$ directamente, **sin atravesar** $T_x$ timesteps de LSTM. Esto mitiga vanishing gradient para dependencies largas.

### 4.3 Interpretabilidad

Las matrices $\alpha_{ij}$ son visualizables: en NMT EN-FR muestran diagonales claras cuando el orden es similar, y patrones de cross cuando hay reorderings (ej. adjetivos post-nominales en frances).

**Caveat**: estudios posteriores (Jain & Wallace 2019) muestran que attention weights **no siempre** reflejan contribucion causal -- usar con cautela como herramienta explicativa.

---

# Parte III: Variantes de Atencion

---

## 5. Atencion Multiplicativa / Dot-Product (Luong et al. 2015)

### 5.1 Dot-product simple

$$e_{ij} = s_i^T h_j$$

**Requiere**: $s_i$ y $h_j$ con la misma dimensionalidad. **Sin parametros extra**. Mas rapido que aditivo.

### 5.2 General bilinear

$$e_{ij} = s_i^T W_a h_j$$

con $W_a \in \mathbb{R}^{d_s \times d_h}$. Aprendible, intermedio entre aditivo y dot-product.

### 5.3 Scaled dot-product (Vaswani et al. 2017)

$$e_{ij} = \frac{s_i^T h_j}{\sqrt{d_k}}$$

### Por que escalar por $\sqrt{d_k}$?

Asumiendo $s_i, h_j$ independientes con componentes zero-mean y varianza 1, el producto punto $s_i^T h_j$ tiene:

- Media 0.
- **Varianza $d_k$** (suma de $d_k$ productos independientes).

Con $d_k$ grande (ej. 512), los scores crecen → softmax satura → gradientes cero. Escalar por $\sqrt{d_k}$ normaliza la varianza a 1.

### 5.4 Comparacion

| Variante | Formula | Parametros | Velocidad | Cuando usar |
|---|---|---|---|---|
| Aditivo (Bahdanau) | $V^T \tanh(W s + U h)$ | $V, W, U$ | Lento | Dims pequenas, mas expresivo |
| Dot-product | $s^T h$ | Ninguno | Rapido | Dims iguales, dims pequenas |
| Scaled dot-product | $s^T h / \sqrt{d_k}$ | Ninguno | Rapido | **Transformer, default moderno** |
| General bilinear | $s^T W h$ | $W$ | Medio | Dims diferentes |

---

## 6. Local vs Global Attention (Luong 2015)

### 6.1 Global attention

Atender sobre **todas** las posiciones del encoder. Es lo de Bahdanau y lo que se ha descrito hasta aqui.

### 6.2 Local attention

Atender solo sobre una **ventana** de posiciones alrededor de una posicion predicha $p_t$:

$$p_t = T_x \cdot \sigma(v_p^T \tanh(W_p s_t))$$

con $v_p, W_p$ aprendibles. La ventana es $[p_t - D, p_t + D]$ con $D$ fijo.

Dentro de la ventana, computar atencion soft. Fuera, $\alpha = 0$.

Ventajas:
- **Computacionalmente mas barato**: $O(D)$ en vez de $O(T_x)$.
- Reduce noise de posiciones irrelevantes.

Desventaja: requiere aprender $p_t$ bien; si falla, pierde informacion.

En la practica, **global attention domina** con el Transformer y secuencias <1K tokens; local se usa en models eficientes (Longformer, BigBird).

---

# Parte IV: Soft vs Hard Attention

---

## 7. Soft Attention

$$c_t = \sum_i \alpha_{ti} h_i, \quad \alpha_{ti} = \text{softmax}(\text{score}(s_t, h_i))$$

Propiedades:
- **Diferenciable** end-to-end.
- Gradientes via backprop estandar.
- Computacionalmente $O(T_x)$ por paso.

Problema: **atiende a todas las posiciones**. Si el modelo deberia enfocarse en una sola, soft attention "smearea" con atencion pequena pero distinta de cero en todas.

---

## 8. Hard Attention (Xu et al. 2015)

Muestrear una posicion $s_t \sim \text{Multinoulli}(\alpha_t)$ y atender solo a esa:

$$c_t = h_{s_t}$$

Problema: muestreo no es diferenciable. Soluciones:

### 8.1 REINFORCE

Policy gradient: considerar $s_t$ como una accion, $-\log P(y \mid x)$ como reward negativo. Estimator:

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{1}{N} \sum_n \left[ \frac{\partial \log p(y \mid \tilde{s}^n, a)}{\partial \theta} + \lambda_r (\log p(y \mid \tilde{s}^n, a) - b) \frac{\partial \log p(\tilde{s}^n \mid a)}{\partial \theta} \right]$$

con baseline $b$ para reducir varianza. **Frecuentemente inestable**.

### 8.2 Variational lower bound

Tratar las locaciones como variables latentes, optimizar ELBO:

$$\log p(y \mid a) \geq \sum_s p(s \mid a) \log p(y \mid s, a)$$

### 8.3 Gumbel-Softmax (Jang, Gu, Poole 2017)

Aproximar la categorical discretizada con una softmax continua reparametrizable. Permite training end-to-end con temperatures bajas → cerca de hard, pero diferenciable.

---

## 9. Soft vs Hard: Comparacion

| Aspecto | Soft | Hard |
|---|---|---|
| Diferenciable | Si | No (requiere REINFORCE/Gumbel) |
| Training | Estable | Inestable |
| Memoria | $O(T_x)$ computo | $O(1)$ atendido (pero muestreo ruidoso) |
| Interpretabilidad | "Borrosa" | Discreta (facil de visualizar) |
| Performance | Mejor en la mayoria de tareas | Marginalmente mejor en algunos casos de vision |

**Conclusion practica**: usar **soft** salvo razon especifica para hard (ej. deployment con constraints estrictos de latencia y memoria).

### 9.1 Doubly stochastic attention (Xu 2015)

Regularizacion para soft attention que fuerza: $\sum_t \alpha_{ti} \approx 1$ para cada posicion $i$. Objetivo:

$$L_d = -\log P(y \mid x) + \lambda \sum_i \left(1 - \sum_t \alpha_{ti}\right)^2$$

Empuja al modelo a "visitar" todas las posiciones de la fuente al menos una vez a lo largo de la generacion. En image captioning: mejora cobertura de la imagen.

---

# Parte V: Pointer-Generator y Coverage

---

## 10. Pointer-Generator (See, Liu, Manning 2017)

### 10.1 El problema

Modelos seq2seq puramente generativos para summarization tienen 3 problemas:

1. Errores factuales (detalles incorrectos).
2. OOV words (nombres, cifras) → `[UNK]`.
3. Repetition (mismas frases varias veces).

### 10.2 Pointer-generator switch

Probabilidad aprendida de **generar** vs **copiar**:

$$p_{\text{gen}} = \sigma(w_{h^*}^T h_t^* + w_s^T s_t + w_x^T x_t + b_{\text{ptr}})$$

Distribucion final sobre vocabulario extendido:

$$P(w) = p_{\text{gen}} \cdot P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \cdot \sum_{i: w_i = w} \alpha_i^t$$

Si $w$ es OOV: $P_{\text{vocab}}(w) = 0$, pero se puede producir via el segundo termino (copying). **Resuelve OOV** y reduce errores factuales.

### 10.3 Coverage vector

$$c^t = \sum_{t'=0}^{t-1} \alpha^{t'}$$

(suma de atenciones previas). Pasar a attention:

$$e_i^t = v^T \tanh(W_h h_i + W_s s_t + w_c c_i^t + b_{\text{attn}})$$

El termino $w_c c_i^t$ permite al modelo "ver" cuanto ha atendido cada posicion y ajustar.

### 10.4 Coverage loss

$$\text{covloss}_t = \sum_i \min(\alpha_i^t, c_i^t)$$

Acotada por $\sum_i \alpha_i^t = 1$. Penaliza atender a posiciones ya cubiertas. Loss total:

$$L = -\sum_t \log P(w_t^*) + \lambda \sum_t \text{covloss}_t$$

con $\lambda = 1$.

Resultados: coverage **elimina casi por completo** la repeticion.

---

# Parte VI: Evolucion al Transformer

---

## 11. Limitaciones de Seq2Seq + Attention con RNNs

1. **Secuencial**: el decoder RNN no se paraleliza en el tiempo (cada $s_t$ depende de $s_{t-1}$).
2. **Encoder BiLSTM tampoco paralelizable** dentro de una misma secuencia.
3. **Atencion sobre encoder solamente**: no hay atencion dentro del encoder ni dentro del decoder (entre sus propias posiciones).

---

## 12. Self-Attention

Generalizacion: atencion donde query, key y value vienen **de la misma secuencia**.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

con:
- $Q = X W_Q \in \mathbb{R}^{T \times d_k}$ (queries)
- $K = X W_K \in \mathbb{R}^{T \times d_k}$ (keys)
- $V = X W_V \in \mathbb{R}^{T \times d_v}$ (values)

En una sola multiplicacion matricial computamos **todas** las atenciones entre todas las posiciones. $O(T^2)$ pero completamente **paralelizable**.

---

## 13. Multi-Head Attention

En vez de una sola atencion, **$h$ cabezales paralelos** con subespacios distintos:

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

Permite al modelo atender a **distintos patrones** simultaneamente (semantica, sintaxis, co-referencia, etc.).

---

## 14. Transformer Architecture (Vaswani 2017)

Encoder: $N$ layers, cada una con:

1. **Self-attention** (multi-head).
2. **FFN** (feedforward): 2 Dense layers con ReLU.
3. **Residual connections + LayerNorm** alrededor de cada sub-layer.

Decoder: $N$ layers, cada una con:

1. **Masked self-attention** (no puede ver futuro).
2. **Cross-attention** (atiende al output del encoder) -- esto es **Bahdanau attention** generalizada.
3. **FFN**.
4. Residual + LayerNorm.

La atencion de Bahdanau (2015) es exactamente la **cross-attention** del decoder del Transformer. Sin esa idea, el Transformer no existiria.

---

## 15. De Bahdanau al LLM

| Ano | Modelo | Innovacion |
|---|---|---|
| 2014 | Sutskever Seq2Seq | Encoder-decoder LSTM |
| 2014 | Cho GRU + encoder-decoder | Simplified RNN |
| **2015** | **Bahdanau Attention** | **Adaptive context vector** |
| 2015 | Luong attention | Dot-product, local vs global |
| 2016 | Google GNMT | Production-grade NMT |
| **2017** | **Vaswani Transformer** | **No RNN, solo attention** |
| 2018 | BERT | Encoder-only, MLM pretraining |
| 2019 | GPT-2 | Decoder-only, scaled autoregressive |
| 2020 | GPT-3 | 175B params, in-context learning |
| 2022+ | LLMs modernos | Claude, GPT-4, Gemini |

Todos los LLMs modernos son **descendientes directos** de la idea de atencion de Bahdanau 2015.

---

# Resumen Ejecutivo

1. **Seq2Seq** factoriza $P(y \mid x)$ autoregresivamente y se entrena con teacher forcing + cross-entropy.
2. **Beam search** + length normalization es el estandar para inferencia.
3. **Bahdanau attention** introduce un context vector adaptativo $c_t = \sum_j \alpha_{tj} h_j$ con $\alpha$ computada via MLP + softmax.
4. Variantes: **additive** (Bahdanau), **dot-product** (Luong), **scaled dot-product** (Transformer).
5. **Soft attention** es diferenciable y domina; **hard attention** requiere REINFORCE.
6. **Doubly stochastic** regulariza para cobertura en image captioning.
7. **Pointer-generator** (See 2017) combina generacion y copying para resolver OOV; **coverage** elimina repeticion.
8. **Self-attention** y el **Transformer** (Vaswani 2017) son la evolucion natural: la atencion reemplaza completamente a la recurrencia.

---

## Referencias

- Sutskever, Vinyals, Le (2014). [Sequence to Sequence Learning with Neural Networks](/papers/seq2seq-sutskever-2014). *NeurIPS*.
- Bahdanau, Cho, Bengio (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](/papers/bahdanau-attention-2015). *ICLR*.
- Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation. *EMNLP*.
- Xu et al. (2015). [Show, Attend and Tell](/papers/show-attend-tell-xu-2015). *ICML*.
- See, Liu, Manning (2017). [Get To The Point: Summarization with Pointer-Generator Networks](/papers/pointer-generator-see-2017). *ACL*.
- Anderson et al. (2018). [Bottom-Up and Top-Down Attention](/papers/bottom-up-attention-anderson-2018). *CVPR*.
- Vaswani et al. (2017). Attention Is All You Need. *NeurIPS*.
- Bengio et al. (2015). Scheduled Sampling for Sequence Prediction. *NeurIPS*.
- Jain & Wallace (2019). Attention is not Explanation. *NAACL*.

Volver a [Teoria](teoria) | Hub de la [Clase 13](/clases/clase-13).
