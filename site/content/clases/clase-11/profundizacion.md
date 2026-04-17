---
title: "Profundizacion - BPTT, Vanishing Gradient y LSTM Math"
weight: 20
math: true
---

> Este documento profundiza en los fundamentos matematicos detras de los conceptos cubiertos en la Clase 11.
> Cubre la derivacion completa de Backpropagation Through Time, el analisis formal del vanishing/exploding gradient
> de Pascanu, Mikolov y Bengio (2013), las matematicas internas de LSTM y GRU, y por que las arquitecturas
> con compuertas resuelven el problema fundamental de las RNNs vanilla.

---

# Parte I: Backpropagation Through Time

---

## 1. Setup y Notacion

Considera una RNN vanilla con parametros $\theta = \{W_{xh}, W_{hh}, W_{hy}, b_h, b_y\}$. Para una secuencia de entrada $x_1, \ldots, x_T$ con etiquetas $\hat{y}_1, \ldots, \hat{y}_T$:

$$
\begin{aligned}
z_t &= W_{hh} \, h_{t-1} + W_{xh} \, x_t + b_h \\
h_t &= \tanh(z_t) \\
o_t &= W_{hy} \, h_t + b_y \\
\hat{p}_t &= \text{softmax}(o_t) \\
L_t &= -\hat{y}_t^T \log \hat{p}_t \\
\mathcal{L} &= \sum_{t=1}^{T} L_t
\end{aligned}
$$

donde $h_0$ es el estado inicial (cero o aprendible) y $L_t$ es la cross-entropy en el paso $t$.

---

## 2. Gradientes Locales

### 2.1 Gradiente respecto a la salida $o_t$

Para softmax + cross-entropy, el gradiente local es la diferencia entre prediccion y target:

$$\delta_t^o = \frac{\partial L_t}{\partial o_t} = \hat{p}_t - \hat{y}_t$$

### 2.2 Gradiente respecto a $W_{hy}$

$$\frac{\partial L_t}{\partial W_{hy}} = \delta_t^o \cdot h_t^T$$

Sumado sobre todos los pasos:

$$\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T} (\hat{p}_t - \hat{y}_t) \cdot h_t^T$$

### 2.3 Gradiente respecto a $h_t$

El estado $h_t$ contribuye a $L_t$ directamente y a todos los $L_{t+1}, \ldots, L_T$ via la recurrencia. Por la regla de la cadena:

$$\delta_t^h = \frac{\partial \mathcal{L}}{\partial h_t} = W_{hy}^T \delta_t^o + W_{hh}^T \, \text{diag}(1 - h_{t+1}^2) \cdot \delta_{t+1}^h$$

donde usamos $\tanh'(z) = 1 - \tanh^2(z)$, y la recurrencia se calcula desde $t = T$ hacia atras con $\delta_{T+1}^h = 0$.

### 2.4 Gradiente respecto a $z_t$

$$\delta_t^z = \frac{\partial \mathcal{L}}{\partial z_t} = \delta_t^h \odot (1 - h_t^2)$$

donde $\odot$ es producto elementwise.

### 2.5 Gradientes respecto a $W_{xh}$ y $W_{hh}$

Como ambas matrices se usan en cada paso:

$$\frac{\partial \mathcal{L}}{\partial W_{xh}} = \sum_{t=1}^{T} \delta_t^z \cdot x_t^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t^z \cdot h_{t-1}^T$$

---

## 3. Pseudocodigo del BPTT

```python
# Forward
h[0] = 0
for t in range(1, T+1):
    z[t] = W_hh @ h[t-1] + W_xh @ x[t] + b_h
    h[t] = tanh(z[t])
    o[t] = W_hy @ h[t] + b_y
    p[t] = softmax(o[t])
    L[t] = -y[t] @ log(p[t])
loss = sum(L)

# Backward
dh_next = 0
dW_hh = dW_xh = dW_hy = 0
db_h = db_y = 0

for t in reversed(range(1, T+1)):
    do = p[t] - y[t]                        # δ^o
    dW_hy += do @ h[t].T
    db_y  += do

    dh = W_hy.T @ do + dh_next              # δ^h
    dz = dh * (1 - h[t]**2)                 # δ^z

    dW_xh += dz @ x[t].T
    dW_hh += dz @ h[t-1].T
    db_h  += dz

    dh_next = W_hh.T @ dz                   # propagar al paso t-1
```

Memoria: $O(T \cdot d_h)$ para almacenar todos los $h_t$ y $z_t$. Tiempo: $O(T \cdot (d_h^2 + d_h d_x))$.

---

# Parte II: Analisis Formal de Vanishing y Exploding Gradient

---

## 4. El Producto de Jacobianas

Reescribiendo la cadena de gradientes hacia atras, el "transporte temporal" desde el paso $t$ al paso $k$ ($k < t$) es:

$$\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$$

Cada factor es:

$$\frac{\partial h_j}{\partial h_{j-1}} = W_{hh}^T \cdot \text{diag}(\sigma'(z_j))$$

El gradiente de $\mathcal{L}_t$ respecto a $h_k$ es entonces:

$$\frac{\partial L_t}{\partial h_k} = \frac{\partial L_t}{\partial h_t} \cdot \prod_{j=k+1}^{t} W_{hh}^T \, \text{diag}(\sigma'(z_j))$$

---

## 5. Cota Superior (Pascanu et al. 2013)

Sea $\| \cdot \|$ una norma matricial compatible con la norma vectorial. Entonces:

$$\left\| \prod_{j=k+1}^{t} W_{hh}^T \, \text{diag}(\sigma'(z_j)) \right\| \leq \prod_{j=k+1}^{t} \| W_{hh}^T \| \cdot \| \text{diag}(\sigma'(z_j)) \|$$

Sea $\gamma$ una cota de $|\sigma'|$:

- $\gamma = 1$ para $\tanh$ (maximo en $z = 0$).
- $\gamma = 1/4$ para sigmoide.
- $\gamma = 1$ para ReLU (cuando activa).

Y sea $\lambda_1$ el **mayor valor singular** de $W_{hh}$:

$$\left\| \frac{\partial h_t}{\partial h_k} \right\| \leq (\lambda_1 \gamma)^{t-k}$$

Esta es la **cota fundamental**:

- Si $\lambda_1 \gamma < 1$ → la cota decae exponencialmente → **vanishing**.
- Si $\lambda_1 \gamma > 1$ → la cota crece exponencialmente → **exploding posible**.

{{< concept-alert type="clave" >}}
**Resultados de Pascanu et al. (2013)**:

- $\lambda_1 < \frac{1}{\gamma}$ es **suficiente** para vanishing gradient.
- $\lambda_1 > \frac{1}{\gamma}$ es **necesario** para exploding gradient.

Para $\tanh$, esto significa que **vanishing es seguro** cuando los valores singulares de $W_{hh}$ son menores que 1, y **exploding es posible** cuando alguno excede 1.
{{< /concept-alert >}}

### 5.1 Ejemplo numerico

Sea $W_{hh}$ una matriz $2 \times 2$ con valores singulares $\sigma_1 = 0.9, \sigma_2 = 0.5$, y activacion tanh ($\gamma = 1$).

Despues de $t - k = 100$ pasos, la magnitud del gradiente esta acotada por:

$$\| \nabla \| \leq 0.9^{100} \approx 2.66 \times 10^{-5}$$

El gradiente que volveria a un input 100 pasos atras es esencialmente cero. La RNN **no puede aprender** dependencias en esa ventana.

Si en cambio $\sigma_1 = 1.1$:

$$\| \nabla \| \leq 1.1^{100} \approx 13780$$

El gradiente explota -- las actualizaciones de pesos seran catastroficas y la perdida diverge.

---

## 6. Vista de Sistemas Dinamicos

Pascanu et al. interpretan estos fenomenos desde la teoria de **sistemas dinamicos**:

- El estado $h_t$ converge (en ausencia de input) a un **atractor** del mapa $h \mapsto F(h) = W_{hh}^T \sigma(W_{hh} h)$.
- Los **bordes entre cuencas de atraccion** son **bifurcaciones**: pequenos cambios en los pesos cambian la estructura de atractores.
- Cruzar una bifurcacion produce **discontinuidades grandes** en el estado final, lo que se traduce en **paredes empinadas** en la superficie de error.

Una pared en la loss landscape se ve asi:

```
   error
    │      ╱╲
    │     ╱  ╲    pared empinada
    │    ╱    ╲
    │   ╱      ╲___ valle
    │__/           ___________
    └─────────────────────→ θ
```

Un paso de SGD normal en una pared empinada **salta lejos del valle**. Esto explica por que el entrenamiento "se rompe" subitamente cuando el optimizador visita una region critica.

---

## 7. Solucion: Gradient Norm Clipping

La solucion practica de Pascanu para exploding gradients:

```text
g ← ∇θ L
if ||g||₂ ≥ τ:
    g ← (τ / ||g||₂) · g
θ ← θ - η · g
```

donde $\tau$ es el **threshold** (tipicamente 1 a 10).

### Justificacion

Sea $g$ el gradiente. Si la pared apunta correctamente lejos de la pared (ortogonal a $\nabla L$ apuntando al valle), simplemente acortar el paso permite mantenerse en la direccion correcta sin saltar fuera.

Crucialmente, **clipping preserva la direccion**, solo escala la magnitud. Es **invariante a la escala** del gradiente, lo que lo hace mas robusto que reducir el learning rate (que afectaria todas las direcciones uniformemente).

### Variantes

| Variante | Operacion |
|---|---|
| **Clip by norm** | $g \leftarrow \min(1, \tau / \|g\|) \cdot g$ |
| **Clip by value** | $g \leftarrow \text{clip}(g, -\tau, \tau)$ elementwise |
| **Clip by global norm** | Norma sobre todos los gradientes concatenados |

En la practica, **clip by norm** o **clip by global norm** son los mas usados.

---

# Parte III: Matematica Interna de LSTM

---

## 8. Constant Error Carrousel (CEC)

La idea geminal de Hochreiter & Schmidhuber (1997): una unidad lineal con auto-conexion de **peso fijo 1.0**:

$$s_j(t) = s_j(t-1) + \text{input}_j(t)$$

La derivada respecto al estado previo es **identidad**:

$$\frac{\partial s_j(t)}{\partial s_j(t-1)} = 1$$

Backpropagar a traves de $T$ pasos no decaira el gradiente -- el "carrousel" lo preserva intacto.

Pero sin control, el CEC sufre de:

1. **Conflicto de escritura**: la misma conexion debe a veces almacenar info y a veces ignorarla.
2. **Conflicto de lectura**: la salida de la celda perturba a otras unidades aun cuando no es relevante.

Solucion: **compuertas multiplicativas**.

---

## 9. LSTM Moderna (con Forget Gate)

La LSTM original tenia solo input/output gates. Gers, Schmidhuber & Cummins (2000) anadieron el forget gate. Esta es la version usada hoy:

$$
\begin{aligned}
i_t &= \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f) \\
g_t &= \tanh(W_{xg} x_t + W_{hg} h_{t-1} + b_g) \\
o_t &= \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

Donde $\sigma$ es sigmoide, $\odot$ producto elementwise.

### 9.1 Por que evita vanishing gradient

El gradiente del cell state hacia atras es:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t \quad \text{(diagonal: cada componente es un escalar)}$$

A traves de $T$ pasos:

$$\frac{\partial c_t}{\partial c_{t-T}} = \prod_{k=t-T+1}^{t} f_k$$

Si $f_k \approx 1$ para todos los pasos relevantes (la red ha aprendido a "recordar"), el gradiente fluye sin atenuacion. **No hay producto matricial**, no hay valores singulares que decaigan exponencialmente.

### 9.2 Truco practico: inicializar $b_f$ positivo

Inicializar el bias del forget gate a un valor positivo (ej. $b_f = 1.0$) hace que al inicio del entrenamiento:

$$f_t \approx \sigma(b_f) \approx 0.73 \quad \text{(con } b_f = 1\text{)}$$

la celda recuerda por defecto. Mejora significativamente la convergencia, especialmente en tareas con dependencias largas.

```python
# PyTorch
for name, param in lstm.named_parameters():
    if 'bias' in name:
        n = param.size(0)
        # PyTorch: gates en orden (i, f, g, o)
        param.data[n//4 : n//2].fill_(1.0)
```

### 9.3 Costo computacional

Por celda con dimension $d$ y entrada $d_x$:

$$\text{params}_{LSTM} = 4 \cdot (d \cdot d_x + d \cdot d + d) = 4d(d + d_x + 1)$$

vs. RNN vanilla:

$$\text{params}_{RNN} = d \cdot d_x + d \cdot d + d = d(d + d_x + 1)$$

LSTM tiene **4 veces** mas parametros.

---

## 10. GRU (Gated Recurrent Unit)

Cho et al. (2014) propusieron una arquitectura mas simple con solo 2 compuertas y 1 estado:

$$
\begin{aligned}
r_t &= \sigma(W_r x_t + U_r h_{t-1}) \quad \text{(reset gate)} \\
z_t &= \sigma(W_z x_t + U_z h_{t-1}) \quad \text{(update gate)} \\
\tilde{h}_t &= \tanh(W x_t + U(r_t \odot h_{t-1})) \quad \text{(candidate)} \\
h_t &= z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t \quad \text{(state)}
\end{aligned}
$$

### 10.1 Por que evita vanishing gradient

$$\frac{\partial h_t}{\partial h_{t-1}} \approx z_t \cdot I + (1 - z_t) \cdot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

Cuando $z_t \approx 1$, $\frac{\partial h_t}{\partial h_{t-1}} \approx I$. El gradiente fluye casi sin atenuacion.

### 10.2 Comparacion con LSTM

| Aspecto | LSTM | GRU |
|---|---|---|
| Compuertas | 4 (i, f, g, o) | 3 (r, z, $\tilde h$) |
| Estados | 2 ($c_t, h_t$) | 1 ($h_t$) |
| Parametros | $4d(d + d_x + 1)$ | $3d(d + d_x + 1)$ |
| Velocidad | Base | ~25% mas rapido |
| Performance | Marginalmente mejor en tareas grandes | Marginalmente mejor en tareas pequenas |

En la practica, la diferencia es pequena. **GRU es mas comun como default** por simplicidad.

---

# Parte IV: Truncated BPTT

---

## 11. El Problema con Secuencias Muy Largas

BPTT full requiere almacenar **toda la secuencia** en memoria para el backward pass. Para secuencias de millones de tokens (libros completos, dominios largos), esto es prohibitivo.

---

## 12. Truncated BPTT

Idea: dividir la secuencia en **chunks** de longitud fija $K$. Forward y backward solo dentro del chunk; al cambiar de chunk, **detach** el estado oculto para no propagar gradientes hacia atras.

```python
hidden = None
for chunk in chunks(sequence, chunk_size=K):
    if hidden is not None:
        hidden = hidden.detach()  # romper el grafo
    output, hidden = model(chunk, hidden)
    loss = criterion(output, targets[chunk_idx])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    optimizer.zero_grad()
```

### 12.1 Trade-off

| Pro | Contra |
|---|---|
| Memoria $O(K)$ en vez de $O(T)$ | No aprende dependencias > $K$ |
| Mas updates por epoca | Sesgo: gradientes truncados |
| Practico para secuencias largas | Perdida de informacion de gradiente lejano |

Tipicamente $K = 35$ a $200$ para language modeling. Para tareas con dependencias muy largas (ej. comprension de novela), truncated BPTT no basta -- se requieren atencion o memoria externa.

---

# Parte V: Encoder-Decoder y Seq2Seq

---

## 13. Arquitectura Encoder-Decoder (Cho 2014, Sutskever 2014)

### 13.1 Encoder

Procesa la secuencia fuente $x_1, \ldots, x_T$ y produce un **vector de contexto** $c$:

$$c = h_T^{\text{enc}}$$

Donde $h_T^{\text{enc}}$ es el estado final del encoder LSTM.

### 13.2 Decoder

Genera la secuencia objetivo $y_1, \ldots, y_{T'}$ condicionado en $c$:

$$
\begin{aligned}
h_t^{\text{dec}} &= f(h_{t-1}^{\text{dec}}, y_{t-1}, c) \\
p(y_t \mid y_{<t}, c) &= g(h_t^{\text{dec}}, y_{t-1}, c)
\end{aligned}
$$

### 13.3 Probabilidad conjunta

$$p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_{t=1}^{T'} p(y_t \mid y_{<t}, c)$$

Loss = negative log-likelihood:

$$L = -\sum_{t=1}^{T'} \log p(y_t \mid y_{<t}, c)$$

### 13.4 Trucos de Sutskever 2014

- **Reverse source**: alimentar la oracion fuente al reves (`C B A` en vez de `A B C`). Mejora BLEU de 25.9 → 30.6.
- **4 capas LSTM** de 1000 celdas: cada capa adicional reduce perplexity ~10%.
- **Beam search** con beam size 12 en inferencia.
- **Gradient clipping** norma 5.

### 13.5 Cuello de botella del vector $c$

Comprimir toda la oracion fuente en un vector $c$ de dimension fija (8000 en Sutskever) limita el rendimiento en oraciones largas. La solucion: **mecanismo de atencion** (Bahdanau, Cho, Bengio 2014), que permite al decoder mirar a todos los estados del encoder en cada paso. Esto evolucionara hacia los **Transformers** (Vaswani 2017).

---

# Parte VI: Image Captioning con CNN + LSTM

---

## 14. Modelo NIC (Neural Image Caption)

Vinyals et al. (2015) extendieron seq2seq a vision-language: el "encoder" es un **CNN preentrenado** en ImageNet, el decoder es un LSTM.

$$
\begin{aligned}
x_{-1} &= \text{CNN}(I) \quad \text{(embedding visual, 4096-dim)} \\
x_t &= W_e \, S_t, \quad t \in \{0, \ldots, N-1\} \\
p_{t+1} &= \text{LSTM}(x_t) \\
L(I, S) &= -\sum_{t=1}^{N} \log p_t(S_t)
\end{aligned}
$$

Decisiones clave:

- La imagen se introduce **una sola vez** ($t = -1$). Alimentarla en cada paso empeora resultados.
- **Beam search** con beam 20.
- CNN preentrenado en ImageNet (transfer learning), **fine-tuned** durante el entrenamiento.

Resultados en Pascal: BLEU-1 = 59 (NIC) vs. 25 (state-of-art previo) vs. 69 (humano).

---

# Resumen Ejecutivo

1. **BPTT** = backprop sobre la red recurrente desplegada. Computacionalmente $O(T \cdot d^2)$, memoria $O(T \cdot d)$.
2. El gradiente recurrente es producto de Jacobianas $W_{hh}^T \, \text{diag}(\sigma')$. Crece o decae exponencialmente segun el mayor valor singular de $W_{hh}$.
3. **Pascanu 2013**: condicion suficiente para vanishing $\lambda_1 < 1/\gamma$, condicion necesaria para exploding $\lambda_1 > 1/\gamma$.
4. **Solucion universal para exploding**: gradient norm clipping con threshold ~5.
5. **LSTM** introduce un cell state $c_t$ con flujo aditivo $c_t = f \odot c_{t-1} + i \odot g$. La derivada $\partial c_t / \partial c_{t-1} = f$ es elementwise, no matricial.
6. **GRU** simplifica a 2 compuertas y 1 estado, con propiedades similares pero menos parametros.
7. **Encoder-decoder** (Cho 2014, Sutskever 2014) generaliza a cualquier mapeo seq2seq. Cuello de botella en el vector $c$ resuelto despues con atencion (Bahdanau 2014) y Transformer (Vaswani 2017).
8. **NIC** (Vinyals 2015) muestra que el patron CNN+LSTM funciona para vision-language.

---

## Referencias

- Hochreiter, Schmidhuber (1997). [Long Short-Term Memory](/papers/lstm-hochreiter-1997). *Neural Computation*.
- Pascanu, Mikolov, Bengio (2013). [On the difficulty of training Recurrent Neural Networks](/papers/difficulty-training-rnns-pascanu-2013). *ICML*.
- Cho et al. (2014). [Learning Phrase Representations using RNN Encoder-Decoder](/papers/gru-cho-2014). *EMNLP*.
- Sutskever, Vinyals, Le (2014). [Sequence to Sequence Learning with Neural Networks](/papers/seq2seq-sutskever-2014). *NeurIPS*.
- Vinyals, Toshev, Bengio, Erhan (2015). [Show and Tell: A Neural Image Caption Generator](/papers/show-and-tell-vinyals-2015). *CVPR*.
- Gers, Schmidhuber, Cummins (2000). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*.
- Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*.

Volver a [Teoria](teoria) | Hub de la [Clase 11](/clases/clase-11).
