---
title: "Profundizacion - MIT 6.S191 (2026): RNNs + Transformers"
weight: 20
math: true
---

> Material complementario al video de Ava Amini correspondiente al Lecture 2 de la edicion 2026
> de MIT 6.S191. Esta nota acompana al lecture y se concibe como **bisagra**
> entre los apuntes UC ya disponibles (clases 11 y 13) y el video MIT mas reciente.
> No re-deriva contenido cubierto en otras paginas: cruza enlaces y solo profundiza
> en aquello que la edicion 2026 introduce o reordena, especialmente la mitad
> dedicada a Transformers, multi-head attention y positional encoding.

---

## 1. Contexto breve

**MIT 6.S191 -- Introduction to Deep Learning** es el curso introductorio del MIT que
desde 2017 dictan Alexander Amini y Ava Amini (anteriormente Soleimany) cada
*Independent Activities Period* (IAP, enero). En la edicion **2026** el curso fue
significativamente reestructurado para reflejar la realidad post-LLM: el Lecture 2
ya no es solo sobre RNNs, sino que se titula explicitamente **"Recurrent Neural
Networks, Transformers, and Attention"** y es presentado por **Ava Amini**. Es decir,
toda la cadena conceptual -- desde la motivacion del modelado secuencial hasta el
Transformer completo de Vaswani et al. (2017) -- se cubre en una sola hora de
clase. Esto contrasta con la edicion **2020** (la que documentamos en
[/videos/mit-6s191-rnn/profundizacion/](/videos/mit-6s191-rnn/profundizacion/)),
donde los Transformers solo aparecian aludidos al final como sucesor.

La consecuencia pedagogica es que el lecture 2026 dedica aproximadamente la primera
mitad a RNNs, vanishing/exploding gradient y LSTM (material que ya cubrimos en
[Clase 11](/clases/clase-11/teoria/) y su [profundizacion](/clases/clase-11/profundizacion/)),
y la segunda mitad a self-attention, multi-head attention, positional encoding y
arquitectura Transformer (que se solapa parcialmente con [Clase 13](/clases/clase-13/teoria/)
pero introduce explicitamente la formalizacion query/key/value y los detalles
ingenieriles del Transformer). Este documento se concentra en la parte que la
clase UC aun no formaliza con suficiente detalle: las derivaciones que justifican
las elecciones especificas del Transformer.

---

## 2. Papers seminales

A continuacion sintetizamos los papers fundamentales que el lecture 2026 atraviesa,
con la formula o insight central de cada uno y enlace a la pagina de paper en el
sitio (cuando existe) o cita directa a arxiv.

### Bengio, Simard & Frasconi (1994) -- "Learning Long-Term Dependencies with Gradient Descent is Difficult"

Cita: *IEEE Transactions on Neural Networks*, vol. 5, no. 2, pp. 157-166, 1994.
Disponible en [https://ieeexplore.ieee.org/document/279181](https://ieeexplore.ieee.org/document/279181).

Es el primer analisis formal del **problema del gradiente desvaneciente** en redes
recurrentes. Bengio et al. demuestran que para que un RNN almacene informacion
robustamente durante muchos pasos, los autovalores de la matriz jacobiana
$\partial h_t / \partial h_{t-1}$ deben caer dentro del disco unitario, lo que
fuerza al producto $\prod_t \partial h_t / \partial h_{t-1}$ a contraerse
exponencialmente -- es decir, gradientes que se desvanecen. Es un resultado de
*imposibilidad parcial*: no es que sea dificil entrenar RNNs largas, es que el
diseno mismo introduce una tension entre estabilidad y memoria. Este insight
motivara, anos despues, la cell state aditiva de LSTM como solucion estructural.

### Pascanu, Mikolov & Bengio (2013) -- "On the Difficulty of Training Recurrent Neural Networks"

Pagina UC: [/papers/difficulty-training-rnns-pascanu-2013/](/papers/difficulty-training-rnns-pascanu-2013/).

Reformula el resultado de 1994 con herramientas modernas (analisis espectral con
norma de matriz, SVD, valores singulares maximos) y propone dos remedios
practicos: **gradient clipping** para exploding gradient (re-escalar el gradiente
si su norma supera un umbral $\tau$) y **regularizacion** que penaliza variaciones
abruptas en la norma del estado oculto. El bound formal $\|\partial h_T / \partial h_t\|
\leq \sigma_{\max}(W_{hh})^{T-t} \cdot \prod \|f'\|$ se trabaja en
[Clase 11 profundizacion](/clases/clase-11/profundizacion/).

### Hochreiter & Schmidhuber (1997) -- "Long Short-Term Memory"

Pagina UC: [/papers/lstm-hochreiter-1997/](/papers/lstm-hochreiter-1997/).

Introduce LSTM como solucion al vanishing gradient: la **constant error carousel**
mediante una cell state $c_t$ con actualizacion aditiva (no multiplicativa)
$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. La derivada
$\partial c_t / \partial c_{t-1} = f_t$ permite al gradiente fluir sin atenuarse
mientras $f_t$ permanezca cerca de 1. Las matematicas completas estan en
[Clase 11 profundizacion seccion 6](/clases/clase-11/profundizacion/).

### Sutskever, Vinyals & Le (2014) -- "Sequence to Sequence Learning with Neural Networks"

Pagina UC: [/papers/seq2seq-sutskever-2014/](/papers/seq2seq-sutskever-2014/).

Define el patron **encoder-decoder** con dos LSTMs apiladas: la encoder consume
$x_1, \ldots, x_{T_x}$ y produce un context vector $c$ (su ultimo hidden state);
la decoder genera $y_1, \ldots, y_{T_y}$ condicionado en $c$. Establece estado del
arte en WMT 2014 ingles-frances con BLEU 34.8. Su limitacion -- el **cuello de
botella** del context vector unico -- motiva directamente Bahdanau 2015.

### Bahdanau, Cho & Bengio (2015) -- "Neural Machine Translation by Jointly Learning to Align and Translate"

Pagina UC: [/papers/bahdanau-attention-2015/](/papers/bahdanau-attention-2015/).

Reemplaza el context fijo por un **context adaptativo** $c_t = \sum_j \alpha_{tj} h_j$
con pesos calculados por una MLP de alineamiento. Es el origen conceptual de
toda la familia attention. La derivacion completa, incluido el alignment model
$e_{tj} = V_a^T \tanh(W_a s_{t-1} + U_a h_j)$, esta en
[Clase 13 profundizacion seccion 3](/clases/clase-13/profundizacion/).

### Luong, Pham & Manning (2015) -- "Effective Approaches to Attention-based Neural Machine Translation"

Cita: arXiv [1508.04025](https://arxiv.org/abs/1508.04025). EMNLP 2015.

Simplifica Bahdanau y propone tres scores alternativos: **dot-product** ($s_t^T h_j$),
**general bilinear** ($s_t^T W_a h_j$) y **concat** (similar a Bahdanau). Tambien
introduce **local attention**: en vez de atender sobre todas las posiciones,
predecir una posicion de pivote $p_t$ y restringir a una ventana
$[p_t - D, p_t + D]$. El resultado es un sistema NMT mas simple, mas rapido y
con BLEU comparable a Bahdanau. La elegancia del dot-product -- sin parametros
extra -- preanuncia el Transformer.

### Vaswani et al. (2017) -- "Attention Is All You Need"

Cita: arXiv [1706.03762](https://arxiv.org/abs/1706.03762). NeurIPS 2017.

Demuestra que **se puede prescindir totalmente de la recurrencia**: una arquitectura
construida unicamente con self-attention multi-cabezal, feedforward por posicion,
residuales y layer normalization supera todos los baselines anteriores en
traduccion. Introduce tres componentes que explicaremos en la seccion 3:
**scaled dot-product attention**, **multi-head attention** y **sinusoidal positional
encoding**. La eliminacion de la recurrencia permite paralelismo total dentro de
la secuencia y abre la puerta al escalamiento masivo que culminara en BERT, GPT y
los LLMs modernos.

---

## 3. Derivaciones formales

Esta seccion cubre las matematicas que la edicion 2026 del lecture introduce de
forma intuitiva pero que vale la pena formalizar. Para BPTT, gradientes a traves
del tiempo y matematica LSTM ver
[clase-11 profundizacion](/clases/clase-11/profundizacion/). Para el alignment math
de Bahdanau y la derivacion de Seq2Seq con cross-entropy ver
[clase-13 profundizacion](/clases/clase-13/profundizacion/).

### 3.1 Justificacion del factor $1/\sqrt{d_k}$ en scaled dot-product attention

Vaswani et al. definen la atencion como

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V.$$

La pregunta es: por que dividir por $\sqrt{d_k}$ en lugar de no normalizar, o
normalizar por $d_k$? La respuesta es un **argumento de varianza** que asume
que las componentes de $q$ y $k$ son independientes con media 0 y varianza 1.

Sea $q, k \in \mathbb{R}^{d_k}$ con componentes $q_i, k_i$ iid, $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$,
$\text{Var}(q_i) = \text{Var}(k_i) = 1$, e independientes entre si. El producto punto es

$$q^T k = \sum_{i=1}^{d_k} q_i k_i.$$

Por linealidad de la esperanza y la independencia de $q_i$ y $k_i$:

$$\mathbb{E}[q^T k] = \sum_i \mathbb{E}[q_i] \mathbb{E}[k_i] = 0.$$

Para la varianza, cada termino $q_i k_i$ tiene
$\mathbb{E}[(q_i k_i)^2] = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = 1 \cdot 1 = 1$
y $\mathbb{E}[q_i k_i] = 0$, asi que $\text{Var}(q_i k_i) = 1$. Por independencia
de los terminos:

$$\text{Var}(q^T k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k.$$

Es decir, la magnitud tipica del logit crece como $\sqrt{d_k}$. Para
$d_k = 64$ los scores fluctuan en el rango $\pm 8$, manejable. Pero para
$d_k = 512$ fluctuan en $\pm 22$, lo que **satura la softmax**: la mayoria de la
masa se concentra en una sola posicion y los gradientes en las demas tienden a 0
(`softmax' (z)_i = p_i (\delta_{ij} - p_j)`, despreciable cuando una $p_i \to 1$).

Dividiendo por $\sqrt{d_k}$ obtenemos
$\text{Var}(q^T k / \sqrt{d_k}) = d_k / d_k = 1$, devolviendo los logits a una
escala estable independiente de la dimension. Esta es la unica razon por la que
$\sqrt{d_k}$ aparece y no otro factor: viene del teorema central del limite
aplicado al producto interno.

Para una discusion comparativa entre dot-product, additive y scaled dot-product,
ver [clase-13 profundizacion seccion 5](/clases/clase-13/profundizacion/).

### 3.2 Multi-head attention: por que $H$ cabezales de dimension $d_k = d_{\text{model}} / H$ tienen el mismo costo que uno solo

Multi-head attention reparte el espacio de atencion en $H$ cabezales. Cada cabezal
$i$ tiene matrices de proyeccion $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ y
$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, con $d_k = d_v = d_{\text{model}} / H$
en la receta del paper. El output es

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O,$$

con $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$ y
$W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

El conteo de FLOPs revela una propiedad clave. Sea $n$ la longitud de secuencia.

**Single-head con dimension $d_{\text{model}}$**:

- Proyecciones $Q, K, V$: $3 \cdot n \cdot d_{\text{model}}^2$ FLOPs.
- Producto $Q K^T$: $n^2 \cdot d_{\text{model}}$ FLOPs.
- Aplicacion del softmax a $V$: $n^2 \cdot d_{\text{model}}$ FLOPs.

Total dominante en attention: $\mathcal{O}(n^2 \cdot d_{\text{model}})$
mas $\mathcal{O}(n \cdot d_{\text{model}}^2)$ en proyecciones.

**$H$ cabezales con dimension $d_k = d_{\text{model}} / H$**:

- Proyecciones por cabezal: $3 \cdot n \cdot d_{\text{model}} \cdot d_k$ FLOPs cada uno.
  Sumando todos: $3 \cdot n \cdot d_{\text{model}} \cdot d_k \cdot H = 3 \cdot n \cdot d_{\text{model}}^2$.
- Producto $Q_i K_i^T$ por cabezal: $n^2 \cdot d_k$ FLOPs cada uno. Total: $n^2 \cdot d_k \cdot H = n^2 \cdot d_{\text{model}}$.
- Multiplicacion por $V_i$: identico, $n^2 \cdot d_{\text{model}}$.
- $W^O$: $n \cdot d_{\text{model}}^2$.

Los totales coinciden hasta constantes pequenas. Es decir, **multi-head no cuesta
mas** que single-head (a igual $d_{\text{model}}$), pero permite a la red
construir $H$ subespacios de atencion distintos en paralelo. Empiricamente cada
cabezal aprende patrones diferentes (sintaxis, co-referencia, posicion relativa,
proximidad lexica) que se combinan en $W^O$.

La leccion ingenieril es importante: **multi-head es un free lunch**. La unica
razon por la que no se hace single-head con $d_{\text{model}}$ enorme es que la
softmax sobre un unico mapa de atencion es menos expresiva que la mezcla de
varios mapas independientes, aun a igual presupuesto.

### 3.3 Positional encoding sinusoidal y la propiedad de invarianza por traslacion

Una RNN procesa la secuencia paso a paso, asi que el orden esta codificado en la
estructura del computo. Pero un Transformer aplica self-attention de forma
permutation-invariante: si reorden los inputs, la salida (modulo permutacion)
es identica. Por lo tanto necesita inyectar **informacion de posicion**
explicitamente. Vaswani et al. proponen sumar al embedding de cada token un
vector $\mathrm{PE}(\text{pos}) \in \mathbb{R}^{d_{\text{model}}}$ definido por

$$\mathrm{PE}(\text{pos}, 2i) = \sin(\text{pos} / 10000^{2i / d_{\text{model}}})$$
$$\mathrm{PE}(\text{pos}, 2i+1) = \cos(\text{pos} / 10000^{2i / d_{\text{model}}})$$

para $i = 0, 1, \ldots, d_{\text{model}}/2 - 1$. Cada par de coordenadas
$(2i, 2i+1)$ codifica la posicion en una "frecuencia angular"
$\omega_i = 1 / 10000^{2i / d_{\text{model}}}$, que decae geometricamente con $i$.

**Propiedad clave (relative position invariance).** Para cualquier offset $k$ entero,
el vector $\mathrm{PE}(\text{pos} + k)$ se puede expresar como una **transformacion
lineal** (independiente de pos) aplicada a $\mathrm{PE}(\text{pos})$. Es decir,
existe una matriz $M_k \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ tal
que para todo pos:

$$\mathrm{PE}(\text{pos} + k) = M_k \cdot \mathrm{PE}(\text{pos}).$$

**Demostracion.** Trabajemos sobre el par $(2i, 2i+1)$ con frecuencia $\omega_i$.
Aplicando las formulas trigonometricas de adicion:

$$\sin(\omega_i (\text{pos} + k)) = \sin(\omega_i \text{pos}) \cos(\omega_i k) + \cos(\omega_i \text{pos}) \sin(\omega_i k),$$
$$\cos(\omega_i (\text{pos} + k)) = \cos(\omega_i \text{pos}) \cos(\omega_i k) - \sin(\omega_i \text{pos}) \sin(\omega_i k).$$

En forma matricial, llamando $u_i(\text{pos}) = (\sin(\omega_i \text{pos}), \cos(\omega_i \text{pos}))^T$:

$$u_i(\text{pos} + k) = R_i(k) \cdot u_i(\text{pos}),$$

donde $R_i(k) = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix}$
es una **rotacion** en el plano $(2i, 2i+1)$ con angulo $-\omega_i k$. La matriz
global $M_k$ es la suma directa de estas rotaciones por cada par:

$$M_k = \bigoplus_{i=0}^{d_{\text{model}}/2 - 1} R_i(k).$$

Lo crucial es que $M_k$ depende **solo del offset $k$, no de la posicion absoluta
pos**. Esto significa que, dada $\mathrm{PE}(\text{pos}_a)$ y $\mathrm{PE}(\text{pos}_b)$,
la diferencia $\text{pos}_b - \text{pos}_a$ es recuperable mediante una operacion
lineal fija. La attention puede entonces aprender a atender a "tokens 5 posiciones
atras" generando una query que rote el key 5 unidades, sin importar donde en la
secuencia ocurra. Esa es la propiedad de **relative-position invariance** que
Vaswani et al. mencionan como motivacion para usar sinusoides en vez de
embeddings aprendidos.

Esta misma observacion -- que las rotaciones lineales en pares de coordenadas
codifican posicion -- es lo que motiva las **rotary position embeddings (RoPE)**
de Su et al. (2021) y las variantes ALiBi, que predominan en LLMs modernos
(LLaMA, GPT-NeoX, Mistral).

---

## 4. Diferencias con la clase UC y con el video 2020

La siguiente tabla resume coberturas entre las tres fuentes principales:

| Tema | MIT 2020 (L2 Soleimany) | MIT 2026 (L2 Amini) | UC clases 11 + 13 |
|---|---|---|---|
| Datos secuenciales y motivacion | Si, pedagogico | Si, condensado | Si |
| RNN vainilla y BPTT | Si, conceptual | Mencion breve | **Si, BPTT formal** |
| Vanishing/exploding gradient | Si, intuitivo | Mencion (refuerza por que LSTM/Transformer) | **Si, analisis espectral** |
| LSTM completo | Si, ecuaciones | Si, condensado | **Si, derivacion formal** |
| GRU | Mencion | Mencion | Si, completo |
| Encoder-decoder seq2seq | Si | Si, breve antes de attention | Si, en clase 13 |
| Bahdanau attention | Si, central | Si, como puente | **Si, formalizado** |
| Self-attention con Q/K/V | No | **Si, central** | Si, en clase 13 |
| Scaled dot-product justificacion | No | Mencion | Si, varianza |
| Multi-head attention | No | **Si, con motivacion** | Si, breve |
| Positional encoding | No | **Si** | Mencion |
| Transformer block completo | No | **Si** | Si, breve |
| Conexion con LLMs | No | **Si, explicita** | Mencion |
| Image captioning + attention espacial | Si | No (eliminado) | No |
| Demos de generacion (Shakespeare, musica) | Si | Si, recortado | No |

**Que profundiza la edicion 2026 que la edicion 2020 no cubre.** Lo central:
toda la mitad de Transformers. Vaswani et al. aparece de manera explicita, con
derivacion de Q/K/V como proyecciones aprendidas del input, justificacion del
escalamiento $1/\sqrt{d_k}$, motivacion de multi-head, y una pasada completa por
el Transformer block (residual + layernorm + FFN). Tambien hay un cambio de tono
mas marcado: las RNNs se presentan como **paso historico** en vez de tecnologia
contemporanea, con el Transformer como destino natural.

**Que profundiza la clase UC que MIT 2026 no cubre.** El curso UC mantiene
ventaja en el rigor formal: BPTT con todas las cadenas de gradientes, demostracion
de por que LSTM evita vanishing usando $\partial c_t / \partial c_{t-1} = f_t$,
analisis espectral con valores singulares de Pascanu et al., y la formalizacion
probabilistica de seq2seq con maximizacion de log-verosimilitud y beam search.
La edicion MIT 2026 los menciona pero no los deriva.

**Espiritu complementario.** Lo recomendable sigue siendo: ver el video MIT 2026
para la *vista de pajaro* completa (RNN -> attention -> Transformer en una hora) y
la intuicion visual; trabajar luego [Clase 11 profundizacion](/clases/clase-11/profundizacion/)
y [Clase 13 profundizacion](/clases/clase-13/profundizacion/) para cerrar las
matematicas.

---

## 5. Conceptos NO cubiertos por MIT 2026 que vale la pena conocer

- **Bidirectional RNNs / BiLSTM**: aludidas pero no profundizadas; centrales para tagging y NLU offline.
- **Encoder-only vs decoder-only Transformers**: BERT (encoder-only, MLM, NLU) vs GPT (decoder-only, autoregresivo, generacion); ambas familias derivan del paper de 2017 pero divergen en pretraining y uso.
- **Transformer-XL** (Dai et al. 2019): atencion segmentada con memoria recurrente entre segmentos para extender el receptive field mas alla del context window.
- **Sparse attention** (Child et al. 2019, Longformer, BigBird): patrones de atencion sub-cuadraticos para secuencias largas (documentos, codigo).
- **FlashAttention** (Dao et al. 2022): implementacion IO-aware de attention que evita materializar la matriz $n \times n$ completa, reduciendo memoria de $\mathcal{O}(n^2)$ a $\mathcal{O}(n)$ y acelerando 2-4x en GPUs modernas; es lo que hace viable contextos de millones de tokens en LLMs actuales.
- **Rotary Position Embeddings (RoPE) y ALiBi**: alternativas modernas al positional encoding sinusoidal de Vaswani et al., que generalizan mejor a longitudes mayores que las vistas en entrenamiento; predominan en LLMs 2023-2026.
- **Modelos espacio-estado (S4, Mamba)**: combinan complejidad lineal (como RNN) con paralelismo en entrenamiento (como Transformer); investigacion activa que compite con attention en contextos largos.
- **Mixture of Experts (MoE)**: enrutamiento sparse de tokens a expertos especializados; clave en LLMs modernos como Mixtral, DeepSeek y Claude.
- **Speculative decoding**: tecnica de inferencia que acelera la generacion autoregresiva usando un modelo pequeno para proponer y uno grande para verificar.

---

## Referencias

- Bengio, Simard & Frasconi (1994). Learning Long-Term Dependencies with Gradient Descent is Difficult. *IEEE TNN* 5(2):157-166.
- Pascanu, Mikolov & Bengio (2013). [On the Difficulty of Training Recurrent Neural Networks](/papers/difficulty-training-rnns-pascanu-2013/). *ICML*.
- Hochreiter & Schmidhuber (1997). [Long Short-Term Memory](/papers/lstm-hochreiter-1997/). *Neural Computation* 9(8).
- Cho et al. (2014). [Learning Phrase Representations Using RNN Encoder-Decoder](/papers/gru-cho-2014/). *EMNLP*.
- Sutskever, Vinyals & Le (2014). [Sequence to Sequence Learning with Neural Networks](/papers/seq2seq-sutskever-2014/). *NeurIPS*.
- Bahdanau, Cho & Bengio (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](/papers/bahdanau-attention-2015/). *ICLR*.
- Luong, Pham & Manning (2015). Effective Approaches to Attention-based Neural Machine Translation. arXiv [1508.04025](https://arxiv.org/abs/1508.04025).
- Xu et al. (2015). [Show, Attend and Tell](/papers/show-attend-tell-xu-2015/). *ICML*.
- Vaswani et al. (2017). Attention Is All You Need. arXiv [1706.03762](https://arxiv.org/abs/1706.03762).

Material relacionado: [video MIT 6.S191 (2020) RNN](/videos/mit-6s191-rnn/profundizacion/) - [Clase 11 profundizacion](/clases/clase-11/profundizacion/) - [Clase 13 profundizacion](/clases/clase-13/profundizacion/).

---

> Material complementario al video **MIT 6.S191 (2026) Lecture 2: Recurrent Neural Networks, Transformers, and Attention**, Ava Amini, 5 de enero de 2026.
> [Video](https://www.youtube.com/watch?v=d02VkQ9MP44) - [Slides oficiales](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf) - [Sitio del curso](https://introtodeeplearning.com/).
> Investigacion adicional como elaboracion independiente. Sin afiliacion oficial con MIT.
