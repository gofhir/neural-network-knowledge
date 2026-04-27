---
title: "Glosario - MIT 6.S191 (2026): RNNs + Transformers"
weight: 30
---

> Glosario autocontenido de los terminos clave del lecture de Ava Amini (MIT 6.S191, 2026). Cada entrada usa el formato `**Termino (English)**` seguido de una definicion concisa en espanol (1 a 3 lineas). Los terminos cuya derivacion matematica se desarrolla en `profundizacion.md` incluyen un puntero al final.

---

## RNN, BPTT y memoria recurrente

**Autoregressive (Autoregresivo)** — Modelo que genera una secuencia paso a paso, donde cada salida $y_t$ se condiciona en las salidas previas $y_1, \ldots, y_{t-1}$. Durante inferencia, el output muestreado se realimenta como input del siguiente paso.

**Backpropagation Through Time (BPTT)** — Generalizacion del algoritmo de backpropagation aplicada al grafo computacional desplegado de una RNN. La perdida se acumula sobre todos los pasos temporales y los gradientes se propagan hacia atras a traves de la cadena recurrente. Ver `profundizacion.md` para la derivacion.

**Cell state (Estado de celda)** — Vector $c_t$ interno de un LSTM que actua como memoria de largo plazo. Se actualiza de forma aditiva ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$), preservando el flujo del gradiente sin atenuacion exponencial.

**Context vector (Vector de contexto)** — Resumen $c$ producido por el encoder y consumido por el decoder en una arquitectura seq2seq. En el modelo clasico es el ultimo hidden state; con attention se recalcula dinamicamente como $c_t = \sum_i \alpha_{t,i} h_i$ en cada paso del decoder.

**Decoder (Decodificador)** — Segunda RNN del patron encoder-decoder que, condicionada en el context vector del encoder, genera la secuencia de salida token a token. Tipicamente opera de forma autorregresiva durante inferencia.

**Embedding** — Representacion densa de baja dimension de una unidad simbolica (palabra, caracter, item) aprendida durante el entrenamiento. Sustituye al one-hot al alimentar la red, dejando palabras semanticamente similares cercanas en el espacio vectorial.

**Encoder (Codificador)** — Primera RNN del patron encoder-decoder que procesa la secuencia de entrada y produce una representacion comprimida (el ultimo hidden state o el conjunto de hidden states) que el decoder consume.

**Exploding gradient (Gradiente explosivo)** — Patologia en la que la norma del gradiente crece exponencialmente al propagarse hacia atras en el tiempo, causando inestabilidad numerica (NaN). Se mitiga con gradient clipping. Ver `profundizacion.md` para la derivacion.

**Forget gate (Compuerta de olvido)** — Vector $f_t \in [0,1]^d$ del LSTM, calculado como $f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)$, que controla cuanta informacion del cell state previo $c_{t-1}$ se preserva.

**Gating (Mecanismo de compuertas)** — Patron arquitectural donde se introduce una multiplicacion elementwise por vectores en $[0,1]^d$ producidos por sigmoides aprendidas. Permite control fino y diferenciable sobre que informacion fluye en cada paso.

**Gradient clipping (Recorte de gradiente)** — Tecnica para mitigar el exploding gradient que escala el vector de gradiente cuando su norma excede un umbral $\tau$: $\hat{g} = (\tau / \|g\|) \cdot g$ si $\|g\| > \tau$. Preserva la direccion y solo limita la magnitud.

**GRU (Gated Recurrent Unit)** — Arquitectura recurrente con compuertas propuesta por Cho et al. (2014) que simplifica LSTM: combina forget e input en una **update gate**, anade una **reset gate** y prescinde del cell state separado. Suele lograr rendimiento comparable a LSTM con menos parametros.

**Hidden state (Estado oculto)** — Vector $h_t \in \mathbb{R}^{d_h}$ que actua como resumen lossy de la historia de la secuencia hasta el paso $t$. Se actualiza recurrentemente segun $h_t = f(h_{t-1}, x_t)$ y es la pieza central de cualquier RNN.

**Input gate (Compuerta de entrada)** — Vector $i_t \in [0,1]^d$ del LSTM que controla cuanta informacion del candidato $\tilde{c}_t$ entra en la cell state. Se calcula como $i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)$.

**LSTM (Long Short-Term Memory)** — Arquitectura recurrente con compuertas (forget, input, output) y una cell state separada del hidden state, propuesta por Hochreiter y Schmidhuber (1997). Permite aprender dependencias a centenas o miles de pasos al preservar el flujo del gradiente.

**One-hot encoding** — Representacion de un simbolo discreto como un vector binario donde solo la posicion correspondiente esta encendida. Para un vocabulario de tamano $V$, ocupa un vector en $\{0,1\}^V$. Es ineficiente en alta dimension y se reemplaza tipicamente con embeddings.

**Output gate (Compuerta de salida)** — Vector $o_t \in [0,1]^d$ del LSTM que filtra que informacion del cell state se expone como hidden state. Combinado con $\tanh(c_t)$ produce $h_t = o_t \odot \tanh(c_t)$.

**Perplexity (Perplejidad)** — Metrica estandar para modelos de lenguaje, definida como $\exp(\text{cross-entropy})$. Interpretable como el numero efectivo de opciones igualmente probables que el modelo considera en cada paso. Menor es mejor.

**Recurrent cell (Unidad recurrente)** — Modulo que computa el estado siguiente $h_t$ a partir de $h_{t-1}$ y $x_t$. Distintas elecciones del modulo definen distintas RNNs: vanilla, LSTM, GRU, etc.

**Sequence (Secuencia)** — Estructura ordenada de elementos $\{x_1, x_2, \ldots, x_T\}$ donde el orden es semanticamente relevante. Datos secuenciales tipicos: texto, audio, video, series temporales, ADN.

**Sequence-to-sequence (Seq2seq)** — Familia de arquitecturas que mapea una secuencia de entrada a una secuencia de salida de longitud potencialmente distinta, tipicamente mediante un encoder-decoder. Introducida formalmente por Sutskever et al. (2014).

**Teacher forcing** — Estrategia de entrenamiento del decoder en seq2seq donde, en cada paso, se alimenta como input la **salida correcta** $y_{t-1}^{\text{true}}$ del dataset en lugar de la prediccion del modelo. Acelera la convergencia pero introduce *exposure bias* en inferencia.

**Vanishing gradient (Gradiente evanescente)** — Patologia en la que la norma del gradiente decae exponencialmente al propagarse hacia atras a traves de muchos pasos temporales, impidiendo el aprendizaje de dependencias a largo plazo. Motiva la introduccion de LSTM y GRU. Ver `profundizacion.md` para la derivacion.

**Weight sharing (Compartir parametros)** — Propiedad por la cual la misma matriz $W_{hh}$, $W_{xh}$, $W_{hy}$ se aplica en cada paso temporal. Permite procesar secuencias de longitud variable sin que crezca el numero de parametros.

---

## Attention y Transformers

**Alignment (Alineamiento)** — Correspondencia aprendida entre posiciones de la secuencia de entrada y posiciones de la secuencia de salida. En traduccion automatica con attention, los pesos $\alpha_{t,i}$ implementan un alineamiento suave (soft alignment) palabra a palabra.

**Attention (Atencion)** — Mecanismo que permite consultar selectivamente partes de una representacion mediante pesos aprendidos $\alpha_{t,i}$, produciendo un context vector dinamico que se adapta a cada paso. Resuelve el cuello de botella del context vector unico de seq2seq.

**Attention mask (Mascara de atencion)** — Matriz binaria (o de $\{0, -\infty\}$) que se suma a los logits antes del softmax para prohibir atender ciertas posiciones (padding, futuros tokens, posiciones invalidas).

**Attention weights (Pesos de atencion)** — Coeficientes $\alpha_{ij} \in [0,1]$, $\sum_j \alpha_{ij} = 1$, obtenidos al aplicar softmax a los scores query-key. Cuantifican cuanto atiende la posicion $i$ a la posicion $j$ y son la fuente principal de interpretabilidad en Transformers.

**Beam search (Busqueda por haz)** — Algoritmo de decodificacion que mantiene los $k$ candidatos parciales mas probables (beam width $k$) en cada paso, en lugar de la unica mejor opcion (greedy). Mejora la calidad de generacion en traduccion y modelado de lenguaje.

**BLEU (Bilingual Evaluation Understudy)** — Metrica estandar para evaluar traduccion automatica que mide superposicion de n-gramas entre la traduccion candidata y una o varias referencias. Toma valores en $[0, 1]$ y es la metrica usual en benchmarks de NMT.

**Causal mask / Masked attention (Mascara causal)** — Mascara triangular inferior que impide a cada posicion $i$ atender posiciones $j > i$. Es lo que vuelve autorregresivo a un decoder Transformer y permite entrenar en paralelo sobre toda la secuencia.

**Encoder-only / Decoder-only / Encoder-decoder Transformer** — Tres familias arquitectonicas: **encoder-only** (BERT) para representacion bidireccional y clasificacion; **decoder-only** (GPT) con causal masking para generacion autorregresiva; **encoder-decoder** (T5, vanilla Transformer) para tareas seq2seq como traduccion.

**Feed-forward network (FFN)** — Subcapa posicional del bloque Transformer aplicada independientemente a cada token: $\text{FFN}(x) = W_2 \, \text{GELU}(W_1 x + b_1) + b_2$. Suele expandir la dimension oculta a $4 d_{\text{model}}$ y aporta la mayor parte de los parametros.

**Key (Clave)** — Vector $k_j = W_K x_j$ con el cual cada posicion se "indexa" para ser consultada. El score de atencion entre query $i$ y key $j$ es el producto punto $q_i \cdot k_j$ (escalado).

**Layer normalization (LayerNorm)** — Normalizacion que estandariza un vector de activaciones a media 0 y varianza 1 a lo largo de la dimension de features (no del batch), seguida de un reescalado afin aprendido $\gamma, \beta$. Estabiliza el entrenamiento de Transformers.

**Multi-head attention (Atencion multi-cabeza)** — Variante que ejecuta $h$ atenciones scaled dot-product en paralelo sobre proyecciones lineales independientes de Q, K, V. Cada cabeza captura un patron de relacion distinto y sus salidas se concatenan y proyectan. Ver `profundizacion.md` para la derivacion.

**Positional encoding (Codificacion posicional)** — Vector que se suma (o concatena) al embedding de cada token para inyectar informacion de orden, ya que la self-attention es permutation-invariant. Puede ser aprendido o fijo (sinusoidal, RoPE, ALiBi).

**Query (Consulta)** — Vector $q_i = W_Q x_i$ con el que la posicion $i$ "pregunta" cuales otras posiciones le son relevantes. Se compara contra todas las keys mediante producto punto para producir los scores de atencion.

**Residual connection / Skip connection (Conexion residual)** — Atajo aditivo $y = x + f(x)$ alrededor de cada subcapa del Transformer (atencion y FFN). Facilita el flujo de gradiente en redes profundas y permite apilar decenas de bloques.

**Scaled dot-product attention** — Forma cerrada $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d_k}) V$. La division por $\sqrt{d_k}$ evita que los logits crezcan con la dimension y saturen el softmax. Ver `profundizacion.md` para la derivacion.

**Self-attention (Auto-atencion)** — Atencion donde queries, keys y values provienen de la misma secuencia: cada token atiende a todos los tokens (incluido a si mismo) de la misma capa. Es el bloque central del Transformer.

**Sinusoidal positional encoding (Codificacion posicional sinusoidal)** — Esquema fijo del paper original donde $PE_{(\text{pos}, 2i)} = \sin(\text{pos}/10000^{2i/d})$ y $PE_{(\text{pos}, 2i+1)} = \cos(\text{pos}/10000^{2i/d})$. Permite extrapolar a longitudes no vistas y codifica posiciones relativas implicitamente.

**Softmax temperature (Temperatura del softmax)** — Hiperparametro $T$ que escala los logits antes del softmax: $p_i = \exp(z_i/T) / \sum_j \exp(z_j/T)$. Con $T \to 0$ se vuelve greedy/argmax; con $T \to \infty$ tiende al uniforme. Controla la diversidad en sampling.

**Value (Valor)** — Vector $v_j = W_V x_j$ que se devuelve cuando la posicion $j$ es "atendida". La salida de atencion es la combinacion convexa $\sum_j \alpha_{ij} v_j$ ponderada por los attention weights.

---

> Material adaptado de **MIT 6.S191 (2026) Lecture 2** (Ava Amini), reutilizando el glosario 2020 (Ava Soleimany) como base. Sin afiliacion oficial con MIT.

---

> Glosario complementario al video **MIT 6.S191 (2026) Lecture 2**, Ava Amini.
> Para derivaciones formales ver [`profundizacion`](../profundizacion). Para el recorrido por las diapositivas ver [`notas`](../notas).
