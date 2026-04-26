---
title: "Glosario - MIT 6.S191 RNNs"
weight: 30
math: true
---

> Glosario de terminos clave aparecidos en el lecture de Ava Soleimany. Cada entrada incluye el termino en espanol seguido del original en ingles entre parentesis y una definicion concisa de una a tres lineas.

---

**Alineamiento (Alignment)** - Correspondencia aprendida entre posiciones de la secuencia de entrada y posiciones de la secuencia de salida. En traduccion automatica con attention, los pesos $\alpha_{t,i}$ implementan un alineamiento suave (soft alignment) palabra a palabra.

**Atencion (Attention)** - Mecanismo que permite a un decoder consultar selectivamente partes de la representacion del encoder mediante pesos aprendidos $\alpha_{t,i}$, produciendo un context vector dinamico $C_t$ en cada paso del decoder. Resuelve el cuello de botella del context vector unico.

**Autoregresivo (Autoregressive)** - Modelo que genera una secuencia paso a paso, donde cada salida $y_t$ se condiciona en las salidas previas $y_1, \ldots, y_{t-1}$. Durante inferencia, el output muestreado se realimenta como input del siguiente paso.

**Backpropagation Through Time (BPTT)** - Generalizacion del algoritmo de backpropagation aplicada al grafo computacional desplegado de una RNN. La perdida se acumula sobre todos los pasos temporales y los gradientes se propagan hacia atras a traves de la cadena recurrente.

**BLEU (Bilingual Evaluation Understudy)** - Metrica estandar para evaluar traduccion automatica que mide superposicion de n-gramas entre la traduccion candidata y una o varias traducciones de referencia. Toma valores en $[0, 1]$ y es la metrica usual en benchmarks de NMT.

**Cell state** - Vector $c_t$ interno de un LSTM que actua como memoria de largo plazo. Se actualiza de forma aditiva ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$), lo que permite el flujo de gradiente sin atenuacion exponencial.

**Clasificacion de sentimiento (Sentiment classification)** - Tarea N-a-1 que asigna una etiqueta categorica (positivo / negativo, escala 1-5, etc.) a una secuencia textual completa. Usa el ultimo estado oculto $h_T$ como representacion para la prediccion.

**Compartir parametros (Weight sharing)** - Propiedad por la cual la misma matriz $W_{hh}$, $W_{xh}$, $W_{hy}$ se aplica en cada paso temporal. Permite procesar secuencias de longitud variable sin que crezca el numero de parametros.

**Compuerta de entrada (Input gate)** - Vector $i_t \in [0,1]^d$ del LSTM que controla cuanta informacion del candidato $\tilde{c}_t$ entra en la cell state. Se calcula como $i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)$.

**Compuerta de olvido (Forget gate)** - Vector $f_t \in [0,1]^d$ del LSTM que controla cuanta informacion del cell state previo $c_{t-1}$ se preserva. Cuando $f_t \approx 0$, se "olvida" la memoria pasada; cuando $f_t \approx 1$, se conserva integramente.

**Compuerta de salida (Output gate)** - Vector $o_t \in [0,1]^d$ del LSTM que filtra que informacion del cell state se expone como hidden state hacia el siguiente paso y hacia la capa superior. Combinado con $\tanh(c_t)$ produce $h_t = o_t \odot \tanh(c_t)$.

**Decoder** - Segunda RNN del patron encoder-decoder que, condicionada en el context vector del encoder, genera la secuencia de salida palabra por palabra. Tipicamente opera de forma autorregresiva durante inferencia.

**Embedding** - Representacion densa de baja dimension de una unidad simbolica (palabra, caracter, item) aprendida durante el entrenamiento. Sustituye al one-hot encoding al alimentar la RNN, permitiendo que palabras semanticamente similares queden cercanas en el espacio.

**Encoder** - Primera RNN del patron encoder-decoder que procesa la secuencia de entrada y produce una representacion comprimida (el ultimo estado oculto o un conjunto de hidden states) que el decoder consume.

**Estado oculto (Hidden state)** - Vector $h_t \in \mathbb{R}^{d_h}$ que actua como resumen lossy de la historia de la secuencia hasta el paso $t$. Se actualiza recurrentemente segun $h_t = f(h_{t-1}, x_t)$ y es la pieza central de cualquier RNN.

**Generacion de musica (Music generation)** - Tarea 1-a-N donde una RNN aprende a generar secuencias musicales (notas, acordes, ritmos) condicionada en un contexto inicial. La clase MIT lo ejemplifica con un modelo character-level entrenado sobre ABC notation.

**Gradient clipping** - Tecnica para mitigar el exploding gradient que escala el vector de gradiente cuando su norma excede un umbral $\tau$: $\hat{g} = (\tau / \|g\|) \cdot g$ si $\|g\| > \tau$. Preserva la direccion y solo limita la magnitud.

**Gradiente explosivo (Exploding gradient)** - Patologia en la que la norma del gradiente crece exponencialmente al propagarse hacia atras en el tiempo, causando inestabilidad numerica (NaN) durante el entrenamiento. Se mitiga con gradient clipping.

**Gradiente evanescente (Vanishing gradient)** - Patologia en la que la norma del gradiente decae exponencialmente al propagarse hacia atras a traves de muchos pasos temporales, impidiendo el aprendizaje de dependencias a largo plazo. Motiva la introduccion de LSTM y GRU.

**GRU (Gated Recurrent Unit)** - Arquitectura recurrente con compuertas propuesta por Cho et al. (2014) que simplifica LSTM: combina las compuertas de olvido e input en una **update gate**, anade una **reset gate** y prescinde del cell state separado. Suele tener rendimiento comparable a LSTM con menos parametros.

**LSTM (Long Short-Term Memory)** - Arquitectura recurrente con compuertas (forget, input, output) y una cell state separada del hidden state, propuesta por Hochreiter y Schmidhuber (1997). Permite aprender dependencias a centenas o miles de pasos al preservar el flujo del gradiente.

**Mecanismo de gating (Gating mechanism)** - Patron arquitectural donde se introduce una multiplicacion elementwise por vectores $\in [0,1]^d$ (compuertas) producidos por sigmoides aprendidas. Permite control fino y diferenciable sobre que informacion fluye en cada paso.

**Modelado de lenguaje (Language modeling)** - Tarea de estimar la distribucion de probabilidad sobre la siguiente unidad (caracter o palabra) condicionada en el contexto previo: $P(x_t \mid x_1, \ldots, x_{t-1})$. Es la tarea de pretraining mas comun para RNNs (y mas tarde para Transformers).

**One-hot encoding** - Representacion de un simbolo discreto como un vector binario donde solo la posicion correspondiente esta encendida. Para un vocabulario de tamano $V$, ocupa un vector en $\{0,1\}^V$. Es ineficiente en alta dimension y usualmente se reemplaza con embeddings.

**Perplejidad (Perplexity)** - Metrica estandar para modelos de lenguaje, definida como $\exp(\text{cross-entropy})$. Interpretable como el numero efectivo de opciones igualmente probables que el modelo considera en cada paso. Menor es mejor.

**Secuencia (Sequence)** - Estructura ordenada de elementos $\{x_1, x_2, \ldots, x_T\}$ donde el orden es semanticamente relevante. Datos secuenciales tipicos: texto, audio, video, series temporales, ADN.

**Secuencia a secuencia (Sequence-to-sequence, Seq2seq)** - Familia de arquitecturas que mapea una secuencia de entrada a una secuencia de salida de longitud potencialmente distinta, tipicamente mediante un encoder-decoder. Introducida formalmente por Sutskever et al. (2014).

**Softmax** - Funcion $\sigma(z)_i = \exp(z_i) / \sum_j \exp(z_j)$ que convierte un vector de logits en una distribucion de probabilidad sobre clases. Se usa en la salida del decoder para seleccionar el siguiente token y en el calculo de pesos de attention.

**Tanh (Tangente hiperbolica)** - Funcion de activacion $\tanh(z) = (e^z - e^{-z})/(e^z + e^{-z})$ con rango $[-1, 1]$. Es la activacion estandar para el estado oculto de RNNs vanilla y para el candidato $\tilde{c}_t$ del LSTM.

**Teacher forcing** - Estrategia de entrenamiento del decoder en seq2seq donde, en cada paso, se alimenta como input la **salida correcta** $y_{t-1}^{\text{true}}$ del dataset en lugar de la prediccion del modelo $\hat{y}_{t-1}$. Acelera convergencia pero introduce *exposure bias* en inferencia.

**Traduccion automatica (Machine translation)** - Tarea de mapear una oracion en un idioma origen a una oracion en un idioma destino. Es la aplicacion canonica del patron encoder-decoder con attention y el escenario donde se introdujeron Bahdanau et al. (2015) y los Transformers.

**Unidad recurrente (Recurrent cell)** - Modulo que computa el estado siguiente $h_t$ a partir de $h_{t-1}$ y $x_t$. Distintas elecciones del modulo producen distintas RNNs: vanilla, LSTM, GRU, etc.

---

> Material adaptado de **MIT 6.S191 (2020) Lecture 2: Deep Sequence Modeling**, Alexander Amini & Ava Soleimany. [Video](https://www.youtube.com/watch?v=SEnXr6v2ifU) - [Slides oficiales](http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf) - [Sitio del curso](http://introtodeeplearning.com/2020/). Notas en espanol con investigacion complementaria. Sin afiliacion oficial con MIT.
