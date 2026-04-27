---
title: "Glosario - MIT 6.S191 (2026)"
weight: 30
math: true
---

> Glosario autosuficiente para el lecture **Deep Sequence Modeling** (Ava Amini, 5 enero 2026). Cubre conceptos de modelado secuencial, RNNs, BPTT, gating, attention y la arquitectura Transformer. Cada entrada incluye el termino en ingles seguido de la traduccion al espanol entre parentesis cuando aplica, y una definicion concisa de 1-3 lineas.

---

**AlphaFold** - Modelo de prediccion de estructura proteinica 3D basado en attention/Transformers (Jumper et al., Nature 2021). Citado en slide 80 como aplicacion de self-attention en secuencias biologicas.

**Attention head (Cabeza de atencion)** - Modulo computacional individual que ejecuta las operaciones de self-attention sobre una proyeccion linear de Q, K y V. Cada head atiende a un aspecto distinto del input; varios heads en paralelo forman multi-head attention.

**Attention is All You Need** - Titulo del paper de Vaswani et al. (NeurIPS 2017) que introdujo el Transformer. Da nombre al section header de la slide 65.

**Attention mask (Mascara de atencion)** - Matriz que codifica que tan similar es cada key respecto al query; se computa antes de extraer los values. En el lecture aparece como paso 1 del search analogy (slide 68).

**Attention score (Puntaje de atencion)** - Similitud por pares entre cada query y cada key, calculada via dot product $Q \cdot K^T$. Es el insumo crudo que luego se escala y se pasa por softmax.

**Attention weighting (Ponderacion de atencion)** - Matriz resultado de aplicar softmax al puntaje escalado: $\text{softmax}(Q \cdot K^T / \sqrt{d})$. Define cuanto peso recibe cada value en la mezcla final.

**Backpropagation (Retropropagacion)** - Algoritmo para entrenar redes neuronales: (1) tomar la derivada de la perdida respecto a cada parametro, (2) ajustar los parametros para minimizar la perdida (slide 42).

**Backward pass (Pase hacia atras)** - Fase del entrenamiento en la que los gradientes se propagan desde la perdida hacia los parametros via la regla de la cadena. En RNNs corre desde la perdida total $L$ hacia $W_{xh}$, $W_{hh}$ y $W_{hy}$.

**BERT (Bidirectional Encoder Representations from Transformers)** - Modelo de lenguaje basado en Transformer encoder (Devlin et al., NAACL 2019). Citado en slide 80 como aplicacion de self-attention en NLP.

**Binary classification (Clasificacion binaria)** - Tarea one-to-one ejemplificada con "Will I pass this class?" (slide 8). Recibe un vector de input fijo y produce una etiqueta de dos clases.

**BPTT** - Ver **Backpropagation Through Time**.

**Backpropagation Through Time (BPTT)** - Algoritmo de entrenamiento de RNNs que aplica backprop sobre el grafo computacional desplegado a traves del tiempo. Las perdidas $L_0, \ldots, L_T$ se agregan en una perdida total $L$ y los gradientes fluyen hacia atras reusando $W_{hh}, W_{xh}, W_{hy}$ (slides 41-44, Mozer 1989).

**Cell state (Estado de celda)** - Termino usado en slide 17 para referirse al estado interno $h_t$ que la RNN cell mantiene y actualiza en cada paso. En LSTMs (slide 54) refiere especificamente a la memoria de largo plazo separada del hidden state.

**Computational graph (Grafo computacional)** - Representacion de la RNN unrolled a traves del tiempo, con cada paso $t$ como nodo. Visualiza forward y backward pass y deja explicito el reuso de las mismas matrices de pesos (slide 26).

**Cosine similarity (Similitud coseno)** - Metrica de similitud entre dos vectores basada en el producto punto normalizado. La slide 73 menciona que el dot product $Q \cdot K^T$ tras el escalado equivale conceptualmente a una cosine similarity.

**Cross-entropy (Entropia cruzada)** - Funcion de perdida estandar para clasificacion. En la slide 57 aparece como `tf.nn.softmax_cross_entropy_with_logits(y, predicted)` para sentiment classification.

**Decoder (Decodificador)** - Componente que transforma una representacion latente a la secuencia de salida. En el lecture aparece de forma implicita en multi-task generation; el Transformer original tiene un encoder-decoder.

**Dense network (Red densa)** - Red feed-forward fully-connected. En slide 64 se evalua como "Idea 1: feed everything into dense network" y se descarta por no ser escalable y no preservar orden.

**Dot product (Producto punto)** - Operacion $Q \cdot K^T$ usada como metrica de similitud entre query y key (slide 73). Es el corazon del calculo del attention score.

**Embedding** - Representacion densa de tamano fijo de un simbolo discreto. La slide 36 lo define como "transform indexes into a vector of fixed size" y muestra one-hot vs learned embedding.

**Encoder (Codificador)** - Componente que transforma una secuencia de input a una representacion latente. En self-attention aplicada (BERT, ViT) el encoder Transformer es el bloque protagonico.

**Encoding bottleneck (Cuello de botella de codificacion)** - Limitacion de RNNs por la cual toda la historia debe pasar por un hidden state de tamano fijo. Aparece como icono de balde en las slides 59 y 61.

**ESM (Evolutionary Scale Modeling)** - Familia de modelos Transformer para secuencias proteinicas (Lin et al., Science 2023). Citado en slide 80.

**Exploding gradient (Gradiente que explota)** - Patologia en la que la norma del gradiente crece exponencialmente al propagarse hacia atras. Ocurre cuando los factores acumulados de $W_{hh}$ tienen valores > 1 (slide 47). Se mitiga con gradient clipping.

**Feed-forward network (Red feed-forward)** - Red sin recurrencia donde la informacion fluye solo de entrada a salida. Es el punto de partida del lecture (slides 11-12) antes de introducir la recurrencia.

**Forward pass (Pase hacia adelante)** - Fase en la que las activaciones se computan desde el input hasta la prediccion $\hat{y}_t$. En RNNs corre a traves del unrolled graph en el sentido del tiempo.

**Foundation model (Modelo fundacional)** - Modelo grande preentrenado que sirve de base para muchas tareas. La slide 78 dice "Attention is the foundational building block of the Transformer architecture" y el summary (slide 81) anuncia LLMs como continuacion.

**Gate (Compuerta)** - Mecanismo que selectivamente deja pasar o bloquea informacion en una unidad recurrente. La slide 54 lo define como "use gates to selectively add or remove information within each recurrent unit".

**Gated cell (Celda con compuertas)** - Variante de RNN cell que incorpora gates para controlar el flujo de informacion. LSTM y GRU son las dos instancias mencionadas (slide 54).

**GPT (Generative Pre-trained Transformer)** - Familia de modelos de lenguaje autoregresivos basados en Transformer decoder. La slide 80 cita GPT-3 (Brown et al., NeurIPS 2020).

**Gradient clipping** - Tecnica para prevenir exploding gradients escalando el vector de gradiente cuando su norma excede un umbral. Mencionada en slide 47.

**GRU (Gated Recurrent Unit)** - Arquitectura recurrente con compuertas mas simple que LSTM. Mencionada brevemente en slide 54 como ejemplo de gated cell, sin detalle interno en el lecture 2026.

**Hidden state (Estado oculto)** - Vector $h_t$ que resume la historia de la secuencia hasta el paso $t$. Se actualiza segun $h_t = \tanh(W_{hh}^T h_{t-1} + W_{xh}^T x_t)$ y es la pieza central de la recurrencia (slide 23).

**Image captioning (Generacion de pies de imagen)** - Tarea one-to-many ejemplificada en slide 8 con "A baseball player throws a ball": entra una imagen, sale una secuencia de palabras.

**Indexing (Indexacion)** - Paso intermedio del pipeline de embedding que asigna un indice entero a cada palabra del vocabulario (slide 36): "a -> 1, cat -> 2, ..., walk -> N".

**Input vector (Vector de entrada)** - El vector $x_t$ que entra a la RNN cell en el paso $t$. Aparece como callout azul en slide 22.

**Key (K)** - Una de las tres proyecciones lineales de la entrada en self-attention. Se compara con el query para calcular attention scores. En la analogia de YouTube (slide 68) los keys son los titulos de los videos en el corpus.

**Learned embedding (Embedding aprendido)** - Representacion densa entrenable que mapea indices de palabras a vectores en $\mathbb{R}^d$. La slide 36 lo contrasta con one-hot mostrando un scatter 2D donde palabras semanticamente similares quedan cercanas.

**Long-term dependencies (Dependencias de largo plazo)** - Relaciones entre elementos de la secuencia que estan separados por muchos pasos. Ejemplo: "I grew up in France ... I speak fluent ___" (slide 38). Las RNNs vanilla las capturan mal por vanishing gradients.

**LSTM (Long Short-Term Memory)** - Arquitectura recurrente con gates (forget, input, output) y cell state separada que permite preservar informacion a traves de muchos pasos. Mencionada en slide 54 como ejemplo de gated cell.

**Machine translation (Traduccion automatica)** - Tarea many-to-many ejemplificada en slide 8 con un caracter chino traducido a latino. Tambien aparece en el grid del slide 29.

**Many-to-many** - Patron de mapeo donde una secuencia de entrada produce una secuencia de salida. Ejemplos del lecture: machine translation, music generation, forecasting (slides 8 y 29).

**Many-to-one** - Patron de mapeo donde una secuencia de entrada produce una unica salida. Ejemplo del lecture: sentiment classification (slides 8, 29 y 57).

**Multi-head attention (Atencion multi-cabeza)** - Aplicacion en paralelo de varios self-attention heads, cada uno enfocado en un aspecto distinto del input. La slide 79 lo visualiza con tres views diferentes de Iron Man.

**Music generation (Generacion de musica)** - Tarea many-to-many ejemplificada en slide 56: una RNN recibe notas (E, F#, G, C) y predice las siguientes (F#, G, C, A). Es la tarea del Lab 1.

**One-hot encoding** - Representacion binaria donde solo una posicion del vector esta encendida. La slide 36 muestra "cat" = $[0, 1, 0, 0, 0, 0]$ donde el 1 marca el indice de "cat" en el vocabulario.

**One-to-many** - Patron de mapeo donde una entrada fija produce una secuencia de salida. Ejemplos del lecture: image captioning, text generation (slides 8 y 29).

**One-to-one** - Patron de mapeo input-output sin secuencia (vanilla NN). Ejemplo del lecture: binary classification "Will I pass this class?" (slides 8 y 29).

**Order (Orden)** - Tercer criterio de diseno para sequence models (slide 30): el modelo debe mantener informacion sobre el orden de los elementos. Motivado por el ejemplo "good, not bad" vs "bad, not good" (slide 39).

**Output vector (Vector de salida)** - La prediccion $\hat{y}_t = W_{hy}^T h_t$ producida por la RNN en cada paso (slide 24).

**Parallelization (Paralelizacion)** - Capacidad de procesar elementos de una secuencia en paralelo. Es una limitacion de RNNs (icono reloj en slide 59) y una desired capability del Transformer (slide 62).

**Parameter sharing (Compartir parametros)** - Cuarto criterio de diseno (slide 30): las mismas matrices de pesos $W_{hh}, W_{xh}, W_{hy}$ se reusan en cada paso temporal (slide 26).

**Perceptron** - Unidad neuronal basica revisada en slide 10: $y = g(\sum_i w_i x_i)$, con inputs $x^{(1)}, \ldots, x^{(m)}$, pesos $w_1, \ldots, w_m$ y activacion $g$.

**Pointwise multiplication (Multiplicacion elementwise)** - Operacion $\odot$ aplicada componente a componente entre dos vectores. En la slide 54 aparece como circulo rojo para combinar la salida de la sigmoide con el flujo de datos en un gate.

**Position encoding (Codificacion de posicion)** - Vectores $p_0, p_1, \ldots, p_T$ que se suman al embedding para preservar el orden cuando se procesa la secuencia en paralelo. Es el paso 1 de self-attention (slide 71).

**Position-aware encoding (Codificacion consciente de posicion)** - Resultado de sumar embedding mas position encoding: vector que combina identidad y orden. Es el input efectivo del bloque self-attention (slide 71).

**Predict next word (Predecir la siguiente palabra)** - Tarea de modelado secuencial usada para introducir embeddings (slides 31-36). Ejemplo: "This morning I took my cat for a ___" -> "walk".

**Query (Q)** - Una de las tres proyecciones lineales de la entrada en self-attention. Representa lo que se busca. En la analogia de YouTube (slide 68) el query es la barra de busqueda con texto "deep learning".

**Recurrence (Recurrencia)** - Mecanismo por el cual la salida en el paso $t$ depende del estado del paso anterior. Es la idea central de las RNNs (slides 14-17, summary slide 81).

**Recurrence relation (Relacion de recurrencia)** - Ecuacion $h_t = f_W(x_t, h_{t-1})$ que define como evoluciona el estado de la RNN. Aparece en slide 17 como pieza fundacional.

**Recurrent cell (Celda recurrente)** - Modulo que computa $h_t$ a partir de $x_t$ y $h_{t-1}$ con un self-loop. Introducida explicitamente con esa etiqueta en slide 15.

**RNN (Recurrent Neural Network, Red Neuronal Recurrente)** - Red que procesa secuencias manteniendo un estado oculto $h_t$ y reusando la misma funcion $f_W$ y los mismos pesos en cada paso (slide 17).

**Scaled dot-product attention (Atencion por producto punto escalado)** - Operacion central del Transformer: $A(Q, K, V) = \text{softmax}(Q \cdot K^T / \sqrt{d}) \cdot V$. El factor de escala $\sqrt{d}$ estabiliza los gradientes (slides 73-76).

**Self-attention (Auto-atencion)** - Mecanismo donde Q, K y V provienen del mismo input, permitiendo que cada posicion atienda a todas las demas. Reemplaza la recurrencia y permite paralelizacion (slides 65-78).

**Self-loop (Bucle propio)** - Flecha que regresa de la celda hacia si misma en el diagrama RNN, indicando que el estado se realimenta al siguiente paso. Aparece etiquetado como "recurrent cell" en slide 15.

**Sentiment classification (Clasificacion de sentimiento)** - Tarea many-to-one que asigna una etiqueta (positivo / negativo) a una secuencia de palabras. Ejemplificada con "I love this class!" -> \<positive\> (slide 57) y tweets reales (slide 58).

**Sequence (Secuencia)** - Estructura ordenada de elementos $x_1, x_2, \ldots, x_T$. Ejemplos del lecture: audio, video, mercado bursatil, ADN, ECG, clima (slides 6-7).

**Sequence modeling (Modelado de secuencias)** - Tarea de aprender funciones sobre secuencias de entrada y/o salida. Es el tema del lecture entero, motivado en slide 8 con los 4 tipos de mapping.

**Sequence-to-sequence (Seq2seq, Secuencia-a-secuencia)** - Patron donde una secuencia de entrada se mapea a una secuencia de salida. Aparece implicito en machine translation (slide 8) y music generation (slide 56).

**Sigmoid (Sigmoide)** - Funcion $\sigma(z) = 1/(1+e^{-z})$ con rango $(0, 1)$. En slide 54 aparece como caja amarilla "sigmoid neural net layer" que produce los valores del gate.

**Softmax** - Funcion $\sigma(z)_i = \exp(z_i) / \sum_j \exp(z_j)$ que normaliza un vector de logits a una distribucion de probabilidad. En self-attention se aplica al puntaje escalado para producir el attention weighting (slide 75).

**Softmax cross-entropy** - Combinacion de softmax y cross-entropy que se usa como perdida en clasificacion. Aparece en slide 57 como `tf.nn.softmax_cross_entropy_with_logits`.

**Time step (Paso temporal)** - Indice $t$ que ordena los elementos de la secuencia. La slide 12 introduce la notacion $x_t, \hat{y}_t$ para marcar el paso actual.

**Token** - Unidad atomica de una secuencia textual (palabra, subword, caracter). En el ejemplo de slide 18 los tokens son "I", "love", "recurrent", "neural".

**Transformer** - Arquitectura basada enteramente en self-attention introducida por Vaswani et al. (2017). El lecture la presenta como base para BERT, GPT, AlphaFold, ESM, ViT (slides 78, 80).

**Unrolling (Desenrollado)** - Representacion de la RNN como una cadena de cells repetidas a traves del tiempo, una por cada paso. Es la base para visualizar BPTT (slide 25-26).

**Value (V)** - Una de las tres proyecciones lineales de la entrada en self-attention. Representa el contenido a extraer una vez decidido cuanto atender a cada posicion. En la analogia de YouTube (slide 69) el value es el video seleccionado.

**Vanilla RNN (RNN basica)** - Variante simple de RNN sin gates, definida por $h_t = \tanh(W_{hh}^T h_{t-1} + W_{xh}^T x_t)$. Sufre de vanishing y exploding gradients (slide 29 la nombra "Vanilla").

**Vanishing gradient (Gradiente que se desvanece)** - Patologia en la que la norma del gradiente decae exponencialmente al propagarse hacia atras a traves de muchos pasos. Ocurre cuando los factores de $W_{hh}$ tienen valores < 1 (slide 48); causa sesgo hacia dependencias de corto plazo.

**Variable-length sequences (Secuencias de longitud variable)** - Primer criterio de diseno (slide 30): el modelo debe poder procesar secuencias de tamanos distintos sin recompilar. Motivado en slide 37 con frases de 4, 6 y 9 palabras.

**ViT (Vision Transformer)** - Aplicacion de self-attention a imagenes dividiendolas en parches (Dosovitskiy et al., ICLR 2020). Citado en slide 80 con la imagen del golden retriever con grid overlay.

**Vocabulary (Vocabulario)** - Conjunto de tokens unicos del corpus. Es el primer paso del pipeline de embedding (slide 36): "this, cat, for, my, took, walk, I, a, morning".

**$W_{hh}$ (Hidden-to-hidden weight matrix)** - Matriz que multiplica al estado oculto previo $h_{t-1}$ en la actualizacion del estado: $h_t = \tanh(W_{hh}^T h_{t-1} + W_{xh}^T x_t)$. Reusada en cada paso temporal.

**$W_{hy}$ (Hidden-to-output weight matrix)** - Matriz que proyecta el estado oculto al espacio de salida: $\hat{y}_t = W_{hy}^T h_t$ (slide 24).

**$W_{xh}$ (Input-to-hidden weight matrix)** - Matriz que multiplica al input $x_t$ en la actualizacion del estado oculto. Aparece en la formula de slide 23.

---

> Material adaptado de **MIT 6.S191 (2026) Lecture 2: Deep Sequence Modeling**, Ava Amini (5 enero 2026). [Slides oficiales](http://introtodeeplearning.com) - [Sitio del curso](http://introtodeeplearning.com). Notas en espanol con investigacion complementaria. Sin afiliacion oficial con MIT.
