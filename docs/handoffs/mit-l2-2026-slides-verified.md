# MIT 6.S191 (2026) Lecture 2 — Slides verificados (Roberto leyó el PDF directamente)

> Log slide-por-slide del contenido **real** del PDF (83 páginas), generado leyendo el PDF como imágenes con la herramienta Read. Este archivo es la fuente de verdad para reescribir notas/profundización/glosario desde cero.

## Slides 1-20

| # | Título | Contenido |
|---|--------|-----------|
| 1 | (sin título) | Title slide: "Deep Sequence Modeling", **Ava Amini**, MIT Introduction to Deep Learning, **January 5, 2026** |
| 2 | (sin título) | "Given an image of a ball, can you predict where it will go next?" — solo bola |
| 3 | (sin título) | Misma pregunta, bola con **flechas en múltiples direcciones** + "???" — sin contexto, ambiguo |
| 4 | (sin título) | Misma pregunta, **secuencia de 4 bolas** (3 atenuadas + 1 actual) sin flecha — el contexto pasado |
| 5 | (sin título) | Misma pregunta, **secuencia de 4 bolas + flecha hacia derecha** — la predicción se vuelve obvia con contexto |
| 6 | Sequences in the Wild | Waveform azul + label **"Audio"** debajo (categoría) |
| 7 | Sequences in the Wild | Collage de 6 imágenes (sin labels): stock market (gráficos numéricos), video (cámara DSLR enfocando persona), DNA (secuencias ATCG), ECG (electrocardiograma), runner (atleta en motion blur), weather satellite (mapa esférico de ozono, fecha "Sep 08, 2019") |
| 8 | **Sequence Modeling Applications** | 4 columnas: **One-to-One** Binary Classification ("Will I pass this class?"), **Many-to-One** Sentiment Classification (tweet+emoji), **One-to-Many** Image Captioning ("A baseball player throws a ball"), **Many-to-Many** Machine Translation (文/A) |
| 9 | Neurons with Recurrence | Section header (slide azul) |
| 10 | The Perceptron Revisited | Diagrama: $x^{(1)}, x^{(2)}, x^{(m)}$ con pesos $w_1, w_2, w_m$ → $z$ → $y = g(z)$ → $\hat{y}$ |
| 11 | Feed-Forward Networks Revisited | Diagrama de capa input → hidden → output, $\boldsymbol{x} \in \mathbb{R}^m$, $\hat{\boldsymbol{y}} \in \mathbb{R}^n$ |
| 12 | Feed-Forward Networks Revisited | Caja simplificada: $\boldsymbol{x}_t \in \mathbb{R}^m$ → caja → $\hat{\boldsymbol{y}}_t \in \mathbb{R}^n$ (introduce subíndice de tiempo $t$) |
| 13 | Handling Individual Time Steps | Lado izq: caja con $x_t \to \hat{y}_t$. Lado der: 3 cajas separadas en $t=0,1,2$, **sin conexión entre ellas**. Fórmula: $\hat{y}_t = f(x_t)$. **Anotaciones de animación:** rectángulo azul rodeando $x_0, x_1$ (entradas vistas) y rectángulo púrpura rodeando $\hat{y}_2$ (predicción "futura" sin contexto) — Ava usa esto para señalar que sin recurrencia no se puede predecir $\hat{y}_2$ usando lo aprendido en $t=0,1$ |
| 14 | Neurons with Recurrence | Lado izq: caja con $x_t \to \hat{y}_t$. Lado der: 3 cajas conectadas con **$h_0, h_1$** que pasan información entre pasos. Fórmula: $\hat{y}_t = f(x_t, h_{t-1})$ con etiquetas **output, input, past memory** |
| 15 | Neurons with Recurrence | Igual a slide 14 + **self-loop "recurrent cell"** etiquetado en la caja izquierda con $h_t$ |
| 16 | Recurrent Neural Networks (RNNs) | Section header |
| 17 | Recurrent Neural Networks (RNNs) | Diagrama RNN con self-loop. Fórmula formal: $h_t = f_W(x_t, h_{t-1})$ con etiquetas **cell state, function with weights W, input, old state**. Nota: "the same function and set of parameters are used at every time step. RNNs have a state $h_t$ that is updated at each time step as a sequence is processed" |
| 18 | RNN Intuition | Pseudocódigo Python: `my_rnn = RNN(); hidden_state = [0,0,0,0]; sentence = ["I","love","recurrent","neural"]` (init resaltado en verde) |
| 19 | RNN Intuition | Mismo código + `for word in sentence: prediction, hidden_state = my_rnn(word, hidden_state)` (loop resaltado) |
| 20 | RNN Intuition | Mismo código + `next_word_prediction = prediction; # >>> "networks!"` (resaltado) |

**Observaciones críticas:**
- **NO HAY** slides sobre "naive approaches" (fixed window, bag of words). El lecture 2026 saltó esa sección que estaba en el 2020.
- El bloque RNN comienza en slide 9 (no 15), las aplicaciones one-to-many/many-to-many están en slide 8 (no en 16-19).
- **CORRECCIÓN al log original:** los 4 criterios de diseño SÍ existen — están en slide 30 (ver más abajo).

---

## Slides 21-40

| # | Título | Contenido |
|---|---|---|
| 21 | RNN State Update and Output | Solo el diagrama base: $\hat{y}_t$ (output, púrpura) ← RNN cell (con $h_t$ y self-loop) ← $x_t$ (input, azul). Sin texto ni fórmulas a la derecha — frame inicial del build |
| 22 | RNN State Update and Output | Igual + agrega callout azul "**Input Vector** $x_t$" a la derecha |
| 23 | RNN State Update and Output | Igual + agrega callout verde "**Update Hidden State**": $h_t = \tanh(\boldsymbol{W}_{hh}^T h_{t-1} + \boldsymbol{W}_{xh}^T x_t)$ |
| 24 | RNN State Update and Output | Igual + agrega callout púrpura "**Output Vector**": $\hat{y}_t = \boldsymbol{W}_{hy}^T h_t$ — slide **completa** con las 3 ecuaciones y las 3 matrices distintas: $W_{xh}, W_{hh}, W_{hy}$ |
| 25 | RNNs: Computational Graph Across Time | Diagrama RNN folded "=" texto: "Represent as computational graph unrolled across time" (introduce el concepto de unrolling) |
| 26 | RNNs: Computational Graph Across Time | Diagrama unrolled completo: $\hat{y}_0, \hat{y}_1, \hat{y}_2, \ldots, \hat{y}_t$ con $\boldsymbol{W}_{hy}$ subiendo a cada output; $\boldsymbol{W}_{hh}$ conectando cells horizontalmente; $\boldsymbol{W}_{xh}$ desde cada input $x_0, x_1, x_2, \ldots, x_t$. Cada timestep tiene una pérdida $L_0, L_1, L_2, L_3$ que se agregan en una pérdida total $L$ (caja naranja arriba). Texto: "Re-use the **same weight matrices** at every time step". Indicador "Forward pass" arriba izq. |
| 27 | RNNs from Scratch in TensorFlow | Código completo de `class MyRNNCell(tf.keras.layers.Layer)`. **`__init__(self, rnn_units, input_dim, output_dim)`**: inicializa 3 weight matrices con `self.add_weight([rnn_units, input_dim])` para `W_xh`, `[rnn_units, rnn_units]` para `W_hh`, `[output_dim, rnn_units]` para `W_hy`. Hidden state init a `tf.zeros([rnn_units, 1])`. **`call(self, x)`**: 3 bloques resaltados en verde — (1) `self.h = tf.math.tanh(self.W_hh * self.h + self.W_xh * x)`, (2) `output = self.W_hy * self.h`, (3) `return output, self.h`. Logo TensorFlow naranja arriba derecha. Diagrama RNN a la derecha. |
| 28 | RNN Implementation: TensorFlow & PyTorch | Dos one-liners. **TF:** `from tf.keras.layers import SimpleRNN` / `model = SimpleRNN(rnn_units)` + logo TF. **PyTorch:** `from torch.nn import RNN` / `model = RNN(input_size, rnn_units)` + logo PyTorch. Diagrama RNN a la derecha. |
| 29 | RNNs for Sequence Modeling | Grid 4-arquitecturas (estructura igual a slide 8 pero **labels distintos** orientado a RNN). **One to One** — "Vanilla" NN — *Binary classification*. **Many to One** — *Sentiment Classification*. **One to Many** — *Text Generation*, *Image Captioning*. **Many to Many** — *Translation & Forecasting*, *Music Generation*. Pie: "… and many other architectures and applications" + ⭐ **6.S191 Lab!** |
| 30 | Sequence Modeling: Design Criteria | Lista numerada explícita "To model sequences, we need to:" (1) Handle **variable-length** sequences. (2) Track **long-term** dependencies. (3) Maintain information about **order**. (4) **Share parameters** across the sequence. Cierre: "**Recurrent Neural Networks (RNNs)** meet these sequence modeling design criteria". Diagrama RNN simplificado a la derecha. |

| 31 | A Sequence Modeling Problem: Predict the Next Word | Section header (slide azul) — sin footer |
| 32 | A Sequence Modeling Problem: Predict the Next Word | Frase: **"This morning I took my cat for a walk."** — sin nada más. **Footer cambia** a `H. Suresh, 6.S191 2018.` (atribución a la versión 2018 del lecture por Harini Suresh — indica slides reciclados/adaptados) |
| 33 | A Sequence Modeling Problem: Predict the Next Word | Igual + agrega anotación: las palabras "This morning I took my cat for a" resaltadas en **verde** con label verde "**given these words**", la palabra "walk" resaltada en **naranja** |
| 34 | A Sequence Modeling Problem: Predict the Next Word | Igual + agrega label naranja "**predict the next word**" sobre "walk" |
| 35 | A Sequence Modeling Problem: Predict the Next Word | Igual + agrega segunda fila titulada **"Representing Language to a Neural Network"**. Dos cajas: ❌ "deep" → red box → "learning" *(Neural networks cannot interpret words)*  vs  ✅ vector $[0.1, 0.8, 0.6]$ → green box → $[0.9, 0.2, 0.4]$ *(Neural networks require numerical inputs)*. **Footer Suresh desaparece** en esta slide |
| 36 | Encoding Language for a Neural Network | Repite las dos cajas ❌/✅ del slide anterior arriba. Abajo subtítulo: *"Embedding: transform indexes into a vector of fixed size."* y 3 columnas: **1. Vocabulary** (Corpus of words: this, cat, for, my, took, walk, I, a, morning) → **2. Indexing** (Word to index: a→1, cat→2, …, walk→N) → **3. Embedding** (Index to fixed-sized vector). En "3. Embedding" se muestran 2 opciones: **One-hot embedding** "cat" = $[0, 1, 0, 0, 0, 0]$ con flecha $i$-th index; **Learned embedding** scatter 2D con palabras (run/walk/dog/cat alrededor de un eje y day/sun/happy/sad alrededor del otro). Sin footer Suresh |
| 37 | Handle Variable Sequence Lengths | 3 frases de longitudes muy distintas, todas con la última palabra resaltada en naranja sobre fondo verde:<br>• "The food was **great**" (4 palabras)<br>• "We visited a restaurant for **lunch**" (6 palabras)<br>• "We were hungry but cleaned the house before **eating**" (9 palabras) |
| 38 | Model Long-Term Dependencies | Frase: *"**France** is where I grew up, but I now live in Boston. I speak fluent ____."* Decoración con corazones azul/blanco/rojo (bandera francesa) + texto **"J'aime 6.S191!"** + corazones rojo/blanco/azul. Pie: *"We need information from **the distant past** to accurately predict the correct word."* Footer Suresh 2018 |
| 39 | Capture Differences in Sequence Order | Dos frases con mismo vocabulario pero orden distinto:<br>• "The food was good, not bad at all." (icono burger+drink positivo a la izquierda)<br>• vs.<br>• "The food was bad, not good at all." (icono burger+drink dentro de círculo rojo de prohibición a la derecha)<br>Footer Suresh 2018 |
| 40 | Sequence Modeling: Design Criteria | **DUPLICADO EXACTO de la slide 30** — recap de los 4 criterios después de motivar cada uno individualmente. Patrón pedagógico bookend: slide 30 introduce la lista, slides 31-39 desarrollan cada criterio con ejemplos, slide 40 cierra repitiendo la lista |

**Observaciones del bloque 21-40:**
- Las slides 32-34 y 38-39 llevan footer **`H. Suresh, 6.S191 2018.`** — son adaptadas/recicladas del lecture original 2018 por Harini Suresh. Las del 2026 (35, 36, 37, 40) dicen solo `1/5/26`.
- La slide 36 ("Encoding Language for a Neural Network") **es la única slide del lecture entero que cubre embeddings explícitamente** — vocabulary → indexing → embedding (one-hot vs learned). Material crítico que el log original había omitido o tergiversado.
- El slide 40 es **idéntico al 30** (no es typo) — es un recap intencional. Cualquier nota debe mencionar este patrón.

## Slides 41-60

| # | Título | Contenido |
|---|---|---|
| 41 | Backpropagation Through Time (BPTT) | Section header (slide azul) sin footer |
| 42 | Recall: Backpropagation in Feed Forward Models | Diagrama 3-capas (input azul $x$, hidden verde, output púrpura $y$) con flechas rojas backward entre todas las neuronas. Flecha grande negra subiendo (forward) y roja bajando (backward) al lado. Texto: **"Backpropagation algorithm:** 1. Take the derivative (gradient) of the loss with respect to each parameter. 2. Shift parameters in order to minimize loss" |
| 43 | RNNs: Backpropagation Through Time | Mismo diagrama unrolled de slide 26 (4 cells + ..., $L_0..L_3$ → $L$ total) con etiquetas $W_{xh}$, $W_{hh}$, $W_{hy}$. Solo "Forward pass" indicador (todas flechas negras) — frame inicial del build |
| 44 | RNNs: Backpropagation Through Time | Igual + agrega flechas **rojas** "Backward pass": (1) desde $L$ hacia atrás a cada $L_i$, (2) desde $L_i$ → $\hat{y}_i$ → cell, (3) entre cells horizontalmente vía $W_{hh}$. Footer cambia a **`Mozer Complex Systems 1989.`** — citación al paper original de BPTT |
| 45 | Standard RNN Gradient Flow | Diagrama simplificado horizontal: 4 cells + ... → ... `h_0`, ..., `h_t` con flechas backward rojas, etiquetas $W_{hh}$ entre cells, $W_{xh}$ subiendo desde inputs $x_0, x_1, x_2, ..., x_t$ (azul) |
| 46 | Standard RNN Gradient Flow | Igual + agrega texto inferior: *"Computing the gradient wrt $h_0$ involves **many factors of $W_{hh}$** + **repeated gradient computation**!"* |
| 47 | Standard RNN Gradient Flow: **Exploding Gradients** | Igual + agrega caja redondeada: "Many values > 1: **exploding gradients**. **Gradient clipping** to scale big gradients" |
| 48 | Standard RNN Gradient Flow: **Vanishing Gradients** | Igual + caja "Many values < 1: **vanishing gradients**" con lista numerada: 1. Activation function, 2. Weight initialization, 3. Network architecture. La caja del slide anterior (exploding) queda **atenuada/grisada** al fondo (build acumulativo) |
| 49 | The Problem of Long-Term Dependencies | Flow chart vertical: "Why are vanishing gradients a problem?" → "Multiply **small numbers** together" → ↓ → "Errors due to further back time steps have smaller and smaller gradients" → ↓ → "**Bias parameters to capture short-term dependencies**" |
| 50 | The Problem of Long-Term Dependencies | Igual + agrega frase upper-right: *"The clouds are in the ___"* (sin diagrama todavía) |
| 51 | The Problem of Long-Term Dependencies | Igual + agrega diagrama RNN derecha: 5 cells unrolled, outputs $\hat{y}_0..\hat{y}_4$, inputs $x_0..x_4$. **$\hat{y}_3$ resaltado teal** (la predicción de "sky") y **$x_0, x_1$ resaltados teal** (las palabras informativas "The clouds") — distancia corta = dependencia fácil |
| 52 | The Problem of Long-Term Dependencies | Igual + agrega segundo ejemplo abajo: *"I grew up in France, … and I speak fluent ___"* (sin diagrama todavía) |
| 53 | The Problem of Long-Term Dependencies | Igual + agrega segundo diagrama RNN abajo: cells extendidos $x_0, x_1, ..., x_t, x_{t+1}$ con $\hat{y}_t$ resaltado **naranja** (predicción "French") y $x_0, x_1$ resaltados **naranja** (palabras informativas "France") — distancia LARGA = dependencia difícil. Contraste visual entre ambas |
| 54 | Gating Mechanisms in Neurons | "Idea: use **gates** to selectively **add** or **remove** information within **each recurrent unit** with". Componentes legenda: **σ** (sigmoid neural net layer, caja amarilla) → **×** (pointwise multiplication, círculo rojo). Caja verde grande "**gated cell** — LSTM, GRU, etc.". Caption: "Gates optionally let information through the cell". Bottom: *"**Long Short Term Memory (LSTMs)** networks rely on a gated cell to track information throughout many time steps."* — **única slide del lecture donde se mencionan LSTMs/GRUs explícitamente** |
| 55 | RNN Applications & Limitations | Section header (slide azul) sin footer |
| 56 | Example Task: Music Generation | RNN con 4 cells: inputs (notas en azul) E, F#, G, C → outputs (notas en púrpura) F#, G, C, A. Lado derecho: **Input:** sheet music. **Output:** next character in sheet music. Caja amarilla con imagen "Listening to 3rd movement" + botón ▶ (video del Schubert generado por RNN del MIT). Badge ⭐ **6.S191 Lab!**. Footer: **`Huawei.`** (atribución) |
| 57 | Example Task: Sentiment Classification | RNN con 4 cells: inputs (palabras) "I love this class!" → output sentiment **\<positive\>** (círculo púrpura). **Input:** sequence of words. **Output:** probability of having positive sentiment. Snippet código TF: `loss = tf.nn.softmax_cross_entropy_with_logits(y, predicted)`. Footer: **`Socher+, EMNLP 2013.`** |
| 58 | Example Task: Sentiment Classification | Igual + agrega "**Tweet sentiment classification**" panel derecho con dos tweets reales: (1) Ivar Hagendoorn "@MIT Introduction to #DeepLearning is definitely one of the best courses..." con emoji 😃, (2) Angels-Cave "I wouldn't mind a bit of snow right now. We haven't had any..." con emoji 😢. Footer: **`H. Suresh, 6.S191 2018.`** |
| 59 | Limitations of Recurrent Models | Mismo diagrama RNN sentiment de izquierda. Panel derecho titulado "**Limitations of RNNs**" con 3 items + iconos: 🪣 **Encoding bottleneck**, ⏰ **Slow, no parallelization**, 🧠 **Not long memory** — **slide bisagra hacia Transformers** |
| 60 | Goal of Sequence Modeling | Diagrama de 3 capas con 6 timesteps ($x_0, x_1, x_2, ..., x_{t-2}, x_{t-1}, x_t$). Capa **input** (azul) → capa **feature vector** (amarillas barras horizontales conectadas con flechas →) → capa **output** $\hat{y}_*$ (púrpura). Tres labels izquierda: "Sequence of inputs", "Sequence of features", "Sequence of outputs". Texto superior: "**RNNs: recurrence to model sequence dependencies**". Eje t debajo. Es el setup conceptual para introducir attention/transformers |

**Observaciones del bloque 41-60:**
- Slide 44 cita Mozer 1989 (paper fundacional de BPTT). Slide 56 cita Huawei (origen del demo de música). Slide 57 cita Socher+ EMNLP 2013 (paper de sentiment con RNN).
- Slide 47 ↔ 48 son un patrón de comparación lado-a-lado de exploding vs vanishing gradients (la caja anterior queda atenuada cuando aparece la nueva).
- Slides 50-53 construyen el ejemplo dual "clouds in the sky" (corta) vs "I speak French" (larga) con código de color teal/naranja para visualizar visualmente cuándo el RNN puede vs no puede.
- **Slide 54 es la única mención de LSTM/GRU en todo el lecture 2026** — Ava no entra en detalles internos del gate como hacía la versión 2020. El lecture pasa de "gates existen" directamente a las limitaciones de RNN (slide 59) y al pivot hacia Transformers.
- Slide 59 con los 3 iconos de limitaciones es la **bisagra explícita** hacia Self-Attention/Transformers (que viene en 61+).

## Slides 61-83

| # | Título | Contenido |
|---|---|---|
| 61 | Goal of Sequence Modeling | Mismo diagrama de slide 60 (input azul → feature vector amarillo → output púrpura, 6 timesteps con flechas → entre cells) + agrega panel izquierdo "**Limitations of RNNs**" con los 3 iconos: 🪣 Encoding bottleneck, ⏰ Slow, no parallelization, 🧠 Not long memory. Slide bisagra: el diagrama es el ideal y al lado se recuerdan los problemas de RNNs |
| 62 | Goal of Sequence Modeling | Igual + **reemplaza** el panel izquierdo por "**Desired Capabilities**" con 3 iconos: 🌊 Continuous stream, ⤴⤵ Parallelization, 🧠 Long memory. Texto superior agrega: "**Can we eliminate the need for recurrence entirely?**" |
| 63 | Goal of Sequence Modeling | Igual texto superior + el diagrama cambia: ahora **input es una sola barra horizontal** (no celdas separadas), feature vector es una sola barra horizontal sin flechas → entre timesteps, output también barra horizontal. El icono de "Long memory" se vuelve un solo bloque $x_0$ — todos los inputs colapsados en uno solo |
| 64 | Goal of Sequence Modeling | Igual + agrega panel izquierdo "**Idea 1: Feed everything into dense network**" con: ✅ No recurrence, ❌ Not scalable, ❌ No order, ❌ No long memory. Abajo: 🧠 "**Idea: Identify and attend to what's important**" — pivot explícito a self-attention |
| 65 | Attention Is All You Need | Section header (slide azul) — referencia directa al título del paper Vaswani 2017 |
| 66 | Intuition Behind Self-Attention | "Attending to the most important parts of an input." Imagen de Iron Man volando con flechas horizontales blancas atravesando la imagen (representando atención sobre regiones). Caja resaltada "1. **Identify** which parts to attend to" con flecha azul a "Similar to a search problem!". Item 2: "Extract the features with high attention" |
| 67 | A Simple Example: Search | Emoji 🤔 (thinking face) con thought bubble "**How can I learn more about neural networks?**". Lado derecho: collage de muchas miniaturas de videos/imágenes (representa el corpus a buscar). Sin footer |
| 68 | Understanding Attention with Search | Mockup de YouTube en dark mode con search box "deep learning" (resaltado azul como **Query (Q)**). 3 resultados: (1) "GIANT SEA TURTLES • AMAZING CORAL REEF FISH..." atenuado como **Key (K₁)**; (2) **"MIT 6.S191 (2020): Introduction to Deep Learning"** por Alexander Amini resaltado en naranja como **Key (K₂)**; (3) "The Kobe Bryant Fadeaway Shot" atenuado como **Key (K₃)**. Llaves laterales agrupando los 3 keys con texto "How similar is the key to the query?". Bottom: "**1. Compute attention mask:** how similar is each key to the desired query?" |
| 69 | Understanding Attention with Search | Igual + el resultado seleccionado (MIT) ahora tiene además un **box púrpura** alrededor etiquetado **Value (V)** — el contenido real extraído. Bottom cambia a: "**2. Extract values based on attention:** Return the values highest attention" |
| 70 | Learning Self-Attention with Neural Networks | "Goal: identify and attend to most important features in input." Lista numerada con **solo step 1 resaltado** (resto en gris claro): **1. Encode position information**, 2. Extract query, key, value for search, 3. Compute attention weighting, 4. Extract features with high attention. Lado derecho: barra azul $x$ con palabras "He / tossed / the / tennis / ball / to / serve" debajo. Bottom: "**Data is fed in all at once! Need to encode position information to understand order.**" |
| 71 | Learning Self-Attention with Neural Networks | Igual + visualización del position encoding: input "He tossed..." → 7 columnas de **embedding** (azul vertical) **⊕** (suma) **position information** $p_0, p_1, ..., p_6$ → 7 columnas verdes "**Position-aware encoding**". Step 1 resaltado |
| 72 | Learning Self-Attention with Neural Networks | Igual + agrega step 2 resaltado "**Extract query, key, value for search**". Lado derecho: 3 multiplicaciones de matrices: **Positional embedding** (verde) × **Linear layer** (color) = **Output** (color). Tres copias: una azul → **Q Query**, una naranja → **K Key**, una púrpura → **V Value**. Footer: **`Vaswani+, NeurIPS 2017.`** |
| 73 | Learning Self-Attention with Neural Networks | Igual + step 3 resaltado "**Compute attention weighting**". Lado derecho cambia: "**Attention score:** compute pairwise similarity between each **query** and **key**". Pregunta: "How to compute similarity between two sets of features?". Visualización de dos vectores **Q** (azul) y **K** (naranja) con flechas desde origen. "**Dot product**" → caja roja punteada con fórmula $\dfrac{Q \cdot K^T}{\text{scaling}}$ = "**Similarity metric**". Caption: "Also known as the 'cosine similarity'" |
| 74 | Learning Self-Attention with Neural Networks | Igual + reemplaza la visualización de vectores por **matrices completas**: matriz $Q$ (azul) ⋅ matriz $K^T$ (naranja) → fórmula $\dfrac{Q \cdot K^T}{\text{scaling}}$. Mismo step 3 resaltado |
| 75 | Learning Self-Attention with Neural Networks | Igual step 3 + visualización clave: **matriz de atención 7×7** (gradientes rojos, diagonal más oscura) con filas/columnas labeled "He / tossed / the / tennis / ball / to / serve". Fórmula final: $\text{softmax}\left(\dfrac{Q \cdot K^T}{\text{scaling}}\right)$ — etiquetada "**Attention weighting**" en rojo |
| 76 | Learning Self-Attention with Neural Networks | Igual + step 4 resaltado "**Extract features with high attention**" (ahora todos los 4 steps están en negro). Lado derecho: matriz **Attention weighting** (roja) × matriz **Value** (púrpura, V) = matriz **Output** (gris). Fórmula completa: $\text{softmax}\left(\dfrac{Q \cdot K^T}{\text{scaling}}\right) \cdot V = A(Q, K, V)$ con cada componente subrayado en su color |
| 77 | Learning Self-Attention with Neural Networks | Igual + reemplaza visualización por **diagrama completo del self-attention head** (estilo Vaswani): **Positional Encoding** (verde) → 3 **Linear** layers (Query azul, Key naranja, Value púrpura) → **MatMul** → **Scale** → **Softmax** → **Matmul** final → output. Caja roja inferior izquierda: "**These operations form a self-attention head that can plug into a larger network. Each head attends to a different part of input.**" Fórmula en caja roja inferior derecha. **Esta es la slide donde se ensambla TODO** |
| 78 | Learning Self-Attention with Neural Networks | Mismo diagrama del head completo + frase punchline inferior centrada: "**Attention is the foundational building block of the Transformer architecture.**" (sin la caja roja anterior) |
| 79 | Applying Multiple Self-Attention Heads | Visualización con 3 imágenes de Iron Man arriba: **Attention weighting** (silueta blanca de Iron Man sobre gris) × **Value** (escena completa Iron Man) = **Output** (Iron Man enfocado). Debajo, 3 outputs distintos: "**Output of attention head 1**" (cara/casco de Iron Man), "**Output of attention head 2**" (edificio/fondo), "**Output of attention head 3**" (objeto en la distancia/contexto distinto). Visualiza que múltiples heads atienden a partes distintas del input |
| 80 | Self-Attention Applied | 3 columnas con aplicaciones: **Language Processing** — imagen avocado-armchair "An armchair in the shape of an avocado" — **Transformers: BERT, GPT** — refs: *Devlin et al., NAACL 2019* / *Brown et al., NeurIPS 2020* + ⭐ **6.S191 Lab and Lectures!**. **Biological Sequences** — imagen estructura proteínica 3D — **Protein Structure Models** — refs: *Jumper et al., Nature 2021* (AlphaFold) / *Lin et al., Science 2023* (ESM). **Computer Vision** — golden retriever en pasto con grid overlay — **Vision Transformers** — ref: *Dosovitskiy et al., ICLR 2020* (ViT) |
| 81 | Deep Learning for Sequence Modeling: Summary | Lista numerada de 6 takeaways:<br>1. RNNs are well suited for **sequence modeling** tasks<br>2. Model sequences via a **recurrence relation**<br>3. Training RNNs with **backpropagation through time**<br>4. Models for **music generation**, classification, machine translation, and more<br>5. Self-attention to model **sequences without recurrence**<br>6. Self-attention is the basis for many **large language models** – stay tuned!<br>Fondo decorativo: waveform multicolor (espectro arcoíris) |
| 82 | 6.S191: Introduction to Deep Learning | Slide negra con waveform arcoíris. "**Lab 1: Deep Learning in Python and Music Generation with RNNs**". Link: `http://introtodeeplearning.com#schedule`. Pasos: 1. Open the lab in Google Colab, 2. Start executing code blocks and filling in the #TODOs, 3. Need help? Find a TA/instructor! |
| 83 | 6.S191: Introduction to Deep Learning | Slide negra con waveform arcoíris. "**Kickoff Reception at One Kendall Square!**". Register here: `luma.com/47iswo1i`. "Join us at 5:00pm for a special kickoff reception at 1 Kendall Square! Food and drinks will be provided! Special thanks to John Werner and Link Ventures for hosting." — slide específica del evento MIT (no contenido pedagógico) |

**Observaciones del bloque 61-83:**
- **Slides 61-64 son la transición pedagógica clave:** parten de "limitations of RNNs" (61), formulan el "what we want" (62), muestran el primer intento naïve "feed everything into dense" (63-64), y terminan con la idea de "identify and attend" — perfecto pivot a self-attention.
- **Slides 70-78 son el corazón técnico de self-attention** (9 slides). Es un build progresivo de los 4 pasos: encode position → Q,K,V → similarity scoring → softmax weighting → multiply by V → output. Construyen toda la fórmula $\text{softmax}(QK^T/\sqrt{d}) \cdot V$ paso a paso, terminando con el diagrama completo del attention head.
- **Footer Vaswani+ NeurIPS 2017** aparece en slides 72, 73 (ausente), 75, 76 — citación al paper "Attention Is All You Need". Footer Suresh 2018 ya no aparece desde slide 60 (todo el bloque 60-83 es contenido nuevo o de Ava 2026).
- **Slide 79 con Iron Man** visualiza multi-head attention de forma muy intuitiva (cada head ve parte distinta de la imagen). NO es del paper original, es invención didáctica de Ava.
- **Slide 80 cita 5 papers** clave para conectar el lecture con aplicaciones reales: BERT (Devlin), GPT (Brown), AlphaFold (Jumper), ESM (Lin), ViT (Dosovitskiy). El emoji ⭐ "**6.S191 Lab and Lectures!**" apunta a labs de Transformers/LLMs en clases siguientes.
- **Slide 81 es el resumen final pedagógico** — 6 takeaways en 1 slide. Esto es lo que el alumno debería poder responder al terminar el lecture.
- **Slides 82-83 son administrativas** (lab logistics + reception) — no aportan contenido de ML pero confirman que este es el primer lecture técnico del programa intensivo (Lab 1 = el primer lab, Kickoff = inicio del bootcamp).

---

## Resumen estructural del lecture completo (83 slides)

| Bloque | Slides | Tema |
|---|---|---|
| Intro motivacional | 1-8 | Pelota → secuencias en la naturaleza → 4 tipos one-to-one/many-to-one/etc |
| Construcción del RNN | 9-23 | Perceptron → feed-forward → time steps → recurrence → fórmulas $h_t, \hat{y}_t$ |
| Computational graph | 24-29 | Unrolling, BPTT graph, código TF/PyTorch, RNN para sequence modeling |
| Design criteria + motivación | 30-40 | 4 criterios → predict next word → embeddings → variable-length / long-term / order |
| BPTT mecánico | 41-44 | Backprop en feed-forward → BPTT con flechas backward |
| Vanishing/exploding gradients | 45-53 | Gradient flow → clipping → vanishing problem → ejemplos clouds vs French |
| Gating (LSTM/GRU teaser) | 54 | Gates como solución (1 sola slide, sin detalles internos) |
| Aplicaciones RNN | 55-58 | Music generation, sentiment classification |
| Limitaciones → pivot | 59-64 | Limitations → desired capabilities → dense fail → "attend" |
| Self-attention conceptual | 65-69 | Iron Man intuition → YouTube search Q/K/V analogy |
| Self-attention técnico | 70-78 | 4 pasos build: position encoding → Q,K,V → similarity → softmax → multiply → diagrama completo |
| Multi-head + aplicaciones | 79-80 | Multi-head Iron Man → BERT/GPT/AlphaFold/ViT |
| Cierre | 81-83 | Summary 6 takeaways → Lab 1 → reception |
