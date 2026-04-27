---
title: "Notas - MIT 6.S191 (2026) Deep Sequence Modeling"
weight: 10
math: true
---

> Recorrido tematico de las 83 diapositivas del lecture, organizado por contenido (no por slide). Citas a slides especificas en *cursiva*.

**Video original:** [YouTube](https://www.youtube.com/watch?v=d02VkQ9MP44)
**Slides:** [PDF local](/videos/mit-6s191-l2-2026/slides.pdf)
**Lecturer:** Ava Amini, MIT Introduction to Deep Learning, 5 de enero de 2026.

El lecture mantiene el armazon clasico del 2020 (motivacion -> RNN -> BPTT -> vanishing -> aplicaciones) pero a partir de la slide 60 pivota explicitamente hacia **self-attention** y termina con una vista panoramica del Transformer y sus aplicaciones modernas (BERT, GPT, AlphaFold, ViT). LSTMs y GRUs ocupan **una sola slide** (la 54) y se mencionan apenas como teaser: el enfasis pedagogico se desplazo del gating a la atencion.

---

## 1. Motivacion: por que modelar secuencias *(slides 1-8)*

El lecture abre con una **metafora visual progresiva** alrededor de una pelota *(slides 2-5)*. Primero aparece la pelota sola con la pregunta "dada una imagen de una pelota, puedes predecir donde ira despues?". La respuesta natural es ambigua, lo que se ilustra agregando flechas en multiples direcciones y signos "???" *(slide 3)*. Luego se introduce el contexto: una secuencia de cuatro pelotas atenuadas, sin flecha *(slide 4)*. Finalmente, con esa secuencia mas una flecha hacia la derecha, la prediccion se vuelve obvia *(slide 5)*. La leccion implicita: **lo que convierte un problema ambiguo en uno tratable es la informacion temporal pasada**.

A continuacion, "Sequences in the Wild" muestra que las secuencias estan en todas partes: una waveform de audio *(slide 6)*, y un collage con stocks, video, ADN, ECG, un corredor en motion blur y un mapa satelital de ozono *(slide 7)*. La idea es que el dominio "secuencia" cubre desde fonemas y bases nucleotidicas hasta frames de video y series financieras.

La motivacion se cierra con un **catalogo de las 4 configuraciones canonicas** de tareas secuenciales *(slide 8, "Sequence Modeling Applications")*:

- **One-to-One** -- *Binary Classification* ("Will I pass this class?").
- **Many-to-One** -- *Sentiment Classification* sobre un tweet con emoji.
- **One-to-Many** -- *Image Captioning* (un baseball player lanzando una pelota).
- **Many-to-Many** -- *Machine Translation* con caracteres `文` y `A`.

Este grid reaparecera mas adelante *(slide 29)* relabelado en terminos de RNN concretas, con un badge `6.S191 Lab!` apuntando al lab de generacion de musica.

> Nota: a diferencia del lecture 2020, **no hay slides sobre fixed-window ni bag-of-words**. El 2026 saltea esa critica de "naive approaches" y va directo de la motivacion al perceptron.

---

## 2. Construccion del RNN: del perceptron a la recurrencia *(slides 9-23)*

La construccion del RNN se hace **incrementalmente**, partiendo de objetos ya conocidos y agregando una pieza por slide. El objetivo pedagogico es mostrar que la recurrencia no aparece de la nada: emerge naturalmente cuando uno intenta procesar muchos pasos temporales con feed-forward.

### 2.1 Del perceptron a la caja temporal *(slides 10-12)*

Se parte de un perceptron clasico con entradas $x^{(1)}, x^{(2)}, \ldots, x^{(m)}$, pesos $w_1, \ldots, w_m$, suma $z$ y activacion $\hat{y} = g(z)$ *(slide 10)*. Luego se generaliza a una **feed-forward network** con $\boldsymbol{x} \in \mathbb{R}^m$ y $\hat{\boldsymbol{y}} \in \mathbb{R}^n$ *(slide 11)*. Finalmente, esa red se colapsa visualmente en una **caja unica** con subindice de tiempo: $\boldsymbol{x}_t \to \text{caja} \to \hat{\boldsymbol{y}}_t$ *(slide 12)*. Es la primera vez que aparece la idea de "tiempo".

### 2.2 Time steps independientes y el problema *(slide 13)*

Se replica la caja en $t=0, 1, 2$ pero **sin conexion entre cajas**. La formula es deliberadamente trivial: $\hat{y}_t = f(x_t)$. Ava enmarca con un rectangulo azul los inputs vistos $x_0, x_1$ y con un rectangulo purpura el output futuro $\hat{y}_2$ para hacer evidente que **sin recurrencia, lo aprendido en $t=0,1$ no puede informar la prediccion en $t=2$**. Esa es la motivacion concreta para introducir un canal de memoria.

### 2.3 Recurrencia y self-loop *(slides 14-15)*

El siguiente paso introduce los **estados ocultos** $h_0, h_1$ que pasan informacion entre cajas, con la formula

$$\hat{y}_t = f(x_t, h_{t-1})$$

etiquetada como **output, input, past memory** *(slide 14)*. El cambio crucial respecto del slide 13 es que ahora cada caja recibe no solo $x_t$ sino tambien la "memoria pasada" $h_{t-1}$. La slide siguiente *(15)* introduce el **`recurrent cell`** como un solo nodo con un `self-loop` etiquetado $h_t$: la representacion compacta del mismo computo.

### 2.4 RNN formal y `pseudocodigo` *(slides 16-20)*

La definicion formal aparece en *slide 17*:

$$h_t = f_W(x_t, h_{t-1})$$

con etiquetas **cell state, function with weights W, input, old state**. La nota es importante: "the same function and set of parameters are used at every time step. RNNs have a state $h_t$ that is updated at each time step". Aqui aparece, sin gritarlo, el principio de **parameter sharing**.

Para anclar la idea en codigo, se construye un pseudocodigo Python en 3 slides *(18-20)* que reproduce literalmente el espiritu de la recurrencia:

```python
my_rnn = RNN()
hidden_state = [0, 0, 0, 0]
sentence = ["I", "love", "recurrent", "neural"]

for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)

next_word_prediction = prediction
# >>> "networks!"
```

El highlight verde se mueve progresivamente entre la inicializacion, el loop y la prediccion final, dejando claro que el `hidden_state` se reusa de iteracion en iteracion.

### 2.5 Las tres ecuaciones del RNN *(slides 21-24)*

El bloque cierra con un build progresivo titulado **"RNN State Update and Output"**. La slide 21 muestra solo el diagrama base: $\hat{y}_t$ (output, purpura) <- `RNN cell` (con $h_t$ y self-loop) <- $x_t$ (input, azul). En las slides siguientes se agregan callouts uno por uno:

- *(slide 22)* **Input Vector**: $x_t$.
- *(slide 23)* **Update Hidden State**:
  $$h_t = \tanh(\boldsymbol{W}_{hh}^T h_{t-1} + \boldsymbol{W}_{xh}^T x_t)$$
- *(slide 24)* **Output Vector**:
  $$\hat{y}_t = \boldsymbol{W}_{hy}^T h_t$$

Esta es la primera vez que se explicitan **tres matrices distintas** -- $W_{xh}$, $W_{hh}$, $W_{hy}$ -- y se escoge `tanh` como activacion. La eleccion reaparecera mas tarde como pieza relevante en la discusion de vanishing gradients.

---

## 3. Computational graph y codigo (TF/PyTorch) *(slides 24-29)*

### 3.1 Unrolling

Una vez tenemos las ecuaciones, conviene **desenrollar** el RNN en el tiempo *(slide 25)*. La slide muestra el RNN folded con un `=` y el texto "Represent as computational graph unrolled across time": una sola idea que sin embargo es la base para entender BPTT, parameter sharing y vanishing gradients.

El unrolled completo *(slide 26)* tiene:

- Los outputs $\hat{y}_0, \hat{y}_1, \hat{y}_2, \ldots, \hat{y}_t$ con $\boldsymbol{W}_{hy}$ subiendo a cada uno.
- $\boldsymbol{W}_{hh}$ conectando cells horizontalmente.
- $\boldsymbol{W}_{xh}$ desde cada input $x_0, x_1, x_2, \ldots, x_t$.
- Una **perdida por timestep** $L_0, L_1, L_2, L_3$ que se agrega en una **perdida total** $L$ (caja naranja arriba).
- Un indicador "Forward pass" arriba a la izquierda.

El texto clave: "Re-use the **same weight matrices** at every time step". Aqui se materializa la idea de parameter sharing como **literal**: las flechas de $W_{hh}$ entre cells son la misma matriz, las flechas de $W_{xh}$ desde cada input son la misma matriz, etc.

### 3.2 RNN desde cero en TensorFlow *(slide 27)*

Para mostrar que la matematica se traduce directo a codigo, Ava muestra una clase `MyRNNCell(tf.keras.layers.Layer)` completa:

- En `__init__(self, rnn_units, input_dim, output_dim)` se inicializan tres matrices con `self.add_weight`:
  - `W_xh` con shape `[rnn_units, input_dim]`,
  - `W_hh` con shape `[rnn_units, rnn_units]`,
  - `W_hy` con shape `[output_dim, rnn_units]`.
  - El estado oculto se inicializa con `tf.zeros([rnn_units, 1])`.
- En `call(self, x)` aparecen tres bloques resaltados en verde que mapean uno-a-uno con las ecuaciones del slide 23-24:

  ```python
  self.h = tf.math.tanh(self.W_hh * self.h + self.W_xh * x)
  output = self.W_hy * self.h
  return output, self.h
  ```

La intencion pedagogica es eliminar la magia: la implementacion **es exactamente lo que dice la formula**.

### 3.3 Los one-liners *(slide 28)*

Acto seguido se muestra que en la practica casi nadie escribe RNNs a mano. Dos snippets de una linea cada uno:

- **TensorFlow**: `from tf.keras.layers import SimpleRNN; model = SimpleRNN(rnn_units)`.
- **PyTorch**: `from torch.nn import RNN; model = RNN(input_size, rnn_units)`.

El framework abstrae el `for word in sentence` y la actualizacion del `hidden_state`.

### 3.4 RNNs para sequence modeling *(slide 29)*

La slide reaparece el grid de cuatro arquitecturas *(estructura igual al slide 8 pero con labels distintos orientados a RNN)*:

- **One to One** -- "Vanilla" NN -- *Binary classification*.
- **Many to One** -- *Sentiment Classification*.
- **One to Many** -- *Text Generation* / *Image Captioning*.
- **Many to Many** -- *Translation & Forecasting* / *Music Generation*.

El pie cierra con "... and many other architectures and applications" mas un badge `6.S191 Lab!` apuntando al lab practico.

---

## 4. Criterios de diseno + predict-next-word + embeddings *(slides 30-40)*

### 4.1 Los 4 criterios de diseno *(slide 30)*

Despues de tener el RNN como objeto, Ava plantea explicitamente los **criterios que cualquier modelo de secuencias debe cumplir** *(slide 30, "Sequence Modeling: Design Criteria")*:

1. Manejar **variable-length** sequences.
2. Rastrear **long-term** dependencies.
3. Mantener informacion sobre el **order**.
4. **Share parameters** a lo largo de la secuencia.

El cierre es retorico: "Recurrent Neural Networks (RNNs) meet these sequence modeling design criteria". Es el setup perfecto para mostrar **uno por uno** por que cada criterio importa.

### 4.2 Predict the next word *(slides 31-34)*

El section header *(slide 31)* introduce el problema canonico: predict the next word. La frase ejemplo es "**This morning I took my cat for a walk.**" *(slide 32)*. Las slides 33-34 anotan la frase: las palabras "This morning I took my cat for a" en verde con label "given these words", y la palabra "walk" en naranja con label "predict the next word".

> *Detalle de procedencia:* las slides 32-34 cargan el footer **`H. Suresh, 6.S191 2018.`** -- son slides recicladas/adaptadas del lecture 2018 por Harini Suresh. No es contenido nuevo de Ava 2026, pero sigue siendo la ilustracion canonica del problema.

### 4.3 Embeddings: representando lenguaje numericamente *(slides 35-36)*

Para que la red pueda procesar palabras, primero hay que **convertirlas en numeros**. La slide 35 contrasta dos opciones:

- ❌ "deep" -> caja roja -> "learning" (las redes no interpretan palabras).
- ✅ vector $[0.1, 0.8, 0.6]$ -> caja verde -> $[0.9, 0.2, 0.4]$ (las redes requieren inputs numericos).

La slide 36 ("Encoding Language for a Neural Network") es la **unica slide del lecture entero que cubre embeddings explicitamente**, y el flujo es claro:

1. **Vocabulary** -- corpus de palabras (this, cat, for, my, took, walk, I, a, morning).
2. **Indexing** -- mapeo word-to-index (a -> 1, cat -> 2, ..., walk -> N).
3. **Embedding** -- index -> vector de tamano fijo.

Para el paso 3 se muestran dos opciones:

- **One-hot embedding**: `cat = [0, 1, 0, 0, 0, 0]` (un 1 en el i-th index).
- **Learned embedding**: scatter 2D donde palabras semanticamente cercanas (`run`, `walk`, `dog`, `cat`) se agrupan en un eje, y palabras de otra dimension semantica (`day`, `sun`, `happy`, `sad`) en otro.

La intuicion: con learned embeddings, **distancia geometrica refleja similitud semantica**, mientras que one-hot las hace todas equidistantes.

### 4.4 Los criterios, ilustrados uno por uno *(slides 37-39)*

Las siguientes tres slides aterrizan los criterios 1-3 con ejemplos concretos:

- **Variable-length** *(slide 37)*: tres frases de longitudes muy distintas, todas con la ultima palabra resaltada en naranja:
  - "The food was **great**" (4 palabras)
  - "We visited a restaurant for **lunch**" (6 palabras)
  - "We were hungry but cleaned the house before **eating**" (9 palabras)

  El modelo debe poder lidiar con todas sin padding artificial.

- **Long-term dependencies** *(slide 38)*: la frase "**France** is where I grew up, but I now live in Boston. I speak fluent ____." Decoraciones con corazones y "J'aime 6.S191!" enfatizan que la palabra clave ("France") esta a docenas de tokens de distancia de la prediccion. La leccion: "We need information from **the distant past** to accurately predict the correct word."

- **Order** *(slide 39)*: dos frases con el **mismo vocabulario** pero orden distinto:
  - "The food was good, not bad at all." (icono burger positivo)
  - "The food was bad, not good at all." (icono burger en circulo de prohibicion)

  Bag-of-words no podria distinguirlas; un RNN si, porque procesa el orden secuencialmente.

### 4.5 Recap *(slide 40)*

La slide 40 es **un duplicado exacto del slide 30**. No es un error: es un patron pedagogico bookend. Slide 30 introduce los 4 criterios; slides 31-39 los desarrollan uno por uno; slide 40 los repite para cerrar el bloque.

---

## 5. Backpropagation Through Time mecanico *(slides 41-44)*

La seccion BPTT comienza con un section header *(slide 41)* y un recordatorio del backprop estandar en feed-forward *(slide 42)*: diagrama de tres capas (input azul, hidden verde, output purpura) con flechas rojas backward y el algoritmo en dos pasos -- "1. Take the derivative (gradient) of the loss with respect to each parameter. 2. Shift parameters in order to minimize loss".

Sobre el unrolled RNN *(slide 43)* se vuelve a mostrar el grafo con $W_{xh}$, $W_{hh}$, $W_{hy}$ y las perdidas $L_0..L_3 \to L$, ahora con el indicador "Forward pass" en negro. La slide 44 agrega progresivamente las **flechas rojas backward**:

1. Desde la perdida total $L$ hacia atras a cada $L_i$.
2. Desde cada $L_i$ a su $\hat{y}_i$ y de ahi al cell.
3. **Entre cells horizontalmente**, viajando via $W_{hh}$.

El footer en este punto cambia a `Mozer Complex Systems 1989.`, citacion al paper original que formaliza BPTT. La idea visual: el gradiente respecto a un peso recibe contribuciones de **todos** los timesteps, porque el peso aparece en todas las transiciones.

---

## 6. Vanishing/exploding gradients y long-term dependencies *(slides 45-53)*

### 6.1 Standard RNN gradient flow *(slides 45-46)*

Se simplifica el diagrama a un flujo horizontal: 4 cells -> ... -> $h_t$ con flechas backward rojas, $W_{hh}$ entre cells y $W_{xh}$ subiendo desde inputs *(slide 45)*. Se anota el problema *(slide 46)*: "Computing the gradient wrt $h_0$ involves **many factors of $W_{hh}$** + **repeated gradient computation**".

Este es el corazon del problema: cuando el gradiente viaja $T$ pasos hacia atras, se multiplica por $W_{hh}$ aproximadamente $T$ veces.

### 6.2 Exploding y vanishing en paralelo *(slides 47-48)*

La slide 47 enmarca el primer fallo: "Many values > 1: **exploding gradients**. **Gradient clipping** to scale big gradients" en una caja redondeada al lado del diagrama.

La slide 48 introduce el caso opuesto, **vanishing gradients**: "Many values < 1: vanishing gradients" con una lista numerada de mitigaciones:

1. Activation function.
2. Weight initialization.
3. Network architecture.

Visualmente, la caja de "exploding" del slide anterior queda **atenuada/grisada al fondo** -- es un patron de comparacion lado-a-lado. La nueva caja (vanishing) queda al frente con sus tres lineas de defensa.

### 6.3 Por que es un problema *(slide 49)*

Un flow chart vertical aterriza la consecuencia practica:

- "Why are vanishing gradients a problem?"
- -> "Multiply small numbers together"
- -> "Errors due to further back time steps have smaller and smaller gradients"
- -> "Bias parameters to capture short-term dependencies"

En otras palabras: el RNN **aprende a confiar solo en lo cercano** y se vuelve incapaz de modelar dependencias largas.

### 6.4 El ejemplo dual: "clouds in the sky" vs "I speak French" *(slides 50-53)*

La forma mas convincente de explicar long-term dependencies es contrastando dos casos:

- **Distancia corta** *(slides 50-51)*: "The clouds are in the ___". El diagrama RNN tiene 5 cells unrolled. La prediccion $\hat{y}_3$ esta resaltada **teal**, las palabras informativas $x_0, x_1$ ("The clouds") tambien estan resaltadas teal. La distancia entre informacion y prediccion es de pocos pasos: el RNN puede manejarlo.

- **Distancia larga** *(slides 52-53)*: "I grew up in France, ... and I speak fluent ___". El diagrama agrega cells extendidos $x_0, x_1, ..., x_t, x_{t+1}$. Ahora $\hat{y}_t$ se resalta **naranja** y las palabras informativas $x_0, x_1$ ("France") tambien naranja. La distancia es decenas de tokens: el gradiente no llega, el RNN no aprende la dependencia.

El contraste teal vs naranja es deliberado: el lector entiende visualmente cuando el RNN puede vs cuando no puede.

---

## 7. Gating como solucion (LSTM/GRU teaser) *(slide 54)*

La slide 54 ("Gating Mechanisms in Neurons") es **la unica del lecture donde se mencionan LSTM/GRU explicitamente**. La idea se presenta de forma compacta:

- "Idea: use **gates** to selectively **add** or **remove** information within **each recurrent unit**".
- Componentes legenda: **σ** (sigmoid neural net layer, caja amarilla) y **×** (pointwise multiplication, circulo rojo).
- Una caja verde grande etiquetada **"gated cell -- LSTM, GRU, etc."**.
- Caption: "Gates optionally let information through the cell".
- Bottom: "Long Short Term Memory (LSTMs) networks rely on a gated cell to track information throughout many time steps."

Y eso es **todo**. No hay forget/input/output gates desagregadas, no hay ecuaciones del cell state, no hay derivacion de por que LSTM evita el vanishing. El lecture trata a las arquitecturas con compuertas como una nota al pie historica antes de hacer el pivot real hacia attention.

> Para la mecanica completa de LSTM/GRU (ecuaciones de las compuertas, propiedad $\partial c_t / \partial c_{t-1} = f_t$, comparacion entre LSTM y GRU), ver el documento de [profundizacion](./profundizacion.md) o la [Clase 11 del curso UC](/clases/clase-11/teoria).

---

## 8. Aplicaciones de RNN: musica y sentiment *(slides 55-58)*

El bloque comienza con un section header *(slide 55)* y luego presenta dos aplicaciones canonicas.

### 8.1 Music generation *(slide 56)*

Un RNN con 4 cells: inputs (notas en azul) `E, F#, G, C` -> outputs (notas en purpura) `F#, G, C, A`. El setup:

- **Input**: sheet music.
- **Output**: next character in sheet music.

Una caja amarilla muestra "Listening to 3rd movement" con un boton play -- es el famoso ejemplo del MIT donde un RNN entrenado en partituras de Schubert genera el "tercer movimiento" de la sinfonia inacabada. El badge `6.S191 Lab!` apunta al laboratorio donde el alumno replica esto. Footer: `Huawei.` (atribucion a la fuente del demo).

### 8.2 Sentiment classification *(slides 57-58)*

Un RNN many-to-one: inputs `"I love this class!"` -> output `<positive>`. El setup:

- **Input**: sequence of words.
- **Output**: probability of having positive sentiment.

Aparece una linea de codigo TensorFlow con la perdida tipica de clasificacion:

```python
loss = tf.nn.softmax_cross_entropy_with_logits(y, predicted)
```

Footer: `Socher+, EMNLP 2013.` (paper original de sentiment con RNN).

La slide 58 expande con un panel "Tweet sentiment classification" mostrando dos tweets reales:

- 😃 "@MIT Introduction to #DeepLearning is definitely one of the best courses..."
- 😢 "I wouldn't mind a bit of snow right now. We haven't had any..."

Footer cambia a `H. Suresh, 6.S191 2018.` -- otra slide reciclada de la version 2018.

---

## 9. Limitaciones de RNN y pivot a self-attention *(slides 59-64)*

Esta es **la transicion pedagogica clave** del lecture.

### 9.1 Las 3 limitaciones de los RNN *(slide 59)*

A la izquierda permanece el diagrama del sentiment RNN; a la derecha aparece un panel titulado "Limitations of RNNs" con tres iconos:

- 🪣 **Encoding bottleneck**: toda la secuencia comprimida en un unico hidden state final.
- ⏰ **Slow, no parallelization**: cada paso depende del anterior, por construccion.
- 🧠 **Not long memory**: aun con LSTMs, mas alla de cierto horizonte la memoria se diluye.

Esta slide es la **bisagra explicita** hacia Transformers.

### 9.2 Goal of sequence modeling *(slide 60)*

Un diagrama de 3 capas con 6 timesteps ($x_0, x_1, x_2, ..., x_{t-2}, x_{t-1}, x_t$):

- Capa **input** (azul) -> capa **feature vector** (barras amarillas conectadas con flechas) -> capa **output** $\hat{y}_*$ (purpura).
- Tres labels a la izquierda: "Sequence of inputs", "Sequence of features", "Sequence of outputs".
- Texto superior: "RNNs: recurrence to model sequence dependencies".

Es el **setup conceptual** sobre el cual luego se itera.

### 9.3 Limitations -> Desired Capabilities *(slides 61-62)*

La slide 61 agrega a la izquierda el panel "Limitations of RNNs" con los 3 iconos del slide 59. La slide 62 lo reemplaza por **"Desired Capabilities"** con tres iconos nuevos:

- 🌊 **Continuous stream**.
- ⤴⤵ **Parallelization**.
- 🧠 **Long memory**.

Y agrega arriba la pregunta provocadora: "**Can we eliminate the need for recurrence entirely?**"

### 9.4 Idea 1: feed everything into dense network *(slides 63-64)*

El diagrama cambia: ahora el input es **una sola barra horizontal** (no celdas separadas), el feature vector tambien es una sola barra sin flechas entre timesteps, y el output igual. El icono "Long memory" se vuelve un solo bloque $x_0$: **todos los inputs colapsados en uno solo** *(slide 63)*.

La slide 64 evalua esta idea con un panel "Idea 1: Feed everything into dense network":

- ✅ No recurrence.
- ❌ Not scalable.
- ❌ No order.
- ❌ No long memory.

Y el cierre, con un emoji de cerebro: "**Idea: Identify and attend to what's important**". Pivot explicito a self-attention.

---

## 10. Self-attention conceptual: la analogia YouTube/search *(slides 65-69)*

### 10.1 Section header y Iron Man *(slides 65-66)*

El section header *(slide 65)* es directamente el titulo del paper de Vaswani: "Attention Is All You Need".

La slide 66 ("Intuition Behind Self-Attention") muestra una imagen de Iron Man volando con flechas horizontales blancas atravesando la imagen, representando atencion sobre regiones. Dos pasos:

1. **Identify** which parts to attend to (caja resaltada con flecha azul a "Similar to a search problem!").
2. Extract the features with high attention.

La metafora `attention = search` es lo que articula la analogia que viene.

### 10.2 La analogia search *(slide 67)*

Un emoji 🤔 con thought bubble "How can I learn more about neural networks?" y un collage de muchas miniaturas de videos. La pregunta es como encontrar lo relevante en un mar de contenido.

### 10.3 YouTube como ilustracion de Q/K/V *(slides 68-69)*

Mockup de YouTube en dark mode con search box "deep learning" (resaltado azul como **Query (Q)**). Tres resultados:

- (1) "GIANT SEA TURTLES • AMAZING CORAL REEF FISH..." atenuado como **Key (K₁)**.
- (2) **"MIT 6.S191 (2020): Introduction to Deep Learning"** por Alexander Amini resaltado en naranja como **Key (K₂)**.
- (3) "The Kobe Bryant Fadeaway Shot" atenuado como **Key (K₃)**.

Llaves laterales agrupan los tres keys con texto "How similar is the key to the query?". Bottom: "**1. Compute attention mask:** how similar is each key to the desired query?" *(slide 68)*.

La slide 69 agrega un **box purpura** alrededor del resultado MIT etiquetado **Value (V)** -- el contenido real extraido. Bottom cambia a: "**2. Extract values based on attention:** Return the values [with] highest attention".

La triada Q/K/V ya esta presentada en lenguaje cotidiano antes de aparecer en formulas.

---

## 11. Self-attention tecnico: los 4 pasos hasta el head completo *(slides 70-78)*

Esta es la seccion mas densa del lecture: **9 slides para construir el self-attention head completo**, paso a paso. El titulo unificado es "Learning Self-Attention with Neural Networks". La lista de 4 pasos aparece a la izquierda en cada slide y se va resaltando uno a la vez:

1. Encode position information.
2. Extract query, key, value for search.
3. Compute attention weighting.
4. Extract features with high attention.

### 11.1 Step 1 -- Encode position information *(slides 70-71)*

La slide 70 lista los 4 pasos con solo el primero highlighted, y al lado derecho una barra azul $x$ con las palabras "He / tossed / the / tennis / ball / to / serve" debajo. Bottom: "Data is fed in all at once! Need to encode position information to understand order."

La slide 71 visualiza el position encoding: la frase pasa por un embedding (azul vertical), se le suma elementwise (`⊕`) la position information $p_0, p_1, ..., p_6$, y resulta una **"Position-aware encoding"** (verde). Sin esto, el modelo no tendria ninguna nocion de orden, porque el computo siguiente es totalmente paralelo.

### 11.2 Step 2 -- Extract Q, K, V *(slide 72)*

Step 2 highlighted. A la derecha aparecen tres multiplicaciones matriciales: la **positional embedding** (verde) se multiplica por una **linear layer** distinta para producir cada uno de los tres outputs:

- Linear azul -> **Q (Query)**.
- Linear naranja -> **K (Key)**.
- Linear purpura -> **V (Value)**.

Footer: `Vaswani+, NeurIPS 2017.`

La idea: las tres matrices `W_Q`, `W_K`, `W_V` son parametros aprendibles que **proyectan el mismo input** a tres representaciones distintas, una por rol.

### 11.3 Step 3 -- Compute attention weighting *(slides 73-75)*

Step 3 highlighted con titulo "Attention score: compute pairwise similarity between each query and key".

- **Slide 73**: vectores Q (azul) y K (naranja) con flechas desde el origen, dot product, formula
  $$\frac{Q \cdot K^T}{\text{scaling}}$$
  etiquetada como "Similarity metric". Caption: "Also known as the 'cosine similarity'". El `scaling` (en el paper original $\sqrt{d_k}$) se introduce sin entrar en por que; lo que importa pedagogicamente es que **dot product = similitud**.

- **Slide 74**: misma idea pero con **matrices completas** en lugar de vectores -- $Q \cdot K^T$ ahora es una multiplicacion matricial. Es el salto a "todo a la vez" caracteristico de la atencion.

- **Slide 75**: la estrella visual del bloque. Se muestra una **matriz de atencion 7x7** (gradientes rojos, diagonal mas oscura) con filas y columnas labeled "He / tossed / the / tennis / ball / to / serve". La formula final es
  $$\text{softmax}\left(\frac{Q \cdot K^T}{\text{scaling}}\right)$$
  etiquetada "Attention weighting". El softmax convierte similitudes en probabilidades por fila, que es lo que permite interpretar la matriz como "cuanto atiende cada token a cada otro token".

### 11.4 Step 4 -- Extract features with high attention *(slide 76)*

Los cuatro steps quedan en negro (todos completos). A la derecha: matriz **Attention weighting** (roja) × matriz **Value** (purpura, V) = matriz **Output** (gris). La formula completa, con cada componente subrayado en su color:

$$A(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\text{scaling}}\right) \cdot V$$

Es la **definicion completa del scaled dot-product attention** del paper de Vaswani.

### 11.5 El head completo *(slides 77-78)*

La slide 77 reemplaza la visualizacion por el **diagrama completo del self-attention head** (estilo Vaswani):

- Positional Encoding (verde)
- -> 3 Linear layers (Query azul, Key naranja, Value purpura)
- -> MatMul
- -> Scale
- -> Softmax
- -> MatMul final
- -> output.

Una caja roja inferior dice: "These operations form a self-attention head that can plug into a larger network. Each head attends to a different part of input." Esta es **la slide donde se ensambla todo**.

La slide 78 muestra el mismo diagrama mas la frase punchline centrada: "**Attention is the foundational building block of the Transformer architecture.**"

---

## 12. Multi-head, aplicaciones modernas y cierre *(slides 79-83)*

### 12.1 Multi-head con Iron Man *(slide 79)*

Tres imagenes de Iron Man arriba: **Attention weighting** (silueta blanca) × **Value** (escena completa) = **Output** (Iron Man enfocado). Debajo, tres outputs distintos:

- Output of attention head 1: cara/casco de Iron Man.
- Output of attention head 2: edificio/fondo.
- Output of attention head 3: objeto en la distancia / contexto distinto.

La leccion intuitiva: **multiples heads aprenden a atender a partes distintas del input simultaneamente**. Cada head es un especialista; juntos cubren mas "aspectos" del contenido. (Esta visualizacion no esta en el paper original; es una adaptacion didactica de Ava.)

### 12.2 Self-attention applied: 3 dominios *(slide 80)*

Tres columnas con aplicaciones reales:

- **Language Processing**: imagen avocado-armchair "An armchair in the shape of an avocado". Modelos: **BERT, GPT**. Refs: *Devlin et al., NAACL 2019* y *Brown et al., NeurIPS 2020*. Badge `6.S191 Lab and Lectures!` apuntando a labs futuros.
- **Biological Sequences**: imagen de estructura proteinica 3D. Modelos: **AlphaFold** (*Jumper et al., Nature 2021*), **ESM** (*Lin et al., Science 2023*).
- **Computer Vision**: golden retriever en pasto con grid overlay. Modelo: **Vision Transformers (ViT)** (*Dosovitskiy et al., ICLR 2020*).

El mensaje: la misma maquinaria (self-attention) funciona en lenguaje, biologia y vision. Es la primera pista hacia la idea de **arquitectura general de proposito**.

### 12.3 Summary *(slide 81)*

Lista de **6 takeaways** sobre fondo de waveform multicolor:

1. RNNs are well suited for **sequence modeling** tasks.
2. Model sequences via a **recurrence relation**.
3. Training RNNs with **backpropagation through time**.
4. Models for **music generation**, classification, machine translation, and more.
5. Self-attention to model **sequences without recurrence**.
6. Self-attention is the basis for many **large language models** -- stay tuned!

Estos seis puntos son lo que el alumno deberia poder responder al terminar el lecture.

### 12.4 Logistica final *(slides 82-83)*

Las dos ultimas slides son administrativas y no pedagogicas:

- *(slide 82)* "Lab 1: Deep Learning in Python and Music Generation with RNNs". Pasos: abrir el lab en Google Colab, ejecutar bloques y rellenar `#TODO`s, pedir ayuda a TA si hace falta.
- *(slide 83)* "Kickoff Reception at One Kendall Square!" -- evento de inicio del bootcamp con comida y bebida, patrocinado por John Werner y Link Ventures.

Confirman que este es el primer lecture tecnico del programa intensivo: Lab 1 es el primer lab y la kickoff reception es el inicio del programa.

---

## Notas sobre el lecture

Algunas observaciones meta que vale la pena tener presentes al estudiar el material:

- **Continuidad y pivot.** El lecture 2026 mantiene el armazon clasico del 2020 (motivacion -> RNN -> BPTT -> vanishing -> aplicaciones) hasta la slide 60. A partir de ahi pivota explicitamente hacia attention y Transformers, terminando con BERT/GPT/AlphaFold/ViT como estado del arte. Esto refleja como el campo se desplazo entre 2020 y 2026.

- **LSTM/GRU minimizados.** En el lecture 2020 las arquitecturas con compuertas ocupan una porcion sustancial del lecture (forget/input/output gates desagregadas, ecuaciones, intuicion del cell state). En el 2026 ocupan **una sola slide** (la 54), sin internals. El enfasis pedagogico se desplazo del gating a la atencion -- consistente con la trayectoria del campo.

- **Slides recicladas.** Las slides 32-34, 38-39, 56-58 cargan footers de autores originales (`H. Suresh, 6.S191 2018.`, `Mozer Complex Systems 1989.`, `Huawei.`, `Socher+, EMNLP 2013.`). Son slides recicladas de versiones anteriores del curso o de fuentes externas, no creadas por Ava desde cero. Esto es normal en cursos academicos y ayuda a mantener trazabilidad de las contribuciones.

- **Bookend pedagogico.** La slide 40 es **identica a la 30** (los 4 criterios de diseno). No es un typo: es un patron deliberado donde se introduce la lista, se desarrollan los puntos uno por uno con ejemplos y luego se cierra el bloque repitiendo la lista. Vale tener este patron en mente al revisar las notas.

- **El embedding aparece una sola vez.** La slide 36 es la unica del lecture que cubre embeddings (vocabulary -> indexing -> one-hot vs learned). Si el alumno se queda con dudas, este tema se cubre con mas detalle en la profundizacion y en el material complementario sobre word2vec.

- **Iron Man como hilo visual.** La metafora de Iron Man se usa dos veces: primero para introducir la intuicion de attention *(slide 66)* y despues para visualizar multi-head attention *(slide 79)*. Es una eleccion didactica de Ava que no esta en el paper original.

---

> Material adaptado de **MIT 6.S191 (2026) Lecture 2: Deep Sequence Modeling**, Ava Amini, 5 de enero de 2026. [Video](https://www.youtube.com/watch?v=d02VkQ9MP44) - [Slides locales](/videos/mit-6s191-l2-2026/slides.pdf) - [Sitio del curso](http://introtodeeplearning.com). Notas en espanol con investigacion complementaria. Sin afiliacion oficial con MIT.
