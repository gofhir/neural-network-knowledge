## 1. Dependencias a Largo Plazo: El Problema del Flujo de Gradientes

### 1.1 Vanishing Gradients en RNNs Vanilla

Las RNNs recurrentes enfrentan un problema fundamental al entrenar con secuencias largas: el **problema del gradiente evanescente** (vanishing gradient). Cuando se propaga el error hacia atrás a través de muchos pasos temporales mediante backpropagation through time (BPTT), el gradiente tiende a decaer exponencialmente.

*(slide 61-66)*

Matemáticamente, el gradiente se multiplica por muchos números pequeños sucesivamente:

$$\frac{\partial L}{\partial h_t} = \prod_{i=t}^{T} \frac{\partial h_i}{\partial h_{i-1}}$$

Si cada factor es menor a 1, el producto tiende a cero. Esto impide que la red aprenda dependencias a largo plazo: los parámetros se actualizan muy poco para errores causados por inputs muy lejanos temporalmente.

**Consecuencias prácticas:**

- Los parámetros se sesguan para capturar únicamente dependencias **cortas**.
- En tareas como modelado de lenguaje, la red no puede recordar información relevante de pasos temporales distantes.
- Ejemplo: en la oración "The clouds are in the ___", el verbo plural "are" está lejos del sustantivo "clouds", y la red vanilla RNN lucharía por usar esa información.

### 1.2 Soluciones Parciales

Antes de las compuertas, se intentaron tres trucos:

**Truco #1: Funciones de Activación**

Usar ReLU en lugar de tanh o sigmoide ayuda porque la derivada de ReLU es 1 cuando $x > 0$, evitando decaimiento. Sin embargo, es una solución incompleta.

**Truco #2: Inicialización de Pesos**

Inicializar los pesos recurrentes $W_{hh}$ como matrices identidad ayuda a prevenir que decaigan a cero durante entrenamiento. El bias se inicializa a cero.

**Truco #3: Gradient Clipping**

Para el problema opuesto (gradientes explosivos), se normalizan los gradientes cuando su norma excede un umbral, evitando actualizaciones desestabilizadoras.

Estos trucos son útiles pero **insuficientes para dependencias realmente largas**. La solución de fondo requiere arquitecturas especializadas.

---

## 2. LSTM: Long Short-Term Memory

### 2.1 Idea Central: Compuertas que Controlan el Flujo de Información

La innovación clave de LSTM (propuesta por Hochreiter & Schmidhuber, 1997) es introducir **compuertas** que controlan selectivamente qué información entra, sale, u se olvida en el estado interno (cell state).

*(slide 67-72)*

El mecanismo opera en cuatro fases:

$$
\begin{aligned}
\text{1) Forget: } & f_t = \sigma(W_{hf} h_{t-1} + W_{xf} x_t) \\
\text{2) Store: } & i_t = \sigma(W_{hi} h_{t-1} + W_{xi} x_t) \\
& g_t = \tanh(W_{hg} h_{t-1} + W_{xg} x_t) \\
\text{3) Update: } & c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
\text{4) Output: } & o_t = \sigma(W_{ho} h_{t-1} + W_{xo} x_t) \\
& h_t = o_t \odot \tanh(c_t)
\end{aligned}
$$

**Componentes:**

- **$c_t$** (cell state): memoria interna que persiste entre pasos. Es el "depósito" principal de información.
- **$h_t$** (hidden state): salida pública del LSTM, versión filtrada de $c_t$.
- **$f_t$** (forget gate): controlada por sigmoide, decide cuánto del estado previo **olvida**.
- **$i_t$** (input gate): controlada por sigmoide, decide cuánto del nuevo candidato **almacena**.
- **$g_t$** (candidate): computa un candidato mediante tanh.
- **$o_t$** (output gate): filtra qué información de la celda se envía como salida.

### 2.2 Por Qué Resuelve Vanishing Gradients

El gradiente del cell state se propaga sin obstáculos:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

Esta operación es **multiplicación elemento a elemento**, no una multiplicación matricial completa. Esto permite que:

1. Si el forget gate $f_t \approx 1$, el gradiente fluye sin atenuación.
2. El flujo de gradientes es más directo que en RNNs vanilla, donde se multiplica por $W_{hh}$.

Así, LSTMs pueden aprender dependencias que abarcan **cientos o miles** de pasos temporales.

### 2.3 Resumen Visual: Los Cuatro Pasos

1. **Forget:** la compuerta $f_t$ multiplica el estado previo $c_{t-1}$ por un factor entre 0 y 1, permitiendo "olvidar" información irrelevante.

2. **Store:** la compuerta $i_t$ junto con el candidato $g_t$ decide qué nueva información añadir a la celda.

3. **Update:** se combina: $c_t = f_t \odot c_{t-1} + i_t \odot g_t$. El estado se actualiza de forma selectiva.

4. **Output:** la compuerta $o_t$ filtra la celda mediante una no-linealidad (tanh) antes de producir el estado visible $h_t$.

---

## 3. Aplicaciones Reales: Cuando las RNNs Brillan

### 3.1 Generación de Música

Una tarea clásica de RNN es entrenar un modelo a nivel de caracteres en notación musical (ej. notas como E, F#, G, C) para **generar secuencias de música**.

*(slide 73)*

**Arquitectura:**

- Input: secuencia de caracteres musicales (ej. [E, F#, G, C]).
- Output: distribución de probabilidad sobre el siguiente carácter.
- Training: cada paso predice el carácter siguiente; loss es cross-entropy.
- Inference: se muestrea del softmax para generar secuencias largas.

El modelo aprende patrones rítmicos, progresiones de acordes, y estructura melódica. Los resultados (demostrados en lab) producen música coherente, aunque sin estructura armónica profunda.

### 3.2 Clasificación de Sentimientos

Tarea: clasificar el sentimiento (positivo/negativo) de una secuencia de palabras.

*(slide 74-75)*

**Arquitectura many-to-one:**

- Input: palabras "I love this class!" (codificadas como embeddings).
- Hidden states: procesadas secuencialmente mediante LSTM.
- Output: solo del último estado $h_T$, que clasifica el sentimiento.
- Loss: softmax cross-entropy con etiquetas {positivo, negativo}.

Ejemplo real de Twitter: el modelo clasifica correctamente frases como "The @MIT Introduction to #DeepLearning is definitely one of the best courses of its kind currently available online" como positiva.

### 3.3 Traducción Automática: Arquitectura Encoder-Decoder

La tarea de traducción automática (ej. inglés → francés) requiere mapear una secuencia de entrada a una secuencia de salida de **longitud diferente**.

*(slide 76-78)*

**Patrón encoder-decoder:**

1. **Encoder RNN:** procesa la secuencia en inglés palabra por palabra, acumulando información en el hidden state.
2. **Context vector $C$:** el estado final del encoder, que codifica el significado de la oración entera.
3. **Decoder RNN:** inicializado con $C$, genera la traducción palabra por palabra en francés.

**Ejemplo:**

```
Entrada:  "the dog eats" (inglés)
Encoder:  [the] → [the, dog] → [the, dog, eats]
Context:  h_final ← contiene semántica de la oración
Decoder:  h_0 = context → "le" → "chien" → "mange"
```

Durante training, los outputs reales (teacher forcing) se realimentan al decoder. Durante inference, el decoder genera autorregressivamente muestreando de las distribuciones predichas.

---

## 4. El Cuello de Botella del Encoder

### 4.1 Problema Fundamental

En el modelo seq2seq estándar, **toda la información del input debe comprimirse en un único vector $C$** de dimensión fija. Esto crea un cuello de botella.

*(slide 79)*

Cuando el input es largo (oraciones de 20+ palabras), el context vector $C$ debe capturar:

- Estructura sintáctica completa.
- Significado de cada palabra.
- Relaciones gramaticales.
- Orden de palabras.

Pero el decoder debe generar traducción sin acceso directo a palabras específicas del input. Cuando genera "mange" (comes), no tiene conexión directa a "eats"; solo acceso a $C$.

**Resultado:** el rendimiento degrada significativamente con oraciones largas, porque $C$ pierde información importante.

---

## 5. Mecanismo de Atención: Acceso Adaptativo a la Memoria

### 5.1 Idea Core: Atención Selectiva

Para resolver el cuello de botella, se introduce un **mecanismo de atención** que permite al decoder "mirar" adaptativamente a diferentes partes del input mientras genera cada palabra.

*(slide 80-82)*

**Concepto clave:** en lugar de un único context vector $C$, tenemos un context vector adaptativo $C_t$ que **cambia en cada paso $t$ del decoder**, ponderando los hidden states del encoder según su relevancia.

$$C_t = \sum_{i=1}^{T} \alpha_{t,i} \, h_i^{\text{encoder}}$$

donde $\alpha_{t,i}$ son pesos de atención (softmax normalizado) que indican cuánta atención poner en la posición $i$ del input cuando generamos la palabra $t$ del output.

### 5.2 Cálculo de los Pesos de Atención (Bahdanau et al., 2015)

Los coeficientes $\alpha_{t,i}$ se calculan usando una red pequeña aditiva:

$$\text{score}(s_{t-1}, h_i) = V^T \tanh(W_1 s_{t-1} + W_2 h_i)$$

$$\alpha_{t,i} = \frac{\exp(\text{score}(s_{t-1}, h_i))}{\sum_j \exp(\text{score}(s_{t-1}, h_j))}$$

**Interpretación:**

- El estado del decoder $s_{t-1}$ "pregunta" qué parte del input necesita.
- El score combina el estado actual con cada hidden state del encoder.
- El softmax convierte scores en probabilidades, permitiendo que la red **enfoque** en posiciones relevantes.

### 5.3 Ejemplo Concreto: Traducción

Para traducir "El auto rojo de Carlos está averiado" a inglés:

- Al generar "Carlos" → $\alpha$ alto en "Carlos" del input.
- Al generar "red" → $\alpha$ alto en "rojo" del input.
- Al generar "broken" → $\alpha$ alto en "averiado" del input.

Las conexiones visuales (attention weights) permiten **alineación palabra-a-palabra** entre lenguajes.

---

## 6. Arquitectura de Atención Completa

### 6.1 Encoder Bidireccional

*(slide 83)*

Para capturar contexto tanto pasado como futuro, el encoder es típicamente bidireccional (BiLSTM):

- **Forward LSTM:** procesa izquierda a derecha, produce $\overrightarrow{h_i}$.
- **Backward LSTM:** procesa derecha a izquierda, produce $\overleftarrow{h_i}$.
- **Annotation:** $h_i = [\overrightarrow{h_i}; \overleftarrow{h_i}]$ (concatenación).

Esto permite que cada posición $i$ del encoder tenga información de toda la secuencia.

### 6.2 Decoder con Atención

El decoder genera cada palabra usando el context vector $C_t$:

$$y_t = \text{softmax}(W_{out} [s_t; C_t])$$

$$s_t = \text{LSTM}(s_{t-1}, y_{t-1}, C_t)$$

El estado del decoder $s_t$ se condiciona en:

- El estado previo $s_{t-1}$.
- La palabra previa $y_{t-1}$ (embedded).
- El context vector $C_t$ computado dinámicamente.

---

## 7. Aplicaciones de Atención: Más Allá de Traducción

### 7.1 Image Captioning con Atención Espacial

*(slide 84-85)*

Tarea: generar descripciones textuales de imágenes.

**Modificación de atención:** en lugar de atender a posiciones temporales, atendemos a **regiones espaciales de una imagen**.

- Se extrae un grid de features de la imagen usando una CNN (ej. VGG).
- El decoder genera cada palabra mientras "mira" a diferentes regiones.
- Las attention maps visualizan exactamente a dónde mira el modelo para cada palabra generada.

Ejemplo: "A woman is throwing a **frisbee** in a park."
- Al generar "frisbee", el mapa de atención se ilumina en la región del objeto frisbee.

### 7.2 Sumarización Abstractiva

*(slide 86-87)*

Tarea: generar resúmenes **nuevos** (no solo seleccionar oraciones) de documentos largos.

Usar Pointer-Generator Networks permite mezclar:

- **Generación:** crear palabras nuevas no presentes en el documento.
- **Pointing:** copiar palabras del documento original mediante attention.

El modelo aprende a **copiar hechos clave** (mediante pointer) mientras **parafrasea** (mediante generación). Esto mejora significativamente sobre sumarización puramente extractiva.

### 7.3 Bottom-Up Attention para Visión

*(slide 88)*

Mejora sobre grid-based attention: usar características de **objetos detectados** (Faster R-CNN) en lugar de grillas uniformes.

- Los objetos son una abstracción más natural que píxeles o grid cells.
- El modelo atiende a "elefante", "bebé elefante" como unidades discretas.
- Resulta en captions más semánticamente coherentes.

---

## 8. Resumen y Síntesis

### 8.1 Evolución de Architecturas para Secuencias

1. **RNN Vanilla:** simple pero sufre vanishing gradients.
2. **LSTM:** introduce compuertas que permiten flujo de gradientes ininterrumpido.
3. **Attention:** permite acceso selectivo a la memoria del encoder, resolviendo el cuello de botella.

### 8.2 Propiedades Fundamentales

- **Flexibilidad:** RNNs manejan secuencias de longitud variable mediante compartición profunda de parámetros.
- **Memorización:** LSTMs pueden aprender dependencias a largo plazo gracias a cell states.
- **Localización:** Attention permite que el decoder enfoque en información relevante, mejorando rendimiento en oraciones largas y tareas con múltiples modalidades.

### 8.3 Impacto Práctico

Las arquitecturas de seq2seq con atención dominaron tareas de NLP (traducción, Q&A, sumarización, image captioning) durante los años 2014-2017 hasta la llegada de Transformers. Son conceptos fundamentales que preceden y subyacen en modelos modernos.

---

## 9. Conexión con Transformers

Aunque no cubierto en detalle en estas páginas, la atención es el **precursor conceptual directo** del Transformer (Vaswani et al., 2017). Los Transformers eliminan la recurrencia y utilizan **atención pura** (multi-head self-attention) para procesar secuencias de forma paralela, mejorando significativamente en velocidad y rendimiento.

Los conceptos de alignment, weighted averaging, y learnable memory access introducidos por attention son centrales en la arquitectura Transformer y todo lo que ha seguido en deep learning moderno.
