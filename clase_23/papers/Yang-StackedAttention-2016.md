---
título: "Stacked Attention Networks for Image Question Answering"
autores: "Zichao Yang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Smola"
afiliaciones: "Carnegie Mellon University; Microsoft Research, Redmond"
venue: "CVPR 2016"
año: 2016
arxiv: "1511.02274"
link: "https://arxiv.org/abs/1511.02274"
clase: 23
tema: "Visual Question Answering — atención multi-paso (multi-hop)"
---

# Stacked Attention Networks for Image Question Answering (SAN)

> **Cita.** Zichao Yang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Smola. *Stacked Attention Networks for Image Question Answering*. CVPR 2016. arXiv:1511.02274 (v2, 26 ene 2016). Enlace: https://arxiv.org/abs/1511.02274

---

## 1. Resumen ejecutivo

Este paper introdujo en VQA (Visual Question Answering) una idea que hoy damos por sentada: **responder una pregunta visual no es un acto único de "mirar" la imagen, sino un proceso de razonamiento en varios pasos**, donde la atención se refina progresivamente hasta concentrarse en la región que contiene la respuesta. Los autores proponen las **Stacked Attention Networks (SAN)**, una arquitectura que consulta (*query*) la imagen **múltiples veces** usando una representación semántica de la pregunta como consulta, apilando dos o más capas de atención. La primera capa localiza de forma difusa todos los conceptos referidos por la pregunta; las siguientes filtran el ruido y apuntan con precisión a la región indicativa de la respuesta.

El aporte es triple, según los propios autores: (1) proponen la red de atención apilada para VQA; (2) realizan evaluaciones comprehensivas en **cuatro benchmarks** (DAQUAR-ALL, DAQUAR-REDUCED, COCO-QA y VQA), superando el estado del arte previo por márgenes sustanciales; y (3) presentan un análisis detallado **visualizando los mapas de atención** de cada capa, demostrando empíricamente el carácter multi-paso del razonamiento.

Para Roberto: este es el eslabón conceptual que conecta la atención de *Show, Attend and Tell* (Xu et al., 2015, en captioning) y la atención de traducción automática de Bahdanau (2014) con la **top-down attention** que verás en la clase 23 dentro de Pythia y *Bottom-Up and Top-Down Attention* (Anderson et al., 2018). SAN atiende sobre una **grilla espacial uniforme** (14×14 celdas); Pythia y BUTD atienden sobre **regiones de objetos detectados**. Esa es exactamente la línea evolutiva que conviene tener clara.

---

## 2. Contexto — VQA temprano, post-Antol 2015

VQA emerge alrededor de 2014–2015 como una de las tareas multimodales más activas, en la intersección de visión por computador y NLP. La tarea es directa de enunciar: dada una imagen y una pregunta en lenguaje natural sobre ella, producir la respuesta correcta (también en lenguaje natural). Su atractivo es que requiere razonar conjuntamente sobre contenido visual y texto, y sirve como prueba de comprensión integrada.

El panorama de modelos previos a SAN, según la sección de *Related Work* del paper, se puede resumir así:

- **Fusión global imagen + pregunta.** El enfoque más común extraía **un único vector de imagen global** con una CNN (típicamente la activación de la última capa *fully-connected* / inner product de una red como GoogLeNet o VGGNet) y codificaba la pregunta como un vector con una LSTM. Luego combinaba ambos vectores —por concatenación, suma o producto elemento a elemento— y predecía la respuesta. Modelos representativos:
  - **Malinowski et al. (Ask-Your-Neurons, ref. [19] del paper):** framework encoder-decoder; una LSTM codifica imagen y pregunta, otra LSTM decodifica la respuesta, alimentando la *feature* de imagen a cada celda LSTM. Probaron LSTM unidireccionales y bidireccionales.
  - **Ren et al. (VSE, ref. [21]):** varios modelos neuronales, incluyendo encoder-decoder; reportan que la **concatenación de features de imagen con bag-of-words** de la pregunta funciona mejor en sus experimentos.
  - **Antol et al. (VQA, ref. [1]):** codifican preguntas con LSTM y combinan con vectores de imagen por **multiplicación elemento a elemento**.
  - **Ma, Lu, Li (IMG-CNN, ref. [17]):** usan una **CNN para modelar la pregunta** y operaciones de convolución para combinar vectores de pregunta e imagen.

- **El problema de la representación global.** El defecto compartido es estructural: al colapsar la imagen en **un solo vector**, se pierde la información espacial. Los autores lo expresan con nitidez en la introducción: estos modelos "*often fail to give precise answers when such answers are related to a set of fine-grained regions in an image*". Si la respuesta depende de un objeto pequeño y específico (un pájaro sobre el manillar de una bicicleta), un vector global mezcla ese objeto con todo el resto de la escena (bicicleta, ventana, calle, canastos) e introduce **ruido** que arrastra la predicción a respuestas subóptimas.

El ejemplo canónico del paper (Figura 1) ilustra la motivación: la pregunta *"what are sitting in the basket on a bicycle?"* sobre una imagen que contiene `bicycles`, `window`, `street`, `baskets` y `dogs`. Para responder `dogs`, el modelo debe (i) localizar los objetos y conceptos referidos en la pregunta —`basket`, `bicycle`, `sitting in`—, (ii) descartar lo irrelevante, y (iii) apuntar finalmente a la región más indicativa de la respuesta, los perros dentro del canasto. Hacer esto con un vector global es, en la práctica, imposible.

---

## 3. La idea central — atención espacial guiada por la pregunta, y apilarla

La contribución conceptual de SAN tiene dos componentes que conviene separar:

**(a) Atención espacial guiada por la pregunta.** En lugar de un vector de imagen global, SAN extrae un **mapa de features espacial** (cada celda representa una región de la imagen). La pregunta, codificada como vector semántico, actúa como **consulta** (*query*) para calcular una **distribución de probabilidad de atención** sobre las regiones: cuáles miran más, cuáles menos. Esto es exactamente el mecanismo de atención de Bahdanau (2014) trasladado de "palabras de origen" a "regiones de imagen", la misma idea que Xu et al. (2015) aplicaron a captioning. La novedad de SAN es traerlo a VQA.

**(b) Apilar la atención (lo *stacked*).** Aquí está el aporte distintivo. Los autores argumentan que **una sola capa de atención no basta para preguntas complejas**. Para preguntas que involucran relaciones sutiles entre múltiples objetos (*"what are sitting in the basket on a bicycle"* involucra basket, bicycle y la relación espacial *sitting in*), una atención de un solo paso reparte la masa de probabilidad de forma difusa entre varios candidatos sin lograr decidir. La solución: **iterar el proceso de atención en varias capas apiladas**, donde cada capa toma como consulta el resultado refinado de la anterior y extrae información de atención visual progresivamente más fina.

La metáfora operativa, tomada de la visualización del propio paper (Figura 1b):

1. **Primera capa de atención:** la atención se **dispersa sobre todos los conceptos referidos** —`bicycle`, `basket`, objetos dentro del canasto. Es una localización amplia, de candidatos.
2. **Segunda capa de atención:** la atención se **estrecha** y se concentra en la región que efectivamente contiene la respuesta, los perros (`dogs`).

Este "*locate roughly, then sharpen*" es la firma del razonamiento multi-hop que SAN introdujo en VQA.

---

## 4. Arquitectura

SAN se compone de **tres módulos**: (1) el modelo de imagen, (2) el modelo de pregunta y (3) el modelo de atención apilada. Los reviso uno por uno con las ecuaciones del paper.

### 4.1. Modelo de imagen — CNN con mapa espacial (VGGNet)

A diferencia de los modelos previos que usaban la activación de la última capa *inner product* (un vector global), SAN toma las features de la **última capa de pooling** de VGGNet, que **conserva información espacial**:

$$f_I = \mathrm{CNN}_{\text{vgg}}(I) \tag{1}$$

El procedimiento concreto:

- Se reescala la imagen a **448×448 píxeles**.
- Se toma la salida de la última *pooling layer*, con dimensión **512 × 14 × 14**.
- Por tanto hay **14 × 14 = 196 regiones**, cada una representada por un vector de **512** dimensiones. Cada región corresponde a un parche de **32 × 32 píxeles** de la imagen original (448 / 14 = 32).

Denotando $f_i$, con $i \in [0, 195]$, el vector de features de cada región, se aplica un perceptrón de una capa para transformar cada vector de región a la **misma dimensión que el vector de pregunta**:

$$v_I = \tanh(W_I f_I + b_I) \tag{2}$$

donde $v_I$ es una matriz cuya $i$-ésima columna $v_i$ es la feature visual de la región $i$. Esta proyección es lo que permite combinar regiones y pregunta en el mismo espacio.

### 4.2. Modelo de pregunta — LSTM o CNN

Los autores exploran **dos** codificadores de pregunta (ambos competitivos, dando rendimiento similar en los experimentos).

**LSTM (Figura 3).** Dada la pregunta $q = [q_1, \dots, q_T]$ con $q_t$ el vector one-hot de la palabra en posición $t$, primero se incrustan las palabras con una matriz de embedding $W_e$ y luego se alimentan a la LSTM paso a paso:

$$x_t = W_e q_t, \quad t \in \{1, 2, \dots T\} \tag{8}$$
$$h_t = \mathrm{LSTM}(x_t), \quad t \in \{1, 2, \dots T\} \tag{9}$$

Las ecuaciones internas de la celda LSTM (compuerta de entrada $i$, olvido $f$, salida $o$ y celda de memoria $c$) son las estándar:

$$i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i) \tag{3}$$
$$f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f) \tag{4}$$
$$o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o) \tag{5}$$
$$c_t = f_t c_{t-1} + i_t \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c) \tag{6}$$
$$h_t = o_t \tanh(c_t) \tag{7}$$

El **último estado oculto** se toma como representación de la pregunta: $v_Q = h_T$.

**CNN (Figura 4).** Alternativamente, siguiendo a Kim (2014), se incrustan las palabras y se concatenan en una matriz $x_{1:T} = [x_1, x_2, \dots, x_T]$ (ec. 10). Se aplican **tres filtros convolucionales** de tamaño 1 (unigrama), 2 (bigrama) y 3 (trigrama):

$$h_{c,t} = \tanh(W_c x_{t:t+c-1} + b_c) \tag{11}$$
$$h_c = [h_{c,1}, h_{c,2}, \dots, h_{c, T-c+1}] \tag{12}$$

Luego **max-pooling sobre el tiempo** para cada tamaño de convolución:

$$\tilde{h}_c = \max_t [h_{c,1}, h_{c,2}, \dots, h_{c, T-c+1}] \tag{13}$$

y se concatenan los tres para formar el vector de pregunta:

$$h = [\tilde{h}_1, \tilde{h}_2, \tilde{h}_3], \quad v_Q = h \tag{14}$$

### 4.3. Capa de atención (una sola)

Con la matriz de features de imagen $v_I$ y el vector de pregunta $v_Q$, una **única** capa de atención se calcula así. Primero se pasan ambos por una red de una capa y un softmax para generar la distribución de atención sobre las regiones:

$$h_A = \tanh(W_{I,A} v_I \oplus (W_{Q,A} v_Q + b_A)) \tag{15}$$
$$p_I = \mathrm{softmax}(W_P h_A + b_P) \tag{16}$$

donde $v_I \in \mathbb{R}^{d \times m}$ ($d$ = dimensión de la representación de imagen, $m$ = número de regiones = 196), $v_Q \in \mathbb{R}^{d}$, $W_{I,A}, W_{Q,A} \in \mathbb{R}^{k \times d}$ y $W_P \in \mathbb{R}^{1 \times k}$. Entonces $p_I \in \mathbb{R}^m$ es un vector de $m$ dimensiones con la **probabilidad de atención de cada región** dada la pregunta. El símbolo $\oplus$ denota la suma de una matriz $W_{I,A} v_I \in \mathbb{R}^{k \times m}$ y un vector $(W_{Q,A} v_Q + b_A) \in \mathbb{R}^{k}$, realizada **sumando el vector a cada columna de la matriz** (broadcasting).

A partir de la distribución de atención se calcula la **suma ponderada** de los vectores de región:

$$\tilde{v}_I = \sum_i p_i v_i \tag{17}$$

y se combina con el vector de pregunta para formar una **consulta refinada**:

$$u = \tilde{v}_I + v_Q \tag{18}$$

$u$ es una consulta refinada porque codifica simultáneamente la información de la pregunta **y** la información visual relevante a la respuesta potencial. Frente a los modelos que simplemente combinan el vector de pregunta con el vector de imagen global, la atención construye un $u$ **más informativo**, porque pone más peso en las regiones visuales relevantes a la pregunta.

### 4.4. Apilar las capas — el corazón de SAN

Para preguntas complicadas, una sola capa de atención no localiza la región correcta. SAN **itera** el proceso $K$ veces. Para la $k$-ésima capa de atención:

$$h_A^k = \tanh(W_{I,A}^k v_I \oplus (W_{Q,A}^k u^{k-1} + b_A^k)) \tag{19}$$
$$p_I^k = \mathrm{softmax}(W_P^k h_A^k + b_P^k) \tag{20}$$

donde $u^0$ se inicializa como $v_Q$. La feature de imagen agregada se suma a la consulta de la capa anterior para formar la **nueva consulta**:

$$\tilde{v}_I^k = \sum_i p_i^k v_i \tag{21}$$
$$u^k = \tilde{v}_I^k + u^{k-1} \tag{22}$$

Es decir: en cada capa la consulta es el vector combinado pregunta+imagen de la capa anterior, $u^{k-1}$. Tras seleccionar la región, se actualiza la consulta como $u^k = \tilde{v}_I^k + u^{k-1}$. Esto se repite $K$ veces y la consulta final $u^K$ se usa para inferir la respuesta:

$$p_{\text{ans}} = \mathrm{softmax}(W_u u^K + b_u) \tag{23}$$

VQA se formula como **clasificación**: como casi todas las respuestas son de una sola palabra, la salida es una distribución softmax sobre el vocabulario de respuestas (por ejemplo, las 1000 respuestas más frecuentes en VQA).

Un detalle importante de diseño: la conexión residual implícita en $u^k = \tilde{v}_I^k + u^{k-1}$ (la consulta nueva **acumula** sobre la vieja, en lugar de reemplazarla) preserva la información de la pregunta a través de las capas y estabiliza el entrenamiento. Es conceptualmente similar a una *skip connection*.

```
Imagen 448×448 ──VGG──▶ mapa 512×14×14 (196 regiones) ──W_I──▶ v_I

Pregunta ──LSTM/CNN──▶ v_Q = u^0
                                   │
            ┌──────────────────────┘
            ▼
   Capa atención 1:  p_I^1 ◀── softmax(atención(v_I, u^0))
                     u^1 = Σ p_i^1 v_i  +  u^0
            │
            ▼
   Capa atención 2:  p_I^2 ◀── softmax(atención(v_I, u^1))
                     u^2 = Σ p_i^2 v_i  +  u^1
            │
            ▼
   p_ans = softmax(W_u u^2 + b_u)  ──▶  "dogs"
```

---

## 5. El razonamiento multi-hop — por qué apilar ayuda

El argumento del paper es que una respuesta suele relacionarse con **una región pequeña** de la imagen. Usar el vector global introduce ruido de regiones irrelevantes. Una sola capa de atención mitiga esto, pero para preguntas con varios objetos y relaciones no logra **decidir** cuál de los candidatos es la respuesta: reparte la masa de probabilidad.

**Ejemplo trabajado** (Figura 1b, *"what are sitting in the basket on a bicycle?"*):

- **Capa 1 — localización amplia.** La distribución de atención $p_I^1$ identifica de forma aproximada las zonas relevantes a `basket`, `bicycle` y `sitting in`. La atención está **dispersa** sobre múltiples objetos. La consulta acumulada $u^1$ ya incorpora algo de información visual, pero todavía no discrimina la respuesta.
- **Capa 2 — refinamiento.** Con $u^1$ como consulta (que ya "sabe" que mira un canasto sobre una bicicleta), la capa 2 produce $p_I^2$ **mucho más concentrada** en la región del canasto donde están los perros. La consulta final $u^2$ apunta a `dogs`.

La intuición de por qué la segunda capa puede ser más precisa: la primera consulta es puramente lingüística ($v_Q$), mientras que la segunda consulta ($u^1$) es **multimodal** —ya contiene evidencia visual de la primera pasada. Es como mirar una foto, formarte una hipótesis ("hay un canasto en una bicicleta"), y volver a mirar con esa hipótesis en mente para verificar qué hay dentro. Cada hop condiciona el siguiente sobre lo ya observado.

Esto es razonamiento **iterativo y compositivo**: la respuesta no se deriva en un solo paso de fusión, sino mediante refinamientos sucesivos de "dónde mirar".

---

## 6. Visualización de la atención

Una de las razones por las que este paper es tan citado es su **evidencia visual** del proceso multi-hop. En la sección 4.5, los autores visualizan las salidas de las capas de atención sobre ejemplos del conjunto de prueba de COCO-QA. El procedimiento: la distribución de probabilidad de atención tiene tamaño 14×14 y la imagen original 448×448, así que **sobremuestrean** la distribución de atención y le aplican un **filtro gaussiano** para llevarla al mismo tamaño que la imagen; la parte brillante de la imagen es la región detectada (con alta atención).

La Figura 5 presenta seis ejemplos (con más en el apéndice, Figura 7), cubriendo tipos *Object*, *Number*, *Color* y *Location*. Para cada ejemplo se muestran tres imágenes: original, salida de la primera capa de atención, salida de la segunda capa.

El hallazgo cualitativo, consistente en todos los ejemplos:

> En la **primera capa**, la atención está **dispersa sobre muchos objetos** de la imagen, correspondiendo en buena medida a los objetos y conceptos referidos en la pregunta. En la **segunda capa**, la atención está **mucho más enfocada** en las regiones que llevan a la respuesta correcta.

Ejemplo concreto del paper (Figura 5f): la pregunta *"what is the color of the horns?"*. En la salida de la primera capa, el modelo reconoce de forma aproximada a una mujer en la imagen. En la salida de la segunda capa, la atención se enfoca **en la cabeza de la mujer**, lo que conduce a la respuesta del color del cuerno: `red`.

El **valor interpretativo** es doble: por un lado, confirma empíricamente la hipótesis del diseño (que apilar atención produce localización progresivamente más fina); por otro, ofrece **explicabilidad** —se puede inspeccionar literalmente dónde mira el modelo en cada paso para producir su respuesta, algo muy valioso para diagnosticar fallos.

---

## 7. Experimentos

### 7.1. Datasets (cuatro benchmarks)

| Dataset | Origen | Entrenamiento | Prueba | Tipos de pregunta | Notas |
|---|---|---|---|---|---|
| **DAQUAR-ALL** | Malinowski-Fritz [18] | 6 795 preguntas | 5 673 preguntas | Object, Color, Number | 795 / 654 imágenes; escenas interiores; se excluyen respuestas multi-palabra, quedando 90% del set original |
| **DAQUAR-REDUCED** | versión reducida de DAQUAR | 3 876 muestras | 297 muestras | — | 37 categorías de objetos, 25 imágenes de prueba; respuestas de una sola palabra cubren 98% del set |
| **COCO-QA** | Ren et al. [21] | 78 736 muestras | 38 948 muestras | Object (70%), Number (7%), Color (17%), Location (6%) | generado parseando captions de MS COCO; 8 000 / 4 000 imágenes; respuestas de una sola palabra |
| **VQA** | Antol et al. [1] | 248 349 preguntas | 121 512 validación | Yes/No, Number, Other | imágenes de COCO; 3 preguntas por imagen, 10 respuestas humanas por pregunta; el más grande de los cuatro |

Para VQA, siguen a Antol et al. usando las **1000 respuestas más frecuentes** como salidas posibles, que cubren el **82.67%** de todas las respuestas. Dividen el set de validación en dos mitades, val1 y val2: entrenan con train+val1 y validan/prueban localmente con val2 (Tabla 6); además evalúan el mejor modelo SAN(2, CNN) en el **servidor de prueba oficial** (Tabla 5).

### 7.2. Métricas de evaluación

- **Accuracy de clasificación** (todas las respuestas de una palabra → problema de clasificación).
- **WUPS (Wu-Palmer Similarity)** en umbrales **0.9 y 0.0**: mide la similitud entre dos palabras según la subsecuencia común más larga en el árbol taxonómico; bajo un umbral, se pone a cero. Es la métrica de referencia en DAQUAR/COCO-QA.
- Para **VQA**, siguiendo a Antol et al., usan la métrica: $\min(\#\text{etiquetas humanas que coinciden}/3, 1)$, que da crédito completo cuando tres o más de las diez etiquetas humanas coinciden, y crédito parcial si hay menos coincidencias.

### 7.3. Configuración y entrenamiento

- **Modelo de imagen:** VGGNet con parámetros **fijos** (no se hace fine-tuning de la CNN); se usa la salida de la última pooling layer, 512×14×14.
- **Modelo de pregunta:** para DAQUAR y COCO-QA, dimensión de embedding y de la LSTM = **500**; para el CNN de pregunta, tamaños de filtro unigrama/bigrama/trigrama = **128, 256, 256** (vector de pregunta de **640** dimensiones). Para VQA, al ser más grande, **duplican** el tamaño del LSTM y del CNN.
- **Atención:** experimentan con **una y dos** capas de atención. Reportan explícitamente que usar **tres o más capas no mejora** el rendimiento.
- **Optimización:** SGD con momentum 0.9, batch de 100, mejor learning rate por grid search, con **gradient clipping** y **dropout**.

### 7.4. Nomenclatura

Los nombres de modelo codifican la configuración: **SAN(k, modelo)**, donde $k \in \{1, 2\}$ es el número de capas de atención y `modelo` $\in \{$LSTM, CNN$\}$ es el codificador de pregunta. Así, **SAN(2, CNN)** es el de dos capas de atención con codificador CNN —el mejor en la mayoría de los benchmarks.

---

## 8. Resultados numéricos (grounded en el PDF)

### 8.1. DAQUAR-ALL (Tabla 1, en %)

| Método | Accuracy | WUPS0.9 | WUPS0.0 |
|---|---|---|---|
| Multi-World [18] | 7.9 | 11.9 | 38.8 |
| Ask-Your-Neurons: Language [19] | 19.1 | 25.2 | 65.1 |
| Ask-Your-Neurons: Language+IMG [19] | 21.7 | 28.0 | 65.0 |
| IMG-CNN [17] | 23.4 | 29.6 | 63.0 |
| **SAN(1, LSTM)** | 28.9 | 34.7 | 68.5 |
| **SAN(1, CNN)** | 29.2 | 35.1 | 67.8 |
| **SAN(2, LSTM)** | 29.3 | 34.9 | 68.1 |
| **SAN(2, CNN)** | **29.3** | **35.1** | **68.6** |
| Human [18] | 50.2 | 50.8 | 67.3 |

Las SAN de dos capas superan al IMG-CNN [17] y al Ask-Your-Neurons [19] por **5.9%** y **7.6%** absoluto en accuracy, respectivamente.

### 8.2. DAQUAR-REDUCED (Tabla 2, en %)

| Método | Accuracy | WUPS0.9 | WUPS0.0 |
|---|---|---|---|
| Multi-World [18] | 12.7 | 18.2 | 51.5 |
| Ask-Your-Neurons: Language+IMG [19] | 34.7 | 40.8 | 79.5 |
| 2-VIS+BLSTM [21] | 35.8 | 46.8 | 82.2 |
| IMG-CNN [17] | 39.7 | 44.9 | 83.1 |
| SAN(1, LSTM) / SAN(1, CNN) | 45.2 | 49.6 | 84.0 / 83.7 |
| **SAN(2, LSTM)** | **46.2** | **51.2** | **85.1** |
| SAN(2, CNN) | 45.5 | 50.2 | 83.6 |
| Human [18] | 60.3 | 61.0 | 79.0 |

SAN(2, LSTM) supera a IMG-CNN [17], 2-VIS+BLSTM [21], Ask-Your-Neurons [19] y Multi-World [18] por **6.5%, 10.4%, 11.5% y 33.5%** absoluto en accuracy.

### 8.3. COCO-QA (Tabla 3, en %)

| Método | Accuracy | WUPS0.9 | WUPS0.0 |
|---|---|---|---|
| BOW [21] | 37.5 | 48.5 | 82.8 |
| LSTM [21] | 36.8 | 47.6 | 82.3 |
| IMG [21] | 43.0 | 58.6 | 85.9 |
| IMG+BOW [21] | 55.9 | 66.8 | 89.0 |
| 2-VIS+BLSTM [21] | 55.1 | 65.3 | 88.6 |
| IMG-CNN [17] | 55.0 | 65.4 | 88.6 |
| SAN(1, LSTM) | 59.6 | 69.6 | 90.1 |
| SAN(1, CNN) | 60.7 | 70.6 | 90.5 |
| SAN(2, LSTM) | 61.0 | 71.0 | 90.7 |
| **SAN(2, CNN)** | **61.6** | **71.6** | **90.9** |

En este dataset más grande, las SAN de dos capas superan a los mejores baselines (IMG-CNN [17] e IMG+BOW / 2-VIS+BLSTM [21]) por **5.1%** y **6.6%** en accuracy.

### 8.4. COCO-QA por clase (Tabla 4, accuracy en %)

| Método | Objects | Number | Color | Location |
|---|---|---|---|---|
| IMG+BOW [21] | 58.7 | 44.1 | 52.0 | 49.4 |
| 2-VIS+BLSTM [21] | 58.2 | 44.8 | 49.5 | 47.3 |
| SAN(1, LSTM) | 62.5 | 49.0 | 54.8 | 51.6 |
| SAN(1, CNN) | 63.6 | 48.7 | 56.7 | 52.7 |
| SAN(2, LSTM) | 63.6 | **49.8** | 57.9 | 52.8 |
| **SAN(2, CNN)** | **64.5** | 48.6 | **57.9** | **54.0** |

Frente a los mejores baselines, SAN(2, CNN) mejora **7.2%** en *Color*, **6.1%** en *Objects*, **5.7%** en *Location* y **4.2%** en *Number*.

### 8.5. VQA — servidor oficial (Tabla 5, en %)

| Método | test-dev All | Yes/No | Number | Other | test-std All |
|---|---|---|---|---|---|
| Question [1] | 48.1 | 75.7 | 36.7 | 27.1 | — |
| Image [1] | 28.1 | 64.0 | 0.4 | 3.8 | — |
| Q+I [1] | 52.6 | 75.6 | 33.7 | 37.4 | — |
| LSTM Q [1] | 48.8 | 78.2 | 35.7 | 26.6 | — |
| LSTM Q+I [1] | 53.7 | 78.9 | 35.2 | 36.4 | 54.1 |
| **SAN(2, CNN)** | **58.7** | 79.3 | 36.6 | **46.1** | **58.9** |

SAN(2, CNN) supera a LSTM Q+I (el mejor baseline de Antol et al.) por **4.8%** absoluto. La mayor ganancia está en el tipo *Other* (**9.7%**), seguido de *Number* (1.4%) y un leve 0.4% en *Yes/No*.

### 8.6. VQA — partición propia (Tabla 6, en %)

| Método | All | Yes/No | Number | Other |
|---|---|---|---|---|
| SAN(1, LSTM) | 56.6 | 78.1 | 41.6 | 44.8 |
| SAN(1, CNN) | 56.9 | 78.8 | 42.0 | 45.0 |
| SAN(2, LSTM) | 57.3 | 78.3 | **42.2** | 45.9 |
| **SAN(2, CNN)** | **57.6** | 78.6 | 41.8 | **46.4** |

### 8.7. Stacked (2) vs single (1) — la ganancia del apilamiento

El mensaje central del paper, expresado numéricamente: **en los cuatro datasets, las SAN de dos capas siempre superan a las de una capa.** Las cifras de la ganancia promedio:

- **COCO-QA:** en promedio, las dos capas superan a una capa por **2.2%** en *Color*, **1.3%** en *Location*, **1.0%** en *Objects* y **0.4%** en *Number*.
- **VQA:** las dos capas mejoran sobre una capa **1.4%** en *Other*, **0.2%** en *Number*, y plano en *Yes/No*.

Es una ganancia consistente aunque modesta, lo cual es coherente con la observación de que **tres o más capas no ayudan**: el beneficio del razonamiento iterativo se satura rápido en estos benchmarks. Nótese además que el tipo *Yes/No* casi no se beneficia de mejor modelado visual, porque (como confirma Antol et al.) la respuesta a un Yes/No depende muchísimo de la pregunta misma.

### 8.8. Análisis de errores (sección 4.6)

Los autores muestrean **100 imágenes** del set de prueba de COCO-QA donde SAN se equivoca, y clasifican los errores en cuatro categorías:

| Categoría de error | % | Descripción |
|---|---|---|
| Atención en región equivocada | **22%** | el modelo mira mal |
| Atención correcta, respuesta equivocada | **42%** | mira bien pero responde mal |
| Ambiguo / aceptable | **31%** | la respuesta difiere de la etiqueta pero es razonable (ej. etiqueta `pot`, predicción `vase`) |
| Etiqueta claramente errónea | **5%** | la etiqueta del dataset está mal (ej. predicción `trains` correcta, etiqueta `cars` errónea) |

Este desglose es revelador: **el 42% de los errores ocurren con atención correcta** —el cuello de botella ya no es "dónde mirar" sino el **razonamiento/clasificación** posterior. Y un **36% combinado** (31% + 5%) son casos donde la "predicción errónea" es en realidad aceptable o donde la etiqueta de oro está mal, lo que sugiere que la accuracy reportada **subestima** el rendimiento real.

---

## 9. Limitaciones

1. **Atención sobre grilla uniforme, no sobre objetos.** SAN atiende sobre una grilla regular de 14×14 celdas. Cada celda es un parche fijo de 32×32 píxeles, sin noción de objetos, sus límites o su semántica. Un objeto pequeño puede caer en una sola celda (poca resolución), y un objeto grande se reparte en muchas. Esta es precisamente la limitación que **Anderson et al. (2018, Bottom-Up and Top-Down Attention)** corrigieron, atendiendo sobre **regiones propuestas por un detector de objetos** (Faster R-CNN), donde cada "región de atención" es un objeto o parte saliente con significado.

2. **Número fijo de hops.** $K$ es un hiperparámetro fijo (2 en el mejor caso); el modelo no decide adaptativamente cuántos pasos de razonamiento necesita cada pregunta. Una pregunta trivial (`Yes/No`) recibe el mismo número de hops que una compositiva.

3. **El cuello de botella se desplazó al razonamiento.** Como muestra el análisis de errores, el 42% de los fallos ocurre con atención correcta. SAN mejora *dónde mirar*, pero la composición y el razonamiento relacional siguen siendo limitados —algo que arquitecturas posteriores (módulos neuronales, transformers multimodales) atacarían.

4. **CNN congelada.** La VGGNet no se ajusta (fine-tune) durante el entrenamiento, lo que limita la adaptación de las features visuales a la tarea de VQA.

5. **Sesgos de lenguaje en VQA.** El modesto beneficio en *Yes/No* (y el alto rendimiento del baseline "solo pregunta") evidencia que VQA tiene fuertes sesgos de lenguaje que la atención visual no resuelve por sí sola.

---

## 10. Impacto y legado

SAN es, junto con un puñado de trabajos contemporáneos, **el paper que popularizó la atención visual en VQA**. Su legado:

- **Estableció la atención visual guiada por pregunta como componente estándar** de los modelos de VQA. Después de SAN, prácticamente todos los modelos competitivos incorporan algún mecanismo de atención sobre la imagen.
- **Introdujo el razonamiento multi-paso (multi-hop) en VQA.** La idea de "localizar grueso, luego refinar" inspiró líneas de trabajo posteriores en atención iterativa, co-atención (preguntas e imagen se atienden mutuamente, Lu et al. 2016) y módulos de razonamiento.
- **Antecedente directo de la top-down attention de *Bottom-Up and Top-Down Attention* (Anderson et al., 2018)** y de **Pythia**. La top-down attention de BUTD es conceptualmente la capa de atención de SAN (consulta = pregunta; pesos softmax sobre regiones), pero aplicada a **regiones de objetos detectados** en lugar de a una grilla. La diferencia "bottom-up" es que las regiones provienen de un detector (Faster R-CNN sobre Visual Genome) en vez de una grilla fija.
- **La idea de razonamiento iterativo persiste.** La noción de refinar progresivamente una consulta multimodal a través de múltiples capas reaparece en los transformers multimodales modernos (LXMERT, ViLBERT, etc.), donde múltiples capas de cross-attention cumplen un rol análogo al apilamiento de SAN.
- **Demostró el valor de la visualización de atención como herramienta de interpretabilidad** en modelos multimodales, una práctica que se volvió estándar.

---

## 11. Conexión con la clase 23 (VQA y Pythia)

Este es el punto clave para conectar el paper con el material del curso. En la clase 23, la **top-down attention de Pythia** (slide 12: "obtener los objetos importantes para responder la pregunta", con un softmax de *attention weights* sobre $K$ regiones) es **heredera directa de la idea que SAN introdujo**.

La línea evolutiva, en una frase: **SAN atiende a una grilla; Pythia/BUTD atienden a regiones de objetos detectados.**

| Aspecto | SAN (Yang et al., 2016) | Pythia / BUTD (Anderson et al., 2018) |
|---|---|---|
| Unidad de atención | Grilla uniforme 14×14 = 196 celdas (parches de 32×32 px) | $K$ regiones de objetos detectados (Faster R-CNN sobre Visual Genome) |
| Origen de las regiones | Mapa de pooling de VGGNet (sin semántica) | Detector de objetos (semántica explícita: cada región es un objeto) |
| Cálculo de la atención | softmax sobre celdas, consulta = pregunta refinada | softmax de *attention weights* sobre regiones, consulta = pregunta |
| Multi-hop | Sí (apilar 2 capas) | Típicamente una capa top-down, pero combinada con la señal bottom-up del detector |
| Interpretabilidad | mapa de calor sobre la imagen | bounding boxes de objetos atendidos |

La continuidad conceptual es exacta: en ambos casos, **la pregunta actúa como consulta que pondera con un softmax un conjunto de regiones visuales, y la suma ponderada de features alimenta la predicción de la respuesta**. SAN comparte las ecuaciones (15)–(18) con la top-down attention de Pythia; lo que cambia es **qué son las regiones**: celdas de grilla vs. objetos. El paso de "grilla" a "objetos" es lo que dio el salto de calidad que Pythia capitaliza (ganó el VQA Challenge 2018).

Para situarlo en la cronología del módulo: Bahdanau (2014, atención en traducción) → Xu et al. (2015, *Show, Attend and Tell*, atención en captioning) → **Yang et al. (2016, SAN, atención multi-hop en VQA)** → Anderson et al. (2018, BUTD/top-down sobre objetos) → Pythia (2018, ingeniería + escala) → transformers multimodales (2019+).

---

## 12. Notas y enlaces

- **Relación con el mecanismo de atención general.** SAN es una aplicación del mecanismo de atención de Bahdanau (2014, ref. [2]) y de la atención visual de Xu et al. (2015, *Show, Attend and Tell*, ref. [30]). Los autores son explícitos: SAN es "una extensión significativa" de [30] en que se usan **múltiples** capas de atención para soportar razonamiento multi-paso, algo no explorado antes en VQA.
- **Relación con bottom-up attention.** SAN es **top-down puro**: la pregunta dirige la atención sobre una grilla agnóstica de objetos. BUTD añade una señal **bottom-up** (saliencia de objetos vía detector). Pythia operacionaliza esta combinación a escala.
- **VGGNet fija + última pooling layer.** El truco clave de implementación frente a trabajos previos: tomar la **última pooling layer** (512×14×14) en vez de la última capa fully-connected, para preservar la estructura espacial. Sin este mapa espacial, no hay sobre qué atender.
- **VQA como clasificación.** SAN trata VQA como clasificación sobre un vocabulario fijo de respuestas (top-1000 en VQA), no como generación. Esto funciona porque casi todas las respuestas son de una sola palabra, pero limita la expresividad —una restricción común en la era 2016.
- **Reproducibilidad.** CNN congelada, SGD con momentum 0.9, batch 100, gradient clipping, dropout, $K=2$. Tres o más capas no mejoran.

### Referencias relacionadas del paper

- [2] Bahdanau, Cho, Bengio. *Neural machine translation by jointly learning to align and translate*. arXiv:1409.0473, 2014. (Origen del mecanismo de atención.)
- [30] Xu et al. *Show, attend and tell: Neural image caption generation with visual attention*. arXiv:1502.03044, 2015. (Atención visual en captioning, predecesor directo.)
- [1] Antol et al. *VQA: Visual Question Answering*. arXiv:1505.00468, 2015. (Dataset VQA y baselines.)
- [21] Ren, Kiros, Zemel. *Exploring models and data for image question answering*. arXiv:1505.02074, 2015. (Dataset COCO-QA y baselines.)
- [18] Malinowski, Fritz. *A multi-world approach to question answering about real-world scenes based on uncertain input*. NeurIPS 2014. (Dataset DAQUAR.)
- [17] Ma, Lu, Li. *Learning to answer questions from image using convolutional neural network*. arXiv:1506.00333, 2015. (IMG-CNN, baseline principal.)
- [11] Kim. *Convolutional neural networks for sentence classification*. arXiv:1408.5882, 2014. (Base del codificador CNN de pregunta.)
- [23] Simonyan, Zisserman. *Very deep convolutional networks for large-scale image recognition*. arXiv:1409.1556, 2014. (VGGNet.)

> **Para leer después en clave clase 23:** Anderson et al. (2018), *Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering* — la materialización de "atención sobre objetos detectados" que Pythia lleva a producción.
