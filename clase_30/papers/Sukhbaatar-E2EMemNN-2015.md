# End-To-End Memory Networks — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *End-To-End Memory Networks*.
- **Autores:** Sainbayar Sukhbaatar (Dept. of Computer Science, Courant Institute, New York University), Arthur Szlam, Jason Weston, Rob Fergus (Facebook AI Research, New York).
- **Venue:** *Advances in Neural Information Processing Systems 28* (NeurIPS / NIPS 2015).
- **Año:** 2015. **Preprint:** arXiv:1503.08895v5 [cs.NE], 24 nov 2015, [arxiv.org/abs/1503.08895](https://arxiv.org/abs/1503.08895).
- **Código:** MemN2N disponible en [github.com/facebook/MemNN](https://github.com/facebook/MemNN).
- **Abreviatura del modelo:** **MemN2N** (Memory Network, end-to-end / "N2N").

Este paper introduce una red neuronal con un **modelo de atención recurrente sobre una memoria externa posiblemente grande**. La arquitectura es una forma de Memory Network (Weston, Chopra & Bordes, 2015, referencia [23]), pero —y esta es su tesis central— **se entrena extremo a extremo (end-to-end)** y por tanto requiere **mucha menos supervisión durante el entrenamiento**, lo que la hace aplicable a escenarios realistas. El modelo también se puede ver como una extensión de **RNNsearch** (Bahdanau et al., 2015) al caso en que se realizan **múltiples pasos de cómputo —los autores los llaman "hops" (saltos)— por cada símbolo de salida**.

El problema que resuelve es concreto. La Memory Network original era poderosa pero **no era fácil de entrenar por backpropagation y requería supervisión en cada capa de la red**: en el dataset de QA, había que indicarle explícitamente al modelo *cuáles* oraciones de soporte (supporting facts) eran relevantes para cada pregunta. Eso es poco práctico, porque en la mayoría de las tareas reales (modelado de lenguaje, QA realista) esa anotación no existe. MemN2N elimina ese requisito reemplazando las operaciones de **selección dura (hard max)** de la memoria por un **promedio ponderado continuo vía softmax (soft attention)**. Como la función de entrada a salida es ahora suave (smooth), se pueden computar gradientes y retropropagar el error a través de múltiples accesos a memoria, hasta la entrada, sin necesidad de etiquetas intermedias.

La flexibilidad del modelo permite aplicarlo a tareas tan diversas como **question answering sintético** (bAbI, 20 tareas) y **modelado de lenguaje** (Penn Treebank y Text8). En QA es competitivo con las Memory Networks fuertemente supervisadas pero con mucha menos supervisión; en modelado de lenguaje rinde de forma comparable —y ligeramente superior— a RNNs y LSTMs afinados de complejidad similar. En ambos casos, el hallazgo transversal es que **múltiples hops de cómputo mejoran el desempeño**.

Para el curso IA UC importa porque MemN2N es uno de los modelos centrales de la Clase 30 (modelos con memoria externa) y, sobre todo, porque es un **precursor directo de la atención de los Transformers**: la self-attention de "Attention Is All You Need" puede leerse como múltiples hops de soft attention apilados, exactamente el mecanismo que este paper formaliza dos años antes.

## 2. Contexto histórico: dos grandes desafíos y el resurgimiento de la memoria explícita

El paper se abre nombrando **dos grandes desafíos en investigación de IA**: (1) construir modelos que puedan dar **múltiples pasos de cómputo** al servicio de responder una pregunta o completar una tarea, y (2) modelos que puedan describir **dependencias de largo plazo** en datos secuenciales. Hacia 2014–2015 había un resurgimiento de modelos que usaban **almacenamiento explícito y una noción de atención** —Memory Networks (Weston et al.), Neural Turing Machines (Graves et al., 2014), RNNsearch (Bahdanau et al.)— donde el almacenamiento se dota de una representación continua y las lecturas/escrituras se modelan con redes neuronales. Manipular ese almacenamiento ofrece una vía a *ambos* desafíos a la vez.

El contraste con las RNN/LSTM clásicas es la clave del argumento. En esos modelos, **la memoria es el estado de la red**: es latente e inherentemente inestable a lo largo de escalas temporales largas. Las LSTM lo mitigan con celdas de memoria locales que "fijan" estado pasado, pero —citando a Mikolov et al. (2014)— las ganancias sobre RNNs bien entrenadas son modestas. MemN2N difiere en que usa una **memoria global, con funciones de lectura y escritura compartidas**.

El antecedente inmediato que el paper busca destronar es la **Memory Network de Weston, Chopra & Bordes (2015)**. Su limitación práctica decisiva: el modelo de ese trabajo **no era fácil de entrenar vía backpropagation** y **requería supervisión en cada capa**. En el dataset de QA, el subconjunto de oraciones de soporte se le indicaba explícitamente al modelo durante el entrenamiento. MemN2N nace precisamente para quitar esa muleta: su continuidad significa que **se puede entrenar end-to-end desde pares entrada-salida**, y por tanto aplicarse a tareas donde tal supervisión no está disponible.

El otro pariente cercano es la **Neural Turing Machine** (Graves et al., 2014), que también usa memoria continua. La diferencia: la NTM usa acceso por **contenido** *y* por **dirección**, mientras que MemN2N solo permite explícitamente el primero (aunque las features temporales de la Sección 4.1 habilitan una especie de acceso por dirección). Como MemN2N siempre escribe cada memoria secuencialmente, es más simple: no necesita operaciones como el *sharpening*. Y se aplica a razonamiento textual, cualitativamente distinto de las operaciones abstractas de ordenamiento y recall que ataca la NTM. Respecto a **RNNsearch / Bahdanau et al. (2015)**, la "memoria" de MemN2N es análoga a su mecanismo de atención, pero con dos diferencias: Bahdanau atiende sobre una sola oración (los estados ocultos del encoder), mientras que MemN2N atiende sobre **muchas** memorias; y MemN2N hace **varios hops** antes de emitir una salida, lo que el paper mostrará empíricamente que es crucial.

## 3. Contribución central

La contribución es una arquitectura, **MemN2N**, que toma una versión *continua* de la Memory Network y la hace **entrenable extremo a extremo sin supervisión de los hops**. Cuatro ideas la componen:

1. **Lectura por soft attention.** En lugar del *hard max* (seleccionar la memoria de mayor score), se usa una distribución de probabilidad **softmax** sobre las memorias, computada por **producto interno** entre la consulta embebida y cada memoria. La respuesta es una **suma ponderada** de los vectores de salida de la memoria. Esto vuelve todo diferenciable.
2. **Embeddings de entrada/salida (matrices A, B, C, W).** Cada elemento de la memoria tiene una representación de *entrada* (matriz A) usada para casar con la consulta, y una de *salida* (matriz C) usada para construir la respuesta; la consulta se embebe con B; la predicción final con W.
3. **Múltiples hops apilados con weight tying.** Las capas de memoria se apilan (típicamente K = 3), cada una refina el estado interno *u*, y se comparten pesos entre capas con dos esquemas (*adjacent* y *layer-wise*) para regularizar y reducir parámetros.
4. **Aplicabilidad dual.** El mismo armazón sirve para **QA** (bAbI) y para **modelado de lenguaje** (Penn Treebank, Text8), demostrando que no es un truco específico de tarea.

La idea de diseño que une todo: como toda la cadena de entrada a salida es suave, **el error se retropropaga a través de los múltiples accesos a memoria hasta la entrada**, eliminando la necesidad de etiquetas de soporte intermedias. El modelo "deduce por sí mismo, en entrenamiento y en test, qué oraciones son relevantes y cuáles son distractores irrelevantes".

## 4. Método

### 4.1. Capa única (single layer): un hop de memoria

El modelo recibe un conjunto discreto de entradas $x_1, \dots, x_n$ a almacenar en memoria, una consulta $q$, y produce una respuesta $a$. Todos los $x_i$, $q$ y $a$ contienen símbolos de un diccionario de $V$ palabras.

**Representación de memoria de entrada.** Cada $x_i$ se convierte en un **vector de memoria** $m_i$ de dimensión $d$, embebiendo $x_i$ con una matriz de embedding $A$ (de tamaño $d \times V$). La consulta $q$ se embebe con otra matriz $B$ (misma dimensión) para obtener el **estado interno** $u$. El *match* entre $u$ y cada memoria $m_i$ se computa por **producto interno seguido de softmax**:

$$p_i = \mathrm{Softmax}(u^T m_i)$$

con $\mathrm{Softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$. Así $p$ es un **vector de probabilidad sobre las entradas** — el peso de atención que el modelo asigna a cada memoria.

**Representación de memoria de salida.** Cada $x_i$ tiene además un **vector de salida** $c_i$ (en el caso simple, vía otra matriz de embedding $C$). El **vector respuesta** $o$ es la **suma sobre los $c_i$ ponderada por el vector de probabilidad**:

$$o = \sum_i p_i\, c_i$$

Como la función de entrada a salida es suave, se pueden computar gradientes y retropropagar a través de ella —el mismo enfoque que toman Bahdanau et al. (2015) y Graves et al. (2014).

**Predicción final.** La suma del vector de salida $o$ y el embedding de entrada $u$ pasa por una matriz de pesos final $W$ (tamaño $V \times d$) y un softmax:

$$\hat{a} = \mathrm{Softmax}(W(o + u))$$

En entrenamiento, las tres matrices de embedding $A$, $B$, $C$ y la matriz $W$ se aprenden **conjuntamente** minimizando una **cross-entropy** estándar entre $\hat{a}$ y la etiqueta verdadera $a$, vía descenso de gradiente estocástico.

### 4.2. Múltiples capas (hops) y weight tying

Para $K$ hops, las capas de memoria se apilan. La entrada a las capas por encima de la primera es la **suma de la salida $o^k$ y la entrada $u^k$** de la capa $k$:

$$u^{k+1} = u^k + o^k$$

Cada capa tiene sus propias matrices $A^k, C^k$. En la cima de la red, la entrada a $W$ combina entrada y salida de la capa superior: $\hat{a} = \mathrm{Softmax}(W u^{K+1}) = \mathrm{Softmax}(W(o^K + u^K))$. El modelo "hace varios pasos de cómputo antes de producir una salida destinada al mundo exterior".

Hay **dos esquemas de weight tying** (atado de pesos):

- **Adjacent (adyacente):** el embedding de salida de una capa es el de entrada de la de arriba, $A^{k+1} = C^k$; además la matriz de predicción se ata al último embedding de salida ($W^T = C^K$) y el embedding de la pregunta al de entrada de la primera capa ($B = A^1$). Es el esquema usado por defecto en los experimentos de QA.
- **Layer-wise (estilo RNN):** entrada y salida son las mismas a través de todas las capas ($A^1 = \dots = A^K$, $C^1 = \dots = C^K$). Aquí se añade un mapeo lineal $H$ a la actualización: $u^{k+1} = H u^k + o^k$. Con este esquema, **MemN2N se puede ver como una RNN tradicional** donde se separan salidas internas (considerar una memoria) y externas (predecir una etiqueta). Es el esquema usado en modelado de lenguaje.

Una versión de tres capas se ilustra en la Figura 1(b) del paper. En esencia, MemN2N es como la Memory Network de [23] salvo que **las operaciones hard max dentro de cada capa se reemplazan por un ponderado continuo del softmax**.

### 4.3. Codificaciones de posición y temporal (detalles para QA)

Dos representaciones de oración se exploran:

- **Bag-of-words (BoW):** se embebe cada palabra y se suman los vectores, p. ej. $m_i = \sum_j A x_{ij}$. Defecto: **no captura el orden de las palabras**, que es importante en varias tareas.
- **Position Encoding (PE):** $m_i = \sum_j l_j \cdot A x_{ij}$, donde $\cdot$ es multiplicación elemento a elemento y $l_j$ es un vector columna con estructura $l_{kj} = (1 - j/J) - (k/d)(1 - 2j/J)$ ($J$ = número de palabras en la oración, $d$ = dimensión del embedding). Así **el orden de las palabras afecta a $m_i$**. La misma representación se usa para preguntas, memorias de entrada y de salida.

**Temporal Encoding.** Muchas tareas de QA requieren noción de **contexto temporal** (saber que Sam está en el dormitorio *después* de la cocina). Se modifica el vector de memoria: $m_i = \sum_j A x_{ij} + T_A(i)$, donde $T_A(i)$ es la $i$-ésima fila de una matriz especial $T_A$ que codifica información temporal; análogamente la salida con $T_C$. $T_A$ y $T_C$ se **aprenden** y están sujetas a las mismas restricciones de atado que $A$ y $C$. Las oraciones se indexan en **orden inverso** (reflejando su distancia relativa a la pregunta, de modo que $x_1$ es la última oración de la historia).

**Random Noise (RN).** Para regularizar $T_A$ e inducir invariancia temporal, en entrenamiento se añade aleatoriamente un **10% de memorias vacías ("dummy")** a las historias.

### 4.4. Detalles de entrenamiento

Cross-entropy + SGD. Tasa de aprendizaje $\eta = 0.01$ con anneals de $\eta/2$ cada 25 épocas hasta las 100 épocas; sin momentum ni weight decay; inicialización gaussiana ($\mu = 0$, $\sigma = 0.1$); batch de 32; **gradient clipping** a norma $\ell_2$ de 40. Una técnica notable es el **Linear Start (LS)**: comenzar el entrenamiento con los softmax de cada capa de memoria *removidos* (modelo enteramente lineal salvo el softmax final), y reinsertarlos cuando la pérdida de validación deja de bajar. LS ayuda a **evitar mínimos locales** (la tarea 16 baja de 53.6% a 1.6% de error con LS). Por la varianza alta según la inicialización, cada entrenamiento se repite 10 veces y se escoge el de menor error de entrenamiento.

## 5. Experimentos

### 5.1. QA sintético — bAbI (20 tareas)

Las tareas bAbI (Weston et al., 2015, versión 1.1) constan de un conjunto de afirmaciones seguidas de una pregunta cuya respuesta suele ser una sola palabra. Hay **20 tipos de tareas** que prueban distintas formas de razonamiento y deducción (un soporte, dos soportes, tres soportes, relaciones de argumentos, sí/no, conteo, listas/conjuntos, negación, coreferencia, deducción, inducción, razonamiento posicional, de tamaño, temporal, path finding, motivación del agente, etc.). El vocabulario es pequeño ($V = 177$). Solo *un subconjunto* de las afirmaciones contiene la información necesaria; el resto son **distractores**. La diferencia clave con Weston et al.: ese subconjunto de soporte **ya no se le entrega al modelo**. Se usan dos versiones: 1k y 10k problemas de entrenamiento por tarea.

Configuración por defecto: **K = 3 hops** con atado *adjacent*. Las líneas base de comparación:

- **MemNN** (fuertemente supervisado): el mejor enfoque de Weston et al. (2015), con hard max entrenado directamente con los supporting facts, modelado de n-gramas, capas no lineales y número adaptativo de hops.
- **MemNN-WSH**: versión débilmente supervisada heurística (sin etiquetas de soporte).
- **LSTM**: LSTM estándar, también débilmente supervisado.

**Resultados (Tabla 1, 1k):** el mejor MemN2N llega a **12.6%** de error medio (PE+LS+RN, joint) frente a **6.7%** del MemNN fuertemente supervisado y **51.3%** del LSTM. En 10k: **4.2%** (con no-linealidad) vs **3.2%** del MemNN supervisado vs **36.4%** del LSTM. Es decir, **MemN2N se acerca razonablemente al modelo supervisado pese a usar mucha menos supervisión, y supera cómodamente a todas las líneas base débilmente supervisadas**. Hallazgos finos:

- **PE > BoW**, sobre todo en tareas donde el orden importa (4, 5, 15, 18).
- **LS** ayuda a evitar mínimos locales (tarea 16: 53.6% → 1.6%).
- **RN** da un boost pequeño pero consistente, especialmente en 1k.
- El **joint training** sobre las 20 tareas ayuda.
- **Más hops mejoran**: 1 hop da 25.8% de error medio; 2 hops, 15.6%; 3 hops, 13.3% (1k, joint). La Figura 2 muestra cómo el modelo aprende a concentrar la atención sobre las oraciones de soporte correctas a lo largo de los hops, sin que se le indiquen.

### 5.2. Modelado de lenguaje — Penn Treebank y Text8

Aquí se opera a **nivel de palabra**: las $N$ palabras previas se embeben separadamente en memoria, cada celda guarda una sola palabra (no hace falta BoW ni mapeo lineal). Como no hay pregunta, $q$ se fija a un vector constante 0.1 (sin embebido). El softmax de salida predice la siguiente palabra del vocabulario $V$. Se usa **weight tying layer-wise (estilo RNN)** y se aplican **ReLU** a la mitad de las unidades de cada capa. La "secuencia" sobre la que la red es recurrente **no está en el texto, sino en los hops de memoria**.

Datasets: **Penn Treebank** (929k/73k/82k palabras train/val/test, vocabulario 10k) y **Text8** (100M caracteres de Wikipedia, vocabulario ~44k). Gradient clipping a norma $L = 50$ (crucial); annealing de Mikolov et al. (factor 1.5).

**Resultados (Tabla 2, perplejidad en test):** MemN2N logra **menor perplejidad en ambos datasets**: en Penn Treebank **111** (7 hops, memoria 200) frente a **129/115** de RNN/LSTM/SCRN; en Text8 **147** (7 hops) frente a **154** del LSTM. MemN2N tiene ~1.5× los parámetros de una RNN con el mismo número de unidades ocultas (la LSTM tiene ~4×). De nuevo, **aumentar el número de hops mejora la perplejidad**. La Figura 3 muestra que algunos hops se concentran en palabras recientes y otros tienen atención amplia sobre toda la memoria —consistente con la idea de que un buen modelo de lenguaje combina un n-grama suavizado con un cache—, y que esos dos tipos de hop tienden a alternarse. A diferencia de una RNN tradicional, el cache **no decae exponencialmente**: mantiene activación promedio pareja a lo largo de toda la memoria, lo que el paper sugiere como fuente de la mejora.

## 6. Limitaciones reconocidas

El propio paper es modesto en sus conclusiones:

- **No iguala al supervisado.** MemN2N todavía no logra igualar exactamente el desempeño de las Memory Networks entrenadas con supervisión fuerte, y ambos modelos **fallan en varias de las tareas de QA 1k** (p. ej. razonamiento posicional, path finding y, sin no-linealidad, inducción básica).
- **Escalabilidad de la atención suave.** Las búsquedas suaves (smooth lookups) **pueden no escalar bien** cuando se requiere una memoria muy grande, porque el softmax pondera *todas* las memorias. Como trabajo futuro, los autores proponen explorar nociones **multiescala de atención o hashing**.
- **Acceso solo por contenido.** A diferencia de la NTM, MemN2N solo permite acceso por contenido explícitamente; el acceso por dirección llega solo de forma limitada vía las features temporales.
- **Varianza por inicialización.** En algunas tareas hay gran varianza según la semilla, mitigada con 10 reentrenamientos y selección del mejor — señal de un paisaje de optimización difícil.

## 7. Impacto: el puente hacia la atención de los Transformers

El aporte conceptual de mayor alcance de este paper es haber aislado y vuelto entrenable, sin supervisión intermedia, el patrón **"computar pesos de atención por producto interno + softmax, leer por suma ponderada, repetir"**. Visto en retrospectiva, ese es exactamente el núcleo de la atención de los Transformers (Vaswani et al., 2017):

- El **producto interno $u^T m_i$ seguido de softmax** de la Ecuación (1) es el ancestro directo del *scaled dot-product attention* (query · key → softmax). El estado interno $u$ juega el rol de la *query*; los $m_i$, el de las *keys*; los $c_i$ (vectores de salida), el de los *values*; y la suma ponderada $o = \sum_i p_i c_i$ es exactamente la salida de una capa de atención.
- Los **múltiples hops apilados** de MemN2N son el antecedente de **apilar capas de self-attention**: la self-attention de un Transformer puede leerse como múltiples hops de soft attention donde la "memoria" son las representaciones de todos los tokens de la secuencia.
- La separación **embedding de entrada (A) / embedding de salida (C)** prefigura la distinción **key / value** (en contraste con usar la misma proyección para casar y para leer).

MemN2N demostró que la soft attention multi-hop es entrenable end-to-end y mejora con la profundidad de cómputo — dos lecciones que los Transformers llevarían al extremo. Es uno de los eslabones que conectan las Memory Networks y RNNsearch con la era de la atención que domina el NLP moderno.

## 8. Conexión con la Clase 30 (Modelos con memoria externa)

La Clase 30 del curso IA UC dedica varias slides a las **End-to-End Memory Networks (Sukhbaatar et al., 2015)** dentro del módulo de modelos con memoria externa, usando el ejemplo "**¿Quién dirigió El Origen?**" y una **base de conocimiento estructurada como tripletas** (sujeto–relación–objeto). El mapeo del paper a esa exposición:

- **La base de conocimiento de tripletas** = el conjunto $\{x_i\}$ que se escribe a memoria. Cada hecho ("El Origen — dirigida_por — Christopher Nolan") se embebe en un vector de memoria de entrada $m_i$ (matriz A) y uno de salida $c_i$ (matriz C). La pregunta "¿Quién dirigió El Origen?" se embebe en $u$ (matriz B).
- **La lectura de la memoria** = la Ecuación (1): el producto interno $u^T m_i$ + softmax asigna alta probabilidad $p_i$ a la tripleta relevante (la de "El Origen"), y la suma ponderada $o = \sum_i p_i c_i$ recupera el objeto correcto. Esto ilustra en vivo la **soft attention** que la clase contrasta con el lookup duro de una base de datos clásica.
- **Por qué importa "sin supervisión de los hops"**: la clase enfatiza que el modelo *aprende solo* a atender el hecho correcto entre muchos distractores, sin que se le diga cuál es — la diferencia exacta con la Memory Network original.
- **Múltiples hops** = preguntas que requieren encadenar hechos (razonamiento multi-paso), donde el primer hop localiza una entidad intermedia y el segundo el objeto final, tal como muestran las tablas de pesos de atención del paper.

Para profundizar:

- Fundamento transversal: [/fundamentos/memory-augmented-networks](/fundamentos/memory-augmented-networks)
- Clase: [/clases/clase-30](/clases/clase-30)
- Paper relacionado (origen de bAbI y de las tareas de QA): [/papers/babi-weston-2015](/papers/babi-weston-2015)
- Paper sucesor (generalización entrada/salida → clave/valor): [/papers/key-value-memnn-miller-2016](/papers/key-value-memnn-miller-2016)
