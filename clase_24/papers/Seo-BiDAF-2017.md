# Bidirectional Attention Flow for Machine Comprehension (BiDAF)

> Análisis técnico exhaustivo para el curso IA UC (Diplomado en Inteligencia Artificial, PUC Chile).
> Paper de referencia de la era pre-Transformer en *machine comprehension* y *span extraction*.

## 1. Metadata

| Campo | Detalle |
|-------|---------|
| **Título** | Bidirectional Attention Flow for Machine Comprehension |
| **Autores** | Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi |
| **Afiliaciones** | University of Washington (Seo, Farhadi, Hajishirzi); Allen Institute for Artificial Intelligence — AI2 (Kembhavi, Farhadi) |
| **Venue** | ICLR 2017 (conference paper) |
| **arXiv** | 1611.01603 (v6, 21 Jun 2018; primera versión nov. 2016) |
| **Código / demo** | `allenai.github.io/bi-att-flow/` (incluye demo interactiva) |
| **Tareas evaluadas** | Question Answering extractivo (SQuAD) y cloze-test (CNN/DailyMail) |
| **Tamaño del modelo** | ~2.6 millones de parámetros |
| **Hardware reportado** | ~20 h en una sola Titan X (SQuAD); ~60 h en ocho Titan X (CNN/DailyMail) |

Nota sobre la nomenclatura: el primer autor, Minjoon Seo, realizó la mayor parte del trabajo durante una pasantía en AI2. La sigla del modelo aparece en el paper en mayúsculas espaciadas (B I DAF); en este documento usaremos **BiDAF**.

---

## 2. Contexto: SQuAD y las limitaciones de los *attention readers* previos

A finales de 2016, *machine comprehension* (MC) — responder una pregunta sobre un párrafo de contexto — vivía un momento bisagra. El cuello de botella histórico no era arquitectónico sino de datos: conjuntos tempranos como **MCTest** (Richardson et al., 2013) eran demasiado pequeños para entrenar modelos neuronales de extremo a extremo. La irrupción de datasets masivos de tipo *cloze* (**CNN/DailyMail** de Hermann et al., 2015; **Children's Book Test** de Hill et al., 2016) hizo viable entrenar redes profundas. Y en 2016, **SQuAD** (Rajpurkar et al., 2016) cambió las reglas: más de 100.000 preguntas escritas por humanos sobre artículos de Wikipedia, donde **la respuesta es siempre un span contiguo del contexto**.

Esa restricción de SQuAD — respuesta = subcadena del párrafo — define la tarea como *span extraction*: el modelo no genera texto, sino que predice dos índices (inicio y fin) sobre el párrafo. Las métricas oficiales son **Exact Match (EM)**, que exige coincidencia literal con alguna respuesta humana, y **F1**, una métrica más suave que mide la media ponderada de precisión y recall a nivel de tokens/caracteres. SQuAD reparte 90k/10k tuplas pregunta–contexto en train/dev, con un test oculto, y se convirtió de inmediato en el banco de pruebas competitivo de la época (leaderboard público en `stanford-qa.com`).

El ingrediente clave del avance previo había sido el **mecanismo de atención neuronal**, que permite al sistema enfocarse en la región del contexto más relevante para la pregunta (en analogía con la atención sobre regiones de una imagen en Visual QA). Pero los *attention readers* de 2015–2016 compartían tres limitaciones que BiDAF identifica explícitamente como su punto de ataque:

1. **Resumen prematuro en un vector de tamaño fijo.** La práctica dominante usaba los pesos de atención para extraer la información relevante del contexto y **comprimirla en un único vector de tamaño fijo**. Comprimir un párrafo entero en un vector pierde información: detalles que podrían ser cruciales para localizar el span se diluyen antes de llegar a las capas de decisión.

2. **Atención temporalmente dinámica (con memoria).** En el dominio textual, los pesos de atención del paso actual solían ser **función del vector atendido del paso anterior** (estilo Bahdanau et al., 2015). Esto acopla los pasos temporales: un error de atención en un paso temprano contamina los siguientes.

3. **Atención unidireccional.** La atención fluía en un solo sentido — típicamente la *query* atiende al párrafo de contexto (o a la imagen) — sin que el contexto atendiera de vuelta a la *query*. Trabajos previos como Memory Networks (Weston et al., 2015), AS Reader (Kadlec et al., 2016), Match-LSTM (Wang & Jiang, 2016) o Attention-over-Attention (Cui et al., 2016) compartían en distinto grado estas características.

Curiosamente, el dominio de visión ya apuntaba en la dirección correcta: Lu et al. (2016) mostraron en VQA que **atender también desde la imagen de vuelta hacia las palabras de la pregunta** mejoraba resultados. BiDAF traslada y formaliza esa intuición al dominio del lenguaje, pero con una diferencia decisiva: Lu et al. usaban los pesos de atención directamente en la capa de salida, mientras que BiDAF deja que la atención *fluya* hacia una capa de modelado posterior.

---

## 3. Idea central: la atención debe FLUIR en ambas direcciones, sin resumir prematuramente

La propuesta de BiDAF se puede condensar en dos principios complementarios.

**Principio 1 — Flujo de atención (attention-flow), no resumen de atención (attention-summarization).** La capa de atención **no** comprime el contexto y la pregunta en vectores fijos. En su lugar, calcula un vector atendido **para cada paso temporal** (cada token del contexto), y ese vector atendido — junto con las representaciones de las capas anteriores — **se deja fluir** hacia la *modeling layer* posterior. Cada token del contexto conserva su propia representación *query-aware*. Esto reduce la pérdida de información por resumen prematuro.

**Principio 2 — Atención bidireccional.** Se computan dos atenciones complementarias derivadas de una **matriz de similitud compartida**:
- **Context-to-Query (C2Q):** para cada palabra del contexto, ¿qué palabras de la pregunta son más relevantes?
- **Query-to-Context (Q2C):** ¿qué palabras del contexto tienen la mayor similitud con alguna palabra de la pregunta? (es decir, cuáles son críticas para responder).

Ambas direcciones aportan información complementaria. La intuición es que C2Q identifica "para esta palabra del contexto, qué pide la pregunta", mientras que Q2C identifica "globalmente, qué palabras del contexto importan dada la pregunta".

**Principio 3 — Atención sin memoria (memory-less).** Aunque la atención se calcula iterativamente a lo largo del tiempo, **la atención en cada paso depende solo de la query y del contexto en ese paso**, no de la atención del paso anterior. Los autores hipotetizan que esto produce una **división del trabajo**: la capa de atención se concentra en aprender la relación pregunta–contexto, y la *modeling layer* se concentra en aprender las interacciones internas dentro de la representación *query-aware*. Además, evita que atenciones incorrectas en pasos previos afecten al paso actual. Experimentalmente, la atención sin memoria supera a la dinámica.

La combinación de estos tres principios da una **representación del contexto consciente de la pregunta (query-aware) sin resumen temprano**, calculada por un proceso jerárquico multi-etapa que representa el contexto a distintos niveles de granularidad.

---

## 4. Arquitectura por capas

BiDAF es un proceso jerárquico de **seis capas** (Figura 1 del paper). Las tres primeras se aplican **tanto al contexto como a la query**; las tres últimas fusionan ambos y producen la respuesta.

Notación: contexto $\{x_1, \dots, x_T\}$ ($T$ palabras), query $\{q_1, \dots, q_J\}$ ($J$ palabras). El hiperparámetro central es la dimensión oculta $d$ (en los experimentos, $d = 100$).

### Capa 1 — Character Embedding (Char-CNN)

Mapea cada palabra a un espacio vectorial usando **CNN a nivel de caracteres**, siguiendo a Kim (2014). Cada carácter se embebe en un vector; estos vectores se tratan como entradas 1D a la CNN (la dimensión del carácter es el tamaño del canal de entrada). Las salidas de la CNN se someten a **max-pooling sobre todo el ancho** de la palabra para obtener un vector de tamaño fijo por palabra.

Rol: capturar morfología y manejar palabras **fuera de vocabulario (OOV) o raras**, que un embedding de palabra entero no representaría bien. En los experimentos se usan **100 filtros 1D de ancho 5**.

### Capa 2 — Word Embedding (GloVe)

Mapea cada palabra a un espacio vectorial usando **vectores pre-entrenados GloVe** (Pennington et al., 2014), que permanecen fijos. Rol complementario al char-embedding: GloVe captura mejor la **semántica de la palabra como un todo**.

La concatenación de los vectores de carácter y de palabra se pasa por una **Highway Network de dos capas** (Srivastava et al., 2015), cuyas compuertas aprendidas regulan cuánta señal de cada fuente (carácter vs. palabra) propagar. Las salidas son dos secuencias de vectores $d$-dimensionales, organizadas como matrices:

$$X \in \mathbb{R}^{d \times T} \quad \text{(contexto)}, \qquad Q \in \mathbb{R}^{d \times J} \quad \text{(query)}.$$

### Capa 3 — Contextual Embedding (BiLSTM)

Una **LSTM bidireccional** (Hochreiter & Schmidhuber, 1997) se coloca sobre los embeddings anteriores para modelar las **interacciones temporales entre palabras**. Se concatenan las salidas de las LSTM forward y backward (cada una con salida $d$-dimensional), produciendo vectores columna de dimensión $2d$:

$$H \in \mathbb{R}^{2d \times T} \quad \text{(del contexto } X\text{)}, \qquad U \in \mathbb{R}^{2d \times J} \quad \text{(de la query } Q\text{)}.$$

Los autores observan que estas tres primeras capas computan rasgos **a distintos niveles de granularidad** (carácter, palabra, frase/contexto), de modo análogo al cómputo multi-etapa de rasgos en las CNN de visión computacional.

### Capa 4 — Attention Flow Layer

El corazón del modelo. Enlaza y fusiona información de contexto y query produciendo, para cada palabra del contexto, una representación *query-aware*. **No** resume las dos modalidades en vectores únicos; deja fluir los vectores de atención hacia la capa de modelado. Entradas: $H$ y $U$. Salida: $G$ (las representaciones *query-aware* del contexto). Se detalla en la sección 5.

### Capa 5 — Modeling Layer (BiLSTM)

Recibe $G$ (representaciones *query-aware* de las palabras del contexto) y captura la **interacción entre las palabras del contexto condicionada a la query**. Esto difiere de la capa contextual (capa 3), que captura interacciones entre palabras del contexto **independientes de la query**. Se usan **dos capas de BiLSTM** con tamaño de salida $d$ por dirección, produciendo:

$$M \in \mathbb{R}^{2d \times T}.$$

Cada vector columna de $M$ contiene información contextual de la palabra respecto a **todo el párrafo y la query**.

### Capa 6 — Output Layer

Específica de la aplicación — la modularidad de BiDAF permite intercambiarla sin tocar el resto. Para QA, predice los índices de inicio y fin del span (sección 6). Para cloze-test (CNN/DailyMail), se modifica ligeramente.

---

## 5. La Attention Flow Layer en detalle

### Matriz de similitud compartida

Ambas direcciones de atención se derivan de una **matriz de similitud compartida** $S \in \mathbb{R}^{T \times J}$ entre los embeddings contextuales del contexto ($H$) y de la query ($U$), donde $S_{tj}$ indica la similitud entre la $t$-ésima palabra del contexto y la $j$-ésima palabra de la query:

$$S_{tj} = \alpha(H_{:t}, U_{:j}) \in \mathbb{R} \tag{1}$$

donde $\alpha$ es una **función escalar entrenable** que codifica la similitud entre sus dos vectores de entrada, $H_{:t}$ es la $t$-ésima columna de $H$ y $U_{:j}$ la $j$-ésima columna de $U$. La elección concreta es:

$$\alpha(h, u) = w_{(S)}^{\top}\,[h;\, u;\, h \circ u]$$

donde $w_{(S)} \in \mathbb{R}^{6d}$ es un vector de pesos entrenable, $\circ$ es producto elemento a elemento, y $[\,;\,]$ es concatenación por filas. El término $h \circ u$ (producto Hadamard) inyecta una señal multiplicativa de coincidencia componente a componente, complementando los términos lineales $h$ y $u$. La dimensión $6d$ proviene de concatenar tres vectores $2d$-dimensionales.

Esta matriz $S$ es el objeto central: **se calcula una vez** y alimenta ambas direcciones de atención, garantizando coherencia entre C2Q y Q2C.

### Context-to-Query Attention (C2Q)

Indica qué palabras de la query son más relevantes para cada palabra del contexto. Sea $a_t \in \mathbb{R}^J$ el vector de pesos de atención sobre las palabras de la query para la $t$-ésima palabra del contexto, con $\sum_j a_{tj} = 1$. Se calcula aplicando softmax sobre la fila $t$ de $S$:

$$a_t = \mathrm{softmax}(S_{t:}) \in \mathbb{R}^J$$

y cada vector de query atendido es la suma ponderada de las columnas de $U$:

$$\tilde{U}_{:t} = \sum_{j} a_{tj}\, U_{:j}.$$

Así, $\tilde{U} \in \mathbb{R}^{2d \times T}$ contiene los vectores de query atendidos para **todo el contexto** (una columna por palabra del contexto).

### Query-to-Context Attention (Q2C)

Indica qué palabras del contexto tienen la similitud más alta con **alguna** palabra de la query, y son por tanto críticas para responder. Se obtienen los pesos sobre las palabras del contexto tomando, para cada fila (palabra del contexto), el **máximo a lo largo de las columnas** de $S$, seguido de softmax:

$$b = \mathrm{softmax}\big(\mathrm{max}_{\mathrm{col}}(S)\big) \in \mathbb{R}^T.$$

El operador $\mathrm{max}_{\mathrm{col}}$ colapsa la dimensión de la query: para cada palabra del contexto se queda con su mejor coincidencia contra cualquier palabra de la query. El vector de contexto atendido es:

$$\tilde{h} = \sum_{t} b_t\, H_{:t} \in \mathbb{R}^{2d},$$

la suma ponderada de las palabras más importantes del contexto respecto a la query. Este único vector $\tilde{h}$ se **replica (tile) $T$ veces** a lo largo de las columnas, produciendo $\tilde{H} \in \mathbb{R}^{2d \times T}$.

Nota de diseño: la asimetría entre C2Q (un vector atendido distinto por cada palabra del contexto) y Q2C (un único vector global replicado) refleja que C2Q es de grano fino mientras Q2C aporta una señal global de "qué partes del contexto importan".

### Vector combinado $G$ con $\beta$

Finalmente, los embeddings contextuales y los vectores de atención se combinan en $G$, donde cada columna es la **representación query-aware** de la palabra de contexto correspondiente:

$$G_{:t} = \beta(H_{:t},\, \tilde{U}_{:t},\, \tilde{H}_{:t}) \in \mathbb{R}^{d_G} \tag{2}$$

donde $\beta$ es una función vectorial entrenable que **fusiona sus tres vectores de entrada** y $d_G$ es su dimensión de salida. Aunque $\beta$ puede ser una red arbitraria (p.ej. un MLP), la elección simple por concatenación funciona bien:

$$\beta(h, \tilde{u}, \tilde{h}) = [\,h;\, \tilde{u};\, h \circ \tilde{u};\, h \circ \tilde{h}\,] \in \mathbb{R}^{8d \times T}, \qquad d_G = 8d.$$

Lectura de los cuatro términos:
- $h$: la representación contextual original de la palabra de contexto.
- $\tilde{u}$: lo que la query "dice" sobre esa palabra (C2Q).
- $h \circ \tilde{u}$: coincidencia multiplicativa palabra-contexto vs. query atendida.
- $h \circ \tilde{h}$: coincidencia de la palabra con el resumen global Q2C del contexto.

El resultado $G$ es una representación rica de **$8d$ dimensiones por palabra de contexto**, que conserva tanto la señal original como ambas direcciones de atención — exactamente lo que significa "flujo de atención sin resumen prematuro".

---

## 6. Output Layer: predicción de spans y pérdida

La tarea de QA requiere encontrar una **sub-frase del párrafo** prediciendo sus índices de **inicio** y **fin**.

**Inicio.** La distribución de probabilidad sobre el índice de inicio en todo el párrafo:

$$p^1 = \mathrm{softmax}\big(w_{(p^1)}^{\top}\,[G; M]\big) \tag{3}$$

donde $w_{(p^1)} \in \mathbb{R}^{10d}$ es un vector de pesos entrenable. La dimensión $10d$ proviene de concatenar $G$ (de dimensión $8d$) con $M$ (de dimensión $2d$).

**Fin.** Para el índice de fin, $M$ se pasa por **otra BiLSTM** que produce $M^2 \in \mathbb{R}^{2d \times T}$, y luego:

$$p^2 = \mathrm{softmax}\big(w_{(p^2)}^{\top}\,[G; M^2]\big) \tag{4}$$

La LSTM adicional para el fin permite condicionar la predicción del cierre del span sobre la del inicio de manera implícita (a través de la dinámica recurrente).

**Función de pérdida (entrenamiento).** Es la suma de las log-verosimilitudes negativas de los índices verdaderos de inicio y fin, promediada sobre los ejemplos:

$$L(\theta) = -\frac{1}{N} \sum_{i}^{N} \Big[ \log\big(p^1_{y^1_i}\big) + \log\big(p^2_{y^2_i}\big) \Big] \tag{5}$$

donde $\theta$ es el conjunto de pesos entrenables (filtros de la CNN, celdas LSTM, $w_{(S)}$, $w_{(p^1)}$, $w_{(p^2)}$), $N$ el número de ejemplos, $y^1_i$ e $y^2_i$ los índices verdaderos de inicio y fin del ejemplo $i$, y $p^k_j$ el $j$-ésimo valor del vector $p^k$.

**Inferencia (test).** Se elige el span $(k, l)$ con $k \le l$ que **maximiza el producto** $p^1_k\, p^2_l$. Esta búsqueda se resuelve en **tiempo lineal con programación dinámica**, recorriendo el párrafo y manteniendo el mejor inicio visto hasta cada posición.

---

## 7. Resultados

### SQuAD (test oculto)

Resultados sobre el test oculto de SQuAD, reflejando el leaderboard al 6 de diciembre de 2016. EM y F1 (single model y ensemble):

| Modelo | Single EM | Single F1 | Ens. EM | Ens. F1 |
|--------|:---------:|:---------:|:-------:|:-------:|
| Logistic Regression Baseline (Rajpurkar et al.) | 40.4 | 51.0 | — | — |
| Dynamic Chunk Reader (Yu et al.) | 62.5 | 71.0 | — | — |
| Fine-Grained Gating (Yang et al.) | 62.5 | 73.3 | — | — |
| Match-LSTM (Wang & Jiang) | 64.7 | 73.7 | 67.9 | 77.0 |
| Multi-Perspective Matching (IBM Watson) | 65.5 | 75.1 | 68.2 | 77.2 |
| Dynamic Coattention Networks (Xiong et al.) | 66.2 | 75.9 | 71.6 | 80.4 |
| R-Net (Microsoft Research Asia) | 68.4 | 77.5 | 72.1 | 79.7 |
| **BiDAF (Ours)** | **68.0** | **77.3** | **73.3** | **81.1** |

El **ensemble de BiDAF alcanza EM 73.3 y F1 81.1**, superando a todos los enfoques previos del leaderboard al momento de la presentación. El ensemble se compone de **12 corridas** con arquitectura e hiperparámetros idénticos; en test se elige la respuesta con la mayor suma de scores de confianza entre las 12 corridas.

Como referencia humana de la época (no en la tabla del paper pero contexto estándar): el techo humano en SQuAD 1.1 ronda EM ~82 / F1 ~91, de modo que BiDAF dejaba aún un margen considerable.

**Detalles de entrenamiento.** Tokenización con PTB Tokenizer; $d = 100$; 100 filtros 1D de ancho 5 para la char-CNN; optimizador **AdaDelta** (Zeiler, 2012); minibatch 60; learning rate inicial 0.5; 12 épocas; **dropout 0.2** en CNN, todas las LSTM y la transformación lineal antes del softmax; **moving averages** de los pesos con decaimiento exponencial 0.999 (en test se usan los promedios móviles, no los pesos crudos). ~2.6M parámetros; ~20 h en una Titan X.

### CNN/DailyMail (cloze test)

Accuracy de validación/test (∗ indica métodos ensemble):

| Modelo | CNN val | CNN test | DM val | DM test |
|--------|:-------:|:--------:|:------:|:-------:|
| Attentive Reader (Hermann et al.) | 61.6 | 63.0 | 70.5 | 69.0 |
| AS Reader (Kadlec et al.) | 68.6 | 69.5 | 75.0 | 73.9 |
| Iterative Attention (Sordoni et al.) | 72.6 | 73.3 | — | — |
| EpiReader (Trischler et al.) | 73.4 | 74.0 | — | — |
| Stanford AR (Chen et al.) | 73.8 | 73.6 | 77.6 | 76.6 |
| GA Reader (Dhingra et al.) | 73.0 | 73.8 | 76.7 | 75.7 |
| AoA Reader (Cui et al.) | 73.1 | 74.4 | — | — |
| ReasoNet (Shen et al.) | 72.9 | 74.7 | — | — |
| **BiDAF (Ours)** | **76.3** | **76.9** | **80.3** | **79.6** |
| GA Reader∗ (ensemble) | 76.4 | 77.4 | 79.1 | 78.1 |
| Stanford AR∗ (ensemble) | 77.2 | 77.6 | 80.2 | 79.2 |

**BiDAF (single-run)** supera a todos los modelos previos single-run en ambos datasets, y en el test de **DailyMail incluso supera al mejor método ensemble**. Solo dos ensembles (GA Reader∗ y Stanford AR∗) lo superan en CNN.

**Adaptación al cloze.** Como en CNN/DailyMail la respuesta es siempre **una sola palabra (entidad anonimizada)**, solo se predice el índice de inicio $p^1$ (se omite $p^2$ de la pérdida). Se enmascaran las palabras no-entidad en la clasificación final. Como la entidad correcta puede aparecer múltiples veces, se **suman las probabilidades de todas las instancias** de la entidad correcta (estrategia tipo Kadlec et al., 2016) antes de calcular la pérdida. Cada artículo se trocea en oraciones de una ventana de 19 palabras alrededor de cada entidad; las RNN no propagan entre oraciones, lo que paraleliza el entrenamiento. Minibatch 48, 8 épocas con *early stop*, ~60 h en ocho Titan X.

---

## 8. Ablation study

Sobre el **dev set** de SQuAD (single runs salvo el ensemble final):

| Configuración | EM | F1 |
|---------------|:--:|:--:|
| No char embedding | 65.0 | 75.4 |
| No word embedding | 55.5 | 66.8 |
| No C2Q attention | 57.2 | 67.7 |
| No Q2C attention | 63.6 | 73.7 |
| Dynamic attention | 63.5 | 73.6 |
| **BiDAF (single)** | **67.7** | **77.3** |
| **BiDAF (ensemble)** | **72.6** | **80.7** |

Lecturas clave:

- **Char y word embedding son ambos importantes.** Quitar el word embedding es devastador (F1 cae a 66.8), confirmando que GloVe aporta la semántica de palabra completa; quitar el char embedding cuesta ~2 puntos de F1, asociado a OOV/raras.
- **C2Q es crítica.** Reemplazar el vector de query atendido $\tilde{U}$ por el promedio de las salidas de la LSTM contextual de la pregunta **hace caer más de 10 puntos en ambas métricas** (F1 de 77.3 a 67.7). Es el componente individual más valioso.
- **Q2C también ayuda.** Sin los términos con $\tilde{H}$ en $G$, F1 baja de 77.3 a 73.7 (~3.6 puntos).
- **El flujo de atención supera a la atención dinámica.** El modelo "Dynamic attention" (atención computada dentro de la LSTM de modelado, al estilo Bahdanau/Wang&Jiang) rinde F1 73.6 — **más de 3 puntos por debajo** del esquema estático/flujo de BiDAF (77.3). Pese a ser un mecanismo más simple, separar la atención de la capa de modelado produce un conjunto de rasgos más rico en las primeras 4 capas, que luego la *modeling layer* incorpora.

**Variaciones de $\alpha$ y $\beta$ (Apéndice B, dev set).** El paper estudia alternativas a la función de similitud y de fusión:

| Variación | EM | F1 |
|-----------|:--:|:--:|
| Eqn. 1: dot product — $\alpha = h^\top u$ | 65.5 | 75.5 |
| Eqn. 1: linear — $w_{lin}^\top[h;u]$ | 59.5 | 69.7 |
| Eqn. 1: bilinear — $h^\top W_{bi} u$ | 61.6 | 71.8 |
| Eqn. 1: linear after MLP | 66.2 | 76.4 |
| Eqn. 2: MLP after concat (ReLU sobre $\beta$) | 67.1 | 77.0 |
| **BiDAF (definición original)** | **68.0** | **77.3** |

La definición elegida de $\alpha$ (con el término $h \circ u$) supera a alternativas comunes de la literatura (dot, linear, bilinear). Añadir un MLP en $\beta$ **no ayuda**: rinde ligeramente peor que la concatenación simple — un argumento a favor de la navaja de Occam en el diseño de la fusión.

---

## 9. Visualización e interpretabilidad

Un aporte didáctico del paper es mostrar que **la matriz de atención es interpretable** — se puede leer qué información fue crucial.

**Espacios de embedding (Tabla 2).** Para palabras-pregunta frecuentes (When, Where, Who, etc.) se buscan las palabras del contexto con mayor similitud coseno, en dos espacios: el del *word embedding* (capa 2) y el del *phrase/contextual embedding* (capa 3). En el espacio de palabra, "When", "Where", "Who" **no** están bien alineadas con posibles respuestas. En el espacio contextual — una sola capa por debajo de la atención — el cambio es drástico: **"When" empieza a coincidir con años** (1945, 1991, 1971...), **"Where" con ubicaciones** (Rotterdam, area, location...), y **"Who" con nombres** (Guiscard, John, Thomas, Elway...). Esto evidencia que la capa contextual ya alinea los tipos de pregunta con sus respuestas plausibles.

**t-SNE de los meses (Figura 2a).** Visualizando los nombres de meses, en el espacio de palabra **"May" aparece separado del resto** porque tiene múltiples significados en inglés (mes vs. verbo modal "may"). La capa contextual, usando el contexto circundante, **logra separar los dos usos de "May"** — un ejemplo limpio de desambiguación contextual previa a los embeddings contextuales tipo ELMo/BERT.

**Matrices de atención (Figura 3).** Para tuplas pregunta–contexto reales: cada fila es una palabra de la pregunta, cada columna una palabra del contexto. En un ejemplo "Where" se ilumina sobre ubicaciones (at, Stadium, Levi's, Santa); en otro "many" se ilumina sobre cantidades y símbolos numéricos (hundreds, 15, 13, 9). Además, **las entidades de la pregunta típicamente atienden a las mismas entidades del contexto** (Super Bowl → Super Bowl, Warsaw → Warsaw), dando al modelo una señal para localizar la respuesta. Esta es la visualización que el PDF del profe muestra en la **slide 31**.

**Comparación con el baseline tradicional (Figura 2b, 2c).** Un diagrama de Venn muestra que BiDAF responde correctamente **más del 86%** de las preguntas que también acierta el baseline basado en rasgos lingüísticos (Rajpurkar et al., 2016), y el 14% restante no presenta un patrón claro. Esto sugiere que la arquitectura neuronal **captura gran parte de la información de los rasgos lingüísticos manuales**. Desglosado por la primera palabra de la pregunta, BiDAF supera al baseline en todas las categorías.

**Análisis de errores (Apéndice A).** Sobre 50 preguntas EM-incorrectas: **50% por límites imprecisos del span** (p.ej. predice "1 to 7" cuando la respuesta es "articles 1 to 7"), **28% por complejidad/ambigüedad sintáctica**, **14% por paráfrasis**, **4% por conocimiento externo**, **2% por requerir múltiples oraciones**, **2% por errores de tokenización/preprocesamiento**. Que la mitad de los errores sean de frontera del span anticipa una línea de mejora futura (búsqueda de span más fina).

---

## 10. Limitaciones

- **Pre-Transformer.** BiDAF se apoya enteramente en **BiLSTM** para el modelado secuencial. La recurrencia es inherentemente secuencial: no paraleliza bien sobre la longitud de la secuencia y arrastra el costo de propagar dependencias largas paso a paso. La *modeling layer* (2 BiLSTM) más la BiLSTM extra del fin de span hacen el modelo **computacionalmente costoso** en inferencia y entrenamiento (de ahí las ~20 h en Titan X para SQuAD).
- **Sin pre-entrenamiento profundo de lenguaje.** Los únicos pesos pre-entrenados son los GloVe (fijos). No hay pre-entrenamiento auto-supervisado masivo; toda la capacidad de comprensión se aprende desde las ~90k preguntas de SQuAD. Esto es lo que BERT volcaría de cabeza un año después.
- **Atención de un solo "hop".** La capa de atención se aplica una vez. Los autores señalan como trabajo futuro extender BiDAF a **múltiples hops** (estilo Memory Networks).
- **Span extraction puro.** SQuAD 1.1 garantiza que la respuesta es un span existente; BiDAF no maneja "no answer" (eso llegaría con SQuAD 2.0) ni respuestas abstractivas/generativas.
- **Errores de frontera dominantes.** El 50% de los errores por límites imprecisos del span muestra que el mecanismo de decodificación start/end, aunque eficiente, no resuelve bien la granularidad exacta de la respuesta.
- **Superado por BERT en 2018.** El fine-tuning de **BERT** (Devlin et al., 2018) sobre SQuAD superó ampliamente a BiDAF y a la mayoría de arquitecturas a medida de la era, alcanzando y luego superando el rendimiento humano, volviendo en gran medida obsoletas las arquitecturas específicas de QA basadas en RNN.

---

## 11. Impacto y legado

BiDAF se consolidó como la **arquitectura de referencia para span extraction en la era pre-BERT**. Su huella:

- **Estándar de facto para QA extractivo (2017).** Durante el periodo 2017, BiDAF fue el baseline obligado en el leaderboard de SQuAD y la base sobre la que se construyeron muchas variantes. Su demo interactiva pública lo hizo además muy citado en docencia.
- **Influencia directa en QANet** (Yu et al., 2018), que conserva la idea de una capa de atención contexto-query bidireccional pero **reemplaza las RNN por convoluciones + self-attention** (anticipando el Transformer), logrando entrenamiento mucho más rápido. La capa de fusión de QANet es heredera directa del $G$ de BiDAF.
- **Popularización de tres ideas de diseño** que sobrevivieron a la era RNN: (1) la **matriz de similitud compartida** entre dos secuencias como objeto central de la atención cruzada; (2) el principio de **no resumir prematuramente** (dejar fluir representaciones por token); (3) la **atención bidireccional** entre modalidades/secuencias como fuente de información complementaria — eco directo de la *co-attention* en VQA (Lu et al., 2016).
- **Embeddings contextuales como anticipo.** La observación de que la capa contextual (BiLSTM) desambigua "May" o alinea "When→años" prefigura conceptualmente lo que **ELMo** (2018, también de AI2/UW, mismo ecosistema) formalizaría como embeddings contextuales pre-entrenados.
- **Decodificación de span con DP** ($\arg\max_{k \le l} p^1_k p^2_l$) se volvió práctica estándar, reutilizada incluso en los heads de QA de BERT y sucesores.

En síntesis: BiDAF no introdujo un mecanismo radicalmente nuevo, sino que **articuló con claridad y validó empíricamente** los principios de "flujo, no resumen" y "atención bidireccional" que se convirtieron en vocabulario común del campo.

---

## 12. Conexión con la Clase 24

En el PDF de la Clase 24 del profesor, BiDAF aparece en las **slides 29–31** como el segundo gran hito de *machine comprehension* con atención, presentado bajo el lema **"attention should flow both ways"** (la atención debe fluir en ambas direcciones). Las slides muestran:

- **Slide 29–30:** la arquitectura completa de seis capas (la Figura 1 del paper), enfatizando que C2Q y Q2C se derivan de la misma matriz de similitud y que la representación *query-aware* $G$ fluye hacia la *modeling layer* sin resumirse.
- **Slide 31:** la **visualización de los pesos de atención** (Figura 3 del paper) — cómo "Where" atiende a ubicaciones y "many" a cantidades — usada para argumentar la interpretabilidad del mecanismo.

**Rol pedagógico en la secuencia de la clase.** BiDAF se presenta inmediatamente después del **Stanford Attentive Reader** (Chen et al.). La contraposición es deliberada y didáctica:

- El **Stanford Attentive Reader** ejemplifica la **atención unidireccional** clásica: la pregunta se resume en un vector que atiende sobre el contexto, produciendo una distribución de atención sobre las palabras del párrafo. Es el "antes".
- **BiDAF** es el "después": atención **bidireccional** (C2Q + Q2C) y, sobre todo, **sin resumen prematuro** — cada token del contexto conserva su representación atendida. En el ablation, justamente el componente cuya ausencia más penaliza (C2Q, −10 puntos de F1) es el que el Attentive Reader no tiene en grano fino.

La progresión de la clase (atención unidireccional → atención bidireccional con flujo → la posterior llegada de los Transformers/BERT) ofrece a Roberto una narrativa clara de cómo evolucionó la comprensión lectora neuronal: de comprimir la pregunta en un vector, a hacer fluir representaciones ricas por token en ambas direcciones, a finalmente pre-entrenar self-attention a gran escala. BiDAF es el puente conceptual entre el *attentive reading* temprano y la era Transformer.

---

## 13. Notas y enlaces

- **arXiv:** https://arxiv.org/abs/1611.01603 (Seo, Kembhavi, Farhadi, Hajishirzi — ICLR 2017).
- **Código / demo interactiva:** `allenai.github.io/bi-att-flow/`.
- **SQuAD:** Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text", EMNLP 2016. Leaderboard histórico: `stanford-qa.com`.
- **CNN/DailyMail cloze:** Hermann et al., "Teaching Machines to Read and Comprehend", NIPS 2015.
- **Componentes heredados:** Char-CNN (Kim, 2014); GloVe (Pennington et al., 2014); Highway Networks (Srivastava et al., 2015); LSTM (Hochreiter & Schmidhuber, 1997); AdaDelta (Zeiler, 2012); Dropout (Srivastava et al., 2014).
- **Co-attention en visión (inspiración bidireccional):** Lu et al., "Hierarchical Question-Image Co-Attention for VQA", NIPS 2016.
- **Sucesores directos:** QANet (Yu et al., 2018, convolución + self-attention); BERT (Devlin et al., 2018, pre-entrenamiento que superó la era de arquitecturas a medida).
- **Hiperparámetros memorizables:** $d=100$; 100 filtros char-CNN ancho 5; ~2.6M parámetros; $\alpha$ con $w_{(S)} \in \mathbb{R}^{6d}$; $\beta$ con salida $8d$; $w_{(p^1)} \in \mathbb{R}^{10d}$.
- **Resultado estrella:** SQuAD ensemble **EM 73.3 / F1 81.1**; ablation C2Q **−10 pts F1**; flujo vs. dinámica **+3 pts**.

---

*Documento de análisis para uso interno del curso IA UC. Todas las cifras provienen directamente del texto del paper (ICLR 2017, arXiv 1611.01603).*
