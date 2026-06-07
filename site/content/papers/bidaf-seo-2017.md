---
title: "BiDAF (Bidirectional Attention Flow for Machine Comprehension)"
weight: 115
math: true
---

{{< paper-card
    title="Bidirectional Attention Flow for Machine Comprehension"
    authors="Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi"
    year="2017"
    venue="ICLR 2017 (arXiv 1611.01603)"
    pdf="/papers/bidaf-seo-2017.pdf"
    arxiv="1611.01603" >}}
La arquitectura de referencia para *span extraction* en la era pre-Transformer. Su tesis cabe en una frase: la atencion debe **fluir en ambas direcciones** y **sin resumir prematuramente**. En vez de comprimir el contexto y la pregunta en un vector fijo, BiDAF calcula una **matriz de similitud compartida** $S \in \mathbb{R}^{T \times J}$ entre los embeddings contextuales del parrafo ($H$) y de la query ($U$), y de ahi deriva dos atenciones complementarias: **Context-to-Query** (que palabras de la pregunta importan para cada palabra del contexto) y **Query-to-Context** (que palabras del contexto son criticas). Esas representaciones *query-aware* fluyen, token por token, hacia una *modeling layer* posterior. Sobre SQuAD el ensemble alcanza **EM 73.3 / F1 81.1**, superando a todo el leaderboard de fines de 2016. BiDAF es el puente conceptual entre el *attentive reading* temprano y la era BERT.
{{< /paper-card >}}

---

## El problema -- limites de los attention readers

A finales de 2016 la comprension lectora neuronal (*machine comprehension*) vivia un momento bisagra. El cuello de botella historico no era arquitectonico sino de datos: conjuntos tempranos como MCTest eran demasiado pequenos para entrenar modelos de extremo a extremo. La llegada de datasets masivos de tipo *cloze* (CNN/DailyMail, Children's Book Test) y, sobre todo, de [SQuAD (Rajpurkar 2016)](/papers/squad-rajpurkar-2016) -- mas de 100.000 preguntas humanas sobre Wikipedia donde **la respuesta es siempre un span contiguo del contexto** -- hizo viable y competitivo el campo.

Esa restriccion define la tarea como *span extraction*: el modelo no genera texto, predice **dos indices** (inicio y fin) sobre el parrafo. Las metricas oficiales son **Exact Match (EM)**, que exige coincidencia literal con alguna respuesta humana, y **F1**, que mide la superposicion de tokens entre prediccion y referencia.

El ingrediente del avance previo habia sido el [mecanismo de atencion](/fundamentos/mecanismo-atencion), que permite enfocar la region del contexto mas relevante para la pregunta. Pero los *attention readers* de 2015-2016 compartian tres limitaciones que BiDAF ataca explicitamente:

1. **Resumen prematuro en un vector de tamano fijo.** La practica dominante usaba los pesos de atencion para comprimir el contexto relevante en un **unico vector**. Comprimir un parrafo entero pierde informacion: detalles cruciales para localizar el span se diluyen antes de llegar a las capas de decision.
2. **Atencion temporalmente dinamica (con memoria).** Los pesos de atencion del paso actual solian ser **funcion del vector atendido del paso anterior** (estilo Bahdanau 2015). Esto acopla los pasos: un error de atencion temprano contamina los siguientes.
3. **Atencion unidireccional.** La atencion fluia en un solo sentido -- tipicamente la query atiende al contexto -- sin que el contexto atendiera de vuelta a la query. El [Stanford Attentive Reader (Chen 2016)](/papers/stanford-attentive-reader-chen-2016), Match-LSTM, AS Reader o Attention-over-Attention compartian en distinto grado estas caracteristicas.

Curiosamente, el dominio de vision ya apuntaba en la direccion correcta: la *co-attention* en VQA (Lu et al. 2016) mostraba que atender tambien **desde la imagen de vuelta hacia la pregunta** mejoraba resultados. BiDAF traslada esa intuicion al lenguaje, con una diferencia decisiva: deja que la atencion *fluya* hacia una capa de modelado posterior en lugar de usarla directamente en la salida.

---

## Idea central -- attention flow bidireccional, sin resumen prematuro

La propuesta se condensa en tres principios.

**Principio 1 -- Flujo de atencion, no resumen de atencion.** La capa de atencion **no** comprime el contexto y la pregunta en vectores fijos. Calcula un vector atendido **para cada token del contexto**, y ese vector -- junto con las representaciones de las capas anteriores -- **fluye** hacia la *modeling layer*. Cada token conserva su propia representacion *query-aware*.

**Principio 2 -- Atencion bidireccional.** Se computan dos atenciones complementarias derivadas de una **matriz de similitud compartida**:

- **Context-to-Query (C2Q):** para cada palabra del contexto, que palabras de la pregunta son mas relevantes.
- **Query-to-Context (Q2C):** que palabras del contexto tienen la mayor similitud con alguna palabra de la pregunta (cuales son criticas para responder).

C2Q identifica "para esta palabra del contexto, que pide la pregunta"; Q2C identifica "globalmente, que palabras del contexto importan dada la pregunta".

**Principio 3 -- Atencion sin memoria (memory-less).** La atencion en cada paso depende **solo** de la query y del contexto en ese paso, no de la atencion del paso anterior. Los autores hipotetizan que esto produce una **division del trabajo**: la capa de atencion aprende la relacion pregunta-contexto, y la *modeling layer* aprende las interacciones internas dentro de la representacion *query-aware*. Experimentalmente la atencion sin memoria supera a la dinamica.

La combinacion da una representacion del contexto **consciente de la pregunta sin resumen temprano**, calculada por un proceso jerarquico que representa el contexto a distintos niveles de granularidad.

---

## Arquitectura por capas

BiDAF es un proceso jerarquico de **seis capas**. Las tres primeras se aplican **tanto al contexto como a la query**; las tres ultimas fusionan ambos y producen la respuesta. Notacion: contexto $\{x_1, \dots, x_T\}$ ($T$ palabras), query $\{q_1, \dots, q_J\}$ ($J$ palabras). El hiperparametro central es la dimension oculta $d = 100$.

**Capa 1 -- Character Embedding (Char-CNN).** Mapea cada palabra con una CNN a nivel de caracteres (Kim 2014): cada caracter se embebe, se aplica convolucion 1D y luego max-pooling sobre el ancho de la palabra. Rol: capturar morfologia y manejar palabras **fuera de vocabulario (OOV) o raras**. Se usan 100 filtros de ancho 5.

**Capa 2 -- Word Embedding (GloVe).** Vectores GloVe pre-entrenados y fijos, que capturan la semantica de la palabra como un todo. La concatenacion de los vectores de caracter y de palabra pasa por una **Highway Network de dos capas**, cuyas compuertas regulan cuanta senal de cada fuente propagar. Salidas:

$$X \in \mathbb{R}^{d \times T} \quad \text{(contexto)}, \qquad Q \in \mathbb{R}^{d \times J} \quad \text{(query)}.$$

**Capa 3 -- Contextual Embedding ([BiLSTM](/fundamentos/lstm-gru)).** Una LSTM bidireccional modela las interacciones temporales entre palabras. Se concatenan las salidas forward y backward (cada una $d$-dimensional), produciendo vectores de dimension $2d$:

$$H \in \mathbb{R}^{2d \times T} \quad \text{(del contexto)}, \qquad U \in \mathbb{R}^{2d \times J} \quad \text{(de la query)}.$$

Las tres primeras capas computan rasgos a distintos niveles de granularidad (caracter, palabra, frase), de modo analogo al computo multi-etapa de las CNN de vision.

**Capa 4 -- Attention Flow Layer.** El corazon del modelo. Enlaza y fusiona contexto y query produciendo, para cada palabra del contexto, una representacion *query-aware* $G$. No resume; deja fluir. Se detalla abajo.

**Capa 5 -- Modeling Layer (BiLSTM).** Recibe $G$ y captura la interaccion entre las palabras del contexto **condicionada a la query** -- algo distinto de la capa 3, que captura interacciones independientes de la query. Dos capas de BiLSTM producen $M \in \mathbb{R}^{2d \times T}$.

**Capa 6 -- Output Layer.** Especifica de la aplicacion; la modularidad de BiDAF permite intercambiarla. Para QA predice los indices de inicio y fin del span.

---

## Attention Flow Layer en detalle

### Matriz de similitud compartida

Ambas direcciones de atencion se derivan de una **matriz compartida** $S \in \mathbb{R}^{T \times J}$, donde $S_{tj}$ es la similitud entre la palabra $t$ del contexto y la palabra $j$ de la query:

$$S_{tj} = \alpha(H_{:t}, U_{:j}) \in \mathbb{R}, \qquad \alpha(h, u) = w_{(S)}^{\top}\,[h;\, u;\, h \circ u]$$

donde $w_{(S)} \in \mathbb{R}^{6d}$ es un vector entrenable, $\circ$ es producto elemento a elemento (Hadamard) y $[\,;\,]$ concatenacion. El termino $h \circ u$ inyecta una senal multiplicativa de coincidencia componente a componente, complementando los terminos lineales. La dimension $6d$ viene de concatenar tres vectores $2d$-dimensionales. Esta matriz **se calcula una sola vez** y alimenta ambas direcciones, garantizando coherencia entre C2Q y Q2C.

### Context-to-Query (C2Q)

Que palabras de la query importan para cada palabra del contexto. Se aplica softmax sobre la fila $t$ de $S$ y se pondera $U$:

$$a_t = \mathrm{softmax}(S_{t:}) \in \mathbb{R}^J, \qquad \tilde{U}_{:t} = \sum_{j} a_{tj}\, U_{:j}.$$

Asi $\tilde{U} \in \mathbb{R}^{2d \times T}$ contiene un vector de query atendido **por cada palabra del contexto**.

### Query-to-Context (Q2C)

Que palabras del contexto tienen la similitud mas alta con **alguna** palabra de la query. Se toma, por fila, el maximo a lo largo de las columnas de $S$, seguido de softmax:

$$b = \mathrm{softmax}\big(\mathrm{max}_{\mathrm{col}}(S)\big) \in \mathbb{R}^T, \qquad \tilde{h} = \sum_{t} b_t\, H_{:t} \in \mathbb{R}^{2d}.$$

El operador $\mathrm{max}_{\mathrm{col}}$ colapsa la dimension de la query: para cada palabra del contexto se queda con su mejor coincidencia contra cualquier palabra de la pregunta. Este unico vector $\tilde{h}$ se **replica (tile) $T$ veces**, produciendo $\tilde{H} \in \mathbb{R}^{2d \times T}$. La asimetria entre C2Q (un vector distinto por palabra del contexto) y Q2C (un vector global replicado) refleja que C2Q es de grano fino mientras Q2C aporta una senal global de "que partes del contexto importan".

### Vector combinado $G$

Los embeddings contextuales y los vectores de atencion se combinan en $G$, donde cada columna es la representacion *query-aware* de la palabra de contexto:

$$G_{:t} = \beta(H_{:t},\, \tilde{U}_{:t},\, \tilde{H}_{:t}), \qquad \beta(h, \tilde{u}, \tilde{h}) = [\,h;\, \tilde{u};\, h \circ \tilde{u};\, h \circ \tilde{h}\,] \in \mathbb{R}^{8d \times T}.$$

Lectura de los cuatro terminos: $h$ es la representacion contextual original; $\tilde{u}$ es lo que la query dice sobre esa palabra (C2Q); $h \circ \tilde{u}$ es coincidencia multiplicativa palabra-vs-query-atendida; $h \circ \tilde{h}$ es coincidencia con el resumen global Q2C. El resultado es una representacion de **$8d$ dimensiones por palabra**, que conserva la senal original y ambas direcciones de atencion -- exactamente lo que significa "flujo sin resumen prematuro".

---

## Output layer -- prediccion de spans

La tarea de QA requiere encontrar una subfrase del parrafo prediciendo sus indices de **inicio** y **fin**.

**Inicio.** Distribucion sobre el indice de inicio en todo el parrafo:

$$p^1 = \mathrm{softmax}\big(w_{(p^1)}^{\top}\,[G; M]\big), \qquad w_{(p^1)} \in \mathbb{R}^{10d}.$$

La dimension $10d$ viene de concatenar $G$ ($8d$) con $M$ ($2d$).

**Fin.** $M$ se pasa por **otra BiLSTM** que produce $M^2 \in \mathbb{R}^{2d \times T}$, y luego:

$$p^2 = \mathrm{softmax}\big(w_{(p^2)}^{\top}\,[G; M^2]\big).$$

La LSTM adicional permite condicionar implicitamente el fin del span sobre el inicio.

**Perdida (entrenamiento).** Suma de log-verosimilitudes negativas de los indices verdaderos de inicio y fin:

$$L(\theta) = -\frac{1}{N} \sum_{i}^{N} \Big[ \log\big(p^1_{y^1_i}\big) + \log\big(p^2_{y^2_i}\big) \Big].$$

**Inferencia.** Se elige el span $(k, l)$ con $k \le l$ que maximiza el producto $p^1_k\, p^2_l$, resuelto en **tiempo lineal con programacion dinamica**.

---

## Resultados

### SQuAD (test oculto, leaderboard al 6 dic 2016)

| Modelo | Single EM | Single F1 | Ens. EM | Ens. F1 |
|--------|:---------:|:---------:|:-------:|:-------:|
| Logistic Regression Baseline | 40.4 | 51.0 | -- | -- |
| Match-LSTM (Wang & Jiang) | 64.7 | 73.7 | 67.9 | 77.0 |
| Multi-Perspective Matching (IBM) | 65.5 | 75.1 | 68.2 | 77.2 |
| Dynamic Coattention (Xiong et al.) | 66.2 | 75.9 | 71.6 | 80.4 |
| R-Net (MSRA) | 68.4 | 77.5 | 72.1 | 79.7 |
| **BiDAF (Ours)** | **68.0** | **77.3** | **73.3** | **81.1** |

El **ensemble de BiDAF alcanza EM 73.3 / F1 81.1**, superando a todo el leaderboard al momento de la presentacion. El ensemble combina 12 corridas con arquitectura e hiperparametros identicos; en test se elige la respuesta con la mayor suma de scores de confianza. Como referencia de la epoca, el techo humano en SQuAD 1.1 ronda EM ~82 / F1 ~91, de modo que BiDAF dejaba aun un margen considerable.

Detalles: $d=100$; optimizador AdaDelta; minibatch 60; learning rate 0.5; 12 epocas; dropout 0.2; moving averages de pesos con decaimiento 0.999; ~2.6M parametros; ~20 h en una Titan X.

### CNN/DailyMail (cloze test, accuracy)

| Modelo | CNN test | DM test |
|--------|:--------:|:-------:|
| Attentive Reader (Hermann et al.) | 63.0 | 69.0 |
| AS Reader (Kadlec et al.) | 69.5 | 73.9 |
| Stanford AR (Chen et al.) | 73.6 | 76.6 |
| **BiDAF (Ours, single)** | **76.9** | **79.6** |
| Stanford AR* (ensemble) | 77.6 | 79.2 |

BiDAF single-run supera a todos los modelos previos single-run en ambos datasets; en DailyMail incluso supera al mejor metodo ensemble. Como en cloze la respuesta es una sola palabra (entidad anonimizada), solo se predice $p^1$ y se suman las probabilidades de todas las instancias de la entidad correcta antes de la perdida.

### Ablation (dev set de SQuAD)

| Configuracion | EM | F1 |
|---------------|:--:|:--:|
| No word embedding | 55.5 | 66.8 |
| No char embedding | 65.0 | 75.4 |
| No C2Q attention | 57.2 | 67.7 |
| No Q2C attention | 63.6 | 73.7 |
| Dynamic attention | 63.5 | 73.6 |
| **BiDAF (single)** | **67.7** | **77.3** |

Lecturas clave: **C2Q es el componente individual mas valioso** -- sin el, F1 cae mas de 10 puntos (77.3 a 67.7). Q2C tambien ayuda (~3.6 puntos). El **flujo de atencion supera a la atencion dinamica** en mas de 3 puntos de F1: separar la atencion de la capa de modelado produce un conjunto de rasgos mas rico. Quitar GloVe es devastador (F1 a 66.8); quitar el char embedding cuesta ~2 puntos. En el apendice, la definicion elegida de $\alpha$ con el termino $h \circ u$ supera a dot, linear y bilinear, y anadir un MLP en $\beta$ no ayuda.

---

## Interpretabilidad

Un aporte didactico del paper es mostrar que **la matriz de atencion es interpretable**.

- **Espacios de embedding.** Para palabras-pregunta (When, Where, Who) se buscan las palabras del contexto con mayor similitud coseno. En el espacio de palabra (capa 2) no estan bien alineadas; en el espacio contextual (capa 3, una capa por debajo de la atencion) el cambio es drastico: "When" empieza a coincidir con anos (1945, 1991...), "Where" con ubicaciones, "Who" con nombres. La capa contextual ya alinea tipos de pregunta con respuestas plausibles.
- **Desambiguacion de "May".** En el espacio de palabra "May" aparece separado del resto por su doble sentido (mes vs. verbo modal). La capa contextual logra separar los dos usos -- un anticipo limpio de lo que [ELMo/BERT](/fundamentos/embeddings-contextualizados) formalizarian como embeddings contextuales.
- **Matrices de atencion.** Para tuplas reales, "Where" se ilumina sobre ubicaciones y "many" sobre cantidades; las entidades de la pregunta atienden a las mismas entidades del contexto (Super Bowl -> Super Bowl), dando una senal para localizar la respuesta.
- **Analisis de errores.** Sobre 50 preguntas EM-incorrectas: **50% por limites imprecisos del span** (p.ej. "1 to 7" cuando la respuesta es "articles 1 to 7"), 28% por complejidad sintactica, 14% por parafrasis, 4% por conocimiento externo. Que la mitad de los errores sean de frontera anticipa una linea de mejora futura.

---

## Limitaciones (pre-Transformer)

- **Pre-Transformer.** BiDAF se apoya enteramente en BiLSTM. La recurrencia es inherentemente secuencial: no paraleliza bien sobre la longitud de la secuencia y arrastra el costo de propagar dependencias largas paso a paso (de ahi las ~20 h en Titan X). Contrasta con el paralelismo de la [self-attention](/fundamentos/self-attention) que llegaria con el Transformer.
- **Sin pre-entrenamiento profundo de lenguaje.** Los unicos pesos pre-entrenados son los GloVe (fijos). Toda la capacidad de comprension se aprende desde las ~90k preguntas de SQuAD -- justo lo que [BERT (Devlin 2018)](/papers/bert-devlin-2018) volcaria de cabeza un ano despues con pre-entrenamiento auto-supervisado masivo.
- **Atencion de un solo "hop".** La capa de atencion se aplica una vez; los autores senalan multiples hops como trabajo futuro.
- **Span extraction puro.** SQuAD 1.1 garantiza que la respuesta existe; BiDAF no maneja "no answer" (eso llegaria con SQuAD 2.0) ni respuestas abstractivas.
- **Errores de frontera dominantes.** El 50% de errores por limites imprecisos del span muestra que la decodificacion start/end, aunque eficiente, no resuelve bien la granularidad exacta.
- **Superado por BERT en 2018.** El fine-tuning de BERT sobre SQuAD supero ampliamente a BiDAF, alcanzando y luego superando el rendimiento humano, volviendo obsoletas las arquitecturas a medida basadas en RNN.

---

## Por que importa hoy

BiDAF se consolido como la **arquitectura de referencia para span extraction en la era pre-BERT**. Su huella sobrevive a la era RNN:

- **Influencia directa en QANet** (Yu et al. 2018), que conserva la atencion contexto-query bidireccional pero reemplaza las RNN por convoluciones + self-attention, anticipando el Transformer. Su capa de fusion hereda directamente el $G$ de BiDAF.
- **Tres ideas de diseno duraderas:** (1) la **matriz de similitud compartida** entre dos secuencias como objeto central de la atencion cruzada; (2) el principio de **no resumir prematuramente**; (3) la **atencion bidireccional** como fuente de informacion complementaria.
- **Embeddings contextuales como anticipo.** La observacion de que la BiLSTM contextual desambigua "May" o alinea "When -> anos" prefigura lo que ELMo formalizaria (mismo ecosistema AI2/UW).
- **Decodificacion de span con DP** ($\arg\max_{k \le l} p^1_k p^2_l$) se volvio practica estandar, reutilizada incluso en los heads de QA de BERT y sucesores.

BiDAF no introdujo un mecanismo radicalmente nuevo, sino que **articulo con claridad y validó empiricamente** los principios de "flujo, no resumen" y "atencion bidireccional" que se convirtieron en vocabulario comun del campo.

---

## Conexion con la clase 24

En el PDF de la [Clase 24](/clases/clase-24), BiDAF aparece en las **slides 29-31** como el segundo gran hito de *machine comprehension* con atencion, bajo el lema "attention should flow both ways":

- **Slides 29-30:** la arquitectura completa de seis capas, enfatizando que C2Q y Q2C se derivan de la misma matriz de similitud y que la representacion *query-aware* $G$ fluye hacia la *modeling layer* sin resumirse.
- **Slide 31:** la visualizacion de los pesos de atencion -- como "Where" atiende a ubicaciones y "many" a cantidades -- para argumentar la interpretabilidad.

BiDAF se presenta inmediatamente despues del [Stanford Attentive Reader](/papers/stanford-attentive-reader-chen-2016), y la contraposicion es deliberada:

- El **Stanford Attentive Reader** ejemplifica la **atencion unidireccional** clasica: la pregunta se resume en un vector que atiende sobre el contexto. Es el "antes".
- **BiDAF** es el "despues": atencion **bidireccional** (C2Q + Q2C) y, sobre todo, **sin resumen prematuro**. En el ablation, justamente el componente cuya ausencia mas penaliza (C2Q, -10 puntos de F1) es el que el Attentive Reader no tiene en grano fino.

La progresion de la clase -- atencion unidireccional, atencion bidireccional con flujo, y la posterior llegada de Transformers/BERT -- ofrece una narrativa clara de como evoluciono la [comprension lectora neuronal](/fundamentos/machine-reading-comprehension): de comprimir la pregunta en un vector, a hacer fluir representaciones ricas por token, a finalmente pre-entrenar self-attention a gran escala.

---

## Notas y enlaces

- **arXiv:** [1611.01603](https://arxiv.org/abs/1611.01603) (Seo, Kembhavi, Farhadi, Hajishirzi -- ICLR 2017).
- **Codigo / demo interactiva:** `allenai.github.io/bi-att-flow/`.
- **Componentes heredados:** Char-CNN (Kim 2014); GloVe (Pennington 2014); Highway Networks (Srivastava 2015); LSTM (Hochreiter & Schmidhuber 1997); AdaDelta (Zeiler 2012).
- **Inspiracion bidireccional:** Lu et al., "Hierarchical Question-Image Co-Attention for VQA", NIPS 2016.
- **Sucesores:** QANet (Yu et al. 2018, convolucion + self-attention); BERT (Devlin 2018, pre-entrenamiento que supero la era de arquitecturas a medida).
- **Hiperparametros memorizables:** $d=100$; 100 filtros char-CNN ancho 5; ~2.6M parametros; $w_{(S)} \in \mathbb{R}^{6d}$; $\beta$ con salida $8d$; $w_{(p^1)} \in \mathbb{R}^{10d}$. Resultado estrella: SQuAD ensemble **EM 73.3 / F1 81.1**.

Ver fundamentos: [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension) - [Question Answering](/fundamentos/question-answering) - [Mecanismo de atencion](/fundamentos/mecanismo-atencion) - [Self-attention](/fundamentos/self-attention) - [LSTM y GRU](/fundamentos/lstm-gru).

Ver papers: [SQuAD (Rajpurkar 2016)](/papers/squad-rajpurkar-2016) - [Stanford Attentive Reader (Chen 2016)](/papers/stanford-attentive-reader-chen-2016) - [BERT (Devlin 2018)](/papers/bert-devlin-2018).

Ver clase: [Clase 24 -- Machine Reading Comprehension](/clases/clase-24).
