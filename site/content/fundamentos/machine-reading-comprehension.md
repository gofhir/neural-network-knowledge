---
title: "Machine Reading Comprehension"
weight: 91
math: true
---

**Machine Reading Comprehension (MRC)** es la tarea de hacer que un sistema lea un texto y responda preguntas sobre él. Es la formulación más directa de "enseñar a las máquinas a leer y comprender": se le entrega al modelo un **pasaje** $P$ (un párrafo, un artículo, un documento) y una **pregunta** $Q$ sobre ese pasaje, y se espera una **respuesta** $A$ que esté justificada por el texto. A diferencia del question answering de dominio abierto —donde el sistema debe primero buscar dónde está la respuesta en una colección enorme—, en MRC el contexto relevante ya está dado: la dificultad está enteramente en **comprender** ese pasaje corto lo suficientemente bien como para extraer o generar la respuesta.

MRC ocupa un lugar central en la historia del NLP moderno porque fue el banco de pruebas donde se afinaron muchos de los mecanismos que hoy damos por sentados: la atención query-context, la predicción de spans, el fine-tuning de modelos pre-entrenados. Esta página consolida los conceptos transversales del área —definición, formulaciones de la tarea, la red neuronal genérica, la evolución de arquitecturas desde el Attentive Reader hasta BERT, la predicción de spans, el rol de la atención, datasets, limitaciones y la situación en la era LLM— y sirve como fundamento de la **[Clase 24](/clases/clase-24)** del curso IA UC.

---

## 1. Definición: MRC como instancia de QA sobre un pasaje

MRC es, formalmente, una **instancia de [question answering](/fundamentos/question-answering)** restringida a un contexto explícito y acotado. La formulación canónica es:

$$ (P, Q) \longrightarrow A $$

donde:

- $P$ (**passage**) es el texto que el sistema debe leer. Puede ser una sola oración, un párrafo de Wikipedia, un artículo de noticias completo de 800 tokens, o más.
- $Q$ (**question**) es la pregunta formulada sobre el contenido de $P$.
- $A$ (**answer**) es la respuesta, cuya forma depende de la formulación de la tarea (una palabra, un span, una opción, texto libre).

La distinción operativa frente al QA de dominio abierto es el **retrieval**. En open-domain QA el sistema recibe solo $Q$ y debe localizar el contexto relevante en una colección gigante (Wikipedia entera, la web) antes de poder responder. En MRC clásico, el pasaje ya viene dado: no hay que buscar, hay que **entender**. Por eso MRC se llama también *reading comprehension*: la analogía pedagógica es la prueba de comprensión lectora escolar, donde al estudiante se le da un texto y preguntas sobre él.

{{< concept-alert type="clave" >}}
La pregunta de fondo de MRC no es "¿dónde está la respuesta?" sino "¿qué nivel de comprensión del texto se necesita para responder?". Esta distinción es la que hace de MRC un termómetro de comprensión —y la que motivó las auditorías críticas de datasets que veremos en la Sección 8.
{{< /concept-alert >}}

La motivación práctica es enorme: asistentes que responden sobre documentos, buscadores que dan respuestas directas en vez de enlaces, sistemas de soporte que leen manuales, herramientas clínicas que responden sobre una historia médica. Toda la generación de productos tipo "chatea con tu PDF" descansa conceptualmente sobre MRC.

---

## 2. Formulaciones de la tarea

MRC no es una sola tarea sino una familia. El eje que las separa es **qué forma tiene la respuesta $A$** y cómo se evalúa. Las cuatro formulaciones canónicas:

| Formulación | Forma de $A$ | Cómo se predice | Dataset típico |
| --- | --- | --- | --- |
| **Cloze** | Una palabra/entidad omitida del texto | Clasificación sobre entidades candidatas del pasaje | CNN/DailyMail, Children's Book Test |
| **Multiple choice** | Una opción de un conjunto cerrado | Clasificación sobre las $k$ opciones dadas | MCTest, RACE, SWAG |
| **Span extraction** | Una subcadena contigua del pasaje | Predecir índices de inicio y fin | SQuAD, NewsQA |
| **Free-form / generative** | Texto libre, no necesariamente literal | Generación autoregresiva (encoder-decoder) | MS MARCO, NarrativeQA |

### 2.1 Cloze

En la formulación **cloze** (de *Cloze test*, prueba de completar) la pregunta es una oración con una palabra o entidad reemplazada por un marcador como `@placeholder`, y la tarea es predecir qué iba ahí. El dataset CNN/DailyMail (Hermann 2015) es el ejemplo paradigmático: toma un *bullet point* editorial del artículo, borra una entidad, y pide recuperarla. Es barata de construir a escala masiva porque no requiere anotación humana, pero —como veremos— produce un task más fácil de lo que aparenta.

### 2.2 Multiple choice

Se entrega la pregunta junto con un conjunto cerrado de opciones (típicamente 4) y el sistema elige una. Reduce MRC a un problema de **clasificación** sobre las opciones. RACE (exámenes de inglés de China) y MCTest son ejemplos. La ventaja evaluativa es que la métrica es trivial (accuracy); la desventaja es que el modelo puede explotar artefactos de las opciones distractoras sin entender el pasaje.

### 2.3 Span extraction

La formulación más influyente. La respuesta es **siempre una subcadena contigua del pasaje**, y el modelo predice dos índices: dónde empieza y dónde termina el span. Es la formulación de **SQuAD** (Rajpurkar 2016), que dominó la investigación de MRC entre 2016 y 2019. Su atractivo: las respuestas están garantizadas en el texto (no hay alucinación posible), pero la pregunta y la respuesta son lenguaje natural genuino, no entidades anonimizadas. La detallamos en la Sección 5.

### 2.4 Free-form / generative

La respuesta es texto libre que puede no aparecer literalmente en el pasaje: el sistema parafrasea, sintetiza o razona sobre la evidencia. Requiere un decoder generativo (arquitectura [seq2seq](/fundamentos/seq2seq) encoder-decoder). MS MARCO (consultas reales de Bing) y NarrativeQA son ejemplos. Es la formulación más cercana a cómo responde un humano —y la dominante en la era de los LLMs.

---

## 3. La red neuronal genérica para MRC

Pese a la variedad de formulaciones, casi todos los modelos neuronales de MRC pre-Transformer comparten una **plantilla de tres pasos**. La probabilidad de una respuesta $a$ dado el contexto $c$ y la pregunta $q$ se modela como:

$$ p(a \mid c, q) = \mathrm{softmax}\big(W(a)\, g(c, q)\big) $$

donde $g(c, q)$ es una representación conjunta del par contexto-pregunta y $W(a)$ es la matriz de pesos que la proyecta sobre el espacio de respuestas. Los tres pasos:

**Paso 1 — Encode.** Codificar $c$ y $q$ por separado en representaciones vectoriales. En la era pre-Transformer esto se hace con RNNs bidireccionales (BiLSTM o BiGRU) sobre embeddings de palabra (típicamente GloVe), produciendo una secuencia de embeddings contextuales para el pasaje $\{\tilde{p}_1, \dots, \tilde{p}_m\}$ y una representación de la pregunta $q$ (un vector único o una secuencia).

**Paso 2 — Combinar.** Fusionar la información de $c$ y $q$ en una representación conjunta $g(c, q)$. Aquí es donde vive la innovación arquitectónica: puede ser un simple MLP, pero lo que funciona de verdad es algún tipo de **[atención](/fundamentos/mecanismo-atencion)** que deja a la pregunta "consultar" el pasaje (o viceversa). Este paso es el corazón de MRC.

**Paso 3 — Producir la respuesta.** Mapear $g(c, q)$ a una respuesta según la formulación:

- **Clasificador** sobre candidatos (cloze, multiple choice): softmax sobre entidades u opciones.
- **Dos predictores de posición** (span extraction): softmax sobre tokens para inicio y para fin.
- **Decoder generativo** (free-form): generación token a token.

{{< concept-alert type="info" >}}
Esta plantilla "encode → combinar con atención → predecir" es notablemente estable. Lo que cambia entre el Attentive Reader (2015), BiDAF (2017) y BERT for QA (2018) es **cómo** se combina $c$ y $q$ y **cuán pre-entrenado** está el encoder, no la estructura de tres pasos.
{{< /concept-alert >}}

---

## 4. Evolución de arquitecturas

El corazón de este fundamento es la genealogía de modelos de MRC. Cada uno refina cómo la pregunta y el pasaje interactúan vía atención.

### 4.1 Attentive Reader (Hermann 2015)

El primer modelo neuronal de MRC de impacto, introducido junto al dataset CNN/DailyMail en *Teaching Machines to Read and Comprehend* (NIPS 2015). La idea: codificar la pregunta en un vector $q$, codificar el pasaje con una BiLSTM, y usar **atención de la query sobre el pasaje** para producir un vector de salida que resume "qué parte del pasaje importa para esta pregunta". Ese resumen se combina con $q$ a través de una capa no lineal (una MLP con $\tanh$) antes de predecir. El defecto conceptual —que sus sucesores atacarían— es que **resume el pasaje en un único vector de tamaño fijo** antes de decidir, perdiendo detalle.

### 4.2 Stanford Attentive Reader (Chen 2016)

Danqi Chen, Jason Bolton y Christopher Manning ([ficha del paper](/papers/stanford-attentive-reader-chen-2016)) simplificaron el Attentive Reader y, sorprendentemente, lo mejoraron en 7-10 puntos de accuracy. La innovación clave es la **atención bilineal**. Dado el embedding de la pregunta $q \in \mathbb{R}^h$ y los embeddings contextuales del pasaje $\tilde{p}_i \in \mathbb{R}^h$ (de una BiGRU bidireccional), los pesos de atención se calculan con un término bilineal aprendido $W_s \in \mathbb{R}^{h \times h}$:

$$ \alpha_i = \mathrm{softmax}_i\, \big( q^\top W_s\, \tilde{p}_i \big) $$

$$ o = \sum_i \alpha_i\, \tilde{p}_i $$

El término bilineal $q^\top W_s \tilde{p}_i$ es más expresivo que el producto punto $q^\top \tilde{p}_i$: aprende **qué dimensiones de la pregunta deben alinearse con qué dimensiones del pasaje** antes de medir similitud, en vez de comparar dimensión por dimensión. El vector $o$ es la combinación ponderada de los embeddings del pasaje. La predicción se restringe a las entidades candidatas presentes en el pasaje:

$$ a = \arg\max_{a \in p \cap E}\, W_a^\top o $$

Las otras dos simplificaciones de Chen —predecir directamente desde $o$ sin una capa extra, y restringir el argmax a entidades del pasaje— son por parsimonia; los autores son explícitos en que **solo la atención bilineal importa de verdad**. Es la arquitectura didáctica canónica de MRC con atención (aparece en CS224n de Stanford) por su balance entre simplicidad y completitud.

### 4.3 BiDAF (Seo 2017)

*Bidirectional Attention Flow* ([ficha del paper](/papers/bidaf-seo-2017)), de Minjoon Seo et al., introduce dos principios que se volvieron vocabulario común del campo, resumidos en el lema **"attention should flow both ways"**:

**1. Atención bidireccional.** En vez de que solo la pregunta atienda al pasaje, BiDAF computa **dos** atenciones complementarias a partir de una **matriz de similitud compartida** $S \in \mathbb{R}^{T \times J}$ (con $T$ palabras de contexto y $J$ de query):

$$ S_{tj} = w_{(S)}^\top\, [\, h_t;\, u_j;\, h_t \circ u_j \,] $$

donde $h_t$ y $u_j$ son los embeddings contextuales del token $t$ del contexto y $j$ de la query, $\circ$ es producto Hadamard, y $w_{(S)} \in \mathbb{R}^{6d}$ es aprendido. De esta matriz salen:

- **Context-to-Query (C2Q):** para cada palabra del contexto, qué palabras de la pregunta son relevantes. $a_t = \mathrm{softmax}(S_{t:})$, y $\tilde{U}_{:t} = \sum_j a_{tj}\, U_{:j}$.
- **Query-to-Context (Q2C):** qué palabras del contexto son globalmente más críticas para la pregunta. $b = \mathrm{softmax}(\max_{\text{col}}(S))$, produciendo un vector global replicado a lo largo del contexto.

**2. Attention flow sin resumen prematuro.** A diferencia del Attentive Reader, BiDAF **no** comprime el pasaje en un vector fijo. Cada token del contexto conserva su propia representación *query-aware*, formada por concatenación:

$$ G_{:t} = [\, h_t;\, \tilde{u}_t;\, h_t \circ \tilde{u}_t;\, h_t \circ \tilde{h}_t \,] \in \mathbb{R}^{8d} $$

Esta secuencia $G$ —una representación rica por token, no un resumen— fluye hacia una *modeling layer* (BiLSTM) que captura interacciones entre palabras del contexto **condicionadas a la query**. El ablation del paper confirma que C2Q es el componente más valioso (quitarlo cuesta ~10 puntos de F1) y que el flujo supera a la atención dinámica con memoria. BiDAF predice spans (Sección 5) y fue el estándar de facto para QA extractivo en 2017.

### 4.4 BERT for QA (Devlin 2018)

[BERT](/fundamentos/bert) cambió las reglas. En vez de diseñar una arquitectura de atención a medida, se **concatena la pregunta y el pasaje en una sola secuencia** y se deja que la [self-attention](/fundamentos/self-attention) del [Transformer](/fundamentos/transformer) modele todas las interacciones:

```
[CLS] question tokens [SEP] passage tokens [SEP]
```

La self-attention bidireccional dentro de cada capa hace que cada token del pasaje atienda a cada token de la pregunta y viceversa —es la atención bidireccional de BiDAF, pero generalizada y multiplicada por las decenas de cabezas y capas del Transformer, y **pre-entrenada** sobre miles de millones de tokens. Para span extraction se añaden dos vectores aprendidos: un vector de inicio $S$ y uno de fin $E$. Para cada token $i$ con representación final $T_i$:

$$ p^{\text{start}}_i = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}, \qquad p^{\text{end}}_i = \frac{e^{E \cdot T_i}}{\sum_j e^{E \cdot T_j}} $$

El salto de rendimiento sobre BiDAF fue tan grande que volvió obsoletas las arquitecturas de QA basadas en RNN: BERT superó el techo humano en SQuAD 1.1. La lección fue que **el pre-entrenamiento auto-supervisado masivo importa más que la ingeniería de la capa de atención**.

### 4.5 Generative MRC (BART / T5)

La última etapa devuelve a MRC a la formulación generativa. Modelos encoder-decoder pre-entrenados como **T5** y **BART** tratan MRC como text-to-text: el encoder lee pasaje y pregunta concatenados, y el decoder **genera la respuesta** token a token, sin restringirla a un span del texto. Esto permite respuestas abstractivas (parafraseadas, sintetizadas, razonadas sobre múltiples fragmentos), a costa de reintroducir el riesgo de **alucinación** —la respuesta puede no estar literalmente soportada por el pasaje. Es el paradigma que adoptan los LLMs modernos.

| Modelo | Año | Combinación $c$-$q$ | Predicción | Pre-entrenamiento |
| --- | --- | --- | --- | --- |
| Attentive Reader | 2015 | Atención query→passage, resumen fijo | Clasificación entidad | Solo embeddings |
| Stanford AR | 2016 | Atención bilineal $q^\top W_s \tilde{p}_i$ | Clasificación entidad | GloVe fijos |
| BiDAF | 2017 | Atención bidireccional C2Q + Q2C, flow | Span (start/end) | GloVe fijos |
| BERT for QA | 2018 | Self-attention sobre `[Q;P]` concatenados | Span (start/end) | MLM masivo |
| BART / T5 | 2020 | Encoder-decoder, cross-attention | Generación libre | Denoising masivo |

---

## 5. Span prediction en detalle

La predicción de spans es la mecánica más importante de MRC moderno. La respuesta es una subcadena contigua, así que el modelo solo necesita predecir **dónde empieza** y **dónde termina**.

**Predicción.** Sobre la secuencia de representaciones de los $T$ tokens del pasaje, el modelo produce dos distribuciones de probabilidad —una para el inicio y otra para el fin— vía softmax sobre las posiciones:

$$ p^{\text{start}} = \mathrm{softmax}(s_1, \dots, s_T), \qquad p^{\text{end}} = \mathrm{softmax}(e_1, \dots, e_T) $$

donde $s_i$ y $e_i$ son scores escalares por token (en BERT, los productos $S \cdot T_i$ y $E \cdot T_i$; en BiDAF, proyecciones de $[G; M]$).

**Loss.** El entrenamiento minimiza la suma de dos cross-entropies, una para la posición verdadera de inicio $y^1$ y otra para la de fin $y^2$, promediada sobre los $N$ ejemplos:

$$ \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \Big[ \log p^{\text{start}}_{y^1_i} + \log p^{\text{end}}_{y^2_i} \Big] $$

Es decir, cada token correcto de inicio y de fin se trata como una etiqueta de clasificación independiente sobre las $T$ posiciones.

**Decodificación del span óptimo.** En inferencia, las dos distribuciones son independientes, así que hay que combinarlas respetando una restricción de validez: el fin no puede ir antes que el inicio. Se elige el par $(k, l)$ que maximiza el producto de probabilidades sujeto a $k \le l$:

$$ (k^*, l^*) = \arg\max_{k \le l}\; p^{\text{start}}_k \cdot p^{\text{end}}_l $$

(En práctica se acota también la longitud máxima del span y se trabaja en log-espacio: $\arg\max\, [\log p^{\text{start}}_k + \log p^{\text{end}}_l]$.) Esta búsqueda se resuelve en **tiempo lineal con programación dinámica**, recorriendo el pasaje y manteniendo el mejor inicio visto hasta cada posición. Para tareas cloze de una sola palabra (CNN/DailyMail) se omite la predicción de fin y se predice solo el inicio.

{{< concept-alert type="advertencia" >}}
El análisis de errores de BiDAF reveló que el **50% de sus fallos en SQuAD eran de frontera del span** —predecir "1 to 7" cuando la respuesta era "articles 1 to 7". La granularidad exacta del span es un problema persistente: el modelo localiza la región correcta pero yerra los límites por uno o dos tokens.
{{< /concept-alert >}}

---

## 6. El rol de la atención en MRC

La [atención](/fundamentos/mecanismo-atencion) no es un detalle de implementación en MRC: es **el mecanismo central**. La razón es estructural. Responder una pregunta sobre un pasaje requiere alinear la pregunta con la parte del pasaje que la responde —y eso es exactamente lo que computa la atención: una distribución $\alpha$ sobre los tokens del pasaje que codifica "cuánto importa cada palabra para esta pregunta".

Toda la genealogía de la Sección 4 puede leerse como una historia de **qué tan rica es la atención**:

- Attentive Reader: la pregunta atiende al pasaje, una dirección, resumen en un vector.
- Stanford AR: la misma idea con un término bilineal que aprende el alineamiento.
- BiDAF: atención bidireccional (pregunta↔pasaje) sin resumir.
- BERT: self-attention densa, multi-cabeza, multi-capa, pre-entrenada.

**Interpretabilidad.** Una de las virtudes didácticas de la atención en MRC es que es **visualizable**. La matriz de atención de BiDAF (una fila por palabra de la pregunta, una columna por palabra del contexto) puede graficarse como un heatmap y leerse directamente: en los ejemplos del paper, "Where" se ilumina sobre ubicaciones (Stadium, Levi's, Santa), "many" sobre cantidades (hundreds, 15, 13), y las entidades de la pregunta atienden a las mismas entidades del contexto (Super Bowl → Super Bowl). Estos heatmaps son evidencia de que el modelo aprendió a alinear tipos de pregunta con tipos de respuesta —y prefiguraron los embeddings contextuales tipo ELMo (la capa BiLSTM de BiDAF ya desambiguaba "May" mes vs. verbo modal).

---

## 7. Datasets de MRC

El campo avanza al ritmo de sus benchmarks. Los canónicos:

| Dataset | Año | Formulación | Tamaño | Característica |
| --- | --- | --- | --- | --- |
| **CNN/DailyMail** | 2015 | Cloze | ~1.3M | Entidades anonimizadas, construido por scraping de bullets editoriales |
| **SQuAD 1.1** | 2016 | Span extraction | 100K+ | Preguntas humanas sobre Wikipedia, respuesta = span garantizado |
| **SQuAD 2.0** | 2018 | Span + unanswerable | 150K+ | Añade preguntas sin respuesta en el pasaje |
| **MS MARCO** | 2016 | Free-form / generative | 1M+ | Consultas reales de Bing, respuestas abstractivas |

**[CNN/DailyMail](/papers/cnn-dailymail-hermann-2015)** (Hermann 2015) fue el primer dataset masivo de MRC, lo que hizo viable entrenar redes profundas. Su construcción cloze con anonimización de entidades fue ingeniosa pero —como mostró Chen 2016— produjo un task demasiado fácil.

**[SQuAD](/papers/squad-rajpurkar-2016)** (Rajpurkar 2016) fue la respuesta: 100.000+ preguntas escritas por humanos sobre artículos de Wikipedia, con respuestas que son spans arbitrarios de texto. Se evalúa con **Exact Match (EM)** y **F1** a nivel de token (ver [métricas de QA](/fundamentos/qa-evaluation-metrics)). Dominó la investigación de MRC durante años y su leaderboard fue el campo de batalla competitivo.

**SQuAD 2.0** (Rajpurkar 2018) añadió un giro crucial: preguntas **sin respuesta** en el pasaje. El modelo debe aprender a abstenerse ("no answer") en vez de adivinar siempre un span —un test de robustez que SQuAD 1.1 no medía.

**MS MARCO** (Bajaj 2016) usa consultas reales de motor de búsqueda y respuestas generativas, acercándose al QA realista y a la formulación free-form.

---

## 8. Limitaciones y desafíos

MRC parece resuelto en los benchmarks clásicos —pero esa misma facilidad encierra las advertencias más importantes del área.

**Comprensión vs. pattern matching (la crítica de Chen 2016).** El hallazgo más influyente sobre MRC es que **rendir bien en un benchmark no implica comprender**. Chen, Bolton y Manning auditaron a mano 100 ejemplos de CNN/DailyMail y encontraron que ~25% eran ruido (errores de correferencia o casos ambiguos no respondibles) y que **solo 2 de 100 requerían razonar sobre múltiples oraciones**: la inmensa mayoría se resolvía identificando una sola oración relevante. El task se parecía más a **extracción de relaciones de una oración** que a comprensión de discurso. Con un techo realista de ~75% (por el ruido), sus modelos ya estaban ahí —el dataset estaba esencialmente resuelto, no por razonamiento profundo sino por matching superficial robusto a paráfrasis.

**Razonamiento multi-hop.** Muchas preguntas reales requieren combinar información de **varias oraciones o documentos** ("¿en qué ciudad nació el director de la película que ganó el Óscar en 1994?"). Los modelos clásicos —y muchos LLMs— tienden a hacer atajos en vez de razonar en cadena. Datasets como HotpotQA fueron diseñados específicamente para forzar multi-hop.

**Unanswerable questions.** SQuAD 1.1 garantizaba que toda pregunta tenía respuesta en el pasaje, lo que entrenaba modelos a **siempre adivinar un span**. SQuAD 2.0 expuso lo frágil que era esto: saber cuándo *no* hay respuesta es parte de comprender.

**Robustez adversarial.** Jia & Liang (2017) mostraron que **insertar una oración distractora** gramaticalmente correcta pero irrelevante al final del pasaje hacía colapsar la accuracy de los modelos SQuAD de la época. Esto reveló que dependían de matching de palabras superficial, no de comprensión —el problema "Clever Hans" del NLP.

Estas críticas catalizaron la cultura de **dataset auditing**: no aceptar un benchmark al pie de la letra, sino preguntarse qué mide realmente.

---

## 9. MRC en la era de los LLMs

La llegada de los LLMs reconfiguró MRC más que ninguna otra tarea de NLP.

**Long-context.** El cuello de botella histórico de MRC era el límite de contexto: BiDAF y BERT topaban en cientos o pocos miles de tokens, obligando a truncar o segmentar pasajes largos. Los LLMs modernos manejan ventanas de cientos de miles de tokens, permitiendo MRC sobre libros enteros, contratos completos o historias clínicas largas sin segmentación.

**Retrieval-Augmented Generation (RAG).** El paradigma dominante hoy combina lo mejor del open-domain QA y del MRC: un *retriever* localiza los pasajes relevantes en una colección gigante y se los inyecta como contexto a un LLM generativo, que hace la comprensión y produce la respuesta. RAG es, en el fondo, **MRC generativo donde el pasaje se recupera dinámicamente** en vez de venir dado. Es la arquitectura de la mayoría de los asistentes documentales actuales.

**In-context QA.** Con [in-context learning](/fundamentos/in-context-learning), un LLM instruction-tuned hace MRC zero-shot vía prompt —simplemente pegando el pasaje y la pregunta— sin fine-tuning específico. La formulación volvió a ser **free-form generativa**: el modelo parafrasea, sintetiza y razona sobre la evidencia en lenguaje natural, cerrando el círculo desde la predicción de spans de vuelta hacia la generación abierta. El reto que persiste es el mismo de siempre, ahora amplificado: distinguir comprensión genuina de pattern matching, y garantizar **faithfulness** —que la respuesta esté soportada por el pasaje y no alucinada.

---

## 10. Conexión con el curso

MRC integra varios hilos del curso IA UC y es el tema central de la **[Clase 24](/clases/clase-24)**.

- **[Clase 24 (Reading Comprehension)](/clases/clase-24)**: la clase principal de este fundamento. Presenta el Stanford Attentive Reader (slides 23-27) y BiDAF (slides 29-31) como los dos hitos de MRC con atención pre-Transformer.
- **[Question Answering](/fundamentos/question-answering)**: MRC es la instancia de QA con contexto dado. El fundamento de QA cubre el espectro completo (factoid, open-domain, MRC).
- **[Métricas de evaluación de QA](/fundamentos/qa-evaluation-metrics)**: Exact Match y F1, las métricas de span extraction usadas en SQuAD.
- **[Mecanismo de atención](/fundamentos/mecanismo-atencion)**: el mecanismo central de MRC, desde la atención bilineal hasta la bidireccional.
- **[Self-attention](/fundamentos/self-attention)** y **[Transformer](/fundamentos/transformer)**: la generalización que BERT trajo a MRC.
- **[BERT](/fundamentos/bert)**: el pre-entrenamiento que volvió obsoletas las arquitecturas de QA a medida.

La narrativa pedagógica que MRC ofrece es limpia: de **comprimir la pregunta en un vector** (Attentive Reader) a **dejar fluir representaciones por token en ambas direcciones** (BiDAF) a **pre-entrenar self-attention a gran escala** (BERT) y finalmente a **generar la respuesta** (T5/BART, LLMs). Es el puente conceptual entre la era pre-Transformer y la era de los modelos pre-entrenados, y muestra por qué los benchmarks tuvieron que volverse más difíciles para seguir siendo informativos.

---

## Recursos relacionados

### Fundamentos

- [Question Answering](/fundamentos/question-answering) — la tarea madre de la que MRC es instancia.
- [Métricas de evaluación de QA](/fundamentos/qa-evaluation-metrics) — Exact Match y F1.
- [Mecanismo de atención](/fundamentos/mecanismo-atencion) — el mecanismo central de MRC.
- [Self-attention](/fundamentos/self-attention) — la atención que BERT lleva a MRC.
- [BERT](/fundamentos/bert) — encoder pre-entrenado para QA.
- [Transformer](/fundamentos/transformer) — backbone arquitectural.
- [Seq2seq](/fundamentos/seq2seq) — encoder-decoder para MRC generativo.

### Papers

- [CNN/DailyMail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015) — primer dataset masivo de MRC y el Attentive Reader.
- [Stanford Attentive Reader (Chen 2016)](/papers/stanford-attentive-reader-chen-2016) — atención bilineal y la auditoría crítica del benchmark.
- [BiDAF (Seo 2017)](/papers/bidaf-seo-2017) — atención bidireccional y attention flow.
- [SQuAD (Rajpurkar 2016)](/papers/squad-rajpurkar-2016) — el benchmark de span extraction.
- [BERT (Devlin 2018)](/papers/bert-devlin-2018) — pre-entrenamiento que dominó MRC.

### Clases

- [Clase 24 (Reading Comprehension)](/clases/clase-24) — la clase principal.

*Última actualización: 2026-06-07.*
