# A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task

> Análisis técnico exhaustivo para el curso IA UC (Diplomado IA, PUC Chile) — Clase 24, Reading Comprehension. El paper introduce el *Stanford Attentive Reader* y, a la vez, demuestra empíricamente que el dataset CNN/Daily Mail era mucho más fácil de lo que se creía.

## 1. Metadata

| Campo | Valor |
|---|---|
| Título | A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task |
| Autores | Danqi Chen, Jason Bolton, Christopher D. Manning |
| Afiliación | Computer Science Department, Stanford University (Stanford NLP Group) |
| Venue | ACL 2016 (54th Annual Meeting of the Association for Computational Linguistics) |
| Preprint | arXiv:1606.02858v2 [cs.CL], 8 de agosto de 2016 |
| Código | https://github.com/danqi/rc-cnn-dailymail |
| Financiamiento | DARPA DEFT Program, AFRL contrato FA8750-13-2-0040 |
| Resultados clave | 73.6% (CNN) y 76.6% (Daily Mail), superando el estado del arte previo por 7–10% |

El paper tiene una estructura inusual para un trabajo de ACL: combina una contribución de modelado (un sistema neuronal nuevo y mejor) con una contribución de *análisis de datos* (una auditoría manual del benchmark). Esta dualidad es lo que lo hizo influyente: no solo movió la aguja en accuracy, sino que cuestionó la validez misma del task que estaba midiendo.

## 2. Contexto: el dataset CNN/Daily Mail de Hermann et al. (2015)

El problema de fondo que motiva todo el trabajo es la escasez de datos anotados para reading comprehension (RC). Antes de 2015, los datasets de RC supervisado —MCTest (Richardson et al., 2013), los de Berant et al. (2014) y Wang et al. (2015)— consistían en apenas cientos de documentos. La anotación requería expertise y diseño cuidadoso, lo que la encarecía enormemente. Con solo cientos de ejemplos es imposible entrenar modelos estadísticos potentes, en particular redes profundas, que es precisamente donde uno esperaría capturar razonamiento textual complejo.

Hermann et al. (2015), investigadores de DeepMind, tuvieron una idea ingeniosa para romper este cuello de botella: explotar el hecho de que los artículos de noticias de CNN y Daily Mail vienen acompañados de *bullet points* que los resumen. La construcción del dataset funciona así:

- Cada item es una tripleta `(passage p, question q, answer a)`.
- El **passage** es el artículo de noticias completo.
- La **question** es una tarea *cloze* (de completar): se toma uno de los bullet points del artículo y se reemplaza una entidad por un marcador `@placeholder`.
- El **answer** es la entidad que fue eliminada.

La intuición es que un bullet point resume uno o varios aspectos del artículo, así que si el computador entiende el contenido del artículo debería poder inferir la entidad faltante. Es una forma barata de crear datos supervisados a escala masiva: el resultado son **380.298 ejemplos de entrenamiento para CNN y 879.450 para Daily Mail**.

### La anonimización de entidades

El detalle de diseño más importante —y el que después resultaría problemático— es la **anonimización de entidades**. El texto pasa por un pipeline de NLP de Google: se tokeniza, se pasa a minúsculas, y se corre reconocimiento de entidades nombradas (NER) y resolución de correferencia. Para cada cadena de correferencia que contenga al menos una entidad nombrada, todos los elementos de la cadena se reemplazan por un marcador `@entityn`, con un índice `n` distinto por cadena.

Hermann et al. argumentan, de forma convincente, que esta estrategia es *necesaria*: obliga al sistema a entender el passage que tiene delante, en lugar de usar conocimiento del mundo o un simple modelo de lenguaje para adivinar la respuesta sin entender el texto. Si las entidades estuvieran sin anonimizar, un modelo podría responder "Obama" a una pregunta sobre política estadounidense simplemente porque es estadísticamente probable, sin haber leído nada.

Pero la anonimización tiene un costo doble que Chen et al. explotan en su crítica:

1. **Los sistemas se benefician** de que el NER y la correferencia ya estén hechos: les llega trabajo gratis.
2. **Los sistemas sufren** cuando esos módulos fallan, porque el error queda "horneado" en los datos. En el ejemplo de la Figura 1 del paper, "the character" debería ser correferente con `@entity14`, pero el pipeline no lo capturó.

Además, la anonimización vuelve la tarea más difícil incluso *para humanos*: a veces es imposible determinar la respuesta correcta cuando todo está reemplazado por marcadores abstractos sin contenido semántico.

### Estadísticas del dataset

| Métrica | CNN | Daily Mail |
|---|---|---|
| # Train | 380.298 | 879.450 |
| # Dev | 3.924 | 64.835 |
| # Test | 3.198 | 53.182 |
| Passage: avg. tokens | 761.8 | 813.1 |
| Passage: avg. sentences | 32.3 | 28.9 |
| Question: avg. tokens | 12.5 | 14.3 |
| Avg. # entities | 26.2 | 26.2 |

Los passages son largos —alrededor de 30 oraciones y 800 tokens— y cada pregunta tiene unos 12–14 tokens. En promedio hay 26 entidades candidatas por ejemplo, así que el espacio de respuestas no es trivial: un clasificador aleatorio acertaría cerca del 4%.

### Por qué hacía falta un examen riguroso

El planteamiento de Hermann era atractivo, pero quedaban dos preguntas abiertas que nadie había respondido con rigor: ¿qué nivel de reading comprehension se necesita *realmente* para resolver esta tarea algo artificial? Y ¿qué han aprendido de verdad los modelos que rinden bien en ella? Al momento de escribir, solo había dos papers con resultados sobre este task —Hermann et al. (2015) y Hill et al. (2016)—, ambos con redes neuronales, y ninguno había auditado el dataset a mano. Chen et al. se proponen exactamente eso.

## 3. La doble contribución

El paper tiene dos contribuciones entrelazadas:

**(a) Un modelo más simple y mejor.** Chen et al. construyen el *Stanford Attentive Reader* (lo llaman simplemente "Neural net" en sus tablas), una variante simplificada del Attentive Reader de Hermann. Pese a tener menos componentes, obtiene **73.6% en CNN y 76.6% en Daily Mail**, superando el estado del arte previo por 7–10%. La lección de modelado: la atención bilineal simple supera al mecanismo de atención más elaborado del original.

**(b) Un análisis manual que muestra que el dataset es más fácil de lo creído.** Chen et al. muestrean 100 ejemplos del dev de CNN y los clasifican a mano según el tipo de razonamiento que exigen. El hallazgo demoledor: cerca del **25% de los ejemplos son ruido** (errores de correferencia o casos ambiguos/imposibles), y de los ejemplos *respondibles*, la inmensa mayoría se resuelve identificando una sola oración relevante. Solo 2 de 100 requieren razonar sobre múltiples oraciones. El techo realista de performance está alrededor de **75%**, y sus sistemas ya están prácticamente ahí.

Las dos contribuciones se refuerzan: el modelo establece un *lower bound* fuerte de lo que se puede lograr, y el análisis manual establece un *upper bound* del techo del dataset. Cuando ambos casi coinciden, la conclusión es ineludible: el task está esencialmente resuelto.

Las conclusiones que el propio paper enumera son: (i) el dataset es más fácil de lo que se creía; (ii) sistemas NLP convencionales rinden mucho mejor de lo sugerido; (iii) las representaciones distribuidas del deep learning son muy efectivas reconociendo paráfrasis; (iv) por la naturaleza de las preguntas, los sistemas actuales son más extractores de relaciones de una sola oración que entendedores de discurso amplio; (v) los sistemas presentados están cerca del techo para los casos de una oración y no ambiguos; y (vi) las perspectivas de acertar el 20% final son pobres, porque la mayoría involucra problemas de preparación de datos.

## 4. Arquitectura del Stanford Attentive Reader

El modelo neuronal se basa en el AttentiveReader de Hermann et al. (2015) y se describe en tres pasos. Dada la tripleta `(p, q, a)` con `p = {p₁,...,pₘ}` y `q = {q₁,...,qₗ}` como secuencias de tokens del passage y la pregunta, el objetivo es inferir la entidad correcta $a \in p \cap E$ que corresponde al placeholder, donde $E$ es el conjunto de todos los marcadores de entidad abstractos. Nótese la restricción dura: **la respuesta correcta siempre aparece en el passage**.

### Paso 1: Encoding

Primero, todas las palabras se mapean a vectores $d$-dimensionales mediante una matriz de embeddings $E \in \mathbb{R}^{d \times |V|}$. Así obtenemos $p_1, \ldots, p_m \in \mathbb{R}^d$ y $q_1, \ldots, q_l \in \mathbb{R}^d$.

Luego se usa una RNN bidireccional poco profunda con tamaño oculto $\tilde{h}$ para codificar los embeddings contextuales $\tilde{p}_i$ de cada palabra del passage:

$$\overrightarrow{h}_i = \mathrm{RNN}(\overrightarrow{h}_{i-1}, p_i), \quad i = 1, \ldots, m$$

$$\overleftarrow{h}_i = \mathrm{RNN}(\overleftarrow{h}_{i+1}, p_i), \quad i = m, \ldots, 1$$

y se concatenan las dos direcciones:

$$\tilde{p}_i = \mathrm{concat}(\overrightarrow{h}_i, \overleftarrow{h}_i) \in \mathbb{R}^h, \quad h = 2\tilde{h}$$

Una segunda RNN bidireccional mapea la pregunta $q_1, \ldots, q_l$ a un único embedding $q \in \mathbb{R}^h$ (típicamente concatenando los estados finales de ambas direcciones). La celda recurrente elegida es la **GRU** (Cho et al., 2014), no la LSTM, porque rinde de forma similar pero es computacionalmente más barata.

### Paso 2: Attention (atención bilineal)

El objetivo es comparar el embedding de la pregunta con todos los embeddings contextuales del passage y seleccionar la información relevante. Se calcula una distribución de probabilidad $\alpha$ según el grado de relevancia entre cada palabra $\tilde{p}_i$ (en su contexto) y la pregunta $q$:

$$\alpha_i = \mathrm{softmax}_i\, q^\top W_s \tilde{p}_i$$

$$o = \sum_i \alpha_i \tilde{p}_i$$

Aquí $W_s \in \mathbb{R}^{h \times h}$ es el **término bilineal**, que permite calcular una similitud entre $q$ y $\tilde{p}_i$ de forma más flexible que con un simple producto punto. El producto punto $q^\top \tilde{p}_i$ obligaría a comparar las dimensiones una a una; el término bilineal $q^\top W_s \tilde{p}_i$ aprende una transformación que alinea el espacio de la pregunta con el del passage antes de medir similitud. El softmax sobre $i$ convierte estos scores en pesos de atención que suman 1, y $o$ es la combinación ponderada de todos los embeddings contextuales: un vector que resume "qué parte del passage importa para esta pregunta".

### Paso 3: Predicción

Usando el vector de salida $o$, el sistema predice la respuesta más probable:

$$a = \arg\max_{a \in p \cap E}\, W_a^\top o$$

Concretamente, se aplica un softmax sobre $W_a^\top o$ restringido a las entidades candidatas y se entrena con un objetivo de log-verosimilitud negativa (negative log-likelihood). La restricción $a \in p \cap E$ es clave: solo se compite entre las entidades que aparecen en el passage, no sobre todo el vocabulario.

### Comparación con el Attentive Reader original de Hermann

El modelo "básicamente sigue" al AttentiveReader, pero introduce tres diferencias. Para sorpresa de los propios autores, observan una mejora de **7–10%** sobre el original:

1. **Término bilineal en lugar de capa tanh.** El original calculaba la relevancia atención-pregunta con una capa $\tanh$ (una MLP con no linealidad). Chen et al. la reemplazan por el término bilineal $q^\top W_s \tilde{p}_i$. La efectividad de esta atención bilineal simple ya había sido demostrada por Luong et al. (2015) para traducción automática neuronal. **De las tres diferencias, esta es la única que los autores consideran realmente importante.**

2. **Uso directo de $o$ para predecir.** Tras obtener los embeddings contextuales ponderados $o$, Chen et al. predicen directamente. El modelo original combinaba $o$ y el embedding de la pregunta $q$ mediante *otra* capa no lineal antes de la predicción final. Encontraron que podían eliminar esa capa sin perder performance, bajo la idea de que basta con que el modelo aprenda a devolver la entidad a la que da máxima atención.

3. **Predicción restringida a entidades del passage.** El original consideraba *todas* las palabras del vocabulario $V$ al predecir. Chen et al. lo juzgan innecesario y solo predicen entre las entidades que aparecen en el passage.

Los autores son explícitos: de los tres cambios, solo el primero (bilineal) parece importante; los otros dos solo buscan mantener el modelo simple. Esta honestidad es parte del valor del paper —no inflan la contribución, sino que aíslan qué importa de verdad.

### Comparación con MemN2N (Hill et al., 2016)

El paper también discute el enfoque de Hill et al. (2016), basado en memory networks (Weston et al., 2015), al que consideran "altamente similar en espíritu". La mayor diferencia está en cómo codifican el passage: usan solo una **ventana de 5 palabras** alrededor de cada entidad candidata, con un enfoque de *positional unigram*. Si una ventana tiene 5 palabras $x_1, \ldots, x_5$, se codifica como $\sum_{i=1}^{5} E_i(x_i)$, con 5 matrices de embedding separadas que aprender. Codifican la ventana de 5 palabras alrededor del placeholder de forma similar e ignoran el resto de la pregunta, usando un producto punto para la relevancia. Que este modelo tan local funcione bien es, en sí mismo, otra señal de cuánto del task se resuelve con matching de contexto puramente local.

## 5. Resultados

### Detalles de entrenamiento

Para la red neuronal: vocabulario $|V| = 50\mathrm{k}$ palabras más frecuentes (incluyendo marcadores de entidad y placeholder), resto mapeado a `<unk>`. Embeddings de dimensión $d = 100$ inicializados con **GloVe** preentrenado de 100 dimensiones (Pennington et al., 2014). Parámetros de atención y salida inicializados uniformemente en $(-0.01, 0.01)$; pesos de la GRU desde $\mathcal{N}(0, 0.1)$. Tamaño oculto $h = 128$ para CNN y $256$ para Daily Mail. SGD vanilla con learning rate fijo de $0.1$, mini-batch de 32 (ejemplos ordenados por largo de passage), dropout de $0.2$ en la capa de embeddings, y gradient clipping cuando la norma supera 10. Hasta 30 epochs, seleccionando el mejor en dev. Cada modelo se corre 5 veces con semillas distintas y se reporta el promedio; también ensembles que promedian las probabilidades de los 5.

Adicionalmente, los autores notan que los índices de los marcadores de entidad se generan arbitrariamente, así que prueban **relabelar** los marcadores según su primera aparición en el passage y la pregunta (la primera entidad pasa a `@entity1`, la segunda a `@entity2`, etc.). Esto hace que el entrenamiento converja más rápido y trae ganancias leves.

El runtime fue de unas 3 horas por epoch en CNN y 12 horas por epoch en Daily Mail, en una sola GPU GeForce GTX TITAN X.

### Tabla de resultados principales

| Modelo | CNN Dev | CNN Test | DM Dev | DM Test |
|---|---|---|---|---|
| Frame-semantic model † | 36.3 | 40.2 | 35.5 | 35.5 |
| Word distance model † | 50.5 | 50.9 | 56.4 | 55.5 |
| Deep LSTM Reader † | 55.0 | 57.0 | 63.3 | 62.2 |
| Attentive Reader † | 61.6 | 63.0 | 70.5 | 69.0 |
| Impatient Reader † | 61.8 | 63.8 | 69.0 | 68.0 |
| MemNNs (window memory) ‡ | 58.0 | 60.6 | N/A | N/A |
| MemNNs (window + self-sup.) ‡ | 63.4 | 66.8 | N/A | N/A |
| MemNNs (ensemble) ‡ | 66.2* | 69.4* | N/A | N/A |
| **Ours: Classifier** | 67.1 | 67.9 | 69.1 | 68.3 |
| **Ours: Neural net** | 72.5 | 72.7 | 76.9 | 76.0 |
| **Ours: Neural net (ensemble)** | 76.2* | 76.5* | 79.5* | 78.7* |
| **Ours: Neural net (relabeling)** | 73.8 | **73.6** | 77.6 | **76.6** |
| **Ours: Neural net (relabeling, ensemble)** | 77.2* | 77.6* | 80.2* | 79.2* |

(† = de Hermann et al. 2015; ‡ = de Hill et al. 2016; * = ensemble)

Observaciones:

- El **clasificador convencional** obtiene 67.9% en CNN test. No solo supera todos los enfoques simbólicos de Hermann et al., sino también todos sus sistemas neuronales y el mejor resultado de single-system de Hill et al. (2016). Que un clasificador de features supere a las redes neuronales del paper original ya es una señal fuerte de que el task no era tan difícil.
- El **modelo neuronal single** supera los resultados previos por un amplio margen (más de 5%). El relabeling agrega 0.6% (CNN) y 0.9% (Daily Mail), llevando el estado del arte a 73.6% y 76.6%.
- Los **ensembles** de 5 modelos consistentemente agregan 2–4% adicionales.

Concurrentemente, Kadlec et al. (2016) y Kobayashi et al. (2016) también experimentaron sobre estos datasets con resultados competitivos, pero el modelo de Chen et al. los supera y es estructuralmente más simple. Todos estos esfuerzos convergen a números similares, lo que refuerza la hipótesis del techo.

## 6. Análisis manual de 100 ejemplos

El corazón del paper. Chen et al. muestrean uniformemente 100 ejemplos del dev de CNN y los clasifican a mano. Si un ejemplo satisface más de una categoría, lo asignan a la primera (más fácil). La taxonomía:

- **Exact match:** las palabras alrededor del placeholder también aparecen en el passage alrededor de un marcador de entidad; la respuesta es autoevidente.
- **Sentence-level paraphrasing:** el texto de la pregunta está entailment/parafraseado por *exactamente una* oración del passage, así que la respuesta se identifica definitivamente desde esa oración.
- **Partial clue:** no hay un match semántico completo, pero se infiere la respuesta por pistas parciales (solapamiento de palabras/conceptos).
- **Multiple sentences:** requiere procesar múltiples oraciones para inferir la respuesta.
- **Coreference errors:** ejemplos con errores críticos de correferencia para la entidad respuesta o entidades clave de la pregunta. Se tratan básicamente como "no respondibles".
- **Ambiguous / very hard:** casos donde los autores creen que ni un humano podría obtener la respuesta correcta con confianza.

### Tabla de breakdown

| No. | Categoría | (%) |
|---|---|---|
| 1 | Exact match | 13 |
| 2 | Paraphrasing | 41 |
| 3 | Partial clue | 19 |
| 4 | Multiple sentences | 2 |
| 5 | Coreference errors | 8 |
| 6 | Ambiguous / hard | 17 |

### Conclusiones del análisis

Dos hallazgos sorprenden a los autores:

1. **"Coreference errors" + "ambiguous/hard" = 25%.** Una cuarta parte del sample es esencialmente ruido no respondible (salvo por suerte). Esto pone una barrera dura: entrenar un modelo mucho por encima de **75%** de accuracy es prácticamente imposible, porque el 25% restante está corrupto en origen.

2. **Solo 2 de 100 ejemplos requieren múltiples oraciones.** Esto es mucho menos de lo que Hermann et al. sugerían. La hipótesis que emerge: en la mayoría de los casos *respondibles*, la tarea se reduce a identificar la oración única más relevante y luego inferir la respuesta a partir de ella. Es decir, el task se parece mucho más a **extracción de relaciones de una sola oración** que a comprensión de discurso amplio.

### Performance por categoría

| Categoría | Classifier | Neural net |
|---|---|---|
| Exact match | 13 (100.0%) | 13 (100.0%) |
| Paraphrasing | 32 (78.1%) | 39 (95.1%) |
| Partial clue | 14 (73.7%) | 17 (89.5%) |
| Multiple sentences | 1 (50.0%) | 1 (50.0%) |
| Coreference errors | 4 (50.0%) | 3 (37.5%) |
| Ambiguous / hard | 2 (11.8%) | 1 (5.9%) |
| **All** | **66 (66.0%)** | **74 (74.0%)** |

Esta tabla es quizás la más reveladora del paper:

- **Exact match:** ambos sistemas aciertan el 100%. Casos triviales.
- **Ambiguous/hard y coreference errors:** ambos rinden mal, como se esperaba. Aquí no hay nada que aprender.
- **La diferencia entre los dos sistemas está casi enteramente en paraphrasing y partial clue.** La red neuronal sube de 78.1% a 95.1% en paráfrasis y de 73.7% a 89.5% en pistas parciales. Esto muestra de forma limpia que **las redes neuronales son mejores aprendiendo matches semánticos que involucran paráfrasis o variación léxica** entre dos oraciones. Es el aporte real del deep learning aquí: no razonamiento complejo, sino robustez ante reformulaciones.
- Los autores concluyen que la red ya logra performance *casi óptima* en todos los casos de una sola oración y no ambiguos. No queda headroom útil para enfoques de comprensión más sofisticados sobre este dataset.

## 7. Implicancias: el dataset estaba esencialmente resuelto

Cuando el techo realista es ~75% (por el 25% de ruido) y el mejor sistema single ya llega a 73.6–76.6%, la conclusión es ineludible: **el dataset CNN/Daily Mail estaba esencialmente resuelto**. No por razonamiento profundo, sino porque (a) un cuarto de los ejemplos es ruido inalcanzable y (b) la mayoría del resto se resuelve con matching de una sola oración robusto a paráfrasis.

Esto tiene una implicancia metodológica enorme para la comunidad. El dataset había sido celebrado como un paso hacia "enseñar a las máquinas a leer y comprender", pero Chen et al. muestran que el nivel de razonamiento e inferencia requerido es "todavía bastante simple". El mensaje implícito es que se necesitan benchmarks más difíciles, con:

- Menos ruido en la construcción (sin errores de correferencia horneados).
- Preguntas que genuinamente requieran integrar múltiples oraciones.
- Respuestas que no se reduzcan a seleccionar entre entidades anonimizadas.

Este paper es uno de los catalizadores intelectuales detrás de **SQuAD** (Rajpurkar et al., 2016, que aparece el mismo año): un dataset construido por crowdsourcing humano, con respuestas que son *spans* de texto arbitrarios (no entidades de un conjunto cerrado), preguntas escritas por personas, y un protocolo de evaluación más limpio. La auditoría de Chen et al. ayudó a justificar por qué la comunidad necesitaba dar ese salto.

## 8. El clasificador entity-centric basado en features

Antes de la red, Chen et al. construyen deliberadamente un clasificador convencional, en el espíritu de Wang et al. (2015). La motivación es metodológica: sospechan que los baselines de Hermann et al. eran débiles. En particular, Hermann usó un frame-semantic parser cuya pobre cobertura, argumentan, subestima lo que un sistema NLP estándar de los últimos 15 años (basado en QA factoide y extracción de relaciones) puede lograr. De hecho, el modelo frame-semantic de Hermann era marcadamente inferior a su propio baseline de distancia de palabras.

El setup es un problema de **ranking**: se diseña un vector de features $f_{p,q}(e)$ para cada entidad candidata $e$ y se aprende un vector de pesos $\theta$ tal que la respuesta correcta $a$ rankee más alto que las demás candidatas:

$$\theta^\top f_{p,q}(a) > \theta^\top f_{p,q}(e), \quad \forall e \in E \cap p \setminus \{a\}$$

Las 8 plantillas de features:

1. Si la entidad $e$ aparece en el passage.
2. Si $e$ aparece en la pregunta.
3. La frecuencia de $e$ en el passage.
4. La primera posición de aparición de $e$ en el passage.
5. **n-gram exact match:** si hay match exacto entre el texto que rodea al placeholder y el texto que rodea a $e$. Hay features para todas las combinaciones de match a izquierda y/o derecha, de una o dos palabras.
6. **Word distance:** alinea el placeholder con cada ocurrencia de $e$ y computa la distancia mínima promedio de cada palabra no-stopword de la pregunta a la entidad en el passage.
7. **Sentence co-occurrence:** si $e$ co-ocurre con otra entidad o verbo de la pregunta, en alguna oración del passage.
8. **Dependency parse match:** se parsea por dependencias la pregunta y todas las oraciones del passage, y se extrae un indicador de si $w \xrightarrow{r} @placeholder$ y $w \xrightarrow{r} e$ aparecen ambos (y simétricamente para $@placeholder \xrightarrow{r} w$ y $e \xrightarrow{r} w$).

Para entrenar usan **LambdaMART** (Wu et al., 2010) de RankLib —forests de árboles de decisión boosteados, exitosos en competencias de Kaggle—, scoreando 1/0 sobre la primera propuesta rankeada. El parser de dependencias es el neuronal de Stanford (Chen y Manning, 2014).

### Ablación de features

| Features | Accuracy |
|---|---|
| Full model | 67.1 |
| − whether e is in the passage | 67.1 |
| − whether e is in the question | 67.0 |
| − frequency of e | 63.7 |
| − position of e | 65.9 |
| − **n-gram match** | **60.5** |
| − word distance | 65.4 |
| − sentence co-occurrence | 66.0 |
| − dependency parse match | 65.6 |

La tabla muestra accuracy *tras quitar* cada feature, así que un número bajo indica una feature importante. Las dos clases más importantes son **n-gram match** (quitarla baja a 60.5%) y **frequency of entity** (baja a 63.7%). Esto cuadra perfecto con el análisis manual: si el task se resuelve mayormente por matching local de una oración, las features que capturan ese matching superficial son las que mandan. El clasificador llega a **67.9% en CNN test** con features puramente superficiales, lo que en sí mismo es un argumento contundente de que el task no exige razonamiento profundo.

## 9. Limitaciones del propio trabajo

El paper es metodológicamente honesto, pero conviene marcar sus límites:

- **El análisis manual es de 100 ejemplos, solo de CNN.** No se auditó Daily Mail a mano. Con $n=100$, los porcentajes tienen intervalos de confianza amplios (el 25% de ruido podría razonablemente estar entre ~17% y ~34%). Los autores mitigan esto publicando los índices exactos de su muestra (Tablas 7 y 8 del apéndice) para reproducibilidad, pero la muestra sigue siendo pequeña.
- **La taxonomía tiene juicio subjetivo.** Distinguir "paraphrasing" de "partial clue", o decidir qué es "ambiguous/hard", depende del criterio del anotador. No se reporta acuerdo inter-anotador.
- **El techo de ~75% es una estimación, no una cota dura.** Los propios autores reconocen que un modelo "a veces puede adivinar con suerte", así que un sistema podría superar el 75% sin entender realmente el 25% ruidoso.
- **El modelo no aporta innovación arquitectónica profunda.** Su valor está en la simplificación y el rigor empírico, no en un mecanismo nuevo —la atención bilineal venía de Luong et al. (2015). El paper no lo oculta, pero conviene tenerlo claro.
- **Las conclusiones son específicas de este dataset.** El argumento de que "RC es extracción de relaciones de una oración" aplica a CNN/Daily Mail por cómo fue construido, no a RC en general. La sección 6 (Related Tasks) lo deja claro al contrastar con MCTest (>50% de preguntas requieren múltiples oraciones), Children Book Test y bAbI.

## 10. Impacto

Este paper se volvió un clásico por dos vías independientes.

**Danqi Chen como figura del campo de MRC.** Danqi Chen pasó de este trabajo de 2016 a convertirse en una de las investigadoras más influyentes de machine reading comprehension y open-domain question answering. Su tesis doctoral de Stanford ("Neural Reading Comprehension and Beyond", 2018) es referencia obligada, y su trabajo posterior incluye DrQA (open-domain QA sobre Wikipedia) y contribuciones centrales a dense retrieval. Hoy es profesora en Princeton. Este paper, escrito al inicio de su doctorado bajo Christopher Manning, ya exhibe el estilo que la caracterizaría: rigor empírico, modelos simples y bien entendidos, y disposición a cuestionar supuestos del campo.

**El Stanford Attentive Reader como arquitectura didáctica canónica.** Más allá de su impacto en investigación, la arquitectura se volvió la pieza pedagógica estándar para enseñar reading comprehension con atención. Aparece en **CS224n** (Natural Language Processing with Deep Learning, Stanford), el curso que Manning imparte, precisamente porque tiene el balance ideal: es simple de derivar a mano (encoding bi-GRU + atención bilineal + predicción argmax), captura la idea esencial de la atención query-context, y conecta naturalmente con modelos posteriores como BiDAF y, eventualmente, BERT para QA. Es el "hola mundo" de la comprensión lectora neuronal.

El paper también es citado como ejemplo paradigmático de **dataset auditing**: la práctica de no aceptar un benchmark al pie de la letra, sino auditar manualmente qué mide realmente. Esta cultura crítica —que más tarde produciría trabajos sobre artefactos de anotación, atajos espurios y "Clever Hans" en NLP— tiene en este paper uno de sus antecedentes más limpios.

## 11. Conexión con la Clase 24

En el PDF de la Clase 24 del curso, las **slides 23–27** usan el Stanford Attentive Reader como el modelo central para enseñar Reading Comprehension. El rol pedagógico es exactamente el que el paper habilita: presentar una arquitectura de atención mínima pero completa, que el estudiante pueda seguir ecuación por ecuación.

La matemática de las **slides 26–27** corresponde de forma directa a las ecuaciones (2)–(4) del paper:

- **Slide 26 — atención bilineal:** $\alpha_i = \mathrm{softmax}_i\, q^\top W_s \tilde{p}_i$ y el vector de salida $o = \sum_i \alpha_i \tilde{p}_i$. La clase enfatiza por qué el término bilineal $W_s$ es más expresivo que un producto punto: aprende qué dimensiones de la pregunta deben alinearse con qué dimensiones del passage.
- **Slide 27 — predicción:** $a = \arg\max_{a \in p \cap E}\, W_a^\top o$, con la restricción de que la respuesta debe ser una entidad presente en el passage.

El encadenamiento pedagógico de la clase es: (1) plantear el problema de RC como inferir una entidad faltante; (2) codificar passage y pregunta con bi-GRU; (3) usar atención para que la pregunta "consulte" el passage; (4) predecir restringiéndose a candidatos válidos. El paper de Chen et al. provee tanto la formalización limpia de este pipeline como la lección crítica —el dataset era fácil— que la clase puede usar para motivar la transición histórica hacia SQuAD y, más adelante, hacia modelos preentrenados como BERT aplicados a span extraction. Para Roberto, que viene de modelos contextuales (Clase 20, ELMo/BERT/GPT) y summarization (Clase 22, T5/BERTSum), este paper cierra el arco: muestra el puente entre la era pre-Transformer (bi-GRU + atención task-specific) y la era de fine-tuning, y deja explícito por qué los benchmarks tuvieron que volverse más difíciles para seguir siendo informativos.

## 12. Notas y enlaces

- **Paper:** Chen, D., Bolton, J., Manning, C. D. (2016). *A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task.* ACL 2016. arXiv:1606.02858.
- **Código original:** https://github.com/danqi/rc-cnn-dailymail
- **Dataset CNN/Daily Mail:** https://github.com/deepmind/rc-data (Hermann et al., 2015).
- **Dataset original de Hermann et al. (2015):** "Teaching Machines to Read and Comprehend", NIPS 2015 — introduce el Attentive Reader, Impatient Reader y Deep LSTM Reader.
- **Atención bilineal:** Luong, Pham, Manning (2015), "Effective Approaches to Attention-based Neural Machine Translation", EMNLP — origen del término bilineal usado aquí.
- **GRU:** Cho et al. (2014), "Learning phrase representations using RNN encoder–decoder", EMNLP.
- **GloVe:** Pennington, Socher, Manning (2014), EMNLP — embeddings de inicialización.
- **LambdaMART:** Wu et al. (2010), "Adapting boosting for information retrieval measures" — el ranker del clasificador.
- **MemN2N window-based:** Hill et al. (2016), "The Goldilocks Principle", ICLR — el competidor neuronal contemporáneo.
- **Sucesores naturales:** SQuAD (Rajpurkar et al., 2016) como respuesta directa a la necesidad de benchmarks más difíciles; tesis doctoral de Danqi Chen, "Neural Reading Comprehension and Beyond" (Stanford, 2018).
- **Curso:** Stanford CS224n usa el Stanford Attentive Reader como modelo didáctico canónico de RC con atención.
- **Glosario rápido:** *cloze* = tarea de completar una palabra/entidad faltante; *anonimización de entidades* = reemplazo de cadenas de correferencia por marcadores `@entityn`; *atención bilineal* = score $q^\top W_s \tilde{p}_i$ con matriz aprendida $W_s$; *techo del dataset* = accuracy máxima realista dada la fracción de ejemplos no respondibles (~75% aquí).
