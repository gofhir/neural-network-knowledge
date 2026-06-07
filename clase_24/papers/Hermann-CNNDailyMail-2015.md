# Teaching Machines to Read and Comprehend — Hermann et al. (2015)

Análisis técnico del paper que introdujo el dataset **CNN/Daily Mail** y los modelos **Attentive Reader** e **Impatient Reader**, sentando las bases de la comprensión lectora supervisada a gran escala y de la oleada de datasets de Machine Reading Comprehension (MRC) que vendría después.

## 1. Metadata

| Campo | Valor |
|---|---|
| Título | Teaching Machines to Read and Comprehend |
| Autores | Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, Phil Blunsom |
| Afiliaciones | Google DeepMind (todos); University of Oxford (Kočiský y Blunsom, marcados con ‡) |
| Venue | Advances in Neural Information Processing Systems 28 (NeurIPS / NIPS 2015) |
| Preprint | arXiv:1506.03340 — versión v3, 19 de noviembre de 2015 |
| Categoría arXiv | cs.CL |
| Contacto | `{kmh,tkocisky,etg,lespeholt,wkay,mustafasul,pblunsom}@google.com` |
| Código y datos | `http://www.github.com/deepmind/rc-data/` (script generador, no el corpus por licencia) |
| Contribuciones clave | (1) Metodología para construir corpora supervisados de comprensión lectora a escala de ~1M de ejemplos; (2) anonimización/permutación de entidades para aislar comprensión del pasaje; (3) tres modelos neuronales con atención (Deep LSTM, Attentive, Impatient Reader) |

El paper combina dos contribuciones que históricamente se reportan por separado pero que son inseparables aquí: un **dataset** (CNN/Daily Mail) y una **familia de arquitecturas** (los Readers). Es importante notar que Mustafa Suleyman, cofundador de DeepMind y posteriormente CEO de Microsoft AI, figura como coautor, lo que ubica el trabajo en el núcleo del DeepMind temprano post-adquisición por Google (2014).

## 2. Contexto histórico: por qué faltaban datasets de comprensión lectora

Hacia 2015 el campo de la comprensión lectora automática (machine reading comprehension) estaba atrapado en un cuello de botella de datos. El paper lo plantea con claridad en su introducción: el progreso "desde algoritmos superficiales de recuperación de información tipo bag-of-words hacia máquinas capaces de leer y comprender documentos ha sido lento". Las dos familias de enfoques tradicionales eran:

1. **Gramáticas hechas a mano** (hand-engineered grammars), referencia [1] — sistemas basados en reglas como el de Riloff y Thelen para tests de comprensión lectora.
2. **Extracción de información**: detectar triples predicado-argumento que luego pueden consultarse como una base de datos relacional, referencia [2] (la línea "Machine Reading" de la Universidad de Washington).

El problema central que el paper identifica es doble: *(a)* la ausencia de datasets de entrenamiento a gran escala, y *(b)* la dificultad de estructurar modelos estadísticos lo suficientemente flexibles como para aprender a explotar la estructura del documento. Estos dos factores se retroalimentaban: sin datos supervisados grandes, los métodos de aprendizaje automático supervisado simplemente no podían entrar al espacio, dejándolo en manos de enfoques no supervisados que usaban plantillas o analizadores sintáctico-semánticos para extraer tuplas de relación y formar un grafo de conocimiento consultable.

Los datasets que existían eran diminutos. El paper cita explícitamente que los corpora de triples documento–pregunta–respuesta estaban "limitados a cientos de ejemplos y por lo tanto útiles sobre todo para *testing*", citando MCTest [9] (Richardson et al.), un dataset de comprensión de dominio abierto con apenas unos cientos de historias. Con cientos de ejemplos no se puede entrenar una red neuronal profunda; sólo se puede evaluar un sistema construido por otros medios.

Una alternativa explorada por la comunidad había sido generar **narrativas y queries sintéticas** [3, 4] — el linaje de Memory Networks (Weston et al.) y los bAbI tasks. Estos enfoques permiten generar cantidades casi ilimitadas de datos supervisados y aislar fenómenos individuales simulados, y de hecho habían mostrado que las redes neuronales eran prometedoras para modelar comprensión lectora. Pero el paper advierte sobre la trampa histórica: en lingüística computacional, muchos enfoques análogos **fracasaron en la transición de datos sintéticos a entornos reales**, porque esos "mundos cerrados" inevitablemente no capturan la complejidad, riqueza y ruido del lenguaje natural (cita a Winograd [5], *Understanding Natural Language*, 1972). El mundo cerrado de bAbI no garantiza que el modelo funcione sobre prosa periodística real.

La tensión conceptual de fondo, que recorre todo el paper, es la distinción entre **comprensión real versus pattern matching y conocimiento del mundo a priori**. Un modelo puede acertar una pregunta no porque "entendió" el documento, sino porque memorizó estadísticas de co-ocurrencia del lenguaje. El paper formaliza el objetivo como estimar la probabilidad condicional

$$p(a \mid c, q),$$

donde $c$ es el documento de contexto, $q$ la query y $a$ la respuesta. Y subraya que para una evaluación enfocada se desea **excluir información adicional**, como el conocimiento del mundo obtenido de estadísticas de co-ocurrencia, con el fin de testear la capacidad central del modelo de **detectar y comprender las relaciones lingüísticas entre entidades del documento de contexto**. Esta es la semilla intelectual que justifica la anonimización (sección 4).

## 3. La idea clave: convertir resúmenes en bullet points en queries cloze

La innovación que destraba el cuello de botella es elegante y oportunista. Los autores observan que las **oraciones de resumen y paráfrasis, junto con sus documentos asociados, pueden convertirse fácilmente en triples contexto–query–respuesta** usando algoritmos simples de detección de entidades y anonimización.

El insumo proviene de dos sitios de noticias que estructuralmente regalan resúmenes: **CNN** (`cnn.com`) y el **Daily Mail** (`dailymail.co.uk`). Ambos proveedores complementan cada artículo con una serie de **bullet points** que resumen aspectos de la información contenida. Recolectaron **93k artículos de CNN** y **220k del Daily Mail**.

El punto de importancia crítica que el paper enfatiza es que **estos puntos de resumen son abstractivos y no simplemente copian oraciones del documento**. Esto es lo que hace que la tarea no sea trivial: si los bullets fueran extractos literales, encontrar la respuesta sería localizar la oración copiada. Al ser paráfrasis abstractivas, el modelo debe realizar generalización léxica y resolución de correferencia para conectar la query con el pasaje.

El pipeline de construcción del triple es:

1. Tomar un artículo y sus bullet points asociados.
2. Convertir cada bullet point en una pregunta tipo **Cloze** [12] — el "procedimiento Cloze" de Wilson Taylor (1953), originalmente una técnica de medición de legibilidad donde se borra una palabra y el lector debe rellenarla.
3. La conversión Cloze consiste en **reemplazar una entidad a la vez por un placeholder** (denotado `X` en los ejemplos). Cada bullet con $k$ entidades puede así generar hasta $k$ queries distintas.

El resultado es un corpus combinado de aproximadamente **1M de data points** (Tabla 1). El documento es el artículo; la query es el bullet con una entidad enmascarada; la respuesta es la entidad enmascarada. La tarea del modelo es predecir qué entidad del documento rellena el hueco.

Un ejemplo del propio paper (Daily Mail validation):
- **Query**: "Producer X will not press charges against Jeremy Clarkson, his lawyer says."
- **Answer**: "Oisin Tymon"

El modelo debe leer el artículo (que cuenta que un productor de la BBC golpeado por Clarkson no presentará cargos, y que la víctima fue Oisin Tymon) para resolver `X`.

## 4. Anonimización y permutación de entidades

Esta es probablemente la decisión de diseño más influyente del paper, y la que más debate generaría después. El objetivo declarado es proveer un corpus para evaluar la capacidad de leer y comprender **un único documento**, no el conocimiento del mundo ni la co-ocurrencia.

El paper ilustra el problema con tres queries Cloze construidas desde titulares del Daily Mail validation set:

- a) "The hi-tech bra that helps you beat breast X"
- b) "Could Saccharin help beat X?"
- c) "Can fish oils help fight prostate X?"

Un modelo de lenguaje de n-gramas entrenado sobre el Daily Mail predeciría trivialmente que **X = cancer**, sin importar el contenido del documento de contexto, simplemente porque "cancer" es una entidad muy frecuentemente "curada" en ese corpus. Esto es exactamente la **solución degenerada** vía priors del modelo de lenguaje que se quiere prohibir: el modelo acierta sin leer.

Para impedirlo, el procedimiento de anonimización y aleatorización tiene tres pasos:

1. **Coreferencia**: usar un sistema de resolución de correferencia para establecer los correferentes en cada data point (qué menciones se refieren a la misma entidad).
2. **Reemplazo abstracto**: reemplazar todas las entidades por **marcadores de entidad abstractos** (`@entityN`, o `entN` en el texto del paper) según la correferencia. Todas las menciones de "Jeremy Clarkson" se vuelven `ent212`; "BBC" se vuelve `ent381`; etc.
3. **Permutación aleatoria**: **permutar aleatoriamente estos marcadores cada vez que se carga un data point**. El mismo documento, recargado, asigna identidades de marcador distintas a las mismas entidades.

La Tabla 3 del paper muestra el contraste. Versión original:

> "The BBC producer allegedly struck by Jeremy Clarkson will not press charges against the 'Top Gear' host..."

Versión anonimizada:

> "the ent381 producer allegedly struck by ent212 will not press charges against the 'ent153' host..."

El razonamiento sobre por qué esto fuerza comprensión es el corazón del argumento: un lector humano puede contestar ambas versiones. Pero en la versión anonimizada **el documento de contexto es obligatorio** para responder, mientras que la versión original podría responderse por alguien con el conocimiento de fondo adecuado (cualquiera que sepa quién es Clarkson). Tras este procedimiento, "la única estrategia restante para responder preguntas es hacerlo explotando el contexto presentado con cada pregunta". Por eso el rendimiento "mide verdaderamente la capacidad de comprensión lectora".

Dos consecuencias técnicas finas:

- La **permutación** es lo que mata el prior del modelo de lenguaje: aunque el modelo quisiera memorizar "ent212 = persona famosa que aparece mucho", la identidad de `ent212` cambia entre recargas, de modo que no hay señal estable a memorizar. Obliga a tratar los marcadores como **variables ligadas localmente** al documento.
- Los modelos **no distinguen entre marcadores de entidad y palabras regulares** (Tabla 2 y texto). El vocabulario incluye todos los tipos de palabra de documentos, preguntas, los maskers de entidad y el marcador de entidad desconocida. Esto endurece la tarea y hace los modelos más generales: el modelo debe **aprender** a diferenciar entidades de variables a partir de la secuencia de entrada, sin que se le privilegien las entidades.

El paper reconoce honestamente que un sistema de producción se beneficiaría de usar todas las fuentes de información disponibles (pistas del lenguaje y co-ocurrencia); la anonimización es una decisión de *evaluación científica*, no de *deployment*.

## 5. Estadísticas del dataset

La Tabla 1 reporta las estadísticas del corpus. Los artículos se recolectaron desde **abril de 2007 para CNN** y **junio de 2010 para el Daily Mail**, ambos hasta fines de **abril de 2015**. La validación es de **marzo de 2015** y el test de **abril de 2015** — una partición temporal limpia que evita fuga entre splits. Se filtraron artículos de más de 2000 tokens y queries cuya entidad-respuesta no aparecía en el contexto.

| Métrica | CNN train | CNN valid | CNN test | DM train | DM valid | DM test |
|---|---|---|---|---|---|---|
| # meses | 95 | 1 | 1 | 56 | 1 | 1 |
| # documentos | 90,266 | 1,220 | 1,093 | 196,961 | 12,148 | 10,397 |
| # queries | 380,298 | 3,924 | 3,198 | 879,450 | 64,835 | 53,182 |
| Máx # entidades | 527 | 187 | 396 | 371 | 232 | 245 |
| Prom # entidades | 26.4 | 26.5 | 24.5 | 26.5 | 25.5 | 26.0 |
| Prom # tokens | 762 | 763 | 716 | 813 | 774 | 780 |
| Tamaño vocabulario | 118,497 (CNN) | | | 208,045 (DM) | | |

Observaciones para un practitioner:

- La escala es de **~380k queries (CNN) + ~880k queries (Daily Mail) en entrenamiento**, frente a los "cientos de ejemplos" de los datasets previos. Es un salto de tres a cuatro órdenes de magnitud.
- El Daily Mail es ~2.3× más grande en documentos y ~2.3× en queries que CNN, y tiene documentos algo más largos (813 vs 762 tokens promedio).
- El ratio queries/documento es ~4.2 en CNN y ~4.5 en Daily Mail, consistente con la idea de generar varias queries Cloze por documento (una por entidad enmascarable en los bullets).
- El promedio de **~26 entidades por documento** es clave: el modelo debe elegir la respuesta correcta de entre ~26 candidatos en promedio. Eso fija el techo de "majority baseline" muy por debajo de lo trivial.

La **Tabla 2** complementa con el porcentaje acumulado de veces que la respuesta correcta está contenida en las N entidades más frecuentes del documento:

| Top N | CNN | Daily Mail |
|---|---|---|
| 1 | 30.5% | 25.6% |
| 2 | 47.7% | 42.4% |
| 3 | 58.1% | 53.7% |
| 5 | 70.6% | 68.1% |
| 10 | 85.1% | 85.5% |

Esta tabla es diagnóstica: si la respuesta estuviera siempre en la entidad más frecuente, bastaría con un baseline de frecuencia. Que la respuesta esté en el Top-1 sólo el 30.5% (CNN) / 25.6% (DM) de las veces confirma que el majority baseline no resuelve la tarea, pero que llegar al Top-10 cubre ~85% muestra que el espacio de candidatos efectivo no es enorme — un detalle que la crítica posterior (sección 9) explotaría.

## 6. Modelos baseline simbólicos y de distancia

Antes de las redes, el paper define baselines y benchmarks para establecer la dificultad de la tarea desde una perspectiva de NLP clásico.

**Baselines de frecuencia (no neuronales, triviales):**
- **Maximum frequency** (majority baseline): elige la entidad más frecuente observada en el documento de contexto.
- **Exclusive frequency** (exclusive majority): elige la entidad más frecuente del contexto **que no aparece en la query**. La intuición es que el placeholder es poco probable que se mencione dos veces en una misma query Cloze, así que la respuesta tiende a ser una entidad ausente de la pregunta.

**Frame-Semantic Parsing benchmark (simbólico):** intenta identificar predicados y sus argumentos — el clásico "quién le hizo qué a quién". Usa un parser frame-semantic estado del arte [13, 14] para extraer triples entidad-predicado $(e_1, V, e_2)$ tanto de la query como del documento, y luego resuelve la query con un conjunto de reglas ordenadas por precedencia, con un trade-off recall/precision creciente (Tabla 4):

| # | Estrategia | Patrón ∈ q | Patrón ∈ d | Ejemplo (Cloze / Contexto) |
|---|---|---|---|---|
| 1 | Exact match | $(p, V, y)$ | $(x, V, y)$ | X loves Suse / Kim loves Suse |
| 2 | be.01.V match | $(p, \text{be.01.V}, y)$ | $(x, \text{be.01.V}, y)$ | X is president / Mike is president |
| 3 | Correct frame | $(p, V, y)$ | $(x, V, z)$ | X won Oscar / Tom won Academy Award |
| 4 | Permuted frame | $(p, V, y)$ | $(y, V, x)$ | X met Suse / Suse met Tom |
| 5 | Matching entity | $(p, V, y)$ | $(x, Z, y)$ | X likes candy / Tom loves candy |
| 6 | Back-off | — | — | Entidad más frecuente del contexto ausente de la query |

Donde $x$ es la entidad propuesta como respuesta y $V$ es un frame PropBank totalmente calificado (p.ej. `give.01.V`). El algoritmo heurístico se afinó iterativamente sobre el validation set. Como el parser usa información lingüística intensivamente, este benchmark corre sobre la versión **no anonimizada** del corpus — pero el paper aclara que esto no da ventaja real porque el enfoque frame-semantic no puede generalizar vía un modelo de lenguaje más allá del que usa durante el parsing; por tanto el objetivo de evaluar comprensión se mantiene.

**Word Distance benchmark:** alinea el placeholder de la query con cada entidad candidata del documento y calcula una medida de distancia entre la query y el contexto alrededor de la entidad alineada. El score suma las distancias de cada palabra de $q$ a su palabra alineada más cercana en $d$, donde la alineación se define por coincidencia directa de palabras o por el sistema de correferencia. La penalización máxima por palabra ($m = 8$) se afina sobre validation.

El rol de estos baselines es triple, declarado en la sección de evaluación: (1) establecer la dificultad de la tarea aplicando un rango amplio de modelos; (2) comparar métodos parse-based versus neuronales; (3) dentro de los neuronales, hacer ablación de qué aporta cada componente.

## 7. Arquitecturas neuronales

Los tres modelos neuronales comparten una capa de salida común. Se estima la probabilidad del tipo de palabra $a$ dada la query $q$ sobre el documento $d$ como:

$$p(a \mid d, q) \propto \exp\big(W(a)\, g(d, q)\big), \quad \text{s.t. } a \in V,$$

donde $V$ es el vocabulario, $W(a)$ indexa la fila $a$ de la matriz de pesos $W$, y los tipos de palabra hacen doble función como índices. La función $g(d, q)$ devuelve un **embedding vectorial conjunto** del par documento–query. Crucialmente, el modelo no privilegia entidades ni variables: debe aprender a diferenciarlas en la secuencia de entrada. Toda la diferencia entre los tres modelos está en cómo se computa $g(d, q)$.

### 7.1 Deep LSTM Reader

El primer modelo prueba si un encoder LSTM profundo puede manejar secuencias largas. Se alimenta el documento palabra por palabra a un Deep LSTM encoder; tras un delimitador `|||`, se alimenta también la query. (Alternativamente se experimenta con el orden inverso: query primero, luego documento — los setups **cqa** y **qca** respectivamente.) Así el modelo procesa cada par documento–query como **una sola secuencia larga**.

Se emplea una celda Deep LSTM con **skip connections** desde cada entrada $x(t)$ hacia cada capa oculta, y desde cada capa oculta hacia la salida $y(t)$:

$$x'(t, k) = x(t) \,\|\, y'(t, k-1),$$
$$y(t) = y'(t, 1) \,\|\, \dots \,\|\, y'(t, K),$$
$$i(t, k) = \sigma\big(W_{kxi}\, x'(t,k) + W_{khi}\, h(t-1,k) + W_{kci}\, c(t-1,k) + b_{ki}\big),$$
$$f(t, k) = \sigma\big(W_{kxf}\, x(t) + W_{khf}\, h(t-1,k) + W_{kcf}\, c(t-1,k) + b_{kf}\big),$$
$$c(t, k) = f(t,k)\, c(t-1,k) + i(t,k)\, \tanh\big(W_{kxc}\, x'(t,k) + W_{khc}\, h(t-1,k) + b_{kc}\big),$$
$$o(t, k) = \sigma\big(W_{kxo}\, x'(t,k) + W_{kho}\, h(t-1,k) + W_{kco}\, c(t,k) + b_{ko}\big),$$
$$h(t, k) = o(t,k)\, \tanh\big(c(t,k)\big),$$
$$y'(t, k) = W_{ky}\, h(t,k) + b_{ky},$$

donde $\|$ es concatenación de vectores, $h(t,k)$ es el estado oculto de la capa $k$ en el tiempo $t$, e $i, f, o$ son las compuertas de entrada, olvido y salida. El Deep LSTM Reader queda definido por

$$g^{\text{LSTM}}(d, q) = y(|d| + |q|),$$

es decir, **la salida en el último paso temporal** después de haber leído todo. El problema conceptual, que motiva los modelos siguientes, es que el LSTM debe **propagar dependencias a través de distancias largas** para conectar query y respuesta, y el vector oculto de ancho fijo forma un **cuello de botella** para ese flujo de información.

### 7.2 Attentive Reader

Para sortear el cuello de botella se introduce un mecanismo de atención inspirado en resultados recientes en traducción [6] (Bahdanau et al.) y reconocimiento de imágenes [7]. El modelo codifica documento y query con **LSTMs bidireccionales separados de una sola capa**.

Sean $\overrightarrow{y}(t)$ y $\overleftarrow{y}(t)$ las salidas forward y backward. El **encoding de la query** de longitud $|q|$ se forma concatenando las salidas finales forward y backward:

$$u = \overrightarrow{y_q}(|q|) \,\|\, \overleftarrow{y_q}(1).$$

Para el **documento**, la salida compuesta de cada token en posición $t$ es:

$$y_d(t) = \overrightarrow{y_d}(t) \,\|\, \overleftarrow{y_d}(t).$$

La representación $r$ del documento se forma como una **suma ponderada** de estos vectores de salida, donde los pesos se interpretan como el grado en que la red atiende a un token particular al responder la query:

$$m(t) = \tanh\big(W_{ym}\, y_d(t) + W_{um}\, u\big),$$
$$s(t) \propto \exp\big(w_{ms}^{\top}\, m(t)\big),$$
$$r = y_d\, s,$$

interpretando $y_d$ como una matriz cuyas columnas son las representaciones compuestas $y_d(t)$ de cada token. La variable $s(t)$ es la **atención normalizada** en el token $t$ (un softmax sobre los scores $w_{ms}^{\top} m(t)$). El embedding del documento $r$ es la suma ponderada de los embeddings de token. El modelo se completa con la combinación no lineal conjunta:

$$g^{\text{AR}}(d, q) = \tanh\big(W_{rg}\, r + W_{ug}\, u\big).$$

El paper observa que el Attentive Reader puede verse como una **generalización de Memory Networks** [3] aplicadas a question answering: Memory Networks atiende a nivel de oración (cada oración como bag of embeddings), mientras que el Attentive Reader atiende a nivel de **token**, donde cada token está embebido con su contexto pasado y futuro completo gracias al encoder bidireccional. Este es un grano de atención más fino.

### 7.3 Impatient Reader

El Impatient Reader va más allá: equipa al modelo con la capacidad de **releer el documento a medida que lee cada token de la query**. En cada token $i$ de la query, computa un vector de representación del documento $r(i)$ usando el embedding bidireccional $y_q(i) = \overrightarrow{y_q}(i) \,\|\, \overleftarrow{y_q}(i)$:

$$m(i, t) = \tanh\big(W_{dm}\, y_d(t) + W_{rm}\, r(i-1) + W_{qm}\, y_q(i)\big), \quad 1 \le i \le |q|,$$
$$s(i, t) \propto \exp\big(w_{ms}^{\top}\, m(i, t)\big),$$
$$r(0) = r_0, \quad r(i) = y_d^{\top}\, s(i) + \tanh\big(W_{rr}\, r(i-1)\big), \quad 1 \le i \le |q|.$$

El término clave es $r(i-1)$ dentro de $m(i, t)$: la atención sobre el documento en el paso $i$ de la query depende del estado de relectura acumulado en el paso anterior. Es un mecanismo de atención que **acumula recurrentemente información del documento a medida que ve cada token de la query**, produciendo al final una representación conjunta:

$$g^{\text{IR}}(d, q) = \tanh\big(W_{rg}\, r(|q|) + W_{qg}\, u\big).$$

Conceptualmente, el "impaciente" no espera a leer toda la query para mirar el documento; reenfoca su atención token a token de la query. El Apéndice C.2 visualiza esto: la atención del Impatient Reader empieza distribuida "fairly arbitrary" y se enfoca en la entidad correcta sólo una vez que la pregunta se ha parseado lo suficiente.

## 8. Experimentos y resultados

El setup de entrenamiento: todos los hiperparámetros se afinaron en los validation sets respectivos. Para el Deep LSTM Reader se exploraron hidden sizes $[64, 128, 256]$, profundidades $[1, 2, 4]$, learning rates $[10^{-3}, 5\!\times\!10^{-4}, 10^{-4}, 5\!\times\!10^{-5}]$, batch sizes $[16, 32]$ y dropout $[0.0, 0.1, 0.2]$; se reportó el mejor (setup **qca**). Para los modelos de atención: hidden sizes $[64, 128, 256]$, una sola capa, learning rates $[10^{-4}, 5\!\times\!10^{-5}, 2.5\!\times\!10^{-5}, 10^{-5}]$, batch sizes $[8, 16, 32]$ y dropout $[0, 0.1, 0.2, 0.5]$. Todos con **asynchronous RmsProp** [20], momentum 0.9 y decay 0.95. Los modelos de atención finales usaron hidden size 256 (Apéndice A, Tabla 6).

La métrica es **accuracy** (fracción de queries cuya entidad-respuesta se predice correctamente). Tabla 5:

| Modelo | CNN valid | CNN test | DM valid | DM test |
|---|---|---|---|---|
| Maximum frequency | 30.5 | 33.2 | 25.6 | 25.5 |
| Exclusive frequency | 36.6 | 39.3 | 32.7 | 32.8 |
| Frame-semantic model | 36.3 | 40.2 | 35.5 | 35.5 |
| Word distance model | 50.5 | 50.9 | 56.4 | 55.5 |
| Deep LSTM Reader | 55.0 | 57.0 | 63.3 | 62.2 |
| Uniform Reader | 39.0 | 39.4 | 34.6 | 34.4 |
| **Attentive Reader** | **61.6** | **63.0** | **70.5** | **69.0** |
| **Impatient Reader** | **61.8** | **63.8** | **69.0** | **68.0** |

Lecturas:

- **El Uniform Reader es la ablación crítica.** Es idéntico al Attentive Reader salvo que **fija todos los parámetros $m(t)$ iguales**, es decir, ignora las variables de atención (atención uniforme). Su rendimiento se desploma a ~39% (CNN) / ~34% (DM), por debajo incluso del Deep LSTM. Esto aísla causalmente la contribución de la atención: la diferencia entre ~39% y ~63% en CNN es atención pura. Es uno de los primeros ablation studies limpios del valor de la atención en MRC.
- **Atención > LSTM puro.** Attentive (63.0 test CNN) e Impatient (63.8) superan al Deep LSTM (57.0), confirmando la hipótesis de que la atención es ingrediente clave por la necesidad de propagar información a larga distancia. Notablemente, los modelos de atención usan LSTMs de **una sola capa** y aun así ganan al Deep LSTM multicapa.
- **El Deep LSTM Reader sorprende positivamente:** ~57-62% test, demostrando que la arquitectura secuencial simple puede abstraer secuencias de hasta dos mil tokens razonablemente bien.
- **Entre los baselines no neuronales, el Word distance es sorprendentemente fuerte** (50.9 CNN test, 55.5 DM test), muy por encima del frame-semantic (40.2 / 35.5). El paper lo explica por la naturaleza de los datos: los highlights del Daily Mail frecuentemente tienen **solapamiento léxico significativo** con pasajes del artículo, lo que favorece a la distancia de palabras. Ejemplo del paper: la query "Tom Hanks is friends with X's manager, Scooter Brown" tiene en el contexto "...turns out he is good friends with Scooter Brown, manager for Carly Rae Jepson"; el word distance alinea correctamente, el frame-semantic falla en capturar las relaciones de amistad/management. Los autores anticipan que sobre datasets con preguntas reales (no Cloze) este baseline rendiría mucho peor.
- **El frame-semantic falla por dos razones:** (1) cobertura pobre — muchas relaciones no son captadas por el parser PropBank porque no se ajustan a la estructura predicado-argumento por defecto, efecto agravado por el tipo de lenguaje de los highlights; (2) no escala trivialmente a situaciones donde **varias oraciones (y por tanto frames)** son necesarias para responder, lo cual es cierto para la mayoría de las queries.
- **Daily Mail es más fácil que CNN** para casi todos los modelos (p.ej. Attentive 69.0 DM vs 63.0 CNN test), probablemente por el mayor solapamiento léxico mencionado.

La Figura 2 muestra **Precision@Recall** en CNN validation, reforzando la superioridad de los modelos atentivos. El Apéndice B muestra que el rendimiento se degrada **ligeramente** con documentos más largos, pero el efecto se vuelve **negligible pasados ~500 tokens**.

**Heatmaps de atención (Figura 3 y Apéndice C):** se visualiza la atención como un mapa de calor sobre el documento. Los autores muestran que para responder correctamente el modelo realiza **generalización léxica** (p.ej. 'killed' → 'deceased') y **resolución de correferencia/anáfora** (p.ej. 'ent119 was killed' → 'he was identified'), pero también integra estas señales con **heurísticas crudas** como la proximidad de las palabras de la query al candidato de respuesta. El Apéndice C documenta honestamente los **fallos**: queries ambiguas donde múltiples entidades son respuestas plausibles tras la anonimización (típicamente ubicaciones geográficas precedidas por "in"), y fallos del clustering de correferencia (p.ej. "Kate Middleton" y "The Duchess of Cambridge" no agrupados, llevando a confundir `ent15` con `ent81`). Esta inclusión de negativos prefigura la crítica posterior sobre el ruido del dataset.

## 9. Limitaciones: la crítica de Chen et al. (2016)

El paper de Hermann tiene limitaciones que sus autores reconocen parcialmente (queries ambiguas, errores de correferencia en los heatmaps negativos), pero la crítica más célebre vino después y conviene anclarla con precisión.

En **"A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task"** (Chen, Bolton y Manning, ACL 2016), el grupo de Stanford analizó manualmente una muestra del dataset y construyó un clasificador con **features simples**. Los hallazgos centrales:

- Un sistema basado en **entity-centric features clásicas** (8 features superficiales: si la entidad aparece en la query, frecuencia, posición de la primera aparición, n-gram match con la query, etc.) más una capa neuronal mínima alcanzaba accuracy comparable o superior a los Readers originales. El clasificador de features de Chen llegó a ~72.4% (CNN) y ~75.8% (Daily Mail), superando los ~63%/~69% de los Attentive/Impatient Readers de Hermann.
- Por inspección manual de ~100 ejemplos, estimaron que **sólo alrededor del 25%** de las preguntas requieren razonamiento e inferencia genuinos sobre múltiples oraciones; cerca de un **tercio o más** son resolubles por coincidencia/parafraseo de una sola oración (paraphrasing), y otra fracción por reconocimiento de entidad parcial.
- Crucialmente, identificaron que en torno a un **~25% de los ejemplos son ruidosos o no resolubles** ni siquiera por un humano, debido a errores en el sistema de correferencia (entidades mal agrupadas o mal anonimizadas), queries ambiguas, o respuestas que requieren conocimiento ausente del contexto tras la anonimización. Esto fija un **techo práctico de accuracy alrededor del 75%**: el modelo no puede acertar el 25% ruidoso porque no hay señal correcta que aprender.

La consecuencia conceptual es incómoda: el dataset que se diseñó precisamente para medir "comprensión real versus pattern matching" resultó ser, en buena parte, **resoluble por pattern matching superficial sobre entidades**. La anonimización eliminó el prior del modelo de lenguaje, pero no eliminó las pistas léxicas y posicionales locales (proximidad, solapamiento) que un clasificador simple explota — exactamente las "heurísticas crudas" que los propios heatmaps de Hermann ya insinuaban. Y el ruido de la correferencia automática significó que una fracción sustancial de la supervisión era incorrecta.

Esto no invalida el paper de Hermann — su metodología de generación a escala y sus arquitecturas atentivas fueron genuinamente seminales — pero recalibró las expectativas: **CNN/Daily Mail mide más "lectura local guiada por atención" que "comprensión profunda con inferencia multi-oración"**, y motivó datasets posteriores diseñados explícitamente para requerir razonamiento más profundo y con anotación humana de mayor calidad.

## 10. Impacto y legado

El impacto de este paper fue desproporcionado respecto de su tamaño. Detonó la **oleada moderna de datasets de Machine Reading Comprehension** y de arquitecturas con atención para QA:

- **SQuAD** (Rajpurkar et al., 2016, Stanford): respondió directamente a las limitaciones de CNN/Daily Mail. En vez de queries Cloze sintéticas con respuestas de entidad anonimizada, SQuAD usa **preguntas formuladas por humanos** (crowdsourcing) sobre párrafos de Wikipedia, con respuestas que son **spans de texto arbitrarios** (no sólo entidades). Eliminó el ruido de la correferencia automática y exigió comprensión de span extraction.
- **MS MARCO** (Microsoft, 2016): queries reales de búsqueda de Bing, respuestas generadas por humanos, multi-documento — atacando la limitación de "single document" que Hermann mismo señalaba en su conclusión.
- **Children's Book Test** (Hill et al., 2016, Facebook), **Who-did-What**, **NewsQA**, **RACE**, etc.: toda una familia de benchmarks Cloze y QA que siguieron el molde.

En el plano de arquitecturas, el linaje es directo:

- El **Stanford Attentive Reader** (Chen et al., 2016) es una reformulación y simplificación del Attentive Reader de Hermann. Chen mostró que con una función de atención bilineal $s(t) \propto \exp(q^{\top} W y_d(t))$ — más simple que el $\tanh$ de Hermann — y prediciendo directamente sobre entidades, se obtenía mejor rendimiento. Esta versión es la que se enseña como modelo canónico de atención para MRC.
- De ahí en adelante: **Bi-Directional Attention Flow (BiDAF)**, **R-NET**, **DrQA**, y eventualmente la transición a representaciones pre-entrenadas (**BERT** y descendientes) que dominaron SQuAD a partir de 2018–2019. El mecanismo de atención query-aware sobre tokens del documento que Hermann formalizó es el ancestro conceptual de todas ellas.

El paper también consolidó la práctica de **visualizar la atención como herramienta de interpretabilidad** y de incluir análisis de errores cualitativos (los apéndices de heatmaps positivos y negativos), un estándar que se volvió común en NLP neuronal.

## 11. Conexión con la Clase 24 (Question Answering Models)

En el material de la Clase 24 (Question Answering Models) del curso, el PDF del profesor presenta el **CNN Dataset como una tarea Cloze (slide 22)** para motivar el **Stanford Attentive Reader**. El rol pedagógico de este paper en ese contexto es el de **puente histórico y conceptual**:

1. **Motiva por qué QA neuronal necesita atención.** La narrativa "Deep LSTM → cuello de botella de vector fijo → atención" que se desarrolla en la sección 7 es exactamente el arco didáctico que justifica por qué los modelos de QA modernos atienden sobre el pasaje en lugar de comprimirlo en un solo vector. El Uniform Reader (atención uniforme, ~39%) versus Attentive (~63%) es el experimento que convence a un estudiante del valor de la atención sin invocar todavía la complejidad de los Transformers.

2. **Introduce la formulación Cloze como puente hacia QA extractivo.** La tarea Cloze de CNN/Daily Mail — predecir una entidad enmascarada leyendo el pasaje — es la versión simplificada de la tarea de span extraction de SQuAD que el Stanford Attentive Reader resuelve. Pedagógicamente se presenta Cloze primero (respuesta = una entidad de un conjunto pequeño) y luego se generaliza a spans arbitrarios.

3. **Conecta con la línea del curso sobre atención y embeddings contextualizados.** El Attentive Reader de Hermann y su sucesor de Stanford son el eslabón entre los embeddings de palabra estáticos (Clase 18) / contextualizados (Clase 20, ELMo/BERT) y la aplicación concreta a responder preguntas. La atención bidireccional sobre tokens del documento es la idea que BERT lleva al extremo con self-attention multicapa.

4. **Enseña la lección metodológica del benchmark.** La crítica de Chen 2016 (sección 9) es material didáctico de primer orden: muestra que un dataset puede medir algo distinto de lo que pretende, que los baselines simples son esenciales para calibrar la dificultad real, y que el progreso aparente en un benchmark debe interrogarse. Para un practitioner como Roberto, que construye sistemas de matching/scoring, la analogía es directa: un score alto en un benchmark no garantiza comprensión; hay que auditar qué fracción del benchmark es resoluble por heurísticas triviales.

## 12. Notas y enlaces

- **arXiv**: [1506.03340](https://arxiv.org/abs/1506.03340) (v3, 19 nov 2015).
- **Venue**: NeurIPS/NIPS 2015, *Advances in Neural Information Processing Systems 28*.
- **Generador de datos**: [github.com/deepmind/rc-data](https://github.com/deepmind/rc-data) — publica el *script* que reconstruye el corpus desde las URLs originales; el texto de los artículos no se distribuye directamente por licencia.
- **Crítica fundamental**: Chen, Bolton & Manning (2016), "A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task", ACL 2016 — establece el techo de ~75% y el ~25% de ruido, e introduce el Stanford Attentive Reader.
- **Linaje de datos**: la conversión de bullets abstractivos a Cloze se inspira en trabajo de summarization [10, 11] (Svore et al.; Woodsend & Lapata, "Automatic generation of story highlights").
- **Linaje de atención**: Bahdanau, Cho & Bengio (2014, traducción neuronal) [6] y Mnih et al. (recurrent visual attention) [7]; Memory Networks de Weston et al. [3] y End-to-End Memory Networks [4] como antecesores directos del mecanismo token-level.
- **Procedimiento Cloze**: el nombre proviene de Wilson L. Taylor (1953), "Cloze procedure: a new tool for measuring readability", *Journalism Quarterly* — originalmente una técnica de medición de legibilidad humana, reutilizada aquí como tarea de aprendizaje automático.
- **Uso posterior**: el split abstractivo (artículo + highlights) de CNN/Daily Mail se reutilizaría masivamente como dataset de **summarization abstractiva** (Nallapati et al., 2016; See et al., Pointer-Generator 2017; y todos los modelos de resumen tipo PEGASUS/BART/T5), conectando directamente con la Clase 22 del curso. El mismo corpus sirvió a dos tareas distintas según se usara el bullet como query Cloze (QA) o como target de resumen (summarization).
