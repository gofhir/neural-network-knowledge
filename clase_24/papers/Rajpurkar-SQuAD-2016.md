# SQuAD: 100,000+ Questions for Machine Comprehension of Text

> Análisis técnico exhaustivo para el curso IA UC (Diplomado en Inteligencia Artificial, PUC Chile). Clase 24 — Question Answering y Machine Reading Comprehension.

## 1. Metadata

| Campo | Valor |
|---|---|
| Título | SQuAD: 100,000+ Questions for Machine Comprehension of Text |
| Autores | Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang |
| Afiliación | Computer Science Department, Stanford University |
| Venue | EMNLP 2016 (Conference on Empirical Methods in Natural Language Processing) |
| arXiv | arXiv:1606.05250v3 [cs.CL], 11 de octubre de 2016 |
| Versión del dataset | SQuAD v1.0 (todos los resultados experimentales del paper son sobre v1.0) |
| Tamaño | 107,785 pares pregunta-respuesta sobre 536 artículos de Wikipedia |
| Disponibilidad | https://stanford-qa.com — libre y gratuito |
| Reproducibilidad | Código, datos y experimentos en CodaLab |

El artículo es el paper fundacional del que probablemente sea el benchmark más influyente del NLP de la segunda mitad de la década de 2010. Cuatro autores, todos de Stanford, con Percy Liang como senior author y Pranav Rajpurkar como primer autor (el mismo Rajpurkar que después lideraría trabajos en ML médico). El contexto temporal es clave: el paper se publicó en junio de 2016 en arXiv y, ya en la versión de octubre que estamos analizando, los autores reportan que en los cuatro meses transcurridos modelos neuronales más sofisticados habían cerrado más de la mitad del gap inicial. SQuAD nació como benchmark y como leaderboard al mismo tiempo, y esa doble naturaleza explica buena parte de su impacto.

## 2. Contexto: el hueco que SQuAD llena

La tesis central del paper en términos de motivación es una observación sobre la historia del machine learning: los datasets grandes y realistas mueven campos enteros. Los autores invocan dos ejemplos canónicos, ImageNet para reconocimiento de objetos (Deng et al., 2009) y el Penn Treebank para parsing sintáctico (Marcus et al., 1993). La pregunta implícita es: ¿por qué la comprensión lectora (Reading Comprehension, RC) no tiene su ImageNet?

La respuesta, según los autores, es que los datasets existentes de RC sufrían de uno de dos defectos mutuamente excluyentes:

**(i) Alta calidad pero demasiado pequeños.** MCTest (Richardson et al., 2013) contiene 660 historias creadas por crowdworkers, con 4 preguntas por historia y 4 opciones de respuesta por pregunta (2640 preguntas en total). Es un dataset real, difícil, que requiere razonamiento de sentido común y razonamiento sobre múltiples oraciones, pero es demasiado pequeño para entrenar modelos estadísticos expresivos modernos. Lo mismo aplica al dataset de Berant et al. (2014) sobre procesos biológicos. El problema no es la dificultad sino la escala: con miles de ejemplos no se puede entrenar una red profunda sin overfitting masivo.

**(ii) Grandes pero semi-sintéticos.** Aquí entran los datasets cloze. Hermann et al. (2015) construyeron el corpus CNN/Daily Mail (1.4M de preguntas) borrando entidades de los resúmenes abstractivos de los artículos de noticias; la tarea es rellenar la entidad faltante a partir del artículo original. El Children's Book Test (CBT, Hill et al., 2015, 688K) consiste en predecir una palabra borrada de una oración dadas las 20 oraciones anteriores. Estos datasets son enormes porque se generan automáticamente a partir de datos que ocurren de forma natural, pero esa misma automatización es su talón de Aquiles: son semi-sintéticos y no comparten las características de las preguntas explícitas de comprensión lectora. Críticamente, el paper cita a Chen et al. (2016), que demostraron que el dataset CNN/Daily Mail requería mucho menos razonamiento del que se pensaba y que el rendimiento estaba "casi saturado". Es decir, el ruido y la simplicidad estructural del cloze automático hacían que el techo del benchmark fuera bajo y poco informativo.

A esto se suma una tercera familia: los datasets puramente sintéticos como bAbI (Weston et al., 2015), que estratifica las tareas por tipo de razonamiento requerido. Son útiles para diagnóstico controlado, pero el lenguaje generado proceduralmente no captura la riqueza ni la ambigüedad del lenguaje natural real.

Hay también una distinción conceptual fina que el paper resalta y que conviene tener clara: en las queries cloze la respuesta es meramente *sugerida* (suggested) por el pasaje, mientras que en SQuAD las respuestas están *entailed* (implicadas/derivables) por el pasaje. Además, las respuestas cloze son palabras o entidades individuales, mientras que las de SQuAD a menudo incluyen no-entidades y frases mucho más largas.

SQuAD se posiciona exactamente en la intersección vacía: grande (casi dos órdenes de magnitud más grande que MCTest) y de alta calidad (preguntas escritas por humanos sobre pasajes reales de Wikipedia). La Tabla 1 del paper hace este survey explícito.

## 3. Idea central: extractive QA, la respuesta es un span

El corazón de SQuAD es una decisión de diseño elegante: **la respuesta a cada pregunta es un segmento contiguo de texto — un *span* — dentro del pasaje de lectura correspondiente.** No hay opciones múltiples, no hay generación libre de texto, no hay banco de respuestas candidatas predefinido. El sistema debe seleccionar, entre todos los spans posibles del pasaje, aquel que responde la pregunta.

Esta formulación, conocida hoy como *extractive question answering*, tiene dos virtudes que el paper enfatiza y que explican su adopción masiva:

**Es realista.** A diferencia del multiple choice (que introduce el sesgo de poder eliminar distractores) y del cloze (que reduce todo a predecir una entidad), responder con un span obliga al sistema a lidiar con un número grande de candidatos. Si el pasaje tiene $L$ palabras, hay $O(L^2)$ spans posibles. El sistema no puede simplemente clasificar entre 4 alternativas; tiene que localizar las fronteras exactas de la respuesta. Los autores reconocen que las preguntas con respuestas basadas en spans son más restringidas que las preguntas interpretativas de exámenes estandarizados avanzados, pero aun así encuentran una rica diversidad de tipos de pregunta y de respuesta.

**Es evaluable automáticamente.** Este es el punto pragmático decisivo. Una respuesta de forma libre (free-form) es difícil de evaluar: ¿cómo decide una máquina si "la fuerza de gravedad" y "gravity" son la misma respuesta correcta? Con spans, la respuesta correcta es un fragmento literal del texto, lo que permite métricas automáticas robustas (Exact Match y F1 token-level, ver sección 6). La restricción de span "viene con el beneficio importante de que las respuestas basadas en spans son más fáciles de evaluar que las respuestas de forma libre". Esta evaluabilidad automática es lo que permite un leaderboard cerrado y comparaciones reproducibles entre cientos de sistemas — el motor social del benchmark.

El ejemplo canónico del paper (Figura 1) es el pasaje sobre precipitación meteorológica:

> *In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity...*

Con preguntas como "What causes precipitation to fall?" → **gravity**, "What is another main form of precipitation besides drizzle, rain, snow, sleet and hail?" → **graupel**, y "Where do water droplets collide with ice crystals to form precipitation?" → **within a cloud**. Nótese que las tres respuestas son spans literales del pasaje, de longitudes distintas (una palabra, una palabra, una frase de tres palabras), y que la primera requiere razonar que "under" en "falls under gravity" denota causa y no localización.

## 4. Construcción del dataset

La recolección se realizó en tres etapas: curación de pasajes, crowdsourcing de pares pregunta-respuesta, y obtención de respuestas adicionales.

**Curación de pasajes.** Para garantizar artículos de calidad, los autores usaron los PageRanks internos de Wikipedia de Project Nayuki para obtener los top 10,000 artículos de la Wikipedia en inglés, de los cuales muestrearon 536 artículos uniformemente al azar. De cada artículo extrajeron párrafos individuales, eliminando imágenes, figuras y tablas, y descartando párrafos de menos de 500 caracteres. El resultado fueron **23,215 párrafos** sobre temas que van desde celebridades musicales hasta conceptos abstractos. La partición se hizo a nivel de artículo (no de párrafo, lo que evita fuga de información entre splits): training 80%, development 10%, test 10%.

**Recolección de pares pregunta-respuesta.** Se emplearon crowdworkers vía la plataforma Daemo (Gaikwad et al., 2015) con Amazon Mechanical Turk como backend. Los requisitos de calidad para los trabajadores eran exigentes: tasa de aceptación de HITs del 97%, mínimo de 1000 HITs completados, y ubicación en Estados Unidos o Canadá. Se les pidió dedicar 4 minutos por párrafo y se les pagó USD 9 por hora. En cada párrafo, el crowdworker formulaba y respondía hasta 5 preguntas sobre el contenido: la pregunta se escribía en un campo de texto y la respuesta se resaltaba (highlight) en el párrafo. Un detalle de diseño importante para combatir el sesgo de copiar literalmente del texto: se animó a los trabajadores a usar sus propias palabras, reforzado con un prompt recordatorio al inicio de cada párrafo y, de manera más drástica, **deshabilitando la funcionalidad de copy-paste** sobre el texto del párrafo. Esto induce variación léxica y sintáctica entre pregunta y pasaje, que es precisamente lo que hace la tarea difícil.

**Recolección de respuestas adicionales.** Para estimar el rendimiento humano y robustecer la evaluación, se obtuvieron **al menos 2 respuestas adicionales** por cada pregunta en los conjuntos de development y test (dando un total de al menos 3 respuestas por pregunta en esos splits). En esta tarea secundaria, al crowdworker se le mostraban solo las preguntas y los párrafos del artículo, y se le pedía seleccionar el span más corto que respondiera la pregunta. Si una pregunta no era respondible por un span del párrafo, se le pedía enviarla sin marcar respuesta. Velocidad recomendada: 5 preguntas en 2 minutos, mismo pago de USD 9 por hora. Un dato que anticipa SQuAD 2.0: sobre dev y test, **el 2.6% de las preguntas fueron marcadas como no respondibles** por al menos uno de los crowdworkers adicionales.

| Métrica de construcción | Valor |
|---|---|
| Artículos de Wikipedia | 536 (muestreados de top 10,000 por PageRank) |
| Párrafos extraídos | 23,215 |
| Pares pregunta-respuesta | 107,785 |
| Preguntas por párrafo | hasta 5 |
| Respuestas por pregunta (dev/test) | ≥ 3 |
| Split | 80% train / 10% dev / 10% test (por artículo) |
| Pago a crowdworkers | USD 9 / hora |
| Plataforma | Daemo sobre Amazon Mechanical Turk |
| Preguntas marcadas no respondibles (dev/test) | 2.6% |

## 5. Análisis del dataset

Para entender las propiedades de SQuAD, los autores analizan el conjunto de development en tres ejes: diversidad de tipos de respuesta, dificultad por tipo de razonamiento requerido, y divergencia sintáctica entre pregunta y oración de respuesta.

**Diversidad en las respuestas.** Las respuestas se categorizan automáticamente. Primero se separan las numéricas de las no-numéricas; las no-numéricas se categorizan usando constituency parses y POS tags de Stanford CoreNLP; las frases de nombre propio se subdividen en persona, ubicación y otras entidades usando tags de NER. La Tabla 2 muestra la distribución:

| Tipo de respuesta | Porcentaje | Ejemplo |
|---|---|---|
| Date | 8.9% | 19 October 1512 |
| Other Numeric | 10.9% | 12 |
| Person | 12.9% | Thomas Coke |
| Location | 4.4% | Germany |
| Other Entity | 15.3% | ABC Sports |
| Common Noun Phrase | 31.8% | property damage |
| Adjective Phrase | 3.9% | second-largest |
| Verb Phrase | 5.5% | returned to Earth |
| Clause | 3.7% | to avoid trivialization |
| Other | 2.7% | quietly |

Agrupando: fechas y otros números suman 19.8%; los nombres propios de tres tipos suman 32.6%; las frases nominales comunes son 31.8%; y el resto (15.8%) son frases adjetivales, verbales, cláusulas y otros. El punto que los autores quieren destacar es que SQuAD va mucho más allá de las entidades de nombre propio — a diferencia de los datasets cloze, casi la mitad de las respuestas no son entidades. Esto es lo que hace la tarea más rica y más cercana a la comprensión lectora real.

**Razonamiento requerido.** Para caracterizar la dificultad, los autores muestrearon 4 preguntas de cada uno de los 48 artículos del dev set (192 ejemplos) y los etiquetaron manualmente. Un resultado central: **todos los ejemplos presentan algún tipo de divergencia léxica o sintáctica entre la pregunta y la respuesta en el pasaje** (un ejemplo puede caer en más de una categoría):

| Tipo de razonamiento | Porcentaje | Descripción |
|---|---|---|
| Lexical variation (synonymy) | 33.3% | Las correspondencias clave entre pregunta y oración de respuesta son sinónimos (p. ej. "sometimes called" ↔ "referred to as") |
| Lexical variation (world knowledge) | 9.1% | Las correspondencias requieren conocimiento del mundo para resolverse |
| Syntactic variation | 64.1% | Tras parafrasear la pregunta a forma declarativa, su estructura de dependencias no coincide con la de la oración de respuesta ni siquiera con modificaciones locales |
| Multiple sentence reasoning | 13.6% | Hay anáfora, o se requiere fusión de múltiples oraciones (p. ej. resolver "They" a "The V&A Theatre & Performance galleries") |
| Ambiguous | 6.1% | Los autores no concuerdan con la respuesta del crowdworker, o la pregunta no tiene respuesta única |

La variación sintáctica domina (64.1%): es el desafío más frecuente. Esto justifica que el modelo baseline invierta tanto en features de dependency tree paths.

**Estratificación por divergencia sintáctica.** Los autores desarrollan un método automático para cuantificar la divergencia sintáctica. La idea (Figura 3): se detectan *anchors*, pares palabra-lema comunes a la pregunta y a la oración de respuesta. Para cada anchor, se extraen dos paths no lexicalizados de los árboles de dependencias: uno desde el anchor en la pregunta hasta la wh-word, y otro desde el anchor en la oración de respuesta hasta el span de respuesta. Se mide la *edit distance* entre estos dos paths (mínimo número de inserciones o borrados para transformar uno en el otro), y la divergencia sintáctica se define como la edit distance mínima sobre todos los anchors posibles. En el ejemplo de "Bainbridge's", el edit cost es $1 + 2 + 1 = 4$. La divergencia sintáctica ignora la variación léxica, y una divergencia pequeña no implica que la pregunta sea fácil, porque puede haber otros candidatos con divergencia similarmente pequeña. El histograma (Figura 4a) muestra un rango amplio de divergencia en el dataset.

## 6. Métricas de evaluación: Exact Match y F1 token-level

SQuAD introdujo el par de métricas que se volvería estándar de facto para extractive QA. Ambas **ignoran puntuación y artículos** (a, an, the) tras una normalización.

**Exact Match (EM).** Mide el porcentaje de predicciones que coinciden *exactamente* con cualquiera de las respuestas ground truth. Es binaria por pregunta: o la predicción normalizada es idéntica a alguna de las respuestas de referencia (acierto = 1) o no lo es (0). Es estricta y por eso siempre queda por debajo del F1.

**F1 token-level (macro-promediado).** Mide el solapamiento promedio entre la predicción y la respuesta ground truth. Se tratan tanto la predicción como el ground truth como *bags of tokens* y se calcula su F1. Formalmente, sea $P$ el conjunto (bag) de tokens predichos y $G$ el de tokens de la respuesta gold. El número de tokens compartidos es $|P \cap G|$ (contando multiplicidad). Entonces:

$$\text{precision} = \frac{|P \cap G|}{|P|}, \qquad \text{recall} = \frac{|P \cap G|}{|G|}, \qquad F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}.$$

Como en el dev/test cada pregunta tiene al menos 3 respuestas gold, el F1 de una pregunta se toma como el **máximo** sobre todas las respuestas de referencia:

$$F_1(\text{pregunta}) = \max_{g \in \text{gold}} F_1(\text{pred}, g).$$

Finalmente se promedia sobre todas las preguntas (macro-average):

$$F_1 = \frac{1}{N} \sum_{i=1}^{N} \max_{g \in \text{gold}_i} F_1(\text{pred}_i, g).$$

El EM análogamente toma el máximo (un acierto si la predicción coincide con *cualquiera* de las gold answers). El uso de múltiples respuestas gold y el operador max es deliberado: humanos razonables discrepan sobre los límites exactos de un span (incluir o no frases no esenciales, como "monsoon trough" versus "movement of the monsoon trough"). Tomar el máximo sobre 3 referencias hace la métrica robusta a esta variabilidad legítima sin penalizar al sistema por elegir una frontera defendible distinta de una referencia arbitraria.

**Ejemplo de cómputo de F1.** Supóngase que la pregunta tiene gold answers {"within a cloud", "a cloud"} y el sistema predice "in a cloud". Tras normalizar (quitar artículos), la predicción es bag {in, cloud} y las gold {within, cloud} y {cloud}. Contra la primera gold: tokens compartidos = {cloud} = 1, precision = 1/2, recall = 1/2, F1 = 0.5. Contra la segunda gold: predicción {in, cloud} vs {cloud}, compartidos = 1, precision = 1/2, recall = 1/1, F1 = $2 \cdot 0.5 \cdot 1 / 1.5 \approx 0.667$. Se toma el máximo: F1 de la pregunta = 0.667. El EM sería 0 porque ninguna coincidencia es exacta.

## 7. Modelos baseline

Antes de los modelos, una restricción de ingeniería compartida por los cuatro métodos: en lugar de considerar los $O(L^2)$ spans posibles, solo se usan spans que son *constituents* en el constituency parse de Stanford CoreNLP. Ignorando puntuación y artículos, el 77.3% de las respuestas correctas del dev set son constituents — lo que pone un **techo efectivo del 77.3%** sobre la accuracy de estos métodos. Durante el entrenamiento, cuando la respuesta correcta no es un constituent, se usa el constituent más corto que la contiene como target.

**Sliding Window Baseline.** Para cada respuesta candidata, se computa el solapamiento unigram/bigram entre la oración que la contiene (excluyendo el candidato mismo) y la pregunta. Se quedan todos los candidatos con solapamiento máximo y se elige el mejor con el enfoque de sliding-window de Richardson et al. (2013). También se implementó la extensión basada en distancia, pero usando solo la oración que contiene el candidato como contexto (en lugar del pasaje completo) por eficiencia.

**Logistic Regression.** El modelo fuerte del paper. Extrae varios tipos de features por cada respuesta candidata, discretizando cada feature continua en 10 buckets de igual tamaño, para un total de **180 millones de features**, la mayoría lexicalizadas o de dependency tree path. Los grupos de features (Tabla 4) incluyen: Matching Word Frequencies (suma de TF-IDF de palabras comunes a pregunta y oración, con features separadas para izquierda, derecha, dentro del span y oración completa), Matching Bigram Frequencies, Root Match (si las raíces de los árboles de dependencias coinciden), Lengths, Span Word Frequencies, Constituent Label, Span POS Tags, features Lexicalized (lemmas de palabras de la pregunta combinados con lemmas de palabras a distancia 2 del span), y Dependency Tree Paths. El loss es la log-verosimilitud multiclase, optimizado con AdaGrad (learning rate inicial 0.1), updates por batch de todas las preguntas de un párrafo (comparten candidatos), regularización L2 con coeficiente $0.1 / (\text{número de batches})$, tres pasadas sobre los datos de entrenamiento.

**Resultados (Tabla 5).**

| Método | EM Dev | EM Test | F1 Dev | F1 Test |
|---|---|---|---|---|
| Random Guess | 1.1% | 1.3% | 4.1% | 4.3% |
| Sliding Window | 13.2% | 12.5% | 20.2% | 19.7% |
| Sliding Win. + Dist. | 13.3% | 13.0% | 20.2% | 20.0% |
| Logistic Regression | 40.0% | 40.4% | 51.0% | 51.0% |
| Human | 80.3% | 77.0% | 90.5% | 86.8% |

El rendimiento humano se estima tratando la *segunda* respuesta de cada pregunta como la "predicción humana" y las otras como ground truth: **86.8% F1 / 77.0% EM** en test. La regresión logística supera ampliamente al sliding window (51.0% vs ~20% F1) pero queda muy por debajo del humano. Un diagnóstico revelador: el modelo selecciona la oración correcta que contiene la respuesta con **79.3% de accuracy**; el grueso de la dificultad está en encontrar el span exacto dentro de la oración.

**Ablation de features (Tabla 6).** Quitar features lexicalizadas y de dependency tree paths es lo más dañino: el F1 dev cae de 51.0% a 35.8% al quitar ambas, a 45.4% sin lexicalizadas, y a 46.4% sin dependency paths. Las demás features individuales apenas mueven la aguja (50.3%–50.6%). Con features lexicalizadas el modelo sobreajusta fuertemente el train (91.7%), pero aumentar L2 perjudica el dev. Comparado con Chen et al. (2016), los dependency tree path features juegan un rol mucho mayor en SQuAD, consistente con que la variación sintáctica es el desafío dominante (64.1%).

**Estratificación por tipo de respuesta (Tabla 7).** El modelo rinde mejor en dates (72.1% F1) y other numeric (62.5%) — categorías con pocos candidatos plausibles y respuestas mayormente de un token. Sufre con entidades nombradas (person, location, other entity) por la mayor cantidad de candidatos, y peor con los "other answer types" (que suman 47.6% del dataset). Los humanos, en cambio, tienen rendimiento más uniforme (82.4%–95.4%). La estratificación por divergencia sintáctica (Figura 5) muestra que el rendimiento de la regresión logística *degrada* a mayor divergencia, mientras que **el humano es insensible a la divergencia sintáctica** — sugiriendo que el entendimiento profundo no se distrae con diferencias superficiales. Los autores proponen que medir esta degradación es útil para juzgar si un modelo generaliza "de la manera correcta".

## 8. El gap humano-máquina como invitación

La estructura retórica del paper culmina en un número: 86.8% F1 humano versus 51.0% F1 del mejor modelo de los autores. Ese gap de ~36 puntos no es un fracaso sino una invitación. Los autores lo enmarcan explícitamente como "un buen problema desafío para investigación futura" y señalan "amplio espacio para avances en modelado y aprendizaje".

Lo notable es que el paper ya documenta la respuesta de la comunidad en tiempo real: en la versión de octubre de 2016 reportan que, desde la liberación del dataset en junio, Wang y Jiang (2016) con un modelo Match-LSTM + Answer Pointer ya habían obtenido 70.3% F1 sobre SQuAD v1.1, "más que reduciendo a la mitad" el gap entre la regresión logística y el humano. Esta es la mecánica de un benchmark exitoso: define una métrica clara, una baseline medible, un techo humano, y deja que cientos de equipos compitan en un leaderboard público. El gap cuantificado convierte el progreso en algo legible y comparable.

## 9. Limitaciones

El paper es honesto sobre sus restricciones, varias de las cuales motivaron trabajo posterior:

**Toda respuesta es un span (no hay unanswerable).** En SQuAD v1.0, cada pregunta tiene por construcción una respuesta que es un span del pasaje. Esto significa que un sistema puede asumir que *siempre* hay una respuesta y nunca necesita abstenerse. El propio dato del 2.6% de preguntas marcadas como no respondibles por crowdworkers adicionales anticipa el problema. Esta limitación es exactamente lo que **SQuAD 2.0** (Rajpurkar, Jia, Liang, 2018) vino a arreglar, agregando más de 50,000 preguntas no respondibles escritas adversarialmente para que se vean plausibles, forzando a los sistemas a decidir *si* responder además de *qué* responder.

**Sesgo de Wikipedia.** Los 536 artículos provienen de la Wikipedia en inglés, sesgados hacia artículos de alto PageRank. Esto restringe el dominio, el registro (texto enciclopédico formal), el idioma (solo inglés) y la cobertura cultural. Modelos entrenados en SQuAD no necesariamente transfieren a texto conversacional, técnico, multilingüe o de baja calidad.

**Preguntas formuladas mirando el pasaje.** Los crowdworkers escribieron las preguntas *teniendo el párrafo a la vista*. Aunque se deshabilitó el copy-paste para forzar reformulación, el proceso induce un sesgo: las preguntas tienden a tener alto solapamiento de contenido con el pasaje y a presuponer que la respuesta existe ahí. Esto difiere de escenarios de QA reales donde el usuario formula la pregunta *sin* haber leído el documento (como en open-domain QA o búsqueda). Es un sesgo de "pregunta retrospectiva" que infla artificialmente la facilidad de localización.

**Span contiguo y restricción de constituents en los baselines.** La respuesta debe ser un fragmento contiguo, lo que excluye respuestas que requieran agregación, conteo, o síntesis de fragmentos no adyacentes. Además, el techo del 77.3% impuesto por restringir candidatos a constituents es una limitación de los baselines específicos del paper (no del dataset en sí, que los modelos neuronales posteriores atacarían sin esa restricción).

## 10. Impacto

SQuAD se convirtió en el benchmark central del Machine Reading Comprehension (MRC) entre 2016 y 2019, y su leaderboard público fue el motor competitivo del subcampo. La trayectoria es ilustrativa:

- **2016:** regresión logística 51.0% F1 (este paper); Match-LSTM + Answer Pointer 70.3% F1 (Wang y Jiang).
- **2016-2018:** una sucesión de arquitecturas con atención bidireccional y mecanismos de pointer (BiDAF, R-Net, DrQA, QANet, entre otras) escalando el F1 progresivamente.
- **2018:** BERT (Devlin et al.) marca el punto de inflexión. El fine-tuning de un Transformer preentrenado sobre SQuAD **superó el rendimiento humano** (86.8% F1) en el leaderboard de v1.1, un hito ampliamente citado como evidencia del poder del preentrenamiento contextual.
- **2018 en adelante:** SQuAD 2.0 reintroduce el desafío al agregar preguntas no respondibles, volviendo a abrir un gap humano-máquina que tomó tiempo cerrar.

Más allá de los números, SQuAD estableció un *patrón metodológico*: dataset grande de alta calidad + métricas automáticas robustas (EM/F1) + leaderboard público con conjunto de test oculto. Ese patrón se replicó en decenas de benchmarks posteriores (GLUE, SuperGLUE, Natural Questions, TriviaQA, HotpotQA, CoQA). La formulación de extractive QA con span también quedó como la interfaz estándar de las pipelines de QA en producción, incluyendo las cabezas de "question answering" de las librerías como Hugging Face Transformers, que exponen directamente el cómputo de logits de inicio y fin de span sobre el pasaje.

## 11. Conexión con la Clase 24

En el material del curso, el PDF del profesor presenta SQuAD en el slide 28 como el ejemplo paradigmático de **extractive QA**, con la consigna "the answer is a span" — exactamente la idea central de la sección 3 de este análisis. El ejemplo que usa el profesor es el de **Marco Polo** (un pasaje sobre el explorador con una pregunta cuya respuesta es un span del texto), que cumple el mismo rol pedagógico que el ejemplo de la precipitación de la Figura 1 del paper: mostrar de forma concreta que la respuesta no se genera ni se elige de una lista, sino que se *localiza* dentro del pasaje.

Los slides 41-42 introducen las métricas **Exact Match y F1** que analizamos en la sección 6. Pedagógicamente, SQuAD es el puente natural en la Clase 24 entre dos mundos: por un lado, la comprensión lectora como tarea cognitiva (entender un texto y responder); por otro, la maquinaria de NLP moderno que la resuelve (representaciones contextuales, atención, fine-tuning de Transformers). El gap humano-máquina del 2016 y su posterior cierre por BERT en 2018 es, además, una narrativa didáctica perfecta: ilustra en un solo benchmark por qué el preentrenamiento contextual fue una revolución, conectando la Clase 24 con el material de embeddings contextualizados (ELMo/BERT/GPT) de las clases previas del módulo de NLP.

Para Roberto, la lección transferible a sistemas reales (incluyendo búsqueda y matching en contextos clínicos): la elección de la *formulación de la tarea* y de la *métrica* es tan determinante como la arquitectura del modelo. SQuAD triunfó no por un truco de modelado sino por convertir comprensión lectora en algo medible automáticamente y a escala. El operador max sobre múltiples referencias gold, en particular, es un patrón directamente aplicable a cualquier evaluación donde existan múltiples respuestas correctas defendibles.

## 12. Notas y enlaces

- **Paper:** Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP 2016. arXiv:1606.05250.
- **Dataset:** https://stanford-qa.com (SQuAD v1.0; los resultados del paper son todos sobre v1.0; v1.1 corrige errores menores de tokenización y es la usada en el leaderboard histórico).
- **Reproducibilidad:** worksheets en CodaLab (código, datos y experimentos completos).
- **Trabajo siguiente:** Rajpurkar, P., Jia, R., Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD* (SQuAD 2.0), ACL 2018 — agrega 50,000+ preguntas no respondibles adversariales, atacando directamente la limitación de "toda respuesta es un span".
- **Hito de superación humana:** Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — primer modelo en superar el F1 humano (86.8%) en el leaderboard de SQuAD v1.1.
- **Herramientas usadas en el paper:** Stanford CoreNLP (constituency parsing, POS tagging, NER, dependency parsing); Daemo + Amazon Mechanical Turk (crowdsourcing); Project Nayuki's Wikipedia PageRanks (curación de pasajes); AdaGrad (optimización del baseline).
- **Datasets de referencia para contexto:** MCTest (Richardson et al., 2013), CNN/Daily Mail (Hermann et al., 2015), CBT (Hill et al., 2015), bAbI (Weston et al., 2015), WikiQA (Yang et al., 2015), TREC-QA (Voorhees y Tice, 2000).
