# Análisis interno — Rajpurkar et al. (2016) "SQuAD: 100,000+ Questions for Machine Comprehension of Text"

> Documento complementario al material público del site. Aquí se profundiza en el contexto histórico de los datasets de QA pre-2016, la metodología de construcción del Stanford Question Answering Dataset, las decisiones de diseño que lo volvieron el benchmark dominante de comprensión lectora 2016-2019, las métricas Exact Match y F1 que se canonizaron a partir de él, los baselines del paper original, la cronología de modelos que lo atacaron (de Match-LSTM a BERT), las limitaciones que motivaron SQuAD 2.0 y los ataques adversariales de Jia & Liang, y la conexión directa con la cabeza `XLNetForQuestionAnswering` del laboratorio 20 del Diplomado IA UC.

- **Paper**: Rajpurkar, Zhang, Lopyrev, Liang. *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. arXiv:1606.05250v3 (11 Oct 2016). EMNLP 2016.
- **Autores**: Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang — Stanford Computer Science Department.
- **PDF local**: [`Rajpurkar-SQuAD1-2016.pdf`](./Rajpurkar-SQuAD1-2016.pdf)
- **Sitio oficial y leaderboard**: <https://rajpurkar.github.io/SQuAD-explorer/>
- **Dataset**: originalmente en `https://stanford-qa.com` (redirige al sitio oficial). Splits y archivos JSON descargables.

---

## 1. Contexto histórico: el problema de no tener un buen dataset de QA en 2015-2016

Para situar el impacto de SQuAD hay que entender la situación de los datasets de reading comprehension (RC) y question answering (QA) en 2015-2016. La comunidad llevaba dos décadas construyendo datasets, pero ninguno cumplía simultáneamente las tres condiciones que necesitaban los modelos neuronales emergentes: **grande, natural y con respuestas verificables**.

### 1.1 Datasets pequeños y de alta calidad

La tradición empieza con Hirschman et al. (1999, **Deep Read**): 600 preguntas reales sobre lecturas de tercero a sexto grado, anotadas a mano. Era de alta calidad pero diminuto. La línea continúa con MCTest (Richardson, Burges & Renshaw 2013): **660 historias** creadas por crowdworkers, cada una con 4 preguntas y 4 opciones de respuesta. Era multiple-choice estilo examen, requería commonsense y razonamiento entre múltiples oraciones, pero con **2,640 preguntas totales** era completamente insuficiente para entrenar redes neuronales modernas. Las extensiones (Berant et al. 2014 sobre procesos biológicos, Clark & Etzioni 2016 sobre exámenes de ciencias de cuarto grado) sumaban algunos cientos de ejemplos más, sin cambiar el orden de magnitud.

El problema era estructural: las anotaciones de alta calidad escalaban linealmente con el costo humano. Construir un dataset de 100K preguntas anotadas a mano costaría cientos de miles de dólares y meses de trabajo, y nadie había encontrado el modelo de crowdsourcing correcto para hacerlo.

### 1.2 Datasets sintéticos

En el otro extremo estaba **bAbI** (Weston, Bordes, Chopra & Mikolov 2015): 20 tareas de razonamiento generadas algorítmicamente a partir de una simulación de mundo. La ventaja es que se puede generar el volumen que uno quiera; la desventaja es que el lenguaje es plantillas. Cada tarea bAbI exhibe un único tipo de razonamiento (single supporting fact, three supporting facts, induction, deduction, etc.) y los modelos pueden aprender a "resolver bAbI" sin aprender comprensión lectora general. Los **Algebra word problems** (Kushman et al. 2014) y los problemas aritméticos de Hosseini et al. (2014) son otro punto en este espectro: 514 problemas escolares con estructura matemática verificable, pero dominio muy estrecho.

### 1.3 Datasets cloze-style: masivos pero engañosos

A mediados de 2015 aparecen dos datasets construidos automáticamente a escala industrial:

| Dataset | Año | Tamaño | Mecanismo de construcción |
|---|---|---|---|
| **CNN / Daily Mail** (Hermann et al.) | 2015 | 1.4M ejemplos | Borrar una entidad nombrada del resumen abstractivo del artículo; predecir cuál es a partir del artículo |
| **Children's Book Test (CBT)** (Hill et al.) | 2015 | 688K ejemplos | Borrar una palabra del enunciado 21 dadas 20 oraciones de contexto del libro infantil |

Ambos son **cloze-style** ("fill in the blank"): la pregunta no es realmente una pregunta sino una oración con un hueco, y la respuesta es una sola palabra o entidad. La ventaja: se generan automáticamente a partir de texto naturalmente ocurrente, así que escalan sin costo humano. La desventaja la documentó devastadoramente Chen, Bolton & Manning (2016) en "A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task": el dataset estaba **casi saturado**. Un sistema entity-centric simple basado en pattern matching superficial obtenía 73% de exactitud (vs 75% del Attentive Reader neuronal contemporáneo), y un análisis manual de 100 ejemplos mostraba que un porcentaje significativo era trivial por anclaje léxico (la entidad correcta era la que aparecía cerca de las palabras del query) o ambiguo incluso para humanos. La conclusión de Chen et al. fue que el cloze-style no medía verdadera comprensión sino entity tracking.

### 1.4 QA open-domain: WikiQA, TREC-QA

Una tercera tradición venía de **open-domain QA**: dado un query de usuario y una colección de documentos, retornar la respuesta. TREC-QA (Voorhees & Tice 2000) con 1,479 preguntas y WikiQA (Yang, Yih & Meek 2015) con 3,047 ejemplos eran de tamaño moderado, basados en query logs reales, pero la tarea era principalmente **sentence selection** (elegir qué oración contiene la respuesta), no extracción del span exacto. Eran insuficientes en tamaño y en granularidad.

### 1.5 La triple condición que faltaba

A mediados de 2016 era evidente que el campo necesitaba un dataset que fuera simultáneamente:

1. **Grande**, del orden de 100K ejemplos o más, para entrenar arquitecturas neuronales con cientos de millones de parámetros.
2. **Natural**, escrito por humanos en lenguaje espontáneo, no sintético ni cloze, con preguntas reales.
3. **Verificable**, con respuestas no ambiguas y métricas automáticas robustas que permitan correr leaderboards públicos.

A esto se agregaba un cuarto requisito implícito: la tarea tenía que ser **extractiva** — la respuesta debía estar literalmente en el texto, como un span — para evitar el problema de evaluar generación libre de texto (donde BLEU y ROUGE son métricas pobres para respuestas cortas) sin renunciar a la naturalidad. SQuAD fue diseñado exactamente para cumplir estas cuatro condiciones a la vez.

---

## 2. Construcción del dataset: tres fases en Mechanical Turk

El dataset consiste en **107,785 pares (pregunta, contexto, respuesta)** sobre **23,215 párrafos** extraídos de **536 artículos de Wikipedia**. La construcción tomó tres fases, todas vía crowdsourcing.

### 2.1 Selección de artículos: PageRank sobre Wikipedia

El paper explicita una decisión sutil pero importante: **no se eligieron artículos al azar**. Se usó la lista de los top-10,000 artículos de Wikipedia en inglés rankeados por **PageRank interno de Wikipedia** (cómputo de **Project Nayuki**, un proyecto independiente que aplica PageRank al grafo de links internos de la enciclopedia). De esos 10,000 artículos más "centrales" en términos de hipertexto, se samplearon **536 uniformemente al azar**.

La razón de filtrar por PageRank no se discute en el paper pero es evidente: artículos centrales tienden a ser sobre temas conocidos, bien escritos, sin esbozos (stubs) ni controversias de edición. Los artículos resultantes cubren un rango amplio — desde celebridades musicales hasta conceptos abstractos — pero excluyen el long-tail enciclopédico donde la calidad es errática.

De cada artículo se extrajeron **párrafos individuales**, eliminando imágenes, figuras, tablas y descartando párrafos de menos de 500 caracteres. Esto da los 23,215 párrafos que se anotaron. Los artículos se dividieron al azar:

| Split | Porcentaje | Tamaño aproximado |
|---|---|---|
| Train | 80% | ~87,000 pares Q-A |
| Dev | 10% | ~10,570 pares Q-A |
| Test | 10% | ~9,533 pares Q-A |

La división es **a nivel de artículo**, no de párrafo o de pregunta. Esto es deliberado y crucial: si un modelo memoriza patrones de un artículo durante entrenamiento, no puede explotarlos en dev/test porque esos artículos no aparecen. Garantiza generalización razonable.

El test set permaneció **oculto** desde la publicación. Solo se accede a él a través del leaderboard oficial: el equipo sube su modelo, los autores lo corren contra el test y publican el score. Esta política es uno de los aportes metodológicos del paper — copia el esquema de ImageNet y lo extiende a NLP — y previno overfitting al test durante años.

### 2.2 Generación de preguntas-respuestas (Mechanical Turk, Round 1)

Los crowdworkers fueron contratados vía la plataforma **Daemo** (Gaikwad et al. 2015), que usa Amazon Mechanical Turk como backend pero agrega un sistema de gobernanza de los anotadores. Los requisitos fueron explícitos y estrictos:

- Tasa de aceptación de HITs ≥ 97%.
- Mínimo 1,000 HITs previos completados.
- Ubicación en Estados Unidos o Canadá (filtro para angloparlantes nativos o de alta fluidez).
- Pago $9/hora durante 4 minutos por párrafo.

A cada anotador se le pidió escribir **hasta 5 preguntas por párrafo**, junto con sus respuestas. Las preguntas se entran en un campo de texto libre. **Las respuestas se obtienen seleccionando con el mouse un span dentro del párrafo** — esta es la decisión arquitectónica clave que define toda la tarea: las respuestas son siempre subcadenas literales del contexto.

Para evitar que los anotadores hicieran copy-paste o usaran las mismas palabras que el párrafo (lo que generaría preguntas triviales por overlap léxico), el interfaz tenía:

- Un recordatorio al inicio de cada párrafo de "usar tus propias palabras".
- **Copy-paste deshabilitado** sobre el texto del párrafo.
- Ejemplos de buenas y malas preguntas con justificación al inicio de cada tarea.

Esto es ingeniería de crowdsourcing sutil pero load-bearing. Como veremos en la sección de limitaciones, no fue suficiente — los anotadores aún tendían a usar paráfrasis cercanas — pero hizo que las preguntas no fueran transcripciones literales.

### 2.3 Anotaciones adicionales (Round 2, solo dev/test)

Para el dev y test sets se ejecutó una segunda ronda crítica: **obtener al menos 2 respuestas adicionales por cada pregunta**, dando un total de **3 respuestas por pregunta** en dev/test (la original + 2 nuevas). En esta ronda los crowdworkers veían las preguntas (sin las respuestas originales) y debían seleccionar el span más corto del párrafo que respondiera la pregunta. Si la pregunta no era contestable por un span, podían enviarla en blanco.

Hallazgos relevantes de Round 2:

- **2.6% de las preguntas fueron marcadas como no contestables** por al menos uno de los anotadores adicionales. Esto adelanta el problema que SQuAD 2.0 (Rajpurkar, Jia & Liang 2018) atacó directamente.
- La inclusión de múltiples respuestas no es un detalle estético: es **necesaria para definir métricas robustas**. Las respuestas humanas a "¿quién es Bainbridge's?" pueden variar en granularidad — "el primer department store del mundo", "department store", "Bainbridge's" — y un modelo que predice una variante razonable debe ser premiado, no penalizado por elegir una respuesta válida pero distinta a la del anotador original.

La existencia de 3 respuestas también permite **medir performance humano** comparándolas entre sí, que es lo que entrega el techo de **86.8% F1** mencionado en el abstract.

### 2.4 La decisión extractiva como restricción productiva

Vale la pena enfatizar la jugada de diseño: **forzar que las respuestas sean spans literales del contexto** es a la vez una restricción y una liberación.

La restricción: SQuAD no puede expresar preguntas cuyo respuesta requiere síntesis ("¿en qué se parecen X e Y?") o reformulación profunda. No reemplaza a un sistema de QA generativo.

La liberación: convierte la tarea en **problema de clasificación sobre $O(L^2)$ spans del párrafo**, no de generación. La evaluación se vuelve mecánica (basta comparar strings). La predicción se vuelve un problema bien definido para arquitecturas pointer-network o de start/end span. Y, crucialmente, alinea la tarea con lo que las CNNs y Transformers hacen bien: contextualizar tokens y producir scores.

Esta restricción técnica fue la palanca que permitió la explosión de modelos 2016-2019. Toda la familia de pointer networks (Wang & Jiang 2016), BiDAF, R-Net, QANet y BERT está construida alrededor del supuesto "predecir índice de inicio y de fin del span".

---

## 3. Métricas: Exact Match y F1

El paper canoniza dos métricas que se vuelven estándar para QA extractivo durante toda la siguiente década.

### 3.1 Normalización del texto

Antes de comparar predicción y ground truth, ambos strings se normalizan:

1. Se convierten a minúsculas.
2. Se eliminan los **artículos** `a`, `an`, `the`.
3. Se eliminan **signos de puntuación**.
4. Se colapsan los espacios en blanco.

Esto evita penalizar al modelo por diferencias triviales como "the Rankine cycle" vs "Rankine cycle" o "Germany" vs "Germany,".

### 3.2 Exact Match (EM)

EM mide el porcentaje de predicciones cuyo string normalizado **coincide exactamente** con al menos una de las respuestas gold normalizadas:

$$\text{EM} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\text{normalize}(\hat{y}_i) \in \{\text{normalize}(y_i^{(1)}), \dots, \text{normalize}(y_i^{(k)})\}\right]$$

Es una métrica binaria por ejemplo, severa: una sola palabra de diferencia (incluyendo paréntesis, fechas con guiones distintos, etc.) hunde el score a 0 para ese ejemplo. El paper reporta 77.0% EM humano en test set, vs 86.8% F1. La brecha de 9.8 puntos entre EM y F1 humano refleja que los humanos también difieren en granularidad sin estar fundamentalmente en desacuerdo.

### 3.3 F1 token-level

F1 mide el **overlap a nivel de token** entre predicción y gold answer:

$$F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

donde:

- $\text{precision} = \frac{|\text{tokens}(\hat{y}) \cap \text{tokens}(y)|}{|\text{tokens}(\hat{y})|}$
- $\text{recall} = \frac{|\text{tokens}(\hat{y}) \cap \text{tokens}(y)|}{|\text{tokens}(y)|}$

Se trata la predicción y la respuesta gold como **bags of tokens** (multisets, aunque en práctica las respuestas son cortas y los multiset y set casi coinciden). Para múltiples respuestas gold, se toma el **máximo F1** sobre todas:

$$F_1(\hat{y}_i, y_i^{(1..k)}) = \max_{j \in 1..k} F_1(\hat{y}_i, y_i^{(j)})$$

Y luego se promedia macro sobre todas las preguntas del dev/test set:

$$\text{F1}_{\text{dataset}} = \frac{1}{N} \sum_{i=1}^{N} F_1(\hat{y}_i, y_i^{(1..k)})$$

### 3.4 Por qué F1 y no BLEU/ROUGE

Una decisión deliberada del paper es **no usar BLEU**. BLEU (Papineni et al. 2002) está diseñada para evaluar traducción automática contra múltiples referencias largas. Para respuestas cortas (mediana ~3 tokens en SQuAD), BLEU se degenera:

- BLEU-4 (que necesita 4-gramas) es indefinida para respuestas de 1-3 palabras.
- BLEU con brevity penalty severo penaliza respuestas cortas correctas.
- BLEU no maneja bien sinonimia ni equivalencia parcial a nivel léxico.

F1 token-level es la métrica natural para respuestas cortas: mide overlap léxico simple, premia respuestas parcialmente correctas (un modelo que predice "Bainbridge's department store" cuando la gold es "Bainbridge's" recibe parcial), no requiere n-gramas largos, y se generaliza fácilmente a múltiples gold answers vía el máximo.

ROUGE-L tendría sentido pero la diferencia con F1 token-level es marginal para spans cortos. F1 ganó por simpleza.

### 3.5 Implementación de referencia

El script `evaluate-v1.1.py` distribuido por los autores fija la normalización exacta. Cualquier discrepancia entre implementaciones (por ejemplo, qué cuenta como "puntuación" o si se elimina el espacio entre números en "1, 000") cambia ligeramente los scores. La convención de la comunidad fue siempre **usar el script oficial** para reproducibilidad.

---

## 4. Estadísticas del dataset

### 4.1 Tamaños

| Cantidad | Valor |
|---|---|
| Total de pares Q-A | **107,785** |
| Artículos | **536** |
| Párrafos | **23,215** |
| Train | ~87,599 pares |
| Dev | ~10,570 pares |
| Test | ~9,533 pares |
| Respuestas por pregunta (train) | 1 |
| Respuestas por pregunta (dev/test) | ≥ 3 |

Comparado con MCTest (2,640 pares), SQuAD es casi **dos órdenes de magnitud más grande**. Comparado con CNN/Daily Mail (1.4M), es 13× más chico, pero la calidad es radicalmente superior porque las preguntas son naturales y no cloze.

### 4.2 Longitudes características

| Métrica | Valor aproximado |
|---|---|
| Longitud media de pregunta | ~10 palabras |
| Longitud media de respuesta | ~3 palabras |
| Longitud media de párrafo | ~120 palabras |
| Longitud mínima de párrafo (filtrada) | 500 caracteres |

La asimetría es relevante: las preguntas son ~3× más largas que las respuestas. Esto refleja una pragmática natural: las preguntas explican qué se busca; las respuestas son tan cortas como puedan serlo.

### 4.3 Distribución de tipos de respuesta (Tabla 2 del paper)

Los autores categorizaron automáticamente las respuestas usando constituency parses, POS tags y NER tags de Stanford CoreNLP:

| Tipo de respuesta | Porcentaje | Ejemplo |
|---|---|---|
| Date | 8.9% | "19 October 1512" |
| Other Numeric | 10.9% | "12" |
| Person | 12.9% | "Thomas Coke" |
| Location | 4.4% | "Germany" |
| Other Entity | 15.3% | "ABC Sports" |
| Common Noun Phrase | 31.8% | "property damage" |
| Adjective Phrase | 3.9% | "second-largest" |
| Verb Phrase | 5.5% | "returned to Earth" |
| Clause | 3.7% | "to avoid trivialization" |
| Other | 2.7% | "quietly" |

Lecturas:

- **19.8% son números** (Date + Other Numeric). Categoría más fácil para baselines: los párrafos contienen pocos números y filtrar por POS tag los identifica.
- **32.6% son entidades nombradas** (Person + Location + Other Entity). También relativamente fácil: NER tags acotan los candidatos.
- **31.8% son common noun phrases**: la categoría dominante y la más difícil. No hay filtro automático que las identifique sin ambigüedad.
- **15.8% restante** (Adjective Phrase + Verb Phrase + Clause + Other) es la cola larga sintácticamente diversa que rompe los baselines basados en NER.

Comparar con datasets cloze-style: CNN/Daily Mail y CBT tienen **100% de respuestas que son entidades nombradas o palabras individuales**. SQuAD es mucho más diverso, lo que lo hace más difícil pero también más representativo de QA real.

### 4.4 Distribución del tipo de razonamiento (Tabla 3 del paper)

Los autores etiquetaron manualmente 192 ejemplos (4 preguntas × 48 artículos del dev set) según el tipo de razonamiento requerido. Las categorías no son mutuamente excluyentes (un ejemplo puede caer en varias):

| Tipo de razonamiento | % | Descripción |
|---|---|---|
| Lexical variation (synonymy) | 33.3% | La correspondencia entre pregunta y respuesta es por sinónimos. Ejemplo: "called" en pregunta corresponde a "referred to as" en oración. |
| Lexical variation (world knowledge) | 9.1% | Requiere conocimiento del mundo. Ejemplo: "governing bodies" corresponde a "European Parliament and Council". |
| Syntactic variation | 64.1% | Tras paráfrasear la pregunta a declarativa, su estructura sintáctica no coincide con la de la oración respuesta. |
| Multiple sentence reasoning | 13.6% | Hay anáfora o se requiere fusionar información de múltiples oraciones. |
| Ambiguous | 6.1% | Los autores no están de acuerdo con la respuesta del crowdworker o la pregunta no tiene respuesta única. |

Observaciones:

- **64.1% requiere variación sintáctica**. La mayoría de SQuAD no es simple matching léxico — hay reestructuración de cláusulas, paráfrasis, conversiones entre voz activa y pasiva.
- **42.4% requiere alguna variación léxica** (33.3% sinónimos + 9.1% world knowledge). Esto motivó los modelos de attention sobre embeddings densos (BiDAF, R-Net) por encima de matching exacto.
- **13.6% requiere razonamiento multi-oración**. Este es el subset que QANet y BERT explotan agresivamente vía atención sobre el contexto completo.
- **Casi todos los ejemplos tienen alguna divergencia** entre pregunta y respuesta. No hay preguntas triviales que se resuelven solo por anclaje léxico — al menos según esta anotación manual.

### 4.5 Stratificación por divergencia sintáctica

Los autores desarrollaron una métrica automática de **distancia de edición entre paths del dependency tree** que conectan la pregunta y la oración respuesta. Concretamente:

1. Identificar **anchors**: pares (palabra, lema) comunes a pregunta y oración respuesta.
2. Para cada anchor, computar el path en el dependency tree desde el anchor hasta el wh-word (en la pregunta) y desde el anchor hasta el span de respuesta (en la oración).
3. Calcular la edit distance (deleciones + inserciones + substituciones) entre los dos paths.
4. La divergencia sintáctica de la pregunta es el **mínimo edit distance sobre todos los anchors posibles**.

El histograma de la Figura 4a muestra una distribución amplia entre 0 y 8, con moda en 2-3. La Figura 5 muestra que la performance del modelo de regresión logística **cae monótonamente** con la divergencia (de ~60% F1 en divergencia 0 a ~30% F1 en divergencia 7), mientras que **la performance humana es plana** (~90% F1 a través de todo el rango). Esta es una observación devastadora: el baseline no captura nada de la generalización composicional que los humanos hacen sin esfuerzo. La interpretación de los autores: medir el grado de degradación con divergencia es una buena prueba de generalización para futuros modelos.

---

## 5. Análisis humano del razonamiento requerido

La Tabla 3 del paper, complementada con la Figura 3 (un ejemplo concreto de cómputo de divergencia sintáctica) y la Figura 4 (más ejemplos con edit distances 0 y 6), constituye uno de los aportes metodológicos más útiles de SQuAD: **una taxonomía operacional del razonamiento de QA**. Vale la pena profundizar en cada categoría con ejemplos del propio paper.

### 5.1 Lexical variation por sinonimia (33.3%)

Ejemplo: pregunta "What is the Rankine cycle sometimes called?", oración "The Rankine cycle is sometimes referred to as a practical Carnot cycle". La correspondencia es **called ↔ referred to as**. El modelo necesita un mapeo de sinonimia para conectar la pregunta con la respuesta. Los baselines basados en n-grama overlap fallan; los modelos con embeddings densos (Word2Vec, GloVe) lo manejan parcialmente porque "called" y "referred" están cerca en el espacio embedding; los modelos contextualizados (ELMo, BERT) lo manejan robustamente.

### 5.2 Lexical variation por world knowledge (9.1%)

Ejemplo: "Which governing bodies have veto power?" / "The European Parliament and the Council of the European Union have powers of amendment and veto". La correspondencia **governing bodies ↔ European Parliament + Council of the European Union** no es sinonimia — es un hecho del mundo (estas son las governing bodies de la EU). Esta categoría es la más resistente a las técnicas distributivas porque no es relación léxica sino factual. Solo modelos con grandes cantidades de conocimiento del mundo (es decir, pre-trained LMs sobre corpus masivos) pueden capturarla.

### 5.3 Syntactic variation (64.1%)

Ejemplo: "What Shakespeare scholar is currently on the faculty?" / "Current faculty include the anthropologist Marshall Sahlins, ..., Shakespeare scholar David Bevington". La estructura sintáctica difiere: la pregunta tiene wh-fronting; la respuesta usa una lista de apósitos. Para que un modelo conecte el "scholar" como anchor entre ambas, necesita aplicar transformaciones sintácticas. Esta es la categoría dominante (64.1%) y la razón por la cual los modelos puramente lexicales fallan masivamente.

### 5.4 Multiple sentence reasoning (13.6%)

Ejemplo: "What collection does the V&A Theatre & Performance galleries hold?" / "The V&A Theatre & Performance galleries opened in March 2009. ... They hold the UK's biggest national collection of material about live performance." Para responder hay que resolver la anáfora **They ↔ V&A Theatre & Performance galleries** y conectar dos oraciones separadas. Es razonamiento entre oraciones, no dentro de una. Los modelos basados en attention sobre el párrafo completo (BiDAF en adelante) manejan esto naturalmente; los baselines basados en sentence selection fallan.

### 5.5 Ambiguous (6.1%)

Ejemplo: "What is the main goal of criminal punishment?" / "Achieving crime control via incapacitation and deterrence is a major goal of criminal punishment". La respuesta del crowdworker pudo ser cualquiera de varios spans válidos. Los autores reconocen esta categoría como **ruido inherente del dataset** — es el techo de 13.2 puntos (100% - 86.8% F1 humano) que no se puede romper.

### 5.6 Implicaciones para la arquitectura de modelos

La distribución 33% sinonimia + 9% world knowledge + 64% variación sintáctica + 14% razonamiento multi-oración + 6% ambigüedad explica empíricamente por qué la cadena de modelos que atacó SQuAD evolucionó como evolucionó:

- **Sliding window / BoW**: solo captura matching léxico exacto. Pierde 100% de los casos con sinonimia o variación sintáctica.
- **Logistic regression con features sintácticos** (paper de SQuAD): añade dependency paths, captura algunas variaciones sintácticas locales. Llega al 51% F1.
- **Pointer networks + attention** (Match-LSTM, BiDAF): captura attention entre tokens de pregunta y contexto, maneja sinonimia y razonamiento multi-oración. Llega a 70-77% F1.
- **Pre-trained LMs** (ELMo features sobre BiDAF, luego BERT): capturan sinonimia robusta y algo de world knowledge vía corpora masivos. Llega a 90%+ F1.
- **BERT fine-tuned**: domina todas las categorías excepto las ambiguas. 93.2% F1, supera humanos.

---

## 6. Baselines del paper

Los autores implementaron y evaluaron cuatro métodos. Su análisis de qué hace bien y qué no cada uno es un mapa claro del problema.

### 6.1 Random Guess

Para calibrar: elegir un span aleatorio del párrafo da **1.1% EM / 4.1% F1 en dev**, **1.3% EM / 4.3% F1 en test**. Es la línea de fondo.

### 6.2 Sliding Window

Para cada candidato (los constituyentes en el parse del párrafo), computar el overlap de unigramas y bigramas con la pregunta. Quedarse con los que tienen máximo overlap. Entre esos, elegir el mejor por el "sliding window approach" de Richardson et al. (2013) — esencialmente promediar el TF-IDF de las palabras de la pregunta dentro de una ventana centrada en el candidato.

Performance: **13.2% EM / 20.2% F1 en dev**, **12.5% EM / 19.7% F1 en test**. Una variante con distance extension (penalizar candidatos lejos de las palabras matched) da márgenes despreciables: **13.3% / 20.2%**.

Lo que muestra: matching léxico simple captura algo pero está muy lejos de un sistema útil. El 20% F1 es básicamente un baseline "el modelo aprendió a localizar la oración respuesta y picar algo razonable".

### 6.3 Logistic Regression — el modelo principal del paper

Es el modelo "fuerte" que los autores construyen. Extrae múltiples grupos de features (Tabla 4 del paper), discretiza cada feature continua en 10 buckets, y resulta en **180 millones de features** totales. Es lexicalizado y muy sparse.

**Grupos de features:**

| Grupo | Descripción |
|---|---|
| Matching Word Frequencies | Suma de TF-IDF de palabras compartidas entre pregunta y oración del span. Separado para left, span, right, whole. |
| Matching Bigram Frequencies | Análogo con bigramas (TF-IDF generalizado de Shirakawa et al. 2015). |
| Root Match | Si los roots del parse coinciden o se contienen. |
| Lengths | Número de palabras a la izquierda, derecha, dentro del span, total. |
| Span Word Frequencies | Suma de TF-IDF de palabras del span. Sirve para descartar spans de palabras comunes. |
| Constituent Label | Etiqueta del constituyente (NP, VP, etc.), opcionalmente cruzada con el wh-word. |
| Span POS Tags | Secuencia de POS del span, opcionalmente cruzada con el wh-word. |
| Lexicalized | Lemas de las palabras de la pregunta cruzados con lemas de palabras a distancia ≤ 2 del span en el parse. |
| Dependency Tree Paths | Para cada palabra común a pregunta y oración, el path en el parse de la palabra hasta el span, opcionalmente cruzado con el path al wh-word. |

**Generación de candidatos:** en vez de considerar todos los $O(L^2)$ spans posibles del párrafo, se restringe a los **constituyentes del constituency parse** de Stanford CoreNLP. Los autores miden que **77.3% de las respuestas correctas del dev set son constituyentes**, lo que pone un techo efectivo de 77.3% sobre el approach. Durante training, si la respuesta no es un constituyente, se usa el constituyente más corto que la contiene.

**Entrenamiento:**

- Loss: multiclass log-likelihood sobre los candidatos del párrafo.
- Optimizador: AdaGrad, lr inicial 0.1.
- Batching: una pregunta = un batch sobre todos los candidatos del párrafo (eficiencia porque comparten candidatos).
- Regularización: L2 con coeficiente 0.1 dividido por número de batches.
- 3 pasadas sobre el train set.

**Performance:** **40.0% EM / 51.0% F1 en dev**, **40.4% EM / 51.0% F1 en test**. Mucho mejor que el sliding window, pero lejos del humano.

**Análisis del paper:**

- El modelo selecciona la oración respuesta correctamente con **79.3% accuracy**. El grueso del error está en elegir el span exacto dentro de la oración, no en encontrar la oración.
- Ablation (Tabla 6 del paper): las features más importantes son **lexicalized** (sin ellas, F1 dev cae a 45.4%) y **dependency tree paths** (sin ellos, F1 dev cae a 46.4%). Las features básicas (length, span POS, root match) aportan muy poco individualmente.

Tabla de ablation completa del paper:

| Configuración | Train F1 | Dev F1 |
|---|---|---|
| Logistic Regression (full) | 91.7% | **51.0%** |
| – Lex., – Dep. Paths | 33.9% | 35.8% |
| – Lexicalized | 53.5% | 45.4% |
| – Dep. Paths | 91.4% | 46.4% |
| – Match. Word Freq. | 91.7% | 48.1% |
| – Span POS Tags | 91.7% | 49.7% |
| – Match. Bigram Freq. | 91.7% | 50.3% |
| – Constituent Label | 91.7% | 50.4% |
| – Lengths | 91.8% | 50.5% |
| – Span Word Freq. | 91.7% | 50.5% |
| – Root Match | 91.7% | 50.6% |

Nota: con lexicalized features el modelo **overfittea masivamente** (91.7% train vs 51% dev). El paper observa que incrementar L2 hurts dev performance — el overfitting es estructural por la dimensionalidad (180M features).

- Performance estratificada por tipo de respuesta (Tabla 7 del paper):

| Tipo | LR Dev F1 | Human Dev F1 |
|---|---|---|
| Date | 72.1% | 93.9% |
| Other Numeric | 62.5% | 92.9% |
| Person | 56.2% | 95.4% |
| Location | 55.4% | 94.1% |
| Other Entity | 52.2% | 92.6% |
| Common Noun Phrase | 46.5% | 88.3% |
| Adjective Phrase | 37.9% | 86.8% |
| Verb Phrase | 31.2% | 82.4% |
| Clause | 34.3% | 84.5% |
| Other | 34.8% | 86.1% |

Los humanos son **uniformemente buenos** (82-95% F1) en todas las categorías. El modelo LR es **muy desigual**: bueno en números y entidades (donde POS tags y NER hacen el trabajo), pobre en frases verbales y cláusulas (donde se requiere comprensión profunda). La conclusión: el baseline LR es **superficie + un poco de sintaxis**, no comprensión.

### 6.4 Performance humano

Como se describió en §2.3, el dev/test tiene ≥3 respuestas por pregunta. Para medir performance humano, los autores tratan la **segunda respuesta** de cada pregunta como predicción y las otras como ground truth. El resultado:

- **77.0% EM, 86.8% F1 en test**.
- **80.3% EM, 90.5% F1 en dev**.

La interpretación: los humanos están bastante de acuerdo en qué responder (86.8% F1) pero difieren en granularidad — uno responde "monsoon trough" y otro "movement of the monsoon trough", lo que da F1 alto pero EM bajo. **La brecha LR vs humano es 35.8 puntos F1** — espacio enorme para mejora.

---

## 7. Impacto: la "carrera" de modelos QA en SQuAD 1.x

SQuAD se publicó en junio 2016. En los siguientes 30 meses, la comunidad cerró la brecha del 35.8% F1 hasta superar el humano. Este es el resumen de la cronología.

### 7.1 Cronología de modelos

| Año | Modelo | Equipo | Dev F1 | Test F1 | Innovación principal |
|---|---|---|---|---|---|
| 2016-06 | Logistic Regression | Stanford (paper original) | 51.0 | 51.0 | Features manuales + dependency paths |
| 2016-08 | **Match-LSTM + Answer Pointer** | Wang & Jiang (SMU) | 70.0 | 73.7 | Pointer Network sobre matching tokens |
| 2016-11 | **BiDAF** (Bi-Directional Attention Flow) | Seo et al. (UW + AI2) | 77.3 | 77.3 | Atención bidireccional contexto↔pregunta sin sumarización |
| 2017-04 | **DCN+** | Salesforce (Xiong, Zhong, Socher) | 78.1 | 78.9 | Dynamic Coattention con mixed objective |
| 2017-05 | **R-Net** | Microsoft Research Asia | 79.5 | 79.7 | Gated self-attention sobre el contexto |
| 2017-09 | **R-Net + ensemble** | MSRA | 84.0 | 84.7 | Ensemble de 25 R-Net |
| 2018-04 | **QANet** | Google Brain + CMU (Yu et al.) | 84.6 | 84.6 | Primer modelo **sin RNNs**: CNNs + self-attention. 3-13× más rápido. |
| 2018-10 | **BERT-base + fine-tune** | Google (Devlin et al.) | 88.5 | — | Pre-trained Transformer bidireccional |
| 2018-10 | **BERT-large + fine-tune** | Google | 90.9 | 91.8 | Tamaño + bidireccional + MLM/NSP. **Supera humanos**. |
| 2018-12 | **BERT-large ensemble + TriviaQA** | Google | 91.8 | **93.2** | Augmentation con TriviaQA, ensemble |

### 7.2 Línea por línea

**Match-LSTM con Answer Pointer (Wang & Jiang 2016)** fue el primer modelo neuronal serio en SQuAD. La idea: codificar pregunta y contexto con LSTMs, computar attention contextualizada de cada token del contexto sobre la pregunta (match-LSTM, una variante de Wang & Jiang 2015 para entailment), y luego usar **Pointer Networks** (Vinyals et al. 2015) para apuntar a los índices de inicio y fin del span. El uso de Pointer Networks fue conceptualmente brillante: en vez de generar la respuesta, **se generan los índices en el contexto**, lo que respeta la naturaleza extractiva del problema. Esta arquitectura set el patrón "encoder + pointer to span" que todos los modelos posteriores siguieron.

**BiDAF (Seo et al. 2017, ICLR)** añadió la idea clave de **bidirectional attention flow sin sumarización**. En vez de resumir la pregunta en un vector y atender el contexto a ese vector (Match-LSTM lo hacía implícitamente), BiDAF computa dos matrices de attention: contexto→pregunta (qué tokens de la pregunta son relevantes para cada token del contexto) y pregunta→contexto (qué tokens del contexto son relevantes para cada token de la pregunta). Las representaciones contextualizadas resultantes pasan por más capas de modeling antes de predecir start/end. BiDAF se convirtió en la arquitectura de referencia durante 18 meses.

**R-Net (MSRA 2017)** introdujo **self-attention sobre el contexto**, una idea adelantada a los Transformers de 7 meses después. La motivación: dependencias de largo alcance en el contexto requieren attention dentro del contexto, no solo entre contexto y pregunta. R-Net dominó el leaderboard durante meses.

**QANet (Yu et al. 2018, ICLR)** fue el quiebre arquitectónico. Reemplazó completamente las RNNs por **convolutional + self-attention encoder blocks**, idea inspirada en el Transformer recién publicado (Vaswani et al. 2017). Resultado: misma performance que ensembles de RNN-based con **3-13× speedup**. QANet también introdujo **data augmentation con back-translation** (En → Fr → En) que añadía variabilidad léxica. Fue el último modelo SOTA antes de BERT.

**BERT (Devlin et al. 2018)** rompió definitivamente la barrera humana. La cabeza de QA es minimalista (Section 7.3 del análisis BERT): dos vectores aprendidos $S$ y $E$, score de inicio $S \cdot T_i$ para cada token del contexto, score de fin análogo, span = par $(i, j)$ con $j \ge i$ que maximiza $S \cdot T_i + E \cdot T_j$. **2048 parámetros nuevos por encima de los 340M de BERT-large**. Tres meses de leaderboard fueron suficientes para que la era BERT desplazara a todos los modelos especializados.

### 7.3 Lecciones de la carrera

| Lección | Evidencia |
|---|---|
| Las RNNs no eran necesarias para QA | QANet y BERT, ambos sin RNNs, dominaron. |
| La attention bidireccional es lo crítico | BiDAF, R-Net y BERT explotan attention en múltiples ejes. |
| El pre-training masivo desbloquea representaciones que ningún modelo task-specific puede aprender | BERT salta +5 puntos F1 sobre QANet con prácticamente la misma arquitectura por encima. |
| El ensemble + data augmentation aporta 1-2 puntos | TriviaQA augmentation de BERT-large añade 1.4 F1. |
| La barrera humana en SQuAD 1.x se rompió en ~30 meses | 51% F1 (2016-06) → 93.2% F1 (2018-12). |

### 7.4 Lo que se rompió primero

Es ilustrativo notar **qué subset de SQuAD se rompió primero**. Las categorías Date, Other Numeric, Person, Location fueron las primeras en saturar — los modelos llegaron a 95%+ F1 en estas categorías antes que en common noun phrases o verb phrases. La razón es que estas categorías tienen filtros NER/POS robustos y los candidatos por pregunta son pocos. **La última frontera fue siempre common noun phrases y razonamiento multi-oración**, donde BERT aportó la mayor ganancia marginal.

---

## 8. Limitaciones reconocidas y críticas posteriores

A pesar de su éxito, SQuAD 1.0 mostró rápidamente varias limitaciones. Algunas las reconocían los propios autores; otras emergieron de la literatura posterior.

### 8.1 Solo Wikipedia → sesgo de dominio

Los 536 artículos provienen exclusivamente de Wikipedia en inglés, top-10K por PageRank. Esto implica:

- Estilo enciclopédico, prosa controlada, baja ambigüedad referencial.
- Temas predominantemente conocidos (sin long-tail enciclopédico).
- Lenguaje formal, poca slang, poca jerga técnica fuera de las disciplinas más conocidas.

Modelos entrenados en SQuAD no se transfieren bien a **noticias** (estilo distinto, citas, opiniones), **literatura** (narrativa, voz, anáfora compleja), **conversación** (turnos, contexto compartido implícito), **dominios técnicos** (jerga médica, legal). Esto motivó datasets sucesores como TriviaQA (preguntas de trivia con contexto Wikipedia + Web), Natural Questions (preguntas reales de Google Search), MS MARCO (preguntas conversacionales), QuAC (QA conversacional) y DROP (preguntas que requieren cómputo numérico).

### 8.2 Solo preguntas con respuesta → SQuAD 2.0

La limitación más profunda que los propios autores atacaron dos años después: en SQuAD 1.0, **toda pregunta tiene respuesta** en el contexto. Esto enseña al modelo un sesgo perverso: siempre va a producir un span, aunque no haya información relevante. En aplicaciones reales (motor de búsqueda, asistente conversacional), la pregunta "qué temperatura tendrá Santiago mañana" sobre un párrafo sobre la geología de Chile **no tiene respuesta**, y un sistema robusto debe reconocerlo.

**SQuAD 2.0** (Rajpurkar, Jia & Liang 2018, ACL) añadió 50K preguntas adversariales sin respuesta, escritas por crowdworkers que veían el contexto y debían formular preguntas que parecieran responsables pero no lo fueran. El resultado fue dramático: modelos que tenían 90% F1 en SQuAD 1.1 caían a 65-70% F1 en SQuAD 2.0. La adaptación arquitectónica (predecir `[CLS]` como "no answer", umbral $\tau$ sobre el score) descrita en la Section 7.4 del análisis BERT permite a BERT manejar SQuAD 2.0 razonablemente.

### 8.3 Anclaje léxico por construcción

Crítica más sutil. Las preguntas en SQuAD fueron escritas por crowdworkers **mirando el párrafo**. Aunque el interfaz incentivaba "usar tus propias palabras" y deshabilitaba el copy-paste, los humanos tienen un sesgo natural a usar las mismas palabras que acaban de leer. El resultado: muchas preguntas tienen **alta superposición léxica con la oración respuesta**, lo que las hace tractables vía matching simple. Este artefacto se invisibilizó en SQuAD 1.0 pero apareció dramáticamente en:

**Adversarial SQuAD (Jia & Liang 2017, EMNLP)**: los autores (el mismo Percy Liang, en honesta autocrítica) construyeron un test set adversarial agregando una **oración distractor** al final de cada párrafo. La oración distractor contiene palabras similares a las de la pregunta pero responde una pregunta distinta. Ejemplo: si la pregunta original es "What city did Tesla move to in 1880?", se agrega al final del párrafo "Tadakatsu moved to the city of Chicago in 1881". El modelo, entrenado para hacer matching, se distrae con "moved to the city" y responde "Chicago" en vez de la respuesta correcta.

Los resultados de Jia & Liang fueron devastadores. Modelos que tenían 75-85% F1 en SQuAD 1.0 caían a **30-40% F1 en el adversarial set**. BiDAF cayó de 75.5 a 34.3 F1 con un solo distractor concatenado. Match-LSTM cayó de 71.4 a 27.3 F1. Esto demostró que los modelos no entendían las preguntas — explotaban anclaje léxico superficial. La conclusión:

> Models that perform well on SQuAD are largely learning to pick up on surface-level cues such as keyword matching, rather than reasoning about the actual meaning of questions and passages.

Esto motivó toda una línea de investigación en **robustness adversarial** en QA que sigue activa en 2026 (TextAttack, CheckList, Contrast Sets de Gardner et al. 2020, behavioral testing en general).

### 8.4 Solo extractivo

Por diseño, SQuAD excluye preguntas cuya respuesta requiere síntesis, cómputo o reformulación. Esto deja fuera categorías importantes:

- **Cómputo**: "¿qué año cumplió Tesla 50?" requiere aritmética. DROP (Dua et al. 2019) atacó este caso.
- **Lista**: "¿cuáles son los tres ríos más largos?" requiere agregar información. SQuAD permite listas literales pero no agregaciones.
- **Abstractivo / generativo**: "¿en qué se parecen X e Y?" requiere generar texto nuevo. Esto solo se ataca con modelos generativos (T5, GPT-3) sobre datasets como NarrativeQA, ELI5.

La consecuencia indirecta: arquitecturas QA optimizadas para SQuAD (start/end pointer) **no se transfieren naturalmente a QA generativo**. Cuando llegó T5 (2020) reformulando todo NLP como text-to-text, el formato de respuesta cambió y los modelos extractivos quedaron como nicho (re-ranking, retrieval, IE).

### 8.5 Pocos pasos de razonamiento

SQuAD prácticamente no contiene **multi-hop reasoning**: preguntas que requieren combinar evidencia de múltiples documentos. La razón es que el contexto es **un solo párrafo**. Las preguntas multi-hop necesitan retrieval + razonamiento. Datasets sucesores: HotpotQA (Yang et al. 2018), MuSiQue, 2WikiMultiHopQA.

### 8.6 Ruido en el dataset

Análisis posteriores (Pavlick & Kwiatkowski 2019, etc.) encontraron que **~3-5% de SQuAD tiene errores** — preguntas mal escritas, respuestas incorrectas, ambigüedades no resueltas. Esto pone un techo práctico al F1 alcanzable de ~95-96%, no 100%. Los modelos que reportan 94-95% F1 ya están en el régimen de error de anotación.

### 8.7 El leaderboard como arma de doble filo

La existencia del leaderboard público fue brillante para acelerar el campo pero introdujo dos vicios:

1. **Overfitting al test**. Aunque el test estaba "oculto", la posibilidad de subir múltiples modelos y ver el score creó un canal débil de información. Equipos optimizaban en dev y ajustaban en test indirectamente.
2. **Carrera de ensembles**. Los modelos top eran ensembles de 25-50 modelos individuales, sin valor científico claro pero con +1-2 puntos F1. Esto opacaba qué modelo individual era realmente mejor.

---

## 9. Conexión con la clase 20 del Diplomado IA UC

La clase 20 cubre los **encoders contextualizados modernos**: ELMo, BERT, GPT family. SQuAD es **el motor que impulsó este progreso**. Sin un benchmark estandarizado, grande, natural y verificable, la comunidad no habría tenido cómo comparar arquitecturas. La cronología de la sección 7 muestra que cada salto arquitectónico (Match-LSTM, BiDAF, R-Net, QANet, BERT) fue medido contra SQuAD y compitió por el top del leaderboard. SQuAD jugó el rol que ImageNet jugó en visión (Deng et al. 2009): catalizador de progreso vía benchmark común.

La razón concreta por la cual BERT incluye `BertForQuestionAnswering` como cabeza canónica por defecto en la liberación oficial es **porque SQuAD era el benchmark de referencia en ese momento**. La cabeza es minimalista — dos vectores $S, E \in \mathbb{R}^H$ que producen scores de start/end token-wise sobre el contexto — y se aplica idénticamente a SQuAD 1.1, SQuAD 2.0 (con la extensión de no-answer) y a la familia de tareas extractivas downstream.

En el material del site, los fundamentos relevantes son:

- `fundamentos/bert.md`: la cabeza QA de BERT y su entrenamiento sobre SQuAD.
- `fundamentos/embeddings-contextualizados.md`: por qué ELMo y BERT capturan la variación sintáctica que aniquila al baseline LR del paper SQuAD.
- `fundamentos/pretraining-bert.md`: la razón por la cual el pre-training masivo es lo que cierra la brecha al humano en SQuAD.

SQuAD también es el caso de estudio del paradigma **pretrain-finetune en NLP**: modelo pre-entrenado en corpus masivo + cabeza simple + fine-tune end-to-end sobre el dataset target. Este patrón, validado primero en SQuAD, se generalizó luego a GLUE, SuperGLUE, y eventualmente al paradigma instruct/RLHF de los LLMs modernos.

---

## 10. Conexión con el laboratorio 20

El laboratorio 20 del Diplomado incluye, en la celda 13, un ejemplo concreto de cabeza QA usando XLNet:

```python
from transformers import XLNetForQuestionAnswering, XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])
outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
loss, start_scores, end_scores = outputs[:3]
```

Este snippet es **el formato canónico de la cabeza de QA estilo SQuAD**, exportado a XLNet (Yang et al. 2019), que es una extensión de BERT con permutation language modeling. Tres puntos clave a explicar:

### 10.1 `start_positions=[1], end_positions=[3]` — el formato SQuAD

El input es la secuencia tokenizada `[CLS] Hello , my dog is cute [SEP]` (8 tokens con specials). Los **índices 1 y 3 corresponden a "Hello" y a "my"** respectivamente (asumiendo `[CLS]` en posición 0). Esto significa que en este ejemplo de juguete, la respuesta sería el span `Hello, my`.

Esta es **exactamente** la convención de SQuAD: la respuesta es un span del contexto definido por dos índices de token (inicio y fin, ambos inclusive, fin ≥ inicio). El loss durante entrenamiento es:

$$\mathcal{L} = -\log P(\text{start} = s^* \mid x) - \log P(\text{end} = e^* \mid x)$$

donde $s^*$ y $e^*$ son los índices ground truth y las probabilidades vienen de softmax sobre los logits start/end. El loss del ejemplo se calcula contra `start_positions=1, end_positions=3`.

### 10.2 Por qué el ejemplo es sintético

El ejemplo del lab **no es una pregunta real** — es solo "Hello, my dog is cute" sin contexto de pregunta. En SQuAD real, el input sería:

```
[CLS] question_tokens [SEP] context_tokens [SEP]
```

con la pregunta como segmento $A$ y el contexto como segmento $B$. Por ejemplo:

```
[CLS] What causes precipitation to fall ? [SEP]
In meteorology , precipitation is any product of the condensation of atmospheric water vapor that falls under gravity ... [SEP]
```

Y los `start_positions` y `end_positions` apuntarían al token "gravity" en el contexto (digamos índice 28).

El ejemplo del lab demuestra **el API de Hugging Face**, no un caso de uso realista. Para entrenar en SQuAD habría que cargar el dataset (vía `datasets` library), tokenizar pregunta + contexto juntos con `truncation='only_second'` (truncar solo el contexto), mapear las posiciones de char-level del SQuAD a token-level posiciones, y armar batches con un collator apropiado. El paper de Devlin (2018) tiene los hyperparams de fine-tuning de referencia: 3 épocas, batch 32, lr 3e-5.

### 10.3 Decodificación en inferencia: top-k spans

En training, el loss usa los índices reales. En inferencia, hay que **decodificar el span** a partir de los scores. La estrategia estándar:

1. Computar `start_logits[i]` para todos los tokens del contexto.
2. Computar `end_logits[j]` para todos los tokens del contexto.
3. Para todos los pares $(i, j)$ con $0 \le i \le j$ y $j - i \le \text{max\_answer\_length}$ (típicamente 30 tokens), computar el span score:

$$\text{score}(i, j) = \text{start\_logits}[i] + \text{end\_logits}[j]$$

4. Devolver el span con máximo score.

Para mayor robustez, el approach de **n-best decoding** de Devlin et al.:

1. Tomar top-$k$ candidatos de start (digamos top-20) y top-$k$ de end.
2. Generar todos los $k^2 = 400$ pares.
3. Filtrar los inválidos ($j < i$, span demasiado largo, span fuera del contexto).
4. Ordenar por score descendente y devolver el top.

En SQuAD 2.0 con preguntas sin respuesta, se agrega un comparador: si el score del span `[CLS]` (es decir, start=0, end=0) supera al mejor span por más del umbral $\tau$, predecir "no answer". El umbral $\tau$ se calibra en dev.

### 10.4 Por qué esto importa

El laboratorio expone al estudiante al **API canónico de QA en Hugging Face**, que funciona idénticamente para BERT, RoBERTa, XLNet, ALBERT, DeBERTa y casi todos los encoders modernos. El estudiante que entiende este patrón puede:

- Fine-tunear cualquiera de estos modelos sobre SQuAD u otros datasets extractivos (TriviaQA, Natural Questions, MLQA).
- Adaptar el patrón a QA en español: BETO + SQuAD-es, mBERT + xquad.
- Combinar QA extractivo con retrieval para implementar pipelines RAG simples (top-k retrieve + cross-encoder rerank + QA extraction).

El lab termina conectando este conocimiento con los caminos del curso, particularmente el camino de **comprensión y extracción de información**.

---

## 11. Notas de integración al site

Cosas que un material público derivado de este análisis podría incluir y que el resto del material del site no cubre:

1. **Tabla de la cronología de modelos en SQuAD 1.x** (Match-LSTM → BERT) — útil para enseñar la evolución arquitectónica 2016-2018.
2. **Discusión de Adversarial SQuAD (Jia & Liang 2017)** — fundamental para entender por qué los benchmarks "saturados" no implican que el problema esté resuelto.
3. **Comparación de tipos de respuesta y razonamiento** (Tablas 2 y 3 del paper) — material útil para fundamentos de QA.
4. **Conexión cabeza QA de BERT → SQuAD** — explicar por qué BERT trae `BertForQuestionAnswering` por defecto.
5. **Discusión sobre métricas Exact Match y F1** — particularmente por qué BLEU no funciona para respuestas cortas.
6. **Limitación extractive-only y motivación de SQuAD 2.0** — útil para introducir el problema de "answerable vs unanswerable" en QA real.

---

## 12. Lectura recomendada complementaria

- **SQuAD 2.0** (Rajpurkar, Jia & Liang 2018, ACL) — extensión con 50K preguntas adversariales sin respuesta. Pide al modelo abstenerse cuando no hay evidencia.
- **Match-LSTM with Answer Pointer** (Wang & Jiang 2016) — primer modelo neuronal serio en SQuAD. Introduce el patrón pointer network para spans.
- **BiDAF** (Seo, Kembhavi, Farhadi & Hajishirzi 2017, ICLR) — la arquitectura de referencia de QA neuronal pre-BERT. Bidirectional attention flow sin sumarización.
- **QANet** (Yu et al. 2018, ICLR) — primer modelo de QA sin RNNs, basado en CNN + self-attention. Antesala arquitectónica de BERT.
- **A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task** (Chen, Bolton & Manning 2016) — demuestra que cloze-style datasets están casi saturados con métodos triviales. Justificación implícita de SQuAD.
- **Adversarial Examples for Evaluating Reading Comprehension Systems** (Jia & Liang 2017, EMNLP) — el paper que destrozó los modelos top de SQuAD agregando una sola oración distractor.
- **TriviaQA** (Joshi et al. 2017, ACL) — dataset alternativo con preguntas más naturales y contextos múltiples (Wikipedia + Web).
- **Natural Questions** (Kwiatkowski et al. 2019, TACL) — dataset de preguntas reales de Google Search con contexto Wikipedia. Más difícil y realista que SQuAD.
- **HotpotQA** (Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning 2018, EMNLP) — QA multi-hop sobre múltiples documentos Wikipedia. Va más allá de SQuAD en complejidad de razonamiento.
- **DROP** (Dua et al. 2019, NAACL) — QA con razonamiento numérico (suma, conteo, máximo). Va más allá de SQuAD en tipos de respuesta.
- **CheckList: Beyond Accuracy** (Ribeiro et al. 2020, ACL) — framework de behavioral testing para modelos NLP. Sucesor metodológico de Jia & Liang.
