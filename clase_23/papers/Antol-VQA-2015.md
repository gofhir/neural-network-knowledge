---
título: "VQA: Visual Question Answering"
autores: "Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence Zitnick, Dhruv Batra, Devi Parikh"
venue: "IEEE International Conference on Computer Vision (ICCV) 2015"
año: 2015
arxiv: "1505.00468"
link: "https://arxiv.org/abs/1505.00468"
sitio_oficial: "https://www.visualqa.org"
clase: 23
tema: "Visual Question Answering"
---

# VQA: Visual Question Answering

## Frontmatter de cita

- **Título:** VQA: Visual Question Answering
- **Autores:** Aishwarya Agrawal\*, Jiasen Lu\*, Stanislaw Antol\* (Virginia Tech), Margaret Mitchell (Microsoft Research, Redmond), C. Lawrence Zitnick (Facebook AI Research), Dhruv Batra y Devi Parikh (Georgia Institute of Technology). Los tres primeros autores contribuyeron por igual.
- **Venue:** International Conference on Computer Vision (ICCV) 2015.
- **Año:** 2015 (la versión arXiv que aquí se analiza es la v7, fechada el 27 de octubre de 2016, ya con la sección del VQA Challenge y los leaderboards actualizados).
- **arXiv ID:** 1505.00468 (cs.CL).
- **Enlace:** [https://arxiv.org/abs/1505.00468](https://arxiv.org/abs/1505.00468)
- **Sitio oficial:** [www.visualqa.org](https://www.visualqa.org) — demo en CloudCV: [http://cloudcv.org/vqa](http://cloudcv.org/vqa)

Este es el paper fundacional que define la tarea de *Visual Question Answering* tal como la conocemos hoy, introduce el dataset VQA v1, propone la métrica de consenso humano y establece la familia de baselines LSTM+CNN que dominaría el subcampo durante varios años. Es la base directa de todo lo que se enseña en la clase 23 del curso (VQA e Image Captioning) y el ancestro del dataset que usa Pythia.

## Contexto histórico: por qué surge VQA en 2015

A mediados de 2015, la visión por computador vivía un momento de euforia tras el éxito de las CNN profundas (AlexNet en 2012, VGGNet y GoogLeNet en 2014). La tarea de moda en la intersección de visión y lenguaje era el *image captioning*: generar una oración que describiera una imagen. Trabajos como Show and Tell de Google, los modelos de Karpathy y Fei-Fei, y los de Microsoft habían producido resultados llamativos en pocos meses. La narrativa dominante era que el captioning era "un paso hacia la resolución de la IA", porque combinaba visión, NLP y razonamiento.

Los autores de VQA observan algo incómodo en esa narrativa. Como dicen en la introducción, "el estado del arte actual demuestra que una comprensión a nivel de escena, gruesa, de una imagen emparejada con estadísticas de n-gramas de palabras basta para generar descripciones razonables". En otras palabras, un modelo de captioning puede producir descripciones plausibles **sin entender realmente la imagen**, explotando regularidades estadísticas del lenguaje (las imágenes de playas suelen mencionar arena, las de cocinas mencionan platos). Esto sugiere que el captioning **no es tan "AI-complete" como se cree**.

Aquí aparece la pregunta de diseño que motiva todo el paper: ¿qué hace que una tarea sea verdaderamente "AI-complete"? Los autores proponen tres criterios para una tarea ideal de próxima generación:

1. Debe requerir **conocimiento multimodal** más allá de un solo subdominio (no basta con visión, no basta con lenguaje).
2. Debe tener una **métrica de evaluación cuantitativa bien definida** que permita seguir el progreso. El captioning falla aquí: evaluar una descripción libre es notoriamente difícil (BLEU, METEOR, CIDEr correlacionan mal con el juicio humano).
3. Implícitamente, debe ser **automáticamente evaluable**, no requerir jueces humanos costosos cada vez.

VQA cumple los tres. Es una versión moderna del **Test de Turing visual**: en vez de una conversación abierta, se le hace al sistema una pregunta sobre una imagen y se evalúa si responde como lo haría un humano. La elegancia del diseño está en que, como las respuestas tienden a ser cortas (una o pocas palabras), la tarea sigue siendo *abierta* y rica, pero **automáticamente evaluable** mediante coincidencia exacta con respuestas humanas. Esto resuelve simultáneamente la riqueza del problema y la tratabilidad de su medición, algo que el captioning no lograba.

El paper también enmarca VQA en aplicaciones reales concretas: asistir a personas con discapacidad visual (que pueden fotografiar su entorno y preguntar) o permitir que analistas de inteligencia consulten activamente contenido visual. Estas aplicaciones justifican que las preguntas y respuestas sean de forma libre y abierta, no encajonadas en plantillas.

## Definición de la tarea

Formalmente, un sistema de VQA recibe como entrada:

- una **imagen** $I$, y
- una **pregunta en lenguaje natural** $q$ sobre esa imagen, de forma libre y abierta (*free-form, open-ended*),

y debe producir como salida una **respuesta en lenguaje natural** $a$, también de forma libre.

El paper define **dos modalidades de respuesta** para evaluación:

1. **Open-ended (abierta):** el sistema genera una respuesta libre. En la práctica, los modelos del paper la implementan como una clasificación sobre las $K$ respuestas más frecuentes, pero conceptualmente la respuesta puede ser cualquier cadena.
2. **Multiple-choice (selección múltiple):** el sistema elige entre 18 respuestas candidatas predefinidas para cada pregunta. Esta variante es más fácil de evaluar (basta elegir de una lista) y útil para algoritmos que no pueden generar texto libre.

Las preguntas resultan ser sorprendentemente diversas en el tipo de capacidad que exigen. El paper las ilustra con ejemplos que cubren todo el espectro de la IA:

- **Reconocimiento de objetos de grano fino:** "¿Qué tipo de queso tiene la pizza?"
- **Detección de objetos:** "¿Cuántas bicicletas hay?"
- **Reconocimiento de actividad:** "¿Está llorando este hombre?"
- **Razonamiento sobre base de conocimiento:** "¿Es una pizza vegetariana?"
- **Razonamiento de sentido común:** "¿Esta persona está esperando compañía?", "¿Tiene esta persona visión 20/20?"

Para análisis y evaluación, las respuestas se agrupan en tres grandes **tipos de respuesta**, una taxonomía que se volvió canónica en todo el campo:

| Tipo de respuesta | Descripción | Ejemplo de pregunta |
|---|---|---|
| **Yes/No** | Respuesta binaria (a veces "maybe") | "¿Está rota la pizza?" |
| **Number** | Una cantidad numérica | "¿Cuántas porciones de pizza hay?" |
| **Other** | Todo lo demás (colores, objetos, lugares...) | "¿Qué color son sus ojos?" |

Esta clasificación tripartita no es arbitraria: refleja que cada tipo activa mecanismos cognitivos distintos (verificación binaria, conteo, reconocimiento abierto) y que los modelos rinden muy diferente en cada uno. Casi todas las tablas de resultados del paper y de la literatura posterior reportan accuracy desagregada en estas tres columnas más el "All".

## El dataset VQA v1

El dataset combina dos fuentes de imágenes, una decisión metodológica clave:

**Imágenes reales (MS COCO).** Se usan las imágenes del dataset Microsoft Common Objects in Context (MS COCO), elegido porque sus escenas contienen múltiples objetos y contexto rico, ideal para preguntas interesantes. En total:

- **204.721 imágenes** de MS COCO (123.287 de train+val y 81.434 de test).
- Cada imagen ya venía con cinco *captions* de una sola oración.

**Escenas abstractas (clipart).** Los autores crean además un nuevo dataset de **50.000 escenas abstractas** generadas con clipart. La motivación es brillante: estas escenas eliminan la necesidad de visión de bajo nivel (segmentación, detección ruidosa) y permiten a los investigadores enfocarse en el **razonamiento de alto nivel** que VQA exige, sin que el cuello de botella sea el reconocimiento de píxeles. El conjunto incluye 20 modelos humanos tipo "paperdoll" (distintos géneros, razas, edades, con 8 expresiones), más de 100 objetos y 31 animales en varias poses, con extremidades ajustables para poses continuas.

**Volumen total de preguntas y respuestas.** El proceso de recolección produjo cifras a gran escala:

- **Tres preguntas por imagen/escena**, recolectadas vía Amazon Mechanical Turk (AMT).
- En total, **más de ~0,76 M preguntas** (614.163 para imágenes reales sobre las 204.721 imágenes COCO, y 150.000 para las 50.000 escenas abstractas).
- **~10 M respuestas** en total (7.984.119 respuestas para imágenes reales — incluyendo las que se dieron sin ver la imagen — y 1.950.000 para escenas abstractas).

**Las "10 respuestas por pregunta".** Esta es una de las decisiones de diseño más influyentes del paper. Para cada pregunta se recogieron **10 respuestas de 10 trabajadores únicos**, asegurando además que quien respondía no fuera quien escribió la pregunta. ¿Por qué 10? Porque las preguntas abiertas producen **discrepancias legítimas**: ante "¿de qué color es la mesa?", un humano puede decir "white", otro "tan", otro "off-white", y los tres tienen razón. Capturar 10 respuestas permite modelar esta distribución de respuestas válidas y construir una métrica robusta basada en consenso (ver más abajo). A los sujetos se les pidió responder con frases breves, no oraciones completas, y de forma objetiva ("evita lenguaje conversacional o tu opinión").

Además se les preguntó su **confianza** ("¿Crees que pudiste responder correctamente?", con opciones "no", "maybe", "yes"), lo que permite el análisis de acuerdo inter-humano descrito más adelante.

**Splits.** Para imágenes reales se siguen los splits de COCO: train/val/test, con el test subdividido en test-dev (para depuración, envíos ilimitados), test-standard (el "oficial" para reportar en papers, con leaderboard público), test-challenge (para determinar ganadores del challenge) y test-reserve (para detectar overfitting; si los puntajes en test-standard y test-reserve difieren mucho, salta una alerta roja). Las escenas abstractas usan splits 20K/10K/20K sin subsplits.

## Análisis del dataset

El paper dedica una sección completa a entender qué hay realmente dentro del dataset, y los hallazgos son reveladores.

**Tipos de pregunta.** Agrupando las preguntas por sus primeras palabras (las primeras cuatro), emerge una enorme variedad: "What is...", "Is there...", "How many...", "Does the...", "What color...", "Which...", etc. Un hallazgo importante: **la distribución de tipos de pregunta es muy similar entre imágenes reales y escenas abstractas**, lo que valida que las escenas abstractas elicitan el mismo tipo de razonamiento que las reales. Las preguntas "What is..." destacan por tener la mayor diversidad de respuestas posibles.

**Longitud de preguntas.** La mayoría de las preguntas tienen entre **4 y 10 palabras**, con un pico alrededor de 5-6 palabras. La distribución de longitudes es casi idéntica entre imágenes reales y abstractas.

**Longitud de respuestas.** Aquí está el dato que hace toda la tarea tratable: las respuestas son extremadamente cortas.

| Longitud | Imágenes reales | Escenas abstractas |
|---|---|---|
| 1 palabra | 89,32 % | 90,51 % |
| 2 palabras | 6,91 % | 5,89 % |
| 3 palabras | 2,74 % | 2,49 % |

Que el **89,32 %** de las respuestas en imágenes reales sean de una sola palabra es justamente lo que permite usar coincidencia exacta como métrica fiable (ver siguiente sección). Hay 23.234 respuestas únicas de una palabra para imágenes reales y 3.770 para escenas abstractas. Los autores advierten contra la tentación de pensar que la brevedad facilita el problema: las preguntas requieren razonamiento complejo para llegar a respuestas "engañosamente simples".

**Distribución de respuestas por tipo.** Las preguntas "yes/no" se responden con "yes"/"no" (a veces "maybe") y constituyen el **38,37 %** de las respuestas en imágenes reales y **40,66 %** en abstractas. Dentro de ellas hay un **sesgo hacia "yes"** (58,83 % en reales, 55,86 % en abstractas). Las preguntas "How many..." son numéricas: el **12,31 %** (reales) y **14,48 %** (abstractas) de las preguntas son de tipo "number", y entre ellas el "2" es la respuesta más frecuente (26,04 % en reales, 39,85 % en abstractas). Este sesgo en la distribución de respuestas es precisamente la semilla del problema de *language priors* que se discute más adelante.

**¿Se necesita la imagen, o basta el sentido común?** El paper aborda directamente la pregunta crítica: ¿cuántas preguntas pueden responderse sin mirar la imagen, solo con sentido común? Se hicieron estudios AMT pidiendo a sujetos responder **sin ver la imagen**. El resultado es contundente: para preguntas que no son "yes/no", los humanos sin imagen solo aciertan el **~21 %** de las veces. Esto demuestra que entender la información visual es crítico y que el sentido común por sí solo no basta. (El detalle incómodo es que las máquinas, como veremos, hacen mucho mejor que ese 21 % humano sin imagen, justamente porque explotan sesgos estadísticos del dataset).

Se hizo también un estudio de la **edad percibida** necesaria para responder cada pregunta: toddler (3-4), younger child (5-8), older child (9-12), teenager (13-17), adult (18+). La distribución: toddler 15,3 %, younger child 39,7 %, older child 28,4 %, teenager 11,2 %, adult 5,5 %. La edad promedio percibida es 8,92 años y el grado de sentido común promedio es 31,01 sobre 100. Las preguntas juzgadas como "de adulto" requieren conocimiento especializado; las "de toddler" son más genéricas.

**Acuerdo inter-humano.** ¿Concuerdan los humanos entre sí? El paper mide esto con la métrica de evaluación (Tabla 1, fila Question + Image): los humanos concuerdan en **83,30 %** para imágenes reales y **87,49 %** para escenas abstractas. En promedio cada pregunta tiene **2,70 respuestas únicas** (reales) y 2,39 (abstractas). El acuerdo es mucho mayor (>95 %) en preguntas "yes/no" y menor (<76 %) en las demás, en parte porque la coincidencia es de cadena exacta y no contempla sinónimos, plurales, etc. — un humano que dice "couch" y otro que dice "sofa" cuentan como en desacuerdo.

**Captions vs. preguntas.** Una pregunta natural: ¿bastaría dar un caption genérico para responder las preguntas? El paper lo prueba (Tabla 1, fila Question + Caption): dar el caption en vez de la imagen mejora respecto a no tener nada, pero queda muy por debajo de tener la imagen real (por ejemplo 57,47 % vs 83,30 % en reales). Un test de Kolmogórov-Smirnov sobre las distribuciones de sustantivos, verbos y adjetivos confirma que captions y preguntas+respuestas capturan información **estadísticamente distinta** ($p < .001$). Los captions describen lo genérico de la escena; las preguntas buscan detalles específicos. Son complementarios, no equivalentes.

## Métrica de evaluación

Esta es una de las contribuciones más duraderas del paper. Para la tarea open-ended, la accuracy de una respuesta predicha se define como:

$$
\text{accuracy} = \min\left(\frac{\#\,\text{humanos que dieron esa respuesta}}{3},\ 1\right)
$$

Es decir, una respuesta se considera **100 % correcta si al menos 3 de los 10 anotadores dieron exactamente esa respuesta**. Si solo 1 de los 10 la dio, recibe 1/3 ≈ 0,33; si 2 la dieron, 2/3 ≈ 0,67; si 3 o más, 1,0.

Antes de comparar, todas las respuestas se normalizan: minúsculas, números convertidos a dígitos, y se eliminan puntuación y artículos.

**¿Por qué se diseñó así?** El razonamiento es profundo:

1. **Robustez ante discrepancias legítimas.** Como vimos, ante "¿de qué color es la mesa?" varios colores pueden ser correctos. Exigir coincidencia con *una* respuesta de referencia sería injusto. Usar las 10 respuestas y requerir consenso de 3 captura la idea de "una respuesta es correcta si una fracción razonable de humanos coincide".

2. **Evita métricas blandas problemáticas.** Los autores rechazan explícitamente métricas de similitud semántica como Word2Vec, "porque a menudo agrupan palabras que queremos distinguir, como 'left' y 'right'". También rechazan BLEU y ROUGE (de traducción automática y summarization) porque solo son fiables para oraciones de múltiples palabras: dado que el 89,32 % de las respuestas son de una sola palabra, no hay coincidencias de n-gramas de orden alto y estas métricas degeneran a coincidencia exacta de todos modos; además correlacionan mal con el juicio humano.

3. **Consistencia con el acuerdo humano.** La métrica se diseña para que el "techo" humano sea medible con la misma fórmula. Para que las accuracies de máquina sean comparables con las "human accuracies", las accuracies de máquina se promedian sobre los $\binom{10}{9}$ conjuntos de anotadores (es decir, se evalúa contra cada subconjunto de 9 anotadores y se promedia), evitando el sesgo de que el modelo "vio" sus propias referencias.

Para la tarea multiple-choice, se construyen **18 respuestas candidatas** por pregunta, a partir de cuatro fuentes: **Correct** (la respuesta más común de las 10), **Plausible** (respuestas plausibles pero incorrectas, generadas pidiendo a 3 sujetos que respondan *sin ver la imagen* — así son plausibles por sentido común pero no por la imagen), **Popular** (las 10 respuestas más populares del dataset, p. ej. "yes", "no", "2", "1", "white"...; incluirlas dificulta que el algoritmo infiera el tipo de pregunta a partir de las opciones), y **Random** (respuestas correctas de preguntas aleatorias). Las 18 son únicas, pero como 10 sujetos respondieron cada pregunta, puede haber más de una opción con accuracy no nula.

## Modelos baseline

El paper establece una batería de baselines deliberadamente diseñada para revelar **de dónde viene el rendimiento** de cualquier modelo de VQA. Esto es metodológicamente ejemplar.

**Baselines simples:**

- **random:** elige al azar entre las top 1K respuestas.
- **prior ("yes"):** siempre responde "yes" (la respuesta más popular). Sorprendentemente difícil de batir en algunos cortes.
- **per Q-type prior:** responde la respuesta más popular *para ese tipo de pregunta*. Para "How many", responder "2"; para "What color", responder "white"; etc.
- **nearest neighbor:** dada una imagen-pregunta de test, encuentra las $K$ preguntas+imágenes más cercanas en train y devuelve la respuesta de referencia más frecuente entre ellas.

**Canales del modelo neuronal (un modelo de 2 canales: visión + lenguaje):**

*Canal de imagen:*
- **I:** las activaciones de la última capa oculta de **VGGNet**, un embedding de 4096 dimensiones.
- **norm I:** las mismas activaciones pero normalizadas en $\ell_2$.

*Canal de pregunta:*
- **Bag-of-Words Question (BoW Q):** las 1000 palabras más frecuentes más un bag-of-words de las top 10 primera/segunda/tercera palabras de las preguntas (30 dims), concatenado para dar 1030 dims. Aprovecha la fuerte correlación entre las primeras palabras y la respuesta.
- **LSTM Q:** una LSTM de una capa oculta produce un embedding de 1024 dims. Cada palabra se codifica con un embedding de 300 dims + tanh antes de entrar a la LSTM.
- **deeper LSTM Q:** una LSTM de **dos capas** ocultas que produce un embedding de 2048 dims (concatenando estado de celda y estado oculto de cada capa, cada uno 512 dims), luego proyectado a 1024 dims.

**El mejor modelo: deeper LSTM Q + norm I.** Es la arquitectura icónica de este paper (Figura 8). Su mecánica:

1. La pregunta "How many horses are in this image?" pasa palabra a palabra por la **LSTM de dos capas**, produciendo un embedding de 1024 dims.
2. La imagen pasa por **VGGNet**; se toman las 4096 activaciones de la última capa oculta, se **normalizan en $\ell_2$**, y se proyectan a 1024 dims con una capa fully-connected + tanh para igualar el espacio de la pregunta.
3. **Fusión por producto elemento a elemento** (*point-wise / element-wise multiplication*) de los dos embeddings de 1024 dims. Este es el corazón del modelo: la multiplicación elemento a elemento actúa como una interacción multiplicativa entre las dimensiones de lenguaje y visión, mucho más expresiva que una simple concatenación.
4. El embedding fusionado pasa por un **MLP** (2 capas ocultas, 1000 unidades, dropout 0,5, tanh) y un **softmax sobre $K = 1000$ respuestas** posibles (las 1000 más frecuentes, que cubren el 82,67 % de las respuestas de train+val).

Todo el modelo se entrena end-to-end con **cross-entropy**. Importante: los parámetros de VGGNet se **congelan** (preentrenados en ImageNet) y no se fine-tunean. Esto significa que el canal de visión es un extractor de features fijo, lo que limita cuánto puede adaptarse a la tarea — un detalle relevante para entender por qué el modelo se apoya tanto en el lenguaje.

## Resultados experimentales

Los resultados (Tabla 2, test-dev imágenes reales) cuentan una historia que sacudió al campo.

| Método | Open-Ended All | OE Yes/No | OE Number | OE Other | Multiple-Choice All |
|---|---|---|---|---|---|
| prior ("yes") | 29,66 | 70,81 | 00,39 | 01,15 | 29,66 |
| per Q-type prior | 37,54 | 71,03 | 35,77 | 09,38 | 39,45 |
| nearest neighbor | 42,70 | 71,89 | 24,36 | 21,94 | 48,49 |
| **I** (solo imagen) | 28,13 | 64,01 | 00,42 | 03,77 | 30,53 |
| **BoW Q** (solo lenguaje) | 48,09 | 75,66 | 36,70 | 27,14 | 53,68 |
| **LSTM Q** (solo lenguaje) | 48,76 | 78,20 | 35,68 | 26,59 | 54,75 |
| BoW Q + I | 52,64 | 75,55 | 33,67 | 37,37 | 58,97 |
| LSTM Q + I | 53,74 | 78,94 | 35,24 | 36,42 | 57,17 |
| **deeper LSTM Q + norm I** | **57,75** | **80,50** | **36,77** | **43,08** | **62,70** |

Y la comparación humano vs. máquina (test-standard) para el mejor modelo:

- **deeper LSTM Q + norm I:** 58,16 % open-ended / 63,09 % multiple-choice (test-standard).
- **Humanos (Question + Image):** 83,30 % en imágenes reales (Tabla 1).

La brecha de **~25 puntos** entre máquina y humano deja claro, en 2015, que VQA es un problema lejos de estar resuelto — exactamente la propiedad "AI-complete" que los autores buscaban.

**El hallazgo que más impactó: los modelos "ciegos" funcionan demasiado bien.** Observa la tabla con cuidado:

- El modelo de **solo imagen (I)** rinde **peor** (28,13 % OE) que incluso el prior trivial "yes" (29,66 %). La imagen sola, sin pregunta, es casi inútil.
- Pero los modelos de **solo lenguaje** (per Q-type prior, BoW Q, LSTM Q) funcionan **sorprendentemente bien**: LSTM Q alcanza **48,76 %** open-ended ignorando completamente la imagen, superando incluso al baseline de nearest neighbor (42,70 %) que sí usa la imagen.
- Agregar la imagen al mejor modelo de lenguaje solo sube de ~48,76 % a 57,75 %: la imagen aporta, pero menos de lo que uno esperaría de una tarea "visual".

Desagregando por tipo de pregunta (Tabla 3) se ve dónde ayuda la imagen y dónde no:

- En preguntas que requieren razonamiento ("Is the...", "How many..."), las features visuales a nivel de escena **casi no aportan información**: la accuracy con y sin imagen es casi igual.
- En preguntas que pueden responderse con información a nivel de escena ("What sport...", "What animal...") **sí** hay mejora con imagen.

El modelo es bueno reconociendo objetos comunes ("wii", "tennis", "bathroom") pero malo contando (pobre en "5", "6", "8", "10"). El conteo es un talón de Aquiles persistente del campo.

**Estudios de ablación.** El paper ablaciona cuidadosamente su mejor modelo (Tabla 4) para entender qué componente aporta cada cosa, otro ejercicio de rigor metodológico:

- **Sin normalización $\ell_2$ de la imagen:** la normalización aporta +0,16 % open-ended y +0,24 % multiple-choice. Pequeño pero consistente.
- **Concatenación en vez de producto elemento a elemento:** la **fusión multiplicativa supera a la concatenación en +0,95 % open-ended y +1,24 % multiple-choice**, con la mitad de parámetros en la capa siguiente. Este resultado justifica empíricamente la decisión de diseño central: la interacción multiplicativa entre lenguaje y visión es genuinamente mejor que pegar los vectores.
- **Tamaño de $K$:** $K = 1000$ supera a $K = 500$ (+0,82 % OE), y $K = 2000$ supera ligeramente a $K = 1000$ (+0,40 % OE). Hay rendimientos decrecientes al ampliar el vocabulario de respuestas.
- **Truncar el vocabulario de preguntas:** quitar palabras que aparecen menos de 5 u 11 veces (reduciendo el vocabulario hasta 65-76 %) **no perjudica** el rendimiento (incluso mejora marginalmente), lo que indica que las palabras raras de las preguntas aportan poca señal.
- **Dataset filtrado:** entrenar solo con respuestas de alta confianza empeora el modelo (-1,13 % OE), sugiriendo que el ruido de las respuestas dudosas en realidad ayuda como regularización o cobertura.

Estas ablaciones, leídas hoy, confirman lo modesto que era el aprovechamiento de la imagen: ningún ajuste del canal visual mueve la aguja tanto como cambiar el canal de lenguaje, otra señal del peso desproporcionado de los priors lingüísticos.

## El problema de los language priors

Este es, retrospectivamente, el legado teórico más importante del paper, aunque los autores lo presentan casi de pasada. El hallazgo de que un modelo "ciego" alcanza casi el 49 % de accuracy revela una patología estructural del dataset.

Los autores escriben: "nuestros resultados cuantitativos y análisis sugieren que esto podría deberse a que el modelo de lenguaje explota sutiles regularidades estadísticas sobre la distribución de respuestas (p. ej., '¿De qué color es el plátano?' puede responderse con 'yellow' sin mirar la imagen)".

¿Qué está pasando exactamente? El dataset, construido con humanos, hereda los **sesgos del mundo y del lenguaje**:

- Los plátanos suelen ser amarillos, así que "¿de qué color es el plátano?" → "yellow" casi siempre acierta.
- Las preguntas "How many..." se responden "2" el 26 % de las veces.
- Las preguntas "yes/no" tienen sesgo hacia "yes" (58,83 %).
- "Is there a..." casi siempre tiene respuesta "yes" (porque la gente pregunta por cosas que están).

Un modelo puede **memorizar la distribución condicional $P(\text{respuesta} \mid \text{tipo de pregunta})$** y "hacer trampa": responder bien sin nunca mirar la imagen. La tarea, que se diseñó como "visual", puede resolverse parcialmente como un problema de NLP puro de priors estadísticos. Esto es grave: un modelo puede obtener buen accuracy **sin haber aprendido a ver**, lo que invalida la métrica como medida de comprensión visual genuina.

Este problema es exactamente lo que motiva a **Goyal et al. (2017)** a crear **VQA v2**, que construye, para cada pregunta, **pares de imágenes complementarias** con respuestas *distintas* (p. ej., una pregunta "¿de qué color es...?" emparejada con dos imágenes donde la respuesta es diferente). Al balancear el dataset, el prior de lenguaje deja de funcionar: un modelo ciego ya no puede acertar porque la misma pregunta tiene respuestas opuestas según la imagen. VQA v2 obliga a los modelos a *mirar*.

La clase 23 destaca este problema precisamente al presentar **Pythia** (el modelo ganador del VQA Challenge 2018), que se entrena sobre VQA v2 justamente para mitigar el atajo de los priors. La línea conceptual va directa: el paper de Antol 2015 *descubre* el problema → VQA v2 lo *corrige en los datos* → Pythia y los modelos modernos lo *combaten en la arquitectura* (atención, features de detección tipo bottom-up). Entender el problema de language priors es, por tanto, entender por qué existe toda la maquinaria moderna de VQA.

## Limitaciones reconocidas por los autores

El paper es honesto sobre sus límites:

1. **Las preguntas son abiertas y no específicas de dominio.** Para aplicaciones concretas (asistencia a personas ciegas, deportes) convendría recolectar preguntas específicas del dominio. Curiosamente, observan que las preguntas reales de usuarios ciegos (Bigham et al. [3]) **rara vez se responden con captions**, lo que refuerza la necesidad de VQA real.

2. **La métrica de coincidencia exacta no maneja sinónimos ni plurales.** "couch"/"sofa", "1"/"one" cuentan como distintas. Los autores reconocen que la determinación automática de sinónimos es difícil porque la granularidad correcta de una respuesta varía según la pregunta. Esto deprime artificialmente el acuerdo inter-humano en preguntas no-binarias (<76 %).

3. **El conteo es débil.** El mejor modelo rinde mal en cantidades altas ("5", "6", "10"). El conteo requiere razonamiento espacial que las features globales de VGGNet no capturan.

4. **VGGNet congelado.** El canal visual es un extractor fijo; el modelo no puede adaptar las features visuales a la tarea, limitando el aprovechamiento de la imagen (y contribuyendo, indirectamente, a la dependencia del lenguaje).

5. **Sesgos del dataset (language priors).** Aunque los autores los señalan, no los corrigen en v1 — esa corrección queda para VQA v2. En el paper se presenta más como una observación interesante que como un defecto a remediar, algo que la comunidad reevaluaría con dureza en los años siguientes.

## Impacto y legado

El impacto de este paper es difícil de exagerar; es uno de los más citados de la intersección visión-lenguaje (decenas de miles de citas).

**El VQA Challenge y Workshop.** Los autores montaron un servidor de evaluación y un **challenge anual con workshop** (el primero en CVPR 2016), con leaderboards públicos para open-ended y multiple-choice, en imágenes reales y abstractas. Esto institucionalizó VQA como subcampo con métrica común y competencia anual, replicando el modelo de éxito de ImageNet. Los leaderboards de octubre de 2016 (Tabla 5, Figura 13) ya mostraban decenas de equipos (snubi-naverlabs, MM_PaloAlto, LV-NUS, etc.) compitiendo, con el mejor superando el 60 % open-ended.

**Definió un subcampo entero.** VQA pasó de ser una idea a un área de investigación con cientos de papers por año: mecanismos de atención (Stacked Attention Networks, 2016), atención co-atencional (Hierarchical Co-Attention, Lu et al. 2016), fusión bilineal (MCB, MLB, MUTAN), features bottom-up basadas en detección (Anderson et al. 2018), módulos de razonamiento (Neural Module Networks), y datasets de razonamiento composicional (CLEVR).

**Conexión a VQA v2 (Goyal 2017) y Pythia.** Como se explicó, VQA v2 corrige los language priors balanceando el dataset. **Pythia** (Jiang et al. 2018), ganador del VQA Challenge 2018, se construye sobre VQA v2 con features bottom-up de Faster R-CNN, atención, y un pipeline de entrenamiento muy afinado. Es el modelo que la clase 23 usa como caso de estudio moderno, y su linaje conceptual se remonta directamente a este paper de 2015.

**Conexión a los VLMs modernos.** La línea evolutiva es nítida:

- **2015 — Antol VQA:** define la tarea, dataset, métrica, baseline LSTM+CNN con fusión multiplicativa.
- **2016-2018 — Atención y fusión:** SAN, co-atención, MCB, bottom-up attention, Pythia.
- **2019-2021 — Transformers multimodales:** ViLBERT, LXMERT, UNITER, OSCAR preentrenan en visión+lenguaje y hacen fine-tuning en VQA.
- **2022+ — VLMs generativos:** BLIP / BLIP-2, Flamingo, y modelos comerciales como **GPT-4V**, **Gemini** y **Claude** con visión, que responden preguntas sobre imágenes de forma conversacional, en lenguaje natural, sin la restricción de las top-K respuestas.

Lo notable es que la tarea que estos modelos gigantes resuelven es, conceptualmente, la misma que Antol et al. definieron en 2015. El benchmark VQA sigue siendo una de las pruebas estándar para evaluar VLMs modernos. El paper no solo creó un dataset: creó una **forma de pensar la comprensión visual** como un problema de pregunta-respuesta.

## Conexión con la clase 23

Este paper es la **piedra fundacional** de todo lo que la profesora Bianca Del Solar enseña sobre VQA en la clase 23 (VQA + Image Captioning). Los puntos de conexión directa:

1. **Definición de la tarea.** Cuando la clase define VQA como "dada una imagen y una pregunta, producir una respuesta en lenguaje natural", está usando exactamente la formulación de Antol et al. 2015. La taxonomía yes/no / number / other que aparece en cualquier slide de VQA proviene de aquí.

2. **El dataset que usa Pythia deriva de aquí.** Pythia se entrena sobre **VQA v2**, que es la versión balanceada del dataset VQA v1 introducido en este paper. Las mismas imágenes COCO, la misma estructura de 10 respuestas por pregunta, la misma métrica de consenso de 3/10. Sin este paper, no hay Pythia.

3. **El problema de los language priors es un tema central de la clase.** La clase 23 destaca por qué los modelos pueden "hacer trampa" usando solo el lenguaje, y por qué se necesitó VQA v2 + arquitecturas con atención. Ese problema **se descubre en este paper** (el modelo ciego con 49 %). Comprender este resultado es comprender la motivación de todo el pipeline moderno que la clase enseña.

4. **La métrica de evaluación.** Cualquier ejercicio o notebook de la clase que reporte "VQA accuracy" usa la fórmula $\min(\#/3, 1)$ de este paper. Es el estándar de facto del campo.

5. **El baseline LSTM + CNN.** La arquitectura de dos canales (pregunta vía LSTM/embedding, imagen vía CNN, fusión, MLP, softmax) es el esqueleto pedagógico desde el cual la clase construye hacia modelos con atención. Entender la fusión por producto elemento a elemento de Antol es entender el punto de partida que la atención vino a mejorar.

En síntesis: este paper define el **qué** (la tarea), el **con qué** (dataset y métrica), el **cómo básico** (baseline neuronal) y el **problema clave** (language priors) que estructuran toda la unidad de VQA del curso.

## Notas y enlaces

**Papers relacionados clave:**

- **Goyal, Khot, Summers-Stay, Batra, Parikh (2017)** — "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering" (CVPR 2017). Introduce **VQA v2** balanceado con pares de imágenes complementarias. Corrección directa del problema de language priors descubierto aquí.
- **Jiang, Natarajan, Chen, Bansal, Parikh (2018)** — **Pythia v0.1**, ganador del VQA Challenge 2018, sobre VQA v2 con features bottom-up. El modelo moderno de la clase 23.
- **Anderson et al. (2018)** — "Bottom-Up and Top-Down Attention for Image Captioning and VQA" (CVPR 2018). Features de detección (Faster R-CNN) + atención, base de Pythia.
- **Yang et al. (2016)** — "Stacked Attention Networks for Image Question Answering". Primera atención visual influyente para VQA.
- **Lu et al. (2016)** — "Hierarchical Question-Image Co-Attention for VQA".
- **Johnson et al. (2017)** — "CLEVR", dataset diagnóstico de razonamiento composicional, complementario a VQA.
- **Li et al. (2022)** — "BLIP" / "BLIP-2", VLMs generativos modernos evaluados en VQA.

**Recursos:**

- Sitio oficial y challenge: [www.visualqa.org](https://www.visualqa.org) y [visualqa.org/challenge.html](https://visualqa.org/challenge.html)
- Demo en CloudCV: [cloudcv.org/vqa](http://cloudcv.org/vqa)
- Interfaz de recolección COCO-UI: [github.com/tylin/coco-ui](https://github.com/tylin/coco-ui)
- Dataset base de imágenes: MS COCO ([cocodataset.org](https://cocodataset.org))

**Cifras clave para recordar:**

- ~204.721 imágenes COCO + 50.000 escenas abstractas.
- ~0,76 M preguntas, ~10 M respuestas.
- 3 preguntas por imagen, 10 respuestas por pregunta.
- 89,32 % de respuestas son de una sola palabra (imágenes reales).
- Métrica: $\min(\#\text{humanos}/3, 1)$.
- Mejor modelo (deeper LSTM Q + norm I): 58,16 % open-ended / 63,09 % multiple-choice (test-standard); techo humano 83,30 %.
- Modelo "ciego" (LSTM Q sin imagen): 48,76 % open-ended → evidencia de los language priors.
