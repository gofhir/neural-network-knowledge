# LAMBADA: predicción de palabra que exige un contexto de discurso amplio

> Análisis técnico del paper *The LAMBADA dataset: Word prediction requiring a broad discourse context* (Paperno et al., ACL 2016). Material del curso IA UC — Diplomado en Inteligencia Artificial, PUC Chile.

## Metadata

| Campo | Detalle |
|---|---|
| Título | The LAMBADA dataset: Word prediction requiring a broad discourse context |
| Autores | Denis Paperno, Germán Kruszewski (co-primeros autores), Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, Raquel Fernández |
| Afiliaciones | CIMeC — Center for Mind/Brain Sciences, University of Trento; Institute for Logic, Language & Computation, University of Amsterdam |
| Senior authorship | Marco Baroni, Gemma Boleda, Raquel Fernández |
| Venue | ACL 2016 (Association for Computational Linguistics) |
| Preprint | arXiv:1606.06031v1 [cs.CL], 20 de junio de 2016 |
| Acrónimo | LAnguage Modeling Broadened to Account for Discourse Aspects |
| Recursos | Corpus de entrenamiento + dev set descargables; test set reservado para una competencia pública (http://clic.cimec.unitn.it/lambada/) |

LAMBADA es un dataset de evaluación, no un modelo ni una arquitectura. Su contribución es metodológica: una forma de aislar y medir la comprensión de contexto amplio en modelos de lenguaje, construida con un protocolo de filtrado humano deliberadamente estricto.

## Contexto: la perplejidad promedio oculta lo que un modelo entiende

Hacia 2015-2016, la evaluación estándar de un modelo de lenguaje (LM) consistía en medir su perplejidad sobre un texto de prueba representativo. La perplejidad es el exponencial de la entropía cruzada promedio por token:

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(w_i \mid w_{1:i-1})\right)$$

El problema, que el paper articula con precisión, es que un texto natural promedio está dominado por tokens que se predicen localmente. Una proporción enorme de las palabras de un corpus se determina con la sintaxis y la colocación inmediata: artículos, preposiciones, concordancias, continuaciones de frases hechas. Un modelo puede alcanzar una perplejidad excelente capturando esas regularidades estadísticas de corto alcance sin entender absolutamente nada del discurso que lo rodea.

Los autores ilustran el punto con el sistema conversacional end-to-end de Vinyals y Le (2015), que produce diálogos como:

```
Human:   what is your job?
Machine: i'm a lawyer
Human:   what do you do?
Machine: i'm a doctor
```

Cada respuesta es localmente plausible, pero tomadas juntas son incoherentes: el sistema es "loro-like" (*parrot-like*). Produce fragmentos sensatos pero no integra el significado del contexto amplio. El riesgo metodológico que el paper denuncia es la *ilusión de comprensión*: la efectividad de las redes neuronales para extraer generalizaciones estadísticas de grandes corpus puede hacer creer que alcanzan un grado de entendimiento más profundo del que realmente tienen.

La pregunta de investigación, entonces, es: ¿cómo construimos un test que mida *solo* la capacidad de integrar contexto de discurso amplio, descontando lo que se resuelve con patrones locales? La perplejidad promedio no sirve porque mezcla ambas cosas. Se necesita un benchmark que aísle quirúrgicamente la dependencia de largo alcance.

## Idea central: predecir la última palabra solo si ves todo el pasaje

LAMBADA plantea una tarea de predicción de palabra (*word prediction*), el marco clásico del modelado de lenguaje, pero con una restricción de diseño que cambia todo. Cada ítem es un pasaje narrativo donde hay que adivinar la **última palabra de la oración final** (la *target word*). La propiedad definitoria del dataset:

- Un hablante de inglés acierta la palabra objetivo **si ve el pasaje completo** (en promedio 4.6 oraciones de contexto más la oración objetivo).
- Ese mismo hablante **falla si solo ve la última oración** (la oración objetivo sin su contexto previo).

El ejemplo canónico del paper (Ejemplo 1) lo deja claro:

> **Contexto:** "Yes, I thought I was going to lose the baby." "I was scared too," he stated... "This baby wasn't exactly planned for."
> **Oración objetivo:** "Do you honestly think that I would want you to have a ___?"
> **Palabra objetivo:** miscarriage

La oración objetivo aislada — *"¿Honestamente crees que querría que tuvieras un ___?"* — admite una multitud de continuaciones. Es el contexto amplio (la conversación sobre perder al bebé, un embarazo no planeado) el que fija sin ambigüedad la palabra *miscarriage*. La elegancia del diseño es que esta propiedad no se postula: se **verifica empíricamente con humanos**. Un pasaje entra a LAMBADA solo si gente real demuestra el patrón "acierta con contexto, falla sin él".

La consecuencia directa para los modelos es contundente: para tener éxito en LAMBADA, un modelo **no puede apoyarse en el contexto local**. La construcción del dataset garantiza que la información necesaria está distribuida en el discurso amplio, fuera de la oración objetivo. Esto es lo que la perplejidad promedio no podía aislar.

## Construcción: filtrado en dos etapas con verificación humana

### Fuente: BookCorpus

Los pasajes provienen del Book Corpus de Zhu et al. (2015), una colección de novelas no publicadas. La elección no es casual: al ser novelas inéditas, se **minimiza la utilidad del conocimiento de mundo general y de recursos externos**. Un modelo no puede resolver el pasaje recurriendo a hechos famosos, Wikipedia o trama conocida; debe usar el texto presente. Esto contrasta deliberadamente con noticias o novelas célebres.

Tras eliminar duplicados y filtrar material potencialmente ofensivo con una lista de stop words, el corpus contiene **5.325 novelas y 465 millones de palabras**. Se dividió en dos particiones de igual tamaño:

- **Partición de entrenamiento:** 2.662 novelas, más de 200 millones de palabras (203M reportados después). De aquí se entrenan los LMs que serán evaluados.
- **Partición dev+test:** de aquí se construye LAMBADA.

La división es por novela completa, no por pasaje. Esto es crucial: los pasajes de LAMBADA son **autocontenidos** y no pueden resolverse explotando el resto de la novela (información de fondo sobre personajes, propiedades del mundo ficticio). El mismo método novela-disjunta separa dev de test.

### Definición de contexto

El contexto es el **número mínimo de oraciones completas** antes de la oración objetivo tal que acumulen al menos **50 tokens** (umbral elegido en un estudio piloto). La tarea es adivinar la última palabra de la oración objetivo. La restricción de que la palabra objetivo sea la última de la oración no es necesaria para el objetivo de investigación, pero hace la tarea más natural para los sujetos humanos.

### Filtrado automático previo

Para reducir tiempo y costo, primero se descartaron pasajes fáciles para LMs estándar (probablemente resolubles con contexto local). Se usó una combinación de **cuatro modelos de lenguaje**: un RNN preentrenado (Mikolov et al., 2011) y tres modelos entrenados sobre el Book Corpus (un 4-grama estándar, un RNN y un feed-forward). Importante: estos modelos de filtrado son **distintos** de los que luego se evalúan en LAMBADA. Regla: cualquier pasaje cuya palabra objetivo tuviera probabilidad $\geq 0{,}00175$ según *cualquiera* de los cuatro modelos fue excluido.

### Filtrado humano en tres pasos (CrowdFlower)

Sobre los pasajes que sobrevivieron al filtro automático, se aplicó el protocolo humano que define la identidad del dataset:

1. Un sujeto adivina la palabra objetivo con el **pasaje completo**. Si acierta, continúa.
2. Un **segundo** sujeto adivina con el pasaje completo. Si también acierta, continúa.
3. Más sujetos intentan adivinar con la **oración objetivo sola**, hasta acertar o hasta acumular **10 intentos fallidos** (con 3 guesses permitidos por oración). Si nadie la adivina con la oración sola, el pasaje **entra a LAMBADA**.

El paso 2 se añadió tras un piloto: el paso 1 solo no garantizaba que el ítem fuera fácil con contexto (la salida mezclaba casos obvios con casos difíciles que algún sujeto particularmente hábil o con suerte acertó). Exigir **dos sujetos consecutivos** que coincidan exactamente sube la barra de "guessable con contexto amplio". Se aseguró que ningún sujeto juzgara el mismo ítem en ambas condiciones (pasaje y oración).

Las tasas de descarte muestran lo estricto del filtro:

| Etapa | Ítems descartados |
|---|---|
| Paso 1 (falla con contexto completo) | 84-86% |
| Paso 2 (segundo sujeto falla con contexto) | 6-7% adicional |
| Paso 3 (alguien acierta con la oración sola) | 3-5% adicional |
| **Sobrevivientes** | **~1 de cada 25** |

El costo: $0,22 por página en pasos 1 y 2 (10 pasajes/página), $0,15 por página en paso 3 (20 oraciones/página); **$1,24 promedio por ítem final**. Diseños alternativos (paso 3 antes que el 2 o que el 1) resultaron más caros. Los anotadores recibieron más de 200.000 pasajes en la etapa 1.

Los autores defienden el enfoque *perfect-match* (hit-or-miss): se exige coincidencia exacta de la palabra y que nadie la provea con contexto local. La alternativa — aceptar continuaciones plausibles o sinónimas en contexto amplio — es metodológica y prácticamente inviable a esta escala, porque determinar qué respuestas alternativas "encajan bien" requeriría anotación manual masiva. Mantienen solo ítems determinables de manera inequívoca por humanos.

### Estadísticas finales

| Atributo | Valor |
|---|---|
| Total de pasajes LAMBADA | 10.022 |
| Dev | 4.869 pasajes (de 1.331 novelas disjuntas) |
| Test | 5.153 pasajes (de 1.332 novelas disjuntas) |
| Oraciones de contexto (promedio) | 4,6 + 1 oración objetivo |
| Longitud total (promedio) | 75,4 tokens (dev) / 75 tokens (test) |
| Corpus de entrenamiento para LMs | 2.662 novelas, 203 millones de palabras |

El corpus de entrenamiento es del **mismo dominio** que dev+test, en gran cantidad pero **sin el filtrado**. Esto es deliberado: LAMBADA quiere evaluar modelos de propósito general en comprensión de contexto amplio (como hacían los sujetos humanos, usando habilidades generales de comprensión), no fomentar modelos ad-hoc que solo predigan la última palabra de pasajes tipo LAMBADA. El dev set puede usarse para fine-tuning a las particularidades del dataset.

## Propiedades: qué tipo de capacidad mide

El análisis lingüístico del dataset es la parte más rica del paper y revela qué se está midiendo realmente.

### La palabra objetivo suele estar (o estar implicada) en el contexto

Para que la palabra sea predecible solo con contexto amplio, debe estar **fuertemente sugerida** en el discurso. Empíricamente: **más del 80%** de los pasajes de LAMBADA incluyen la palabra objetivo (o su lema) en el contexto, frente a **menos del 15%** en los datos de entrada (input). Esta es una huella diagnóstica: el filtrado humano selecciona pasajes donde la palabra está anclada referencialmente en el discurso previo, no en la oración objetivo. Aun así, la presencia de la palabra no hace trivial la tarea — el sujeto debe identificar *cuál* de las palabras del contexto es la continuación correcta, lo que sigue exigiendo razonamiento.

### Distribución de categorías gramaticales (POS)

| Categoría | Proporción en LAMBADA |
|---|---|
| Nombres propios (PN) | 48% |
| Sustantivos comunes (CN) | 37% |
| Verbos (V) | 7,7% |
| Pronombres | 0,3% |
| Adjetivos, adverbios, otros | resto (minoritario) |

Los **nombres propios están masivamente sobrerrepresentados** respecto al input. La razón: cuando el contexto exige una expresión referencial, la restricción de "una sola palabra" excluye sintagmas nominales con artículo, y la **co-referencia** (rastrear a qué entidad apunta el texto) parece ser más fácil que otros fenómenos de discurso en esta tarea. El Ejemplo 2 lo muestra: "And Polish, to boot," said ___ → *Gabriel*, donde hay que rastrear quién habla en el diálogo.

Los sustantivos comunes (más de un tercio del dataset) exhiben una mezcla de fenómenos:

- **Co-referencia** directa (Ejemplo 3: *chains*, mencionado antes).
- **Co-referencia parcial / bridging**: shutter → *camera* (Ejemplo 5), donde el obturador implica la cámara.
- **Sinonimia / cuasi-sinonimia**: 'lose the baby' → *miscarriage* (Ejemplo 1).
- **Inferencia de participantes prototípicos de un evento**: un desayuno con comida típica permite adivinar *coffee* (Ejemplo 7) aunque nunca se mencione el café.

Verbos, adjetivos y adverbios son raros en LAMBADA porque muchos se adivinan con contexto local (verbos frecuentes como *ask, answer, call*; adverbios de clase cerrada como *now, too, well*). El filtrado los elimina porque pasan el paso 3. Esto sugiere que **rastrear fenómenos ligados a eventos (secuencias tipo script) es más difícil para los sujetos que la co-referencia**, al menos como se enmarca la tarea — los adverbios de clase abierta (*innocently, confidently*) son difíciles tanto con contexto local como amplio.

### Cuando la palabra NO está en el contexto

Cerca del **16%** de LAMBADA tiene el lema de la palabra objetivo *ausente* del contexto (Figura 2c). En esos casos:

- En ~1/3 la palabra está "casi ahí": misma raíz, distinta categoría (death → *died*, Ejemplo 9) o expresión sinónima ('deprived you of water' → *dehydrated*).
- En el resto se exige **inferencia de discurso más compleja**: participantes prototípicos de una escena (*coffee*), acciones sugeridas por el discurso (icy road → *driving*, Ejemplo 10), o propiedades cualitativas de situaciones (Ejemplo 8: *lonely*).

(El ~1% aparente de nombres propios fuera de contexto en la Figura 2c se debe a errores de lematización, p. ej. *Wynn–Wynns*; una revisión manual confirmó que **todos** los nombres propios objetivo están en el contexto.)

### Otras observaciones

Los ítems de LAMBADA contienen **discurso directo citado** con más frecuencia que el input (71% vs. 61%), lo que sugiere que el discurso más dialógico facilita la predicción de la palabra final.

El paper sintetiza la complejidad con el Ejemplo 1: resolver *miscarriage* requiere desde (morfo)fonología (el artículo *a* descarta *abortion* por la fonología del indefinido), pasando por morfosintaxis (el hueco pide un sustantivo común singular), hasta pragmática (entender qué infiere el participante masculino de las palabras de la femenina), más razonamiento general. LAMBADA, por construcción, es un muestreo de "casi cualquier aspecto de la comprensión textual": co-referencia, desambiguación de sentido, entailment, todo cae bajo el paraguas de la predicción de palabra.

## Modelos y resultados: el abismo entre máquina y humano

### Modelos evaluados

| Tipo | Modelos |
|---|---|
| Modelos de lenguaje | RNN simple (Elman), LSTM, N-grama (SRILM) con y sin cache, Memory Network |
| Baselines tailored | Sup-CBOW (red que predice desde bag-of-words del pasaje), Unsup-CBOW (similitud coseno pasaje–palabra) |
| Baselines aleatorios | Palabra aleatoria del vocabulario, palabra aleatoria del pasaje, palabra capitalizada aleatoria del pasaje |

Los modelos se entrenaron de forma no supervisada sobre los 203M de palabras de entrenamiento (predecir la siguiente palabra dado el contexto previo), salvo Sup-CBOW, que se entrenó sobre ~9M de pasajes de forma similar a LAMBADA extraídos de las novelas de entrenamiento. Vocabulario restringido a las **60.000 palabras más frecuentes** (cubre el 95% de las palabras objetivo del dev set). Hiperparámetros tuneados sobre accuracy en dev.

El LSTM es arquitectónicamente similar al Deep LSTM Reader de Hermann et al. (2015); la Memory Network es similar a la que obtuvo los mejores resultados en CBT (Hill et al., 2016). Es decir, no son hombres de paja: son modelos competitivos de la época con capacidad declarada de manejar contexto largo.

### Control set

Para descartar que la baja performance se deba a modelos malos, se construyó un **control set**: 5.000 pasajes de la misma forma y tamaño, de las mismas novelas de test, pero **sin ningún filtrado**. Es un benchmark de modelado de lenguaje estándar sobre el mismo corpus.

### Resultados

La métrica de interés es **accuracy** (acierto exacto de la palabra objetivo), porque a diferencia del modelado estándar sabemos que los humanos pueden predecir la palabra con precisión. Como la accuracy en LAMBADA muestra un efecto de piso (*bottoming*), también se reportan perplejidad y rango mediano de la palabra correcta.

| Conjunto | Método | Accuracy | Perplejidad | Rango mediano |
|---|---|---|---|---|
| **LAMBADA** | Random vocabulary word | 0 | 60000 | 30026 |
| | Random word from passage | 1,6 | — | — |
| | Random capitalized word from passage | 7,3 | — | — |
| | Unsup-CBOW | 0 | — | — |
| | Sup-CBOW | 0 | — | — |
| | N-Gram | 0,1 | 3125 | 993 |
| | N-Gram w/cache | 0,1 | **768** | 87 |
| | RNN | 0 | 14725 | 7831 |
| | LSTM | 0 | 5357 | 324 |
| | Memory Network | 0 | 16318 | 846 |
| **Control** | N-Gram | 19,1 | 285 | 17 |
| | N-Gram w/cache | 19,1 | 270 | 18 |
| | RNN | 15,4 | 277 | 24 |
| | LSTM | **21,9** | 149 | 12 |
| | Memory Network | 8,5 | 566 | 46 |

(Los valores de los baselines aleatorios y CBOW en el control set rondan accuracy 0-3,5%, confirmando que sin filtrado el control es resoluble por LMs reales pero no por azar.)

La lectura es demoledora. En el **control set**, los LMs son excelentes: tres modelos (los dos N-grama y el LSTM) aciertan en ~1/5 de los casos (19-22%). Son modelos genuinamente buenos en modelado de lenguaje estándar. En **LAMBADA**, todos colapsan a accuracy esencialmente cero. Ni siquiera superan la heurística trivial de "elegir una palabra aleatoria del pasaje" (1,6%) ni "una palabra capitalizada aleatoria del pasaje" (7,3%, que explota el sesgo de nombres propios).

Y aquí está el punto del paper: aunque los humanos aciertan en ~86% (implícito en el protocolo de filtrado, que exige dos humanos consecutivos correctos), ningún modelo del estado del arte de la época pasa del 1% de accuracy real (los abstracts dicen "ninguno alcanza accuracy sobre 1%"). El **gap humano-máquina es el resultado central**: ~86% vs. <1%.

Observaciones comparativas (usando perplejidad y rango, dado el piso de accuracy):

- Los N-grama tradicionales superan a las redes neuronales, probablemente por la dificultad de tunear bien estas últimas.
- El mejor desempeño relativo es **N-Gram w/cache** (perplejidad 768), porque el cache toma en cuenta estadísticas del pasaje. Aun así no acierta la palabra.
- El baseline de "palabra capitalizada aleatoria" (7,3%) muestra que el sesgo a nombres propios existe pero no basta — confirma que la dificultad real está en integrar contexto amplio, no en predecir la palabra exacta.

## Por qué es difícil para los modelos

La dificultad no es accidental: es estructural y, en parte, **construida por diseño**. Hay dos capas.

Primero, la capa intrínseca. La información necesaria para resolver el ítem reside fuera de la oración objetivo, distribuida en 4-5 oraciones de discurso. Un modelo que solo modela bien la distribución condicional local — que es lo que captura la mayoría de la "señal fácil" del lenguaje — no tiene de dónde sacar la respuesta. Necesita mantener en memoria entidades, resolver co-referencia, hacer bridging (shutter→camera), inferir participantes prototípicos (desayuno→coffee) e integrar pistas distantes (icy road→driving). Estos son fenómenos de **integración de información distante**, exactamente lo que la perplejidad promedio no estresa.

Segundo, la capa de diseño, que los autores reconocen con honestidad: uno de los primeros filtros fue **descartar pasajes que LMs simples predecían bien**. Por construcción, los LMs estándar están condenados a fallar en LAMBADA. Pero los autores argumentan que esto no invalida el benchmark: los humanos *sí* resuelven la tarea, así que un modelo que afirme tener buena comprensión del lenguaje debería poder hacerlo también. El reto es precisamente encontrar la forma de rodear esa dificultad inherente.

Su hipótesis sobre el camino a seguir: a pesar del resultado decepcionante de la Memory Network "vanilla", la capacidad de almacenar información en una memoria de más largo plazo será crucial, acoplada con la habilidad de razonar sobre lo almacenado para recuperar la información correcta. También sugieren que mecanismos de atención (Bahdanau et al., 2014) podrían ayudar — una observación que, con el diario del lunes, anticipa el camino real hacia la solución.

## Limitaciones

Los propios autores enmarcan su evaluación como **preliminar**, un *proof-of-concept* de la dificultad. Limitaciones a tener en cuenta:

- **Formato de last-word prediction.** La restricción de predecir la última palabra de la oración objetivo es artificial; se eligió por naturalidad para humanos, no por necesidad teórica. Acota qué fenómenos entran (favorece nombres propios y sustantivos por la concordancia de "una sola palabra").
- **Dominio de novelas.** BookCorpus es ficción narrativa. La elección minimiza el conocimiento de mundo externo (una virtud para el objetivo), pero limita la generalización a otros géneros (noticias, técnico, conversacional). La alta proporción de discurso directo es un artefacto del género.
- **Enfoque perfect-match estricto.** Solo se aceptan ítems con una única respuesta inequívoca verificada por humanos. Esto excluye casos donde varias continuaciones serían plausibles — legítimos como comprensión de discurso, pero no medibles con coincidencia exacta. El dataset mide un subconjunto "limpio" del fenómeno.
- **Tuning preliminar de los modelos.** Los autores admiten que más tuning, atención u otros mecanismos podrían mejorar resultados; no afirman que su evaluación sea exhaustiva.

## Impacto: el benchmark de contexto largo que los LLMs terminaron cerrando

LAMBADA se convirtió en un **benchmark estándar** para medir comprensión de contexto largo en modelos de lenguaje, y su trayectoria posterior es uno de los ejemplos más nítidos de cómo la escala cerró un gap que en 2016 parecía infranqueable.

En 2016, con <1% de accuracy frente a ~86% humano, el mensaje era que las arquitecturas existentes estaban "lejísimos" de la comprensión genuina de discurso. Lo que vino después:

- **Continuous cache / atención sobre contexto** y modelos de comprensión lectora empezaron a moverse del piso de accuracy poco después de la publicación.
- **GPT-2 (2019)** reportó accuracy de LAMBADA explícitamente como una de sus métricas insignia de comprensión de contexto largo, mostrando saltos grandes en zero-shot a medida que escalaba el modelo (de ~1.5B parámetros). LAMBADA pasó de "imposible" a "indicador de progreso".
- **GPT-3 (2020)** reportó LAMBADA en su tabla de resultados como prueba de comprensión de dependencias de largo alcance, alcanzando en few-shot niveles muy por encima del estado del arte previo y acercándose al rango humano.

La moraleja es doble. Por un lado, valida la tesis de Paperno et al.: la integración de contexto amplio era una capacidad ausente y real, no un artefacto, y midiéndola se detectó algo que la perplejidad promedio escondía. Por otro, ilustra que el camino a la solución no fue el que el paper apostó en primera instancia (Memory Networks con razonamiento explícito), sino la **escala de Transformers autoregresivos** entrenados sobre corpus enormes, donde la atención provee de facto el acceso a contexto distante que las RNN/LSTM de 2016 no lograban explotar. El benchmark sobrevivió a la hipótesis arquitectónica de sus autores.

LAMBADA también consolidó una metodología: **aprovechar la performance humana en predicción de palabra para construir benchmarks**. La idea de filtrar con humanos para aislar una capacidad específica (no solo recolectar etiquetas) influyó en el diseño de evaluaciones posteriores.

## Conexión con la Clase 24

El PDF de la Clase 24 del curso lista LAMBADA entre los **datasets pre-2015** de la era de comprensión lectora / modelado de lenguaje (slide 21), junto a CNN/Daily Mail (CNNDM) y Children's Book Test (CBT). Conviene situarlo en esa familia, porque el propio paper se compara explícitamente con sus parientes:

- **CNN/Daily Mail (Hermann et al., 2015):** artículos con resúmenes; la tarea es adivinar una entidad nombrada removida del resumen. Como LAMBADA, exige mirar el contexto amplio (el artículo). Diferencias: género (noticias vs. novelas), ítems limitados a entidades nombradas, y — clave — en CNNDM el modelo debe **resumir** el artículo, mientras que en LAMBADA la oración objetivo es una **continuación** de la narrativa, no un resumen. LAMBADA pide entender qué desarrollo es plausible para un fragmento narrativo o un diálogo.
- **CBT (Hill et al., 2016):** excerpts de libros con una palabra removida de la última oración en una secuencia de 21 oraciones. La distinción crucial: **CBT no fue filtrado para ser human-guessable solo con contexto amplio**. En el análisis post-hoc de Hill et al., en muchos casos donde los anotadores adivinaban con contexto amplio, también podían hacerlo con la última oración sola; y en ~1/5 no podían adivinar ni con contexto. Solo una porción pequeña de CBT prueba realmente comprensión de contexto amplio — que es justo el foco exclusivo de LAMBADA. Este es el aporte metodológico diferencial.
- **MSRCC (Zweig y Burges, 2011):** el origen de la idea de completado de excerpts de libros, pero con contexto limitado a oraciones individuales, sin medir comprensión de pasajes amplios.

LAMBADA es, en esencia, una tarea **cloze / last-word** emparentada con los cloze tasks de CNN/DM y CBT, pero con el filtro humano que la hace el test más puro de comprensión de contexto amplio del grupo. Para el hilo de la Clase 24, esto conecta directamente con la **evaluación de modelos de lenguaje** y con la transición a la era Transformer/GPT que la clase cubre al final: LAMBADA es el puente donde se ve, con números concretos, por qué los modelos de 2016 no "entendían" el discurso y cómo la generación siguiente (GPT-2/GPT-3) reportó precisamente esta métrica para demostrar que sí empezaban a hacerlo. Es el benchmark que materializa el salto de "perplejidad promedio engañosa" a "medición aislada de contexto largo", y que documenta el cierre del gap humano-máquina vía escala.

Para Roberto, hay una lección de evaluación transferible a sistemas ML de producción (incluido matching/MDM en FHIR): una métrica promedio agregada puede ocultar fallas sistemáticas en el subconjunto de casos difíciles. LAMBADA es, conceptualmente, el equivalente a construir un *slice* de evaluación deliberadamente adversarial — casos donde la señal fácil no basta — para descubrir qué *no* aprende el modelo. La práctica de filtrar tu test set para aislar la capacidad que de verdad te importa (en vez de promediar sobre la distribución natural) es directamente aplicable a la validación de scorers donde los casos triviales inflan las métricas.

## Notas y enlaces

- **Paper:** Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., Fernández, R. (2016). *The LAMBADA dataset: Word prediction requiring a broad discourse context*. ACL 2016. arXiv:1606.06031.
- **Recursos del dataset:** corpus de entrenamiento + dev set en http://clic.cimec.unitn.it/lambada/ (el test set se reservó para una competencia pública). Material suplementario con detalles técnicos en la misma URL.
- **Corpus fuente:** BookCorpus (Zhu et al., 2015), *Aligning books and movies*, ICCV 2015.
- **Parientes citados:** CNN/Daily Mail (Hermann et al., 2015, NIPS); CBT — Children's Book Test (Hill et al., 2016, ICLR); MSRCC (Zweig y Burges, 2011, MSR-TR-2011-129).
- **Modelos de base mencionados:** RNN (Elman, 1990); LSTM (Hochreiter y Schmidhuber, 1997); SRILM N-grama (Stolcke, 2002); End-to-end Memory Networks (Sukhbaatar et al., 2015); CBOW (Mikolov et al., 2013); atención (Bahdanau et al., 2014).
- **Cifras clave para recordar:** 10.022 pasajes (4.869 dev / 5.153 test); contexto promedio 4,6 oraciones / ~75 tokens; >80% de los ítems tienen la palabra objetivo en el contexto; 48% nombres propios; ~1 de cada 25 pasajes de entrada sobrevive al filtro; gap humano (~86%) vs. modelos (<1%, mejor relativo N-Gram w/cache con perplejidad 768).
