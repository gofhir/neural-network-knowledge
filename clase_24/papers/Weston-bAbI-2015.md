# Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks (bAbI)

## 1. Metadata

| Campo | Detalle |
|---|---|
| Título | *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks* |
| Autores | Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merriënboer, Armand Joulin, Tomas Mikolov |
| Afiliación | Facebook AI Research (FAIR), 770 Broadway, Nueva York, USA |
| Año | 2015 (versión v10 fechada el 31 de diciembre de 2015) |
| Publicación | Bajo revisión como conference paper en ICLR 2016 |
| Identificador | arXiv:1502.05698 [cs.AI] |
| Recursos | Tareas: `http://fb.ai/babi`; código generador: `https://github.com/facebook/bAbI-tasks` |
| Dataset asociado | bAbI (20 tareas sintéticas de QA) |

El nombre "bAbI" alude a la idea de un "baby AI": un conjunto mínimo de competencias que cualquier sistema con aspiraciones de comprensión de lenguaje debería dominar antes de abordar problemas más ambiciosos.

## 2. Contexto: QA como proxy de "AI-complete"

El objetivo de fondo declarado por los autores es construir un agente de diálogo inteligente. El problema es que evaluar diálogo abierto de forma automática es notoriamente difícil: no hay una métrica limpia que diga si una respuesta conversacional es "correcta". Por eso el paper hace un movimiento estratégico: en lugar de evaluar diálogo, evalúan **comprensión lectora vía question answering (QA)**. QA tiene la propiedad de que, en escenarios de verdadero/falso, opción múltiple o respuesta de una palabra, la corrección es inequívoca y se mide trivialmente como acierto o error.

La tesis central es que QA es un dominio extraordinariamente amplio: "más o menos cualquier tarea que uno pueda imaginar puede plantearse en este formato". Esto convierte a QA en un marco unificador bajo el cual se pueden expresar capacidades muy distintas. La frase "AI-complete" del título es deliberada: la comprensión plena de lenguaje es, en la práctica, tan difícil como la IA en general. Pero en lugar de atacar ese problema monolítico, los autores proponen **descomponer la comprensión en habilidades atómicas medibles**, cada una aislada en su propia tarea.

El paper sitúa esta idea en una larga tradición de **datos sintéticos** en machine learning: el problema XOR que motivó las redes neuronales (Minsky & Papert, 1969), los datasets de círculos y anillos que motivaron clustering espectral y aprendizaje semi-supervisado (Ng et al., 2002; Zhu et al., 2003), las ecuaciones de Mackey-Glass para series de tiempo, e incluso datasets clásicos de UCI como *waveform*. Más cercanamente, los datos sintéticos sirvieron para desarrollar la Neural Turing Machine (Graves et al., 2014) y las Memory Networks (Weston et al., 2014), esta última directamente relevante al trabajo.

El argumento epistemológico es agudo. Los autores citan a Halevy et al. (2009) — *"the unreasonable effectiveness of data"* — para señalar que cuando se trabaja con grandes volúmenes de datos reales, los investigadores tienden a converger hacia modelos más simples, porque "modelos simples con muchos datos vencen a modelos elaborados con menos datos". Un N-gram para modelado de lenguaje funciona bien relativo a sus competidores, pero está lejos de entender realmente el texto. La conclusión: como investigadores podemos quedar atrapados en mínimos locales en el espacio de algoritmos, y los datos sintéticos son una vía para romper ese estancamiento, porque permiten construir tareas diagnósticas donde el fracaso de un modelo es interpretable.

## 3. Idea central: 20 tareas, una habilidad por tarea

El principio de diseño es explícitamente análogo al **software testing**: cada tarea es idealmente un "caso de prueba hoja", lo más independiente posible de las demás, que ejercita de la manera más simple posible un aspecto del comportamiento deseado. Tareas "no hoja" subsiguientes pueden construirse combinando habilidades. La promesa es que si un sistema falla en una tarea, sabemos *exactamente* qué habilidad le falta, lo que permite proponer mejoras dirigidas — algo imposible en datasets reales donde cada pregunta mezcla coreferencia, deducción, sentido común, etc.

Todas las tareas son **sin ruido**: un humano que lea el idioma puede en principio alcanzar 100% de exactitud. No requieren formación en semántica formal, machine learning, lógica ni representación de conocimiento. La supervisión en entrenamiento incluye la respuesta verdadera y el conjunto de **hechos de soporte** (supporting facts) relevantes — que el modelo puede usar o no. Las respuestas se limitan a una sola palabra (`¿Dónde está Mark? A: bathroom`) o una lista de palabras (`¿Qué tiene Mark? A: milk, football`).

| # | Tarea | Habilidad aislada |
|---|---|---|
| 1 | Single Supporting Fact | Recuperar la respuesta de un único hecho relevante entre distractores |
| 2 | Two Supporting Facts | Encadenar dos hechos para responder |
| 3 | Three Supporting Facts | Encadenar tres hechos |
| 4 | Two Argument Relations | Distinguir sujeto y objeto; sensibilidad al orden de palabras (bag-of-words falla) |
| 5 | Three Argument Relations | Distinguir dador/receptor/objeto en relaciones ternarias |
| 6 | Yes/No Questions | Responder verdadero/falso |
| 7 | Counting | Contar objetos con una propiedad |
| 8 | Lists/Sets | Producir un conjunto de respuestas (lista de palabras) |
| 9 | Simple Negation | Modelar negaciones ("ya no está en...") |
| 10 | Indefinite Knowledge | Modelar posibilidad vs. certeza (respuesta "maybe") |
| 11 | Basic Coreference | Resolver el referente más cercano de un pronombre |
| 12 | Conjunction | Manejar múltiples sujetos en una oración ("Mary and Jeff went...") |
| 13 | Compound Coreference | Pronombre que refiere a múltiples actores ("they") |
| 14 | Time Reasoning | Interpretar expresiones temporales explícitas (afternoon, yesterday) |
| 15 | Basic Deduction | Deducción por herencia de propiedades (silogismo) |
| 16 | Basic Induction | Inducción por herencia de propiedades |
| 17 | Positional Reasoning | Razonamiento espacial sobre bloques de colores (estilo SHRDLU) |
| 18 | Size Reasoning | Razonamiento sobre tamaños relativos de objetos |
| 19 | Path Finding | Encontrar la ruta entre ubicaciones (problema de búsqueda) |
| 20 | Agent's Motivations | Inferir por qué un agente realiza una acción (estados mentales) |

Las tareas 8 y 9 están inspiradas en el trabajo previo sobre *lambda dependency-based compositional semantics* (Liang et al., 2013). El paper se posiciona cerca del **Winograd Schema Challenge** (Levesque et al., 2011) en cuanto a la interpretabilidad directa de los resultados, pero se diferencia en dos puntos: las tareas bAbI son **autocontenidas** (vienen con datos de entrenamiento *y* de evaluación, no solo evaluación) y son **más diversas**. A diferencia de ARISTO (exámenes de ciencias) o MCTest (660 historias, conjunto de entrenamiento demasiado pequeño), bAbI permite controlar la cantidad de ejemplos de entrenamiento y garantiza que el conocimiento y razonamiento de sentido común necesario para el test esté contenido en el train.

## 4. Generación sintética: el simulador

Todas las tareas se generan con un **simulador que se comporta como un juego clásico de aventura de texto** (text adventure game), en la tradición de Bordes et al. (2010) y Weston et al. (2014), pero más complejo. La idea es **anclar (ground)** el lenguaje en un mundo artificial coherente y controlado, donde las etiquetas verdaderas se conocen por construcción.

El mundo simulado se compone de **entidades** de varios tipos (locaciones, objetos, personas) y de **acciones** que operan sobre ellas. Las entidades tienen estados internos: su ubicación, qué objetos cargan (encima o dentro, como mesas o cajas), el estado mental de los actores (hambriento, etc.) y propiedades como tamaño, color y comestibilidad. Para las locaciones se codifican las conexiones espaciales (qué hay al este, qué hay arriba). Para los actores se pueden especificar reglas pre-definidas que controlan su comportamiento (si tienen hambre, buscan comida); si no hay regla, se ejecutan acciones válidas aleatorias.

El repertorio de acciones del simulador es:
`go <location>`, `get <object>`, `get <object1> from <object2>`, `put <object1> in/on <object2>`, `give <object> to <actor>`, `drop <object>`, `set <entity> <state>`, `look`, `inventory`, `examine <object>`.

Un conjunto de **restricciones universales** garantiza coherencia: un actor no puede tomar algo que ya tiene (o que tiene otro), no puede ir a un lugar no conectado, no puede soltar algo que no posee, etc. Para cada tarea se limita el conjunto de acciones necesarias (la tarea 1 solo necesita `go`; la tarea 2 usa `go`, `get` y `drop`). La secuencia de comandos define una "historia" ejecutable: `joe go playground; bob go office; joe get football`. El sistema entonces consulta el estado del mundo (`where football?`) y, como tiene acceso al estado interno, calcula la respuesta verdadera de forma trivial.

Para producir texto con **variedad léxica**, se aplica una gramática automática simple: a cada verbo se le asignan sinónimos (`get` se reemplaza por *picked up*, *got*, *grabbed* o *took*; `drop` por *dropped*, *left*, *discarded* o *put down*), y objetos/actores pueden tener reemplazos (sustituir *Daniel* por *he* en la tarea 11). Los adverbios son cruciales para tareas como la 14 (razonamiento temporal).

**Por qué sintético.** El control total sobre el mundo permite: (i) conocer la etiqueta verdadera por grounding, sin anotación manual; (ii) generar tantos ejemplos como se quieran, midiendo cuántos hacen falta para resolver una tarea; (iii) aislar habilidades para diagnóstico de fallos específicos; y (iv) cerrar un *feedback loop* donde nuevas tareas se diseñan, posiblemente de forma adversarial, para romper los modelos recién propuestos. El paper también libera las tareas (i) en hindi y (ii) con palabras en inglés barajadas (ilegibles para humanos). Un buen algoritmo debería rendir parecido en las tres versiones — lo que *no* ocurriría con métodos que dependen de recursos externos específicos de un idioma, simulando a un aprendiz expuesto por primera vez a una lengua.

Los autores son explícitos sobre los límites del simulador: las oraciones son cortas y con poco anidamiento, el vocabulario es pequeño (150 palabras, típicamente 4 actores, 6 locaciones y 3 objetos por tarea). Y subrayan que estas tareas **no sustituyen** datos reales, sino que los **complementan**.

## 5. Formato de los datos

Cada ejemplo es una **historia** (lista de oraciones numeradas), seguida de una **pregunta**, su **respuesta** y los **hechos de soporte** (índices de las oraciones relevantes). Ejemplos concretos del paper:

**Tarea 1 — Single Supporting Fact:**
```
Mary went to the bathroom.
John moved to the hallway.
Mary travelled to the office.
Where is Mary? A: office
```
La respuesta exige recuperar el último hecho relevante sobre Mary, ignorando el distractor sobre John.

**Tarea 2 — Two Supporting Facts:**
```
John is in the playground.
John picked up the football.
Bob went to the kitchen.
Where is the football? A: playground
```
Hay que encadenar "John tiene el football" + "John está en el playground".

**Tarea 3 — Three Supporting Facts:**
```
John picked up the apple.
John went to the office.
John went to the kitchen.
John dropped the apple.
Where was the apple before the kitchen? A: office
```

**Tarea 5 — Three Argument Relations:**
```
Mary gave the cake to Fred.
Fred gave the cake to Bill.
Jeff was given the milk by Bill.
Who gave the cake to Fred? A: Mary
Who did Fred give the cake to? A: Bill
```

**Tarea 15 — Basic Deduction:**
```
Sheep are afraid of wolves.
Cats are afraid of dogs.
Mice are afraid of cats.
Gertrude is a sheep.
What is Gertrude afraid of? A: wolves
```

**Tarea 19 — Path Finding:**
```
The kitchen is north of the hallway.
The bathroom is west of the bedroom.
The den is east of the hallway.
The office is south of the bedroom.
How do you go from den to kitchen? A: west, north
```

**Tarea 20 — Agent's Motivations:**
```
John is hungry.
John goes to the kitchen.
John grabbed the apple there.
Daniel is hungry.
Where does Daniel go? A: kitchen
Why did John go to the kitchen? A: hungry
```

Este formato — historia numerada + pregunta + respuesta de una palabra/lista + supporting facts — se convirtió en el estándar de facto para benchmarks de razonamiento sobre texto y es directamente reconocible en cualquier implementación posterior de Memory Networks.

## 6. Modelos evaluados

Los modelos se agrupan en **tres pistas de supervisión**:

- **Weakly supervised**: solo reciben pares pregunta-respuesta en entrenamiento.
- **Strong supervision**: además reciben el conjunto de hechos de soporte en entrenamiento (pero no en test). Dan cotas superiores de rendimiento respecto a la versión débilmente supervisada de la misma clase de modelo.
- **External resources**: pueden usar datos etiquetados de otras fuentes (coreferencia, semantic role labeling), además de supervisión fuerte.

Protocolo experimental: **1000 preguntas de entrenamiento y 1000 de test por tarea**, reportando exactitud de test. Una tarea se considera **"pasada" si se obtiene ≥ 95% de exactitud** (el paper aclara que el umbral de 95% y los 1000 ejemplos son elecciones arbitrarias). La consigna metodológica es importante: **un único modelo debe evaluarse en todas las tareas, sin tuning por tarea**, y luego probarse en tareas reales.

Los métodos comparados:

1. **N-gram classifier** (baseline, débilmente supervisado). Inspirado en Richardson et al. (2013), adaptado a producir una respuesta de 1 palabra: construye un bag-of-N-grams sobre las oraciones de la historia que comparten al menos una palabra con la pregunta, y entrena un clasificador lineal. (Usar *todas* las oraciones, sin filtrar, daba peores resultados.)

2. **LSTM** (débilmente supervisado). Lee la historia hasta llegar a la pregunta y luego emite una respuesta. Está en desventaja por ser solo débilmente supervisado.

3. **Memory Networks (MemNN)** (Weston et al., 2014, supervisión fuerte). Un "controlador" neuronal realiza inferencia sobre memorias almacenadas (las oraciones previas). El modelo original hace **2 hops** de inferencia: encuentra el primer hecho de soporte con máximo score de match con la pregunta, luego el segundo con máximo match respecto a la pregunta *y* el primer hecho. La función de matching mapea el bag-of-words de pregunta y hechos a un espacio de embedding sumando word embeddings; los embeddings se aprenden con supervisión fuerte.

4. **Extensiones de MemNN** propuestas en este paper:
   - **Adaptive Memories (AM)**: número variable de hops en lugar de 2 fijos. Se puntúa un hecho especial $m_\emptyset$ y se itera $o_i = O([x, m_{o_1}, \dots, m_{o_{i-1}}], m)$ hasta predecir $m_\emptyset$ (con tope duro de 10 iteraciones para evitar bucles infinitos). El mismo truco con una palabra especial $w_\emptyset$ permite emitir **respuestas multi-palabra** (necesario para tareas 8 y 19).
   - **N-grams (NG)**: bag de 3-gramas en lugar de bag-of-words, para capturar orden de palabras.
   - **Nonlinearity (NL)**: una red neuronal de 2 capas con no-linealidad $\tanh$ en la función de matching, $E(x) = \tanh(W \tanh(\Phi_x(x)^\top U))$.

5. **Structured SVM** (recursos externos). Sistema NLP en cascada clásico: corre el sistema de coreferencia de Stanford (Raghunathan et al., 2010) y el SRL de SENNA (Collobert et al., 2011) como preprocesamiento, y construye features (pares de palabras, distancia, orden, pares de verbos SRL, pares verbo-argumento). Busca hasta tres hechos de soporte por búsqueda exhaustiva (no greedy, a diferencia de MemNN).

**Por qué brillan las MemNN.** La función de scoring de MemNN tiene la forma de un modelo de embedding:

$$s(x, y) = \Phi_x(x)^\top U^\top U\, \Phi_y(y)$$

donde $U$ es una matriz $n \times D$. El módulo $O$ recupera $k$ memorias de soporte mediante $\arg\max$ (ecuaciones 1-2), y el módulo $R$ produce la respuesta rankeando palabras del diccionario (ecuación 3). La arquitectura de memoria externa explícita encaja naturalmente con tareas que requieren **recuperar y encadenar hechos dispersos** en una historia — exactamente lo que bAbI mide. El N-gram y el LSTM no tienen ese mecanismo de recuperación direccionable, y por eso quedan muy por detrás.

## 7. Resultados

La tabla siguiente reproduce la exactitud de test (%) de la Tabla 3 del paper para los métodos principales. Las extensiones de MemNN se denotan AM (adaptive memory), NG (N-grams), NL (nonlinear). La columna "Min. ej." indica el mínimo de ejemplos para alcanzar ≥ 95% (o FAIL con 1000), y "Multitask" el rendimiento del modelo AM+NG+NL entrenado en todas las tareas a la vez.

| Tarea | N-gram | LSTM | MemNN (2014) | AM | AM+NG | AM+NL | AM+NG+NL | SVM (ext.) | Min. ej. | Multitask |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 Single Supporting Fact | 36 | 50 | 100 | 99 | 100 | 100 | 100 | 100 | 250 | 100 |
| 2 Two Supporting Facts | 2 | 20 | 74 | 74 | 100 | 100 | 100 | 100 | 500 | 100 |
| 3 Three Supporting Facts | 7 | 20 | 17 | 94 | 99 | 100 | 100 | 98 | 500 | 98 |
| 4 Two Arg. Relations | 50 | 61 | 98 | 27 | 100 | 69 | 100 | 80 | 500 | 80 |
| 5 Three Arg. Relations | 20 | 70 | 83 | 21 | 86 | 83 | 98 | 99 | 1000 | 99 |
| 6 Yes/No Questions | 49 | 48 | 99 | 23 | 53 | 99 | 100 | 100 | 500 | 100 |
| 7 Counting | 52 | 49 | 69 | 51 | 86 | 78 | 85 | 86 | FAIL | 86 |
| 8 Lists/Sets | 40 | 45 | 70 | 52 | 88 | 90 | 91 | 93 | FAIL | 93 |
| 9 Simple Negation | 62 | 64 | 100 | 8 | 63 | 71 | 100 | 100 | 500 | 100 |
| 10 Indefinite Knowledge | 45 | 44 | 99 | 91 | 54 | 57 | 98 | 98 | 1000 | 98 |
| 11 Basic Coreference | 29 | 72 | 100 | 49 | 100 | 100 | 100 | 100 | 250 | 100 |
| 12 Conjunction | 9 | 74 | 96 | 100 | 100 | 100 | 100 | 100 | 250 | 100 |
| 13 Compound Coref. | 26 | 94 | 99 | 100 | 100 | 100 | 100 | 100 | 250 | 100 |
| 14 Time Reasoning | 19 | 27 | 99 | 99 | 100 | 100 | 99 | 99 | 500 | 99 |
| 15 Basic Deduction | 20 | 21 | 96 | 100 | 73 | 74 | 100 | 100 | 100 | 100 |
| 16 Basic Induction | 43 | 23 | 24 | 100 | 100 | 27 | 100 | 100 | 100 | 94 |
| 17 Positional Reasoning | 46 | 51 | 61 | 49 | 46 | 54 | 65 | 57 | FAIL | 72 |
| 18 Size Reasoning | 52 | 52 | 62 | 74 | 50 | 57 | 95 | 54 | 1000 | 93 |
| 19 Path Finding | 0 | 8 | 49 | 3 | 9 | 0 | 36 | 15 | FAIL | 19 |
| 20 Agent's Motivations | 76 | 91 | 95 | 100 | 100 | 100 | 100 | 100 | 250 | 100 |
| **Media** | 34 | 49 | 79 | 63 | 79 | 75 | 93 | 87 | — | 92 |

*(Los valores se transcriben de la Tabla 3 del paper; el mapeo exacto de algunas columnas intermedias de variantes MemNN puede estar sujeto a la ambigüedad del texto OCR de las cabeceras rotadas, pero los patrones cualitativos y las columnas clave — N-gram, LSTM, MemNN original, AM+NG+NL y SVM — son los reportados.)*

**Análisis.** Las MemNN estándar superan claramente a los baselines N-gram y LSTM, consistente con Weston et al. (2014). Pero la MemNN original **falla** (test < 95%) en varias tareas, algunas esperadas y otras no:

- **Fallos esperados** por limitaciones de modelado: con $k=2$ hechos, respuestas de una sola palabra y bag-of-words, no resuelve las tareas 3, 4, 5, 7, 8 y 18.
- **Fallos inesperados**: yes/no (6) e indefinite knowledge (10). En retrospectiva, la función de scoring *lineal* de la MomNN estándar no puede modelar el match entre query, hecho de soporte y respuesta sí/no, porque eso requiere **interacciones de tres vías** — exactamente lo que arregla la no-linealidad (NL).

Las extensiones aportan mejoras complementarias:
- **AM** ayuda en tareas que requieren más de dos hechos (3 y 16) y, en menor medida, en las que requieren salida multi-palabra (8 y 19).
- **NG** ayuda cuando importa el orden de palabras (4 y 15), pero no sustituye a la no-linealidad.
- **NL** resuelve las interacciones de tres vías (6 y 10), pero no modela orden de palabras (falla en 4).
- **AM+NG+NL** (combinación) **promueve 9 tareas de fracaso a éxito** respecto a la MemNN original, alcanzando media de 93.

El **Structured SVM**, pese a sus recursos externos, **no supera a la MemNN extendida** (también falla en 9 tareas). Gana en 6, 9 y 10 (sus conjunciones de features capturan las no-linealidades), pero su ranking sobre muchas posibilidades introduce errores en tareas de tres (a veces dos) hechos de soporte (3, 16, 2). Su búsqueda **no-greedy** sí ayuda en **path finding (19)**, donde la búsqueda es esencial.

Las tareas que quedan como **problemas abiertos**:
- **Counting (7)** requiere 10000 ejemplos y **Lists/Sets (8)** requiere 5000 → marcadas FAIL bajo el presupuesto de 1000.
- **Positional Reasoning (17)** y **Path Finding (19)** no se resuelven **ni con 10000 ejemplos**. Estas (y formas más avanzadas de inducción/deducción) requieren un **algoritmo de búsqueda general** incorporado al procedimiento de inferencia, del que MemNN y todos los demás métodos carecen.

El entrenamiento **multitask** (última columna) da rendimiento alentadoramente similar al de entrenar tarea por tarea (media 92 vs. 93), mostrando que un solo modelo puede aprender múltiples aspectos de comprensión y razonamiento simultáneamente.

## 8. Limitaciones

Los propios autores son francos sobre los límites:

- **Sintético ≠ lenguaje real.** Las oraciones son cortas, con poco anidamiento; el vocabulario es minúsculo (150 palabras). La complejidad sintáctica, la ambigüedad genuina y la riqueza del lenguaje natural real quedan fuera. El paper insiste: *"estas tareas no son un sustituto de datos reales, sino que deben complementarlos"*.
- **Riesgo de overfitting a la gramática del simulador.** Como el texto se genera con una gramática automática de sinónimos, un modelo podría aprender los patrones del *generador* en lugar de razonamiento genuino. Las versiones en hindi y con palabras barajadas son una mitigación parcial: un método que dependa de recursos externos específicos de inglés rendiría peor en ellas, delatando que no aprende de cero.
- **Supervisión más fuerte de lo realista.** Los mejores resultados usan **supervisión fuerte** (los hechos de soporte en entrenamiento), que rara vez está disponible en escenarios reales. El paper señala que en el caso *débilmente supervisado* con ≤ 1000 ejemplos no se conoce ningún método general (no hand-engineered) que resuelva las tareas.
- **Cobertura incompleta del razonamiento.** Un análisis completo de inducción y deducción está "claramente fuera del alcance" del trabajo; las tareas 15-16 son versiones básicas.

**Distinción con datasets reales.** El paper contrasta explícitamente bAbI con benchmarks reales y complementarios: *Teaching Machines to Read and Comprehend* (Hermann et al., 2015, el dataset CNN/Daily Mail de cloze), large-scale simple QA con Memory Networks (Bordes et al., 2015) y el Children's Book Test (Hill et al., 2015). La consigna metodológica es clara: aunque un método funcione bien en las 20 tareas, debe demostrarse útil también en datos reales. bAbI es un **banco de pruebas diagnóstico**, no un objetivo final.

## 9. Impacto

El paper reporta que, ya desde su publicación online, las tareas bAbI influyeron directamente en el desarrollo de varios algoritmos prometedores de razonamiento con memoria:

- **End-to-End Memory Networks (MemN2N)** de Sukhbaatar et al. (2015): versión *débilmente supervisada* y entrenable de extremo a extremo de las MemNN, que elimina la necesidad de supervisión sobre hechos de soporte (la principal crítica de realismo del paper original). MemN2N además se mostró efectiva en tareas reales (Hill et al., 2015).
- **Dynamic Memory Networks (DMN)** de Kumar et al. (2015): introduce un mecanismo de atención episódica iterativa sobre la memoria.
- **Neural Reasoner** de Peng et al. (2015).

Más ampliamente, bAbI consolidó el paradigma de las **arquitecturas con memoria externa direccionable** — junto a las Neural Turing Machines de Graves et al. (2014), citadas en el contexto — como una línea central de la agenda de *reasoning* en deep learning. Durante varios años, "resolver bAbI" (las 20 tareas con un solo modelo débilmente supervisado) fue un hito estándar de referencia para nuevas arquitecturas de razonamiento, incluyendo modelos posteriores como las Recurrent Entity Networks y los Relation Networks. El formato de datos historia-pregunta-respuesta-soporte se volvió canónico.

El valor duradero de bAbI no es como dataset "difícil" — modelos modernos lo resuelven casi por completo — sino como **instrumento de diagnóstico**: su capacidad de aislar exactamente qué habilidad le falta a un modelo, y de cerrar el feedback loop entre diseño de tareas y diseño de algoritmos.

## 10. Conexión con la Clase 24

La Clase 24 del curso (Question Answering / Reading Comprehension) lista, entre los datasets de referencia (slide 21), "bAbI / Children's Book Test". bAbI cumple un rol pedagógico específico dentro del panorama de QA: representa el extremo **sintético y diagnóstico** del espectro, en oposición directa a los datasets **reales y a escala** que el resto de la clase introduce.

El contraste pedagógico es el siguiente:

- **bAbI (sintético/diagnóstico):** mundo cerrado generado por simulador, 150 palabras de vocabulario, respuestas de una palabra, supervisión sobre hechos de soporte, una habilidad por tarea. Permite responder la pregunta *"¿qué tipo exacto de razonamiento le falta a mi modelo?"*. El fracaso es interpretable.
- **SQuAD (Rajpurkar et al., 2016):** preguntas y respuestas crowdsourced sobre párrafos reales de Wikipedia; la respuesta es un *span* del texto. Lenguaje natural genuino, ambigüedad real, escala (100k+ preguntas). Mide comprensión lectora extractiva sobre texto auténtico.
- **MS MARCO / CNN-Daily Mail (Hermann et al., 2015):** QA sobre consultas reales de usuarios y artículos periodísticos; comprensión a escala industrial.

La lección de la clase es que **ambos extremos son necesarios y complementarios**. Los datasets reales (SQuAD, MS MARCO) miden si un sistema sirve para el mundo real, pero su fracaso es opaco: cuando un modelo se equivoca, no sabemos si fue por coreferencia, deducción, falta de sentido común o ruido del crowdsourcing. bAbI invierte esa propiedad: sacrifica el realismo para ganar **interpretabilidad diagnóstica**. Es precisamente la analogía con el *software testing* que el paper propone — bAbI son los *unit tests* de la comprensión de lenguaje, mientras que SQuAD/MS MARCO son las *pruebas de integración* en producción.

Para un practitioner, la moraleja transferible (y muy alineada con la práctica de QA y de sistemas de matching/retrieval) es: antes de optimizar una métrica agregada sobre datos reales, conviene tener un conjunto de pruebas sintéticas que aíslen capacidades, para saber *por qué* algo falla y no solo *que* falla.

## 11. Notas y enlaces

- **arXiv:** https://arxiv.org/abs/1502.05698 (versión v10, 31-dic-2015; bajo revisión para ICLR 2016).
- **Dataset:** originalmente `http://fb.ai/babi`; generador de tareas en `https://github.com/facebook/bAbI-tasks`.
- **Modelo base relacionado:** Weston, Chopra & Bordes, *Memory Networks*, arXiv:1410.3916 (2014).
- **Continuaciones directas citadas en el paper:**
  - Sukhbaatar et al., *End-to-End Memory Networks (MemN2N)*, NIPS 2015.
  - Kumar et al., *Ask Me Anything: Dynamic Memory Networks*, arXiv:1506.07285 (2015).
  - Peng et al., *Towards Neural Network-Based Reasoning (Neural Reasoner)*, arXiv:1508.05508 (2015).
- **Datasets reales complementarios citados:** Hermann et al. (2015, CNN/Daily Mail, NIPS); Bordes et al. (2015, large-scale simple QA); Hill et al. (2015, Children's Book Test, arXiv:1511.02301).
- **Antecedentes de tareas sintéticas:** Minsky & Papert (1969, XOR); Graves et al. (2014, Neural Turing Machines); Halevy et al. (2009, *The unreasonable effectiveness of data*).
- **Benchmarks de QA contemporáneos referenciados:** ARISTO (Allen Institute for AI); MCTest (Richardson et al., 2013); Winograd Schema Challenge (Levesque et al., 2011).
- **Detalle clave de evaluación:** umbral de éxito ≥ 95% de exactitud; 1000 ejemplos de train y 1000 de test por tarea; un único modelo evaluado en las 20 tareas sin tuning por tarea (ambos valores declarados arbitrarios por los autores).
