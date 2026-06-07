---
title: "bAbI (20 Prerequisite Toy Tasks for QA)"
weight: 119
math: true
---

{{< paper-card
    title="Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    authors="Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merriënboer, Armand Joulin, Tomas Mikolov"
    year="2015"
    venue="arXiv 1502.05698 (ICLR 2016)"
    pdf="/papers/babi-weston-2015.pdf"
    arxiv="1502.05698" >}}
bAbI propone descomponer la comprensión de lenguaje en **20 tareas sintéticas de question answering**, cada una aislando *una* habilidad de razonamiento (encadenar hechos, contar, deducir, resolver coreferencia, buscar rutas). El principio de diseño es el del **software testing**: cada tarea es un caso de prueba mínimo, y si un modelo falla en ella sabemos *exactamente* qué le falta. Todas las tareas se generan con un **simulador tipo aventura de texto** que ancla el lenguaje en un mundo controlado, de modo que la etiqueta verdadera se conoce por construcción y no hay ruido (un humano puede alcanzar 100%). El paper evalúa **Memory Networks** y extensiones contra baselines N-gram, LSTM y un SVM con recursos externos, mostrando que la memoria externa direccionable encaja naturalmente con tareas que requieren recuperar y encadenar hechos dispersos.
{{< /paper-card >}}

---

## El problema -- QA como proxy de "AI-complete"

El objetivo de fondo de los autores (Facebook AI Research) es construir agentes de diálogo. El obstáculo es que **evaluar diálogo abierto automáticamente es muy difícil**: no existe una métrica limpia que diga si una respuesta conversacional es "correcta". El paper hace entonces un movimiento estratégico: en lugar de evaluar diálogo, evalúa **comprensión lectora vía question answering**. En escenarios de verdadero/falso, opción múltiple o respuesta de una palabra, la corrección es inequívoca y se mide trivialmente como acierto o error.

La tesis es que QA es un dominio extraordinariamente amplio: casi cualquier tarea imaginable puede plantearse en este formato, lo que lo convierte en un marco unificador. El término **"AI-complete"** del título es deliberado: la comprensión plena del lenguaje es, en la práctica, tan difícil como la IA en general. Pero en lugar de atacar ese problema monolítico, los autores proponen **descomponer la comprensión en habilidades atómicas medibles**, cada una aislada en su propia tarea.

El argumento epistemológico es agudo. Citando a Halevy et al. (2009) -- *"the unreasonable effectiveness of data"* -- los autores observan que con grandes volúmenes de datos reales los investigadores tienden a converger hacia modelos simples, porque "modelos simples con muchos datos vencen a modelos elaborados con menos datos". Un N-gram funciona bien relativo a sus competidores, pero está lejos de *entender* el texto. La conclusión: como comunidad podemos quedar atrapados en mínimos locales del espacio de algoritmos, y los datos sintéticos son una vía para romper ese estancamiento, porque permiten construir tareas diagnósticas donde el fracaso de un modelo es **interpretable**. El nombre "bAbI" alude a un *baby AI*: el conjunto mínimo de competencias que cualquier sistema con aspiraciones de comprensión debería dominar primero.

---

## Idea central -- 20 tareas, una habilidad por tarea

El principio de diseño es explícitamente análogo al **software testing**: cada tarea es idealmente un "caso de prueba hoja", lo más independiente posible de las demás, que ejercita de la forma más simple un aspecto del comportamiento deseado. Tareas más complejas pueden construirse combinando habilidades. La promesa es que si un sistema falla en una tarea, sabemos *exactamente* qué habilidad le falta -- algo imposible en datasets reales donde cada pregunta mezcla coreferencia, deducción, sentido común y ruido.

Todas las tareas son **sin ruido**: un humano que lea el idioma puede en principio alcanzar 100% de exactitud. No requieren formación en semántica formal, lógica ni representación de conocimiento. La supervisión de entrenamiento incluye la respuesta verdadera y los **hechos de soporte** (supporting facts) relevantes. Las respuestas se limitan a una sola palabra (`¿Dónde está Mary? A: office`) o una lista de palabras.

| # | Tarea | Habilidad aislada |
|---|---|---|
| 1 | Single Supporting Fact | Recuperar la respuesta de un único hecho entre distractores |
| 2 | Two Supporting Facts | Encadenar dos hechos |
| 3 | Three Supporting Facts | Encadenar tres hechos |
| 4 | Two Argument Relations | Distinguir sujeto/objeto; sensibilidad al orden de palabras |
| 5 | Three Argument Relations | Distinguir dador/receptor/objeto (relación ternaria) |
| 6 | Yes/No Questions | Responder verdadero/falso |
| 7 | Counting | Contar objetos con una propiedad |
| 8 | Lists/Sets | Producir un conjunto de respuestas |
| 9 | Simple Negation | Modelar negaciones |
| 10 | Indefinite Knowledge | Posibilidad vs. certeza (respuesta "maybe") |
| 11 | Basic Coreference | Resolver el referente más cercano de un pronombre |
| 12 | Conjunction | Múltiples sujetos en una oración |
| 13 | Compound Coreference | Pronombre que refiere a múltiples actores |
| 14 | Time Reasoning | Interpretar expresiones temporales |
| 15 | Basic Deduction | Deducción por herencia de propiedades (silogismo) |
| 16 | Basic Induction | Inducción por herencia de propiedades |
| 17 | Positional Reasoning | Razonamiento espacial (estilo SHRDLU) |
| 18 | Size Reasoning | Tamaños relativos de objetos |
| 19 | Path Finding | Encontrar la ruta entre ubicaciones (búsqueda) |
| 20 | Agent's Motivations | Inferir por qué un agente actúa (estados mentales) |

bAbI se posiciona cerca del **Winograd Schema Challenge** en cuanto a la interpretabilidad directa de los resultados, pero se diferencia en dos puntos: las tareas son **autocontenidas** (vienen con datos de entrenamiento *y* de evaluación) y son **más diversas**. A diferencia de MCTest (660 historias, entrenamiento demasiado pequeño), bAbI permite controlar la cantidad de ejemplos y garantiza que el conocimiento necesario para el test esté contenido en el train.

Un ejemplo concreto de la **Tarea 1**:

```
Mary went to the bathroom.
John moved to the hallway.
Mary travelled to the office.
Where is Mary? A: office
```

Y un ejemplo de razonamiento, **Tarea 15 (Basic Deduction)**:

```
Sheep are afraid of wolves.
Cats are afraid of dogs.
Mice are afraid of cats.
Gertrude is a sheep.
What is Gertrude afraid of? A: wolves
```

---

## Generación sintética -- el simulador

Todas las tareas se generan con un **simulador que se comporta como un juego clásico de aventura de texto**, en la tradición de Bordes et al. (2010) y Weston et al. (2014). La idea es **anclar (ground)** el lenguaje en un mundo artificial coherente, donde las etiquetas verdaderas se conocen por construcción.

El mundo se compone de **entidades** de varios tipos (locaciones, objetos, personas) y de **acciones** sobre ellas. Las entidades tienen estados internos: ubicación, qué objetos cargan, estado mental de los actores (hambriento, etc.) y propiedades como tamaño, color y comestibilidad. Para las locaciones se codifican las conexiones espaciales. El repertorio de acciones incluye `go`, `get`, `put`, `give`, `drop`, `set`, `look`, `inventory`, `examine`. Una secuencia de comandos define una "historia" ejecutable (`joe go playground; joe get football`), y el sistema, que tiene acceso al estado interno del mundo, calcula la respuesta verdadera de forma trivial.

Para producir **variedad léxica** se aplica una gramática automática simple: a cada verbo se le asignan sinónimos (`get` $\to$ *picked up*, *got*, *grabbed*, *took*), y actores pueden sustituirse por pronombres (*Daniel* $\to$ *he* en la tarea 11). El control total del mundo permite: (i) conocer la etiqueta por grounding, sin anotación manual; (ii) generar tantos ejemplos como se quieran, midiendo cuántos hacen falta para resolver cada tarea; (iii) aislar habilidades para diagnóstico de fallos; y (iv) cerrar un **feedback loop** donde nuevas tareas se diseñan, incluso de forma adversarial, para romper los modelos recién propuestos.

El paper también libera las tareas (i) en hindi y (ii) con palabras en inglés barajadas (ilegibles para humanos). Un buen algoritmo debería rendir parecido en las tres versiones -- lo que *no* ocurre con métodos que dependen de recursos externos específicos de un idioma. Los autores son explícitos sobre los límites: oraciones cortas, poco anidamiento, vocabulario de 150 palabras (típicamente 4 actores, 6 locaciones, 3 objetos por tarea). Y subrayan que estas tareas **no sustituyen** datos reales, sino que los **complementan**.

---

## Modelos -- Memory Networks

Los modelos se agrupan en **tres pistas de supervisión**: *weakly supervised* (solo pares pregunta-respuesta), *strong supervision* (además los hechos de soporte en entrenamiento, no en test) y *external resources* (datos etiquetados de otras fuentes). El protocolo: **1000 preguntas de train y 1000 de test por tarea**, con una tarea considerada **"pasada" si supera el 95% de exactitud** (umbral declarado arbitrario). La consigna metodológica clave: **un único modelo debe evaluarse en las 20 tareas sin tuning por tarea**.

Los métodos comparados:

1. **N-gram classifier** (baseline débil): bag-of-N-grams sobre las oraciones que comparten palabra con la pregunta, más un clasificador lineal.
2. **LSTM** (débil): lee la historia hasta la pregunta y emite la respuesta.
3. **Memory Networks (MemNN)** (Weston et al., 2014, supervisión fuerte): un controlador neuronal realiza inferencia sobre memorias (oraciones previas). El modelo original hace **2 hops**: encuentra el primer hecho con máximo match contra la pregunta, luego el segundo con máximo match respecto a la pregunta *y* el primer hecho.
4. **Extensiones de MemNN** propuestas aquí: **Adaptive Memories (AM)** -- número variable de hops y salida multi-palabra; **N-grams (NG)** -- bag de 3-gramas para capturar orden; **Nonlinearity (NL)** -- red de 2 capas con $\tanh$ en el matching.
5. **Structured SVM** (recursos externos): cascada NLP clásica con coreferencia de Stanford y SRL de SENNA, búsqueda exhaustiva de hasta tres hechos de soporte.

La función de scoring de MemNN tiene forma de modelo de embedding:

$$s(x, y) = \Phi_x(x)^\top U^\top U\, \Phi_y(y)$$

donde $U$ es una matriz $n \times D$. El módulo $O$ recupera $k$ memorias de soporte mediante $\arg\max$ y el módulo $R$ rankea palabras del diccionario para producir la respuesta. La arquitectura de **memoria externa explícita** encaja naturalmente con tareas que requieren recuperar y encadenar hechos dispersos -- justo lo que bAbI mide. El N-gram y el LSTM no tienen ese mecanismo de recuperación direccionable, y por eso quedan muy por detrás.

---

## Resultados

Exactitud de test media (%) sobre las 20 tareas, según la Tabla 3 del paper:

| Método | Media |
|---|---|
| N-gram (baseline débil) | 34 |
| LSTM (débil) | 49 |
| MemNN original (2014, fuerte) | 79 |
| MemNN AM+NG+NL (este paper, fuerte) | **93** |
| Structured SVM (recursos externos) | 87 |
| MemNN AM+NG+NL multitask | 92 |

Las MemNN estándar superan claramente a los baselines, pero la versión original **falla** (< 95%) en varias tareas. Los **fallos esperados** vienen de las limitaciones de modelado: con $k=2$ hechos, respuestas de una palabra y bag-of-words, no resuelve las tareas 3, 4, 5, 7, 8 y 18. Los **fallos inesperados** son yes/no (6) e indefinite knowledge (10): la función de scoring *lineal* no puede modelar el match entre query, hecho de soporte y respuesta, porque eso requiere **interacciones de tres vías** -- exactamente lo que arregla la no-linealidad.

Las extensiones son complementarias: **AM** ayuda en tareas de más de dos hechos (3, 16); **NG** cuando importa el orden de palabras (4, 15); **NL** resuelve las interacciones de tres vías (6, 10). La combinación **AM+NG+NL promueve 9 tareas de fracaso a éxito** respecto a la MemNN original. El Structured SVM, pese a sus recursos externos, no la supera (también falla en 9 tareas), aunque su búsqueda **no-greedy** ayuda en path finding (19).

Quedan como **problemas abiertos**: counting (7) y lists/sets (8) requieren muchos más de 1000 ejemplos; positional reasoning (17) y path finding (19) no se resuelven **ni con 10000 ejemplos**, porque exigen un **algoritmo de búsqueda general** del que todos los métodos carecen. El entrenamiento **multitask** (media 92) es alentadoramente similar a entrenar tarea por tarea (93), mostrando que un solo modelo puede aprender múltiples aspectos de comprensión simultáneamente.

---

## Limitaciones -- el costo de ser sintético

Los propios autores son francos sobre los límites:

- **Sintético $\neq$ lenguaje real.** Oraciones cortas, poco anidamiento, vocabulario de 150 palabras. La complejidad sintáctica, la ambigüedad genuina y la riqueza del lenguaje natural quedan fuera. *"Estas tareas no son un sustituto de datos reales, sino que deben complementarlos."*
- **Riesgo de overfitting a la gramática del simulador.** Un modelo podría aprender los patrones del *generador* en vez de razonamiento genuino. Las versiones en hindi y con palabras barajadas son una mitigación parcial.
- **Supervisión más fuerte de lo realista.** Los mejores resultados usan los **hechos de soporte** en entrenamiento, rara vez disponibles en escenarios reales. En el caso débilmente supervisado con $\le 1000$ ejemplos no se conocía ningún método general que resolviera las tareas.
- **Cobertura incompleta del razonamiento.** Un análisis completo de inducción y deducción queda fuera de alcance; las tareas 15-16 son versiones básicas.

---

## Por qué importa hoy

El valor duradero de bAbI no es como dataset "difícil" -- los modelos modernos lo resuelven casi por completo -- sino como **instrumento de diagnóstico**: su capacidad de aislar exactamente qué habilidad le falta a un modelo y de cerrar el feedback loop entre diseño de tareas y diseño de algoritmos.

Desde su publicación, bAbI influyó directamente en una línea de arquitecturas con memoria: **End-to-End Memory Networks (MemN2N)** de Sukhbaatar et al. (2015), entrenables de extremo a extremo y *débilmente supervisadas* -- eliminando la principal crítica de realismo del paper original; **Dynamic Memory Networks** de Kumar et al. (2015), con atención episódica iterativa; y trabajos posteriores como Recurrent Entity Networks y Relation Networks. Durante años, "resolver bAbI" (las 20 tareas con un solo modelo débilmente supervisado) fue un hito de referencia para nuevas arquitecturas de razonamiento, y el formato historia-pregunta-respuesta-soporte se volvió canónico.

Más ampliamente, bAbI consolidó el paradigma de las **arquitecturas con memoria externa direccionable** -- junto a las Neural Turing Machines (Graves et al., 2014) -- como una línea central de la agenda de *reasoning* en deep learning. La moraleja transferible para un practitioner: antes de optimizar una métrica agregada sobre datos reales, conviene tener un conjunto de pruebas sintéticas que aíslen capacidades, para saber *por qué* algo falla y no solo *que* falla.

---

## Conexión con la Clase 24 -- QA sintético vs. realista

La [Clase 24](/clases/clase-24) (Question Answering / Reading Comprehension) lista bAbI entre los datasets de referencia. bAbI cumple un rol pedagógico específico: representa el extremo **sintético y diagnóstico** del espectro de QA, en oposición directa a los datasets **reales y a escala** que el resto de la clase introduce.

El contraste es nítido:

- **bAbI (sintético/diagnóstico):** mundo cerrado generado por simulador, 150 palabras de vocabulario, respuestas de una palabra, supervisión sobre hechos de soporte, una habilidad por tarea. Responde *"¿qué tipo exacto de razonamiento le falta a mi modelo?"*. El fracaso es interpretable.
- **[SQuAD](/papers/squad-rajpurkar-2016) (Rajpurkar et al., 2016):** preguntas crowdsourced sobre párrafos reales de Wikipedia; la respuesta es un *span* del texto. Lenguaje natural genuino, ambigüedad real, escala (100k+ preguntas). Su fracaso es opaco.
- **[CNN/Daily Mail](/papers/cnn-dailymail-hermann-2015) (Hermann et al., 2015) y [Children's Book Test](/papers/childrens-book-test-hill-2016) (Hill et al., 2016):** QA de tipo cloze sobre artículos periodísticos y libros, de la misma era que bAbI pero anclados en texto real.

La lección de la clase es que **ambos extremos son necesarios y complementarios**. Los datasets reales miden si un sistema sirve para el mundo, pero cuando un modelo se equivoca no sabemos si fue por coreferencia, deducción, falta de sentido común o ruido del crowdsourcing. bAbI invierte esa propiedad: sacrifica el realismo para ganar **interpretabilidad diagnóstica**. Es la analogía con el *software testing* que el paper propone -- bAbI son los *unit tests* de la comprensión de lenguaje, mientras que SQuAD/CNN-DM son las *pruebas de integración* en producción.

---

## Notas y enlaces

- El paper (versión v10, 31-dic-2015) estuvo bajo revisión para ICLR 2016. Tareas originalmente en `http://fb.ai/babi`; generador en [facebook/bAbI-tasks](https://github.com/facebook/bAbI-tasks).
- Modelo base relacionado: Weston, Chopra & Bordes, *Memory Networks*, arXiv:1410.3916 (2014).
- Continuaciones directas: Sukhbaatar et al., *End-to-End Memory Networks* (NIPS 2015); Kumar et al., *Dynamic Memory Networks* (arXiv:1506.07285); Peng et al., *Neural Reasoner* (arXiv:1508.05508).
- Detalle de evaluación: umbral de éxito $\ge 95\%$; 1000 ejemplos de train y 1000 de test por tarea; un único modelo en las 20 tareas sin tuning (ambos valores declarados arbitrarios).

Ver fundamentos: [Question Answering](/fundamentos/question-answering).

Ver papers: [CNN/Daily Mail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015) - [Children's Book Test (Hill 2016)](/papers/childrens-book-test-hill-2016) - [SQuAD (Rajpurkar 2016)](/papers/squad-rajpurkar-2016).

Ver clase: [Clase 24 -- Question Answering](/clases/clase-24).
