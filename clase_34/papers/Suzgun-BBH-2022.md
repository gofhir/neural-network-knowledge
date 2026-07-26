# Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them (BBH) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them*.
- **Autores:** Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou y Jason Wei. La mayoría en **Google Research**; Suzgun afiliado a la **Universidad de Stanford**.
- **Año / preprint:** 2022. arXiv:2210.09261v1 (17 de octubre de 2022).
- **Datos y código:** disponibles públicamente en `github.com/suzgunmirac/BIG-Bench-Hard` (datos, prompts y salidas del modelo Codex).
- **Linaje:** es la respuesta directa a dos trabajos previos del mismo ecosistema: el benchmark **BIG-Bench** (Srivastava et al., 2022) y el paper de **chain-of-thought prompting** (Wei et al., 2022b). Combina ambos: toma las tareas más difíciles de BIG-Bench y les aplica CoT.

El paper parte de una observación incómoda. BIG-Bench fue diseñado explícitamente para reunir tareas "más allá de las capacidades de los modelos de lenguaje actuales", pero al escalar **PaLM a 540B parámetros**, el mejor modelo ya superaba el desempeño promedio de evaluadores humanos (*average human-rater*) en el **65%** de las tareas. Quedaba entonces una pregunta abierta: en las tareas donde los modelos aún se quedan cortos frente al humano promedio, ¿es que esas tareas son fundamentalmente irresolubles con la tecnología actual, o simplemente se las estaba evaluando con la técnica de prompting equivocada?

Para responderla, los autores curan **BIG-Bench Hard (BBH)**: un subconjunto de **23 tareas** (27 subtareas) de BIG-Bench para las cuales **ningún modelo previo había superado al humano promedio**. Sobre este subconjunto aplican **chain-of-thought (CoT) prompting** —insertar pasos de razonamiento intermedios en los ejemplos few-shot— y lo comparan contra el prompting estándar *answer-only* (solo respuesta). El resultado central: CoT permite a **PaLM 540B superar al humano promedio en 10 de 23 tareas** y a **Codex (`code-davinci-002`) en 17 de 23**, frente a apenas 5 de 23 con answer-only. Muchas de estas tareas requieren razonamiento multi-paso, de modo que el prompting sin CoT usado en el BIG-Bench original **subestimaba sistemáticamente** las capacidades reales de los modelos.

Para la **Clase 34 (Razonamiento)**, este paper es la evidencia empírica citada en la slide 29: con chain-of-thought, Codex supera al humano promedio en 17 de 23 tareas de BBH, tareas donde antes los LLMs no lo lograban. Es el puente entre "los modelos parecen no razonar bien" y "los modelos sí razonan, pero hay que darles el espacio para hacerlo paso a paso".

## 2. Contexto: evaluar razonamiento y el subconjunto "hard"

### 2.1. Qué es BIG-Bench

**BIG-Bench** (*Beyond the Imitation Game Benchmark*, Srivastava et al., 2022) es un benchmark colaborativo que busca medir cuantitativamente las capacidades y límites de los modelos de lenguaje. Reúne **más de 200 tareas** basadas en texto (209 en total según el paper de BBH), aportadas por la comunidad (*crowd-sourced*), abarcando categorías que van desde NLP tradicional hasta matemáticas, razonamiento de sentido común y respuesta a preguntas. Su diversidad es intencional: cubrir un abanico amplio de habilidades para no sobreajustar la evaluación a un solo tipo de competencia.

Un elemento crucial de BIG-Bench es que los organizadores reclutaron **evaluadores humanos** para resolver manualmente cada tarea y puntuarla contra las etiquetas correctas (*golden labels*), estableciendo así **baselines humanos**. El paper de BBH reporta, para las 23 tareas seleccionadas, un **humano promedio de 67.7%** y un **humano máximo de 94.4%**. Estos puntajes no son representativos de toda la población —los propios organizadores advierten cautela al interpretarlos—, pero sí expresan la **dificultad empírica** de cada tarea: cuán difícil es para una persona, y por tanto, probablemente, para un modelo.

### 2.2. Por qué existe un subconjunto "hard"

La paradoja que motiva el trabajo es esta: aunque BIG-Bench se creó para desafiar a los modelos, PaLM 540B con five-shot prompting ya vencía al humano promedio en el 65% de las tareas. La pregunta natural es entonces qué caracteriza al 35% restante. ¿Son tareas que ningún escalamiento resolverá, o tareas que exigen un **protocolo de prompting distinto** al few-shot estándar del BIG-Bench original?

Los autores identifican las tareas donde el mejor desempeño reportado de cualquier modelo queda **por debajo** del humano promedio, y de ahí construyen BBH. La hipótesis de trabajo es que buena parte de esas tareas involucran **razonamiento en múltiples pasos** (aritmético, lógico, simbólico, espacial, temporal), y que el prompting answer-only obliga al modelo a "saltar" directamente a la respuesta sin espacio para computar los pasos intermedios, ocultando así su capacidad real.

## 3. Contribución: BBH como benchmark

La contribución del paper es doble:

1. **Un benchmark curado y difícil (BBH).** No cualquier tarea difícil, sino un subconjunto **limpio, tratable y desafiante**: 23 tareas con datos y metadatos de calidad, baselines humanos, métricas objetivas (opción múltiple o coincidencia exacta) y donde ningún modelo previo había vencido al humano promedio. Se libera públicamente con datos, prompts y salidas del modelo, de modo que sirva como estándar reproducible para medir progreso en razonamiento. BBH se volvió, de hecho, uno de los benchmarks de referencia para evaluar razonamiento en LLMs en los años siguientes.

2. **Una demostración metodológica.** El paper muestra que la elección del protocolo de prompting cambia radicalmente la conclusión sobre lo que un modelo "puede" hacer. Answer-only prompting **subestima** las capacidades; CoT las revela. Esto tiene una implicación epistemológica importante: los rankings de un benchmark no miden solo al modelo, sino la interacción modelo-protocolo, y usar el protocolo equivocado puede llevar a declarar prematuramente que una tarea es irresoluble.

Un tercer aporte, más analítico, es el estudio de la **interacción entre CoT y escala**: CoT es una estrategia *emergente* que solo aporta ganancias en modelos suficientemente grandes, y puede "desbloquear" desempeño en tareas cuya curva de escalamiento era plana con answer-only.

## 4. Método

### 4.1. Selección de las 23 tareas

El criterio de selección es explícito y reproducible. Partiendo de las **209** tareas de BIG-Bench, se aplica una cascada de filtros (Tabla 1 del paper):

| Paso | # Tareas | Criterio |
|---|---|---|
| Todas | 209 | Todas las tareas de BIG-Bench |
| Filtro 1 | 187 | Descartar tareas con más de tres subtareas |
| Filtro 2 | 130 | Descartar tareas con menos de 103 ejemplos (3 para few-shot, 100 para evaluación) |
| Filtro 3 | 85 | Descartar tareas sin baseline humano |
| Filtro 4 | 78 | Descartar tareas que no usan opción múltiple ni coincidencia exacta como métrica |
| Filtro 5 | 36 | Descartar tareas donde el mejor modelo ya vence al humano promedio |
| Filtro 6 | 23 | Descartar tareas extremadamente difíciles, fuera del alcance del trabajo |

El punto clave es el **Filtro 5**: de las 78 tareas limpias con métrica objetiva, el mejor modelo ya superaba al humano promedio en **42**, dejando **36** donde ningún modelo lo lograba. Ese es el corazón del "hard": tareas donde, hasta ese momento, la máquina perdía contra la persona promedio.

De esas 36, se descartan **13** por ser **extremadamente difíciles o fuera de alcance**, según inspección manual de los propios autores. Los ejemplos son ilustrativos de *por qué* se descartan: *Checkmate in One* exige rastrear el estado de un tablero de ajedrez a lo largo de una secuencia larga de jugadas (imposible para no ajedrecistas); *Real or Fake Text* tiene entradas demasiado largas; *Moral Permissibility* tiene una formulación ambigua y, además, los autores prefieren no delegar en un modelo juicios morales. Estas 13 se dejan como trabajo futuro para modelos o métodos más potentes.

Las **23 tareas restantes** forman BBH. Dos de ellas —*Logical Deduction* y *Tracking Shuffled Objects*— tienen tres subtareas cada una (de ahí las 27 subtareas). Para todas menos tres tareas se toma una muestra aleatoria de **250 ejemplos** de evaluación; para *Causal Judgement*, *Penguins in a Table* y *Snarks* se usan todos los ejemplos disponibles (187, 146 y 178 respectivamente). En total hay **6.511 ejemplos** de evaluación. Como referencia de costo, los autores estiman que evaluar BBH con `text-davinci-002` costaría **195,33 USD** al precio de OpenAI de entonces (0,02 USD por cada 1.000 tokens).

Las 23 tareas se agrupan además en dos familias que atraviesan todo el análisis: **algorítmicas** (11 tareas, marcadas con superíndice λ en la Tabla 3 del paper: pueden resolverse con un algoritmo basado en reglas sin usar NLP) y de **NLP / lenguaje natural** (12 tareas). Ejemplos algorítmicos: *Boolean Expressions*, *Multi-Step Arithmetic*, *Navigate*, *Word Sorting*, *Dyck Languages*, *Object Counting*, *Tracking Shuffled Objects*, *Web of Lies*. Ejemplos de NLP: *Causal Judgement*, *Disambiguation QA*, *Hyperbaton* (orden de adjetivos), *Snarks* (detección de sarcasmo), *Movie Recommendation*, *Ruin Names*.

### 4.2. Los dos protocolos de prompting

El paper contrasta dos configuraciones few-shot. Ambas incluyen en el prompt una **descripción de la tarea** y las **opciones de respuesta** (Figura 3), lo cual ya es un baseline reforzado respecto del BIG-Bench original.

**Answer-only prompting (AO).** El protocolo estándar de Brown et al. (2020): se anteponen varios ejemplos de entrada-salida (pregunta → respuesta directa) antes de la pregunta de inferencia. El modelo debe producir la respuesta sin mostrar trabajo intermedio. Se refuerza incluyendo instrucciones y el espacio de respuestas posible, lo que se sabe que mejora el desempeño (Min et al., 2022b: los modelos se benefician fuertemente de conocer el espacio de salida deseado).

**Chain-of-thought prompting (CoT).** Los ejemplos few-shot se aumentan con una **cadena de razonamiento** (pasos intermedios) antes de la respuesta final. Los autores **escriben manualmente tres exemplars CoT por cada tarea** de BBH (todos los prompts se liberan en el apéndice). A cada anotación CoT se le antepone la frase **"Let's think step by step"** (Kojima et al., 2022). La Figura 2 muestra ejemplos reales: en *Navigate*, el modelo rastrea coordenadas y orientación paso a paso —"empezamos en el origen (0,0), mirando el eje y positivo; (1) girar a la izquierda…"— hasta concluir si volvió al punto de partida; en *Word Sorting*, ordena palabras comparando primera letra, luego segunda letra, sub-lista por sub-lista.

### 4.3. Modelos y protocolo de evaluación

Se evalúan tres familias, en varias escalas:

- **Codex** (Chen et al., 2021a): `code-cushman-001`, `code-davinci-002` (el modelo más fuerte, entrenado con código **y** texto).
- **InstructGPT** (Ouyang et al., 2022): `text-ada-001`, `text-babbage-001`, `text-curie-001`, `text-davinci-002`.
- **PaLM** (Chowdhery et al., 2022): 8B, 62B, 540B.

La **decodificación es greedy** (muestreo con temperatura $\tau = 0$). La respuesta final se extrae buscando la frase clave que el modelo debe producir ("*the answer is*"), y se mide **exactitud por coincidencia exacta (exact match, EM)** contra la etiqueta verdadera. Nótese que esto difiere de la clasificación por *ranking/scoring* usada en otros trabajos: aquí se le dan al modelo todas las opciones de una vez, genera una salida y se compara por EM, en lugar de puntuar la verosimilitud de cada opción por separado.

## 5. Resultados

### 5.1. El resultado central: 17 de 23 con Codex

La Tabla 2 del paper resume el desempeño agregado sobre las 23 tareas. Los baselines son: **humano promedio 67.7%**, humano máximo 94.4%, y el **mejor resultado previo de BIG-Bench 50.9%** (que vencía al humano promedio en **0 de 23** tareas, por construcción del subconjunto).

| Modelo / protocolo | BBH (23) | # tareas > humano promedio |
|---|---|---|
| Mejor resultado previo BIG-Bench | 50.9 | 0 / 23 |
| PaLM 540B — answer-only | 52.3 | 6 / 23 |
| PaLM 540B — **CoT** | **65.2** (+12.9) | **10 / 23** |
| InstructGPT (text-davinci-002) — answer-only | 51.8 | 4 / 23 |
| InstructGPT — **CoT** | **68.4** (+16.6) | **15 / 23** |
| Codex (code-davinci-002) — answer-only | 56.6 | 5 / 23 |
| Codex — **CoT** | **73.9** (+16.7) | **17 / 23** |

Tres lecturas se desprenden de la tabla:

1. **Answer-only ya subestimaba.** En el BIG-Bench original, ningún modelo (incluido PaLM 540B) vencía al humano promedio en **ninguna** tarea que cumpliera los criterios de BBH. Sin embargo, el mismo PaLM 540B con answer-only reforzado (instrucciones + opciones) en este paper ya vence en **6 de 23** y es 1,4% mejor que el resultado reportado de BIG-Bench. Es decir, parte de la "dificultad" era un artefacto del formato del prompt.

2. **CoT aporta mejoras de dos dígitos en las tres familias.** +12,9 en PaLM, +16,6 en InstructGPT, +16,7 en Codex. Para el mejor modelo, Codex, CoT lleva el conteo de tareas ganadas de **5 a 17 de 23**.

3. **Superar al promedio no es superar al experto.** Codex con CoT vence al humano **promedio** por más de 6%, pero **sigue por debajo del mejor humano** por más de 20%. Esta es una advertencia central del paper y un matiz que conviene explicitar frente a la slide 29: "supera al humano" significa **superar al evaluador humano promedio**, no al experto. Los autores subrayan que superar el promedio en un conjunto de tareas difíciles **no debe confundirse con verdadera comprensión o razonamiento**.

### 5.2. Dónde ayuda más CoT: la división NLP vs. algorítmico

La ganancia de CoT no es uniforme; se concentra en las tareas **algorítmicas / multi-paso**. Al desagregar por familia (Codex):

| Familia | Answer-only | CoT | Δ |
|---|---|---|---|
| NLP (12 tareas) | 66.4 | 73.5 | **+7.1** |
| Algorítmicas (11 tareas) | 45.9 | 74.4 | **+28.5** |

El contraste es dramático: en las tareas algorítmicas Codex pasa de 45,9% (peor que el humano promedio de 63,5% para esa familia) a 74,4% (por encima), un salto de **+28,5 puntos**. En las de NLP la ganancia existe pero es modesta (+7,1). La razón es intuitiva: CoT facilita la **descomposición de problemas complejos multi-paso en subproblemas más pequeños y resolubles**, que es exactamente lo que exige la aritmética, la lógica o el rastreo de estado, pero aporta poco cuando la tarea depende de conocimiento del mundo o de matices semánticos que el razonamiento explícito no puede generar de la nada.

Las mayores ganancias por tarea individual con Codex ilustran el punto: **Tracking Shuffled Objects (+60,4)**, **Multi-Step Arithmetic (+46,4)**, **Navigate (+46,0)**, **Temporal Sequences (+19,8)**. Todas son tareas donde hay que ejecutar un procedimiento paso a paso y mantener un estado intermedio. La Figura 6 muestra el mecanismo: en *Multi-Step Arithmetic*, el modelo descompone `((4 + 7 * 4 - -5) - (-4 - 1 - -4 - 4))` en `A - B`, calcula cada subexpresión respetando el orden de operaciones, y llega a 42; en *Hyperbaton*, enumera la jerarquía canónica de adjetivos (opinión, tamaño, edad, forma, color, origen, material, propósito) y verifica cada opción contra ella.

Los autores agrupan las tareas en cuatro categorías cualitativas según qué demandan:

- **Razonamiento algorítmico y aritmético multi-paso** (la mayoría de BBH): aritmético, lógico (*Boolean Expressions*, *Logical Deduction*), geométrico (*Geometric Shapes*), jerárquico (*Dyck Languages*), espacial (*Navigate*) y temporal (*Temporal Sequences*). Aquí Codex —entrenado con código— destaca al explotar patrones algorítmicos de los exemplars.
- **Comprensión de lenguaje natural**: desambiguación, resolución de entidades, reglas gramaticales, sarcasmo (*Disambiguation QA*, *Hyperbaton*, *Snarks*, *Salient Translation Error Detection*). Aquí PaLM e InstructGPT (entrenados sobre todo con lenguaje natural) suelen superar a Codex.
- **Uso de conocimiento del mundo**: hechos y suposiciones culturales (*Sports Understanding*, *Movie Recommendation*, *Date Understanding*, *Causal Judgement*, *Ruin Names*). CoT ayuda en algunas (*Sports Understanding*, *Movie Recommendation*) pero **empeora** en *Causal Judgement* y *Ruin Names*.
- **Conocimiento y razonamiento multilingüe**: una sola tarea, *Salient Translation Error Detection*, donde la mejora por CoT solo aparece en PaLM.

### 5.3. CoT desbloquea desempeño emergente

Un análisis secundario, pero conceptualmente importante, estudia la **interacción entre CoT y escala** (Figura 4). Para los modelos pequeños (de `text-ada-001` a `text-curie-001`, y PaLM 8B), CoT tiene ganancia **negativa o nula**: razonar en voz alta no ayuda si el modelo no es lo bastante capaz. La brecha entre CoT y answer-only **crece con la escala** hasta el modelo más grande. CoT es, por tanto, una **estrategia de prompting emergente** (Wei et al., 2022a): solo funciona por encima de cierto tamaño.

Más aún, en tres tareas cuya **curva de escalamiento es plana** con answer-only —*Multi-Step Arithmetic*, *Tracking Shuffled Objects*, *Web of Lies*— el desempeño se mantiene cerca del azar sin importar la escala; pero **con CoT** el desempeño transita de aleatorio a claramente por encima del azar a medida que el modelo crece (Figura 5). Esto es *emergent task performance*: CoT no solo mejora tareas ya resueltas, sino que **habilita** tareas que el escalamiento por sí solo no resolvía. La excepción notable es *Causal Judgement* (57,8% con answer-only, sin mejora con CoT; azar 50%, humano máximo 100%): una tarea de curva plana que CoT **no** desbloquea, y que se deja como desafío para técnicas futuras.

### 5.4. Cuándo CoT falla

CoT no es una panacea. En **tres tareas** queda **por debajo** de answer-only en las tres familias de modelos: *Causal Judgement*, *Ruin Names* y *Snarks*. Dos de ellas requieren **conocimiento del mundo** (presuposiciones comunes, percepción y uso del humor) que el razonamiento explícito no puede fabricar. *Snarks* mide detección de sarcasmo: la Figura 7 muestra casos donde el modelo, al "razonar", clasifica ambas frases como neutrales y termina eligiendo al azar. Es difícil detectar sarcasmo sin contexto situacional, y verbalizar pasos no crea ese contexto faltante. La lección: **CoT amplifica el razonamiento procedimental, pero no suple conocimiento ausente ni juicio pragmático.**

## 6. Limitaciones

- **El baseline humano es frágil.** Los propios organizadores de BIG-Bench advierten no interpretar los puntajes humanos como representativos. La evaluación humana tomó casi un año, durante el cual algunas tareas cambiaron de formato y contenido; los evaluadores tenían pocas horas por día, podían usar recursos externos (Internet) y a veces la descripción de la tarea era difícil de seguir. Superar "al humano promedio" debe leerse con cautela.
- **Superar el promedio ≠ razonar.** El paper es enfático: vencer al humano promedio en tareas difíciles **no equivale a verdadera comprensión o razonamiento** del lenguaje. Codex sigue 20+ puntos por debajo del mejor humano.
- **Prompts CoT escritos a mano.** Los tres exemplars por tarea fueron compuestos manualmente por los autores; el desempeño depende de esa curación, y no se explora sensibilidad a la formulación de las cadenas.
- **Cobertura acotada.** Se descartaron 13 tareas "extremadamente difíciles" que quedan sin abordar; BBH no cubre razonamiento que exija memoria de trabajo muy larga, conocimiento especializado profundo o juicio moral.
- **Modelos opacos.** Los autores reconocen no conocer con certeza el tamaño, arquitectura ni corpus de entrenamiento de Codex e InstructGPT (no saben si son modelos únicos, mezclas de expertos o ensembles), lo que limita conclusiones sobre *por qué* Codex es superior en tareas algorítmicas.

## 7. Conexión con la Clase 34 (Razonamiento) y con CoT (Wei et al., 2022)

BBH es la contraparte empírica de la tesis central de la Clase 34: los LLMs **sí pueden razonar en múltiples pasos, pero necesitan que el prompt les dé espacio para hacerlo**. La slide 29 cita el resultado clave —Codex con CoT supera al humano en 17 de 23 tareas de BBH, tareas donde antes los LLMs no lo lograban— y este paper es su fuente. El matiz didáctico a transmitir es que ese "humano" es el evaluador **promedio**, no el experto, y que la ganancia se concentra en tareas **algorítmicas / simbólicas multi-paso**, no en las que dependen de conocimiento o pragmática.

La relación con **chain-of-thought prompting (Wei et al., 2022b)** es de complementariedad. Wei et al. **introducen** la técnica y muestran que aumentar los ejemplos few-shot con pasos de razonamiento intermedios mejora tareas de aritmética, sentido común y manipulación simbólica. BBH la **estresa**: la aplica al conjunto de tareas deliberadamente más difícil de un benchmark diverso y de gran escala, y demuestra que ahí es precisamente donde CoT rinde sus mayores frutos (+28,5 puntos en tareas algorítmicas con Codex). Ambos papers comparten autores (Jason Wei, Denny Zhou, Ed Chi, Quoc Le) y forman una unidad conceptual: uno propone el método, el otro prueba que **cambia la conclusión sobre qué tareas son "resolubles"**. BBH también dialoga con *emergent abilities* (Wei et al., 2022a): CoT es una estrategia emergente que solo aparece a escala suficiente, y que puede aplanar-a-emergente curvas de escalamiento antes muertas.

Para el curso, el mapa mental es: **razonamiento en LLMs = capacidad latente + protocolo que la revela**. Answer-only oculta; CoT revela. Y la revelación tiene un perfil claro: brilla en lo procedimental (aritmética, lógica, rastreo de estado, orden), es tibia en lo semántico y falla en lo que exige conocimiento o juicio ausente.

**Enlaces internos sugeridos:**

- Clase: [/clases/clase-34](/clases/clase-34) — Razonamiento (Sebastián Amenábar).
- Paper hermano — el método: [/papers/chain-of-thought-wei-2022](/papers/chain-of-thought-wei-2022) — CoT prompting, que BBH lleva al límite.
- Fundamento transversal: [/fundamentos/in-context-learning](/fundamentos/in-context-learning) — few-shot prompting, espacio de salida, exemplars.
- Concepto relacionado: emergencia a escala (Wei et al., 2022a) — capacidades que aparecen solo en modelos grandes.

## 8. Nota final: relevancia para salud

Para decisiones clínicas, la lección más útil de BBH no es que "los LLMs superan al humano en 17 de 23 tareas", sino **cómo** llegan a esa cifra y qué la limita. Primero, medir razonamiento exige tareas **deliberadamente difíciles y de múltiples pasos**, no promedios cómodos: en medicina el error se concentra justamente en los casos complejos —diagnósticos diferenciales largos, cálculo de dosis, cronologías de eventos, cadenas causales—, que son el análogo de las tareas algorítmicas de BBH. Segundo, el paper muestra que **el protocolo importa tanto como el modelo**: un LLM puede parecer incompetente con answer-only y competente con CoT, de modo que evaluar un asistente clínico sin exigirle mostrar su razonamiento paso a paso puede llevar a **subestimar o sobreestimar** su fiabilidad. Tercero, y más importante, "superar al humano promedio" **no es** superar al experto ni equivale a comprensión real: Codex seguía 20+ puntos bajo el mejor humano, y CoT **fallaba** precisamente donde faltaba conocimiento del mundo o juicio pragmático —el terreno donde vive buena parte de la decisión médica. Antes de confiar en un modelo para cualquier decisión clínica, la exigencia mínima es someterlo a un BBH del dominio: tareas difíciles, razonamiento explícito y verificable, comparación contra el mejor experto (no el promedio), y desconfianza especial en las tareas donde el razonamiento paso a paso no puede compensar el conocimiento ausente.
