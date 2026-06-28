# Tracking the World State with Recurrent Entity Networks — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Tracking the World State with Recurrent Entity Networks*.
- **Autores:** Mikael Henaff (Facebook AI Research y Courant Institute, NYU), Jason Weston, Arthur Szlam, Antoine Bordes y Yann LeCun (Facebook AI Research; LeCun también en Courant Institute, NYU).
- **Venue:** ICLR 2017 (*Published as a conference paper at ICLR 2017*).
- **Año:** 2016 (preprint) / 2017 (conferencia). **Preprint:** arXiv:1612.03969v3 (10 may 2017), [arxiv.org/abs/1612.03969](https://arxiv.org/abs/1612.03969).
- **Código:** [github.com/facebook/MemNN/tree/master/EntNet-babi](https://github.com/facebook/MemNN/tree/master/EntNet-babi). Implementación en Torch.

Este paper introduce el **Recurrent Entity Network** (abreviado **EntNet**), un modelo de red neuronal aumentada con memoria cuya tesis central es que un agente inteligente debe **mantener un modelo del mundo** (*world state*) que se actualiza continuamente a medida que recibe información, y no solo recuperarla cuando le hacen una pregunta. La frase que abre el paper —"la esencia de la inteligencia es la capacidad de predecir"— enmarca la motivación: para razonar y planificar, un agente necesita una estimación del estado actual del mundo y un modelo de cómo ese estado evoluciona ante nuevos eventos. El ejemplo canónico del paper es lingüístico: al leer "John walks out of the kitchen", el agente debería actualizar la ubicación de John, la lista de personas presentes en cada habitación y, si John llevaba una bolsa, también la ubicación de esa bolsa y el inventario de la cocina.

La contribución técnica es una **memoria dinámica de bloques** (*memory cells* o *slots*), donde idealmente cada bloque rastrea una **entidad** del mundo (una persona, un objeto, una ubicación). Cada bloque tiene dos piezas: una **clave** `w_j` (key) y un **valor** o contenido `h_j` (hidden state). Al leer cada frase, un **gate** (compuerta) decide cuáles entidades son relevantes y por tanto cuáles bloques actualizar; los bloques irrelevantes quedan intactos. El procesamiento es **online** y en **una sola pasada** sobre el texto: el modelo razona sobre la marcha (*on-the-fly*) mientras lee, en contraste con las Memory Networks (Sukhbaatar et al., 2015), que almacenan toda la secuencia de entrada y razonan solo en el momento de responder.

El resultado empírico estelar: el EntNet es el **primer modelo en resolver las 20 tareas bAbI** en el régimen de 10 000 ejemplos de entrenamiento (0 tareas falladas, error medio 0.5 %), fijando un nuevo estado del arte sobre el benchmark de comprensión de historias. Para la Clase 30 (Modelos con memoria externa) este es el paper de cierre: representa el salto desde "memoria como archivo que se consulta" hacia "memoria como estado estructurado por entidades que se mantiene vivo".

## 2. Contexto histórico: de leer la memoria a mantener un estado del mundo

Hacia 2015–2016, las redes aumentadas con memoria externa eran un área en plena efervescencia. La **Memory Network** (Weston et al., 2014) y su versión entrenable extremo a extremo, la **End-to-End Memory Network / MemN2N** (Sukhbaatar et al., 2015), habían establecido el paradigma dominante para comprensión de historias: almacenar explícitamente toda la secuencia de entrada en una matriz de memoria y, al recibir una pregunta, ejecutar varios "hops" de atención softmax sobre esas memorias para componer la respuesta. En paralelo, la **Neural Turing Machine** (Graves et al., 2014) y el **Differentiable Neural Computer** (Graves et al., 2016) ofrecían memorias de tamaño fijo con operaciones de lectura y escritura diferenciables basadas en contenido y en localización, gobernadas por un controlador central (típicamente un LSTM).

El paper diagnostica una limitación compartida por estas familias. Las Memory Networks **leen** la entrada pero no mantienen un **estado dinámico de las entidades**: vuelcan las frases a memoria de forma esencialmente pasiva y delegan todo el razonamiento al momento de la consulta, mediante atención repetida sobre el texto crudo almacenado. No hay una representación que evolucione frase a frase capturando "dónde está cada cosa ahora". La NTM/DNC sí mantienen una memoria que se reescribe, pero a través de un controlador sofisticado y secuencial que produce vectores de interfaz (claves, ponderaciones) combinados vía softmax para leer y escribir en la matriz externa; el grueso del razonamiento recae en ese controlador central.

El EntNet propone una arquitectura distinta y, en cierto sentido, más simple: en lugar de un controlador central, una **batería de RNN con compuertas que comparten parámetros**, donde cada RNN gobierna su propio bloque de memoria local sin interacción directa con los demás. El paper ofrece dos lecturas equivalentes de esta idea. (1) Múltiples "procesadores" idénticos funcionando en paralelo, cada uno con su memoria local distribuida. (2) Un banco de RNN con compuertas cuyos estados ocultos corresponden a conceptos y atributos latentes, y cuyos parámetros (compartidos) describen las **leyes del mundo** según las cuales esos atributos se actualizan. El paper traza una analogía elegante: así como el *weight tying* en una CNN refleja la invarianza de las estadísticas de la imagen a través de las localizaciones espaciales, aquí el compartir parámetros entre bloques refleja una invarianza de las leyes del mundo a través de las instancias de objetos. Una misma regla ("si una persona se mueve a un lugar, su ubicación cambia") debe aplicarse por igual a John, a Mary o a Sandra.

Otra diferencia crucial frente a la NTM/DNC: en vez de una atención softmax (que normaliza a través de las memorias y por tanto las acopla), el EntNet usa una **compuerta independiente por bloque**. Esto permite que **varias localizaciones de memoria se actualicen simultáneamente** en un solo paso, sin competir entre sí por la masa de probabilidad de un softmax. El paper también lo distingue de los modelos LSTM/GRU (celdas escalares con interacción completa entre ellas, sin término de matching contenido-entrada), del Dynamic Memory Network de Xiong et al. (2016) (que liga memorias a tokens y las actualiza secuencialmente, no en paralelo) y de las Gated Graph Networks de Li et al. (2015) (de las que el EntNet sería el caso particular de un grafo sin aristas).

## 3. Contribución central

La aportación del paper se puede resumir en cuatro puntos:

1. **Un modelo nuevo, el EntNet**, equipado con una memoria dinámica de largo plazo de tamaño fijo que mantiene y actualiza una representación del estado del mundo a medida que recibe datos nuevos.
2. **Razonamiento on-the-fly en una sola pasada.** A diferencia de la Memory Network, el EntNet razona mientras lee, no solo cuando se le pide responder. Esto lo convierte en un modelo apto para procesamiento *online* de flujos de texto.
3. **Memoria estructurada por entidades.** Cada bloque de memoria está diseñado para —e idealmente aprende a— rastrear una entidad del mundo, con su clave fija que la identifica y su contenido dinámico que codifica sus atributos (ubicación, objetos que porta, compañía).
4. **Estado del arte en bAbI.** Es el primer método en resolver las 20 tareas bAbI en el régimen de 10k ejemplos, además de resolver una tarea sintética de modelado del mundo que requiere un gran número de hechos de soporte (donde LSTM y MemN2N fracasan) y generalizar más allá de su horizonte de entrenamiento, y obtener resultados competitivos en el Children's Book Test leyendo en una sola pasada.

## 4. Método: las tres partes del modelo

El EntNet procesa datos secuenciales y consta de tres componentes: un **encoder de entrada**, una **memoria dinámica** y una **capa de salida**.

### 4.1. Encoder de entrada

El encoder resume cada elemento de la secuencia (típicamente una frase o ventana de palabras) en un vector de longitud fija `s_t`. Uno es libre de elegir cualquier encoder de secuencias estándar —bag-of-words, el estado final de un RNN, etc.—, pero los autores adoptan un encoder simple: una **máscara multiplicativa aprendida seguida de una suma**. Dada la entrada en el tiempo `t` como secuencia de embeddings de palabras `{e_1, ..., e_k}`, la representación es

> `s_t = Σ_i f_i ⊙ e_i`

donde `⊙` es el producto de Hadamard (elemento a elemento) y los vectores de máscara `{f_1, ..., f_k}` son los mismos en cada paso temporal y se aprenden junto al resto del modelo. El diseño es flexible: si todas las máscaras valen 1, el encoder degenera en un bag-of-words puro; alternativamente puede aprender un *positional encoding* como el de Sukhbaatar et al. (2015). Esta máscara permite que el modelo aprenda a ponderar posiciones dentro de la frase (por ejemplo, dar más peso al sujeto o al verbo de movimiento).

### 4.2. Memoria dinámica: gate, candidato, update y normalización

El corazón del modelo. La memoria es una red recurrente con compuertas y un esquema de *weight tying* (parcialmente) estructurado por bloques. El estado oculto completo se divide en bloques `h_1, ..., h_m`; en los experimentos `m` va de 5 a 20 bloques, cada uno de 20 a 100 unidades. A cada bloque `j` se le asocia un vector clave `w_j`. En cada paso temporal `t`, dado el input codificado `s_t`, cada bloque se actualiza mediante cuatro ecuaciones (en su forma más general):

> **(1) Gate:** `g_j ← σ(s_tᵀ h_j + s_tᵀ w_j)`
>
> **(2) Candidato:** `h̃_j ← φ(U h_j + V w_j + W s_t)`
>
> **(3) Update:** `h_j ← h_j + g_j ⊙ h̃_j`
>
> **(4) Normalización:** `h_j ← h_j / ‖h_j‖`

Cada pieza tiene un rol preciso:

- **El gate `g_j`** (con `σ` la sigmoide) determina cuánto debe actualizarse la `j`-ésima memoria. Contiene **dos términos** que el paper distingue explícitamente. El término de **contenido** `s_tᵀ h_j` abre la compuerta para los slots cuyo *contenido actual* coincide con la entrada (direccionamiento basado en contenido). El término de **localización** `s_tᵀ w_j` abre la compuerta para los slots cuya *clave* coincide con la entrada (direccionamiento basado en clave). Esta combinación es lo que permite "encontrar" la entidad correcta tanto por su nombre (clave) como por lo que sabemos de ella (contenido).
- **El candidato `h̃_j`** es el nuevo valor propuesto para la memoria, combinando el contenido previo `h_j`, la clave `w_j` y la entrada `s_t` vía las matrices `U`, `V`, `W` (parámetros entrenables **compartidos entre todos los bloques** — de ahí el *weight tying*). La activación `φ` puede ser ReLU paramétrica (PReLU; He et al., 2015) o la identidad. Las matrices pueden además fijarse a valores especiales (identidad, cero) para obtener variantes más simples del modelo.
- **El update `h_j ← h_j + g_j ⊙ h̃_j`** suma el candidato ponderado por el gate. Si `g_j ≈ 0`, el bloque queda esencialmente intacto; solo las entidades relevantes se modifican.
- **La normalización a la esfera unitaria** cumple una función sutil pero importante: permite **olvidar** información antigua. Como todas las memorias viven en la esfera unitaria, toda su información está contenida en su **fase** (dirección). Sumar cualquier vector a una memoria (que no sea ella misma) reduce la distancia coseno entre la memoria original y la actualizada; por tanto, a medida que se agrega información nueva, la antigua se va olvidando.

En el diagrama del paper (Figura 1), las ecuaciones (1)–(2) están representadas por el módulo `f_θ` (con `θ` el conjunto de parámetros entrenables), y las ecuaciones (3)–(4) por el "gate", ya que cumplen una función análoga.

### 4.3. Capa de salida: atención sobre las memorias

Cuando el modelo debe producir una salida, se le presenta un vector de consulta `q` y se aplica un módulo de atención sobre los contenidos de la memoria:

> `p_j = Softmax(qᵀ h_j)`
>
> `u = Σ_j p_j h_j`
>
> `y = R φ(q + H u)`

donde `H` y `R` son parámetros entrenables adicionales. El softmax `p_j` produce una distribución de atención sobre los bloques según su afinidad con la consulta; `u` es la lectura ponderada de las memorias; y `y` es la respuesta final. El paper observa que este módulo de salida puede verse como una **Memory Network de un solo hop** con una no-linealidad `φ` añadida entre el estado interno y la matriz decodificadora. Si los slots corresponden a palabras específicas (caso del *tied variant*), `p` se interpreta directamente como una distribución sobre respuestas candidatas. Todo el modelo —encoder, memoria y salida— se entrena por **backpropagation through time**, recibiendo gradientes en los pasos donde el lector debe producir una salida.

### 4.4. El ejemplo motivador: Mary, el balón y el jardín

El paper ilustra el funcionamiento con un ejemplo de operación que conviene retener porque es exactamente el espíritu del cierre de la Clase 30. El modelo es libre de aprender las claves `w_j`; una elección natural es **asociar un slot de memoria con cada entidad** de la historia. Es más, esta elección puede inyectarse como conocimiento previo atando (*tying*) las claves a los embeddings de palabras específicas (por ejemplo, las entidades nombradas que produciría un *tagger*) — esta es la **variante "tied"** de los experimentos, que tiene la ventaja de poder manejar entidades no vistas en entrenamiento siempre que sus embeddings se inicialicen razonablemente.

Considérense dos frases consecutivas:

- *"Mary picked up the ball."* Al ingerir esta frase, deseamos que se activen los gates de los slots de "Mary" **y** de "ball" — posible gracias al término de localización `s_tᵀ w_j`, que usa la clave. El modelo modifica la entrada de "Mary" para indicar que ahora porta el balón, y la de "ball" para indicar que está siendo portado por Mary.
- *"Mary went to the garden."* Queremos modificar de nuevo "Mary" (ahora está en el jardín) y también "ball" (su nueva ubicación). Crucialmente, la palabra "ball" **no aparece** en esta segunda frase; sin embargo, el gate de "ball" puede activarse gracias al término de **contenido** `s_tᵀ h_j`, porque la información sobre Mary quedó guardada en la memoria de "ball" en el paso anterior.

Si los gates y las funciones de actualización tienen los pesos adecuados para ejecutar estos pasos, la memoria queda en un estado donde preguntas como "Where is the ball?" o "Where is Mary?" se responden directamente leyendo los slots relevantes, **sin necesidad de razonamiento complejo adicional**. Esa es la promesa del modelo: convertir el razonamiento secuencial costoso (en el momento de la pregunta) en una lectura simple sobre un estado ya mantenido.

## 5. Experimentos

### 5.1. Tarea sintética de modelo del mundo

Una tarea de juguete diseñada para medir directamente la capacidad de mantener un modelo del mundo en memoria. Dos agentes se colocan al azar en una grilla de 10×10; en cada paso un agente elegido al azar cambia de dirección o avanza. Tras `T` pasos, el modelo debe dar las **ubicaciones de ambos agentes**, revelando así su modelo interno del mundo. La tarea es difícil porque exige combinar hasta `T − 2` hechos de soporte y mantener actualizadas dos ubicaciones que cambian en momentos distintos. Resultados (Tabla 1a, error de 0 a 1):

| Modelo | T=10 | T=20 | T=40 |
|---|---|---|---|
| MemN2N | 0.09 | 0.633 | 0.896 |
| LSTM | 0 | 0.157 | 0.226 |
| **EntNet** | **0** | **0** | **0** |

El MemN2N (con número de hops igual a `T − 2`) es el peor y se degrada rápido con la longitud. El LSTM aguanta mejor pero también pierde precisión. El EntNet (con solo 5 slots y `d = 20`) resuelve la tarea en todos los casos. Más aún, el paper muestra **generalización más allá del horizonte de entrenamiento** (Tabla 1b): entrenado con secuencias de longitud 1 a 20, mantiene error casi nulo hasta T=50 (0.01), subiendo gradualmente (0.08 a T=80) — evidencia de que ha aprendido la *dinámica* del mundo, no solo a memorizar trayectorias de cierta longitud.

### 5.2. Tareas bAbI (el resultado estelar)

Las 20 tareas bAbI (Weston et al., 2015) son datasets sintéticos de question-answering que prueban un amplio espectro de habilidades de razonamiento (hechos de soporte múltiples, conteo, negación, deducción, inducción, razonamiento posicional, búsqueda de caminos, etc.) y son el benchmark estándar para redes con memoria. Se mide el error medio y el número de tareas falladas (>5 % de error). Los autores usan la versión 1.2 con 10k muestras, embeddings de `d = 100` y **20 slots de memoria**, limitando la capacidad a las 70 frases más recientes (130 para la tarea 3). Resultados clave de la Tabla 2:

| Modelo | Tareas falladas (>5%) | Error medio |
|---|---|---|
| NTM | 16 | 20.1 |
| D-NTM | 9 | 12.8 |
| MemN2N | 3 | 4.2 |
| DNC | 2 | 3.8 |
| DMN+ | 1 | 2.8 |
| **EntNet** | **0** | **0.5** |

El EntNet es el **primer modelo en resolver las 20 tareas** (0 falladas), superando a todos los demás tanto en número de tareas resueltas como en error medio. El salto más espectacular es en la **tarea 16 (basic induction)**, donde todos los competidores fallan estrepitosamente (45–55 % de error) mientras el EntNet logra 0.2 %. Conviene la honestidad: el EntNet no es uniformemente el mejor en cada tarea individual (en la 3, "3 supporting facts", obtiene 4.1 % frente al 1.1 % del DMN+), pero su consistencia general es lo que lo distingue.

**Análisis de interpretabilidad.** Para verificar que el modelo realmente mantiene un modelo del mundo y no solo lo necesario para una pregunta puntual, los autores entrenaron en la tarea 2 una variante con BoW y claves atadas a embeddings de entidades, y luego midieron la distancia coseno entre `φ(H h_j)` y cada fila de la matriz decodificadora `R`. La Tabla 3 muestra que cada slot de memoria (una entidad) tiene almacenada la **ubicación correcta** de su entidad al final de la historia — incluyendo la respuesta a la pregunta ("milk" → "garden") y también las ubicaciones de las demás entidades sobre las que no se preguntó. Esto confirma empíricamente la tesis: el modelo construye un estado del mundo completo, no un resumen oportunista.

### 5.3. Children's Book Test (CBT)

Un benchmark realista de modelado de lenguaje semántico (completar palabras) construido sobre libros infantiles de Project Gutenberg. El modelo lee 20 frases consecutivas y debe completar una palabra faltante en la frase 21, eligiendo entre 10 candidatos. Se evalúan las subtareas de **Named Entities** y **Common Nouns** (las más exigentes semánticamente). Los autores distinguen el setup **single-pass** (el modelo codifica el documento antes de ver la consulta) del **multi-pass** (puede atender sobre la historia usando la consulta). En CBT usan una variante simplificada del modelo (`U = V = 0`, `W = I`, `φ` identidad, **sin** normalización — que aquí perjudica porque oculta información útil basada en frecuencia), con claves atadas a los 10 candidatos.

| Modelo (single-pass) | Named Entities | Common Nouns |
|---|---|---|
| LSTMs (context+query) | 0.418 | 0.560 |
| Window LSTM | 0.436 | 0.582 |
| EntNet (general) | 0.484 | 0.540 |
| **EntNet (simple)** | **0.616** | **0.588** |

El EntNet simplificado **supera a todos los modelos single-pass** en ambas subtareas, y bate a la Memory Network sin auto-supervisión. Persiste una brecha frente a los modelos multi-pass de comprensión lectora más sofisticados (AoA Reader, NSE, que alcanzan 0.72–0.73 en Named Entities haciendo múltiples capas de atención con conocimiento de la pregunta). Pero el punto es alentador: leyendo en una sola pasada y construyendo una representación de propósito general de la historia, el EntNet responde un conjunto razonablemente diverso de consultas.

## 6. Limitaciones reconocidas

El paper es directo sobre sus límites:

- **Eficiencia de muestras.** El gran resultado (20/20 tareas) es solo en el régimen de **10k ejemplos**. Con 1k muestras el rendimiento **cae considerablemente** (Tabla 6: 15 tareas falladas, error medio 29.6 %, peor que el MemN2N). Los autores enfatizan que resolver bAbI en el régimen de 1k "sigue siendo un problema abierto" que requerirá mejorar la eficiencia de muestras de los modelos de razonamiento.
- **Variante "tied" no ayuda en general.** Atar las claves a embeddings de entidades, aunque hace el modelo más interpretable, no mejoró el rendimiento en la mayoría de las tareas y lo perjudicó en algunas (de ahí que no se aplicara en la Tabla 2).
- **Brecha frente a multi-pass en CBT.** En comprensión lectora real, los modelos que hacen múltiples pasadas de atención con conocimiento de la pregunta siguen por delante.
- **Solo entrada textual.** Aunque la arquitectura es general, en este trabajo solo se aplicó a tareas con texto. Los autores señalan como trabajo futuro combinar la capacidad de *tracking* del EntNet con modelos predictivos (forward models) que anticipen la evolución del mundo — el segundo objetivo de la IA que mencionan en la introducción y que el EntNet no aborda.

## 7. Impacto y lectura conceptual

El EntNet consolidó una idea que resonaría en la investigación posterior sobre memoria y razonamiento: en lugar de una memoria genérica direccionable como un archivo, una **memoria estructurada por entidades**, donde cada slot rastrea un objeto del mundo y se actualiza con compuertas independientes en paralelo. Esta concepción de *world modeling* —mantener vivo un estado del mundo que evoluciona con cada percepto— anticipa preocupaciones centrales de la IA contemporánea sobre el seguimiento de estado en agentes, el razonamiento sobre entidades y la memoria persistente. El hecho de ser el primero en resolver las 20 tareas bAbI lo convirtió en un hito de referencia del benchmark. Su elegancia arquitectónica (un banco de RNN con compuertas y parámetros compartidos, sin controlador central ni softmax acoplador) ofreció una alternativa más simple y paralelizable a la NTM/DNC, mostrando que buena parte del razonamiento puede distribuirse entre procesadores locales en vez de centralizarse.

## 8. Conexión con la Clase 30 (Modelos con memoria externa)

La Clase 30 recorre la evolución de la memoria externa en redes neuronales y **cierra precisamente con las Entity Networks (Henaff et al., 2017)**, usando el ejemplo de **Mary / cocina / manzana**: rastrear dónde está cada entidad a medida que la historia avanza. El paper aporta el mismo ejemplo en su versión "Mary / balón / jardín" (§4.4 de este análisis), que es exactamente la intuición que la clase quiere transmitir: la memoria deja de ser un repositorio pasivo y pasa a ser un **estado del mundo mantenido por entidades**.

Los **pasos del modelo** que la clase enumera mapean uno a uno con las ecuaciones del método:

1. **Codificar el input** → encoder de máscara multiplicativa + suma, `s_t = Σ_i f_i ⊙ e_i` (§4.1).
2. **Gate: decidir cuáles entidades actualizar** → `g_j ← σ(s_tᵀ h_j + s_tᵀ w_j)`, con sus términos de contenido y de localización (§4.2).
3. **Información a agregar (candidato)** → `h̃_j ← φ(U h_j + V w_j + W s_t)` (§4.2).
4. **Actualizar** → `h_j ← h_j + g_j ⊙ h̃_j` (§4.2).
5. **Normalizar** → `h_j ← h_j / ‖h_j‖`, el mecanismo de olvido (§4.2).
6. **Atención sobre las memorias** para responder → `p_j = Softmax(qᵀ h_j)`, `u = Σ_j p_j h_j`, `y = R φ(q + H u)` (§4.3).

Para situar el EntNet en el arco de la clase, conviene cruzarlo con los hitos previos del módulo de memoria externa. Es la culminación de una línea que parte de las **tareas bAbI** como benchmark de comprensión de historias ([/papers/babi-weston-2015](/papers/babi-weston-2015)) y de la **End-to-End Memory Network** ([/papers/e2e-memnn-sukhbaatar-2015](/papers/e2e-memnn-sukhbaatar-2015)), de la que el EntNet hereda el encoder posicional y el módulo de salida (que es, literalmente, una MemN2N de un solo hop con una no-linealidad extra) pero a la que supera con su idea distintiva: mantener estado dinámico de entidades en vez de solo almacenar y atender el texto crudo. El fundamento transversal de [/fundamentos/memory-augmented-networks](/fundamentos/memory-augmented-networks) provee el marco común (memorias externas, direccionamiento por contenido y por localización, lectura/escritura diferenciables) dentro del cual el EntNet se ubica como el modelo que estructura la memoria por entidades y actualiza varios slots en paralelo. La clase completa vive en [/clases/clase-30](/clases/clase-30).
