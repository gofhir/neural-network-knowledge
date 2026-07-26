# Mastering the Game of Go without Human Knowledge (AlphaGo Zero) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Mastering the game of Go without human knowledge*.
- **Autores:** David Silver*, Julian Schrittwieser*, Karen Simonyan*, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel y Demis Hassabis (DeepMind, Londres; *contribución equitativa).
- **Venue:** *Nature*, vol. 550, pp. 354–359, octubre de 2017.
- **Correspondencia:** David Silver (davidsilver@google.com).

AlphaGo Zero es el programa que aprendió a jugar Go a nivel sobrehumano **partiendo de cero (*tabula rasa*)**, sin usar una sola partida humana, sin conocimiento de dominio más allá de las reglas del juego, y **derrotó 100 a 0 a la versión de AlphaGo que había vencido a Lee Sedol** en Seúl (AlphaGo Lee). Su tesis, expresada ya en el resumen del paper, es directa: los sistemas entrenados por aprendizaje supervisado a partir de datos de expertos quedan sujetos a la disponibilidad de esos datos y, sobre todo, a un **techo de rendimiento** impuesto por la calidad humana; en contraste, un sistema entrenado desde su propia experiencia por aprendizaje reforzado puede, en principio, **exceder las capacidades humanas** y operar en dominios donde la pericia humana escasea. AlphaGo Zero demuestra empíricamente esa afirmación en el que se considera "el gran desafío de la IA".

Para la **Clase 33 (Aprendizaje por Imitación y Aprendizaje Reforzado Inverso)** este paper es el caso de estudio estrella. El profesor Rodrigo Toro Icarte lo cita (slide 39) como el ejemplo canónico de que el **RL puro por self-play supera al maestro**, mientras que el aprendizaje por imitación (clonación de comportamiento a partir de demostraciones) rara vez lo hace: el imitador tiende, en el mejor de los casos, a igualar al experto que copia. AlphaGo Zero es la refutación viva del "techo humano": no solo iguala, sino que aplasta a la versión de AlphaGo que sí había usado imitación de partidas humanas como punto de partida.

**Aclaración terminológica importante.** La slide del curso rotula este caso como "AlphaZero", pero cita este paper (Silver et al., Nature 2017), que corresponde a **AlphaGo Zero** —el sistema específico para Go. **AlphaZero** es un trabajo posterior (Silver et al., *Science* 2018) que generaliza el mismo algoritmo, sin cambios específicos de dominio, a ajedrez, shogi y Go. La distinción se desarrolla en la sección 7. En lo que sigue nos referimos siempre a AlphaGo Zero salvo indicación expresa.

## 2. Contexto: AlphaGo (2016) sí usó imitación; qué elimina Zero

Para entender la contribución de Zero hay que recordar cómo funcionaba **AlphaGo (2016)** —el paper analizado en la Clase 31. Aquella arquitectura (denominada en este trabajo *AlphaGo Fan*, y su sucesora *AlphaGo Lee*) partía de dos redes neuronales separadas: una **red de políticas** que producía probabilidades de jugada y una **red de valor** que evaluaba posiciones. Y, crucialmente, la red de políticas se **inicializaba por aprendizaje supervisado para predecir con precisión las jugadas de expertos humanos** —30 millones de posiciones del servidor KGS—; solo después se refinaba con aprendizaje reforzado por gradiente de política. Ese primer paso supervisado es, en el vocabulario de la Clase 33, **aprendizaje por imitación / clonación de comportamiento (behavioral cloning)**: el modelo aprende a reproducir la decisión del experto en cada estado.

AlphaGo Zero elimina, de forma total, todo lo humano y todo lo hecho a mano. El paper enumera con precisión las diferencias respecto de AlphaGo Fan y AlphaGo Lee:

1. **Se entrena únicamente por aprendizaje reforzado de self-play, partiendo de juego aleatorio**, sin supervisión ni datos humanos. Desaparece el arranque supervisado —desaparece la imitación.
2. Usa **solo las piedras blancas y negras del tablero** como features de entrada, en lugar de los 48 planos de features derivados de conocimiento experto de Go que usaba AlphaGo.
3. Usa **una sola red neuronal**, en vez de redes de políticas y de valor separadas.
4. Usa una **búsqueda en árbol más simple** que se apoya en esa única red para evaluar posiciones y muestrear jugadas, **sin ejecutar ningún rollout de Monte Carlo** (AlphaGo evaluaba las hojas combinando la red de valor con simulaciones rápidas hasta el final de la partida usando una *rollout policy* hecha a mano).

En términos de la clase: AlphaGo (2016) = **imitación + RL**; AlphaGo Zero (2017) = **RL puro**. Toda la maquinaria de imitación que en 2016 se consideraba indispensable para arrancar, en 2017 resulta no solo prescindible, sino un lastre para el rendimiento asintótico.

## 3. Contribución central

El aporte principal, en palabras de los autores, es **demostrar que se puede alcanzar rendimiento sobrehumano sin conocimiento humano de dominio**, incluso en el más difícil de los dominios. Técnicamente, el paper introduce un nuevo algoritmo de aprendizaje reforzado que **incorpora la búsqueda hacia adelante (lookahead) dentro del bucle de entrenamiento**, logrando mejora rápida y aprendizaje preciso y estable. Los ingredientes conceptuales:

- Una **red única** $f_\theta$ con torso residual (ResNet) y dos cabezales, que combina los roles de política y de valor.
- Un **MCTS guiado por esa red** que actúa como un potente **operador de mejora de política**: la búsqueda produce jugadas más fuertes que la política cruda de la red.
- Un **esquema de iteración de política por self-play** en el que la red se entrena para imitar la política mejorada por la búsqueda, y la red mejorada hace la búsqueda siguiente aún más fuerte.

La sutileza que conviene subrayar para la Clase 33: AlphaGo Zero **sí "imita" algo**, pero no imita a un humano —**se imita a sí mismo mejorado por la búsqueda**. La red aprende a predecir sus propias selecciones de jugada (las que salen del MCTS) y el ganador de sus propias partidas. El maestro no es externo ni tiene techo: es una versión ligeramente superior del propio alumno, generada por el cómputo de la búsqueda. Ese "bootstrap" es lo que permite escapar del límite humano.

## 4. Método

### 4.1. Arquitectura de red única (política + valor sobre torso ResNet)

La red $f_\theta$ toma como entrada la representación cruda del tablero y su historia, y produce simultáneamente un vector de probabilidades de jugada y un escalar de valor:

$$(p, v) = f_\theta(s)$$

donde $p_a = \Pr(a\mid s)$ es la probabilidad de seleccionar cada jugada (incluido el paso) y $v$ estima la probabilidad de que el jugador actual gane desde la posición $s$.

La **entrada** es una pila de imágenes $19\times 19\times 17$: 8 planos binarios con las piedras del jugador actual en los últimos 8 pasos, 8 planos con las del oponente, y 1 plano constante que indica el color a jugar. La historia es necesaria porque el Go no es plenamente observable desde las piedras actuales (hay prohibición de repeticiones), y el plano de color es necesario porque el komi no es observable.

El cuerpo es una **torre residual**: un bloque convolucional inicial seguido de **19 o 39 bloques residuales** (según la instancia, 20 o 40 bloques en total). Cada bloque residual aplica dos convoluciones de 256 filtros $3\times 3$ con batch normalization y rectificadores, más una **conexión de salto** que suma la entrada del bloque. De la torre salen dos **cabezales**:

- **Cabezal de política:** convolución $1\times 1$ de 2 filtros, batch norm, rectificador, y una capa lineal totalmente conectada que produce $19\cdot 19 + 1 = 362$ logits (todas las intersecciones más la jugada de paso).
- **Cabezal de valor:** convolución $1\times 1$ de 1 filtro, batch norm, rectificador, capa oculta de tamaño 256, rectificador, capa lineal a un escalar y una $\tanh$ que produce un valor en $[-1, 1]$.

Que política y valor **compartan el torso** no es un detalle de eficiencia solamente. El paper muestra (Fig. 4) que combinar ambas salidas en una red única, frente a redes separadas, **mejora el rendimiento en unos 600 Elo**: el objetivo dual regulariza la red hacia una representación común que sirve a los dos usos.

### 4.2. MCTS como operador de mejora de política

El MCTS de AlphaGo Zero es una variante simplificada del APV-MCTS de AlphaGo. Cada arista $(s,a)$ del árbol almacena un conteo de visitas $N(s,a)$, un valor de acción total $W(s,a)$, un valor de acción medio $Q(s,a)$ y una probabilidad a priori $P(s,a)$. Cada simulación tiene tres fases:

- **Selección.** Desde la raíz se desciende eligiendo en cada paso la acción que maximiza $Q(s,a) + U(s,a)$, con una variante del algoritmo **PUCT**:
  $$U(s,a) = c_{\text{puct}}\, P(s,a)\,\frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}$$
  Este control de búsqueda prefiere inicialmente acciones con **alta probabilidad a priori y pocas visitas**, y asintóticamente aquellas con **alto valor de acción** —el equilibrio entre exploración y explotación.
- **Expansión y evaluación.** Al llegar a una hoja $s_L$, esta se evalúa **una sola vez** por la red: $(P(s_L,\cdot), V(s_L)) = f_\theta(s_L)$. Los priors $P$ se almacenan en las aristas salientes de $s_L$. No hay rollouts: la red *es* la función de evaluación.
- **Backup.** En una pasada hacia atrás se incrementan los conteos de visitas y se actualiza cada valor de acción al promedio de las evaluaciones de la red en el subárbol correspondiente: $Q(s,a) = \tfrac{1}{N(s,a)}\sum_{s'\mid s,a\to s'} V(s')$.

Al terminar la búsqueda, el MCTS devuelve las **probabilidades de búsqueda** $\pi$, proporcionales al conteo de visitas exponenciado:

$$\pi(a\mid s_0) \propto N(s_0, a)^{1/\tau}$$

donde $\tau$ es una temperatura que controla la exploración. Los autores lo formalizan como $\pi = \alpha_\theta(s)$.

El punto conceptual clave —y la razón de que la clase lo cite— es este: **las probabilidades de búsqueda $\pi$ suelen seleccionar jugadas mucho más fuertes que las probabilidades crudas $p$ de la red $f_\theta$**. Por eso el MCTS "puede verse como un potente operador de mejora de política". El self-play con búsqueda —usar la política mejorada por MCTS para jugar cada movimiento, y el ganador $z$ como muestra del valor— actúa a su vez como un potente **operador de evaluación de política**.

### 4.3. Bucle de self-play / iteración de política

La idea central del algoritmo es aplicar repetidamente esos dos operadores en un procedimiento clásico de **iteración de política**:

1. La red se inicializa con pesos aleatorios $\theta_0$.
2. En cada iteración $i\ge 1$ se generan partidas de self-play. En cada paso $t$ se ejecuta $\pi_t = \alpha_{\theta_{i-1}}(s_t)$ con la red de la iteración anterior, y se juega muestreando de $\pi_t$.
3. La partida termina cuando ambos jugadores pasan, cuando el valor de búsqueda cae por debajo de un umbral de rendición, o al superar una longitud máxima; se puntúa para dar la recompensa terminal $r_T \in \{-1, +1\}$. Cada paso se guarda como $(s_t, \pi_t, z_t)$, con $z_t = \pm r_T$ desde la perspectiva del jugador en $t$.
4. En paralelo, se entrenan nuevos parámetros $\theta_i$ para que $(p, v) = f_{\theta_i}(s)$ se acerque a las probabilidades de búsqueda mejoradas y al ganador del self-play $(\pi, z)$. Estos nuevos parámetros se usan en la siguiente iteración, haciendo la búsqueda aún más fuerte.

El pipeline de producción corre tres componentes asíncronos en paralelo: un **optimizador** que actualiza continuamente $\theta_i$ desde los datos recientes de self-play; un **evaluador** que compara cada nuevo checkpoint contra el mejor jugador actual $\alpha_{\theta^*}$ (400 partidas, y solo se promueve el nuevo si gana con margen $>55\%$); y el **generador de self-play**, que usa el mejor jugador $\alpha_{\theta^*}$ para producir 25.000 partidas por iteración. Se añade **ruido de Dirichlet** $\eta\sim\text{Dir}(0.03)$ a los priors de la raíz (con $\varepsilon = 0.25$) para garantizar exploración, y se usa $\tau = 1$ en las primeras 30 jugadas de cada partida (para diversidad) y $\tau \to 0$ después.

### 4.4. Objetivo de entrenamiento

La red se entrena por descenso de gradiente sobre una función de pérdida que suma un término de error cuadrático medio (valor), un término de entropía cruzada (política) y regularización L2:

$$l = (z - v)^2 - \pi^\top \log p + c\lVert\theta\rVert^2$$

Cada término tiene un rol nítido, útil de leer con la lente de la Clase 33:

- $(z - v)^2$ empuja la **red de valor** a predecir quién ganó realmente el self-play. Es evaluación de política por regresión de Monte Carlo.
- $-\pi^\top \log p$ es la entropía cruzada entre la política de la red $p$ y las probabilidades de búsqueda $\pi$. Es **destilar la política mejorada por MCTS de vuelta en la red** —"imitar", pero al maestro-que-soy-yo-más-la-búsqueda, no a un humano.
- $c\lVert\theta\rVert^2$ es regularización L2, con $c = 10^{-4}$; los términos de valor y política se ponderan por igual (razonable porque las recompensas son unitarias, $r\in\{-1,+1\}$).

## 5. Resultados

**Entrenamiento y eficiencia.** La primera instancia (20 bloques) se entrenó **~3 días sin intervención humana**, generando **4,9 millones de partidas de self-play** con 1.600 simulaciones por MCTS (~0,4 s de "pensamiento" por jugada) y actualizando parámetros desde 700.000 mini-lotes de 2.048 posiciones. El aprendizaje **progresó de forma suave**, sin las oscilaciones ni el olvido catastrófico que la literatura previa sugería para el aprendizaje multiagente inestable.

**100 a 0 contra AlphaGo Lee.** Sorprendentemente, AlphaGo Zero **superó a AlphaGo Lee tras solo 36 horas** —AlphaGo Lee había requerido varios meses de entrenamiento. A las 72 horas se lo enfrentó a la versión exacta que venció a Lee Sedol, bajo las mismas condiciones de tiempo (2 horas) del match de Seúl. **AlphaGo Zero ganó 100 partidas a 0**, y lo hizo usando **una sola máquina con 4 TPU**, frente a las 48 TPU distribuidas de AlphaGo Lee.

**RL puro contra imitación (el experimento decisivo para la clase).** Los autores entrenaron una segunda red, de arquitectura idéntica, por **aprendizaje supervisado** de las jugadas de expertos del dataset KGS. Los resultados de la Fig. 3 son el corazón del argumento de la Clase 33:

- El aprendizaje supervisado (imitación) logró **mejor rendimiento inicial** y fue mejor prediciendo el resultado de partidas profesionales humanas —arranca rápido, porque copia directamente decisiones expertas.
- Pero el jugador auto-aprendido (RL) **rindió mucho mejor en general y derrotó al jugador entrenado con datos humanos dentro de las primeras 24 horas**.
- Notablemente, aunque el supervisado alcanzó **mayor precisión de predicción de jugadas humanas** (60,4 % en el test de KGS frente a apenas 49,0 % del modelo RL de 20 bloques), **jugó peor**. Predecir bien la jugada humana no equivale a jugar bien: el RL "puede estar aprendiendo una estrategia cualitativamente distinta del juego humano".

Esta es exactamente la moraleja del profesor: **la imitación arranca rápido pero se topa con el techo del maestro; el RL puro por self-play lo supera**.

**Curva Elo y rendimiento final.** Una segunda instancia (40 bloques) se entrenó **~40 días**, con 29 millones de partidas de self-play y 3,1 millones de mini-lotes. En un torneo con 5 s por jugada, las puntuaciones Elo fueron:

- Red cruda de AlphaGo Zero (sin búsqueda, eligiendo la jugada de máxima probabilidad): **3.055**.
- AlphaGo Fan (venció a Fan Hui): **3.144**.
- AlphaGo Lee (venció a Lee Sedol): **3.739**.
- AlphaGo Master (venció 60-0 a top profesionales online, enero 2017): **4.858**.
- **AlphaGo Zero: 5.185**.

En un match directo de 100 partidas con controles de 2 horas, **AlphaGo Zero venció a AlphaGo Master 89 a 11**. Que la red cruda ya alcance 3.055 Elo sin ningún lookahead ilustra cuánto conocimiento vive en los pesos; que el MCTS añada más de 2.000 Elo ilustra el poder del operador de mejora de política.

**Redescubrimiento y superación del conocimiento humano.** AlphaGo Zero **redescubrió por sí solo** elementos fundamentales del Go —fuseki (aperturas), tesuji (tácticas), vida y muerte, ko, yose (final), sente (iniciativa), forma, influencia y territorio— todos desde primeros principios. Redescubrió **josekis profesionales** (secuencias de esquina): p. ej., a las 47 horas prefería la invasión 3-3, común en el juego profesional humano; pero **más tarde descubrió y prefirió variantes nuevas, previamente desconocidas**. Curiosamente, el *shicho* ("escalera"), uno de los primeros conceptos que aprenden los humanos, lo comprendió tarde. La humanidad acumuló conocimiento de Go en millones de partidas a lo largo de milenios; AlphaGo Zero lo reconstruyó —y lo superó— en días.

**Ablación arquitectónica.** Comparando redes separadas vs. combinadas y convolucionales vs. residuales sobre un dataset fijo, el paso a **red residual** aportó **~600 Elo** y **combinar política y valor** en una red aportó otros **~600 Elo** —la arquitectura y el algoritmo contribuyen por separado.

## 6. AlphaGo Zero vs. AlphaGo (2016): la tabla del contraste

Vale consolidar el contraste que la Clase 33 necesita, dado que el análisis de AlphaGo (2016) ya existe en la Clase 31:

| Aspecto | AlphaGo Lee (2016) | AlphaGo Zero (2017) |
|---|---|---|
| Arranque | **Supervisado desde partidas humanas (imitación)** + RL | RL puro desde juego aleatorio |
| Datos humanos | 30 M posiciones KGS | **Ninguno** |
| Redes | Política y valor separadas (conv) | **Una red** política+valor (ResNet) |
| Features de entrada | 48 planos hechos a mano | Solo piedras crudas + historia |
| Evaluación de hojas | Red de valor + **rollouts** con política rápida | Solo la red (**sin rollouts**) |
| Hardware (evaluación) | 48 TPU, distribuido | **4 TPU, una máquina** |
| Resultado del enfrentamiento | — | **Zero gana 100-0** |

La lectura de la clase: cada pieza de imitación y de ingeniería humana que se retira **no degrada**, sino que **mejora** el sistema.

## 7. AlphaGo Zero vs. AlphaZero (aclaración)

Conviene ser preciso sobre la nomenclatura de la slide. **AlphaGo Zero** —el paper aquí analizado— es específico de Go: aunque no usa datos humanos, sí incorpora algún conocimiento de dominio del Go, que el propio paper enumera con honestidad como los elementos que "habría que reemplazar para aprender otro juego (de Markov alternante)": las reglas del juego usadas dentro del MCTS para simular y puntuar, la puntuación **Tromp-Taylor**, la estructura de imagen $19\times 19$ adaptada a la grilla del tablero, y la **invariancia a rotaciones y reflexiones** del Go, explotada para aumentar el dataset y muestrear simetrías durante la búsqueda.

**AlphaZero** (Silver et al., *Science* 2018) es el trabajo posterior que **generaliza el mismo algoritmo** —una red residual con cabezales política+valor, MCTS como operador de mejora, self-play e iteración de política— a **ajedrez, shogi y Go** con un único algoritmo y sin ajustes específicos de dominio (elimina, entre otras cosas, el aumento por simetría, que en ajedrez y shogi no aplica). En resumen: la slide dice "AlphaZero" pero cita AlphaGo Zero; para el argumento de la clase (RL puro por self-play que supera al maestro humano) **ambos sirven**, y el paper citado es el que introduce por primera vez la eliminación total de datos humanos. La generalización a múltiples juegos es mérito del AlphaZero de 2018, no de este paper.

## 8. Limitaciones

- **Dominio de información perfecta y suma cero.** El método es "más directamente aplicable a juegos de suma cero de información perfecta". El Go es determinista, plenamente accesible (con historia), de dos jugadores y con un ganador bien definido. Nada de esto es gratuito fuera de los juegos.
- **El self-play requiere un modelo/simulador perfecto del entorno.** El MCTS **necesita las reglas del juego** para simular las posiciones resultantes de una secuencia de jugadas y para puntuar los estados terminales. Sin un simulador exacto y barato del mundo, el operador de mejora de política —el corazón del método— no existe. Esta es la limitación más importante de cara a dominios reales.
- **Recompensa verificable y densa en la terminal.** El resultado $z\in\{-1,+1\}$ es objetivo, automático e inequívoco al final de cada partida. No hay ambigüedad sobre "quién ganó".
- **Costo computacional.** Aunque la inferencia final corre en 4 TPU, el entrenamiento consumió millones de partidas de self-play y días (o 40 días para la instancia grande) de cómputo intensivo.
- **Cierto conocimiento de dominio persiste** (reglas, simetrías, grilla, Tromp-Taylor), como se detalló en la sección 7.

## 9. Conexión con la Clase 33 (RL vs. imitación)

AlphaGo Zero es el contraejemplo definitivo al que la Clase 33 recurre para delimitar cuándo conviene imitar y cuándo conviene reforzar. La tabla mental de ventajas y desventajas del profesor se ilumina con este paper:

- **El aprendizaje por imitación (behavioral cloning) tiene un techo: el maestro.** El experimento supervisado del propio paper lo demuestra: la red que copia a expertos humanos aprende rápido, predice mejor las jugadas humanas y, sin embargo, se estanca por debajo del RL. Imitar es aprender a *ser como* el demostrador; nunca a superarlo.
- **El RL puro puede exceder al humano, porque su señal no es "haz lo que hace el experto" sino "gana".** Al optimizar directamente la recompensa (el resultado de la partida) en lugar de la verosimilitud de la acción experta, el agente es libre de descubrir estrategias que ningún humano jugó —los nuevos josekis son la evidencia visible de ello.
- **El precio del RL puro es la exploración y la señal de recompensa.** La imitación evita el costoso problema de exploración (el experto ya muestra qué hacer); el RL debe descubrirlo desde el juego aleatorio, lo que solo es viable aquí porque existe un **simulador perfecto barato** (las reglas del Go) y una **recompensa verificable**. Cuando esos dos supuestos faltan, la balanza vuelve a inclinarse hacia la imitación.
- **AlphaGo Zero "imita", pero a un maestro sin techo.** La entropía cruzada $-\pi^\top\log p$ es literalmente destilación de una política por imitación; la diferencia con la clonación de comportamiento humano es que el demostrador ($\pi$ del MCTS) es siempre una versión mejorada del propio agente. Ese bucle es lo que la Clase 31 llamó iteración de política y lo que la Clase 33 contrasta con imitar a un humano fijo.

Para el contexto completo de esta clase, ver [/clases/clase-33](/clases/clase-33). Para el análisis de la versión de AlphaGo (2016) que **sí** usó imitación de partidas humanas, ver [AlphaGo (Silver et al., 2016)](/papers/silver-alphago-2016) en la Clase 31, y los fundamentos en [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado).

---

**Nota para el lector experto en FHIR/salud.** El poder de AlphaGo Zero descansa en dos supuestos que en salud casi nunca se cumplen: un **modelo perfecto y barato del mundo** (las reglas del Go permiten simular millones de partidas sin costo ni riesgo) y una **recompensa verificable** (ganar o perder es inequívoco al final de cada partida). En un problema como el *matching* de pacientes (record linkage / MDM), no existe un simulador que genere pares "verdadero/falso" a voluntad sin un juez externo, y no hay una recompensa objetiva que se dispare sola: la única señal de verdad es el criterio de un experto humano —el data steward que resuelve los casos ambiguos— o un *ground truth* costoso y escaso. No hay self-play posible cuando no se puede jugar contra uno mismo ni verificar el resultado. Por eso, en estos dominios sin modelo del mundo y sin recompensa verificable, **la imitación de expertos sigue siendo el camino práctico**: se aprende de las decisiones de los stewards (clonación de comportamiento) o, cuando se busca inferir el criterio implícito detrás de esas decisiones, se recurre al **aprendizaje reforzado inverso** —el otro pilar de esta clase— para recuperar la función de recompensa que un humano optimiza al declarar dos registros como el mismo paciente. AlphaGo Zero marca el techo de lo que el RL puro puede lograr; sirve, precisamente, para reconocer cuándo *no* tenemos las condiciones para invocarlo.
