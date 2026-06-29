# Progressive Neural Networks — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Progressive Neural Networks*.
- **Autores:** Andrei A. Rusu\*, Neil C. Rabinowitz\*, Guillaume Desjardins\*, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, Raia Hadsell (\* contribución equivalente). Todos de **Google DeepMind**, Londres.
- **Venue / preprint:** arXiv:1606.04671 (junio 2016; la copia analizada es la v4 de octubre 2022). Trabajo ampliamente citado como referencia canónica de los métodos *expansibles* de aprendizaje continuo.
- **Año:** 2016.

**Nota importante para el curso.** Este paper **no aparece citado en las slides de la Clase 32**. Lo incluimos porque es el **método arquitectónico seminal** contra el olvido catastrófico: cuando la clase describe la familia de "métodos basados en arquitectura" —los que evitan el olvido añadiendo capacidad nueva y cuya memoria/parámetros crecen con cada tarea— Progressive Neural Networks (PNN) es *el* ejemplo fundacional de esa categoría. Conviene leerlo como el contrapunto extremo de la familia de regularización (EWC, también de DeepMind y de varios de estos mismos autores): donde EWC tolera algo de olvido a cambio de no crecer, PNN ofrece **cero olvido por construcción** a cambio de **crecer en parámetros**.

La tesis del paper es directa. Aprender una secuencia de tareas exige dos cosas simultáneas que normalmente están en tensión: **transferir** conocimiento de tareas previas y **no olvidar** lo aprendido (*catastrophic forgetting*). El *finetuning*, método estándar de transferencia, falla en el segundo punto: es "un proceso destructivo que descarta la función previamente aprendida". PNN resuelve la tensión moviendo el problema al **diseño de la arquitectura**, no al de la función de pérdida: por cada tarea nueva instancia una **columna** de red nueva, **congela** las columnas anteriores (de ahí la inmunidad total al olvido) y conecta las columnas previas a la nueva mediante **conexiones laterales** que habilitan la transferencia de *features*. Nunca se sobrescribe nada.

Las contribuciones declaradas son tres: (1) la arquitectura progresiva en sí, como combinación novedosa de ingredientes conocidos para resolver secuencias complejas de tareas; (2) una evaluación extensa en *deep reinforcement learning* (Pong, Atari, laberintos 3D), comparándola con *finetuning*; y (3) un análisis novedoso basado en Información de Fisher y perturbaciones para medir **dónde y cómo** ocurre la transferencia entre tareas.

## 2. Contexto histórico: olvido catastrófico, transferencia y el límite del finetuning

Hacia 2016 la transferencia entre redes neuronales se hacía casi siempre por *pretrain-and-finetune*: se preentrena un modelo en un dominio fuente (con datos abundantes), se adapta la capa de salida al dominio objetivo y se ajusta toda la red por *backpropagation*. El paper reconoce que este paradigma fue pionero en Hinton & Salakhutdinov (2006) y se generalizó con éxito, pero enumera sus límites cuando lo que se quiere es aprender una **secuencia** de tareas:

- **No sabe con qué modelo inicializar.** Si quiero aprovechar conocimiento de varias experiencias previas, ¿qué modelo uso para inicializar el siguiente? El método "parece requerir no solo aprendizaje que soporte transferencia sin olvido catastrófico, sino también conocimiento previo de la similitud entre tareas".
- **Es destructivo.** El *finetuning* puede recuperar rendimiento experto en el dominio objetivo, pero "descarta la función previamente aprendida". Se podría copiar cada modelo antes de adaptarlo para recordarlas todas, pero entonces vuelve el problema de la inicialización.
- **La destilación no siempre aplica.** La destilación (Hinton, Vinyals & Dean, 2015) ofrece una vía al *multitask*, pero "requiere un reservorio de datos de entrenamiento persistente para todas las tareas", supuesto que no siempre se cumple en aprendizaje continuo.

El paper sitúa así el aprendizaje continuo (*continual / lifelong learning*) como meta de larga data del *machine learning*: agentes que aprenden y **recuerdan** una serie de tareas en secuencia, **y además** transfieren conocimiento para converger más rápido. PNN integra ambos deseos directamente en la arquitectura. Genealógicamente el paper se reconoce emparentado con arquitecturas **incrementales y constructivas** previas: la *cascade-correlation* de Fahlman & Lebiere (1990), diseñada justamente para eliminar el olvido añadiendo extractores de *features* de forma incremental; los *multi-column deep networks* de Ciresan et al. (2012); y la arquitectura *block-modular* de Terekhov et al. (2015). La diferencia de PNN es el uso de **conexiones laterales** para acceder a *features* previamente aprendidos y lograr *composicionalidad profunda*.

## 3. Contribución central: cero olvido por construcción, transferencia por conexiones laterales

La idea central se puede enunciar en una frase: **separar físicamente la capacidad de cada tarea (columnas) y conectar lo viejo con lo nuevo en un solo sentido (lateral, congelado).** De ahí se derivan dos propiedades que la Clase 32 querría destacar:

1. **Inmunidad total al olvido (por diseño, no por regularización).** Como los parámetros de las columnas anteriores `{Θ(j); j < k}` se mantienen **congelados** (son constantes para el optimizador) durante el entrenamiento de la columna nueva `Θ(k)`, y como las conexiones laterales solo van de columnas previas `j` hacia la nueva `k` (nunca al revés), **las columnas previas no se ven afectadas en el *forward pass***. No hay interferencia entre tareas y, por tanto, **no hay olvido catastrófico**. Esto es categóricamente distinto de EWC: ahí el olvido se *penaliza* (puede ocurrir un poco); aquí es *imposible* porque los pesos viejos literalmente no cambian.

2. **Transferencia sin destrucción.** A diferencia del *finetuning*, que solo incorpora conocimiento previo en la inicialización y luego lo puede "desaprender", PNN **retiene un pool de modelos preentrenados durante todo el entrenamiento** y aprende conexiones laterales para extraer de ellos los *features* útiles. El conocimiento previo deja de ser transitorio y puede integrarse **en cada capa** de la jerarquía de *features* ("composicionalidad más rica"). Como las columnas nuevas se inicializan al azar pero tienen acceso lateral a lo viejo, son libres de **reutilizar, modificar o ignorar** los *features* previos según convenga a la tarea. El paper subraya que **no hace ningún supuesto sobre la relación entre tareas** —pueden ser ortogonales o incluso adversariales—, a diferencia del *finetuning*, que asume implícitamente "solapamiento" entre dominios.

## 4. El método: columnas, conexiones laterales, adapters y crecimiento

### 4.1. Columnas

Una red progresiva **empieza con una sola columna**: una red profunda de `L` capas, con activaciones ocultas `h_i^(1)` y parámetros `Θ(1)` entrenados hasta converger en la tarea 1. Al pasar a la tarea 2, se **congela** `Θ(1)` y se instancia una **columna nueva** con parámetros `Θ(2)` (inicialización aleatoria), donde cada capa `h_i^(2)` recibe entrada **tanto de su propia columna** (`h_{i-1}^(2)`) **como de la columna anterior** (`h_{i-1}^(1)`) vía conexiones laterales. Generalizado a `K` tareas, la activación de la capa `i` de la columna `k` es:

```
h_i^(k) = f( W_i^(k) · h_{i-1}^(k)  +  Σ_{j<k} U_i^(k:j) · h_{i-1}^(j) )
```

donde `W_i^(k)` es la matriz de pesos de la capa `i` de la columna `k`; `U_i^(k:j)` son las **conexiones laterales** desde la capa `i-1` de la columna previa `j` hacia la capa `i` de la columna `k`; y `f` es una no linealidad por elemento (ReLU, `f(x)=max(0,x)`). La Figura 1 del paper muestra una red de `K=3` columnas: las dos columnas de la izquierda fueron entrenadas en las tareas 1 y 2; la tercera, añadida para la tarea final, tiene acceso a **todos** los *features* aprendidos antes.

### 4.2. Conexiones laterales con adapters

En la práctica las conexiones laterales lineales se reemplazan por **adapters** no lineales, que cumplen dos funciones: (i) mejorar el condicionamiento inicial y (ii) hacer **reducción de dimensionalidad**. El problema que resuelven: a medida que crece el índice `k`, el vector de *features anteriores* `h_{i-1}^(<k)` (la concatenación de las activaciones de todas las columnas previas) crece, y con él la cantidad de parámetros laterales. El adapter es un MLP de una capa oculta que **proyecta** esas activaciones anteriores a un subespacio de dimensión `n_i` antes de inyectarlas. Antes de pasar las activaciones laterales por el MLP, se multiplican por un **escalar aprendido** (inicializado pequeño y aleatorio) cuyo rol es ajustar las distintas escalas de las distintas entradas. Con esta proyección, la cantidad de parámetros laterales queda **en el mismo orden que `Θ(1)`** en lugar de crecer sin control. Para capas convolucionales la reducción de dimensionalidad se hace con **convoluciones 1×1**. La ecuación con adapter (omitiendo sesgos) es:

```
h_i^(k) = σ( W_i^(k) · h_{i-1}^(k)  +  U_i^(k:j) · σ( V_i^(k:j) · α_{i-1}^(<k) · h_{i-1}^(<k) ) )
```

donde `V_i^(k:j)` es la matriz de proyección y `α` el escalar de escala aprendido.

### 4.3. El crecimiento

El crecimiento es la consecuencia inevitable del diseño: **cada tarea añade una columna entera** (capas nuevas con pesos nuevos). En la versión básica del paper, el número de unidades ocultas y *feature maps* crece **linealmente** con el número de columnas, y el número de **parámetros crece cuadráticamente** (porque cada columna nueva se conecta lateralmente a todas las previas). Los adapters contienen el factor de crecimiento de los laterales, pero no eliminan el crecimiento de fondo.

### 4.4. Aplicación a RL

Aunque PNN es de aplicación general, el paper se centra en *deep reinforcement learning*. Cada columna se entrena para resolver un MDP (proceso de decisión de Markov) distinto: la columna `k` define una política `π^(k)(a|s)` que toma el estado `s` del ambiente y produce probabilidades sobre acciones (`π^(k)(a|s) := h_L^(k)(s)`). El entrenamiento usa el framework **A3C** (Asynchronous Advantage Actor-Critic, Mnih et al., 2016), que aprende política y función de valor en paralelo sobre CPU y converge más rápido que DQN en este régimen de muchos experimentos secuenciales.

## 5. Experimentos: Pong Soup, Atari y Labyrinth

El paper evalúa PNN en tres dominios de RL y mide la transferencia con un **transfer score**: rendimiento relativo (área bajo la curva de aprendizaje) frente a una columna única entrenada solo en la tarea objetivo (*baseline 1*). Los *baselines* clave son: *baseline 2* (columna única preentrenada en la fuente, con solo la capa de salida reajustada), *baseline 3* (lo mismo pero con *finetuning* completo — el paradigma estándar de transferencia), y *baseline 4* (arquitectura progresiva de 2 columnas pero con la columna previa inicializada al azar y congelada, para aislar el efecto de la transferencia real).

- **Pong Soup.** Variantes sintéticas de Pong (Noisy, Black, White, Zoom, V-flip, H-flip, VH-flip) donde se sabe que hay aspectos transferibles. *Baseline 2* (solo capa de salida) **falla** y da transferencia negativa en la mayoría de casos. PNN supera a *baseline 3* (finetuning completo) tanto en media como en mediana, y la diferencia es más marcada en la media —más sensible a *outliers*—, lo que sugiere que PNN **explota mejor la transferencia cuando es posible**. PNN también supera a *baseline 4*, confirmando que efectivamente usa los *features* de las columnas previas y no solo la capacidad extra.

- **Atari.** Transferencia entre juegos Atari elegidos al azar (fuentes: Pong, River Raid, Seaquest; objetivos: Alien, Asterix, Boxing, Centipede, Gopher, Hero, James Bond, Krull, Robotank, Road Runner, Star Gunner, Wizard of Wor), evaluando PNN con 2, 3 y 4 columnas. Es un escenario duro: los Atari difieren mucho en visuales, controles y estrategia (Pong es vertical, Breakout horizontal; otros pares no comparten nada). PNN da **transferencia positiva en 8 de 12** tareas objetivo (solo 2 casos negativos), frente a *baseline 3* que solo logra 5 de 12. En el caso Seaquest→Gopher (juegos disímiles), el *finetuning* da transferencia **negativa** mientras PNN no, "quizá porque es más capaz de ignorar los *features* irrelevantes". Más columnas siguen mejorando la transferencia.

- **Labyrinth.** Laberintos 3D con observabilidad parcial donde el agente forrajea (recompensa positiva por manzanas/fresas, negativa por hongos/limones). De nuevo PNN da más transferencia positiva que cualquier *baseline*; *baseline 2* da transferencia negativa incluso en niveles fáciles porque no puede aprender *features* visuales de bajo nivel nuevos, que aquí importan porque los ítems de recompensa cambian de tarea a tarea.

**Análisis de transferencia (AFS/APS).** Como PNN no destruye los *features* viejos, el paper puede medir **dónde** ocurre la transferencia. Con la **Average Fisher Sensitivity** (basada en la matriz de Información de Fisher de la política) y su contraparte por perturbación (APS), encuentran patrones interpretables: de Pong a H-Flip se reutiliza la visión de bajo y medio nivel pero la capa totalmente conectada debe reaprenderse; la transferencia positiva en Atari ocurre en un "punto dulce" entre depender mucho de los *features* fuente y aprender muchos *features* nuevos; la transferencia negativa coincide con depender por completo de las capas convolucionales previas sin aprender visión nueva.

## 6. Limitaciones reconocidas

El propio paper es explícito: PNN es "un peldaño hacia un agente de aprendizaje continuo completo", no la solución final. Sus límites:

- **El número de parámetros crece con cada tarea.** Es la limitación central y la que conecta directamente con la Clase 32: la red crece (linealmente en unidades, cuadráticamente en parámetros) y **no escala a muchas tareas**. El Apéndice muestra además que **solo una fracción de la capacidad nueva se usa realmente**, y que esa subutilización **aumenta** con más columnas (el espectro de AFS se vuelve más disperso). Eso sugiere que el crecimiento se podría mitigar añadiendo menos capacidad, con *pruning* (LeCun et al., 1990 — *Optimal Brain Damage*) o con compresión/destilación en línea durante el aprendizaje — pero todo eso queda como trabajo futuro.
- **Requiere la etiqueta de tarea en inferencia.** PNN retiene la capacidad de resolver las `K` tareas en *test time*, pero **elegir qué columna usar para inferir requiere conocer la etiqueta de la tarea**. Es un supuesto fuerte: PNN resuelve el *task-incremental* setting, no el más difícil *class-incremental* sin identificador de tarea.

## 7. Impacto y lugar en el dominio

PNN se consolidó como el **método arquitectónico fundacional** contra el olvido catastrófico: la referencia que define la familia de enfoques *expansibles* o de *parameter isolation* en aprendizaje continuo. Su aporte conceptual perdura: demostró que es posible una **inmunidad total al olvido** sin renunciar a la transferencia, *a costa* de un crecimiento de capacidad. Esa tensión —cero olvido vs. modelo que crece— quedó como uno de los ejes que organizan todo el campo, y motivó la línea de trabajos posteriores que buscan los beneficios de PNN **sin el crecimiento**: máscaras sobre una red fija (Piggyback, PackNet), expansión *selectiva* (Dynamically Expandable Networks), o métodos de regularización (EWC, SI) que aceptan algo de olvido a cambio de un footprint constante. PNN es, en ese mapa, el extremo "sin compromiso con el olvido, máximo compromiso con la memoria".

## 8. Conexión con la Clase 32 (Olvido catastrófico)

La Clase 32 organiza las defensas contra el olvido catastrófico en familias; PNN es el **ejemplo canónico de los métodos basados en arquitectura**. El punto que la clase destaca de esta familia —que **la memoria/parámetros crecen con cada tarea**— es *literalmente* la limitación que Rusu et al. reconocen en su sección de limitaciones. Tres lecturas para la clase:

- **El mecanismo del olvido y su antídoto.** El olvido catastrófico ocurre porque el *backprop* sobre la tarea nueva **sobrescribe** los pesos importantes para las tareas viejas. PNN ataca la raíz: si los pesos viejos están **congelados**, no hay nada que sobrescribir. Es la forma más limpia de "entender" por qué olvidamos —viéndo el método que lo vuelve imposible.

- **El trade-off que define el campo.** PNN encarna el extremo del *trade-off* estabilidad-plasticidad: estabilidad perfecta (cero olvido) pagada con plasticidad/escalabilidad limitada (crece sin parar). Compararlo con la familia de regularización (EWC) en la misma clase hace tangible el espectro de soluciones.

- **Hacia métodos arquitectónicos que no crecen.** La limitación de crecimiento de PNN es exactamente el problema que resuelven los métodos de **enmascaramiento sobre una red fija**: ver [Piggyback (Mallya et al., 2018)](/papers/piggyback-mallya-2018), que aprende **máscaras binarias** sobre una red base congelada —obteniendo el aislamiento de parámetros de PNN (y por tanto su inmunidad al olvido) pero con un costo de almacenamiento por tarea minúsculo y constante en lugar de una columna entera.

Enlaces relacionados del curso: fundamento de [aprendizaje continuo](/fundamentos/aprendizaje-continuo), la [Clase 32](/clases/clase-32) y el paper [Piggyback (Mallya et al., 2018)](/papers/piggyback-mallya-2018).
