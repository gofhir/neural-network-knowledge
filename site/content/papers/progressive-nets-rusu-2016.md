---
title: "Progressive Neural Networks (2016)"
weight: 366
math: true
---

{{< paper-card
    title="Progressive Neural Networks"
    authors="Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, et al. (DeepMind)"
    year="2016"
    venue="arXiv / DeepMind"
    pdf="/papers/progressive-nets-rusu-2016.pdf"
    arxiv="1606.04671" >}}
El **método arquitectónico seminal** contra el olvido catastrófico. Por cada tarea nueva, una red progresiva instancia una **columna** completa de red, **congela** las columnas anteriores (cero olvido por diseño) y las conecta a la nueva mediante **conexiones laterales** que transfieren *features*. Nunca se sobrescribe nada, así que el olvido es imposible por construcción —pero el precio es que **los parámetros crecen con cada tarea**. Evaluado en *deep reinforcement learning* (Pong, Atari, laberintos 3D). No aparece citado en las slides de la [Clase 32](/clases/clase-32), pero es **el** ejemplo fundacional de la familia de métodos basados en arquitectura.
{{< /paper-card >}}

---

## Por qué lo incluimos (no está en las slides)

Este paper **no aparece citado en la Clase 32**, pero lo incluimos porque es el **método arquitectónico seminal** contra el olvido catastrófico. Cuando la clase describe la familia de "métodos basados en arquitectura" —los que evitan el olvido añadiendo capacidad nueva, y cuya memoria/parámetros crecen con cada tarea— Progressive Neural Networks (PNN) es **el** ejemplo fundacional de esa categoría.

Conviene leerlo como el contrapunto extremo de la familia de regularización (EWC, también de DeepMind y de varios de estos mismos autores): donde [EWC](/papers/ewc-kirkpatrick-2017) tolera algo de olvido a cambio de no crecer, PNN ofrece **cero olvido por construcción** a cambio de **crecer en parámetros**.

## La tensión que resuelve

Aprender una secuencia de tareas exige dos cosas que normalmente están en conflicto: **transferir** conocimiento de tareas previas y **no olvidar** lo aprendido (*catastrophic forgetting*). El *finetuning*, método estándar de transferencia hacia 2016, falla en lo segundo: es "un proceso destructivo que descarta la función previamente aprendida". Copiar cada modelo antes de adaptarlo recuperaría la memoria, pero reabre el problema de con qué modelo inicializar el siguiente; y la destilación, otra vía al *multitask*, exige conservar un reservorio de datos de todas las tareas, supuesto que el aprendizaje continuo no siempre cumple.

PNN mueve el problema del **diseño de la pérdida** al **diseño de la arquitectura**. La idea cabe en una frase: **separar físicamente la capacidad de cada tarea (columnas) y conectar lo viejo con lo nuevo en un solo sentido (lateral, congelado).** De ahí salen dos propiedades:

1. **Inmunidad total al olvido (por diseño, no por regularización).** Los parámetros de las columnas anteriores se mantienen **congelados** durante el entrenamiento de la nueva, y las conexiones laterales solo van de columnas previas hacia la nueva (nunca al revés). Las columnas viejas no se ven afectadas en el *forward pass*: no hay interferencia y, por tanto, no hay olvido. Es categóricamente distinto de EWC, donde el olvido se *penaliza* (puede ocurrir algo); aquí es *imposible* porque los pesos viejos literalmente no cambian.

2. **Transferencia sin destrucción.** PNN retiene un *pool* de modelos preentrenados durante todo el entrenamiento y aprende a extraer de ellos los *features* útiles. El conocimiento previo deja de ser transitorio y se integra **en cada capa** de la jerarquía. La columna nueva se inicializa al azar pero, con acceso lateral a lo viejo, es libre de **reutilizar, modificar o ignorar** los *features* previos. El paper no hace ningún supuesto sobre la relación entre tareas: pueden ser ortogonales o incluso adversariales.

## El método: columnas, conexiones laterales y adapters

### Columnas

Una red progresiva **empieza con una sola columna**: una red profunda de $L$ capas con parámetros $\Theta^{(1)}$, entrenada hasta converger en la tarea 1. Al pasar a la tarea 2 se **congela** $\Theta^{(1)}$ y se instancia una **columna nueva** $\Theta^{(2)}$ (inicialización aleatoria), donde cada capa recibe entrada tanto de su propia columna como de la anterior vía conexiones laterales. Generalizado a $K$ tareas, la activación de la capa $i$ de la columna $k$ es:

$$h_i^{(k)} = f\!\left( W_i^{(k)} h_{i-1}^{(k)} + \sum_{j<k} U_i^{(k:j)} h_{i-1}^{(j)} \right)$$

donde $W_i^{(k)}$ es la matriz de pesos propia, $U_i^{(k:j)}$ son las **conexiones laterales** desde la columna previa $j$, y $f$ es una no linealidad (ReLU). Una red de $K=3$ columnas tiene su tercera columna —la de la tarea final— con acceso a **todos** los *features* aprendidos antes.

### Conexiones laterales con adapters

En la práctica los laterales lineales se reemplazan por **adapters** no lineales que (i) mejoran el condicionamiento inicial y (ii) hacen **reducción de dimensionalidad**. El problema: a medida que crece $k$, el vector de *features* previos (concatenación de las activaciones de todas las columnas anteriores) crece, y con él los parámetros laterales. El adapter es un MLP de una capa oculta que **proyecta** esas activaciones a un subespacio menor antes de inyectarlas; las activaciones laterales se multiplican primero por un **escalar aprendido** que ajusta las distintas escalas. Así los parámetros laterales quedan en el mismo orden que $\Theta^{(1)}$ en vez de crecer sin control. Para capas convolucionales la reducción se hace con **convoluciones 1×1**.

### El crecimiento (la limitación central)

El crecimiento es la consecuencia inevitable del diseño: **cada tarea añade una columna entera**. En la versión básica, las unidades ocultas crecen **linealmente** con el número de columnas y los **parámetros crecen cuadráticamente** (porque cada columna nueva se conecta lateralmente a todas las previas). Los adapters contienen el factor de crecimiento de los laterales, pero no eliminan el crecimiento de fondo.

### Aplicación a RL

Aunque PNN es de aplicación general, el paper se centra en *deep reinforcement learning*. Cada columna resuelve un MDP distinto: la columna $k$ define una política $\pi^{(k)}(a\mid s)$ que mapea el estado a probabilidades sobre acciones. El entrenamiento usa **A3C** (Asynchronous Advantage Actor-Critic), que aprende política y valor en paralelo sobre CPU y converge rápido en el régimen de muchos experimentos secuenciales.

## Experimentos: Pong Soup, Atari y Labyrinth

La transferencia se mide con un **transfer score**: rendimiento (área bajo la curva de aprendizaje) relativo a una columna única entrenada solo en la tarea objetivo. Los *baselines* clave son: solo capa de salida reajustada (*baseline 2*), *finetuning* completo (*baseline 3*, el paradigma estándar) y una columna previa inicializada al azar y congelada (*baseline 4*, para aislar la transferencia real de la mera capacidad extra).

- **Pong Soup.** Variantes sintéticas de Pong (Noisy, Black, White, Zoom, V-flip, H-flip…) con transferencia conocida. Reajustar solo la capa de salida **falla** (transferencia negativa). PNN supera al *finetuning* completo en media y mediana, y también al *baseline 4*, confirmando que **sí usa los *features* de las columnas previas** y no solo la capacidad añadida.

- **Atari.** Transferencia entre juegos elegidos al azar, un escenario duro (los Atari difieren mucho en visuales, controles y estrategia). PNN da **transferencia positiva en 8 de 12** tareas objetivo (solo 2 negativas), frente a 5 de 12 del *finetuning*. En Seaquest→Gopher (juegos disímiles) el *finetuning* da transferencia **negativa** y PNN no, "quizá porque es más capaz de ignorar los *features* irrelevantes". Más columnas siguen mejorando la transferencia.

- **Labyrinth.** Laberintos 3D con observabilidad parcial donde el agente forrajea. De nuevo PNN aporta más transferencia positiva que cualquier *baseline*; reajustar solo la salida da transferencia negativa incluso en niveles fáciles porque no puede aprender *features* visuales nuevos.

Como PNN **no destruye** los *features* viejos, el paper puede medir **dónde** ocurre la transferencia, con la *Average Fisher Sensitivity* (basada en la Información de Fisher de la política) y una contraparte por perturbación. Encuentra patrones interpretables: de Pong a H-Flip se reutiliza la visión de bajo y medio nivel pero la capa totalmente conectada se reaprende; la transferencia positiva en Atari ocurre en un "punto dulce" entre depender de los *features* fuente y aprender muchos nuevos.

## Limitaciones reconocidas

- **Los parámetros crecen con cada tarea.** Es la limitación central y la que conecta con la Clase 32: la red crece (linealmente en unidades, cuadráticamente en parámetros) y **no escala a muchas tareas**. El apéndice muestra además que **solo una fracción de la capacidad nueva se usa realmente**, y que esa subutilización **aumenta** con más columnas. El crecimiento podría mitigarse con *pruning* o compresión/destilación en línea, pero queda como trabajo futuro.
- **Requiere la etiqueta de tarea en inferencia.** PNN conserva la capacidad de resolver las $K$ tareas, pero **elegir qué columna usar exige conocer la etiqueta de la tarea**. Resuelve el escenario *task-incremental*, no el más difícil *class-incremental* sin identificador.

## Lugar en el dominio y conexión con la Clase 32

PNN se consolidó como el **método arquitectónico fundacional** contra el olvido catastrófico: la referencia que define la familia de enfoques *expansibles* o de *parameter isolation*. Demostró que es posible una **inmunidad total al olvido** sin renunciar a la transferencia, *a costa* de un crecimiento de capacidad. Esa tensión —cero olvido vs. modelo que crece— quedó como uno de los ejes que organizan el campo del [aprendizaje continuo](/fundamentos/aprendizaje-continuo).

La Clase 32 organiza las defensas contra el olvido en familias, y PNN es el **ejemplo canónico de los métodos basados en arquitectura**. Tres lecturas:

- **El mecanismo del olvido y su antídoto.** El olvido catastrófico ocurre porque el *backprop* sobre la tarea nueva **sobrescribe** los pesos importantes para las viejas. PNN ataca la raíz: si los pesos viejos están **congelados**, no hay nada que sobrescribir.
- **El trade-off que define el campo.** PNN encarna el extremo del *trade-off* estabilidad-plasticidad: estabilidad perfecta (cero olvido) pagada con escalabilidad limitada (crece sin parar). Compararlo con [EWC](/papers/ewc-kirkpatrick-2017) en la misma clase hace tangible el espectro de soluciones.
- **Hacia métodos arquitectónicos que no crecen.** La limitación de crecimiento es exactamente el problema que resuelven los métodos de **enmascaramiento sobre una red fija**: ver [Piggyback (Mallya et al., 2018)](/papers/piggyback-mallya-2018), que aprende **máscaras binarias** sobre una red base congelada —obteniendo el aislamiento de parámetros de PNN (y por tanto su inmunidad al olvido) pero con un costo de almacenamiento por tarea minúsculo y constante en lugar de una columna entera.

## Notas y enlaces

- Preprint: arXiv:1606.04671 (junio 2016).
- Afiliación: Google DeepMind, Londres.
- Enlaces del curso: [Clase 32](/clases/clase-32) · fundamento de [aprendizaje continuo](/fundamentos/aprendizaje-continuo) · [Piggyback (Mallya et al., 2018)](/papers/piggyback-mallya-2018) · [EWC (Kirkpatrick et al., 2017)](/papers/ewc-kirkpatrick-2017).
