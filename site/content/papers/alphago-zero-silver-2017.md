---
title: "AlphaGo Zero: Mastering Go without Human Knowledge (2017)"
weight: 374
math: true
---

{{< paper-card
    title="Mastering the Game of Go without Human Knowledge"
    authors="David Silver, Julian Schrittwieser, Karen Simonyan, et al. (DeepMind)"
    year="2017"
    venue="Nature 2017"
    pdf="/papers/alphago-zero-silver-2017.pdf" >}}
AlphaGo Zero es el programa que aprendió Go a nivel sobrehumano **partiendo de cero (*tabula rasa*)**: sin una sola partida humana, sin más conocimiento de dominio que las reglas del juego, y **derrotando 100 a 0 a AlphaGo Lee** —la versión que había vencido a Lee Sedol. Su tesis es directa: los sistemas entrenados por imitación de datos expertos quedan atados a la disponibilidad de esos datos y, sobre todo, a un **techo de rendimiento impuesto por la calidad humana**; en cambio, un sistema entrenado desde su propia experiencia por [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado) puede **exceder las capacidades humanas**. Un nuevo algoritmo integra la búsqueda hacia adelante (MCTS) dentro del bucle de entrenamiento con una **red única** (política + valor sobre torso ResNet). Es el caso de estudio estrella de la [Clase 33](/clases/clase-33) para mostrar que el **RL puro por self-play supera al maestro** (slide 39), donde la [imitación](/fundamentos/aprendizaje-por-imitacion) rara vez lo logra.
{{< /paper-card >}}

---

## Contexto: AlphaGo (2016) sí usó imitación; qué elimina Zero

Para dimensionar el aporte de Zero conviene recordar cómo funcionaba [AlphaGo (2016)](/papers/alphago-silver-2016) —el paper de la [Clase 31](/clases/clase-31)—. Aquella arquitectura (llamada aquí *AlphaGo Fan*, y su sucesora *AlphaGo Lee*) partía de dos redes separadas: una de **políticas** y una de **valor**. Y, crucialmente, la red de políticas se **inicializaba por aprendizaje supervisado para predecir las jugadas de expertos humanos** —30 millones de posiciones del servidor KGS—; solo después se refinaba con RL por gradiente de política. Ese primer paso supervisado es, en el vocabulario de la clase, [aprendizaje por imitación / clonación de comportamiento](/fundamentos/aprendizaje-por-imitacion): el modelo aprende a reproducir la decisión del experto en cada estado.

AlphaGo Zero elimina todo lo humano y todo lo hecho a mano. El paper enumera cuatro diferencias respecto de AlphaGo:

1. **Se entrena únicamente por RL de self-play, partiendo de juego aleatorio**, sin supervisión ni datos humanos. Desaparece el arranque supervisado —desaparece la imitación.
2. Usa **solo las piedras blancas y negras del tablero** como entrada, en lugar de los 48 planos de features derivados de conocimiento experto de Go.
3. Usa **una sola red neuronal**, en vez de redes de políticas y de valor separadas.
4. Usa una **búsqueda en árbol más simple**, apoyada en esa única red, **sin ningún rollout de Monte Carlo**.

En términos de la clase: AlphaGo (2016) = **imitación + RL**; AlphaGo Zero (2017) = **RL puro**. Toda la maquinaria de imitación que en 2016 se consideraba indispensable para arrancar, en 2017 resulta no solo prescindible, sino un lastre para el rendimiento asintótico.

## Método y contribución

**Red única.** Una red $f_\theta$ con torso residual (ResNet, 19 o 39 bloques residuales de 256 filtros) y dos cabezales produce simultáneamente política y valor a partir de la representación cruda del tablero (una pila $19\times 19\times 17$: piedras propias y del oponente en los últimos 8 pasos, más el color a jugar):

$$(p, v) = f_\theta(s)$$

Combinar política y valor en una red única (frente a redes separadas) mejora el rendimiento en unos **600 Elo**: el objetivo dual regulariza la red hacia una representación común.

**MCTS como operador de mejora de política.** La búsqueda selecciona acciones maximizando $Q(s,a) + U(s,a)$ con una variante de PUCT, evalúa cada hoja **una sola vez** con la red (sin rollouts) y devuelve las probabilidades de búsqueda $\pi(a\mid s_0) \propto N(s_0, a)^{1/\tau}$. El punto clave: **$\pi$ suele seleccionar jugadas mucho más fuertes que la política cruda $p$ de la red**. Por eso el MCTS "es un potente operador de mejora de política".

**Iteración de política por self-play.** La red se entrena para acercar $(p, v)$ a la política mejorada por la búsqueda y al ganador del self-play $(\pi, z)$, con la pérdida:

$$l = (z - v)^2 - \pi^\top \log p + c\lVert\theta\rVert^2$$

El término $-\pi^\top\log p$ es entropía cruzada entre la política de la red y la del MCTS: es **destilar la política mejorada por búsqueda de vuelta en la red**. La sutileza para la clase: AlphaGo Zero **sí "imita" algo, pero no a un humano —se imita a sí mismo mejorado por la búsqueda**. El maestro no tiene techo: es una versión ligeramente superior del propio alumno, generada por el cómputo del MCTS.

## Resultados

- **100 a 0 contra AlphaGo Lee.** Zero superó a AlphaGo Lee **tras solo 36 horas** (Lee había requerido meses), y a las 72 horas lo venció **100 partidas a 0**, usando **una sola máquina con 4 TPU** frente a las 48 TPU distribuidas de Lee.
- **RL puro vs. imitación (el experimento decisivo).** Una red idéntica entrenada por aprendizaje supervisado de jugadas KGS **arrancó más rápido** y predijo mejor la jugada humana (**60.4 %** de precisión vs. 49.0 % del modelo RL), pero **jugó peor** y fue superada por el jugador auto-aprendido dentro de las primeras 24 horas. Predecir bien la jugada humana no equivale a jugar bien.
- **Curva Elo.** Red cruda de Zero (sin búsqueda): **3.055**; AlphaGo Lee: **3.739**; AlphaGo Master: **4.858**; **AlphaGo Zero: 5.185**. En match directo, Zero venció a Master **89 a 11**. El paso a red residual aportó ~600 Elo y combinar política+valor otros ~600.
- **Redescubrimiento y superación.** Zero reconstruyó desde primeros principios fuseki, tesuji, vida y muerte, ko y josekis profesionales; más tarde **descubrió y prefirió variantes nuevas, previamente desconocidas** para los humanos.

## Limitaciones

- **Dominio de información perfecta y suma cero.** El método es más directamente aplicable a juegos deterministas, de dos jugadores y con ganador bien definido.
- **El self-play requiere un simulador perfecto del entorno.** El MCTS **necesita las reglas del juego** para simular posiciones y puntuar estados terminales. Sin un simulador exacto y barato del mundo, el operador de mejora de política no existe: esta es la limitación clave de cara a dominios reales.
- **Recompensa verificable.** El resultado $z\in\{-1,+1\}$ es objetivo e inequívoco al final de cada partida.
- **Persiste algún conocimiento de dominio.** Reglas dentro del MCTS, puntuación Tromp-Taylor, grilla $19\times 19$ e invariancia a rotaciones/reflexiones del Go.

## Aclaración: Zero vs. AlphaZero

La slide del curso rotula el caso como "AlphaZero", pero cita **este** paper (Silver et al., *Nature* 2017), que corresponde a **AlphaGo Zero** —específico de Go—. **AlphaZero** (Silver et al., *Science* 2018) es el trabajo posterior que **generaliza el mismo algoritmo** (red residual política+valor, MCTS como operador de mejora, self-play) a ajedrez, shogi y Go sin ajustes por dominio. Para el argumento de la clase ambos sirven; el paper citado es el que introduce por primera vez la eliminación total de datos humanos.

## Por qué importa para la Clase 33

AlphaGo Zero es el contraejemplo definitivo con que la [Clase 33](/clases/clase-33) delimita cuándo conviene imitar y cuándo reforzar:

- **La imitación (behavioral cloning) tiene un techo: el maestro.** La red que copia a expertos humanos aprende rápido y predice mejor las jugadas humanas, pero se estanca por debajo del RL. Imitar es aprender a *ser como* el demostrador, nunca a superarlo.
- **El RL puro puede exceder al humano** porque su señal no es "haz lo que hace el experto" sino "gana": al optimizar directamente la recompensa, el agente descubre estrategias que ningún humano jugó.
- **El precio del RL puro es la exploración y la recompensa.** Solo es viable aquí porque existen un **simulador perfecto barato** (las reglas del Go) y una **recompensa verificable**. Cuando esos supuestos faltan, la balanza vuelve a inclinarse hacia la imitación —el escenario típico de dominios reales como el matching de pacientes, donde no hay self-play posible.

Junto a [AlphaGo (2016)](/papers/alphago-silver-2016) —que sí usó imitación de partidas humanas— y a [DQN](/papers/dqn-nature-mnih-2015) de la [Clase 31](/clases/clase-31), AlphaGo Zero delimita el espacio de diseño del deep RL moderno.
