---
title: "AlphaGo: Mastering the game of Go (2016)"
weight: 355
math: true
---

{{< paper-card
    title="Mastering the game of Go with deep neural networks and tree search"
    authors="David Silver, Aja Huang, Chris J. Maddison, et al. (DeepMind)"
    year="2016"
    venue="Nature 2016"
    pdf="/papers/alphago-silver-2016.pdf" >}}
AlphaGo fue el primer programa que derrotó a un jugador profesional humano de Go en el tablero completo de 19×19 sin handicap: venció al campeón europeo **Fan Hui (2 dan) por 5 a 0** en octubre de 2015, y meses después su versión sucesora venció a **Lee Sedol (9 dan) por 4-1** en marzo de 2016. La tesis técnica: el Go —intratable por fuerza bruta, con ~$250^{150}$ secuencias posibles— se vuelve abordable combinando tres ingredientes de aprendizaje profundo: una **policy network** (CNN entrenada por imitación de partidas humanas y refinada por RL en self-play), una **value network** (evalúa posiciones) y un **Monte Carlo Tree Search (MCTS)** que usa ambas para guiar la búsqueda. AlphaGo ganó el **99.8%** de las partidas (494 de 495) contra otros programas de Go. Es la aplicación cumbre del aprendizaje reforzado de la [Clase 31](/clases/clase-31).
{{< /paper-card >}}

---

## Contexto: Go como el gran desafío de la IA

Todo juego de información perfecta tiene una **función de valor óptima** $v^*(s)$ que determina el resultado desde cualquier estado $s$ bajo juego perfecto. En principio se resuelve recorriendo un árbol de búsqueda con aproximadamente $b^d$ secuencias, donde $b$ es la **amplitud** (jugadas legales por posición) y $d$ la **profundidad** (largo de la partida). El ajedrez tiene $b\approx 35$, $d\approx 80$; el Go tiene $b\approx 250$, $d\approx 150$. Eso da del orden de $250^{150}$ secuencias —un número astronómico que vuelve la **búsqueda exhaustiva por fuerza bruta sencillamente intratable**.

El campo conocía dos formas de podar ese espacio. Primero, reducir la **profundidad** evaluando posiciones: truncar el árbol en un estado $s$ y reemplazar el subárbol por una aproximación $v(s)\approx v^*(s)$. Esto dio rendimiento sobrehumano en ajedrez (Deep Blue), pero **se creía intratable en Go** por la dificultad de evaluar una posición. Segundo, reducir la **amplitud** muestreando jugadas desde una política $p(a\mid s)$. Los **rollouts de Monte Carlo** llevan la idea al extremo: muestrean partidas completas hasta el final y promedian los resultados; lograron nivel sobrehumano en backgammon, pero solo **nivel amateur en Go**.

El estado del arte previo era el **MCTS**: usa rollouts para estimar el valor de cada nodo, y conforme corren más simulaciones el árbol crece y las estimaciones mejoran. Los programas más fuertes (Crazy Stone, Zen, Pachi, Fuego) eran MCTS con políticas someras —combinaciones lineales de features hechas a mano. El salto de AlphaGo fue reemplazar esas representaciones lineales por **redes convolucionales profundas**, leyendo el tablero como una imagen 19×19, la misma intuición que había triunfado en clasificación de imágenes y en los juegos de Atari de [DQN](/papers/dqn-nature-mnih-2015).

## Las tres redes y la búsqueda

AlphaGo introduce tres componentes de aprendizaje profundo más un algoritmo de búsqueda que los integra. El claim más sorprendente: **sin ninguna búsqueda hacia adelante**, las redes neuronales por sí solas ya juegan al nivel de los mejores programas MCTS. La idea unificadora es **reducir profundidad y amplitud vía aprendizaje profundo**: la value network reduce la profundidad (no hace falta jugar hasta el final para evaluar), la policy network reduce la amplitud (no hace falta considerar todas las jugadas), y MCTS las orquesta.

### Policy network supervisada (SL) — aprender de expertos

La SL policy network $p_\sigma(a\mid s)$ es una CNN de **13 capas** que termina en un softmax sobre todas las jugadas legales. Se entrena por ascenso de gradiente estocástico para maximizar la verosimilitud de la jugada humana observada:

$$\Delta\sigma \propto \frac{\partial \log p_\sigma(a\mid s)}{\partial\sigma}$$

Se entrenó con **30 millones de posiciones** del KGS Go Server (partidas 6-9 dan) y alcanzó **57.0% de precisión** prediciendo la jugada del experto, contra el 44.4% del estado del arte previo. Detalle clave: **mejoras pequeñas en precisión se traducen en mejoras grandes en fuerza de juego**, pero las redes más grandes son más lentas de evaluar en la búsqueda —una tensión recurrente. En paralelo se entrena una **rollout policy** $p_\pi$ rápida pero imprecisa (24.2% de precisión, solo 2 microsegundos por jugada vs. 3 ms de la red completa): esa velocidad es la que hace viables los rollouts dentro de la búsqueda.

### Policy network por RL — refinar con self-play y gradiente de política

La segunda etapa es **aprendizaje reforzado por gradiente de política**. La RL policy network $p_\rho$ tiene la misma estructura que la SL y se **inicializa con sus pesos**. Luego juega contra **versiones anteriores de sí misma**, muestreadas al azar de un pool de oponentes (esto estabiliza el entrenamiento y evita el sobreajuste a la política actual). La recompensa es **dispersa**: $r(s)=0$ en todos los pasos no terminales, y el resultado terminal $z_t=\pm 1$ es $+1$ por ganar y $-1$ por perder. Los pesos se actualizan en la dirección que maximiza el resultado esperado —exactamente el algoritmo **REINFORCE** (Williams, 1992), con un baseline $v(s_t)$ para reducir varianza:

$$\Delta\rho \propto \frac{\partial \log p_\rho(a_t\mid s_t)}{\partial\rho}\, z_t$$

La RL policy ganó **más del 80%** de las partidas contra la SL policy y —sin búsqueda alguna— el **85%** contra Pachi (el MCTS open-source más fuerte, con 100.000 simulaciones por jugada). El enfoque supervisado previo apenas ganaba el 11% contra Pachi.

### Value network — evaluar posiciones por regresión

La etapa final estima una función de valor $v_\theta(s)$ que predice el resultado de partidas jugadas por la política RL. Es una red de arquitectura similar pero con salida **escalar** (una unidad tanh) en vez de una distribución. Se entrena por regresión, minimizando el error cuadrático medio entre el valor predicho y el resultado real $z$:

$$\Delta\theta \propto \frac{\partial v_\theta(s)}{\partial\theta}\,(z - v_\theta(s))$$

Aquí aparece una **trampa de sobreajuste** que conviene enseñar como advertencia general del RL: entrenar con posiciones de partidas completas falla porque las posiciones sucesivas están fuertemente correlacionadas (difieren en una sola piedra) pero comparten el mismo target. La red **memorizó** los resultados (MSE 0.37 en test vs. 0.19 en train). La solución fue generar **30 millones de posiciones distintas, cada una de una partida de self-play separada**, lo que redujo el MSE a 0.226/0.234. Una sola evaluación de $v_\theta$ se aproximó a la precisión de los rollouts, pero con **15.000 veces menos cómputo**.

### Búsqueda: MCTS guiado por las redes

AlphaGo combina ambas redes en un MCTS asíncrono. Cada arista $(s,a)$ almacena un valor de acción $Q(s,a)$, un conteo de visitas $N(s,a)$ y una probabilidad a priori $P(s,a)$. Cada simulación tiene cuatro fases:

- **Selección.** Desde la raíz se desciende eligiendo en cada paso la acción que maximiza el valor más un bono de exploración:
  $$a_t = \arg\max_a\big(Q(s_t,a) + u(s_t,a)\big), \qquad u(s,a) \propto \frac{P(s,a)}{1+N(s,a)}$$
  El bono es proporcional al prior (de la policy network) y decae con las visitas, favoreciendo al inicio jugadas con alto prior y pocas visitas, y asintóticamente las de alto valor (variante de PUCT).
- **Expansión.** Al llegar a una hoja, la posición se procesa una vez por la policy network y sus salidas se guardan como priors $P(s,a)=p_\sigma(a\mid s)$.
- **Evaluación.** La hoja $s_L$ se evalúa de dos maneras complementarias: por la value network $v_\theta(s_L)$ y por un rollout hasta el final con la política rápida $p_\pi$. Se combinan con un parámetro de mezcla $\lambda$:
  $$V(s_L) = (1-\lambda)\,v_\theta(s_L) + \lambda\,z_L$$
- **Backup.** Al final de cada simulación se actualizan los valores y conteos de todas las aristas recorridas.

Tras la búsqueda, AlphaGo elige **la jugada más visitada** desde la raíz (más robusta que maximizar el valor de acción). Dos hallazgos contraintuitivos: la **SL policy funcionó mejor que la RL policy como fuente de priors** (los humanos proponen un haz diverso de jugadas prometedoras, mientras el RL optimiza para la única mejor jugada), pero la **value network derivada de la RL policy sí funcionó mejor**. El RL ayuda a *evaluar* posiciones, mientras la diversidad humana ayuda a *guiar* la exploración.

## Resultados

En un torneo interno (5 s de cómputo por jugada), AlphaGo de una sola máquina resultó muchos rangos dan más fuerte que cualquier programa previo, ganando **494 de 495 partidas (99.8%)** contra Crazy Stone, Zen, Pachi, Fuego y GnuGo. Incluso dando a los rivales **cuatro piedras de handicap**, ganó entre 77% y 99%. La versión distribuida (1.202 CPUs, 176 GPUs) ganó el 77% contra la de una sola máquina. En la **ablación de componentes**, evaluar solo con value network ($\lambda=0$) o solo con rollouts ($\lambda=1$) ya superaba a todos los programas, pero la **evaluación mixta ($\lambda=0.5$) fue la mejor** (≥95% contra las otras variantes): value network y rollouts son complementarios.

Del 5 al 9 de octubre de 2015, AlphaGo enfrentó a **Fan Hui** (profesional 2 dan, campeón europeo) en un match formal de cinco partidas con komi 7.5, sin handicap y reglas chinas. **AlphaGo ganó 5-0** los juegos formales: la primera vez que un programa derrotó a un profesional humano en el Go completo sin handicap.

## Limitaciones

- **Dependencia de datos de expertos.** La SL policy requiere 30 millones de posiciones de partidas humanas para arrancar el pipeline: AlphaGo no aprende de cero. Esta es la limitación que **AlphaGo Zero** (2017) eliminaría, aprendiendo enteramente por self-play desde el tablero vacío —y resultando *más* fuerte.
- **Conocimiento de dominio.** Los inputs son **48 planos de features 19×19** derivados de las reglas (libertades, capturas, *ladders*, etc.) y la rollout policy usa patrones locales hechos a mano. AlphaGo Zero también simplificaría esto al estado crudo del tablero.
- **Costo computacional inmenso.** La versión de competición exigió clusters con cientos de CPUs y GPUs y semanas de entrenamiento. Lejos del alcance de un laboratorio modesto.
- **Especificidad del dominio.** AlphaGo está construido para Go (información perfecta, dos jugadores, suma cero, transiciones deterministas). La generalización a otros dominios la abordaría después la línea AlphaZero/MuZero.

## Por qué importa para la Clase 31

AlphaGo es la **aplicación cumbre** de los conceptos que la [Clase 31](/clases/clase-31) introduce —no como un programa de Go, sino como demostración integrada del [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado):

- **Política.** AlphaGo materializa dos: la SL policy (imitación) y la RL policy (optimizada por recompensa), mostrando la diferencia entre *aprender a imitar* y *aprender a ganar*.
- **Gradiente de política (REINFORCE).** El refinamiento por self-play es REINFORCE puro con baseline: la regla $\Delta\rho \propto \nabla\log p_\rho(a_t\mid s_t)\,z_t$ es el policy gradient que la clase deriva, llevado a escala real.
- **Función de valor.** La value network es la estimación de $v^*(s)$ aprendida por regresión de Monte Carlo. La trampa de correlación/sobreajuste es una lección transferible sobre por qué la independencia de muestras importa en RL.
- **Self-play.** El bucle de jugar contra versiones pasadas de sí mismo es el motor de mejora autónoma sin supervisión humana adicional.
- **RL más allá de Atari.** Si [DQN](/papers/dqn-nature-mnih-2015) mostró el RL profundo basado en valor (Q-learning) desde píxeles en Atari, AlphaGo muestra el otro gran eje —**policy gradient + función de valor + búsqueda**— en un problema de planificación combinatoria que la fuerza bruta no puede tocar. Los dos papers juntos delimitan el espacio de diseño del deep RL moderno, una columna del trabajo en [robótica](/dominios/robotica) y agentes secuenciales.

AlphaGo es uno de los hitos históricos de la IA: a diferencia de Deep Blue (función de evaluación hecha a mano y fuerza bruta), aprendió sus funciones de selección y evaluación **directamente del juego, con métodos de propósito general**. La derrota de Lee Sedol en marzo de 2016, vista por más de 200 millones de personas, lo convirtió en símbolo cultural del avance de la IA y abrió la línea AlphaGo Zero → AlphaZero → MuZero.
