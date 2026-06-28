# Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Mastering the game of Go with deep neural networks and tree search*.
- **Autores:** David Silver*, Aja Huang*, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel y Demis Hassabis (Google DeepMind; *contribución equitativa).
- **Venue:** *Nature*, vol. 529, pp. 484–489, 28 de enero de 2016. **DOI:** 10.1038/nature16961. Recibido 11 nov 2015; aceptado 5 ene 2016. **Portada de Nature** ese número.
- **Correspondencia:** David Silver (davidsilver@google.com), Demis Hassabis (demishassabis@google.com).

AlphaGo es el programa que, por primera vez en la historia, derrotó a un jugador profesional humano de Go en el tablero completo de 19×19 sin handicap: venció al campeón europeo **Fan Hui (2 dan profesional) por 5 a 0** en los juegos formales (octubre de 2015), un hito que se creía a "al menos una década de distancia". Pocos meses después de esta publicación, la versión sucesora derrotaría a **Lee Sedol (9 dan)** por 4-1 en Seúl, marzo de 2016, en uno de los acontecimientos más mediáticos de la historia de la IA.

La tesis técnica del paper es que el Go —considerado "el más desafiante de los juegos clásicos para la inteligencia artificial" por su enorme espacio de búsqueda y la dificultad de evaluar posiciones— se vuelve tratable combinando tres ingredientes: (1) una **policy network** (red de políticas, una CNN) entrenada primero por aprendizaje supervisado de partidas de expertos y luego refinada por aprendizaje reforzado mediante self-play; (2) una **value network** (red de valor) que evalúa qué tan buena es una posición; y (3) un **Monte Carlo Tree Search (MCTS)** que usa ambas redes para guiar la búsqueda en árbol. El resultado: AlphaGo ganó el **99.8% de las partidas (494 de 495)** contra otros programas de Go.

Para la Clase 31 (Aprendizaje Reforzado) esto importa porque AlphaGo es la **aplicación cumbre del RL**: pone en escena, juntos y a escala, los tres conceptos centrales que la clase introduce —**política**, **función de valor** y **gradiente de política**—, todo alimentado por **self-play**. Es el contrapunto natural a DQN (Mnih et al., 2015): si DQN demostró RL profundo en Atari aprendiendo desde píxeles, AlphaGo demostró que el RL profundo, acoplado a búsqueda, podía conquistar el problema de decisión que durante décadas resistió a la IA.

## 2. Contexto histórico: Go como el gran desafío de la IA

Todo juego de información perfecta tiene una **función de valor óptima** $v^*(s)$ que determina el resultado de la partida desde cualquier estado $s$ bajo juego perfecto. En principio se resuelve recorriendo recursivamente un árbol de búsqueda con aproximadamente $b^d$ secuencias de jugadas, donde $b$ es la **amplitud** (jugadas legales por posición) y $d$ la **profundidad** (largo de la partida). El ajedrez tiene $b\approx 35$, $d\approx 80$; el Go tiene $b\approx 250$, $d\approx 150$. Eso da del orden de $250^{150}$ secuencias posibles —un número astronómico que vuelve la **búsqueda exhaustiva por fuerza bruta sencillamente intratable**.

El campo había aprendido dos principios para podar ese espacio. Primero, reducir la **profundidad** por *evaluación de posición*: truncar el árbol en un estado $s$ y reemplazar el subárbol por una aproximación $v(s)\approx v^*(s)$. Esto dio rendimiento sobrehumano en ajedrez (Deep Blue), damas y othello, pero **se creía intratable en Go** por la complejidad de evaluar una posición. Segundo, reducir la **amplitud** muestreando jugadas desde una política $p(a\mid s)$. Los **rollouts de Monte Carlo** llevan esta idea al extremo: muestrean partidas completas hasta el final sin ramificar, y promedian los resultados para estimar el valor de una posición; lograron nivel sobrehumano en backgammon y Scrabble, pero solo **nivel amateur débil en Go**.

El estado del arte previo a AlphaGo era el **Monte Carlo Tree Search (MCTS)**: usa rollouts para estimar el valor de cada nodo, y a medida que corren más simulaciones el árbol crece y las estimaciones mejoran; asintóticamente converge al juego óptimo. Los programas más fuertes (Crazy Stone, Zen, Pachi, Fuego) eran MCTS reforzado con políticas entrenadas para predecir jugadas humanas, usadas para estrechar la búsqueda a un haz de jugadas de alta probabilidad. Pero ese trabajo previo estaba **limitado a políticas y funciones de valor someras** (combinaciones lineales de features hechas a mano). El salto de AlphaGo fue reemplazar esas representaciones lineales por **redes neuronales convolucionales profundas**, leyendo el tablero como una imagen 19×19 —la misma intuición arquitectónica que había triunfado en clasificación de imágenes, reconocimiento facial y los juegos de Atari de DQN.

## 3. Contribución central

AlphaGo introduce **tres componentes de aprendizaje profundo y un algoritmo de búsqueda que los integra**:

1. **Policy networks (redes de políticas)** que seleccionan jugadas, entrenadas por una **combinación novedosa** de aprendizaje supervisado a partir de partidas humanas y aprendizaje reforzado a partir de self-play.
2. **Value network (red de valor)** que evalúa posiciones del tablero —la pieza que se creía imposible para Go.
3. Un **algoritmo de búsqueda nuevo** que combina la simulación de Monte Carlo con las redes de política y de valor (APV-MCTS).

El claim más sorprendente del paper: **sin ninguna búsqueda hacia adelante (sin lookahead)**, las redes neuronales por sí solas ya juegan al nivel de los mejores programas MCTS que simulan miles de partidas aleatorias. La red de valor sola alcanza la precisión de los rollouts de Monte Carlo usando la política RL, pero con **15.000 veces menos cómputo**. La búsqueda completa, que combina redes y rollouts, las supera a todas.

La idea de diseño que unifica todo es la **reducción de profundidad y amplitud vía aprendizaje profundo**: la value network reduce la profundidad (no hace falta jugar hasta el final para evaluar), la policy network reduce la amplitud (no hace falta considerar todas las jugadas legales), y MCTS las orquesta. Como nota el paper en la discusión, durante el match contra Fan Hui AlphaGo evaluó **miles de veces menos posiciones** que Deep Blue contra Kasparov, compensando con posiciones elegidas más inteligentemente (policy network) y evaluadas más precisamente (value network) —un enfoque "quizás más cercano a cómo juegan los humanos".

## 4. El pipeline de entrenamiento

El entrenamiento procede en varias etapas encadenadas (Fig. 1 del paper), donde cada red alimenta a la siguiente.

### 4.1. Policy network supervisada (SL) — aprender de expertos

La SL policy network $p_\sigma(a\mid s)$ es una CNN de **13 capas** que alterna convoluciones con no linealidades rectificadoras y termina en un softmax sobre todas las jugadas legales. Se entrena por **ascenso de gradiente estocástico** para maximizar la verosimilitud de la jugada humana $a$ observada en el estado $s$:

$$\Delta\sigma \propto \frac{\partial \log p_\sigma(a\mid s)}{\partial\sigma}$$

Se entrenó con **30 millones de posiciones** del KGS Go Server (partidas de jugadores 6-9 dan). Alcanzó **57.0% de precisión** prediciendo la jugada del experto en el test set (55.7% usando solo posición cruda e historia de jugadas), contra el 44.4% del estado del arte de otros grupos. Detalle clave que la clase debe subrayar: **mejoras pequeñas en precisión se traducen en mejoras grandes en fuerza de juego**. Las redes más grandes son más precisas pero más lentas de evaluar durante la búsqueda —una tensión de diseño recurrente en RL aplicado.

En paralelo se entrena una **rollout policy** $p_\pi$ rápida pero menos precisa: un softmax lineal sobre features de patrones locales 3×3, con **24.2% de precisión** pero tarda solo **2 microsegundos** por jugada (vs. 3 ms de la red completa). Esta velocidad es la que hace viables los rollouts dentro de la búsqueda.

### 4.2. Policy network por RL — refinar con self-play y gradiente de política

La segunda etapa es **aprendizaje reforzado por gradiente de política**. La RL policy network $p_\rho$ tiene la misma estructura que la SL y se **inicializa con sus pesos** ($\rho=\sigma$). Luego juega contra **versiones anteriores de sí misma**, muestreadas aleatoriamente de un pool de oponentes (esto estabiliza el entrenamiento y evita el sobreajuste a la política actual —una receta de self-play que la clase debe destacar).

La recompensa es **dispersa**: $r(s)=0$ en todos los pasos no terminales, y el resultado terminal $z_t=\pm r(s_T)$ es $+1$ por ganar y $-1$ por perder, desde la perspectiva del jugador en el paso $t$. Los pesos se actualizan por ascenso de gradiente estocástico en la dirección que **maximiza el resultado esperado** —esto es exactamente el algoritmo **REINFORCE** (Williams, 1992):

$$\Delta\rho \propto \frac{\partial \log p_\rho(a_t\mid s_t)}{\partial\rho}\, z_t$$

En los Methods se precisa que se usa REINFORCE **con un baseline** $v(s_t)$ para reducir varianza (en la segunda pasada, el baseline es la propia value network). Resultados: la RL policy ganó **más del 80%** de las partidas contra la SL policy, y —sin ninguna búsqueda— ganó el **85%** contra Pachi (el MCTS open-source más fuerte, que corre 100.000 simulaciones por jugada). El estado del arte previo basado solo en supervisado ganaba apenas el 11% contra Pachi.

### 4.3. Value network — evaluar posiciones por regresión

La etapa final estima una función de valor $v^{p_\rho}(s)$ que predice el resultado de partidas jugadas por la política RL para ambos jugadores. La value network $v_\theta(s)$ tiene arquitectura similar a la policy network pero su salida es **un único escalar** (una unidad tanh) en vez de una distribución. Se entrena por regresión, minimizando el **error cuadrático medio** entre el valor predicho y el resultado real $z$:

$$\Delta\theta \propto \frac{\partial v_\theta(s)}{\partial\theta}\,(z - v_\theta(s))$$

Aquí aparece una **trampa de sobreajuste que la clase debe enseñar como advertencia general del RL**: entrenar con posiciones de partidas completas falla porque las posiciones sucesivas están fuertemente correlacionadas (difieren en una sola piedra) pero comparten el mismo target de regresión. La red **memorizó** los resultados (MSE 0.37 en test vs. 0.19 en train). La solución fue generar un dataset nuevo de **30 millones de posiciones distintas, cada una de una partida de self-play separada**. Con esto el MSE bajó a 0.226/0.234 (train/test), indicando sobreajuste mínimo. Una sola evaluación de $v_\theta$ se aproximó a la precisión de los rollouts con la RL policy, usando **15.000 veces menos cómputo**.

## 5. Búsqueda: MCTS guiado por las redes (APV-MCTS)

AlphaGo combina ambas redes en un algoritmo MCTS asíncrono (APV-MCTS, Fig. 3). Cada arista $(s,a)$ del árbol almacena un valor de acción $Q(s,a)$, un conteo de visitas $N(s,a)$ y una probabilidad a priori $P(s,a)$. Cada simulación tiene cuatro fases:

- **Selección.** Desde la raíz se desciende eligiendo en cada paso la acción que maximiza el valor más un bono de exploración:
  $$a_t = \arg\max_a\big(Q(s_t,a) + u(s_t,a)\big), \qquad u(s,a) \propto \frac{P(s,a)}{1+N(s,a)}$$
  El bono es proporcional a la probabilidad a priori (de la policy network) y decae con las visitas, **favoreciendo inicialmente jugadas con alto prior y pocas visitas, y asintóticamente las de alto valor** (una variante del algoritmo PUCT).
- **Expansión.** Al llegar a una hoja, se expande: la posición se procesa **una vez** por la policy network $p_\sigma$, y sus salidas se almacenan como priors $P(s,a)=p_\sigma(a\mid s)$ para cada acción.
- **Evaluación.** La hoja $s_L$ se evalúa de **dos maneras complementarias**: (1) por la value network $v_\theta(s_L)$, y (2) por un rollout hasta el final con la política rápida $p_\pi$, obteniendo el resultado $z_L$. Se combinan con un parámetro de mezcla $\lambda$:
  $$V(s_L) = (1-\lambda)\,v_\theta(s_L) + \lambda\,z_L$$
- **Backup.** Al final de la simulación, los valores de acción y conteos de todas las aristas recorridas se actualizan al promedio de las evaluaciones que pasaron por ellas.

Tras completar la búsqueda, AlphaGo elige **la jugada más visitada** desde la raíz (más robusta a outliers que maximizar el valor de acción).

Dos hallazgos contraintuitivos que la clase debe discutir. Primero, **la SL policy funcionó mejor que la RL policy dentro de MCTS** como fuente de priors, presumiblemente porque los humanos seleccionan un haz diverso de jugadas prometedoras, mientras el RL optimiza para la única mejor jugada (menos diversidad para guiar la búsqueda). Segundo, **la value network derivada de la RL policy sí funcionó mejor** que la derivada de la SL. Es decir: el RL ayuda a *evaluar* posiciones pero la diversidad humana ayuda a *guiar* la exploración —un equilibrio sutil entre explotación y exploración.

Las evaluaciones de redes profundas cuestan varios órdenes de magnitud más que las heurísticas tradicionales, por lo que AlphaGo usa búsqueda **asíncrona multihilo**: simulaciones en CPUs, redes en GPUs. La versión final usó 40 hilos de búsqueda, 48 CPUs y 8 GPUs; la versión distribuida escaló a **1.202 CPUs y 176 GPUs**.

## 6. Experimentos y resultados

**Torneo contra otros programas (escala Elo).** En un torneo interno con 5 s de cómputo por jugada, AlphaGo de una sola máquina resultó **muchos rangos dan más fuerte** que cualquier programa previo, ganando 494 de 495 partidas (**99.8%**) contra Crazy Stone, Zen, Pachi, Fuego y GnuGo. Incluso dando a los rivales **cuatro piedras de handicap**, AlphaGo ganó el 77%, 86% y 99% contra Crazy Stone, Zen y Pachi. La versión distribuida ganó el 77% contra la de una sola máquina y el 100% contra los demás programas. En escala Elo (anclada al rating de Fan Hui, ~2.908), AlphaGo distribuido supera holgadamente a todos los programas y se acerca al rango profesional.

**Ablación de componentes (Fig. 4b).** Las variantes que evalúan solo con value network ($\lambda=0$) o solo con rollouts ($\lambda=1$) ya superan a todos los demás programas; pero la **evaluación mixta ($\lambda=0.5$) es la mejor**, ganando $\geq$95% contra las otras variantes. Esto muestra que value network y rollouts son **complementarios**: la value network aproxima el resultado de la política fuerte pero lenta $p_\rho$, mientras los rollouts puntúan con precisión partidas de la política débil pero rápida $p_\pi$.

**Match contra Fan Hui.** Del 5 al 9 de octubre de 2015, AlphaGo distribuido enfrentó a **Fan Hui**, profesional 2 dan y campeón europeo 2013-2015, en un match formal de cinco partidas con arbitraje imparcial, komi 7.5, sin handicap y reglas chinas. **AlphaGo ganó 5-0** los juegos formales (y 3-2 los informales con controles de tiempo más cortos). Fue la primera vez que un programa derrotó a un profesional humano en el Go completo sin handicap.

## 7. Limitaciones

- **Dependencia de datos de expertos.** La SL policy network requiere **30 millones de posiciones de partidas humanas** del KGS para arrancar el pipeline. AlphaGo no aprende de cero: se apoya en el conocimiento humano acumulado como punto de partida. Esta es la limitación que **AlphaGo Zero** (Silver et al., 2017) eliminaría, aprendiendo enteramente por self-play desde reglas y tablero vacío, sin una sola partida humana —y resultando *más* fuerte.
- **Conocimiento de dominio en los rollouts y features.** Aunque las redes se entrenan "de extremo a extremo", los inputs son **48 planos de features 19×19** derivados de las reglas (color de piedra, libertades, capturas, *ladders*, etc.) y la rollout policy usa patrones locales hechos a mano. AlphaGo Zero también simplificaría esto a solo el estado crudo del tablero.
- **Costo computacional inmenso.** La versión de competición exigió clusters con cientos de CPUs y GPUs y semanas de entrenamiento (3 semanas para la SL policy con 50 GPUs; una semana para la value network). Lejos del alcance de cualquier laboratorio modesto.
- **Especificidad del dominio.** AlphaGo está construido para Go (información perfecta, dos jugadores, suma cero, transiciones deterministas). La generalización a otros dominios requeriría rehacer features y arquitectura —algo que la línea AlphaZero/MuZero abordaría después.

## 8. Impacto y legado

AlphaGo es uno de los **hitos históricos de la inteligencia artificial**, comparable a la victoria de Deep Blue sobre Kasparov en 1997 pero conceptualmente más profundo: Deep Blue usaba una función de evaluación hecha a mano y fuerza bruta especializada; AlphaGo aprendió sus funciones de selección y evaluación **directamente del juego, mediante métodos de propósito general** de aprendizaje supervisado y reforzado. El paper enmarca el Go como ejemplar de las dificultades de la IA —decisión desafiante, espacio de búsqueda intratable, solución óptima demasiado compleja para aproximar directamente— y concluye que alcanzar nivel profesional "da esperanza de que el rendimiento a nivel humano pueda lograrse ahora en otros dominios aparentemente intratables".

Esa promesa se cumplió: AlphaGo abrió la línea **AlphaGo Zero** (2017, self-play puro), **AlphaZero** (2018, mismo algoritmo para Go, ajedrez y shogi sin conocimiento de dominio) y **MuZero** (2020, aprendiendo incluso el modelo del entorno). La demostración de que **deep RL + búsqueda** podían superar el desafío que resistió décadas reactivó el interés mundial en el aprendizaje reforzado y fue **portada de Nature**. La derrota de Lee Sedol en marzo de 2016, vista por más de 200 millones de personas, convirtió a AlphaGo en un símbolo cultural del avance de la IA.

## 9. Conexión con la Clase 31 (Aprendizaje Reforzado)

AlphaGo es la **aplicación cumbre** de los conceptos que la Clase 31 introduce, y conviene leerlo precisamente como tal —no como un programa de Go, sino como una demostración integrada de RL:

- **Política ($\pi$ / policy network).** La clase define la política como la distribución sobre acciones que el agente sigue. AlphaGo materializa dos: la SL policy (imitación) y la RL policy (optimizada por recompensa). El estudiante ve la diferencia entre *aprender a imitar* y *aprender a ganar*.
- **Gradiente de política (REINFORCE).** El refinamiento por self-play es REINFORCE puro con baseline: la regla $\Delta\rho \propto \nabla\log p_\rho(a_t\mid s_t)\,z_t$ es el algoritmo de policy gradient que la clase deriva. AlphaGo lo escala a un dominio real y muestra el truco de varianza (baseline = value network).
- **Función de valor.** La value network es la estimación de $v^*(s)$ que la clase formaliza, aprendida por regresión de Monte Carlo sobre resultados de self-play. La trampa de correlación/sobreajuste es una lección transferible sobre por qué la independencia de muestras importa en RL.
- **Self-play como generador de experiencia.** El bucle de jugar contra versiones pasadas de sí mismo es la receta que la clase presenta como motor de mejora autónoma sin supervisión humana adicional.
- **RL más allá de Atari/DQN.** Si DQN (Mnih et al., 2015) mostró que el RL profundo podía aprender control desde píxeles en Atari mediante **value-based learning (Q-learning)**, AlphaGo muestra el otro gran eje del RL —**policy gradient + función de valor + búsqueda**— y lo lleva a un problema de planificación combinatoria que la fuerza bruta no puede tocar. Los dos papers juntos delimitan el espacio de diseño del deep RL moderno.

Para profundizar en los fundamentos teóricos del aprendizaje reforzado, ver [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado). Para el contexto completo de esta clase, ver [/clases/clase-31](/clases/clase-31). Para el otro pilar del deep RL —el aprendizaje basado en valor con DQN en Atari—, ver el análisis de [DQN (Mnih et al., 2015)](/papers/dqn-nature-mnih-2015).
