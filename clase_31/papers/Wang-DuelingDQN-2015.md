# Dueling Network Architectures for Deep Reinforcement Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Dueling Network Architectures for Deep Reinforcement Learning*.
- **Autores:** Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas — todos de **Google DeepMind** (Londres, Reino Unido).
- **Venue:** ICML 2016 (Proceedings of the 33rd International Conference on Machine Learning). Ganador del *Best Paper Award* de ICML 2016.
- **Año:** 2015 (preprint) / 2016 (publicación). **Preprint:** arXiv:1511.06581v3 (5 abr 2016), [arxiv.org/abs/1511.06581](https://arxiv.org/abs/1511.06581).
- **Reproducibilidad:** los experimentos usan el Arcade Learning Environment (Bellemare et al., 2013), 57 juegos de Atari 2600; reusan optimizadores e hiperparámetros de van Hasselt et al. (2015).

Este paper no propone un **algoritmo** nuevo de aprendizaje reforzado: propone una **arquitectura de red neuronal** nueva, y ese matiz es toda su tesis. Los autores lo dicen explícitamente: la mayoría de los avances recientes en RL profundo se habían dedicado a inventar mejores algoritmos de control (Double DQN, prioritized replay) o a meter en RL arquitecturas ya existentes (convnets, LSTMs, autoencoders). Aquí toman el camino complementario: innovar en la **arquitectura de la red** de modo que sea intrínsecamente más adecuada para RL libre de modelo. El beneficio de ese enfoque es que la nueva red "se puede combinar fácilmente con algoritmos existentes y futuros" — avanza una red (Figura 1 del paper) pero usa algoritmos ya publicados, sin tocar la ecuación de aprendizaje.

La idea central, que da nombre a la **dueling architecture** ("arquitectura en duelo"), es separar la red en **dos flujos** (streams): uno estima la **función de valor de estado** $V(s)$ —cuán bueno es estar en un estado, independientemente de la acción— y otro estima la **función de ventaja** $A(s,a)$ —cuánto mejor o peor es cada acción respecto al promedio en ese estado. Ambos flujos comparten el mismo módulo convolucional de extracción de características, y un módulo de agregación especial los recombina en la estimación de $Q(s,a)$. El resultado es **un único Q-network con dos streams** que reemplaza el Q-network de un solo stream usado en DQN, sin imponer ningún cambio al algoritmo de RL subyacente.

Para la Clase 31 (Aprendizaje Reforzado) esto importa porque la dueling architecture es una **mejora arquitectónica directa del DQN** que se estudia en la clase: misma interfaz de entrada/salida, mismo bucle de Q-learning con experience replay y target network, pero una red que aprende más rápido *qué estados son valiosos* sin tener que aprender el efecto de cada acción en cada estado. Es, además, uno de los seis componentes que después se combinarían en **Rainbow** (Hessel et al., 2018).

## 2. Contexto: por qué estimar $V(s)$ y $A(s,a)$ por separado

DQN (Mnih et al., 2015) estima directamente la función $Q(s,a;\theta)$ con una sola red: las capas convolucionales procesan los píxeles, y una secuencia de capas totalmente conectadas produce un vector de Q-valores, uno por acción. Esto funciona, pero tiene una ineficiencia estructural que el paper diagnostica con precisión.

**La observación clave:** para muchos estados, *no hace falta* estimar el valor de cada elección de acción. El ejemplo recurrente del paper es el juego Enduro (conducir esquivando autos): saber si conviene moverse a la izquierda o a la derecha solo importa cuando una colisión es inminente. En la enorme mayoría de los frames —carretera despejada— la elección de acción no tiene ninguna repercusión sobre lo que pasa a continuación. En esos estados, lo que de verdad determina el retorno es el **valor del estado**, no la acción específica. Sin embargo, para los algoritmos basados en bootstrapping (como Q-learning), estimar bien $V(s)$ es de gran importancia *para todos los estados*, porque ese valor se propaga hacia atrás en cada actualización temporal.

DQN, al estimar $Q$ directamente, mezcla ambas cosas: cada Q-valor es $V(s)$ más la ventaja de esa acción, y la red tiene que aprender esa suma para cada par $(s,a)$ por separado. El paper ilustra el desperdicio con un dato concreto del juego Seaquest: tras entrenar con DDQN, el **action gap** promedio (la diferencia entre el Q-valor de la mejor y la segunda mejor acción) es de aproximadamente **0.04**, mientras que el valor de estado promedio ronda **15**. Es decir, la señal que distingue las acciones es ~375 veces más pequeña que la magnitud del valor que comparten. En un solo stream, pequeñas cantidades de ruido en las actualizaciones pueden reordenar las acciones y hacer que la política casi-greedy cambie abruptamente. Separar $V$ de $A$ vuelve la arquitectura robusta a este efecto de escala.

El antecedente teórico es viejo: la noción de mantener funciones de valor y de ventaja separadas se remonta a Baird (1993) y al *advantage updating* (Harmon et al., 1995; Harmon & Baird, 1996), donde la actualización de Bellman se descomponía en dos. La diferencia es que en aquellos trabajos la representación y el algoritmo estaban acoplados; aquí están **desacoplados por construcción**: la red produce $V$ y $A$, pero el algoritmo (DDQN, SARSA, prioritized replay) sigue siendo el mismo y trata la salida como un $Q$ ordinario.

## 3. Contribución central

La definición de la ventaja relaciona las tres funciones:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

con la propiedad de que $\mathbb{E}_{a\sim\pi(s)}[A^\pi(s,a)] = 0$. Intuitivamente, $V$ mide cuán bueno es estar en el estado $s$; $A$ resta ese valor del $Q$ para obtener una medida *relativa* de la importancia de cada acción.

La dueling architecture representa **ambas funciones con un solo modelo profundo** cuya salida las combina para producir $Q(s,a)$. Concretamente: un stream de capas totalmente conectadas emite un **escalar** $V(s;\theta,\beta)$, y el otro stream emite un **vector** $A(s,a;\theta,\alpha)$ de dimensión $|A|$. Aquí $\theta$ son los parámetros del tronco convolucional compartido, y $\alpha,\beta$ los de los dos streams.

**El problema de identificabilidad y el truco de restar la media.** La tentación ingenua es combinar así:

$$Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + A(s,a;\theta,\alpha) \quad (7)$$

Pero esta ecuación es **no identificable**: dado $Q$, no se pueden recuperar $V$ y $A$ de forma única. Si se suma una constante a $V$ y se resta esa misma constante a $A$, el $Q$ resultante es idéntico. Esa libertad sobrante hace que $V$ y $A$ no converjan a estimaciones con sentido, y el desempeño práctico se degrada.

Para forzar identificabilidad, el paper propone restar el **máximo** de la ventaja:

$$Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + \Big(A(s,a;\theta,\alpha) - \max_{a'} A(s,a';\theta,\alpha)\Big) \quad (8)$$

Con esto, para la acción óptima $a^* = \arg\max_{a'} Q$ se obtiene $Q(s,a^*) = V(s)$, de modo que el stream $V$ queda anclado al valor real y el stream $A$ recupera su semántica de ventaja.

La variante que **realmente se usa** en todos los experimentos reemplaza el máximo por la **media**:

$$Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + \Big(A(s,a;\theta,\alpha) - \frac{1}{|A|}\sum_{a'} A(s,a';\theta,\alpha)\Big) \quad (9)$$

El compromiso es elegante: restar la media pierde la semántica estricta de $V$ y $A$ (quedan desplazados por una constante), pero **aumenta la estabilidad de la optimización**, porque ahora las ventajas solo necesitan cambiar tan rápido como su propia media, en vez de tener que compensar cualquier cambio en la ventaja de la acción óptima como ocurría con el máximo en (8). Un punto crucial: restar la media **no altera el ranking relativo** de los $A$ (ni de los $Q$), de modo que la política greedy o ε-greedy basada en $Q$ se preserva exactamente. Al actuar, basta evaluar el stream de ventaja para decidir.

El paper subraya que la ecuación (9) **es parte de la red, no un paso algorítmico aparte**: se implementa como una capa de agregación y el entrenamiento sigue siendo retropropagación pura. $V$ y $A$ se computan automáticamente, sin supervisión extra ni modificación del algoritmo. La red aprende **qué estados son (o no son) valiosos sin tener que aprender el efecto de cada acción en cada estado** — esto es lo que el título llama el factoring que "generaliza el aprendizaje a través de las acciones".

## 4. Método: los dos streams y el módulo de agregación

La estructura concreta de la red usada en Atari:

- **Tronco convolucional compartido** (idéntico a DQN / van Hasselt et al., 2015): 3 capas convolucionales — la primera con 32 filtros 8×8 stride 4, la segunda con 64 filtros 4×4 stride 2, la tercera con 64 filtros 3×3 stride 1.
- **Bifurcación en dos streams**, cada uno con una capa totalmente conectada de **512 unidades**. El stream de valor termina en **una sola salida** (el escalar $V$); el stream de ventaja termina en **$|A|$ salidas** (una por acción válida; en ALE las acciones van de 3 a 18). Hay rectificadores (ReLU) entre todas las capas adyacentes.
- **Módulo de agregación verde** (en la Figura 1 del paper): implementa la ecuación (9) para combinar ambos streams en el vector de Q-valores. La salida final tiene la misma forma que la de un DQN estándar, de modo que la red es *drop-in*: se pueden reciclar todos los algoritmos de Q-networks (DDQN, SARSA, prioritized replay) sin cambios.

**Detalles de entrenamiento que importan.** Como ambos streams retropropagan gradientes hacia la última capa convolucional compartida, el gradiente combinado que entra a esa capa se **reescala por $1/\sqrt{2}$**, un truco simple que mejora levemente la estabilidad. Además se aplica **gradient clipping** (norma ≤ 10), práctica común en redes recurrentes (Bengio et al., 2013) pero no estándar en RL profundo; el paper verifica que buena parte de la mejora de su línea base "Single Clip" sobre "Single" viene precisamente de este clipping, por lo que lo incorpora en todas las variantes para que la comparación sea justa.

**Saliency maps como evidencia del mecanismo.** Para mostrar que los dos streams aprenden roles distintos, los autores computan los Jacobianos de $V$ y de $A$ respecto a los píxeles de entrada (método de Simonyan et al., 2013) en Enduro. El stream de **valor** atiende al horizonte de la carretera —donde aparecen autos nuevos que afectarán el desempeño futuro— y al marcador. El stream de **ventaja** casi no atiende a la imagen cuando no hay autos enfrente (su elección de acción es irrelevante), pero se activa apenas hay un auto en curso inmediato de colisión, momento en que la acción sí importa. Es la confirmación visual de la intuición de la Sección 2.

## 5. Experimentos

**5.1. Evaluación de política (corridor environment).** Antes de Atari, el paper aísla el efecto de la arquitectura en una tarea de evaluación de política con valores $Q^\pi(s,a)$ calculables exactamente (un entorno de "corredor" de tres tramos). Comparan un MLP de un solo stream contra la versión dueling, en tres variantes con **5, 10 y 20 acciones** (las acciones extra son no-ops añadidos). Con 5 acciones, ambas arquitecturas convergen a velocidad similar. Pero **a medida que crece el número de acciones, la dueling architecture supera claramente** al single-stream, y la brecha se ensancha con más acciones. La razón: el stream $V$ aprende un valor general compartido entre muchas acciones similares en $s$, lo que acelera la convergencia. Esto predice exactamente el patrón que aparecerá en Atari.

**5.2. Atari (57 juegos).** Misma arquitectura e hiperparámetros para los 57 juegos, solo píxeles crudos y score como entrada. Entrenan la dueling network con el algoritmo **DDQN**. Para aislar la contribución arquitectónica, reentrenan también un DDQN de un solo stream con exactamente el mismo procedimiento (gradient clipping, 1024 unidades en la primera capa FC para igualar el número de parámetros): esa línea base es **Single Clip**, frente a **Single** (el modelo original de van Hasselt et al.).

Resultados principales (Tabla 1, en % de desempeño humano):

| Agente | 30 no-ops Media / Mediana | Human Starts Media / Mediana |
|---|---|---|
| Nature DQN | 227.9% / 79.1% | 219.6% / 68.5% |
| Single | 307.3% / 117.8% | 332.9% / 110.9% |
| Single Clip | 341.2% / 132.6% | 302.8% / 114.1% |
| **Duel Clip** | **373.1% / 151.5%** | **343.8% / 117.1%** |
| Prior. Single | 434.6% / 123.7% | 386.7% / 112.9% |
| **Prior. Duel Clip** | **591.9% / 172.1%** | **567.0% / 115.3%** |

Hallazgos clave:

- **Duel Clip** supera a Single Clip (de capacidad equivalente) en **75.4%** de los juegos (43/57) y a la línea base Single en **80.7%** (46/57). Alcanza desempeño a nivel humano en 42 de 57 juegos.
- **La ventaja crece con el número de acciones.** Entre los juegos con 18 acciones, Duel Clip es mejor el **86.6%** de las veces (26/30). Esto confirma directamente la predicción del experimento del corredor: cuantas más acciones, más rinde separar el valor de estado de la ventaja.
- **Robustez a human starts.** Bajo la métrica más exigente (100 puntos de inicio sampleados de un experto humano, que castiga a los agentes que solo memorizan secuencias en el Atari determinista), Duel Clip sigue ganando: mejor que Single en 70.2% (40/57) y en juegos de 18 acciones, 83.3% (25/30).
- **Combinación con prioritized replay.** Como la priorización del replay y la dueling architecture atacan aspectos *ortogonales* del aprendizaje, su combinación es prometedora — y lo confirman: **Prior. Duel Clip** establece el nuevo estado del arte en ALE, con media de 591.9% bajo 30 no-ops. (Hubo que reajustar levemente learning rate y norma de clipping porque la priorización interactúa de forma sutil con el clipping: muestrear transiciones de alto TD-error produce gradientes de mayor norma.)

## 6. Limitaciones reconocidas

El paper es modesto en autocrítica, pero los límites son legibles:

- **Mejora arquitectónica, no algorítmica.** La dueling architecture no cambia *qué* se optimiza ni cómo se exploran los estados; hereda todas las patologías del Q-learning profundo (sobreestimación —parcialmente mitigada por usar DDQN—, dependencia de un target network, sensibilidad a hiperparámetros). No resuelve la exploración: Montezuma's Revenge sigue en 0.0 para todas las variantes.
- **Beneficio condicionado al número de acciones.** Con pocas acciones (p. ej. 5 en el corredor, o juegos de 3-4 acciones), la ventaja sobre el single-stream es marginal. El método rinde donde hay muchas acciones similares; en espacios de acción pequeños el costo extra de los dos streams casi no se paga.
- **No hay garantía formal de qué representa cada stream.** Al usar la resta de la media (ec. 9 en lugar de 8), $V$ y $A$ quedan desplazados por una constante y pierden su interpretación estricta; el paper acepta este sacrificio por estabilidad, pero significa que el "valor" aprendido no es exactamente $V^\pi$.
- **Interacciones sutiles entre extensiones.** La propia combinación con prioritized replay requirió reajuste manual de hiperparámetros sobre un subconjunto de 9 juegos, señal de que apilar mejoras no es trivial.
- **Pérdidas en algunos juegos.** No es universalmente superior: hay juegos (Freeway, Breakout, Video Pinball en ciertas métricas) donde la dueling network rinde por debajo del single-stream.

## 7. Impacto

La dueling architecture se volvió uno de los **componentes canónicos del DQN moderno**. Su mayor legado es ser una de las seis mejoras integradas en **Rainbow** (Hessel et al., 2018), el agente que combinó Double Q-learning, prioritized replay, dueling networks, multi-step learning, distributional RL (C51) y noisy nets, y que demostró que estas mejoras son en gran medida complementarias y aditivas. Junto con Double DQN (van Hasselt et al., 2015) y Prioritized Experience Replay (Schaul et al., 2016), la dueling network forma el trío de mejoras "de bajo costo y alto retorno" que casi cualquier implementación seria de value-based RL incorpora por defecto.

Su valor pedagógico es notable: enseña que **el sesgo inductivo de la arquitectura puede sustituir parte del trabajo del algoritmo**. La factorización $Q = V + (A - \text{media}\,A)$ es un ejemplo limpio de cómo imponer estructura conocida del problema (que el valor de estado y la ventaja son cantidades de naturaleza distinta y de escalas distintas) directamente en la red, en vez de esperar que un único stream las descubra desde cero. Ganó el Best Paper Award de ICML 2016 precisamente por esa elegancia: una idea simple, barata de implementar, combinable con todo lo demás, y con mejoras dramáticas y bien aisladas.

## 8. Conexión con la Clase 31 (Aprendizaje Reforzado)

La Clase 31 introduce el DQN (Mnih et al., 2015) como el puente entre Q-learning tabular y el RL profundo: una red que aproxima $Q(s,a)$ desde píxeles, estabilizada por experience replay y target network. La dueling architecture es la **siguiente pieza natural** de esa historia y encaja sin fricción:

- **No cambia el algoritmo de la clase.** El bucle de DQN/DDQN —muestrear minibatch del replay buffer, calcular el target $y = r + \gamma\,Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$, dar un paso de descenso de gradiente— queda intacto. Lo único que se reemplaza es la red interna: donde antes había un MLP de un stream, ahora hay dos streams y una capa de agregación. Para el estudiante, es entender que "la red de DQN" puede mejorarse por dentro sin tocar la ecuación de Bellman.
- **Hace tangible la distinción $V$ vs $Q$ vs $A$.** La clase define $V^\pi(s)$, $Q^\pi(s,a)$ y, opcionalmente, la ventaja $A = Q - V$. La dueling architecture es donde esas tres cantidades dejan de ser notación y se vuelven *dos cabezas físicas* de una red, con la identidad $\mathbb{E}_a[A]=0$ implementada literalmente como la resta de la media. Es el laboratorio conceptual ideal para fijar esos conceptos.
- **Ilustra el rol del sesgo inductivo.** Conecta el RL con la lección transversal del curso: meter conocimiento del problema en la arquitectura (como hicieron las convnets con la invarianza traslacional) acelera el aprendizaje. Aquí el conocimiento es "en muchos estados la acción no importa, el valor sí".

Material relacionado:

- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado) — MDP, ecuación de Bellman, Q-learning, las funciones $V$, $Q$ y $A$ que esta arquitectura factoriza.
- Clase: [/clases/clase-31](/clases/clase-31) — Aprendizaje Reforzado, DQN y sus extensiones.
- Paper base: [/papers/dqn-nature-mnih-2015](/papers/dqn-nature-mnih-2015) — el DQN de un solo stream que la dueling architecture mejora (experience replay + target network).
- Extensión complementaria: [/papers/per-schaul-2015](/papers/per-schaul-2015) — Prioritized Experience Replay, con el que el paper combina la dueling network para fijar el estado del arte (Prior. Duel Clip) y que, junto a esta, forma parte de Rainbow.
