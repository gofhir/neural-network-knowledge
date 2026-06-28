# Deep Reinforcement Learning with Double Q-learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Deep Reinforcement Learning with Double Q-learning*.
- **Autores:** Hado van Hasselt, Arthur Guez, David Silver (Google DeepMind).
- **Venue:** AAAI 2016 (Thirtieth AAAI Conference on Artificial Intelligence). Copyright AAAI, 2016.
- **Año:** 2015 (preprint). **Preprint:** arXiv:1509.06461v3 (8 dic 2015), [arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461).
- **Linaje:** generaliza *Double Q-learning* (van Hasselt, NeurIPS 2010), originalmente formulado para el caso tabular, al régimen de aproximación de funciones con redes neuronales profundas, tomando como base el algoritmo **DQN** (Mnih et al., Nature 2015).

Este es un paper que resuelve un problema concreto y bien delimitado: el algoritmo **Q-learning** —y por extensión **DQN**— **sobreestima sistemáticamente los valores de acción**. El paper hace cuatro afirmaciones que demuestra una por una: (1) muestra *por qué* Q-learning puede ser sobreoptimista incluso en problemas grandes y deterministas, debido a los errores de estimación inherentes al aprendizaje; (2) muestra, midiendo los value estimates en juegos de Atari, que estas sobreestimaciones son **más comunes y severas en la práctica** de lo que se reconocía; (3) muestra que la idea de Double Q-learning puede usarse *a escala* para reducir ese sesgo, produciendo aprendizaje más estable; y (4) propone una implementación específica —**Double DQN**— que reutiliza la arquitectura y la red ya existentes en DQN, sin redes ni parámetros adicionales, y obtiene resultados estado-del-arte en el dominio Atari 2600.

El punto que lo hace memorable para la enseñanza es la **economía de la solución**: el sesgo de sobreestimación, que parece un defecto profundo del aprendizaje por refuerzo basado en valores, se corrige con un cambio de **prácticamente una línea** en el cálculo del *target* de DQN. No se introduce ninguna red nueva, ningún hiperparámetro nuevo, ningún costo computacional apreciable. Esa relación entre el tamaño del cambio y el tamaño de la mejora es lo que convirtió a Double DQN en componente estándar.

Para la **Clase 31 (Aprendizaje Reforzado)** esto importa porque el curso enseña DQN como el puente entre Q-learning tabular y deep RL; Double DQN es la primera y más limpia de las mejoras que se le aplican, y la que mejor ilustra que el aprendizaje por refuerzo no es solo "poner una red neuronal" sino entender los sesgos estadísticos que la red amplifica.

## 2. Contexto: el operador max y el sesgo de sobreoptimismo

El objetivo del aprendizaje por refuerzo (Sutton y Barto, 1998) es aprender buenas políticas para problemas de decisión secuencial, optimizando una señal de recompensa futura acumulada. **Q-learning** (Watkins, 1989) es uno de los algoritmos más populares para esto: aprende estimaciones del valor óptimo de cada acción, definido como la suma esperada de recompensas futuras al tomar esa acción y luego seguir la política óptima. En problemas grandes no se puede tabular un valor por cada par estado-acción, así que se aprende una función de valor parametrizada $Q(s,a;\theta_t)$, y la actualización estándar empuja $Q(S_t, A_t; \theta_t)$ hacia un *target*:

$$Y_t^{Q} \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t).$$

Aquí está el problema. Q-learning "a veces aprende valores de acción irrealistamente altos porque incluye un paso de maximización sobre valores de acción estimados, que tiende a preferir valores sobreestimados a subestimados". La causa raíz, que el paper expone con claridad quirúrgica, es que **el mismo conjunto de valores se usa tanto para SELECCIONAR la acción como para EVALUARla**. El operador $\max_a Q(S_{t+1}, a; \theta_t)$ hace dos cosas simultáneamente: elige cuál acción es la mejor (el `argmax`) y reporta el valor de esa elección (el valor en ese `argmax`). Cuando las estimaciones tienen ruido —y *siempre* lo tienen durante el aprendizaje, porque los valores verdaderos son inicialmente desconocidos— el `max` se inclina hacia las acciones cuyo ruido las hace lucir mejor de lo que son. El resultado es un sesgo hacia arriba: **overoptimism bias**.

Trabajos previos atribuían las sobreestimaciones a aproximación de funciones insuficientemente flexible (Thrun y Schwartz, 1993) o a ruido del ambiente (van Hasselt, 2010, 2011). La contribución conceptual del paper es **unificar esas explicaciones**: las sobreestimaciones pueden ocurrir cuando los valores de acción son inexactos *por cualquier causa* —ruido, aproximación, no-estacionariedad— con independencia de la fuente del error de aproximación. Dado que las estimaciones imprecisas son la norma durante el aprendizaje, las sobreestimaciones son mucho más comunes de lo que se apreciaba.

El paper hace una distinción importante para no confundir conceptos: este sobreoptimismo **no es lo mismo que el "optimismo frente a la incertidumbre"** (optimism in the face of uncertainty), una técnica de exploración deseable en la que se da un bono a estados o acciones de valor incierto. Al contrario, las sobreestimaciones de Q-learning aparecen *después* de actualizar, produciendo "optimismo frente a la certeza aparente", y —como ya notaba Thrun y Schwartz (1993)— pueden **impedir** el aprendizaje de una política óptima.

### 2.1. Por qué el sesgo es real, no un artefacto

El paper no se queda en la intuición; lo prueba. El **Teorema 1** muestra que en un estado donde todos los valores óptimos verdaderos son iguales a $V_*(s)$, si las estimaciones $Q_t$ son en promedio insesgadas pero no todas correctas (con error cuadrático medio $C > 0$ sobre $m$ acciones), entonces

$$\max_a Q_t(s,a) \ge V_*(s) + \sqrt{\frac{C}{m-1}},$$

y esta cota inferior es justa (*tight*). Bajo las mismas condiciones, la cota inferior del error absoluto del estimador de Double Q-learning es **cero**. Es decir: aun cuando las estimaciones sean correctas en promedio, errores de estimación de cualquier fuente empujan el `max` hacia arriba, mientras que el desacoplamiento de Double Q-learning permite que el sesgo se anule. Notablemente, el teorema **no asume independencia** entre los errores de las distintas acciones.

El paper refuerza esto con un experimento determinista (Figura 2): ajusta polinomios de grado 6 o 9 a funciones de valor verdaderas conocidas ($\sin(s)$, $2\exp(-s^2)$) sobre 10 acciones, y muestra que el `max` de las estimaciones queda casi siempre por encima del valor verdadero, mientras que el estimador estilo Double Q-learning (usando un segundo conjunto de muestras) queda mucho más cerca de cero. Crucialmente, las tres filas del experimento demuestran que el sesgo aparece tanto con aproximación poco flexible (errores asintóticos irreducibles) como con aproximación *suficientemente flexible* para ajustar todos los puntos —contradiciendo la idea de Thrun y Schwartz de que el problema era solo de flexibilidad. La sobreestimación combinada con *bootstrapping* tiene además el efecto pernicioso de **propagar** la información relativa equivocada sobre qué estados valen más, contaminando toda la función de valor.

## 3. Contribución central

La contribución es desacoplar la **selección** de la acción de su **evaluación** en el cálculo del *target*, reutilizando una pieza que DQN **ya tiene**: la red objetivo (*target network*).

Para ver de dónde sale la idea, conviene reescribir el `max` de Q-learning de forma equivalente, separando explícitamente el `argmax` (selección) del valor (evaluación):

$$Y_t^{Q} = R_{t+1} + \gamma\, Q\big(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t\big).$$

En esta forma se ve que **los mismos pesos $\theta_t$ aparecen dos veces**: una para elegir la acción greedy, otra para evaluarla. Ese acoplamiento es exactamente lo que produce el sesgo. **Double Q-learning** (van Hasselt, 2010), en su forma original, aprende *dos* funciones de valor con pesos $\theta$ y $\theta'$, asignando cada experiencia al azar para actualizar una u otra, y desacopla así:

$$Y_t^{\text{DoubleQ}} \equiv R_{t+1} + \gamma\, Q\big(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta'_t\big).$$

La selección sigue usando los pesos online $\theta_t$ (estimamos la política greedy según los valores actuales), pero la **evaluación** del valor de esa política usa el segundo conjunto de pesos $\theta'_t$.

La observación clave del paper es que **no hace falta introducir una segunda red para tener ese segundo conjunto de pesos**: la arquitectura de DQN ya incluye una *target network* con parámetros $\theta_t^-$, que es una copia periódica de la red online (copiada cada $\tau$ pasos y mantenida fija entre medio). Aunque no está totalmente desacoplada de la red online, "la target network de la arquitectura DQN provee un candidato natural para la segunda función de valor, sin tener que introducir redes adicionales". Esto convierte una idea elegante pero costosa (dos redes) en un cambio gratuito.

## 4. Método: el target de Double DQN frente al de DQN

DQN usa la target network *solo* para evaluar, manteniendo el `max` acoplado:

$$Y_t^{\text{DQN}} \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t^-).$$

Double DQN deja **todo lo demás de DQN intacto** y reemplaza únicamente el *target*. La acción se **selecciona** con la red online $\theta_t$ y se **evalúa** con la target network $\theta_t^-$:

$$\boxed{\,Y_t^{\text{DoubleDQN}} \equiv R_{t+1} + \gamma\, Q\big(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-\big)\,}$$

Comparado con la fórmula original de Double Q-learning, los pesos $\theta'_t$ de la segunda red se reemplazan simplemente por los pesos $\theta_t^-$ de la target network. La actualización de la target network no cambia respecto a DQN: sigue siendo una copia periódica de la red online. En palabras del paper, "esta versión de Double DQN es quizás el cambio mínimo posible de DQN hacia Double Q-learning". El objetivo de diseño es explícito: capturar la mayor parte del beneficio de Double Q-learning manteniendo el resto del algoritmo DQN sin tocar, para una comparación justa y con sobrecarga computacional mínima.

Conviene notar la diferencia operativa frente a DQN: donde DQN escribe `max(target_net(s'))`, Double DQN escribe `target_net(s')[ argmax(online_net(s')) ]`. La selección de la acción migra de la target network a la red online; la evaluación se queda en la target network. En código real esto es, literalmente, mover el `argmax` de una red a la otra: el "cambio de una línea" del que habla la comunidad.

## 5. Experimentos: Atari 2600

El banco de pruebas es el **Arcade Learning Environment** (Bellemare et al., 2013) sobre juegos de Atari 2600, siguiendo de cerca el montaje experimental y la arquitectura de red de Mnih et al. (2015): una CNN con 3 capas convolucionales y una capa oculta totalmente conectada (~1.5M parámetros), que toma los últimos cuatro frames como entrada y emite el valor de cada acción. Un único algoritmo con hiperparámetros fijos debe aprender a jugar cada juego solo a partir de los píxeles de pantalla, entrenando ~200M frames por juego (~1 semana en una GPU). Es un testbed exigente: las entradas son de alta dimensión y la mecánica varía mucho entre juegos, de modo que las buenas soluciones dependen del algoritmo y no del *tuning* específico.

**Menos sobreestimación.** La Figura 3 mide los *value estimates* durante el entrenamiento (promediando $\max_a Q(S_t, a; \theta)$ sobre fases de evaluación de $T=125{,}000$ pasos) y los compara con el valor descontado *real* de la mejor política aprendida. En DQN, las curvas de valor terminan **mucho más altas** que los valores verdaderos —DQN es "consistente y a veces enormemente sobreoptimista". En juegos como **Asterix** y **Wizard of Wor** la sobreestimación es extrema (nótese la escala logarítmica) e inestable: el momento en que los value estimates de DQN se disparan **coincide** con la caída de su puntaje. Double DQN produce estimaciones mucho más cercanas al valor real y un aprendizaje notablemente más estable. Las sobreestimaciones se observaron para DQN en **los 49 juegos** probados, en grados variables.

**Mejor puntaje.** Para resumir a través de juegos, los puntajes se normalizan respecto a un agente aleatorio y a un jugador humano. Bajo la condición *no-ops* (5 min de juego, 49 juegos, Tabla 1), Double DQN mejora claramente sobre DQN:

| | DQN | Double DQN |
|---|---|---|
| Mediana | 93.5% | 114.7% |
| Media | 241.1% | 330.3% |

Bajo la condición más exigente de *human starts* (30 min, arranques desde trayectorias de expertos humanos, Tabla 2), la brecha se mantiene y crece con una versión afinada (*tuned*):

| | DQN | Double DQN | Double DQN (tuned) |
|---|---|---|---|
| Mediana | 47.5% | 88.4% | 116.7% |
| Media | 122.0% | 273.1% | 475.2% |

La única diferencia entre DQN y Double DQN en estas evaluaciones es el *target* ($Y_t^{\text{DoubleDQN}}$ en vez de $Y^{\text{DQN}}$), usando los **mismos hiperparámetros** afinados para DQN —una comparación deliberadamente adversa para Double DQN. La versión *tuned* introduce ajustes menores (más frames entre copias de la target network, de 10.000 a 30.000, para reducir aún más la sobreestimación; menor exploración; un sesgo compartido en la capa final). Ejemplos de mejora notable en los datos crudos: **Road Runner** (de 233% a 617%), **Asterix** (de 70% a 180%), **Zaxxon** (de 54% a 111%) y **Double Dunk** (de 17% a 397%). La robustez frente a *human starts* sugiere que las soluciones de Double DQN **generalizan** y no explotan el determinismo de los ambientes memorizando secuencias de acciones.

El mensaje empírico es doble: Double DQN no solo produce estimaciones de valor **más precisas**, sino también **mejores políticas** —la línea recta azul (valor real de la política final de Double DQN) suele estar por encima de la naranja (la de DQN), confirmando que reducir el sesgo de sobreestimación mejora la calidad de la política, no solo la honestidad del número.

## 6. Limitaciones

- **Desacoplamiento solo parcial.** El propio paper reconoce que en Double DQN la selección y la evaluación "no están totalmente desacopladas" (*not fully decoupled*), porque la target network es una copia *retardada* de la red online, no una red entrenada independientemente como en el Double Q-learning original. Es un compromiso pragmático: se gana casi todo el beneficio a costo cero, pero el desacoplamiento es imperfecto.
- **No elimina, atenúa.** Double DQN *reduce* la sobreestimación; no garantiza estimaciones insesgadas en el régimen profundo. La versión *tuned* todavía necesita aumentar el periodo de la target network para reducir el sesgo "aún más", lo que evidencia que el sesgo residual persiste.
- **Costo del régimen Atari.** Cada juego requiere ~1 semana de GPU y 200M frames; el estudio hereda el costo experimental de DQN.
- **Hiperparámetros heredados.** La comparación principal usa hiperparámetros afinados para DQN, lo que es honesto y conservador, pero implica que el potencial pleno de Double DQN podría estar subestimado en la versión no-*tuned*.
- **Alcance del análisis teórico.** El Teorema 1 caracteriza el sesgo en estados con valores óptimos iguales y bajo supuestos específicos; es una demostración de existencia y de cota, no una caracterización completa del sesgo en redes profundas reales con bootstrapping correlacionado.

## 7. Impacto y adopción

Double DQN se convirtió en un **componente estándar** del deep RL basado en valores. Su atractivo —máximo beneficio por mínimo cambio— lo volvió una mejora "por defecto" que casi cualquier implementación de DQN incorpora. La contribución más perdurable, más allá del algoritmo, es haber instalado en la comunidad la conciencia de que **el sesgo de sobreestimación es un problema real y medible** en deep RL, no una curiosidad tabular.

Su lugar canónico es como uno de los **seis ingredientes de Rainbow** (Hessel et al., 2018), el agente que combina Double DQN, *prioritized experience replay* (Schaul et al., 2015), *dueling networks* (Wang et al., 2015), aprendizaje multi-paso, RL distribucional y *noisy nets*, demostrando que estas mejoras son en gran medida **complementarias**. Double DQN y Dueling DQN, en particular, son ortogonales: una corrige *cómo se evalúa* el `max` (el sesgo), la otra reorganiza *cómo la red representa* el valor (separando $V(s)$ y la ventaja $A(s,a)$); combinarlas suma.

## 8. Conexión con la Clase 31 (Aprendizaje Reforzado)

La Clase 31 enseña la transición de Q-learning tabular a **DQN** (ver [/papers/dqn-nature-mnih-2015](/papers/dqn-nature-mnih-2015)): la red neuronal como aproximador de $Q(s,a)$, la *experience replay* y la *target network* como los dos ingredientes que estabilizan el entrenamiento. Double DQN es la **mejora directa más natural** sobre ese DQN, y por eso es la primera parada después de enseñarlo:

- **Reutiliza lo que la clase ya explicó.** El estudiante ya entendió por qué DQN necesita una target network (estabilizar el blanco del *bootstrapping*). Double DQN le da a esa misma red un *segundo propósito* —evaluar la acción que la red online selecciona— sin agregar nada. Es la mejor demostración de que entender bien una arquitectura abre mejoras gratuitas.
- **Hace tangible un sesgo estadístico.** La clase puede mostrar el `argmax` migrando de la target network a la red online como un cambio de una línea, y conectar ese cambio con la Figura 3 (curvas de valor de DQN despegándose de la verdad). Es un puente perfecto entre teoría (el `max` sesga) y práctica (los juegos donde el puntaje se desploma cuando el valor se infla).
- **Prepara el terreno para las demás mejoras.** Una vez visto Double DQN, el camino sigue hacia [Dueling DQN](/papers/dueling-dqn-wang-2015) (arquitectura $V + A$), *prioritized replay* y, finalmente, su integración en Rainbow. Double DQN enseña que las mejoras de deep RL suelen ser quirúrgicas y combinables.
- **Fundamento transversal.** El sesgo de sobreestimación —el `max` de estimaciones ruidosas siempre tira hacia arriba— es un fenómeno general de los estimadores por refuerzo, no un detalle de Atari. Forma parte del cuerpo conceptual de [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado) y del recorrido completo de [/clases/clase-31](/clases/clase-31).
