---
title: "Double DQN (2015)"
weight: 350
math: true
---

{{< paper-card
    title="Deep Reinforcement Learning with Double Q-learning"
    authors="Hado van Hasselt, Arthur Guez, David Silver"
    year="2015"
    venue="AAAI 2016"
    pdf="/papers/double-dqn-van-hasselt-2015.pdf"
    arxiv="1509.06461" >}}
Paper de Google DeepMind que diagnostica y corrige un defecto sistemático de DQN: el operador `max` del *target* **sobreestima** los valores de acción porque el mismo conjunto de pesos **selecciona Y evalúa** la acción. **Double DQN** desacopla ambas funciones —la red online elige la acción, la *target network* (que DQN ya tiene) la evalúa— en un **cambio de prácticamente una línea**, sin redes ni hiperparámetros nuevos. Resultado: estimaciones de valor más honestas, aprendizaje más estable y mejores puntajes en Atari 2600. Se convirtió en componente estándar del deep RL y en uno de los seis ingredientes de **Rainbow**.
{{< /paper-card >}}

---

## Contexto: el operador `max` sobreoptimista

El objetivo del [aprendizaje por refuerzo](/fundamentos/aprendizaje-reforzado) es aprender políticas para problemas de decisión secuencial maximizando la recompensa futura acumulada. **Q-learning** (Watkins, 1989) aprende estimaciones del valor óptimo de cada acción. En problemas grandes no se puede tabular un valor por cada par estado-acción, así que se aprende una función parametrizada $Q(s,a;\theta_t)$ y la actualización empuja $Q(S_t, A_t; \theta_t)$ hacia un *target*:

$$Y_t^{Q} \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t).$$

Aquí está el problema. Q-learning "a veces aprende valores de acción irrealistamente altos porque incluye un paso de maximización sobre valores estimados, que tiende a preferir valores sobreestimados a subestimados". La causa raíz que el paper expone con claridad quirúrgica: **el mismo conjunto de valores se usa tanto para SELECCIONAR la acción como para EVALUARla**. El operador `max` hace dos cosas a la vez —elige cuál acción es la mejor (el `argmax`) y reporta el valor de esa elección—. Cuando las estimaciones tienen ruido —y *siempre* lo tienen durante el aprendizaje— el `max` se inclina hacia las acciones cuyo ruido las hace lucir mejor de lo que son. El resultado es un sesgo hacia arriba: **overoptimism bias**.

La contribución conceptual es **unificar** explicaciones previas (aproximación poco flexible, ruido del ambiente): las sobreestimaciones aparecen cuando los valores son inexactos *por cualquier causa*. Como las estimaciones imprecisas son la norma durante el aprendizaje, las sobreestimaciones son mucho más comunes de lo que se reconocía. El paper aclara que este sobreoptimismo **no es** el deseable "optimismo frente a la incertidumbre" usado para exploración: es "optimismo frente a la certeza aparente" que puede **impedir** aprender la política óptima.

### Por qué el sesgo es real, no un artefacto

El **Teorema 1** lo prueba: en un estado donde todos los valores óptimos verdaderos son iguales a $V_*(s)$, si las estimaciones son en promedio insesgadas pero no todas correctas (error cuadrático medio $C > 0$ sobre $m$ acciones), entonces

$$\max_a Q_t(s,a) \ge V_*(s) + \sqrt{\frac{C}{m-1}},$$

y la cota es justa (*tight*). Bajo las mismas condiciones, la cota inferior del error de Double Q-learning es **cero**. Notablemente, el teorema **no asume independencia** entre los errores de las distintas acciones. Un experimento determinista (ajustando polinomios a funciones de valor conocidas) confirma que el `max` queda casi siempre por encima de la verdad, incluso con aproximación *suficientemente flexible* para ajustar todos los puntos —contradiciendo la idea de que el problema era solo de flexibilidad—. Peor aún: combinada con *bootstrapping*, la sobreestimación **propaga** información equivocada sobre qué estados valen más, contaminando toda la función de valor.

## Contribución central: desacoplar selección de evaluación

Conviene reescribir el `max` separando explícitamente el `argmax` (selección) del valor (evaluación):

$$Y_t^{Q} = R_{t+1} + \gamma\, Q\big(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t\big).$$

Así se ve que **los mismos pesos $\theta_t$ aparecen dos veces**: una para elegir la acción greedy, otra para evaluarla. Ese acoplamiento es lo que produce el sesgo. **Double Q-learning** (van Hasselt, 2010), en su forma original, aprende *dos* funciones de valor con pesos $\theta$ y $\theta'$ y desacopla:

$$Y_t^{\text{DoubleQ}} \equiv R_{t+1} + \gamma\, Q\big(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta'_t\big).$$

La selección usa los pesos online $\theta_t$; la **evaluación** usa el segundo conjunto $\theta'_t$.

La observación clave del paper: **no hace falta una segunda red para tener ese segundo conjunto de pesos**. La arquitectura de DQN ya incluye una *target network* con parámetros $\theta_t^-$, una copia periódica de la red online mantenida fija entre actualizaciones. "La target network de la arquitectura DQN provee un candidato natural para la segunda función de valor, sin tener que introducir redes adicionales". Esto convierte una idea elegante pero costosa (dos redes) en un cambio gratuito.

## Método: el target de Double DQN frente al de DQN

DQN usa la target network *solo* para evaluar, manteniendo el `max` acoplado:

$$Y_t^{\text{DQN}} \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t^-).$$

Double DQN deja **todo lo demás de DQN intacto** y reemplaza únicamente el *target*. La acción se **selecciona** con la red online $\theta_t$ y se **evalúa** con la target network $\theta_t^-$:

$$Y_t^{\text{DoubleDQN}} \equiv R_{t+1} + \gamma\, Q\big(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-\big).$$

Comparado con la fórmula original de Double Q-learning, los pesos $\theta'_t$ se reemplazan por los pesos $\theta_t^-$ de la target network, que se sigue actualizando como en DQN (copia periódica). En palabras del paper, "esta versión de Double DQN es quizás el cambio mínimo posible de DQN hacia Double Q-learning".

En código real la diferencia es literal: donde DQN escribe `max(target_net(s'))`, Double DQN escribe `target_net(s')[ argmax(online_net(s')) ]`. La selección de la acción migra de la target network a la red online; la evaluación se queda en la target network. Ese es el "cambio de una línea" del que habla la comunidad.

## Experimentos: Atari 2600

El banco de pruebas es el **Arcade Learning Environment** sobre juegos de Atari 2600, replicando el montaje de Mnih et al. (2015): una CNN con 3 capas convolucionales y una capa oculta densa (~1.5M parámetros) que toma los últimos cuatro frames y emite el valor de cada acción. Un único algoritmo con hiperparámetros fijos aprende cada juego solo desde los píxeles, entrenando ~200M frames (~1 semana de GPU por juego).

**Menos sobreestimación.** Midiendo los *value estimates* durante el entrenamiento y comparándolos con el valor descontado *real* de la mejor política aprendida, DQN resulta "consistente y a veces enormemente sobreoptimista". En juegos como **Asterix** y **Wizard of Wor** la sobreestimación es extrema (escala logarítmica) e inestable: el momento en que los valores de DQN se disparan **coincide** con la caída de su puntaje. Double DQN produce estimaciones mucho más cercanas a la verdad y aprendizaje notablemente más estable. Las sobreestimaciones se observaron en **los 49 juegos** probados.

**Mejor puntaje.** Puntajes normalizados respecto a un agente aleatorio y un jugador humano:

| Condición *no-ops* (49 juegos) | DQN | Double DQN |
|---|---|---|
| Mediana | 93.5% | 114.7% |
| Media | 241.1% | 330.3% |

Bajo la condición más exigente de *human starts* (arranques desde trayectorias de expertos), la brecha crece con una versión afinada (*tuned*):

| Condición *human starts* | DQN | Double DQN | Double DQN (tuned) |
|---|---|---|---|
| Mediana | 47.5% | 88.4% | 116.7% |
| Media | 122.0% | 273.1% | 475.2% |

La única diferencia entre DQN y Double DQN es el *target*, usando los **mismos hiperparámetros** afinados para DQN —una comparación deliberadamente adversa para Double DQN—. La versión *tuned* introduce ajustes menores (periodo de la target network de 10.000 a 30.000 frames, entre otros). La robustez frente a *human starts* sugiere que las soluciones de Double DQN **generalizan** y no memorizan secuencias de acciones explotando el determinismo. El mensaje empírico es doble: Double DQN entrega estimaciones de valor **más precisas** *y* **mejores políticas**.

## Limitaciones

- **Desacoplamiento solo parcial.** La selección y la evaluación "no están totalmente desacopladas", porque la target network es una copia *retardada* de la red online, no una red entrenada de forma independiente. Es un compromiso pragmático: casi todo el beneficio a costo cero, pero desacoplamiento imperfecto.
- **No elimina, atenúa.** Double DQN *reduce* la sobreestimación; no garantiza estimaciones insesgadas. La versión *tuned* todavía aumenta el periodo de la target network para reducir el sesgo "aún más".
- **Costo del régimen Atari.** ~1 semana de GPU y 200M frames por juego; hereda el costo experimental de DQN.
- **Alcance teórico.** El Teorema 1 caracteriza el sesgo bajo supuestos específicos; es una demostración de existencia y de cota, no una caracterización completa del sesgo en redes profundas reales.

## Impacto y conexión con la Clase 31

Double DQN se volvió un **componente estándar** del deep RL basado en valores: máximo beneficio por mínimo cambio. Su contribución más perdurable, más allá del algoritmo, fue instalar en la comunidad la conciencia de que **el sesgo de sobreestimación es un problema real y medible** en deep RL, no una curiosidad tabular. Su lugar canónico es como uno de los **seis ingredientes de Rainbow** (Hessel et al., 2018), junto con prioritized replay, [Dueling DQN](/papers/dueling-dqn-wang-2015), aprendizaje multi-paso, RL distribucional y noisy nets —mejoras en gran medida complementarias—.

La [Clase 31](/clases/clase-31) enseña la transición de Q-learning tabular a [DQN](/papers/dqn-nature-mnih-2015): la red como aproximador de $Q(s,a)$, la *experience replay* y la *target network* como los dos ingredientes que estabilizan el entrenamiento. Double DQN es la **mejora directa más natural** sobre ese DQN, y por eso es la primera parada después de enseñarlo:

- **Reutiliza lo que la clase ya explicó.** El estudiante ya entendió por qué DQN necesita una target network; Double DQN le da a esa misma red un *segundo propósito* —evaluar la acción que la red online selecciona— sin agregar nada.
- **Hace tangible un sesgo estadístico.** Mostrar el `argmax` migrando de la target network a la red online conecta teoría (el `max` sesga) con práctica (los juegos donde el puntaje se desploma cuando el valor se infla).
- **Prepara el terreno.** El camino sigue hacia [Dueling DQN](/papers/dueling-dqn-wang-2015) (arquitectura $V + A$), prioritized replay y Rainbow. Double DQN y Dueling DQN son ortogonales: una corrige *cómo se evalúa* el `max`, la otra *cómo la red representa* el valor.
- **Fundamento transversal.** El sesgo de sobreestimación —el `max` de estimaciones ruidosas siempre tira hacia arriba— es un fenómeno general de los estimadores por refuerzo, parte del cuerpo conceptual de [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado).
