# Respuestas — Tarea Tutorial DQN (CartPole)

> Respaldadas por la ejecución real del Colab (`Tutorial_DQN_DINTA_v18_1.ipynb`): DQN resolvió CartPole en **85 episodios** y obtuvo **210/210 en test** (ε=0). Las validaciones empíricas provienen de ablations propias sobre el mismo DQN.

---

## Pregunta incrustada (celda 17) — ¿Por qué el estado incluye las velocidades? ¿No basta con las posiciones?

Sin las velocidades el estado deja de ser **markoviano**. La física de CartPole es de **segundo orden**: la aceleración del bastón depende de su posición *y* su velocidad. Con solo las posiciones $(x, \theta)$, dos situaciones opuestas son **indistinguibles** — el bastón a $+5°$ *cayéndose* ($\dot\theta > 0$) versus *volviendo al centro* ($\dot\theta < 0$): misma observación, acción óptima contraria. Ninguna función $Q(x, \theta)$ puede acertar a ambos casos. Por eso no basta con las posiciones: se necesitan las velocidades para que el estado prediga el futuro.

---

## Tarea 1 — ¿De dónde salen los datos que utiliza DQN para entrenar su red?

**El agente genera sus propios datos** interactuando con el ambiente. En cada paso guarda la transición $(s_t, a_t, r_{t+1}, s_{t+1}, \text{done})$ en el **experience replay buffer** (en el código, una `deque` de `maxlen=2500`). Ese buffer *es* el conjunto de entrenamiento de la red.

Dos matices que lo distinguen de la supervisión clásica:

1. **Las etiquetas son auto-generadas por bootstrapping.** El target de cada muestra no lo pone un humano, sino la propia red vía la ecuación de Bellman:

   $$y = r + \gamma \max_{a'} Q_\theta(s', a')$$

   La red se entrena contra una versión de sí misma.

2. **El dataset es dinámico y no i.i.d.** Crece y rota mientras el agente actúa; las transiciones consecutivas están muy correlacionadas. El **muestreo aleatorio** de mini-batches desde el buffer las des-correlaciona y permite aplicar SGD como si fueran independientes.

> **Validación empírica:** al entrenar *sin* replay buffer (aprendiendo solo de la última transición), DQN colapsa a **9.4** de recompensa — **peor que el azar (21)**. La des-correlación del buffer no es un detalle de eficiencia: es lo que hace que DQN converja.

---

## Tarea 2 — En CartPole, ¿por qué es importante que el estado incluya las velocidades del auto y del bastón?

Porque sin ellas el estado deja de ser **markoviano** y la tarea se vuelve irresoluble en principio. La física es de segundo orden: con solo $(x, \theta)$, el bastón a $+5°$ *cayéndose* ($\dot\theta > 0$) y a $+5°$ *volviendo* ($\dot\theta < 0$) dan la misma observación pero exigen acciones opuestas; ninguna $Q(x, \theta)$ puede asignar el valor correcto a ambos. Formalmente, quitar las velocidades convierte el MDP en un **POMDP** (parcialmente observable), donde Q-learning pierde sus garantías de convergencia. El DQN de Atari resuelve esto **apilando 4 frames** consecutivos para inferir las velocidades por diferencias de posición; aquí Gym nos las entrega directamente en el estado, así que una red sin memoria basta.

> **Validación empírica:** al enmascarar las dos dimensiones de velocidad (dejando solo posiciones) y reentrenar el mismo DQN, la política cae de **~210 a ~46** y ya no resuelve la tarea — el costo medido de romper la propiedad de Markov.

---

### En una frase cada una

- **Incrustada / Tarea 2:** las velocidades hacen que $s$ sea un estado markoviano; sin ellas la posición es una observación ambigua y $q_*$ no está bien definida como función del estado.
- **Tarea 1:** los datos salen del propio agente — transiciones $(s, a, r, s')$ guardadas en el replay buffer, con etiquetas auto-generadas por bootstrapping de Bellman.
