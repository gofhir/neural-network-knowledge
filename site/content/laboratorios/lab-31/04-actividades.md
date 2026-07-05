---
title: "Tarea: las dos preguntas"
weight: 4
---

El notebook cierra con dos preguntas conceptuales. Aquí van resueltas con el respaldo teórico de las páginas anteriores y, donde se puede, **validadas empíricamente** en [Experimentos](03-experimentos-y-analisis).

## Pregunta 1 — ¿De dónde salen los datos de entrenamiento de DQN?

> DQN entrena una red neuronal, pero entrenar una red requiere un conjunto de datos. ¿De dónde salen los datos que usa DQN?

**Respuesta.** No hay un dataset externo ni etiquetas humanas: **el agente genera sus propios datos interactuando con el ambiente**. En cada paso guarda la transición $(s_t, a_t, r_{t+1}, s_{t+1}, \text{done})$ en el **experience replay buffer** (aquí una `deque` de `maxlen=2500`). Ese buffer *es* el conjunto de entrenamiento de la red.

Dos matices que lo hacen distinto de la supervisión clásica:

1. **Las etiquetas son auto-generadas por bootstrapping.** El target de cada muestra no viene de un humano, sino de la propia red vía Bellman:
   $$
   y = r + \gamma\max_{a'}Q_\theta(s',a')
   $$
   La red se entrena contra una versión de sí misma. Es lo que en RL se llama **aprendizaje por bootstrapping**: la etiqueta se construye con el modelo actual.

2. **El dataset es dinámico, correlacionado y no-i.i.d.** Crece y rota mientras el agente actúa; las transiciones consecutivas están muy correlacionadas (el estado $t+1$ casi igual al $t$). El **muestreo aleatorio** de mini-batches desde el buffer las des-correlaciona y permite aplicar SGD como si fueran i.i.d.

{{< callout type="info" >}}
**Validación empírica.** El experimento [sin replay buffer](03-experimentos-y-analisis) entrena aprendiendo solo de la última transición (sin muestreo aleatorio). El resultado confirma la teoría: sin el buffer el entrenamiento se degrada — la des-correlación que aporta el replay es lo que hace converger a DQN.
{{< /callout >}}

## Pregunta 2 — ¿Por qué el estado necesita las velocidades del carro y del bastón?

> En CartPole, ¿por qué es importante que el estado incluya las velocidades del carro y del bastón? ¿No basta con conocer sus posiciones?

**Respuesta.** Porque sin las velocidades el estado **deja de ser markoviano** y la tarea se vuelve irresoluble en principio.

La física de CartPole es de **segundo orden**: la aceleración del bastón depende de su posición *y* su velocidad. Con solo las posiciones $(x, \theta)$, dos situaciones opuestas son **indistinguibles**:

- Bastón a $+5°$ **cayéndose** hacia afuera ($\dot\theta > 0$) → emergencia, hay que corregir ya.
- Bastón a $+5°$ **volviendo** al centro ($\dot\theta < 0$) → todo bien, no toques.

Misma observación posicional, acción óptima **opuesta**. Ninguna función $Q(x,\theta)$ puede asignar el Q-value correcto a ambos casos: el mismo input tendría que dar dos outputs distintos.

Formalmente, quitar las velocidades convierte el MDP en un **POMDP** (parcialmente observable), donde Q-learning pierde sus garantías de convergencia. Es exactamente el problema que el DQN de Atari resuelve **apilando 4 frames** consecutivos: al ver varias posiciones seguidas, la red puede *inferir* las velocidades por diferencias. Aquí Gym nos entrega las velocidades directamente en el estado, así que la propiedad de Markov se cumple y una red sin memoria basta.

{{< callout type="info" >}}
**Validación empírica (el experimento estrella).** En [Experimentos](03-experimentos-y-analisis) enmascaramos las dos dimensiones de velocidad del estado (dejando solo posiciones) y reentrenamos el mismo DQN. El resultado mide directamente el costo de romper Markov: la política se degrada drásticamente respecto al baseline que sí ve las velocidades. Deja de ser un argumento teórico y pasa a ser un número.
{{< /callout >}}

## En una frase cada una

- **P1:** Los datos salen del propio agente: transiciones $(s,a,r,s')$ guardadas en el replay buffer; las etiquetas se auto-generan por bootstrapping de Bellman.
- **P2:** Las velocidades son las que hacen que $s$ sea un **estado markoviano**; sin ellas la posición es una observación ambigua y $q_*$ no está bien definida como función del estado.

---

**Siguiente:** [Experimentos propios y análisis](03-experimentos-y-analisis) — las ablations que convierten estas dos respuestas en datos medidos.
