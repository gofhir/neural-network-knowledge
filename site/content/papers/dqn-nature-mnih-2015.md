---
title: "DQN: Human-level control (Nature 2015)"
weight: 349
math: true
---

{{< paper-card
    title="Human-level control through deep reinforcement learning"
    authors="Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al."
    year="2015"
    venue="Nature 2015"
    pdf="/papers/dqn-nature-mnih-2015.pdf" >}}
La versión madura y definitiva del **Deep Q-Network (DQN)**, publicada en portada de *Nature*. Un único agente aprende a jugar **49 juegos de Atari 2600** directamente desde los píxeles crudos y el puntaje, con *la misma* arquitectura, hiperparámetros y algoritmo, alcanzando o superando el nivel de un *tester* humano profesional. La contribución central frente al preprint de 2013 es la **red objetivo (target network)**: una copia congelada de la red $Q$ que se actualiza solo cada $C$ pasos para estabilizar el objetivo de TD. Junto con **experience replay** y **reward clipping**, resuelve la inestabilidad que impedía entrenar Q-learning con redes neuronales profundas. Las ablaciones demuestran que tanto el replay como la red objetivo son cruciales. Hito fundacional del deep reinforcement learning.
{{< /paper-card >}}

---

## Contexto

El aprendizaje reforzado tiene raíces profundas en la psicología (Thorndike, 1911) y la neurociencia —las señales fásicas de las neuronas dopaminérgicas se parecen notablemente a los algoritmos de *temporal difference* (Schultz, Dayan & Montague, 1997). Pero hasta 2013, los agentes de RL exitosos estaban confinados a dominios con características diseñadas a mano (*handcrafted features*) o espacios de estado de baja dimensión: TD-Gammon (Tesauro, 1995) en backgammon fue un éxito aislado; el RL para fútbol robótico dependía de representaciones cuidadosamente elaboradas.

El **DQN de 2013** (Mnih et al., NIPS Deep Learning Workshop) introdujo la idea de combinar Q-learning con una red convolucional profunda y *experience replay*, evaluada sobre 7 juegos. Fue un resultado llamativo pero preliminar: sin red objetivo, con pocos juegos y sin evaluación rigurosa contra humanos. Esta versión de **Nature 2015** es la forma madura de ese trabajo. Las diferencias clave respecto a 2013:

- **La red objetivo (target network):** el aporte estabilizador que el preprint *no* tenía. Es la novedad metodológica central.
- **Recorte del error (error clipping)** además del recorte de recompensas.
- **Evaluación a gran escala:** 49 juegos (no 7), comparación cuantitativa contra un *tester* humano profesional y contra el mejor aproximador lineal, análisis t-SNE de las representaciones y ablaciones que aíslan cada componente.

El paper se publicó como artículo destacado de *Nature* —con repercusión de portada— porque por primera vez un único sistema de propósito general, recibiendo solo píxeles y puntaje, dominaba un abanico amplio y variado de tareas.

## El problema: por qué Q-learning con redes diverge

DQN resuelve la **inestabilidad** del RL cuando se usa un aproximador no lineal (una red neuronal) para representar la función de valor-acción $Q$. El paper enumera tres causas:

1. Las **correlaciones** en la secuencia de observaciones consecutivas.
2. Pequeñas actualizaciones a $Q$ pueden cambiar significativamente la política y, por ende, la **distribución de los datos**.
3. Las **correlaciones entre los valores $Q$ y el objetivo** $r + \gamma \max_{a'} Q(s', a')$.

La idea de diseño que une la solución: estabilizar Q-learning profundo requiere **descorrelacionar los datos** (con replay) y **descorrelacionar el objetivo de la predicción** (con la red objetivo). Las dos modificaciones atacan dos de las tres fuentes de inestabilidad.

## Formalización: MDP y Q-learning

El agente interactúa con el emulador eligiendo en cada paso una acción $a_t \in A = \{1, \dots, K\}$. Como la pantalla actual $x_t$ no basta para entender la situación (problema parcialmente observable), el algoritmo trabaja sobre **secuencias** $s_t = x_1, a_1, x_2, \dots, x_t$, definiendo un MDP grande pero finito. El objetivo es maximizar el retorno futuro descontado $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$, con $\gamma = 0.99$. La función de valor-acción óptima:

$$Q^*(s,a) = \max_\pi \mathbb{E}\!\left[ R_t \mid s_t = s,\, a_t = a,\, \pi \right]$$

obedece la ecuación de Bellman, y DQN la aproxima con una red convolucional, $Q(s, a; \theta) \approx Q^*(s,a)$. El algoritmo es **model-free** (aprende con muestras del emulador sin estimar la dinámica) y **off-policy** (aprende sobre la política voraz mientras sigue una política $\epsilon$-greedy exploratoria).

## La red objetivo (target network)

El entrenamiento minimiza una pérdida que cambia en cada iteración $i$:

$$L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_i^{-}) - Q(s, a; \theta_i) \right)^2 \right]$$

Lo crucial está en los parámetros: $\theta_i$ son los pesos de la red $Q$ activa, mientras que $\theta_i^{-}$ son los de la **red objetivo** usados para calcular el target. Estos últimos **solo se sincronizan con la red $Q$ cada $C$ pasos** y se mantienen fijos entre actualizaciones.

El paper explica con precisión por qué esto estabiliza: en Q-learning online estándar, una actualización que aumenta $Q(s_t, a_t)$ frecuentemente *también* aumenta $Q(s_{t+1}, a)$ para todo $a$, y por ende eleva el propio objetivo, pudiendo provocar oscilaciones o divergencia. Generar los objetivos con un conjunto de parámetros más antiguo introduce un **retraso (delay)** entre el momento en que se actualiza $Q$ y el momento en que esa actualización afecta a los objetivos, haciendo la divergencia mucho menos probable. Esta es la idea que distingue la versión madura de *Nature* del preprint de 2013.

## Experience replay

Las transiciones $e_t = (s_t, a_t, r_t, s_{t+1})$ se almacenan en una memoria $D$ (las últimas $N = 1$ millón) y el entrenamiento muestrea minibatches **uniformemente al azar** de ese reservorio. Tres ventajas sobre Q-learning online:

1. **Eficiencia de datos:** cada transición se reutiliza en muchas actualizaciones de peso.
2. **Rompe correlaciones:** aleatorizar las muestras reduce la varianza de las actualizaciones, evitando el sesgo de aprender de muestras consecutivas fuertemente correlacionadas.
3. **Evita bucles de retroalimentación nocivos:** el replay promedia la distribución de comportamiento sobre muchos estados pasados, suavizando el aprendizaje. Aprender de un buffer obliga a aprender *off-policy*, lo que motiva la elección de Q-learning.

El paper reconoce una limitación: el muestreo uniforme da igual peso a todas las transiciones y el buffer sobrescribe ciegamente las más antiguas. Una estrategia que enfatizara las transiciones más informativas —análogo al *prioritized sweeping*— sería mejor, sembrando la idea que más tarde fructificaría en *prioritized experience replay*.

## Reward clipping y arquitectura

**Recorte de recompensas:** todas las recompensas positivas se fijan en $+1$ y las negativas en $-1$ ($0$ intacto). Esto limita la escala de las derivadas del error y permite usar *la misma* tasa de aprendizaje a través de juegos cuyos puntajes varían en órdenes de magnitud. Adicionalmente, el término de error TD se recorta al intervalo $[-1, 1]$ (una forma de pérdida de Huber) para mayor estabilidad.

La **red convolucional** recibe solo el estado y produce un valor $Q$ por cada acción en la salida (un único *forward pass* calcula todas las acciones):

- **Entrada:** imagen $84 \times 84 \times 4$ (4 frames apilados tras preprocesar).
- **Conv 1:** 32 filtros $8 \times 8$, stride 4 + ReLU.
- **Conv 2:** 64 filtros $4 \times 4$, stride 2 + ReLU.
- **Conv 3:** 64 filtros $3 \times 3$, stride 1 + ReLU.
- **Densa:** 512 unidades rectificadoras.
- **Salida:** lineal, una por acción válida (4 a 18 según el juego).

El **preprocesamiento** toma el máximo píxel a píxel entre el frame actual y el anterior (elimina el parpadeo de sprites de Atari), extrae la luminancia, reescala a $84 \times 84$ y apila los $m = 4$ frames más recientes. Con **frame-skipping** ($k = 4$) el agente actúa solo cada 4-º frame. Se entrena con RMSProp, minibatches de 32, $\epsilon$ recocido de 1.0 a 0.1 sobre el primer millón de frames, y 50 millones de frames por juego.

## Resultados

Se entrenó una red por juego, pero con *idéntica* arquitectura, algoritmo e hiperparámetros a través de los 49 juegos (estos últimos elegidos por búsqueda informal sobre 5 juegos de validación). El conocimiento previo inyectado fue mínimo: que la entrada son imágenes, el puntaje, el número de acciones y el contador de vidas.

- DQN **supera a los mejores métodos de RL existentes** (incluido el mejor aproximador lineal) en **43 de los 49 juegos**.
- Alcanzó un nivel **comparable al del *tester* humano profesional**, logrando **más del 75 % del puntaje humano en 29 juegos** (más de la mitad).
- El desempeño se normaliza como $100 \times \frac{\text{DQN} - \text{aleatorio}}{\text{humano} - \text{aleatorio}}$ (100 % = humano). DQN va desde superar masivamente al humano (Video Pinball, Boxing, Breakout, Star Gunner, por encima del 1000 %) hasta quedar muy por debajo en los del fondo.
- Las **curvas de entrenamiento** son estables: tanto el puntaje promedio como el valor $Q$ predicho sobre estados *held-out* crecen suavemente, evidenciando que entrenar redes grandes con señal de RL y SGD se logró de forma estable.

El análisis **t-SNE** de la última capa oculta muestra que la red mapea cerca no solo estados perceptualmente similares, sino también estados perceptualmente distintos pero **cercanos en recompensa esperada** (por ejemplo, pantallas casi completas y completas en Space Invaders). Las representaciones incluso generalizan a datos de políticas distintas (estados de juego humano y del agente caen en clústeres superpuestos).

## Ablaciones: replay y red objetivo son cruciales

El experimento que valida el diseño entrenó agentes con **todas las combinaciones** de replay encendido/apagado y red objetivo separada sí/no. El resultado es contundente: **ambos componentes son críticos** —desactivar el replay o la red objetivo degrada drásticamente el desempeño. Una prueba adicional muestra que reemplazar la CNN por un aproximador lineal (manteniendo replay y red objetivo) también colapsa el rendimiento: los tres componentes —replay, red objetivo y CNN profunda— son indispensables.

## Limitaciones reconocidas

- **Planificación temporalmente extendida.** Los juegos que demandan estrategias de largo plazo siguen siendo un desafío para *todos* los agentes. El ejemplo emblemático es **Montezuma's Revenge**, donde DQN queda en el fondo absoluto (esencialmente nivel aleatorio): las recompensas son escasas, diferidas y requieren secuencias largas de subobjetivos. Contrasta con Breakout, donde DQN *sí* descubre la estrategia de cavar un túnel lateral para enviar la pelota detrás del muro.
- **El recorte de recompensas pierde magnitud.** Fijar todo a $\pm 1$ permite una sola tasa de aprendizaje, pero impide al agente diferenciar recompensas de distinta magnitud.
- **Replay uniforme y no priorizado.** Igual peso a todas las transiciones; sesgar hacia eventos salientes sería una mejora futura.
- **Observabilidad parcial.** Apilar 4 frames es una heurística; estados que requieren memoria más allá de esa ventana quedan fuera de alcance.

## Impacto y conexión neurocientífica

DQN-Nature es un **hito histórico del deep reinforcement learning**: el primer agente artificial que aprende a sobresalir en un abanico diverso de tareas desde percepción cruda con una sola configuración. Su publicación en *Nature* marcó el inicio de la era moderna del deep RL y abrió el camino a Double DQN, Dueling DQN, prioritized experience replay, Rainbow, y eventualmente AlphaGo y AlphaZero del mismo grupo de DeepMind.

El paper enmarca su aporte también en términos neurocientíficos: el *experience replay* tiene un correlato biológico plausible en la reactivación comprimida de trayectorias durante periodos *offline* (reposo en vigilia), un mecanismo putativo para actualizar funciones de valor vía los ganglios basales. Y el *end-to-end* RL —usar la recompensa para moldear las representaciones de la CNN— se apoya en evidencia de que las señales de recompensa influyen en las representaciones de la corteza visual primate.

## Conexión con la Clase 31

Este paper *es* el material de la [Clase 31](/clases/clase-31) en su segmento de Deep Q-Learning. Los **"dos trucos clave"** que la clase enseña —experience replay y target network— provienen literalmente de aquí: el replay descorrelaciona los *datos* y la red objetivo descorrelaciona el *objetivo de la predicción*.

El [laboratorio de la clase](/laboratorios/lab-31) implementa exactamente el algoritmo de este paper: una red $Q$ y una red objetivo $\hat{Q}$, un buffer de replay del que se muestrean minibatches, la pérdida de TD con el objetivo calculado por la red congelada, y el reseteo $\hat{Q} \leftarrow Q$ cada $C$ pasos. **La red objetivo es el detalle que separa el lab "que funciona" del "que diverge":** igual que en la ablación del paper, sin ella el entrenamiento oscila o diverge. Los fundamentos —MDP, ecuación de Bellman, on-policy vs off-policy, $\epsilon$-greedy— se desarrollan en el fundamento de [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado).

En la genealogía del deep RL, el [DQN 2013 de Mnih et al.](/papers/dqn-atari-mnih-2013) es el preludio (sin red objetivo, sobre 7 juegos) y el [Double DQN de van Hasselt et al. 2015](/papers/double-dqn-van-hasselt-2015) es la secuela inmediata que corrige el *sesgo de sobreestimación* del $\max$ en el objetivo de DQN.

## Notas y enlaces

- Venue: *Nature*, vol. 518, pp. 529-533, 26 de febrero de 2015. DOI: 10.1038/nature14236.
- Autores: Google DeepMind (Londres). Mnih, Kavukcuoglu y Silver con contribución equitativa.
- Recursos del curso: [Clase 31](/clases/clase-31) · [Laboratorio 31](/laboratorios/lab-31) · [Fundamento: Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado) · [DQN Atari (Mnih 2013)](/papers/dqn-atari-mnih-2013) · [Double DQN (van Hasselt 2015)](/papers/double-dqn-van-hasselt-2015)
