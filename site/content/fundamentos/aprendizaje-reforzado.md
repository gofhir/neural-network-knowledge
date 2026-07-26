---
title: "Aprendizaje Reforzado (RL)"
weight: 108
math: true
---

El **aprendizaje reforzado** (reinforcement learning, RL) es el paradigma de machine learning en el que un **agente aprende a tomar decisiones secuenciales por ensayo y error**, interactuando con un ambiente y guiándose por una señal de **recompensa**. A diferencia de los otros dos grandes paradigmas, aquí no hay un conjunto de respuestas correctas que copiar: el agente debe *descubrir*, por sí mismo, qué acciones producen más recompensa a la larga, y debe hacerlo mientras esas mismas acciones determinan qué situaciones verá después. Es el marco formal que subyace a logros tan diversos como un programa que vence al campeón mundial de Go, un robot que aprende a caminar, o el [RLHF](/fundamentos/rlhf) que alinea a un modelo de lenguaje como ChatGPT con las preferencias humanas. Este fundamento es el núcleo conceptual de la [Clase 31](/clases/clase-31): recorre desde el bucle agente-ambiente y su formalización como proceso de decisión de Markov, pasando por Q-Learning y su escalado profundo con DQN, hasta los métodos basados en política (PPO) y los hitos que definieron el campo (AlphaGo).

---

## 1. Qué es RL: aprender por ensayo y error

La idea central es un agente situado en un **ambiente** con el que interactúa en un bucle cerrado. En cada paso de tiempo $t$, el agente observa un **estado** $s_t$, elige una **acción** $a_t$, y el ambiente responde devolviéndole una **recompensa** $r_{t+1}$ (un escalar) y un nuevo estado $s_{t+1}$. El ciclo se repite. El agente no recibe instrucciones sobre qué acción es correcta; solo una señal numérica que evalúa, después del hecho, cuán bien le fue. Su objetivo no es maximizar la recompensa inmediata sino la **recompensa acumulada a lo largo del tiempo** —lo que obliga a razonar sobre consecuencias diferidas: una acción puede dar poca recompensa ahora pero abrir la puerta a mucha más después.

Conviene contrastarlo con los otros paradigmas. En el **aprendizaje supervisado** existe un conjunto de pares (entrada, etiqueta correcta) y el modelo aprende a imitar esa correspondencia: hay un maestro que dice la respuesta. En el **no supervisado** no hay etiquetas y el modelo busca estructura latente (clusters, factores, densidad). El RL ocupa un territorio propio: hay una señal de retroalimentación, pero es **evaluativa, no instructiva** —dice "qué tan bueno fue lo que hiciste", no "qué deberías haber hecho"—, llega **diferida** (la recompensa puede depender de decisiones tomadas mucho antes), y los datos **no son i.i.d.** sino generados por las propias decisiones del agente. Esto último introduce el rasgo más distintivo del RL: el agente influye en la distribución de datos que recibe.

{{< concept-alert type="clave" >}}
El sello del RL es el **bucle de interacción** y la **recompensa diferida**. El agente no aprende de un dataset fijo, sino de las consecuencias de sus propias acciones, donde el premio o castigo puede llegar muchos pasos después de la decisión que lo causó. Por eso el problema central no es ajustar una función a datos dados, sino **descubrir qué secuencia de acciones maximiza la recompensa a largo plazo** en un mundo que responde a lo que el agente hace.
{{< /concept-alert >}}

Esta tensión define todo lo que sigue: cómo formalizar el mundo (el MDP), cómo medir el valor de largo plazo (las funciones de valor), cómo equilibrar probar cosas nuevas contra aprovechar lo conocido (exploración/explotación), y cómo aprender la política de forma estable.

El RL tiene raíces profundas fuera de la computación. La **psicología del condicionamiento** —la ley del efecto de Thorndike (1911): las acciones seguidas de satisfacción tienden a repetirse— es su antecedente conductual directo. Y la **neurociencia** ofrece una de las analogías más notables del área: las señales fásicas de las neuronas dopaminérgicas en el cerebro de los primates se comportan como un *error de diferencia temporal*, la misma cantidad que está en el centro de Q-Learning. Esa convergencia entre un algoritmo de ingeniería y un mecanismo biológico de aprendizaje es parte de por qué el RL se considera un marco normativo de la toma de decisiones, no solo una técnica más de machine learning.

---

## 2. Formalización: el proceso de decisión de Markov (MDP)

El RL se formaliza mediante un **proceso de decisión de Markov** (Markov Decision Process, MDP), la estructura matemática que captura "agente que decide en un mundo estocástico". Un MDP queda definido por la tupla $(S, A, P, R, \gamma)$:

- **$S$**, el conjunto de **estados** posibles del ambiente.
- **$A$**, el conjunto de **acciones** disponibles para el agente.
- **$P(s' \mid s, a)$**, la **función de transición**: la probabilidad de pasar al estado $s'$ tras ejecutar la acción $a$ en el estado $s$. Aquí vive la estocasticidad y la dinámica del mundo.
- **$R(s, a)$**, la **recompensa** esperada al ejecutar $a$ en $s$ (a veces escrita $R(s,a,s')$).
- **$\gamma \in [0, 1)$**, el **factor de descuento**, que pondera cuánto valen las recompensas futuras frente a las inmediatas.

El comportamiento del agente se describe mediante una **política** $\pi(a \mid s)$: una distribución de probabilidad sobre acciones condicionada al estado (o, si es determinista, un mapeo $\pi(s) = a$). La política es *lo que el agente aprende*. El objetivo es encontrar la política $\pi^*$ que maximiza el **retorno** —la suma de recompensas descontadas desde el instante $t$ en adelante—:

$$
G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k+1}.
$$

El factor de descuento $\gamma$ cumple dos roles. **Matemáticamente**, con $\gamma < 1$ garantiza que la serie converja a un valor finito incluso en horizontes infinitos. **Conceptualmente**, codifica una preferencia por la inmediatez: $\gamma$ cercano a 0 hace al agente miope (solo le importa el premio inmediato), $\gamma$ cercano a 1 lo vuelve previsor (valora recompensas lejanas casi tanto como las cercanas). Elegir $\gamma$ —típicamente $0{,}9$ a $0{,}99$— es una decisión de diseño que define el horizonte efectivo de planificación.

| Elemento | Símbolo | Qué representa |
|---|---|---|
| Estado | $s$ | Situación actual del ambiente |
| Acción | $a$ | Decisión del agente |
| Transición | $P(s'\mid s,a)$ | Dinámica estocástica del mundo |
| Recompensa | $r$ | Señal evaluativa escalar |
| Descuento | $\gamma$ | Peso de lo futuro vs. lo inmediato |
| Política | $\pi(a\mid s)$ | Estrategia que se aprende |
| Retorno | $G_t$ | Recompensa descontada acumulada |

El supuesto que hace tratable todo esto es la **propiedad de Markov**: el estado actual contiene toda la información relevante para predecir el futuro, de modo que la transición y la recompensa dependen *solo* del estado y la acción presentes, **no de la historia completa** de cómo se llegó allí. Formalmente, $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots)$. Esta propiedad es lo que permite definir funciones de valor recursivas y, con ellas, las ecuaciones de Bellman. Cuando el estado *no* es plenamente observable —como en Atari, donde una sola pantalla no revela la velocidad de la pelota— se recupera la markovianidad apilando varias observaciones recientes para reconstruir un estado suficiente.

---

## 3. Funciones de valor y las ecuaciones de Bellman

Para razonar sobre recompensa de largo plazo necesitamos cuantificar "cuán bueno" es estar en un estado, o tomar una acción en él. Eso lo hacen las **funciones de valor**, siempre definidas respecto de una política $\pi$.

La **función de valor de estado** $V^\pi(s)$ es el retorno esperado partiendo de $s$ y siguiendo $\pi$ después:

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[\, G_t \mid s_t = s \,\right].
$$

La **función de valor-acción** $Q^\pi(s, a)$ —el *action-value* o "función Q"— es el retorno esperado de tomar la acción $a$ en $s$ y *luego* seguir $\pi$:

$$
Q^\pi(s, a) = \mathbb{E}_\pi\!\left[\, G_t \mid s_t = s,\, a_t = a \,\right].
$$

La función $Q$ es especialmente útil porque, si la conocemos, decidir es trivial: basta tomar la acción de mayor valor, sin necesidad de un modelo del ambiente. Su relación con $V$ es $V^\pi(s) = \sum_a \pi(a\mid s)\, Q^\pi(s,a)$.

La estructura recursiva del retorno —"recompensa inmediata más valor descontado de lo que viene"— da lugar a las **ecuaciones de Bellman**. La *ecuación de expectativa* de Bellman expresa el valor de un estado en términos del valor de sus sucesores:

$$
V^\pi(s) = \sum_a \pi(a\mid s) \sum_{s'} P(s'\mid s,a)\big[\, R(s,a) + \gamma\, V^\pi(s') \,\big].
$$

Esta ecuación es el corazón de toda la teoría: convierte un problema de horizonte infinito en una relación de punto fijo local. Entre todas las políticas existe al menos una **política óptima** $\pi^*$ que domina a las demás en todo estado; su función de valor óptima $V^*(s) = \max_\pi V^\pi(s)$ y su $Q^*(s,a) = \max_\pi Q^\pi(s,a)$ satisfacen las **ecuaciones de optimalidad de Bellman**:

$$
Q^*(s,a) = \mathbb{E}\!\left[\, r + \gamma \max_{a'} Q^*(s',a') \,\right], \qquad V^*(s) = \max_a Q^*(s,a).
$$

La aparición del operador $\max$ —en lugar de promediar sobre la política— es lo que la distingue de la ecuación de expectativa: el agente óptimo no sigue una política fija, sino que en cada estado elige la mejor acción. Y de aquí se obtiene la política óptima de forma directa: $\pi^*(s) = \arg\max_a Q^*(s,a)$. **Aprender $Q^*$ equivale a resolver el problema.** Toda la familia de algoritmos *value-based* —Q-Learning, DQN y sus variantes— persigue exactamente este objetivo.

---

## 4. Exploración vs. explotación; on-policy vs. off-policy

Hay un dilema que es exclusivo del RL y no tiene análogo en el aprendizaje supervisado: el de **exploración versus explotación**. Para maximizar la recompensa, el agente debería **explotar** lo que ya sabe —tomar la acción que cree mejor—. Pero para *saber* qué es mejor, primero debe **explorar** —probar acciones cuyo valor aún es incierto—. Explotar demasiado pronto condena al agente a un óptimo local: nunca descubre que existía una opción superior porque jamás la intentó. Explorar demasiado desperdicia recompensa en acciones que ya se sabe que son malas. El balance es esencial, y no es un detalle de implementación: la garantía de convergencia de Q-Learning depende de que *todas* las acciones se prueben en *todos* los estados infinitas veces, lo que solo ocurre si hay exploración persistente.

La estrategia más común es **$\varepsilon$-greedy**: con probabilidad $1 - \varepsilon$ el agente *explota* (toma $\arg\max_a Q(s,a)$) y con probabilidad $\varepsilon$ *explora* (elige una acción al azar). Es habitual usar $\varepsilon$ alto al principio (mucha exploración) y reducirlo gradualmente (*annealing*) a medida que las estimaciones maduran; DQN, por ejemplo, lo decae de $1{,}0$ a $0{,}1$ sobre el primer millón de pasos.

Una distinción ortogonal y fundamental es **on-policy versus off-policy**. Un método **on-policy** aprende el valor de la *misma* política que usa para comportarse (explorar): SARSA es el ejemplo canónico. Un método **off-policy** aprende el valor de una política *objetivo* (típicamente la greedy óptima) mientras *se comporta* siguiendo una política distinta y más exploratoria: Q-Learning es el ejemplo canónico, gracias al $\max_{a'}$ en su regla de actualización, que evalúa la mejor acción posible aunque el agente haya tomado otra. Ser off-policy es lo que permite a un agente aprender de experiencias pasadas almacenadas, de demostraciones de otros, o de un buffer de replay —una propiedad que será decisiva para DQN.

---

## 5. Q-Learning: control tabular sin modelo

[**Q-Learning**](/papers/q-learning-watkins-1992), introducido por Watkins en 1989 y demostrado convergente por Watkins y Dayan en 1992, es el algoritmo fundacional del RL sin modelo. Su idea es aprender directamente la tabla de valores $Q(s,a)$ —una entrada por cada par estado-acción— mediante actualizaciones incrementales basadas en la experiencia. La regla de actualización es:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \,\big].
$$

Aquí $\alpha$ es la **tasa de aprendizaje** y el término entre corchetes es el **error de diferencia temporal** (TD error): la discrepancia entre la estimación actual $Q(s,a)$ y un *objetivo* mejor informado, $r + \gamma \max_{a'} Q(s',a')$, construido con la recompensa real observada más el valor descontado del mejor sucesor. Cada actualización empuja la estimación hacia el objetivo; si el error es cero, la estimación ya satisface la ecuación de optimalidad de Bellman. Esta técnica de "estimar a partir de estimaciones" se llama **bootstrapping** y es la herencia directa de los métodos de diferencias temporales de Sutton.

Q-Learning tiene tres propiedades que lo hicieron canónico:

- Es **model-free** (sin modelo): nunca necesita conocer ni estimar $P(s'\mid s,a)$ ni $R(s,a)$. Toda la dinámica queda absorbida implícitamente en los valores $Q$, aprendidos por puro ensayo y error.
- Es **off-policy**: el $\max_{a'}$ hace que aprenda el valor de la política greedy óptima aunque, mientras explora con $\varepsilon$-greedy, ejecute una política subóptima.
- **Converge** a $Q^*$ con probabilidad 1, bajo condiciones razonables: que todo par $(s,a)$ se visite infinitas veces y que la tasa de aprendizaje cumpla las condiciones de Robbins-Monro ($\sum \alpha = \infty$, $\sum \alpha^2 < \infty$). Esta garantía teórica —que "había eludido a la mayoría de los métodos de RL"— es el peso histórico del paper de Watkins y Dayan.

Vale notar el contraste con su primo **on-policy SARSA**, cuya actualización usa $r + \gamma\,Q(s', a')$ con la acción $a'$ que el agente *realmente* tomará, en lugar del $\max$. SARSA aprende el valor de la política exploratoria que sigue, por lo que tiende a comportamientos más "cautelosos" (evita el borde del precipicio que la exploración podría hacerle pisar), mientras que Q-Learning aprende la política óptima asumiendo comportamiento greedy futuro. Esta diferencia de un solo término —$\max_{a'}$ versus $Q(s',a')$— es el ejemplo más limpio de la distinción off-policy / on-policy de la sección anterior.

Su límite es estructural: la representación **tabular** tiene una entrada por par estado-acción, lo que es inviable cuando el espacio de estados es enorme o continuo (imágenes, control de alta dimensión). No se puede tener una fila por cada configuración posible de píxeles de una pantalla de Atari. Esta barrera es exactamente la que Deep Q-Learning vino a romper.

---

## 6. Deep Q-Learning (DQN): escalar Q con redes neuronales

La idea de **Deep Q-Learning** ([DQN](/papers/dqn-atari-mnih-2013), Mnih et al. 2013; versión madura en [Nature 2015](/papers/dqn-nature-mnih-2015)) es reemplazar la tabla por una **red neuronal** $Q(s,a;\theta)$ que aproxima la función de valor-acción, entrenada para minimizar el mismo error TD que en el caso tabular. Esto permite *generalizar* entre estados similares y manejar entradas de alta dimensión: la red de DQN recibe los píxeles crudos de Atari y produce un valor $Q$ por acción, aprendiendo a jugar 49 juegos distintos con una única arquitectura, los mismos hiperparámetros y nivel comparable o superior al de un humano profesional.

El problema es que combinar Q-Learning con un aproximador no lineal es **inestable** —a veces diverge—. El paper diagnostica tres fuentes de inestabilidad, a veces llamadas la **"tríada mortal"** (bootstrapping + aproximación de función + entrenamiento off-policy): (1) las **correlaciones** entre observaciones consecutivas; (2) que pequeñas actualizaciones a $Q$ cambian la política y por ende la distribución de los datos; y (3) las **correlaciones entre los valores $Q$ y los objetivos** $r + \gamma \max_{a'} Q(s',a')$, que comparten los mismos pesos. DQN ataca estos problemas con **dos trucos clave**:

- **Experience replay.** Las transiciones $(s, a, r, s')$ se guardan en una memoria (buffer) y el entrenamiento muestrea minibatches *al azar* de ese reservorio. Esto **descorrelaciona los datos** (rompe las correlaciones temporales de muestras consecutivas), permite **reutilizar cada experiencia** muchas veces (eficiencia de datos) y evita bucles de retroalimentación nocivos. Aprender de un buffer obliga a ser off-policy —y ahí Q-Learning encaja perfecto—.
- **Target network (red objetivo).** Se mantiene una copia *congelada* $\hat{Q}(\cdot;\theta^-)$ de la red para calcular los objetivos TD, y solo se sincroniza con la red en entrenamiento cada $C$ pasos. Esto **descorrelaciona el objetivo de la predicción**: introduce un retraso que evita que una actualización de $Q$ arrastre simultáneamente al objetivo, lo que de otro modo provocaría oscilaciones o divergencia.

La pérdida resultante es $L(\theta) = \mathbb{E}_{(s,a,r,s')\sim U(D)}\big[(r + \gamma \max_{a'} \hat{Q}(s',a';\theta^-) - Q(s,a;\theta))^2\big]$. Las ablaciones del paper confirman que **ambos trucos son indispensables**: quitar cualquiera de los dos colapsa el desempeño.

Sobre esta base surgió una familia de **mejoras quirúrgicas y combinables**:

| Mejora | Problema que ataca | Idea |
|---|---|---|
| [Double DQN](/papers/double-dqn-van-hasselt-2015) | Sobreestimación del $\max$ | Desacopla *selección* y *evaluación* de la acción |
| [Dueling DQN](/papers/dueling-dqn-wang-2015) | Eficiencia de la arquitectura | Factoriza $Q = V + A$ (valor + ventaja) |
| [Prioritized Replay](/papers/per-schaul-2015) | Eficiencia de uso de datos | Samplea por TD-error, no uniforme |

**Double DQN** corrige el *sesgo de sobreestimación*: el operador $\max$ sobre estimaciones ruidosas siempre tira hacia arriba, inflando los valores; la solución usa una red para *elegir* la mejor acción y otra para *evaluarla*. **Dueling DQN** reorganiza la red en dos flujos —uno estima el valor del estado $V(s)$, otro la *ventaja* $A(s,a)$ de cada acción sobre la media— y los combina, lo que ayuda en estados donde la acción importa poco. **Prioritized Experience Replay** muestrea con más frecuencia las transiciones de mayor TD-error, las que más enseñan, en vez de uniformemente. **Rainbow** (Hessel et al. 2017) demostró que estas mejoras son complementarias al integrarlas todas en un solo agente que supera a cada una por separado.

---

## 7. Métodos basados en política (policy-based)

La familia *value-based* aprende $Q$ y deriva la política como $\arg\max_a Q$. La familia complementaria, **policy-based**, **optimiza la política directamente**, parametrizándola como $\pi_\theta(a\mid s)$ y ajustando $\theta$ por ascenso de gradiente sobre el retorno esperado. Es la opción natural cuando el espacio de acciones es **continuo** —control robótico, donde tomar el $\arg\max$ de una función Q es inviable— y cuando se quiere una política estocástica.

El punto de partida es el **policy gradient**, formalizado por **REINFORCE**: el gradiente del retorno esperado es $\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a\mid s)\, G_t\big]$, que sube la probabilidad de las acciones seguidas de buen retorno y la baja para las malas. Su problema es la **alta varianza** del estimador, que hace el aprendizaje lento y ruidoso.

El **actor-critic** reduce esa varianza combinando dos componentes: un *actor* (la política $\pi_\theta$) y un *crítico* (una función de valor $V$ o $Q$) que evalúa las acciones del actor. En lugar del retorno crudo $G_t$ se usa la **ventaja** $A(s,a) = Q(s,a) - V(s)$ —cuánto mejor es una acción que el promedio del estado—, una señal mucho menos ruidosa. Esto encarna un *trade-off* clásico de sesgo y varianza: el retorno Monte Carlo $G_t$ es insesgado pero de altísima varianza, mientras que el crítico introduce algo de sesgo (sus estimaciones son imperfectas) a cambio de reducir drásticamente la varianza, acelerando y estabilizando el aprendizaje. El actor-critic es, en cierto sentido, la síntesis de las dos familias: aprovecha una función de valor (lo propio de value-based) para mejorar el gradiente de una política explícita (lo propio de policy-based). **[A3C](/papers/a3c-mnih-2016)** (Asynchronous Advantage Actor-Critic, Mnih et al. 2016) lleva esto a escala usando múltiples *workers* asíncronos que exploran copias del ambiente en paralelo y actualizan parámetros compartidos; la diversidad de sus experiencias descorrelaciona los datos —cumpliendo el rol del experience replay— y estabiliza el entrenamiento sin necesidad de un buffer.

**[PPO](/papers/ppo-schulman-2017)** (Proximal Policy Optimization, Schulman et al. 2017) es hoy el algoritmo policy-based más usado, por su equilibrio entre simplicidad y robustez. Su idea central es el **objetivo recortado** (*clipped objective*): al actualizar la política, PPO limita cuánto puede alejarse de la política anterior, recortando la razón de probabilidades $r_t(\theta) = \pi_\theta(a\mid s)/\pi_{\theta_{\text{old}}}(a\mid s)$ a un intervalo $[1-\epsilon, 1+\epsilon]$. Esto evita actualizaciones demasiado grandes que "rompan" la política —la estabilidad de TRPO, su predecesor de región de confianza, pero sin su costosa maquinaria de segundo orden—. PPO es, además, **la base del [RLHF](/fundamentos/rlhf)**: en ese contexto la "política" es un modelo de lenguaje, la "acción" es generar el siguiente token, y la "recompensa" la entrega un *reward model* entrenado con preferencias humanas, mientras un término KL contra el modelo de referencia cumple el mismo rol de "no alejarse demasiado" que motivó el recorte de PPO.

---

## 8. Hitos y aplicaciones

El RL pasó de curiosidad académica a motor de algunos de los resultados más visibles de la IA.

**Juegos como banco de pruebas.** Los videojuegos ofrecen ambientes ricos, reproducibles y con recompensa clara (el puntaje). **DQN** dominando Atari desde píxeles crudos (2015) fue el resultado que convirtió al deep RL en un campo. Los juegos siguen siendo el laboratorio donde se prueban los algoritmos antes de llevarlos al mundo real.

**[AlphaGo](/papers/alphago-silver-2016)** (Silver et al. 2016) es probablemente el hito más célebre: el primer programa en vencer a un campeón profesional de **Go**, un juego de complejidad astronómica ($\sim 10^{170}$ posiciones) que se creía fuera del alcance de las máquinas por décadas. Su receta combina **deep RL** (redes que estiman valor y política), **búsqueda de árbol de Monte Carlo** (MCTS, que planifica simulando jugadas futuras guiada por esas redes) y **self-play** (el agente mejora jugando millones de partidas contra versiones de sí mismo, generando su propio currículo de entrenamiento sin datos humanos al final). AlphaGo demostró que la combinación de aprendizaje y búsqueda podía superar la intuición humana en un dominio de planificación profunda.

**Robótica.** El control de robots —manipulación, locomoción, navegación— es un dominio natural para el RL: las acciones son continuas, las consecuencias diferidas, y diseñar a mano el controlador óptimo es intratable. Aquí dominan los métodos policy-based como PPO, capaces de manejar espacios de acción continuos. Ver el [dominio de robótica](/dominios/robotica) para el recorrido completo.

**RLHF para modelos de lenguaje.** La aplicación que devolvió al RL al centro de la atención: usar feedback humano para alinear LLMs. InstructGPT y ChatGPT se afinaron con PPO maximizando una recompensa aprendida de preferencias humanas. Es el puente directo entre el RL clásico de esta clase y los modelos generativos modernos.

---

## 9. Conexión con el curso y resumen

El aprendizaje reforzado conecta con varios hilos del curso. El más directo es el [RLHF de la Clase 20](/fundamentos/rlhf): la alineación de LLMs por refuerzo es una aplicación de PPO, y entender qué es una política, una recompensa y la ventaja es requisito para entenderlo. El segundo es el [dominio de robótica](/dominios/robotica), donde los métodos basados en política controlan agentes en espacios continuos. Y de fondo, el RL aporta una perspectiva distinta a todo el curso: en lugar de aprender de un dataset estático, aprender *interactuando* —un marco que reaparece cada vez que un sistema debe tomar decisiones secuenciales bajo incertidumbre.

El recorrido conceptual es acumulativo. Todo arranca con el **bucle agente-ambiente** y la **recompensa diferida** (sección 1), que se formalizan en el **MDP** —estados, acciones, transición, recompensa, descuento, política y retorno— bajo la **propiedad de Markov** (sección 2). Las **funciones de valor** $V$ y $Q$ y las **ecuaciones de Bellman** dan el lenguaje para razonar sobre valor de largo plazo, y la ecuación de optimalidad define la **política óptima** (sección 3). El dilema **exploración/explotación** y la distinción **on/off-policy** (sección 4) son transversales. Sobre esa base, **Q-Learning** aprende $Q^*$ tabular sin modelo (sección 5); **DQN** lo escala con redes neuronales, estabilizadas por **experience replay** y **target network**, y refinado por Double/Dueling/PER/Rainbow (sección 6). La familia complementaria **policy-based** —REINFORCE, actor-critic, A3C, PPO— optimiza la política directamente (sección 7). Y los **hitos** —Atari, AlphaGo, robótica, RLHF— muestran el alcance del paradigma (sección 8).

---

## Para profundizar

- [Q-Learning (Watkins y Dayan, 1992)](/papers/q-learning-watkins-1992) — el algoritmo fundacional del RL sin modelo y su prueba de convergencia.
- [DQN Atari (Mnih et al., 2013)](/papers/dqn-atari-mnih-2013) — el preludio que combinó Q-learning con CNN y experience replay.
- [DQN Nature (Mnih et al., 2015)](/papers/dqn-nature-mnih-2015) — la versión madura con target network y control a nivel humano en 49 juegos.
- [Double DQN (van Hasselt et al., 2015)](/papers/double-dqn-van-hasselt-2015) — corrige el sesgo de sobreestimación del $\max$.
- [Dueling DQN (Wang et al., 2015)](/papers/dueling-dqn-wang-2015) — factoriza la función Q en valor y ventaja.
- [Prioritized Experience Replay (Schaul et al., 2015)](/papers/per-schaul-2015) — samplea el buffer por TD-error.
- [A3C (Mnih et al., 2016)](/papers/a3c-mnih-2016) — actor-critic con ventaja y workers asíncronos.
- [PPO (Schulman et al., 2017)](/papers/ppo-schulman-2017) — el objetivo recortado, el policy gradient más usado y base del RLHF.
- [AlphaGo (Silver et al., 2016)](/papers/alphago-silver-2016) — deep RL + MCTS + self-play para vencer al campeón de Go.

**Recursos relacionados:** [RLHF](/fundamentos/rlhf) · [Dominio: Robótica](/dominios/robotica) · [Clase 31: Aprendizaje Reforzado](/clases/clase-31) · [Clase 33: Imitación e IRL](/clases/clase-33) · Fundamentos: [Aprendizaje Reforzado Inverso](/fundamentos/aprendizaje-reforzado-inverso) · [Aprendizaje por Imitación](/fundamentos/aprendizaje-por-imitacion) · [Generalización en RL](/fundamentos/generalizacion-en-rl)
