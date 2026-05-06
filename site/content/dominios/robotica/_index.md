---
title: "Robótica / RL"
weight: 6
sidebar:
  open: true
---

# Robótica / RL

## El problema central

Hasta aquí los demás dominios — texto, imagen, audio, video, multimodal — siguen un patrón común: el modelo recibe entrada y emite salida en un paso. **Reinforcement Learning (RL)** rompe ese patrón. Un agente actúa repetidamente en un entorno: cada acción cambia el estado, el entorno responde con una recompensa escalar, y el objetivo es maximizar la recompensa acumulada a lo largo del tiempo. Esto introduce dependencias temporales largas, exploración del espacio de acciones, y la pregunta del *credit assignment* — qué decisión pasada es responsable de la recompensa actual.

Tres tensiones específicas vertebran el campo: (1) **exploración vs explotación** — el agente debe descubrir acciones nuevas sin perder las que ya sabe que funcionan; (2) **eficiencia de muestra** — RL clásico necesita millones de interacciones, lo cual es viable en simulación pero infactible en robótica física donde cada interacción cuesta tiempo y desgaste; (3) **alineamiento** — cómo formular la "recompensa" cuando viene de preferencias humanas (RLHF), de instrucciones en lenguaje natural (robot foundation models), o cuando el espacio de objetivos es demasiado complejo para una función numérica fija. Estas tres tensiones marcan los saltos cualitativos entre las cinco eras de la disciplina.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era de RL clásico" years="1989-2010" >}}
    {{< hito year="1989" name="Q-learning" status="minimal" >}}
      Watkins: algoritmo *off-policy* que aprende la función $Q(s, a)$ — valor esperado de tomar acción $a$ en estado $s$ y seguir óptimamente — vía actualizaciones temporales de diferencia. **Por qué importó:** primer algoritmo de RL con prueba de convergencia bajo condiciones razonables; base de DQN dos décadas después.
    {{< /hito >}}
    {{< hito year="1992" name="REINFORCE / Policy Gradients" status="minimal" >}}
      Williams: método de gradiente sobre la política directamente, sin estimar valor. La política $\pi_\theta$ se actualiza en la dirección que aumenta la recompensa esperada. **Por qué importó:** alternativa fundamental a Q-learning; base de PPO, RLHF y casi todos los métodos modernos.
    {{< /hito >}}
    {{< hito year="1994" name="SARSA / TD-learning" status="minimal" >}}
      Rummery & Niranjan: variante *on-policy* de Q-learning — actualiza $Q$ usando la acción que efectivamente se tomó, no la óptima. **Por qué importó:** marca la distinción on-policy vs off-policy que estructura todo el campo posterior.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de Deep RL temprano" years="2013-2016" >}}
    {{< hito year="2013/2015" name="DQN" status="minimal" >}}
      Mnih et al. (DeepMind): combinaron Q-learning con redes neuronales convolucionales para jugar Atari directamente desde píxeles. Truco clave: *experience replay* + target network para estabilizar el entrenamiento. NIPS workshop 2013, Nature paper 2015. **Por qué importó:** primera aplicación exitosa de deep learning a RL; nivel humano en 49 juegos sin diseño manual de features.
    {{< /hito >}}
    {{< hito year="2015" name="DDPG" status="minimal" >}}
      Lillicrap et al. (DeepMind): extiende DQN a acciones continuas vía actor-critic determinista. **Por qué importó:** habilitó RL sobre control continuo (robótica simulada, MuJoCo) — paso necesario antes de pasar a robots reales.
    {{< /hito >}}
    {{< hito year="2016" name="A3C" status="minimal" >}}
      Mnih et al.: *Asynchronous Advantage Actor-Critic* — múltiples actores paralelos en CPU, sin replay buffer. Más rápido y eficiente que DQN en muchos entornos. **Por qué importó:** mostró que RL podía paralelizarse a escala; base conceptual de PPO un año después.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era AlphaGo" years="2016-2019" >}}
    {{< hito year="2016" name="AlphaGo" status="minimal" >}}
      Silver et al. (DeepMind): combinación de policy + value networks entrenadas con supervised learning sobre partidas humanas + RL por self-play, decodificadas en tiempo real con Monte Carlo Tree Search. Derrotó al campeón mundial Lee Sedol 4-1. **Por qué importó:** Go había sido el caso paradigmático "imposible para IA en 10 años"; AlphaGo lo resolvió primero.
    {{< /hito >}}
    {{< hito year="2017" name="AlphaZero" status="minimal" >}}
      Silver et al.: aprende Go, ajedrez y shogi desde cero — solo las reglas del juego, sin partidas humanas. Self-play puro + MCTS. Superó a Stockfish (mejor motor de ajedrez) tras 24h de entrenamiento. **Por qué importó:** demostró que la supervisión humana era prescindible cuando el entorno permite self-play.
    {{< /hito >}}
    {{< hito year="2019" name="AlphaStar" status="minimal" >}}
      Vinyals et al. (DeepMind): nivel grandmaster en StarCraft II — juego de información incompleta, tiempo real, espacio de acciones masivo. Combinó IL desde partidas humanas, league play y multi-agent RL. **Por qué importó:** mostró que las técnicas escalaban más allá de juegos perfectos como Go o ajedrez.
    {{< /hito >}}
    {{< hito year="2019" name="MuZero" status="minimal" >}}
      Schrittwieser et al.: variante de AlphaZero que **aprende su propio modelo del entorno** en lugar de usar las reglas explícitas. Iguala AlphaZero en Go/ajedrez/shogi, y supera a DQN en Atari. **Por qué importó:** unificó model-based y model-free RL; abrió la puerta a aplicar la receta a entornos sin reglas formales.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era PPO + RLHF" years="2017-2023" >}}
    {{< hito year="2017" name="PPO" status="minimal" >}}
      Schulman et al. (OpenAI): *Proximal Policy Optimization* — clip del ratio de probabilidad para evitar updates demasiado grandes que colapsen la política. Más simple y robusto que TRPO. **Por qué importó:** se volvió el algoritmo estándar de policy optimization, base de RLHF, OpenAI Five (Dota 2), y muchas pipelines de robótica.
    {{< /hito >}}
    {{< hito year="2022" name="InstructGPT / RLHF" status="covered" link="/fundamentos/sft" >}}
      Ouyang et al. (OpenAI): GPT-3 fine-tuneado con SFT (datos de demostración) + RLHF (modelo de recompensa entrenado sobre preferencias humanas, optimizado con PPO). Marcó la diferencia entre GPT-3 y ChatGPT.
    {{< /hito >}}
    {{< hito year="2022-2023" name="RLAIF / Constitutional AI" status="minimal" >}}
      Bai et al. (Anthropic): reemplazar parte de las preferencias humanas por preferencias generadas por un modelo guiado por una "constitución" de principios. **Por qué importó:** escala el alineamiento más allá de lo que humanos pueden anotar; base de Claude.
    {{< /hito >}}
    {{< hito year="2023" name="DPO" status="covered" link="/fundamentos/dpo" >}}
      Rafailov et al.: *Direct Preference Optimization* — reformula RLHF como una pérdida supervisada directa sobre pares de preferencias, eliminando el modelo de recompensa explícito y PPO. Equivalente teórico, mucho más simple en la práctica.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de Robot Foundation Models" years="2022-presente" >}}
    {{< hito year="2022" name="SayCan" status="minimal" >}}
      Ahn et al. (Google): combina un LLM (que propone planes en lenguaje) con value functions aprendidas (que evalúan factibilidad para el robot). El robot ejecuta solo planes que el LLM proponga *y* sean factibles físicamente. **Por qué importó:** primer puente serio entre LLMs y control robótico real.
    {{< /hito >}}
    {{< hito year="2022" name="RT-1" status="minimal" >}}
      Brohan et al. (Google): primer modelo Transformer entrenado sobre 130k demostraciones reales de manipulación. Entrada: imagen + instrucción en lenguaje. Salida: acción del brazo robótico. **Por qué importó:** demostró que el patrón de foundation models funcionaba en robótica física.
    {{< /hito >}}
    {{< hito year="2023" name="RT-2" status="minimal" >}}
      Brohan et al. (Google DeepMind): co-fine-tunea un VLM (PaLM-E o PaLI-X) sobre datos de robot, tratando la acción como tokens de texto. **Por qué importó:** mostró transferencia de conocimiento web → robot — el modelo razona sobre objetos nunca vistos en el dataset robótico usando lo aprendido en internet.
    {{< /hito >}}
    {{< hito year="2024" name="OpenVLA" status="minimal" >}}
      Kim et al. (Stanford): VLA open-source de 7B parámetros entrenado sobre Open X-Embodiment dataset (970k trajectorias, 22 robots). **Por qué importó:** democratizó la receta VLA — open weights que la comunidad pudo extender, fine-tunear y desplegar.
    {{< /hito >}}
    {{< hito year="2024" name="π0 (Physical Intelligence)" status="minimal" >}}
      Black et al. (Physical Intelligence): VLA generalist multi-embodiment con destreza fina, basado en flow matching sobre acciones. Una sola política controla brazos, manipuladores móviles y humanoides. **Por qué importó:** primer modelo que mostró control continuo de alta frecuencia (50Hz) con foundation model — necesario para tareas reales como doblar ropa.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}

## Era 1 — RL clásico (1989-2010)

### Problema heredado

Antes de RL existía la teoría del control óptimo (Bellman, 1957) y los procesos de decisión de Markov, pero faltaba algo crucial: ¿cómo aprender a actuar bien cuando **no se conoce el modelo del entorno**? El control óptimo asume conocer la dinámica $P(s' \mid s, a)$ y la recompensa $R(s, a)$; en problemas reales, ambas son desconocidas y solo se accede a ellas a través de muestreo.

### Idea clave

**Aprender de la experiencia, no del modelo.** Q-learning (Watkins, 1989) propuso aprender directamente la función de valor óptima $Q^*(s, a)$ — la recompensa acumulada esperada de tomar acción $a$ en estado $s$ y luego actuar óptimamente — sin necesidad de conocer la dinámica. La actualización es local: cada vez que el agente ejecuta una acción y recibe una recompensa, ajusta $Q$ ligeramente hacia el valor inferido del siguiente estado. Bajo condiciones razonables (todas las acciones son visitadas infinitas veces, learning rate decreciente), Q-learning converge al óptimo.

REINFORCE (Williams, 1992) tomó un camino distinto: en lugar de aprender valores y derivar política, **optimizar directamente la política** $\pi_\theta(a \mid s)$ vía gradiente. Si una trayectoria recibió alta recompensa, aumentar la probabilidad de las acciones que tomó. Este enfoque escala mejor a espacios de acciones continuos y es la base teórica de PPO y RLHF tres décadas después.

### Qué la destronó

Q-learning y REINFORCE eran exitosos en problemas pequeños (gridworlds, control simple) pero no escalaban. La función $Q$ se representaba con tablas indexadas por $(s, a)$ — inviable en cualquier problema con espacio de estados grande. Aproximarla con redes neuronales era inestable: pequeños cambios en pesos podían desestabilizar todo el bootstrap. La solución llegaría con DQN (2013) y dos trucos específicos.

## Era 2 — Deep RL temprano (2013-2016)

### Problema heredado

Aplicar Q-learning con una red neuronal como aproximador de $Q$ resultaba inestable: la red predice tanto el valor actual como el target (que también depende de la red), creando un objetivo que se mueve durante el entrenamiento. Las correlaciones entre transiciones consecutivas violan la suposición de muestras i.i.d. del SGD. Décadas de intentos fallidos.

### Idea clave

**DQN: experience replay + target network.** Mnih et al. (DeepMind, 2013/2015) resolvieron ambos problemas con dos trucos. Primero, **experience replay**: en lugar de aprender de la transición más reciente, guardar todas las transiciones $(s, a, r, s')$ en un buffer y muestrear lotes aleatorios. Esto rompe la correlación temporal y reutiliza datos. Segundo, **target network**: usar una copia congelada de la red para calcular el target, actualizándola lentamente. Esto estabiliza el bootstrap.

DDPG (Lillicrap et al., 2015) extendió DQN a acciones continuas vía actor-critic determinista — necesario para robótica donde "torque" es continuo, no discreto. A3C (Mnih et al., 2016) eliminó el replay buffer al usar múltiples actores asíncronos en paralelo, mostrando que RL podía escalar con CPU sin GPU.

### Qué la destronó

DQN y A3C funcionaban en Atari y MuJoCo, pero seguían siendo *frágiles*: hiperparámetros sensibles, dificultad de generalizar. Y el campo quería atacar problemas más ambiciosos — Go, StarCraft, robots reales. La frontera se movió a combinar RL con búsqueda explícita (Monte Carlo Tree Search) y self-play.

## Era 3 — La era AlphaGo (2016-2019)

### Problema heredado

Go era el caso emblemático "imposible para IA en 10 años" según expertos en 2015. El espacio de estados de Go ($\sim 10^{170}$ posiciones legales) hace inviable tabular $Q$. Las CNNs podían evaluar posiciones, pero no producir un jugador competitivo solas.

### Idea clave

**Combinar redes neuronales con búsqueda explícita.** AlphaGo (Silver et al., DeepMind, 2016) usó dos redes — una *policy network* que sugiere movimientos plausibles, una *value network* que evalúa posiciones — entrenadas con supervised learning sobre partidas humanas + RL por self-play. En tiempo real, decodificadas vía Monte Carlo Tree Search: el árbol explora movimientos guiado por la policy y poda con la value. Derrotó al campeón mundial Lee Sedol 4-1.

AlphaZero (2017) eliminó el supervised pretraining: solo las reglas del juego + self-play puro. Aprendió Go, ajedrez y shogi desde cero, superando a Stockfish (motor de ajedrez especializado, 30 años de desarrollo) en 24 horas de entrenamiento. AlphaStar (2019) extendió la receta a StarCraft II — información incompleta, tiempo real, espacio de acciones masivo. MuZero (2019) cerró el círculo aprendiendo su propio modelo del entorno, eliminando la última asunción (conocer las reglas).

### Qué la destronó

La era AlphaGo dominó juegos cerrados con reglas explícitas o aprendibles. Pero fuera de juegos, los problemas abiertos — robótica con lenguaje natural, alineamiento de LLMs — no podían formularse como search trees discretos. El siguiente capítulo del campo se movió hacia **policy optimization sin búsqueda**, simple y escalable: PPO.

## Era 4 — PPO + RLHF (2017-2023)

### Problema heredado

Tras AlphaGo, OpenAI quería atacar problemas como Dota 2 — espacio de acciones continuo y enorme, sin posibilidad de MCTS. Los métodos de policy gradient existentes (TRPO) eran complicados y costosos. Necesitaban algo simple que escalase.

### Idea clave

**PPO: clip del ratio de probabilidades.** Schulman et al. (OpenAI, 2017) propusieron *Proximal Policy Optimization*: en cada update, clipear el ratio $\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}$ para evitar pasos demasiado grandes que colapsen la política. Más simple que TRPO, casi tan robusto. PPO se volvió el algoritmo estándar de policy optimization durante la siguiente década — base de OpenAI Five (Dota 2 a nivel campeón mundial), del entrenamiento de robots en simulación, y crucialmente, de RLHF.

**RLHF — la aplicación que cambió todo.** InstructGPT (Ouyang et al., OpenAI, 2022) tomó GPT-3 y lo fine-tuneó con tres pasos: (1) SFT sobre demostraciones humanas, (2) entrenar un modelo de recompensa sobre comparaciones humanas pareadas, (3) optimizar la política con PPO usando ese modelo como recompensa. El resultado fue ChatGPT — el modelo que llevó RL a la conversación masiva.

DPO (Rafailov et al., 2023) demostró que el objetivo de RLHF se puede reescribir como una pérdida supervisada directa sobre pares de preferencias, eliminando el modelo de recompensa explícito y PPO. Equivalente teórico, mucho más simple en la práctica. Constitutional AI / RLAIF (Anthropic) escaló el alineamiento reemplazando preferencias humanas por preferencias generadas por un modelo guiado por una "constitución" de principios.

### Qué la destronó

PPO+RLHF y sus variantes son el estado del arte en alineamiento de LLMs en 2025. Pero la frontera del campo se movió a algo más ambicioso: **aplicar la receta foundation-model + RLHF a robótica física**. Eso requería arquitecturas que conectaran percepción visual, lenguaje y control continuo en un solo modelo.

## Era 5 — Robot Foundation Models (2022-presente)

### Problema heredado

Robótica clásica diseñaba pipelines especializadas: visión → planning → control, cada componente entrenado por separado. RT-1 mostró que un Transformer end-to-end podía superar esa pipeline, pero solo aprendía las tareas específicas en su dataset. La pregunta abierta: ¿se puede hacer un foundation model robótico — un modelo que aprenda de datos web masivos + algunos datos robóticos y generalice a tareas no vistas?

### Idea clave

**Vision-Language-Action (VLA) models.** SayCan (Ahn et al., Google, 2022) fue el puente: combina un LLM que propone planes en lenguaje con value functions aprendidas que evalúan factibilidad física, ejecutando solo planes plausibles *y* factibles. RT-1 (Brohan et al., 2022) entrenó un Transformer puro sobre 130k demostraciones reales — primer modelo robótico de propósito general a escala. RT-2 (2023) co-fine-tuneó un VLM (PaLM-E, PaLI-X) sobre datos de robot tratando la acción como tokens, transfiriendo conocimiento web → robot: el modelo razona sobre objetos nunca vistos en el dataset robótico usando lo aprendido en internet.

OpenVLA (Stanford, 2024) democratizó la receta con open weights y entrenamiento sobre Open X-Embodiment (970k trajectorias, 22 embodiments). π0 (Physical Intelligence, 2024) llevó la idea a control continuo de alta frecuencia (50Hz) usando flow matching sobre acciones — necesario para tareas reales como doblar ropa o ensamblar piezas. Una sola política controla brazos industriales, manipuladores móviles y humanoides.

### Qué viene

Las apuestas activas en RL/robótica: **multi-embodiment** (un modelo controla cualquier robot), **generalización por lenguaje** (instrucciones naturales complejas con razonamiento), **sim-to-real masivo** (entrenar en simulación enorme, transferir a físico), **razonamiento jerárquico** (planeación de largo horizonte + control bajo nivel), **recompensas aprendidas** (modelos de recompensa de visión-lenguaje en lugar de funciones diseñadas), **RL de razonamiento** (post-DeepSeek-R1, los LLMs aprenden a razonar via RL puro sobre cadenas de pensamiento), y el **robotic data flywheel** (despliegue → recolección → entrenamiento → mejor modelo). La pregunta abierta de 2025: si un foundation model multi-embodiment alcanza nivel humano en manipulación general, ¿qué tan rápido se vuelve económicamente viable un humanoide en cada hogar?
