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
