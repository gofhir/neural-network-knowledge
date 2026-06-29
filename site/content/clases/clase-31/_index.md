---
title: "Clase 31 - Aprendizaje Reforzado"
weight: 310
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga
**Curso 3:** Relacional, GANs, RL, Meta-Learning, Razonamiento y Memoria

Clase sobre **aprendizaje reforzado (RL)**: la rama de la IA que estudia cómo crear **agentes que aprenden a resolver problemas por ensayo y error**, maximizando la recompensa acumulada al interactuar con un ambiente. A diferencia del aprendizaje supervisado (entrada → salida en un paso), el RL introduce **decisiones secuenciales**, exploración del espacio de acciones y el problema del *credit assignment*. La clase formaliza el paradigma como un **proceso de decisión de Markov (MDP)** —estado, acción, recompensa, política π(a|s), descuento γ—, desarrolla **Q-Learning** (la ecuación de Bellman, la tabla Q, ε-greedy) y su escalamiento a **Deep Q-Learning (DQN)** con red neuronal, experience replay y target network, y cierra con el trabajo práctico (un DQN).

La clase conecta con el [RLHF (Clase 20)](/clases/clase-20) —que usa PPO para alinear LLMs con preferencias humanas— y con la [robótica](/dominios/robotica), el dominio natural del control por refuerzo.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las diapositivas: motivación, paradigma RL (MDP, política), Q-Learning, Deep Q-Learning (DQN)" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: MDP y retorno, ecuaciones de Bellman, Q-Learning, DQN (pérdida, replay, target), policy gradient/PPO, AlphaGo" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Q-Learning tabular y DQN desde cero en triple framework (PyTorch, TensorFlow, JAX)" icon="code" >}}
  {{< card link="/clases/clase-32" title="Clase siguiente: Olvido Catastrófico" subtitle="Aprendizaje continuo, EWC, replay" icon="arrow-right" >}}
  {{< card link="/clases/clase-30" title="Clase anterior: Modelos con memoria externa" subtitle="Memory Networks, NTM, memoria explícita" icon="arrow-left" >}}
  {{< card link="/clases/clase-20" title="Relacionada: BERT/GPT/ChatGPT (RLHF)" subtitle="PPO para alinear LLMs con preferencias humanas" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/aprendizaje-reforzado" title="Aprendizaje Reforzado" subtitle="MDP, Bellman, Q-Learning, DQN, policy gradient, actor-critic, PPO, AlphaGo" icon="book-open" >}}
  {{< card link="/fundamentos/rlhf" title="RLHF" subtitle="Reinforcement Learning from Human Feedback: PPO aplicado a LLMs" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/q-learning-watkins-1992" title="Q-Learning (1992)" subtitle="Watkins & Dayan — el algoritmo fundacional, con prueba de convergencia" icon="document-text" >}}
  {{< card link="/papers/dqn-atari-mnih-2013" title="DQN: Atari (2013)" subtitle="Mnih et al. — deep RL desde pixeles, experience replay" icon="document-text" >}}
  {{< card link="/papers/dqn-nature-mnih-2015" title="DQN: Human-level (Nature 2015)" subtitle="Mnih et al. — target network, 49 juegos a nivel humano" icon="document-text" >}}
  {{< card link="/papers/double-dqn-van-hasselt-2015" title="Double DQN (2015)" subtitle="van Hasselt et al. — corrige la sobreestimación del max" icon="document-text" >}}
  {{< card link="/papers/dueling-dqn-wang-2015" title="Dueling DQN (2015)" subtitle="Wang et al. — separar valor V(s) y ventaja A(s,a)" icon="document-text" >}}
  {{< card link="/papers/per-schaul-2015" title="Prioritized Experience Replay (2015)" subtitle="Schaul et al. — muestrear transiciones por su TD-error" icon="document-text" >}}
{{< /cards >}}

## Papers canónicos (complementarios)

{{< cards >}}
  {{< card link="/papers/a3c-mnih-2016" title="A3C (2016)" subtitle="Mnih et al. — actor-critic asíncrono, la familia policy-based" icon="document-text" >}}
  {{< card link="/papers/ppo-schulman-2017" title="PPO (2017)" subtitle="Schulman et al. — clipped surrogate, el RL más usado y base del RLHF" icon="document-text" >}}
  {{< card link="/papers/alphago-silver-2016" title="AlphaGo (2016)" subtitle="Silver et al. — deep RL + MCTS + self-play, hito histórico" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/robotica" title="Dominio: Robótica / RL" subtitle="Línea de tiempo: de Q-learning y TD-Gammon a DQN, AlphaGo y RLHF" icon="globe-alt" >}}
{{< /cards >}}
