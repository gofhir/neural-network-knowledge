---
title: "Lab 31 - Aprendizaje Reforzado: DQN sobre CartPole"
weight: 310
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga
**Fecha:** Junio 2026
**Notebook origen:** `clase_31/material/Laboratorio/Tutorial_DQN_DINTA_v18.ipynb`
**Notebook ejecutado:** [lab31.ipynb](/notebooks/lab31.ipynb) · [HTML](/notebooks-html/lab31.html)

## Encuadre

La contraparte práctica de la [clase 31](/clases/clase-31): implementar de cero un **Deep Q-Network (DQN)** ([Mnih et al. 2015](/papers/dqn-nature-mnih-2015)) y verlo resolver **CartPole**, el péndulo invertido clásico de control. Es el puente exacto de la clase: cuando la tabla Q del [Q-Learning tabular](/clases/clase-31/practica/01-q-learning-tabular-desde-cero) no escala a estados continuos, se aproxima con una red neuronal.

El notebook (basado en el material de [Evan Hennis](https://github.com/ehennis/ReinforcementLearning)) recorre el algoritmo completo: la red $Q_\theta$, el **experience replay buffer**, la política **ε-greedy** y el loop de entrenamiento que minimiza el **error de Bellman**.

| Pieza | Implementación en el lab |
|---|---|
| Aproximador $Q_\theta$ | MLP `4 → 24 → 24 → 2` con salida **lineal** (regresión de Q-values, no softmax) |
| Dataset de entrenamiento | experience replay buffer (`deque`, `maxlen=2500`) de transiciones $(s,a,r,s')$ |
| Política de exploración | ε-greedy con decaimiento (`epsilon *= 0.995` por paso de gradiente) |
| Target de Bellman | $y = r + \gamma \max_{a'} Q_\theta(s',a')$ (o $y=r$ si terminal) |
| Pérdida | MSE entre $Q_\theta(s,a)$ y el target |
| Estabilización | experience replay **sí**; target network **ausente** (simplificación pedagógica) |
| Evaluación | 100 episodios de test con ε=0 (explotación pura, sin entrenar) |

## Resultados consolidados

Entrenamiento real ejecutado en CPU (la red es diminuta; la GPU no ayuda):

| Métrica | Valor medido (Colab) |
|---|---|
| Baseline aleatorio | **21.0** pasos |
| Episodios hasta resolver (avg₁₀ > 195) | **85** |
| Primer episodio con 210 | **76** |
| **Test (100 eps, ε=0)** | **210.0** — política perfecta |
| Tiempo de entrenamiento | ~30 min en Colab (CPU) |

![Curva de recompensa de DQN en CartPole](/laboratorios/lab-31/reward-curve.png)

### Las lecciones del lab

1. **DQN mejora 10× sobre el azar** (210 vs 21) y aprende una política *perfecta* (210/210 en test), no solo "buena".
2. **La exploración termina antes de resolver.** `epsilon` decae por *paso de gradiente*, no por episodio: hacia el episodio ~65 ya está en el piso (0.001), mientras que la tarea se resuelve en el 85. El tramo final de mejora ocurre con ε≈0 — el mérito es del **replay buffer**, no de más exploración.
3. **Sin experience replay, DQN colapsa** por debajo del azar (ablation medida): la des-correlación de transiciones es lo que hace converger el SGD.
4. **Sin las velocidades en el estado (POMDP), DQN no resuelve** la tarea: rompe la propiedad de Markov y el mismo input exige acciones opuestas. Validación empírica directa de la pregunta 2 de la tarea.
5. **La loss no baja monótonamente** (sube y baja) porque el target se mueve — *bootstrapping*. Normal en RL, y la razón por la que la target network importa.

## Bloques del lab

{{< cards >}}
  {{< card link="01-marco-rl-y-cartpole" title="El marco RL y CartPole" subtitle="MDP, política, retorno descontado, Q-values óptimos, ecuación de Bellman, Gymnasium, el ambiente CartPole y el baseline aleatorio" icon="globe-alt" >}}
  {{< card link="02-dqn-implementacion" title="Implementación de DQN" subtitle="La clase DeepQNetwork: red Q, replay buffer, política ε-greedy, target de Bellman, el loop de entrenamiento y la fase de test contra datos reales" icon="code" >}}
  {{< card link="03-experimentos-y-analisis" title="Experimentos propios y análisis" subtitle="4 ablations medidas: enmascarar velocidades (POMDP), quitar el replay buffer, agregar target network, y sensibilidad a γ y ε-decay" icon="beaker" >}}
  {{< card link="04-actividades" title="Tarea: las dos preguntas" subtitle="De dónde salen los datos de entrenamiento (replay buffer + bootstrapping) y por qué el estado necesita las velocidades (Markov)" icon="academic-cap" >}}
{{< /cards >}}

## Papers relacionados

{{< cards >}}
  {{< card link="/papers/dqn-nature-mnih-2015" title="DQN: Human-level (Nature 2015)" subtitle="Mnih et al. — el paper de referencia del lab: target network, 49 juegos Atari a nivel humano" icon="document-text" >}}
  {{< card link="/papers/dqn-atari-mnih-2013" title="DQN: Atari (2013)" subtitle="Mnih et al. — deep RL desde pixeles, la introducción del experience replay" icon="document-text" >}}
  {{< card link="/papers/double-dqn-van-hasselt-2015" title="Double DQN (2015)" subtitle="van Hasselt et al. — corrige la sobreestimación del max que discute el lab" icon="document-text" >}}
  {{< card link="/papers/per-schaul-2015" title="Prioritized Experience Replay (2015)" subtitle="Schaul et al. — muestrear el replay por TD-error, mejora directa del buffer del lab" icon="document-text" >}}
  {{< card link="/papers/q-learning-watkins-1992" title="Q-Learning (1992)" subtitle="Watkins & Dayan — el algoritmo fundacional que DQN lleva a redes profundas" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/aprendizaje-reforzado" title="Aprendizaje Reforzado" subtitle="MDP, Bellman, Q-Learning, DQN, policy gradient, actor-critic, PPO, AlphaGo" icon="book-open" >}}
  {{< card link="/fundamentos/rlhf" title="RLHF" subtitle="El mismo RL que aquí resuelve CartPole, aplicado (vía PPO) a alinear LLMs" icon="book-open" >}}
{{< /cards >}}

## Referencias e implementaciones de producción

El notebook cierra remitiendo a tres repositorios "para implementaciones eficientes de DQN y otros algoritmos". Son el puente entre los papers de la clase y el RL de producción — vale la pena situarlos y actualizarlos:

| Repositorio | Framework | Algoritmos | Rol y estado |
|---|---|---|---|
| [`openai/baselines`](https://github.com/openai/baselines) | TensorFlow 1.x | DQN, PPO, A2C, ACER, ACKTR, TRPO, DDPG, GAIL, HER | El conjunto de referencia histórico de OpenAI. **En modo mantenimiento** desde ~2020. Su **sucesor de facto es [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** (PyTorch): la cadena fue `baselines` → `stable-baselines` → `stable-baselines3`. Hoy se recomienda SB3, no este. |
| [`astooke/rlpyt`](https://github.com/astooke/rlpyt) | PyTorch | DQN + **Double** + **Dueling** + **Categorical/Rainbow** + **R2D2** + **PER**; A2C, PPO; DDPG, TD3, SAC | Framework de Adam Stooke (Berkeley) que unifica las tres familias de RL model-free. **Implementa exactamente las mejoras sobre DQN que estudiamos** ([Double](/papers/double-dqn-van-hasselt-2015), [Dueling](/papers/dueling-dqn-wang-2015), [PER](/papers/per-schaul-2015)) y que este lab *no* tiene. Es el "DQN completo". |
| [`lcswillems/torch-ac`](https://github.com/lcswillems/torch-ac) | PyTorch | A2C (A3C síncrono), PPO | Paquete minimalista y legible de Lucas Willems, con políticas recurrentes + multiprocessing, pensado para MiniGrid. El complemento *policy-based* del lab (DQN es *value-based*). |

**Por qué importan para el lab:**

1. **Cierran el arco value-based → policy-based.** El lab hace DQN (value-based); torch-ac y las partes PG de rlpyt son PPO/A2C (policy-based), el salto que conecta con [PPO](/papers/ppo-schulman-2017) y [RLHF](/fundamentos/rlhf).
2. **rlpyt materializa lo "ausente" del notebook:** target network, Double, Dueling y PER — todo lo que el lab marca como simplificado está implementado ahí.
3. **La recomendación está desactualizada:** hoy el punto de partida es **Stable-Baselines3**, no `openai/baselines`.

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-31" title="Clase 31 - Teoría" subtitle="Paradigma RL, MDP, Q-Learning, Deep Q-Learning, policy gradient, PPO, AlphaGo" icon="academic-cap" >}}
  {{< card link="/clases/clase-31/practica/02-dqn-desde-cero" title="Práctica: DQN desde cero (triple framework)" subtitle="El mismo DQN en PyTorch, TensorFlow y JAX, con target network incluida" icon="code" >}}
  {{< card link="/dominios/robotica" title="Dominio: Robótica / RL" subtitle="Línea de tiempo: de Q-learning y TD-Gammon a DQN, AlphaGo y RLHF" icon="globe-alt" >}}
  {{< card link="/laboratorios/lab-30" title="Lab 30 - Modelos con memoria externa (anterior)" subtitle="Key-Value Memory Networks sobre WikiMovies QA" icon="arrow-left" >}}
  {{< card link="/laboratorios/lab-32" title="Lab 32 - Olvido Catastrófico (siguiente)" subtitle="Aprendizaje incremental: Naive vs Rehearsal vs EWC sobre Permuted MNIST. El último lab del curso" icon="arrow-right" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Recorrido celda a celda de las 38 celdas del notebook + entrenamiento real ejecutado (DQN resuelve CartPole en 85 episodios, test 210/210) + 4 experimentos propios medidos (enmascarar velocidades, quitar replay, agregar target network, sensibilidad a hiperparámetros). Las 2 preguntas de la tarea resueltas (dentro del notebook) y validadas empíricamente. Notebook Colab ejecutado con la curva de recompensa embebida; análisis de referencias de producción (baselines/rlpyt/torch-ac).
