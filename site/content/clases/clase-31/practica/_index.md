---
title: "Practica desde 0 - Aprendizaje Reforzado"
weight: 30
sidebar:
  open: true
---

La clase 31 cubre el **aprendizaje reforzado (RL)**: agentes que aprenden por ensayo y error maximizando recompensa acumulada. Esta práctica construye los **dos pilares** en **mínima escala**: el **Q-Learning tabular** (la base, sobre un gridworld a mano) y el **DQN** (Deep Q-Learning, con red, experience replay y target network, sobre CartPole). El salto entre ambos es el corazón de la clase: cuando la tabla Q no escala, se aproxima con una red neuronal. Cada camino se muestra en **triple framework** (PyTorch, TensorFlow y JAX).

## Caminos

{{< cards >}}
  {{< card link="01-q-learning-tabular-desde-cero" title="01 - Q-Learning tabular desde cero" subtitle="MDP, tabla Q, regla de Bellman y ε-greedy sobre un gridworld (NumPy + las 3 representaciones)" icon="code" >}}
  {{< card link="02-dqn-desde-cero" title="02 - DQN desde cero" subtitle="Red Q + experience replay + target network sobre CartPole, en PyTorch, TensorFlow y JAX" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 30 - Modelos con memoria externa](../../clase-30) (clase anterior del bloque).
- Nociones de probabilidad y de redes neuronales (MLP, backprop).
- Python intermedio y NumPy; PyTorch básico. Útil: nociones de TensorFlow/Keras y JAX.
- GPU **no necesaria**: el gridworld y CartPole corren en CPU en segundos/minutos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - Q-Learning tabular | NumPy | PyTorch / TensorFlow / JAX (representación de la tabla Q) |
| 02 - DQN | PyTorch 2.x | TensorFlow 2.x, JAX + Flax/optax |

## El hilo conductor

1. **Q-Learning tabular**: aprende una **tabla** Q(s,a) por la regla de Bellman. Exacto y con convergencia probada, pero **no escala** (un valor por cada par estado-acción).
2. **DQN**: reemplaza la tabla por una **red** Q(s,a;θ). Para que el entrenamiento no diverja (la "tríada mortal": aproximación + bootstrapping + off-policy), añade dos estabilizadores: **experience replay** (rompe correlaciones) y **target network** (target estable).

---

**Ver tambien:** [Clase 31 - Teoria](../teoria) · [Clase 31 - Profundizacion](../profundizacion) · Fundamentos: [Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado) · [RLHF](/fundamentos/rlhf).
