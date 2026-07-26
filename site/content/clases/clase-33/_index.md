---
title: "Clase 33 - Aprendizaje por Imitación e IRL"
weight: 330
sidebar:
  open: true
---

**Profesor:** Rodrigo Toro Icarte
**Curso 3 / Tópicos de profundización:** Relacional, GANs, RL, Meta-Learning, Razonamiento y Memoria

Continuación directa de la [Clase 31 (Aprendizaje Reforzado)](/clases/clase-31). Si el RL clásico asume una **recompensa dada** y aprende por ensayo y error, esta clase ataca tres preguntas que ese marco deja abiertas. Primero, **¿generalizan** las políticas aprendidas por RL, o solo memorizan el ambiente de entrenamiento? Segundo, **¿qué hacemos cuando no sabemos definir la recompensa?** —el **aprendizaje reforzado inverso (IRL)** la infiere a partir de demostraciones expertas. Tercero, **¿y si simplemente imitamos al experto?** —el **aprendizaje por imitación** (behavioral cloning, DAgger) convierte el control en aprendizaje supervisado. La clase cierra comparando refuerzo e imitación con el caso de estudio de **AlphaGo Zero**: el RL puro puede superar a los humanos, la imitación rara vez supera a su maestro.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las diapositivas: generalización en RL, IRL, imitación (BC + DAgger), y refuerzo vs. imitación (AlphaGo Zero)" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: IRL (LP de Ng-Russell, feature expectations, MaxEnt), cota O(T²ε) del BC y no-regret de DAgger, GAIL (occupancy matching), generalización" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Behavioral Cloning y DAgger desde cero en triple framework (PyTorch, TensorFlow, JAX)" icon="code" >}}
  {{< card link="/laboratorios/lab-33" title="Laboratorio: DAgger sobre Breakout" subtitle="Imitación de un experto DQN en Atari Breakout con Dataset Aggregation (Gymnasium)" icon="variable" >}}
  {{< card link="/clases/clase-32" title="Clase anterior: Olvido Catastrófico" subtitle="Aprendizaje continuo, EWC, replay" icon="arrow-left" >}}
  {{< card link="/clases/clase-31" title="Relacionada: Aprendizaje Reforzado" subtitle="MDP, Q-Learning, DQN — la base sobre la que se construye esta clase" icon="arrow-left" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/aprendizaje-reforzado-inverso" title="Aprendizaje Reforzado Inverso" subtitle="Inferir la recompensa desde demostraciones: Ng-Russell, apprenticeship, MaxEnt, GAIL" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-por-imitacion" title="Aprendizaje por Imitación" subtitle="Behavioral cloning, distribution shift, DAgger, RL+imitación" icon="book-open" >}}
  {{< card link="/fundamentos/generalizacion-en-rl" title="Generalización en RL" subtitle="Overfitting en deep RL, train/test splits, CoinRun, regularización" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-reforzado" title="Aprendizaje Reforzado" subtitle="MDP, Bellman, Q-Learning, DQN, policy gradient (la base)" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/generalization-rl-witty-2018" title="Generalización en Deep RL (2018)" subtitle="Witty et al. — medir y caracterizar dónde falla un agente" icon="document-text" >}}
  {{< card link="/papers/overfitting-rl-zhang-2018" title="Overfitting en Deep RL (2018)" subtitle="Zhang et al. — el gridworld con train/test splits; memorización" icon="document-text" >}}
  {{< card link="/papers/quantifying-generalization-cobbe-2019" title="Quantifying Generalization: CoinRun (2019)" subtitle="Cobbe et al. — miles de niveles y regularización clásica en RL" icon="document-text" >}}
  {{< card link="/papers/irl-ng-russell-2000" title="Algorithms for Inverse RL (2000)" subtitle="Ng & Russell — el paper fundacional del IRL" icon="document-text" >}}
  {{< card link="/papers/apprenticeship-abbeel-ng-2004" title="Apprenticeship Learning via IRL (2004)" subtitle="Abbeel & Ng — feature expectations; aprender a conducir" icon="document-text" >}}
  {{< card link="/papers/apprenticeship-parking-abbeel-2008" title="Apprenticeship: Parking (2008)" subtitle="Abbeel et al. — aprender la función de costo; estacionar" icon="document-text" >}}
  {{< card link="/papers/dagger-ross-2011" title="DAgger (2011)" subtitle="Ross et al. — el algoritmo del lab; garantía lineal vía no-regret" icon="document-text" >}}
  {{< card link="/papers/gato-reed-2022" title="Gato: A Generalist Agent (2022)" subtitle="Reed et al. — imitación masiva multimodal; 604 tareas" icon="document-text" >}}
  {{< card link="/papers/alphago-zero-silver-2017" title="AlphaGo Zero (2017)" subtitle="Silver et al. — RL puro por self-play supera a la imitación" icon="document-text" >}}
{{< /cards >}}

## Papers canónicos (complementarios)

{{< cards >}}
  {{< card link="/papers/maxent-irl-ziebart-2008" title="Maximum Entropy IRL (2008)" subtitle="Ziebart et al. — resuelve la ambigüedad del IRL con probabilidad" icon="document-text" >}}
  {{< card link="/papers/gail-ho-ermon-2016" title="GAIL (2016)" subtitle="Ho & Ermon — imitación adversaria; el puente IRL↔GAN" icon="document-text" >}}
  {{< card link="/papers/alphago-silver-2016" title="AlphaGo (2016)" subtitle="Silver et al. — la versión que sí partía de imitación humana" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/robotica" title="Dominio: Robótica / RL" subtitle="Línea de tiempo: de Q-learning y DQN a AlphaGo, imitación e IRL" icon="globe-alt" >}}
{{< /cards >}}
