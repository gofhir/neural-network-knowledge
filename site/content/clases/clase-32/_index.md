---
title: "Clase 32 - Olvido Catastrófico y Aprendizaje Continuo"
weight: 320
sidebar:
  open: true
---

**Profesor:** Alain Raymond
**Curso 3 / Tópicos de profundización:** Relacional, GANs, RL, Meta-Learning, Razonamiento y Memoria

Clase final del curso, sobre un problema muy real del uso continuo de modelos: el **olvido catastrófico**. Cuando una red entrenada en una tarea se reentrena en una tarea nueva, **olvida** lo aprendido — porque optimizar solo los datos nuevos mueve los pesos fuera del óptimo de los anteriores. El **aprendizaje continuo (continual learning)** busca que el modelo incorpore datos nuevos sin degradar lo viejo y sin tener que reentrenar con todo desde cero. La clase formaliza el problema (el dilema **estabilidad-plasticidad**), define los **tres escenarios** —Task / Domain / Class Incremental— y recorre las **tres familias de métodos**: **regularización** (EWC, LwF), **memoria/replay** (Experience Replay, GEM, iCaRL) y **arquitectura** (Progressive Nets, PiggyBack, SupSup, HAT, L2P).

La clase cierra el bloque relacional conectando con la [destilación](/clases/clase-22) (LwF), el [prompting y los Transformers](/clases/clase-20) (L2P), y el [transfer learning](/fundamentos/transfer-learning). El mensaje final: **no es un problema resuelto**.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 51 diapositivas: motivación, aprendizaje incremental, los 3 escenarios, métodos (regularización/memoria/arquitectura)" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: ACC/BWT/FWT, EWC y Fisher, Synaptic Intelligence, distillation/iCaRL, GEM como QP, máscaras, los 3 escenarios" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="EWC y Experience Replay desde cero en triple framework (PyTorch, TensorFlow, JAX)" icon="code" >}}
  {{< card link="/laboratorios/lab-32" title="Laboratorio 32" subtitle="Permuted MNIST: mide el olvido y compara Naive vs Rehearsal vs EWC. Curva de λ y las 4 actividades" icon="beaker" >}}
  {{< card link="/clases/clase-31" title="Clase anterior: Aprendizaje Reforzado" subtitle="Q-Learning, DQN, policy gradient" icon="arrow-left" >}}
  {{< card link="/clases/clase-26" title="Relacionada: Meta-aprendizaje" subtitle="Aprender a aprender, pocos datos por tarea" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/aprendizaje-continuo" title="Aprendizaje Continuo y Olvido Catastrófico" subtitle="Los 3 escenarios, las 3 familias de métodos, métricas ACC/BWT/FWT" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer Learning" subtitle="El campo vecino: reutilizar conocimiento previo" icon="book-open" >}}
  {{< card link="/fundamentos/memory-augmented-networks" title="Memory-Augmented Networks" subtitle="Memoria externa, relacionada con los métodos de replay" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/ewc-kirkpatrick-2017" title="EWC (2017)" subtitle="Kirkpatrick et al. — regularización con la matriz de Fisher" icon="document-text" >}}
  {{< card link="/papers/lwf-li-2016" title="Learning without Forgetting (2016)" subtitle="Li & Hoiem — distillation para preservar tareas viejas" icon="document-text" >}}
  {{< card link="/papers/gem-lopez-paz-2017" title="GEM (2017)" subtitle="Lopez-Paz & Ranzato — memoria episódica + proyección de gradientes" icon="document-text" >}}
  {{< card link="/papers/piggyback-mallya-2018" title="Piggyback (2018)" subtitle="Mallya et al. — máscaras binarias sobre una red congelada" icon="document-text" >}}
  {{< card link="/papers/supsup-wortsman-2020" title="SupSup (2020)" subtitle="Wortsman et al. — supermascaras, infiere la tarea sin task ID" icon="document-text" >}}
  {{< card link="/papers/hat-serra-2018" title="HAT (2018)" subtitle="Serrà et al. — atención dura por tarea sobre las unidades" icon="document-text" >}}
  {{< card link="/papers/l2p-wang-2022" title="L2P (2022)" subtitle="Wang et al. — prompts aprendibles sobre un Transformer congelado" icon="document-text" >}}
  {{< card link="/papers/continual-survey-mundt-2020" title="Survey (Mundt 2020)" subtitle="Mundt et al. — panorama del continual learning (citado en la clase)" icon="document-text" >}}
{{< /cards >}}

## Papers canónicos (complementarios)

{{< cards >}}
  {{< card link="/papers/icarl-rebuffi-2017" title="iCaRL (2017)" subtitle="Rebuffi et al. — exemplars + distillation + NME, baseline class-incremental" icon="document-text" >}}
  {{< card link="/papers/synaptic-intelligence-zenke-2017" title="Synaptic Intelligence (2017)" subtitle="Zenke et al. — importancia de pesos online, hermano de EWC" icon="document-text" >}}
  {{< card link="/papers/progressive-nets-rusu-2016" title="Progressive Neural Networks (2016)" subtitle="Rusu et al. — columnas nuevas por tarea, cero olvido por diseño" icon="document-text" >}}
  {{< card link="/papers/three-scenarios-van-de-ven-2019" title="Three Scenarios (2019)" subtitle="van de Ven & Tolias — la taxonomía Task/Domain/Class Incremental" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/ingenieria-ml" title="Dominio: Ingeniería de ML" subtitle="El uso continuo de modelos en producción: drift, reentrenamiento, continual learning" icon="globe-alt" >}}
{{< /cards >}}
