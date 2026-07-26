---
title: "Clase 34 - Razonamiento"
weight: 340
sidebar:
  open: true
---

**Profesor:** Sebastián Amenábar
**Curso 3 / Tópicos de profundización:** Relacional, GANs, RL, Meta-Learning, Razonamiento y Memoria

Clase panorámica sobre la frontera más discutida de la IA: el **razonamiento**. Parte de la cognición —la **escalera de causalidad** de Judea Pearl (asociación, intervención, contrafactuales), la **abstracción**, la **sistematicidad** y los **sistemas 1 y 2**— para argumentar que el deep learning es esencialmente un **sistema asociativo** que memoriza patrones. Luego recorre cómo dotar de razonamiento a las redes: primero con **estructura** ([memoria externa](/fundamentos/redes-de-memoria), redes composicionales como MAC), y en la era de los LLMs con **prompting e inferencia** —del *scratchpad* al **Chain-of-Thought**, la self-consistency, el Tree-of-Thoughts, y el **test-time compute** de los modelos de razonamiento (o1, DeepSeek-R1). Cierra con cautela: los límites de generalización y robustez que persisten (Math-Perturb, ARC, acertijos perturbados).

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las diapositivas: causalidad, abstracción/sistematicidad, System 1/2, neuro-simbólico, CoT, test-time compute, límites" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: jerarquía causal, CoT/self-consistency/ToT, Pass@k y leyes de cobertura, GRPO, la crítica de Yue, la inteligencia de Chollet" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Chain-of-Thought y Self-Consistency desde cero + un mini-verificador, en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-34" title="Laboratorio: Razonamiento aplicado" subtitle="Tool use, PEFT/LoRA, optimización de prompts (DSPy/GEPA) y multimodal" icon="variable" >}}
  {{< card link="/clases/clase-35" title="Clase siguiente: Introducción al Análisis de Audio" subtitle="Fourier, FFT, sampling, STFT, MFCC" icon="arrow-right" >}}
  {{< card link="/clases/clase-33" title="Clase anterior: Imitación e IRL" subtitle="Aprendizaje por imitación, IRL, generalización en RL" icon="arrow-left" >}}
  {{< card link="/clases/clase-30" title="Relacionada: Modelos con memoria externa" subtitle="NTM, DNC — dar estructura a la red para razonar" icon="academic-cap" >}}
  {{< card link="/clases/clase-20" title="Relacionada: BERT/GPT/ChatGPT (RLHF)" subtitle="In-context learning, RLHF, la base de los LLMs" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/razonamiento" title="Razonamiento en IA" subtitle="Escalera de causalidad, abstracción, sistematicidad, System 1/2, neuro-simbólico" icon="book-open" >}}
  {{< card link="/fundamentos/chain-of-thought" title="Chain-of-Thought y prompting para razonar" subtitle="Scratchpad, CoT, self-consistency, Tree-of-Thoughts" icon="book-open" >}}
  {{< card link="/fundamentos/test-time-compute" title="Cómputo en tiempo de inferencia" subtitle="Pass@k, muestreo repetido, RL con recompensa verificable, o1/R1" icon="book-open" >}}
  {{< card link="/fundamentos/in-context-learning" title="In-Context Learning" subtitle="El comportamiento emergente base de los LLMs" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/arc-chollet-2019" title="On the Measure of Intelligence — ARC (2019)" subtitle="Chollet — inteligencia como eficiencia; el benchmark que resiste la memorización" icon="document-text" >}}
  {{< card link="/papers/mac-hudson-2018" title="Compositional Attention Networks — MAC (2018)" subtitle="Hudson & Manning — razonamiento composicional diferenciable en CLEVR" icon="document-text" >}}
  {{< card link="/papers/scratchpad-nye-2021" title="Show Your Work: Scratchpads (2021)" subtitle="Nye et al. — pasos intermedios; el precursor de CoT" icon="document-text" >}}
  {{< card link="/papers/emergent-abilities-wei-2022" title="Emergent Abilities of LLMs (2022)" subtitle="Wei et al. — capacidades que aparecen solo con escala" icon="document-text" >}}
  {{< card link="/papers/chain-of-thought-wei-2022" title="Chain-of-Thought Prompting (2022)" subtitle="Wei et al. — razonar paso a paso; GSM8K 18→57" icon="document-text" >}}
  {{< card link="/papers/bbh-suzgun-2022" title="BIG-Bench Hard + CoT (2022)" subtitle="Suzgun et al. — CoT supera al humano promedio en 17/23 tareas" icon="document-text" >}}
  {{< card link="/papers/ye-durrett-explanations-2022" title="The Unreliability of Explanations (2022)" subtitle="Ye & Durrett — una cadena fluida puede no ser factual" icon="document-text" >}}
  {{< card link="/papers/self-consistency-wang-2022" title="Self-Consistency (2022)" subtitle="Wang et al. — muestrear muchas cadenas y votar" icon="document-text" >}}
  {{< card link="/papers/tree-of-thoughts-yao-2023" title="Tree of Thoughts (2023)" subtitle="Yao et al. — CoT como búsqueda en árbol con backtracking" icon="document-text" >}}
  {{< card link="/papers/large-language-monkeys-brown-2024" title="Large Language Monkeys (2024)" subtitle="Brown et al. — Pass@k y el rol del verificador" icon="document-text" >}}
  {{< card link="/papers/deepseek-r1-2025" title="DeepSeek-R1 (2025)" subtitle="RL con recompensa verificable; GRPO; el aha-moment" icon="document-text" >}}
{{< /cards >}}

## Papers complementarios

{{< cards >}}
  {{< card link="/papers/logic-rl-xie-2025" title="Logic-RL (2025)" subtitle="Xie et al. — razonamiento con RL basado en reglas sobre puzzles lógicos" icon="document-text" >}}
  {{< card link="/papers/rl-reasoning-yue-2025" title="¿El RL incentiva el razonamiento? (2025)" subtitle="Yue et al. — la crítica: RL reordena, no expande la frontera" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/texto" title="Dominio: Texto / NLP" subtitle="De los embeddings a los LLMs y el razonamiento" icon="globe-alt" >}}
{{< /cards >}}
