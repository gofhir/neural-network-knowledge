---
title: "Cómputo en tiempo de inferencia (test-time compute)"
weight: 115
math: true
---

Durante una década, la receta para mejorar un modelo fue **escalar el entrenamiento**: más datos, más parámetros, más cómputo de *training*. El **cómputo en tiempo de inferencia** (test-time compute) es el eje complementario que definió a los "modelos de razonamiento" de 2024-2025 (o1, DeepSeek-R1): en lugar de —o además de— entrenar más, **gastar más cómputo cuando el modelo responde**, dándole espacio para pensar, muestrear alternativas y verificar. Este fundamento acompaña a la [Clase 34](/clases/clase-34) y desarrolla el paradigma: del muestreo repetido (Pass@k) al aprendizaje reforzado con recompensa verificable, y el debate sobre qué es lo que ese RL realmente aporta. Es el complemento de [Chain-of-Thought](/fundamentos/chain-of-thought) y [Razonamiento en IA](/fundamentos/razonamiento).

---

## 1. La idea: cambiar cómputo por acierto en inferencia

Las técnicas de [Chain-of-Thought](/fundamentos/chain-of-thought) ya insinuaban el patrón: dejar que el modelo escriba más pasos (más tokens) mejora el razonamiento. El test-time compute lo lleva al centro: **el desempeño de un LLM se puede aumentar gastando más cómputo en inferencia**, sin cambiar sus pesos. Hay dos formas complementarias:

- **En serie** (razonar más largo): cadenas de razonamiento más extensas, con auto-verificación y reflexión.
- **En paralelo** (razonar más veces): muestrear muchas respuestas y seleccionar/agregar (self-consistency, Pass@k).

---

## 2. Pass@k y el poder del muestreo repetido

Si a un LLM le pides **una** respuesta puede fallar, pero si le pides **muchas**, la probabilidad de que *alguna* sea correcta crece. La métrica **Pass@k** mide justamente eso: la fracción de problemas resueltos por **al menos una** de $k$ muestras. Brown et al. (2024), en *Large Language Monkeys*, cuantifican esta **cobertura** (*coverage*) como función de $k$ y encuentran que crece de forma **predecible** —a menudo casi log-lineal— sobre cuatro órdenes de magnitud (de 1 a 10.000 muestras):

$$
c(k) \;\approx\; \exp\!\big(a\, k^{b}\big),
$$

una ley de potencia exponenciada. El resultado es sorprendente: modelos pequeños y baratos, muestreados miles de veces, pueden **cubrir** más problemas que un modelo caro de una sola pasada (p.ej. en SWE-bench, un modelo pasa de ~16% con una muestra a ~56% con cientos). → [análisis](/papers/large-language-monkeys-brown-2024)

{{< concept-alert type="advertencia" >}}
**Cobertura ≠ acierto.** Que *exista* una muestra correcta entre 10.000 no sirve si no sabes **cuál** es. Aquí aparece la distinción más importante del paradigma: los problemas **verificables** (código con tests, matemáticas con checker) permiten *seleccionar* automáticamente la muestra correcta; los **no verificables** (donde falta un buen verificador) dejan la cobertura como una promesa incumplida. En estos, la selección por voto mayoritario o modelo de recompensa **se estanca** mucho antes que la cobertura.
{{< /concept-alert >}}

---

## 3. El momento o1 y R1: RL con recompensa verificable

El salto de 2024-2025 fue **internalizar** el razonamiento largo mediante **aprendizaje reforzado**. La idea (o1 de OpenAI, **DeepSeek-R1**): generar muchas trazas de razonamiento por problema y, usando RL, **premiar las trazas que llegan al resultado correcto**. La recompensa no viene de un modelo de preferencias humanas (como en [RLHF](/fundamentos/rlhf)) sino de un **verificador objetivo** —el resultado es correcto o no—, por lo que solo aplica a **problemas verificables** (matemáticas, código, lógica).

**DeepSeek-R1-Zero** llevó esto al extremo: RL puro sobre el modelo base, **sin fine-tuning supervisado previo**, con recompensas basadas en reglas (corrección + formato). Durante el entrenamiento **emergieron** comportamientos de razonamiento —cadenas cada vez más largas, auto-verificación, reflexión— y el célebre **"aha moment"**, donde el modelo aprende a detenerse y reconsiderar. DeepSeek-R1 usa **GRPO** (Group Relative Policy Optimization), una variante de [PPO](/papers/ppo-schulman-2017) que estima la ventaja **normalizando dentro de un grupo** de respuestas muestreadas, sin red de valor:

$$
A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}.
$$

Como R1-Zero producía texto poco legible (mezcla de idiomas), DeepSeek-R1 añade un pipeline multi-etapa (cold-start SFT → RL → rechazo/SFT → RL final) y **destila** el razonamiento a modelos pequeños. → [análisis](/papers/deepseek-r1-2025)

---

## 4. El debate: ¿RL crea razonamiento o solo lo reordena?

El entusiasmo se topó con una pregunta incómoda, planteada por Yue et al. (2025): **¿el RL realmente enseña a razonar, o solo hace más probables patrones que el modelo base ya tenía?** Su evidencia usa Pass@k: el modelo entrenado con RL supera al base en **Pass@1** (una muestra), pero al aumentar $k$ el **modelo base iguala o supera** al modelo con RL en cobertura. Interpretación:

{{< concept-alert type="clave" >}}
El RL con recompensa verificable **estrecha la distribución** hacia los caminos correctos que el modelo base **ya podía generar** —mejora la *eficiencia de muestreo*, no la *frontera de capacidades*. Como dice la clase: *RL no enseña nuevos comportamientos (como la verificación), sino que son patrones aprendidos durante el pre-entrenamiento, y el RL hace que se muestreen con mayor probabilidad.* La destilación, en cambio, sí puede **introducir** patrones nuevos desde un modelo más capaz.
{{< /concept-alert >}}

Esto no anula el valor del RL (un modelo que acierta en la primera muestra es mucho más útil en la práctica), pero **recalibra** la narrativa: la capacidad de razonamiento reside en gran medida en el **pre-entrenamiento**. → [análisis](/papers/rl-reasoning-yue-2025)

Trabajos como **Logic-RL** (Xie et al., 2025) refuerzan el marco usando entornos sintéticos verificables (puzzles lógicos *knights and knaves*) para estudiar de forma controlada cómo emergen —y hasta dónde transfieren— estos comportamientos. → [análisis](/papers/logic-rl-xie-2025)

---

## 5. La precaución: robustez y generalización

El paradigma sigue teniendo grietas que la clase subraya:

- **Robustez.** Benchmarks como **Math-Perturb** perturban problemas para distinguir razonamiento de memorización; los modelos caen notablemente, señal de que parte del "razonamiento" es recuperación de plantillas.
- **Generalización frágil.** Variantes triviales de acertijos clásicos (Monty Hall, el cruce del río) hacen fallar incluso a modelos de razonamiento, que recuperan la solución memorizada del original.
- **Solo dominios verificables.** Todo el andamiaje del RL con recompensa verificable depende de poder **comprobar** la respuesta. En dominios sin ground-truth objetivo, no aplica directamente.

---

## 6. Relevancia para salud y sistemas clínicos

El test-time compute ofrece una palanca tentadora en medicina: gastar más cómputo en los casos difíciles (más deliberación, más muestras) para mejorar la calidad de una decisión. Pero las advertencias de este fundamento son especialmente agudas en el contexto clínico. Primero, **cobertura ≠ acierto**: muestrear 100 diagnósticos posibles no ayuda si no hay un verificador confiable que seleccione el correcto —y en medicina el ground-truth suele ser incierto o diferido. Segundo, el RL con recompensa verificable **no aplica** a la mayoría de las decisiones clínicas, que carecen de un checker objetivo como el de un problema de matemáticas. Tercero, el hallazgo de Yue et al. recuerda que la capacidad de razonamiento clínico de un modelo viene de su **pre-entrenamiento**, no de un afinado de razonamiento que solo la reordena. La conclusión pragmática: usar más cómputo de inferencia donde exista una forma de **verificar**, y desconfiar de la fluidez donde no la haya.

---

## Referencias

- Brown, B. et al. (2024). *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*. arXiv:2407.21787.
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.
- Xie, T. et al. (2025). *Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning*. arXiv:2502.14768.
- Yue, Y. et al. (2025). *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* arXiv:2504.13837.
- Fundamentos hermanos: [Razonamiento en IA](/fundamentos/razonamiento) · [Chain-of-Thought](/fundamentos/chain-of-thought) · [RLHF](/fundamentos/rlhf).
