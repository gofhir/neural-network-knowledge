---
title: "Teoría - Razonamiento"
weight: 10
math: true
---

> **Recorrido de la Clase 34** del Diplomado IA UC (Sebastián Amenábar). Una clase panorámica sobre la frontera más discutida de la IA: el **razonamiento**. Arranca en la cognición —los tipos de razonamiento de Judea Pearl, la abstracción, la sistematicidad, los sistemas 1 y 2— para explicar por qué el deep learning es esencialmente un **sistema asociativo** que memoriza patrones. Luego recorre los intentos de dotar de razonamiento a las redes: primero con **estructura** (memoria externa, redes composicionales) y después, en la era de los LLMs, con **técnicas de prompting e inferencia** (Chain-of-Thought, test-time compute, modelos de razonamiento como o1 y DeepSeek-R1). Cierra con una nota de cautela: los límites que persisten.

---

## 1. Tipos de razonamiento: la escalera de causalidad

La clase abre con la **escalera de la causalidad** de Judea Pearl, tres niveles de preguntas cada vez más profundas:

- **Razonamiento asociativo** — basado en **correlaciones**. ¿Qué me dice observar $X$ sobre $Y$?
- **Razonamiento intervencional** (*hacer*) — ¿qué pasará si **realizo** cierta acción?
- **Razonamiento contrafactual** (*imaginar*) — dados los resultados de una acción, ¿qué habría pasado si hubiera actuado distinto? ¿Qué sé ahora que antes no sabía?

Casi todo el machine learning vive en el primer peldaño. Subir a intervención y contrafactuales es el territorio del razonamiento genuino. Se desarrolla en el fundamento [Razonamiento en IA](/fundamentos/razonamiento).

---

## 2. Las habilidades para razonar

La clase identifica dos capacidades cognitivas de base:

**Construcción de abstracciones.** Extraer el concepto subyacente a partir de instancias específicas (de perros concretos, el concepto "perro"; de ahí, la abstracción jerárquica "raza"). Una abstracción **captura lo esencial y omite lo irrelevante**, facilitando el procesamiento. *El conocimiento son abstracciones.*

**Sistematicidad.** Aplicar esas abstracciones en **contextos distintos** a los del entrenamiento y **componer conceptos** ("un mono tocando guitarra en la playa"). Las redes muestran *cierta* sistematicidad —los generadores de imágenes componen conceptos— pero **incompleta**: fallan al pedirles un reloj marcando una hora específica, o combinaciones forma-color no vistas.

Juntas, abstracción y sistematicidad permiten construir **modelos causales**: el ejemplo de la clase razona sobre por qué no crecieron las plantas (fertilizante, pH, absorción de nitrógeno) —un modelo que soporta intervención y contrafactuales.

{{< concept-alert type="recordar" >}}
Borges lo capturó en *Funes el memorioso*: Funes recordaba todo, pero *"no era muy capaz de pensar. Pensar es olvidar diferencias, es generalizar, abstraer."* Memorizar no es razonar; razonar exige **descartar lo irrelevante** para generalizar.
{{< /concept-alert >}}

El **lenguaje** aparece como habilitador del pensamiento abstracto y los escenarios hipotéticos (ilustrado con *El hombre imaginario* de Nicanor Parra), y se subraya que **aprender rápido está fuertemente relacionado con razonar** —el vínculo con el [meta-aprendizaje](/fundamentos/meta-aprendizaje).

---

## 3. Deep learning como sistema asociativo

La tesis central de la primera mitad: el deep learning es **memorización de patrones** (Sistema 1, reactivo), no razonamiento deliberado (Sistema 2). La evidencia:

- En **aprendizaje reforzado**, el DRL funciona bien cuando la recompensa está **cerca de la acción** (juegos asociativos), pero fracasa con recompensa **esporádica** como *Montezuma's Revenge*. Técnicas como **Go-Explore** —guardar estados prometedores y volver a ellos— inyectan una estructura causal ("¿qué hubiera pasado si...?") de forma **externa**, no intrínseca al modelo.
- La distinción **Sistema 1** (reactivo, rápido, automático) vs **Sistema 2** (deliberado, lento, con esfuerzo) enmarca todo: el DL clásico es Sistema 1; el razonamiento exige Sistema 2.

---

## 4. Dar estructura a la red: el enfoque neuro-simbólico

Primer intento de aumentar el razonamiento: **darle a la red la estructura de un computador clásico**, que sí generaliza sistemáticamente (un algoritmo de grafos funciona en cualquier grafo).

- **Neural Turing Machine / Differentiable Neural Computer** (Graves, 2016): acoplar una red a una memoria direccionable. Cubiertas en la [Clase 30](/clases/clase-30) ([redes de memoria](/fundamentos/redes-de-memoria)).
- **Compositional Attention Networks (MAC)** (Hudson & Manning, 2018): descomponer el razonamiento visual en pasos de atención (control, lectura, escritura), logrando razonamiento composicional diferenciable en **CLEVR**. → [paper](/papers/mac-hudson-2018)
- **CLEVR-CoGenT** mide la sistematicidad: entrenar con ciertas combinaciones forma-color y evaluar con otras. Los modelos **tropiezan** al componer atributos no vistos juntos —evidencia de sistematicidad incompleta.

---

## 5. LLMs: el giro moderno

Los LLMs pre-entrenados ([GPT-3](/papers/gpt-3-brown-2020)) tienen excelente desempeño en tareas diversas y exhiben comportamientos **emergentes** como el [in-context learning](/fundamentos/in-context-learning). Pero **fallan en aritmética simple** (sumar dígitos) —señal de que memorizan patrones más que ejecutar algoritmos. La pregunta: ¿se puede inducir razonamiento sin cambiar la arquitectura?

### 5.1 Bloc de notas y Chain-of-Thought

- **Scratchpad** (Nye, 2021): guiar al modelo para que **emita los pasos intermedios** de un cálculo antes de la respuesta. *"Como al mono recolectando cocos, lo guiamos para que aprenda el algoritmo."* → [paper](/papers/scratchpad-nye-2021)
- **Chain-of-Thought** (Wei, 2022): lograr lo mismo **sin entrenar**, solo con el prompt —dar ejemplos few-shot con razonamiento paso a paso. En GSM8K, PaLM 540B salta de ~18% a ~57%. → [paper](/papers/chain-of-thought-wei-2022)
- **BIG-Bench Hard** (Suzgun, 2022): con CoT, Codex supera el desempeño humano **promedio** en 17 de 23 tareas donde antes ningún LLM lo lograba. → [paper](/papers/bbh-suzgun-2022)

{{< concept-alert type="clave" >}}
El Chain-of-Thought es una **habilidad emergente**: solo funciona a partir de cierta escala (~100B parámetros). En modelos pequeños, pedir el razonamiento paso a paso **empeora** el resultado. Ver [habilidades emergentes](/papers/emergent-abilities-wei-2022) y el fundamento [Chain-of-Thought](/fundamentos/chain-of-thought).
{{< /concept-alert >}}

### 5.2 ¿De dónde surge el CoT?

El CoT empezó a ser marcadamente efectivo en **text-davinci-002** (Ye & Durrett, 2022, documentan el salto). Una hipótesis discutida en la clase lo atribuye al **entrenamiento con gran cantidad de código**, donde el razonamiento estructurado es omnipresente —hipótesis plausible pero **no probada por ese paper**. Ye & Durrett además advierten que las explicaciones generadas pueden **no ser factuales** ni implicar la predicción: una cadena fluida no garantiza razonamiento correcto. → [paper](/papers/ye-durrett-explanations-2022)

### 5.3 InstructGPT, RLHF y sicofancia

El [RLHF](/fundamentos/rlhf) (InstructGPT, cubierto en la [Clase 20](/clases/clase-20)) alinea los LLMs con preferencias humanas, pero introduce la **sicofancia**: el modelo tiende a decir lo que el usuario quiere oír, un efecto colateral de optimizar preferencias humanas.

---

## 6. Potenciar LLMs con más cómputo de inferencia

### 6.1 De la cadena al árbol y al muestreo

- **Tree-of-Thought y Agent-Debate** (Yao, 2023): generalizar el CoT a una **búsqueda en árbol** con auto-evaluación y backtracking; o hacer que varios agentes debatan. → [paper](/papers/tree-of-thoughts-yao-2023)
- **Pass@k** (Brown, 2024, *Large Language Monkeys*): si muestreamos muchas respuestas, es probable que **alguna** sea correcta. La cobertura crece de forma predecible con el número de muestras. → [paper](/papers/large-language-monkeys-brown-2024)

### 6.2 El momento o1 y R1

**Test-time compute**: generar muchas respuestas por problema y usar **Aprendizaje Reforzado** para dar recompensa positiva a las trazas de razonamiento que **llegan al resultado correcto**. Es aplicable solo sobre **problemas verificables** (matemáticas, código). Es la receta de **ChatGPT o1** y **DeepSeek-R1**. → [paper](/papers/deepseek-r1-2025)

- **aha-moments** (Logic-RL, Xie 2025): durante el RL emergen comportamientos de auto-verificación y reflexión. → [paper](/papers/logic-rl-xie-2025)
- **La precaución** (Yue et al., 2025): la evidencia apunta a que el RL **no enseña comportamientos nuevos** (como la verificación), sino que son patrones que el modelo **ya aprendió en el pre-entrenamiento**, y el RL los hace más probables porque son útiles. → [paper](/papers/rl-reasoning-yue-2025)

Se desarrolla en el fundamento [Test-time compute](/fundamentos/test-time-compute).

{{< concept-alert type="advertencia" >}}
**Diferencia con RLHF.** El RL para razonamiento usa una **recompensa verificable** (el resultado es correcto o no) sobre problemas objetivos; el [RLHF](/fundamentos/rlhf) usa un **modelo de recompensa de preferencias humanas**. El primero solo aplica donde hay un verificador; el segundo puede alinear comportamiento subjetivo (a costa de efectos como la sicofancia).
{{< /concept-alert >}}

---

## 7. Los límites persisten

La clase cierra con cautela:

- **Robustez.** **Math-Perturb** perturba problemas matemáticos para medir si el modelo razona o memoriza; grandes caídas delatan memorización.
- **Generalización frágil.** Variantes triviales de acertijos clásicos —el problema de **Monty Hall** con puertas transparentes, el cruce del río con un bote de mayor capacidad— hacen que los modelos **recuperen la solución memorizada** en vez de razonar sobre el enunciado real. ChatGPT (con razonamiento) a veces acierta, a veces afirma que no hay solución; distintos modelos fallan de distinta forma.
- **Abstracción genuina.** El **ARC-Challenge** (Chollet, 2019) mide la **eficiencia en adquirir habilidades nuevas** con pocos ejemplos —resistente a la memorización— y sigue siendo difícil para los LLMs. → [paper](/papers/arc-chollet-2019)
- **Comportamientos inesperados.** El caso del *fine-tuning* de GPT-3.5 sobre 140.000 mensajes de Slack ilustra cómo el afinado puede producir conductas no anticipadas.

{{< concept-alert type="clave" >}}
La lección de cierre: una respuesta fluida y bien argumentada **no es** prueba de razonamiento correcto. En dominios de alto riesgo —como la medicina— hay que **verificar** el razonamiento, no confiar en su superficie.
{{< /concept-alert >}}

---

## 8. Recursos y cierre

La clase remite a recursos útiles: la [guía de prompting](https://www.promptingguide.ai/), *Gorilla* (LLMs conectados a APIs masivas), el análisis de Yao Fu sobre el origen de las capacidades emergentes y el rol del código, y *Towards Complex Reasoning*. El [laboratorio](/laboratorios/lab-34) aterriza estos conceptos en práctica: uso de herramientas (tool use), fine-tuning eficiente (PEFT/LoRA), optimización de prompts (DSPy/GEPA) y tareas multimodales.

---

**Ver también:** [Clase 34 - Profundización](/clases/clase-34/profundizacion) · [Clase 34 - Práctica](/clases/clase-34/practica) · Fundamentos: [Razonamiento en IA](/fundamentos/razonamiento) · [Chain-of-Thought](/fundamentos/chain-of-thought) · [Test-time compute](/fundamentos/test-time-compute).
