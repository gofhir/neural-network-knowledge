---
title: "DeepSeek-R1: RL para razonamiento (2025)"
weight: 389
math: true
---

{{< paper-card
    title="DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
    authors="DeepSeek-AI"
    year="2025"
    venue="arXiv:2501.12948"
    pdf="/papers/deepseek-r1-2025.pdf" >}}
DeepSeek-R1 es la instancia abierta y documentada del **"momento o1 & R1"**: generar muchas respuestas por problema y usar aprendizaje reforzado para premiar las trazas que llegan al resultado correcto, sobre problemas **verificables por reglas**. Su tesis fuerte —**DeepSeek-R1-Zero**— aplica RL directamente sobre el modelo base (DeepSeek-V3-Base, MoE de 671B con 37B activados) **omitiendo por completo el fine-tuning supervisado (SFT) previo**, con recompensas basadas solo en corrección y formato. De ahí emergen de forma autónoma autoverificación, reflexión, cadenas de pensamiento largas y un célebre *aha moment*. El modelo final DeepSeek-R1 queda **a la par de OpenAI-o1-1217** en matemática y código. Es el ejemplo canónico de la [Clase 34](/clases/clase-34) del [test-time compute](/fundamentos/test-time-compute) inducido por RL con recompensa verificable.
{{< /paper-card >}}

---

## Contexto: cómputo en inferencia y el momento o1

El razonamiento —matemática, deducción lógica, programación— es una capacidad emergente en LLMs a suficiente escala, y el *chain-of-thought* (CoT) la potencia forzando pasos intermedios. La idea de fondo es que asignar **más cómputo en tiempo de inferencia** —más tokens de "pensamiento" antes de responder— eleva la precisión en tareas complejas.

El enfoque post-entrenamiento tradicional usa SFT sobre trazas anotadas por humanos, seguido de RL. R1 identifica dos límites de depender de demostraciones humanas: **escalabilidad y sesgo** (anotar trazas de calidad es caro e introduce sesgos), y **techo humano** (al restringir al modelo a replicar procesos humanos, su desempeño queda acotado por los ejemplos). El "momento o1" (OpenAI, 2024) mostró públicamente el salto cualitativo de las CoT largas y auto-correctoras; R1 es la respuesta abierta, y su aporte conceptual es demostrar que **no se necesita ninguna traza de razonamiento humana**: bastan preguntas difíciles, un verificador confiable y suficiente cómputo de RL.

## Método / Contribución

### GRPO: el motor de RL

R1-Zero y R1 se entrenan con **Group Relative Policy Optimization (GRPO)**, que simplifica y abarata a [PPO](/papers/ppo-schulman-2017). La diferencia clave es cómo se estima la ventaja. PPO requiere una **red de valor (critic)** del tamaño del *policy* para estimar $V(s)$. **GRPO la elimina**: para cada pregunta $q$ muestrea un **grupo** de $G$ salidas y estima la ventaja de cada una por normalización dentro del grupo,

$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$

El promedio del grupo cumple el rol de *baseline* que en PPO cumplía el crítico, pero sin costo de un modelo adicional. El objetivo conserva los ingredientes de PPO: cociente de probabilidades entre política nueva y vieja, operador **clip** para estabilidad y penalización **KL** contra una referencia $\pi_{ref}$. Esto encaja con la lógica de "generar muchas respuestas por problema": el grupo de rollouts que ya se necesita para explorar es también el que provee la baseline estadística.

### Recompensas basadas en reglas

Para R1-Zero la señal es **puramente por reglas**, con dos componentes de igual peso: **precisión** (si la respuesta final es correcta —comparación matemática determinista o compilador ejecutando tests de código) y **formato** (encapsular el razonamiento entre `<think>...</think>` y la respuesta entre `<answer>...</answer>`). Los autores **se abstienen de usar modelos de recompensa neuronales**, porque son susceptibles a *reward hacking* en el RL a gran escala. Solo se restringe la forma, no el contenido, para observar la progresión natural del modelo.

### El pipeline multietapa de R1

R1-Zero sufre mala legibilidad y mezcla de idiomas. DeepSeek-R1 lo corrige con cuatro fases sobre V3-Base: **(1) arranque en frío** (SFT sobre miles de ejemplos legibles), **(2) primera RL** con GRPO más una recompensa de consistencia de idioma, **(3) muestreo por rechazo + SFT** (~600k muestras de razonamiento filtradas + ~200k no-razonamiento = ~800k ejemplos), y **(4) RL final** que combina reward por reglas (razonamiento) con reward de preferencias humanas (utilidad, inocuidad) para dominios generales.

### Destilación

El razonamiento de R1 se transfiere a modelos abiertos pequeños (Qwen y Llama, de 1.5B a 70B) por **SFT directo sobre los 800k ejemplos**, sin aplicar RL a los estudiantes. Es "empaquetar" las trazas del maestro grande y enseñárselas por imitación.

## Resultados

- **R1-Zero (RL puro).** En AIME 2024, el pass@1 promedio salta de **15.6% a 77.9%** durante el entrenamiento; con auto-consistencia (cons@16) llega a **86.7%**, superando el promedio de los competidores humanos. La longitud de respuesta crece sostenidamente: el modelo *aprende por sí solo a pensar más tiempo*, sin traza humana alguna.
- **R1 frente a o1.** Queda a la par de o1-1217: AIME 2024 **79.8** vs 79.2, MATH-500 **97.3** vs 96.4, LiveCodeBench **65.9** vs 63.4. En Codeforces supera al 96.3% de los participantes humanos. El salto sobre V3-Base es enorme (AIME 39.2 → 79.8).
- **Destilación.** El diminuto **Qwen-1.5B destilado supera a GPT-4o y Claude-3.5-Sonnet** en los benchmarks matemáticos. R1-Distill-Qwen-32B llega a 72.6 en AIME, mientras que aplicar RL puro directo a Qwen-32B alcanza apenas 47.0: **destilar rinde mucho mejor** que RL a gran escala sobre el modelo pequeño.

### El "aha moment"

En una versión intermedia de R1-Zero, resolviendo $\sqrt{a - \sqrt{a+x}} = x$, el modelo escribe en medio de su cadena: *"Wait, wait. Wait. That's an aha moment I can flag here. Let's reevaluate this step-by-step..."*. Se detiene, reconoce que debe reconsiderar y reinicia el razonamiento, sin haber sido entrenado para hacerlo. Cuantitativamente se manifiesta como un aumento súbito del uso de "wait". El mensaje: comportamientos como autoverificación y reflexión **emergen orgánicamente** del RL cuando se premia solo el resultado correcto.

## Limitaciones

- **Solo dominios verificables.** El pilar es un **verificador confiable**. Para tareas sin regla de corrección objetiva (escritura), no se puede usar reward por reglas; sustituirlo por un modelo neuronal lo vuelve vulnerable a *reward hacking*. **Escalar el RL puro a tareas no verificables sigue abierto.**
- **Legibilidad y mezcla de idiomas.** R1 sigue optimizado solo para chino e inglés.
- **Sensibilidad a prompts.** El *few-shot* degrada el desempeño; se recomienda *zero-shot*.
- **Uso de herramientas, salida estructurada e ingeniería de software** aún subóptimos; y *overthinking* (pensar de más en preguntas simples).

## Por qué importa para la Clase 34

R1 materializa el "momento o1 & R1" de la [Clase 34](/clases/clase-34): muestrear muchas respuestas por problema y dar reward positivo a las trazas que llegan al resultado correcto, sobre problemas verificables. El grupo de $G$ rollouts de GRPO *es* ese conjunto de respuestas, y la ventaja normalizada al grupo *es* la forma de premiar relativamente las mejores. El *aha moment* es la evidencia de que los comportamientos de razonamiento emergen del RL sin ser programados.

La diferencia con [RLHF](/fundamentos/rlhf) define el alcance del método. En RLHF la recompensa la entrega un **modelo neuronal entrenado sobre preferencias humanas**: señal subjetiva, aprendida y explotable. En R1-Zero la recompensa es **verificable por reglas**: la respuesta es correcta o no, el código pasa los tests o no. Esa señal objetiva y no *hackeable* permite RL a gran escala sin SFT previo ni anotación de trazas. La lección transversal —central en el debate que abre [Yue et al.](/papers/rl-reasoning-yue-2025)— es que el RL desbloquea capacidades del modelo base cuando la señal es confiable; el cuello de botella no es el algoritmo sino la existencia de un verificador. R1 es la aplicación cumbre del [test-time compute](/fundamentos/test-time-compute) incentivado por RL.
