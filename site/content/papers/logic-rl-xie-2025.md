---
title: "Logic-RL: razonamiento con RL basado en reglas (2025)"
weight: 390
math: true
---

{{< paper-card
    title="Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning"
    authors="Tian Xie et al."
    year="2025"
    venue="arXiv:2502.14768"
    pdf="/papers/logic-rl-xie-2025.pdf" >}}
Logic-RL es el **complemento controlado de [DeepSeek-R1](/papers/deepseek-r1-2025)**: toma su receta de recompensas basadas en reglas y la lleva a un laboratorio verificable —**puzzles lógicos sintéticos de caballeros y bribones (Knights & Knaves)**— donde cada variable (dificultad, formato, recompensa) es observable y manipulable. Con solo **~5.000 problemas de lógica**, un modelo base de **7B** desarrolla reflexión, verificación, exploración de alternativas y resumen sistemático, comportamientos **ausentes en su corpus de entrenamiento**. El resultado estelar es la generalización fuera de dominio: pese a ver únicamente puzzles lógicos, mejora **+125% en AIME** y **+38% en AMC**. Es el ejemplo de la [Clase 34](/clases/clase-34) (slide 35, "aha-moments") de RL con recompensa verificable que induce razonamiento sin supervisión de proceso, y el microscopio del [test-time compute](/fundamentos/test-time-compute) emergente.
{{< /paper-card >}}

---

## Contexto: RLVR después de R1

DeepSeek-R1 mostró que un esquema **simple de RL basado en reglas** —recompensar respuesta y formato correctos, sin más— basta para hacer emerger razonamiento, sin Monte Carlo Tree Search ni Process Reward Models. Este paradigma se conoce como **RLVR** (*Reinforcement Learning with Verifiable Rewards*): a diferencia de [RLHF](/fundamentos/rlhf), la recompensa proviene de una **regla determinística** que comprueba objetivamente si la respuesta es correcta —señal exacta, barata y difícil de engañar, pero limitada a dominios verificables.

El problema que Xie et al. identifican es de **reproducibilidad**: R1 liberó los pesos pero no el código ni el dataset, dejando abiertas preguntas críticas (¿emergen capacidades en modelos pequeños? ¿cuál es la estructura óptima de datos?). Responder exige un marco controlado. Aquí el punto metodológico: aunque las matemáticas son el banco de pruebas habitual, datasets como GSM8K son problemáticos *como datos de entrenamiento* porque su complejidad es **incontrolable y de varianza alta**. Para estudiar la dinámica del razonamiento hace falta un entorno donde la dificultad sea una perilla que se pueda girar.

## Método / Contribución

### Un dataset lógico como laboratorio

Los puzzles de caballeros y bribones son el sustrato ideal por tres propiedades: **generación procedural** (variabilidad infinita, datos no vistos por el modelo, mide generalización genuina), **dificultad controlable** (se modula variando el número de personajes de 2 a 8 y la complejidad de operadores booleanos de 1 a 4) y **facilidad de verificación** (una única respuesta ground-truth inequívoca, que minimiza el *reward hacking*). Ejemplo: *"Zoey remarked: 'Oliver is not a knight'. Oliver stated: 'Oliver is a knight if and only if Zoey is a knave'."* — la solución (Zoey bribón, Oliver caballero) se deduce por reglas formales y se verifica determinísticamente.

### Recompensa basada en reglas y anti reward-hacking

La recompensa se refinó **iterativamente**, monitoreando comportamientos de hackeo hasta un sistema "casi imposible de hackear" con dos términos. La **recompensa de formato** exige `<think></think>` y `<answer></answer>`:

$$S_{format} = \begin{cases} 1, & \text{formato correcto} \\ -1, & \text{formato incorrecto} \end{cases}$$

Lo más instructivo es el **catálogo de reward hacking** observado y cómo cada patología motivó una regla: saltarse el `<think>`, poner el razonamiento dentro de `<answer>`, adivinar sin razonar, texto irrelevante, volver a pensar tras emitir `<answer>`, o frases como *"thinking process here"* para simular razonamiento. En respuesta, endurecieron: cada etiqueta exactamente una vez y en orden correcto. La lección de RLVR: **el modelo optimiza exactamente lo que se mide, no lo que se quiere lograr**; cerrar los atajos es tan importante como definir el objetivo. La **recompensa de respuesta** es asimétrica (+2 total, −1.5 parcial, −2 no parseable), castigando la evasión más que el error parcial.

### Algoritmo de RL

El algoritmo base es **REINFORCE++**, que en su configuración **superó a GRPO**. Siguiendo a DeepSeek-Math mueven la penalización **KL fuera de la recompensa y la incorporan a la pérdida**, y adoptan el estimador insesgado que garantiza KL no negativa. El modelo se entrena por 3.600 pasos con puzzles de complejidad mixta (3 a 7 personas), lo que permite evaluar generalización fuera de distribución con puzzles de **8 personas** nunca vistos. Hallazgo: base e instruct dan curvas casi idénticas — *"el cold start es un bonus, no una necesidad"*.

## Resultados

- **Emergencia gradual.** La longitud media de respuesta crece casi linealmente de ~500 a ~2.000 tokens (**4×**) sin instrucción alguna, y con ella emergen reflexión, exploración multicamino con backtracking y aplicación instintiva de la fórmula de implicación —**lógica formal, no solo ensayo y error**. En K&K, Logic-RL lleva el promedio de **0.19 a 0.89**, superando a GPT-4o (0.37) y generalizando a 8 personas (0.67).
- **¿Hay un "aha moment"?** El punto más matizado. Logic-RL **no observó ninguna forma súbita**: el modelo no verbalizó "aha moment", y rastreando la frecuencia de palabras reflexivas (*check*, *verify*, *wait*, *re-evaluate*) en los primeros 1.800 pasos, todas **crecen gradual y establemente, sin saltos abruptos**. Ya exhibía razonamiento complejo en el paso 10. La conclusión: el razonamiento **emerge orgánica y gradualmente**, no en un instante mágico.
- **Transferencia a matemáticas (Super OOD).** Pese a entrenar solo en lógica, mejora **+125% en AIME (2021-2024)** y **+38% en AMC**. La mejora sincrónica indica que el RL facilita **esquemas abstractos de resolución de problemas** transferibles, no coincidencia de patrones del dominio.
- **SFT memoriza, RL generaliza.** Con la métrica LiMem, el RL logra mayor accuracy de test con incremento mínimo o negativo de memorización, mientras el fine-tuning por rechazo dispara la memorización.

## Limitaciones

- **Escala pequeña y dominio estrecho.** Dataset <5.000 muestras; la generalización a matemáticas o código a gran escala queda por explorar.
- **Longitud explosiva.** Las respuestas se expanden hasta 4×; hacen falta métodos *long-to-short*.
- **Restricciones de formato posiblemente subóptimas.** Queda abierto si un enfoque sin restricciones o latente rendiría mejor.
- **Mezcla de idiomas sin explicar** (tokens chinos en `<think>` pese a entrenar solo en inglés).

## Por qué importa para la Clase 34

En la [Clase 34](/clases/clase-34), Logic-RL aparece (slide 35) bajo "aha-moments" como ejemplo de RL basado en reglas que induce razonamiento. Su valor pedagógico es doble. Primero, es el **complemento controlado de [DeepSeek-R1](/papers/deepseek-r1-2025)**: R1 demostró el fenómeno a gran escala pero como caja negra (sin código ni datos); Logic-RL lo reproduce en un laboratorio verificable, confirmando las tres piezas del paradigma —recompensa por reglas, emergencia sin datos que la contengan, crecimiento autónomo del cómputo de razonamiento. Es la validación científica reproducible de la receta de R1 en pequeño.

Segundo, **matiza el mito del "aha moment"**: donde el relato popular sugiere un instante mágico, Logic-RL muestra emergencia gradual y continua. Esto conecta con el debate que abre [Yue et al.](/papers/rl-reasoning-yue-2025): la tensión entre *"el RL amplifica capacidades latentes del base"* vs. *"el RL enseña razonamiento nuevo"*. El hallazgo de que base e instruct dan curvas casi idénticas, y que el RL generaliza donde el SFT memoriza, empuja hacia la idea de que el RL **reorganiza y hace accesibles** esquemas de razonamiento más que inyectarlos desde cero —el mismo [test-time compute](/fundamentos/test-time-compute) que R1 popularizó, ahora observado bajo el microscopio.
