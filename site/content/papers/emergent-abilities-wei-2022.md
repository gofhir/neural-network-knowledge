---
title: "Emergent Abilities of Large Language Models (2022)"
weight: 382
math: true
---

{{< paper-card
    title="Emergent Abilities of Large Language Models"
    authors="Jason Wei et al. (Google, Stanford, DeepMind)"
    year="2022"
    venue="TMLR 2022"
    pdf="/papers/emergent-abilities-wei-2022.pdf" >}}
Escalar modelos de lenguaje —más cómputo, más parámetros, más datos— mejora el rendimiento de forma **predecible**, capturada por *leyes de escala* que abarcan más de siete órdenes de magnitud. Sobre ese fondo predecible, Wei et al. dirigen la atención a un fenómeno **impredecible**: ciertas habilidades **no están presentes en modelos pequeños y sí aparecen en modelos grandes**, sin poder anticiparse extrapolando la curva de los menores. A eso llaman *habilidad emergente*. La tesis, tomada del físico Philip Anderson (*More Is Different*, 1972): **"cambios cuantitativos producen cambios cualitativos en el comportamiento"**. Entre los ejemplos está el propio [Chain-of-Thought](/papers/chain-of-thought-wei-2022): el [razonamiento](/fundamentos/razonamiento) multi-paso solo supera al prompting estándar a partir de ~$10^{23}$ FLOPs. Es la referencia canónica de por qué el razonamiento por CoT requiere escala, base de la [Clase 34](/clases/clase-34).
{{< /paper-card >}}

---

## Contexto: leyes de escala y el salto a las capacidades

Antes de este paper, la comunidad de NLP había consolidado que **la escala mejora el rendimiento de forma metódicamente predecible**: las *scaling laws* de Kaplan et al. (2020) y Hoffmann et al. (2022) muestran que la pérdida de entropía cruzada cae de forma suave y regular en función del cómputo, los parámetros y el tamaño del dataset. Esa suavidad permite *presupuestar* un entrenamiento. El problema que motiva a Wei et al. es que **esa predecibilidad de la pérdida no se traslada automáticamente al rendimiento en tareas downstream**: para ciertas tareas el desempeño no mejora de forma continua ni puede anticiparse.

El concepto de emergencia proviene de la física y las ciencias de la complejidad. La analogía más útil es la de una **transición de fase**: el agua no se enfría gradualmente hasta ser hielo, cambia de estado abruptamente al cruzar los 0 °C. Del mismo modo, una habilidad emergente muestra rendimiento cercano al azar hasta cierto umbral y luego un **salto** a un desempeño muy superior. Un matiz importante: **no existe un único proxy de la escala** —FLOPs, parámetros o datos—, por lo que conviene ver la emergencia como función de muchas variables correlacionadas, no de un solo eje.

## Contribución central: el marco de la emergencia

El trabajo **no propone un método ni un modelo**: es un **survey conceptual** que reorganiza resultados dispersos (GPT-3, LaMDA, Gopher, Chinchilla, PaLM, BIG-Bench) bajo un mismo marco. Su definición operativa es deliberadamente acotada: **una habilidad es emergente si no está presente en modelos pequeños pero sí en modelos grandes**. De ahí la propiedad clave: no podría haberse predicho extrapolando la ley de escala desde los modelos pequeños. En una curva de escala:

$$\text{rendimiento} \approx \text{azar} \quad \text{hasta un umbral crítico, luego} \quad \text{rendimiento} \gg \text{azar}.$$

Los autores son explícitos sobre lo que *no* afirman: **la escala a la que una habilidad emerge no es una propiedad inmutable**. Puede ocurrir con menos cómputo si el modelo se entrena con datos de mayor calidad, otra arquitectura u otro objetivo. El umbral es empírico y contingente, no una ley fundamental.

## Evidencia: tareas y curvas por escala

El paper organiza la evidencia en dos categorías. La primera son las **habilidades emergentes en prompting few-shot** (Fig. 2, ocho ejemplos, cinco familias de modelos): aritmética multi-dígito (que salta a $2 \cdot 10^{22}$ FLOPs, **13B**, en GPT-3); transliteración fonética; recuperar palabras de letras desordenadas; QA en persa; **TruthfulQA** (solo el Gopher de **280B** salta a +20 puntos sobre el azar); **MMLU** (hay que escalar a **70B–280B**); y **Word in Context** (solo emerge con PaLM a $2.5 \cdot 10^{24}$ FLOPs, **540B**).

La segunda categoría son las **estrategias de prompting aumentado** (Fig. 3), donde una técnica se considera emergente si no ayuda —o perjudica— hasta cierta escala:

- **Chain-of-Thought** (razonamiento multi-paso, Fig. 3A): sobre problemas matemáticos verbales (GSM8K), **CoT solo supera al prompting estándar al escalar a $10^{23}$ FLOPs (~100B parámetros)**. Por debajo, generar cadenas de razonamiento no ayuda o empeora el resultado. Es el resultado más relevante para la Clase 34.
- **Seguimiento de instrucciones** (Fig. 3B): el *instruction finetuning* **perjudica** a modelos de **8B** o menores y solo mejora a ~100B.
- **Ejecución de programas / scratchpad** (Fig. 3C): finetunear para predecir salidas intermedias solo ayuda a partir de **~40M parámetros** —la escala de emergencia más baja de la Tabla 1.
- **Calibración P(True)** (Fig. 3D): solo supera a los métodos estándar en el mayor modelo, **52B**.

## Ejemplos emergentes y el caso de Chain-of-Thought

Vale detenerse en por qué CoT es el ejemplo paradigmático del razonamiento. El paper ofrece una **intuición estructural**: si una tarea de razonamiento requiere $l$ pasos de cómputo secuencial, resolverla podría exigir un modelo de **profundidad $O(l)$ capas**. Un modelo pequeño, con pocas capas, no tendría la profundidad para encadenar los pasos; por eso el razonamiento explícito no puede "activarse" hasta que la red es lo bastante profunda y grande. Es una hipótesis, no una demostración —los autores admiten que "hay pocas explicaciones convincentes de por qué emergen las habilidades"—, pero conecta razonamiento con escala. El caso **WiC** ilustra la impredecibilidad histórica: cuando GPT-3 175B falló, se atribuyó a su objetivo autorregresivo y se sugirió un modelo bidireccional; luego bastó **seguir escalando** un decoder-only (PaLM 540B) para resolverlo sin cambios arquitectónicos.

## Limitaciones

- **Es un survey.** Todas las cifras son de terceros; no controla las diferencias entre familias (datos, arquitectura, objetivo), por lo que atribuir la emergencia limpiamente "a la escala" es difícil.
- **Falta de explicación mecanicista.** El análisis de la pérdida de entropía cruzada muestra mejora latente incluso donde las métricas downstream siguen al azar, pero no explica por qué esas métricas son emergentes ni predice el umbral.
- **El rol de las métricas.** El propio paper anticipa la crítica principal posterior: usar *exact match* "puede disfrazar mejoras incrementales como emergencia". El debate lo continuaría Schaeffer et al. (*Are Emergent Abilities a Mirage?*, NeurIPS 2023), argumentando que muchas curvas son artefacto de métricas discontinuas; no está cerrado, porque a un usuario le importa si el modelo *acierta la respuesta*, no la log-verosimilitud.
- **El umbral no es intrínseco:** depende de datos, arquitectura y objetivo, y puede bajar con nuevas técnicas.

## Por qué importa para la Clase 34

La [Clase 34](/clases/clase-34) (Razonamiento) sitúa a los comportamientos emergentes —el [In-Context Learning](/fundamentos/in-context-learning) entre ellos— como base del paradigma de prompting popularizado por [GPT-3](/papers/gpt-3-brown-2020). Este paper es el fundamento de esa afirmación y, sobre todo, la explicación de **por qué el razonamiento por Chain-of-Thought no es gratis**. El mensaje operativo es directo: **CoT es una habilidad emergente, no una técnica universal**; no se puede tomar un modelo pequeño, añadirle un prompt de CoT y esperar razonamiento. La técnica necesita un **sustrato de escala** para activarse. Un estudiante debería llevarse tres ideas:

1. **Las scaling laws predicen la pérdida, no las capacidades.** El rendimiento en razonamiento puede seguir en el azar mientras la pérdida de pre-entrenamiento ya mejora suavemente; la capacidad aparece de golpe.
2. **CoT es emergente:** funciona a partir de cierta escala y no antes. El [razonamiento](/fundamentos/razonamiento) explícito y la escala están acoplados.
3. **El umbral es empírico y movible:** mejores datos, arquitecturas u objetivos, y técnicas posteriores (instruction tuning, RLHF, destilación de razonamiento) pueden bajarlo. La escala es una condición típicamente necesaria, no mágicamente suficiente ni eternamente fija.
