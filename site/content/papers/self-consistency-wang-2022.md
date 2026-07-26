---
title: "Self-Consistency Improves Chain of Thought (2022)"
weight: 386
math: true
---

{{< paper-card
    title="Self-Consistency Improves Chain of Thought Reasoning in Language Models"
    authors="Xuezhi Wang et al. (Google)"
    year="2022"
    venue="ICLR 2023"
    pdf="/papers/self-consistency-wang-2022.pdf" >}}
**Self-consistency** es una estrategia de *decoding* que reemplaza el *greedy decoding* usado con [Chain-of-Thought](/fundamentos/chain-of-thought). En lugar de decodificar una única cadena de razonamiento tomando el token más probable en cada paso, self-consistency **muestrea un conjunto diverso de cadenas** y luego **marginaliza** sobre ellas para quedarse con la respuesta final más frecuente por **voto de mayoría**. La intuición viene del razonamiento humano: un problema complejo admite múltiples caminos distintos que convergen a la misma respuesta correcta, y esa convergencia da confianza. El método es **completamente no supervisado**: funciona *off-the-shelf* sobre modelos preentrenados, sin entrenamiento, fine-tuning, modelos auxiliares ni anotación. Sobre PaLM-540B, GPT-3, LaMDA-137B y UL2-20B mejora a CoT con márgenes notables (**+17.9% en GSM8K, +12.2% en AQuA, +6.4% en StrategyQA**). Es la materialización del slide 33 de la [Clase 34](/clases/clase-34): el puente entre [Pass@k](/fundamentos/test-time-compute) y una métrica realizable.
{{< /paper-card >}}

---

## Contexto: Chain-of-Thought, greedy decoding y su fragilidad

**Wei et al. (2022)** mostraron que inducir al modelo a generar **pasos intermedios** de razonamiento en lenguaje natural —[Chain-of-Thought](/papers/chain-of-thought-wei-2022)— mejora fuertemente el desempeño en tareas multi-paso. El problema es **cómo se decodifica** esa cadena. La práctica estándar era el **greedy decoding**: elegir en cada posición el token de máxima probabilidad, produciendo **una sola** cadena. Esto tiene dos debilidades. Primero, el greedy tiende a repeticiones y óptimos locales. Segundo, y más grave, el razonamiento multi-paso es **frágil**: si esa única cadena contiene un error en cualquier paso, la respuesta final es errónea, sin mecanismo de recuperación.

Hay una tensión conceptual que el paper subraya. Las tareas de razonamiento tienen una **respuesta única y fija**, por lo que los investigadores gravitaban hacia decoding determinístico; el muestreo se asociaba con generación **abierta** de texto, donde la diversidad es deseable. El hallazgo es contraintuitivo: **incluso cuando la respuesta es fija, introducir diversidad en el proceso de razonamiento resulta muy beneficioso**. Self-consistency habita "un espacio interesante entre la generación abierta y la generación óptima con respuesta fija", y logra las mejoras de enfoques previos (entrenar un *verifier*, Cobbe et al. 2021; un *re-ranker*, Thoppilan et al. 2022) **sin ningún componente entrenado adicional**.

## Método: muestrear y marginalizar

El procedimiento tiene tres pasos: (1) inducir al modelo con exemplars de CoT como en Wei et al.; (2) en vez de greedy, **muestrear** un conjunto de cadenas candidatas, cada una con su respuesta final; (3) **marginalizar** sobre las cadenas eligiendo la respuesta más frecuente. Formalmente, sea $a_i$ la respuesta de la $i$-ésima de $m$ muestras y $r_i$ la cadena de razonamiento (variable latente). Tras muestrear los pares $(r_i, a_i)$, se marginaliza sobre $r_i$ por voto de mayoría:

$$\arg\max_{a} \sum_{i=1}^{m} \mathbb{1}(a_i = a)$$

La palabra clave es **marginalizar**: la cadena es una latente que no interesa por sí misma; interesa la distribución sobre la respuesta final tras sumar sobre todos los caminos. Se puede ponderar cada par por su probabilidad, pero el hallazgo práctico es que **el voto de mayoría simple rinde casi igual** que la suma ponderada (74.4% vs. 74.1% en GSM8K), porque las probabilidades condicionales de distintos pares resultan muy parecidas —el modelo no está bien calibrado, lo que explica por qué otros entrenaban re-rankers.

**Por qué funciona.** La hipótesis: como los LLMs no son razonadores perfectos, producen cadenas con errores, pero **los procesos correctos, aunque diversos, tienden a coincidir más en su respuesta final que los incorrectos**. Dos caminos correctos aterrizan en el mismo número; dos errores distintos aterrizan, con alta probabilidad, en números distintos. Así la respuesta correcta acumula votos y las incorrectas se dispersan. El método es un **self-ensemble** sobre un único modelo (no combina modelos distintos: ensamblar LaMDA-137B + PaLM-540B da 36.9% en GSM8K frente al 74.4% de self-consistency solo sobre PaLM). Es compatible con temperature, top-$k$ y nucleus sampling, y robusto a sus parámetros.

## Resultados

En régimen few-shot, con los mismos prompts de Wei et al. (8 exemplars aritméticos), promediando 10 corridas de 40 muestras. Self-consistency mejora sobre CoT en los cuatro modelos y todas las tareas, y **las ganancias crecen con la escala**: +3–6% sobre UL2-20B pero +9–23% sobre LaMDA-137B y GPT-3. Algunas cifras sobre PaLM-540B:

- **GSM8K:** 56.5% → **74.4% (+17.9)**.
- **AQuA:** 35.8% → **48.3% (+12.5)**.
- **SVAMP:** 79.0% → **86.6% (+7.6)**.
- **MultiArith (LaMDA-137B):** 51.8% → **75.7% (+23.9)**, la mayor ganancia absoluta.

En sentido común y simbólico obtiene SoTA en 5 de 6 tareas (StrategyQA 75.3% → 81.6%; ARC-challenge 85.2% → 88.7%). Muestrear más caminos mejora consistentemente, pero **la curva satura rápido**: con 5–10 caminos ya se captura la mayor parte de la ganancia. Notablemente, self-consistency **repara casos donde CoT perjudica**: donde añadir CoT baja la accuracy frente al prompting estándar (ANLI, RTE), self-consistency cierra la brecha y la supera (ANLI-R1: estándar 69.1%, CoT 68.8%, self-consistency **78.5%**). Frente a otras estrategias, gana a *sample-and-rank*, a beam search (que produce **menos diversidad**) y a ensembles por permutación de exemplars. Un beneficio secundario: el **% de consistencia correlaciona con la exactitud**, dando una señal incipiente de "saber cuándo no sabe".

## Limitaciones

1. **Costo de cómputo.** Requiere muestrear decenas de caminos en vez de uno; se mitiga porque el desempeño satura rápido (5–10 caminos bastan).
2. **Requiere respuesta única identificable.** Aplica solo a problemas cuya respuesta proviene de un conjunto fijo, parseable y comparable; extenderlo a generación abierta exige una métrica de consistencia no trivial.
3. **Racionales potencialmente incorrectos.** El modelo puede generar caminos sin sentido aun cuando la respuesta agregada es correcta; los racionales solo deben usarse para inspeccionar, con precaución.

## Por qué importa para la Clase 34

Self-consistency es una bisagra entre varias ideas de la [Clase 34](/clases/clase-34):

- **Pass@k (slide 33).** La observación de que muestreando muchas respuestas alguna sea correcta es una cota optimista que supone un **oráculo** que sabe cuál es la buena —oráculo que no tenemos. Self-consistency es el **puente entre Pass@k y una métrica realizable**: sin conocer la respuesta, usa la **frecuencia** como sustituto del oráculo. Es el mecanismo que **cosecha** la promesa de Pass@k sin supervisión.
- **[Test-time compute](/fundamentos/test-time-compute).** Es un ejemplo temprano y limpio de **escalar cómputo en inferencia** en vez de en entrenamiento: los pesos no cambian, solo cuánto se gasta al responder (un camino vs. 40). Se compra exactitud con muestras. Es la lógica que años después estructuraría a los *reasoning models* que "piensan más" antes de responder.
- **Reducción de varianza.** El greedy es una única muestra ruidosa; self-consistency agrega muchas por moda, un estimador de menor varianza que mitiga la estocasticidad y la trampa del óptimo local.
- **Relación con CoT y ToT.** Self-consistency **presupone** [CoT](/fundamentos/chain-of-thought) —sin cadenas explícitas no hay diversidad que marginalizar—; es una capa de decoding *encima* de él. Frente a Tree of Thoughts, muestrea caminos **independientes** en paralelo y agrega solo al final (búsqueda plana, sin ramificación ni evaluación intermedia), el caso más simple de la familia que explora el espacio de razonamientos en tiempo de inferencia.

En salud, self-consistency reduce la varianza de una única generación ruidosa y —vía la correlación consistencia-exactitud— permite marcar los casos de **baja consistencia** para revisión humana. Nunca sustituye la verificación clínica: los racionales pueden ser incorrectos aunque la respuesta agregada sea correcta.
