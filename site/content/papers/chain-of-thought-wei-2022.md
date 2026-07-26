---
title: "Chain-of-Thought Prompting (2022)"
weight: 383
math: true
---

{{< paper-card
    title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    authors="Jason Wei et al. (Google Brain)"
    year="2022"
    venue="NeurIPS 2022"
    pdf="/papers/chain-of-thought-wei-2022.pdf" >}}
Basta con **mostrarle al modelo unos pocos ejemplos que incluyan los pasos intermedios de razonamiento** para que un modelo de lenguaje suficientemente grande empiece a resolver problemas que antes fallaba, sin ajustar un solo parámetro. La técnica —el **chain-of-thought prompting**— transforma cada exemplar de few-shot de un par $\langle \text{entrada}, \text{salida} \rangle$ en un triple $\langle \text{entrada}, \text{cadena de pensamiento}, \text{salida} \rangle$: una serie de pasos de razonamiento en lenguaje natural que conducen a la respuesta. El hallazgo central es que esta capacidad es **emergente con la escala**: no ayuda —incluso perjudica— en modelos pequeños, y solo produce ganancias a partir de aproximadamente **100.000 millones (100B) de parámetros**. El resultado estrella: con solo **ocho exemplars**, PaLM 540B pasa de **17.9% a 56.9%** en GSM8K, superando el estado del arte previo que requería finetuning y un verificador. Es el paper central de la [Clase 34](/clases/clase-34) y el ancestro directo del [chain-of-thought](/fundamentos/chain-of-thought) como paradigma de razonamiento en LLMs.
{{< /paper-card >}}

---

## Contexto: los LLMs fallan en aritmética simple pese a su escala

Escalar los modelos de lenguaje confiere beneficios predecibles que siguen **leyes de escala suaves y monótonas** (Kaplan et al., 2020). El problema que motiva el paper es que **escalar el tamaño del modelo, por sí solo, no basta** para tareas difíciles de razonamiento aritmético, de sentido común y simbólico. Es contraintuitivo: un modelo de 175B parámetros que escribe ensayos fluidos puede fallar en un problema aritmético que un niño resuelve. La razón es que la aritmética de varios pasos exige **encadenar operaciones**, y el *standard prompting* obliga al modelo a producir la respuesta en un solo paso hacia adelante (*one forward pass*), sin espacio para desplegar el cálculo. En esas tareas la curva de escala del prompting estándar es **plana**.

Existían dos líneas previas, ambas con limitaciones. Los **racionales entrenados** (Ling et al., 2017; Cobbe et al., 2021, con GSM8K y un verificador) exigen crear un conjunto grande y caro de racionales de alta calidad. El **few-shot prompting** (Brown et al., 2020) es barato pero **funciona mal en tareas de razonamiento** y no mejora al escalar. La contribución del paper es **combinar las fortalezas de ambos evitando sus limitaciones**: usar few-shot (sin dataset ni finetuning) pero con exemplars que incluyen la cadena de razonamiento. Es la generalización del [scratchpad de Nye et al. (2021)](/papers/scratchpad-nye-2021) —cómputo intermedio para tareas simbólicas/programáticas— a razonamiento en lenguaje natural.

## Método y contribución

El método aumenta cada exemplar del few-shot con una cadena de pensamiento. Es puramente de prompting, con dos virtudes prácticas: no requiere dataset de entrenamiento y **un único checkpoint sirve para muchas tareas**. El ejemplo canónico (Figura 1, el mismo de la **slide 28**) contrasta ambos formatos con **Natalia y sus clips**. Ante la pregunta de test, el standard prompting responde directamente y falla; el chain-of-thought despliega los pasos —"la cafetería tenía 23 manzanas, usó 20 para el almuerzo, quedan $23 - 20 = 3$, compró 6 más, así que tiene $3 + 6 = 9$"— y acierta. **El problema no cambió; cambió el formato de la demostración.**

Los autores compusieron manualmente **ocho exemplars** con cadenas de pensamiento, usados para todos los benchmarks, sin *prompt engineering* previo y con *greedy decoding* (mejorado después por el [self-consistency de Wang et al. (2022)](/papers/self-consistency-wang-2022), que toma la respuesta mayoritaria sobre muchas cadenas muestreadas). Se evalúan cinco familias —GPT-3, LaMDA, PaLM, UL2 20B y Codex— cubriendo un amplio rango de escalas, esencial para detectar la emergencia. La cadena de pensamiento tiene cuatro propiedades atractivas: **cómputo adaptativo** (más tokens intermedios para problemas más difíciles, la semilla del *test-time compute*), **interpretabilidad**, **generalidad** y **facilidad** (se elicita en modelos *off-the-shelf*).

## Resultados

El resultado emblemático está en GSM8K: PaLM 540B pasa de **17.9% (standard) a 56.9% (CoT)**, un salto de **+39.0 puntos** que **más que triplica** el desempeño y supera el SOTA previo (55%, finetuning + verificador). En los otros benchmarks con PaLM 540B: SVAMP 69.4 → 79.0, ASDiv 72.1 → 73.9, AQuA 25.2 → 35.8, MAWPS 79.2 → 93.3. Tres conclusiones transversales:

- **Es una habilidad emergente de la escala.** LaMDA 8B en GSM8K *cae* de 3.2% a 1.6% con CoT; los modelos pequeños producen cadenas *fluidas pero ilógicas*. Recién a ~100B parámetros aparece el salto.
- **Las ganancias son mayores en problemas más difíciles.** En GSM8K (menor baseline) el desempeño más que se duplicó; en SingleOp (un solo paso) las mejoras fueron mínimas.
- **Generaliza más allá de la aritmética.** En razonamiento de sentido común supera el SOTA en StrategyQA (75.6% vs 69.4%) y a un aficionado deportivo humano en Sports Understanding (95.4% vs 84%); en razonamiento simbólico (concatenación de letras, coin flip) facilita la **generalización de longitud** a cadenas más largas que las vistas.

Tres ablaciones aíslan qué importa: prompting con **solo la ecuación** no ayuda en GSM8K; **solo cómputo variable** (emitir una secuencia de puntos `. . .`) rinde igual que el baseline; y poner la **cadena después de la respuesta** también rinde igual. Convergen en que lo decisivo es **el razonamiento secuencial en lenguaje natural desplegado antes de responder**, no el cómputo extra ni la activación de conocimiento.

## Limitaciones

- **No garantiza "razonamiento" real.** La cadena emula el proceso humano, pero no responde si la red está realmente razonando; queda abierto.
- **No hay garantía de caminos correctos ni factuales.** Una cadena puede llevar a respuestas correctas o incorrectas; los autores advierten no usarla como factual en el mundo real sin más cuidado.
- **Emerge solo a gran escala,** lo que la hace **costosa de servir**. Inducir razonamiento en modelos pequeños queda como trabajo futuro.

## Por qué importa para la Clase 34

La [Clase 34](/clases/clase-34) recorre la distinción **System 1 vs System 2** (Kahneman): el prompting estándar es System 1 puro —respuesta en un único paso por asociación—, mientras la cadena de pensamiento intenta dotar al modelo de algo parecido a **System 2**: un proceso secuencial y deliberado antes de comprometerse con una respuesta. Es la bisagra del arco de razonamiento en LLMs que traza la clase: el [scratchpad](/papers/scratchpad-nye-2021) demostró el principio en tareas simbólicas, la cadena de pensamiento lo generaliza y descubre la emergencia, y el *test-time compute* (o1, R1) lo lleva al extremo entrenando al modelo para generar cadenas largas por sí mismo. Tres ideas para internalizar: **formato > parámetros** (el mismo peso resuelve o falla según cómo se pida razonar), **emergencia** (ciertas capacidades no se extrapolan desde modelos pequeños, rompiendo la intuición de las leyes de escala suaves) y **razonamiento explícito ≠ razonamiento correcto** (la cadena es interpretable y suele ser fiel, pero no hay garantía de validez). El [chain-of-thought](/fundamentos/chain-of-thought) es hoy el fundamento transversal sobre el que se apoya todo el razonamiento moderno en LLMs.
