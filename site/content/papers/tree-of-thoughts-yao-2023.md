---
title: "Tree of Thoughts (2023)"
weight: 387
math: true
---

{{< paper-card
    title="Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
    authors="Shunyu Yao et al. (Princeton, Google DeepMind)"
    year="2023"
    venue="NeurIPS 2023"
    pdf="/papers/tree-of-thoughts-yao-2023.pdf" >}}
**Tree of Thoughts (ToT)** es un marco de inferencia que **generaliza a Chain-of-Thought (CoT)**: en vez de producir una cadena lineal de pasos de izquierda a derecha, deja que el modelo **explore un árbol de "pensamientos"** —unidades coherentes de texto que sirven de pasos intermedios— generando varios candidatos por paso, **autoevaluándolos** y navegando con **algoritmos de búsqueda clásicos (BFS/DFS con backtracking)** que deciden qué ramas expandir, cuáles podar y cuándo retroceder. La tesis: el mecanismo autorregresivo token a token de los LLM se parece al "Sistema 1" del pensamiento humano —rápido, asociativo— y puede aumentarse con un "Sistema 2" deliberado que mantiene alternativas y planifica. El salto empírico es dramático: en **Game of 24**, GPT-4 con CoT resuelve apenas el **4 %** de los problemas, mientras ToT alcanza el **74 %**. Es una pieza central de la [Clase 34](/clases/clase-34) porque conecta el razonamiento de LLMs con la búsqueda clásica de la IA.
{{< /paper-card >}}

---

## Contexto: los límites del razonamiento lineal

El punto de partida es una observación incómoda: por debajo de todo el progreso en razonamiento matemático y simbólico, los LLM siguen operando con el **mismo mecanismo autorregresivo original** —decisiones a nivel de token, una a una, de izquierda a derecha—. Apoyándose en la teoría de los procesos duales de la psicología cognitiva (Sloman, Stanovich, Kahneman), los autores identifican que a ese "Sistema 1" le falta un "Sistema 2" de planificación que (1) mantenga y explore alternativas en vez de comprometerse con una sola, y (2) evalúe el estado presente mirando adelante o retrocediendo.

Ese marco permite diagnosticar con precisión los métodos de prompting existentes. Formalizados como casos de un continuo —input-output (IO), [Chain-of-Thought](/fundamentos/chain-of-thought) y Self-Consistency (CoT-SC)—, todos comparten dos fallas: **localmente** no exploran continuaciones distintas dentro de un proceso de pensamiento (no ramifican), y **globalmente** carecen de planificación, lookahead o backtracking. El análisis de errores lo cuantifica: en Game of 24, cerca del **60 %** de las muestras de CoT ya han fracasado tras generar el primer paso (las primeras tres palabras, por ejemplo "4 + 9"). Una decisión inicial mala condena toda la cadena, y el decoding lineal no tiene forma de deshacerla.

## Método: los cuatro componentes de ToT

La propuesta enmarca cualquier problema como una **búsqueda en un árbol** donde cada nodo es un estado $s = [x, z_{1\cdots i}]$: la entrada más los pensamientos parciales acumulados. La innovación fundamental está en **de dónde salen las heurísticas**: no están programadas (como en [Deep Blue](/papers/alphago-silver-2016)) ni aprendidas con entrenamiento dedicado (como en AlphaGo), sino que las provee **el propio LLM razonando en lenguaje natural sobre los estados**. Una instancia concreta responde a cuatro preguntas de diseño modulares:

- **Descomposición del pensamiento.** Diseñar el tamaño del paso intermedio: un par de palabras (Crosswords), una línea de ecuación (Game of 24) o un párrafo de plan (Creative Writing). Debe ser suficientemente pequeño para generar candidatos diversos, pero suficientemente grande para poder evaluarlo.
- **Generador $G(p_\theta, s, k)$.** Produce $k$ candidatos por paso, ya sea por muestreo i.i.d. desde un prompt CoT (cuando el espacio es rico) o por propuesta secuencial con un "propose prompt" (cuando es restringido, para evitar duplicados).
- **Evaluador $V(p_\theta, S)$.** La heurística que le dice a la búsqueda qué estados conservar. Valora cada estado independientemente (un escalar 1-10 o `sure`/`likely`/`impossible`) combinando lookahead y sentido común, o **vota** entre estados comparándolos —una self-consistency "paso a paso"—.
- **Algoritmo de búsqueda.** **BFS** mantiene los $b$ estados más prometedores por nivel (árboles poco profundos, $T \le 3$); **DFS** explora el más prometedor hasta la solución o hasta que el evaluador lo declara imposible ($V \le v_{th}$), entonces **poda el subárbol y retrocede al padre**. Se dejan A\* y MCTS para trabajo futuro.

## Resultados

En **Game of 24** (100 juegos difíciles del sitio 4nums.com, BFS con $b=5$), el contraste es contundente: IO 7.3 %, CoT 4.0 %, CoT-SC (k=100) 9.0 %, frente a ToT con $b=1$ en **45 %** y $b=5$ en **74 %**. Incluso el mejor de 100 muestras de CoT (49 %) queda muy por debajo: el escalamiento por muestreo ciego rinde mucho menos que la búsqueda deliberada.

En **Creative Writing** (pasaje coherente de 4 párrafos), ToT logra un puntaje GPT-4 de **7.56** vs. CoT 6.93 vs. IO 6.19, y los humanos lo prefieren sobre CoT en 41 de 100 pares (vs. 21 en contra). En **Mini Crosswords 5×5** (DFS), ToT lleva el éxito a nivel de palabra del <16 % de IO/CoT al **60 %**, resolviendo 4 de 20 juegos. Las ablaciones son reveladoras: quitar el backtracking desploma el éxito de palabra a 20 %, y quitar la poda empeora el desempeño general —**el backtracking y una buena heurística de poda son componentes críticos, no adornos**—.

## Limitaciones

- **Costo de inferencia.** ToT requiere de 5 a 100 veces más tokens generados que CoT. En Creative Writing usa unas 5× más tokens y dinero ($0.32 vs. $0.06-0.07). La flexibilidad modular permite ajustar el compromiso costo-desempeño.
- **Dependencia del evaluador.** La heurística es tan buena como la capacidad del modelo de autoevaluarse. En Crosswords, el evaluador podó estados correctos por no reconocer palabras raras u obsoletas.
- **Necesidad selectiva.** La búsqueda deliberada no hace falta en tareas que GPT-4 ya domina (GSM8K, StrategyQA); su valor se materializa en problemas de planificación/búsqueda genuinos.
- **Solo inferencia.** Se usa un LLM off-the-shelf; los autores conjeturan que entrenar el modelo con decisiones contrafactuales de alto nivel podría potenciar aún más estas capacidades —una intuición que anticipa la línea de los modelos de razonamiento entrenados con RL—.

## Por qué importa para la Clase 34

ToT es, literalmente, un **puente entre dos tradiciones**: aporta a los LLM las intuiciones clásicas del problem solving de Newell y Simon (búsqueda heurística en un espacio combinatorio), y a la vez los LLM aportan a esos métodos clásicos algo que no tenían —una forma de resolver problemas difíciles de formalizar, como la escritura creativa, donde no existe función de evaluación programable—.

- **Precursor del [test-time compute](/fundamentos/test-time-compute).** ToT enuncia casi al pasar la apuesta que la generación de modelos de razonamiento (o1, R1) convertiría en paradigma: **gastar más cómputo en la inferencia** —explorando, evaluando y descartando caminos— vale más que forzar una respuesta en una sola pasada. La diferencia de implementación importa: ToT orquesta la búsqueda **externamente** con un algoritmo que llama al LLM como generador y evaluador, mientras los modelos de razonamiento posteriores **internalizan** esa deliberación en una sola cadena larga aprendida vía RL.
- **Posición frente a CoT y [Self-Consistency](/papers/self-consistency-wang-2022).** CoT rompió el mapeo directo entrada-salida introduciendo pasos intermedios, pero mantuvo la linealidad; CoT-SC agregó diversidad **entre** cadenas completas, pero sin exploración **dentro** de cada una. ToT unifica ambas ideas: exploración local (ramificar) más planificación global (evaluar, mirar adelante, retroceder), con IO/CoT/CoT-SC recuperados como casos degenerados de árboles de amplitud o profundidad 1.
- **Interpretabilidad.** Al operar sobre lenguaje legible en vez de valores implícitos de token, ToT mejora la interpretabilidad de las decisiones del modelo y la posibilidad de alineamiento humano —un beneficio que la [Clase 34](/clases/clase-34) valora especialmente para dominios de alto riesgo como la decisión clínica—.
