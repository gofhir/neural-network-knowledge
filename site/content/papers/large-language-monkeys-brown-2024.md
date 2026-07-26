---
title: "Large Language Monkeys: Repeated Sampling (2024)"
weight: 388
math: true
---

{{< paper-card
    title="Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"
    authors="Bradley Brown et al. (Stanford, Oxford)"
    year="2024"
    venue="arXiv:2407.21787"
    pdf="/papers/large-language-monkeys-brown-2024.pdf" >}}
El paper explora una idea sencilla con consecuencias profundas: **el cómputo de inferencia como un eje de escalamiento independiente del de entrenamiento**. En vez de dar al modelo un solo intento por problema, se lo hace **muestrear repetidamente** muchas soluciones candidatas con temperatura positiva y luego se selecciona una con un verificador. La métrica central es la **cobertura** (*coverage*): la fracción de problemas resueltos por **al menos una** muestra. El hallazgo: la cobertura crece de forma suave y predecible a lo largo de **cuatro órdenes de magnitud** (de 1 a 10.000 muestras), modelable con una ley de potencia exponenciada —evidencia de *leyes de escala en tiempo de inferencia*—. La segunda tesis es una advertencia crucial: escalar la cobertura solo se traduce en desempeño real cuando existe un **verificador**. Es una pieza de la [Clase 34](/clases/clase-34), citada bajo *Pass@k*.
{{< /paper-card >}}

---

## Contexto: cómputo de entrenamiento vs. cómputo de inferencia

Durante la última década, la mejora de los LLM se explicó casi enteramente por **escalar el cómputo de entrenamiento**: modelos más grandes, corridas más largas, datasets mayores. Las leyes de escala de entrenamiento (Kaplan et al., Chinchilla) formalizaron esa relación como una ley de potencia entre pérdida y cómputo. El cómputo de **inferencia**, en cambio, recibió inversión escasa: aunque técnicas como chain-of-thought elevan la calidad al costo de salidas más largas, en la práctica usuarios y desarrolladores **restringen el modelo a un solo intento por problema**.

El paper propone tratar la inferencia como una segunda palanca de escala mediante la técnica más simple imaginable: **muestreo repetido**. El título alude al *teorema del mono infinito* —un mono tecleando al azar durante tiempo infinito acabaría escribiendo cualquier texto—; si un LLM genera suficientes intentos independientes, es probable que alguno acierte. El precedente inspirador es **AlphaCode** (Li et al., *Science* 2022), que descubrió que el desempeño sigue mejorando hasta un millón de muestras por problema.

## Contribución: cobertura, precisión y Pass@k

El paper articula tres observaciones: (1) el muestreo repetido produce **grandes mejoras de cobertura**, haciendo posible **amplificar un modelo débil con muchas muestras** hasta superar a uno más capaz de un solo intento; (2) la relación cobertura-muestras se modela con una **ley de potencia exponenciada**; (3) sin verificadores automáticos, los métodos de selección se **estancan más allá de ~100 muestras**.

La efectividad depende de **dos propiedades separables**: la **cobertura** (¿qué fracción de problemas resolvemos con *cualquiera* de las muestras?) y la **precisión** (¿con qué frecuencia *identificamos* las muestras correctas?). En el contexto de código, la cobertura **es exactamente la métrica pass@k** de Chen et al. (2021). Para reducir varianza se usa el estimador insesgado:

$$\text{pass@}k = \frac{1}{\#\text{problemas}} \sum_{i=1}^{\#\text{problemas}} \left(1 - \frac{\binom{N-C_i}{k}}{\binom{N}{k}}\right),$$

donde $C_i$ es el número de muestras correctas del problema $i$: el término $\binom{N-C_i}{k}/\binom{N}{k}$ es la probabilidad de que las $k$ muestras elegidas sean **todas** incorrectas, y su complemento, la de que al menos una acierte.

## Resultados de cobertura

Sobre cinco tareas de tipo pasa/falla (GSM8K, MATH, MiniF2F-MATH, CodeContests, SWE-bench Lite) con hasta **10.000 muestras por problema**, la cobertura mejora suavemente en todas. El resultado más llamativo: en **SWE-bench Lite**, DeepSeek-Coder-V2-Instruct resuelve **15,9 % de los issues con una sola muestra pero 56 % con 250 muestras**, superando el estado del arte de un solo intento (43 %) por 13 puntos. Cuando a todos los modelos se les da un solo intento, GPT-4o los supera; pero al aumentar $k$, los tres modelos más débiles superan su desempeño de un solo intento. Esa es la tesis de la **amplificación**.

El efecto es robusto entre tamaños y familias, con los **modelos más pequeños mostrando los aumentos más pronunciados**: en CodeContests, Gemma-2B crece **más de 300×** (pass@1 0,02 % → pass@10k 7,1 %); en MATH, Pythia-160M crece de 0,27 % a **57 %**. La excepción es Pythia en CodeContests: **cobertura cero** incluso con 10.000 muestras, porque fue entrenada con pocos datos de código —el muestreo repetido amplifica una capacidad que el modelo **ya posee latentemente**; si la probabilidad de acierto es exactamente cero, ningún presupuesto la rescata—.

Modelando el logaritmo de la cobertura como $\log(c) \approx a\,k^{b}$, y exponenciando, $c \approx \exp(a\,k^{b})$: un ajuste de **ley de potencia exponenciada** que sobre eje $x$ logarítmico se ve casi log-lineal en varios órdenes de magnitud. La economía es el aporte práctico clave: en SWE-bench Lite, **cinco muestras de DeepSeek** (0,0072 USD/intento) resuelven más issues que un solo intento de Claude 3.5 Sonnet o GPT-4o (29,62 % vs. 26,70 % y 24,00 %) a **más de 3× menos costo** (10,8 vs. 51 y 39 USD).

## El rol del verificador: la limitación central

Aquí está el corazón crítico: toda la sección anterior mide **cobertura**, que asume un verificador oráculo perfecto. Pero la cobertura es solo una **cota superior** del desempeño real; convertirla en aciertos requiere resolver la **precisión** —encontrar "la aguja en el pajar"—. De las cinco tareas, solo GSM8K y MATH carecen de verificador automático. Probando tres métodos de selección sobre 10.000 muestras (votación por mayoría, modelo de recompensa + Best-of-N, y su combinación), el resultado es contundente: con Llama-3-8B-Instruct en MATH, la **cobertura crece de 82,9 % (100 muestras) a 98,44 % (10.000)**, pero el mejor método de selección apenas pasa de **40,50 % a 41,41 %**. Los tres se **estancan alrededor de las 100 muestras** mientras la cobertura sigue subiendo: la brecha **crece** con el número de muestras.

Para la votación por mayoría la saturación es intuitiva —una solución correcta rara no cambia la respuesta más común—; lo preocupante es que el modelo de recompensa tampoco escala. ¿Es que verificar es tan difícil como resolver? Evaluando manualmente 105 cadenas de razonamiento de muestras correctas en GSM8K, **más del 90 % son fieles**: hay señal que un verificador podría explotar, pero los actuales no la aprovechan.

## Limitaciones

- **La cobertura sin verificador es una ilusión de rigor.** El beneficio central solo se materializa con verificación confiable; sin ella, la respuesta correcta puede estar presente pero ser indistinguible.
- **Verificadores imperfectos.** Incluso los "automáticos" fallan: en SWE-bench Lite, el 11,3 % de los problemas tiene tests inestables (*flaky*); en CodeContests, 35 de 122 problemas tienen soluciones correctas que fallan sus tests por salidas múltiples o casos de prueba mal generados.
- **Dependencia de capacidad latente.** El muestreo no crea capacidades; solo amplifica las que el modelo ya tiene (cobertura cero de Pythia en código).

## Por qué importa para la Clase 34

Este paper es una de las piezas fundacionales del paradigma de **[cómputo en tiempo de test](/fundamentos/test-time-compute)** que domina la generación actual de modelos de razonamiento. Su relevancia opera en tres niveles:

- **Pass@k y la intuición del muestreo.** El paper cuantifica la afirmación citada en la clase: si se muestrean muchas respuestas, es probable que alguna sea correcta. La cobertura *es* pass@k, y su crecimiento log-lineal le da forma matemática a esa intuición —con el matiz esencial de que **pass@k es una cota superior optimista** que solo se materializa con un buen verificador—.
- **El puente hacia o1 y [R1](/papers/deepseek-r1-2025).** El muestreo repetido es la forma más simple de gastar más cómputo en inferencia. Los modelos de razonamiento posteriores internalizan la idea: en vez de muestrear muchas trayectorias cortas *en paralelo* y verificarlas externamente, aprenden por RL a producir una sola trayectoria larga que explora, verifica y corrige *dentro* de su cadena de pensamiento. Este trabajo estableció empíricamente que el cómputo de inferencia es un eje de escala legítimo, con sus propias leyes de potencia —la premisa que hace sensato entrenar modelos que "piensen más"—.
- **[Self-Consistency](/papers/self-consistency-wang-2022) como caso particular.** La votación por mayoría que el paper evalúa como método de precisión es exactamente Self-Consistency. El aporte crítico es mostrar **por qué se estanca**: es insensible a las soluciones correctas raras, así que satura alrededor de las 100 muestras aun cuando la cobertura sigue creciendo. Esto motiva la investigación en **verificadores** —modelos de recompensa de proceso, process supervision— como complemento indispensable del muestreo.
