---
title: "Razonamiento en IA"
weight: 113
math: true
---

El **razonamiento** es la capacidad de ir más allá de reconocer patrones para **derivar conclusiones nuevas**, aplicar conocimiento a situaciones no vistas y explicar *por qué* algo ocurre. Es, quizás, la frontera más discutida de la inteligencia artificial moderna: los modelos de deep learning brillan como **sistemas asociativos** —memorizan y generalizan patrones estadísticos— pero tropiezan cuando una tarea exige varios pasos deliberados de inferencia. Este fundamento acompaña a la [Clase 34](/clases/clase-34): recorre qué es razonar (la escalera de causalidad de Pearl, la abstracción, la sistematicidad), por qué el deep learning puro se queda corto, y cómo la comunidad intentó dotar de razonamiento a las redes —de las arquitecturas con estructura ([memoria externa](/fundamentos/redes-de-memoria)) a las técnicas de prompting como el [Chain-of-Thought](/fundamentos/chain-of-thought) y el [cómputo en tiempo de inferencia](/fundamentos/test-time-compute).

---

## 1. Tipos de razonamiento: la escalera de causalidad

Judea Pearl propuso una jerarquía de tres niveles —la **escalera de la causalidad**— que ordena las preguntas que un sistema puede responder:

1. **Razonamiento asociativo** (*ver*). Basado en **correlaciones**: ¿qué me dice observar $X$ sobre $Y$? Es $P(Y\mid X)$. Todo el deep learning supervisado vive aquí.
2. **Razonamiento intervencional** (*hacer*). ¿Qué pasará si **realizo** cierta acción? Es $P(Y\mid \text{do}(X))$, distinto de la simple correlación porque implica intervenir sobre el sistema, no solo observarlo.
3. **Razonamiento contrafactual** (*imaginar*). Dados los resultados de una acción, ¿qué habría pasado si hubiera actuado distinto? *"¿Por qué mis plantas no crecieron?"* exige imaginar un mundo alternativo.

{{< concept-alert type="clave" >}}
La mayor parte del machine learning moderno opera en el **primer peldaño** (asociación). Subir a intervención y contrafactuales —el terreno del razonamiento causal— requiere **modelos causales** del mundo, no solo estadística de co-ocurrencias. Esa brecha es el hilo conductor de la Clase 34.
{{< /concept-alert >}}

---

## 2. Las habilidades que subyacen al razonar

La clase identifica dos capacidades cognitivas como cimiento del razonamiento:

**Construcción de abstracciones.** Extraer el concepto subyacente a partir de instancias específicas (de muchos perros concretos, el concepto "perro"; de este, la abstracción jerárquica "raza"). Una abstracción captura lo esencial y **omite lo irrelevante**, lo que facilita el procesamiento. En palabras de la clase: *el conocimiento son abstracciones*.

**Sistematicidad.** Aplicar esas abstracciones en **contextos distintos** a los vistos durante el entrenamiento, y **componer conceptos** ("un mono tocando guitarra en la playa"). Las redes neuronales muestran *cierta* sistematicidad —los generadores de imágenes componen conceptos— pero **incompleta**: fallan en composiciones que exigen combinaciones no vistas (un reloj marcando una hora específica, colores y formas nunca emparejados). Juntas, abstracción y sistematicidad permiten construir y usar **modelos causales**.

{{< concept-alert type="recordar" >}}
Borges lo capturó en *Funes el memorioso*: Funes recordaba todo pero *"no era muy capaz de pensar. Pensar es olvidar diferencias, es generalizar, abstraer."* Un sistema que solo memoriza (asociación perfecta) no razona; razonar exige **descartar** lo irrelevante para generalizar.
{{< /concept-alert >}}

---

## 3. Sistema 1 y Sistema 2

Una lente influyente (Kahneman) distingue dos modos de pensamiento, que la clase mapea a la IA:

- **Sistema 1** — comportamientos **reactivos**, rápidos, automáticos, intuitivos. Es lo que el deep learning asociativo hace bien: reconocer un rostro, completar una frase.
- **Sistema 2** — comportamientos **deliberados**, lentos, secuenciales, que requieren esfuerzo y control. Resolver un acertijo lógico, planificar, verificar un cálculo.

El deep learning clásico es esencialmente **Sistema 1**. Buena parte de la investigación en razonamiento busca dotar a los modelos de capacidades de **Sistema 2** —deliberación explícita, búsqueda, verificación—, ya sea con arquitecturas especiales o con técnicas de inferencia.

---

## 4. Deep learning como sistema asociativo

La evidencia de que el deep learning es fundamentalmente asociativo aparece en varios frentes:

- En **aprendizaje reforzado**, el DRL funciona bien en juegos donde la señal de recompensa está **cerca de la acción** (asociativos), pero fracasa en juegos de recompensa **esporádica** como *Montezuma's Revenge*. Técnicas como **Go-Explore** —guardar estados prometedores y volver a ellos si la exploración falla— inyectan una **estructura causal** ("¿qué hubiera pasado si...?") de forma externa, no intrínseca al modelo.
- Los **LLMs** pre-entrenados ([GPT-3](/papers/gpt-3-brown-2020)) tienen excelente desempeño en tareas diversas y exhiben comportamientos [emergentes](/fundamentos/in-context-learning) como el *in-context learning*, pero **fallan en aritmética simple** (sumar dígitos), señal de que memorizan patrones más que ejecutar algoritmos.

---

## 5. Dar estructura a la red: el enfoque neuro-simbólico

Una primera vía para aumentar el razonamiento fue **darle a la red neuronal la estructura de un computador clásico** —que sí generaliza sistemáticamente (un algoritmo de grafos funciona en *cualquier* grafo). Dos hitos:

- Las **[redes con memoria externa](/fundamentos/redes-de-memoria)** —Neural Turing Machine y Differentiable Neural Computer (Graves et al., 2016)— acoplan una red a una memoria direccionable, imitando la separación cómputo/memoria de una máquina de Turing. (Cubiertas en la [Clase 30](/clases/clase-30).)
- Las **Compositional Attention Networks (MAC)** (Hudson & Manning, 2018) descomponen el razonamiento visual en una secuencia de pasos de atención (control, lectura, escritura), logrando razonamiento composicional diferenciable en CLEVR. El benchmark **CLEVR-CoGenT** —entrenar con ciertas combinaciones forma-color y evaluar con otras— mide justamente la **sistematicidad**: los modelos tropiezan al componer atributos no vistos juntos.

---

## 6. Razonamiento en LLMs: el giro moderno

El descubrimiento clave de la era LLM es que **no siempre hace falta cambiar la arquitectura**: a veces basta cambiar *cómo se le pide* al modelo que responda, o *cuánto cómputo se le da en inferencia*. Esta línea —el corazón práctico de la Clase 34— se desarrolla en dos fundamentos hermanos:

- **[Chain-of-Thought y prompting para razonar](/fundamentos/chain-of-thought)**: del *scratchpad* (Nye, 2021) al Chain-of-Thought (Wei, 2022), self-consistency y Tree-of-Thoughts. La idea: hacer que el modelo **escriba sus pasos intermedios**, transformando una respuesta de un solo golpe (Sistema 1) en una deliberación explícita (Sistema 2).
- **[Cómputo en tiempo de inferencia (test-time compute)](/fundamentos/test-time-compute)**: del muestreo repetido (Pass@k, *Large Language Monkeys*) al entrenamiento por RL con recompensa verificable (DeepSeek-R1, o1) y el debate sobre si el RL **crea** capacidades o solo **reordena** las que el modelo ya tenía (Yue et al., 2025).

---

## 7. Los límites persisten

A pesar del progreso, el razonamiento de los LLMs sigue siendo frágil, como ilustran los ejemplos de la clase:

- **Sobreajuste a plantillas conocidas.** Variantes sutiles del problema de Monty Hall o del acertijo del "lobo, la cabra y el repollo" hacen que los modelos **recuperen la solución memorizada** en vez de razonar sobre el enunciado real. Un modelo puede resolver el clásico y fallar en su variante trivial.
- **Robustez.** Benchmarks como **Math-Perturb** perturban problemas matemáticos para medir si el modelo razona o memoriza; grandes caídas de desempeño delatan memorización.
- **Abstracción genuina.** El **ARC** (Abstraction and Reasoning Corpus) de Chollet mide la **eficiencia en adquirir habilidades nuevas** con pocos ejemplos —resistente a la memorización a fuerza de datos— y sigue siendo difícil para los LLMs.

{{< concept-alert type="advertencia" >}}
La lección de cierre de la clase: una respuesta fluida y bien argumentada **no es** prueba de razonamiento correcto. Los LLMs pueden producir explicaciones plausibles que no se sostienen (Ye & Durrett, 2022). En dominios de alto riesgo —como la medicina— hay que **verificar** el razonamiento, no confiar en su superficie.
{{< /concept-alert >}}

---

## 8. Relevancia para salud y sistemas clínicos

Para quien construye software clínico, la distinción asociación/razonamiento es más que filosófica. Un modelo que **correlaciona** síntomas con diagnósticos (peldaño 1) es útil pero peligroso cuando se le pide **intervenir** ("¿qué pasa si administro este fármaco?", peldaño 2) o razonar **contrafactualmente** ("¿por qué este paciente no respondió?", peldaño 3) —preguntas causales que la correlación no puede responder. Las técnicas de razonamiento explícito (Chain-of-Thought) aportan **trazabilidad** —un registro auditable de los pasos que llevaron a una decisión—, pero la advertencia de Ye & Durrett es crítica: una cadena de razonamiento convincente puede ser incorrecta o inventada. En contextos como el matching de pacientes o el apoyo a decisiones clínicas, el razonamiento debe ser **verificable**, no solo persuasivo.

---

## Referencias

- Pearl, J. & Mackenzie, D. (2018). *The Book of Why*. — la escalera de la causalidad.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. — Sistema 1 y Sistema 2.
- Hudson, D. & Manning, C. (2018). *Compositional Attention Networks for Machine Reasoning* (MAC). ICLR.
- Chollet, F. (2019). *On the Measure of Intelligence* (ARC). arXiv:1911.01547.
- Fundamentos hermanos: [Chain-of-Thought](/fundamentos/chain-of-thought) · [Test-time compute](/fundamentos/test-time-compute) · [Redes de memoria](/fundamentos/redes-de-memoria).
