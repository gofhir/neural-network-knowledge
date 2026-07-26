---
title: "Chain-of-Thought y prompting para razonar"
weight: 114
math: true
---

El **Chain-of-Thought** (CoT, cadena de pensamiento) es la técnica que destrabó el razonamiento en los grandes modelos de lenguaje sin tocar sus pesos: en lugar de pedirle al modelo la respuesta de un solo golpe, se le pide que **escriba los pasos intermedios** —igual que un estudiante que "muestra su trabajo". Este cambio, aparentemente trivial, convierte una respuesta reactiva (Sistema 1) en una **deliberación explícita** (Sistema 2) y mejora drásticamente el desempeño en tareas de varios pasos. Este fundamento acompaña a la [Clase 34](/clases/clase-34) y desarrolla la familia de técnicas de prompting para razonar: del *scratchpad* al Chain-of-Thought, la self-consistency y el Tree-of-Thoughts. Es el complemento de fundamentos de [Razonamiento en IA](/fundamentos/razonamiento) y [Test-time compute](/fundamentos/test-time-compute).

---

## 1. El problema: un solo forward pass no basta

Un LLM produce cada token con una cantidad **fija** de cómputo (una pasada por la red). Si le pides directamente el resultado de un problema de varios pasos —una suma de muchos dígitos, un problema de palabras de matemáticas—, debe "resolverlo todo de una" dentro de esa única pasada, sin espacio para cálculos intermedios. Por eso [GPT-3](/papers/gpt-3-brown-2020), pese a su escala, **falla en aritmética simple**: no tiene dónde "anotar" los pasos.

La observación clave es que darle al modelo **espacio para pensar en voz alta** —tokens intermedios donde desarrollar el cálculo— le permite descomponer el problema y usar más cómputo, proporcional a la longitud del razonamiento.

---

## 2. Scratchpad: el precursor

Nye et al. (2021) introdujeron el **scratchpad** ("bloc de notas"): entrenar al modelo para que **emita los pasos intermedios** de un cálculo antes de la respuesta final. En vez de mapear `2+3 → 5` directamente, el modelo aprende a escribir el procedimiento (los acarreos de una suma, la evaluación término a término de un polinomio, la traza de ejecución de un programa Python línea por línea) y *luego* la respuesta. La metáfora de la clase: *"como al mono recolectando cocos, lo guiamos fuertemente para que aprenda el algoritmo."*

El scratchpad requería **fine-tuning** con trazas de ejemplo. Su límite práctico es la ventana de contexto (los pasos intermedios ocupan tokens) y la necesidad de supervisión con trazas. Pero probó la idea central: **externalizar el cómputo intermedio** mejora radicalmente las tareas multi-paso. → [análisis](/papers/scratchpad-nye-2021)

---

## 3. Chain-of-Thought: sin fine-tuning, solo prompting

Wei et al. (2022) dieron el salto decisivo: lograr lo mismo **sin entrenar**, solo con el prompt. El Chain-of-Thought provee unos pocos **ejemplos few-shot** donde cada demostración es un triple $\langle \text{entrada}, \text{cadena de razonamiento}, \text{salida}\rangle$. Al ver ejemplos resueltos "paso a paso", el modelo **imita ese formato** y desarrolla su propio razonamiento para el problema nuevo.

$$
\text{Prompt} = \underbrace{(x_1, c_1, y_1), \dots, (x_k, c_k, y_k)}_{k \text{ ejemplos con cadena } c_i}, \; x_{\text{test}} \;\longrightarrow\; \text{modelo genera } (c_{\text{test}}, y_{\text{test}}).
$$

El ejemplo canónico es **GSM8K** (problemas de matemáticas de primaria): con prompting estándar, PaLM 540B rondaba el 18%; con Chain-of-Thought saltó a ~57%, superando incluso a modelos afinados con verificador. En **BIG-Bench Hard** (Suzgun et al., 2022), agregar CoT permitió a Codex superar el desempeño humano promedio en 17 de 23 tareas donde antes ningún LLM lo lograba.

{{< concept-alert type="clave" >}}
El Chain-of-Thought es una **habilidad emergente**: solo aparece a partir de cierta **escala** (~100B parámetros). En modelos pequeños, pedir el razonamiento paso a paso **empeora** el resultado —producen cadenas incoherentes. Esto conecta CoT con las [habilidades emergentes](/fundamentos/razonamiento) de los LLMs grandes: es una capacidad que no se extrapola de modelos menores.
{{< /concept-alert >}}

### 3.1 ¿De dónde sale esta capacidad?

Empíricamente, el CoT empezó a ser marcadamente efectivo en la versión **text-davinci-002** de GPT-3 (Ye & Durrett, 2022, documentan el salto). Una hipótesis popular —discutida en la clase— atribuye la mejora al **entrenamiento con grandes cantidades de código**, donde el razonamiento estructurado paso a paso es omnipresente. Es una hipótesis plausible pero **no probada**: el paper de Ye & Durrett documenta el salto sin afirmar su causa.

---

## 4. Cuidado: la cadena puede mentir

Un hallazgo incómodo (Ye & Durrett, 2022): las explicaciones que genera un LLM pueden **no ser factuales ni implicar su propia predicción**, incluso en tareas simples. Una cadena de razonamiento fluida y convincente **no garantiza** que el modelo haya razonado correctamente —puede ser una racionalización *post-hoc*. La buena noticia: las explicaciones lógicamente consistentes **co-ocurren** más con predicciones correctas, así que sirven como señal (imperfecta) para verificar. La lección: el CoT aporta **trazabilidad**, no **corrección garantizada**. → [análisis](/papers/ye-durrett-explanations-2022)

---

## 5. Self-Consistency: muestrear y votar

Wang et al. (2022) mejoraron el CoT con una idea de la estadística: en vez de generar **una** cadena con decoding greedy, **muestrear muchas** cadenas diversas y quedarse con la respuesta **más frecuente** (voto por mayoría). La intuición: un problema correcto admite **múltiples caminos de razonamiento** que convergen a la misma respuesta, mientras que los errores tienden a dispersarse. Formalmente, se **marginaliza** sobre los caminos de razonamiento:

$$
\hat y = \arg\max_{y} \sum_{i=1}^{m} \mathbb{1}\big[\,y_i = y\,\big], \qquad (c_i, y_i) \sim p_\theta(\cdot \mid \text{prompt}).
$$

Self-consistency mejora sustancialmente sobre CoT en GSM8K, SVAMP, AQuA y otros. Su costo: $m$ veces más cómputo de inferencia —el primer ejemplo de **cambiar cómputo por precisión** en tiempo de inferencia. → [análisis](/papers/self-consistency-wang-2022)

---

## 6. Tree-of-Thoughts: de la cadena al árbol

Yao et al. (2023) generalizaron el CoT de una **cadena lineal** a un **árbol de búsqueda**. En el **Tree-of-Thoughts** (ToT), el modelo:

1. **Genera** varios "pensamientos" candidatos (pasos intermedios) en cada punto.
2. **Auto-evalúa** cada estado parcial (mediante value o votación) para estimar cuán prometedor es.
3. **Busca** por el árbol (BFS/DFS) con posibilidad de **backtracking** cuando un camino no lleva a ninguna parte.

Esto acopla el razonamiento del LLM con la **búsqueda clásica de IA** —el corazón del Sistema 2— y logra saltos enormes en tareas que exigen exploración: en el *Game of 24*, el CoT resolvía ~4% y el ToT ~74%. El precio, otra vez, es **más cómputo de inferencia**. → [análisis](/papers/tree-of-thoughts-yao-2023)

{{< concept-alert type="recordar" >}}
La familia CoT → self-consistency → ToT dibuja una progresión clara: **una cadena** (razonar en voz alta) → **muchas cadenas** (muestrear y votar) → **un árbol** (buscar y retroceder). Todas comparten el mismo motor: **gastar más cómputo en inferencia** para razonar mejor —la puerta de entrada al [test-time compute](/fundamentos/test-time-compute) y a los modelos de razonamiento como o1 y DeepSeek-R1.
{{< /concept-alert >}}

---

## 7. Relevancia para salud y sistemas clínicos

En software clínico, el Chain-of-Thought tiene un atractivo inmediato: **auditabilidad**. Un modelo que expone su razonamiento paso a paso —por qué considera que dos registros son el mismo paciente, cómo llegó a una sugerencia de dosis— produce una **traza revisable** por un humano, muy superior a una respuesta opaca. Pero la advertencia de Ye & Durrett es ineludible en medicina: esa traza es una **hipótesis revisable, no una prueba**. Una cadena convincente puede ocultar un error o ser una racionalización inventada. La práctica correcta combina la trazabilidad del CoT con **verificación externa** (contra la fuente de datos, contra reglas clínicas) y, donde sea posible, técnicas como self-consistency para reducir la varianza de decisiones críticas.

---

## Referencias

- Nye, M. et al. (2021). *Show Your Work: Scratchpads for Intermediate Computation with Language Models*. arXiv:2112.00114.
- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS.
- Suzgun, M. et al. (2022). *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them*. arXiv:2210.09261.
- Ye, X. & Durrett, G. (2022). *The Unreliability of Explanations in Few-Shot Prompting for Textual Reasoning*. NeurIPS.
- Wang, X. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023.
- Yao, S. et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. NeurIPS.
- Fundamentos hermanos: [Razonamiento en IA](/fundamentos/razonamiento) · [Test-time compute](/fundamentos/test-time-compute).
