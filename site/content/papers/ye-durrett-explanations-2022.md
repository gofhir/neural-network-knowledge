---
title: "The Unreliability of Explanations (2022)"
weight: 385
math: true
---

{{< paper-card
    title="The Unreliability of Explanations in Few-Shot Prompting for Textual Reasoning"
    authors="Xi Ye, Greg Durrett (UT Austin)"
    year="2022"
    venue="NeurIPS 2022"
    pdf="/papers/ye-durrett-explanations-2022.pdf" >}}
¿Mejora el *in-context learning* cuando agregamos explicaciones al prompt de un LLM? Ye y Durrett lo miden sobre razonamiento textual (QA y NLI) y hallan un resultado de dos caras. Sobre **utilidad**: enchufar explicaciones da mejoras solo pequeñas o moderadas para OPT, GPT-3 (davinci) e InstructGPT (text-davinci-001) —incluso puede degradar—, con una **única excepción notable: text-davinci-002**, que mejora de forma sustancial en las tres tareas. Sobre **fiabilidad**: las explicaciones que genera un LLM **pueden no ser factuales** (alucinan hechos que el contexto contradice) ni **implicar la predicción** que acompañan, aun en un dataset sintético trivial. Lo que salva al trabajo de ser puro escepticismo es constructivo: la falta de factualidad **correlaciona con predicciones incorrectas**, así que la explicación sirve como **señal de verificación post-hoc** para calibrar. Este es el paper de la slide 30 de la [Clase 34](/clases/clase-34), "¿de dónde surge el CoT?".
{{< /paper-card >}}

---

## Contexto: el auge del "hazte explicar" y la pregunta por su fiabilidad

Los LLMs escalados aprenden tareas "en contexto" a partir de unos pocos ejemplos, sin actualizar parámetros (Brown et al., 2020), pero ese aprendizaje sigue siendo poco entendido: es sensible al orden de los ejemplos y a veces ni siquiera usa las etiquetas como uno esperaría (Min et al., 2022). Las herramientas clásicas de interpretabilidad (LIME, saliencia, gradientes) no sirven para un modelo accedido como caja negra vía API. Frente a esa opacidad surge una idea atractiva: **dejar que el modelo se explique a sí mismo**, la línea de los *scratchpads* (Nye et al., 2021) y del [Chain-of-Thought](/fundamentos/chain-of-thought) (Wei et al., 2022). Ye y Durrett observan que la evidencia previa de éxito se concentraba en tareas **simbólicas** (aritmética, ejecución de programas), y preguntan lo que casi nadie preguntaba en 2022: ¿el beneficio se traslada a razonamiento **textual**? Y más allá de la accuracy, ¿son **fiables** esas explicaciones, o son solo texto convincente que puede engañar al usuario?

## Método

Se evalúa sobre tres datasets en inglés (250 ejemplos de test cada uno): **Synth** (QA multi-hop sintético, diseñado simétrico para eliminar atajos, con factualidad juzgable por reglas), **AdvHotpot** (HotpotQA con dos párrafos de soporte y dos distractores adversariales) y **E-SNLI** (NLI con explicaciones humanas abstractas, donde "factualidad" casi no aplica). Se comparan dos paradigmas de prompting: **Explain-then-Predict (E-P)**, donde la explicación va antes de la etiqueta y puede influir en ella —la categoría del CoT—, y **Predict-then-Explain (P-E)**, donde la predicción va primero y, con decodificación greedy, la explicación posterior ya no la altera. Se prueban cuatro modelos: OPT (175B), GPT-3 (davinci), InstructGPT (text-davinci-001) y text-davinci-002.

La pieza constructiva es un **calibrador lineal** de dos parámetros que combina las probabilidades $\mathbf{p}$ con un escalar $v$ que aproxima la factualidad de la explicación:

$$\hat{\mathbf{p}} = \mathrm{softmax}\!\left(W\,[\mathbf{p}; v] + b\right)$$

Como no hay forma automática perfecta de medir factualidad, se la aproxima por **solapamiento léxico** entre las oraciones de la explicación y los párrafos del contexto, tomando el mínimo sobre las oraciones (todas deben ser factuales para que la explicación entera lo sea).

## Resultados

**Utilidad.** Para InstructGPT las ganancias son modestas: en Synth, E-P sube 54.8 → 58.5; en AdvHotpot, 56.8 → 59.4. La excepción es **text-davinci-002**, que despega en las tres tareas: Synth **72.0 → 86.9 (E-P)**, AdvHotpot **77.7 → 82.4**, E-SNLI **69.1 → 75.6**. Este es justamente el salto que destaca la slide del curso.

**Fiabilidad.** Aparece un desacople entre predicción y "razonamiento": los LLMs generan explicaciones **consistentes** (>80% con el prompt adecuado) pero **menos factuales**. En Synth con InstructGPT + P-E, la consistencia trepa a 95.2% pero la factualidad cae a 51.6%. El caso paradigmático: el modelo responde "Croatian" con una prosa impecable que **inventa** que la persona es fotógrafa croata cuando el contexto dice claramente que es ucraniana.

**La falta de fiabilidad como señal.** El giro constructivo: accuracy y factualidad **correlacionan**. En AdvHotpot, factualidad y correctitud de InstructGPT coinciden el **80.0%** de las veces, muy por encima de la accuracy bruta (62.0%). Chequear la factualidad y rechazar pares no factuales sube P-E en Synth de 52.4% a **74.8%**. Con el calibrador basado en explicaciones, E-SNLI alcanza **68.5%**, unos **12 puntos** sobre el few-shot vanilla (56.8%), superando también al calibrador basado solo en probabilidades.

## Aclaración: la hipótesis del código es del profesor, no del paper

La slide 30 usa este trabajo como respaldo de "el CoT despega en text-davinci-002" y de ahí se hipotetiza que la diferencia viene del entrenamiento con código. Conviene separar con precisión:

- **Lo que el paper sí demuestra:** empíricamente, las explicaciones dan mejoras sustanciales solo con text-davinci-002. Dato sólido y reproducible.
- **Lo que el paper NO afirma:** que el salto se deba al código. Los autores dicen explícitamente que **no saben** qué lo causa, que las diferencias entre text-davinci-002 e InstructGPT **no están documentadas** en ninguna publicación ni blog, y que por esa falta de transparencia **dudan en hacer afirmaciones científicas**.
- **De dónde viene la hipótesis del código:** de la comunidad (notablemente Yao Fu et al., 2022), no de Ye y Durrett. Es plausible y ampliamente citada, pero externa a este paper y aún especulativa.

En síntesis: este es la **evidencia del salto**, no la explicación del salto.

## Limitaciones

- **La feature de factualidad es débil.** El solapamiento léxico es una señal ruidosa; un modelo de *entailment* fuerte la haría mejor, pero rompería el setting de caja negra pura.
- **Alcance acotado.** Los resultados aplican a QA y NLI textual, no se extrapolan sin más al razonamiento simbólico donde el CoT sí rinde grande.
- **Opacidad de los modelos.** No poder saber en qué difiere text-davinci-002 impide cualquier conclusión causal.
- **Requiere datos extra** para entrenar el calibrador (aunque los aprovecha porque no caben en el prompt).

## Por qué importa para la Clase 34

Este paper ocupa un lugar bisagra en la [Clase 34](/clases/clase-34). Por un lado **fecha empíricamente el nacimiento del [CoT](/fundamentos/chain-of-thought) efectivo** en razonamiento textual, dando material a la discusión "¿de dónde surge el CoT?". Por otro es un **contrapunto crítico saludable**: justo cuando Wei et al. (2022) mostraban que "pensar paso a paso" desbloquea capacidades, Ye y Durrett recuerdan que **una cadena verbalizada no es un registro fiel del proceso interno del modelo** —puede ser consistente pero no factual, convincente pero falsa—. Esa distinción entre *plausibilidad* y *fidelidad* anticipa toda la línea posterior sobre *faithfulness* del CoT. El aporte constructivo —usar la (in)fiabilidad como señal para calibrar y abstenerse— envejece muy bien: es un precursor conceptual de los verificadores, del [self-consistency](/papers/self-consistency-wang-2022) y de los *LLM-as-a-judge*. La lección transversal: el razonamiento explícito de un LLM es a la vez una **herramienta de desempeño** y una **superficie de verificación**, y ambas cosas exigen escepticismo medido. En salud esto es directo y grave: una explicación plausible **no** garantiza que la decisión sea correcta ni esté fundamentada; hay que **verificar, no confiar en la narrativa**.
