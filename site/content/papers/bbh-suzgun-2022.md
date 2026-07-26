---
title: "BIG-Bench Hard y Chain-of-Thought (2022)"
weight: 384
math: true
---

{{< paper-card
    title="Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them"
    authors="Mirac Suzgun et al. (Google, Stanford)"
    year="2022"
    venue="arXiv:2210.09261 (Findings ACL 2023)"
    pdf="/papers/bbh-suzgun-2022.pdf" >}}
BIG-Bench se diseñó para reunir tareas "más allá de las capacidades de los modelos actuales", pero al escalar PaLM a 540B parámetros el mejor modelo ya superaba al evaluador humano promedio en el **65%** de las tareas. Los autores curan **BIG-Bench Hard (BBH)**: un subconjunto de **23 tareas** para las cuales **ningún modelo previo había superado al humano promedio**. Sobre ellas aplican [chain-of-thought prompting](/papers/chain-of-thought-wei-2022) —insertar pasos de razonamiento intermedios en los ejemplos few-shot— frente al prompting estándar *answer-only*. El resultado central: CoT permite a **Codex (`code-davinci-002`) superar al humano promedio en 17 de 23 tareas** (PaLM 540B en 10, InstructGPT en 15), frente a apenas 5 de 23 con answer-only. Con un matiz que el paper subraya: ese "humano" es el evaluador **promedio, no el experto** —Codex sigue 20+ puntos por debajo del mejor humano—, y la ganancia se concentra en tareas **algorítmicas / simbólicas multi-paso**. Es la evidencia empírica de la slide 29 de la [Clase 34](/clases/clase-34): los LLMs sí razonan, pero hay que darles el espacio para hacerlo paso a paso.
{{< /paper-card >}}

---

## Contexto: evaluar razonamiento y el subconjunto "hard"

**BIG-Bench** (Srivastava et al., 2022) es un benchmark colaborativo de más de 200 tareas de texto aportadas por la comunidad, con **baselines humanos** establecidos por evaluadores que resolvieron manualmente cada tarea. Para las 23 tareas de BBH, el humano promedio es **67.7%** y el humano máximo **94.4%**. La paradoja que motiva el trabajo: aunque BIG-Bench se creó para desafiar a los modelos, PaLM 540B ya vencía al humano promedio en el 65% de las tareas. ¿Qué caracteriza al 35% restante? ¿Son tareas fundamentalmente irresolubles, o simplemente se las evaluaba con la técnica de prompting equivocada? La hipótesis de trabajo es que buena parte involucran **razonamiento multi-paso** (aritmético, lógico, simbólico, espacial, temporal), y que el answer-only obliga al modelo a "saltar" a la respuesta sin espacio para computar los pasos intermedios, ocultando su capacidad real.

## Método y contribución

La contribución es doble: un **benchmark curado y difícil** (23 tareas limpias, con baselines humanos y métricas objetivas de opción múltiple o coincidencia exacta, liberado públicamente con datos, prompts y salidas de Codex) y una **demostración metodológica** de que la elección del protocolo de prompting cambia radicalmente la conclusión sobre lo que un modelo "puede" hacer. La selección aplica una cascada de filtros sobre las **209** tareas de BIG-Bench; el filtro clave descarta las 42 tareas donde el mejor modelo ya vencía al humano promedio, dejando **36**, de las que se descartan 13 "extremadamente difíciles" (como *Checkmate in One*, que exige rastrear un tablero de ajedrez a lo largo de una secuencia larga). Las 23 restantes forman BBH (27 subtareas, 6.511 ejemplos), agrupadas en dos familias: **algorítmicas** (11 tareas, resolubles con un algoritmo basado en reglas) y de **NLP / lenguaje natural** (12).

Se contrastan dos protocolos few-shot, ambos reforzados con descripción de la tarea y opciones de respuesta. En **answer-only (AO)** el modelo produce la respuesta sin trabajo intermedio. En **chain-of-thought (CoT)** los ejemplos se aumentan con una cadena de razonamiento; los autores escriben manualmente **tres exemplars CoT por tarea**, anteponiendo "Let's think step by step" (Kojima et al., 2022). Se evalúan Codex, InstructGPT y PaLM en varias escalas, con decodificación *greedy* y exactitud por *exact match*.

## Resultados

La tabla agregada compara contra el humano promedio (67.7%) y el mejor resultado previo de BIG-Bench (50.9%, que por construcción vencía al humano en 0 de 23 tareas):

| Modelo / protocolo | BBH (23) | # tareas > humano promedio |
|---|---|---|
| Mejor resultado previo BIG-Bench | 50.9 | 0 / 23 |
| PaLM 540B — answer-only | 52.3 | 6 / 23 |
| PaLM 540B — **CoT** | **65.2** (+12.9) | **10 / 23** |
| InstructGPT — answer-only | 51.8 | 4 / 23 |
| InstructGPT — **CoT** | **68.4** (+16.6) | **15 / 23** |
| Codex — answer-only | 56.6 | 5 / 23 |
| Codex — **CoT** | **73.9** (+16.7) | **17 / 23** |

Tres lecturas se desprenden. Primero, **answer-only ya subestimaba**: parte de la "dificultad" era un artefacto del formato del prompt. Segundo, **CoT aporta mejoras de dos dígitos en las tres familias**, llevando a Codex de 5 a 17 tareas ganadas. Tercero, **superar al promedio no es superar al experto**: Codex con CoT vence al humano promedio por más de 6%, pero sigue **más de 20% por debajo del mejor humano** —una advertencia central que conviene explicitar frente a la slide 29.

La ganancia **no es uniforme**: se concentra en las tareas algorítmicas. Al desagregar por familia (Codex), en NLP pasa de 66.4 a 73.5 (**+7.1**), pero en algorítmicas de 45.9 a 74.4 (**+28.5**). Las mayores ganancias por tarea individual son *Tracking Shuffled Objects* (+60.4), *Multi-Step Arithmetic* (+46.4) y *Navigate* (+46.0): todas requieren ejecutar un procedimiento paso a paso y mantener un estado intermedio. CoT también es una **estrategia emergente**: en modelos pequeños su ganancia es negativa o nula, y la brecha con answer-only crece con la escala. Más aún, en tareas cuya curva de escalamiento era plana (*Multi-Step Arithmetic*, *Tracking Shuffled Objects*, *Web of Lies*), CoT **desbloquea** desempeño emergente que el escalamiento por sí solo no lograba.

## Limitaciones

- **El baseline humano es frágil.** La evaluación humana tomó casi un año, con tareas que cambiaron de formato y evaluadores con pocas horas y acceso a Internet. Superar "al humano promedio" debe leerse con cautela.
- **Superar el promedio ≠ razonar.** El paper es enfático: vencer al humano promedio en tareas difíciles no equivale a comprensión real, y Codex sigue 20+ puntos bajo el mejor humano.
- **CoT no es panacea.** En tres tareas (*Causal Judgement*, *Ruin Names*, *Snarks*) queda **por debajo** de answer-only: CoT amplifica el razonamiento procedimental pero **no suple conocimiento ausente ni juicio pragmático** (p. ej., detectar sarcasmo sin contexto situacional).
- **Prompts CoT escritos a mano** y modelos opacos (tamaño y corpus de Codex/InstructGPT desconocidos), lo que limita conclusiones sobre *por qué* Codex domina en tareas algorítmicas.

## Por qué importa para la Clase 34

BBH es la contraparte empírica de la tesis central de la [Clase 34](/clases/clase-34): los LLMs **sí pueden razonar en múltiples pasos, pero necesitan que el prompt les dé espacio para hacerlo**. La relación con el [chain-of-thought de Wei et al. (2022)](/papers/chain-of-thought-wei-2022) es de complementariedad: uno **introduce** el método, el otro lo **estresa** aplicándolo al conjunto deliberadamente más difícil de un benchmark diverso, y demuestra que ahí es donde CoT rinde sus mayores frutos (+28.5 puntos en tareas algorítmicas). Comparten autores (Wei, Zhou, Chi, Le) y forman una unidad conceptual apoyada en el fundamento transversal del [chain-of-thought](/fundamentos/chain-of-thought). El mapa mental para el curso: **razonamiento en LLMs = capacidad latente + protocolo que la revela**. Answer-only oculta; CoT revela. Y la revelación tiene un perfil claro: brilla en lo procedimental (aritmética, lógica, rastreo de estado, orden), es tibia en lo semántico y falla en lo que exige conocimiento o juicio ausente —un matiz decisivo para cualquier aplicación de alto riesgo.
