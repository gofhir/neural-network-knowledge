---
title: "Compositional Attention Networks — MAC (2018)"
weight: 380
math: true
---

{{< paper-card
    title="Compositional Attention Networks for Machine Reasoning"
    authors="Drew A. Hudson, Christopher D. Manning (Stanford)"
    year="2018"
    venue="ICLR 2018"
    pdf="/papers/mac-hudson-2018.pdf" >}}
La **red MAC** (Memory, Attention and Composition) es una arquitectura neuronal **totalmente diferenciable** diseñada para un razonamiento **explícito y composicional**. Su tesis: se puede dotar a una red profunda de una **estructura que favorezca el razonamiento** sin renunciar a la diferenciabilidad end-to-end ni recurrir a supervisión de programa. MAC descompone cada problema en una secuencia de $p$ pasos de razonamiento basados en atención, ejecutados por una celda recurrente que mantiene una separación estricta entre **control** (qué operación hacer, derivada de la pregunta) y **memoria** (el resultado intermedio acumulado desde la imagen). Sobre el dataset **CLEVR** alcanza **98.9%**, más que reduciendo a la mitad el error del mejor modelo previo, y es **5× más eficiente en datos**. Es la encarnación paradigmática de "dar estructura a la red para razonar" que estudia la [Clase 34](/clases/clase-34), y hereda de la línea de [memoria externa](/fundamentos/redes-de-memoria) de las [NTM de Graves](/papers/ntm-graves-2014).
{{< /paper-card >}}

---

## Contexto: VQA, sesgos y CLEVR

El *visual question answering* exige responder preguntas en lenguaje natural sobre imágenes. La primera generación de modelos VQA adquiría solo una **comprensión superficial**: **explotaban sesgos del dataset** en vez de razonar (responder "¿de qué color es el plátano?" sin mirar la imagen, porque suelen ser amarillos). **CLEVR** (Johnson et al., 2017) se creó como dataset **diagnóstico**: imágenes renderizadas de objetos 3D con preguntas composicionales de múltiples pasos, **insesgadas**, que exigen relaciones transitivas y lógicas, conteo y comparaciones sin atajos. El ejemplo canónico ("¿el bloque frente al cilindro amarillo pequeño y la cosa a la derecha del objeto verde grande brillante tienen el mismo color?") obliga a resolver referencias indirectas encadenadas. Cada pregunta trae además un **programa funcional** de árbol —recurso que algunos modelos usan como supervisión y que MAC deliberadamente ignora—. El diagnóstico teórico: las redes profundas, "grandes motores de correlación", **fallan en tareas composicionales y estructuradas**.

## Contribución: una tercera vía

Existían dos respuestas previas, ambas con costos:

- **Module networks** (Andreas et al.; PG+EE): componen módulos neuronales según árboles de expresión, pero **dependen de programas funcionales provistos externamente**, parsers frágiles y entrenamiento multi-etapa con RL; su rigidez socava la generalización.
- **CNNs aumentadas** (Relation Networks, FiLM): más entrenables, pero **monolíticas** —fusionan pregunta e imagen y pierden transparencia y estructura explícita.

MAC busca el equilibrio: la versatilidad de los enfoques neuronales end-to-end y la estructura explícita del razonamiento simbólico. Realiza razonamiento estructurado **encadenando una celda recurrente MAC**, inspirada en la organización de los computadores, que separa **control** de **memoria**. Este diseño universal actúa como un **prior estructural** que induce a la red a descomponer el problema en operaciones de atención inferidas de los datos **sin supervisión fuerte**. Con auto-atención entre celdas, MAC representa de forma blanda **grafos de razonamiento acíclicos arbitrarios** (DAGs) manteniéndose diferenciable por retropropagación.

## Método: la celda MAC

Una red MAC encadena $p$ celdas, cada una un paso de razonamiento, sobre una **base de conocimiento** $K$ (la imagen, procesada con ResNet101 y dos CNNs hasta $14\times14$ regiones) y una **descripción de tarea** $q$ (la pregunta, procesada con un biLSTM que produce palabras contextuales $cw_s$, la representación global $q$ y proyecciones posicionales $q_i$ por paso). Cada celda mantiene dos estados duales de dimensión $d$: **control** $c_i$ (la operación del paso, un promedio ponderado por atención sobre las palabras) y **memoria** $m_i$ (el resultado intermedio). La clave —tomada de la arquitectura de computadores— es que la interacción entre las modalidades visual y textual está **mediada exclusivamente por distribuciones de probabilidad** (atención blanda, compuertas sigmoidales), manteniendo una **separación estricta entre espacios representacionales**, en contraste con los enfoques que fusionan todo en un vector.

- **Unidad de control (CU).** Combina $q_i$ con el control previo $c_{i-1}$ y proyecta el resultado sobre las palabras de la pregunta, produciendo una distribución de atención cuyo promedio da $c_i$. Este "anclaje" en las palabras regulariza el espacio de operaciones y mejora la transparencia:
  $$cv_{i,s} = \mathrm{softmax}\big(W^{1\times d}(cq_i \odot cw_s)\big), \qquad c_i = \sum_{s=1}^{S} cv_{i,s}\cdot cw_s$$
- **Unidad de lectura (RU).** Recupera $r_i$ de la imagen mediante atención de dos etapas: primero mide la interacción directa entre cada región $k_{h,w}$ y la memoria previa $m_{i-1}$ —habilitando **razonamiento transitivo**—, luego concatena $k_{h,w}$ para captar información nueva, y finalmente pondera por el control $c_i$.
- **Unidad de escritura (WU).** Integra $r_i$ con $m_{i-1}$ (paso obligatorio) y opcionalmente aplica **auto-atención** sobre memorias previas (usando control como clave y memoria como valor —un eco directo de las **Key-Value Memory Networks**) y una **compuerta de memoria** sigmoidal que permite "saltarse" pasos, ajustando dinámicamente la longitud efectiva del razonamiento.

La unidad de salida concatena $q$ y la memoria final $m_p$ y clasifica sobre las 28 respuestas candidatas de CLEVR.

## Resultados

Sobre CLEVR (700k ejemplos, $d=512$, $p=12$), **sin usar los programas funcionales**, MAC alcanza **98.94%**, superando a FiLM (97.7%) y al humano (92.6%), e incluso a PG+EE entrenado con 700k programas (96.9%). Destaca en **conteo** (97.1%) y **comparación numérica** (99.5%) —categorías difíciles para VQA—, porque la atención favorece la **agregación global** que el conteo requiere. En **CLEVR-Humans** (preguntas escritas por personas) logra **81.5%**, +5.6 puntos sobre el siguiente mejor, gracias a que la atención blanda ignora variaciones lingüísticas irrelevantes.

El segundo gran aporte es la **eficiencia**. MAC converge **40× más rápido** que Relation Networks y **10×** más rápido que FiLM (9.5 horas vs. 4 días). La **eficiencia de datos** es aún más dramática: con solo **10%** de CLEVR (70k), MAC es el **único que generaliza bien** (85.5%) mientras los demás fracasan (49.0%–54.9%, apenas sobre el baseline de 42.1%). Las ablaciones confirman qué importa: usar la pregunta completa en vez del control atendido cuesta −18.5%; **fundir control y memoria** en un solo estado cae de 98.9% a 93.75% (y a 20.2% con el 10% de datos); y la longitud de la red correlaciona con el rendimiento hasta $p=8$, señal de que MAC **usa efectivamente** las celdas para razonar.

## Limitaciones

- **Dominio sintético.** CLEVR usa objetos 3D renderizados y vocabulario controlado; la utilidad en tareas del mundo real queda como conjetura.
- **Errores residuales.** La mayoría son de conteo *off-by-one* y oclusiones fuertes; el modelo tiende a **subestimar** con oclusión pesada, señal de un conteo "continuo" más que discreto —no razona simbólicamente sino con aproximaciones blandas.
- **Número fijo de pasos.** La longitud $p$ es un hiperparámetro; la compuerta de memoria acorta el razonamiento efectivo, pero no hay un mecanismo completo de "detención adaptativa".

## Por qué importa para la Clase 34

En la sección "Intentando aumentar a DL" de la [Clase 34](/clases/clase-34), MAC ilustra la estrategia de **dar a la red una estructura que favorezca el razonamiento composicional**. El aporte conceptual: **la composicionalidad no exige supervisión simbólica**. Las module networks obtienen estructura a costa de programas y RL multi-etapa; la atención monolítica obtiene entrenabilidad a costa de perder estructura. MAC demuestra que se puede tener **estructura composicional explícita, diferenciable y sin supervisión de programa**, imponiendo un **prior arquitectónico** en lugar de un prior de datos: la estructura no se ejecuta como programa discreto, **emerge de forma blanda** de la secuencia de mapas de atención.

La conexión con la línea de [memoria externa](/fundamentos/redes-de-memoria) es explícita en el propio paper. MAC hereda de las [Neural Turing Machines](/papers/ntm-graves-2014) y el Differentiable Neural Computer de Graves la noción de un controlador que **lee y escribe** sobre memoria vía atención blanda. Pero, mientras las NTM/DNC operan sobre **múltiples slots de una memoria global compartida** (con riesgo de *content blurring*), MAC usa una **memoria recurrente** donde cada celda construye una nueva memoria sobre las anteriores. Su auto-atención de escritura es además un eco directo de las Key-Value Memory Networks. Para el razonamiento clínico multimodal, el valor de MAC no es su precisión sino su **arquitectura auditable**: la separación control–memoria y los mapas de atención por paso permitirían **exhibir la traza del razonamiento** —qué región de la imagen y qué términos pesaron en cada paso—, requisito prácticamente innegociable para la validación regulatoria, con la misma lógica trasladable al *record linkage* y al *master patient index*.
