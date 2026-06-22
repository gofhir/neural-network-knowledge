---
title: "MoCo v2: Improved Baselines (2020)"
weight: 322
math: true
---

{{< paper-card
    title="Improved Baselines with Momentum Contrastive Learning"
    authors="Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He"
    year="2020"
    venue="arXiv (technical report)"
    pdf="/papers/moco-v2-chen-2020.pdf"
    arxiv="2003.04297" >}}
Reporte técnico breve (3 páginas) de Facebook AI Research que aparece apenas un mes después de SimCLR. Su tesis es deliberadamente modesta: dos de las mejoras de diseño que SimCLR introdujo —una **cabeza de proyección MLP** y **aumentación de datos más fuerte** (sobre todo *blur* gaussiano)— no son propiedad de SimCLR, sino trucos **ortogonales** que se pueden trasplantar a cualquier framework contrastivo. Los autores los montan sobre [MoCo v1](/papers/moco-he-2019) manteniendo intacta su **cola de negativos** y su **encoder de momentum**. El resultado, **MoCo v2**, supera a [SimCLR](/papers/simclr-chen-2020) en clasificación lineal sobre ImageNet (**71.1% vs 69.3%**) **sin batches gigantes ni TPUs**: batch 256 en una máquina de 8 GPUs. Es el cierre exacto de la comparación SimCLR vs MoCo que ve la [Clase 28](/clases/clase-28).
{{< /paper-card >}}

---

## Contexto

A fines de 2019 y comienzos de 2020 el [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) se consolidó como el concepto central del aprendizaje de representaciones no supervisado de imágenes. MoCo v1 había mostrado que el preentrenamiento no supervisado **puede superar a su contraparte supervisada** en varias tareas de detección y segmentación; SimCLR, un mes antes de este reporte, redujo aún más la brecha en clasificación lineal entre lo no supervisado y lo supervisado. En cuestión de meses el SSL pasó de curiosidad académica a competir de igual a igual.

Todos estos métodos comparten la misma pérdida contrastiva, **InfoNCE**:

$$\mathcal{L}_{q,k^+,\{k^-\}} = -\log \frac{\exp(q\cdot k^+/\tau)}{\exp(q\cdot k^+/\tau) + \sum_{k^-}\exp(q\cdot k^-/\tau)}$$

donde $q$ es la *query*, $k^+$ el positivo (otra vista aumentada de la misma imagen), $\{k^-\}$ los negativos y $\tau$ la temperatura. La tarea pretexto es **discriminación de instancia**: dos vistas aumentadas de la misma imagen forman un par positivo, cualquier otra imagen es negativo.

La diferencia clave entre los dos frameworks es **cómo se mantienen los negativos**:

- **End-to-end (SimCLR):** los negativos vienen del mismo batch y se actualizan por retropropagación de extremo a extremo. Esto **acopla el tamaño de batch al número de negativos** — para tener muchos negativos hay que usar batches enormes (4k–8k), lo que exige TPUs.
- **MoCo (cola + momentum):** los negativos se mantienen en una **cola** (un diccionario de claves de batches anteriores) y solo las queries y claves positivas se codifican en cada batch. Un **encoder de momentum** (media móvil exponencial del encoder de la query) mantiene consistentes las representaciones de la cola. La consecuencia decisiva: **MoCo desacopla el tamaño de batch del número de negativos**.

SimCLR introdujo tres mejoras sobre la variante end-to-end: (i) un batch mucho mayor para proveer más negativos; (ii) reemplazar la cabeza `fc` por una **cabeza MLP**; y (iii) **aumentación más fuerte**. El insight de MoCo v2 es que (ii) y (iii) son ortogonales al framework: MoCo ya resuelve el problema de los negativos por otra vía (la cola), así que solo necesita importar los otros dos trucos.

## Contribución central

La contribución es de una simplicidad calculada: **MoCo v2 = MoCo v1 + cabeza MLP + aumentación fuerte (+ schedule coseno)**. No hay arquitectura nueva ni pérdida nueva; los cambios *"require only a few lines of code changes to MoCo v1"*. El valor está en demostrar empíricamente que la combinación es **lo mejor de ambos mundos**:

1. **Eficiencia de MoCo:** la cola provee abundantes negativos sin batches grandes, manteniendo el costo en GPUs commodity.
2. **Calidad de SimCLR:** la cabeza MLP y el *blur* recuperan (y superan) la ganancia que SimCLR atribuía a su diseño.

El resultado neto: MoCo v2 **supera a SimCLR sin batches gigantes ni TPUs**. Esto invierte la narrativa implícita de SimCLR (que sugería que los batches grandes eran parte esencial de la receta) por una conclusión de ingeniería: *"large batches are not necessary for good accuracy, and state-of-the-art results can be made more accessible"*.

## Las tres modificaciones

Todos los experimentos usan **ResNet-50** como encoder y se evalúan con dos protocolos: clasificación lineal en ImageNet (features congeladas, top-1 a 1-crop) y transferencia a detección VOC (Faster R-CNN C4).

**Cabeza de proyección MLP.** Se reemplaza la cabeza `fc` de MoCo por un **MLP de 2 capas** (oculta de 2048-d con ReLU). Detalle clave: la MLP **solo se usa en el preentrenamiento**; en clasificación lineal o transferencia se descarta. Con el $\tau=0.07$ por defecto de MoCo v1, la MLP mejora de 60.6% a 62.9%; al re-sintonizar a $\tau=0.2$ (óptimo con MLP), la accuracy salta a **66.2%**.

**Aumentación más fuerte (blur).** Se agrega el **blur gaussiano** de SimCLR. La distorsión de color fuerte de SimCLR tiene ganancias decrecientes sobre los baselines ya mejorados de MoCo, así que el ingrediente nuevo útil es el blur: por sí solo (sin MLP) sube la baseline a **63.4%**. Hallazgo metodológico de peso: pese a tener menor accuracy lineal que solo la MLP (63.4% vs 66.2%), el blur logra **mejor detección VOC** — *"linear classification accuracy is not monotonically related to transfer performance in detection"*.

**Schedule coseno.** Para comparar de forma justa con SimCLR (que lo adopta), se añade un schedule de learning rate **coseno** (half-period). Con MLP + blur + coseno a 200 épocas, MoCo v2 llega a **67.5%**.

| Caso | MLP | aug+ | cos | épocas | ImageNet | VOC AP |
|---|---|---|---|---|---|---|
| supervisado | | | | | 76.5 | 53.5 |
| MoCo v1 | | | | 200 | 60.6 | 55.9 |
| (a) | ✓ | | | 200 | 66.2 | 56.4 |
| (b) | | ✓ | | 200 | 63.4 | 56.8 |
| (c) | ✓ | ✓ | | 200 | 67.3 | 57.2 |
| (d) | ✓ | ✓ | ✓ | 200 | 67.5 | 57.0 |
| (e) | ✓ | ✓ | ✓ | 800 | **71.1** | 57.4 |

La fila (e) es la *headline*: con 800 épocas de preentrenamiento, MoCo v2 alcanza **71.1%** en ImageNet. En detección VOC, MoCo v1 ya superaba al supervisado (55.9 vs 53.5 AP) y MoCo v2 lo extiende a 57.4.

## MoCo v2 vs SimCLR

| Método | épocas | batch | ImageNet acc. |
|---|---|---|---|
| MoCo v1 | 200 | 256 | 60.6 |
| SimCLR | 200 | 256 | 61.9 |
| SimCLR | 200 | 8192 | 66.6 |
| **MoCo v2** | 200 | 256 | **67.5** |
| SimCLR | 1000 | 4096 | 69.3 |
| **MoCo v2** | 800 | 256 | **71.1** |

Las dos comparaciones que importan:

- **Mismas épocas y batch (200 ép., batch 256):** MoCo v2 logra 67.5%, **5.6% por encima** de SimCLR (61.9%), y además **mejor que el resultado de batch grande de SimCLR** (66.6% con batch 8192). MoCo v2 con batch chico vence a SimCLR con batch enorme.
- **Mejor de cada uno:** MoCo v2 con 800 épocas y batch 256 alcanza **71.1%**, superando a SimCLR con 1000 épocas y batch 4096 (**69.3%**).

## El argumento del costo computacional

Aquí está el corazón del mensaje de accesibilidad. La medición se hace en **8 GPUs V100 de 16 GB** en PyTorch:

| Mecanismo | batch | memoria / GPU | tiempo / 200 ép. |
|---|---|---|---|
| MoCo | 256 | 5.0 G | 53 hrs |
| end-to-end | 256 | 7.4 G | 65 hrs |
| end-to-end | 4096 | 93.0 G † | n/a |

Dos conclusiones:

1. **El batch de 4k es intratable** incluso en una máquina de 8 GPUs de gama alta: requeriría ~93 GB por GPU (imposible en V100 de 16 GB). Esto explica por qué SimCLR necesita TPUs.
2. **Aun con el mismo batch 256, end-to-end es más caro** que MoCo en memoria (7.4 G vs 5.0 G) y tiempo (65 hrs vs 53 hrs), porque retropropaga a **ambos** encoders ($q$ y $k$), mientras MoCo solo retropropaga al encoder de la query; el encoder de claves se actualiza por momentum, sin gradiente.

## Limitaciones

- **No es un método nuevo.** Por diseño, MoCo v2 combina ideas existentes; su contribución es empírica y de ingeniería, no conceptual.
- **La accuracy lineal no es monótona con la transferencia.** El propio paper documenta que el blur mejora detección pero baja la clasificación lineal relativa a la MLP — una advertencia sobre el protocolo de evaluación dominante.
- **Ganancias decrecientes de la aumentación.** No toda la receta de aumentación de SimCLR transfiere su utilidad; la distorsión de color fuerte aporta poco.
- **Brecha residual con el supervisado.** En clasificación lineal, 71.1% sigue por debajo del 76.5% supervisado (aunque en detección VOC ya lo supera).

## Por qué importa para la Clase 28

La [Clase 28](/clases/clase-28) dedica una parte a comparar **SimCLR vs MoCo** como los dos métodos contrastivos canónicos, y MoCo v2 es la pieza que cierra esa comparación:

- **La tabla de la clase** reporta MoCo v2 con **71.1%** top-1 frente a SimCLR con **69.3%** — exactamente los números de la Tabla 2 de este reporte.
- **El slide "Detalles Importantes"** señala que SimCLR necesita batch 4096 (y por ende muchas TPU) mientras MoCo tiene menores requisitos. Este paper lo *demuestra*: el batch 4k pide ~93 GB/GPU mientras MoCo v2 corre en 5.0 GB/GPU.
- **El mecanismo que la clase explica** —encoder de momentum + cola que desacopla el batch del número de negativos— es justamente lo que permite a MoCo v2 usar batch chico sin sacrificar negativos.

En síntesis: si SimCLR es la versión "elegante pero cara" del [contraste](/fundamentos/aprendizaje-contrastivo) end-to-end, **MoCo v2 es la versión "lo mejor de ambos mundos"** — toma los dos trucos baratos de SimCLR (MLP head + blur) y los monta sobre la cola eficiente de MoCo, ganando en accuracy *y* en costo.

## Notas y enlaces

- Preprint: arXiv:2003.04297v1 (9 mar 2020), [arxiv.org/abs/2003.04297](https://arxiv.org/abs/2003.04297).
- Antecesor MoCo v1: [/papers/moco-he-2019](/papers/moco-he-2019).
- Paper hermano SimCLR: [/papers/simclr-chen-2020](/papers/simclr-chen-2020).
- Fundamento transversal: [/fundamentos/aprendizaje-contrastivo](/fundamentos/aprendizaje-contrastivo).
- La línea continuó con MoCo v3 (adaptación a Vision Transformers); la cabeza de proyección + aumentación fuerte se volvió estándar incluso en métodos sin negativos (BYOL, SimSiam).
