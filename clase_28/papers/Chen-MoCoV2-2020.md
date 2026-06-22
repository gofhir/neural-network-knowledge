# Improved Baselines with Momentum Contrastive Learning (MoCo v2) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Improved Baselines with Momentum Contrastive Learning*.
- **Autores:** Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He — Facebook AI Research (FAIR).
- **Tipo:** *Technical note* / reporte técnico breve (3 páginas de contenido). No es un paper de algoritmo nuevo: es una actualización de baselines.
- **Año:** 2020. **Preprint:** arXiv:2003.04297v1 (9 mar 2020), [arxiv.org/abs/2003.04297](https://arxiv.org/abs/2003.04297).
- **Antecesores directos:** MoCo v1 (He, Fan, Wu, Xie, Girshick, arXiv:1911.05722, 2019) y SimCLR (Chen, Kornblith, Norouzi, Hinton, arXiv:2002.05709, 2020).

Este documento se publica apenas un mes después de SimCLR, y su tesis es deliberadamente modesta: **dos de las mejoras de diseño que SimCLR introdujo no son propiedad de SimCLR, sino trucos ortogonales que se pueden trasplantar a cualquier framework contrastivo**. Concretamente, los autores toman MoCo v1 y le agregan (i) una *cabeza de proyección MLP* en lugar de la cabeza `fc` original y (ii) *aumentación de datos más fuerte* (sobre todo *blur* gaussiano), manteniendo intacta la **cola de negativos (queue)** y el **encoder de momentum** que definen a MoCo. El resultado, bautizado **MoCo v2**, supera a SimCLR en clasificación lineal sobre ImageNet **sin necesitar batches gigantes ni TPUs**.

La frase que resume el aporte está en el abstract: *"with simple modifications to MoCo —namely, using an MLP projection head and more data augmentation— we establish stronger baselines that outperform SimCLR and do not require large training batches"*. El objetivo declarado es de accesibilidad: *"We hope this will make state-of-the-art unsupervised learning research more accessible"*. Mientras SimCLR exige batches de 4k–8k que solo son tratables en TPUs, los baselines de MoCo v2 corren en *"a typical 8-GPU machine"* y obtienen mejores resultados.

Para la Clase 28 (Aprendizaje Autosupervisado) este paper es exactamente la lección que la clase saca de la comparación SimCLR vs MoCo: la tabla de la clase muestra MoCo v2 con **71.1%** top-1, por encima de SimCLR con **69.3%**, y el slide *"Detalles Importantes"* enfatiza que SimCLR necesita batch 4096 (muchas TPU) mientras MoCo tiene menores requisitos computacionales. MoCo v2 *es* la materialización de ese punto: lo mejor de ambos mundos —la eficiencia de la cola de MoCo más los trucos de SimCLR— en una máquina commodity.

## 2. Contexto histórico: MoCo v1, SimCLR y la carrera contrastiva de 2019–2020

A fines de 2019 y comienzos de 2020 el aprendizaje contrastivo se consolidó como el concepto central del aprendizaje de representaciones no supervisado de imágenes. El paper resume el momentum del campo: MoCo v1 mostró que el preentrenamiento no supervisado **puede superar a su contraparte supervisada en ImageNet** en varias tareas de detección y segmentación; y SimCLR *"further reduces the gap in linear classifier performance between unsupervised and supervised pre-training"*. Es decir, en cuestión de meses el preentrenamiento auto-supervisado pasó de curiosidad académica a competir de igual a igual con el supervisado.

Todos estos métodos comparten la misma función de pérdida contrastiva, **InfoNCE** (van den Oord et al., 2018), que el paper escribe explícitamente:

$$\mathcal{L}_{q,k^+,\{k^-\}} = -\log \frac{\exp(q\cdot k^+/\tau)}{\exp(q\cdot k^+/\tau) + \sum_{k^-}\exp(q\cdot k^-/\tau)}$$

donde $q$ es la representación de la *query*, $k^+$ la del *positivo* (otra vista aumentada de la misma imagen), $\{k^-\}$ las de los *negativos*, y $\tau$ es la temperatura. La tarea pretexto es **discriminación de instancia** (Wu et al., 2018): dos vistas aumentadas de la misma imagen forman un par positivo; cualquier otra imagen es negativo.

La diferencia clave entre los dos frameworks es **cómo se mantienen los negativos** (Fig. 1 del paper):

- **End-to-end (SimCLR):** los negativos provienen del mismo batch y se actualizan por retropropagación de extremo a extremo. Esto **acopla el tamaño de batch al número de negativos** — para tener muchos negativos hay que usar batches enormes (4k–8k), lo que requiere TPUs.
- **MoCo (queue + momentum):** los negativos se mantienen en una **cola** (un diccionario de claves de batches anteriores), y solo las *queries* y *claves positivas* se codifican en cada batch. Un **encoder de momentum** (actualizado como media móvil exponencial del encoder de la query) mantiene la consistencia de las representaciones entre la cola actual y las claves más antiguas. La consecuencia decisiva: **MoCo desacopla el tamaño de batch del número de negativos**.

SimCLR introdujo tres mejoras sobre la variante end-to-end de discriminación de instancia: (i) un batch sustancialmente mayor (4k u 8k) para proveer más negativos; (ii) reemplazar la cabeza `fc` de proyección por una **cabeza MLP**; y (iii) **aumentación de datos más fuerte**. El insight de MoCo v2 es que (ii) y (iii) son ortogonales al framework: en MoCo *"a large number of negative samples are readily available; the MLP head and data augmentation are orthogonal to how contrastive learning is instantiated"*. Es decir, MoCo ya resuelve el problema de los negativos por otra vía (la cola), así que solo necesita importar los otros dos trucos.

## 3. Contribución central

La contribución es de una simplicidad calculada: **MoCo v2 = MoCo v1 + cabeza MLP + aumentación fuerte (+ schedule coseno)**. No hay arquitectura nueva, no hay pérdida nueva. Los autores destacan que las mejoras *"require only a few lines of code changes to MoCo v1"*. El valor está en la demostración empírica de que esta combinación —la cola de MoCo con los trucos de SimCLR— es **lo mejor de ambos mundos**:

1. **Eficiencia de MoCo:** la cola provee abundantes negativos sin necesitar batches grandes, manteniendo el costo en GPUs commodity.
2. **Calidad de SimCLR:** la cabeza MLP y el blur recuperan (y superan) la ganancia de calidad que SimCLR atribuía a su diseño.

El resultado neto es que MoCo v2 **supera a SimCLR sin necesitar batches gigantes ni TPUs**: con batch 256 en 8 GPUs alcanza mejores números que SimCLR con batch 4096 en TPUs. Esto invierte la narrativa implícita de SimCLR (que sugería que batches grandes eran parte esencial de la receta) y la reemplaza por una conclusión de ingeniería: *"large batches are not necessary for good accuracy, and state-of-the-art results can be made more accessible"*.

## 4. Método: las ablaciones de MoCo v2

Todos los experimentos usan un **ResNet-50** estándar como encoder, y se evalúan con dos protocolos: (i) **clasificación lineal en ImageNet** (features congeladas, se entrena solo un clasificador lineal supervisado; se reporta top-1 a 1-crop 224×224) y (ii) **transferencia a detección de objetos VOC** (Faster R-CNN con backbone C4, fine-tuning end-to-end en VOC 07+12 trainval, evaluado en VOC 07 test con métricas COCO). Los hiperparámetros y el codebase son los de MoCo v1, salvo lo indicado.

### 4.1. Cabeza de proyección MLP

Siguiendo a SimCLR, se reemplaza la cabeza `fc` de MoCo por un **MLP de 2 capas** (capa oculta de 2048-d con ReLU). Detalle importante que el paper subraya: la MLP **solo afecta la etapa de preentrenamiento no supervisado**; en la etapa de clasificación lineal o de transferencia *no se usa* (se descarta y se entrena sobre el backbone). Esto es coherente con la intuición de SimCLR de que la cabeza de proyección "protege" a la representación del backbone del colapso inducido por el objetivo contrastivo.

Al introducir el MLP los autores buscan el $\tau$ óptimo respecto a la accuracy lineal en ImageNet:

| $\tau$ | 0.07 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| sin MLP | 60.6 | 60.7 | 59.0 | 58.2 | 57.2 | 56.4 |
| con MLP | 62.9 | 64.9 | **66.2** | 65.7 | 65.0 | 64.3 |

Con el $\tau=0.07$ por defecto de MoCo v1, agregar la MLP mejora de **60.6% a 62.9%**; pero al re-sintonizar a $\tau=0.2$ (óptimo con MLP) la accuracy salta a **66.2%**. Detalle revelador: la ganancia en detección VOC al agregar la MLP es mucho menor que el salto en ImageNet — primera señal de que la accuracy lineal no predice bien la transferencia.

### 4.2. Aumentación más fuerte (blur)

Se extiende la aumentación original de MoCo agregando el **blur gaussiano** de SimCLR. Los autores notan que la distorsión de color más fuerte de SimCLR tiene *ganancias decrecientes* en sus baselines (más altos), así que el principal ingrediente nuevo útil es el blur. Hallazgos:

- El blur **por sí solo** (sin MLP) mejora la baseline de MoCo en ImageNet en 2.8%, hasta **63.4%** (Tabla 1b).
- Curiosamente, su accuracy de detección es **más alta** que la de usar solo la MLP (Tabla 1b vs 1a), pese a tener accuracy lineal mucho menor (63.4% vs 66.2%). El paper concluye: *"linear classification accuracy is not monotonically related to transfer performance in detection"* — una advertencia metodológica de peso.
- Con la MLP, agregar el blur sube ImageNet a **67.3%** (Tabla 1c).

### 4.3. Schedule de learning rate coseno

Para comparar de forma justa con SimCLR (que lo adopta), se añade un **schedule de learning rate coseno** (half-period, Loshchilov & Hutter, 2017). Tabla 1d: con MLP + blur + coseno a 200 épocas, MoCo v2 llega a **67.5%**. Es una mejora marginal sobre 67.3% pero alinea el setup con SimCLR para que la comparación sea limpia.

### 4.4. Tabla de ablaciones (Tabla 1, resumen)

| Caso | MLP | aug+ | cos | épocas | ImageNet acc. | VOC AP |
|---|---|---|---|---|---|---|
| supervisado | | | | | 76.5 | 53.5 |
| MoCo v1 | | | | 200 | 60.6 | 55.9 |
| (a) | ✓ | | | 200 | 66.2 | 56.4 |
| (b) | | ✓ | | 200 | 63.4 | 56.8 |
| (c) | ✓ | ✓ | | 200 | 67.3 | 57.2 |
| (d) | ✓ | ✓ | ✓ | 200 | 67.5 | 57.0 |
| (e) | ✓ | ✓ | ✓ | 800 | **71.1** | 57.4 |

La fila (e) es la headline: con 800 épocas de preentrenamiento, MoCo v2 alcanza **71.1%** en ImageNet. Nótese que en detección VOC, MoCo v1 ya superaba al supervisado (55.9 vs 53.5 AP), y MoCo v2 lo extiende a 57.4.

## 5. Experimentos: MoCo v2 vs SimCLR y el argumento del costo

### 5.1. Comparación de accuracy (Tabla 2)

| Método | MLP | aug+ | cos | épocas | batch | ImageNet acc. |
|---|---|---|---|---|---|---|
| MoCo v1 | | | | 200 | 256 | 60.6 |
| SimCLR | ✓ | ✓ | ✓ | 200 | 256 | 61.9 |
| SimCLR | ✓ | ✓ | ✓ | 200 | 8192 | 66.6 |
| **MoCo v2** | ✓ | ✓ | ✓ | 200 | 256 | **67.5** |
| SimCLR | ✓ | ✓ | ✓ | 1000 | 4096 | 69.3 |
| **MoCo v2** | ✓ | ✓ | ✓ | 800 | 256 | **71.1** |

Las comparaciones clave:

- **Mismas épocas y batch (200 ép., batch 256):** MoCo v2 logra 67.5%, **5.6% por encima** de SimCLR (61.9%), y además **mejor que el resultado de batch grande de SimCLR** (66.6% con batch 8192). Es decir, MoCo v2 con batch chico vence a SimCLR con batch enorme.
- **Mejores resultados de cada uno:** MoCo v2 con 800 épocas y batch 256 alcanza **71.1%**, superando a SimCLR con 1000 épocas y batch 4096 (**69.3%**). *"With 800-epoch pre-training, MoCo v2 achieves 71.1%, outperforming SimCLR's 69.3% with 1000 epochs."*

(Los autores agradecen a los autores de SimCLR por proveer los resultados numéricos de su Fig. 9, lo que permite la comparación directa.)

### 5.2. El argumento del costo computacional (Tabla 3)

Aquí está el corazón del mensaje de accesibilidad. La medición se hace en **8 GPUs V100 de 16 GB**, en PyTorch:

| Mecanismo | batch | memoria / GPU | tiempo / 200 ép. |
|---|---|---|---|
| MoCo | 256 | 5.0 G | 53 hrs |
| end-to-end | 256 | 7.4 G | 65 hrs |
| end-to-end | 4096 | 93.0 G † | n/a |

El caso *end-to-end* refleja el costo de SimCLR pero en GPUs (no en las TPUs originales). Dos conclusiones:

1. **El batch de 4k es intratable** incluso en una máquina de 8 GPUs de gama alta: requeriría 93 GB por GPU (estimación marcada con †), imposible en V100 de 16 GB. Esto explica por qué SimCLR necesita TPUs.
2. **Aun con el mismo batch de 256, end-to-end es más caro** que MoCo en memoria (7.4 G vs 5.0 G) y tiempo (65 hrs vs 53 hrs), *porque retropropaga a ambos encoders* ($q$ y $k$), mientras **MoCo solo retropropaga al encoder de la query** ($q$). El encoder de claves se actualiza por momentum, sin gradiente.

Las Tablas 2 y 3 juntas sostienen la tesis del paper: *"large batches are not necessary for good accuracy, and state-of-the-art results can be made more accessible"*. El sacrificio de ingeniería de SimCLR (TPUs, batches gigantes) no compra ventaja; la cola de MoCo lo logra mejor y más barato.

## 6. Limitaciones

El paper es un reporte técnico breve y honesto sobre su alcance; las limitaciones son en parte explícitas y en parte de lectura entre líneas:

- **No es un método nuevo.** Por diseño, MoCo v2 es una combinación de ideas existentes (MoCo + dos trucos de SimCLR). Su contribución es empírica y de ingeniería, no conceptual. Esto es virtud (accesibilidad) y límite (no avanza la teoría del contraste).
- **La accuracy lineal no es monótona con la transferencia.** El propio paper documenta que el blur mejora detección pero baja clasificación lineal relativa a la MLP (Tabla 1b vs 1a). Esto significa que optimizar para el benchmark lineal de ImageNet puede ser engañoso para tareas downstream — una limitación del protocolo de evaluación dominante, más que del método.
- **Ganancias decrecientes de la aumentación.** La distorsión de color fuerte de SimCLR aporta poco sobre los baselines ya mejorados de MoCo; no toda la receta de aumentación transfiere su utilidad.
- **Brecha residual con el supervisado.** En clasificación lineal de ImageNet, 71.1% sigue por debajo del 76.5% supervisado, aunque en detección VOC ya lo supera. El gap no se cierra del todo en el régimen lineal.
- **Sin nuevos dominios.** Toda la evaluación es ImageNet (clasificación) y VOC (detección); no hay exploración de otros dominios o modalidades — coherente con que es una nota de baselines.

## 7. Impacto

MoCo v2 tuvo un impacto desproporcionado a su brevedad (3 páginas). Estableció el principio de que **el mecanismo de mantenimiento de negativos (cola vs batch) es ortogonal a los trucos de diseño (cabeza, aumentación)**, lo que reordenó cómo la comunidad pensaba el aprendizaje contrastivo: en vez de "frameworks rivales" (MoCo vs SimCLR), se entendieron como un menú de componentes intercambiables. Democratizó el SSL de imágenes al demostrar que **resultados SOTA caben en 8 GPUs**, no solo en pods de TPU, lo que amplió drásticamente quién podía investigar en el área.

La línea MoCo continuó evolucionando (la familia MoCo v3 después adaptaría el enfoque a Vision Transformers), y la idea de cabeza de proyección + aumentación fuerte se volvió estándar en métodos posteriores, incluso los que abandonaron los negativos (BYOL, SimSiam — varios del mismo grupo de FAIR). MoCo v2 quedó como la baseline contrastiva *eficiente* de referencia para la era 2020.

## 8. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

La Clase 28 dedica una parte a comparar **SimCLR vs MoCo** como los dos métodos contrastivos canónicos, y MoCo v2 es la pieza que cierra esa comparación:

- **La tabla de la clase** reporta MoCo v2 con **71.1%** top-1 en ImageNet (clasificación lineal, ResNet-50), por encima de SimCLR con **69.3%**. Esos son exactamente los números de la Tabla 2 de este paper (fila MoCo v2 800 ép. vs SimCLR 1000 ép.). Cuando la clase muestra "MoCo v2: 71.1 > SimCLR: 69.3", está citando directamente el resultado headline de este reporte.

- **El slide "Detalles Importantes"** señala que SimCLR necesita batch 4096 (y por ende muchas TPU) mientras que MoCo tiene menores requisitos computacionales. Este paper *demuestra empíricamente* esa afirmación con la Tabla 3: el batch 4k de SimCLR pide ~93 GB/GPU (intratable en V100), mientras MoCo v2 con batch 256 corre en 5.0 GB/GPU. La lección de la clase —que MoCo es más accesible computacionalmente— **es literalmente la tesis de MoCo v2**.

- **El mecanismo que la clase explica** —encoder de momentum + cola de negativos que desacopla el batch del número de negativos— es lo que permite que MoCo v2 use batch chico sin sacrificar negativos. Entender por qué MoCo no necesita batch grande (porque la cola guarda negativos de batches pasados, mantenidos consistentes por el encoder de momentum que solo retropropaga al encoder de query) es entender por qué la Tabla 3 le da la razón.

En síntesis para el curso: si SimCLR es la versión "elegante pero cara" del contraste end-to-end, **MoCo v2 es la versión "lo mejor de ambos mundos"** — toma los dos trucos baratos de SimCLR (MLP head + blur) y los monta sobre la cola eficiente de MoCo, ganando en accuracy *y* en costo. Es el ejemplo perfecto de la moraleja de la clase: en SSL contrastivo, la ingeniería del mecanismo de negativos importa tanto como la calidad de las representaciones.

## 9. Enlaces internos

- Paper antecesor MoCo v1: [/papers/he-moco-2019](/papers/he-moco-2019)
- Paper hermano SimCLR: [/papers/simclr-chen-2020](/papers/simclr-chen-2020)
- Fundamento transversal: [/fundamentos/aprendizaje-contrastivo](/fundamentos/aprendizaje-contrastivo)
- Clase: [/clases/clase-28](/clases/clase-28)
