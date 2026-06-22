---
title: "SimCLR: A Simple Framework for Contrastive Learning (2020)"
weight: 320
math: true
---

{{< paper-card
    title="A Simple Framework for Contrastive Learning of Visual Representations"
    authors="Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton"
    year="2020"
    venue="ICML 2020"
    pdf="/papers/simclr-chen-2020.pdf"
    arxiv="2002.05709" >}}
SimCLR (Google Brain) defiende una tesis casi provocadora para el [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) de imágenes: no hacen falta arquitecturas especializadas, *memory banks* ni mecanismos sofisticados para alcanzar el estado del arte. Basta combinar bien cuatro piezas conocidas: **augmentaciones compuestas** (crop + color), un **encoder** (ResNet), una **cabeza de proyección MLP** y la pérdida contrastiva **NT-Xent** con temperatura, usando los demás ejemplos del batch como negativos. Un clasificador lineal sobre sus representaciones logra **69.3% top-1 en ImageNet** con ResNet-50 estándar (76.5% al escalar a ResNet-50 4×, igualando al supervisado), y con solo el **1% de las etiquetas** alcanza 75.5% top-5. La aportación no es un componente nuevo sino el estudio sistemático de su *composición*. Es la pieza canónica de la familia [contrastiva](/fundamentos/aprendizaje-contrastivo) en la [Clase 28](/clases/clase-28).
{{< /paper-card >}}

---

## Contexto

Hacia 2020, aprender representaciones sin etiquetas caía en dos clases. Los **enfoques generativos** (VAEs, GANs) modelan o generan píxeles, algo caro y posiblemente innecesario para representar bien. Los **enfoques discriminativos** entrenan en *pretext tasks* donde entradas y etiquetas se derivan del conjunto no etiquetado: predicción de parches relativos, jigsaw, colorización, predicción de rotación. SimCLR es crítico con esta segunda línea: esas tareas se apoyan en "heurísticas un tanto *ad hoc* que limitan la generalidad de las representaciones".

La alternativa que ganaba tracción era el **aprendizaje contrastivo en el espacio latente**: hacer que dos vistas de una imagen coincidan bajo transformaciones. La idea se remonta a Becker & Hinton (1992) —no es casual que Hinton firme SimCLR— y a Hadsell, Chopra & LeCun (2006), que aprenden contrastando pares positivos contra negativos. Dosovitskiy et al. (2014) propusieron tratar **cada instancia como su propia clase** (*instance discrimination*); Wu et al. (2018) lo escalaron con un *memory bank*, enfoque adoptado por trabajos contemporáneos como [MoCo](/papers/moco-he-2019), CMC y PIRL. SimCLR toma una línea distinta —usar las muestras *del propio batch* como negativos— y se posiciona deliberadamente como la versión *despojada* de toda esa maquinaria.

## El marco de cuatro componentes

SimCLR aprende **maximizando el acuerdo entre dos vistas aumentadas del mismo ejemplo** mediante una pérdida contrastiva en el espacio latente. Cuatro piezas:

1. **Augmentación estocástica de datos.** Transforma cada ejemplo en dos vistas correlacionadas $\tilde{x}_i$ y $\tilde{x}_j$ (un par positivo). Aplica en secuencia *random crop* + redimensionado, distorsión de color, y desenfoque gaussiano. La composición *crop + color* es crítica.
2. **Encoder base $f(\cdot)$.** Extrae $h_i = f(\tilde{x}_i)$. Sin restricciones de arquitectura; se usa ResNet tomando la salida tras el *average pooling* (2048-d en ResNet-50).
3. **Cabeza de proyección $g(\cdot)$.** Un **MLP con una capa oculta** que mapea al espacio donde se aplica la pérdida: $z_i = g(h_i) = W^{(2)}\sigma(W^{(1)} h_i)$ con ReLU, a 128 dimensiones. Clave: la pérdida se define sobre los $z$, pero **para downstream se usa $h$ y se descarta $g(\cdot)$**.
4. **Pérdida contrastiva NT-Xent** (ver abajo).

El detalle de ingeniería que evita el *memory bank*: de un minibatch de $N$ ejemplos se generan $2N$ vistas y **no se muestrean negativos explícitamente** — dado un par positivo, las otras $2(N-1)$ vistas del batch son los negativos. Por eso el **tamaño del batch es tan determinante**: con $N = 8192$, cada par positivo enfrenta 16.382 negativos.

Lo que SimCLR *no* tiene es donde reside su tesis: no usa el *memory bank* de Wu et al., ni la cola con encoder de momentum de MoCo, ni el campo receptivo restringido de AMDIM, ni la partición fija de CPC. Se diferencia del entrenamiento supervisado estándar en ImageNet *únicamente* en tres puntos: la augmentación, la cabeza no-lineal y la función de pérdida.

## La pérdida NT-Xent

Con $\text{sim}(u, v) = u^\top v / (\lVert u\rVert \lVert v\rVert)$ la **similitud coseno** entre vectores normalizados con $\ell_2$, la pérdida para un par positivo $(i, j)$ es:

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i, z_k)/\tau)}$$

donde $\tau$ es el **parámetro de temperatura** y el indicador excluye el propio ejemplo. La pérdida total promedia sobre todos los pares positivos del minibatch en ambos sentidos. Se la bautiza **NT-Xent** (*Normalized Temperature-scaled Cross Entropy*): una entropía cruzada softmax sobre similitudes.

Su gradiente **pondera los negativos por su dificultad relativa**, lo que ayuda a aprender de *hard negatives* sin necesidad de minería explícita. La normalización $\ell_2$ y una temperatura adecuada importan mucho: sin normalización, la precisión de la tarea contrastiva sube pero la representación empeora; temperaturas demasiado altas degradan el top-1.

**Detalles que importan.** Con batches grandes, SGD se vuelve inestable, así que se usa el optimizador **LARS**. Se entrena en Cloud TPUs (32–128 cores); un ResNet-50 con batch 4096 por 100 épocas toma ~1.5 h en 128 TPU v3. **Global BN** corrige una fuga de información local: al estar los pares positivos en el mismo dispositivo, el modelo podría explotar las estadísticas de Batch Normalization por dispositivo, así que se agregan sobre todos.

## Los cuatro hallazgos clave

**1. La composición de augmentaciones es crucial (crop + color).** El paper reencuadra la augmentación como el mecanismo que **define la tarea predictiva**, no como mero truco de regularización. Vía *random cropping* se generan tareas que subsumen las de métodos más complejos (predicción global→local, vistas adyacentes), desacoplando la tarea de la arquitectura. En la ablación surge el resultado central: **ninguna transformación individual basta**; al componer, la tarea se vuelve más difícil pero la representación mejora drásticamente. La composición ganadora es **random crop + distorsión de color**. La conjetura: los parches de una imagen comparten distribución de color, y los histogramas solos bastan para distinguir imágenes; sin la distorsión de color, la red explota ese atajo.

**2. El contrastivo necesita augmentación más fuerte que el supervisado.** Una distorsión de color **más intensa** mejora la evaluación lineal del modelo no supervisado (de 59.6% a 64.5% con blur), y AutoAugment —hallado *con* supervisión— no la supera. Al revés, para el modelo supervisado la augmentación de color fuerte **no ayuda o perjudica** (de 77.0% a 75.4%). Augmentaciones inútiles para el supervisado pueden ayudar mucho al contrastivo.

**3. La cabeza de proyección no-lineal mejora la representación.** Comparando identidad, proyección lineal y la MLP no-lineal por defecto: la no-lineal es **+3% sobre la lineal** y **>10% sobre ninguna**. Hallazgo contraintuitivo: incluso con la cabeza no-lineal, la capa $h$ *anterior* a la proyección es >10% mejor que $z = g(h)$. La conjetura: la pérdida induce a $z$ a ser invariante a las transformaciones, por lo que $g$ *descarta* información útil para downstream (color, orientación). Se verifica midiendo cuánta info de la transformación se recupera desde $h$ frente a $g(h)$ (rotación: 67.6% vs 25.6%). Por eso se descarta la cabeza para tareas posteriores.

**4. Batch grande y más épocas ayudan mucho — más que al supervisado.** Con pocas épocas, los batches grandes tienen ventaja clara; con más pasos las diferencias se reducen. La razón es directa sin *memory bank*: a diferencia del supervisado, un batch mayor aporta **más negativos**, facilitando la convergencia; entrenar más tiempo también. Los números finales usan **1000 épocas**. Además, el contrastivo se beneficia de redes más profundas y anchas *más* que el supervisado: la brecha entre ambos se achica al crecer el modelo.

## Resultados

| Evaluación | Configuración | Resultado |
|---|---|---|
| **Lineal ImageNet** top-1/top-5 | ResNet-50 estándar | **69.3% / 89.0%** |
| Lineal ImageNet | ResNet-50 2× | 74.2% / 92.0% |
| Lineal ImageNet | ResNet-50 4× (375M) | **76.5% / 93.2%** (= supervisado) |
| **Semisupervisado 1%** etiquetas | ResNet-50 / 2× / 4× | 75.5% / 83.0% / 85.8% top-5 |
| **Semisupervisado 10%** etiquetas | ResNet-50 / 2× / 4× | 87.8% / 91.2% / 92.6% top-5 |

- **Evaluación lineal.** Con ResNet-50 estándar, SimCLR (69.3% top-1) supera a CPCv2 (63.8%), PIRL (63.6%), MoCo (60.6%) y Local Aggregation (60.2%) sobre la misma arquitectura. Escalando a ResNet-50 4×, iguala al ResNet-50 supervisado.
- **Pocas etiquetas.** Afinando con solo 1% o 10% de las etiquetas (~12.8 y ~128 imágenes por clase), supera tanto a métodos de *label-propagation* (UDA, FixMatch, S4L) como a los representacionales (BigBiGAN, PIRL, CPCv2). El baseline supervisado con 1% solo llega a 48.4% top-5.
- **Transferencia.** Sobre 12 datasets de imágenes naturales, afinado **supera significativamente al supervisado fuerte en 5** y este solo gana en 2 (Pets, Flowers).

## Limitaciones

- **Costo computacional del batch enorme.** SimCLR necesita batches de miles de ejemplos y cientos a miles de épocas, lo que exige **32–128 cores de TPU v3** en paralelo (batch 4096) — un régimen fuera del alcance de la mayoría de los laboratorios académicos. La dependencia de muchos negativos *en el mismo batch* es justo lo que [MoCo](/papers/moco-he-2019) sorteó con una cola de momentum, y lo que BYOL y SimSiam cuestionaron al aprender sin negativos explícitos.
- **Cabeza de proyección poco entendida.** La mejor representación está *antes* de la capa optimizada, lo que evidencia una pérdida de información aún no explicada del todo.
- **Pregunta conceptual abierta.** El paper no resuelve si el éxito se debe a la maximización de información mutua o a la forma específica de la pérdida contrastiva.

## Por qué importa para la Clase 28

SimCLR consolidó al [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) como el paradigma dominante del [autosupervisado](/fundamentos/aprendizaje-autosupervisado) visual de inicios de los 2020, mostrando que la simplicidad —augmentaciones compuestas + encoder + MLP de proyección + NT-Xent + batch grande— bastaba para igualar al supervisado. Inspiró una progenie (SimCLRv2, BYOL, SimSiam, SwAV) y, junto con MoCo, fijó la evaluación lineal como protocolo de facto.

En la [Clase 28](/clases/clase-28) (Aprendizaje Autosupervisado) es el representante canónico de la familia contrastiva en imágenes: aparece en el slide dedicado a Chen et al. y en la tabla comparativa de métodos. Su énfasis en que la *augmentación define la tarea* y en el papel del *projection head* es el puente natural hacia el contrastivo **multimodal** texto-imagen de [CLIP](/papers/clip-radford-2021), que generaliza la idea de "maximizar el acuerdo entre vistas" a pares imagen-texto.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/2002.05709
- Código: https://github.com/google-research/simclr
- Venue: ICML 2020 (PMLR vol. 119). Google Research, Brain Team.
