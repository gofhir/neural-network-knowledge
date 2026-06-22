---
title: "Shuffle and Learn: Temporal Order Verification (2016)"
weight: 314
math: true
---

{{< paper-card
    title="Shuffle and Learn: Unsupervised Learning using Temporal Order Verification"
    authors="Ishan Misra, C. Lawrence Zitnick, Martial Hebert"
    year="2016"
    venue="ECCV 2016"
    pdf="/papers/shuffle-and-learn-misra-2016.pdf"
    arxiv="1603.08561" >}}
Piedra fundacional del **aprendizaje autosupervisado a partir de video** (Misra et al., CMU + Facebook AI Research). Su tesis es simple y poderosa: el orden temporal de un video es una señal de supervisión *gratuita*, sin etiquetas humanas. En vez de predecir categorías, la red verifica si una tupla de fotogramas está en el orden temporal correcto o desordenada ("shuffled"). Esa representación, aprendida sin semántica, resulta **complementaria a ImageNet**, captura **pose humana**, y al transferirla mejora +12.4% en UCF101 y +4.7% en HMDB51 sobre entrenar desde cero, llegando a superar al pre-entrenamiento supervisado en pose.
{{< /paper-card >}}

---

## Contexto: la flecha del tiempo como supervisión gratis

En 2016 el aprendizaje de representaciones sin etiquetas estaba fragmentado. Desde imágenes estáticas existían autoencoders, Deep Boltzmann Machines y el cercano trabajo de Doersch et al., que usa el **contexto espacial** (predecir la posición relativa de parches) como pretexto. Desde video, la tradición dominante imponía **continuidad espaciotemporal**: como la apariencia cambia suavemente, se regularizaba con *temporal coherence* o *slow feature analysis*. Otra rama —con LSTMs— predecía fotogramas explícitamente.

*Shuffle and Learn* rompe con ambas. **No impone suavidad** sobre las características (el verbo es "verificar el orden", no "suavizar") y **no predice fotogramas** (esquiva la explosión combinatoria del espacio de imágenes). Esto conecta con una distinción central del paper, tomada del *sequence learning*: las tareas secuenciales se dividen en **predicción** y **verificación**. Predecir un fotograma es el análogo visual de *word2vec*, pero inviable: las palabras viven en un vocabulario finito, mientras que predecir los píxeles de una imagen de 256×256 implica $256^{2 \times 3 \times 256}$ hipótesis. La salida es la **verificación**: en lugar de predecir el contenido de la secuencia, se predice solo su *validez* mediante una clasificación binaria.

La motivación es cognitiva: aprender de datos secuenciales es natural en humanos. La pregunta de juguete —"¿hacia dónde irá la pelota en movimiento?"— exige predecir a partir de la estructura temporal. Es la intuición que LeCun popularizó como el "pastel": la mayor parte del aprendizaje (la masa) debe venir de señales autosupervisadas abundantes —ordenar, completar, anticipar— y solo la guinda de las etiquetas.

## El pretexto: verificación de orden temporal

La contribución es un **pretext task** de verificación: dada una tupla de fotogramas de un video, predecir si están en el orden temporal correcto (positivo) o desordenados (negativo). El porqué de que esta tarea trivial induzca buenas representaciones es el corazón conceptual: para juzgar la validez de una secuencia, la red debe **razonar sobre cómo se transforman los objetos y sus localizaciones a través del tiempo**. Para saber si "una persona levantando una taza" está en orden, debe modelar implícitamente cómo se mueve el cuerpo humano — justo lo útil para reconocimiento de acciones y estimación de pose.

**¿Cuántos fotogramas?** Con dos fotogramas la tarea es ambigua bajo movimiento cíclico (no se distingue si se levanta o se deja la taza). Se usan **tres fotogramas**. Dada la secuencia $\{f_1,\dots,f_n\}$, la tupla $(f_b,f_c,f_d)$ es **orden correcto** (clase 1) si $b<c<d$ **o** $d<c<b$ —se admiten ambos sentidos, porque un clip invertido sigue siendo temporalmente plausible—, y **mal ordenada** (clase 0) si $b<d<c$ o $c<b<d$. Tres fotogramas no resuelven teóricamente toda ambigüedad cíclica (citando a Shannon), pero combinados con muestreo inteligente eliminan la mayoría de casos. Usar 4 o 5 fotogramas no mejoró.

## Muestreo por flujo óptico

El reto crítico es **cómo muestrear positivos y negativos**. Un muestreo uniforme genera muchos ejemplos ambiguos en ventanas con poco movimiento, donde los tres fotogramas se parecen demasiado. La solución: **muestrear solo de ventanas de alto movimiento**, medido con **flujo óptico grueso a nivel de fotograma** (Farnebäck). La magnitud media del flujo por fotograma sirve de **peso** que sesga el muestreo.

El procedimiento toma cinco fotogramas $a<b<c<d<e$ de una ventana de alto movimiento:

- **Positivos**: $(f_b, f_c, f_d)$.
- **Negativos**: $(f_b, f_a, f_d)$ y $(f_b, f_e, f_d)$.
- **Aumento por inversión**: invertir cualquier tupla genera ejemplos extra (p. ej. $(f_d, f_c, f_b)$ es positivo).

Hay un detalle de diseño sutil y muy influyente: durante el entrenamiento se **mantienen fijos el fotograma inicial $f_b$ y el final $f_d$, cambiando solo el del medio** tanto en positivos como en negativos. Así la red se ve forzada a enfocarse en la posición temporal del fotograma central —la señal que importa— en lugar de explotar **atajos** (iluminación, fondo). Es un precursor directo de toda la literatura posterior sobre *shortcut learning* en SSL. Además se exige que el fotograma central positivo $f_c$ no sea demasiado similar (vía SSD sobre RGB) a $f_a$ o $f_e$, para descartar negativos ambiguos. Los parámetros de muestreo recomendados: $\tau_{max}=60$ (distancia máxima de los positivos, controla su dificultad) y $\tau_{min}=15$ (distancia mínima de los negativos, valores bajos = más difíciles).

## Arquitectura: red Siamesa de triplete

Se usa una **red Siamesa de triplete**: tres pilas paralelas con **parámetros compartidos**, cada una con la arquitectura **CaffeNet** (variante de AlexNet) desde `conv1` hasta `fc7`. Cada pila recibe uno de los tres fotogramas y produce su `fc7`; las tres salidas se **concatenan** hacia una capa lineal que clasifica la tupla en dos clases (ordenada / desordenada), minimizando entropía cruzada regularizada.

Como las capas se comparten, la red tiene **el mismo número de parámetros que AlexNet** salvo la capa final. Esto es clave para la transferencia: en test se extrae la representación `conv1`–`fc7` de un *único* fotograma usando una sola pila. Pre-entrenamiento: ~900k tuplas de los videos de UCF101 **sin usar las etiquetas de acción**, 100k iteraciones, *learning rate* $10^{-3}$, *mini-batch* de 128, con *batch normalization*.

Dos decisiones de muestreo importan según la ablación: ventanas grandes para positivos ayudan, pero grandes para negativos perjudican; y conviene un **mayor porcentaje de negativos** (mejor configuración ~25% positivos). Crucialmente, la accuracy en la tarea pretext y la de reconocimiento de acciones están **correlacionadas** — evidencia de que un buen pretexto predice una buena transferencia.

## Qué captura: pose, no semántica

Antes de los números, el paper aporta evidencia cualitativa nítida. Los **vecinos más cercanos** con `fc7` sobre UCF101 muestran que **ImageNet se enfoca en la semántica de la escena**, mientras que la **red autosupervisada se enfoca en la pose de la persona** — información complementaria. La **visualización de unidades pool5** revela muchas unidades sensibles a partes del cuerpo y pose. Y el experimento de *fill in the blanks* (dado un fotograma inicial y final, predecir uno intermedio plausible) resuelve la ambigüedad direccional en acciones cíclicas con gran movimiento, fallando justo donde hay poco movimiento u objetos pequeños — exactamente donde el muestreo por flujo óptico tiene menos señal.

## Transferencia: action recognition y pose

**Reconocimiento de acciones** en UCF101 (101 clases) y HMDB51 (51 clases). Se inicializa la *spatial network* (solo RGB) con los pesos autosupervisados, se reinicia `fc8` y se hace *finetuning*.

| Dataset | Inicialización | Mean Accuracy |
|---|---|---|
| UCF101 | Random | 38.6 |
| UCF101 | **Tuple verification (Ours)** | **50.2** |
| HMDB51 | Random | 13.3 |
| HMDB51 | UCF Supervised | 15.2 |
| HMDB51 | **Tuple verification (Ours)** | **18.1** |

La ganancia de **+12.4%** sobre *scratch* en UCF101 y **+4.7%** en HMDB51 demuestra lo informativa que es la tarea. Más notable aún: en HMDB51 la red autosupervisada (18.1) **supera** a la pre-entrenada de forma supervisada en UCF101 (15.2). ImageNet supervisado sigue por delante (67.1 / 28.5), pero una señal *gratuita* recorta buena parte de la brecha. Comparada con otras tareas de ordenamiento (two-close, two-order, *DrLim*, *temporal coherence*, *object patch*), la verificación de **tres fotogramas gana por margen claro** (50.9 / 19.8 en el split 1), validando que tres fotogramas > dos y que verificar orden es más rico que mera suavidad temporal. Y combinada con ImageNet sube de 28.5 a 29.9 en HMDB51: **añade información complementaria** incluso sobre supervisión.

**Estimación de pose** en FLIC y MPII (predicción de keypoints estilo DeepPose, métrica PCK / PCKh@0.5). La red autosupervisada logra **84.7 (FLIC) y 85.8 (MPII)**, superando a *random init* (74.5 / 76.1), a *object patch*, a *DrLim* y al **pre-entrenamiento supervisado en UCF (+7.6% en FLIC, +2.1% en MPII)**. Es competitiva con ImageNet (85.8 / 85.1) y combinada con él da el mejor resultado (86.2 / 87.6). Lo cuantitativo confirma lo cualitativo: la verificación de orden aprende **pose humana** de videos sin etiquetas.

## Limitaciones

- **Tres fotogramas no resuelven toda la ambigüedad cíclica**; depende del muestreo inteligente (flujo óptico, restricción SSD). Acciones de bajo movimiento u objetos pequeños siguen fallando.
- **Dependencia del flujo óptico y del dominio de acción humana**: la representación se especializa en pose; no está claro cuánto transfiere a dominios sin movimiento humano dominante.
- **Arquitectura modesta y secuencias cortas**: CaffeNet/AlexNet y solo tres fotogramas; el trabajo futuro apunta a secuencias largas y CNN+RNN.
- **Brecha aún abierta con ImageNet supervisado** en reconocimiento de acciones (50.2 vs 67.1 en UCF101): la señal gratuita acorta la distancia pero no la cierra.

## Impacto: SSL temporal en video

*Shuffle and Learn* es un hito fundacional de los **pretext tasks temporales**. Usar el orden del tiempo como supervisión gratuita inauguró una línea fértil: predicción de la *arrow of time*, *sorting sequences* con más fotogramas, *Odd-One-Out networks*, predicción del *pace* de reproducción, y los métodos contrastivos/predictivos modernos sobre video (CPC temporal, *time-contrastive learning*, y arquitecturas que predicen representaciones futuras en espacio latente como JEPA). Articuló además la dicotomía **predicción vs. verificación** que sigue vigente: cuando predecir píxeles es inviable, *verificar una propiedad de la secuencia* es un sustituto tratable y sorprendentemente rico.

Tres ideas de diseño tuvieron eco duradero: **(1)** construir el pretexto para forzar el aprendizaje de lo que importa (cambiar solo el fotograma central anticipa la lucha contra los atajos en SSL); **(2)** muestrear positivos/negativos según una medida de informatividad (aquí, flujo óptico) en vez de uniformemente; **(3)** demostrar **complementariedad** con la supervisión en lugar de plantear el SSL como reemplazo total. Validar la representación vía *nearest neighbors* y visualización de unidades se volvió práctica estándar para interpretar modelos autosupervisados.

## Conexión con la Clase 28

La [Clase 28](/clases/clase-28) presenta el aprendizaje autosupervisado como el paradigma donde **la supervisión proviene de la propia estructura de los datos**. Este paper es el ejemplo canónico de **pretext temporal** (Misra et al. 2016). El mapeo es directo:

- **La señal gratuita = la flecha del tiempo.** Donde otros pretexts usan estructura espacial (parches, colorización, jigsaws), este usa la estructura *temporal*. El orden de los fotogramas es una etiqueta que el mundo provee gratis: encarna la idea de LeCun de "predecir/ordenar el futuro a partir del pasado" como motor del aprendizaje no supervisado.
- **Pretexto → downstream.** El patrón canónico —inventar una tarea con etiquetas automáticas, entrenar, transferir el backbone— se ilustra perfectamente: pretexto = verificación de orden; downstream = acciones y pose. La correlación medida entre accuracy pretext y downstream es la evidencia de que *un buen pretexto predice una buena transferencia*.
- **Diseñar contra atajos.** Mantener fijos $f_b$ y $f_d$ anticipa una gran lección del campo: un pretexto mal diseñado se resuelve por *shortcuts* sin aprender nada útil.
- **Complementariedad con la supervisión.** Los experimentos ImageNet + Tuple son evidencia temprana del SSL como pre-entrenamiento que se combina con etiquetas, no como su reemplazo.

Para profundizar, ver el fundamento transversal de [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) y el dominio de [video](/dominios/video), donde este paper se sitúa junto a los pretexts espaciales y los métodos contrastivos como uno de los pilares temporales del paradigma.
