---
title: "SlowFast: dos vías por framerate (2019)"
weight: 424
math: true
---

{{< paper-card
    title="SlowFast Networks for Video Recognition"
    authors="Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He (Facebook AI Research)"
    year="2019"
    venue="ICCV 2019 / arXiv:1812.03982"
    arxiv="1812.03982"
    pdf="/papers/slowfast-feichtenhofer-2019.pdf" >}}
Tratar $x$ e $y$ **simétricamente** está justificado porque las imágenes naturales son casi **isótropas**; en $I(x,y,t)$ la premisa se rompe, porque el movimiento es la contraparte espacio-temporal de la orientación y **no todas las orientaciones espacio-temporales son igualmente probables**. Tratar espacio y tiempo por igual —lo que hacen los kernels cúbicos $N\times N\times N$ del inflado de C3D e [I3D](/papers/i3d-carreira-2017)— es un error de diseño. La alternativa: dos *pathways* sobre **el mismo clip crudo a dos velocidades**. La vía **Slow** ($\tau=16$) ve 4 fotogramas de 64 con toda la capacidad de canales: semántica. La vía **Fast** (stride $\tau/\alpha=2$) ve 32 fotogramas con una fracción $\beta=1/8$ de los canales: movimiento fino. Se unen por cuatro **conexiones laterales** Fast → Slow, y la Fast cuesta ~**20% del cómputo** y ~**1.6% de los parámetros**. Todas las cifras son **desde cero, sin ImageNet**: **79.8% top-1** en Kinetics-400 (16×8 R101+NL) a 234 GFLOPs × 30 vistas contra 77.7% del SOTA previo a 359, y 4×16 R50 logra **75.6%** con **36.1 GFLOPs**, empatando a Two-Stream I3D con **6× menos cómputo por vista** y sin flujo. En AVA sube a **28.2 mAP** y gana el challenge 2019. El eslogan: *el eje temporal es una dimensión especial*.
{{< /paper-card >}}

---

## Contexto: la asimetría del video que nadie explotaba

La simetría entre $x$ e $y$ se sostiene porque todas las orientaciones espaciales son igualmente probables; en video el análogo de la orientación es el **movimiento**, y ahí la distribución está sesgada, porque casi todo el mundo visible está en reposo en un instante dado. El paper cita el **problema de la apertura**: un borde aislado se percibe moviéndose perpendicular a sí mismo, percepto racional solo bajo un prior de movimiento lento. La conclusión es factorizar. El sustento en [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones): **la semántica categórica evoluciona lentamente** —unas manos que se agitan siguen siendo "manos"— mientras que **el movimiento evoluciona mucho más rápido que la identidad de su sujeto**.

La motivación biológica: el paper se declara **parcialmente inspirado** en las células ganglionares de la retina de primates, y admite que *"la analogía es tosca y prematura"* — nunca se valida.

| Célula | Proporción | Frec. temporal | Sensibilidad |
| --- | --- | --- | --- |
| Parvocelular (P) | ~80% | Baja | Detalle fino y **color** |
| Magnocelular (M) | ~15–20% | **Alta** | **Ni** detalle ni color |

**Afirma** solo tres correspondencias: dos vías a resolución temporal baja y alta; la Fast captura movimiento con menos detalle, como las células M; y es liviana, como su proporción. **No** afirma modelar el sistema visual, y hasta *pathway* en vez de *stream* es retórica declarada: SlowFast es **un stream a dos framerates**.

> Frente al [Two-Stream clásico](/papers/two-stream-simonyan-2014), la separación deja de ser por **modalidad** (RGB contra flujo precomputado) y pasa a ser por **resolución temporal**: misma modalidad, distinto framerate y distinto presupuesto de canales. La especialización no se impone, se induce — y la segunda corriente sale casi gratis, porque al dedicarse solo al movimiento no necesita capacidad para apariencia.

Eso elimina la primera desventaja que la [Clase 38](/clases/clase-38) atribuye al Two-Stream: "necesita calcular el flujo óptico de cada video", paso no diferenciable que el paper llama "metodológicamente insatisfactorio" por ser una representación **diseñada a mano**.

---

## El pathway Slow

Su rasgo definitorio es el **stride temporal grande $\tau$**: 1 de cada $\tau$ fotogramas. Con $\tau=16$ a 30 fps son ~2 por segundo, o sea $T=4$ fotogramas de un clip crudo de **64** (~2.13 s). Y **no hace ningún downsampling temporal**, ni pooling ni stride en el tiempo: con cuatro fotogramas, comprimir más *"sería perjudicial cuando el stride de entrada es grande"*.

En compensación concentra **toda la capacidad de canales**: una [ResNet](/papers/resnet-he-2015) 3D con el ancho estándar de ResNet-50 (64 → 256 → 512 → 1024 → 2048). Con 32.4M de parámetros y 27.3 GFLOPs alcanza sola **72.6% top-1** en [Kinetics-400](/papers/kinetics-kay-2017), un baseline fuerte que hace más significativa la ganancia de la segunda vía.

---

## El pathway Fast

**1. Alta resolución temporal, de punta a punta.** Stride $\tau/\alpha$; con $\alpha=8$ es **2**, así que muestrea $\alpha T=32$ fotogramas del mismo clip de 64, **8× más denso**. Y **no hay ninguna capa de downsampling temporal en toda la vía** hasta el global pooling: sus tensores tienen 32 fotogramas en conv1, res2 y res5. Correlativamente usa convoluciones temporales **no degeneradas** en cada bloque ($3\times1^2$ por bottleneck, más un conv1 de $5\times7^2$), que valen la pena justamente porque la resolución temporal fina se conserva.

**2. Baja capacidad de canales.** El corazón del truco: una fracción $\beta$ de los canales de Slow, típicamente $1/8$ (8 en conv1 en vez de 64). Los FLOPs de una capa convolucional son **cuadráticos** en la razón de escalado de canales, así que llevarlos a $1/8$ reduce el cómputo por capa en torno a $1/64$: margen de sobra para pagar el $8\times$ en fotogramas. Resultado, **6.4 GFLOPs** contra 27.3 de Slow y **0.53M de parámetros** contra 32.4M — un incremento total de 8.8 GFLOPs sobre Slow-only, ~20% del cómputo, que no por casualidad coincide con el ~15–20% de células M.

**3. Sin capacidad espacial fuerte.** La vía Fast **no recibe tratamiento especial en la dimensión espacial** (mismos kernels y strides que Slow): su debilidad espacial es *consecuencia* de tener menos canales. El tradeoff se declara —*"es deseable debilitar su modelado espacial mientras fortalece el temporal"*— y la apariencia la aporta la otra vía.

Que "delgada pero rápida" sea correcto no es teórico: la vía Fast **sola alcanza apenas 51.7% top-1**, y aun así aporta **+3.0 puntos** en Kinetics y **+5.2 mAP** en AVA. Su valor está en la complementariedad.

---

## Conexiones laterales

Sin fusión intermedia las dos vías se ignoran. El paper las une con **conexiones laterales** —técnica que Feichtenhofer ya usó para [fusionar two-stream](/papers/two-stream-fusion-feichtenhofer-2016)— una por stage: **tras `pool1`, `res2`, `res3` y `res4`** (no tras res5, donde ya ocurre el global pooling), y **unidireccionales Fast → Slow** porque la variante bidireccional rindió igual. Las formas no coinciden —Slow tiene $\{T,S^2,C\}$ y Fast $\{\alpha T,S^2,\beta C\}$—, y hay tres transformaciones candidatas, con dimensiones en `pool1` (Fast: $32\times56^2\times8$):

| Estrategia | Transformación de Fast | En `pool1` |
| --- | --- | --- |
| (i) Time-to-channel: reshape+transpose, y con $\alpha\beta=1$ habilita suma | $\{T,S^2,\alpha\beta C\}$ | $\to 4\times56^2\times64$ |
| (ii) Time-strided sampling: 1 de cada $\alpha$, descarta 7/8 del tiempo | $\{T,S^2,\beta C\}$ | $\to 4\times56^2\times8$ |
| (iii) Time-strided convolution: **aprende** la agregación | conv 3D $5\times1^2$, stride $\alpha$ | $\to 4\times56^2\times16$ |

Ablation con SlowFast 4×16 R-50:

| Variante | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| Slow-only | 72.6 | 90.3 | 27.3 |
| Fast-only | 51.7 | 78.5 | 6.4 |
| **Sin** laterales (concatena salidas finales) | 73.5 | 90.3 | 34.2 |
| Time-to-channel, suma | 74.5 | 91.3 | 34.2 |
| Time-strided sampling | 75.4 | 91.8 | 34.9 |
| **Time-strided convolution** (defecto) | **75.6** | **92.1** | 36.1 |

**Sin laterales en la jerarquía la idea no funciona**: concatenar solo las salidas finales da 73.5%, +0.9 sobre Slow-only. **T-conv gana** (+3.0 puntos por +8.8 GFLOPs), pero **time-strided sampling queda a 0.2 puntos** pese a descartar 7 de cada 8 fotogramas: lo importante ocurre *dentro* de la vía Fast, y la lateral solo inyecta señal. Tras res5 se hace global average pooling por vía y se concatena al clasificador.

---

## La instanciación concreta

Con $\alpha=8$, $\beta=1/8$, $\tau=16$, ResNet-50 y entrada $224\times224$; notación $\{T\times S^2, C\}$, filtros temporales no degenerados con $\dagger$. **La columna temporal es constante en ambas vías** —32 en Fast, 4 en Slow— y toda la reducción es espacial.

| Stage | Slow pathway | Fast pathway | Output Slow / Fast |
| --- | --- | --- | --- |
| data layer (clip de 64) | stride $16,1^2$ | stride $2,1^2$ | $4\times224^2$ / $32\times224^2$ |
| conv1 | $1\times7^2,64$, stride $1,2^2$ | $5\times7^2,8\dagger$, stride $1,2^2$ | $4\times112^2$ / $32\times112^2$ |
| pool1 | max $1\times3^2$, stride $1,2^2$ | max $1\times3^2$, stride $1,2^2$ | $4\times56^2$ / $32\times56^2$ |
| res2 | $[1\times1^2,64;1\times3^2,64;1\times1^2,256]\times3$ | $[3\times1^2,8\dagger;1\times3^2,8;1\times1^2,32]\times3$ | $4\times56^2$ / $32\times56^2$ |
| res3 | $[1\times1^2,128;1\times3^2,128;1\times1^2,512]\times4$ | $[3\times1^2,16\dagger;1\times3^2,16;1\times1^2,64]\times4$ | $4\times28^2$ / $32\times28^2$ |
| res4 | $[3\times1^2,256\dagger;1\times3^2,256;1\times1^2,1024]\times6$ | $[3\times1^2,32\dagger;1\times3^2,32;1\times1^2,128]\times6$ | $4\times14^2$ / $32\times14^2$ |
| res5 | $[3\times1^2,512\dagger;1\times3^2,512;1\times1^2,2048]\times3$ | $[3\times1^2,64\dagger;1\times3^2,64;1\times1^2,256]\times3$ | $4\times7^2$ / $32\times7^2$ |
| salida | global average pool, concat, fc | | # clases |

Ninguna vía toca el eje temporal, por razones opuestas: en Slow ya queda muy poco, en Fast es su especialidad.

El hallazgo contraintuitivo: **la vía Slow es esencialmente 2D en las capas tempranas.** A diferencia de C3D o I3D, que inflan *todos* los filtros a cúbicos, usa convoluciones temporales no degeneradas **solo en res4 y res5**, porque **ponerlas antes degrada la precisión**. El argumento es geométrico: con $\tau=16$ pasan ~0.53 s entre fotogramas de Slow, un objeto se desplaza muchísimo más que los 7 píxeles del campo receptivo de conv1, y un filtro cúbico ahí correlacionaría parches que ya no se solapan. En res4 el campo receptivo es amplio y el objeto sigue dentro.

> Corolario: **el ritmo con que se introduce el modelado temporal debe escalar con el campo receptivo espacial.** Los kernels cúbicos uniformes violan esa condición abajo; la vía Fast sí puede filtrar en el tiempo *porque* su stride temporal es 8× menor (~0.067 s). El fundamento [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones) discute cuándo el inflado sigue haciendo falta.

---

## Resultados y ablations

Kinetics-400 (GFLOPs por vista × vistas):

| Modelo | Flujo | Pretrain | top-1 | top-5 | GFLOPs × vistas |
| --- | --- | --- | --- | --- | --- |
| [I3D](/papers/i3d-carreira-2017) | | ImageNet | 72.1 | 90.3 | 108 × N/A |
| Two-Stream I3D | sí | ImageNet | 75.7 | 92.0 | 216 × N/A |
| Nonlocal R101 (SOTA previo) | | ImageNet | 77.7 | 93.3 | 359 × 30 |
| R(2+1)D + flujo (mejor sin ImageNet) | sí | — | 73.9 | 90.9 | 304 × 115 |
| **SlowFast 4×16, R50** | | — | **75.6** | 92.1 | **36.1** × 30 |
| **SlowFast 8×8, R50** | | — | 77.0 | 92.6 | 65.7 × 30 |
| **SlowFast 16×8, R101** | | — | 78.9 | 93.5 | 213 × 30 |
| **SlowFast 16×8, R101+NL** | | — | **79.8** | **93.9** | 234 × 30 |

Contra **I3D RGB** (con ImageNet), SlowFast 4×16 R50 gana **+3.5 puntos** con **3× menos cómputo por vista**; contra **Two-Stream I3D** (con ImageNet y con flujo) empata con **6× menos FLOPs por vista**, y 8×8 R50 lo supera; contra **Nonlocal R101** gana **+2.1 puntos** con 1.5× menos cómputo. Y donde otros usan más de 100 vistas en inferencia, a SlowFast le bastan **30**: la vía Fast ya cubre densamente cada clip.

| $\beta$ | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| Slow-only | 72.6 | 90.3 | 27.3 |
| 1/4 | 75.6 | 91.7 | 54.5 |
| **1/6** | **75.8** | 92.0 | 41.8 |
| **1/8** (defecto) | 75.6 | **92.1** | 36.1 |
| 1/32 | 74.2 | 91.3 | 28.6 |

Lo notable es la robustez: **todo** el rango de $1/32$ a $1/4$ mejora sobre Slow-only, y $\beta=1/32$ agrega 1.3 GFLOPs por +1.6 puntos. En el otro extremo, $\beta=1/4$ cuesta 51% más que 1/8 para la **misma** top-1 y un top-5 *peor*: más canales en la vía rápida son desperdicio puro, la validación de que **no debe** modelar apariencia. De $\alpha$ el paper dice que *"está en el centro del concepto SlowFast"*.

Otras formas de debilitar su capacidad espacial:

| Entrada a la vía Fast | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| **RGB** ($\beta=1/8$, defecto) | **75.6** | **92.1** | 36.1 |
| RGB media resolución ($112^2$), $\beta=1/4$ | 74.7 | 91.8 | 34.4 |
| **Escala de grises** | 75.5 | 91.9 | 34.1 |
| Time difference | 74.5 | 91.6 | 34.2 |
| **[Flujo óptico](/fundamentos/flujo-optico)** | 73.8 | 91.3 | 35.1 |

Todas superan el 72.6% de Slow-only: la vía Fast codifica movimiento, no apariencia. La **escala de grises está a 0.1 puntos del RGB** y ahorra ~5% de FLOPs, así que la vía rápida **no necesita color** (consistente con las células M). Y **el flujo óptico es la peor de las cinco**: no lo evitamos porque sea caro, sino porque **el RGB a alta tasa temporal representa mejor el movimiento que el flujo precomputado**, que al estar diseñado a mano descarta información aprovechable.

El control decisivo, sobre la misma 3D ResNet-50:

| Régimen (3D R-50, 36.7 GFLOPs) | Pretrain | top-1 | top-5 |
| --- | --- | --- | --- |
| Receta de Nonlocal | ImageNet | 73.4 | 90.9 |
| Receta de Nonlocal, desde cero | — | 69.4 | 88.6 |
| **Receta de los autores**, desde cero | — | **73.5** | 90.8 |

La receta antigua desde cero pierde **4.0 puntos**; la de SlowFast **iguala** a ImageNet con SGD sincronizado de batch 1024 sobre 128 GPUs, warm-up lineal de 8k iteraciones para tolerar $\eta=1.6$ y 256 epochs. **La brecha de "entrenar desde cero" era de optimización, no de datos**, y con ImageNet la diferencia fue de **±0.3%**.

> La ironía que cierra el arco de la clase: el argumento central de [I3D](/papers/i3d-carreira-2017) es que heredar ImageNet vía el inflado era **indispensable** para las redes 3D profundas, y dos años después SlowFast entrena **desde inicialización aleatoria** y alcanza el estado del arte. No cambió la arquitectura, cambió la **madurez de Kinetics como "ImageNet del video"** (~240k videos contra los ~10k de UCF-101 y HMDB-51) y una receta de optimización inexistente en 2017. El inflado resolvía una escasez de datos; no era una verdad permanente.

**AVA** (detección espacio-temporal, Faster R-CNN con SlowFast de backbone). Slow-only R-50 4×16 = **19.0 mAP** contra SlowFast = **24.2 mAP**: **+5.2 mAP (28% relativo)** por la sola idea SlowFast, cuando el flujo daba **+1.1 mAP** a I3D y **+1.7** a ATR duplicando el cómputo. En v2.1, R101 8×8 con Kinetics-400 logra **26.3 mAP** frente a 21.7 del mejor modelo único previo, y **28.2** en su mejor configuración; un ensamble de 7 llegó a **34.3 mAP**, **ganando el challenge 2019**. Mejora en **57 de 60 categorías** ("hand clap" +27.7 AP) y solo empeora, marginalmente, en tres cuasi-estáticas.

**Charades** (~30 s por actividad) es el test de largo alcance: 16×8 R101 sube de **39.0 mAP** de su baseline Slow-only a **42.1**, y a **45.2** con non-local y Kinetics-600 — contra 39.7 del mejor previo, a **2.7× menos costo**. En **Kinetics-600** llega a **81.8%** contra 71.9% de I3D.

---

## Limitaciones

- **El costo absoluto sigue alto:** el mejor modelo cuesta $234\times30 = 7020$ GFLOPs por video; el "económico" 4×16 R50, 1083. Feichtenhofer atacaría esto con **X3D** (2020).
- **El protocolo de 30 vistas optimiza leaderboard, no despliegue.** Reportarlo explícitamente es un aporte metodológico valioso —ese costo "había sido largamente ignorado"—, pero producción usaría menos vistas, con una pérdida que el paper no cuantifica.
- **La ventana temporal es de segundos, no de minutos:** 64 fotogramas por defecto (~2.13 s), 128 en $16\times8$, y el largo alcance se recupera promediando 10 clips, agregación sin memoria ni orden. La desventaja de "no capturar relaciones temporales largas" queda **mejorada, no resuelta**.
- **La fusión es unidireccional y de topología fija:** la bidireccional se despacha como "resultados similares", no se exploran más de dos vías ni $\alpha$/$\beta$ por stage, y casi todos los ablations viven en una sola configuración (4×16 R-50).

---

## Por qué importa hoy

SlowFast se volvió el **baseline obligado en detección de acciones**, sobre todo en AVA: ganar el challenge 2019 con ese margen obligó durante años a que todo trabajo de localización espacio-temporal reportara frente a él, y el patrón "SlowFast + Faster R-CNN con RoI 3D y RoIAlign" quedó como referencia de facto. **PySlowFast** importó tanto como el paper: FAIR liberó un framework completo de investigación en video (entrenamiento distribuido, evaluación multi-vista, zoo de modelos) que bajó la barrera de entrada y luego albergó X3D y MViT. Y resolvió la dependencia del flujo **eliminando el problema**: calcular TV-L1 sobre 240k videos dejó de ser estándar.

Sobre los **video transformers**: el mismo grupo publicó **MViT (Multiscale Vision Transformers, ICCV 2021)**, con dos autores compartidos, y el estado del arte migró de las ConvNets a la atención. Lo que sobrevive de SlowFast ahí no son las dos vías, sino **las jerarquías multiescala**: MViT rechaza el diseño del [Vision Transformer](/fundamentos/vision-transformer) de resolución y ancho constantes y construye una pirámide donde la resolución espacio-temporal se reduce mientras los canales se expanden con la profundidad — el principio de "resolución alta con pocos canales / resolución baja con muchos", reorganizado del eje temporal al de profundidad. SlowFast se lee retrospectivamente como el argumento de que **resolución temporal y capacidad de canales son recursos intercambiables**.

---

## Notas y enlaces

- **En el curso:** la [Clase 38](/clases/clase-38) recorre CNN2D → CNN2D+RNN → [Two-Stream](/papers/two-stream-simonyan-2014) → C3D → [I3D](/papers/i3d-carreira-2017) con las desventajas de cada familia, y SlowFast ataca las de las dos últimas. La [teoría](/clases/clase-38/teoria) lo sitúa en esa genealogía; la [profundización](/clases/clase-38/profundizacion), en el inflado.
- **Linaje:** [Two-Stream (2014)](/papers/two-stream-simonyan-2014) plantea la separación original y [Two-Stream Fusion (2016)](/papers/two-stream-fusion-feichtenhofer-2016), del mismo primer autor, aporta las conexiones laterales; [I3D (2017)](/papers/i3d-carreira-2017) y [Kinetics (2017)](/papers/kinetics-kay-2017) fijan el paradigma que SlowFast relativiza; [R(2+1)D (2018)](/papers/r2plus1d-tran-2018) y [S3D (2018)](/papers/s3d-xie-2018) atacan la misma simetría cúbica factorizando dentro de cada capa. Ver también [ResNet](/papers/resnet-he-2015) y la [Clase 36](/clases/clase-36), con los datasets y el marco de evaluación del video.
- **Código:** [PySlowFast](https://github.com/facebookresearch/SlowFast). Datasets: Kinetics-400/600, Charades y AVA.
