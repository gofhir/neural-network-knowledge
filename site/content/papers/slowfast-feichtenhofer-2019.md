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
Tratar $x$ e $y$ **simétricamente** está justificado porque las imágenes naturales son aproximadamente **isótropas**; en una señal $I(x,y,t)$ la premisa se rompe, porque el movimiento es la contraparte espacio-temporal de la orientación y **no todas las orientaciones espacio-temporales son igualmente probables**. Entonces no hay razón para tratar espacio y tiempo por igual, que es justo lo que hacen los kernels cúbicos $N\times N\times N$ del inflado de C3D e [I3D](/papers/i3d-carreira-2017). La propuesta es factorizar en dos *pathways* que procesan **el mismo clip crudo a dos velocidades**. La vía **Slow** ($\tau=16$) ve 4 fotogramas de 64 y concentra toda la capacidad de canales: semántica espacial. La vía **Fast** (stride $\tau/\alpha=2$) ve 32 fotogramas del mismo clip con una fracción $\beta=1/8$ de los canales: movimiento de grano fino. Se unen por cuatro **conexiones laterales** unidireccionales Fast → Slow, y la vía Fast cuesta ~**20% del cómputo** y ~**1.6% de los parámetros** (0.53M contra 32.4M). Todas las cifras, **entrenando desde cero, sin ImageNet**: **79.8% top-1** en Kinetics-400 (16×8, R101+NL) a 234 GFLOPs × 30 vistas, contra 77.7% del estado del arte previo a 359; y 4×16 R50 logra **75.6%** con **36.1 GFLOPs**, empatando a Two-Stream I3D con **6× menos cómputo por vista**, sin flujo óptico y sin ImageNet. En AVA sube a **28.2 mAP** y gana el challenge 2019. El eslogan de la conclusión: *el eje temporal es una dimensión especial*.
{{< /paper-card >}}

---

## Contexto: la asimetría del video que nadie explotaba

El argumento de apertura es de estadística de la señal. La simetría entre $x$ e $y$ se sostiene porque todas las orientaciones espaciales son igualmente probables; en video el análogo de la orientación es el **movimiento**, y ahí la distribución está sesgada, porque la mayor parte del mundo está en reposo en un instante dado. El paper cita el **problema de la apertura**: un borde en movimiento aislado se percibe moviéndose perpendicular a sí mismo, percepto racional solo si el prior favorece movimientos lentos. Conclusión de diseño: hay que **factorizar** la red en vez de usar kernels cúbicos.

El sustento en [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones): **la semántica categórica evoluciona lentamente** —unas manos que se agitan no dejan de ser "manos"—, mientras que **el movimiento puede evolucionar mucho más rápido que la identidad de su sujeto**: aplaudir, agitar, saltar.

La motivación biológica conviene leerla con precisión. El paper se declara **parcialmente inspirado** en las células ganglionares de la retina de primates y añade textualmente que *"admitidamente la analogía es tosca y prematura"*.

| Tipo de célula | Proporción | Frecuencia temporal | Sensibilidad |
| --- | --- | --- | --- |
| Parvocelular (P) | ~80% | Baja | Detalle espacial fino y **color** |
| Magnocelular (M) | ~15–20% | **Alta** | **No** sensible a detalle ni a color |

**Afirma** solo tres correspondencias: dos vías a resolución temporal baja y alta; la Fast captura movimiento con menos detalle espacial, como las células M; y es liviana, como su pequeña proporción. **No** afirma modelar el sistema visual ni que la separación M/P se corresponda con lo aprendido. Incluso *pathway* en vez de *stream* es retórica declarada: SlowFast puede describirse como **un solo stream operando a dos framerates**.

> Frente al [Two-Stream clásico](/papers/two-stream-simonyan-2014), la separación deja de ser por **modalidad** (RGB contra flujo precomputado) y pasa a ser por **resolución temporal**: misma modalidad, distinto framerate y distinto presupuesto de canales. La especialización no se impone desde la entrada, se induce.

Eso resuelve la primera desventaja que la [Clase 38](/clases/clase-38) atribuye al Two-Stream —"necesita calcular el flujo óptico de cada video"—, un preprocesamiento costoso y no diferenciable que el paper llama "metodológicamente insatisfactorio" por ser una representación **diseñada a mano**.

---

## El pathway Slow

Su rasgo definitorio es el **stride temporal grande $\tau$**: procesa 1 de cada $\tau$ fotogramas. Con $\tau=16$ a 30 fps son ~2 fotogramas muestreados por segundo — ve $T=4$ fotogramas de un clip crudo de **64** (~2.13 s). Cuatro instantáneas.

Y **no hace ningún downsampling temporal**, ni pooling ni stride en el tiempo: si solo quedan cuatro fotogramas, comprimirlos más sería destructivo. El paper lo dice sin rodeos — hacerlo *"sería perjudicial cuando el stride de entrada es grande"*.

En compensación concentra **toda la capacidad de canales**: una [ResNet](/papers/resnet-he-2015) 3D con el ancho estándar de ResNet-50 (64 → 256 → 512 → 1024 → 2048). Con 32.4M de parámetros y 27.3 GFLOPs alcanza, sola, **72.6% top-1** en [Kinetics-400](/papers/kinetics-kay-2017): un baseline fuerte, lo que vuelve más significativa la ganancia de la segunda vía.

---

## El pathway Fast

**1. Alta resolución temporal, de punta a punta.** Stride $\tau/\alpha$; con $\alpha=8$ es **2**, así que muestrea $\alpha T=32$ fotogramas del mismo clip de 64, **8× más denso**. Y no basta con la entrada: **no hay ninguna capa de downsampling temporal en toda la vía** hasta el global pooling final — sus tensores tienen 32 fotogramas en conv1, en res2 y en res5. Correlativamente usa convoluciones temporales **no degeneradas** en cada bloque ($3\times1^2$ en el primer conv de cada bottleneck de res2 a res5, más un conv1 de $5\times7^2$): si se conserva resolución temporal fina, vale la pena filtrar en el tiempo.

**2. Baja capacidad de canales.** El corazón del truco: una fracción $\beta$ de los canales de Slow, típicamente $1/8$ (8 canales en conv1 en vez de 64, 256 en res5 en vez de 2048). El argumento es aritmético: los FLOPs de una capa convolucional son **cuadráticos** en la razón de escalado de canales, así que llevar los canales a $1/8$ reduce el cómputo por capa en torno a $1/64$ — margen de sobra para pagar el $8\times$ en fotogramas. Resultado: **6.4 GFLOPs** contra 27.3 de Slow y **0.53M de parámetros** contra 32.4M. Con las laterales incluidas, el incremento sobre Slow-only es de 8.8 GFLOPs (36.1 − 27.3), que el paper redondea a ~20% del cómputo — y no es casualidad que coincida con el ~15–20% de células M.

**3. Sin capacidad espacial fuerte.** Precisión importante: la vía Fast **no recibe tratamiento especial en la dimensión espacial** (mismos kernels y strides espaciales que Slow); su debilidad espacial es *consecuencia* de tener menos canales, no una decisión aparte. El tradeoff se declara: *"es deseable para la vía Fast debilitar su capacidad de modelado espacial mientras fortalece su capacidad de modelado temporal"*. La apariencia la aporta la otra vía, y eso es menos redundante que duplicarla.

Que "delgada pero rápida" sea la elección correcta no es teórico: la vía Fast **sola alcanza apenas 51.7% top-1**, y sin embargo aporta hasta **+3.0 puntos** en Kinetics y **+5.2 mAP** en AVA. Su valor está en la complementariedad, no en su desempeño aislado.

---

## Conexiones laterales

Sin fusión intermedia las dos vías se ignoran. El paper las une con **conexiones laterales** —técnica que Feichtenhofer ya usó para [fusionar redes two-stream](/papers/two-stream-fusion-feichtenhofer-2016)— una por stage: **después de `pool1`, `res2`, `res3` y `res4`** (no hay tras res5, donde ya ocurre el global pooling). Dirección **unidireccional, Fast → Slow**; la bidireccional dio resultados similares y se quedaron con la simple. Las formas no coinciden —Slow tiene $\{T,S^2,C\}$ y Fast $\{\alpha T,S^2,\beta C\}$—, y hay tres transformaciones candidatas, aquí con dimensiones en `pool1` (Fast: $32\times56^2\times8$):

| Estrategia | Transformación | En `pool1` | Nota |
| --- | --- | --- | --- |
| (i) Time-to-channel | $\{T,S^2,\alpha\beta C\}$ | $\to 4\times56^2\times64$ | Reshape+transpose; con $\alpha\beta=1$ iguala los canales de Slow y permite fusión por suma |
| (ii) Time-strided sampling | $\{T,S^2,\beta C\}$ | $\to 4\times56^2\times8$ | 1 de cada $\alpha$ fotogramas: descarta 7/8 del eje temporal |
| (iii) Time-strided convolution | conv 3D $5\times1^2$, stride $\alpha$ | $\to 4\times56^2\times16$ | **Aprende** cómo agregar la ventana; kernel 5 con stride 8 solapa parcialmente |

Ablation en Kinetics-400 con SlowFast 4×16 R-50:

| Variante | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| Slow-only | 72.6 | 90.3 | 27.3 |
| Fast-only | 51.7 | 78.5 | 6.4 |
| **Sin** laterales (concatena salidas finales) | 73.5 | 90.3 | 34.2 |
| Time-to-channel, suma | 74.5 | 91.3 | 34.2 |
| Time-to-channel, concatenación | 74.3 | 91.0 | 39.8 |
| Time-strided sampling | 75.4 | 91.8 | 34.9 |
| **Time-strided convolution** (defecto) | **75.6** | **92.1** | 36.1 |

**Sin laterales en la jerarquía la idea no funciona**: concatenar solo las salidas finales da 73.5%, apenas +0.9 sobre Slow-only. **T-conv gana**, +3.0 puntos por +8.8 GFLOPs. Y el gotcha: **time-strided sampling queda a 0.2 puntos** aunque descarte 7 de cada 8 fotogramas, lo que sugiere que lo importante ocurre *dentro* de la vía Fast y que la lateral solo necesita inyectar una señal, no transportar el detalle. Tras res5 hay global average pooling por vía y los dos vectores se concatenan hacia el clasificador.

---

## La instanciación concreta

Con $\alpha=8$, $\beta=1/8$, $\tau=16$, ResNet-50 y entrada $224\times224$. Notación $\{T\times S^2, C\}$; filtros temporales no degenerados marcados con $\dagger$.

| Stage | Slow pathway | Fast pathway | Output Slow / Fast |
| --- | --- | --- | --- |
| raw clip / data layer | stride $16,1^2$ | stride $2,1^2$ | $4\times224^2$ / $32\times224^2$ |
| conv1 | $1\times7^2,64$, stride $1,2^2$ | $5\times7^2,8\dagger$, stride $1,2^2$ | $4\times112^2$ / $32\times112^2$ |
| pool1 | max $1\times3^2$, stride $1,2^2$ | max $1\times3^2$, stride $1,2^2$ | $4\times56^2$ / $32\times56^2$ |
| res2 | $[1\times1^2,64;1\times3^2,64;1\times1^2,256]\times3$ | $[3\times1^2,8\dagger;1\times3^2,8;1\times1^2,32]\times3$ | $4\times56^2$ / $32\times56^2$ |
| res3 | $[1\times1^2,128;1\times3^2,128;1\times1^2,512]\times4$ | $[3\times1^2,16\dagger;1\times3^2,16;1\times1^2,64]\times4$ | $4\times28^2$ / $32\times28^2$ |
| res4 | $[3\times1^2,256\dagger;1\times3^2,256;1\times1^2,1024]\times6$ | $[3\times1^2,32\dagger;1\times3^2,32;1\times1^2,128]\times6$ | $4\times14^2$ / $32\times14^2$ |
| res5 | $[3\times1^2,512\dagger;1\times3^2,512;1\times1^2,2048]\times3$ | $[3\times1^2,64\dagger;1\times3^2,64;1\times1^2,256]\times3$ | $4\times7^2$ / $32\times7^2$ |
| salida | global average pool, concat, fc | | # clases |

**La columna temporal es constante en ambas vías** —32 en Fast, 4 en Slow, en todos los stages—; toda la reducción es espacial ($224 \to 112 \to 56 \to 28 \to 14 \to 7$). Ninguna toca el eje temporal, por razones opuestas: en Slow porque ya queda muy poco, en Fast porque es su especialidad.

El hallazgo contraintuitivo: **la vía Slow es esencialmente 2D en las capas tempranas.** A diferencia de C3D o I3D, que inflan *todos* los filtros a cúbicos, usa convoluciones temporales no degeneradas **solo en res4 y res5**. No es economía: los autores observaron que **ponerlas en capas tempranas degrada la precisión**, y el argumento es geométrico. Con $\tau=16$ pasan ~0.53 s entre fotogramas consecutivos de Slow, así que un objeto en movimiento se desplazó muchísimo más que los 7 píxeles del campo receptivo de conv1; un filtro cúbico ahí correlacionaría parches que ya no se solapan. En res4, con campo receptivo amplio, el objeto sí sigue dentro del campo.

> Corolario de diseño: **el ritmo con que se introduce el modelado temporal debe escalar con el campo receptivo espacial.** Los kernels cúbicos uniformes violan esa condición en las capas bajas. La vía Fast sí puede filtrar en el tiempo en cada bloque *porque* su stride temporal es 8× menor (~0.067 s) y ahí la correlación local existe. El fundamento [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones) discute cuándo el inflado sigue siendo necesario y cuándo no.

---

## Resultados y ablations

Kinetics-400, con GFLOPs por vista × número de vistas:

| Modelo | Flujo | Pretrain | top-1 | top-5 | GFLOPs × vistas |
| --- | --- | --- | --- | --- | --- |
| [I3D](/papers/i3d-carreira-2017) | | ImageNet | 72.1 | 90.3 | 108 × N/A |
| Two-Stream I3D | sí | ImageNet | 75.7 | 92.0 | 216 × N/A |
| Nonlocal R101 (SOTA previo) | | ImageNet | 77.7 | 93.3 | 359 × 30 |
| I3D | sí | — | 71.6 | 90.0 | 216 × N/A |
| R(2+1)D + flujo (mejor sin ImageNet) | sí | — | 73.9 | 90.9 | 304 × 115 |
| **SlowFast 4×16, R50** | | — | **75.6** | 92.1 | **36.1** × 30 |
| **SlowFast 8×8, R50** | | — | 77.0 | 92.6 | 65.7 × 30 |
| **SlowFast 8×8, R101** | | — | 77.9 | 93.2 | 106 × 30 |
| **SlowFast 16×8, R101** | | — | 78.9 | 93.5 | 213 × 30 |
| **SlowFast 16×8, R101+NL** | | — | **79.8** | **93.9** | 234 × 30 |

Contra **I3D RGB** (72.1%, 108 GFLOPs, con ImageNet), SlowFast 4×16 R50 gana **+3.5 puntos** con **3× menos cómputo por vista**. Contra **Two-Stream I3D** (75.7%, 216 GFLOPs, con ImageNet y con flujo) empata usando **6× menos FLOPs por vista**, sin flujo y sin ImageNet; y 8×8 R50 lo supera con 77.0% a 65.7 GFLOPs, aún 3.3× más barato. Contra el SOTA **Nonlocal R101**, el mejor SlowFast gana **+2.1 puntos** con 1.5× menos cómputo. Además, varios trabajos previos usan más de 100 vistas en inferencia (R(2+1)D 115, ARTNet 250); SlowFast usa **30** y le basta, porque la vía Fast ya cubre densamente el intervalo *dentro* de cada clip: la densidad temporal se paga una vez en la red, no al muestrear.

| $\beta$ | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| Slow-only | 72.6 | 90.3 | 27.3 |
| 1/4 | 75.6 | 91.7 | 54.5 |
| **1/6** | **75.8** | 92.0 | 41.8 |
| **1/8** (defecto) | 75.6 | **92.1** | 36.1 |
| 1/12 | 75.2 | 91.8 | 32.8 |
| 1/16 | 75.1 | 91.7 | 30.6 |
| 1/32 | 74.2 | 91.3 | 28.6 |

Lo notable es la robustez: **todo** el rango de $1/32$ a $1/4$ mejora sobre Slow-only, y $\beta=1/32$ agrega apenas 1.3 GFLOPs por +1.6 puntos. En el otro extremo, $\beta=1/4$ cuesta 54.5 GFLOPs (51% más que 1/8) para la **misma** top-1 y un top-5 *peor*: más canales en la vía rápida son desperdicio, la validación empírica de que **no debe** modelar apariencia. Sobre $\alpha$, el paper insiste en que su presencia *"está en el centro del concepto SlowFast"* porque es lo que fuerza la especialización de las subredes.

Variantes de entrada a la vía Fast, pensadas para debilitar su capacidad espacial de otras maneras:

| Entrada a la vía Fast | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| **RGB** ($\beta=1/8$, defecto) | **75.6** | **92.1** | 36.1 |
| RGB media resolución ($112^2$), $\beta=1/4$ | 74.7 | 91.8 | 34.4 |
| **Escala de grises** | 75.5 | 91.9 | 34.1 |
| Time difference (actual − anterior) | 74.5 | 91.6 | 34.2 |
| **[Flujo óptico](/fundamentos/flujo-optico)** | 73.8 | 91.3 | 35.1 |

Todas superan el 72.6% de Slow-only: la evidencia más fuerte de que la vía Fast representa movimiento, no apariencia. La **escala de grises está a 0.1 puntos del RGB** y ahorra ~5% de FLOPs — la vía rápida **no necesita color**, consistente con la insensibilidad al color de las células M. Y **el flujo óptico es la peor de las cinco opciones**: no es que el flujo sea caro y por eso lo evitemos, es que **el RGB a alta tasa temporal es mejor representación de movimiento que el flujo precomputado**, porque el flujo, diseñado a mano, descarta información aprovechable.

El control que cierra el arco, sobre la misma 3D ResNet-50:

| Modelo | Pretrain | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- | --- |
| Receta de Nonlocal | ImageNet | 73.4 | 90.9 | 36.7 |
| Receta de Nonlocal, desde cero | — | 69.4 | 88.6 | 36.7 |
| **Receta de los autores**, desde cero | — | **73.5** | 90.8 | 36.7 |

La receta antigua desde cero pierde **4.0 puntos**; la de SlowFast desde cero **iguala** a ImageNet, con SGD sincronizado de batch 1024 sobre 128 GPUs, warm-up lineal de 8k iteraciones para tolerar $\eta=1.6$, schedule cosenoidal de medio período y 256 epochs. **La brecha de "entrenar desde cero" era brecha de receta de optimización, no de datos.** Al probar ImageNet sobre SlowFast la diferencia fue de **±0.3%**: dejó de aportar.

> Aquí está la ironía que cierra el arco de la clase. El argumento central de [I3D](/papers/i3d-carreira-2017) es que heredar ImageNet vía el inflado era **indispensable** para las redes 3D profundas, y la clase lo lista como su ventaja principal. Dos años después SlowFast entrena **desde inicialización aleatoria** y alcanza el estado del arte. Lo que cambió no fue la arquitectura sino la **madurez de Kinetics como "ImageNet del video"** (~240k videos en K400, ~392k en K600, contra los ~10k de UCF-101 y HMDB-51), más una receta de optimización que en 2017 no era práctica estándar. El inflado fue una solución brillante a un problema de escasez de datos y recetas inmaduras, no una verdad arquitectónica permanente.

**AVA** (detección espacio-temporal, 60 clases, Faster R-CNN con SlowFast como backbone). Ablation limpio: Slow-only R-50 4×16 = **19.0 mAP** contra SlowFast = **24.2 mAP**, un **+5.2 mAP (28% relativo)** atribuible únicamente a la idea SlowFast; añadir flujo daba **+1.1 mAP** a I3D y **+1.7** a ATR, duplicando el cómputo. En v2.1, SlowFast R101 8×8 con solo Kinetics-400 logra **26.3 mAP** frente a 21.7 del mejor modelo único previo, y **28.2 mAP** en su mejor configuración; en v2.2 alcanza 30.7 con test multiescala, y un ensamble de 7 modelos llegó a **34.3 mAP**, **ganando el AVA action detection challenge 2019**. Mejora en **57 de 60 categorías** —máximos en "hand clap" +27.7 AP, "swim" +27.4, "run/jog" +18.8, "dance" +15.9— y empeora solo en tres, cuasi-estáticas y marginalmente: "answer phone" (−0.1), "lie/sleep" (−0.2), "shoot" (−0.4).

**Charades** (multi-etiqueta, ~30 s por actividad) es el test de estrés de largo alcance: SlowFast 16×8 R101 sube de **39.0 mAP** de su propio baseline Slow-only a **42.1**, y a **45.2 mAP** con non-local y pre-entrenamiento en Kinetics-600 — contra 39.7 del mejor previo (STRG R101+NL) a **2.7× menos costo** (234 contra 630 GFLOPs por vista). En **Kinetics-600** los SlowFast van de 78.8% a **81.8%**, contra 71.9% de I3D.

---

## Limitaciones

- **El costo absoluto sigue alto.** Ser más eficiente *por vista* no lo hace barato: el mejor modelo cuesta $234\times30 = 7020$ GFLOPs por video, y hasta el "económico" 4×16 R50 son 1083. Feichtenhofer atacaría esto al año siguiente con **X3D** (CVPR 2020).
- **El protocolo de 30 vistas optimiza leaderboard, no despliegue.** El paper reporta el factor ×30 explícitamente —aporte metodológico valioso, porque ese costo "había sido largamente ignorado"—, pero producción usaría menos vistas y perdería precisión de forma no cuantificada aquí.
- **La ventana temporal es de segundos, no de minutos.** Por defecto ve 64 fotogramas (~2.13 s) y la variante $16\times8$ llega a 128 (~4.3 s). El largo alcance se recupera solo promediando 10 clips, agregación sin memoria y sin orden: la desventaja de "no capturar relaciones temporales largas" queda **mejorada, no resuelta**.
- **La analogía biológica no se valida.** El paper mismo la califica de tosca y prematura, y su único contacto empírico (insensibilidad al color) es débil como evidencia.
- **La fusión es unidireccional y de topología fija.** Cuatro conexiones prefijadas; la bidireccional se reporta como "resultados similares" sin más análisis. No se exploran más de dos vías ni $\alpha$ o $\beta$ variables por stage.
- **Los ablations viven en una sola configuración**, casi siempre SlowFast 4×16 R-50; la transferencia de esos óptimos a modelos mayores se asume más que se verifica.

---

## Por qué importa hoy

SlowFast se convirtió en el **baseline obligado en detección de acciones**, muy especialmente en AVA: ganar el challenge 2019 con ese margen hizo que durante años cualquier trabajo en localización espacio-temporal tuviera que reportar frente a él, y el patrón "backbone SlowFast + Faster R-CNN con RoI 3D y RoIAlign" quedó como referencia de facto. **PySlowFast** fue tan importante como el paper: FAIR liberó no solo el modelo sino un framework completo de investigación en video, con entrenamiento distribuido, evaluación multi-vista y un zoo de modelos pre-entrenados; bajó la barrera de entrada y luego albergó X3D y MViT. También resolvió la dependencia del flujo óptico de la forma más elegante posible: no con un módulo de flujo aprendido ni con una pérdida auxiliar, sino **eliminando el problema**. Después de SlowFast, calcular TV-L1 sobre datasets de 240k videos dejó de ser parte del pipeline estándar.

Sobre la transición a los **video transformers**: el mismo grupo publicó **MViT (Multiscale Vision Transformers, ICCV 2021)**, con dos autores compartidos, y con ello el estado del arte migró de las ConvNets a la atención. Lo que sobrevive de SlowFast en ese salto no es la arquitectura de dos vías, sino la idea más profunda: **las jerarquías multiescala**. MViT rechaza el diseño del [Vision Transformer](/fundamentos/vision-transformer) de resolución y ancho constantes y construye una pirámide donde la resolución espacio-temporal se reduce mientras la dimensión de canales se expande con la profundidad — el mismo principio de "resolución alta con pocos canales / resolución baja con muchos canales" que SlowFast explotaba en paralelo sobre el eje temporal, ahora reorganizado sobre el eje de profundidad. Leído retrospectivamente, SlowFast es el argumento de que **el presupuesto de resolución temporal y el de capacidad de canales son recursos intercambiables**.

---

## Notas y enlaces

- **En el curso:** la [Clase 38](/clases/clase-38) recorre la evolución CNN2D → CNN2D+RNN → [Two-Stream](/papers/two-stream-simonyan-2014) → C3D → [I3D](/papers/i3d-carreira-2017) enumerando desventajas de cada familia; SlowFast es el cierre natural porque ataca de frente la lista de las dos últimas. La [teoría](/clases/clase-38/teoria) sitúa el paper en esa genealogía y la [profundización](/clases/clase-38/profundizacion) trata la matemática del inflado y de los kernels espacio-temporales.
- **El cambio de eje:** la clase presenta el Two-Stream como separación por **modalidad**; SlowFast cambia el eje a la **resolución temporal**. Tres consecuencias: desaparece un preprocesamiento costoso y no diferenciable, el modelo pasa a ser end-to-end, y la segunda corriente se vuelve casi gratis porque, al dedicarse solo al movimiento, no necesita capacidad para apariencia — algo que el Two-Stream, con el mismo backbone en ambos streams, nunca aprovechó.
- **Fundamentos:** [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones), [flujo óptico](/fundamentos/flujo-optico), [inflado de convoluciones](/fundamentos/inflado-de-convoluciones) y [Vision Transformer](/fundamentos/vision-transformer).
- **Linaje:** [Two-Stream (2014)](/papers/two-stream-simonyan-2014) plantea la separación original; [Two-Stream Fusion (2016)](/papers/two-stream-fusion-feichtenhofer-2016), del mismo primer autor, aporta las conexiones laterales que SlowFast reutiliza; [I3D (2017)](/papers/i3d-carreira-2017) y [Kinetics (2017)](/papers/kinetics-kay-2017) establecen el paradigma que SlowFast relativiza; [R(2+1)D (2018)](/papers/r2plus1d-tran-2018) y [S3D (2018)](/papers/s3d-xie-2018) son los otros ataques contemporáneos a la simetría de los kernels cúbicos, por factorización espacio/tiempo dentro de cada capa en vez de por framerate. El backbone es una [ResNet](/papers/resnet-he-2015) 3D, y la [Clase 36](/clases/clase-36) provee los datasets y el marco de evaluación del análisis de video.
- **Código:** [github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) (PySlowFast). Datasets: Kinetics-400, Kinetics-600, Charades y AVA.
