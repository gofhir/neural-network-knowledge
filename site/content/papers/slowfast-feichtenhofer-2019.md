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
El paper parte de una observación estadística que la generación de redes 3D había pasado por alto: tratar $x$ e $y$ **simétricamente** está justificado porque las imágenes naturales son aproximadamente **isótropas**, pero en una señal $I(x,y,t)$ la premisa se rompe — el movimiento es la contraparte espacio-temporal de la orientación y **no todas las orientaciones espacio-temporales son igualmente probables**. Si es así, no hay razón para tratar espacio y tiempo por igual, que es exactamente lo que hacen los kernels cúbicos $N\times N\times N$ del inflado de C3D e [I3D](/papers/i3d-carreira-2017). La propuesta es factorizar: dos *pathways* que procesan **el mismo clip crudo a dos velocidades**. Una vía **Slow** con stride temporal grande ($\tau=16$) ve 4 fotogramas de 64 y concentra toda la capacidad de canales, a cargo de la semántica espacial; una vía **Fast** con stride $\tau/\alpha=2$ ve 32 fotogramas del mismo clip con una fracción $\beta=1/8$ de los canales, a cargo del movimiento de grano fino. Se unen por cuatro **conexiones laterales** unidireccionales Fast → Slow. La vía Fast consume ~**20% del cómputo** y ~**1.6% de los parámetros** (0.53M contra 32.4M). Todas las cifras, **entrenando desde cero, sin ImageNet**: **79.8% top-1** en Kinetics-400 (16×8, R101+NL) a 234 GFLOPs × 30 vistas, contra 77.7% del estado del arte previo a 359 GFLOPs; y 4×16 R50 logra **75.6%** con **36.1 GFLOPs**, empatando a Two-Stream I3D con **6× menos cómputo por vista**, sin flujo óptico y sin ImageNet. En AVA sube a **28.2 mAP** y gana el challenge 2019. El eslogan de la conclusión: *el eje temporal es una dimensión especial*.
{{< /paper-card >}}

---

## Contexto: la asimetría del video que nadie explotaba

El argumento de apertura es de estadística de la señal, no de ingeniería. La simetría entre $x$ e $y$ se sostiene porque todas las orientaciones espaciales son igualmente probables. En video el análogo de la orientación es el **movimiento**, y ahí la distribución está sesgada: la mayor parte del mundo está en reposo en un instante dado, así que los movimientos lentos son más probables. El paper cita el **problema de la apertura** como evidencia: un borde en movimiento aislado se percibe moviéndose perpendicular a sí mismo, percepto racional solo si el prior favorece movimientos lentos.

De ahí el diagnóstico: si las orientaciones espacio-temporales no son equiprobables, hay que **factorizar** la red en vez de usar kernels cúbicos. La justificación en [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) es clara: **la semántica categórica evoluciona lentamente** —unas manos que se agitan no dejan de ser "manos", y color, textura e iluminación también se refrescan lento—, mientras que **el movimiento puede evolucionar mucho más rápido que la identidad de su sujeto**: aplaudir, agitar, saltar.

La motivación biológica conviene leerla con precisión. El paper dice estar **parcialmente inspirado** en las células ganglionares de la retina de primates y añade, textualmente, que *"admitidamente la analogía es tosca y prematura"*.

| Tipo de célula | Proporción | Frecuencia temporal | Sensibilidad |
| --- | --- | --- | --- |
| Parvocelular (P) | ~80% | Baja | Detalle espacial fino y **color** |
| Magnocelular (M) | ~15–20% | **Alta** | **No** sensible a detalle espacial ni a color |

Lo que **afirma** como analogía es solo esto: dos vías a resoluciones temporales baja y alta; la vía Fast captura movimiento con menos detalle espacial, como las células M; y es liviana, como su pequeña proporción. Lo que **no** afirma es que la arquitectura modele el sistema visual ni que la separación M/P se corresponda funcionalmente con lo aprendido. Incluso *pathway* en lugar de *stream* es una elección retórica declarada: los autores dicen que SlowFast puede describirse como **un solo stream operando a dos framerates**.

> Frente al [Two-Stream clásico](/papers/two-stream-simonyan-2014), la separación deja de ser por **modalidad** (RGB contra flujo óptico precomputado) y pasa a ser por **resolución temporal**. Ambas vías reciben la misma modalidad; difieren en el framerate al que la muestrean y en el presupuesto de canales. La especialización no se impone desde la entrada: se induce.

Eso resuelve de frente la primera desventaja que la [Clase 38](/clases/clase-38) le atribuye al Two-Stream —"necesita calcular el flujo óptico de cada video"—, un preprocesamiento costoso y no diferenciable. El paper es duro: el flujo es "metodológicamente insatisfactorio" por ser una representación **diseñada a mano**.

---

## El pathway Slow

Puede ser cualquier ConvNet que opere sobre el clip como volumen espacio-temporal; su rasgo definitorio es un **stride temporal grande $\tau$**: procesa 1 de cada $\tau$ fotogramas. Con $\tau=16$ sobre video a 30 fps la tasa de refresco es de ~2 fotogramas por segundo — ve $T=4$ fotogramas de un clip crudo de **64** (~2.13 s). Cuatro instantáneas.

El detalle contraintuitivo: **no hace ningún downsampling temporal**, ni pooling ni stride en el tiempo. Es consecuencia lógica del muestreo agresivo de entrada —si solo quedan cuatro fotogramas, comprimirlos más sería destructivo— y el paper lo dice sin rodeos: hacerlo *"sería perjudicial cuando el stride de entrada es grande"*.

En compensación concentra **toda la capacidad de canales**: una [ResNet](/papers/resnet-he-2015) 3D con el ancho estándar de ResNet-50 (64 → 256 → 512 → 1024 → 2048). Con 32.4M de parámetros y 27.3 GFLOPs alcanza, sola, **72.6% top-1** en [Kinetics-400](/papers/kinetics-kay-2017): un baseline fuerte, lo que hace más significativa la ganancia de la segunda vía.

---

## El pathway Fast

**1. Alta resolución temporal, de punta a punta.** Stride temporal $\tau/\alpha$; con $\alpha=8$ es **2**, así que muestrea $\alpha T=32$ fotogramas del mismo clip de 64, **8× más denso**. Y no basta con la entrada: la vía Fast **no tiene ninguna capa de downsampling temporal en toda la red** hasta el global pooling final — sus tensores tienen 32 fotogramas en conv1, en res2 y en res5. Correlativamente usa convoluciones temporales **no degeneradas** en cada bloque ($3\times1^2$ en el primer conv de cada bottleneck de res2 a res5, más un conv1 de $5\times7^2$): como conserva resolución temporal fina, vale la pena poner filtros temporales.

**2. Baja capacidad de canales.** Es el corazón del truco: una fracción $\beta$ de los canales de Slow, típicamente $1/8$ (8 canales en conv1 en vez de 64, 256 en res5 en vez de 2048). El argumento es aritmético: los FLOPs de una capa convolucional son **cuadráticos** en la razón de escalado de canales, así que llevar los canales a $1/8$ reduce el cómputo por capa en torno a $1/64$ — margen de sobra para pagar el $8\times$ en fotogramas. Resultado: **6.4 GFLOPs** contra 27.3 de Slow y **0.53M de parámetros** contra 32.4M. Con las conexiones laterales incluidas, el incremento total sobre Slow-only es de 8.8 GFLOPs (36.1 − 27.3); el paper lo redondea a ~20% del cómputo, y no es casualidad que coincida con el ~15–20% de células M.

**3. Sin capacidad espacial fuerte.** Precisión importante: la vía Fast **no recibe ningún tratamiento especial en la dimensión espacial** —usa los mismos kernels y strides espaciales que Slow—; su debilidad espacial es *consecuencia* de tener menos canales, no una decisión separada. El tradeoff se declara: *"es un tradeoff deseable para la vía Fast debilitar su capacidad de modelado espacial mientras fortalece su capacidad de modelado temporal"*. La apariencia la aporta la otra vía, y eso es menos redundante que duplicarla.

Que "delgada pero rápida" sea la elección correcta no es teórico: la vía Fast **sola alcanza apenas 51.7% top-1**, un modelo mediocre, y sin embargo aporta hasta **+3.0 puntos** a la vía Slow en Kinetics y **+5.2 mAP** en AVA. Su valor está en la complementariedad, no en su desempeño aislado.

---

## Conexiones laterales

Sin fusión intermedia las dos vías se ignoran. El paper las une con **conexiones laterales** —técnica que Feichtenhofer ya había usado para [fusionar redes two-stream](/papers/two-stream-fusion-feichtenhofer-2016)— insertadas una por stage: **después de `pool1`, `res2`, `res3` y `res4`** (no hay una tras res5 porque ahí ya ocurre el global pooling). La dirección es **unidireccional, Fast → Slow**; la variante bidireccional dio resultados similares y se quedaron con la simple. El problema es que las formas no coinciden: Slow tiene $\{T,S^2,C\}$ y Fast tiene $\{\alpha T,S^2,\beta C\}$. Las tres transformaciones evaluadas, con dimensiones en `pool1` (donde Fast es $32\times56^2\times8$):

| Estrategia | Transformación | En `pool1` | Comentario |
| --- | --- | --- | --- |
| (i) Time-to-channel | $\{T,S^2,\alpha\beta C\}$ | $\to 4\times56^2\times64$ | Reshape + transpose: empaqueta los $\alpha$ fotogramas en los canales de uno. Con $\alpha\beta=1$ iguala los canales de Slow, así que permite fusión por suma. |
| (ii) Time-strided sampling | $\{T,S^2,\beta C\}$ | $\to 4\times56^2\times8$ | Toma 1 de cada $\alpha$ fotogramas: descarta 7/8 de la información temporal. |
| (iii) Time-strided convolution | Conv 3D $5\times1^2$, $2\beta C$ salidas, stride $\alpha$ | $\to 4\times56^2\times16$ | **Aprende** cómo agregar la ventana temporal. Kernel 5 con stride 8: solapa parcialmente. |

Ablation sobre Kinetics-400, SlowFast 4×16 R-50:

| Variante | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| Slow-only | 72.6 | 90.3 | 27.3 |
| Fast-only | 51.7 | 78.5 | 6.4 |
| **Sin** laterales (solo concatena salidas finales) | 73.5 | 90.3 | 34.2 |
| Time-to-channel, suma | 74.5 | 91.3 | 34.2 |
| Time-to-channel, concatenación | 74.3 | 91.0 | 39.8 |
| Time-strided sampling | 75.4 | 91.8 | 34.9 |
| **Time-strided convolution** (por defecto) | **75.6** | **92.1** | 36.1 |

Tres lecturas. **Sin conexiones laterales en la jerarquía la idea no funciona**: concatenar solo las salidas finales da 73.5%, apenas +0.9 sobre Slow-only. **T-conv gana**, con +3.0 puntos sobre Slow-only por +8.8 GFLOPs. Y el gotcha: **time-strided sampling queda a solo 0.2 puntos** aunque descarte 7 de cada 8 fotogramas, lo que sugiere que lo importante ocurre *dentro* de la vía Fast y que la conexión lateral solo necesita inyectar una señal, no transportar todo el detalle. Tras res5 hay global average pooling por vía y los dos vectores se concatenan hacia el clasificador.

---

## La instanciación concreta

Con $\alpha=8$, $\beta=1/8$, $\tau=16$, backbone ResNet-50, entrada $224\times224$. Notación $\{T\times S^2, C\}$; filtros temporales no degenerados marcados con $\dagger$.

| Stage | Slow pathway | Fast pathway | Output $T\times S^2$ |
| --- | --- | --- | --- |
| raw clip | — | — | $64\times224^2$ |
| data layer | stride $16,1^2$ | stride $2,1^2$ | $4\times224^2$ / $32\times224^2$ |
| conv1 | $1\times7^2, 64$, stride $1,2^2$ | $5\times7^2, 8\ \dagger$, stride $1,2^2$ | $4\times112^2$ / $32\times112^2$ |
| pool1 | max $1\times3^2$, stride $1,2^2$ | max $1\times3^2$, stride $1,2^2$ | $4\times56^2$ / $32\times56^2$ |
| res2 | $[1\times1^2,64;\ 1\times3^2,64;\ 1\times1^2,256]\times3$ | $[3\times1^2,8\ \dagger;\ 1\times3^2,8;\ 1\times1^2,32]\times3$ | $4\times56^2$ / $32\times56^2$ |
| res3 | $[1\times1^2,128;\ 1\times3^2,128;\ 1\times1^2,512]\times4$ | $[3\times1^2,16\ \dagger;\ 1\times3^2,16;\ 1\times1^2,64]\times4$ | $4\times28^2$ / $32\times28^2$ |
| res4 | $[3\times1^2,256\ \dagger;\ 1\times3^2,256;\ 1\times1^2,1024]\times6$ | $[3\times1^2,32\ \dagger;\ 1\times3^2,32;\ 1\times1^2,128]\times6$ | $4\times14^2$ / $32\times14^2$ |
| res5 | $[3\times1^2,512\ \dagger;\ 1\times3^2,512;\ 1\times1^2,2048]\times3$ | $[3\times1^2,64\ \dagger;\ 1\times3^2,64;\ 1\times1^2,256]\times3$ | $4\times7^2$ / $32\times7^2$ |
| — | global average pool, concatenate, fc | | # clases |

**La columna temporal es constante en ambas vías** —32 en Fast, 4 en Slow, en todos los stages—; toda la reducción es espacial ($224 \to 112 \to 56 \to 28 \to 14 \to 7$). Ninguna toca el eje temporal, por razones opuestas: en Slow porque ya queda muy poco, en Fast porque es su especialidad.

El hallazgo contraintuitivo: **la vía Slow es esencialmente 2D en las capas tempranas.** A diferencia de C3D o I3D, que inflan *todos* los filtros a cúbicos, usa convoluciones temporales no degeneradas **solo en res4 y res5**; de conv1 a res3 sus filtros son $1\times7^2$, $1\times1^2$ o $1\times3^2$. No es economía: los autores observaron que **poner convoluciones temporales en capas tempranas degrada la precisión**, y el argumento es geométrico. Con $\tau=16$ pasan ~0.53 s entre fotogramas consecutivos de la vía Slow, así que un objeto en movimiento se desplazó muchísimo más que los 7 píxeles del campo receptivo de conv1; un filtro cúbico ahí correlacionaría parches que ya no se solapan en absoluto. En res4, con campo receptivo amplio, el objeto sí sigue dentro del campo.

> Corolario de diseño: **el ritmo con que se introduce el modelado temporal debe escalar con el campo receptivo espacial.** Los kernels cúbicos uniformes violan esa condición en las capas bajas. La vía Fast sí puede tener convoluciones temporales en cada bloque *porque* su stride temporal es 8× menor (~0.067 s), y ahí la correlación local existe. El fundamento [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones) discute cuándo el inflado sigue siendo necesario y cuándo no.

---

## Resultados y ablations

Kinetics-400, GFLOPs por vista × número de vistas:

| Modelo | Flujo | Pretrain | top-1 | top-5 | GFLOPs × vistas |
| --- | --- | --- | --- | --- | --- |
| [I3D](/papers/i3d-carreira-2017) | | ImageNet | 72.1 | 90.3 | 108 × N/A |
| Two-Stream I3D | sí | ImageNet | 75.7 | 92.0 | 216 × N/A |
| [S3D-G](/papers/s3d-xie-2018) | sí | ImageNet | 77.2 | 93.0 | 143 × N/A |
| Nonlocal R101 (SOTA previo) | | ImageNet | 77.7 | 93.3 | 359 × 30 |
| I3D | sí | — | 71.6 | 90.0 | 216 × N/A |
| [R(2+1)D](/papers/r2plus1d-tran-2018) | | — | 72.0 | 90.0 | 152 × 115 |
| R(2+1)D + flujo (mejor previo sin ImageNet) | sí | — | 73.9 | 90.9 | 304 × 115 |
| **SlowFast 4×16, R50** | | — | **75.6** | 92.1 | **36.1** × 30 |
| **SlowFast 8×8, R50** | | — | 77.0 | 92.6 | 65.7 × 30 |
| **SlowFast 8×8, R101** | | — | 77.9 | 93.2 | 106 × 30 |
| **SlowFast 16×8, R101** | | — | 78.9 | 93.5 | 213 × 30 |
| **SlowFast 16×8, R101+NL** | | — | **79.8** | **93.9** | 234 × 30 |

La comparación con I3D es la que importa para el arco de la clase. Contra **I3D RGB** (72.1%, 108 GFLOPs, con ImageNet), SlowFast 4×16 R50 gana **+3.5 puntos** con **3× menos cómputo por vista**. Contra **Two-Stream I3D** (75.7%, 216 GFLOPs, con ImageNet y con flujo), empata usando **6× menos FLOPs por vista**, sin flujo y sin ImageNet — y 8×8 R50 lo supera con 77.0% a 65.7 GFLOPs, aún 3.3× más barato. Contra el SOTA **Nonlocal R101** (77.7%, 359 GFLOPs), el mejor SlowFast gana **+2.1 puntos** con 1.5× menos cómputo. Además, varios trabajos previos usan más de 100 vistas en inferencia (R(2+1)D 115, ARTNet 250); SlowFast usa **30** y le basta, porque la vía Fast ya cubre densamente el intervalo *dentro* de cada clip: la densidad temporal se paga una vez en la red, no repetidamente al muestrear clips.

| $\beta$ | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| Slow-only | 72.6 | 90.3 | 27.3 |
| 1/4 | 75.6 | 91.7 | 54.5 |
| **1/6** | **75.8** | 92.0 | 41.8 |
| **1/8** (por defecto) | 75.6 | **92.1** | 36.1 |
| 1/12 | 75.2 | 91.8 | 32.8 |
| 1/16 | 75.1 | 91.7 | 30.6 |
| 1/32 | 74.2 | 91.3 | 28.6 |

Lo notable es la robustez: **todo** el rango de $1/32$ a $1/4$ mejora sobre Slow-only. En el extremo barato, $\beta=1/32$ agrega apenas 1.3 GFLOPs y aporta +1.6 puntos. En el extremo caro la asimetría es elocuente: $\beta=1/4$ cuesta 54.5 GFLOPs (51% más que 1/8) para la **misma** top-1 y un top-5 *peor*. Más canales en la vía rápida son desperdicio: la validación empírica de que **no debe** modelar apariencia. Sobre $\alpha$, el paper insiste en que su presencia *"está en el centro del concepto SlowFast"*, porque es lo que fuerza la especialización de las subredes; con $\alpha=8$ la ganancia en AVA sobre Slow-only es de +5.2 mAP.

Variantes de entrada a la vía Fast, pensadas para debilitar su capacidad espacial de otras maneras:

| Entrada a la vía Fast | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| **RGB** ($\beta=1/8$, por defecto) | **75.6** | **92.1** | 36.1 |
| RGB media resolución ($112^2$), $\beta=1/4$ | 74.7 | 91.8 | 34.4 |
| **Escala de grises** | 75.5 | 91.9 | 34.1 |
| Time difference (actual − anterior) | 74.5 | 91.6 | 34.2 |
| **[Flujo óptico](/fundamentos/flujo-optico)** | 73.8 | 91.3 | 35.1 |

Todas superan el 72.6% de Slow-only: la evidencia más fuerte de que lo que la vía Fast representa es movimiento, no apariencia. Dos gotchas. La **escala de grises está a 0.1 puntos del RGB** y ahorra ~5% de FLOPs: la vía rápida **no necesita color**, consistente con la insensibilidad al color de las células M. Y **el flujo óptico es la peor de las cinco opciones** (73.8%). Ese es el resultado clave para la narrativa de la clase: no es que el flujo sea caro de calcular y por eso lo evitemos, es que **el RGB a alta tasa temporal es mejor representación de movimiento que el flujo precomputado**, porque el flujo, diseñado a mano, descarta información que la red podría usar.

Y el control que cierra el arco, sobre la misma 3D ResNet-50 en tres regímenes:

| Modelo | Pretrain | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- | --- |
| 3D R-50, receta de Nonlocal | ImageNet | 73.4 | 90.9 | 36.7 |
| 3D R-50, receta de Nonlocal, desde cero | — | 69.4 | 88.6 | 36.7 |
| 3D R-50, **receta de los autores**, desde cero | — | **73.5** | 90.8 | 36.7 |

La receta antigua desde cero pierde **4.0 puntos**; la de SlowFast desde cero **iguala** al pre-entrenamiento en ImageNet (73.5 contra 73.4), con SGD sincronizado de batch 1024 sobre 128 GPUs, warm-up lineal de 8k iteraciones para tolerar $\eta=1.6$, schedule cosenoidal de medio período y 256 epochs. **La brecha de "entrenar desde cero" era brecha de receta de optimización, no de datos.** Y al probar ImageNet sobre SlowFast la diferencia fue de **±0.3%**: dejó de aportar.

> Aquí está la ironía que cierra el arco de la clase. El argumento central de [I3D](/papers/i3d-carreira-2017) es que heredar ImageNet vía el inflado era **indispensable** para las redes 3D profundas, y la clase lo lista como su ventaja principal. Dos años después SlowFast entrena **desde inicialización aleatoria** y alcanza el estado del arte. Lo que cambió no fue la arquitectura sino la **madurez de Kinetics como "ImageNet del video"** (~240k videos en K400, ~392k en K600, contra los ~10k de UCF-101 y HMDB-51), más una receta de optimización que en 2017 no era práctica estándar. El inflado fue una solución brillante a un problema de escasez de datos y recetas inmaduras, no una verdad arquitectónica permanente.

**AVA** (detección espacio-temporal, 60 clases, Faster R-CNN con SlowFast como backbone). Ablation limpio: Slow-only R-50 4×16 = **19.0 mAP** contra SlowFast = **24.2 mAP**, un **+5.2 mAP (28% relativo)** atribuible únicamente a la idea SlowFast. El contraste con el flujo es demoledor: añadirlo daba **+1.1 mAP** a I3D y **+1.7** a ATR, duplicando el cómputo. Contra el estado del arte de v2.1, SlowFast R101 8×8 con solo Kinetics-400 logra **26.3 mAP** frente a 21.7 del mejor modelo único previo, y **28.2 mAP** en su mejor configuración; en v2.2 alcanza 30.7 con test multiescala, y un ensamble de 7 modelos logró **34.3 mAP** en el test server, **ganando el AVA action detection challenge 2019**. Mejora en **57 de 60 categorías**, con máximos en "hand clap" +27.7 AP, "swim" +27.4, "run/jog" +18.8, "dance" +15.9, "eat" +12.5; empeora solo en tres, cuasi-estáticas y marginalmente: "answer phone" (−0.1), "lie/sleep" (−0.2), "shoot" (−0.4).

**Charades** (multi-etiqueta, ~30 s por actividad) es el test de estrés de largo alcance: SlowFast 16×8 R101 sube de **39.0 mAP** de su propio baseline Slow-only a **42.1**, +0.4 con bloques non-local y **45.2 mAP** con pre-entrenamiento en Kinetics-600 — contra 39.7 del mejor previo (STRG R101+NL) a **2.7× menos costo** (234 contra 630 GFLOPs por vista). En **Kinetics-600** los SlowFast van de 78.8% a **81.8%**, contra 71.9% de I3D.

---

## Limitaciones

- **El costo absoluto sigue siendo alto.** Ser más eficiente *por vista* no lo hace barato: el mejor modelo cuesta $234\times30 = 7020$ GFLOPs por video, y hasta el "económico" 4×16 R50 son 1083. Está fuera de alcance para inferencia en tiempo real en hardware modesto; el propio Feichtenhofer atacaría esto al año siguiente con **X3D** (CVPR 2020).
- **El protocolo de 30 vistas optimiza leaderboard, no despliegue.** El paper es honesto y reporta el factor ×30 explícitamente —aporte metodológico valioso, porque ese costo "había sido largamente ignorado"—, pero un sistema en producción usaría muchas menos vistas y perdería precisión de forma no cuantificada aquí.
- **La ventana temporal es de segundos, no de minutos.** Por defecto ve 64 fotogramas (~2.13 s a 30 fps) y la variante $16\times8$ llega a 128 (~4.3 s). El largo alcance se recupera solo promediando 10 clips, una agregación sin memoria y sin orden: la tercera desventaja que la clase atribuye al Two-Stream queda **mejorada, no resuelta**.
- **La analogía biológica no se valida.** El paper mismo la califica de tosca y prematura, y su único punto de contacto empírico (insensibilidad al color) es débil como evidencia.
- **La fusión es unidireccional y de topología fija.** Cuatro conexiones prefijadas; la bidireccional se reporta como "resultados similares" sin más análisis. No se exploran más de dos vías (¿tres velocidades?) ni $\alpha$ o $\beta$ variables por stage.
- **Los ablations viven en una sola configuración.** Casi todos usan SlowFast 4×16 R-50; la transferencia de esos óptimos a configuraciones mayores se asume más que se verifica.

---

## Por qué importa hoy

SlowFast se convirtió en el **baseline obligado en detección de acciones**, muy especialmente en AVA. Ganar el challenge 2019 con un margen tan amplio (26.3 contra 21.7 mAP en condiciones comparables) hizo que durante años cualquier trabajo en localización espacio-temporal tuviera que reportar frente a él, y el patrón "backbone SlowFast + Faster R-CNN con RoI 3D y RoIAlign" quedó como referencia de facto. **PySlowFast** fue tan importante como el paper: FAIR liberó no solo el modelo sino un framework completo de investigación en video, con recetas de entrenamiento distribuido, protocolos de evaluación multi-vista y un zoo de modelos pre-entrenados; bajó sustantivamente la barrera de entrada, y luego albergó X3D y MViT. También resolvió la dependencia del flujo óptico de la forma más elegante posible: no con un módulo de flujo aprendido ni con una pérdida auxiliar, sino **eliminando el problema**. Después de SlowFast, calcular TV-L1 sobre datasets de 240k videos dejó de ser parte del pipeline estándar.

Sobre la transición a los **video transformers**: el mismo grupo publicó **MViT (Multiscale Vision Transformers, ICCV 2021)**, con dos autores compartidos, y con ello el estado del arte migró de las ConvNets a la atención. Lo que sobrevive de SlowFast en ese salto no es la arquitectura de dos vías sino la idea más profunda: **las jerarquías multiescala**. MViT rechaza el diseño del [Vision Transformer](/fundamentos/vision-transformer) de resolución y ancho constantes y construye una pirámide donde la resolución espacio-temporal se reduce mientras la dimensión de canales se expande con la profundidad — exactamente el principio de "resolución alta con pocos canales / resolución baja con muchos canales" que SlowFast había explotado en paralelo sobre el eje temporal, ahora reorganizado sobre el eje de profundidad. Leído retrospectivamente, SlowFast es el argumento de que **el presupuesto de resolución temporal y el de capacidad de canales son recursos intercambiables**.

Dos lecciones trascienden el video. Primera: cuando dos fuentes de señal tienen escalas de variación distintas, conviene procesarlas a resoluciones y presupuestos distintos en vez de forzar un tratamiento uniforme — la vía que atiende la señal rápida necesita **frecuencia, no ancho**, y la que atiende la estable necesita **ancho, no frecuencia**. Segunda: una rama que rinde mal por sí sola (51.7%) puede aportar muchísimo en conjunto, porque su valor está en la complementariedad. Evaluar componentes solo por su métrica individual lleva a descartar exactamente las piezas que más aportan al ensamble, un error que reaparece en cualquier sistema con múltiples señales de entrada, sea video, texto o registros estructurados.

---

## Notas y enlaces

- **En el curso:** la [Clase 38](/clases/clase-38) recorre la evolución CNN2D → CNN2D+RNN → [Two-Stream](/papers/two-stream-simonyan-2014) → C3D → [I3D](/papers/i3d-carreira-2017) enumerando desventajas de cada familia; SlowFast es el cierre natural porque ataca de frente la lista de las dos últimas. La [teoría](/clases/clase-38/teoria) sitúa el paper en esa genealogía y la [profundización](/clases/clase-38/profundizacion) trata la matemática del inflado y de los kernels espacio-temporales.
- **El cambio de eje:** la clase presenta el Two-Stream como separación por **modalidad**. SlowFast conserva la intuición de dos corrientes especializadas pero cambia el eje a la **resolución temporal**, con tres consecuencias: desaparece un preprocesamiento costoso y no diferenciable, el modelo pasa a ser end-to-end, y la segunda corriente se vuelve casi gratis porque, al dedicarse solo al movimiento, no necesita capacidad para apariencia — algo que el Two-Stream, con el mismo backbone en ambos streams, nunca aprovechó.
- **Fundamentos:** [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones), [flujo óptico](/fundamentos/flujo-optico) (qué se está reemplazando y por qué), [inflado de convoluciones](/fundamentos/inflado-de-convoluciones) y [Vision Transformer](/fundamentos/vision-transformer) (la vía por la que llegan MViT y sucesores).
- **Linaje:** [Two-Stream (2014)](/papers/two-stream-simonyan-2014) plantea la separación original; [Two-Stream Fusion (2016)](/papers/two-stream-fusion-feichtenhofer-2016), del mismo primer autor, aporta las conexiones laterales que SlowFast reutiliza; [I3D (2017)](/papers/i3d-carreira-2017) y [Kinetics (2017)](/papers/kinetics-kay-2017) establecen el paradigma que SlowFast relativiza; [R(2+1)D (2018)](/papers/r2plus1d-tran-2018) y [S3D (2018)](/papers/s3d-xie-2018) son los otros dos ataques contemporáneos a la simetría de los kernels cúbicos, por factorización espacio/tiempo dentro de cada capa en vez de por framerate. El backbone es una [ResNet](/papers/resnet-he-2015) 3D, y la [Clase 36](/clases/clase-36) provee los datasets y el marco de evaluación del análisis de video.
- **Código:** [github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) (PySlowFast). Datasets evaluados: Kinetics-400, Kinetics-600, Charades y AVA.
