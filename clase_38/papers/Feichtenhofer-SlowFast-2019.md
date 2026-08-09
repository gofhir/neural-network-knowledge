# SlowFast Networks for Video Recognition — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *SlowFast Networks for Video Recognition*.
- **Autores:** Christoph Feichtenhofer, Haoqi Fan, **Jitendra Malik** y **Kaiming He**, los cuatro en **Facebook AI Research (FAIR)**.
- **Venue:** *IEEE/CVF International Conference on Computer Vision* (**ICCV 2019**).
- **Preprint:** arXiv:1812.03982v3 (29 oct 2019), [arxiv.org/abs/1812.03982](https://arxiv.org/abs/1812.03982).
- **Código:** [github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) (más tarde renombrado **PySlowFast**).
- **Datasets evaluados:** Kinetics-400, Kinetics-600, Charades y AVA.

El paper propone una arquitectura de dos vías (*pathways*) que procesan **el mismo clip crudo a dos velocidades temporales distintas**: una vía **Slow** de bajo framerate y alta capacidad de canales, encargada de la semántica espacial, y una vía **Fast** de alto framerate y **baja** capacidad de canales, encargada del movimiento a resolución temporal fina. Ambas se fusionan por **conexiones laterales** unidireccionales (Fast → Slow) a lo largo de la jerarquía. La vía Fast es deliberadamente barata: consume aproximadamente el **20% del cómputo total**.

Las cifras clave, todas obtenidas **entrenando desde cero, sin pre-entrenamiento en ImageNet**:

- **Kinetics-400:** el mejor modelo (SlowFast 16×8, R101+NL) alcanza **79.8% top-1 / 93.9% top-5** a **234 GFLOPs × 30 vistas**, lo que supera al estado del arte previo (Nonlocal R101, 77.7%) en **2.1 puntos** usando **menos cómputo por vista** (234 vs 359 GFLOPs). Frente al mejor resultado previo *sin* ImageNet (R(2+1)D con flujo, 73.9%), la mejora absoluta es de **5.9 puntos**. La variante más económica, SlowFast 4×16 R50, logra **75.6%** con apenas **36.1 GFLOPs** por vista.
- **Kinetics-600:** **81.8% top-1 / 95.1% top-5** (16×8, R101+NL), contra 79.0% de la entrada ganadora del ActivityNet Challenge 2018.
- **Charades** (multi-etiqueta, actividades de ~30 s): **45.2 mAP** con pre-entrenamiento en Kinetics-600, contra 39.7 mAP del mejor previo (STRG R101+NL) y a **menor costo** (234×30 vs 630×30 GFLOPs).
- **AVA** (detección espacio-temporal de acciones): **28.2 mAP** en validación de v2.1, **5.6 mAP** por encima del mejor resultado previo comparable (21.7 de ATR). En v2.2, **30.7 mAP** con test multiescala, y un ensamble de 7 modelos alcanzó **34.3 mAP** en el test server, **ganando el AVA action detection challenge 2019**.

El eslogan de la conclusión resume la tesis: *"el eje temporal es una dimensión especial"*.

## 2. Contexto: la asimetría del video que nadie estaba explotando

### 2.1. El argumento de la isotropía

En imágenes $I(x,y)$ es costumbre tratar $x$ e $y$ **simétricamente**: mismos kernels cuadrados, mismos strides. Eso está **justificado** porque las imágenes naturales son, a primera aproximación, **isótropas** (todas las orientaciones son igualmente probables) e invariantes al desplazamiento.

Pero en señales de video $I(x,y,t)$ la premisa se rompe. El movimiento es la contraparte espacio-temporal de la orientación, y **no todas las orientaciones espacio-temporales son igualmente probables**: los movimientos lentos son más probables que los rápidos —la mayor parte del mundo que vemos está en reposo en un instante dado—, algo que la literatura de percepción explota en modelos bayesianos del movimiento humano. El paper cita el problema de la apertura como evidencia: si vemos un borde en movimiento aislado lo percibimos moviéndose perpendicular a sí mismo, aunque en principio pudiera tener una componente tangencial arbitraria; ese percepto es **racional si el prior favorece movimientos lentos**.

La conclusión de diseño es directa: si las orientaciones espacio-temporales **no** son equiprobables, **no hay razón para tratar espacio y tiempo simétricamente**, que es exactamente lo que hacen implícitamente las convoluciones espacio-temporales con kernels cúbicos ($N \times N \times N$, el inflado de C3D/I3D). Hay que **factorizar** la arquitectura para tratar estructuras espaciales y eventos temporales por separado.

La justificación en el dominio del reconocimiento: **la semántica categórica evoluciona lentamente**. Unas manos que se agitan no dejan de ser "manos" durante la acción; una persona sigue siendo "persona" aunque pase de caminar a correr, y colores, texturas e iluminación también se refrescan lento. En cambio, **el movimiento puede evolucionar mucho más rápido que la identidad de su sujeto**: aplaudir, agitar, sacudir, saltar. Para eso conviene alta resolución temporal.

### 2.2. La motivación biológica: células M y P (y hasta dónde llega la analogía)

El paper dice que su método está **parcialmente inspirado** en estudios sobre las **células ganglionares de la retina** de primates, y —esto importa para ser justos con el texto— añade explícitamente que *"admitidamente la analogía es tosca y prematura"* (*"rough and premature"*). La biología es **motivación e ilustración**, no derivación ni evidencia.

Los hechos biológicos citados:

| Tipo de célula | Proporción | Frecuencia temporal | Sensibilidad |
| --- | --- | --- | --- |
| **Parvocelular (P)** | ~80% | Baja (responde lento a estímulos) | Detalle espacial fino y **color** |
| **Magnocelular (M)** | ~15–20% | **Alta** (responde a cambios temporales rápidos) | **No** sensible a detalle espacial ni a color |

Y las tres correspondencias que el paper afirma como analogía:

1. El modelo tiene dos vías que trabajan separadamente a resoluciones temporales **baja** y **alta**.
2. La vía Fast está diseñada para capturar movimiento rápido con **menos detalle espacial**, análogo a las células M.
3. La vía Fast es **liviana**, similar a la pequeña proporción de células M.

Lo que el paper **no** afirma: que la arquitectura sea un modelo del sistema visual, ni que la separación M/P se corresponda funcionalmente con lo que aprenden las vías. El uso de *pathway* en lugar de *stream* es una elección retórica deliberada: los autores dicen que SlowFast *puede describirse como una arquitectura de un solo stream operando a dos framerates*, y que usan "pathway" solo por la resonancia biológica.

El único punto donde biología y experimento se tocan de forma no trivial es el ablation en **escala de grises** (Tabla 5c): la vía Fast en gris rinde 75.5% contra 75.6% en RGB. Los autores señalan que esto es *consistente* con la insensibilidad al color de las células M. Coincidencia elegante, pero consistencia, no confirmación.

## 3. Contribución central

SlowFast es una arquitectura genérica de **dos vías con distinto framerate y distinta capacidad, unidas por conexiones laterales**. Formalmente hay tres hiperparámetros que la definen:

- $\tau$: el **stride temporal** de la vía Slow sobre los fotogramas crudos. Valor típico $\tau = 16$.
- $\alpha > 1$: la **razón de framerate** entre Fast y Slow. La vía Fast usa stride $\tau/\alpha$. Valor típico $\alpha = 8$.
- $\beta < 1$: la **razón de capacidad de canales** de Fast respecto de Slow. Valor típico $\beta = 1/8$.

Si la vía Slow muestrea $T$ fotogramas, el clip crudo tiene $T \times \tau$ fotogramas y la vía Fast muestrea $\alpha T$ fotogramas del **mismo** clip crudo. Denotando la forma del tensor de features de Slow como $\{T, S^2, C\}$, la de Fast es $\{\alpha T, S^2, \beta C\}$.

Lo que distingue esto del **two-stream** clásico de Simonyan y Zisserman, argumentado explícitamente en Related Work:

1. Two-Stream **no explora velocidades temporales distintas**, que es *el* concepto de SlowFast.
2. Two-Stream usa **el mismo backbone en ambos streams**; la vía Fast es **más liviana** (por $\beta$).
3. SlowFast **no calcula flujo óptico**, por lo que se aprende **end-to-end desde los datos crudos**. El paper es duro con el flujo: es "metodológicamente insatisfactorio" porque es una representación **diseñada a mano** y los métodos two-stream "a menudo no se aprenden end-to-end junto con el flujo".

La separación deja de ser por **modalidad** (RGB vs. flujo precomputado) y pasa a ser por **resolución temporal**.

## 4. El pathway Slow: ver poco, pero entender bien

La vía Slow puede ser cualquier modelo convolucional que opere sobre un clip como volumen espacio-temporal. Su rasgo definitorio es un **stride temporal grande $\tau$**: procesa solo **1 de cada $\tau$** fotogramas. Con $\tau = 16$ y video a 30 fps, la velocidad de refresco es de aproximadamente **2 fotogramas muestreados por segundo**: en la instanciación por defecto la vía Slow ve $T = 4$ fotogramas de un clip crudo de **64** (~2.13 s). Cuatro instantáneas.

El gotcha más contraintuitivo del paper: **la vía Slow no hace downsampling temporal**. No hay pooling temporal ni convoluciones con stride temporal. Es consecuencia lógica del muestreo agresivo: si solo quedan 4 fotogramas, **comprimirlos más sería destructivo**. El paper lo dice sin rodeos: *"optamos por no realizar downsampling temporal en esta instanciación, ya que hacerlo sería perjudicial cuando el stride de entrada es grande"*.

En compensación, la vía Slow concentra **toda la capacidad de canales**: es un ResNet 3D con stride temporal y el ancho estándar de ResNet-50 (64 → 256 → 512 → 1024 → 2048). Con 32.4M de parámetros y 27.3 GFLOPs, sola alcanza **72.6% top-1** en Kinetics-400 — un baseline fuerte, lo que hace más significativas las ganancias que aporta la vía Fast.

## 5. El pathway Fast: delgado, ciego al detalle, pero temporalmente fiel

En paralelo corre la vía Fast, con tres propiedades de diseño explícitas.

### 5.1. Alto framerate

Opera con stride temporal **$\tau/\alpha$**. Con $\tau=16$ y $\alpha=8$, el stride es **2**: muestrea $\alpha T = 32$ fotogramas del mismo clip de 64, **8× más denso** que Slow. El paper insiste en que la presencia de $\alpha$ *"está en el centro del concepto SlowFast"*: es lo que indica que las dos vías trabajan a velocidades distintas y lo que **fuerza la especialización** de las dos subredes.

### 5.2. Features de alta resolución temporal en toda la jerarquía

No basta con alta resolución de entrada. La vía Fast **no tiene ninguna capa de downsampling temporal** —ni pooling ni convoluciones con stride temporal— **en toda la red**, hasta el global pooling final. Sus tensores **siempre tienen $\alpha T = 32$ fotogramas**: en conv1, en res2, en res5, en todas. La fidelidad temporal se mantiene de punta a punta.

Correlativamente usa **convoluciones temporales no degeneradas** (kernel temporal $> 1$) en **cada bloque**: $3 \times 1^2$ en el primer conv de cada bottleneck de res2 a res5, más un conv1 de $5 \times 7^2$. La lógica es coherente: como esta vía sí conserva resolución temporal fina, **vale la pena** poner filtros temporales para capturar movimiento detallado.

### 5.3. Baja capacidad de canales: el corazón del truco

La vía Fast usa una fracción $\beta$ de los canales de Slow, típicamente $\beta = 1/8$: 8 canales en conv1 en vez de 64, 256 en res5 en vez de 2048.

El argumento del costo es aritmético y contundente: **los FLOPs de una capa convolucional son cuadráticos en la razón de escalado de canales**. Reducir canales a $1/8$ reduce el cómputo por capa en torno a $1/64$, lo que da margen de sobra para pagar el $8\times$ de fotogramas. Resultado neto: la vía Fast sola cuesta **6.4 GFLOPs** contra 27.3 de Slow, y tiene **0.53M de parámetros** contra 32.4M — **~1.6% de los parámetros** del modelo. En la instanciación completa de 36.1 GFLOPs esos 6.4 son ~18%; contando también las conexiones laterales, el incremento total sobre Slow-only es 8.8 GFLOPs (36.1 − 27.3), ~24% del total. El paper redondea a **"~20% del cómputo total"**, y no es casualidad que coincida con el ~15–20% de células M.

### 5.4. Baja capacidad espacial (por diseño y por omisión)

La baja capacidad de canales se interpreta además como una **capacidad más débil para representar semántica espacial**. Técnicamente la vía Fast no recibe **ningún tratamiento especial** en la dimensión espacial: usa los mismos kernels y strides espaciales que Slow. Su debilidad espacial es consecuencia de tener menos canales, no una decisión arquitectónica separada.

El tradeoff se explicita: *"es un tradeoff deseable para la vía Fast debilitar su capacidad de modelado espacial mientras fortalece su capacidad de modelado temporal"*. La apariencia **la aporta la otra vía**, y hacerlo así es menos redundante que duplicarla.

Motivados por esa interpretación, los autores exploran otras formas de debilitar la capacidad espacial (Tabla 5c): media resolución ($112 \times 112$), escala de grises, "time difference" (restar el fotograma anterior) y flujo óptico. Todas funcionan y todas superan a Slow-only. Es la evidencia más fuerte de la tesis: **lo que la vía Fast representa es movimiento, no apariencia**. Y la prueba de que es complementaria y no redundante es que **sola alcanza apenas 51.7% top-1** —un modelo mediocre— y sin embargo aporta hasta **+3.0 puntos** a la vía Slow.

## 6. Las conexiones laterales: reconciliar dos ejes temporales

Sin fusión, las dos vías se ignoran. El paper las une con **conexiones laterales**, técnica que Feichtenhofer ya había usado para fusionar redes two-stream basadas en flujo y que en detección de objetos se popularizó con las Feature Pyramid Networks.

**Dónde se insertan:** una por cada "stage" del ResNet, concretamente **justo después de `pool1`, `res2`, `res3` y `res4`** — cuatro conexiones. No hay una después de res5 porque ahí ya se hace el global pooling y la concatenación final.

**Dirección:** **unidireccional, de Fast hacia Slow**. Los autores probaron fusión bidireccional y encontraron resultados **similares**, así que se quedaron con la más simple.

**El problema:** las formas no coinciden. Slow tiene $\{T, S^2, C\}$ y Fast tiene $\{\alpha T, S^2, \beta C\}$. Hay que transformar Fast para poder sumar o concatenar. Las tres estrategias evaluadas:

| Estrategia | Transformación de $\{\alpha T, S^2, \beta C\}$ | Con $\alpha=8$, $\beta=1/8$, en `pool1` (Fast: $32 \times 56^2 \times 8$) | Comentario |
| --- | --- | --- | --- |
| **(i) Time-to-channel (TtoC)** | $\{T, S^2, \alpha\beta C\}$ | $32 \times 56^2 \times 8 \to 4 \times 56^2 \times 64$ | Reshape + transpose: empaqueta los $\alpha$ fotogramas dentro de los canales de uno. Con $\alpha\beta = 1$, iguala exactamente los canales de Slow, así que **permite fusión por suma**. |
| **(ii) Time-strided sampling (T-sample)** | $\{T, S^2, \beta C\}$ | $32 \times 56^2 \times 8 \to 4 \times 56^2 \times 8$ | Simplemente toma 1 de cada $\alpha$ fotogramas. **Descarta 7/8 de la información temporal.** |
| **(iii) Time-strided convolution (T-conv)** | Conv 3D de kernel $5 \times 1^2$, $2\beta C$ canales de salida, stride $= \alpha$ | $32 \times 56^2 \times 8 \to 4 \times 56^2 \times 16$ | **Aprende** cómo agregar la ventana temporal. Kernel temporal 5 con stride 8: solapa parcialmente. |

La salida se fusiona en la vía Slow por **suma** o **concatenación**.

**Resultados del ablation (Tabla 5a, Kinetics-400, SlowFast 4×16 R-50):**

| Variante | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| Slow-only | 72.6 | 90.3 | 27.3 |
| Fast-only | 51.7 | 78.5 | 6.4 |
| SlowFast **sin** conexión lateral (solo concatena salidas finales) | 73.5 | 90.3 | 34.2 |
| SlowFast **TtoC, sum** | 74.5 | 91.3 | 34.2 |
| SlowFast **TtoC, concat** | 74.3 | 91.0 | 39.8 |
| SlowFast **T-sample** | 75.4 | 91.8 | 34.9 |
| SlowFast **T-conv** (por defecto) | **75.6** | **92.1** | 36.1 |

Lecturas obligatorias de esta tabla:

- La fusión ingenua (solo concatenar las salidas finales, sin laterales) da **73.5%**, apenas **+0.9** sobre Slow-only. Sin conexiones laterales en la jerarquía, la idea *no funciona*.
- **T-conv gana** y es la elección por defecto: **+3.0 puntos** sobre Slow-only (75.6 vs 72.6) por **+8.8 GFLOPs**.
- T-sample queda a solo 0.2 puntos de T-conv y es más barata. Gotcha: **descartar 7 de cada 8 fotogramas en la fusión sigue funcionando muy bien**, lo que sugiere que lo importante ocurre *dentro* de la vía Fast y que la fusión solo necesita inyectar una señal, no transportar todo el detalle.
- TtoC por suma le gana a TtoC por concatenación (74.5 vs 74.3) **y** cuesta 5.6 GFLOPs menos. Empaquetar tiempo en canales es la peor de las tres.

Tras res5 se hace **global average pooling en cada vía por separado** y los dos vectores se **concatenan** como entrada al clasificador fully-connected.

## 7. La instanciación concreta: SlowFast sobre ResNet-50

Esta es la Tabla 1 del paper, con $\alpha = 8$, $\beta = 1/8$, $\tau = 16$, backbone ResNet-50, entrada de entrenamiento $224 \times 224$. Notación de kernels: $\{T \times S^2, C\}$; strides como $\{$stride temporal, stride espacial$^2\}$. Los filtros temporales **no degenerados** (kernel temporal $>1$) van marcados con $^{\dagger}$.

| Stage | Slow pathway | Fast pathway | Output $T \times S^2$ |
| --- | --- | --- | --- |
| raw clip | — | — | $64 \times 224^2$ |
| data layer | stride $16, 1^2$ | stride $2, 1^2$ | Slow: $4 \times 224^2$ / Fast: $32 \times 224^2$ |
| conv1 | $1 \times 7^2, 64$ — stride $1, 2^2$ | $5 \times 7^2, 8$ $^{\dagger}$ — stride $1, 2^2$ | Slow: $4 \times 112^2$ / Fast: $32 \times 112^2$ |
| pool1 | $1 \times 3^2$ max — stride $1, 2^2$ | $1 \times 3^2$ max — stride $1, 2^2$ | Slow: $4 \times 56^2$ / Fast: $32 \times 56^2$ |
| res2 | [ $1\times1^2, 64$ ; $1\times3^2, 64$ ; $1\times1^2, 256$ ] $\times 3$ | [ $3\times1^2, 8$ $^{\dagger}$ ; $1\times3^2, 8$ ; $1\times1^2, 32$ ] $\times 3$ | Slow: $4 \times 56^2$ / Fast: $32 \times 56^2$ |
| res3 | [ $1\times1^2, 128$ ; $1\times3^2, 128$ ; $1\times1^2, 512$ ] $\times 4$ | [ $3\times1^2, 16$ $^{\dagger}$ ; $1\times3^2, 16$ ; $1\times1^2, 64$ ] $\times 4$ | Slow: $4 \times 28^2$ / Fast: $32 \times 28^2$ |
| res4 | [ $3\times1^2, 256$ $^{\dagger}$ ; $1\times3^2, 256$ ; $1\times1^2, 1024$ ] $\times 6$ | [ $3\times1^2, 32$ $^{\dagger}$ ; $1\times3^2, 32$ ; $1\times1^2, 128$ ] $\times 6$ | Slow: $4 \times 14^2$ / Fast: $32 \times 14^2$ |
| res5 | [ $3\times1^2, 512$ $^{\dagger}$ ; $1\times3^2, 512$ ; $1\times1^2, 2048$ ] $\times 3$ | [ $3\times1^2, 64$ $^{\dagger}$ ; $1\times3^2, 64$ ; $1\times1^2, 256$ ] $\times 3$ | Slow: $4 \times 7^2$ / Fast: $32 \times 7^2$ |
| — | global average pool, concatenate, fc | | # clases |

Dos observaciones que vale la pena internalizar:

**La columna temporal es constante en ambas vías**: 32 en Fast y 4 en Slow, en todos los stages. **Ninguna de las dos hace downsampling temporal.** Toda la reducción es espacial ($224 \to 112 \to 56 \to 28 \to 14 \to 7$). El eje temporal es intocable en ambas, por razones opuestas: en Slow porque ya queda muy poco, en Fast porque es precisamente su especialidad.

**El hallazgo contraintuitivo: la vía Slow es 2D en las capas tempranas.** A diferencia de C3D o I3D —que inflan *todos* los filtros a cúbicos—, la vía Slow usa convoluciones temporales no degeneradas **solo en res4 y res5**. De conv1 a res3, **todos sus filtros son esencialmente kernels 2D** ($1 \times 7^2$, $1 \times 1^2$, $1 \times 3^2$).

No es una simplificación por economía, sino una decisión motivada por observación experimental: **usar convoluciones temporales en capas tempranas degrada la precisión**. El argumento de los autores es geométrico: cuando los objetos se mueven rápido y el stride temporal es grande, **hay poca correlación dentro de un campo receptivo temporal a menos que el campo receptivo espacial sea suficientemente grande** (es decir, a menos que estemos en capas profundas). Con $\tau=16$, entre dos fotogramas consecutivos de la vía Slow pasan ~0.53 s: un objeto en movimiento se desplazó muchísimo más que los 7 píxeles del campo receptivo de conv1. Un filtro cúbico allí intentaría correlacionar parches espaciales que ya no se solapan en absoluto — puro ruido. En res4, con campo receptivo amplio, el objeto **sí** sigue dentro del campo y el filtro temporal tiene algo real que medir.

Corolario de diseño: **el ritmo con que se introduce el modelado temporal debe escalar con el campo receptivo espacial**. Kernels cúbicos uniformes violan esa condición en las capas bajas. La vía Fast sí tiene convoluciones temporales en cada bloque *porque* su stride temporal es 8× menor (~0.067 s entre fotogramas), y ahí la correlación local existe.

## 8. Setup experimental

### 8.1. Datasets

| Dataset | Train | Val | Clases | Tarea / métrica |
| --- | --- | --- | --- | --- |
| **Kinetics-400** | ~240k videos | 20k | 400 | Clasificación, top-1/top-5 |
| **Kinetics-600** | ~392k videos | 30k | 600 | Clasificación, top-1/top-5 |
| **Charades** | ~9.8k videos | 1.8k | 157 | Multi-etiqueta, actividades de ~30 s, mAP |
| **AVA v2.1** | 211k segmentos | 57k | 60 (evaluadas) | Detección espacio-temporal, mAP con IoU a nivel de frame de 0.5 |

Nota de rigor: el conjunto de validación de Kinetics-600 **se solapa con el train de Kinetics-400**, por lo que los autores **no pre-entrenan en Kinetics-400** para los experimentos de Kinetics-600. AVA proviene de **437 películas**, con etiquetas espacio-temporales para **un fotograma por segundo**, cada persona con bounding box y posiblemente múltiples acciones; la dificultad está en la **detección de la acción**, no en localizar al actor.

### 8.2. Entrenamiento desde cero — el detalle a destacar

**Los modelos en Kinetics se entrenan desde inicialización aleatoria ("from scratch"), sin ImageNet ni ningún pre-entrenamiento.** Es un quiebre frontal con la tesis central de I3D, que argumentaba que heredar pesos de ImageNet vía el inflado era el ingrediente que desbloqueaba las 3D ConvNets profundas. Y no es que SlowFast no *pueda* usar ImageNet: los autores lo probaron y encontraron que rinde **similar (±0.3%)** en ambas variantes. **ImageNet dejó de aportar.**

La receta que lo hace posible (Apéndice):

- **SGD sincronizado sobre 128 GPUs** (receta de large-minibatch de Goyal et al.), **8 clips por GPU**, total **1024**. Inicialización de He et al. (2015), BN con estadísticas **dentro de cada grupo de 8 clips**.
- **Schedule cosenoidal de medio período**: $\eta \cdot 0.5\,[\cos(\tfrac{n}{n_{\max}}\pi) + 1]$, con **$\eta = 1.6$** de base para Kinetics-400.
- **Warm-up lineal en las primeras 8k iteraciones.** Con lr base 1.6 y batch 1024, sin calentamiento el entrenamiento divergiría; el warm-up es lo que hace viable el régimen de batch grande.
- Kinetics-400: **256 epochs** (60k iteraciones) cuando $T \le 4$, **196 epochs** cuando $T > 4$ ("es suficiente entrenar menos cuando un clip tiene más fotogramas"). Momentum 0.9, weight decay $10^{-4}$, dropout 0.5. Kinetics-600: **2×** epochs y $\eta = 0.8$.
- Augmentación: recorte aleatorio de $224 \times 224$ o su volteo horizontal, con lado menor en $[256, 320]$ ($[256, 340]$ para R-101).
- Los modelos $16 \times 8$ se **inicializan desde sus contrapartes $8 \times 8$** con la mitad de los epochs; los que llevan bloques **non-local (NL)** se inicializan desde sus contrapartes sin NL, y NL se aplica **solo sobre los features (fusionados) de Slow en res4**.
- Charades: fine-tuning desde Kinetics, salida **sigmoide por clase**, 24k iteraciones, batch 16, lr base 0.0375 (desde K400) o 0.02 (desde K600), decay escalonado 10×, y **max-pooling temporal** de scores en inferencia.

### 8.3. Evaluación e informe honesto de costo

**Inferencia:** **10 clips** muestreados uniformemente en el eje temporal; para cada uno se escala el lado menor a **256** y se toman **3 crops de $256 \times 256$** como aproximación al testeo fully-convolucional. Total **30 vistas**, promediando scores softmax. Gotcha: la resolución de inferencia es $256^2$, distinta de los $224^2$ de entrenamiento.

Aporte metodológico menor pero valioso: los autores **reportan explícitamente GFLOPs por vista × número de vistas**, porque los papers de la época diferían salvajemente en su estrategia de cropping/clipping y ese costo "había sido largamente ignorado".

### 8.4. AVA: arquitectura de detección

El detector es un **Faster R-CNN** con modificaciones mínimas y SlowFast como backbone. Cambios: **stride espacial de res5 puesto a 1** (en vez de 2) con **dilatación 2** en sus filtros, lo que **duplica la resolución espacial de res5**. Las RoI 2D se extienden a RoI 3D **replicándolas a lo largo del eje temporal**; luego **RoIAlign** espacialmente y **global average pooling** temporalmente; los features se max-poolean y van a un clasificador **sigmoide por clase**.

Las propuestas de región vienen de un detector de personas **off-the-shelf, no entrenado conjuntamente**: Faster R-CNN con backbone **ResNeXt-101-FPN** (Detectron), pre-entrenado en ImageNet y en keypoints humanos de COCO, fine-tuneado en AVA. Alcanza **93.9 AP@50** en validación; con umbral de confianza $>0.8$, las propuestas tienen **recall 91.1%** y **precisión 90.7%**. Entrenamiento: init desde los modelos de Kinetics-400, 14k iteraciones (**68 epochs** para ~211k datos), warm-up lineal en las primeras **1k** iteraciones, weight decay $10^{-7}$, cajas ground-truth como muestras.

## 9. Resultados y ablations

### 9.1. Kinetics-400: comparación con el estado del arte (Tabla 2)

| Modelo | Flujo | Pretrain | top-1 | top-5 | GFLOPs × vistas |
| --- | --- | --- | --- | --- | --- |
| I3D | | ImageNet | 72.1 | 90.3 | 108 × N/A |
| **Two-Stream I3D**  | sí |  ImageNet | 75.7 | 92.0 | **216** × N/A |
| S3D-G  | sí |  ImageNet | 77.2 | 93.0 | 143 × N/A |
| Nonlocal R50 | | ImageNet | 76.5 | 92.6 | 282 × 30 |
| **Nonlocal R101** (SOTA previo) | | ImageNet | 77.7 | 93.3 | **359** × 30 |
| R(2+1)D Flow  | sí |  — | 67.5 | 87.2 | 152 × 115 |
| STC | | — | 68.7 | 88.5 | N/A |
| ARTNet | | — | 69.2 | 88.3 | 23.5 × **250** |
| S3D | | — | 69.4 | 89.1 | 66.4 × N/A |
| ECO | | — | 70.0 | 89.4 | N/A |
| I3D  | sí |  — | 71.6 | 90.0 | 216 × N/A |
| R(2+1)D | | — | 72.0 | 90.0 | 152 × 115 |
| **R(2+1)D Flow** (mejor previo sin ImageNet)  | sí |  — | **73.9** | 90.9 | 304 × 115 |
| **SlowFast 4×16, R50** | | — | 75.6 | 92.1 | **36.1** × 30 |
| **SlowFast 8×8, R50** | | — | 77.0 | 92.6 | 65.7 × 30 |
| **SlowFast 8×8, R101** | | — | 77.9 | 93.2 | 106 × 30 |
| **SlowFast 16×8, R101** | | — | 78.9 | 93.5 | 213 × 30 |
| **SlowFast 16×8, R101+NL** | | — | **79.8** | **93.9** | 234 × 30 |

**Comparación explícita con I3D**, que es lo que más importa para la Clase 38:

- **SlowFast 4×16 R50 (75.6%, 36.1 GFLOPs/vista, sin pretrain) contra I3D RGB (72.1%, 108 GFLOPs, con ImageNet):** **+3.5 puntos** con **3× menos cómputo por vista** y **sin ImageNet**.
- **Contra Two-Stream I3D (75.7%, 216 GFLOPs, con ImageNet y con flujo):** SlowFast 4×16 R50 empata (75.6 vs 75.7) con **6× menos FLOPs por vista**, **sin flujo óptico** y **sin ImageNet**. SlowFast 8×8 R50 lo supera (77.0%) con 65.7 GFLOPs, aún **3.3× más barato**.
- **Contra I3D con flujo sin ImageNet (71.6%, 216 GFLOPs):** **+4.0 puntos** con **6× menos cómputo**.
- **Contra el SOTA Nonlocal R101 (77.7%, 359 GFLOPs, ImageNet):** el mejor SlowFast gana **+2.1 puntos** con **1.5× menos cómputo por vista**.

Y el punto de eficiencia de inferencia: varios trabajos previos usan muestreo temporal extremadamente denso, con **más de 100 vistas** (R(2+1)D usa 115, ARTNet 250). SlowFast usa **30** y le basta, precisamente porque la vía Fast ya cubre densamente el intervalo temporal dentro de cada clip.

La Figura 2 sistematiza el aporte de la vía Fast: para cada configuración $T \times \tau$ y backbone, SlowFast supera consistentemente a su contraparte Slow-only, con ganancias de **+3.3, +3.0, +3.4, +2.1, +2.0 y +1.7 puntos** (de 2×32 R50 hasta 16×8 R101). Doblar los fotogramas de la vía Slow mejora la precisión al **doble de costo**; agregar la vía Fast mejora **más** con un incremento **pequeño**. Hay incluso un punto donde SlowFast da **mayor precisión y menor costo** que un Slow-only temporalmente pesado.

### 9.2. Ablation de $\beta$: la razón de canales (Tabla 5b)

| $\beta$ | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| Slow-only | 72.6 | 90.3 | 27.3 |
| 1/4 | 75.6 | 91.7 | 54.5 |
| **1/6** | **75.8** | 92.0 | 41.8 |
| **1/8** (por defecto) | 75.6 | **92.1** | 36.1 |
| 1/12 | 75.2 | 91.8 | 32.8 |
| 1/16 | 75.1 | 91.7 | 30.6 |
| 1/32 | 74.2 | 91.3 | 28.6 |

Los mejores valores son **1/6 y 1/8**, pero lo notable es la **robustez**: **todo** el rango de $\beta = 1/32$ a $1/4$ mejora sobre Slow-only. El caso extremo es el más elocuente: con $\beta = 1/32$ la vía Fast agrega apenas **1.3 GFLOPs** (~5% relativo) y aporta **+1.6 puntos**. Y notar la asimetría del extremo opuesto: $\beta = 1/4$ cuesta **54.5 GFLOPs** (51% más que 1/8) para la **misma** precisión top-1 de 75.6 y un top-5 *peor*. Más canales en la vía rápida no solo no ayudan: son desperdicio. Esa es la validación empírica de que la vía Fast **no debe** intentar modelar apariencia.

### 9.3. Ablation de entradas espacialmente debilitadas (Tabla 5c)

| Entrada a la vía Fast | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- |
| **RGB** ($\beta=1/8$, por defecto) | **75.6** | **92.1** | 36.1 |
| RGB media resolución ($112^2$), $\beta=1/4$ | 74.7 | 91.8 | 34.4 |
| **Escala de grises** | 75.5 | 91.9 | 34.1 |
| Time difference (frame actual − anterior) | 74.5 | 91.6 | 34.2 |
| **Flujo óptico** | 73.8 | 91.3 | 35.1 |

Todas superan el 72.6% de Slow-only. Dos gotchas:

1. **La escala de grises está a 0.1 puntos del RGB y ahorra ~5% de FLOPs.** La vía rápida **no necesita color**. Consistente con las células M.
2. **Alimentar la vía Fast con flujo óptico es la *peor* de las cinco opciones (73.8%).** Este es el resultado más importante del paper para la narrativa de la Clase 38: no es que el flujo sea difícil de calcular y por eso lo evitamos; es que **el RGB a alta tasa temporal es mejor representación del movimiento que el flujo óptico precomputado**. El flujo, como representación diseñada a mano, descarta información que la red podría usar.

### 9.4. Entrenamiento desde cero: el control (Tabla 6)

| Modelo | Pretrain | top-1 | top-5 | GFLOPs |
| --- | --- | --- | --- | --- |
| 3D R-50 (receta de Nonlocal) | ImageNet | 73.4 | 90.9 | 36.7 |
| 3D R-50, receta de Nonlocal, desde cero | — | 69.4 | 88.6 | 36.7 |
| 3D R-50, **receta de los autores**, desde cero | — | **73.5** | 90.8 | 36.7 |

Puro rigor experimental. **Misma arquitectura exacta**, tres regímenes. La receta original entrenada desde cero pierde **4.0 puntos** (73.4 → 69.4); la receta de SlowFast desde cero **iguala** al pre-entrenamiento en ImageNet (73.5 vs 73.4). Conclusión: **la brecha de "entrenar desde cero" era una brecha de receta de optimización, no de datos** — y las comparaciones de SlowFast no están viciadas por un sistema de entrenamiento debilitado.

### 9.5. AVA: detección de acciones (Tablas 7, 8, 9 y Figura 3)

El ablation limpio (Tabla 9): **Slow-only R-50 4×16 = 19.0 mAP** contra **SlowFast R-50 4×16 con $\alpha=8$ = 24.2 mAP**. Mejora de **+5.2 mAP, 28% relativo**, atribuible **únicamente** a la idea SlowFast.

El contraste con el flujo óptico es demoledor: los trabajos previos observaban mejoras **leves** al añadirlo — **+1.1 mAP** para I3D y **+1.7 mAP** para ATR. La vía Fast aporta **+5.2 mAP**. Y los métodos two-stream con flujo pueden **duplicar** el costo, mientras la vía Fast es liviana.

Por categoría (Figura 3), SlowFast mejora en **57 de 60 categorías**. Mayores ganancias absolutas: **"hand clap" +27.7 AP**, **"swim" +27.4**, **"run/jog" +18.8**, **"dance" +15.9**, **"eat" +12.5** — exactamente donde modelar dinámica es vital. Grandes aumentos relativos en "jump/leap", "hand wave", "put down", "throw", "hit", "cut". Empeora en solo **3** y marginalmente: "answer phone" (−0.1), "lie/sleep" (−0.2), "shoot" (−0.4) — acciones cuasi-estáticas donde la vía rápida no tiene nada que aportar.

Comparación con el estado del arte en AVA v2.1 (todos los SlowFast con $T\times\tau = 8\times8$, R101):

| Modelo | Flujo | Pretrain de video | val mAP | test mAP |
| --- | --- | --- | --- | --- |
| I3D | | Kinetics-400 | 14.5 | — |
| I3D  | sí |  Kinetics-400 | 15.6 | — |
| ACRN, S3D  | sí |  Kinetics-400 | 17.4 | — |
| ATR, R50+NL | | Kinetics-400 | 20.0 | — |
| ATR, R50+NL  | sí |  Kinetics-400 | 21.7 | — |
| Ensamble de 9 modelos  | sí |  Kinetics-400 | 25.6 | 21.1 |
| I3D (Girdhar et al.) | | Kinetics-600 | 21.9 | 21.0 |
| **SlowFast** | | Kinetics-400 | **26.3** | — |
| **SlowFast** | | Kinetics-600 | 26.8 | — |
| **SlowFast, +NL** | | Kinetics-600 | 27.3 | 27.1 |
| **SlowFast\*, +NL** (propuestas propias en train) | | Kinetics-600 | **28.2** | — |

Con solo pre-entrenamiento en Kinetics-400, SlowFast logra **26.3 mAP**: **+5.6 mAP** sobre el mejor previo comparable de modelo único (21.7 de ATR) y **+7.3 mAP** sobre el mejor que no usa flujo. En **AVA v2.2** (anotaciones más consistentes): **29.0 mAP** con 8×8, **29.8** con 16×8, **30.7** con test multiescala y volteo horizontal. Un ensamble de 7 modelos alcanzó **34.3 mAP** en el test server y quedó **primero en el AVA action detection challenge 2019**.

### 9.6. Charades (Tabla 4) y Kinetics-600 (Tabla 3)

Charades, todas las variantes con $T\times\tau = 16\times8$, R-101:

| Modelo | Pretrain | mAP | GFLOPs × vistas |
| --- | --- | --- | --- |
| CoViAR, R-50 | ImageNet | 21.9 | N/A |
| Asyn-TF, VGG16 | ImageNet | 22.4 | N/A |
| MultiScale TRN | ImageNet | 25.2 | N/A |
| Nonlocal, R101 | ImageNet+K400 | 37.5 | 544 × 30 |
| STRG, R101+NL | ImageNet+K400 | 39.7 | 630 × 30 |
| Baseline propio (Slow-only) | K400 | 39.0 | 187 × 30 |
| **SlowFast** | K400 | **42.1** | 213 × 30 |
| **SlowFast, +NL** | K400 | 42.5 | 234 × 30 |
| **SlowFast, +NL** | K600 | **45.2** | 234 × 30 |

SlowFast mejora **+3.1 mAP** sobre su propio baseline Slow-only (39.0 → 42.1), NL agrega **+0.4** más, y K600 lleva a **45.2**. Contra STRG (39.7 a 630 GFLOPs/vista), SlowFast+NL con K600 gana **+5.5 mAP** a **2.7× menos costo**. Importa porque Charades tiene actividades de rango largo (~30 s promedio), donde uno esperaría que una arquitectura de ventana corta sufriera.

En **Kinetics-600**, los SlowFast van de **78.8%** (4×16 R50, 36.1 GFLOPs) a **81.8%** (16×8 R101+NL), contra 71.9% de I3D y 79.0% de StNet-IRv2 (que además usa ImageNet+K400).

## 10. Limitaciones

- **El costo de inferencia sigue siendo alto en absoluto.** El mejor modelo cuesta $234 \times 30 = 7020$ GFLOPs por video. Ser más eficiente *por vista* que Nonlocal R101 no lo hace barato: queda fuera de alcance para inferencia en tiempo real en un dispositivo modesto y muy lejos de procesar streams de video largo a escala. El propio Feichtenhofer atacaría esto al año siguiente con **X3D** (CVPR 2020).
- **La evaluación densa con múltiples vistas infla el cómputo real.** El paper reporta honestamente el factor ×30, pero eso significa que el atractivo "36.1 GFLOPs" es realmente **1083 GFLOPs por video**. El protocolo de 10 clips × 3 crops optimiza precisión de leaderboard, no eficiencia de despliegue; un sistema de producción usaría muchas menos vistas y perdería precisión de forma no cuantificada aquí.
- **La ventana temporal sigue siendo de segundos, no de minutos.** La instanciación por defecto ve un clip crudo de **64 fotogramas** (~2.13 s a 30 fps); la más larga ($16 \times 8$), 128 fotogramas (~4.3 s). El largo alcance se recupera solo por **promediado de scores** de 10 clips (o max-pooling temporal en Charades), una agregación sin memoria y sin orden. **SlowFast no modela relaciones temporales largas dentro de la red**: la vía Fast resuelve el grano fino, no el largo alcance.
- **La analogía biológica no se valida.** El paper mismo la califica de tosca y prematura, y el único punto de contacto empírico (insensibilidad al color) es débil como evidencia.
- **La fusión es unidireccional y de topología fija.** Cuatro conexiones en posiciones prefijadas, de Fast a Slow. La bidireccional dio "resultados similares", reportado como no-hallazgo sin más análisis. Tampoco se exploran más de dos vías (¿tres velocidades?) ni $\alpha$ o $\beta$ variables por stage.
- **Los ablations usan mayoritariamente una sola configuración.** Casi toda la Tabla 5 es SlowFast 4×16 R-50; la transferencia de esos óptimos ($\beta = 1/8$, T-conv) a configuraciones más grandes se asume, no se verifica.

## 11. Impacto y legado

**SlowFast se convirtió en la referencia obligada y el baseline por defecto en detección de acciones**, particularmente en AVA. Ganar el challenge 2019 con un margen tan amplio (26.3 vs 21.7 mAP en condiciones comparables) hizo que durante años cualquier trabajo nuevo en localización espacio-temporal tuviera que reportar frente a él. La combinación backbone SlowFast + Faster R-CNN con RoIAlign 3D se volvió el patrón de referencia.

**PySlowFast** ([github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)) fue tan importante como el paper: FAIR liberó no solo el modelo sino un **framework de investigación en video** completo, con recetas de entrenamiento distribuido, protocolos de evaluación multi-vista y un zoo de modelos pre-entrenados. Bajó sustancialmente la barrera de entrada y albergó luego X3D y MViT.

**Resolvió la dependencia del flujo óptico de forma elegante.** No con un módulo de flujo aprendido ni con una pérdida auxiliar, sino **eliminando el problema**: si el objetivo del stream de flujo era inyectar movimiento, basta muestrear RGB más densamente y dedicarle una vía delgada. La Tabla 5c cierra el argumento mostrando que el flujo como entrada a la vía Fast es *peor* que el RGB. Después de SlowFast, calcular TV-L1 sobre datasets de 240k videos dejó de ser parte del pipeline estándar.

**La transición hacia los video transformers.** El mismo grupo de FAIR publicó **MViT (Multiscale Vision Transformers, ICCV 2021)** —Fan, Xiong, Mangalam, Li, Yan, Malik y Feichtenhofer, con autores compartidos con SlowFast— y con ello el estado del arte migró de las ConvNets a la atención. Lo que sobrevive de SlowFast no es la arquitectura de dos vías, sino la idea más profunda: **las jerarquías multiescala**. MViT rechaza el diseño de ViT de resolución y ancho constantes y construye una pirámide donde la resolución espacio-temporal **se reduce** mientras la dimensión de canales **se expande** con la profundidad — el mismo principio de "resolución alta con pocos canales / resolución baja con muchos canales" que SlowFast explotó en paralelo sobre el eje temporal, ahora reorganizado sobre el eje de profundidad. Retrospectivamente, SlowFast es el argumento de que **el presupuesto de resolución temporal y el de capacidad de canales son recursos intercambiables**, y esa contabilidad atraviesa a MViT y a los modelos de video posteriores.

## 12. Conexión con la Clase 38

La Clase 38 (*Redes Convolucionales para reconocimiento en video — Modelos pre-entrenados*, prof. Bianca Del Solar Medrano) recorre la evolución CNN2D → CNN2D+RNN → **Two-Stream** → **C3D** → **I3D**, enumerando ventajas y desventajas de cada familia. SlowFast es el cierre natural de esa línea porque **ataca de frente la lista de desventajas de las dos últimas**.

### 12.1. Contra las tres desventajas de Two-Stream

| Desventaja según la clase | Qué hace SlowFast |
| --- | --- |
| **(a) Necesita calcular el flujo óptico de cada video** | **Resuelta.** SlowFast no calcula flujo: la vía Fast obtiene el movimiento muestreando RGB crudo a stride $\tau/\alpha = 2$, todo end-to-end. El paper va más lejos: usar flujo como entrada a la vía Fast rinde **73.8%**, *peor* que RGB (**75.6%**). En AVA el flujo aportaba +1.1 mAP a I3D y +1.7 a ATR; la vía Fast aporta **+5.2 mAP** a una fracción del costo, en vez de duplicarlo. |
| **(b) Solo considera la apariencia de un fotograma** | **Resuelta parcialmente.** La vía Slow ve $T=4$ fotogramas (16 en la configuración $16\times8$), no uno, y tiene convoluciones temporales en res4 y res5: es un muestreo esparcido con modelado temporal en las capas altas, no una instantánea única. |
| **(c) No puede capturar relaciones temporales largas** | **Mejorada, no resuelta.** La vía Fast cubre el clip densamente ($\alpha T = 32$ fotogramas sin downsampling temporal), lo que resuelve el grano *fino*. El rango largo sigue dependiendo de promediar 10 clips en inferencia, así que la desventaja **persiste** (sección 10). Los 45.2 mAP en Charades (~30 s) muestran que escala mejor de lo esperado, pero por fuerza del backbone, no por modelado explícito de largo alcance. |

### 12.2. Contra las tres desventajas de I3D

| Desventaja según la clase | Qué hace SlowFast |
| --- | --- |
| **(a) Gran cantidad de parámetros** | **Atacada estructuralmente.** El inflado de I3D multiplica los parámetros de cada filtro por el factor temporal $N$. SlowFast **deja 2D las capas tempranas de la vía Slow** (conv1 a res3) y **adelgaza la vía Fast a $\beta = 1/8$**: 0.53M contra 32.4M, ~1.6% del total. La segunda vía es casi gratis en parámetros. |
| **(b) Computacionalmente costoso** | **Resuelta con margen.** SlowFast 4×16 R50 iguala a Two-Stream I3D (75.6 vs 75.7) con **36.1 GFLOPs por vista contra 216**: **6× más barato**, sin flujo y sin ImageNet. Y el mejor SlowFast (79.8%) supera a Nonlocal R101 (77.7%) usando 234 contra 359. La razón es la cuadraticidad de los FLOPs en los canales: pagar $8\times$ en fotogramas cuesta $1/64$ por capa si se reducen los canales a $1/8$. |
| **(c) La inferencia no es más rápida** | **Atacada directamente.** El paper señala que muchos trabajos previos usan **>100 vistas** en inferencia (R(2+1)D: 115; ARTNet: 250), un costo "largamente ignorado". SlowFast usa **30** y le basta, *precisamente porque* la vía Fast ya cubre densamente el intervalo dentro de cada clip: la densidad temporal se paga una vez dentro de la red, en vez de repetidamente al muestrear clips. |

### 12.3. La reinterpretación del "two-stream"

En el vocabulario de la clase: **Two-Stream separa por modalidad** (un stream sobre RGB para apariencia, otro sobre flujo óptico precomputado para movimiento). **SlowFast conserva la intuición —dos corrientes especializadas— y cambia el eje de separación a la resolución temporal.** Ambas vías reciben **la misma modalidad** (RGB del mismo clip crudo); lo que difiere es **el framerate al que la muestrean** y **la capacidad que se les asigna**. La expertise no se impone: se induce desde el muestreo y el presupuesto de canales.

El cambio de eje tiene tres consecuencias que valen como lección de diseño general: desaparece un preprocesamiento costoso y no diferenciable; el modelo pasa a ser end-to-end; y la segunda corriente se vuelve casi gratis porque, al dedicarse solo al movimiento, **no necesita capacidad para apariencia** — algo que Two-Stream, usando el mismo backbone en ambos streams, nunca aprovechó.

### 12.4. Nota sobre el pre-entrenamiento: qué cambió entre 2017 y 2019

Este punto **relativiza el argumento central de I3D**. Toda su construcción —el inflado, el *boring-video fixed point*, el reescalado de pesos por $1/N$— existe para heredar ImageNet, porque en 2017 las 3D ConvNets entrenadas desde cero rendían notoriamente peor; la clase lo lista como *ventaja* de I3D. Dos años después SlowFast entrena **desde inicialización aleatoria** y bate a todos, y al probar ImageNet reporta diferencias de **±0.3%**. Qué cambió:

1. **La escala de datos.** Kinetics-400 con ~240k videos de entrenamiento (y K600 con ~392k) es un régimen que en 2016, con UCF-101 y HMDB-51 (del orden de 10k videos), no existía. La razón de fondo por la que I3D necesitaba ImageNet era que **no había suficiente video etiquetado**.
2. **La receta de optimización, que es el factor decisivo y está medido.** La Tabla 6 lo aísla quirúrgicamente: **la misma** 3D ResNet-50 desde cero con la receta previa da **69.4%**; con la receta de los autores, **73.5%** — igualando el 73.4% del pre-entrenamiento en ImageNet. Los **4.0 puntos** que se atribuían a "no tener ImageNet" eran brecha de optimización. Los ingredientes: SGD sincronizado con **batch 1024** sobre 128 GPUs (large-minibatch de Goyal et al., 2017), **warm-up lineal** de 8k iteraciones para tolerar $\eta = 1.6$, **schedule cosenoidal**, inicialización de He et al., BN por grupo de 8 clips y entrenamiento **largo** (256 epochs). Nada de eso era práctica estándar en 2017.
3. **El contexto de la época.** Es el mismo período en que Kaiming He —coautor aquí— publicaba *Rethinking ImageNet Pre-training* (2018), argumentando que en detección de objetos el pre-entrenamiento **acelera la convergencia pero no mejora la precisión final** si se entrena lo suficiente con la receta correcta. SlowFast es esa tesis en el dominio del video. (Contexto externo al paper, no una afirmación del texto.)

La lección: **el inflado de I3D era una solución brillante a un problema de escasez de datos y recetas inmaduras, no una verdad arquitectónica permanente.** Y hay un giro que cierra el arco: SlowFast argumenta que inflar kernels a cúbicos era además **conceptualmente equivocado** —tratar espacio y tiempo simétricamente contradice la estadística del video— y lo demuestra con el hallazgo de que poner convoluciones temporales en las capas tempranas de la vía Slow **degrada** la precisión. La herramienta que hizo posible a I3D es exactamente la que SlowFast identifica como el error de diseño a corregir.

---

**Nota final — relevancia práctica.** Dos lecciones de SlowFast trascienden el video. La primera: **cuando dos fuentes de señal tienen escalas de variación distintas, conviene procesarlas a resoluciones distintas y con presupuestos de capacidad distintos, en vez de forzar un tratamiento uniforme** — la vía que atiende la señal rápida necesita **frecuencia, no ancho**; la que atiende la señal estable necesita **ancho, no frecuencia**. La segunda: **una rama que rinde mal por sí sola (51.7% top-1) puede aportar muchísimo en conjunto (+3.0 puntos, +5.2 mAP), porque su valor está en la complementariedad y no en su desempeño aislado.** Evaluar componentes solo por su métrica individual lleva a descartar exactamente las piezas que más aportan al ensamble — un error de juicio que reaparece con la misma forma en cualquier sistema con múltiples señales de entrada, sea video, texto o registros estructurados.
