---
title: "R(2+1)D: factorizar la convolución 3D (2018)"
weight: 423
math: true
---

{{< paper-card
    title="A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    authors="Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri (Facebook AI Research, Dartmouth)"
    year="2018"
    venue="CVPR 2018 / arXiv:1711.11248"
    arxiv="1711.11248"
    pdf="/papers/r2plus1d-tran-2018.pdf" >}}
Un **estudio empírico controlado**, más que un paper de arquitectura. Si una CNN 2D sobre cuadros individuales sigue siendo competitiva en reconocimiento de acciones, ¿de verdad sirven las convoluciones 3D? Los autores fijan todo lo demás —mismo backbone residual, 18 capas, $112\times112$, entrenamiento **desde cero**— y varían **solo el tipo de convolución espacio-temporal**: cinco variantes (R2D, f-R2D, R3D, MCx, rMCx) más una sexta propia, **R(2+1)D**, donde cada conv 3D de $t\times d\times d$ se descompone en una espacial $1\times d\times d$ seguida de una temporal $t\times1\times1$, con un ReLU intercalado y con la dimensionalidad intermedia elegida para **igualar exactamente el número de parámetros** de la conv 3D original. Sin esa igualdad el experimento no diría nada; con ella, la diferencia es atribuible a la *forma* del operador y no a la capacidad. En **Kinetics** con clips de 16 cuadros, R(2+1)D-18 logra **56.8% clip top-1 / 68.0% video top-1** contra 52.5/64.2 de R3D, con 33.3M parámetros contra 33.4M y aproximadamente los mismos FLOPs. Con 34 capas y solo RGB desde cero alcanza **72.0% top-1**, **4.5 puntos** sobre [I3D](/papers/i3d-carreira-2017)-RGB también entrenado desde cero (67.5%). El argumento fino está en el error de **entrenamiento**: R(2+1)D ajusta *mejor* los datos de entrenamiento con los mismos parámetros, lo que descarta la regularización y apunta a **optimizabilidad**. Du Tran es el primer autor de [C3D](/papers/c3d-tran-2015): este es su propia revisión crítica tres años después.
{{< /paper-card >}}

---

## Contexto: el autor de C3D revisa su propio trabajo

El paper abre con el diagnóstico de 2018: el video **"aún no ha presenciado su momento AlexNet"**. Y la evidencia militaba **contra** las convoluciones 3D: [I3D](/papers/i3d-carreira-2017) le ganaba a iDT por un margen mucho menor que el que separaba a las CNN de los descriptores manuales en imágenes, y una ResNet-152 **2D** sobre cuadros individuales conseguía **46.5% clip@1 y 64.6% video@1** en [Sports-1M](/papers/large-scale-video-karpathy-2014) contra **46.1/61.1 de C3D**. Una red incapaz de modelar movimiento le ganaba a nivel de video a la 3D CNN de referencia: resultado "sorprendente y frustrante", del que sale la hipótesis nihilista que el paper refuta.

El factor de confusión era el **pre-entrenamiento en ImageNet**: las 2D heredaban millones de imágenes y las 3D partían de cero, forzadas además a ser poco profundas por la escasez de video etiquetado (C3D tiene 8 capas convolucionales). I3D atacó eso con el [inflado de convoluciones](/fundamentos/inflado-de-convoluciones); R(2+1)D **no hereda nada de ImageNet** y apuesta a que con aprendizaje residual, datasets grandes ([Kinetics](/papers/kinetics-kay-2017) ~300K videos, Sports-1M 1.1M) y una factorización que facilite la optimización, la 3D CNN gana por mérito propio.

> La [Clase 38](/clases/clase-38) presenta C3D e I3D como eslabones consecutivos y le atribuye a I3D las desventajas de tener muchos parámetros y ser costoso. Este paper es la respuesta directa a esa objeción, escrita por el propio autor de C3D.

---

## Las cinco arquitecturas comparadas

Todas comparten el esqueleto R3D: `conv1` de $3\times7\times7$ con 64 canales (stride $1\times2\times2$), cuatro grupos `conv2_x`–`conv5_x` de bloques $3\times3\times3$ con 64/128/256/512 canales, downsampling espacio-temporal en `conv3_1`, `conv4_1` y `conv5_1`, y cierre con **global average pooling sobre todo el volumen espacio-temporal** (512 dim) más FC y softmax. Bloques ResNet *vanilla* con la [conexión residual](/fundamentos/arquitectura-redes) $z_i = z_{i-1} + \mathcal{F}(z_{i-1}; \theta_i)$.

| Variante | Dónde vive la conv 3D | # params (18 capas) |
|---|---|---|
| R2D | ninguna: el tiempo se reinterpreta como canales | 11.4M |
| f-R2D | ninguna: 2D por cuadro, fusión en el pooling final | 11.4M |
| R3D | todos los grupos | 33.4M |
| MC2–MC5 | grupos tempranos (3D abajo, 2D arriba) | 11.4M–16.9M |
| rMC2–rMC5 | grupos profundos (2D abajo, 3D arriba) | 27.9M–33.3M |
| **R(2+1)D** | todos, factorizada en $1\times d\times d$ + $t\times1\times1$ | **33.3M** |

**R2D** *reshapea* $3\times L\times H\times W$ a $3L\times H\times W$: `conv1` usa filtros $N_{i-1}\times d\times d$ que convolucionan solo en 2D, y la salida es $N_i\times H_i\times W_i$, **sin eje temporal**. El paper lo enuncia comprimido ("ignoran el ordenamiento temporal"), pero `conv1` **sí puede codificar orden**: tiene pesos distintos para cada uno de los $3L$ canales y podría aprender algo parecido a una derivada temporal, la *early fusion* de [Karpathy et al.](/papers/large-scale-video-karpathy-2014). Lo que R2D destruye es peor: **tras `conv1` el eje temporal desapareció**, y eso impide todo razonamiento temporal posterior (por eso tampoco hay striding temporal). Es ≈7× más rápida que f-R2D y la menos precisa: con 16 cuadros queda **1.6% bajo f-R2D a nivel de video** (58.9 vs 60.5).

**f-R2D** procesa los $L$ cuadros independientemente con filtros 2D compartidos e integra solo en el pooling global: el baseline de bag-of-frames de la [Clase 36](/clases/clase-36). **R3D** propaga el eje temporal con filtros $N_{i-1}\times t\times d\times d$ y $t=3$: el C3D moderno, residual y profundo, a casi 3× el costo en parámetros.

**MCx** supone que el movimiento es de nivel bajo o medio y que las capas semánticas no necesitan eje temporal: MC5 pasa a 2D el grupo 5, MC4 los grupos 4 y 5, y así hasta MC2 (MC1 se omite porque equivale a f-R2D). **rMCx** invierte la hipótesis. Como los grupos profundos tienen 256 y 512 canales, ahí vive la mayoría de los pesos: 3D arriba cuesta casi lo mismo que R3D completa y 3D solo abajo es casi tan barato como una red 2D. Esa asimetría es el argumento económico central.

> Gotcha: el downsampling espacio-temporal es **striding convolucional 3D**, así que al pasar una conv a 2D queda solo espacial. Por eso MCx y rMCx dan tamaños temporales distintos en la última capa convolucional y el costo no varía monótonamente: rMC3 es *más eficiente* que f-R2D, porque sí hace striding temporal en `conv3_1`.

---

## Convoluciones mixtas: la evidencia real

Kinetics, validación, ResNet-18 desde cero:

| Red | # params | Clip@1 (8f) | Video@1 (8f) | Clip@1 (16f) | Video@1 (16f) |
|---|---|---|---|---|---|
| R2D | 11.4M | 46.7 | 59.5 | 47.0 | 58.9 |
| f-R2D | 11.4M | 48.1 | 59.4 | 50.3 | 60.5 |
| R3D | 33.4M | 49.4 | 61.8 | 52.5 | 64.2 |
| MC2 | 11.4M | 50.2 | 62.5 | 53.1 | 64.2 |
| MC3 | 11.7M | 50.7 | 62.9 | 53.7 | 64.7 |
| MC4 | 12.7M | 50.5 | 62.5 | 53.7 | 65.1 |
| MC5 | 16.9M | 50.3 | 62.5 | 53.7 | 65.1 |
| rMC2 | 33.3M | 49.8 | 62.1 | 53.1 | 64.9 |
| rMC3 | 33.0M | 49.8 | 62.3 | 53.2 | 65.0 |
| rMC4 | 32.0M | 49.9 | 62.3 | 53.4 | 65.1 |
| rMC5 | 27.9M | 49.4 | 61.2 | 52.1 | 63.1 |
| **R(2+1)D** | **33.3M** | **52.8** | **64.8** | **56.8** | **68.0** |

**El movimiento sí importa.** La brecha entre las ResNets 2D y cualquier modelo con conv 3D es de **1.3–4%** con 8 cuadros y **1.8–6.7%** con 16. Todos ven el mismo input; la única diferencia es qué hacen con el tiempo, y que la brecha crezca con clips más largos es el argumento decisivo.

**MCx gana en eficiencia, no en exactitud.** El titular real no es "3D abajo es mejor" sino "3D abajo es *suficiente*": las MC ResNets dan 3-4% a nivel de clip sobre ResNets 2D comparables e **igualan a R3D con 2.93× menos parámetros** (33.4M / 11.4M).

**Y acá el matiz honesto: la evidencia "bottom-heavy" es más débil de lo que suele citarse.** A nivel de clip MCx le gana consistentemente a rMCx, pero por 0.4–0.9 puntos. A **nivel de video con 16 cuadros la ventaja se evapora**: rMC3 obtiene 65.0 contra 64.7 de MC3, y rMC4 empata con MC4 y MC5 en 65.1. El único perdedor claro es rMC5 (63.1), el caso degenerado de meter 3D solo en el último grupo. Cualquier lectura fuerte de "el movimiento es de bajo nivel" sobre-interpreta diferencias de menos de un punto.

Importa porque [S3D](/papers/s3d-xie-2018) (Xie et al., ECCV 2018), contemporáneo y con un experimento casi idéntico, concluye lo **opuesto**: partiendo de I3D sobre Inception, la variante **top-heavy** —el rMCx de este paper— resulta a la vez más rápida y más precisa. Cuatro diferencias rompen la comparabilidad: el **backbone** (ResNet-18 vanilla contra Inception-v1 inflada, con reparto de canales y FLOPs por etapa muy distinto); la **interacción con el striding temporal** (aquí pasar a 2D *elimina* el downsampling temporal de esa etapa, lo que en una red top-heavy obliga a arrastrar tensores temporales grandes por las capas 2D iniciales); la **resolución y longitud** ($112\times112$ con 8–16 cuadros contra $224\times224$ con 64); y el **pre-entrenamiento** (desde cero contra ImageNet, que sesga a favor de dejar 2D las capas tempranas heredadas). La coda es que [SlowFast](/papers/slowfast-feichtenhofer-2019) adoptó la postura de S3D, con convoluciones temporales no degeneradas solo en res4 y res5. La moraleja: **la ubicación óptima de la capacidad temporal no es una ley del dominio sino una propiedad del backbone, la resolución y el esquema de downsampling**.

---

## R(2+1)D: la factorización

Se reemplazan los $N_i$ filtros 3D de $N_{i-1}\times t\times d\times d$ por $M_i$ filtros **espaciales** de $N_{i-1}\times 1\times d\times d$ seguidos de $N_i$ filtros **temporales** de $M_i\times t\times 1\times 1$, con un **ReLU intercalado**. El subespacio intermedio se dimensiona así:

$$M_i = \left\lfloor \frac{t\, d^2\, N_{i-1}\, N_i}{d^2\, N_{i-1} + t\, N_i} \right\rfloor$$

La derivación es de una línea. La conv 3D tiene $P_{3D} = N_{i-1}\, t\, d^2\, N_i$ parámetros; el bloque factorizado tiene

$$P_{(2+1)D} = \underbrace{N_{i-1}\, d^2\, M_i}_{\text{espacial}} + \underbrace{M_i\, t\, N_i}_{\text{temporal}} = M_i\,(d^2 N_{i-1} + t N_i)$$

Igualando $P_{(2+1)D} = P_{3D}$ y despejando sale exactamente la fórmula. **Ese es el corazón metodológico del paper**: sin esa elección, cualquier mejora sobre R3D sería atribuible a más capacidad; con ella, R3D y R(2+1)D-18 tienen 33.4M y 33.3M parámetros y aproximadamente los mismos FLOPs. Lo contraintuitivo es que **el subespacio intermedio es más ancho que la salida**. Con $N_{i-1} = N_i = N$, $t = 3$, $d = 3$:

$$M_i = \frac{3 \cdot 9 \cdot N^2}{N(9 + 3)} = \frac{27N}{12} = 2.25\,N$$

En `conv5_x` (512 canales) el bloque usa **1152 filtros espaciales** de $1\times3\times3$ antes de proyectar a 512 con filtros temporales de $3\times1\times1$; análogamente 144 para bloques 64→64, 230 para 64→128, 460 para 128→256 y 921 para 256→512. La factorización **no es un recorte sino un re-gasto** del mismo presupuesto: compra ancho espacial y una no-linealidad extra a cambio de renunciar al acoplamiento conjunto de espacio y tiempo dentro de un mismo filtro. Los dos beneficios argumentados son **duplicar las no-linealidades** sin cambiar parámetros (el argumento de VGG) y **facilitar la optimización**. FSTCN (2015) factoriza la *red* y no la *capa*; P3D (2017) intercala tres bloques distintos y su P3D-A es el pariente más cercano, pero usa bottlenecks y **no iguala el conteo de parámetros**. R(2+1)D es deliberadamente **homogéneo**.

> Gotcha verificado sobre el stem. El apéndice describe `conv1` con **45 filtros de $1\times7\times7$ y 64 de $3\times1\times1$**, y afirma que iguala los parámetros de R3D. La cuenta no cuadra: la conv 3D tiene $3\cdot64\cdot3\cdot49 = 28{,}224$ parámetros y el bloque con $M=45$ tiene $3\cdot45\cdot49 + 45\cdot64\cdot3 = 15{,}255$; la fórmula habría dado $M_1 = 83$ (28,137 parámetros, sí equiparable). En el stem la igualdad es aproximada y el 45 es el número que quedó en la implementación de referencia, que sobrevive en `torchvision` (`R2Plus1dStem`: `Conv3d(3, 45, (1,7,7))` → `Conv3d(45, 64, (3,1,1))`). Conviene saberlo antes de reproducir la tabla de parámetros a mano.

---

## El argumento fino: optimización, no regularización

Graficando error de **entrenamiento** y de validación por época para R3D y R(2+1)D, con 18 y con 34 capas a lo largo de 45 épocas, **R(2+1)D tiene menor error de entrenamiento**, no solo menor error de test, y la brecha en las pérdidas de entrenamiento es **particularmente grande en la red de 34 capas**.

Si solo tuviera menor error de *test*, la explicación natural sería **regularización**: la factorización restringe el espacio de hipótesis —impone que el filtro sea separable, un subconjunto de rango bajo de los filtros 3D posibles— y actuaría como prior contra el sobreajuste. Sería una conclusión mucho más modesta. Pero tener menor error de **entrenamiento** con el **mismo número de parámetros** la descarta: un modelo estrictamente menos expresivo no debería ajustar *mejor* los datos de entrenamiento. Que lo haga significa que el problema no es de capacidad sino de **optimizabilidad**, y que la factorización cambia el paisaje de la pérdida, no el conjunto de funciones alcanzables. Que el efecto **crezca con la profundidad** es coherente: las patologías de optimización se agravan con la profundidad, el mismo régimen donde las conexiones residuales fueron necesarias en imágenes. La lección transferible: **cómo se parametriza un operador afecta al entrenamiento incluso a paridad de parámetros y de expresividad nominal**.

---

## Resultados

**Kinetics.** R(2+1)D-18 supera a MCx, rMCx y R3D por **2.1–3.4%** con 8 cuadros y por **3.1–4.7%** con 16; y a las ResNets 2D por **4.7–6.1%** y **6.3–9.8%**. Contra FLOPs domina la frontera, con **3–3.8%** sobre R3D **al mismo costo computacional**. Nota de protocolo: **video top-1** promedia 10 clips con recortes centrales espaciados uniformemente (100 en Sports-1M, donde los videos pasan de 5 minutos) y la brecha con **clip top-1** es de 10–12 puntos, así que cruzar ambas métricas entre papers es un error clásico.

**Longitud del clip.** Variando $L \in \{8, 16, 24, 32, 40, 48\}$, la exactitud a nivel de clip **sigue creciendo**, pero la de video **hace pico en 32 cuadros**. Como el pooling global no tiene parámetros, todos estos modelos tienen el mismo conteo:

| Train (cuadros) | Finetune | Test (cuadros) | Tiempo entren. (h) | Clip@1 | Video@1 |
|---|---|---|---|---|---|
| 8 | — | 8 | 11.8 | 52.8 | 64.8 |
| 8 | — | 32 | 11.8 | 51.6 | 59.0 |
| 32 | — | 32 | 59.8 | 60.1 | 69.4 |
| 8 | 32 | 32 | 20.5 | 59.8 | 68.0 |

(Tiempos con 64 GPUs en paralelo.) **Alargar el clip solo en test es contraproducente**: evaluar con 32 cuadros el modelo entrenado con 8 baja 1.2% en clip y **5.8% en video**; entrenar con clips largos produce modelos cualitativamente distintos, con filtros que aprenden patrones de mayor plazo. El atajo eficiente es hacer fine-tuning del modelo de 32 cuadros inicializándolo con el de 8: llega a 59.8% contra 60.1% de entrenar desde cero, en **20.5 h en lugar de 59.8 h**, porque el modelo de 8 cuadros es 7.3× más rápido en FLOPs. En inferencia, 20 recortes quedan ~0.5% bajo 100 y son 5× más rápidos.

**Sports-1M (R(2+1)D-34).**

| Método | Clip@1 | Video@1 | Video@5 |
|---|---|---|---|
| C3D | 46.1 | 61.1 | 85.2 |
| 2D ResNet-152 | 46.5 | 64.6 | 86.4 |
| Conv pooling | — | 71.7 | 90.4 |
| P3D | 47.9 | 66.4 | 87.4 |
| R3D-RGB-8f | 53.8 | — | — |
| R(2+1)D-RGB-8f | 56.1 | 72.0 | 91.2 |
| R(2+1)D-RGB-32f | **57.0** | 73.0 | 91.5 |
| R(2+1)D-Two-Stream-32f | — | **73.3** | **91.9** |

R(2+1)D-RGB supera a **C3D por 10.9%** y a **P3D por 9.1%** en clip@1, y al 2D ResNet por 10.5%, pese a que ResNet-152 y P3D tienen 152 capas contra las 34 de R(2+1)D. Contra su propio baseline R3D-34 con 8 cuadros RGB la ventaja es de **2.3%** (56.1 vs 53.8), lo que aísla el aporte de la descomposición controlando arquitectura y datos.

**Comparación con I3D en Kinetics.**

| Método | Pre-entrenamiento | Top-1 | Top-5 |
|---|---|---|---|
| I3D-RGB | ninguno | 67.5 | 87.2 |
| I3D-RGB | ImageNet | 72.1 | 90.3 |
| I3D-Flow | ImageNet | 65.3 | 86.2 |
| I3D-Two-Stream | ImageNet | **75.7** | **92.0** |
| R(2+1)D-RGB | ninguno | 72.0 | 90.0 |
| R(2+1)D-Flow | ninguno | 67.5 | 87.2 |
| R(2+1)D-Two-Stream | ninguno | 73.9 | 90.9 |
| R(2+1)D-RGB | Sports-1M | 74.3 | 91.4 |
| R(2+1)D-Flow | Sports-1M | 68.5 | 88.1 |
| R(2+1)D-Two-Stream | Sports-1M | 75.4 | 91.9 |

Las tres comparaciones que permite la tabla son distintas, y conviene ser explícito con los asteriscos. **Sin pre-entrenamiento, R(2+1)D gana claro**: 72.0 vs 67.5 en RGB, **+4.5%**, la comparación más limpia del paper. **Pre-entrenado en Sports-1M supera a I3D pre-entrenado en ImageNet** por +2.2% en RGB (74.3 vs 72.1) y +3.2% en flujo (68.5 vs 65.3), pero acá las **fuentes de pre-entrenamiento son distintas** y el asterisco corta en las dos direcciones: Sports-1M aporta 1.1M videos mucho más afines al dominio que las imágenes de ImageNet, aunque con etiquetas más ruidosas y restringidas a deportes. **En two-stream I3D sigue adelante por 0.3%** (75.7 vs 75.4), y el propio paper señala la causa: R(2+1)D usa flujo óptico de **Farnebäck** por eficiencia mientras I3D usa **TV-L1**, más preciso y un orden de magnitud más lento, así que parte de la diferencia final es de **preprocesamiento, no de arquitectura**.

**Transferencia** (promedio de los 3 splits estándar):

| Método | Pre-entrenamiento | UCF-101 | HMDB-51 |
|---|---|---|---|
| Temporal Segment Network | ImageNet | 94.2 | 69.4 |
| I3D-RGB | ImageNet+Kinetics | 95.6 | 74.8 |
| I3D-Two-Stream | ImageNet+Kinetics | **98.0** | **80.7** |
| R(2+1)D-RGB | Sports-1M | 93.6 | 66.6 |
| R(2+1)D-Two-Stream | Sports-1M | 95.0 | 72.7 |
| R(2+1)D-RGB | Kinetics | 96.8 | 74.5 |
| R(2+1)D-Flow | Kinetics | 95.5 | 76.4 |
| R(2+1)D-Two-Stream | Kinetics | 97.3 | 78.7 |

R(2+1)D supera a todos los métodos de la comparación excepto I3D, que además usa ImageNet. Y aparece un hallazgo lateral muy reutilizable: **Kinetics es mejor fuente de pre-entrenamiento que Sports-1M** por márgenes grandes, +3.2 puntos en UCF-101 (96.8 vs 93.6 en RGB) y **+7.9 en HMDB-51** (74.5 vs 66.6), pese a ser 3.6× más chico. La lectura honesta: **R(2+1)D es la mejor arquitectura de video entrenada desde cero de 2018, y su ventaja sobre I3D no está establecida cuando I3D puede usar ImageNet y buen flujo óptico**. Lo que sí queda establecido es que la dependencia de ImageNet, que en 2017 parecía indispensable, dejó de serlo.

---

## Limitaciones

- **El costo de entrenamiento sigue siendo alto**, y no es un problema exclusivo de I3D: entrenar R(2+1)D-18 con clips de 32 cuadros en Kinetics toma **59.8 horas con 64 GPUs en paralelo**, del orden de 3800 GPU-horas para un modelo de 18 capas.
- **El mejor resultado depende del flujo óptico precomputado**, un paso costoso fuera de la red que contradice la promesa end-to-end. Y el balance es pobre: con pre-entrenamiento Sports-1M, RGB solo da 74.3 contra 75.4 de two-stream, o sea **1.1 punto** por duplicar el cómputo.
- **La ventana temporal es corta.** La exactitud a nivel de video hace pico en **32 cuadros**, poco más de un segundo a 25 fps; más allá, la agregación es un promedio de clips independientes sin modelo del orden entre ellos.
- **No tener pre-entrenamiento en ImageNet es una desventaja estructural.** La factorización no ofrece nada análogo al *boring-video fixed point* de I3D: se podría inicializar la parte espacial $1\times d\times d$ desde una red 2D, pero el ancho intermedio $M_i = 2.25\,N_i$ no coincide con los canales de ninguna ResNet de imagen.
- **Alcance del estudio.** Un solo tipo de red (ResNet) y un uso **homogéneo** de la descomposición en todas las capas: no exploran combinarla con la asignación no uniforme de capacidad temporal que sus propias secciones MC/rMC insinúan.

---

## Por qué importa hoy

R(2+1)D se volvió el **backbone de video por defecto** de los años siguientes por una razón pragmática: es un ResNet, y la factorización se implementa con dos `Conv3d` de kernels degenerados, sin operadores nuevos. Está en `torchvision.models.video.r2plus1d_18` con pesos de Kinetics-400, junto a `r3d_18` y `mc3_18`: tres arquitecturas de este paper llegaron a la librería estándar de PyTorch, algo poco común para un paper de ablaciones. Su lugar en la historia de las ideas es compartido con [S3D](/papers/s3d-xie-2018): dos papers del mismo año, desde tradiciones distintas (ResNet contra Inception), que **consolidaron la separabilidad espacio-temporal** como principio de diseño del [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones). Después de 2018 prácticamente ningún modelo de video usa convoluciones 3D densas sin factorizar.

La cadena del propio Du Tran es explícita: [C3D](/papers/c3d-tran-2015) (2015) instala las features espacio-temporales aprendidas → **R(2+1)D** (2018) factoriza el operador y muestra que el problema era de optimización → **CSN** (*Channel-Separated Convolutional Networks*, ICCV 2019) separa además el eje de canales, con exactitud comparable a costos mucho menores → **IG-65M** (Ghadiyaram, Tran, Mahajan, CVPR 2019) usa R(2+1)D como backbone para pre-entrenamiento débilmente supervisado sobre 65 millones de videos de Instagram, la respuesta definitiva a la falta de ImageNet: si el problema era el pre-entrenamiento a escala, se construye uno de video. En paralelo, [SlowFast](/papers/slowfast-feichtenhofer-2019) recogió el otro hilo, el de *dónde* poner la capacidad temporal.

El puente a los video transformers es literal. **TimeSformer** (Bertasius, Wang, **Torresani**, ICML 2021), que comparte co-autor con este paper, propone *divided space-time attention*: atender primero en el espacio y luego en el tiempo, en operaciones separadas dentro de cada bloque. Es la misma factorización trasladada de la convolución a la atención, con idéntica justificación —mismo presupuesto, mejor optimizable, más barato que la versión conjunta—, y **ViViT** explora la misma familia con su *factorised encoder*. La lección de que **factorizar el operador espacio-temporal es casi siempre mejor que aplicarlo de forma conjunta** sobrevivió al cambio completo de arquitectura.

---

## Notas y enlaces

- La [Clase 38](/clases/clase-38) recorre la escalera CNN2D + temporal pooling → CNN2D + RNN → Two-Stream → [C3D](/papers/c3d-tran-2015) → [I3D](/papers/i3d-carreira-2017); la [teoría](/clases/clase-38/teoria) da el recorrido y la [profundización](/clases/clase-38/profundizacion) desarrolla la matemática de los operadores espacio-temporales.
- La [práctica de la clase](/clases/clase-38/practica) implementa el bloque (2+1)D con la fórmula de $M_i$ y verifica numéricamente el empate de parámetros contra la conv 3D equivalente: es el ejercicio que vuelve concreto el $2.25N$ del subespacio intermedio.
- Contraste obligado: [I3D](/papers/i3d-carreira-2017) ataca el lado de los **datos y la inicialización** vía [inflado de convoluciones](/fundamentos/inflado-de-convoluciones); R(2+1)D ataca el lado del **operador y la optimización**. El veredicto de la historia es que se combinaron.
- [S3D](/papers/s3d-xie-2018) llega a la conclusión inversa sobre dónde poner la conv 3D y [SlowFast](/papers/slowfast-feichtenhofer-2019) sigue esa línea: vale leer los tres juntos como lección sobre no extrapolar ablaciones entre backbones.
- [Kinetics](/papers/kinetics-kay-2017) y [Sports-1M](/papers/large-scale-video-karpathy-2014) habilitan el entrenamiento desde cero; el backbone es [ResNet](/papers/resnet-he-2015) vanilla y el marco de decisiones de diseño está en [arquitectura de redes](/fundamentos/arquitectura-redes). La [Clase 36](/clases/clase-36) introdujo el baseline de bag-of-frames, que aquí reaparece como f-R2D.
- Para video clínico —endoscopía, ecocardiografía, análisis de marcha— tres cosas se trasladan casi sin cambios: entrenar con clips cortos y hacer fine-tuning con clips largos recupera casi toda la exactitud a un tercio del cómputo; la afinidad y limpieza del dataset fuente pesan más que su tamaño bruto; y la distinción entre exactitud a nivel de clip y de video (57% contra 73%) cambia por completo la lectura de la utilidad diagnóstica de un modelo.
