# A Closer Look at Spatiotemporal Convolutions for Action Recognition (R(2+1)D) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *A Closer Look at Spatiotemporal Convolutions for Action Recognition*.
- **Autores:** Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, **Yann LeCun** y Manohar Paluri. Todos en **Facebook Research** (hoy Meta AI); Torresani además en **Dartmouth College**.
- **Venue:** *IEEE Conference on Computer Vision and Pattern Recognition* (CVPR 2018).
- **Preprint:** arXiv:1711.11248v3 (12 abr 2018), [arxiv.org/abs/1711.11248](https://arxiv.org/abs/1711.11248).
- **Modelo estrella:** **R(2+1)D** — un ResNet de video donde cada convolución 3D se factoriza en una convolución espacial 2D seguida de una convolución temporal 1D.
- **Implementación:** Caffe2, entrenamiento distribuido síncrono sobre clusters de GPU (hasta 64 GPUs en paralelo).

El paper no es principalmente un paper de arquitectura: es un **estudio empírico controlado**. La pregunta de partida es incómoda para el campo: si las CNN 2D aplicadas a fotogramas individuales siguen siendo competitivas en reconocimiento de acciones, ¿de verdad sirven las convoluciones 3D? Los autores responden fijando todo lo demás —el mismo backbone residual, la misma profundidad (18 capas), el mismo input, el mismo protocolo— y variando **solo el tipo de convolución espacio-temporal**. Comparan cinco variantes: R2D, f-R2D, R3D, MCx y rMCx, y proponen una sexta, R(2+1)D.

Las cifras clave: en el set de validación de **Kinetics**, con clips de 16 fotogramas y 18 capas entrenadas desde cero, R(2+1)D obtiene **56.8% clip top-1 / 68.0% video top-1**, contra 52.5/64.2 de R3D y 50.3/60.5 de f-R2D, con **33.3M parámetros** (prácticamente los mismos 33.4M de R3D) y aproximadamente el mismo costo en FLOPs. Con 34 capas, **R(2+1)D-RGB entrenado desde cero en Kinetics alcanza 72.0% top-1, superando por 4.5% a I3D-RGB entrenado también desde cero (67.5%)**. En **Sports-1M** logra 73.3% video top-1, superando a C3D por 10.9% y a P3D por 9.1% en clip top-1 con solo 34 capas contra las 152 de P3D. En transferencia, R(2+1)D pre-entrenado en Kinetics llega a **97.3% en UCF-101 y 78.7% en HMDB-51**, casi a la par de I3D (98.0 / 80.7) que usa además ImageNet.

El dato biográfico importa: **Du Tran es el primer autor de C3D** (Tran et al., ICCV 2015), el modelo que popularizó las convoluciones 3D para video. Este paper es su propia revisión crítica tres años más tarde, y desmonta dos limitaciones de C3D: la poca profundidad y la convolución 3D monolítica.

## 2. Contexto: el estado de la cuestión en 2018

La frase que abre el paper resume el diagnóstico: el dominio del video **"aún no ha presenciado su momento AlexNet"**. En imágenes hubo una secuencia clara de avances de diseño —filtros espaciales más pequeños (VGG), convoluciones multi-escala (Inception), aprendizaje residual (ResNet), conexiones densas (DenseNet)— con un ganador identificable en cada etapa. En video, no.

Peor aún: la evidencia empírica militaba contra las convoluciones 3D. Dos hechos incómodos:

1. El mejor modelo de video de la época (**I3D**, Carreira y Zisserman 2017) le ganaba al mejor método hecho a mano (**iDT**, improved Dense Trajectories) por un margen mucho menos impresionante que el que separaba a las CNN de los descriptores manuales en imágenes.
2. Una CNN 2D de imagen —**ResNet-152** operando sobre fotogramas individuales— lograba un desempeño *notablemente cercano* al estado del arte en Sports-1M. En la Tabla 4 del paper, ese ResNet-152 2D consigue **46.5% clip@1 y 64.6% video@1**, contra 46.1/61.1 de C3D. Es decir: una red sin ninguna capacidad de modelar movimiento **le ganaba a nivel de video a la 3D CNN de referencia**.

Los autores llaman a este resultado "sorprendente y frustrante". De ahí la hipótesis nihilista que el paper se propone refutar: que el razonamiento temporal *no es esencial* para reconocer acciones, porque los fotogramas estáticos ya contienen la información de clase.

Hay un factor de confusión estructural que explicaba buena parte de esa paridad: el **pre-entrenamiento en ImageNet**. Las arquitecturas 2D heredaban gratis millones de imágenes etiquetadas; las 3D CNN partían de cero. Y el problema se realimentaba: como la dimensionalidad de parámetros de una conv 3D es alta y los datos de video etiquetados eran escasos, las 3D CNN estaban forzadas a ser **poco profundas**. C3D tiene 8 capas convolucionales. I3D atacó exactamente ese cuello de botella con el truco del inflado (heredar pesos 2D de ImageNet replicándolos en el tiempo y dividiendo por $N$), lo que le permitió ser profunda *y* pre-entrenada.

R(2+1)D toma el camino opuesto: **no hereda nada de ImageNet**. Su apuesta es que si se combina (a) aprendizaje residual, (b) datasets de video suficientemente grandes para entrenar desde cero (Sports-1M con 1.1M videos, Kinetics con ~300K) y (c) una factorización que facilite la optimización, la 3D CNN gana por mérito propio. La comparación con I3D queda entonces atravesada por ese asterisco, y el paper es honesto en explicitarlo.

## 3. Contribución central

Dos aportes entrelazados:

1. **Un estudio empírico controlado de redes residuales para video.** Los autores restringen todo a ResNets "vanilla" (bloques de dos convoluciones con ReLU, sin bottlenecks), fijan 18 capas y el mismo input, y comparan cinco formas de convolución espacio-temporal. La restricción es metodológicamente valiosa: cualquier diferencia observada es atribuible al tipo de convolución, no al backbone ni al pre-entrenamiento. El bloque residual es el estándar:

   $$z_i = z_{i-1} + \mathcal{F}(z_{i-1}; \theta_i)$$

   con $\mathcal{F}$ implementando la composición de dos convoluciones y sus ReLU. La red culmina en un **global average pooling sobre todo el volumen espacio-temporal** (512 dimensiones) y una capa fully-connected con softmax.

2. **El bloque R(2+1)D**, que factoriza cada convolución 3D en una espacial 2D seguida de una temporal 1D, con una elección deliberada de la dimensionalidad intermedia que **iguala el número de parámetros** con la conv 3D original.

Notación del paper, que conviene fijar: el clip de entrada $x$ tiene tamaño $3 \times L \times H \times W$ ($L$ fotogramas, RGB), y $z_i$ es el tensor que produce el $i$-ésimo bloque residual.

## 4. Las cinco arquitecturas comparadas

La Figura 1 del paper las dibuja lado a lado. Todas parten del mismo esqueleto R3D de la Tabla 1: `conv1` con filtro $3\times 7\times 7$ y 64 canales (stride $1\times2\times2$, solo downsampling espacial), y luego cuatro grupos `conv2_x` a `conv5_x` con bloques de $3\times3\times3$ y 64/128/256/512 canales. El downsampling espacio-temporal ocurre en `conv3_1`, `conv4_1` y `conv5_1` con stride $2\times2\times2$.

### 4.1. R2D: 2D sobre todo el clip

R2D **reinterpreta la dimensión temporal como canales**. El tensor 4D de entrada $3 \times L \times H \times W$ se *reshapea* a un tensor 3D de $3L \times H \times W$. Los filtros de `conv1` son entonces de tamaño $N_{i-1}\times d \times d$ y se convolucionan **solo en 2D** sobre las dimensiones espaciales. Cada filtro produce una salida de un solo canal, y el tensor de salida $z_i$ es $N_i \times H_i \times W_i$: **sin eje temporal**.

Este es el detalle que conviene entender bien, porque el paper lo enuncia de forma comprimida ("ignoran el ordenamiento temporal"). Estrictamente, `conv1` *sí* puede codificar orden: el filtro tiene pesos distintos para cada uno de los $3L$ canales de entrada, así que podría aprender algo parecido a una derivada temporal —es la "early fusion" de Karpathy et al. Lo que R2D destruye es peor: **después de `conv1` el eje temporal desapareció**, y eso *impide que ocurra cualquier razonamiento temporal en las capas subsiguientes*. Como los mapas de features ya no tienen significado temporal, **no se aplica striding temporal** en esta red. Consecuencias prácticas: R2D es la más barata (≈7× más rápida que f-R2D) y la menos precisa, y su degradación con clips largos es diagnóstica — con 16 fotogramas queda **1.6% por debajo de f-R2D a nivel de video** (58.9 vs 60.5), porque un único punto de fusión temporal maneja mal entradas largas.

### 4.2. f-R2D: 2D por fotograma ("frame-based")

Procesa los $L$ fotogramas **independientemente** con los mismos filtros 2D compartidos. No hay modelado temporal en ninguna capa convolucional; la única integración temporal es el **global spatiotemporal pooling** del final, que fusiona las representaciones extraídas por separado de cada fotograma. Es exactamente el "CNN2D + temporal pooling" con que abre la escalera de la Clase 38. Como no hay striding temporal, la salida de la última conv es $L \times 7 \times 7$.

### 4.3. R3D: 3D completa

Preserva el eje temporal y lo propaga por toda la red. $z_i$ es 4D de tamaño $N_i \times L \times H_i \times W_i$, y cada filtro es 4-dimensional de tamaño $N_{i-1} \times t \times d \times d$, con **$t = 3$** (igual que en C3D e I3D). Es el C3D moderno: residual, 18 o 34 capas. Cuesta **33.4M parámetros** contra los 11.4M de las variantes 2D — casi 3× más.

### 4.4. MCx: Mixed Convolutions (3D abajo, 2D arriba)

La hipótesis: el modelado de movimiento es una operación de nivel bajo/medio, implementable con conv 3D en las capas tempranas, mientras que el razonamiento espacial sobre esos features de movimiento —en las capas altas, de mayor abstracción semántica— no necesita eje temporal. Se define **MC5** reemplazando todas las convs 3D del grupo 5 por 2D; **MC4** convierte los grupos 4 y 5; y así hasta **MC3** y **MC2**. MC1 se omite porque equivale a f-R2D sobre un clip.

Gotcha de implementación importante: **el downsampling espacio-temporal está implementado como striding convolucional 3D**. Cuando una conv 3D se reemplaza por 2D, ese striding pasa a ser solo espacial. Por eso MCx y rMCx producen tensores de tamaños temporales distintos en la última capa convolucional, y por eso el costo computacional no varía de forma monótona (rMC3 resulta *más eficiente* que f-R2D, porque sí hace striding temporal en `conv3_1` y con eso encoge los tensores de todas las capas 2D posteriores).

### 4.5. rMCx: Reversed Mixed Convolutions (2D abajo, 3D arriba)

La hipótesis inversa: las capas tempranas capturan apariencia (2D basta) y el modelado temporal rinde más en profundidad, donde ya hay semántica. **rMC3** usa 2D en los grupos 1 y 2, y 3D desde el grupo 3 en adelante; análogamente rMC2, rMC4 y rMC5.

Consecuencia de conteo que vale internalizar: como los grupos profundos tienen 256 y 512 canales, es ahí donde vive la mayoría de los parámetros. Poner 3D arriba (rMCx: 27.9–33.3M) cuesta casi lo mismo que R3D completa; poner 3D solo abajo (MCx: 11.4–16.9M) es casi tan barato como una red 2D. Esa asimetría es el argumento económico central a favor de MC.

## 5. El hallazgo sobre convoluciones mixtas, y el desacuerdo con S3D

Los números de Kinetics (validación, ResNet-18 desde cero):

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

Tres lecturas:

**Primera: el movimiento sí importa.** La brecha entre las ResNets 2D (R2D, f-R2D) y cualquier modelo con conv 3D es de **1.3–4%** con 8 fotogramas y crece a **1.8–6.7%** con 16. Todos los modelos ven el mismo input y procesan todos los fotogramas del clip; la única diferencia es qué hacen con el tiempo. El crecimiento de la brecha con clips más largos es el argumento decisivo: **el modelado temporal rinde más cuanto más larga es la ventana**, algo invisible si uno solo mide con 8 fotogramas.

**Segunda: MCx gana en eficiencia, no en exactitud.** El abstract dice que las MC ResNets dan "3-4% de ganancia en clip-level sobre ResNets 2D de capacidad comparable" y **"igualan el desempeño de las 3D ResNets, que tienen 3 veces más parámetros"** (33.4M / 11.4M = 2.93×, verificado). El titular real de MC no es "3D abajo es mejor", es "3D abajo es *suficiente*".

**Tercera, y es el gotcha: la evidencia a favor de "bottom-heavy" es más débil de lo que suele citarse.** A nivel de clip, MCx le gana consistentemente a rMCx, pero por márgenes de 0.4–0.9 puntos. A **nivel de video con 16 fotogramas la ventaja se evapora**: rMC3 obtiene 65.0 contra 64.7 de MC3, y rMC4 empata con MC4 y MC5 en 65.1. El único perdedor claro es rMC5 (63.1), que es el caso degenerado de meter 3D solo en el último grupo. Cualquier lectura fuerte de "el movimiento es de bajo nivel" está sobre-interpretando diferencias de menos de un punto.

Esto se vuelve relevante porque **S3D** (Xie, Sun, Huang, Tu, Murphy — *Rethinking Spatiotemporal Feature Learning*, ECCV 2018), contemporáneo y con un experimento conceptualmente idéntico, llega a la conclusión **opuesta**: partiendo de I3D sobre Inception y reemplazando progresivamente convs 3D por 2D, encuentran que la variante **top-heavy** (2D abajo, 3D arriba — el rMCx de este paper) es *simultáneamente más rápida y más precisa* que la bottom-heavy, en Kinetics y en Something-Something. Su modelo final, S3D, aplica además la misma factorización separable espacio/tiempo que R(2+1)D.

¿Por qué no coinciden? Hay al menos cuatro diferencias que rompen la comparabilidad, y son el tipo de detalle que decide una ablación:

- **Backbone.** ResNet-18 con bloques vanilla aquí, Inception-v1 inflada allá; la distribución de canales y de FLOPs por etapa es muy distinta.
- **Interacción con el striding temporal.** Aquí, reemplazar 3D por 2D *elimina* el downsampling temporal de esa etapa. En una red top-heavy eso obliga a arrastrar tensores temporales grandes por las capas 2D iniciales; en la bottom-heavy, se comprimen temprano. Los dos papers no controlan esto igual.
- **Resolución y longitud.** $112 \times 112$ con 8–16 fotogramas aquí; $224 \times 224$ con 64 fotogramas en el linaje I3D/S3D. La utilidad del razonamiento temporal profundo depende de cuánta ventana haya.
- **Pre-entrenamiento.** Desde cero aquí, con inicialización ImageNet allá. Las capas tempranas heredadas de ImageNet son por construcción buenos extractores 2D de apariencia, lo que sesga a favor de dejarlas 2D.

La coda histórica es que **SlowFast** (Feichtenhofer et al., ICCV 2019) adoptó la postura de S3D: en su pathway *Slow* usa convoluciones temporales no degeneradas **solo en las etapas res4 y res5**, y reporta que ponerlas en las etapas tempranas degrada la exactitud. La moraleja no es que uno de los dos papers se equivocó, sino que **la ubicación óptima de la capacidad temporal no es una ley del dominio sino una propiedad del backbone, la resolución y el esquema de downsampling**, y hay que re-medirla al cambiar cualquiera de los tres.

## 6. R(2+1)D: la factorización formal

Aquí está el aporte arquitectónico. Se reemplazan los $N_i$ filtros 3D de tamaño $N_{i-1} \times t \times d \times d$ por un **bloque (2+1)D** compuesto de:

- $M_i$ filtros **espaciales** 2D de tamaño $N_{i-1} \times 1 \times d \times d$,
- seguidos de $N_i$ filtros **temporales** 1D de tamaño $M_i \times t \times 1 \times 1$,
- con un **ReLU intercalado** entre ambos.

El hiperparámetro $M_i$ es la dimensionalidad del subespacio intermedio donde se proyecta la señal entre la convolución espacial y la temporal. Se elige así:

$$M_i = \left\lfloor \frac{t\, d^2\, N_{i-1}\, N_i}{d^2\, N_{i-1} + t\, N_i} \right\rfloor$$

La derivación es de una línea y conviene hacerla para ver por qué la comparación es justa. La conv 3D tiene

$$P_{3D} = N_{i-1}\, t\, d^2\, N_i$$

parámetros, y el bloque (2+1)D tiene

$$P_{(2+1)D} = \underbrace{N_{i-1}\, d^2\, M_i}_{\text{espacial}} + \underbrace{M_i\, t\, N_i}_{\text{temporal}} = M_i\,(d^2 N_{i-1} + t N_i).$$

Igualando $P_{(2+1)D} = P_{3D}$ y despejando $M_i$ se obtiene exactamente la fórmula anterior. **Ese es el corazón metodológico del paper**: sin esa elección, cualquier mejora de R(2+1)D sobre R3D sería atribuible a tener más capacidad, y el experimento no diría nada. Con ella, R3D y R(2+1)D-18 tienen 33.4M y 33.3M parámetros respectivamente, y aproximadamente los mismos FLOPs, de modo que la diferencia de exactitud es atribuible a la **forma** de la convolución.

Un detalle contraintuitivo que la fórmula revela: **el subespacio intermedio es más ancho que la salida**. Para un bloque con $N_{i-1} = N_i = N$, $t = 3$, $d = 3$:

$$M_i = \frac{3 \cdot 9 \cdot N^2}{N(9 + 3)} = \frac{27N}{12} = 2.25\,N.$$

Es decir, en `conv5_x` (512 canales) el bloque usa **1152 filtros espaciales** de $1\times3\times3$ antes de proyectar a 512 con filtros temporales de $3\times1\times1$. Análogamente: 144 para bloques de 64→64, 230 para 64→128, 460 para 128→256, 921 para 256→512. La factorización no es un recorte: es un **re-gasto** del mismo presupuesto de parámetros, comprando ancho espacial y una no-linealidad extra a cambio de renunciar al acoplamiento conjunto espacio-tiempo dentro de un mismo filtro. Si hay striding espacial o temporal en la conv 3D original, se descompone correspondientemente en su dimensión.

Las **dos razones del beneficio** que argumenta el paper:

1. **Duplicar el número de no-linealidades.** El ReLU adicional entre la parte espacial y la temporal duplica las no-linealidades de la red *sin cambiar el número de parámetros*, lo que aumenta la complejidad de las funciones representables. El paper conecta explícitamente con VGG, que aproxima el efecto de un filtro grande apilando filtros chicos con no-linealidades intercaladas.
2. **Facilitar la optimización.** Forzar la separación en componentes espacial y temporal hace el problema más fácil de optimizar que un filtro 3D donde apariencia y dinámica están "conjuntamente entrelazadas". Esto es lo que la Sección 7 documenta.

El paper se sitúa cuidadosamente respecto de dos vecinos. **FSTCN** (Sun et al. 2015) factoriza la *red* (varias capas espaciales abajo, dos capas temporales paralelas arriba); R(2+1)D factoriza la *capa*, y por eso alterna espacial-temporal a lo largo de toda la red. **P3D** (Qiu et al. 2017) propone tres bloques distintos (espacial→temporal, espacial y temporal en paralelo, y espacial→temporal con skip) y los **intercala** en secuencia; su bloque P3D-A es el pariente más cercano, pero usa bottlenecks y **no está diseñado para igualar el conteo de parámetros con la conv 3D**. R(2+1)D es deliberadamente **homogéneo**: el mismo bloque en todas las capas, sin bottlenecks.

**Gotcha verificado sobre `conv1`.** El apéndice describe el stem: en lugar de 64 filtros 3D de $3\times7\times7$, R(2+1)D usa **45 filtros 2D de $1\times7\times7$ y 64 filtros 1D de $3\times1\times1$**, y afirma que tiene el mismo número de parámetros que R3D. Haciendo la cuenta, no cuadra: la conv 3D tiene $3\cdot64\cdot3\cdot49 = 28{,}224$ parámetros, mientras que el bloque con $M=45$ tiene $3\cdot45\cdot49 + 45\cdot64\cdot3 = 15{,}255$; la fórmula habría dado $M_1 = 83$ (28,137 parámetros, sí equiparable). O sea: en el stem la igualdad de parámetros es aproximada y el 45 es simplemente el número que quedó en la implementación de referencia — y que sobrevive hasta hoy en `torchvision` (`R2Plus1dStem`: `Conv3d(3, 45, (1,7,7))` → `Conv3d(45, 64, (3,1,1))`). Es un detalle menor en el conteo total, pero conviene saberlo antes de intentar reproducir la tabla de parámetros a mano.

La Figura 6 del apéndice visualiza los filtros aprendidos de `conv1`: los 45 filtros espaciales de $7\times7$ y los 64 filtros temporales, cada uno mostrado como una matriz $45 \times 3$ que expone cómo combina los 45 canales espaciales a través de los 3 fotogramas.

## 7. La evidencia sobre optimización

Este es el argumento más fino del paper, y el que lo distingue de una simple mejora empírica. La Figura 3 grafica error de **entrenamiento** (líneas finas) y de **validación** por época, para R3D y R(2+1)D, con 18 capas y con 34 capas, a lo largo de 45 épocas.

El resultado: **R(2+1)D tiene menor error de entrenamiento**, no solo menor error de test. Y la brecha en las pérdidas de entrenamiento es **particularmente grande en la red de 34 capas**.

Por qué importa la distinción. Si R(2+1)D solo tuviera menor error de *test*, la explicación natural sería **regularización**: la factorización restringe el espacio de hipótesis —impone que el filtro espacio-temporal sea separable, un subconjunto de rango bajo de los filtros 3D posibles— y esa restricción actúa como prior que reduce el sobreajuste. Sería una conclusión mucho más modesta.

Pero tener menor error de **entrenamiento** con **el mismo número de parámetros** descarta esa lectura: un modelo estrictamente menos expresivo no debería poder ajustar *mejor* los datos de entrenamiento. Que lo haga significa que el problema no es de capacidad sino de **optimizabilidad** — SGD encuentra mejores soluciones en la parametrización factorizada que en la conjunta. La factorización cambia el paisaje de la pérdida, no el conjunto de funciones alcanzables en el límite. Y que el efecto **crezca con la profundidad** (34 ≫ 18 capas) es coherente: las patologías de optimización se agravan con la profundidad, el mismo régimen donde las conexiones residuales fueron necesarias en imágenes.

La lección transferible: **cómo se parametriza un operador afecta al entrenamiento incluso a paridad de parámetros y de expresividad nominal**. Es la misma familia de fenómenos que las convoluciones separables en profundidad y las reparametrizaciones tipo RepVGG.

## 8. Setup experimental

**Datasets.** Cuatro benchmarks:

| Dataset | Tamaño | Clases | Rol |
|---|---|---|---|
| Sports-1M | 1.1M videos (>5 min promedio) | 487 deportes finos | pre-entrenamiento y benchmark |
| Kinetics | ~300K videos (~240K train) | 400 acciones humanas | benchmark principal (validación) |
| UCF-101 | ~13K videos | 101 | transferencia (3 splits) |
| HMDB-51 | ~6K videos | 51 | transferencia (3 splits) |

Kinetics y Sports-1M son los benchmarks primarios *porque son lo bastante grandes para entrenar modelos profundos desde cero*, que es el requisito del diseño experimental. En Kinetics se reporta sobre el **set de validación** (las anotaciones de test no son públicas); en UCF-101 y HMDB-51 se promedia sobre los **3 splits** estándar.

**Preprocesamiento y política de entrenamiento (desde cero).** Fotogramas escalados a $128 \times 171$, recortes aleatorios de $112 \times 112$. Se muestrean $L$ fotogramas consecutivos con *temporal jittering*. Batch normalization en todas las capas convolucionales. Mini-batch de **32 clips por GPU**. Aunque Kinetics tiene solo ~240K videos de entrenamiento, el **tamaño de época se fija en 1M** justamente para explotar el jittering temporal (cada pasada ve recortes temporales distintos del mismo video). Learning rate inicial **0.01**, dividido por 10 cada 10 épocas, con las **primeras 10 épocas de warm-up** para el entrenamiento distribuido (receta de Goyal et al., *Accurate, Large Minibatch SGD*). **45 épocas** en total, SGD distribuido síncrono en Caffe2.

**Política de fine-tuning.** Al ajustar en Kinetics un modelo pre-entrenado en Sports-1M: learning rate base 10× menor (**0.001**), reducido 10× cada 4 épocas, terminando en **15 épocas**.

**Longitudes de clip.** El estudio comparativo usa $L = 8$ y $L = 16$; el estudio de longitud extiende a 8, 16, 24, 32, 40 y 48 fotogramas. Como el pooling global no tiene parámetros aprendibles, **todos esos modelos tienen exactamente el mismo número de parámetros** y solo difieren en la longitud del input.

**Evaluación por clips.** Detalle fácil de pasar por alto que afecta a toda la lectura de las tablas. Se reportan **clip top-1** (sobre un clip individual) y **video top-1**, promediando las predicciones de **10 clips** con recortes centrales espaciados uniformemente en el video de Kinetics. En Sports-1M, donde los videos superan los 5 minutos de promedio, se usan **100 clips por video**. Comparar clip@1 de un paper con video@1 de otro es un error clásico: la brecha entre ambas es de 10–12 puntos.

**Flujo óptico.** Para el modelo de 34 capas se entrena una segunda corriente sobre flujo óptico y se fusionan los scores por promedio. Se usa el **método de Farnebäck** por su eficiencia — decisión que el paper reconoce que le cuesta exactitud frente al **TV-L1** que usa I3D, un orden de magnitud más lento.

## 9. Resultados

### 9.1. Kinetics: las cinco arquitecturas

La tabla de la Sección 5 ya recogió las cifras. R(2+1)D-18 supera a **MCx, rMCx y R3D** por **2.1–3.4%** con 8 fotogramas y por **3.1–4.7%** con 16; y a las **ResNets 2D** por **4.7–6.1%** y **6.3–9.8%** respectivamente. La Figura 4 grafica exactitud video top-1 contra FLOPs: R(2+1)D domina la frontera, con **3–3.8%** de ganancia sobre R3D **al mismo costo computacional**. El ranking relativo es **consistente** entre 8 y 16 fotogramas, pero las brechas son mayores con 16.

### 9.2. Longitud del clip: entrenar y testear a distintas longitudes

Con R(2+1)D-18 variando $L \in \{8, 16, 24, 32, 40, 48\}$: la **exactitud a nivel de clip sigue creciendo** al agregar fotogramas, pero la **exactitud a nivel de video hace pico en 32 fotogramas**. Como todos los modelos tienen los mismos parámetros, la pregunta natural es qué causa la diferencia. Dos experimentos la responden:

| Train (frames) | Finetune | Test (frames) | Tiempo entrenamiento (h) | Clip@1 | Video@1 |
|---|---|---|---|---|---|
| 8 | — | 8 | 11.8 | 52.8 | 64.8 |
| 8 | — | 32 | 11.8 | 51.6 | 59.0 |
| 32 | — | 32 | 59.8 | 60.1 | 69.4 |
| 8 | 32 | 32 | 20.5 | 59.8 | 68.0 |

(Tiempos medidos con 64 GPUs en paralelo.)

Los hallazgos:

1. **Alargar el clip solo en test es contraproducente.** Tomar el modelo entrenado con 8 fotogramas y evaluarlo con 32 **baja** 1.2% en clip y **5.8% en video**. La mejora no se obtiene gratis estirando el input en inferencia; hay que entrenar con la longitud objetivo. Esto es consistente con Varol et al. (LTC).
2. **Entrenar con clips largos produce modelos cualitativamente distintos**, cuyos filtros aprenden patrones temporales de mayor plazo. No es solo "ver más contexto".
3. **El atajo eficiente existe.** Hacer fine-tuning del modelo de 32 fotogramas inicializándolo con el de 8 llega a **59.8% vs 60.1%** de entrenar desde cero con 32 (un 7% de ganancia sobre el modelo de 8), pero en **20.5 h en lugar de 59.8 h** — porque el modelo de 8 fotogramas es **7.3× más rápido en FLOPs**. Es el mejor trade-off tiempo/exactitud de la tabla y una receta directamente aplicable.

Sobre cuántos clips promediar en inferencia: con el modelo de 32 fotogramas, usar **20 recortes está solo ~0.5% por debajo de usar 100**, y la predicción es **5× más rápida**.

### 9.3. Sports-1M (R(2+1)D-34)

| Método | Clip@1 | Video@1 | Video@5 |
|---|---|---|---|
| DeepVideo | 41.9 | 60.9 | 80.2 |
| C3D | 46.1 | 61.1 | 85.2 |
| 2D ResNet-152* | 46.5 | 64.6 | 86.4 |
| Conv pooling | — | 71.7 | 90.4 |
| P3D* | 47.9 | 66.4 | 87.4 |
| R3D-RGB-8frame | 53.8 | — | — |
| R(2+1)D-RGB-8frame | 56.1 | 72.0 | 91.2 |
| R(2+1)D-Flow-8frame | 44.5 | 65.5 | 87.2 |
| R(2+1)D-Two-Stream-8frame | — | 72.2 | 91.4 |
| R(2+1)D-RGB-32frame | **57.0** | 73.0 | 91.5 |
| R(2+1)D-Flow-32frame | 46.4 | 68.4 | 88.7 |
| R(2+1)D-Two-Stream-32frame | — | **73.3** | **91.9** |

(*cifras tomadas de P3D.)

R(2+1)D-RGB supera a **C3D por 10.9%** y a **P3D por 9.1%** en clip@1, y al 2D ResNet por 10.5% — pese a que ResNet-152 y P3D tienen **152 capas** contra las **34** de R(2+1)D (o 67, si se cuenta cada descomposición como dos capas). Contra su propio baseline R3D-34 con 8 fotogramas RGB, la ventaja es de **2.3%** (56.1 vs 53.8), lo que confirma el aporte de la descomposición controlando arquitectura y datos. El 73.3% video top-1 era el mejor resultado publicado en Sports-1M al momento.

### 9.4. Kinetics y la comparación con I3D — con sus asteriscos

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

Hay que leer esta tabla con cuidado, porque las tres comparaciones que permite son distintas:

1. **A paridad de "sin pre-entrenamiento", R(2+1)D gana claro.** RGB desde cero: **72.0 vs 67.5 = +4.5%**. Es la comparación más limpia del paper y la que sostiene la tesis: la factorización más residual vale más que la conv 3D inflada, cuando ninguno de los dos parte de ImageNet.
2. **Pre-entrenado en Sports-1M, R(2+1)D supera a I3D pre-entrenado en ImageNet**: **+2.2%** en RGB (74.3 vs 72.1) y **+3.2%** en flujo (68.5 vs 65.3). Pero acá el asterisco es doble y corta en las dos direcciones: R(2+1)D usa 1.1M videos de Sports-1M como fuente de pre-entrenamiento, mucho más *afín al dominio* que las imágenes de ImageNet, aunque también con etiquetas más ruidosas y restringidas a deportes.
3. **En two-stream, I3D sigue adelante por 0.3%** (75.7 vs 75.4). El propio paper lo admite. Y hay una explicación identificada: **R(2+1)D usa flujo de Farnebäck y I3D usa TV-L1**, más preciso. Es decir, parte de la diferencia final es de *preprocesamiento*, no de arquitectura.

La lectura honesta es entonces: **R(2+1)D es la mejor arquitectura de video entrenada desde cero de 2018, y su ventaja sobre I3D no está establecida cuando I3D puede usar ImageNet y buen flujo óptico.** Lo que sí queda establecido es que la dependencia de ImageNet, que en 2017 parecía indispensable para redes de video profundas, **dejó de serlo**.

### 9.5. Transferencia a UCF-101 y HMDB-51

| Método | Pre-entrenamiento | UCF-101 | HMDB-51 |
|---|---|---|---|
| Two-Stream | ImageNet | 88.0 | 59.4 |
| FSTCN | ImageNet | 88.1 | 59.1 |
| Conv Pooling | Sports-1M | 88.6 | — |
| P3D | ImageNet+Sports-1M | 88.6 | — |
| Two-Stream Fusion | ImageNet | 92.5 | 65.4 |
| Spatiotemporal ResNet | ImageNet | 93.4 | 66.4 |
| Temporal Segment Network | ImageNet | 94.2 | 69.4 |
| I3D-RGB | ImageNet+Kinetics | 95.6 | 74.8 |
| I3D-Flow | ImageNet+Kinetics | 96.7 | 77.1 |
| I3D-Two-Stream | ImageNet+Kinetics | **98.0** | **80.7** |
| R(2+1)D-RGB | Sports-1M | 93.6 | 66.6 |
| R(2+1)D-Flow | Sports-1M | 93.3 | 70.1 |
| R(2+1)D-Two-Stream | Sports-1M | 95.0 | 72.7 |
| R(2+1)D-RGB | Kinetics | 96.8 | 74.5 |
| R(2+1)D-Flow | Kinetics | 95.5 | 76.4 |
| R(2+1)D-Two-Stream | Kinetics | 97.3 | 78.7 |

R(2+1)D supera a **todos** los métodos de la comparación excepto I3D, que además usa ImageNet. Dos observaciones concretas:

- **Kinetics es mejor fuente de pre-entrenamiento que Sports-1M**, y por márgenes grandes: +3.2 puntos en UCF-101 (96.8 vs 93.6 en RGB) y +7.9 en HMDB-51 (74.5 vs 66.6). Kinetics es más chico pero mucho más limpio y diverso en acciones humanas; Sports-1M es más grande pero acotado a deportes y con etiquetas débiles. Es una lección sobre calidad vs cantidad en pre-entrenamiento. (Los modelos de las filas "Kinetics" son los entrenados desde cero en Kinetics, no los ajustados desde Sports-1M, precisamente para aislar el efecto del dataset.)
- La brecha residual con I3D-Two-Stream (97.3 vs 98.0 y 78.7 vs 80.7) es consistente con las dos desventajas ya identificadas: **sin ImageNet** y **con flujo de peor calidad**.

## 10. Limitaciones

- **El costo de entrenamiento sigue siendo alto.** No es un problema exclusivo de I3D. La Tabla 3 lo cuantifica sin adornos: entrenar R(2+1)D-18 con clips de 32 fotogramas en Kinetics toma **59.8 horas con 64 GPUs en paralelo** — del orden de 3800 GPU-horas para un modelo de solo 18 capas. Con 8 fotogramas baja a 11.8 h y el fine-tuning progresivo lo deja en 20.5 h, pero el piso de entrada al campo sigue siendo un cluster.
- **El mejor resultado depende del flujo óptico.** Igual que I3D, R(2+1)D necesita una segunda corriente sobre flujo pre-computado, un paso costoso fuera de la red que contradice la promesa end-to-end. Y el balance es pobre: en Kinetics con pre-entrenamiento Sports-1M, RGB solo da 74.3 contra 75.4 de two-stream — la fusión aporta **1.1 punto** a cambio de duplicar el cómputo y agregar un pipeline de flujo, que además (Farnebäck) es menos preciso que el TV-L1 de I3D.
- **La ventana temporal es corta.** La exactitud a nivel de video **hace pico en 32 fotogramas**, poco más de un segundo a 25 fps. El modelo no razona sobre estructura de largo plazo: más allá de esa ventana la agregación es un promedio de predicciones de clips independientes, sin ningún modelo del orden entre clips.
- **La falta de pre-entrenamiento en ImageNet es una desventaja estructural.** La factorización no ofrece nada análogo al *boring-video fixed point* de I3D para heredar pesos 2D. Se podría inicializar la parte espacial $1\times d\times d$ desde una red 2D, pero el ancho intermedio $M_i = 2.25 N_i$ no coincide con los canales de ninguna ResNet de imagen, así que el mapeo no es directo. Sin un dataset de video grande disponible, R(2+1)D queda en desventaja frente a las arquitecturas infladas.
- **Alcance del estudio.** Los autores lo delimitan: un solo tipo de red (ResNet) y un uso **homogéneo** de la descomposición en todas las capas. Dejan para trabajo futuro buscar arquitecturas más adecuadas al bloque (2+1)D — y no exploran combinar la factorización con la asignación no uniforme de capacidad temporal que sus propias secciones MC/rMC insinúan.

## 11. Impacto y legado

R(2+1)D se convirtió en el **backbone de video por defecto** de los años siguientes, por una razón muy pragmática: es un ResNet. Cualquier ingeniero que sabía trabajar con ResNets podía usarlo, y la factorización se implementa con dos `Conv3d` de kernels degenerados, sin operadores nuevos. Está disponible en **`torchvision.models.video.r2plus1d_18`** con pesos pre-entrenados en Kinetics-400, junto a `r3d_18` y `mc3_18` — es decir, tres de las arquitecturas de este paper llegaron directamente a la librería estándar de PyTorch, algo poco común para un paper de ablaciones. Facebook liberó además el código y los modelos en el repositorio **VMZ (Video Model Zoo)**.

Su lugar en la historia de las ideas es compartido con **S3D**: los dos papers, publicados en el mismo año y desde tradiciones distintas (ResNet vs Inception), **consolidaron la separabilidad espacio-temporal** como principio de diseño. Después de 2018, prácticamente ningún modelo de video usa convoluciones 3D densas sin factorizar. Du Tran llevó la idea un paso más allá en **CSN (Channel-Separated Convolutional Networks**, ICCV 2019), donde separa además el eje de canales, obteniendo redes 3D con exactitud comparable a costos mucho menores; y en **IG-65M** (Ghadiyaram, Tran, Mahajan, CVPR 2019) usó R(2+1)D como backbone para pre-entrenamiento débilmente supervisado sobre 65 millones de videos de Instagram — que es, en cierto sentido, la respuesta definitiva a la desventaja de no tener ImageNet: si el problema era la falta de pre-entrenamiento a escala, se construye uno de video.

**SlowFast** (Feichtenhofer et al. 2019) recogió el otro hilo: la pregunta de *dónde* poner la capacidad temporal. Su respuesta —dos pathways con resoluciones temporales distintas y convoluciones temporales no degeneradas solo en las etapas profundas del pathway lento— mostró que la asignación **no uniforme** de capacidad temporal, insinuada por las secciones MC/rMC de este paper, era la dirección fértil.

Y el linaje llega a los **video transformers** de forma bastante literal. **TimeSformer** (Bertasius, Wang, **Torresani**, ICML 2021) —con un co-autor de este mismo paper— propone *divided space-time attention*: atender primero en el espacio y luego en el tiempo, en operaciones separadas dentro de cada bloque. Es la factorización de R(2+1)D trasladada de la convolución a la atención, con la misma justificación: mismo presupuesto, mejor optimizable, más barato que la atención espacio-temporal conjunta. **ViViT** (Arnab et al. 2021) explora la misma familia con su *factorised encoder*. La lección —**factorizar el operador espacio-temporal es casi siempre mejor que aplicarlo de forma conjunta**— sobrevivió al cambio completo de arquitectura.

## 12. Conexión con la Clase 38

La Clase 38 recorre la escalera CNN2D + temporal pooling → CNN2D + RNN → Two-Stream → C3D → I3D. Este paper se inserta en dos lugares a la vez, y ahí está su valor didáctico.

**Primero: es la revisión crítica del propio autor de C3D.** Du Tran firmó C3D en 2015 y firma esto en 2018. Vale contrastar qué desventajas de C3D que la clase enumera quedan resueltas y cuáles no:

| Desventaja de C3D | ¿Resuelta en R(2+1)D? |
|---|---|
| Poca profundidad (8 capas conv) | **Sí.** 18 y 34 capas gracias al aprendizaje residual, y la factorización hace que la profundidad ayude más (la brecha de error de entrenamiento con R3D crece de 18 a 34 capas). |
| Demasiados parámetros / costo alto | **Parcialmente.** A paridad de parámetros con R3D (33.3M vs 33.4M), R(2+1)D es más preciso; pero el conteo absoluto no baja. La variante que sí abarata es MCx (11.4M igualando a R3D). Y el entrenamiento sigue costando decenas de horas en 64 GPUs. |
| Sin pre-entrenamiento en ImageNet | **No, y es asumido como precio.** Se compensa con datasets de video grandes (Sports-1M, Kinetics). |
| Ventana temporal corta (16 fotogramas) | **Marginalmente.** Se extiende a 32, donde la exactitud a nivel de video hace pico; más allá no ayuda. El razonamiento de largo plazo sigue pendiente. |
| Convolución 3D monolítica difícil de optimizar | **Sí.** Es la contribución central: menor error de entrenamiento a igual capacidad. |

**Segundo: relaciona las dos estrategias rivales para el mismo problema.** El problema es idéntico en ambos papers: *obtener capacidad espacio-temporal sin pagar el costo completo de una conv 3D densa entrenada desde cero*. Las respuestas son alternativas, no incompatibles:

- **I3D — inflado.** Ataca el lado de los *datos y la inicialización*. Mantiene la conv 3D densa y consigue que sus pesos no partan de ruido, replicando en el tiempo los filtros 2D de ImageNet y dividiendo por $N$. La arquitectura no se cuestiona; se importa el conocimiento.
- **R(2+1)D — factorización.** Ataca el lado del *operador y la optimización*. No importa nada, pero rediseña el bloque para que el mismo presupuesto de parámetros se distribuya en un subespacio espacial más ancho ($2.25\times$) con una no-linealidad extra, y para que SGD lo entrene mejor.

Y el veredicto de la historia es que **se combinaron**: la práctica posterior usa bloques factorizados (R(2+1)D, S3D, CSN) *y* pre-entrenamiento masivo (Kinetics, IG-65M, y luego los pre-entrenamientos de los video transformers). La clase presenta como desventaja de I3D que tiene demasiados parámetros y es costoso; este paper es la respuesta directa a esa objeción, y su forma de responder —controlar la capacidad, aislar la variable, medir el error de entrenamiento y no solo el de test— es tan instructiva como el modelo que produce.

---

**Nota final — relevancia para video clínico.** Tres cosas de este paper se trasladan casi sin cambios a un pipeline de video médico (endoscopía, ecocardiografía, análisis de marcha, gestos quirúrgicos). Primero, la receta de eficiencia de la Tabla 3: **entrenar con clips cortos y luego hacer fine-tuning con clips largos** recupera el 99.5% de la exactitud a un tercio del tiempo de cómputo — con presupuesto de GPU acotado, es la diferencia entre poder iterar y no poder. Segundo, que **Kinetics supere a Sports-1M como fuente de pre-entrenamiento pese a ser 3.6× más chico**: en dominios clínicos la tentación es acumular más video sin curar, y acá hay evidencia cuantitativa de que la afinidad y la limpieza del dataset fuente pesan más que el tamaño bruto. Tercero, la separación entre exactitud a nivel de clip y a nivel de video es exactamente la distinción que importa clínicamente: un modelo puede acertar el 57% de los clips y el 73% de los estudios, y cuál de las dos cifras se reporta cambia por completo la lectura de su utilidad diagnóstica.
