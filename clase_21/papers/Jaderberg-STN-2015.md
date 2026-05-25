---
title: "Spatial Transformer Networks"
authors: ["Max Jaderberg", "Karen Simonyan", "Andrew Zisserman", "Koray Kavukcuoglu"]
year: 2015
venue: "NeurIPS 2015"
slug: "stn-jaderberg-2015"
arxiv: "1506.02025"
affiliation: "Google DeepMind"
tags: ["spatial-transformer", "differentiable-sampling", "attention", "scene-text-recognition", "fine-grained-recognition", "tps", "affine", "deepmind"]
---

# Spatial Transformer Networks (Jaderberg, Simonyan, Zisserman, Kavukcuoglu — NeurIPS 2015)

## Resumen ejecutivo

Las CNNs convencionales son extraordinariamente potentes para visión, pero su invariancia geométrica es limitada y rígida: el max-pooling proporciona invariancia local en ventanas pequeñas (típicamente $2 \times 2$), la convolución comparte pesos para tolerar traslaciones, y el data augmentation aporta rotaciones o zooms ad-hoc. Ninguno de estos mecanismos permite a la red **descubrir y aplicar** la transformación geométrica que conviene a cada muestra concreta.

El **Spatial Transformer (ST)** propuesto por Jaderberg et al. (DeepMind, NeurIPS 2015) introduce un módulo diferenciable, sin parámetros fijos en tiempo de inferencia, que **regresa los parámetros de una transformación espacial condicionada en el input** y la aplica al feature map antes de pasar a las capas siguientes. El módulo se compone de tres piezas: una **localisation network** que predice $\theta$, un **grid generator** que mapea una grilla regular del output a coordenadas del input vía $T_\theta$, y un **sampler** que extrae los valores del input mediante interpolación bilineal diferenciable. Todo el módulo se entrena por backprop usando únicamente la loss de la tarea final — no se necesitan etiquetas de transformación.

Los autores demuestran ganancias sustanciales en cuatro escenarios: (i) clasificación de MNIST distorsionado (rotación, traslación, escala, perspectiva, deformación elástica), (ii) reconocimiento multi-dígito en Street View House Numbers (SVHN) con ST anidados en el stack convolucional, (iii) clasificación fine-grained de aves en CUB-200-2011 con múltiples STs paralelos que descubren partes (cabeza, cuerpo) sin supervisión adicional, y (iv) co-localización semi-supervisada con loss de triplete sobre embeddings. El ST se convierte rápidamente en una primitiva canónica de la visión profunda: aparece como bloque de pre-procesamiento en **scene text recognition** (RARE, ASTER, MORAN), inspira las **Deformable Convolutions** y los **Dynamic Filter Networks**, y prefigura conceptualmente la atención espacial usada en Vision Transformers y la diferenciabilidad de samplers en NeRF.

Para el curso, este paper es un prerrequisito ineludible para entender la slide de la clase 21 *"Text Recognition — Image Preprocessing Stage: STN, TPS, Other networks"*: la rectificación de texto curvado antes del recognizer (CRNN, ASTER, MORAN) usa exactamente el sampler bilineal + TPS introducidos aquí.

## Contexto histórico

A mediados de 2015, las CNNs ya dominaban clasificación (AlexNet 2012, VGG 2014, GoogLeNet/Inception 2014), detección (R-CNN, Fast R-CNN), segmentación (FCN 2015) y reconocimiento de acciones. Pero subsistía una brecha conceptual: **¿cómo introduce una CNN invariancia o equivariancia a transformaciones grandes y globales del input?** Las respuestas existentes eran insuficientes o costosas:

**Max-pooling local.** El pooling $2 \times 2$ provee invariancia translacional dentro de la ventana de pooling. Como las capas se apilan, el campo receptivo crece y la invariancia efectiva también, pero sólo de forma jerárquica y limitada — no maneja rotaciones de $30^\circ$ ni cambios de escala $\times 2$ sin pagar un costo en discriminabilidad. Cohen y Welling (ICLR 2015) y Lenc–Vedaldi (CVPR 2015) miden empíricamente que las representaciones intermedias de CNNs no son fuertemente invariantes a transformaciones grandes; son aproximadamente equivariantes con error creciente.

**Convolution weight sharing.** La convolución comparte filtros a lo largo del eje espacial, lo que produce equivariancia translacional exacta (en teoría). No produce equivariancia rotacional ni de escala.

**Data augmentation.** Augmentar el training set con rotaciones, zooms, recortes aleatorios es la solución práctica dominante. Pero (a) infla el dataset, (b) requiere conocer las transformaciones esperadas a priori, y (c) no permite a la red **decidir en inferencia** qué transformación aplicar a la muestra de turno.

**Capsules y transforming auto-encoders (Hinton 2011, Tieleman 2014).** Primer intento serio de equivariancia explícita: cada cápsula codifica la pose de una parte. Conceptualmente potente, pero los modelos de la época no escalaban a problemas grandes ni eran fácilmente diferenciables end-to-end.

**Scattering networks (Bruna–Mallat 2013), invariant scattering filters, locally scale-invariant convolutions (Kanazawa 2014), Gens–Domingos deep symmetry networks (NIPS 2014).** Construyen invariancia por diseño matemático (wavelets, grupos de simetría). Elegantes, pero rígidos: la invariancia se baja al modelo manualmente.

**Hard/soft attention con RL (Mnih 2014 RAM, Ba 2015 DRAM, Xu 2015 Show-Attend-Tell).** Atención sobre regiones del input vía recorte. La variante "hard" requiere REINFORCE (gradientes de alta varianza); la "soft" gaussiana de DRAW (Gregor 2015) era diferenciable pero limitada a kernel gaussiano.

**El gap.** No existía un módulo (a) plenamente diferenciable, (b) entrenable end-to-end por backprop estándar, (c) capaz de aplicar una transformación espacial **arbitraria y aprendida** sobre el feature map y (d) que no requiriera supervisión de la transformación. STN cierra este gap con elegancia: factoriza la transformación en *predicción de parámetros + grilla geométrica + sampler bilineal*, todo diferenciable. La idea es tan natural que en retrospectiva parece obvia, pero a la fecha del paper era una contribución arquitectónica clara y reusable.

## Arquitectura del Spatial Transformer

El ST es un sub-módulo que recibe un feature map $U \in \mathbb{R}^{H \times W \times C}$ y produce otro $V \in \mathbb{R}^{H' \times W' \times C}$, aplicando una transformación geométrica $T_\theta$ aprendida y condicionada en el propio $U$. Tres componentes:

### Localisation network

Es una red pequeña $f_{\text{loc}}: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^d$ que regresa los parámetros $\theta = f_{\text{loc}}(U)$ de la transformación. La dimensión $d$ depende de la familia de transformaciones elegida:

- Afín 2D: $d = 6$.
- Proyectiva (homografía): $d = 8$.
- Atención (translación + zoom isotrópico): $d = 3$.
- Thin-Plate Spline (TPS) con $K$ puntos de control: $d = 2K$ (típicamente $K = 16$).

La arquitectura interna es libre: en el paper se usan FCs para inputs MNIST y mezclas conv + FC para inputs más grandes (SVHN, CUB). El detalle crítico es la **inicialización del último layer**: pesos a cero y bias igual a la transformación identidad. Así el ST arranca como un cable transparente y descubre transformaciones útiles gradualmente.

### Grid generator

Define una grilla regular en coordenadas del output $G = \{G_i\} = \{(x_i^t, y_i^t)\}$ con $H' \cdot W'$ puntos. Las coordenadas se normalizan a $[-1, 1]$ (height y width independientes). Aplica la transformación $T_\theta$ para obtener las **coordenadas fuente** en el input:

$$
(x_i^s, y_i^s) = T_\theta(G_i)
$$

Para una afín 2D:

$$
\begin{pmatrix} x_i^s \\ y_i^s \end{pmatrix}
= A_\theta \begin{pmatrix} x_i^t \\ y_i^t \\ 1 \end{pmatrix}
= \begin{pmatrix} \theta_{11} & \theta_{12} & \theta_{13} \\ \theta_{21} & \theta_{22} & \theta_{23} \end{pmatrix}
\begin{pmatrix} x_i^t \\ y_i^t \\ 1 \end{pmatrix}
$$

Nota conceptual importante: la transformación va **de output a input** (backward warping). Para cada pixel del output, se calcula desde qué posición del input se debe muestrear. Es el mismo esquema que en computer graphics (texture mapping) — evita huecos en el output.

### Sampler

Para cada coordenada fuente $(x_i^s, y_i^s)$, el sampler extrae un valor del input. El paper define un kernel genérico:

$$
V_i^c = \sum_{n=1}^{H} \sum_{m=1}^{W} U_{nm}^c \, k(x_i^s - m; \Phi_x) \, k(y_i^s - n; \Phi_y)
$$

Dos opciones explícitas:

1. **Nearest neighbour** (no diferenciable en $x_i^s, y_i^s$ — sólo discretiza):
$$
V_i^c = \sum_{n,m} U_{nm}^c \, \delta(\lfloor x_i^s + 0.5 \rfloor - m) \, \delta(\lfloor y_i^s + 0.5 \rfloor - n)
$$

2. **Bilinear** (la opción usada en todos los experimentos, sub-diferenciable):
$$
V_i^c = \sum_{n=1}^{H} \sum_{m=1}^{W} U_{nm}^c \, \max(0, 1 - |x_i^s - m|) \, \max(0, 1 - |y_i^s - n|)
$$

Los gradientes se calculan analíticamente:

$$
\frac{\partial V_i^c}{\partial U_{nm}^c} = \max(0, 1 - |x_i^s - m|) \, \max(0, 1 - |y_i^s - n|)
$$

$$
\frac{\partial V_i^c}{\partial x_i^s} = \sum_{n,m} U_{nm}^c \, \max(0, 1 - |y_i^s - n|) \cdot
\begin{cases}
0 & \text{si } |m - x_i^s| \geq 1 \\
+1 & \text{si } m \geq x_i^s \\
-1 & \text{si } m < x_i^s
\end{cases}
$$

y análogamente para $\partial V_i^c / \partial y_i^s$. Estos gradientes fluyen hacia atrás a través del grid generator hasta $\theta$ — calcular $\partial x_i^s / \partial \theta$ es trivial porque $T_\theta$ es una composición de matrices conocidas. La sumatoria $H \times W$ en la práctica se evalúa sólo en los 4 vecinos del pixel fuente, así que es $O(1)$ por pixel del output, perfecta para GPU.

El sampling se aplica idénticamente en cada canal: $V^c$ usa el mismo grid para todos los $C$ canales — esto preserva consistencia espacial inter-canal.

## Tipos de transformación parametrizada

El ST es agnóstico a la familia $T_\theta$ siempre que ésta sea diferenciable en $\theta$. El paper discute y experimenta con varias:

**Afín 2D (6 parámetros).** Cubre traslación, escala anisotrópica, rotación y shear. Si el determinante de la sub-matriz $2 \times 2$ es menor a 1, el grid se contrae y el efecto es un crop con zoom. Es la default en SVHN, CUB y muchos casos.

**Atención (3 parámetros).**
$$
A_\theta = \begin{pmatrix} s & 0 & t_x \\ 0 & s & t_y \end{pmatrix}
$$
Equivale a un crop rectangular centrado en $(t_x, t_y)$ con zoom isotrópico $s$. Es la versión más restringida — útil cuando se sabe que la transformación deseada es sólo "mirar acá con este zoom".

**Proyectiva / homografía (8 parámetros).** Cubre perspectiva. Útil cuando el input contiene texto o objetos vistos desde ángulos oblicuos.

**Thin-Plate Spline (TPS) con $K$ puntos de control (Bookstein 1989).** Parametrización de $2K$ coordenadas que describe **deformaciones no rígidas**. La transformación se construye como suma de un término afín más una serie de funciones radiales $\phi(r) = r^2 \log r^2$ centradas en los puntos de control. Es **la opción correcta para rectificar texto curvado** o caracteres elásticamente deformados — el ST aprende a "des-curvar" la entrada antes del recognizer. Es el bloque base de RARE, ASTER y MORAN.

**Forma estructurada $T_\theta = M_\theta B$.** El paper sugiere generalizaciones donde la grilla base $B$ también se aprende además de $\theta$, abriendo la puerta a familias compuestas o piecewise-affine.

**Caso 3D (apéndice A.3).** Extensión natural: $\theta$ es una matriz $3 \times 4$ afín en $\mathbb{R}^3$, el sampler usa interpolación trilineal. Los autores demuestran un clasificador MNIST 3D donde el dígito está embebido y rotado en un volumen $60^3$; el ST aprende a proyectar el volumen a 2D para que las capas siguientes clasifiquen.

## Propiedades clave

**Diferenciable end-to-end.** Toda la pipeline (loss de tarea $\to$ sampler $\to$ grid $\to$ localisation net) admite backprop estándar. No se necesita REINFORCE, no se necesita supervisión de transformación.

**Modular y "drop-in".** Se puede insertar en cualquier punto de cualquier red, no sólo al input. Insertarlo en capas intermedias permite warpear feature maps abstractos. En SVHN-Multi los autores apilan 4 STs intercalados con capas conv.

**No requiere etiquetas de transformación.** La supervisión proviene únicamente de la loss final (cross-entropy de clasificación, triplet loss para co-localización, etc.). La red descubre por gradient descent qué transformación reduce la loss.

**Múltiples STs paralelos.** En CUB-200, $K = 2$ y $K = 4$ STs paralelos atienden cada uno a una parte distinta del ave. Cada uno produce un crop $224 \times 224$ que alimenta una Inception independiente; los descriptores se concatenan y clasifican con un softmax 200-way. Los STs co-adaptan: uno se especializa en cabeza, otro en cuerpo, sin etiquetas de keypoint.

**Múltiples STs en serie.** Permiten transformaciones cada vez más abstractas — el ST de capa profunda opera sobre features semánticas y no sobre píxeles.

**Costo computacional bajo.** En SVHN-Multi el ST-CNN Multi (4 STs) es sólo ~6% más lento que la baseline CNN. El sampler bilineal evaluado en GPU es trivial.

**Downsampling implícito útil.** Como el output $V$ puede tener resolución $H' \times W'$ distinta a la de $U$, el ST puede recortar y reducir simultáneamente — en CUB usan crops $448 \to 224$.

## Experimentos del paper

### MNIST distorsionado (Tabla 1)

Cuatro datasets con dígitos distorsionados:
- **R**: rotación uniforme entre $-90^\circ$ y $+90^\circ$.
- **RTS**: rotación $\pm 45^\circ$ + escala $[0.7, 1.2]$ + traslación dentro de canvas $42 \times 42$.
- **P**: distorsión proyectiva (perturbación normal de las esquinas).
- **E**: deformación elástica (TPS aplicado al input con $\sigma = 1.5$ px).

Modelos: FCN, CNN, ST-FCN (FCN con ST de entrada), ST-CNN. El ST opera con afín (Aff), proyectiva (Proj) o TPS de 16 puntos. Resultados (error %):

| Modelo | R | RTS | P | E |
|---|---|---|---|---|
| FCN | 2.1 | 5.2 | 3.1 | 3.2 |
| CNN | 1.2 | 0.8 | 1.5 | 1.4 |
| ST-FCN Aff | 1.2 | 0.8 | 1.5 | 2.7 |
| ST-FCN Proj | 1.3 | 0.9 | 1.4 | 2.6 |
| ST-FCN TPS | 1.1 | 0.8 | 1.4 | 2.4 |
| ST-CNN Aff | 0.7 | 0.5 | 0.8 | 1.2 |
| ST-CNN Proj | 0.8 | 0.6 | 0.8 | 1.3 |
| ST-CNN TPS | 0.7 | 0.5 | 0.8 | **1.1** |

Observaciones clave:
- ST-CNN supera a CNN en cada distorsión.
- TPS gana en datos elásticamente deformados (E) porque puede revertir deformaciones no rígidas.
- Cuando se combina ST con FCN (sin convolución ni pooling), ST-FCN alcanza el error del CNN baseline — el ST proporciona invariancia espacial que la FCN no tiene de fábrica.
- En entorno con clutter (canvas $60 \times 60$ con dígitos trasladados + 6 distractores), FCN logra 13.2% error, CNN 3.5%, ST-FCN 2.0%, **ST-CNN 1.7%**.

### Street View House Numbers (Tabla 2)

Dataset SVHN multi-dígito (~200k imágenes, secuencias de 1–5 dígitos por imagen, escalas y arreglos espaciales muy variables). Crops $64 \times 64$ y $128 \times 128$. Modelo baseline: CNN de 11 capas con 5 softmax independientes (uno por posición). El ST-CNN Multi inserta un ST antes de cada una de las primeras 4 capas conv; la **localisation network es minimalista** (fc[32]-fc[32]).

Error (% secuencia):

| Modelo | 64px | 128px |
|---|---|---|
| Maxout CNN (Goodfellow 2013) | 4.0 | – |
| CNN (baseline propia) | 4.0 | 5.6 |
| DRAM (Ba 2015) con MC averaging | 3.9 | 4.5 |
| ST-CNN Single | 3.7 | 3.9 |
| ST-CNN Multi | **3.6** | **3.9** |

ST-CNN supera a DRAM (modelo recurrente con atención reforzada por RL y ensemble) usando una sola pasada forward. Visualizando las afines predichas en cada ST, se observa que cada uno recorta progresivamente la región relevante de la secuencia.

### CUB-200-2011 fine-grained birds (Tabla 3)

Backbone: Inception con batch norm, pre-entrenado en ImageNet. Baseline a $224$ px: 82.3% top-1. Se entrenan modelos con 2 o 4 STs paralelos (cada uno con localisation net derivada de Inception); cada ST produce un crop $224 \times 224$ que alimenta una Inception independiente; los descriptores 1024-D se concatenan y clasifican con softmax 200-way. Resultado (accuracy %):

| Modelo | Accuracy |
|---|---|
| Cimpoi '15 | 66.7 |
| Zhang '14 (part R-CNN) | 74.9 |
| Branson '14 | 75.7 |
| Lin '15 (bilinear CNN) | 80.9 |
| Simon '15 (constellations) | 81.0 |
| CNN baseline 224px | 82.3 |
| 2×ST-CNN 224px | 83.1 |
| 2×ST-CNN 448px | 83.9 |
| 4×ST-CNN 448px | **84.1** |

Visualización: en 2×ST-CNN, un transformer aprende a detectar **cabezas** y el otro a fijarse en el **cuerpo central**, sin ninguna anotación de keypoint. Es localización emergente sólo por gradient descent sobre cross-entropy de clase.

### Co-localización (apéndice A.2, Tabla 5)

Escenario semi-supervisado: dado un conjunto de imágenes con objetos de clase común pero desconocida, localizar el objeto en cada una. Se usa **triplet loss con hinge margin** sobre embeddings de crops producidos por el ST:

$$
\sum_{n=1}^{N} \sum_{m \neq n}^{M} \max\!\Big(0,\; \|e(I_n^{T}) - e(I_m^{T})\|_2^2 - \|e(I_n^{T}) - e(I_n^{\text{rand}})\|_2^2 + \alpha\Big)
$$

donde $I_n^T = T_\theta(I_n)$ es el crop producido por el ST e $I_n^{\text{rand}}$ es un parche aleatorio. Resultado: en MNIST de un dígito por imagen sobre canvas $84 \times 84$, el ST localiza correctamente el 100% en el caso translated y entre 75–94% en el caso con clutter, sin ningún label de bounding box. La Fig. 4 muestra la convergencia del ST a la posición del dígito a lo largo de 180 pasos SGD.

### MNIST addition (apéndice A.1, Tabla 4)

Dos dígitos transformados independientemente en canales separados de un input $42 \times 42 \times 2$. Tarea: predecir la suma (19 clases). Con dos STs paralelos, cada uno se especializa en un canal — los outputs concatenados van a un FCN. Error: FCN 47.7%, CNN 14.7%, ST-FCN 18.5%, **2×ST-FCN TPS 5.8%**.

### MNIST 3D (apéndice A.3)

Extensión a 3D: voxel input $60^3$ con un dígito MNIST extruido y rotado en 3D. El ST aprende una transformación afín $3 \times 4$ y proyecta el volumen 3D a una imagen 2D que las capas siguientes clasifican.

## Detalles de implementación reportados en el apéndice

El apéndice del paper documenta los hiperparámetros y arquitecturas exactas, lo que es valioso para reproducir y para entender los compromisos prácticos. Resumo los puntos no triviales:

**MNIST distorsionado (A.4).** Todas las redes usan ReLU + softmax. Las FCN tienen dos hidden layers FC más una capa de clasificación. Las CNN tienen una capa conv $9 \times 9$ stride 1 sin padding, max-pool $2 \times 2$ stride 2, conv $7 \times 7$ stride 1 sin padding, otra max-pool $2 \times 2$ stride 2, FC final. Las localisation networks del ST-FCN tienen 3 capas FC; las del ST-CNN tienen 2 capas conv $5 \times 5$ con 20 filtros sobre input $2\times$ downsampled, max-pool $2 \times 2$ y FC[20]. Para los datasets TC y RTS se aplica average pooling tras el ST para downsamplear por factor 2 antes de la red de clasificación, mitigando aliasing. Todas las redes se entrenan 150k iteraciones con SGD, batch 256, lr base 0.01 con scheduled decay $\times 0.1$ cada 50k iteraciones, sin weight decay, sin dropout. El reporte es el promedio de 3 corridas con seeds distintas.

**SVHN (A.5).** Hiperparámetros tunados sobre 5k imágenes de validación. 400k iteraciones SGD batch 128, lr base 0.01 con $\times 0.1$ cada 80k, weight decay $5 \times 10^{-4}$, dropout 0.5 en todas las capas salvo la primera conv y las localisation networks. La lr de las localisation nets es **una décima** de la lr base — detalle crítico para que el ST converja sin destruir las features. Backbone: conv[48,5,1,2]-max[2]-conv[64,5,1,2]-conv[128,5,1,2]-max[2]-conv[160,5,1,2]-conv[192,5,1,2]-max[2]-conv[192,5,1,2]-conv[192,5,1,2]-max[2]-conv[192,5,1,2]-fc[3072]-fc[3072] + 5 softmax fc[11] paralelos. El ST-CNN Single inserta un ST con localisation `conv[32,5,1,2]-max[2]-conv[32,5,1,2]-fc[32]` antes de la primera conv del backbone. El ST-CNN Multi inserta 4 STs, cada uno con localisation `fc[32]-fc[32]` — **minimalismo extremo** justamente porque actúan sobre feature maps ya descriptivas, no sobre píxeles crudos.

**CUB-200 (A.6).** Backbone: Inception + BN preentrenado en ImageNet (27.1% top-1 ILSVRC val), fine-tuned en CUB. Los STs son **atención (location + scale, scale fijo a 50%)**, no afín general. Cada ST muestrea un crop $224 \times 224$ del input ($224$ o $448$ px). La localisation network es compartida entre todos los transformers (output $2N$-D donde $N$ es el número de STs) y deriva de Inception removiendo la última capa de pooling para conservar resolución $7 \times 7 \times 1024$, seguida de conv $1 \times 1$ con 128 canales, FC[128], FC[$2N$]. La capa final está inicializada para mosaiquear el plano espacial — i.e., cada ST se inicializa apuntando a una región distinta de la imagen, lo que previene colapso a la misma región. Lr base 0.1 con $\times 0.1$ a 10k/20k/25k iteraciones; lr de localisation $\times 10^{-4}$; weight decay $10^{-5}$; dropout 0.7 antes del softmax 200-way. Augmentation: random sampling $224$ de $256$ side, horizontal flip. Los autores probaron también transformaciones más complejas (afín, location+scale) y observaron **overfitting severo** dado el tamaño pequeño del dataset (6k train).

**Co-localización (A.2).** Encoding $e()$: CNN preentrenada en MNIST clasificación, se concatenan activaciones de las 3 layers (sin softmax) como descriptor. ST con atención (scale + translation); localisation: CNN de ~100k parámetros con 8 filtros $9 \times 9$ stride 4, max-pool $2 \times 2$ stride 2, FC[8]-FC[8]-FC[3]. Margin de triplet $\alpha = 1$. Para cada clase de dígito, se generan 100 muestras distorsionadas y se optimiza la transformación SGD muestreando pares $(n, m)$. Métrica: IoU $> 0.5$ entre bbox predicho y groundtruth.

Estos detalles refuerzan tres lecciones prácticas: **(a)** la lr de la localisation network suele ir muy por debajo de la lr base (por factor 10 en SVHN, $10^{-4}$ en CUB) — sin esto, el ST oscila y rompe entrenamiento; **(b)** la **inicialización a identidad** se usa cuando hay 1 ST; cuando hay varios STs paralelos, se inicializan **a posiciones distintas** para evitar que todos converjan a la misma región; **(c)** las localisation networks son sorprendentemente pequeñas — fc[32]-fc[32] basta para SVHN — lo que es coherente con la idea de regresar pocos parámetros geométricos.

## Análisis y ablaciones discutidas

**El ST se entrena sólo con la loss de clasificación.** En todos los experimentos, la única supervisión es la etiqueta de la tarea final. La transformación emerge porque facilita la tarea de la capa siguiente.

**TPS > Proj > Aff en datos no rígidos.** En MNIST elástico (E), la diferencia es marcada (TPS 1.1% vs Aff 1.2%). En datos puramente rígidos (R, RTS), los tres tipos son comparables, lo que es esperable.

**Múltiples STs paralelos > 1 ST.** En CUB la mejora 1 → 2 → 4 STs es monotónica (82.3 → 83.1 → 84.1%). En MNIST addition, 2×ST-FCN baja error de 18.5% (1 ST) a 5.8%.

**Inicialización a identidad es crítica.** Los autores reportan que los pesos de la regression layer de la localisation net se inicializan a cero y el bias a la transformación identidad. Si no, el ST puede colapsar a un crop degenerado y los gradientes mueren.

**Sampler bilineal es suficiente.** Aunque cualquier kernel sub-diferenciable funcionaría, en la práctica el bilineal es óptimo en coste/calidad. Para downsampling agresivo puede haber aliasing — los autores aplican average pooling después del ST en algunos modelos para mitigarlo.

**No degrada velocidad.** ST-CNN Multi en SVHN es sólo ~6% más lento que CNN. El cost dominante sigue siendo conv.

## Limitaciones

**Riesgo de colapso a identidad o trivial.** Si la inicialización es mala o la loss no penaliza claramente la transformación, el ST puede aprender la identidad y aportar nada. Los autores no lo mencionan explícitamente como limitación, pero la práctica posterior (ASTER, MORAN) lo confirma: hay que inicializar cuidadosamente y, en algunos casos, regularizar.

**Bilineal aliasing en downsampling.** Si $V$ tiene resolución mucho menor que $U$ y el sampler bilineal sólo mira 4 vecinos, hay aliasing. Solución parche en el paper: average pool tras el ST.

**Sensible a inicialización.** El paper inicializa la regression layer a cero + bias identidad — esto es estándar pero hay que respetarlo. ASTER y MORAN (años después) añaden una **iniciación con keypoints predichos** para acelerar la convergencia del TPS.

**Sin manejo de transformaciones discontinuas.** Un solo ST aplica **una sola** transformación global al feature map. Oclusiones, múltiples objetos con poses muy distintas o discontinuidades (e.g. dos textos con perspectivas opuestas en la misma imagen) requieren STs paralelos — y el número de paralelos limita el número de objetos modelados.

**Número de STs paralelos es hiperparámetro fijo.** En CUB se eligen 2 o 4 a mano. No hay un mecanismo de selección dinámica. Esto se aborda en trabajos posteriores con attention soft.

**No equivariante "puro".** El ST aprende a aplicar una transformación, pero no produce representaciones explícitamente equivariantes — las layers downstream siguen viendo features pasadas por una warp. Para equivariancia formal hay otras familias (group-equivariant CNNs, Cohen–Welling 2016, capsules).

**Capacidad de la localisation network.** Si $f_{\text{loc}}$ es muy pequeña, no captura bien la transformación necesaria; si es muy grande, sobreajusta. Es otro hiperparámetro a tunear.

## Impacto en Scene Text Recognition

Esta es la línea de impacto **directamente relevante** para la clase 21. El STN aparece explícitamente en la slide *"Image preprocessing stage: STN, TPS, Other networks"*. La cadena cronológica:

**RARE (Shi, Wang, Lyu, Yao, Bai — CVPR 2016).** "Robust scene text recognition with Automatic REctification". Primer uso explícito de STN + TPS como módulo de rectificación de texto. La localisation net predice las coordenadas de $K = 20$ puntos de control en el input, el TPS resuelve la deformación inversa y el sampler produce una imagen rectificada que entra a un sequence recognizer attention-based. Mejora notable sobre CRNN en irregular text (curvado, perspectiva).

**ASTER (Shi, Yang, Lyu, Bai — TPAMI 2018).** "An Attentional Scene Text Recognizer with Flexible Rectification". Refina RARE: usa un **STN bidireccional**, predice keypoints de la línea superior e inferior del texto, aplica TPS y luego un encoder-decoder con atención sobre la imagen rectificada. Gana ~5% en datasets de irregular text (ICDAR15, CUTE80, Total-Text). Es el baseline canónico hasta ~2020.

**MORAN (Luo, Jin, Sun — Pattern Recognition 2019).** "Multi-Object Rectified Attention Network". Generaliza ASTER: rectificación píxel a píxel (no por puntos de control) más recognizer attention-based. Mantiene el sampler bilineal de STN como bloque base.

**RARE → ASTER → MORAN → SAR → SATRN → ABINet → ParSeq.** La línea de scene text recognition de la era 2016–2022 puede ordenarse en dos paradigmas: (i) **rectify then recognize** (RARE, ASTER, MORAN — todos usan STN+TPS o variantes) y (ii) **recognize without rectification** (CRNN puro, SATRN con 2D self-attention, ABINet con language model, ABCNet con BezierAlign). La clase 21 contrasta justamente estos paradigmas: STN+TPS es **el preprocessing canónico** para curved/perspective text antes de los enfoques no-rectificadores.

**Contraste pedagógico con ABCNet (Liu et al. 2020, CVPR oral).** ABCNet **no usa STN ni rectifica**. En su lugar, modela cada texto curvado con una **curva de Bézier** (8 puntos) y muestrea features directamente a lo largo de la curva con **BezierAlign**. Conceptualmente: en lugar de "des-curvar" el input y luego aplicar un recognizer rectilíneo, ABCNet representa la curva y mantiene la operación rectilínea local sobre features curveadas. Pedagógicamente es el contraste perfecto: STN/TPS = enderezar; BezierAlign = aceptar la curva. La clase 21 menciona ambos enfoques.

## Otros impactos en visión profunda

**Deformable Convolutions (Dai, Qi, Xiong, Li, Zhang, Hu, Wei — ICCV 2017).** Generaliza la idea STN al **kernel** mismo. En vez de aprender una transformación global y aplicar conv estándar, aprende offsets por posición de kernel: $y(p_0) = \sum_{p_n \in R} w(p_n) \cdot x(p_0 + p_n + \Delta p_n)$. Es esencialmente STN aplicado per-receptive-field, y usa el mismo sampler bilineal sub-diferenciable. Lectura obligatoria para detection (DCNv1, DCNv2 son backbones populares).

**Dynamic Filter Networks (De Brabandere, Jia, Tuytelaars, Van Gool — NeurIPS 2016).** Generalización dual: en vez de aprender una transformación espacial, aprende el **filtro convolucional** condicionado en el input. Conceptualmente "predicción de pesos en tiempo de inferencia".

**Vision Transformer attention (Dosovitskiy et al. 2020).** Filosóficamente cercano: la atención de un patch a otros patches es una forma de "elegir dónde mirar" diferenciable. ViT no usa STN, pero comparte el espíritu de "atención sobre regiones aprendida por gradiente".

**Differentiable rendering / NeRF (Mildenhall et al. 2020).** El **sampler bilineal diferenciable** es uno de los bloques fundacionales: NeRF y todas las técnicas de differentiable rendering muestrean valores de un campo continuo de forma diferenciable. La técnica matemática es la misma que aquí (interpolación bilineal/trilineal con gradientes analíticos).

**Pose estimation y crop-and-resize en object detection.** Fast R-CNN (Girshick 2015) y R-FCN ya usaban "RoI pooling" para extraer features de regiones, pero no diferenciable en las coordenadas. **RoIAlign** (Mask R-CNN, He 2017) introduce sampling bilineal para hacer RoI pooling diferenciable — directamente inspirado en el sampler STN.

**Self-supervised correspondence learning, optical flow (RAFT, FlowNet).** El warping diferenciable de features por flujos predichos usa el mismo sampler.

## Conexión con la clase 21 y con otros papers del curso

**Clase 21 — Scene Text Detection y Recognition.** El paper STN es **prerrequisito directo** para entender la slide *"Image preprocessing stage: STN, TPS, Other networks"*. Antes de aplicar CRNN (Shi 2017) o un recognizer attention-based al texto detectado, se rectifica con STN + TPS si el texto es curvado o en perspectiva. ASTER y MORAN son los recognizers canónicos basados en esta receta.

**Contraste con [Liu-ABCNet-2020](Liu-ABCNet-2020.md).** ABCNet elimina la etapa de rectificación: en vez de des-curvar el input, modela la curva con Bézier y aplica BezierAlign (un sampler bilineal a lo largo de una curva, también diferenciable — heredando del STN la técnica de sampling). Comparar STN+TPS vs BezierAlign es un excelente ejercicio: ambos son "diferenciar el muestreo geométrico", pero uno endereza y el otro acepta la curva.

**Conexión con clase 09 (CNN).** El STN extiende el aparato CNN: no reemplaza convolución ni pooling, los complementa con un mecanismo de transformación espacial explícito. Releer el paper habiendo entendido bien CNN es esencial — el STN sólo tiene sentido como "lo que las CNN no hacen".

**Conexión con clase 14 (attention).** STN es una forma de **hard attention diferenciable**: en vez de muestrear regiones con políticas reforzadas (RL) o soft attention gaussiana, parametriza una transformación geométrica con pocos parámetros y la aplica con sampler bilineal. Conceptualmente: STN es atención sobre regiones, no sobre tokens.

**Conexión con clase 17 (pose recognition).** La idea de "aplicar una transformación espacial aprendida" reaparece en pose: en algunos recognizers de pose, se warpea la imagen al canonical pose antes de procesar (SMPL, denoiser-based pipelines). El sampler bilineal diferenciable es el mismo bloque.

**Conexión con fundamento `scene-text-recognition.md`.** La sección de preprocessing del fundamento debe mencionar STN + TPS como receta canónica de rectificación.

## Conclusión

El paper STN aporta una primitiva arquitectónica simple y profunda: **muestreo geométrico diferenciable** desde un input hacia un output, con la transformación condicionada en el propio input vía una pequeña red de regresión y aplicada vía interpolación bilineal. Es un módulo "drop-in" entrenable por backprop estándar, sin etiquetas de transformación, que recupera invariancia frente a transformaciones grandes que el max-pooling y la convolución no manejan bien por sí solos.

Sus resultados experimentales son sólidos pero quizá menos influyentes que su **idea matemática**: la técnica de "sampling bilineal con gradiente analítico sobre coordenadas fuente" se ha convertido en un building block ubicuo. Aparece en RoIAlign (detection), en deformable convolutions, en NeRF y differentiable rendering, en optical flow estimation, en RAFT, en VITON y try-on networks, y de forma central en **scene text recognition** (RARE, ASTER, MORAN), donde TPS sobre STN es la receta canónica para rectificar texto curvado o en perspectiva.

Para el curso, este paper es la base teórica para entender por qué la clase 21 dedica una slide entera a "Image preprocessing stage: STN, TPS, Other networks" — y por qué ese paradigma de *rectify-then-recognize* compite (y a veces pierde) con paradigmas más recientes que aceptan la geometría arbitraria del texto sin rectificar (ABCNet con BezierAlign, SATRN con 2D self-attention). Cualquier ingeniero que quiera entender la línea histórica de scene text recognition tiene que pasar por aquí.

## Referencias bibliográficas clave (selección comentada)

- **Bookstein, F. (1989).** *Principal warps: Thin-plate splines and the decomposition of deformations.* IEEE PAMI. Base matemática del TPS, citado como ref [2] del paper.
- **Hinton (1981, 2011); Tieleman (2014).** Línea de pre-historia: cápsulas, transforming auto-encoders, modelos generativos compositionales — el conceptual de "equivariancia explícita" del que STN es la versión escalable y diferenciable.
- **Cohen & Welling (ICLR 2015); Lenc & Vedaldi (CVPR 2015).** Análisis empírico de invariancia/equivariancia de CNNs — justifican el problema que STN viene a resolver.
- **Ba, Mnih, Kavukcuoglu (ICLR 2015) DRAM; Mnih et al. (NIPS 2014) RAM; Gregor et al. (ICML 2015) DRAW; Xu et al. (ICML 2015) Show-Attend-Tell.** Familia de attention models — STN es la alternativa diferenciable (sin RL) y geométrica (no sólo gaussiana).
- **Goodfellow et al. (2013) Multi-digit recognition; Netzer et al. (2011) SVHN.** Baselines y dataset de SVHN.
- **Wah et al. (2011) CUB-200-2011.** Dataset de fine-grained birds.
- **Shi, Wang, Lyu, Yao, Bai (CVPR 2016) RARE; Shi, Yang, Lyu, Bai (TPAMI 2018) ASTER; Luo, Jin, Sun (PR 2019) MORAN.** Línea de scene text recognition con rectificación STN+TPS. Conexión directa con clase 21.
- **Dai et al. (ICCV 2017) Deformable Convolutions; De Brabandere et al. (NIPS 2016) Dynamic Filter Networks.** Generalizaciones de la idea STN.
- **He et al. (ICCV 2017) Mask R-CNN — RoIAlign.** Heredero directo del sampler bilineal STN en detection.
