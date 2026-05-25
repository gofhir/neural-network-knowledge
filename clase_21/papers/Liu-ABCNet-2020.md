---
title: "ABCNet — Real-time Scene Text Spotting with Adaptive Bezier-Curve Network"
authors:
  - Yuliang Liu
  - Hao Chen
  - Chunhua Shen
  - Tong He
  - Lianwen Jin
  - Liangwei Wang
year: 2020
venue: "CVPR 2020 (oral)"
slug: liu-abcnet-2020
arxiv: "https://arxiv.org/abs/2002.10200"
tags:
  - scene-text-spotting
  - bezier-curves
  - end-to-end
  - anchor-free
  - fcos
  - bezier-align
---

## Resumen ejecutivo

ABCNet (Adaptive Bezier-Curve Network) propone el primer pipeline end-to-end de scene text spotting que representa texto de forma arbitraria mediante **curvas Bezier cúbicas paramétricas**. En lugar de polígonos densos (14–16 vértices manuales) o segmentación pixel a pixel costosa, una curva Bezier cúbica usa solo 4 puntos de control por lado del texto (8 puntos en total) y captura líneas curvas, en perspectiva o irregulares con costo computacional despreciable. La segunda contribución es **BezierAlign**, una variante de RoIAlign que muestrea features a lo largo de la curva (no en una grilla rectangular ni en un cuadrilátero), permitiendo "rectificar" el texto curvo antes del recognizer. El sistema usa un detector anchor-free al estilo FCOS sobre ResNet-50 + FPN, suma una rama de regresión de 16 canales para los puntos de control y conecta un recognizer ligero (6 conv + 1 BLSTM + FC, CTC loss). En Total-Text alcanza F-measure 69.5 (lexicon-free) y 78.4 (full lexicon) a 6.9 FPS, superando Mask TextSpotter'19; la variante ABCNet-F llega a 22.8 FPS manteniendo 61.9 F-measure, abriendo la puerta a aplicaciones en tiempo real.

## 1. Contexto histórico — STR antes de 2020

### 1.0. De OCR clásico a scene text recognition

El campo de OCR (Optical Character Recognition) tradicional, dominado por motores como Tesseract durante décadas, asumía dos cosas que en escenas naturales se rompen: (a) texto recto, alineado axis-aligned sobre fondo blanco contrastado (documentos escaneados), y (b) tipografía limitada y conocida. Cuando la comunidad de visión computacional empezó a atacar texto "in the wild" — letreros, etiquetas, grafiti, productos en estanterías — emergió un problema fundamentalmente distinto: el texto puede aparecer rotado, en perspectiva, con iluminación heterogénea, oclusiones, tipografías arbitrarias, en cualquier escala, y crucialmente, **siguiendo curvas no lineales**. Pensemos en el logo de un letrero arqueado, el texto enrollado alrededor de una taza, una pegatina sobre un casco, o el nombre de un equipo curvado sobre un escudo.

El primer benchmark serio fue ICDAR 2003, seguido por ICDAR 2013/2015 (texto recto y rotado), pero los puntos de inflexión fueron **Total-Text (2017)** y **CTW1500 (2019)**, que forzaron a la comunidad a tratar texto curvo como first-class citizen. Mientras tanto, deep learning replicó la división del campo en dos sub-tareas: **text detection** (¿dónde está el texto?) y **text recognition** (¿qué dice?). Sistemas modulares (EAST + CRNN, CRAFT + TPS-CRNN) entrenaban ambas etapas separadas; sistemas end-to-end (FOTS, Mask TextSpotter) las fusionaban con beneficios obvios en shared features y joint optimization.

### 1.1. Dos familias dominantes hasta 2019

El "scene text spotting" (detección + reconocimiento conjunto de texto en imágenes naturales) era, hasta CVPR 2020, un dominio fracturado entre dos enfoques:

- **Character-based / box-regression**: TextBoxes, EAST, FOTS, TextDragon. Predicen bounding boxes axis-aligned o cuadriláteros rotados, y a veces tags a nivel de carácter. Son rápidos pero **incapaces de capturar texto curvo**: cualquier letrero "WELCOME TO GATORLAND" con arco lo recortan en un rectángulo enorme con mucho fondo, contaminando el recognizer.
- **Segmentation-based**: Mask TextSpotter, CharNet, PSENet, PAN. Producen máscaras a nivel de pixel; en teoría capturan cualquier forma, pero (a) requieren post-procesamiento no diferenciable (component grouping, fitting polygonal), (b) son sensibles a texto cercano que se "pega" en la máscara, (c) el costo per-pixel es alto, y (d) muchas variantes (Mask TextSpotter v1) exigen anotaciones character-level que son caras y casi inexistentes en datos reales.

La Figura 2 del paper muestra la genealogía: Li et al. 2017, He et al. 2018 (TextAlign), Liu et al. 2018 (FOTS), Liao et al. 2018 (Mask TextSpotter), Sun et al. 2018 (TextNet), Qin et al. 2019 (RoI Masking), Xing et al. 2019 (CharNet), Feng et al. 2019 (TextDragon con RoISlide). Cada uno introduce un sampler distinto (RoI Pooling, RoIRotate, RoIAlign, RoI Transform, RoI Masking, RoISlide), confirmando que **el cuello de botella era la alineación geométrica entre detector y recognizer**.

### 1.2. Por qué el texto irregular era el cuello de botella

Tres datasets de la época (Total-Text 2017, CTW1500 2019, ICDAR2019-ArT) consolidaron benchmarks donde **al menos una instancia de cada imagen es curva**. La anotación oficial usa polígonos de 10 vértices (Total-Text extendido) o 14 vértices (CTW1500), y el resultado de evaluar TextBoxes / FOTS sobre Total-Text caía 30+ puntos de F-measure frente a benchmarks rectos (ICDAR2013/2015). Para "Welcome" en un arco, un cuadrilátero rotado deja la mitad del texto fuera; un polígono de 14 puntos rectifica bien pero requiere predecir 28 coordenadas correladas, y los métodos de regresión existentes (Mask TextSpotter, CharNet) recurrían a segmentación para evitar esa explosión paramétrica.

La pregunta que ABCNet contesta es: **¿existe una representación paramétrica compacta que (a) sea suficientemente expresiva para texto curvo del mundo real, (b) tenga bajo costo computacional y (c) permita una alineación geométrica diferenciable con el recognizer?** La respuesta es la curva Bezier cúbica.

## 2. Contribución central — representación Bezier cúbica

### 2.1. Espectro de representaciones de bounding box

Para situar el aporte:

| Representación | Parámetros por instancia | Capacidad | Costo |
| --- | --- | --- | --- |
| Horizontal bbox | 4 ($x, y, w, h$) | Solo texto horizontal | Mínimo |
| Cuadrilátero rotado | 8 (4 esquinas) | Texto inclinado, perspectiva leve | Bajo |
| Polígono denso | 28 (14 vértices) | Texto curvo, irregular | Alto, ruidoso |
| Mask binaria | $H \times W$ pixeles | Cualquier forma | Muy alto |
| **Bezier cúbica (ABCNet)** | **16 (8 puntos de control)** | **Texto curvo arbitrario** | **Bajo** |

Bezier cúbica logra el sweet spot: 16 escalares (lo mismo que dos cuadriláteros) capturan texto curvo arbitrario con suavidad C^∞ y permiten muestreo paramétrico exacto.

### 2.2. Curvas Bezier — la matemática

Una curva Bezier de grado $n$ se define como una combinación convexa de $n+1$ puntos de control $b_i \in \mathbb{R}^2$ ponderados por los **polinomios de Bernstein** $B_{i,n}(t)$:

$$
c(t) = \sum_{i=0}^{n} b_i \, B_{i,n}(t), \quad 0 \le t \le 1,
$$

donde

$$
B_{i,n}(t) = \binom{n}{i} t^i (1-t)^{n-i}, \quad i = 0, 1, \dots, n.
$$

Para la cúbica $n=3$ se expande a:

$$
c(t) = (1-t)^3 b_0 + 3(1-t)^2 t \, b_1 + 3(1-t) t^2 \, b_2 + t^3 b_3.
$$

**Propiedades que justifican la elección**:

- **Partición de la unidad**: $\sum_i B_{i,n}(t) = 1$ para todo $t$. La curva queda dentro del *casco convexo* (convex hull) de sus puntos de control, lo que da estabilidad numérica al regresar.
- **Endpoint interpolation**: $c(0) = b_0$ y $c(1) = b_3$. Los puntos extremos del texto coinciden con dos puntos de control, lo que da una correspondencia geométrica intuitiva con el inicio y el fin de la línea de texto.
- **Tangencia en los extremos**: $c'(0) = 3(b_1 - b_0)$ y $c'(1) = 3(b_3 - b_2)$. Los puntos de control intermedios $b_1, b_2$ controlan la *dirección* de entrada y salida de la curva.
- **Invariancia afín**: si transformas los puntos de control con una matriz afín, la curva se transforma con la misma matriz. Esto es fundamental para que data augmentation (rotación, escala, traslación) sea consistente.
- **Suavidad**: una cúbica tiene continuidad $C^\infty$, sin esquinas espurias.

La derivada es:

$$
c'(t) = \sum_{i=0}^{n-1} n (b_{i+1} - b_i) B_{i,n-1}(t),
$$

útil para calcular la normal a la curva (necesaria al muestrear en BezierAlign).

### 2.3. ¿Por qué cúbica y no cuadrática o quintic?

Los autores reportan evidencia empírica: observan los polígonos de Total-Text y CTW1500 y comprueban que una cúbica fittea con error sub-pixel en >95% de los casos. La intuición geométrica es simple:

- Una **lineal** ($n=1$) es un segmento — solo texto recto.
- Una **cuadrática** ($n=2$) tiene un solo punto intermedio de control: captura arcos simples pero no S-curves.
- Una **cúbica** ($n=3$) puede tener un punto de inflexión (S-curve), suficiente para letreros enrollados, círculos parciales y texto sobre objetos curvos como botellas o tazas.
- Una **quintic** ($n=5$) capturaría más oscilaciones pero la anotación humana no es lo bastante precisa para distinguirlas, y la regresión se vuelve mal condicionada.

Para texto se usan **dos curvas Bezier cúbicas** (top boundary + bottom boundary), 4 puntos cada una = **8 puntos de control = 16 coordenadas**.

### 2.4. Generación de ground truth Bezier desde polígonos

Total-Text trae polígonos con $m+1$ vértices por boundary (típicamente $m=5$ para Total-Text, $m=7$ para CTW1500). Para convertir cada boundary en una cúbica, los autores resuelven un **least-squares lineal**:

$$
\begin{bmatrix}
B_{0,3}(t_0) & B_{1,3}(t_0) & B_{2,3}(t_0) & B_{3,3}(t_0) \\
B_{0,3}(t_1) & B_{1,3}(t_1) & B_{2,3}(t_1) & B_{3,3}(t_1) \\
\vdots & \vdots & \vdots & \vdots \\
B_{0,3}(t_m) & B_{1,3}(t_m) & B_{2,3}(t_m) & B_{3,3}(t_m)
\end{bmatrix}
\begin{bmatrix}
b_{0x} & b_{0y} \\
b_{1x} & b_{1y} \\
b_{2x} & b_{2y} \\
b_{3x} & b_{3y}
\end{bmatrix}
=
\begin{bmatrix}
p_{0x} & p_{0y} \\
p_{1x} & p_{1y} \\
\vdots & \vdots \\
p_{mx} & p_{my}
\end{bmatrix}.
$$

Los $t_i$ se calculan por el método de **parametrización por arc-length**: se acumula la longitud de la polilínea, se normaliza a $[0, 1]$ y se asigna a cada vértice. $b_0$ y $b_3$ se fijan como el primer y último vértice anotado (endpoint interpolation), y solo se resuelven $b_1, b_2$.

La Figura 5 del paper muestra una propiedad sutil pero importante: **el ground truth Bezier es a menudo *más suave y mejor* que la anotación polygonal humana**. Los anotadores no son perfectos al trazar arcos a mano, mientras que la cúbica ajustada por mínimos cuadrados promedia el ruido.

## 3. Arquitectura del modelo

### 3.0. Visión panorámica del flujo

El sistema completo se puede leer como una tubería de seis estaciones encadenadas:

1. **Imagen RGB** entra al sistema (típicamente 800×800 px en inferencia estándar).
2. **ResNet-50** extrae features jerárquicas C2–C5.
3. **FPN** fusiona top-down esos niveles produciendo P3–P7 (multi-escala).
4. **Detection head FCOS-style** predice densamente por pixel: clase, center-ness, bbox y los 16 offsets de los puntos de control Bezier.
5. **BezierAlign** toma las predicciones Bezier y las usa para muestrear el feature map original sobre la región curva, produciendo un mini-mapa rectificado de tamaño fijo (típicamente 7×32).
6. **Recognition branch** (6 conv + BLSTM + FC + CTC) lee el texto.

Lo elegante es que entre 4 y 5 el sistema se mantiene completamente diferenciable: BezierAlign usa interpolación bilineal sobre la curva paramétrica, y el gradiente fluye desde la loss de CTC hasta los puntos de control regresados, y de ahí al backbone. No hay paso no diferenciable como en métodos basados en segmentación, donde el "fit polygon a la máscara" rompe la cadena.

### 3.1. Backbone y multi-escala

- **Backbone**: ResNet-50 estándar.
- **Neck**: Feature Pyramid Network (FPN, Lin et al. 2017), con 5 niveles P3–P7 a resoluciones 1/8, 1/16, 1/32, 1/64, 1/128 del input.
- **Detection branch**: opera sobre los 5 niveles FPN.
- **Recognition branch**: opera sobre 3 niveles 1/4, 1/8, 1/16 (más alta resolución para preservar detalle de caracteres).

### 3.2. Cabezales (anchor-free, estilo FCOS)

ABCNet adopta el paradigma **FCOS** (Tian et al. 2019): predicción densa pixel a pixel, sin anchor boxes. Cada pixel $(x, y)$ del feature map predice:

1. **Clasificación**: 1 canal (texto vs no-texto), sigmoid + focal loss.
2. **Center-ness**: 1 canal, mide qué tan centrado está el pixel respecto al GT box; pondera la confianza durante NMS para suprimir predicciones cerca del borde.
3. **Bbox regression**: 4 canales $(l, t, r, b)$ — distancias al borde izquierdo, superior, derecho e inferior del cuadrilátero envolvente.
4. **Bezier control points regression**: **16 canales** = 8 puntos × 2 coordenadas, predichos como offsets relativos al mínimo $(x_{min}, y_{min})$ del cuadrilátero:

$$
\Delta_x = b_{ix} - x_{min}, \quad \Delta_y = b_{iy} - y_{min}.
$$

Predecir offsets relativos (y no absolutos) tiene dos ventajas:
- **Invariancia traslacional**: el modelo aprende la forma de la curva, no su posición.
- **Robustez a puntos de control fuera del crop**: si $b_1$ o $b_2$ caen ligeramente fuera del frame por la geometría de un arco, el offset relativo sigue siendo predecible.

Los autores enfatizan que predecir 16 canales adicionales por pixel agrega **una sola capa convolucional** y es prácticamente gratis: 22.8 FPS sin Bezier vs 22.5 FPS con Bezier (Tabla 5 del paper). El costo del Bezier es 0.3 FPS, un **1.3% overhead**.

### 3.3. BezierAlign — la pieza clave

RoIAlign (Mask R-CNN) muestrea features dentro de un **rectángulo axis-aligned**: divide la RoI en una grilla $h_{out} \times w_{out}$, hace bilinear interpolation en cada celda y produce un feature map de tamaño fijo. RoIRotate (FOTS) extiende esto a cuadriláteros rotados. Pero ninguno funciona para texto curvo: el rectángulo deja afuera trozos de texto o incluye fondo.

**BezierAlign** muestrea las features sobre la región delimitada por dos curvas Bezier (top boundary $tp$ y bottom boundary $bp$). Dado pixel de salida en posición $(g_{iw}, g_{ih})$ del output map $h_{out} \times w_{out}$:

1. Calcula el parámetro horizontal:

$$
t = \frac{g_{iw}}{w_{out}}.
$$

2. Evalúa los puntos correspondientes en ambas curvas: $tp = c_{top}(t)$ y $bp = c_{bottom}(t)$ vía la ecuación (1).

3. Interpola linealmente entre ambos puntos según la coordenada vertical:

$$
op = bp \cdot \frac{g_{ih}}{h_{out}} + tp \cdot \left(1 - \frac{g_{ih}}{h_{out}}\right).
$$

4. Aplica **bilinear interpolation** en $op$ sobre el feature map original.

El resultado: el output map de tamaño fijo es una "rectificación" del texto curvo a un rectángulo, donde la dimensión horizontal recorre el texto a lo largo del trazo y la vertical recorre el grosor del texto. El recognizer ve siempre texto horizontal.

La Figura 7 del paper compara los tres samplings sobre la misma instancia ("BISTRO" en arco): horizontal sampling deja la mitad afuera, quadrilateral mejora pero distorsiona, BezierAlign queda perfectamente alineado.

**Hyper-parámetros**: la Tabla 4 ablation muestra que $(n_h, n_w) = (7, 32)$ es el sweet spot: F-measure 61.9 a 22.8 FPS. Más muestras (28, 128) bajan a 53.4 F-measure por overfitting al ruido del feature map; menos (6, 32) bajan a 59.6 pero suben a 23.2 FPS.

### 3.4. Recognition head ligero

| Capa | Parámetros (kernel, stride) | Output (n, c, h, w) |
| --- | --- | --- |
| conv × 4 | (3, 1) | (n, 256, h, w) |
| conv × 2 | (3, (2,1)) | (n, 256, h, w) |
| avg pool en h | — | (n, 256, 1, w) |
| Channels-Permute | — | (w, n, 256) |
| BLSTM | — | (w, n, 512) |
| FC | — | (w, n, n_class) |

- 6 convoluciones (4 con stride 1, 2 con stride (2,1) para colapsar altura).
- Average pooling vertical.
- Una **BLSTM** (bidirectional LSTM) que procesa la secuencia.
- FC a 97 clases (case-sensitive English + dígitos + símbolos + EOF + "unseen" en CTW1500).
- **CTC Loss** (Connectionist Temporal Classification, Graves et al. 2006) para alinear secuencia predicha vs GT sin necesidad de character-level boundaries.

Es deliberadamente ligero: el paper cita que la versión más compleja (2D attention de Mask TextSpotter v2) requería pasos adicionales costosos que ABCNet evita gracias a la rectificación previa.

### 3.5. Pérdida total

$$
\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{center} + \mathcal{L}_{bbox} + \mathcal{L}_{bezier} + \mathcal{L}_{rec},
$$

donde:
- $\mathcal{L}_{cls}$ — focal loss para texto/no-texto.
- $\mathcal{L}_{center}$ — BCE para center-ness.
- $\mathcal{L}_{bbox}$ — IoU loss (o GIoU, Rezatofighi et al. 2019) sobre el cuadrilátero.
- $\mathcal{L}_{bezier}$ — Smooth L1 sobre los 16 offsets de los puntos de control.
- $\mathcal{L}_{rec}$ — CTC cross-entropy sobre la secuencia.

## 4. Entrenamiento

### 4.0. Estrategia general

Entrenar un modelo end-to-end de detección y reconocimiento de texto presenta dos desafíos acoplados que ABCNet resuelve con una receta clásica de transfer learning: **pre-training sobre un corpus sintético masivo + fine-tuning sobre datos reales pequeños**. La razón es asimétrica entre las dos ramas. La detection branch requiere fundamentalmente **diversidad geométrica** (muchos arcos, perspectivas, escalas), que se puede generar barato con texto sintético inyectado en fondos naturales. La recognition branch requiere **diversidad léxica y tipográfica** (muchas palabras, fuentes, idiomas), también generable sintéticamente. Solo el "estilo natural" (degradación de iluminación, blur, oclusiones reales) requiere data real, y por eso el fine-tuning sobre Total-Text o CTW1500 (1k–1.5k imágenes) es suficiente para cerrar el gap de dominio.

### 4.1. Datasets

- **SynText150k** (sintético, contribución del paper): los autores filtran 40k imágenes de fondo libres de texto de COCO-Text, preparan mapas de segmentación y profundidad usando los modelos de Pont-Tuset (multiscale grouping) y Laina (depth estimation), y luego inyectan texto sintético con el método VGG synthetic (Gupta et al. 2016) **modificado para generar arcos y curvas** además de texto recto. El resultado: 94,723 imágenes con mayormente texto recto + 54,327 con texto curvo. Anotación polygonal automática → Bezier ground truth vía el least-squares de §2.4.
- **COCO-Text**: 15k imágenes filtradas con anotaciones a nivel de palabra en inglés.
- **ICDAR-MLT**: 7k imágenes (multi-language, con subset en inglés).
- Fine-tuning: Total-Text (1,255 train) o CTW1500 (1k train) según el benchmark de evaluación.

### 4.2. Hyper-parámetros

- 4× Tesla V100 GPUs.
- Batch size: 32 imágenes.
- Optimizer: SGD (estándar Detectron2/AdelaiDet).
- Learning rate inicial: 0.01, decay a 0.001 en 70k iter, a 0.0001 en 120k iter.
- Total: 150k iteraciones, ~3 días.
- Pre-training en SynText150k + COCO-Text + MLT, luego fine-tune en dataset target.
- Data augmentation: random scale (short side ∈ [560, 800], long side < 1333), random crop (mantiene al menos la mitad del crop original y sin cortar texto cuando es posible).

### 4.3. Inference

- Detection: short side = 800 para ABCNet estándar, 600 para ABCNet-F (fast).
- Multi-scale: ABCNet-MS testea en múltiples escalas y promedia.
- Post-procesamiento: NMS sobre cuadriláteros, luego usa los puntos Bezier predichos para BezierAlign + recognizer.

## 5. Resultados experimentales

### 5.0. Métricas

La métrica estándar de end-to-end scene text spotting es **F-measure word-level**: una predicción cuenta como verdadero positivo si (a) su Bezier predicho tiene IoU ≥ 0.5 con el ground truth a nivel polygon y (b) la string reconocida coincide exactamente con la palabra ground truth (case-sensitive en algunos benchmarks, case-insensitive en otros). Hay dos sub-protocolos:

- **None / lexicon-free**: el recognizer debe producir la palabra correcta sin ayuda externa.
- **Full / strong lexicon**: el sistema puede acotar las predicciones a un diccionario (el set de todas las palabras del test set). Esto permite corregir errores tipográficos pequeños (por ejemplo "PEACHIREE" → "PEACHTREE").

ABCNet reporta ambos protocolos. Como referencia: pasar de None a Full típicamente suma 8–12 puntos de F-measure porque rescata muchas predicciones con uno o dos caracteres erróneos.

### 5.1. Total-Text — comparación con SOTA

| Método | Datos pre-train | Backbone | F-measure (None / Full) | FPS |
| --- | --- | --- | --- | --- |
| TextBoxes | SynText800k + IC13/IC15 + TT | ResNet-50-FPN | 36.3 / 48.9 | 1.4 |
| Mask TextSpotter'18 | SynText800k + IC13/IC15 + TT | ResNet-50-FPN | 52.9 / 71.8 | 4.8 |
| Two-stage (Sun) | SynText800k + IC13/IC15 + TT | ResNet-50-SAM | 45.0 / — | — |
| TextNet | SynText800k + IC13/IC15 + TT | ResNet-50-SAM | 54.0 / — | 2.7 |
| Li et al. 2019 | SynText840k + IC13/IC15/TT/MLT/AddF2k | ResNet-101-FPN | 57.80 / — | 1.4 |
| Mask TextSpotter'19 | SynText800k + IC13/IC15/TT/AddF2k | ResNet-50-FPN | 65.3 / 77.4 | 2.0 |
| Qin et al. 2019 | SynText200k + IC15/COCO-Text/TT/MLT + 30k privados | ResNet-50-MSF | 67.8 / — | 4.8 |
| CharNet | SynText800k + IC15/MLT/TT | ResNet-50-Hourglass57 | 66.2 / — | 1.2 |
| TextDragon | SynText800k + IC15/TT | VGG16 | 48.8 / 74.8 | — |
| **ABCNet-F** | SynText150k + COCO-Text + TT + MLT | ResNet-50-FPN | **61.9 / 74.1** | **22.8** |
| **ABCNet** | id. | id. | 64.2 / 75.7 | 17.9 |
| **ABCNet-MS** | id. | id. | **69.5 / 78.4** | 6.9 |

Lectura clave:

- **ABCNet-MS supera a Mask TextSpotter'19** (la SOTA anterior end-to-end): 69.5 vs 65.3 en lexicon-free.
- **ABCNet-F a 22.8 FPS** es **>11× más rápido** que Mask TextSpotter'19 (2.0 FPS) con caída pequeña en F-measure.
- Usa **menos datos sintéticos** (150k vs 800k de SynText) — la calidad del SynText curvo importa más que la cantidad.

### 5.2. Ablation BezierAlign (Tabla 3)

Sobre el mismo modelo ABCNet, varía únicamente el sampler:

| Sampling | F-measure (%) |
| --- | --- |
| Horizontal sampling | 38.4 |
| Quadrilateral sampling | 44.7 |
| **BezierAlign** | **61.9** |

**+17.2 puntos de F-measure** al pasar de cuadrilátero a Bezier. Esto es la evidencia más fuerte del paper: la rectificación geométrica precisa importa tanto como el recognizer.

### 5.3. Ablation costo del Bezier (Tabla 5)

| Configuración | Inference time |
| --- | --- |
| Sin Bezier curve detection | 22.8 FPS |
| **Con Bezier curve detection** | **22.5 FPS** |

Costo: **0.3 FPS = 1.3% overhead** sobre detección de cuadrilátero estándar. Confirma que la regresión de 16 coordenadas extra por pixel es virtualmente gratis cuando el backbone ya está computado.

### 5.4. Ablation número de sampling points (Tabla 4)

| $(n_h, n_w)$ | F-measure | FPS |
| --- | --- | --- |
| (6, 32) | 59.6 | 23.2 |
| **(7, 32)** | **61.9** | **22.8** |
| (14, 64) | 58.1 | 19.9 |
| (21, 96) | 54.8 | 18.0 |
| (28, 128) | 53.4 | 15.1 |
| (30, 30) | 59.9 | 21.4 |

El óptimo es **bajo y rectangular** (7 alto × 32 ancho), reflejando la forma natural de una palabra latina. Aumentar la resolución no ayuda y hasta hurts performance.

### 5.5. CTW1500 — texto curvo + chino

CTW1500 anota a nivel de línea de texto (no de palabra) e incluye chino. Los autores tratan al chino como clase "unseen" (clase 96) durante entrenamiento.

| Método | Datos | F-measure (None / Strong Full) |
| --- | --- | --- |
| FOTS | SynText800k + CTW1500 | 21.1 / 39.7 |
| Two-Stage* | id. | 37.2 / 69.9 |
| RoIRotate* | id. | 38.6 / 70.9 |
| LSTM* | id. | 39.2 / 71.5 |
| TextDragon | id. | 39.7 / 72.4 |
| **ABCNet** | SynText150k + CTW1500 | **45.2 / 74.1** |

Margen de **+5.5 puntos lexicon-free** sobre TextDragon. El paper menciona honestamente que la métrica word-accuracy a line-level es severa: un solo carácter mal reconocido en una línea larga da score = 0 para esa línea.

## 6. Limitaciones reconocibles

Aunque el paper no las enumera en sección dedicada, una lectura crítica identifica:

1. **Asume líneas de texto**: una cúbica de top + bottom presupone una orientación dominante. Texto **vertical chino o japonés real** (donde la dirección de lectura es vertical) requiere intercambiar roles de top/bottom o re-parametrizar; el paper trata CTW1500 chino como caso degenerado y lo marca "unseen".

2. **Oclusiones severas**: si una palabra está parcialmente tapada, los puntos de control intermedios $b_1, b_2$ pierden señal y la regresión Bezier puede irse a un mínimo local. No se reportan experimentos con datasets occluded.

3. **Dependencia de la calidad sintética**: los 150k SynText curvos son críticos. Sin ellos, el modelo no aprende curvas. La receta de Gupta et al. modificada es ad-hoc y no se incluye en el código original (aunque el repositorio AdelaiDet lo publica luego).

4. **FPS dependiente del input size**: ABCNet-F a 22.8 FPS usa short side = 600. En letreros lejanos con texto pequeño (típicos de driving datasets) esto puede no resolver bien el texto. Escalar la imagen reduce FPS dramáticamente.

5. **Recognizer caso-sensitive limitado**: 97 clases cubren bien inglés latino + dígitos pero no diacríticos (acentos en español, ñ, ü), ni alfabetos no latinos. El paper deja esto como trabajo futuro.

6. **Cúbica fija**: no hay grado adaptativo. Texto que requiere doble inflexión (raro pero existe en logos artísticos) no se modela bien con $n=3$.

7. **Sensibilidad al ground truth Bezier**: la conversión polygon→Bezier vía least-squares introduce un error que se acumula. Para CTW1500 ($m=7$) el fit es mejor que para Total-Text extendido ($m=5$).

## 6.1. Análisis cualitativo de fallos

La Figura 9 del paper muestra explícitamente comparaciones entre quadrilateral warping y BezierAlign warping en tres ejemplos: una etiqueta circular "TELEPHONE" (warp cuadrilátero produce "KIYS" incorrecto, BezierAlign produce "TELEPHONE" correcto), "PEACHTREE" en arco (cuadrilátero produce "PEMEPREE", BezierAlign produce "PEACHTREE"), y "SHOP RITE" en arco (cuadrilátero produce "SKOPRN", BezierAlign produce "SHOP RITE"). El patrón es claro: cuando la geometría del texto se desvía de un cuadrilátero, el sampler convencional produce strings que conservan **algunos** caracteres pero introducen errores sistemáticos por el aliasing visual al estirar el texto curvo a un rectángulo. BezierAlign elimina ese aliasing en la raíz.

Los autores también reconocen errores residuales: ABCNet a veces falla en reconocer un solo carácter dentro de líneas largas (especialmente en CTW1500, donde una línea puede tener 10+ palabras y un mismatch de 1 carácter da score 0 para toda la línea). Esto sugiere que la BLSTM + CTC del recognizer ligero es un cuello de botella secundario después de resolver el problema geométrico — algo que ABCNet v2 luego ataca con attention adaptativa.

## 7. Impacto y legado

ABCNet abrió una línea fértil:

- **ABCNet v2** (Liu et al., TPAMI 2022): extiende a (a) cabezales BiFPN, (b) attention adaptiva en el recognizer en lugar de CTC + BLSTM, (c) entrenamiento con character-aware soft-label. Sube a ~76 F-measure en Total-Text manteniendo tiempo real.
- **TESTR** (Zhang et al., CVPR 2022): Transformer-based, usa los Bezier control points como queries en un DETR-like detector. Cierra la brecha entre detection y NER de texto en una sola atención.
- **SPTS** (Peng et al., 2022): "Single-Point Text Spotting" lleva la lógica al extremo, reemplazando 8 puntos de control por un solo punto + secuencia auto-regresiva de la palabra. Reduce anotación a 1 click por palabra.
- **DBNet++**, **PAN++**, **FCENet** (Fourier Contour Embedding) — métodos posteriores que compiten con representaciones paramétricas alternativas (DCT, Fourier).
- **Handwriting recognition**: ideas de Bezier control points aparecen en sistemas como HMER de fórmulas matemáticas y handwritten text recognition multilingüe.
- **Adelaide AdelaiDet**: librería oficial donde ABCNet (junto con FCOS, BlendMask, MEInst, CondInst) consolidó el ecosistema anchor-free para vision tasks denso.

El concepto general — **regresar control points + alinear geométricamente con un sampler diferenciable** — entra al mainstream de cualquier tarea donde la geometría no sea axis-aligned: detección de lane markings, segmentación de hojas en agronomía, contornos médicos.

## 8. Conexión con el lab y la clase del Diplomado IA UC

### 8.1. Lugar en la clase 21

La clase 21 del módulo de Visión Computacional cubre **OCR y scene text recognition**. El pipeline canónico se enseña en dos etapas:

1. **Text detection**: localizar regiones con texto (CRAFT, EAST, DBNet).
2. **Text recognition**: leer el texto dentro de cada región (CRNN, TPS-CRNN, ABINet, PARSeq).

ABCNet entra como el primer ejemplo de **integración end-to-end** donde detection + recognition comparten backbone y se entrenan jointly. Esto motiva tres preguntas pedagógicas:

- ¿Cuándo conviene un pipeline two-stage (detector + recognizer separados) vs end-to-end?
- ¿Cómo se diseña un sampler diferenciable que respete la geometría del objeto?
- ¿Qué representación de bounding box elegir según la naturaleza del dominio?

### 8.2. Lugar en el Lab 21

El lab introduce un pipeline de scene text recognition aplicado (típicamente con EasyOCR, PaddleOCR, o un modelo HuggingFace). ABCNet sirve como **referencia conceptual** del modelo "completo" que esos toolkits aproximan en producción. Concretamente:

- PaddleOCR PP-OCRv3 usa una variante de DBNet (segmentation-based) + SVTR recognizer. ABCNet es la alternativa regression-based con curvas explícitas.
- EasyOCR usa CRAFT detector + CRNN recognizer. Comparado con ABCNet: menos integrado, dos modelos separados, sin BezierAlign.

Una pregunta de discusión natural en el lab: si tu dominio tiene texto en envases curvos (botellas, latas, taxis) o señalética con arcos, ¿es preferible entrenar ABCNet desde cero, fine-tunear con PaddleOCR, o usar un Vision-Language Model (LLaVA, Qwen-VL) que lee texto vía atención?

### 8.3. Para el dominio FHIR-adjacent

Aunque tu interés primario es FHIR clínico, hay aplicaciones tangentes:

- **OCR de etiquetas farmacéuticas en envases curvos** (cilindros de medicamentos) — caso de uso real donde texto curvo + perspectiva justifica una representación Bezier.
- **OCR de wristbands de pacientes** (identificación) — texto en cintas curvadas alrededor de la muñeca.
- **Document understanding** — donde el texto está mayormente recto, ABCNet es over-engineered, y un DBNet + CRNN basta.

El criterio de selección que ABCNet enseña: **la representación geométrica debe ajustarse a la distribución empírica del texto en tu dominio**. No uses cúbicas si tu texto es recto; no uses cuadriláteros si tu texto se curva.

## 9. Notas matemáticas adicionales sobre Bezier

### 9.0. Historia breve

Pierre Bézier desarrolló estas curvas en los años 60 en Renault para diseñar carrocerías de automóviles; independientemente, Paul de Casteljau hizo lo mismo en Citroën. Las curvas adoptaron el nombre de Bézier porque Renault publicó primero. La base matemática — los polinomios de Bernstein — es bastante anterior: Sergei Bernstein los introdujo en 1912 para una prueba constructiva del teorema de aproximación de Weierstrass (toda función continua sobre $[0,1]$ se puede aproximar uniformemente por polinomios). Esto le da a las curvas Bezier una propiedad muy útil para deep learning: son **universalmente aproximadoras** de curvas continuas en el sentido de que aumentando el grado se puede acercar arbitrariamente cualquier forma suave. ABCNet elige el grado mínimo ($n=3$) que captura el espacio de formas de texto natural.

### 9.1. Derivada y normal

$$
c'(t) = 3 \big[(1-t)^2 (b_1 - b_0) + 2(1-t)t (b_2 - b_1) + t^2 (b_3 - b_2)\big].
$$

La normal unitaria en $c(t)$ se obtiene rotando $c'(t)$ 90°. BezierAlign no la usa explícitamente porque interpola entre top y bottom (que ya están en lados opuestos), pero variantes posteriores (TESTR) sí calculan normales para muestreo más fino.

### 9.2. Subdivisión de De Casteljau

El algoritmo de De Casteljau evalúa $c(t)$ por interpolaciones lineales recursivas:

$$
b_i^{(k)}(t) = (1-t) b_i^{(k-1)}(t) + t \, b_{i+1}^{(k-1)}(t), \quad b_i^{(0)} = b_i.
$$

Con $b_i^{(n)}(t) = c(t)$. Es numéricamente más estable que evaluar Bernstein directo y se usa en CAD/CAGD. Para inference de ABCNet, evaluar Bernstein directo basta porque $n=3$ es muy bajo.

### 9.3. Polinomios de Bernstein — propiedades

- Forman base del espacio de polinomios de grado $\le n$.
- Son no negativos en $[0,1]$.
- Se transforman bajo elevación de grado: una cúbica se puede reescribir como cuártica con 5 puntos de control equivalentes, útil para comparar curvas de distinto grado.
- Convergen uniformemente: el polinomio de Bernstein de $f$ converge a $f$ a tasa $O(1/\sqrt{n})$ (teorema clásico, base de la prueba constructiva del teorema de Stone-Weierstrass).

Para profundizar, mira el [fundamento bezier-curves](../../fundamentos/bezier-curves/) (en construcción) donde se desarrollan estas propiedades con visualizaciones.

### 9.4. Por qué cúbicas (no quintic) son suficientes — intuición espectral

Una cúbica tiene 4 grados de libertad por dimensión (8 por curva 2D). El espectro de frecuencias de una palabra arqueada en un letrero contiene un componente DC (la posición media), un componente de inclinación (slope), y un componente de curvatura (segunda derivada). Más allá de eso (oscilaciones de alta frecuencia) está el ruido humano de anotación. Una cúbica captura DC + slope + curvatura + un grado adicional de inflexión, lo cual cubre el espectro de señal real y filtra el ruido.

## 10. Referencias clave del paper

- **[9] He, Kaiming et al. ResNet (CVPR 2016)** — backbone de prácticamente todo el sistema. Residual connections enablearon entrenar redes de 50+ capas.
- **[22] Lin, T.-Y. FPN (CVPR 2017)** — neck multi-escala. Da features de alta resolución (P3) para textos pequeños y baja resolución (P7) para textos grandes.
- **[37] Tian, Zhi et al. FCOS (ICCV 2019)** — detector anchor-free per-pixel. ABCNet adopta su filosofía y agrega 16 canales para Bezier.
- **[8] He, K. Mask R-CNN (ICCV 2017)** — fuente de RoIAlign, que BezierAlign extiende.
- **[3] Ch'ng, C.-K. Total-Text (ICDAR 2017)** — primer benchmark de texto curvo en escenas naturales.
- **[26] Liu, Yuliang. CTW1500 (PR 2019)** — benchmark adicional, anotación line-level, incluye chino.
- **[6] Graves, A. CTC (ICML 2006)** — Connectionist Temporal Classification, la loss que permite entrenar sequence recognition sin alineación carácter a carácter.
- **[35] Shi, B. CRNN (TPAMI 2017)** — arquitectura de la recognition branch ligera (CNN + BLSTM + CTC).
- **[13] Hochreiter, Schmidhuber. LSTM (1997)** — base de la BLSTM del recognizer.
- **[29] Lorentz. Bernstein polynomials** — referencia matemática clásica para las bases polinómicas usadas.
- **[7] Gupta, A. SynText (CVPR 2016)** — método sintético VGG que ABCNet adapta para generar texto curvo.
- **[20] Liao, M. Mask TextSpotter v3 (TPAMI 2019)** — el baseline principal a superar.

Rezatofighi (GIoU) y Bahdanau (attention), aunque citados como inspiración general, no aparecen en la lista directa porque ABCNet usa IoU loss estándar y attention solo aparece como contraste con métodos previos.

## 10.1. Diálogo con otros papers del módulo NLP/Vision

ABCNet conecta con varias ideas estudiadas en clases previas:

- **FCOS** (Tian et al. 2019, clase de detección): el paradigma anchor-free. ABCNet es una de las primeras aplicaciones exitosas de FCOS fuera de COCO. Demuestra que la filosofía "predecir per-pixel sin anchors" generaliza a tareas donde la geometría del objeto no es axis-aligned. La lección transferible: una vez que dominas un detector denso, agregar regresores adicionales (16 canales para Bezier, 4 canales para keypoints, $k$ canales para cualquier parametrización) es barato.

- **CRNN** (Shi et al. 2017): la recognition branch de ABCNet es una simplificación directa de CRNN. La diferencia es que CRNN se entrena de forma aislada sobre patches recortados, mientras que en ABCNet se entrena end-to-end vía BezierAlign. El gradiente que la BLSTM produce influencia los puntos de control Bezier, cerrando el loop.

- **CTC loss** (Graves et al. 2006): el "trick" que permite entrenar secuencia a secuencia sin alineaciones explícitas. CTC introduce un símbolo blank y suma probabilidades sobre todas las alineaciones posibles, lo que hace que el recognizer aprenda donde "estirar" o "comprimir" el texto. En ABCNet, CTC opera sobre el feature map BezierAlign de ancho 32, lo que da espacio suficiente para palabras de hasta ~15 caracteres.

- **Mask R-CNN / RoIAlign** (He et al. 2017): el ancestro directo de BezierAlign. La diferencia clave es que RoIAlign muestrea sobre una grilla rectangular, mientras BezierAlign muestrea a lo largo de la curva. La filosofía es la misma: bilinear interpolation diferenciable evita perder precisión de localización al hacer pooling.

- **Spatial Transformer Networks (STN, Jaderberg 2015)** y **TPS (Thin Plate Splines)**: alternativas que ABCNet compara explícitamente. STN y TPS pueden rectificar texto curvo pero requieren predecir parámetros de transformación complejos (TPS necesita 20+ puntos control + función radial), mientras ABCNet rectifica con solo 8 puntos. Y crucialmente, los puntos Bezier de ABCNet **tienen significado geométrico interpretable** (son los extremos y vértices del texto), mientras los puntos de control TPS son abstractos.

## 11. Lectura crítica final

ABCNet es un ejemplo paradigmático de **"la representación correcta resuelve el problema"**. La sustitución de polígonos densos por una curva Bezier paramétrica tiene tres efectos en cascada:

1. **Reduce la dimensionalidad** del problema de regresión (16 vs 28+ escalares), facilitando que un detector denso anchor-free aprenda con poca data.
2. **Habilita un sampler diferenciable** (BezierAlign) que rectifica el texto en feature space, eliminando la necesidad de TPS, STN o post-procesamiento no diferenciable.
3. **Desacopla detection y recognition** sin sacrificar end-to-end-ness: la rama de reconocimiento ve siempre texto rectificado y puede ser ligera (CRNN-style con BLSTM + CTC), mientras la rama de detección hace todo el trabajo geométrico.

El trade-off es honesto: ABCNet **no es el más preciso** en su categoría (versiones posteriores de Mask TextSpotter con character supervision lo superan en accuracy puro), pero **domina la frontera Pareto FPS-vs-accuracy**. Para un practicante que necesita poner OCR en producción con latencia bounded — exactamente el caso de uso del Diplomado IA UC y de aplicaciones reales en healthcare, retail, logística — ABCNet representa el punto donde la teoría matemática (curvas Bezier, polinomios de Bernstein) se encuentra con la ingeniería pragmática (anchor-free, light recognizer, sampler diferenciable).

La lección transferible para cualquier proyecto de visión computacional: **antes de complicar el modelo, busca una representación más compacta del problema**.
