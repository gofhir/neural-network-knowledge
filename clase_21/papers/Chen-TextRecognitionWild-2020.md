---
title: "Text Recognition in the Wild: A Survey"
authors: ["Xiaoxue Chen", "Lianwen Jin", "Yuanzhi Zhu", "Canjie Luo", "Tianwei Wang"]
year: 2020
venue: "arXiv 2020 (J. ACM Vol. 1, No. 1, December 2020)"
slug: text-recognition-wild-chen-2020
---

# Análisis interno — Chen et al. (2020) "Text Recognition in the Wild: A Survey"

> Documento complementario al material público del site (clase 21 — OCR / Scene Text Recognition / Document AI). Aquí se profundiza en por qué este survey se volvió la referencia organizativa del campo de STR entre 2020 y 2022, qué taxonomía propone para clasificar 200+ papers, qué datasets y métricas codifica como estándar, qué tabla comparativa cierra el estado del arte pre-Transformer multimodal, y qué limitaciones tiene visto desde 2025-2026 (post TrOCR, PARSeq, foundation models multimodales).

- **Paper**: Chen, Jin, Zhu, Luo, Wang. *Text Recognition in the Wild: A Survey*. arXiv:2005.03492v3 (3 Dec 2020). Publicado también como artículo en *J. ACM* Vol. 1, No. 1, Diciembre 2020 (34 páginas).
- **Versiones**: v1 (May 2020), v2 (revisiones menores), v3 (Dec 2020) — la versión definitiva. Es importante saber que el survey se cierra con la literatura publicada antes de fines de 2020, por lo que **no incluye** TrOCR (Li et al. 2021), PARSeq (Bautista & Atienza 2022), MaskOCR (Lyu et al. 2022) ni los foundation models multimodales tipo Donut o LayoutLMv3 que dominan el campo desde 2022.
- **Código y recursos**: el repositorio oficial del survey con tablas vivas y enlaces a implementaciones es `https://github.com/HCIILAB/Scene-Text-Recognition`, mantenido por el laboratorio HCII de la South China University of Technology, donde trabajan los autores.
- **Institución**: College of Electronic and Information Engineering, South China University of Technology (Guangzhou, China). Es uno de los laboratorios más prolíficos en STR y document AI en idioma chino. Lianwen Jin (segundo autor) es además uno de los referentes mundiales de OCR para *Chinese text in the wild* y co-organizador de varias competencias ICDAR (RCTW-17, LSVT, ReCTS, MLT-2019).

---

## 1. Resumen ejecutivo

Un survey no aporta resultados experimentales nuevos: aporta **estructura**. Cuando un campo crece a la velocidad del *deep learning* aplicado a visión, la cantidad de papers publicados en 2-3 años desborda la capacidad de cualquier *practitioner* recién llegado para mapear qué es estado del arte, qué es derivativo, qué dataset es relevante y qué métrica reportar. Chen, Jin y colaboradores ocupan ese hueco para *Scene Text Recognition* (STR) — el subcampo de visión por computadora que se ocupa de leer texto en imágenes naturales, capturadas con teléfonos o cámaras *in the wild*, en condiciones de iluminación, oclusión, distorsión, fuente, color y orientación arbitrarias.

El paper cumple cuatro funciones esenciales para el curso IA UC clase 21 y para cualquier ingeniero que quiera entrar al campo. Primero, codifica el **vocabulario**: separa explícitamente STR de OCR clásico (Tesseract sobre documentos escaneados), enumera los problemas (localización, verificación, detección, segmentación, reconocimiento, end-to-end) y los issues especiales (script identification, enhancement, tracking, NLP downstream). Segundo, define la **pipeline canónica** de un STR moderno en cuatro etapas (preprocessing, feature extraction, sequence modeling, prediction) que se vuelve la rejilla mental con la que se enseña el campo durante los siguientes años. Tercero, **inventaría** los datasets más usados — sintéticos (Synth90k, SynthText, Verisimilar, UnrealText), regulares (IIIT5K, SVT, IC03/11/13, SVHN), irregulares (SVT-P, CUTE80, IC15, COCO-Text, Total-Text) y multilingües (RCTW-17, MTWI, CTW, CTW1500, LSVT, ArT, ReCTS, MLT) — con sus tamaños, ground truth disponibles y enlaces. Cuarto, presenta una **tabla comparativa monumental** (Table 4 del paper) con la accuracy de 50+ métodos sobre 8 benchmarks, que permite comparar de un vistazo CRNN, RARE, STAR-Net, ASTER, MORAN, SAR, AON, NRTR, ESIR y métodos de 2019-2020.

El valor del survey no está en cada celda de esa tabla, sino en haberla armado, mantenido y publicado con código abierto.

---

## 2. Por qué Scene Text Recognition no es OCR

Para alguien con formación en sistemas de información (FHIR, Go, documentos médicos), la distinción importa. **OCR clásico** — Tesseract, ABBYY FineReader, los sistemas comerciales de los noventa y dos mil — se diseñó para documentos escaneados: papel A4 o carta, fondo blanco, fuente regular tipo Times o Arial, alineación frontal, sin distorsión perspectiva, sin oclusión, capturado a 200-300 DPI con un escáner plano. Bajo esas condiciones la accuracy supera el 99% desde hace dos décadas. **STR** es la tarea de leer texto en una foto tomada con un teléfono a un letrero, una vitrina, una placa, un envase, un cartel callejero o un electrodoméstico.

La Tabla 1 y la Figura 1 del survey resumen la diferencia en cuatro ejes:

1. **Background**. El OCR escaneado asume fondo blanco uniforme. STR convive con paredes de ladrillo, pasto, agua, papel arrugado, texturas urbanas, otros textos cercanos, y reflejos. Peor: la textura del fondo puede parecerse visualmente al texto (ladrillos paralelos, ramas, ondas), provocando falsos positivos en la detección y errores de segmentación en el reconocimiento.
2. **Form (forma del texto)**. En un documento, la fuente es regular, el tamaño consistente y el arreglo uniforme. En la calle aparecen textos en mil colores, fuentes ornamentales, tamaños variables, *bold/italic* arbitrario, mezclas mayúscula-minúscula con propósito estético, escritura curvada sobre logos, texto vertical, espejado, en forma de arco, etc.
3. **Noise**. La imagen *in the wild* sufre iluminación no uniforme, baja resolución (texto que ocupa pocos píxeles porque está lejos), motion blur (foto sacada caminando), enfoque imperfecto, ruido de sensor en condiciones de poca luz.
4. **Access**. El escaneado garantiza imagen frontal y que el texto ocupe la mayor parte de la página. STR se enfrenta a distorsión perspectiva (mirar el texto en ángulo), curvatura física (texto sobre una taza, una pelota, una pasta de dientes), deformación geométrica arbitraria.

La consecuencia metodológica es clara: **las técnicas de OCR clásico no transfieren**. La pipeline de binarización + segmentación de caracteres + clasificador por carácter, que funciona en documento escaneado, fracasa en escena. Por eso STR vive en otro régimen: necesita modelos que sean simultáneamente *vision* (para extraer features robustas a distorsión y ruido) y *sequence* (para modelar la salida como secuencia de caracteres con dependencias contextuales).

El punto de quiebre histórico está en 2014. Jaderberg, Simonyan, Vedaldi y Zisserman (Oxford VGG) publican el *Synth90k dataset*: 9 millones de imágenes sintéticas generadas renderizando 90.000 palabras inglesas comunes sobre fondos naturales con transformaciones aleatorias (rotación, escala, blur, color, distorsión perspectiva). Es la primera vez que se entrena un STR *data-hungry* sin necesidad de anotar manualmente cada palabra de la calle. A partir de ahí los métodos *deep learning* superan rápidamente a los basados en *handcrafted features* tipo HOG, MSER, SWT (Stroke Width Transform) y CC (Connected Components). El survey marca este momento como la frontera entre la "era handcrafted" y la "era deep" de STR.

---

## 3. La taxonomía del survey: pipeline en cuatro etapas

El survey aporta como contribución principal el diagrama de la Figura 4 (página 7), donde la pipeline *segmentation-free* moderna se descompone en cuatro estaciones secuenciales. Esta es la rejilla que se usa para clasificar a posteriori los 50+ métodos de la Tabla 1.

### 3.1 Image Preprocessing Stage

Tres familias de técnicas operan antes del *feature extractor*:

- **Background removal**. Eliminar el fondo y dejar solo los píxeles del texto. La binarización clásica (Otsu, adaptive thresholding) funciona en documento pero falla en escena. Luo et al. (2020) usan GANs (red generativa adversaria) para aprender a separar foreground y background, reduciendo el costo cognitivo del reconocedor downstream.
- **Text Image Super-Resolution (TextSR)**. Cuando el texto ocupa pocos píxeles (lejano, foto pequeña), reconstruir una versión de alta resolución antes de reconocer. Wang et al. usan una red de super-resolución entrenada **conjuntamente** con el reconocedor, no como módulo independiente — la pérdida del reconocedor backpropaga al super-resolver para que reconstruya los detalles que importan a la lectura, no a la PSNR genérica.
- **Rectification networks**. La pieza más influyente. Una *Spatial Transformer Network* (STN, Jaderberg et al. 2015) se inserta como módulo diferenciable que aprende a desdistorsionar el texto antes de entregarlo al CNN. Para texto curvado o con perspectiva fuerte, Shi et al. en ASTER (2018) y Jeonghun et al. (2019) adoptan *Thin-Plate Spline* (TPS): una transformación no rígida controlada por puntos fiduciarios aprendidos. Variantes posteriores son MORAN (Luo et al. 2019) con múltiples objetos rectificados independientemente, ESIR (Zhan et al. 2019) con iterative TPS y *line-fitting transformation*, y Yang et al. (2019) con red de rectificación restringida por simetría.

La crítica del propio survey es honesta: módulos de rectificación complejos consumen memoria y tiempo, y conforme mejora la detección de texto irregular *upstream*, conviene reconsiderar si el rectificador es necesario.

### 3.2 Feature Extraction Stage

El backbone CNN que mapea la imagen de entrada (típicamente $32 \times 100$ píxeles para una palabra inglesa) a un mapa de features apto para reconocimiento. Las opciones canónicas:

- **VGGNet** (Simonyan & Zisserman 2014): el backbone original de CRNN (Shi 2015). Stack de bloques `conv-conv-pool`, 13 capas convolucionales, simple pero costoso en parámetros.
- **ResNet** (He et al. 2016): la mayoría de métodos modernos (STAR-Net, ASTER, MORAN, AON, NRTR, SAR) usan ResNet como backbone. Las *skip connections* permiten redes más profundas (50, 101 capas) sin colapsar el gradiente.
- **DenseNet** (Huang et al. 2017): conexiones densas entre todas las capas dentro de un bloque. Más eficiente en parámetros, usado por algunos métodos de 2018-2019.
- **Recursive CNN / Gated Recurrent Convolution**. Lee et al. (2016) introducen *recursive CNN* para STR — la misma capa convolucional se aplica $k$ veces, aumentando profundidad efectiva sin aumentar parámetros. Wang et al. (2017) extienden esto con un *gate* tipo LSTM dentro de la convolución (GRCNN) para controlar la modulación contextual.
- **Binary CNN**. Liu et al. proponen convoluciones binarias para STR *real-time*, acelerando inferencia a costa de algunos puntos de accuracy.
- **CNN + Attention**. He et al., Yang et al. y otros combinan CNN con módulos de atención visual para resaltar foreground del texto y suprimir ruido del background. Este es el germen de los métodos *2D-attention* posteriores (SAR).

### 3.3 Sequence Modeling Stage

Después del CNN, el mapa de features de tamaño $T \times C$ (donde $T$ es la dimensión a lo largo del eje del texto) se interpreta como una secuencia. El default desde 2015 hasta 2019 es **BiLSTM** (Hochreiter & Schmidhuber 1997 + dirección bidireccional): dos LSTM, uno hacia adelante y otro hacia atrás, concatenando sus *hidden states*. La justificación es que para leer "BANANA" hay que mirar tanto el contexto izquierdo como el derecho de cada carácter.

Sin embargo, varios autores cuestionan que BiLSTM sea necesario:

- **CNN sliding window / 1D-CNN**. Yin et al. (2017) y Borisyuk et al. (2018) reemplazan BiLSTM por una CNN unidimensional profunda, argumentando que con receptive field suficientemente grande el contexto local que necesita STR ya se captura, sin el costo computacional de la recurrencia y sin sus problemas de vanishing gradient.
- **Transformer**. Lyu et al. (2019, NRTR), Sheng et al. y otros introducen el *encoder-decoder Transformer* (Vaswani et al. 2017) para STR. Beneficios: paralelización completa de la secuencia (vs. recurrencia secuencial), mejor modelado de dependencias largas, escalabilidad. NRTR (*No-Recurrence sequence-to-sequence Transformer*) fue uno de los primeros en mostrar que un Transformer puro supera a BiLSTM+attention en STR.

El survey anticipa con claridad que la dirección dominante post-2020 será Transformer. La tendencia se confirma con TrOCR (Microsoft 2021), PARSeq (2022) y MaskOCR (2022) — todos basados en Vision Transformer o variantes.

### 3.4 Prediction Stage

La última estación convierte la secuencia de features en la secuencia de caracteres de salida. Dos técnicas dominan el campo:

**Connectionist Temporal Classification (CTC)** — propuesta por Graves, Fernández, Gomez y Schmidhuber (ICML 2006) para *unsegmented sequence labelling* en reconocimiento de voz. La idea brillante: en lugar de obligar al modelo a alinear cada timestep con un carácter de salida (lo que requeriría anotación de fronteras de carácter, costosa), CTC introduce un símbolo *blank* $\varepsilon$ y suma sobre todos los caminos posibles que mapean a la misma transcripción.

Formalmente, sea $y = (y_1, \dots, y_T)$ la secuencia de features de entrada, donde cada $y_t$ es una distribución sobre el alfabeto $\mathcal{L} \cup \{\varepsilon\}$. Un camino $\pi \in (\mathcal{L} \cup \{\varepsilon\})^T$ es una secuencia de etiquetas de longitud $T$. La función $\mathcal{B}$ colapsa repeticiones y borra el *blank*: $\mathcal{B}(a\varepsilon ab b) = aab$. La probabilidad de una transcripción $l$ dada la entrada $y$ es:

$$
p(l \mid y) = \sum_{\pi : \mathcal{B}(\pi) = l} p(\pi \mid y), \quad p(\pi \mid y) = \prod_{t=1}^{T} y^t_{\pi_t}
$$

Computar esa suma directamente es exponencial, pero el algoritmo *forward-backward* la calcula en $O(T \cdot |l|)$. He et al. (2016) y Shi et al. (CRNN 2015) trasladaron CTC de voz a STR, y desde entonces aparece en CRNN, STAR-Net, GRCNN, FAN, EnEdiTC, GTC, ABCNet, entre muchos otros.

Limitaciones de CTC reconocidas por el survey: (i) sufre el *peaky distribution problem* — las predicciones se concentran en muy pocos timesteps y el modelo se vuelve overconfident, lo que Liu et al. (2018) tratan con regularización por máxima entropía; (ii) le cuesta repeticiones (el dictado "AA" requiere un *blank* entre ambas); (iii) **no escala a 2D** — CTC asume secuencia 1D, por lo que para texto curvado o multi-oriented requiere un módulo de rectificación previo. Wan et al. (2020) proponen una extensión 2D-CTC con dimensión adicional de altura, pero el problema 2D no está completamente resuelto.

**Attention-based decoder** — Bahdanau, Cho y Bengio (ICLR 2015) propusieron *attention* en traducción automática para alinear soft entre tokens fuente y target. Trasladado a STR, el decoder GRU emite carácter a carácter consultando un *glimpse vector* sobre los features del encoder:

$$
\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{i=1}^{N} \exp(e_{t,i})}, \quad e_{t,j} = \tanh(W_s s_{t-1} + W_h h_j + b)
$$

donde $s_{t-1}$ es el estado oculto previo del decoder y $h_j$ son los features del encoder. El glimpse es $g_t = \sum_j \alpha_{t,j} h_j$ y la predicción $o_t = \mathrm{softmax}(W_o s_t + b_o)$, con $s_t = \mathrm{GRU}(o_{prev}, g_t, s_{t-1})$.

Variantes documentadas por el survey:

- **2D attention** para texto irregular: SAR (Show, Attend and Read — Li, Wang, Shen, Zhang 2019), AON (Cheng et al. 2018), Yang et al. (2017) — atienden sobre el mapa de features 2D en lugar de aplastarlo a 1D.
- **Modelado de lenguaje implícito mejorado**: Chen et al. (2020) introducen *higher-order character language model*; Wang et al. (2018, MAAN) usan *memory-augmented attention*.
- **Bidireccional**: Shi et al. (ASTER) usan dos decoders en direcciones opuestas y combinan, evitando el sesgo direccional.
- **Transformer attention**: NRTR, ScRN y otros reemplazan la atención sobre RNN por atención multi-head pura.
- **Attention drift**. Cheng et al. (2017, FAN — *Focusing Attention Network*) identifican que el módulo de atención puede desalinearse del target región y proponen supervisión adicional sobre la localización. Wang et al. (2019) argumentan que el problema viene de usar el historial recurrente y proponen desacoplar la atención del estado del decoder.

El veredicto del survey: la atención supera a CTC en accuracy en *isolated word recognition*, pero CTC es más rápido y robusto en oraciones largas. Cong et al. (2019) hacen una comparación empírica sistemática que respalda esta conclusión. Métodos híbridos (Hu et al. 2020, GTC; Litman et al. 2020, SCATTER) combinan ambos para tener lo mejor de ambos mundos durante el entrenamiento.

---

## 4. Datasets canónicos

El inventario de datasets es uno de los aportes más útiles del survey para *practitioners*. La Table 3 (página 16) resume language, tamaño y tipo. Selección curada:

### 4.1 Sintéticos

| Dataset | Idioma | Instancias | Propósito |
|---|---|---|---|
| **Synth90k** (Jaderberg 2014) | Inglés | ~9 M | El estándar de entrenamiento STR latino. Render de 90k palabras con transformaciones. |
| **SynthText** (Gupta et al. 2016) | Inglés | ~6 M (en 800k imágenes) | Texto sobre escenas naturales segmentadas. Incluye localización + transcripción. |
| **Verisimilar Synthesis** (Zhan et al. 2018) | Inglés | ~5 M | Embedding de texto en posiciones semánticamente apropiadas (mapa de saliency). |
| **UnrealText** (Long & Yao 2020) | Inglés | ~12 M (en 600k imágenes) | Renderizado en Unreal Engine 4 con física y mallas 3D. Sintético más realista. |

Sin Synth90k y SynthText, la mayoría de los métodos modernos no podrían entrenarse — los datasets realísticos contienen miles, no millones, de instancias.

### 4.2 Realísticos regulares (latín)

| Dataset | Train / Test | Léxico | Tipo |
|---|---|---|---|
| **IIIT5K-Words** | 2.000 / 3.000 | 50w + 1.000w | Imágenes web e in-situ. |
| **SVT** (Street View Text) | 100 / 250 | 50w | Google Street View. |
| **IC03** (ICDAR 2003) | 258 / 251 (867 cropped) | 50w + full | Competencia ICDAR. |
| **IC11** | 485 imágenes | — | Extensión IC03. |
| **IC13** | 420 / 141 (1.015 cropped) | sin léxico | ¡Cuidado!: 215 instancias duplicadas con IC03 test. |
| **SVHN** (Street View House Numbers) | >600.000 dígitos | — | Solo dígitos, números de casa. |

### 4.3 Realísticos irregulares

| Dataset | Instancias | Distorsión |
|---|---|---|
| **SVT-P** (StreetViewText-Perspective) | 639 cropped | Perspectiva no-frontal. |
| **CUTE80** | 288 cropped | Texto curvado en logos y carteles. |
| **IC15** (ICDAR 2015 Incidental) | 1.000 / 500 (2.077 cropped) | Google Glasses, sin control, multi-oriented. |
| **COCO-Text** | 145.859 cropped | Primer dataset masivo en escenas naturales con atributos. |
| **Total-Text** | 11.459 cropped | Foco en texto curvado, multi-oriented. |

### 4.4 Multilingüe (énfasis en chino)

El survey dedica una sección importante al chino, justificadamente: (i) China tiene la mayor base de usuarios STR en el mundo, (ii) el alfabeto chino tiene miles de categorías (vs. ~62 alfanuméricas latinas), (iii) hay desbalance fuerte entre caracteres comunes y raros, (iv) muchos pares de caracteres se parecen visualmente.

| Dataset | Idioma | Instancias |
|---|---|---|
| **RCTW-17** (Reading Chinese in the Wild) | Chino | 12.514 imágenes |
| **MTWI** (Multi-Type Web Images) | Chino + Latín | 20.000 |
| **CTW** (Chinese Text in the Wild) | Chino | 32.285 imágenes / 1M+ caracteres |
| **SCUT-CTW1500** | Chino + Latín | 1.500 (3.530 curved) |
| **LSVT** (Large-Scale Street View) | Chino | 450.000 (mix supervisión fuerte y débil) |
| **ArT** (Arbitrary-Shaped Text) | Chino + Latín | 10.166 |
| **ReCTS-25k** | Chino | 25.000 imágenes (signboards) |
| **MLT-2019** | 10 idiomas (Árabe, Bangla, Chino, Devanagari, Inglés, Francés, Alemán, Italiano, Japonés, Coreano) | 20.000 |

---

## 5. Métricas estándar

### 5.1 Para reconocimiento (latino)

**Word Recognition Accuracy (WRA)** — la métrica dominante en latino. Es simplemente la fracción de palabras correctamente predichas:

$$
\mathrm{WRA} = \frac{W_r}{W}
$$

donde $W$ es el total de palabras y $W_r$ las correctas. Su complemento es el *Word Error Rate*, $\mathrm{WER} = 1 - \mathrm{WRA}$.

WRA tiene un comportamiento *all-or-nothing*: una palabra con un solo carácter mal cuenta como error. Para latino, dado que las palabras son cortas (4-10 caracteres), WRA es una métrica razonable.

### 5.2 Para reconocimiento multilingüe

**Normalized Edit Distance (NED)** — adoptada por las competencias ICDAR para chino y multilingüe:

$$
\mathrm{NED} = \frac{1}{N} \sum_{i=1}^{N} \frac{D(s_i, \hat{s}_i)}{\max(l_i, \hat{l}_i)}
$$

donde $D(\cdot)$ es la distancia de Levenshtein entre la predicción $s_i$ y el ground truth $\hat{s}_i$, $l_i$ y $\hat{l}_i$ son sus longitudes y $N$ el número de líneas de texto. La métrica final reportada es típicamente $1 - \mathrm{NED}$. La normalización por el máximo de longitudes evita que líneas largas dominen el promedio. NED captura la diferencia parcial: si el modelo lee 9 de 10 caracteres correctamente, NED refleja el 90% acertado, mientras WRA marca error total.

### 5.3 Para end-to-end

Para sistemas que hacen detección + reconocimiento conjuntos, las métricas combinan IoU de bounding box con string match:

- **Precision, Recall, F-score basados en NED**. Una predicción se considera correcta si la bbox tiene IoU > 0.5 con el ground truth Y el string predicho coincide (o tiene $1 - \mathrm{NED}$ alto). F-score es la media armónica.
- **End-to-end recognition vs. word spotting**. En *end-to-end* se evalúa todo: se detecta y se reconoce. En *word spotting* solo se evalúan palabras presentes en un vocabulario dado. Hay tres niveles: **strongly contextualised (S)** — vocabulario por imagen con 100 palabras; **weakly contextualised (W)** — vocabulario de train/test; **generic (G)** — diccionario de ~90k palabras de Jaderberg.

---

## 6. Métodos representativos por estación

### 6.1 Detección

Aunque el foco es STR (reconocimiento), el survey resume métodos de detección porque el end-to-end los integra. Los nombres a memorizar:

- **TextBoxes** (Liao et al. 2017) y **TextBoxes++** (Liao et al. 2018): SSD adaptado a cajas largas y oblicuas.
- **EAST** (Zhou et al. 2017): regresión directa de quad/rotated-rectangle por píxel, una de las primeras detecciones rápidas multi-oriented.
- **CRAFT** (Baek et al. 2019): *Character Region Awareness for Text Detection*. Predice mapas de score por carácter y por afinidad entre caracteres, lo que permite construir polígonos de palabras o líneas a partir de caracteres.
- **PSENet** (Wang et al. 2019, Progressive Scale Expansion): kernels concéntricos que crecen desde el centro de cada instancia para separar textos cercanos.
- **DBNet** (Liao et al. 2020, Differentiable Binarization): binarización aprendible que produce mapas más nítidos en tiempo real.

### 6.2 Reconocimiento

- **CRNN** (Shi, Bai & Yao 2015) — el clásico. VGG + BiLSTM + CTC. Es la baseline contra la que se compara todo lo demás durante 5 años.
- **RARE** (Shi et al. 2016) — primer reconocedor con STN integrada.
- **STAR-Net** (Liu et al. 2016) — *SpaTial Attention Residue Network*: ResNet backbone + attention espacial + CTC.
- **ASTER** (Shi et al. 2018) — *Attentional Scene TExt Recognizer*: rectificación TPS aprendida + ResNet + BiLSTM + attention bidireccional. Referencia para texto irregular.
- **MORAN** (Luo, Jin & Sun 2019) — *Multi-Object Rectified Attention Network*: rectificación multi-objeto donde cada parte del texto se rectifica con su propia transformación.
- **SAR** (Li, Wang, Shen, Zhang 2019) — *Show, Attend and Read*: 2D attention sobre el mapa de features, descarta rectificación explícita.
- **AON** (Cheng et al. 2018) — *Arbitrarily-Oriented Network*: extrae features en cuatro orientaciones y los combina.
- **NRTR** (Sheng, Chen & Xu 2019) — *No-Recurrence sequence-to-sequence Transformer*: primer reconocedor STR puramente Transformer, sin RNN.
- **DAN** (Wang et al. 2020) — *Decoupled Attention Network*: separa la alineación de atención del estado del decoder.

### 6.3 End-to-end

- **Mask TextSpotter** (Lyu, Liao, Yang, Bai, Bai 2018; extensión Liao et al. 2019): Mask R-CNN adaptado para detectar y reconocer texto de forma arbitraria con anotaciones character-level.
- **CharNet** (Xing et al. 2019): detección + reconocimiento character-level conjunto.
- **TextDragon** (Feng, He, Yin, Zhang, Liu 2019): framework para texto de forma arbitraria con CTC.
- **ABCNet** (Liu, Chen, Shen, He, Jin, Wang 2020): adaptive Bezier-curve network — modela el contorno del texto como curvas de Bézier paramétricas y propone *BezierAlign* (análogo a RoIAlign pero sobre Bézier) para extraer features rectificadas. Este es el otro paper de cabecera de la clase 21, también descargado.

---

## 7. Tabla comparativa (estado del arte 2015-2020)

La Table 4 del survey (página 22) es la tabla más útil. Selección curada con accuracy en los benchmarks principales (sin léxico, salvo indicación):

| Método (año) | IIIT5K | SVT | IC03 | IC13 | IC15 | SVTP | CUTE80 |
|---|---|---|---|---|---|---|---|
| Wang ABBYY (2011) | 24.3 | 35.0 | 56.0 | — | — | 40.5 | — |
| Jaderberg dict (2014) | — | 71.7 | 89.6 | 81.8 | — | — | — |
| Shi CRNN (2015) | 78.2 | 80.8 | 89.4 | 86.7 | — | — | — |
| Shi RARE (2016) | 81.9 | 81.9 | 90.1 | 88.6 | — | 71.8 | 59.2 |
| Lee R2AM (2016) | 78.4 | 80.7 | 88.7 | 90.0 | — | — | — |
| Shi STAR-Net (2016) | 83.3 | 83.6 | 89.9 | 89.1 | — | 73.5 | — |
| Cheng FAN (2017) | 87.4 | 85.9 | 94.2 | 93.3 | 70.6 | — | — |
| Cheng AON (2018) | 87.0 | 82.8 | 91.5 | — | 68.2 | 73.0 | 76.8 |
| Bai EP (2018) | 88.3 | 87.5 | 94.6 | 94.4 | 73.9 | — | — |
| Shi ASTER (2018) | 93.4 | 89.5 | 94.5 | 91.8 | 76.1 | 78.5 | 79.5 |
| Liu MORAN (2019) | 91.2 | 88.3 | 95.0 | 92.4 | 68.8 | 76.1 | 77.4 |
| Li SAR (2019) | 91.5 | 84.5 | — | 91.0 | 69.2 | 76.4 | 83.3 |
| Sheng NRTR (2019) | 90.1 | 91.5 | 95.8 | 95.8 | — | 79.4 | — |
| Yang ScRN (2019) | 94.4 | 88.9 | 95.0 | 93.9 | 78.7 | 80.8 | 87.5 |
| Wang DAN (2020) | 94.3 | 89.2 | 95.0 | 93.9 | 74.5 | 80.0 | 84.4 |
| Yu SRN (2020) | 94.8 | 91.5 | — | 95.5 | 82.7 | 85.1 | 87.8 |
| Qiao SEED (2020) | 93.8 | 89.6 | — | 92.8 | 80.0 | 81.4 | 83.6 |

Cinco lecturas:

1. Entre 2011 y 2020 la accuracy en IIIT5K pasa de 24 a 94+ — diez veces el bajo techo de los métodos pre-deep.
2. ASTER (2018) es el punto de inflexión que pasa la barrera del 90% en latín regular.
3. NRTR (2019) es el primer Transformer puro que aparece arriba del podio, anticipando el cambio.
4. Los datasets irregulares (IC15, SVTP, CUTE80) tienen 10-15 puntos menos que los regulares — el reto no estaba resuelto en 2020.
5. SRN (Yu et al. 2020) introduce módulo de razonamiento semántico global, anunciando la dirección "reconocimiento + lenguaje" que luego dominará TrOCR.

---

## 8. Retos abiertos según el survey (sección 5)

El capítulo de cierre identifica siete líneas:

1. **Generalization ability**. Modelos entrenados en sintético rinden bien en IIIT5K/SVT pero caen en COCO-Text. No generalizan a fuentes nuevas, tamaños pequeños o caracteres largos. El gap con la lectura humana sigue siendo enorme.
2. **Evaluation protocols**. La inconsistencia entre experimentos (qué datos de entrenamiento se usaron, qué anotación, qué léxico) hace incomparable la Table 4 entre filas. El survey pide un benchmark unificado — petición que Baek et al. (2019) "*What is wrong with scene text recognition model comparisons?*" ya empezó a responder.
3. **Data issues**. Los datasets reales son muy chicos (miles de instancias). Hay que mejorar la síntesis (e.g., UnrealText con motor 3D) y explorar self/unsupervised learning sobre datos no anotados.
4. **Scenarios**. Las aplicaciones reales (tarjetas bancarias, IDs, licencias de conducir) demandan precisión casi perfecta y son privadas. La investigación académica no las cubre.
5. **Image preprocessing**. Background removal y TextSR siguen siendo subexplorados.
6. **End-to-end systems**. Quedan lejos del OCR escaneado y deben balancear la velocidad de convergencia distinta entre detección y reconocimiento.
7. **Languages**. La mayoría de algoritmos solo cubren latín. Chino, árabe, devanagari requieren tratamiento específico.
8. **Security**. Los modelos STR son vulnerables a ataques adversariales — relevante para autenticación.
9. **STR + NLP**. La combinación con NLP (text VQA, document understanding, information extraction) es el siguiente frontier.

---

## 9. Por qué importa para el curso IA UC

El survey aterriza la clase 21 (OCR / STR / Document AI) en tres dimensiones:

- **Pipeline mental**. Las cuatro estaciones (preprocessing → feature extraction → sequence modeling → prediction) son la rejilla para entender cualquier paper de OCR moderno. Cuando llegamos a TrOCR (clase 21+), la pregunta natural es: ¿cómo se mapean preprocessing/feature/sequence/prediction al esquema encoder-decoder Transformer? (respuesta: ViT como feature+sequence, RoBERTa-decoder como prediction; preprocessing implícito en data augmentation).
- **Conexiones cruzadas con otras clases**:
  - **Clase 17 (Pose Recognition)**: ambos son tareas de *vision regression* sobre puntos. La rectificación TPS de ASTER es conceptualmente prima de los grafos cinemáticos de OpenPose — ambos modelan deformación geométrica con puntos fiduciarios.
  - **Clase 18 (Word Embeddings)**: el módulo de prediction de STR emite caracteres, no palabras — pero los métodos *semantic-aware* de 2020 (SRN, SEED) usan embeddings de palabras pre-entrenadas para guiar la predicción.
  - **Clase 09 (CNN)**: el backbone (VGG, ResNet, DenseNet) es exactamente la materia de clase 09. STR demuestra cómo un CNN diseñado para clasificación se reutiliza como *feature extractor* para una tarea totalmente diferente.
  - **Clase 14 (Transformer)**: NRTR es el puente. El paso de BiLSTM a Transformer en STR sigue la misma motivación que en NLP — paralelización y dependencias largas. La crónica de la clase 14 sobre attention se aplica casi literalmente al decoder de STR.
- **Datasets como cultura**. Conocer IIIT5K, SVT, IC13, IC15, SVTP, CUTE80 es como conocer ImageNet, COCO o GLUE: ningún paper de STR se puede leer sin entender qué prueba sobre qué conjunto. El survey los inventa todos en un solo lugar.

Para un practitioner de FHIR / documentos médicos, la conexión es directa: el OCR de cédulas, recetas médicas manuscritas, etiquetas de medicamentos y carnés de identidad cae en STR, no en OCR clásico. El survey indica cuál es la pipeline que conviene replicar (CRNN o ASTER como baseline, DBNet o CRAFT para detección) y qué dataset multilingüe usar (MLT-2019 si hay mezcla de latín y otros scripts).

---

## 10. Limitaciones del survey desde 2026

El survey se cierra a fines de 2020. Cinco años después, hay cuatro cosas que el lector debe saber que no aparecen:

1. **TrOCR** (Li et al. 2021, Microsoft). Reemplaza completamente CNN + BiLSTM por un encoder Vision Transformer (ViT/BEiT/DeiT pre-entrenado en ImageNet) y un decoder Transformer pre-entrenado en NLP (RoBERTa o variantes). Pre-entrenamiento con cientos de millones de líneas sintéticas de texto. Establece nuevo SOTA en IAM (handwriting) y IC13, IC15. La era CRNN-style termina.
2. **PARSeq** (Bautista & Atienza, ECCV 2022). *Permutation-based attention*: durante el entrenamiento se permuta el orden de los tokens de salida, permitiendo al decoder aprender *internal language modeling* bidireccional. Casi alcanza performance humana en benchmarks latinos.
3. **MaskOCR** (Lyu et al. 2022). Aplica *masked image modeling* (BEiT-style) al pre-entrenamiento de STR — el encoder ViT aprende a reconstruir patches de texto enmascarados, lo que mejora robustez a oclusión y variación de fuente.
4. **Foundation models multimodales para OCR / Document AI**. Donut (Kim et al. 2022) hace document understanding *end-to-end* sin OCR explícito. LayoutLMv3 (Huang et al. 2022) combina texto + layout + imagen. GPT-4V (OpenAI 2023), Claude 3.5/3.7 con visión y Gemini 1.5/2.0 (Google) realizan OCR multilingüe zero-shot directamente desde imagen, sin entrenamiento específico STR. Esta convergencia OCR → modelo multimodal general es probablemente el cambio más grande post-2020.

Otras limitaciones intrínsecas del survey: el sesgo hacia latín y chino (los autores son HCII-SCUT, los datasets multilingües privilegian Asia); la cobertura ligera de handwriting (IAM solo se menciona de pasada); la ausencia de papers de 2020 publicados muy cerca del corte (ej. SRN, SEED solo aparecen en la tabla, no en la discusión narrativa).

---

## 11. Referencias clave del survey

Las cinco referencias que son obligatorias para construir un mapa mental del campo a partir de este survey:

1. **Jaderberg, Simonyan, Vedaldi, Zisserman (2014)**. "Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition". NIPS-W. → Synth90k. El paper que crea el dataset que entrena a todo el campo.
2. **Shi, Bai, Yao (2015 / TPAMI 2017)**. "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition". → CRNN. La baseline canónica VGG + BiLSTM + CTC.
3. **Graves, Fernández, Gomez, Schmidhuber (2006)**. "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks". ICML. → CTC. La función de pérdida que hace posible STR end-to-end sin segmentación de caracteres.
4. **Shi, Yang, Wang, Lyu, Yao, Bai (2018, TPAMI 2019)**. "ASTER: An Attentional Scene Text Recognizer with Flexible Rectification". → ASTER. El reconocedor de irregular más influyente: TPS + ResNet + BiLSTM + attention bidireccional.
5. **Lyu, Liao, Yang, Bai, Bai (2018 / TPAMI 2019)**. "Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes". → Mask TextSpotter. End-to-end con anotación character-level.

Como sexta referencia complementaria, el otro paper de la clase 21 — **Liu, Chen, Shen, He, Jin, Wang (2020)**, "ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network". Modela texto curvado con curvas de Bézier paramétricas, evitando rectificación explícita. Es el último paper que aparece en las Tablas del survey y conecta con la frontera 2020.

---

## 12. Cierre

El valor de un survey bien hecho es que ahorra meses de exploración. Para un ingeniero entrando a STR en 2026, este paper de Chen et al. sigue siendo la mejor puerta de entrada **hasta 2020**, complementado obligatoriamente con TrOCR (2021), PARSeq (2022) y la literatura de foundation models multimodales (2023+). La taxonomía de cuatro estaciones, la lista de datasets, el contraste CTC vs. attention y las métricas WRA/NED son herramientas conceptuales que no envejecen: aun cuando los modelos cambien, los datasets y las métricas siguen siendo los mismos. Por eso el survey se merece estar en la bibliografía permanente de la clase 21 del Diplomado IA UC.
