---
title: "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"
authors:
  - Baoguang Shi
  - Xiang Bai
  - Cong Yao
year: 2017
venue: "TPAMI 2017 (arXiv 2015)"
slug: crnn-shi-2017
tags:
  - scene-text-recognition
  - sequence-recognition
  - ctc
  - blstm
  - end-to-end
  - ocr
---

# CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition

**Autores:** Baoguang Shi, Xiang Bai, Cong Yao (Huazhong University of Science and Technology, Wuhan).
**Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2017. Versión inicial en arXiv: 1507.05717, julio 2015.

---

## 1. Resumen ejecutivo

CRNN (Convolutional Recurrent Neural Network) es la arquitectura que en 2015 logra unificar, por primera vez, el reconocimiento de texto en escena (scene text recognition, STR) en un único modelo **end-to-end** entrenable que: (i) opera sobre imágenes de palabras recortadas con longitud arbitraria, (ii) no requiere segmentación a nivel de carácter, (iii) no está restringido a un léxico cerrado y (iv) tiene apenas 8.3M parámetros. El esqueleto es deliberadamente simple: una CNN tipo VGG extrae un mapa de características cuya altura se colapsa a 1, transformándolo en una secuencia de descriptores ordenados de izquierda a derecha; sobre esa secuencia se monta un BLSTM profundo (dos capas bidireccionales apiladas) que produce, para cada timestep, una distribución sobre el alfabeto extendido con un símbolo *blank*; finalmente, una capa de transcripción basada en **CTC (Connectionist Temporal Classification, Graves 2006)** integra todas las alineaciones posibles entre frames y caracteres para definir una verosimilitud diferenciable sobre la palabra objetivo.

Las consecuencias son profundas. CRNN reemplaza pipelines heterogéneos —binarización, detección de líneas, segmentación de caracteres, clasificación individual y reranking con HMM o CRF— por un solo grafo de cómputo entrenable. Solo necesita pares (imagen recortada, palabra). Bate o iguala el estado del arte en IIIT5k, SVT, IC03 e IC13, y a la vez es 30 a 60 veces más liviano que los modelos competidores de Jaderberg (490M / 304M parámetros). Como prueba de generalidad, los autores lo aplican a reconocimiento óptico de partituras musicales (OMR) y superan en márgenes muy amplios a productos comerciales (Capella Scan, PhotoScore).

Para el curso, CRNN es el ejemplar canónico de la etapa "Sequence Modeling" descrita en la clase 21: cualquier reconocedor moderno de texto en escena —ASTER, MORAN, SAR, NRTR, ABINet, PARSeq— hereda directamente el patrón **CNN encoder → sequence module → CTC o attention decoder**. ABCNet (Liu 2020, también de la clase 21) sustituye CTC por un decodificador attention-based, pero conserva el esqueleto convolucional + BLSTM sobre features rectificadas. Comprender CRNN es comprender la espina dorsal del reconocimiento de secuencias visuales de la última década.

---

## 2. Contexto histórico

### 2.1 La era pre-deep learning

Reconocer texto en escena (no documentos) fue durante décadas un problema considerado intratable. Tesseract y motores OCR clásicos asumen documentos escaneados con fondo blanco, líneas rectas, fuentes conocidas y alto contraste. Las palabras en una fotografía urbana —letrero de tienda, placa de calle, camiseta, graffiti— violan todos esos supuestos: iluminación variable, oclusión, perspectiva, fuentes decorativas, colores arbitrarios, fondo cluttered.

El paradigma dominante hasta 2012-2013 era una **pipeline en cinco etapas**: (1) binarización con métodos tipo Otsu o adaptativos para separar foreground del fondo; (2) detección y segmentación de caracteres individuales mediante connected components o sliding window; (3) extracción de descriptores hand-crafted (HOG, SIFT, Strokelets de Yao 2014); (4) clasificación carácter a carácter con SVM, RF o redes shallow; (5) post-procesamiento lingüístico con HMM, CRF, n-gramas o léxicos para corregir errores y resolver ambigüedades. Cada etapa se entrenaba y ajustaba por separado; errores en binarización se propagaban irrecuperablemente, y la segmentación de caracteres era el cuello de botella endémico —el clásico problema "Sayre's paradox": para segmentar un carácter necesitas reconocerlo, pero para reconocerlo necesitas haberlo segmentado.

Wang, Babenko y Belongie (ICCV 2011) introdujeron el dataset SVT (Street View Text) y el primer pipeline serio de end-to-end recognition con sliding window + Random Ferns + pictorial structure. Mishra et al. (BMVC 2012) propusieron "higher order language priors" con CRF sobre caracteres. Yao et al. (CVPR 2014) propusieron Strokelets, descriptores multi-escala aprendidos. Almazán et al. (TPAMI 2014) plantearon embeddings conjuntos de imagen y string (label embedding) que convierten reconocimiento en retrieval.

### 2.2 La irrupción del deep learning (2012-2014)

El éxito de AlexNet (Krizhevsky 2012) en ImageNet detonó intentos de aplicar CNN a STR. Tres líneas principales:

**Carácter-céntrica:** Wang, Wu, Coates y Ng (ICPR 2012) entrenan una CNN que clasifica caracteres aislados; integran con sliding window y NMS para detectar caracteres en la imagen, luego usan beam search sobre un grafo léxico. Alsharif y Pineau (ICLR 2014) extienden con HMM maxout. Estos sistemas heredan el problema fundamental: requieren **anotaciones a nivel de carácter** (bounding box por letra), costosas de producir y propensas a inconsistencias.

**Palabra-céntrica:** Jaderberg, Simonyan, Vedaldi y Zisserman ("Reading Text in the Wild", IJCV 2015 / NIPS DL Workshop 2014) entrenan una CNN gigante (490M parámetros, 90.000 clases) que clasifica directamente una imagen de palabra recortada como una de 90k palabras inglesas pre-definidas. Funciona espectacularmente bien en accuracy, pero tiene tres limitaciones críticas: (i) **vocabulario cerrado** —no puede leer números de teléfono, matrículas, nombres propios, palabras chinas, partituras musicales—; (ii) modelo enorme (490M parámetros, casi 2GB); (iii) no escala a alfabetos con combinatoria abierta (chino tiene >50k caracteres).

**Pipelines híbridos:** PhotoOCR de Bissacco et al. (Google, ICCV 2013) construye un sistema industrial: deep CNN para clasificación de caracteres + segmentación con CRF + reranking con n-gramas. Logra 78% en SVT pero requiere **7.9 millones de imágenes reales con anotaciones a nivel de carácter** —un dataset privado de Google inviable de reproducir académicamente.

### 2.3 El gap que CRNN cierra

En 2015, ningún método cumplía simultáneamente:

1. **End-to-end entrenable** desde imagen cruda hasta string, con un solo loss.
2. **Vocabulario abierto** —capaz de producir cualquier string, no solo elementos de un diccionario.
3. **Sin anotaciones a nivel de carácter** —solo etiquetas word-level.
4. **Longitud variable** —sin asumir ancho fijo ni número fijo de caracteres.
5. **Compacto** —desplegable en hardware modesto.

Las RNN ya se habían usado para STR (Graves para handwriting, Su y Lu ACCV 2014 con HOG sequence + RNN), pero requerían un paso de extracción de features manual previo, rompiendo el end-to-end. La idea germinal de CRNN es **conectar la CNN directamente al RNN mediante un map-to-sequence operator y entrenar todo con CTC**, idea que en speech recognition ya había explotado Graves (DeepSpeech, 2014) pero que nadie había llevado limpiamente a visión.

---

## 3. Arquitectura CRNN

CRNN se descompone en tres bloques apilados, de abajo hacia arriba: **convolutional layers** (extracción), **recurrent layers** (modelado de secuencia) y **transcription layer** (decodificación). Todo el grafo es diferenciable y se entrena con un único loss CTC.

### 3.1 Bloque convolucional: feature sequence extraction

La columna vertebral es una variante de VGG-VeryDeep (Simonyan-Zisserman 2014) adaptada a entrada rectangular. La configuración exacta (Tabla 1 del paper) es:

| Capa | Configuración |
|------|---------------|
| Input | $W \times 32$, grayscale |
| Conv1 | 64 mapas, $3{\times}3$, s=1, p=1 |
| MaxPool1 | $2{\times}2$, s=2 |
| Conv2 | 128 mapas, $3{\times}3$, s=1, p=1 |
| MaxPool2 | $2{\times}2$, s=2 |
| Conv3 | 256 mapas, $3{\times}3$, s=1, p=1 |
| Conv4 | 256 mapas, $3{\times}3$, s=1, p=1 |
| MaxPool3 | **$1{\times}2$**, s=2 (rectangular) |
| Conv5 | 512 mapas, $3{\times}3$, s=1, p=1 + BatchNorm |
| Conv6 | 512 mapas, $3{\times}3$, s=1, p=1 + BatchNorm |
| MaxPool4 | **$1{\times}2$**, s=2 (rectangular) |
| Conv7 | 512 mapas, $2{\times}2$, s=1, p=0 |

Hay dos tweaks claves respecto a una VGG canónica:

1. **Pooling rectangular $1 \times 2$ en MaxPool3 y MaxPool4** (en lugar del clásico $2 \times 2$). Sólo se sub-muestra el alto, no el ancho. Esto preserva mayor resolución horizontal, lo cual es crítico porque cada columna del feature map final corresponderá a un timestep del RNN. Para una palabra de 10 caracteres con imagen de $100 \times 32$, se obtiene una secuencia de 25 frames —suficiente granularidad para alinear caracteres delgados como `i` o `l` con varios frames cada uno.

2. **BatchNormalization tras Conv5 y Conv6**, esencial para entrenar conjuntamente capas profundas + LSTM. Sin BN, el entrenamiento se torna inestable por el cambio de estadísticas internas.

Tras Conv7 (kernel $2 \times 2$ con padding 0), el feature map tiene altura exactamente 1: dimensiones $(512, 1, W')$ con $W' = W/4$. Esa altura colapsada es el truco arquitectural fundamental.

### 3.2 Map-to-sequence

Operación trivial pero conceptualmente decisiva. El tensor $(C, H{=}1, W')$ se reinterpreta como una **secuencia de $W'$ vectores de $C$ canales**:

$$\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_{W'}), \quad \mathbf{x}_t \in \mathbb{R}^{512}.$$

Cada $\mathbf{x}_t$ es el descriptor de una **columna** del feature map; equivalentemente, es la representación convolucional de una **franja vertical** de la imagen original. Esa franja —el campo receptivo de esa columna— es el "carácter en bruto" que el RNN deberá procesar. Importantemente, las franjas se **solapan** (el receptive field excede el ancho de un stride), así que múltiples timesteps consecutivos pueden ver porciones de un mismo carácter. Eso es por diseño: CTC necesita poder emitir un carácter en varios frames contiguos y luego colapsarlos.

La inversión de esta operación es lo que conecta los gradientes del RNN de vuelta al CNN durante backprop; los autores lo llaman explícitamente "Map-to-Sequence layer".

### 3.3 Recurrent layers: Deep BLSTM

Sobre la secuencia $\mathbf{x}$ se monta un **BLSTM profundo de 2 capas** con 256 unidades ocultas por dirección. Cada capa BLSTM consta de un LSTM forward (procesa $\mathbf{x}_1 \to \mathbf{x}_{W'}$) y uno backward ($\mathbf{x}_{W'} \to \mathbf{x}_1$), cuyas salidas por timestep se concatenan. Stacking dos capas BLSTM permite abstracción jerárquica (lower-level encoda morfología local de strokes, higher-level encoda relaciones inter-carácter).

Los autores justifican explícitamente las tres ventajas del RNN sobre el CNN puro:

1. **Contexto:** "il" es ambiguo letra por letra (ambos son trazos verticales) pero la diferencia de altura discrimina; el RNN ve varios frames simultáneamente.
2. **Backprop conjunta:** los gradientes del CTC fluyen a través del BLSTM hasta la CNN, permitiendo que el extractor convolucional aprenda features útiles específicamente para sequence prediction —no para clasificación holística.
3. **Longitud arbitraria:** un BLSTM procesa secuencias de cualquier $W'$, en contraste con FC layers de tamaño fijo.

La salida de la última BLSTM, tras una proyección lineal, es una matriz $\mathbf{y} \in \mathbb{R}^{W' \times |\mathcal{L}'|}$ con softmax por filas: $\mathbf{y}_t$ es la distribución sobre el alfabeto extendido $\mathcal{L}' = \mathcal{L} \cup \{\text{blank}\}$ en el frame $t$.

Para inglés alfanumérico, $|\mathcal{L}| = 36$ (26 letras case-insensitive + 10 dígitos) y $|\mathcal{L}'| = 37$.

### 3.4 Transcription layer: CTC

La capa final transforma la secuencia $\mathbf{y}$ frame-level en un string variable-length aplicando **Connectionist Temporal Classification** (Graves, Fernández, Gomez, Schmidhuber — ICML 2006).

---

## 4. CTC en detalle matemático

CTC resuelve el problema fundamental: dada una salida de longitud $T = W'$ frames y un target de longitud $|\mathbf{l}| \leq T$ caracteres, ¿cómo definir una verosimilitud diferenciable cuando **no sabemos qué frame produjo qué carácter**?

### 4.1 Formulación

Sea $\mathcal{L}$ el alfabeto, $\mathcal{L}' = \mathcal{L} \cup \{-\}$ (con `blank` denotado `-`). Un **path** $\pi \in \mathcal{L}'^T$ es una asignación de un símbolo (incluido blank) a cada uno de los $T$ frames:

$$\pi = (\pi_1, \pi_2, \ldots, \pi_T), \quad \pi_t \in \mathcal{L}'.$$

Asumiendo **independencia condicional entre frames**, la probabilidad de un path es:

$$p(\pi | \mathbf{y}) = \prod_{t=1}^{T} y_{\pi_t}^t,$$

donde $y_k^t$ es la probabilidad de la clase $k$ en el frame $t$ según el softmax del BLSTM.

El **operador de colapso** $\mathcal{B}: \mathcal{L}'^T \to \mathcal{L}^{\leq T}$ transforma un path en un string aplicando dos pasos:

1. Colapsar caracteres repetidos consecutivos.
2. Eliminar todos los blanks.

**Ejemplos canónicos:**

- $\mathcal{B}(\texttt{-{}-hh-e-l-ll-oo-{}-}) = \texttt{hello}$
- $\mathcal{B}(\texttt{s-t-aa-t-e}) = \texttt{state}$
- $\mathcal{B}(\texttt{-s-t-a-t-e-}) = \texttt{state}$

El blank es indispensable: sin él, no se podría distinguir `aa` (doble a, dos caracteres) de `aa` (una sola a alineada en dos frames). El blank actúa como **separador explícito** entre caracteres iguales consecutivos: `a-a` colapsa a `aa`, mientras `aa` colapsa a `a`.

### 4.2 Probabilidad de un label sequence

La probabilidad de una palabra $\mathbf{l}$ dado el output del BLSTM se obtiene marginalizando sobre **todas las alineaciones que colapsan a $\mathbf{l}$**:

$$\boxed{p(\mathbf{l} | \mathbf{y}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} p(\pi | \mathbf{y}) = \sum_{\pi : \mathcal{B}(\pi) = \mathbf{l}} \prod_{t=1}^{T} y_{\pi_t}^t.}$$

El número de paths que colapsan a $\mathbf{l}$ es exponencial en $T$ (combinatorialmente: ¿en qué frames pongo cada carácter? ¿cuántas repeticiones le doy? ¿dónde inserto blanks?). Pero la suma se calcula **exactamente en tiempo $O(T \cdot |\mathbf{l}|)$** mediante un algoritmo forward-backward análogo al de HMM.

### 4.3 Forward-backward

Se construye un string extendido $\mathbf{l}'$ insertando un blank entre cada carácter de $\mathbf{l}$ y al inicio/fin: para $\mathbf{l} = \texttt{cat}$, se obtiene $\mathbf{l}' = \texttt{-c-a-t-}$ de longitud $2|\mathbf{l}|+1 = 7$.

Se define la **variable forward** $\alpha_t(s)$ = probabilidad de que en el frame $t$ se haya emitido el prefijo $\mathbf{l}'_{1:s}$:

$$\alpha_t(s) = \sum_{\pi: \mathcal{B}(\pi_{1:t}) = \mathbf{l}_{1:f(s)}, \, \pi_t = l'_s} \prod_{t'=1}^t y_{\pi_{t'}}^{t'},$$

con recursión

$$\alpha_t(s) = y_{l'_s}^t \cdot \left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) + [\text{si } l'_s \neq l'_{s-2} \text{ y } l'_s \neq \text{blank}] \cdot \alpha_{t-1}(s-2) \right).$$

Análogamente la **backward** $\beta_t(s)$. La verosimilitud total es:

$$p(\mathbf{l}|\mathbf{y}) = \alpha_T(|\mathbf{l}'|) + \alpha_T(|\mathbf{l}'| - 1),$$

y los gradientes respecto a $y_k^t$ son:

$$\frac{\partial p(\mathbf{l}|\mathbf{y})}{\partial y_k^t} = \frac{1}{y_k^t} \sum_{s : l'_s = k} \alpha_t(s) \beta_t(s).$$

El loss CTC es $\mathcal{L}_{\text{CTC}} = -\log p(\mathbf{l}^* | \mathbf{y})$. Es **diferenciable** y se propaga por backprop normal al BLSTM y de allí al CNN.

### 4.4 Conditional independence: el supuesto que duele

CTC asume $p(\pi | \mathbf{y}) = \prod_t y_{\pi_t}^t$, es decir, que cada frame es independiente dado $\mathbf{y}$. Esto es **doble simplificación**: el BLSTM internamente sí modela dependencias temporales en la representación oculta, pero la capa de salida emite cada frame independientemente sin condicionar en lo emitido previamente. Esa es la motivación de los decoders **attention-based** posteriores (ASTER, NRTR, ABINet, PARSeq): el decoder autoregresivo condiciona explícitamente $p(\mathbf{l}_i | \mathbf{l}_{<i}, \mathbf{y})$ y puede aprovechar n-gramas lingüísticos. CTC compensa en parte porque el BLSTM ya ve contexto, pero sigue siendo subóptimo cuando hay dependencias largas en el output (correcciones tipo language model).

---

## 5. Decoding

### 5.1 Lexicon-free (greedy o beam)

**Greedy:** se toma el argmax por frame y se aplica $\mathcal{B}$:

$$\mathbf{l}^* \approx \mathcal{B}(\arg\max_\pi p(\pi|\mathbf{y})) = \mathcal{B}((\arg\max_k y_k^1, \ldots, \arg\max_k y_k^T)).$$

Es un greedy aproximado a $\arg\max_\mathbf{l} p(\mathbf{l}|\mathbf{y})$ —no exacto porque maximizar el path no equivale a maximizar el string (varios paths pueden colapsar al mismo string). En la práctica funciona bien y es lo que CRNN reporta en la columna "None" de la Tabla 2.

**Beam search:** se mantiene un beam de strings parciales con probabilidades acumuladas; mejora marginalmente sobre greedy.

### 5.2 Lexicon-based

Cuando hay un léxico $\mathcal{D}$ (50 palabras, 1000, 50000), se busca

$$\mathbf{l}^* = \arg\max_{\mathbf{l} \in \mathcal{D}} p(\mathbf{l}|\mathbf{y}).$$

Para léxicos pequeños se evalúa exhaustivamente. Para léxicos grandes (50k palabras Hunspell) el costo es prohibitivo. **CRNN introduce un truco elegante con BK-tree:** se decodifica primero greedy para obtener $\mathbf{l}'$, luego se restringe la búsqueda a los vecinos de $\mathbf{l}'$ dentro de edit distance $\delta$:

$$\mathbf{l}^* = \arg\max_{\mathbf{l} \in \mathcal{N}_\delta(\mathbf{l}')} p(\mathbf{l}|\mathbf{y}).$$

Burkhard-Keller trees (BK-trees, 1973) son árboles métricos para espacios discretos con métrica como edit distance; permiten lookup de vecinos en $O(\log |\mathcal{D}|)$. La Figura 4 del paper muestra el trade-off $\delta$ vs accuracy: con $\delta = 0$ se obtiene 89.4% pero solo si la greedy ya estaba en el léxico; $\delta = 3$ alcanza 95.5% con 370ms/sample; $\delta = 5$ llega a 95.9% pero cuesta 2.4 s. Los autores fijan $\delta = 3$ como sweet spot.

---

## 6. Training

### 6.1 Dataset: Synth90k de Jaderberg

El detalle subestimado: **CRNN se entrena exclusivamente con datos sintéticos**. El dataset **Synth90k** (Jaderberg et al., NIPS DL Workshop 2014) son 8-9 millones de imágenes de palabras generadas por un motor sintético que combina ~90k palabras inglesas con tipografías, colores, bordes, transformaciones afines, ruido y backgrounds naturales. Cada imagen está perfectamente etiquetada con la palabra completa —no requiere bounding boxes ni char-level annotations.

Esto es transformador: significa que el **costo de anotación es cero** (todo el ground truth lo genera el sintetizador) y aún así el modelo generaliza a fotos reales sin fine-tuning. Es la victoria del **synth-to-real transfer** en STR.

### 6.2 Hiperparámetros

- **Optimizer:** ADADELTA (Zeiler 2012) con $\rho = 0.9$. ADADELTA ajusta learning rates por-dimensión automáticamente; los autores reportan convergencia más rápida que SGD+momentum, sin necesidad de tuning manual.
- **Tamaño de imagen:** todas las imágenes de entrenamiento se redimensionan a $100 \times 32$ (preservando altura, ancho fijo para batching).
- **Batch normalization** crucial tras Conv5 y Conv6.
- **Tiempo:** ~50 horas de entrenamiento en una NVIDIA Tesla K40 (un GPU de 2014).
- **Frameworks:** Torch7 con LSTM custom en CUDA, transcription CTC y BK-tree en C++.
- **Inferencia:** 0.16 s/sample sin léxico; 0.53 s/sample con léxico 50k.

### 6.3 Tamaño del modelo

**8.3M parámetros, 33MB en float32.** Comparado con Jaderberg 2015a (490M, ~2GB) y Jaderberg 2015b (304M), CRNN es ~30-60× más pequeño y por eso "puede portarse a dispositivos móviles" —un argumento de despliegue, no solo académico.

---

## 7. Resultados experimentales

### 7.1 Datasets de evaluación

- **IIIT5k** (Mishra et al. 2012): 3000 imágenes de palabras recogidas de internet, con léxico de 50 y de 1000 palabras por imagen.
- **SVT** (Wang et al. 2011): 647 imágenes recortadas de Google Street View, con léxico de 50 palabras.
- **IC03** (ICDAR 2003): 860 imágenes filtradas (>3 chars, alfanuméricas), léxicos 50, full y 50k.
- **IC13** (ICDAR 2013): 1015 imágenes recortadas, hereda gran parte de IC03.

Importante: CRNN **no se ajusta en ningún training set real**. Se entrena solo en Synth90k y se evalúa zero-shot en los benchmarks.

### 7.2 Comparativa (Tabla 2)

| Método | IIIT5k-50 | IIIT5k-1k | IIIT5k-None | SVT-50 | SVT-None | IC03-50 | IC03-Full | IC03-50k | IC03-None | IC13-None |
|--------|-----------|-----------|-------------|--------|----------|---------|-----------|----------|-----------|-----------|
| ABBYY | 24.3 | - | - | 35.0 | - | 56.0 | 55.0 | - | - | - |
| Mishra 2012 | 64.1 | 57.5 | - | 73.2 | - | 81.8 | 67.8 | - | - | - |
| PhotoOCR | - | - | - | 90.4 | 78.0 | - | - | - | - | 87.6 |
| Almazán 2014 | 91.2 | 82.1 | - | 89.2 | - | - | - | - | - | - |
| Jaderberg 2014 (deep feat) | - | - | - | 86.1 | - | 96.2 | 91.5 | - | - | - |
| Jaderberg 2015a (90k clf) | 97.1 | 92.7 | - | 95.4 | 80.7* | 98.7 | 98.6 | 93.3 | 93.1* | 90.8* |
| Jaderberg 2015b (structured) | 95.5 | 89.6 | - | 93.2 | 71.7 | 97.8 | 97.0 | 93.4 | 89.6 | 81.8 |
| **CRNN** | **97.6** | **94.4** | **78.2** | **96.4** | **80.8** | **98.7** | 97.6 | **95.5** | 89.4 | 86.7 |

(* indica que Jaderberg 2015a no es estrictamente lexicon-free; sus outputs están constreñidos a 90k palabras.)

Observaciones críticas:

- En **constrained lexicon**, CRNN gana en IIIT5k-50, IIIT5k-1k, SVT-50, IC03-50k; iguala en IC03-50; queda apenas por debajo solo en IC03-Full.
- En **lexicon-free** estricto (None), CRNN es el primer método con resultados publicados en IIIT5k (78.2%) y SVT (80.8%) genuinamente abiertos. PhotoOCR alcanza 78.0% en SVT-None pero requiere 7.9M de imágenes reales con char-level annotations.
- Jaderberg 2015a parece superar en algunos None (80.7 SVT, 93.1 IC03, 90.8 IC13) pero su output está restringido a 90k palabras: no es comparable en sentido estricto.
- **Tamaño:** CRNN 8.3M vs Jaderberg 2015a 490M vs Jaderberg 2015b 304M (Tabla 3).

### 7.3 Análisis de propiedades (Tabla 3)

| Método | E2E | Conv Ftrs | CharGT-Free | Unconstrained | Tamaño |
|--------|-----|-----------|-------------|---------------|--------|
| Wang 2012 | No | Sí | No | Sí | - |
| PhotoOCR | No | No | No | Sí | - |
| Jaderberg 2015a | Sí | Sí | Sí | **No** | 490M |
| Jaderberg 2015b | Sí | Sí | Sí | Sí | 304M |
| **CRNN** | **Sí** | **Sí** | **Sí** | **Sí** | **8.3M** |

CRNN es el único método que cumple las cinco propiedades simultáneamente con el modelo más pequeño.

### 7.4 Generalidad: Optical Music Recognition

Para demostrar que CRNN no es solo un STR engine sino un framework genérico de image-based sequence recognition, los autores lo aplican a reconocimiento de partituras (OMR): predicción de secuencia de pitches a partir de fragmentos de partitura.

- 2650 imágenes recogidas de MuseScore, aumentadas a 265k vía rotación/ruido/backgrounds.
- Configuración simplificada: 4ª y 6ª conv removidas; BLSTM reemplazado por LSTM unidireccional de 2 capas (menos datos → menos capacidad).
- Métricas: fragment accuracy + average edit distance.

| | Capella Scan | PhotoScore | CRNN |
|---|---|---|---|
| Clean | 51.9% / 1.75 | 55.0% / 2.34 | **74.6% / 0.37** |
| Synthesized | 20.0% / 2.31 | 28.0% / 1.85 | **81.5% / 0.30** |
| Real-World | 43.5% / 3.05 | 20.4% / 3.00 | **84.0% / 0.30** |

CRNN aplasta a los productos comerciales por márgenes de 20-60 puntos. La razón: Capella y PhotoScore dependen de binarización + detección de staff lines, ambos pasos frágiles ante iluminación variable y ruido; CRNN learna features convolucionales robustas y contextualiza con BLSTM.

---

## 8. Análisis, ablations e insights

### 8.1 BLSTM vs LSTM unidireccional

Los autores no incluyen tabla de ablation explícita, pero el experimento de OMR es revelador: cuando reducen capacidad eliminando bidireccionalidad, conservan la calidad solo porque también reducen complejidad del problema (partituras son menos densas que texto). En STR, el bidireccional aporta típicamente 3-5 puntos según ablations posteriores en la literatura (ASTER, Shi 2018).

### 8.2 Pooling rectangular

El choice de $1 \times 2$ en lugar de $2 \times 2$ en las últimas dos max-pools es una de las decisiones de diseño más subestimadas. Preserva resolución horizontal de modo que $W' = W/4$ en lugar de $W/16$. Para una imagen $100 \times 32$, eso significa 25 frames de salida en lugar de 6. Sin esa decisión, palabras largas como "congratulations" (15 chars) caerían fuera del rango representable. Esta es la razón arquitectural de por qué CRNN maneja longitudes arbitrarias.

### 8.3 Batch Normalization como enabler

Sin BN, entrenar conjuntamente CNN profunda + BLSTM no convergía limpiamente. Los autores lo reportan en passing pero es crítico: BN estabiliza la distribución de las activaciones que entran al BLSTM, mitigando el problema de gradient instability que es endémico en RNN sobre features deep.

### 8.4 ADADELTA vs SGD+momentum

Los autores reportan que ADADELTA converge más rápido y elimina el tuning manual de learning rate. Este detalle se replicaría en muchos sucesores aunque hoy es Adam el optimizer por defecto.

### 8.5 BK-tree para léxicos grandes

Una innovación práctica notable: la combinación greedy → BK-tree neighbor search permite escalar lexicon-based decoding a léxicos de 50k+ palabras sin sacrificar accuracy. Time-accuracy plot (Fig. 4) muestra el frontera de Pareto.

---

## 9. Limitaciones reconocibles

### 9.1 Texto irregular

CRNN asume texto **rectificado y aproximadamente horizontal**. Imágenes con texto curvo, perspectiva fuerte, o rotación severa colapsan la asunción de que columnas verticales del feature map corresponden a caracteres consecutivos. Esta es la motivación de:

- **ASTER (Shi 2018):** Spatial Transformer Network rectifier que primero deforma la imagen para enderezar el texto, luego corre un CRNN-like con attention decoder.
- **MORAN (Luo 2019):** Multi-Object Rectification Network pixel-wise.
- **SAR (Li 2019):** attention 2D, abandona la asunción de altura colapsada.

### 9.2 Conditional independence

CTC factoriza por frame sin condicional sobre output previo. Esto introduce errores tipo "burlington" → "burligton" donde un language model trivial lo corregiría. Decoders attention-based autoregresivos (NRTR, ABINet, PARSeq) condicionan explícitamente.

### 9.3 Altura fija

La asunción de input height = 32 es bottleneck. Texto vertical (japonés, chino vertical), texto a múltiples escalas en la misma imagen, o caracteres altos (mayúsculas latinas con descenders mezcladas con kanji) son casos donde una sola altura no basta. Spatial transformers y multi-scale features mitigan.

### 9.4 No interpretabilidad por carácter

Métodos char-level dan posición y confianza por letra. CRNN solo emite el string final; la asignación frame→carácter está disponible vía argmax pero es ruidosa (CTC paths no son interpretables como segmentación). Para downstream tasks que requieran word spotting con localización fina, hay que reconstruir a posteriori.

### 9.5 Sensibilidad a longitudes extremas

Palabras muy cortas (1-2 chars) producen secuencias degeneradas; palabras muy largas (>20 chars) exceden $W'$ típico. Mitigan con padding y ajuste de ancho.

---

## 10. Sucesores y descendencia

CRNN es el ancestro común de prácticamente toda la familia de reconocedores de scene text. La taxonomía (curiosa por compartirla con la clase 21):

- **ASTER** (Shi, Yang, Wang, Lyu, Bai — TPAMI 2018): mismo Shi, segunda iteración. Spatial Transformer Network + CRNN + attention decoder. Maneja texto irregular.
- **MORAN** (Luo, Jin, Sun — Pattern Recognition 2019): pixel-wise rectification + attention.
- **CA-FCN / FAN** (Cheng 2017): focusing attention para evitar attention drift.
- **SAR** (Li, Wang, Shen, Lyu — AAAI 2019): attention 2D que opera sobre feature map sin colapsar altura. Importante para irregular text.
- **NRTR** (Sheng, Chen, Xu — ICDAR 2019): reemplaza BLSTM por Transformer encoder + Transformer decoder. CTC desaparece, attention reigns.
- **MASTER** (Lu, Yang, Wang, Wei, Lin, Wang, Bai — Pattern Recognition 2021): Multi-Aspect attention.
- **ABINet** (Fang, Xie, Wang, Mao, Zhang — CVPR 2021): introduce un language model explícito en bidirectional fashion sobre el output del visual encoder. Logra que las dependencias lingüísticas se modelen ya no implícitamente sino con un decoder dedicado.
- **PARSeq** (Bautista, Atienza — ECCV 2022): permutation language modeling, ensemble de auto-regressive decoders en distintos órdenes. SOTA en 2022.

Y en la línea **scene text spotting end-to-end** (detección + reconocimiento juntos), CRNN-like recognizers se acoplan a detectores:

- **Mask TextSpotter** (Liao 2019): Mask R-CNN detector + char-level recognizer.
- **ABCNet** (Liu, Chen, Liu, Jin, Bai — CVPR 2020): Bezier curve detector + attention-based recognizer sobre features rectificadas. Está en la clase 21 como paper paralelo a este; comparten ADN porque ambos descienden de CRNN.

La etapa "Sequence Modeling" del diagrama Text Recognition Stages de la clase 21 es precisamente lo que CRNN canoniza: convolutional encoder → secuencia → predicción frame-level → transcripción.

---

## 11. Conexión con la clase 21 y el curso

### 11.1 Lugar en el dominio "image+text"

La clase 21 organiza el reconocimiento de texto en escena en cuatro etapas:

1. **Image preprocessing** — rectificación, normalización.
2. **Feature extraction** — CNN.
3. **Sequence modeling** — RNN/Transformer.
4. **Prediction** — CTC o attention.

CRNN es el primer modelo que implementa las cuatro etapas en un grafo único entrenable end-to-end. Es el "Hello World" de STR moderno.

### 11.2 Cruces con otros papers del curso

- **CTC (Graves 2006)** — el paper canónico que CRNN reutiliza. Disponible como `clase_21/papers/Graves-CTC-2006.md` (en construcción paralela). Comprender CTC es prerequisito para CRNN.
- **ABCNet (Liu 2020)** — successor en el mismo curso. ABCNet sustituye CTC por attention y maneja texto curvo con Bezier curves, pero el encoder convolucional + sequence module es heredero directo.
- **Chen TextRecognitionWild (2020)** — survey en la misma clase; CRNN aparece como el método foundational en su taxonomía.

### 11.3 Lugar en la línea de tiempo del dominio

- 2006: CTC (Graves) → enabler matemático.
- 2012-2014: deep STR fragmentado (Wang, Bissacco, Jaderberg).
- **2015: CRNN — unificación end-to-end.**
- 2017: TPAMI publication, consagración.
- 2018-2019: ASTER, SAR, NRTR — herederos que mejoran cada componente.
- 2020-2022: ABINet, PARSeq — atención + language modeling explícito.

### 11.4 Lecciones de diseño que perduran

Más allá del resultado puntual, CRNN deja lecciones arquitecturales que el campo absorbió:

1. **Colapsar una dimensión espacial para convertir feature maps en secuencias.** El patrón "altura 1, ancho variable" se replica en ASTER, MORAN y muchos más.
2. **CTC como bridge entre extractor visual y output simbólico.** Sigue siendo el método de elección cuando alignment ground truth no está disponible y se quiere evitar costo autoregresivo. Speech recognition lo usaba; CRNN lo cementa en visión.
3. **Synth-to-real transfer:** entrenar con sintético y desplegar sin fine-tuning. Patrón replicado masivamente en STR moderno.
4. **Modelos compactos por reemplazo de FC con weight-sharing:** la diferencia 8.3M vs 490M es estructural (no hay FC, todo es conv + LSTM compartido en el tiempo).
5. **End-to-end como argumento de diseño:** una vez que se ve CRNN, las pipelines fragmentadas se vuelven inaceptables. El campo entero pivotea.

---

## 12. Conclusión

CRNN (Shi, Bai, Yao — 2015/2017) es uno de esos papers raros que parecen, en retrospectiva, obvios: tomar una CNN tipo VGG, colapsar la altura, montar un BLSTM, aplicar CTC. Pero la combinación no se había hecho con éxito en visión, y los detalles importan: pooling rectangular para preservar resolución horizontal, BatchNorm para estabilizar el entrenamiento conjunto, BK-tree para escalar léxicos grandes, ADADELTA para evitar tuning manual. La arquitectura unifica feature extraction, sequence modeling y transcripción en un solo grafo, entrenable con solo pares (imagen, palabra), generaliza desde datos sintéticos a fotos reales, y se reduce a 8.3M parámetros desplegables en móviles.

Su impacto es estructural: define el template arquitectónico que la próxima década de scene text recognition usará, refinará y extenderá. Cualquier reconocedor STR moderno —y por extensión cualquier sistema de OCR neural, captioning de elementos visuales con secuencia, OMR, handwriting recognition— descansa, explícita o implícitamente, sobre las tres ideas que CRNN cristaliza: **convolutional encoder + sequence model + alignment-free loss**. En el mapa conceptual de la clase 21, CRNN es el nodo central de "Sequence Modeling Stage": sin él, ABCNet, ASTER, ABINet y PARSeq no existirían en su forma actual.

Para Roberto: si la pregunta es "¿con qué empezar para entender STR moderno?", la respuesta es leer CTC (Graves 2006) y CRNN (Shi 2015). Todo lo demás es extensión.

---

## Referencias clave dentro del paper

- Graves et al. 2006 — CTC paper canónico, fundamento matemático.
- Hochreiter & Schmidhuber 1997 — LSTM.
- Simonyan & Zisserman 2014 — VGG backbone.
- Ioffe & Szegedy 2015 — Batch Normalization.
- Zeiler 2012 — ADADELTA.
- Jaderberg et al. 2014 — Synth90k dataset.
- Burkhard & Keller 1973 — BK-trees.

## Conexiones con material del curso IA UC

- `clase_21/papers/Graves-CTC-2006.md` — fundamento matemático del CTC loss.
- `clase_21/papers/Liu-ABCNet-2020.md` — successor con attention decoder en lugar de CTC.
- `clase_21/papers/Chen-TextRecognitionWild-2020.md` — survey que coloca CRNN en perspectiva histórica.
- `fundamentos/ctc-loss.md` — derivación didáctica del CTC.
- `fundamentos/bidirectional-rnn.md` — BLSTM en detalle.
- `clase_14/` — Transformers como sucesor del BLSTM en NRTR/PARSeq.
