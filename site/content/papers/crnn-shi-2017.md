---
title: "CRNN (Convolutional Recurrent Neural Network)"
weight: 106
math: true
---

{{< paper-card
    title="An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"
    authors="Shi, Bai, Yao"
    year="2017"
    venue="IEEE TPAMI 2017 (arXiv 2015)"
    pdf="/papers/crnn-shi-2017.pdf"
    arxiv="1507.05717" >}}
Unifica el reconocimiento de texto en escena en un grafo end-to-end: CNN tipo VGG colapsa la altura del feature map y produce una secuencia de descriptores, un BLSTM apilado modela el contexto horizontal, y CTC integra todas las alineaciones frame-caracter en un loss diferenciable. Funciona con vocabulario abierto, sin char-level annotations, en 8.3M parametros, entrenado solo con datos sinteticos. Es el baseline universal de scene text recognition (STR) moderno.
{{< /paper-card >}}

---

## El problema

Reconocer texto en escena (no documentos) fue durante decadas intratable. Hasta 2014 el paradigma era una pipeline de cinco etapas: binarizacion, segmentacion de caracteres, extraccion hand-crafted, clasificacion por caracter y post-procesamiento con HMM o CRF. La segmentacion era el cuello de botella endemico (paradoja de Sayre: para segmentar un caracter hay que reconocerlo, pero para reconocerlo hay que segmentarlo).

Cuando llego deep learning, surgieron tres lineas competidoras pero incompletas:

1. **Caracter-centrica** (Wang 2012, Alsharif 2014): CNN por caracter aislado + sliding window. Requiere **anotaciones a nivel de caracter** (bounding box por letra), costosas e inconsistentes.
2. **Palabra-centrica** (Jaderberg 2014/2015): una CNN gigante de 490M parametros clasifica la imagen como una de 90.000 palabras inglesas. Spectacular en accuracy pero **vocabulario cerrado**: no puede leer numeros de telefono, matriculas, nombres propios, partituras o caracteres chinos.
3. **Pipelines hibridos** (PhotoOCR, Bissacco 2013): clasificador deep + CRF + n-gramas. Requiere 7.9M imagenes reales con char-level annotations, dataset privado de Google.

En 2015, **ningun metodo cumplia simultaneamente** ser end-to-end, vocabulario abierto, sin anotacion char-level, de longitud variable y compacto. CRNN cierra exactamente ese gap.

---

## Arquitectura CRNN

CRNN apila tres bloques diferenciables que se entrenan con un unico loss CTC.

### CNN feature extractor

Variante de VGG-VeryDeep con dos tweaks claves para sequence prediction. Input grayscale $W \times 32$:

| Capa | Configuracion |
|------|---------------|
| Conv1 / MaxPool1 | 64 mapas $3{\times}3$ / $2{\times}2$ |
| Conv2 / MaxPool2 | 128 mapas $3{\times}3$ / $2{\times}2$ |
| Conv3, Conv4 / MaxPool3 | 256 mapas / **$1{\times}2$ rectangular** |
| Conv5, Conv6 + BN / MaxPool4 | 512 mapas / **$1{\times}2$ rectangular** |
| Conv7 | 512 mapas $2{\times}2$, padding 0 |

Los **dos tweaks decisivos**:

1. **Pooling rectangular $1 \times 2$** en MaxPool3/4 (en lugar de $2{\times}2$): solo sub-muestrea el alto. Preserva resolucion horizontal, de modo que $W' = W/4$ en lugar de $W/16$. Para una imagen $100 \times 32$, se obtienen 25 frames de salida en lugar de 6, suficientes para palabras largas como "congratulations".
2. **Batch Normalization tras Conv5/6**: sin BN, entrenar conjuntamente CNN profunda + BLSTM no convergia limpiamente.

Tras Conv7 el feature map tiene altura **exactamente 1**: dimensiones $(C, 1, W')$ con $C = 512$. Esa altura colapsada es el truco arquitectural fundamental del paper.

### Map-to-sequence

Operacion trivial pero conceptualmente decisiva. El tensor $(512, 1, W')$ se reinterpreta como una secuencia de $W'$ vectores de 512 canales:

$$\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_{W'}), \quad \mathbf{x}_t \in \mathbb{R}^{512}.$$

Cada $\mathbf{x}_t$ es el descriptor de una **franja vertical** de la imagen original (su campo receptivo). Las franjas se **solapan**: multiples timesteps pueden ver porciones del mismo caracter. Eso es por diseno: CTC necesita poder emitir un caracter en varios frames contiguos y luego colapsarlos.

### BLSTM profundo

Sobre $\mathbf{x}$ se monta un **BLSTM de 2 capas apiladas** con 256 unidades ocultas por direccion. Tres ventajas respecto a un CNN puro o un LSTM unidireccional:

1. **Contexto bidireccional**: "il" es ambiguo letra por letra (ambos son trazos verticales) pero el contexto lo resuelve.
2. **Backprop conjunto**: los gradientes del CTC fluyen a traves del BLSTM hasta la CNN, permitiendo que el extractor convolucional aprenda features especificamente utiles para sequence prediction.
3. **Longitud arbitraria**: a diferencia de capas FC, un BLSTM procesa secuencias de cualquier $W'$.

La salida es una matriz $\mathbf{y} \in \mathbb{R}^{W' \times |\mathcal{L}'|}$ con softmax por filas: $\mathbf{y}_t$ es la distribucion sobre el alfabeto extendido $\mathcal{L}' = \mathcal{L} \cup \{\text{blank}\}$ en el frame $t$. Para ingles alfanumerico, $|\mathcal{L}| = 36$ y $|\mathcal{L}'| = 37$.

### Transcription layer (CTC)

La capa final transforma $\mathbf{y}$ frame-level en un string variable-length aplicando **Connectionist Temporal Classification** (Graves 2006).

---

## CTC en compacto

CTC resuelve el problema central: dado $T = W'$ frames y un target de longitud $|\mathbf{l}| \leq T$, definir una verosimilitud diferenciable **sin saber que frame produjo que caracter**. La derivacion completa esta en [/fundamentos/ctc-loss](/fundamentos/ctc-loss) y el paper canonico [/papers/ctc-graves-2006](/papers/ctc-graves-2006); aqui solo lo esencial.

**Operador de colapso** $\mathcal{B}: \mathcal{L}'^T \to \mathcal{L}^{\leq T}$: (1) colapsar caracteres repetidos consecutivos; (2) eliminar blanks. Ejemplo:

$$\mathcal{B}(\texttt{-{}-hh-e-l-ll-oo-{}-}) = \texttt{hello}.$$

El blank es indispensable como **separador explicito** entre caracteres iguales consecutivos: `a-a` colapsa a `aa`, mientras `aa` colapsa a `a`.

**Probabilidad de un label** $\mathbf{l}$ se marginaliza sobre todos los paths que colapsan a $\mathbf{l}$, bajo el supuesto de **independencia condicional entre frames**:

$$p(\mathbf{l} | \mathbf{y}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T} y_{\pi_t}^t.$$

El numero de paths es exponencial, pero la suma se calcula exactamente en $O(T \cdot |\mathbf{l}|)$ con **forward-backward** estilo HMM. El loss CTC $\mathcal{L}_{\text{CTC}} = -\log p(\mathbf{l}^* | \mathbf{y})$ es diferenciable y propaga gradientes al BLSTM y de alli al CNN.

El supuesto de independencia condicional es la principal limitacion teorica: cada frame es independiente dado $\mathbf{y}$, sin condicionar en lo emitido previamente. El BLSTM compensa parcialmente (su estado oculto si tiene memoria), pero la capa de salida no autoregresa. Esa es la motivacion de los decoders **attention-based** posteriores (ASTER, NRTR, ABINet, PARSeq).

---

## Decoding

**Greedy (lexicon-free):** argmax por frame seguido de $\mathcal{B}$:

$$\mathbf{l}^* \approx \mathcal{B}((\arg\max_k y_k^1, \ldots, \arg\max_k y_k^T)).$$

Es aproximado (varios paths pueden colapsar al mismo string) pero funciona bien en la practica. Es la columna "None" de los benchmarks.

**Beam search:** mantiene un beam de strings parciales con probabilidades acumuladas; mejora marginalmente sobre greedy.

**Lexicon-constrained:** dado un lexico $\mathcal{D}$ (50, 1000 o 50.000 palabras),

$$\mathbf{l}^* = \arg\max_{\mathbf{l} \in \mathcal{D}} p(\mathbf{l}|\mathbf{y}).$$

Para lexicos pequenos se evalua exhaustivamente. Para lexicos grandes (Hunspell 50k), CRNN introduce un truco con **BK-tree** (Burkhard-Keller 1973): se decodifica primero greedy a $\mathbf{l}'$, luego se restringe la busqueda a vecinos de $\mathbf{l}'$ dentro de edit distance $\delta$. Lookup en $O(\log |\mathcal{D}|)$. Los autores fijan $\delta = 3$ como sweet spot (95.5% accuracy, 370 ms/sample).

---

## Training

- **Dataset:** Synth90k (Jaderberg 2014), 8-9M imagenes de palabras sinteticas con ~90k palabras inglesas. **Costo de anotacion cero** (todo lo genera el sintetizador) y aun asi generaliza a fotos reales **sin fine-tuning**. Victoria del synth-to-real transfer.
- **Optimizer:** ADADELTA ($\rho = 0.9$). Convergencia mas rapida que SGD+momentum sin tuning de learning rate.
- **Imagen:** redimensionada a $100 \times 32$. Curriculum natural short-to-long emerge del muestreo de Synth90k.
- **Tiempo:** ~50 horas en una Tesla K40 (GPU de 2014).
- **Frameworks:** Torch7 con LSTM custom en CUDA y CTC en C++.
- **Modelo:** 8.3M parametros, 33 MB en float32. Comparado con Jaderberg 2015a (490M, ~2GB) y Jaderberg 2015b (304M), CRNN es ~30-60x mas pequeno y desplegable en moviles.

---

## Resultados

Evaluado **zero-shot** (entrenado solo en Synth90k) en cuatro benchmarks: IIIT5k (3000 imagenes), SVT (647), IC03 (860), IC13 (1015). Cada uno con lexicos opcionales de 50, Full o 50k palabras y un setting "None" lexicon-free.

| Metodo | IIIT5k-50 | IIIT5k-1k | IIIT5k-None | SVT-50 | SVT-None | IC03-50 | IC03-50k | IC13-None |
|--------|-----------|-----------|-------------|--------|----------|---------|----------|-----------|
| ABBYY | 24.3 | - | - | 35.0 | - | 56.0 | - | - |
| Mishra 2012 | 64.1 | 57.5 | - | 73.2 | - | 81.8 | - | - |
| PhotoOCR (Google) | - | - | - | 90.4 | 78.0 | - | - | 87.6 |
| Jaderberg 2015a (90k) | 97.1 | 92.7 | - | 95.4 | 80.7* | 98.7 | 93.3 | 90.8* |
| Jaderberg 2015b | 95.5 | 89.6 | - | 93.2 | 71.7 | 97.8 | 93.4 | 81.8 |
| **CRNN** | **97.6** | **94.4** | **78.2** | **96.4** | **80.8** | **98.7** | **95.5** | 86.7 |

(* Jaderberg 2015a constrene el output a 90k palabras: no es lexicon-free estricto.)

Observaciones:

- Gana en **constrained lexicon** (IIIT5k-50/1k, SVT-50, IC03-50k); iguala en IC03-50.
- Es el primer metodo con resultados publicados en setting **lexicon-free estricto** (IIIT5k-None 78.2%, SVT-None 80.8%) sin requerir char-level annotations ni vocabulario cerrado.
- **Tamano:** 8.3M vs 490M de Jaderberg 2015a y 304M de Jaderberg 2015b. Es ~60x mas pequeno que el mejor competidor.

**Generalidad (OMR):** los autores aplican CRNN a reconocimiento optico de partituras con configuracion simplificada (4a y 6a conv removidas, BLSTM reemplazado por LSTM unidireccional). Sobre 2650 imagenes recogidas de MuseScore aumentadas a 265k via rotacion/ruido/backgrounds, CRNN aplasta a Capella Scan y PhotoScore por 20-60 puntos en accuracy de fragmento (74-84% vs 20-55% de los competidores). La razon es estructural: Capella y PhotoScore dependen de binarizacion + deteccion de staff lines, ambos pasos fragiles ante iluminacion variable; CRNN aprende features convolucionales robustas y contextualiza con BLSTM, demostrando que no es solo un STR engine sino un framework generico de image-based sequence recognition.

**Ablations implicitas:** el paper no incluye tabla formal de ablation, pero varias decisiones se justifican empiricamente:

- **Pooling rectangular $1{\times}2$** vs $2{\times}2$: sin el, palabras largas caen fuera del rango representable. Es la razon arquitectural por la que CRNN maneja longitudes arbitrarias.
- **BatchNorm tras Conv5/6**: enabler de la convergencia del entrenamiento conjunto CNN+BLSTM.
- **BLSTM 2 capas** vs LSTM unidireccional: bidireccional aporta tipicamente 3-5 puntos segun ablations posteriores en la literatura.
- **ADADELTA** vs SGD+momentum: converge mas rapido y elimina tuning manual de learning rate.

---

## Limitaciones

1. **Texto rectificado horizontal**: CRNN asume que columnas verticales del feature map corresponden a caracteres consecutivos. Texto curvo, perspectiva fuerte o rotacion severa rompen la asuncion. Motivacion de [ASTER](/papers/stn-jaderberg-2015) (STN + CRNN + attention), MORAN (rectificacion pixel-wise) y SAR (attention 2D).
2. **Conditional independence del CTC**: cada frame se emite sin condicionar en lo previo. Errores tipo "burlington" -> "burligton" que un language model trivial corregiria. Decoders autoregresivos (NRTR, ABINet, PARSeq) los corrigen.
3. **Altura fija (32 px)**: bottleneck para texto vertical (japones, chino), multi-escala en la misma imagen, o mezcla de tamanos extremos.
4. **No interpretabilidad por caracter**: CRNN emite solo el string. La asignacion frame->caracter via argmax es ruidosa y no se interpreta como segmentacion. Para word spotting con localizacion fina hay que reconstruir a posteriori.
5. **Longitudes extremas**: palabras de 1-2 caracteres producen secuencias degeneradas; palabras de >20 chars exceden $W'$ tipico.

---

## Sucesores

CRNN es el ancestro comun de practicamente toda la familia de reconocedores de scene text. La descendencia inmediata:

- **ASTER** (Shi 2018) — mismo Shi, segunda iteracion. **Spatial Transformer Network** rectifier que primero endereza la imagen, luego CRNN-like con attention decoder. Maneja texto irregular.
- **MORAN** (Luo 2019) — rectificacion pixel-wise + attention.
- **SAR** (Li 2019) — attention 2D que opera sobre feature map sin colapsar altura.
- **NRTR** (Sheng 2019) — reemplaza BLSTM por Transformer encoder + Transformer decoder. CTC desaparece, attention reigns.
- **ABINet** (Fang 2021) — introduce un language model explicito bidireccional sobre el output del visual encoder; las dependencias linguisticas se modelan con un decoder dedicado.
- **PARSeq** (Bautista 2022) — permutation language modeling, ensemble de decoders autoregresivos en distintos ordenes. SOTA en 2022.

Y en **end-to-end scene text spotting** (deteccion + reconocimiento juntos):

{{< paper-card
    title="ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network"
    authors="Liu et al."
    year="2020"
    venue="CVPR 2020" >}}
Detector Bezier curve + recognizer attention-based sobre features rectificadas. Comparte ADN con CRNN (encoder convolucional + sequence module) pero sustituye CTC por attention y maneja texto curvo nativo. Cubierto en detalle en [/papers/abcnet-liu-2020](/papers/abcnet-liu-2020).
{{< /paper-card >}}

---

## Conexion con la clase 21

La [clase 21](/clases/clase-21) organiza el reconocimiento de texto en escena en cuatro etapas: **image preprocessing → feature extraction → sequence modeling → prediction (CTC o attention)**. El diagrama "Text Recognition Stages" del slide del curso describe exactamente CRNN: es el primer modelo que implementa las cuatro etapas en un grafo unico entrenable end-to-end. Es el "Hello World" del STR moderno.

ABCNet (Liu 2020) usa la misma idea pero con attention recognizer en lugar de CTC, y agrega deteccion con curvas Bezier. Comprender CRNN es comprender la espina dorsal del reconocimiento de secuencias visuales de la ultima decada.

**Lecciones de diseno que perduran:**

1. **Colapsar una dimension espacial** para convertir feature maps en secuencias. Patron replicado en ASTER, MORAN y muchos mas.
2. **CTC como bridge** entre extractor visual y output simbolico cuando alignment ground truth no esta disponible.
3. **Synth-to-real transfer**: entrenar con sintetico y desplegar sin fine-tuning. Patron replicado masivamente.
4. **Modelos compactos via weight-sharing temporal**: la diferencia 8.3M vs 490M parametros es estructural (no hay FC, todo es conv + LSTM compartido).
5. **End-to-end como argumento de diseno**: una vez que se ve CRNN, las pipelines fragmentadas se vuelven inaceptables.

**Linea de tiempo del dominio:**

- 2006: [CTC (Graves)](/papers/ctc-graves-2006) — enabler matematico.
- 2012-2014: deep STR fragmentado (Wang, Bissacco, Jaderberg).
- **2015: CRNN — unificacion end-to-end.**
- 2017: publicacion en TPAMI, consagracion.
- 2018-2019: ASTER, SAR, NRTR — herederos que mejoran cada componente.
- 2020-2022: [ABCNet](/papers/abcnet-liu-2020), ABINet, PARSeq — attention + language modeling explicito.

---

## Notas y enlaces

**Fundamentos:**

- [CTC loss](/fundamentos/ctc-loss) — derivacion didactica del forward-backward.
- [Scene text recognition](/fundamentos/scene-text-recognition) — taxonomia completa del dominio.
- [LSTM y GRU](/fundamentos/lstm-gru) — base del sequence module.
- [Redes convolucionales](/fundamentos/redes-convolucionales) — backbone del feature extractor.

**Papers relacionados:**

- [CTC (Graves 2006)](/papers/ctc-graves-2006) — fundamento matematico del loss; prerequisito para CRNN.
- [LSTM (Hochreiter 1997)](/papers/lstm-hochreiter-1997) — celda recurrente base del BLSTM.
- [STN (Jaderberg 2015)](/papers/stn-jaderberg-2015) — Spatial Transformer; rectifier que ASTER usa sobre CRNN.
- [ABCNet (Liu 2020)](/papers/abcnet-liu-2020) — successor con attention decoder y deteccion con Bezier curves.
- [Text Recognition in the Wild (Chen 2020)](/papers/text-recognition-wild-chen-2020) — survey que coloca CRNN en perspectiva historica.

**Clase:** [Clase 21 — Image + Text](/clases/clase-21).

**Recursos externos:** la implementacion oficial en Torch7 esta disponible en el repositorio de Bgshih en GitHub; multiples re-implementaciones en PyTorch (crnn.pytorch de Meijieru) y TensorFlow circulan ampliamente. Es uno de los baselines mas implementados de la decada.
