---
title: "Text Recognition in the Wild: A Survey"
weight: 101
math: true
---

{{< paper-card
    title="Text Recognition in the Wild: A Survey"
    authors="Chen, Jin, Yi, Lyu"
    year="2020"
    venue="arXiv 2005.03492 (later ACM Computing Surveys)"
    pdf="/papers/text-recognition-wild-chen-2020.pdf"
    arxiv="2005.03492" >}}
Survey de referencia que estructuró el campo de *Scene Text Recognition* (STR) entre 2020 y 2022. Articula una **taxonomía canónica en cuatro etapas** (preprocessing → feature extraction → sequence modeling → prediction), inventaría datasets, métricas y métodos representativos, y organiza más de 200 papers en una tabla comparativa que permitió comparar de un vistazo CRNN, ASTER, MORAN, NRTR y todo el estado del arte pre-Transformer multimodal.
{{< /paper-card >}}

---

## Por qué un survey importa

Un survey no aporta resultados experimentales nuevos: aporta **estructura**. Cuando un campo crece a la velocidad del *deep learning* aplicado a visión, la cantidad de papers publicados en dos o tres años desborda a cualquier *practitioner* recién llegado. Chen, Jin y colaboradores ocupan ese hueco para *Scene Text Recognition* (STR), el subcampo de visión por computadora que se ocupa de leer texto en imágenes naturales capturadas con teléfonos o cámaras *in the wild*, en condiciones de iluminación, oclusión, distorsión, fuente, color y orientación arbitrarias.

Antes del survey, el campo era una nube de papers sueltos. Después del survey, todo se discute con su vocabulario: las cuatro estaciones de la pipeline, la separación entre datasets *regular* e *irregular*, las métricas WRA y NED, el contraste CTC vs. attention. La versión final (v3, diciembre 2020) cierra con la literatura pre-Transformer multimodal: **no incluye** TrOCR (2021), PARSeq (2022) ni los foundation models multimodales actuales, lo que define a la vez su valor y su límite.

Para la clase 21 del Diplomado IA UC, este survey es el mapa que permite ubicar dentro del campo cualquier paper posterior: ABCNet, Mask TextSpotter, TrOCR, Donut o LayoutLMv3 se entienden mejor cuando se sabe qué estación de la pipeline reorganizan o suprimen.

La distinción con **OCR clásico** (Tesseract sobre documentos escaneados) es importante: STR convive con fondos complejos, fuentes ornamentales, distorsión perspectiva, baja resolución y *motion blur*, condiciones donde la pipeline binarización + segmentación de caracteres + clasificador fracasa. Por eso STR vive en otro régimen: necesita modelos simultáneamente *vision* (robustos a distorsión y ruido) y *sequence* (que modelen la salida como cadena de caracteres con dependencias contextuales).

El punto de quiebre histórico está en 2014, cuando Jaderberg et al. (Oxford VGG) publican el dataset sintético **Synth90k**: nueve millones de imágenes renderizando 90.000 palabras sobre fondos naturales. A partir de ahí, los métodos *deep learning* superan rápidamente a los basados en *handcrafted features* (HOG, MSER, SWT).

---

## Taxonomía pipeline canónica

La contribución principal del survey es el diagrama de la Figura 4: la pipeline *segmentation-free* moderna se descompone en cuatro estaciones secuenciales.

### Image preprocessing

Tres familias de técnicas operan antes del extractor de features:

- **Background removal**. Eliminar el fondo y dejar solo los píxeles del texto. La binarización clásica (Otsu) falla en escena; Luo et al. (2020) usan GANs para aprender a separar foreground y background.
- **Text Image Super-Resolution (TextSR)**. Cuando el texto ocupa pocos píxeles, reconstruir una versión de alta resolución antes de reconocer. Lo interesante es entrenar el super-resolver **conjuntamente** con el reconocedor: la pérdida del reconocedor backpropaga al super-resolver para que reconstruya los detalles que importan a la lectura, no a la PSNR genérica.
- **Rectification networks**. La pieza más influyente. Una *Spatial Transformer Network* (STN, Jaderberg et al. 2015) se inserta como módulo diferenciable que aprende a desdistorsionar el texto antes de entregarlo al CNN. Para texto curvado o con perspectiva fuerte, **ASTER** (Shi et al. 2018) adopta *Thin-Plate Spline* (TPS): transformación no rígida controlada por puntos fiduciarios aprendidos. Variantes posteriores: MORAN (multi-objeto), ESIR (TPS iterativa), Yang et al. 2019 (restringida por simetría).

El survey advierte que módulos de rectificación complejos consumen memoria y tiempo, y que conforme mejora la detección de texto irregular *upstream*, conviene reconsiderar si el rectificador es estrictamente necesario.

### Feature extraction

El backbone CNN que mapea la imagen de entrada (típicamente $32 \times 100$ píxeles para una palabra inglesa) a un mapa de features apto para reconocimiento:

- **VGGNet** (Simonyan & Zisserman 2014): backbone original de CRNN, simple pero costoso en parámetros.
- **ResNet** (He et al. 2016): la mayoría de métodos modernos (STAR-Net, ASTER, MORAN, AON, NRTR, SAR) lo usan; las *skip connections* permiten redes profundas sin colapsar el gradiente.
- **DenseNet** (Huang et al. 2017): conexiones densas dentro de bloque, más eficiente en parámetros.
- **Recursive CNN / GRCNN**: la misma capa convolucional aplicada $k$ veces, aumentando profundidad efectiva sin más parámetros.
- **CNN + attention visual**: resalta foreground del texto y suprime ruido del background; es el germen de los métodos *2D-attention* posteriores (SAR).

### Sequence modeling

Después del CNN, el mapa de features de tamaño $T \times C$ se interpreta como una secuencia. El default desde 2015 hasta 2019 es **BiLSTM**: dos LSTM, uno hacia adelante y otro hacia atrás, concatenando *hidden states*. La justificación es que para leer "BANANA" hay que mirar tanto el contexto izquierdo como el derecho de cada carácter.

Sin embargo, varios autores cuestionan que BiLSTM sea necesario:

- **1D-CNN profunda**: con receptive field suficientemente grande, el contexto local que necesita STR ya se captura, sin el costo de la recurrencia.
- **Transformer**: **NRTR** (Sheng et al. 2019) fue uno de los primeros en mostrar que un Transformer puro supera a BiLSTM+attention en STR, anticipando la dirección dominante post-2020 (TrOCR, PARSeq, MaskOCR).

### Prediction

La última estación convierte la secuencia de features en la secuencia de caracteres. Dos técnicas dominan el campo.

**Connectionist Temporal Classification (CTC)**, propuesta por Graves et al. (2006) para *unsegmented sequence labelling*. La idea brillante: en lugar de obligar al modelo a alinear cada *timestep* con un carácter (lo que requeriría anotación de fronteras), CTC introduce un símbolo *blank* $\varepsilon$ y suma sobre todos los caminos posibles que mapean a la misma transcripción:

$$
p(l \mid y) = \sum_{\pi : \mathcal{B}(\pi) = l} \prod_{t=1}^{T} y^t_{\pi_t}
$$

donde $\mathcal{B}$ colapsa repeticiones y borra el *blank*. La suma se calcula en $O(T \cdot |l|)$ con el algoritmo *forward-backward*. CTC aparece en CRNN, STAR-Net, GRCNN, FAN, ABCNet, entre muchos otros. Sus limitaciones: sufre el *peaky distribution problem* (predicciones sobreconcentradas), le cuesta repeticiones (requiere *blank* entre caracteres iguales) y no escala a 2D (Wan et al. 2020 proponen 2D-CTC, pero el problema no está resuelto).

**Attention-based decoder**, basado en Bahdanau et al. (2015). Un decoder GRU emite carácter a carácter consultando un *glimpse vector* sobre los features del encoder:

$$
\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{i=1}^{N} \exp(e_{t,i})}, \quad g_t = \sum_j \alpha_{t,j} h_j
$$

Variantes: **2D attention** para texto irregular (SAR, AON); **bidireccional** (ASTER); **Transformer attention** (NRTR); soluciones al **attention drift** (FAN — Cheng et al. 2017 — agrega supervisión sobre la localización; DAN — Wang et al. 2020 — desacopla atención del estado del decoder).

El veredicto del survey: la atención supera a CTC en accuracy en *isolated word recognition*, pero CTC es más rápido y robusto en oraciones largas. Métodos híbridos (GTC, SCATTER) combinan ambos durante el entrenamiento.

---

## Datasets canónicos

| Categoría | Dataset | Tamaño | Característica |
|---|---|---|---|
| **Sintético** | Synth90k (Jaderberg 2014) | ~9 M palabras | Render de 90k palabras con transformaciones |
| Sintético | SynthText (Gupta 2016) | ~6 M en 800k imgs | Texto sobre escenas naturales con localización |
| Sintético | UnrealText (Long & Yao 2020) | ~12 M en 600k imgs | Renderizado en Unreal Engine 4 con física 3D |
| **Regular (latín)** | IIIT5K-Words | 2.000 train / 3.000 test | Imágenes web e in-situ |
| Regular | SVT (Street View Text) | 100 / 250 | Google Street View |
| Regular | IC03 / IC11 / IC13 | ~250-485 | Competencias ICDAR |
| Regular | SVHN | >600.000 dígitos | Números de casa |
| **Irregular (latín)** | SVT-P | 639 cropped | Perspectiva no-frontal |
| Irregular | CUTE80 | 288 cropped | Texto curvado en logos y carteles |
| Irregular | IC15 (Incidental) | 1.000 / 500 | Google Glasses, multi-oriented |
| Irregular | COCO-Text | 145.859 cropped | Escenas naturales con atributos |
| Irregular | Total-Text | 11.459 cropped | Foco en texto curvado |
| **Multilingüe** | RCTW-17 / CTW | 12.514 / 32.285 imgs | Chino in the wild |
| Multilingüe | LSVT | 450.000 | Chino, mix supervisión fuerte/débil |
| Multilingüe | MLT-2019 | 20.000 | 10 idiomas: árabe, bangla, chino, devanagari, inglés, francés, alemán, italiano, japonés, coreano |

Sin Synth90k y SynthText, la mayoría de los métodos modernos no podrían entrenarse: los datasets realistas contienen miles, no millones, de instancias. Por eso el *pretrain sintético + finetune realista* es la receta estándar del campo.

---

## Métricas

### Word Recognition Accuracy (WRA)

Métrica dominante en latín:

$$
\mathrm{WRA} = \frac{W_r}{W}, \quad \mathrm{WER} = 1 - \mathrm{WRA}
$$

donde $W$ es el total de palabras y $W_r$ las correctas. Comportamiento *all-or-nothing*: una palabra con un solo carácter mal cuenta como error.

### Normalized Edit Distance (NED)

Adoptada por ICDAR para chino y multilingüe, captura diferencias parciales:

$$
\mathrm{NED} = \frac{1}{N} \sum_{i=1}^{N} \frac{D(s_i, \hat{s}_i)}{\max(l_i, \hat{l}_i)}
$$

donde $D(\cdot)$ es la distancia de Levenshtein entre predicción $s_i$ y ground truth $\hat{s}_i$. La métrica reportada típicamente es $1 - \mathrm{NED}$. Si el modelo lee 9 de 10 caracteres correctamente, NED refleja el 90% acertado, mientras WRA marca error total.

### End-to-end F-measure

Para sistemas que hacen detección + reconocimiento conjuntos: una predicción es correcta si la *bounding box* tiene $\mathrm{IoU} > 0.5$ con el ground truth **y** el string predicho coincide (o tiene $1 - \mathrm{NED}$ alto). F-score es la media armónica de *precision* y *recall*. Tres niveles según vocabulario disponible: **strongly contextualised** (vocabulario de 100 palabras por imagen), **weakly contextualised** (vocabulario de train/test), **generic** (~90k palabras).

---

## Métodos representativos

| Estación | Métodos | Idea clave |
|---|---|---|
| **Detección** | TextBoxes / TextBoxes++ (Liao 2017-18) | SSD adaptado a cajas largas y oblicuas |
| Detección | EAST (Zhou 2017) | Regresión directa de quad/rotated-rectangle por píxel |
| Detección | CRAFT (Baek 2019) | Score por carácter + afinidad entre caracteres |
| Detección | PSENet (Wang 2019) | Kernels concéntricos para separar textos cercanos |
| Detección | DBNet (Liao 2020) | Binarización aprendible diferenciable, *real-time* |
| **Reconocimiento** | CRNN (Shi 2015) | VGG + BiLSTM + CTC — la baseline canónica |
| Reconocimiento | RARE (Shi 2016) | Primer reconocedor con STN integrada |
| Reconocimiento | STAR-Net (Liu 2016) | ResNet + atención espacial + CTC |
| Reconocimiento | ASTER (Shi 2018) | TPS + ResNet + BiLSTM + attention bidireccional |
| Reconocimiento | MORAN (Luo 2019) | Rectificación multi-objeto |
| Reconocimiento | SAR (Li 2019) | 2D attention sobre features, sin rectificación explícita |
| Reconocimiento | NRTR (Sheng 2019) | Primer reconocedor STR puramente Transformer |
| Reconocimiento | DAN (Wang 2020) | Desacopla alineación de atención del estado del decoder |
| **End-to-end** | Mask TextSpotter (Lyu 2018-19) | Mask R-CNN para texto de forma arbitraria |
| End-to-end | CharNet (Xing 2019) | Detección + reconocimiento character-level |
| End-to-end | TextDragon (Feng 2019) | Forma arbitraria con CTC |
| End-to-end | **ABCNet (Liu 2020)** | Curvas de Bézier paramétricas + BezierAlign |

La tabla comparativa monumental del survey (Table 4) reporta accuracy de 50+ métodos sobre ocho benchmarks. Cinco lecturas: (i) entre 2011 y 2020 la accuracy en IIIT5K pasa de 24 a 94+; (ii) **ASTER** (2018) es el punto de inflexión sobre el 90% en latín regular; (iii) **NRTR** (2019) es el primer Transformer puro en el podio; (iv) datasets irregulares (IC15, SVTP, CUTE80) tienen 10-15 puntos menos que los regulares; (v) **SRN** (Yu 2020) introduce razonamiento semántico global, anunciando la dirección "reconocimiento + lenguaje" que dominará TrOCR.

---

## Retos abiertos

La sección de cierre identifica las líneas que el campo dejó abiertas a fines de 2020:

- **Generalization ability**. Modelos entrenados en sintético rinden bien en IIIT5K/SVT pero caen en COCO-Text; no generalizan a fuentes nuevas, tamaños pequeños o caracteres largos.
- **Evaluation protocols**. Inconsistencia entre experimentos (datos de entrenamiento, anotación, léxico) hace incomparables filas de la misma tabla. Baek et al. 2019 ("*What is wrong with scene text recognition model comparisons?*") empezó a responder.
- **Data issues**. Datasets reales muy chicos (miles de instancias). Mejorar síntesis (UnrealText) y explorar self/unsupervised learning sobre datos no anotados.
- **Scenarios prácticos**. Tarjetas bancarias, IDs, licencias — demandan precisión casi perfecta y son privadas; la investigación académica no las cubre.
- **Image preprocessing**. Background removal y TextSR siguen subexplorados.
- **End-to-end**. Lejos del OCR escaneado; balancear velocidades de convergencia distintas entre detección y reconocimiento.
- **Idiomas**. La mayoría de algoritmos solo cubre latín; chino, árabe y devanagari requieren tratamiento específico.
- **Seguridad**. Modelos STR vulnerables a ataques adversariales — crítico en autenticación.
- **STR + NLP**. La combinación con *text VQA*, *document understanding* e *information extraction* es la siguiente frontera.

---

## Limitaciones del survey

El survey se cierra a fines de 2020. Cinco años después, hay cuatro cosas que el lector debe saber que no aparecen:

- **TrOCR** (Li et al. 2021, Microsoft). Reemplaza completamente CNN + BiLSTM por un encoder Vision Transformer (ViT/BEiT) y un decoder Transformer pre-entrenado en NLP (RoBERTa). Pre-entrenamiento con cientos de millones de líneas sintéticas. Nuevo SOTA en IAM, IC13, IC15. **La era CRNN-style termina aquí.**
- **PARSeq** (Bautista & Atienza, ECCV 2022). *Permutation-based attention*: durante el entrenamiento se permuta el orden de los tokens de salida, permitiendo al decoder aprender *internal language modeling* bidireccional. Casi alcanza performance humana en benchmarks latinos.
- **MaskOCR** (Lyu et al. 2022). Aplica *masked image modeling* (BEiT-style) al pre-entrenamiento de STR; el encoder ViT aprende a reconstruir patches enmascarados, mejorando robustez a oclusión.
- **Foundation models multimodales**. Donut (Kim et al. 2022) hace document understanding *end-to-end* sin OCR explícito; LayoutLMv3 (Huang et al. 2022) combina texto + layout + imagen; GPT-4V, Claude con visión y Gemini realizan OCR multilingüe zero-shot directamente desde imagen. Esta convergencia OCR → modelo multimodal general es probablemente el cambio más grande post-2020.

Otras limitaciones intrínsecas: sesgo hacia latín y chino (los autores son del laboratorio HCII-SCUT, los datasets multilingües privilegian Asia); cobertura ligera de *handwriting* (IAM se menciona apenas); papers de 2020 cercanos al corte (SRN, SEED) aparecen en la tabla pero no en la discusión narrativa.

---

## Por qué importa hoy

El survey aterriza la clase 21 en tres dimensiones:

- **Pipeline mental**. Las cuatro estaciones son la rejilla para entender cualquier paper de OCR moderno. Cuando llegamos a TrOCR, la pregunta natural es: ¿cómo se mapean preprocessing/feature/sequence/prediction al encoder-decoder Transformer? (respuesta: ViT como feature + sequence, RoBERTa-decoder como prediction; preprocessing implícito en *data augmentation*).
- **Conexiones cruzadas**. Con la clase 09 (CNN como *feature extractor*), la clase 14 (Transformer en NLP y su traslado a NRTR), la clase 17 (pose recognition comparte la lógica de puntos fiduciarios con TPS), la clase 18 (embeddings de palabras pre-entrenadas guían SRN y SEED).
- **Datasets como cultura**. Conocer IIIT5K, SVT, IC13, IC15, SVTP, CUTE80, COCO-Text y MLT-2019 es como conocer ImageNet, COCO o GLUE: ningún paper STR se lee sin entender qué prueba sobre qué conjunto.

Para un *practitioner* de FHIR / documentos médicos, la conexión es directa: el OCR de cédulas, recetas manuscritas, etiquetas de medicamentos y carnés de identidad cae en STR, no en OCR clásico. El survey indica qué pipeline replicar (CRNN o ASTER como baseline, DBNet o CRAFT para detección) y qué dataset multilingüe usar si hay mezcla de scripts (MLT-2019). Para entender **ABCNet** (clase 21) en contexto, este survey es el paso obligatorio: ABCNet es el último renglón de las tablas, la coda que cierra la era pre-Transformer multimodal y abre la frontera 2020+.

---

## Notas y enlaces

- **Cómo leer este survey**: ir directo a la Figura 4 (pipeline en cuatro etapas), luego Tabla 3 (datasets), Tabla 4 (comparativa de métodos) y sección 5 (retos abiertos). El valor conceptual del survey está en esas cuatro piezas; el resto profundiza pero es opcional.
- **Repositorio oficial**: [HCIILAB/Scene-Text-Recognition](https://github.com/HCIILAB/Scene-Text-Recognition) con tablas vivas y enlaces a implementaciones.
- **Versiones**: v1 (mayo 2020), v3 (diciembre 2020, definitiva).
- **Fundamentos**: [Scene Text Recognition](/fundamentos/scene-text-recognition) · [CTC Loss](/fundamentos/ctc-loss) · [Mecanismo de atención](/fundamentos/mecanismo-atencion).
- **Papers relacionados**: [CRNN (Shi 2017)](/papers/crnn-shi-2017) · [CTC (Graves 2006)](/papers/ctc-graves-2006) · [STN (Jaderberg 2015)](/papers/stn-jaderberg-2015) · [ABCNet (Liu 2020)](/papers/abcnet-liu-2020) · [Total-Text (Chng 2017)](/papers/total-text-chng-2017).
- **Clase**: [Clase 21 — OCR / Scene Text Recognition / Document AI](/clases/clase-21).
