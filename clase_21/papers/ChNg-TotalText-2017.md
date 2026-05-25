---
title: "Total-Text: A Comprehensive Dataset for Scene Text Detection and Recognition"
authors: ["Chee Kheng Ch'ng", "Chee Seng Chan"]
year: 2017
venue: "ICDAR 2017 (extended arXiv version 1710.10400)"
slug: total-text-chng-2017
tags: ["scene-text", "dataset", "curved-text", "polygon-annotation", "detection", "recognition", "OCR"]
arxiv: "https://arxiv.org/abs/1710.10400"
github: "https://github.com/cs-chan/Total-Text-Dataset"
---

# Total-Text: A Comprehensive Dataset for Scene Text Detection and Recognition

**Chee Kheng Ch'ng, Chee Seng Chan** — Centre of Image & Signal Processing, University of Malaya. ICDAR 2017 (extended).

## Resumen ejecutivo

Total-Text es el primer dataset de scene text que ataca explícita y sistemáticamente el problema del **texto curvado** en escenas naturales. Hasta 2017, los benchmarks dominantes del campo (ICDAR'03/11/13, MSRA-TD500, SVT, IIIT5K, COCO-Text, ICDAR'15) trabajaban con texto horizontal o, en el mejor caso, con texto multi-orientado anotado mediante cuadriláteros de cuatro puntos. Ninguno representaba bien el texto que aparece en logos de negocios, fachadas, parques temáticos o turismo: arcos, círculos, ondas. El único antecedente público, CUTE80, sólo tenía 80 imágenes (mayoritariamente camisetas de fútbol) y diversidad escénica mínima.

Los autores aportan tres contribuciones que cambiarían la trayectoria del campo:

1. **1555 imágenes** (1255 train / 300 test) con **9330 instancias de palabra** anotadas a nivel de polígono de N vértices variables, donde **~46 %** del total son curvadas (4265 instancias), distribuidas en cuatro variantes (horizontal-curve 57.1 %, vertical-curve 23.5 %, circular 17.3 %, wavy 2 %) y combinadas en imágenes que mezclan hasta tres orientaciones simultáneas (horizontal + multi-oriented + curved).
2. **Protocolo de anotación polygon-shaped** (no quadrilateral fijo), con pares de vértices top/bottom que facilitan rectificación posterior, transcripción word-level con casing preservado, máscara binaria pixel-level y zonas `do not care` para textos en otros idiomas, watermarks o caracteres ilegibles.
3. **Baseline experimental** con DeconvNet (Noh et al. 2015) fine-tuneado: F-score 0.36 (P=0.40, R=0.33), evidencia cuantitativa de que las arquitecturas pre-2017 — incluso las basadas en segmentación, que ya batían SOTA en MSRA-TD500 — colapsan frente a texto curvado.

El impacto del dataset es difícil de exagerar. Total-Text se convierte en el benchmark canónico para irregular text detection y end-to-end recognition: TextSnake (Long 2018), Mask TextSpotter (Lyu 2018), TextDragon (Feng 2019), CRAFT (Baek 2019), PAN, PSENet, DBNet y, especialmente, **ABCNet (Liu CVPR 2020)** — el paper troncal de la clase 21 — lo usan como banco principal para reportar F-measure curved.

## El problema antes de Total-Text

### Benchmarks dominantes y su sesgo horizontal

El ecosistema de scene text en 2017 estaba dominado por una familia de datasets cuya geometría implícita era el **rectángulo axis-aligned** o, a lo sumo, el cuadrilátero rotado:

- **ICDAR'03** (Lucas 2003), **ICDAR'11** (Shahab 2011), **ICDAR'13** (Karatzas 2013): 509, 484 y 462 imágenes respectivamente, todas con texto **horizontal**. Anotación axis-aligned bounding box. El campo había saturado: F-score ≈ 0.9 en ICDAR'13.
- **SVT** (Wang 2010), **IIIT5K** (Mishra 2012): recognition word-level a partir de Google Street View, mayoría horizontal o ligeramente rotado.
- **MSRA-TD500** (Yao CVPR 2012): introduce arbitrary orientation (300 train / 200 test) con minimum area rectangle. Es el primer salto al texto rotado, pero sólo contiene **2 instancias curvadas en todo el dataset**, según observación directa de los autores.
- **ICDAR'15 "Incidental Scene Text"** (Karatzas 2015): 1670 imágenes capturadas con dispositivos wearables, texto fuera de foco, anotación quadrilateral de 4 puntos — la primera concesión al hecho de que un rectángulo axis-aligned no captura perspective distortion, pero todavía insuficiente para curvas.
- **COCO-Text** (Veit 2016): 63 686 imágenes, 173 589 regiones — el mayor en volumen — pero la anotación es **axis-oriented rectangle**, lo que destruye cualquier estructura curvada que la imagen pudiera tener.
- **CUTE80** (Risnumawan 2014): único antecedente con curved text explícito, pero apenas **80 imágenes** y mayormente camisetas de fútbol; no permitía entrenar deep learning ni evaluar generalización.

### El gap geométrico

La observación que articula el paper es geométrica antes que estadística: una línea recta se describe por $y = mx + c$ y se aproxima por dos puntos; un cuadrilátero rotado por cuatro puntos. Una curva, en cambio, "es libre de restricción de variación angular a lo largo de la línea". Formalmente, si un texto sigue una baseline parametrizada por $\gamma(t) = (x(t), y(t))$ con $t \in [0,1]$ y curvatura $\kappa(t) = |\dot\gamma \times \ddot\gamma| / |\dot\gamma|^3$, entonces los métodos basados en quadrilateral asumen $\kappa(t) \approx 0$ para todo $t$. En los logos circulares de Total-Text $\kappa(t)$ es aproximadamente constante y no nula; en los wavy es alternante con cambios de signo. Ningún cuadrilátero captura esa geometría.

Forzar a un detector entrenado en cuadriláteros a recuperar texto curvado tiene tres consecuencias prácticas:

- El **fondo no-texto se filtra dentro del bounding box** del groundtruth — pérdida de precisión en evaluación, y peor: ruido de supervisión durante el entrenamiento, porque el modelo aprende a etiquetar como "texto" píxeles de fondo que sistemáticamente caen dentro del rectángulo envolvente.
- El **stage de rectificación** falla: los reconocedores (CRNN, ASTER, SAR) asumen texto rectificable a un strip horizontal de altura fija, lo que sólo es válido si la anotación captura la baseline correctamente. Sin pares (top, bottom) ordenados, no hay homografía o TPS que recupere el orden de los caracteres en un arco.
- La **non-maximum suppression** estándar (basada en IoU rectangular) merge incorrectamente palabras curvadas adyacentes cuyos bounding boxes axis-aligned se solapan aunque los polígonos no lo hagan.

La Figura 5 del paper lo ilustra con cuatro logos curvos ("THE CHAPLINS", "STARBUCKS COFFEE", "POLLUTION INTERDITE", "GRAND TURK"): el rectángulo rojo sobreestima el área en >50 %, mientras que el polígono verde se ajusta tightly al baseline curvado. En el caso de "STARBUCKS COFFEE", el círculo anular de la marca incluye dentro del rectángulo el logo central de la sirena — un objeto visual que el detector aprenderá erróneamente a asociar con clase "texto" si entrena con groundtruth rectangular.

## Caracterización del dataset

### Volumen y splits

| Atributo | Valor |
|---|---|
| Imágenes totales | 1555 |
| Train / Test split | 1255 / 300 |
| Palabras anotadas | 9330 |
| Instancias curvadas | 4265 (~45.7 %) |
| Promedio instancias / imagen | 6.0 |
| Orientaciones promedio / imagen | 1.8 |
| Idioma | Inglés (otros → `do not care`) |
| Granularidad transcripción | Word-level con casing |

### Las tres orientaciones simultáneas

A diferencia de ICDAR'15 (que mezcla horizontal + multi-oriented) o CUTE80 (sólo curved), Total-Text fue construido para forzar a las arquitecturas a manejar **horizontal + multi-oriented + curved** en una misma imagen. La Figura 6a categoriza imágenes según combinaciones:

- **Top — Una orientación**: HC, VC, Cir, W (sólo horizontal curve, sólo vertical curve, sólo circular, sólo wavy).
- **Middle — Dos orientaciones**: Cir+H, MO+HC, W+H.
- **Bottom — Tres orientaciones**: H+MO+VC, H+MO+HC, H+MO+Cir.

Más de la mitad de las imágenes contienen al menos dos orientaciones distintas (Figura 7b). Esta diversidad obliga a los algoritmos a abandonar el "orientation assumption" — la heurística de proyectar una recta media para agrupar caracteres en una línea, viable en MSRA-TD500 pero impracticable cuando la línea es polinomial.

### Variantes de curvatura

La Figura 7c desglosa la distribución de curvas:

- **Horizontal curve**: 57.1 %
- **Vertical curve**: 23.5 %
- **Circular**: 17.3 %
- **Wavy**: 2 %

La predominancia del arco simétrico (horizontal curve) refleja una preferencia perceptual humana documentada en *The Science of Social Vision* (Adams 2011, ref 21 del paper): los diseñadores prefieren composiciones simétricas, lo que se traslada a logos y signage.

### Procedencia escénica

La Figura 7d muestra dónde aparece el texto curvado en el corpus:

- **Business-related** (Nandos, Starbucks, restaurantes, tiendas): 61.9 %
- **Tourist spots** (Beverly Hills, Harajuku, parques, museos): 21.1 %
- **Non-business**: 8.2 %
- **Formal information**: 5.6 %
- **Others / Advertising / Clubs**: < 5 % combinado

Esta distribución es relevante para pipelines de OCR aplicados a retail, logística de tiendas, mapeo urbano y reconocimiento de marcas — más que para documentos escaneados.

## Annotation protocol

### Polígono de N vértices

El cambio metodológico clave es abandonar el quadrilateral fijo y adoptar polígonos con **número variable de vértices** por instancia. La Figura 7a muestra el histograma de #vértices por instancia: la mediana ronda los 6-8 puntos, con cola larga hasta >20 vértices para curvas complejas.

Los autores estructuran los vértices en **pares (top, bottom)** que recorren la baseline superior e inferior de la palabra. Esta convención no es accidental: facilita rectificación posterior por TPS (Thin-Plate Spline) o, en arquitecturas más recientes, parametrización Bezier (ABCNet). Cada par top/bottom define una columna vertical local del texto, lo que permite construir un sampling grid que "enderece" la palabra a un strip horizontal — exactamente lo que Mask TextSpotter y ABCNet harán después.

### Granularidad word-level con `do not care`

La transcripción sigue la definición de COCO-Text: una palabra es una *uninterrupted sequence of characters separated by a space*. Casing se preserva (`BARBER` ≠ `barber`). Regiones marcadas como `do not care` cubren:

- Texto en idiomas no-inglés.
- Watermarks digitales.
- Caracteres ilegibles o severamente ocluidos.

Los detectores deben **filtrar** estas regiones antes de evaluación — sólo cuentan True Positives sobre regiones `care`.

### Pares (top, bottom) y rectificación posterior

La estructura por pares no es sólo una elección visual. Si etiquetamos los vértices como $\{T_1, T_2, \ldots, T_N\}$ (top baseline) y $\{B_1, B_2, \ldots, B_N\}$ (bottom baseline), entonces cada cuadrilátero local $(T_i, T_{i+1}, B_{i+1}, B_i)$ define una **celda** que mapea a un rectángulo de altura fija en el espacio rectificado. La operación de rectificación que TextSnake, Mask TextSpotter y ABCNet implementan es esencialmente:

$$
\text{rectify}(x_{ij}) = \mathcal{B}(T_i, T_{i+1}, B_{i+1}, B_i; u_{ij}, v_{ij})
$$

donde $\mathcal{B}$ es una interpolación bilineal sobre la celda y $(u_{ij}, v_{ij})$ son las coordenadas normalizadas en el output grid. ABCNet generaliza esto sustituyendo el polígono por dos curvas de Bézier parametrizadas y obteniendo $T_i, B_i$ por muestreo uniforme — pero la convención del groundtruth de Total-Text es lo que hace posible esa parametrización a posteriori.

### Múltiples representaciones complementarias

Cada imagen viene con:

1. **Coordenadas spaciales del polígono** (N vértices).
2. **Bounding box rectangular** (deprecated por los autores pero incluido por compatibilidad).
3. **Transcripción** word-level.
4. **Orientación** etiquetada per-instancia (Horizontal / Multi-oriented / Curved / NA).
5. **Máscara binaria pixel-level** (Figura 8): 1 = región de texto, 0 = fondo, suministrada para alimentar pipelines de **segmentación semántica** (FCN, U-Net, DeconvNet, Mask R-CNN).

La inclusión simultánea de polígono y máscara es la apuesta arquitectural del dataset: anticipa que la siguiente generación de detectores serán segmentation-based, no proposal-based, y deben tener groundtruth coherente con ese paradigma.

### Comparación de formatos de anotación

| Formato | Parámetros | Datasets | Limitación |
|---|---|---|---|
| Axis-aligned bbox | 4 (x, y, w, h) | ICDAR'03/'11/'13, COCO-Text | No captura rotación ni curva |
| Quadrilateral (4 puntos) | 8 (x₁..x₄, y₁..y₄) | ICDAR'15, MSRA-TD500 | No captura curva, fondo dentro del box |
| **Polygon (N puntos)** | 2N variable | **Total-Text**, CTW-1500 | Anotación manual costosa |
| Pixel-level mask | H×W bool | Total-Text (complementario) | Pérdida de instance separation |

## Metadata y estadísticas

### Comparación con benchmarks contemporáneos

| Dataset | Año | Imágenes | Instancias | Orientación | Anotación |
|---|---|---|---|---|---|
| ICDAR'13 | 2013 | 462 | ~2k | Horizontal | Bbox |
| MSRA-TD500 | 2012 | 500 | ~1.5k | Multi-oriented | Min-area rect |
| ICDAR'15 | 2015 | 1670 | ~17k | Multi-oriented | Quadrilateral |
| COCO-Text | 2016 | 63 686 | 173 589 | Mixed | Axis-aligned bbox |
| CUTE80 | 2014 | 80 | ~280 | Curved | Polygon |
| **Total-Text** | 2017 | **1555** | **9330** | **H + MO + Curved** | **Polygon N pts** |
| CTW-1500 | 2017 | 1500 | ~10k | Curved (line-level) | Polygon 14 pts |

CTW-1500 (Yuliang Liu 2017) emergió contemporáneamente con un enfoque distinto: anotación a nivel de **línea de texto** (no word) con polígono fijo de 14 puntos. Total-Text mantiene granularidad word-level — más alineada con pipelines end-to-end de detection + recognition — y polígono de N variable.

## Evaluation protocol

### DetEval modificado

Los autores adoptan **DetEval** (Wolf & Jolion 2006, IJDAR), el mismo protocolo de ICDAR, pero modifican el cálculo del **minimum intersection area** para acomodar groundtruth polygonal. En esencia:

- **Precision** $P = \frac{|D \cap G|}{|D|}$ donde $D$ es la detección y $G$ es el groundtruth polygonal.
- **Recall** $R = \frac{|D \cap G|}{|G|}$.
- **F-measure** $F = \frac{2PR}{P + R}$.

La intersección se computa en el plano del polígono, no sobre el rectángulo envolvente — distinción crítica porque, como la Tabla II del paper demuestra empíricamente, evaluar la misma detección contra groundtruth rectangular vs. polygonal puede variar precision en >0.4 puntos absolutos.

### Lexicon modes

Para evaluación end-to-end recognition (no explícita en el paper original pero adoptada por trabajos posteriores como ABCNet), Total-Text soporta cuatro modos de léxico:

- **None**: vocabulario abierto.
- **Weak / Strong**: subconjuntos de palabras candidatas.
- **Full**: vocabulario completo del test set (típicamente unos miles de palabras).

El reporte estándar en papers post-2018 es F-measure **None** y F-measure **Full**, donde Full sirve como upper-bound de qué tan bien el detector localiza, descontando errores de transcripción.

### Matching de detecciones a groundtruth en DetEval

DetEval distingue entre tres tipos de match para evitar penalizar splits/merges legítimos:

- **One-to-one**: una detección $D_i$ matchea exactamente un groundtruth $G_j$ si simultáneamente $|D_i \cap G_j| / |D_i| > t_p$ (precision threshold, típicamente 0.4) y $|D_i \cap G_j| / |G_j| > t_r$ (recall threshold, típicamente 0.8).
- **One-to-many** (split): una detección $G_j$ del groundtruth está cubierta colectivamente por múltiples $D_i$. Se aplica una penalización $f_{sc}$ (típicamente 0.8) para desincentivar splits artificiales.
- **Many-to-one** (merge): múltiples groundtruths son cubiertos por una sola detección — penalización análoga.

En texto curvado, esta distinción es crítica porque un detector polygon-aware puede recuperar correctamente la palabra como una unidad, mientras que un detector rectangular tiende a producir splits que un DetEval menos sofisticado castigaría injustamente.

### IoU sobre polígonos

El cálculo de $|D \cap G|$ y $|D \cup G|$ entre dos polígonos $D$ y $G$ se implementa por **Sutherland-Hodgman clipping** seguido de cómputo de área via Shoelace formula:

$$
|P| = \frac{1}{2} \left| \sum_{i=0}^{N-1} (x_i y_{i+1} - x_{i+1} y_i) \right|
$$

Esta es la operación que `shapely.geometry.Polygon` o `cv2.intersectConvexConvex` ejecutan en pipelines modernos. Total-Text estandariza esta evaluación al distribuir su propio script de DetEval modificado junto con el dataset.

## Baseline methods on Total-Text

### Resultados cualitativos previos

Antes de su experimento principal, los autores corren tres detectores SOTA pre-2017 sobre muestras de Total-Text (Figura 4):

- **Yin et al.** (T-PAMI 2014, ref 22): basado en MSER + stroke detection. Falla en curved text — sus bounding boxes rojos cortan caracteres del logo "STARBUCKS COFFEE" arqueado.
- **Huang et al.** (ECCV 2014, ref 6): MSER + CNN-induced trees. Sufre el mismo failure mode.
- **Shi et al. SegLink** (CVPR 2017, ref 7): linking segments. Mejora porque agrupa segmentos locales, pero la conectividad global asume baseline recta, lo que rompe la palabra curvada en fragmentos disconexos.

Ninguno supera 0.5 F-measure en muestras curvadas. Esto confirma cualitativamente que el gap no es de hyperparameter tuning, es estructural.

### Experimento principal: DeconvNet fine-tuned

Los autores eligen **DeconvNet** (Noh et al. ICCV 2015, ref 24) como herramienta de investigación por dos razones:

1. SOTA en segmentación semántica en PASCAL VOC.
2. **Múltiples capas deconvolucionales** que permiten observar progresivamente cómo la red refina la localización — útil para diagnóstico, no sólo para benchmark.

**Setup**:

- Pre-entrenamiento en **COCO-Text** (sólo categoría "legible"), única opción a escala suficiente para los 252 M parámetros de DeconvNet.
- Última capa convolucional reducida de 21 clases (PASCAL VOC) a 2 (text / non-text).
- Imágenes resampleadas a 256×256, patches con <10 % text region eliminados, ~200k patches de entrenamiento + 80k de validación.
- Data augmentation: horizontal flipping + random crop 224×224.
- Fine-tuning en Total-Text training split.

**Inferencia**: input 224×224 → forward pass → saliency map binarizada a threshold 0.5 → connected components → polígonos.

### Resultados

**Tabla I del paper**:

| Dataset | Recall | Precision | F-score |
|---|---|---|---|
| Total-Text | 0.33 | 0.40 | **0.36** |

Un F-score de 0.36 sobre Total-Text — usando una de las mejores arquitecturas de segmentación de la época, pre-entrenada en el dataset más grande disponible — es la métrica que define el gap. En MSRA-TD500, métodos contemporáneos como CCTN (He et al. 2016) o Zhang et al. (CVPR 2016) reportaban F-scores entre 0.75 y 0.84. En Total-Text bajan a 0.36.

### Diagnóstico de fallos

Los autores identifican dos causas raíz (Figura 11):

1. **Robustez insuficiente a fondos challenging**: ladrillos, vegetación, paredes texturadas — el modelo tiende a activarse sobre estructuras repetitivas que se parecen visualmente a strokes de texto. Atribuyen esto al training data, donde fondos cercanos al texto eran labeled como "text" debido a la looseness del bbox de COCO-Text.
2. **Múltiples palabras agrupadas como una sola región**: el output es a nivel de "text region", no de "word instance". Falta un módulo de **text line supervision** que separe palabras adyacentes — Zhang et al. (CVPR 2016) lo implementan con un Text Block FCN (TBN) que aporta +0.3 F-score (de 0.5 a 0.84).

La Figura 12 muestra un failure mode adicional: en curved text, la confianza decae en los **extremos** de la curva. Las activaciones intermedias (Figura 9) revelan que las capas tempranas (14×14 deconv) detectan blobs grosseros, mientras que las capas finales (224×224) recuperan la geometría — pero la confianza promedio es baja en bordes.

## Análisis cualitativo

### Diversidad visual del corpus

La Figura 13 cataloga seis tipos de challenge visual presentes en Total-Text:

- **Distorsión de perspectiva** (Fig 13a): signage fotografiado en ángulo.
- **Variación de fonts** (13b): serifs, sans-serif, scripts, decorativos.
- **Variación de tamaños** (13c): de letras de logo gigantes a tags pequeñas en escaparates.
- **Fondos complejos** (13d): ladrillos, rejas, vegetación.
- **Iluminación desigual** (13e): sombras, neón nocturno.
- **Bajo contraste** (13f): metal sobre metal, texto grabado en piedra.

Esta combinación supera en variabilidad escénica a ICDAR'13/'15 y compite con COCO-Text — con la ventaja de tener anotación polygonal precisa.

### Ablación de groundtruth polygonal vs. rectangular

La **Tabla II** del paper compara, para 14 detecciones individuales, las métricas obtenidas evaluando contra groundtruth polygonal vs. rectangular:

- Ejemplo "PURE" (Fig 15c): Precision sube de 0.13 (polygon) a 0.16 (rectangle), Recall 1 → 1. El rectángulo *infla* precision artificialmente porque la detección entera cae dentro del rectángulo amplio.
- Ejemplo "ICE-CREAM": P 0.23 → 0.44, R 0.99 → 0.98.
- Ejemplo "COSTA": P 0.86 → 0.95, R 0.64 → 0.43. Aquí la métrica revela el sesgo opuesto: el rectángulo amplio reduce recall porque la detección polygonal interseca menos del rectángulo que del polígono ajustado.

La conclusión metodológica es directa: **groundtruth polygonal produce métricas más fieles** al desempeño real del detector. Evaluar curved text contra rectángulos es ejercicio engañoso.

## Impacto medible

El paper se publica en octubre de 2017. Los dos años siguientes son explosivos para irregular scene text. La hoja de ruta de impacto:

### 2018 — Primera generación curved-aware

- **TextSnake** (Long et al. ECCV 2018): representa texto como una secuencia de discos a lo largo de un centerline, con radio $r(t)$ y orientación local $\theta(t)$ por punto. Cada palabra es una unión de discos $\bigcup_{t} D(c(t), r(t))$ donde $c(t)$ es la curva central y $D$ es el disco euclidiano. Reporta F=78.4 en Total-Text — un salto desde 0.36 (DeconvNet baseline) a 0.784, en menos de un año. Es la prueba de que **la representación geométrica importa más que la profundidad de la red**.
- **Mask TextSpotter** (Lyu et al. ECCV 2018): combina Mask R-CNN con un character-level segmentation head. F=52.9 (None) / 71.8 (Full) en Total-Text — el primer end-to-end curved spotter.
- **Liu et al.** (CVPR 2018, FOTS extension): F=57.8 en Total-Text.

### 2019 — Refinamiento de pipelines

- **TextDragon** (Feng et al. ICCV 2019): combinación de instance-level detection + RoISlide para curved RoI. Mejora la rectificación.
- **CRAFT** (Baek et al. CVPR 2019): character region + affinity field. F=78.7 en Total-Text.
- **PSENet** (Wang et al. CVPR 2019): progressive scale expansion. F=80.9 Total-Text.
- **PAN** (Wang et al. ICCV 2019): pixel aggregation. F=82.1 Total-Text.

### 2020 — Bezier curve parametrization (ABCNet)

- **ABCNet** (Liu et al. CVPR 2020): **el paper troncal de la clase 21**. Representa cada texto curvado mediante dos curvas de **Bézier de tercer orden** (8 parámetros por curva, 16 totales — vs 2N variables del polígono libre). BezierAlign sustituye RoIAlign para sampling rectificado. Resultado en Total-Text: **F=69.5 None / 78.4 Full** con backbone single-scale, y **F=69.5 / 80.6** con multi-scale (ABCNet-MS). Más relevante aún: end-to-end framework inference rate **~17 fps**, vs <5 fps de Mask TextSpotter — apertura del camino al deployment real-time.
- **DBNet** (Liao et al. AAAI 2020): differentiable binarization. F=84.7 Total-Text.

### Métrica de saturación

A 2022 (post-cutoff parcial pero verificable hasta clase 21), los F-scores SOTA en Total-Text rondan 0.87-0.89. La pendiente de mejora se aplana, lo que sugiere que el dataset (con sus 1555 imágenes) empieza a ser **insuficiente para discriminar arquitecturas modernas** — escenario familiar para quien vivió la saturación de ICDAR'13.

### Patrón de mejora arquitectural

Si trazamos los F-scores SOTA en Total-Text contra el año:

| Año | Método | F-score (None) | Innovación clave |
|---|---|---|---|
| 2017 | DeconvNet (baseline) | 0.36 | Segmentation FCN |
| 2018 | TextSnake | 0.78 | Discos + centerline |
| 2018 | Mask TextSpotter | 0.53 / 0.72 (Full) | Mask R-CNN + char head |
| 2019 | CRAFT | 0.79 | Character region + affinity |
| 2019 | PSENet | 0.81 | Progressive scale expansion |
| 2019 | PAN | 0.82 | Pixel aggregation + FPEM |
| 2020 | ABCNet (single-scale) | 0.67 | Bezier curves + BezierAlign |
| 2020 | ABCNet-MS (multi-scale) | 0.78 / 0.81 (Full) | + multi-scale inference |
| 2020 | DBNet | 0.85 | Differentiable binarization |
| 2021 | DBNet++ | 0.87 | Adaptive scale fusion |
| 2022 | TESTR / SwinTextSpotter | ~0.88 | Transformer-based |

El salto de 2017 → 2018 (0.36 → 0.78) corresponde al cambio de representación geométrica. El refinamiento posterior (0.78 → 0.88) viene de mejorar el backbone, loss y multi-scale — incrementos marginales pero consistentes. Esta dinámica confirma que **el bottleneck no era de capacidad de modelado sino de formulación del output**.

## Limitaciones del dataset

Los autores son honestos sobre lo que Total-Text no resuelve:

1. **Escala**: 1555 imágenes es pequeño para deep learning moderno. Pre-training en COCO-Text o synthetic data (SynthText 800k, MJSynth 9M) sigue siendo necesario.
2. **Idioma**: sólo inglés. Texto chino, árabe, devanagari, cirílico queda fuera. Esto motivará **MLT** (Nayef et al. ICDAR 2017) y datasets multilingües posteriores.
3. **Curated, no incidental**: las imágenes fueron seleccionadas con curved text en mente. La distribución no representa una deployment real, donde la mayoría de los frames de un robot móvil o smartphone no contendrán texto curvado.
4. **Granularidad word vs. line**: la decisión de anotar a nivel de palabra (vs. línea, como CTW-1500) ayuda a pipelines de recognition pero dificulta evaluar agrupamiento layout-level.
5. **Errores de anotación manual**: el corpus fue anotado por los autores + 3 miembros del laboratorio, con cross-checking. Inevitablemente hay ruido — particularmente en decisión de qué cuenta como `do not care`.
6. **Sin evaluación end-to-end estandardizada en el paper original**: el paper sólo reporta detection F-score. El protocolo end-to-end se consolidó después, en trabajos posteriores que usaron Total-Text.

## Datasets sucesores y complementarios

Total-Text inaugura una familia de datasets que extienden distintas dimensiones del problema:

- **CTW-1500** (Yuliang Liu et al. ACMM 2017): 1500 imágenes con polígono de 14 puntos a nivel de **línea**. Complemento natural a Total-Text (word-level). El estándar de facto es reportar resultados en **ambos**.
- **MLT 2017 / 2019** (Nayef et al.): multilingüe con 9 scripts (latín, chino, árabe, hebreo, etc.). Resuelve la limitación de idioma.
- **ArT — Arbitrary-shaped Text** (Chng et al. ICDAR 2019, mismos autores): 10 166 imágenes, fusión de Total-Text + CTW-1500 + nuevas imágenes. Es la evolución directa de Total-Text a gran escala.
- **LSVT** (Sun et al. 2019): 50 000 imágenes street view con weakly + fully annotated splits.
- **ReCTS** (2019): texto chino en signage de restaurantes.
- **HierText** (Long et al. CVPR 2022): jerarquía paragraph → line → word, también curvada.

## Por qué importa para la clase 21

Total-Text es el dataset **sobre el que ABCNet reporta sus números headline** en la slide "Quantitative results on Total-Text" del PDF de clase. Tres conexiones explícitas con el material:

1. **Métrica directa**: ABCNet single-scale F=67.1, ABCNet con synthetic pre-training F=69.5 (None) / 78.4 (Full). Estos números sólo tienen sentido sobre Total-Text — la comparación entre Mask TextSpotter (F=52.9 None / 71.8 Full) y ABCNet ocurre sobre este benchmark.
2. **Justificación geométrica de Bezier**: la decisión de ABCNet de representar curvas mediante dos polinomios de Bézier de orden 3 es una respuesta directa al formato de anotación de Total-Text — los 2N puntos del polígono se interpolan a 8+8=16 parámetros de Bézier, con error de reconstrucción <1 píxel en >99 % de los casos según Liu et al. La compatibilidad con la geometría del groundtruth de Total-Text es lo que hace posible BezierAlign y, por extensión, el end-to-end real-time.
3. **Continuidad pedagógica**: en la trayectoria de la clase 21 (detection objetos → detection texto → recognition texto), Total-Text es el momento en que el dominio "texto" deja de ser una extensión trivial de detection genérica y se convierte en un problema con representación geométrica propia. Sin Total-Text, no existen TextSnake, Mask TextSpotter ni ABCNet — y la slide de quantitative results sería sobre ICDAR'15 quadrilateral, un problema mucho más restringido.

### Conexión con otros papers de la clase 21

- **Liu ABCNet 2020**: usa Total-Text como benchmark principal.
- **Chen TextRecognitionWild 2020**: survey que cataloga Total-Text como uno de los benchmarks dominantes de irregular text desde 2018.
- **Girshick Fast R-CNN 2015**: base arquitectural de RoI pooling, que TextSnake/Mask TextSpotter adaptan a curvas — adaptación motivada por la imposibilidad de RoI rectangular de capturar Total-Text.
- **Tian FCOS 2019**: anchor-free framework que ABCNet reutiliza, eligiendo anchor-free precisamente porque las shapes de Total-Text son demasiado variadas para anchor templates.
- **Rezatofighi GIoU 2019**: loss alternativa para bounding boxes — en Total-Text, GIoU se generaliza a polygon IoU, otro indicador de que la métrica clásica IoU rectangular es insuficiente.

## Cierre

Total-Text es un paper relativamente corto (10 páginas), con experimentos modestos (un solo baseline DeconvNet, F=0.36) y sin novedad arquitectural propia. Su valor está enteramente en haber **identificado correctamente el gap geométrico** y haber producido el groundtruth que permite cerrarlo. La lección de método es clara: en problemas donde la geometría del output importa, **la anotación es la palanca de avance**, no la arquitectura. Cinco años después, el ecosistema de irregular scene text — desde TextSnake hasta DBNet++ pasando por ABCNet — descansa sobre la decisión de Ch'ng y Chan de anotar 9330 instancias con N vértices variables en lugar de 4.

Para Roberto, ingeniero senior trabajando en pipelines de OCR para producción: la heurística práctica derivada del paper es **auditar el formato de anotación del dataset antes de elegir arquitectura**. Si los polígonos tienen >4 vértices, las arquitecturas quadrilateral-only (EAST, vanilla SegLink) van a saturar muy por debajo del SOTA. Si los polígonos son rectos (4 vértices), invertir en polygon-based detectors es overkill computacional.

## Lecciones transferibles a otros dominios

El patrón de Total-Text —ir un nivel de geometría más libre que el estándar de la época— es transferible a otros problemas estructurados:

- **Detección de objetos no convexos**: instance segmentation con Mask R-CNN sigue la misma lógica (pasar de bbox a máscara). Coco's pixel-mask GT habilitó el salto.
- **Pose estimation**: pasar de bounding box humano a 17 keypoints (COCO) y luego a 3D SMPL (clase 17 del curso) reproduce el mismo movimiento.
- **Medical imaging**: en segmentación de tumores con bordes irregulares, anotar máscaras polygonales rather than bbox es exactamente el mismo argumento.
- **FHIR matching** (dominio profesional de Roberto): si los identificadores de pacientes están anotados sólo como matches binarios (match / no-match), los modelos saturan. Anotar el **tipo de match** (exact name + DOB, exact name + addr, soundex + DOB, etc.) habilita modelos más finos —misma lección, distinto dominio—.

En todos los casos, el move ganador es **enriquecer el espacio de output del groundtruth** antes que escalar la red. Total-Text es un ejemplo limpio porque el coste de anotación es lineal en N vértices y el beneficio en F-score es supra-lineal.

## Reproducibilidad y disponibilidad

El dataset está publicado en `https://github.com/cs-chan/Total-Text-Dataset` con licencia académica. Incluye:

- Imágenes train/test originales.
- Groundtruth en formato `.txt` con polígonos N-puntos.
- Groundtruth en formato `.mat` (MATLAB) para compatibilidad con DetEval original.
- Máscaras binarias pixel-level (`.png`).
- Script DetEval modificado.
- Conversión a formatos COCO-JSON y YOLOv5-polygon-txt en forks comunitarios posteriores.

A 2026, el repositorio supera las 700 estrellas en GitHub y se mantiene vigente como parte del estándar de evaluación para cualquier paper de scene text que reclame ser SOTA en irregular text — testimonio del impacto duradero de una contribución que, en su núcleo, fue una decisión de anotación bien tomada.
