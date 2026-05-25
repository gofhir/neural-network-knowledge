---
title: "Total-Text (Curved Scene Text Dataset)"
weight: 105
math: true
---

{{< paper-card
    title="Total-Text: A Comprehensive Dataset for Scene Text Detection and Recognition"
    authors="Ch'ng, Chan"
    year="2017"
    venue="ICDAR 2017"
    pdf="/papers/total-text-chng-2017.pdf"
    arxiv="1710.10400" >}}
Primer dataset de scene text construido sistematicamente para **texto curvado**. **1555 imagenes** (1255 train / 300 test), **9330 instancias** anotadas como **poligonos de N vertices variables** con pares (top, bottom), combinando tres orientaciones en cada imagen: **horizontal + multi-oriented + curved**. El baseline DeconvNet alcanza apenas **F=0.36**, evidencia cuantitativa de que las arquitecturas pre-2017 colapsan ante curvas y de que el bottleneck es la **representacion geometrica del output**, no la profundidad de la red.
{{< /paper-card >}}

---

## El problema antes de Total-Text

El ecosistema de scene text en 2017 estaba dominado por geometrias rectangulares o, a lo sumo, cuadrilateros rotados:

| Dataset | Ano | Imagenes | Anotacion | Orientacion |
| --- | --- | --- | --- | --- |
| ICDAR'03/'11/'13 | 2003-13 | ~500 c/u | Axis-aligned bbox | Horizontal |
| MSRA-TD500 | 2012 | 500 | Min-area rect | Multi-oriented |
| ICDAR'15 | 2015 | 1670 | Quadrilateral 4 pts | Multi-oriented |
| COCO-Text | 2016 | 63 686 | Axis-aligned bbox | Mixed (anotacion lossy) |
| CUTE80 | 2014 | **80** | Polygon | Curved (camisetas de futbol) |

El gap es geometrico antes que estadistico. Una linea recta se aproxima por dos puntos; un cuadrilatero rotado por cuatro; una curva parametrizada por $\gamma(t) = (x(t), y(t))$ con curvatura

$$
\kappa(t) = \frac{|\dot\gamma \times \ddot\gamma|}{|\dot\gamma|^3}
$$

no se aproxima por **ningun** cuadrilatero. En logos circulares $\kappa(t)$ es aproximadamente constante y no nula; en wavy es alternante. Forzar a un detector entrenado en cuadrilateros a recuperar texto curvado produce tres consecuencias:

1. **Fondo no-texto se filtra dentro del bounding box** del groundtruth -> ruido de supervision, el modelo aprende a etiquetar fondo como texto.
2. **El stage de rectificacion falla**: reconocedores CRNN/ASTER/SAR asumen texto rectificable a strip horizontal de altura fija, valido solo si la baseline esta capturada correctamente.
3. **Non-maximum suppression rectangular** mergea palabras curvadas adyacentes cuyos polygons no se solapan.

El antecedente directo, CUTE80, tenia apenas 80 imagenes -> insuficiente para entrenar deep learning ni para benchmark. Total-Text llena ese hueco con escala y diversidad escenica reales.

## El dataset

| Atributo | Valor |
| --- | --- |
| Imagenes | 1555 (1255 train + 300 test) |
| Instancias palabra | 9330 |
| Instancias curvadas | 4265 (~46 %) |
| Promedio instancias/imagen | 6.0 |
| Orientaciones promedio/imagen | 1.8 |
| Idioma | Ingles (otros -> `do not care`) |
| Granularidad transcripcion | Word-level con casing preservado |

### Las tres orientaciones simultaneas

A diferencia de ICDAR'15 (horizontal + multi-oriented) o CUTE80 (solo curved), Total-Text fue construido para forzar a los detectores a manejar **horizontal + multi-oriented + curved en la misma imagen**. Mas de la mitad del corpus contiene al menos dos orientaciones distintas, lo que invalida la heuristica clasica de proyectar una recta media para agrupar caracteres.

### Variantes de curvatura

| Variante | Proporcion |
| --- | --- |
| Horizontal curve | 57.1 % |
| Vertical curve | 23.5 % |
| Circular | 17.3 % |
| Wavy | 2 % |

La predominancia del arco simetrico refleja una preferencia perceptual humana: los disenadores prefieren composiciones simetricas, lo que se traslada a logos y signage.

### Procedencia escenica

- **Business-related** (Nandos, Starbucks, restaurantes, tiendas): 61.9 %
- **Tourist spots** (Beverly Hills, Harajuku, parques, museos): 21.1 %
- **Non-business**: 8.2 %
- **Formal information**: 5.6 %

Distribucion relevante para OCR en retail, logistica de tiendas, mapeo urbano y reconocimiento de marcas -> mas que para documentos escaneados.

## Annotation protocol

### Poligono de N vertices con pares (top, bottom)

El cambio metodologico clave es abandonar el cuadrilatero fijo y adoptar poligonos con **numero variable de vertices**. La mediana ronda los 6-8 puntos por instancia, con cola hasta >20 vertices en curvas complejas.

Los vertices se estructuran en pares $(T_i, B_i)$ que recorren la baseline superior e inferior de la palabra. Esta convencion no es accidental: cada cuadrilatero local $(T_i, T_{i+1}, B_{i+1}, B_i)$ define una **celda** que mapea a un rectangulo de altura fija en el espacio rectificado. La rectificacion posterior es:

$$
\text{rectify}(x_{ij}) = \mathcal{B}(T_i, T_{i+1}, B_{i+1}, B_i;\, u_{ij}, v_{ij})
$$

donde $\mathcal{B}$ es interpolacion bilineal y $(u_{ij}, v_{ij})$ son coordenadas normalizadas en el output grid. **ABCNet generaliza esto** sustituyendo el poligono por dos curvas Bezier parametrizadas y obteniendo $T_i, B_i$ por muestreo uniforme.

### Granularidad word-level con `do not care`

Una palabra es una *uninterrupted sequence of characters separated by a space*. Casing preservado (`BARBER` != `barber`). Regiones marcadas como `do not care` cubren texto no-ingles, watermarks digitales y caracteres ilegibles -> los detectores deben filtrarlas antes de la evaluacion.

### Multiples representaciones

Cada imagen viene con:

1. **Coordenadas del poligono** (N vertices).
2. **Bounding box rectangular** (deprecated por los autores, incluido por compatibilidad).
3. **Transcripcion word-level**.
4. **Orientacion** etiquetada por instancia (H / MO / Curved / NA).
5. **Mascara binaria pixel-level** -> alimenta pipelines de segmentacion semantica (FCN, DeconvNet, Mask R-CNN).

### Comparacion de formatos

| Formato | Parametros | Datasets | Limitacion |
| --- | --- | --- | --- |
| Axis-aligned bbox | 4 | ICDAR'03/'11/'13, COCO-Text | No captura rotacion ni curva |
| Quadrilateral | 8 | ICDAR'15, MSRA-TD500 | Fondo dentro del box, no curva |
| **Polygon (N pts)** | 2N variable | **Total-Text**, CTW-1500 | Anotacion manual costosa |
| Pixel-level mask | H x W bool | Total-Text (complementario) | Sin separacion de instancias |

## Evaluation

Total-Text adopta **DetEval** (Wolf & Jolion 2006) modificado para groundtruth poligonal:

- **Precision** $P = |D \cap G| / |D|$
- **Recall** $R = |D \cap G| / |G|$
- **F-measure** $F = 2PR / (P + R)$

La interseccion se computa en el plano del poligono via **Sutherland-Hodgman clipping** + Shoelace:

$$
|P| = \frac{1}{2} \left| \sum_{i=0}^{N-1} (x_i y_{i+1} - x_{i+1} y_i) \right|
$$

Distincion critica: evaluar la misma deteccion contra groundtruth rectangular vs. poligonal puede variar precision en **>0.4 puntos absolutos** (Tabla II del paper). Forzar metrica rectangular sobre texto curvado es ejercicio enganoso.

DetEval distingue tres tipos de match para no penalizar splits/merges legitimos:

- **One-to-one**: $|D_i \cap G_j|/|D_i| > t_p$ y $|D_i \cap G_j|/|G_j| > t_r$.
- **One-to-many** (split): un GT cubierto por multiples detecciones -> penalizacion $f_{sc} \approx 0.8$.
- **Many-to-one** (merge): multiples GT cubiertos por una deteccion -> penalizacion analoga.

### Lexicon modes para end-to-end

Cuatro modos de lexico estandarizados por trabajos posteriores:

- **None**: vocabulario abierto.
- **Weak / Strong**: subconjuntos de candidatas.
- **Full**: vocabulario completo del test set.

El reporte estandar post-2018 es F-measure **None** y **Full**; este ultimo sirve como upper-bound de localizacion descontando errores de transcripcion.

## Baselines on Total-Text

Los autores corren tres detectores SOTA pre-2017 sobre muestras (Yin et al. T-PAMI 2014, Huang et al. ECCV 2014, Shi et al. SegLink CVPR 2017): **ninguno supera 0.5 F-measure** en curved text. El experimento principal usa **DeconvNet** (Noh et al. ICCV 2015) pre-entrenado en COCO-Text (categoria "legible") y fine-tuneado en Total-Text:

| Metodo | Recall | Precision | F-score |
| --- | --- | --- | --- |
| DeconvNet fine-tuned | 0.33 | 0.40 | **0.36** |

En MSRA-TD500, metodos contemporaneos reportaban F-scores entre 0.75 y 0.84. En Total-Text bajan a **0.36**. La metrica define el gap.

**Diagnostico de fallos** (Figuras 11-12 del paper):

1. **Robustez insuficiente a fondos challenging**: ladrillos, vegetacion, paredes texturadas activan al detector como si fueran strokes.
2. **Multiples palabras agrupadas como una sola region**: falta supervision a nivel de text-line para separar palabras adyacentes.
3. **Confianza decae en los extremos de la curva**: las activaciones de las capas finales recuperan la geometria, pero la confianza promedio es baja en bordes.

## Impacto

El paper se publica en octubre de 2017. Los dos anos siguientes son explosivos para irregular scene text:

| Ano | Metodo | F-score (None) en Total-Text | Innovacion clave |
| --- | --- | --- | --- |
| 2017 | DeconvNet baseline | 0.36 | Segmentation FCN |
| 2018 | TextSnake | 0.78 | Discos a lo largo de un centerline |
| 2018 | Mask TextSpotter | 0.53 / 0.72 (Full) | Mask R-CNN + character head |
| 2019 | CRAFT | 0.79 | Character region + affinity |
| 2019 | PSENet | 0.81 | Progressive scale expansion |
| 2019 | PAN | 0.82 | Pixel aggregation + FPEM |
| 2020 | **ABCNet** | **0.69 / 0.78 (Full)** | **Bezier curves + BezierAlign** |
| 2020 | DBNet | 0.85 | Differentiable binarization |
| 2021 | DBNet++ | 0.87 | Adaptive scale fusion |
| 2022 | TESTR / SwinTextSpotter | ~0.88 | Transformer-based |

El salto **2017 -> 2018 (0.36 -> 0.78)** corresponde al cambio de representacion geometrica, no de capacidad de red. Esto confirma que **el bottleneck no era de modelado sino de formulacion del output**. El refinamiento posterior (0.78 -> 0.88) viene de mejorar backbone, loss y multi-scale -> incrementos marginales pero consistentes.

A 2022 los F-scores SOTA rondan 0.87-0.89. La pendiente se aplana, lo que sugiere que el dataset (1555 imagenes) empieza a ser insuficiente para discriminar arquitecturas modernas -> escenario familiar de saturacion, como ocurrio con ICDAR'13.

## Limitaciones

Los autores son honestos sobre lo que Total-Text no resuelve:

1. **Escala**: 1555 imagenes es pequeno para deep learning moderno. Pre-training en COCO-Text o synthetic (SynthText 800k, MJSynth 9M) sigue siendo necesario.
2. **Idioma**: solo ingles. Chino, arabe, devanagari, cirilico fuera -> motivara MLT 2017/2019.
3. **Curated, no incidental**: imagenes seleccionadas con curved text en mente. No representa la distribucion real de un robot movil o smartphone donde la mayoria de frames no tienen curvas.
4. **Granularidad word vs. line**: la decision de anotar a nivel de palabra ayuda a recognition pero dificulta evaluar agrupamiento layout-level (gap que CTW-1500 llena).
5. **Errores de anotacion manual**: corpus anotado por los autores + 3 miembros del laboratorio, con cross-checking, pero hay ruido inevitable -> particularmente en decision de `do not care`.
6. **Sin end-to-end estandarizado en el paper original**: solo se reporta detection. El protocolo end-to-end se consolido despues, en trabajos que reutilizaron el dataset.

## Datasets sucesores

Total-Text inaugura una familia de datasets que extienden distintas dimensiones del problema:

- **CTW-1500** (Yuliang Liu et al. ACMM 2017): 1500 imagenes con poligono de **14 puntos a nivel de linea**. Complemento natural a Total-Text (word-level) -> el estandar de facto es reportar resultados en ambos.
- **MLT 2017 / 2019** (Nayef et al.): multilingue con 9 scripts (latino, chino, arabe, hebreo, etc.).
- **ArT** (Ch'ng et al. ICDAR 2019, mismos autores): 10 166 imagenes, fusion de Total-Text + CTW-1500 + nuevas imagenes -> la evolucion directa a gran escala.
- **LSVT** (Sun et al. 2019): 50 000 imagenes street view con weakly + fully annotated splits.
- **ReCTS** (2019): texto chino en signage de restaurantes.
- **HierText** (Long et al. CVPR 2022): jerarquia paragraph -> line -> word, tambien curvada.

## Conexion con la clase 21

Total-Text es el dataset **sobre el que ABCNet reporta sus numeros headline** en la slide "Quantitative results on Total-Text" del PDF. Tres conexiones explicitas con el material:

1. **Metrica directa**: ABCNet single-scale F=67.1, ABCNet con synthetic pre-training F=69.5 (None) / 78.4 (Full). La comparacion entre Mask TextSpotter (52.9 / 71.8) y ABCNet (69.5 / 78.4) ocurre sobre este benchmark.
2. **Justificacion geometrica de Bezier**: la decision de ABCNet de representar curvas mediante dos polinomios de Bezier de orden 3 es una respuesta directa al formato de Total-Text -> los 2N puntos del poligono se interpolan a 8+8=16 parametros Bezier, con error de reconstruccion <1 pixel en >99 % de los casos. Esa compatibilidad geometrica es lo que hace posible BezierAlign y, por extension, el end-to-end real-time (~17 fps).
3. **Continuidad pedagogica**: en la trayectoria detection objetos -> detection texto -> recognition texto, Total-Text es el momento en que el dominio texto deja de ser una extension trivial de detection generica y se convierte en un problema con representacion geometrica propia. Sin Total-Text no existen TextSnake, Mask TextSpotter ni ABCNet.

## Lecciones transferibles

El patron de Total-Text -ir un nivel de geometria mas libre que el estandar de la epoca- es transferible:

- **Instance segmentation** (Mask R-CNN): pasar de bbox a mascara sigue la misma logica.
- **Pose estimation**: bbox humano -> 17 keypoints (COCO) -> 3D SMPL reproduce el mismo movimiento.
- **Medical imaging**: anotar mascaras poligonales en tumores con bordes irregulares en lugar de bbox.
- **MDM / patient matching**: anotar el **tipo de match** (exact name + DOB, exact name + addr, soundex + DOB, etc.) en lugar de match binario habilita modelos mas finos.

En todos los casos, el move ganador es **enriquecer el espacio de output del groundtruth** antes que escalar la red. Total-Text es un ejemplo limpio porque el coste de anotacion es lineal en N vertices y el beneficio en F-score es supra-lineal (0.36 -> 0.78 en un ano).

## Cierre

Total-Text es un paper relativamente corto (10 paginas), con un solo baseline DeconvNet y sin novedad arquitectural propia. Su valor esta enteramente en haber identificado correctamente el **gap geometrico** y haber producido el groundtruth que permite cerrarlo. La leccion de metodo es clara: en problemas donde la geometria del output importa, **la anotacion es la palanca de avance**, no la arquitectura. Cinco anos despues el ecosistema irregular scene text -desde TextSnake hasta DBNet++ pasando por ABCNet- descansa sobre la decision de Ch'ng y Chan de anotar 9330 instancias con N vertices variables en lugar de 4.

## Notas y enlaces

- **Fundamentos relacionados**:
  - [Scene Text Recognition](/fundamentos/scene-text-recognition)
  - [Deteccion de objetos](/fundamentos/deteccion-de-objetos)
- **Papers conectados**:
  - [ABCNet (Liu CVPR 2020)](/papers/abcnet-liu-2020) -> usa Total-Text como benchmark principal y motiva Bezier curves desde su formato de anotacion.
  - [Text Recognition in the Wild (Chen 2020)](/papers/text-recognition-wild-chen-2020) -> survey que cataloga Total-Text como benchmark dominante de irregular text post-2018.
- **Clase**: [Clase 21 — Reconocimiento de texto en escenas](/clases/clase-21)
- **Repositorio oficial**: [`cs-chan/Total-Text-Dataset`](https://github.com/cs-chan/Total-Text-Dataset) (licencia academica, >700 estrellas, mantenido a 2026).
