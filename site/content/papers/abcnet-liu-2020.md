---
title: "ABCNet (Adaptive Bezier-Curve Network)"
weight: 100
math: true
---

{{< paper-card
    title="ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network"
    authors="Liu, Chen, Bian, Shen, Liu"
    year="2020"
    venue="CVPR 2020 (oral)"
    pdf="/papers/abcnet-liu-2020.pdf"
    arxiv="2002.10200" >}}
Primer **scene text spotter end-to-end** que representa texto irregular mediante **curvas Bezier cubicas parametricas** (8 puntos de control = 16 escalares) en lugar de poligonos densos o mascaras pixel a pixel. Introduce **BezierAlign**, un sampler diferenciable que rectifica el texto curvo a lo largo de la curva antes del recognizer. Sobre un detector anchor-free estilo FCOS + ResNet-50 + FPN alcanza **F-measure 69.5 / 78.4 en Total-Text** y la variante rapida corre a **22.8 FPS**, dominando la frontera Pareto velocidad-precision.
{{< /paper-card >}}

---

## El problema

Antes de 2020 el scene text spotting (deteccion + reconocimiento conjunto de texto en imagenes naturales) estaba fracturado en dos familias:

- **Box-regression**: TextBoxes, EAST, FOTS, TextDragon. Rapidos pero predicen bounding boxes rectangulares o cuadrilateros rotados, **incapaces de capturar texto curvo**. Un letrero "WELCOME" en arco lo recortan en un rectangulo enorme con mucho fondo, contaminando el recognizer.
- **Segmentation-based**: Mask TextSpotter, CharNet, PSENet. Producen mascaras a nivel de pixel; capturan cualquier forma, pero requieren **post-procesamiento no diferenciable** (component grouping, fitting polygonal), son sensibles a texto cercano que se "pega" en la mascara, y suelen exigir anotaciones character-level caras.

Los benchmarks **Total-Text (2017)** y **CTW1500 (2019)** forzaron a la comunidad a tratar texto curvo como first-class citizen. La anotacion oficial usaba **poligonos de 10-14 vertices** (28+ escalares correlados a regresar), lo que explica por que casi todos los metodos previos recurrian a segmentacion para evitar la explosion parametrica.

La pregunta que ABCNet responde: ¿existe una representacion parametrica compacta que (a) sea suficientemente expresiva para texto curvo del mundo real, (b) tenga bajo costo computacional y (c) permita una alineacion geometrica **diferenciable** con el recognizer? La respuesta es la curva Bezier cubica.

## Ideas principales

### Representacion Bezier cubica

Una curva Bezier de grado $n$ se define como combinacion convexa de $n+1$ puntos de control $b_i \in \mathbb{R}^2$ ponderados por los **polinomios de Bernstein**:

$$
c(t) = \sum_{i=0}^{n} b_i \, B_{i,n}(t), \quad B_{i,n}(t) = \binom{n}{i} t^i (1-t)^{n-i}, \quad t \in [0,1].
$$

Para la cubica ($n=3$):

$$
c(t) = (1-t)^3 b_0 + 3(1-t)^2 t \, b_1 + 3(1-t) t^2 \, b_2 + t^3 b_3.
$$

Propiedades que justifican la eleccion:

- **Endpoint interpolation**: $c(0)=b_0$ y $c(1)=b_3$ coinciden con inicio y fin del texto.
- **Convex hull**: $\sum_i B_{i,n}(t) = 1$, la curva queda dentro del casco convexo de los puntos de control (estabilidad numerica al regresar).
- **Invariancia afin**: si transformas los puntos de control, la curva se transforma con la misma matriz — clave para data augmentation consistente.
- **Suavidad $C^\infty$** sin esquinas espurias.

Para texto se usan **dos curvas cubicas** (top + bottom boundary) = **8 puntos de control = 16 coordenadas**. La cubica es el grado minimo que captura S-curves (un punto de inflexion), suficiente para letreros enrollados, circulos parciales y texto sobre objetos curvos como botellas o tazas.

| Representacion | Parametros | Capacidad | Costo |
| --- | --- | --- | --- |
| Horizontal bbox | 4 | Solo texto horizontal | Minimo |
| Cuadrilatero rotado | 8 | Inclinado, perspectiva leve | Bajo |
| Poligono denso | 28 (14 vertices) | Curvo, irregular | Alto, ruidoso |
| Mask binaria | $H \times W$ | Cualquier forma | Muy alto |
| **Bezier cubica** | **16** | **Curvo arbitrario** | **Bajo** |

### Generacion de ground truth Bezier

Total-Text trae poligonos con $m+1$ vertices por boundary. Para convertir a cubica, se resuelve un **least-squares lineal**:

$$
\mathbf{B} \cdot \mathbf{b} = \mathbf{p},
$$

donde $\mathbf{B}$ es la matriz de Bernstein evaluada en parametros $t_i$ obtenidos por **arc-length parametrization** (longitud acumulada normalizada a $[0,1]$), $\mathbf{b}$ son los puntos de control y $\mathbf{p}$ los vertices del poligono. Se fijan $b_0$ y $b_3$ como extremos (endpoint interpolation) y solo se resuelven $b_1, b_2$. Una propiedad sutil: el ground truth Bezier resulta a menudo **mas suave que la anotacion poligonal humana**, porque la cubica ajustada promedia el ruido del trazado manual.

### BezierAlign — el sampler diferenciable

RoIAlign (Mask R-CNN) muestrea features en un rectangulo axis-aligned; RoIRotate (FOTS) lo extiende a cuadrilateros rotados. **Ninguno funciona para texto curvo**.

BezierAlign muestrea features sobre la region delimitada por dos curvas Bezier. Para cada pixel de salida $(g_{iw}, g_{ih})$ del output map $h_{out} \times w_{out}$:

1. Calcula el parametro horizontal $t = g_{iw} / w_{out}$.
2. Evalua los puntos en ambas curvas: $tp = c_{top}(t)$ y $bp = c_{bottom}(t)$.
3. Interpola linealmente entre top y bottom segun la coordenada vertical: $op = bp \cdot (g_{ih}/h_{out}) + tp \cdot (1 - g_{ih}/h_{out})$.
4. Aplica **bilinear interpolation** en $op$ sobre el feature map original.

Equivalente en display:

$$
op = bp \cdot \frac{g_{ih}}{h_{out}} + tp \cdot \left(1 - \frac{g_{ih}}{h_{out}}\right).
$$

El resultado: un output map rectangular donde la dimension horizontal recorre el texto a lo largo del trazo y la vertical recorre su grosor. El recognizer ve siempre **texto rectificado**, sin importar cuan curvo era el original.

### Detector anchor-free estilo FCOS

ABCNet adopta el paradigma **FCOS** (Tian et al. 2019): prediccion densa pixel a pixel sobre los 5 niveles FPN, sin anchor boxes. Cada pixel predice:

- 1 canal de clasificacion (sigmoid + focal loss).
- 1 canal de center-ness.
- 4 canales de bbox regression $(l, t, r, b)$ con IoU loss.
- **16 canales adicionales** para los 8 puntos de control Bezier, predichos como offsets relativos al $(x_{min}, y_{min})$ del cuadrilatero envolvente:

$$
\Delta_x = b_{ix} - x_{min}, \quad \Delta_y = b_{iy} - y_{min}.
$$

Predecir offsets relativos da invariancia traslacional y robustez cuando $b_1, b_2$ caen ligeramente fuera del crop por la geometria del arco.

## Arquitectura

- **Backbone**: ResNet-50.
- **Neck**: FPN con 5 niveles **P3-P7** (strides 1/8 a 1/128).
- **Detection head** (anchor-free, sobre P3-P7): clasificacion + center-ness + 4-ch bbox + **16-ch control points**.
- **BezierAlign**: muestrea features a lo largo de la curva, produce un mini-mapa rectificado de **7 × 32**.
- **Recognition head** (sobre niveles 1/4, 1/8, 1/16): 6 convs (4 con stride 1, 2 con stride (2,1) para colapsar altura) + average pool vertical + **BLSTM** + FC a 97 clases + **CTC loss**.

Loss total:

$$
\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{center} + \mathcal{L}_{bbox} + \mathcal{L}_{bezier} + \mathcal{L}_{rec},
$$

donde $\mathcal{L}_{bezier}$ es Smooth L1 sobre los 16 offsets y $\mathcal{L}_{rec}$ es CTC sobre la secuencia. El sistema es **completamente diferenciable**: el gradiente fluye desde la CTC loss del recognizer, vuelve por BezierAlign (interpolacion bilineal en una curva parametrica), llega a los puntos de control regresados y de ahi al backbone.

## Resultados experimentales

### Total-Text (lexicon-free / full lexicon)

| Metodo | Backbone | F-measure (None / Full) | FPS |
| --- | --- | --- | --- |
| TextBoxes | ResNet-50-FPN | 36.3 / 48.9 | 1.4 |
| Mask TextSpotter'18 | ResNet-50-FPN | 52.9 / 71.8 | 4.8 |
| TextNet | ResNet-50-SAM | 54.0 / — | 2.7 |
| Mask TextSpotter'19 | ResNet-50-FPN | 65.3 / 77.4 | 2.0 |
| CharNet | Hourglass57 | 66.2 / — | 1.2 |
| Qin et al. 2019 | ResNet-50-MSF | 67.8 / — | 4.8 |
| **ABCNet-F** (fast) | ResNet-50-FPN | **61.9 / 74.1** | **22.8** |
| **ABCNet** | ResNet-50-FPN | 64.2 / 75.7 | 17.9 |
| **ABCNet-MS** (multi-scale) | ResNet-50-FPN | **69.5 / 78.4** | 6.9 |

Lectura clave: ABCNet-MS **supera a Mask TextSpotter'19** (la SOTA end-to-end anterior) en F-measure, y ABCNet-F es **>11× mas rapido** con caida pequena. Ademas usa **menos datos sinteticos** (150k vs 800k del SynText estandar): la calidad del SynText curvo importa mas que la cantidad.

La metrica end-to-end es severa: una prediccion cuenta como verdadero positivo solo si (a) su Bezier tiene IoU ≥ 0.5 con el GT poligonal y (b) la string reconocida coincide exactamente (case-sensitive) con la palabra GT. El protocolo **None** (lexicon-free) obliga al recognizer a producir la palabra correcta sin ayuda, mientras **Full** (strong lexicon) permite acotar las predicciones a un diccionario del test set, rescatando errores tipograficos pequenos (por ejemplo "PEACHIREE" → "PEACHTREE"). Pasar de None a Full suma tipicamente 8-12 puntos.

### Ablation BezierAlign

Mismo modelo, variando solo el sampler:

| Sampling | F-measure (%) |
| --- | --- |
| Horizontal sampling | 38.4 |
| Quadrilateral sampling | 44.7 |
| **BezierAlign** | **61.9** |

**+17.2 puntos** al pasar de cuadrilatero a Bezier. Es la evidencia mas fuerte del paper: la rectificacion geometrica precisa importa tanto como el recognizer.

### Costo del Bezier

| Configuracion | FPS |
| --- | --- |
| Sin Bezier curve detection | 22.8 |
| **Con Bezier curve detection** | **22.5** |

**0.3 FPS = 1.3% de overhead**. Los 16 canales extra por pixel son virtualmente gratis cuando el backbone ya esta computado.

### Sampling points

El optimo es $(n_h, n_w) = (7, 32)$ — **bajo y rectangular**, reflejando la forma natural de una palabra latina. Aumentar la resolucion no ayuda y hurts performance: $(28, 128)$ baja a 53.4 F-measure por overfitting al ruido del feature map.

### CTW1500 (texto curvo + chino)

| Metodo | F-measure (None / Strong Full) |
| --- | --- |
| TextDragon | 39.7 / 72.4 |
| **ABCNet** | **45.2 / 74.1** |

Margen de **+5.5 puntos lexicon-free**. La metrica word-accuracy a line-level es severa: un solo caracter mal en una linea larga da score 0 para esa linea.

## Limitaciones reconocibles

1. **Asume lineas de texto con orientacion dominante**: una cubica top + cubica bottom presupone direccion de lectura horizontal-ish. **Texto vertical chino o japones real** requiere re-parametrizar; ABCNet trata el chino de CTW1500 como clase "unseen".
2. **Oclusiones severas**: si una palabra esta parcialmente tapada, $b_1, b_2$ pierden senal y la regresion Bezier puede caer en un minimo local. No hay experimentos con datasets occluded.
3. **Dependencia de la sintesis**: los 150k SynText curvos son criticos. Sin ellos el modelo no aprende curvas; la receta de Gupta et al. modificada es ad-hoc.
4. **FPS dependiente del input size**: ABCNet-F usa short side = 600. En letreros lejanos con texto pequeno (driving datasets) puede no resolver bien el texto, y escalar reduce FPS dramaticamente.
5. **Recognizer cerrado a 97 clases**: cubre ingles latino + digitos pero no diacriticos (acentos en español, ñ, ü) ni alfabetos no latinos.
6. **Grado fijo ($n=3$)**: texto con doble inflexion (raro en logos artisticos) no se modela bien con una cubica.
7. **Error acumulado del GT Bezier**: la conversion poligono→Bezier introduce error de fit que se propaga al entrenamiento.

## Por que importa hoy

ABCNet abrio una linea fertil de **Bezier-based methods** en scene text spotting:

- **ABCNet v2** (TPAMI 2022): BiFPN, attention adaptativa en el recognizer, soft-label character-aware. Sube a ~76 F-measure en Total-Text manteniendo tiempo real.
- **TESTR** (CVPR 2022): Transformer-based, usa los Bezier control points como queries en un detector DETR-like.
- **SPTS** (2022): "Single-Point Text Spotting" lleva la logica al extremo, reemplazando 8 puntos por **un solo punto + secuencia autoregresiva**.
- **AdelaiDet** (libreria oficial Adelaide): ABCNet consolido el ecosistema anchor-free para tareas densas junto a FCOS, BlendMask, CondInst.

El concepto general — **regresar control points + alinear geometricamente con un sampler diferenciable** — entra al mainstream de cualquier tarea donde la geometria no es axis-aligned: lane markings en driving, contornos medicos, segmentacion de hojas en agronomia. La leccion transferible: antes de complicar el modelo, **busca una representacion mas compacta del problema**. Bezier cubicas son una eleccion paradigmatica porque sustentan la teoria clasica de Bernstein (universalidad de aproximacion) y la ingenieria moderna (parametros pocos, gradiente limpio).

Tambien conecta con vision foundation models: Qwen-VL, LLaVA y GPT-4V leen texto curvo via atencion implicita sin parametrizacion explicita de la geometria. ABCNet representa la alternativa **explicita y geometricamente interpretable**: cuando necesitas latencia bounded, bounding outputs auditables y control fino, una representacion parametrica sigue siendo preferible al black-box attention.

## Conexion con la clase y el lab

La **clase 21** del modulo de Vision Computacional cubre OCR y scene text recognition. El pipeline canonico se ensena en dos etapas — text detection (CRAFT, EAST, DBNet) seguido de text recognition (CRNN, TPS-CRNN, ABINet, PARSeq) — y ABCNet entra como el primer ejemplo de **integracion end-to-end** donde ambas etapas comparten backbone y se entrenan jointly. Esto motiva tres preguntas pedagogicas: ¿cuando conviene two-stage vs end-to-end?, ¿como se disena un sampler diferenciable que respete la geometria del objeto?, ¿que representacion de bounding box elegir segun la naturaleza del dominio?

El **[lab 21](/laboratorios/lab-21)** trabaja **directamente con ABCNet**: lo instala sobre AdelaiDet/Detectron2, carga el checkpoint `attn_R_50` preentrenado en Total-Text, y diseca su salida bit a bit — los `instances.beziers` `(N,16)` (los 8 puntos de control por palabra) y los `instances.recs` `(N,25)` (indices del charset, donde el indice 95 = 口 es el placeholder de "desconocido" que explica el warning CJK del demo). Luego construye dos aplicaciones que reaprovechan el modelo sin reentrenar: OCR de marcas sobre [Freiburg Groceries](/papers/freiburg-groceries-jund-2016) (transfer *zero-shot* a aleman, porque ABCNet aprendio glifos latinos, no idiomas) y mineria geoespacial de texto sobre [Google Street View](/papers/street-view-geolocalization-zamir-2010). Pregunta de discusion natural: si tu dominio tiene texto en envases curvos (botellas, latas, taxis) o senaletica con arcos — incluso etiquetas farmaceuticas en cilindros, wristbands de pacientes en healthcare — ¿es preferible entrenar ABCNet desde cero, fine-tunear, o usar un Vision-Language Model? El criterio que ABCNet ensena: **la representacion geometrica debe ajustarse a la distribucion empirica del texto en tu dominio**. No uses cubicas si tu texto es recto; no uses cuadrilateros si tu texto se curva.

## Notas y enlaces

- **Fundamentos relacionados**:
  - [Scene text recognition](/fundamentos/scene-text-recognition/)
  - [Curvas de Bezier](/fundamentos/bezier-curves/)
  - [Deteccion anchor-free](/fundamentos/anchor-free-detection/)
  - [Deteccion de objetos](/fundamentos/deteccion-de-objetos/)
- **Papers relacionados**:
  - [FCOS — Tian et al. 2019](/papers/fcos-tian-2019/) — detector anchor-free per-pixel base de ABCNet
  - [FPN — Lin et al. 2017](/papers/fpn-lin-2017/) — neck multi-escala
  - [GIoU — Rezatofighi et al. 2019](/papers/giou-rezatofighi-2019/) — IoU loss diferenciable usada en bbox regression
  - [CRNN — Shi et al. 2017](/papers/crnn-shi-2017/) — base de la recognition head ligera (CNN + BLSTM + CTC)
  - [Total-Text — Ch'ng et al. 2017](/papers/total-text-chng-2017/) — primer benchmark de texto curvo en escenas naturales
  - [ResNet — He et al. 2015](/papers/resnet-he-2015/) — backbone
  - [Bahdanau attention — 2015](/papers/bahdanau-attention-2015/) — comparado con CTC como alternativa para el recognizer
- **Clase**: [Clase 21 — OCR y scene text recognition](/clases/clase-21/)
- **Lab**: [Lab 21 — scene text recognition aplicado](/laboratorios/lab-21/)
