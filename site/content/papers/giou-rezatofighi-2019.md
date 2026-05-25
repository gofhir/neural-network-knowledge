---
title: "Generalized IoU (GIoU)"
weight: 102
math: true
---

{{< paper-card
    title="Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
    authors="Rezatofighi, Tsoi, Gwak, Sadeghian, Reid, Savarese"
    year="2019"
    venue="CVPR 2019"
    pdf="/papers/giou-rezatofighi-2019.pdf"
    arxiv="1902.09630" >}}
Propone una **versión diferenciable y acotada en $[-1, 1]$ de IoU** que se comporta como métrica matemática (cumple la desigualdad triangular como $1-\text{IoU}$) y, sobre todo, **tiene gradiente útil incluso cuando las cajas no se solapan**. Arregla el *gradient mismatch* entre la pérdida $\ell_2$/Smooth $L_1$ que se optimiza y la métrica IoU con la que se evalúa, y se convierte en la base de toda la familia DIoU/CIoU/EIoU adoptada hoy por YOLOv4+, RTMDet, DETR y los detectores anchor-free.
{{< /paper-card >}}

---

## El problema

Los detectores 2D modernos (Faster R-CNN, Mask R-CNN, YOLOv3, SSD) entrenan la cabeza de regresión con **pérdidas $\ell_p$ sobre coordenadas** (Smooth $L_1$ sobre $(x_c, y_c, \log w, \log h)$, MSE sobre $(x, y, w, h)$), pero se evalúan con **IoU**. Esa desconexión genera tres patologías concretas:

1. **Gradient mismatch.** La figura 1 del paper muestra configuraciones con $\|\cdot\|_2$ idéntica pero IoU radicalmente distintos: tres ejemplos con $\ell_2 = 8.41$ producen $\text{IoU} \in \{0.26, 0.49, 0.65\}$. La pérdida $\ell_p$ trata cada coordenada como variable independiente y **no codifica la estructura conjunta** del bounding box.

2. **Plateau de IoU directo.** Una solución obvia es usar $\mathcal{L}_{IoU} = 1 - \text{IoU}$ directamente (UnitBox, Yu 2016). Funciona en régimen *overlapping*, pero **falla cuando $|A \cap B| = 0$**: la pérdida se satura en 1 para *cualquier* configuración no solapada, $\nabla \text{IoU} = 0$, y la red no recibe señal de hacia dónde mover la predicción. Es justo el régimen temprano del entrenamiento, donde más se necesita el gradiente.

3. **Falta de invariancia a escala.** IoU es invariante: dos cajas que se solapan al 80% tienen IoU = 0.8 sin importar la resolución. $\ell_2$ no lo es: el error en píxeles crece con la escala, sesgando el entrenamiento hacia objetos grandes. Faster R-CNN compensa con **log-space + normalización por anchor**, un parche artesanal.

Los workarounds existentes (anchor boxes densas, parametrización log, Smooth $L_1$) atacan síntomas. La solución limpia es **optimizar directamente la métrica**.

---

## Definición de GIoU

Sean $A, B$ dos formas convexas y $C$ la **menor caja convexa que las contiene** (*smallest enclosing convex object*). Entonces:

$$
\boxed{\text{GIoU}(A, B) = \underbrace{\frac{|A \cap B|}{|A \cup B|}}_{\text{IoU}} \;-\; \frac{|C \setminus (A \cup B)|}{|C|}}
$$

El segundo término penaliza el **espacio vacío dentro de la envolvente** que no cubre ninguna de las dos cajas. Cuanto más separadas, mayor el área de $C$ relativa a $|A \cup B|$, mayor la penalización.

Para *axis-aligned* 2D:

$$
\begin{aligned}
x_1^c &= \min(x_1^p, x_1^g), & x_2^c &= \max(x_2^p, x_2^g), \\
y_1^c &= \min(y_1^p, y_1^g), & y_2^c &= \max(y_2^p, y_2^g),
\end{aligned}
$$

con $A^c = (x_2^c - x_1^c)(y_2^c - y_1^c)$.

### Propiedades

| Propiedad | GIoU | IoU | $\ell_2$ |
|-----------|------|-----|----------|
| Rango | $[-1, 1]$ | $[0, 1]$ | $[0, \infty)$ |
| Invariante a escala | sí | sí | **no** |
| Métrica matemática ($1-\cdot$) | sí (Kosub 2016) | sí | sí |
| Gradiente sin overlap | **sí** | no (plateau) | sí pero sin estructura |
| Codifica estructura geométrica | sí | sí | **no** |
| Igualdad con IoU | $A=B$ o $A\cup B = C$ | — | — |

Adicionalmente, $\text{GIoU}(A,B) \leq \text{IoU}(A,B)$ siempre (es una cota inferior), y $\text{GIoU} \to -1$ cuando $|A \cup B|/|C| \to 0$, es decir, cuando las cajas se alejan al infinito relativo a la envolvente.

### Algoritmo

```python
def giou(B_p, B_g):
    # 1. Asegurar x2 > x1, y2 > y1
    x1p, y1p = min(B_p[0], B_p[2]), min(B_p[1], B_p[3])
    x2p, y2p = max(B_p[0], B_p[2]), max(B_p[1], B_p[3])
    x1g, y1g, x2g, y2g = B_g

    A_p = (x2p - x1p) * (y2p - y1p)
    A_g = (x2g - x1g) * (y2g - y1g)

    # 2. Interseccion (con clipping)
    xi1, yi1 = max(x1p, x1g), max(y1p, y1g)
    xi2, yi2 = min(x2p, x2g), min(y2p, y2g)
    I = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

    # 3. Union + IoU
    U = A_p + A_g - I
    iou = I / U

    # 4. Caja envolvente C y penalizacion
    xc1, yc1 = min(x1p, x1g), min(y1p, y1g)
    xc2, yc2 = max(x2p, x2g), max(y2p, y2g)
    A_c = (xc2 - xc1) * (yc2 - yc1)

    return iou - (A_c - U) / A_c
```

Todos los operadores son diferenciables casi en todas partes; la rama del clipping introduce una discontinuidad en frontera de medida cero, que en la práctica los frameworks manejan con ReLU-like clipping.

---

## Como loss

$$
\mathcal{L}_{GIoU} = 1 - \text{GIoU} \in [0, 2].
$$

A diferencia de $\mathcal{L}_{IoU} \in [0, 1]$ que se satura en 1 sin overlap, $\mathcal{L}_{GIoU}$ sigue creciendo hasta 2 mientras las cajas se alejan.

### Gradiente en régimen sin overlap

Cuando $I = 0$:

$$
\mathcal{L}_{GIoU} = 2 - \frac{U}{A^c} = 2 - \frac{A^p + A^g}{A^c}.
$$

Como $A^g$ es fijo, minimizar la pérdida obliga a la red a:

- **Aumentar $A^p$** (agrandar la caja predicha hacia la GT), y/o
- **Reducir $A^c$** (mover la predicción para que la envolvente se encoja).

Ambas dinámicas empujan $B^p$ hacia $B^g$ con **gradiente no nulo en todo el espacio** de configuraciones no solapadas. Es la diferencia conceptual con $\mathcal{L}_{IoU}$.

### Estabilidad numérica

El paper demuestra que la pérdida está bien definida sobre todo $\mathbb{R}^4$ para $B^p$:

- $A^g > 0$ por definición (GT no degenerada).
- El paso de "asegurar $x_2 > x_1$" fuerza $A^p \geq 0$.
- El clipping de la intersección garantiza $I \geq 0$.
- $U \geq I$ siempre, denominador de IoU $> 0$.
- $A^c \geq A^g > 0$, denominador de la penalización acotado.
- $A^c \geq U$, por lo que $|C \setminus (A \cup B)|/|C| \in [0, 1)$.

Resultado: $\mathcal{L}_{GIoU} \in [0, 2]$ para cualquier predicción, **sin overflow, divisiones por cero ni NaN**. Esto es valioso en práctica porque otras pérdidas geométricas (IoU rotada, Distance-IoU sin clip) tienen casos patológicos que las hacen explotar al inicio del entrenamiento.

### Ejemplo numérico

GT $= (0, 0, 10, 10)$, área 100. Dos predicciones con la misma $\ell_2$:

| Predicción | Caja | $\ell_2$ | IoU | GIoU |
|-----------|------|----------|-----|------|
| A (alargada en $y$) | $(0, 0, 10, 20)$ | 10 | 0.5 | 0.5 |
| B (desplazada en $x$) | $(10, 0, 20, 10)$ | 10 | 0 | **0** |

Misma $\ell_2$, predicción A claramente mejor (overlap parcial) que B (adyacente, sin overlap). GIoU las distingue (0.5 vs 0); $\ell_2$ no.

---

## Resultados

Los autores reemplazan la pérdida nativa por $\mathcal{L}_{IoU}$ o $\mathcal{L}_{GIoU}$ sin cambiar arquitectura, anchors, learning rate ni schedule.

| Detector | Backbone | Baseline | Δ AP vs baseline | Δ AP75 vs baseline |
|----------|----------|----------|------------------|--------------------|
| YOLOv3 (VOC 07)  | DarkNet-608 | MSE | **+3.45%** | **+5.56%** |
| YOLOv3 (COCO val) | DarkNet-608 | MSE | **+6.69%** | +9.12% |
| YOLOv3 (COCO test) | DarkNet-608 | MSE | +5.71% | +8.01% |
| Faster R-CNN (VOC 07) | ResNet-50 | Smooth $L_1$ | +5.95% | **+12.85%** |
| Faster R-CNN (COCO val) | ResNet-50 | Smooth $L_1$ | +2.50% | +2.05% |
| Mask R-CNN (COCO val) | ResNet-50 | Smooth $L_1$ | +2.73% | +2.02% |

Patrón consistente: **GIoU > IoU > $\ell_p$**. Dos observaciones del paper:

1. La brecha es mayor en **AP75** que en AP a 0.5: la ventaja se amplifica cuando el umbral es estricto y la calidad de localización importa más.
2. La ganancia en Faster/Mask R-CNN es más modesta porque la **densidad de anchors** ya filtra predicciones sin overlap (RPN selecciona positivos con $\text{IoU} \geq 0.7$), por lo que el plateau de IoU casi no se activa. En YOLOv3, sin esa pre-selección densa, GIoU brilla más.

La figura 3 del paper muestra que el average IoU sobre cajas predichas converge **más rápido y más alto** con GIoU loss que con MSE, aunque la pérdida de clasificación queda ligeramente inferior por falta de re-tuneo del peso relativo entre cabezas.

---

## Variantes posteriores

GIoU es el primer eslabón de una familia que añade penalizaciones geométricas adicionales:

| Variante | Año | Término extra | Adopción típica |
|----------|-----|---------------|-----------------|
| **GIoU** | 2019 | Espacio vacío en envolvente | YOLOv5 (opcional), DETR, FCOS |
| **DIoU** | 2020 | $+ \rho^2(b^p, b^g)/c^2$ — distancia normalizada entre centros | DETR variantes |
| **CIoU** | 2020 | DIoU $+ \alpha v$ — distancia + aspect ratio | **YOLOv4, YOLOv5, YOLOv7, YOLOv8** |
| **EIoU** | 2022 | Desacopla $w, h$ del aspect ratio de CIoU | Detectores experimentales |
| **SIoU** | 2022 | Componente direccional (angle cost) | YOLO custom |
| **WIoU** | 2023 | Pesos dinámicos estilo focal | Detectores robustos a outliers |
| **$\alpha$-IoU** | 2021 | Exponente $\alpha$ sobre IoU | Control fino del gradiente |

Vistas como familia:

$$
\mathcal{L} = 1 - \text{IoU} + \mathcal{R}(B^p, B^g)
$$

donde $\mathcal{R}$ es un regularizador geométrico. GIoU usa $\mathcal{R} = |C \setminus (A \cup B)|/|C|$; DIoU/CIoU añaden distancia entre centros y aspect ratio. **CIoU es la default en YOLOv4 (Bochkovskiy 2020)** y se propaga al resto de la familia YOLO; **RTMDet (2022)** reporta ~0.5 AP extra al pasar de GIoU a CIoU.

---

## Limitaciones

1. **Convergencia lenta sin overlap.** La señal de gradiente existe (a diferencia de IoU puro) pero se aplana cuando $A^c$ es muy grande respecto a $A \cup B$. DIoU lo resuelve con el término explícito de distancia entre centros.

2. **Colapso a IoU cuando $A \subset B$ o viceversa.** Si una caja contiene a la otra, $A \cup B = C$ y $|C \setminus (A \cup B)| = 0$, por lo que $\text{GIoU} = \text{IoU}$ y el término extra no aporta nada. Este modo de falla específico motivó DIoU.

3. **No considera distancia entre centros.** GIoU codifica el "espacio vacío en la envolvente" pero no penaliza explícitamente que los centros estén descentrados. Dos predicciones con misma GIoU pueden tener centros muy distintos respecto al GT.

4. **Aspect ratio no es un término separado.** Predicción centrada en GT pero alargada (GT cuadrada, pred rectangular) puede tener GIoU razonable solo por el solapamiento. Para texto escena (aspect ratios extremos) esto importa.

5. **Tuneo del peso vs clasificación.** $\mathcal{L}_{GIoU} \in [0, 2]$ acotada cambia el balance gradiente respecto a MSE no acotado. Requiere re-tunear el peso de la rama de regresión; no es plug-and-play perfecto.

6. **Cajas rotadas / polígonos.** El paper restringe la solución analítica a *axis-aligned*. Para *rotated boxes* (DOTA, texto inclinado) o polígonos arbitrarios (texto curvo), el cómputo de $|C|$ requiere convex hull con aproximaciones suaves (Rot-GIoU, KFIoU, ProbIoU).

---

## Por qué importa hoy

GIoU transformó "loss de regresión" de un detalle de implementación en un eje de mejora explícito. Tres consecuencias prácticas:

- **Estándar en frameworks.** `torchvision.ops.generalized_box_iou_loss` y `mmdet.models.losses.GIoULoss` exponen la implementación drop-in. En MMDetection y Detectron2 los configs estándar incluyen GIoU/CIoU como opciones nativas.
- **Pieza estructural en anchor-free.** Detectores sin anchors (FCOS, CenterNet, ATSS, GFL) dependen aún más de pérdidas IoU-based porque no tienen el regularizador implícito de las anchor boxes. Aquí GIoU/CIoU no es marginal: es estructural.
- **Base para detección de texto escena.** Datasets como ICDAR 2015, Total-Text, CTW1500 tienen instancias con escalas y aspect ratios extremos. La invariancia a escala de GIoU/CIoU y su gradiente fuera de overlap explican por qué los detectores de texto post-2020 (EAST, PSENet, DBNet, FCENet) reportan mejoras especialmente en *small-text recall*. **ABCNet (Liu 2020)** usa IoU loss en su rama de regresión geométrica sobre puntos de control Bézier; reemplazarla por CIoU es la mejora natural.

---

## Conexión con clase 21

La Clase 21 cubre **Scene Text Detection y Recognition**, donde la geometría del bounding region (caja, polígono o curva Bézier) es central. Tres conexiones directas:

- **FCOS (Tian et al. ICCV 2019).** Detector anchor-free que usa IoU loss en la regresión de $(l, t, r, b)$ desde cada píxel. Las re-implementaciones modernas la sustituyen por GIoU/CIoU. Su rama de *centerness* cumple un rol complementario al término de centros de DIoU: penaliza predicciones lejos del centro del objeto.
- **ABCNet (Liu et al. CVPR 2020).** Parametriza texto curvo con curvas de Bézier; usa IoU loss sobre el bounding region de los puntos de control. Es candidato directo a beneficiarse de CIoU porque (a) las predicciones iniciales rara vez solapan la GT en texto inclinado y (b) los aspect ratios extremos del texto castigan duro a $\ell_p$.
- **FPN + GIoU.** En detectores piramidales (Faster R-CNN, FCOS, Mask R-CNN) las ganancias de GIoU se ven más fuerte en niveles altos de la pirámide (objetos grandes), donde las anchors estándar quedan más desalineadas. Cambio de Smooth $L_1$ a GIoU es la mejora más universal de regresión en estos pipelines.

La cita típica en estos trabajos es *"we use IoU/GIoU loss for bounding box regression following [Rezatofighi 2019]"*; este paper es el "por qué" detrás de la cita.

---

## Notas y enlaces

**Fundamentos transversales:**

- [Detección de objetos](/fundamentos/deteccion-de-objetos)
- [Detección anchor-free](/fundamentos/anchor-free-detection)
- [Funciones de pérdida](/fundamentos/funciones-perdida)

**Papers relacionados:**

- [FCOS (Tian 2019)](/papers/fcos-tian-2019) — anchor-free, usa IoU loss en regresión
- [ABCNet (Liu 2020)](/papers/abcnet-liu-2020) — texto curvo Bézier, candidato directo a CIoU
- [FPN (Lin 2017)](/papers/fpn-lin-2017) — pirámide multi-escala, donde GIoU se inserta en la cabeza de regresión
- [Faster R-CNN (Ren 2015)](/papers/faster-rcnn-ren-2015) — Smooth $L_1$ original, baseline del experimento principal

**Clase:**

- [Clase 21 — Scene Text Detection y Recognition](/clases/clase-21)

**Linaje histórico:**

```
2016: UnitBox (IoU loss para face detection)
  └─ 2019: GIoU (resuelve el plateau con enclosing penalty)
       └─ 2020: DIoU/CIoU (anade distancia entre centros + aspect ratio)
            └─ 2022+: EIoU, SIoU, WIoU, alpha-IoU (refinamientos)
```

**Regla práctica:** GIoU para validar la dirección (un solo término extra, sin hiperparámetros), CIoU para producción (mejor convergencia, default en YOLOv4+).
