---
title: "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
authors:
  - Hamid Rezatofighi
  - Nathan Tsoi
  - JunYoung Gwak
  - Amir Sadeghian
  - Ian Reid
  - Silvio Savarese
year: 2019
venue: CVPR 2019
slug: giou-rezatofighi-2019
arxiv: "1902.09630"
tags:
  - object-detection
  - bounding-box-regression
  - loss-functions
  - iou
  - metric-learning
---

## Resumen ejecutivo

Rezatofighi y colegas atacan un problema central de los detectores 2D modernos: la métrica con la que se evalúa la calidad de una caja (Intersection over Union, IoU) **no es la misma función que se optimiza durante el entrenamiento**. Faster R-CNN usa Smooth $L_1$, YOLOv3 usa MSE sobre $(x_c, y_c, w, h)$. Esa desconexión genera dos patologías: (a) configuraciones con $\ell_2$ idénticos pueden tener IoU radicalmente distintos, y (b) cuando las cajas no se solapan, $\text{IoU} = 0$ y el gradiente se desvanece, de modo que la red no recibe señal de en qué dirección moverse. La propuesta es **Generalized IoU (GIoU)**, definida como $\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}$, donde $C$ es la menor caja envolvente. GIoU vive en $[-1, 1]$, es diferenciable en todos los casos, y se reduce a IoU cuando hay solapamiento perfecto. Como pérdida $\mathcal{L}_{GIoU} = 1 - \text{GIoU}$ mejora de forma consistente Faster R-CNN, Mask R-CNN y YOLOv3 en COCO y VOC. El paper inaugura toda una familia de "IoU-based losses" que culmina en DIoU, CIoU y se vuelve estándar desde YOLOv4 en adelante.

---

## 1. El problema: métrica vs loss

El detector ideal optimiza directamente la métrica con la que será evaluado. En detección de objetos eso significa optimizar IoU. Sin embargo, las arquitecturas dominantes (R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD) entrenan cabezales de regresión con pérdidas $\ell_n$ sobre representaciones paramétricas del bounding box.

### 1.1 Gradient mismatch

Dado un bounding box predicho $B^p = (x_1^p, y_1^p, x_2^p, y_2^p)$ y ground truth $B^g = (x_1^g, y_1^g, x_2^g, y_2^g)$, una pérdida $\ell_2$ típica es:

$$
\mathcal{L}_{\ell_2}(B^p, B^g) = \sum_{i \in \{1,2\}} (x_i^p - x_i^g)^2 + (y_i^p - y_i^g)^2.
$$

El problema es geométrico. La figura 1 del paper muestra tres configuraciones con **idéntica** $\|\cdot\|_2 = 8.41$ pero $\text{IoU} \in \{0.26, 0.49, 0.65\}$. La pérdida $\ell_2$ ve estas configuraciones como equivalentes; la métrica de evaluación las trata como dramáticamente diferentes. El mismo fenómeno ocurre con parametrización $(x_c, y_c, w, h)$: tres ejemplos con $\|\cdot\|_1 = 9.07$ producen $\text{IoU} \in \{0.27, 0.59, 0.66\}$.

La intuición de fondo es que $\ell_p$ trata cada coordenada como variable independiente, ignorando la estructura conjunta (un bounding box es un objeto 4-dimensional cuya semántica depende de la combinación de coordenadas, no de cada una por separado). Mover $x_1$ hacia adentro y $x_2$ hacia afuera por la misma cantidad mantiene el centro pero cambia el área; mover ambos por la misma cantidad traslada sin cambiar el área. $\ell_2$ no distingue estos casos.

### 1.2 Falta de invariancia a escala

IoU es invariante a escala: dos cajas que se solapan en 80% tienen IoU = 0.8 sin importar si están en una imagen 100×100 o 1000×1000. $\ell_2$ no: el error en píxeles crece linealmente con la escala. Esto es por lo que Faster R-CNN parametriza offsets en log-space y normaliza por el tamaño de la anchor, una corrección artesanal que GIoU vuelve innecesaria.

### 1.3 IoU como loss directo: el problema del plateau

Una respuesta obvia es usar $\mathcal{L}_{IoU} = 1 - \text{IoU}$ como pérdida (UnitBox, Yu 2016). Funciona cuando las cajas se solapan, pero falla cuando no: si $|A \cap B| = 0$, entonces $\text{IoU} = 0$ para **cualquier** configuración no solapada. Eso significa $\nabla_{B^p} \text{IoU} = 0$ en una región amplia del espacio de predicciones, y el optimizador no recibe señal de hacia dónde mover la predicción para empezar a solapar el ground truth.

Esto se llama el problema del **plateau de IoU**. En etapas tempranas del entrenamiento, donde las predicciones están lejos del target, es exactamente donde más se necesita gradiente útil, y donde IoU lo niega.

### 1.4 Por qué los workarounds artesanales no bastan

Los detectores modernos compensan estas patologías con varios mecanismos:

- **Anchor boxes** (Faster R-CNN, SSD, YOLOv2+): predefinen un grid denso de cajas iniciales con escalas y aspect ratios típicos, de modo que el regresor solo predice **offsets pequeños** respecto a la anchor. Esto evita el régimen "predicción muy lejos del GT" pero introduce hiperparámetros adicionales (cuántas anchors, qué escalas, qué aspect ratios) y aumenta el número de positivos/negativos a procesar.
- **Log-space parameterization:** Girshick (R-CNN, Fast R-CNN) predice $(\Delta x, \Delta y, \log \Delta w, \log \Delta h)$ relativo al anchor. El logaritmo regulariza variaciones grandes de escala y desacopla parcialmente $(x, y)$ de $(w, h)$. Pero la pérdida sigue siendo $\ell_2$ o Smooth $L_1$ sobre estos targets, y sigue siendo geométricamente ciega.
- **Smooth $L_1$** (Huber loss): combina $\ell_2$ cerca de cero con $\ell_1$ lejos, ofreciendo robustez a outliers. Mejora estabilidad numérica pero no resuelve el gradient mismatch fundamental.

Todos estos son **parches** alrededor de un problema cuya solución limpia es: optimiza directamente la métrica. Es ahí donde GIoU se inserta.

---

## 2. Definición formal de GIoU

### 2.1 Construcción

Sean $A, B \subseteq \mathbb{S} \in \mathbb{R}^n$ dos formas convexas arbitrarias. Sea $C$ la **menor forma convexa que encierra a ambas** (smallest enclosing convex object). Entonces:

$$
\boxed{\text{GIoU}(A, B) = \underbrace{\frac{|A \cap B|}{|A \cup B|}}_{\text{IoU}} \;-\; \frac{|C \setminus (A \cup B)|}{|C|}}
$$

El segundo término penaliza el **espacio vacío dentro de la cápsula envolvente** que no está cubierto por ninguna de las dos formas. Cuanto más lejos estén $A$ y $B$, más grande es $|C|$ relativo a $|A \cup B|$, y mayor la penalización.

Para axis-aligned 2D bounding boxes, $C$ es simplemente la caja axis-aligned más pequeña que contiene ambas:

$$
\begin{aligned}
x_1^c &= \min(x_1^p, x_1^g), & x_2^c &= \max(x_2^p, x_2^g), \\
y_1^c &= \min(y_1^p, y_1^g), & y_2^c &= \max(y_2^p, y_2^g).
\end{aligned}
$$

Con área $A^c = (x_2^c - x_1^c)(y_2^c - y_1^c)$.

### 2.2 Propiedades

1. **Métrica:** $\mathcal{L}_{GIoU} = 1 - \text{GIoU}$ cumple no-negatividad, identidad de indiscernibles, simetría y desigualdad triangular (referencia Kosub 2016).
2. **Invariancia a escala:** $\text{GIoU}(sA, sB) = \text{GIoU}(A, B)$ para todo factor $s > 0$.
3. **Cota inferior de IoU:** $\text{GIoU}(A, B) \leq \text{IoU}(A, B)$, con igualdad solo cuando $A = B$ o cuando $A \cup B = C$ (las cajas llenan completamente la envolvente). La cota se vuelve más ajustada cuanto más cerca están las formas.
4. **Rango simétrico:** $-1 \leq \text{GIoU}(A, B) \leq 1$.
   - $\text{GIoU} = 1 \iff A = B$ (idéntico a IoU = 1).
   - $\text{GIoU} \to -1$ cuando $|A \cup B| / |C| \to 0$, es decir, cuando las cajas son infinitamente pequeñas relativas a la envolvente (están muy alejadas).

### 2.3 Algoritmo en pseudocódigo

```python
def giou(B_p, B_g):
    """
    B_p, B_g: (x1, y1, x2, y2) axis-aligned bounding boxes.
    """
    # 1. Asegurar x2 > x1, y2 > y1 en la prediccion
    x1p, y1p = min(B_p[0], B_p[2]), min(B_p[1], B_p[3])
    x2p, y2p = max(B_p[0], B_p[2]), max(B_p[1], B_p[3])
    x1g, y1g, x2g, y2g = B_g

    # 2. Areas individuales
    A_p = (x2p - x1p) * (y2p - y1p)
    A_g = (x2g - x1g) * (y2g - y1g)

    # 3. Interseccion
    xi1, yi1 = max(x1p, x1g), max(y1p, y1g)
    xi2, yi2 = min(x2p, x2g), min(y2p, y2g)
    if xi2 > xi1 and yi2 > yi1:
        I = (xi2 - xi1) * (yi2 - yi1)
    else:
        I = 0.0

    # 4. Union
    U = A_p + A_g - I
    iou = I / U

    # 5. Caja envolvente C
    xc1, yc1 = min(x1p, x1g), min(y1p, y1g)
    xc2, yc2 = max(x2p, x2g), max(y2p, y2g)
    A_c = (xc2 - xc1) * (yc2 - yc1)

    # 6. GIoU
    giou_val = iou - (A_c - U) / A_c
    return giou_val

def giou_loss(B_p, B_g):
    return 1.0 - giou(B_p, B_g)
```

Todos los operadores (`min`, `max`, divisiones) son diferenciables casi en todas partes; la condición `if xi2 > xi1 and yi2 > yi1` introduce una rama no diferenciable únicamente en la frontera de medida cero. En la práctica los frameworks (PyTorch, TF) usan ReLU-like clipping para mantenerlo bien comportado.

---

## 3. GIoU como loss

### 3.1 Definición y rango

$$
\mathcal{L}_{GIoU}(B^p, B^g) = 1 - \text{GIoU}(B^p, B^g), \quad \mathcal{L}_{GIoU} \in [0, 2].
$$

A diferencia de $\mathcal{L}_{IoU} \in [0, 1]$ que satura en 1 cuando no hay overlap, $\mathcal{L}_{GIoU}$ sigue creciendo hasta 2 a medida que las cajas se alejan.

### 3.2 Comportamiento cuando $\text{IoU} = 0$

El paso clave del paper. Si $I = 0$ y por lo tanto $\text{IoU} = 0$:

$$
\mathcal{L}_{GIoU} = 1 - \text{GIoU} = 1 + \frac{A^c - U}{A^c} = 2 - \frac{U}{A^c}.
$$

Minimizar $\mathcal{L}_{GIoU}$ en el régimen sin overlap equivale a **maximizar** $\frac{U}{A^c} = \frac{A^p + A^g}{A^c}$. Como $A^g$ es fijo, la red está forzada a:

- **Aumentar $A^p$** (hacer la caja predicha más grande para que se acerque a la GT), y/o
- **Reducir $A^c$** (mover la caja predicha hacia la GT para que la envolvente se encoja).

Ambas dinámicas empujan la predicción hacia el ground truth. El gradiente es **no nulo** en todo el espacio de configuraciones sin overlap, eliminando el plateau de IoU.

### 3.3 Estabilidad numérica

El paper prueba (sección 3 del paper, transcrito):

- $A^g > 0$ por definición (GT no degenerada).
- El paso 1 del algoritmo fuerza $A^p \geq 0$.
- $I \geq 0$ por el clipping del paso 4.
- $U \geq I$ siempre, por lo que el denominador de IoU es $> 0$.
- $A^c \geq A^g > 0$ siempre, por lo que el denominador de la penalización está acotado.
- $A^c \geq U$ siempre, por lo que la penalización $\frac{A^c - U}{A^c} \in [0, 1)$.

Esto garantiza $0 \leq \mathcal{L}_{GIoU} \leq 2$ para cualquier predicción $B^p \in \mathbb{R}^4$, sin overflow, divisiones por cero ni NaN.

### 3.4 Ejemplo numérico de $\ell_2$ vs GIoU

Considere GT = $(0, 0, 10, 10)$ (área 100) y dos predicciones con la misma $\ell_2$:

- **Pred A:** $(0, 0, 10, 20)$ — caja alargada en $y$. $\ell_2 = 10$. $I = 100$, $U = 200$, $\text{IoU} = 0.5$. $C = (0, 0, 10, 20)$, $A^c = 200$. $\text{GIoU} = 0.5 - (200 - 200)/200 = 0.5$.
- **Pred B:** $(10, 0, 20, 10)$ — caja desplazada en $x$. $\ell_2 = 10$. $I = 0$, $U = 200$, $\text{IoU} = 0$. $C = (0, 0, 20, 10)$, $A^c = 200$. $\text{GIoU} = 0 - (200 - 200)/200 = 0$.

Misma $\ell_2$, mismo nivel de "distancia coordenada", pero predicción A es claramente mejor (overlap parcial) y predicción B es claramente peor (sin overlap, adyacente). GIoU las distingue (0.5 vs 0), $\ell_2$ no.

---

## 4. Experimentos

Los autores integran $\mathcal{L}_{GIoU}$ en tres detectores estándar reemplazando sus pérdidas de regresión nativas, sin cambiar arquitectura, anchors, learning rate ni schedule.

### 4.1 YOLOv3 (MSE → IoU → GIoU)

YOLOv3 usa DarkNet-608. Para entrenar con $\mathcal{L}_{IoU}$ o $\mathcal{L}_{GIoU}$ los autores simplemente reemplazan MSE en la regresión, manteniendo la pérdida MSE en clasificación.

**PASCAL VOC 2007** (50K iteraciones, 9963 imágenes, 20 clases):

| Loss | AP (IoU) | AP (GIoU) | AP75 (IoU) | AP75 (GIoU) |
|------|----------|-----------|------------|-------------|
| MSE  | 0.461    | 0.451     | 0.486      | 0.467       |
| $\mathcal{L}_{IoU}$ | 0.466 | 0.460 | 0.504 | 0.498 |
| $\mathcal{L}_{GIoU}$ | **0.477** | **0.469** | **0.513** | **0.499** |

GIoU mejora AP en +3.45% relativo sobre MSE y +1.08 puntos sobre IoU loss. La mejora en AP75 (umbral estricto, requiere localización precisa) es +5.56% relativo.

**MS COCO val 2014** (502K iteraciones, 80 clases):

| Loss | AP (IoU) | AP (GIoU) | AP75 (IoU) | AP75 (GIoU) |
|------|----------|-----------|------------|-------------|
| MSE  | 0.314    | 0.302     | 0.329      | 0.317       |
| $\mathcal{L}_{IoU}$ | 0.322 | 0.313 | 0.345 | 0.335 |
| $\mathcal{L}_{GIoU}$ | **0.335** | **0.325** | **0.359** | **0.348** |

Mejora relativa de +6.69% en AP. En COCO test 2018 (server): +5.71% AP y +8.01% AP75.

La figura 3 del paper muestra que el average IoU sobre cajas predichas (no AP, sino calidad localización pura) durante entrenamiento converge más rápido y más alto con GIoU loss que con MSE. La pérdida de clasificación, en cambio, queda ligeramente inferior con GIoU porque no se rebalanceó el peso entre regresión y clasificación; los autores reconocen que un mejor tuning podría aumentar aún más AP.

### 4.2 Faster R-CNN (Smooth L1 → IoU → GIoU)

Backbone ResNet-50, multiplicador $\times 10$ para las pérdidas $\mathcal{L}_{IoU}, \mathcal{L}_{GIoU}$ contra clasificación.

**PASCAL VOC 2007:**

| Loss | AP (IoU) | AP (GIoU) | AP75 (IoU) | AP75 (GIoU) |
|------|----------|-----------|------------|-------------|
| Smooth $L_1$ | 0.370 | 0.361 | 0.358 | 0.346 |
| $\mathcal{L}_{IoU}$ | 0.384 | 0.375 | 0.395 | 0.382 |
| $\mathcal{L}_{GIoU}$ | **0.392** | **0.382** | **0.404** | **0.395** |

Salto enorme en AP75: +12.85% relativo. La figura 4 muestra que la ventaja de GIoU sobre Smooth $L_1$ se amplifica conforme el umbral de IoU sube de 0.5 a 0.95.

**MS COCO 2018 val:** AP de 0.360 → 0.369 (+2.50%), AP75 de 0.390 → 0.398 (+2.05%).

### 4.3 Mask R-CNN (Smooth L1 → IoU → GIoU)

**MS COCO 2018 val:**

| Loss | AP (IoU) | AP (GIoU) | AP75 (IoU) | AP75 (GIoU) |
|------|----------|-----------|------------|-------------|
| Smooth $L_1$ | 0.366 | 0.356 | 0.397 | 0.385 |
| $\mathcal{L}_{IoU}$ | 0.374 | 0.364 | 0.404 | 0.393 |
| $\mathcal{L}_{GIoU}$ | **0.376** | **0.366** | **0.405** | **0.395** |

Mejora más modesta en Faster/Mask R-CNN que en YOLOv3. Los autores ofrecen dos explicaciones:

1. **Densidad de anchors:** Faster R-CNN tiene anchors muy densas y selecciona positivos con IoU ≥ 0.7 contra GT. Las predicciones rara vez empiezan sin solapamiento, por lo que la principal ventaja de GIoU (gradient en no-overlap) se activa menos.
2. **Hyperparam tuning:** el factor $\times 10$ fue tuneado en VOC, probablemente subóptimo para COCO.

### 4.4 Síntesis

| Detector | Backbone | Baseline loss | Mejora AP relativa con GIoU | Mejora AP75 relativa |
|----------|----------|---------------|-----|------|
| YOLOv3 (VOC) | DarkNet-608 | MSE | +3.45% | +5.56% |
| YOLOv3 (COCO val) | DarkNet-608 | MSE | +6.69% | +9.12% |
| YOLOv3 (COCO test) | DarkNet-608 | MSE | +5.71% | +8.01% |
| Faster R-CNN (VOC) | ResNet-50 | Smooth $L_1$ | +5.95% | +12.85% |
| Faster R-CNN (COCO val) | ResNet-50 | Smooth $L_1$ | +2.50% | +2.05% |
| Mask R-CNN (COCO val) | ResNet-50 | Smooth $L_1$ | +2.73% | +2.02% |

Patrón consistente: GIoU > IoU > pérdidas $\ell_p$, y la brecha es mayor en (a) detectores con anchors menos densas (YOLOv3 vs Faster R-CNN), y (b) métricas más estrictas (AP75 > AP).

---

## 5. Análisis geométrico

### 5.1 Visualización del gradiente

La figura 2 del paper muestra 10K pares de cajas muestreadas aleatoriamente. En el régimen overlapping (IoU > 0), la nube cae sobre la línea $\text{IoU} = \text{GIoU}$ con desviaciones pequeñas. En el régimen non-overlapping (IoU = 0), $\text{GIoU}$ se dispersa en el rango $[-1, 0]$ según cuán lejos estén las cajas: aquí toda la señal de gradiente proviene del término $-\frac{|C \setminus (A \cup B)|}{|C|}$.

Adicionalmente, en el régimen de bajo overlap ($\text{IoU} \leq 0.2$), GIoU puede cambiar mucho más rápido que IoU con pequeñas variaciones de coordenadas, lo que se traduce en gradientes más empinados y convergencia más rápida.

### 5.2 Parametrización: $(x_1, y_1, x_2, y_2)$ vs $(x_c, y_c, w, h)$

La figura 1 del paper presenta el mismo argumento en dos parametrizaciones. Para corners, $\ell_2$ ve la distancia euclidiana en $\mathbb{R}^4$ de los corners; para center-size, $\ell_1$ ve la distancia Manhattan. Ambas son **invariantes** a la elección de $(x_c, y_c, w, h)$ vs $(x_1, y_1, x_2, y_2)$ módulo un cambio de coordenadas, pero **no son invariantes a transformaciones geométricamente sensatas** como una traslación pura vs un escalado.

GIoU, al estar definido directamente en términos de áreas (cantidad geométricamente invariante), no depende de la parametrización. Esto es una ventaja conceptual: la red puede usar internamente cualquier parametrización conveniente y la pérdida sigue siendo la métrica correcta.

### 5.3 Gradiente analítico en caso non-overlap

Para $I = 0$, con la simplificación $\mathcal{L}_{GIoU} = 2 - \frac{U}{A^c}$:

$$
\frac{\partial \mathcal{L}_{GIoU}}{\partial x_1^p} = -\frac{1}{A^c} \frac{\partial A^p}{\partial x_1^p} + \frac{U}{(A^c)^2} \frac{\partial A^c}{\partial x_1^p}.
$$

Con $\frac{\partial A^p}{\partial x_1^p} = -(y_2^p - y_1^p)$ (la caja se encoge si $x_1^p$ aumenta) y $\frac{\partial A^c}{\partial x_1^p}$ depende de si $x_1^p < x_1^g$ (entonces $x_1^c = x_1^p$ y $\frac{\partial A^c}{\partial x_1^p} = -(y_2^c - y_1^c)$) o no ($x_1^c = x_1^g$ constante, derivada cero). Es decir, el gradiente tiene **estructura geométrica explícita** que codifica "muévete hacia la GT".

---

## 6. Limitaciones reconocibles

A pesar del salto cualitativo sobre IoU loss, GIoU sigue teniendo debilidades concretas que la literatura posterior atacó.

### 6.1 Convergencia lenta sin overlap

Cuando $A$ y $B$ están muy alejados, $\frac{|C \setminus (A \cup B)|}{|C|}$ se aproxima a 1 pero su derivada respecto a las coordenadas se aplana. La señal de gradiente existe (a diferencia de IoU puro) pero es pequeña. En la práctica esto se traduce en que GIoU loss converge más lento que pérdidas con términos de distancia explícita entre centros (ver DIoU/CIoU abajo).

### 6.2 Caso degenerado: caja predicha contiene la GT (o viceversa)

Si $B^p \supset B^g$, entonces $A \cup B = B^p$ y $C = B^p$, por lo que $|C \setminus (A \cup B)| = 0$ y $\text{GIoU} = \text{IoU}$. En este caso GIoU **no añade información** respecto a IoU; el término extra de penalización se anula. La red ve el mismo gradiente que con IoU loss puro, perdiendo la ventaja del enclosing.

Análogamente, si dos cajas tienen el mismo $A \cup B$ (mismo área de unión) y el mismo $A^c$ (misma envolvente) pero distinta intersección, GIoU y IoU coinciden en cuán informativos son.

### 6.3 No considera distancia entre centros

GIoU codifica el "espacio vacío en la envolvente" pero no penaliza explícitamente que los centros estén descentrados. Dos predicciones con misma GIoU pueden tener centros muy diferentes respecto al GT, y geométricamente la que tiene centros alineados suele ser preferible para downstream tasks (tracking, NMS).

### 6.4 Aspect ratio no es un término separado

GIoU se basa puramente en áreas. Una predicción que comparte centro con la GT pero tiene aspect ratio distinto (digamos GT cuadrada vs pred muy alargada) puede tener GIoU razonable simplemente por el solapamiento de área, sin penalización adicional por la disonancia de forma. Para tareas como detección de texto donde la geometría es discriminativa (texto generalmente alargado horizontalmente), esto puede ser limitante.

### 6.5 Interacción con clasificación

El paper reporta que en YOLOv3, al reemplazar MSE por GIoU loss en la rama de regresión, la pérdida de clasificación queda ligeramente más alta (peor) que en baseline (figura 3b del paper). Esto se debe a que el balance entre la magnitud de la pérdida de regresión y la de clasificación cambió: MSE no está acotado y crece sin límite, mientras que $\mathcal{L}_{GIoU} \in [0, 2]$. El gradiente combinado favorece distinto a las dos cabezas. Los autores reconocen que un tuning cuidadoso del peso relativo recuperaría la calidad de clasificación, pero no lo exploraron en profundidad. Esto significa que en producción **GIoU loss usualmente requiere re-tunear el peso de la regresión vs clasificación**, no es plug-and-play perfecto.

### 6.6 Caso de cajas inclinadas o no axis-aligned

El paper explícitamente limita su solución analítica a axis-aligned bounding boxes. Para rotated boxes (relevante en detección aérea, texto inclinado, detección 3D de cuboides), el cómputo de $|A \cap B|$ y $|C|$ requiere algoritmos de intersección de polígonos. Aunque la teoría se extiende, el costo computacional sube considerablemente y la diferenciabilidad de la convex hull requiere truco (smooth approximation). Esto se aborda en literatura subsecuente (Rot-GIoU, KFIoU, ProbIoU).

---

## 7. Variantes posteriores

### 7.1 DIoU y CIoU (Zheng et al., AAAI 2020)

**DIoU (Distance IoU):** agrega un término de distancia entre centros, normalizado por la diagonal de la envolvente.

$$
\mathcal{L}_{DIoU} = 1 - \text{IoU} + \frac{\rho^2(b^p, b^g)}{c^2}
$$

donde $\rho$ es la distancia euclidiana entre centros $b^p, b^g$ y $c$ es la diagonal de $C$. Esto soluciona la convergencia lenta en non-overlap y el caso degenerado de contención: el término $\rho^2/c^2$ entrega gradiente útil incluso cuando $A \cup B = C$.

**CIoU (Complete IoU):** añade además un término de consistencia de aspect ratio:

$$
\mathcal{L}_{CIoU} = \mathcal{L}_{DIoU} + \alpha v, \quad v = \frac{4}{\pi^2} \left(\arctan \frac{w^g}{h^g} - \arctan \frac{w^p}{h^p}\right)^2
$$

con $\alpha$ un balance ponderado por IoU. CIoU es la **default en YOLOv4** (Bochkovskiy 2020) y se adopta extensamente en YOLOv5, YOLOv7, RTMDet.

### 7.2 EIoU, SIoU, WIoU, $\alpha$-IoU

- **EIoU (Efficient IoU, Zhang 2022):** descompone el término de aspect ratio de CIoU en penalizaciones separadas para $w$ y $h$, observando que el arctan acoplado de CIoU puede generar gradientes contradictorios.
- **SIoU (Scylla IoU, Gevorgyan 2022):** introduce un término direccional (angle cost) entre centros, argumentando que la dirección del vector $b^g - b^p$ importa además de la magnitud.
- **WIoU (Wise IoU, Tong 2023):** asigna pesos dinámicos según calidad: ejemplos de alta calidad reciben menos peso, ejemplos de baja calidad reciben más; estilo focal loss aplicado a la regresión.
- **$\alpha$-IoU (He 2021):** generaliza $\mathcal{L}_{IoU/GIoU/DIoU/CIoU}$ con un exponente $\alpha$ aplicado a IoU, dando control fino sobre el shape del gradiente.

### 7.3 Adopción en detectores modernos

| Detector | Año | Loss de regresión |
|----------|-----|-------------------|
| YOLOv3 | 2018 | MSE (default), GIoU (post-paper) |
| YOLOv4 | 2020 | **CIoU** |
| YOLOv5 | 2020 | CIoU (default), GIoU disponible |
| YOLOv7 | 2022 | CIoU |
| YOLOv8 | 2023 | CIoU + DFL (Distribution Focal Loss) |
| FCOS | 2019 | GIoU (centerness branch usa BCE) |
| RetinaNet | 2017 | Smooth L1 → GIoU en re-implementaciones |
| DETR | 2020 | L1 + GIoU combinados |
| RTMDet | 2022 | GIoU + Quality Focal Loss |
| ABCNet (texto) | 2020 | IoU loss |

GIoU es **el loss de transición** que abrió la puerta a toda esta familia; en producción la mayoría de los frameworks ofrecen GIoU/DIoU/CIoU como opción de configuración estándar.

---

## 8. Por qué importa para Scene Text Recognition (Clase 21)

Aunque el paper se centra en detección genérica de objetos, los conceptos son directamente aplicables a Scene Text Detection (STD) y Scene Text Recognition (STR), temas centrales de la Clase 21.

### 8.1 ABCNet y la geometría del texto

ABCNet (Liu et al. CVPR 2020) propone parametrizar texto curvo con curvas de Bézier. Para el bounding region de control points usa **IoU loss directamente**. Reemplazar IoU por GIoU/CIoU es un paso natural:

- En texto escena el bounding box suele estar inclinado o curvado; las predicciones iniciales rara vez solapan la GT, por lo que el plateau de IoU es especialmente perjudicial.
- Los detectores de texto suelen tener anchors menos densas que Faster R-CNN (más cerca de YOLOv3), maximizando la ganancia de GIoU.

Trabajos recientes de STD (EAST, PSENet, DBNet, FCENet) adoptan variantes IoU-based para la rama de regresión geométrica.

### 8.2 Generalización a polígonos y curvas

El paper define GIoU sobre **formas convexas arbitrarias** con envolvente convexa $C$. Esto se extiende naturalmente a:

- **Rotated bounding boxes:** GIoU rotada (Rot-GIoU) se usa en detección aérea (DOTA dataset) y texto inclinado. La envolvente convexa de dos rectángulos rotados se calcula con convex hull.
- **PolyIoU:** para polígonos arbitrarios (texto curvo, instance segmentation), $|A|, |B|, |A \cap B|, |C|$ se calculan con algoritmos de intersección de polígonos (Sutherland-Hodgman, GH-clipping).
- **Bezier-area:** ABCNet podría re-formularse con un GIoU sobre el área bajo dos curvas Bézier, pero la implementación es no trivial.

### 8.3 STR end-to-end y backpropagation through detection

Los pipelines STR end-to-end (Mask TextSpotter v3, ABCNet, MANGO) backpropagan a través del módulo de detección. Una loss de regresión más informativa (GIoU/CIoU) significa mejores gradientes para el detector, que a su vez genera mejores ROI features para el reconocedor de caracteres. Las ganancias se acumulan: un detector que localiza más precisamente entrega ROI features mejor alineadas, lo que reduce ambigüedad en el reconocedor (que en STR suele ser un Transformer o un CTC-RNN sensible al centrado del crop).

### 8.4 Robustez en escenarios long-tail de texto escena

Datasets de texto escena como ICDAR 2015, Total-Text, CTW1500 contienen instancias de texto con escalas extremadamente variables (texto de letrero vs subtítulo en imagen), aspect ratios extremos (texto vertical chino vs banner horizontal larguísimo) y muchas instancias por imagen (densidad alta). En estos regímenes:

- Pérdidas $\ell_p$ sobre $(x_1, y_1, x_2, y_2)$ fallan más en instancias pequeñas porque su error absoluto domina la pérdida, sesgando el entrenamiento hacia texto grande.
- IoU loss puede plateau en texto pequeño porque las predicciones iniciales suelen estar desplazadas más allá del área del GT (overlap = 0).
- GIoU/CIoU, al ser invariantes a escala y dar gradiente fuera de overlap, mantienen señal útil en todo el rango de tamaños.

Esto explica empíricamente por qué los detectores de texto post-2020 que adoptaron GIoU/CIoU reportan mejoras especialmente en small-text recall (la subtarea históricamente más difícil del STR pipeline).

---

## 9. Conexión con la clase y otros papers

### 9.1 FPN (Lin et al. CVPR 2017)

FPN cambia la receta de extracción multi-escala usando una pirámide top-down con conexiones laterales. La regresión de bounding boxes se hace por cada nivel de la pirámide; cuando se combina con GIoU en lugar de Smooth $L_1$, las ganancias se observan más fuerte en niveles altos (objetos grandes) porque las cajas en esos niveles tienden a tener menos overlap con anchors estándar (que están optimizadas para tamaños medianos).

### 9.2 Faster R-CNN (Ren et al. NeurIPS 2015)

El experimento de la sección 4.2 reemplaza Smooth $L_1$ del bbox refinement head por GIoU. La estructura two-stage de Faster R-CNN (RPN + ROI head) hace que el segundo stage rara vez vea predicciones sin overlap (porque RPN ya filtra), explicando por qué la ganancia de GIoU es más modesta aquí.

### 9.3 FCOS (Tian et al. ICCV 2019)

FCOS es anchor-free. Predice 4 distancias $(l, t, r, b)$ desde cada pixel hasta los bordes de la caja, y usa **IoU loss** como pérdida de regresión por defecto. En implementaciones modernas FCOS se entrena con GIoU. FCOS añade un branch de "centerness" que reduce predicciones de baja calidad lejanas del centro, lo que cumple un rol complementario al de DIoU (que penaliza distancia entre centros explícitamente en la loss de regresión).

### 9.4 UnitBox (Yu et al. ACM MM 2016)

UnitBox fue el primero en proponer IoU directamente como loss para detección facial; Rezatofighi cita este trabajo como antecedente principal. UnitBox sufre el plateau cuando no hay overlap, y GIoU lo resuelve. La línea histórica:

```
2016: UnitBox (IoU loss para face detection)
  └─ 2019: GIoU (resuelve el plateau con enclosing penalty)
       └─ 2020: DIoU/CIoU (añade distancia + aspect ratio)
            └─ 2022+: EIoU, SIoU, WIoU, alpha-IoU (refinamientos)
```

### 9.5 Conexión con detección anchor-free moderna

Detectores anchor-free como FCOS, CenterNet, ATSS, GFL (Generalized Focal Loss) dependen aún más de losses IoU-based porque no tienen el "regularizador implícito" de las anchor boxes. En estos detectores GIoU/CIoU no es solo una mejora marginal sino una pieza estructural. RTMDet (Lyu et al. 2022) reporta que cambiar de GIoU a CIoU le da ~0.5 AP, mostrando que la elección específica de la variante aún importa.

---

## 10. Referencias clave

- **Yu et al. (2016) — UnitBox** (ACM MM 2016): primer uso de IoU como loss directa de regresión para detección facial. Demuestra que IoU es diferenciable en su régimen overlapping; deja abierto el problema del plateau.
- **Ren et al. (2015) — Faster R-CNN** (NeurIPS 2015): introduce RPN + ROI head, usa Smooth $L_1$. El target principal de "mejora por reemplazo de loss" en este paper.
- **Redmon & Farhadi (2018) — YOLOv3**: arquitectura de referencia para los experimentos. Usa MSE sobre $(x_c, y_c, \sqrt{w}, \sqrt{h})$.
- **He et al. (2017) — Mask R-CNN** (ICCV 2017): extiende Faster R-CNN con rama de segmentación. La rama de bbox sigue usando Smooth $L_1$.
- **Lin et al. (2017) — FPN** (CVPR 2017): feature pyramid network, complementario al loss de regresión.
- **Zheng et al. (2020) — DIoU/CIoU** (AAAI 2020): sucesor directo, añade términos de distancia entre centros y aspect ratio.
- **Bochkovskiy et al. (2020) — YOLOv4**: primer detector mainstream en adoptar CIoU como loss default.
- **Kosub (2016)** (arXiv 1612.02696): prueba formal de que $1 - \text{Jaccard}$ satisface la desigualdad triangular, justificando que GIoU también es métrica matemática.
- **Rahman & Wang (2016)** y **Berman et al. (2018) — Lovász-Softmax**: optimización de IoU surrogada en segmentación semántica, work paralelo conceptual.

---

## Notas para Roberto

- **Por qué este paper es seminal en la práctica:** transformó "loss de regresión" de un detalle de implementación en un eje de mejora explícito. Hoy si entrenas un detector y reportas Smooth $L_1$ sin haber considerado al menos GIoU, estás dejando AP sobre la mesa.
- **Conexión con tu pipeline FHIR/MDM:** el patrón "métrica de evaluación $\neq$ loss de entrenamiento" es exactamente el mismo dilema que aparece en MDM cuando entrenas un GBM XGBoost con logloss pero te evalúan con F1-at-threshold o precision-recall AUC. La lección de GIoU es: cuando puedas hacer la métrica diferenciable, hazlo, y cuando no puedas, busca un surrogate que tenga gradiente útil en todo el espacio (no que se desvanezca en el régimen "no overlap" = "no match").
- **Para tu paper de retornos decrecientes en MDM LATAM:** el patrón de GIoU > IoU > $\ell_2$ es el mismo patrón de "mejor loss = pequeña mejora robusta sin cambiar arquitectura". Suele ser de las mejoras más universales: bajo riesgo, pequeño upside, fácil de A/B test. El equivalente en MDM sería: ¿hay un loss que respete la métrica de matching (F1@threshold ajustado por cobertura) en lugar de cross-entropy estándar?
- **Sobre la complejidad de implementación:** GIoU loss añade ~10 líneas de código respecto a Smooth $L_1$. No tiene hiperparámetros (a diferencia de CIoU que tiene $\alpha$, o WIoU que tiene varios). Si tu detector usa una loss $\ell_p$ y quieres explorar mejoras de regresión, GIoU es el primer experimento que vale la pena correr antes de saltar a DIoU/CIoU. La regla práctica es: GIoU para validar la dirección, CIoU para producción.
- **Sobre cómo encaja en torchvision/MMDetection:** ambas librerías exponen `generalized_box_iou_loss` (torchvision) y `GIoULoss` (mmdet) como módulos drop-in. La API espera cajas en formato `(x1, y1, x2, y2)` y aplica el algoritmo del paper línea a línea. No hay aproximaciones; lo que estos frameworks computan es exactamente $\mathcal{L}_{GIoU} = 1 - \text{GIoU}$ de este paper.
- **Conexión con la línea de papers de la Clase 21:** GIoU es uno de los building blocks transversales de detección moderna. Si la clase trata STR como tarea downstream, conviene tenerlo en cabeza para entender cómo papers como ABCNet (Liu 2020) y FCENet eligen sus losses de regresión geométrica. La cita típica en estos papers es "we use IoU/GIoU loss for bounding box regression following [Rezatofighi 2019]", y aquí tienes el "por qué" completo del por qué esa cita aparece.

---

## Apéndice: prueba de invariancia a escala (informal)

Sea $s > 0$ un factor de escala uniforme. Considera $A' = sA$, $B' = sB$, $C' = sC$ (la envolvente de las versiones escaladas es la versión escalada de la envolvente, por axis-alignment).

- $|sA| = s^n |A|$ en $\mathbb{R}^n$.
- $|sA \cap sB| = s^n |A \cap B|$.
- $|sA \cup sB| = s^n |A \cup B|$.
- $|sC \setminus (sA \cup sB)| = s^n |C \setminus (A \cup B)|$.

Sustituyendo:

$$
\text{GIoU}(sA, sB) = \frac{s^n |A \cap B|}{s^n |A \cup B|} - \frac{s^n |C \setminus (A \cup B)|}{s^n |C|} = \text{GIoU}(A, B).
$$

Los factores $s^n$ se cancelan en cada cociente. Esta cancelación es precisamente lo que **no ocurre** en pérdidas $\ell_p$: $\|sA - sB\|_2 = s \|A - B\|_2$, por lo que el error escala con $s$ y la red ve gradientes proporcionalmente más grandes en cajas grandes que en pequeñas (sesgo de escala).

## Apéndice: el caso 1D para intuición

Para construir intuición geométrica, considera el caso 1D: dos intervalos $A = [a_1, a_2]$ y $B = [b_1, b_2]$ en $\mathbb{R}$. La envolvente es $C = [\min(a_1, b_1), \max(a_2, b_2)]$.

- Si $A \cap B \neq \emptyset$: $\text{IoU} = \frac{\min(a_2, b_2) - \max(a_1, b_1)}{\max(a_2, b_2) - \min(a_1, b_1)} \in (0, 1]$, y $C = A \cup B$, por lo que $|C \setminus (A \cup B)| = 0$ y $\text{GIoU} = \text{IoU}$.
- Si $A \cap B = \emptyset$ (intervalos disjuntos): $\text{IoU} = 0$. Supongamos $a_2 < b_1$. Entonces $|A \cup B| = (a_2 - a_1) + (b_2 - b_1)$ y $|C| = b_2 - a_1$. La penalización es $\frac{b_1 - a_2}{b_2 - a_1}$ (el gap entre los intervalos sobre la longitud total de la envolvente). Así $\text{GIoU} = -\frac{b_1 - a_2}{b_2 - a_1} \in (-1, 0)$.

En 1D la fórmula se reduce a una **medida normalizada del gap** entre los intervalos. Esta es la generalización exacta de la métrica de Jaccard a casos disjuntos, y la base intuitiva del paper.

## Apéndice: GIoU como caso particular de una familia

Visto desde una perspectiva más general, $\mathcal{L}_{GIoU}$ pertenece a la familia:

$$
\mathcal{L} = 1 - \text{IoU} + \mathcal{R}(B^p, B^g)
$$

donde $\mathcal{R}$ es un regularizador geométrico. Las variantes posteriores eligen $\mathcal{R}$ distintos:

- **GIoU:** $\mathcal{R} = \frac{|C \setminus (A \cup B)|}{|C|}$ — espacio vacío en envolvente.
- **DIoU:** $\mathcal{R} = \frac{\rho^2(b^p, b^g)}{c^2}$ — distancia normalizada entre centros.
- **CIoU:** $\mathcal{R} = \mathcal{R}_{DIoU} + \alpha v$ — distancia + aspect ratio.

Esta forma factorizada explica por qué los tres reportes de papers comparativos colocan GIoU como "mid-tier" frente a CIoU: GIoU codifica una dimensión adicional sobre IoU (espacio vacío), DIoU codifica otra (distancia entre centros), CIoU codifica las tres. Cuando $A \subset B$ o viceversa, solo DIoU/CIoU mantienen gradiente útil, mientras que GIoU colapsa a IoU. Este es el modo de falla específico que motivó la siguiente generación.
