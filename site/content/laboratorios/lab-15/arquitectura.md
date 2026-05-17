---
title: "Arquitectura de Faster R-CNN: las 4 piezas"
weight: 10
math: true
---

Antes de entender el codigo del lab hay que tener clara la arquitectura del modelo que vamos a usar (y luego fine-tunear). `torchvision.models.detection.fasterrcnn_resnet50_fpn` es una version moderna del paper [Faster R-CNN](/papers/faster-rcnn-ren-2015/) (Ren et al. 2015) con tres componentes adicionales que no estaban en el original: ResNet-50 como backbone, FPN como neck multi-escala, y RoIAlign para extraer features de propuestas.

## Vista de pajaro: 4 piezas en serie

```text
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌────────────┐
│ Imagen  │ → │ Backbone │ → │   RPN   │ → │ Clasifica- │ → Detecciones
│ entrada │   │  (CNN)   │   │ "donde  │   │ dor de RoI │
│         │   │          │   │  mirar" │   │ "que hay"  │
└─────────┘   └──────────┘   └─────────┘   └────────────┘
```

Mas la **pieza 0** (`transform`) que normaliza la imagen antes de todo.

---

## Pieza 0 — `transform`: preprocesamiento

```python
(transform): GeneralizedRCNNTransform(
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    Resize(min_size=(800,), max_size=1333, mode='bilinear')
)
```

Esta pieza **no aprende**. Hace dos cosas:

1. **Normalize**: resta la media y divide por la desviacion estandar de **ImageNet**. La ResNet-50 fue entrenada esperando esa distribucion estadistica.
2. **Resize**: escala la imagen de manera que el lado corto sea $\geq 800$ y el lado largo $\leq 1333$ px, manteniendo aspect ratio.

**Implicacion practica:** puedes pasar imagenes de cualquier tamano. El transform se encarga internamente. Esa es la respuesta a la **pregunta 2 de la tarea final** del notebook.

---

## Pieza 1 — `backbone`: extraccion de features multi-escala

### 1a — ResNet-50 (`backbone.body`)

La CNN clasica que extrae features. Estructura:

| Stage | Capas | Output (de imagen 800x800) | Stride | Canales |
| --- | --- | --- | --- | --- |
| stem | conv 7x7 + bn + relu + maxpool | (64, 200, 200) | /4 | 64 |
| layer1 | 3 Bottlenecks | $C_2$: (256, 200, 200) | /4 | 256 |
| layer2 | 4 Bottlenecks | $C_3$: (512, 100, 100) | /8 | 512 |
| layer3 | 6 Bottlenecks | $C_4$: (1024, 50, 50) | /16 | 1024 |
| layer4 | 3 Bottlenecks | $C_5$: (2048, 25, 25) | /32 | 2048 |

Cada **Bottleneck** es:

```text
input → [conv 1x1 reduce] → [conv 3x3] → [conv 1x1 expand] → + input (skip) → ReLU
```

La unidad clave de [ResNet](/papers/resnet-he-2015/): las conexiones residuales permiten entrenar 50, 100 o 150 capas sin que el gradiente se desvanezca.

**Detalle del lab:** torchvision usa `FrozenBatchNorm2d` (BN con estadisticas congeladas) porque los detectores usan batch sizes pequenos (1-6 imagenes), y BN con batches chicos es inestable.

### 1b — FPN (`backbone.fpn`)

El problema de quedarse solo con $C_5$ (lo que hacia el paper Faster R-CNN original):

- $C_2$ tiene **alta resolucion** pero **pobre semantica** -> objetos pequenos visibles pero la red no sabe que son.
- $C_5$ tiene **alta semantica** pero **pobre resolucion** -> objetos pequenos desaparecen.

La **Feature Pyramid Network** ([Lin et al. 2017](/papers/fpn-lin-2017/)) combina ambos via **top-down + lateral connections**:

```text
C5 ──[1x1 conv]── P5
                  │
                  upsample 2x (nearest neighbor)
                  ▼
C4 ──[1x1 conv]── + → P4
                  │
                  upsample 2x
                  ▼
C3 ──[1x1 conv]── + → P3
                  │
                  upsample 2x
                  ▼
C2 ──[1x1 conv]── + → P2
```

Mas un **$P_6$** que torchvision anade con MaxPool 1x1 stride 2 sobre $P_5$ para detectar objetos muy grandes.

Resultado: 5 feature maps $\{P_2, P_3, P_4, P_5, P_6\}$ todos con **256 canales** y semantica fuerte en todos los niveles. Resoluciones (200, 100, 50, 25, 13) aprox.

---

## Pieza 2 — `rpn`: la Region Proposal Network

```python
(rpn): RegionProposalNetwork(
    (anchor_generator): AnchorGenerator()
    (head): RPNHead(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        (cls_logits): Conv2d(256, 3, kernel_size=(1, 1))
        (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1))
    )
)
```

### Anchors — cajas de referencia preplantadas

En cada celda de cada feature map, se plantan **3 anchors fijos** (3 aspect ratios: 1:1, 1:2, 2:1). La escala depende del nivel:

| Nivel | Stride | Area del anchor | Para detectar objetos de... |
| --- | --- | --- | --- |
| $P_2$ | /4 | $32^2$ | ~32 px (muy pequenos) |
| $P_3$ | /8 | $64^2$ | ~64 px |
| $P_4$ | /16 | $128^2$ | ~128 px (medianos) |
| $P_5$ | /32 | $256^2$ | ~256 px |
| $P_6$ | /64 | $512^2$ | ~512 px (muy grandes) |

Total de anchors en una imagen tipica: **~160.000**.

### Que hace la RPN

3 convoluciones:

1. **Conv 3x3** sobre el feature map (integra el vecindario de cada celda).
2. **Conv 1x1 cls** -> 3 canales = 3 scores de **objectness** (objeto vs fondo) por celda.
3. **Conv 1x1 reg** -> 12 canales = 3 anchors × 4 deltas $(\Delta x, \Delta y, \Delta w, \Delta h)$.

### Aplicacion de los deltas a los anchors

$$x_{prop} = x_a + \Delta x \cdot w_a, \quad y_{prop} = y_a + \Delta y \cdot h_a$$
$$w_{prop} = w_a \cdot e^{\Delta w}, \quad h_{prop} = h_a \cdot e^{\Delta h}$$

La parametrizacion log para $w, h$ es estable, simetrica (2x y 0.5x estan equidistantes en log-espacio) y mantiene los tamanos positivos.

### Filtrado: de 160k anchors a ~1000 propuestas

1. Aplicar deltas a los anchors.
2. Clip a los bordes de la imagen.
3. Filtrar cajas degeneradas ($w < 1$ o $h < 1$).
4. Pre-NMS top-1000 por nivel (por score de objectness).
5. **NMS** con threshold IoU 0.7.
6. Post-NMS top-1000-2000 globales.

### Loss de entrenamiento

$$L_{RPN} = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*)$$

- $L_{cls}$: binary cross-entropy (objeto vs fondo).
- $L_{reg}$: **smooth L1** (Huber loss) sobre los 4 deltas. Solo se aplica a anchors **positivos** (de ahi el factor $p_i^*$).

### Asignacion de etiquetas a anchors

| Condicion | Etiqueta |
| --- | --- |
| IoU > 0.7 con alguna GT **o** maximo IoU para esa GT | Positivo |
| IoU < 0.3 con todas las GT | Negativo |
| Otro caso | Ignorado |

Mini-batch balanceado: 256 anchors por imagen, ratio hasta 1:1 positivos:negativos.

---

## Pieza 3 — `roi_heads`: clasificacion y refinamiento final

```python
(roi_heads): RoIHeads(
    (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0','1','2','3'], output_size=(7,7), sampling_ratio=2)
    (box_head): TwoMLPHead(
        (fc6): Linear(in_features=12544, out_features=1024)
        (fc7): Linear(in_features=1024, out_features=1024)
    )
    (box_predictor): FastRCNNPredictor(
        (cls_score): Linear(in_features=1024, out_features=91)
        (bbox_pred): Linear(in_features=1024, out_features=364)
    )
)
```

### 3.1 — `box_roi_pool`: MultiScaleRoIAlign

Para cada propuesta de la RPN:

1. **Asignacion a nivel** de la piramide: $k = \lfloor 4 + \log_2(\sqrt{wh}/224) \rfloor$. Propuestas pequenas -> $P_2$, grandes -> $P_5$.
2. **RoIAlign**: extrae un tensor de tamano fijo **(256, 7, 7)** con interpolacion bilineal **sin cuantizacion**. Aporta ~+1.3 box AP vs RoIPool clasico (paper [Mask R-CNN](/papers/mask-rcnn-he-2017/)).

### 3.2 — `box_head`: TwoMLPHead

Convierte el tensor (256, 7, 7) en un vector de 1024 dim:

- Aplanar: 256 × 7 × 7 = 12544.
- Linear(12544, 1024) + ReLU.
- Linear(1024, 1024) + ReLU.

### 3.3 — `box_predictor`: FastRCNNPredictor

Dos cabezas lineales paralelas:

- **`cls_score`** = `Linear(1024, 91)`: clasificacion multi-clase (90 categorias COCO + 1 background).
- **`bbox_pred`** = `Linear(1024, 364)` (= 91 × 4): regresion refinada **por clase**. Cada clase aprende sus propios refinamientos tipicos (personas verticales, autos horizontales, etc.).

### Post-procesamiento (inferencia)

1. Aplicar deltas a las propuestas.
2. Softmax sobre logits de clase.
3. Filtrar por score (default 0.05 en torchvision, 0.5-0.9 en el lab).
4. **NMS por clase**: para que un perro y un gato proximos no se eliminen mutuamente.
5. Top-K final (default K=100).

---

## Output final del modelo

```python
predictions = model(images)
# predictions es una lista de dicts, uno por imagen:
predictions[0] = {
    'boxes':  Tensor[N, 4],   # (x1, y1, x2, y2) en pixeles imagen original
    'labels': Tensor[N],      # indices de clase (1-90 para COCO; 0 = background, no aparece)
    'scores': Tensor[N]       # probabilidad de la clase predicha, ordenadas descendentemente
}
```

donde N es el numero de detecciones que sobrevivieron al filtro.

---

## Resumen visual

```text
Imagen RGB (3, 800, 800)
    │
    │  ─── Pieza 0: transform ───
    │
    ▼
Imagen normalizada
    │
    │  ─── Pieza 1: backbone ───
    │  ResNet-50 -> 4 feature maps C2-C5
    │  FPN -> 5 feature maps uniformes P2-P6
    │
    ▼
{P2, P3, P4, P5, P6}
    │
    │  ─── Pieza 2: RPN ───
    │  Plantar ~160k anchors
    │  Predecir objectness + deltas
    │  NMS -> ~1000-2000 propuestas
    │
    ▼
~1000-2000 propuestas (cajas con score de objectness, sin clase)
    │
    │  ─── Pieza 3: roi_heads ───
    │  MultiScaleRoIAlign -> tensores (256, 7, 7) por propuesta
    │  TwoMLPHead -> vectores de 1024
    │  FastRCNNPredictor -> 91 clases + 91*4 deltas refinados
    │  Filtro score + NMS por clase
    │
    ▼
~5-30 detecciones finales: {'boxes', 'labels', 'scores'}
```

---

## Por que importa entender esto antes del fine-tuning

Cuando en la **Parte 2** del lab reemplacemos solo el `FastRCNNPredictor` para detectar mapaches (91 clases -> 2 clases), todo el resto del modelo se mantiene con los pesos pre-entrenados de COCO:

- El backbone ya sabe extraer features ricos del mapache (pelaje, ojos, postura).
- La FPN ya combina escalas correctamente.
- La RPN ya propone cajas alrededor de cosas que parecen objetos.
- El `box_head` ya produce vectores discriminativos.

Lo unico que falta aprender desde cero son **~10.000 parametros** de las dos `Linear` finales — 0.025% del modelo. Por eso 200 imagenes son suficientes.
