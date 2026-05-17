---
title: "Mask R-CNN"
weight: 53
math: true
---

{{< paper-card
    title="Mask R-CNN"
    authors="He, Gkioxari, Dollar, Girshick"
    year="2017"
    venue="ICCV 2017"
    pdf="/papers/mask-rcnn-he-2017.pdf"
    arxiv="1703.06870" >}}
Extiende Faster R-CNN con una **rama de segmentacion paralela** y reemplaza RoI Pool por **RoIAlign** sin cuantizacion. Ganador del **ICCV 2017 Best Paper Award** (Marr Prize), gana COCO 2016/2017 en deteccion, segmentacion de instancias y keypoints simultaneamente. RoIAlign aporta tambien ~+1.3 box AP "gratis" a Faster R-CNN.
{{< /paper-card >}}

---

## La pregunta que responde

Antes de 2017 habia dos tareas con frameworks distintos:

- **Deteccion** (Faster R-CNN): bounding boxes + clases.
- **Segmentacion semantica** (FCN, DeepLab): clase por pixel, sin distinguir instancias.

Faltaba la tercera: **segmentacion de instancias** = bbox + mascara de pixeles + distinguir cada instancia individual.

## Ideas principales

- **Arquitectura**: Faster R-CNN con una **tercera rama** en el RoI head, en paralelo a clasificacion y regresion. La rama predice $K$ mascaras binarias $m \times m$ (una por clase) usando un mini-FCN.
- **Decoupling de mascara y clase**: cada clase tiene su propia mascara con **sigmoid + BCE binario** (no softmax multi-clase). La clase se decide en la rama de clasificacion, y se selecciona la mascara correspondiente. Esto elimina la competencia entre clases en la rama de mascara.
- **RoIAlign**: el corazon del paper. Reemplaza RoIPool eliminando **dos cuantizaciones**:
  1. RoIPool cuantizaba las coordenadas del RoI al stride del feature map (perdida de ~16 px con stride 16).
  2. RoIPool cuantizaba los bordes de cada bin de la grilla 7x7.
- **Interpolacion bilineal**: para cada bin de la grilla, define 4 puntos de muestreo (sampling_ratio=2 -> 2x2) en coordenadas float. Cada punto se calcula con interpolacion bilineal de los 4 pixeles vecinos del feature map. Promedia o max-pool los 4 puntos.
- **Generalidad demostrada**: cambiando solo la rama final, **Mask R-CNN tambien hace keypoint detection** tratando cada keypoint como una mascara one-hot. Gano COCO Keypoints 2017.

## RoIAlign — impacto cuantitativo (Tabla 2c)

| Metodo | mask AP | mask AP@0.5 | mask AP@0.75 |
| --- | --- | --- | --- |
| RoIPool | 26.9 | 48.8 | 26.4 |
| RoIWarp | 27.2 | 49.2 | 27.1 |
| **RoIAlign** | **30.2** | 51.0 | **31.8** |

- **+3.3 puntos de AP** solo cambiando la operacion de extraccion.
- **+5.4 puntos en AP@0.75** (metrica de localizacion estricta).
- Para deteccion (no segmentacion): **+1.3 box AP** gratis.

## Resultados completos (Tabla 1)

COCO test-dev, segmentacion de instancias:

| Modelo | Backbone | mask AP | AP_S |
| --- | --- | --- | --- |
| MNC (2015 winner) | ResNet-101-C4 | 24.6 | 4.7 |
| FCIS+++ (2016 winner) | ResNet-101-C5-dilated | 33.6 | — |
| **Mask R-CNN** | ResNet-101-FPN | **35.7** | 15.5 |
| **Mask R-CNN** | ResNeXt-101-FPN | **37.1** | 16.9 |

Supera al ganador 2016 sin bells & whistles (sin multi-scale train/test, sin OHEM, sin ensembles).

## Conexion con el laboratorio

El lab usa `fasterrcnn_resnet50_fpn`, **no** `maskrcnn_resnet50_fpn`. No tiene rama de mascara. Pero **hereda RoIAlign** via `MultiScaleRoIAlign`:

```python
(box_roi_pool): MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=(7, 7),
    sampling_ratio=2   # 2x2 = 4 puntos por bin
)
```

Es parte del porque los resultados del lab son mejores que el Faster R-CNN original de 2015: la implementacion moderna se beneficia de RoIAlign aunque sea para deteccion sola.

## Filosofia de diseno

Mask R-CNN es un caso de estudio en **decoupling** como motor de progreso en deep learning:

- Desacopla clasificacion de segmentacion (sigmoid binario por clase, no softmax multi-clase).
- Desacopla localizacion de extraccion (RoIAlign elimina la cuantizacion).

Y se beneficia del decoupling previo de Faster R-CNN (RPN separado del detector, backbone compartida pero cabezas distintas). Esto permite usarlo como base para **Mask R-CNN, Keypoint R-CNN, Cascade Mask R-CNN, HTC, DetectoRS** cambiando solo las cabezas. Misma filosofia aplica al fine-tuning del lab: reemplazamos solo `FastRCNNPredictor`.
