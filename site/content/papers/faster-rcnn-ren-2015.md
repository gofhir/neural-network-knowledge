---
title: "Faster R-CNN"
weight: 51
math: true
---

{{< paper-card
    title="Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
    authors="Ren, He, Girshick, Sun"
    year="2015"
    venue="NeurIPS 2015 / TPAMI 2017"
    pdf="/papers/faster-rcnn-ren-2015.pdf"
    arxiv="1506.01497" >}}
Elimina el cuello de botella de propuestas externas (Selective Search) introduciendo la **Region Proposal Network** (RPN), una red totalmente convolucional que comparte features con el detector. Reduce el costo de propuestas de ~1500 ms a ~10 ms por imagen y logra 5 fps end-to-end con VGG-16, ganador de ILSVRC y COCO 2015.
{{< /paper-card >}}

---

## Ideas principales

- **Problema previo**: Fast R-CNN (2015) ya compartia el computo CNN entre regiones, pero las propuestas se generaban con **Selective Search** en CPU (~2 s/imagen). El detector tardaba 200 ms en GPU pero las propuestas dominaban el costo total.
- **RPN**: una red totalmente convolucional que opera sobre el feature map del backbone y predice simultaneamente **objectness** (binario, objeto vs fondo) y **deltas de caja** $(\Delta x, \Delta y, \Delta w, \Delta h)$ para cada anchor.
- **Anchors**: $k = 9$ cajas de referencia preplantadas (3 escalas $\times$ 3 aspect ratios) en cada posicion del feature map. La red predice **offsets relativos al anchor**, no coordenadas absolutas. Esto hace la prediccion **invariante a la traslacion** y mucho mas estable que regresion desde cero.
- **Pirámide de anchors** como reemplazo de pirámides de imagenes (lentas) o pirámides de filtros (DPM clasico): una única escala de imagen + un único feature map + 9 anchors por celda.
- **Asignacion de etiquetas**: anchor positivo si IoU > 0.7 con alguna GT o si es el de mayor IoU para esa GT; negativo si IoU < 0.3; ignorado en medio. Mini-batch balanceado de 256 anchors (1:1 positivo:negativo).
- **Multi-task loss**: $L = L_{cls} + \lambda \cdot p^* \cdot L_{reg}$ con $L_{cls}$ binary cross-entropy y $L_{reg}$ **smooth L1** sobre los 4 deltas (solo para anchors positivos).
- **Parametrizacion log para escalas**: $t_w = \log(w / w_a)$, $t_h = \log(h / h_a)$. Convierte la regresion multiplicativa a aditiva en log-espacio, simetrica respecto a 2x y 0.5x.
- **4-step alternating training**: paso 1 entrena RPN, paso 2 entrena Fast R-CNN con propuestas del paso 1, paso 3 re-entrena RPN con backbone congelado, paso 4 ajusta Fast R-CNN con backbone congelado. Resultado: una red unificada con backbone compartida.
- **NMS sobre propuestas RPN**: threshold IoU 0.7 reduce ~20k anchors crudos a ~2000 propuestas, luego top-300 para alimentar al detector.

## Resultados clave

| Sistema | VOC 2007 mAP | COCO mAP@[.5:.95] | Velocidad |
| --- | --- | --- | --- |
| SS + Fast R-CNN (VGG) | 70.0% | 19.7% | 0.5 fps |
| **RPN + Fast R-CNN (VGG)** | **73.2%** (07+12) | **21.9%** | **5 fps** |
| RPN + Fast R-CNN (ZF) | — | — | 17 fps |

- En VOC 2007 con datos COCO+VOC: 78.8% mAP.
- 1er lugar en ImageNet Detection 2015, ImageNet Localization 2015, COCO Detection 2015.
- Reduce el paso de propuestas de **1510 ms a 10 ms** (150x).

## Limitaciones reconocibles

- Single-scale feature map limita la deteccion de objetos muy pequenos -> resuelto por **FPN** (2017).
- **RoI Pooling** cuantiza dos veces, perdiendo precision sub-pixel -> resuelto por **RoIAlign** en Mask R-CNN.
- Anchors hard-coded requieren elegir escalas/aspect ratios manualmente -> eliminados por **DETR** (2020).

## Conexion con el laboratorio

`torchvision.models.detection.fasterrcnn_resnet50_fpn` instancia una version moderna del modelo del paper con tres cambios respecto al original 2015:

1. Backbone **ResNet-50 + FPN** en vez de VGG-16/ZF (mayor capacidad y multi-escala).
2. **MultiScaleRoIAlign** en vez de RoIPool (heredado de Mask R-CNN).
3. **Joint training** en vez de 4-step alternating (mas simple, similar accuracy).

Todo lo demas del paper aplica directamente: anchors, RPN head con dos cabezas 1x1 (cls + reg), parametrizacion log, smooth L1, NMS por clase.
