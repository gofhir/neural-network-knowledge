---
title: "Feature Pyramid Networks (FPN)"
weight: 52
math: true
---

{{< paper-card
    title="Feature Pyramid Networks for Object Detection"
    authors="Lin, Dollar, Girshick, He, Hariharan, Belongie"
    year="2017"
    venue="CVPR 2017"
    pdf="/papers/fpn-lin-2017.pdf"
    arxiv="1612.03144" >}}
Construye una **piramide de feature maps con semantica fuerte en todos los niveles** combinando bottom-up (la backbone normal) con un top-down enriquecido y lateral connections. Resuelve el problema de deteccion multi-escala con coste marginal y se convierte en componente estandar de Faster R-CNN, Mask R-CNN, RetinaNet y todos los detectores modernos.
{{< /paper-card >}}

---

## El problema

Las CNNs producen una jerarquia natural de feature maps con strides crecientes ($C_2, C_3, C_4, C_5$ con strides 4, 8, 16, 32). Pero hay un **trade-off** entre resolucion y semantica:

- $C_2$ (alta resolucion): bordes, texturas. **Pobre semantica** -> mal para clasificar.
- $C_5$ (alta semantica): conceptos abstractos. **Pobre resolucion** -> objetos pequenos desaparecen.

Soluciones previas:

- **Pirámide de imagenes** (R-CNN clasico): redimensionar la imagen a multiples escalas y procesar cada una. Lento (~4x).
- **Single feature map** (Fast/Faster R-CNN): solo $C_5$. Falla con objetos pequenos.
- **Pirámide de features sin fusion** (SSD): predecir en cada nivel sin enriquecerlos. Niveles bajos siguen siendo semanticamente pobres.

## Ideas principales

- **Bottom-up pathway**: la backbone CNN estandar (ResNet) produce los feature maps $\{C_2, C_3, C_4, C_5\}$.
- **Top-down pathway**: empezando de $C_5$, **upsampling 2x con vecino mas cercano** (sin parametros) para alucinar mapas de alta resolucion con semantica fuerte.
- **Lateral connections**: para cada nivel bottom-up, aplicar **conv 1x1** que iguala canales (todos a 256) y **sumar elemento a elemento** con el top-down upsampleado.
- **Conv 3x3 final** en cada nivel para suavizar artefactos del upsampling.
- Resultado: piramide $\{P_2, P_3, P_4, P_5\}$ con **256 canales uniformes** y **semantica fuerte en todos los niveles**.
- **Cabezas compartidas**: el mismo RPN y RoI head se aplican a todos los niveles. Esto refleja la creencia de que la semantica es similar entre niveles, solo cambia la escala.
- **Asignacion de RoIs a niveles**: formula $k = \lfloor 4 + \log_2(\sqrt{wh}/224) \rfloor$. Propuestas pequenas -> $P_2$ (resolucion); grandes -> $P_5$ (semantica).
- **Anchors por nivel**: 1 escala por nivel ($32^2, 64^2, 128^2, 256^2, 512^2$ para $P_2$-$P_6$) $\times$ 3 aspect ratios = 15 anchors a traves de la piramide (vs 9 apilados en una sola escala del Faster R-CNN original).

## Resultados (Tabla 1 del paper)

Sobre COCO, recall de propuestas RPN:

| Variante | AR@1k | AR_s (pequenos) |
| --- | --- | --- |
| RPN single-scale $C_4$ | 47.4 | 14.4 |
| RPN single-scale $C_5$ | 47.5 | 14.5 |
| **FPN completo** | **56.3** (+8) | **27.4** (+12.9) |
| FPN sin top-down | 50.0 | 23.4 |
| FPN sin laterales | 46.5 | — |

Para deteccion completa (Faster R-CNN + ResNet-50):

| Backbone | mAP@[.5:.95] |
| --- | --- |
| ResNet-50 single-scale | 31.9 |
| **ResNet-50 + FPN** | **36.2** (+4.3) |

## Impacto

FPN se convirtio inmediatamente en estandar:

- **Mask R-CNN** (2017): usa FPN por defecto.
- **RetinaNet** (2017): one-stage construido sobre FPN.
- **EfficientDet** (2019): generaliza FPN a BiFPN bidireccional.
- **YOLOv4, v5, v8**: variantes de FPN (PANet, etc.).

En `torchvision.models.detection` todos los modelos llevan **"fpn"** explicito en el nombre. Es el componente que los hace competitivos.

## Conexion con el laboratorio

En el `print(frcnn_model)` del lab veras:

```python
(fpn): FeaturePyramidNetwork(
    (inner_blocks): ModuleList(   # las 4 laterales 1x1
        (0): Conv2d(256,  256, kernel_size=(1, 1))   # para C2
        (1): Conv2d(512,  256, kernel_size=(1, 1))   # para C3
        (2): Conv2d(1024, 256, kernel_size=(1, 1))   # para C4
        (3): Conv2d(2048, 256, kernel_size=(1, 1))   # para C5
    )
    (layer_blocks): ModuleList(   # las 4 convs 3x3 suavizadoras
        (0-3): Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
    )
    (extra_blocks): LastLevelMaxPool()   # anade P6 desde P5 (extension de torchvision)
)
```

Y la asignacion a niveles aparece en `(box_roi_pool): MultiScaleRoIAlign(featmap_names=['0','1','2','3'], output_size=(7,7), sampling_ratio=2)` que implementa la formula del paper.
