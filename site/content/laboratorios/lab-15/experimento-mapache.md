---
title: "Experimento didactico: el modelo COCO no conoce mapaches"
weight: 30
math: true
---

La Parte 2 del lab empieza con un experimento corto pero **muy revelador**: pasarle una imagen de mapache al modelo Faster R-CNN entrenado en COCO y ver que pasa. Esto motiva por que necesitamos fine-tuning y como funciona el transfer learning.

## El setup

```python
!wget https://raw.githubusercontent.com/bing0037/Raccoon_dataset/master/images/raccoon-1.jpg -O raccoon_example.jpg
object_detection_api('./raccoon_example.jpg', threshold=0.9)
```

Carga una foto del dataset Raccoon de bing0037 (la primera imagen, raccoon-1.jpg) y la pasa por la `object_detection_api` definida en la [Parte 1](inferencia-coco) con threshold alto (0.9).

## El resultado

### Con `threshold=0.9`: nada detectado

![Mapache sin detecciones (modelo COCO, threshold 0.9)](/laboratorios/lab-15/raccoon-coco-threshold-09.jpg)

La imagen aparece sin cajas verdes. Ninguna deteccion supera el umbral 0.9.

### Con `threshold=0.5`: **bear:0.87**

![Mapache clasificado como 'bear:0.87' (modelo COCO, threshold 0.5)](/laboratorios/lab-15/raccoon-coco-threshold-05-bear.jpg)

Bajando el threshold a 0.5, aparece una caja verde con la etiqueta **`bear:0.87`**. El modelo predice "oso" sobre el mapache, con confianza alta (87%).

Este resultado es la **mejor demostracion empirica** del fenomeno: el backbone reconoce que es un animal de tamano mediano con pelaje y postura cuadrupeda, y el clasificador final — al no tener "raccoon" en su vocabulario — elige la categoria mas parecida fenotipicamente entre sus 80 opciones: **bear**.

## Por que esto es educativo

El resultado demuestra empiricamente **dos cosas conceptualmente importantes**:

### 1. Faster R-CNN es un detector closed-set

El modelo **no puede decir "esto es algo que no conozco"**. Solo puede elegir entre las 91 clases (incluyendo background) con las que fue entrenado. Cuando ve un mapache, busca la categoria mas parecida fenotipicamente entre sus 80 opciones reales:

- `bear`: similar tamano, pelaje denso, postura cuadrupeda. **Gana** con score 0.5x.
- `cat`: pelo, pero proporciones distintas.
- `dog`: postura similar, pero hocico alargado vs hocico corto del mapache.
- `teddy bear`: pelaje similar, pero postura tipicamente sentada en juguete.

Score 0.5 es una **alerta interna**: el modelo dice "no estoy seguro, pero esta es la mejor opcion". Con scores >0.95 (como las zebras) el modelo esta muy confiado. 0.5 es la zona de duda.

### 2. El backbone ya conoce los mapaches

Aunque el clasificador final no tiene la etiqueta correcta, **el resto del modelo si extrae features correctos del mapache**:

- **Backbone (ResNet + FPN)**: detecta pelaje, ojos redondos, postura cuadrupeda, tamano mediano.
- **RPN**: genera una propuesta de caja correctamente localizada alrededor del mapache.
- **`box_head`**: produce un vector de 1024 dim que representa al mapache.

Lo unico que falla es el **mapping final** del vector a una etiqueta. El `cls_score` es una `Linear(1024, 91)` que solo tiene 91 outputs, y "raccoon" no es uno de ellos.

## La estrategia del fine-tuning

Esto motiva tres decisiones de diseno:

1. **No tocar el backbone** (mucho): ya sabe ver al mapache. Solo congelaremos las capas iniciales (`trainable_backbone_layers=3` deja libres las semanticas).
2. **Reemplazar el clasificador final** (`FastRCNNPredictor`): de 91 clases -> 2 clases (background + raccoon). Esta es la unica pieza que **borramos y empezamos desde cero**.
3. **Entrenar con pocas imagenes** (200 mapaches): posible solo gracias a que el resto del modelo ya esta aprendido. Si entrenaramos desde cero con 200 imagenes, seria un desastre.

```text
Modelo COCO (Parte 1):                          Modelo fine-tuneado (Parte 2):

backbone (COCO)                                 backbone (COCO, capas iniciales congeladas)
    │                                               │
    ▼                                               ▼
fpn (COCO)                                      fpn (COCO)
    │                                               │
    ▼                                               ▼
rpn (COCO)                                      rpn (COCO)
    │                                               │
    ▼                                               ▼
roi_heads:                                      roi_heads:
    box_roi_pool                                   box_roi_pool
    box_head (COCO)                                box_head (COCO)
    box_predictor                       ─►         box_predictor
        cls_score: Linear(1024, 91)                   cls_score: Linear(1024, 2)    ⚡ aleatorio
        bbox_pred: Linear(1024, 364)                  bbox_pred: Linear(1024, 8)    ⚡ aleatorio
```

## La leccion de transfer learning

El experimento del mapache es la mejor demostracion concreta de **por que funciona el transfer learning**:

> Las primeras capas de cualquier CNN entrenada en imagenes naturales aprenden **features universales** (bordes, texturas, partes). Cualquier dataset visual se beneficia de ese conocimiento. Solo las capas finales necesitan especializarse en la tarea concreta.

El paper [Yosinski et al. 2014 - Transferable Features](/papers/transferable-features-yosinski-2014) cuantifico esto: las primeras capas son transferibles entre tareas casi sin perdida, las ultimas son cada vez mas especificas.

## Cuanto entrenamos en la Parte 2

Despues del reemplazo del clasificador, los parametros del modelo se reparten asi:

| Componente | Parametros aprox. | Estado |
| --- | --- | --- |
| Stem + layer1 (ResNet) | ~250k | **Congelados** (no entrenan) |
| layer2 + layer3 + layer4 (ResNet) | ~23M | Entrenables con LR bajo |
| FPN | ~3M | Entrenables |
| RPN | ~600k | Entrenables |
| box_head (FCs 12544->1024->1024) | ~14M | Entrenables |
| **box_predictor (nuevo)** | **~10k** | **Aleatorio, entrenable** |

Total entrenable: **~40M parametros**. De esos, solo **~10k son nuevos**. El otro 99.975% viene pre-entrenado y solo se ajusta ligeramente.

Esa es la esencia: **aprovechar al maximo lo que ya sabes, aprender lo minimo necesario**.

## Sigue: el dataset

El siguiente paso es preparar los datos para el fine-tuning. Ver [Dataset Raccoon y DataLoader custom](dataset-y-dataloader).
