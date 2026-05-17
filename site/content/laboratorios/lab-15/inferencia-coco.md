---
title: "Parte 1: Inferencia con Faster R-CNN pre-entrenado en COCO"
weight: 20
math: true
---

La primera mitad del lab usa el modelo **sin modificarlo**: solo lo carga, le pasa imagenes y observa que detecta. Esto sirve para familiarizarse con la API y entender que clases conoce el modelo antes de pensar en fine-tuning.

## Setup inicial

### Imports y device

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import torchvision.transforms as T
import cv2, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)   # 'cuda' si Colab tiene GPU habilitada
```

`torchvision.ops.nms` es la implementacion CUDA optimizada de Non-Maximum Suppression. `FastRCNNPredictor` es la clase que reemplazaremos en la Parte 2.

### Instanciar el modelo

```python
frcnn_model = fasterrcnn_resnet50_fpn(
    pretrained=True,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=3
)
```

Argumentos relevantes:

| Argumento | Significado |
| --- | --- |
| `pretrained=True` | Descarga pesos pre-entrenados en COCO 2017 (~160 MB cacheado en `~/.cache/torch/hub/`) |
| `num_classes=91` | 80 clases COCO + 1 background + 10 huecos `'N/A'` |
| `pretrained_backbone=True` | Backbone con pesos ImageNet (redundante si `pretrained=True`) |
| `trainable_backbone_layers=3` | Deja entrenable layer2+layer3+layer4 (congela stem + layer1) |

El argumento `trainable_backbone_layers` controla cuantos stages se permite ajustar durante fine-tuning. Default `3` significa congelar **las capas iniciales** (que aprenden features universales como bordes y texturas) y dejar libres **las capas semanticas** (donde reside el conocimiento especifico de "objeto").

### `print(frcnn_model)` — recorrer la arquitectura

Imprime el arbol completo de submodulos. Salida abreviada:

```text
FasterRCNN(
  (transform): GeneralizedRCNNTransform(...)
  (backbone): BackboneWithFPN(
    (body): IntermediateLayerGetter(...)          # ResNet-50 sin avgpool/fc
    (fpn): FeaturePyramidNetwork(...)
  )
  (rpn): RegionProposalNetwork(
    (anchor_generator): AnchorGenerator()
    (head): RPNHead(...)
  )
  (roi_heads): RoIHeads(
    (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0','1','2','3'], output_size=(7,7), sampling_ratio=2)
    (box_head): TwoMLPHead(...)
    (box_predictor): FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=91)
      (bbox_pred): Linear(in_features=1024, out_features=364)
    )
  )
)
```

Las 4 piezas (transform, backbone, rpn, roi_heads) son las que cubrimos en detalle en [arquitectura](arquitectura).

### `eval()` + `to(device)`

```python
frcnn_model.eval()
frcnn_model = frcnn_model.to(device)
```

Dos efectos:

1. **`eval()`** cambia el comportamiento del `forward()`. En modo training devuelve el **dict de losses**; en modo eval devuelve **las detecciones**. Tambien desactiva el efecto de `Dropout` (no usado aqui) y el update de `BatchNorm` (aqui es FrozenBN igual).
2. **`to(device)`** mueve todos los parametros a GPU. Necesario para procesar tensores que esten en GPU sin error de device mismatch.

## La lista `CATEGORY_NAMES` (clases COCO)

```python
CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    # ... 91 entradas en total ...
]
```

**91 entradas en total**:

- Indice 0: `__background__` (no aparece en predicciones, pero el modelo lo conoce).
- Indices 1-90: 80 categorias reales + 11 `'N/A'` (huecos de IDs descartados al crear COCO).

Los huecos vienen del hecho de que el dataset original [COCO 2014](/papers/coco-lin-2014/) propuso 91 categorias pero descarto 11 durante la anotacion (hat, shoe, plate, mirror, window, etc.). **Los IDs nunca se renumeraron** para mantener compatibilidad.

## `get_prediction(img_path, threshold)`

La funcion nucleo de la inferencia del lab:

```python
def get_prediction(img_path, threshold):
    img = Image.open(img_path)                          # cargar con PIL
    img = T.ToTensor()(img).to(device)                  # a tensor (3, H, W) float32 [0,1] en GPU
    pred = frcnn_model([img])                           # forward del modelo

    pred_class = [CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    pred_t = pred_t[-1] if len(pred_t) > 0 else -1
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]
    return pred_boxes, pred_class, pred_score
```

### Que devuelve el modelo

`frcnn_model([img])` recibe una **lista** (no un tensor) y devuelve una lista de dicts:

```python
pred = [
    {
        'boxes':  Tensor[N, 4],   # (x1, y1, x2, y2) en pixeles imagen original
        'labels': Tensor[N],      # int64, indices de clase
        'scores': Tensor[N]       # float32, probabilidades ordenadas descendentemente
    }
]
```

Las N detecciones ya pasaron por NMS por clase, score threshold interno (0.05) y top-K (100) dentro del modelo.

### Filtrado por threshold

El bloque final filtra las detecciones aprovechando que **ya vienen ordenadas por score**. Encuentra el ultimo indice con `score > threshold` y trunca las listas. Es funcional pero algo fragil: si torchvision cambiara el orden en el futuro, podria descartar detecciones validas.

Forma mas robusta:

```python
mask = [s > threshold for s in pred_score]
pred_boxes = [b for b, m in zip(pred_boxes, mask) if m]
```

## `object_detection_api(img_path, threshold, ...)`

```python
def object_detection_api(img_path, threshold=0.5, rect_th=1, text_size=0.5, text_th=2):
    boxes, pred_cls, pred_scrs = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])),
                      (int(boxes[i][1][0]), int(boxes[i][1][1])),
                      color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i] + ':' + "{:.2f}".format(pred_scrs[i]),
                    (int(boxes[i][0][0]), int(boxes[i][0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([]); plt.yticks([])
    plt.show()
```

Llama a `get_prediction`, recarga la imagen con OpenCV, **convierte BGR a RGB** (porque OpenCV lee en BGR por defecto, una convencion historica de Intel IPL), dibuja rectangulos verdes con etiquetas y muestra con matplotlib.

## Las 3 imagenes de prueba

```python
!wget https://ichef.bbci.co.uk/.../zebras.jpg.webp -O zebras.jpg
!wget https://mundoenlinea.cl/.../Teletrabajo.jpg -O ofice.jpg
!wget https://img.freepik.com/.../multicultural-group.jpg -O eating.jpg
```

| Imagen | Threshold | Detecciones esperadas |
| --- | --- | --- |
| `zebras.jpg` | 0.7 | 2-4 `zebra` con scores ~0.97 |
| `ofice.jpg` | 0.9 | `person`, `laptop`/`tv`, `cup`, `chair`, `keyboard`, posiblemente `cell phone`, `book`, `mouse` |
| `eating.jpg` | 0.9 | varios `person`, `dining table`, `cup`, `wine glass`, `bottle`, comidas |

Por que cada imagen usa thresholds distintos:

- **Zebras**: pocos objetos grandes y claros. Threshold bajo (0.7) deja margen y todas las detecciones tienen scores altisimos.
- **Oficina y comida**: escenas densas con muchos objetos. Threshold 0.9 filtra las detecciones dudosas que el modelo siempre intenta proponer.

## Resultados reales

### Zebras (threshold 0.7)

![Deteccion de cebras con Faster R-CNN COCO](/laboratorios/lab-15/coco-inference-zebras.jpg)

El modelo detecta las cebras con scores altisimos (~0.99). Las cajas verdes encuadran cada cebra individualmente — incluso las que aparecen muy juntas se distinguen porque pertenecen a la misma clase y el NMS por clase las trata como instancias separadas.

### Oficina (threshold 0.9)

![Deteccion en escena de oficina](/laboratorios/lab-15/coco-inference-office.jpg)

Una escena densa con multiples objetos. Con threshold alto, el modelo solo reporta las detecciones mas confiables: `person`, `laptop`, `cup`, `chair`, etc. Si bajas el threshold a 0.5 veras decenas de detecciones mas (algunas validas, otras dudosas).

### Gente comiendo (threshold 0.9)

![Deteccion de personas comiendo](/laboratorios/lab-15/coco-inference-eating.jpg)

Multiples personas + objetos en la mesa. Demuestra el manejo de **detecciones densas** donde las cajas se solapan mucho. NMS por clase evita que detecciones del mismo objeto se eliminen entre si.

## Observaciones evidentes

Tres patrones se hacen claros con estos resultados:

1. **El score como medida de confianza**: scores >0.95 son robustos, 0.7-0.95 plausibles, 0.5-0.7 dudosos.
2. **Errores tipicos**: doble deteccion (mismo objeto detectado dos veces si esta parcialmente ocluido), clase incorrecta pero cercana (donut como pizza, wine glass como cup).
3. **Objetos pequenos perdidos**: un mouse o un boligrafo pueden no detectarse si threshold es alto. FPN ayuda pero no es magico.

## Limites de COCO

Lo mas educativo: ver **lo que el modelo no detecta**.

- **Servilleta**: no es categoria COCO -> ignorada.
- **Smartphone**: COCO la llama `cell phone` -> detectada.
- **Sushi**: no es categoria COCO -> podria caer como `pizza` o no detectarse.
- **Lentes**: era categoria descartada (`'N/A'`) -> nunca detectada.

Esto motiva la **Parte 2** del lab: para detectar **mapaches** (clase **no presente en COCO**), hay que hacer [fine-tuning](fine-tuning-setup).
