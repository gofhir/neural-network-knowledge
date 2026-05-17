---
title: "Reemplazo del clasificador: el corazon del fine-tuning"
weight: 50
math: true
---

Esta es la celda mas importante de toda la Parte 2 del lab. **Aqui pasa la "magia" del transfer learning**: reemplazar **solo el clasificador final** del modelo, manteniendo todo lo demas con los pesos pre-entrenados de COCO.

## El codigo

```python
# Instanciamos Faster RCNN preentrenado
frcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)

# Obtenemos el tamano de entrada que tiene el clasificador de Faster RCNN
in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features

# Definimos cuantas clases queremos (mapaches + background)
num_classes = 2

# Reemplazamos el clasificador
frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
frcnn_model.to(device)
```

5 lineas, pero cada una hace algo conceptualmente importante.

---

## Linea 1: re-instanciar el modelo pre-entrenado

```python
frcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
```

### Por que de nuevo?

Ya cargamos `frcnn_model` en la Parte 1 con configuracion COCO. En la Parte 2 lo recargamos **desde un punto de partida limpio** porque en la Parte 1 fuiste modificando su estado (`eval()`, `.to(device)`). Re-instanciar lo deja con configuracion default.

**No vuelve a descargar pesos**: torchvision los cachea en `~/.cache/torch/hub/checkpoints/` despues de la primera descarga. Solo construye la arquitectura y carga del cache. Tiempo: ~2-3 segundos.

### Diferencia con la version de la Parte 1

```python
# Parte 1 (Celda 8) - explicito:
frcnn_model = fasterrcnn_resnet50_fpn(
    pretrained=True, progress=True, num_classes=91,
    pretrained_backbone=True, trainable_backbone_layers=3
)

# Parte 2 (Celda 70) - mas corto:
frcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
```

La Parte 2 omite `trainable_backbone_layers`. Esto **usa el default de torchvision, que es 3** (igual que la Parte 1). Misma configuracion, escrita mas corta.

---

## Linea 2: extraer el tamano de entrada del clasificador

```python
in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
```

### Navegacion por dot notation

```text
frcnn_model                          # FasterRCNN
    .roi_heads                       # RoIHeads
        .box_predictor               # FastRCNNPredictor
            .cls_score               # Linear(1024, 91)
                .in_features         # → 1024 (int)
```

`in_features` es un atributo de `torch.nn.Linear` que guarda el tamano de entrada (el primer argumento del constructor `Linear(in, out)`).

### Por que leer dinamicamente

Para `fasterrcnn_resnet50_fpn`, `in_features` siempre es 1024. **Podrias escribir `in_features = 1024` directamente**, pero es fragil: si torchvision cambiara el tamano en una version futura tu codigo fallaria. Leer del modelo es robusto. Este es **un patron clasico** en PyTorch para reemplazar capas finales.

---

## Linea 3: definir el numero de clases

```python
num_classes = 2
```

**1 raccoon + 1 background = 2 clases en total**.

⚠️ **Para tu propio dataset con N clases reales**: `num_classes = N + 1`. Esta es una de las cosas a modificar y es **respuesta a la pregunta 1 de la tarea final** del notebook.

---

## Linea 4: el reemplazo del predictor

```python
frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

🔥 **Esta es la linea critica del fine-tuning.**

Crea un nuevo `FastRCNNPredictor(1024, 2)` y lo asigna como atributo a `roi_heads`, reemplazando el predictor pre-entrenado.

### Que hay dentro de `FastRCNNPredictor(1024, 2)`

Mirando el codigo fuente de torchvision:

```python
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas
```

Es decir, solo crea **dos capas lineales**:

| Capa | Antes (COCO) | Despues (Raccoon) |
| --- | --- | --- |
| `cls_score` | `Linear(1024, 91)` | `Linear(1024, 2)` |
| `bbox_pred` | `Linear(1024, 364)` (91x4) | `Linear(1024, 8)` (2x4) |

### Las nuevas capas se inicializan aleatoriamente

Las dos `Linear` recien creadas tienen pesos inicializados con **Kaiming init** (la inicializacion default de PyTorch para `Linear`). **Sus pesos NO heredan nada del modelo COCO.**

Esto es **intencional**:

- Las clases son distintas (91 -> 2), no hay forma de "heredar" mappings.
- Es el clasificador el que debe aprender a decir "esto es raccoon".

### Lo que NO cambia

Todo lo demas del modelo permanece **con los pesos pre-entrenados de COCO**:

| Componente | Estado despues del reemplazo |
| --- | --- |
| `transform` | sin pesos aprendibles |
| `backbone.body` (ResNet-50) | **pesos COCO**, capas iniciales congeladas |
| `backbone.fpn` | **pesos COCO** |
| `rpn.anchor_generator` | sin pesos |
| `rpn.head` (conv 3x3 + cls + reg) | **pesos COCO** |
| `roi_heads.box_roi_pool` | sin pesos |
| `roi_heads.box_head` (fc6, fc7) | **pesos COCO** |
| `roi_heads.box_predictor.cls_score` | ⚡ **aleatorio** (Linear(1024, 2)) |
| `roi_heads.box_predictor.bbox_pred` | ⚡ **aleatorio** (Linear(1024, 8)) |

### Cuantos parametros nuevos

- `cls_score`: $1024 \times 2 + 2 = 2.050$ parametros.
- `bbox_pred`: $1024 \times 8 + 8 = 8.200$ parametros.
- **Total nuevos**: ~10.000 parametros.

Comparado con los ~41M del modelo completo: **0.025%**. Eso es lo unico que aprendemos desde cero. **El otro 99.975% viene pre-entrenado.**

---

## Linea 5: mover el modelo a GPU

```python
frcnn_model.to(device)
```

⚠️ **Obligatorio despues del reemplazo**.

Cuando creas `FastRCNNPredictor(in_features, num_classes)`, las nuevas capas se construyen **en CPU por default**. Si no las mueves a GPU explicitamente, tendras un error clasico:

```text
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

Porque el resto del modelo esta en GPU (lo movimos antes) y las nuevas capas estan en CPU. `.to(device)` aplicado al modelo entero mueve **todos** los parametros (incluidos los nuevos).

---

## Verificacion

Despues de la celda puedes confirmar el reemplazo:

```python
print(frcnn_model.roi_heads.box_predictor)
```

Output esperado:

```text
FastRCNNPredictor(
    (cls_score): Linear(in_features=1024, out_features=2, bias=True)
    (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
)
```

`out_features=2` y `out_features=8` confirman que el reemplazo funciono.

---

## Por que funciona el transfer learning

Recuerda el experimento previo del [mapache](experimento-mapache):

> Con threshold 0.5, el modelo COCO predijo `bear:0.5x` sobre un mapache.

Eso significa:

1. **El backbone ya extrae features correctos del mapache** (pelaje, ojos, postura).
2. **El RPN ya genera buenas propuestas** alrededor del mapache.
3. **El `box_head` ya produce un vector representativo** del mapache.
4. **El unico problema fue el `box_predictor`**: como solo tenia 91 outputs y "raccoon" no era ninguno, eligio "bear" como el mas parecido.

Al reemplazar **solo el `box_predictor`** con uno nuevo de 2 salidas (raccoon + background), el modelo solo tiene que aprender:

> "Cuando el vector de 1024 dim parece un mapache, dispara la salida 1 (raccoon)."

Eso lo aprende rapido con 160 imagenes. Si hubieramos reseteado mas capas, tendriamos que ensenarle desde mas atras → mucho mas datos requeridos.

---

## Sigue: utilities del entrenamiento

Antes del loop de entrenamiento, el lab define varias funciones helper: IoU, NMS por clase, calculo de metricas, warmup del learning rate. Ver [Utilities del entrenamiento](utilities).
