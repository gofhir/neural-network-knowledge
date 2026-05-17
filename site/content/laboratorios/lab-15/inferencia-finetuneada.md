---
title: "Lanzamiento, resultados reales, inferencia post-fine-tuning"
weight: 80
math: true
---

Esta pagina cubre las ultimas celdas del lab: definir los thresholds + optimizer, ejecutar el entrenamiento real, cargar el mejor checkpoint y probar el modelo fine-tuneado sobre imagenes nuevas de mapaches.

## Hyperparameters + optimizer + lanzamiento

```python
SCORE_THRESHOLD = 0.5
IOU_OVERLAP_THRESHOLD = 0.5
IOU_POSITIVE_THRESHOLD = 0.7

params = [p for p in frcnn_model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

train_model(frcnn_model, train_loader, val_loader, optimizer, lr_scheduler, device)
```

### Los 3 thresholds

| Threshold | Valor | Uso |
| --- | --- | --- |
| `SCORE_THRESHOLD` | 0.5 | Score minimo para considerar una prediccion en `eval_epoch` |
| `IOU_OVERLAP_THRESHOLD` | 0.5 | Usado por NMS extra en `filter_by_class_nms` |
| `IOU_POSITIVE_THRESHOLD` | 0.7 | Umbral de matching prediction-vs-GT para contar TP |

⚠️ El lab usa **0.7** como threshold de matching (vs el estandar 0.5 de PASCAL VOC). Esto hace la evaluacion **mas estricta**: una prediccion necesita 70% de solape con la GT para contar como TP. Por eso las metricas del lab seran mas bajas que un benchmark COCO estandar.

### El optimizer: SGD con momentum + weight decay

```python
optimizer = torch.optim.SGD(
    params, 
    lr=0.005,           # learning rate base
    momentum=0.9,        # promedio ponderado de gradientes pasados
    weight_decay=0.0005  # regularizacion L2
)
```

Por que SGD y no Adam:

1. **Tradicion**: el paper Faster R-CNN usa SGD. Las recetas funcionan bien.
2. **SGD generaliza mejor** en vision: Adam converge mas rapido pero a soluciones mas "agudas" que pueden generalizar peor.
3. **Adam con LR alto rompe transfer learning**: los nuevos pesos aleatorios destrozan los pre-entrenados.

Para fine-tuning de detectores **SGD + momentum es el default seguro**.

### El scheduler StepLR

```python
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
```

Reduce el `lr` por un factor de `0.8` cada `2` epocas:

| Epoca | lr efectivo (descontando warmup) |
| --- | --- |
| 0 | 0.005 |
| 1 | 0.005 |
| 2 | 0.004 |
| 3 | 0.004 |

Con `epochs=4`, solo hay una reduccion a la mitad del entrenamiento.

⚠️ El comentario del codigo dice `decreases the learning rate by 0.7 every 2 epochs` pero el codigo dice `gamma=0.8`. Inconsistencia de comentario; el codigo manda.

### Solo parametros entrenables

```python
params = [p for p in frcnn_model.parameters() if p.requires_grad]
```

Filtra los parametros con `requires_grad=True`. Excluye:

- Stem + layer1 de ResNet (congelados por `trainable_backbone_layers=3`).
- Todos los `FrozenBatchNorm2d`.

De ~41M parametros totales, **~28M son entrenables**. El resto son layers congelados + estadisticas BN.

---

## Resultados reales del entrenamiento

Ejemplo de ejecucion con dataset Raccoon (160 train + 40 val), 4 epocas, GPU T4 en Colab. Tiempo total: **~5 minutos**.

| Epoca | Train loss (moving) | TP | FN | FP | Recall | Precision |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.357 | 30 | 16 | **153** | 65.2% | 16.4% |
| 1 | 0.134 | 35 | 11 | 44 | 76.1% | 44.3% |
| 2 | 0.095 | 36 | 10 | 21 | 78.3% | 63.2% |
| **3** | **0.084** | **36** | **10** | **19** | **78.3%** | **65.5%** ⭐ |

**Modelo guardado**: epoca 3 (mejor precision: 0.655 > 0.632 de epoca 2).

### La caida espectacular de Falsos Positivos: 153 → 19

La observacion mas reveladora. En la **epoca 0**, el modelo predijo "raccoon" 183 veces pero solo 30 eran reales. Estaba alucinando mapaches en sombras, arboles, todo lo que tuviera textura/forma cercana.

Causa: el `Linear(1024, 2)` recien inicializado **sesga** las predicciones — al principio asigna scores moderados a casi todo lo que pase el threshold, porque no ha aprendido la frontera de decision.

A medida que entrena, el clasificador aprende a **rechazar** propuestas que no son mapaches → FPs colapsan. Reduccion: -109, -23, -2. Curva clasica de saturacion: la mayor mejora viene en la primera epoca.

### Recall plafona en 78%

Desde epoca 2, el modelo detecta 36/46 mapaches y se queda ahi. Los 10 que nunca encuentra son probablemente:

- Mapaches muy pequenos (escala que ni FPN cubre bien).
- Mapaches parcialmente ocluidos.
- Mapaches con poses raras.
- Cajas GT con bordes ambiguos donde IoU < 0.7 (el threshold estricto del lab).

Con el threshold estandar de 0.5 (no 0.7), el recall seguramente seria >90%.

### Train loss: 0.357 → 0.084

Caida del 76%. Convergencia muy rapida. En 4 epocas × 27 batches = 108 iteraciones, el modelo ya esta cerca del optimo local. Senal de:

- Transfer learning funcionando perfecto.
- Warmup hizo su trabajo (0 NaNs).
- `lr=0.005` apropiado para fine-tuning con reset del clasificador.

### F1 del modelo final

$$F_1 = 2 \cdot \frac{P \cdot R}{P + R} = 2 \cdot \frac{0.655 \cdot 0.783}{0.655 + 0.783} = 0.713$$

Para 160 imagenes y 4 epocas es un buen resultado.

---

## Inferencia con el modelo fine-tuneado

### Paso 1: redefinir `CATEGORY_NAMES`

```python
CATEGORY_NAMES = [
    '__background__', 'raccoon'
]
```

**Sobreescribe** la variable global que en Parte 1 tenia 91 entradas de COCO. Ahora `get_prediction` mapeara `label=1 → 'raccoon'`.

⚠️ Patron fragil: si te olvidas de ejecutar esta celda, los mapaches saldrian etiquetados como `person` (la clase 1 de COCO).

### Paso 2: cargar el checkpoint

```python
checkpoint = torch.load('frcnn_raccoon.model')
frcnn_model.load_state_dict(checkpoint['model'])
frcnn_model.eval()
```

Tres lineas, tres operaciones:

1. **`torch.load`**: deserializa el dict que `train_model` guardo en disco.
2. **`load_state_dict`**: aplica los pesos al modelo. Requiere que la arquitectura coincida con la del checkpoint (en nuestro caso si, porque ya hicimos el reemplazo del `box_predictor`).
3. **`.eval()`**: modo inferencia. Vuelve a devolver detecciones (no losses).

### Paso 3: inferencia sobre imagenes de validacion

```python
object_detection_api('Raccoon_dataset/images/raccoon-42.jpg', threshold=0.9)
object_detection_api('Raccoon_dataset/images/raccoon-31.jpg', threshold=0.9)
object_detection_api('Raccoon_dataset/images/raccoon-191.jpg', threshold=0.95)
```

Estas 3 imagenes estan en `raccoon_test_data.txt` — son **validacion**, no train. El modelo **nunca las vio** durante el entrenamiento.

#### raccoon-42 (threshold 0.9)

![Mapache 42 detectado tras fine-tuning](/laboratorios/lab-15/finetuned-raccoon-42.jpg)

#### raccoon-31 (threshold 0.9)

![Mapache 31 detectado tras fine-tuning](/laboratorios/lab-15/finetuned-raccoon-31.jpg)

#### raccoon-191 (threshold 0.95)

![Mapache 191 detectado tras fine-tuning](/laboratorios/lab-15/finetuned-raccoon-191.jpg)

Cada imagen con caja verde y etiqueta `raccoon:0.9X`. **Score altisimo** porque:

- Modelo entrenado especificamente en este dataset.
- Imagenes de val tienen estilo similar a las de train.

**Antes del fine-tuning**: el modelo COCO clasificaba al mapache como `bear:0.87` (ver [experimento didactico](experimento-mapache)). **Despues**: lo clasifica correctamente como `raccoon` con confianza >0.9. Esta diferencia se logro entrenando **solo ~10.000 parametros** durante **4 epocas** sobre **160 imagenes**. Es la esencia del transfer learning bien aplicado.

### Que se demostro

1. **Transfer learning funciona**: con ~10.000 parametros nuevos sobre 160 imagenes, el modelo aprende a detectar una clase que no estaba en COCO.
2. **Pipeline modular**: misma `object_detection_api` y misma `get_prediction` sirven para COCO (91 clases) o para custom (2 clases). Solo cambian pesos y `CATEGORY_NAMES`.
3. **Paradigma "pre-train + fine-tune"**: es la receta de produccion. Nunca entrenes Faster R-CNN desde cero — carga COCO, reemplaza la cabeza, ajusta pocas epocas.

---

## Sigue: tarea final

Dos preguntas conceptuales a responder. Ver [Tarea final](tarea-final).
