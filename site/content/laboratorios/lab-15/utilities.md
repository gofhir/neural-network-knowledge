---
title: "Utilities del entrenamiento: IoU, metricas, NMS, warmup"
weight: 60
math: true
---

Antes del loop de entrenamiento, el lab define 5 funciones helper que se usaran dentro de `train_one_epoch` y `eval_epoch`. Conceptualmente distintas pero todas necesarias.

## 1. `iou(bb1, bb2)` — Intersection over Union

```python
def iou(bb1, bb2):
    x_left   = max(bb1['x1'], bb2['x1'])
    y_top    = max(bb1['y1'], bb2['y1'])
    x_right  = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    return iou
```

### Para que sirve

Durante la evaluacion, para decidir si una prediccion es **True Positive** o **False Positive**:

| IoU(prediccion, GT) | Decision |
| --- | --- |
| ≥ 0.5 | TP (la prediccion esta cerca de un objeto real) |
| < 0.5 | FP (la prediccion no corresponde a ningun objeto real) |

Si una caja GT **no es cubierta** por ninguna prediccion con IoU ≥ 0.5 → FN.

### El algoritmo

La interseccion de dos cajas eje-alineadas es una nueva caja cuyos limites son:

- Borde izquierdo: **max** de los `x1` (el mas a la derecha de los izquierdos).
- Borde derecho: **min** de los `x2` (el mas a la izquierda de los derechos).
- Analogo para `y`.

Si los bordes resultantes estan invertidos -> no se solapan -> IoU = 0.

La formula final:

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{interseccion}}{|A| + |B| - \text{interseccion}}$$

(Restar la interseccion evita contarla dos veces en la union.)

### Comparacion con `torchvision.ops.box_iou`

PyTorch tiene una implementacion vectorizada nativa:

```python
from torchvision.ops import box_iou
ious = box_iou(boxes1, boxes2)   # tensores (N, 4) -> matriz (N, M)
```

Mucho mas rapida para muchas comparaciones (un solo kernel CUDA). El lab usa la implementacion manual con dict por didactica.

---

## 2. `print_stats(stats, best_stats, epoch)` — logging

Funcion de logging que imprime las metricas de la epoca actual junto al "mejor valor visto hasta ahora", actualizando los bests sobre la marcha.

### Inputs

```python
stats = {
    'true_positives':  {0: 8},    # 8 TPs en la clase 0 (raccoon)
    'false_positives': {0: 2},    # 2 FPs
    'false_negatives': {0: 1}     # 1 FN
}

best_stats = {
    'true_positives':  {0: (best_pct, best_val, best_epoch)},
    'false_positives': {0: (best_val, best_epoch)},
    'false_negatives': {0: (best_pct, best_val, best_epoch)}
}
```

Asimetria: TP y FN guardan triple `(pct, val, epoch)`; FP solo `(val, epoch)` porque no hay un denominador obvio para porcentaje.

### Calculos

- **TP %** = $\frac{TP}{TP + FN}$ = **Recall**.
- **FN %** = $\frac{FN}{FN + TP}$ = **1 - Recall** (tasa de olvido).
- **FP**: conteo absoluto.

Para TP el "mejor" se actualiza cuando el porcentaje **sube**. Para FN cuando **baja**. Para FP cuando el conteo **baja**.

### Output tipico

```text
True positives:
  raccoon       0.875   (7)         Best epoch: 12  0.925   (8)

False negatives (hard positive):
  raccoon       0.125   (1)         Best epoch: 12  0.075   (0)

False positives (hard negative):
  raccoon                (3)        Best epoch: 12         (1)
```

### Nota: el "best" mezcla metricas distintas

El "Best epoch" se calcula por metrica independientemente. **El mejor recall puede ser en la epoca 12** y **los mejores FPs en la epoca 5**. No representa una sola "epoca ganadora". La decision de que checkpoint guardar se hace en `train_model` con otra logica.

---

## 3. `stats_2_metrics(stats)` — calculo limpio de metricas

```python
def stats_2_metrics(stats):
    metrics = {'True positives': {}, 'False negatives': {}, 'False positives': {},
               'Precision': {}, 'Recall': {}}
    for category in Category:
        for k in metrics:
            metrics[k][category] = 0

    for category, value in stats['true_positives'].items():
        if stats['false_negatives'][category] + value > 0:
            metrics['True positives'][category] = value / (value + stats['false_negatives'][category])

    # ... similar para FN, FP ...

    for category in Category:
        metrics['Recall'][category] = metrics['True positives'][category]
        metrics['Precision'][category] = stats['true_positives'][category] / (
            stats['true_positives'][category] +
            stats['false_positives'][category] + 0.0000001)
    return metrics
```

Es la "version limpia" de lo que hace `print_stats`, pero **sin imprimir** y **sin actualizar bests**. Solo convierte conteos crudos en metricas calculadas.

### Las 5 metricas

| Metrica | Formula | Pregunta |
| --- | --- | --- |
| **Precision** | $\frac{TP}{TP + FP}$ | "De todo lo que dije que era raccoon, %real?" |
| **Recall** | $\frac{TP}{TP + FN}$ | "De todos los raccoons reales, %detectados?" |
| TP % | igual que Recall | duplicado por consistencia |
| FN % | $\frac{FN}{FN + TP}$ = 1 - Recall | tasa de olvido |
| FP | conteo absoluto | falsos positivos |

### Truco del epsilon

```python
metrics['Precision'][...] = TP / (TP + FP + 0.0000001)
```

Si en una epoca el modelo no genera ningun TP ni FP, el denominador seria 0 -> `ZeroDivisionError`. El epsilon (`1e-7`) hace que la division retorne ~0 sin crashear.

### Trade-off Precision vs Recall

- **Threshold alto** (ej. 0.9) -> menos detecciones, mas conservador -> **Precision sube, Recall baja**.
- **Threshold bajo** (ej. 0.3) -> mas detecciones, mas agresivo -> **Recall sube, Precision baja**.

Por eso se reportan ambos. O metricas combinadas (F1, AP).

---

## 4. `filter_by_class_nms(detection)` — NMS por clase

```python
def filter_by_class_nms(detection):
    boxes = detection['boxes']; detection['boxes'] = torch.tensor([])
    labels = detection['labels']; detection['labels'] = torch.tensor([], dtype=labels.dtype)
    scores = detection['scores']; detection['scores'] = torch.tensor([])

    for category in Category:
        category_idxs = (labels == category + 1)
        relevant_boxes = boxes[category_idxs]
        relevant_scores = scores[category_idxs]
        keep_idxs = nms(relevant_boxes, relevant_scores, IOU_OVERLAP_THRESHOLD)

        detection['boxes']  = torch.cat([detection['boxes'],  relevant_boxes[keep_idxs]])
        detection['scores'] = torch.cat([detection['scores'], relevant_scores[keep_idxs]])
        detection['labels'] = torch.cat([detection['labels'],
                                          torch.full_like(keep_idxs, category+1)])
    return detection
```

### Recordatorio: por que NMS por clase

Imagina una imagen con un perro y un gato cerca. El modelo puede producir:

- Caja A: clase=dog, score=0.92, ubicacion X.
- Caja B: clase=cat, score=0.88, ubicacion X (solapada con A).

Un **NMS global** eliminaria B porque tiene mucho IoU con A. Pero **son objetos distintos**. NMS **por clase** los trata en silos separados y deja ambas.

### Pasos

1. **Extraer y vaciar**: guarda los tensores originales en variables locales y vacia el dict.
2. **Iterar por categoria**: para cada clase real (`category + 1` porque el modelo usa 0=bg, 1=raccoon, ...).
3. **NMS dentro de la categoria**: `torchvision.ops.nms` toma cajas + scores + threshold IoU y devuelve los indices que sobreviven.
4. **Acumular**: concatena las detecciones sobrevivientes al dict de salida.

### Por que el +1 en `category + 1`

`Category = [0]` es la categoria **real** (sin contar background). En el modelo es indice `1` (porque 0 = background). El offset `+1` mapea entre ambas convenciones, consistente con el offset del DataLoader.

### Detalle: ya hay NMS interno en Faster R-CNN

Faster R-CNN de torchvision **ya hace NMS por clase internamente** durante el forward. Este NMS extra es un **filtro de seguridad** adicional, con un threshold posiblemente distinto (`IOU_OVERLAP_THRESHOLD` definido despues, tipicamente 0.5). En entrenamientos serios podrias omitirlo.

---

## 5. `warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)` — warmup del LR

```python
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
```

### Por que hace falta warmup

Recuerda el reemplazo del clasificador en [fine-tuning-setup](fine-tuning-setup):

- El backbone, FPN, RPN, box_head ya estan bien entrenados (vienen de COCO).
- El **nuevo `box_predictor`** tiene pesos aleatorios.

Si arrancas con un `lr` alto, los pesos aleatorios producen gradientes **enormes y caoticos** que se propagan hacia atras. Esos gradientes:

1. **Destruyen** los pesos pre-entrenados del resto del modelo en los primeros pasos.
2. **Causan inestabilidad numerica** (NaNs, gradientes explosivos).

El warmup mitiga esto: `lr` inicial bajo permite que el predictor aleatorio "se asiente" sin destruir el conocimiento pre-existente.

### Visualizacion

```text
lr
│
│              ━━━━━━━━━━━━━━━━━━━━━━━━━ (full lr)
│           ╱
│        ╱
│     ╱  (warmup)
│  ╱
│╱
└────────────────────────────────────────► iteracion
   warmup_iters
```

### La funcion `f(x)`

`x` = numero de iteracion actual (no la epoca).

- En `x = 0`: `alpha = 0` -> retorna `warmup_factor` (tipicamente `1/1000 = 0.001`).
- En `x = warmup_iters`: `alpha = 1` -> retorna `1` (lr completo).
- En medio: interpolacion lineal entre ambos.

Es decir, `f(x)` es un **multiplicador** que sube linealmente de `warmup_factor` a `1` durante las primeras `warmup_iters` iteraciones.

### `LambdaLR`

```python
torch.optim.lr_scheduler.LambdaLR(optimizer, f)
```

Multiplica el `lr` base del optimizer por `f(epoch_actual)` en cada paso. El `lr` efectivo es:

$$lr_{efectivo}(x) = lr_{base} \cdot f(x)$$

### Uso tipico

```python
optimizer = SGD(params, lr=0.005)
scheduler = warmup_lr_scheduler(optimizer, warmup_iters=1000, warmup_factor=1/1000)

for iter in range(total_iters):
    # forward + backward
    optimizer.step()
    scheduler.step()
```

| Iteracion | `f(x)` | `lr` efectivo |
| --- | --- | --- |
| 0 | 0.001 | 0.000005 |
| 500 | 0.5005 | 0.0025 |
| 1000+ | 1 | 0.005 |

### En el lab

`warmup_lr_scheduler` se usa **dentro de `train_one_epoch`** pero **solo en la primera epoca**. Con 27 batches por epoca, el warmup dura `min(1000, 26) = 26` iteraciones. Despues del warmup, el `lr` se mantiene en su valor base (sin decay agresivo).

En entrenamientos mas sofisticados, despues del warmup tendrias un decay (cosine, step, etc.). El lab omite esa parte por simplicidad.

---

## Resumen de las 5 funciones

| Funcion | Tipo | Proposito |
| --- | --- | --- |
| `iou` | Calculo | IoU entre dos cajas (usado en eval para decidir TP/FP) |
| `print_stats` | Logging | Imprime metricas con bests historicos |
| `stats_2_metrics` | Calculo | Convierte conteos crudos en precision/recall |
| `filter_by_class_nms` | Post-procesamiento | NMS por clase de las predicciones |
| `warmup_lr_scheduler` | Optimizacion | Sube `lr` gradualmente al inicio |

---

## Sigue: el loop de entrenamiento

Las siguientes celdas (76, 77, 78) definen `train_one_epoch`, `eval_epoch` y `train_model`. Son las funciones que orquestan todo lo anterior. **(Pendiente en este recorrido — proximas paginas del lab).**
