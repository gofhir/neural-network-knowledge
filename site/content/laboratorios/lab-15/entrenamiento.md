---
title: "Loop de entrenamiento: train_one_epoch, eval_epoch, train_model"
weight: 70
math: true
---

Con las [utilities](utilities) ya definidas, el lab construye tres funciones que ejecutan una epoca de entrenamiento, una epoca de validacion y orquestan ambas a traves de varias epocas guardando el mejor checkpoint.

## `train_one_epoch` — ejecutar una epoca completa

```python
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    print('Epoch: [{}]'.format(epoch))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    iterator = tqdm(data_loader)
    nans = 0
    moving_loss = 0.0

    for images, targets in iterator:
        images = [image.to(device) for image in images]
        images_rgb = [image[:3,:,:] for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images_rgb, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            nans += 1
        else:
            losses.backward()
            optimizer.step()

            if moving_loss == 0.0:
                moving_loss = loss_value
            else:
                moving_loss = moving_loss * 0.95 + loss_value * 0.05

        iterator.set_description(
            'Epoch: {}; Loss: {:.5f}; NANs: {}'.format(epoch, moving_loss, nans)
        )

        if lr_scheduler is not None:
            lr_scheduler.step()
    return moving_loss
```

### El cambio mas importante: `model.train()` vs `model.eval()`

Esta es la diferencia mas critica entre Parte 1 (inferencia) y Parte 2 (entrenamiento). Faster R-CNN tiene comportamientos distintos segun el modo:

| Modo | `model(images)` devuelve | `model(images, targets)` devuelve |
| --- | --- | --- |
| `eval()` | detecciones `[{boxes, labels, scores}]` | detecciones (ignora targets) |
| `train()` | error (necesita targets) | **dict de losses** |

En modo entrenamiento, el forward devuelve **un dict con 4 losses**:

```python
loss_dict = {
    'loss_classifier':  tensor,   # multi-clase CE del RoI head (91 -> 2 clases en el lab)
    'loss_box_reg':     tensor,   # smooth L1 de la regresion refinada (RoI head)
    'loss_objectness':  tensor,   # binary CE de la RPN (objeto vs fondo)
    'loss_rpn_box_reg': tensor    # smooth L1 de la regresion de la RPN
}
```

Las 4 son las del [multi-task loss del paper Faster R-CNN](/papers/faster-rcnn-ren-2015). RPN (2 losses) + RoI head (2 losses).

### Warmup solo en epoca 0

Recuerda que el `box_predictor` fue inicializado aleatoriamente (Linear(1024, 2) y Linear(1024, 8) nuevos). Sus gradientes iniciales son **enormes y caoticos** — si arrancas con `lr=0.005` completo, destruyen los pesos pre-entrenados del resto del modelo en los primeros pasos.

[`warmup_lr_scheduler`](utilities) sube el `lr` linealmente de `lr_base/1000` a `lr_base` durante las primeras `min(1000, len(data_loader)-1)` iteraciones. Para 160 imagenes y `batch_size=6`, son **26 iteraciones** (casi una epoca completa).

A partir de la epoca 1 ya no se crea scheduler nuevo. `lr_scheduler` queda en `None`, y la linea final `lr_scheduler.step()` se salta.

### El detalle `images_rgb = [image[:3,:,:] for image in images]`

Salvaguarda defensiva: si alguna imagen viene con 4 canales (RGBA), se descarta el canal alpha. En el dataset Raccoon todas son JPG (3 canales), asi que es redundante pero buena practica.

### El forward + loss + backward

```python
optimizer.zero_grad()                                # limpiar gradientes acumulados
loss_dict = model(images_rgb, targets)               # forward → dict de 4 losses
losses = sum(loss for loss in loss_dict.values())    # suma sin ponderacion
loss_value = losses.item()                            # a float Python (no tensor)

if math.isfinite(loss_value):
    losses.backward()    # gradientes
    optimizer.step()     # actualizar pesos
```

Patron estandar de PyTorch: `zero_grad → forward → backward → step`. La unica peculiaridad es el **chequeo de NaN/Inf**: si el loss es invalido, se skipea el backward para no propagar gradientes corruptos a todo el modelo.

### Exponential moving average del loss

```python
moving_loss = moving_loss * 0.95 + loss_value * 0.05
```

EMA con factor 0.95. Suaviza fluctuaciones batch-a-batch para que `tqdm.set_description` muestre la tendencia, no el ruido. El loss reciente pesa 5%, el promedio historico 95%.

---

## `eval_epoch` — calcular TPs, FPs, FNs en validacion

```python
def eval_epoch(model, data_loader, device):
    model.eval()

    stats = {}
    stats['true_positives']  = {category: 0 for category in Category}
    stats['false_positives'] = {category: 0 for category in Category}
    stats['false_negatives'] = {category: 0 for category in Category}

    for images, targets in tqdm(data_loader):
        images = [image.to(device) for image in images]
        images_rgb = [image[:3, :, :] for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        batch_detections = model(images_rgb, targets)
        batch_detections = [{k: v.cpu() for k, v in det.items()} for det in batch_detections]

        for batch_idx, detection in enumerate(batch_detections):
            image_idx = targets[batch_idx]['image_id']
            detection = filter_by_class_nms(detection)
            
            # Filtrar por score threshold
            boxes = detection['boxes']
            del detection['boxes']
            detection_df = pd.DataFrame.from_dict(detection)
            relevant_dets = detection_df[detection_df['scores'] > SCORE_THRESHOLD]

            # Recargar GT desde el dataset (ineficiente pero funcional)
            my_img, my_targets = val_data.__getitem__(image_idx)
            bbs = my_targets['boxes']
            bb_categories = my_targets['labels']

            # Para cada GT: buscar match → TP o FN
            positives = []
            for i in range(bbs.shape[0]):
                gt_category = bb_categories[i] - 1
                gt_polygon = {'x1': bbs[i,0], 'y1': bbs[i,1], 'x2': bbs[i,2], 'y2': bbs[i,3]}
                positive = False
                for j, (_, (pred_category, score)) in enumerate(relevant_dets.iterrows()):
                    box = boxes[j]
                    pred_category -= 1
                    pred_polygon = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
                    if iou(gt_polygon, pred_polygon) > IOU_POSITIVE_THRESHOLD:
                        if gt_category == pred_category:
                            positive = True
                            positives.append(pred_polygon)
                            break
                if positive:
                    stats['true_positives'][gt_category.item()] += 1
                else:
                    stats['false_negatives'][gt_category.item()] += 1

            # Predicciones no emparejadas → FP
            for j, (_, (pred_category, score)) in enumerate(relevant_dets.iterrows()):
                box = boxes[j]
                pred_category -= 1
                pred_polygon = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
                if pred_polygon not in positives:
                    stats['false_positives'][pred_category] += 1
    return stats
```

### El algoritmo en lenguaje natural

> Para cada imagen del set de validacion:
>   1. Predecir con el modelo.
>   2. Filtrar predicciones con `score > SCORE_THRESHOLD` (default 0.5).
>   3. Para cada caja GT: buscar entre las predicciones alguna con `IoU > IOU_POSITIVE_THRESHOLD` (default 0.7) Y la misma clase.
>      - Si match → **TP**, marcar prediccion como usada.
>      - Si no → **FN** (GT no detectada).
>   4. Predicciones no marcadas → **FP**.

### Los offsets `- 1`

Recuerda: el modelo predice labels 1..N (con 0=background), pero el dict `stats` usa indices 0..N-1. De ahi los `gt_category - 1` y `pred_category - 1` que normalizan ambas convenciones.

### Limitacion: NO calcula mAP@[.5:.95]

Esta funcion es **una version simplificada** de la metrica COCO oficial. Lo que SI hace: TP/FP/FN explicitos a un solo IoU threshold. Lo que NO hace:

- No promedia sobre multiples thresholds de IoU (mAP@[.5:.95]).
- No calcula AP (area bajo curva precision-recall).
- Matching greedy en vez de optimo (no Hungarian).

Para produccion usarias `torchmetrics.detection.MeanAveragePrecision` o `pycocotools`. Pero ver TP/FP/FN explicitos es **mucho mas instructivo** que un solo numero opaco.

---

## `train_model` — orquestador completo

```python
def train_model(model, train_data_loader, val_data_loader, optimizer, lr_scheduler, device):
    file_name = 'frcnn_raccoon.model'
    MODELS_ROOT = '.'
    save_path = os.path.join(MODELS_ROOT, file_name)
    best_precision = 0

    best_stats = {}
    best_stats['true_positives']  = {category: (0, 0, -1) for category in Category}
    best_stats['false_negatives'] = {category: (None, None, -1) for category in Category}
    best_stats['false_positives'] = {category: (1, -1) for category in Category}

    epochs = 4
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, train_data_loader, device, epoch)
        lr_scheduler.step()

        with torch.no_grad():
            epoch_stats = eval_epoch(model, val_data_loader, device)
        print_stats(epoch_stats, best_stats, epoch)
        epoch_metrics = stats_2_metrics(epoch_stats)
        print(epoch_metrics)

        if epoch_metrics['Precision'][0] > best_precision:
            best_precision = epoch_metrics['Precision'][0]
            with open(save_path, 'wb') as f:
                state = {
                    "optimizer": optimizer.state_dict(),
                    "model": model.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "precision": epoch_metrics['Precision'][0],
                }
                torch.save(state, f)
```

### El flujo de una epoca

```text
for epoch in range(4):
    1. train_loss = train_one_epoch(...)
    2. lr_scheduler.step()             ← StepLR (no warmup; ya termino en ep 0)
    3. with torch.no_grad():
           epoch_stats = eval_epoch(...)
    4. print_stats(epoch_stats, best_stats, epoch)
       print(stats_2_metrics(epoch_stats))
    5. if precision mejora:
           torch.save(state)
```

### El criterio de "mejor checkpoint": precision

⚠️ El lab guarda el modelo cuando la **precision** mejora, no F1 ni mAP. Es una decision arbitraria. Para tu propio dataset, podrias preferir F1 (balancea precision y recall) o AP (area bajo curva).

### El `torch.no_grad()` en eval

Desactiva el tracking de gradientes durante la validacion. Ahorra memoria y un poco de tiempo, ya que no construye el grafo computacional. En eval no hacemos backward → no necesitamos gradientes.

### Que se guarda en el checkpoint

```python
state = {
    "optimizer": optimizer.state_dict(),   # momentum, m1/m2 si fuera Adam
    "model": model.state_dict(),           # pesos (~160 MB)
    "scheduler": lr_scheduler.state_dict(),# step actual del decay
    "epoch": epoch,                        # epoca en que se guardo
    "precision": ...                       # precision de validacion
}
```

Solo `model.state_dict()` es esencial para inferencia. El resto permite **continuar entrenamiento desde el checkpoint** (util para finetune en multiples sesiones).

### Patron "save only the best"

Como `torch.save` esta dentro del `if`, **si la epoca N no mejora, no se sobreescribe el archivo**. El checkpoint final tiene los pesos del **mejor modelo segun precision**, no del ultimo entrenamiento. Esto evita perder un buen modelo si la ultima epoca empeora por overfitting.

---

## Sigue: lanzamiento + resultados reales

Ver [Lanzamiento del entrenamiento + inferencia post-fine-tuning](inferencia-finetuneada).
