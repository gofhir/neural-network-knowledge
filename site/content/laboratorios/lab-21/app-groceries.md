---
title: "App 1 · Freiburg Groceries"
weight: 4
math: true
---

> **Celdas 38-61 del notebook.** Primera aplicación: leer texto en el dataset [Freiburg Groceries](/papers/freiburg-groceries-jund-2016) (5000 imágenes de productos de supermercado), introduciendo inferencia por batches y búsqueda aproximada de marcas.

## El planteamiento: transfer zero-shot

El dataset Freiburg Groceries (Jund et al., 2016) tiene 5000 imágenes de 25 categorías de productos, originalmente para **clasificación**. El lab lo reaprovecha para **OCR de marcas**, y el punto central es: *el modelo nunca vio estas fotos*. Funciona porque ABCNet aprendió a leer **glifos latinos genéricos**, no "el dataset TotalText" — la forma de una "C" es la misma en una señal de calle que en una caja de cereal.

![CEREAL0000 con las detecciones de texto sobre un producto alemán](/laboratorios/lab-21/groceries-cereal.jpg)

## De `DefaultPredictor` a `build_model` (celda 44)

`DefaultPredictor` procesa imágenes **de una en una** — subóptimo para 5000. Para batchear, se construye el modelo a mano:

```python
from detectron2.modeling import build_model
from adet.checkpoint import AdetCheckpointer
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper

model = build_model(cfg)                          # arquitectura (pesos aleatorios)
AdetCheckpointer(model).load('tt_attn_R_50.pth')  # carga los pesos entrenados
_ = model.eval()                                  # desactiva dropout, congela BatchNorm

aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST, 'choice')
mapper = DatasetMapper(is_train=False, augmentations=[aug], image_format=cfg.INPUT.FORMAT)
```

Las piezas que `DefaultPredictor` ocultaba, ahora explícitas:

- **`build_model`** crea el esqueleto; **`AdetCheckpointer`** (de `adet`, no de Detectron2) carga los pesos sabiendo mapear las claves propias de ABCNet (cabeza Bézier, recognizer). Con el checkpointer de Detectron2 habría `missing keys` en esas capas.
- **`model.eval()`** es crítico: sin él, dropout y BatchNorm dan predicciones no deterministas y peores.
- **`ResizeShortestEdge`** reescala para que el lado corto = `MIN_SIZE_TEST`, con tope en `MAX_SIZE_TEST` para el lado largo, **manteniendo aspect ratio** (deformar arruinaría las letras).
- **`DatasetMapper`** carga la imagen del disco, aplica `aug`, la convierte a tensor `(C,H,W)` y lee en BGR (`image_format=cfg.INPUT.FORMAT`) — aquí **sí** se respeta el formato.

### `ResizeShortestEdge` — triple framework

El "redimensionar por lado corto con tope en lado largo, manteniendo aspect ratio" es un patrón genérico de preprocesamiento de detección:

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch.nn.functional as F

def resize_shortest_edge(img, short=1000, max_size=1824):  # (C,H,W)
    c, h, w = img.shape
    scale = short / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)
    nh, nw = round(h*scale), round(w*scale)
    return F.interpolate(img[None], size=(nh, nw),
                         mode='bilinear', align_corners=False)[0]
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

def resize_shortest_edge(img, short=1000, max_size=1824):  # (H,W,C)
    hf = tf.cast(tf.shape(img)[0], tf.float32)
    wf = tf.cast(tf.shape(img)[1], tf.float32)
    scale = short / tf.minimum(hf, wf)
    scale = tf.where(tf.maximum(hf, wf)*scale > max_size,
                     max_size/tf.maximum(hf, wf), scale)
    nh = tf.cast(tf.round(hf*scale), tf.int32)
    nw = tf.cast(tf.round(wf*scale), tf.int32)
    return tf.image.resize(img, (nh, nw), method='bilinear')
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax.numpy as jnp
from jax.image import resize

def resize_shortest_edge(img, short=1000, max_size=1824):  # (H,W,C)
    h, w = img.shape[:2]
    scale = short / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)
    nh, nw = round(h*scale), round(w*scale)
    return resize(img, (nh, nw, img.shape[2]), method='bilinear')
```
{{< /tab >}}
{{< /tabs >}}

> Diferencias idiomáticas: PyTorch usa layout `(C,H,W)` y `interpolate` necesita un batch dummy (`img[None]`); TF/JAX usan `(H,W,C)`. En TF el control de flujo en grafo se hace con `tf.where` (no `if`, porque `h,w` son simbólicos).

## El loop de batching (celda 48)

```python
BATCH_SIZE = 8
for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    model_input = [mapper(image) for image in dataset[i:i+BATCH_SIZE]]
    pred = model(model_input)                       # ← lista de dicts, no array crudo
    for j in range(len(pred)):
        predicted_text = [visualizer._decode_recognition(r)
                          for r in pred[j]['instances'].recs]
        dataset[i+j]['words'] = predicted_text
```

🎯 La diferencia clave: `model(...)` (el modelo crudo) recibe una **lista de dicts ya preprocesados** y procesa todo el batch de una pasada, devolviendo una **lista** de predicciones. `DefaultPredictor(array)` recibía un array NumPy crudo y devolvía un solo dict — por eso no batchea. La GPU tiene miles de núcleos: procesar 1 imagen los deja ociosos; un batch los aprovecha.

> El `dataset` es una lista de dicts `{'file_name', 'image_id'}` construida con `glob` + `.sort()` (el sort da reproducibilidad de los `image_id`). Las imágenes se cargan **bajo demanda** dentro del mapper, no todas a RAM.

## Análisis: las palabras más frecuentes (celda 53)

```python
all_words = []
for image in dataset:
    all_words += [word.lower() for word in image['words']]   # .lower() unifica MILCH/Milch/milch
frequency = Counter(all_words)
frequency.most_common(40)
```

Top real medido (de 5000 imágenes):

| Rank | Palabra (freq) | Tipo |
|---|---|---|
| 1-5 | bio (254) · real (165) · milch (103) · reis (86) · honig (79) | sustantivos alemanes + cadena "Real" |
| … | zucker, kuchen, tee, tomaten, bohnen, apfel, saft, mais | sustantivos alemanes limpios |
| marcas | haribo, edeka, bonduelle, nescafe, teekanne, ricola | bien leídas |
| errores | **droetker** (= Dr. Oetker), **musli** (= Müsli), **apfel** (= Äpfel), **uncle**+**bens** (= Uncle Ben's) | ver abajo |

### Tres hallazgos cuantitativos

1. **El transfer cross-idioma funciona.** Decenas de palabras alemanas perfectamente formadas (milch 103×, reis 86×), sin basura. El modelo entrenado en inglés lee alemán porque comparten alfabeto latino.
2. **El límite del charset.** `musli` (debería ser "Müsli") y `apfel` ("Äpfel") — la **ü** y la **Ä** no están en ASCII 32-126, así que se colapsan a la letra base. La diéresis perdida es la pista de la [Actividad 2](actividades).
3. **Detección palabra-por-palabra.** "Uncle Ben's" aparece partido en `uncle` (40×) + `bens` (44×): el detector segmenta en palabras y cada una se lee aislada (pista de la [Actividad 1](actividades)).

> El error `droetker` (= "Dr. Oetker") es **sistemático**: el modelo siempre lee igual ese logo. La consistencia lo hace utilizable como clave estable, aunque esté "mal".

## Búsqueda de marcas: exacta vs aproximada (celdas 55-60)

```python
frequency['nestle']          # → 17 (match exacto)
```

![Una de las 17 imágenes con match exacto de "nestle"](/laboratorios/lab-21/nestle-match.jpg)

El match exacto encuentra 17 productos Nestlé, pero descarta las lecturas imperfectas ("nestl", "nesle"). La solución es [fuzzy matching](/fundamentos/fuzzy-string-matching):

```python
from fuzzywuzzy import fuzz
keyword, threshold = 'nestle', 80
for image in dataset:
    for word in image['words']:
        if word.lower() != keyword:                  # solo lo nuevo (no los 17 exactos)
            if fuzz.ratio(word.lower(), keyword) > threshold:
                # match aproximado: 'nestl' (≈91), 'nestie' (≈83)...
                ...
```

`fuzz.ratio` convierte la **distancia de Levenshtein** en un ratio 0-100. Con threshold 80 se rescataron **3 instancias extra** (≈20 total). El umbral encarna el trade-off **precision/recall**: alto → pocas pero correctas; bajo → muchas pero con basura ("needle", "castle"). La elección depende del costo del error en la aplicación.

> **Conexión con record linkage / FHIR MDM:** este es exactamente el problema de *entity resolution* — emparejar "José González" con "Jose Gonzalez" o "J. Gonzales". El trade-off del umbral es el que se calibra en un *scorer*: demasiado bajo genera falsos positivos (fusionar dos pacientes = peligroso clínicamente), demasiado alto duplica registros. La diferencia es que en FHIR se combinan varios campos (nombre + fecha nacimiento + identificador) para subir precisión sin sacrificar recall, algo que el fuzzy puro sobre un solo campo no logra. Ver [fuzzy string matching](/fundamentos/fuzzy-string-matching).

---

**Anterior:** [disección del output](diseccion-output) · **Siguiente:** [App 2 · Google Street View](app-streetview)
