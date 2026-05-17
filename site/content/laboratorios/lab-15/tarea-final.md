---
title: "Tarea final: respuestas razonadas"
weight: 90
math: true
---

El notebook cierra con dos preguntas conceptuales que recorren el ciclo completo del fine-tuning y el preprocesamiento interno del modelo. Esta pagina consolida las respuestas con el razonamiento detras de cada una.

---

## Pregunta 1

> *Si quisiera utilizar mi propio dataset con `X` clases distintas, ¿que partes del codigo debiese modificar?*

### Respuesta

Para usar tu propio dataset con `N` clases reales hay que modificar **5 puntos del codigo**:

#### 1) `RaccoonDataLoader` (Celda 59) — formato + offset de labels

El dataset Raccoon usa `label=0` en el `.txt` y el DataLoader hace `+1` para llevarlo a `1` (porque en Faster R-CNN el `0` es background):

```python
labels = annotations_array[:,4] + 1   # convierte 0 → 1 (raccoon)
```

Para tu dataset:

- Si tus labels en el archivo **empiezan en 0** (clases 0, 1, ..., N-1) → mantener `+1`. Tus labels finales seran 1, 2, ..., N.
- Si tus labels **ya empiezan en 1** → eliminar el `+1` (sino se desfasan).
- Si tu formato de anotaciones no es el TXT plano del lab (ej. XML PASCAL VOC, JSON COCO) → reescribir la logica de parseo de `__init__` y/o `__getitem__`.

#### 2) `num_classes` (Celdas 63 y 70)

```python
num_classes = 2   # background + raccoon
```

Cambiar a:

```python
num_classes = N + 1   # background + las N clases reales
```

El **`+1`** es critico — es la convencion de torchvision para el background.

#### 3) `Category` (Celda 63)

```python
Category = list(range(num_classes - 1))   # [0] para 1 clase real
```

Se ajusta automaticamente al cambiar `num_classes`. Con 3 clases reales seria `[0, 1, 2]`. **No requiere edicion manual** si modificaste `num_classes`.

#### 4) `int2class` en `print_stats` (Celda 73)

```python
int2class = {0: 'raccoon'}
```

Usado por la funcion de logging para imprimir el nombre legible de la clase. Para tu dataset:

```python
int2class = {0: 'perro', 1: 'gato', 2: 'pajaro'}   # ejemplo con 3 clases
```

#### 5) `CATEGORY_NAMES` (Celda 84)

```python
CATEGORY_NAMES = ['__background__', 'raccoon']
```

Usado por `get_prediction` para mapear indices a nombres en las visualizaciones. Para tu dataset:

```python
CATEGORY_NAMES = ['__background__', 'perro', 'gato', 'pajaro']
```

⚠️ **El orden importa**: debe coincidir con los labels post-offset del DataLoader. Si en el `.txt` `dog=0`, `cat=1`, `bird=2`, tras el `+1` seran `dog=1`, `cat=2`, `bird=3`. La lista `CATEGORY_NAMES` debe estar en ese orden:

- indice 0 = `__background__`
- indice 1 = `perro`
- indice 2 = `gato`
- indice 3 = `pajaro`

#### Resumen visual

```text
┌────────────────────────────────────────────────────────────────┐
│ Cambios necesarios para dataset propio con N clases:            │
├────────────────────────────────────────────────────────────────┤
│ 1. RaccoonDataLoader  → ajustar parseo + offset de labels       │
│ 2. num_classes = N + 1                                          │
│ 3. Category = list(range(N))   (auto-ajustado)                  │
│ 4. int2class = {0: 'clase1', 1: 'clase2', ..., N-1: 'claseN'}  │
│ 5. CATEGORY_NAMES = ['__background__', 'clase1', ..., 'claseN'] │
└────────────────────────────────────────────────────────────────┘
```

### Lo que NO hay que modificar

Sorprendentemente, **el resto del codigo es agnostico al numero de clases**:

- `train_one_epoch`, `eval_epoch`, `train_model`: funcionan para cualquier N.
- `iou`, `filter_by_class_nms`, `warmup_lr_scheduler`: tampoco dependen de N.
- `FastRCNNPredictor`: torchvision lo dimensiona automaticamente segun `num_classes`.

Es la potencia del **transfer learning bien hecho**: la mayor parte del codigo es reutilizable.

---

## Pregunta 2

> *¿La implementacion de PyTorch de Faster R-CNN puede recibir como entrada imagenes de cualquier tamano? En caso contrario, especifique de que tamano deben ser.*

### Respuesta

**Si, puede recibir imagenes de cualquier tamano.** El modelo tiene una capa interna de preprocesamiento llamada `GeneralizedRCNNTransform` que adapta cualquier imagen al tamano que el resto del modelo espera.

### Como funciona el `transform` interno

En el `print(frcnn_model)` veias:

```text
(transform): GeneralizedRCNNTransform(
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    Resize(min_size=(800,), max_size=1333, mode='bilinear')
)
```

Aplica **dos operaciones** a cada imagen:

1. **Normalizacion**: resta la media de ImageNet y divide por la desviacion estandar. Los pesos de ResNet-50 fueron entrenados esperando esa distribucion.

2. **Resize**: redimensiona manteniendo el aspect ratio bajo dos restricciones:
   - El **lado corto** debe ser **al menos 800 px**.
   - El **lado largo** debe ser **como maximo 1333 px**.

### Ejemplos del resize

| Imagen original | Resultado |
| --- | --- |
| 1024×768 (HD) | 1067×800 |
| 4000×3000 (DSLR) | 1067×800 |
| 640×480 (VGA) | 1067×800 (upscale) |
| 5000×800 (panoramica) | 1333×213 (el largo dominaria, cap a 1333) |
| 100×100 (icono) | 800×800 (upscale agresivo) |

### Por que mantener aspect ratio y no forzar cuadrado

Si forzaramos todo a un tamano cuadrado fijo (ej. 800×800), tendriamos un problema fundamental: **deformacion**.

- Un autobus alargado (ratio 3:1) se veria como un cubo.
- Una persona vertical (ratio 1:2) se veria achatada.

Eso destruiria la informacion geometrica que el detector usa para reconocer objetos. Por eso el modelo prefiere **paddear con ceros** dentro del batch para igualar dimensiones, en vez de aplicar resize cuadrado.

### Implicacion practica

Por eso pudiste pasarle al modelo durante el lab:

- `zebras.jpg` (800×600 aprox)
- `ofice.jpg` (678×381)
- `eating.jpg` (~500×750)
- `raccoon-XX.jpg` (tamanos variados)

Sin tener que redimensionarlas tu. El modelo se encarga internamente.

### Caso limite: imagenes muy pequenas

Si pasas una imagen 50×50, el resize agresivo a 800×800 producira detecciones muy malas porque la informacion original es pobre. **El modelo no fallara**, pero la calidad sera mala. Lo mismo con imagenes gigantes (16k×16k): seran reducidas a 1333×... y se pierde detalle.

### Caso limite: distintos tamanos en un mismo batch

Cuando el `DataLoader` agrupa imagenes de tamanos distintos en un batch, **el transform las paddea con ceros** para que coincidan dimensionalmente despues del resize. Por eso la funcion `collate` del lab usa `tuple(zip(*batch))` en vez de `torch.stack` — permite tamanos heterogeneos.

### Donde encontrar mas detalle

- [Arquitectura del lab — Pieza 0](arquitectura/#pieza-0--transform-preprocesamiento)
- [Profundizacion de la clase 15](/clases/clase-15/profundizacion/)
- [Fundamento de deteccion de objetos](/fundamentos/deteccion-de-objetos/)

---

## Lecciones aprendidas del lab

Mas alla de las dos respuestas, el lab dejo varias lecciones transversales que vale la pena consolidar:

### 1. Transfer learning es la receta de produccion

Con **~10.000 parametros nuevos** (las dos `Linear` del `FastRCNNPredictor`) entrenados sobre **160 imagenes** durante **4 epocas (~5 minutos)**, el modelo paso de no conocer mapaches (los confundia con osos) a detectarlos con 78% de recall y 65% de precision.

Si entrenaras Faster R-CNN desde cero con 160 imagenes, **el resultado seria desastroso**. La diferencia es el **99.975% pre-entrenado**.

### 2. `model.train()` vs `model.eval()` cambia el comportamiento del forward

Es la fuente #1 de bugs en fine-tuning de Faster R-CNN:

- `eval()`: devuelve detecciones.
- `train()`: devuelve dict de losses (y requiere `targets`).

Si te olvidas de `train()` antes de entrenar, los gradientes no se calculan. Si te olvidas de `eval()` antes de inferir, falla.

### 3. El warmup del learning rate protege los pesos pre-entrenados

Empezar con `lr` completo cuando hay capas reinicializadas aleatoriamente **destruye** los pesos pre-entrenados en las primeras iteraciones. El warmup (subir `lr` linealmente de `lr/1000` a `lr` durante ~26 iteraciones) permite que los pesos aleatorios "se asienten" sin causar danos.

### 4. El offset `+1` en los labels es facil de olvidar

En la convencion de torchvision, `0 = background` y las clases reales empiezan en `1`. Si tu dataset usa `0` para tu primera clase, hay que sumar 1. Si olvidas hacerlo, el modelo aprendera que tu clase es fondo → entrenamiento roto.

### 5. RoIAlign + FPN son las dos piezas que hacen funcionar la version moderna

El paper Faster R-CNN original (2015) usaba VGG-16 + RoIPool + single feature map. La version de torchvision usa **ResNet-50 + FPN + RoIAlign**, heredando lo mejor de papers posteriores (ResNet 2015, FPN 2017, Mask R-CNN 2017). Esto da +5-10 puntos de mAP gratis.

### 6. La metrica COCO `mAP@[.5:.95]` es mucho mas estricta que `mAP@0.5`

El lab usa `IOU_POSITIVE_THRESHOLD=0.7` (estricto). Si reportas resultados, especifica siempre el threshold de IoU usado. Un detector con "70% mAP" puede ser 80% mAP@0.5 y 30% mAP@0.75.

---

## Cierre

El lab cubrio el ciclo completo de un detector moderno: arquitectura, inferencia con modelo pre-entrenado, motivacion para fine-tuning, dataset preparation, reemplazo del clasificador, loop de entrenamiento, evaluacion con TP/FP/FN, checkpointing, y carga del mejor modelo para inferencia.

Es el patron canonico que usaras en cualquier proyecto real de deteccion de objetos: cargar un detector pre-entrenado de `torchvision.models.detection`, reemplazar la cabeza segun tus clases, fine-tunear con pocos datos, y desplegar.
