---
title: "Disección del output"
weight: 3
math: true
---

> **Celdas 19-37 del notebook.** La parte conceptualmente más importante: pasar de "mirar imágenes bonitas" a entender la salida de ABCNet **bit a bit** —desde los puntos de control Bézier y los índices de caracteres hasta el texto legible.

## Del script al `DefaultPredictor` (celdas 20-21)

```python
from detectron2.engine.defaults import DefaultPredictor
from adet.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file('AdelaiDet/configs/BAText/TotalText/attn_R_50.yaml')
cfg.MODEL.WEIGHTS = 'tt_attn_R_50.pth'
predictor = DefaultPredictor(cfg)
```

`get_cfg()` se importa de `adet.config`, **no** de `detectron2.config`: AdelaiDet extiende el sistema de config con claves propias de ABCNet (`MODEL.BATEXT.*`, charset, cabeza Bézier). El config funciona por **capas de override**: defaults → `merge_from_file` (el `.yaml` del modelo) → asignación directa (`cfg.MODEL.WEIGHTS`).

`DefaultPredictor(cfg)` hace todo de una: construye la arquitectura, **carga los pesos**, pone el modelo en `eval()`, lo mueve a GPU y configura el preprocesamiento (resize, BGR↔RGB, normalización).

> **Gotcha de formato de color:** `DefaultPredictor` espera **BGR** por defecto (herencia de OpenCV). `read_image(path)` sin `format=` devuelve **RGB**. El texto es bastante robusto al intercambio de canales (las formas de las letras no dependen del color), por eso aquí funciona; pero lo explícito y correcto es `read_image(path, format="BGR")`. Más adelante, en las apps, el profe sí lo usa.

## El objeto `Instances` (celdas 22-26)

```python
img = read_image('datasets/totaltext/test_images/0000089.jpg')
pred = predictor(img)           # → {'instances': Instances(...)}
instances = pred['instances']
```

`Instances` es una estructura tipo "tabla de columnas" de Detectron2: cada **fila es una detección** y cada **columna un atributo** (todas de longitud N). ABCNet rellena:

| Campo | Qué es | Forma |
|---|---|---|
| `instances.scores` | confianza por detección | (N,) |
| `instances.beziers` | puntos de control de las curvas | (N, 16) |
| `instances.recs` | índices de caracteres reconocidos | (N, 25) |

## `instances.beziers`: el corazón de ABCNet

Para `0000089.jpg`, `instances.beziers` tiene forma **(12, 16)** — 12 palabras, 16 valores cada una. Esos 16 números codifican **dos curvas Bézier cúbicas**:

```
[ x0 y0  x1 y1  x2 y2  x3 y3 | x4 y4  x5 y5  x6 y6  x7 y7 ]
  └──── curva SUPERIOR ─────┘   └──── curva INFERIOR ─────┘
   4 puntos de control × 2        4 puntos de control × 2
```

`16 = 2 curvas × 4 puntos × 2 coordenadas`. Es exactamente la representación de la [teoría de la clase 21](/clases/clase-21): la mínima parametrización que captura cualquier curvatura suave de texto. Las coordenadas están en **píxeles absolutos** de la imagen original.

### Geometría real de tres palabras

**Fila 0 — "TURN" (horizontal):** las `y` del borde superior son ~271 y las del inferior ~312, casi constantes. Texto plano; una caja rectangular lo describiría igual.

**Fila 2 — "COFFEE" (curvo):**
```
top:    (200, 20.7) (278.6, 22.2) (338.4, 64.3) (360.9, 139.4)   ← y sube de 21 a 139
bottom: (315, 162) (293.6, 110.3) (253.2, 74.3) (191.8, 71.5)
```
La `y` del borde superior **sube de 21 a 139 px** conforme crece `x`: "COFFEE" está **arqueado**. Una caja axis-aligned envolvería el arco con mucho fondo vacío y solaparía palabras vecinas; la Bézier lo sigue con 8 puntos. **Este es el caso que motivó ABCNet.**

**Fila 3 — "REAL":** otro arco, descendente (`y` del borde superior baja de 130 a 36).

![Disección de 0000089.jpg: 12 palabras con sus contornos Bézier y texto reconocido](/laboratorios/lab-21/diseccion-0000089.jpg)

> El borde **superior** va de izquierda→derecha y el **inferior** de derecha→izquierda; concatenando los 8 puntos se traza el **perímetro cerrado** del polígono.

### Evaluar una curva Bézier cúbica — triple framework

La fórmula es $B(t) = (1-t)^3 P_0 + 3(1-t)^2 t\,P_1 + 3(1-t)t^2 P_2 + t^3 P_3$. Así se evalúa en los tres frameworks (concepto transversal, el mismo `bezier_cubic` que ABCNet usa internamente para construir la grilla de BezierAlign):

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch

def bezier_cubic(ctrl, t):           # ctrl: (4,2), t: (N,)
    t = t.unsqueeze(1)
    b = torch.stack([(1-t)**3, 3*(1-t)**2*t,
                     3*(1-t)*t**2, t**3], dim=1).squeeze(-1)  # (N,4)
    return b @ ctrl                  # (N,2)
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

def bezier_cubic(ctrl, t):
    t = tf.reshape(t, (-1, 1))
    b = tf.concat([(1-t)**3, 3*(1-t)**2*t,
                   3*(1-t)*t**2, t**3], axis=1)               # (N,4)
    return tf.matmul(b, ctrl)        # (N,2)
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax.numpy as jnp

def bezier_cubic(ctrl, t):
    t = t[:, None]
    b = jnp.concatenate([(1-t)**3, 3*(1-t)**2*t,
                         3*(1-t)*t**2, t**3], axis=1)          # (N,4)
    return b @ ctrl                  # (N,2)
```
{{< /tab >}}
{{< /tabs >}}

> Para rectificar (BezierAlign), se evalúan ambas curvas en una grilla de `t`, se interpola entre superior e inferior y se hace **muestreo bilineal** del feature map: `grid_sample` (PyTorch) / `tfa.image.resampler` (TF) / `jax.scipy.ndimage.map_coordinates` (JAX).

## `instances.recs`: el texto en crudo + el charset

`instances.recs` tiene forma **(12, 25)** — 12 palabras, longitud máxima 25 — de **índices enteros** a un charset. El padding es **96**. El charset de AdelaiDet/BAText es ASCII imprimible empezando en el espacio:

$$\text{CTLABELS}[i] = \texttt{chr}(32 + i)\quad\text{para } i \in [0, 94]$$

Es decir: `0 → ' '`, `16 → '0'`, `17 → '1'`, `33 → 'A'`, `65 → 'a'`… Y los dos índices especiales:

- **Índice 95 → `'口'`**: AdelaiDet **hardcodea** el índice 95 como carácter "desconocido / fuera de vocabulario", usando el ideograma chino 口 como placeholder. **Este es el origen real del warning CJK del demo**: cuando el modelo duda, emite 95 y el visualizer dibuja 口.
- **Índice 96 → blank / padding (EOS)**, se omite al decodificar.

### La imagen decodificada a mano

Aplicando `chr(32+i)` a cada fila de `recs`:

| # | Índices (sin padding) | Palabra |
|---|---|---|
| 0 | 52,53,50,46 | **TURN** |
| 1 | 33,40,37,33,36 | **AHEAD** |
| 2 | 35,47,38,38,37,37 | **COFFEE** |
| 3 | 50,37,33,44 | **REAL** |
| 4 | 35,65,70,69 | **Cafe** |
| 5 | 33,44,44,37,57 | **ALLEY** |
| 6 | 52,69,65 | **Tea** |
| 7 | 50,41,39,40,52 | **RIGHT** |
| 8 | 51,45,33,44,44 | **SMALL** |
| 9 | 45,37,52,37,50,51 | **METERS** |
| 10 | 17,16 | **10** |
| 11 | 41,46 | **IN** |

La escena es un cartel de cafetería: *"REAL COFFEE · Tea · Cafe"* + direcciones *"TURN RIGHT · 10 METERS AHEAD · IN ALLEY"*. Mezcla mayúsculas y minúsculas correctamente, y leyó un número ("10").

## La decodificación oficial confirma todo (celda 37)

```python
predicted_text = []
for text_prediction in instances.recs:
    predicted_text.append(visualizer._decode_recognition(text_prediction))
print(predicted_text)
# ['TURN','AHEAD','COFFEE','REAL','Cafe','ALLEY','Tea','RIGHT','SMALL','METERS','10','IN']
```

`_decode_recognition` aplica el charset que reconstruimos. Su lógica esencial:

```python
def _decode_recognition(self, rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < 95:        s += self.CTLABELS[c]   # ASCII imprimible
        elif c == 95:     s += u'口'              # ← el glifo del warning
        # c == 96 (blank) → se omite
    return s
```

La salida coincide **exactamente** con la decodificación a mano. Tres cosas confirmadas de golpe: (1) la fórmula `chr(32+i)` es correcta; (2) el índice **95 = 口** explica el warning CJK; (3) el índice **96 = blank** se omite.

> El `TextVisualizer` de `adet` (celda 31) es el que dibuja las curvas y el texto. Requiere `instances.to('cpu')`: las predicciones viven en GPU (`device='cuda:0'`), pero matplotlib/NumPy solo operan en CPU. Olvidar el `.to('cpu')` da `TypeError: can't convert cuda:0 device tensor to numpy`. Es un patrón universal en PyTorch: **bajar a CPU antes de visualizar**.

## Lo que cierra el círculo

1. **Teoría:** ABCNet representa texto curvo con 2 curvas Bézier = 16 parámetros.
2. **Datos reales:** `beziers[2]` son los 16 números de "COFFEE", y su curva superior realmente se arquea (y: 21→139).
3. **Rectificación:** BezierAlign endereza ese arco a un rectángulo.
4. **Reconocimiento:** `recs[2] = [35,47,38,38,37,37]` → "COFFEE".
5. **Charset:** índice 95 = 口 (el warning), 96 = blank (el padding).

---

**Anterior:** [demo end-to-end](demo-abcnet) · **Siguiente:** [App 1 · Freiburg Groceries](app-groceries)
