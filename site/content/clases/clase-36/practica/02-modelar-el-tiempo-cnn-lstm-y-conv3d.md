---
title: "Modelar el tiempo: CNN+LSTM y Conv3D desde cero"
weight: 2
math: true
---

El [camino 01](/clases/clase-36/practica/01-clasificador-2d-cnn-y-fusion-temporal) demostró que el **2D CNN + promedio** fracasa (accuracy de azar) en una tarea donde la etiqueta depende del **orden** de los frames, porque la agregación simétrica es invariante a permutaciones. Este capítulo aplica las dos soluciones que propone la Clase 36 —el **2D CNN + RNN** ([LRCN](/papers/lrcn-donahue-2015)) y la **convolución 3D** ([C3D](/papers/c3d-tran-2015))— y verifica que **ambas resuelven la tarea**, recuperando el orden temporal. Reutilizamos el dataset de "puntos que cruzan la grilla" del camino 01.

> **Lecturas de apoyo:** los fundamentos [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) y la [Clase 11 (RNN/LSTM)](/clases/clase-11); la [profundización](/clases/clase-36/profundizacion) de la clase.

---

## 1. Solución A: 2D CNN + LSTM (LRCN)

La idea de LRCN: una **CNN** extrae features por frame, y una **LSTM** los procesa **en orden**, manteniendo un estado que recuerda la historia. Al no ser simétrica en el tiempo, la LSTM **sí distingue** una secuencia de su inversa.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class FrameEncoder(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.conv = nn.Conv2d(1, d, 3, padding=1); self.d = d
    def forward(self, x):                       # [N, 1, H, W]
        return F.relu(self.conv(x)).mean(dim=(2, 3))   # [N, d]

class CNNLSTM(nn.Module):
    def __init__(self, d=8, hidden=16):
        super().__init__()
        self.enc = FrameEncoder(d)
        self.lstm = nn.LSTM(d, hidden, batch_first=True)   # <-- procesa la secuencia
        self.fc = nn.Linear(hidden, 2)
    def forward(self, clip):                    # [B, T, H, W]
        B, T, H, W = clip.shape
        feats = self.enc(clip.reshape(B*T, 1, H, W)).reshape(B, T, -1)
        out, _ = self.lstm(feats)               # respeta el orden temporal
        return self.fc(out[:, -1])              # último estado -> logits
```

## 2. Solución B: convolución 3D (C3D)

La convolución 3D usa un kernel que se extiende también en el **tiempo** ($k_t \times k_h \times k_w$), capturando el movimiento directamente. En PyTorch, `nn.Conv3d` opera sobre `[B, C, T, H, W]`.

```python
class Conv3DNet(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.conv = nn.Conv3d(1, d, kernel_size=3, padding=1)   # kernel 3x3x3
        self.fc = nn.Linear(d, 2)
    def forward(self, clip):                    # [B, T, H, W]
        x = clip.unsqueeze(1)                   # [B, 1, T, H, W]
        h = F.relu(self.conv(x)).mean(dim=(2, 3, 4))   # pooling espacio-temporal
        return self.fc(h)
```

## 3. La prueba: ambas resuelven la tarea

Reusamos `make_dataset` y `train_eval` del camino 01, y comparamos las tres arquitecturas:

```python
from itertools import starmap
X, Y = make_dataset()                           # del camino 01

print("2D CNN + promedio :", round(train_eval(TwoDAvg(),  X, Y), 3))   # ~0.50 (azar)
print("2D CNN + LSTM     :", round(train_eval(CNNLSTM(),  X, Y), 3))   # ~1.00
print("Conv3D            :", round(train_eval(Conv3DNet(), X, Y), 3))  # ~1.00
```

El contraste es contundente: el modelo base queda en el azar, mientras que **CNN+LSTM y Conv3D alcanzan ~100%**. La diferencia no es la CNN (es prácticamente la misma en los tres) ni la cantidad de datos: es que las dos últimas **respetan el orden** de los frames.

{{< concept-alert type="clave" >}}
Ambas soluciones capturan el orden, pero de forma distinta: la **LSTM** lo hace **secuencialmente** (procesa frame por frame, no paralelizable —la desventaja que menciona la clase), mientras que la **Conv3D** lo hace con un kernel que abarca varios frames a la vez (paralelizable, pero con más parámetros y memoria). Es exactamente el trade-off entre las familias de arquitecturas de video que recorre la [profundización](/clases/clase-36/profundizacion).
{{< /concept-alert >}}

---

## 4. Modelar el tiempo en triple framework

El componente temporal —LSTM o Conv3D— existe nativamente en los tres frameworks. Aquí las piezas clave.

### TensorFlow

```python
import tensorflow as tf

# CNN + LSTM: TimeDistributed aplica la CNN a cada frame, LSTM procesa la secuencia
def build_cnn_lstm_tf(T, H, W):
    inp = tf.keras.Input((T, H, W, 1))
    x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"))(inp)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    x = tf.keras.layers.LSTM(16)(x)             # respeta el orden
    return tf.keras.Model(inp, tf.keras.layers.Dense(2)(x))

# Conv3D: kernel espacio-temporal
def build_conv3d_tf(T, H, W):
    inp = tf.keras.Input((T, H, W, 1))
    x = tf.keras.layers.Conv3D(8, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(2)(x))
```

### JAX (Flax)

```python
import jax.numpy as jnp
from flax import linen as fnn

class Conv3DJAX(fnn.Module):
    @fnn.compact
    def __call__(self, clip):                   # [B, T, H, W]
        x = clip[..., None]                     # [B, T, H, W, 1]
        h = fnn.relu(fnn.Conv(8, (3, 3, 3), padding="SAME")(x))
        h = h.mean(axis=(1, 2, 3))              # pooling espacio-temporal
        return fnn.Dense(2)(h)

# Para CNN+LSTM en JAX se usa fnn.RNN(fnn.LSTMCell(...)) sobre los features por frame,
# con la CNN aplicada vía jax.vmap sobre el eje temporal.
```

En los tres, la receta es la misma que enseña la clase: reemplazar la agregación temporal ciega por un módulo que **respeta el orden** —recurrente (LSTM) o convolucional en el tiempo (Conv3D).

---

## 5. Qué nos llevamos

- El **2D CNN + LSTM** ([LRCN](/papers/lrcn-donahue-2015)) y la **convolución 3D** ([C3D](/papers/c3d-tran-2015)) resuelven la tarea de orden donde el modelo base fracasa —evidencia directa de por qué el video necesita arquitecturas especializadas.
- La **LSTM** modela el tiempo secuencialmente (no paralelizable); la **Conv3D**, con kernels espacio-temporales (paralelizable, más costosa). Es el trade-off entre las familias de la clase.
- El componente temporal es nativo en PyTorch, TensorFlow y JAX; lo que cambia el resultado no es el framework, sino **respetar el orden de los frames**.

De aquí en adelante, el campo escala estas ideas: [two-stream](/papers/two-stream-simonyan-2014) agrega flujo óptico, [I3D](/papers/i3d-carreira-2017) infla modelos 2D pre-entrenados, y [TSN](/papers/tsn-wang-2016) muestrea segmentos —pero todas parten de la misma lección que acabamos de ver: **modelar el movimiento requiere modelar el tiempo**.

---

**Ver también:** [Clase 36 - Teoría](/clases/clase-36/teoria) · [Clase 36 - Profundización](/clases/clase-36/profundizacion) · [Camino 01: 2D CNN + fusión temporal](/clases/clase-36/practica/01-clasificador-2d-cnn-y-fusion-temporal) · [Laboratorio](/laboratorios/lab-36).
