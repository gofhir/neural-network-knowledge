---
title: "Clasificador 2D CNN + fusión temporal desde cero"
weight: 1
math: true
---

La [teoría de la Clase 36](/clases/clase-36/teoria) advierte que el enfoque más simple para clasificar video —pasar cada frame por una **CNN 2D** y agregar las predicciones— **descarta el sentido temporal**. Este capítulo lo construye desde cero y, sobre una tarea de juguete diseñada a propósito, **demuestra empíricamente** esa limitación: el modelo alcanza apenas el azar en una tarea que depende del orden de los frames. Es la falla que motiva todo el resto de la clase.

> **Lecturas de apoyo:** el fundamento [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones); la [profundización](/clases/clase-36/profundizacion) formaliza por qué la agregación por promedio es invariante al orden.

---

## 1. Una tarea donde el orden ES la etiqueta

Para exponer la limitación necesitamos una tarea cuya respuesta dependa **solo del orden** de los frames. Construimos "videos" de juguete: un punto blanco que se desplaza sobre una grilla. Si se mueve de **izquierda a derecha**, la etiqueta es `0`; de **derecha a izquierda**, la etiqueta es `1`. **Ambas clases contienen exactamente los mismos frames** —solo cambia su orden.

```python
import numpy as np
rng = np.random.default_rng(0)

def make_clip(direction, T=6, H=8, W=8):
    """Un punto que cruza la grilla. direction=+1 (izq→der) o -1 (der→izq)."""
    clip = np.zeros((T, H, W), dtype=np.float32)
    row = rng.integers(1, H-1)
    cols = np.linspace(1, W-2, T).astype(int)
    if direction < 0: cols = cols[::-1]          # invierte el orden temporal
    for t, c in enumerate(cols):
        clip[t, row, c] = 1.0
    return clip

def make_dataset(n=2000, T=6):
    X, Y = [], []
    for _ in range(n):
        d = rng.choice([+1, -1])
        X.append(make_clip(d, T)); Y.append(0 if d > 0 else 1)
    return np.array(X), np.array(Y, dtype=np.int64)

X, Y = make_dataset()
print(X.shape, Y.shape)   # (2000, 6, 8, 8)  (2000,)
```

{{< concept-alert type="clave" >}}
El conjunto de frames de un clip "izquierda→derecha" es **idéntico** al de su versión "derecha→izquierda" —los mismos 6 frames, en orden inverso. Cualquier modelo que **no mire el orden** verá las dos clases como indistinguibles. Esta tarea es un detector de si una arquitectura modela el tiempo.
{{< /concept-alert >}}

---

## 2. El modelo base: CNN por frame + promedio

El enfoque de la clase: una CNN 2D $\phi$ produce una predicción por frame, y se **promedian** (fusión temporal por *average pooling*). En PyTorch:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class FrameCNN(nn.Module):
    """CNN 2D pequeña: procesa UN frame -> logits de 2 clases."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.fc = nn.Linear(8, 2)
    def forward(self, x):                      # x: [B, 1, H, W]
        h = F.relu(self.conv(x))
        h = h.mean(dim=(2, 3))                 # global average pooling espacial
        return self.fc(h)                      # [B, 2]

class TwoDAvg(nn.Module):
    """Aplica la CNN a cada frame y PROMEDIA las predicciones (fusión temporal)."""
    def __init__(self):
        super().__init__(); self.frame = FrameCNN()
    def forward(self, clip):                   # clip: [B, T, H, W]
        B, T, H, W = clip.shape
        x = clip.reshape(B*T, 1, H, W)
        logits = self.frame(x).reshape(B, T, 2)
        return logits.mean(dim=1)              # <-- promedio temporal (simétrico)
```

Entrenémoslo y midamos su accuracy:

```python
def train_eval(model, X, Y, epochs=30, lr=1e-2):
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt), Yt)
        loss.backward(); opt.step()
    acc = (model(Xt).argmax(1) == Yt).float().mean().item()
    return acc

print("2D CNN + promedio:", round(train_eval(TwoDAvg(), X, Y), 3))   # ~0.50
```

El resultado es **~0.50 —azar puro**. El modelo **no puede** separar las dos clases, por más que entrenemos. No es un problema de capacidad ni de datos: es **estructural**.

---

## 3. Por qué falla: invarianza al orden

La razón es matemática. La fusión por promedio calcula

$$
\hat y = \frac{1}{T}\sum_{t=1}^{T} \phi(x_t),
$$

y una **suma es invariante a permutaciones**: reordenar los frames $x_1, \dots, x_T$ no cambia el resultado. Como los dos clips (izq→der y der→izq) contienen los **mismos frames**, producen la **misma** predicción promedio —el modelo les asigna, necesariamente, la misma clase. Cualquier agregación simétrica (promedio, max, suma) tiene este defecto.

{{< concept-alert type="advertencia" >}}
Esta es exactamente la limitación que enumera la clase: el 2D CNN por frame **descarta el sentido temporal y el movimiento**. Nuestra tarea de juguete la aísla en su forma más pura, pero el mismo problema afecta a acciones reales: *abrir* vs. *cerrar* una puerta, *sentarse* vs. *pararse*, *empujar* vs. *tirar* —todas comparten frames y se distinguen por el **orden**.
{{< /concept-alert >}}

---

## 4. El mismo modelo base en TensorFlow y JAX

Para dejar claro que la limitación es del **enfoque**, no del framework, aquí el modelo base en los otros dos. El promedio temporal —el punto débil— es la misma línea en todos.

### TensorFlow

```python
import tensorflow as tf

class TwoDAvgTF(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")
        self.fc = tf.keras.layers.Dense(2)
    def call(self, clip):                       # [B, T, H, W]
        B, T, H, W = clip.shape
        x = tf.reshape(clip, (B*T, H, W, 1))
        h = tf.reduce_mean(self.conv(x), axis=(1, 2))
        logits = tf.reshape(self.fc(h), (B, T, 2))
        return tf.reduce_mean(logits, axis=1)   # promedio temporal
```

### JAX (Flax)

```python
import jax.numpy as jnp
from flax import linen as fnn

class TwoDAvgJAX(fnn.Module):
    @fnn.compact
    def __call__(self, clip):                   # [B, T, H, W]
        B, T, H, W = clip.shape
        x = clip.reshape(B*T, H, W, 1)
        h = fnn.relu(fnn.Conv(8, (3, 3), padding="SAME")(x)).mean(axis=(1, 2))
        logits = fnn.Dense(2)(h).reshape(B, T, 2)
        return logits.mean(axis=1)              # promedio temporal
```

Las tres colapsan el eje temporal con un promedio —y las tres fallan igual en la tarea de orden. El framework no cambia la conclusión.

---

## 5. Qué nos llevamos

- El **2D CNN + fusión por promedio** es el enfoque base para clasificar video: simple y paralelizable.
- Pero la agregación simétrica es **invariante al orden** de los frames: matemáticamente incapaz de distinguir acciones que solo difieren en su secuencia temporal.
- En nuestra tarea de juguete esto se ve como **accuracy de azar (~50%)**, y es una limitación **estructural**, no de entrenamiento.

En el [camino 02](/clases/clase-36/practica/02-modelar-el-tiempo-cnn-lstm-y-conv3d) recuperamos el orden con una **LSTM** (procesa los frames en secuencia) y con una **convolución 3D** (kernel que se extiende en el tiempo), y vemos la accuracy saltar al 100%.

---

**Ver también:** [Clase 36 - Teoría](/clases/clase-36/teoria) · [Clase 36 - Profundización](/clases/clase-36/profundizacion) · [Camino 02: Modelar el tiempo](/clases/clase-36/practica/02-modelar-el-tiempo-cnn-lstm-y-conv3d) · [Laboratorio](/laboratorios/lab-36).
