---
title: "Teoria - Redes Convolucionales y AlexNet"
weight: 10
math: true
---

## 1. Que es una Red Convolucional (CNN)

Una red convolucional es una arquitectura de red neuronal disenada para procesar datos con estructura espacial (imagenes). A diferencia de una red densa (MLP), las CNNs explotan que los pixeles cercanos estan relacionados entre si, usando **filtros** que se deslizan sobre la imagen.

| Tipo de capa | Que hace |
|---|---|
| **Convolucion** | Aplica un filtro para detectar patrones locales (bordes, texturas, formas) |
| **Pooling** | Reduce el tamano espacial, concentrando la informacion mas relevante |
| **Fully Connected (FC)** | Combina todas las caracteristicas para producir una prediccion |

---

## 2. Como funciona una Convolucion por dentro

Un filtro es una pequena matriz de numeros (pesos). Se **desliza** sobre la imagen y en cada posicion hace un **producto punto** entre el filtro y el parche de imagen que cubre.

```text
Imagen (fragmento 3x3):    Filtro (3x3):         Producto punto:
+---+---+---+              +----+----+----+
| 1 | 2 | 3 |              |  0 | -1 |  0 |      1x0  + 2x(-1) + 3x0
+---+---+---+       x      +----+----+----+   =  4x(-1)+ 5x5   + 6x(-1)  = 5
| 4 | 5 | 6 |              | -1 |  5 | -1 |      7x0  + 8x(-1) + 9x0
+---+---+---+              +----+----+----+
| 7 | 8 | 9 |              |  0 | -1 |  0 |
+---+---+---+              +----+----+----+
```

La red **aprende los valores del filtro** durante el entrenamiento.

---

## 3. Filtro Laplaciano

El Laplaciano es un ejemplo clasico de filtro detector de bordes. Compara cada pixel con sus vecinos: si son similares (zona uniforme), el resultado es cercano a 0. Si el pixel central es muy distinto (borde), el resultado es alto.

```text
Filtro Laplaciano:
+----+----+----+
|  0 | -1 |  0 |
+----+----+----+
| -1 |  4 | -1 |
+----+----+----+
|  0 | -1 |  0 |
+----+----+----+
```

La red no usa el Laplaciano explicitamente, pero los filtros que aprende en conv1 terminan siendo matematicamente similares.

### Ejemplo: Convolucion con filtro personalizado

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torch.nn.functional as F

# Definir imagen de ejemplo (1 batch, 1 canal, 5x5)
imagen = torch.rand(1, 1, 5, 5)

# Filtro Laplaciano como kernel personalizado
filtro = torch.tensor([[0., -1., 0.],
                       [-1., 4., -1.],
                       [0., -1., 0.]]).reshape(1, 1, 3, 3)

# Aplicar convolucion manualmente
salida = F.conv2d(imagen, filtro, padding=1)
print("Forma de salida:", salida.shape)  # (1, 1, 5, 5)
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

# Definir imagen de ejemplo (1 batch, 5x5, 1 canal)
imagen = tf.random.uniform((1, 5, 5, 1))

# Filtro Laplaciano como kernel personalizado (alto, ancho, canales_in, canales_out)
filtro = tf.constant([[0., -1., 0.],
                      [-1., 4., -1.],
                      [0., -1., 0.]], shape=(3, 3, 1, 1))

# Aplicar convolucion manualmente
salida = tf.nn.conv2d(imagen, filtro, strides=1, padding="SAME")
print("Forma de salida:", salida.shape)  # (1, 5, 5, 1)
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax
import jax.numpy as jnp
from jax import lax

# Definir imagen de ejemplo (1 batch, 1 canal, 5x5)
imagen = jax.random.uniform(jax.random.key(0), (1, 1, 5, 5))

# Filtro Laplaciano como kernel personalizado
filtro = jnp.array([[0., -1., 0.],
                    [-1., 4., -1.],
                    [0., -1., 0.]]).reshape(1, 1, 3, 3)

# Aplicar convolucion con lax.conv
salida = lax.conv(imagen, filtro, window_strides=(1, 1), padding="SAME")
print("Forma de salida:", salida.shape)  # (1, 1, 5, 5)
```
{{< /tab >}}
{{< /tabs >}}

---

## 4. Invarianza a Traslaciones

{{< concept-alert type="clave" >}}
**MaxPool** proporciona invarianza a pequenos desplazamientos: aunque la activacion se mueva un pixel, el maximo de la ventana sigue siendo el mismo. Esto permite que la red reconozca objetos independientemente de su posicion exacta.
{{< /concept-alert >}}

```text
Activacion en (2,2):        Imagen movida, activacion en (2,3):
+---+---+---+---+           +---+---+---+---+
| 0 | 0 | 0 | 0 |           | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |           | 0 | 0 | 0 | 0 |
| 0 | 9 | 0 | 0 |           | 0 | 0 | 9 | 0 |
| 0 | 0 | 0 | 0 |           | 0 | 0 | 0 | 0 |

MaxPool 2x2: max = 9        MaxPool 2x2: max = 9  (mismo resultado)
```

---

## 5. Jerarquia de Caracteristicas

Cada capa ve la salida de la anterior, no los pixeles originales. Esto produce una jerarquia emergente:

```mermaid
graph TD
    C1["conv1<br/>Bordes, gradientes de color, orientaciones"]:::l1
    C2["conv2<br/>Esquinas, texturas"]:::l2
    C3["conv3<br/>Formas geometricas, patrones"]:::l3
    C4["conv4<br/>Partes de objetos (ojos, ruedas, hojas)"]:::l4
    C5["conv5<br/>Objetos completos"]:::l5

    C1 --> C2 --> C3 --> C4 --> C5

    classDef l1 fill:#1e40af,color:#fff,stroke:#1e3a8a
    classDef l2 fill:#2563eb,color:#fff,stroke:#1e40af
    classDef l3 fill:#3b82f6,color:#fff,stroke:#2563eb
    classDef l4 fill:#60a5fa,color:#fff,stroke:#3b82f6
    classDef l5 fill:#059669,color:#fff,stroke:#047857
```

Esta jerarquia no esta impuesta: emerge porque es la estrategia mas eficiente para reducir el error de clasificacion.

### Ejemplo: Visualizacion de mapas de caracteristicas

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torchvision.models as models
import matplotlib.pyplot as plt

# Cargar modelo preentrenado y una imagen de ejemplo
modelo = models.alexnet(pretrained=True).eval()
imagen = torch.rand(1, 3, 224, 224)  # Reemplazar con imagen real

# Extraer salida de conv1
conv1_salida = modelo.features[0](imagen)  # (1, 64, 55, 55)

# Visualizar los primeros 16 mapas de caracteristicas
fig, ejes = plt.subplots(4, 4, figsize=(8, 8))
for i, eje in enumerate(ejes.flat):
    eje.imshow(conv1_salida[0, i].detach().numpy(), cmap="viridis")
    eje.axis("off")
plt.suptitle("Mapas de caracteristicas - conv1")
plt.show()
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar modelo preentrenado
modelo_base = tf.keras.applications.VGG16(weights="imagenet")
imagen = tf.random.uniform((1, 224, 224, 3))  # Reemplazar con imagen real

# Crear modelo parcial hasta la primera capa conv
modelo_parcial = tf.keras.Model(
    inputs=modelo_base.input,
    outputs=modelo_base.layers[1].output  # Primera conv
)
conv1_salida = modelo_parcial(imagen)  # (1, 224, 224, 64)

# Visualizar los primeros 16 mapas de caracteristicas
fig, ejes = plt.subplots(4, 4, figsize=(8, 8))
for i, eje in enumerate(ejes.flat):
    eje.imshow(conv1_salida[0, :, :, i].numpy(), cmap="viridis")
    eje.axis("off")
plt.suptitle("Mapas de caracteristicas - conv1")
plt.show()
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt

# Definir capa convolucional simple
capa_conv = nn.Conv(features=16, kernel_size=(3, 3))
imagen = jax.random.uniform(jax.random.key(0), (1, 224, 224, 3))

# Inicializar parametros y obtener salida
params = capa_conv.init(jax.random.key(1), imagen)
conv1_salida = capa_conv.apply(params, imagen)  # (1, 222, 222, 16)

# Visualizar los primeros 16 mapas de caracteristicas
fig, ejes = plt.subplots(4, 4, figsize=(8, 8))
for i, eje in enumerate(ejes.flat):
    eje.imshow(conv1_salida[0, :, :, i], cmap="viridis")
    eje.axis("off")
plt.suptitle("Mapas de caracteristicas - conv1")
plt.show()
```
{{< /tab >}}
{{< /tabs >}}

---

## 6. Formulas clave

### Dimension de salida de una capa convolucional o de pooling

$$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$

Donde:
- $I$ = tamano de entrada
- $K$ = tamano del kernel
- $P$ = padding
- $S$ = stride

### Cantidad de parametros de una capa Conv2d

$$\text{params} = C_{out} \times (C_{in} \times K_H \times K_W + 1)$$

### Cantidad de parametros de una capa Linear

$$\text{params} = \text{out\_features} \times (\text{in\_features} + 1)$$

---

## 7. Arquitectura Original de AlexNet

AlexNet (2012) clasificaba imagenes del dataset **ImageNet** (1000 categorias) con entradas de 3 x 224 x 224 (RGB).

### Diagrama de flujo de datos

```mermaid
graph TD
    IN["Input<br/>3 x 224 x 224"]:::input
    C1["conv1 — 96 x 55 x 55<br/>MaxPool — 96 x 27 x 27"]:::conv
    C2["conv2 — 256 x 27 x 27<br/>MaxPool — 256 x 13 x 13"]:::conv
    C3["conv3 — 384 x 13 x 13"]:::conv
    C4["conv4 — 384 x 13 x 13"]:::conv
    C5["conv5 — 256 x 13 x 13<br/>MaxPool — 256 x 6 x 6"]:::conv
    FL["Flatten — 9216"]:::flat
    F6["fc6 — 4096"]:::fc
    F7["fc7 — 4096"]:::fc
    F8["fc8 — 1000 clases"]:::output

    IN --> C1 --> C2 --> C3 --> C4 --> C5 --> FL --> F6 --> F7 --> F8

    classDef input fill:#1e40af,color:#fff,stroke:#1e3a8a
    classDef conv fill:#2563eb,color:#fff,stroke:#1e40af
    classDef flat fill:#6b7280,color:#fff,stroke:#4b5563
    classDef fc fill:#3b82f6,color:#fff,stroke:#2563eb
    classDef output fill:#059669,color:#fff,stroke:#047857
```

### Capa a capa

| Capa | Configuracion | Salida | Parametros | Detecta |
|------|--------------|--------|-----------|---------|
| conv1 | 96 filtros 11x11, stride=4 | 96 x 27 x 27 | 34,944 | Bordes, colores basicos |
| conv2 | 256 filtros 5x5 | 256 x 13 x 13 | 614,656 | Texturas, esquinas |
| conv3 | 384 filtros 3x3 | 384 x 13 x 13 | 885,120 | Formas complejas |
| conv4 | 384 filtros 3x3 | 384 x 13 x 13 | 1,327,488 | Partes de objetos |
| conv5 | 256 filtros 3x3 | 256 x 6 x 6 | 884,992 | Objetos completos |
| fc6 | Linear(9216, 4096) | 4096 | 37,752,832 | - |
| fc7 | Linear(4096, 4096) | 4096 | 16,781,312 | - |
| fc8 | Linear(4096, 1000) | 1000 | 4,097,000 | Clasificacion |
| **Total** | | | **62,378,344** | |

> Las capas FC concentran ~95% de los parametros, aunque son solo 3 capas.

### Ejemplo: Definicion de AlexNet

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_clases=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, num_clases),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Aplanar
        return self.classifier(x)
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def crear_alexnet(num_clases=1000):
    """Definir AlexNet usando la API funcional de Keras."""
    entrada = layers.Input(shape=(224, 224, 3))
    # Capas convolucionales
    x = layers.Conv2D(96, 11, strides=4, padding="valid", activation="relu")(entrada)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(256, 5, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    # Clasificador
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    salida = layers.Dense(num_clases)(x)
    return Model(inputs=entrada, outputs=salida)
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
from flax import linen as nn
import jax.numpy as jnp

class AlexNet(nn.Module):
    """AlexNet implementado con Flax/Linen."""
    num_clases: int = 1000

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Capas convolucionales
        x = nn.relu(nn.Conv(96, kernel_size=(11, 11), strides=(4, 4))(x))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.relu(nn.Conv(256, kernel_size=(5, 5), padding="SAME")(x))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.relu(nn.Conv(384, kernel_size=(3, 3), padding="SAME")(x))
        x = nn.relu(nn.Conv(384, kernel_size=(3, 3), padding="SAME")(x))
        x = nn.relu(nn.Conv(256, kernel_size=(3, 3), padding="SAME")(x))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        # Clasificador
        x = x.reshape((x.shape[0], -1))  # Aplanar
        x = nn.Dense(4096)(x); x = nn.relu(x); x = nn.Dropout(0.5)(x, deterministic=not train)
        x = nn.Dense(4096)(x); x = nn.relu(x); x = nn.Dropout(0.5)(x, deterministic=not train)
        return nn.Dense(self.num_clases)(x)
```
{{< /tab >}}
{{< /tabs >}}

---

## 8. Actividad 1 — Adaptar para 102 clases

Para un dataset de 102 clases (ej. Oxford Flowers), solo la **ultima capa** `fc8` necesita cambiar:

| Capa | Original | Modificado |
|------|----------|-----------|
| fc8 | `Linear(4096, 1000)` | `Linear(4096, 102)` |

```python
class MiAlexNet(nn.Module):
    def __init__(self):
        super(MiAlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11,11), stride=(4,4), padding=(2,2)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )
        # ... conv2-conv5 sin cambios ...
        self.fc8 = nn.Sequential(nn.Linear(4096, 102))  # CAMBIO: 1000 -> 102
```

---

## 9. Actividad 2 — Adaptar para imagenes 64 x 64

{{< concept-alert type="clave" >}}
Con imagenes de 64x64, el kernel=11 y stride=4 de conv1 son demasiado agresivos. Las dimensiones colapsan antes de llegar a fc6. La solucion es modificar conv1 y conv2 para preservar las dimensiones espaciales hasta llegar a 6x6 antes del flatten.
{{< /concept-alert >}}

### Cambios necesarios

| Capa | Original | Modificado | Razon |
|------|----------|-----------|-------|
| conv1 kernel | (11,11) | **(3,3)** | Adecuado para imagen pequena |
| conv1 stride | (4,4) | **(1,1)** | Evita colapso espacial |
| conv1 padding | (2,2) | **(1,1)** | Mantiene salida 64x64 |
| conv2 padding | (2,2) | **(0,0)** | Reduce 31 a 27 para llegar a 13 tras MaxPool |

### Nuevo flujo de dimensiones

```mermaid
graph TD
    IN["Input<br/>3 x 64 x 64"]:::input
    C1["conv1 (k=3, s=1, p=1)<br/>MaxPool — 96 x 31 x 31"]:::conv
    C2["conv2 (k=5, s=1, p=0)<br/>MaxPool — 256 x 13 x 13"]:::conv
    C34["conv3, conv4<br/>384 x 13 x 13"]:::conv
    C5["conv5 + MaxPool<br/>256 x 6 x 6"]:::conv
    FL["Flatten — 9216<br/><small>fc6 sin cambios</small>"]:::flat

    IN --> C1 --> C2 --> C34 --> C5 --> FL

    classDef input fill:#1e40af,color:#fff,stroke:#1e3a8a
    classDef conv fill:#2563eb,color:#fff,stroke:#1e40af
    classDef flat fill:#6b7280,color:#fff,stroke:#4b5563
```
