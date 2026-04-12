---
title: "Conceptos Fundamentales de CNNs"
weight: 30
math: true
---

> Preguntas y respuestas sobre como operan internamente las redes convolucionales.

---

## 1. Que es un mapa de activacion

Un mapa de activacion es el resultado de **un filtro recorriendo toda la imagen**. Es una grilla 2D donde cada valor indica "cuanto se parece esta region al patron que busca el filtro".

```text
Imagen 224x224x3
       |
       |  Filtro #0 (3x3x3) pasa por TODAS las 224x224 posiciones
       |
       v
Mapa de activacion #0: 224x224 valores

  Valor alto (ej: 8.5) -> "encontre el patron aqui"
  Valor bajo (ej: 0.1) -> "no hay nada relevante aqui"
```

Cada filtro siempre produce exactamente **un** mapa de activacion.

### Ejemplo: Extraer mapas de activacion de capas intermedias

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# Cargar modelo preentrenado
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.eval()

# Preparar imagen
img = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(Image.open("gato.jpg")).unsqueeze(0)

# Extraer features de capas intermedias usando hooks
features = {}
def guardar_activacion(nombre):
    def hook(modelo, entrada, salida):
        features[nombre] = salida.detach()
    return hook

model.features[4].register_forward_hook(guardar_activacion("bloque1"))   # tras bloque 1
model.features[16].register_forward_hook(guardar_activacion("bloque3"))  # tras bloque 3

with torch.no_grad():
    model(img)

print(f"Bloque 1: {features['bloque1'].shape}")  # [1, 64, 112, 112]
print(f"Bloque 3: {features['bloque3'].shape}")  # [1, 256, 28, 28]
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

# Cargar modelo preentrenado
base_model = tf.keras.applications.VGG16(weights="imagenet")

# Crear modelo que retorna activaciones de capas intermedias
capas_objetivo = ["block1_pool", "block3_pool", "block5_pool"]
salidas = [base_model.get_layer(nombre).output for nombre in capas_objetivo]
modelo_features = tf.keras.Model(inputs=base_model.input, outputs=salidas)

# Preparar imagen
img = tf.keras.preprocessing.image.load_img("gato.jpg", target_size=(224, 224))
img_array = tf.expand_dims(tf.keras.applications.vgg16.preprocess_input(
    tf.keras.preprocessing.image.img_to_array(img)), 0)

# Extraer features de capas intermedias
activaciones = modelo_features.predict(img_array)
for nombre, act in zip(capas_objetivo, activaciones):
    print(f"{nombre}: {act.shape}")  # ej: block1_pool: (1, 112, 112, 64)
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class VGGConFeatures(nn.Module):
    """VGG simplificado que retorna features intermedias"""
    @nn.compact
    def __call__(self, x):
        intermedias = {}
        # Bloque 1: 2 conv + pool
        for _ in range(2):
            x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        intermedias["bloque1"] = x
        # Bloque 2: 2 conv + pool
        for _ in range(2):
            x = nn.relu(nn.Conv(128, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        intermedias["bloque2"] = x
        return x, intermedias  # retornar salida final y features intermedias
```
{{< /tab >}}
{{< /tabs >}}

---

## 2. Que significa `conv3-64`

Notacion del paper de VGG: una capa convolucional con filtros de 3x3 que tiene 64 filtros (= 64 canales de salida).

| Notacion | Kernel | Filtros |
|----------|--------|---------|
| conv3-64 | 3x3 | 64 |
| conv3-128 | 3x3 | 128 |
| conv3-512 | 3x3 | 512 |

---

## 3. De donde salen los 64 canales

De los **64 filtros** que definimos en la capa. El numero 64 es una **decision del disenador**, no un calculo.

{{< concept-alert type="clave" >}}
El patron de VGG es doblar los canales en cada bloque: 64 -> 128 -> 256 -> 512 -> 512. Esto compensa la perdida de informacion espacial del MaxPool: menos pixeles, pero mas canales para representar patrones mas abstractos.
{{< /concept-alert >}}

---

## 4. Los 64 filtros no producen lo mismo

No, porque cada filtro tiene **pesos distintos**. Los 64 filtros recorren la misma imagen, pero cada uno tiene sus propios valores en el kernel 3x3x3.

```text
Mismo parche, tres filtros distintos:

Filtro #0 (bordes verticales):  resultado = 12.3
Filtro #1 (bordes horizontales): resultado = -0.5
Filtro #2 (zonas brillantes):   resultado = 85.7
```

Al inicio del entrenamiento los pesos son aleatorios. Despues del entrenamiento con backpropagation, cada filtro converge a detectar un patron util.

---

## 5. Numero de filtros vs. posiciones

Son independientes:

| Concepto | Que determina | Quien lo decide |
|----------|--------------|-----------------|
| **Posiciones** (224x224) | Tamano espacial H x W | Tamano de imagen + stride + padding |
| **Filtros** (64) | Canales de salida | El disenador de la arquitectura |

$$H_{\text{out}} = \frac{H_{\text{in}} + 2P - K}{S} + 1$$

---

## 6. Que reduce el espacio y que cambia canales

| Operacion | Efecto en H x W | Efecto en Canales |
|-----------|-----------------|-------------------|
| Conv stride=1, pad=1 | No cambia | Lo define `out_channels` |
| Conv stride=2 | Divide por 2 | Lo define `out_channels` |
| MaxPool 2x2, stride=2 | Divide por 2 | No cambia |
| Conv 1x1 | No cambia | Cambia canales |
| Global Average Pool | H x W -> 1x1 | No cambia |

**Regla clave:** Las convoluciones controlan los canales; el stride y el pooling controlan el espacio.

---

## 7. Campo receptivo capa a capa

```text
Capas tempranas:  Alta resolucion x Pocos canales  -> "DONDE estan los bordes"
Capas profundas:  Baja resolucion x Muchos canales  -> "QUE objeto hay"
```

---

## 8. Las capas convolucionales NO clasifican

Las capas convolucionales extraen features. La clasificacion ocurre solo al final (FC + Softmax).

```mermaid
graph LR
    subgraph Conv["Capas Conv — entiende la imagen"]
        direction LR
        IN["224x224x3<br/>foto de un gato"]:::input --> CP["Conv + ReLU + Pool<br/>x5 bloques"]:::conv --> FM["7x7x512<br/>512 mapas de features"]:::conv
    end
    subgraph FC["Capas FC — decide que es"]
        direction LR
        FL["Flatten<br/>25,088"]:::flat --> F1["FC 4,096"]:::fc --> F2["FC 4,096"]:::fc --> F3["FC 1,000"]:::fc --> SM["Softmax<br/>85% gato"]:::output
    end

    FM --> FL

    classDef input fill:#1e40af,color:#fff,stroke:#1e3a8a
    classDef conv fill:#2563eb,color:#fff,stroke:#1e40af
    classDef flat fill:#6b7280,color:#fff,stroke:#4b5563
    classDef fc fill:#3b82f6,color:#fff,stroke:#2563eb
    classDef output fill:#059669,color:#fff,stroke:#047857
```

---

## 9. Las capas FC intermedias

### Por que no saltar de 25,088 a 1,000 directo

Las capas intermedias permiten aprender **combinaciones no-lineales** entre features:

```text
FC1: "Si hay pelaje Y ojos redondos Y NO ruedas -> probablemente animal"
FC2: "Si animal Y color dorado Y tamano grande -> perro dorado grande"
Final: neurona contribuye a clase "golden retriever"
```

### El 89% de los parametros de VGG estan en las FC

```text
Todas las conv (13 capas):  14.7M   (10.6%)
FC1: 25,088 x 4,096:      102.8M   (74.3%)
FC2: 4,096 x 4,096:        16.8M   (12.1%)
FC3: 4,096 x 1,000:         4.1M   (3.0%)
```

---

## 10. Global Average Pooling

{{< concept-alert type="clave" >}}
En lugar de aplanar (Flatten) el tensor 7x7x512 en un vector de 25,088, **Global Average Pooling** promedia cada canal en un solo numero, produciendo un vector de 512. Esto reduce las capas FC de ~102M parametros a ~0.5M parametros (~200x menos).
{{< /concept-alert >}}

### Comparacion

```mermaid
graph LR
    subgraph VGG["VGG (Flatten)"]
        direction LR
        V1["7x7x512"]:::conv --> V2["Flatten<br/>25,088"]:::flat --> V3["FC<br/>102M params"]:::fc
    end
    subgraph ResNet["ResNet (Avg Pool)"]
        direction LR
        R1["7x7x512"]:::conv --> R2["Global Avg Pool<br/>512"]:::pool --> R3["FC<br/>0.5M params"]:::fc2
    end

    classDef conv fill:#2563eb,color:#fff,stroke:#1e40af
    classDef flat fill:#6b7280,color:#fff,stroke:#4b5563
    classDef fc fill:#ef4444,color:#fff,stroke:#dc2626
    classDef pool fill:#f59e0b,color:#fff,stroke:#d97706
    classDef fc2 fill:#059669,color:#fff,stroke:#047857
```

### No se pierde informacion relevante

A esta altura de la red, importa **si** el patron esta presente, no **donde**:

```text
Canal #23 detecta "ojos de perro":
  Con ojos en la imagen:  promedio = 0.32 -> "hay ojos de perro"
  Sin ojos en la imagen:  promedio = 0.0  -> "no hay ojos de perro"
```

### Ejemplo: Global Average Pooling en practica

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torch.nn as nn

# Simular salida de ultima capa convolucional: batch=1, 512 canales, 7x7
feature_maps = torch.randn(1, 512, 7, 7)

# Global Average Pooling: promedia cada canal en un solo numero
gap = nn.AdaptiveAvgPool2d(1)
vector = gap(feature_maps).squeeze()  # [512]

# Clasificador final: solo 512 x num_clases parametros
clasificador = nn.Linear(512, 1000)
logits = clasificador(vector)

print(f"Feature maps: {feature_maps.shape}")  # [1, 512, 7, 7]
print(f"Tras GAP:     {vector.shape}")        # [512]
print(f"Params FC:    {512 * 1000 + 1000:,}") # 513,000 (vs 25M con flatten)
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

# Simular salida de ultima capa convolucional: batch=1, 7x7, 512 canales
feature_maps = tf.random.normal((1, 7, 7, 512))

# Global Average Pooling: promedia cada canal en un solo numero
gap = tf.keras.layers.GlobalAveragePooling2D()
vector = gap(feature_maps)  # [1, 512]

# Clasificador final: solo 512 x num_clases parametros
clasificador = tf.keras.layers.Dense(1000)
logits = clasificador(vector)

print(f"Feature maps: {feature_maps.shape}")  # (1, 7, 7, 512)
print(f"Tras GAP:     {vector.shape}")        # (1, 512)
print(f"Params FC:    {clasificador.count_params():,}")
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax.numpy as jnp
from flax import linen as nn
import jax

# Simular salida de ultima capa convolucional: batch=1, 7x7, 512 canales
feature_maps = jnp.ones((1, 7, 7, 512))

# Global Average Pooling: promedia sobre dimensiones espaciales (H, W)
vector = jnp.mean(feature_maps, axis=(1, 2))  # [1, 512]

# Clasificador final
clasificador = nn.Dense(1000)
params = clasificador.init(jax.random.PRNGKey(0), vector)
logits = clasificador.apply(params, vector)

print(f"Feature maps: {feature_maps.shape}")  # (1, 7, 7, 512)
print(f"Tras GAP:     {vector.shape}")        # (1, 512)
print(f"Logits:       {logits.shape}")        # (1, 1000)
```
{{< /tab >}}
{{< /tabs >}}

### Todas las arquitecturas modernas lo usan

```text
VGG (2014):      flatten -> FC 4,096 -> FC 4,096 -> FC 1,000  (122M params en FC)
GoogLeNet (2014): AvgPool -> FC 1,000                          (1M params en FC)
ResNet (2015):    AvgPool -> FC 1,000                          (2M params en FC)
```

---

## 11. Flujo completo de VGG-16

```mermaid
graph TD
    IN["Entrada: 224x224x3 RGB"]:::input
    B1["Bloque 1 — Bordes y colores<br/>conv3-64 x2 + ReLU + MaxPool<br/>112x112x64"]:::b1
    B2["Bloque 2 — Texturas simples<br/>conv3-128 x2 + ReLU + MaxPool<br/>56x56x128"]:::b2
    B3["Bloque 3 — Texturas complejas<br/>conv3-256 x3 + MaxPool<br/>28x28x256"]:::b3
    B4["Bloque 4 — Partes de objetos<br/>conv3-512 x3 + MaxPool<br/>14x14x512"]:::b4
    B5["Bloque 5 — Objetos completos<br/>conv3-512 x3 + MaxPool<br/>7x7x512"]:::b5
    CL["Flatten 25,088 — FC 4,096 — FC 4,096 — FC 1,000 — Softmax"]:::output

    IN --> B1 --> B2 --> B3 --> B4 --> B5 --> CL

    classDef input fill:#1e40af,color:#fff,stroke:#1e3a8a
    classDef b1 fill:#2563eb,color:#fff,stroke:#1e40af
    classDef b2 fill:#3b82f6,color:#fff,stroke:#2563eb
    classDef b3 fill:#60a5fa,color:#fff,stroke:#3b82f6
    classDef b4 fill:#3b82f6,color:#fff,stroke:#2563eb
    classDef b5 fill:#2563eb,color:#fff,stroke:#1e40af
    classDef output fill:#059669,color:#fff,stroke:#047857
```
