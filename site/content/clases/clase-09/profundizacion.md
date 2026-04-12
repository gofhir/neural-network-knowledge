---
title: "Profundizacion - Analisis de Arquitecturas"
weight: 20
math: true
---

## 1. El Problema que Motiva Todo

Toda arquitectura de red neuronal profunda intenta responder:

> Como construir una funcion lo suficientemente expresiva para representar conceptos complejos, sin que sea imposible de entrenar ni de usar en produccion?

| Tension | Lado A | Lado B |
|---------|--------|--------|
| Capacidad vs. eficiencia | Mas parametros = mas expresividad | Mas parametros = mas computo |
| Profundidad vs. entrenabilidad | Mas capas = mas abstraccion | Mas capas = vanishing gradient |
| Generalizacion vs. memorizacion | Buena accuracy en train | Mala accuracy en test |

### El benchmark: ImageNet (ILSVRC)

| Ano | Ganador | Top-5 Error | Innovacion clave |
|-----|---------|-------------|-----------------|
| 2010 | Hand-crafted | ~28% | -- |
| 2012 | **AlexNet** | **16.4%** | ReLU, Dropout, GPU |
| 2014 | **GoogLeNet** | **6.7%** | Inception modules |
| 2014 | **VGG** | **7.3%** | Profundidad + 3x3 |
| 2015 | **ResNet** | **3.57%** | Skip connections |
| (humano) | -- | ~5% | -- |

En 2015, ResNet supero el rendimiento humano en Top-5 error.

---

## 2. Campo Receptivo: Profundo

### Definicion precisa

El campo receptivo de una neurona en capa $n$ es el conjunto de pixeles de la imagen de entrada que influyen en ella. Para convoluciones con filtro $k$ y stride $s=1$:

$$RF_n = RF_{n-1} + (k-1)$$

Para filtros 3x3: $RF_n = (2n+1) \times (2n+1)$

### Tres estrategias para aumentar el campo receptivo

{{< concept-alert type="clave" >}}
Todas las innovaciones en arquitecturas CNN son, en el fondo, estrategias para **aumentar el campo receptivo de forma eficiente** mientras se mantiene el entrenamiento estable.
{{< /concept-alert >}}

1. **Filtros grandes (AlexNet):** 11x11 = 121 parametros. Costoso.
2. **Apilar filtros pequenos (VGG):** 2 capas de 3x3 = campo 5x5 con menos parametros + no-linealidad extra.
3. **Filtros en paralelo (Inception):** La red aprende que escala es relevante.
4. **Skip connections + profundidad (ResNet):** Profundidad arbitraria sin degradacion.

### Ejemplo: Calcular campo receptivo

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
def calcular_campo_receptivo(capas):
    """Calcula el campo receptivo acumulado capa a capa.
    Cada capa es una tupla (kernel_size, stride).
    """
    rf = 1       # campo receptivo inicial (1 pixel)
    stride_acum = 1  # stride acumulado
    for k, s in capas:
        rf = rf + (k - 1) * stride_acum  # expandir campo receptivo
        stride_acum *= s
    return rf

# Ejemplo: primeras capas de VGG-16 (conv3x3 + conv3x3 + maxpool)
capas_vgg = [(3, 1), (3, 1), (2, 2),   # Bloque 1: RF = 6x6
             (3, 1), (3, 1), (2, 2)]    # Bloque 2: RF = 16x16
print(f"Campo receptivo tras bloque 2: {calcular_campo_receptivo(capas_vgg)}x{calcular_campo_receptivo(capas_vgg)}")
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
def calcular_campo_receptivo(capas):
    """Calcula el campo receptivo acumulado capa a capa.
    Cada capa es una tupla (kernel_size, stride).
    """
    rf = 1       # campo receptivo inicial (1 pixel)
    stride_acum = 1  # stride acumulado
    for k, s in capas:
        rf = rf + (k - 1) * stride_acum
        stride_acum *= s
    return rf

# Ejemplo: primeras capas de ResNet (conv7x7 stride 2 + maxpool 3x3 stride 2)
capas_resnet = [(7, 2), (3, 2), (3, 1), (3, 1)]
print(f"Campo receptivo tras primer bloque residual: {calcular_campo_receptivo(capas_resnet)}")
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax.numpy as jnp

def calcular_campo_receptivo(capas):
    """Calcula el campo receptivo acumulado capa a capa.
    Cada capa es una tupla (kernel_size, stride).
    """
    rf = 1
    stride_acum = 1
    for k, s in capas:
        rf = rf + (k - 1) * stride_acum
        stride_acum *= s
    return rf

# Comparar campo receptivo de distintas estrategias
una_capa_7x7 = [(7, 1)]             # RF = 7
tres_capas_3x3 = [(3, 1)] * 3       # RF = 7 (mismo RF, menos parametros)
print(f"1 capa 7x7: RF={calcular_campo_receptivo(una_capa_7x7)}")
print(f"3 capas 3x3: RF={calcular_campo_receptivo(tres_capas_3x3)}")
```
{{< /tab >}}
{{< /tabs >}}

---

## 3. VGG en Profundidad

### La decision de usar solo filtros 3x3

**Argumento 1:** El filtro 3x3 cubre las 8 posiciones vecinas de cada pixel mas el centro. Es el mas pequeno con riqueza espacial completa.

**Argumento 2 (eficiencia de parametros):**

Para campo receptivo 5x5 con $C$ canales de entrada:

| Opcion | Parametros |
|--------|-----------|
| 1 capa 5x5 | $25C^2$ |
| 2 capas 3x3 | $18C^2$ (28% menos) |

Para campo receptivo 7x7:

| Opcion | Parametros |
|--------|-----------|
| 1 capa 7x7 | $49C^2$ |
| 3 capas 3x3 | $27C^2$ (45% menos) |

### Configuraciones evaluadas

| Config | Capas | Parametros |
|--------|-------|-----------|
| A | 11 | 133M |
| D (VGG-16) | 16 | **138M** |
| E (VGG-19) | 19 | **144M** |

---

## 4. Inception en Profundidad

### Por que usar filtros 1x1

Un filtro 1x1 actua como una **proyeccion lineal** sobre el eje de canales. Es analogo a un embedding: comprime la informacion sin perder informacion espacial.

### Ahorro de parametros

Primer bloque Inception (filtros 5x5, input 192 canales, output 32):

| Caso | Parametros |
|------|-----------|
| Sin 1x1 | 153,600 |
| Con 1x1 (16 intermedios) | 15,872 (~90% menos) |

### Ejemplo: Convolucion 1x1 como proyeccion

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torch.nn as nn

# Simular entrada: batch=1, 192 canales, 28x28 espacial
x = torch.randn(1, 192, 28, 28)

# Convolucion 1x1: proyecta de 192 a 16 canales (reduccion ~12x)
proyeccion = nn.Conv2d(192, 16, kernel_size=1)
x_reducido = proyeccion(x)

print(f"Entrada:  {x.shape}")          # [1, 192, 28, 28]
print(f"Salida:   {x_reducido.shape}")  # [1, 16, 28, 28] — misma resolucion, menos canales
print(f"Params 1x1: {192 * 16 + 16:,}")  # 3,088 parametros (pesos + bias)
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

# Simular entrada: batch=1, 28x28, 192 canales
x = tf.random.normal((1, 28, 28, 192))

# Convolucion 1x1: proyecta de 192 a 16 canales (reduccion ~12x)
proyeccion = tf.keras.layers.Conv2D(16, kernel_size=1)
x_reducido = proyeccion(x)

print(f"Entrada:  {x.shape}")          # (1, 28, 28, 192)
print(f"Salida:   {x_reducido.shape}")  # (1, 28, 28, 16) — misma resolucion, menos canales
print(f"Params 1x1: {proyeccion.count_params():,}")
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax
import jax.numpy as jnp
from flax import linen as nn

# Convolucion 1x1: proyeccion de canales sin cambiar resolucion espacial
proyeccion = nn.Conv(features=16, kernel_size=(1, 1))
x = jnp.ones((1, 28, 28, 192))  # entrada: 192 canales
params = proyeccion.init(jax.random.PRNGKey(0), x)
x_reducido = proyeccion.apply(params, x)

print(f"Entrada:  {x.shape}")          # (1, 28, 28, 192)
print(f"Salida:   {x_reducido.shape}")  # (1, 28, 28, 16) — misma resolucion, menos canales
```
{{< /tab >}}
{{< /tabs >}}

### Clasificadores auxiliares y vanishing gradient

Las redes profundas sufren vanishing gradient. Inception inyecta gradiente en capas intermedias con clasificadores auxiliares. En inferencia solo se usa el final.

---

## 5. ResNet en Profundidad

### El experimento clave

Red plain de 56 capas tiene mayor error que la de 20 capas, tanto en train como en test. No es overfitting: es un problema de optimizacion.

### La hipotesis

> No todos los sistemas son igualmente faciles de optimizar. Es mas dificil aprender la funcion identidad directamente que aprender una perturbacion de cero.

### Bloque residual formal

Sea $H(x)$ el mapeo deseado. La red aprende:

$$F(x) := H(x) - x$$

Por lo tanto: $H(x) = F(x) + x$

Si la identidad es optima, basta con llevar los pesos de $F(x)$ a cero.

### Bottleneck (ResNet-50+)

Usa 3 capas (1x1 reduce, 3x3 convolucion, 1x1 restaura) para reducir costo computacional.

| Version | FLOPs |
|---------|-------|
| ResNet-18 | 1.8 x 10^9 |
| ResNet-50 | 3.8 x 10^9 |
| ResNet-152 | 11.3 x 10^9 |

### Ejemplo: Bottleneck Block

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch.nn as nn

class Bottleneck(nn.Module):
    """Bloque bottleneck usado en ResNet-50/101/152"""
    expansion = 4  # el canal de salida es 4x el cuello de botella

    def __init__(self, in_channels, bottleneck_channels):
        super().__init__()
        out_channels = bottleneck_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)  # reducir
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)  # conv espacial
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)  # restaurar
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Skip connection con proyeccion si cambian dimensiones
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))   # 1x1 reduce
        out = self.relu(self.bn2(self.conv2(out)))  # 3x3 espacial
        out = self.bn3(self.conv3(out))             # 1x1 restaura
        return self.relu(out + self.shortcut(x))    # skip connection
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf
from tensorflow.keras import layers

class Bottleneck(layers.Layer):
    """Bloque bottleneck usado en ResNet-50/101/152"""
    expansion = 4

    def __init__(self, bottleneck_channels):
        super().__init__()
        out_ch = bottleneck_channels * self.expansion
        self.conv1 = layers.Conv2D(bottleneck_channels, 1, use_bias=False)   # reducir
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(bottleneck_channels, 3, padding="same", use_bias=False)  # conv espacial
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_ch, 1, use_bias=False)               # restaurar
        self.bn3 = layers.BatchNormalization()
        self.proj = layers.Conv2D(out_ch, 1, use_bias=False)  # proyeccion skip

    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))   # 1x1 reduce
        out = tf.nn.relu(self.bn2(self.conv2(out)))  # 3x3 espacial
        out = self.bn3(self.conv3(out))              # 1x1 restaura
        return tf.nn.relu(out + self.proj(x))        # skip connection
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
from flax import linen as nn

class Bottleneck(nn.Module):
    """Bloque bottleneck usado en ResNet-50/101/152"""
    bottleneck_channels: int
    expansion: int = 4

    @nn.compact
    def __call__(self, x, train: bool = True):
        out_ch = self.bottleneck_channels * self.expansion
        # 1x1 reduce dimensionalidad
        out = nn.Conv(self.bottleneck_channels, (1, 1), use_bias=False)(x)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)
        # 3x3 convolucion espacial
        out = nn.Conv(self.bottleneck_channels, (3, 3), padding="SAME", use_bias=False)(out)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)
        # 1x1 restaura dimensionalidad
        out = nn.Conv(out_ch, (1, 1), use_bias=False)(out)
        out = nn.BatchNorm(use_running_average=not train)(out)
        # Proyeccion skip si es necesario
        shortcut = nn.Conv(out_ch, (1, 1), use_bias=False)(x)
        return nn.relu(out + shortcut)
```
{{< /tab >}}
{{< /tabs >}}

---

## 6. Guia de Decision: Que Arquitectura Usar

| Criterio | Recomendacion |
|----------|--------------|
| Baseline simple | ResNet-18/34 |
| Maxima accuracy | ResNet-152 o Inception-v4 |
| Dispositivos moviles | MobileNet |
| Pocas operaciones | GoogLeNet |
| Transfer learning | ResNet-50 (preentrenado) |

---

## 7. Interpretabilidad: Analisis Profundo

### Feature Visualization: gradient ascent en el input

En lugar de actualizar pesos, actualizamos la imagen para maximizar activaciones. Problemas: imagenes ruidosas y patrones de tablero.

**Soluciones de regularizacion:**
- L1 regularization y Total Variation (penalizar ruido)
- Robustez a transformaciones (jitter, rotate, scale)
- Espacio decorrelado (optimizar en frecuencias de Fourier)

### Attribution: causalidad en redes neuronales

| Metodo | Tipo | Descripcion |
|--------|------|-------------|
| Gradient (Vanilla) | Backprop | $\partial\text{output}/\partial\text{input}$ directamente |
| Guided Backprop | Backprop | Solo propaga gradientes positivos por ReLUs |
| Occlusion | Perturbacion | Desliza parche negro y mide caida |
| Extremal Perturbation | Perturbacion | Aprende mascara optima por optimizacion |

{{< concept-alert type="clave" >}}
La interpretabilidad es critica para detectar **sesgos y shortcuts daninos**. Ejemplo real: una red clasificaba "caballo" basandose en el watermark de copyright, no en el animal. Attribution lo revelo claramente.
{{< /concept-alert >}}

### Filter-Concept Overlap

La relacion entre filtros y conceptos no es 1:1:
- Un filtro puede responder a multiples conceptos (polisemia)
- Un concepto puede activar multiples filtros
