---
title: "Pooling y Dimensiones"
weight: 30
math: true
---

## 1. Max Pooling

**Max Pooling** es la operacion de reduccion dimensional mas comun en CNNs. Toma una ventana de tamano fijo y selecciona el valor maximo dentro de ella, descartando el resto.

### Mecanica

Dado un feature map de entrada, la ventana de pooling se desliza con un stride determinado y en cada posicion retiene solo el maximo:

$$\text{MaxPool}(i, j) = \max_{m,n \in \text{ventana}} \text{entrada}(i + m, j + n)$$

### Ejemplo: MaxPool $3 \times 3$ con stride 2

Sobre un feature map de $27 \times 27$:

$$O = \left\lfloor\frac{27 - 3}{2}\right\rfloor + 1 = 13$$

La salida es $13 \times 13$. La dimension se reduce a aproximadamente la mitad, pero la profundidad (numero de canales) no cambia.

### Formula general

La formula de dimension de salida del pooling es identica a la de convolucion:

$$O = \left\lfloor\frac{W - K}{S}\right\rfloor + 1$$

Donde $K$ es el tamano de la ventana de pooling y $S$ el stride. En pooling no se suele usar padding.

{{< concept-alert type="clave" >}}
Max Pooling **no tiene parametros entrenables**. Es una operacion fija que reduce las dimensiones espaciales del feature map manteniendo las caracteristicas mas prominentes (los maximos). Esto introduce invarianza local a traslaciones y reduce el costo computacional de las capas siguientes.
{{< /concept-alert >}}

### Implementacion en PyTorch

```python
import torch.nn as nn

pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

import torch
x = torch.randn(1, 96, 55, 55)
y = pool(x)
print(y.shape)  # torch.Size([1, 96, 27, 27])
```

---

## 2. Average Pooling

**Average Pooling** calcula el promedio de los valores dentro de la ventana en lugar del maximo:

$$\text{AvgPool}(i, j) = \frac{1}{K^2} \sum_{m,n \in \text{ventana}} \text{entrada}(i + m, j + n)$$

Se usa menos que Max Pooling en las capas intermedias, pero es comun como ultima operacion de reduccion antes de las capas fully connected en arquitecturas modernas (como ResNet, que usa `nn.AdaptiveAvgPool2d(1)` para reducir cualquier tamano espacial a $1 \times 1$).

```python
# Average Pooling clasico
avg_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))

# Adaptive Average Pooling (la salida siempre es 1x1)
adaptive_pool = nn.AdaptiveAvgPool2d(1)
x = torch.randn(1, 256, 6, 6)
print(adaptive_pool(x).shape)  # torch.Size([1, 256, 1, 1])
```

---

## 3. Flattening

Despues de las capas convolucionales y de pooling, el tensor tiene forma $C \times H \times W$. Para conectar con las capas fully connected, se necesita aplanar este tensor a un vector 1D:

$$\text{dim\_flat} = C \times H \times W$$

En AlexNet, despues de conv5 la salida es $256 \times 6 \times 6$:

$$\text{dim\_flat} = 256 \times 6 \times 6 = 9{,}216$$

Este valor es el `in_features` de la primera capa Linear (fc6).

```python
self.flat = nn.Flatten()
# transforma tensor de [batch, 256, 6, 6] a [batch, 9216]
```

---

## 4. Flujo completo de dimensiones en AlexNet

El siguiente recorrido muestra como la entrada $3 \times 224 \times 224$ se transforma capa por capa hasta la salida de 1000 clases.

### Fase convolucional

| Etapa | Operacion | Calculo de dimension | Salida |
|-------|-----------|---------------------|--------|
| Entrada | — | — | $3 \times 224 \times 224$ |
| conv1 - Conv | Conv2d(3, 96, 11, s=4, p=2) | $\lfloor(224-11+4)/4\rfloor+1=55$ | $96 \times 55 \times 55$ |
| conv1 - Pool | MaxPool(3, s=2) | $\lfloor(55-3)/2\rfloor+1=27$ | $96 \times 27 \times 27$ |
| conv2 - Conv | Conv2d(96, 256, 5, s=1, p=2) | $\lfloor(27-5+4)/1\rfloor+1=27$ | $256 \times 27 \times 27$ |
| conv2 - Pool | MaxPool(3, s=2) | $\lfloor(27-3)/2\rfloor+1=13$ | $256 \times 13 \times 13$ |
| conv3 | Conv2d(256, 384, 3, s=1, p=1) | $\lfloor(13-3+2)/1\rfloor+1=13$ | $384 \times 13 \times 13$ |
| conv4 | Conv2d(384, 384, 3, s=1, p=1) | $\lfloor(13-3+2)/1\rfloor+1=13$ | $384 \times 13 \times 13$ |
| conv5 - Conv | Conv2d(384, 256, 3, s=1, p=1) | $\lfloor(13-3+2)/1\rfloor+1=13$ | $256 \times 13 \times 13$ |
| conv5 - Pool | MaxPool(3, s=2) | $\lfloor(13-3)/2\rfloor+1=6$ | $256 \times 6 \times 6$ |

### Fase fully connected

| Etapa | Operacion | Salida |
|-------|-----------|--------|
| Flatten | $256 \times 6 \times 6$ | $9{,}216$ |
| fc6 | Linear(9216, 4096) + ReLU | $4{,}096$ |
| fc7 | Linear(4096, 4096) + ReLU | $4{,}096$ |
| fc8 | Linear(4096, 1000) | $1{,}000$ |

{{< concept-alert type="clave" >}}
La fase convolucional reduce progresivamente las dimensiones espaciales ($224 \to 55 \to 27 \to 13 \to 6$) mientras aumenta la profundidad ($3 \to 96 \to 256 \to 384 \to 256$). La fase fully connected toma el vector aplanado de $9{,}216$ dimensiones y lo comprime hasta las $1{,}000$ clases de salida.
{{< /concept-alert >}}

---

## 5. Verificacion con PyTorch

Una forma practica de verificar que las dimensiones son correctas es propagar un tensor aleatorio por la red:

```python
import torch

modelo = AlexNet()
x = torch.randn(1, 3, 224, 224)

# Verificar capa por capa
x1 = modelo.conv1(x);  print(f"conv1: {x1.shape}")  # [1, 96, 27, 27]
x2 = modelo.conv2(x1);  print(f"conv2: {x2.shape}")  # [1, 256, 13, 13]
x3 = modelo.conv3(x2);  print(f"conv3: {x3.shape}")  # [1, 384, 13, 13]
x4 = modelo.conv4(x3);  print(f"conv4: {x4.shape}")  # [1, 384, 13, 13]
x5 = modelo.conv5(x4);  print(f"conv5: {x5.shape}")  # [1, 256, 6, 6]
xf = modelo.flat(x5);   print(f"flat:  {xf.shape}")  # [1, 9216]
```

Si alguna capa esta mal definida, PyTorch lanzara un error de dimensiones incompatibles al intentar la propagacion. Esta tecnica es la forma mas rapida de depurar una arquitectura.
