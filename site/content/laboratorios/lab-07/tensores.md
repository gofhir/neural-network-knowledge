---
title: "Tensores en PyTorch"
weight: 10
math: true
---

## 1. Que es un Tensor

Un tensor es una **matriz N-dimensional** que constituye la estructura de datos fundamental de PyTorch. Asi como NumPy trabaja con `ndarray`, PyTorch trabaja con `torch.Tensor`. La diferencia clave es que los tensores de PyTorch pueden ejecutarse en GPU y soportan diferenciacion automatica (autograd), lo que los hace esenciales para entrenar redes neuronales.

Segun el numero de dimensiones, un tensor recibe distintos nombres:

| Dimensiones | Nombre | Ejemplo |
|---|---|---|
| 0 | Escalar | Un numero: `3.14` |
| 1 | Vector | Una lista: `[1, 2, 3]` |
| 2 | Matriz | Una tabla de datos |
| 3 | Tensor 3D | Una imagen RGB (canales, alto, ancho) |
| 4 | Tensor 4D | Un batch de imagenes |

{{< concept-alert type="clave" >}}
Toda la informacion que fluye a traves de una red neuronal --- imagenes, texto, audio, embeddings, predicciones --- se representa como tensores. Dominar su creacion y manipulacion es prerequisito para cualquier trabajo en PyTorch.
{{< /concept-alert >}}

---

## 2. Creacion de Tensores

### Desde datos existentes

```python
import torch

# Desde una lista de Python
t = torch.tensor([1, 2, 3, 4])
print(t)        # tensor([1, 2, 3, 4])
print(t.shape)  # torch.Size([4])

# Desde una lista 2D
m = torch.tensor([[1, 2], [3, 4]])
print(m.shape)  # torch.Size([2, 2])

# Desde un array de NumPy
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(arr)
```

### Funciones de creacion

```python
# Tensor de ceros
zeros = torch.zeros(3, 4)       # Matriz 3x4 de ceros

# Tensor de unos
ones = torch.ones(2, 3, 4)     # Tensor 3D de unos

# Tensor con valores aleatorios (distribucion normal)
randn = torch.randn(3, 224, 224)  # Simula una imagen RGB 224x224

# Tensor con valores aleatorios (distribucion uniforme [0, 1))
rand = torch.rand(5, 5)

# Tensor identidad
eye = torch.eye(4)              # Matriz identidad 4x4

# Rango de valores
arange = torch.arange(0, 10, 2)  # tensor([0, 2, 4, 6, 8])

# Tensor lleno de un valor especifico
full = torch.full((3, 3), 7.0)   # Matriz 3x3 llena de 7.0
```

---

## 3. Operaciones con Tensores

### Aritmetica basica

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Suma y resta (element-wise)
suma = a + b          # tensor([5., 7., 9.])
resta = a - b         # tensor([-3., -3., -3.])

# Multiplicacion por escalar
mult = 3 * a          # tensor([3., 6., 9.])

# Multiplicacion element-wise
elem = a * b          # tensor([4., 10., 18.])

# Division
div = b / a           # tensor([4., 2.5, 2.])
```

### Multiplicacion matricial

La multiplicacion de matrices es la operacion central de las redes neuronales. Cada capa `Linear` internamente realiza $y = xW^T + b$.

```python
# Matrices 2x3 y 3x4
A = torch.randn(2, 3)
B = torch.randn(3, 4)

# Tres formas equivalentes de multiplicar matrices
C1 = torch.mm(A, B)     # Funcion dedicada
C2 = A @ B              # Operador @ (recomendado)
C3 = torch.matmul(A, B) # Funcion general (soporta broadcasting)

print(C1.shape)  # torch.Size([2, 4])
```

{{< concept-alert type="clave" >}}
`torch.mm` solo acepta matrices 2D. `torch.matmul` y el operador `@` soportan tensores de cualquier dimension con broadcasting automatico, lo que los hace mas versatiles para trabajar con batches.
{{< /concept-alert >}}

### Reducciones

```python
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

t.sum()          # tensor(10.) - suma total
t.sum(dim=0)     # tensor([4., 6.]) - suma por columna
t.sum(dim=1)     # tensor([3., 7.]) - suma por fila
t.mean()         # tensor(2.5)
t.max()          # tensor(4.)
t.argmax()       # tensor(3) - indice del maximo (aplanado)
t.argmax(dim=1)  # tensor([1, 1]) - indice del maximo por fila
```

---

## 4. Indexacion y Slicing

Los tensores se indexan de forma similar a NumPy:

```python
t = torch.randn(4, 5)

# Elemento individual
t[0, 0]        # Primer elemento

# Fila completa
t[0]           # Primera fila (shape: [5])

# Columna completa
t[:, 0]        # Primera columna (shape: [4])

# Sub-tensor
t[1:3, 2:4]    # Filas 1-2, columnas 2-3

# Indexacion booleana
mask = t > 0
positivos = t[mask]  # Todos los valores positivos
```

---

## 5. Manipulacion de Forma

Cambiar la forma de un tensor sin alterar sus datos es una operacion frecuente:

```python
t = torch.randn(2, 3, 4)
print(t.shape)  # torch.Size([2, 3, 4])

# view: cambia la forma (requiere memoria contigua)
t_flat = t.view(2, 12)       # Aplana las ultimas dos dimensiones
t_flat2 = t.view(-1)         # Aplana completamente: shape [24]

# reshape: similar a view pero no requiere memoria contigua
t_reshaped = t.reshape(6, 4)

# squeeze: elimina dimensiones de tamanio 1
t_extra = torch.randn(1, 3, 1, 4)
print(t_extra.shape)          # torch.Size([1, 3, 1, 4])
print(t_extra.squeeze().shape) # torch.Size([3, 4])

# unsqueeze: agrega una dimension de tamanio 1
t_2d = torch.randn(3, 4)
t_3d = t_2d.unsqueeze(0)     # Agrega dimension de batch
print(t_3d.shape)             # torch.Size([1, 3, 4])

# permute: reordena las dimensiones
img = torch.randn(224, 224, 3)    # H x W x C (formato PIL)
img_pytorch = img.permute(2, 0, 1) # C x H x W (formato PyTorch)
print(img_pytorch.shape)           # torch.Size([3, 224, 224])

# transpose: intercambia dos dimensiones
t_T = t_2d.transpose(0, 1)   # shape [4, 3]
```

---

## 6. Gestion de Dispositivo: CPU vs GPU

Los tensores viven en un dispositivo especifico. Todos los tensores involucrados en una operacion deben estar en el **mismo dispositivo**, de lo contrario PyTorch lanzara un error.

```python
# Verificar disponibilidad de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Crear tensor directamente en GPU
t_gpu = torch.randn(3, 3, device=device)

# Mover tensor existente a GPU
t_cpu = torch.randn(3, 3)
t_gpu = t_cpu.to(device)      # Forma general (recomendada)
t_gpu = t_cpu.cuda()          # Equivalente si hay GPU

# Mover de vuelta a CPU
t_cpu = t_gpu.cpu()

# Verificar dispositivo
print(t_gpu.device)           # cuda:0
print(t_cpu.device)           # cpu
```

{{< concept-alert type="clave" >}}
`.to(device)` crea una **copia** del tensor en el dispositivo destino. El tensor original permanece donde estaba. Esto es diferente a operaciones in-place como `.cuda_()` (con underscore). La forma recomendada es siempre usar `.to(device)`.
{{< /concept-alert >}}

---

## 7. Convencion de Dimensiones por Tipo de Dato

PyTorch espera convenciones especificas para cada tipo de dato. Respetar estas convenciones es obligatorio para que las capas funcionen correctamente.

### Imagenes

El formato estandar para imagenes en PyTorch es **(batch, canales, alto, ancho)** o **NCHW**:

```python
# Un batch de 32 imagenes RGB de 224x224
batch_imagenes = torch.randn(32, 3, 224, 224)
#                             N   C   H    W

# Una sola imagen necesita dimension de batch
imagen = torch.randn(1, 3, 224, 224)
```

### Texto

Para secuencias de texto, el formato tipico es **(batch, longitud_secuencia, dimension_embedding)**:

```python
# Un batch de 16 oraciones, cada una de 50 tokens, embedding de 300
batch_texto = torch.randn(16, 50, 300)
```

### Videos

Para video se agrega la dimension temporal: **(batch, canales, frames, alto, ancho)**:

```python
# Un batch de 8 videos, 3 canales, 16 frames, 112x112
batch_video = torch.randn(8, 3, 16, 112, 112)
```

{{< concept-alert type="recordar" >}}
La dimension de **batch siempre es la primera** en PyTorch. Incluso si se evalua un solo elemento, debe tener la dimension de batch: `imagen.unsqueeze(0)` convierte un tensor de shape `[3, 224, 224]` a `[1, 3, 224, 224]`.
{{< /concept-alert >}}

---

## 8. Tipos de Dato

```python
# PyTorch infiere el tipo automaticamente
t_float = torch.tensor([1.0, 2.0])    # float32 (default)
t_int = torch.tensor([1, 2])          # int64 (default)

# Especificar tipo explicitamente
t = torch.tensor([1, 2], dtype=torch.float32)

# Convertir tipo
t_f16 = t.half()      # float16 (para eficiencia en GPU)
t_f64 = t.double()    # float64
t_i32 = t.int()       # int32

# Verificar tipo
print(t.dtype)         # torch.float32
```

Los pesos de las redes neuronales y las funciones de perdida trabajan por defecto con `float32`. Si los datos de entrada son de otro tipo (por ejemplo `uint8` para imagenes), es necesario convertirlos antes de pasarlos al modelo.
