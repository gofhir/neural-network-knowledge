---
title: "Modulos y Capas"
weight: 20
math: true
---

## 1. nn.Module: la Base de Todo Modelo

En PyTorch, toda red neuronal es una clase que hereda de `torch.nn.Module`. Este es el bloque de construccion fundamental: cada capa, cada funcion de activacion y cada modelo completo es un `nn.Module`.

Para crear un modelo valido se deben implementar dos metodos:

- **`__init__(self)`** --- el constructor, donde se definen todas las capas y componentes de la arquitectura.
- **`forward(self, x)`** --- define como fluyen los datos a traves de la red, conectando las capas en el orden deseado.

```python
import torch
import torch.nn as nn

class MiModelo(nn.Module):
    def __init__(self):
        super(MiModelo, self).__init__()
        self.fc1 = nn.Linear(100, 100)

    def forward(self, x):
        x = self.fc1(x)
        return x
```

{{< concept-alert type="clave" >}}
PyTorch **no** calcula la propagacion hacia atras manualmente. Al definir `forward()`, el mecanismo de autograd construye automaticamente el grafo computacional y `backward()` calcula los gradientes. Solo es necesario definir el paso forward.
{{< /concept-alert >}}

---

## 2. Capas Comunes

### nn.Linear (Capa Densa / Fully Connected)

Aplica una transformacion lineal $y = xW^T + b$. Es la capa basica de los MLP (Multilayer Perceptrons).

```python
# Capa con 784 entradas y 256 salidas
fc = nn.Linear(784, 256)

# Los parametros se inicializan automaticamente
print(fc.weight.shape)  # torch.Size([256, 784])
print(fc.bias.shape)    # torch.Size([256])

# Uso
x = torch.randn(32, 784)  # Batch de 32, 784 features
y = fc(x)                  # shape: [32, 256]
```

### nn.Conv2d (Convolucion 2D)

Aplica filtros convolucionales sobre imagenes. Cada filtro detecta un patron local (borde, textura, forma).

```python
# 3 canales de entrada (RGB), 64 filtros de 3x3
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

x = torch.randn(32, 3, 224, 224)  # Batch de 32 imagenes RGB
y = conv(x)                        # shape: [32, 64, 224, 224]
```

### Funciones de Activacion

Las activaciones introducen no-linealidad, sin la cual la red seria equivalente a una sola transformacion lineal sin importar cuantas capas tenga.

```python
relu = nn.ReLU()           # max(0, x) - la mas comun
sigmoid = nn.Sigmoid()     # Rango [0, 1]
softmax = nn.Softmax(dim=1) # Probabilidades que suman 1
tanh = nn.Tanh()           # Rango [-1, 1]
```

### nn.BatchNorm2d (Normalizacion por Batch)

Normaliza las activaciones de cada canal usando la media y varianza del batch. Estabiliza y acelera el entrenamiento.

```python
# BatchNorm para 64 canales (debe coincidir con la salida de la capa anterior)
bn = nn.BatchNorm2d(64)

x = torch.randn(32, 64, 56, 56)
y = bn(x)  # shape: [32, 64, 56, 56] - misma forma, valores normalizados
```

### nn.Dropout

Desactiva neuronas aleatoriamente durante el entrenamiento para reducir overfitting. Durante evaluacion (`model.eval()`), todas las neuronas estan activas.

```python
dropout = nn.Dropout(p=0.5)  # 50% de probabilidad de desactivar cada neurona

x = torch.randn(32, 256)
y = dropout(x)  # Durante training: algunos valores son 0
                 # Durante eval: todos los valores pasan (escalados)
```

### nn.MaxPool2d (Pooling)

Reduce la resolucion espacial tomando el maximo en ventanas locales.

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.randn(32, 64, 224, 224)
y = pool(x)  # shape: [32, 64, 112, 112] - resolucion reducida a la mitad
```

---

## 3. Composicion con nn.Sequential

Para arquitecturas simples donde las capas se ejecutan una tras otra, `nn.Sequential` evita tener que escribir `forward()` manualmente:

```python
# Forma clasica (explicita)
class ModeloExplicito(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Forma con Sequential (compacta)
modelo_seq = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    nn.Softmax(dim=1)
)
```

Ambas formas son equivalentes funcionalmente. La diferencia es que `nn.Sequential` no requiere definir una clase ni escribir `forward()`, pero pierde flexibilidad para arquitecturas con ramas paralelas o conexiones residuales.

---

## 4. Modelos Custom con Arquitectura Compleja

Cuando la arquitectura no es secuencial (por ejemplo, conexiones residuales o ramas paralelas), se necesita una clase custom:

```python
class ModeloConResiduales(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        residual = x                # Guardar entrada
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = x + residual            # Conexion residual (skip connection)
        x = self.relu(x)
        return x
```

---

## 5. Ejemplo Completo: MiAlexNet

El laboratorio implementa una version de AlexNet adaptada al dataset Flowers (102 clases):

```python
class MiAlexNet(nn.Module):
    def __init__(self, num_classes=102):
        super(MiAlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv1: 3 -> 64 canales, kernel 11x11
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2: 64 -> 192 canales, kernel 5x5
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3: 192 -> 384 canales, kernel 3x3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4: 384 -> 256 canales
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5: 256 -> 256 canales
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),   # FC6
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),           # FC7
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),    # FC8 -> 102 clases
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch, 256*6*6]
        x = self.classifier(x)
        return x
```

{{< concept-alert type="clave" >}}
La linea `x.view(x.size(0), -1)` aplana el tensor de features de shape `[batch, 256, 6, 6]` a `[batch, 9216]` para poder alimentar las capas fully connected. El `-1` le indica a PyTorch que calcule automaticamente esa dimension.
{{< /concept-alert >}}

---

## 6. Inspeccion de Parametros

```python
model = MiAlexNet()

# Ver todos los parametros con sus nombres
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Contar parametros totales
total = sum(p.numel() for p in model.parameters())
print(f"Parametros totales: {total:,}")

# Contar solo parametros entrenables
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametros entrenables: {trainable:,}")
```

### Congelar parametros

Para fine-tuning, es comun congelar las capas preentrenadas y solo entrenar las nuevas:

```python
# Congelar todos los parametros
for param in model.parameters():
    param.requires_grad = False

# Descongelar solo el clasificador
for param in model.classifier.parameters():
    param.requires_grad = True
```

---

## 7. Mover Modelo a GPU

Al igual que los tensores, el modelo completo debe moverse al dispositivo donde se ejecutara:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MiAlexNet()
model = model.to(device)  # Mueve todos los parametros y buffers a GPU

# Los datos tambien deben estar en el mismo dispositivo
x = torch.randn(32, 3, 224, 224).to(device)
output = model(x)
```
