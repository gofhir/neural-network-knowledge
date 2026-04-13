---
title: "Pipeline de Entrenamiento"
weight: 30
math: true
---

## 1. Vision General

Entrenar una red neuronal en PyTorch sigue un pipeline de 5 pasos que se repiten en cada iteracion:

1. **Forward pass** --- pasar los datos por el modelo para obtener predicciones
2. **Calcular la perdida** --- comparar predicciones con las etiquetas reales
3. **Backward pass** --- calcular los gradientes de la perdida respecto a cada parametro
4. **Actualizar pesos** --- el optimizador ajusta los parametros en la direccion que reduce la perdida
5. **Limpiar gradientes** --- resetear los gradientes a cero para la siguiente iteracion

```
forward → loss → backward → step → zero_grad → (repetir)
```

{{< concept-alert type="clave" >}}
El orden de estos 5 pasos es critico. Invertir `zero_grad` y `backward` acumula gradientes de iteraciones anteriores. Omitir `backward` antes de `step` hace que el optimizador no tenga gradientes para actualizar.
{{< /concept-alert >}}

---

## 2. Funciones de Perdida

La funcion de perdida (loss function) cuantifica **que tan equivocadas estan las predicciones** del modelo. El tipo de problema determina que funcion usar:

### Clasificacion: CrossEntropyLoss

Para problemas donde se predice una de $N$ clases. Internamente combina `LogSoftmax` + `NLLLoss`, por lo que **no** se debe aplicar Softmax en la ultima capa del modelo.

$$\text{CrossEntropy}(y, \hat{y}) = -\sum_{c=1}^{N} y_c \log(\hat{y}_c)$$

```python
from torch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss()

# preds: logits del modelo, shape [batch, num_clases]
# targets: indices de clase, shape [batch] (NO one-hot)
preds = torch.randn(32, 10)       # 32 muestras, 10 clases
targets = torch.randint(0, 10, (32,))  # Etiquetas: numeros de 0 a 9

loss = criterion(preds, targets)
print(loss)  # tensor(2.4531) - un escalar
```

### Regresion: MSELoss

Para problemas donde se predice un valor continuo.

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

```python
from torch.nn import MSELoss

criterion = MSELoss()

preds = torch.randn(32, 1)    # Predicciones
targets = torch.randn(32, 1)  # Valores reales

loss = criterion(preds, targets)
```

---

## 3. Optimizadores

El optimizador implementa el algoritmo que ajusta los pesos del modelo para minimizar la perdida. Se crea vinculandolo a los parametros del modelo:

### SGD (Stochastic Gradient Descent)

```python
from torch.optim import SGD

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Adam (Adaptive Moment Estimation)

Combina momentum y tasas de aprendizaje adaptativas por parametro. En la practica converge mas rapido que SGD para la mayoria de los casos.

```python
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=0.001)
```

### Operaciones del optimizador

```python
# Limpiar gradientes acumulados
optimizer.zero_grad()

# Despues de calcular la perdida y hacer backward:
loss.backward()      # Calcula gradientes
optimizer.step()     # Actualiza parametros usando los gradientes
```

{{< concept-alert type="clave" >}}
`optimizer.zero_grad()` debe llamarse **antes** de `loss.backward()`. PyTorch acumula gradientes por defecto (util para gradient accumulation), asi que si no se limpian, los gradientes de la iteracion anterior se suman a los actuales.
{{< /concept-alert >}}

---

## 4. Backpropagation en PyTorch

El calculo de gradientes es automatico gracias al sistema de **autograd**:

```python
# Forward pass
output = model(x)
loss = criterion(output, target)

# Backward pass - calcula d(loss)/d(param) para cada parametro
loss.backward()

# Ahora cada parametro tiene su gradiente calculado:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad shape = {param.grad.shape}")
```

`loss.backward()` recorre el grafo computacional en orden inverso, calculando el gradiente de la perdida respecto a cada tensor que tenga `requires_grad=True`. Este es el mecanismo que hace posible entrenar redes profundas con millones de parametros.

---

## 5. model.train() vs model.eval()

Algunas capas (BatchNorm, Dropout) se comportan diferente durante entrenamiento y evaluacion:

| Capa | `model.train()` | `model.eval()` |
|---|---|---|
| **Dropout** | Desactiva neuronas aleatoriamente | Todas las neuronas activas |
| **BatchNorm** | Usa media/varianza del batch actual | Usa media/varianza acumuladas (running stats) |

```python
# Durante entrenamiento
model.train()
output = model(x_train)

# Durante evaluacion o inferencia
model.eval()
with torch.no_grad():
    output = model(x_test)
```

---

## 6. torch.no_grad()

Durante la evaluacion no necesitamos gradientes (no vamos a entrenar). `torch.no_grad()` desactiva el tracking de autograd, lo que **reduce el consumo de memoria** y **acelera la computacion**:

```python
# Sin torch.no_grad: PyTorch guarda tensores intermedios para backprop
output = model(x)  # Consume memoria extra

# Con torch.no_grad: solo calcula el resultado
with torch.no_grad():
    output = model(x)  # Mas rapido y menos memoria
```

{{< concept-alert type="recordar" >}}
Siempre usar `model.eval()` **y** `torch.no_grad()` juntos durante evaluacion. `model.eval()` cambia el comportamiento de Dropout/BatchNorm, y `torch.no_grad()` desactiva el grafo de gradientes. Son complementarios, no redundantes.
{{< /concept-alert >}}

---

## 7. Loop de Entrenamiento Completo

Este es el patron estandar que se usa en practicamente todo proyecto de PyTorch:

```python
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

# Configuracion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiAlexNet(num_classes=102).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
n_epochs = 20

# DataLoaders (ver seccion DataLoaders)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Entrenamiento
for epoch in range(1, n_epochs + 1):
    model.train()
    running_loss = 0.0
    total_correctas = 0
    total_muestras = 0

    for x, target in train_loader:
        # Mover datos a GPU
        x = x.to(device)
        target = target.to(device)

        # 1. Limpiar gradientes
        optimizer.zero_grad()

        # 2. Forward pass
        output = model(x)

        # 3. Calcular perdida
        loss = criterion(output, target)

        # 4. Backward pass
        loss.backward()

        # 5. Actualizar pesos
        optimizer.step()

        # Metricas
        running_loss += loss.item()
        preds = output.argmax(dim=1)
        total_correctas += (preds == target).sum().item()
        total_muestras += target.size(0)

    accuracy = total_correctas / total_muestras
    avg_loss = running_loss / len(train_loader)
    print(f"Epoca {epoch}/{n_epochs} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f}")
```

---

## 8. Evaluacion del Modelo

```python
def evaluar(model, dataloader, device):
    model.eval()
    total_correctas = 0
    total_muestras = 0

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)

            output = model(x)
            preds = output.argmax(dim=1)

            total_correctas += (preds == target).sum().item()
            total_muestras += target.size(0)

    accuracy = total_correctas / total_muestras
    return accuracy

# Evaluar en train y test
train_acc = evaluar(model, train_loader, device)
test_acc = evaluar(model, test_loader, device)
print(f"Train accuracy: {train_acc:.2%}")
print(f"Test accuracy: {test_acc:.2%}")
```

La diferencia entre train accuracy y test accuracy es el **gap de generalizacion**. Un gap grande indica overfitting: el modelo memorizo los datos de entrenamiento pero no aprendio patrones generalizables.

---

## 9. Guardar y Cargar Modelos

```python
# Guardar los pesos del modelo
torch.save(model.state_dict(), 'modelo_alexnet.pth')

# Cargar los pesos en un modelo nuevo
model_nuevo = MiAlexNet(num_classes=102)
model_nuevo.load_state_dict(torch.load('modelo_alexnet.pth', map_location=device))
model_nuevo = model_nuevo.to(device)
model_nuevo.eval()
```

{{< concept-alert type="clave" >}}
`state_dict()` guarda solo los parametros aprendidos (pesos y biases), no la arquitectura. Para cargar los pesos es necesario crear primero una instancia del modelo con la misma arquitectura. Esto es mas flexible que guardar el modelo completo con `torch.save(model, ...)`.
{{< /concept-alert >}}
