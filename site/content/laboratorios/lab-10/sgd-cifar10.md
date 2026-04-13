---
title: "SGD en CIFAR10"
weight: 20
math: true
---

## De funciones simples a imagenes reales

La Actividad 1 uso gradient descent manual con funciones de 2-3 parametros. Ahora pasamos a un problema real: clasificar imagenes de CIFAR10 (10 clases) con una CNN que tiene miles de parametros. El principio es el mismo — minimizar el loss ajustando pesos — pero ahora usamos PyTorch.

---

## Datos: CIFAR10

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=2)
```

- **50,000 imagenes** de entrenamiento, **10,000** de test
- **10 clases:** avion, auto, pajaro, gato, ciervo, perro, rana, caballo, barco, camion
- Cada imagen es **32x32 pixeles** con 3 canales (RGB)
- `Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))` transforma los pixeles de [0,1] a [-1,1]
- `batch_size=512`: en vez de usar todas las imagenes a la vez (como en GD puro), se procesan de a 512 — esto es **Stochastic Gradient Descent (SGD)**

---

## El modelo: una CNN pequeña

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)       # 3 canales → 6 filtros de 5x5
        self.pool = nn.MaxPool2d(2, 2)         # reduce tamaño a la mitad
        self.conv2 = nn.Conv2d(6, 16, 5)       # 6 canales → 16 filtros de 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # aplanar y reducir a 120
        self.fc2 = nn.Linear(120, 10)           # 120 → 10 clases

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # conv1 → relu → pool
        x = self.pool(F.relu(self.conv2(x)))   # conv2 → relu → pool
        x = x.view(-1, 16 * 5 * 5)            # aplanar
        x = F.relu(self.fc1(x))                # fc1 → relu
        x = self.fc2(x)                        # fc2 → salida
        return x
```

El flujo de dimensiones:

```
Input:  3 x 32 x 32
conv1:  6 x 28 x 28  →  pool: 6 x 14 x 14
conv2: 16 x 10 x 10  →  pool: 16 x 5 x 5
flatten: 400
fc1: 120
fc2: 10 (una por clase)
```

Es un modelo intencionalmente pequeno — el foco del lab no es la arquitectura sino como la **optimizacion** afecta el entrenamiento.

---

## Loop de entrenamiento

```python
def train(net, trainloader, optimizer, criterion, lr, epochs=60, scheduler=None):
    total_loss = []
    net.train()
    for epoch in range(epochs):
        running_loss = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)                # forward
            loss = criterion(outputs, labels)    # calcular error
            loss.backward()                      # calcular gradientes
            optimizer.step()                     # actualizar pesos
            optimizer.zero_grad()                # limpiar gradientes

            running_loss.append(loss.item())
        total_loss.append(np.mean(running_loss))
    return total_loss
```

Paso a paso:

| Linea | Que hace |
|-------|---------|
| `outputs = net(inputs)` | Forward pass — la red predice |
| `loss = criterion(outputs, labels)` | Calcula el error (CrossEntropyLoss) |
| `loss.backward()` | Backpropagation — calcula el gradiente de cada peso |
| `optimizer.step()` | Actualiza los pesos usando el gradiente |
| `optimizer.zero_grad()` | Limpia los gradientes para el siguiente batch |

{{< concept-alert type="clave" >}}
`optimizer.zero_grad()` es crucial. Sin esta linea, los gradientes se **acumulan** entre batches, lo que corrompe la actualizacion de pesos.
{{< /concept-alert >}}

---

## Efecto del learning rate en SGD

El lab entrena el mismo modelo con 4 valores de learning rate:

```python
for lr in [0.0001, 0.001, 0.01, 0.1]:
    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    loss = train(net, trainloader, optimizer, criterion, lr, epochs=20)
```

Cada vez se crea un modelo **nuevo** (pesos aleatorios) y se entrena con un lr diferente. El grafico resultante muestra como baja el loss para cada lr.

---

## Momentum y Nesterov

Ademas del learning rate, SGD tiene dos hiperparametros adicionales:

**Momentum ($\rho$):** suaviza el movimiento incorporando la historia del gradiente. En vez de moverse solo segun el gradiente actual, mantiene "inercia" de la direccion previa.

$$v_t = \rho \cdot v_{t-1} + \nabla L(\theta)$$
$$\theta = \theta - \alpha \cdot v_t$$

**Nesterov:** variante del momentum donde primero se aplica el momentum y luego se calcula el gradiente en la posicion proyectada. Es una "mirada al futuro" que mejora la convergencia.

El lab compara momentum con valores `[0.0, 0.3, 0.6, 0.9]` y luego `nesterov=True` vs `nesterov=False`.
