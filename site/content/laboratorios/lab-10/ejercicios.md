---
title: "Ejercicios Practicos"
weight: 40
math: true
---

## Actividad Practica 1 — Parametros de SGD

**Enunciado:** Revise la documentacion de PyTorch para `torch.optim.SGD` y comente los diferentes parametros.

Referencia: [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)

---

## Actividad Practica 2 — Learning rates altos

**Enunciado:**

1. Explique el grafico obtenido con learning rates `[0.0001, 0.001, 0.01, 0.1]`. Por que hay tanta diferencia?
2. Entrene el modelo con learning rates `[1, 10, 100]`.
3. Comente los resultados comparando ambos graficos.

```python
import torch.optim as optim
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()
epochs = 20

fig = plt.figure(figsize=(18, 16))

for lr in [1, 10, 100]:
    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    loss = train(net, trainloader, optimizer, criterion, lr, epochs=epochs)
    test(testloader, net)
    plt.plot(loss, label=lr, linewidth=4)

plt.xlabel("Epochs", fontsize=35)
plt.xticks(fontsize=35)
plt.ylabel("Loss", fontsize=35)
plt.yticks(fontsize=35)
plt.legend(prop={'size': 42})
plt.show()
```

---

## Actividad Practica 3 — Mejor lr para Adam

**Enunciado:**

1. Encuentre un mejor valor de learning rate para Adam (puede cambiar los betas si lo cree necesario).
2. Comente los resultados y explique la diferencia con SGD.

Referencia: [torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

---

## Actividad Practica 4 — Schedulers

**Enunciado:**

Una forma de acelerar el entrenamiento es adaptar el learning rate durante el proceso: pasos grandes al inicio para avanzar rapido, pasos chicos al final para afinar cerca del optimo.

**Criterio 1 — Por epocas (StepLR):** multiplica el lr por un factor `gamma` cada `step_size` epocas.

**Criterio 2 — Por estancamiento (ReduceLROnPlateau):** si el loss no mejora durante `patience` epocas, reduce el lr por un factor.

### 4a — StepLR

```python
criterion = nn.CrossEntropyLoss()
fig = plt.figure(figsize=(10, 6))

for opt in [(0.1, optim.SGD), (0.01, optim.Adam), (0.1, optim.Adagrad)]:
    net = Net().to(device)
    optimizer = opt[1](net.parameters(), lr=opt[0])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss = train(net, trainloader, optimizer, criterion, opt[1].__name__, epochs=epochs, scheduler=scheduler)
    test(testloader, net)
    plt.plot(loss, label=opt[1].__name__, linewidth=2)

plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={'size': 12})
plt.title("StepLR — step=10, gamma=0.1")
plt.show()
```

### 4b — ReduceLROnPlateau

```python
criterion = nn.CrossEntropyLoss()
fig = plt.figure(figsize=(10, 6))

for opt in [(1e-1, optim.SGD), (1e-2, optim.Adam), (1e-1, optim.Adagrad)]:
    net = Net().to(device)
    optimizer = opt[1](net.parameters(), lr=opt[0])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss = train(net, trainloader, optimizer, criterion, opt[1].__name__, epochs=epochs, scheduler=scheduler)
    test(testloader, net)
    plt.plot(loss, label=opt[1].__name__, linewidth=2)

plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={'size': 12})
plt.title("ReduceLROnPlateau")
plt.show()
```
