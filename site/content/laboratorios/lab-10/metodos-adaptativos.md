---
title: "Metodos Adaptativos y Schedulers"
weight: 30
math: true
---

## Mas alla de SGD

SGD usa el mismo learning rate para todos los parametros. Los metodos adaptativos mejoran esto asignando un learning rate individual a cada parametro.

---

## AdaGrad

Acumula la historia de gradientes al cuadrado para cada parametro. Los parametros que recibieron gradientes grandes en el pasado tendran un learning rate mas bajo, y viceversa.

$$\theta_j = \theta_j - \frac{\alpha}{\sqrt{G_j + \epsilon}} \cdot \nabla L(\theta_j)$$

Donde $G_j$ es la suma de gradientes al cuadrado para el parametro $j$.

**Ventaja:** ajusta automaticamente el lr por parametro.
**Desventaja:** el lr solo puede bajar (nunca sube), lo que puede hacer que el entrenamiento se detenga prematuramente.

---

## Adam

Combina momentum (primer momento) con la normalizacion de AdaGrad (segundo momento):

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla L(\theta)$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla L(\theta))^2$$
$$\theta = \theta - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- $\beta_1 = 0.9$ — controla el momentum (media movil del gradiente)
- $\beta_2 = 0.999$ — controla la normalizacion (media movil del gradiente al cuadrado)
- $\hat{m}_t, \hat{v}_t$ son versiones corregidas por sesgo

**En la practica, Adam es el optimizador mas usado** porque converge rapido y requiere menos ajuste de hiperparametros.

---

## Comparacion en el lab

```python
for opt in [(0.1, optim.SGD), (0.01, optim.Adam), (0.1, optim.Adagrad)]:
    net = Net().to(device)
    optimizer = opt[1](net.parameters(), lr=opt[0])
    loss = train(net, trainloader, optimizer, criterion, opt[1], epochs=20)
```

Notar que Adam usa `lr=0.01` mientras SGD usa `lr=0.1`. Adam se beneficia de learning rates mas bajos porque ya incorpora normalizacion interna.

---

## Schedulers — ajustar el lr durante el entrenamiento

En vez de usar un lr fijo, se puede ir reduciendo durante el entrenamiento. La logica: al principio pasos grandes para avanzar rapido, al final pasos chicos para afinar.

### StepLR — reducir cada N epocas

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

Multiplica el lr por `gamma=0.1` cada 10 epocas. Si empiezas con `lr=0.1`:
- Epocas 1-10: lr = 0.1
- Epocas 11-20: lr = 0.01
- Epocas 21-30: lr = 0.001

### ReduceLROnPlateau — reducir cuando el loss se estanca

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
```

Monitorea el loss. Si no mejora despues de `patience` epocas, reduce el lr por un factor. Es mas inteligente que StepLR porque se adapta al comportamiento real del entrenamiento.

{{< concept-alert type="clave" >}}
El scheduler no reemplaza la eleccion del optimizador — es un complemento. La combinacion tipica en la practica es **Adam + ReduceLROnPlateau**.
{{< /concept-alert >}}
