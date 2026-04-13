---
title: "Resolucion del Laboratorio"
weight: 50
math: true
---

Resolucion de las actividades del laboratorio.

---

## Actividad 2.1 -- Grafico de Sigmoid y su derivada

### Grafico de la funcion Sigmoid

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

activation_function = torch.nn.Sigmoid()

X = []
Fx = []

for x in np.arange(-10, 10, 0.1):
    X.append(x)
    x_tensor = torch.Tensor([x])
    result = activation_function(x_tensor)
    Fx.append(float(result))

plt.scatter(X, Fx)
plt.title('Funcion Sigmoid')
plt.show()
```

**Resultado:** la curva tiene forma de "S". Valores muy negativos producen salidas cercanas a 0, y valores muy positivos producen salidas cercanas a 1. La transicion ocurre en el rango $[-5, 5]$ aproximadamente, con el punto de inflexion en $x = 0$ donde $\sigma(0) = 0.5$.

### Grafico de la derivada de Sigmoid

```python
from torch.autograd import Variable

X = []
gradiente = []

for x in np.arange(-10, 10, 0.1):
    X.append(x)
    x_tensor = Variable(torch.Tensor([x]), requires_grad=True)
    result = activation_function(x_tensor)
    result.backward()
    gradiente.append(float(x_tensor.grad))

plt.scatter(X, gradiente)
plt.title('Derivada Sigmoid')
plt.show()
```

**Resultado:** la derivada tiene forma de campana centrada en $x = 0$. El valor maximo es $\sigma'(0) = 0.25$. Para $|x| > 5$, el gradiente es practicamente cero.

---

## Actividad 2.2 -- ReLU vs Sigmoid

**Pregunta:** Entre un modelo con funcion de activacion ReLU y otro con funcion de activacion Sigmoid, cual aprenderia mas rapido?

**Respuesta:**

ReLU aprende mas rapido. La razon principal esta en la magnitud de los gradientes:

- La derivada de **Sigmoid** tiene un gradiente maximo de **0.25** (en $x = 0$). Para valores de entrada mayores o menores que 3, el gradiente se acerca rapidamente a 0, por lo que las neuronas saturadas reciben actualizaciones de pesos muy pequenas.

- La derivada de **ReLU** para todo $x > 0$ es **1** (constante). Esto significa que el gradiente es 4 veces mayor que el maximo de sigmoid, y ademas es constante -- no disminuye con la magnitud de la entrada.

En redes profundas, estos factores se multiplican capa por capa. Con sigmoid, $0.25^L$ se acerca rapidamente a cero (*vanishing gradient*), mientras que con ReLU el gradiente se mantiene en 1 para activaciones positivas. Krizhevsky et al. (2012) demostraron empiricamente que ReLU converge hasta 6 veces mas rapido que tanh en una CNN de 4 capas sobre CIFAR-10.

---

## Actividad 3.1 -- Identificacion de fenomenos

### Pesos inicializados con valor menor a 1 (valor = 0.1)

**Fenomeno:** **Desvanecimiento de gradiente** (*vanishing gradient*).

Los pesos menores a 1 se multiplican en todas las capas y el valor se reduce exponencialmente. Con pesos de 0.1 en una red de 10 capas, el factor de escala en la primera capa es del orden de $0.1^9 \approx 10^{-9}$, haciendo que los gradientes sean practicamente nulos y que la red no aprenda.

En los graficos se observa que:
- El gradiente de la primera capa se mantiene extremadamente cercano a cero durante todo el entrenamiento
- La funcion de perdida desciende de forma lentisima o se estanca

### Pesos inicializados con valor mayor a 1 (valor = 1.5)

**Fenomeno:** **Explosion de gradiente** (*exploding gradient*).

Los pesos mayores a 1 se van multiplicando en todas las capas y el valor crece exponencialmente. Con pesos de 1.5 en 10 capas, el factor de escala es $1.5^9 \approx 38.44$ en el forward, y un efecto similar ocurre en el backward con los gradientes.

En los graficos se observa que:
- Los gradientes tienen magnitudes muy grandes e inestables
- La funcion de perdida oscila violentamente o diverge

---

## Actividad 3.2 -- Xavier Glorot

### Codigo del modelo

```python
class DeepNNXavier(nn.Module):

    def __init__(self):
        super(DeepNNXavier, self).__init__()
        self.layer_0 = torch.nn.Linear(2, 2)
        self.layer_1 = torch.nn.Linear(2, 2)
        self.layer_2 = torch.nn.Linear(2, 2)
        self.layer_3 = torch.nn.Linear(2, 2)
        self.layer_4 = torch.nn.Linear(2, 2)
        self.layer_5 = torch.nn.Linear(2, 2)
        self.layer_6 = torch.nn.Linear(2, 2)
        self.layer_7 = torch.nn.Linear(2, 2)
        self.layer_8 = torch.nn.Linear(2, 2)
        self.layer_9 = torch.nn.Linear(2, 1)
        torch.nn.init.xavier_uniform_(self.layer_0.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_1.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_2.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_3.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_4.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_5.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_6.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_7.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_8.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_9.weight.data)

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        return x
```

### Creacion e instancia del modelo

```python
model = DeepNNXavier()
print(model.layer_0.weight.data)
# Ejemplo de salida:
# tensor([[ 1.1083, -0.2710],
#         [-1.0929,  0.5795]])
```

Los pesos ya no son todos iguales -- son valores aleatorios muestreados de una distribucion uniforme con varianza $\frac{2}{fan\_in + fan\_out} = \frac{2}{2 + 2} = 0.5$.

### Entrenamiento

```python
fit(model)
```

### Pregunta 1: Que efecto tiene Xavier?

**Respuesta:** Los gradientes se mantienen en un rango razonable -- no hay desvanecimiento ni explosion. La funcion de perdida desciende de forma consistente y suave, sin los problemas observados con inicializacion constante. Los gradientes no son ni extremadamente pequenos ni extremadamente grandes.

### Pregunta 2: A que se deben los efectos observados?

**Respuesta:** El metodo de Xavier Glorot inicializa los pesos con varianza $\frac{2}{fan\_in + fan\_out}$, lo que asegura que la varianza de la senal se mantenga igual entre la entrada y la salida de cada capa. Al mantener esta varianza constante a traves de las capas, se evita tanto el vanishing como el exploding gradient, permitiendo que los gradientes fluyan de forma estable durante el backward pass y que todas las capas aprendan a una velocidad similar.
