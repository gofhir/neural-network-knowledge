---
title: "Ejercicios"
weight: 40
math: true
---

Ejercicios extraidos del practico de la Clase 06. Se organizan en dos secciones: funciones de activacion e inicializacion de pesos.

---

## Seccion 1: Funciones de Activacion

### Actividad 2.1 -- Visualizar Sigmoid y su derivada

**Contexto:** En el notebook se explorar la funcion ReLU, graficando tanto la funcion como su derivada. Ahora se debe repetir el ejercicio con la funcion **Sigmoid**.

**Tareas:**

1. Confeccionar el grafico de la funcion Sigmoid para valores entre $-10$ y $10$
2. Confeccionar el grafico de su derivada en el mismo rango

**Codigo base para la funcion:**

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

**Codigo base para la derivada:**

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

**Que observar:** la sigmoid comprime cualquier valor real al rango $(0, 1)$. Su derivada tiene un maximo de 0.25 en $x = 0$ y tiende rapidamente a cero para $|x| > 3$.

---

### Actividad 2.2 -- Comparacion ReLU vs Sigmoid

**Pregunta:** Entre un modelo con funcion de activacion ReLU y otro con funcion de activacion Sigmoid, cual cree usted que aprenderia mas rapido? Justifique su respuesta apoyandose en los graficos generados anteriormente.

**Pistas para la respuesta:**

- Comparar el gradiente maximo de cada funcion
- Analizar en que rango del dominio cada funcion tiene gradiente significativo
- Considerar que ocurre cuando se multiplican gradientes a traves de muchas capas

---

## Seccion 2: Inicializacion de Pesos

El laboratorio utiliza una red neuronal de **10 capas** sin funcion de activacion para aislar el efecto de la inicializacion. La red tiene la siguiente arquitectura:

- 9 capas ocultas de 2 neuronas cada una ($\text{Linear}(2, 2)$)
- 1 capa de salida de 1 neurona ($\text{Linear}(2, 1)$)
- Entrada: vector $[0.5, 0.5]$, etiqueta: $1$
- Optimizador: Adam con $\text{lr} = 10^{-4}$

**Modelo base (ya proporcionado en el notebook):**

```python
class DeepNN(nn.Module):

    def __init__(self, weights_initial_value):
        super(DeepNN, self).__init__()
        self.layer_0 = nn.Linear(2, 2)
        self.layer_1 = nn.Linear(2, 2)
        # ... capas 2 a 8 ...
        self.layer_9 = nn.Linear(2, 1)

        # Inicializar todos los pesos al mismo valor constante
        for i in range(10):
            layer = getattr(self, f'layer_{i}')
            nn.init.constant_(layer.weight.data, weights_initial_value)

    def forward(self, x):
        for i in range(10):
            x = getattr(self, f'layer_{i}')(x)
        return x
```

Los experimentos previos en el notebook muestran los resultados de inicializar con valor 0.1 (menor a 1) y con valor 1.5 (mayor a 1).

---

### Actividad 3.1 -- Identificar los fenomenos

**Pregunta 1:** Como se llama el fenomeno observado al inicializar los pesos con valores menor a 1?

**Pregunta 2:** Como se llama el fenomeno observado al inicializar los pesos con valores mayor a 1?

**Pistas:** observar los graficos de gradientes y loss generados por la funcion `fit()`. Para pesos < 1, los gradientes tienden a cero. Para pesos > 1, los gradientes crecen de forma descontrolada.

---

### Actividad 3.2 -- Xavier Glorot

**Tarea:** Repetir el experimento anterior, pero inicializando los pesos a traves del metodo de Xavier Glorot usando `torch.nn.init.xavier_uniform_`.

Se debe:

1. Redefinir la clase del modelo, reemplazando `nn.init.constant_` por `nn.init.xavier_uniform_`
2. Crear una instancia del nuevo modelo
3. Llamar a la funcion `fit()` con el modelo como parametro

**Preguntas despues de ejecutar:**

- Que efecto puede ver de la aplicacion de este nuevo metodo de inicializacion de pesos?
- A que se deben los efectos observados?

**Referencia:** [PyTorch Xavier Uniform](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_)
