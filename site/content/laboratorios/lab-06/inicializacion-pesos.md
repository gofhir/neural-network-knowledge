---
title: "Inicializacion de Pesos"
weight: 30
math: true
---

La inicializacion de pesos es una de las decisiones mas criticas al disenar una red neuronal profunda. Una mala eleccion puede hacer que el entrenamiento sea extremadamente lento o directamente imposible. Esta seccion explica los problemas que surgen con inicializaciones ingenuas y presenta las estrategias que los resuelven.

---

## Por que importa la inicializacion

Consideremos una red profunda con $L$ capas, sin funciones de activacion. La salida del modelo en el forward pass es:

$$y = W^{[L]} \cdot W^{[L-1]} \cdots W^{[2]} \cdot W^{[1]} \cdot \vec{x}$$

Esto equivale a elevar la matriz de pesos a la potencia $L-1$:

$$y = W^{[L]} \cdot \left(W\right)^{L-1} \cdot \vec{x}$$

El comportamiento de esta multiplicacion depende criticamente de los **valores propios** de la matriz $W$.

---

## Vanishing input signal (senal que se desvanece)

Si los pesos son menores que 1, por ejemplo:

$$W^{[l]} = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}$$

Entonces:

$$y = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}^{L-1} \cdot \vec{x}$$

Con $L = 10$ capas, el factor de escala es $0.5^9 \approx 0.002$. La senal de entrada se reduce a practicamente cero al llegar a la salida. Este es el fenomeno de **vanishing input signal**.

---

## Exploding input signal (senal que explota)

Si los pesos son mayores que 1, por ejemplo:

$$W^{[l]} = \begin{bmatrix} 1.5 & 0 \\ 0 & 1.5 \end{bmatrix}$$

Entonces:

$$y = \begin{bmatrix} 1.5 & 0 \\ 0 & 1.5 \end{bmatrix}^{L-1} \cdot \vec{x}$$

Con $L = 10$ capas, el factor de escala es $1.5^9 \approx 38.44$. La senal crece exponencialmente, produciendo valores numericamente inestables. Este es el fenomeno de **exploding input signal**.

---

## Vanishing gradient (gradiente que se desvanece)

El mismo problema ocurre en el backward pass. El gradiente se calcula como el producto de las derivadas locales:

$$\frac{\partial y}{\partial x_i} = f'^{[1]} \cdot f'^{[2]} \cdots f'^{[L-1]} \cdot f'^{[L]} \cdot \vec{g}$$

Si las derivadas $f'^{[l]}$ tienen valores menores a 1 (como ocurre con sigmoid, cuyo maximo es 0.25), los gradientes se multiplican y se reducen exponencialmente:

$$0.5^{L-1} \rightarrow 0$$

Las capas cercanas a la entrada reciben gradientes practicamente nulos y **dejan de aprender**. Este es el problema de **vanishing gradient**.

{{< concept-alert type="clave" >}}
El vanishing gradient es especialmente grave en redes profundas con funciones de activacion como sigmoid o tanh. Es una de las razones por las que ReLU se convirtio en la activacion estandar para capas ocultas.
{{< /concept-alert >}}

---

## Exploding gradient (gradiente que explota)

Si las derivadas locales tienen valores mayores a 1:

$$1.5^{L-1} \rightarrow \infty$$

Los gradientes crecen exponencialmente, produciendo actualizaciones de pesos enormes que desestabilizan el entrenamiento. El modelo oscila violentamente y no converge. Este es el problema de **exploding gradient**.

---

## Inicializacion Xavier/Glorot

Glorot y Bengio (2010) propusieron una estrategia de inicializacion que mantiene la varianza de la senal estable entre capas. La idea central es:

**Hacer que la varianza de la entrada y la salida de cada capa sean iguales.**

Para lograr esto, los pesos se inicializan con valores aleatorios de una distribucion gaussiana (o uniforme) con media cero y varianza:

$$\text{Var}(W_i) = \frac{2}{fan\_in + fan\_out}$$

donde:

- $fan\_in$: numero de entradas a la capa
- $fan\_out$: numero de salidas de la capa

### Variante uniforme

En la variante uniforme, los pesos se muestrean de:

$$W \sim U\left[-\sqrt{\frac{6}{fan\_in + fan\_out}}, \sqrt{\frac{6}{fan\_in + fan\_out}}\right]$$

### Por que funciona

Al igualar la varianza de entrada y salida, se evita que la senal crezca o se reduzca al pasar por cada capa. Esto mantiene los gradientes en un rango razonable durante el backward pass, permitiendo que todas las capas aprendan a una velocidad similar.

{{< concept-alert type="clave" >}}
Xavier/Glorot fue disenado para funciones de activacion simetricas (sigmoid, tanh). Para ReLU, que anula la mitad de las entradas, se necesita una variante diferente: la inicializacion He.
{{< /concept-alert >}}

---

## Inicializacion He (Kaiming)

He et al. (2015) propusieron una variante especifica para redes con activacion ReLU. Dado que ReLU anula todas las entradas negativas (la mitad en promedio), la varianza se reduce a la mitad en cada capa. Para compensar:

$$\text{Var}(W_i) = \frac{2}{fan\_in}$$

Esta inicializacion duplica la varianza respecto a Xavier, compensando la "muerte" de la mitad de las activaciones.

---

## Implementacion en PyTorch

PyTorch ofrece funciones de inicializacion en el modulo `torch.nn.init`:

### Funciones principales

```python
import torch.nn as nn

# Inicializacion con valor constante (para experimentacion)
torch.nn.init.constant_(layer.weight.data, val)

# Inicializacion Xavier/Glorot uniforme
torch.nn.init.xavier_uniform_(layer.weight.data)

# Inicializacion Xavier/Glorot normal
torch.nn.init.xavier_normal_(layer.weight.data)

# Inicializacion He/Kaiming normal (para ReLU)
torch.nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')

# Inicializacion He/Kaiming uniforme (para ReLU)
torch.nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')
```

### Ejemplo completo: red de 10 capas con Xavier

```python
import torch
import torch.nn as nn

class DeepNNXavier(nn.Module):

    def __init__(self):
        super(DeepNNXavier, self).__init__()
        self.layer_0 = nn.Linear(2, 2)
        self.layer_1 = nn.Linear(2, 2)
        self.layer_2 = nn.Linear(2, 2)
        self.layer_3 = nn.Linear(2, 2)
        self.layer_4 = nn.Linear(2, 2)
        self.layer_5 = nn.Linear(2, 2)
        self.layer_6 = nn.Linear(2, 2)
        self.layer_7 = nn.Linear(2, 2)
        self.layer_8 = nn.Linear(2, 2)
        self.layer_9 = nn.Linear(2, 1)

        # Aplicar Xavier a todas las capas
        for i in range(10):
            layer = getattr(self, f'layer_{i}')
            nn.init.xavier_uniform_(layer.weight.data)

    def forward(self, x):
        for i in range(10):
            x = getattr(self, f'layer_{i}')(x)
        return x
```

### Ejemplo: red con He para ReLU

```python
class DeepNNHe(nn.Module):

    def __init__(self):
        super(DeepNNHe, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 2) for _ in range(9)
        ] + [nn.Linear(2, 1)])

        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
```

---

## Conexion entre inicializacion y activacion

La eleccion de la estrategia de inicializacion esta directamente ligada a la funcion de activacion utilizada:

| Activacion | Inicializacion recomendada | Razon |
|-----------|---------------------------|-------|
| Sigmoid / Tanh | Xavier/Glorot | Activaciones simetricas, no anulan entradas |
| ReLU | He/Kaiming | ReLU anula mitad de entradas, necesita mayor varianza |
| Leaky ReLU | He/Kaiming | Similar a ReLU, con ajuste por pendiente negativa |
| GELU / ELU | Xavier o He | Depende de la red y el problema |

---

## Resumen

La inicializacion de pesos no es una solucion definitiva al problema de vanishing/exploding gradients, pero es una herramienta fundamental para hacer factible el entrenamiento de redes profundas. Junto con la eleccion correcta de funcion de activacion y tecnicas como batch normalization, permite que los gradientes fluyan de forma estable a traves de las capas durante el entrenamiento.

| Inicializacion | Efecto en gradientes | Efecto en loss |
|---------------|---------------------|----------------|
| Constante < 1 | Desvanecimiento (vanishing) | No converge |
| Constante > 1 | Explosion (exploding) | Inestable |
| Xavier/Glorot | Estables | Convergencia suave |
| He/Kaiming | Estables con ReLU | Convergencia suave |
