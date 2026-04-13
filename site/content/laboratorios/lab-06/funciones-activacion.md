---
title: "Funciones de Activacion"
weight: 20
math: true
---

Las funciones de activacion son un componente esencial de las redes neuronales. Sin ellas, una red profunda con multiples capas seria equivalente a una sola transformacion lineal, independientemente del numero de capas. Esta seccion analiza las principales funciones de activacion, sus propiedades matematicas, sus problemas y como implementarlas en PyTorch.

---

## Por que se necesita no-linealidad

La arquitectura de una red neuronal artificial (ANN) se define por tres elementos:

1. **Estructura** -- numero de capas y neuronas por capa
2. **Pesos** $W_{ij}$ -- conectan la neurona $i$ con la neurona $j$
3. **Funcion de activacion** -- modula la senal enviada desde una neurona a la siguiente

Un perceptron calcula:

$$\hat{y} = f\left(\sum_i w_i x_i + b\right)$$

Si $f$ fuera la identidad (sin activacion), la composicion de multiples capas seguiria siendo una funcion lineal:

$$y = W_n \cdot W_{n-1} \cdots W_1 \cdot x = W_{total} \cdot x$$

{{< concept-alert type="clave" >}}
Sin funciones de activacion no lineales, un perceptron multicapa (MLP) no puede resolver problemas de mayor complejidad que un perceptron simple. La no-linealidad es lo que permite a las redes profundas aproximar funciones arbitrariamente complejas.
{{< /concept-alert >}}

---

## Sigmoid

### Formula y derivada

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\frac{\partial \sigma(x)}{\partial x} = \sigma(x)(1 - \sigma(x))$$

### Buenas propiedades

- Suave y derivable en todo su dominio
- Lleva la salida a valores entre 0 y 1, lo que es util para interpretacion como probabilidad en problemas de clasificacion

### Malas propiedades

**Problema 1: Vanishing gradient.** Cuando las neuronas se saturan (valores de entrada muy grandes o muy pequenos), el gradiente se acerca a cero. Recordemos la regla de actualizacion:

$$w_i^{new} = w_i^{old} - \eta \frac{\partial E}{\partial w_i}$$

Si $\frac{\partial \sigma}{\partial x} \approx 0$, entonces $\frac{\partial E}{\partial w_i} \approx 0$, lo que produce convergencia extremadamente lenta. El gradiente maximo de la sigmoid es apenas **0.25** (en $x = 0$), y para $|x| > 3$ el gradiente es practicamente nulo.

En deep learning con muchas capas, este efecto se amplifica exponencialmente -- es el problema conocido como *vanishing gradient*, especialmente critico en redes recurrentes (RNNs).

**Problema 2: Salida no centrada en cero.** La sigmoid produce valores en $(0, 1)$, nunca negativos. Esto causa que los gradientes de los pesos sean todos positivos o todos negativos, produciendo un efecto de **zig-zag** en la convergencia durante la optimizacion.

---

## Tanh (Tangente Hiperbolica)

### Formula y derivada

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

$$\frac{\partial \tanh(x)}{\partial x} = 1 - \tanh^2(x)$$

### Buenas propiedades

- Suave y derivable
- **Centrada en cero** -- su salida esta en el rango $(-1, 1)$, resolviendo el problema de zig-zag de la sigmoid
- Util para clasificacion

### Malas propiedades

- Neuronas saturadas tienen gradiente cercano a cero (mismo problema de *vanishing gradient*)
- Convergencia lenta
- Extremadamente sensible para valores de salida cercanos a cero

---

## ReLU (Rectified Linear Unit)

### Formula y derivada

$$\text{ReLU}(x) = \max(0, x)$$

$$\frac{\partial \text{ReLU}(x)}{\partial x} = \begin{cases} 1 & \text{si } x > 0 \\ 0 & \text{si } x < 0 \end{cases}$$

### Buenas propiedades

- **No se satura** para entradas positivas -- el gradiente es siempre 1
- **Convergencia rapida** -- hasta 6 veces mas rapido que usando tanh (Krizhevsky et al., 2012, en CIFAR-10 con CNN de 4 capas)
- **Eficiente computacionalmente** -- se calcula como una simple comparacion con cero

### Malas propiedades

- No es derivable en $x = 0$
- **Se satura para entradas negativas** -- el gradiente es exactamente 0, lo que puede "matar" neuronas permanentemente (*dying ReLU*)
- No acota la salida, por lo que no es directamente util como capa de salida en clasificacion

{{< concept-alert type="clave" >}}
ReLU es la funcion de activacion mas utilizada en capas ocultas de redes profundas. Su gradiente constante de 1 para valores positivos evita el vanishing gradient y permite entrenar redes mucho mas profundas que con sigmoid o tanh.
{{< /concept-alert >}}

---

## Leaky ReLU y Parametric ReLU

### Formula

$$\text{LeakyReLU}(x) = \max(\alpha x, x)$$

donde $\alpha$ es un valor pequeno (tipicamente 0.01). En la variante **Parametric ReLU (PReLU)**, $\alpha$ es un parametro aprendible.

### Buenas propiedades

- **No se satura** para ninguna entrada -- las neuronas no dejan de aprender
- Convergencia rapida (similar a ReLU)
- Eficiente computacionalmente

### Malas propiedades

- Introduce un hiperparametro adicional ($\alpha$)
- No acota la salida

---

## Otras funciones modernas

### ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha(e^x - 1) & \text{si } x \leq 0 \end{cases}$$

Combina las ventajas de ReLU (no satura para positivos) con una salida media mas cercana a cero para entradas negativas.

### GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

donde $\Phi(x)$ es la funcion de distribucion acumulada de la normal estandar. GELU es la activacion utilizada en modelos Transformer como BERT y GPT. A diferencia de ReLU, GELU es suave y no tiene discontinuidades.

---

## Softmax (capa de salida)

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Softmax se usa normalmente en la **capa de salida** para problemas de clasificacion multiclase. Convierte un vector de valores reales en un vector de probabilidades:

- Es derivable
- Funciona como un "maximizador suave" -- amplifica los valores mas altos y atenua los bajos
- Incentiva competencia entre salidas (*the winner takes it all*)

---

## Comparativa resumen

| Funcion | Rango | Centrada | Gradiente max | Saturacion | Uso tipico |
|---------|-------|----------|---------------|------------|------------|
| Sigmoid | $(0, 1)$ | No | 0.25 | Ambos lados | Salida binaria |
| Tanh | $(-1, 1)$ | Si | 1.0 | Ambos lados | Capas ocultas (legacy) |
| ReLU | $[0, \infty)$ | No | 1.0 | Solo negativos | Capas ocultas (estandar) |
| Leaky ReLU | $(-\infty, \infty)$ | No | 1.0 | No | Capas ocultas |
| Softmax | $(0, 1)$ | -- | -- | -- | Salida multiclase |

---

## Visualizacion en PyTorch

PyTorch permite explorar las funciones de activacion y sus gradientes de forma directa usando `autograd`.

### Graficar una funcion de activacion

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

activation = torch.nn.Sigmoid()

X = []
Fx = []

for x in np.arange(-10, 10, 0.1):
    X.append(x)
    x_tensor = torch.Tensor([x])
    result = activation(x_tensor)
    Fx.append(float(result))

plt.scatter(X, Fx)
plt.title('Funcion Sigmoid')
plt.xlabel('x')
plt.ylabel('sigma(x)')
plt.show()
```

### Graficar la derivada (gradiente)

```python
from torch.autograd import Variable

activation = torch.nn.Sigmoid()

X = []
gradiente = []

for x in np.arange(-10, 10, 0.1):
    X.append(x)
    x_tensor = Variable(torch.Tensor([x]), requires_grad=True)
    result = activation(x_tensor)
    result.backward()
    gradiente.append(float(x_tensor.grad))

plt.scatter(X, gradiente)
plt.title('Derivada Sigmoid')
plt.xlabel('x')
plt.ylabel("sigma'(x)")
plt.show()
```

### Comparar ReLU y Sigmoid

```python
activations = {
    'ReLU': torch.nn.ReLU(),
    'Sigmoid': torch.nn.Sigmoid(),
    'Tanh': torch.nn.Tanh(),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, fn) in zip(axes, activations.items()):
    X, Fx = [], []
    for x in np.arange(-5, 5, 0.1):
        X.append(x)
        Fx.append(float(fn(torch.Tensor([x]))))
    ax.plot(X, Fx)
    ax.set_title(name)
    ax.grid(True)

plt.tight_layout()
plt.show()
```

{{< concept-alert type="clave" >}}
Para evaluar una funcion de activacion en PyTorch, se instancia como un objeto (`torch.nn.ReLU()`) y luego se aplica a tensores. Para obtener su derivada, se usa `requires_grad=True` en el tensor de entrada y se llama a `.backward()` sobre el resultado.
{{< /concept-alert >}}
