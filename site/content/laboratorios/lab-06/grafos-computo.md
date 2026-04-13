---
title: "Grafos de Computo"
weight: 10
math: true
---

Los grafos de computo son la estructura fundamental que permite a los frameworks de deep learning (PyTorch, TensorFlow, JAX) implementar backpropagation de forma automatica. Esta seccion desarrolla la teoria detras de estas representaciones y muestra como PyTorch las utiliza internamente.

---

## Que es un grafo de computo

Un grafo de computo es un **grafo dirigido** donde:

- Los **nodos** representan operaciones matematicas (suma, multiplicacion, funciones de activacion)
- Las **aristas** representan el flujo de datos (tensores) entre operaciones

Cualquier funcion matematica, por compleja que sea, puede descomponerse en una secuencia de operaciones elementales representadas como un grafo.

**Ejemplo:** para la funcion $g = (x + y) \cdot z$, el grafo tiene dos nodos:

1. Un nodo de **suma**: $a = x + y$
2. Un nodo de **multiplicacion**: $g = a \cdot z$

Las variables $x$, $y$ y $z$ son las entradas (hojas del grafo) y $g$ es la salida.

{{< concept-alert type="clave" >}}
Los grafos de computo permiten descomponer funciones complejas en operaciones simples cuyas derivadas son conocidas. Esto es lo que hace posible calcular gradientes de forma automatica.
{{< /concept-alert >}}

---

## El ciclo de entrenamiento

Antes de profundizar en los grafos, es importante entender el contexto en el que se utilizan. El entrenamiento de un modelo de deep learning sigue tres pasos que se repiten en cada iteracion:

1. **Forward pass** -- se pasa un input a traves del modelo para obtener una prediccion $\hat{y}$
2. **Calculo del error** -- se compara la prediccion con la etiqueta real $y$ mediante una funcion de perdida $E(\vec{w})$
3. **Backward pass** -- se calcula el gradiente del error respecto a cada peso $w_i$ y se actualizan los pesos

La actualizacion de pesos sigue la **regla delta**:

$$\Delta w = -\eta \cdot \frac{\partial E(w)}{\partial w}$$

$$w \leftarrow w + \Delta w$$

donde $\eta$ es el *learning rate* (tasa de aprendizaje). Los pesos se actualizan segun el error cometido, y para eso necesitamos calcular el gradiente del error respecto a cada peso.

---

## Forward pass: calculando la salida

En el forward pass, los datos fluyen desde las entradas hacia la salida del modelo. Cada nodo del grafo recibe sus entradas, aplica su operacion y pasa el resultado al siguiente nodo.

Consideremos un perceptron con funcion de error cuadratico:

$$\hat{y} = f\left(\sum_{i=1}^{n} w_i \cdot x_i + w_0\right)$$

$$E(\vec{w}) = \frac{1}{2} \cdot (y - \hat{y})^2$$

El forward pass consiste en:

1. Calcular cada producto $w_i \cdot x_i$
2. Sumar todos los productos y el bias: $z = \sum w_i \cdot x_i + b$
3. Aplicar la funcion de activacion: $\hat{y} = f(z)$
4. Calcular el error: $E = \frac{1}{2}(y - \hat{y})^2$

Cada una de estas operaciones corresponde a un nodo en el grafo de computo.

---

## Backward pass: regla de la cadena

El backward pass es donde los grafos de computo muestran su verdadero poder. El objetivo es calcular $\frac{\partial E(\vec{w})}{\partial w_i}$ -- como cambia el error cuando cambia un peso especifico.

### Descomposicion en variables auxiliares

La clave esta en descomponer la funcion compuesta en variables auxiliares con derivadas conocidas:

$$l(w_i) = w_i \cdot x_i \quad \Rightarrow \quad \frac{\partial l}{\partial w_i} = x_i$$

$$z(l) = \sum l(w_i) \quad \Rightarrow \quad \frac{\partial z}{\partial l} = 1$$

$$\hat{y}(z) = z + b \quad \Rightarrow \quad \frac{\partial \hat{y}}{\partial z} = 1$$

$$u(\hat{y}) = y - \hat{y} \quad \Rightarrow \quad \frac{\partial u}{\partial \hat{y}} = -1$$

$$E(u) = u^2 \quad \Rightarrow \quad \frac{\partial E}{\partial u} = 2u$$

La funcion original queda expresada como una composicion:

$$E(\vec{w}) = E\big(u\big(\hat{y}\big(z\big(l(\vec{w})\big)\big)\big)\big)$$

### Aplicacion de la regla de la cadena

Gracias a esta descomposicion, aplicamos la **regla de la cadena** para obtener el gradiente completo:

$$\frac{\partial E(\vec{w})}{\partial w_i} = \frac{\partial l}{\partial w_i} \cdot \frac{\partial z}{\partial l} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial u}{\partial \hat{y}} \cdot \frac{\partial E}{\partial u}$$

{{< concept-alert type="clave" >}}
En el grafo de computo, para propagar el gradiente necesitamos: (1) calcular las derivadas locales en cada nodo, y (2) multiplicar las derivadas locales de derecha a izquierda (backward) hasta alcanzar el peso $w_i$ en cuestion.
{{< /concept-alert >}}

Cada nodo del grafo solo necesita conocer su **derivada local** -- la derivada de su operacion respecto a su entrada. El gradiente total se obtiene multiplicando todas las derivadas locales a lo largo del camino desde la salida hasta el peso de interes.

---

## Backpropagation

El algoritmo de backpropagation formaliza este proceso para redes de multiples capas:

- Aplica la **regla de la cadena de forma recursiva**
- Puede usarse en redes de tamano arbitrario
- Funciona con cualquier tipo de red y funcion diferenciable
- El rendimiento (convergencia) no esta garantizado

Para un perceptron multicapa (MLP), las capas intermedias (*hidden layers*) no tienen acceso directo a la etiqueta real $y$. Backpropagation resuelve este problema propagando el error desde la capa de salida hacia atras, capa por capa.

### Ejemplo numerico

Consideremos $g = (x + y) \cdot z$ con $x = 2$, $y = 3$, $z = -4$.

**Forward:**

$$a = x + y = 2 + 3 = 5$$
$$g = a \cdot z = 5 \cdot (-4) = -20$$

**Backward** (derivadas locales):

$$\frac{\partial g}{\partial a} = z = -4, \quad \frac{\partial g}{\partial z} = a = 5$$

$$\frac{\partial a}{\partial x} = 1, \quad \frac{\partial a}{\partial y} = 1$$

**Gradientes finales** (regla de la cadena):

$$\frac{\partial g}{\partial x} = \frac{\partial a}{\partial x} \cdot \frac{\partial g}{\partial a} = 1 \cdot (-4) = -4$$

$$\frac{\partial g}{\partial y} = \frac{\partial a}{\partial y} \cdot \frac{\partial g}{\partial a} = 1 \cdot (-4) = -4$$

$$\frac{\partial g}{\partial z} = a = 5$$

---

## PyTorch autograd

PyTorch implementa grafos de computo de forma dinamica a traves de su sistema **autograd**. Cada vez que se realiza una operacion con tensores que tienen `requires_grad=True`, PyTorch construye el grafo automaticamente.

### Conceptos fundamentales

| Concepto | Descripcion |
|----------|-------------|
| `requires_grad=True` | Indica que un tensor debe rastrear operaciones para calcular gradientes |
| `.backward()` | Inicia la retropropagacion desde un tensor escalar |
| `.grad` | Almacena el gradiente calculado respecto a ese tensor |
| `.grad_fn` | Referencia a la operacion que creo el tensor (nodo del grafo) |

### Ejemplo basico

```python
import torch

# Definir entrada con rastreo de gradientes
x = torch.tensor([2.0], requires_grad=True)

# Forward pass: operaciones construyen el grafo
y = x ** 2 + 3 * x + 1  # y = x^2 + 3x + 1

# Backward pass: calcular dy/dx
y.backward()

# dy/dx = 2x + 3 = 2(2) + 3 = 7
print(x.grad)  # tensor([7.])
```

### Ejemplo con multiples operaciones

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)

# Forward
z = (x * w).sum()  # producto punto
loss = (z - 1.0) ** 2  # error cuadratico

# Backward
loss.backward()

print(f"dL/dx = {x.grad}")  # gradientes respecto a x
print(f"dL/dw = {w.grad}")  # gradientes respecto a w
```

{{< concept-alert type="clave" >}}
En PyTorch, el grafo de computo se construye dinamicamente durante el forward pass (*define-by-run*). Esto permite usar condicionales y bucles normales de Python dentro del modelo, a diferencia de frameworks con grafos estaticos.
{{< /concept-alert >}}

### zero_grad y acumulacion de gradientes

Un detalle importante: PyTorch **acumula** gradientes por defecto. Si se llama `.backward()` multiples veces sin limpiar, los gradientes se suman. Por eso es necesario llamar a `optimizer.zero_grad()` antes de cada paso de backpropagation:

```python
optimizer.zero_grad()   # limpiar gradientes anteriores
loss.backward()         # calcular nuevos gradientes
optimizer.step()        # actualizar pesos
```

---

## Resumen

El proceso completo de entrenamiento, visto a traves de grafos de computo:

1. **Forward:** los datos fluyen por el grafo, cada nodo calcula su salida y almacena la informacion necesaria para calcular derivadas
2. **Loss:** se calcula el error entre la prediccion y la etiqueta
3. **Backward:** los gradientes fluyen en sentido inverso, multiplicando derivadas locales (regla de la cadena)
4. **Update:** los pesos se ajustan en la direccion que reduce el error

Este mecanismo es la base de todo el aprendizaje en deep learning, y PyTorch lo implementa de forma transparente a traves de `autograd`.
