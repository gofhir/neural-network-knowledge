---
title: "Backpropagation"
weight: 40
math: true
---

Backpropagation es el algoritmo que hace posible el entrenamiento de redes profundas. Proporciona un metodo eficiente para calcular el gradiente de la funcion de perdida respecto a **cada peso** de la red, sin importar su profundidad, mediante la aplicacion sistematica de la **regla de la cadena**.

> **Rumelhart, Hinton & Williams (1986).** *"Learning representations by back-propagating errors."* Nature, 323, 533-536.

---

## 1. La Regla de la Cadena

Si $y = f(g(x))$, donde $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Para funciones multivariadas:

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

{{< concept-alert type="clave" >}}
**Esta es la base matematica de backpropagation**: la perdida depende de las salidas, que dependen de las activaciones ocultas, que dependen de los pesos. La regla de la cadena permite calcular como un cambio en cualquier peso afecta la perdida final.
{{< /concept-alert >}}

### Ejemplo concreto

Sea $L = (y - t)^2$, $y = \sigma(z)$, $z = wx + b$:

$$\frac{\partial L}{\partial w} = \underbrace{2(y - t)}_{\partial L / \partial y} \cdot \underbrace{\sigma(z)(1 - \sigma(z))}_{\partial y / \partial z} \cdot \underbrace{x}_{\partial z / \partial w}$$

---

## 2. Grafos Computacionales

Una red neuronal se representa como un **Grafo Aciclico Dirigido (DAG)**:

- **Nodos** = operaciones o variables
- **Aristas** = flujo de datos

El forward pass evalua nodos en orden topologico. El backward pass los recorre en orden inverso, acumulando gradientes. Cada nodo solo necesita conocer su **derivada local**.

---

## 3. Ejemplo Numerico Completo

Red de 2 capas: entrada $x_1 = 0.5, x_2 = 0.3$, capa oculta con 2 neuronas (sigmoid), salida con 1 neurona (sigmoid), target $t = 1.0$.

### Forward Pass

```text
Pesos capa oculta:  W^(1) = [[0.15, 0.20], [0.25, 0.30]], b^(1) = [0.35, 0.35]
Pesos capa salida:  W^(2) = [0.40, 0.45], b^(2) = 0.60

Pre-activacion oculta:
  z_1 = 0.15*0.5 + 0.20*0.3 + 0.35 = 0.485
  z_2 = 0.25*0.5 + 0.30*0.3 + 0.35 = 0.565

Activacion sigmoid:
  h_1 = sigma(0.485) = 0.6190
  h_2 = sigma(0.565) = 0.6376

Pre-activacion salida:
  z = 0.40*0.6190 + 0.45*0.6376 + 0.60 = 1.1345

Salida:
  y = sigma(1.1345) = 0.7565

Loss:
  L = (1/2)(0.7565 - 1.0)^2 = 0.02966
```

### Backward Pass

```text
Paso 1: dL/dy = y - t = -0.2435

Paso 2: dy/dz = y(1-y) = 0.1842
         delta^(2) = (-0.2435)(0.1842) = -0.04484

Paso 3 - Gradientes capa salida:
         dL/dw_31 = delta^(2) * h_1 = -0.02776
         dL/dw_32 = delta^(2) * h_2 = -0.02859

Paso 4 - Propagar error hacia atras:
         dL/dh_1 = delta^(2) * w_31 = -0.01794
         dL/dh_2 = delta^(2) * w_32 = -0.02018

Paso 5 - Deltas capa oculta:
         delta_1^(1) = -0.01794 * 0.6190 * 0.3810 = -0.004231
         delta_2^(1) = -0.02018 * 0.6376 * 0.3624 = -0.004662

Paso 6 - Gradientes capa oculta:
         dL/dw_11 = delta_1^(1) * x_1 = -0.002116
         dL/dw_12 = delta_1^(1) * x_2 = -0.001269
         dL/dw_21 = delta_2^(1) * x_1 = -0.002331
         dL/dw_22 = delta_2^(1) * x_2 = -0.001399
```

{{< concept-alert type="clave" >}}
**Observacion critica**: Los gradientes de la capa oculta (~0.002) son ~10x mas pequenos que los de la capa de salida (~0.028). Con solo 2 capas ya se observa la atenuacion. Esto es el inicio del problema de **vanishing gradients**.
{{< /concept-alert >}}

---

## 4. Patron General

Para una red de $L$ capas, el gradiente de los pesos de la capa $l$ es:

{{< math-formula title="Gradiente por capa" >}}
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \cdot \left(a^{(l-1)}\right)^T
{{< /math-formula >}}

Donde la senal de error se propaga recursivamente:

$$\delta^{(l)} = \left( \left(W^{(l+1)}\right)^T \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})$$

El gradiente completo es un **producto de Jacobianos**:

$$\frac{\partial L}{\partial x} = J_{f_1}^T \cdot J_{f_2}^T \cdots J_{f_K}^T \cdot \frac{\partial L}{\partial y}$$

---

## 5. Pseudocodigo

```text
ALGORITMO: Backpropagation
==========================

--- FORWARD PASS ---
1.  a^(0) = x
2.  PARA l = 1 HASTA L:
3.      z^(l) = W^(l) * a^(l-1) + b^(l)
4.      a^(l) = sigma(z^(l))
5.  y = a^(L)
6.  L = Loss(y, t)

--- BACKWARD PASS ---
7.  delta^(L) = dLoss/dy . sigma'(z^(L))
8.  PARA l = L-1 BAJANDO HASTA 1:
9.      delta^(l) = ( (W^(l+1))^T * delta^(l+1) ) . sigma'(z^(l))

--- ACTUALIZAR PARAMETROS ---
10. PARA l = 1 HASTA L:
11.     W^(l) = W^(l) - eta * delta^(l) * (a^(l-1))^T
12.     b^(l) = b^(l) - eta * delta^(l)
```

---

## 6. Vanishing y Exploding Gradients

### El problema con Sigmoid

El valor maximo de $\sigma'(z)$ es **0.25**. Despues de $K$ capas:

$$\|\text{gradiente capa 1}\| \leq (0.25)^K \cdot \|\text{gradiente salida}\|$$

| Capas desde la salida | Magnitud del gradiente |
|---|---|
| 0 (salida) | 1.0 |
| 2 | 0.0625 |
| 5 | 0.000977 |
| 10 | **0.00000095** |

Con 10 capas sigmoid, el gradiente se reduce un millon de veces.

### Soluciones

| Problema | Solucion | Mecanismo |
|---|---|---|
| Vanishing | **ReLU** | Derivada = 1 para $z > 0$ |
| Vanishing | **Skip connections** (ResNets) | $\frac{\partial}{\partial x}(x + F(x)) = 1 + F'(x)$ |
| Vanishing | **LSTM / GRU gates** | Preserva gradientes en el tiempo |
| Exploding | **Gradient clipping** | Recorta la norma del gradiente |
| Ambos | **Xavier / He init** | Preserva varianza entre capas |
| Ambos | **Batch Normalization** | Re-centra y re-escala activaciones |

{{< math-formula title="Skip connection - por que funciona" >}}
\frac{\partial}{\partial x}(x + F(x)) = I + \frac{\partial F}{\partial x}
{{< /math-formula >}}

El termino $I$ (identidad) garantiza que el gradiente siempre fluye, independientemente de $F'(x)$.

---

## 7. Diferenciacion Automatica

Backpropagation es un caso especifico de **diferenciacion automatica en modo reverso** (reverse-mode AD). Los frameworks modernos lo implementan automaticamente:

| Propiedad | Forward Mode | Reverse Mode (Backprop) |
|---|---|---|
| Direccion | Input $\to$ Output | Output $\to$ Input |
| Un pase calcula | $\partial(\text{outputs})/\partial(\text{un input})$ | $\partial(\text{un output})/\partial(\text{todos inputs})$ |
| Para redes neuronales | Inviable | **Ideal** (un solo output: la loss) |
| Costo en memoria | Bajo | Alto (almacena activaciones) |

El **trade-off de memoria** del modo reverso motiva tecnicas como **gradient checkpointing**, que sacrifica computo adicional para reducir el consumo de memoria.

---

## Para Profundizar

- [Clase 10 - Profundizacion, Parte II](/clases/clase-10/profundizacion/) -- Derivacion completa, Jacobianos, AD
- [Clase 10 - Historia Matematica](/clases/clase-10/historia-matematica/) -- Rumelhart 1986, Werbos 1974
