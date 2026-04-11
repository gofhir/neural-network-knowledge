---
title: "Teoria - Grafos de Computo, Activaciones e Inicializacion"
weight: 10
math: true
---

## 1. Algoritmo de Aprendizaje y Grafos de Computo

### 1.1 Pasos de Entrenamiento

El entrenamiento de una red neuronal consta de 3 pasos fundamentales que se repiten ciclicamente:

1. **Forward (Propagacion hacia adelante):** Los datos de entrada pasan a traves del modelo, que genera una prediccion.
2. **Backward (Retropropagacion):** Se compara la prediccion con la etiqueta real mediante una funcion de perdida, y se propaga el error hacia atras para calcular los gradientes.
3. **Weights Update (Actualizacion de pesos):** Los pesos del modelo se actualizan segun el error cometido.

### 1.2 Descenso del Gradiente (Regla Delta)

{{< math-formula title="Regla Delta" >}}
\Delta w = -\eta \cdot \frac{\partial E(w)}{\partial w}, \quad w \leftarrow w + \Delta w
{{< /math-formula >}}

Donde:
- $E(w)$ es la funcion de error (loss)
- $\eta$ es el **learning rate**
- $\frac{\partial E(w)}{\partial w}$ es el gradiente del error respecto a los pesos

### 1.3 Entrenamiento de un Perceptron

Para un perceptron simple con funcion de activacion $f$:

$$\hat{y} = f\left(\sum_i w_i x_i + w_0\right)$$

La funcion de error (MSE) es:

$$E(w) = \frac{1}{2}(y - \hat{y})^2$$

El gradiente del error respecto a cada peso:

$$\frac{\partial E(w)}{\partial w_i} = (y - \hat{y}) \cdot f'(z) \cdot x_i$$

### 1.4 Perceptron Multicapa (MLP)

- Los MLPs combinan multiples perceptrones en varias capas (Deep Feed Forward Networks)
- Pueden aproximar cualquier funcion matematica (Teorema de Aproximacion Universal)
- El algoritmo de **Backpropagation** resuelve el problema de entrenar capas ocultas aplicando la **Regla de la Cadena** de forma recursiva

### 1.5 Grafos de Computo

Los grafos de computo son representaciones que permiten expresar y evaluar funciones matematicas. Consisten en grafos dirigidos donde los **nodos** son operaciones matematicas y las **aristas** representan el flujo de variables.

```mermaid
graph LR
    x([x]):::input --> add(("+")):::op
    y([y]):::input --> add
    add --> mul(("*")):::op
    z([z]):::input --> mul
    mul --> g([g]):::output

    classDef input fill:#2563eb,color:#fff,stroke:#1e40af
    classDef op fill:#f59e0b,color:#fff,stroke:#d97706
    classDef output fill:#059669,color:#fff,stroke:#047857
```

Los frameworks (TensorFlow, PyTorch) los utilizan para implementar backpropagation de forma automatica.

---

## 2. Funciones de Activacion

### 2.1 Funcion Sigmoide

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Derivada:** $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**Problemas:**
- Neuronas saturadas tienen gradiente cercano a cero (**vanishing gradient**)
- La salida no esta centrada en cero (produce zig-zag en la convergencia)

### 2.2 Tangente Hiperbolica (Tanh)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

- Centrada en cero (mejora sobre Sigmoide)
- Aun sufre de vanishing gradient en saturacion

### 2.3 ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

{{< concept-alert type="clave" >}}
ReLU converge mucho mas rapido que Sigmoide (hasta 6x segun Krizhevsky et al., 2012) porque su gradiente en la zona positiva es constante (= 1), mientras que el gradiente de Sigmoide es como maximo 0.25.
{{< /concept-alert >}}

**Problema:** Se satura para entradas negativas (gradiente = 0, la neurona "muere").

### 2.4 Leaky ReLU / PReLU

$$\text{PReLU}(x) = \max(\alpha x, x)$$

Donde $\alpha$ es un parametro pequeno (ej. 0.01). Las neuronas nunca dejan de aprender.

### 2.5 Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Convierte salidas en probabilidades que suman 1. Usada en capas de salida para clasificacion multiclase.

### 2.6 Tabla Comparativa

| Funcion | Rango | Centrada en 0 | Vanishing Gradient | Uso tipico |
|---------|-------|---------------|-------------------|------------|
| Sigmoide | (0, 1) | No | Si | Salida binaria |
| Tanh | (-1, 1) | Si | Si | Capas ocultas (RNN) |
| ReLU | [0, +inf) | No | Solo negativos | Capas ocultas (CNN) |
| Leaky ReLU | (-inf, +inf) | No | No | Capas ocultas |
| Softmax | (0, 1) | N/A | N/A | Capa salida (multiclase) |

---

## 3. Inicializacion de Pesos

### 3.1 El Problema

En redes profundas existen 4 problemas criticos:

**Forward:** Vanishing input signal (pesos < 1) y Exploding input signal (pesos > 1)

**Backward:** Vanishing gradient y Exploding gradient

Para una red de $L$ capas sin activacion:

$$y = W^{[L]} \cdot W^{[L-1]} \cdots W^{[1]} \cdot x$$

Si los pesos son < 1, $y \to 0$ cuando $L$ es grande. Si son > 1, $y \to \infty$.

### 3.2 Inicializacion de Xavier Glorot

{{< concept-alert type="clave" >}}
Xavier Glorot inicializa los pesos desde una distribucion gaussiana con varianza $\text{Var}(W_i) = \frac{2}{\text{fan\_in} + \text{fan\_out}}$, donde fan_in es el numero de entradas y fan_out el numero de salidas de la capa. Esto mantiene la varianza estable a traves de las capas.
{{< /concept-alert >}}

En PyTorch:

```python
torch.nn.init.xavier_uniform_(tensor, gain=1.0)
```

---

## 4. Conceptos Clave de PyTorch

### 4.1 Tensores

| Dimensiones | Nombre | Ejemplo |
|-------------|--------|---------|
| 0 | Escalar | `5.0` |
| 1 | Vector | `[1, 2, 3]` |
| 2 | Matriz | `[[1, 2], [3, 4]]` |
| 3+ | Tensor | Imagen RGB: alto x ancho x 3 canales |

Los tensores de PyTorch tienen dos superpoderes: pueden ejecutarse en **GPU** y pueden rastrear operaciones para calcular **gradientes automaticamente**.

```python
# Tensor que rastrea gradientes
x = torch.tensor([2.0], requires_grad=True)
```

### 4.2 Optimizador Adam

Adam (Adaptive Moment Estimation) adapta el learning rate para cada peso individualmente combinando:

1. **Momentum:** Memoria de la direccion de actualizacion anterior
2. **Adaptativo:** Ajusta el paso segun la magnitud de los gradientes

$$w = w - \eta \cdot \text{gradiente}$$

```python
optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()     # Limpia gradientes
loss_value.backward()     # Calcula gradientes
optimizer.step()          # Actualiza pesos
```
