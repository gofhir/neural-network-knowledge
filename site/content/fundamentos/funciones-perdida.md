---
title: "Funciones de Perdida"
weight: 30
math: true
---

La funcion de perdida es el corazon del aprendizaje: define **que significa equivocarse** y, por lo tanto, que debe optimizar la red. Sin una buena funcion de perdida, ningun optimizador puede ayudar.

---

## 1. Jerarquia: Loss vs Cost vs Objective

Es importante distinguir tres niveles:

| Termino | Alcance | Formula |
|---|---|---|
| **Loss Function** $L$ | Un solo ejemplo | $L(y_i, \hat{y}_i)$ |
| **Cost Function** $J$ | Agregado sobre el dataset | $J(\theta) = \frac{1}{N} \sum_i L(y_i, \hat{y}_i)$ |
| **Objective Function** $F$ | Cost + Regularizacion | $F(\theta) = J(\theta) + \lambda R(\theta)$ |

{{< concept-alert type="clave" >}}
La **funcion objetivo** es lo que realmente se minimiza durante el entrenamiento. Incluye tanto el error de prediccion (cost) como la penalizacion por complejidad (regularizacion). Minimizar solo el cost lleva a overfitting.
{{< /concept-alert >}}

El principio fundamental es la **Minimizacion del Riesgo Empirico (ERM)**:

{{< math-formula title="Minimizacion del riesgo empirico" >}}
\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i; \theta))
{{< /math-formula >}}

---

## 2. Funciones de Perdida para Regresion

### MSE (Mean Squared Error)

$$L(y, \hat{y}) = (y - \hat{y})^2, \quad J(\theta) = \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2$$

- Penaliza fuertemente los errores grandes (crecimiento cuadratico) -- sensible a outliers
- Derivable desde Maximum Likelihood asumiendo ruido Gaussiano
- El minimizador es la **media condicional** $\mathbb{E}[y|x]$

### MAE (Mean Absolute Error)

$$L(y, \hat{y}) = |y - \hat{y}|$$

- Robusta a outliers (penalidad lineal), pero no diferenciable en $y = \hat{y}$
- El minimizador es la **mediana condicional** de $y|x$

### Huber Loss

$$L_\delta = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{si } |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{si } |y - \hat{y}| > \delta \end{cases}$$

Lo mejor de ambos mundos: cuadratica para errores pequenos, lineal para grandes.

| Propiedad | MSE (L2) | MAE (L1) | Huber |
|---|---|---|---|
| Sensibilidad a outliers | Alta | Baja | Baja |
| Diferenciable | Si | No en 0 | Si |
| Optimo que busca | Media | Mediana | Depende de $\delta$ |

### Ejemplo de codigo: MSE, MAE y Huber Loss

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torch.nn as nn

# Datos de ejemplo: predicciones y valores reales
y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])

# MSE - penaliza fuertemente errores grandes
mse = nn.MSELoss()
print(f"MSE: {mse(y_pred, y_true):.4f}")  # 0.1475

# MAE - robusta a outliers
mae = nn.L1Loss()
print(f"MAE: {mae(y_pred, y_true):.4f}")  # 0.3250

# Huber Loss - combinacion de MSE y MAE (delta=1.0 por defecto)
huber = nn.SmoothL1Loss()
print(f"Huber: {huber(y_pred, y_true):.4f}")  # 0.1138
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

# Datos de ejemplo: predicciones y valores reales
y_true = tf.constant([3.0, -0.5, 2.0, 7.0])
y_pred = tf.constant([2.5, 0.0, 2.1, 7.8])

# MSE - penaliza fuertemente errores grandes
mse = tf.keras.losses.MeanSquaredError()
print(f"MSE: {mse(y_true, y_pred).numpy():.4f}")

# MAE - robusta a outliers
mae = tf.keras.losses.MeanAbsoluteError()
print(f"MAE: {mae(y_true, y_pred).numpy():.4f}")

# Huber Loss - combinacion de MSE y MAE (delta=1.0 por defecto)
huber = tf.keras.losses.Huber(delta=1.0)
print(f"Huber: {huber(y_true, y_pred).numpy():.4f}")
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax.numpy as jnp
import optax

# Datos de ejemplo: predicciones y valores reales
y_true = jnp.array([3.0, -0.5, 2.0, 7.0])
y_pred = jnp.array([2.5, 0.0, 2.1, 7.8])

# MSE - penaliza fuertemente errores grandes
mse = jnp.mean((y_pred - y_true) ** 2)
print(f"MSE: {mse:.4f}")

# MAE - robusta a outliers
mae = jnp.mean(jnp.abs(y_pred - y_true))
print(f"MAE: {mae:.4f}")

# Huber Loss - combinacion de MSE y MAE
huber = optax.huber_loss(y_pred, y_true, delta=1.0).mean()
print(f"Huber: {huber:.4f}")
```
{{< /tab >}}
{{< /tabs >}}

---

## 3. Funciones de Perdida para Clasificacion

### Binary Cross-Entropy (BCE)

Para clasificacion binaria con $y \in \{0, 1\}$ y $\hat{y} = \sigma(z)$:

$$L(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

### Categorical Cross-Entropy (CCE)

Para clasificacion multiclase con $C$ clases y $y$ one-hot:

$$L(y, \hat{y}) = -\sum_{c=1}^{C} y_c \log(\hat{y}_c) = -\log(\hat{y}_k) \quad \text{donde } k \text{ es la clase verdadera}$$

### Tabla de seleccion practica

| Tipo de problema | Funcion de perdida | PyTorch |
|---|---|---|
| Clasificacion ($N$ clases) | Cross-Entropy | `nn.CrossEntropyLoss` |
| Clasificacion binaria | BCE | `nn.BCEWithLogitsLoss` |
| Regresion | MSE | `nn.MSELoss` |
| Regresion robusta | Huber / MAE | `nn.SmoothL1Loss` / `nn.L1Loss` |

### Ejemplo de codigo: Binary y Categorical Cross-Entropy

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torch.nn as nn

# --- Binary Cross-Entropy ---
# Logits (sin sigmoid) y etiquetas binarias
logits_bin = torch.tensor([0.5, -1.2, 2.0, -0.3])
y_bin = torch.tensor([1.0, 0.0, 1.0, 0.0])

bce = nn.BCEWithLogitsLoss()  # aplica sigmoid internamente
print(f"BCE: {bce(logits_bin, y_bin):.4f}")

# --- Categorical Cross-Entropy ---
# Logits (sin softmax) para 3 clases, batch de 2 ejemplos
logits_cat = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
y_cat = torch.tensor([0, 1])  # indices de clase (no one-hot)

ce = nn.CrossEntropyLoss()  # aplica softmax internamente
print(f"CE: {ce(logits_cat, y_cat):.4f}")
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

# --- Binary Cross-Entropy ---
# Logits (sin sigmoid) y etiquetas binarias
logits_bin = tf.constant([0.5, -1.2, 2.0, -0.3])
y_bin = tf.constant([1.0, 0.0, 1.0, 0.0])

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
print(f"BCE: {bce(y_bin, logits_bin).numpy():.4f}")

# --- Categorical Cross-Entropy ---
# Logits (sin softmax) para 3 clases, batch de 2 ejemplos
logits_cat = tf.constant([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
y_cat = tf.constant([0, 1])  # indices de clase

ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(f"CE: {ce(y_cat, logits_cat).numpy():.4f}")
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax.numpy as jnp
import optax

# --- Binary Cross-Entropy ---
# Logits (sin sigmoid) y etiquetas binarias
logits_bin = jnp.array([0.5, -1.2, 2.0, -0.3])
y_bin = jnp.array([1.0, 0.0, 1.0, 0.0])

bce = optax.sigmoid_binary_cross_entropy(logits_bin, y_bin).mean()
print(f"BCE: {bce:.4f}")

# --- Categorical Cross-Entropy ---
# Logits (sin softmax) para 3 clases, batch de 2 ejemplos
logits_cat = jnp.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
y_cat = jnp.array([0, 1])  # indices de clase

# Convertir a one-hot y calcular cross-entropy
y_onehot = jax.nn.one_hot(y_cat, num_classes=3)
ce = optax.softmax_cross_entropy(logits_cat, y_onehot).mean()
print(f"CE: {ce:.4f}")
```
{{< /tab >}}
{{< /tabs >}}

---

## 4. Softmax + Cross-Entropy: La Combinacion Canonica

La funcion **Softmax** convierte logits en probabilidades:

$$\hat{y}_c = \text{softmax}(z)_c = \frac{\exp(z_c)}{\sum_{j=1}^{C} \exp(z_j)}$$

{{< concept-alert type="clave" >}}
En PyTorch, `nn.CrossEntropyLoss` aplica Softmax + Cross-Entropy internamente. **No aplicar softmax antes**, o la red recibira doble softmax y el entrenamiento fallara.
{{< /concept-alert >}}

### La derivacion clave: el gradiente se simplifica

Al combinar Softmax con Cross-Entropy, el gradiente respecto a los logits se reduce a:

{{< math-formula title="Gradiente Softmax + Cross-Entropy" >}}
\frac{\partial L}{\partial z} = \hat{y} - y
{{< /math-formula >}}

### Ejemplo numerico

Logits: $z = [2.0, 1.0, 0.1]$. Clase verdadera: $k = 0$.

```text
Paso 1 - Softmax:
  exp(z) = [7.389, 2.718, 1.105],  suma = 11.212
  y_hat  = [0.659, 0.242, 0.099]

Paso 2 - Cross-Entropy Loss:
  L = -log(0.659) = 0.417

Paso 3 - Gradiente:
  dL/dz = y_hat - y = [0.659, 0.242, 0.099] - [1, 0, 0]
        = [-0.341, 0.242, 0.099]
```

Esta simplicidad ($\hat{y} - y$) es una razon fundamental por la cual la combinacion Softmax + CE (y analogamente Sigmoid + BCE) son las elecciones canonicas para clasificacion.

### Ejemplo de codigo: Softmax + Cross-Entropy y su gradiente simplificado

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torch.nn as nn

# Logits y clase verdadera
logits = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
y = torch.tensor([0])  # clase 0

# CrossEntropyLoss combina softmax + CE internamente
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, y)
loss.backward()

# Verificar que el gradiente es (y_hat - y)
y_hat = torch.softmax(logits, dim=1)
y_onehot = torch.zeros_like(y_hat).scatter_(1, y.unsqueeze(1), 1.0)
gradiente_esperado = y_hat - y_onehot

print(f"Loss: {loss.item():.4f}")
print(f"Gradiente automatico:  {logits.grad.data}")
print(f"Gradiente manual (ŷ-y): {gradiente_esperado.data}")
# Ambos gradientes son identicos: [-0.341, 0.242, 0.099]
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

# Logits y clase verdadera
logits = tf.Variable([[2.0, 1.0, 0.1]])
y = tf.constant([0])  # clase 0

# GradientTape para calcular gradientes automaticamente
with tf.GradientTape() as tape:
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y, logits, from_logits=True
    )

# Gradiente automatico via backprop
grad_auto = tape.gradient(loss, logits)

# Verificar que el gradiente es (y_hat - y)
y_hat = tf.nn.softmax(logits)
y_onehot = tf.one_hot(y, depth=3)
grad_manual = y_hat - y_onehot

print(f"Loss: {loss.numpy()[0]:.4f}")
print(f"Gradiente automatico:  {grad_auto.numpy()}")
print(f"Gradiente manual (ŷ-y): {grad_manual.numpy()}")
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax
import jax.numpy as jnp
import optax

# Logits y clase verdadera
logits = jnp.array([2.0, 1.0, 0.1])
y_onehot = jnp.array([1.0, 0.0, 0.0])  # clase 0

# Definir funcion de perdida y calcular gradiente con jax.grad
def loss_fn(z):
    return optax.softmax_cross_entropy(z, y_onehot)

loss, grad_auto = jax.value_and_grad(loss_fn)(logits)

# Verificar que el gradiente es (y_hat - y)
y_hat = jax.nn.softmax(logits)
grad_manual = y_hat - y_onehot

print(f"Loss: {loss:.4f}")
print(f"Gradiente automatico:  {grad_auto}")
print(f"Gradiente manual (ŷ-y): {grad_manual}")
```
{{< /tab >}}
{{< /tabs >}}

---

## 5. Propiedades de una Buena Funcion de Perdida

- **Diferenciabilidad**: requerida para backpropagation
- **Acotacion inferior**: $L \geq 0$ siempre
- **Calibracion**: Cross-entropy es una *proper scoring rule* -- si la red predice la probabilidad correcta, obtiene la menor perdida posible
- **Convexidad** (deseable): garantiza que todo minimo local es global para modelos lineales

---

## Para Profundizar

- [Clase 08 - Funciones de Perdida y Regularizacion](/clases/clase-08/) -- MSE, Cross-Entropy, tareas auxiliares
- [Clase 10 - Profundizacion, Parte I](/clases/clase-10/profundizacion/) -- Derivaciones formales, Huber loss, interpretacion Bayesiana
