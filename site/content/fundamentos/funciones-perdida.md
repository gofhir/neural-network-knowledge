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
