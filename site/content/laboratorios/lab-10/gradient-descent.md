---
title: "Gradient Descent — Fundamentos"
weight: 10
math: true
---

## El problema

Un modelo de machine learning necesita aprender los parametros correctos a partir de datos. Este laboratorio muestra como funciona ese proceso usando dos funciones simples como ejemplo.

---

## Las funciones objetivo

El lab define dos funciones que el modelo debe aprender a aproximar:

**Funcion lineal:**

$$Y = 4 + 3 \cdot x$$

**Funcion cuadratica:**

$$Y = 4 + 3 \cdot x - 2 \cdot x^2$$

```python
def getTargetLinear(x, noise=False):
    y = 4 + 3 * x
    if noise:
        return y + np.random.randn(y.size)
    return y

def getTargetCuadratic(x, noise=False):
    y = 4 + 3 * x - 2 * x**2
    if noise:
        return y + np.random.randn(y.size) * 0.5
    return y
```

Nosotros sabemos que los coeficientes son `4, 3` (lineal) y `4, 3, -2` (cuadratica). El modelo **no los sabe** — tiene que descubrirlos a partir de los datos.

El parametro `noise=True` agrega ruido gaussiano a los datos para simular la realidad, donde los datos nunca se ajustan perfectamente a una ecuacion.

---

## El modelo

Para aproximar cada funcion, el modelo propone una hipotesis parametrizada por $\theta$:

**Para la lineal:**

$$\hat{Y} = h(\theta, x) = \theta_0 + \theta_1 \cdot x$$

**Para la cuadratica:**

$$\hat{Y} = h(\theta, x) = \theta_0 + \theta_1 \cdot x + \theta_2 \cdot x^2$$

Los valores $\theta_i$ son los **parametros que el modelo debe aprender**. Empiezan con valores aleatorios y se van ajustando iterativamente. Si el modelo aprende correctamente, al final deberian quedar cerca de los coeficientes reales:

| Funcion | $\theta_0$ | $\theta_1$ | $\theta_2$ |
|---------|-----------|-----------|-----------|
| Lineal (real) | 4 | 3 | — |
| Cuadratica (real) | 4 | 3 | -2 |

---

## La funcion de perdida (loss)

Para saber si el modelo va bien o mal se necesita una metrica de error. Se usa el **Mean Squared Error (MSE)**:

$$L(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h(\theta, x_i) - y_i)^2$$

```python
def cal_loss(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)  # Ŷ = X · θ
    cost = (1/2*m) * np.sum(np.square(predictions - y))
    return cost
```

La logica: toma la diferencia entre prediccion y valor real, la eleva al cuadrado y promedia. Si el modelo es perfecto, el loss es 0. Mientras mas lejos este la prediccion, mas alto el loss.

---

## El gradiente — la direccion para mejorar

El modelo tiene un error. La pregunta es: hacia donde debe mover los $\theta$ para reducirlo?

La **derivada parcial** del loss respecto a cada $\theta_j$ da la pendiente — la direccion en que el error crece mas rapido:

$$\frac{\partial L(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h(\theta, x_i) - y_i) \cdot X_i^j$$

El modelo se mueve en **direccion contraria** (por eso se resta), multiplicado por el learning rate $\alpha$:

$$\theta_j = \theta_j - \alpha \cdot \frac{\partial L(\theta)}{\partial \theta_j}$$

---

## La funcion gradient_descent

```python
def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, theta.size))

    for it in range(iterations):
        prediction = np.dot(X, theta)                                    # 1. predecir
        theta = theta - learning_rate * (1/m) * (X.T.dot(prediction - y))  # 2. actualizar pesos
        theta_history[it,:] = theta.T
        cost_history[it] = cal_loss(theta, X, y)                          # 3. registrar error

    return theta, cost_history, theta_history
```

En cada iteracion:

1. **Predice** con los $\theta$ actuales
2. **Calcula el gradiente** y **actualiza** $\theta$ en direccion contraria
3. **Registra el error** para monitorear la convergencia

Despues de muchas iteraciones, el error va bajando y los $\theta$ se acercan a los valores reales.

---

## Efecto del learning rate

El lab prueba 4 combinaciones de learning rate e iteraciones:

| Learning rate | Iteraciones | Comportamiento |
|---------------|-------------|----------------|
| 0.001 | 2000 | Pasos muy chicos, necesita muchas iteraciones |
| 0.01 | 500 | Equilibrio razonable |
| 0.05 | 200 | Pasos mas grandes, converge rapido |
| 0.1 | 100 | Pasos grandes, converge muy rapido pero puede ser inestable |

En los graficos del lab se ve como la linea roja (prediccion del modelo) se va acercando a los puntos azules (datos reales). Con un learning rate alto llega rapido; con uno bajo se mueve despacio pero con mas precision.

{{< concept-alert type="clave" >}}
El learning rate es un **tradeoff entre velocidad y estabilidad**: muy alto puede diverger (el error sube en vez de bajar), muy bajo converge pero puede tardar demasiado.
{{< /concept-alert >}}

---

## Metricas durante el entrenamiento

Es importante monitorear como evoluciona el loss durante el entrenamiento para detectar problemas:

| Comportamiento del loss | Diagnostico |
|------------------------|-------------|
| Nunca baja | Problema grave — los pesos no se actualizan correctamente |
| Baja pero sube de nuevo | Learning rate demasiado alto |
| Baja muy lento | Learning rate demasiado bajo |
| Baja constantemente | Entrenamiento correcto |
