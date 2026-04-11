# Clase 10 - Profundizacion: Funciones de Perdida, Backpropagation y Learning Rate

**Diplomado Inteligencia Artificial - UC**
**Documento complementario a la Clase 10**
**Fecha:** 2026-04-10

> Este documento profundiza en los fundamentos teoricos que subyacen a los algoritmos
> de optimizacion cubiertos en la Clase 10. Cubre funciones de perdida, backpropagation
> y la teoria del learning rate con demostraciones matematicas y ejemplos numericos concretos.

---

## Tabla de Contenidos

**Parte I: Funciones de Perdida**
1. [Definicion Formal y Jerarquia](#1-definicion-formal-y-jerarquia)
2. [Funciones de Perdida para Regresion](#2-funciones-de-perdida-para-regresion)
3. [Funciones de Perdida para Clasificacion](#3-funciones-de-perdida-para-clasificacion)
4. [Softmax + Cross-Entropy: La Combinacion Canonica](#4-softmax--cross-entropy-la-combinacion-canonica)
5. [Propiedades de una Buena Funcion de Perdida](#5-propiedades-de-una-buena-funcion-de-perdida)
6. [Regularizacion: L1, L2 y Elastic Net](#6-regularizacion-l1-l2-y-elastic-net)

**Parte II: Backpropagation**
7. [Contexto Historico](#7-contexto-historico)
8. [La Regla de la Cadena (Chain Rule)](#8-la-regla-de-la-cadena-chain-rule)
9. [Grafos Computacionales](#9-grafos-computacionales)
10. [Forward Pass: Ejemplo Numerico Completo](#10-forward-pass-ejemplo-numerico-completo)
11. [Backward Pass: Derivacion Paso a Paso](#11-backward-pass-derivacion-paso-a-paso)
12. [Flujo de Gradientes y la Matriz Jacobiana](#12-flujo-de-gradientes-y-la-matriz-jacobiana)
13. [Vanishing y Exploding Gradients](#13-vanishing-y-exploding-gradients)
14. [Diferenciacion Automatica](#14-diferenciacion-automatica)
15. [Pseudocodigo Completo de Backpropagation](#15-pseudocodigo-completo-de-backpropagation)

**Parte III: Learning Rate y Paisajes de Optimizacion**
16. [El Paisaje de Optimizacion de Redes Neuronales](#16-el-paisaje-de-optimizacion-de-redes-neuronales)
17. [Teoria Formal del Learning Rate](#17-teoria-formal-del-learning-rate)
18. [Estrategias de Learning Rate Scheduling](#18-estrategias-de-learning-rate-scheduling)
19. [Inicializacion de Pesos y su Relacion con el Learning Rate](#19-inicializacion-de-pesos-y-su-relacion-con-el-learning-rate)
20. [Batch Size y Learning Rate](#20-batch-size-y-learning-rate)
21. [Papers Fundamentales Referenciados](#21-papers-fundamentales-referenciados)

---

# PARTE I: FUNCIONES DE PERDIDA

---

## 1. Definicion Formal y Jerarquia

### Definicion matematica

Una **funcion de perdida** es un mapeo:

```
L : Y x Y --> R>=0
```

Dado un valor real `y` y una prediccion `y_hat = f(x; theta)` producida por un modelo `f` con parametros `theta`, la funcion `L(y, y_hat)` cuantifica la discrepancia entre ambos.

El objetivo del entrenamiento es encontrar los parametros `theta*` que minimicen la **perdida esperada** (riesgo) sobre la distribucion real de datos `P(x, y)`:

```
theta* = arg min_theta  E_{(x,y)~P} [ L(y, f(x; theta)) ]
```

Como `P` es desconocida, aproximamos con el **riesgo empirico** sobre N muestras de entrenamiento:

```
theta* = arg min_theta  (1/N) SUM_{i=1}^{N} L(y_i, f(x_i; theta))
```

Este es el principio de **Minimizacion del Riesgo Empirico (ERM)**.

### Jerarquia: Loss vs Cost vs Objective

Estos tres terminos estan relacionados pero son tecnicamente distintos:

```
+--------------------------------------------------+
| Funcion Objetivo  F(theta)                        |
|                                                    |
|   = Funcion de Costo J(theta) + Regularizacion    |
|                                                    |
|   = (1/N) SUM L(y_i, y_hat_i) + lambda * R(theta) |
|             ^                         ^            |
|     Funcion de Perdida          Regularizador      |
|     (un solo ejemplo)           (penalizacion)     |
+--------------------------------------------------+
```

| Termino | Alcance | Formula |
|---|---|---|
| **Loss Function** `L` | Un **solo** ejemplo | `L(y_i, y_hat_i)` |
| **Cost Function** `J` | El **agregado** sobre todo el dataset | `J(theta) = (1/N) SUM L(y_i, y_hat_i)` |
| **Objective Function** `F` | Cost + Regularizacion | `F(theta) = J(theta) + lambda * R(theta)` |

En la practica, muchos autores usan "loss function" y "cost function" de forma intercambiable.

---

## 2. Funciones de Perdida para Regresion

### 2.1 Mean Squared Error (MSE) / L2 Loss

**Perdida por ejemplo:**
```
L(y, y_hat) = (y - y_hat)^2
```

**Funcion de costo:**
```
J(theta) = (1/N) SUM_{i=1}^{N} (y_i - y_hat_i)^2
```

**Gradiente respecto a y_hat:**
```
dL/d(y_hat) = -2(y - y_hat) = 2(y_hat - y)
```

**Gradiente respecto a los parametros theta (via chain rule):**
```
dJ/d(theta) = (1/N) SUM_{i=1}^{N} 2(y_hat_i - y_i) * d(y_hat_i)/d(theta)
```

**Derivacion desde Maximum Likelihood:**

Si asumimos que el target `y` se genera a partir de `y_hat` con ruido Gaussiano aditivo:
```
y = y_hat + epsilon,    epsilon ~ N(0, sigma^2)
```

Entonces `P(y|x,theta) = (1/sqrt(2*pi*sigma^2)) * exp(-(y - y_hat)^2 / (2*sigma^2))`

Tomando el negativo del log-likelihood:
```
-log P(y|x,theta) = (y - y_hat)^2 / (2*sigma^2) + constante
```

Minimizar el NLL es equivalente a minimizar MSE. La constante `sigma^2` solo escala la perdida.

**Propiedades:**
- Penaliza fuertemente los errores grandes (crecimiento cuadratico) --> **sensible a outliers**
- Suave, diferenciable en todo punto
- Convexa respecto a `y_hat`
- El minimizador unico es la **media condicional** E[y|x]

**Cuando usar:** Eleccion por defecto para regresion. Adecuada cuando los outliers son raros o cuando los errores grandes deben penalizarse.

---

### 2.2 Mean Absolute Error (MAE) / L1 Loss

**Perdida por ejemplo:**
```
L(y, y_hat) = |y - y_hat|
```

**Gradiente:**
```
dL/d(y_hat) = -sign(y - y_hat) = { -1  si y_hat < y
                                     +1  si y_hat > y
                                     indefinido si y_hat = y }
```

**Derivacion desde Maximum Likelihood:**

MAE corresponde a asumir una **distribucion de Laplace** para el ruido:
```
P(y|x,theta) = (1/(2b)) * exp(-|y - y_hat|/b)
```

**Propiedades:**
- **Robusta a outliers** (penalidad lineal, no cuadratica)
- No diferenciable en `y = y_hat`
- El minimizador es la **mediana condicional** de `y|x`
- Gradiente constante puede causar inestabilidad cerca del minimo (oscilacion)

**Cuando usar:** Datos con outliers significativos, o cuando interesa la mediana mas que la media.

---

### 2.3 Huber Loss (Smooth L1)

**Perdida por ejemplo:**
```
L_delta(y, y_hat) = { (1/2)(y - y_hat)^2            si |y - y_hat| <= delta
                    { delta * |y - y_hat| - (1/2)*delta^2   si |y - y_hat| > delta
```

**Gradiente:**
```
dL/d(y_hat) = { (y_hat - y)                si |y - y_hat| <= delta
              { -delta * sign(y - y_hat)    si |y - y_hat| > delta
```

**Lo mejor de ambos mundos:**
- Cuadratica para errores pequenos (como MSE) --> suave cerca del optimo
- Lineal para errores grandes (como MAE) --> robusta a outliers
- Diferenciable en todo punto (incluyendo `|y - y_hat| = delta`)
- `delta -> infinito` recupera MSE; `delta -> 0` se aproxima a MAE

### Tabla comparativa de perdidas de regresion

| Propiedad | MSE (L2) | MAE (L1) | Huber |
|---|---|---|---|
| Sensibilidad a outliers | Alta (cuadratica) | Baja (lineal) | Baja (cola lineal) |
| Diferenciable | Si, en todo punto | No en 0 | Si, en todo punto |
| Optimo que busca | Media condicional | Mediana condicional | Depende de delta |
| Convexa | Si | Si | Si |

---

## 3. Funciones de Perdida para Clasificacion

### 3.1 Binary Cross-Entropy (BCE) / Log Loss

Para clasificacion binaria donde `y in {0, 1}` y `y_hat = sigma(z) in (0, 1)` donde `sigma` es la funcion sigmoid.

**Perdida por ejemplo:**
```
L(y, y_hat) = -[ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]
```

**Derivacion desde Maximum Likelihood:**

Modelamos la salida como una distribucion Bernoulli:
```
P(y|x,theta) = y_hat^y * (1 - y_hat)^(1-y)
```

El negativo del log-likelihood:
```
-log P(y|x,theta) = -[ y * log(y_hat) + (1-y) * log(1-y_hat) ]
```

Esto es exactamente la Binary Cross-Entropy.

**Gradiente respecto a y_hat:**
```
dL/d(y_hat) = -(y/y_hat) + (1-y)/(1-y_hat) = (y_hat - y) / (y_hat * (1-y_hat))
```

**Gradiente respecto al logit z (donde y_hat = sigma(z)) -- LA SIMPLIFICACION CLAVE:**

Dado que `sigma'(z) = sigma(z) * (1 - sigma(z)) = y_hat * (1 - y_hat)`:

```
dL/dz = dL/d(y_hat) * d(y_hat)/dz
      = [(y_hat - y) / (y_hat*(1-y_hat))] * y_hat*(1-y_hat)
      = y_hat - y
```

**Resultado elegante: `dL/dz = y_hat - y`**. Esta simplicidad es una razon fundamental por la cual sigmoid + BCE se usan juntos.

**Ejemplo numerico:**
- Prediccion: `y_hat = 0.8`, Real: `y = 1`
- `L = -[1 * log(0.8) + 0 * log(0.2)] = -log(0.8) = 0.223`
- `dL/dz = 0.8 - 1 = -0.2` (empujar para aumentar la probabilidad)

---

### 3.2 Categorical Cross-Entropy (CCE)

Para clasificacion multiclase con `C` clases, `y` one-hot encoded, `y_hat` es la salida de softmax.

**Perdida por ejemplo:**
```
L(y, y_hat) = -SUM_{c=1}^{C} y_c * log(y_hat_c)
```

Como `y` es one-hot (solo un `y_c = 1`), se simplifica a:
```
L(y, y_hat) = -log(y_hat_k)     donde k es la clase verdadera
```

**Conexion con teoria de la informacion:**

La cross-entropy entre distribuciones `p` (real) y `q` (predicha) es:
```
H(p, q) = -SUM_c p_c * log(q_c) = H(p) + D_KL(p || q)
```

Donde `H(p)` es la entropia de `p` y `D_KL` es la divergencia KL. Como `H(p)` es constante respecto al modelo, **minimizar cross-entropy equivale a minimizar la divergencia KL**.

---

### 3.3 Sparse Categorical Cross-Entropy

Matematicamente **identica** a Categorical CE. La diferencia es de implementacion:

| Aspecto | Categorical CE | Sparse Categorical CE |
|---|---|---|
| Formato de labels | Vector one-hot `[0,0,1,0]` | Indice entero `2` |
| Memoria por label | O(C) por ejemplo | O(1) por ejemplo |
| Formula | `-SUM_c y_c * log(y_hat_c)` | `-log(y_hat_k)` |

Sparse CE es preferida cuando el numero de clases `C` es grande (ej: vocabulario de 50,000+ palabras en NLP).

---

### 3.4 Hinge Loss (SVM)

Usada en Support Vector Machines. Requiere labels `y in {-1, +1}` y un score crudo `z` (sin activacion).

**Perdida por ejemplo:**
```
L(y, z) = max(0, 1 - y * z)
```

**Gradiente:**
```
dL/dz = { 0     si y*z >= 1   (clasificado correctamente con margen)
        { -y    si y*z < 1    (margen violado)
```

**Propiedades:**
- No produce estimaciones de probabilidad
- Solo le importa el margen de decision
- Gradientes sparse (cero una vez clasificado correctamente)
- Menos comun en deep learning moderno que cross-entropy

---

## 4. Softmax + Cross-Entropy: La Combinacion Canonica

### La funcion Softmax

Convierte un vector de logits `z = (z_1, ..., z_C)` en una distribucion de probabilidad:

```
y_hat_c = softmax(z)_c = exp(z_c) / SUM_{j=1}^{C} exp(z_j)
```

### Derivacion detallada del gradiente simplificado

Sea `k` la clase verdadera. La perdida es:
```
L = -log(y_hat_k) = -log(exp(z_k) / SUM_j exp(z_j)) = -z_k + log(SUM_j exp(z_j))
```

**Caso 1: c = k (la clase verdadera)**
```
dL/dz_k = -1 + exp(z_k) / SUM_j exp(z_j) = -1 + y_hat_k = y_hat_k - 1
```

**Caso 2: c != k (cualquier otra clase)**
```
dL/dz_c = 0 + exp(z_c) / SUM_j exp(z_j) = y_hat_c
```

**Expresion unificada usando el vector one-hot y:**
```
dL/dz_c = y_hat_c - y_c     para todo c = 1, ..., C
```

O en forma vectorial:
```
dL/dz = y_hat - y
```

### Por que esto importa

1. **Eficiencia computacional:** No necesitamos calcular el Jacobiano de softmax separadamente. El gradiente es una simple resta.
2. **Estabilidad numerica:** Softmax y log por separado pueden causar overflow/underflow. Las implementaciones combinadas usan el truco **log-sum-exp**:
   ```
   log(SUM exp(z_j)) = z_max + log(SUM exp(z_j - z_max))
   ```
3. **Sin gradientes que desaparecen por saturacion del softmax:** Incluso cuando las salidas de softmax estan cerca de 0 o 1, el gradiente `y_hat - y` se mantiene bien comportado. Comparar con MSE + softmax, donde el gradiente incluye terminos `y_hat_c * (1 - y_hat_c)` que desaparecen cerca de la saturacion.

### Ejemplo numerico completo

Red con 3 clases. Logits: `z = [2.0, 1.0, 0.1]`. Clase verdadera: `k = 0`.

**Paso 1: Softmax**
```
exp(z) = [e^2.0, e^1.0, e^0.1] = [7.389, 2.718, 1.105]
SUM = 7.389 + 2.718 + 1.105 = 11.212
y_hat = [7.389/11.212, 2.718/11.212, 1.105/11.212] = [0.659, 0.242, 0.099]
```

**Paso 2: Cross-Entropy Loss**
```
L = -log(y_hat_0) = -log(0.659) = 0.417
```

**Paso 3: Gradiente**
```
dL/dz = y_hat - y = [0.659, 0.242, 0.099] - [1, 0, 0] = [-0.341, 0.242, 0.099]
```

Interpretacion: El gradiente empuja z_0 hacia arriba (para aumentar la probabilidad de la clase correcta) y z_1, z_2 hacia abajo.

---

## 5. Propiedades de una Buena Funcion de Perdida

### 5.1 Convexidad

Una funcion `L` es **convexa** si para todo `theta_1, theta_2` y `lambda in [0, 1]`:
```
L(lambda*theta_1 + (1-lambda)*theta_2) <= lambda*L(theta_1) + (1-lambda)*L(theta_2)
```

- Garantiza que todo minimo local es global
- Para modelos lineales, MSE y cross-entropy son convexas en los parametros
- Para redes neuronales con capas ocultas, la funcion compuesta es **no-convexa** (por la composicion no-lineal)

### 5.2 Diferenciabilidad

- Requerida para optimizacion basada en gradientes (backpropagation)
- Como minimo, necesitamos **subdiferenciabilidad** (MAE, Hinge, ReLU)
- Perdidas suaves (MSE, cross-entropy) son preferidas: gradientes continuos -> optimizacion estable
- **Gradientes Lipschitz continuos** (segunda derivada acotada) permiten garantias de convergencia mas fuertes

### 5.3 Acotacion inferior

- `L >= 0` siempre (asegura que el problema tiene un minimo bien definido)

### 5.4 Consistencia con la tarea

- **Fisher consistency**: La perdida es Fisher-consistente si minimizarla sobre la distribucion real produce el clasificador Bayes-optimo
- Cross-entropy y hinge loss son Fisher-consistentes

### 5.5 Calibracion

- Una perdida que produce probabilidades **calibradas** significa: cuando el modelo predice `P(y=1) = 0.7`, ~70% de esos casos son realmente positivos
- Cross-entropy es una **proper scoring rule** (minimizada cuando las probabilidades predichas igualan a las reales)
- Hinge loss NO produce probabilidades calibradas

---

## 6. Regularizacion: L1, L2 y Elastic Net

La funcion objetivo total es:
```
F(theta) = (1/N) SUM_i L(y_i, f(x_i; theta)) + lambda * R(theta)
```

### 6.1 L2 Regularizacion (Ridge / Weight Decay)

```
R(theta) = (1/2) SUM_j theta_j^2 = (1/2) ||theta||_2^2
```

**Gradiente:**
```
dR/d(theta_j) = theta_j
```

**La actualizacion se convierte en:**
```
theta_j <-- theta_j * (1 - eta*lambda) - eta * dJ/d(theta_j)
```

El factor `(1 - eta*lambda)` **encoge multiplicativamente** los pesos en cada paso --> "weight decay".

**Interpretacion Bayesiana:** L2 es equivalente a un **prior Gaussiano** sobre los pesos:
```
P(theta) = PROD_j N(theta_j | 0, 1/lambda)
```

**Efecto:** Encoge todos los pesos hacia cero pero raramente los hace exactamente cero. Prefiere muchos pesos pequenos sobre pocos grandes.

---

### 6.2 L1 Regularizacion (Lasso)

```
R(theta) = SUM_j |theta_j| = ||theta||_1
```

**Subgradiente:**
```
dR/d(theta_j) = sign(theta_j)
```

**Interpretacion Bayesiana:** Equivale a un **prior de Laplace**:
```
P(theta) = PROD_j Laplace(theta_j | 0, 1/lambda)
```

**Efecto:** Induce **sparsity** -- lleva muchos pesos a exactamente cero. Actua como **seleccion automatica de features**.

---

### 6.3 Elastic Net (L1 + L2)

```
R(theta) = alpha * ||theta||_1 + (1-alpha)/2 * ||theta||_2^2
```

Combina sparsity de L1 con el efecto de agrupamiento de L2.

### Tabla resumen de regularizacion

| Regularizador | Formula | Sparsity | Diferenciable | Prior |
|---|---|---|---|---|
| **L2 (Ridge)** | `(lambda/2) * \|\|theta\|\|_2^2` | No (encoge, nunca cero) | Si | Gaussiano |
| **L1 (Lasso)** | `lambda * \|\|theta\|\|_1` | Si (ceros exactos) | No (en 0) | Laplace |
| **Elastic Net** | Mix L1 + L2 | Si (parcial) | No (en 0) | Mixto |

### Notas practicas importantes

- Los **biases tipicamente NO se regularizan** (penalizar el bias no sirve de nada)
- En deep learning, **L2 (weight decay) es la mas comun**, especialmente en `AdamW` (que desacopla el weight decay de la actualizacion del gradiente)
- **Dropout, batch normalization, data augmentation y early stopping** son formas **implicitas** de regularizacion que no aparecen en la funcion objetivo

---

# PARTE II: BACKPROPAGATION

---

## 7. Contexto Historico

El paper fundamental:

> **Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986)**. "Learning representations by back-propagating errors." *Nature*, 323, 533-536.

### El problema que resolvio

Antes de 1986, las redes neuronales estaban limitadas a perceptrones de una sola capa. Minsky y Papert (1969) habian demostrado que los perceptrones de una capa no pueden aprender funciones no linealmente separables (como XOR). El campo necesitaba una forma de entrenar redes **multicapa** -- calcular como ajustar los pesos en **capas ocultas** que no tienen conexion directa con el error de salida.

Backpropagation proporciono un metodo eficiente para computar el gradiente de la funcion de perdida respecto a **cada peso** en la red, independientemente de la profundidad, mediante la aplicacion sistematica de la **regla de la cadena**.

### Antecedentes

- Bryson & Ho (1969) describieron el metodo en teoria de control
- Werbos (1974) lo describio en su tesis doctoral
- LeCun (1985) lo desarrollo independientemente
- Pero fueron Rumelhart, Hinton y Williams quienes demostraron su efectividad practica

---

## 8. La Regla de la Cadena (Chain Rule)

### Definicion formal

Si `y = f(g(x))`, donde `u = g(x)`, entonces:

```
dy/dx = dy/du * du/dx
```

### Regla de la cadena multivariada (general)

Si `L = L(y_1, y_2, ..., y_m)` y cada `y_j = y_j(x_1, x_2, ..., x_n)`, entonces:

```
dL/dx_i = SUM_{j=1}^{m} (dL/dy_j) * (dy_j/dx_i)
```

**Esta es la base matematica de backpropagation**: la perdida depende de las salidas, que dependen de las activaciones ocultas, que dependen de los pesos. La regla de la cadena permite descomponer el gradiente en un producto de derivadas locales.

### Ejemplo concreto

Sea `L = (y - t)^2`, donde `y = sigma(z)`, `z = w*x + b`, y `sigma(z) = 1/(1+e^(-z))`.

```
dL/dw = dL/dy * dy/dz * dz/dw
```

Cada factor es una **derivada local**:
- `dL/dy = 2(y - t)`
- `dy/dz = sigma(z) * (1 - sigma(z))`
- `dz/dw = x`

Resultado: `dL/dw = 2(y - t) * sigma(z) * (1 - sigma(z)) * x`

---

## 9. Grafos Computacionales

Una red neuronal se representa como un **Grafo Aciclico Dirigido (DAG)** donde:

- **Nodos** = operaciones (suma, multiplicacion, funciones de activacion) o variables (pesos, biases, inputs)
- **Aristas** = flujo de datos (tensores)

### Ejemplo: `L = (sigma(w*x + b) - t)^2`

```
[x] --\
       [*] --> [z=w*x] --\
[w] --/                    [+] --> [a=z+b] --> [sigma] --> [y] --> [-] --> [e=y-t] --> [^2] --> [L]
                          /                                        /
[b] ---------------------/                              [t] ------/
```

**Importancia del DAG:**
1. **Forward pass**: evalua nodos en orden topologico (inputs --> output)
2. **Backward pass**: recorre en orden topologico inverso, acumulando gradientes via chain rule
3. Cada nodo solo necesita conocer su **derivada local**

---

## 10. Forward Pass: Ejemplo Numerico Completo

### Arquitectura de la red

Red simple de 2 capas (1 capa oculta, 1 salida):
- Input: `x_1 = 0.5, x_2 = 0.3`
- Capa oculta: 2 neuronas, activacion sigmoid
- Capa de salida: 1 neurona, activacion sigmoid
- Target: `t = 1.0`
- Loss: `MSE = (1/2)(y - t)^2`

### Pesos y biases iniciales

**Capa oculta** (2 inputs --> 2 neuronas):
```
W^(1) = | w_11  w_12 | = | 0.15  0.20 |
        | w_21  w_22 |   | 0.25  0.30 |

b^(1) = | 0.35 |
        | 0.35 |
```

**Capa de salida** (2 neuronas --> 1 salida):
```
W^(2) = | w_31  w_32 | = | 0.40  0.45 |

b^(2) = 0.60
```

### Paso 1: Pre-activacion de la capa oculta

```
z_1^(1) = w_11 * x_1 + w_12 * x_2 + b_1
        = 0.15 * 0.5 + 0.20 * 0.3 + 0.35
        = 0.075 + 0.06 + 0.35
        = 0.485

z_2^(1) = w_21 * x_1 + w_22 * x_2 + b_2
        = 0.25 * 0.5 + 0.30 * 0.3 + 0.35
        = 0.125 + 0.09 + 0.35
        = 0.565
```

### Paso 2: Activacion sigmoid de la capa oculta

```
sigma(z) = 1 / (1 + e^(-z))

h_1 = sigma(0.485) = 1 / (1 + e^(-0.485)) = 1 / 1.6161 = 0.6190
h_2 = sigma(0.565) = 1 / (1 + e^(-0.565)) = 1 / 1.5685 = 0.6376
```

### Paso 3: Pre-activacion de la capa de salida

```
z^(2) = w_31 * h_1 + w_32 * h_2 + b^(2)
      = 0.40 * 0.6190 + 0.45 * 0.6376 + 0.60
      = 0.2476 + 0.2869 + 0.60
      = 1.1345
```

### Paso 4: Activacion de salida

```
y = sigma(1.1345) = 1 / (1 + e^(-1.1345)) = 1 / 1.3215 = 0.7565
```

### Paso 5: Calculo del Loss

```
L = (1/2)(y - t)^2 = (1/2)(0.7565 - 1.0)^2 = (1/2)(0.0593) = 0.02966
```

### Resumen del forward pass

| Variable | Valor |
|---|---|
| `z_1^(1)` | 0.485 |
| `z_2^(1)` | 0.565 |
| `h_1` | 0.6190 |
| `h_2` | 0.6376 |
| `z^(2)` | 1.1345 |
| `y` | 0.7565 |
| `L` | 0.02966 |

---

## 11. Backward Pass: Derivacion Paso a Paso

Ahora calculamos los gradientes de `L` respecto a **cada peso y bias**, moviéndonos desde la salida hacia la entrada.

### Paso 1: Gradiente de la Loss respecto a la salida

```
dL/dy = y - t = 0.7565 - 1.0 = -0.2435
```

### Paso 2: Gradiente a traves del sigmoid de salida

La derivada del sigmoid es `sigma'(z) = sigma(z) * (1 - sigma(z))`:

```
dy/dz^(2) = y * (1 - y) = 0.7565 * 0.2435 = 0.1842
```

Definimos la **senal de error** (delta) de salida:

```
delta^(2) = dL/dz^(2) = dL/dy * dy/dz^(2) = (-0.2435)(0.1842) = -0.04484
```

### Paso 3: Gradientes de los pesos de salida

```
dL/dw_31 = delta^(2) * h_1 = (-0.04484)(0.6190) = -0.02776
dL/dw_32 = delta^(2) * h_2 = (-0.04484)(0.6376) = -0.02859
dL/db^(2) = delta^(2) = -0.04484
```

### Paso 4: Propagar el error hacia la capa oculta

```
dL/dh_1 = delta^(2) * w_31 = (-0.04484)(0.40) = -0.01794
dL/dh_2 = delta^(2) * w_32 = (-0.04484)(0.45) = -0.02018
```

### Paso 5: Gradiente a traves del sigmoid de la capa oculta

```
delta_1^(1) = dL/dh_1 * h_1 * (1 - h_1)
            = (-0.01794)(0.6190)(0.3810)
            = (-0.01794)(0.2358)
            = -0.004231

delta_2^(1) = dL/dh_2 * h_2 * (1 - h_2)
            = (-0.02018)(0.6376)(0.3624)
            = (-0.02018)(0.2310)
            = -0.004662
```

### Paso 6: Gradientes de los pesos de la capa oculta

```
dL/dw_11 = delta_1^(1) * x_1 = (-0.004231)(0.5) = -0.002116
dL/dw_12 = delta_1^(1) * x_2 = (-0.004231)(0.3) = -0.001269
dL/dw_21 = delta_2^(1) * x_1 = (-0.004662)(0.5) = -0.002331
dL/dw_22 = delta_2^(1) * x_2 = (-0.004662)(0.3) = -0.001399
dL/db_1  = delta_1^(1) = -0.004231
dL/db_2  = delta_2^(1) = -0.004662
```

### Resumen de todos los gradientes

| Parametro | Gradiente | Interpretacion |
|---|---|---|
| `w_31` | -0.02776 | Aumentar (mover contra el gradiente) |
| `w_32` | -0.02859 | Aumentar |
| `b^(2)` | -0.04484 | Aumentar |
| `w_11` | -0.002116 | Aumentar (pero menos, mas lejos de la salida) |
| `w_12` | -0.001269 | Aumentar |
| `w_21` | -0.002331 | Aumentar |
| `w_22` | -0.001399 | Aumentar |
| `b_1` | -0.004231 | Aumentar |
| `b_2` | -0.004662 | Aumentar |

**Observacion clave**: Todos los gradientes son negativos porque la prediccion (0.7565) fue menor que el target (1.0). Los pesos necesitan **aumentar** para reducir la perdida. Notar tambien que los gradientes de la capa oculta son ~10x mas pequenos que los de la capa de salida -- esto es el inicio del problema de **vanishing gradients**.

### Paso 7: Actualizacion de pesos (con eta = 0.5)

```
w_31_new = 0.40 - 0.5 * (-0.02776) = 0.40 + 0.01388 = 0.41388
w_32_new = 0.45 - 0.5 * (-0.02859) = 0.45 + 0.01430 = 0.46430
w_11_new = 0.15 - 0.5 * (-0.002116) = 0.15 + 0.001058 = 0.151058
... (analogamente para todos los demas)
```

---

## 12. Flujo de Gradientes y la Matriz Jacobiana

### Patron general para una red de L capas

El gradiente en la capa `l` es:

```
dL/dW^(l) = delta^(l) * (a^(l-1))^T
```

Donde la senal de error `delta^(l)` se propaga recursivamente:

```
delta^(l) = ( (W^(l+1))^T * delta^(l+1) ) ⊙ sigma'(z^(l))
```

(`⊙` = multiplicacion elemento a elemento)

### Interpretacion intuitiva del flujo

```
Capa de Salida        Capa Oculta 2        Capa Oculta 1        Input
                                                                      
  delta^(3) --W^(3)^T--> delta^(2) --W^(2)^T--> delta^(1)    
     |                      |                      |          
     v                      v                      v          
  dL/dW^(3)              dL/dW^(2)              dL/dW^(1)     
```

Cada capa:
1. **Recibe** la senal de error `delta` de la capa superior
2. **La escala** por la transpuesta de la matriz de pesos (distribuyendo "culpa" proporcionalmente)
3. **La modula** por la derivada de la activacion local `sigma'(z)` (las neuronas mas "activas" pasan mas gradiente)
4. **Produce** gradientes de pesos por producto externo con las activaciones de entrada
5. **Pasa** la senal de error residual hacia atras

### La Matriz Jacobiana

El **Jacobiano** de una funcion vectorial `f: R^n --> R^m` es la matriz `m x n` de derivadas parciales:

```
J = | df_1/dx_1  ...  df_1/dx_n |
    |    ...     ...     ...     |
    | df_m/dx_1  ...  df_m/dx_n |
```

**Rol en backpropagation:** Para una capa que mapea `z in R^n` a `a in R^m` via `a = f(z)`:

```
dL/dz = J_f^T * dL/da
```

Esto es un **vector-Jacobian product (VJP)**. En backpropagation, **nunca** construimos explicitamente el Jacobiano completo -- solo computamos su producto con el vector gradiente entrante.

**Ejemplo con capa sigmoid** (activacion por elementos):

El Jacobiano es diagonal:
```
J_sigma = diag(sigma'(z_1), sigma'(z_2), ..., sigma'(z_n))
```

El VJP se simplifica a multiplicacion por elementos:
```
dL/dz_i = sigma'(z_i) * dL/da_i
```

**Ejemplo con capa lineal** `a = W*x`:

El Jacobiano respecto a `x` es `W`, entonces:
```
dL/dx = W^T * dL/da
```

**Para la red completa** con capas `f_1, f_2, ..., f_K`:

```
dL/dx = J_{f_1}^T * J_{f_2}^T * ... * J_{f_K}^T * dL/dy
```

El gradiente es un **producto de Jacobianos transpuestos** -- esta es la esencia matematica de backpropagation.

---

## 13. Vanishing y Exploding Gradients

### La raiz matematica del problema

Para una red profunda con K capas, el gradiente en la capa 1 involucra un producto:

```
dL/dW^(1) ~ PROD_{k=1}^{K-1} ( W^(k+1) * diag(sigma'(z^(k))) )
```

La magnitud del gradiente depende del **producto de normas espectrales** de estas matrices.

### Vanishing Gradients

**Con activacion sigmoid:** `sigma'(z) = sigma(z) * (1 - sigma(z))`

El valor maximo de `sigma'(z)` es **0.25** (en `z = 0`). En cada capa, el gradiente se multiplica por un factor de a lo mas 0.25. Despues de K capas:

```
||gradiente en capa 1|| <= (0.25)^K * ||gradiente en salida||
```

**Ejemplo numerico (K = 10 capas sigmoid):**

| Capa | Magnitud del gradiente |
|---|---|
| 10 (salida) | 1.0 |
| 9 | 0.25 |
| 8 | 0.0625 |
| 7 | 0.01563 |
| 5 | 0.000977 |
| 3 | 0.0000610 |
| 1 | **0.00000095** |

Las capas tempranas practicamente no aprenden.

### Exploding Gradients

Si las matrices de pesos tienen normas espectrales mayores que 1 y las derivadas de activacion no atenuan suficiente:

```
PROD_{k=1}^{K} ||W^(k)|| * ||sigma'(z^(k))|| >> 1
```

Los gradientes se vuelven enormes -> actualizaciones masivas -> overflow numerico -> NaN.

### Soluciones

| Problema | Solucion | Como funciona |
|---|---|---|
| Vanishing | **ReLU** `max(0,z)` | Derivada es 1 para `z > 0`, sin atenuacion |
| Vanishing | **Conexiones residuales** (ResNets) | Gradiente tiene camino aditivo: `d/dx(x + F(x)) = 1 + F'(x)` |
| Vanishing | **LSTM / GRU gates** | "Carrusel de error constante" preserva gradientes en el tiempo |
| Exploding | **Gradient clipping** | `g <-- g * theta/\|\|g\|\|` si `\|\|g\|\| > theta` |
| Ambos | **Inicializacion cuidadosa** (Xavier/He) | Pesos iniciales preservan varianza entre capas |
| Ambos | **Batch Normalization** | Re-centra y re-escala activaciones, estabilizando gradientes |

---

## 14. Diferenciacion Automatica

Backpropagation es un caso especifico de **diferenciacion automatica en modo reverso (reverse-mode AD)**. AD es distinto tanto de la diferenciacion simbolica como de la numerica (diferencias finitas).

### Forward Mode AD

Computa derivadas junto con la evaluacion de la funcion, propagando **vectores tangentes** hacia adelante.

- **Costo**: Un pase forward computa derivadas respecto a **una variable de entrada**
- Para `n` inputs, necesitamos `n` pases
- **Eficiente cuando**: `n` (inputs) es pequeno relativo a `m` (outputs)

### Reverse Mode AD (= Backpropagation)

Propaga **valores adjuntos** (gradientes) hacia atras desde las salidas hacia las entradas.

- **Costo**: Un pase backward computa derivadas respecto a **todas las variables de entrada** simultaneamente
- Para `m` outputs, necesitamos `m` pases
- **Eficiente cuando**: `m` (outputs) es pequeno relativo a `n` (inputs)

### Por que backpropagation usa el modo reverso

En el entrenamiento de redes neuronales:
- **Dimension de salida**: 1 (el loss escalar `L`)
- **Dimension de entrada**: **millones** de parametros

El modo reverso computa TODOS los gradientes en **un solo pase backward** -- costo computacional proporcional al forward pass (~2-3x). El modo forward requeriria un pase por parametro -- completamente inviable.

### Tabla comparativa

| Propiedad | Forward Mode | Reverse Mode (Backprop) |
|---|---|---|
| Direccion | Input --> Output | Output --> Input |
| Propaga | Vectores tangentes | Valores adjuntos |
| Un pase da | `d(todos outputs)/d(un input)` | `d(un output)/d(todos inputs)` |
| Costo para Jacobiano completo | O(n) pases | O(m) pases |
| Mejor cuando | `n << m` | `m << n` |
| Redes neuronales | Inviable | Ideal (m=1) |
| Memoria | Baja | **Alta** (debe almacenar activaciones) |

> **Trade-off de memoria**: El modo reverso requiere almacenar todas las activaciones intermedias del forward pass para usarlas durante el backward pass. Es por esto que la **memoria GPU** es el cuello de botella en redes grandes, y tecnicas como **gradient checkpointing** sacrifican computo adicional para reducir memoria.

---

## 15. Pseudocodigo Completo de Backpropagation

### Version basica (un ejemplo)

```
ALGORITMO: Backpropagation para Red Fully Connected
====================================================

INPUT:
  - Ejemplo (x, t): x = input, t = target
  - Red con L capas
  - Pesos W^(l) y biases b^(l) para l = 1, ..., L
  - Funcion de activacion sigma() y su derivada sigma'()
  - Funcion de perdida Loss(y, t) y su derivada dLoss/dy
  - Learning rate eta

--- FORWARD PASS ---

1.  a^(0) = x                              // Input es la "activacion" de capa 0

2.  PARA l = 1 HASTA L:
3.      z^(l) = W^(l) * a^(l-1) + b^(l)   // Pre-activacion
4.      a^(l) = sigma(z^(l))               // Post-activacion
5.  FIN PARA

6.  y = a^(L)                              // Salida de la red
7.  L = Loss(y, t)                         // Calcular perdida

--- BACKWARD PASS ---

8.  // Senal de error de la capa de salida
9.  delta^(L) = dLoss/dy ⊙ sigma'(z^(L))

10. PARA l = L-1 BAJANDO HASTA 1:
11.     delta^(l) = ( (W^(l+1))^T * delta^(l+1) ) ⊙ sigma'(z^(l))
12. FIN PARA

--- CALCULAR GRADIENTES ---

13. PARA l = 1 HASTA L:
14.     dL/dW^(l) = delta^(l) * (a^(l-1))^T    // Producto externo
15.     dL/db^(l) = delta^(l)
16. FIN PARA

--- ACTUALIZAR PARAMETROS ---

17. PARA l = 1 HASTA L:
18.     W^(l) = W^(l) - eta * dL/dW^(l)
19.     b^(l) = b^(l) - eta * dL/db^(l)
20. FIN PARA
```

### Version mini-batch SGD

```
ALGORITMO: Mini-batch SGD con Backpropagation
==============================================

1.  Inicializar pesos W^(l) y biases b^(l) aleatoriamente (ej: Xavier)

2.  PARA epoch = 1 HASTA E:
3.      Mezclar dataset D aleatoriamente
4.      Particionar D en mini-batches de tamano B

5.      PARA CADA mini-batch {(x_1,t_1), ..., (x_B,t_B)}:

6.          Inicializar acumuladores de gradiente a cero

7.          PARA i = 1 HASTA B:
8.              Forward pass con x_i para obtener y_i
9.              Backward pass para obtener dL/dW^(l), dL/db^(l)
10.             Acumular gradientes
11.         FIN PARA

12.         // Actualizar con gradiente promedio
13.         PARA l = 1 HASTA L:
14.             W^(l) = W^(l) - (eta/B) * grad_W^(l)
15.             b^(l) = b^(l) - (eta/B) * grad_b^(l)
16.         FIN PARA

17.     FIN PARA (mini-batches)
18. FIN PARA (epochs)
```

### Complejidad computacional

| Operacion | Costo por ejemplo |
|---|---|
| Forward pass | O(SUM_{l=1}^{L} n_l * n_{l-1}) |
| Backward pass | Mismo orden que forward (~2x en practica) |
| Memoria (activaciones) | O(SUM_{l=1}^{L} n_l) |
| Memoria (pesos) | O(SUM_{l=1}^{L} n_l * n_{l-1}) |

---

# PARTE III: LEARNING RATE Y PAISAJES DE OPTIMIZACION

---

## 16. El Paisaje de Optimizacion de Redes Neuronales

### 16.1 Optimizacion Convexa vs No-Convexa

**Funcion convexa (definicion formal):** Una funcion `f: R^n --> R` es convexa si para todo `x, y` en su dominio y todo `lambda in [0, 1]`:

```
f(lambda*x + (1-lambda)*y) <= lambda*f(x) + (1-lambda)*f(y)
```

Equivalentemente, para funciones dos veces diferenciables, `f` es convexa si su **Hessiano** `H(x)` es positivo semi-definido en todo punto (todos los eigenvalues >= 0).

**Propiedad clave de optimizacion convexa:** Todo minimo local es un minimo global. GD sobre funciones convexas **garantiza** convergencia al optimo global.

**Las redes neuronales son no-convexas** porque componen transformaciones no-lineales con lineales. Esto implica:
- Multiples minimos locales
- Proliferacion de saddle points
- Sin garantia de encontrar el minimo global
- GD puede converger a distintas soluciones dependiendo de la inicializacion

### 16.2 Superficies de Perdida en Alta Dimension

La superficie de perdida `L(theta)` mapea vectores de parametros `theta in R^n` (donde n puede ser millones) a valores escalares de perdida:

- **Extremadamente alta dimension**: Redes modernas tienen millones a miles de millones de parametros
- **No-convexa pero estructurada**: Diferentes minimos encontrados por SGD estan frecuentemente conectados por caminos de baja perdida ("mode connectivity")
- **Muchas soluciones buenas equivalentes**: En redes sobreparametrizadas, muchos minimos locales tienen valores de loss cercanos al global

### 16.3 Definiciones formales

**Minimo local:** Un punto `theta*` es minimo local si existe un vecindario `N(theta*)` tal que `f(theta*) <= f(theta)` para todo `theta in N(theta*)`. Condiciones:
1. Gradiente cero: `nabla f(theta*) = 0`
2. Hessiano positivo semi-definido: todos los eigenvalues de `H(theta*) >= 0`

**Minimo global:** `theta*` tal que `f(theta*) <= f(theta)` para **todo** `theta` en el dominio.

**Saddle point:** Punto critico donde `nabla f(theta*) = 0` pero el Hessiano tiene eigenvalues de **signo mixto** -- la funcion sube en algunas direcciones y baja en otras.

### 16.4 Por que los Saddle Points son mas problematicos que los Minimos Locales

**Paper:** Dauphin, Pascanu, Gulcehre, Cho, Ganguli, Bengio (2014). *"Identifying and attacking the saddle point problem in high-dimensional non-convex optimization."* NeurIPS 2014.

**Argumento probabilistico:** En un punto critico, el Hessiano tiene `n` eigenvalues. Para que sea un minimo local, **todos** deben ser positivos. Si cada eigenvalue tiene probabilidad ~1/2 de ser positivo:

```
P(minimo local) = P(todos n eigenvalues positivos) ~ (1/2)^n
```

Con `n = 1,000,000` parametros: `(1/2)^1000000` -- astronomicamente improbable. Los **saddle points vastamente superan en numero** a los minimos locales.

**Por que son problematicos:**
- Rodeados de plateaus de alto error que ralentizan dramaticamente el aprendizaje
- Gradientes cercanos a cero (porque `nabla f = 0` en el saddle)
- Metodos de primer orden (SGD) pueden tardar exponencialmente en escapar
- Dan la "falsa impresion" de estar en un minimo local

### 16.5 La superficie es "sorprendentemente suave" (Li et al., 2018)

**Paper:** Li, Xu, Taylor, Studer, Goldstein (2018). *"Visualizing the Loss Landscape of Neural Nets."* NeurIPS 2018.

**Hallazgos clave:**
1. **Skip connections causan "convexificacion" dramatica**: ResNets tienen paisajes mucho mas suaves que redes equivalentes sin ellas (VGG)
2. **La profundidad sin skip connections crea caos**: Redes poco profundas tienen paisajes benignos, pero al agregar profundidad sin residuales, la superficie se vuelve caotica
3. **Redes mas anchas retrasan la transicion caotica**: Mas filtros por capa = superficie mas suave
4. **La "planitud" del minimo correlaciona con generalizacion**: Minimos planos generalizan mejor que minimos agudos

---

## 17. Teoria Formal del Learning Rate

### 17.1 Rol formal en la convergencia de GD

Gradient descent actualiza parametros como:
```
theta_{t+1} = theta_t - eta * nabla f(theta_t)
```

El learning rate `eta` controla el trade-off entre:
- **Muy grande**: Sobrepasar el minimo, divergencia, oscilaciones
- **Muy pequeno**: Convergencia extremadamente lenta
- **Justo**: Convergencia estable hacia un minimo

### 17.2 Continuidad de Lipschitz y L-suavidad

**Definicion (L-smooth / gradiente Lipschitz continuo):** Una funcion diferenciable `f` es L-smooth si:

```
||nabla f(x) - nabla f(y)|| <= L * ||x - y||    para todo x, y
```

`L` es la **constante de suavidad** (o constante de Lipschitz del gradiente). Intuitivamente, la curvatura de la funcion esta acotada por `L`.

Para funciones dos veces diferenciables, esto equivale a:
```
||H(x)|| <= L    para todo x    (norma espectral del Hessiano)
```

### 17.3 El Lema de Descenso y la condicion eta < 2/L

**Lema de Descenso** (resultado fundamental en teoria de optimizacion):

Para una funcion L-smooth, gradient descent con paso `eta` satisface:

```
f(x - eta * nabla f(x)) <= f(x) - eta * (1 - L*eta/2) * ||nabla f(x)||^2
```

**Analisis:**
- El termino `eta * (1 - L*eta/2)` debe ser **positivo** para que la funcion decrezca
- Esto requiere: `1 - L*eta/2 > 0`
- Lo cual da: **`eta < 2/L`**
- El decrecimiento garantizado se **maximiza** cuando `eta = 1/L` (paso optimo fijo)

**Resumen de condiciones del paso:**

| Condicion | Efecto |
|---|---|
| `eta < 2/L` | Garantiza descenso (la funcion decrece en cada paso) |
| `eta = 1/L` | Paso fijo optimo (maximiza progreso garantizado) |
| `eta >= 2/L` | Sin garantia de convergencia; puede divergir |

### 17.4 Tasas de convergencia

Con paso `eta = 1/L`:

| Escenario | Tasa | Iteraciones para epsilon-precision |
|---|---|---|
| No-convexa, L-smooth | `min ||nabla f||^2 = O(1/t)` | `O(1/epsilon)` para punto estacionario |
| Convexa, L-smooth | `f(x_t) - f* = O(1/t)` | `O(1/epsilon)` |
| mu-fuertemente convexa, L-smooth | `f(x_t) - f* = O((1-mu/L)^t)` | `O(L/mu * log(1/epsilon))` -- tasa lineal |

El ratio `kappa = L/mu` es el **numero de condicion**; mayor kappa = convergencia mas lenta.

### 17.5 Por que el Learning Rate es el hiperparametro mas importante

Segun Yoshua Bengio (2012) y Leslie Smith (2018):

1. **Control directo**: Determina directamente el tamano del paso en el espacio de parametros
2. **Sensibilidad**: Muy alto = divergencia; muy bajo = convergencia a minimos pobres. El rango "bueno" es relativamente estrecho
3. **Interaccion con todo**: Batch size, momentum, weight decay, arquitectura -- todos cambian el learning rate efectivo
4. **Efecto de regularizacion**: Learning rates grandes actuan como regularizacion implicita, inyectando ruido y prefiriendo minimos planos
5. **Regla practica de Bengio**: Empezar con un LR que cause divergencia (ej: 1.0), dividir por 3 repetidamente hasta que el entrenamiento se estabilice

---

## 18. Estrategias de Learning Rate Scheduling

### 18.1 Step Decay

El mas simple: multiplicar el LR por un factor constante en hitos de epocas fijos.

```
eta_t = eta_0 * gamma^(floor(t / step_size))
```

**Ejemplo clasico (entrenamiento de ImageNet):**
- `eta_0 = 0.1`, multiplicar por `gamma = 0.1` en epocas 30, 60, 90

### 18.2 Decaimiento Exponencial

Decay suave y continuo:
```
eta_t = eta_0 * gamma^t     (gamma ligeramente < 1, ej: 0.95 por epoca)
```

### 18.3 Cosine Annealing (Loshchilov & Hutter, SGDR, ICLR 2017)

**Paper:** *"Stochastic Gradient Descent with Warm Restarts"* (arXiv: 1608.03983)

**Formula** (dentro de un ciclo de longitud T):
```
eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T))
```

Decrece suavemente de `eta_max` a `eta_min` siguiendo una curva medio-coseno.

**Con warm restarts (SGDR):**
- Despues de cada ciclo de longitud `T_i`, el LR "reinicia" a `eta_max`
- Los ciclos pueden crecer: `T_{i+1} = T_mult * T_i`
- Permite al optimizador escapar de minimos locales periodicamente

**Por que funciona:** La forma coseno pasa mas tiempo en LRs moderados comparado con decay lineal. Los reinicios ayudan a explorar diferentes cuencas de atraccion.

### 18.4 Warmup (Calentamiento)

Empezar con un LR muy pequeno e incrementar gradualmente hasta el valor objetivo.

**Tipos comunes:**
- **Warmup lineal:** `eta_t = eta_target * (t / T_warmup)` para `t < T_warmup`
- **Warmup exponencial:** `eta_t = eta_target * (t / T_warmup)^2`

**Por que es necesario (Gilmer et al., NeurIPS 2024 - "Why Warmup the Learning Rate?"):**

1. **Estabiliza el entrenamiento temprano**: Al inicio, los gradientes son ruidosos y el paisaje esta mal condicionado. Pasos grandes pueden empujar los parametros a regiones inestables
2. **Permite LRs objetivo mas altos**: El warmup fuerza a la red hacia areas mejor condicionadas antes de dar pasos grandes
3. **Estabiliza optimizadores adaptativos**: Para Adam, las estadisticas de momentos son poco confiables al inicio (pocas muestras). El warmup da tiempo para que se acumulen
4. **Critico para Transformers**: Sin warmup, los gradientes en capas bajas de Transformers profundos desaparecen en pocos pasos
5. **Esencial para batches grandes**: La regla de escalamiento lineal se rompe al inicio del entrenamiento

**Regla practica:** Warmup es casi universalmente usado en Transformers. Dura tipicamente 1-10% de los pasos totales.

### 18.5 Cyclical Learning Rates (Leslie Smith, 2015/2017)

**Paper:** *"Cyclical Learning Rates for Training Neural Networks"* (arXiv: 1506.01186)

En vez de decrecer monotonamente, el LR oscila ciclicamente entre un minimo (`base_lr`) y un maximo (`max_lr`).

**Politica triangular:**
```
cycle = floor(1 + iteration / (2 * step_size))
x = |iteration / step_size - 2 * cycle + 1|
lr = base_lr + (max_lr - base_lr) * max(0, 1 - x)
```

**Por que funciona:** Aumentar periodicamente el LR puede tener efectos negativos a corto plazo (loss sube temporalmente) pero beneficios a largo plazo: permite al optimizador atravesar plateaus de saddle points y escapar minimos locales pobres.

### 18.6 One-Cycle Policy (Leslie Smith, 2018)

**Paper:** *"Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"* (arXiv: 1708.07120)

**Tres fases:**

```
LR
 ^
 |         /\
 |        /  \
 |       /    \
 |      /      \
 |     /        \
 |    /          \_____
 |   /                 |
 +---|----|----|----|-->  Epochs
     Fase1 Fase2  Fase3
    (warm)  (decay) (aniq)
```

1. **Fase 1 - Warmup (~45% del entrenamiento):** LR sube linealmente de `lr_min` a `max_lr`
2. **Fase 2 - Decay (~45%):** LR baja de `max_lr` a `lr_min`
3. **Fase 3 - Aniquilacion (~10%):** LR cae varias ordenes de magnitud adicionales

**Momentum ciclico (relacion inversa con LR):**
- Cuando LR es alto -> momentum BAJO (0.85)
- Cuando LR es bajo -> momentum ALTO (0.95)

**Super-convergencia:** Redes entrenadas con 1cycle alcanzan la misma precision en **1/5 a 1/10** de las epocas. Ejemplo: ResNet-56 en CIFAR-10 logra 92.3% en ~50 epocas vs 360+ con entrenamiento estandar.

### 18.7 Learning Rate Range Test / LR Finder

**Procedimiento:**
1. Empezar con un LR muy pequeno (ej: 1e-7)
2. Entrenar una epoca, **incrementando** el LR exponencialmente despues de cada mini-batch
3. Registrar la loss en cada paso
4. Graficar loss vs learning rate (escala log)

**Como leer el grafico:**

```
Loss
 ^
 |  ___________
 |              \
 |               \         <-- zona de descenso rapido
 |                \
 |                 \____
 |                      \
 |                       |  <-- LR recomendado aqui (mayor pendiente negativa)
 |                       |
 |                        \___/  <-- divergencia
 +--|--------|---------|-----> LR (log scale)
   1e-7     1e-4      1e-1
```

- **max_lr recomendado**: Donde la loss esta descendiendo mas rapidamente (NO en el punto minimo)
- **base_lr**: Tipicamente 1/10 a 1/3 de max_lr

**Base matematica:** El LR finder estima empiricamente la frontera `eta < 2/L` para la constante de Lipschitz local.

---

## 19. Inicializacion de Pesos y su Relacion con el Learning Rate

### 19.1 Xavier/Glorot Initialization (Glorot & Bengio, 2010)

Disenada para redes con activacion sigmoid o tanh. Objetivo: mantener la varianza de activaciones Y gradientes constante entre capas.

**Derivacion:** Para una capa `y = W*x` con `fan_in` inputs y `fan_out` outputs:
- Forward: `Var(y) = fan_in * Var(W) * Var(x)`. Para `Var(y) = Var(x)`: `Var(W) = 1/fan_in`
- Backward: `Var(delta_x) = fan_out * Var(W) * Var(delta_y)`. Para preservar: `Var(W) = 1/fan_out`
- **Compromiso (promediando):**

```
Var(W) = 2 / (fan_in + fan_out)
```

**Distribuciones:**
- Normal: `W ~ N(0, 2/(fan_in + fan_out))`
- Uniforme: `W ~ U(-a, a)` donde `a = sqrt(6 / (fan_in + fan_out))`

### 19.2 He/Kaiming Initialization (He et al., 2015)

Disenada para activaciones ReLU. ReLU elimina ~la mitad de las activaciones (todas las negativas), lo que efectivamente reduce la varianza a la mitad.

**Compensacion:**
```
Var(W) = 2 / fan_in
```

**Distribuciones:**
- Normal: `W ~ N(0, 2/fan_in)`
- Uniforme: `W ~ U(-a, a)` donde `a = sqrt(6 / fan_in)`

Para Leaky ReLU con pendiente negativa `alpha`:
```
Var(W) = 2 / ((1 + alpha^2) * fan_in)
```

### 19.3 Por que una mala inicializacion hace que cualquier LR falle

**Pesos muy pequenos:**
- Activaciones se encogen exponencialmente: `||a_L|| ~ (sigma_W)^L * ||x||`
- Gradientes tambien se encogen -> ni un LR grande ayuda (gradiente * lr sigue siendo negligible)

**Pesos muy grandes:**
- Activaciones explotan exponencialmente
- Gradientes explotan -> cualquier LR no-trivial causa actualizaciones masivas -> NaN

**Balance critico:** Xavier/He colocan la red en un regimen donde:
- `Var(activaciones) = constante` entre capas
- La constante de Lipschitz `L` del loss es razonable
- La condicion `eta < 2/L` es alcanzable con LRs practicos

**La inicializacion determina el punto de partida en la superficie de perdida.** Li et al. (2018) mostraron que diferentes puntos de inicio caen en regiones con geometria muy distinta -- algunas suaves, otras caoticas.

---

## 20. Batch Size y Learning Rate

### 20.1 Regla de Escalamiento Lineal (Goyal et al., 2017)

**Paper:** *"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"* (arXiv: 1706.02677)

**La regla:** Cuando el batch size se multiplica por `k`, multiplicar el LR por `k`:

```
lr_new = lr_base * (batch_size_new / batch_size_base)
```

**Ejemplo:** Baseline con batch 256 y lr = 0.1. Para batch 8192 (32x): lr = 0.1 * 32 = 3.2.

### 20.2 Por que batches mas grandes necesitan LR mas grande

**Argumento de reduccion de varianza:**
```
Var(g_B) = Var(nabla L_i) / B
```

Duplicar el batch reduce la varianza del gradiente a la mitad. Con menos ruido, podemos dar pasos mas grandes de forma segura.

**Ruido de SGD como regularizacion:**
```
noise ~ sqrt(lr / B) * sigma
```

Para mantener el mismo nivel de ruido al aumentar B por k, debemos aumentar lr por k.

### 20.3 Limitaciones

1. **Se rompe al inicio del entrenamiento**: Cuando la red cambia rapidamente, los gradientes no son constantes entre pasos -> por eso se necesita **warmup**
2. **Se rompe para batches muy grandes**: Mas alla de un "batch size critico", los rendimientos decrecientes hacen que la regla lineal sobreestime el LR apropiado
3. **Escalamiento de raiz cuadrada** (alternativa): `lr ~ sqrt(k)` preserva la escala de ruido en vez de la magnitud del gradiente

### 20.4 Tabla resumen de relaciones

| Concepto | Formula |
|---|---|
| Actualizacion GD | `theta_{t+1} = theta_t - eta * nabla f(theta_t)` |
| Condicion L-smoothness | `\|\|nabla f(x) - nabla f(y)\|\| <= L * \|\|x - y\|\|` |
| Lema de descenso | `f(x - eta*nabla f) <= f(x) - eta*(1-L*eta/2)*\|\|nabla f\|\|^2` |
| LR maximo para convergencia | `eta < 2/L` |
| LR optimo fijo | `eta = 1/L` |
| Xavier init | `Var(W) = 2/(fan_in + fan_out)` |
| He init | `Var(W) = 2/fan_in` |
| Cosine annealing | `eta_t = eta_min + 0.5*(eta_max-eta_min)*(1+cos(pi*t/T))` |
| Regla escalamiento lineal | `lr_new = lr_base * (B_new / B_base)` |
| Varianza gradiente SGD | `Var(g_B) = Var(nabla L_i) / B` |
| Escala ruido SGD | `noise ~ sqrt(lr/B) * sigma` |

---

## 21. Papers Fundamentales Referenciados

### Cubiertos en la Clase 10

| Ano | Paper | Contribucion |
|---|---|---|
| 2015 | Adam (Kingma & Ba) | Optimizador adaptativo con momentum, el mas usado |
| 2018 | ADMM as Continuous Dynamical Systems | Framework de descomposicion de optimizacion |
| 2019 | Ranger (RAdam + Lookahead) | Optimizador "todo-en-uno" |
| 2019 | Lookahead (Zhang, Lucas, Hinton, Ba) | Meta-optimizador k-steps forward, 1 back |
| 2020 | Gradient Centralization (Yong et al.) | Centrar gradientes a media cero (1 linea de codigo) |
| 2021 | AngularGrad (Roy et al.) | Usar info angular entre gradientes consecutivos |

### Referenciados en este documento de profundizacion

| Ano | Paper | Relevancia |
|---|---|---|
| 1986 | Rumelhart, Hinton, Williams | Paper fundamental de backpropagation |
| 2010 | Glorot & Bengio | Inicializacion Xavier |
| 2014 | Dauphin et al. | Saddle points mas problematicos que minimos locales |
| 2015 | He et al. | Inicializacion Kaiming para ReLU |
| 2015 | Leslie Smith | Cyclical Learning Rates |
| 2017 | Goyal et al. | Regla de escalamiento lineal batch/lr |
| 2017 | Loshchilov & Hutter | SGDR / Cosine Annealing |
| 2018 | Li et al. | Visualizacion de paisajes de perdida |
| 2018 | Leslie Smith | Super-Convergence / 1cycle policy |
| 2024 | Gilmer et al. | Por que hacer warmup del LR |

---

## Diagrama de Relaciones entre Conceptos

```
FUNCION OBJETIVO
F(theta) = J(theta) + lambda * R(theta)
     |                        |
     v                        v
FUNCION DE COSTO         REGULARIZACION
(1/N) SUM L(y,y_hat)    L1 / L2 / Elastic Net
     |
     v
FUNCION DE PERDIDA (por ejemplo)
MSE / BCE / CCE / Hinge
     |
     v
GRADIENTE (via backpropagation)
dF/d(theta) = dJ/d(theta) + lambda * dR/d(theta)
     |
     v
REGLA DE ACTUALIZACION
theta_new = theta_old - eta * gradiente
     |              |
     v              v
OPTIMIZADOR     LEARNING RATE
SGD/Adam/etc    Scheduling/Warmup/etc
     |              |
     v              v
CONVERGENCIA <------+
(o divergencia)     ^
     |              |
     v              |
PAISAJE DE PERDIDA -+
(convexo/no-convexo, saddle points,
 minimos locales, superficie suave/caotica)
```
