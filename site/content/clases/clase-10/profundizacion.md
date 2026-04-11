---
title: "Profundizacion - Loss Functions, Backpropagation y Learning Rate"
weight: 20
math: true
---

> Este documento profundiza en los fundamentos teoricos que subyacen a los algoritmos
> de optimizacion cubiertos en la Clase 10. Cubre funciones de perdida, backpropagation
> y la teoria del learning rate con demostraciones matematicas y ejemplos numericos concretos.

---

# Parte I: Funciones de Perdida

---

## 1. Definicion Formal y Jerarquia

### Definicion matematica

Una **funcion de perdida** es un mapeo $L : \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$.

Dado un valor real $y$ y una prediccion $\hat{y} = f(x; \theta)$, la funcion $L(y, \hat{y})$ cuantifica la discrepancia entre ambos.

El objetivo del entrenamiento es encontrar:

$$\theta^* = \arg\min_\theta \; \mathbb{E}_{(x,y) \sim P} \left[ L(y, f(x; \theta)) \right]$$

Como $P$ es desconocida, aproximamos con el **riesgo empirico**:

$$\theta^* = \arg\min_\theta \; \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i; \theta))$$

Este es el principio de **Minimizacion del Riesgo Empirico (ERM)**.

### Jerarquia: Loss vs Cost vs Objective

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
| **Loss Function** $L$ | Un **solo** ejemplo | $L(y_i, \hat{y}_i)$ |
| **Cost Function** $J$ | El **agregado** sobre todo el dataset | $J(\theta) = \frac{1}{N} \sum L(y_i, \hat{y}_i)$ |
| **Objective Function** $F$ | Cost + Regularizacion | $F(\theta) = J(\theta) + \lambda R(\theta)$ |

---

## 2. Funciones de Perdida para Regresion

### 2.1 Mean Squared Error (MSE) / L2 Loss

**Perdida por ejemplo:**

$$L(y, \hat{y}) = (y - \hat{y})^2$$

**Funcion de costo:**

$$J(\theta) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**Gradiente respecto a $\hat{y}$:**

$$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$$

**Derivacion desde Maximum Likelihood:** Si asumimos ruido Gaussiano $y = \hat{y} + \epsilon$ con $\epsilon \sim \mathcal{N}(0, \sigma^2)$, el negativo del log-likelihood es:

$$-\log P(y|x,\theta) = \frac{(y - \hat{y})^2}{2\sigma^2} + \text{constante}$$

Minimizar el NLL es equivalente a minimizar MSE.

**Propiedades:**
- Penaliza fuertemente los errores grandes (crecimiento cuadratico) -- **sensible a outliers**
- Suave, diferenciable en todo punto, convexa respecto a $\hat{y}$
- El minimizador unico es la **media condicional** $\mathbb{E}[y|x]$

### 2.2 Mean Absolute Error (MAE) / L1 Loss

$$L(y, \hat{y}) = |y - \hat{y}|$$

- **Robusta a outliers** (penalidad lineal, no cuadratica)
- No diferenciable en $y = \hat{y}$
- El minimizador es la **mediana condicional** de $y|x$

### 2.3 Huber Loss (Smooth L1)

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{si } |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{si } |y - \hat{y}| > \delta \end{cases}$$

Lo mejor de ambos mundos: cuadratica para errores pequenos, lineal para grandes.

### Tabla comparativa

| Propiedad | MSE (L2) | MAE (L1) | Huber |
|---|---|---|---|
| Sensibilidad a outliers | Alta | Baja | Baja |
| Diferenciable | Si | No en 0 | Si |
| Optimo que busca | Media | Mediana | Depende de $\delta$ |

---

## 3. Funciones de Perdida para Clasificacion

### 3.1 Binary Cross-Entropy (BCE) / Log Loss

Para clasificacion binaria donde $y \in \{0, 1\}$ y $\hat{y} = \sigma(z) \in (0, 1)$:

$$L(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

**Gradiente respecto al logit $z$** (donde $\hat{y} = \sigma(z)$):

{{< math-formula title="Gradiente simplificado BCE + Sigmoid" >}}
\frac{\partial L}{\partial z} = \hat{y} - y
{{< /math-formula >}}

{{< concept-alert type="clave" >}}
**Resultado elegante**: $\frac{\partial L}{\partial z} = \hat{y} - y$. Esta simplicidad es una razon fundamental por la cual sigmoid + BCE se usan juntos.
{{< /concept-alert >}}

### 3.2 Categorical Cross-Entropy (CCE)

Para clasificacion multiclase con $C$ clases, $y$ one-hot encoded, $\hat{y}$ salida de softmax:

$$L(y, \hat{y}) = -\sum_{c=1}^{C} y_c \log(\hat{y}_c) = -\log(\hat{y}_k) \quad \text{donde } k \text{ es la clase verdadera}$$

### 3.3 Hinge Loss (SVM)

$$L(y, z) = \max(0, 1 - y \cdot z)$$

---

## 4. Softmax + Cross-Entropy: La Combinacion Canonica

### La funcion Softmax

$$\hat{y}_c = \text{softmax}(z)_c = \frac{\exp(z_c)}{\sum_{j=1}^{C} \exp(z_j)}$$

### Gradiente simplificado

{{< math-formula title="Gradiente Softmax + Cross-Entropy" >}}
\frac{\partial L}{\partial z} = \hat{y} - y
{{< /math-formula >}}

### Ejemplo numerico completo

Red con 3 clases. Logits: $z = [2.0, 1.0, 0.1]$. Clase verdadera: $k = 0$.

```
Paso 1 - Softmax:
  exp(z) = [7.389, 2.718, 1.105]
  SUM = 11.212
  y_hat = [0.659, 0.242, 0.099]

Paso 2 - Cross-Entropy Loss:
  L = -log(0.659) = 0.417

Paso 3 - Gradiente:
  dL/dz = y_hat - y = [0.659, 0.242, 0.099] - [1, 0, 0] = [-0.341, 0.242, 0.099]
```

---

## 5. Propiedades de una Buena Funcion de Perdida

- **Convexidad**: Garantiza que todo minimo local es global (para modelos lineales; redes neuronales son no-convexas)
- **Diferenciabilidad**: Requerida para backpropagation
- **Acotacion inferior**: $L \geq 0$ siempre
- **Fisher consistency**: Minimizar la perdida sobre la distribucion real produce el clasificador Bayes-optimo
- **Calibracion**: Cross-entropy es una **proper scoring rule**

---

## 6. Regularizacion: L1, L2 y Elastic Net

La funcion objetivo total: $F(\theta) = \frac{1}{N} \sum_i L(y_i, f(x_i; \theta)) + \lambda R(\theta)$

### 6.1 L2 Regularizacion (Ridge / Weight Decay)

$$R(\theta) = \frac{1}{2} \|\theta\|_2^2$$

La actualizacion se convierte en:

$$\theta_j \leftarrow \theta_j (1 - \eta\lambda) - \eta \frac{\partial J}{\partial \theta_j}$$

El factor $(1 - \eta\lambda)$ **encoge multiplicativamente** los pesos en cada paso -- "weight decay".

**Interpretacion Bayesiana:** Equivale a un **prior Gaussiano** sobre los pesos.

### 6.2 L1 Regularizacion (Lasso)

$$R(\theta) = \|\theta\|_1$$

Induce **sparsity** -- lleva muchos pesos a exactamente cero. Equivale a un **prior de Laplace**.

### 6.3 Elastic Net (L1 + L2)

$$R(\theta) = \alpha \|\theta\|_1 + \frac{1-\alpha}{2} \|\theta\|_2^2$$

| Regularizador | Sparsity | Diferenciable | Prior |
|---|---|---|---|
| **L2 (Ridge)** | No (encoge, nunca cero) | Si | Gaussiano |
| **L1 (Lasso)** | Si (ceros exactos) | No (en 0) | Laplace |
| **Elastic Net** | Si (parcial) | No (en 0) | Mixto |

---

# Parte II: Backpropagation

---

## 7. Contexto Historico

> **Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986)**. "Learning representations by back-propagating errors." *Nature*, 323, 533-536.

Backpropagation proporciono un metodo eficiente para computar el gradiente de la funcion de perdida respecto a **cada peso** en la red, independientemente de la profundidad, mediante la aplicacion sistematica de la **regla de la cadena**.

---

## 8. La Regla de la Cadena (Chain Rule)

Si $y = f(g(x))$, donde $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

### Regla de la cadena multivariada

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

**Esta es la base matematica de backpropagation**: la perdida depende de las salidas, que dependen de las activaciones ocultas, que dependen de los pesos.

### Ejemplo concreto

Sea $L = (y - t)^2$, $y = \sigma(z)$, $z = wx + b$:

$$\frac{\partial L}{\partial w} = \underbrace{2(y - t)}_{\partial L / \partial y} \cdot \underbrace{\sigma(z)(1 - \sigma(z))}_{\partial y / \partial z} \cdot \underbrace{x}_{\partial z / \partial w}$$

---

## 9. Grafos Computacionales

Una red neuronal se representa como un **Grafo Aciclico Dirigido (DAG)**:
- **Nodos** = operaciones o variables
- **Aristas** = flujo de datos

1. **Forward pass**: evalua nodos en orden topologico
2. **Backward pass**: recorre en orden inverso, acumulando gradientes via chain rule
3. Cada nodo solo necesita conocer su **derivada local**

---

## 10. Forward Pass: Ejemplo Numerico Completo

Red simple de 2 capas: Input $x_1 = 0.5, x_2 = 0.3$, capa oculta con 2 neuronas (sigmoid), salida con 1 neurona (sigmoid), target $t = 1.0$.

```
Pesos capa oculta:     W^(1) = [[0.15, 0.20], [0.25, 0.30]], b^(1) = [0.35, 0.35]
Pesos capa salida:     W^(2) = [0.40, 0.45], b^(2) = 0.60

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

---

## 11. Backward Pass: Derivacion Paso a Paso

```
Paso 1: dL/dy = y - t = -0.2435

Paso 2: dy/dz = y(1-y) = 0.1842
         delta^(2) = (-0.2435)(0.1842) = -0.04484

Paso 3: dL/dw_31 = delta^(2) * h_1 = -0.02776
         dL/dw_32 = delta^(2) * h_2 = -0.02859

Paso 4: dL/dh_1 = delta^(2) * w_31 = -0.01794
         dL/dh_2 = delta^(2) * w_32 = -0.02018

Paso 5: delta_1^(1) = -0.01794 * 0.6190 * 0.3810 = -0.004231
         delta_2^(1) = -0.02018 * 0.6376 * 0.3624 = -0.004662

Paso 6: dL/dw_11 = delta_1^(1) * x_1 = -0.002116
         dL/dw_12 = delta_1^(1) * x_2 = -0.001269
         dL/dw_21 = delta_2^(1) * x_1 = -0.002331
         dL/dw_22 = delta_2^(1) * x_2 = -0.001399
```

**Observacion clave**: Los gradientes de la capa oculta son ~10x mas pequenos que los de la capa de salida -- esto es el inicio del problema de **vanishing gradients**.

---

## 12. Flujo de Gradientes y la Matriz Jacobiana

### Patron general para una red de L capas

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \cdot \left(a^{(l-1)}\right)^T$$

Donde la senal de error se propaga recursivamente:

$$\delta^{(l)} = \left( \left(W^{(l+1)}\right)^T \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})$$

El gradiente de la red completa es un **producto de Jacobianos transpuestos**:

$$\frac{\partial L}{\partial x} = J_{f_1}^T \cdot J_{f_2}^T \cdots J_{f_K}^T \cdot \frac{\partial L}{\partial y}$$

---

## 13. Vanishing y Exploding Gradients

### Vanishing Gradients con sigmoid

El valor maximo de $\sigma'(z)$ es **0.25**. Despues de $K$ capas:

$$\|\text{gradiente en capa 1}\| \leq (0.25)^K \cdot \|\text{gradiente en salida}\|$$

| Capa | Magnitud del gradiente |
|---|---|
| 10 (salida) | 1.0 |
| 9 | 0.25 |
| 7 | 0.01563 |
| 5 | 0.000977 |
| 1 | **0.00000095** |

### Soluciones

| Problema | Solucion | Como funciona |
|---|---|---|
| Vanishing | **ReLU** | Derivada = 1 para $z > 0$ |
| Vanishing | **Conexiones residuales** (ResNets) | $\frac{\partial}{\partial x}(x + F(x)) = 1 + F'(x)$ |
| Vanishing | **LSTM / GRU gates** | Preserva gradientes en el tiempo |
| Exploding | **Gradient clipping** | Recorta la norma del gradiente |
| Ambos | **Inicializacion Xavier/He** | Preserva varianza entre capas |
| Ambos | **Batch Normalization** | Re-centra y re-escala activaciones |

---

## 14. Diferenciacion Automatica

Backpropagation es un caso especifico de **diferenciacion automatica en modo reverso (reverse-mode AD)**.

| Propiedad | Forward Mode | Reverse Mode (Backprop) |
|---|---|---|
| Direccion | Input --> Output | Output --> Input |
| Un pase da | $\partial(\text{outputs})/\partial(\text{un input})$ | $\partial(\text{un output})/\partial(\text{todos inputs})$ |
| Redes neuronales | Inviable | Ideal ($m=1$) |
| Memoria | Baja | **Alta** (almacena activaciones) |

> **Trade-off de memoria**: El modo reverso requiere almacenar todas las activaciones intermedias del forward pass. Tecnicas como **gradient checkpointing** sacrifican computo adicional para reducir memoria.

---

## 15. Pseudocodigo Completo de Backpropagation

```
ALGORITMO: Backpropagation para Red Fully Connected
====================================================

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

# Parte III: Learning Rate y Paisajes de Optimizacion

---

## 16. El Paisaje de Optimizacion

### Optimizacion Convexa vs No-Convexa

Una funcion $f$ es **convexa** si:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$$

**Las redes neuronales son no-convexas** -- implica multiples minimos locales, saddle points, y sin garantia de encontrar el minimo global.

### Por que los Saddle Points son mas problematicos que los Minimos Locales

En un punto critico con $n$ eigenvalues del Hessiano:

$$P(\text{minimo local}) \approx \left(\frac{1}{2}\right)^n$$

Con $n = 1{,}000{,}000$ parametros, esta probabilidad es astronomicamente pequena. Los **saddle points vastamente superan en numero** a los minimos locales.

---

## 17. Teoria Formal del Learning Rate

### L-suavidad

Una funcion $f$ es L-smooth si:

$$\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|$$

### El Lema de Descenso

{{< math-formula title="Lema de Descenso" >}}
f(x - \eta \nabla f(x)) \leq f(x) - \eta \left(1 - \frac{L\eta}{2}\right) \|\nabla f(x)\|^2
{{< /math-formula >}}

{{< concept-alert type="clave" >}}
Para que la funcion decrezca se requiere $\eta < \frac{2}{L}$. El paso optimo fijo es $\eta = \frac{1}{L}$.
{{< /concept-alert >}}

### Tasas de convergencia con $\eta = 1/L$

| Escenario | Tasa | Iteraciones para $\epsilon$-precision |
|---|---|---|
| No-convexa, L-smooth | $\min \|\nabla f\|^2 = O(1/t)$ | $O(1/\epsilon)$ |
| Convexa, L-smooth | $f(x_t) - f^* = O(1/t)$ | $O(1/\epsilon)$ |
| $\mu$-fuertemente convexa | $f(x_t) - f^* = O((1-\mu/L)^t)$ | $O(\kappa \log(1/\epsilon))$ |

El ratio $\kappa = L/\mu$ es el **numero de condicion**.

---

## 18. Estrategias de Learning Rate Scheduling

### 18.1 Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor}$$

### 18.2 Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

### 18.3 Warmup

$$\eta_t = \eta_{\text{target}} \cdot \frac{t}{T_{\text{warmup}}} \quad \text{para } t < T_{\text{warmup}}$$

**Por que es necesario:**
1. Estabiliza el entrenamiento temprano
2. Permite LRs objetivo mas altos
3. Critico para Transformers
4. Esencial para batches grandes

### 18.4 One-Cycle Policy (Leslie Smith, 2018)

```
LR
 ^
 |         /\
 |        /  \
 |       /    \
 |      /      \
 |     /        \_____
 +---|----|----|----|--> Epochs
     Warmup  Decay  Aniquilacion
```

Redes entrenadas con 1cycle alcanzan la misma precision en **1/5 a 1/10** de las epocas.

---

## 19. Inicializacion de Pesos

### Xavier/Glorot (2010)

$$\text{Var}(W) = \frac{2}{\text{fan\_in} + \text{fan\_out}}$$

### He/Kaiming (2015)

$$\text{Var}(W) = \frac{2}{\text{fan\_in}}$$

La inicializacion determina el punto de partida en la superficie de perdida. Una mala inicializacion hace que cualquier LR falle.

---

## 20. Batch Size y Learning Rate

### Regla de Escalamiento Lineal (Goyal et al., 2017)

$$\text{lr\_new} = \text{lr\_base} \cdot \frac{B_{\text{new}}}{B_{\text{base}}}$$

**Argumento de reduccion de varianza:** $\text{Var}(g_B) = \text{Var}(\nabla L_i) / B$. Duplicar el batch reduce la varianza a la mitad, permitiendo pasos mas grandes.

**Escala de ruido SGD:** $\text{noise} \sim \sqrt{\text{lr}/B} \cdot \sigma$. Para mantener el mismo nivel de ruido al aumentar $B$ por $k$, debemos aumentar lr por $k$.

---

## 21. Papers Fundamentales

| Ano | Paper | Contribucion |
|---|---|---|
| 1986 | Rumelhart, Hinton, Williams | Backpropagation |
| 2010 | Glorot & Bengio | Inicializacion Xavier |
| 2014 | Dauphin et al. | Saddle points > minimos locales |
| 2015 | He et al. | Inicializacion Kaiming |
| 2017 | Goyal et al. | Escalamiento lineal batch/lr |
| 2017 | Loshchilov & Hutter | Cosine Annealing / SGDR |
| 2018 | Li et al. | Visualizacion paisajes de perdida |
| 2018 | Leslie Smith | Super-Convergence / 1cycle |

---

## Diagrama de Relaciones

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
```
