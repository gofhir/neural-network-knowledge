# Clase 10 - Raices Matematicas e Historia de la Optimizacion en Deep Learning

**Diplomado Inteligencia Artificial - UC**
**Documento complementario historico-matematico**
**Fecha:** 2026-04-10

> Este documento traza la linea historica completa: desde el calculo de Euler y Lagrange
> en el siglo XVIII, pasando por Cauchy inventando gradient descent en 1847 para astronomia,
> hasta Adam en 2015. Cada idea nacio como solucion a un problema concreto de su epoca.

---

## Tabla de Contenidos

**Parte I: Las Raices Matematicas (1669-1964)**
1. [Newton y la Busqueda de Raices (1669)](#1-newton-y-la-busqueda-de-raices-1669)
2. [Euler, Lagrange y el Calculo de Variaciones (1744-1788)](#2-euler-lagrange-y-el-calculo-de-variaciones-1744-1788)
3. [Gauss y los Minimos Cuadrados (1809)](#3-gauss-y-los-minimos-cuadrados-1809)
4. [Cauchy y el Nacimiento del Gradient Descent (1847)](#4-cauchy-y-el-nacimiento-del-gradient-descent-1847)
5. [Demostracion: Por que el Negativo del Gradiente es la Direccion Optima](#5-demostracion-por-que-el-negativo-del-gradiente-es-la-direccion-optima)
6. [Metodos de Segundo Orden: Newton, Gauss-Newton, Levenberg-Marquardt](#6-metodos-de-segundo-orden)
7. [Multiplicadores de Lagrange y Optimizacion Restringida](#7-multiplicadores-de-lagrange-y-optimizacion-restringida)
8. [Optimizacion Convexa: de Dantzig a Boyd](#8-optimizacion-convexa-de-dantzig-a-boyd)

**Parte II: El Nacimiento de las Redes Neuronales (1943-1989)**
9. [McCulloch-Pitts: La Primera Neurona Matematica (1943)](#9-mcculloch-pitts-la-primera-neurona-matematica-1943)
10. [Hebb: La Primera Regla de Aprendizaje (1949)](#10-hebb-la-primera-regla-de-aprendizaje-1949)
11. [Robbins-Monro: Los Cimientos de SGD (1951)](#11-robbins-monro-los-cimientos-de-sgd-1951)
12. [Rosenblatt y el Perceptron (1958)](#12-rosenblatt-y-el-perceptron-1958)
13. [Minsky-Papert: El Invierno de la IA (1969)](#13-minsky-papert-el-invierno-de-la-ia-1969)
14. [Werbos: Backpropagation en una Tesis Doctoral (1974)](#14-werbos-backpropagation-en-una-tesis-doctoral-1974)
15. [Hopfield y la Fisica Estadistica (1982)](#15-hopfield-y-la-fisica-estadistica-1982)
16. [Rumelhart, Hinton, Williams: La Popularizacion (1986)](#16-rumelhart-hinton-williams-la-popularizacion-1986)
17. [El Teorema de Aproximacion Universal (1989-1991)](#17-el-teorema-de-aproximacion-universal-1989-1991)
18. [LeCun y las Redes Convolucionales (1989)](#18-lecun-y-las-redes-convolucionales-1989)

**Parte III: La Evolucion de los Optimizadores (1964-2018)**
19. [Polyak y el Heavy Ball (1964)](#19-polyak-y-el-heavy-ball-1964)
20. [Nesterov y la Aceleracion Optima (1983)](#20-nesterov-y-la-aceleracion-optima-1983)
21. [Momentum como Ecuacion Diferencial](#21-momentum-como-ecuacion-diferencial)
22. [AdaGrad: Learning Rates Adaptativos (2011)](#22-adagrad-learning-rates-adaptativos-2011)
23. [RMSProp: Nacido en una Clase de Coursera (2012)](#23-rmsprop-nacido-en-una-clase-de-coursera-2012)
24. [Adam: Momentos Adaptativos (2014)](#24-adam-momentos-adaptativos-2014)
25. [Los Breakthroughs que Terminaron el Invierno de la IA](#25-los-breakthroughs-que-terminaron-el-invierno-de-la-ia)
26. [El Gran Debate: Adam vs SGD](#26-el-gran-debate-adam-vs-sgd)

**Apendice**
- [Linea de Tiempo Completa](#linea-de-tiempo-completa)
- [El Arco Narrativo de la Optimizacion](#el-arco-narrativo-de-la-optimizacion)

---

# PARTE I: LAS RAICES MATEMATICAS (1669-1964)

---

## 1. Newton y la Busqueda de Raices (1669)

Isaac Newton describio un metodo iterativo para encontrar raices de polinomios en *De analysi* (escrito 1669, publicado 1711). Joseph Raphson publico una version simplificada en 1690.

### El metodo Newton-Raphson

Para encontrar la raiz de `g(x) = 0`:

```
x^(k+1) = x^(k) - g(x^(k)) / g'(x^(k))
```

**Interpretacion geometrica:** En cada paso, aproximar `g` por su recta tangente y encontrar donde cruza cero.

### Conexion con la optimizacion

Para **minimizar** `f(x)`, la condicion necesaria es `f'(x*) = 0`. Aplicando Newton-Raphson a `g(x) = f'(x)`:

```
x^(k+1) = x^(k) - f'(x^(k)) / f''(x^(k))
```

### Version multivariada

Para `f: R^n --> R`, la condicion es `nabla f(x*) = 0`. Usando la expansion de Taylor de segundo orden:

```
x^(k+1) = x^(k) - [H(x^(k))]^{-1} * nabla f(x^(k))
```

Donde `H` es la **matriz Hessiana** (segundas derivadas parciales):
```
H_{ij} = d^2f / (dx_i * dx_j)
```

**Convergencia cuadratica** cerca de la solucion:
```
||x^(k+1) - x*|| <= C * ||x^(k) - x*||^2
```

Dramaticamente mas rapido que gradient descent (convergencia lineal), pero cada iteracion cuesta `O(n^3)` por la inversion de la Hessiana -- impracticable para redes neuronales con millones de parametros.

### El Decremento de Newton

```
lambda(x) = (nabla f(x)^T * H(x)^{-1} * nabla f(x))^{1/2}
```

Mide la proximidad al optimo: `(1/2)*lambda^2` aproxima `f(x) - f(x*)`.

---

## 2. Euler, Lagrange y el Calculo de Variaciones (1744-1788)

### El problema que inicio todo

**Johann Bernoulli (1696)** planteo el **problema de la braquistocrona**: encontrar la curva de descenso mas rapido bajo gravedad entre dos puntos. Este problema lanzo el campo del calculo de variaciones.

### La pregunta fundamental

Entre todas las funciones `y(x)` que satisfacen ciertas condiciones de frontera, cual minimiza un **funcional**?

```
J[y] = INTEGRAL_a^b L(x, y(x), y'(x)) dx
```

Donde `L` es el **Lagrangiano** (la funcion dentro de la integral).

### Derivacion de la Ecuacion de Euler-Lagrange

**Paso 1 - Variacion:** Sea `y*(x)` la funcion optima. Consideramos perturbaciones:
```
y_tilde(x) = y*(x) + epsilon * eta(x)
```
donde `eta(a) = eta(b) = 0` y `epsilon` es pequeno.

**Paso 2 - Condicion necesaria:** Definimos `Phi(epsilon) = J[y* + epsilon*eta]`. Para que `y*` sea minimizador:
```
d(Phi)/d(epsilon) |_{epsilon=0} = 0    para toda eta admisible
```

**Paso 3 - Calcular la derivada:**
```
d(Phi)/d(epsilon)|_0 = INTEGRAL_a^b [ dL/dy * eta + dL/dy' * eta' ] dx
```

**Paso 4 - Integracion por partes** del segundo termino:
```
INTEGRAL dL/dy' * eta' dx = [dL/dy' * eta]_a^b - INTEGRAL d/dx(dL/dy') * eta dx
```

El termino de frontera desaparece porque `eta(a) = eta(b) = 0`.

**Paso 5 - Combinar:**
```
INTEGRAL_a^b [ dL/dy - d/dx(dL/dy') ] * eta(x) dx = 0
```

**Paso 6 - Lema fundamental:** Como esto debe valer para toda `eta`, el integrando debe ser cero:

```
+-----------------------------------------+
|  dL/dy - d/dx(dL/dy') = 0               |
|                                          |
|  LA ECUACION DE EULER-LAGRANGE           |
+-----------------------------------------+
```

### Conexion con la optimizacion moderna

La ecuacion de Euler-Lagrange es el **analogo en dimension infinita** de poner `nabla f = 0`:

| Dimension finita | Dimension infinita |
|---|---|
| Variable `x in R^n` | Funcion `y(x)` |
| Funcion `f(x)` | Funcional `J[y]` |
| `nabla f = 0` | Ecuacion de Euler-Lagrange |
| Gradient descent | Flujo de gradiente / metodos PDE |

Conexiones modernas:
- **Neural ODEs** (Chen et al., 2018): redes neuronales como sistemas dinamicos continuos
- **Transporte optimo** (Monge 1781, Kantorovich 1942): formulado como problema variacional
- **Physics-Informed Neural Networks (PINNs)**: imponen ecuaciones diferenciales (derivadas de Euler-Lagrange) en la loss function

---

## 3. Gauss y los Minimos Cuadrados (1809)

Carl Friedrich Gauss desarrollo el **metodo de minimos cuadrados** en *Theoria motus corporum coelestium* (1809), motivado por determinar orbitas planetarias a partir de observaciones astronomicas. (Legendre lo publico en 1805, generando una disputa de prioridad.)

### Formulacion

Dadas observaciones `(t_i, y_i)` y un modelo `y_hat(t; theta)`, minimizar:

```
S(theta) = (1/2) SUM_{i=1}^{m} r_i(theta)^2 = (1/2) ||r(theta)||^2
```

donde `r_i(theta) = y_i - y_hat(t_i; theta)` son los **residuos**.

### Conexion directa con MSE

La funcion de costo MSE usada en redes neuronales es exactamente el metodo de Gauss:
```
J(theta) = (1/N) SUM (y_i - f(x_i; theta))^2
```

Gauss lo uso para orbitas planetarias. Nosotros lo usamos para entrenar redes neuronales. La matematica es la misma -- lo que cambio es la escala y la complejidad del modelo `f`.

---

## 4. Cauchy y el Nacimiento del Gradient Descent (1847)

### El paper que lo empezo todo

Augustin-Louis Cauchy publico **"Methode generale pour la resolution des systemes d'equations simultanees"** en *Comptes Rendus de l'Academie des Sciences* (vol. 25, pp. 536-538, 1847). Solo 3 paginas que cambiaron la historia de la optimizacion.

### El problema que Cauchy intentaba resolver

Cauchy NO trabajaba en machine learning -- intentaba resolver sistemas de **ecuaciones no-lineales simultaneas** que surgian de la **mecanica celeste y astronomia**:

```
f_1(x_1, x_2, ..., x_n) = 0
f_2(x_1, x_2, ..., x_n) = 0
...
f_m(x_1, x_2, ..., x_n) = 0
```

### La reformulacion clave de Cauchy

Su insight crucial: **convertir el problema de buscar raices en un problema de minimizacion**:

```
F(x_1, ..., x_n) = SUM_{i=1}^{m} f_i(x_1, ..., x_n)^2
```

Observacion: `F >= 0` siempre, y `F = 0` si y solo si todas las ecuaciones se satisfacen. Asi, encontrar la solucion del sistema equivale a encontrar el **minimo** de `F`.

### El metodo iterativo de Cauchy

Dada una aproximacion actual, computar la siguiente:

```
x_j^(k+1) = x_j^(k) - alpha * dF/dx_j |_{x^(k)}
```

En notacion vectorial moderna:

```
x^(k+1) = x^(k) - alpha * nabla F(x^(k))
```

Cauchy argumento que eligiendo `alpha` suficientemente pequeno, se garantiza:

```
F(x^(k+1)) < F(x^(k))
```

### Significado historico

Este fue el **primer algoritmo explicito de optimizacion iterativa** usando informacion de derivadas. Cauchy entendio todos los ingredientes clave:
- Conversion de busqueda de raices a minimizacion
- Uso de derivadas parciales para determinar la direccion de descenso
- El rol del tamano del paso en garantizar convergencia
- La naturaleza iterativa del procedimiento

> **La ecuacion de Cauchy de 1847 es, en esencia, exactamente lo que `loss.backward(); optimizer.step()` hace en PyTorch hoy.**

---

## 5. Demostracion: Por que el Negativo del Gradiente es la Direccion Optima

Esta es la justificacion matematica rigurosa de por que nos movemos en la direccion `- nabla f`.

### Teorema

Entre todos los vectores unitarios `d` con `||d|| = 1`, la derivada direccional `D_d f(x)` es minimizada cuando `d = -nabla f(x) / ||nabla f(x)||`.

### Demostracion

**Paso 1: Derivada direccional.**

La derivada direccional de `f: R^n --> R` en el punto `x` en la direccion `d` (con `||d|| = 1`) es:

```
D_d f(x) = lim_{t->0} [f(x + t*d) - f(x)] / t = nabla f(x)^T * d
```

Esto mide la tasa de cambio de `f` cuando nos movemos en la direccion `d`.

**Paso 2: Aplicar la desigualdad de Cauchy-Schwarz.**

Para cualesquiera vectores `a, b in R^n`:
```
|a^T * b| <= ||a|| * ||b||
```

con igualdad si y solo si `a = lambda * b` para algun escalar `lambda`.

Aplicando con `a = nabla f(x)` y `b = d`:
```
|nabla f(x)^T * d| <= ||nabla f(x)|| * ||d|| = ||nabla f(x)||
```

ya que `||d|| = 1`. Esto nos da:
```
-||nabla f(x)|| <= nabla f(x)^T * d <= ||nabla f(x)||
```

**Paso 3: Alcanzar la cota inferior.**

La derivada direccional alcanza su **valor minimo** `-||nabla f(x)||` cuando:
```
nabla f(x)^T * d = -||nabla f(x)||
```

Por la condicion de igualdad de Cauchy-Schwarz, esto ocurre cuando `d = lambda * nabla f(x)` con `lambda < 0`. Combinado con `||d|| = 1`:

```
+---------------------------------------------------+
|  d* = -nabla f(x) / ||nabla f(x)||                |
|                                                     |
|  La direccion de maximo descenso es el NEGATIVO     |
|  del gradiente normalizado.                  QED    |
+---------------------------------------------------+
```

### Perspectiva desde Taylor

Por el teorema de Taylor:
```
f(x + alpha*d) = f(x) + alpha * nabla f(x)^T * d + O(alpha^2)
```

Para `alpha > 0` pequeno, el cambio en `f` es aproximadamente `alpha * nabla f(x)^T * d`. Para hacer esto lo mas negativo posible (mayor descenso), queremos minimizar `nabla f(x)^T * d` sujeto a `||d|| = 1`, lo cual da `d* = -nabla f / ||nabla f||` como se demostro arriba.

---

## 6. Metodos de Segundo Orden

### Metodo de Gauss-Newton

Para problemas de minimos cuadrados no-lineales, el gradiente y la Hessiana de `S(theta)` son:

```
nabla S = J^T * r
H_S = J^T * J + SUM_i r_i * nabla^2 r_i
```

donde `J` es el **Jacobiano** del vector de residuos: `J_{ij} = dr_i/d(theta_j)`.

**Aproximacion Gauss-Newton:** Descartar el termino de segundo orden:
```
H_S ~ J^T * J
```

El paso de actualizacion:
```
theta^(k+1) = theta^(k) - (J^T * J)^{-1} * J^T * r
```

### Levenberg-Marquardt (1944/1963)

Kenneth Levenberg (1944) y Donald Marquardt (1963) propusieron una version regularizada:

```
theta^(k+1) = theta^(k) - (J^T*J + lambda*I)^{-1} * J^T * r
```

El parametro de amortiguamiento `lambda >= 0` **interpola entre Gauss-Newton y gradient descent**:

| `lambda` | Comportamiento |
|---|---|
| `lambda -> 0` | Gauss-Newton (convergencia rapida cerca de la solucion) |
| `lambda -> infinito` | Gradient descent con paso pequeno (seguro lejos de la solucion) |

El algoritmo adapta `lambda` dinamicamente: **aumenta** cuando un paso falla, **disminuye** cuando tiene exito.

### Metodos Quasi-Newton: BFGS (1970)

Calcular la Hessiana completa es costoso. **BFGS** (Broyden-Fletcher-Goldfarb-Shanno, 1970) aproxima `H^{-1}` iterativamente usando solo informacion de gradientes.

**L-BFGS** (Limited-memory BFGS): almacena solo los ultimos `m` pares de vectores, reduciendo almacenamiento de `O(n^2)` a `O(m*n)`. Es el metodo dominante para optimizacion suave a gran escala (regresion logistica, CRFs).

---

## 7. Multiplicadores de Lagrange y Optimizacion Restringida

### Origen

Lagrange introdujo los multiplicadores en *Mecanique analytique* (1788) para resolver problemas de mecanica con restricciones.

### El metodo

**Problema:** Minimizar `f(x)` sujeto a `g_i(x) = 0` para `i = 1, ..., m`.

**Lagrangiano:**
```
L(x, lambda) = f(x) + SUM_{i=1}^{m} lambda_i * g_i(x)
```

**Condiciones necesarias:**
```
nabla_x L = nabla f(x) + SUM_i lambda_i * nabla g_i(x) = 0
g_i(x) = 0,    i = 1, ..., m
```

**Interpretacion geometrica:** En el optimo, `nabla f` debe ser combinacion lineal de los gradientes de las restricciones. Si no, existiria una direccion factible en la que `f` decrece.

### Condiciones KKT (Karush 1939, Kuhn-Tucker 1951)

Para desigualdades `g_i(x) <= 0`:

1. **Estacionariedad:** `nabla f(x*) + SUM_i mu_i * nabla g_i(x*) = 0`
2. **Factibilidad primal:** `g_i(x*) <= 0`
3. **Factibilidad dual:** `mu_i >= 0`
4. **Holgura complementaria:** `mu_i * g_i(x*) = 0`

### Aplicacion en ML: Support Vector Machines

El **SVM hard-margin** (Vapnik & Chervonenkis, 1963):

**Problema primal:**
```
min_{w,b} (1/2) ||w||^2    sujeto a    y_i(w^T*x_i + b) >= 1
```

**Problema dual** (sustituyendo las condiciones de primer orden):
```
max_alpha SUM_i alpha_i - (1/2) SUM_{i,j} alpha_i*alpha_j*y_i*y_j * x_i^T*x_j
sujeto a alpha_i >= 0, SUM_i alpha_i*y_i = 0
```

Los datos solo aparecen via productos internos `x_i^T * x_j`, habilitando el **kernel trick**: reemplazar `x_i^T * x_j` con `K(x_i, x_j)`.

---

## 8. Optimizacion Convexa: de Dantzig a Boyd

### Programacion Lineal: Simplex (Dantzig, 1947)

George Dantzig desarrollo el **metodo simplex** mientras trabajaba en logistica para la Fuerza Aerea de EE.UU. durante la WWII.

La region factible es un **politopo convexo**, y el optimo se alcanza en un **vertice**. El simplex camina por las aristas del politopo, mejorando el objetivo en cada paso.

### Metodo del Elipsoide (Khachiyan, 1979)

Probo que la programacion lineal es resoluble en **tiempo polinomial** -- teoricamente importante pero lento en practica.

### Metodos de Punto Interior (Karmarkar, 1984)

En vez de caminar por el borde del politopo (simplex), moverse por el **interior** usando una barrera logaritmica:

```
min_x  t * c^T*x - SUM_i ln(x_i)    sujeto a    A*x = b
```

La barrera `-SUM ln(x_i)` previene que los iterados toquen la frontera. Al `t -> infinito`, la solucion se aproxima al optimo.

### Nesterov y la Aceleracion (1983)

Ya en el contexto de optimizacion convexa, Nesterov demostro el gradiente acelerado que logra `O(1/k^2)` -- el limite teorico para metodos de primer orden. (Detallado en la Parte III.)

### Boyd & Vandenberghe (2004)

Su libro *Convex Optimization* unifico el campo y lo hizo accesible. Introdujeron **Disciplined Convex Programming (DCP)**: reglas de composicion para verificar convexidad automaticamente.

---

# PARTE II: EL NACIMIENTO DE LAS REDES NEURONALES (1943-1989)

---

## 9. McCulloch-Pitts: La Primera Neurona Matematica (1943)

**Paper:** McCulloch & Pitts, "A Logical Calculus of the Ideas Immanent in Nervous Activity," *Bull. Math. Biophys.*, 1943.

### Formulacion

La primera neurona artificial como unidad de umbral logico binario:

```
y = theta( SUM_i w_i * x_i - T )
```

Donde:
- `x_i in {0, 1}` = entradas binarias
- `w_i in {-1, +1}` = pesos **fijos** (no aprendidos) -- excitatorios (+1) o inhibitorios (-1)
- `T` = umbral
- `theta(z)` = funcion escalon de Heaviside: `theta(z) = 1 si z >= 0, sino 0`

### Resultado clave

Demostraron que redes de estas neuronas pueden computar **cualquier proposicion de logica proposicional** (AND, OR, NOT), haciendolas Turing-completas cuando se organizan con retroalimentacion.

**Pero:** los pesos eran fijos por diseno -- no habia algoritmo de aprendizaje.

---

## 10. Hebb: La Primera Regla de Aprendizaje (1949)

**Libro:** Donald Hebb, *The Organization of Behavior*, 1949.

### El postulado de Hebb

"Cuando un axon de la celula A esta suficientemente cerca para excitar a la celula B y repetidamente participa en disparar B, algun proceso de crecimiento ocurre de tal forma que la eficiencia de A para disparar B aumenta."

**Formulacion matematica (formalizada posteriormente):**

```
Delta w_{ij} = eta * x_i * y_j
```

- `Delta w_{ij}` = cambio en el peso sinaptico de neurona `i` a neurona `j`
- `eta > 0` = tasa de aprendizaje
- `x_i` = actividad pre-sinaptica
- `y_j` = actividad post-sinaptica

**Regla no-supervisada basada en correlacion**: fortalece conexiones entre neuronas simultaneamente activas.

### Problema

El aprendizaje Hebbiano puro es **inestable** -- los pesos crecen sin limite ya que `Delta w >= 0` siempre cuando ambas neuronas disparan. Esto llevo a modificaciones posteriores como la regla de Oja (1982) que efectivamente realiza PCA.

---

## 11. Robbins-Monro: Los Cimientos Matematicos de SGD (1951)

**Paper:** Robbins & Monro, "A Stochastic Approximation Method," *Annals of Mathematical Statistics*, 1951.

### El problema

Encontrar la raiz `theta*` de una funcion `M(theta) = E[Y|theta]` donde `M(theta)` no puede computarse directamente pero se pueden obtener observaciones ruidosas `Y`.

### El algoritmo

```
theta_{n+1} = theta_n - a_n * Y_n(theta_n)
```

### Las condiciones de convergencia de Robbins-Monro

La secuencia `{theta_n}` converge a `theta*` si:

```
1) SUM_n a_n = infinito          (asegura poder alcanzar theta* desde cualquier punto)
2) SUM_n a_n^2 < infinito        (asegura que el ruido se promedia eventualmente)
```

La eleccion clasica que satisface ambas: `a_n = c/n`.

### Conexion directa con SGD

SGD es una aplicacion directa de Robbins-Monro. Para minimizar `L(theta) = E[l(theta; X)]`, buscamos donde `nabla L(theta) = 0`. La actualizacion SGD:

```
theta_{n+1} = theta_n - a_n * nabla l(theta_n; X_n)
```

es exactamente Robbins-Monro con `M(theta) = nabla L(theta)` y `Y_n = nabla l(theta_n; X_n)` como observacion ruidosa.

### Lo que explican estas condiciones

- **Por que los schedules de LR importan:** Learning rates constantes violan `SUM a_n^2 < infinito` y previenen convergencia exacta (el iterado oscila alrededor del optimo)
- **Por que `1/n` o `1/sqrt(n)` decay estan motivados teoricamente**
- **En practica moderna**: se usan LR constantes (violando las condiciones) porque la convergencia exacta a un minimo no siempre es deseable (el ruido de SGD actua como regularizacion implicita)

---

## 12. Rosenblatt y el Perceptron (1958)

**Paper:** Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain," *Psychological Review*, 1958.

### El modelo

```
y = sign(w * x + b) = sign(SUM_i w_i * x_i + b)
```

### El algoritmo de aprendizaje del Perceptron

```
Inicializar w = 0
Repetir hasta convergencia:
    Para cada ejemplo (x^(t), d^(t)):
        Computar y_hat = sign(w * x^(t))
        Si y_hat != d^(t):
            w <-- w + d^(t) * x^(t)
```

### Teorema de Convergencia del Perceptron (sketch de la demostracion)

**Teorema:** Si los datos son linealmente separables, el algoritmo converge en un numero finito de actualizaciones.

**Demostracion:**

Suponer que existe `w*` con `||w*|| = 1` y margen `gamma > 0` tal que `d^(t) * (w* * x^(t)) >= gamma` para todo `t`. Sea `R = max_t ||x^(t)||`.

Rastrear dos cantidades despues de `k` actualizaciones por error:

**(a) Cota inferior de `w* * w_k`:**

En cada actualizacion sobre un punto mal clasificado:
```
w* * w_k = w* * w_{k-1} + d^(t)(w* * x^(t)) >= w* * w_{k-1} + gamma
```
Por induccion: `w* * w_k >= k * gamma`

**(b) Cota superior de `||w_k||^2`:**

Como el punto fue mal clasificado (`d^(t)(w_{k-1} * x^(t)) <= 0`):
```
||w_k||^2 = ||w_{k-1}||^2 + 2*d^(t)(w_{k-1} * x^(t)) + ||x^(t)||^2 <= ||w_{k-1}||^2 + R^2
```
Por induccion: `||w_k||^2 <= k * R^2`

**(c) Combinando:** Por Cauchy-Schwarz, `w* * w_k <= ||w*|| * ||w_k|| = ||w_k||`:
```
k * gamma <= ||w_k|| <= sqrt(k * R^2) = R * sqrt(k)
```

Por lo tanto: **`k <= R^2 / gamma^2`** -- el numero de actualizaciones es finito. QED.

---

## 13. Minsky-Papert: El Invierno de la IA (1969)

**Libro:** Minsky & Papert, *Perceptrons: An Introduction to Computational Geometry*, MIT Press, 1969.

### El problema XOR -- demostracion formal

XOR: `f(x_1, x_2) = x_1 XOR x_2`

| x_1 | x_2 | XOR |
|-----|-----|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Demostracion de no-separabilidad lineal:**

Suponer que existen `w_1, w_2, b` tales que `sign(w_1*x_1 + w_2*x_2 + b) = XOR(x_1, x_2)`:

De las cuatro restricciones:
```
(0,0) -> 0:  b <= 0                   ... (i)
(0,1) -> 1:  w_2 + b > 0              ... (ii)
(1,0) -> 1:  w_1 + b > 0              ... (iii)
(1,1) -> 0:  w_1 + w_2 + b <= 0       ... (iv)
```

Sumando (ii) y (iii): `w_1 + w_2 + 2b > 0`
De (iv): `w_1 + w_2 <= -b`
Entonces: `-b + 2b > 0` => `b > 0`
Pero de (i): `b <= 0`. **CONTRADICCION.** QED.

### Impacto

Aunque reconocieron que redes multicapa podrian superar estas limitaciones, expresaron escepticismo sobre encontrar algoritmos de aprendizaje efectivos. Esto contribuyo al **primer invierno de la IA** (~1969-1982), redirigiendo financiamiento hacia la IA simbolica.

---

## 14. Werbos: Backpropagation en una Tesis Doctoral (1974)

**Tesis:** Paul Werbos, "Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences," PhD, Harvard, 1974.

Werbos formulo lo que llamo "dynamic feedback" o el metodo de "derivadas ordenadas" -- esencialmente la regla de la cadena aplicada sistematicamente a traves de un grafo computacional.

### Su insight clave

Computo de `dL/dw_i` eficientemente aplicando la regla de la cadena **en orden reverso** a traves del grafo. Esto es exactamente el **metodo adjunto** de la teoria de control optimo, conexion que Werbos hizo explicita.

El costo: `O(W)` donde `W` es el numero de pesos, versus `O(W^2)` para calcular cada derivada por separado.

**Nota:** Backpropagation fue descubierto independientemente multiples veces:
- Linnainmaa (1970) -- en diferenciacion automatica
- Werbos (1974) -- para redes neuronales
- Parker (1985)
- Rumelhart, Hinton, Williams (1986) -- quienes lo popularizaron

---

## 15. Hopfield y la Fisica Estadistica (1982)

**Paper:** Hopfield, "Neural Networks and Physical Systems with Emergent Collective Computational Abilities," *PNAS*, 1982.

### El modelo

Red de `N` neuronas binarias con conexiones simetricas (`w_{ij} = w_{ji}`, `w_{ii} = 0`). Actualizacion asincrona:

```
s_i <-- sign(SUM_j w_{ij} * s_j - theta_i)
```

### La funcion de energia (insight clave)

```
E = -(1/2) SUM_i SUM_j w_{ij} * s_i * s_j + SUM_i theta_i * s_i
```

**Teorema:** Bajo actualizaciones asincronas, la energia `E` es **monotonamonte no-creciente**. Cada actualizacion o decrece `E` o la deja igual. Como `E` esta acotada (numero finito de estados), la red siempre converge a un **minimo local** = memoria almacenada.

### Conexion con la fisica estadistica

Las **Maquinas de Boltzmann** (Hinton & Sejnowski, 1983-1985) extendieron las redes de Hopfield con:
- Actualizaciones **estocasticas** via distribucion de Boltzmann:
```
P(s_i = 1) = sigma(SUM_j w_{ij}*s_j / T) = 1 / (1 + exp(-SUM_j w_{ij}*s_j / T))
```
- **Unidades ocultas** que aprenden representaciones internas

La distribucion de equilibrio sigue la distribucion de Boltzmann de la mecanica estadistica:
```
P(s) = (1/Z) * exp(-E(s)/T)
```

---

## 16. Rumelhart, Hinton, Williams: La Popularizacion (1986)

**Paper:** "Learning Representations by Back-propagating Errors," *Nature*, 323:533-536, 1986.

### Insights matematicos clave

**(a) Funciones de activacion diferenciables:** Reemplazar la funcion escalon con el **sigmoid**:
```
sigma(z) = 1 / (1 + e^(-z))
```
Con la derivada elegante: `sigma'(z) = sigma(z) * (1 - sigma(z))`

Esto hizo toda la red **diferenciable de extremo a extremo**.

**(b) La regla delta generalizada:**

Senal de error de la capa de salida:
```
delta_k = (y_k - t_k) * f'(net_k)
```

Senal de error de la capa oculta (el paso crucial de backpropagation):
```
delta_j = f'(net_j) * SUM_k w_{jk} * delta_k
```

Actualizacion de pesos:
```
Delta w_{ij} = -eta * delta_j * o_i
```

**(c) Demostracion empirica** de que backpropagation podia aprender representaciones internas. XOR resuelto trivialmente por una red 2-2-1.

**(d) Momentum:** Introdujeron el termino de momentum para acelerar convergencia:
```
Delta w_{ij}(t) = -eta * delta_j * o_i + alpha * Delta w_{ij}(t-1)
```

---

## 17. El Teorema de Aproximacion Universal (1989-1991)

### Cybenko (1989)

**Paper:** "Approximation by Superpositions of a Sigmoidal Function," *Math. Control Signals Syst.*, 1989.

**Enunciado formal:** Sea `sigma` cualquier funcion sigmoidal continua. Entonces las sumas finitas de la forma:

```
G(x) = SUM_{j=1}^{N} alpha_j * sigma(w_j^T * x + b_j)
```

son **densas** en `C(I_n)` (el espacio de funciones continuas en el cubo unitario n-dimensional) respecto a la norma supremo. Es decir, para cualquier `f in C(I_n)` y cualquier `epsilon > 0`, existen `N`, `alpha_j`, `w_j`, `b_j` tales que:

```
|G(x) - f(x)| < epsilon    para todo x in I_n
```

**Tecnica de demostracion:** Uso del teorema de Hahn-Banach y el teorema de representacion de Riesz. Por contradiccion: si las sumas no fueran densas, existiria una medida con signo no-nula `mu` que anula todas las sumas, y se demuestra que esto implica `mu = 0`.

### Hornik (1991)

Generalizo significativamente:
- La capacidad de aproximacion **no** es especial del sigmoid -- vale para virtualmente cualquier funcion de activacion no-constante, acotada y continua
- Es la **arquitectura multicapa misma** (no la activacion especifica) la que da el poder de aproximacion

### Limitaciones importantes

1. **Existencia, no construccion:** Garantiza que la red *existe* pero no dice como encontrarla
2. **Ancho vs profundidad:** El ancho `N` requerido puede ser exponencialmente grande. La profundidad logra compresion exponencial -- motivacion teorica para deep learning
3. **Sin garantia de generalizacion:** Aproximar en datos de entrenamiento no dice nada sobre test
4. **ReLU:** Leshno et al. (1993) mostraron que vale para cualquier activacion no-polinomial

---

## 18. LeCun y las Redes Convolucionales (1989)

**Paper:** LeCun et al., "Backpropagation Applied to Handwritten Zip Code Recognition," *Neural Computation*, 1989.

### Innovaciones clave

**(a) Convolucion y pesos compartidos:**
```
z^(l)_{i,j,k} = SUM_m SUM_p SUM_q w^(l)_{k,m,p,q} * a^(l-1)_{i+p, j+q, m} + b^(l)_k
```

Los mismos pesos del filtro se comparten en todas las posiciones espaciales. Reduce parametros dramaticamente: de ~10^6 (fully connected) a 25 por filtro (5x5).

**(b) Insight matematico:** La convolucion explota dos priors sobre datos visuales:
- **Conectividad local:** Pixeles interactuan principalmente con vecinos cercanos
- **Equivarianza traslacional:** `f(translate(x)) = translate(f(x))`

Esto evoluciono en **LeNet-5** (1998), la arquitectura CNN canonica.

---

# PARTE III: LA EVOLUCION DE LOS OPTIMIZADORES (1964-2018)

---

## 19. Polyak y el Heavy Ball (1964)

**Paper:** Boris Polyak, "Some methods of speeding up the convergence of iteration methods," 1964.

### La analogia fisica

Polyak lo llamo "heavy ball" (bola pesada) porque la iteracion modela una **bola pesada rodando** sobre la superficie definida por la funcion objetivo, sujeta a friccion. La bola tiene **inercia**: no se detiene inmediatamente cuando el gradiente es cero, sino que sobrepasa y oscila, eventualmente asentandose en el minimo.

### Regla de actualizacion

```
x_{k+1} = x_k - alpha * nabla f(x_k) + beta * (x_k - x_{k-1})
                  ^                        ^
            fuerza gravitacional       momentum (inercia)
```

- `alpha` = tamano del paso (fuerza gravitacional)
- `beta` = parametro de momentum (1 menos friccion)
- `(x_k - x_{k-1})` = termino de momentum

### Parametros optimos

Para funciones mu-fuertemente convexas con gradientes L-Lipschitz (numero de condicion `kappa = L/mu`):

```
alpha* = 4 / (sqrt(L) + sqrt(mu))^2
beta*  = ((sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu)))^2
```

### Tasa de convergencia

```
||x_k - x*|| <= ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^k * ||x_0 - x*||
```

El metodo requiere **O(sqrt(kappa))** iteraciones, comparado con **O(kappa)** para gradient descent estandar.

**Ejemplo:** Si `kappa = 10,000`:
- Gradient descent: ~10,000 iteraciones
- Heavy ball: ~100 iteraciones

Una mejora **dramatica** para problemas mal condicionados.

---

## 20. Nesterov y la Aceleracion Optima (1983)

**Paper:** Yurii Nesterov, "A method of solving a convex programming problem with convergence rate O(1/k^2)," *Soviet Mathematics Doklady*, 1983. Solo 4 paginas -- uno de los papers mas influyentes en teoria de optimizacion.

### Regla de actualizacion (NAG)

```
y_k     = x_k + ((k-1)/(k+2)) * (x_k - x_{k-1})
x_{k+1} = y_k - (1/L) * nabla f(y_k)
```

La diferencia clave con Polyak: el gradiente se evalua en el punto "lookahead" `y_k`, no en `x_k`. El coeficiente de momentum `(k-1)/(k+2)` no es constante sino que crece hacia 1.

### El insight de "mirar adelante"

- **Polyak:** computar gradiente en `x_k`, LUEGO agregar momentum
- **Nesterov:** PRIMERO dar paso de momentum a `y_k`, LUEGO evaluar gradiente ahi

Esta evaluacion anticipatoria da una "correccion" -- evaluamos el gradiente donde esperamos terminar, no donde estamos.

### Tasa de convergencia y optimalidad

Para funciones convexas con gradientes L-Lipschitz:

```
f(x_k) - f(x*) <= 2L * ||x_0 - x*||^2 / (k+1)^2 = O(1/k^2)
```

Comparacion:
- **Gradient descent:** `O(1/k)`
- **Nesterov:** `O(1/k^2)`

### Que significa "optimo"

**Nemirovski y Yudin (1983)** establecieron un marco de **complejidad de oraculos**. Modelaron la optimizacion como un juego: un algoritmo consulta un oraculo por valores de funcion y gradientes, y el oraculo puede ser adversarial. Demostraron:

```
Para CUALQUIER metodo de primer orden aplicado a funciones convexas L-smooth,
despues de k pasos:

    f(x_k) - f(x*) >= Omega(L * ||x_0 - x*||^2 / k^2)
```

**Nesterov iguala esta cota inferior.** Ningun metodo de primer orden puede hacerlo asintoticamente mejor. NAG alcanza el **limite teorico de la informacion** para metodos de primer orden.

---

## 21. Momentum como Ecuacion Diferencial

### Gradient descent como ODE de primer orden (flujo de gradiente)

```
dX/dt = -nabla f(X)
```

Sistema disipativo: `f(X(t))` decrece monotonicamente. Corresponde a una particula **sobreabsorta** deslizandose sin inercia.

### La ODE de Nesterov (Su, Boyd, Candes, 2016)

**Paper:** "A Differential Equation for Modeling Nesterov's Accelerated Gradient Method: Theory and Insights," *JMLR*, 2016.

Tomando el limite continuo del metodo de Nesterov (paso --> 0), se obtiene la ODE de segundo orden:

```
+-----------------------------------------------+
|  X'' + (3/t) * X' + nabla f(X) = 0            |
|                                                 |
|  Ecuacion de oscilador amortiguado              |
+-----------------------------------------------+
```

Los tres terminos corresponden a:
- **X''** = aceleracion (inercia), le da masa a la particula
- **(3/t) * X'** = amortiguamiento dependiente del tiempo (friccion que **decrece**)
- **nabla f(X)** = fuerza restauradora del potencial `f`

### Interpretacion fisica

La ecuacion describe una bola rodando en el paisaje de potencial `f(x)` con coeficiente de friccion `3/t`:

- **Tiempos tempranos** (t pequeno): friccion alta -> sistema sobreabsorbido -> movimiento directo al minimo
- **Tiempos tardios** (t grande): friccion baja -> sistema subabsorbido -> oscilaciones con amplitud decreciente

Esto coincide con la observacion empirica de que NAG oscila alrededor del optimo.

### Contraste con Polyak

El heavy ball de Polyak corresponde a una ODE con **friccion constante**:
```
X'' + gamma * X' + nabla f(X) = 0
```

Un oscilador armonico amortiguado estandar. La diferencia es fundamental: friccion constante da aceleracion local pero no la tasa global optima para funciones convexas generales.

### Resultado clave

Su et al. demostraron que las soluciones de la ODE de Nesterov satisfacen:
```
f(X(t)) - f(x*) = O(1/t^2)
```

igualando la tasa discreta `O(1/k^2)`. La constante `3/t` es critica: con `r/t` para `r < 3`, la tasa `O(1/t^2)` se pierde.

---

## 22. AdaGrad: Learning Rates Adaptativos (2011)

**Paper:** Duchi, Hazan, Singer, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization," *JMLR*, 2011.

### Motivacion: online learning y features sparse

En aplicaciones como NLP y click-through rate prediction, los vectores de features son extremadamente sparse. Una palabra que aparece 1 vez en 10,000 ejemplos recibe el mismo learning rate que una que aparece en todos. Esto es ineficiente -- features raros deberian tener learning rates mas altos porque cada observacion es altamente informativa.

### Por que los gradientes al cuadrado aparecen naturalmente

En el analisis de regret, Duchi et al. demostraron que la eleccion optima de la funcion proximal (la geometria para proyecciones en online learning) depende de la **estructura del segundo momento** de los gradientes observados. La matriz `G_t = SUM g_tau * g_tau^T` es un estimador empirico del segundo momento no-centrado `E[g*g^T]`.

La suma de gradientes al cuadrado emerge como la cantidad natural a rastrear porque la cota optima de regret involucra la norma dual definida por esta matriz.

### Cotas de regret

- Caso general: `O(sqrt(d * T))`
- Features sparse: `O(ln(d) * sqrt(T))` -- exponencialmente mejor en dimension

### El defecto fatal

`G_t` solo acumula (nunca olvida) -> el learning rate efectivo `eta / sqrt(G_t) -> 0` cuando `t -> infinito`. Para online convex learning esto esta bien. Para deep learning con gradientes densos, el learning rate decae a cero demasiado rapido.

---

## 23. RMSProp: Nacido en una Clase de Coursera (2012)

### El origen mas inusual en la historia de la optimizacion

RMSProp **nunca fue publicado en un paper peer-reviewed**. Fue introducido por Geoffrey Hinton en la **Lectura 6.5** de su curso de Coursera "Neural Networks for Machine Learning" (2012), con slides de Tijmen Tieleman.

La cita canonica es literalmente: *"Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning."*

A pesar de este origen informal, RMSProp se convirtio en uno de los optimizadores mas usados.

### El fix a AdaGrad

El insight de Hinton fue simple: reemplazar la **suma acumulativa** de gradientes al cuadrado con una **media movil exponencial (EMA)**:

```
E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2
theta_{t+1} = theta_t - eta / sqrt(E[g^2]_t + epsilon) * g_t
```

Donde `rho` (tipicamente 0.9) controla la tasa de decaimiento. El nombre "RMSProp" = "Root Mean Square Propagation" -- el denominador estima el RMS de gradientes recientes.

### Por que funciona

La EMA da una **ventana deslizante** sobre magnitudes de gradientes. A diferencia de AdaGrad, el learning rate efectivo no decae a cero -- se adapta a la curvatura **reciente** de la superficie de perdida.

---

## 24. Adam: Momentos Adaptativos (2014)

**Paper:** Kingma & Ba, "Adam: A Method for Stochastic Optimization," arXiv:1412.6980, ICLR 2015.

### El nombre

"Adam" viene de **"Adaptive Moment estimation"**. No es un acronimo (no es "ADAM") -- es simplemente un nombre inspirado en el concepto.

### Como combina momentum y learning rates adaptativos

- **Primer momento `m_t`** = EMA de gradientes = **momentum** (analogo al beta de Polyak)
- **Segundo momento `v_t`** = EMA de gradientes al cuadrado = **learning rate adaptativo estilo RMSProp**
- Adam combina ambos: momentum da la **direccion**, el denominador adaptativo da el **escalamiento por parametro**

### Derivacion de la correccion de sesgo

Como `m_0 = 0` y `v_0 = 0`, los estimadores iniciales estan sesgados hacia cero. Expandiendo la recurrencia:

```
m_t = (1-beta_1) * SUM_{i=1}^{t} beta_1^{t-i} * g_i
E[m_t] = E[g_t] * (1 - beta_1^t)
```

Entonces `E[m_t] != E[g_t]`; esta desviado por un factor de `(1 - beta_1^t)`. Dividir por este factor da el estimador insesgado `m_hat_t`.

Sin correccion de sesgo, los primeros pasos tendrian estimadores de momento artificialmente pequenos, causando pasos excesivamente grandes (denominador `sqrt(v_hat_t)` demasiado pequeno).

### El problema de convergencia (Reddi, Kale, Kumar, 2018)

**Paper:** "On the Convergence of Adam and Beyond," ICLR 2018.

Mostraron:

1. **La demostracion original esta errada.** Asumia que una cantidad `Gamma_t` es siempre positiva semi-definida, pero esto no vale para la EMA de Adam.

2. **Contraejemplo explicito.** Construyeron un problema 1D donde la solucion optima es `x* = -1`, pero Adam converge a `x = +1` (el peor punto). La funcion produce gradientes grandes ocasionalmente pero pequenos la mayoria del tiempo. La memoria corta de Adam descuenta el gradiente grande demasiado rapido.

3. **Causa raiz.** La EMA en `v_t` puede **decrecer** con el tiempo, haciendo que el learning rate efectivo **aumente** en ciertas dimensiones, violando la condicion de learning rate no-creciente.

4. **Condicion de fallo.** Adam puede divergir cuando `beta_1 < sqrt(beta_2)`, que se satisface con los defaults (0.9 < sqrt(0.999) ~ 0.9995).

**Fix AMSGrad:**
```
v_hat_t = max(v_hat_{t-1}, v_t)
```

Asegura que el denominador nunca decrece. Tiene demostracion de convergencia valida. Sin embargo, empiricamente AMSGrad rara vez supera a Adam.

---

## 25. Los Breakthroughs que Terminaron el Invierno de la IA

### Segundo Invierno de la IA (~1988-2006)

A pesar del entusiasmo por backpropagation, los problemas practicos se acumularon:
- **Vanishing gradients** (Hochreiter 1991, Bengio 1994): gradientes decaen como `(0.25)^L` con sigmoid
- SVMs (Vapnik 1995) ofrecian garantias teoricas y funcionaban mejor con datasets pequenos
- Poder computacional y datos limitados

### Los breakthroughs

| Ano | Breakthrough | Contribucion |
|---|---|---|
| 2006 | **Deep Belief Networks** (Hinton et al.) | Pretraining no-supervisado + fine-tuning. Demostro que redes profundas pueden entrenarse |
| 2010 | **ReLU** (Nair & Hinton) | `max(0,z)`: derivada = 1 para `z > 0`, sin vanishing gradient |
| 2012 | **AlexNet** (Krizhevsky, Sutskever, Hinton) | CNN profunda entrenada en GPUs gano ImageNet por 10+ puntos. **El momento en que deep learning "llego"** |
| 2014 | **Dropout** (Srivastava et al.) | Regularizacion: entrenar ensemble exponencial de sub-redes |
| 2015 | **Batch Normalization** (Ioffe & Szegedy) | Normalizar activaciones: permite LR mas altos, actua como regularizador |
| 2015 | **ResNets** (He et al.) | Skip connections: `y = F(x) + x`. Gradiente fluye sin impedimento: `dL/dx = dL/dy * (dF/dx + I)`. Redes de 152+ capas |
| 2015 | **Adam** (Kingma & Ba) | Momentum + adaptividad con correccion de sesgo |
| 2017 | **Transformer** (Vaswani et al.) | Atencion es todo lo que necesitas. Base de GPT, BERT, etc. |

### Por que ReLU fue tan importante matematicamente

```
ReLU(z) = max(0, z)
ReLU'(z) = { 1  si z > 0
           { 0  si z < 0
```

Con sigmoid: despues de 10 capas, gradiente ~ `(0.25)^10 = 9.5 x 10^{-7}`
Con ReLU: despues de 10 capas, gradiente ~ `(1)^10 = 1` (para neuronas activas)

### Por que ResNets resolvieron la profundidad

```
y = F(x, {W_i}) + x       (conexion residual)
```

Gradiente a traves de la skip connection:
```
dL/dx = dL/dy * (dF/dx + I)
```

La matriz identidad `I` asegura que el gradiente es **al menos** `dL/dy`, previniendo vanishing gradients sin importar la profundidad.

---

## 26. El Gran Debate: Adam vs SGD

### Wilson et al. (2017) - "The Marginal Value of Adaptive Methods"

**Hallazgos provocativos:**

1. **Construccion teorica:** Problema de clasificacion binaria linealmente separable donde GD/SGD logran 0% error de test, pero AdaGrad/Adam/RMSProp logran errores de test **cercanos al 50%** (azar). Los metodos adaptativos memorizan los datos pero no encuentran el separador de margen maximo.

2. **Resultados empiricos:** A traves de multiples benchmarks, SGD + momentum consistentemente logro igual o mejor rendimiento de **test** que Adam -- aunque Adam frecuentemente logro mejor **training** loss.

### La tension que continua

| Aspecto | Pro-SGD | Pro-Adam |
|---|---|---|
| Generalizacion | SGD encuentra minimos "planos" que generalizan mejor | Adam converge mucho mas rapido |
| Tuning | Requiere mas tuning de LR | Funciona "out of the box" |
| Arquitecturas | Mejor para CNNs en vision | **Esencial** para Transformers y NLP |
| Regularizacion | El ruido de SGD actua como regularizacion implicita | AdamW corrige muchos de los problemas |

### Resolucion moderna

El debate paso de "cual es mejor?" a "cuando usar cada uno?":
- **CNNs en vision:** SGD + momentum frecuentemente gana
- **Transformers/NLP:** Adam/AdamW domina y es practicamente obligatorio
- La **arquitectura y la tarea** importan mas que el optimizador aislado

---

# APENDICE

---

## Linea de Tiempo Completa

```
1669  Newton           Busqueda iterativa de raices
1696  Bernoulli        Braquistocrona (lanza calculo de variaciones)
1744  Euler            Ecuacion de Euler-Lagrange
1788  Lagrange         Mecanique analytique, multiplicadores
1809  Gauss            Metodo de minimos cuadrados
1847  CAUCHY           >>> GRADIENT DESCENT <<<
1939  Karush           Condiciones KKT (desigualdades)
1943  McCulloch-Pitts  Primera neurona matematica
1944  Levenberg        Minimos cuadrados amortiguados
1947  Dantzig          Metodo simplex (programacion lineal)
1949  Hebb             Primera regla de aprendizaje
1951  Robbins-Monro    Fundamentos matematicos de SGD
1951  Kuhn-Tucker      Condiciones KKT redescubiertas
1958  Rosenblatt       Perceptron + teorema de convergencia
1963  Marquardt        Levenberg-Marquardt
1964  POLYAK           >>> MOMENTUM (Heavy Ball) <<<
1969  Minsky-Papert    Limitaciones del perceptron -> 1er Invierno IA
1970  BFGS             Metodo quasi-Newton
1974  Werbos           Backpropagation (tesis doctoral)
1979  Khachiyan        LP en tiempo polinomial
1982  Hopfield         Redes de energia + conexion con fisica
1983  NESTEROV         >>> GRADIENTE ACELERADO OPTIMO O(1/k^2) <<<
1984  Karmarkar        Metodos de punto interior
1985  Hinton           Maquinas de Boltzmann
1986  Rumelhart+       Backpropagation popularizado
1989  Cybenko          Teorema de Aproximacion Universal
1989  LeCun            CNNs para digitos
1991  Hornik           UAT generalizado
1997  Hochreiter       LSTM (vanishing gradients en RNNs)
2004  Boyd             Convex Optimization (libro)
2006  Hinton           Deep Belief Networks -> renacimiento DL
2010  Nair & Hinton    ReLU
2011  DUCHI+           >>> ADAGRAD (learning rate adaptativo) <<<
2012  Hinton (Coursera) >>> RMSPROP (fix a AdaGrad) <<<
2012  Krizhevsky       AlexNet (deep learning "llega")
2012  Zeiler           Adadelta (sin learning rate)
2014  KINGMA & BA      >>> ADAM (momentos adaptativos) <<<
2015  Ioffe & Szegedy  Batch Normalization
2015  He et al.        ResNets (skip connections)
2016  Su, Boyd, Candes Momentum como ODE de segundo orden
2017  Wilson et al.    "Adam no generaliza tan bien como SGD"
2017  Vaswani et al.   Transformer
2018  Reddi et al.     Demostracion de Adam esta errada -> AMSGrad
2019  Loshchilov       AdamW (weight decay desacoplado)
```

---

## El Arco Narrativo de la Optimizacion

```
MATEMATICA PURA                        REDES NEURONALES
                                       
Euler 1744                             McCulloch-Pitts 1943
(calculo de variaciones)               (neurona matematica)
     |                                      |
Cauchy 1847                            Hebb 1949
(gradient descent para                 (regla de aprendizaje)
 mecanica celeste)                          |
     |                                 Rosenblatt 1958
     |                                 (perceptron)
     |                                      |
     |                                 Minsky-Papert 1969
     |                                 (limitaciones -> invierno)
     |                                      |
Polyak 1964                            Werbos 1974
(momentum)                             (backpropagation)
     |                                      |
Nesterov 1983                          Rumelhart+ 1986
(aceleracion optima)                   (popularizacion)
     |                                      |
     +------ CONVERGENCIA --------> Cybenko 1989 (UAT)
     |                              LeCun 1989 (CNNs)
     |                                      |
     |                              [SEGUNDO INVIERNO ~1988-2006]
     |                                      |
Duchi 2011 (AdaGrad)                   Hinton 2006 (DBN)
     |                              Krizhevsky 2012 (AlexNet + GPU)
Hinton 2012 (RMSProp)                      |
     |                                      |
Kingma 2014 (Adam)    <--- FUSION ---> Deep Learning moderno
     |                                      |
     +---- Debate Adam vs SGD -----> Transformers (2017+)
     |                                      |
     +---- ODE perspective (2016) -> Neural ODEs (2018)
```

La historia muestra un patron recurrente:
1. **La teoria matematica establece lo posible** (McCulloch-Pitts, convergencia del perceptron, aproximacion universal)
2. **Las limitaciones practicas se identifican** (Minsky-Papert, vanishing gradients)
3. **Breakthroughs algoritmicos/ingenieriles las superan** (backpropagation, ReLU, skip connections, GPUs)

Cada generacion construyo directamente sobre los fundamentos matematicos de sus predecesores. La ecuacion de Cauchy de 1847 sigue siendo el corazon de todo.
