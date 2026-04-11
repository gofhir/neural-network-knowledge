# Clase 10 - Algoritmos de Optimizacion y Learning Rate

**Diplomado Inteligencia Artificial - UC**
**Profesora:** Francisca Cattan Castillo
**Fecha:** 2026-04-10

---

## Tabla de Contenidos

1. [Motivacion y Contexto](#1-motivacion-y-contexto)
2. [Concepto de Optimizacion](#2-concepto-de-optimizacion)
3. [Gradiente Descendente (GD)](#3-gradiente-descendente-gd)
4. [Descenso de Gradiente Estocastico (SGD)](#4-descenso-de-gradiente-estocastico-sgd)
5. [Problemas de SGD](#5-problemas-de-sgd)
6. [Learning Rate](#6-learning-rate)
7. [SGD con Momentum](#7-sgd-con-momentum)
8. [Nesterov Accelerated Gradient (NAG)](#8-nesterov-accelerated-gradient-nag)
9. [Comparacion Momentum vs Nesterov](#9-comparacion-momentum-vs-nesterov)
10. [Adaptive Gradient (AdaGrad)](#10-adaptive-gradient-adagrad)
11. [Adaptive Moments (Adam)](#11-adaptive-moments-adam)
12. [Adaptar el Learning Rate durante el Entrenamiento](#12-adaptar-el-learning-rate-durante-el-entrenamiento)
13. [Early Stopping](#13-early-stopping)
14. [Papers y Optimizadores Modernos](#14-papers-y-optimizadores-modernos)
15. [Resumen Comparativo de Optimizadores](#15-resumen-comparativo-de-optimizadores)

---

## 1. Motivacion y Contexto

### El problema fundamental

Dado un conjunto de datos (ej: imagenes de animales), queremos que un **modelo** aprenda a clasificarlos en un **espacio distribuido** donde cada clase (label) quede separada de las demas. El modelo toma los datos de entrada y los mapea a un espacio de representaciones donde las distintas clases forman clusters separables.

### La funcion de perdida

La **funcion de perdida** (loss function) evalua que tan bien el algoritmo modela un conjunto de datos. Mide el **error** entre la salida del modelo y la salida esperada. El proceso iterativo busca reducir progresivamente esta perdida hasta lograr una buena separacion.

### Forward Pass y Backpropagation

El flujo basico de una red neuronal es:

```
x (input) --> [x W] (multiplicacion por pesos) --> [+ b] (suma del bias) --> y (output)
```

- **Forward pass**: Los datos pasan por la red para obtener una prediccion.
- **Error**: La diferencia entre el valor predicho y el real.
- **Mision**: Disminuir el error **ajustando los pesos** de cada nodo.
- **Como**: Los cambios se propagan hacia atras (**backpropagation**) usando derivadas parciales.
- Como los pesos son vectores, las derivadas parciales se llaman **gradientes**.

### Convergencia vs Divergencia

Un concepto clave ilustrado con la funcion `f(x) = x^2 * sin(x)`:

| | Convergencia | Divergencia |
|---|---|---|
| **Step coefficient** | 0.005 (pequeno) | 0.05 (grande) |
| **Comportamiento** | Descenso suave al minimo | Oscilaciones erraticas |
| **Resultado** | Llega al optimo (4.9, -23.7) | Se aleja del optimo (5.4, -22.1) |

**Leccion clave**: El tamano del paso (step coefficient / learning rate) determina si el proceso converge o diverge.

---

## 2. Concepto de Optimizacion

### Objetivo

El objetivo de un algoritmo de optimizacion es **modificar los pesos del modelo** para **minimizar** la funcion de perdida. Estos algoritmos son **iterativos**: dan pasos que acercan la solucion a un optimo en cada iteracion.

### Funcion objetivo

La funcion que queremos minimizar se compone de dos partes:

```
L(W) = L(f(x), y; W) + alpha * Omega(W)
```

Donde:
- `L(f(x), y; W)` = **Funcion de perdida**: mide el error entre la prediccion `f(x)` y el valor real `y`, parametrizada por los pesos `W`
- `alpha * Omega(W)` = **Regularizador**: penaliza la complejidad del modelo (previene overfitting)
- `f(x)` = el modelo que recibe la entrada `x`
- `y` = el valor real (ground truth)
- `W` = los pesos del modelo

### El gradiente como brujula

- La **derivada** nos entrega la **pendiente** de una funcion en un punto.
- El **gradiente** es la generalizacion multidimensional: nos indica la **direccion** en la cual debemos mover los pesos para acercarnos al optimo.

### Regla de actualizacion fundamental

```
w_i^new = w_i^old - eta * (dL / dw_i)
```

Donde:
- `w_i^new` = nuevo valor del peso
- `w_i^old` = valor actual del peso
- `eta` (eta) = **learning rate** (tasa de aprendizaje)
- `dL/dw_i` = el **gradiente** (derivada parcial de la perdida respecto al peso)

El signo **negativo** es crucial: nos movemos en la direccion **opuesta** al gradiente (que apunta hacia arriba), para descender hacia el minimo.

> **Intuicion**: El gradiente apunta en la direccion de mayor crecimiento. Para minimizar, nos movemos en la direccion contraria.

---

## 3. Gradiente Descendente (GD)

### Algoritmo clasico

En Gradient Descent (GD), calculamos la funcion de perdida considerando **todos** los elementos del set de entrenamiento:

```
L(W) = SUM_n [ L(f(x_n), y_n; W) ] + alpha * Omega(W)
```

Donde `n` recorre **todos los datos** del dataset.

### Regla de actualizacion

```
w_i^new = w_i^old - eta * (dL / dw_i)
```

La perdida se calcula sobre el dataset completo, y luego se actualiza una vez.

### Problema fundamental

**El paso (eta) importa enormemente:**
- Si `eta` es muy grande: los pasos sobredimensionados saltan el minimo (divergencia)
- Si `eta` es muy pequeno: convergencia extremadamente lenta
- Ademas, con datasets grandes, calcular el gradiente sobre todos los datos en cada iteracion es **muy costoso computacionalmente**

---

## 4. Descenso de Gradiente Estocastico (SGD)

### Motivacion

Si el dataset es muy grande, GD clasico es prohibitivamente lento porque:
- Cada iteracion requiere pasar por **todo** el dataset
- Se hacen demasiados calculos por cada actualizacion de pesos

### Solucion: Mini-batches

SGD muestrea **subconjuntos** de datos llamados **batches**:

```
L(W) = SUM_n' [ L(f(x_n'), y_n'; W) ] + alpha * Omega(W)
```

Ahora `n'` es un **minibatch** (subconjunto pequeno) en vez de todos los datos.

### Conceptos clave: Epoca vs Iteracion

| Concepto | Definicion |
|---|---|
| **Epoca** | Un ciclo completo donde el modelo pasa por **todo** el set de datos |
| **Iteracion** | Una actualizacion de pesos usando un **batch** |
| **Batch size** | Tamano del subconjunto de datos por iteracion |

**Ejemplo concreto**:
- Dataset: 1000 datos
- Batch size = 100
- Iteraciones por epoca = 1000 / 100 = **10 iteraciones**
- Batch size = 500 -> 2 iteraciones por epoca
- Batch size = 1000 (todo el dataset) -> 1 iteracion por epoca = GD clasico

### Relacion entre batch size y epocas

- **Mas epocas** = mayor diversidad de patrones vistos = mejor generalizacion (en teoria)
- **Batch size pequeno** = mas iteraciones por epoca = mas actualizaciones pero con gradientes mas ruidosos
- **Batch size grande** = menos iteraciones = gradientes mas estables pero mas costosos

### SGD vs GD visualmente

En una superficie de perdida con curvas de nivel (contour plot):
- **GD** (negro): camino directo y suave hacia el minimo
- **SGD** (rojo): camino mas errático y zigzagueante, pero eventualmente llega

La naturaleza estocastica introduce **ruido**, lo cual puede ser beneficioso para escapar de minimos locales.

---

## 5. Problemas de SGD

SGD tiene tres problemas principales:

### 5.1 Saddle Points (Puntos de Silla)

Un saddle point es un punto donde el gradiente es cero pero **no es un minimo**. La superficie sube en una direccion y baja en otra. SGD puede quedarse "atascado" en estos puntos porque el gradiente es muy pequeno o nulo.

### 5.2 Optimos Locales

La funcion de perdida puede tener multiples minimos. SGD puede converger a un **minimo local** en vez del **minimo global** (el verdadero optimo). El minimo global puede estar "lejos" en el espacio de parametros.

### 5.3 Tamano de los Pasos

El gradiente determina la **direccion**, pero el **tamano del paso** depende del learning rate `eta`:

- **Learning rate grande**: pasos grandes que pueden saltar sobre el minimo, oscilando de un lado a otro del valle
- **Learning rate pequeno**: pasos diminutos que convergen muy lentamente

Ambos extremos son problematicos. Se necesita un balance.

---

## 6. Learning Rate

### El dilema central

```
w_i^new = w_i^old - eta * (dL / dw_i)
                     ^^^
                El learning rate (eta) controla el tamano del paso
```

### Comportamiento segun el valor de eta

| Learning Rate | Comportamiento | Loss vs Epoch |
|---|---|---|
| **Muy alto** | Diverge, la loss sube | Curva ascendente |
| **Alto** | Converge rapido al inicio, luego oscila en un plateau | Curva que baja y se estanca alto |
| **Bajo** | Converge muy lentamente | Curva que baja gradualmente pero no termina |
| **Bueno** | Converge rapido y llega a un buen minimo | Curva que baja rapido y se estabiliza bajo |

### Intuicion visual

- Con un learning rate **pequeno**, el optimizador da pasos diminutos bajando cuidadosamente al valle
- Con un learning rate **grande**, el optimizador salta de un lado a otro del valle sin poder asentarse en el fondo

---

## 7. SGD con Momentum

### Motivacion

El gradiente de SGD depende solo del **batch actual**, lo cual genera variabilidad:
- Diferentes grupos de datos pueden producir gradientes muy distintos
- El optimizador puede oscilar o cambiar de direccion abruptamente

### Idea central

Momentum agrega la idea de **mantener la direccion** del gradiente, incorporando **historia** de como se ha movido en el pasado. Es analogo al momentum fisico: un objeto en movimiento tiende a mantener su direccion.

### Formulas

**Sin momentum (SGD clasico):**
```
w_{t+1} = w_t - eta * nabla_f(x_t)
```

**Con momentum:**
```
v_{t+1} = rho * v_t + nabla_f(x_t)       # velocidad acumulada
w_{t+1} = w_t - eta * v_{t+1}             # actualizacion de pesos
```

Donde:
- `v_{t+1}` = velocidad (momentum acumulado)
- `rho` = coeficiente de momentum (tipicamente 0.0 a 1.0, comun: 0.9)
- `nabla_f(x_t)` = gradiente actual

### Ejemplo numerico (1 dimension, rho = 0.1)

| Batch | Gradiente | v_{t+1} = rho * v_t + gradiente | Calculo |
|---|---|---|---|
| Batch 1 | -2 | -2 | 0.1*0 + (-2) = -2 |
| Batch 2 | -4 | -4.2 | 0.1*(-2) + (-4) = -4.2 |
| Batch 3 | -3 | -3.42 | 0.1*(-4.2) + (-3) = -3.42 |
| Batch 4 | 2 | 1.658 | 0.1*(-3.42) + 2 = 1.658 |

**Observaciones:**
- Cuando los gradientes mantienen la misma direccion (batches 1-3, todos negativos), el momentum **amplifica** el paso (de -2 a -4.2)
- Cuando el gradiente cambia de direccion (batch 4, positivo), el momentum **amortigua** el cambio (seria 2, pero queda 1.658)
- Esto produce trayectorias mas suaves

### Efecto visual

- **SGD sin momentum**: oscilaciones en zigzag alrededor del camino optimo (curvas de nivel)
- **SGD con momentum**: trayectoria mas suave y directa hacia el minimo

### Momentum insuficiente vs suficiente

- **Momentum insuficiente** (rho bajo): el optimizador puede quedar atrapado en un minimo local
- **Momentum suficiente** (rho alto): el optimizador tiene suficiente "inercia" para superar colinas entre minimos locales y encontrar mejores soluciones

---

## 8. Nesterov Accelerated Gradient (NAG)

### Idea fundamental

Nesterov es una variante "predictiva" del momentum. En vez de calcular el gradiente en la posicion actual, **primero se mueve** en la direccion del momentum y **luego** calcula el gradiente desde esa posicion avanzada.

### Analogia

- **Momentum clasico**: Es como una bola bajando un cerro que no sabe que hay mas abajo (ciega)
- **Nesterov**: Es como un esquiador que **mira hacia adelante** para frenar antes de una curva cerrada

### Formulas (3 pasos)

**Paso 1 - Update del momentum actual:**
```
w_t' = w_t - rho * v_t        # "mirar hacia adelante" usando el momentum
```

**Paso 2 - Correccion del gradiente:**
```
v_{t+1} = rho * v_t - alpha * (dL(x) / dw_t')    # calcular gradiente en la posicion avanzada
```

**Paso 3 - Update final:**
```
w_{t+1} = w_t + v_{t+1}       # hacer el salto al nuevo peso
```

### Desglose intuitivo

1. `rho * v_t` = la velocidad (momentum) nos va a llevar en cierta direccion
2. `alpha * (dL/dw_t')` = tasa de correccion basada en el gradiente **desde la posicion predicha**
3. La combinacion produce un paso mas informado

---

## 9. Comparacion Momentum vs Nesterov

### Diferencia grafica

**Momentum clasico:**
```
Posicion actual --> [Gradiente] + [Velocidad] = Paso real
                     (desde aqui)  (acumulada)
```

**Nesterov:**
```
Posicion actual --> [Velocidad] --> Posicion predicha --> [Gradiente] = Paso real
                    (primero)                             (desde alla)
```

### Tabla comparativa

| Caracteristica | Momentum Clasico | Nesterov (NAG) |
|---|---|---|
| **Punto de calculo** | Se calcula en la posicion **actual** | Se calcula en la posicion **predecida** |
| **Comportamiento** | Como una bola bajando un cerro sin saber que hay mas abajo | Como un esquiador que mira adelante para frenar antes de una curva |
| **Convergencia** | Rapida, pero puede **oscilar** al final | Mas **estable** porque evita que el modelo sobrepase el minimo tan agresivamente |
| **Ventaja** | Simple, efectivo | Mejor control cerca del optimo |

---

## 10. Adaptive Gradient (AdaGrad)

### Motivacion

En SGD (con o sin momentum), **todos los pesos** comparten el **mismo learning rate**. Pero en la realidad:
- Algunos pesos necesitan actualizaciones grandes (features poco frecuentes)
- Otros necesitan actualizaciones pequenas (features frecuentes)
- El learning rate optimo puede variar dependiendo del batch

### Idea central

AdaGrad da a **cada peso su propio learning rate** que se adapta segun la historia de sus gradientes.

### Formulas

**Actualizacion del peso i:**
```
w_t^i = w_{t-1}^i - eta_{w^i} * (dL(x) / dw^i)
```

**Learning rate adaptativo para el peso i:**
```
eta_{w^i} = eta / sqrt( SUM_{j=1}^{t} (G_j)^2 )
```

Donde:
- `eta` = learning rate global (hiperparametro)
- `G_j` = gradientes historicos del peso `i` hasta el tiempo `t`
- La suma acumula el cuadrado de todos los gradientes pasados

### Efecto de normalizacion

- **Gradientes muy pronunciados** (grandes) -> denominador grande -> learning rate **reducido** (se frena)
- **Gradientes muy bajos** (pequenos) -> denominador pequeno -> learning rate **acelerado** (se impulsa)

Es una especie de **normalizacion automatica** de los gradientes.

### Problema de AdaGrad

El denominador `sqrt(SUM G_j^2)` **siempre crece** (es una suma de cuadrados, siempre positiva). Esto significa que:
- El learning rate **tiende a cero** con el tiempo
- El entrenamiento puede **detenerse prematuramente** porque los pasos se vuelven infinitesimalmente pequenos

> **Nota:** Este problema motivo el desarrollo de RMSProp (no cubierto en detalle en esta clase) y eventualmente Adam.

### Paper original

AdaGrad fue propuesto por Duchi, Hazan & Singer (2011): *"Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"*. Fue uno de los primeros optimizadores adaptativos influyentes.

---

## 11. Adaptive Moments (Adam)

### Motivacion

Adam combina lo mejor de dos mundos:
- **Momentum** de SGD (suavizado de gradientes usando primer momento)
- **Escalamiento adaptativo** de AdaGrad (learning rate por parametro usando segundo momento)

### Estructura de la formula

```
w_i^t = w_i^{t-1} - eta * r_t / (sqrt(v_t) + epsilon)
```

Donde:
- `r_t` = **primer momento** (media exponencial del gradiente) ~ **Momentum**
- `v_t` = **segundo momento** (media exponencial del gradiente al cuadrado) ~ **Escalamiento adaptativo**
- `epsilon` = constante pequena para estabilidad numerica (evitar division por cero)

### Calculo de los momentos

**Primer momento (media):**
```
r_t = (1 - gamma_1) * G_i^t + gamma_1 * r_{t-1}
```

**Segundo momento (varianza):**
```
v_t = (1 - gamma_2) * (G_i^t)^2 + gamma_2 * v_{t-1}
```

Donde:
- `gamma_1` (beta_1 en el paper) = tasa de decaimiento del primer momento (tipico: 0.9)
- `gamma_2` (beta_2 en el paper) = tasa de decaimiento del segundo momento (tipico: 0.999)
- `G_i^t` = gradiente del peso `i` en tiempo `t`

### Correccion de sesgo (Bias Correction)

Ambos estimadores parten de cero, lo cual introduce un sesgo hacia cero en las primeras iteraciones. Adam aplica una **correccion**:

```
r_t_corregido = r_t / (1 - gamma_1^t)
v_t_corregido = v_t / (1 - gamma_2^t)
```

A medida que `t` crece, `gamma^t` tiende a cero y la correccion se vuelve irrelevante.

**Formula final de Adam:**
```
w_i^t = w_i^{t-1} - eta * r_t_corregido / (sqrt(v_t_corregido) + epsilon)
```

### Hiperparametros tipicos de Adam

| Parametro | Valor tipico | Descripcion |
|---|---|---|
| `eta` (alpha) | 0.001 | Learning rate global |
| `beta_1` (gamma_1) | 0.9 | Decaimiento del primer momento |
| `beta_2` (gamma_2) | 0.999 | Decaimiento del segundo momento |
| `epsilon` | 1e-8 | Estabilidad numerica |

### Paper original

**"Adam: A Method for Stochastic Optimization"** - Diederik P. Kingma & Jimmy Ba (2014, ICLR 2015)
- arXiv: 1412.6980
- Contribuciones clave:
  - Algoritmo eficiente computacionalmente con minimo uso de memoria
  - Bueno para problemas con gradientes ruidosos o sparse
  - Introduce tambien **AdaMax** (variante con norma infinito)
  - Invariante al re-escalamiento diagonal del gradiente
  - Se convirtio en el optimizador **mas usado** en deep learning moderno

### Por que Adam es tan popular

1. **Combina momentum + adaptividad**: aprovecha las fortalezas de ambos enfoques
2. **Pocos hiperparametros**: y los defaults funcionan bien en la mayoria de casos
3. **Robusto**: funciona bien con gradientes sparse, ruidosos, y en problemas no estacionarios
4. **Facil de implementar**: pocas lineas de codigo
5. **Correccion de sesgo**: maneja correctamente las primeras iteraciones

---

## 12. Adaptar el Learning Rate durante el Entrenamiento

### Motivacion

Un learning rate fijo puede no ser optimo durante todo el entrenamiento:
- Al inicio, queremos pasos **grandes** para avanzar rapido
- Cerca del optimo, queremos pasos **pequenos** para ajustar fino

### Estrategias de Scheduling

#### 12.1 Decaimiento basado en epocas (Step Decay)

Cada cierta cantidad de epocas, **reducir** el learning rate por un factor.

**Ejemplo real con ResNet:**

| Epocas | Learning Rate |
|---|---|
| 0-30 | 0.1 |
| 30-60 | 0.01 |
| 60-90 | 0.001 |
| 90-120 | 0.0001 |

Cada 30 epocas se multiplica el LR por 0.1. El efecto visual en la curva de training loss es dramatico: cada reduccion produce una caida abrupta seguida de un nuevo plateau.

#### 12.2 Decaimiento basado en validacion (ReduceLROnPlateau)

- Monitorear la funcion de perdida del **conjunto de validacion**
- Si la perdida no mejora durante `n` epocas consecutivas, reducir el learning rate
- Mas adaptativo que el step decay fijo

#### 12.3 Otras estrategias comunes (no cubiertas en detalle)

- **Cosine annealing**: LR sigue una curva coseno, bajando suavemente
- **Warmup**: LR empieza bajo y sube gradualmente antes de aplicar decay
- **Cyclical LR**: LR oscila entre un minimo y un maximo

---

## 13. Early Stopping

### Concepto

Si durante el entrenamiento el error del **conjunto de validacion** comienza a **empeorar** (subir) mientras el error de entrenamiento sigue bajando, es senal de **overfitting**. Early stopping detiene el entrenamiento en ese punto.

### Comportamiento tipico

```
Error
 ^
 |  \
 |   \         ___--- Test error (sube = overfitting)
 |    \    ---/
 |     \--/
 |      \
 |       \________ Training error (sigue bajando)
 |
 +-----|-----------> Training cycles
       ^
       Early stopping point
```

### Implementacion practica

1. Dividir datos en train/validation/test
2. Entrenar monitoreando la **validation loss**
3. Guardar el modelo cada vez que la validation loss mejore (**best checkpoint**)
4. Si la validation loss no mejora durante `patience` epocas, detener el entrenamiento
5. Restaurar el mejor checkpoint

### Relacion con learning rate

Early stopping y learning rate scheduling son **complementarios**:
- El scheduler puede reducir el LR cuando el progreso se estanca
- Early stopping detiene completamente cuando ya no hay progreso posible

---

## 14. Papers y Optimizadores Modernos

La clase referencia varios optimizadores posteriores a Adam. A continuacion un analisis profundo de cada uno:

### 14.1 Adam (2015) - El fundacional

**Paper:** *"Adam: A Method for Stochastic Optimization"*
**Autores:** Diederik P. Kingma, Jimmy Ba
**Publicado en:** ICLR 2015 | arXiv: 1412.6980

Ya cubierto en detalle en la seccion 11. Es el punto de referencia contra el cual se comparan todos los optimizadores modernos.

### 14.2 ADMM as Continuous Dynamical Systems (2018)

**Paper:** *"ADMM and Accelerated ADMM as Continuous Dynamical Systems"*
**arXiv:** 1805.06579

ADMM (Alternating Direction Method of Multipliers) es un framework de optimizacion que descompone problemas complejos en subproblemas mas simples. Este paper analiza ADMM desde la perspectiva de sistemas dinamicos continuos, proporcionando nuevas intuiciones teoricas sobre su convergencia y aceleracion.

### 14.3 Ranger (2019)

**Repositorio:** github.com/lessw2020/Ranger-Deep-Learning-Optimizer

Ranger combina dos ideas:
- **RAdam** (Rectified Adam): corrige problemas de varianza en las primeras iteraciones de Adam
- **Lookahead**: mecanismo de "mirada adelante" (ver siguiente seccion)

Es un optimizador "todo-en-uno" que busca robustez sin tuning extensivo.

### 14.4 Lookahead (2019)

**Paper:** *"Lookahead Optimizer: k steps forward, 1 step back"*
**Autores:** Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba
**Publicado en:** NeurIPS 2019 | arXiv: 1907.08610

**Idea clave:** Lookahead es un **meta-optimizador** que envuelve a cualquier optimizador existente (SGD, Adam, etc.):

1. Mantiene dos conjuntos de pesos: **fast weights** (phi) y **slow weights** (theta)
2. El optimizador interno avanza `k` pasos con los fast weights
3. Los slow weights se actualizan interpolando: `theta = theta + alpha * (phi - theta)`
4. Los fast weights se resetean a los slow weights

**Beneficios:**
- Reduce la varianza del optimizador interno
- Mejora la estabilidad del entrenamiento
- Costo computacional y de memoria negligible
- Funciona con SGD, Adam, y cualquier otro optimizador

**Analogia:** Es como explorar `k` pasos hacia adelante, evaluar donde se llego, y luego dar un paso conservador en esa direccion general.

### 14.5 Gradient Centralization (2020)

**Paper:** *"Gradient Centralization: A New Optimization Technique for Deep Neural Networks"*
**Autores:** Hongwei Yong, Jianqiang Huang, Xiansheng Hua, Lei Zhang
**arXiv:** 2004.01461

**Idea clave:** Centralizar los vectores gradiente para que tengan **media cero** antes de usarlos para actualizar los pesos.

**Implementacion:** Literalmente **una linea de codigo**:
```python
# Antes de actualizar, restar la media del gradiente
gradient = gradient - gradient.mean()
```

**Beneficios:**
- Actua como regularizacion implicita (en el espacio de pesos y features de salida)
- Mejora la Lipschitzness de la funcion de perdida (entrenamiento mas estable)
- Mejora la generalizacion en clasificacion, deteccion y segmentacion
- Compatible con cualquier optimizador basado en gradientes

**Interpretacion teorica:** GC es equivalente a un metodo de descenso de gradiente proyectado con una funcion de perdida restringida.

### 14.6 AngularGrad (2021)

**Paper:** *"AngularGrad: A New Optimization Technique for Angular Convergence of Convolutional Neural Networks"*
**Autores:** S.K. Roy, M.E. Paoletti, J.M. Haut, S.R. Dubey, P. Kar, A. Plaza, B.B. Chaudhuri
**arXiv:** 2105.10190

**Idea clave:** Usar la **informacion angular** (direccion) entre gradientes consecutivos, no solo su magnitud.

- Es el primer trabajo en explotar la informacion angular del gradiente para optimizacion
- Genera scores de control de paso usando datos angulares de iteraciones previas
- Produce trayectorias de optimizacion mas "suaves"

**Dos variantes:**
1. **AngularGrad-Tangent**: usa la funcion tangente para calcular la informacion angular
2. **AngularGrad-Cosine**: usa la funcion coseno

**Garantias teoricas:** Mismas cotas de regret que Adam, pero con mejor rendimiento empirico en benchmarks.

---

## 15. Resumen Comparativo de Optimizadores

### Tabla general

| Optimizador | LR Adaptativo | Momentum | Ventaja principal | Desventaja principal |
|---|---|---|---|---|
| **GD** | No | No | Gradiente exacto | Muy lento en datasets grandes |
| **SGD** | No | No | Rapido por iteracion | Ruidoso, puede oscilar |
| **SGD + Momentum** | No | Si | Suaviza oscilaciones | Puede sobrepasar minimos |
| **SGD + Nesterov** | No | Si (predictivo) | Mejor control cerca del optimo | Mas complejo que momentum |
| **AdaGrad** | Si (por peso) | No | Adapta LR automaticamente | LR tiende a cero |
| **Adam** | Si (por peso) | Si (1er y 2do momento) | Robusto, pocos hiperparametros | Puede no generalizar tan bien como SGD en algunos casos |

### Flujo de decision practico

```
Problema nuevo?
  |
  v
Empezar con Adam (lr=0.001, defaults)
  |
  v
Funciona bien? ----Si----> Listo!
  |
  No
  v
Probar SGD + Momentum + LR Scheduler
  |
  v
Funciona bien? ----Si----> Listo!
  |
  No
  v
Experimentar con:
  - AdamW (Adam + weight decay desacoplado)
  - Ranger (RAdam + Lookahead)
  - Gradient Centralization (agregar una linea)
```

### Jerarquia evolutiva de los optimizadores

```
GD (1847, Cauchy)
 |
 +-- SGD (muestreo estocastico)
      |
      +-- SGD + Momentum (suavizado historico)
      |    |
      |    +-- Nesterov (momentum predictivo)
      |
      +-- AdaGrad (LR adaptativo por peso)
           |
           +-- RMSProp (media movil exponencial, no cubierto)
           |
           +-- Adam (Momentum + Adaptividad) [2015]
                |
                +-- AdaMax (norma infinito)
                +-- RAdam (correccion de varianza)
                +-- AdamW (weight decay desacoplado)
                +-- Lookahead (meta-optimizador) [2019]
                +-- Ranger (RAdam + Lookahead) [2019]
                +-- Gradient Centralization [2020]
                +-- AngularGrad [2021]
```

---

## Conceptos Clave para Recordar

1. **Optimizacion = minimizar la funcion de perdida ajustando pesos**
2. **El gradiente indica la direccion; el learning rate controla el tamano del paso**
3. **SGD vs GD**: SGD usa mini-batches para eficiencia computacional
4. **Momentum**: acumula historia de gradientes para suavizar el camino
5. **Nesterov**: calcula el gradiente desde la posicion predicha (mira adelante)
6. **AdaGrad**: learning rate adaptativo por peso, pero tiende a cero
7. **Adam**: combina momentum + adaptividad con correccion de sesgo
8. **Learning rate scheduling**: cambiar el LR durante entrenamiento (step decay, plateau)
9. **Early stopping**: detener cuando la validation loss empeora (previene overfitting)
10. **No hay optimizador universalmente "mejor"**: depende del problema, datos y arquitectura

---

## Formulas Resumen

### SGD basico
```
w_{t+1} = w_t - eta * nabla_f(x_t)
```

### SGD + Momentum
```
v_{t+1} = rho * v_t + nabla_f(x_t)
w_{t+1} = w_t - eta * v_{t+1}
```

### Nesterov
```
w_t' = w_t - rho * v_t
v_{t+1} = rho * v_t - alpha * nabla_L(w_t')
w_{t+1} = w_t + v_{t+1}
```

### AdaGrad
```
eta_{w^i} = eta / sqrt(SUM_j (G_j)^2)
w_t^i = w_{t-1}^i - eta_{w^i} * nabla_L
```

### Adam
```
r_t = (1-beta1)*g_t + beta1*r_{t-1}          # primer momento
v_t = (1-beta2)*g_t^2 + beta2*v_{t-1}        # segundo momento
r_hat = r_t / (1 - beta1^t)                   # correccion de sesgo
v_hat = v_t / (1 - beta2^t)                   # correccion de sesgo
w_t = w_{t-1} - eta * r_hat / (sqrt(v_hat) + epsilon)
```
