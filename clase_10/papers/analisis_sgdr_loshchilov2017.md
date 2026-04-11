# Analisis del Paper: SGDR — Stochastic Gradient Descent with Warm Restarts

**Autores**: Ilya Loshchilov & Frank Hutter
**Institucion**: University of Freiburg, Germany
**Publicado en**: ICLR 2017 (Conference Paper)
**arXiv**: 1608.03983 (enviado Ago 2016, version final Mayo 2017)
**Codigo fuente**: https://github.com/loshchil/SGDR

> PDF descargado en: [papers/4_SGDR_Loshchilov2017.pdf](4_SGDR_Loshchilov2017.pdf)

---

## Datos Clave del Paper

| Aspecto | Detalle |
|---|---|
| **Ano** | 2017 (ICLR 2017, arXiv Ago 2016) |
| **Citas** | Uno de los papers mas citados en optimizacion para Deep Learning (~7,000+ citas) |
| **Autores notables** | Ilya Loshchilov & Frank Hutter (tambien autores de AdamW y CMA-ES) |
| **Idea central** | Reiniciar periodicamente el learning rate usando cosine annealing para mejorar el rendimiento "anytime" de SGD |
| **Impacto** | Se convirtio en scheduler estandar; implementado como `CosineAnnealingWarmRestarts` en PyTorch |

---

## 1. Problema que Resuelve

Entrenar redes neuronales profundas (DNNs) con SGD requiere definir un **schedule de learning rate** (como disminuir el learning rate a lo largo del entrenamiento). Los metodos comunes en 2016 eran:

- **Step decay**: Dividir el learning rate por un factor fijo en epochs predefinidos (e.g., dividir por 5 en epochs 60, 120, 160)
- **Learning rates adaptativos**: AdaDelta, Adam (que aproximan la inversa del Hessiano)

**Problemas con estos enfoques**:
1. **Step decay** requiere decidir de antemano los epochs exactos donde reducir el learning rate — mal "anytime performance"
2. El schedule es **rigido**: si entrenas 200 epochs tienes un schedule; si quieres entrenar 300 necesitas redisenar todo
3. **Sin exploracion**: una vez que el learning rate baja, el optimizador queda atrapado en el minimo local mas cercano
4. Los **optimizadores adaptativos** (Adam, AdaDelta) en esa epoca no superaban a SGD con momentum bien tuneado en tareas como CIFAR e ImageNet

**Solucion propuesta**: SGDR — simular **warm restarts** periodicos de SGD, donde el learning rate se reinicia a un valor alto y decae siguiendo una funcion **coseno**. Esto permite al optimizador escapar de minimos locales pobres y lograr buen rendimiento en cualquier momento del entrenamiento.

```text
SCHEDULE CLASICO (step decay):
  lr
  0.1 |████████████████████
      |                    ████████████████████
  0.02|                                        ████████████████████
      |                                                            ████
  0.004                                                                ████
      └──────────────────────────────────────────────────────────────────────
       0    60              120             160                    200 epochs
       Solo 3 "saltos" predefinidos. Sin flexibilidad.

SGDR (cosine annealing + warm restarts):
  lr
  0.05|*        *              *
      | *       * *            * *
      |  *      *  *           *   *
      |   *     *   *          *    *
      |    **   *    **        *     **
      |      ** *      **     *       ***
      |        **        **** *          ******
  0.0 |         *            **               ********
      └──────────────────────────────────────────────────
       0   T_0  restart    restart                    epochs
       Multiples ciclos. Periodos crecientes (T_mult=2).
```

---

## 2. Modelo Formal (Seccion 3 del paper)

### 2.1. SGD con Momentum (base)

El punto de partida es SGD con momentum de Nesterov, el metodo que en 2016 producia los mejores resultados en CIFAR y ImageNet:

```text
v_{t+1} = mu_t * v_t - eta_t * grad(f_t(x_t))     (actualizacion de velocidad)
x_{t+1} = x_t + v_{t+1}                             (actualizacion de parametros)

donde:
  v_t     = vector de velocidad (inicializado en 0)
  eta_t   = learning rate decreciente
  mu_t    = momentum rate (tipicamente 0.9)
  grad(f_t(x_t)) = gradiente del loss en el minibatch t
```

### 2.2. Formula de Cosine Annealing (Ecuacion 5 del paper)

La contribucion central del paper es la formula para el learning rate en el i-esimo "run" (ciclo):

```text
                            1                        T_cur
  eta_t = eta_min^i  +  ─────── (eta_max^i - eta_min^i)(1 + cos(───── * pi))
                            2                         T_i

donde:
  eta_min^i = learning rate minimo en el i-esimo ciclo
  eta_max^i = learning rate maximo en el i-esimo ciclo
  T_cur     = numero de epochs transcurridos desde el ultimo restart
  T_i       = duracion total del i-esimo ciclo (en epochs)
```

**Comportamiento de la formula**:

```text
Cuando T_cur = 0 (inicio del ciclo):
  cos(0) = 1
  eta_t = eta_min + 1/2 * (eta_max - eta_min) * (1 + 1)
  eta_t = eta_min + (eta_max - eta_min)
  eta_t = eta_max                              ← learning rate MAXIMO

Cuando T_cur = T_i (fin del ciclo):
  cos(pi) = -1
  eta_t = eta_min + 1/2 * (eta_max - eta_min) * (1 + (-1))
  eta_t = eta_min + 0
  eta_t = eta_min                              ← learning rate MINIMO
```

### 2.3. Ejemplo numerico paso a paso

```text
Configuracion: eta_max = 0.05, eta_min = 0, T_0 = 10, T_mult = 1

Ciclo 1 (epochs 0-10, T_i = 10):
  Epoch 0:  T_cur=0   → cos(0/10 * pi) = cos(0)     =  1.0  → eta = 0.0500
  Epoch 1:  T_cur=1   → cos(1/10 * pi) = cos(0.314)  =  0.95 → eta = 0.0488
  Epoch 2:  T_cur=2   → cos(2/10 * pi) = cos(0.628)  =  0.81 → eta = 0.0452
  Epoch 3:  T_cur=3   → cos(3/10 * pi) = cos(0.942)  =  0.59 → eta = 0.0398
  Epoch 5:  T_cur=5   → cos(5/10 * pi) = cos(pi/2)   =  0.00 → eta = 0.0250
  Epoch 7:  T_cur=7   → cos(7/10 * pi) = cos(2.199)  = -0.59 → eta = 0.0103
  Epoch 9:  T_cur=9   → cos(9/10 * pi) = cos(2.827)  = -0.95 → eta = 0.0013
  Epoch 10: T_cur=10  → cos(10/10 * pi)= cos(pi)     = -1.00 → eta = 0.0000
  ← RESTART: T_cur vuelve a 0, eta salta a 0.05 de nuevo

Ciclo 2 (epochs 10-20, T_i = 10):
  Epoch 10: T_cur=0  → eta = 0.0500   ← salto repentino!
  Epoch 15: T_cur=5  → eta = 0.0250
  Epoch 20: T_cur=10 → eta = 0.0000
  ← RESTART ...
```

### 2.4. Parametros T_0 y T_mult

```text
T_0: Duracion del PRIMER ciclo (en epochs)
T_mult: Factor de multiplicacion para la duracion de cada ciclo sucesivo

  T_i = T_0 * T_mult^(i-1)     para el i-esimo ciclo

Ejemplos:

  T_0 = 50, T_mult = 1 (periodos FIJOS):
    Ciclo 1: 50 epochs
    Ciclo 2: 50 epochs
    Ciclo 3: 50 epochs
    Total en 200 epochs: 4 restarts

  T_0 = 1, T_mult = 2 (periodos que se DUPLICAN):
    Ciclo 1:  1 epoch
    Ciclo 2:  2 epochs
    Ciclo 3:  4 epochs
    Ciclo 4:  8 epochs
    Ciclo 5: 16 epochs
    Ciclo 6: 32 epochs
    Ciclo 7: 64 epochs   (total: 1+2+4+8+16+32+64 = 127 epochs)
    Ciclo 8: 128 epochs  (total: 255 epochs)

  T_0 = 10, T_mult = 2 (periodos que se DUPLICAN, inicio mas largo):
    Ciclo 1: 10 epochs
    Ciclo 2: 20 epochs
    Ciclo 3: 40 epochs
    Ciclo 4: 80 epochs   (total: 10+20+40+80 = 150 epochs)
```

### 2.5. Visualizacion del learning rate con T_mult = 2

```text
  eta
  0.05|*                                                     
      | *    *                                                
      |  *   * *                                              
      |   *  *  *     *                                       
      |    * *   *    * *                                     
      |     **    *   *  *        *                            
      |      *     ** *   **      * *                          
      |             * *     ***   *  **                        
      |              **       ****    ****                     
  0.0 |               *          *       ********              
      └───────────────────────────────────────────────────────
       0  T_0  2*T_0      4*T_0           8*T_0        epochs
       |←1→|←──2──→|←────4────→|←────────8────────→|
       Cada ciclo dura el DOBLE que el anterior
```

### 2.6. Incumbent Solution (solucion recomendada)

Un detalle importante: despues de cada warm restart, el learning rate sube bruscamente, lo que **empeora temporalmente** el rendimiento. Por eso, los autores recomiendan no usar siempre el ultimo modelo sino la **incumbent solution**:

```text
- Durante el primer ciclo: la recomendacion es x_t (el modelo actual)
- Despues del primer restart: la recomendacion es el modelo
  obtenido al FINAL del ultimo ciclo completado (cuando eta_t = eta_min)

→ No requiere conjunto de validacion separado para decidir
```

---

## 3. Por Que Funciona: Intuicion

### 3.1. Cosine annealing vs step decay

```text
STEP DECAY:
  - Transiciones abruptas → el modelo se "sacude" repentinamente
  - Los epochs entre escalones son "desperdiciados" (lr constante, poca exploracion)
  - Requiere definir manualmente los puntos de cambio

COSINE ANNEALING:
  - Transicion SUAVE de lr alto a lr bajo
  - Fase inicial (lr alto): EXPLORA el espacio de parametros
  - Fase final (lr bajo): CONVERGE suavemente a un minimo
  - Un solo hiperparametro: T_i (duracion del ciclo)
```

### 3.2. Warm restarts como mecanismo de escape

```text
  Loss
  landscape:

       ╱╲          ╱╲
      ╱  ╲   ╱╲   ╱  ╲
     ╱    ╲ ╱  ╲ ╱    ╲
    ╱      A    B      ╲        A = minimo local angosto (sharp)
   ╱                    ╲       B = minimo local ancho (flat)
  ╱                      ╲

  Sin restarts:
    SGD converge a A (el minimo mas cercano) y se queda ahi.

  Con warm restart:
    1. SGD converge a A (lr baja)
    2. RESTART: lr sube → el modelo "salta" fuera de A
    3. Si A es un minimo angosto, el salto lo saca facilmente
    4. B (minimo ancho) retiene mejor al modelo porque
       el gradiente no es tan pronunciado en sus bordes
    5. SGD converge a B → mejor generalizacion

  Minimos ANCHOS (flat) generalizan mejor que los ANGOSTOS (sharp)
  porque son mas robustos a perturbaciones en los datos.
```

### 3.3. Anytime performance

```text
  El concepto de "anytime performance" es clave en SGDR:

  DEFAULT SCHEDULE:
    Test error
    25│  ****
      │      ****
    20│          ****
      │              ****
    15│                  ********************************
      │                                                  ****
    10│                                                      ****
      │                                                          ****
     5│                                                              ****
      │                                                                  **
     4│
      └──────────────────────────────────────────────────────────────────────
       0     20    40    60    80   100   120   140   160   180   200 epochs
       ↑ El error solo baja significativamente en los 3 step drops.
       Si paras en epoch 50, tienes un modelo MEDIOCRE.

  SGDR:
    Test error
    25│  *
      │   *  *
    20│    *  *  *
      │     *  *  *   *
    15│      *  *  *  * *
      │       *  *  *  * *    *
    10│        *  *  *  *  *   * *
      │         *   *   *   *  *  * *
     5│          *   *   *   *  *  *  ***
      │           *   *   *   *   *       ****
     4│            *   *   *   *              ****
      └──────────────────────────────────────────────────────────────────────
       0     20    40    60    80   100   120   140   160   180   200 epochs
       ↑ El error baja continuamente. Si paras en epoch 50,
       ya tienes un modelo RAZONABLE (buena anytime performance).
```

---

## 4. Resultados Experimentales (Seccion 4)

### 4.1. Configuracion experimental

```text
Red:       Wide Residual Networks (WRN-d-k)
           WRN-28-10: depth=28, widening factor k=10, 36.5M parametros
           WRN-28-20: depth=28, widening factor k=20, 145.8M parametros
Datasets:  CIFAR-10 (10 clases, 50K train, 10K test, 32x32 color)
           CIFAR-100 (100 clases, 50K train, 10K test, 32x32 color)
Optimizer: SGD con momentum (0.9), weight decay 0.0005
Baseline:  Zagoruyko & Komodakis (2016), lr=0.05, step decay en epochs 60, 120, 160
Budget:    200 epochs totales para todas las configuraciones
```

### 4.2. Resultados Single-Model en CIFAR (Tabla 1 del paper)

| Metodo | Red | CIFAR-10 (%) | CIFAR-100 (%) |
|---|---|---|---|
| original-ResNet (He et al., 2015) | 110 capas, 1.7M | 6.43 | 25.16 |
| pre-act-ResNet (He et al., 2016) | 1001 capas, 10.2M | 4.62 | 22.71 |
| WRN default (eta_0=0.05, step decay) | 28-10, 36.5M | 4.13 | 20.21 |
| **SGDR T_0=200, T_mult=1** | 28-10, 36.5M | **3.86** | **19.98** |
| **SGDR T_0=10, T_mult=2** | 28-10, 36.5M | 4.03 | 19.58 |
| WRN default (eta_0=0.05) | 28-20, 145.8M | 3.96 | 19.67 |
| **SGDR T_0=200, T_mult=1** | 28-20, 145.8M | **3.77** | 19.24 |
| **SGDR T_0=10, T_mult=2** | 28-20, 145.8M | **3.74** | **18.70** |

**Observaciones clave**:
- SGDR con T_0=200 (sin restarts, solo cosine annealing) ya supera al step decay default
- Con T_mult=2 y WRN-28-20, SGDR logra **3.74%** en CIFAR-10 y **18.70%** en CIFAR-100
- La mejora mas notable: SGDR alcanza rendimiento comparable **2-4x mas rapido** (buena anytime performance)

### 4.3. Resultados con Ensembles (Tabla 2 del paper)

Una ventaja unica de SGDR: los **snapshots** tomados al final de cada ciclo (cuando eta_t = eta_min) son modelos diversos que se pueden combinar en un ensemble "gratis":

| Configuracion | CIFAR-10 (%) | CIFAR-100 (%) |
|---|---|---|
| 1 run, 1 snapshot (baseline SGDR) | 4.03 | 19.57 |
| 1 run, 3 snapshots (ensemble gratis) | 3.51 | 17.75 |
| 3 runs, 3 snapshots c/u | 3.25 | 16.64 |
| **16 runs, 3 snapshots c/u** | **3.14** | **16.21** |

```text
  ENSEMBLE CON SNAPSHOTS DE SGDR:

  lr
  0.05|*         *              *
      | *        * *            * *
      |  **      *  **          *  **
      |    ***   *    ***       *    ***
      |       ***       ***    *       ****
  0.0 |         S1        S2  S3          S4    ← Snapshots
      └──────────────────────────────────────────
       0     T_0    2*T_0    3*T_0     4*T_0

  S1, S2, S3, S4 = modelos guardados al final de cada ciclo
  Ensemble = promedio de las predicciones de S1, S2, S3, S4

  Ventaja: Un solo entrenamiento produce MULTIPLES modelos
  para ensemble, sin costo adicional de entrenamiento.
```

**State-of-the-art al momento de publicacion**:
- **3.14%** en CIFAR-10 (mejor resultado publicado)
- **16.21%** en CIFAR-100 (mejor resultado publicado)

### 4.4. Experimentos en EEG (Seccion 4.4)

Para demostrar generalidad, los autores probaron SGDR en un dominio completamente diferente: clasificacion de actividad cerebral (EEG, 14 sujetos, ~1000 trials por sujeto).

```text
Resultados:
  - SGDR logra rendimiento similar al baseline SIN necesidad de
    definir el numero total de epochs de antemano
  - Con snapshots: mejora de 1-2% sobre un solo modelo
  - Con snapshots de multiples hyperparameters: mejora de 2-3%
```

### 4.5. Experimentos en ImageNet downsampled (Seccion 4.5)

```text
Dataset: ImageNet completo (1000 clases) pero con imagenes reducidas a 32x32 pixels
Red: WRN-28-10
Configuraciones de eta_max: 0.050, 0.025, 0.01, 0.005

Resultados con SGDR (T_0=10, T_mult=2, eta_max=0.01):
  Top-1 error: 39.24%
  Top-5 error: 17.17%

→ Comparable a AlexNet (40.7% top-1, 18.2% top-5) entrenado
  en imagenes FULL SIZE (50x mas pixeles por imagen)
→ SGDR demuestra mejor anytime performance que el schedule default
```

### 4.6. SGDR vs Default: Overfitting

```text
El paper muestra (Figura 7 en el apendice) que:

  DEFAULT SCHEDULE:
    - Training loss baja rapidamente
    - Test error SUBE despues de epoch ~120 → OVERFITTING
    - Gap train-test crece con el tiempo

  SGDR:
    - Training loss oscila (por los restarts)
    - Test error baja de forma mas estable
    - Overfitting MUY LEVE comparado con el default
    - Los restarts actuan como una forma de REGULARIZACION implicita
```

---

## 5. Impacto Historico y Legado

```text
2016 (Ago): Paper publicado en arXiv
  → Introduce cosine annealing con warm restarts para SGD

2016 (Nov): Huang et al. (2016a) - "Snapshot Ensembles"
  → Directamente inspirado por SGDR
  → Usan cosine annealing cycles para crear ensembles "gratis"
  → Citan explicitamente a SGDR como base

2017 (Feb): Publicado en ICLR 2017
  → Se establece como referencia en optimizacion de redes

2017: Loshchilov & Hutter publican "Decoupled Weight Decay" (AdamW)
  → Usan cosine annealing de SGDR como scheduler default
  → AdamW + cosine annealing = combo dominante en NLP/Transformers

2018+: Adopcion masiva en la comunidad
  - PyTorch implementa:
    * torch.optim.lr_scheduler.CosineAnnealingLR
    * torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
  - TensorFlow/Keras tambien adoptan cosine decay

2018-2020: Transformers y BERT
  - Los Transformers (Vaswani et al., 2017) usan warmup + cosine decay
  - BERT, GPT-2, T5 todos usan variantes de cosine annealing
  - El "linear warmup + cosine decay" se vuelve el scheduler default

2020+: Era de Large Language Models
  - GPT-3, LLaMA, Chinchilla usan cosine annealing
  - El scheduler de coseno es el ESTANDAR para LLMs
  - Se prefiere un solo ciclo largo (sin restarts) para LLMs
    pero la base matematica es la misma de SGDR

HOY (2026):
  - Cosine annealing es EL scheduler mas usado en Deep Learning
  - Practicamente todo entrenamiento de modelos grandes lo usa
  - La formula del paper (Ecuacion 5) esta implementada en
    TODOS los frameworks de Deep Learning
  - Las variantes con warm restarts se usan para fine-tuning
    y transfer learning
```

### Uso en PyTorch (codigo referencial)

```text
# Cosine Annealing sin restarts (un solo ciclo):
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=200,        # duracion total del ciclo
    eta_min=0         # learning rate minimo
)

# Cosine Annealing CON warm restarts (SGDR):
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,           # duracion del primer ciclo
    T_mult=2,         # factor de multiplicacion
    eta_min=0         # learning rate minimo
)

# Uso tipico en loop de entrenamiento:
for epoch in range(200):
    train(...)
    scheduler.step()
```

---

## 6. Comparacion con Otros Schedulers

```text
| Scheduler         | Formula                      | Ventajas              | Desventajas            |
|-------------------|------------------------------|-----------------------|------------------------|
| Step Decay        | lr = lr * gamma cada N eps   | Simple                | Rigido, mal anytime    |
| Exponential Decay | lr = lr_0 * gamma^t          | Suave                 | Decae demasiado rapido |
| Linear Decay      | lr = lr_0 * (1 - t/T)        | Predecible            | Sin exploracion        |
| Cosine Annealing  | lr = 1/2*(1+cos(t/T*pi))     | Suave, buen anytime   | Un solo ciclo          |
| SGDR (este paper) | Cosine + warm restarts       | Anytime, ensembles    | Mas hyperparameters    |
| OneCycleLR        | Warmup + cosine              | Estandar en LLMs      | Sin restarts           |
```

---

## 7. Resumen en Una Pagina

```text
PROBLEMA:  Los schedules de learning rate clasicos (step decay) son rigidos
           y tienen mal "anytime performance" — si paras el entrenamiento
           antes de tiempo, el modelo puede ser mediocre.

SOLUCION:  SGDR — Cosine Annealing con Warm Restarts

FORMULA CENTRAL:
  eta_t = eta_min + 1/2 * (eta_max - eta_min) * (1 + cos(T_cur/T_i * pi))

  donde T_cur = epochs desde el ultimo restart
        T_i   = duracion del ciclo i-esimo = T_0 * T_mult^(i-1)

COMO FUNCIONA:
  1. El lr decae suavemente siguiendo una curva coseno de eta_max a eta_min
  2. Al final de cada ciclo, el lr se REINICIA a eta_max (warm restart)
  3. Cada ciclo puede ser mas largo que el anterior (T_mult > 1)
  4. Los restarts permiten escapar de minimos locales angostos
  5. Los snapshots al final de cada ciclo forman ensembles "gratis"

HIPERPARAMETROS CLAVE:
  - eta_max: learning rate inicial (tipicamente 0.05)
  - eta_min: learning rate minimo (tipicamente 0)
  - T_0:     duracion del primer ciclo (1, 10, 50, 100 o 200 epochs)
  - T_mult:  factor multiplicativo (1 = fijo, 2 = duplicar cada ciclo)

RESULTADOS:
  - CIFAR-10:  3.86% single model (vs 4.13% default) — mejora 6.5%
  - CIFAR-10:  3.14% ensemble (state-of-the-art 2017)
  - CIFAR-100: 18.70% single model (vs 20.21% default) — mejora 7.5%
  - CIFAR-100: 16.21% ensemble (state-of-the-art 2017)
  - 2-4x mas rapido en alcanzar buen rendimiento (anytime performance)
  - Menos overfitting que el schedule default

VENTAJA UNICA:
  Los modelos al final de cada ciclo son "diversos" y se pueden
  combinar en un ensemble SIN costo adicional de entrenamiento.
  Un solo entrenamiento → multiples modelos para ensemble.

LEGADO:
  - CosineAnnealingWarmRestarts en PyTorch
  - Base del scheduler estandar para Transformers y LLMs
  - Complementa a AdamW (del mismo autor) como combo dominante
  - Influyo en Snapshot Ensembles, OneCycleLR, y warmup+cosine decay
  - La formula del coseno es hoy EL scheduler mas usado en Deep Learning
```
