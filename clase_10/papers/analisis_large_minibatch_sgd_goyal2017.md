# Analisis del Paper: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour

**Autores**: Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He
**Institucion**: Facebook (ahora Meta) AI Research
**Publicado en**: arXiv:1706.02677, 2017
**Conferencia relacionada**: Resultados presentados en contexto de CVPR/ICLR

> PDF descargado en: [papers/7_LargeMinibatchSGD_Goyal2017.pdf](7_LargeMinibatchSGD_Goyal2017.pdf)

---

## Datos Clave del Paper

| Aspecto | Detalle |
|---|---|
| **Ano** | 2017 (arXiv junio 2017, v2 abril 2018) |
| **Citas** | Uno de los papers mas influyentes en entrenamiento distribuido (~6,000+ citas) |
| **Autores notables** | Kaiming He (ResNet, Mask R-CNN), Ross Girshick (R-CNN, Fast/Faster R-CNN), Piotr Dollar (FAIR) |
| **Idea central** | Escalar SGD a minibatches muy grandes (hasta 8192) sin perder accuracy, usando una linear scaling rule y gradual warmup |
| **Impacto** | Entrenaron ResNet-50 en ImageNet en **1 hora** con 256 GPUs, estableciendo el paradigma para entrenamiento distribuido a gran escala |

---

## 1. Problema que Resuelve

El Deep Learning mejora consistentemente con mas datos y modelos mas grandes, pero esto incrementa dramaticamente el **tiempo de entrenamiento**. El enfoque natural es usar **entrenamiento distribuido** con SGD sincronico, dividiendo los minibatches entre multiples GPUs (data parallelism).

```text
PROBLEMA CENTRAL:

  Modelo grande + Dataset grande = Tiempo de entrenamiento LARGO
  
  Solucion obvia: usar mas GPUs en paralelo (data parallelism)
  
  PERO: Al usar mas GPUs, el minibatch TOTAL crece
         (cada GPU procesa su parte)
  
  Minibatch grande → Problemas de OPTIMIZACION
                    → La accuracy se DEGRADA
                    → El entrenamiento DIVERGE

  EJEMPLO CONCRETO:
    Baseline:  8 GPUs x 32 imgs/GPU = minibatch 256  → 29 horas
    Objetivo: 256 GPUs x 32 imgs/GPU = minibatch 8192 → ~1 hora
    
    Sin tecnicas especiales, el minibatch de 8192
    produce PEOR accuracy que el de 256
```

**Estado del arte en 2017**:
- Se sabia que minibatches grandes degradaban la accuracy
- Krizhevsky (2014) propuso una linear scaling rule pero reporto 1% de perdida de error al escalar de 128 a 1024
- No existia una guia practica y robusta para escalar a minibatches de miles de imagenes
- Algunos trabajos (Keskar et al., 2017) argumentaban que minibatches grandes encontraban **sharp minima** que generalizaban peor

**Contribucion clave del paper**: Demostrar que el problema es de **optimizacion** (no de generalizacion), y que con tecnicas simples se puede escalar sin perdida de accuracy.

---

## 2. Modelo Formal

### 2.1. SGD Estandar con Minibatches

El aprendizaje supervisado busca minimizar una funcion de perdida:

```text
L(w) = (1/|X|) * SUM_{x in X} l(x, w)

donde:
  w     = pesos de la red
  X     = conjunto de entrenamiento
  l(x,w) = perdida por muestra (ej: cross-entropy + regularizacion)
```

Minibatch SGD actualiza los pesos en cada iteracion t:

```text
w_{t+1} = w_t - eta * (1/n) * SUM_{x in B} grad_l(x, w_t)

donde:
  B   = minibatch (subconjunto de X)
  n   = |B| = tamano del minibatch
  eta = learning rate
```

### 2.2. Linear Scaling Rule

La idea central del paper, sorprendentemente simple:

```text
+----------------------------------------------------------+
|  LINEAR SCALING RULE:                                     |
|                                                           |
|  Cuando el tamano del minibatch se MULTIPLICA por k,      |
|  MULTIPLICAR el learning rate por k.                      |
|                                                           |
|  Todos los demas hiperparametros se mantienen IGUALES.    |
+----------------------------------------------------------+
```

**Justificacion matematica**:

Consideremos k iteraciones de SGD con minibatch pequeno de tamano n y learning rate eta:

```text
CASO 1: k iteraciones con minibatch PEQUENO (tamano n, learning rate eta)

  w_{t+k} = w_t - eta * (1/n) * SUM_{j<k} SUM_{x in B_j} grad_l(x, w_{t+j})
                                                              ^^^^^^^^^
                                                  Nota: los gradientes se evaluan
                                                  en DIFERENTES puntos w_{t+j}

CASO 2: 1 iteracion con minibatch GRANDE (tamano kn, learning rate eta_hat)

  w_hat_{t+1} = w_t - eta_hat * (1/kn) * SUM_{j<k} SUM_{x in B_j} grad_l(x, w_t)
                                                                         ^^^
                                                              Todos los gradientes
                                                              evaluados en el MISMO w_t
```

**Hipotesis clave**: Si asumimos que los gradientes no cambian mucho entre iteraciones consecutivas:

```text
  grad_l(x, w_t) ≈ grad_l(x, w_{t+j})   para j < k

  Entonces, si eta_hat = k * eta:

  w_hat_{t+1} = w_t - (k*eta)/(kn) * SUM_j SUM_x grad_l(x, w_t)
              = w_t - eta/n * SUM_j SUM_x grad_l(x, w_t)
              ≈ w_{t+k}

  → ¡Las actualizaciones son EQUIVALENTES!
```

**Ejemplo numerico paso a paso**:

```text
CONFIGURACION BASE (1 GPU):
  n = 256 imagenes por minibatch
  eta = 0.1
  k = 1 (1 GPU)

ESCALANDO A 8 GPUs:
  Cada GPU procesa 32 imagenes → total kn = 8 * 32 = 256
  Pero queremos kn = 8 * 256 = 2048
  k = 2048/256 = 8
  eta_hat = 8 * 0.1 = 0.8

ESCALANDO A 256 GPUs:
  k = 256, n = 32 por GPU → kn = 8192
  Razon: kn/256 = 8192/256 = 32
  eta_hat = 0.1 * (8192/256) = 0.1 * 32 = 3.2

  FORMULA PRACTICA:
  eta_ref = 0.1 * (kn / 256)
  
  kn = 256  → eta = 0.1
  kn = 512  → eta = 0.2
  kn = 1024 → eta = 0.4
  kn = 2048 → eta = 0.8
  kn = 4096 → eta = 1.6
  kn = 8192 → eta = 3.2
```

### 2.3. Gradual Warmup Strategy

La linear scaling rule falla al inicio del entrenamiento porque los pesos cambian rapidamente y la aproximacion `grad_l(x, w_t) ≈ grad_l(x, w_{t+j})` no se cumple. La solucion es un **warmup gradual**:

```text
ESTRATEGIA DE WARMUP GRADUAL:

  Epoca 0   1   2   3   4   5   6   7 ...
  eta:  |---|---|---|---|---|------------...
        0.1                 3.2
         \                 /
          \   RAMPA       /
           \ LINEAL     /
            \         /
             -------
  
  - Empezar con eta = 0.1 (learning rate base)
  - Incrementar LINEALMENTE durante 5 epochs
  - Llegar a eta_hat = k * eta = 3.2 al final del epoch 5
  - Despues del warmup, usar el schedule normal
    (reducir eta por 1/10 en epochs 30, 60, 80)

  INCREMENTO POR ITERACION:
    delta_eta = (eta_hat - eta) / (num_iteraciones_en_5_epochs)
    
    Con ~5005 iteraciones por epoch (1.28M imagenes / 256):
    delta_eta = (3.2 - 0.1) / (5 * 5005) ≈ 0.000124 por iteracion
```

**Comparacion de las tres estrategias de warmup**:

```text
  LEARNING RATE vs EPOCHS:

  eta
  3.2 |                          ........... gradual warmup
      |                      ...
      |                  ...
      |              ...        __________ constant warmup
      |          ...            |
      |      ...                |
      |  ...                    |
  0.1 |.........________________|__________ no warmup
      |                                     (empieza directo en 3.2)
      +----+----+----+----+----+----+----→ epochs
      0    1    2    3    4    5    6    7


  RESULTADOS EN IMAGENET (ResNet-50, kn=8192):

  | Estrategia       | top-1 error (%) |
  |-------------------|-----------------|
  | Baseline (kn=256) | 23.60 +/- 0.12  |
  | No warmup         | 24.84 +/- 0.37  | ← +1.2% peor
  | Constant warmup   | 25.88 +/- 0.56  | ← +2.3% peor (!)
  | Gradual warmup    | 23.74 +/- 0.09  | ← Solo +0.14% peor
```

### 2.4. Batch Normalization con Large Minibatches

Batch Normalization (BN) introduce una complicacion sutil: la perdida de cada muestra **depende de las otras muestras** en el minibatch (a traves de las estadisticas de media y varianza).

```text
SIN Batch Normalization:
  l(x, w) es independiente de otras muestras
  → Se puede dividir el minibatch libremente entre GPUs

CON Batch Normalization:
  l_B(x, w) depende de TODO el minibatch B
  → Las estadisticas de BN (media, varianza) cambian
    segun el tamano n del minibatch

  SOLUCION DEL PAPER:
  ┌─────────────────────────────────────────────────┐
  │ Mantener n = 32 (BN batch size por GPU) FIJO    │
  │ independientemente del minibatch total kn.       │
  │                                                  │
  │ Las estadisticas de BN se computan LOCALMENTE    │
  │ en cada GPU, NO a traves de todos los workers.   │
  └─────────────────────────────────────────────────┘

  ¿Por que?
  - Con n fijo, cada GPU ve minibatches de tamano 32
  - Las estadisticas de BN son las MISMAS que en el baseline
  - La funcion de perdida subyacente NO CAMBIA
  - Si se usara BN sobre todo el minibatch (kn=8192),
    la funcion de perdida seria DIFERENTE
```

### 2.5. Inicializacion gamma de BN

Tecnica adicional para facilitar la optimizacion con minibatches grandes:

```text
BN ESTANDAR:
  y = gamma * (x - mu)/sigma + beta
  Inicializacion: gamma = 1 para TODAS las capas BN

PROPUESTA DEL PAPER:
  gamma = 0 para la ULTIMA capa BN de cada bloque residual
  gamma = 1 para todas las demas capas BN

  EFECTO: Al inicio del entrenamiento, cada bloque residual
  se comporta como la IDENTIDAD:

  Bloque residual:  y = x + F(x)
  Con gamma=0:      y = x + 0 * (...) = x    (identidad pura)

  → La red empieza siendo "mas simple" (como si tuviera menos capas)
  → Los gradientes fluyen limpiamente por los shortcuts
  → La optimizacion al inicio es mas estable
  → Particularmente util con minibatches grandes

  RESULTADOS:
  | kn   | gamma=1 (estandar) | gamma=0 (propuesto) |
  |------|--------------------|---------------------|
  | 256  | 23.84 +/- 0.18     | 23.60 +/- 0.12      |
  | 8192 | 24.11 +/- 0.07     | 23.74 +/- 0.09      |
  
  Mejora mayor para minibatches grandes (+0.37 vs +0.24)
```

---

## 3. Subtleties and Pitfalls de SGD Distribuido (Seccion 3)

El paper detalla varias trampas de implementacion que pueden degradar silenciosamente la accuracy:

### 3.1. Weight Decay

```text
INCORRECTO: Escalar la cross-entropy loss y creer que es
            equivalente a escalar el learning rate

  l(x,w) = (lambda/2) * ||w||^2 + epsilon(x,w)
                ^^^^^^^^^^^^         ^^^^^^^^^^^
                weight decay     cross-entropy (depende de datos)

  Actualizacion correcta:
  w_{t+1} = w_t - eta*lambda*w_t - eta*(1/n)*SUM grad_epsilon(x, w_t)

  REMARK 1: Escalar la cross-entropy loss NO es equivalente
  a escalar el learning rate cuando hay weight decay.
  Solo el learning rate debe escalarse.
```

### 3.2. Momentum Correction

```text
SGD con Momentum (formulacion de referencia):

  u_{t+1} = m * u_t + (1/n) * SUM grad_l(x, w_t)
  w_{t+1} = w_t - eta * u_{t+1}

  Formulacion alternativa (absorbiendo eta en v):
  v_{t+1} = m * v_t + eta * (1/n) * SUM grad_l(x, w_t)
  w_{t+1} = w_t - v_{t+1}

  PROBLEMA: Cuando eta cambia (ej. durante warmup),
  la segunda forma necesita CORRECCION DE MOMENTUM:

  v_{t+1} = m * (eta_{t+1}/eta_t) * v_t + eta_{t+1} * (1/n) * SUM grad
                  ^^^^^^^^^^^^^^^^
                  factor de correccion

  REMARK 2: Aplicar momentum correction despues de
  cambiar el learning rate (si se usa la forma con v).

  Sin esta correccion, el termino historico v_t tiene
  la magnitud incorrecta y causa inestabilidad.
```

### 3.3. Gradient Aggregation

```text
Con k GPUs, cada una con minibatch local de n muestras:

  CORRECTO:   Normalizar por el minibatch TOTAL kn
  INCORRECTO: Normalizar por el minibatch LOCAL n

  IMPLEMENTACION PRACTICA:
  - Cada GPU computa: SUM l(x, w_t) / n  (perdida local promedio)
  - allreduce SUMA los gradientes de k GPUs
  - Resultado: SUM de k promedios locales
  - Se debe dividir por k para obtener el promedio global
  
  TRUCO: Usar allreduce con promedio (no suma),
  o escalar la perdida por 1/k antes del allreduce.

  REMARK 3: Normalizar la per-worker loss por el
  minibatch TOTAL kn, no por el per-worker size n.
```

### 3.4. Data Shuffling

```text
  REMARK 4: Usar UN SOLO random shuffle del training set
  por epoch, distribuido entre todos los k workers.

  INCORRECTO: Cada GPU shufflea independientemente
  → Puede causar que algunas muestras se vean
    multiples veces y otras no se vean nunca
  → Resultados inconsistentes y no reproducibles
```

---

## 4. Sistema de Comunicacion (Seccion 4)

### 4.1. Arquitectura de Hardware

```text
INFRAESTRUCTURA FACEBOOK BIG BASIN:

  ┌─────────────────────────────────────────────┐
  │               SERVIDOR (Big Basin)           │
  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │
  │  │ P100│ │ P100│ │ P100│ │ P100│           │
  │  │ GPU │ │ GPU │ │ GPU │ │ GPU │           │
  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘           │
  │     │       │       │       │    NVLink      │
  │  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐           │
  │  │ P100│ │ P100│ │ P100│ │ P100│           │
  │  │ GPU │ │ GPU │ │ GPU │ │ GPU │           │
  │  └─────┘ └─────┘ └─────┘ └─────┘           │
  │           8 GPUs por servidor                │
  │           Mellanox 50Gbit Ethernet           │
  └──────────────────┬──────────────────────────┘
                     │
        Wedge100 Ethernet switches
                     │
  ┌──────────────────┴──────────────────────────┐
  │  32 servidores = 256 GPUs                    │
  │  (44 servidores para 352 GPUs max)           │
  └─────────────────────────────────────────────┘

  Ancho de banda necesario para ResNet-50:
  - 25M parametros * 4 bytes = 100MB
  - Backprop toma 120ms en 1 GPU
  - allreduce necesita ~2x los datos = 200MB
  - Bandwidth pico: 200MB / 0.125s ≈ 1600 MB/s ≈ 12.8 Gbit/s
  - Con overhead: ~15 Gbit/s (cubierto por 50Gbit Ethernet)
```

### 4.2. allreduce Eficiente

```text
PIPELINE DE COMUNICACION:

  Backprop                allreduce
  Capa N  ████             ████  → agregacion capa N
  Capa N-1 ████████         ████  → agregacion capa N-1
  Capa N-2 ████████████     ████  → agregacion capa N-2
  ...

  Los gradientes de cada capa se agregan EN PARALELO
  con el backprop de la capa anterior (pipelining).
  → Oculta la mayor parte de la latencia de comunicacion.

  Tres fases del allreduce:
  1. Intra-servidor: 8 GPUs → 1 buffer (via NCCL)
  2. Inter-servidor: buffers compartidos y sumados
     (halving/doubling algorithm para latency-limited,
      bucket/ring algorithm para bandwidth-limited)
  3. Broadcast: resultado final a cada GPU (via NCCL)
```

---

## 5. Resultados Experimentales (Seccion 5)

### 5.1. Configuracion Experimental

```text
SETUP:
  - Modelo: ResNet-50
  - Dataset: ImageNet (ILSVRC-2012)
    - ~1.28 millones de imagenes de entrenamiento
    - 50,000 imagenes de validacion
    - 1000 clases
  - Input: 224x224 pixeles (crop aleatorio + flip horizontal)
  - Entrenamiento: 90 epochs
  - Optimizador: Nesterov Momentum SGD (m=0.9)
  - Weight decay: lambda = 0.0001
  - BN batch size: n = 32 (fijo por GPU)
  - Schedule: reducir eta por 1/10 en epochs 30, 60, 80
  - Metrica: top-1 validation error (mediana de ultimos 5 epochs)
  - Cada resultado: promedio +/- std de 5 runs independientes
```

### 5.2. Resultado Principal: Minibatch 8k iguala al Baseline

```text
TABLA DE RESULTADOS PRINCIPALES:

  | Configuracion     | k GPUs | n/GPU | kn total | eta  | top-1 error (%) |
  |-------------------|--------|-------|----------|------|-----------------|
  | Baseline          | 8      | 32    | 256      | 0.1  | 23.60 +/- 0.12  |
  | No warmup         | 256    | 32    | 8192     | 3.2  | 24.84 +/- 0.37  |
  | Constant warmup   | 256    | 32    | 8192     | 3.2  | 25.88 +/- 0.56  |
  | Gradual warmup    | 256    | 32    | 8192     | 3.2  | 23.74 +/- 0.09  |
                                                          ^^^^^^^^^^^^^^^^^
                                                   ¡Solo 0.14% mas que baseline!
                                                   (dentro del ruido estadistico)

  CONCLUSION: Con gradual warmup + linear scaling rule,
  un minibatch de 8192 iguala la accuracy de uno de 256.
```

### 5.3. Minibatch Size vs. Error (curva completa)

```text
  top-1 validation error (%)
  40 |                                                    *
     |                                               *
  35 |                                          *
     |
  30 |
     |                                     *
  25 |                                *
     |  *    *    *    *    *    *
  23 |  ===========================  (zona estable)
  20 |
     +---+----+----+----+----+----+----+----+----+----→
      64  128  256  512  1k   2k   4k   8k  16k  32k  64k
                        minibatch size (kn)

  OBSERVACIONES:
  - De kn=64 a kn=8k: error ESTABLE (~23.5-23.7%)
  - kn=16k: empieza a degradarse (24.79%)
  - kn=32k: error sube a 27.55%
  - kn=64k: error sube a 33.96%
  - El "codo" esta alrededor de kn=8k para ResNet-50
```

### 5.4. Resultados Detallados por Minibatch Size

```text
  | kn     | eta              | top-1 error (%) |
  |--------|------------------|-----------------|
  | 256    | 0.1              | 23.60 +/- 0.12  |
  | 128    | 0.05             | 23.49 +/- 0.12  |
  | 512    | 0.2              | 23.48 +/- 0.09  |
  | 1k     | 0.4              | 23.53 +/- 0.08  |
  | 2k     | 0.8              | 23.49 +/- 0.11  |
  | 4k     | 1.6              | 23.56 +/- 0.12  |
  | 8k     | 3.2              | 23.74 +/- 0.09  |
  | 16k    | 6.4              | 24.79 +/- 0.27  |
  | 32k    | 12.8             | 27.55 +/- 0.28  |
  | 64k    | 25.6             | 33.96 +/- 0.80  |

  Las curvas de training error COINCIDEN con el baseline
  para kn <= 8k (despues del periodo de warmup).
  Para kn >= 16k, divergen desde el inicio.
```

### 5.5. Optimizacion vs. Generalizacion

```text
HALLAZGO CLAVE DEL PAPER:

  El problema con minibatches grandes es de OPTIMIZACION,
  NO de GENERALIZACION.

  Evidencia:
  1. Sin warmup (kn=8k): training error +2.1% mayor
     → Dificultad de optimizacion, no overfitting
  
  2. Con gradual warmup: training curves COINCIDEN
     → Se resuelve la dificultad de optimizacion
     → Y la accuracy de validacion tambien coincide
  
  3. Transfer learning (ImageNet → COCO):
     - Modelos pre-entrenados con kn=256 a kn=8k
       dan MISMA accuracy en deteccion/segmentacion
     → No hay degradacion de generalizacion

  ┌──────────────────────────────────────────────────┐
  │  "Si las dificultades de optimizacion se         │
  │   resuelven, NO hay degradacion aparente         │
  │   de generalizacion con minibatches grandes,     │
  │   incluso escalando de kn=256 a kn=8k."          │
  └──────────────────────────────────────────────────┘
```

### 5.6. ResNet-101

```text
  | Modelo     | kn   | eta  | top-1 error (%) |
  |------------|------|------|-----------------|
  | ResNet-101 | 256  | 0.1  | 22.08 +/- 0.06  |
  | ResNet-101 | 8k   | 3.2  | 22.36 +/- 0.09  |

  Diferencia: solo 0.28%
  Tiempo de entrenamiento: 92.5 minutos con 256 GPUs
  
  → El minibatch de 8k esta en el limite del regimen util
    para ResNet-101 (similar a ResNet-50)
```

### 5.7. Generalizacion a Deteccion y Segmentacion (COCO)

```text
TRANSFER LEARNING: ImageNet pre-training → Mask R-CNN en COCO

  | kn pre-training | ImageNet error | COCO box AP | COCO mask AP |
  |-----------------|----------------|-------------|--------------|
  | 256             | 23.60 +/- 0.12 | 35.9 +/- 0.1| 33.9 +/- 0.1|
  | 512             | 23.48 +/- 0.09 | 35.8 +/- 0.1| 33.8 +/- 0.2|
  | 1k              | 23.53 +/- 0.08 | 35.9 +/- 0.2| 33.9 +/- 0.2|
  | 2k              | 23.49 +/- 0.11 | 35.9 +/- 0.1| 33.9 +/- 0.1|
  | 4k              | 23.56 +/- 0.12 | 35.8 +/- 0.1| 33.8 +/- 0.1|
  | 8k              | 23.74 +/- 0.09 | 35.8 +/- 0.1| 33.9 +/- 0.2|
  | 16k             | 24.79 +/- 0.27 | 35.1 +/- 0.3| 33.2 +/- 0.3|

  CONCLUSION: Mientras la accuracy de ImageNet se mantenga
  baja (kn <= 8k), la generalizacion a deteccion y
  segmentacion NO SE DEGRADA.
  
  Incluso la linear scaling rule aplicada directamente
  a Mask R-CNN (1→8 GPUs) mantiene box/mask AP constante.
```

### 5.8. Tiempos de Ejecucion

```text
TIEMPO POR EPOCH:

  kn=256  (8 GPUs, 1 servidor):   ~16 min/epoch
  kn=8192 (256 GPUs, 32 servidores): ~0.5 min/epoch (~30 seg)
  
  TOTAL (90 epochs):
  kn=256:  ~29 horas (baseline)
  kn=8192: ~1 hora   (256 GPUs con gradual warmup)
  
  EFICIENCIA DE ESCALADO:
  
  images/segundo
  32k |                                          * actual
      |                                      * ideal
  16k |                               *  *
      |                          * *
   8k |                     * *
      |                * *
   4k |           * *
      |       * *
   2k |   * *
      | **
   1k |*
      +--+----+----+----+----+----+----→
       8  16   32   64  128  256  352
                    # GPUs

  Eficiencia: ~90% de escalado lineal perfecto
  (8 GPUs → 256 GPUs)
  
  Overhead: solo ~12% mas tiempo por iteracion
  al escalar el minibatch 44x (de 256 a 11264).
  La mayor parte de la comunicacion se OCULTA
  mediante pipelining con el backprop.
```

---

## 6. Comparacion de Learning Rate Rules

```text
TABLA COMPARATIVA (kn=8192, ResNet-50):

  | Regla                        | eta          | top-1 error (%)  |
  |------------------------------|-------------|------------------|
  | Linear scaling (propuesta)   | 0.1 * 32    | 23.74 +/- 0.09   | MEJOR
  | eta fijo (no escalar)        | 0.10        | 41.67 +/- 0.10   | Terrible
  | Square root scaling          | 0.1*sqrt(32)| 26.22 +/- 0.03   | Peor
  
  Para kn=256 (baseline):
  | eta = 0.05                   | 23.92 +/- 0.10 |
  | eta = 0.10                   | 23.60 +/- 0.12 | MEJOR para baseline
  | eta = 0.20                   | 23.68 +/- 0.09 |

  → La linear scaling rule es claramente superior
  → Eta fijo con minibatch grande: desastre total
  → Square root scaling: mucho peor que linear
```

---

## 7. Impacto Historico y Legado

```text
2014: Krizhevsky propone linear scaling rule informalmente
      pero reporta 1% de degradacion (minibatch 128→1024)

2017: ESTE PAPER:
      → Demuestra linear scaling rule hasta kn=8k SIN degradacion
      → Introduce gradual warmup (ahora tecnica estandar)
      → Resuelve sutilezas de BN, weight decay, momentum
      → Entrena ImageNet en 1 HORA (record en su momento)
      → Demuestra que el problema es optimizacion, no generalizacion

2017-2018: Carrera por entrenar ImageNet cada vez mas rapido:
      → You et al. (LARS/LAMB): 32 minutos (2017)
      → Akiba et al.: 15 minutos (2018)
      → Jia et al. (Tencent): 6.6 minutos con 2048 GPUs (2018)
      → Sony: 224 segundos con 3456 GPUs (2018)

2018+: Las tecnicas de este paper se vuelven ESTANDAR:
      → Linear scaling rule: adoptada en PyTorch, TensorFlow
      → Gradual warmup: usado en BERT, GPT, ViT, y casi
        todo modelo moderno de Deep Learning
      → BN local (por GPU): practica estandar

2020+: LLMs y Foundation Models:
      → Los principios de este paper son la BASE del
        entrenamiento distribuido de GPT-3, LLaMA, etc.
      → Warmup es UBICUO en el entrenamiento de Transformers
      → La idea de "escalar el LR con el batch size"
        influyo en el desarrollo de LARS, LAMB, y
        otros optimizadores para large-batch training

CONTRIBUCION DURADERA:
  No fue solo un resultado empirico; fue una GUIA PRACTICA
  con recetas claras que democratizo el entrenamiento
  distribuido a gran escala.
```

---

## 8. Resumen en Una Pagina

```text
PROBLEMA:  Entrenar redes profundas con minibatches grandes
           causa degradacion de accuracy por dificultades de
           optimizacion al inicio del entrenamiento.

SOLUCION:  Dos tecnicas simples pero poderosas:

  1. LINEAR SCALING RULE:
     Si multiplicas el minibatch por k, multiplica el LR por k.
     Ejemplo: minibatch 256 con eta=0.1
            → minibatch 8192 con eta=3.2

  2. GRADUAL WARMUP:
     Empezar con LR bajo (eta=0.1) y subir linealmente
     hasta el LR objetivo (eta=3.2) durante 5 epochs.
     Evita la divergencia en las primeras iteraciones.

  + DETALLES CRITICOS DE IMPLEMENTACION:
     - BN: estadisticas locales por GPU (n=32 fijo)
     - Inicializacion gamma=0 en ultima BN de cada residual block
     - Weight decay aplicado correctamente (no escalar la loss)
     - Momentum correction al cambiar el LR
     - Un solo shuffle global por epoch
     - Normalizar loss por minibatch total kn

RESULTADOS CLAVE:
  - ResNet-50 en ImageNet:
    Baseline (kn=256, 8 GPUs):    23.60% error, 29 horas
    Propuesto (kn=8192, 256 GPUs): 23.74% error, 1 HORA
    Diferencia: solo 0.14% (dentro del ruido estadistico)

  - Funciona para kn de 64 a 8192 (estable)
  - Degrada para kn >= 16k (limite de la tecnica)
  - ~90% de eficiencia de escalado (8→256 GPUs)
  - Generaliza a deteccion (Mask R-CNN en COCO)
  - Generaliza a ResNet-101 (solo +0.28% en 92.5 min)

POR QUE FUNCIONA:
  La linear scaling rule hace que un paso grande con
  minibatch kn sea equivalente a k pasos pequenos con
  minibatch n (bajo la hipotesis de gradientes estables).
  El warmup cubre el periodo inicial donde los gradientes
  cambian demasiado rapido para que la hipotesis se cumpla.

LEGADO:
  - Gradual warmup es HOY tecnica estandar en TODO
    entrenamiento de Deep Learning (BERT, GPT, ViT, etc.)
  - La linear scaling rule es la base del entrenamiento
    distribuido moderno
  - Demostro que large batch = problema de OPTIMIZACION,
    no de GENERALIZACION
  - Abrio la puerta al entrenamiento de modelos masivos
    (GPT-3, LLaMA, etc.) en clusters de miles de GPUs
```
