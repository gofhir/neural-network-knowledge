# Analisis del Paper: Cyclical Learning Rates for Training Neural Networks

**Autor**: Leslie N. Smith
**Institucion**: U.S. Naval Research Laboratory, Code 5514, Washington, D.C.
**Publicado en**: arXiv:1506.01186 (2015, revisado Abril 2017)
**Conferencia**: IEEE Winter Conference on Applications of Computer Vision (WACV) 2017

> PDF descargado en: [papers/5_CyclicalLR_Smith2017.pdf](5_CyclicalLR_Smith2017.pdf)

---

## Datos Clave del Paper

| Aspecto | Detalle |
|---|---|
| **Ano** | 2017 (primera version arXiv en Junio 2015) |
| **Citas** | Altamente citado en la comunidad de Deep Learning (>5,000 citas) |
| **Autor notable** | Leslie N. Smith — investigador pionero en politicas de learning rate, autor tambien del paper "Super-Convergence" |
| **Idea central** | Variar ciclicamente el learning rate entre un minimo y un maximo durante el entrenamiento, en lugar de decrementarlo monotonamente |
| **Impacto** | Introdujo el LR range test y las politicas ciclicas que se convirtieron en herramientas estandar; inspiro directamente warm restarts (SGDR) y one-cycle policy |

---

## 1. Problema que Resuelve

El learning rate es el hiperparametro **mas importante** al entrenar redes neuronales profundas. La sabiduria convencional en 2015 dictaba que el learning rate debia ser un valor fijo que decrece monotonamente durante el entrenamiento.

```text
PROBLEMA CLASICO:
  - Elegir el learning rate optimo requiere muchos experimentos
  - Learning rate muy bajo → convergencia lenta, queda atrapado en saddle points
  - Learning rate muy alto → divergencia, el entrenamiento explota
  - Los schedules clasicos (step decay, exponential decay) requieren
    tuning manual de multiples hiperparametros adicionales

METODOS EXISTENTES EN 2015:
  - Fixed LR: un valor constante durante todo el entrenamiento
  - Step decay: reducir LR por un factor cada N epochs
  - Exponential decay: LR = LR_0 * gamma^iteracion
  - Adaptive methods: AdaGrad, RMSProp, Adam
    → Ajustan LR por parametro, pero con costo computacional adicional

LIMITACION FUNDAMENTAL:
  Un LR fijo o que solo decrece puede quedar atrapado en
  saddle points (puntos de silla) del landscape de la loss.
  Los saddle points tienen gradientes pequenos → avance lento.
  Aumentar temporalmente el LR ayuda a atravesarlos.
```

**Solucion propuesta**: Cyclical Learning Rates (CLR) — dejar que el learning rate oscile ciclicamente entre un valor minimo (base_lr) y un maximo (max_lr), lo que:
1. Elimina la necesidad de experimentar para encontrar el LR optimo
2. Permite atravesar saddle points gracias a los incrementos periodicos del LR
3. Alcanza la misma o mejor accuracy en **menos iteraciones**
4. No requiere costo computacional adicional

---

## 2. Intuicion: Por que Funciona Variar el Learning Rate

```text
LANDSCAPE DE LA LOSS FUNCTION:

  Loss
   |
   |  \.                    ./\
   |   \.   saddle point   /   \.
   |    \. . . . . . . . ./     \.
   |     \_____plateau____/       \___minimo___
   |
   └──────────────────────────────────────────── parametros

  Con LR fijo bajo:
    → El optimizador avanza muy lento por el plateau
    → Puede quedar "atrapado" en el saddle point
    → Converge eventualmente, pero tarda muchas iteraciones

  Con CLR (LR que sube y baja ciclicamente):
    → Cuando LR sube: "empuja" al optimizador fuera del saddle point
    → Cuando LR baja: permite precision para acercarse al minimo
    → Efecto neto: convergencia mas rapida y a veces a MEJORES minimos
```

El paper argumenta que aumentar el learning rate puede tener un **efecto negativo a corto plazo** (la loss sube temporalmente), pero produce un **beneficio a largo plazo** (encuentra mejores regiones del espacio de parametros).

Ademas, si los limites base_lr y max_lr se eligen correctamente (via el LR range test), el learning rate optimo estara **siempre dentro del rango**, por lo que se usara durante parte de cada ciclo.

---

## 3. Modelo Formal

### 3.1. Politica Triangular (triangular policy)

La politica base propuesta es la **triangular**: el LR sube linealmente de base_lr a max_lr, y luego baja linealmente de max_lr a base_lr. Esto forma un triangulo:

```text
  LR
  max_lr ──────────/\──────────/\──────────/\
                  /  \        /  \        /  \
                 /    \      /    \      /    \
                /      \    /      \    /      \
  base_lr ────/────────\──/────────\──/────────\──
              |         |  |         |  |         |
              |  ciclo 1|  |  ciclo 2|  |  ciclo 3|
              |<stepsize>| 
              |<--cycle-->|
              
  stepsize = numero de iteraciones en MEDIO ciclo
  cycle = 2 * stepsize iteraciones
```

**Formula matematica**:

```text
cycle = floor(1 + epochCounter / (2 * stepsize))

x = abs(epochCounter / stepsize - 2 * cycle + 1)

lr = base_lr + (max_lr - base_lr) * max(0, 1 - x)
```

**Donde**:
- `epochCounter` = numero de iteracion actual (no epoch, sino iteracion de entrenamiento)
- `stepsize` = numero de iteraciones en medio ciclo
- `cycle` = numero del ciclo actual
- `x` = posicion relativa dentro del ciclo (va de 0 a 1 y vuelve a 0)
- `lr` = learning rate calculado para la iteracion actual

**Ejemplo paso a paso**:

```text
Parametros: base_lr = 0.001, max_lr = 0.006, stepsize = 2000

Iteracion 0:
  cycle = floor(1 + 0/4000) = 1
  x = abs(0/2000 - 2*1 + 1) = abs(-1) = 1
  lr = 0.001 + (0.006 - 0.001) * max(0, 1-1) = 0.001 + 0 = 0.001

Iteracion 1000:
  cycle = floor(1 + 1000/4000) = 1
  x = abs(1000/2000 - 2*1 + 1) = abs(0.5 - 1) = 0.5
  lr = 0.001 + 0.005 * max(0, 1-0.5) = 0.001 + 0.0025 = 0.0035

Iteracion 2000 (pico del ciclo):
  cycle = floor(1 + 2000/4000) = 1
  x = abs(2000/2000 - 2*1 + 1) = abs(0) = 0
  lr = 0.001 + 0.005 * max(0, 1-0) = 0.001 + 0.005 = 0.006  ← max_lr

Iteracion 3000:
  cycle = floor(1 + 3000/4000) = 1
  x = abs(3000/2000 - 2*1 + 1) = abs(0.5) = 0.5
  lr = 0.001 + 0.005 * max(0, 0.5) = 0.001 + 0.0025 = 0.0035

Iteracion 4000 (fin del ciclo 1):
  cycle = floor(1 + 4000/4000) = 2
  x = abs(4000/2000 - 2*2 + 1) = abs(-1) = 1
  lr = 0.001 + 0.005 * max(0, 0) = 0.001  ← base_lr de nuevo
```

### 3.2. Politica Triangular2

Igual que triangular, pero la **amplitud del ciclo se reduce a la mitad** en cada ciclo sucesivo:

```text
  LR
  max_lr ──────/\
              /  \         /\
             /    \       /  \        /\
            /      \     /    \      /  \      /\
  base_lr ─/────────\───/──────\────/────\────/──\────
            ciclo 1     ciclo 2    ciclo 3   ciclo 4

  Formula: lr = base_lr + (max_lr - base_lr) * max(0, 1-x) * (1 / 2^(cycle-1))

  Ciclo 1: amplitud completa     (max_lr - base_lr)
  Ciclo 2: amplitud / 2          (max_lr - base_lr) / 2
  Ciclo 3: amplitud / 4          (max_lr - base_lr) / 4
  ...
```

**Ventaja**: Permite exploracion amplia al inicio y refinamiento progresivo. Al final del entrenamiento, el LR se estabiliza cerca de base_lr, lo que favorece la convergencia fina.

### 3.3. Politica exp_range

Similar a triangular, pero la amplitud decrece exponencialmente con un factor gamma:

```text
  lr = base_lr + (max_lr - base_lr) * max(0, 1-x) * gamma^(iteracion)

  donde gamma es un valor ligeramente menor a 1 (e.g., gamma = 0.99994)
```

```text
  LR
  max_lr ──/\
          /  \.
         /    \. /\
        /      \/  \.  /\
       /        .    \./  \.  /\.
  base_lr──────────────────────\./\.──────
           ciclo 1   ciclo 2   ciclo 3

  La envolvente superior decae exponencialmente con gamma^iteracion
```

### 3.4. Comparacion visual de las tres politicas

```text
  TRIANGULAR:          TRIANGULAR2:         EXP_RANGE:
  /\    /\    /\       /\                   /\
 /  \  /  \  /  \     /  \  /\             /  \.
/    \/    \/    \   /    \/  \ /\        /    \.  /\
                    /         \/  \/\   /       \./  \.
Amplitud constante  Amplitud /2 cada   Amplitud decae
en cada ciclo       ciclo              exponencialmente
```

---

## 4. LR Range Test: Metodo para Encontrar base_lr y max_lr

Este es uno de los aportes mas practicos del paper. El **LR range test** es un procedimiento simple para determinar los limites optimos del learning rate:

### 4.1. Procedimiento

```text
PASO 1: Configurar un entrenamiento corto (unas pocas epochs)
        con el LR aumentando linealmente de un valor muy bajo
        a un valor muy alto.

        Ejemplo: LR va de 0.0001 a 0.02 durante 8 epochs

PASO 2: Graficar accuracy (o loss) vs learning rate

PASO 3: Identificar dos puntos clave:

  Accuracy
  0.6│                    ___________
     │                   /           \
  0.5│                  /             \
     │                 /               \
  0.4│                /                 \
     │               /                   \
  0.3│              /                     \
     │             /                       \
  0.2│     _______/                         \
     │    /                                  \
  0.1│___/                                    \____
     └──────────────────────────────────────────────
     0.0001  0.001    0.005   0.01   0.015   0.02
                    Learning Rate
              ^                      ^
              |                      |
          base_lr               max_lr
     (accuracy empieza        (accuracy empieza
      a subir)                 a bajar/oscilar)
```

### 4.2. Reglas para elegir los limites

```text
Metodo 1 (visual):
  base_lr = LR donde la accuracy empieza a mejorar
  max_lr  = LR donde la accuracy empieza a caer o a oscilar

Metodo 2 (regla practica):
  max_lr  = LR mas grande que aun converge
  base_lr = max_lr / 3  o  max_lr / 4

  Justificacion: el LR optimo esta tipicamente a un factor
  de 2 del LR mas grande que converge [Bengio, 2012]
```

### 4.3. Ejemplo practico con CIFAR-10

```text
LR range test con la arquitectura de Caffe para CIFAR-10:

  - LR crece linealmente de 0 a 0.02 durante 8 epochs
  - La accuracy empieza a subir en LR ~ 0.001
  - La accuracy se vuelve inestable en LR ~ 0.006

  Resultado:
    base_lr = 0.001
    max_lr  = 0.006
```

---

## 5. Eleccion del Tamano del Ciclo (stepsize)

```text
stepsize = numero de iteraciones en medio ciclo

Calculo de iteraciones por epoch:
  iteraciones_por_epoch = num_imagenes_train / batch_size

Ejemplo CIFAR-10:
  50,000 imagenes / 100 batch_size = 500 iteraciones/epoch

Recomendacion del paper:
  stepsize = 2 a 10 veces las iteraciones por epoch

  Ejemplo: stepsize = 2 * 500 = 1,000  (ciclo = 2,000 iteraciones = 4 epochs)
           stepsize = 8 * 500 = 4,000  (ciclo = 8,000 iteraciones = 16 epochs)

Observacion experimental:
  stepsize = 8 * epoch da resultados solo LIGERAMENTE mejores
  que stepsize = 2 * epoch → el metodo es robusto al tamano del ciclo
```

**Recomendacion adicional**: es mejor detener el entrenamiento al final de un ciclo (cuando el LR esta en base_lr y la accuracy esta en su pico).

---

## 6. Resultados Experimentales

### 6.1. CIFAR-10 con Arquitectura de Caffe

| Metodo | Politica de LR | Iteraciones | Accuracy (%) |
|---|---|---|---|
| Baseline | fixed | 70,000 | 81.4 |
| CLR | triangular2 | **25,000** | **81.4** |
| Decay | decay | 25,000 | 78.5 |
| Exponential | exp | 70,000 | 79.1 |
| CLR | exp_range | 42,000 | **82.2** |

**Hallazgos clave**:
- `triangular2` alcanza la misma accuracy (81.4%) en **2.8x menos iteraciones** (25K vs 70K)
- `exp_range` supera a todos con **82.2%**, una mejora absoluta de 0.8% sobre el baseline
- La politica `decay` (solo decrecer) logra solo 78.5% — evidencia de que **subir y bajar** el LR es esencial

### 6.2. CIFAR-10 con Sigmoid + Batch Normalization

```text
  Accuracy
  0.8│ ──────────────────────── BN + CLR (72.2%)
     │
  0.7│
     │ ────────────────── BN sin CLR (60.8%)
  0.6│
     │
  0.5│
     │
     └──────────────────────────────────
     0     1     2     3     4     5     6
                Iteraciones (x 10^4)

  CLR mejora la accuracy de 60.8% a 72.2% (+11.4 puntos)
  con la misma arquitectura y batch normalization.
```

### 6.3. CLR con Metodos Adaptativos (CIFAR-10)

| Metodo | Politica de LR | Iteraciones | Accuracy (%) |
|---|---|---|---|
| Nesterov | fixed | 70,000 | 82.1 |
| Nesterov + CLR | triangular (0.001-0.006) | 25,000 | 81.3 |
| Adam | fixed | 70,000 | 81.4 |
| Adam + CLR | triangular (0.0005-0.002) | 25,000 | 79.8 |
| Adam + CLR | triangular | 70,000 | 81.1 |
| RMSProp | fixed | 70,000 | 75.2 |
| RMSProp + CLR | triangular (0.0001-0.0003) | 25,000 | 72.8 |
| RMSProp + CLR | triangular | 70,000 | 75.1 |
| AdaGrad | fixed | 70,000 | 74.6 |
| AdaGrad + CLR | triangular (0.003-0.035) | 25,000 | 76.0 |
| AdaDelta | fixed | 70,000 | 67.3 |
| AdaDelta + CLR | triangular (0.01-0.1) | 25,000 | 67.3 |

**Observaciones**:
- Para Nesterov y Adam, CLR permite alcanzar en 25K iteraciones la accuracy que toma 70K con LR fijo
- Para AdaGrad, CLR **mejora** el resultado incluso con LR fijo (76.0% vs 74.6%)
- CLR es **compatible** con metodos adaptativos — no son mutuamente excluyentes
- Los beneficios de CLR son mayores con SGD/Nesterov que con Adam (Adam ya adapta el LR internamente)

### 6.4. ResNets, Stochastic Depth y DenseNets (CIFAR-10 y CIFAR-100)

| Arquitectura | CIFAR-10 (LR) | CIFAR-100 (LR) |
|---|---|---|
| ResNet (LR=0.1) | 92.8 | 71.2 |
| ResNet (LR=0.2) | 93.3 | 71.6 |
| ResNet (LR=0.3) | 91.8 | 71.9 |
| **ResNet + CLR (0.1-0.3)** | **93.6** | **72.5** |
| Stochastic Depth (LR=0.1) | 94.6 | 75.2 |
| Stochastic Depth (LR=0.2) | 94.5 | 75.2 |
| Stochastic Depth (LR=0.3) | 94.2 | 74.6 |
| **SD + CLR (0.1-0.3)** | **94.5** | **75.4** |
| DenseNet (LR=0.1) | 94.5 | 75.2 |
| DenseNet (LR=0.2) | 94.5 | 75.3 |
| DenseNet (LR=0.3) | 94.2 | 74.5 |
| **DenseNet + CLR (0.1-0.2)** | **94.9** | **75.9** |

**Hallazgos**:
- CLR mejora o iguala el mejor resultado con LR fijo en **todas** las arquitecturas
- Para DenseNet en CIFAR-10: 94.9% con CLR vs 94.5% con mejor LR fijo (+0.4%)
- El LR range test permite identificar el rango [0.1, 0.3] directamente, eliminando la busqueda manual
- CLR funciona bien incluso con batch normalization

### 6.5. ImageNet

#### AlexNet

```text
  LR range test para AlexNet en ImageNet:

  Accuracy
  0.25│
     │  ___________
  0.20│ /           \
     │/             \
  0.15│               \
     │                 \
  0.10│                  \
     │                    \
  0.05│                     \
     │                      \_____
   0 │
     └────────────────────────────────
     0   0.005 0.01  0.02  0.03  0.04
              Learning Rate

  base_lr ~ 0.005, max_lr ~ 0.02
```

| Metodo | Iteraciones | Top-1 Accuracy (%) |
|---|---|---|
| AlexNet fixed LR | 400,000 | 58.0 |
| AlexNet triangular2 | 400,000 | **58.4** |
| AlexNet exp | 300,000 | 56.0 |
| AlexNet exp (460K) | 460,000 | 56.5 |
| AlexNet exp_range | 300,000 | 56.5 |

#### GoogLeNet

| Metodo | Iteraciones | Accuracy (%) |
|---|---|---|
| GoogLeNet fixed | 420,000 | 63.0 |
| GoogLeNet triangular2 | 420,000 | **64.4** |
| GoogLeNet exp | 240,000 | 58.2 |
| GoogLeNet exp_range | 240,000 | 60.2 |

**Hallazgo en ImageNet**: CLR mejora accuracy de AlexNet (58.0 → 58.4) y GoogLeNet (63.0 → 64.4) con el **mismo numero de iteraciones**, demostrando que el beneficio no es solo en velocidad sino tambien en calidad final.

---

## 7. Implementacion en Codigo

El paper incluye la implementacion en Torch 7, que es sorprendentemente simple:

```text
-- Implementacion de la politica triangular en pseudocodigo:

funcion calcular_lr(iteracion, base_lr, max_lr, stepsize):
    cycle = floor(1 + iteracion / (2 * stepsize))
    x = abs(iteracion / stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * max(0, 1 - x)
    retornar lr

-- Para triangular2, agregar:
    lr = base_lr + (max_lr - base_lr) * max(0, 1 - x) / (2^(cycle - 1))

-- Para exp_range, agregar:
    lr = base_lr + (max_lr - base_lr) * max(0, 1 - x) * (gamma^iteracion)
```

**Parametros de entrada** (solo 3 para la politica triangular):
1. `base_lr` — learning rate minimo (limite inferior)
2. `max_lr` — learning rate maximo (limite superior)
3. `stepsize` — iteraciones en medio ciclo

---

## 8. Por que Solo Decrementar el LR No es Suficiente

El paper incluye un experimento crucial: la politica `decay`, que es como triangular pero **solo decrece** (empieza en max_lr y baja linealmente a base_lr, luego se queda ahi):

```text
  DECAY (solo decrece):         TRIANGULAR (sube y baja):
  LR                            LR
  max_lr ─\                     max_lr ──/\────/\────/\
           \                            /  \  /  \  /  \
            \                          /    \/    \/    \
             \___________             /                  \
  base_lr ───────────────  base_lr ──/────────────────────\

  Accuracy: 78.5%              Accuracy: 81.4%
```

Este resultado demuestra que el beneficio de CLR no viene solo de reducir el LR, sino del **comportamiento ciclico** — la combinacion de subir y bajar es lo que permite escapar de saddle points y explorar mejor el landscape de la loss.

---

## 9. Guia Practica para Usar CLR

```text
RECETA RAPIDA PARA APLICAR CLR:

1. ELEGIR ARQUITECTURA y dataset

2. EJECUTAR LR RANGE TEST:
   - Entrenar 1-8 epochs con LR creciendo linealmente
   - De LR_min_muy_bajo (e.g., 1e-5) a LR_max_alto (e.g., 1)
   - Graficar accuracy vs LR
   - base_lr = donde accuracy empieza a subir
   - max_lr  = donde accuracy empieza a caer
   (Alternativa: max_lr = mayor LR que converge, base_lr = max_lr/3)

3. ELEGIR STEPSIZE:
   - stepsize = 2 * iteraciones_por_epoch  (opcion simple)
   - stepsize = 8 * iteraciones_por_epoch  (ligeramente mejor)

4. ELEGIR POLITICA:
   - triangular:  buena opcion por defecto
   - triangular2: recomendada si se entrena muchos ciclos
   - exp_range:   util si se quiere decaimiento mas suave

5. ENTRENAR:
   - Intentar detener al final de un ciclo (LR en base_lr)
   - Comparar con LR fijo como sanity check
   - Al menos 3 ciclos completos para mejores resultados
```

---

## 10. Impacto Historico y Legado

```text
2015: Smith publica la primera version en arXiv
      → Introduce CLR y el LR range test

2017: Version revisada publicada en WACV
      → Agrega experimentos con ResNets, DenseNets, ImageNet

2017: Loshchilov & Hutter publican SGDR (Warm Restarts)
      → Concepto similar: reiniciar el LR ciclicamente
      → Usa cosine annealing en vez de triangular
      → Directamente inspirado por el trabajo de Smith

2018: Smith publica "Super-Convergence"
      → Extiende CLR con la "1cycle policy"
      → Un solo ciclo grande con LR que sube hasta un maximo alto
        y luego baja a un valor muy bajo
      → Logra entrenar redes 10x mas rapido

2018: fast.ai (Jeremy Howard) populariza CLR y 1cycle
      → Lo incorpora como metodo por defecto en la libreria fastai
      → fit_one_cycle() se vuelve el metodo estandar de entrenamiento

2019+: CLR se integra en frameworks principales:
  - PyTorch: torch.optim.lr_scheduler.CyclicLR
  - PyTorch: torch.optim.lr_scheduler.OneCycleLR
  - TensorFlow: tf.keras.optimizers.schedules
  - Keras: callbacks de LR scheduling

Hoy (2026):
  - El LR range test es una herramienta ESTANDAR para cualquier
    nuevo proyecto de Deep Learning
  - La 1cycle policy (descendiente directo de CLR) es el schedule
    mas usado en la practica
  - Los conceptos de warm-up + cosine decay usados en Transformers
    modernos (GPT, BERT, LLaMA) tienen raices en las ideas de CLR
  - PyTorch incluye CyclicLR y OneCycleLR como schedulers nativos
```

---

## 11. Conexion con Otros Metodos

```text
FAMILIA DE METODOS INSPIRADOS POR CLR:

  CLR (Smith, 2015/2017)
    |
    +──→ SGDR / Cosine Annealing with Warm Restarts
    |    (Loshchilov & Hutter, 2017)
    |    → Usa coseno en vez de triangular
    |    → lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T))
    |
    +──→ Super-Convergence / 1Cycle Policy
    |    (Smith & Topin, 2018)
    |    → Un solo ciclo con LR muy alto
    |    → Entrenamiento hasta 10x mas rapido
    |
    +──→ Warm-up + Decay (usado en Transformers)
         → Linear warm-up = primera mitad de un ciclo triangular
         → Cosine decay = similar a la segunda mitad
         → GPT, BERT, etc. usan este patron
```

---

## 12. Resumen en Una Pagina

```text
PROBLEMA:  Encontrar el learning rate optimo requiere muchos
           experimentos costosos; los schedules clasicos que solo
           decrecen son suboptimos.

SOLUCION:  Variar el learning rate CICLICAMENTE entre un minimo
           y un maximo durante el entrenamiento.

METODO PARA ELEGIR LOS LIMITES:
  LR Range Test: entrenar pocas epochs con LR creciente y observar
  donde la accuracy sube (base_lr) y donde cae (max_lr).

TRES POLITICAS PROPUESTAS:
  - triangular:  amplitud constante (la mas simple)
  - triangular2: amplitud se reduce a la mitad cada ciclo
  - exp_range:   amplitud decae exponencialmente con gamma^iteracion

PARAMETROS (solo 3):
  - base_lr:  learning rate minimo
  - max_lr:   learning rate maximo
  - stepsize: iteraciones en medio ciclo (recomendado: 2-8 epochs)

POR QUE FUNCIONA:
  1. El LR alto periodico ayuda a escapar saddle points
  2. El LR optimo siempre esta dentro del rango [base_lr, max_lr]
  3. Efecto de regularizacion: el ruido del LR variable
     previene overfitting (similar a otros metodos estocasticos)

RESULTADOS CLAVE:
  - CIFAR-10: misma accuracy en 2.8x MENOS iteraciones (25K vs 70K)
  - CIFAR-10 exp_range: MEJOR accuracy que baseline (82.2% vs 81.4%)
  - ResNet + CLR: 93.6% vs 93.3% mejor LR fijo
  - DenseNet + CLR: 94.9% vs 94.5% mejor LR fijo
  - ImageNet/GoogLeNet: 64.4% vs 63.0% con LR fijo
  - Compatible con Adam, Nesterov, RMSProp, AdaGrad

COSTO COMPUTACIONAL ADICIONAL: practicamente CERO
  (solo una formula simple para calcular el LR en cada iteracion)

LEGADO:
  - LR range test → herramienta estandar en Deep Learning
  - Inspiro 1cycle policy, SGDR, warm restarts
  - PyTorch: torch.optim.lr_scheduler.CyclicLR
  - Fundamento de los warm-up schedules usados en Transformers
```
