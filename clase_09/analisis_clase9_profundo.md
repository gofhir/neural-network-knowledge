# Análisis Profundo — Clase 9: Deep Network Architectures & Interpretability
**Profesor:** Miguel Fadic | Escuela de Ingeniería UC

> Este documento va mucho más allá de los slides: explica el *por qué* detrás de cada decisión de diseño, el contexto histórico, las intuiciones matemáticas y las razones por las que una arquitectura gana a otra.

---

## Tabla de Contenidos

1. [El problema que motiva todo: escalar redes profundas](#1-el-problema-que-motiva-todo)
2. [Contexto histórico: de LeNet a la era moderna](#2-contexto-histórico)
3. [Campo Receptivo: el concepto que lo une todo](#3-campo-receptivo-profundo)
4. [VGG: la filosofía de la simplicidad extrema](#4-vgg-en-profundidad)
5. [Inception: la arquitectura que "todo al mismo tiempo"](#5-inception-en-profundidad)
6. [ResNet: redefinir qué aprende una capa](#6-resnet-en-profundidad)
7. [¿Por qué una arquitectura u otra? Guía de decisión](#7-por-qué-una-arquitectura-u-otra)
8. [Interpretabilidad: el problema filosófico](#8-interpretabilidad-el-problema-filosófico)
9. [Feature Visualization: la red como generador](#9-feature-visualization-en-profundidad)
10. [Attribution: causalidad en redes neuronales](#10-attribution-en-profundidad)
11. [Extremal Perturbation: la atribución como optimización](#11-extremal-perturbation-en-profundidad)
12. [Síntesis: lecciones que trascienden las arquitecturas](#12-síntesis)

---

## 1. El Problema que Motiva Todo

### ¿Cuál es la pregunta central del diseño de arquitecturas?

Toda arquitectura de red neuronal profunda intenta responder la misma pregunta:

> **¿Cómo construir una función lo suficientemente expresiva para representar conceptos complejos del mundo real, sin que sea imposible de entrenar ni de usar en producción?**

Esta pregunta tiene tres tensiones fundamentales:

| Tensión | Lado A | Lado B |
|---------|--------|--------|
| Capacidad vs. eficiencia | Más parámetros = más expresividad | Más parámetros = más cómputo y memoria |
| Profundidad vs. entrenabilidad | Más capas = más abstracción | Más capas = gradientes que se desvanecen |
| Generalización vs. memorización | Aprender el dataset = buena accuracy en train | No generalizar = mala accuracy en test |

Cada arquitectura que vemos en esta clase es una solución distinta a estas tres tensiones.

### El benchmark: ImageNet

Toda la historia de las arquitecturas de esta clase gira alrededor de un dataset y competencia: **ILSVRC (ImageNet Large Scale Visual Recognition Challenge)**.

- **1.2 millones** de imágenes de entrenamiento
- **1000 clases** de objetos
- Métrica principal: **Top-5 error** (la clase correcta está entre las 5 mejores predicciones)
- Se realiza anualmente desde 2010

| Año | Ganador | Top-5 Error | Innovación clave |
|-----|---------|-------------|-----------------|
| 2010 | Hand-crafted features | ~28% | — |
| 2012 | **AlexNet** | **16.4%** | ReLU, Dropout, GPU |
| 2013 | ZFNet | 11.7% | AlexNet refinado |
| 2014 | **GoogLeNet** | **6.7%** | Inception modules |
| 2014 | **VGG** | **7.3%** | Profundidad + 3×3 |
| 2015 | **ResNet** | **3.57%** | Skip connections |
| (humano) | — | ~5% | — |

> En 2015, ResNet superó el rendimiento humano en Top-5 error. Esta es la línea de tiempo que da contexto a por qué cada arquitectura existe.

---

## 2. Contexto Histórico

### El problema pre-2012: las redes superficiales

Antes de 2012, las redes neuronales convolucionales existían desde los años 90 (LeNet-5 de Yann LeCun, 1998), pero no escalaban. Las razones:

1. **Cómputo insuficiente:** Las GPUs modernas no existían para este uso.
2. **Vanishing gradient:** Con más de 5-6 capas, los gradientes se hacían insignificantes y los pesos de las primeras capas no aprendían.
3. **Datos insuficientes:** ImageNet no existía; los datasets eran de decenas de miles de imágenes.
4. **Activaciones saturadas:** Las funciones sigmoid y tanh saturan, produciendo gradientes cercanos a cero.

### AlexNet (2012): la ruptura

AlexNet no fue simplemente "una red más grande". Fue un conjunto de decisiones que resolvieron los problemas anteriores simultáneamente:

| Decisión | Problema resuelto |
|----------|-----------------|
| **ReLU** en lugar de sigmoid/tanh | Gradientes no se saturan → vanishing gradient reducido |
| **Dropout** (0.5 en capas FC) | Regularización fuerte → menos overfitting |
| **Data augmentation** | Imágenes artificialmente variadas → más generalización |
| **Dos GPUs** en paralelo | Cómputo suficiente para entrenar en tiempo razonable |
| **Normalización local (LRN)** | Inspirada en neurofisiología → compitencia entre neuronas |

AlexNet tenía ~60M de parámetros y 5 capas convolucionales. Ganó por 10 puntos porcentuales sobre el segundo puesto, lo que fue una revolución en el campo.

### El insight de AlexNet que abrió la puerta

AlexNet demostró que **la profundidad importa**. Esto disparó una pregunta: ¿cuánto más profundo podemos ir y cómo?

---

## 3. Campo Receptivo: Profundo

### Definición precisa

El **campo receptivo** de una neurona en la capa `n` es el conjunto de píxeles de la imagen de entrada que influyen en esa neurona, a través de todas las capas intermedias.

**No confundir:**
- El **filtro** es 3×3 (en el espacio de la capa anterior)
- El **campo receptivo** es el tamaño de la región en la **imagen original** que corresponde a ese filtro

### Cálculo del campo receptivo

Para una secuencia de convoluciones con filtro `k` y stride `s=1`:

```
RF_n = RF_{n-1} + (k-1)
```

Para filtros 3×3 (k=3) con stride 1:

| Capa | Campo Receptivo (en el input) |
|------|------------------------------|
| 1 | 3×3 |
| 2 | 5×5 |
| 3 | 7×7 |
| 4 | 9×9 |
| n | (2n+1)×(2n+1) |

### ¿Por qué esto es fundamental para el diseño de arquitecturas?

El campo receptivo determina **cuánto contexto visual puede usar una neurona** para tomar su decisión. 

- Una neurona con campo receptivo 3×3 solo puede ver bordes.
- Una neurona con campo receptivo 224×224 puede ver la imagen completa.

**La pregunta de diseño es:** ¿Cómo alcanzar un campo receptivo grande de forma eficiente?

### Tres estrategias para aumentar el campo receptivo

#### Estrategia 1: Filtros grandes (AlexNet)
AlexNet usa un filtro 11×11 en la primera capa con stride 4.

```
Pros: Campo receptivo grande de golpe
Contras: 11×11 = 121 parámetros por filtro vs 9 para 3×3
         Muy costoso computacionalmente
```

#### Estrategia 2: Apilar filtros pequeños (VGG)
2 capas de 3×3 = campo receptivo 5×5 con **menos parámetros**

```
Pros: Eficiente en parámetros, añade no-linealidad extra (ReLU entre capas)
Contras: Más capas = más profundidad = más riesgo de vanishing gradient
```

#### Estrategia 3: Filtros en paralelo de distintos tamaños (Inception)
```
Pros: La red aprende qué escala es relevante para cada capa
Contras: Más complejo, requiere reducción de canales para no explotar en parámetros
```

#### Estrategia 4: Skip connections + profundidad (ResNet)
```
Pros: Profundidad arbitraria sin degradación de gradiente
Contras: Más complejo de implementar, mayor consumo de memoria durante forward
```

> **Insight central:** Todas las innovaciones en arquitecturas CNN son, en el fondo, estrategias para **aumentar el campo receptivo de forma eficiente** mientras se mantiene el entrenamiento estable.

---

## 4. VGG en Profundidad

### El paper y su contexto

**"Very Deep Convolutional Networks for Large-Scale Image Recognition"** (Simonyan & Zisserman, Oxford, 2014).

El equipo de Oxford (Visual Geometry Group, de ahí VGG) se propuso responder sistemáticamente: **¿qué pasa si aumentamos la profundidad manteniendo todo lo demás constante?**

Su metodología fue rigurosa: mantuvieron fijo el tamaño de filtro (3×3) y variaron solo la profundidad, desde 11 hasta 19 capas.

### La decisión de usar solo filtros 3×3: análisis completo

Esta es **la idea más importante de VGG** y merece análisis detallado.

#### ¿Por qué 3×3 y no 5×5 o 7×7?

**Argumento 1: Captura todas las relaciones espaciales**

Un filtro 3×3 cubre las 8 posiciones vecinas de cada píxel más el píxel central:
```
↖ ↑ ↗
← · →
↙ ↓ ↘
```
Es el filtro más pequeño que puede capturar: arriba/abajo, izquierda/derecha, diagonal, y centro. Ningún filtro de menor tamaño captura esta riqueza espacial.

**Argumento 2: Eficiencia de parámetros** (el más importante)

Analicemos el costo de distintas estrategias para un campo receptivo 5×5:

```
Opción A: 1 capa con filtro 5×5
Parámetros: 5 × 5 × C_in × C_out = 25 × C_in × C_out

Opción B: 2 capas con filtro 3×3
Parámetros capa 1: 3 × 3 × C_in × C_mid  = 9 × C_in × C_mid
Parámetros capa 2: 3 × 3 × C_mid × C_out = 9 × C_mid × C_out
Total (si C_in = C_mid = C_out = C): 18C²  vs  25C²
```

**Reducción: 28% menos parámetros** con el mismo campo receptivo.

Para campo receptivo 7×7:
```
Opción A: 1 capa 7×7 → 49 × C_in × C_out
Opción B: 3 capas 3×3 → 27 × C² (reducción del 44%)
```

**Argumento 3: Más no-linealidades**

Con 2 capas 3×3 en lugar de 1 capa 5×5, hay **una ReLU adicional** entre ellas. Más no-linealidades = función más expresiva con la misma capacidad receptiva.

```
Opción A (1 capa 5×5):
  Input → Conv5×5 → ReLU → Output
  No-linealidades: 1

Opción B (2 capas 3×3):
  Input → Conv3×3 → ReLU → Conv3×3 → ReLU → Output
  No-linealidades: 2
```

### La arquitectura de VGG: decisiones de diseño

#### Bloques con MaxPooling

VGG organiza las capas en **5 bloques**, cada uno terminando en MaxPool 2×2 stride 2. Esto reduce el mapa de activaciones a la mitad en cada bloque.

```
Entrada: 224×224
Bloque 1 → MaxPool → 112×112 (reducción ×2)
Bloque 2 → MaxPool →  56×56  (reducción ×4)
Bloque 3 → MaxPool →  28×28  (reducción ×8)
Bloque 4 → MaxPool →  14×14  (reducción ×16)
Bloque 5 → MaxPool →   7×7   (reducción ×32)
```

Mientras el mapa espacial se achica, los **canales aumentan**:
```
64 → 128 → 256 → 512 → 512
```

Este patrón "embudo espacial / expansión de canales" es un estándar en todas las CNN modernas.

#### ¿Por qué doblar los canales en cada bloque?

Al hacer MaxPool, se pierde la mitad de la información espacial (cada 2×2 píxeles → 1 píxel). Para no perder capacidad representacional, se dobla el número de canales. Esto mantiene el **volumen total de información** aproximadamente constante entre bloques.

```
Bloque 1: 112×112×64  = 802,816 valores
Bloque 2:  56×56×128  = 401,408 valores (mitad)  ← perdemos info espacial
Bloque 3:  28×28×256  = 200,704 valores
...
```

#### Las capas Fully Connected: el cuello de botella de VGG

Las últimas 3 capas FC de VGG son el mayor problema de la arquitectura:

```
7×7×512 → FC-4096 → FC-4096 → FC-1000
```

El paso `7×7×512 = 25088` → `4096` requiere una matriz de **25,088 × 4,096 = 102M parámetros** solo en esa capa.

De los ~138M de parámetros totales de VGG-16:
- Capas convolucionales: ~14.7M (solo 10.6% del total)
- Capas FC: ~123.6M (89.4% del total)

> **Esto es la gran debilidad de VGG:** la mayor parte de sus parámetros están en las capas FC, que son las menos eficientes. Inception y ResNet resuelven esto con Average Pooling global.

### ¿Por qué VGG sigue siendo relevante?

A pesar de sus ineficiencias, VGG se usa ampliamente porque:

1. **Simplicidad:** Es fácil de entender, modificar y depurar.
2. **Features de alta calidad:** Sus representaciones intermedias son excelentes para Transfer Learning en tareas de visión.
3. **Perceptual Loss:** En tareas de style transfer y super-resolución, las activaciones de VGG capturan bien la "percepción" visual. (Ej: el paper de Johnson et al. sobre neural style transfer usa VGG).
4. **Transferibilidad demostrada:** Décadas de uso práctico han validado su utilidad.

### Limitaciones de VGG

| Limitación | Impacto |
|-----------|---------|
| 138M parámetros | No cabe en dispositivos móviles |
| ~30 G-Ops para inferencia | Lento en producción |
| Sin skip connections | Difícil de entrenar más profundo |
| FC layers enormes | 89% de parámetros en capas poco eficientes |

---

## 5. Inception en Profundidad

### La motivación filosófica

El equipo de Google (Szegedy et al.) partió de una observación diferente a VGG: en lugar de preguntar "¿cuántas capas?", preguntaron **"¿qué hace falta en cada capa?"**

La respuesta fue: **no sabemos de antemano qué escala espacial es relevante en cada capa**. A veces un objeto importante es pequeño (filtro 1×1), a veces mediano (filtro 3×3), a veces grande (filtro 5×5). ¿Por qué forzar una sola escala?

### El módulo Inception: diseño "todo a la vez"

La idea central es operar en **paralelo** con múltiples escalas y **dejar que la red aprenda** cuál es más útil para cada tarea.

#### Versión naïve del módulo Inception

```
                    Capa anterior
                    (batch, H, W, C)
          ┌─────────────┬─────────────┬──────────────┐
          │             │             │              │
        1×1 conv      3×3 conv      5×5 conv    3×3 MaxPool
       (depth=64)    (depth=128)   (depth=32)    (padding=same)
          │             │             │              │
          └─────────────┴─────────────┴──────────────┘
                         Concatenación (en canales)
                    (batch, H, W, 64+128+32+C)
```

**Problema:** Si la capa anterior tiene 256 canales, el costo de la convolución 5×5 es:
```
H × W × 256 × 5 × 5 × 32 = enorme
```

Además, los canales se acumulan en cada módulo: cada módulo Inception aumenta la profundidad en `(64+128+32+pool_proj)`. En una red con 9 módulos, esto explota.

#### La solución: convoluciones 1×1 como "cuellos de botella"

La innovación clave de Inception es usar **convoluciones 1×1 antes de las 3×3 y 5×5** para reducir la profundidad (número de canales).

```
Capa anterior (256 canales)
       │
   1×1 conv (16 canales)   ← BOTTLENECK: reduce de 256 a 16
       │
   5×5 conv (32 canales)   ← ahora mucho más barato
```

### La convolución 1×1: explicación completa

Este es uno de los conceptos más importantes y menos intuitivos de las CNN modernas.

#### ¿Qué hace matemáticamente una convolución 1×1?

Una convolución 1×1 toma un tensor `(H, W, C_in)` y produce `(H, W, C_out)`. Para cada posición `(h, w)`:

```
output[h, w, j] = Σᵢ W[j, i] × input[h, w, i] + b[j]
```

Donde `W` es de tamaño `(C_out, C_in)`.

**Esto es exactamente una capa fully connected aplicada independientemente en cada posición espacial.**

#### La analogía con embeddings

Imagina que cada posición `(h, w)` tiene un "vector de características" de dimensión `C_in`. Una convolución 1×1 aplica una **proyección lineal** a ese vector, comprimiéndolo o expandiéndolo a `C_out` dimensiones.

```
C_in = 256 canales
1×1 conv con C_out = 16

Cada posición (h,w):
  vector de 256 dims → proyección lineal → vector de 16 dims
```

Esto es **exactamente** lo que hace una capa de embedding en NLP: proyectar un espacio de alta dimensión a uno más compacto.

#### ¿Qué aprende la red en estas convoluciones?

La red aprende qué **combinaciones lineales** de los canales anteriores son relevantes para el siguiente cómputo. Es una forma de "mezclar" información de distintos canales sin cambiar las dimensiones espaciales.

### El cálculo de ahorro de parámetros: paso a paso

Primer módulo Inception de GoogLeNet (capa `inception_3a`):
- Input: `28×28×192`
- Rama 5×5: 32 filtros de 5×5

**Sin 1×1 bottleneck:**
```
Parámetros = C_in × 5 × 5 × C_out
           = 192 × 5 × 5 × 32
           = 153,600

Operaciones por posición espacial = 192 × 25 = 4,800
Total operaciones = 28 × 28 × 4,800 = 3,763,200
```

**Con 1×1 bottleneck (16 canales intermedios):**
```
Capa 1×1: 192 × 1 × 1 × 16 = 3,072 parámetros
Capa 5×5:  16 × 5 × 5 × 32 = 12,800 parámetros
Total = 15,872 parámetros  (↓ 89.7%)

Operaciones 1×1 por posición = 192
Operaciones 5×5 por posición = 16 × 25 = 400
Total operaciones = 28×28 × (192 + 400) = 464,464  (↓ 87.7%)
```

### La estructura completa de GoogLeNet

GoogLeNet tiene 22 capas con parámetros y ~6.8M de parámetros totales (comparado con 138M de VGG-16).

```
7×7 conv, stride 2 → 3×3 MaxPool, stride 2
3×3 conv, LRN      → 3×3 MaxPool, stride 2

inception_3a (256 filtros totales)
inception_3b (480 filtros totales)
3×3 MaxPool, stride 2

inception_4a (512) ← Auxiliary classifier 1
inception_4b (512)
inception_4c (512)
inception_4d (528)
inception_4e (832) ← Auxiliary classifier 2
3×3 MaxPool, stride 2

inception_5a (832)
inception_5b (1024)

7×7 Average Pool (global)
Dropout 40%
Linear 1000
Softmax
```

### Average Pooling Global vs. Capas FC: explicación profunda

GoogLeNet fue pionero en reemplazar las capas FC por **Global Average Pooling (GAP)**.

**¿Qué hace el GAP?**

Después de la última capa convolucional, el tensor es `(batch, 7, 7, 1024)`. GAP colapsa las dimensiones espaciales tomando el promedio:

```
GAP: (batch, 7, 7, 1024) → (batch, 1024)
```

Cada uno de los 1024 valores es el **promedio de activación** del canal correspondiente sobre toda la imagen.

**¿Por qué es mejor que FC?**

1. **Parámetros:** GAP tiene 0 parámetros. Una FC equivalente tendría `7×7×1024 × 1000 = 50M` parámetros.
2. **Invarianza espacial:** El GAP promedia sobre toda la imagen, por lo que es invariante a traslaciones del objeto. Esto mejora la generalización.
3. **Interpretabilidad:** Cada canal tiene una interpretación semántica clara, lo que facilita las Class Activation Maps (CAMs).
4. **Regularización implícita:** Al no tener los parámetros de las FC, hay mucho menos overfitting.

### Los clasificadores auxiliares: estrategia anti-vanishing-gradient

En una red de 22 capas, el gradiente de la loss en la capa final debe propagarse hasta la capa 1. Con cada capa, el gradiente se multiplica por la derivada de la activación. Si estas derivadas son < 1 (lo que ocurre frecuentemente), el gradiente se hace exponencialmente pequeño.

**La solución de Inception:** Agregar clasificadores intermedios que calculan su propia loss:

```
Loss_total = Loss_principal + 0.3 × Loss_aux1 + 0.3 × Loss_aux2
```

Los clasificadores auxiliares "inyectan" gradiente directamente en capas intermedias, evitando que el gradiente tenga que recorrer toda la red.

> **Nota:** Esta técnica fue necesaria en 2014. Las redes modernas con Batch Normalization y skip connections hacen innecesarios los clasificadores auxiliares. En Inception-v4 ya no se usan.

---

## 6. ResNet en Profundidad

### El problema de la degradación: explicación completa

En 2015, Kaiming He et al. observaron algo perturbador en experimentos en CIFAR-10:

```
Red plain de 20 capas:  error train ≈ 7%, error test ≈ 8%
Red plain de 56 capas:  error train ≈ 9%, error test ≈ 11%
```

La red más profunda tiene **peor error de entrenamiento**. Esto no es overfitting (el error de test también es peor). La red de 56 capas tiene 36 capas extra que simplemente empeoran el resultado.

**¿Por qué pasa esto?**

Añadir una capa es añadir una transformación. Para que sea neutral, esa capa debería aprender la función identidad `f(x) = x`. Pero en la práctica, esto es **difícil de aprender**.

Una red con 56 capas, en principio, podría comportarse exactamente como una de 20 capas si las 36 capas extra aprendieran la identidad. Pero no lo hace, porque:

1. El espacio de búsqueda para encontrar la identidad es enorme.
2. El gradiente en capas tempranas es muy pequeño, por lo que esas capas convergen muy lento.
3. Con múltiples no-linealidades, es difícil componer la función identidad perfecta.

### La hipótesis residual: replanteamiento del problema

La genialidad de He et al. fue **reformular** qué aprende cada bloque.

**Formulación clásica:** Un bloque aprende `H(x)` directamente.

**Formulación residual:** En lugar de aprender `H(x)`, un bloque aprende:
```
F(x) = H(x) - x    →    H(x) = F(x) + x
```

La shortcut connection suma `x` directamente a la salida de las capas intermedias.

**¿Por qué esto es más fácil de optimizar?**

Si la solución óptima es la identidad (`H(x) = x`), entonces `F(x) = 0`. Llevar pesos a cero es trivial en gradient descent; las capas simplemente aprenden a no hacer nada.

```
Caso: la identidad es óptima

Formulación clásica: aprender H(x) = x
  → Los pesos deben configurarse para replicar la entrada exactamente
  → Muy difícil con múltiples capas y no-linealidades

Formulación residual: aprender F(x) = 0
  → Los pesos solo necesitan aproximarse a cero
  → Trivial: cualquier inicialización cerca de cero funciona
```

### El bloque residual: análisis detallado

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x                        # ← guardar entrada
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual                # ← suma residual
        out = self.relu(out)
        return out
```

#### La shortcut connection cuando cambian las dimensiones

Cuando el bloque cambia el número de canales (ej: de 64 a 128), la shortcut no puede sumarse directamente (dimensiones distintas). Se usa una **proyección 1×1**:

```python
if C_in != C_out:
    shortcut = nn.Conv2d(C_in, C_out, 1, stride=2)
else:
    shortcut = nn.Identity()

output = F(x) + shortcut(x)
```

### El bloque bottleneck: estrategia de eficiencia

Para ResNet-50, 101, 152, usar bloques de 2 capas 3×3 sería demasiado costoso. Se usa el **bottleneck block** con 3 capas:

```
256 canales de entrada
    ↓
  1×1, 64 canales   ← comprimir: reduce 256→64 (ahorra cómputo)
    ↓
  3×3, 64 canales   ← convolución espacial (cómputo reducido)
    ↓
  1×1, 256 canales  ← expandir: restaura 64→256 (preserva info)
    ↓
    + shortcut(256)
```

**Comparación de costos:**

| Bloque | Parámetros | FLOPs |
|--------|-----------|-------|
| 2× Conv3×3 (básico) | 2 × 3×3×64×64 = 73,728 | ~118M |
| Bottleneck 1×1-3×3-1×1 | 256×1×64 + 64×9×64 + 64×1×256 = 69,632 | ~17M |

El bottleneck es más eficiente con prácticamente la misma capacidad.

### Batch Normalization: por qué ResNet lo usa en todas las capas

ResNet aplica **Batch Normalization (BN)** después de cada convolución, antes de la activación.

#### ¿Qué hace Batch Normalization?

Para un mini-batch de activaciones, BN normaliza la distribución a media 0 y varianza 1, luego aplica parámetros aprendibles `γ` y `β`:

```
BN(x) = γ × (x - μ_batch) / σ_batch + β
```

#### ¿Por qué ayuda a entrenar redes muy profundas?

**Problema (Internal Covariate Shift):** Durante el entrenamiento, la distribución de activaciones en cada capa cambia constantemente porque los pesos anteriores cambian. Esto hace que cada capa deba "readaptarse" constantemente.

**Solución con BN:**
1. Las activaciones siempre tienen distribución aproximadamente normal → las capas reciben siempre inputs similares.
2. Los gradientes fluyen mejor: BN evita que las activaciones se saturen (no hay píxeles en la zona plana de sigmoid).
3. Permite **learning rates más altos** → entrenamiento más rápido.
4. **Regularización implícita:** Las estadísticas del batch son ruidosas, lo que actúa como regularizador.

#### ¿Por qué ResNet no usa Dropout?

VGG usa Dropout(0.5) en las capas FC. ResNet no usa Dropout. La razón es que:

1. BN ya provee regularización suficiente.
2. Las skip connections hacen al modelo más robusto por sí solo.
3. Dropout interactúa mal con BN en redes profundas (hay investigación reciente sobre esto).

### ResNet: resultados y por qué funcionó

Con conexiones residuales, **más capas = mejor rendimiento**:

| Versión | Capas | Top-1 ImageNet | Parámetros |
|---------|-------|---------------|-----------|
| ResNet-18 | 18 | 69.8% | 11.7M |
| ResNet-34 | 34 | 73.3% | 21.8M |
| ResNet-50 | 50 | 76.1% | 25.6M |
| ResNet-101 | 101 | 77.4% | 44.6M |
| ResNet-152 | 152 | 78.3% | 60.2M |

El error **sigue bajando** con más capas. Esto demostró que el problema de degradación era de optimización, no de capacidad.

---

## 7. ¿Por qué Una Arquitectura u Otra? Guía de Decisión

### La pregunta práctica

En la práctica, cuando un ingeniero o investigador enfrenta una tarea de visión, debe elegir arquitectura. Esta sección da los criterios.

### Criterio 1: Recursos computacionales disponibles

| Contexto | Arquitectura recomendada | Razón |
|----------|-------------------------|-------|
| Servidor con GPU potente | ResNet-50/101, EfficientNet | Balance óptimo eficiencia/accuracy |
| Servidor con múltiples GPUs | ResNet-152, Inception-v4 | Máxima accuracy, no importa el costo |
| Tiempo real en servidor | ResNet-18, GoogLeNet | Menos de 10 G-Ops |
| Móvil / Edge computing | MobileNet, EfficientNet-B0 | Diseñadas específicamente para esto |
| No hay GPU | AlexNet, ResNet-18 | Los más rápidos en CPU |

### Criterio 2: Tamaño del dataset

| Dataset | Estrategia | Arquitectura |
|---------|-----------|-------------|
| < 1,000 imágenes por clase | Transfer learning, solo entrenar capas finales | Cualquiera pre-entrenada en ImageNet |
| 1,000 - 10,000 por clase | Fine-tuning de capas superiores | ResNet-50, VGG-16 |
| > 100,000 por clase | Entrenamiento desde cero viable | Elegir según criterio computacional |
| Dataset médico / específico | Fine-tuning cuidadoso | ResNet, EfficientNet |

#### ¿Por qué el tamaño importa tanto?

Las capas tempranas de una CNN aprenden features universales (bordes, texturas). Las capas profundas aprenden features específicas del dominio.

Con datasets pequeños:
- Entrenar desde cero → las capas tempranas nunca aprenden bien (no hay suficientes ejemplos)
- Fine-tuning → las capas tempranas ya tienen buenas features de ImageNet; solo hay que adaptar las capas finales al nuevo dominio

### Criterio 3: Naturaleza de la tarea

| Tarea | Mejor opción | Por qué |
|-------|-------------|---------|
| Clasificación simple | ResNet-50 | Mejor balance general |
| Detección de objetos | ResNet (backbone) + FPN | Skip connections facilitan multi-escala |
| Segmentación | U-Net (VGG/ResNet encoder) | Necesita features de distintas resoluciones |
| Transfer learning general | VGG-16 o ResNet-50 | Features bien estudiadas y transferibles |
| Generación (GAN, VAE) | VGG (perceptual loss) | Sus features capturan "percepción" visual |

### Criterio 4: Interpretabilidad requerida

Si la tarea requiere **explicar las predicciones** (medicina, auditoría, legal):

- **ResNet** tiene skip connections que facilitan la propagación de señales de atribución.
- **Inception** es más difícil de interpretar por su estructura paralela.
- **VGG** es el más simple de visualizar (todo secuencial).

### El gráfico de Canziani et al.: leído correctamente

El gráfico que mostró el profesor tiene 3 ejes simultáneos:
- X: Operaciones (G-Ops) → tiempo de inferencia
- Y: Top-1 accuracy
- Tamaño burbuja: Número de parámetros → memoria necesaria

**Lectura óptima:** Buscar burbujas pequeñas, arriba a la izquierda.

**Frontera de Pareto de arquitecturas:**
1. `SqueezeNet`: Mínimos parámetros, pero baja accuracy
2. `MobileNet-v2`: Pocas ops, buena accuracy → **mejor para móvil**
3. `ResNet-50`: Balance óptimo general → **mejor para servidor**
4. `Inception-v4`: Máxima accuracy → **mejor cuando accuracy es crítica**
5. `VGG-16/19`: **Nunca es la opción óptima** en el gráfico de Pareto (grande y lento para su accuracy)

---

## 8. Interpretabilidad: el Problema Filosófico

### ¿Por qué es difícil interpretar una CNN?

Una CNN con 50 capas tiene millones de operaciones no-lineales encadenadas. La relación entre la entrada y la salida es:

```
y = f_50(f_49(...f_2(f_1(x))...))
```

Donde cada `f_i` es una transformación no-lineal. Entender por qué `y = "gato"` dado `x = [imagen]` es equivalente a entender la composición de 50 funciones no-lineales. No hay una fórmula cerrada.

### El espectro de la interpretabilidad

```
Más simple de entender ←──────────────────────→ Más complejo

Regresión lineal → Árbol de decisión → Random Forest → SVM → Red neuronal profunda
      ↑                                                           ↑
  completamente                                          caja negra casi
  interpretable                                          completamente opaca
```

Las CNN están en el extremo derecho. Las técnicas que vemos en clase son intentos de iluminar esa caja negra.

### Dos preguntas distintas de interpretabilidad

**Pregunta 1: Intrínseca** — *¿Qué ha aprendido la red en general?*
- No depende de una imagen específica
- Respuesta: Feature Visualization
- Útil para: depurar el proceso de entrenamiento, entender capacidades del modelo

**Pregunta 2: Extrínseca** — *¿Por qué la red predijo X para esta imagen específica?*
- Depende de una imagen concreta
- Respuesta: Attribution
- Útil para: auditar predicciones individuales, detectar bias

### Las dos amenazas a la confianza en un modelo

#### Amenaza 1: Bias (sesgo)

El modelo aprende una correlación espuria que funciona en el dataset de entrenamiento pero no refleja la causalidad real.

**Ejemplos reales:**
- Un clasificador de neumonía basado en radiografías aprendió que las **radiografías portátiles** (tomadas a pacientes graves) se correlacionan con neumonía. Predicha en una radiografía portátil normal → alta probabilidad de neumonía.
- Un clasificador de razas de perros aprendió el **fondo** de las fotos (husky en nieve) más que el perro.
- El clasificador de caballos que aprendió el **watermark de copyright** de las imágenes de entrenamiento.

#### Amenaza 2: Shortcuts

El modelo encuentra un atajo para maximizar la accuracy de entrenamiento que no generaliza.

**Ejemplo:** Si en el dataset de entrenamiento todos los lobos están en nieve y todos los huskies están en casa, el modelo puede aprender "nieve = lobo" sin nunca aprender la apariencia del animal.

### ¿Por qué la interpretabilidad es ahora obligatoria en producción?

- **GDPR (Europa):** "Derecho a explicación" para decisiones automatizadas que afecten a personas.
- **FDA (medicina):** Los sistemas de diagnóstico por IA deben poder explicar sus predicciones.
- **Ética en IA:** Los sesgos en modelos de crédito, contratación y justicia penal han sido documentados extensamente.
- **Debugging:** Encontrar por qué un modelo falla es imposible sin entender qué aprendió.

---

## 9. Feature Visualization en Profundidad

### El fundamento matemático

Dado un modelo `Φ: ℝ^n → ℝ^m` (de imagen a activaciones), la Feature Visualization busca:

```
x* = argmax_{x ∈ ℝ^n}  obj(Φ(x)) - R(x)
```

Donde:
- `obj(Φ(x))` es el objetivo (activación a maximizar)
- `R(x)` es un término de regularización que penaliza imágenes "no naturales"
- La optimización es **gradient ascent** en el espacio de imágenes

### Objetivos de optimización: taxonomía completa

#### Neurona individual: `layer_n[x_0, y_0, z_0]`

Maximiza la activación de una neurona específica en posición espacial `(x_0, y_0)` y canal `z_0`.

```
Útil para: entender qué detecta una neurona muy específica
Problema: muy ruidoso, un solo ejemplo puede no ser representativo
```

#### Canal completo: `layer_n[:, :, z_0]`

Maximiza la activación promedio del canal `z_0` sobre todas las posiciones espaciales.

```
Útil para: entender qué detecta un "filtro" completo
Más robusto que neurona individual
```

#### Layer/DeepDream: `||layer_n[:, :, :]||^2`

Maximiza la norma L2 de todas las activaciones en una capa. Esto **amplifica** los patrones que ya estaban presentes en la imagen.

```
Útil para: visualización artística, entender qué clase de patrones hay en una capa
Es la base de DeepDream de Google
```

#### Class Logits: `pre_softmax[k]`

Maximiza el logit de la clase `k` antes de la capa softmax.

```
Útil para: ver la "imagen ideal" de una clase
Mejor que softmax porque no hay competencia entre clases
```

**¿Por qué logits > probabilidad softmax?**

Recordemos softmax:
```
P(k) = exp(z_k) / Σ_j exp(z_j)
```

Para maximizar `P(k)`, la red puede hacer dos cosas:
1. Aumentar `z_k` (la clase objetivo)
2. Disminuir todos los demás `z_j`

Esto lleva a imágenes que "suprimen" todas las otras clases, no a imágenes que representan bien la clase objetivo. Los logits evitan esta distorsión.

### El problema de las imágenes adversariales

**Sin regularización**, la optimización converge a **imágenes adversariales**: imágenes que el humano ve como ruido pero que activan el modelo con alta confianza.

```
Imagen adversarial: x* que maximiza P("gato") = 99.9%
Pero x* se ve como ruido blanco para un humano
```

Esto pasa porque:
- La función `Φ(x)` es altamente no-convexa
- Hay muchísimos inputs que activan fuertemente una neurona
- La mayoría de esos inputs son "ruido estructurado" que nuestros ojos no reconocen

### Las tres técnicas de regularización explicadas

#### 1. Penalización de frecuencia

Las imágenes naturales tienen la propiedad de que las **frecuencias altas** (variaciones rápidas entre píxeles vecinos) son raras y pequeñas en comparación con las frecuencias bajas.

```
Espectro de potencia de imágenes naturales: P(f) ∝ 1/f²
(las altas frecuencias tienen muy poca potencia)
```

Al penalizar frecuencias altas, forzamos a la imagen a parecerse a imágenes naturales.

**Técnicas:**
- **L1:** Penaliza la magnitud de los píxeles. `R(x) = λ × Σ |x_i|`
- **Total Variation:** Penaliza diferencias entre píxeles vecinos. `R(x) = λ × Σ |x_{i+1} - x_i| + |x_{i+1} - x_j|`
- **Blur:** En cada paso, aplica un filtro gaussiano a la imagen.

#### 2. Robustez a transformaciones (Transformation Robustness)

Antes de evaluar la activación en cada paso de gradient ascent, se aplica una **transformación aleatoria** a la imagen:

```python
# Pseudocódigo del proceso
for step in range(n_steps):
    x_transformed = random_transform(x, jitter=1, rotate=5°, scale=1.1)
    activation = model(x_transformed)[target]
    activation.backward()
    x = x + lr * x.grad
```

**¿Por qué funciona?**

Fuerza a la imagen a activar el modelo independientemente de pequeñas perturbaciones. El resultado es un patrón más "robusto" y visualmente coherente, porque el optimizador no puede explotar coordenadas exactas de píxeles.

#### 3. Espacio de imagen decorrelado

**El problema:** En el espacio RGB, los colores rojo, verde y azul están fuertemente correlacionados en imágenes naturales (un píxel brillante suele ser brillante en los 3 canales). Optimizar directamente en este espacio genera ruido cromático.

**La solución:** Transformar la imagen a un espacio donde los canales están decorrelados (similar a PCA o DCT/Fourier), optimizar allí, y transformar de vuelta.

```
Imagen RGB (correlada)
      ↓  transformación (Fourier/PCA)
Espacio decorrelado
      ↓  gradient ascent
Espacio decorrelado (optimizado)
      ↓  transformación inversa
Imagen RGB mejorada
```

El gradiente en el espacio decorrelado distribuye las actualizaciones de manera más uniforme y "natural".

### Diversidad en Feature Visualization: el concepto de polisemia neuronal

Un resultado perturbador de la investigación de Distill (Olah et al.): las neuronas no son monosémicas (un concepto = una neurona). Muchas neuronas son **polisémicas** (una neurona responde a múltiples conceptos no relacionados).

**Ejemplo real de la clase:** El canal 143 de `mixed4a` en GoogLeNet
- Optimización simple → "cabezas de perro"
- Optimización con diversidad → "cabezas de perro" + "orejas" + "snouts" + otros patrones

**¿Por qué existe la polisemia?**

Hipótesis de superposición (*superposition hypothesis*, Elhage et al., 2022):
- Si hay más conceptos que neuronas, la red es forzada a "comprimir" múltiples conceptos en una neurona
- Las redes modernas aprenden muchos más conceptos que neuronas tienen
- La polisemia es una estrategia óptima de compresión

**Implicación para la interpretabilidad:** No podemos estudiar neuronas individuales y concluir que "esta neurona detecta X". Las neuronas tienen significados múltiples y contextuales.

### La jerarquía de features: evidencia experimental

Lo que vemos en GoogLeNet (y es consistente con todas las CNN):

| Capas | Features | Interpretación |
|-------|----------|---------------|
| conv2d0 (capa 1) | Bordes orientados, colores primarios | Detectores de gradiente local |
| conv2d1-2 | Bordes más complejos, esquinas | Combinaciones de detectores básicos |
| mixed3a-3b | Texturas (puntos, líneas, grillas) | Patrones repetitivos locales |
| mixed4a-4b | Patrones complejos (flores, redes, escalas) | Texturas de mayor escala |
| mixed4c-4d | Partes de objetos (ojos, ruedas, patas) | Composición de texturas |
| mixed4e-5a | Objetos parciales (caras, flores) | Partes reconocibles |
| mixed5b | Objetos completos y escenas | Representaciones semánticas |

**¿Por qué esta jerarquía es esperable?**

Teorema de composición: cualquier función suave puede aproximarse componiendo funciones simples. Las CNN aprenden exactamente esta composición de forma emergente, sin que nadie programara "esta capa aprende bordes".

---

## 10. Attribution en Profundidad

### La pregunta correcta de la atribución

Attribution responde: **¿Cuánto contribuyó cada feature de la entrada a una predicción específica?**

Formalmente, dado un modelo `Φ: X → Y` y una entrada `x₀`, se busca un **mapa de relevancia** `R: X → ℝ` tal que:

```
R(i) ≈ "cuánto contribuyó el píxel i a la predicción Φ(x₀)"
```

### Métodos basados en backpropagation

#### Gradiente simple (Vanilla Gradient)

```
R(i) = |∂Φ(x₀)/∂x₀[i]|
```

El gradiente de la predicción respecto al input. Es el método más simple.

**Pros:** Fácil de calcular, eficiente  
**Contras:** Mide *sensibilidad*, no *importancia*. Un píxel puede tener gradiente alto aunque no sea importante para el objeto (ej: puede estar en el fondo).

**Intuición:** Piénsalo como "¿cuánto cambiaría la predicción si perturbara infinitesimalmente este píxel?". Esto no es lo mismo que "¿cuánto importa este píxel para la predicción actual?".

#### Guided Backpropagation

Modifica las reglas de backprop: solo propaga gradientes que son **positivos tanto en el forward como en el backward**:

```python
# En el backward de ReLU:
# Normal:         grad_input = grad_output * (output > 0)
# Guided:         grad_input = grad_output * (output > 0) * (grad_output > 0)
```

**Intuición:** "Solo me interesan las activaciones que incrementaron la predicción".  
**Resultado:** Mapas más nítidos que el gradiente simple.  
**Limitación:** No tiene garantías de que muestre features causalmente relevantes.

#### Grad-CAM (Gradient-weighted Class Activation Maps)

Combina el gradiente con las activaciones de la última capa convolucional:

```
weights_c = (1/Z) Σ_i Σ_j (∂y_c / ∂A^k_{ij})   ← promedio espacial del gradiente

Grad-CAM^c = ReLU(Σ_k weights_c^k × A^k)
```

Donde `A^k` es el mapa de activación del canal `k`.

**Intuición:** "Los canales que más cambian cuando la predicción de clase `c` cambia son los más importantes. El mapa de esos canales es el mapa de atribución".

**Pros:** Produce mapas de alta calidad semántica, sin modificar la red  
**Contras:** Resolución limitada por el tamaño de la última capa conv (ej: 7×7 para ResNet)

### Métodos basados en perturbación

#### Oclusión (Occlusion Sensitivity)

El método más intuitivo: deslizar un parche de oclusión (negro) sobre la imagen y medir cuánto cae la predicción.

```python
for y in range(H):
    for x in range(W):
        img_masked = img.copy()
        img_masked[y:y+patch, x:x+patch] = 0  # ocluir región
        sensitivity[y, x] = original_pred - model(img_masked)[class]
```

**Pros:** Muy intuitivo, mide causalidad real  
**Contras:** O(H×W) evaluaciones del modelo → extremadamente lento

#### RISE (Randomized Input Sampling for Explanation)

Aplica miles de máscaras aleatorias y promedia su efecto:

```
R(i) ≈ (1/N) Σ_n [m_n(i) × Φ(m_n ⊙ x₀)]
```

**Intuición:** Si la predicción es alta cuando el píxel `i` está visible (máscara = 1), entonces `i` es importante.

**Pros:** Estocásticamente estima la importancia de cada píxel  
**Contras:** Requiere muchas evaluaciones (ej: N=5000)

### Comparativa visual de métodos

Del slide de la clase (cabbage butterfly, kite, tile roof):

| Método | Granularidad | Ruido | Coste |
|--------|-------------|-------|-------|
| Gradient | Píxel fino | Alto | Bajo (1 backward) |
| Guided Backprop | Fino | Medio | Bajo |
| Grad-CAM | Grueso (semántico) | Bajo | Bajo |
| Occlusion | Medio | Bajo | Muy alto |
| Extremal Perturbation | Compacto, preciso | Muy bajo | Medio |
| RISE | Medio | Bajo | Alto |

---

## 11. Extremal Perturbation en Profundidad

### El problema con los métodos anteriores

Todos los métodos anteriores tienen una limitación fundamental: **no optimizan directamente para encontrar la región más informativa**. Son aproximaciones (gradiente), promediados estocásticos (RISE) o costosos barridos (occlusion).

**Extremal Perturbation** formula la atribución como un **problema de optimización** directamente:

> Encontrar la máscara `m` de tamaño fijo que, al revelar solo esa región de la imagen, **preserva máximamente la predicción original del modelo**.

### La formulación matemática completa

```
m* = argmax_{m}  Φ(m ⊗ x₀)[c]

sujeto a:
  ∫∫ m(u,v) du dv = a × H × W    (restricción de área)
  m(u,v) ∈ [0, 1]                 (máscara suave)
```

Donde:
- `m ⊗ x₀` es la perturbación: `m(u,v) × x₀(u,v) + (1-m(u,v)) × blur(x₀)(u,v)`
- La región no-revelada se reemplaza por la versión difuminada (no por negro, para evitar distribución fuera-de-distribución)
- `a` es el área permitida (ej: 0.1 = 10% de la imagen)

### ¿Por qué usar blur en lugar de negro?

Si ocultamos píxeles poniéndolos en negro (valor 0), le estamos pasando al modelo un input que **nunca vio durante el entrenamiento** (imágenes con grandes parches negros artificiales). Esto puede causar respuestas arbitrarias del modelo.

Al usar blur, estamos diciéndole al modelo "esta región tiene información no-informativa pero plausible". Es una perturbación más honesta.

```
Región revelada:     m(u,v) ≈ 1 → x₀(u,v) (imagen original)
Región ocultada:     m(u,v) ≈ 0 → blur(x₀)(u,v) (versión suavizada = fondo neutral)
```

### La pirámide de perturbaciones

Para hacer la optimización robusta, torchray usa una **pirámide de blurs** con distintas escalas:

```
blur(x₀) = promedio de [blur_scale1(x₀), blur_scale2(x₀), ..., blur_scaleN(x₀)]
```

Esto evita que la máscara se especialice en explotar los artefactos de un nivel específico de blur.

### La función de recompensa contrastiva

El parámetro `contrastive_reward` en el laboratorio es clave. En lugar de simplemente maximizar la activación de la clase objetivo, la recompensa contrastiva:

```
reward = Φ(m ⊗ x₀)[clase_objetivo] - max_{j ≠ clase_objetivo} Φ(m ⊗ x₀)[j]
```

Esto maximiza la diferencia entre la clase objetivo y la mejor clase alternativa. Produce máscaras más **discriminativas** (que realmente separan la clase de las otras) en lugar de máscaras que simplemente activan la red en general.

### El parámetro de área: decisión de granularidad

```
área = 5%:
  La máscara solo puede revelar el 5% de la imagen
  → Encuentra el mínimo suficiente para identificar el objeto
  → Más preciso, pero puede perderse contexto necesario

área = 10%:
  Más contexto, máscaras más estables
  → Balance habitual

área = 20%:
  Revela bastante del objeto
  → Más fácil de optimizar, máscaras más suaves
```

### Interpretación comparativa: base vs. fine-tuned

Este experimento del laboratorio es uno de los más instructivos:

**Modelo base (entrenado desde cero en flores):**

```
La máscara se dispersa por el fondo:
┌─────────────────┐
│  [calor]        │   ← el modelo aprendió que el suelo/contexto
│     [calor]     │     predice mejor la clase que la flor misma
│        [flor]   │   ← la flor real tiene poco peso
└─────────────────┘
```

**¿Por qué?** Con pocas imágenes de entrenamiento, el modelo aprende correlaciones del contexto (este tipo de suelo aparece con esta flor) porque es más "fácil" que aprender la forma/color de la flor.

**Modelo fine-tuned (pre-entrenado en ImageNet):**

```
La máscara se concentra en la flor:
┌─────────────────┐
│                 │
│    [calor]      │   ← la flor es la fuente de información
│   [máximo]      │
└─────────────────┘
```

**¿Por qué?** Las representaciones pre-aprendidas en ImageNet ya saben detectar formas, texturas y colores. El fine-tuning solo adapta estas representaciones robustas al dominio flores, sin necesidad de aprender shortcuts del contexto.

---

## 12. Síntesis: Lecciones que Trascienden las Arquitecturas

### Lección 1: La eficiencia es un principio de diseño, no un afterthought

VGG fue gigante porque nadie pensó en la eficiencia. Inception (1×1 conv) y ResNet (bottleneck) incorporaron la eficiencia como principio central de diseño. Las arquitecturas modernas (EfficientNet, MobileNet, Vision Transformers) van aún más lejos.

**Regla:** Antes de añadir parámetros, pregunta: ¿puedo obtener el mismo campo receptivo / capacidad con menos?

### Lección 2: Reformular el problema puede ser más poderoso que resolverlo mejor

ResNet no encontró una mejor forma de entrenar redes profundas. ResNet **cambió la pregunta**: en lugar de "aprende H(x)", pregunta "aprende F(x) = H(x) - x". Esta reformulación hizo trivial lo que antes era difícil.

**Regla:** Cuando algo es difícil de optimizar, prueba reformularlo.

### Lección 3: Las arquitecturas tienen suposiciones implícitas sobre el dominio

VGG asume homogeneidad: todas las capas usan el mismo tamaño de filtro. Inception asume multi-escala: los objetos aparecen a distintas escalas. ResNet asume "añadir capas es arriesgado": mejor aprender perturbaciones.

Si estas suposiciones no se cumplen en tu dominio, la arquitectura puede ser subóptima.

### Lección 4: La interpretabilidad revela los fundamentos del aprendizaje

Feature Visualization muestra que las CNN aprenden una jerarquía de conceptos de forma emergente, sin programarla. Attribution muestra que los modelos pueden aprender el concepto equivocado y aún así tener buena accuracy. Estas herramientas son el microscopio del deep learning.

### Lección 5: Accuracy en test no es suficiente

El ejemplo del watermark del caballo y el modelo de flores que aprende el fondo demuestran que un modelo puede tener alta accuracy y estar completamente equivocado en lo que aprendió. La interpretabilidad cierra este gap.

---

## Resumen Visual: Evolución de las Arquitecturas

```
     Parámetros
         ↑
  150M ──│────────────────────────VGG-19──────────────────────────────
         │                      ●
  100M ──│
         │
   50M ──│    ResNet-152                           
         │       ●                         
   25M ──│     ResNet-50                   Inception-v4
         │       ●                             ●          
   10M ──│                   GoogLeNet
         │                       ●
    5M ──│
         │
    1M ──│─────────────────────────────────────────────────────────────→ Accuracy
              55%    60%    65%    70%    75%    80%    85%

Conclusión: ResNet-50 y GoogLeNet son óptimos en el trade-off parámetros/accuracy.
            VGG tiene muchos parámetros para su accuracy.
            Inception-v4 lidera en accuracy a un costo razonable.
```

---

## Preguntas Frecuentes y Conceptos Comunes de Confusión

### ¿Un filtro 1×1 "ve" un solo píxel? ¿Para qué sirve entonces?

No confundir dimensión espacial con riqueza de información. Un filtro 1×1 opera sobre **todos los canales** de esa posición. Si la capa anterior tiene 256 canales, el filtro 1×1 ve un vector de 256 valores (un vector de características rico) y lo proyecta a otra dimensión.

### ¿Por qué ResNet-50 tiene más FLOPs que ResNet-34 si es "más eficiente"?

ResNet-50 usa bottlenecks (1×1-3×3-1×1) que tienen más operaciones totales que los bloques simples (3×3-3×3) de ResNet-34, pero con 50 capas vs 34. La eficiencia del bottleneck se nota en redes aún más profundas (101, 152) donde escala mejor.

### ¿Los clasificadores auxiliares de Inception siguen usándose?

No en Inception-v4 ni en arquitecturas modernas. BN y los residuales resolvieron el problema del vanishing gradient de forma más elegante. Los clasificadores auxiliares son un artefacto histórico de 2014.

### ¿Feature Visualization y Saliency Maps son lo mismo?

No. 
- **Feature Visualization**: optimiza la imagen para maximar una activación (no depende de una imagen real)
- **Saliency Maps / Attribution**: dados una imagen real y una predicción, identifica qué píxeles causaron esa predicción

Son complementarios: Feature Visualization para entender el modelo en general; Attribution para entender una predicción específica.

### ¿Por qué la imagen generada por Feature Visualization no se parece a una foto real?

Porque el espacio de imágenes que maximizan una neurona es vastísimo, y la mayoría de esos inputs son ruido estructurado. La regularización (frecuencia, transformaciones, decorrelación) intenta empujar la solución hacia el subespacio de "imágenes naturales", pero aún así las imágenes generadas son "alucinaciones" del modelo, no recreaciones de fotos reales.
