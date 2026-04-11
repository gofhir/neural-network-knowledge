# Analisis del Paper: Visualizing the Loss Landscape of Neural Nets

**Autores**: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein
**Instituciones**: University of Maryland, United States Naval Academy, Cornell University
**Publicado en**: 32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montreal, Canada

> PDF descargado en: [papers/3_LossLandscape_Li2018.pdf](3_LossLandscape_Li2018.pdf)

---

## Datos Clave del Paper

| Aspecto | Detalle |
|---|---|
| **Ano** | 2018 (arXiv: Nov 2017, conferencia: NeurIPS 2018) |
| **Citas** | Uno de los papers mas influyentes en comprension de Deep Learning (>4,000 citas) |
| **Autores notables** | Tom Goldstein (U. Maryland, reconocido en optimizacion), Christoph Studer (Cornell, procesamiento de senales) |
| **Idea central** | Metodo de "filter normalization" para visualizar loss landscapes de forma significativa y comparable entre arquitecturas |
| **Impacto** | Cambio la forma en que la comunidad entiende por que ciertas arquitecturas (ResNets) son mas entrenables que otras |
| **Codigo** | https://github.com/tomgoldstein/loss-landscape |

---

## 1. Problema que Resuelve

Entrenar redes neuronales requiere minimizar una funcion de loss altamente no-convexa en un espacio de **millones de dimensiones**. Surge una pregunta fundamental: por que SGD encuentra buenos minimos en algunas arquitecturas pero falla en otras?

```text
PREGUNTAS ABIERTAS EN 2018:
  - Por que las redes profundas SIN skip connections son dificiles de entrenar?
  - Por que batch size pequeno generaliza mejor que batch size grande?
  - Existe una relacion entre la "forma" del loss landscape y la generalizacion?
  - Por que ResNets funcionan y VGGs profundas no?
```

**El problema especifico de visualizacion**: Los loss landscapes viven en espacios de millones de dimensiones. Para visualizarlos, se proyectan a 1D o 2D usando direcciones aleatorias. Pero estas visualizaciones ingenuas son **enganosas** porque no consideran la **invariancia de escala** de las redes neuronales.

```text
PROBLEMA DE ESCALA:
  Red A: pesos grandes → landscape parece "plano" (perturbaciones pequenas relativas)
  Red B: pesos pequenos → landscape parece "afilado" (perturbaciones grandes relativas)

  Pero Red A y Red B pueden ser EQUIVALENTES!
  (batch normalization hace que la escala de los pesos sea irrelevante)

  → Las comparaciones de "sharpness" entre minimizadores son FALSAS
    sin una normalizacion adecuada
```

**Solucion propuesta**: Filter normalization — un metodo simple que normaliza las direcciones aleatorias para que sean comparables entre redes con diferentes escalas de pesos.

---

## 2. Fundamentos de Visualizacion de Loss Functions (Seccion 3)

### 2.1. El loss function de una red neuronal

```text
Dada una red con parametros theta, datos {x_i} y etiquetas {y_i}:

  L(theta) = (1/m) * SUM_{i=1}^{m} l(x_i, y_i; theta)

donde:
  theta = vector de TODOS los pesos de la red (millones de dimensiones)
  l(x_i, y_i; theta) = loss por cada ejemplo (e.g., cross-entropy)
  m = numero de ejemplos de entrenamiento
```

Visualizar L(theta) directamente es imposible — necesitamos reducir la dimensionalidad.

### 2.2. Interpolacion lineal 1D

```text
Dados dos conjuntos de parametros theta y theta':
  theta(alpha) = (1 - alpha) * theta + alpha * theta'

Se grafica:  f(alpha) = L(theta(alpha))

          Loss
          |  *
          | * *
          |*   *
          |     *          *
          |      *       *
          |       *    *
          |        * *
          |         *
          +--------------------> alpha
          0                   1
        theta               theta'

Usos:
  - Estudiar la "barrera" entre dos minimos
  - Comparar sharpness entre soluciones de batch grande vs pequeno
  - Goodfellow et al. (2015) usaron este metodo

Limitaciones:
  - Solo muestra UNA linea en un espacio de millones de dimensiones
  - Las no-convexidades pueden NO aparecer en 1D
  - No considera batch normalization ni invariancia de escala
```

### 2.3. Contour Plots con direcciones aleatorias (2D)

Se eligen un punto central theta* y dos vectores de direccion delta y eta:

```text
Caso 1D (linea):
  f(alpha) = L(theta* + alpha * delta)

Caso 2D (superficie):
  f(alpha, beta) = L(theta* + alpha * delta + beta * eta)

Esto genera una superficie 3D o un contour plot 2D:

         beta
          ^
          |  ___________
          | /  contornos \
          ||   de loss    |
          | \  alrededor  /
          |  \  de theta*/
          |   ----------
          +-------------------> alpha
                theta* esta en el centro (alpha=0, beta=0)
```

**Problema critico**: Si delta y eta son puramente aleatorios (Gaussianos), la escala de la perturbacion depende de la magnitud de los pesos, haciendo las comparaciones entre redes **sin sentido**.

---

## 3. Modelo Formal: Filter Normalization (Seccion 4)

### 3.1. El problema de la invariancia de escala

```text
Con activaciones ReLU y/o batch normalization:

  Si multiplicamos los pesos de la capa l por 10:
    w_l → 10 * w_l

  Y dividimos los de la capa l+1 por 10:
    w_{l+1} → w_{l+1} / 10

  LA RED COMPUTA EXACTAMENTE LO MISMO.

  Pero los loss landscapes lucen COMPLETAMENTE DIFERENTES:
  - La version con pesos grandes parece "plana"
  - La version con pesos pequenos parece "afilada"

  Dinh et al. (2017) explotaron esto para construir pares
  de redes EQUIVALENTES con sharpness arbitrariamente diferente,
  demostrando que las metricas de sharpness sin normalizar
  son INUTIL para predecir generalizacion.
```

### 3.2. Metodo de Filter Normalization

Para generar una direccion normalizada d compatible con los parametros theta:

```text
ALGORITMO FILTER NORMALIZATION:
  
  Entrada: parametros theta, direccion aleatoria d ~ N(0, I)
  
  Para cada capa i de la red:
    Para cada filtro j de la capa i:
      1. Tomar d_{i,j} (la porcion de d correspondiente al filtro j, capa i)
      2. Tomar theta_{i,j} (los pesos del filtro j, capa i)
      3. Reemplazar:
      
         d_{i,j} ← (d_{i,j} / ||d_{i,j}||) * ||theta_{i,j}||
      
      donde || . || es la norma de Frobenius

  Resultado: d tiene la misma "escala" que theta, filtro por filtro
```

### 3.3. Paso a paso con ejemplo numerico

```text
Ejemplo: Red con 2 capas convolucionales

CAPA 1, Filtro 1:
  theta_{1,1} = [0.3, -0.5, 0.2, 0.4]     ||theta_{1,1}|| = 0.74
  d_{1,1}     = [1.2, 0.8, -0.3, 0.5]      ||d_{1,1}||     = 1.56

  Normalizacion:
  d_{1,1} ← (d_{1,1} / 1.56) * 0.74
  d_{1,1} = [0.57, 0.38, -0.14, 0.24]      ||d_{1,1}|| = 0.74  ✓

CAPA 1, Filtro 2:
  theta_{1,2} = [0.1, -0.1, 0.05, 0.08]    ||theta_{1,2}|| = 0.17
  d_{1,2}     = [0.9, -1.1, 0.7, 0.3]      ||d_{1,2}||     = 1.63

  Normalizacion:
  d_{1,2} ← (d_{1,2} / 1.63) * 0.17
  d_{1,2} = [0.094, -0.115, 0.073, 0.031]  ||d_{1,2}|| = 0.17  ✓

EFECTO:
  - Filtros con pesos GRANDES → direccion de perturbacion GRANDE
  - Filtros con pesos PEQUENOS → direccion de perturbacion PEQUENA
  - La perturbacion es PROPORCIONAL a la escala del filtro
  - Esto elimina el efecto artificial de la escala de los pesos
```

### 3.4. Diferencia con layer-wise normalization

```text
Layer-wise normalization (Im et al.):
  Normaliza toda la capa como un solo bloque
  d_i ← (d_i / ||d_i||) * ||theta_i||

  Problema: dentro de una capa, filtros con pesos grandes
  dominan la perturbacion, y filtros pequenos son ignorados.

Filter-wise normalization (propuesta del paper):
  Normaliza CADA FILTRO individualmente
  d_{i,j} ← (d_{i,j} / ||d_{i,j}||) * ||theta_{i,j}||

  Ventaja: cada filtro recibe una perturbacion proporcional
  a su propia escala → captura la geometria INTRINSECA

Para capas fully connected (FC):
  Un filtro = los pesos que generan UNA neurona de salida
  (equivalente a una Conv 1x1 con 1 feature map de salida)
```

### 3.5. Formulacion completa de la visualizacion

```text
Dados:
  theta* = minimizador encontrado por SGD
  delta  = direccion aleatoria filter-normalized
  eta    = otra direccion aleatoria filter-normalized (independiente)

Plot 2D:
  f(alpha, beta) = L(theta* + alpha * delta + beta * eta)

  Se evalua en una grilla, e.g., alpha, beta in [-1, 1]
  con resolucion de 51x51 puntos (2601 evaluaciones de loss)

  Cada evaluacion requiere un forward pass sobre TODO el dataset
  → Computacionalmente costoso pero necesario para alta resolucion
```

---

## 4. El Dilema Sharp vs Flat (Seccion 5)

### 4.1. El debate historico

```text
HIPOTESIS PREVALENTE (Keskar et al., 2017):
  - SGD con batch pequeno → minimos "flat" → buena generalizacion
  - SGD con batch grande  → minimos "sharp" → mala generalizacion

CONTRA-ARGUMENTO (Dinh et al., 2017):
  - Se pueden construir redes equivalentes con sharpness arbitraria
  - La "sharpness" no es invariante a reparametrizaciones
  - → La relacion sharpness-generalizacion puede ser un artefacto

CONTRIBUCION DE ESTE PAPER:
  - Demuestra que SIN normalizacion, los plots de sharpness son ENGANOSOS
  - CON filter normalization, SI hay correlacion entre sharpness y generalizacion
  - Pero la diferencia es mucho mas sutil de lo que se creia
```

### 4.2. Experimento demostrativo: VGG-9 en CIFAR-10

```text
Configuracion:
  Red: VGG-9 con batch normalization
  Epochs: fijo (300)
  Dos condiciones:
    (a) Batch size 128 (pequeno)  → test error 7.37%
    (b) Batch size 8192 (grande) → test error 11.07%

SIN normalizacion, SIN weight decay:
  - Batch pequeno: landscape AMPLIO (parece flat)
  - Batch grande:  landscape ESTRECHO (parece sharp)
  → "Confirma" la hipotesis sharpness-generalizacion

SIN normalizacion, CON weight decay (5e-4):
  - Batch pequeno: landscape ESTRECHO (parece sharp!)
  - Batch grande:  landscape AMPLIO (parece flat!)
  → CONTRADICE la hipotesis! (batch pequeno sigue generalizando mejor)

  La explicacion: weight decay cambia la ESCALA de los pesos
  - Batch pequeno + weight decay → pesos mas pequenos → parece sharp
  - Batch grande + weight decay → pesos mas grandes → parece flat
  - Es un ARTEFACTO de la escala, no de la geometria real

CON filter normalization:
  - La diferencia de sharpness es SUTIL pero CONSISTENTE
  - Batch grande produce minimos ligeramente mas sharp
  - La correlacion con generalizacion se mantiene
  - Las comparaciones son SIGNIFICATIVAS
```

---

## 5. Estructura No-Convexa de Loss Surfaces (Seccion 6)

### 5.1. Efecto de la profundidad de la red

Este es uno de los hallazgos mas impactantes del paper. Se comparan redes con y sin skip connections a diferentes profundidades en CIFAR-10:

```text
                     CON skip connections        SIN skip connections
                     (ResNet)                    (ResNet-noshort / VGG-like)

  20 capas:          Convexo, suave              Convexo, suave
                     Test error: 7.37%           Test error: 8.18%

  56 capas:          Convexo, suave              CAOTICO, no-convexo
                     Test error: 5.89%           Test error: 13.31%

  110 capas:         Convexo, suave              MUY CAOTICO, inentrenable
                     Test error: 5.79%           Test error: 16.44%

DIAGRAMA ASCII - Vista esquematica de los contour plots:

  ResNet-56 (con skip):        ResNet-56-noshort (sin skip):
  +-------------------+        +-------------------+
  |                   |        |  /\    /\  /\     |
  |    ___________    |        | /  \  / _\/ /\    |
  |   /           \   |        |/   _\/    \/  \   |
  |  |   suave,    |  |        |\  / /\  /\  /\_\  |
  |  |  contornos  |  |        | \/ /  \/  \/   /  |
  |  |  concentri- |  |        |  \/   /\  /\  /   |
  |  |    cos      |  |        |  /\  /  \/  \/    |
  |   \___________/   |        | /  \/    /\  /\   |
  |                   |        |/    \   /  \/  \   |
  +-------------------+        +-------------------+
   Error: 5.89%                 Error: 13.31%
   Landscape: convexo           Landscape: caotico
```

### 5.2. Por que las skip connections son cruciales

```text
SIN skip connections (profundidad creciente):
  20 capas → 56 capas → 110 capas
  convexo  → caotico  → MUY caotico

  A medida que la red se hace mas profunda:
  1. Aparecen grandes regiones de non-convexity
  2. Los gradientes dejan de apuntar al minimizador
  3. El loss crece dramaticamente al moverse en cualquier direccion
  4. Los minimizadores se vuelven "sharp" (estrechos)
  5. La red se vuelve INENTRENABLE (SGD no converge)
  6. 156 capas sin skip: SGD NO PUDO entrenar ni con lr muy bajo

CON skip connections:
  20 capas → 56 capas → 110 capas
  convexo  → convexo  → convexo!

  Las skip connections PREVIENEN la transicion a comportamiento caotico.
  El loss landscape se mantiene suave y casi convexo
  independientemente de la profundidad.
  
  → ESTO EXPLICA por que ResNets pueden tener cientos de capas
    mientras que VGGs se limitan a ~19 capas.
```

### 5.3. Efecto del ancho de la red (Wide ResNets)

```text
Wide-ResNet-56 en CIFAR-10 (k = multiplicador de filtros):

  CON skip connections:
    k=1: Error 5.89%   Landscape: convexo, algunas irregularidades
    k=2: Error 5.07%   Landscape: mas suave
    k=4: Error 4.34%   Landscape: muy suave, contornos concentricos
    k=8: Error 3.93%   Landscape: extremadamente suave

  SIN skip connections:
    k=1: Error 13.31%  Landscape: muy caotico
    k=2: Error 10.26%  Landscape: caotico pero menos
    k=4: Error 9.69%   Landscape: irregular
    k=8: Error 8.70%   Landscape: bastante suave

CONCLUSION:
  Aumentar el ANCHO de la red tambien "suaviza" el loss landscape.
  Redes mas anchas → loss landscape mas convexo → mejor generalizacion.
  Skip connections + ancho = maxima suavidad.
```

### 5.4. Implicaciones para la inicializacion

```text
Observacion clave de los loss landscapes:

  +-----------------------------------------+
  |  Region de LOSS ALTO                     |
  |  (no-convexa, caotica)                   |
  |                                          |
  |  +-----------------------------------+  |
  |  |  Region de LOSS BAJO               |  |
  |  |  (convexa, suave)                  |  |
  |  |                                    |  |
  |  |       * minimizador                |  |
  |  |                                    |  |
  |  +-----------------------------------+  |
  |            frontera ~loss 4              |
  +-----------------------------------------+

  Con inicializacion de Glorot/He:
    Loss inicial tipico < 2.5 → DENTRO de la region convexa
    → SGD "ve" un landscape benigno y converge facilmente

  Para redes profundas sin skip connections:
    La region convexa se ENCOGE o DESAPARECE
    → La inicializacion cae en la region caotica
    → Gradientes son "shattered" (ruidosos, no informativos)
    → SGD NO puede escapar → entrenamiento FALLA
```

---

## 6. Medicion Cuantitativa de Non-Convexity (Seccion 6, final)

### 6.1. Eigenvalores del Hessiano

```text
Para medir la no-convexidad de forma cuantitativa:

  Hessiano H = matriz de segundas derivadas de L(theta)
  
  Eigenvalores de H:
    lambda_max = eigenvalor maximo (curvatura mas positiva)
    lambda_min = eigenvalor minimo (curvatura mas negativa)

  Ratio: |lambda_min| / |lambda_max|
    ≈ 0   → region convexa (curvaturas negativas insignificantes)
    > 0.3 → region significativamente no-convexa

  Se calcula usando el metodo de Lanczos con reinicio implicito
  (solo requiere productos Hessiano-vector via automatic differentiation)
```

### 6.2. Heat maps de non-convexity

```text
Se genera un heat map coloreando cada punto del loss landscape:

  ResNet-56 (con skip):
  +-------------------+
  |  azul  azul  azul |    azul = convexo
  |  azul  AZUL  azul |    (|lambda_min/lambda_max| ≈ 0)
  |  azul  azul  azul |
  +-------------------+
  → Casi completamente convexo en toda la region

  ResNet-56-noshort (sin skip):
  +-------------------+
  |  amar  AMAR  amar |    amarillo = no-convexo
  |  azul  azul  amar |    (|lambda_min/lambda_max| > 0.3)
  |  amar  azul  AMAR |
  +-------------------+
  → Regiones convexas rodeadas de regiones caoticas

  DenseNet-121:
  +-------------------+
  |  azul  azul  azul |
  |  azul  AZUL  azul |    Completamente convexo
  |  azul  azul  azul |    (incluso mas que ResNet)
  +-------------------+
```

---

## 7. Visualizacion de Trayectorias de Optimizacion (Seccion 7)

### 7.1. Por que las direcciones aleatorias fallan para trayectorias

```text
Problema:
  Las trayectorias de SGD viven en un subespacio de MUY BAJA DIMENSION
  del espacio de parametros.

  Dos vectores aleatorios en R^n tienen similitud coseno esperada:
    E[cos(u,v)] ≈ sqrt(2 / (pi * n))

  Para n = 1,000,000 (parametros tipicos):
    E[cos(u,v)] ≈ 0.0008

  → Una direccion aleatoria es CASI ORTOGONAL a la trayectoria
  → Proyectar la trayectoria en direcciones aleatorias
     captura ~0% de la variacion
  → El plot muestra RUIDO, no la trayectoria real

Resultado visual con direcciones aleatorias:
  +-------------------+
  |                   |
  |    .  ..  . .     |    Puntos dispersos sin estructura
  |  .   . .    .     |    (zoom a escala ~0.001)
  |    .    ..  .     |    → INUTIL para visualizacion
  |  .  .  .   .     |
  +-------------------+
```

### 7.2. Solucion: PCA sobre la trayectoria

```text
ALGORITMO:
  1. Registrar theta_i en cada epoca i (i = 0, 1, ..., n)
  2. Construir matriz de diferencias:
     M = [theta_0 - theta_n, theta_1 - theta_n, ..., theta_{n-1} - theta_n]
  3. Aplicar PCA a M
  4. Tomar las dos componentes principales como ejes de visualizacion
  5. Proyectar la trayectoria y el loss landscape

Resultado:
  - El 1er componente PCA captura ~25-88% de la variacion
  - El 2do componente PCA captura ~6-17% de la variacion
  - → Juntos capturan la mayor parte del movimiento de SGD

Trayectoria tipica sobre contour plot:
  +---------------------------+
  |        contornos de loss  |
  |  * inicio                 |
  |   \                       |
  |    \  (baja rapido,       |
  |     \  perpendicular a    |
  |      \  contornos)        |
  |       \                   |
  |        *---*--*-*-*       |
  |          (converge, se    |
  |           mueve paralelo  |
  |     * fin  a contornos)   |
  +---------------------------+
  
  Observaciones:
  - Al inicio: movimiento PERPENDICULAR a contornos (descenso puro)
  - Al final: movimiento mas ESTOCASTICO (cerca del minimo)
  - Cambios de learning rate (schedule) producen saltos visibles
```

### 7.3. Comparacion de optimizadores

```text
VGG-9 en CIFAR-10, proyectado via PCA:

SGD (con weight decay):
  - Trayectoria con curvas amplias
  - 1er PCA: ~25% de variacion (la trayectoria es mas compleja)
  - Converge a un minimo amplio y plano

Adam (con weight decay):
  - Trayectoria mas directa
  - 1er PCA: ~33% de variacion
  - Converge mas rapido pero a un minimo similar

SGD (sin weight decay):
  - 1er PCA: ~29% de variacion
  - Trayectoria mas errática al final

Adam (sin weight decay):
  - 1er PCA: ~40% de variacion
  - Trayectoria mas concentrada en pocas direcciones
```

---

## 8. Resultados Experimentales Consolidados

### 8.1. Arquitecturas evaluadas

| Arquitectura | Capas | Skip Connections | Dataset | Test Error |
|---|---|---|---|---|
| ResNet-20 | 20 | Si | CIFAR-10 | 7.37% |
| ResNet-56 | 56 | Si | CIFAR-10 | 5.89% |
| ResNet-110 | 110 | Si | CIFAR-10 | 5.79% |
| ResNet-20-noshort | 20 | No | CIFAR-10 | 8.18% |
| ResNet-56-noshort | 56 | No | CIFAR-10 | 13.31% |
| ResNet-110-noshort | 110 | No | CIFAR-10 | 16.44% |
| Wide-ResNet-56 (k=8) | 56 | Si | CIFAR-10 | 3.93% |
| Wide-ResNet-56-NS (k=8) | 56 | No | CIFAR-10 | 8.70% |
| DenseNet-121 | 121 | Si (dense) | CIFAR-10 | - |
| VGG-9 | 9 | No | CIFAR-10 | 6.0-11.07% |

### 8.2. Correlacion sharpness-generalizacion

```text
CON filter normalization:

  Observacion 1: Minimizadores mas FLAT → MENOR test error
    ResNet-56 (flat):     5.89%
    ResNet-56-NS (sharp): 13.31%

  Observacion 2: Skip connections → landscape PLANO
    Con skip: contornos suaves, concentricos
    Sin skip: contornos irregulares, caoticos

  Observacion 3: Mas ancho → mas PLANO → mejor generalizacion
    k=1: 5.89%  (contornos irregulares)
    k=8: 3.93%  (contornos casi perfectamente concentricos)

  Observacion 4: Small batch → ligeramente mas flat que large batch
    Batch 128:  7.37%  (contornos ligeramente mas anchos)
    Batch 8192: 11.07% (contornos ligeramente mas estrechos)
    (Diferencia SUTIL despues de filter normalization)
```

### 8.3. Tabla resumen de hallazgos clave

| Factor | Efecto en Loss Landscape | Efecto en Generalizacion |
|---|---|---|
| Skip connections | Previene transicion a caos | Mejora dramatica en redes profundas |
| Mas profundidad (sin skip) | Convexo → caotico | Degradacion severa |
| Mas profundidad (con skip) | Se mantiene convexo | Mejora consistente |
| Mas ancho | Mas suave y convexo | Mejora consistente |
| Batch size grande | Ligeramente mas sharp | Peor generalizacion |
| Weight decay | Cambia escala (artefacto sin normalizar) | Mejora generalizacion |

---

## 9. Impacto Historico y Legado

```text
2015: Goodfellow et al. — primeros plots 1D de loss surfaces
      → Conclusion (incorrecta): "los loss landscapes son casi lineales"

2017: Keskar et al. — "sharp minima generalizan peor"
      Dinh et al.  — "la sharpness no es invariante, es inutil"
      → Debate irresuelto sobre sharpness y generalizacion

2018: ESTE PAPER — resuelve el debate:
      → Filter normalization permite comparaciones SIGNIFICATIVAS
      → CON normalizacion, sharpness SI correlaciona con generalizacion
      → Muestra visualmente POR QUE skip connections son necesarias
      → El loss landscape explica la trainability de redes profundas

2019+: Impacto en investigacion posterior:
  - SAM (Sharpness-Aware Minimization, Foret et al., 2021):
    → Optimizador que BUSCA minimos flat explicitamente
    → Directamente inspirado por la relacion flatness-generalizacion
    → State-of-the-art en multiples benchmarks

  - Neural Architecture Search (NAS):
    → La geometria del landscape como criterio de busqueda
    → Arquitecturas con landscapes suaves son preferidas

  - Understanding generalization:
    → Base para estudios de generalizacion basados en geometria
    → Flat minima hypothesis se fortalece con normalizacion adecuada

  - Landscape analysis de Transformers:
    → Tecnicas de este paper aplicadas a analizar
       por que layer normalization es crucial en Transformers
    → Paralelo directo con skip connections en ResNets

Hoy (2026):
  - Filter normalization es ESTANDAR para visualizar loss landscapes
  - El repositorio loss-landscape sigue siendo la herramienta de referencia
  - La relacion "landscape suave ↔ buena generalizacion" es ampliamente aceptada
  - SAM y variantes (ASAM, GSAM) usan principios de este paper
```

---

## 10. Resumen en Una Pagina

```text
PROBLEMA:  No se entendia por que ciertas arquitecturas (ResNets) son
           mas entrenables que otras (VGGs profundas), ni como visualizar
           loss landscapes de forma significativa y comparable.

SOLUCION:  Filter normalization — normalizar las direcciones aleatorias
           de visualizacion filtro por filtro para eliminar artefactos
           de escala.

COMO:
  1. Elegir theta* (minimizador entrenado) como centro
  2. Generar direccion aleatoria d ~ N(0, I)
  3. Para cada filtro j en cada capa i:
     d_{i,j} ← (d_{i,j} / ||d_{i,j}||) * ||theta_{i,j}||
  4. Repetir para segunda direccion eta (plots 2D)
  5. Evaluar f(alpha, beta) = L(theta* + alpha*delta + beta*eta)
  6. Generar contour plot o surface plot

HALLAZGOS PRINCIPALES:
  1. Sin normalizacion, las comparaciones de sharpness son ENGANOSAS
     (la escala de los pesos distorsiona la geometria aparente)
  2. Con filter normalization, sharpness correlaciona con generalizacion
  3. Skip connections PREVIENEN la transicion de convexo a caotico
     al aumentar la profundidad
  4. Sin skip connections, redes de 56+ capas tienen landscapes
     CAOTICOS y son dificiles/imposibles de entrenar
  5. Redes mas ANCHAS tienen landscapes mas SUAVES
  6. Las trayectorias de optimizacion viven en subespacios de baja
     dimension → usar PCA (no direcciones aleatorias) para visualizarlas

RESULTADOS CLAVE:
  - ResNet-56 (con skip):  landscape convexo,  error 5.89%
  - ResNet-56 (sin skip):  landscape caotico,  error 13.31%
  - Wide-ResNet-56 (k=8):  landscape muy suave, error 3.93%
  - Redes profundas sin skip (>56 capas): INENTRENABLES
  - Batch pequeno → minimos ligeramente mas flat (efecto sutil)

CONFIGURACION EXPERIMENTAL:
  - Dataset: CIFAR-10 (principal)
  - Optimizador: SGD con Nesterov momentum, batch 128
  - Learning rate: 0.1, decae x10 en epochs 150, 225, 275
  - Weight decay: 0.0005
  - Epochs: 300
  - Resolucion de plots: 51x51 puntos

IMPACTO:
  - Resolvio el debate "sharp vs flat minima"
  - Explico visualmente por que skip connections son necesarias
  - Filter normalization se volvio estandar para visualizacion
  - Inspiro SAM y otros optimizadores basados en sharpness
  - Codigo abierto: github.com/tomgoldstein/loss-landscape
```
