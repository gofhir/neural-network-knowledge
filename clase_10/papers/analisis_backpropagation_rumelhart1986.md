# Analisis del Paper: Learning Representations by Back-Propagating Errors

**Autores**: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
**Instituciones**: Institute for Cognitive Science, UC San Diego (Rumelhart, Williams); Department of Computer Science, Carnegie-Mellon University (Hinton)
**Publicado en**: Nature, Vol. 323, pp. 533-536, 9 October 1986
**Nota**: El paper de Nature es de 3 paginas; la version extendida (technical report) tiene ~23 paginas con detalles matematicos completos

> PDF descargado en: [papers/1_Backpropagation_Rumelhart1986.pdf](1_Backpropagation_Rumelhart1986.pdf)

---

## Datos Clave del Paper

| Aspecto | Detalle |
|---|---|
| **Ano** | 1986 (publicado en Nature, 9 de octubre) |
| **Citas** | Uno de los papers mas citados de toda la ciencia (>90,000 citas en Google Scholar) |
| **Autores notables** | Geoffrey Hinton ("padre del Deep Learning", Premio Turing 2018, Nobel de Fisica 2024), David Rumelhart (pionero de redes neuronales, psicologia cognitiva) |
| **Idea central** | Un algoritmo para calcular eficientemente el gradiente del error respecto a cada peso en una red neuronal multicapa, permitiendo que la red aprenda representaciones internas utiles |
| **Impacto** | Hizo posible el entrenamiento de redes neuronales profundas; es la base de practicamente todo el Deep Learning moderno |

---

## 1. Problema que Resuelve

### 1.1. El problema del credit assignment en redes multicapa

Antes de 1986, las redes neuronales estaban limitadas al **Perceptron** (Rosenblatt, 1958), que solo tenia una capa de pesos ajustables. Minsky y Papert (1969) demostraron formalmente que el Perceptron no puede resolver problemas que no sean **linealmente separables** (como XOR).

```text
PROBLEMA XOR — imposible para un Perceptron:

  Input1  Input2  |  Output deseado
    0       0     |      0
    0       1     |      1
    1       0     |      1
    1       1     |      0

  Grafico en 2D:
     Input2
      1 |  (1)         (0)
        |   ●           ○
        |
      0 |  (0)         (1)
        |   ○           ●
        +------------------
           0            1   Input1

  No existe UNA linea recta que separe los ● de los ○
  → Se necesitan al menos 2 lineas → se necesita una capa oculta
```

La solucion era obvia: agregar **capas ocultas** (hidden layers). Pero el problema era: **como ajustar los pesos de las capas ocultas?**

```text
                  ERROR
                    ↓
  Input → [Capa Oculta] → [Capa de Salida] → Output
              ???              ← gradiente conocido

  En la capa de salida: sabemos el error (output - target)
  En la capa oculta:    NO hay target directo
                         ¿Como saber que "culpa" tiene cada neurona oculta?

  Este es el "credit assignment problem"
```

### 1.2. Intentos previos

- **Perceptron learning rule** (Rosenblatt, 1958): Solo funciona para una capa
- **Boltzmann machines** (Hinton & Sejnowski, 1983): Podian aprender con capas ocultas pero eran extremadamente lentos (muestreo estocastico)
- **Derivaciones parciales del algoritmo**: Werbos (1974) y Parker (1985) habian descrito formas de backpropagation, pero sin demostrar que las redes realmente aprendian representaciones internas utiles

### 1.3. Contribucion clave de este paper

Rumelhart, Hinton y Williams no solo formalizaron el algoritmo de backpropagation, sino que **demostraron empiricamente** que:
1. Las redes multicapa pueden aprender representaciones internas utiles
2. Estas representaciones "emergen" automaticamente del proceso de aprendizaje
3. El algoritmo es practico y funciona en problemas reales

---

## 2. Modelo Formal

### 2.1. Arquitectura de la Red (Feedforward Network)

```text
  ARQUITECTURA MULTICAPA (ejemplo con 1 capa oculta):

  Capa de         Capa            Capa de
  Entrada         Oculta          Salida
  (input)         (hidden)        (output)

   x1 ─────┬──────→ h1 ───┬──────→ o1
            │╲      ╱│╲    │╲
   x2 ─────┼──╲──╱──┼──╲──┼──╲───→ o2
            │╲  ╲╱╲  │╲  ╲│╲  ╲
   x3 ─────┼──╳──╳──┼──╳──┼──╳──→ o3
            │╱  ╱╲╱  │╱  ╱│╱  ╱
   x4 ─────┼──╱──╲──┼──╱──┼──╱
            │╱      ╲│╱    │╱
   x5 ─────┘────────→ h2 ──┘

         w_ji           w_kj
     (pesos input    (pesos hidden
      → hidden)       → output)

  Total de parametros: (5×2) + (2×3) = 16 pesos + 5 biases = 21
```

### 2.2. Forward Pass (propagacion hacia adelante)

Para cada neurona j, la activacion se calcula en dos pasos:

```text
PASO 1 — Entrada neta (net input):

  net_j = Σ_i  w_ji * y_i + b_j

  donde:
    w_ji = peso de la conexion de la neurona i a la neurona j
    y_i  = salida de la neurona i (de la capa anterior)
    b_j  = bias de la neurona j

PASO 2 — Funcion de activacion (squashing function):

  y_j = f(net_j) = 1 / (1 + exp(-net_j))    ← Sigmoide logistica

  Propiedades de la sigmoide:
    - Rango: (0, 1)
    - Es diferenciable en todo punto
    - Su derivada tiene una forma elegante:
      f'(x) = f(x) * (1 - f(x))

  Grafico de la sigmoide:

  y
  1.0 ┤                        ___________
      │                    ╱
  0.5 ┤               ╱
      │          ╱
  0.0 ┤_________╱
      └───────────────────────────────────→ net_j
         -6  -4  -2   0   2   4   6
```

### 2.3. Funcion de Error

El paper utiliza el **error cuadratico medio** (sum-of-squares error):

```text
E = (1/2) * Σ_c Σ_j (t_cj - y_cj)²

donde:
  c = indice del patron de entrenamiento
  j = indice de la neurona de salida
  t_cj = valor target (deseado) para la neurona j en el patron c
  y_cj = valor de salida (producido) para la neurona j en el patron c

Para un solo patron:
  E = (1/2) * Σ_j (t_j - y_j)²
```

### 2.4. Backpropagation — La regla de aprendizaje

El objetivo es calcular ∂E/∂w_ji para cada peso de la red, y luego actualizar los pesos en la direccion que reduce el error.

**Regla de actualizacion de pesos (gradient descent)**:

```text
Δw_ji = -η * (∂E/∂w_ji)

donde η (eta) es el learning rate
```

**Calculo del gradiente usando la chain rule**:

```text
∂E/∂w_ji = (∂E/∂net_j) * (∂net_j/∂w_ji)

Como net_j = Σ_i w_ji * y_i, entonces:
  ∂net_j/∂w_ji = y_i

Definimos δ_j = -(∂E/∂net_j)   ← "señal de error" de la neurona j

Entonces:
  Δw_ji = η * δ_j * y_i
```

**Calculo de delta para neuronas de SALIDA**:

```text
δ_j = (t_j - y_j) * f'(net_j)

Con sigmoide: f'(net_j) = y_j * (1 - y_j)

Entonces:
  δ_j = (t_j - y_j) * y_j * (1 - y_j)
        ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
        error          derivada de la
        directo        activacion
```

**Calculo de delta para neuronas OCULTAS (la clave del algoritmo)**:

```text
δ_j = f'(net_j) * Σ_k (δ_k * w_kj)

donde:
  k = neuronas de la capa SIGUIENTE (que reciben salida de j)
  δ_k = deltas ya calculados de la capa siguiente
  w_kj = peso de j hacia k

Con sigmoide:
  δ_j = y_j * (1 - y_j) * Σ_k (δ_k * w_kj)

INTUICION: El error de una neurona oculta es la suma ponderada
de los errores de las neuronas a las que esta conectada,
multiplicada por la derivada de su activacion.

→ Los errores se "propagan hacia atras" (back-propagate)
  desde la salida hasta la entrada.
```

### 2.5. Ejemplo Paso a Paso — Aprendiendo XOR

```text
RED PARA XOR:
  2 inputs → 2 neuronas ocultas → 1 neurona de salida
  Activacion: sigmoide
  Learning rate: η = 0.5

ESTADO INICIAL (pesos aleatorios pequenos):

  w_h1_x1 = 0.15    w_h1_x2 = 0.20    b_h1 = 0.35
  w_h2_x1 = 0.25    w_h2_x2 = 0.30    b_h2 = 0.35
  w_o_h1  = 0.40    w_o_h2  = 0.45    b_o  = 0.60

PATRON DE ENTRENAMIENTO: x1=1, x2=1, target=0

────────────────────────────────────────────────────────
FORWARD PASS:
────────────────────────────────────────────────────────

Neurona h1:
  net_h1 = 0.15*1 + 0.20*1 + 0.35 = 0.70
  y_h1   = σ(0.70) = 1/(1+exp(-0.70)) = 0.6682

Neurona h2:
  net_h2 = 0.25*1 + 0.30*1 + 0.35 = 0.90
  y_h2   = σ(0.90) = 1/(1+exp(-0.90)) = 0.7109

Neurona de salida o:
  net_o = 0.40*0.6682 + 0.45*0.7109 + 0.60 = 1.1872
  y_o   = σ(1.1872) = 0.7662

Error: E = 0.5*(0 - 0.7662)² = 0.2935

────────────────────────────────────────────────────────
BACKWARD PASS:
────────────────────────────────────────────────────────

Delta de la salida:
  δ_o = (t - y_o) * y_o * (1 - y_o)
      = (0 - 0.7662) * 0.7662 * (1 - 0.7662)
      = -0.7662 * 0.7662 * 0.2338
      = -0.1372

Deltas de las ocultas (BACKPROPAGATION):
  δ_h1 = y_h1*(1-y_h1) * (δ_o * w_o_h1)
       = 0.6682*0.3318 * (-0.1372 * 0.40)
       = 0.2217 * (-0.0549)
       = -0.01217

  δ_h2 = y_h2*(1-y_h2) * (δ_o * w_o_h2)
       = 0.7109*0.2891 * (-0.1372 * 0.45)
       = 0.2055 * (-0.0617)
       = -0.01269

────────────────────────────────────────────────────────
ACTUALIZACION DE PESOS (Δw = η * δ * input):
────────────────────────────────────────────────────────

Pesos oculta → salida:
  Δw_o_h1 = 0.5 * (-0.1372) * 0.6682 = -0.04584
  w_o_h1  = 0.40 + (-0.04584) = 0.3542

  Δw_o_h2 = 0.5 * (-0.1372) * 0.7109 = -0.04876
  w_o_h2  = 0.45 + (-0.04876) = 0.4012

Pesos entrada → oculta:
  Δw_h1_x1 = 0.5 * (-0.01217) * 1 = -0.00609
  w_h1_x1  = 0.15 + (-0.00609) = 0.1439

  (analogamente para los demas pesos)

→ Despues de muchas iteraciones con los 4 patrones de XOR,
  la red converge a una solucion correcta.
```

### 2.6. Momentum (mejora importante)

Los autores proponen agregar un termino de **momentum** para acelerar la convergencia:

```text
Δw_ji(t) = η * δ_j * y_i  +  α * Δw_ji(t-1)

donde:
  α = coeficiente de momentum (tipicamente 0.9)
  Δw_ji(t-1) = cambio de peso en la iteracion anterior

EFECTO:
  - Acelera convergencia en direcciones consistentes del gradiente
  - Amortigua oscilaciones en direcciones inconsistentes
  - Permite atravesar mesetas planas de la superficie de error

  Sin momentum:           Con momentum:
  ╱╲╱╲╱╲→ destino        ╱─────→ destino
  (zigzaguea)            (va mas directo)
```

### 2.7. Algoritmo Completo

```text
ALGORITMO BACKPROPAGATION:
═══════════════════════════════════════════════════

Entrada: Dataset D, arquitectura de red, η, α
Salida:  Pesos entrenados

1. Inicializar pesos aleatoriamente (valores pequenos)

2. REPETIR hasta convergencia:

   Para cada patron (x, t) en D:

     a) FORWARD PASS:
        Para cada capa l = 1, 2, ..., L:
          net_j = Σ_i w_ji * y_i + b_j
          y_j = σ(net_j)

     b) CALCULAR ERROR:
        E = (1/2) * Σ_j (t_j - y_j)²

     c) BACKWARD PASS:
        Para capa de salida:
          δ_j = (t_j - y_j) * y_j * (1 - y_j)

        Para cada capa oculta l = L-1, ..., 1:
          δ_j = y_j * (1-y_j) * Σ_k (δ_k * w_kj)
                                 ↑
                           capa l+1

     d) ACTUALIZAR PESOS:
        Para cada peso w_ji:
          Δw_ji = η * δ_j * y_i + α * Δw_ji_anterior
          w_ji  = w_ji + Δw_ji

3. RETORNAR pesos finales

═══════════════════════════════════════════════════

Complejidad computacional:
  Forward:  O(W)   donde W = numero total de pesos
  Backward: O(W)   ← misma complejidad que forward!
  Total:    O(W) por patron, O(N*W) por epoca
```

---

## 3. Resultados Experimentales

### 3.1. XOR y problemas logicos

El paper demuestra que backpropagation resuelve exitosamente el problema XOR, que era imposible para el Perceptron:

```text
Despues del entrenamiento:

  Input   Target   Output
  (0,0)     0      0.01   ✓
  (0,1)     1      0.99   ✓
  (1,0)     1      0.99   ✓
  (1,1)     0      0.01   ✓

La red aprendio automaticamente representaciones internas:
  h1 ≈ OR(x1, x2)
  h2 ≈ AND(x1, x2)
  output ≈ h1 AND NOT h2 = XOR
```

### 3.2. Codificacion de identidad (Encoder problem)

Uno de los experimentos mas elegantes del paper:

```text
ENCODER 8-3-8:
  8 inputs → 3 neuronas ocultas → 8 outputs
  Tarea: reproducir la entrada en la salida (autoencoder)

  Input/Output (one-hot encoding):
  10000000 → [???] → 10000000
  01000000 → [???] → 01000000
  00100000 → [???] → 00100000
  ...
  00000001 → [???] → 00000001

  La capa oculta tiene solo 3 neuronas para representar 8 patrones
  → Debe "inventar" un codigo binario de 3 bits

  Representaciones aprendidas en la capa oculta:
  Patron 1: [0, 0, 0]
  Patron 2: [0, 0, 1]
  Patron 3: [0, 1, 0]
  Patron 4: [0, 1, 1]
  Patron 5: [1, 0, 0]
  Patron 6: [1, 0, 1]
  Patron 7: [1, 1, 0]
  Patron 8: [1, 1, 1]

  → La red descubrio INDEPENDIENTEMENTE la codificacion binaria!
  → Esto demuestra que backpropagation puede aprender
    representaciones internas utiles y significativas.
```

### 3.3. Simetria y prediccion de familia (Family Trees)

```text
PROBLEMA DE RELACIONES FAMILIARES:
  Input: (persona1, relacion)
  Output: persona2

  Ejemplo: (Colin, padre) → James
           (Victoria, madre) → Colin

  Red: input → hidden1 → hidden2 → output

  RESULTADO: Las neuronas ocultas aprendieron features
  distribuidas que codifican:
  - Generacion (abuelo, padre, hijo)
  - Rama familiar (izquierda, derecha)
  - Genero (masculino, femenino)
  - Nacionalidad (inglesa, italiana)

  Estas features NO fueron programadas — emergieron
  del proceso de aprendizaje.
```

### 3.4. Negacion y completacion de patrones

El paper muestra que la red puede aprender transformaciones complejas como la negacion (invertir bits) y completar patrones parciales, demostrando la capacidad de generalizacion de las representaciones internas.

### 3.5. Reconocimiento de vocales y consonantes

```text
TAREA: Clasificar letras como vocales o consonantes
basandose en representaciones fonologicas distribuidas.

  Input: representacion fonetica de una letra
  Output: vocal (1) o consonante (0)

  La red aprendio a GENERALIZAR a letras no vistas
  durante el entrenamiento, demostrando que las
  representaciones internas capturan la estructura
  subyacente del problema.
```

---

## 4. Aspectos Teoricos Clave

### 4.1. Por que funciona: la chain rule del calculo

```text
Backpropagation es simplemente una aplicacion EFICIENTE
de la regla de la cadena (chain rule) del calculo:

  dE/dw = (dE/dy) * (dy/dnet) * (dnet/dw)
           ^^^^^^   ^^^^^^^^^   ^^^^^^^^^
           como E    derivada    como net
           cambia    de la       cambia
           con y     activacion  con w

Para capas profundas, las cadenas se extienden:

  dE/dw_capa1 = (dE/dy_L) * (dy_L/dnet_L) * (dnet_L/dy_{L-1})
                * (dy_{L-1}/dnet_{L-1}) * ... * (dnet_1/dw_capa1)

  → El gradiente "fluye" desde la salida hacia la entrada
    a traves de multiplicaciones sucesivas.
```

### 4.2. Requisito fundamental: funciones de activacion diferenciables

```text
El Perceptron usaba una funcion escalon (step function):

  Step function:          Sigmoide (backprop):
  y                       y
  1│      ┌──────         1│           ________
   │      │                │        ╱
  0│──────┘                │     ╱
   └──────────→ x         0│___╱
                            └──────────→ x
  Derivada: 0 en todo      Derivada: continua
  punto excepto 0           y bien definida
  (no sirve para gradiente) (permite gradiente)
```

### 4.3. Minimos locales vs globales

Los autores reconocen que gradient descent puede quedar atrapado en **minimos locales**:

```text
  E (error)
  │ ╲        ╱╲        ╱╲
  │  ╲      ╱  ╲      ╱  ╲
  │   ╲    ╱    ╲    ╱    ╲
  │    ╲  ╱      ╲  ╱      ╲
  │     ╲╱  A     ╲╱  B     ╲
  └────────────────────────────→ w
         ↑           ↑
      minimo       minimo
      local        global

  Gradient descent puede converger a A en vez de B.

  Mitigaciones propuestas en el paper:
  - Momentum (ayuda a escapar minimos locales poco profundos)
  - Multiples inicializaciones aleatorias
  - Stochastic gradient descent (ruido inherente)
```

Sin embargo, los autores argumentan (y la practica posterior confirmo) que en redes con muchos parametros, los minimos locales tienden a ser de calidad similar al global.

### 4.4. Batch vs Online Learning

```text
BATCH (por epocas):
  1. Acumular gradientes para TODOS los patrones
  2. Actualizar pesos UNA vez por epoca
  + Gradiente exacto
  - Lento, requiere mucha memoria

ONLINE (patron por patron):
  1. Presentar UN patron
  2. Actualizar pesos inmediatamente
  + Mas rapido en la practica
  + El ruido ayuda a escapar minimos locales
  - Gradiente ruidoso (aproximacion)

El paper propone AMBOS modos.
El modo online se convirtio en la base del SGD moderno.
```

---

## 5. Impacto Historico y Legado

### 5.1. Linea temporal

```text
1958  Perceptron (Rosenblatt) — solo una capa, aprendizaje limitado
1969  Perceptrons (Minsky & Papert) — demuestran limitaciones → "AI Winter"
1974  Werbos describe backprop en su tesis doctoral (poca difusion)
1982  Hopfield reaviva interes en redes neuronales
1985  Parker redescubre backprop
1986  ★ RUMELHART, HINTON & WILLIAMS publican en Nature ★
      → Demuestran que FUNCIONA y que aprende representaciones
      → Revive completamente el campo de redes neuronales
1989  LeCun aplica backprop a reconocimiento de digitos (LeNet)
1990s Backprop se vuelve algoritmo estandar, pero redes profundas
      sufren del "vanishing gradient problem"
2006  Hinton propone deep belief networks (pretraining)
2012  AlexNet (entrenada con backprop) gana ImageNet → era del Deep Learning
2017  Transformers (Vaswani et al.) — entrenados con backprop
2024  Hinton recibe Premio Nobel de Fisica por fundamentos de redes neuronales
```

### 5.2. Por que ESTE paper y no los anteriores

```text
Werbos (1974): Describia el algoritmo pero en contexto de
               sistemas de control, en una tesis doctoral
               poco difundida. No demostro aprendizaje de
               representaciones.

Parker (1985): Redescubrio la misma matematica pero sin
               los experimentos que mostraban que las redes
               realmente aprendian representaciones utiles.

Rumelhart, Hinton & Williams (1986):
  ✓ Formalizacion clara y elegante del algoritmo
  ✓ Publicacion en NATURE (maxima visibilidad cientifica)
  ✓ Demostracion EMPIRICA de que las redes aprenden
    representaciones internas significativas
  ✓ Multiples experimentos convincentes (XOR, encoder,
    family trees, etc.)
  ✓ Parte del influyente libro "Parallel Distributed
    Processing" (PDP) que fue un bestseller academico
```

### 5.3. Legado directo en tecnicas modernas

```text
TECNICA MODERNA                  RAIZ EN ESTE PAPER
──────────────────────────────────────────────────────
SGD (Stochastic Gradient         Online backpropagation
  Descent)
Adam, RMSprop, etc.              Extensiones del momentum
                                   propuesto aqui
Autoencoders                     Encoder experiment (8-3-8)
Word Embeddings (Word2Vec)       Idea de representaciones
                                   distribuidas aprendidas
Transfer Learning                Representaciones internas
                                   reutilizables
Deep Learning completo           Todo se entrena con
                                   backpropagation
Automatic Differentiation        Formalizacion de la chain
                                   rule eficiente
PyTorch autograd / TensorFlow    Implementaciones modernas
  GradientTape                     de backpropagation
```

### 5.4. Limitaciones que se descubrieron despues

```text
1. VANISHING GRADIENT PROBLEM (Hochreiter, 1991):
   En redes muy profundas, los gradientes se hacen
   exponencialmente pequenos al propagarse hacia atras.

   δ_capa1 = f'(...) * w * f'(...) * w * ... * f'(...) * w * δ_salida
                                                          ↑
   Si |f'*w| < 1 para cada capa, el producto → 0

   Soluciones posteriores:
   - ReLU en vez de sigmoide (Nair & Hinton, 2010)
   - Batch Normalization (Ioffe & Szegedy, 2015)
   - Residual connections (He et al., 2015)
   - Mejores inicializaciones (Xavier, He init)

2. COSTO COMPUTACIONAL:
   En 1986, los computadores eran demasiado lentos para
   redes grandes. El Deep Learning tuvo que esperar a las GPUs.

3. DATOS INSUFICIENTES:
   Los datasets de los 80s y 90s eran pequenos.
   Backprop necesitaba mucha data para generalizar bien.
   → Resuelto con Internet, ImageNet, etc.
```

---

## 6. Resumen en Una Pagina

```text
PROBLEMA:  Antes de 1986, no existia un metodo practico para
           entrenar redes neuronales con capas ocultas. El
           Perceptron (1 capa) no podia resolver problemas
           no-lineales como XOR.

SOLUCION:  Backpropagation — un algoritmo que calcula el gradiente
           del error respecto a CADA peso de la red, propagando
           señales de error desde la salida hacia la entrada
           usando la regla de la cadena del calculo.

COMO FUNCIONA:
  1. Forward pass: calcular salidas capa por capa
  2. Calcular error en la salida: E = (1/2) Σ(t-y)²
  3. Backward pass: propagar deltas hacia atras
     - Salida:  δ = (t-y) * f'(net)
     - Ocultas: δ = f'(net) * Σ(δ_siguiente * w)
  4. Actualizar pesos: Δw = η * δ * input + α * Δw_anterior

REQUISITO CLAVE:
  Funciones de activacion DIFERENCIABLES (sigmoide en vez de step)

RESULTADOS DEMOSTRADOS:
  - Resuelve XOR y otros problemas no-lineales
  - El encoder 8-3-8 descubre codificacion binaria solo
  - Las redes aprenden representaciones internas significativas
    (generacion, genero, etc. en el problema de family trees)
  - Generaliza a patrones no vistos

IMPACTO:
  - Paper mas influyente en la historia del Deep Learning
  - >90,000 citas
  - Hizo viable el entrenamiento de redes multicapa
  - Base de TODO el Deep Learning moderno
  - Cada vez que se entrena una red con PyTorch, TensorFlow,
    JAX, etc., se esta usando backpropagation

AUTORES:
  Rumelhart (psicologia cognitiva), Hinton (CS, Premio Nobel 2024),
  Williams (CS) — Publicado en Nature, 1986

LEGADO:
  Sin este paper, no existirian CNNs, RNNs, Transformers,
  GPT, BERT, AlphaGo, ni ninguna de las tecnologias de IA
  que hoy usamos. Es, literalmente, el algoritmo que hace
  posible la inteligencia artificial moderna.
```
