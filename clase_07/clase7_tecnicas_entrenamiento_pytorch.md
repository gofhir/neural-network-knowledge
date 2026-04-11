# Clase 7 — Tecnicas para Redes Profundas: ReLU, Dropout, Normalizaciones y PyTorch

Diplomado IA: Aprendizaje Profundo I — Profesor: Alain Raymond

---

## Tabla de Contenidos

1. [ReLU: Funcion de Activacion](#1-relu-funcion-de-activacion)
2. [Dropout: Regularizacion](#2-dropout-regularizacion)
3. [Normalizaciones: BatchNorm y LayerNorm](#3-normalizaciones-batchnorm-y-layernorm)
4. [Guia Practica: Entrenamiento de DNNs](#4-guia-practica-entrenamiento-de-dnns)
5. [Introduccion a PyTorch](#5-introduccion-a-pytorch)
6. [Laboratorio 7: PyTorch en Practica](#6-laboratorio-7-pytorch-en-practica)

---

## 1. ReLU: Funcion de Activacion

### 1.1. El problema con Sigmoide y Tanh

Las funciones de activacion tradicionales (Sigmoide, Tanh) presentan un problema critico: **saturacion**.

```text
Sigmoide: σ(x) = 1 / (1 + e^(-x))

Problema: cuando x es muy grande o muy negativo,
la derivada se acerca a 0 → gradientes "muertos"

         1 ─────────────────────*
                              /
       0.5 ─ ─ ─ ─ ─ ─ ─ ─/─ ─
                          /
         0 ──*───────────
            -6  -4  -2  0  2  4  6

  Zonas marcadas con * = SATURACION (gradiente ≈ 0)
```

**Consecuencia**: En redes profundas, durante backpropagation los gradientes se multiplican capa por capa. Si cada capa aporta gradientes cercanos a 0, el gradiente total se desvanece exponencialmente. Esto se conoce como el **vanishing gradient problem** y hace que las capas iniciales de la red practicamente dejen de aprender.

### 1.2. ReLU: Rectified Linear Unit

ReLU resuelve el problema de saturacion con una funcion extremadamente simple:

```text
ReLU(x) = max(0, x)

         y
         │      /
         │     /
         │    /
         │   /
  ───────┼──/────── x
         │
         │

  Para x > 0: gradiente = 1 (nunca se satura)
  Para x < 0: gradiente = 0 (neurona "muerta")
```

**Ventajas de ReLU**:
- **Computacionalmente eficiente**: solo requiere una comparacion con 0
- **No se satura** para valores positivos → gradientes fluyen sin problemas
- **Promueve sparsity**: muchas neuronas se "apagan" (salida = 0), lo que actua como una forma de regularizacion implicita

**Desventaja — Dying ReLU**: Si una neurona siempre recibe entradas negativas, su gradiente sera siempre 0 y nunca se actualizara. Queda permanentemente "muerta".

### 1.3. Variantes de ReLU

Para mitigar el problema de "dying ReLU", se han propuesto variantes:

| Funcion | Formula | Caracteristica |
|---|---|---|
| **Leaky ReLU** | `max(0.01x, x)` | Permite un pequeno gradiente para x < 0 |
| **ELU** | `x si x>0, α(e^x - 1) si x≤0` | Salidas negativas suaves, media cercana a 0 |
| **GELU** | `x * Φ(x)` donde Φ es la CDF gaussiana | Usada en Transformers (BERT, GPT) |
| **Swish/SiLU** | `x * σ(x)` | Smooth, no-monotonica, usada en EfficientNet |
| **Mish** | `x * tanh(softplus(x))` | Similar a Swish pero mas suave |

En la practica, **ReLU sigue siendo la mas usada** como punto de partida. Las variantes se eligen segun la arquitectura y el problema.

### 1.4. ReLU en PyTorch

```python
import torch
import torch.nn as nn

# Como modulo (dentro de un modelo)
relu = nn.ReLU()

# Como funcion (en el forward pass)
import torch.nn.functional as F
output = F.relu(input)

# Ejemplo
tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(relu(tensor))  # tensor([0., 0., 0., 1., 2.])
```

> **Referencia**: [PyTorch nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)

---

## 2. Dropout: Regularizacion

### 2.1. Primero: que problema resuelve Dropout?

Cuando una red neuronal tiene muchos parametros y pocos datos, tiende a **memorizar** los datos de entrenamiento en vez de aprender patrones generales. A esto se le llama **overfitting** (sobreajuste).

```text
Ejemplo intuitivo:

  Imagina que estudias para una prueba SOLO memorizando
  las respuestas de pruebas anteriores, palabra por palabra.

  → En pruebas viejas sacas 100% (entrenamiento perfecto)
  → En la prueba nueva sacas 30% (no generalizas)

  Eso es overfitting. La red "memorizo" en vez de "entender".
```

**Regularizacion** es cualquier tecnica que le dificulta a la red memorizar, forzandola a aprender patrones mas generales. Dropout es una de las tecnicas de regularizacion mas exitosas.

### 2.2. Que es Dropout concretamente

Dropout agrega un **paso extra** dentro de la red durante el entrenamiento:

> En cada iteracion de entrenamiento, cada neurona tiene una probabilidad **p** de ser "apagada" (su valor se pone en 0).

Veamoslo con un ejemplo concreto. Supongamos una capa con 6 neuronas y p = 0.5:

```text
ITERACION 1:                    ITERACION 2:
Valores de la capa:             Valores de la capa:
[3.2, 1.5, 0.8, 2.1, 0.3, 1.7] [3.2, 1.5, 0.8, 2.1, 0.3, 1.7]

Se lanzan 6 "monedas" (p=0.5):  Se lanzan 6 "monedas" otra vez:
[si,  no,  si,  si,  no,  si ]  [no,  si,  no,  si,  si,  no ]

Resultado despues de Dropout:    Resultado despues de Dropout:
[3.2, 0.0, 0.8, 2.1, 0.0, 1.7] [0.0, 1.5, 0.0, 2.1, 0.3, 0.0]
      ^^^              ^^^       ^^^        ^^^              ^^^
      apagadas                   apagadas

→ Cada iteracion "apaga" neuronas DISTINTAS al azar
```

Visualmente en la red completa:

```text
Red normal (todas las conexiones):

  Entrada     Capa oculta     Salida
    O─────────O
    O─────────O─────────O
    O─────────O
    O─────────O─────────O
    O─────────O

  Todas las neuronas participan siempre.

Con Dropout (iteracion 1, p=0.5):

    O─────────O
    O         X (apagada, vale 0)
    O─────────O─────────O
    O         X (apagada, vale 0)
    O─────────O─────────O

  Solo ALGUNAS neuronas participan, las demas son ignoradas.
  Es como si entrenaramos una red MAS PEQUENA.

Con Dropout (iteracion 2, p=0.5):

    O         X (apagada)
    O─────────O─────────O
    O─────────O
    O─────────O─────────O
    O         X (apagada)

  Ahora se apagaron OTRAS neuronas distintas.
  Es como si entrenaramos OTRA red mas pequena, diferente.
```

### 2.3. Por que esto funciona? Dos formas de entenderlo

**Forma 1: El "equipo de trabajo" (la mas intuitiva)**

```text
Imagina un equipo de 10 personas haciendo un proyecto.

SIN Dropout (sin regularizacion):
  → Pedro se vuelve "el experto" y hace todo el trabajo critico
  → El resto se acomoda y depende de Pedro
  → Si Pedro falta un dia, el equipo no funciona
  → El equipo "memorizo" que Pedro resuelve todo

CON Dropout:
  → Cada dia, al azar, la mitad del equipo no viene
  → Un dia falta Pedro, asi que Maria tiene que aprender
  → Otro dia falta Maria, asi que Juan se esfuerza
  → TODOS terminan aprendiendo a hacer el trabajo
  → Si cualquiera falta, el equipo sigue funcionando

Resultado: El equipo es ROBUSTO.
  Cada persona aprendio features UTILES por si misma,
  no dependiendo de una sola persona.
```

Esto es exactamente lo que pasa en la red:
- **Sin Dropout**: Algunas neuronas se vuelven "super-especializadas" y las demas se vuelven parasitarias. La red memoriza patrones muy especificos.
- **Con Dropout**: Como cualquier neurona puede desaparecer, TODAS tienen que aprender a ser utiles independientemente. La red aprende patrones mas generales y robustos.

**Forma 2: Muchas redes en una (Ensambles)**

```text
Si tuvieramos dinero infinito, podriamos entrenar 1000 redes
distintas y promediar sus predicciones. Esto se llama "ensamble"
y casi siempre mejora los resultados.

  Red 1: "Creo que es un gato"  ──┐
  Red 2: "Creo que es un gato"  ──┤
  Red 3: "Creo que es un perro" ──┼──→ Voto: "Es un gato" (2 de 3)
                                        (prediccion mas confiable)

El problema: entrenar 1000 redes es carisimo.

Dropout lo hace GRATIS:
  - Cada iteracion, al apagar neuronas distintas,
    estamos entrenando una "sub-red" diferente
  - Con 10 neuronas y p=0.5, hay 2^10 = 1024
    posibles sub-redes!
  - Al final, es COMO SI hubieramos entrenado
    miles de redes y promediado sus resultados
```

### 2.4. Y cuando quiero USAR el modelo (inferencia)?

Aqui viene una pregunta natural: si entrene el modelo apagando neuronas, tengo que seguir apagandolas cuando hago predicciones?

**NO.** En inferencia usamos TODAS las neuronas. Pero hay un detalle matematico:

```text
PROBLEMA:

  Durante entrenamiento con p=0.5:
    Solo la MITAD de las neuronas estan activas.
    La capa siguiente se acostumbro a recibir
    "la mitad" de la informacion.

  En inferencia, TODAS las neuronas estan activas:
    Ahora llega EL DOBLE de informacion de lo esperado!
    → Las predicciones se descontrolan.

SOLUCION: Escalar la salida

  Si durante entrenamiento solo pasaba el 50% de la info,
  en inferencia multiplicamos todo por 0.5 para que
  la CANTIDAD TOTAL de informacion sea la misma.
```

**Ejemplo numerico**:

```text
Capa con 4 neuronas, p = 0.5
Valores originales: [2.0, 4.0, 1.0, 3.0]

ENTRENAMIENTO (una iteracion):
  Mascara aleatoria:   [1,   0,   1,   0  ]
  Resultado:           [2.0, 0.0, 1.0, 0.0]
  Suma = 3.0

  En PROMEDIO, la suma esperada = 0.5 * (2+4+1+3) = 5.0

INFERENCIA (todas activas):
  Sin escalar:         [2.0, 4.0, 1.0, 3.0]
  Suma = 10.0  ← EL DOBLE de lo que esperaba la red!

  Con escalar (* 0.5): [1.0, 2.0, 0.5, 1.5]
  Suma = 5.0   ← Ahora la magnitud es la que la red espera
```

> **PyTorch simplifica esto**: Usa "inverted dropout". En vez de escalar en inferencia, escala durante entrenamiento (divide por (1-p)). Asi en inferencia no hay que hacer nada. Es equivalente matematicamente pero mas simple de implementar.

```text
Inverted Dropout en PyTorch (lo que realmente pasa):

ENTRENAMIENTO:
  Valores:           [2.0, 4.0, 1.0, 3.0]
  Mascara (p=0.5):   [1,   0,   1,   0  ]
  Aplicar mascara:   [2.0, 0.0, 1.0, 0.0]
  Dividir por (1-p): [4.0, 0.0, 2.0, 0.0]  ← escala AQUI
                           ÷0.5

INFERENCIA:
  Valores:           [2.0, 4.0, 1.0, 3.0]  ← no se toca nada
```

### 2.5. Dropout en PyTorch

```python
import torch
import torch.nn as nn

# --- Ejemplo basico ---
dropout = nn.Dropout(p=0.5)

tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# En modo entrenamiento: apaga neuronas al azar
dropout.train()
print(dropout(tensor))  
# Posible salida: tensor([2., 0., 6., 0., 10.])
#                        ↑       ↑       ↑
#                    1/0.5=2  3/0.5=6  5/0.5=10  (inverted dropout)
#                        0.     0.  ← neuronas apagadas

# En modo evaluacion: no hace nada
dropout.eval()
print(dropout(tensor))  # tensor([1., 2., 3., 4., 5.]) ← intacto
```

```python
# --- Dentro de un modelo real ---
class MiRed(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Apaga 50% de neuronas
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)      # Capa lineal
        x = self.relu(x)     # Activacion
        x = self.dropout(x)  # Dropout (solo activo en .train())
        x = self.fc2(x)      # Capa de salida
        return x

model = MiRed()

# CRITICO: hay que cambiar el modo del modelo
model.train()  # Para entrenar → Dropout ACTIVO
model.eval()   # Para predecir → Dropout DESACTIVADO
```

**Donde poner Dropout en la red?**

```text
Capa Lineal → ReLU → Dropout → Capa Lineal → ReLU → Dropout → Salida
                      ^^^^^^                          ^^^^^^
                   Se pone DESPUES de la activacion,
                   ENTRE capas. Nunca en la ultima capa.
```

**Valores tipicos de p**:

| Valor de p | Donde se usa | Que tan agresivo es |
|---|---|---|
| **p = 0.1** | Transformers, redes ya regularizadas | Sutil |
| **p = 0.2 - 0.3** | Capas convolucionales | Moderado |
| **p = 0.5** | Capas fully-connected (valor clasico) | Agresivo |

> **Referencia**: [PyTorch nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) |
> Paper original: Srivastava et al. (2014). *Dropout: A simple way to prevent neural networks from overfitting*. JMLR.

---

## 3. Normalizaciones: BatchNorm y LayerNorm

### 3.1. Primero: que problema resuelven las normalizaciones?

Imagina que estas aprendiendo a cocinar siguiendo una receta. Pero cada vez que miras la receta, las unidades cambian:

```text
Intento 1: "Agrega 200g de harina"        → OK, lo haces bien
Intento 2: "Agrega 0.44 libras de harina"  → Hmm, tengo que convertir...
Intento 3: "Agrega 7.05 onzas de harina"   → Esto es confuso...
Intento 4: "Agrega 3.17 tazas de harina"   → Ya no se cuanto es!

  → Cada vez la MISMA cantidad, pero en ESCALAS distintas.
  → Pierdes tiempo convirtiendo en vez de aprendiendo a cocinar.
```

Exactamente eso le pasa a una red neuronal profunda. Cada capa recibe datos de la capa anterior, pero a medida que los pesos se actualizan durante el entrenamiento, la **escala y distribucion** de esos datos cambia constantemente:

```text
Red profunda, iteracion 1:
  Capa 1 produce: [0.5, 1.2, 0.8]     ← valores "pequenos"
  Capa 2 se adapta a recibir valores entre 0 y 1.5

Red profunda, iteracion 100:
  Capa 1 produce: [15.3, 42.7, 28.1]  ← ahora valores "grandes"!
  Capa 2 esta confundida: esperaba valores pequenos,
  ahora recibe valores 30x mas grandes!

  → La capa 2 tiene que RE-APRENDER a interpretar sus entradas.
  → En vez de aprender patrones utiles, gasta tiempo adaptandose
    a que la escala cambia todo el rato.
```

A esto se le llama **Internal Covariate Shift** (Ioffe & Szegedy, 2015): las distribuciones internas de la red cambian durante el entrenamiento, haciendo que cada capa tenga que "perseguir" una entrada que se mueve.

**La solucion**: despues de cada capa, normalizar las activaciones para que SIEMPRE tengan media ≈ 0 y varianza ≈ 1. Asi cada capa recibe datos en una escala predecible y puede enfocarse en aprender patrones utiles.

### 3.2. Que es normalizar concretamente

Normalizar es transformar datos para que tengan **media = 0** y **desviacion estandar = 1**. Es la tipica Z-score de estadistica:

```text
         x - μ
x_norm = ─────
           σ

Donde:
  μ = media de los datos
  σ = desviacion estandar de los datos
```

Veamoslo con un ejemplo concreto. Supongamos que una capa produce estos valores para 4 muestras:

```text
Valores originales:  [2.0, 8.0, 4.0, 6.0]

Paso 1: Calcular media
  μ = (2 + 8 + 4 + 6) / 4 = 5.0

Paso 2: Calcular desviacion estandar
  σ = sqrt(((2-5)² + (8-5)² + (4-5)² + (6-5)²) / 4)
    = sqrt((9 + 9 + 1 + 1) / 4)
    = sqrt(5)
    ≈ 2.24

Paso 3: Normalizar cada valor
  (2 - 5) / 2.24 = -1.34
  (8 - 5) / 2.24 = +1.34
  (4 - 5) / 2.24 = -0.45
  (6 - 5) / 2.24 = +0.45

Resultado:           [-1.34, +1.34, -0.45, +0.45]
                      Media ≈ 0, Desviacion estandar ≈ 1
```

Ahora la capa siguiente siempre recibe valores "bien portados" centrados en 0. Ya no importa si la capa anterior producia valores entre 0 y 1, o entre -1000 y 1000. Despues de normalizar, todo queda en la misma escala.

### 3.3. El truco de gamma y beta: no perder expresividad

Normalizar resuelve el problema de estabilidad. Pero crea un problema nuevo: **forzar media=0 y varianza=1 puede ser demasiado restrictivo**.

Veamoslo con una analogia:

```text
Imagina que eres profesor y tus alumnos escriben ensayos
de distintos largos: 200, 500, 1500, 3000 palabras.

PROBLEMA ORIGINAL (sin normalizar):
  Los largos son tan distintos que no puedes compararlos.
  → Decides estandarizar: "todos escriban exactamente 500 palabras"

PROBLEMA NUEVO (normalizar a media=0, var=1):
  Ahora TODOS los ensayos tienen el mismo largo.
  Pero algunos temas NECESITAN mas espacio!
  Un ensayo sobre "mi color favorito" → 500 palabras sobra
  Un ensayo sobre "la historia de Chile" → 500 palabras no alcanza

SOLUCION:
  Primero estandarizas a 500 (para tener una base comun).
  Luego dejas que cada alumno AJUSTE el largo segun lo que necesite:
    largo_final = factor_escala * 500 + ajuste
  
  Asi eliminas el caos original PERO no pierdes flexibilidad.
```

Eso es exactamente lo que hacen γ (gamma) y β (beta). Son dos numeros **que la red aprende durante el entrenamiento** (igual que aprende los pesos):

```text
Paso 1 - Normalizar (siempre igual):
  x_norm = (x - μ) / σ     → media=0, varianza=1

Paso 2 - Re-escalar (la red decide):
  y = γ * x_norm + β

  γ (gamma) = "cuanto estirar o comprimir" (escala)
  β (beta)  = "cuanto desplazar" (centro)
```

Ejemplo numerico para ver por que esto importa. Supongamos que normalizamos y obtenemos:

```text
x_norm = [-1.34, +1.34, -0.45, +0.45]    (media=0, var=1)
```

Ahora la red puede hacer cosas distintas segun lo que APRENDA para γ y β:

```text
Caso 1: La red aprende γ=1, β=0 (identidad)
  y = 1 * [-1.34, +1.34, -0.45, +0.45] + 0
    = [-1.34, +1.34, -0.45, +0.45]
  → "La normalizacion estandar esta bien, no cambio nada"

Caso 2: La red aprende γ=2, β=3
  y = 2 * [-1.34, +1.34, -0.45, +0.45] + 3
    = [0.32,  5.68,  2.10,  3.90]
  → "Necesito valores positivos centrados en 3"
  → Media ≈ 3.0, Varianza ≈ 4.0

Caso 3 (caso extremo): La red aprende γ=σ_original, β=μ_original
  → ¡DESHACE completamente la normalizacion!
  → Recupera los valores originales
  → "Para esta capa, normalizar no ayuda, prefiero los datos como estaban"
```

El punto clave: **γ y β le dan a la red una valvula de escape**.

```text
SIN γ y β:
  "Forzamos media=0 y var=1 SIEMPRE, sin importar si eso es optimo"
  → Puede perjudicar a algunas capas que necesitan otra distribucion

CON γ y β:
  "Empezamos con media=0 y var=1 (estable), pero si la red
   descubre que otra distribucion funciona mejor, puede aprenderla"
  → Lo mejor de ambos mundos: estabilidad + flexibilidad

  Es como poner rueditas a una bicicleta:
  estabilizan al principio, pero no limitan lo que puedes hacer despues.
```

> **Detalle tecnico**: γ y β se inicializan en γ=1 y β=0 (es decir, empiezan sin cambiar la normalizacion). A medida que la red entrena, el optimizador (SGD, Adam, etc.) los ajusta junto con todos los demas pesos. Hay un par de γ y β **por cada feature/canal**.

### 3.4. Batch Normalization: normalizar "en columna"

Ya entendemos QUE es normalizar. Ahora la pregunta es: **sobre que datos calculo la media y la varianza?**

Batch Normalization (Ioffe & Szegedy, 2015) dice: para cada feature, calcula la media y varianza **a traves de todas las muestras del batch**.

```text
Imagina una tabla donde cada fila es una muestra
y cada columna es un feature:

         Feature 1   Feature 2   Feature 3
         ─────────   ─────────   ─────────
Muestra 1:   2.0        10.0        0.5
Muestra 2:   8.0        20.0        1.5
Muestra 3:   4.0        30.0        0.8
Muestra 4:   6.0        40.0        1.2
              ↕           ↕           ↕
           BN calcula  BN calcula  BN calcula
           media y σ   media y σ   media y σ
           de ESTA     de ESTA     de ESTA
           columna     columna     columna

Feature 1: μ=5.0, σ=2.24 → normalizar [2,8,4,6] con estos valores
Feature 2: μ=25.0, σ=11.18 → normalizar [10,20,30,40] con estos valores
Feature 3: μ=1.0, σ=0.37 → normalizar [0.5,1.5,0.8,1.2] con estos valores
```

Ejemplo numerico completo para el Feature 1:

```text
Valores del Feature 1 en el batch: [2.0, 8.0, 4.0, 6.0]

1. Media del batch:  μ = (2+8+4+6)/4 = 5.0
2. Varianza:         σ² = ((2-5)²+(8-5)²+(4-5)²+(6-5)²)/4 = 5.0
3. Normalizar:       x_norm = (x - 5.0) / √(5.0 + ε)
                     = [-1.34, +1.34, -0.45, +0.45]
4. Escalar (γ=1, β=0 inicialmente):
                     y = 1 * x_norm + 0 = [-1.34, +1.34, -0.45, +0.45]
```

Visualmente en la matriz completa:

```text
         Feature 1   Feature 2   Feature 3
         ─────────   ─────────   ─────────
Muestra 1:  -1.34      -1.34      -1.34
Muestra 2:  +1.34      -0.45      +1.34
Muestra 3:  -0.45      +0.45      -0.45
Muestra 4:  +0.45      +1.34      +0.45

Cada COLUMNA tiene media ≈ 0 y varianza ≈ 1
(cada feature esta normalizado independientemente)
```

### 3.5. BatchNorm en inferencia: un detalle importante

En entrenamiento, BatchNorm usa la media y varianza **del batch actual**. Pero en inferencia (cuando usas el modelo para predecir), hay un problema:

```text
PROBLEMA:
  En inferencia, tal vez solo tienes UNA muestra.
  No hay "batch" del cual calcular estadisticas!

  Ademas, quieres que la prediccion sea DETERMINISTA:
  el mismo input siempre debe dar el mismo output.
  Pero si la prediccion depende de QUE OTRAS muestras
  estan en el batch, no es determinista.

SOLUCION:
  Durante entrenamiento, BatchNorm va guardando un
  PROMEDIO MOVIL (running mean/var) de las estadisticas
  de todos los batches que ha visto.

  En inferencia, usa esas estadisticas acumuladas,
  NO las del batch actual.
```

```text
Entrenamiento:
  Batch 1: μ=5.0, σ²=5.0
  Batch 2: μ=4.8, σ²=4.5
  Batch 3: μ=5.2, σ²=5.3
  ...
  Running mean ≈ 5.0   (promedio acumulado)
  Running var  ≈ 5.0   (promedio acumulado)

Inferencia:
  Llega UNA muestra con Feature 1 = 7.0
  x_norm = (7.0 - 5.0) / √(5.0) = 0.89
  → Usa la running mean/var, NO necesita un batch
```

> **Por eso model.eval() es CRITICO**: cambia BatchNorm de usar estadisticas del batch actual a usar las acumuladas. Si olvidas llamar model.eval(), tus predicciones seran ruidosas e inconsistentes.

### 3.6. Limitacion de BatchNorm

```text
Batch grande (128 muestras):
  μ y σ calculadas sobre 128 valores → estadisticas ESTABLES
  → BatchNorm funciona bien

Batch pequeno (4 muestras):
  μ y σ calculadas sobre 4 valores → estadisticas MUY RUIDOSAS
  → Un valor atipico cambia drasticamente la media
  → BatchNorm funciona MAL

  Ejemplo: Feature 1 en un batch de 4
  [2.0, 3.0, 2.5, 100.0]  ← un outlier
  μ = 26.9  ← ¡la media no representa a nadie!
```

Ademas, BatchNorm no funciona bien con **secuencias de largo variable** (como en NLP), porque cada posicion de la secuencia tendria estadisticas diferentes.

### 3.7. Layer Normalization: normalizar "en fila"

Layer Normalization (Ba et al., 2016) resuelve las limitaciones de BatchNorm cambiando la direccion: en vez de normalizar por columna (a traves del batch), normaliza **por fila** (a traves de los features de cada muestra).

```text
         Feature 1   Feature 2   Feature 3
         ─────────   ─────────   ─────────
Muestra 1:   2.0        10.0        0.5    ← LN: μ y σ de ESTA fila
Muestra 2:   8.0        20.0        1.5    ← LN: μ y σ de ESTA fila
Muestra 3:   4.0        30.0        0.8    ← LN: μ y σ de ESTA fila
Muestra 4:   6.0        40.0        1.2    ← LN: μ y σ de ESTA fila
             ←─────────────────────────→
              LN normaliza HORIZONTALMENTE
              cada muestra por separado
```

Ejemplo numerico para la Muestra 1:

```text
Valores de Muestra 1: [2.0, 10.0, 0.5]

1. Media de esta muestra:  μ = (2.0 + 10.0 + 0.5) / 3 = 4.17
2. Varianza:               σ² = ((2-4.17)² + (10-4.17)² + (0.5-4.17)²) / 3
                              = (4.69 + 34.03 + 13.47) / 3 = 17.40
3. Normalizar:             x_norm = (x - 4.17) / √(17.40)
                           = [-0.52, +1.40, -0.88]

Muestra 1 normalizada: [-0.52, +1.40, -0.88]
                        Media ≈ 0, Varianza ≈ 1
```

La diferencia clave con BatchNorm:

```text
BatchNorm:
  "Tomo TODOS los valores del Feature 1 en el batch
   y los normalizo juntos"
  → Necesito multiples muestras (un batch)
  → En inferencia uso estadisticas acumuladas

LayerNorm:
  "Tomo TODOS los features de la Muestra 1
   y los normalizo juntos"
  → Solo necesito UNA muestra (no depende del batch)
  → Calculo es identico en entrenamiento e inferencia
```

### 3.8. Por que LayerNorm es ideal para Transformers

```text
En NLP, cada token es una "muestra" con un vector de features:

  "El gato come"

  "El"   → [0.3, 1.2, -0.5, ..., 0.8]  ← 512 features
  "gato" → [1.1, 0.4,  0.9, ..., -0.2] ← 512 features
  "come" → [0.7, -0.3, 1.4, ..., 0.5]  ← 512 features

LayerNorm normaliza CADA TOKEN por separado:
  → No importa cuantos tokens tenga la secuencia
  → No importa el tamano del batch
  → Cada token se normaliza usando sus propios 512 features
  → Funciona igual con secuencias de largo 3 o largo 1000

BatchNorm necesitaria normalizar el Feature 1 de TODOS
los tokens de TODAS las muestras del batch juntos:
  → Problematico con secuencias de largo variable
  → Estadisticas ruidosas si hay pocas muestras
```

### 3.9. Comparacion BN vs LN

| Aspecto | Batch Normalization | Layer Normalization |
|---|---|---|
| **Normaliza sobre** | Columna (a traves del batch) | Fila (a traves de features) |
| **Depende del batch size** | Si (batches grandes funcionan mejor) | No (funciona con batch_size=1) |
| **Parametros aprendibles** | γ, β por feature | γ, β por feature |
| **Inferencia** | Usa running statistics (acumuladas) | Calcula en el momento |
| **Ideal para** | CNNs, imagenes | Transformers, RNNs, secuencias |
| **model.eval() afecta?** | Si (cambia a running stats) | No (siempre se comporta igual) |

**Resultados empiricos** (del paper original de BatchNorm):

| Modelo | Pasos para 72.2% accuracy | Max accuracy |
|---|---|---|
| Inception (sin BN) | 31.0 x 10^6 | 72.2% |
| BN-Baseline | 13.3 x 10^6 | 72.7% |
| BN-x5 (learning rate x5) | 2.1 x 10^6 | 73.0% |
| BN-x30 (learning rate x30) | 2.7 x 10^6 | 74.8% |

Con BatchNorm, el modelo logra mejor accuracy en **15x menos iteraciones** y ademas permite usar learning rates mucho mas altos sin que el entrenamiento se desestabilice.

### 3.10. Normalizaciones en PyTorch

```python
import torch
import torch.nn as nn

# ─── Batch Normalization ───
# Para datos 1D (fully connected): BatchNorm1d
bn1d = nn.BatchNorm1d(num_features=64)   # 64 features

# Para datos 2D (imagenes): BatchNorm2d
bn2d = nn.BatchNorm2d(num_features=3)    # 3 canales (RGB)

# ─── Layer Normalization ───
ln = nn.LayerNorm(normalized_shape=100)  # normaliza sobre 100 features
```

#### Gamma y beta en PyTorch: el parametro `affine` / `elementwise_affine`

En PyTorch, γ y β se controlan con UN parametro: **`affine`** (en BatchNorm) o **`elementwise_affine`** (en LayerNorm).

```python
# ─── BatchNorm: parametro "affine" ───

# affine=True (POR DEFECTO): incluye γ y β aprendibles
bn = nn.BatchNorm1d(num_features=5, affine=True)
# Es lo mismo que escribir:
bn = nn.BatchNorm1d(5)  # affine=True es el default

# affine=False: solo normaliza, sin γ ni β
bn_sin_gamma_beta = nn.BatchNorm1d(num_features=5, affine=False)
```

```python
# ─── LayerNorm: parametro "elementwise_affine" ───

# elementwise_affine=True (POR DEFECTO): incluye γ y β aprendibles
ln = nn.LayerNorm(normalized_shape=100, elementwise_affine=True)
# Es lo mismo que escribir:
ln = nn.LayerNorm(100)  # elementwise_affine=True es el default

# elementwise_affine=False: solo normaliza, sin γ ni β
ln_sin_gamma_beta = nn.LayerNorm(normalized_shape=100, elementwise_affine=False)
```

Pero, ¿donde estan γ y β? PyTorch los llama **`weight`** (γ) y **`bias`** (β):

```python
bn = nn.BatchNorm1d(5)  # 5 features

# γ y β son parametros del modulo, como los pesos de una capa lineal
print(bn.weight)  # γ: tensor([1., 1., 1., 1., 1.])  ← inicializado en 1
print(bn.bias)    # β: tensor([0., 0., 0., 0., 0.])  ← inicializado en 0
```

```text
¿Por que γ=1 y β=0 al inicio?

  Recordemos la formula:  y = γ * x_norm + β

  Con γ=1, β=0:  y = 1 * x_norm + 0 = x_norm
  → Al principio NO CAMBIA la normalizacion
  → Es como decir: "empecemos con normalizacion pura,
     y si la red necesita otra escala, que la aprenda"

  Con el tiempo, el optimizador (SGD, Adam, etc.) ajusta
  γ y β igual que cualquier otro peso de la red.
```

Veamoslo funcionando con un ejemplo completo:

```python
import torch
import torch.nn as nn

# ─── Ejemplo: ver como γ y β afectan la salida ───
bn = nn.BatchNorm1d(3, momentum=None)  # 3 features

# Datos: batch de 4 muestras, 3 features
x = torch.tensor([
    [2.0, 10.0, 0.5],
    [8.0, 20.0, 1.5],
    [4.0, 30.0, 0.8],
    [6.0, 40.0, 1.2]
])

# --- Paso 1: Con γ=1, β=0 (valores iniciales) ---
print(bn.weight)  # tensor([1., 1., 1., 1., 1.])  γ
print(bn.bias)    # tensor([0., 0., 0., 0., 0.])  β

resultado = bn(x)
print(resultado.mean(dim=0))  # ≈ [0, 0, 0]  ← media ≈ 0
print(resultado.var(dim=0))   # ≈ [1, 1, 1]  ← varianza ≈ 1

# --- Paso 2: Cambiar γ y β manualmente para ver el efecto ---
with torch.no_grad():
    bn.weight.fill_(2.0)   # γ = 2  (estirar)
    bn.bias.fill_(5.0)     # β = 5  (desplazar)

resultado2 = bn(x)
print(resultado2.mean(dim=0))  # ≈ [5, 5, 5]  ← media ≈ β = 5
print(resultado2.var(dim=0))   # ≈ [4, 4, 4]  ← varianza ≈ γ² = 4

# --- Paso 3: Sin γ ni β (affine=False) ---
bn_puro = nn.BatchNorm1d(3, affine=False, momentum=None)
resultado3 = bn_puro(x)
print(resultado3.mean(dim=0))  # ≈ [0, 0, 0]  ← siempre media=0
print(resultado3.var(dim=0))   # ≈ [1, 1, 1]  ← siempre var=1
# Con affine=False, NO TIENE weight ni bias:
# print(bn_puro.weight)  → AttributeError!
```

```text
Resumen: γ y β en PyTorch

  ┌─────────────────────────────────────────────────────────┐
  │  Concepto        │  Nombre en PyTorch  │ Valor inicial  │
  ├─────────────────────────────────────────────────────────┤
  │  γ (gamma)       │  .weight            │  1.0           │
  │  β (beta)        │  .bias              │  0.0           │
  │  ¿Activarlos?    │  affine=True/False  │  True (default)│
  │                  │  elementwise_affine │  True (default)│
  └─────────────────────────────────────────────────────────┘

  En la practica: casi SIEMPRE se dejan activados (el default).
  No necesitas tocarlos manualmente; el optimizador los ajusta
  durante el entrenamiento igual que los pesos de las capas.
```

#### Ejemplos practicos de normalizacion

```python
# ─── Ejemplo practico con BatchNorm1d ───
tensor = 100 * torch.randn((3, 5))  # batch=3, features=5
bn = nn.BatchNorm1d(5, momentum=None)
resultado = bn(tensor)
print(f"Media por feature: {resultado.mean(dim=0)}")  # ≈ [0, 0, 0, 0, 0]
print(f"Var por feature: {resultado.var(dim=0)}")      # ≈ [1, 1, 1, 1, 1]

# ─── Ejemplo practico con BatchNorm2d ───
imagenes = 20 * torch.randn((4, 3, 2, 2))  # batch=4, 3 canales, 2x2
bn2d = nn.BatchNorm2d(3, momentum=None, eps=0.0)
resultado = bn2d(imagenes)
# Media por canal ≈ 0, Varianza por canal ≈ 1

# ─── Ejemplo practico con LayerNorm ───
tensor = 100 * torch.randn((20, 100))
ln = nn.LayerNorm(100, eps=0, elementwise_affine=True)
resultado = ln(tensor)
# Media por muestra ≈ 0, Varianza por muestra ≈ 1
```

**Donde poner normalizacion en la red?**

```text
Opcion clasica (pre-activacion):
  Capa Lineal → BatchNorm → ReLU → Capa Lineal → BatchNorm → ReLU → Salida

En Transformers:
  Attention → LayerNorm → FFN → LayerNorm → ...

NUNCA en la capa de salida (queremos la prediccion sin normalizar).
```

> **Referencia**: [PyTorch nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) | [PyTorch nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) | Paper original: Ioffe & Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML.

---

## 4. Guia Practica: Entrenamiento de DNNs

### 4.1. Los 4 pasos para entrenar una red neuronal

```text
┌─────────────────────┐     ┌─────────────────────┐
│ 1. Determinar el    │ ──→ │ 2. Elegir/Disenar   │
│    problema         │     │    un modelo         │
└─────────────────────┘     └─────────────────────┘
                                      │
                                      ▼
┌─────────────────────┐     ┌─────────────────────┐
│ 4. Evaluar el       │ ←── │ 3. Entrenar el      │
│    modelo           │     │    modelo            │
└─────────────────────┘     └─────────────────────┘
```

### 4.2. Paso 1: Determinar el problema

**Preguntas clave**:
- Que tarea quiero resolver?
- Cuales son mis datos de entrenamiento, validacion y test?

**Tipos de problema y funciones de perdida**:

| Tipo | Descripcion | Funcion de Perdida |
|---|---|---|
| **Regresion** | Predecir valores continuos | Mean Squared Error (MSE) |
| **Clasificacion** | Predecir una de N clases | Cross Entropy |

**Sobre los datos de entrenamiento**:
- Deben poder transformarse a **tensores** (formato que entiende el modelo)
- Se agrupan en **minibatches** para entrenamiento paralelo
- Se debe ver todos los datos antes de repetir → concepto de **epoca**

### 4.3. Paso 2: Elegir o disenar un modelo

Dos opciones:

| Opcion | Ventaja | Desventaja |
|---|---|---|
| **Modelo preentrenado** | Rapido, aprovecha conocimiento previo | Menos flexible |
| **Modelo desde cero** | Totalmente personalizado | Costoso, requiere expertise |

**Estructura de una red profunda**: Es una coleccion de **modulos** ejecutados secuencialmente. Cada modulo transforma una representacion numerica en otra.

**Modulos disponibles**:

| Modulo | Entrada | Salida | Uso |
|---|---|---|---|
| **Linear** | Definida por usuario | Definida por usuario | Capas fully-connected |
| **Sigmoid** | Cualquiera | Misma que entrada | Activacion tradicional |
| **ReLU** | Cualquiera | Misma que entrada | Activacion moderna |
| **Conv2d** | (C_in, H, W) | (C_out, O(H), O(W)) | Imagenes |
| **MaxPool2d** | (C_in, H, W) | (C_in, O(H), O(W)) | Reduccion espacial |

**Ejemplo: MLP en pseudocodigo**:

```python
# x es la entrada de tamano 3
x = Linear1(x)   # x ahora es de tamano 7
x = Sigmoid(x)   # x sigue siendo de tamano 7
x = Linear2(x)   # x ahora es de tamano 5
x = Sigmoid(x)   # x sigue siendo de tamano 5
output = Linear3(x)  # output es de tamano 1
```

**Restriccion**: Las dimensionalidades de salida de un modulo deben coincidir con las de entrada del siguiente.

### 4.4. Paso 3: Entrenar el modelo

El entrenamiento sigue un loop iterativo:

```text
1. Inicializar modelo (pesos aleatorios o preentrenados)
2. Para cada epoca:
   3. Para cada minibatch (x, y) del dataset:
      4. Forward pass: output = model(x)
      5. Calcular perdida: loss = loss_fn(output, y)
      6. Backpropagation: calcular gradientes
      7. Actualizar pesos: optimizer.step()
```

**Pseudocodigo**:

```python
for epoch in range(1, num_epochs + 1):
    for x, y in train_dataset:
        model_output = model(x)
        loss = funcion_perdida(model_output, y)
        grads = backpropagation(loss)
        model = update_model(model, lr, grads)
```

### 4.5. Paso 4: Evaluar el modelo

**Metricas principales**:
- **Perdida (Loss)**: queremos que sea lo mas baja posible
- **Exactitud (Accuracy)**: porcentaje de predicciones correctas (solo para clasificacion)

```python
output = model(input)                   # (batch_size, n_clases)
preds = output.argmax(dim=1)            # indice de la clase con mayor probabilidad
n_correctas = (preds == targets).sum()  # comparar con ground truth
accuracy = n_correctas / len(targets)
```

---

## 5. Introduccion a PyTorch

### 5.1. Por que usar un framework de Deep Learning

- Implementar backpropagation manualmente es dificil y propenso a errores
- Los frameworks proveen implementaciones **eficientes** y optimizadas
- Operaciones en GPU de forma transparente
- Autograd: calculo automatico de gradientes

### 5.2. Frameworks principales

| Framework | Mantenido por |
|---|---|
| **PyTorch** | Meta AI (Facebook) |
| TensorFlow | Google |
| Keras | Comunidad (sobre TensorFlow) |
| JAX | Google |

**En este curso se usa PyTorch.**

### 5.3. Caracteristicas de PyTorch

- Biblioteca de alto nivel para redes neuronales
- Escrito en Python, backend en C++ para rendimiento
- Facil de usar, gran modularidad
- Enfocado en experimentacion rapida
- Soporta CNNs, RNNs y cualquier arquitectura
- Corre en CPU y GPU de forma transparente
- Orientado a objetos (basado en clases)

### 5.4. Modularidad de PyTorch

```text
PyTorch provee modulos para:
├── Capas neuronales (nn.Linear, nn.Conv2d, ...)
├── Funciones de perdida (nn.CrossEntropyLoss, nn.MSELoss, ...)
├── Optimizadores (optim.SGD, optim.Adam, ...)
├── Inicializacion de pesos
├── Funciones de activacion (nn.ReLU, nn.GELU, ...)
├── Regularizacion (nn.Dropout, nn.BatchNorm2d, ...)
└── Se pueden crear modulos custom
```

---

## 6. Laboratorio 7: PyTorch en Practica

### 6.1. Tensores: la unidad fundamental

Un tensor es un arreglo multidimensional, la estructura de datos basica en PyTorch.

```python
import torch

# Crear tensor aleatorio (batch=2, ancho=4, alto=3)
tensor = torch.randn((2, 4, 3)).float()
print(tensor.shape)  # torch.Size([2, 4, 3])

# Desde un arreglo de Python
arreglo = [[1, 2, 3], [4, 5, 6]]
tensor = torch.tensor(arreglo)

# Operaciones
suma = tensor + tensor
producto = tensor * tensor  # elemento a elemento

# Indexacion (igual que NumPy)
tensor_indexado = tensor[0:2, 0]
```

**Dimension batch**: Es la primera dimension del tensor. Permite procesar multiples muestras en paralelo en GPU.

```python
# 10 imagenes RGB de 64x64
batch_imagenes = torch.randn(10, 3, 64, 64)
#                             ↑   ↑  ↑   ↑
#                          batch  C  H   W
```

### 6.2. Dispositivos: CPU vs GPU

```python
tensor = torch.randn((1, 2, 3))        # En CPU por defecto
tensor_gpu = tensor.cuda()              # Copia a GPU
tensor_gpu = tensor.to("cuda")          # Alternativa

# IMPORTANTE: No se pueden operar tensores en dispositivos distintos
# tensor_cpu + tensor_gpu → ERROR
```

### 6.3. Definicion de modelos

Todos los modelos en PyTorch heredan de `nn.Module`:

```python
import torch
import torch.nn as nn

class MiAlexNet(nn.Module):

    def __init__(self):
        super(MiAlexNet, self).__init__()
        # Bloques Convolucionales
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Bloque Clasificador (fully connected)
        self.classifier = nn.Sequential(
            nn.Dropout(),          # Dropout para regularizacion
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 5),    # 5 clases
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)   # Aplanar para el clasificador
        x = self.classifier(x)
        return x

# Crear y verificar
model = MiAlexNet()
entrada_falsa = torch.randn(15, 3, 224, 224)  # 15 imagenes 224x224 RGB
print(model(entrada_falsa).shape)  # torch.Size([15, 5])
```

**Funciones de activacion disponibles**:

```python
from torch.nn import ReLU, Sigmoid, Softmax

# ReLU: max(0, x)
relu = ReLU()

# Sigmoid: 1 / (1 + e^(-x)), rango (0, 1)
sigmoid = Sigmoid()

# Softmax: distribucion de probabilidad sobre clases
softmax = Softmax(dim=1)  # dim=1 para normalizar por fila
```

**Modulos de regularizacion y normalizacion**:

```python
from torch.nn import Dropout, BatchNorm1d, BatchNorm2d, LayerNorm

# Dropout
drop = Dropout(p=0.5)

# BatchNorm para vectores
bn1d = BatchNorm1d(num_features=64)

# BatchNorm para imagenes (normaliza por canal)
bn2d = BatchNorm2d(num_features=3)

# LayerNorm (normaliza por muestra)
ln = LayerNorm(normalized_shape=100)
```

### 6.4. Manejo de datos

**Datasets predefinidos** (torchvision):

```python
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor

# Descargar MNIST con transformacion a tensor
mnist_train = MNIST(root=".", train=True, download=True, transform=ToTensor())
print(len(mnist_train))  # 60000

# Cada elemento es (tensor_imagen, label)
imagen, label = mnist_train[0]
```

**Dataset personalizado**:

```python
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from os import listdir
from os.path import join

class Flowers(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(listdir(root))
        self.data = []
        for i, cls in enumerate(self.classes):
            for img in listdir(join(root, cls)):
                self.data.append((join(root, cls, img), i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Uso
transforms = Compose([Resize((224, 224)), ToTensor()])
dataset = Flowers('flowers_dataset/train', transform=transforms)
```

**DataLoader**: Itera sobre el dataset en minibatches.

```python
from torch.utils.data import DataLoader

train_dl = DataLoader(dataset, batch_size=128, shuffle=True)

for x, target in train_dl:
    print(x.shape)       # (128, 3, 224, 224)
    print(target.shape)  # (128,)
    break
```

### 6.5. Funcion de perdida y optimizador

```python
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

# Funcion de perdida para clasificacion
loss_fn = CrossEntropyLoss()

# Optimizador (vinculado a los parametros del modelo)
optimizer = Adam(model.parameters(), lr=0.001)
```

> **Nota**: `CrossEntropyLoss` en PyTorch ya incluye `LogSoftmax` internamente. **No** pongas `Softmax` en la ultima capa de tu modelo si usas `CrossEntropyLoss`.

### 6.6. Loop de entrenamiento completo

```python
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

model = MiAlexNet()
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Activa Dropout y BatchNorm en modo entrenamiento
    total_loss = 0.0
    total_correctas = 0
    total_muestras = 0

    for x, target in train_dl:
        # 1. Limpiar gradientes
        optimizer.zero_grad()

        # 2. Forward pass
        output = model(x)

        # 3. Calcular perdida
        loss = loss_fn(output, target)

        # 4. Backpropagation
        loss.backward()

        # 5. Actualizar pesos
        optimizer.step()

        # Metricas
        total_loss += loss.item()
        preds = output.argmax(dim=1)
        total_correctas += (preds == target).sum().item()
        total_muestras += len(target)

    accuracy = total_correctas / total_muestras
    print(f"Epoca {epoch+1}: Loss={total_loss:.4f}, Accuracy={accuracy:.4f}")
```

**Los 5 pasos clave en cada iteracion**:

```text
optimizer.zero_grad()          # 1. Limpiar gradientes acumulados
output = model(x)              # 2. Forward pass
loss = loss_fn(output, target) # 3. Calcular perdida
loss.backward()                # 4. Backpropagation (calcula gradientes)
optimizer.step()               # 5. Actualizar pesos
```

### 6.7. Evaluacion en conjunto de test

```python
model.eval()  # IMPORTANTE: desactiva Dropout y pone BatchNorm en modo inferencia

total_correctas = 0
total_muestras = 0

with torch.no_grad():  # No calcular gradientes (ahorra memoria y tiempo)
    for x, target in test_dl:
        output = model(x)
        preds = output.argmax(dim=1)
        total_correctas += (preds == target).sum().item()
        total_muestras += len(target)

accuracy = total_correctas / total_muestras
print(f"Test Accuracy: {accuracy:.4f}")
```

### 6.8. Guardar y cargar modelos

```python
# Guardar pesos
torch.save(model.state_dict(), "pesos_modelo.pth")

# Cargar pesos
modelo_nuevo = MiAlexNet()                        # Crear modelo vacio
pesos = torch.load("pesos_modelo.pth")            # Cargar pesos del disco
modelo_nuevo.load_state_dict(pesos)               # Asignar pesos al modelo
```

### 6.9. Transfer Learning con AlexNet preentrenado

```python
from torchvision.models import alexnet

# Cargar modelo preentrenado en ImageNet
model = alexnet(pretrained=True)

# Congelar los features (no entrenar las convoluiones)
model.features.requires_grad_(False)

# Reemplazar el clasificador para nuestras 5 clases
model.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 5),  # 5 clases en vez de 1000 de ImageNet
)
```

---

## 7. Resumen y Conexiones

```text
Problema: Entrenar redes profundas es dificil
│
├── Gradientes se desvanecen
│   └── Solucion: ReLU (y variantes)
│
├── Overfitting
│   └── Solucion: Dropout (regularizacion)
│
├── Activaciones inestables / entrenamiento lento
│   └── Solucion: BatchNorm / LayerNorm
│
└── Implementacion compleja
    └── Solucion: PyTorch (autograd, modulos, GPU)
```

| Tecnica | Tipo | Donde se aplica | Efecto principal |
|---|---|---|---|
| **ReLU** | Activacion | Entre capas | Evita vanishing gradients |
| **Dropout** | Regularizacion | Despues de capas densas | Previene overfitting |
| **BatchNorm** | Normalizacion | Despues de capas (antes de activacion) | Estabiliza y acelera entrenamiento |
| **LayerNorm** | Normalizacion | Despues de capas | Ideal para secuencias y Transformers |

---

## 8. Referencias

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A simple way to prevent neural networks from overfitting*. JMLR, 15(56), 1929-1958.
- Ioffe, S., & Szegedy, C. (2015). *Batch normalization: Accelerating deep network training by reducing internal covariate shift*. ICML.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer normalization*. arXiv:1607.06450.
- Nair, V., & Hinton, G. E. (2010). *Rectified linear units improve restricted Boltzmann machines*. ICML.
- [Documentacion oficial de PyTorch](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
