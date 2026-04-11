# Clase 8 — Funciones de Perdida, Regularizacion y Tareas Auxiliares

Profesor: Carlos Aspillaga

---

## Indice

1. [Funciones de Perdida](#1-funciones-de-perdida)
   - Contexto y definicion
   - MSE (Mean Squared Error)
   - Cross-Entropy + Softmax
   - Cuando usar cual
2. [Regularizacion](#2-regularizacion)
   - Motivacion
   - Regularizacion L2 (Weight Decay)
   - Regularizacion L1
   - Comparacion L1 vs L2 vs Dropout
3. [Tareas Auxiliares](#3-tareas-auxiliares)
   - Motivacion
   - Ejemplos
   - Modo de uso y CombinedLoss
4. [Trabajo Practico](#4-trabajo-practico)

---

## 1. Funciones de Perdida

### 1.1. Contexto: el problema de Machine Learning

El flujo basico de Machine Learning es:

```text
Datos → Modelo → Prediccion
                     ↓
              ¿Que tan bien predijo?
              Funcion de Perdida
                     ↓
              Corregir (optimizacion)
                     ↓
              Modelo actualizado
```

El modelo tiene una **representacion** (su espacio de hipotesis). Por ejemplo, un modelo lineal simple:

```text
prediccion = A*x1 + B*x2 + C

Si prediccion > 0 → clase 1
Si prediccion <= 0 → clase 2
```

Los parametros A, B, C son lo que el modelo aprende. La **funcion de perdida** le dice que tan lejos esta de la respuesta correcta.

### 1.2. Que es una funcion de perdida

Es una **metrica del error de prediccion**. Mide que tan mal le fue al modelo.

```text
El MEJOR modelo es aquel que MINIMIZA la funcion de perdida
en promedio para TODOS los datos posibles.
```

Como no tenemos acceso a "todos los datos posibles", la estimamos usando el set de entrenamiento.

Expresado matematicamente:

```text
f* ≈ f*_Tr = argmin  (1/N) Σ L(f(x_i), y_i)
             f ∈ H          x_i ∈ Tr

Donde:
  f*     = el mejor modelo
  H      = espacio de hipotesis (todas las redes posibles con esa arquitectura)
  L      = funcion de perdida
  f(x_i) = prediccion del modelo para la entrada x_i
  y_i    = etiqueta real (la respuesta correcta)
  N      = numero de datos de entrenamiento
  Tr     = set de entrenamiento
```

En palabras simples: probamos muchas configuraciones de pesos y nos quedamos con la que tenga el error promedio mas bajo.

### 1.3. MSE (Mean Squared Error)

**Formula**:

```text
MSE = (1/N) Σ (ŷ_i - y_i)²

Donde:
  ŷ_i = prediccion del modelo
  y_i = valor real
  N   = numero de datos
```

**Ejemplo numerico** (de la clase):

Dos modelos que "en promedio" se equivocan por 5.0 puntos. ¿Cual tiene mejor MSE?

```text
Modelo A:                          Modelo B:
  Real  Pred  Dif   Dif²            Real  Pred  Dif   Dif²
  5.0   0.0   5.0   25             5.0   10.0  5.0   25
  10.0  10.0  0.0    0             10.0   5.0  5.0   25
  20.0  10.0  10.0  100            20.0  15.0  5.0   25
  Prom: 5.0   MSE: 41.7            Prom: 5.0   MSE: 25.0
                                                     ^^^^
                                                     MEJOR!
```

**Observacion clave**: ambos modelos tienen el mismo error promedio (5.0), pero MSE penaliza MAS al Modelo A porque tiene un error de 10.0 (que al cuadrado es 100). MSE **castiga desproporcionadamente los errores grandes**.

**Cuando usar MSE**:

- Problemas de **regresion** (predecir un valor continuo: precio, temperatura, etc.)
- Cuando importa **que tan lejos** esta la prediccion del valor real
- Cuando quieres penalizar mas los errores grandes

**En PyTorch**:

```python
import torch.nn as nn

# Forma 1: como modulo
loss_fn = nn.MSELoss()
loss = loss_fn(predictions, target)
loss.backward()

# Forma 2: como funcion
import torch.nn.functional as F
loss = F.mse_loss(predictions, target)
loss.backward()
```

### 1.4. Cross-Entropy

Cross-Entropy viene de la **Teoria de la Informacion**. En Machine Learning la usamos como una medida de distancia entre dos distribuciones de probabilidad:

- **p(x)**: la distribucion REAL (la respuesta correcta)
- **q(x)**: la distribucion ESTIMADA (lo que la red predice)

**Formula**:

```text
Cross-Entropy = - Σ y_i * log(ŷ_i)

Donde:
  y_i  = etiqueta real (one-hot: un 1 en la clase correcta, 0 en el resto)
  ŷ_i  = probabilidad estimada por el modelo para la clase i
```

**Ejemplo numerico** (clasificacion de 10 clases, la clase correcta es la 3):

```text
Distribucion real (one-hot):        Distribucion estimada (red):
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]    [0.01, 0.02, 0.89, 0.0, 0.03, 0.01, 0.01, 0.02, 0.0, 0.01]
        ↑                                          ↑
    clase 3 = 1                              clase 3 = 0.89

  Cross-Entropy = -(0*log(0.01) + 0*log(0.02) + 1*log(0.89) + ...)
                = -log(0.89)
                = 0.117  ← bajo (buena prediccion)
```

Como y_i es one-hot (solo un 1), la formula se simplifica:

```text
Cross-Entropy = -log(ŷ_clase_correcta)

  Si la red da 0.89 a la clase correcta: -log(0.89) = 0.117  ← bajo
  Si la red da 0.01 a la clase correcta: -log(0.01) = 4.605  ← alto!
```

**Problema con log(0)**: si la red predice exactamente 0.0 para alguna clase, log(0) = -infinito. PyTorch maneja esto internamente sumando un epsilon (1e-10) para evitar el problema.

### 1.5. Softmax: convertir numeros crudos en probabilidades

La red produce numeros "crudos" (logits) que pueden ser cualquier valor. Cross-Entropy necesita **probabilidades** (valores entre 0 y 1 que sumen 1). Softmax hace esa conversion:

```text
softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
```

**Ejemplo** (de la clase):

```text
Logits (salida cruda de la red):    Softmax (probabilidades):
  [-58.0]                             [0.000000]
  [ 18.3]                             [0.999684]  ← la mas alta
  [  0.008]                           [0.000000]
  [  0.935]                           [0.000000]
  [ -0.156]                           [0.000000]
  [-88.72]                            [0.000000]
  [  0.01]                            [0.000000]
  [ 10.24]                            [0.000316]
  [  3.333]                           [0.000000]
  [  2.5]                             [0.000000]

  → El valor mas alto (18.3) concentra casi toda la probabilidad
  → Los valores negativos se acercan a 0
  → Todo suma 1.0
```

**En PyTorch**, `nn.CrossEntropyLoss` hace **Softmax + Cross-Entropy** internamente. No necesitas aplicar softmax antes:

```python
# PyTorch hace softmax + cross-entropy todo junto
loss_fn = nn.CrossEntropyLoss()

# La red produce logits (numeros crudos), NO probabilidades
logits = model(x)           # [-58.0, 18.3, 0.008, ...]
loss = loss_fn(logits, label)  # PyTorch aplica softmax internamente

# Si necesitas las probabilidades para otra cosa:
probs = torch.softmax(logits, dim=1)
```

### 1.6. MSE para clasificacion: por que NO funciona bien

La clase muestra que usar MSE para clasificacion tiene un problema fundamental. Si el modelo predice UN numero (la clase como valor numerico):

```text
Clases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Si el modelo SIEMPRE predice "0":
  MSE = (0² + 1² + 2² + 3² + 4² + 5² + 6² + 7² + 8² + 9²) / 10 = 28.5

Si el modelo SIEMPRE predice "5" (la clase del medio):
  MSE = (5² + 4² + 3² + 2² + 1² + 0² + 1² + 2² + 3² + 4²) / 10 = 11.0  ← MENOR!
```

MSE **prefiere predecir valores intermedios**, no porque sean mas probables, sino porque estan mas cerca de todo. Esto no tiene sentido para clasificacion donde las clases no tienen un orden numerico significativo (la clase 5 no esta "entre" la 3 y la 7 de forma semantica).

### 1.7. Cuando usar cual (resumen)

| Tipo de problema | Funcion de perdida | Ejemplo |
|---|---|---|
| **Clasificacion** (N clases) | Cross-Entropy (`nn.CrossEntropyLoss`) | digitos 0-9, tipo de animal |
| **Clasificacion binaria** (si/no) | Binary Cross-Entropy (`nn.BCEWithLogitsLoss`) | spam?, enfermo? |
| **Regresion** (valor continuo) | MSE (`nn.MSELoss`) | precio, temperatura |
| **Regresion robusta** (con outliers) | L1 / MAE (`nn.L1Loss`) | tiempos de respuesta |

**Preguntas de la clase** para pensar:

| Problema | ¿MSE o Cross-Entropy? | Razon |
|---|---|---|
| Estimar probabilidad de compra (0% a 100%) | MSE | Es un valor continuo |
| Sentimiento positivo/negativo/neutro | Cross-Entropy | Son categorias discretas |
| Estimar precio del dolar | MSE | Es un valor continuo |
| Cantidad de alumnos en un curso (discreta) | MSE | Aunque es discreta, hay orden numerico |
| Predecir color de cada pixel (RGB) | MSE | Cada canal es un valor continuo 0-255 |
| Aproximar funcion continua no lineal | MSE | Regresion sobre funcion continua |

---

## 2. Regularizacion

### 2.1. Motivacion: overfitting

Cuando un modelo es muy complejo (muchos parametros) y los datos son pocos, el modelo **memoriza** los datos de entrenamiento en vez de aprender patrones generales. A esto se le llama **overfitting**.

```text
Sin regularizacion (red con 1000 neuronas, pocos datos):
  La red aprende limites de clasificacion MUY complejos
  que se ajustan PERFECTO a los datos de entrenamiento,
  pero fallan con datos nuevos.

  Train accuracy: 100%
  Test accuracy:  60%   ← MAL, memorizo

Con regularizacion:
  Los limites de clasificacion son mas SUAVES y generales.
  No se ajustan perfecto al entrenamiento, pero generalizan mejor.

  Train accuracy: 95%
  Test accuracy:  85%   ← MEJOR, generaliza
```

En la clase 7 vimos **Dropout** como tecnica de regularizacion. La clase 8 introduce dos tecnicas adicionales: **L2** y **L1**, que trabajan directamente sobre los **pesos** de la red.

### 2.2. Regularizacion L2 (Weight Decay)

La idea: **penalizar los pesos grandes**. Si un peso es muy grande, significa que la red depende mucho de un feature especifico, lo que puede ser overfitting.

**Formula**:

```text
Loss_total = Loss_original + λ * Σ w_i²
                              ↑       ↑
                          peso de la  suma de todos
                          regulariz.  los pesos al cuadrado

Donde:
  λ (lambda) = que tanta importancia le das a la regularizacion
  w_i = cada peso de la red
```

**Ejemplo numerico**:

```text
Sin regularizacion:
  Loss = CrossEntropy(prediccion, real) = 0.5
  → El optimizador solo se preocupa de predecir bien

Con regularizacion L2 (λ=0.2):
  Loss = CrossEntropy + 0.2 * (w1² + w2² + w3² + ...)
  Loss = 0.5 + 0.2 * (10² + 5² + 3² + ...)
  Loss = 0.5 + 0.2 * 134
  Loss = 0.5 + 26.8 = 27.3
  → El optimizador tiene que predecir bien Y ADEMAS mantener los pesos chicos
```

**Efecto**: los pesos grandes son "caros" (aumentan el loss). La red prefiere distribuir la importancia entre muchos pesos chicos en vez de depender de pocos pesos grandes.

```text
Sin L2:  pesos = [50.0, -30.0, 0.01, 0.01, 0.0, 0.0]
  → Pocas neuronas hacen todo (overfitting)

Con L2:  pesos = [3.2, -2.1, 1.5, -1.8, 0.9, -0.7]
  → Pesos mas parejos, limites de clasificacion mas suaves
```

**En PyTorch**: L2 se implementa como `weight_decay` en el optimizador. Es un solo parametro:

```python
# Sin regularizacion
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

# Con regularizacion L2 (lambda=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.2)
#                                                    ^^^^^^^^^^^^^^^
#                                                    esto es λ (lambda)

# PyTorch hace internamente:
#   loss_total = loss + weight_decay * sum(param ** 2)
```

**Valores tipicos de weight_decay**:

```text
  0.0:     sin regularizacion
  0.0001:  regularizacion sutil (comun en practica)
  0.001:   regularizacion moderada
  0.01:    regularizacion fuerte
  0.2:     regularizacion muy fuerte (del lab)
  100000:  regularizacion extrema → la red no aprende NADA
           (los pesos se quedan en ~0)
```

### 2.3. Regularizacion L1

Similar a L2, pero usa el **valor absoluto** en vez del cuadrado:

```text
Loss_total = Loss_original + λ * Σ |w_i|
```

**Diferencia clave con L2**:

```text
L2 (cuadrado):
  Penaliza MUCHO los pesos grandes, poco los chicos
  Efecto: pesos se hacen CHICOS pero NO cero
  pesos: [0.01, 0.02, 0.03, 0.01, 0.02, 0.01]  ← todos chicos, ninguno cero

L1 (absoluto):
  Penaliza IGUAL pesos grandes y chicos
  Efecto: muchos pesos se hacen EXACTAMENTE cero
  pesos: [0.0, 0.5, 0.0, 0.0, 0.3, 0.0]  ← muchos en cero (sparse)
```

**L1 produce sparsity** (pesos en cero): es como si la red "seleccionara" automaticamente que features son importantes y descartara el resto. Esto es util cuando tienes muchos features y sospechas que solo algunos importan.

**En PyTorch**: L1 no viene como parametro del optimizador. Hay que implementarlo manualmente:

```python
# L1 manual
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss_fn(predictions, labels) + l1_lambda * l1_norm
loss.backward()
```

### 2.4. Comparacion: L1 vs L2 vs Dropout

| Tecnica | Que hace | Efecto | En PyTorch |
|---|---|---|---|
| **L2** | Penaliza pesos² | Pesos chicos (distribuidos) | `weight_decay=0.2` en optimizer |
| **L1** | Penaliza \|pesos\| | Pesos en cero (sparse) | Manual: `sum(p.abs().sum())` |
| **Dropout** | Apaga neuronas al azar | Redundancia (todas aprenden) | `nn.Dropout(p=0.5)` |

Se pueden **combinar** (y es comun hacerlo):

```python
# Red con Dropout + L2 (weight_decay)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),        # Dropout
    nn.Linear(256, 10),
)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # L2
```

---

## 3. Tareas Auxiliares

### 3.1. Motivacion

A veces la tarea principal es dificil y la red no tiene suficiente "señal" para aprender bien. Una **tarea auxiliar** es una tarea adicional que se entrena al mismo tiempo para ayudar a la red a aprender mejores representaciones.

```text
Ejemplo: detectar si una persona esta sonriendo en una foto

  Solo tarea principal (Smiling):
    La red tiene que aprender todo sobre caras
    a partir de una sola señal binaria (smile/no smile)

  Con tarea auxiliar (Smiling + Young):
    La red aprende a detectar sonrisas Y juventud al mismo tiempo.
    Para predecir ambas cosas, las capas compartidas aprenden
    representaciones mas RICAS de la cara (bordes, texturas, formas)
    que ayudan a AMBAS tareas.

    → La tarea principal mejora porque las features son mejores
```

### 3.2. Ejemplo del laboratorio: CelebA

El laboratorio usa el dataset **CelebA** (rostros de celebridades) con multiples atributos anotados:

```text
Atributos disponibles (40 en total):
  Smiling, Young, Male, Eyeglasses, Bald, Bangs,
  Big_Nose, Black_Hair, Blond_Hair, Heavy_Makeup, ...

Tarea principal:     ¿Esta sonriendo? (Smiling) → binario
Tarea auxiliar 1:    ¿Es joven? (Young) → binario
Tarea auxiliar 2:    ¿Donde estan los landmarks faciales? → regresion (x,y de ojos, nariz, boca)
```

### 3.3. Arquitectura con tarea auxiliar

La red comparte las capas convolucionales (que aprenden features visuales) y tiene **dos cabezas** de salida:

```text
                    Capas compartidas              Cabeza principal
  Imagen → Conv1 → Conv2 → FC1 → FC2 ─────────→ FC3 → ¿Smiling?
                                     │
                                     └──────────→ FC4 → ¿Young?  (tarea auxiliar)
                                                  Cabeza auxiliar
```

En PyTorch (del laboratorio):

```python
class FaceModel(nn.Module):
    def __init__(self, auxiliary_task_dim=10):
        super().__init__()
        # Capas compartidas (aprenden features comunes)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)

        # Cabeza principal (tarea: Smiling)
        self.fc3 = nn.Linear(84, 1)

        # Cabeza auxiliar (tarea: Young o Landmarks)
        if auxiliary_task_dim is not None:
            self.fc4 = nn.Linear(84, auxiliary_task_dim)

    def forward(self, x):
        # Capas compartidas
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Dos salidas
        main_task = self.fc3(x)                       # prediccion principal
        auxiliary_task = self.fc4(x) if self.fc4 else None  # prediccion auxiliar
        return main_task, auxiliary_task
```

### 3.4. CombinedLoss: como combinar las perdidas

Las dos tareas generan dos losses. Se combinan con un peso lambda (λ):

```text
Loss_total = Loss_principal + λ * Loss_auxiliar

  λ = 0.0:  la tarea auxiliar no tiene efecto (se ignora)
  λ = 0.2:  la tarea auxiliar tiene poca influencia (valor tipico)
  λ = 1.0:  ambas tareas tienen la misma importancia
```

En PyTorch (del laboratorio):

```python
class CombinedLoss(nn.Module):
    def __init__(self, auxiliary_task, auxiliary_weight):
        super().__init__()
        self.auxiliary_task = auxiliary_task
        self.aux_weight = auxiliary_weight  # esto es λ

    def forward(self, main_pred, aux_pred, main_labels, aux_labels):
        if aux_labels is None:
            # Sin tarea auxiliar: solo cross-entropy principal
            return F.binary_cross_entropy_with_logits(main_pred, main_labels)
        else:
            # Con tarea auxiliar: combinar ambas
            main_loss = F.binary_cross_entropy_with_logits(main_pred, main_labels)

            if self.auxiliary_task == 'Landmarks':
                aux_loss = F.mse_loss(aux_pred, aux_labels)  # regresion para landmarks
            else:
                aux_loss = F.binary_cross_entropy_with_logits(aux_pred, aux_labels)  # clasificacion

            return main_loss + self.aux_weight * aux_loss
```

**Nota importante**: la tarea auxiliar puede usar una funcion de perdida DISTINTA a la tarea principal. En el ejemplo, la tarea principal (Smiling) usa Binary Cross-Entropy, y la tarea auxiliar de Landmarks usa MSE (porque predecir coordenadas x,y es regresion).

### 3.5. Tres experimentos del laboratorio

```text
Experimento 1: Solo tarea principal (Smiling)
  auxiliary_task = None
  → Baseline para comparar

Experimento 2: Tarea principal + auxiliar binaria (Smiling + Young)
  auxiliary_task = 'Young'
  auxiliary_weight = 0.2
  → La tarea auxiliar es similar (clasificar un atributo de la cara)

Experimento 3: Tarea principal + auxiliar de regresion (Young + Landmarks)
  primary_task = 'Young'
  auxiliary_task = 'Landmarks'
  auxiliary_weight = 0.1
  → La tarea auxiliar es predecir coordenadas de puntos faciales
  → Fuerza a la red a entender la GEOMETRIA de la cara
```

### 3.6. Cuando usar tareas auxiliares

```text
UTIL cuando:
  - La tarea principal tiene pocos datos
  - Hay tareas relacionadas con datos disponibles
  - Quieres que la red aprenda mejores representaciones internas
  - Las tareas comparten estructura (ej: ambas necesitan entender caras)

NO util cuando:
  - Las tareas no estan relacionadas (ej: detectar sonrisas + predecir clima)
  - La tarea auxiliar es mucho mas facil o dificil que la principal
  - λ esta mal calibrado (muy alto: la auxiliar domina; muy bajo: no tiene efecto)
```

### 3.7. Problemas con λ (del laboratorio)

Si las tareas tienen **ordenes de magnitud distintos** en su loss, hay que ajustar λ cuidadosamente:

```text
main_loss ≈ 0.5 (cross-entropy, valores entre 0 y ~5)
aux_loss  ≈ 500.0 (MSE de landmarks, valores grandes)

Con λ=0.2:
  Loss = 0.5 + 0.2 * 500 = 100.5
  → La tarea auxiliar DOMINA completamente!
  → La red ignora la tarea principal

Solucion: usar λ MUY chico
  Loss = 0.5 + 0.001 * 500 = 1.0  ← balanceado
```

Si hay **multiples tareas auxiliares**, se puede usar un λ por cada una:

```text
Loss = Loss_principal + λ1 * Loss_aux1 + λ2 * Loss_aux2 + λ3 * Loss_aux3

Cada λ se ajusta segun la escala de cada loss auxiliar.
```

---

## 4. Trabajo Practico

### 4.1. Experimento 1: Funciones de Perdida (CIFAR-10)

El laboratorio entrena una CNN simple en **CIFAR-10** (imagenes a color de 10 clases: avion, auto, pajaro, gato, etc.) y compara Cross-Entropy vs MSE.

```python
# CIFAR-10 (imagenes 32x32 a color)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
# 10 clases: avion, auto, pajaro, gato, ciervo, perro, rana, caballo, barco, camion

# Red CNN simple
class Net(nn.Module):
    def __init__(self, output_dim=10):
        self.conv1 = nn.Conv2d(3, 6, 5)    # 3 canales RGB
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

# Con Cross-Entropy: output_dim=10 (una salida por clase)
# Con MSE: output_dim=1 (una sola salida, el "numero" de la clase)
```

### 4.2. Experimento 2: Regularizacion L2

Usa un dataset 2D simple (dos clases en un plano) con una red grande (1000 neuronas por capa) para mostrar overfitting:

```python
# Red MUY grande para datos simples (overfitting garantizado)
class ClassificationModel(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(2, 1000)     # 2 entradas, 1000 neuronas!
        self.layer2 = nn.Linear(1000, 1000)  # 1000 → 1000
        self.layer3 = nn.Linear(1000, 1)     # 1000 → 1 salida

# Sin regularizacion
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

# Con regularizacion L2
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.2)
#                                                   ^^^^^^^^^^^^^^^
#                                                   la UNICA diferencia
```

### 4.3. Experimento 3: Tareas Auxiliares (CelebA)

Usa CelebA con una CNN para clasificar atributos faciales, comparando:
- Solo tarea principal
- Tarea principal + auxiliar binaria
- Tarea principal + auxiliar de regresion (landmarks)

---

## 5. Resumen de la Clase

```text
Funciones de Perdida:
  MSE:            para regresion (predecir numeros)
  Cross-Entropy:  para clasificacion (elegir clases)
  NO usar MSE para clasificacion

Regularizacion (evitar overfitting):
  L2 (Weight Decay): penaliza pesos grandes → pesos distribuidos
                     Un parametro en PyTorch: weight_decay=0.2
  L1:               penaliza |pesos| → muchos pesos en cero (sparsity)
                     Implementacion manual
  Dropout:          apaga neuronas al azar → redundancia
  Se pueden combinar

Tareas Auxiliares:
  Entrenar dos tareas al mismo tiempo con capas compartidas
  La tarea auxiliar ayuda a aprender mejores features
  Loss = Loss_principal + λ * Loss_auxiliar
  λ controla la importancia de la tarea auxiliar
  Cuidado con la escala de los losses
```

---

> **Referencias**:
> - Material de clase: Carlos Aspillaga, Diplomado IA UC
> - PyTorch docs: [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) | [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
> - Dataset: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
