# Laboratorio 5 — Redes Convolucionales: AlexNet

Diplomado IA: Aprendizaje Profundo I

---

## 1. ¿Qué es una Red Convolucional (CNN)?

Una red convolucional es una arquitectura de red neuronal diseñada para procesar datos con estructura espacial (imágenes). A diferencia de una red densa (MLP), las CNNs explotan que los píxeles cercanos están relacionados entre sí, usando **filtros** que se deslizan sobre la imagen.

Hay tres tipos de capas principales:

| Tipo de capa | ¿Qué hace? |
|---|---|
| **Convolución** | Aplica un filtro para detectar patrones locales (bordes, texturas, formas) |
| **Pooling** | Reduce el tamaño espacial, concentrando la información más relevante |
| **Fully Connected (FC)** | Combina todas las características para producir una predicción |

---

## 2. ¿Cómo funciona una Convolución por dentro?

Un filtro es una pequeña matriz de números (pesos). Se **desliza** sobre la imagen y en cada posición hace un **producto punto** entre el filtro y el parche de imagen que cubre. El resultado es un único número que mide "qué tanto se parece ese parche al patrón del filtro".

```text
Imagen (fragmento 3×3):    Filtro (3×3):         Producto punto:
┌───┬───┬───┐              ┌────┬────┬────┐
│ 1 │ 2 │ 3 │              │  0 │ -1 │  0 │      1×0  + 2×(-1) + 3×0
├───┼───┼───┤       ×      ├────┼────┼────┤   =  4×(-1)+ 5×5   + 6×(-1)  = 5
│ 4 │ 5 │ 6 │              │ -1 │  5 │ -1 │      7×0  + 8×(-1) + 9×0
├───┼───┼───┤              ├────┼────┼────┤
│ 7 │ 8 │ 9 │              │  0 │ -1 │  0 │
└───┴───┴───┘              └────┴────┴────┘
```

El filtro se desplaza por toda la imagen (según el stride), produciendo un **mapa de activación**: una nueva imagen donde cada píxel indica cuánto se detectó ese patrón en esa ubicación.

La red **aprende los valores del filtro** durante el entrenamiento — nadie los define a mano.

---

## 3. ¿Qué es un Filtro Laplaciano?

El Laplaciano es un ejemplo clásico de filtro detector de bordes. La idea es simple: **compara cada píxel con sus vecinos**. Si todos son similares (zona uniforme), el resultado es ≈0. Si el píxel central es muy distinto (borde), el resultado es alto.

```text
Filtro Laplaciano:
┌────┬────┬────┐
│  0 │ -1 │  0 │
├────┼────┼────┤
│ -1 │  4 │ -1 │
├────┼────┼────┤
│  0 │ -1 │  0 │
└────┴────┴────┘
```

**En zona uniforme** (todos los píxeles ≈ 20):

```text
= 0 - 20 + 0 - 20 + 80 - 20 + 0 - 20 + 0 = 0   ← sin borde
```

**En un borde** (centro=200, vecinos=20):

```text
= 0 - 20 + 0 - 20 + 800 - 20 + 0 - 200 + 0 = 540  ← borde detectado
```

La red no usa el Laplaciano explícitamente — pero los filtros que aprende en conv1 terminan siendo matemáticamente similares a él, porque el entrenamiento descubre que detectar bordes es útil.

---

## 4. ¿Qué es la Invarianza a Traslaciones?

Sin invarianza, una red entrenada con gatos en el centro de la imagen no reconocería un gato en la esquina — aprendió "gato en el centro", no "gato".

**MaxPool resuelve esto.** Aunque la activación se mueva un píxel, el máximo de la ventana sigue siendo el mismo:

```text
Activación fuerte en posición (2,2):    Imagen movida → activación en (2,3):

┌───┬───┬───┬───┐                       ┌───┬───┬───┬───┐
│ 0 │ 0 │ 0 │ 0 │                       │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┤                       ├───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │                       │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┤                       ├───┼───┼───┼───┤
│ 0 │ 9 │ 0 │ 0 │                       │ 0 │ 0 │ 9 │ 0 │
├───┼───┼───┼───┤                       ├───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │                       │ 0 │ 0 │ 0 │ 0 │
└───┴───┴───┴───┘                       └───┴───┴───┴───┘

MaxPool 2×2 ventana izquierda:          MaxPool 2×2 ventana derecha:
max(0, 0, 0, 9) = 9  ✓                  max(0, 9, 0, 0) = 9  ✓
```

**Ambos dan 9** aunque la activación se desplazó. Eso es invarianza: pequeños movimientos del objeto no cambian la respuesta de la red.

> Limitación: MaxPool solo da invarianza a **pequeños** desplazamientos (dentro de la ventana). Para objetos muy desplazados o rotados se necesitan otras técnicas (data augmentation, etc.).

---

## 5. ¿Por qué cada capa detecta cosas más complejas?

Nadie programa qué debe detectar cada capa — la red lo aprende sola. Ocurre porque **cada capa ve la salida de la anterior**, no los píxeles originales.

### La jerarquía de características

```text
Píxeles RGB
    ↓
conv1 — ve píxeles directamente
        aprende: bordes, gradientes de color, orientaciones

    ↓
conv2 — ve mapas de bordes (salida de conv1)
        aprende: esquinas (borde horiz + borde vert),
                 texturas (bordes repetidos)

    ↓
conv3 — ve combinaciones de esquinas y texturas
        aprende: formas geométricas simples, patrones

    ↓
conv4 — ve formas
        aprende: partes de objetos (ojos, ruedas, hojas)

    ↓
conv5 — ve partes de objetos
        aprende: objetos completos o partes muy distintivas
```

### ¿Por qué ocurre esto espontáneamente?

El entrenamiento ajusta los pesos para minimizar el error de clasificación. Para clasificar bien, la red descubre que:

1. Primero conviene detectar cosas simples (bordes) — son fáciles de encontrar y están en toda imagen.
2. Combinar bordes da esquinas — más informativas.
3. Combinar esquinas da formas — aún más informativas.
4. Y así sucesivamente.

Esta jerarquía emerge porque es la estrategia más eficiente para reducir el error. No está impuesta — es descubierta.

### Visualización real (Zeiler & Fergus, 2013)

Los investigadores visualizaron qué activa cada capa de AlexNet y encontraron exactamente este patrón:

```text
Layer 1:  bordes, gradientes de color, orientaciones básicas
Layer 2:  texturas, cuadrículas, patrones repetitivos
Layer 3:  formas simples, mallas, texto
Layer 4:  partes específicas: patas, ojos, hojas, ruedas
Layer 5:  objetos o partes muy características
```

---

## 6. Fórmulas clave

### Dimensión de salida de una capa convolucional o de pooling

```
O = floor((I - K + 2P) / S) + 1
```

Donde:
- `I` = tamaño de entrada (alto o ancho)
- `K` = tamaño del kernel (filtro)
- `P` = padding (píxeles añadidos alrededor)
- `S` = stride (paso del filtro)
- `floor()` = parte entera inferior

### Cantidad de parámetros de una capa Conv2d

```
params = C_out × (C_in × K_H × K_W + 1)
```

El `+1` es el bias por filtro.

### Cantidad de parámetros de una capa Linear

```
params = out_features × (in_features + 1)
```

---

## 7. Arquitectura Original de AlexNet

AlexNet fue diseñada en 2012 para clasificar imágenes del dataset **ImageNet** (1000 categorías) con entradas de **3 × 224 × 224** (RGB).

### Diagrama de flujo de datos

```
Input:   3 × 224 × 224

conv1 ──► 96 × 55 × 55 ──► MaxPool ──► 96 × 27 × 27
conv2 ──► 256 × 27 × 27 ──► MaxPool ──► 256 × 13 × 13
conv3 ──► 384 × 13 × 13
conv4 ──► 384 × 13 × 13
conv5 ──► 256 × 13 × 13 ──► MaxPool ──► 256 × 6 × 6

Flatten ──► 9216

fc6 ──► 4096
fc7 ──► 4096
fc8 ──► 1000 (clases)
```

---

### Capa a capa: ¿qué hace cada una?

#### `conv1` — Primera extracción de características
```python
nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=(4,4), padding=(2,2))
nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
nn.ReLU()
```
- **Entrada:** 3 × 224 × 224 (imagen RGB)
- **Convolución:** 96 filtros de 11×11. Stride=4 reduce rápido el mapa espacial.
  - O = floor((224 - 11 + 4) / 4) + 1 = floor(217/4) + 1 = **55**
  - Salida conv: **96 × 55 × 55**
- **MaxPool 3×3 stride 2:**
  - O = floor((55 - 3) / 2) + 1 = **27**
  - Salida pool: **96 × 27 × 27**
- **ReLU:** Introduce no-linealidad. Elimina valores negativos (f(x) = max(0, x)).
- **Parámetros conv:** 96 × (3 × 11 × 11 + 1) = **34,944**
- **Detecta:** bordes, orientaciones, colores básicos.

---

#### `conv2` — Patrones de mediana complejidad
```python
nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(1,1), padding=(2,2))
nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
nn.ReLU()
```
- **Entrada:** 96 × 27 × 27
- **Convolución:** 256 filtros de 5×5. Padding=2 preserva dimensiones.
  - O = floor((27 - 5 + 4) / 1) + 1 = **27**
  - Salida conv: **256 × 27 × 27**
- **MaxPool 3×3 stride 2:**
  - O = floor((27 - 3) / 2) + 1 = **13**
  - Salida pool: **256 × 13 × 13**
- **Parámetros conv:** 256 × (96 × 5 × 5 + 1) = **614,656**
- **Detecta:** texturas, esquinas, combinaciones de bordes.

---

#### `conv3` — Patrones más abstractos
```python
nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=(1,1))
nn.ReLU()
```
- **Entrada:** 256 × 13 × 13
- Padding=1 con kernel=3 preserva dimensiones: O = (13 - 3 + 2) / 1 + 1 = **13**
- **Salida:** 384 × 13 × 13
- **Sin MaxPool** en esta capa.
- **Parámetros:** 384 × (256 × 3 × 3 + 1) = **885,120**
- **Detecta:** formas más complejas compuestas de texturas.

---

#### `conv4` — Refinamiento de características
```python
nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=(1,1))
nn.ReLU()
```
- **Entrada:** 384 × 13 × 13
- Igual que conv3: preserva dimensiones.
- **Salida:** 384 × 13 × 13
- **Parámetros:** 384 × (384 × 3 × 3 + 1) = **1,327,488**
- **Detecta:** partes de objetos (ruedas, ojos, ventanas, etc.)

---

#### `conv5` — Última capa convolucional
```python
nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
nn.ReLU()
```
- **Entrada:** 384 × 13 × 13
- **Convolución:** reduce canales a 256, mantiene 13×13.
  - Salida conv: **256 × 13 × 13**
- **MaxPool 3×3 stride 2:**
  - O = floor((13 - 3) / 2) + 1 = **6**
  - Salida pool: **256 × 6 × 6**
- **Parámetros conv:** 256 × (384 × 3 × 3 + 1) = **884,992**
- **Detecta:** objetos completos o partes distintivas.

---

#### `flat` — Aplanamiento (Flatten)
```python
nn.Flatten()
```
- Convierte el tensor 3D en un vector 1D.
- **256 × 6 × 6 = 9,216 neuronas**
- Necesario para pasar a capas lineales.

---

#### `fc6` — Primera capa densa
```python
nn.Linear(9216, 4096)
nn.ReLU()
```
- Conecta los 9,216 valores extraídos por las convoluciones con 4,096 neuronas.
- **Parámetros:** 4096 × (9216 + 1) = **37,752,832**
- Es la capa más grande de la red.

---

#### `fc7` — Segunda capa densa
```python
nn.Linear(4096, 4096)
nn.ReLU()
```
- **Parámetros:** 4096 × (4096 + 1) = **16,781,312**

---

#### `fc8` — Capa de salida (clasificación)
```python
nn.Linear(4096, 1000)
```
- Produce 1,000 scores, uno por clase.
- **No tiene ReLU** — la activación la aplica la función de pérdida (CrossEntropyLoss usa Softmax internamente).
- **Parámetros:** 1000 × (4096 + 1) = **4,097,000**

---

### Resumen de parámetros por capa

| Capa | Parámetros |
|------|-----------|
| conv1 | 34,944 |
| conv2 | 614,656 |
| conv3 | 885,120 |
| conv4 | 1,327,488 |
| conv5 | 884,992 |
| fc6 | 37,752,832 |
| fc7 | 16,781,312 |
| fc8 | 4,097,000 |
| **Total** | **62,378,344** |

> Las capas FC concentran ~95% de los parámetros, aunque son solo 3 capas.

---

## 8. Actividad 1 — Adaptar para 102 clases

### ¿Qué hay que cambiar?

AlexNet clasifica 1,000 clases (ImageNet). Para un dataset de 102 clases (ej. Oxford Flowers), solo la **última capa** `fc8` necesita cambiar. Todo lo demás — la extracción de características — puede reutilizarse tal cual.

### El cambio

| Capa | Original | Modificado |
|------|----------|-----------|
| fc8 | `Linear(4096, 1000)` | `Linear(4096, 102)` |

### Impacto en parámetros

| | Original | Modificado | Diferencia |
|---|---|---|---|
| fc8 | 4,097,000 | 417,894 | **−3,679,106** |
| **Total** | 62,378,344 | 58,699,238 | **−3,679,106** |

### Código

```python
class MiAlexNet(nn.Module):
    def __init__(self):
        super(MiAlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11,11), stride=(4,4), padding=(2,2)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )
        self.flat = nn.Flatten()
        self.fc6 = nn.Sequential(nn.Linear(9216, 4096), nn.ReLU())
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU())

        # CAMBIO: 1000 → 102
        self.fc8 = nn.Sequential(nn.Linear(4096, 102))

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x)
        x = self.flat(x)
        x = self.fc6(x); x = self.fc7(x); x = self.fc8(x)
        return x
```

---

## 9. Actividad 2 — Adaptar para imágenes 64 × 64

### Problema

Con imágenes de 64×64 (en lugar de 224×224), si usamos AlexNet sin modificar:

- `conv1` tiene `kernel=11` y `stride=4` → diseñado para 224×224, demasiado agresivo para una entrada pequeña.
- Las dimensiones colapsan antes de llegar a `fc6`, haciendo imposible mantener el resto de la red igual.

### Análisis: ¿Qué pasa con las dimensiones originales sobre 64×64?

Usando los parámetros originales de conv1 y conv2 con entrada 64×64:

```
conv1 (k=11, s=4, p=2): floor((64-11+4)/4)+1 = floor(57/4)+1 = 15
MaxPool (k=3, s=2):     floor((15-3)/2)+1 = 7
→ 96 × 7 × 7

conv2 (k=5, s=1, p=2): floor((7-5+4)/1)+1 = 7
MaxPool (k=3, s=2):    floor((7-3)/2)+1 = 3
→ 256 × 3 × 3

conv3,4 (preservan):   256/384 × 3 × 3
conv5 + MaxPool:       floor((3-3)/2)+1 = 1
→ 256 × 1 × 1 = 256
```

El mapa espacial colapsa a **1×1** — solo quedan 256 valores para `fc6`. La red pierde toda la información espacial.

### Solución: Modificar conv1 y conv2 para preservar las dimensiones espaciales

El objetivo es que las dimensiones espaciales lleguen a **6×6** antes del flatten, igual que el AlexNet original con 224×224. Así `fc6` y el resto de capas no necesitan cambiar.

Se trabaja hacia atrás desde el objetivo:

```
Necesitamos 256×6×6 = 9216 antes de fc6  →  fc6 sin cambios ✓

MaxPool5 (k=3, s=2) necesita entrada 13×13:  floor((13-3)/2)+1 = 6 ✓
conv3,4,5 (p=1) preservan: necesitan 13×13
MaxPool2 (k=3, s=2) necesita entrada 27×27:  floor((27-3)/2)+1 = 13 ✓
conv2 (k=5, s=1, p=0) preserva canales:      floor((31-5+0)/1)+1 = 27 ✓
MaxPool1 (k=3, s=2) necesita entrada 64×64:  floor((64-3)/2)+1 = 31 ✓
conv1 (k=3, s=1, p=1):                       floor((64-3+2)/1)+1 = 64 ✓
```

#### Nuevo flujo de dimensiones

```
Input:   3 × 64 × 64

conv1 (k=3, s=1, p=1):   floor((64-3+2)/1)+1 = 64
MaxPool (k=3, s=2):      floor((64-3)/2)+1 = 31
→ 96 × 31 × 31

conv2 (k=5, s=1, p=0):   floor((31-5+0)/1)+1 = 27
MaxPool (k=3, s=2):      floor((27-3)/2)+1 = 13
→ 256 × 13 × 13

conv3 (k=3, s=1, p=1):   13 → 13  → 384 × 13 × 13
conv4 (k=3, s=1, p=1):   13 → 13  → 384 × 13 × 13
conv5 (k=3, s=1, p=1):   13 → 13  → 256 × 13 × 13
MaxPool (k=3, s=2):      floor((13-3)/2)+1 = 6
→ 256 × 6 × 6

Flatten: 256 × 6 × 6 = 9216
```

`fc6` permanece `Linear(9216, 4096)` — sin cambios.

### Resumen de cambios

| Capa | Original | Modificado | Razón |
|------|----------|-----------|-------|
| conv1 kernel | (11,11) | **(3,3)** | Adecuado para imagen pequeña |
| conv1 stride | (4,4) | **(1,1)** | Evita colapso espacial en 64×64 |
| conv1 padding | (2,2) | **(1,1)** | Mantiene salida 64×64 con k=3 |
| conv2 padding | (2,2) | **(0,0)** | Reduce 31→27 para llegar a 13 tras MaxPool |
| fc6 | sin cambio | **sin cambio** | El flatten sigue siendo 9216 |

### Impacto en parámetros

| Capa | Original (act. 1) | Modificado | Diferencia |
|------|----------|-----------|-----------|
| conv1 | 34,944 | **2,688** | **−32,256** |
| conv2 | 614,656 | 614,656 | Sin cambio (mismo kernel y canales) |
| fc6 | 37,752,832 | 37,752,832 | Sin cambio |
| **Total** | 58,699,238 | **58,666,982** | **−32,256** |

> Conv1 reduce parámetros porque el kernel pasa de 11×11×3×96 a 3×3×3×96. Conv2 no cambia porque solo se ajustó el padding, no el kernel ni los canales.

### Código

```python
class MiAlexNet(nn.Module):
    def __init__(self):
        super(MiAlexNet, self).__init__()

        # CAMBIO: kernel (11,11)→(3,3), stride (4,4)→(1,1), padding (2,2)→(1,1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )
        # CAMBIO: padding (2,2)→(0,0) para reducir 31→27
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )
        self.flat = nn.Flatten()

        # Sin cambio: el flatten sigue siendo 256×6×6 = 9216
        self.fc6 = nn.Sequential(nn.Linear(9216, 4096), nn.ReLU())
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU())
        self.fc8 = nn.Sequential(nn.Linear(4096, 102))

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x)
        x = self.flat(x)
        x = self.fc6(x); x = self.fc7(x); x = self.fc8(x)
        return x
```

---

## 10. Actividad 3 — Agregar conv6 y reducir FC a 1024

### Punto de partida

Se vuelve al **modelo original** (AlexNet con entrada 224×224 y 1,000 clases). No se aplican los cambios de las actividades anteriores.

### Cambios requeridos

1. Agregar `conv6` después de `conv5`: reduce de 256 → 128 filtros.
2. Ajustar `fc6` para recibir el nuevo tamaño del flatten.
3. Cambiar `fc6`, `fc7` para trabajar con **1024** dimensiones en vez de 4096.

### Análisis de dimensiones con conv6

`conv5` deja un mapa de **256 × 6 × 6**. Agregamos:

```python
conv6: nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
```

Con `padding=1` y `kernel=3`, las dimensiones espaciales se preservan:

```
O = floor((6 - 3 + 2) / 1) + 1 = 6
```

```
conv6: 256 × 6 × 6 → 128 × 6 × 6

Flatten: 128 × 6 × 6 = 4608
```

### Cadena completa con conv6

```
Input:   3 × 224 × 224

conv1 → 96 × 27 × 27   (sin cambio)
conv2 → 256 × 13 × 13  (sin cambio)
conv3 → 384 × 13 × 13  (sin cambio)
conv4 → 384 × 13 × 13  (sin cambio)
conv5 → 256 × 6 × 6    (sin cambio)
conv6 → 128 × 6 × 6    ← NUEVA CAPA

Flatten: 4608

fc6 → 1024   ← cambio: 9216→4608 entrada, 4096→1024 salida
fc7 → 1024   ← cambio: 4096→1024
fc8 → 1000   (sin cambio en clases)
```

### Resumen de cambios

| Capa | Original | Modificado | Razón |
|------|----------|-----------|-------|
| conv6 | no existe | `Conv2d(256, 128, 3, 1, 1)` | Nueva capa reductora |
| fc6 | `Linear(9216, 4096)` | `Linear(4608, 1024)` | Nuevo flatten + menos dim |
| fc7 | `Linear(4096, 4096)` | `Linear(1024, 1024)` | Reducción de dimensiones |
| fc8 | `Linear(4096, 1000)` | `Linear(1024, 1000)` | Mantiene 1000 clases |

### Impacto en parámetros

| Capa | Original | Modificado | Diferencia |
|------|----------|-----------|-----------|
| conv5 | 884,992 | 884,992 | Sin cambio |
| conv6 | — | 128 × (256×3×3 + 1) = **295,040** | +295,040 |
| fc6 | 37,752,832 | 1024 × (4608 + 1) = **4,718,592** | −33,034,240 |
| fc7 | 16,781,312 | 1024 × (1024 + 1) = **1,049,600** | −15,731,712 |
| fc8 | 4,097,000 | 1000 × (1024 + 1) = **1,025,000** | −3,072,000 |
| **Total** | 62,378,344 | ~11,258,960 | **−51,119,384** |

> La red pierde ~51M parámetros (~82% menos). La nueva conv6 agrega apenas 295K, pero las capas FC se reducen drásticamente al pasar de 4096 → 1024 dimensiones.

### Código

```python
class MiAlexNet(nn.Module):
    def __init__(self):
        super(MiAlexNet, self).__init__()

        # conv1-conv5 sin cambios (modelo original 224×224)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11,11), stride=(4,4), padding=(2,2)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU()
        )

        # NUEVA CAPA: conv6 reduce 256 → 128 canales, preserva 6×6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU()
        )

        self.flat = nn.Flatten()

        # CAMBIO: entrada 4608 (128×6×6), salida 1024
        self.fc6 = nn.Sequential(nn.Linear(4608, 1024), nn.ReLU())
        # CAMBIO: 4096 → 1024
        self.fc7 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU())
        # CAMBIO: entrada 1024
        self.fc8 = nn.Sequential(nn.Linear(1024, 1000))

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x)
        x = self.conv6(x)   # ← nueva capa en el forward
        x = self.flat(x)
        x = self.fc6(x); x = self.fc7(x); x = self.fc8(x)
        return x
```

---

## 11. Comparativa final de las tres actividades

| | AlexNet Original | Act. 1 (102 clases) | Act. 2 (64×64) | Act. 3 (conv6 + 1024) |
|---|---|---|---|---|
| **Input** | 3×224×224 | 3×224×224 | 3×64×64 | 3×224×224 |
| **conv1 stride** | 4 | 4 | **2** | 4 |
| **conv6** | — | — | — | **128 filtros** |
| **Flatten** | 9,216 | 9,216 | 1,024 | **4,608** |
| **fc6** | 9216→4096 | 9216→4096 | **1024→4096** | **4608→1024** |
| **fc7** | 4096→4096 | 4096→4096 | 4096→4096 | **1024→1024** |
| **fc8** | 4096→1000 | **4096→102** | **4096→102** | **1024→1000** |
| **Total params** | ~62.4M | ~58.7M | ~28.8M | ~11.3M |

### Insight principal

> Las capas fully connected (`fc6`, `fc7`, `fc8`) concentran la mayor parte de los parámetros de AlexNet. Cualquier cambio que afecte el tamaño del tensor antes del `Flatten` —ya sea por imágenes más pequeñas o por añadir capas que reduzcan canales— produce una reducción enorme en la red.

---

## 12. Funciones y conceptos de PyTorch usados en el lab

### `nn.Module` — La base de todo modelo

```python
class MiAlexNet(nn.Module):
    def __init__(self):
        super(MiAlexNet, self).__init__()  # inicializa la clase padre
```

Toda red neuronal en PyTorch hereda de `nn.Module`. Esto le da:

- Registro automático de parámetros entrenables.
- El método `forward()` que define cómo fluyen los datos.
- Métodos como `.parameters()`, `.named_children()`, `.train()`, `.eval()`.

---

### `nn.Sequential` — Agrupador de capas en orden

```python
self.conv1 = nn.Sequential(
    nn.Conv2d(...),
    nn.MaxPool2d(...),
    nn.ReLU()
)
```

Ejecuta las capas **en el orden en que se definen**. Es equivalente a escribir:

```python
x = conv(x)
x = pool(x)
x = relu(x)
```

Pero más limpio cuando las capas siempre van juntas.

---

### `nn.Conv2d` — La convolución 2D

```python
nn.Conv2d(
    in_channels=3,       # canales de entrada (ej: 3 para RGB)
    out_channels=96,     # cuántos filtros aprender
    kernel_size=(11,11), # tamaño del filtro
    stride=(4,4),        # paso del filtro (cuánto se mueve)
    padding=(2,2)        # píxeles de relleno alrededor
)
```

**¿Qué aprende?** Cada filtro aprende a detectar un patrón visual. Las capas iniciales detectan bordes y colores; las finales, partes de objetos.

**Efecto del stride:** A mayor stride, más se reduce el mapa espacial de salida.

**Efecto del padding:** Preserva las dimensiones espaciales cuando `padding = (kernel_size - 1) / 2`.

---

### `nn.MaxPool2d` — Pooling de máximo

```python
nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
```

Divide el mapa de activaciones en ventanas y **toma el valor máximo** de cada una. Efectos:

- Reduce la resolución espacial → menos parámetros en capas siguientes.
- Hace la red invariante a pequeñas traslaciones de los objetos.
- No tiene parámetros entrenables.

**¿Por qué MAX y no promedio?** El máximo retiene la característica más fuerte detectada. El promedio la diluiría.

```
Ejemplo con ventana 2×2:
┌─────┐
│1  3 │  →  3  (toma el máximo)
│2  1 │
└─────┘
```

---

### `nn.ReLU` — Función de activación

```python
nn.ReLU()
# Equivale a: f(x) = max(0, x)
```

**¿Por qué se necesita?** Sin activación, apilar capas lineales sería equivalente a una sola capa lineal. ReLU introduce **no-linealidad**, permitiendo aprender funciones complejas.

**¿Por qué ReLU y no sigmoid/tanh?**

- Sigmoid y tanh saturan (gradientes ≈ 0 para valores grandes) → *vanishing gradient*.
- ReLU no satura para valores positivos → gradientes fluyen bien.
- ReLU es más rápida de calcular.

```text
sigmoid: f(x) = 1 / (1 + e^(-x))   → satura en ambos extremos
tanh:    f(x) = (e^x - e^-x) / ...  → satura en ambos extremos
ReLU:    f(x) = max(0, x)            → no satura para x > 0
```

---

### `nn.Linear` — Capa densa (Fully Connected)

```python
nn.Linear(in_features=9216, out_features=4096)
```

Multiplica cada entrada por un peso y suma un bias:

```text
y = x · W^T + b
```

Donde `W` es una matriz de forma `(out_features, in_features)`.

**Parámetros:** `out_features × (in_features + 1)` — el `+1` es el bias.

---

### `nn.Flatten` — Aplanamiento

```python
nn.Flatten()
```

Convierte un tensor multidimensional en un vector 1D (manteniendo el batch):

```text
Tensor:  [batch, 256, 6, 6]  →  Vector: [batch, 9216]
```

Es necesario para conectar las capas convolucionales (que trabajan en 2D) con las capas lineales (que trabajan en 1D).

---

### `forward()` — El flujo de datos

```python
def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    ...
    return x
```

Define **cómo se conectan** las capas. PyTorch llama a este método automáticamente cuando haces `modelo(input)`. Es aquí donde se deben incluir las capas nuevas (como `conv6` en la Actividad 3).

---

### `requires_grad` — Parámetros entrenables

```python
p.requires_grad  # True si el parámetro se actualiza en el entrenamiento
```

Los pesos y biases de `Conv2d` y `Linear` tienen `requires_grad=True` por defecto. Las capas sin parámetros (ReLU, MaxPool, Flatten) no tienen gradientes.

---

## 13. Conceptos adicionales útiles

### `nn.Dropout` — Regularización (no está en el lab, pero AlexNet lo usa)

```python
nn.Dropout(p=0.5)  # desactiva el 50% de neuronas aleatoriamente en cada paso
```

Se usa entre capas FC para evitar **overfitting**. Durante la evaluación se desactiva automáticamente.

---

### `model.train()` vs `model.eval()`

```python
modelo.train()  # activa Dropout y BatchNorm en modo entrenamiento
modelo.eval()   # desactiva Dropout, BatchNorm usa estadísticas acumuladas
```

Siempre usar `.eval()` al evaluar/inferir con el modelo.

---

### `torch.no_grad()` — Evaluación sin gradientes

```python
with torch.no_grad():
    output = modelo(input)
```

Desactiva el cálculo de gradientes. Más rápido y consume menos memoria. Siempre usar al evaluar o predecir, nunca al entrenar.

---

## 14. Función auxiliar: `contar_parametros`

```python
def contar_parametros(modelo):
    contador = 0
    for nombre, modulo in modelo.named_children():
        params = sum(p.numel() for p in modulo.parameters() if p.requires_grad)
        print("Cantidad Parámetros Capa '{}': {}.".format(nombre, params))
        contador += params
    print("La cantidad total de parámetros es: {}.".format(contador))
    return contador
```

Uso para verificar cualquier modelo:

```python
modelo = MiAlexNet()
contar_parametros(modelo)
```
