# Conceptos Fundamentales de CNNs — Preguntas y Respuestas
**Diplomado IA UC — Clase 9 | Notas de estudio**

> Este documento resuelve las dudas conceptuales más comunes sobre cómo operan internamente las redes convolucionales. Está escrito de forma didáctica, con ejemplos numéricos concretos.

---

## Tabla de Contenidos

1. [¿Qué es un mapa de activación?](#1-qué-es-un-mapa-de-activación)
2. [¿Qué significa `conv3-64`?](#2-qué-significa-conv3-64)
3. [¿Qué significa `224×224×64`?](#3-qué-significa-224×224×64)
4. [¿De dónde salen los 64 canales?](#4-de-dónde-salen-los-64-canales)
5. [¿Los 64 filtros no producen lo mismo si recorren la misma imagen?](#5-los-64-filtros-no-producen-lo-mismo)
6. [¿El número de filtros depende de cuántas posiciones caben?](#6-número-de-filtros-vs-posiciones)
7. [¿Qué reduce el tamaño espacial y qué cambia los canales?](#7-qué-reduce-el-espacio-y-qué-cambia-canales)
8. [¿Cómo crece el campo receptivo capa a capa?](#8-campo-receptivo-capa-a-capa)
9. [¿Las capas convolucionales clasifican?](#9-las-capas-convolucionales-clasifican)
10. [¿Para qué son las capas FC intermedias y por qué 4096?](#10-las-capas-fc-intermedias)
11. [¿Qué es Global Average Pooling y por qué reemplaza a las FC?](#11-global-average-pooling)
12. [Resumen visual: el flujo completo de VGG-16](#12-flujo-completo-vgg-16)

---

## 1. ¿Qué es un mapa de activación?

Un mapa de activación es el resultado de **un filtro recorriendo toda la imagen** (o toda la salida de la capa anterior). Es una grilla 2D donde cada valor indica "cuánto se parece esta región al patrón que busca el filtro".

```
Imagen 224×224×3
       │
       │  Filtro #0 (3×3×3) pasa por TODAS las 224×224 posiciones
       │  En cada posición: mira los 3 canales RGB, multiplica, suma → 1 número
       │
       ▼
Mapa de activación #0: 224×224 valores

  Valor alto (ej: 8.5) → "encontré el patrón aquí"
  Valor bajo (ej: 0.1) → "no hay nada relevante aquí"
  Valor negativo → ReLU lo pone en 0 → "definitivamente no"
```

Cada filtro hace un **barrido completo** de toda la entrada. Un filtro siempre produce exactamente **un** mapa de activación.

---

## 2. ¿Qué significa `conv3-64`?

Es una notación abreviada del paper de VGG:

```
conv3-64
  │   │
  │   └── 64 filtros (= 64 canales de salida)
  └────── kernel de 3×3
```

Significa: "una capa convolucional con filtros de 3×3 que tiene 64 filtros".

Otros ejemplos de la misma notación:

| Notación | Kernel | Filtros | Canales de salida |
|----------|--------|---------|-------------------|
| conv3-64 | 3×3 | 64 | 64 |
| conv3-128 | 3×3 | 128 | 128 |
| conv3-512 | 3×3 | 512 | 512 |
| conv7-64 (ResNet) | 7×7 | 64 | 64 |
| conv1-16 (Inception) | 1×1 | 16 | 16 |

---

## 3. ¿Qué significa `224×224×64`?

Pensemos primero en la imagen original:

```
Una foto RGB tiene 3 "capas" de píxeles:

224×224 píxeles ROJOS      ← canal 0
224×224 píxeles VERDES     ← canal 1
224×224 píxeles AZULES     ← canal 2

Total: 224 × 224 × 3
       ancho  alto  canales
```

Después de `conv3-64`, los 3 canales de color se transforman en **64 canales de features**:

```
224×224 mapa de activación #0     ← lo que detectó el filtro 0 (ej: bordes verticales)
224×224 mapa de activación #1     ← lo que detectó el filtro 1 (ej: bordes horizontales)
224×224 mapa de activación #2     ← lo que detectó el filtro 2 (ej: esquinas)
...
224×224 mapa de activación #63    ← lo que detectó el filtro 63 (ej: transiciones de color)

Total: 224 × 224 × 64
       ancho  alto  canales (uno por filtro)
```

**En vez de 3 "capas" de color, ahora tienes 64 "capas" de features detectadas.** Cada capa es el resultado de un filtro distinto recorriendo toda la imagen.

---

## 4. ¿De dónde salen los 64 canales?

De los **64 filtros** que definimos en la capa. El número 64 es una **decisión del diseñador**, no un cálculo. Simonyan (autor de VGG) eligió 64 porque experimentalmente funcionaba bien.

```
Filtro #0:  busca bordes horizontales     → produce 1 mapa de 224×224
Filtro #1:  busca bordes verticales       → produce 1 mapa de 224×224
Filtro #2:  busca bordes diagonales       → produce 1 mapa de 224×224
Filtro #3:  busca transiciones de color   → produce 1 mapa de 224×224
...
Filtro #63: busca algún otro patrón       → produce 1 mapa de 224×224

Se apilan los 64 mapas → tensor 224×224×64
```

Es un hiperparámetro, como el learning rate. Podría haber sido 32, 128 o 1000. Lo que distintas arquitecturas eligieron:

```
AlexNet (2012):  96 filtros en la primera capa
VGG (2014):      64 filtros
ResNet (2015):   64 filtros
MobileNet:       32 filtros  ← optimizado para móviles
```

El patrón de VGG es doblar los canales en cada bloque:

```
Bloque 1:   64 filtros   ← suficientes para patrones básicos (bordes, colores)
Bloque 2:  128 filtros   ← más combinaciones de patrones
Bloque 3:  256 filtros   ← texturas y formas
Bloque 4:  512 filtros   ← partes de objetos
Bloque 5:  512 filtros   ← objetos complejos
```

---

## 5. ¿Los 64 filtros no producen lo mismo?

No, porque cada filtro tiene **pesos distintos**. Los 64 filtros recorren exactamente la misma imagen, pero cada uno tiene sus propios valores en el kernel 3×3×3. Esos pesos son lo que la red **aprende** durante el entrenamiento.

### Ejemplo concreto en una posición

Los 64 filtros miran el mismo parche de 3×3 píxeles en posición (100, 50):

```
Parche de la imagen:

Canal R:  [120, 130, 125]
          [128, 135, 130]
          [122, 127, 123]

(+ canales G y B, 27 valores en total)
```

Cada filtro tiene pesos diferentes y produce un resultado diferente:

```
Filtro #0 (aprendió a detectar bordes verticales):
Pesos R:  [-1,  0,  1]
          [-1,  0,  1]
          [-1,  0,  1]
...
Resultado: Σ(27 productos) = 12.3    ← "hay un borde vertical aquí"


Filtro #1 (aprendió a detectar bordes horizontales):
Pesos R:  [-1, -1, -1]
          [ 0,  0,  0]
          [ 1,  1,  1]
...
Resultado: Σ(27 productos) = -0.5    ← "no hay borde horizontal aquí"


Filtro #2 (aprendió a detectar zonas brillantes):
Pesos R:  [0.1, 0.1, 0.1]
          [0.1, 0.1, 0.1]
          [0.1, 0.1, 0.1]
...
Resultado: Σ(27 productos) = 85.7    ← "esta zona es brillante"
```

**Misma imagen, misma posición, 3 resultados completamente distintos.**

### ¿De dónde salen esos pesos?

Al inicio del entrenamiento son **aleatorios**. Después del entrenamiento con backpropagation, cada filtro converge a detectar un patrón útil para la tarea. Nadie programa "filtro #0 = bordes verticales". Eso **emerge** del entrenamiento.

---

## 6. Número de filtros vs. posiciones

Son dos cosas completamente independientes:

| Concepto | Qué determina | Quién lo decide |
|----------|--------------|-----------------|
| **Posiciones** (224×224) | El tamaño espacial de la salida (H × W) | El tamaño de la imagen + stride + padding |
| **Filtros** (64) | La cantidad de canales de salida | El diseñador de la arquitectura |

```
Tamaño de salida espacial (cuántas posiciones caben):
  H_out = (224 + 2×padding - kernel_size) / stride + 1
  H_out = (224 + 2×1 - 3) / 1 + 1 = 224

Canales de salida:
  = número de filtros que el diseñador eligió = 64
```

### Analogía

Imagina una foto impresa de 224×224 y una lupa de 3×3:

- **Una persona** pasa la lupa por toda la foto buscando "bordes verticales" → anota en una hoja de 224×224 dónde encontró bordes → **1 mapa**
- Si pones **64 personas**, cada una con instrucciones distintas ("tú busca bordes", "tú busca colores cálidos", "tú busca texturas"), cada una produce **su propio mapa** de 224×224
- Apilas las 64 hojas → `224×224×64`

El número 64 es **cuántas personas (filtros) decidiste contratar**, no cuántas veces cabe la lupa en la foto.

---

## 7. ¿Qué reduce el espacio y qué cambia canales?

| Operación | Efecto en H×W (espacio) | Efecto en Canales |
|-----------|------------------------|-------------------|
| Conv stride=1, pad=1 | **No cambia** | Lo define `out_channels` (número de filtros) |
| Conv stride=2 | **÷2** | Lo define `out_channels` |
| MaxPool 2×2, stride=2 | **÷2** | **No cambia** |
| Conv 1×1 | **No cambia** | Cambia canales (reduce o expande) |
| Global Average Pool | **H×W → 1×1** | **No cambia** |
| Flatten | N/A | Aplana todo en un vector 1D |

**Regla clave:** Las convoluciones controlan los canales; el stride y el pooling controlan el espacio.

### En VGG-16

Los canales **solo crecen** (64 → 128 → 256 → 512 → 512). Lo que se achica es el mapa espacial:

```
conv3-64, conv3-64       cada filtro recorre 224×224 posiciones → mapa 224×224
MaxPool 2×2              ───────────────────────────────────── ahora 112×112

conv3-128, conv3-128     cada filtro recorre 112×112 posiciones → mapa 112×112
MaxPool 2×2              ───────────────────────────────────── ahora 56×56

conv3-256 ×3             cada filtro recorre 56×56 posiciones
MaxPool 2×2              ───────────────────────────────────── ahora 28×28

conv3-512 ×3             cada filtro recorre 28×28 posiciones
MaxPool 2×2              ───────────────────────────────────── ahora 14×14

conv3-512 ×3             cada filtro recorre 14×14 posiciones
MaxPool 2×2              ───────────────────────────────────── ahora 7×7
```

El MaxPool es el que va **comprimiendo el espacio**, forzando a la red a resumir información espacial en representaciones cada vez más abstractas.

---

## 8. Campo receptivo capa a capa

Cada MaxPool reduce el mapa espacial, pero eso significa que en las últimas capas, cada filtro tiene un campo receptivo que cubre **prácticamente toda la imagen original**.

```
Capa 1:   cada filtro ve 3×3 píxeles       → "hay un borde aquí"
Capa 3:   cada filtro ve 7×7 píxeles       → "hay una textura aquí"
Capa 5:   cada filtro ve ~receptive field completo → "hay un perro en la imagen"
```

Y en la última capa tienes **512 de estos detectores**, cada uno con vista de toda la imagen:

```
Filtro #0:   vio toda la imagen → "¿hay un perro?"
Filtro #1:   vio toda la imagen → "¿hay una rueda?"
Filtro #2:   vio toda la imagen → "¿hay un ojo?"
Filtro #3:   vio toda la imagen → "¿hay pelaje?"
...
Filtro #511: vio toda la imagen → "¿hay un fondo de pasto?"
```

### La transición de "dónde" a "qué"

```
Capas tempranas:  Alta resolución × Pocos canales   →  "DÓNDE están los bordes"
Capas profundas:  Baja resolución × Muchos canales   →  "QUÉ objeto hay"
```

---

## 9. ¿Las capas convolucionales clasifican?

**No.** Las capas convolucionales extraen features. La clasificación ocurre solo al final.

### ¿Qué hace cada componente?

| Componente | ¿Clasifica? | ¿Qué hace? |
|-----------|------------|------------|
| Conv + ReLU | No | Detecta patrones, produce mapas de features |
| MaxPool | No | Reduce resolución espacial |
| BatchNorm | No | Estabiliza distribución de activaciones |
| FC + Softmax (final) | **Sí** | Combina features → probabilidades por clase |
| Aux classifiers (Inception) | Solo en training | Inyecta gradiente, se descarta en inferencia |

### ¿Qué hace ReLU?

ReLU no clasifica. Solo pone los negativos en cero:

```
Valor del filtro: -3.7  →  ReLU → 0    ("no encontré este patrón aquí")
Valor del filtro:  5.2  →  ReLU → 5.2  ("sí encontré este patrón aquí")
```

No hay probabilidades, no hay clases. Solo "encontré / no encontré".

### ¿Y los clasificadores auxiliares de GoogLeNet?

GoogLeNet tiene 3 softmax, pero los 2 intermedios **solo existen durante el entrenamiento** para inyectar gradiente en capas intermedias y combatir el vanishing gradient:

```
Loss = Loss_principal + 0.3 × Loss_aux1 + 0.3 × Loss_aux2
```

En inferencia (predicción), los auxiliares se ignoran completamente. Solo se usa el softmax final.

### El flujo real

```
Capas conv (entiende la imagen)         Capas FC (decide qué es)
───────────────────────────────         ────────────────────────

224×224×3 (foto de un gato)             7×7×512 = 25,088 features
    ↓ conv + ReLU + pool                     ↓ flatten
    ↓ conv + ReLU + pool                FC: 25,088 → 4,096
    ↓ conv + ReLU + pool                     ↓ ReLU
    ↓ conv + ReLU + pool                FC: 4,096 → 4,096
    ↓ conv + ReLU + pool                     ↓ ReLU
7×7×512                                 FC: 4,096 → 1,000
                                             ↓ Softmax
"512 mapas describiendo                 [0.01, 0.02, ..., 0.85, ..., 0.001]
 lo que hay en la imagen"                "85% probabilidad de ser gato"
```

La red primero **entiende** la imagen (conv layers) y después **decide** qué es (FC + softmax).

---

## 10. Las capas FC intermedias

### ¿Por qué no saltar directo de 25,088 a 1,000?

Podrías, pero una sola capa lineal solo puede hacer sumas ponderadas simples. Las capas intermedias permiten aprender **combinaciones no-lineales** entre features:

```
Capa FC1 (25,088 → 4,096):
  "Si hay pelaje (feature #23) Y hay ojos redondos (feature #187)
   Y NO hay ruedas (feature #401) → activar neurona #52
   que representa 'probablemente un animal'"

Capa FC2 (4,096 → 4,096):
  "Si 'probablemente un animal' (#52) Y 'color dorado' (#88)
   Y 'tamaño grande' (#201) → activar neurona #77
   que representa 'perro dorado grande'"

Capa final (4,096 → 1,000):
  neurona #77 contribuye fuertemente a la clase "golden retriever"
```

### ¿Por qué 4,096 específicamente?

Viene de **AlexNet** (2012). Krizhevsky eligió 4,096 y funcionó bien. VGG simplemente copió el patrón. No hay justificación matemática profunda.

```
AlexNet (2012):  9,216 → 4,096 → 4,096 → 1,000   ← eligió 4,096
VGG (2014):     25,088 → 4,096 → 4,096 → 1,000   ← copió el patrón
```

### El gran problema: el 89% de los parámetros de VGG están aquí

```
Capa                     Parámetros       % del total
──────────────────────────────────────────────────────
Todas las conv (capas 1-13)  14.7M          10.6%
FC1: 25,088 × 4,096        102.8M          74.3%    ← ¡la mayoría!
FC2: 4,096 × 4,096          16.8M          12.1%
FC3: 4,096 × 1,000           4.1M           3.0%
──────────────────────────────────────────────────────
Total                       138.4M         100.0%
```

Las capas FC son enormemente ineficientes. Por eso las arquitecturas posteriores las eliminaron.

---

## 11. Global Average Pooling

### El problema

Después de las convoluciones tienes un tensor de `7×7×512`. Necesitas convertirlo en un vector para clasificar. Hay dos formas:

### Forma 1: Flatten (VGG)

Aplanar todo en una fila:

```
Canal 0:  [v0, v1, v2, ..., v48]     ← 49 valores (7×7)
Canal 1:  [v49, v50, ..., v97]       ← 49 valores
...
Canal 511: [v24,990, ..., v25,087]   ← 49 valores

Flatten → [v0, v1, ..., v25,087]     ← vector de 25,088

Luego FC de 25,088 × 4,096 = 102 MILLONES de parámetros
```

### Forma 2: Global Average Pooling (ResNet, Inception)

En vez de conservar los 49 valores de cada canal, **promediarlos en uno solo**:

```
Canal 0 (7×7 valores):
┌────┬────┬────┬────┬────┬────┬────┐
│ 2.1│ 0.5│ 3.2│ 1.1│ 0.0│ 2.8│ 1.5│
├────┼────┼────┼────┼────┼────┼────┤
│ 1.8│ 0.3│ 4.1│ 0.9│ 0.2│ 3.1│ 1.2│
├────┼────┼────┼────┼────┼────┼────┤
│ 0.7│ 1.2│ 2.5│ 0.8│ 1.1│ 2.0│ 0.9│
├────┼────┼────┼────┼────┼────┼────┤
│ ...│    │    │    │    │    │    │
└────┴────┴────┴────┴────┴────┴────┘

Promedio de los 49 valores = 1.73  ← UN solo número para todo el canal

Canal 1:   promedio = 0.42
Canal 2:   promedio = 3.88
...
Canal 511: promedio = 2.15

Resultado: vector de 512 valores (uno por canal)
```

### Comparación directa

```
VGG (flatten):       7×7×512  →  vector de 25,088  →  FC de 102M params
ResNet (avg pool):   7×7×512  →  vector de 512     →  FC de 0.5M params
                                                        ↑
                                                    ~200× menos parámetros
```

### ¿No se pierde información al promediar?

Se pierde la información de **dónde** exactamente dentro del mapa 7×7 estaba la activación. Pero a esta altura de la red, lo que importa es **si** el patrón está presente, no **dónde**:

```
Canal #23 detecta "ojos de perro":

Con ojos en la imagen:
┌────┬────┬────┬────┬────┬────┬────┐
│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│
│ 0.0│ 0.0│ 8.5│ 0.0│ 7.2│ 0.0│ 0.0│   ← activaciones altas
│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│
│ ...│    │    │    │    │    │    │
└────┴────┴────┴────┴────┴────┴────┘
Promedio = 0.32  → "hay algo de ojos de perro en la imagen"


Sin ojos en la imagen:
┌────┬────┬────┬────┬────┬────┬────┐
│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│
│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│ 0.0│
│ ...│    │    │    │    │    │    │
└────┴────┴────┴────┴────┴────┴────┘
Promedio = 0.0  → "no hay ojos de perro"
```

El promedio distingue "presente" vs "ausente". Para clasificar, eso es suficiente.

### ¿Por qué todas las arquitecturas modernas usan esto?

Porque elimina las capas FC gigantes sin perder accuracy:

```
VGG (2014):      flatten → FC 4,096 → FC 4,096 → FC 1,000   (122M params en FC)
GoogLeNet (2014): AvgPool → FC 1,000                          (1M params en FC)
ResNet (2015):    AvgPool → FC 1,000                          (2M params en FC)
EfficientNet:     AvgPool → FC 1,000                          (~1M params en FC)
```

Es una de las lecciones más importantes de GoogLeNet que todas las arquitecturas posteriores adoptaron.

---

## 12. Flujo completo de VGG-16

```
═══════════════════════════════════════════════════════════════════════
ENTRADA: Foto RGB de 224×224×3
═══════════════════════════════════════════════════════════════════════

BLOQUE 1: "Detectar bordes y colores básicos"
  conv3-64:  64 filtros de 3×3×3    → 224×224×64   (27 pesos por filtro)
  ReLU
  conv3-64:  64 filtros de 3×3×64   → 224×224×64   (576 pesos por filtro)
  ReLU
  MaxPool 2×2                       → 112×112×64
───────────────────────────────────────────────────────────────────────

BLOQUE 2: "Detectar texturas simples"
  conv3-128: 128 filtros de 3×3×64  → 112×112×128
  ReLU
  conv3-128: 128 filtros de 3×3×128 → 112×112×128
  ReLU
  MaxPool 2×2                       → 56×56×128
───────────────────────────────────────────────────────────────────────

BLOQUE 3: "Detectar texturas complejas y formas"
  conv3-256 ×3                      → 56×56×256
  MaxPool 2×2                       → 28×28×256
───────────────────────────────────────────────────────────────────────

BLOQUE 4: "Detectar partes de objetos"
  conv3-512 ×3                      → 28×28×512
  MaxPool 2×2                       → 14×14×512
───────────────────────────────────────────────────────────────────────

BLOQUE 5: "Detectar objetos completos"
  conv3-512 ×3                      → 14×14×512
  MaxPool 2×2                       → 7×7×512
───────────────────────────────────────────────────────────────────────

TRANSICIÓN: De mapas espaciales a vector
  Flatten: 7×7×512                  → vector de 25,088
───────────────────────────────────────────────────────────────────────

CLASIFICACIÓN: "Decidir qué objeto es"
  FC1: 25,088 → 4,096 + ReLU       (102.8M parámetros)
  Dropout 0.5
  FC2: 4,096 → 4,096 + ReLU        (16.8M parámetros)
  Dropout 0.5
  FC3: 4,096 → 1,000               (4.1M parámetros)
  Softmax

═══════════════════════════════════════════════════════════════════════
SALIDA: [0.01, 0.00, ..., 0.85, ..., 0.00]
        "85% probabilidad de ser golden retriever"
═══════════════════════════════════════════════════════════════════════

Resumen:
  Espacio:   224 → 112 → 56 → 28 → 14 → 7    (se achica ×32)
  Canales:     3 →  64 → 128 → 256 → 512 → 512 (crece)
  Significado: píxeles → bordes → texturas → partes → objetos → clase
```

---

## Conceptos clave para recordar

| Pregunta | Respuesta |
|---------|-----------|
| ¿Quién define los canales de salida? | El número de filtros (decisión del diseñador) |
| ¿Quién define el tamaño espacial? | El stride, padding y MaxPool (cálculo matemático) |
| ¿Los filtros dan el mismo resultado? | No — cada filtro tiene pesos distintos aprendidos |
| ¿Las conv clasifican? | No — solo extraen features. La clasificación es FC + Softmax |
| ¿Por qué 4,096 en las FC? | Decisión empírica de AlexNet, VGG lo copió |
| ¿Qué es Global Average Pooling? | Promediar cada canal en un solo número, eliminando las FC gigantes |
| ¿Por qué se doblan los canales? | Para compensar la pérdida de información espacial del MaxPool |
| ¿Qué pasa capa a capa? | El espacio se achica, los canales crecen: de "dónde" a "qué" |
