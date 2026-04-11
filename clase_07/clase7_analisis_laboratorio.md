# Laboratorio 7 — Analisis paso a paso

Analisis celda por celda del notebook del Laboratorio 7: PyTorch.

---

## Celda 1: El Tensor — la unidad fundamental

### Codigo

```python
import torch

batch_dim = 2
ancho = 4
alto = 3
tensor_ejemplo = torch.randn((batch_dim, ancho, alto)).float()
print("El tensor es: {}\n".format(tensor_ejemplo))
print("La forma del tensor es: {}".format(tensor_ejemplo.shape))
print("El tipo del tensor es: {}".format(type(tensor_ejemplo)))
```

### Salida

```text
El tensor es: tensor([[[-1.1938,  0.2638,  0.5017],
         [ 0.5649, -1.1356,  0.1043],
         [-0.2367,  1.1050, -0.1637],
         [-0.7329,  0.5200, -0.3071]],

        [[-0.4596, -0.6214, -1.7263],
         [-0.1324, -1.1799, -0.2345],
         [-0.0471,  0.0234,  1.0838],
         [-0.3815, -1.1738,  0.1757]]])

La forma del tensor es: torch.Size([2, 4, 3])
El tipo del tensor es: <class 'torch.Tensor'>
```

### Analisis linea por linea

---

#### `import torch`

Importa la libreria **PyTorch**. Todo en PyTorch empieza con `torch`. Es como importar `numpy` pero para redes neuronales.

---

#### `batch_dim = 2`, `ancho = 4`, `alto = 3`

Son solo variables de Python con numeros. Definen las **dimensiones** del tensor que vamos a crear. Los nombres son para que sea legible, pero a PyTorch le da igual como se llamen.

---

#### `torch.randn((batch_dim, ancho, alto))`

**`torch.randn`** crea un tensor lleno de numeros **aleatorios** con distribucion normal (media=0, desviacion estandar=1). Los valores tipicos caen entre -3 y +3.

El argumento es una **tupla con la forma** (shape) del tensor:

```text
torch.randn((2, 4, 3))
              ↑  ↑  ↑
              │  │  └─ dimension 2: tamaño 3 (alto)
              │  └──── dimension 1: tamaño 4 (ancho)
              └─────── dimension 0: tamaño 2 (batch)

Total de numeros: 2 × 4 × 3 = 24 numeros aleatorios
```

Otros metodos para crear tensores:

```text
torch.randn((2,4,3))    → aleatorios distribucion normal (media=0, std=1)
torch.rand((2,4,3))     → aleatorios uniformes entre 0 y 1
torch.zeros((2,4,3))    → todos ceros
torch.ones((2,4,3))     → todos unos
torch.empty((2,4,3))    → sin inicializar (basura en memoria, mas rapido)
torch.full((2,4,3), 7)  → todos con valor 7
torch.arange(10)        → [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
torch.linspace(0, 1, 5) → [0.0, 0.25, 0.5, 0.75, 1.0]
torch.tensor([1, 2, 3]) → desde una lista de Python
```

---

#### `.float()`

Convierte el tensor al tipo **float32** (32 bits por numero). En este caso es redundante porque `torch.randn` ya crea tensores float32 por defecto, pero es buena practica ser explicito.

Tipos de datos en PyTorch:

```text
.float()    → float32   (el mas comun, usado para entrenar redes)
.double()   → float64   (mas precision, mas lento, raro en deep learning)
.half()     → float16   (menos precision, mas rapido en GPU)
.int()      → int32     (enteros)
.long()     → int64     (enteros grandes, usado para etiquetas/labels)
.bool()     → booleano  (True/False)
```

En la practica casi siempre usas `.float()` para los datos y `.long()` para las etiquetas.

---

#### La salida del tensor

```text
tensor([[[-1.1938,  0.2638,  0.5017],     ← batch 0, fila 0
         [ 0.5649, -1.1356,  0.1043],     ← batch 0, fila 1
         [-0.2367,  1.1050, -0.1637],     ← batch 0, fila 2
         [-0.7329,  0.5200, -0.3071]],    ← batch 0, fila 3

        [[-0.4596, -0.6214, -1.7263],     ← batch 1, fila 0
         [-0.1324, -1.1799, -0.2345],     ← batch 1, fila 1
         [-0.0471,  0.0234,  1.0838],     ← batch 1, fila 2
         [-0.3815, -1.1738,  0.1757]]])   ← batch 1, fila 3
```

Visualmente, son **2 "paginas"** (batch), cada una con **4 filas** y **3 columnas**:

```text
Batch 0 (primera "pagina"):         Batch 1 (segunda "pagina"):
  ┌─────────────────────────┐       ┌─────────────────────────┐
  │ -1.19   0.26   0.50     │       │ -0.46  -0.62  -1.73     │
  │  0.56  -1.14   0.10     │       │ -0.13  -1.18  -0.23     │
  │ -0.24   1.11  -0.16     │       │ -0.05   0.02   1.08     │
  │ -0.73   0.52  -0.31     │       │ -0.38  -1.17   0.18     │
  └─────────────────────────┘       └─────────────────────────┘
       4 filas × 3 columnas              4 filas × 3 columnas
```

---

#### `.shape`

Devuelve la **forma** (dimensiones) del tensor como `torch.Size`, que se comporta como una tupla:

```text
tensor_ejemplo.shape → torch.Size([2, 4, 3])

  Dimension 0: 2  (batch_dim)
  Dimension 1: 4  (ancho)
  Dimension 2: 3  (alto)
```

Propiedades relacionadas:

```python
tensor_ejemplo.shape      # torch.Size([2, 4, 3]) — la forma
tensor_ejemplo.shape[0]   # 2 — tamaño de la dimension 0
tensor_ejemplo.shape[1]   # 4 — tamaño de la dimension 1
tensor_ejemplo.ndim        # 3 — numero de dimensiones (3D)
tensor_ejemplo.numel()    # 24 — total de elementos (2×4×3)
tensor_ejemplo.size()     # torch.Size([2, 4, 3]) — igual que .shape
tensor_ejemplo.dtype      # torch.float32 — tipo de dato
tensor_ejemplo.device     # device(type='cpu') — donde esta (CPU o GPU)
```

---

#### `type(tensor_ejemplo)`

Devuelve la **clase de Python** del objeto: `<class 'torch.Tensor'>`. Esto confirma que es un tensor de PyTorch, no una lista, un ndarray de NumPy, ni otra cosa.

No confundir con `.dtype` que es el tipo de los **datos internos**:

```text
type(tensor_ejemplo)       → <class 'torch.Tensor'>   (la clase del objeto)
tensor_ejemplo.dtype       → torch.float32             (el tipo de los numeros)
```

---

### En el contexto de redes neuronales

Este tensor de shape `(2, 4, 3)` podria representar:

```text
Como datos tabulares:
  2 muestras, cada una con 4 filas de 3 features

Como imagenes:
  2 imagenes (batch=2), de 4×3 pixeles, 1 canal
  (aunque para imagenes el formato es (batch, canales, alto, ancho))

Como texto:
  2 frases (batch=2), de 4 tokens, con embedding de 3 dimensiones
```

La dimension 0 es siempre el **batch** en PyTorch. Esto es una convencion que TODAS las capas de PyTorch esperan.

---

### Metodos utiles de un tensor (referencia rapida)

```text
Crear:
  torch.randn(shape)      numeros aleatorios (normal)
  torch.zeros(shape)      todo ceros
  torch.tensor([1,2,3])   desde lista

Informacion:
  .shape                   dimensiones
  .dtype                   tipo de datos
  .device                  CPU o GPU
  .numel()                 total de elementos
  .ndim                    numero de dimensiones

Cambiar forma:
  .reshape(new_shape)      cambiar dimensiones
  .view(new_shape)         cambiar dimensiones (misma memoria)
  .unsqueeze(dim)          agregar dimension de tamaño 1
  .squeeze()               quitar dimensiones de tamaño 1
  .flatten()               aplanar todo a 1D
  .T                       transponer (2D)
  .permute(dims)           reordenar dimensiones

Matematicas:
  .mean()                  media
  .sum()                   suma
  .std()                   desviacion estandar
  .min(), .max()           minimo, maximo
  .abs()                   valor absoluto
  .pow(2)                  potencia
  .sqrt()                  raiz cuadrada

Conversion:
  .float()                 a float32
  .long()                  a int64
  .numpy()                 a numpy array
  .tolist()                a lista de Python
  .item()                  a numero Python (solo si es 1 elemento)
  .to("cuda")              mover a GPU
  .cpu()                   mover a CPU

Comparacion:
  tensor == otro           igualdad elemento a elemento
  tensor > valor           mayor que (booleano)
  torch.allclose(a, b)     casi iguales (con tolerancia)
```

---

## Celda 2: Crear tensores desde listas y operaciones basicas

### Codigo

```python
arreglo = [[1, 2, 3], [4, 5, 6]]
tensor_desde_arreglo = torch.tensor(arreglo)
print(tensor_desde_arreglo, tensor_desde_arreglo.shape)
tensor_desde_arreglo2 = torch.tensor(arreglo)
suma = tensor_desde_arreglo +  tensor_desde_arreglo2
print(suma)
```

### Salida

```text
tensor([[1, 2, 3],
        [4, 5, 6]]) torch.Size([2, 3])
tensor([[ 2,  4,  6],
        [ 8, 10, 12]])
```

### Analisis linea por linea

---

#### `arreglo = [[1, 2, 3], [4, 5, 6]]`

Es una **lista de listas** de Python (puro, nada de PyTorch todavia). Dos listas internas de 3 elementos cada una. En Python esto no tiene operaciones matematicas utiles:

```python
# Python puro:
[[1,2,3],[4,5,6]] + [[1,2,3],[4,5,6]]
# → [[1,2,3],[4,5,6],[1,2,3],[4,5,6]]  ← CONCATENA listas, no suma numeros!
```

---

#### `torch.tensor(arreglo)`

Convierte la lista de Python a un **tensor de PyTorch**. Ahora si tiene operaciones matematicas:

```text
Lista Python:           Tensor PyTorch:
[[1, 2, 3],             tensor([[1, 2, 3],
 [4, 5, 6]]                     [4, 5, 6]])

- No se puede sumar     - Se puede sumar
- No tiene shape         - Tiene shape: (2, 3)
- No corre en GPU        - Puede correr en GPU
- Es lenta               - Es rapida (optimizada en C++)
```

**Nota sobre el tipo de dato**: como la lista tiene enteros, PyTorch crea un tensor de tipo `int64` (`.long()`), NO `float32`. Por eso la salida muestra `tensor([[1, 2, 3], ...])` sin puntos decimales. Si fueran floats:

```python
torch.tensor([[1.0, 2.0, 3.0]])  # → float32 (con decimales)
torch.tensor([[1, 2, 3]])        # → int64 (sin decimales)
```

---

#### `tensor_desde_arreglo.shape` → `torch.Size([2, 3])`

```text
Shape: (2, 3)
         ↑  ↑
         │  └─ 3 columnas (features)
         └──── 2 filas (muestras)

Es una MATRIZ de 2×3:
  ┌─────────────┐
  │  1   2   3  │  ← fila 0
  │  4   5   6  │  ← fila 1
  └─────────────┘
```

A diferencia del tensor anterior `(2, 4, 3)` que era 3D, este es **2D** (una matriz comun). No tiene dimension de batch explicita.

---

#### `tensor_desde_arreglo2 = torch.tensor(arreglo)`

Crea un **segundo tensor** con los mismos valores. Son dos objetos **independientes** en memoria, aunque tengan los mismos numeros.

```text
tensor_desde_arreglo:    tensor_desde_arreglo2:
  [[1, 2, 3],              [[1, 2, 3],
   [4, 5, 6]]               [4, 5, 6]]
  ↑                         ↑
  objeto A en memoria       objeto B en memoria (copia)
```

Si modificas uno, el otro no cambia.

---

#### `suma = tensor_desde_arreglo + tensor_desde_arreglo2`

Suma **elemento a elemento**. Cada numero se suma con su correspondiente en la misma posicion:

```text
  tensor_1:          tensor_2:          suma:
  [[1, 2, 3],    +   [[1, 2, 3],    =   [[ 2,  4,  6],
   [4, 5, 6]]        [4, 5, 6]]         [ 8, 10, 12]]

  Posicion [0,0]: 1 + 1 = 2
  Posicion [0,1]: 2 + 2 = 4
  Posicion [0,2]: 3 + 3 = 6
  Posicion [1,0]: 4 + 4 = 8
  Posicion [1,1]: 5 + 5 = 10
  Posicion [1,2]: 6 + 6 = 12
```

**Requisito**: ambos tensores deben tener la **misma forma** (o ser compatibles via broadcasting). Si no:

```python
a = torch.tensor([[1, 2, 3]])       # shape (1, 3)
b = torch.tensor([[1, 2], [3, 4]])  # shape (2, 2)
a + b  # → ERROR: las formas no coinciden
```

---

### Otras operaciones elemento a elemento

La suma es solo una de muchas. Todas funcionan igual (posicion a posicion):

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[1, 2, 3], [4, 5, 6]])

a + b    # suma:           [[ 2,  4,  6], [ 8, 10, 12]]
a - b    # resta:           [[ 0,  0,  0], [ 0,  0,  0]]
a * b    # multiplicacion:  [[ 1,  4,  9], [16, 25, 36]]
a / b    # division:        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
a ** 2   # potencia:        [[ 1,  4,  9], [16, 25, 36]]
a > 3    # comparacion:     [[False,False,False], [True,True,True]]
```

**Importante**: `a * b` es multiplicacion **elemento a elemento**, NO multiplicacion de matrices. Para multiplicar matrices se usa `a @ b` o `torch.matmul(a, b)`:

```python
# Elemento a elemento (lo que hace *)
[[1,2],[3,4]] * [[1,2],[3,4]] = [[1,4],[9,16]]

# Multiplicacion de matrices (lo que hace @)
[[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
```

---

### Diferencia clave: lista de Python vs tensor de PyTorch

```text
Operacion              Lista Python          Tensor PyTorch
──────────             ──────────────        ──────────────
[1,2] + [3,4]          [1,2,3,4]             tensor([4,6])
                       (concatena)           (suma numeros)

[1,2] * 3              [1,2,1,2,1,2]         tensor([3,6])
                       (repite)              (multiplica)

[1,2] * [3,4]          ERROR                 tensor([3,8])
                                             (multiplica elem.)

len([1,2,3])           3                     —
torch.tensor([1,2]).shape  —                 torch.Size([2])
```

La lista de Python es para guardar datos. El tensor de PyTorch es para hacer **matematicas rapidas** con esos datos.

---

## Celda 3: Multiplicacion elemento a elemento (Hadamard)

### Codigo

```python
tensor_1 = torch.randn((10,2,3))
tensor_2 = torch.randn((10,2,3))
tensor_3 = tensor_1*tensor_2
print(tensor_1.shape)
print(tensor_2)
print(tensor_3)
```

### Salida

```text
torch.Size([10, 2, 3])
tensor([[[ 0.3367,  0.1288,  0.2345],
         [ 0.2303, -1.1229, -0.1863]],

        [[-2.1787, -1.4303, -1.0485],
         [ 0.8607,  0.5088,  0.0416]],

        ...  (10 "paginas" en total)

        [[ 0.4525, -0.0025, -0.5484],
         [-0.1290, -0.6914, -0.0941]]])

tensor([[[ 0.5025, -0.0022, -0.5375],    ← tensor_1[0,0] * tensor_2[0,0]
         [-0.0895, -0.5219,  0.0241]],

        ...
```

### Analisis linea por linea

---

#### `tensor_1 = torch.randn((10,2,3))`

Crea un tensor de forma `(10, 2, 3)`. Comparado con la celda anterior que era `(2, 4, 3)`, ahora tenemos:

```text
torch.Size([10, 2, 3])
              ↑  ↑  ↑
              │  │  └─ 3 columnas (features)
              │  └──── 2 filas
              └─────── 10 elementos en el batch

Total: 10 × 2 × 3 = 60 numeros
```

Son **10 matrices de 2×3**. Visualmente:

```text
Batch 0:          Batch 1:          ...  Batch 9:
┌──────────┐      ┌──────────┐           ┌──────────┐
│ a  b  c  │      │ g  h  i  │           │ ...      │
│ d  e  f  │      │ j  k  l  │           │ ...      │
└──────────┘      └──────────┘           └──────────┘
```

En el contexto de redes neuronales, esto podria representar:
- **10 imagenes** en el batch, cada una de 2×3 pixeles
- **10 frases** con 2 tokens y embedding de 3 dimensiones
- **10 muestras** con 2 filas de 3 features

---

#### `tensor_3 = tensor_1 * tensor_2`

El operador `*` hace **multiplicacion elemento a elemento** (tambien llamada **producto de Hadamard**). Cada numero en `tensor_1` se multiplica con el numero en la **misma posicion** de `tensor_2`:

```text
tensor_1[0]:          tensor_2[0]:          tensor_3[0]:
┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
│  a   b   c   │  ×   │  d   e   f   │  =   │  a×d  b×e  c×f   │
│  g   h   i   │      │  j   k   l   │      │  g×j  h×k  i×l   │
└──────────────┘      └──────────────┘      └──────────────────┘

Esto se repite para los 10 elementos del batch.
```

Ejemplo concreto con numeros pequeños:

```text
  tensor_1[0,0] = [ 0.34,  0.13,  0.23]
  tensor_2[0,0] = [ 1.50, -0.50,  2.00]
  tensor_3[0,0] = [ 0.51, -0.07,  0.46]   ← cada par multiplicado
                   ↑       ↑       ↑
                  0.34×1.5  0.13×(-0.5)  0.23×2.0
```

**Requisito**: ambos tensores deben tener **exactamente la misma forma** (o ser compatibles via broadcasting).

---

#### `*` vs `@`: la diferencia critica

Esta es una confusion comun en PyTorch. Son DOS operaciones completamente distintas:

```text
Operador  Nombre                   Que hace
────────  ───────────────────────  ────────────────────────────────────
  *       Hadamard (elem. a elem.) Multiplica posicion a posicion
  @       Matmul (matriz)          Multiplicacion algebraica de matrices
```

Para verlo con numeros:

```python
a = torch.tensor([[1., 2.],
                  [3., 4.]])

b = torch.tensor([[1., 2.],
                  [3., 4.]])

# Elemento a elemento (*):
a * b
# → [[1×1, 2×2],   = [[1,  4],
#    [3×3, 4×4]]      [9, 16]]

# Multiplicacion de matrices (@):
a @ b
# → [[1×1+2×3, 1×2+2×4],   = [[7, 10],
#    [3×1+4×3, 3×2+4×4]]      [15, 22]]
```

Para tensores 3D (como los de esta celda), `@` hace la multiplicacion de matrices en las ultimas 2 dimensiones y aplica sobre el batch:

```text
tensor_1 @ tensor_2
  (10, 2, 3) @ (10, 2, 3)  → ERROR: formas incompatibles para matmul
  (10, 2, 3) @ (10, 3, 2)  → resultado (10, 2, 2)  ← necesita transponer
```

---

#### ¿Cuando se usa `*` en redes neuronales?

La multiplicacion elemento a elemento aparece en muchos lugares:

```text
Donde se usa *:
  Funciones de activacion:
    sigmoid(x) * (1 - sigmoid(x))       ← derivada del sigmoid
    ReLU: x * (x > 0).float()           ← mascara de activacion

  Mecanismo de atencion:
    scores = query * key / sqrt(d)       ← scaled dot product (simplificado)

  Gating (LSTM, GRU):
    output = tanh(h) * sigmoid(gate)    ← controla cuanto pasa

  Dropout:
    x_out = x * mask                    ← apagar neuronas (mask de 0s y 1s)

  Normalizacion:
    x_out = x * gamma + beta            ← escalar y desplazar (BatchNorm/LayerNorm)
```

---

### ¿Por que shape `(10, 2, 3)` y no `(2, 3)` con un loop?

En vez de hacer un loop sobre los 10 ejemplos del batch, PyTorch aplica la operacion **sobre todo el batch a la vez**:

```python
# Forma LENTA (no se hace):
resultado = []
for i in range(10):
    resultado.append(tensor_1[i] * tensor_2[i])
resultado = torch.stack(resultado)

# Forma RAPIDA (PyTorch):
resultado = tensor_1 * tensor_2   # aplica sobre el batch entero
```

Ambas dan el mismo resultado, pero la segunda es:
- **Rapida**: aprovecha operaciones vectorizadas en CPU/GPU
- **Concisa**: una sola linea
- **Eficiente en memoria**: no crea listas intermedias

Esta es la razon por la que trabajamos con tensores 3D en vez de matrices 2D: el batch esta incluido como una dimension mas.

---

### Resumen de operadores en PyTorch

```text
Operador  Nombre                Formas compatibles
────────  ──────────────────    ─────────────────────────────
a + b     Suma elem. a elem.    Misma forma (o broadcastable)
a - b     Resta elem. a elem.   Misma forma (o broadcastable)
a * b     Mult. elem. a elem.   Misma forma (o broadcastable)
a / b     Division elem. a elem Misma forma (o broadcastable)
a @ b     Mult. de matrices     (..., n, k) @ (..., k, m) → (..., n, m)
a ** n    Potencia              Cualquier tensor
a.T       Transpuesta           Solo 2D
a.permute Reordenar dims        Cualquier tensor ND
```

La regla simple:
- **`*`** = cada elemento con su pareja en la misma posicion
- **`@`** = multiplicacion algebraica de matrices (filas × columnas)

---

## Celda 4: Indexacion y slicing de tensores

### Codigo

```python
print(tensor_1.shape)

lista = [1, 2, 3, 4]

print(lista[0:2])

tensor_indexado = tensor_1[0:2, 0, 0]
print(tensor_indexado, tensor_indexado.shape)
```

### Salida

```text
torch.Size([10, 2, 3])
[1, 2]
tensor([-0.2455, -0.1349]) torch.Size([2])
```

### Analisis linea por linea

---

#### `lista[0:2]` → `[1, 2]`

Primero se muestra como funciona el **slicing en Python puro** para comparar con PyTorch despues.

La notacion `[inicio:fin]` selecciona elementos desde `inicio` hasta `fin` (sin incluirlo):

```text
lista = [1, 2, 3, 4]
         ↑  ↑  ↑  ↑
         0  1  2  3   ← indices

lista[0:2]  → indices 0 y 1 → [1, 2]
lista[1:3]  → indices 1 y 2 → [2, 3]
lista[2:]   → desde indice 2 hasta el final → [3, 4]
lista[:2]   → desde el inicio hasta indice 1 → [1, 2]
lista[-1]   → ultimo elemento → 4
lista[-2:]  → ultimos 2 elementos → [3, 4]
```

---

#### `tensor_1[0:2, 0, 0]`

Aqui se aplica **indexacion multidimensional** al tensor de shape `(10, 2, 3)`. La coma separa los indices de cada dimension:

```text
tensor_1[  0:2  ,  0  ,  0  ]
            ↑       ↑     ↑
         dim 0    dim 1  dim 2
         (batch)  (fila) (col)

  dim 0: 0:2  → slice: toma batches 0 y 1  (mantiene la dimension)
  dim 1: 0    → entero: toma solo fila 0    (elimina la dimension)
  dim 2: 0    → entero: toma solo columna 0 (elimina la dimension)
```

Paso a paso:

```text
tensor_1 tiene shape (10, 2, 3):

  Batch 0:              Batch 1:              ...  Batch 9:
  ┌──────────────┐      ┌──────────────┐           ┌──────────────┐
  │ X  .  .      │      │ Y  .  .      │           │ .  .  .      │
  │ .  .  .      │      │ .  .  .      │           │ .  .  .      │
  └──────────────┘      └──────────────┘           └──────────────┘
    ↑ fila 0, col 0       ↑ fila 0, col 0

tensor_1[0:2, 0, 0] = [X, Y]  ← solo esos dos valores
```

Resultado: `tensor([-0.2455, -0.1349])` con shape `(2,)`.

---

#### Por que el resultado tiene shape `(2,)` y no `(2, 1, 1)`

Esta es la regla clave del indexado:

```text
Tipo de indice     Efecto en la forma
──────────────     ──────────────────────────────────────────────
slice (0:2)        MANTIENE la dimension (con el tamaño del slice)
entero (0)         ELIMINA la dimension

tensor_1[0:2, 0, 0]:
  dim 0: slice → dimension que queda, tamaño 2
  dim 1: entero → dimension eliminada
  dim 2: entero → dimension eliminada

Resultado: (2,)   ← solo queda la dimension del slice
```

Si en cambio usaras slices en todas las dimensiones:

```python
tensor_1[0:2, 0:1, 0:1]  # slice en todas → shape (2, 1, 1)
tensor_1[0:2, :, :]       # todos en dim 1 y 2 → shape (2, 2, 3)
tensor_1[0, :, :]         # entero en dim 0 → shape (2, 3)
tensor_1[0, 0, :]         # entero en dim 0 y 1 → shape (3,)
tensor_1[0, 0, 0]         # entero en todas → escalar (shape [])
```

---

#### Formas de indexar un tensor `(10, 2, 3)`

```text
Expresion              Shape resultado   Descripcion
─────────────────────  ───────────────   ──────────────────────────────
tensor_1[0]            (2, 3)            primer batch completo
tensor_1[0:2]          (2, 2, 3)         primeros 2 batches
tensor_1[:, 0]         (10, 3)           fila 0 de todos los batches
tensor_1[:, :, 0]      (10, 2)           columna 0 de todos los batches
tensor_1[0, 0, 0]      () escalar        un numero especifico
tensor_1[0:2, 0, 0]    (2,)              col 0, fila 0, batches 0 y 1
tensor_1[-1]           (2, 3)            ultimo batch
tensor_1[::2]          (5, 2, 3)         cada 2 batches (0,2,4,6,8)
```

---

#### ¿Por que importa el indexado en redes neuronales?

```text
Caso de uso                          Codigo tipico
─────────────────────────────────    ──────────────────────────────────
Tomar el primer elemento del batch   output[0]
Tomar las predicciones finales       output[:, -1, :]   (ultimo token)
Separar clases del output            logits[:, clase_idx]
Acceder a una capa especifica        features[:, layer, :]
Tomar solo los primeros k ejemplos   X[:k], y[:k]
Saltar una dimension                 tensor[None, :, :]  (agrega dim)
```

Ejemplo practico: en un modelo de texto con output shape `(batch, tokens, vocab)`:

```python
# Predecir el proximo token: solo nos importa el ultimo token de cada secuencia
ultimo_token = output[:, -1, :]   # shape (batch, vocab)
prediccion = ultimo_token.argmax(dim=-1)   # shape (batch,)
```

---

### Comparacion: slicing en lista Python vs tensor PyTorch

```text
Operacion           Lista Python         Tensor PyTorch
─────────────────   ─────────────        ─────────────────────────
lista[0]            primer elemento      tensor[0] → reduce 1 dim
lista[0:2]          sublista [e0, e1]    tensor[0:2] → mantiene dim
lista[-1]           ultimo elemento      tensor[-1] → reduce 1 dim
lista[:, 0]         ERROR                tensor[:, 0] → multidim OK
lista[mask]         ERROR                tensor[mask] → indexado booleano
```

Los tensores soportan indexado **multidimensional** con comas, algo imposible en listas de Python. Esta es una de las ventajas fundamentales de usar PyTorch.

---

## Celda 5: El formato estandar de imagenes en PyTorch

### Codigo

```python
torch.randn(10, 3, 64, 64).shape
```

### Salida

```text
torch.Size([10, 3, 64, 64])
```

### Analisis

---

#### `torch.randn(10, 3, 64, 64)`

Esta es la primera vez que aparece un tensor **4D**. Es el formato estandar para representar imagenes en PyTorch:

```text
torch.Size([10,  3,  64,  64])
             ↑   ↑    ↑    ↑
             │   │    │    └─ ancho (W = width)
             │   │    └────── alto  (H = height)
             │   └─────────── canales de color (C = channels)
             └─────────────── imagenes en el batch (N = batch size)

Convencion: (N, C, H, W) — el estandar de PyTorch
```

En este caso concreto:
- **10** imagenes en el batch
- **3** canales de color (RGB: Rojo, Verde, Azul)
- **64×64** pixeles de resolucion

Total de numeros: 10 × 3 × 64 × 64 = **122,880 valores**

---

#### ¿Por que 3 canales?

Una imagen a color tiene 3 capas superpuestas, una por canal:

```text
Imagen RGB de 64×64:

Canal R (rojo):          Canal G (verde):         Canal B (azul):
┌────────────────┐       ┌────────────────┐       ┌────────────────┐
│ 0.8  0.2  ...  │       │ 0.1  0.9  ...  │       │ 0.4  0.3  ...  │
│ 0.3  0.7  ...  │       │ 0.6  0.2  ...  │       │ 0.8  0.5  ...  │
│ ...            │       │ ...            │       │ ...            │
└────────────────┘       └────────────────┘       └────────────────┘
      64×64                    64×64                    64×64

Cada pixel es la combinacion de los 3 valores: (R, G, B)
```

Para imagenes en escala de grises: **1 canal** → shape `(N, 1, H, W)`.

---

#### La convencion (N, C, H, W) en PyTorch

PyTorch usa **canales primero** (channels-first). Esto NO es universal — TensorFlow/Keras usa por defecto canales al final:

```text
PyTorch:     (N, C, H, W)   → channels-first
TensorFlow:  (N, H, W, C)   → channels-last

Misma imagen de 10 RGB 64×64:
  PyTorch:    (10, 3, 64, 64)
  TensorFlow: (10, 64, 64, 3)
```

Todas las capas de PyTorch (`Conv2d`, `MaxPool2d`, `BatchNorm2d`) esperan el formato `(N, C, H, W)`. Si pasas una imagen en el formato incorrecto, los resultados son silenciosamente erroneos.

---

#### Ejemplos de shapes de imagenes reales

```text
Dataset           Shape del batch (N=32)     Descripcion
───────────────   ──────────────────────     ─────────────────────────────
MNIST             (32,  1, 28, 28)           28×28 pixeles, grises (1 canal)
CIFAR-10          (32,  3, 32, 32)           32×32 pixeles, color (3 canales)
Flowers-102       (32,  3, 224, 224)         224×224 pixeles, color (ImageNet)
Imagenes medicas  (32,  1, 512, 512)         512×512, escala de grises
Video (1 frame)   (32,  3, 720, 1280)        HD 720p
Video (T frames)  (32, T, 3, 224, 224)       5D: batch+tiempo+canales+HxW
```

El laboratorio usa **Flowers-102** con shape `(batch, 3, 224, 224)` — el tamaño estandar de **ImageNet**, que es el que espera AlexNet.

---

#### ¿Como llega una imagen a este formato?

```text
Imagen en disco (archivo .jpg):
  → PIL/OpenCV: array (H, W, C) = (224, 224, 3)  ← channels-last, enteros 0-255

  → transforms.ToTensor():
      convierte a tensor (C, H, W) = (3, 224, 224)
      normaliza a float entre 0.0 y 1.0

  → transforms.Normalize(mean, std):
      ajusta los valores al rango esperado por la red

  → DataLoader (batch de 10):
      apila 10 tensores → (10, 3, 224, 224)
      agrega la dimension N automaticamente
```

Esto es exactamente lo que hace el laboratorio con el dataset de flores antes de pasarlas a MiAlexNet.

---

## Celda 6: Nota del laboratorio — La dimension batch

### Texto del notebook

> La dimension de batch es importante pues es la que nos permite poder entrenar de forma paralela en nuestras GPUs. El estandar de Pytorch es que esta es la primera dimension de nuestros tensores siempre. Aunque solo evaluar un elemento en nuestra red, este debe tener una dimension de batch en su primer lugar o sino no funcionara. Tambien es necesario mencionar que Pytorch espera siempre que la dimension del canal vaya antes que las dimensiones de ancho y alto en imagenes.

### Analisis

---

#### Por que el batch va primero

La GPU puede ejecutar **muchos calculos en paralelo** al mismo tiempo. Si le das un batch de 32 imagenes, no las procesa una a una — las procesa **todas a la vez** en paralelo:

```text
Sin batch (ineficiente):
  imagen_1 → red → pred_1   ← espera
  imagen_2 → red → pred_2   ← espera
  ...
  imagen_32 → red → pred_32  ← 32 viajes separados

Con batch (eficiente):
  [imagen_1, imagen_2, ..., imagen_32] → red → [pred_1, ..., pred_32]
                                              ↑
                                   todo en 1 solo paso paralelo
```

Al poner el batch en la **dimension 0**, PyTorch puede aplicar las capas de la red a todas las muestras del batch en una sola operacion matricial. Esto es lo que hace que entrenar con GPU sea 10x-100x mas rapido que en CPU.

---

#### La regla del batch aunque sea 1 solo elemento

Si quieres evaluar UNA sola imagen, no puedes pasarla directamente — tienes que agregar la dimension de batch de todas formas:

```python
# Una imagen de 3 canales, 64x64:
imagen = torch.randn(3, 64, 64)   # shape (3, 64, 64)

# Pasarla directo a la red → ERROR
# modelo(imagen)  ← la red espera (N, C, H, W), no (C, H, W)

# Correcto: agregar dimension de batch con unsqueeze(0)
imagen_con_batch = imagen.unsqueeze(0)  # shape (1, 3, 64, 64)
pred = modelo(imagen_con_batch)         # funciona

# Alternativa equivalente:
imagen_con_batch = imagen[None, :, :, :]  # shape (1, 3, 64, 64)
```

```text
imagen.unsqueeze(0):
  (3, 64, 64)  →  (1, 3, 64, 64)
                   ↑
                   batch = 1 (un solo elemento)
```

---

#### El orden de dimensiones en PyTorch es una convencion, no una ley

PyTorch no "sabe" que la dimension 0 es el batch — es una **convencion** que todas las capas respetan. Si pasas un tensor con las dimensiones en el orden equivocado, PyTorch no da error; simplemente calcula algo incorrecto:

```text
Tensor correcto:   (10, 3, 64, 64)  →  10 imagenes RGB de 64x64
Tensor invertido:  (3, 10, 64, 64)  →  PyTorch lo acepta, pero trata
                                        los 3 canales como si fueran
                                        el batch — resultado basura
```

Por eso el notebook enfatiza la convencion: **siempre `(N, C, H, W)`**.

---

#### Resumen de convenciones de forma en PyTorch

```text
Tipo de dato          Convencion          Significado
──────────────────    ────────────────    ─────────────────────────────────
Datos 1D (tabular)    (N, F)              N muestras, F features
Secuencias (texto)    (N, T, E)           N secuencias, T tokens, E embedding
Imagenes              (N, C, H, W)        N imagenes, C canales, H alto, W ancho
Video                 (N, T, C, H, W)     N videos, T frames, C canales, H, W

N = batch size   (siempre primero)
C = channels     (antes que H y W en imagenes)
```

La convencion de channels-first `(N, C, H, W)` que usa PyTorch es diferente a TensorFlow que usa channels-last `(N, H, W, C)`. En el laboratorio esto importa al cargar pesos preentrenados de otros frameworks.

---

## Celda 7: Mover tensores a la GPU

### Codigo

```python
tensor_nuevo = torch.randn((1,2,3))     # Por defecto el tensor está en CPU
tensor_nuevo_gpu = tensor_nuevo.cuda()  # Creé una copia de tensor_nuevo en GPU!
otra_forma = tensor_nuevo.to("cuda")    # Creé otra copia de tensor_nuevo en GPU!
print(otra_forma)
```

### Salida

```text
tensor([[[ 0.3019,  1.3206, -0.5961],
         [-0.6800, -0.2182, -1.8108]]], device='cuda:0')
```

### Analisis

---

#### El tensor vive en un dispositivo

Todo tensor en PyTorch existe en un **dispositivo** especifico: CPU o GPU. Por defecto, siempre se crea en CPU:

```text
tensor_nuevo = torch.randn((1,2,3))
  → device: cpu
  → La memoria esta en la RAM normal del computador
```

Cuando se imprime un tensor en CPU, no aparece el device porque es el default. Cuando esta en GPU, si aparece:

```text
CPU: tensor([[ 0.30,  1.32, -0.60], ...])
GPU: tensor([[ 0.30,  1.32, -0.60], ...], device='cuda:0')
                                                     ↑
                                              GPU numero 0
```

---

#### `.cuda()` vs `.to("cuda")`

Son equivalentes, pero `.to()` es el metodo **moderno y recomendado**:

```text
Metodo                  Funciona con      Nota
─────────────────────   ─────────────     ──────────────────────────────────
tensor.cuda()           Solo GPU Nvidia   Forma antigua, directa
tensor.to("cuda")       Solo GPU Nvidia   Forma moderna, equivalente
tensor.to("cuda:0")     GPU especifica    Util si hay multiples GPUs
tensor.to("cuda:1")     Segunda GPU       Selecciona GPU por indice
tensor.to("cpu")        CPU               Trae de vuelta a CPU
tensor.to(device)       Variable          Flexibe: device puede ser "cpu" o "cuda"
```

El patron mas comun en el laboratorio:

```python
# Al inicio del script, detectar si hay GPU disponible:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Luego mover todo al dispositivo correcto:
tensor = tensor.to(device)
model = model.to(device)
```

Esto permite que el mismo codigo corra en CPU (si no hay GPU) y en GPU (si la hay) sin cambiarlo.

---

#### `.cuda()` crea una COPIA, no mueve el original

El comentario del notebook lo dice explicitamente: `# Cree una copia`. El tensor original sigue en CPU:

```python
tensor_nuevo     = torch.randn((1,2,3))   # en CPU
tensor_nuevo_gpu = tensor_nuevo.cuda()    # copia en GPU

tensor_nuevo.device      # cpu  ← el original NO se movio
tensor_nuevo_gpu.device  # cuda:0
```

Si quieres que la variable apunte a la version en GPU (sobrescribir):

```python
tensor_nuevo = tensor_nuevo.to("cuda")   # ahora tensor_nuevo apunta a GPU
```

---

#### Por que la GPU es necesaria para este laboratorio

El laboratorio entrena **MiAlexNet** con el dataset **Flowers-102** (imagenes de 224×224). Para tener una idea de la escala:

```text
Un batch de 32 imagenes:
  shape: (32, 3, 224, 224)
  numeros: 32 × 3 × 224 × 224 = 4,816,896 valores

MiAlexNet tiene ~57 millones de parametros.

En CPU: cada paso del entrenamiento tarda varios segundos.
En GPU: el mismo paso tarda milisegundos.
        → 100x mas rapido.
```

La GPU puede hacer estas operaciones en paralelo porque tiene miles de nucleos pequeños (vs los pocos nucleos grandes de la CPU):

```text
CPU: 8-32 nucleos grandes, buenos para tareas complejas secuenciales
GPU: 3000-10000 nucleos pequeños, buenos para muchos calculos simples en paralelo

Multiplicar matrices grandes (lo que hace una red neuronal)
→ es exactamente el tipo de problema que la GPU hace mejor.
```

---

#### El error mas comun: mezclar CPU y GPU

Si intentas operar tensores en dispositivos distintos, PyTorch da error:

```python
a = torch.randn(3)          # CPU
b = torch.randn(3).cuda()   # GPU

a + b  # → RuntimeError: Expected all tensors to be on the same device
```

Todo lo que interactua tiene que estar en el **mismo dispositivo**. Por eso en el laboratorio se mueve tanto el modelo como los datos a GPU al inicio:

```python
model = MiAlexNet().to(device)

for images, labels in dataloader:
    images = images.to(device)   # datos → GPU
    labels = labels.to(device)   # etiquetas → GPU
    outputs = model(images)      # todo en GPU → OK
```

---

## Celda 8: Definicion del modelo y la capa Linear

### Texto del notebook

> Definir un modelo de aprendizaje profundo consiste en definir una nueva clase que herede de `torch.nn.Module`. Esta clase debe implementar dos metodos:
>
> - `__init__(self)`: el constructor. Aqui se definen todas las capas, funciones de activacion, pooling, etc.
> - `forward(self, input)`: define como fluye la informacion entre capas. Debe retornar un tensor.
>
> Todos los elementos arquitectonicos estan en `torch.nn` y derivan de `torch.nn.Module`.

### Codigo

```python
import torch
from torch.nn import Linear, ReLU, Sigmoid, Sequential, Softmax, BatchNorm1d, BatchNorm2d, Dropout, LayerNorm

tensor_prueba = torch.randn((3, 5))
print(tensor_prueba, tensor_prueba.shape)

# Capa Lineal que va de 5 dimensiones a 8
capa_lineal = Linear(5, 8)
tensor_nuevo = capa_lineal(tensor_prueba)
print("Tensor después de Capa Lineal:")
print(tensor_nuevo, tensor_nuevo.shape)
```

### Salida

```text
tensor([[ 1.6242, -1.4484, -0.7869, -1.3575, -0.3389],
        [-0.0597, -1.0161, -0.9935,  0.5773,  0.1553],
        [ 0.4160,  1.2555, -2.5687,  0.4599, -0.4211]]) torch.Size([3, 5])

Tensor después de Capa Lineal:
tensor([[-0.3925, -1.0241, -1.5916, -0.3037, -0.8452, -0.4214, -1.1762,  0.4189],
        [-0.3703,  0.0737, -0.3450, -0.1637, -0.3950,  0.5478, -0.3754,  0.2379],
        [ 0.8189, -0.6418,  0.1086, -0.0366,  0.2350, -0.4037, -0.6229, -0.3475]],
       grad_fn=<AddmmBackward0>) torch.Size([3, 8])
```

### Analisis

---

#### La estructura de cualquier modelo PyTorch

Todo modelo en PyTorch es una **clase que hereda de `nn.Module`**. La estructura minima es siempre la misma:

```python
import torch.nn as nn

class MiModelo(nn.Module):
    def __init__(self):
        super().__init__()          # obligatorio: inicializa nn.Module
        # aqui se DECLARAN las capas
        self.capa1 = nn.Linear(5, 8)
        self.capa2 = nn.ReLU()

    def forward(self, x):
        # aqui se CONECTAN las capas (como fluye la informacion)
        x = self.capa1(x)
        x = self.capa2(x)
        return x
```

```text
__init__:    "Cuales son las piezas de la red?"
             → declara las capas como atributos (self.capa1, etc.)

forward:     "Como se conectan esas piezas?"
             → define el flujo: input → capa1 → capa2 → output
```

En esta celda, en vez de definir un modelo completo, se usa `Linear` directamente como objeto independiente — es lo mismo que tenerlo dentro de un `nn.Module`, solo que suelto.

---

#### `Linear(5, 8)` — la capa completamente conectada

`Linear(in_features, out_features)` es la capa mas basica de una red neuronal: cada neurona de entrada se conecta con cada neurona de salida.

```text
Linear(5, 8):
  5 neuronas de entrada → 8 neuronas de salida

  ●──┬──────────────────────────────● neurona salida 0
  ●──┼──────────────────────────────● neurona salida 1
  ●──┼──────────────────────────────● neurona salida 2
  ●──┼──────────────────────────────● neurona salida 3
  ●──┼──────────────────────────────● neurona salida 4
     └──────────────────────────────● neurona salida 5
                                    ● neurona salida 6
                                    ● neurona salida 7

  5 entradas × 8 salidas = 40 pesos (W)
  + 8 sesgos (bias, uno por neurona de salida)
  = 48 parametros en total
```

La operacion matematica que realiza:

```text
salida = entrada @ W^T + bias

  entrada: (3, 5)   — 3 muestras, 5 features
  W:       (8, 5)   — 8 neuronas, cada una con 5 pesos
  W^T:     (5, 8)   — traspuesta para multiplicar
  bias:    (8,)     — un sesgo por neurona de salida

  (3, 5) @ (5, 8) = (3, 8)   ← la forma del resultado
```

---

#### El cambio de shape: `(3, 5)` → `(3, 8)`

```text
Antes de Linear(5, 8):          Despues de Linear(5, 8):
┌─────────────────┐             ┌──────────────────────────────┐
│ f0 f1 f2 f3 f4  │  muestra 0  │ n0 n1 n2 n3 n4 n5 n6 n7     │
│ f0 f1 f2 f3 f4  │  muestra 1  │ n0 n1 n2 n3 n4 n5 n6 n7     │
│ f0 f1 f2 f3 f4  │  muestra 2  │ n0 n1 n2 n3 n4 n5 n6 n7     │
└─────────────────┘             └──────────────────────────────┘
      (3, 5)                                (3, 8)

La dimension del batch (3) se mantiene siempre.
Solo cambia la ultima dimension: 5 features → 8 neuronas.
```

Esto es exactamente lo que hace cada capa densa en una red: **transformar el espacio de representacion**. Pasar de 5 dimensiones a 8 significa que la red puede crear 8 nuevas "caracteristicas" combinando las 5 originales.

---

#### `grad_fn=<AddmmBackward0>` — el rastro del gradiente

La salida incluye `grad_fn=<AddmmBackward0>`. Esto significa que PyTorch esta **registrando automaticamente** esta operacion para poder calcular gradientes despues.

```text
tensor_prueba:         NO tiene grad_fn (es un dato de entrada, no aprende)
tensor_nuevo:          SI tiene grad_fn (paso por una capa con parametros)

grad_fn=<AddmmBackward0>:
  Addmm = "Add + Matrix Multiply"  ← la operacion que se hizo (xW^T + b)
  Backward0 = "funcion para calcular el gradiente de esta operacion"
```

Cuando luego se llame a `loss.backward()`, PyTorch seguira esta cadena de `grad_fn` hacia atras (backpropagation) para calcular cuanto cambiar cada peso de `capa_lineal`.

```text
forward:   input → Linear → output    (calculo normal)
backward:  d_loss/d_W ← ← ← d_loss   (siguiendo grad_fn en reversa)
```

Si un tensor no tiene `grad_fn`, significa que no paso por ninguna capa con parametros aprendibles (o que se desactivo el tracking con `torch.no_grad()`).

---

#### Los elementos de `torch.nn` mencionados en el notebook

```text
Capa               Que hace                          Parametros clave
─────────────────  ────────────────────────────────  ─────────────────────────
Linear(n, m)       Capa densa: n entradas, m salidas  in, out features
Conv2d(c, f, k)    Convolucion 2D                     in_ch, out_ch, kernel
ReLU()             Activacion: max(0, x)              —
Sigmoid()          Activacion: 1/(1+e^-x)             —
Softmax(dim)       Distribucion de probabilidad       dim (sobre que dim aplicar)
MaxPool2d(k)       Pooling maximo 2D                  kernel_size
Dropout(p)         Apaga neuronas con prob p           p (default 0.5)
BatchNorm2d(c)     Normaliza por batch (imagenes)     num_channels
BatchNorm1d(f)     Normaliza por batch (datos 1D)     num_features
LayerNorm(shape)   Normaliza por capa (texto/NLP)     normalized_shape
Sequential(...)    Encadena modulos en orden           modulos en secuencia
```

`Sequential` es un atajo para no tener que escribir el `forward` cuando las capas van una tras otra:

```python
# Sin Sequential:
class Red(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Con Sequential (equivalente):
red = nn.Sequential(
    nn.Linear(5, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

---

## Celda 9: Funciones de activacion — ReLU

### Codigo

```python
# Funciones de Activación
# ReLU
relu = ReLU()
print("Antes de ReLU: ")
print(tensor_prueba)
print("Después de ReLU: ")
print(relu(tensor_prueba))
```

### Salida

```text
Antes de ReLU: 
tensor([[ 1.8557,  0.1690, -1.6810, -0.3846,  0.3428],
        [ 0.1939,  0.9907, -2.5539, -1.3668,  0.6404],
        [ 0.6933,  0.0300,  0.3220, -0.2189,  0.8146]])
Después de ReLU: 
tensor([[1.8557, 0.1690, 0.0000, 0.0000, 0.3428],
        [0.1939, 0.9907, 0.0000, 0.0000, 0.6404],
        [0.6933, 0.0300, 0.3220, 0.0000, 0.8146]])
```

### Analisis

---

#### ¿Que hace ReLU?

**ReLU** (Rectified Linear Unit) es la funcion de activacion mas usada en redes profundas. Su definicion es la mas simple posible:

```text
ReLU(x) = max(0, x)

  Si x > 0  →  devuelve x       (lo deja pasar igual)
  Si x ≤ 0  →  devuelve 0       (lo apaga)
```

Aplicado elemento a elemento sobre el tensor:

```text
Antes:   [ 1.86,  0.17, -1.68, -0.38,  0.34]
                         ↑       ↑
                       negativos → se hacen 0

Despues: [ 1.86,  0.17,  0.00,  0.00,  0.34]
```

La forma del tensor **no cambia**: `(3, 5)` → `(3, 5)`. ReLU no transforma dimensiones, solo filtra valores.

---

#### ReLU no tiene parametros que aprender

A diferencia de `Linear`, `ReLU` no tiene pesos ni bias. Es una funcion fija:

```text
Linear(5, 8):   tiene 48 parametros (40 pesos + 8 bias) → se ajustan con backprop
ReLU():         tiene 0 parametros → siempre hace max(0, x)
```

Por eso en el notebook se crea simplemente con `ReLU()` sin argumentos.

---

#### Por que necesitamos funciones de activacion

Sin activacion, apilar capas lineales no sirve de nada:

```text
Sin activacion:
  capa1(x) = W1·x + b1         (transformacion lineal)
  capa2(x) = W2·(W1·x + b1) + b2
           = (W2·W1)·x + (W2·b1 + b2)
           = W_combinada·x + b_combinada   ← sigue siendo lineal!

  2 capas lineales = 1 capa lineal.  100 capas lineales = 1 capa lineal.
  → No hay "profundidad" real.

Con activacion:
  capa1(x) = W1·x + b1
  relu1    = max(0, capa1(x))        ← introduce no-linealidad
  capa2(x) = W2·relu1 + b2
  relu2    = max(0, capa2(x))        ← otra no-linealidad
  ...
  → Cada capa puede aprender patrones que la anterior no puede.
```

La activacion es lo que le da a la red la capacidad de aprender funciones **no lineales** (curvas, bordes, texturas, conceptos abstractos).

---

#### ReLU en la practica: lo que se ve en la salida

Mirando la salida fila por fila:

```text
Fila 0: [ 1.86,  0.17, -1.68, -0.38,  0.34]
         →     [ 1.86,  0.17,  0.00,  0.00,  0.34]
                              ↑       ↑
                           -1.68    -0.38  → 0

Fila 1: [ 0.19,  0.99, -2.55, -1.37,  0.64]
         →     [ 0.19,  0.99,  0.00,  0.00,  0.64]

Fila 2: [ 0.69,  0.03,  0.32, -0.22,  0.81]
         →     [ 0.69,  0.03,  0.32,  0.00,  0.81]
```

En promedio, con datos aleatorios de distribucion normal, **~50% de los valores** caen en negativo y se apagan. Esto es intencional: la red aprende a usar solo las "neuronas" relevantes para cada entrada.

---

#### Comparacion con otras funciones de activacion

```text
Funcion    Formula              Rango        Ventaja / Uso tipico
─────────  ───────────────────  ───────────  ──────────────────────────────
ReLU       max(0, x)            [0, +∞)      Rapida, evita vanishing gradient
                                             → capas ocultas de CNNs, MLPs

Sigmoid    1 / (1 + e^-x)       (0, 1)       Salida interpretable como prob.
                                             → capa final de clasificacion binaria

Tanh       (e^x - e^-x)/        (-1, 1)      Centrada en 0 (mejor que Sigmoid)
           (e^x + e^-x)                      → RNNs clasicas

Softmax    e^xi / Σ e^xj        (0, 1)       Distribucion de probabilidad
                                             → capa final de clasificacion multi-clase

Leaky      max(0.01x, x)        (-∞, +∞)     Evita neuronas "muertas" de ReLU
ReLU                                         → alternativa a ReLU
```

El problema que resuelve ReLU vs Sigmoid/Tanh:

```text
Sigmoid y Tanh tienen gradientes muy pequeños para valores extremos:
  sigmoid'(x) → 0 cuando x → ±∞

→ "Vanishing gradient": en redes profundas, los gradientes se vuelven
   casi 0 al propagarse hacia atras → las capas iniciales no aprenden.

ReLU tiene gradiente 1 para x > 0 → los gradientes fluyen sin atenuarse.
→ Permite entrenar redes mucho mas profundas.
```

---

#### ReLU en AlexNet (el modelo del laboratorio)

En **MiAlexNet**, ReLU aparece despues de cada capa convolucional y de cada capa densa:

```text
Conv → ReLU → MaxPool → Conv → ReLU → ... → Linear → ReLU → Linear → ReLU → Linear

La ultima capa (clasificacion en 102 clases) NO tiene ReLU.
→ El output puede ser cualquier numero real (logits).
→ La loss (CrossEntropyLoss) aplica Softmax internamente.

---

## Celda 10: Funciones de activacion — Sigmoid

### Codigo

```python
# Sigmoid
s = Sigmoid()
print("Antes de Sigmoid: ")
print(tensor_prueba)
print("Después de Sigmoid: ")
print(s(tensor_prueba))
```

### Salida

```text
Antes de Sigmoid: 
tensor([[ 1.6242, -1.4484, -0.7869, -1.3575, -0.3389],
        [-0.0597, -1.0161, -0.9935,  0.5773,  0.1553],
        [ 0.4160,  1.2555, -2.5687,  0.4599, -0.4211]])
Después de Sigmoid: 
tensor([[0.8354, 0.1902, 0.3128, 0.2046, 0.4161],
        [0.4851, 0.2658, 0.2702, 0.6405, 0.5387],
        [0.6025, 0.7782, 0.0712, 0.6130, 0.3963]])
```

### Analisis

---

#### ¿Que hace Sigmoid?

**Sigmoid** comprime cualquier numero real al rango **(0, 1)**. Su formula:

```text
σ(x) = 1 / (1 + e^(-x))
```

El efecto:

```text
x muy negativo → σ(x) → 0    (por ejemplo: σ(-5) = 0.007)
x = 0          → σ(x) = 0.5  (punto medio)
x muy positivo → σ(x) → 1    (por ejemplo: σ(5) = 0.993)

Forma de "S" (sigmoide):
  1.0 ┤                    ╭──────────
      │                ╭───╯
  0.5 ┤            ────╯
      │        ╭───╯
  0.0 ┤───────╯
      └────────────────────────
            -5  -2   0   2   5
```

---

#### Verificando con los numeros de la salida

Tomando algunos valores del tensor de entrada:

```text
Entrada:  1.6242  →  Sigmoid:  0.8354   (positivo alto → cerca de 1)
Entrada: -1.4484  →  Sigmoid:  0.1902   (negativo → cerca de 0)
Entrada: -2.5687  →  Sigmoid:  0.0712   (muy negativo → muy cerca de 0)
Entrada:  1.2555  →  Sigmoid:  0.7782   (positivo → por encima de 0.5)
Entrada: -0.0597  →  Sigmoid:  0.4851   (casi 0 → casi 0.5)
```

Los valores se "aplastan" dentro del rango (0, 1). La forma `(3, 5)` se mantiene identica.

---

#### ¿Para que sirve Sigmoid?

La clave es que el rango (0, 1) permite **interpretar la salida como una probabilidad**:

```text
Clasificacion binaria (SI / NO):
  output = sigmoid(logit)
  output = 0.85  → "85% de probabilidad de que sea SI"
  output = 0.12  → "12% de probabilidad → muy probablemente NO"

Ejemplos de uso:
  ¿Esta imagen contiene un gato?           → sigmoid → 0.92  (si, probablemente)
  ¿Este email es spam?                      → sigmoid → 0.03  (no, probablemente)
  ¿El cliente va a comprar?                 → sigmoid → 0.67  (probabilidad media-alta)
```

---

#### Diferencia clave con ReLU

```text
                  ReLU               Sigmoid
Rango:            [0, +∞)            (0, 1)
Negatives:        se vuelven 0       se comprimen cerca de 0
Positivos:        pasan igual        se comprimen cerca de 1
Gradiente:        1 (para x>0)       pequeno para x extremo
Uso tipico:       capas ocultas      capa de salida binaria
Tiene 0 exacto:   SI (x ≤ 0)        NO (siempre entre 0 y 1)
```

El valor **0.0712** que aparece en la salida (para la entrada -2.5687) muestra que Sigmoid nunca llega exactamente a 0 — solo se acerca. ReLU en cambio produce ceros exactos.

---

#### El problema de Sigmoid en capas ocultas: vanishing gradient

Sigmoid tiene un defecto cuando se usa en capas internas de redes profundas:

```text
Derivada de Sigmoid: σ'(x) = σ(x) · (1 - σ(x))

  Maximo de σ'(x) ocurre en x=0: σ'(0) = 0.5 · 0.5 = 0.25
  Para x=3:  σ'(3) ≈ 0.045
  Para x=5:  σ'(5) ≈ 0.007
  Para x=10: σ'(10) ≈ 0.000045  ← casi cero

En backpropagation se multiplican los gradientes de cada capa:
  0.25 × 0.25 × 0.25 × 0.25 × 0.25 = 0.001  (5 capas)
  → El gradiente que llega a las capas iniciales es casi 0
  → Las primeras capas no aprenden nada
```

Por eso Sigmoid se usa **solo en la capa de salida** para clasificacion binaria, no en capas ocultas. Para capas ocultas se usa ReLU.

---

#### Sigmoid vs `BCEWithLogitsLoss` en el laboratorio

En el laboratorio (clase 8) se uso `BCEWithLogitsLoss` en vez de aplicar Sigmoid + `BCELoss` por separado:

```python
# Forma manual (menos estable numericamente):
pred_prob = sigmoid(logit)          # aplicar sigmoid
loss = F.binary_cross_entropy(pred_prob, label)

# Forma recomendada (mas estable):
loss = F.binary_cross_entropy_with_logits(logit, label)
# → aplica sigmoid internamente con mayor precision numerica
```

`BCEWithLogitsLoss` combina sigmoid + BCE en una sola operacion mas numericamente estable. En la practica, casi nunca se pone Sigmoid en la ultima capa si se usa esta loss.

---

## Celda 11: Funciones de activacion — Softmax

### Codigo

```python
# Softmax
soft = Softmax(dim=1)
print("Antes de Softmax: ")
print(tensor_prueba)
print("Después de Softmax: ")
soft_tensor = soft(tensor_prueba)
print(soft_tensor)
print(soft_tensor.sum(dim=0))
```

### Salida

```text
Antes de Softmax: 
tensor([[ 1.6242, -1.4484, -0.7869, -1.3575, -0.3389],
        [-0.0597, -1.0161, -0.9935,  0.5773,  0.1553],
        [ 0.4160,  1.2555, -2.5687,  0.4599, -0.4211]])
Después de Softmax: 
tensor([[0.7535, 0.0349, 0.0676, 0.0382, 0.1058],
        [0.2038, 0.0783, 0.0801, 0.3853, 0.2526],
        [0.2065, 0.4780, 0.0104, 0.2157, 0.0894]])
tensor([1.1637, 0.5912, 0.1581, 0.6392, 0.4478])
```

### Analisis

---

#### ¿Que hace Softmax?

**Softmax** convierte un vector de numeros arbitrarios en una **distribucion de probabilidad**: todos los valores quedan entre 0 y 1, y **suman exactamente 1**.

La formula para cada elemento `i` del vector:

```text
Softmax(x)_i = e^(x_i) / Σ_j e^(x_j)

  Numerador:    e^(x_i)         → exponencial del valor i
  Denominador:  suma de e^(xj)  → suma de todos los exponenciales del vector
```

Ejemplo con la primera fila `[1.62, -1.45, -0.79, -1.36, -0.34]`:

```text
Paso 1 — exponenciales:
  e^1.62  = 5.05
  e^-1.45 = 0.23
  e^-0.79 = 0.45
  e^-1.36 = 0.26
  e^-0.34 = 0.71
  Suma    = 6.70

Paso 2 — dividir por la suma:
  5.05/6.70 = 0.7535  ← el mas alto porque 1.62 era el mayor
  0.23/6.70 = 0.0349
  0.45/6.70 = 0.0676
  0.26/6.70 = 0.0382
  0.71/6.70 = 0.1058

Verificacion: 0.7535 + 0.0349 + 0.0676 + 0.0382 + 0.1058 = 1.0 ✓
```

---

#### El parametro `dim=1` — sobre que dimension normalizar

Este es el punto mas confuso de Softmax. El parametro `dim` indica **a lo largo de que dimension** se aplica la normalizacion (donde los valores van a sumar 1).

El tensor tiene shape `(3, 5)`:

```text
           dim=1 (columnas, tamaño 5)
           ←──────────────────────→

  ┌────────────────────────────────────┐  ↑
  │  1.62  -1.45  -0.79  -1.36  -0.34 │  │ dim=0
  │ -0.06  -1.02  -0.99   0.58   0.16 │  │ (filas,
  │  0.42   1.26  -2.57   0.46  -0.42 │  │ tamaño 3)
  └────────────────────────────────────┘  ↓
```

Con `dim=1`: para **cada fila**, los 5 valores se convierten en una distribucion de probabilidad (suman 1). Cada fila = una muestra = una distribucion independiente.

```text
Fila 0: [0.7535, 0.0349, 0.0676, 0.0382, 0.1058]  → suma = 1.0
Fila 1: [0.2038, 0.0783, 0.0801, 0.3853, 0.2526]  → suma = 1.0
Fila 2: [0.2065, 0.4780, 0.0104, 0.2157, 0.0894]  → suma = 1.0
```

---

#### Por que `sum(dim=0)` no da [1, 1, 1, 1, 1]

El codigo imprime `soft_tensor.sum(dim=0)` y el resultado es `[1.1637, 0.5912, 0.1581, 0.6392, 0.4478]` — no son unos.

Esto es correcto y no es un error. `sum(dim=0)` suma **por columnas** (a lo largo de las filas):

```text
soft_tensor:
  fila 0:  [0.7535,  0.0349,  0.0676,  0.0382,  0.1058]
  fila 1:  [0.2038,  0.0783,  0.0801,  0.3853,  0.2526]
  fila 2:  [0.2065,  0.4780,  0.0104,  0.2157,  0.0894]
  ────────────────────────────────────────────────────────
  sum(0):  [1.1638,  0.5912,  0.1581,  0.6392,  0.4478]
             ↑
             0.7535 + 0.2038 + 0.2065 = 1.1638  ← suma de las 3 filas en col 0

Softmax con dim=1 garantiza que CADA FILA sume 1.
No garantiza que cada columna sume 1.
```

Si quisieras que las **columnas** sumaran 1, necesitarias `Softmax(dim=0)`.

---

#### Para que se usa Softmax en clasificacion

Softmax es la capa final en modelos de **clasificacion multiclase**. Convierte los "logits" (puntuaciones crudas de la red) en probabilidades interpretables:

```text
Clasificar una imagen en 10 clases (CIFAR-10):

  Logits (salida cruda):   [-2.1,  0.3,  8.5, -1.4,  0.2, ...]
                                          ↑
                                   clase 2 tiene el mayor logit

  Softmax(logits):         [0.001, 0.009, 0.974, 0.002, ...]
                                           ↑
                            97.4% de probabilidad → "es clase 2 (pajaro)"
```

La clase predicha es simplemente la que tiene la probabilidad mas alta → `argmax(softmax_output)`.

---

#### Softmax vs CrossEntropyLoss

En la practica, **no se pone Softmax en la red** cuando se usa `CrossEntropyLoss`:

```python
# Lo que NO se hace:
class Red(nn.Module):
    def forward(self, x):
        x = self.fc_final(x)
        return nn.Softmax(dim=1)(x)   # NO hacer esto

loss = nn.CrossEntropyLoss()(output, labels)  # ya aplica softmax internamente

# Lo correcto:
class Red(nn.Module):
    def forward(self, x):
        return self.fc_final(x)   # devolver logits crudos

loss = nn.CrossEntropyLoss()(output, labels)  # aplica log-softmax internamente
```

`CrossEntropyLoss` aplica log-softmax + NLLLoss internamente, y es numericamente mas estable que hacerlo por separado. Softmax explicito solo se usa cuando necesitas las probabilidades para algo mas (mostrarlas, umbralizar, etc.).

---

## Celda 12: Sequential — encadenar capas

### Codigo

```python
# Sequential
# Unamos pasos
seq = Sequential(capa_lineal, relu, soft)
tensor_final = seq(tensor_prueba)
print("Aplicando Sequential: ")
print(tensor_final, tensor_final.shape)
```

### Salida

```text
Aplicando Sequential: 
tensor([[0.1174, 0.1174, 0.1174, 0.1174, 0.1174, 0.1174, 0.1174, 0.1784],
        [0.1102, 0.1186, 0.1102, 0.1102, 0.1102, 0.1906, 0.1102, 0.1398],
        [0.2351, 0.1037, 0.1155, 0.1037, 0.1311, 0.1037, 0.1037, 0.1037]],
       grad_fn=<SoftmaxBackward0>) torch.Size([3, 8])
```

### Analisis

---

#### Lo que hace Sequential

`Sequential(capa_lineal, relu, soft)` encadena las tres capas en orden. La salida de cada capa es la entrada de la siguiente:

```text
tensor_prueba (3, 5)
       ↓
  capa_lineal: Linear(5, 8)   →  (3, 8)   multiplica pesos + bias
       ↓
  relu: ReLU()                →  (3, 8)   apaga valores negativos (→ 0)
       ↓
  soft: Softmax(dim=1)        →  (3, 8)   convierte cada fila en probabilidades
       ↓
tensor_final (3, 8)
```

No hay que escribir el `forward` manualmente — Sequential lo construye automaticamente.

---

#### Por que tantos valores iguales (0.1174, 0.1102, 0.1037)

Esta es la observacion mas interesante de la celda. En la fila 0 hay **7 valores identicos** de `0.1174` y solo uno distinto (`0.1784`).

La explicacion esta en la interaccion entre **ReLU y Softmax**:

```text
1. Linear produce 8 valores para la fila 0 (algunos negativos)

2. ReLU convierte todos los negativos a exactamente 0:
   [neg, neg, neg, neg, neg, neg, neg, positivo]
   → [0,   0,   0,   0,   0,   0,   0,   X    ]
                                              ↑
                                        unico sobreviviente

3. Softmax recibe [0, 0, 0, 0, 0, 0, 0, X]:
   e^0 = 1 para todos los ceros → todos iguales
   e^X > 1 para el positivo → un poco mas alto

   Resultado: 7 valores iguales + 1 mas alto
```

Verificacion con los numeros reales:

```text
Fila 0: [0.1174, 0.1174, 0.1174, 0.1174, 0.1174, 0.1174, 0.1174, 0.1784]

  Suma de los 7 iguales: 7 × 0.1174 = 0.8218
  El distinto:                         0.1784
  Total:                               1.0002 ≈ 1.0  ✓

  → Solo 1 de las 8 neuronas sobrevivio ReLU en esta muestra.
    Las otras 7 quedaron en 0 → Softmax les dio probabilidad igual (1/8 aprox).
```

Fila 1 tiene 2 valores distintos (0.1186 y 0.1906) → 2 neuronas sobrevivieron ReLU.
Fila 2 tiene 3 valores distintos → 3 neuronas sobrevivieron.

---

#### El `grad_fn=<SoftmaxBackward0>`

A diferencia de antes donde era `<AddmmBackward0>` (solo Linear), ahora el `grad_fn` apunta a la **ultima operacion de la cadena**: Softmax. Pero internamente PyTorch tiene registrada toda la secuencia:

```text
SoftmaxBackward0  ←  ReLUBackward0  ←  AddmmBackward0
      ↑                   ↑                  ↑
   Softmax              ReLU              Linear

Cuando se llame a loss.backward(), el gradiente fluye de derecha a izquierda
a traves de esta cadena completa.
```

---

#### Sequential vs definir forward manualmente

Estos dos codigos son equivalentes:

```python
# Con Sequential:
modelo = nn.Sequential(
    nn.Linear(5, 8),
    nn.ReLU(),
    nn.Softmax(dim=1)
)
output = modelo(x)

# Con forward explicito:
class MiModelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.soft(x)
        return x

output = MiModelo()(x)
```

`Sequential` es conveniente para arquitecturas simples donde la informacion fluye en linea recta. Para arquitecturas con ramificaciones (como MiAlexNet, que tiene capas de extraccion de features + capas de clasificacion), se necesita el `forward` explicito para controlar el flujo.

---

#### Esta celda como mini-red neuronal completa

El pipeline `Linear → ReLU → Softmax` es en esencia una **red neuronal de 1 capa oculta** para clasificacion:

```text
Entrada (5 features) → Linear → [transformacion] → ReLU → [no-linealidad] → Softmax → Probabilidades (8 clases)

En el laboratorio, MiAlexNet hace esto mismo pero con:
  - 5 capas convolucionales antes (extraccion de features de imagenes)
  - 3 capas Linear al final (clasificacion en 102 clases)
  - Sin Softmax explicito → CrossEntropyLoss lo aplica internamente
```

---

## Celda 13: Dropout

### Codigo

```python
# Dropout
drop = Dropout(p=0.1)  # con probabilidad de dejar en 0 de 0.5.

tensor_dropout = drop(tensor_prueba)
print(tensor_dropout)
```

### Salida

```text
tensor([[ 1.8047, -1.6093, -0.8743, -1.5083, -0.3766],
        [-0.0663, -1.1291, -1.1039,  0.6415,  0.1725],
        [ 0.4622,  1.3950, -2.8541,  0.5110, -0.4679]])
```

### Analisis

---

#### ¿Que hace Dropout?

Dropout apaga aleatoriamente neuronas durante el entrenamiento poniendolas en 0. El parametro `p` es la probabilidad de que cada neurona sea apagada:

```text
Dropout(p=0.1):
  Cada valor tiene 10% de probabilidad de → 0
  Cada valor tiene 90% de probabilidad de → sobrevivir (y escalar)
```

---

#### Por que no hay ningun cero en la salida

Con `p=0.1` (solo 10% de probabilidad) y apenas 15 valores en el tensor `(3, 5)`, es perfectamente normal que en esta ejecucion ninguno haya sido apagado. El dropout es **aleatorio** — cada vez que corre puede dar un resultado diferente.

Si se usara `p=0.5`, estadisticamente la mitad de los 15 valores habria sido apagada.

---

#### El escalado automatico (inverted dropout)

Aunque ninguna neurona fue apagada en esta corrida, **todos los valores si fueron escalados**. Comparando entrada y salida:

```text
tensor_prueba fila 0:   [ 1.6242, -1.4484, -0.7869, -1.3575, -0.3389]
tensor_dropout fila 0:  [ 1.8047, -1.6093, -0.8743, -1.5083, -0.3766]

Factor: 1.8047 / 1.6242 = 1.1111  ← es exactamente 1 / (1 - 0.1) = 1/0.9
```

PyTorch usa **inverted dropout**: los valores que sobreviven se multiplican por `1/(1-p)` para compensar los que fueron apagados. Esto mantiene la escala esperada del tensor igual con y sin dropout:

```text
Sin dropout:      E[salida] = valor original
Con dropout p=0.1:
  - 10% de neuronas → 0           (contribucion = 0)
  - 90% de neuronas → valor × 1/0.9  (contribucion = valor × 1/0.9 × 0.9 = valor)
  E[salida] = valor original       ← la escala esperada se mantiene
```

Sin este escalado, al quitar dropout en inferencia los valores serian sistematicamente mas grandes que durante entrenamiento, rompiendo el modelo.

---

#### Dropout solo actua en modo entrenamiento

```python
drop = Dropout(p=0.1)

# Modo entrenamiento (default):
drop.train()
drop(tensor_prueba)  # apaga y escala

# Modo evaluacion/inferencia:
drop.eval()
drop(tensor_prueba)  # devuelve el tensor SIN modificar (identidad)
```

```text
Entrenamiento: dropout activo → neuronas aleatorias a 0 → escala el resto
Inferencia:    dropout inactivo → pasa todo sin tocar

→ Durante inferencia se usa la red COMPLETA, sin apagar nada.
  El escalado del entrenamiento garantiza que los valores sean comparables.
```

Por eso es critico llamar `model.eval()` antes de evaluar o hacer inferencia, y `model.train()` antes de entrenar. Si se olvida `model.eval()`, el dropout sigue activo durante la prediccion y los resultados son no-deterministicos.

---

#### Nota sobre el comentario del notebook

El codigo dice `# con probabilidad de dejar en 0 de 0.5` pero usa `p=0.1`. El comentario esta desactualizado — la probabilidad real es `p=0.1` (10%), no 0.5 (50%).

---

#### ¿Por que sirve el dropout?

Dropout actua como un **regularizador**: al apagar neuronas aleatoriamente en cada paso, obliga a la red a no depender demasiado de ninguna neurona especifica:

```text
Sin dropout:
  La red puede "memorizar" los datos de entrenamiento.
  Neurona A siempre aprende a detectar exactamente el ruido del dato 17.

Con dropout:
  Neurona A a veces no esta → otras neuronas deben aprender lo mismo.
  La red aprende representaciones redundantes y mas robustas.
  → Generaliza mejor a datos nuevos (menos overfitting).
```

Valores tipicos de `p`:
- `p = 0.1 a 0.3`: dropout suave (como en esta celda)
- `p = 0.5`: dropout estandar (muy comun en capas densas)
- `p > 0.5`: raro, puede causar underfitting

---

## Celda 14: BatchNorm1d — normalizacion por batch

### Codigo

```python
# Batch Normalization 1d
tensor_lineal = 100*torch.randn((3, 5))  # Una imagen con varianza alta
bn = BatchNorm1d(5, momentum=None)
print("La media de los vectores originales por dimensión es: {}".format(tensor_lineal.mean(dim=[0])))
print("La varianza de los vectores originales por dimensión es: {}".format(tensor_lineal.var(dim=[0], unbiased=False)))
print("Las medias por dimensión de BN inicialmente son: {}".format(bn.running_mean))
print("Las varianzas por dimensión de BN inicialmente son: {}".format(bn.running_var))

tensor_bn = bn(tensor_lineal)

print("La media por dimensión del resultado es: {}".format(tensor_bn.mean(dim=[0])))
print("La varianza por dimensión del resultado es: {}".format(tensor_bn.var(dim=[0], unbiased=False)))
print("Las medias por dimensión de BN ahora son: {}".format(bn.running_mean))
print("Las varianzas por dimensión de BN ahora son: {}".format(bn.running_var))

for n, p in bn.named_parameters():
    print(n, p)
```

### Salida

```text
La media de los vectores originales por dimensión es: tensor([ 25.8616, -74.8571,   0.7456,  11.3224, -14.8267])
La varianza de los vectores originales por dimensión es: tensor([ 8383.7021, 13947.2715,  1898.5669,  3095.4729,  8237.3730])

Las medias por dimensión de BN inicialmente son: tensor([0., 0., 0., 0., 0.])
Las varianzas por dimensión de BN inicialmente son: tensor([1., 1., 1., 1., 1.])

La media por dimensión del resultado es: tensor([-1.99e-08, 3.97e-08, 0., 0., -3.97e-08], grad_fn=<MeanBackward1>)
La varianza por dimensión del resultado es: tensor([1., 1., 1., 1., 1.], grad_fn=<VarBackward0>)

Las medias por dimensión de BN ahora son: tensor([ 25.8616, -74.8571,   0.7456,  11.3224, -14.8267])
Las varianzas por dimensión de BN ahora son: tensor([12575.5527, 20920.9062,  2847.8503,  4643.2095, 12356.0605])

weight Parameter containing: tensor([1., 1., 1., 1., 1.], requires_grad=True)
bias Parameter containing:   tensor([0., 0., 0., 0., 0.], requires_grad=True)
```

### Analisis

---

#### El problema que resuelve BatchNorm

Los datos de entrada tienen varianza **enorme**: valores entre -74 y +26 en media, con varianzas de hasta 13,000. Esto causa problemas durante el entrenamiento:

```text
tensor_lineal = 100 * torch.randn((3, 5))
                ↑
           multiplica por 100 para simular datos con escala muy grande

Media por feature:    [ 25.86, -74.86,   0.75,  11.32, -14.83]
Varianza por feature: [8383,   13947,   1898,   3095,   8237 ]
```

Cuando los valores de activacion son muy grandes o muy pequenos, los gradientes se vuelven inestables (explotan o desaparecen). BatchNorm **normaliza** cada feature para que tenga media 0 y varianza 1 dentro del batch.

---

#### Lo que hace BatchNorm1d paso a paso

Para **cada feature** (cada columna del tensor), calcula la media y varianza de los 3 ejemplos del batch, y normaliza:

```text
Feature 0 (columna 0):  valores = [a, b, c]  (los 3 ejemplos del batch)

  1. media_batch = (a + b + c) / 3  = 25.86
  2. var_batch   = Σ(xi - media)²/3 = 8383.7  (varianza sin sesgo)
  3. x_norm      = (x - media_batch) / sqrt(var_batch + ε)
                 → media ≈ 0, varianza ≈ 1

  4. x_out = gamma × x_norm + beta
           = 1.0 × x_norm + 0.0   (gamma=1, beta=0 inicialmente)
           = x_norm                ← identidad al inicio
```

El resultado: la media por feature del tensor de salida es **≈ 0** (los valores como `-1.99e-08` son cero con error de punto flotante) y la varianza es exactamente **1.0**.

---

#### `momentum=None` — media acumulativa

El parametro `momentum` controla como se actualizan las **running stats** (estadisticas para inferencia):

```text
momentum=0.1 (default):
  running_mean = 0.9 × running_mean + 0.1 × media_batch
  → promedio movil exponencial, "olvida" el pasado gradualmente

momentum=None:
  running_mean = (1/N) × running_mean_viejo + (1 - 1/N) × media_batch
  donde N es el numero de batches procesados hasta ahora
  → en el primer batch: running_mean = media_batch
  → media acumulativa: promedia TODOS los batches con igual peso
```

En este caso es el **primer y unico batch**, por lo que con `momentum=None`:

```text
running_mean despues = media_batch = [ 25.86, -74.86,  0.75,  11.32, -14.83]
                                     (exactamente los valores del batch)
```

---

#### Por que running_var es diferente a la varianza del input

```text
Varianza del input  (unbiased=False, divide por N=3):   8383.7
running_var despues:                                    12575.6

Razon: 12575.6 / 8383.7 = 1.5 = N/(N-1) = 3/2
```

PyTorch guarda en `running_var` la **varianza sin sesgo** (divide por N-1=2), aunque la normalizacion interna use la varianza con sesgo (divide por N=3). Son dos estimadores distintos de la misma varianza poblacional — el sin sesgo es el estadisticamente correcto para estimaciones. La normalizacion usa el biased porque opera sobre el batch actual completo.

---

#### Las running stats: para que sirven

Las `running_mean` y `running_var` **no se usan durante el entrenamiento** — solo en inferencia:

```text
Entrenamiento (model.train()):
  → Normaliza usando la media y varianza DEL BATCH ACTUAL
  → Actualiza running_mean y running_var como "memoria"

Inferencia (model.eval()):
  → Normaliza usando running_mean y running_var acumuladas
  → NO usa el batch actual (podria ser 1 solo ejemplo)
```

Esto es crucial: en produccion se evalua un ejemplo a la vez. No hay "batch" del que calcular estadisticas. Las running stats son la solucion: acumulan el comportamiento estadistico visto durante todo el entrenamiento.

---

#### `weight` y `bias` — los parametros gamma y beta

```text
weight (gamma): tensor([1., 1., 1., 1., 1.], requires_grad=True)
bias   (beta):  tensor([0., 0., 0., 0., 0.], requires_grad=True)
```

Estos son los **parametros aprendibles** de BatchNorm:

- `gamma` (weight) = escala. Empieza en 1 → no escala nada al inicio
- `beta` (bias) = desplazamiento. Empieza en 0 → no desplaza nada al inicio

La formula completa: `salida = gamma × x_normalizado + beta`

Con los valores iniciales (gamma=1, beta=0), la salida es igual a `x_normalizado`. A medida que entrena, `gamma` y `beta` se ajustan via backpropagation para que la red pueda **recuperar la expresividad** que necesita. Si la red aprende que cierta feature debe tener media 5 y escala 2, gradualmente gamma→2 y beta→5.

```text
¿Por que gamma y beta tienen requires_grad=True pero running_mean/running_var no?

running_mean/var: se actualizan con una formula fija (media movil), no con backprop
gamma/beta:       se aprenden via gradiente, como cualquier otro peso de la red
```

---

## Celda 15: BatchNorm2d — normalizacion para imagenes (y el error de eps=0.0)

### Codigo

```python
# Batch Normalization 2d
tensor_imagenes = 20*torch.randn((4, 3, 2, 2))  # Una imagen con varianza alta
bn = BatchNorm2d(3, momentum=None, eps=0.0)  # 3 es el número de canales

print("La media de las imágenes originales por canal es: {}".format(tensor_imagenes.mean(dim=[0,2,3])))
print("La varianza de las imágenes originales por canal es: {}".format(tensor_imagenes.var(dim=[0,2,3], unbiased=True)))
...
tensor_bn = bn(tensor_imagenes)  # ← ERROR aqui
```

### Salida

```text
La media de las imágenes originales por canal es: tensor([-4.5984, -6.3164, -3.7078])
La varianza de las imágenes originales por canal es: tensor([234.6071, 733.1655, 537.8582])
Las medias por canal de BN inicialmente son: tensor([0., 0., 0.])
Las varianzas por canal de BN inicialmente son: tensor([1., 1., 1.])

ValueError: batch_norm eps must be positive, but got 0.0
```

### Analisis

---

#### Por que hay un error: el papel de `eps`

La formula de BatchNorm divide por la desviacion estandar:

```text
x_normalizado = (x - media) / sqrt(varianza + eps)
                                              ↑
                                   eps evita division por cero
```

Si `varianza = 0` (todos los valores del canal son identicos) y `eps = 0.0`:

```text
sqrt(0 + 0.0) = sqrt(0) = 0
x / 0  →  infinito o NaN  → el entrenamiento explota
```

Por eso PyTorch **exige** que `eps > 0`. Es una restriccion matematica, no arbitraria.

```text
eps=0.0   → ValueError: batch_norm eps must be positive
eps=1e-5  → correcto (default)
eps=1e-3  → correcto (mas estable para varianzas muy pequeñas)
```

El valor default `eps=1e-5` es suficientemente pequeño para no distorsionar la normalizacion, pero suficientemente grande para proteger contra division por cero.

---

#### BatchNorm2d vs BatchNorm1d — la diferencia clave

`BatchNorm1d` normaliza datos tabulares `(N, F)` — una media por feature.
`BatchNorm2d` normaliza imagenes `(N, C, H, W)` — una media por **canal**.

```text
BatchNorm2d agrupa TODAS las posiciones espaciales del mismo canal:

  (4, 3, 2, 2):
    4 imagenes en el batch
    3 canales (RGB)
    2×2 pixeles

  Para el canal R (canal 0):
    Cuantos valores hay?  4 imagenes × 2 × 2 pixeles = 16 valores
    Se calcula UNA media y UNA varianza para esos 16 valores.
    Todos los pixeles del canal R se normalizan con esa misma media/var.

  Para el canal G (canal 1): otros 16 valores, otra media/var independiente.
  Para el canal B (canal 2): otros 16 valores, otra media/var independiente.
```

Por eso el resultado es un vector de **3 valores** (uno por canal) en lugar de 16:

```text
tensor_imagenes.mean(dim=[0, 2, 3])
                     ↑  ↑  ↑
                     N  H  W   ← colapsa estas 3 dimensiones

Resultado: tensor de shape (3,)  ← una media por canal
```

---

#### Por que `dim=[0, 2, 3]` y no `dim=[1]`

Para calcular la media **por canal**, hay que promediar sobre todas las otras dimensiones: batch (0), alto (2), ancho (3). Dimension 1 es la de los canales — esa es la que se quiere mantener separada.

```text
dim=[0,2,3]:  colapsa batch + espacial → queda (C,) = 3 valores  ← lo que quiere BN2d
dim=[1]:      colapsaria los canales → queda (N,H,W) — no tiene sentido para BN
```

Comparacion con BatchNorm1d:

```text
BatchNorm1d sobre (N, F):
  Promedia sobre dim=[0]  → queda (F,) — una estadistica por feature

BatchNorm2d sobre (N, C, H, W):
  Promedia sobre dim=[0,2,3] → queda (C,) — una estadistica por canal
```

---

#### Lo que si funciona antes del error

Aunque el codigo falla al llegar a `bn(tensor_imagenes)`, lo de antes si se ejecuta y es informativo:

```text
running_mean inicial: [0., 0., 0.]   ← tres ceros, uno por canal
running_var inicial:  [1., 1., 1.]   ← tres unos, uno por canal

Media real del batch por canal:      [-4.60, -6.32, -3.71]
Varianza real del batch por canal:   [234.6, 733.2, 537.9]
```

Estos son valores muy alejados de (0, 1) — exactamente el problema que BatchNorm resolveria si `eps` fuera valido.

---

#### Correccion: usar eps por defecto

Para que funcione, simplemente no pasar `eps=0.0`:

```python
# Con error:
bn = BatchNorm2d(3, momentum=None, eps=0.0)

# Correcto:
bn = BatchNorm2d(3, momentum=None)           # usa eps=1e-5 por defecto
bn = BatchNorm2d(3, momentum=None, eps=1e-5) # equivalente, explicito
```

---

#### Resumen: BatchNorm1d vs BatchNorm2d

```text
Clase           Entrada          Normaliza sobre   Stats por
─────────────   ─────────────    ────────────────  ──────────────
BatchNorm1d(F)  (N, F)           dim 0 (batch)     feature (F stats)
BatchNorm2d(C)  (N, C, H, W)    dims 0,2,3         canal (C stats)

Uso tipico:
  BatchNorm1d → capas Linear (MLPs, clasificador final)
  BatchNorm2d → capas Conv2d (CNNs, AlexNet)
```

En **MiAlexNet**, se usa `BatchNorm2d` despues de las capas convolucionales (trabajan con imagenes `(N, C, H, W)`) y podria usarse `BatchNorm1d` despues de las capas `Linear` al final, aunque en la version del laboratorio se usa directamente Dropout.

---

### Celda 15 corregida — salida real de BatchNorm2d

Con `eps=0.0` eliminado, la celda corre correctamente:

```text
La media de las imágenes originales por canal es: tensor([ 3.6324,  8.0519, -1.4734])
La varianza de las imágenes originales por canal es: tensor([409.2384, 429.5763, 163.2007])

Las medias por canal de BN inicialmente son: tensor([0., 0., 0.])
Las varianzas por canal de BN inicialmente son: tensor([1., 1., 1.])

La media por canal es: tensor([-3.73e-09, 0., 0.], grad_fn=<MeanBackward1>)
La varianza por canal es: tensor([1.0667, 1.0667, 1.0667], grad_fn=<VarBackward0>)

Las medias por canal de BN ahora son: tensor([ 3.6324,  8.0519, -1.4734])
Las varianzas por canal de BN ahora son: tensor([409.2385, 429.5763, 163.2007])

weight: tensor([1., 1., 1.], requires_grad=True)
bias:   tensor([0., 0., 0.], requires_grad=True)
```

#### Por que la varianza del resultado es 1.0667 y no 1.0

En BatchNorm1d la varianza del resultado fue exactamente 1.0. Aqui es **1.0667**. La diferencia viene del numero de valores que se normalizan por canal:

```text
BatchNorm1d sobre (3, 5):
  Por feature: N = 3 muestras  →  var_biased × N/(N-1) = 1.0 × 3/2  = 1.5  (era 1.5)

BatchNorm2d sobre (4, 3, 2, 2):
  Por canal: N = 4 imagenes × 2 × 2 pixeles = 16 valores
  BN normaliza con var_biased (divide por N=16)
  .var(unbiased=True) mide con N-1=15
  ratio: 16/15 = 1.0667
```

BN siempre normaliza usando la varianza **sesgada** (divide por N), pero `.var(unbiased=True)` reporta la **sin sesgo** (divide por N-1). Cuanto mayor es N, menor es la diferencia entre ambas — con N=16 la discrepancia es solo del 6.7%, mientras que con N=3 era del 50%.

#### Comparacion BatchNorm1d vs BatchNorm2d en esta celda

```text
                      BN1d (celda 14)       BN2d (celda 15)
Input shape:          (3, 5)                (4, 3, 2, 2)
Stats por:            feature (5 stats)     canal (3 stats)
N por estadistica:    3                     4×2×2 = 16
Varianza reportada:   1.5                   1.0667
running_mean size:    5                     3
running_var size:     5                     3
```

El principio es identico — la unica diferencia es **sobre que dimensiones** se agrupan los valores para calcular las estadisticas.

---

## Celda 16: LayerNorm — normalizacion por muestra

### Codigo

```python
# Layer Normalization
tensor_lineal = 100*torch.randn((20, 100))  # 20 muestras, 100 features
ln = LayerNorm(100, eps=0, elementwise_affine=True)
print("Media del resultado: ", ln(tensor_lineal).mean(dim=1))
print("Desviación estándar del resultado: ", ln(tensor_lineal).std(dim=1, unbiased=False))

# Cambiemos gamma y beta a mano
ln.weight.copy_(torch.ones(100) * 100.0)  # gamma = 100
ln.bias.copy_(torch.ones(100) * 5.0)      # beta = 5

print("Media del resultado: ", ln(tensor_lineal).mean(dim=1))
print("Desviación estándar del resultado: ", ln(tensor_lineal).std(dim=1, unbiased=False))
```

### Salida

```text
--- Primera evaluacion (gamma=1, beta=0) ---
Media:  tensor([-4.77e-09, ..., -1.79e-08])   ← todos ≈ 0
Desv. std: tensor([1., 1., 1., ..., 1.])       ← todos = 1.0 exacto

weight: tensor([1., 1., ..., 1.], requires_grad=True)   ← 100 valores
bias:   tensor([0., 0., ..., 0.], requires_grad=True)   ← 100 valores

--- Segunda evaluacion (gamma=100, beta=5) ---
Media:  tensor([5., 5., 5., ..., 5.])          ← todos = 5.0 exacto
Desv. std: tensor([100., 100., ..., 100.])      ← todos = 100.0 exacto
```

### Analisis

---

#### LayerNorm normaliza por muestra, no por feature

La diferencia fundamental con BatchNorm:

```text
BatchNorm1d sobre (20, 100):
  Para cada feature (columna): promedia los 20 ejemplos del batch
  → estadistica por FEATURE, atraviesa el batch
  dim de normalizacion: dim=0 (el batch)
  Running stats: SI (acumula para inferencia)

LayerNorm sobre (20, 100):
  Para cada muestra (fila): promedia sus 100 features
  → estadistica por MUESTRA, independiente del batch
  dim de normalizacion: dim=1 (las features)
  Running stats: NO (no necesita)
```

Visualmente:

```text
Tensor (20, 100) — 20 muestras, 100 features:

BatchNorm: normaliza ↕ (por columna, a traves del batch)
  feature_0: media de los 20 valores de esa columna
  feature_1: media de los 20 valores de esa columna
  ...

LayerNorm: normaliza ↔ (por fila, a traves de las features)
  muestra_0: media de sus 100 features
  muestra_1: media de sus 100 features
  ...
```

---

#### Por que std=1.0000 exacto (vs 1.0667 en BN2d)

En BatchNorm2d la varianza medida fue 1.0667 porque BN normaliza con varianza **sesgada** pero se midio con `unbiased=True`. Aqui el resultado es exactamente 1.0 porque:

```text
BN2d: normaliza con var_biased (/ N=16), se midio con var_unbiased (/ N-1=15)
  → discrepancia: 16/15 = 1.0667

LayerNorm: normaliza con std_biased (/ N=100), se mide con std_unbiased=False (/ N=100)
  → misma formula → exactamente 1.0
```

Es la misma matematica — la diferencia es como se midio en el notebook.

---

#### `eps=0` funciona en LayerNorm pero no en BatchNorm

Interesante: en la celda anterior `eps=0.0` en BatchNorm2d causo `ValueError`. Aqui `eps=0` en LayerNorm **no da error**. PyTorch tiene restricciones distintas para cada modulo — LayerNorm no lanza excepcion con eps=0, aunque teoricamente podria haber division por cero si la std del vector fuera exactamente 0.

---

#### `elementwise_affine=True` — los parametros gamma y beta

Con `elementwise_affine=True` (default), LayerNorm tiene parametros aprendibles:

```text
weight (gamma): tensor de shape (100,) — un gamma por feature
bias   (beta):  tensor de shape (100,) — un beta por feature

Iniciados como: gamma=1, beta=0 → identidad (no transforma)
```

Con `elementwise_affine=False`, LayerNorm solo normaliza sin gamma ni beta — no tiene parametros aprendibles. Se usa cuando se quiere normalizacion pura sin permitir que la red ajuste la escala.

---

#### La demostracion de gamma y beta en accion

La segunda parte de la celda es la mas clara de todo el notebook para entender gamma/beta:

```python
ln.weight.copy_(torch.ones(100) * 100.0)  # gamma = 100 para las 100 features
ln.bias.copy_(torch.ones(100) * 5.0)      # beta = 5 para las 100 features
```

Resultado:

```text
Antes (gamma=1, beta=0):
  output = 1 × x_normalizado + 0 = x_normalizado
  media = 0,   std = 1

Despues (gamma=100, beta=5):
  output = 100 × x_normalizado + 5
  media = 100 × 0 + 5 = 5.0     ← beta desplaza la media
  std   = 100 × 1     = 100.0   ← gamma escala la desviacion
```

Todos los 20 ejemplos tienen exactamente la misma media (5.0) y la misma std (100.0) porque LayerNorm garantiza que cada muestra empieza con media=0, std=1 antes de aplicar gamma/beta.

```text
¿Porque la red aprende gamma y beta en vez de simplemente no normalizar?

Sin gamma/beta: la red esta FORZADA a tener activaciones con media=0, std=1.
Con gamma/beta: la red puede recuperar la escala y desplazamiento que necesite.
  → Normalizar primero estabiliza el entrenamiento.
  → Gamma/beta devuelven la capacidad expresiva.
```

---

#### `torch.no_grad()` al modificar parametros

```python
with torch.no_grad():
    ln.weight.copy_(new_weight1)
    ln.bias.copy_(new_weight2)
```

El `torch.no_grad()` es necesario porque `weight` y `bias` son tensores con `requires_grad=True`. Modificarlos directamente dentro de un contexto de gradientes causaria problemas con el grafo computacional. El bloque `no_grad()` suspende el tracking de gradientes temporalmente.

---

#### LayerNorm vs BatchNorm: resumen completo

```text
Caracteristica          BatchNorm              LayerNorm
──────────────────────  ─────────────────────  ──────────────────────
Normaliza sobre         Batch + espacial        Cada muestra sola
                        (a traves de muestras)  (a traves de features)
Running stats           SI (mean, var)          NO
Depende del batch       SI                      NO
Funciona con batch=1    Mal (solo 1 muestra)    Perfectamente
Uso tipico              CNNs, imagenes          Transformers, NLP, RNNs
Num. de estadisticas    1 por canal/feature     1 por muestra
```

LayerNorm es el estandar en arquitecturas de texto (GPT, BERT) porque en NLP el batch size puede ser variable y cada secuencia puede tener distinta longitud — BatchNorm no funciona bien en esos escenarios.

---

## Celda 17: MiAlexNet — definicion completa del modelo

### Codigo

```python
class MiAlexNet(nn.Module):
    def __init__(self):
        super(MiAlexNet, self).__init__()
        # 5 bloques convolucionales + flatten + 3 capas FC
        ...

m = MiAlexNet()
for n, p in m.named_parameters():
    print(n, p.shape)
```

### Salida (parametros)

```text
conv1.0.weight  torch.Size([96, 3, 11, 11])
conv1.0.bias    torch.Size([96])
conv2.0.weight  torch.Size([256, 96, 5, 5])
conv2.0.bias    torch.Size([256])
conv3.0.weight  torch.Size([384, 256, 3, 3])
conv3.0.bias    torch.Size([384])
conv4.0.weight  torch.Size([384, 384, 3, 3])
conv4.0.bias    torch.Size([384])
conv5.0.weight  torch.Size([256, 384, 3, 3])
conv5.0.bias    torch.Size([256])
fc6.0.weight    torch.Size([4096, 9216])
fc6.0.bias      torch.Size([4096])
fc7.0.weight    torch.Size([4096, 4096])
fc7.0.bias      torch.Size([4096])
fc8.0.weight    torch.Size([102, 4096])
fc8.0.bias      torch.Size([102])
```

### Analisis

---

#### Arquitectura general: 5 bloques conv + 3 capas FC

MiAlexNet es una adaptacion de AlexNet (2012) para clasificar **102 tipos de flores** en imagenes de 224×224:

```text
Input: (N, 3, 224, 224)
  │
  ├─ conv1: Conv(3→96, k=11, s=4, p=2) → MaxPool(3,2) → ReLU  → (N, 96, 27, 27)
  ├─ conv2: Conv(96→256, k=5, s=1, p=2) → MaxPool(3,2) → ReLU → (N, 256, 13, 13)
  ├─ conv3: Conv(256→384, k=3, s=1, p=1) → ReLU                → (N, 384, 13, 13)
  ├─ conv4: Conv(384→384, k=3, s=1, p=1) → ReLU                → (N, 384, 13, 13)
  ├─ conv5: Conv(384→256, k=3, s=1, p=1) → MaxPool(3,2) → ReLU → (N, 256, 6, 6)
  │
  ├─ Flatten: (N, 256, 6, 6) → (N, 9216)
  │
  ├─ fc6: Linear(9216→4096) → ReLU  → (N, 4096)
  ├─ fc7: Linear(4096→4096) → ReLU  → (N, 4096)
  └─ fc8: Linear(4096→102)          → (N, 102)  ← logits para 102 clases
```

---

#### Como calcular el tamaño de salida de cada Conv2d

La formula para el tamaño espacial despues de una convolucion:

```text
output = floor( (input - kernel + 2×padding) / stride ) + 1
```

Verificando cada capa:

```text
conv1: (224 - 11 + 2×2) / 4 + 1 = 217/4 + 1 = 54 + 1 = 55
  → despues MaxPool(3,2): (55 - 3) / 2 + 1 = 27  → (96, 27, 27) ✓

conv2: (27 - 5 + 2×2) / 1 + 1 = 26 + 1 = 27
  → despues MaxPool(3,2): (27 - 3) / 2 + 1 = 13  → (256, 13, 13) ✓

conv3: (13 - 3 + 2×1) / 1 + 1 = 13  → (384, 13, 13) ✓  (sin MaxPool)
conv4: (13 - 3 + 2×1) / 1 + 1 = 13  → (384, 13, 13) ✓  (sin MaxPool)

conv5: (13 - 3 + 2×1) / 1 + 1 = 13
  → despues MaxPool(3,2): (13 - 3) / 2 + 1 = 6   → (256, 6, 6) ✓
```

---

#### Por que conv3 y conv4 no tienen MaxPool

MaxPool reduce el tamaño espacial a la mitad. Despues de conv2 ya estamos en 13×13 — si se aplicara MaxPool despues de conv3 y conv4 quedaria 3×3 o 1×1, perdiendo demasiada informacion espacial.

```text
Estrategia: MaxPool solo cuando hay espacio suficiente
  conv1: 224→55→27   ← dos reducciones grandes al inicio
  conv2:  27→27→13   ← una reduccion mas
  conv3:  13→13      ← sin MaxPool (ya es pequeño)
  conv4:  13→13      ← sin MaxPool
  conv5:  13→13→6    ← una ultima reduccion antes del FC
```

Las conv3/conv4 agregan mas filtros (aumentan los canales de 256→384→384) sin reducir el espacio — extraen features mas complejas manteniendo la resolucion.

---

#### Flatten: el puente entre conv y FC

```text
self.flat = nn.Flatten()

Entrada: (N, 256, 6, 6)
Salida:  (N, 256×6×6) = (N, 9216)
```

Los bloques convolucionales producen un volumen 3D por imagen. Las capas `Linear` esperan un vector 1D. `Flatten` colapsa las tres dimensiones `(C, H, W)` en una sola, dejando solo el batch:

```text
(N, 256, 6, 6)  →  (N, 9216)
                       ↑
                  256 × 6 × 6 = 9216
```

Por eso `fc6` tiene `Linear(9216, 4096)` — recibe los 9216 valores aplanados.

---

#### Los parametros de Conv2d explicados

```python
nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=(4,4), padding=2)
```

```text
in_channels:  canales de entrada (3 = RGB)
out_channels: cuantos filtros/kernels aprende (96 patrones distintos)
kernel_size:  tamaño de cada filtro (11×11 pixeles)
stride:       cuanto se mueve el filtro por cada paso (4 = salta 4 pixeles)
padding:      pixeles de relleno alrededor del borde (2 = agrega 2 pixeles)

Forma del peso: (out_channels, in_channels, kH, kW) = (96, 3, 11, 11)
  96 filtros, cada uno de 3×11×11 = 363 valores
  + 96 bias
  = 96×363 + 96 = 34,944 parametros en conv1
```

---

#### Conteo total de parametros

```text
Capa           Pesos                   Params
─────────────  ──────────────────────  ───────────
conv1.0        96×3×11×11 + 96        = 34,944
conv2.0        256×96×5×5 + 256       = 614,656
conv3.0        384×256×3×3 + 384      = 885,120
conv4.0        384×384×3×3 + 384      = 1,327,488
conv5.0        256×384×3×3 + 256      = 884,992
fc6.0          4096×9216 + 4096       = 37,752,832
fc7.0          4096×4096 + 4096       = 16,781,312
fc8.0          102×4096 + 102         = 417,894
─────────────────────────────────────────────────
TOTAL                                 ≈ 58,699,238

→ ~58.7 millones de parametros
→ El 94% estan en las capas FC (fc6 + fc7 = 54.5M)
→ Las conv tienen solo 3.7M (el 6%)
```

---

#### Por que fc8 tiene 102 salidas

```python
self.fc8 = nn.Sequential(nn.Linear(4096, 102))
```

El dataset **Flowers-102** tiene exactamente **102 categorias de flores**. La ultima capa produce 102 numeros (logits) — uno por clase. La clase predicha es `argmax(logits)`.

No tiene ReLU ni Softmax al final — los logits crudos se pasan directamente a `CrossEntropyLoss`, que aplica log-softmax internamente.

---

#### El Dropout comentado

```python
self.fc6 = nn.Sequential( #nn.Dropout(),
                          nn.Linear(9216, 4096),
                          nn.ReLU())
```

El Dropout esta **comentado**. En AlexNet original, el dropout (p=0.5) era fundamental para evitar overfitting en las capas FC enormes. Esta version lo omite — quizas para simplificar el laboratorio o para comparar con/sin regularizacion (Experimento 2 de clase 8).

---

#### La notacion `conv1.0.weight` en named_parameters

El nombre `conv1.0.weight` significa:

```text
conv1   → el atributo self.conv1 (un Sequential)
  .0    → el primer modulo dentro del Sequential (Conv2d, indice 0)
    .weight  → el tensor de pesos de ese Conv2d

Si hubiera BatchNorm:
  conv1.0.weight  → Conv2d weights
  conv1.1.weight  → BatchNorm2d gamma (si estuviera en indice 1)
  conv1.1.bias    → BatchNorm2d beta
  conv1.2         → ReLU (no tiene parametros, no aparece)
```

Los modulos sin parametros (ReLU, MaxPool, Flatten) no aparecen en `named_parameters()` porque no tienen nada que aprender.

---

## Celda 18: Verificar que MiAlexNet funciona con un forward pass

### Codigo

```python
fotos_de_mentira = torch.randn(15, 3, 224, 224)

m(fotos_de_mentira).shape
```

### Salida

```text
torch.Size([15, 102])
```

### Analisis

---

#### Lo que hace esta celda

Crea un tensor de **ruido aleatorio** con la forma correcta de un batch de imagenes y lo pasa por la red completa. Si la arquitectura esta bien definida (dimensiones compatibles en todos los pasos), la red corre sin errores y devuelve el shape esperado.

```text
Input:   torch.randn(15, 3, 224, 224)
           ↑   ↑  ↑    ↑
           │   │  │    └── ancho (W)
           │   │  └─────── alto (H)
           │   └────────── canales RGB
           └────────────── 15 imagenes en el batch

Output: torch.Size([15, 102])
           ↑         ↑
           │         └── 102 logits (una puntuacion por clase de flor)
           └──────────── 15 imagenes (el batch se mantiene)
```

---

#### El recorrido completo de las formas

```text
Entrada:         (15,   3, 224, 224)

conv1 (Conv+Pool+ReLU):   (15,  96,  27,  27)
conv2 (Conv+Pool+ReLU):   (15, 256,  13,  13)
conv3 (Conv+ReLU):        (15, 384,  13,  13)
conv4 (Conv+ReLU):        (15, 384,  13,  13)
conv5 (Conv+Pool+ReLU):   (15, 256,   6,   6)

Flatten:                  (15, 9216)

fc6 (Linear+ReLU):        (15, 4096)
fc7 (Linear+ReLU):        (15, 4096)
fc8 (Linear):             (15, 102)

Salida:          (15, 102)
```

---

#### Por que "fotos de mentira" son suficientes para probar la arquitectura

Los pesos de la red son aleatorios (no entrenados), y los datos son ruido — no se obtendra ninguna prediccion util. Pero eso no importa: el objetivo aqui es **verificar que las dimensiones son compatibles** en toda la cadena de capas.

Esto es una practica estandar al diseñar redes: antes de gastar horas entrenando, se hace un "sanity check" con datos falsos del shape correcto. Si hay un error dimensional (por ejemplo, `Linear(9216, 4096)` pero el flatten produce 9220 valores), se detecta inmediatamente.

---

#### El resultado confirma el diseño

```text
torch.Size([15, 102])
  ↑             ↑
  batch=15      102 clases (Flowers-102)

→ La red esta lista para entrenar.
→ Cada imagen produce 102 logits.
→ argmax(logits) da la clase predicha (0 a 101).
```

En el entrenamiento real, estos 102 logits se pasan a `CrossEntropyLoss` junto con las etiquetas verdaderas (enteros entre 0 y 101), y la red aprende a hacer los logits de la clase correcta mas altos que los demas.

---

## Celda 19: Manejo de datos — Dataset y DataLoader

### Texto del notebook

> Los datasets en deep learning son demasiado grandes para mantener en memoria. PyTorch resuelve esto con dos abstracciones:
>
> - `Dataset`: interfaz para acceder a los datos fisicos.
> - `DataLoader`: agrupa elementos del Dataset en batches.
>
> Para datasets personalizados, se hereda de `torch.utils.data.Dataset` e implementan `__len__` y `__getitem__`.

### Codigo

```python
from torchvision.datasets import MNIST, CIFAR10

mnist_train = MNIST(root=".", train=True, download=True)
cifar = CIFAR10(root=".", train=True, download=True)
```

### Salida

```text
Descarga de MNIST: ~9.91 MB + 28.9 kB + 1.65 MB + 4.54 kB
Descarga de CIFAR10: ~170 MB
```

### Analisis

---

#### El problema: los datos no caben en RAM

En machine learning clasico los datasets son pequeños — caben enteros en memoria. En deep learning no:

```text
MNIST:     60,000 imagenes 28×28 grises       ≈ 47 MB   (cabe en RAM)
CIFAR-10:  50,000 imagenes 32×32 color        ≈ 170 MB  (cabe en RAM)
Flowers:   8,000  imagenes 224×224 color      ≈ 1 GB    (cabe en RAM)
ImageNet:  1.2M   imagenes 224×224 color      ≈ 150 GB  (NO cabe)
Videos:    horas de video a 30fps + audio     ≈ TB      (imposible en RAM)
```

La solucion: **cargar los datos de a batches** (grupos de 32 o 64 imagenes a la vez). Mientras la GPU entrena con el batch actual, el CPU prepara el siguiente batch del disco.

---

#### La separacion Dataset / DataLoader

```text
Dataset                          DataLoader
─────────────────────────────    ──────────────────────────────────────
"Sabe como leer un dato"         "Sabe como armar batches de datos"

- Abre archivos del disco         - Agrupa N elementos del Dataset
- Aplica transformaciones         - Mezcla aleatoriamente (shuffle)
- Devuelve (imagen, etiqueta)     - Paraleliza la carga (num_workers)
- Se accede por indice [i]        - Se usa en un for loop de entrenamiento
```

Analogia: Dataset es la bodega con los productos. DataLoader es el empleado que saca cajas de la bodega y las pone en el mostrador de a grupos.

---

#### Estructura de un Dataset personalizado

```python
from torch.utils.data import Dataset

class MiDataset(Dataset):
    def __init__(self, archivos, etiquetas, transform=None):
        self.archivos = archivos      # lista de rutas a las imagenes
        self.etiquetas = etiquetas    # lista de labels
        self.transform = transform

    def __len__(self):
        return len(self.archivos)     # cuantos elementos tiene el dataset

    def __getitem__(self, index):
        imagen = cargar_imagen(self.archivos[index])   # leer del disco
        if self.transform:
            imagen = self.transform(imagen)
        return imagen, self.etiquetas[index]           # devolver (X, y)
```

El `DataLoader` llama a `__len__` para saber cuantos batches hay, y llama a `__getitem__` con indices distintos para construir cada batch.

---

#### Datasets predefinidos en torchvision

`torchvision.datasets` incluye los datasets mas usados en investigacion de vision:

```text
MNIST:     digitos escritos a mano, 10 clases, 28×28 grises
CIFAR-10:  objetos cotidianos, 10 clases, 32×32 color
CIFAR-100: objetos, 100 clases, 32×32 color
ImageNet:  objetos, 1000 clases, resolucion variable (requiere descarga manual)
Flowers102: flores, 102 clases, resolucion variable (el del laboratorio)
COCO:      deteccion de objetos y segmentacion
STL10:     objetos, 10 clases, 96×96 color
```

Todos tienen la misma interfaz: `Dataset(root, train, download, transform)`.

---

#### Los parametros de MNIST/CIFAR10

```python
MNIST(root=".",         # donde guardar los archivos descargados
      train=True,       # True=entrenamiento (60K), False=test (10K)
      download=True)    # si no existe, descargarlo automaticamente
```

```text
MNIST tiene 4 archivos:
  train-images (9.91 MB): las 60,000 imagenes de entrenamiento
  train-labels (28.9 kB): las 60,000 etiquetas
  test-images  (1.65 MB): las 10,000 imagenes de test
  test-labels  (4.54 kB): las 10,000 etiquetas

CIFAR-10 es un solo archivo comprimido (170 MB) con los 50,000 items.
```

---

#### Como se usa en el loop de entrenamiento

El patron completo Dataset → DataLoader → loop:

```python
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 1. Transformaciones (preprocesamiento)
transform = transforms.Compose([
    transforms.ToTensor(),                          # PIL/numpy → tensor
    transforms.Normalize((0.5,), (0.5,))           # normalizar pixeles
])

# 2. Dataset (sabe leer los datos)
dataset = MNIST(root=".", train=True, transform=transform)

# 3. DataLoader (arma batches)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# 4. Loop de entrenamiento
for imagenes, etiquetas in loader:
    # imagenes: shape (32, 1, 28, 28)
    # etiquetas: shape (32,) con valores 0-9
    pred = modelo(imagenes)
    loss = criterio(pred, etiquetas)
    ...
```

`num_workers=2` significa que 2 procesos paralelos preparan los batches en CPU mientras la GPU entrena — esto elimina el cuello de botella de carga de datos.

---

## Celda 20: `len(mnist_train)` → 60000

### Codigo

```python
len(mnist_train)
```

### Salida

```text
60000
```

### Analisis

`len()` sobre un Dataset llama a su metodo `__len__`. Devuelve **60,000** — el numero de imagenes en el split de entrenamiento de MNIST.

```text
MNIST completo:
  mnist_train (train=True):  60,000 imagenes  ← este
  mnist_test  (train=False): 10,000 imagenes

Total: 70,000 imagenes de digitos escritos a mano
```

Este numero es importante para el DataLoader: con batch_size=32 y 60,000 imagenes, cada epoch tiene `ceil(60000 / 32) = 1875` pasos (batches). El DataLoader usa `__len__` internamente para calcular cuantos batches hay por epoch y cuando detenerse.

---

## Celda 21: Acceder a un elemento del dataset

### Codigo

```python
mnist_train[459]

# display de la imagen:
from IPython.display import display
display(mnist_train[459][0])
```

### Salida

```text
(<PIL.Image.Image image mode=L size=28x28>, 3)

[imagen del digito "3" escrito a mano]
```

### Analisis

---

#### `mnist_train[459]` llama a `__getitem__(459)`

Acceder con corchetes a un Dataset es equivalente a llamar `__getitem__`. Devuelve una **tupla de dos elementos**:

```text
mnist_train[459]  →  (imagen, etiqueta)
                      ↑        ↑
                      PIL      3 (el digito en esa imagen)
```

Esto es el patron estandar de todos los datasets de PyTorch: `(X, y)` donde `X` es la entrada y `y` es el target.

---

#### `PIL.Image.Image mode=L size=28x28`

La imagen viene en formato **PIL** (Python Imaging Library), no como tensor de PyTorch:

```text
PIL.Image.Image:  objeto de imagen de Python, se puede ver, guardar, rotar, etc.
mode=L:           "Luminance" = escala de grises (1 canal, valores 0-255)
size=28x28:       resolucion de la imagen

No es un tensor todavia — no tiene .shape, no puede ir directamente a la red.
```

¿Por que PIL y no tensor? Porque desde PIL es mas facil aplicar transformaciones de data augmentation (rotar, recortar, voltear) antes de convertir. La conversion a tensor ocurre al final del pipeline de transformaciones.

---

#### El pipeline de transformacion: PIL → tensor

Sin `transform`, el dataset devuelve PIL. Con `transform`, convierte automaticamente:

```python
# Sin transform (como en esta celda):
mnist_train = MNIST(root=".", train=True, download=True)
mnist_train[459]  →  (PIL_image, 3)   ← imagen en PIL, no usable por la red

# Con transform:
transform = transforms.Compose([
    transforms.ToTensor(),                  # PIL (28,28) → tensor (1,28,28), float [0,1]
    transforms.Normalize((0.5,), (0.5,))   # normalizar a [-1, 1]
])
mnist_con_transform = MNIST(root=".", train=True, transform=transform)
mnist_con_transform[459]  →  (tensor(1,28,28), 3)   ← listo para la red
```

```text
PIL mode=L, size=28x28:    valores enteros 0-255, shape implicita (28,28)
transforms.ToTensor():     → tensor float32 [0.0, 1.0], shape (1, 28, 28)
                                                                ↑
                                                            canal 1 (grises)
```

---

#### La etiqueta: `3`

El segundo elemento de la tupla es `3` — un entero de Python que indica el digito representado en la imagen. Es el **ground truth** (verdad de campo): lo que la red deberia predecir.

```text
Clases de MNIST: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9  (10 digitos)
Clases de CIFAR-10: avion, auto, pajaro, gato, ciervo, perro, rana, caballo, barco, camion

En ambos casos, la etiqueta es un entero (indice de clase).
CrossEntropyLoss espera exactamente este formato: un entero por muestra.
```

---

#### `display()` — solo para Colab/Jupyter

```python
from IPython.display import display
display(mnist_train[459][0])
```

`display()` es una funcion de IPython para mostrar objetos (imagenes, HTML, etc.) directamente en el notebook. No es PyTorch — es la infraestructura del cuaderno.

`mnist_train[459][0]` accede al primer elemento de la tupla (la imagen PIL). Como PIL sabe como renderizarse en Jupyter, `display()` la muestra como imagen. En un script de Python normal, habria que usar `imagen.save("foto.png")` o `imagen.show()` en su lugar.

---

## Celda 22: Aplicar `ToTensor()` — PIL a tensor

### Codigo

```python
from torchvision.transforms import ToTensor
mnist_train = MNIST(root=".", train=True, download=True, transform=ToTensor())

mnist_train[459]
```

### Salida (resumida)

```text
(tensor([[[0.0000, 0.0000, ..., 0.0000],   ← filas de ceros (margen superior)
           ...
          [0.0000, 0.0078, 0.2314, 0.8980, 0.9961, 1.0000, ...],  ← trazo del 3
          [0.0000, 0.3529, 0.9922, 0.9922, 0.9922, 0.9922, ...],
          ...
          ]]),
 3)
```

Shape del tensor: `(1, 28, 28)` — label: `3`

### Analisis

---

#### Lo que hace `ToTensor()`

```text
PIL Image (mode=L, size=28x28)        tensor float32 (1, 28, 28)
  valores: enteros 0-255         →      valores: float 0.0 - 1.0
  shape implicita: (28, 28)             shape: (1, 28, 28)
  no usable por PyTorch                 listo para la red
```

Dos cosas ocurren:
1. **Cambio de forma**: `(H, W)` → `(C, H, W)` agrega la dimension de canal (1 para grises, 3 para RGB)
2. **Normalizacion**: divide cada pixel por 255 → valores en `[0.0, 1.0]`

---

#### Leer la imagen del tensor

Los valores del tensor codifican el brillo de cada pixel:

```text
0.0000  →  negro (fondo, sin tinta)
1.0000  →  blanco (tinta maxima)
0.5000  →  gris medio

Mirando las primeras filas del tensor:
  Filas 0-5: todo 0.0000 → margen superior negro (no hay tinta)
  Fila 6:    [0, 0, ..., 0.0078, 0.2314, 0.8980, 0.9961, 1.0000, ...]
              ↑ empieza el trazo superior del "3"

Mirando las ultimas filas:
  Filas 24-25: [0, 0.7412, 0.9922, 0.9922, 0.9922, ...]
               ↑ trazo inferior del "3"
  Filas 26-27: todo 0.0000 → margen inferior
```

El "3" es visible en los datos: los ceros forman el fondo negro, y los valores altos (0.9922, 1.0000) forman el trazo blanco del digito.

---

#### Por que 0.9922 y no 1.0000

La mayoria de los trazos del digito son `0.9922` en vez de `1.0000` exacto. Esto es porque:

```text
pixel original = 253 (casi blanco, pero no 255)
ToTensor: 253 / 255 = 0.9922...

pixel original = 255 (blanco puro)
ToTensor: 255 / 255 = 1.0000
```

Los digitos escritos a mano tienen antialiasing — los bordes del trazo tienen valores intermedios, no binarios.

---

#### El shape `(1, 28, 28)` vs `(3, 224, 224)` del laboratorio

```text
MNIST (grises):       (1, 28, 28)    — 1 canal, 28×28 pixeles
Flowers (color):      (3, 224, 224)  — 3 canales RGB, 224×224 pixeles

Para pasar MNIST por MiAlexNet habria que:
  1. Convertir 1 canal → 3 canales (la red espera RGB)
  2. Redimensionar 28×28 → 224×224
  → Por eso MiAlexNet usa el dataset Flowers, no MNIST.
```

---

#### La tupla completa

```text
mnist_train[459]  →  (tensor, 3)
                       ↑       ↑
                  shape (1,28,28)   etiqueta: digito 3
                  float32 [0,1]     int: indice de clase

Para pasarla al modelo:
  imagen, etiqueta = mnist_train[459]
  imagen = imagen.unsqueeze(0)   # → (1, 1, 28, 28)  agrega batch
  # o usando DataLoader que lo hace automaticamente en batches
```

El DataLoader se encargara de apilar N de estas tuplas para formar batches `(N, 1, 28, 28)` listos para entrenar.

---

## Celda 23: Dataset personalizado — clase Flowers

### Codigo

```python
!wget https://www.dropbox.com/.../flowers.tar.gz -q --show-progress
!tar -xzf flowers.tar.gz

class Flowers(torch.utils.data.Dataset):
    def __init__(self, root, transform=None): ...
    def armar_indices(self, root): ...
    def obtener_imagen(self, archivo): ...
    def __getitem__(self, idx): ...
    def __len__(self): ...

transforms = Compose([Resize((224,224)), ToTensor()])
f = Flowers('flowers_dataset/train', transform=transforms)
print(f[0])
```

### Salida

```text
(tensor([[[0.1373, 0.1333, ...],   ← canal R, 224×224
          ...],
         [[0.1294, 0.1255, ...],   ← canal G, 224×224
          ...],
         [[0.1333, 0.1294, ...],   ← canal B, 224×224
          ...]]),
 78)
```

Shape del tensor: `(3, 224, 224)` — label: `78`

### Analisis

---

#### La estructura del dataset en disco

El dataset Flowers esta organizado en carpetas, una por clase:

```text
flowers_dataset/train/
  1/
    image_00001.jpg
    image_00002.jpg
    ...
  2/
    image_00050.jpg
    ...
  78/
    image_00200.jpg    ← esta imagen (la primera del dataset al iterar)
    ...
  102/
    image_00800.jpg
    ...

La carpeta = la clase. El nombre de la carpeta es el indice de clase (1-102).
```

---

#### `armar_indices(root)` — construir el indice del dataset

```python
def armar_indices(self, root):
    for clase in listdir(root):          # itera sobre carpetas (1, 2, ..., 102)
        directorio = join(root, clase)
        for archivo in listdir(directorio):   # itera sobre imagenes de esa clase
            lista_imagenes.append(archivo)    # guarda el nombre del archivo
            imgs_to_class.append(int(clase))  # guarda la clase como entero
```

Resultado: dos listas paralelas de longitud total (numero de imagenes):

```text
lista_imagenes:  ['image_00001.jpg', 'image_00002.jpg', ..., 'image_00800.jpg', ...]
imgs_to_class:   [1,                 1,                 ...,  78,               ...]
                  ↑ imagen 0 es clase 1       imagen N es clase 78 ↑
```

El indice `idx` en `__getitem__` apunta a la misma posicion en ambas listas.

---

#### `__getitem__(idx)` — leer y transformar una imagen

```python
def __getitem__(self, idx):
    nombre_archivo = self.imagenes[idx]       # nombre del archivo
    clase = self.imgs_to_class[idx]           # su clase (entero)
    ruta_img = join(self.root, str(clase), nombre_archivo)  # ruta completa
    img = self.obtener_imagen(ruta_img)       # abre la imagen con PIL
    if self.transform is not None:
        img = self.transform(img)             # aplica transformaciones
    return img, clase                         # devuelve (tensor, label)
```

El recorrido para `idx=0`:

```text
nombre_archivo = 'image_00200.jpg'   (el primero que encontro listdir)
clase          = 78
ruta           = 'flowers_dataset/train/78/image_00200.jpg'
img (PIL)      → Resize(224,224) → ToTensor() → tensor (3, 224, 224)
return         → (tensor, 78)
```

---

#### Las transformaciones: `Compose([Resize, ToTensor])`

```python
transforms = Compose([Resize((224,224)), ToTensor()])
```

`Compose` aplica las transformaciones en orden, una tras otra:

```text
PIL Image (resolucion original, ej. 500×400)
  ↓ Resize((224, 224))
PIL Image (224×224)          ← redimensiona manteniendo los 3 canales
  ↓ ToTensor()
tensor float32 (3, 224, 224) ← convierte a tensor, normaliza 0-255 → 0.0-1.0
```

¿Por que `Resize(224, 224)` primero? Las imagenes de Flowers tienen distintas resoluciones originales. `MiAlexNet` espera exactamente `(3, 224, 224)` — si las imagenes tuvieran distintos tamaños, no se podrian apilar en un batch. `Resize` las uniformiza.

---

#### El tensor de salida `(3, 224, 224)`

```text
Canal 0 (R): tensor de 224×224 floats, valores del canal rojo
Canal 1 (G): tensor de 224×224 floats, valores del canal verde
Canal 2 (B): tensor de 224×224 floats, valores del canal azul

Mirando los valores: [0.1373, 0.1333, 0.1451, ...]
  Son todos relativamente bajos (0.1 - 0.3) → la imagen es bastante oscura
  El fondo de la imagen de la flor 78 parece ser oscuro/negro
```

---

#### El label `78` — indice de clase

El segundo elemento de la tupla es `78` — el nombre de la carpeta convertido a entero. Es uno de los 102 tipos de flores del dataset Oxford Flowers-102.

```text
El numero 78 no tiene significado especial — es simplemente el identificador
de esa categoria de flor en el dataset original de Oxford.

Para CrossEntropyLoss, la red necesita que las clases sean enteros de 0 a N-1.
Ojo: Flowers-102 usa clases de 1 a 102, NO de 0 a 101.
Si el modelo predice clase 0 nunca acertaria (no existe clase 0).
→ Tipicamente se resta 1 a todas las etiquetas antes de entrenar: clase - 1.
```

---

#### Comparacion: Dataset predefinido vs personalizado

```text
                   MNIST (predefinido)        Flowers (personalizado)
Implementacion:    Ya existe en torchvision   Escrita por nosotros
__len__:           Automatico                 return len(self.imagenes)
__getitem__:       Automatico                 abre PIL, transforma, devuelve
Organizacion:      Binario comprimido         Carpetas por clase
Flexibilidad:      Fija                       Podemos adaptarla (augmentation, etc.)
```

La ventaja del Dataset personalizado es total control: se puede agregar data augmentation (rotaciones, flips, cambio de brillo), manejar datasets con estructura no estandar, combinar multiples fuentes de datos, etc.

---

## Celda 24: `len(f)` → 5687

### Codigo

```python
len(f)
```

### Salida

```text
5687
```

### Analisis

El split de entrenamiento de Flowers-102 tiene **5,687 imagenes**, distribuidas en 102 clases. Promedio de ~55 imagenes por clase, aunque la distribucion no es perfectamente uniforme.

```text
Flowers-102 completo:
  train: 5,687 imagenes  ← este
  val:   ~1,020 imagenes (10 por clase)
  test:  ~6,149 imagenes

Comparacion de tamaños:
  MNIST train:   60,000 imagenes
  CIFAR-10 train: 50,000 imagenes
  Flowers train:   5,687 imagenes  ← mucho mas pequeño

→ Con tan pocas imagenes para 102 clases, el riesgo de overfitting es alto.
→ Por eso el Transfer Learning (Actividad 4 del laboratorio) es tan efectivo aqui:
   en vez de entrenar AlexNet desde cero con 5,687 imagenes,
   se parte de pesos ya entrenados en ImageNet (1.2M imagenes).
```

Con batch_size=32 y 5,687 imagenes: `ceil(5687 / 32) = 178` pasos por epoch.

---

## Celda 25: DataLoader — iterar por una epoch

### Codigo

```python
from torch.utils.data import DataLoader
train_dl = DataLoader(f, batch_size=128, shuffle=True)

for n_batch, (x, target) in enumerate(train_dl):
    print("\rN_Batch: {} input: {} - Label:{}".format(n_batch, x.shape, target.shape), end="")
```

### Salida

```text
N_Batch: 44 input: torch.Size([55, 3, 224, 224])- Label:torch.Size([55])
```

### Analisis

---

#### `DataLoader(f, batch_size=128, shuffle=True)`

Toma el Dataset `f` (5,687 imagenes) y lo prepara para iterar en batches:

```text
batch_size=128: agrupa 128 imagenes por batch
shuffle=True:   mezcla aleatoriamente el orden cada epoch
                → evita que la red "memorice" el orden de los datos
```

---

#### El loop: `for n_batch, (x, target) in enumerate(train_dl)`

```text
train_dl produce tuplas (batch_imagenes, batch_labels) en cada iteracion.
enumerate agrega el indice del batch (0, 1, 2, ...).

En cada paso:
  n_batch: indice del batch actual (0 a 44)
  x:       tensor (128, 3, 224, 224) — imagenes del batch
  target:  tensor (128,) — etiquetas del batch

  Excepto el ultimo batch: (55, 3, 224, 224) y (55,)
```

---

#### Por que el ultimo batch tiene 55 imagenes

```text
5,687 imagenes / 128 por batch = 44.43 batches

  Batches 0 a 43: 44 batches completos de 128 imagenes = 44 × 128 = 5,632
  Batch 44:       el resto = 5,687 - 5,632 = 55 imagenes

Total batches: 45 (indices 0 a 44) ← una epoch completa
```

Por defecto el DataLoader incluye este ultimo batch incompleto. Si se quiere descartar, se usa `drop_last=True`:

```python
DataLoader(f, batch_size=128, shuffle=True, drop_last=True)
# → 44 batches de exactamente 128, el 55 se descarta
```

---

#### El truco de `\r` y `end=""`

```python
print("\rN_Batch: ...", end="")
```

`\r` (carriage return) mueve el cursor al inicio de la linea sin bajar. `end=""` evita que `print` agregue un salto de linea. El efecto: **sobreescribe la misma linea** en cada iteracion — funciona como barra de progreso:

```text
Iteracion 0:  N_Batch: 0 input: torch.Size([128, 3, 224, 224])...
Iteracion 1:  N_Batch: 1 input: torch.Size([128, 3, 224, 224])...  ← misma linea
...
Iteracion 44: N_Batch: 44 input: torch.Size([55, 3, 224, 224])...  ← lo que queda
```

En el notebook de Colab, la salida final muestra solo el ultimo estado (batch 44).

---

#### Este loop = una epoch de entrenamiento (sin actualizar pesos)

El loop recorre las 5,687 imagenes exactamente una vez. En un entrenamiento real, dentro del loop estarian:

```python
for n_batch, (x, target) in enumerate(train_dl):
    x = x.to(device)
    target = target.to(device)

    pred = model(x)                      # forward pass
    loss = criterion(pred, target)       # calcular loss

    optimizer.zero_grad()                # limpiar gradientes anteriores
    loss.backward()                      # backward pass
    optimizer.step()                     # actualizar pesos

# → al terminar el loop: una epoch completa
# → entrenar 50 epochs = recorrer los 5,687 ejemplos 50 veces
```

Este es el patron estandar del training loop de PyTorch que se vera en la siguiente parte del laboratorio.

---

## Celda 26: Funcion de perdida — CrossEntropyLoss

### Texto del notebook

> Para encontrar los pesos optimos se minimiza la funcion de perdida. Las mas comunes son:
> - `CrossEntropyLoss`: para clasificacion — mide distancia entre distribuciones de probabilidad.
> - `MSELoss`: para regresion — error cuadratico medio.

### Codigo

```python
from torch.nn import CrossEntropyLoss
funcion_perdida = CrossEntropyLoss()

loss = funcion_perdida(output, target)  # No correr este código, va a dar error!
```

### Salida

```text
NameError: name 'output' is not defined
```

### Analisis

---

#### El error es intencional

El comentario del notebook lo advierte: `# No correr este código, va a dar error!`. La celda es solo un ejemplo de sintaxis — `output` y `target` no estan definidas en este punto del notebook. El error es esperado.

---

#### `CrossEntropyLoss()` — para clasificacion multiclase

```text
funcion_perdida = CrossEntropyLoss()
loss = funcion_perdida(output, target)

  output: tensor (N, C) — logits crudos de la red, uno por clase
          ejemplo: (32, 102) para un batch de 32 con 102 clases de flores

  target: tensor (N,) — etiquetas como enteros, valores entre 0 y C-1
          ejemplo: (32,) con valores en [0, 101]

  loss: escalar — un solo numero, el error promedio del batch
```

Internamente, CrossEntropyLoss hace dos cosas:

```text
1. log_softmax(output): convierte logits a log-probabilidades
2. NLLLoss: mide cuan baja es la log-probabilidad de la clase correcta

Equivalente a:
  probs = softmax(output)            → distribucion de probabilidad
  loss  = -log(probs[clase_correcta]) → castigar si la clase correcta tiene prob baja
```

Cuanto menor es la probabilidad que la red asigna a la clase correcta, mayor es el loss.

---

#### CrossEntropyLoss vs MSELoss — cuando usar cada una

```text
Problema           Tipo de output    Loss correcta
─────────────────  ────────────────  ───────────────────────────
Clasificar flores  102 clases        CrossEntropyLoss
Reconocer digitos  10 clases         CrossEntropyLoss
Precio de casa     valor continuo    MSELoss
Score de sentim.   valor en [-1,1]   MSELoss
¿Compra? (si/no)   probabilidad      BCEWithLogitsLoss

Regla: CATEGORIAS → CrossEntropy.  NUMEROS → MSE.
```

---

#### Por que NO poner Softmax antes de CrossEntropyLoss

`CrossEntropyLoss` aplica softmax internamente. Si la red ya aplica Softmax y luego se pasa a CrossEntropyLoss, se esta aplicando softmax dos veces — el resultado es incorrecto y el entrenamiento falla silenciosamente:

```python
# MAL:
class Red(nn.Module):
    def forward(self, x):
        return nn.Softmax(dim=1)(self.fc(x))   # softmax aqui

loss = CrossEntropyLoss()(output, target)       # y softmax de nuevo dentro

# BIEN:
class Red(nn.Module):
    def forward(self, x):
        return self.fc(x)                       # logits crudos

loss = CrossEntropyLoss()(output, target)       # softmax solo aqui
```

La red siempre debe devolver **logits crudos** cuando se usa CrossEntropyLoss. Eso es exactamente lo que hace `fc8` de MiAlexNet: `nn.Linear(4096, 102)` sin activacion al final.

---

## Celda 27: Optimizador — SGD y Adam

### Texto del notebook

> Para calcular los gradientes y actualizar los pesos necesitamos un algoritmo de optimizacion. El estandar es **Stochastic Gradient Descent (SGD)**, ya implementado en `torch.optim`.

### Codigo

```python
from torch.optim import SGD, Adam

optimizer = SGD(model.parameters())  # vincula el optimizador a los params del modelo
```

### Salida

```text
NameError: name 'model' is not defined
```

### Analisis

---

#### El error es intencional — mismo patron que la celda anterior

`model` no esta definido en este punto del notebook. Es solo un ejemplo de sintaxis. En el training loop real se usara con `m` (la instancia de MiAlexNet).

---

#### Por que el optimizador necesita `model.parameters()`

El optimizador necesita saber **que tensores actualizar** despues de cada `backward()`. Al pasarle `model.parameters()`, le da acceso a todos los pesos y bias del modelo:

```text
model.parameters() devuelve un iterador sobre todos los tensores con requires_grad=True:
  conv1.0.weight  (96, 3, 11, 11)
  conv1.0.bias    (96,)
  conv2.0.weight  (256, 96, 5, 5)
  ...
  fc8.0.weight    (102, 4096)
  fc8.0.bias      (102,)
  → 58.7 millones de parametros en total para MiAlexNet
```

---

#### SGD — Stochastic Gradient Descent

```text
Algoritmo:
  1. Calcular gradiente del loss respecto a cada parametro (backward())
  2. Mover cada parametro en la direccion opuesta al gradiente:
       parametro = parametro - lr × gradiente

  lr (learning rate): cuanto se mueve en cada paso
  Tipico: lr = 0.01 o 0.001
```

"Stochastic" (estocastico) porque en vez de calcular el gradiente sobre todo el dataset (muy lento), lo calcula sobre un batch aleatorio en cada paso — de ahi el DataLoader con shuffle=True.

---

#### Adam — la alternativa mas popular

```python
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=0.001)
```

Adam adapta el learning rate de forma individual para cada parametro, y usa momentum para acelerar el entrenamiento. En la practica converge mas rapido que SGD sin necesidad de ajustar tanto los hiperparametros:

```text
SGD:   clasico, predecible, pero requiere ajustar lr y momentum con cuidado
Adam:  adapta lr automaticamente, funciona bien con el lr default (0.001)
       → mas comun en el laboratorio y en la practica moderna
```

---

#### El rol del optimizador en el training loop

```python
for x, target in train_dl:
    # 1. Forward pass
    output = model(x)
    loss = funcion_perdida(output, target)

    # 2. Limpiar gradientes del paso anterior
    optimizer.zero_grad()

    # 3. Calcular gradientes (backpropagation)
    loss.backward()

    # 4. Actualizar pesos con los gradientes
    optimizer.step()
```

```text
optimizer.zero_grad():  borra los gradientes acumulados del paso anterior
                        (PyTorch acumula gradientes por defecto — hay que limpiar)
loss.backward():        calcula d_loss/d_param para cada parametro del modelo
optimizer.step():       aplica la regla de actualizacion: param -= lr × grad
```

Sin `zero_grad()`, los gradientes se acumularian de paso en paso — el modelo aprenderia sobre gradientes "sucios" de pasos anteriores.

---

## Celda 28: `loss.backward()` — backpropagation en una linea

### Codigo

```python
loss.backward()   # ¡Backpropagation! Eso es todo.
```

### Analisis

PyTorch registra automaticamente cada operacion matematica sobre tensores con `requires_grad=True`, construyendo un **grafo computacional**. Cuando se llama a `loss.backward()`, PyTorch recorre ese grafo hacia atras y calcula el gradiente del loss respecto a cada parametro:

```text
Pesos → conv1 → ... → fc8 → output → CrossEntropyLoss → loss
                                                          ↑
                                           backward() empieza aqui
                                           y fluye hacia la izquierda
```

Aplica la regla de la cadena automaticamente para cada capa. Los gradientes se guardan en `param.grad` de cada parametro con `requires_grad=True`. Sin esto, el optimizador no tendria nada que usar en `step()`.

---

## Celda 29: El training loop completo — los 5 pasos

### Codigo

```python
optimizer.zero_grad()                   # 1. Hacer cero los gradientes
output = model(input)                   # 2. Forward pass
loss = funcion_perdida(output, target)  # 3. Calcular el loss
loss.backward()                         # 4. Backpropagation — calcular gradientes
optimizer.step()                        # 5. Actualizar parametros del modelo
```

### Analisis

---

#### Los 5 pasos en detalle

```text
1. optimizer.zero_grad()
   Borra los .grad de todos los parametros del modelo.
   PyTorch ACUMULA gradientes por defecto — hay que limpiar antes de cada paso.

2. output = model(input)
   Forward pass: datos fluyen por la red, construye el grafo computacional.
   output: tensor (N, 102) de logits para MiAlexNet.

3. loss = funcion_perdida(output, target)
   Calcula el escalar de loss para el batch actual.
   CrossEntropyLoss compara los 102 logits con la clase correcta.

4. loss.backward()
   Backpropagation: recorre el grafo de derecha a izquierda.
   Calcula d_loss/d_param y guarda en param.grad para cada parametro.

5. optimizer.step()
   Usa param.grad para actualizar cada parametro.
   SGD:  param = param - lr * param.grad
   Adam: actualizacion adaptativa por parametro.
```

---

#### Este loop en el contexto completo de entrenamiento

```python
model = MiAlexNet().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

for epoch in range(50):
    model.train()                          # activa dropout y batchnorm en modo train
    for input, target in train_dl:
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()              # 1
        output = model(input)             # 2
        loss = criterion(output, target)  # 3
        loss.backward()                   # 4
        optimizer.step()                  # 5

    model.eval()
    with torch.no_grad():                 # sin calcular gradientes en evaluacion
        ...
```

Estos 5 pasos son el nucleo invariable de todo entrenamiento en PyTorch. Todo lo demas (DataLoader, epochs, logging, metricas) es infraestructura alrededor de este patron.

---

## Celda 30: Evaluacion de rendimiento — Accuracy

### Texto del notebook

> En clasificacion, la metrica es comparar el ground truth con lo predicho. La clase predicha es el **indice de mayor valor** (argmax) en la salida de la red.

### Codigo

```python
output = model(input)              # tensor (batch_size, n_clases) — logits
preds = output.argmax(dim=1)       # indice de mayor valor entre las N clases
n_correctas = (preds == targets).sum()  # tensor de 1s y 0s → suma = correctas
total = targets.shape[0]           # total de ejemplos en el batch
acc = n_correctas / total          # accuracy = correctas / total
```

### Analisis

---

#### `output.argmax(dim=1)` — de logits a clase predicha

La red devuelve un tensor `(batch, 102)` de logits — puntuaciones crudas para cada clase. Para obtener la prediccion se toma el **indice del mayor valor** en la dimension de clases:

```text
output para 1 imagen:
  [-1.2,  0.3,  8.5, -0.4,  2.1,  ..., 0.7]   ← 102 valores
     0     1     2     3     4          101     ← indices de clase

argmax(dim=1) → 2   ← clase con mayor logit = la prediccion

No se necesita Softmax para la prediccion: el argmax no cambia
aunque se aplique softmax (solo escala los valores, no el orden).
```

`dim=1` porque la dimension 1 es la de las clases (la que se quiere colapsar):

```text
output shape: (batch, clases)
                  0       1   ← dim
argmax(dim=1) → shape (batch,)  ← colapsa la dimension de clases
```

---

#### `(preds == targets).sum()` — contar aciertos

```text
preds:    tensor([2,  78, 15,  3, ...])   ← clase predicha por la red
targets:  tensor([2,  10, 15,  3, ...])   ← clase real (ground truth)

preds == targets  →  tensor([True, False, True, True, ...])
                              ↑      ↑      ↑     ↑
                            acierto error  acierto acierto

.sum()  → cuenta los True → numero de predicciones correctas
          (True = 1, False = 0)
```

---

#### `targets.shape[0]` — el total

```text
targets shape: (batch_size,)
targets.shape[0] = batch_size   ← cuantas muestras hay en el batch

acc = n_correctas / total       ← fraccion de aciertos, entre 0.0 y 1.0
```

Un accuracy de `0.7` significa que la red acerto el 70% de las predicciones del batch.

---

#### Por que no basta con el loss para evaluar

```text
Loss (CrossEntropyLoss):
  - Numero continuo (ej: 2.34)
  - Mide que tan seguros estamos, no solo si acertamos
  - Disminuye durante el entrenamiento → util para saber si aprende
  - Dificil de interpretar para un humano

Accuracy:
  - Numero entre 0 y 1 (ej: 0.72 = 72%)
  - Mide directamente "cuantas acertamos"
  - Facil de interpretar
  - NO disminuye monotonamente → puede fluctuar aunque el loss baje

Se usan AMBAS: loss guia el entrenamiento, accuracy evalua el resultado.
```

---

#### Accuracy en el contexto de evaluacion completa

```python
model.eval()
with torch.no_grad():          # no calcular gradientes en evaluacion
    total_correctas = 0
    total = 0
    for input, targets in val_dl:
        input, targets = input.to(device), targets.to(device)
        output = model(input)
        preds = output.argmax(dim=1)
        total_correctas += (preds == targets).sum().item()
        total += targets.shape[0]

    accuracy = total_correctas / total
    print(f"Accuracy: {accuracy:.1%}")  # ejemplo: "Accuracy: 72.3%"
```

`torch.no_grad()` desactiva el calculo de gradientes durante la evaluacion — no se necesitan y consumen memoria y tiempo innecesariamente.

---

## Celda 31: Training loop completo — MiAlexNet en Flowers-102

### Codigo (estructura)

```python
model = MiAlexNet().cuda()
optimizer = Adam(model.parameters(), lr=0.001)
loss_function = CrossEntropyLoss()

model.train()
for epoch in range(1, n_epochs + 1):
    total_correctas = 0.0
    total_muestras = 0.0
    for x, target in train_dl:
        optimizer.zero_grad()
        x, target = x.cuda(), target.cuda()
        output = model(x)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        preds = output.argmax(dim=1)
        total_correctas += (preds == target).sum()
        total_muestras += target.shape[0]
        accuracy = total_correctas / total_muestras
        print("\rEpoca {}: Loss: {:.2f} Accuracy: {:.2f}%".format(...), end="")
    print("")
```

### Salida

```text
Epoca  1: Loss: 4.44  Accuracy:  2.67%
Epoca  2: Loss: 4.49  Accuracy:  3.09%
Epoca  3: Loss: 3.85  Accuracy:  6.38%
Epoca  4: Loss: 3.73  Accuracy:  7.97%
Epoca  5: Loss: 3.45  Accuracy: 10.39%
Epoca  6: Loss: 3.43  Accuracy: 12.96%
Epoca  7: Loss: 3.20  Accuracy: 14.88%
Epoca  8: Loss: 3.44  Accuracy: 19.06%
Epoca  9: Loss: 2.99  Accuracy: 22.16%
Epoca 10: Loss: 2.54  Accuracy: 26.13%
```

### Analisis

---

#### El punto de partida: 2.67% en epoch 1

Con 102 clases, una red que predice **al azar** acertaria el `1/102 ≈ 0.98%`. Que la epoch 1 de ya 2.67% indica que la red empezo a aprender algo desde el primer paso — aunque apenas.

```text
Baseline aleatorio:  ~1%
Epoch 1:              2.67%  → ya aprende algo
Epoch 10:            26.13%  → aprendio bastante, pero aun hay mucho margen
```

---

#### La tendencia general: el loss baja, la accuracy sube

```text
Epoch  1: Loss 4.44  → Epoch 10: Loss 2.54   ← loss bajo ~43%
Epoch  1: Acc  2.67% → Epoch 10: Acc  26.13% ← accuracy subio ~10x
```

La red esta aprendiendo. Pero el ritmo es lento porque:
1. Solo 5,687 imagenes de entrenamiento para 102 clases (~55 por clase)
2. MiAlexNet se entrena **desde cero** (pesos aleatorios al inicio)
3. Solo 10 epochs — la red necesita mas tiempo

---

#### La anomalia en epoch 8: el loss sube de 3.20 a 3.44

```text
Epoch 7: Loss 3.20
Epoch 8: Loss 3.44  ← sube!
Epoch 9: Loss 2.99  ← baja de nuevo
```

Esto es normal — el loss reportado es el del **ultimo batch** de la epoch, no el promedio. El DataLoader con `shuffle=True` hace que el orden cambie cada epoch, y el ultimo batch puede ser mas facil o mas dificil que el de la epoch anterior. Si se reportara el loss promedio de todos los batches, seria mas suave.

La **accuracy acumulada** (que si promedia todo el epoch) sube monotonamente: 14.88% → 19.06% → 22.16%.

---

#### `total_correctas` se acumula durante toda la epoch

```python
for x, target in train_dl:      # 45 batches (5687/128)
    ...
    total_correctas += (preds == target).sum()
    total_muestras  += target.shape[0]
    accuracy = total_correctas / total_muestras
```

La accuracy se calcula sobre **todas las imagenes vistas hasta ese momento** en la epoch, no solo el batch actual. Esto da una estimacion mas estable que la accuracy por batch.

```text
Despues del batch 1:   accuracy = correctas_batch1 / 128
Despues del batch 2:   accuracy = (correctas_b1 + correctas_b2) / 256
...
Despues del batch 45:  accuracy = total_correctas / 5687  ← la que imprime al final
```

---

#### El codigo comentado: Transfer Learning (Actividad 4)

```python
# model = alexnet(pretrained=True)
# model.features.requires_grad_(False)
# model.classifier = nn.Sequential(
#     nn.Dropout(),
#     nn.Linear(256 * 6 * 6, 4096),
#     nn.ReLU(inplace=True),
#     nn.Dropout(),
#     nn.Linear(4096, 4096),
#     nn.ReLU(inplace=True),
#     nn.Linear(4096, 102),
# )
```

Este codigo muestra **Transfer Learning**: en vez de entrenar desde cero, se carga AlexNet con pesos preentrenados en ImageNet (1.2M imagenes, 1000 clases), se **congela el extractor de features** (`requires_grad_(False)` — esos pesos no cambian) y solo se reemplaza y entrena el clasificador final para las 102 clases de flores.

```text
MiAlexNet desde cero:     epoch 10 → 26%   (limitado por pocas imagenes)
AlexNet preentrenado:      epoch 10 → ~80%+ (usa conocimiento de ImageNet)

Transfer Learning es tan efectivo porque:
  Los 5 bloques conv de AlexNet ya saben detectar bordes, texturas, formas
  (aprendido de 1.2M imagenes).
  Solo hay que aprender COMO COMBINAR esas features para distinguir flores.
  → Mucho mas eficiente con datos limitados.
```

Esta es la **Actividad 4** del laboratorio — la diferencia entre ambos enfoques es dramatica y es uno de los conceptos mas importantes del deep learning moderno.

---

#### `model.train()` — modo entrenamiento

```python
model.train()
for epoch in range(...):
    ...
```

`model.train()` activa los modulos que se comportan diferente en entrenamiento vs inferencia:
- **Dropout**: apaga neuronas aleatoriamente (activo en train, inactivo en eval)
- **BatchNorm**: usa estadisticas del batch actual (en train) vs running stats (en eval)

Llamarlo una sola vez antes del loop es suficiente si no se alterna entre train y eval durante el entrenamiento. En este loop no hay evaluacion entre epochs, por eso no se necesita `model.eval()` dentro del loop.

---

## Celda 32: Evaluacion en el conjunto de Test

### Codigo

```python
ds_test = Flowers("flowers_dataset/test", transform=transforms)
test_dl = DataLoader(ds_test, batch_size=1024)

model.eval()
for x, target in test_dl:
    with torch.no_grad():
        x, target = x.cuda(), target.cuda()
        output = model(x)
        loss = loss_function(output, target)
        preds = output.argmax(dim=1)
        total_correctas += (preds == target).sum()
        total_muestras += target.shape[0]
        accuracy = total_correctas / total_muestras
        print("\rLoss: {:.2f} ... Accuracy: {:.2f}%".format(...), end="")
```

### Salida

```text
Loss: 2.74  Correctas: 285.0  Total: 1024.0  Accuracy: 27.83%
```

### Analisis

---

#### Diferencias clave con el loop de entrenamiento

```text
                    Entrenamiento          Evaluacion (test)
────────────────    ──────────────────     ──────────────────────────────
model.train()       SI (antes del loop)    NO — model.eval()
optimizer.zero_grad() SI                   NO (no hay gradientes)
loss.backward()     SI                     NO
optimizer.step()    SI                     NO
torch.no_grad()     NO                     SI (dentro del loop)
shuffle=True        SI (DataLoader)        NO (no importa el orden)
batch_size          128                    1024 (puede ser grande, sin gradientes)
```

`model.eval()` desactiva Dropout y cambia BatchNorm a usar running stats. `torch.no_grad()` desactiva el grafo computacional — sin gradientes, el forward es mas rapido y usa menos memoria, por eso `batch_size=1024` es viable.

---

#### Los resultados: train vs test

```text
Entrenamiento (epoch 10):  Loss 2.54  Accuracy 26.13%
Test:                      Loss 2.74  Accuracy 27.83%
```

Dos observaciones:

**1. La accuracy de test (27.83%) es ligeramente mayor que la de train (26.13%)**

Esto parece contradictorio — normalmente train > test. La explicacion esta en como se midio el training accuracy: se acumulo durante toda la epoch (incluidos los primeros batches cuando la red aun no habia actualizado sus pesos con los ultimos datos). El test se evalua con el modelo **ya completamente entrenado** con los 10 epochs. Compararlos directamente es injusto — para una comparacion justa habria que evaluar el training set despues de la ultima epoch con `model.eval()`.

**2. Los valores son muy cercanos → no hay overfitting severo**

Con solo 5,687 imagenes de entrenamiento y 58.7M parametros, podria esperarse mucho overfitting. Que train y test sean similares sugiere que la red no memorizo los datos — simplemente no entreno suficiente para hacerlo en 10 epochs.

---

#### `torch.no_grad()` dentro del loop vs antes

```python
# Como esta en el notebook (no_grad dentro):
model.eval()
for x, target in test_dl:
    with torch.no_grad():
        output = model(x)
        ...

# Forma mas comun (no_grad envuelve todo):
model.eval()
with torch.no_grad():
    for x, target in test_dl:
        output = model(x)
        ...
```

Ambas son correctas. La segunda es mas limpia; la primera es valida pero crea/destruye el contexto `no_grad` en cada iteracion innecesariamente.

---

#### 26% de accuracy — ¿es bueno?

```text
Baseline aleatorio:            ~1%    (1/102 clases)
MiAlexNet desde cero, 10 ep:  ~27%   (este resultado)
AlexNet preentrenado, 10 ep:  ~80%+  (Transfer Learning, Actividad 4)

Para clasificar 102 tipos de flores con solo ~55 imagenes por clase
entrenando desde cero, un 27% es razonable. El modelo aprendio algo real.
La diferencia enorme con Transfer Learning muestra por que en la practica
casi siempre se usa un modelo preentrenado como punto de partida.
```

---

## Celda 33: Guardar y cargar el modelo

### Texto del notebook

> La forma mas eficiente es guardar solo los pesos (`state_dict`). Para recuperarlos, se crea un modelo nuevo y se carga el `state_dict` con `load_state_dict`.

### Codigo

```python
torch.save(model.state_dict(), "pesos_modelo_entrenado.pth")  # guardar a disco

modelo = MiAlexNet()                                  # modelo con pesos aleatorios
pesos = torch.load("pesos_modelo_entrenado.pth")      # cargar pesos del disco
modelo.load_state_dict(pesos)                         # aplicar pesos al modelo
```

### Salida

```text
<All keys matched successfully>
```

### Analisis

---

#### `model.state_dict()` — que contiene

`state_dict` es un diccionario ordenado que mapea cada capa a sus tensores de pesos:

```text
{
  'conv1.0.weight': tensor([96, 3, 11, 11]),
  'conv1.0.bias':   tensor([96]),
  'conv2.0.weight': tensor([256, 96, 5, 5]),
  ...
  'fc8.0.weight':   tensor([102, 4096]),
  'fc8.0.bias':     tensor([102]),
}

Solo pesos y bias — NO incluye la arquitectura (la clase MiAlexNet).
```

---

#### Por que guardar `state_dict` y no el modelo completo

```python
# Opcion 1 — solo pesos (RECOMENDADO):
torch.save(model.state_dict(), "pesos.pth")

# Opcion 2 — modelo completo (NO recomendado):
torch.save(model, "modelo_completo.pth")
```

```text
Solo pesos:
  + Mas pequeño (solo los numeros)
  + Portable entre versiones de Python/PyTorch
  + Estandar de la industria
  - Requiere tener la clase MiAlexNet disponible al cargar

Modelo completo:
  + Mas comodo (no necesitas definir la clase al cargar)
  - Usa pickle: rompe si cambia la clase, el path, o la version de PyTorch
  - Mas grande
```

---

#### El flujo completo de carga

```python
# Paso 1: crear un modelo vacio (pesos aleatorios)
modelo_nuevo = MiAlexNet()

# Paso 2: cargar los pesos del disco a un diccionario
pesos = torch.load("pesos.pth")

# Paso 3: copiar los pesos al modelo (verifica que las keys coincidan)
modelo_nuevo.load_state_dict(pesos)
# → <All keys matched successfully>

# Paso 4: poner en modo evaluacion para inferencia
modelo_nuevo.eval()
```

---

#### `<All keys matched successfully>`

Esta confirmacion significa que cada clave del diccionario de pesos guardado (`conv1.0.weight`, `fc8.0.bias`, etc.) tiene un tensor correspondiente en el modelo nuevo con el **mismo nombre y la misma forma**. Si hubiera discrepancia (por ejemplo, si se modifica la arquitectura entre guardar y cargar), daria error.

---

#### Usos tipicos del guardado de pesos

```text
Checkpoint durante entrenamiento:
  → Guardar cada N epochs para no perder progreso si hay falla

Mejor modelo:
  → Guardar cuando la val_accuracy alcanza un nuevo maximo

Distribucion:
  → Compartir pesos entrenados para que otros los usen sin reentrenar

Transfer Learning:
  → Los pesos preentrenados de AlexNet en ImageNet se cargan exactamente asi
    pesos = torch.load("alexnet_imagenet.pth")
    model.load_state_dict(pesos)
```

La extension `.pth` es convencion para "PyTorch" pero es simplemente un archivo binario — tambien se ve `.pt` con el mismo proposito.

---

## Actividades del Laboratorio

---

### Actividad 1: Entrenar MiAlexNet por 10 epochs

**Enunciado**: Entrene el modelo MiAlexNet por 10 epocas. ¿Que resultados obtiene en train y test?

**Resultados**:

```text
Train (epoch 10): Accuracy 26.13%   Loss 2.54
Test:             Accuracy 27.83%   Loss 2.74
```

- Partiendo de ~1% aleatorio, llegar a 26-27% en 10 epochs desde cero es progreso real.
- Train ≈ Test → no hay overfitting severo, la red simplemente no entreno suficiente.

---

### Actividad 2: MiAlexNet con Dropout en FC6 y FC7

**Enunciado**: Modifique MiAlexNet para agregar Dropout antes de FC6 y FC7. Entrene 10 epochs. ¿Ve cambios en el rendimiento?

**Cambio**: descomentar `nn.Dropout()` (p=0.5 default) en ambos bloques FC.

**Resultados**:

```text
Epoca  1: Train Accuracy  3.04%
Epoca  2: Train Accuracy  2.76%
Epoca  3: Train Accuracy  3.01%
Epoca  4: Train Accuracy  3.32%
Epoca  5: Train Accuracy  3.09%
Epoca  6: Train Accuracy  2.99%
Epoca  7: Train Accuracy  3.45%
Epoca  8: Train Accuracy  4.69%
Epoca  9: Train Accuracy  5.94%
Epoca 10: Train Accuracy  6.33%
```

**Comparacion**:

```text
                    Epoch 1    Epoch 10
Sin Dropout:         2.67%     26.13%
Con Dropout (p=0.5): 3.04%      6.33%
```

**Analisis**:

El dropout con p=0.5 destruye el aprendizaje en 10 epochs. ¿Por que?

Dropout apaga el **50% de las neuronas** de fc6 (4096 neuronas) y fc7 (4096 neuronas) en cada paso de entrenamiento. Esto hace que la red aprenda mucho mas lento — necesita ver los datos muchas mas veces antes de que los patrones "se fijen".

```text
Epochs 1-7: accuracy 2-3% ≈ azar (1/102 ≈ 1%)
            La red no logra aprender nada con tanto ruido del dropout.

Epochs 8-10: empieza a despegar (4.69% → 6.33%)
             Recien en esta zona el aprendizaje supera el efecto del dropout.
```

**¿Significa esto que el dropout es malo?** No — es que 10 epochs no es suficiente. El dropout es util para **prevenir overfitting**, pero en 10 epochs el modelo sin dropout tampoco overfittea (train ≈ test = 26-27%). El regularizador llega antes de que haya un problema que resolver.

```text
Para que el dropout ayude en este escenario:
  - Entrenar muchas mas epochs (50-100+) hasta que sin dropout el train >> test
  - O reducir p (ej: p=0.2) para un dropout mas suave
  - O usar Transfer Learning (AlexNet preentrenado) donde si hay riesgo de overfitting
```

El AlexNet original entrenaba con dropout p=0.5 pero durante **90 epochs** en ImageNet (1.2M imagenes). Con 10 epochs y 5,687 imagenes, p=0.5 es demasiado agresivo.

---

### Actividad 3: MiAlexNet con BatchNorm2d antes de Conv3, Conv4 y Conv5

**Enunciado**: Agregue BatchNorm2d antes de Conv3, Conv4 y Conv5. Entrene 10 epochs. ¿Ve algun cambio en el entrenamiento?

**Cambio**: agregar `nn.BatchNorm2d(256)` en conv3, `nn.BatchNorm2d(384)` en conv4 y conv5.

**Resultados**:

```text
Epoca  1: Train Accuracy  3.78%
Epoca  2: Train Accuracy  9.25%
Epoca  3: Train Accuracy 16.41%
Epoca  4: Train Accuracy 21.38%
Epoca  5: Train Accuracy 25.25%
Epoca  6: Train Accuracy 30.65%
Epoca  7: Train Accuracy 34.92%
Epoca  8: Train Accuracy 38.95%
Epoca  9: Train Accuracy 42.94%
Epoca 10: Train Accuracy 47.07%
```

**Comparacion de los tres experimentos**:

```text
                         Epoch 5    Epoch 10   Tendencia
Sin nada (Act.1):        10.39%     26.13%     subiendo
Con Dropout p=0.5:        3.09%      6.33%     muy lenta
Con BatchNorm:           25.25%     47.07%     acelerada ↑↑
```

**Analisis**:

El resultado es llamativo: en la epoch 5 con BatchNorm (25.25%) ya se iguala al resultado final del modelo base en epoch 10 (26.13%). Al final de las 10 epochs, BatchNorm casi **duplica** la accuracy (47% vs 26%).

```text
¿Por que BatchNorm acelera tanto?

Cada capa recibe activaciones normalizadas (media≈0, var≈1).
→ Los gradientes fluyen de forma mas estable hacia las capas iniciales.
→ La red puede dar pasos de aprendizaje mas grandes sin desestabilizarse.
→ Menos epochs necesarios para alcanzar el mismo nivel.
```

La tendencia al final (epoch 10 todavia subiendo fuerte) indica que con mas epochs el modelo con BatchNorm seguiria mejorando significativamente — probablemente alcanzaria 60-70% o mas con 30-50 epochs, mientras el modelo base se estancaria antes.

---

### Actividad 4: Transfer Learning — AlexNet preentrenado en ImageNet

**Enunciado**: Use el modelo preentrenado de AlexNet. Entrene 10 epochs. ¿Afecta en el rendimiento?

**Cambios clave**:
- `alexnet(pretrained=True)` — carga pesos entrenados en 1.2M imagenes de ImageNet
- `model.features.requires_grad_(False)` — congela los 5 bloques convolucionales
- Reemplaza `model.classifier` con una nueva cabeza de 102 clases

**Resultados**:

```text
Epoca  1: Train Accuracy 21.47%
Epoca  2: Train Accuracy 51.84%
Epoca  3: Train Accuracy 63.57%
Epoca  4: Train Accuracy 70.12%
Epoca  5: Train Accuracy 74.50%
Epoca  6: Train Accuracy 78.39%
Epoca  7: Train Accuracy 80.94%
Epoca  8: Train Accuracy 81.52%
Epoca  9: Train Accuracy 82.93%
Epoca 10: Train Accuracy 84.16%
```

**Comparacion final de las 4 actividades**:

```text
Experimento                    Epoch 1   Epoch 5   Epoch 10
─────────────────────────────  ───────   ───────   ────────
Act.1: Sin modificar            2.67%    10.39%     26.13%
Act.2: + Dropout (p=0.5)        3.04%     3.09%      6.33%
Act.3: + BatchNorm              3.78%    25.25%     47.07%
Act.4: Transfer Learning       21.47%    74.50%     84.16%
```

**Analisis**:

La epoch 1 del Transfer Learning (21.47%) ya supera el resultado final del modelo base en epoch 10 (26.13%). En epoch 2 (51.84%) ya casi dobla al modelo con BatchNorm en epoch 10.

```text
¿Por que es tan superior?

Los 5 bloques convolucionales de AlexNet fueron entrenados con
1.2 millones de imagenes de 1000 categorias distintas.
Ya saben detectar: bordes, texturas, formas, gradientes, patrones.

Transfer Learning aprovecha ese conocimiento:
  → Los features extraidos por conv1-conv5 ya son utiles para flores
  → Solo hay que entrenar el clasificador (fc6, fc7, fc8) desde cero
  → Mucho menos parametros que aprender con los mismos 5,687 datos

Sin Transfer Learning (desde cero):
  Epoch 1: la red no sabe nada → accuracy ≈ 1% (azar)
  Necesita aprender features basicos Y como clasificar flores

Con Transfer Learning:
  Epoch 1: ya detecta bordes y texturas → empieza en 21%
  Solo necesita aprender COMO COMBINAR esas features para flores
```

Esta es la razon por la que en la practica casi todo proyecto de vision computacional parte de un modelo preentrenado en ImageNet en vez de entrenar desde cero. Los datos disponibles raramente son suficientes para superar lo que da el Transfer Learning.
