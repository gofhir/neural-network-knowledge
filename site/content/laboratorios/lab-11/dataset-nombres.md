---
title: "Dataset y Tensores"
weight: 10
math: true
---

## El problema

Dado un apellido, predecir el idioma/nacionalidad de origen. Es un problema de **clasificación de secuencias de longitud variable**: la entrada es un string de 2 a ~20 caracteres, la salida es una de 18 categorías de idioma.

Es el caso de uso natural para una RNN: la red lee el apellido **carácter por carácter**, mantiene un estado oculto que resume lo visto, y al llegar al último carácter emite la predicción.

---

## El dataset

Se descarga desde el tutorial oficial de PyTorch:

```python
!wget https://download.pytorch.org/tutorial/data.zip
!unzip data.zip
```

Dentro de `data/names/` hay **18 archivos** `[Idioma].txt`, uno por nacionalidad (Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Irish, Italian, Japanese, Korean, Polish, Portuguese, Russian, Scottish, Spanish, Vietnamese). Cada línea es un apellido.

---

## Normalización Unicode → ASCII

Los apellidos vienen con tildes y caracteres especiales (`Ślusàrski`, `Müller`, `Beñat`). Para reducir el alfabeto a un set fijo y manejable, se convierten a ASCII:

```python
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)  # 57

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
```

`unicodedata.normalize('NFD', s)` descompone caracteres con tilde en (letra base + marca de acento). Luego se filtran las marcas (`Mn` = "Mark, Nonspacing") y se conservan solo las letras del alfabeto ASCII. `Ślusàrski → Slusarski`.

El alfabeto final tiene 57 símbolos: 52 letras (a–z, A–Z) + 5 caracteres `" .,;'"`.

---

## Construcción del diccionario `category_lines`

```python
category_lines = {}        # {'French': ['Abel', ...], 'German': [...], ...}
all_categories = []        # ['French', 'German', ...]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    category_lines[category] = readLines(filename)

n_categories = len(all_categories)  # 18
```

`category_lines` es la fuente del dataset; `all_categories` mantiene el **orden fijo** de las clases (necesario para mapear nombre → índice y viceversa).

---

## One-hot encoding por carácter

> **¿Qué es un vector one-hot?** Una forma de representar categorías sin introducir orden numérico falso entre ellas: una "casilla" por categoría, todas en cero excepto una. Si nunca lo has visto, lee primero la sección **[One-Hot Encoding en el fundamento de Representación de Datos](/fundamentos/representacion-datos/#25-one-hot-encoding-representar-categorias)** — explica el "por qué" desde cero.

Una RNN no puede consumir strings directamente — necesita tensores. Cada carácter se representa como un vector one-hot de tamaño `n_letters = 57`:

```python
def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)        # shape: (1, 57)
    tensor[0][letterToIndex(letter)] = 1
    return tensor
```

Para un apellido completo, se apilan los one-hots en un tensor 3D:

```python
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)   # (seq_len, batch=1, 57)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
```

| Variable | Shape | Significado |
|----------|-------|-------------|
| `letterToTensor('A')` | `(1, 57)` | Un carácter one-hot |
| `lineToTensor('Albert')` | `(6, 1, 57)` | 6 caracteres × batch 1 × 57 letras |

La dimensión `batch=1` está fija porque este lab **no hace batching** — cada apellido se procesa solo. Padding y batching reales requieren `pack_padded_sequence` u otras técnicas que se omiten aquí por simplicidad.

---

## Sampleo de ejemplos de entrenamiento

En lugar de un `DataLoader` formal, se hace un sampleo aleatorio en cada iteración:

```python
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)               # idioma random
    line = randomChoice(category_lines[category])         # apellido random del idioma
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
```

Detalle clave: el sampleo es **uniforme por categoría**, no por número de apellidos. Como hay idiomas con miles de apellidos (Russian) y otros con decenas (Korean), un sampleo proporcional desbalancearía el entrenamiento. Aquí cada idioma tiene la misma probabilidad de aparecer en cada step, lo que actúa como un **balanceo natural de clases**.

`category_tensor` es un escalar long con el índice de la clase (0–17), formato esperado por `nn.CrossEntropyLoss`.

---

## Tamaño del dataset y comparación con el tutorial padre

Este lab es una adaptación del [tutorial oficial de PyTorch *NLP From Scratch: Classifying Names with a Character-Level RNN*](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html). Los números totales del dataset son:

| Métrica | Valor |
|---------|-------|
| Apellidos totales | 20 074 |
| Train / Test (tutorial original, 85/15) | 17 063 / 3 011 |
| Idiomas (clases) | 18 |
| Promedio por idioma | ~1 115 |
| Distribución | **Muy desbalanceada** (Russian ~9 400, Korean ~94) |

Por eso el sampleo del lab es **uniforme por categoría** y no proporcional al tamaño: si fuera proporcional, una época estaría dominada por apellidos rusos y el modelo nunca aprendería bien las clases minoritarias.

### Diferencias con el tutorial original

| Decisión | Tutorial PyTorch | Lab UC |
|----------|------------------|--------|
| Alfabeto | 58 chars (incluye `_` como OOV) | 57 chars (sin OOV — descarta caracteres no mapeables) |
| Batch | 64 con gradient accumulation | 1 (manual, sin acumulación) |
| Loss | `NLLLoss` + `LogSoftmax` | `CrossEntropyLoss` (equivalente: ya integra `LogSoftmax`) |
| Train/test split | 85/15 | No hay split — la matriz de confusión se calcula sobre 10 000 muestras aleatorias del set completo |

El lab simplifica para fines didácticos: sin OOV, sin batching real, sin split formal de evaluación.
