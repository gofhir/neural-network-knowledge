---
title: "Actividades 1-3 (multiple choice)"
weight: 50
math: true
---

Cubre las celdas 43-52 del notebook. Tres actividades multiple choice donde tenés que **distinguir cuándo cada tokenizador o estrategia es apropiada** según el dominio. Cada una tiene **una trampa pedagógica deliberada** que apunta a errores comunes de NLP.

---

## Convención de las actividades

Cada actividad sigue el mismo patrón:

1. **Enunciado** que describe un problema con dominio específico.
2. **N opciones implementadas** como funciones Python (`split_text1`, `split_text2`, ..., o `remove_stopwords1`, etc.).
3. **Widget Colab** `#@param` donde elegís la opción.

**Las dos notas clave** en todos los enunciados:
- *"Asuma que todas las librerías necesarias se encuentran previamente cargadas y que todos los métodos se logran ejecutar sin errores"* → la pregunta NO es sobre corrección sintáctica.
- *"En caso de haber más de una correcta, debe seleccionar la más adecuada"* → puede haber varias técnicamente correctas pero solo una es la mejor.

---

## Actividad 1 (celdas 44-46) — Separar tweets en frases

**Dominio**: mensajes de X (Twitter) en inglés.
**Tarea**: separar texto en **frases** (oraciones).

### Las 6 opciones

```python
# OPCIÓN 1: text.split('.')               ← split ingenuo
# OPCIÓN 2: re.split('[.;\\n]', text)     ← regex con ; y newlines
# OPCIÓN 3: text.split()                  ← whitespace (PALABRAS, no oraciones)
# OPCIÓN 4: nltk.word_tokenize(text)      ← PALABRAS, no oraciones
# OPCIÓN 5: nltk.sent_tokenize(text)      ← ORACIONES con Punkt
# OPCIÓN 6: TweetTokenizer().tokenize     ← PALABRAS de Twitter
```

### La trampa pedagógica

Quien lee superficial elige **Opción 6** porque dice "Twitter". **Pero TweetTokenizer es a nivel palabra**, no oración. La pregunta pide **frases (oraciones)**.

### Respuesta correcta: **Opción 5**

`nltk.sent_tokenize(text)` es la única que **es sentence-level real**. El default es inglés (que es lo que pide el enunciado), maneja abreviaciones (`Mr.`, `U.S.A.`), separa por `.`, `!`, `?`.

| Opción | Nivel | Maneja abreviaciones | Adecuada |
|---|---|---|---|
| 1: `split('.')` | Oración (mal) | ✗ | ✗ |
| 2: `re.split('[.;\\n]', ...)` | Oración (mal) | ✗ | ✗ |
| 3: `split()` (whitespace) | **Palabra** | N/A | ✗ Tarea equivocada |
| 4: `word_tokenize` | **Palabra** | Sí | ✗ Tarea equivocada |
| **5: `sent_tokenize`** | **Oración** | **Sí** | **✓ CORRECTA** |
| 6: `TweetTokenizer` | **Palabra** | Sí | ✗ Tarea equivocada |

---

## Actividad 2 (celdas 47-49) — Separar correos formales en palabras

**Dominio**: correos corporativos en **español formal**.
**Tarea**: separar texto en **tokens (palabras)**, para luego eliminar stop-words.

### Las 6 opciones

```python
# OPCIÓN 1: text.split(' ')              ← split por espacio único
# OPCIÓN 2: text.split()                 ← whitespace genérico
# OPCIÓN 3: re.split('[.;, \\n]', text)  ← regex puntuación
# OPCIÓN 4: nltk.word_tokenize(text)     ← Treebank (PALABRAS)
# OPCIÓN 5: spaCy Spanish() + sentencizer + return sent.text ← ORACIONES (¡!)
# OPCIÓN 6: TweetTokenizer().tokenize    ← Twitter
```

### Las dos trampas pedagógicas

**Trampa 1 — Opción 5**: usa spaCy `Spanish()` (suena profesional) pero **el código itera `doc.sents` y devuelve `sent.text`** — eso son **oraciones, no tokens**. Tarea equivocada disfrazada de sofisticación.

**Trampa 2 — Opción 6**: TweetTokenizer suena bien por su capacidad de manejar texto natural, pero el dominio es **correo corporativo formal** (no Twitter). Es overkill y rompe abreviaciones como `Sr.`, `Dr.` que sí necesitamos preservar en correos.

### Respuesta correcta: **Opción 4**

`nltk.word_tokenize(text)`:
- **Es a nivel palabra** ✓
- **Separa puntuación apropiadamente** ✓
- **Compatible con filtrado de stop-words** posterior ✓
- **Robusto en texto formal** ✓ (Penn Treebank rules diseñadas para newswire)
- Preserva `Sr.`, `Dr.`, `Dpto.` como tokens completos.

Verificación directa:

```python
correo = "Estimado Sr. Pérez, adjunto reporte. Saludos cordiales, Dr. Ana López"

word_tokenize(correo)
# ['Estimado', 'Sr.', 'Pérez', ',', 'adjunto', 'reporte', '.', 'Saludos',
#  'cordiales', ',', 'Dr.', 'Ana', 'López']

TweetTokenizer().tokenize(correo)
# ['Estimado', 'Sr', '.', 'Pérez', ...]   ← rompe Sr.
```

---

## Actividad 3 (celdas 50-52) — Eliminación de Stop-Words en Wikipedia ES

**Dominio**: 1,000,000 artículos de Wikipedia en español como train + 200,000 como test.
**Tarea**: filtrar stop-words.

### Pista clave del enunciado

> *"un umbral de frecuencia definido por el usuario (excepto en los últimos 2 casos)"*

Esto te dice:
- **Opciones 1-8** usan threshold de frecuencia (enfoque data-driven).
- **Opciones 9-10** no usan threshold (lista fija).

### Las 4 dimensiones para evaluar

Cada opción se caracteriza por 4 decisiones:

| Dimensión | Valores |
|---|---|
| D1. Fuente del cálculo de frecuencias | `train+test` (leakage) / solo `train` / por separado |
| D2. Dirección del threshold | `> threshold` / `< threshold` (invertido) |
| D3. Operación de filtrado | `not in stop_words` / `in stop_words` (invertido) |
| D4. Lista | Calculada / NLTK español / NLTK **inglés** (error) |

### Análisis sistemático

| Opción | D1 | D2 | D3 | D4 | Veredicto |
|---|---|---|---|---|---|
| 1 | train+test (leakage) | `>` | `not in` | Frecuencia | ✗ Leakage |
| 2 | train+test (leakage) | `>` | `in` (invertido) | Frecuencia | ✗ Leakage + invertido |
| 3 | train+test (leakage) | `<` (invertido) | `not in` | Frecuencia | ✗ Leakage + invertido |
| 4 | solo train | `<` (invertido) | `not in` | Frecuencia | ✗ Threshold invertido |
| **5** | **solo train** | `>` | `not in` | Frecuencia | **✓ CORRECTA** |
| 6 | train + test separados | `<` (invertido) | `not in` | Frecuencia | ✗ |
| 7 | train + test separados | `>` | `not in` | Frecuencia | ✗ Leakage parcial |
| 8 | train + test separados | `>` | `in` (invertido) | Frecuencia | ✗ |
| 9 | — | — | `not in` | NLTK **inglés** | ✗ Idioma incorrecto |
| 10 | — | — | `in` (invertido) | NLTK **inglés** | ✗ |

### Respuesta correcta: **Opción 5**

```python
def remove_stopwords5(train, test, threshold):
    corpus = nltk.Text(train)                # ← solo train (sin leakage)
    frequencies = FreqDist(corpus)
    stop_words = set()
    for word in frequencies.keys():
        if frequencies[word] > threshold:    # ← frecuentes = stop-words
            stop_words.add(word)
    filtered_train = [s for s in train if s not in stop_words]
    filtered_test = [s for s in test if s not in stop_words]  # misma lista
    return filtered_train, filtered_test
```

Por qué es correcta:
1. **D1 — Sin leakage**: usa solo `train` para calcular. Test queda intacto.
2. **D2 — Threshold correcto**: `> threshold` marca palabras frecuentes como stop-words (definición correcta).
3. **D3 — Filtrado correcto**: `not in` elimina las stop-words.
4. **D4 — Estrategia adaptada al dominio**: Wikipedia tiene vocabulario tan amplio que la lista NLTK español genérica (313 palabras) no captura stop-words enciclopédicas (`artículo`, `página`, `referencia`, `categoría`).

### Las 3 lecciones de la actividad

1. **Train/test split es sagrado** — nunca uses test para calcular nada.
2. **Stop-words = alta frecuencia + baja información** — la definición exige `>` no `<`.
3. **Filtrar = eliminar** — `not in stop_words`, no `in stop_words`.

Si entendés las 4 dimensiones, leer las 10 opciones se vuelve mecánico: descartás por cada dimensión hasta quedarte con la única que pasa todas.

---

## Patrones generales de las 3 actividades

Las trampas pedagógicas comunes:

| Trampa | Cómo identificarla |
|---|---|
| **Tarea equivocada disfrazada** | El código usa herramienta correcta pero devuelve nivel equivocado (palabras vs oraciones) |
| **Sobre-aplicación al dominio** | El tokenizer "suena correcto" por el dominio (Twitter, formal) pero resuelve otra cosa |
| **Heurística regex ingenua** | `split('.')` parece simple pero falla en abreviaciones |
| **Data leakage** | Calcular features en train+test contamina la evaluación |
| **Direcciones lógicas invertidas** | `< threshold` para stop-words, `in stop_words` para filtrar |
| **Idioma incorrecto** | Lista inglés para texto español |

### Para tu pipeline real (MDM-FHIR)

Estas actividades modelan **errores comunes en producción**:

- **Confundir niveles**: muchos pipelines clínicos hacen tokenización de palabras cuando necesitan oraciones (e.g., para applicar VADER a "oraciones" que en realidad son palabras sueltas).
- **Data leakage**: el error más insidioso. Especialmente fácil de cometer cuando hay múltiples archivos de configuración compartidos.
- **Threshold invertido**: en sistemas de detección de outliers / anomalías clínicas, confundir `>` con `<` invierte el comportamiento sin generar error visible.

Las actividades te entrenan a **leer código rigurosamente** antes de elegir, no juzgar por el nombre.

---

## Lecturas

- [Tokenización clásica](/fundamentos/tokenizacion-clasica) — sentence vs word tokenization.
- [Bag of Words](/fundamentos/bag-of-words) — por qué stop-words son data-driven.
- [Punkt 2006](/papers/punkt-kiss-strunk-2006) — el sentence tokenizer.

Anterior: [spaCy: POS, NER, Dependency Parsing](spacy-pipeline).
Siguiente: [NLLB-200 traducción multilingüe](nllb-traduccion).
