---
title: "Tokenización con NLTK"
weight: 10
math: true
---

Cubre las celdas 21-25 del notebook. NLTK provee **tres tokenizadores** que vimos en el lab: `sent_tokenize` (Punkt, multilingüe), `word_tokenize` (Penn Treebank), y `TweetTokenizer` (texto de redes sociales). Cada uno representa una filosofía y un trade-off distinto.

Para la teoría completa de tokenización ver [Tokenización clásica](/fundamentos/tokenizacion-clasica). Para el algoritmo Punkt ver [paper Kiss & Strunk 2006](/papers/punkt-kiss-strunk-2006). Para TweetTokenizer ver [paper Gimpel et al. 2011](/papers/twitter-pos-gimpel-2011).

---

## 1. Setup: descarga de modelos

Antes de tokenizar, el lab descarga los datos NLTK necesarios (celda 8):

```python
import nltk
nltk.download('punkt')        # modelos sentence tokenizer (legacy)
nltk.download('punkt_tab')    # versión .tab (requerida desde NLTK 3.8.2+)
```

Sin `punkt_tab`, `sent_tokenize` falla con `LookupError` en versiones recientes. Es un gotcha frecuente.

---

## 2. Sentence tokenization con Punkt

### En inglés (celda 22)

```python
from nltk.tokenize import sent_tokenize
s = "this is a sentence. This is another sentence. Mr. President said something. U.S.A. means United States of America"
sent_tokenize(s)
```

Output:

```python
['this is a sentence.',
 'This is another sentence.',
 'Mr. President said something.',
 'U.S.A. means United States of America']
```

**Cuatro oraciones correctamente segmentadas**, a pesar de tener 6 puntos:
- 4 puntos finales de oración → boundaries detectados.
- 2 puntos dentro de `U.S.A.` y 1 dentro de `Mr.` → reconocidos como abreviaciones.

Esto es la **etapa 1 type-based** de Punkt en acción (ver [Punkt paper](/papers/punkt-kiss-strunk-2006)): `Mr` y `U.S.A` están en el `abbrev_types` del modelo inglés porque aparecen masivamente con punto en el corpus de entrenamiento.

### En español (celda 23) — el bug latente

```python
s = "Esta es una oración. Esta es otra oración. El Sr. presidente dijo algo. U.S.A. significa United States of America"
sent_tokenize(s)
```

**Sin pasar `language='spanish'`**, NLTK usa el modelo inglés por default. Resultado: probablemente **5 fragmentos rotos** porque "Sr." no está en `abbrev_types` del modelo inglés.

**Versión correcta**:

```python
sent_tokenize(s, language='spanish')
# 4 oraciones correctamente segmentadas
```

**Lección operativa**: SIEMPRE pasar `language` explícitamente para texto no inglés. Modelos disponibles tras `nltk.download('punkt')`: 18 idiomas (english, spanish, french, german, italian, portuguese, dutch, danish, swedish, norwegian, finnish, estonian, polish, czech, slovenian, turkish, greek, russian).

---

## 3. Word tokenization Penn Treebank

### El ejemplo "limpio" (celda 24)

```python
sentence = "Such an in-depth-analysis can reveal features that are not easily visible..."
tokens = nltk.word_tokenize(sentence)
```

Output:

```python
['Such', 'an', 'in-depth-analysis', 'can', 'reveal', 'features', 'that',
 'are', 'not', 'easily', 'visible', 'from', 'the', 'variations', 'in',
 'the', 'individual', 'genes']
```

Notar que **`in-depth-analysis` se preserva como un solo token**. Penn Treebank tiene reglas para preservar guiones intra-palabra.

### Las reglas Penn más relevantes

| Caso | Penn output | Por qué |
|---|---|---|
| `"end."` | `["end", "."]` | Puntuación separada |
| `"don't"` | `["do", "n't"]` | Contracciones separadas |
| `"John's"` | `["John", "'s"]` | Posesivos separados |
| `"U.S.A."` | `["U.S.A."]` | Abreviación preservada |
| `"in-depth"` | `["in-depth"]` | Guion intra-palabra preservado |
| `"$1.50"` | `["$", "1.50"]` | Símbolo separado, decimal preservado |

### Lo que `word_tokenize` rompe

Casos donde falla:

| Input | Output (problema) |
|---|---|
| `"http://example.com"` | `["http", ":", "/", "/example.com"]` ← URL destruida |
| `"user@gmail.com"` | `["user", "@", "gmail.com"]` ← email roto |
| `":-)"` | `[":", "-", ")"]` ← emoticon destruido |
| `"#machinelearning"` | `["#", "machinelearning"]` ← hashtag roto |

Para texto que contenga URLs, emails, emoticons o hashtags, **Penn falla**. Solución: TweetTokenizer.

---

## 4. TweetTokenizer (celda 25)

```python
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
s = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
tknzr.tokenize(s)
```

Output:

```python
['This', 'is', 'a', 'cooool',
 '#dummysmiley', ':',
 ':-)', ':-P', '<3',           # ← emoticons preservados
 'and', 'some', 'arrows',
 '<', '>', '->', '<--']
```

**Las diferencias clave** con Penn:
- `#dummysmiley` preservado.
- `:-)`, `:-P`, `<3` preservados como tokens unitarios.
- `->`, `<--` preservados como flechas multicaracter.

### Cómo lo hace

TweetTokenizer aplica **regex específicas** antes de tokenización general, capturando patrones de redes sociales:

```
1. URLs (http://, https://, www.)
2. Phone numbers
3. HTML tags
4. ASCII emoticons (:-), :-(, :-P, <3, etc.)
5. Hashtags (#palabra)
6. At-mentions (@palabra)
7. Numbers, decimales, fracciones
8. ... fallback Penn rules
```

Si un fragmento matchea estas primeras categorías, se preserva. Solo fragmentos no-matched van al tokenizer Penn-style.

### Parámetros útiles

```python
TweetTokenizer(
    preserve_case=True,    # default
    reduce_len=True,       # 'cooool' → 'coool' (máx 3 chars repetidos)
    strip_handles=True,    # '@user' → ''
)
```

Útiles para normalizar antes de pasar a un clasificador downstream.

---

## 5. Comparativa lado a lado

Mismo texto, 4 tokenizadores distintos:

```python
texto = "Don't go! Visit http://example.com :-) #yolo"

# WhitespaceTokenizer
['Don\'t', 'go!', 'Visit', 'http://example.com', ':-)', '#yolo']

# word_tokenize (Penn)
['Do', "n't", 'go', '!', 'Visit', 'http', ':', '//example.com', ':',
 '-', ')', '#', 'yolo']

# TweetTokenizer
["Don't", 'go', '!', 'Visit', 'http://example.com', ':-)', '#yolo']

# spaCy English()
["Do", "n't", 'go', '!', 'Visit', 'http', ':', '/', '/example.com',
 ':-)', '#yolo']
```

**Lección global**: no existe "la tokenización correcta". Existe la apropiada para tu corpus.

| Corpus | Tokenizador recomendado |
|---|---|
| Texto editorial / noticias en inglés | NLTK Treebank |
| Texto formal en español | NLTK Punkt español + Treebank |
| Texto de Twitter / redes sociales | TweetTokenizer |
| Texto biomédico inglés | scispaCy custom |
| Texto clínico español | spaCy `Spanish()` + abreviaciones custom |

---

## 6. Para tu trabajo MDM-FHIR

En texto clínico vas a encontrar todos los patrones difíciles:

```python
text_clinico = "Pte. Sr. González, 65 a., HTA dx 2020. Tto. losartán 50 mg/d. ECG s/p."
```

- Abreviaciones honoríficas (`Sr.`).
- Abreviaciones clínicas (`pte.`, `dx`, `tto.`, `s/p`).
- Códigos médicos (`HTA`, `ECG`).
- Decimales con unidades (`50 mg/d`).

Ningún tokenizador genérico (Penn, TweetTokenizer, spaCy `Spanish()`) maneja todo bien. Necesitás:

1. **Punkt re-entrenado** sobre corpus clínico (e.g., MEDDOCAN) para aprender abreviaciones médicas.
2. **Reglas regex custom** para `pte.`, `dx`, `tto.`, `s/p`, etc.
3. **Preservación de unidades médicas** (`mg/d`, `mmHg`, `ng/mL`).

Esto es **exactamente lo que aborda la práctica local** del lab — scripts 20-23 en [clase_16/practica](https://github.com/) — entrenando Punkt custom sobre MEDDOCAN/Cantemist/PharmaCoNER.

---

## Lecturas

- [Tokenización clásica (fundamento)](/fundamentos/tokenizacion-clasica) — visión general.
- [Punkt — Kiss & Strunk 2006](/papers/punkt-kiss-strunk-2006) — el sentence tokenizer.
- [Twitter POS — Gimpel et al. 2011](/papers/twitter-pos-gimpel-2011) — TweetTokenizer.
- [NLTK — Bird & Loper 2006](/papers/nltk-bird-loper-2006) — el toolkit.

Siguiente: [Estadísticas de texto: Zipf, Heaps, FreqDist](nltk-estadisticas).
