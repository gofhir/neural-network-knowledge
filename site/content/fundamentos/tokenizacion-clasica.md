---
title: "Tokenización clásica"
weight: 275
math: true
---

**Tokenización** es el proceso de dividir texto en unidades discretas — tokens — que un modelo NLP puede procesar. Parece trivial pero es **una de las decisiones más subestimadas** del pipeline: el tokenizador fija el techo de calidad de todo lo que viene después (FreqDist, BoW, embeddings, classifier).

Este fundamento cubre la **tokenización clásica** (NLTK Punkt + Treebank, TweetTokenizer, spaCy `Spanish()`), anterior a subword tokenization. Para BPE moderna ver [BPE](/fundamentos/bpe).

---

## 1. Los dos niveles de tokenización

Toda tokenización opera en uno o dos niveles:

```
Documento crudo
   │
   ▼
[sentence tokenization]  → ["oración 1.", "oración 2.", ...]
   │                              │
   │                              ▼
   │                       [word tokenization]
   │                       → ["palabra", "1", ".", ...]
   ▼
Output según uso
```

**Sentence tokenization** segmenta texto continuo en oraciones individuales. **Word tokenization** segmenta cada oración en palabras / signos / unidades léxicas.

La separación importa: aplicar word tokenization sobre un texto sin segmentar oraciones lleva a perder boundary info necesaria para POS tagging, parsing, NER.

---

## 2. El problema central: el punto tiene tres roles

Un caracter, tres significados ambiguos:

| Rol | Ejemplo | Decisión esperada |
|---|---|---|
| Fin de oración | `"Volvió a casa. Cenó."` | **Separar** en 2 oraciones |
| Abreviación | `"Dr. Smith fue al U.S.A."` | **NO separar**: es interno a la oración |
| Decimal o ordinal | `"3.14"`, `"1990s"` | **NO separar**: parte del número |

Tokenizadores ingenuos como `text.split('.')` fallan catastróficamente:

```python
"Dr. Smith fue al U.S.A.".split('.')
# ['Dr', ' Smith fue al U', 'S', 'A', '']  ← 5 fragmentos rotos
```

Las técnicas serias resuelven esto con:
- **Listas de abreviaciones** hardcoded (spaCy: `Spanish()` incluye `Sr.`, `Dr.`, `Sra.`, `Lic.`).
- **Estadísticas colocacionales** (Punkt aprende abreviaciones del corpus).
- **Reglas regex** sofisticadas (TweetTokenizer).

Ver [paper Punkt](/papers/punkt-kiss-strunk-2006) para el algoritmo no supervisado de detección de abreviaciones que sostiene `nltk.sent_tokenize`.

---

## 3. Tokenizadores principales del ecosistema clásico

### 3.1 NLTK `sent_tokenize` (Punkt)

Default de NLTK para sentence segmentation, multilingüe (18 idiomas):

```python
from nltk import sent_tokenize
sent_tokenize("Esta es una oración. Otra acá.", language='spanish')
# ['Esta es una oración.', 'Otra acá.']
```

**Importante**: pasar `language='spanish'` explícitamente. Sin parámetro, NLTK usa modelo inglés que no conoce abreviaciones españolas como `Sr.`, `Sra.`.

Modelos descargables vía `nltk.download('punkt')` y desde NLTK 3.8.2+ también `nltk.download('punkt_tab')`.

### 3.2 NLTK `word_tokenize` (Treebank)

Tokenizador a nivel palabra siguiendo convenciones **Penn Treebank** (Marcus et al. 1993):

```python
from nltk import word_tokenize
word_tokenize("Don't go to the U.S.A.!")
# ["Do", "n't", "go", "to", "the", "U.S.A.", "!"]
```

Decisiones canónicas de Penn:
- Separar contracciones: `don't → ["do", "n't"]`.
- Separar posesivos: `John's → ["John", "'s"]`.
- Preservar abreviaciones comunes: `U.S.A.` queda intacto.
- Preservar números decimales: `3.14` queda intacto.
- Separar puntuación: `end.` → `["end", "."]`.

**Problema en redes sociales**: destruye emoticons (`:-) → [":", "-", ")"]`), hashtags (`#tag → ["#", "tag"]`), URLs.

### 3.3 NLTK `TweetTokenizer` (Gimpel et al. 2011)

Diseñado para texto Twitter, preserva features de redes sociales:

```python
from nltk.tokenize import TweetTokenizer
TweetTokenizer().tokenize("OMG @user this is cooool! :-) #yolo http://t.co/abc")
# ['OMG', '@user', 'this', 'is', 'cooool', '!', ':-)', '#yolo', 'http://t.co/abc']
```

Reglas clave:
- Preserva emoticons como `:-)`, `:(`, `<3`, `o_O`.
- Preserva hashtags y at-mentions completos.
- Preserva URLs.
- NO separa contracciones.
- Opcionales: `reduce_len=True` (cooool → coool), `strip_handles=True` (@user → vacío).

Ver [paper Gimpel et al. 2011](/papers/twitter-pos-gimpel-2011) para detalle de las reglas.

### 3.4 spaCy `Spanish()` + `sentencizer`

Tokenizador minimal de spaCy para español:

```python
from spacy.lang.es import Spanish
nlp = Spanish()
nlp.add_pipe('sentencizer')
doc = nlp("Sr. Pérez consulta. Tiene HTA.")
for sent in doc.sents:
    print(sent.text)
# Sr. Pérez consulta.
# Tiene HTA.
```

Filosofía opuesta a Punkt: spaCy usa **listas hardcoded** de abreviaciones por idioma. Más predecible pero menos adaptable a dominios especiales (clínico, legal).

### 3.5 Comparativa de tokenizadores

| Tokenizador | Multilingüe | Maneja abreviaciones | Maneja Twitter | Aprende del corpus |
|---|---|---|---|---|
| `split('.')` | N/A | ✗ | ✗ | ✗ |
| `text.split()` (whitespace) | N/A | ✗ | ✗ | ✗ |
| NLTK Punkt (`sent_tokenize`) | ✓ (18 idiomas) | ✓ | ✗ | ✓ |
| NLTK Treebank (`word_tokenize`) | Parcialmente | Parcialmente | ✗ | ✗ |
| NLTK TweetTokenizer | ✗ (inglés) | Parcial | ✓ | ✗ |
| spaCy `Spanish()` + sentencizer | ✓ (~25 idiomas) | ✓ (listas) | ✗ | ✗ |

---

## 4. Tokenización es contextual

**No existe "la tokenización correcta"** — existe la apropiada para tu corpus y tarea downstream. Casos donde la elección difiere:

| Caso | Tokenizador recomendado | Por qué |
|---|---|---|
| Texto noticioso formal en inglés | NLTK Treebank | Maneja abreviaciones, citas, contracciones estándar |
| Texto en español formal | NLTK Punkt español + Treebank | Modelo entrenado en español |
| Tweets, posts de Twitter/X | TweetTokenizer | Preserva emoticons, hashtags, mentions |
| Texto biomédico inglés | scispaCy (custom) | Conoce términos médicos, abreviaciones farma |
| Texto clínico español | spaCy `Spanish()` + extensión custom | Necesita `pte.`, `dx.`, `tto.`, `s/o` |
| Código fuente / texto técnico | Tokenizer custom regex | Preserva `function()`, `x.y.z`, etc. |
| Para BERT/Transformers | WordPiece/SentencePiece (subword) | Maneja OOV vía subwords |

Esto refleja el principio general del NLP clásico: **decisiones de pre-procesamiento son específicas del problema**, no universales.

---

## 5. La cascada de errores

Una tokenización incorrecta **se propaga** a todo el pipeline:

```
Bad tokenization
       ↓
Bad FreqDist        →  Bad TF-IDF  →  Bad clasificador
       ↓
Bad sentence boundaries  →  Bad POS tagging  →  Bad NER  →  Bad relations
       ↓
Bad lemmatization   →  Bad stop-word filtering  →  Bad vocab
```

Por eso vale invertir tiempo en tokenizar bien al inicio. Ahorra debugging downstream.

---

## 6. Entrenamiento de tokenizadores custom

Para dominios especializados (clínico, legal, financiero), los tokenizadores genéricos fallan. Solución: **entrenar un PunktTrainer** sobre tu corpus.

```python
from nltk.tokenize.punkt import PunktTrainer

trainer = PunktTrainer()
trainer.train(my_corpus_text)  # texto crudo, sin etiquetas

# Inspeccionar abreviaciones aprendidas
print(trainer._params.abbrev_types)
# Para clínico aparecerán: {'pte', 'dx', 'tto', 'dr', 'dra', 'sra', ...}

# Usar el tokenizer custom
my_tokenizer = trainer.get_tokenizer()
my_tokenizer.tokenize("El pte. con HTA. Dx: DM2.")
```

Esto sigue el método de **Kiss & Strunk 2006**: aprendizaje no supervisado de abreviaciones desde el corpus mismo. ~30 minutos de cómputo sobre 100k oraciones del dominio.

---

## 7. Conexión con métodos modernos

La tokenización clásica fue **reemplazada** en pipelines neural por **subword tokenization** (BPE, WordPiece, SentencePiece):

| Aspecto | Tokenización clásica | Subword tokenization |
|---|---|---|
| Vocab típico | 5k-50k | 30k-256k subwords |
| OOV | Marcado como `<UNK>` | Cero (todo descomponible) |
| Idiomas | Por idioma | Multilingüe en un solo modelo |
| Granularidad | Palabra completa | Sub-palabra (raíces, afijos) |
| Uso típico | NLTK, spaCy, IR clásico | BERT, GPT, NLLB, Llama |

Pero **tokenización clásica sigue siendo necesaria** en:
- Pipelines CPU-only sin GPU.
- IR de gran escala (Lucene, Solr, Elasticsearch).
- Sentence boundary detection (incluso pipelines BERT preprocesan con Punkt primero).
- Análisis estadístico exploratorio (FreqDist, dispersion plots).
- Casos donde necesitás tokens **legibles para humanos** (no fragmentos BPE como `un`/`break`/`able`).

Ver [BPE](/fundamentos/bpe) para la alternativa moderna.

---

## Lecturas

- Kiss & Strunk (2006), *Unsupervised Multilingual Sentence Boundary Detection*, Computational Linguistics. Ver [paper Punkt](/papers/punkt-kiss-strunk-2006).
- Marcus, Santorini & Marcinkiewicz (1993), *Building a Large Annotated Corpus of English: The Penn Treebank* — el tagset y reglas Treebank.
- Gimpel et al. (2011), *Part-of-Speech Tagging for Twitter*. Ver [paper Twitter POS](/papers/twitter-pos-gimpel-2011).
- Bird & Loper (2006), *NLTK: The Natural Language Toolkit*. Ver [paper NLTK](/papers/nltk-bird-loper-2006).

Aplicación práctica: [Lab 16](/laboratorios/lab-16/).
