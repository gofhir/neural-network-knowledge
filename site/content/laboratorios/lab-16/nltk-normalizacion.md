---
title: "Normalización: stop-words y stemming"
weight: 30
math: true
---

Cubre las celdas 26-31 del notebook. Operaciones de **limpieza léxica** que reducen ruido antes de cualquier análisis: filtrar stop-words y normalizar variantes morfológicas con stemming o lematización.

Para discusión amplia ver [Bag of Words](/fundamentos/bag-of-words). Para los algoritmos canónicos: [Porter Stemmer 1980](/papers/porter-stemmer-1980) y [WordNet 1995](/papers/wordnet-miller-1995).

---

## 1. Stop-words con NLTK

### Cargar lista (celda 27)

```python
from nltk.corpus import stopwords
print(stopwords.words('spanish'))
```

NLTK incluye listas pre-empaquetadas en **23 idiomas**. La lista española tiene ~313 palabras:

```python
['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las',
 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como',
 'más', 'pero', ...]
```

Cubre: pronombres, artículos, preposiciones, conjunciones, verbos auxiliares conjugados (ser, estar, haber, tener en muchas formas).

### Comparativa entre idiomas

```python
len(stopwords.words('english'))     # ~179
len(stopwords.words('spanish'))     # ~313
len(stopwords.words('french'))      # ~157
len(stopwords.words('german'))      # ~232
len(stopwords.words('portuguese'))  # ~207
```

**Por qué español tiene más**: morfología verbal rica. Cada auxiliar (ser, estar, haber, tener) tiene ~50+ formas conjugadas que se incluyen.

### Filtrar stop-words (celda 29)

```python
stop_words = set(stopwords.words('spanish'))      # set para lookup O(1)
text = 'Donald Trump es el 47vo y actual presidente de los Estados Unidos'
tokenized_text = nltk.word_tokenize(text)
' '.join([s for s in tokenized_text if s not in stop_words])
```

Output:

```
'Donald Trump 47vo actual presidente Estados Unidos'
```

De **13 tokens originales quedan 7**. ~46% del texto se filtró. Lo que queda es **el contenido semántico**: Donald Trump (sujeto), 47vo actual presidente (atributo), Estados Unidos (entidad). Las palabras funcionales (`es`, `el`, `y`, `de`, `los`) desaparecen.

Esto es **Zipf en acción operativo**: las palabras más frecuentes son funcionales, filtrarlas reduce ruido sin perder contenido.

### Detalle crítico: `set` vs `list`

```python
stop_words = set(stopwords.words('spanish'))   # O(1) lookup
# vs
stop_words = stopwords.words('spanish')         # O(n) lookup
```

Para 5572 SMS × 15 tokens × 313 stopwords = ~26M comparaciones en list, ~80k en set. **300x más rápido**.

### El problema sistemático de "not"

```python
'no' in stop_words   # True
```

`not` y sus equivalentes están en la lista. Esto es **problemático** en:

- **Sentiment analysis**: `"no me gusta"` → filtrado a `"gusta"` da score positivo cuando es negativo.
- **Texto clínico**: `"no presenta fiebre"` → filtrado a `"presenta fiebre"` invierte el significado.
- **Question answering**: `"¿no es Pedro?"` → `"es Pedro?"` cambia el sentido.

**Para tu pipeline FHIR-MDM**: **NUNCA filtres `no` y negaciones afines en texto clínico** sin antes extraer las negaciones con un sistema tipo NegEx (Chapman 2001).

### Case sensitivity

```python
'el' in stop_words      # True
'El' in stop_words      # False  ← mayúscula NO matchea
'EL' in stop_words      # False
```

La lista NLTK está toda en minúsculas. Patrón correcto:

```python
filtered = [s for s in tokens if s.lower() not in stop_words]
```

El lab no lo hace y es un bug silencioso. Si tu texto tiene mayúsculas al inicio de oración o ALL-CAPS, las stop-words NO se filtran.

---

## 2. Stemming y lemmatization (celda 31)

### Definiciones rigurosas

**Stemming**: reducir una palabra a su raíz por **eliminación de sufijos**. El resultado puede NO ser palabra del idioma.

- `running, runner, runs, ran → run` (stem reconocible)
- `intelligence, intelligent → intellig` (stem que no es palabra)

**Lemmatization**: reducir a la **forma canónica del diccionario** (lemma). El resultado SIEMPRE es palabra válida, pero requiere conocimiento léxico (WordNet) y POS.

- `running → running` (gerundio sustantivo) o `run` (verbo)
- `is, are, was, been → be` (auxiliar to be)
- `mice → mouse` (plural irregular)

### Los 4 métodos comparados en el lab

```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

text = "Artificial intelligence is intelligence demonstrated by machines..."
```

#### Porter (1980)

```
artifici intellig is intellig demonstr by machin . lead ai textbook
defin the field as the studi of intellig agent : ani devic that
perceiv it environ and take action that maxim it chanc of success
achiev it goal .
```

Algoritmo en 5 pasos de eliminación de sufijos. Ver [Porter Stemmer 1980](/papers/porter-stemmer-1980).

#### Snowball / Porter2 (2001)

```
artifici intellig is intellig demonstr by machin . lead ai textbook
defin the field as the studi of intellig agent : ani devic that
perceiv it environ and take action that maxim it chanc of success
achiev it goal .
```

**Casi idéntico a Porter** pero corrige bugs conocidos. Es el mismo Porter mejorado por el mismo autor en 2001. **Para producción usá Snowball**, no Porter original.

#### Lancaster (1990)

```
art intellig is intellig demonst by machin . lead ai textbook defin
the field as the study of intellig ag : any devic that perceiv it
environ and tak act that maxim it chant of success achiev it goal .
```

**Brutalmente agresivo**: `artificial → art`, `agents → ag`, `chance → chant`. Sobre-strippa hasta producir stems inútiles. `aviation → av` (caso famoso). **Casi nadie lo usa hoy**.

#### WordNet Lemmatizer

```
Artificial intelligence is intelligence demonstrate by machine .
Leading AI textbook define the field a the study of intelligent
agent : any device that perceives it environment and take action
that maximize it chance of successfully achieve it goal .
```

**Mucho más conservativo**. Solo cambia palabras donde tiene evidencia de inflexión:
- `machines → machine`, `textbooks → textbook`, `agents → agent` (plurales).
- `demonstrated → demonstrate` (con `pos='v'`).
- `is` NO se reduce a `be` porque WordNetLemmatizer **requiere POS explícito** para irregulares.

Output **legible**, palabras reales. Requiere `nltk.download('wordnet')` y `nltk.download('omw-1.4')`.

### Comparativa rápida

| Aspecto | Porter | Snowball | Lancaster | WordNet |
|---|---|---|---|---|
| Velocidad | ~5 ms/palabra | ~5 ms/palabra | ~5 ms/palabra | ~50 ms/palabra |
| Stems = palabras reales? | No | No | No | Sí |
| Multilingüe? | Solo inglés | Sí (17 idiomas) | Solo inglés | Inglés + OMW (~30 idiomas) |
| Maneja irregulares? | No | No | No | Sí (con POS correcto) |
| Recomendación 2026 | Histórico | **Producción** | No usar | Tareas legibles |

---

## 3. Stemming en español

```python
ss = SnowballStemmer('spanish')
words = ['corriendo', 'corrió', 'corredor', 'corren', 'corrida']
[ss.stem(w) for w in words]
# ['corr', 'corr', 'corredor', 'corren', 'corr']
```

Resultados imperfectos pero útiles para colapsar formas verbales y plurales.

### Sobre vocabulario clínico

```python
ss = SnowballStemmer('spanish')
ss.stem("hipertensión")    # 'hipertension'
ss.stem("losartán")        # 'losartán' (sin cambios, no es derivacional)
ss.stem("pacientes")       # 'pacient'
ss.stem("diabéticos")      # 'diabét'
ss.stem("tratamiento")     # 'trat'
```

Snowball español **preserva términos técnicos** (fármacos, nombres propios) y **colapsa morfología regular** (plurales, derivacionales). Buena propiedad para BoW clínico.

---

## 4. La función `lemmatize` del lab (celda 31)

```python
def lemmatize(lemmatizer, word):
    word = lemmatizer.lemmatize(word, pos='a')   # adjetivo
    word = lemmatizer.lemmatize(word, pos='n')   # sustantivo
    word = lemmatizer.lemmatize(word, pos='v')   # verbo
    return word
```

**Esto NO es la forma correcta de lematizar**. Intenta lematizar como adjetivo, luego sustantivo, luego verbo. Problemas:

- Si una palabra es adjetivo en contexto pero existe como sustantivo, se aplica lematización incorrecta.
- Por ejemplo `meeting`: `pos='a'` → `meeting`. `pos='n'` → `meeting`. `pos='v'` → `meet`. **Depende del orden.**

**Forma correcta**: POS tagging primero, luego lemmatize con el POS real.

```python
def lemmatize_correctly(word, lemmatizer):
    tag = nltk.pos_tag([word])[0][1]
    if tag.startswith('J'):  return lemmatizer.lemmatize(word, pos='a')
    elif tag.startswith('V'): return lemmatizer.lemmatize(word, pos='v')
    elif tag.startswith('N'): return lemmatizer.lemmatize(word, pos='n')
    elif tag.startswith('R'): return lemmatizer.lemmatize(word, pos='r')
    else: return lemmatizer.lemmatize(word)
```

Es lo que **spaCy hace automáticamente** — y por eso lematizar con spaCy es **más preciso** que con NLTK, aunque más lento.

---

## 5. Cuándo usar cada uno

| Tarea | Recomendación |
|---|---|
| BoW para clasificación spam | Snowball stemmer (rápido) |
| TF-IDF para retrieval | Snowball stemmer |
| Análisis de keywords visible al usuario | Lemmatization (WordNet) |
| Pipeline para Transformers (BERT, NLLB) | **Ninguno** — los Transformers usan subword tokenization que ya maneja morfología |
| Procesamiento médico/legal | Lemmatization + lexicón custom |

Para **Transformers**, **nunca apliques stemming antes**. El tokenizer del modelo (BPE/WordPiece/SentencePiece) ya maneja morfología internamente. Pasarle `"intellig"` en lugar de `"intelligence"` confunde al modelo (vio `"intelligence"` durante pre-training).

---

## 6. Aplicación a tu trabajo MDM-FHIR

Para texto clínico español:

```python
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

ss = SnowballStemmer('spanish')
sw_es = set(stopwords.words('spanish'))

# Extender stopwords con jerga clínica
sw_clinico = sw_es | {
    'paciente', 'pte', 'sr', 'sra',           # genéricos pero ubicuos
    'presenta', 'refiere', 'manifiesta',       # verbos clínicos comunes
    'años', 'mes', 'día', 'fecha', 'hora',     # temporales
}

def normalize_clinical(text):
    tokens = nltk.word_tokenize(text)
    # Lowercase + filtrar stopwords + stemming
    return [ss.stem(t.lower()) for t in tokens 
            if t.isalpha() and t.lower() not in sw_clinico]
```

**Para producción**:
- **NO filtres `no`** y negaciones — usa NegEx antes.
- **NO uses autocorrect** genérico — daña abreviaciones médicas (`pte. → patente`).
- **Considerá un lemmatizer médico** si tenés UMLS o SNOMED CT disponible — más preciso que Snowball genérico.

La práctica local del lab — scripts 30-34 en [clase_16/practica](https://github.com/) — implementa esto sobre MEDDOCAN/Cantemist/PharmaCoNER con experimentos comparativos.

---

## Lecturas

- [Bag of Words (fundamento)](/fundamentos/bag-of-words) — contexto operativo.
- [Porter Stemmer 1980](/papers/porter-stemmer-1980) — el algoritmo canónico.
- [WordNet 1995](/papers/wordnet-miller-1995) — el lemmatizer base.

Anterior: [Estadísticas de texto](nltk-estadisticas).
Siguiente: [spaCy: POS, NER, Dependency Parsing](spacy-pipeline).
