---
title: "spaCy: POS, NER, Dependency Parsing"
weight: 40
math: true
---

Cubre las celdas 32-42 del notebook. Cambio de paradigma respecto a NLTK: **spaCy es pipeline industrial**, con una API uniforme (`nlp(text)`) que procesa todo en una llamada. Diseñado para producción, ~10-100x más rápido que NLTK por implementación interna en Cython.

Para tokenización clásica ver [Tokenización clásica](/fundamentos/tokenizacion-clasica). Para comparativa NLTK vs spaCy ver al final de [Estadísticas de texto](nltk-estadisticas).

---

## 1. Setup (celdas 32-33)

```python
!pip install spacy
```

Instala el **framework** spaCy. **NO incluye modelos preentrenados**. Para inglés:

```bash
python -m spacy download en_core_web_sm    # ~12 MB, modelo pequeño
```

Variantes disponibles:
- `_sm` (~12 MB): vocabularios + POS + parser + NER + lemma.
- `_md` (~40 MB): + word vectors GloVe estáticos.
- `_lg` (~560 MB): vectors más grandes.
- `_trf` (~440 MB): basado en RoBERTa Transformer, accuracy SOTA pero requiere GPU.

Para español: `es_core_news_sm/md/lg/trf`. Para biomédico: **scispaCy** (`en_core_sci_md`).

---

## 2. POS tagging + Dependency parsing (celda 35)

```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
displacy.render(doc, style='dep', jupyter=True, options={'distance': 100})
```

Una sola llamada `nlp(text)` ejecuta **todo el pipeline**:

```
texto crudo
   ↓
[tokenizer]    →  Doc con tokens
   ↓
[tagger]       →  cada token tiene .pos_ y .tag_
   ↓
[parser]       →  estructura sintáctica de dependencias
   ↓
[ner]          →  entidades nombradas reconocidas
   ↓
[lemmatizer]   →  cada token tiene .lemma_
   ↓
Doc enriquecido
```

### Output esperado para la frase canónica

Inspeccionando tokens:

```python
for token in doc:
    print(f"{token.text:12} {token.pos_:6} {token.tag_:6} {token.dep_:12} head={token.head.text}")
```

| Token | POS | Tag | Dependencia | Head |
|---|---|---|---|---|
| Apple | PROPN | NNP | nsubj | looking |
| is | AUX | VBZ | aux | looking |
| looking | VERB | VBG | ROOT | (sí mismo) |
| at | ADP | IN | prep | looking |
| buying | VERB | VBG | pcomp | at |
| U.K. | PROPN | NNP | compound | startup |
| startup | NOUN | NN | dobj | buying |
| for | ADP | IN | prep | buying |
| $ | SYM | $ | quantmod | billion |
| 1 | NUM | CD | compound | billion |
| billion | NUM | CD | pobj | for |

`displacy` renderiza este árbol como SVG inline en Jupyter con flechas curvas conectando cada token a su head.

### Aplicación a tu trabajo MDM-FHIR

Sobre `"El paciente presenta hipertensión arterial controlada con losartán 50 mg desde 2020"`, un parser de dependencias extrae estructura:

- `paciente` → `nsubj` de `presenta`.
- `hipertensión` → `dobj` de `presenta`.
- `losartán` → relacionado vía `con` → `controlada`.
- `50 mg` → `compound` de `losartán`.
- `2020` → fecha vía `desde`.

Esto es **mucho más estructurado** que un BoW. Mapeable a recursos FHIR:
- `Condition` con `code=HTA, status=controlled, subject=Patient`.
- `MedicationStatement` con `medication=Losartan, dose=50 mg, effectiveDateTime=2020`.

---

## 3. Named Entity Recognition (celdas 36-37)

```python
text = "When Donald Trump started working on that project in 2007, few people outside of the United States took him seriously."
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
displacy.render(doc, style='ent', jupyter=True)
```

Output:

```
Donald Trump      5    17   PERSON
2007              53   57   DATE
the United States 88   106  GPE
```

### El tagset OntoNotes 5 (18 categorías inglés)

| Tag | Significado | Ejemplos |
|---|---|---|
| `PERSON` | Persona humana | Donald Trump, Quijote |
| `NORP` | Nationality / Religion / Political group | Christian, Republican |
| `FAC` | Edificios, aeropuertos | Empire State, Highway 1 |
| `ORG` | Organizaciones, empresas | Apple, NATO, OMS |
| `GPE` | Países, ciudades, estados | the United States, Madrid |
| `LOC` | Geografía no-GPE | the Andes, Amazon River |
| `PRODUCT` | Productos | iPhone, Coca-Cola |
| `EVENT` | Eventos | World War II |
| `WORK_OF_ART` | Títulos | Don Quijote, Star Wars |
| `LAW` | Leyes | the Constitution, HIPAA |
| `LANGUAGE` | Idiomas | Spanish, Mandarin |
| `DATE` | Fechas | 2007, last Tuesday |
| `TIME` | Tiempos <24h | 3 a.m. |
| `PERCENT` | Porcentajes | 50%, three quarters |
| `MONEY` | Cantidades monetarias | $1 billion |
| `QUANTITY` | Cantidades con unidades | 50 mg, 10 km |
| `ORDINAL` | first, third |
| `CARDINAL` | 12, three, seventy |

### Cómo funciona internamente

`en_core_web_sm` usa **transition-based con CNN** (no BERT — sería `_trf`):

1. **Tokenización**.
2. **Features léxicos**: shape (`Xxxx`), prefixes, suffixes, embeddings.
3. **CNN encoder**: convoluciones sobre la secuencia capturan contexto local.
4. **Transition parser**: decide acciones (`BEGIN`, `IN`, `LAST`, `UNIT`, `OUT`).
5. **Output**: lista de Spans con `start_char`, `end_char`, `label_`.

Accuracy reportada en OntoNotes 5 (newswire):
- `_sm`: F1 ≈ 0.85.
- `_lg`: F1 ≈ 0.86.
- `_trf`: F1 ≈ 0.90.

Para **dominio clínico** estos números caen ~0.50-0.65 — necesitás modelos entrenados específicamente (scispaCy, NER médico custom).

### El problema crítico para tu MDM-FHIR

```python
nlp_es = spacy.load("es_core_news_sm")
doc = nlp_es("El paciente Juan Pérez con DNI 12345678X presenta HTA con losartán 50 mg")
for ent in doc.ents:
    print(ent.text, "→", ent.label_)
```

Output probable:

```
Juan Pérez   → PER
12345678X    → MISC (o nada)
HTA          → MISC (o nada)
losartán     → MISC (o nada)
```

**El modelo español default solo tiene 4 categorías** (`PER`, `LOC`, `ORG`, `MISC`). Mucho más pobre que el inglés. Para entidades clínicas necesitás:

1. **Reglas custom** (regex para DNI, dosis, etc.).
2. **NER custom** entrenado con corpus anotado (MEDDOCAN es buen training set).
3. **Modelo biomédico** especializado (PlanTL-ES/clinical-ner, scispaCy si traducís a inglés).

---

## 4. Tokenización con spaCy (celdas 38-42)

### El patrón "constructor manual"

```python
from spacy.lang.en import English

nlp = English()                    # bare bones (sin modelo entrenado)
nlp.add_pipe('sentencizer')        # agregar componente
doc = nlp("this is a sentence. Mr. President said something. U.S.A. means...")
for sent in doc.sents:
    print(sent.text)
```

`English()` crea un objeto vacío con **solo el tokenizer básico**. Sin POS, NER, parser, lemma. Es **16x más liviano** en RAM que `spacy.load("en_core_web_sm")`. Útil cuando solo necesitás tokenizar rápido.

### Comparativa `Spanish()` vs NLTK Punkt en abreviaciones formales

```python
from spacy.lang.es import Spanish
nlp = Spanish()
nlp.add_pipe('sentencizer')
nlp("El Sr. Pérez consulta. Tiene HTA.").sents  # → 2 oraciones correctas
```

spaCy `Spanish()` tiene **listas hardcoded** de abreviaciones (`Sr.`, `Sra.`, `Dr.`, `Lic.`, `Ing.`, etc.). Más predecible que Punkt (estadístico) pero menos adaptable.

**Donde spaCy falla**: abreviaciones clínicas raras (`pte.`, `dx.`, `tto.`, `s/p`). Para esos casos necesitás entrenar Punkt custom o extender la lista de spaCy.

### Doc → Span → Token

```python
for sent in doc.sents:           # Span por oración
    for token in sent:            # Token por palabra
        print(token.text)
```

Estructura jerárquica de spaCy:

```
Doc
├── Span (oración 1)
│   ├── Token (palabra 1)
│   ├── Token (palabra 2)
│   └── ...
├── Span (oración 2)
│   └── ...
```

`Span` y `Token` son **views** sobre el `Doc`, no copias. Eficiente en memoria.

### Atributos útiles del Token

| Atributo | Significado |
|---|---|
| `.text` | Texto literal |
| `.lemma_` | Lemma |
| `.pos_` | POS Universal |
| `.tag_` | POS detallado (Penn Treebank) |
| `.dep_` | Dependencia sintáctica |
| `.is_alpha` | Solo letras? |
| `.is_stop` | ¿Es stopword? |
| `.is_punct` | ¿Es puntuación? |
| `.like_num` | ¿Parece número? |
| `.like_email` | ¿Parece email? |
| `.like_url` | ¿Parece URL? |
| `.vector` | Word embedding (si modelo lo tiene) |
| `.head` | Token padre en árbol de dependencias |
| `.children` | Tokens hijos |
| `.ent_type_` | Categoría NER si aplica |

### `token.is_stop` — alternativa a NLTK

```python
from spacy.lang.es.stop_words import STOP_WORDS as STOP_ES
len(STOP_ES)   # ~551 (más amplio que NLTK español de 313)
```

```python
filtered = [t.text for t in doc if not t.is_stop and not t.is_punct]
```

---

## 5. NLTK vs spaCy: cuándo cada uno

| Aspecto | NLTK | spaCy |
|---|---|---|
| Diseño | Pedagógico, académico | Industrial, producción |
| Performance | Optimizado para legibilidad | 10-100x más rápido (Cython) |
| API | Muchas funciones independientes | Una función central: `nlp(text)` |
| Modelos | Componentes individuales (MB) | Integrados (~50-500 MB) |
| Stemming | Sí (Porter, Snowball, Lancaster) | NO (solo lemma) |
| POS tagger | Sí pero solo inglés bueno | Sí, 25+ idiomas con calidad similar |
| NER | Limitado (ne_chunk inglés) | Built-in en todos los modelos |
| Dependency parsing | No incluido | Sí, built-in |
| Word embeddings | No | Sí (en modelos md/lg) |
| Visualización | Funciones aisladas | displacy (SVG calidad publicación) |

**Recomendación general 2026**:
- Para **preprocesamiento + pipeline serio**: spaCy.
- Para **experimentos comparativos + flexibilidad**: NLTK.
- Para **producción multilingüe**: spaCy uniformemente.
- Para **NER clínico**: scispaCy (inglés) o NER custom entrenado en MEDDOCAN (español).

---

## Lecturas

- spaCy documentation: `spacy.io`.
- scispaCy (proyecto AllenNLP): biomedical NLP con spaCy. `allenai.github.io/scispacy/`.
- Honnibal & Montani (Explosion AI): autores de spaCy.

Anterior: [Normalización: stop-words y stemming](nltk-normalizacion).
Siguiente: [Actividades 1-3](actividades-1-3).
