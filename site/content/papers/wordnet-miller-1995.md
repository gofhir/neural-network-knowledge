---
title: "WordNet (Introducción)"
weight: 60
math: true
---

{{< paper-card
    title="Introduction to WordNet: An On-line Lexical Database"
    authors="Miller, Beckwith, Fellbaum, Gross, Miller"
    year="1990/1993"
    venue="International Journal of Lexicography 3(4)"
    pdf="/papers/wordnet-miller-1995.pdf" >}}
Construye un lexicón inglés organizado por **relaciones semánticas** (synsets, hipónimos, meronimia, antonimia) en lugar de orden alfabético. Inspirado en teoría psicolingüística sobre cómo el lexicón mental humano está organizado. **El recurso léxico más influyente de la historia del NLP**: incluido en NLTK, spaCy, sklearn; reusado en >50,000 papers académicos; base de Open Multilingual WordNet (30+ idiomas) y BabelNet.
{{< /paper-card >}}

---

## Contexto

A inicios de los 80, los diccionarios digitales eran simples conversiones del formato papel: ordenados alfabéticamente, sin relaciones semánticas explícitas entre palabras. Para tareas como **desambiguación de sentidos** (WSD), **expansión de queries** en IR, o **comprensión semántica**, los investigadores necesitaban un lexicón **estructurado** donde pudieran consultar programáticamente "¿qué palabras son sinónimos de X?", "¿qué es Z una parte de?".

George Miller (padre de la psicología cognitiva, autor de *"The Magical Number Seven, Plus or Minus Two"*) lideró el proyecto en Princeton desde 1985 con un equipo de psicólogos y lingüistas. La motivación: construir un lexicón siguiendo principios **psicolingüísticos**, no convenciones editoriales.

---

## Ideas principales

### 1. La matriz léxica: forms × meanings

```
                Word Forms
              F1     F2     F3   ... Fn
Word Meanings ──────────────────────────
M1            E1,1   E1,2
M2                   E2,2
...                                  ...
```

- **Word form**: ortografía (cómo se escribe).
- **Word meaning**: el concepto léxico asociado.

Dos fenómenos básicos:
- **Sinonimia**: dos forms (F1, F2) expresan el mismo meaning (M1). Misma fila.
- **Polisemia**: un form (F2) expresa múltiples meanings. Misma columna.

WordNet representa cada meaning como un **synset** — el conjunto de forms sinónimas que lo expresan.

### 2. Synsets como unidad fundamental

Un synset es un conjunto de palabras intercambiables en cierto contexto:

- `{board, plank}` → "pieza de madera"
- `{board, committee}` → "grupo de personas reunidas"

Ambos contienen "board", pero **son synsets distintos** representando conceptos distintos. La polisemia se modela como **un word form en múltiples synsets**.

Cuando no hay sinónimos suficientes para distinguir el sentido, WordNet incluye un **gloss** (definición breve):

```
{board, (a person's meals, provided regularly for money)}
```

### 3. Relaciones semánticas tipadas

WordNet conecta synsets mediante relaciones explícitas:

| Relación | Definición | Ejemplo |
|---|---|---|
| Synonymy | Misma synset | `{rise, ascend}` |
| Antonymy | Pares opuestos (entre word forms) | `[rise/fall]`, `[hot/cold]` |
| Hyponymy / Hypernymy | ISA hierarchy | `{maple}` hyp. de `{tree}` hyp. de `{plant}` |
| Meronymy / Holonymy | Part-whole | `{wheel}` meronym de `{car}` |
| Entailment | Verbo X implica Y | `snore` entails `sleep` |
| Cause | Verbo X causa Y | `kill` causes `die` |
| Similar-to | Adjetivos próximos a un par antónimo | `{wet, soaked, drenched}` |

La organización jerárquica de nouns (hyponymy) llega a ~12 niveles de profundidad típicamente:

```
canary → finch → passerine → bird → vertebrate → animal → organism → entity
```

### 4. Particionamiento por POS

WordNet **no mezcla** categorías sintácticas. Cuatro subsistemas independientes:

| POS | Tamaño 1993 | Tamaño actual |
|---|---|---|
| Nouns | 48,800 synsets | ~82,000 |
| Verbs | ~10,000 | ~13,000 |
| Adjectives | ~20,000 | ~21,000 |
| Adverbs | ~3,500 | ~4,500 |

Justificación: **estudios psicolingüísticos** sugieren que nouns, verbs, adjectives se almacenan en subsistemas neurales distintos. La aphasia anómica selectiva (pérdida de capacidad de nombrar objetos sin perder otras funciones) lo apoya empíricamente.

### 5. Morfología en la interfaz, no en la base

WordNet **no guarda formas inflexionales** (`run, runs, running, ran`). En lugar de eso, el interfaz tiene un **morphological analyzer** que reduce a lemma antes del lookup:

- `running → run`
- `mice → mouse`
- `better → good` (irregular)

Esto es lo que NLTK usa cuando llamás `WordNetLemmatizer().lemmatize(word, pos='v')`.

---

## Resultados y validación

El paper es **un system paper / theory paper**, no presenta benchmarks numéricos. La validación es por **adopción institucional**:

- 1993: 51,500 word forms, 70,100 synsets en WordNet.
- 2025: >117,000 synsets en versión actual.
- Cobertura comparable a un diccionario colegial estándar pero **organizado por significado**.

---

## Limitaciones reconocibles

- **Solo inglés** en el proyecto original. Extensiones modernas (EuroWordNet, MultiWordNet, OMW) cubren ~30 idiomas pero con menor cobertura.
- **WordNet español vía OMW-1.4**: ~30,000 synsets, **muy inferior** al inglés (~117,000).
- **Sentidos discretos**: WordNet asume cada palabra polisémica tiene N sentidos. El significado real es continuo y contextual — esto motivó word embeddings.
- **Congelado**: Princeton terminó desarrollo activo en ~2012. Términos nuevos (`bitcoin`, `selfie`, `meme`) están ausentes.
- **No cubre dominios técnicos** (medicina, derecho, química) — para eso existen UMLS, MeSH, SNOMED CT.

---

## Por qué importa hoy

- **>50,000 citas combinadas** en Google Scholar entre los papers fundacionales de WordNet.
- **Estándar de facto** en NLP académico — incluido en NLTK (`nltk.corpus.wordnet`), spaCy, AllenNLP, Stanford CoreNLP.
- Base de **Open Multilingual WordNet** (Bond & Foster 2013) — extensión multilingüe usada cuando descargás `nltk.download('omw-1.4')`.
- Inspiró sucesores: **ConceptNet** (MIT, 2007+, relaciones más diversas), **BabelNet** (Sapienza, WordNet + Wikipedia + Wikidata en 200+ idiomas), **Wikidata**.
- Para **evaluación WSD**: muchos benchmarks usan WordNet synsets como labels gold.

En la era de Transformers, WordNet sigue siendo relevante donde **embeddings necesitan grounding simbólico** — sistemas que requieren explicabilidad (compliance, médico, legal) o reasoning paso a paso. Modelos modernos como **KnowBERT**, **ERNIE**, **Sense2vec** combinan embeddings con lookup de WordNet.

---

## Notas y enlaces

- George Miller recibió numerosos premios; el proyecto WordNet ganó el **ACL Lifetime Achievement Award**.
- Para tu trabajo en NLP clínico español: **WordNet español es pobre** para vocabulario médico. Necesitás **UMLS + SNOMED CT + LOINC** para terminología clínica seria.
- Documentación oficial: `wordnet.princeton.edu` (proyecto Princeton). API en NLTK: `nltk.corpus.wordnet.synsets(word)`.
- Sucesor moderno: **WordNet 3.1** (última versión Princeton, 2011). Extensión multilingüe: **Open Multilingual WordNet** (2013+).

Ver fundamentos: [Bag of Words](/fundamentos/bag-of-words) · [Tokenización clásica](/fundamentos/tokenizacion-clasica).
