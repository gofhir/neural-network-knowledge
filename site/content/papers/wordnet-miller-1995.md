---
title: "WordNet - An On-line Lexical Database"
weight: 162
math: true
---

{{< paper-card
    title="Introduction to WordNet: An On-line Lexical Database"
    authors="Miller, Beckwith, Fellbaum, Gross, Miller"
    year="1995"
    venue="International Journal of Lexicography"
    pdf="/papers/wordnet-miller-1995.pdf" >}}
Presenta **WordNet**, la primera base de datos léxica para inglés organizada según principios psicolingüísticos en lugar del orden alfabético tradicional. Introduce el **synset** (conjunto de sinónimos) como unidad fundamental del lexicón y una red de **relaciones semánticas tipadas** (hipernimia/hiponimia, meronimia, antonimia, entailment, causación) que enlaza más de 70,000 conceptos. Es el recurso léxico simbólico más influyente de la historia del NLP y la base sobre la que se construyen el `WordNetLemmatizer` de NLTK y prácticamente toda tarea de Word Sense Disambiguation hasta hoy.
{{< /paper-card >}}

---

## Contexto

A finales de los 70 y comienzos de los 80, el procesamiento de lenguaje natural carecía de **un recurso léxico digital comprehensivo** para inglés. Los diccionarios existían en papel y en algunas versiones digitalizadas, pero:

- **Estaban organizados alfabéticamente**, no semánticamente. Buscar "all synonyms of run" requería revisión manual extensiva.
- **No tenían estructura formal**: definiciones en prosa, no enlaces explícitos entre conceptos relacionados.
- **No reflejaban teoría psicolingüística**: la organización seguía conveniencias de imprenta, no la estructura cognitiva del lexicón mental humano.

Para tareas como **desambiguación de palabras**, **expansión de queries en IR**, **traducción automática** o **comprensión de texto**, los investigadores necesitaban un lexicón estructurado donde pudieran consultar programáticamente "¿qué palabras son sinónimos de X?", "¿cuáles son los hipónimos de Y?", "¿qué es Z una parte de?".

### El proyecto Princeton

En **1985**, un grupo de psicólogos y lingüistas en Princeton (encabezado por George Miller, padre de la psicología cognitiva) decidió construir un lexicón siguiendo principios **psicolingüísticos** — no por conveniencia editorial, sino para reflejar cómo el lexicón humano (mental lexicon) está organizado.

Miller era figura mayor:

- Su paper de 1956 *"The Magical Number Seven, Plus or Minus Two"* es uno de los más citados de toda la psicología.
- Cofundador de Cognitive Science.
- Pionero de la psicolingüística junto a Noam Chomsky.

La motivación que el paper articula:

> *"Standard alphabetical procedures for organizing lexical information put together words that are spelled alike and scatter words with similar or related meanings haphazardly through the list."*

Es decir: el diccionario alfabético es **anti-cognitivo**. WordNet rompe con esto.

### Investigaciones psicolingüísticas que motivaron el diseño

El paper cita estudios empíricos clave:

1. **Fillenbaum & Jones (1965)**: probaron **word association** — pedir a sujetos la primera palabra que les viene tras un estímulo. Resultado: **el 79% de las asociaciones con sustantivos son sustantivos**, 65% con adjetivos son adjetivos, 43% con verbos son verbos. **El lexicón mental está particionado por categoría sintáctica**.

2. **Collins & Quillian (1969)**: midieron **tiempos de reacción** para verificar afirmaciones como "Un canario puede cantar" vs "Un canario puede volar" vs "Un canario tiene piel". Resultado: **más tiempo a medida que la propiedad pertenece a un nivel jerárquico más alto** (volar = pájaro, tener piel = animal). Sugiere que **la información se almacena jerárquicamente**, con propiedades en el nivel más general y herencia a sub-conceptos.

3. **Anomic aphasia**: pacientes con lesiones en hemisferio izquierdo pierden la capacidad de **nombrar objetos** pero conservan otras facultades. Esto sugiere que **los sustantivos están organizados en un subsistema léxico separado** del de verbos.

WordNet fue construida con estos hallazgos como guía: **categorías separadas para nouns/verbs/adjectives/adverbs**, **estructura jerárquica para nouns** (herencia), **estructura diferente para verbs** (relaciones de entailment), **estructura distinta para adjectives** (espacio N-dimensional).

---

## Ideas principales

WordNet aporta **tres cosas profundamente interrelacionadas**:

1. **El concepto de synset (synonym set)** como unidad fundamental del lexicón.
2. **Una red de relaciones semánticas tipadas** que enlaza synsets entre sí.
3. **Una implementación operacional** distribuida libremente, con APIs para múltiples lenguajes y datasets en docenas de idiomas (vía Open Multilingual WordNet).

A 1993 (fecha de la versión documentada en el paper), WordNet contenía:

- **51,500 palabras simples**
- **44,100 collocations** (frases-palabra como *swimming pool*)
- **70,100 synsets** (conceptos)

Para 2026, la versión actual (WordNet 3.1 + extensions vía OMW-1.4) contiene **>117,000 synsets** y se extiende a >200 idiomas vía proyectos derivados.

### La matriz léxica: forms × meanings

Miller introduce la **lexical matrix** como abstracción fundamental:

```
                Word Forms
              F1     F2     F3   ... Fn
Word Meanings ──────────────────────────
M1            E1,1   E1,2
M2                   E2,2
M3                          E3,3
...                                  ...
Mm                                  Em,n
```

- **Word form**: la realización física (ortografía, fonética).
- **Word meaning**: el concepto léxico asociado.
- Una **celda E[i,j]** indica que el form $F_j$ puede expresar el meaning $M_i$.

**Dos fenómenos básicos** se ven en esta matriz:

| Fenómeno | Definición | Visualización |
|---|---|---|
| **Sinonimia** | Dos formas (F1, F2) pueden expresar el mismo meaning (M1) | Dos celdas en la **misma fila** |
| **Polisemia** | Un form (F2) puede expresar múltiples meanings | Dos celdas en la **misma columna** |

WordNet representa cada meaning $M$ con un **synset** = `{F1, F2, ...}` (lista de formas sinónimas que lo expresan). Las relaciones semánticas son **flechas entre synsets**, no entre formas individuales.

### Synsets — el corazón de WordNet

Un **synset** es un conjunto de formas sinónimas que comparten un significado:

- `{board, plank}` → meaning: "una pieza de madera".
- `{board, committee}` → meaning: "un grupo de personas reunidas para un propósito".

Ambos contienen "board" pero **representan conceptos distintos**. La palabra `board` es **polisémica** (3+ sentidos), pero cada sentido corresponde a un synset diferente.

**Las llaves `{ }`** son notación para sinonimia. **Los brackets `[ ]`** se usan para relaciones lexicales no-semánticas.

Cuando un synset es ambiguo (no hay sinónimos suficientes para distinguirlo), WordNet incluye un **gloss** (definición breve):

```
{board, (a person's meals, provided regularly for money)}
```

El paréntesis es el gloss diferenciador.

### Relaciones semánticas

WordNet organiza synsets mediante **relaciones tipadas**. Las principales:

#### Synonymy (synset itself)

Ya discutido. Define el synset.

#### Antonymy

Relación entre **word forms**, no entre meanings. Curiosamente asimétrica en intuición:

- `{rise, ascend}` y `{fall, descend}` representan conceptos opuestos.
- Pero **[rise/fall]** y **[ascend/descend]** son antonyms; **rise/descend** no se siente igual.

Antonymy es **relación lexical** (entre formas), aunque tiene base semántica.

#### Hyponymy / Hypernymy (ISA relation)

Relación semántica entre meanings:

- `{maple}` es hyponym de `{tree}` (maple es un tipo de tree).
- `{tree}` es hyponym de `{plant}` (tree es un tipo de plant).

Es **transitiva**: si X is_a Y y Y is_a Z, entonces X is_a Z.
Es **asimétrica**: tree no es maple.

Típicamente cada noun tiene **exactamente un hypernym**, lo cual genera una **estructura de árbol** ("inheritance system"). Esta es la base de la organización jerárquica de los sustantivos en WordNet.

**Profundidad típica**: máximo ~12 niveles desde la raíz (`entity`) hasta los terminales. Por ejemplo:

```
canary → finch → passerine → bird → vertebrate → animal → organism → entity
```

8 niveles.

#### Meronymy / Holonymy (HASA / PART-OF relation)

- `wheel` es **meronym** de `car` (wheel es parte de car).
- `car` es **holonym** de `wheel`.

Es transitiva (con cualificaciones — un dedo es parte de mano que es parte de brazo, pero "un dedo es parte de brazo" es discutible).

#### Entailment (entre verbos)

Si verbo X entails verbo Y, entonces hacer X implica hacer Y:

- `snore` entails `sleep` (roncar implica dormir).
- `divorce` entails `marry` (divorciarse implica haberse casado).

#### Cause (entre verbos)

- `kill` causes `die`.
- `show` causes `see`.

#### Pertainymy (entre adjetivos y nouns)

- `dental` pertains_to `tooth`.
- `Italian` pertains_to `Italy`.

#### Similar to (entre adjetivos)

Los adjetivos se organizan en **grupos de similaridad** alrededor de un par antónimo:

- `{wet, soaked, drenched, ...}` similar_to (head)
- antonym
- `{dry, parched, arid, ...}` similar_to (head)

### Organización particional por POS

WordNet **NO mezcla categorías sintácticas**. Cuatro subsistemas independientes:

| POS | Tamaño 1993 | Tamaño actual (~2024) | Estructura organizativa |
|---|---|---|---|
| **Nouns** | ~57,000 forms / 48,800 synsets | ~117,000 synsets | Jerarquía profunda (hypernymy/hyponymy/meronymy) |
| **Verbs** | ~21,000 forms | ~25,000 synsets | Jerarquía + entailment + causación |
| **Adjectives** | ~19,500 forms | ~21,000 synsets | Espacio N-dimensional con antonymy + similar_to |
| **Adverbs** | ~3,500 forms | ~4,500 synsets | Mayormente derivados de adjetivos vía pertainymy |

**Por qué la separación**: estudios psicolingüísticos sugieren que estas categorías se almacenan **en sistemas neurales distintos**. La aphasia anómica selectiva (pierde solo nouns) confirma esto.

### Morfología

WordNet **no almacena formas inflexionales** (run, runs, running, ran). Esto crearía explosión combinatoria.

En su lugar, **el interface** (no la base) tiene un **morphological analyzer** que reduce formas inflexionales a su lemma antes del lookup:

- `running` → `run`
- `mice` → `mouse`
- `better` → `good` (irregular)

**Esto es lo que NLTK usa cuando llamas `lemmatizer.lemmatize(word, pos='v')`**.

---

## Implementación

WordNet se distribuye como:

- **Base de datos** (archivos planos `.dict`, `.idx`, `.exc`).
- **APIs** en C, Java, Python (vía NLTK).
- **Browser** (interface web/desktop para explorar synsets visualmente).

El paper original menciona que la base de datos crece continuamente. Eso ha sido cierto: WordNet ha tenido **8+ versiones mayores** desde 1991 (1.0, 1.5, 1.6, 1.7, 1.7.1, 2.0, 2.1, 3.0, 3.1) hasta 2011, después del cual el proyecto Princeton ha sido relevado por extensiones académicas (Open Multilingual WordNet, BabelNet, etc.).

---

## Resultados

WordNet es probablemente **el recurso léxico más influyente jamás construido**. Algunas métricas:

- **Citas**: a mayo de 2026, los papers fundacionales de WordNet acumulan **>50,000 citas** sumadas (Miller 1995, Fellbaum 1998, etc.).
- **Adopción**: incluido por defecto en NLTK, spaCy, Stanford CoreNLP, AllenNLP. **El estándar de facto** en NLP académico.
- **Derivados**: WordNet ha inspirado **decenas** de proyectos similares: ConceptNet (2007), BabelNet (2010), Open Multilingual WordNet (2014), etc.
- **Premios**: George Miller recibió numerosos honores; el proyecto WordNet fue reconocido con el ACL Lifetime Achievement Award.

### Aplicaciones canónicas

- **Lemmatization**: el `WordNetLemmatizer` de NLTK.
- **Word Sense Disambiguation (WSD)**: dado un texto, decidir qué synset de cada palabra está siendo usado. Tareas SemEval y CoNLL.
- **IR Expansion**: en queries de búsqueda, expandir con sinónimos del synset.
- **Semantic Similarity**: calcular similitud entre palabras vía distancia en el grafo de WordNet (path similarity, Wu-Palmer, Resnik, Lin, Leacock-Chodorow, etc.).
- **Text simplification**: reemplazar palabras complejas por hipernyms más generales.
- **Question Answering**: para preguntas "¿qué es X?", buscar el hypernym de X.

### Antes y después de WordNet

| Antes (1985) | Después |
|---|---|
| Lexicones improvisados por cada proyecto | WordNet = estándar |
| Sin relaciones semánticas explícitas | Synsets + relaciones tipadas |
| Solo inglés vía diccionarios papel | Open Multilingual WordNet en 30+ idiomas |
| Sin estructura jerárquica | Inheritance system para nouns |
| Pocas tareas de NLP semántico | Decenas de tareas que asumen WordNet |

---

## Limitaciones

El paper de 1995 no critica abiertamente sus propias limitaciones, pero el lector moderno las identifica:

### Solo inglés

WordNet original es exclusivamente inglés estadounidense. Para otros idiomas:

- **EuroWordNet** (1998-2002, 8 idiomas europeos).
- **MultiWordNet** (italiano, español).
- **Open Multilingual WordNet** (OMW): meta-proyecto que enlaza WordNets en 30+ idiomas.
- **OMW-1.4** (lo que descargas en NLTK con `nltk.download('omw-1.4')`) es la versión 2017+ de OMW. Contiene WordNets de español, francés, italiano, japonés, alemán, portugués, etc.

**Pero el WordNet español de OMW es mucho más pobre que el inglés** (~30,000 synsets en español vs >117,000 en inglés). La cobertura es desigual.

### Sentidos discretos

WordNet asume que cada palabra polisémica tiene **N sentidos discretos**. Pero el significado es **continuo y contextual** — el sentido "exacto" de `bank` en "I went to the bank" depende del contexto y puede caer entre los sentidos definidos.

Esto es la motivación de **word embeddings** (word2vec, GloVe, contextual embeddings de BERT) que tratan el significado como vector continuo en lugar de membership discreto.

### Lentitud para nuevos términos

WordNet se actualiza lentamente. Términos surgidos post-2011 (cuando se detuvo el desarrollo principal) **no están**:

- `Bitcoin`, `cryptocurrency`, `selfie`, `tweet` (como verbo), `meme`, `troll` (como verbo de internet), `lol`, `app`, `vape`, etc.

Si tu corpus es contemporáneo, **WordNet va a tener gaps** importantes.

### No cubre slang, jerga técnica, idiomas no estándar

`obamacare`, `weaponize`, terminología médica específica como `SARS-CoV-2`, jerga programática (`grep`, `sudo`, `repository`) — todo eso falta o está mal categorizado.

### Está congelada (mostly)

Princeton terminó el desarrollo activo de WordNet en ~2012. Existen forks y extensiones, pero la base de Princeton es estática. Algunos proyectos sucesores:

- **Wikidata** (Wikimedia Foundation, 2012+): grafo de conocimiento más amplio.
- **BabelNet** (Sapienza Univ. Rome): combina WordNet + Wikipedia + Wikidata, multilingüe masivo.
- **ConceptNet** (MIT, 2007+): grafo de sentido común incluyendo relaciones más diversas que las de WordNet.

### Inflexibilidad para domain knowledge

WordNet asume **conocimiento general**. Para dominios especializados (medicina, derecho, biología, química) necesitas **ontologías específicas**:

- **MeSH** (Medical Subject Headings) y **UMLS** para biomedicina.
- **SNOMED CT** para terminología clínica.
- **WordNet Domains** (extensión).

Para texto clínico la combinación más común es: SNOMED CT (vocabulario clínico) + LOINC (laboratorio) + WordNet (vocabulario general). WordNet por sí solo no cubre patología clínica.

---

## Por qué importa hoy

Con la llegada de BERT (2018) y GPT-2/3/4, muchas tareas que usaban WordNet han migrado a representaciones distribuidas (embeddings). Pero WordNet **sigue siendo relevante**:

1. **Como ground truth para evaluación**: muchos benchmarks de WSD usan synsets de WordNet como labels.
2. **Como source de symbolic reasoning**: cuando necesitas explicabilidad o reasoning paso a paso, WordNet provee estructura simbólica que los embeddings no tienen.
3. **Combinado con embeddings**: muchos sistemas modernos usan embeddings + lookup de WordNet (por ejemplo Sense2vec, KnowBERT, ERNIE).
4. **En idiomas low-resource**: para idiomas sin Transformers preentrenados bien, WordNet (vía OMW) puede ser único recurso semántico disponible.

### Inspección directa desde NLTK

```python
from nltk.corpus import wordnet as wn

# Obtener todos los synsets que contienen "run"
synsets = wn.synsets("run")
print(f"'run' tiene {len(synsets)} synsets:")
for s in synsets[:5]:
    print(f"  {s.name()}: {s.definition()}")

# Explorar relaciones de un synset específico
ss = wn.synset("run.v.01")
print(f"\nDefinición: {ss.definition()}")
print(f"Synonyms: {[lemma.name() for lemma in ss.lemmas()]}")
print(f"Hypernyms: {[h.name() for h in ss.hypernyms()]}")
print(f"Hyponyms: {[h.name() for h in ss.hyponyms()][:5]}")
print(f"Antonyms: {[lemma.antonyms() for lemma in ss.lemmas()]}")
```

`run` tiene **~50 synsets** en WordNet, cubriendo desde "correr físicamente" hasta "ejecutar un programa" hasta "operar un negocio" hasta "candidatear políticamente".

WordNet fue parte fundamental del **giro semántico** del NLP en los 90s. Aunque hoy compite con embeddings y LLMs, sigue siendo el **mejor recurso simbólico** para hacer reasoning explícito sobre significado. En cualquier pipeline que combine reglas + estadística, WordNet es punto de partida obligado.

---

## Notas y enlaces

- **Clase asociada**: [Clase 16 - NLP clásico, NLTK, BoW, embeddings](/clases/clase-16).
- **Laboratorio asociado**: [Lab 16 - Pipeline NLP con NLTK/spaCy/NLLB/VADER](/laboratorios/lab-16).
- **Fundamento relacionado**: [Tokenización clásica](/fundamentos/tokenizacion-clasica).
- **Cita BibTeX**:

```bibtex
@article{miller1995wordnet,
  title={WordNet: a lexical database for English},
  author={Miller, George A},
  journal={Communications of the ACM},
  volume={38},
  number={11},
  pages={39--41},
  year={1995}
}
```
