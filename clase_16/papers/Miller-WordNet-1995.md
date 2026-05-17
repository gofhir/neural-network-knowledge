# Introduction to WordNet: An On-line Lexical Database — Miller et al. (1995)

**Autores:** George A. Miller, Richard Beckwith, Christiane Fellbaum, Derek Gross, Katherine Miller (Cognitive Science Laboratory, Princeton University).
**Publicación:** *International Journal of Lexicography*, Volume 3, Number 4, Special Issue (Diciembre 1990), pp. 235-244. **La versión más citada y consultada** es la revisión publicada en 1993-1995 como parte de los "Five Papers on WordNet" disponibles en princeton.edu/wordnet. El "1995" comúnmente atribuido al paper refiere a la publicación de Miller (1995) *WordNet: A Lexical Database for English* en *Communications of the ACM* 38(11), pp. 39-41.
**PDF local:** `Miller-WordNet-1995.pdf` (compilation de los 5 papers, 86 páginas, incluye este introductorio + capítulos sobre nouns, verbs, adjectives, design).
**Conexión con el laboratorio:** El bloque 1 del Práctico 16 (celda 31) usa `WordNetLemmatizer`, que **depende directamente de la base de datos WordNet descrita en este paper**. Cuando ejecutas `nltk.download('wordnet')` y `nltk.download('omw-1.4')` en la celda 8, estás descargando precisamente la base lexical que este equipo de Princeton construyó. WordNet es **EL recurso léxico estándar del NLP en inglés desde 1990 hasta hoy**.

---

## 1. Contexto histórico

A finales de los 70 y comienzos de los 80, el procesamiento de lenguaje natural carecía de **un recurso léxico digital comprehensivo** para inglés. Los diccionarios existían en papel y en algunas versiones digitalizadas, pero:

- **Estaban organizados alfabéticamente**, no semánticamente. Buscar "all synonyms of run" requería revisión manual extensiva.
- **No tenían estructura formal**: definiciones en prosa, no enlaces explícitos entre conceptos relacionados.
- **No reflejaban teoría psicolingüística**: la organización seguía conveniencias de imprenta, no la estructura cognitiva del lexicón mental humano.

Para tareas como **desambiguación de palabras**, **expansión de queries en IR**, **traducción automática**, o **comprensión de texto**, los investigadores necesitaban un lexicón estructurado donde pudieran consultar programáticamente "¿qué palabras son sinónimos de X?", "¿cuáles son los hipónimos de Y?", "¿qué es Z una parte de?".

### El proyecto Princeton

En **1985**, un grupo de psicólogos y lingüistas en Princeton (encabezado por George Miller, padre de la psicología cognitiva) decidió construir un lexicón siguiendo principios **psicolingüísticos** — no por conveniencia editorial, sino para reflejar cómo el lexicón humano (mental lexicon) está organizado.

Miller era figura mayor:
- Su paper de 1956 *"The Magical Number Seven, Plus or Minus Two"* es uno de los más citados de toda la psicología.
- Cofundador de Cognitive Science.
- Pionero de la psicolinguística junto a Noam Chomsky.

La motivación que el paper articula (página 1):
> *"Standard alphabetical procedures for organizing lexical information put together words that are spelled alike and scatter words with similar or related meanings haphazardly through the list."*

Es decir: el diccionario alfabético es **anti-cognitivo**. WordNet rompe con esto.

### Investigaciones psicolingüísticas que motivaron el diseño

El paper cita estudios empíricos clave:

1. **Fillenbaum & Jones (1965)**: probaron **word association** — pedir a sujetos la primera palabra que les viene tras un estímulo. Resultado: **el 79% de las asociaciones con sustantivos son sustantivos**, 65% con adjetivos son adjetivos, 43% con verbos son verbos. **El lexicón mental está particionado por categoría sintáctica**.

2. **Collins & Quillian (1969)**: midieron **tiempos de reacción** para verificar afirmaciones como "Un canario puede cantar" vs "Un canario puede volar" vs "Un canario tiene piel". Resultado: **más tiempo a medida que la propiedad pertenece a un nivel jerárquico más alto** (volar = pájaro, tener piel = animal). Sugiere que **la información se almacena jerárquicamente**, con propiedades en el nivel más general y herencia a sub-conceptos.

3. **Anomic aphasia**: pacientes con lesiones en hemisferio izquierdo pierden la capacidad de **nombrar objetos** pero conservan otras facultades. Esto sugiere que **los sustantivos están organizados en un subsistema léxico separado** del de verbos.

WordNet fue construida con estos hallazgos como guía: **categorías separadas para nouns/verbs/adjectives/adverbs**, **estructura jerárquica para nouns** (herencia), **estructura diferente para verbs** (relaciones de entailment), **estructura distinta para adjectives** (espacio N-dimensional).

---

## 2. Contribución central

WordNet aporta **tres cosas profundamente interrelacionadas**:

1. **El concepto de synset (synonym set)** como unidad fundamental del lexicón.
2. **Una red de relaciones semánticas tipadas** que enlaza synsets entre sí.
3. **Una implementación operacional** distribuida libremente, con APIs para múltiples lenguajes y datasets en docenas de idiomas (vía Open Multilingual WordNet).

A 1993 (fecha de la versión documentada en el paper), WordNet contenía:
- **51,500 palabras simples**
- **44,100 collocations** (frases-palabra como *swimming pool*)
- **70,100 synsets** (conceptos)

Para 2026, la versión actual (WordNet 3.1 + extensions vía OMW-1.4) contiene **>117,000 synsets** y se extiende a >200 idiomas vía proyectos derivados.

---

## 3. Conceptos fundamentales

### 3.1 La matriz léxica: forms × meanings

Miller introduce la **lexical matrix** como abstracción fundamental (página 4):

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
- Una **celda E[i,j]** indica que el form Fj puede expresar el meaning Mi.

**Dos fenómenos básicos** se ven en esta matriz:

| Fenómeno | Definición | Visualización |
|---|---|---|
| **Sinonimia** | Dos formas (F1, F2) pueden expresar el mismo meaning (M1) | Dos celdas en la **misma fila** |
| **Polisemia** | Un form (F2) puede expresar múltiples meanings | Dos celdas en la **misma columna** |

WordNet representa cada meaning M con un **synset** = `{F1, F2, ...}` (lista de formas sinónimas que lo expresan). Las relaciones semánticas son **flechas entre synsets**, no entre formas individuales.

### 3.2 Synsets — el corazón de WordNet

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

### 3.3 Relaciones semánticas

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
- `{maple}` es hyponym de `{tree}` (maple es un tipo de tree)
- `{tree}` es hyponym de `{plant}` (tree es un tipo de plant)

Es **transitiva**: si X is_a Y y Y is_a Z, entonces X is_a Z.
Es **asimétrica**: tree no es maple.

Tipicamente cada noun tiene **exactamente un hypernym**, lo cual genera una **estructura de árbol** ("inheritance system"). Esta es la base de la organización jerárquica de los sustantivos en WordNet.

**Profundidad típica**: máximo ~12 niveles desde la raíz (`entity`) hasta los terminales. Por ejemplo:
```
canary → finch → passerine → bird → vertebrate → animal → organism → entity
```
8 niveles.

#### Meronymy / Holonymy (HASA / PART-OF relation)
- `wheel` es **meronym** de `car` (wheel es parte de car).
- `car` es **holonym** de `wheel`.

Es transitiva (con qualificaciones — un dedo es parte de mano que es parte de brazo, pero "un dedo es parte de brazo" es discutible).

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

### 3.4 Organización particional por POS

WordNet **NO mezcla categorías sintácticas**. Cuatro subsistemas independientes:

| POS | Tamaño 1993 | Tamaño actual (~2024) | Estructura organizativa |
|---|---|---|---|
| **Nouns** | ~57,000 forms / 48,800 synsets | ~117,000 synsets | Jerarquía profunda (hypernymy/hyponymy/meronymy) |
| **Verbs** | ~21,000 forms | ~25,000 synsets | Jerarquía + entailment + causación |
| **Adjectives** | ~19,500 forms | ~21,000 synsets | Espacio N-dimensional con antonymy + similar_to |
| **Adverbs** | ~3,500 forms | ~4,500 synsets | Mayormente derivados de adjetivos vía pertainymy |

**Por qué la separación**: estudios psicolingüísticos sugieren que estas categorías se almacenan **en sistemas neurales distintos**. La aphasia anómica selectiva (pierde solo nouns) confirma esto.

### 3.5 Morfología

WordNet **no almacena formas inflexionales** (run, runs, running, ran). Esto crearía explosión combinatoria.

En su lugar, **el interface** (no la base) tiene un **morphological analyzer** que reduce formas inflexionales a su lemma antes del lookup:
- `running` → `run`
- `mice` → `mouse`
- `better` → `good` (irregular)

**Esto es lo que NLTK usa cuando llamas `lemmatizer.lemmatize(word, pos='v')`**.

---

## 4. Implementación

WordNet se distribuye como:
- **Base de datos** (archivos planos `.dict`, `.idx`, `.exc`).
- **APIs** en C, Java, Python (vía NLTK).
- **Browser** (interface web/desktop para explorar synsets visualmente).

El paper original (sección 3.5) menciona que la base de datos crece continuamente. Eso ha sido cierto: WordNet ha tenido **8+ versiones mayores** desde 1991 (1.0, 1.5, 1.6, 1.7, 1.7.1, 2.0, 2.1, 3.0, 3.1) hasta 2011, después del cual el proyecto Princeton ha sido relevado por extensiones académicas (Open Multilingual WordNet, BabelNet, etc.).

---

## 5. Limitaciones reconocidas

El paper de 1995 no critica abiertamente sus propias limitaciones, pero el lector moderno las identifica:

### 5.1 Solo inglés

WordNet original es exclusivamente inglés estadounidense. Para otros idiomas:
- **EuroWordNet** (1998-2002, 8 idiomas europeos).
- **MultiWordNet** (italiano, español).
- **Open Multilingual WordNet** (OMW): meta-proyecto que enlaza WordNets en 30+ idiomas.
- **OMW-1.4** (lo que descargas en NLTK con `nltk.download('omw-1.4')`) es la versión 2017+ de OMW. Contiene WordNets de español, francés, italiano, japonés, alemán, portugués, etc.

**Pero el WordNet español de OMW es mucho más pequeño que el inglés** (~30,000 synsets en español vs >117,000 en inglés). La cobertura es desigual.

### 5.2 Sentidos discretos

WordNet asume que cada palabra polisémica tiene **N sentidos discretos**. Pero el significado es **continuo y contextual** — el sentido "exacto" de `bank` en "I went to the bank" depende del contexto y puede caer entre los sentidos definidos.

Esto es la motivación de **word embeddings** (word2vec, GloVe, contextual embeddings de BERT) que tratan el significado como vector continuo en lugar de membership discreto.

### 5.3 Lentitud para nuevos términos

WordNet se actualiza lentamente. Términos surgidos post-2011 (cuando se detuvo el desarrollo principal) **no están**:
- `Bitcoin`, `cryptocurrency`, `selfie`, `tweet` (como verbo), `meme`, `troll` (como verbo de internet), `lol`, `app`, `vape`, etc.

Si tu corpus es contemporáneo, **WordNet va a tener gaps** importantes.

### 5.4 No cubre slang, jerga técnica, idiomas no estándar

`obamacare`, `weaponize`, terminología médica específica como `SARS-CoV-2`, jerga programática (`grep`, `sudo`, `repository`) — todo eso falta o está mal categorizado.

### 5.5 Está congelada (mostly)

Princeton terminó el desarrollo activo de WordNet en ~2012. Existen forks y extensiones, pero la base de Princeton es estática. Algunos proyectos sucesores:
- **Wikidata** (Wikimedia Foundation, 2012+): grafo de conocimiento más amplio.
- **BabelNet** (Sapienza Univ. Rome): combina WordNet + Wikipedia + Wikidata, multilingüe masivo.
- **ConceptNet** (MIT, 2007+): grafo de sentido común incluyendo relaciones más diversas que las de WordNet.

### 5.6 Inflexibilidad para domain knowledge

WordNet asume **conocimiento general**. Para dominios especializados (medicina, derecho, biología, química) necesitas **ontologías específicas**:
- **MeSH** (Medical Subject Headings) y **UMLS** para biomedicina.
- **SNOMED CT** para terminología clínica.
- **WordNet Domains** (extensión).

**Para tu trabajo en FHIR-MDM** la combinación más común es: SNOMED CT (vocabulario clínico) + LOINC (laboratorio) + WordNet (vocabulario general). WordNet por sí solo no cubre patología clínica.

---

## 6. Impacto y legado

WordNet es probablemente **el recurso léxico más influyente jamás construido**. Algunas métricas:

- **Citas**: a mayo de 2026, los papers fundacionales de WordNet acumulan **>50,000 citas** sumadas (Miller 1995, Fellbaum 1998, etc.).
- **Adopción**: incluido por defecto en NLTK, spaCy, Stanford CoreNLP, AllenNLP. **El estándar de facto** en NLP académico.
- **Derivados**: WordNet ha inspirado **decenas** de proyectos similares: ConceptNet (2007), BabelNet (2010), Open Multilingual WordNet (2014), etc.
- **Premios**: George Miller recibió numerosos honores; el proyecto WordNet fue Reconocido con el ACL Lifetime Achievement Award.

### Aplicaciones canónicas

- **Lemmatization**: el `WordNetLemmatizer` de NLTK que usas en la celda 31.
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

### Era post-Transformer

Con la llegada de BERT (2018) y GPT-2/3/4, muchas tareas que usaban WordNet han migrado a representaciones distribuidas (embeddings). Pero WordNet **sigue siendo relevante**:

1. **Como ground truth para evaluación**: muchos benchmarks de WSD usan synsets de WordNet como labels.
2. **Como source de symbolic reasoning**: cuando necesitas explicabilidad o reasoning paso a paso, WordNet provee estructura simbólica que los embeddings no tienen.
3. **Combined con embeddings**: muchos sistemas modernos usan embeddings + lookup de WordNet (e.g., Sense2vec, KnowBERT, ERNIE).
4. **En idiomas low-resource**: para idiomas sin Transformers pretrenados bien, WordNet (vía OMW) puede ser único recurso semántico disponible.

---

## 7. Conexión directa con el Práctico 16

| Celda del lab | Concepto del paper |
|---|---|
| 8 | `nltk.download('wordnet')` — descarga la base WordNet 3.0 (~32 MB). |
| 8 | `nltk.download('omw-1.4')` — descarga la **extension multilingüe** Open Multilingual WordNet. Required por NLTK 3.6.5+ para que `WordNetLemmatizer` funcione (cambio interno de NLTK). |
| 31 | `from nltk.stem import WordNetLemmatizer` — importa el lemmatizer que consulta WordNet. |
| 31 | `lemmatizer = WordNetLemmatizer()` — instancia el lemmatizer. **No requiere parámetros** porque usa los archivos descargados. |
| 31 | `lemmatizer.lemmatize(word, pos='v')` — para cada palabra, busca su lemma asumiendo POS=verbo. **Esto consulta directamente los archivos de WordNet**. |
| 31 | Output de WordNet en la frase del lab: `Artificial intelligence is intelligence demonstrate by machine . Leading AI textbook define the field a the study of intelligent agent : any device that perceives it environment and take action that maximize it chance of successfully achieve it goal .` |
| | • `machines → machine` — plural reducido a singular vía morphology lookup. |
| | • `textbooks → textbook` — idem. |
| | • `agents → agent`, `actions → action`, `goals → goal` — todos plurales. |
| | • `demonstrated → demonstrate` — verbo pasado a presente vía `pos='v'`. |
| | • `defines` no parece — pero `define` está en el lemma form. |
| | • `is` NO se convierte a `be` porque el `pos='a'` (intentado primero por la función `lemmatize` del lab) no hace nada con `is` y los siguientes intentos no llegan a la respuesta correcta. **Esta es la limitación de la función `lemmatize` del lab que discutimos en la celda 31.** |

**Para inspeccionar WordNet directamente** (interesante):

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

Vas a ver que `run` tiene **~50 synsets** en WordNet, cubriendo desde "correr físicamente" hasta "ejecutar un programa" hasta "operar un negocio" hasta "candidatear políticamente".

**Para tu trabajo en FHIR-MDM**, WordNet español vía OMW-1.4:

```python
# Sinónimos de "doctor" en español
synsets = wn.synsets("doctor", lang="spa")
for ss in synsets:
    print(f"{ss.name()}: {ss.lemma_names('spa')}")
```

Vas a ver que el WordNet español es **mucho más pobre** que el inglés. Para terminología clínica seria necesitarás SNOMED CT, no WordNet.

---

## 8. Lecturas relacionadas

**Foundational:**
- Miller (1995), *WordNet: A Lexical Database for English*, Communications of the ACM 38(11) — el paper short más citado.
- Fellbaum (1998), *WordNet: An Electronic Lexical Database*, MIT Press — el libro de referencia (~50,000 citas).
- Miller, Beckwith, Fellbaum, Gross & Miller (1990), *Introduction to WordNet: An On-line Lexical Database*, International Journal of Lexicography 3(4) — el paper original al que este texto se refiere.

**Trabajos relacionados pre-WordNet:**
- Quillian (1968), *Semantic Memory* — primer modelo computacional de memoria semántica jerárquica.
- Collins & Quillian (1969), *Retrieval Time from Semantic Memory* — el experimento de tiempos de reacción que validó la hipótesis jerárquica.

**Extensions y sucesores:**
- Vossen (1998), *EuroWordNet* — extensión a 8 idiomas europeos.
- Bond & Foster (2013), *Linking and Extending an Open Multilingual WordNet* — Open Multilingual WordNet (OMW), que descargas con `omw-1.4`.
- Navigli & Ponzetto (2010), *BabelNet* — combinación WordNet + Wikipedia + Wikidata en 200+ idiomas.
- Speer, Chin & Havasi (2017), *ConceptNet 5.5* — alternativa con relaciones más diversas (motivation, location, function, etc.).

**Aplicaciones canónicas de WordNet:**
- Resnik (1995), *Using Information Content to Evaluate Semantic Similarity in a Taxonomy* — la primera medida cuantitativa de similitud léxica basada en WordNet.
- Banerjee & Pedersen (2002), *An Adapted Lesk Algorithm for Word Sense Disambiguation Using WordNet* — WSD vía solapamiento de glosses.
- Hirst & St-Onge (1998), *Lexical Chains as Representations of Context* — usa WordNet para detectar coherencia temática en texto.

**Para entender por qué los Transformers dejaron WordNet en segundo plano:**
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space* (word2vec) — el primer modelo neural que aprendió relaciones semánticas SIN WordNet, solo desde texto crudo.
- Devlin et al. (2018), *BERT* — embeddings contextuales que codifican sentido sin necesidad de synsets discretos.

**Para texto clínico (relevante a tu trabajo):**
- Bodenreider (2004), *The Unified Medical Language System (UMLS): Integrating Biomedical Terminology* — UMLS es el equivalente de WordNet para medicina, integra SNOMED CT, LOINC, MeSH, etc.
- Pradhan et al. (2014), *SemEval-2014 Task 7: Analysis of Clinical Text* — benchmarks para NLP clínico, comparable a SemEval para inglés general.

WordNet fue parte fundamental del **giro semántico** del NLP en los 90s. Aunque hoy compitiendo con embeddings y LLMs, sigue siendo el **mejor recurso simbólico** para hacer reasoning explícito sobre significado. En cualquier pipeline que combine reglas + estadística, WordNet es punto de partida obligado.
