# NLTK: The Natural Language Toolkit — Bird & Loper (2006)

**Autores:** Steven Bird (University of Melbourne) y Edward Loper (University of Pennsylvania).
**Publicación:** Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions, Sydney. Esta versión es la del paper "corto" de 4 páginas; existe también la versión 2002 más larga del workshop de ACL ("NLTK: The Natural Language Toolkit") y el libro 2009 *Natural Language Processing with Python*.
**PDF local:** `NLTK-Bird-Loper-2006.pdf`
**Conexión con el laboratorio:** El bloque 1 del Práctico 16 (celdas 5-31) **usa NLTK como herramienta principal** para tokenización, frecuencias, stop-words, stemming, lematización y manipulación de corpora. Este paper es **la justificación de diseño** de esa herramienta.

---

## 1. Contexto histórico

Para entender la importancia de NLTK hay que mirar el estado del arte en 2001-2006:

| Dimensión | Antes de NLTK | Lo que NLTK trajo |
|---|---|---|
| **Lenguajes** | NLP académico vivía en Perl, C++, Java y Tcl (ej. GATE). | Python como lengua común. |
| **Distribución** | Cada laboratorio tenía sus propios scripts y corpora privados. | Un único paquete con todo lo necesario para enseñar NLP. |
| **Pedagogía** | Los cursos universitarios construían tooling propio desde cero. | Plataforma única reutilizable: tareas, demos, proyectos. |
| **Corpora** | El acceso a Brown, Penn Treebank, WordNet, etc. requería gestionar múltiples APIs y formatos. | Una interfaz uniforme (`nltk.corpus.brown`, `nltk.corpus.treebank`, etc.). |
| **Algoritmos** | Implementaciones aisladas, sin marco común. | Jerarquía de interfaces (`TokenizerI`, `ParserI`, `TaggerI`) que permite intercambiar implementaciones. |

NLTK nace en 2001 acompañando el curso de Lingüística Computacional de UPenn, dictado por Steven Bird. Para 2006 ya había sido adoptado por al menos 13 universidades (entre ellas MIT, Edinburgh, Amsterdam, Pittsburgh, Macquarie, Melbourne).

**Decisión clave: Python.** Bird y Loper argumentan que Python ofrece:
- Curva de aprendizaje suave (no requiere expertise en tipos, memoria, builds).
- Sintaxis transparente que se parece al pseudocódigo de los libros de texto.
- Buen manejo de strings y unicode.
- Generadores (introducidos en Python 2.2, popularizados en 2.4) que permiten implementaciones interactivas y *lazy* de algoritmos.
- Librería estándar fuerte (Tkinter para GUI, módulos numéricos, etc.).

En 2006 este "Python para NLP" era una apuesta — el lenguaje aún no dominaba ML. Hoy parece obvio.

---

## 2. Contribución central

NLTK aporta tres cosas que en conjunto cambiaron cómo se enseña y prototipa NLP:

1. **Un suite de módulos minimal-coupling.** Una jerarquía plana donde cada módulo (tokenizer, tagger, parser, chunker, classifier, …) implementa una interfaz limpia y puede usarse independiente de los demás.
2. **Una arquitectura de "blackboard" sobre la clase `Token`.** En vez de un pipeline donde cada etapa descarta el input de la previa, los `Token`s acumulan propiedades (`TEXT`, `TAG`, `SUBTOKENS`, `SENSE`, etc.) de manera monotónica. Esto **anticipa** ideas que aparecerían más tarde en spaCy (`Doc.tensor` con anotaciones acumuladas) y en HuggingFace `Datasets` (columnas que se van agregando).
3. **Una colección integrada de corpora, datasets y demos GUI.** Antes había que descargar Brown desde una cinta, configurar Penn Treebank con una licencia, etc. NLTK los empaqueta en `nltk-data` con una API uniforme.

---

## 3. Arquitectura y diseño

### 3.1 La clase `Token` como tipo central

```python
>>> from nltk.token import *
>>> Token(TEXT="Hello World!")
<Hello World!>
>>> Token(TEXT="python", TAG="NN")
<python/NN>
>>> tok = Token(TEXT="Hello World!")
>>> WhitespaceTokenizer().tokenize(tok)
>>> print(tok['SUBTOKENS'])
[<Hello>, <World!>]
```

Cada `Token` es un *mapping parcial* de nombres de propiedad a valores. Esta abstracción es **deliberadamente más laxa que un struct**: cualquier tarea puede agregar propiedades sin que las demás se rompan.

Comparación con arquitecturas alternativas:
- **Pipeline (común en GATE, OpenNLP):** Tokenizer → Tagger → Parser, cada etapa toma el output anterior como string/lista. Perdés información del paso previo.
- **Blackboard (NLTK):** Tokenizer agrega `SUBTOKENS`. Tagger agrega `TAG` a cada subtoken. Parser agrega `TREE`. Todo coexiste. Permite que un componente posterior reconsidere decisiones tempranas.

### 3.2 Módulos principales

| Módulo | Propósito |
|---|---|
| `nltk.token`, `nltk.tokenizer` | Token data structure, WhitespaceTokenizer, RegexpTokenizer |
| `nltk.corpus` | Brown, Treebank, Gutenberg, WordNet, stopwords, Names, Genesis, Inaugural, … |
| `nltk.tagger` | POS taggers (default, lookup, regexp, n-gram, Brill) |
| `nltk.parser` | Chart parsers, recursive descent, shift-reduce, probabilistic, chunk parsers |
| `nltk.probability` | FreqDist, ConditionalFreqDist, ProbDistI, suavizado |
| `nltk.stemmer` | Porter, Lancaster, Snowball |
| `nltk.cfg` | Gramáticas libres de contexto y PCFG |
| `nltk.featurestruct` | Estructuras de rasgos para gramáticas unification-based |
| `nltk.sense` | Word-sense disambiguation |
| `nltk.draw` | Visualizadores GUI interactivos (chart parsing, árboles, FSA) |
| `nltk.eval` | Métricas estándar (precision, recall, accuracy, edit distance) |

Las interfaces se distinguen con un sufijo `I` mayúsculo: `ParserI`, `TokenizerI`, `TaggerI`. Cada interfaz tiene un *action method* (`parse`, `tokenize`, `tag`) y opcionalmente *extended action methods* (`parse_n` que devuelve los top-N parsings, `xtokenize` que devuelve un iterador en lugar de una lista).

### 3.3 Corpora distribuidos (tabla 1 del paper)

NLTK 1.4 incluía ya 15 corpora preempaquetados:

| Corpus | Tamaño | Uso típico |
|---|---|---|
| Brown Corpus | 1.15M tokens, 15 géneros, taggeado | Entrenar taggers, clasificación de texto |
| Penn Treebank (sample) | 40k tokens, taggeado + parseado | Desarrollar parsers |
| Project Gutenberg (selection) | 1.7M tokens, 14 textos | Modelado de lenguaje, clasificación |
| CoNLL-2000 Chunking | 270k tokens, chunkeado | Entrenar chunkers |
| WordNet 1.7 | 180k palabras en red semántica | WSD, NL understanding |
| Stopwords Corpus | 2400 stopwords en 11 lenguajes | Information retrieval |
| Names Corpus | 8k nombres (m/f) | Clasificación |
| Roget's Thesaurus | 200k tokens | WSD |
| SEMCOR, SENSEVAL-2 | 880k / 600k tokens, POS + sense | WSD |
| NIST IEER (selection) | 63k tokens, NER markup | Entrenar reconocedores de entidades |
| PP Attachment Corpus | 28k preposicionales | Parser development |
| 20 Newsgroups (selection) | 4000 posts | Text classification |
| Levin Verb Index | 3k verbos | Parser development |
| Wordlist Corpus | 960k palabras, 20k afijos | Spell checking |

Esto es importante: NLTK no es "una librería para hacer NLP" — es una librería **+ una colección de datos + tutoriales + demos** todo empaquetado para que un estudiante pueda ejecutar `nltk.download()` y tener todo lo necesario para una clase.

---

## 4. Experimentos clave y validación

El paper de 2006 es un *system paper*, no un paper de resultados experimentales. La "validación" es la adopción institucional. Tabla 2 del paper lista 13 universidades que ya usaban NLTK en cursos:

- Graz (Austria), Macquarie (Australia), MIT (USA), UNAM (México), Ohio State, Amsterdam, Colorado, Edinburgh, Magdeburg, Malta, Melbourne, UPenn, Pittsburgh, Simon Fraser (Canada).

También documenta contribuciones de terceros que ya estaban integradas:
- **Brill tagger** (Chris Maloof) — transformación basada en reglas con aprendizaje supervisado.
- **HMM tagger** (Trevor Cohn, Phil Blunsom).
- **Parser GPSG con rasgos** (Rob Speer, Bob Berwick).
- **Analizador morfológico FST** (Carl de Marcken, Beracah Yankama, Bob Berwick).
- **Clasificadores decision list y decision tree** (Trevor Cohn).
- **Discourse Representation Theory** (Edward Ivanovic).

---

## 5. Limitaciones y críticas

Aunque el paper no las discute explícitamente, las limitaciones de NLTK que se han hecho evidentes con los años:

1. **Performance.** NLTK está optimizado para legibilidad pedagógica, no velocidad. Para producción de gran escala, spaCy (4-100× más rápido en parsing y NER) o Stanza son mejores opciones.
2. **Modelos preentrenados pobres comparados con DL moderno.** Los taggers y parsers de NLTK son estadísticos clásicos (Brill, HMM, n-gram); no compiten con Transformer-based en accuracy.
3. **APIs inconsistentes entre módulos.** El precio de los "minimally interdependent modules" es que `nltk.tag`, `nltk.parse`, `nltk.classify` tienen patrones de uso ligeramente distintos.
4. **No multilingüe nativo.** Los corpora y modelos preempaquetados son principalmente en inglés; soporte para otros idiomas depende de stemmers y stopwords listados pero pocos taggers/parsers preentrenados.
5. **WordNet-centric.** Mucho del módulo `sense` y de los recursos semánticos asumen WordNet, que existe en buena cobertura solo para inglés.
6. **Sin embeddings nativos.** En 2006 esto no era una crítica, pero para 2014+ NLTK se quedó atrás respecto a gensim (word2vec), fastText, BERT, etc.

---

## 6. Impacto y legado

NLTK definió **la lengua franca pedagógica del NLP por casi 20 años**. Algunos efectos medibles:

- El libro *Natural Language Processing with Python* (Bird, Klein & Loper, 2009) es texto obligatorio en cursos introductorios de NLP en todo el mundo (~250k descargas/año en su versión gratuita).
- Empujó la adopción de Python como lengua franca del NLP (precediendo y allanando el camino para scikit-learn 2007, gensim 2009, spaCy 2015, Transformers 2018).
- La arquitectura blackboard sobre `Token` se ve reflejada en `spacy.tokens.Doc` (atributos extensibles) y en `transformers.tokenization_utils.BatchEncoding`.
- Las funciones que conocés del práctico (`concordance`, `dispersion_plot`, `FreqDist.plot`, `sent_tokenize`, `word_tokenize`, `PorterStemmer`, `stopwords.words('spanish')`) se mantuvieron prácticamente iguales en versiones posteriores — uno de los proyectos open-source con compatibilidad hacia atrás más estables en NLP.

Citas: a mayo de 2026, Google Scholar reporta **~6700 citas para el paper de 2006** y **~12k citas combinadas para el libro de 2009 y el paper de 2002**. Es probablemente el paper más citado en NLP pedagógico jamás escrito.

---

## 7. Conexión directa con el Práctico 16

Cada cosa que el bloque 1 (celdas 5-31) del lab hace, está justificada por este paper:

| Celda del lab | Concepto de NLTK | Sección del paper |
|---|---|---|
| 8 | `nltk.download('gutenberg')`, `nltk.download('genesis')`, etc. | §3 *Installing NLTK*, tabla 1 *Corpora* |
| 11-13 | `Text.concordance()` sobre `text1` (Moby Dick) | §2.2 *corpus module*; corpus Gutenberg en tabla 1 |
| 15 | `text4.dispersion_plot(...)` | §2.4 *Documentation*; demos GUI |
| 17-18 | `FreqDist` y `fdist1.plot(50)` | §2.1 *probability module* |
| 20 | Ley de Heaps sobre tokens de Moby Dick | uso de `text.tokens` (Token interface) |
| 22-25 | `sent_tokenize`, `word_tokenize`, `TweetTokenizer` | §2.1 *tokenizer module*, interface `TokenizerI` |
| 27-29 | `stopwords.words('spanish')` | tabla 1 *Stopwords Corpus, 2400 stopwords for 11 lgs* |
| 31 | `PorterStemmer`, `LancasterStemmer`, `SnowballStemmer`, `WordNetLemmatizer` | §2.3 módulos `stemmer.porter` y `wordnet` |

La frase del paper más relevante para el lab: *"NLTK supports assignments of varying difficulty and scope. In the simplest assignments, students experiment with existing components to perform a wide variety of NLP tasks."* — Eso es **exactamente** lo que las celdas 11-31 del Práctico 16 están haciendo: experimentar con componentes existentes.

---

## 8. Lecturas relacionadas

- Loper & Bird (2002), *NLTK: The Natural Language Toolkit*, ACL Workshop on Effective Tools and Methodologies for Teaching NLP — la versión "0" del paper.
- Bird, Klein & Loper (2009), *Natural Language Processing with Python*, O'Reilly. El libro de referencia (gratuito en nltk.org/book).
- Loper (2004), *NLTK: Building a Pedagogical Toolkit in Python*, PyCon DC.

Para spaCy (que aparece en el bloque 2 del lab, celdas 32-42) — el contraste con NLTK lo veremos en ese bloque: spaCy elige el camino opuesto (pipeline industrial sobre blackboard pedagógico).
