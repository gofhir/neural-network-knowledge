---
title: "Twitter POS - Part-of-Speech Tagging for Twitter"
weight: 165
math: true
---

{{< paper-card
    title="Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments"
    authors="Gimpel, Schneider, O'Connor, Das, Mills, Eisenstein, Heilman, Yogatama, Flanigan, Smith"
    year="2011"
    venue="ACL 2011 (short)"
    pdf="/papers/twitter-pos-gimpel-2011.pdf" >}}
Construye el primer POS tagger especializado para Twitter: un **tagset de 25 etiquetas** que incluye hashtags, at-mentions, URLs, emoticons y discourse markers; un **corpus anotado de 1,827 tweets** (26k tokens, κ = 0.914); y un **CRF con features Twitter-específicos** que alcanza **89.37%** de accuracy (vs. 85.85% del Stanford tagger reentrenado). El tokenizer asociado se portó luego a NLTK como `TweetTokenizer`, herramienta de facto para tokenizar texto de redes sociales.
{{< /paper-card >}}

---

## Contexto

A inicios de los 2010, Twitter se volvió fuente predilecta para estudios de opinión, política y mercados, pero el pipeline NLP clásico **fallaba en tweets**: Finin et al. (2010) mostró que los taggers POS entrenados sobre Wall Street Journal caían ~25% de accuracy al aplicarse a Twitter. Las razones son específicas y todas se relacionan con tokenización + POS:

| Característica de Twitter | Por qué rompe el pipeline NLP estándar |
|---|---|
| **Límite de 140 caracteres** | Fuerza abreviaciones, contracciones no estándar |
| **Ortografía no convencional** | `nite`, `2nite`, `cooool`, `lmaooo` — el lexicón POS no las cubre |
| **Hashtags `#NLP`** | El tokenizer estándar separa `#` de la palabra |
| **At-mentions `@user`** | Idem; pero además son siempre proper nouns en función |
| **URLs `http://t.co/xyz`** | El tokenizer estándar los explota en `["http", ":", "/", "/", ...]` |
| **Emoticons `:-)` `<3` `>:(`** | Los signos se tokenizan como puntuación, destruyendo el emoticón |
| **Capitalización inconsistente** | "I", "i", "I'M", "im" — la heurística "mayúsculas = proper noun" falla |
| **Retweet markers (`RT @user:`, `~`)** | Construcciones discursivas tratadas como signos sueltos |

El equipo de Noah Smith en CMU (con Brendan O'Connor como figura clave del esfuerzo Twitter) decidió construir desde cero un POS tagger especializado para Twitter. Este paper documenta el resultado.

---

## Ideas principales

El paper aporta **tres recursos en conjunto** liberados como recursos abiertos:

1. **Un POS tagset de 25 etiquetas** diseñado para Twitter (17 refinamientos del Universal POS Tagset de Petrov, Das & McDonald 2011 + 8 Twitter-específicas). Cada etiqueta es un **carácter ASCII único**.
2. **Un corpus anotado de 1,827 tweets (26,436 tokens)** con POS gold-standard. 17 anotadores, ~200 person-hours, 2 meses. Inter-annotator agreement: **92.2%** (Cohen κ = 0.914).
3. **Un POS tagger CRF** entrenado sobre ese corpus, con accuracy **89.37%** en test (vs. 85.85% para Stanford POS Tagger reentrenado en el mismo corpus). Reducción relativa de error: **25%**.

Junto al tagger, liberaron un **tokenizer especializado** (la base del actual `nltk.tokenize.casual.TweetTokenizer`).

### El tagset Twitter-específico

#### Nominales (refinamiento de NN)

| Tag | Descripción | Ejemplos | % en corpus |
|---|---|---|---|
| `N` | Common noun | books, someone | 13.7% |
| `O` | Pronoun (no posesivo) | it, you, u, meeee | 6.8% |
| `S` | Nominal + posesivo | books', someone's | 0.1% |
| `^` | Proper noun | lebron, usa, iPad | 6.4% |
| `Z` | Proper noun + posesivo | America's | 0.2% |
| `L` | Nominal + verbal | he's, book'll, iono (= I don't know) | 1.6% |
| `M` | Proper noun + verbal | Mark'll | < 0.05% |

**Decisión clave:** los autores **NO separan contracciones**. El Penn Treebank tokeniza `he's` como `["he", "'s"]` con tags `[PRP, VBZ]`. En Twitter las contracciones son masivas y muchas no son separables limpiamente (`iono` para "I don't know"), por lo que crearon **tags compuestos** (`L`, `M`). Es una decisión pragmática: simplificar tokenización a costa de tener más tags.

#### Open-class words (refinamiento)

| Tag | Descripción | Ejemplos | % |
|---|---|---|---|
| `V` | Verbo (incl. copula, aux, MD) | might, gonna, ought, couldn't, is, eats | 15.1% |
| `A` | Adjetivo | good, fav, lil | 5.1% |
| `R` | Adverbio | 2 (= "too") | 4.6% |
| `!` | Interjección | lol, haha, FTW, yea, right | 2.6% |

Notar: `2` puede ser numeral O adverbio (`2` = "too" en Twitter slang). El contexto determina.

#### Closed-class words (refinamiento)

| Tag | Descripción | Ejemplos | % |
|---|---|---|---|
| `D` | Determiner | the, teh, its, it's | 6.5% |
| `P` | Pre/postposición o conjunción subordinante | while, to, for, `2`, `4` | 8.7% |
| `&` | Conjunción coordinante | and, n, &, +, BUT | 1.7% |
| `T` | Verb particle | out, off, Up | 0.6% |
| `X` | Existential there, predeterminers | both | 0.1% |
| `Y` | X + verbal | there's, all's | < 0.05% |

#### Las cuatro Twitter-específicas (la contribución más original)

| Tag | Descripción | Ejemplos | % |
|---|---|---|---|
| `#` | Hashtag (categoría del tweet) | `#acl` | 1.0% |
| `@` | At-mention (recipiente) | `@BarackObama` | 4.9% |
| `~` | Discourse marker | `RT`, `:` en RT, `≪` separador | 3.4% |
| `U` | URL o email | `http://bit.ly/xyz` | 1.6% |
| `E` | Emoticon | `:-)`, `:b`, `(:`, `<3`, `o_O` | 1.0% |

**Suman ~12% de todos los tokens en tweets.** Ningún tagset previo cubría estos casos.

**Nota sutil sobre hashtags:** `#NLP` puede ser (i) categoría del tweet (tag `#`) o (ii) palabra usada como palabra ("Is #qadaffi going down?" → proper noun). Los autores **eligen contextualizar**: 35% de los hashtags reciben tags distintos a `#` (14% proper noun, 9% common noun, 5% multi-word, 3% verb, 4% otros). Las at-mentions, en cambio, **siempre son proper nouns** semánticamente.

#### Misceláneas

| Tag | Descripción | % |
|---|---|---|
| `$` | Numeral (incluye horas como `9:30`) | 1.5% |
| `,` | Puntuación (`!!!`, `....`, `?!?`) | 11.6% |
| `G` | Foreign words, abreviaciones multi-word (`ily` = I love you), garbage | 1.1% |

La categoría `G` es el "cajón de sastre" — incluye errores de tokenización, símbolos extraños, partes de palabras truncadas. Tiene la accuracy más baja del tagger (26%) por su heterogeneidad.

### El tokenizer especializado

Modificación del **TweetMotif tokenizer** de O'Connor et al. (2010b). Reglas clave:

1. **NO separar contracciones**: `he's`, `don't`, `iono` → un solo token. (Penn separaría `he's` → `["he", "'s"]`.)
2. **NO separar posesivos pegados al nombre**: `Mark's` → un solo token.
3. **Preservar emoticons** como tokens unitarios: `:-)`, `:-P`, `<3`, `o_O`. Requiere reglas regex específicas porque los caracteres `:`, `-`, `)` son normalmente puntuación.
4. **Preservar hashtags y at-mentions** completos: `#machinelearning`, `@user`.
5. **Preservar URLs**: detectar `http://`, `https://`, `www.`, `.com/.net/...` y mantenerlos como tokens unitarios.
6. **Reconocer Twitter discourse markers**: `RT`, `:` en construcción retweet, `≪`.
7. **Manejar caracteres unicode** correctamente (emojis, no-ASCII).

La implementación original está en `github.com/brendano/tweetmotif` y fue luego portada a NLTK como `nltk.tokenize.casual.TweetTokenizer`.

**Por qué tokenizar bien es prerrequisito de POS-taggear bien:** si tu tokenizer **destruye un emoticón** convirtiéndolo en `[":", "-", ")"]`, ningún POS tagger podrá recuperar la información de que ese conjunto era originalmente un emoticón con función discursiva. El tokenizer fija el techo de calidad de todo el pipeline downstream.

### Sistema CRF y features

#### Modelo base: Conditional Random Field

CRF (Lafferty, McCallum & Pereira 2001) es un **modelo discriminativo log-lineal** para etiquetado secuencial. Modela $P(\text{tags} \mid \text{tokens})$ con dependencias entre tags vecinos y features arbitrarios sobre la secuencia. Para POS tagging, CRF es estándar — mejor que HMM porque permite features ricos (afijos, capitalización, vecinos) sin asumir independencia condicional. Gimpel et al. usaron implementación en C++ propia.

#### Features base

Standard POS features que cualquier sistema usaría:

- **Word feature** por cada tipo de palabra observado.
- **Digit/hyphen presence**.
- **Suffixes** hasta longitud 3.
- **Capitalization patterns**.

Estos solos dan accuracy **83.38%**. Bien pero no suficiente.

#### Features Twitter-específicos (la innovación)

Cinco bloques adicionales, evaluados ablativamente:

- **TWORTH** — Regex para detectar at-mentions, hashtags, URLs. Aporta **+1.00%**.
- **NAMES** — Gazetteers de tokens frecuentemente capitalizados (top-N por likelihood de capitalización). Aportó solo **+0.02%** — probablemente el TAGDICT ya captura la mayor parte.
- **TAGDICT** — Tag dictionary del Penn Treebank: para cada palabra que aparece en PTB, agregar como soft feature los tags PTB de esa palabra. Es **type-level domain adaptation**: usar conocimiento del dominio fuente (newswire) para informar el modelo en el dominio target (Twitter). Aporta **+1.06%**.
- **DISTSIM** — Features distribucionales basadas en SVD de matriz de co-ocurrencia sucesor/predecesor, calculada sobre 1.9M tokens de tweets sin etiquetar (134k tweets). Cada palabra obtiene un embedding de 50 dimensiones. Esto es **proto-word2vec** (~2 años antes de word2vec). Aporta **+1.06%**.
- **METAPH** — Algoritmo Metaphone (Philips 1990) reduce variantes ortográficas a un mismo "phonetic key": `{thangs, thanks, thanksss, thanx, thinks, thnx}` → todos a `0NKS`; `{lmao, lmaoo, lmaooooo}` → todos a `LM`. A veces demasiado coarse: `{war, we're, wear, were, where, worry}` → todos a `WR`. Aún así aporta **+0.42%**.

---

## Resultados

### Resultados ablativos finales (Tabla 2)

| Sistema | Test accuracy |
|---|---|
| Annotator agreement (ceiling) | 92.2% |
| **Our tagger (full features)** | **89.37%** |
| – DISTSIM | 88.31% (-1.06%) |
| – TAGDICT | 88.31% (-1.06%) |
| – TWORTH | 88.37% (-1.00%) |
| – METAPH | 88.95% (-0.42%) |
| – NAMES | 89.39% (+0.02%) |
| Our tagger, base features only | 83.38% |
| Stanford tagger (retrained on same data) | 85.85% |

**Reducción de error relativa vs Stanford = 25%.**

### Robustez con poca data

Comentario notable al final del paper:

> *"test set accuracy when training on only 500 tweets drops to 87.66%, a decrease of only 1.7%"*

Con la mitad de los datos (500 tweets etiquetados), el accuracy cae apenas 1.7%. Sugiere que **la curva de aprendizaje se aplana rápido** — los features de TAGDICT y DISTSIM hacen mucho del trabajo con pocos ejemplos etiquetados.

### Análisis por tag (Tabla 3)

Recall por tag en el test set:

| Tag | Recall | Confusión más común |
|---|---|---|
| `@` (at-mention) | 99% | V |
| `,` (puntuación) | 98% | ~ |
| `&` (conj coord) | 98% | ^ |
| `U` (URL) | 97% | , |
| `O` (pronoun) | 97% | ^ |
| `P` (preposition) | 95% | R |
| `D` (determiner) | 95% | ^ |
| `L` (nom+verbal) | 93% | V |
| `V` (verbo) | 91% | N |
| `~` (discourse marker) | 91% | , |
| `$` (numeral) | 89% | P |
| `#` (hashtag) | 89% | ^ |
| `E` (emoticon) | 88% | , |
| `N` (common noun) | 85% | ^ |
| `R` (adverb) | 83% | A |
| `!` (interjection) | 82% | N |
| `A` (adjective) | 79% | N |
| `T` (verb particle) | 72% | P |
| `^` (proper noun) | 71% | N |
| `Z` (proper + poss) | 45% | ^ |
| `G` (garbage) | 26% | , |

**Observaciones clave:**

- Los tags **Twitter-específicos rankean entre los mejores** (`@` 99%, `U` 97%, `~` 91%, `#` 89%, `E` 88%). Las features TWORTH funcionan.
- **Proper nouns son el punto débil**: solo 71% recall. La capitalización inconsistente en Twitter ("obama", "Obama", "OBAMA") confunde al sistema.
- **G (garbage)** es el peor: 26% recall — pero por naturaleza, es el cajón de sastre.

---

## Limitaciones

Los autores son explícitos en los límites:

1. **Proper nouns con capitalización no estándar.** "obama" sin mayúscula confunde al sistema; tampoco lo arregla el feature NAMES porque ese mira capitalización.
2. **Categoría G es heterogénea.** Mezcla foreign words, partial words, símbolos, errores de tokenización. Tratamiento residual.
3. **Solo inglés.** El tagger es 100% inglés-céntrico (entrenado con tweets de zona USA, UI en inglés).
4. **Dataset pequeño.** 1,827 tweets es poco. Aún así demuestran que con buen feature engineering basta — pero hoy se hace con datasets de 10-100x el tamaño.
5. **No usa contexto largo.** CRF mira solo features locales. Un Transformer captura más contexto pero también requiere más datos.

---

## Por qué importa hoy

El paper tuvo un impacto desproporcionado al tamaño (6 páginas) por **liberar recursos abiertos** que se volvieron estándar:

1. **El POS tagset Twitter de Gimpel** es **el primer tagset estándar para texto de redes sociales**. Trabajos posteriores extienden o critican este tagset pero lo usan como referencia.
2. **El corpus anotado** (1,827 tweets con gold POS tags) fue distribuido vía `http://www.ark.cs.cmu.edu/TweetNLP`. Reusado en docenas de papers posteriores.
3. **El tokenizer** se portó a NLTK como `TweetTokenizer` y se volvió la herramienta de facto para tokenizar texto de redes sociales en Python.
4. **Las features METAPH y DISTSIM** anticipan ideas que serían formalizadas con word embeddings (word2vec 2013) y son aún un buen baseline conceptual.
5. **El estilo de paper "case study en rapid engineering"** (200 person-hours, 17 personas, 2 meses) **influenció cómo se publican papers de annotation en NLP**: enfocarse en metodología reproducible y release de recursos.

A mayo de 2026, el paper tiene **~700 citas** en Google Scholar. No es un megapaper en citaciones brutas, pero su impacto vía herramientas reusadas es masivo. **Cualquier paper de social media NLP en los últimos 15 años probablemente usó el TweetTokenizer**, directa o indirectamente.

### Sucesores y evolución

- **Owoputi et al. (2013)**, *Improved Part-of-Speech Tagging for Online Conversational Text with Word Clusters* — el mismo equipo CMU, mejora el tagger con Brown clusters, sube a 93.4% accuracy.
- **Ritter et al. (2011)**, *Named Entity Recognition in Tweets: An Experimental Study*, EMNLP — NER específico para Twitter.
- Con BERT y Transformers (2018+) el accuracy escaló a >95% con un modelo entrenado en datasets más grandes (TweetEval, TweetBERT). Pero para casos sin GPU, **el sistema Gimpel sigue siendo el baseline rápido en CPU**.

### Verificación práctica

```python
from nltk.tokenize import word_tokenize, TweetTokenizer

texto = "OMG @user #yolo this is sooooo cool :-) http://t.co/xyz <3"

# Comparación lado a lado
print("word_tokenize (Penn Treebank):")
print(word_tokenize(texto))
# Esperado: probablemente rompe @user, #yolo, :-) y la URL

print("\nTweetTokenizer:")
print(TweetTokenizer().tokenize(texto))
# Esperado: preserva @user, #yolo, :-) y la URL como tokens unitarios

print("\nTweetTokenizer con normalización:")
print(TweetTokenizer(reduce_len=True, strip_handles=True).tokenize(texto))
# Esperado: "sooooo" -> "soo" (reduce_len), "@user" -> "" (strip_handles)
```

La diferencia entre las dos tokenizaciones es **exactamente la lección de este paper**.

---

## Notas y enlaces

- **Clase asociada**: [Clase 16 - NLP clásico, NLTK, BoW, embeddings](/clases/clase-16).
- **Laboratorio asociado**: [Lab 16 - Pipeline NLP con NLTK/spaCy/NLLB/VADER](/laboratorios/lab-16).
- **Fundamento relacionado**: [Tokenización clásica](/fundamentos/tokenizacion-clasica).
- **Cita BibTeX**:

```bibtex
@inproceedings{gimpel2011part,
  title={Part-of-speech tagging for {Twitter}: Annotation, features, and experiments},
  author={Gimpel, Kevin and Schneider, Nathan and O'Connor, Brendan and Das, Dipanjan and Mills, Daniel and Eisenstein, Jacob and Heilman, Michael and Yogatama, Dani and Flanigan, Jeffrey and Smith, Noah A},
  booktitle={Proceedings of ACL-HLT (Short)},
  pages={42--47},
  year={2011}
}
```
