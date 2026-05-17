# Unsupervised Multilingual Sentence Boundary Detection — Kiss & Strunk (2006)

**Autores:** Tibor Kiss y Jan Strunk (Sprachwissenschaftliches Institut, Ruhr-Universität Bochum, Alemania).
**Publicación:** *Computational Linguistics*, Volume 32, Number 4, pp. 485-525 (Diciembre 2006). 41 páginas. Es la **versión journal-of-record** del trabajo previamente reportado en Kiss & Strunk (2002a, 2002b).
**PDF local:** `Punkt-Kiss-Strunk-2006.pdf`
**Conexión con el laboratorio:** El bloque 1 del Práctico 16 (celdas 21-24) usa `nltk.sent_tokenize`, que **es exactamente el sistema Punkt descrito en este paper**. El nombre "Punkt" significa "punto" en alemán. Cuando ejecutas `nltk.download('punkt')` (celda 8), descargas los modelos preentrenados que produjeron los autores siguiendo el método de este paper.

---

## 1. Contexto histórico

A inicios de los 2000, la **segmentación de oraciones** era un problema considerado "casi resuelto" — pero solo bajo dos suposiciones generalmente aceptadas y muy frágiles:

1. **El sistema tiene acceso a listas de abreviaciones pre-compiladas por humanos** (`Mr.`, `Dr.`, `U.S.A.`, etc.).
2. **El sistema fue entrenado sobre un corpus etiquetado a mano** con marcas explícitas de fin de oración.

Esto bastaba para inglés newswire en el dominio para el que se entrenó. Pero fallaba estrepitosamente en:

- **Idiomas distintos del inglés**: cada idioma tiene su propio inventario de abreviaciones.
- **Géneros distintos** (clínico, jurídico, conversacional): cada uno trae abreviaciones que la lista pre-compilada no cubre.
- **Texto en mayúsculas o minúsculas únicas** (single-case): los sistemas basados en capitalización (la pista "después del punto viene mayúscula") colapsan.
- **Corpora nuevos**: cada vez que aparece un dominio nuevo, hay que volver a anotar a mano.

El estado del arte previo era una colección de sistemas con sus respectivos límites:

| Sistema | Año | Tipo | Limitación principal |
|---|---|---|---|
| **Riley** (decision tree) | 1989 | Supervisado | Requiere 25M de palabras de AP newswire etiquetadas |
| **Satz** (Palmer & Hearst) | 1997 | Supervisado (NN o DT) | Requiere lexicón POS + lista de abreviaciones manual |
| **MxTerminator** (Reynar & Ratnaparkhi) | 1997 | Maximum entropy supervisado | Requiere 39k oraciones WSJ anotadas |
| **RE system** (Silla et al.) | 2003 | Basado en reglas | Requiere ~700 abreviaciones manuales por idioma |
| **Stamatatos et al.** | 1999 | Transformation-based learning (Brill) | Supervisado |
| **Grefenstette** | 1999 | Heurísticas + lexicón completo | Requiere lexicón exhaustivo del idioma |

Kiss y Strunk hicieron la pregunta clave: **¿es posible detectar fronteras de oración sin entrenamiento supervisado, sin listas de abreviaciones, sin POS tagger, y de manera portátil entre idiomas?**

La respuesta de este paper: **sí**, con un sistema de dos etapas basado en estadísticas colocacionales.

---

## 2. Contribución central

Punkt aporta tres cosas en una:

1. **Un algoritmo no supervisado para detectar abreviaciones**, basado en tres propiedades observables del candidato (no del contexto). Reformulación del problema: si encuentras todas las abreviaciones, casi todos los demás puntos son fronteras de oración por descarte.
2. **Una clasificación en dos etapas — type-based + token-based**:
   - **Etapa 1 (type-based)**: decide globalmente si una *palabra-tipo* es candidata a abreviación. Mira evidencia sobre todas las ocurrencias en el corpus.
   - **Etapa 2 (token-based)**: refina por *instancia*. Usa heurísticas (ortográfica, colocacional, frequent sentence starter, ordinal numbers) para corregir casos ambiguos.
3. **Validación masiva multilingüe**: probaron Punkt en corpora de **11 idiomas** (English, Brazilian Portuguese, Dutch, Estonian, French, German, Italian, Norwegian, Spanish, Swedish, Turkish) **sin re-entrenar nada**. Error medio: **1.26%** sobre 200k oraciones evaluadas.

El paper se publicó como el journal-of-record en *Computational Linguistics* (peer-reviewed, journal top en NLP). Esa publicación validó académicamente el método y motivó su adopción en NLTK como el sentence tokenizer por defecto.

---

## 3. Método en detalle

### 3.1 Reformulación del problema

El punto `.` tiene **tres roles ambiguos** en escritura:

1. **Marcador de fin de oración**: "Volvió a casa. Cenó."
2. **Marcador de abreviación**: "Dr. Smith fue al U.S.A. ayer."
3. **Decimal o número ordinal**: "3.14", "1990s" (esto último depende del idioma).

Los autores demuestran (sección 6.1) que **hasta el 30% de los puntos en texto típico son ambiguos** — no son fronteras de oración. Si detectas correctamente cuáles son abreviaciones, **todo lo demás** es frontera de oración por descarte.

Por eso el sistema se concentra en **detectar abreviaciones** primero, y deja sentence boundary detection como una consecuencia.

### 3.2 Las tres propiedades de las abreviaciones (etapa 1)

Una *abbreviation type* es una palabra que aparece con punto en una proporción muy alta de sus ocurrencias y tiene además ciertas propiedades físicas. Punkt formaliza esto con **tres factores cuantificables**:

#### Factor 1 — Collocational bond entre la palabra y el punto

Una abreviación está **fuertemente asociada al punto que la sigue**. Para medir esto Punkt usa una versión modificada del log-likelihood ratio de Dunning (1993):

```
H_0:  P(• | w) = P(•)              # el punto es independiente de la palabra
H_A:  P(• | w) = 0.99               # el punto sigue casi siempre a w
```

La hipótesis alternativa **no es "más frecuente que la media"** sino **"casi 1"**. Esta es una **innovación clave** sobre Dunning original (cuya alternativa es simplemente p1 ≠ p2). El motivo: las abreviaciones se caracterizan porque **virtualmente nunca aparecen sin punto**. La razón log λ resultante captura ese requisito de "casi siempre acompañadas".

#### Factor 2 — Brevity (longitud)

Las abreviaciones tienden a ser cortas. Pero los autores no usan un cutoff duro de longitud (como otros sistemas) — usan un **factor exponencial inverso**:

```
F_length(w) = 1 / exp(length(w))
```

donde `length(w)` es número de caracteres antes del punto final, **excluyendo los puntos internos** (porque "U.S.A." es corta a pesar de tener 5 caracteres).

El factor decae exponencialmente: si w tiene 3 caracteres, F = 1/e³ ≈ 0.05. Si tiene 10 caracteres, F = 1/e¹⁰ ≈ 0.00005. Esto **penaliza fuertemente** las palabras largas como candidatas a abreviación, pero permite excepciones (una palabra muy larga puede ser abreviación si tiene log λ enorme).

#### Factor 3 — Internal periods (puntos internos)

Muchas abreviaciones contienen puntos internos: `U.S.A.`, `i.e.`, `e.g.`, `Ph.D.`, `St.Pte.`. Esto es **evidencia muy fuerte** de que el candidato es abreviación.

```
F_periods(w) = 1 + número de puntos internos en w
```

Una palabra sin puntos internos: F = 1 (sin bonus).
Una palabra con 2 puntos internos: F = 3 (bonus 3x al score).

#### Factor 4 — Penalty para ocurrencias sin punto

Si una palabra **a veces aparece sin punto**, hay evidencia de que NO es abreviación. Pero algunas abreviaciones reales aparecen ocasionalmente sin punto (typos, headlines). Penalty exponencial:

```
F_penalty(w) = 1 / length(w)^C(w, ¬•)
```

donde `C(w, ¬•)` es el conteo de ocurrencias **sin punto** final. Cuantas más, mayor el castigo.

#### Combinación

```
score(w) = log λ(w) · F_length(w) · F_periods(w) · F_penalty(w)
```

Con un threshold de **0.3**: si `score(w) ≥ 0.3`, w se marca como abreviación. **Los autores derivaron este threshold empíricamente sobre inglés y lo mantuvieron fijo para los otros 10 idiomas**, lo cual es una decisión metodológica radical (no calibran threshold por idioma).

### 3.3 Token-based reclassification (etapa 2)

Después de la etapa 1, cada token con punto se marca tentativamente como:
- `<A>` si su tipo es abreviación.
- `<E>` si es elipsis (`...`).
- `<S>` si es frontera de oración (lo que sobra).

La etapa 2 mira el contexto para corregir y refinar. Aplica tres heurísticas:

#### Heurística ortográfica (4.1.1)

Para un token después de un período (potencial frontera de oración), Punkt mira **la palabra siguiente**:
- Si la palabra es **siempre mayúscula** en el corpus (e.g., nombres propios), no es evidencia clara.
- Si la palabra a veces aparece minúscula al principio de oración pero **nunca** minúscula dentro de oración, eso sugiere que SÍ hay frontera.
- En otros casos, devuelve "indecided".

Esto evita el ingenuo "mayúscula = nueva oración" que falla en alemán (todos los sustantivos son mayúsculos) y otros idiomas.

#### Heurística colocacional

Si dos palabras alrededor de un punto forman una **colocación** (i.e., aparecen juntas más frecuentemente de lo esperado por azar), eso sugiere que el punto no es frontera de oración. Ejemplo: "St. Petersburg" forma colocación; "casa. Vino" no.

Punkt usa Dunning log-likelihood ratio con una condición unilateral: la colocación solo cuenta si la frecuencia conjunta es **mayor** que la esperada (no menor).

#### Frequent Sentence Starter

Para cada palabra que aparece después de un `<S>`, Punkt calcula si esa palabra es un "sentence starter típico" del corpus (e.g., "However", "The", "Moreover" en inglés). Si la palabra siguiente al punto candidato es un sentence starter frecuente, hay evidencia adicional a favor de frontera de oración.

#### Tratamiento especial de iniciales y ordinales

Las **iniciales** (`A.`, `B.`, `C.`) son técnicamente abreviaciones pero raras — la etapa 1 type-based no las detecta bien por baja frecuencia. Punkt aplica una heurística adicional: si el token tiene longitud 1 y está seguido inmediatamente de una palabra capitalizada, lo trata como inicial.

Los **ordinales numéricos** (`1.`, `2.`, `3.`) son tratados especialmente para idiomas que los marcan con punto (alemán, sueco, noruego, etc.). En inglés y español no se usa este formato, así que esta heurística es no-op para esos idiomas.

### 3.4 La arquitectura completa en un diagrama

```
┌──────────────────────────────────────┐
│ ETAPA 1: Type-based classification   │
│ (sobre tipos únicos del corpus)      │
│                                       │
│  • Collocational bond (log λ revised)│
│  • Length penalty                     │
│  • Internal periods bonus             │
│  • No-period penalty                  │
│                                       │
│  → Anotación inicial: <A>, <E>, <S>  │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ ETAPA 2: Token-based reclassification│
│ (por instancia, mirando contexto)    │
│                                       │
│  • Orthographic heuristic             │
│  • Collocation heuristic              │
│  • Frequent sentence starter          │
│  • Special: initials, ordinals        │
│                                       │
│  → Anotación final: <A>, <E>, <S>,   │
│                     <A><S>, <E><S>   │
└──────────────────────────────────────┘
```

Un punto que es **a la vez** abreviación Y final de oración (e.g., "Volvió a U.S.A.") recibe la anotación combinada `<A><S>`.

---

## 4. Experimentos clave

### 4.1 Corpora de evaluación

Probaron 11 idiomas, todos sobre **corpora de periódicos** (newspaper). Tamaños:

| Idioma | Tokens evaluados |
|---|---|
| German | 34,256 |
| English | 24,282 |
| Estonian | 23,243 |
| Turkish | 18,942 |
| Dutch | 18,068 |
| Swedish | 17,752 |
| B. Portuguese | 13,725 |
| Spanish | 11,714 |
| Norwegian | 25,531 |
| French | 11,601 |
| Italian | 10,405 |

Total aproximado: 200k oraciones para evaluación.

### 4.2 Resultado headline (Tabla 21)

Error rate de detección de frontera de oración (mientras menor, mejor):

| Idioma | **Punkt** | MxTerminator (entrenado por idioma) |
|---|---|---|
| German | **0.35%** | 0.63% |
| Norwegian | **0.81%** | 1.34% |
| French | **1.54%** | 2.66% |
| Italian | **1.13%** | 2.45% |
| Spanish | **1.06%** | 1.60% |
| English | 1.65% | **1.53%** |
| Dutch | **0.97%** | 1.13% |
| B. Portuguese | 1.11% | **1.10%** |
| Estonian | **2.12%** | 2.79% |
| Swedish | 1.76% | 2.39% |
| Turkish | **1.31%** | 1.77% |
| **Mean** | **1.26%** | 1.76% |

Punkt **gana en 9 de 11 idiomas** contra MxTerminator (un sistema supervisado entrenado por idioma con miles de oraciones etiquetadas), y empata en los otros 2.

### 4.3 Ablation de las heurísticas (Tabla 19)

Probando qué heurística contribuye más:

| Sistema | Estonian | German | French | Mean approx |
|---|---|---|---|---|
| A (solo type-based) | 7.37% | 2.38% | 2.94% | ~2.8% |
| B (+ collocation) | 2.94% | 0.47% | 2.61% | ~1.7% |
| C (+ frequent sentence starter) | 2.80% | 0.42% | 1.96% | ~1.6% |
| D (+ orthographic) | 2.18% | 0.36% | 1.78% | ~1.4% |
| E (+ special orthographic for initials, completo) | 2.12% | 0.35% | 1.54% | ~1.3% |

La **heurística colocacional** es la que más aporta (-1.1% promedio). La etapa 2 token-based **reduce el error a la mitad** comparado con solo etapa 1.

### 4.4 Análisis de errores remanentes

Los autores identifican 5 tipos de error que Punkt no maneja bien:

1. **Homografía**: "in" es preposición frecuente Y abreviación de "inch". Como el tipo "in" aparece sin punto la mayoría del tiempo, Punkt lo clasifica como no-abreviación, y cuando aparece "in." con punto, lo trata como frontera de oración.
2. **Uso inconsistente de abreviaciones**: en sueco "osv." (and so on) a veces tiene punto, a veces no. Si la versión sin punto domina en el corpus, Punkt no detecta "osv." como abreviación.
3. **Data sparseness**: abreviaciones que aparecen solo 1-2 veces no acumulan evidencia colocacional suficiente.
4. **Falta de evidencia ortográfica**: si la palabra que sigue a una abreviación es nombre propio (siempre capitalizado), la heurística ortográfica no decide.
5. **Estructura de texto**: titulares no terminan con punto, listas tienen ordinales, etc. Punkt no modela estructura de documento.

### 4.5 Independencia del threshold (Tabla 20)

Threshold de 0.3 (derivado en inglés) vs threshold óptimo por idioma:

| Idioma | Punkt @ 0.3 | Punkt @ óptimo | Diff |
|---|---|---|---|
| Italian | 1.13% | 1.00% (0.2) | 0.13% |
| French | 1.54% | 1.51% (0.2) | 0.03% |
| English | 1.65% | 1.59% (0.2) | 0.06% |
| Estonian | 2.12% | 2.10% (0.6) | 0.02% |
| Average | — | — | **0.03%** |

**El threshold 0.3 derivado de inglés es casi óptimo para los otros 10 idiomas**. Diferencia promedio: 0.03%. Esto sugiere que el método es **genuinamente language-independent** — los autores no inflaron el resultado con ajustes manuales por idioma.

---

## 5. Limitaciones reconocidas

Algunas explícitas en el paper, otras visibles en retrospectiva:

1. **Asume sistema de escritura alfabético con punto como marcador de fin de oración.** No funciona para chino (no usa espacios), tailandés, árabe sin diacríticos, etc.
2. **No modela estructura de documento.** Titulares sin punto, listas numeradas, código, URLs — todo trata como prosa continua.
3. **Necesita un corpus para aprender.** Es no supervisado pero no zero-shot: necesita ver suficiente texto para acumular estadísticas. Para corpora muy pequeños (<10k tokens) la calidad cae.
4. **Verb-final languages** (turco, japonés) son problemáticas porque ciertos verbos cierran casi todas las oraciones; el sistema puede confundirlos con abreviaciones.
5. **Homografía abreviation/palabra ordinaria** (vimos "in" / "in.").
6. **No detecta sentencias dentro de oraciones**: si una oración compleja tiene una sub-cláusula que en otro contexto sería oración independiente, Punkt no la separa.

Limitaciones no discutidas pero importantes hoy:
7. **No es contextual.** Un Transformer fine-tuned para sentence boundary detection (e.g., basado en BERT) supera fácilmente a Punkt en accuracy, especialmente en texto difícil (medical, legal).
8. **Idiomas de bajos recursos.** Lenguas con poca data digital (lenguas indígenas, africanas) no tienen corpora suficientes para entrenar Punkt eficientemente.
9. **Code-switching.** Texto que mezcla idiomas (común en redes sociales) confunde las estadísticas de un solo idioma.

---

## 6. Impacto y legado

Punkt es **el sentence tokenizer más usado del NLP clásico**. Algunos hechos:

- **NLTK lo adoptó** como `nltk.sent_tokenize` desde la versión 0.9 (2007) y sigue siendo el default en 2026. Modelos pre-entrenados disponibles en 18 idiomas vía `nltk.download('punkt')` (y desde NLTK 3.8.2 también `'punkt_tab'`).
- **Citas:** a mayo de 2026, **~1700 citas en Google Scholar** para el journal paper de 2006, más miles adicionales para las versiones de workshop/conference de 2002.
- **spaCy** usa un sentence segmenter más moderno por defecto (basado en dependency parsing), pero ofrece Punkt como alternativa para texto sin POS info.
- **scikit-learn** no incluye sentence segmentation propio — los usuarios típicamente usan Punkt vía NLTK.
- En **producción industrial**, Punkt sigue corriendo en miles de pipelines NLP que precedieron a la era Transformer: clasificación de documentos, IR, extracción de información clásica, OCR post-processing.
- Su **diseño type-based + token-based** influenció arquitecturas posteriores de IE y NER, donde primero se aprende un vocabulario y luego se reclasifica por instancia.

El paper también es ejemplo de **buena ciencia metodológica** en NLP:
- Threshold derivado de un solo idioma, validado en 10 más sin tuning.
- Comparación honesta contra baselines supervisados (Punkt no siempre gana, y los autores lo reportan).
- Análisis exhaustivo de errores remanentes en lugar de inflar las métricas.

---

## 7. Conexión directa con el Práctico 16

| Celda del lab | Concepto de Punkt |
|---|---|
| 8 | `nltk.download('punkt')` — descarga **modelos Punkt preentrenados** en 18 idiomas. Cada modelo es un pickle con los parámetros aprendidos: listas de abreviaciones detectadas, frequent sentence starters, lookup tables ortográficas. |
| 8 | `nltk.download('punkt_tab')` — **versión re-empaquetada** del mismo modelo en formato `.tab` (más portable, agregada en NLTK 3.8.2, abril 2024). Si solo descargas `punkt` sin `punkt_tab`, `sent_tokenize` falla con LookupError en versiones recientes. |
| 22 | `sent_tokenize(s)` en inglés → invoca `PunktSentenceTokenizer.tokenize(s)` con el modelo inglés. Verás cómo maneja `"Mr. President"`, `"U.S.A."`, etc. — todo gracias al sistema descrito en este paper. |
| 23 | `sent_tokenize(s, language='spanish')` — el mismo algoritmo Punkt pero usando el modelo entrenado sobre corpus en español. La función subyacente lee `~/nltk_data/tokenizers/punkt_tab/spanish/`. |
| 26-29 | Aunque no se usa Punkt explícitamente, las stop-words y stemming vienen DESPUÉS de tokenizar — si Punkt segmenta mal, todo el pipeline corriente abajo se rompe. |
| 36-42 | spaCy (bloque 2 del lab) **NO usa Punkt** — usa su propio sentence segmenter basado en dependency parsing. Esta es una diferencia metodológica importante que vamos a discutir cuando lleguemos a spaCy. |

**Cosas que puedes verificar:**

```python
from nltk.tokenize import PunktSentenceTokenizer
import nltk.data
spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
# Inspeccionar las abreviaciones aprendidas
print(spanish_tokenizer._params.abbrev_types)
# Ver los sentence starters frecuentes
print(spanish_tokenizer._params.sent_starters)
# Inspeccionar el modelo ortográfico
print(list(spanish_tokenizer._params.ortho_context.items())[:20])
```

Esto te muestra **directamente lo que el algoritmo aprendió** del corpus de español — las abreviaciones, frequent starters, y patrones ortográficos. Es una forma muy concreta de ver Punkt en acción.

---

## 8. Lecturas relacionadas

**Precursores directos:**
- Riley (1989), *Some applications of tree-based modelling to speech and language*, DARPA — primer decision tree sentence segmenter.
- Palmer & Hearst (1997), *Adaptive Multilingual Sentence Boundary Disambiguation*, Computational Linguistics — sistema Satz con POS features.
- Reynar & Ratnaparkhi (1997), *A Maximum Entropy Approach to Identifying Sentence Boundaries*, ANLP — sistema MxTerminator.
- Mikheev (2002), *Periods, Capitalized Words, etc.*, Computational Linguistics — análisis lingüístico del problema.

**Estadística colocacional fundacional:**
- Dunning (1993), *Accurate Methods for the Statistics of Surprise and Coincidence*, Computational Linguistics — el log-likelihood ratio que Punkt usa para abreviaciones (con la modificación crítica que vimos en sección 3.2).

**Versiones previas del mismo trabajo por los autores:**
- Kiss & Strunk (2002a), *Scaled Log-Likelihood Ratios for Abbreviation Detection*, technical report.
- Kiss & Strunk (2002b), *Viewing Sentence Boundary Detection as Collocation Identification*, KONVENS.

**Aplicaciones modernas:**
- spaCy documentation, *Sentence Segmentation* — explica cómo spaCy usa dependency parsing en lugar de Punkt.
- Tiedemann (2010), *Sentence Splitting Toolkits Survey* — comparación amplia de Punkt vs alternativas modernas.

**Para tokenización a nivel de palabra** (no oración, que es lo que hace Punkt), ver `Gimpel-Twitter-POS-2011.md` que analiza el TweetTokenizer y por qué la tokenización Penn Treebank estándar falla en texto de redes sociales.
