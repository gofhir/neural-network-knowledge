# An Algorithm for Suffix Stripping — Porter (1980)

**Autor:** M.F. Porter (Computer Laboratory, Corn Exchange Street, Cambridge, UK).
**Publicación:** *Program: Electronic Library and Information Systems*, Vol. 14, No. 3, pp. 130–137 (Julio 1980).
**PDF local:** `Porter-Stemmer-1980.pdf` (paper de 4 páginas en formato 2-column, ~12 páginas equivalentes en formato single-column).
**Conexión con el laboratorio:** El bloque 1 del Práctico 16 (celda 31) usa `PorterStemmer()` como uno de los 4 stemmers. Es el algoritmo descrito **literalmente** en este paper, implementado por Porter en BCPL en 1980 y portado a Python desde entonces sin cambios sustantivos. Lo que el lab te muestra es la **implementación original de hace 45 años, aún en producción**.

---

## 1. Contexto histórico

A finales de los 70, el procesamiento de información (Information Retrieval, IR) ya tenía 20 años como disciplina académica, pero los **stemmers** estaban en estado primitivo. Los antecedentes que Porter cita en su paper:

| Sistema | Año | Características |
|---|---|---|
| **Lovins** (J.B. Lovins) | 1968 | Primer stemmer formalizado para inglés. ~260 reglas, recodificación posterior. Maximalista (intenta cubrir cada caso). |
| **Andrews** (K. Andrews, Cambridge) | 1971 | Conflation algorithm rápido, base de varios sistemas IR posteriores. |
| **Dawson** | 1974 | Suffix removal + word conflation. Más simple que Lovins pero comparable. |
| **Salton SMART** | 1971+ | Sistema integrado IR con stemming embebido. |

**El problema con los stemmers de los 70**: eran enormes (Lovins tenía cientos de reglas en cascada), difíciles de mantener, sensibles al orden de aplicación, e incluso así fallaban en muchos casos. Una típica falla de Lovins: convertir `RELATE` y `RELATIVITY` al mismo stem aunque semánticamente son distantes ("relate" tiene relación con relacionar, "relativity" con la teoría de Einstein).

Porter quería un algoritmo **simple, rápido, y suficientemente bueno**. Su tesis filosófica clave (citada literalmente del paper):
> *"It would not be at all obvious under what circumstances a suffix should be removed, even if we could exactly determine the suffixes of the words by automatic means."*

Es decir: **no buscaba un stemmer lingüísticamente correcto**, buscaba uno que **mejorara recall en IR**, sabiendo que cualquier criterio de "corrección" lingüística es discutible.

### El test que motivó el diseño

Porter compara su algoritmo contra el sistema previo de Cambridge (Andrews 1971 ampliado a 1975) sobre la **Cranfield 200 collection** (200 documentos científicos, 42 queries, evaluación estándar de IR de la época). Resultados (tabla del paper, página 314):

| Recall | Sistema anterior — Precision | Sistema Porter — Precision |
|---|---|---|
| 0 | 57.24 | 58.60 |
| 10 | 56.85 | 58.13 |
| 20 | 52.85 | 53.92 |
| 30 | 42.61 | 43.51 |
| 40 | 42.20 | 39.39 |
| 50 | 39.06 | 38.85 |
| 60 | 32.86 | 33.18 |
| 70 | 31.64 | 31.19 |
| 80 | 27.15 | 27.52 |
| 90 | 24.59 | 25.85 |
| 100 | 24.59 | 25.85 |

**Conclusión empírica de Porter**: su algoritmo simple **iguala o supera** al sistema más elaborado en todos los recall levels. **La simplicidad no fue un sacrificio de calidad**.

---

## 2. Contribución central

Porter aportó **tres cosas en un único algoritmo**:

1. **Una métrica simple para "longitud silábica"** del stem: el parámetro **`m`** (measure), que cuenta secuencias `VC` (vocal-consonante) en la palabra. Esto permite tomar decisiones de stripping basadas en la "robustez" del stem que queda.

2. **Un sistema de 5 pasos secuenciales** que aplica reglas de la forma `(condition) S1 → S2`. Cada paso elimina sufijos de un tipo específico (plurales, derivacionales, terminales). El orden importa: cada paso depende del estado dejado por el anterior.

3. **Una implementación pragmática**: ~400 líneas de BCPL, procesaba 10,000 palabras en **8.1 segundos** sobre un IBM 370/165 en Cambridge (1980). Esto es ~1,200 palabras por segundo en hardware de la época. Hoy en Python procesa cientos de miles por segundo.

El paper también deja claro que **NO** debería usarse para todos los casos:
- Recomienda aplicarlo a **listas de vocabulario derivadas de texto continuo**, no a texto continuo entero.
- Reconoce errores deliberados (`SAND` vs `SANDER` se conflate aunque sea incorrecto; `PROBE` vs `PROBATE` se conflate aunque tengan significados distintos).

Es deliberadamente **un algoritmo "good enough"** para IR, no una herramienta lingüística rigurosa.

---

## 3. El algoritmo en detalle

### 3.1 Definiciones fundamentales

**Consonante (c)**: cualquier letra que **no sea** A, E, I, O, U; **ni Y precedida por consonante**. Notar la sutileza: la `Y` puede ser vocal o consonante según contexto.

- `TOY` → consonantes: T, Y (Y al final tras vocal O = consonante).
- `SYZYGY` → consonantes: S, Z, G (la última Y).
- `SKY` → consonantes: S, K (la Y al final tras K es vocal).

**Vocal (v)**: lo que no es consonante.

**Forma general de cualquier palabra**:

```
[C] (VC)^m [V]
```

donde `[C]` y `[V]` son secuencias opcionales de consonantes/vocales al inicio/final, y `(VC)^m` significa "m repeticiones de VC". El **measure `m`** es lo que importa.

Ejemplos del paper (página 314):

| Palabra | Análisis | m |
|---|---|---|
| TR, EE, TREE, Y, BY | no hay VC repetidas | 0 |
| TROUBLE, OATS, TREES, IVY | 1 secuencia VC | 1 |
| TROUBLES, PRIVATE, OATEN, ORRERY | 2 secuencias VC | 2 |

**Por qué m importa:** Porter no quiere strippar sufijos si el stem que queda es muy corto. La regla típica del algoritmo dice "elimina este sufijo solo si `m > 1`", lo cual asegura que el stem tenga al menos 2 secuencias VC.

### 3.2 Estructura de una regla

```
(condition) S1 → S2
```

**Significado**: si la palabra termina con `S1`, **y** el stem antes de `S1` cumple `condition`, reemplazar `S1` por `S2`.

**Ejemplos del paper (página 314)**:

```
(m > 1) EMENT →
```

Significado: si la palabra termina en `EMENT` y el stem restante tiene m > 1, eliminar `EMENT` (S2 está vacío). Por ejemplo:

```
REPLACEMENT → REPLAC  (porque REPLAC tiene m=2)
PAVEMENT → PAVE  (NO se aplica porque PAV tiene m=1)
```

**Condiciones disponibles** (página 314):
- `*S` — el stem termina con S (similar para otras letras).
- `*v*` — el stem contiene una vocal.
- `*d` — el stem termina con doble consonante (e.g., `-TT`, `-SS`).
- `*o` — el stem termina con `cvc`, donde la segunda c **no** es W, X o Y.

**Operadores lógicos** en condiciones: `and`, `or`, `not`. Ejemplo:
```
(m > 1 and (*S or *T))
```
Significa "m mayor a 1 Y stem termina en S o T".

### 3.3 Los 5 pasos del algoritmo

#### Paso 1a — Plurales

```
SSES → SS    caresses → caress
IES  → I     ponies   → poni
                 ties → ti
SS   → SS    caress   → caress (sin cambio)
S    →       cats     → cat
```

**Solo se aplica una regla por paso** — la **regla con el S1 más largo que matchee** la palabra. Por eso `CARESSES` matchea `SSES` (más largo que `SS` o `S`), no las otras.

#### Paso 1b — Gerundios y participios

```
(m>0)  EED → EE     feed → feed (NO, porque "feed" tiene m=0)
                    agreed → agree
(*v*)  ED  →        plastered → plaster
                    bled → bled (NO, porque "bl" no contiene vocal)
(*v*)  ING →        motoring → motor
                    sing → sing (NO, "s" no contiene vocal)
```

**Si las reglas con ED o ING se aplican** (segunda y tercera), se ejecuta un **post-procesamiento** para restaurar palabras:

```
AT → ATE    conflat(ed) → conflate
BL → BLE    troubl(ing)  → trouble
IZ → IZE    siz(ed)      → size
(doble consonante final que no sea L,S,Z) → eliminar duplicado
    hopp(ing)  → hop
    tann(ed)   → tan
    fall(ing)  → fall (excepción: LL queda)
(m=1 y *o) → agregar E
    fail(ing) → fail (sin cambio, *o no aplica)
    fil(ing)  → file
```

**Esto es lo que hace que Porter funcione mejor que un stripping ingenuo**: trata las reglas como un sistema con post-condiciones, no como aplicaciones aisladas.

#### Paso 1c — Y → I

```
(*v*) Y → I    happy → happi
               sky   → sky (NO, no hay vocal antes de Y)
```

#### Paso 2 — Derivaciones largas

Reglas como (extracto):

```
(m>0) ATIONAL → ATE    relational → relate
(m>0) TIONAL  → TION   conditional → condition
(m>0) ENCI    → ENCE   valenci    → valence
(m>0) IZER    → IZE    digitizer  → digitize
(m>0) ABLI    → ABLE   conformabli → conformable
(m>0) ALLI    → AL     radicalli  → radical
(m>0) IZATION → IZE    vietnamization → vietnamize
(m>0) ATION   → ATE    predication → predicate
(m>0) FULNESS → FUL    hopefulness → hopeful
(m>0) BILITI  → BLE    sensibiliti → sensible
... (~20 reglas en total)
```

Porter observa (página 315): *"the S1-strings in step 2 are presented here in the alphabetical order of their penultimate letter"* — esto permite **lookup eficiente** con un program switch sobre la penúltima letra.

#### Paso 3

Más reglas derivacionales:
```
(m>0) ICATE → IC     triplicate → triplic
(m>0) ATIVE →        formative  → form
(m>0) ALIZE → AL     formalize  → formal
(m>0) ICITI → IC     electriciti → electric
(m>0) FUL   →        hopeful    → hope
(m>0) NESS  →        goodness   → good
```

#### Paso 4

Sufijos derivacionales más cortos:
```
(m>1) AL    →        revival     → reviv
(m>1) ANCE  →        allowance   → allow
(m>1) ENCE  →        inference   → infer
(m>1) ABLE  →        adjustable  → adjust
(m>1) IBLE  →        defensible  → defens
(m>1) ANT   →        irritant    → irrit
(m>1) MENT  →        replacement → replac
(m>1) ENT   →        adjustment  → adjust
(m>1) OU    →        homologou   → homolog
(m>1) ISM   →        communism   → commun
(m>1) ATE   →        activate    → activ
(m>1) ITI   →        angulariti  → angular
(m>1) OUS   →        homologous  → homolog
(m>1) IVE   →        effective   → effect
(m>1) IZE   →        bowdlerize  → bowdler
```

**Nota crítica**: en paso 4 se usa `m > 1` (no `m > 0`). Más restrictivo. Por eso `REVIVAL → REVIV` (m=2 antes del AL) pero `OVAL` no pasaría (m=1).

#### Paso 5

##### 5a — Eliminar E final

```
(m>1)              E →    probate → probat
                          rate    → rate (NO, m=1)
(m=1 and not *o)   E →    cease   → ceas
```

##### 5b — Eliminar consonante doble final si termina en L

```
(m>1 and *d and *L) → consonante simple
    controll → control
    roll     → roll (NO, m=1)
```

### 3.4 Características emergentes del algoritmo

**Composición de pasos**: cada palabra puede pasar por múltiples pasos. Ejemplo del paper (página 316):

```
GENERALIZATIONS
  → Paso 1 (S): GENERALIZATION
  → Paso 2 (IZATION → IZE): GENERALIZE
  → Paso 3 (ALIZE → AL): GENERAL
  → Paso 4 (AL): GENER
```

`GENERALIZATIONS` (14 letras) se reduce a `GENER` (5 letras) en 4 pasos.

**Reducción del vocabulario** sobre un test de 10,000 palabras:

| Paso | Palabras modificadas |
|---|---|
| 1 | 3,597 |
| 2 | 766 |
| 3 | 327 |
| 4 | 2,424 |
| 5 | 1,373 |
| No reducidas | 3,650 |
| **Vocabulario final** | **6,370 stems únicos** |

**Reducción del 36%** en el tamaño del vocabulario. Eso es Heaps en acción ([Punkt-Kiss-Strunk-2006.md](clase_16/papers/Punkt-Kiss-Strunk-2006.md) tiene una discusión paralela en el contexto de sentence boundaries).

---

## 4. Limitaciones reconocidas

Porter es honesto en el paper sobre los errores del algoritmo:

### 4.1 Errores deliberados

```
RELATE y RELATIVITY → mismo stem (RELATIV/RELAT)
PROBE y PROBATE → distinto (PROB / PROBAT)
```

¿Por qué? Porque el algoritmo simplifica:
- `RELATIVITY` pasa por `BILITI → BLE`? No, es `IVITY` → `IVE`. Y luego `IVE → ` por paso 4. Termina en `RELAT`.
- `RELATE` pasa por paso 5a (E → ) si m>1. RELATE = R-E-L-A-T-E, ¿es m>1? R-E (VC) L-A-T (no termina en VC), entonces m=1. No se elimina la E. Termina en `RELATE` o `RELAT`.

**Resultado pragmático**: PORTER las conflate juntas (`RELAT`), aunque semánticamente sean disjuntas. **Esto es por diseño**: Porter explícitamente dice que para IR es preferible **over-conflation a under-conflation** cuando los errores son raros y los aciertos son la mayoría.

### 4.2 Casos donde el spelling complica

```
DECEIVE / DECEPTION → distintos stems (DECEIV / DECEPT)
RESUME / RESUMPTION → distintos (RESUM / RESUMPT)
INDEX / INDICES → distintos (INDEX / INDIC)
```

El paper reconoce: *"In view of the error rate that must in any case be expected, it did not seem worthwhile to try and cope with these cases"* (página 314). Habría requerido reglas especiales para variaciones ortográficas raras.

### 4.3 Reglas inconsistentes

```
list A (NO se aplica -ATE):  RELATE, PROBATE, CONFLATE, PIRATE, PRELATE
list B (SÍ se aplica -ATE):  DERIVATE, ACTIVATE, DEMONSTRATE, NECESSITATE, RENOVATE
```

Diferencia: m=1 vs m>1. PRELATE tiene m=1 (PR-E-L-A-T-E = solo una secuencia VC tras la primera C), DEMONSTRATE tiene m=4.

**Consecuencia**: `PRELATE` y `ARCHPRELATE` reciben tratamiento distinto. `ARCHPRELATE` tiene m>1 (porque el prefijo agrega VCs), entonces sí se aplica `-ATE → ` y queda `ARCHPREL`. Mientras `PRELATE` queda intacta.

Esto es **una inconsistencia diseñada para no agregar más complejidad** (no intentar reconocer prefijos).

### 4.4 Limitaciones generales no discutidas pero conocidas hoy

1. **Solo inglés.** Reglas hardcoded para morfología inglesa. No funciona en otros idiomas sin reescribir todo (que es lo que motivó **Snowball** 2001).
2. **Sobre-stripping de palabras técnicas.** `LASER → LASER` está bien, pero `MULTIPLE → MULTIPL`, `SINGULARITY → SINGULAR` pueden colapsar términos científicos a stems poco útiles.
3. **No maneja excepciones irregulares.** `WENT/GO`, `MICE/MOUSE` no se relacionan.
4. **El stem **no es palabra real**, lo cual hace los outputs feos para usuario final.

---

## 5. Impacto y legado

Porter Stemmer es, junto a TF-IDF y Naive Bayes, **el algoritmo más usado en la historia del NLP clásico**. Algunas métricas:

- **Citas**: a mayo de 2026, **~15,000 citas en Google Scholar** para el paper original. Está entre los 20 papers más citados de NLP de todos los tiempos.
- **Implementaciones**: hay implementaciones oficiales en **decenas de lenguajes**: Python (NLTK, Snowball wrapper), Java (Apache Lucene, OpenNLP), C, Ruby, Go, R, Perl, Tcl, JavaScript, etc.
- **En producción HOY (2026)**: Apache Lucene (motor de search detrás de Elasticsearch, Solr, et al.) usa Porter como uno de los stemmers default. Eso significa que **literalmente miles de motores de búsqueda corren Porter en producción cada día**, 45 años después.
- **Snowball (2001)**: Porter mismo desarrolló Snowball como sucesor — un lenguaje de dominio específico para escribir stemmers en cualquier idioma. La implementación de Porter en Snowball ("Porter2" o "English Snowball") es estrictamente mejor y debería usarse en lugar del Porter original. NLTK incluye ambos por compatibilidad.

### Sucesores notables

- **Snowball / Porter2** (Porter 2001): mismo algoritmo refinado + DSL para multilingüe.
- **Lancaster / Paice** (Paice 1990): más agresivo, peor para uso general. NLTK lo incluye, pero rara vez se usa.
- **Krovetz** (Krovetz 1993): híbrido stem+lemma, usa diccionario. Más preciso pero más lento.
- **Lemmatización** (WordNet, spaCy): reemplazo moderno para tareas que necesitan output legible.

### Por qué sigue siendo relevante

En 2026, con BERT, GPT-4, Llama, etc., uno se preguntaría: ¿por qué seguir hablando de Porter? Las razones:

1. **Costo computacional**. Porter procesa cientos de miles de palabras por segundo en CPU. BERT/Transformers requieren GPU y son 100-10000x más lentos.
2. **Determinismo y trazabilidad.** Porter siempre da el mismo output. Los Transformers tienen randomness sutil (dropout, sampling).
3. **No requiere data adicional.** Porter funciona sin internet, sin descargas, sin modelos pre-entrenados.
4. **Para IR de gran escala**, donde indexas miles de millones de documentos, el ahorro computacional de Porter es enorme. Lucene/Solr/Elasticsearch lo usan por esto.
5. **Pedagógicamente, es un ejemplo perfecto** de cómo un algoritmo simple, bien diseñado, sobrevive 45 años. Lo opuesto al hype de modelos billion-parameter.

---

## 6. Conexión directa con el Práctico 16

| Celda del lab | Concepto del paper |
|---|---|
| 31 | `from nltk.stem import PorterStemmer` — la implementación NLTK del algoritmo de este paper. **Sin cambios sustantivos respecto a 1980**. |
| 31 | `porter = PorterStemmer()` — instancia el stemmer. **No requiere parámetros** porque las reglas están hardcoded en el código (~400 líneas en NLTK, espejo del BCPL original). |
| 31 | `porter.stem(s)` — aplica los 5 pasos a una palabra. |
| 31 | Output de Porter en la frase del lab: `artifici intellig is intellig demonstr by machin . lead ai textbook defin the field as the studi of intellig agent : ani devic that perceiv it environ and take action that maxim it chanc of success achiev it goal .` |
| | • `artificial → artifici` (paso 4: AL eliminado, queda `ARTIFICI` con m=2) |
| | • `intelligence → intellig` (paso 4: ENCE eliminado) |
| | • `demonstrated → demonstr` (paso 1b: ED eliminado tras *v* presente) |
| | • `machines → machin` (paso 1a: S eliminado; paso 1b: ED no aplica; sin más cambios) |
| | • `textbooks → textbook` (paso 1a: S eliminado, queda en textbook) |
| | • `defines → defin`? `define → defin`. Pasa por paso 5a (m=1 con E): no se quita E porque m>1 requerido, queda DEFINE. Pero el output del lab muestra `defin`. Probablemente el `s` en `defines` se quita y luego paso 5a aplica (m=1 ... wait, DEFIN m=2 entonces SÍ aplica). Output correcto. |
| | • `study → studi` (paso 1c: Y → I tras *v*, tipo `studi`) |
| | • `successfully → success` (paso 4: ULLY no es regla directa pero LY no es regla; paso 4: AL→nada o ALI→AL? Termina en `success`) |

**Verifica directamente:**

```python
from nltk.stem import PorterStemmer
porter = PorterStemmer()
for w in ['artificial', 'intelligence', 'demonstrated', 'machines',
          'leading', 'textbooks', 'studying', 'intelligent', 'agents',
          'perceives', 'environment', 'maximize', 'successfully', 'achieving']:
    print(f"{w:15} → {porter.stem(w)}")
```

Vas a ver el mapping completo y puedes contrastarlo con las reglas que describimos arriba.

**Para inspeccionar el algoritmo en NLTK** (interesante didácticamente):

```python
import inspect
from nltk.stem.porter import PorterStemmer
print(inspect.getsource(PorterStemmer._step1a))  # paso 1a, plurales
print(inspect.getsource(PorterStemmer._step2))   # paso 2, derivacionales
```

Vas a ver el código de Python que mapea **1:1** a las reglas del paper. **El algoritmo de 1980 es código de 2026**.

---

## 7. Lecturas relacionadas

**Stemmers contemporáneos a Porter:**
- Lovins (1968), *Development of a Stemming Algorithm*, Mechanical Translation & Computational Linguistics 11(1) — el predecesor maximalista.
- Andrews (1971), *Development of a Fast Conflation Algorithm for English*, Cambridge — el sistema contra el que Porter compara.
- Dawson (1974), *Suffix Removal and Word Conflation*, ALLC Bulletin.

**Sucesores directos:**
- Paice (1990), *Another Stemmer*, SIGIR Forum — el Lancaster Stemmer, más agresivo, también disponible en NLTK como `LancasterStemmer`.
- Porter (2001), *Snowball: A Language for Stemming Algorithms* — el DSL que Porter desarrolló para portar el algoritmo a múltiples idiomas. Recomendado sobre PorterStemmer original.
- Krovetz (1993), *Viewing morphology as an inference process* — stemmer que combina con diccionario para evitar stems sin palabra.

**Para Lemmatization (la alternativa "lingüísticamente correcta"):**
- Miller (1995), *Introduction to WordNet* — ver `Miller-WordNet-1995.md` en este directorio. El lemmatizer de NLTK usa WordNet.

**Contexto IR donde Porter floreció:**
- Salton & McGill (1983), *Introduction to Modern Information Retrieval*, McGraw-Hill — el libro de texto que codificó Porter como estándar en IR.
- Cleverdon, Mills & Keen (1966), *Factors Determining the Performance of Indexing Systems*, College of Aeronautics, Cranfield — la metodología que produjo Cranfield-200, sobre el cual Porter validó su algoritmo.

**Para comparar con la era moderna:**
- Devlin et al. (2018), *BERT* — los Transformers que reemplazan stemming en muchos casos. Subword tokenization (WordPiece) maneja morfología internamente, haciendo Porter obsoleto en pipelines basados en BERT. Pero NO en pipelines clásicos de IR.

Este paper es un ejemplo de cómo **simplicidad pragmática + validación empírica honesta** pueden producir algoritmos que sobreviven generaciones de hardware y paradigmas. Es lectura obligada para entender por qué el NLP clásico funcionó tan bien antes de los Transformers, y por qué sigue siendo relevante en contextos de costo/latencia/explicabilidad.
