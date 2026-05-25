---
title: "Punkt - Unsupervised Multilingual Sentence Boundary Detection"
weight: 164
math: true
---

{{< paper-card
    title="Unsupervised Multilingual Sentence Boundary Detection"
    authors="Kiss, Strunk"
    year="2006"
    venue="Computational Linguistics"
    pdf="/papers/punkt-kiss-strunk-2006.pdf" >}}
Propone **Punkt**, un sistema no supervisado y portátil entre idiomas para detectar fronteras de oración sin listas de abreviaciones, sin POS tagger ni corpus anotado. Reformula el problema como **detección de abreviaciones** usando un log-likelihood ratio modificado y refinamiento por contexto en dos etapas (type-based + token-based). Validado en **11 idiomas** con error medio de **1.26%**, se convirtió en el sentence tokenizer por defecto de NLTK (`nltk.sent_tokenize`).
{{< /paper-card >}}

---

## Contexto

A inicios de los 2000, la segmentación de oraciones se consideraba "casi resuelta", pero los sistemas existentes dependían de dos supuestos frágiles:

1. **Listas de abreviaciones pre-compiladas por humanos** (`Mr.`, `Dr.`, `U.S.A.`, etc.).
2. **Corpus etiquetado a mano** con marcas explícitas de fin de oración.

Esto bastaba para inglés newswire en el dominio entrenado, pero fallaba en:

- **Idiomas distintos del inglés**: cada uno tiene su propio inventario de abreviaciones.
- **Géneros distintos** (clínico, jurídico, conversacional): cada uno trae abreviaciones nuevas.
- **Texto en mayúsculas o minúsculas únicas** (single-case): los sistemas basados en capitalización colapsan.
- **Corpora nuevos**: cada vez que aparece un dominio nuevo, hay que volver a anotar.

El estado del arte previo era una colección de sistemas con sus límites:

| Sistema | Año | Tipo | Limitación principal |
|---|---|---|---|
| **Riley** (decision tree) | 1989 | Supervisado | Requiere 25M de palabras de AP newswire etiquetadas |
| **Satz** (Palmer & Hearst) | 1997 | Supervisado (NN o DT) | Requiere lexicón POS + lista de abreviaciones manual |
| **MxTerminator** (Reynar & Ratnaparkhi) | 1997 | Maximum entropy supervisado | Requiere 39k oraciones WSJ anotadas |
| **RE system** (Silla et al.) | 2003 | Basado en reglas | Requiere ~700 abreviaciones manuales por idioma |
| **Stamatatos et al.** | 1999 | Transformation-based learning (Brill) | Supervisado |
| **Grefenstette** | 1999 | Heurísticas + lexicón completo | Requiere lexicón exhaustivo del idioma |

Kiss y Strunk plantearon la pregunta clave: **¿es posible detectar fronteras de oración sin entrenamiento supervisado, sin listas de abreviaciones, sin POS tagger, y de manera portátil entre idiomas?** La respuesta de este paper: **sí**, con un sistema de dos etapas basado en estadísticas colocacionales.

---

## Ideas principales

### 1. Reformulación del problema

El punto `.` tiene **tres roles ambiguos** en escritura:

1. **Marcador de fin de oración**: "Volvió a casa. Cenó."
2. **Marcador de abreviación**: "Dr. Smith fue al U.S.A. ayer."
3. **Decimal o número ordinal**: "3.14", "1990s".

Los autores demuestran que **hasta el 30% de los puntos en texto típico son ambiguos**. Si detectas correctamente cuáles son abreviaciones, **todo lo demás** es frontera de oración por descarte. Por eso el sistema se concentra en **detectar abreviaciones** primero, y deja sentence boundary detection como una consecuencia.

### 2. Etapa 1 - Type-based classification

Una *abbreviation type* es una palabra que aparece con punto en una proporción muy alta de sus ocurrencias y tiene además ciertas propiedades físicas. Punkt formaliza esto con **cuatro factores cuantificables**:

**Factor 1 - Collocational bond (log-likelihood ratio modificado)**

Una abreviación está **fuertemente asociada al punto que la sigue**. Punkt usa una versión modificada del log-likelihood ratio de Dunning (1993):

$$H_0: P(\bullet \mid w) = P(\bullet) \quad \text{(el punto es independiente de la palabra)}$$

$$H_A: P(\bullet \mid w) = 0.99 \quad \text{(el punto sigue casi siempre a w)}$$

La hipótesis alternativa **no es "más frecuente que la media"** sino **"casi 1"**. Esta es una **innovación clave** sobre Dunning original: las abreviaciones se caracterizan porque **virtualmente nunca aparecen sin punto**.

**Factor 2 - Brevity (longitud)**

Las abreviaciones tienden a ser cortas. En vez de un cutoff duro, Punkt usa un factor exponencial inverso:

$$F_{\text{length}}(w) = \frac{1}{\exp(\text{length}(w))}$$

donde `length(w)` excluye los puntos internos (porque "U.S.A." es corta a pesar de tener 5 caracteres). Si w tiene 3 caracteres, $F \approx 0.05$; si tiene 10, $F \approx 0.00005$.

**Factor 3 - Internal periods (puntos internos)**

Muchas abreviaciones contienen puntos internos: `U.S.A.`, `i.e.`, `e.g.`, `Ph.D.`. Esto es **evidencia muy fuerte**:

$$F_{\text{periods}}(w) = 1 + \text{número de puntos internos en } w$$

**Factor 4 - Penalty para ocurrencias sin punto**

Si una palabra **a veces aparece sin punto**, hay evidencia de que NO es abreviación:

$$F_{\text{penalty}}(w) = \frac{1}{\text{length}(w)^{C(w, \neg\bullet)}}$$

donde $C(w, \neg\bullet)$ es el conteo de ocurrencias sin punto final.

**Combinación final:**

$$\text{score}(w) = \log \lambda(w) \cdot F_{\text{length}}(w) \cdot F_{\text{periods}}(w) \cdot F_{\text{penalty}}(w)$$

Con threshold de **0.3**: si `score(w) ≥ 0.3`, w se marca como abreviación. **Los autores derivaron este threshold sobre inglés y lo mantuvieron fijo para los otros 10 idiomas**.

### 3. Etapa 2 - Token-based reclassification

Después de la etapa 1, cada token con punto se marca tentativamente como:

- `<A>` si su tipo es abreviación.
- `<E>` si es elipsis (`...`).
- `<S>` si es frontera de oración (lo que sobra).

La etapa 2 mira el contexto para corregir y refinar con tres heurísticas:

**Heurística ortográfica**: para un token después de un período, Punkt mira la palabra siguiente. Si es siempre mayúscula en el corpus (e.g., nombres propios), no es evidencia clara; si a veces aparece minúscula al principio de oración pero nunca minúscula dentro, sugiere frontera. Evita el ingenuo "mayúscula = nueva oración" que falla en alemán (todos los sustantivos son mayúsculos).

**Heurística colocacional**: si dos palabras alrededor de un punto forman una colocación (aparecen juntas más de lo esperado por azar), el punto probablemente no es frontera. Ejemplo: "St. Petersburg" forma colocación; "casa. Vino" no.

**Frequent Sentence Starter**: para cada palabra que aparece después de un `<S>`, Punkt calcula si es un "sentence starter típico" (e.g., "However", "The", "Moreover"). Si la palabra siguiente lo es, hay evidencia adicional de frontera.

**Tratamiento especial**: iniciales (`A.`, `B.`) se detectan por longitud 1 + capitalización siguiente; ordinales numéricos (`1.`, `2.`) reciben tratamiento especial para idiomas que los marcan con punto (alemán, sueco, noruego).

### 4. Arquitectura completa

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

## Resultados experimentales

### Corpora de evaluación

Probaron 11 idiomas, todos sobre corpora de periódicos (newspaper):

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

### Resultado headline (Tabla 21)

Error rate de detección de frontera de oración (menor es mejor):

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

Punkt **gana en 9 de 11 idiomas** contra MxTerminator (sistema supervisado entrenado por idioma con miles de oraciones etiquetadas) y empata en los otros 2.

### Ablation de las heurísticas (Tabla 19)

| Sistema | Estonian | German | French | Mean approx |
|---|---|---|---|---|
| A (solo type-based) | 7.37% | 2.38% | 2.94% | ~2.8% |
| B (+ collocation) | 2.94% | 0.47% | 2.61% | ~1.7% |
| C (+ frequent sentence starter) | 2.80% | 0.42% | 1.96% | ~1.6% |
| D (+ orthographic) | 2.18% | 0.36% | 1.78% | ~1.4% |
| E (+ special orthographic for initials, completo) | 2.12% | 0.35% | 1.54% | ~1.3% |

La **heurística colocacional** es la que más aporta (-1.1% promedio). La etapa 2 token-based **reduce el error a la mitad** comparado con solo etapa 1.

### Independencia del threshold (Tabla 20)

Threshold de 0.3 (derivado en inglés) vs threshold óptimo por idioma:

| Idioma | Punkt @ 0.3 | Punkt @ óptimo | Diff |
|---|---|---|---|
| Italian | 1.13% | 1.00% (0.2) | 0.13% |
| French | 1.54% | 1.51% (0.2) | 0.03% |
| English | 1.65% | 1.59% (0.2) | 0.06% |
| Estonian | 2.12% | 2.10% (0.6) | 0.02% |
| Average | — | — | **0.03%** |

**El threshold 0.3 derivado de inglés es casi óptimo para los otros 10 idiomas**. Diferencia promedio: 0.03%. Esto sugiere que el método es **genuinamente language-independent** — los autores no inflaron el resultado con ajustes manuales por idioma.

### Análisis de errores remanentes

Los autores identifican 5 tipos de error que Punkt no maneja bien:

1. **Homografía**: "in" es preposición frecuente Y abreviación de "inch". Como el tipo "in" aparece sin punto la mayoría del tiempo, Punkt lo clasifica como no-abreviación.
2. **Uso inconsistente**: en sueco "osv." (and so on) a veces tiene punto, a veces no. Si la versión sin punto domina, Punkt no la detecta.
3. **Data sparseness**: abreviaciones que aparecen 1-2 veces no acumulan evidencia colocacional suficiente.
4. **Falta de evidencia ortográfica**: si la palabra que sigue a una abreviación es nombre propio (siempre capitalizado), la heurística ortográfica no decide.
5. **Estructura de texto**: titulares no terminan con punto, listas tienen ordinales, etc. Punkt no modela estructura de documento.

---

## Limitaciones

1. **Asume sistema de escritura alfabético con punto como marcador de fin de oración.** No funciona para chino (no usa espacios), tailandés, árabe sin diacríticos, etc.
2. **No modela estructura de documento.** Titulares sin punto, listas numeradas, código, URLs — todo lo trata como prosa continua.
3. **Necesita un corpus para aprender.** Es no supervisado pero no zero-shot: necesita ver suficiente texto para acumular estadísticas. Para corpora muy pequeños (<10k tokens) la calidad cae.
4. **Verb-final languages** (turco, japonés) son problemáticas porque ciertos verbos cierran casi todas las oraciones; el sistema puede confundirlos con abreviaciones.
5. **Homografía abreviation/palabra ordinaria** (caso "in" / "in.").
6. **No detecta sentencias dentro de oraciones**: si una oración compleja tiene una sub-cláusula que en otro contexto sería oración independiente, Punkt no la separa.

Limitaciones no discutidas pero importantes hoy:

7. **No es contextual.** Un Transformer fine-tuned para sentence boundary detection (e.g., basado en BERT) supera fácilmente a Punkt en accuracy, especialmente en texto difícil (medical, legal).
8. **Idiomas de bajos recursos.** Lenguas con poca data digital (lenguas indígenas, africanas) no tienen corpora suficientes para entrenar Punkt eficientemente.
9. **Code-switching.** Texto que mezcla idiomas (común en redes sociales) confunde las estadísticas de un solo idioma.

---

## Por qué importa hoy

Punkt es **el sentence tokenizer más usado del NLP clásico**. Algunos hechos:

- **NLTK lo adoptó** como `nltk.sent_tokenize` desde la versión 0.9 (2007) y sigue siendo el default en 2026. Modelos pre-entrenados disponibles en 18 idiomas vía `nltk.download('punkt')` (y desde NLTK 3.8.2 también `'punkt_tab'`).
- **Citas**: a mayo de 2026, ~1700 citas en Google Scholar para el journal paper de 2006, más miles adicionales para las versiones de workshop/conference de 2002.
- **spaCy** usa un sentence segmenter más moderno por defecto (basado en dependency parsing), pero ofrece Punkt como alternativa para texto sin POS info.
- **scikit-learn** no incluye sentence segmentation propio — los usuarios típicamente usan Punkt vía NLTK.
- En **producción industrial**, Punkt sigue corriendo en miles de pipelines NLP que precedieron a la era Transformer: clasificación de documentos, IR, extracción de información clásica, OCR post-processing.
- Su **diseño type-based + token-based** influenció arquitecturas posteriores de IE y NER, donde primero se aprende un vocabulario y luego se reclasifica por instancia.

El paper también es ejemplo de **buena ciencia metodológica** en NLP:

- Threshold derivado de un solo idioma, validado en 10 más sin tuning.
- Comparación honesta contra baselines supervisados (Punkt no siempre gana, y los autores lo reportan).
- Análisis exhaustivo de errores remanentes en lugar de inflar las métricas.

Filosóficamente, Punkt encarna una lección clave del NLP pre-Transformer: **un modelo estadístico simple con buenas inductive biases lingüísticas puede superar a sistemas supervisados más complejos cuando el problema está bien planteado**. Esa lección sigue vigente cuando hoy se decide entre pipelines clásicos baratos y modelos neuronales caros.

---

## Notas y enlaces

- **Clase asociada**: [Clase 16 - NLP clásico, NLTK, BoW, embeddings](/clases/clase-16).
- **Laboratorio asociado**: [Lab 16 - Pipeline NLP con NLTK/spaCy/NLLB/VADER](/laboratorios/lab-16).
- **Fundamento relacionado**: [Tokenización clásica](/fundamentos/tokenizacion-clasica).
- **Cita BibTeX**:

```bibtex
@article{kiss2006unsupervised,
  title={Unsupervised multilingual sentence boundary detection},
  author={Kiss, Tibor and Strunk, Jan},
  journal={Computational Linguistics},
  volume={32},
  number={4},
  pages={485--525},
  year={2006}
}
```
