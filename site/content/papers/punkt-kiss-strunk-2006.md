---
title: "Punkt — Unsupervised Multilingual Sentence Boundary Detection"
weight: 75
math: true
---

{{< paper-card
    title="Unsupervised Multilingual Sentence Boundary Detection"
    authors="Kiss, Strunk"
    year="2006"
    venue="Computational Linguistics 32(4)"
    pdf="/papers/punkt-kiss-strunk-2006.pdf" >}}
Algoritmo **no supervisado** para segmentar texto en oraciones, basado en detectar **abreviaciones como collocations entre word y punto**. Probado en 11 idiomas (incluyendo español, alemán, francés, italiano, portugués, sueco, turco) con error medio de **1.26%** sin re-entrenar nada. Usa un único threshold derivado de inglés que funciona casi óptimo en los otros 10 idiomas. **Es el sentence tokenizer default de NLTK** (`nltk.sent_tokenize`) y la base de los modelos descargados con `nltk.download('punkt')` en 18 idiomas.
{{< /paper-card >}}

---

## Contexto

A inicios de los 2000, los sentence tokenizers existentes requerían:
- **Listas de abreviaciones pre-compiladas** (`Mr.`, `Dr.`, `U.S.A.`, etc.) por humanos.
- **Corpora etiquetados a mano** con fronteras de oración.

Esto funcionaba para inglés newswire pero fallaba en idiomas distintos, géneros distintos, texto single-case, y corpora nuevos sin anotar. Cada caso requería re-entrenar o curar listas manuales.

Kiss y Strunk preguntaron: **¿es posible detectar fronteras sin entrenamiento supervisado, sin listas manuales, portátil entre idiomas?** Su respuesta: sí, con un sistema basado en estadísticas colocacionales.

---

## Ideas principales

### 1. Reformulación del problema: detectar abreviaciones

El punto `.` tiene tres roles ambiguos:
1. Fin de oración: `"Volvió a casa. Cenó."`
2. Abreviación: `"Dr. Smith fue al U.S.A."`
3. Decimal o ordinal: `"3.14"`, `"1990s"`

**Hasta el 30% de los puntos en texto típico son ambiguos**. Si detectás correctamente cuáles son abreviaciones, **todo lo demás es frontera de oración por descarte**.

Por eso Punkt se concentra en detectar abreviaciones, no en detectar fronteras directamente.

### 2. Las 3 propiedades de las abreviaciones

Una abreviación se caracteriza por **tres propiedades cuantificables** del candidato (no del contexto):

#### Propiedad 1 — Collocational bond

Una abreviación está **fuertemente asociada al punto que la sigue**. Punkt usa una versión modificada del log-likelihood ratio de Dunning (1993):

$$H_0: P(\bullet | w) = P(\bullet)$$
$$H_A: P(\bullet | w) = 0.99$$

La hipótesis alternativa no es "más frecuente que la media" sino **"casi 1"**. Esta es la innovación clave: las abreviaciones se caracterizan porque virtualmente nunca aparecen sin punto.

#### Propiedad 2 — Brevity

Las abreviaciones tienden a ser cortas. Pero Punkt no usa un cutoff duro — usa factor exponencial inverso:

$$F_{\text{length}}(w) = \frac{1}{e^{\text{length}(w)}}$$

Penaliza fuertemente palabras largas pero permite excepciones (una palabra larga puede ser abreviación si tiene `log λ` enorme).

#### Propiedad 3 — Internal periods

Abreviaciones con puntos internos (`U.S.A.`, `i.e.`, `Ph.D.`) son **fuerte evidencia**:

$$F_{\text{periods}}(w) = 1 + \text{número de puntos internos en } w$$

### 3. Arquitectura en dos etapas

```
┌──────────────────────────────────────┐
│ ETAPA 1: Type-based classification   │
│  • Collocational bond (log λ)         │
│  • Length penalty                      │
│  • Internal periods bonus              │
│  • No-period penalty                   │
│  → Anotación: <A>, <E>, <S>           │
└────────────────┬─────────────────────┘
                 ▼
┌──────────────────────────────────────┐
│ ETAPA 2: Token-based reclassification │
│  • Orthographic heuristic              │
│  • Collocation heuristic               │
│  • Frequent sentence starter           │
│  • Initials + ordinals                 │
│  → Anotación final: <A><S>, <E><S>    │
└──────────────────────────────────────┘
```

La etapa 2 refina por instancia: examina si una abreviación específica también es **fin de oración** (`<A><S>`), si la palabra siguiente es siempre mayúscula (no decide), si forma collocation con la palabra previa al punto, etc.

### 4. Threshold único multilingüe

Punkt usa **threshold de 0.3** para clasificar como abreviación. Derivado empíricamente sobre inglés, **retenido para los otros 10 idiomas sin tuning**. Diferencia entre threshold óptimo por idioma y 0.3 fija: **0.03% promedio** — el método es genuinamente language-independent.

---

## Resultados experimentales

**11 idiomas evaluados** sobre corpora de periódicos:

| Idioma | Punkt | MxTerminator (supervisado) |
|---|---|---|
| German | **0.35%** | 0.63% |
| Norwegian | **0.81%** | 1.34% |
| Italian | **1.13%** | 2.45% |
| French | **1.54%** | 2.66% |
| Spanish | **1.06%** | 1.60% |
| English | 1.65% | 1.53% |
| Dutch | **0.97%** | 1.13% |
| Estonian | **2.12%** | 2.79% |
| Swedish | 1.76% | 2.39% |
| Turkish | **1.31%** | 1.77% |
| **Mean** | **1.26%** | 1.76% |

Punkt **gana en 9 de 11 idiomas** contra MxTerminator (sistema supervisado entrenado con miles de oraciones por idioma).

Ablation de heurísticas token-based: la **colocacional** es la más impactante (reduce error en Estonian de 7.37% → 2.94%). La etapa 2 reduce el error a la mitad respecto a solo etapa 1.

---

## Limitaciones reconocibles

- **Asume sistema de escritura alfabético** con punto como marcador de oración. No funciona en chino, tailandés, árabe sin diacríticos.
- **No modela estructura de documento**: titulares sin punto, listas, URLs son problemáticos.
- **Homografía abreviación/palabra ordinaria**: `in` es preposición frecuente Y abreviación de "inch".
- **Verb-final languages** (turco, japonés) son problemáticas porque verbos comunes cierran muchas oraciones.
- **Idiomas low-resource**: necesita suficiente corpus para acumular estadísticas (~10k tokens mínimo).

Limitaciones más visibles hoy: no captura sentencias dentro de oraciones complejas; no maneja code-switching multilingüe en un solo texto.

---

## Por qué importa hoy

- **NLTK lo adoptó** como `nltk.sent_tokenize` desde la versión 0.9 (2007) — sigue siendo el default en 2026.
- Modelos pre-entrenados disponibles en **18 idiomas** vía `nltk.download('punkt')` (y desde NLTK 3.8.2, también `'punkt_tab'`).
- **~1700 citas en Google Scholar** + miles adicionales para las versiones workshop de 2002.
- spaCy usa un sentence segmenter alternativo (dependency parser based) pero ofrece Punkt como opción.
- En **producción industrial** corre en miles de pipelines NLP pre-Transformer: clasificación de documentos, IR, extracción clásica, OCR post-processing.

Es ejemplo de **buena ciencia metodológica** en NLP: threshold derivado en un idioma y validado en 10 más sin tuning; comparación honesta contra baselines supervisados; análisis exhaustivo de errores remanentes.

---

## Notas y enlaces

- El sistema se llama **Punkt** (alemán: "punto") — los autores son Tibor Kiss y Jan Strunk de la Ruhr-Universität Bochum.
- Para **entrenar tu propio Punkt** sobre corpus específico (e.g., texto clínico con abreviaciones médicas), usá `nltk.tokenize.punkt.PunktTrainer().train(text)`.
- Inspeccionar abreviaciones aprendidas: `tokenizer._params.abbrev_types`.
- Versión anterior del mismo trabajo: Kiss & Strunk (2002a/2002b), KONVENS workshops.

Ver fundamentos: [Tokenización clásica](/fundamentos/tokenizacion-clasica) · [Bag of Words](/fundamentos/bag-of-words).
