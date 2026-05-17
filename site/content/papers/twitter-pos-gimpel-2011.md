---
title: "Part-of-Speech Tagging for Twitter"
weight: 95
math: true
---

{{< paper-card
    title="Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments"
    authors="Gimpel, Schneider, O'Connor, Das, Mills, Eisenstein, Heilman, Yogatama, Flanigan, Smith"
    year="2011"
    venue="ACL 2011 (Short Papers)"
    pdf="/papers/twitter-pos-gimpel-2011.pdf" >}}
POS tagger especializado para texto de Twitter, con un **tagset de 25 etiquetas** que incluye categorías Twitter-específicas (`#` hashtag, `@` mention, `~` discourse marker, `U` URL, `E` emoticon). Corpus anotado de 1,827 tweets, ~89.4% accuracy con CRF (vs 85.8% del Stanford POS tagger reentrenado en el mismo dataset). El **tokenizer especializado** que diseñaron fue posteriormente portado a NLTK como `TweetTokenizer`.
{{< /paper-card >}}

---

## Contexto

A inicios de los 2010, el NLP académico recién empezaba a tomar Twitter en serio como fuente de datos. La explosión de uso (Twitter alcanzó 200M de usuarios en 2010) y el acceso via API generó interés en analizar tweets para predicción de elecciones, detección de sentimientos, predicción de mercados, resumen automático, detección de eventos.

**El problema**: la mayoría de estos trabajos saltaban directo a clasificación sin preprocesamiento lingüístico, porque el pipeline NLP clásico fallaba en Twitter. Finin et al. (2010) demostró que los POS taggers estándar (entrenados sobre Wall Street Journal) **caen ~25% accuracy** cuando se aplican a tweets.

Las razones específicas — y todas se relacionan con tokenización + POS:

| Característica de Twitter | Por qué rompe el pipeline estándar |
|---|---|
| Límite de 140 caracteres | Fuerza abreviaciones, contracciones no estándar |
| Ortografía no convencional | `nite`, `2nite`, `cooool`, `lmaooo` — lexicón no las cubre |
| Hashtags `#NLP` | Tokenizer estándar separa `#` de la palabra |
| At-mentions `@user` | Idem; siempre proper nouns en función |
| URLs `http://t.co/xyz` | Tokenizer estándar las explota en `["http", ":", "/", "/", ...]` |
| Emoticons `:-)`, `<3` | Puntuación destruye el emoticon |
| Capitalización inconsistente | Heurística "mayúsculas = proper noun" falla |

El equipo de Noah Smith en CMU decidió construir desde cero un POS tagger para Twitter. **200 person-hours, 17 personas, 2 meses**.

---

## Ideas principales

### 1. Tagset Twitter-específico (25 etiquetas)

Cada etiqueta es **un carácter ASCII único** para simplicidad. Las categorías:

**Nominales**:
- `N` common noun (13.7%) — `books, someone`
- `O` pronoun (6.8%) — `it, you, u, meeee`
- `^` proper noun (6.4%) — `lebron, usa, iPad`
- `L` nominal + verbal (1.6%) — `he's, book'll, iono`

**Open-class**:
- `V` verb incl. modals (15.1%) — `might, gonna, eats`
- `A` adjective (5.1%) — `good, fav, lil`
- `R` adverb (4.6%)
- `!` interjection (2.6%) — `lol, haha, FTW`

**Twitter-específicas** (la contribución más original):
- `#` hashtag (1.0%) — `#acl`
- `@` at-mention (4.9%) — `@BarackObama`
- `~` discourse marker (3.4%) — `RT` y `:` en retweets, `≪` separador
- `U` URL (1.6%) — `http://bit.ly/xyz`
- `E` emoticon (1.0%) — `:-) :b (: <3 o_O`

**Miscellaneous**: `$` numeral, `,` puntuación, `G` foreign words/símbolos.

Decisión clave: **NO separar contracciones**. Penn Treebank tokeniza `he's` como `["he", "'s"]`. Twitter tiene tantas contracciones no estándar (`iono` para "I don't know") que crearon **tags compuestos** (`L`, `M`) para nominal+verbal en lugar de separar.

### 2. El tokenizer especializado

Modificación del **TweetMotif tokenizer** de O'Connor et al. 2010b. Reglas clave:

1. **NO separar contracciones**: `he's`, `don't`, `iono` → un solo token.
2. **NO separar posesivos** cuando están pegados al nombre.
3. **Preservar emoticons** como tokens unitarios: `:-)`, `<3`, `o_O`.
4. **Preservar hashtags y at-mentions** completos.
5. **Preservar URLs**.
6. **Reconocer discourse markers** Twitter: `RT`, `:` en retweets, `≪`.

Este tokenizer fue **portado a NLTK** como `nltk.tokenize.casual.TweetTokenizer` y se volvió la herramienta de facto para tokenizar texto de redes sociales.

### 3. Modelo CRF con features Twitter-específicas

**CRF** (Conditional Random Field — Lafferty et al. 2001) como base.

**Features base**: word form, dígitos/guiones, suffixes hasta longitud 3, capitalization patterns. Solos dan **83.4% accuracy**.

**Features Twitter-específicas** (la innovación):

| Feature | Qué hace | Aporte ablation |
|---|---|---|
| **TWORTH** (Twitter orthography) | Regex para at-mentions, hashtags, URLs | +1.00% |
| **NAMES** (gazetteers) | Tokens frecuentemente capitalizados | +0.02% |
| **TAGDICT** (PTB tag dictionary) | Tags PTB que cada palabra tiene en newswire | +1.06% |
| **DISTSIM** (distributional similarity) | SVD sobre matriz de co-ocurrencia de tweets sin etiquetar | +1.06% |
| **METAPH** (phonetic normalization) | Metaphone keys: `thx/thanks/thanksss/thnx → ONKS` | +0.42% |

DISTSIM es **proto-word2vec** — 2 años antes del paper word2vec de Mikolov. Construyen embeddings de 50 dimensiones via SVD sobre transición sucesor/predecesor.

---

## Resultados

| Sistema | Test accuracy |
|---|---|
| Annotator agreement (ceiling) | 92.2% |
| **CRF con features completas** | **89.37%** |
| Stanford tagger (reentrenado en mismo corpus) | 85.85% |
| Base features only | 83.38% |

**Reducción relativa de error: 25%** vs Stanford. Con solo 500 tweets de training, accuracy cae apenas 1.7% — la curva de aprendizaje se aplana rápido por la riqueza de los features.

**Análisis por tag**: tags Twitter-específicas (`@` 99%, `U` 97%, `#` 89%, `E` 88%) están entre las mejores. **Proper nouns** son el punto débil (71% recall) — la capitalización inconsistente confunde al sistema.

---

## Limitaciones reconocibles

- **Proper nouns con capitalización no estándar**: "obama" sin mayúscula confunde.
- **Categoría G** (garbage) es heterogénea, peor recall (26%).
- **Solo inglés** (corpus filtrado a usuarios con UI inglés, timezone USA).
- **Dataset pequeño**: 1,827 tweets es poco; hoy se haría con 10-100k.
- **No usa contexto largo**: CRF mira features locales; un Transformer captura más.

---

## Por qué importa hoy

El paper tuvo **impacto desproporcionado al tamaño** (6 páginas) por liberar recursos abiertos:

- **Tagset Twitter de Gimpel** es el **primer tagset estándar para redes sociales** — referencia obligatoria para trabajos posteriores.
- **Corpus anotado** (1,827 tweets) reusado en docenas de papers.
- **TweetTokenizer en NLTK**: la herramienta de facto para tokenizar tweets en Python.
- **Features METAPH y DISTSIM** anticipan ideas formalizadas con word2vec (2013).
- **~700 citas en Google Scholar**, pero impacto vía herramientas reusadas es masivo.

**Sucesor directo**: Owoputi et al. (2013), *Improved Part-of-Speech Tagging for Online Conversational Text with Word Clusters* — mismo equipo CMU, accuracy 93.4% con Brown clusters.

En la era post-BERT (2018+), modelos contextuales superan a este sistema con datasets más grandes (`cardiffnlp/twitter-roberta-base` alcanza ~95%+). Pero **para CPU only sin GPU**, el sistema Gimpel sigue siendo baseline rápido válido.

---

## Notas y enlaces

- Repositorio original: `github.com/brendano/tweetmotif` (TweetMotif tokenizer original).
- TweetNLP project: `ark.cs.cmu.edu/TweetNLP`.
- Versión NLTK: `from nltk.tokenize import TweetTokenizer`.
- Parámetros útiles: `TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)`.

Ver fundamentos: [Tokenización clásica](/fundamentos/tokenizacion-clasica) · [Sentiment Analysis](/fundamentos/sentiment-analysis). Ver paper relacionado: [VADER](/papers/vader-hutto-gilbert-2014).
