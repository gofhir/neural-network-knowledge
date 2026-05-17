---
title: "Bag of Words"
weight: 285
math: true
---

**Bag of Words (BoW)** es la representación más antigua de texto como vector numérico, y aunque tiene 70+ años, **sigue siendo la baseline obligatoria** en NLP clásico y un componente operativo en miles de pipelines IR en producción. Representa cada documento como un vector de **conteos de palabras** del vocabulario global, descartando completamente el orden, la sintaxis, y el contexto.

Es **trivial conceptualmente** pero esconde decisiones de diseño que afectan dramáticamente al modelo downstream. Este fundamento cubre BoW puro y sus extensiones (n-grams, TF-IDF).

---

## 1. La idea fundamental

Dado un corpus de documentos $D = \{d_1, d_2, \ldots, d_N\}$:

1. **Construir el vocabulario** $V = \{w_1, w_2, \ldots, w_K\}$: el conjunto de palabras únicas que aparecen en algún documento.
2. **Representar cada documento** $d_i$ como un vector $\mathbf{x}_i \in \mathbb{N}^K$ donde $x_{i,j}$ = cantidad de veces que $w_j$ aparece en $d_i$.

```
documentos:
  d1 = "I love pizza"
  d2 = "Pizza is great"
  d3 = "I hate spam"

vocabulario (tras tokenizar + lowercase):
  V = ['great', 'hate', 'i', 'is', 'love', 'pizza', 'spam']

vectores BoW:
            great  hate   i   is   love  pizza  spam
  d1:        [0,    0,    1,   0,   1,    1,    0]
  d2:        [1,    0,    0,   1,   0,    1,    0]
  d3:        [0,    1,    1,   0,   0,    0,    1]
```

**Cada fila es un documento. Cada columna es una palabra del vocabulario. Cada celda es un conteo.**

El nombre "Bag of Words" (bolsa de palabras) refleja la metáfora: imaginá que metés cada palabra del documento en una bolsa. La bolsa solo recuerda **cuántas de cada tipo** hay, no en qué orden estaban.

---

## 2. Origen histórico

El término aparece en **Zellig Harris (1954)**, *"Distributional Structure"*, Word — paper foundacional de la hipótesis distribucional: *"the meaning of a word is given by the company it keeps"*. Harris no usa el nombre "Bag of Words" pero formaliza la idea de representar texto por co-ocurrencias.

El uso operativo se populariza en los 50-60 con **information retrieval** (Salton et al. en SMART system). En el 70-80 se vuelve estándar con la introducción de **TF-IDF** (Salton & Buckley 1988).

El primer uso explícito del término "bag of words" en literatura computacional es debatido pero aparece consistentemente en libros de IR desde los 90.

---

## 3. Por qué BoW funciona (a pesar de descartar todo)

A primera vista, descartar el orden parece destructivo. Pero BoW funciona razonablemente bien por tres razones:

**A. La estadística de palabras es informativa**. Un documento sobre cocina **tiene** palabras como `harina`, `sal`, `cebolla` con frecuencia distintiva. Un documento médico **tiene** `paciente`, `presenta`, `tratamiento`. La distribución marginal de palabras ya contiene mucha señal de tema.

**B. Para clasificación temática, el orden importa poco**. Las palabras clave determinan el tema casi independiente de su posición.

**C. Para corpus grandes, las correlaciones promedio se imponen**. Un clasificador Naive Bayes asume independencia entre palabras (claramente falso), pero las correlaciones sistemáticas en agregado permiten al modelo aprender patrones distintivos.

Donde BoW **falla**:

- **Tareas que dependen del orden** (sentiment con negación, parsing, comprensión sintáctica).
- **Distinción de frases con mismas palabras**: `"Me gusta mucho el pan"` vs `"Me gusta el pan, mucho"`.
- **Anagramas semánticos**: `"perro muerde hombre"` vs `"hombre muerde perro"`.

Para esos casos necesitás **n-grams** (captura orden local) o modelos contextuales (BERT, RNNs).

---

## 4. Implementación con scikit-learn

`CountVectorizer` es la implementación canónica:

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love pizza", "Pizza is great", "I hate spam"]
vec = CountVectorizer()
X = vec.fit_transform(texts)

print(vec.get_feature_names_out())
# ['great' 'hate' 'is' 'love' 'pizza' 'spam']

print(X.toarray())
# [[0 0 0 1 1 0]
#  [1 0 1 0 1 0]
#  [0 1 0 0 0 1]]
```

**Parámetros importantes**:

| Parámetro | Default | Significado |
|---|---|---|
| `max_features` | None | Quedarse con las K palabras más frecuentes (recorta el vocab). |
| `min_df` | 1 | Mínimo de documentos donde la palabra debe aparecer. |
| `max_df` | 1.0 | Máximo % de documentos donde puede aparecer (filtra demasiado frecuentes). |
| `ngram_range` | (1, 1) | Generar también bigramas, trigramas, etc. |
| `stop_words` | None | Lista de stopwords a filtrar. |
| `lowercase` | True | Convertir a minúscula antes de tokenizar. |
| `token_pattern` | `r"(?u)\b\w\w+\b"` | Patrón regex para tokens. Requiere 2+ alfanuméricos por default. |

**Decisión típica de producción**:

```python
CountVectorizer(
    max_features=15000,
    min_df=5,                  # filtrar hápax y casi-hápax
    max_df=0.95,               # filtrar palabras casi universales
    ngram_range=(1, 2),        # unigrams + bigrams
    lowercase=True,
)
```

---

## 5. La sparsity es inherente

Las matrices BoW son **>99% ceros**. Para un corpus de 5,000 documentos × 10,000 features:

- Matriz densa: ~440 MB en RAM.
- Matriz sparse (CSR): ~2 MB en RAM.

scikit-learn devuelve `scipy.sparse.csr_matrix` por default. **No conviertas a denso** (`.toarray()`) salvo que sea estrictamente necesario.

**Clasificadores que aceptan sparse**: `MultinomialNB`, `BernoulliNB`, `LogisticRegression`, `LinearSVC`, `LinearSVR`, `SGDClassifier`.

**Clasificadores que requieren denso**: `GaussianNB`, modelos basados en árboles típicamente (aunque `XGBoost` y `LightGBM` aceptan sparse).

---

## 6. N-grams: la mejora natural

Bag of Words puro pierde orden. **Bag of N-grams** captura orden **local** generando tokens compuestos por N palabras adyacentes:

| Frase | Unigrams (BoW) | Bigrams | Trigrams |
|---|---|---|---|
| `"el gato come pescado"` | el, gato, come, pescado | el gato, gato come, come pescado | el gato come, gato come pescado |

Con `ngram_range=(1, 3)`, `CountVectorizer` genera **unigrams + bigrams + trigrams** todos juntos como features.

**Trade-off**:
- N-grams más grandes → más features → más dimensionalidad → más data requerida.
- N-grams más grandes → captura más contexto → mejor accuracy en tareas dependientes de orden.

**Sweet spot práctico**: `ngram_range=(1, 2)` para inglés moderno, `(1, 3)` para texto formal en español. 4-gramas y más raramente útiles.

---

## 7. TF-IDF como evolución natural

El conteo bruto (CountVectorizer) tiene un problema: **palabras frecuentes dominan**. Por la Ley de Zipf, palabras como `the`, `of`, `is`, `paciente` aparecen tan masivamente que opacan palabras menos frecuentes pero más informativas.

**TF-IDF** (Term Frequency × Inverse Document Frequency) normaliza:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \cdot \log\frac{|D|}{|\{d \in D : t \in d\}|}$$

Donde:
- $\text{TF}(t, d)$ = frecuencia del término $t$ en el documento $d$.
- $|D|$ = número total de documentos.
- El denominador del log = número de documentos donde aparece $t$.

**Efecto**: palabras que aparecen en muchos documentos (poco discriminativas) se castigan; palabras frecuentes en pocos documentos (muy discriminativas) se amplifican.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X = vec.fit_transform(corpus)
```

**Para tu pipeline**: si vas a clasificar texto, **siempre preferí TF-IDF sobre BoW puro**. Mejora típica: 5-10% en F1.

---

## 8. Las leyes que sostienen BoW

Dos leyes empíricas del lenguaje natural sostienen el éxito de BoW:

**Ley de Zipf**: $f(r) \propto 1/r^\alpha$ con $\alpha \approx 1$. Significa que pocas palabras dominan masivamente, justifica filtrado de stopwords.

**Ley de Heaps**: $V(N) \approx K \cdot N^\beta$ con $\beta \in [0.4, 0.6]$. Significa que el vocabulario crece sin parar pero lento, justifica `max_features` y subword tokenization.

Ambas leyes fueron observadas en corpus de docenas de idiomas. Son universales del lenguaje humano.

---

## 9. BoW en la era de Transformers

¿Sigue siendo relevante BoW + n-grams + TF-IDF + clasificador clásico en 2026, con BERT y GPT-4 disponibles?

**Sí, en muchos casos**:

| Caso | Mejor opción |
|---|---|
| Latencia crítica (<1 ms) | TF-IDF + LogisticRegression |
| Sin GPU disponible | TF-IDF + LogisticRegression |
| Datasets pequeños (<10k docs) | TF-IDF + LinearSVM |
| Necesidad de interpretabilidad | TF-IDF + LogisticRegression (podés ver pesos por palabra) |
| Baseline para validar viabilidad | TF-IDF + cualquier clasificador lineal |
| Multi-tarea con dominio raro | TF-IDF puede superar BERT si no hay fine-tuning data |

**No** para:
- Comprensión sintáctica profunda (necesitás Transformers).
- Tareas con dependencias largas (negación distante, anáfora).
- Estado del arte en benchmarks competitivos.
- Generación de texto.

En arquitecturas industriales modernas se ve combinación: **BERT para entidades + TF-IDF + LogisticRegression para topic classification** sobre el mismo documento.

---

## 10. Cuándo NO usar BoW

Casos donde BoW falla y necesitás algo más sofisticado:

1. **Negación**: `"sin diabetes"` vs `"con diabetes"`. BoW captura las palabras pero no necesariamente la relación.
2. **Sarcasmo / ironía**: `"oh great, another Monday"` — BoW ve `great` (positivo).
3. **Sintaxis crítica**: `"perro muerde hombre"` ≠ `"hombre muerde perro"`.
4. **Dependencias largas**: el sujeto modifica una palabra a 5 tokens de distancia.

Para esos casos usá: dependency parsing, BERT/BETO, modelos contextuales.

---

## Lecturas

- Harris (1954), *"Distributional Structure"*, Word — paper foundacional de la hipótesis distribucional.
- Salton & Buckley (1988), *"Term-Weighting Approaches in Automatic Text Retrieval"* — TF-IDF clásico.
- Sparck Jones (1972), *"A Statistical Interpretation of Term Specificity"* — IDF foundacional.

Ver papers relacionados: [NLTK](/papers/nltk-bird-loper-2006) · [Porter Stemmer](/papers/porter-stemmer-1980) · [WordNet](/papers/wordnet-miller-1995).

Ver fundamentos: [Tokenización clásica](/fundamentos/tokenizacion-clasica) · [BPE](/fundamentos/bpe) (la alternativa moderna).

Aplicación práctica: [Lab 16](/laboratorios/lab-16/) — pipeline BoW completo sobre SMS spam classification.
