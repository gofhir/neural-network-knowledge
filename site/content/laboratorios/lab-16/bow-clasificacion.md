---
title: "Bag of Words + N-grams + Naive Bayes"
weight: 80
math: true
---

Cubre las celdas 72-92 del notebook. **El bloque más largo y sustancial del lab**: pipeline completo de ML supervisado para clasificación de SMS spam vs ham. Tokenizar → preprocesar → vectorizar con BoW → entrenar Naive Bayes → evaluar. Después comparar BoW con N-grams.

Para fundamentos teóricos ver [Bag of Words](/fundamentos/bag-of-words). Para preprocesamiento clásico ver [Tokenización clásica](/fundamentos/tokenizacion-clasica) y [Porter Stemmer 1980](/papers/porter-stemmer-1980).

---

## 1. El dataset SMS Spam Collection

### Origen

- **Autores**: Almeida, Hidalgo & Yamakami (2011), *"Contributions to the Study of SMS Spam Filtering"*, ACM Symposium on Document Engineering.
- **Recolección**: 5,574 SMS reales (2010-2011) de NUS SMS Corpus, Grumbletext, SMS Corpus v0.1 Big.
- **Licencia**: Creative Commons.
- **Distribución**: **86.6% ham, 13.4% spam** — desbalanceado.

### Carga (celdas 77-80)

```python
!if [ ! -f spam.csv ]; then wget -q https://www.dropbox.com/s/rhkzcafowz6yr40/spam.csv; fi

dataset = pd.read_csv('spam.csv', encoding='ISO-8859-1')
```

**Detalle crítico**: `encoding='ISO-8859-1'` (Latin-1). Si usás UTF-8 default, falla con `UnicodeDecodeError: byte 0xa3` — el byte `0xa3` significa `£` (libra esterlina) en Latin-1, no es válido en UTF-8. Para SMS spam británico hay muchos `£` (ofrecen `£900 prize`).

**Después de cargar**:
- 5,572 filas × 5 columnas.
- Columnas útiles: `v1` (label `ham`/`spam`), `v2` (texto).
- Columnas residuales: `Unnamed: 2/3/4` (~50 filas con contenido extra que el CSV mal formateado dejó).

---

## 2. La función `clean_text` (celdas 81-82)

```python
def clean_text(sms, speller, stemmer, stop_words):
    sms = re.sub('[^a-zA-Z]+', '*', sms)         # 1. eliminar números y especiales
    sms = sms.lower()                              # 2. minúsculas
    sms = speller(sms)                             # 3. autocorrect
    tokenized_sms = word_tokenize(sms)             # 4. tokenizar
    tokenized_sms = [s for s in tokenized_sms      # 5. filtrar stopwords
                     if s not in stop_words]
    stemmed_sms = [stemmer.stem(s)                 # 6. stemming
                   for s in tokenized_sms]
    return " ".join(stemmed_sms)
```

Pipeline de **6 etapas** que aplica todo el Bloque 1 (NLTK) de golpe.

### Walkthrough sobre un SMS spam

```
"FREE ringtone! Reply 12345 to claim ur £900 prize! Hurry, only 2 days left!!!"
        │
        ▼ [1] regex [^a-zA-Z]+ → *
"FREE*ringtone*Reply*to*claim*ur*prize*Hurry*only*days*left*"
        │
        ▼ [2] lowercase
"free*ringtone*reply*to*claim*ur*prize*hurry*only*days*left*"
        │
        ▼ [3] autocorrect (puede daño abreviaciones SMS)
"free*ringtone*reply*to*claim*ur*prize*hurry*only*days*left*"
        │
        ▼ [4] word_tokenize
['free', '*', 'ringtone', '*', 'reply', '*', 'to', '*', 'claim', '*', 'ur',
 '*', 'prize', '*', 'hurry', '*', 'only', '*', 'days', '*', 'left', '*']
        │
        ▼ [5] filtrar stopwords inglés (saca 'to', 'only')
['free', '*', 'ringtone', '*', 'reply', '*', 'claim', '*', 'ur',
 '*', 'prize', '*', 'hurry', '*', 'days', '*', 'left', '*']
        │
        ▼ [6] Porter stemmer
['free', '*', 'rington', '*', 'repli', '*', 'claim', '*', 'ur',
 '*', 'prize', '*', 'hurri', '*', 'day', '*', 'left', '*']
        │
        ▼ join
"free * rington * repli * claim * ur * prize * hurri * day * left *"
```

### Problemas reconocibles del pipeline

| Problema | Consecuencia |
|---|---|
| **El `*` como token** | Aparece en TODOS los SMS → feature inútil |
| **Pérdida de números** | `12345`, `£900`, `2 days` → señales fuertes de spam aniquiladas |
| **Pérdida de `!!!`** | Capacidad amplificadora aniquilada |
| **Pérdida de MAYÚSCULAS** | Otra señal fuerte de spam aniquilada |
| **Autocorrect daña SMS slang** | `thx → the`, `ur → your`, `2nite → tonight` |

Esto es el **set-up perfecto para las Actividades 5-10** que cuestionan si BoW + este pipeline es adecuado.

### Performance

Sobre 5,572 SMS, esta función tarda **~5-10 minutos en CPU**. 80% del tiempo es `speller(sms)` — autocorrect es lento. Sin autocorrect, ~30 segundos.

---

## 3. Inicialización (celdas 83-84)

```python
speller = Speller('en')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
```

Tres objetos pasados como **argumentos explícitos** a `clean_text`. Buena práctica (testeable, no depende de globals).

`Speller('en')` tarda **~30 segundos** la primera vez (carga diccionario inglés).

---

## 4. Preprocesamiento masivo (celdas 85-86)

```python
data = []
for i in range(dataset.shape[0]):
    sms = dataset.iloc[i, 1]
    data.append(clean_text(sms, speller, stemmer, stop_words))
```

**Tarda 5-10 minutos**. La celda más lenta del lab.

Para tu propio código, patrón mejor:

```python
data = dataset['v2'].apply(lambda t: clean_text(t, speller, stemmer, stop_words)).tolist()
```

O paralelizando con `multiprocessing.Pool` (8 cores → ~1 minuto).

---

## 5. CountVectorizer (celda 88)

```python
bow_model = CountVectorizer(max_features=10000)
X = bow_model.fit_transform(data).toarray()
y = dataset.iloc[:, 0]
```

### Qué hace

Convierte la lista de strings limpios en **matriz numérica**:

```
X.shape  # (5572, 10000) — matriz NumPy 2D
y.shape  # (5572,)        — labels 'ham'/'spam'
```

Cada fila = un SMS. Cada columna = una palabra del vocabulario. Cada celda = conteo.

### `max_features=10000`

Por la Ley de Heaps el vocabulario crece sin parar. Sin límite, vas a tener 15-20k entradas con muchos hápax inútiles. Cortar al top-10k mantiene ~98% de la señal informativa y reduce dimensionalidad a la mitad.

### `.toarray()` — convertir a denso

```python
X = bow_model.fit_transform(data).toarray()
```

`CountVectorizer.fit_transform` devuelve `scipy.sparse.csr_matrix`. `.toarray()` lo convierte a denso.

**Costo**: matriz sparse ~2 MB → denso ~440 MB (5572 × 10000 × 8 bytes/int).

**Por qué el lab convierte**: porque `GaussianNB.fit(X, y)` **NO acepta sparse**. Tiene que ser denso. Es **decisión técnica subóptima** (`MultinomialNB` acepta sparse y es más natural para BoW counts).

### `token_pattern` default y el `*`

`CountVectorizer` por default usa `token_pattern=r"(?u)\b\w\w+\b"` que **requiere 2+ alfanuméricos**. Esto significa que tokens como `*`, `u`, `n`, `c` (que aparecen en `data` después de `clean_text`) **NO se incluyen**.

**Buena noticia**: el `*` fantasma **NO contamina** el vocabulario final.

### Top 10 palabras esperadas en el vocabulario

```python
import numpy as np
total_freq = X.sum(axis=0)
top_idx = np.argsort(total_freq)[-10:][::-1]
inverse_vocab = {v: k for k, v in bow_model.vocabulary_.items()}
for idx in top_idx:
    print(f"  {inverse_vocab[idx]:15} {total_freq[idx]:>5}")
```

Probable:

| Palabra | Frecuencia |
|---|---|
| `u` (después de stem) | ~1300 |
| `call` | ~700 |
| `free` | ~250 |
| `text` | ~250 |
| `mobil` | ~250 |
| `txt` | ~200 |
| `claim` | ~180 |
| `win` | ~150 |
| `prize` | ~140 |
| `repli` | ~130 |

**Las palabras de spam** (`free`, `win`, `prize`, `claim`, `mobil`) aparecen al tope → señales claras para discriminar.

---

## 6. Train/predict/evaluate (celda 90)

```python
np.random.seed(3)
X_train, X_test, y_train, y_test = train_test_split(X, y)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

### Output esperado

```
0.8585...
```

**~0.86 accuracy**.

### Diseccionando el resultado

**Baseline trivial** ("siempre ham"): accuracy = 0.87 (porque 87% del dataset es ham).

**Tu modelo GaussianNB**: ~0.86. **¡Peor que el baseline trivial!** Esto es un **mal modelo**.

### Métricas por clase (lo que importa)

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

Esperado:

```
              precision    recall  f1-score   support
         ham       0.96      0.89      0.92      1208
        spam       0.51      0.79      0.62       185
   accuracy                            0.86      1393
   macro avg       0.74      0.84      0.78      1393
```

- **Spam precision**: 0.51 → solo 51% de las predicciones "spam" son realmente spam.
- **Spam recall**: 0.79 → detecta 79% del spam real.
- **Ham recall**: 0.89 → 11% de SMS legítimos clasificados como spam erróneamente.

**En producción real esto es catastrófico**: 11% de SMS legítimos perdidos (mensajes familiares, citas médicas, etc.). El servicio se vuelve inusable.

### Por qué GaussianNB falla específicamente

La matriz `X` es **>99% ceros**. La distribución real de conteos es **Bernoulli/Multinomial sparse**, NO Gaussiana.

GaussianNB asume Normal:

$$P(w | \text{spam}) = \mathcal{N}(\mu_{w,\text{spam}}, \sigma^2_{w,\text{spam}})$$

**Esto es totalmente irrealista para BoW counts** (que son enteros, no continuos, dominados por ceros).

### El experimento que el lab no hace: MultinomialNB

```python
from sklearn.naive_bayes import MultinomialNB
clf2 = MultinomialNB()
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
# accuracy ~0.98 — salto de 12 puntos!
```

**`MultinomialNB` es la opción correcta** para BoW counts. La elección del clasificador importa **tanto como** las features.

---

## 7. N-grams (celdas 91-92)

**Una sola línea cambia**:

```python
# Celda 88 (BoW puro):
bow_model = CountVectorizer(max_features=10000)

# Celda 92 (BoW + N-grams):
bow_model = CountVectorizer(max_features=10000, ngram_range=(1, 3))
```

### Qué pasa con la distribución del vocabulario

Mirando la tabla de resultados del experimento:

```
ngram_range=(1, 1): total= 5077 | 1-grams=5077 2-grams=0 3-grams=0
ngram_range=(1, 2): total=10000 | 1-grams=3118 2-grams=6882 3-grams=0
ngram_range=(1, 3): total=10000 | 1-grams=2712 2-grams=4408 3-grams=2880
ngram_range=(2, 2): total=10000 | 1-grams=0 2-grams=10000 3-grams=0
ngram_range=(2, 3): total=10000 | 1-grams=0 2-grams=5927 3-grams=4073
ngram_range=(3, 3): total=10000 | 1-grams=0 2-grams=0 3-grams=10000
```

**Observaciones**:
- `(1, 1)` solo llega a 5077 features → el vocabulario único de unigramas en el corpus es **solo 5077** (el cleaning comprimió ~15-20k crudos a 5077 vía stemming + autocorrect).
- Con bigramas, **siempre saturás** las 10000 plazas. El espacio de bigramas es enorme.
- En `(1, 3)`: 27% unigramas, 44% bigramas, 29% trigramas. Los bigramas **dominan** las features porque hay más combinaciones frecuentes que palabras individuales.

### Bigramas top esperados en spam

- `free entri`
- `win prize`
- `txt stop`
- `repli stop`
- `claim prize`
- `pleas call`

Estos bigramas son **mucho más discriminativos** que las palabras solas. `free` aparece en ham también; `free entri` casi exclusivamente en spam.

### El output esperado

```
0.8593...
```

**Muy similar al de la celda 90**. ¡N-grams NO mejoran el accuracy!

### Por qué

1. **GaussianNB sigue siendo subóptimo** — el cuello de botella es el clasificador, no las features.
2. **El cleaning aniquila señal**: los `*` reemplazan espacios donde bigramas habrían capturado info de puntuación.
3. **`max_features=10000` con espacio ~200k**: la mayoría de bigramas raros se descartan.

**Con clasificador correcto** (MultinomialNB + TF-IDF):
- BoW puro: F1 ~0.96.
- BoW + n-grams: F1 ~0.98.
- Mejora de ~+2% con n-grams, **cuando el clasificador permite verlo**.

---

## 8. Lecciones globales del bloque

El pipeline del lab tiene **3 problemas sistemáticos**:

1. **GaussianNB sobre BoW sparse** — incorrecto. Usar MultinomialNB.
2. **Cleaning excesivo** — `clean_text` aniquila señales fuertes de spam (números, `!!!`, MAYÚSCULAS, `£`).
3. **CountVectorizer raw counts** — TF-IDF (`TfidfVectorizer`) es estrictamente mejor para clasificación.

**Pipeline corregido para 2026**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vec', TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                              min_df=2, max_df=0.95)),
    ('clf', LogisticRegression(max_iter=500, class_weight='balanced')),
])
pipeline.fit(X_train_text, y_train)   # ← X_train_text es lista de strings, no matriz
# Accuracy: ~0.98
```

Sin GaussianNB, sin autocorrect (innecesario), sin pipeline manual de 6 etapas — todo en una línea.

---

## 9. Aplicación a tu trabajo MDM-FHIR

Para clasificación de notas clínicas españolas:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import SnowballStemmer

ss = SnowballStemmer('spanish')

def stem_spanish(text):
    return ' '.join(ss.stem(t.lower()) for t in text.split() if t.isalpha())

pipeline = Pipeline([
    ('vec', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.95,
        preprocessor=stem_spanish,
    )),
    ('clf', LogisticRegression(max_iter=500, class_weight='balanced', C=1.0)),
])

pipeline.fit(reports_train, labels_train)
```

Tareas típicas donde funciona:
- **Triage urgente/no-urgente** (binario).
- **Clasificación por especialidad** (multiclase).
- **Detección de riesgo de readmisión** (binario).

Si necesitás más accuracy y tenés GPU: fine-tunear BETO en español.

---

## Lecturas

- [Bag of Words (fundamento)](/fundamentos/bag-of-words) — visión amplia.
- [Tokenización clásica](/fundamentos/tokenizacion-clasica) — el preprocesamiento.
- [Porter Stemmer 1980](/papers/porter-stemmer-1980) — el algoritmo de stemming usado.
- [NLTK 2006](/papers/nltk-bird-loper-2006) — el toolkit.

Anterior: [VADER + translate-then-analyze](vader-sentiment).
Siguiente: [Actividades 5-10 (BoW conceptual)](actividades-finales).
