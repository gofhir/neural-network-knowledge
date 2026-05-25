---
title: "Bloque 4 — Sentiment Analysis con MLP"
weight: 40
math: true
---

Recorrido del bloque de aplicación downstream (Celdas 55-101 del notebook). Demuestra el uso **práctico** de los embeddings: como features para un clasificador de sentimiento.

## El problema y el pipeline

```
Sentiment140 (1.6M tweets, distant supervision) → sample 20k
  ↓
Limpieza (regex + BeautifulSoup + stopwords) → mediana 6.9 palabras/tweet
  ↓
Tweet vectors (2 estrategias):
  - SUMA:    Σ v_w (magnitudes en [0, 40])
  - PROMEDIO: Σ v_w / N (magnitudes en [0, 6])
  ↓
MLPClassifier sklearn (50 neuronas, Adam, 200 iter max)
  ↓
Evaluación: MAE sobre test (498 tweets, 3 clases: 0/2/4)
```

## El dataset: Sentiment140

| Característica | Valor |
|---|---|
| **Tamaño** | 1.6 millones de tweets etiquetados |
| **Idioma** | Inglés |
| **Periodo** | 2009 (cuando Twitter tenía 5 años) |
| **Etiquetas** | 0 = negativo, 4 = positivo (NO hay neutros en training) |
| **Método de etiquetado** | **Distant supervision por emoticonos** (`:)` = pos, `:(` = neg) |
| **Test set** | ~498 tweets etiquetados **manualmente**, incluye **neutros (2)** |

**Crítico**: el training es **ruidoso** porque tweets sarcásticos como "Great, my flight is cancelled again :)" se etiquetan POSITIVOS.

## El preprocesamiento — limpieza con 7 pasos

```python
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')         # decode HTML entities
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped) # @mentions + URLs
    letters_only = re.sub("[^a-zA-Z]", " ", clean)  # solo letras
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)            # WordPunctTokenizer
    words = [w for w in words if w not in stop_words]  # filter stopwords
    return " ".join(words).strip()
```

### Modos de falla observados (5 ejemplos diagnósticos)

| Tweet # | Problema |
|---|---|
| **#343** | `Not Fun & Furious` → `fun furious` (**inversión semántica por pérdida de negación**) |
| #279 | `don't` → eliminado (NLTK stopwords incluye `don`, `t`) |
| #0 | `awww`, `shoulda` sobreviven pero son OOV en Google News |
| #226 | `Tuesdayï¿½ll` → encoding ISO-8859-1 deja `ï¿½` residuales |
| #175 | De 15 palabras a 3 tokens útiles (80% pérdida) |

→ Material directo para Actividad 7 (mejoras al preprocesamiento).

## Estadísticas post-limpieza (sobre 20k tweets)

| Métrica | Valor |
|---|---|
| Tweets vacíos | 97 (0.5%) |
| Tweets con vector cero (vacíos + todas OOV) | **338 (1.7%)** |
| Longitud media post-limpieza | 6.9 palabras |
| Mediana | 6 palabras |
| Tasa OOV estimada | ~22-25% (vía ratio sum/avg) |
| Distribución target | 10049 / 9951 (balanceado) |

## Tweet vectors: SUMA vs PROMEDIO

```python
def get_embedding_sum(df, w2v, embed_dim):
    out = np.zeros((n, embed_dim))
    for i in range(n):
        for word in df.text[i].split():
            if word in w2v:
                out[i] += w2v[word]
    return out

def get_embedding_avg(df, w2v, embed_dim):
    # idem pero divide por count al final
    if count > 0:
        out[i] /= count
    return out
```

### Comparación de magnitudes (diagnóstico)

| Métrica | SUM | AVG |
|---|---|---|
| norm media | 8.32 | 1.55 |
| norm máxima | 39.87 | 6.02 |
| zeros | 338 | 338 |

→ SUM tiene **factor 40× de variabilidad** entre tweets cortos y largos. AVG normaliza a factor 20× (mucho más uniforme).

## El MLP

```python
clf = MLPClassifier(solver='adam', hidden_layer_sizes=[50,],
                    random_state=1, verbose=1)
clf.fit(X_train_sum, y_train)   # o X_train_avg
```

- **Arquitectura**: 300 → 50 (ReLU) → 2 (softmax)
- **Parámetros entrenables**: 15.152
- **Ratio ejemplos/parámetros**: 20.000 / 15.152 ≈ **1.32** (MUY bajo, riesgo overfit)

## Resultados comparativos finales

| Métrica | SUMA | PROMEDIO |
|---|---|---|
| Accuracy train | 0.985 | 0.938 |
| MAE train | **0.0374** | 0.1356 |
| **MAE test** | 0.3147 | **0.2884** ✅ |
| **Gap train→test** | 0.277 | **0.153** ✅ |
| Loss final tras 200 iter | 0.053 | 0.187 |
| Convergence Warning | Sí | Sí |

### Contexto: ¿es 0.288 bueno?

| Estrategia | MAE en este test |
|---|---|
| Siempre predecir 0 | 0.51 |
| Siempre predecir 1 | 0.50 |
| Random uniforme | ~0.43 |
| Siempre predecir 0.5 | **0.36** |
| **MLP con SUMA** | 0.315 |
| **MLP con PROMEDIO** | **0.288** |
| Esperable bien tuned | ~0.25 |
| State-of-art BERT fine-tuned | ~0.15-0.20 |

→ PROMEDIO es **20% mejor que baseline trivial**, pero ~60% lejos del state-of-art.

## Por qué PROMEDIO gana en test

Tres factores convergentes:

1. **Magnitud como atajo de memorización**: en SUM, la norma del vector crece con la longitud del tweet (rango 0-40). El MLP usa esta señal como atajo para memorizar el training. Pero "longitud" no es feature semánticamente útil → no transfiere al test.

2. **Menos capacidad de memorización**: AVG normaliza implícitamente. Tweets de 3 vs 15 palabras tienen vectores con escalas similares. El MLP no puede usar longitud como feature → aprende patrones más generalizables.

3. **Gap train→test 45% menor**: SUM tiene gap 0.277, AVG tiene gap 0.153. AVG sobreajusta menos.

## Lo que NO captura ningún método (Actividad 6)

Tanto SUM como AVG son **operaciones conmutativas**:

```
suma("not good") = v_not + v_good = v_good + v_not = suma("good not")
```

→ La negación lingüística se pierde. Esto motiva las arquitecturas modernas:
- **RNN/LSTM** (clase 19): procesan secuencias, preservan orden.
- **Transformers** (clase 20): atención + contexto bidireccional.

## El problema del neutro en test (28% del test)

El training es binario (0 vs 4 → 0 vs 1). El test tiene **3 clases** (0, 2, 4 → 0.0, 0.5, 1.0). 139 tweets neutros (28%) NUNCA fueron vistos por el modelo. La métrica MAE con labels continuos castiga proporcionalmente:
- Tweet positivo (label 1.0), modelo predice 0.9 → MAE = 0.1.
- Tweet neutro (label 0.5), modelo predice 0.9 → MAE = 0.4.

→ Los neutros contribuyen ~43% del MAE total a pesar de ser solo 28% del test.

## Cross-links

{{< cards >}}
  {{< card link="../" title="← Lab 18 - Hub" subtitle="Volver al índice del lab" icon="academic-cap" >}}
  {{< card link="../visualizacion-pca" title="Bloque 3 - PCA" subtitle="Lo geométrico" icon="academic-cap" >}}
  {{< card link="../actividades-teoricas" title="Actividades teóricas →" subtitle="Respuestas 4, 5, 6, 7" icon="academic-cap" >}}
{{< /cards >}}
