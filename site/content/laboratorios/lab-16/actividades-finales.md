---
title: "Actividades 5-10 (BoW conceptual)"
weight: 90
math: true
---

Cubre las celdas 93-105 del notebook. **Seis actividades de cierre evaluado** sobre BoW y N-grams. Las primeras tres son multiple choice (`Si`/`No`), las siguientes dos piden pegar **una línea de código modificada**, y la última es **abierta** (respuesta escrita).

---

## Actividad 5 — ¿N-grams incluye también unigramas?

> *Bag-of-Words considera únicamente la frecuencia de aparición de palabras individuales. El código que usa bag of n-grams (`ngram_range=(1,3)`), ¿está considerando también esta información?*

### Respuesta: **Sí**

`ngram_range=(min_n, max_n)` toma todos los n con `min_n ≤ n ≤ max_n`. `(1, 3)` significa `n ∈ {1, 2, 3}` → unigramas + bigramas + trigramas.

### Validación con la doc de sklearn

> *`ngram_range`: tuple (min_n, max_n), default=(1, 1). All values of n such that min_n <= n <= max_n will be used.*

Rango **inclusivo en ambos extremos**.

### Verificación empírica

```python
from collections import Counter
vec = CountVectorizer(max_features=10000, ngram_range=(1, 3))
vec.fit(data)
sizes = Counter(len(k.split()) for k in vec.vocabulary_)
# {1: 2712, 2: 4408, 3: 2880}
```

**Hay 2712 unigramas** en el vocabulario → confirma que están incluidos.

---

## Actividad 6 — ¿BoW puro distingue frases con mismo vocabulario pero distinto orden?

> *"Me gusta mucho el pan con paté" versus "Me gusta el pan con mucho paté". ¿Bag-of-Words podría distinguirlas?*

### Respuesta: **No**

Ambas frases tienen las **mismas 7 palabras**, en orden distinto. Aplicando BoW:

| Palabra | Frase A | Frase B |
|---|---|---|
| Me | 1 | 1 |
| gusta | 1 | 1 |
| mucho | 1 | 1 |
| el | 1 | 1 |
| pan | 1 | 1 |
| con | 1 | 1 |
| paté | 1 | 1 |

**Vectores BoW idénticos**. Por design, BoW solo cuenta frecuencias, sin orden.

### Por qué se llama "Bag of Words"

La metáfora: imaginá que metés cada palabra en una bolsa. La bolsa recuerda **cuántas de cada tipo** hay, no en qué orden estaban. **Eso es BoW** — estructuralmente incapaz de distinguir orden.

### Verificación empírica

```python
from sklearn.feature_extraction.text import CountVectorizer

frases = ["Me gusta mucho el pan con paté", "Me gusta el pan con mucho paté"]
vec = CountVectorizer()
X = vec.fit_transform(frases).toarray()
print((X[0] == X[1]).all())   # True ← idénticos
```

### Ejemplos canónicos donde BoW falla

| Frase A | Frase B | ¿BoW las distingue? |
|---|---|---|
| "Perro muerde hombre" | "Hombre muerde perro" | ❌ No |
| "El gato corrió tras el ratón" | "El ratón corrió tras el gato" | ❌ No |
| "No me gusta esto" | "Me gusta esto, no" | ❌ No (incluso pierde negación) |

---

## Actividad 7 — ¿Bag-of-2grams distingue las mismas frases?

> *Bag-of-2Grams ¿Podría distinguirlas (aunque sea levemente)?*

### Respuesta: **Sí**

Generando bigramas de cada frase:

**Frase A**: `"Me gusta mucho el pan con paté"`

Bigramas: `me gusta`, `gusta mucho`, `mucho el`, `el pan`, `pan con`, `con paté`.

**Frase B**: `"Me gusta el pan con mucho paté"`

Bigramas: `me gusta`, `gusta el`, `el pan`, `pan con`, `con mucho`, `mucho paté`.

**Bigramas exclusivos de A**: `gusta mucho`, `mucho el`, `con paté`.
**Bigramas exclusivos de B**: `gusta el`, `con mucho`, `mucho paté`.

**6 bigramas que difieren** → vectores Bag-of-2grams distintos.

### Verificación

```python
vec = CountVectorizer(ngram_range=(2, 2))
X = vec.fit_transform(frases).toarray()
print((X[0] == X[1]).all())   # False ← diferentes
```

### Lo que esta secuencia 6+7 enseña

| Modelo | ¿Distingue orden? | ¿Por qué? |
|---|---|---|
| BoW (unigramas) | ❌ No | Solo cuenta ocurrencias |
| Bag-of-2grams | ✓ Sí | Captura pares adyacentes |
| Bag-of-3grams | ✓ Sí | Captura tríos adyacentes |
| RNNs / Transformers | ✓ Sí | Captura orden global |

Los n-grams existen específicamente **para capturar el orden que BoW pierde**.

### Limitación de bigramas

Capturan solo **orden local** (palabras adyacentes), NO dependencias largas. Para frases como `"el paciente, según el reporte, tiene fiebre"` vs `"el paciente tiene, según el reporte, fiebre"` (negación o adverbial separado del término por 5 tokens), bigramas no alcanzan — necesitás trigramas+, RNNs, o Transformers.

---

## Actividad 8 — Modificar para solo 3-Grams y 4-Grams

> *¿Qué línea de código tendría que editar?*

### Respuesta

```python
bow_model = CountVectorizer(max_features=10000, ngram_range=(3, 4))
```

`ngram_range=(3, 4)` toma n ∈ {3, 4} → solo trigramas y 4-gramas.

### Errores comunes a evitar

| Respuesta incorrecta | Por qué |
|---|---|
| `ngram_range=(3, 3)` | Solo 3-grams, sin 4-grams |
| `ngram_range=(4, 4)` | Solo 4-grams, sin 3-grams |
| `ngram_range=(1, 4)` | Incluye 1, 2, 3, 4 (demasiado) |
| `ngram_range=3, 4` | Sintaxis ambigua, debe ser tupla |
| `ngram_range=[3, 4]` | Tipo incorrecto, sklearn espera tupla |

---

## Actividad 9 — Modificar para solo 3-Grams

> *¿Qué línea de código tendría que editar?*

### Respuesta

```python
bow_model = CountVectorizer(max_features=10000, ngram_range=(3, 3))
```

`(min_n, max_n)` con valores **iguales** → solo se genera ese N específico.

Confirmado empíricamente del experimento de la Actividad 5:

```
ngram_range=(3, 3): total=10000 | 1-grams=0 2-grams=0 3-grams=10000
```

---

## Actividad 10 — ¿Recomendarías Deep Learning?

> *Comente brevemente si recomendaría utilizar una técnica más avanzada (Deep Learning). Asuma que es para una empresa pequeña con pocos mensajes diarios.*

### El contexto cambia la respuesta

La frase `"empresa pequeña que recibe pocos mensajes diarios"` cambia totalmente la decisión.

### Comparativa cuantitativa

| Dimensión | BoW + MultinomialNB (bien hecho) | BERT fine-tuned |
|---|---|---|
| Accuracy | ~0.98 | ~0.99 (+1 punto) |
| Latencia | <1 ms | 50-200 ms (CPU), 5 ms (GPU) |
| Costo training | Segundos | Horas en GPU |
| RAM | ~50 MB | ~500 MB - 1 GB |
| Hardware | CPU básico | GPU recomendada |
| Datos requeridos | 5k ejemplos | Idealmente 10k-100k |
| Interpretabilidad | Alta | Baja (caja negra) |
| Deploy | pickle de 5 MB | Servidor con CUDA |

### Respuesta sugerida (versión breve)

```
No recomiendo usar Deep Learning en este caso. Para una empresa pequeña con bajo volumen
de mensajes, BoW combinado con un clasificador adecuado (MultinomialNB o LogisticRegression)
ya alcanza accuracy ~98% en este tipo de tarea, con latencia sub-milisegundo, sin necesidad
de GPU, y entrenamiento que toma segundos. Deep Learning aportaría máximo 1-2% adicional de
accuracy pero a costo de mayor complejidad operativa, dependencias de hardware (GPU), pérdida
de interpretabilidad, y tiempo de desarrollo mucho más largo. El costo-beneficio no se
justifica para el volumen y los recursos de una empresa pequeña.
```

### Respuesta más estructurada

```
No recomendaría usar Deep Learning para este caso de uso específico, por cuatro razones:

1. Beneficio marginal: BoW bien implementado (MultinomialNB + TF-IDF + n-grams) alcanza
   accuracy ~98% en clasificación de SMS spam. BERT u otros modelos de DL llegan ~99%,
   solo 1 punto adicional.

2. Costo desproporcionado: Deep Learning requiere GPU, librerías más pesadas,
   tiempos de entrenamiento de horas, y memoria 10-20x mayor que BoW.

3. Volumen bajo: la empresa recibe pocos mensajes diarios. La diferencia de 1% en
   accuracy se traduce en quizás 1 SMS mal clasificado por día. No vale el costo operativo.

4. Interpretabilidad: BoW + Naive Bayes permite explicar las decisiones a usuarios
   ("se clasificó como spam porque contiene palabras X, Y, Z"). Esto es valioso para
   atender quejas y auditar el sistema. Deep Learning es opaco.

Recomendaría en su lugar mejorar el pipeline BoW: cambiar GaussianNB por MultinomialNB,
usar TF-IDF en lugar de CountVectorizer, agregar bigramas, y validar con stratification.
Esto puede llevar el modelo de ~86% del lab a ~98% sin ningún cambio arquitectónico mayor.
```

### Por qué la Actividad 10 es importante pedagógicamente

Cierra el lab con una lección **subestimada** en cursos de IA:

> **El mejor modelo no es el más sofisticado, es el más apropiado al contexto.**

En la era de LLMs (GPT-4, Claude, Llama), hay una tentación constante de "usar AI moderna" para todo. Pero los modelos clásicos (BoW + Naive Bayes, regresión logística, gradient boosting) siguen siendo **competitivos en muchos casos** y tienen ventajas operativas significativas.

### Cuándo preferir cada paradigma

| Caso | Preferí |
|---|---|
| Tarea con complejidad semántica alta (ironía, contexto) | Deep Learning |
| Datasets muy grandes (millones de ejemplos) | Deep Learning |
| Tareas multimodales (texto + imagen + audio) | Deep Learning |
| Cuando 0.1% accuracy adicional justifica el costo (fraude bancario) | Deep Learning |
| Baseline rápido para validar viabilidad | Métodos clásicos |
| Bajos recursos (CPU only, sin GPU) | Métodos clásicos |
| Datos limitados (<10k ejemplos) | Métodos clásicos |
| Necesidad de interpretabilidad (compliance, médico) | Métodos clásicos |
| Latencia crítica (sub-millisegundo) | Métodos clásicos |
| Mantenimiento por equipo no-ML | Métodos clásicos |

---

## Aplicación a tu trabajo MDM-FHIR

Estas lecciones aplican directamente:

| Tarea clínica | Recomendación |
|---|---|
| Triage rápido (urgente/no) | BoW + LogisticRegression |
| Extracción de entidades clínicas | scispaCy o BETO fine-tuned |
| Sentiment de feedback de pacientes | translate-then-analyze (NLLB + VADER) o pysentimiento |
| Diagnóstico asistido | Deep Learning (error costoso) |
| Clasificación de especialidad | BoW + LogisticRegression |

**Patrón general**: empezá siempre con baseline clásico. Si funciona, pará. Si no, escalá complejidad gradualmente.

---

## Cierre del lab

Has terminado las 10 actividades y los 7 bloques del lab. Resumen consolidado:

| Bloque | Celdas | Técnica | Estado |
|---|---|---|---|
| 0 | 0-4 | Intro y mapa | ✓ |
| 1 | 5-31 | NLTK clásico | ✓ |
| 2 | 32-42 | spaCy | ✓ |
| 3 | 43-52 | Actividades 1-3 | ✓ |
| 4 | 53-62 | NLLB-200 traducción | ✓ |
| 5 | 63-71 | VADER + Actividad 4 | ✓ |
| 6 | 72-92 | BoW + N-grams | ✓ |
| 7 | 93-105 | Actividades 5-10 | ✓ |

Recorrido completo: **106 celdas**, **10 actividades**, **7 papers fundacionales** descargados con análisis exhaustivo.

---

## Lecturas

- [Bag of Words (fundamento)](/fundamentos/bag-of-words) — visión completa.
- [Sentiment Analysis (fundamento)](/fundamentos/sentiment-analysis).
- [Tokenización clásica (fundamento)](/fundamentos/tokenizacion-clasica).

Anterior: [Bag of Words + N-grams + Naive Bayes](bow-clasificacion).
Volver al hub: [Lab 16](/laboratorios/lab-16/).
