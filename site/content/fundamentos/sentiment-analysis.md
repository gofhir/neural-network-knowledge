---
title: "Sentiment Analysis"
weight: 295
math: true
---

**Sentiment Analysis** (también llamado *opinion mining*) es el subcampo del NLP que clasifica texto según su carga emocional: positivo, negativo, neutral, o en escalas más finas (1 a 5 estrellas, alegría/tristeza/enojo, etc.). Es **una de las aplicaciones más comerciales** del NLP — se usa en monitoreo de marca, análisis de reviews, detección de crisis en redes, encuestas de pacientes, análisis de feedback corporativo.

Este fundamento cubre los **dos paradigmas dominantes**: rule-based / lexicon-based (VADER, LIWC) y supervised / neural (BERT fine-tuned). Y un patrón híbrido **translate-then-analyze** muy útil en idiomas low-resource.

---

## 1. Las dos filosofías de sentiment analysis

### A. Lexicon-based / Rule-based

**Idea**: tener un diccionario de palabras con scores emocionales pre-asignados. Para clasificar un texto, sumar los scores de las palabras presentes y aplicar reglas heurísticas (negación, intensidad, contraste).

**Características**:
- Cero training data requerido.
- Determinístico (mismo input → mismo output).
- Interpretable (podés ver qué palabras contribuyeron).
- Limitado a las palabras del lexicón.

**Ejemplos**: LIWC (Pennebaker), General Inquirer, ANEW, **VADER** (Hutto & Gilbert 2014).

### B. Supervised / Neural

**Idea**: entrenar un clasificador sobre datos etiquetados (textos con sentiment label conocido). El modelo aprende qué palabras y combinaciones son indicativas de cada clase.

**Características**:
- Requiere training data etiquetada.
- No siempre determinístico (sampling, dropout).
- Caja negra (especialmente Transformers).
- Generaliza a vocabulario fuera del lexicón.

**Ejemplos**: Naive Bayes sobre BoW, BERT/BETO fine-tuned, RoBERTa, GPT-4 como zero-shot classifier.

---

## 2. VADER: el rule-based de referencia

[VADER](/papers/vader-hutto-gilbert-2014) (Hutto & Gilbert 2014) es **la implementación más usada del paradigma lexicon-based**. Composición:

### El lexicón

**7,500+ entradas** con scores entre −4 y +4, validados por crowdsourcing en Amazon Mechanical Turk (10 raters por palabra). Incluye:

- **Palabras estándar**: `good` (+1.9), `great` (+3.1), `horrible` (-2.5).
- **Emoticons**: `:-)` (+1.4), `:(` (-2.2), `<3` (+2.0).
- **Acrónimos**: `lol` (+1.9), `omg` (+1.3), `wtf` (-2.0).
- **Slang**: `meh` (-0.5), `sux` (-1.5), `nah` (-0.5).

### Las 5 reglas heurísticas

| Regla | Ejemplo | Efecto |
|---|---|---|
| **Puntuación** — `!` amplifica | `"good!!!"` vs `"good."` | +0.291 por cada `!` |
| **Capitalización** — ALL-CAPS | `"GREAT"` vs `"great"` | +0.733 |
| **Degree modifiers** | `"very good"` / `"barely good"` | ±0.293 |
| **Conjunción "but"** | `"good but horrible"` → domina cláusula post-but | Mult. 0.5 antes / 1.5 después |
| **Negación** | `"isn't really all that great"` → invierte | Tri-grama previo |

### Output

VADER devuelve un dict:

```python
{'neg': 0.0, 'neu': 0.231, 'pos': 0.769, 'compound': 0.8316}
```

`compound` ∈ [−1, +1] es el agregado normalizado. Umbrales canónicos:

```
compound ≥ +0.05  →  positivo
compound ≤ −0.05  →  negativo
en medio          →  neutral
```

### Limitaciones

- Solo **inglés**.
- No captura **sarcasmo** ni **ironía**.
- Sensible al **dominio** (F1 = 0.96 en tweets, 0.55 en NYT editorials).
- Slang post-2014 no cubierto.

---

## 3. Sentiment con Transformers (BERT, BETO, RoBERTa)

**Patrón canónico** moderno:

1. Tomar un Transformer pre-entrenado (BERT, BETO para español, XLM-RoBERTa multilingüe).
2. Fine-tunear sobre un dataset de sentiment etiquetado.
3. Aplicar a nuevos textos.

**Ejemplos disponibles en HuggingFace**:

| Modelo | Idioma | F1 típico en tweets |
|---|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment` | inglés | ~0.96 |
| `pysentimiento/robertuito-sentiment-analysis` | español | ~0.92 |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | multilingüe | ~0.93 |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 6 idiomas, escala 1-5 | varía |

**Costo**:
- VADER: 5 MB en RAM, <1 ms por texto, sin GPU.
- BERT fine-tuned: 500 MB en RAM, ~50-200 ms por texto en CPU, ~5 ms con GPU.

Para **uso intensivo** (millones de textos/día), VADER sigue siendo económicamente superior. Para **accuracy crítica**, Transformers ganan claramente.

---

## 4. El patrón translate-then-analyze

**Problema**: VADER solo soporta inglés. Modelos de sentiment en español, portugués, chino, etc. existen pero son menos maduros que los ingleses.

**Solución pragmática**:

```
texto en español
        ↓
[NLLB-200] traducir spa_Latn → eng_Latn
        ↓
texto en inglés
        ↓
[VADER] polarity_scores()
        ↓
{'compound': -0.X}
        ↓
clasificación
```

**Ventajas**:
- Aprovecha la calidad del lexicón inglés (más completo que cualquier otro).
- Funciona para los 200 idiomas que NLLB cubre.
- Sin training específico por idioma.

**Limitaciones**:
- **Traducción es lossy**: matices culturales se pierden.
- **Cambios de carga emocional**: palabras neutras en un idioma pueden ser positivas/negativas en inglés (ej: `"special"` en `"efectos especiales"` es neutral en español pero +1.7 en VADER al traducir).
- **Sarcasmo cultural** se pierde.
- **Latencia**: 2 modelos en cadena (NLLB + VADER), mayor que VADER puro.

**Cuándo conviene**:
- Análisis exploratorio rápido en idiomas low-resource.
- Cuando no hay datos para fine-tunear modelo específico.
- Para baselines.

**Cuándo NO**:
- Cuando hay modelos nativos de calidad (e.g., **pysentimiento** para español es mejor que NLLB+VADER).
- Para tareas críticas con consecuencias serias.

Ver [VADER](/papers/vader-hutto-gilbert-2014) y [NLLB](/papers/nllb-team-2022) para los componentes.

---

## 5. Casos donde sentiment analysis falla

| Caso | Por qué falla |
|---|---|
| Sarcasmo: `"Oh great, another Monday"` | Modelos ven `great` (positivo) sin contexto irónico |
| Negación distante: `"never thought I'd hate this so much"` | VADER mira tri-grama; modelos sin contexto largo |
| Ambigüedad cultural | `"está bueno"` en español: positivo o "está OK" |
| Carga contextual: `"increase"` | Positivo en marketing (ventas), negativo en salud (presión) |
| Texto técnico/clínico | `"presents acute pain"` no expresa opinión del autor |
| Aspectos múltiples: `"food great, service awful"` | Output agregado pierde info por aspecto |

Para los últimos dos, técnicas más sofisticadas: **Aspect-Based Sentiment Analysis (ABSA)**, **target-dependent sentiment**.

---

## 6. Aplicación a texto clínico

Sentiment analysis en clínico es **engañoso**:

- **Notas clínicas son descriptivas, no emocionales**. `"paciente presenta dolor severo"` tiene vocabulario "negativo" (dolor, severo) pero la nota no expresa sentimiento del médico — describe un síntoma.
- **VADER aplicado directamente da falsos negativos** masivos: cualquier nota con `pain`, `severe`, `chronic`, `acute` da scores muy negativos.

**Donde sí aplica**:
- **Comentarios de pacientes en encuestas** de satisfacción.
- **Reviews de centros de salud** (Google, Yelp).
- **Notas de enfermería** con tono emocional explícito ("paciente cooperativa", "familia angustiada").
- **Mensajes de pacientes** en plataformas tipo MyChart.

**Pipeline recomendado para feedback de pacientes en español**:

```python
# Opción A: translate-then-analyze
feedback_en = translate(feedback_es, 'spa_Latn', 'eng_Latn')
scores = analyser.polarity_scores(feedback_en)

# Opción B: modelo nativo español (preferible)
from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="es")
analyzer.predict(feedback_es)
```

**Opción B** es mejor si tu dominio es español y tenés GPU. Opción A si no tenés GPU o necesitás múltiples idiomas.

---

## 7. Evaluación de sistemas de sentiment

Métricas estándar:

| Métrica | Significado |
|---|---|
| **Accuracy** | % de predicciones correctas (engañoso si clases desbalanceadas) |
| **Precision (por clase)** | de las predicciones de clase X, qué % son correctas |
| **Recall (por clase)** | de los items de clase X reales, qué % detectaste |
| **F1** | media armónica de precision y recall |
| **Macro F1** | F1 promediado sin ponderar por frecuencia (favorece clases minoritarias) |
| **Weighted F1** | F1 promediado ponderado por frecuencia |
| **Pearson r** | correlación con scores numéricos de humanos (si tu output es continuo) |

**Para datasets desbalanceados** (típico en sentiment, especialmente reviews donde 80% son 4-5 estrellas), `macro F1` es más informativo que `accuracy`.

---

## 8. La trampa del baseline

**Trivial baseline para 3-class sentiment**: predecir siempre la clase mayoritaria. En tweets ~60% son neutral, así que `accuracy = 0.6` se logra sin modelo.

**Tu modelo debe superar significativamente** este baseline. Si tu BERT fine-tuned alcanza accuracy 0.65, está prácticamente fallando — el baseline trivial lo casi iguala.

VADER en tweets alcanza F1 0.96 (no accuracy 0.96 — diferente métrica) — eso es **legítimamente bueno** porque las 3 clases están bien distinguidas.

---

## Lecturas

- Hutto & Gilbert (2014), *VADER*, ICWSM. Ver [paper VADER](/papers/vader-hutto-gilbert-2014).
- Pang & Lee (2008), *"Opinion Mining and Sentiment Analysis"*, Foundations & Trends in IR — survey clásico del campo.
- Socher et al. (2013), *Recursive Deep Models for Semantic Compositionality* (Stanford Sentiment Treebank) — primera demostración seria de DL para sentiment.
- Liu (2012), *Sentiment Analysis and Opinion Mining* — libro de texto del campo.

Ver papers relacionados: [VADER](/papers/vader-hutto-gilbert-2014) · [NLLB](/papers/nllb-team-2022) · [Twitter POS](/papers/twitter-pos-gimpel-2011).

Ver fundamentos: [Bag of Words](/fundamentos/bag-of-words) · [Tokenización clásica](/fundamentos/tokenizacion-clasica).

Aplicación práctica: [Lab 16 — Bloque VADER + Actividad 4](/laboratorios/lab-16/).
