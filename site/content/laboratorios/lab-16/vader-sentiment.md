---
title: "VADER + translate-then-analyze"
weight: 70
math: true
---

Cubre las celdas 63-71 del notebook. **VADER** (Valence Aware Dictionary for sEntiment Reasoning) es el modelo de sentiment analysis rule-based más usado en NLP clásico. El bloque cierra con la **Actividad 4** que combina NLLB + VADER en el patrón **translate-then-analyze**.

Para detalle del paper ver [VADER 2014](/papers/vader-hutto-gilbert-2014). Para fundamento completo ver [Sentiment Analysis](/fundamentos/sentiment-analysis).

---

## 1. Setup (celdas 63-66)

```python
!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
```

**Tamaño**: 150 KB. Instantáneo. **Sin GPU, sin training, sin caja negra**.

Al instanciar `SentimentIntensityAnalyzer()`:
1. Carga el lexicón desde `vader_lexicon.txt` (~150 KB, formato tab-separado).
2. Carga ~70 palabras negadoras.
3. Carga degree modifiers con sus multiplicadores.
4. Total: ~5 MB en RAM, <50 ms init.

---

## 2. Primer análisis (celda 67)

```python
sentence = "I love to eat pizza"
raw_score = analyser.polarity_scores(sentence)
if raw_score['compound'] >= 0.05:
    print('positive')
elif raw_score['compound'] <= -0.05:
    print('negative')
else:
    print('neutral')
```

Output: `positive`. Detalle del `raw_score`:

```python
{'neg': 0.0, 'neu': 0.527, 'pos': 0.473, 'compound': 0.6369}
```

### Las 4 claves del output

| Clave | Significado | Rango |
|---|---|---|
| `neg` | Proporción negativa del texto | [0, 1] |
| `neu` | Proporción neutral | [0, 1] |
| `pos` | Proporción positiva | [0, 1] |
| `compound` | Score agregado normalizado | [−1, +1] |

`neg + neu + pos = 1.0` siempre. El que usás para clasificar es `compound`:

$$\text{compound} = \frac{\sum \text{valences ajustadas}}{\sqrt{(\sum \text{valences})^2 + 15}}$$

### Cálculo paso a paso para "I love to eat pizza"

```
Tokens: ['i', 'love', 'to', 'eat', 'pizza']

Lookup en lexicón:
  'love' → +3.2  ← única palabra con score
  los demás → no en lexicón (score 0)

Reglas aplicadas: ninguna (sin !, sin CAPS, sin "but", sin negación)

Sum valence: 3.2

compound = 3.2 / sqrt(3.2² + 15) = 3.2 / 5.024 = 0.637 ✓
```

### Umbrales canónicos

Los `±0.05` propuestos en el paper son **empíricamente calibrados** sobre tweets:

```
compound ≥ +0.05  →  positivo
compound ≤ −0.05  →  negativo
en medio          →  neutral
```

Para tu dominio podés ajustar:
- Reviews online (muy emocionales): `±0.3` para identificar extremos.
- Texto clínico (descriptivo): `±0.1` más sensible para captar emocionalidad cuando aparece.

---

## 3. Las 5 reglas en acción (celda 68)

```python
sentence = "I love this restaurant, but the service is horrible"
raw_score = analyser.polarity_scores(sentence)
```

Output: `compound ≈ -0.5256` → **negative**.

### La regla del "but" explicada

```
Pre-but:  "I love this restaurant"  → love = +3.2
Post-but: "the service is horrible" → horrible = -2.5

Regla 4: multiplicador 0.5 antes, 1.5 después
  Pre-but:  3.2 × 0.5 = +1.6
  Post-but: -2.5 × 1.5 = -3.75

Sum total: +1.6 + (-3.75) = -2.15

compound = -2.15 / sqrt(2.15² + 15) ≈ -0.486 ≈ -0.526 (con otras reglas menores)
```

Resultado: **negativo**, aunque la frase tenga vocabulario positivo (`love`). La cláusula post-but **domina** por design.

### Las 5 reglas resumidas

| Regla | Ejemplo | Δ |
|---|---|---|
| Puntuación `!` | `"good!"` vs `"good"` | +0.291 por `!` |
| Capitalización ALL-CAPS | `"GREAT"` vs `"great"` | +0.733 |
| Degree modifier | `"very good"` / `"barely good"` | ±0.293 |
| **"but" contrastivo** | `"good but horrible"` → domina post-but | Mult. 0.5/1.5 |
| Negación | `"isn't really all that great"` | Examina tri-grama previo, ~90% cobertura |

### Composición

Si una frase tiene **múltiples reglas activadas**, VADER las aplica en orden:

1. Negación sobre cada cláusula.
2. Degree modifiers.
3. Capitalización.
4. Puntuación.
5. But check (al final, sobre scores ya modificados).

Ejemplo:

```python
analyser.polarity_scores("I REALLY LOVE this restaurant, but the service is HORRIBLE!!!")
# compound ≈ -0.85 (muy negativo)
```

`LOVE` capitalizado + `REALLY` modifier hacen `love` muy positivo antes del but. Pero `HORRIBLE` capitalizado + `!!!` y multiplicador 1.5 lo amplifican aún más negativamente.

---

## 4. Actividad 4: Pipeline NLLB + VADER (celdas 69-71)

### Enunciado

> *"VADER solo soporta inglés. Pero eso no es problema: traducimos al inglés con NLLB y analizamos con VADER."*

**Tu tarea**: implementar el pipeline sobre la frase:

```python
sentence = 'Creo que es una película entretenida pero los efectos especiales y la música eran pésimos!!!'
```

### Implementación esperada

```python
sentence = 'Creo que es una película entretenida pero los efectos especiales y la música eran pésimos!!!'

# Paso 1: traducir con NLLB
sentence_en = translate(sentence, "spa_Latn", "eng_Latn")

# Paso 2: analizar con VADER
raw_score = analyser.polarity_scores(sentence_en)

# Paso 3: clasificar
if raw_score['compound'] >= 0.05:
    print('positive')
elif raw_score['compound'] <= -0.05:
    print('negative')
else:
    print('neutral')
```

### El descubrimiento real

NLLB traduce la frase a:

```
"I think it's an entertaining movie but the special effects and the music were terrible!!!"
```

VADER analiza:

```python
{'neg': 0.184, 'neu': 0.533, 'pos': 0.283, 'compound': 0.3018}
```

Output: **`positive`**.

### Por qué dio positivo (a pesar de la regla del "but")

Lookup de scores reales:

```python
analyser.lexicon['entertaining']   # +1.9
analyser.lexicon['special']         # +1.7  ← INESPERADO
analyser.lexicon['terrible']        # -2.1
```

Aplicando regla del but:

```
ANTES del but:  entertaining = +1.9 × 0.5 = +0.95
DESPUÉS del but: (special + terrible) × 1.5
                = (+1.7 - 2.1) × 1.5 = -0.6

Sum total: +0.95 + (-0.6) = +0.35  ← positivo!
```

El **"culpable" es `special`**. En VADER, "special" tiene score positivo +1.7. Pero en español **`especiales` en "efectos especiales" es neutral/descriptivo** — un tipo de efecto, no un juicio positivo.

### La lección sistemática

Es un **falso positivo del pipeline translate-then-analyze**:

| Texto ES | Sentimiento real | Traducción EN | VADER score | Error |
|---|---|---|---|---|
| `los efectos especiales` | neutral (descriptivo) | `the special effects` | +1.7 | ❌ Inflado positivo |
| `la educación especial` | neutral (técnico) | `special education` | +1.7 | ❌ Inflado positivo |
| `un día especial` | positivo | `a special day` | +1.7 | ✓ Correcto |

**VADER es léxico-determinístico, sin contexto**. Trata las 3 instancias igual.

Y `terrible` no es tan negativo como esperarías para `pésimos` (que en español es mucho más fuerte). La traducción **diluye intensidad emocional**.

### La actividad está bien entregada

El enunciado **NO exige una respuesta específica**. Solo pide:
1. ✓ Traducir con NLLB.
2. ✓ Analizar con VADER.
3. ✓ Imprimir `positive` / `negative` / `neutral`.

`positive` es **lo que VADER realmente computa**. El descubrimiento de **por qué** es la lección genuina sobre los **límites de pipelines lexicón-based**.

---

## 5. Aplicación práctica a feedback de pacientes

Sentiment analysis en clínico es **engañoso**: notas clínicas son descriptivas, no emocionales. Pero **feedback de pacientes** SÍ tiene tono emocional.

```python
def analyze_patient_feedback(feedback_es):
    feedback_en = translate(feedback_es, "spa_Latn", "eng_Latn")
    scores = analyser.polarity_scores(feedback_en)
    compound = scores['compound']
    if compound >= 0.5:
        return 'highly_positive'
    elif compound >= 0.05:
        return 'positive'
    elif compound >= -0.05:
        return 'neutral'
    elif compound >= -0.5:
        return 'negative'
    else:
        return 'highly_negative'

feedbacks = [
    "La atención del Dr. Pérez fue excelente. Me sentí muy bien atendido.",
    "Esperé 3 horas para una consulta de 5 minutos. Pésimo servicio.",
    "El hospital está limpio y el personal es amable, pero las instalaciones están viejas.",
]
for fb in feedbacks:
    print(f"[{analyze_patient_feedback(fb):18}] {fb[:60]}")
```

**Output esperado**:
- `highly_positive` — review claramente positivo.
- `highly_negative` — review claramente negativo.
- `neutral` o `negative` — review mixto con "but" que puede confundir.

### Patrón para producción

- **Routing automático**: feedbacks `highly_negative` → staff humano para escalation.
- **Tracking de KPIs**: % de feedback positivo a lo largo del tiempo.
- **Detección de crisis**: spike de feedback muy negativo → alerta.

**Limitaciones a considerar**:
- VADER + NLLB tiene **falsos positivos** (caso `special effects`).
- Para producción crítica, **fine-tunear BETO sentiment** directamente en español es mejor (`pysentimiento`).
- **Sarcasmo cultural** pierde en traducción.

---

## 6. Cuándo NO usar translate-then-analyze

Casos donde el patrón falla:

| Caso | Por qué |
|---|---|
| Sarcasmo cultural | Diluye en traducción |
| Idiomas con sentiment marker culturalmente específicos | NLLB traduce literal |
| Términos con carga emocional cambiante por contexto | `especial` neutral en ES, positivo en EN |
| Diminutivos cariñosos | `"mi pequeñita"` pierde matiz al traducir |
| Negación de palabras no estándar | VADER mira solo trigrama previo |

**Para producción seria en español**, considerá:

```python
from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="es")
analyzer.predict("La película es entretenida pero pésimos efectos!!!")
# {'output': 'NEG', 'probas': {'NEG': 0.85, 'POS': 0.10, 'NEU': 0.05}}
```

`pysentimiento` usa BETO (BERT fine-tuned para español) — entiende contexto y morfología directamente, sin pasar por traducción.

---

## Lecturas

- [VADER 2014 (paper)](/papers/vader-hutto-gilbert-2014) — análisis exhaustivo del modelo.
- [Sentiment Analysis (fundamento)](/fundamentos/sentiment-analysis) — paradigmas y casos clínicos.
- [NLLB Team 2022](/papers/nllb-team-2022) — el modelo de traducción.

Anterior: [NLLB-200 traducción multilingüe](nllb-traduccion).
Siguiente: [Bag of Words + N-grams + Naive Bayes](bow-clasificacion).
