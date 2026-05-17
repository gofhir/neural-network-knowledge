---
title: "NLLB-200 traducción multilingüe"
weight: 60
math: true
---

Cubre las celdas 53-62 del notebook. **Salto paradigmático**: del NLP clásico (NLTK, spaCy con modelos pequeños) al neural moderno (Transformer encoder-decoder con 600M parámetros). Este bloque carga y usa NLLB-200 distilled para traducir entre inglés y español, y deja preparado el patrón "translate-then-analyze" que se completa en la Actividad 4 (VADER).

Para detalle del paper ver [NLLB Team 2022](/papers/nllb-team-2022). Para arquitectura Transformer ver [Attention is All You Need](/papers/attention-is-all-you-need-vaswani-2017).

---

## 1. Setup (celdas 53-57)

### Install transformers

```python
!pip install transformers==4.56.1
```

**Versión pineada** porque APIs `AutoTokenizer` cambiaron entre 4.40 y 4.60. La 4.56 garantiza que el patrón `pipeline("translation", ..., src_lang=, tgt_lang=)` funcione.

### El link a FLORES-200

El lab menciona:

```
https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
```

**FLORES-200** es el dataset de evaluación que Meta creó junto con NLLB. Contiene **204 idiomas con traducciones profesionales** de las mismas frases. Permite evaluar **40,602 direcciones de traducción** (202 × 201).

Define los **códigos BCP-47 con script Unicode** que usa NLLB:

| Idioma | Código FLORES |
|---|---|
| Inglés | `eng_Latn` |
| Español | `spa_Latn` |
| Francés | `fra_Latn` |
| Quechua | `quy_Latn` |
| Chino simplificado | `zho_Hans` |
| Árabe estándar | `arb_Arab` |

Formato: `<ISO-639-3>_<ISO-15924>`. NLLB distingue scripts: `srp_Cyrl` (serbio cirílico) vs `srp_Latn` (serbio latino).

---

## 2. Carga del modelo (celda 58)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
device = 0 if torch.cuda.is_available() else -1

def translate(text, src_lang, tgt_lang):
    translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer,
                                     src_lang=src_lang, tgt_lang=tgt_lang,
                                     max_length=400, device=device)
    return translation_pipeline(text)[0]['translation_text']
```

### Qué carga

`facebook/nllb-200-distilled-600M` es la versión **distilled** del modelo NLLB:

| Variante | Parámetros | Disco | RAM | Hardware |
|---|---|---|---|---|
| `nllb-200-54B` (MoE) | 54.5B | ~120 GB | masivo | GPU 80GB+ |
| `nllb-200-3.3B` | 3.3B | ~15 GB | 15 GB | GPU 24GB+ |
| **`nllb-200-distilled-600M`** | **600M** | **~2.4 GB** | **~3.5 GB** | **CPU posible** |

Para Colab free tier con GPU T4, traducción tarda ~0.5s por frase. En CPU ~10-15s.

### Cómo funciona internamente cada `translate()`

Cuando llamás `translate("This is a sentence", "eng_Latn", "spa_Latn")`:

```
1. Tokenización del input:
   ["eng_Latn", "<bos>", "This", "▁is", "▁a", "▁sentence", "<eos>"]
   IDs: [256047, 0, 8423, 271, 1234, 891, 2]

2. Forward pass del encoder:
   7 tokens × 1024 dim → vectores contextualizados

3. Generación autoregresiva del decoder:
   - Empieza con token "spa_Latn" como BOS
   - Genera token a token con beam search (k=4 por default)
   - Para cuando produce <eos>

4. Detokenización: subwords → string
```

### Anti-pattern del lab

```python
def translate(text, src_lang, tgt_lang):
    translation_pipeline = pipeline(...)   # ← se recrea CADA vez
    ...
```

Cada llamada **crea un pipeline nuevo** (~200ms overhead). Para 3 frases es trivial. Para 10,000 frases sería desperdicio masivo (~30 minutos perdidos).

**Patrón correcto** para producción:

```python
pipelines_cache = {}

def translate(text, src_lang, tgt_lang):
    key = (src_lang, tgt_lang)
    if key not in pipelines_cache:
        pipelines_cache[key] = pipeline("translation", model=model, tokenizer=tokenizer,
                                         src_lang=src_lang, tgt_lang=tgt_lang,
                                         max_length=400, device=device)
    return pipelines_cache[key](text)[0]['translation_text']
```

### El `max_length=400`

NLLB fue entrenado con secuencias de **~256 tokens** fuente y target. `max_length=400` es generoso. Para texto muy largo (e.g., párrafos enteros de Wikipedia):
1. **Segmentar en oraciones** con Punkt o sentencizer.
2. **Traducir oración por oración**.
3. **Re-armar** el párrafo.

Si pasás un párrafo de 800 tokens directo, NLLB lo **trunca silenciosamente a 256** — perdés la mitad sin warning.

---

## 3. Traducción EN→ES (celdas 59-60)

```python
en_text = [
    'this is a sentence in english that we want to translate to Spanish',
    'This should also go to Spanish',
    'And this to Spanish'
]

es_text_translated = []
for sentence in en_text:
    translated = translate(sentence, "eng_Latn", "spa_Latn")
    print(translated)
    es_text_translated.append(translated)
```

### Las 3 frases gradadas en dificultad

| # | Frase | Dificultad | Output esperado |
|---|---|---|---|
| 1 | Frase completa con sujeto, verbo, objeto | Baja | `"Esta es una oración en inglés que queremos traducir al español"` |
| 2 | Frase con modal verb (`should`) | Media | `"Esto también debería ir al español"` |
| 3 | Fragmento sin verbo finito | **Alta** | `"Y esto al español"` |

### Lecciones del experimento

- **Frase 1**: traducción de calidad humana aceptable. NLLB maneja estructura sujeto-verbo-objeto.
- **Frase 2**: modal `should` → `debería` correctamente.
- **Frase 3**: fragmento `"And this to Spanish"` desafía a NLLB porque fue entrenado en oraciones completas. Puede traducir literal (`"Y esto al español"`) o intentar reconstruir.

**Lección operativa**: NLLB es robusto en frases completas, degrada con fragmentos. **En tu pipeline asegurate de pasar oraciones bien formadas** — combinación canónica: Punkt sentence segmenter + NLLB.

---

## 4. Traducción inversa ES→EN (celdas 61-62) — el round-trip

```python
es_text = es_text_translated
en_text_translated = []
for sentence in es_text:
    translated = translate(sentence, "spa_Latn", "eng_Latn")
    print(translated)
    en_text_translated.append(translated)
```

**Mismo modelo, dirección invertida**. Esto es notable: NLLB es **un solo modelo** que cubre 200 idiomas × 199 direcciones. Cambiar los códigos cambia comportamiento sin cambiar pesos.

Esto contrasta con modelos antiguos como **Marian** o **Helsinki-NLP/opus-mt** que tienen **un modelo por par de idiomas** (e.g., `opus-mt-en-es` para EN→ES y `opus-mt-es-en` para ES→EN).

### Round-trip lossiness

Esperás ver fidelidad alta para la frase 1, decreciente para la 3:

| Iteración | Texto |
|---|---|
| Original (EN) | `this is a sentence in english that we want to translate to Spanish` |
| ES (NLLB) | `Esta es una oración en inglés que queremos traducir al español` |
| EN (NLLB) | `This is a sentence in English that we want to translate to Spanish` |

Cambios:
- `this → This` (capitalización normalizada).
- `english → English` (capitalización del idioma).
- `to Spanish` → `to Spanish` (idéntico).

**Métrica de fidelidad**: muy alta para frase canónica.

### Por qué el round-trip importa

Es un **test diagnóstico** que podés usar en producción:

```python
def translate_with_quality_check(text, src, tgt):
    translated = translate(text, src, tgt)
    back = translate(translated, tgt, src)
    similarity = compute_similarity(text, back)
    if similarity < 0.8:
        return translated, "warning: low fidelity"
    return translated, "ok"
```

Notas críticas:
- **Alta fidelidad de round-trip ≠ alta fidelidad absoluta**. Cambios sutiles (`"I'll see you"` → `"Nos veremos"` → `"We'll see each other"`) pueden parecer fiel pero invierten quién es el sujeto.
- Para texto crítico (consentimientos, diagnósticos), **siempre revisar humanamente**.

---

## 5. Aplicación a tu trabajo MDM-FHIR

NLLB tiene tres usos prácticos en pipeline FHIR clínico:

### A. Translate notas clínicas ES → EN

Para aplicar herramientas que solo soportan inglés (scispaCy, MedSpaCy, BioBERT):

```python
nota_es = "El paciente presenta hipertensión arterial controlada con losartán 50 mg desde 2020."
nota_en = translate(nota_es, "spa_Latn", "eng_Latn")
# "The patient presents with controlled arterial hypertension with losartan 50 mg since 2020."

# Después aplicar scispaCy
import scispacy
nlp = scispacy.load("en_core_sci_md")
doc = nlp(nota_en)
# Reconoce: hypertension, losartan, etc.
```

### B. Mapear terminología cross-lingual

```python
term_es = "diabetes mellitus tipo 2"
term_en = translate(term_es, "spa_Latn", "eng_Latn")
# "type 2 diabetes mellitus"

# Buscar en SNOMED CT inglés
# → http://snomed.info/id/44054006
```

### C. Traducir consultas de pacientes

```python
mensaje_paciente = "Doctor, kako se kaže pomoč na slovenskem?"  # esloveno
translated = translate(mensaje_paciente, "slv_Latn", "spa_Latn")
# Permite responder al paciente en su idioma
```

**Atención**: NLLB **NO está entrenado específicamente en clínico**. Traducciones de terminología médica son literalmente correctas pero **clínicamente subóptimas** (puede confundir matices que importan en diagnóstico).

Para producción seria, **fine-tunear NLLB sobre tu corpus médico** o usar modelos especializados como Med-PaLM (Google) o MedLM.

---

## 6. Caso de demostración: idiomas low-resource

NLLB brilla en idiomas que otros modelos no soportan:

```python
# Inglés → Quechua del sur peruano
translate("Hello, how are you?", "eng_Latn", "quy_Latn")
# "Allinllachu kashanki?" o similar

# Inglés → Kinyarwanda (Ruanda)
translate("The patient has fever", "eng_Latn", "kin_Latn")
# "Umurwayi afite umuriro" o similar
```

NLLB **tiene** quechua, kinyarwanda, fon, igbo, haitiano, etc. — la mayoría de modelos comerciales NO. Es el **valor agregado** sobre Google Translate / DeepL para idiomas low-resource.

Calidad variable: high-resource ~BLEU 35-45, low-resource ~BLEU 15-25.

---

## 7. Conexión con la Actividad 4 (VADER)

NLLB se usa **una vez más** en la Actividad 4: combinar **NLLB + VADER** en el patrón **translate-then-analyze**.

```python
texto_es = "Me encanta este restaurante"
texto_en = translate(texto_es, "spa_Latn", "eng_Latn")
# → "I love this restaurant"
scores = analyser.polarity_scores(texto_en)
# → {'pos': 0.7, 'compound': 0.84} → positivo
```

VADER solo soporta inglés. NLLB cubre 200 idiomas. La combinación te da **sentiment analysis multilingüe** sin entrenar modelos por idioma.

Ver [VADER + translate-then-analyze](vader-sentiment) para el patrón completo y sus limitaciones.

---

## Lecturas

- [NLLB Team 2022 (paper)](/papers/nllb-team-2022) — análisis exhaustivo del modelo.
- [Sentiment Analysis (fundamento)](/fundamentos/sentiment-analysis) — patrón translate-then-analyze.
- [Attention is All You Need](/papers/attention-is-all-you-need-vaswani-2017) — Transformer base.
- [Seq2Seq](/papers/seq2seq-sutskever-2014) — encoder-decoder origen.

Anterior: [Actividades 1-3](actividades-1-3).
Siguiente: [VADER + translate-then-analyze](vader-sentiment).
