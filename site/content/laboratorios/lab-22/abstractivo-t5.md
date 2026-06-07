---
title: "P2 — Abstractivo con T5"
weight: 5
---

> **Celdas 5-15 del notebook (Parte 2).** El cambio de paradigma: de *seleccionar* oraciones a *generar* texto nuevo. El framework text-to-text de [T5](/papers/t5-raffel-2020) y la primera generación.

## El cambio de arquitectura

| | Arquitectura | Qué puede hacer |
|---|---|---|
| **BERT** (Parte 1) | Solo **encoder** | Entender/clasificar → solo *puntúa* oraciones |
| **GPT** | Solo **decoder** | Generar continuando un prompt |
| **T5** (Parte 2) | **Encoder + decoder** | Leer una entrada *y* generar una salida distinta — seq2seq puro |

El encoder lee el artículo; el decoder **genera el resumen token a token**, pudiendo producir palabras que **no están en el original**. Eso hace a T5 genuinamente **abstractivo** — lo que la Parte 1 no podía. Ver el [fundamento T5 Encoder-Decoder](/fundamentos/t5-encoder-decoder).

## Span corruption (no es el MLM de BERT)

El texto del lab dice "entrenamiento basado en el de BERT", pero T5 usa **span corruption** (denoising por spans):

```text
Original: Thank you for inviting me to your party last week.
Input:    Thank you <X> me to your party <Y> week.
Target:   <X> for inviting <Y> last <Z>
```

En vez de enmascarar tokens sueltos (BERT), enmascara **tramos contiguos** y reemplaza cada uno por un único *sentinel token* (`<X>`, `<Y>`). El decoder aprende a **regenerar los tramos faltantes** — justo la capacidad que necesita para resumir. Es el ancestro del *gap sentence generation* de [PEGASUS](/papers/pegasus-zhang-2020).

## El framework text-to-text (la idea central de T5)

*Toda* tarea de NLP se reformula como "texto entra → texto sale", distinguida por un **prefijo**:

```text
summarize: state authorities dispatched emergency crews tuesday...
   → six people hospitalized after a storm in attala county.
```

Traducción, clasificación, similitud, resumen — todas comparten **una arquitectura y una loss** (cross-entropy autoregresiva). El prefijo `"summarize: "` aparece literal en el código. (Detalle histórico: en T5 el prefijo es una *etiqueta opaca* que el modelo asocia a la tarea durante fine-tuning, **no** una instrucción en lenguaje natural como en la era posterior de instruction-tuning/FLAN.)

## Setup: stack moderno (celdas 7-9)

Contraste con la Parte 1 — aquí NO hay `pytorch_pretrained_bert` ni PyTorch 1.1.0 ni `pyrouge`:

```python
!pip install -qqq rouge-score
!pip install -qqq 'transformers[torch]'==4.45.2
!pip install -qqq datasets==3.0.1 evaluate==0.4.3 sentencepiece==0.1.99
```

Versiones **pineadas** (`==`) para reproducibilidad — el reconocimiento explícito de la lección que sufrimos en la Parte 1 (reconstruir un entorno de 2019 era frágil). `sentencepiece` porque T5 usa **SentencePiece**, no WordPiece como BERT.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
```

`T5ForConditionalGeneration` es la clase con cabeza de generación (seq2seq), trae el método `.generate()` que será la estrella de la [Actividad 3](decodificacion).

## Cargar t5-small (celda 12)

```python
t5_model = 't5-small'
device = torch.device('cuda')   # ⚠️ sin fallback: falla si no hay GPU activada
model = T5ForConditionalGeneration.from_pretrained(t5_model).to(device)
tokenizer = T5Tokenizer.from_pretrained(t5_model)
```

| Variante | $d_{model}$ | capas (enc+dec) | Parámetros |
|---|---|---|---|
| **t5-small** ← usamos esta | 512 | 6+6 | **~60 M** |
| t5-base | 768 | 12+12 | ~220 M |
| t5-large | 1024 | 24+24 | ~770 M |
| t5-3B / 11B | 1024 | 24+24 | 2.8B / 11B |

> **El porqué de small:** cabe en la GPU gratis de Colab. **El costo:** 60M es *pequeño* para generación abstractiva. Esto será la causa raíz de la [Actividad 4](evaluacion-rouge) — el paper reporta con T5 mucho mayor.

## Primera generación con beam search (celda 15)

```python
tokenized_text = tokenizer.encode("summarize: " + text, return_tensors="pt").to(device)
summary_ids = model.generate(tokenized_text,
                             num_beams=20, no_repeat_ngram_size=2,
                             num_return_sequences=5,
                             min_length=30, max_length=100, early_stopping=True)
```

A diferencia de BertSum (forward + ranking), T5 **genera autoregresivamente**: produce un token, lo realimenta, hasta `</s>`. Los argumentos:

| Argumento | Qué hace |
|---|---|
| `num_beams=20` | **Beam search**: mantiene las 20 secuencias parciales más probables, elige las mejores |
| `no_repeat_ngram_size=2` | Prohíbe repetir bigramas — combate la repetición patológica (slide 45 de la clase) |
| `num_return_sequences=5` | Devuelve los 5 mejores beams (requiere `num_beams >= 5`) |
| `min_length=30` / `max_length=100` | Límites de longitud del resumen |

### El output (noticia COVID): los 5 resúmenes son casi clones

```text
Resumen 0: the u.s. has over 637,000 confirmed cases and over 30,826 deaths... the president says some states may be able to return to normalcy earlier than that.
Resumen 1: ...the president says... (idéntico, "said" vs "says")
Resumen 3/4: ...+ "we want to get our country back," he says.
```

**Beam search optimiza calidad, no diversidad:** las 5 secuencias viven en la misma vecindad del óptimo → micro-variaciones ("u.s."/"us", "says"/"said"). Esto motiva la [Actividad 3](decodificacion): para diversidad real hace falta **sampling**.

Observaciones de fondo:
- **Abstracción real pero sutil:** "the president says" *no* está en el original ("said some states...") — una palabra nueva, generada. Pero el grueso es **copia casi literal** de la oración de cifras. T5-small abstrae a nivel de palabra suelta, no reescribe.
- **Fallo de saliencia:** el resumen lidera con las cifras de muertes, no con el titular real (la reapertura). Los números densos atraen al modelo.
- **Fidelidad:** ningún número alucinado (637,000 y 30,826 exactos). Aquí copió, y eso *ayudó* a la fidelidad.

> **El contraste que vale el lab:** T5-small se comportó como un "extractivo con costuras suaves" — copió las oraciones clave pero las **fusionó con conectores generados** ("the president says"), algo que BertSum jamás podría hacer. Esa fluidez de fusión es el valor del abstractivo, incluso en su versión más pequeña.

---

**Anterior:** [actividades del extractivo](actividades-extractivo) · **Siguiente:** [generación cualitativa (Act. 2)](generacion-cualitativa)
