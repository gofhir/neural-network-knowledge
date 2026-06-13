---
title: "P2 — Arquitectura: QA generativo (T5/BART seq2seq)"
weight: 4
math: true
---

> **Celdas 0-27 del notebook `QA_EncoderDecoder_Spanish.ipynb` (Parte 2).** El segundo cambio de paradigma del lab: de *localizar* un span en el contexto (extractivo, Parte 1) a **generar** la respuesta token a token con un encoder-decoder. Llegan dos modelos en español — [T5S y BARTO del paper de Araujo](/papers/seq2seq-spanish-araujo-2024) — y un dataset nativo, [SQAC](https://arxiv.org/abs/2107.07253).

## QA generativo: el modelo *escribe* la respuesta

En la Parte 1, BETO no producía texto: elegía dos posiciones (inicio y fin) y la respuesta era literalmente un recorte del contexto. Aquí el planteamiento es otro. El encoder lee pregunta + contexto, y el decoder **genera la respuesta token a token**, igual que en traducción o resumen:

$$P(\text{respuesta} \mid \text{entrada}) = \prod_{t} P(y_t \mid y_{<t}, \text{entrada})$$

La diferencia más profunda no está en la fórmula sino en **sobre qué se aplica el softmax**. En el extractivo, el softmax recorría las *posiciones* del contexto (¿dónde empieza el span?). Aquí recorre **todo el vocabulario** en cada paso: el modelo puede emitir *cualquier* token del idioma, esté o no en el contexto. Eso es lo que lo habilita a reformular, parafrasear e incluso inventar.

La entrada ya **no** usa `[SEP]` ni *segment ids* como BERT. Se aplana en un único string *text-to-text* con etiquetas de tarea (los *task tags* de T5):

```text
question: ¿Cuál es la capital de Chile? context: Santiago es la capital de Chile desde 1541.
```

Pregunta y contexto van separados por simples espacios y prefijos en lenguaje plano (`question:`, `context:`), no por tokens especiales con embeddings de segmento.

```text
                   ┌──────────────┐         ┌──────────────┐
  "question: ...   │              │ estados │              │  y_1 y_2 ... </s>
   context: ..."  ─▶   ENCODER    ├────────▶│   DECODER    ├──▶ respuesta
                   │  (lee todo)  │ ocultos │ (autoregr.)  │   generada
                   └──────────────┘         └──────────────┘
                                                  ▲    │
                                                  └────┘  realimenta y_<t
```

Para el detalle de la arquitectura ver el [fundamento T5 Encoder-Decoder](/fundamentos/t5-encoder-decoder); para el encuadre de la tarea, [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension).

## Extractivo (Parte 1) vs generativo (Parte 2)

| | Extractivo (Parte 1) | Generativo (Parte 2) |
|---|---|---|
| **Arquitectura** | Encoder-only (BERT/BETO) | Encoder-decoder (T5/BART) |
| **Qué hace** | Localiza un *span* en el contexto | Genera texto token a token |
| **Softmax sobre** | Posiciones del contexto | Todo el vocabulario |
| **Input** | `[CLS] pregunta [SEP] contexto [SEP]` | `"question: Q context: C"` |
| **¿Inventa palabras?** | Nunca (solo recorta) | Sí (cualquier token) |
| **¿Reformula?** | No (copia literal) | Sí (paráfrasis posible) |
| **Abstención** | Span nulo apuntando a `[CLS]` | Debe *generar* un "no sé" |
| **Métrica** | EM / F1 (coincidencia de spans) | ROUGE (tolerante a reformulaciones) |

El renglón de la abstención es sutil pero importante: el extractivo tiene un mecanismo *arquitectónico* para decir "no hay respuesta" (apuntar el span a `[CLS]`). El generativo no — tendría que *escribir* explícitamente algo como "no sé", y solo lo hará si lo entrenaron para ello.

## Setup: el stack del paper de Araujo (celdas 0-6)

```python
!git clone https://github.com/huggingface/transformers.git -b v4.52.4
!git clone https://github.com/vgaraujov/Seq2Seq-Spanish-PLMs.git
!pip install -qqq rouge_score
```

Se clona `transformers` v4.52.4 y el repositorio **Seq2Seq-Spanish-PLMs** de Vladimir Araujo (`vgaraujov`), que aporta el script de fine-tuning `run_generativeqa.py` y el acceso a los modelos **BARTO** y **T5S**. No es casual: el primer autor de ese paper, **Vladimir Araujo, es el profesor de esta clase** — estamos usando sus propios modelos y código. Ver el [análisis del paper Seq2Seq Spanish PLMs (Araujo 2024)](/papers/seq2seq-spanish-araujo-2024).

`rouge_score` instala la métrica generativa: a diferencia de EM/F1 (que miden coincidencia literal de spans y castigan cualquier paráfrasis), **ROUGE tolera reformulaciones** comparando solapamiento de n-gramas. Es la métrica adecuada cuando la salida puede no ser idéntica al texto fuente.

## SQAC: un dataset nativo en español (celdas en torno a la carga de datos)

```python
dataset = load_dataset("avacaondata/sqac_fixed")
```

**SQAC** (*Spanish Question Answering Corpus*) es el dato clave de la Parte 2, y su valor está en una palabra: **nativo**. A diferencia del SQuAD-es de la Parte 1 — que es SQuAD traducido automáticamente del inglés — SQAC fue **escrito y anotado por hispanohablantes** sobre fuentes en español: Wikipedia ES, Wikinews y el corpus literario AnCora. Esto evita el *translationese*: las distorsiones sintácticas y léxicas que deja la traducción automática (calcos, orden de palabras del inglés, falsos amigos) y que contaminan el SQuAD-es.

Hay dos detalles que conviene tener presentes para no confundirse:

- **SQAC es extractivo, pero entrena un modelo generativo.** Las anotaciones de SQAC son *spans* (respuestas recortadas del contexto, como SQuAD). Aquí, en vez de enseñar al modelo a *localizar* ese span, lo entrenamos a **generar su texto**. El span deja de ser un par de índices y pasa a ser el *target* que el decoder debe reproducir carácter a carácter.
- **SQAC no tiene preguntas sin respuesta** (a diferencia de SQuAD v2.0). Es de estilo v1.1: toda pregunta tiene respuesta en el contexto. Por eso, en esta Parte 2 **no veremos el mecanismo de abstención** que sí ejercitamos en la Parte 1 — el modelo nunca aprende a decir "no sé" porque nunca se le muestra ese caso.

Fuente del dataset: [Gutiérrez-Fandiño et al., 2021 — arXiv:2107.07253](https://arxiv.org/abs/2107.07253).

## Entrenamiento (opcional): `run_generativeqa.py`

El script entrena dos modelos en español, con configuraciones distintas:

```bash
# T5S
python run_generativeqa.py \
  --model_name_or_path vgaraujov/t5-base-spanish \
  --per_device_train_batch_size 8 \
  --max_source_length 480 --max_target_length 32 \
  --predict_with_generate ...

# BARTO
python run_generativeqa.py \
  --model_name_or_path vgaraujov/bart-base-spanish \
  --per_device_train_batch_size 16 --trust_remote_code ...
```

Los flags **nuevos respecto a la Parte 1** delatan que ahora hay generación:

| Flag | Por qué es nuevo |
|---|---|
| `--max_source_length 480` | Largo de la **entrada** (pregunta + contexto) |
| `--max_target_length 32` | Largo de la **salida generada** — *no existe* en extractivo, donde la respuesta es un span del propio contexto, no algo de longitud propia |
| `--predict_with_generate` | En evaluación ejecuta `.generate()` **autoregresivo** (genera la respuesta de verdad) en vez de solo calcular la loss |

Por dentro, los dos modelos se preentrenaron con objetivos distintos: **T5** con *span corruption* (enmascara tramos contiguos y los regenera) y **BART** con *denoising* general (corrupciones variadas — borrado, permutación, máscara — que el decoder debe deshacer). Mismo *gotcha* que la Parte 1: el entrenamiento toma ~1h, `/content` en Colab es efímero y se pierde al reiniciar — **conviene saltarlo** y usar directamente un checkpoint ya entrenado del Hub.

## El pipeline de predicción seq2seq (celda 27)

Aquí se arma el inferidor con un modelo ya afinado en SQAC, descargado del Hub:

```python
model_checkpoint = "mrm8488/spanish-t5-small-sqac-for-qa"   # type_model = "t5"
# alternativa: "vgaraujov/bart-base-spanish-sqac"           # type_model = "bart"

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
```

Piezas nuevas, todas con sabor seq2seq:

- **`AutoModelForSeq2SeqLM`** — la clase encoder-decoder con cabeza de generación (`.generate()`), en lugar de la cabeza de QA por spans de la Parte 1.
- **`Seq2SeqTrainer`** y **`DataCollatorForSeq2Seq`** — el *trainer* y el *collator* especializados que saben manejar entradas y salidas de longitud variable.
- **`label_pad_token_id=-100`** — el relleno de los *labels* se marca con `-100` para que la *loss* lo **ignore** (no se penaliza al modelo por los tokens de padding del target).

El preprocesamiento usa una función `generate_input` que arma el string `"question: Q context: C"` — los **task tags** de T5, exactamente el mismo patrón que el prefijo `"summarize: "` del [lab-22](/laboratorios/lab-22/abstractivo-t5). La predicción se configura con `predict_with_generate=True` y `fp16` (media precisión para acelerar en GPU):

```python
def run_prediction(questions, context):
    # arma "question: Q context: C", tokeniza
    predictions = trainer.predict(test_dataset)
    return tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
```

`trainer.predict` dispara la generación autoregresiva y `batch_decode(..., skip_special_tokens=True)` convierte los ids generados de vuelta a texto, descartando `</s>` y demás tokens especiales.

> **El detalle que vale el lab:** `run_prediction(questions, context)` tiene la **misma firma** que en la Parte 1, pero el motor por debajo es **totalmente distinto** — antes *extraía* índices de un span; ahora *genera* texto desde cero. Un punto de cuidado al comparar: el T5 del Hub es **`small`** y el BART es **`base`** — tamaños distintos, así que cualquier comparación de calidad entre ambos está sesgada por el número de parámetros, no solo por la arquitectura.

---

**Anterior:** [Actividad extractiva](actividades-extractivo) · **Siguiente:** [Inferencia generativa](inferencia-generativa)
