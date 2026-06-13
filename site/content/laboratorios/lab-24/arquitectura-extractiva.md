---
title: "P1 — Arquitectura: BERT extractivo (span prediction)"
weight: 1
math: true
---

> **Celdas 0-17 del notebook `QA_BERT_Spanish.ipynb` (Parte 1).** Cómo un BERT pre-entrenado se convierte en un lector que **extrae** la respuesta de un texto, el andamiaje para correr el script oficial de HuggingFace en Colab y el dataset SQuAD en español.

## QA extractivo: leer y señalar, no inventar

La Parte 1 ataca el **Question Answering extractivo**, también llamado **Reading Comprehension**. La formulación viene de [SQuAD](/papers/squad-rajpurkar-2016) (Rajpurkar et al., 2016): dado un **contexto** (un párrafo) y una **pregunta**, el modelo debe **extraer un span contiguo del contexto** que responde la pregunta.

La restricción es la clave de todo: **la respuesta es siempre un fragmento literal del texto**. El modelo no genera tokens nuevos, no parafrasea, no inventa — solo señala un *inicio* y un *fin* dentro del contexto.

El ejemplo del notebook lo deja claro:

> **Contexto:** *"…Quito, la capital de Ecuador, tiene una población de 2 millones de habitantes…"*
> **Pregunta:** *"¿Cuál es la población de Quito?"*
> **Respuesta:** *"2 millones"* — un span que **ya está** en el contexto.

Esto contrasta con el QA *generativo* (que produce texto libre, como hace un T5 o un GPT). Aquí el espacio de respuestas posibles está acotado a los $\frac{n(n+1)}{2}$ sub-spans de un contexto de $n$ tokens. Ver el [fundamento de Machine Reading Comprehension](/fundamentos/machine-reading-comprehension) para el panorama de la tarea.

## El mecanismo de span prediction (el corazón de todo)

¿Cómo hace BERT QA extractivo? La idea, sorprendentemente simple, es de la Sección 4.2 del paper de [BERT](/fundamentos/bert) (Devlin et al., 2018).

**Entrada.** Se concatenan pregunta y contexto en una sola secuencia con los tokens especiales:

```
[CLS] ¿ Cuál es la población de Quito ? [SEP] Quito , la capital ... 2 millones ... [SEP]
        \________ pregunta (segment A) ________/  \______ contexto (segment B) ______/
```

BERT procesa esta secuencia y produce un **hidden vector** $T_i \in \mathbb{R}^H$ por cada token $i$ (con $H=768$ en BETO base).

**Dos vectores aprendidos.** El fine-tuning de QA introduce **solo dos parámetros nuevos**: un vector de **inicio** $S \in \mathbb{R}^H$ y un vector de **fin** $E \in \mathbb{R}^H$. Nada más — ni una cabeza grande, ni capas extra.

**Las distribuciones.** Cada token compite por ser el inicio (o el fin) de la respuesta vía un producto punto seguido de softmax sobre toda la secuencia:

$$P_{\text{start}}(i) = \frac{e^{\,T_i \cdot S}}{\sum_j e^{\,T_j \cdot S}}, \qquad P_{\text{end}}(i) = \frac{e^{\,T_i \cdot E}}{\sum_j e^{\,T_j \cdot E}}$$

**La predicción.** La respuesta es el span $(\text{start}, \text{end})$ con $\text{end} \geq \text{start}$ que maximiza el score conjunto:

$$\text{score}(i, j) = T_i \cdot S + T_j \cdot E, \qquad \widehat{(i,j)} = \arg\max_{j \geq i}\; \big(T_i \cdot S + T_j \cdot E\big)$$

Diagrama del flujo:

```
        T_[CLS] T_¿  ...  T_Quito  T_,  ...  T_2  T_millones  ...  T_[SEP]
           │     │         │       │          │      │              │
   S·T ──► ·     ·         ·       ·       ►► (alto) ·              ·     → softmax → P_start
   E·T ──► ·     ·         ·       ·          ·    ►► (alto)        ·     → softmax → P_end
                                              └────────┘
                                          span elegido = "2 millones"
```

**El punto conceptual:** QA extractivo en BERT es **clasificación de posiciones**, no generación. Hay dos clasificadores softmax (uno para el inicio, otro para el fin) que operan sobre las posiciones de la secuencia. Es por eso que un BERT *encoder-only*, sin decoder, basta para la tarea.

## Setup del entorno (celdas 7-12)

A diferencia del lab-22 (que clonaba un fork de 2019), aquí se necesita el **script oficial `run_qa.py`** que vive en `examples/pytorch/question-answering/` del repo `transformers` — y que **no viene** con `pip install transformers`. Hay que clonar el código fuente:

```bash
!git clone --depth 1 --branch v4.52.4 https://github.com/huggingface/transformers.git
%cd transformers
!pip install -e .
```

| Comando | Por qué |
|---|---|
| `--depth 1` | Clon superficial (solo el último commit), descarga mucho más rápido |
| `--branch v4.52.4` | **Versión pineada**: `run_qa.py` evoluciona y asume APIs concretas de `Trainer`/`datasets` |
| `pip install -e .` | Instala `transformers` en modo *editable* desde el código clonado, dando acceso a `examples/` |

Las dependencias del script también van pineadas (`evaluate`, `accelerate`, `datasets`) y se desactiva el logging externo:

```python
import os
os.environ["WANDB_DISABLED"] = "true"   # evita que run_qa.py intente loguear a Weights & Biases
```

> **Gotcha de versiones.** `run_qa.py` está fuertemente acoplado a la versión de `transformers` que lo acompaña; mezclar un `run_qa.py` nuevo con un `transformers` viejo (o viceversa) rompe por APIs incompatibles. Además, tras el `pip install -e .` **Colab pide reiniciar el runtime** (*Restart runtime*) para que tome la nueva instalación — si no se reinicia, los `import` siguen apuntando a la versión vieja.

## SQuAD en español: traducido, no nativo (celdas 13-15)

```python
from datasets import load_dataset
dataset = load_dataset("TheTung/squad_es_v2", "small", trust_remote_code=True)
```

Detalles importantes de este dataset:

- **No es anotado por humanos.** `squad_es_v2` es el SQuAD **inglés traducido automáticamente** al español, con el span de la respuesta **realineado** mediante el método **TranslateAlignRetrieve**: se traduce el contexto, se traduce la respuesta y se busca dónde quedó alineada la respuesta dentro del contexto traducido.
- **Esto introduce *translationese*:** el español resultante arrastra calcos sintácticos y léxicos del inglés. Es un español "de traducción", no natural. Este es justamente el contraste que motiva la Parte 2, que usa **SQAC**, un corpus de QA escrito **nativamente** en español.
- **`"small"`** es un subconjunto del dataset, elegido para que el entrenamiento quepa en los tiempos y la RAM de Colab.

> **Gotcha de seguridad: `trust_remote_code=True`.** Esta bandera autoriza a `datasets` a **ejecutar código Python alojado en el repo del dataset** (el loader custom). Es necesario para que el dataset cargue, pero significa confiar en el autor del repo: nunca lo actives en datasets de procedencia desconocida.

## SQuAD v1.1 vs v2.0: saber cuándo callar

El sufijo `_v2` no es cosmético. [SQuAD 2.0](/papers/squad2-rajpurkar-2018) (Rajpurkar, Jia & Liang, 2018) extiende la v1.1 con un giro fundamental: **preguntas sin respuesta** (*unanswerable*), redactadas **adversarialmente** para parecer respondibles pero sin que el contexto contenga la respuesta.

Esto cambia la tarea: ya no basta con extraer *algún* span; el modelo debe **saber abstenerse**. Cuando una pregunta no tiene respuesta, el span correcto es el **span nulo apuntando a `[CLS]`** (posición 0). La capacidad evaluada es *"saber cuándo NO responder"*, que es justo lo que un sistema de QA realista necesita para no alucinar respuestas. Ver el [fundamento de métricas de evaluación de QA](/fundamentos/qa-evaluation-metrics).

La estructura del campo `answers` refleja esto:

```python
{'text': ['2 millones'], 'answer_start': [73]}   # respondible
{'text': [],             'answer_start': []}      # unanswerable (vacío)
```

Un detalle técnico que el preprocesamiento debe resolver: **`answer_start` es un offset de CARÁCTER** (el carácter 73 del string de contexto), pero BERT trabaja con **tokens**. El preprocesamiento de `run_qa.py` mapea ese offset de carácter al **índice de token** correspondiente, usando los *offset mappings* del tokenizer rápido. Si el mapeo falla (p. ej. el span cae a mitad de un token), la respuesta se marca como nula.

## Entrenamiento (opcional, celdas 16-17)

El entrenamiento usa `run_qa.py` con **BETO** como modelo base:

```bash
!python run_qa.py \
  --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
  --dataset_name TheTung/squad_es_v2 --dataset_config_name small \
  --version_2_with_negative \
  --do_train --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 --doc_stride 128 \
  --output_dir /content/beto-squad-es
```

| Hiperparámetro | Valor | Significado |
|---|---|---|
| `model_name_or_path` | `dccuchile/bert-base-spanish-wwm-cased` | **BETO**: BERT en español de la U. de Chile. *wwm* = **whole word masking** (enmascara palabras completas, no subtokens sueltos); *cased* = distingue mayúsculas |
| `per_device_train_batch_size` | 12 | Secuencias por paso de gradiente |
| `learning_rate` | 3e-5 | LR típico de fine-tuning de BERT |
| `num_train_epochs` | 2 | ~1 hora por época en la GPU de Colab |
| `max_seq_length` | 384 | Largo máximo de la secuencia `[CLS] pregunta [SEP] contexto [SEP]` |
| `doc_stride` | 128 | Solapamiento de la ventana deslizante (ver abajo) |
| `version_2_with_negative` | — | Activa el manejo de preguntas sin respuesta de SQuAD 2.0 |

Esto conecta directamente con el fine-tuning de BERT visto en la [clase-20](/clases/clase-20/) (clasificación de Fake News, generación con GPT-2): el patrón es el mismo —encoder pre-entrenado + cabeza ligera + pocas épocas con LR pequeño—, solo cambia la cabeza (aquí, los dos vectores $S$ y $E$).

### Ventana deslizante: `doc_stride` (el truco de los contextos largos)

¿Qué pasa si el contexto excede los 384 tokens de `max_seq_length`? No se trunca y se pierde la cola — se aplica una **ventana deslizante**: el contexto se parte en varias ventanas que **se solapan** `doc_stride=128` tokens, de modo que ninguna respuesta quede partida justo en el borde de una ventana.

```
contexto (600 tokens):  [=================================================]
ventana 1 (384):        [================]
ventana 2 (384):                  [================]   ← arranca 128 antes de que termine la 1
ventana 3 (384):                            [================]
                                  └─ solapamiento de 128 tokens ─┘
```

Cada ventana se convierte en un ejemplo independiente para BERT; en inferencia, los scores de span de todas las ventanas del mismo documento se combinan y se elige el mejor span global.

> **GOTCHA REAL (vivido).** Roberto corrió este entrenamiento y **la sesión de Colab se cerró a mitad de camino** (probable OOM con `batch_size=12`, o el timeout de inactividad de Colab). Como `/content/` es **efímero**, se perdió **todo el output**: checkpoints, `trainer_state.json`, el log entero. Lecciones:
>
> - **Entrenar es opcional y frágil.** Para esta tarea es más sensato cargar un modelo **ya fine-tuneado** desde el Hub y saltar directo a inferencia.
> - **Si igual entrenas, blíndalo:** `--output_dir` apuntando a **Google Drive** (no a `/content/`), **batch menor + `gradient_accumulation_steps`** para evitar OOM, y redirigir el log con `tee` a un `.txt` (`!python run_qa.py ... 2>&1 | tee /content/drive/MyDrive/train.log`) para no perder la traza si la sesión muere.

---

**Siguiente:** [Inferencia extractiva](inferencia-extractiva)
