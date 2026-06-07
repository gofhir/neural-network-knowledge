---
title: "P2 — Evaluación ROUGE (Actividad 4)"
weight: 8
---

> **Celdas 57-83 del notebook (Parte 2).** Pasar de evaluación cualitativa (mirar resúmenes) a cuantitativa (medir con [ROUGE](/fundamentos/rouge-metric) sobre los 11.490 artículos del test de CNN/DailyMail), y explicar la brecha con el paper.

## Cargar el dataset (celda 59)

```python
from datasets import load_dataset
cnndm = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
```

Cada ejemplo tiene `article` (input), `highlights` (resumen humano = gold) e `id`. Es el **mismo dataset** que usó BertSum en la Parte 1, pero en formato crudo (texto plano), porque T5 solo necesita `"summarize: " + article`.

## Distribución de largos (celdas 61-64)

Antes de medir, estudiar la distribución de largos para configurar bien la evaluación.

![Histograma de largo de artículos](/laboratorios/lab-22/histograma-largo-articulos.jpg)

![Histograma de largo de resúmenes](/laboratorios/lab-22/histograma-largo-resumenes.jpg)

| | percentil 1 | percentil 5 | **promedio** | percentil 95 | percentil 99 |
|---|---|---|---|---|---|
| **Artículos** | 236.7 | 345.0 | **969.3** | 1974.0 | 2436.0 |
| **Resúmenes** | 32.0 | 44.0 | **79.3** | 135.0 | 174.0 |

**El hallazgo central:** el artículo promedio tiene **969 tokens**, pero T5 procesa como máximo 512 (ampliado a 768 en la evaluación). El percentil 95 (1974) es casi **4× el límite**. **Más del 90% de los artículos se trunca** → el modelo lee solo la primera mitad (o menos) y resume con información incompleta. Esto castiga el ROUGE y refuerza el sesgo LEAD.

Los **resúmenes** son cortos y consistentes (~79 tokens, casi todos entre 32 y 135). De ahí salen los límites de generación: `eval_min_length=32` (≈ percentil 1) y `eval_max_length=135` (≈ percentil 95). Buen diseño metodológico.

## Configurar ROUGE (celdas 66-68)

```python
eval_min_length = 32; eval_max_length = 135; eval_article_max_length = 768
eval_num_beams = 1; eval_batch_size = 128

import evaluate
rouge = evaluate.load("rouge")     # Python puro, mucho más limpio que el pyrouge/Perl de la Parte 1
rouge_types = ['rouge1', 'rouge2', 'rougeL']
```

- **ROUGE-1**: solapamiento de unigramas → cobertura de contenido.
- **ROUGE-2**: solapamiento de bigramas → fluidez/orden local.
- **ROUGE-L**: Longest Common Subsequence → estructura global.

ROUGE es **recall-oriented**: pregunta "¿qué fracción del contenido del gold aparece en mi resumen?". Si el modelo **omite** información, el ROUGE baja. `use_stemmer=True` aplica stemming (running ≈ run). Detalle: las labels usan `-100` como "ignorar" (convención HuggingFace), que `compute_metrics` reemplaza por el pad token antes de decodificar.

> **Dos decisiones que afectan el resultado:** `eval_num_beams=1` → **greedy** (por velocidad sobre 11K artículos), peor que beam search; `eval_article_max_length=768` → mitiga el truncamiento (vs. 512) pero sigue truncando (769 < 969 promedio).

## Inspección de un ejemplo (celda 72)

Un ejemplo individual (artículo de cricket, ~150 tokens, generado con `num_beams=20`) dio **ROUGE-1/2/L = 0.5833 / 0.4286 / 0.5556** — altísimo. Pero es **engañosamente bueno**: artículo cortísimo (sin truncar) + 20 beams (no greedy) + estructura de pirámide invertida que dejó copiar la primera oración. **No es representativo del promedio.** ROUGE solo cuenta solapamiento de palabras: no detecta que el resumen omitió a un segundo personaje del gold, ni distingue copia de comprensión.

## Evaluación completa (celdas 74-80)

```python
def preprocess_function(examples):
    inputs = [prefix + doc.strip().replace("\n","") for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=eval_article_max_length, truncation=True)  # ← trunca a 768
    labels = tokenizer(text_target=examples["highlights"], max_length=eval_max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

`DataCollatorForSeq2Seq` hace **padding dinámico** (al largo del batch, no global) y rellena labels con `-100`. El bucle genera con **greedy** (`num_beams=1`) sobre ~90 batches de 128. Tarda 10-30 min en T4.

### El ROUGE real

| Métrica | **Medido (t5-small)** | Reportado en el lab | Ejemplo afortunado |
|---|---|---|---|
| ROUGE-1 | **0.3489** | 0.4112 | 0.5833 |
| ROUGE-2 | **0.1311** | 0.1956 | 0.4286 |
| ROUGE-L | **0.2265** | 0.3835 | 0.5556 |

El promedio real es **muchísimo menor** que el ejemplo individual. El perfil "ROUGE-1 decente, ROUGE-2/L bajos" es típico de un modelo pequeño: acierta *de qué* hablar (palabras) pero no *cómo* estructurarlo (orden, flujo). La brecha más grande es ROUGE-L (−0.16): la organización difiere del gold humano.

## Actividad 4: ¿por qué no alcanzamos el paper?

**Dos razones concretas (de tres posibles):**

1. **Modelo mucho más pequeño** — usamos t5-small (60M); el paper reporta con versiones bastante mayores (t5-base 220M, hasta t5-11B). Menos capacidad → peor resumen.
2. **Truncamiento** — el artículo promedio (969 tokens) supera el límite procesado (768), así que la mayoría perdió su parte final y el modelo resumió con información incompleta.

(Tercera razón válida: **decodificación greedy** `num_beams=1` por velocidad, en lugar de beam search.)

**Mejora propuesta (sin tocar los parámetros usados):**

> **Fine-tunear el modelo sobre CNN/DailyMail.** En el lab usamos T5 tal como viene pre-entrenado, sin entrenarlo para esta tarea; entrenarlo con los datos de CNN/DailyMail le enseñaría el estilo y el largo de los resúmenes de noticias. El paper reporta justamente resultados *después* de fine-tunear. (Alternativas: usar un modelo especializado en resumen como [BART](/papers/bart-lewis-2020) o [PEGASUS](/papers/pegasus-zhang-2020); o dividir los artículos largos en partes para no perder la cola por truncamiento.)

---

## Cierre del laboratorio

> Recorrimos los dos paradigmas: **extractivo** (BertSum — selecciona oraciones, fiel pero rígido) y **abstractivo** (T5 — genera texto, fluido pero arriesgado). El arco completo: del sesgo LEAD y el oracle ruidoso del extractivo, a la alucinación de atribución y el truncamiento del abstractivo. Y la conclusión cuantitativa: con un modelo pequeño, decodificación rápida y artículos truncados, el ROUGE honesto (0.349/0.131/0.227) queda lejos del paper — que se logra con modelos gigantes y configuraciones costosas que no caben en Colab.

**Anterior:** [parámetros de decodificación (Act. 3)](decodificacion) · **Volver al** [índice del lab](../)
