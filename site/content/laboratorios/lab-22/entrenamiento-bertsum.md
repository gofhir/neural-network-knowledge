---
title: "P1 — Estrategia de entrenamiento de BertSum"
weight: 2
---

> **Análisis del comando de entrenamiento (celdas 26-27, comentadas).** El lab carga un `model.pt` pre-entrenado y deja el entrenamiento como opcional. Esta página analiza *por qué* la estrategia está diseñada así — el objeto de estudio es el comando comentado, no su ejecución.

```bash
python train.py -mode train -encoder classifier -dropout 0.1 \
  -bert_data_path ../bert_data/cnndm -model_path ../models/bert_classifier \
  -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 \
  -report_every 50 -save_checkpoint_steps 1000 -batch_size 1000 \
  -decay_method noam -train_steps 50000 -accum_count 2 \
  -use_interval true -warmup_steps 10000
```

Cada flag es una decisión de diseño. Las desmontamos de la más profunda a la más operativa.

## 1. El problema de fondo: *no existen las etiquetas*

BertSum es un clasificador binario por oración: necesita etiquetas $y_i \in \{0,1\}$. **Pero CNN/DailyMail no las tiene.** Tiene resúmenes humanos **abstractivos** — texto nuevo, parafraseado, que **no es un subconjunto de las oraciones del artículo**.

Hay un *mismatch* fundamental: el dataset está pensado para resumen abstractivo y lo forzamos a entrenar un modelo extractivo. La solución es construir las etiquetas sintéticamente con un **oracle**.

## 2. El oracle greedy (la decisión más importante)

La pregunta se vuelve: *¿qué subconjunto de oraciones del artículo, concatenadas, se parece más al resumen humano?* "Parecerse" se mide con [ROUGE](/fundamentos/rouge-metric). Ese subconjunto es la etiqueta.

Encontrar el subconjunto óptimo exacto es **NP-hard**, así que se usa un algoritmo **greedy**:

```text
oracle = []
while True:
    # de las oraciones no elegidas, ¿cuál sube más el ROUGE del conjunto?
    s* = argmax_s ROUGE(oracle + [s], gold_summary)
    if ROUGE no mejora: break
    oracle.append(s*)
```

Se agrega una oración a la vez (la que más sube ROUGE-2) hasta que ninguna mejore. Implicaciones:

- **El oracle es un techo, no una verdad.** En CNN/DM el oracle alcanza ROUGE-L = 48.87; BertSum llega a 39.63. El **gap de ~9 puntos** es infranqueable entrenando contra estas etiquetas.
- **Las etiquetas tienen ruido estructural.** El oracle premia oraciones que *comparten n-gramas* con el resumen, no oraciones *semánticamente importantes*. Una oración trivial que repite palabras del resumen recibe $y_i=1$. (Esto motivó a **MatchSum**, 2020.)
- Entrenas contra una **aproximación de una aproximación**: resumen humano → oracle greedy → predicción del modelo.

> El insight más valioso del lab: **el cuello de botella de un sistema extractivo no es el modelo, es la calidad de las etiquetas oracle.** Por eso el campo migró a lo abstractivo (Parte 2).

## 3. La loss: BCE, y por qué *no* reinforcement learning

$$\mathcal{L} = \sum_{i=1}^{m} \text{BCE}(\hat{Y}_i, y_i^{\text{oracle}})$$

Binary cross-entropy promediada sobre las $m$ oraciones. El modelo completo (BERT 110M + capa de resumen) se fine-tunea conjuntamente.

Competidores de la época (REFRESH, BanditSum, Deep Reinforced) usaban **RL** optimizando ROUGE directamente. BertSum eligió deliberadamente lo **simple**: BCE supervisada contra el oracle. ¿Por qué?

- RL sobre ROUGE es de **alta varianza** y difícil de estabilizar.
- BERT pre-entrenado aporta representaciones tan buenas que un objetivo supervisado simple basta.
- El mensaje: *el pre-entrenamiento importa más que la sofisticación del objetivo* — la tesis de toda la era 2018-2019.

## 4. Scheduler Noam con warmup (`-decay_method noam -warmup_steps 10000`)

Adam ($\beta_1{=}0.9$, $\beta_2{=}0.999$) con el scheduler de *Attention is all you need*:

$$\text{lr} = 2\times10^{-3} \cdot \min\left(\text{step}^{-0.5},\; \text{step}\cdot\text{warmup}^{-1.5}\right)$$

```text
lr
 │       pico en step=10000
 │      ╱‾╲___
 │     ╱      ‾‾‾———___        decae como 1/√step
 │    ╱                ‾‾‾———____
 │   ╱  ← warmup lineal
 │  ╱
 └────┼──────────────────────────────→ step
    10000
```

- **Warmup (0→10000):** el LR sube **linealmente**. Al inicio los pesos de la capa de resumen son aleatorios y los gradientes son enormes; aplicar el LR completo de golpe **destruiría** las representaciones pre-entrenadas de BERT (catastrophic forgetting). El warmup deja que el modelo "se asiente".
- **Decay (>10000):** baja como $1/\sqrt{\text{step}}$ para convergencia fina.

El `lr=1` de la clase `Args` (modo test) ahora tiene sentido: el LR **real** de entrenamiento es 2e-3, reescalado por el scheduler. El warmup de 10000 steps es largo a propósito — para no dañar BERT.

## 5. Gradient accumulation (`-accum_count 2`)

Acumula los gradientes de **2 batches** antes de actualizar pesos → simula un batch efectivo 2× más grande **sin** gastar 2× memoria GPU. El paper entrenó en 3× GTX 1080 Ti (11 GB), con batch efectivo ≈ 36. `batch_size=1000` se mide en **tokens** (*token-level batching*): agrupa documentos hasta llenar ~1000 tokens, dando carga de cómputo estable pese a longitudes variables.

## 6. Model averaging (`-save_checkpoint_steps 1000`)

1. Cada **1000 steps** se evalúa la loss en validación.
2. Se eligen los **top-3 checkpoints** por menor loss.
3. Se **promedian sus pesos** y se reporta el promedio sobre test.

Promediar reduce la varianza del ruido de optimización: tres mínimos locales cercanos promediados dan un punto más robusto. Es *ensembling de pobre* sin el costo de correr 3 modelos en inferencia.

## 7. El gotcha de `train_steps` vs `warmup_steps`

El comando trae `train_steps=50000` (paper), pero el comentario sugiere `TRAIN_STEPS=1000` como "prueba corta".

| `train_steps` | Qué pasa | ROUGE-L esperado |
|---|---|---|
| 1000 | Ni siquiera termina el **warmup** (10000). El LR aún sube; el modelo apenas empezó | ~muy bajo |
| 10000 | Justo termina warmup, recién aprende de verdad | mediocre |
| 50000 | Configuración del paper | 39.63 |

> **Crítico:** con `warmup_steps=10000` y `train_steps=1000`, **entrenarías solo durante la rampa de calentamiento**. El modelo nunca llega a la fase productiva. Para experimentar con pocos steps, **baja también el warmup** (p.ej. `warmup_steps=500 train_steps=2000`), o el experimento no dice nada.

## Síntesis

> BertSum apuesta a que **buen pre-entrenamiento (BERT) + etiquetas oracle baratas + objetivo supervisado simple (BCE) + scheduler que protege las representaciones (Noam warmup) + anti-redundancia en inferencia (trigram blocking)** vencen a arquitecturas más sofisticadas con RL. Y tenía razón: ROUGE-L 39.63 vs ~38 previo. El **eslabón débil** no es el modelo ni el optimizador — es el **oracle**, ese techo de ~9 puntos que ningún ajuste de entrenamiento rompe, y la razón de que el campo migrara a lo abstractivo.

---

**Anterior:** [arquitectura extractiva](arquitectura-bertsum) · **Siguiente:** [inferencia y trigram blocking](inferencia-extractiva)
