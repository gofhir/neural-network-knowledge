---
title: "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization (Zhang et al., 2020)"
slug: pegasus-zhang-2020
authors:
  - Jingqing Zhang
  - Yao Zhao
  - Mohammad Saleh
  - Peter J. Liu
year: 2020
venue: "ICML 2020 (Proceedings of the 37th International Conference on Machine Learning, PMLR 119)"
arxiv: "1912.08777"
url: "https://arxiv.org/abs/1912.08777"
code: "https://github.com/google-research/pegasus"
tags:
  - summarization
  - abstractive-summarization
  - transformer
  - encoder-decoder
  - self-supervised-pretraining
  - gap-sentence-generation
  - rouge
  - low-resource
keywords:
  - PEGASUS
  - GSG
  - gap sentences
  - ROUGE
  - HugeNews
  - C4
  - XSum
  - CNN/DailyMail
  - principal sentence selection
class: 22
type: paper
status: review
---

## Resumen ejecutivo

PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence) es un modelo encoder-decoder Transformer presentado por Zhang, Zhao, Saleh y Liu (Google Research / Imperial College London) en ICML 2020. Su contribución central no es una nueva arquitectura sino un **objetivo de pre-entrenamiento self-supervised diseñado específicamente para que el modelo aprenda a resumir**: el Gap Sentence Generation (GSG). En GSG, se enmascaran oraciones completas del documento de entrada y se exige al decoder regenerarlas como una única secuencia objetivo, lo que simula directamente la dinámica de extracción y reescritura propia de la summarization abstractive.

Los autores comparan tres familias de estrategias para escoger esas oraciones objetivo (Random, Lead y Principal) y demuestran que seleccionar las oraciones más "centrales" del documento mediante una métrica ROUGE-1 F-score contra el resto del texto es la opción óptima. Entrenando con un corpus mixto (C4 + HugeNews, 1.5 mil millones de artículos curados) un modelo de 568M parámetros (PEGASUS-large), alcanzan el estado del arte en los doce benchmarks de summarization evaluados al momento de publicación: XSum, CNN/DailyMail, NEWSROOM, Multi-News, Gigaword, WikiHow, Reddit TIFU, BIGPATENT, arXiv, PubMed, AESLC y BillSum.

Más allá del SOTA, el paper aporta dos observaciones operativamente muy importantes:

1. **Low-resource summarization**: con apenas 1000 ejemplos de fine-tuning, PEGASUS-large supera el SOTA previo (entrenado con datasets completos) en seis de los doce benchmarks. Esto cambia la economía del problema: ya no hace falta recolectar millones de pares documento-resumen.
2. **Paridad humana**: en evaluaciones side-by-side con jueces humanos sobre XSum, CNN/DailyMail y Reddit TIFU, los resúmenes de PEGASUS no son estadísticamente peores que los escritos por humanos ($p<0.01$).

El modelo y los checkpoints fueron publicados como código abierto en `google-research/pegasus` y rápidamente integrados a la biblioteca `transformers` de HuggingFace, donde `google/pegasus-cnn_dailymail`, `google/pegasus-xsum` y `google/pegasus-multi_news` se han vuelto baselines obligados para summarization en español, inglés y dominios técnicos.

## Contexto histórico: 2019-2020, la fiebre encoder-decoder

Para entender la novedad de PEGASUS hay que situarlo en la ola de modelos sequence-to-sequence pre-entrenados que estalló entre finales de 2018 y la primera mitad de 2020. La progresión es la siguiente:

- **BERT** (Devlin et al., 2018): encoder-only, objetivo de Masked Language Modeling (MLM) + Next Sentence Prediction (NSP). Excelente para entender texto, pero no genera secuencias largas.
- **GPT-2** (Radford et al., 2019): decoder-only autoregresivo. Genera, pero no condiciona explícitamente sobre un documento de entrada del que se quiera extraer información.
- **MASS** (Song et al., 2019): primer intento de pre-entrenamiento sequence-to-sequence enmascarando un fragmento contiguo y haciendo que el decoder lo regenere. Sin embargo enmascara un único span aleatorio y no piensa explícitamente en summarization.
- **UniLM** (Dong et al., 2019): combina LM unidireccional, bidireccional y seq-to-seq en un mismo Transformer.
- **T5** (Raffel et al., 2019): unifica todo en "text-to-text", introduce C4 (Colossal Clean Crawled Corpus, 750 GB) y enmascara spans aleatorios.
- **BART** (Lewis et al., 2019): denoising autoencoder con noising functions variadas (text infilling, sentence permutation, etc.). El objetivo final efectivo es text infilling de spans.

El **vacío común** que detectan los autores es que ninguno de estos objetivos de pre-entrenamiento simula la tarea de summarization. T5 enmascara *spans* cortos, BART enmascara *spans* o permuta oraciones, MASS enmascara un único fragmento. La hipótesis de PEGASUS es directa: si el objetivo se parece a la tarea downstream, el transfer learning será más eficiente; en particular, si se enmascaran **oraciones completas importantes** y el modelo aprende a regenerarlas a partir del resto, está aprendiendo en pre-entrenamiento exactamente la dinámica que necesita en summarization (entender qué es importante, condensarlo, reescribirlo).

Como contemporáneo conviene mencionar a Khandelwal et al. (2019), que probaron summarization con un Transformer pre-entrenado sobre Wikipedia y obtenían apenas ROUGE-2 de 13.1 en CNN/DailyMail con 3000 ejemplos. PEGASUS llega al mismo régimen low-resource con 1000 ejemplos y obtiene ROUGE-2 de 19.35 — una diferencia cualitativa más que cuantitativa.

También vale mencionar el trabajo de Radford et al. (2018b) con GPT-2 sobre summarization zero-shot: al promptear con "TL;DR" lograban ROUGE-2 de 8.27 en CNN/DailyMail. Era una prueba de concepto de que los language models grandes podían resumir sin supervisión, pero el rendimiento estaba lejos del SOTA. PEGASUS demuestra que un objetivo **diseñado** para summarization llega mucho más lejos que un objetivo genérico de language modeling, incluso con orden de magnitud menos parámetros.

La inspiración inmediata reconocida en el paper viene de dos líneas. Por un lado, los trabajos de masking de spans contiguos como Joshi et al. (2019, SpanBERT) y Raffel et al. (2019, T5), que mostraron que enmascarar spans en lugar de tokens individuales mejoraba el aprendizaje de fenómenos linguísticos largos. Por otro, Nallapati et al. (2017) propuso SummaRunner, un extractive summarizer basado en RNN que seleccionaba oraciones de forma secuencial greedy — el mismo algoritmo que PEGASUS adopta para la variante Principal-Seq de su selección de gap sentences. La originalidad está en combinar estas dos ideas: tomar la selección por importancia del extractive summarization clásico y convertirla en señal de pre-entrenamiento para un modelo abstractive.

Conviene también destacar que el paper aparece en ICML 2020, el mismo año en que la comunidad NLP estaba transitando del paradigma "fine-tune a single architecture for each task" hacia "build foundation models that adapt with minimal data". PEGASUS se posiciona del lado tradicional (fine-tune por dataset) pero anticipa el régimen low-resource que dominaría a partir de 2021-2022 con instruction tuning, prompt engineering y few-shot prompting. En ese sentido, PEGASUS es el último gran modelo task-specific de la era pre-LLM antes de que GPT-3 (julio 2020) volviera obsoleto el fine-tuning extensivo para muchos casos de uso.

## Gap Sentence Generation (GSG): matemática y proceso

### Formalización

Sea un documento $D = \{s_1, s_2, \ldots, s_n\}$ una secuencia ordenada de $n$ oraciones. El objetivo GSG procede en cuatro pasos:

1. **Selección**: elegir un subconjunto $G \subset D$ de tamaño $m = \lfloor r \cdot n \rfloor$, donde $r$ es el **Gap Sentence Ratio (GSR)**, típicamente $r \in [0.15, 0.45]$.
2. **Construcción del input**: reemplazar cada $s_i \in G$ por el token especial `[MASK1]` en su posición original, preservando el orden del resto. El input al encoder es entonces:
   $$X = \big( [\text{MASK1}] \text{ si } s_i \in G \text{ else } s_i \big)_{i=1}^{n}$$
3. **Construcción del target**: concatenar las oraciones de $G$ **en su orden original dentro del documento**, separadas por un token de separación de oraciones. Este es el target $Y$ que el decoder debe generar.
4. **Loss**: cross-entropy autoregresiva estándar:
   $$\mathcal{L}_{\text{GSG}} = -\sum_{t=1}^{|Y|} \log P_\theta(y_t \mid y_{<t}, X)$$

Es importante notar que, a diferencia de T5 (que reconstruye todos los spans incluyendo separadores numerados) y de BART (que reconstruye toda la entrada original), PEGASUS **solo genera las oraciones enmascaradas**, no la entrada completa. Esto reduce el costo computacional del decoder y mantiene el objetivo más cerca del formato de un resumen.

La Figura 1 del paper muestra que cuando se combina GSG con MLM, el mismo ejemplo se procesa así: las oraciones seleccionadas como gap se reemplazan por `[MASK1]` y entran al target; las oraciones que permanecen reciben además máscaras de tokens individuales `[MASK2]` para el objetivo MLM tradicional sobre el encoder.

### Estrategias de selección de gap sentences

El paper compara tres familias:

#### Random-m

Se eligen $m$ oraciones uniformemente al azar sin reemplazo. Es el baseline natural y se usa también como ablación.

#### Lead-m

Se eligen las primeras $m$ oraciones del documento. Esta heurística respeta el llamado **lead bias** del newsroom: en noticias, las primeras oraciones tienden a contener la información más importante (pirámide invertida). Lead-m es una baseline fuerte en CNN/DailyMail y otros datasets de news.

#### Principal-m

Es la estrategia novedosa. Se asigna a cada oración $s_i$ un score de importancia definido como su ROUGE-1 F-score contra el documento sin esa oración:

$$\text{score}(s_i) = \text{rouge}\big(s_i,\; D \setminus \{s_i\}\big)$$

Luego se toman las top-$m$ oraciones con mayor score. La intuición es que las oraciones que más solapan léxicamente con el resto del documento son las más "centrales": resumen contenido distribuido en otras partes del texto.

Sobre esta idea base hay cuatro variantes según dos ejes:

- **Independent (Ind)** vs **Sequential (Seq)**:
  - **Ind**: cada oración se puntúa una sola vez contra todo el documento; se eligen las top-$m$.
  - **Seq**: se eligen oraciones una a una de forma greedy, maximizando ROUGE-1-F del conjunto seleccionado $S \cup \{s_i\}$ contra el resto del documento $D \setminus (S \cup \{s_i\})$, evitando redundancia entre oraciones elegidas. El algoritmo es el siguiente:

```
Algoritmo 1: Sequential Sentence Selection
1: S := ∅
2: para j ← 1 hasta m:
3:    s_i := rouge(S ∪ {s_i}, D \ (S ∪ {s_i})), ∀i tal que x_i ∉ S
4:    k := arg max_i {s_i}
5:    S := S ∪ {x_k}
6: fin para
```

- **Orig** vs **Uniq**:
  - **Orig**: usar la implementación original de ROUGE-1 con n-gramas contados con multiplicidad.
  - **Uniq**: tratar los n-gramas como conjuntos (sin duplicar).

Esto da cuatro combinaciones: Ind-Orig, Ind-Uniq, Seq-Orig, Seq-Uniq. En las ablaciones de PEGASUS-base sobre C4 (Figura 4a), **Ind-Orig** resulta la mejor estrategia en promedio, seguida muy de cerca por Seq-Uniq. Es la que se usa para el modelo final.

### Hyperparámetros de GSG

- **Gap Sentence Ratio (GSR)**: Figura 4b del paper compara GSR de 15%, 30%, 45%, 50%, 60% y 75%. El óptimo depende del dataset (XSum y CNN/DailyMail prefieren 15-30%, WikiHow 45%, Reddit TIFU 30%), pero **siempre por debajo de 50%** porque enmascarar demasiado pierde el contexto necesario para guiar la generación. Para el modelo final PEGASUS-large se eligió GSR ≈ 30%, con una variante "mixed-stochastic" que muestrea uniformemente entre 15% y 45%.
- **Stochastic vs deterministic**: en el modelo final se añade ruido uniforme de 20% sobre los scores de Principal y se muestrea estocásticamente entre las oraciones top, lo que actúa como regularización.
- **Long en input/target**: $L_{\text{input}} = 512$ tokens, $L_{\text{target}} = 256$ tokens en pre-entrenamiento; se amplía a 1024/256 en fine-tuning para datasets con documentos largos (BIGPATENT, arXiv, PubMed, Multi-News).

## Arquitectura

PEGASUS no propone una arquitectura nueva. Es un encoder-decoder Transformer estándar al estilo Vaswani et al. (2017) con positional encodings sinusoidales (importante porque permiten generalizar a secuencias más largas que las vistas en pre-entrenamiento, como se confirma luego en fine-tuning hasta 1024 tokens).

Dos tamaños:

| Modelo | Layers $L$ | Hidden $H$ | FFN $F$ | Heads $A$ | Params |
|--------|-----------|------------|---------|-----------|--------|
| PEGASUS-base  | 12 | 768  | 3072 | 12 | 223M |
| PEGASUS-large | 16 | 1024 | 4096 | 16 | 568M |

$L$ es el número de bloques **en cada uno** del encoder y del decoder (no en total). PEGASUS-base es comparable a BART-base; PEGASUS-large es comparable a BART-large (406M) y T5-base (220M) en orden de magnitud, aunque T5-large tiene 770M y T5-XXL llega a 11B.

El tokenizer final usa SentencePiece Unigram (Kudo, 2018) con vocabulario de 96k. La ablación de la Figura 5 compara BPE 32k contra Unigram en 32k, 64k, 96k, 128k y 256k; Unigram 96k da el mejor compromiso global.

## Pre-entrenamiento

### Corpora

Se usan dos corpora masivos:

- **C4** (Raffel et al., 2019): la versión limpia del Common Crawl, 350 millones de páginas web, 750 GB. Cobertura general y diversa.
- **HugeNews**: dataset *nuevo* del paper, 1.5 mil millones (1.5B) de artículos colectados de Common Crawl entre 2013 y 2019 (3.8 TB de texto). Se filtró usando un whitelist de dominios de news y heurísticas. **Es el aporte de dataset del paper** y nunca se liberó públicamente (solo los checkpoints entrenados sobre él), lo que ha generado críticas reproducibilidad.

### Hiperparámetros de optimización

- PEGASUS-base: 500k steps, batch 256.
- PEGASUS-large: **500k steps, batch 8192** (el final entrena por 1.5M steps en la versión mixed-stochastic). Se observó convergencia más lenta que en PEGASUS-base, que justifica el aumento de steps.
- Optimizer: **Adafactor** (Shazeer & Stern, 2018) tanto en pre-entrenamiento como en fine-tuning, con square root learning rate decay y dropout 0.1.
- Label smoothing 0.0 en pre-entrenamiento, 0.1 en fine-tuning.

### Combinación MLM + GSG

Una pregunta natural es: ¿conviene combinar el objetivo GSG con el clásico MLM de BERT? La Figura 4a sugiere que **no, en el régimen de pre-entrenamiento largo**.

La receta MLM en el paper sigue exactamente a Devlin et al. (2018): 15% de tokens en las oraciones no-gap se enmascaran, de los cuales 80% se reemplazan por `[MASK2]`, 10% por un token aleatorio y 10% se mantienen sin cambio. Se aplica solo sobre los tokens del encoder.

Resultados de la ablación (PEGASUS-base sobre C4, 500k steps):

- **MLM solamente**: drop sustancial en todas las métricas; ROUGE-1 cae a 32.20-39.33 dependiendo del dataset, claramente inferior a las variantes con GSG.
- **MLM + Ind-Orig**: performance comparable a Random, pero **inferior** a Ind-Orig solo.
- **Ind-Orig solo**: mejor en promedio.

La observación empírica más interesante es temporal: MLM + Ind-Orig converge más rápido (mejor a 100k-200k steps) pero **se estanca** al avanzar el entrenamiento. Por eso PEGASUS-large no incluye MLM en su objetivo final.

## Fine-tuning y evaluación

### Datasets downstream

Se evalúa sobre 12 benchmarks que cubren noticias, ciencia, instrucciones, emails, patentes y leyes:

| Dataset | Tamaño | Dominio | Notas |
|---------|--------|---------|-------|
| **XSum** | 227k | BBC news | Resúmenes extremos de una sola oración |
| **CNN/DailyMail** | 311k | News | Bullet-point summaries (no-anonymized variant) |
| **NEWSROOM** | 1.2M | News (38 editoriales) | Diversidad de estilos |
| **Multi-News** | 56k | News (multi-doc) | Multi-document |
| **Gigaword** | 4M | Headlines | Generar título desde primera oración |
| **WikiHow** | 168k | Instrucciones | How-to articles |
| **Reddit TIFU** | 42k | Stories informales | Sub-reddit "Today I Fucked Up" |
| **BIGPATENT** | 1.3M | Patentes US | 9 categorías |
| **arXiv** | 215k | Papers científicos | Generar abstract |
| **PubMed** | 133k | Papers médicos | Generar abstract |
| **AESLC** | 18k | Emails (Enron) | Generar subject line |
| **BillSum** | 24k | Legislación US | Bills del Congreso |

La diversidad de dominios es deliberada: el paper quiere mostrar que el objetivo GSG transfiere bien aunque el dataset downstream no sea news.

### Hyperparámetros de fine-tuning

Reportados exhaustivamente en el Appendix C. Resumen:

- Learning rate: $5\times10^{-4}$ típico para PEGASUS-base, en el rango $1\times10^{-4}$ a $8\times10^{-4}$ para PEGASUS-large.
- Number of steps: 50k para datasets pequeños y medianos, hasta 300k para BIGPATENT.
- Batch size: 256.
- Beam search durante inferencia: beam size 8, length penalty $\alpha \in [0.6, 0.9]$.
- Label smoothing 0.1.
- Max input tokens: 512 por default, 1024 para arXiv/PubMed/BIGPATENT/Multi-News/CNN-DailyMail (HugeNews).

Una observación práctica sobre el length penalty: en datasets donde el target es muy corto (XSum con max 64 tokens, AESLC con max 32) el length penalty óptimo es bajo ($\alpha \in [0.6, 0.8]$) porque penaliza generar demasiado. En datasets con resúmenes largos (Multi-News, CNN/DailyMail con max 128-256) el length penalty sube a 0.9 para no penalizar la longitud requerida. Esta calibración por dataset es una de las razones por las que en producción suele convenir mantener checkpoints separados por tarea.

Otro detalle relevante para reproducibilidad: el paper reporta que el fine-tuning de PEGASUS-large en cada dataset se hace por **grid search** sobre learning rate y length penalty. Esta búsqueda explica parte de las ganancias frente a baselines que pueden haber usado hiperparámetros default. En contextos donde no se puede hacer grid search exhaustivo, los valores recomendados son: $lr=5\times10^{-5}$, $\alpha=0.8$, beam=8.

## Resultados

### Tabla principal (Tabla 1 del paper)

Métricas ROUGE-1 / ROUGE-2 / ROUGE-L F1. Se compara TransformerBASE (sin pre-entrenamiento, mismo tamaño que PEGASUS-base), PEGASUS-base, SOTA previo, PEGASUS-large (C4), PEGASUS-large (HugeNews).

| Dataset | TransformerBASE | PEGASUS-base | Previous SOTA | PEGASUS-large (C4) | PEGASUS-large (HugeNews) |
|---------|----------------|-------------|---------------|---------------------|--------------------------|
| XSum | 30.83/10.83/24.41 | 39.79/16.58/31.70 | 45.14/22.27/37.25 | 45.20/22.06/36.99 | **47.21/24.56/39.25** |
| CNN/DailyMail | 38.27/15.03/35.48 | 41.79/18.81/38.93 | 44.16/21.28/40.90 | 43.90/21.20/40.76 | **44.17/21.47/41.11** |
| NEWSROOM | 40.28/27.93/36.52 | 42.38/30.06/38.52 | 39.91/28.38/36.87 | 45.07/33.39/41.28 | **45.15/33.51/41.33** |
| Multi-News | 34.36/5.42/15.75 | 42.24/13.27/21.44 | 43.47/14.89/17.41 | 46.74/17.95/24.26 | **47.52/18.72/24.91** |
| Gigaword | 35.70/16.75/32.83 | 36.91/17.66/34.08 | 39.14/19.92/36.57 | 39.12/19.86/36.24 | **39.12/19.86/36.24** |
| WikiHow | 32.48/10.53/23.86 | 36.58/15.64/30.01 | 28.53/9.23/26.54 | **43.06/19.71/34.80** | 41.35/18.51/33.42 |
| Reddit TIFU | 15.89/1.94/12.22 | 24.36/6.09/18.75 | 19.0/3.7/15.1 | **26.54/8.94/21.64** | 26.63/9.01/21.60 |
| BIGPATENT | 42.98/20.51/31.87 | 43.55/20.43/31.80 | 37.52/10.63/22.79 | **53.63/33.16/42.25** | 53.41/32.89/42.07 |
| arXiv | 35.63/7.95/20.00 | 34.81/10.16/22.50 | 41.59/14.26/23.55 | **44.70/17.27/25.80** | 44.67/17.18/25.73 |
| PubMed | 33.94/7.43/19.02 | 39.98/15.15/25.23 | 40.59/15.59/23.59 | **45.49/19.90/27.69** | 45.09/19.56/27.42 |
| AESLC | 15.04/7.39/14.93 | 34.85/18.94/34.10 | 23.67/10.29/23.44 | **37.69/21.85/36.84** | 37.40/21.22/36.45 |
| BillSum | 44.05/21.30/30.98 | 51.42/29.68/37.78 | 40.80/23.83/33.73 | 57.20/39.56/45.80 | **57.31/40.19/45.82** |

Observaciones:

- **PEGASUS-large (HugeNews) gana en news**, especialmente XSum donde el salto frente al SOTA previo (BART-large) es notable: 47.21 vs 45.14 en ROUGE-1.
- **PEGASUS-large (C4) gana en non-news**: WikiHow, Reddit TIFU, BIGPATENT, arXiv, PubMed, AESLC. El corpus genérico transfiere mejor a dominios fuera de noticias.
- **PEGASUS-base ya supera el SOTA previo** en NEWSROOM, Multi-News, WikiHow, Reddit TIFU, BIGPATENT, AESLC, BillSum — siete de doce — pese a tener menos parámetros que el SOTA.
- La diferencia entre TransformerBASE y PEGASUS-base es enorme en datasets pequeños: en AESLC, ROUGE-2 pasa de 7.39 a 18.94 (casi 2.6x); en Reddit TIFU, de 1.94 a 6.09 (3.1x). El pre-entrenamiento importa más cuando el dataset es chico.

### Tabla 2: comparación con otros modelos pre-entrenados

| Modelo | XSum (R1/R2/RL) | CNN/DailyMail | Gigaword |
|--------|-----------------|---------------|----------|
| BERTShare (Rothe et al., 2019) | 38.52/16.12/31.13 | 39.25/18.09/36.45 | 38.13/19.81/35.62 |
| MASS (Song et al., 2019) | 39.75/17.24/31.95 | 42.12/19.50/39.01 | 38.73/19.71/35.96 |
| UniLM (Dong et al., 2019) | — | 43.33/20.21/40.51 | 38.45/19.45/35.75 |
| BART (Lewis et al., 2019) | 45.14/22.27/37.25 | 44.16/21.28/40.90 | — |
| T5 (Raffel et al., 2019) | — | 43.52/21.55/40.69 | — |
| **PEGASUS-large (C4)** | 45.20/22.06/36.99 | 43.90/21.20/40.76 | 38.75/19.86/36.14 |
| **PEGASUS-large (HugeNews)** | **47.21/24.56/39.25** | **44.17/21.47/41.11** | 39.12/19.86/36.24 |

PEGASUS bate a todos. La diferencia es especialmente clara en XSum (el dataset más abstractive de todos, con resúmenes de una sola oración que no pueden simplemente copiarse del documento).

### Low-resource y zero-shot summarization (Figura 6, Tabla E.1)

Este es probablemente el resultado más impactante del paper para la práctica:

- **Zero-shot**: PEGASUS-large sin fine-tuning produce summaries decentes en news; ROUGE-2 de 13.28 en CNN/DailyMail (más de 50% mejor que GPT-2 con ROUGE-2 = 8.27).
- **10 ejemplos**: 15.84 R2 en CNN/DailyMail.
- **100 ejemplos**: ya supera el SOTA previo en BIGPATENT, Reddit TIFU y BillSum.
- **1000 ejemplos**: supera el SOTA previo en seis datasets (Multi-News, WikiHow, Reddit TIFU, BIGPATENT, AESLC, BillSum).
- **10k ejemplos**: comparable a Transformer-base entrenado con dataset completo en datasets de 20k-200k ejemplos.

La intuición es que GSG ya enseñó al modelo a generar texto en formato resumen; el fine-tuning solo le indica el estilo/longitud específicos del dataset target.

### Human evaluation (Tabla 3)

En Amazon Mechanical Turk, tres jueces por ejemplo, escala Likert 1-5. Se comparan PEGASUS-large (HugeNews), PEGASUS-large (C4), TransformerBASE y resúmenes humanos.

| Dataset | Humano | PEGASUS-large (HugeNews) | PEGASUS-large (C4) | TransformerBASE |
|---------|--------|-------------------------|---------------------|-----------------|
| XSum | 3.0 | 3.1 ($p=0.7$) | 3.0 ($p=0.0001$ vs human, peor) | 2.0 ($p=3\text{e-}10$, peor) |
| CNN/DailyMail | 3.1 | 3.6 ($p=0.007$, **mejor que humano**) | 3.6 ($p=0.009$, **mejor que humano**) | 2.9 ($p=0.06$) |
| Reddit TIFU | 3.2 | 3.1 ($p=0.3$) | 3.1 ($p=0.7$) | 1.4 ($p=5\text{e-}23$) |

Estadísticamente, PEGASUS no es peor que un humano en XSum, CNN/DailyMail ni Reddit TIFU. En CNN/DailyMail incluso se evalúa mejor que el resumen humano de referencia. En low-resource (experimento 2 de la tabla), 100 ejemplos ya bastan para alcanzar paridad humana en XSum y CNN/DailyMail (no en Reddit TIFU, donde la diversidad de estilos requiere supervisión completa).

### Test-set overlap

Los autores se cuestionan si las mejoras vienen de **memorización**: HugeNews y C4 son crawls de la web; podrían contener artículos cuyas summaries están en los test sets de CNN/DailyMail o XSum. La Figura 7 mide overlap usando ROUGE-2 recall entre test targets y documentos de pre-entrenamiento y filtra los test examples con similaridad >0.8 o >1.0.

Resultado: solo XSum muestra overlap significativo (15-20%), y filtrar esos ejemplos cambia ROUGE en menos de 1%. La inspección manual de ejemplos con similaridad 1.0 confirma que el modelo no produce copias literales del corpus de pre-entrenamiento. La conclusión es que las ganancias **no se explican por memorización**.

## Limitaciones reconocibles

El paper no incluye una sección explícita de limitaciones (la práctica era menos común en 2020), pero pueden inferirse del análisis y de literatura posterior:

1. **Alucinaciones factuales**: la Figura G.1 del Appendix muestra un caso CNN/DailyMail donde PEGASUS produce un resumen fluido y coherente que confunde "California" con "north carolina". El paper reconoce que ROUGE penaliza injustamente este tipo de outputs en algunos casos, pero también que el modelo aún genera alucinaciones, especialmente para entidades nombradas y números.
2. **Sesgo extractivo residual**: aunque GSG empuja hacia generación abstractive, el modelo todavía reusa frases del documento. Esto es problemático en datasets como XSum donde se espera reformulación.
3. **Solo inglés**: PEGASUS original es monolingüe. Extensiones multilingües (mPEGASUS) llegaron después.
4. **Sin instruction tuning**: cada dataset requiere fine-tuning específico; no hay una "instrucción" universal como en T5 o FLAN. Esto contrasta con la generación posterior de modelos instruction-tuned.
5. **Costo de Principal-Ind**: para documentos muy largos, calcular ROUGE-1 F-score entre cada oración y el resto del documento es $O(n)$ por oración. Manejable en pre-entrenamiento offline, pero no en tiempo real.
6. **HugeNews no es público**: la mejor versión del modelo se entrenó con un corpus privado. Esto compromete la reproducibilidad y forzó a la comunidad a usar `pegasus-large` checkpoints sin poder re-entrenar desde cero con el mismo dato.
7. **Documentos largos**: limitado a 1024 tokens en fine-tuning. Esto excluye papers científicos completos, libros, transcripciones largas, etc. Resuelto parcialmente en PEGASUS-X.
8. **ROUGE como métrica**: Kryscinski et al. (2019) ya señalaba en 2019 que ROUGE penaliza approaches abstractive y no captura factualidad. PEGASUS optimiza explícitamente ROUGE, lo que puede ser un sesgo.
9. **Sesgo del corpus de news**: HugeNews, por construcción, sobre-representa lenguaje periodístico y eventos noticiosos. Esto se nota en el comportamiento del modelo: tiende a producir resúmenes con estructura de lead noticioso (quién hizo qué, dónde, cuándo), lo que puede ser inapropiado para géneros como literatura, ciencia o conversación.
10. **GSR fijo vs adaptativo**: el paper elige un GSR fijo o muestreado uniformemente entre 15-45%, pero distintos documentos podrían beneficiarse de GSRs distintos según su densidad informativa. No se explora pre-entrenamiento adaptativo donde el modelo aprenda a elegir cuántas oraciones enmascarar.
11. **Single-document summarization**: aunque incluye Multi-News en evaluación, el objetivo GSG está pensado para un único documento. No hay un objetivo análogo para multi-document summarization donde el modelo deba consolidar información de fuentes distintas.

## Sucesores e impacto

PEGASUS ha sido extraordinariamente influyente. Los hilos directos:

- **PEGASUS-X** (Phang et al., 2022): extensión a documentos largos hasta 16k tokens usando staggered block-local attention. Misma idea base de GSG pero arquitectura optimizada para long-context.
- **mPEGASUS**: variantes multilingües entrenadas sobre corpus en otros idiomas, especialmente en el contexto de evaluaciones XLSum y WikiLingua.
- **FLAN-T5 y FLAN-PEGASUS**: cuando llegó instruction tuning, los autores y la comunidad mostraron que pre-entrenar con GSG y luego instruction-tunear da resultados superiores en summarization.
- **GSG en otras tareas**: el meta-pattern de "elegir un subconjunto importante del input y hacer al modelo regenerarlo" se ha aplicado a dialogue summarization, code summarization y meeting summarization.

En el ecosistema HuggingFace:

- `google/pegasus-cnn_dailymail`
- `google/pegasus-xsum`
- `google/pegasus-multi_news`
- `google/pegasus-pubmed`
- `google/pegasus-arxiv`
- `google/pegasus-billsum`
- `google/pegasus-large` (sin fine-tuning, para zero-shot o fine-tuning custom)

Todos están integrados en el pipeline `summarization` con un par de líneas:

```python
from transformers import pipeline
summarizer = pipeline("summarization", model="google/pegasus-xsum")
summarizer(documento)
```

Hoy en 2026, aunque modelos más grandes (T5-11B, FLAN-T5-XXL, modelos basados en LLaMA y derivados) cubren summarization como parte de capacidades generales, PEGASUS-large sigue siendo competitivo en ratio calidad/parámetros: 568M params es deployable en una sola GPU consumer.

## Comparación T5 vs BART vs PEGASUS

| Aspecto | T5 (Raffel 2019) | BART (Lewis 2019) | PEGASUS (Zhang 2020) |
|---------|------------------|-------------------|-----------------------|
| Arquitectura | Encoder-decoder | Encoder-decoder | Encoder-decoder |
| Tamaño "large" | 220M (T5-base) - 11B (T5-XXL) | 406M | 568M |
| Pre-training objective | Span corruption (text infilling) con spans cortos | Multiple noising functions (text infilling, sentence permutation, etc.) | **Gap Sentence Generation** sobre oraciones completas seleccionadas por importancia |
| Pre-training data | C4 (750 GB) | 160 GB news/books/web | C4 + HugeNews (3.8 TB) |
| Genera input completo o solo parte enmascarada | Spans entre sentinels | Toda la entrada original | Solo las oraciones enmascaradas (más cerca de un resumen) |
| Tarea para la que está optimizado | Text-to-text general | Generación general, comprensión | **Summarization** (específico) |
| Multitarea en pre-training | Sí (T5-style prompts) | No | No |
| ROUGE en XSum (R1/R2/RL) | — | 45.14/22.27/37.25 | **47.21/24.56/39.25** |
| ROUGE en CNN/DM (R1/R2/RL) | 43.52/21.55/40.69 | 44.16/21.28/40.90 | **44.17/21.47/41.11** |
| Instruction-tunable | Sí (es el corazón de FLAN-T5) | No nativo | No nativo |
| Multilingüe | mT5 disponible | mBART disponible | Solo inglés (PEGASUS original) |

**Conclusión cualitativa**:

- En **summarization extrema** (XSum), PEGASUS gana claramente porque GSG fue diseñado exactamente para ese régimen.
- En **summarization de news con resúmenes más largos** (CNN/DailyMail), los tres modelos están casi empatados; PEGASUS gana por margen estrecho.
- En **tareas no-summarization** (clasificación, QA, parsing), T5 y BART son mejores opciones porque su pre-training objective es más general.
- Para **producción**, T5 ofrece más flexibilidad multitask y multilingüe; BART es buen all-rounder; PEGASUS es la opción más fuerte cuando la tarea es claramente summarization en inglés.

## Conexión con la clase 22 y aplicación práctica

En la clase 22 del curso, PEGASUS se ubica como **el modelo summarization-specific que complementa a T5 (general-purpose)** en el toolkit moderno de NLP. Para Roberto, que viene del mundo FHIR y healthcare, la conexión práctica es clara: en aplicaciones de summarization clínica (resumir notas de evolución, EHR, papers médicos), la elección típica del 2020-2024 ha sido entre:

- **PEGASUS-pubmed**: pre-entrenado en C4 y fine-tuned en PubMed; estándar de hecho para summarization de abstracts médicos.
- **BioBART / ClinicalBART**: variantes de BART en dominio biomédico.
- **T5/FLAN-T5**: cuando se necesita flexibilidad (resumir, traducir, clasificar) en el mismo modelo.

La lección operativa de PEGASUS — **alinear el objetivo de pre-entrenamiento con la tarea downstream** — es trasladable a otros dominios. Un modelo para extracción de entidades clínicas podría beneficiarse de un objetivo que enmascara entidades en lugar de spans aleatorios; un modelo para CDA-to-FHIR conversion podría pre-entrenarse con noising functions específicas de XML/JSON estructurado.

El otro takeaway, complementario, es que **el régimen low-resource cambia la economía de la summarization**: con 100-1000 ejemplos curados de calidad, un PEGASUS-large fine-tuned puede alcanzar paridad con modelos entrenados sobre datasets de cientos de miles. Para un caso de uso FHIR donde recolectar pares (nota clínica, resumen estructurado) es costoso, esto vale literalmente meses de anotación.

Un tercer aprendizaje, más metodológico, es **cómo el paper hace ablations**. Los autores reportan figuras y tablas comparando seis variantes de GSG, seis valores de GSR, siete configuraciones de vocabulario, y dos corpora — todo sobre un modelo más pequeño (PEGASUS-base) antes de escalar al modelo final. Esta disciplina de "ablations sobre modelo pequeño, scaling sobre modelo grande con los mejores settings" es práctica estándar hoy pero en 2019-2020 todavía no era universal. Refleja un cambio en cómo la comunidad piensa el costo de entrenamiento de modelos grandes: cada training run cuenta y conviene tomar decisiones informadas antes del compute final.

## Análisis crítico personal

A casi seis años de su publicación (2020-2026), PEGASUS sigue siendo un paper extraordinariamente sólido. Sus aportes son verificables, reproducibles (con la salvedad de HugeNews) y han pasado la prueba del tiempo. Tres aspectos merecen comentario crítico adicional:

**Primero, sobre la métrica de selección Principal**: usar ROUGE-1 F-score como proxy de importancia tiene una circularidad interesante. ROUGE es la métrica que se usa para evaluar summarization downstream y también el criterio de selección durante pre-entrenamiento. Esto crea un sesgo metodológico: el modelo aprende a producir output que maximiza el solapamiento léxico, lo que es exactamente lo que ROUGE mide. Si la evaluación usara métricas más semánticas (BERTScore, BLEURT, factuality scores), no es obvio que Principal siga ganando frente a Random o Lead. Sería interesante una ablación con la selección de gap sentences basada en embedding similarity (e.g. sentence-BERT) en lugar de ROUGE.

**Segundo, sobre la separación de corpus**: el hallazgo de que HugeNews es mejor para news downstream pero C4 es mejor para non-news refuerza la importancia del **dominio del corpus de pre-entrenamiento**. Esto se ha confirmado en muchos modelos posteriores: BioBERT, ClinicalBERT, SciBERT, LegalBERT. La pregunta abierta es si conviene un PEGASUS-medical entrenado con PubMed full-text y notas clínicas desidentificadas. Existen variantes de la comunidad pero no oficiales.

**Tercero, sobre la longitud**: 512 tokens en pre-entrenamiento limita seriamente la utilidad para documentos largos. Que el modelo generalice a 1024 tokens en fine-tuning es notable pero no resuelve el problema de papers científicos (10-20k tokens), libros, contratos largos. PEGASUS-X (2022) atacó esto con block-local attention, pero el costo es perder algunas garantías del Transformer estándar. Los modelos modernos basados en LLaMA con context windows de 32k-128k tokens hacen este problema menos urgente, pero a costo de mucho mayor compute por inferencia.

En suma: PEGASUS es un paper que vale la pena leer no solo por sus resultados sino por su **metodología limpia** — un objetivo de pre-entrenamiento bien motivado, ablations rigurosas, comparación honesta con baselines, evaluación humana, y análisis de memorización para descartar artefactos. Es un buen ejemplo de cómo escribir un paper de ML aplicado en NLP.

## Lecturas recomendadas y referencias internas

- **Vaswani et al. (2017)** — la arquitectura encoder-decoder Transformer subyacente.
- **Devlin et al. (2018)** — BERT y la receta MLM que PEGASUS evalúa como ablación.
- **Raffel et al. (2019)** — T5, C4 y el framework text-to-text que PEGASUS reutiliza.
- **Lewis et al. (2019)** — BART, el competidor directo en summarization general.
- **Song et al. (2019)** — MASS, predecesor conceptual de GSG (un solo span enmascarado, sin selección por importancia).
- **Narayan et al. (2018)** — XSum, el dataset donde PEGASUS demuestra su ventaja más clara.
- **Lin (2004)** — ROUGE, la métrica y también el criterio de selección de Principal.
- **Kudo (2018)** — SentencePiece Unigram, el tokenizer usado.
- **Kryscinski et al. (2019)** — crítica de ROUGE como métrica de summarization.

Para el curso IA UC, este paper se conecta hacia atrás con la clase 20 (BERT/GPT/ELMo: pre-training paradigms) y la clase 21 (Transformers seq-to-seq y attention), y hacia adelante con cualquier sesión sobre instruction tuning, RLHF y modelos de generación de texto generales (donde el problema de alinear objetivo de pre-training con downstream task vuelve, pero con la palanca del prompt en vez del objetivo de loss).
