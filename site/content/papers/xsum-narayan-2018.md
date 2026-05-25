---
title: "XSum (Extreme Summarization Dataset)"
weight: 116
math: true
---

{{< paper-card
    title="Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization"
    authors="Narayan, Cohen, Lapata"
    year="2018"
    venue="EMNLP 2018"
    pdf="/papers/xsum-narayan-2018.pdf"
    arxiv="1808.08745" >}}
**226.711 artículos de la BBC** con resúmenes profesionales de **una sola oración** escritos por periodistas. A diferencia de CNN/DailyMail —donde LEAD-3 ya alcanza ROUGE-1 ≈ 40— en XSum el **83% de los bigramas del resumen son novel** (no aparecen en el documento). Se convirtió en el benchmark canónico de summarization verdaderamente abstractive y catalizó la era de pre-training generativo (BART, PEGASUS, T5).
{{< /paper-card >}}

---

## El problema: CNN/DailyMail recompensa la extracción

Hacia 2018 el campo de summarization vivía una paradoja incómoda. Desde Rush et al. (2015) y See et al. (2017), una sucesión de encoder-decoder abstractives había sido evaluada en **CNN/DailyMail** (Hermann et al. 2015) y se reportaban mejoras incrementales. Pero un detalle delataba el truco: el baseline **LEAD-3** —tomar las tres primeras oraciones del artículo— alcanzaba ROUGE-1 ≈ 40.68 en DailyMail y ≈ 29.15 en CNN, niveles comparables al estado del arte.

La consecuencia operativa: los modelos "abstractives" entrenados en CNN/DM aprendían a comportarse como extractores sofisticados. El paper documenta que los resúmenes generados por See et al. (2017) tenían menos del 5% de bigramas novedosos, mientras que el gold humano tenía ≈ 54%. **Si la tarea no penaliza la extracción, ningún modelo aprenderá a abstractar.**

Narayan, Cohen y Lapata articulan el problema como diseño de tarea: el campo necesita un dataset donde (1) LEAD-k sea pésimo, (2) el oracle extractivo esté lejos del techo, y (3) el target sea **single-sentence** para eliminar la concatenación de oraciones importantes.

## El dataset XSum

### Recolección y diseño

Los autores explotan una convención editorial de la BBC: cada artículo comienza con una oración introductoria con clase HTML `story-body__introduction`, escrita típicamente por el autor del artículo y que responde "¿de qué trata este artículo?". Esa oración se aísla como **target** y el resto del artículo es el **source**. El método es elegante: no requiere alineamiento heurístico ni anotación manual.

Recolectan **226.711 artículos** del Wayback Machine de BBC.co.uk cubriendo aproximadamente 2010–2017, con cobertura temática amplia (News, Politics, Sports, Business, Tech, Science, Health, Entertainment, etc.).

### Splits

- **Train**: 204.045 (90%).
- **Validation**: 11.332 (5%).
- **Test**: 11.334 (5%).

División por identificadores únicos de URL para evitar data leakage entre splits.

### Estadísticas comparadas

| Dataset    | doc len (words) | summary len (words) | summary len (sent.) |
| ---------- | --------------- | ------------------- | ------------------- |
| CNN        | 760.50          | 45.70               | 3.59                |
| DailyMail  | 653.33          | 54.65               | 3.86                |
| NY Times   | 800.04          | 45.54               | 2.44                |
| **XSum**   | **431.07**      | **23.26**           | **1.00**            |

El documento promedio es **más corto** (431 palabras), el resumen es **mucho más corto** (23 palabras) y siempre tiene **exactamente una oración** —de ahí el nombre "extreme summarization".

### Novelty: el indicador decisivo

El porcentaje de **n-gramas novedosos** del gold (n-gramas que aparecen en el resumen pero no en el documento):

| Dataset    | unigramas | bigramas  | trigramas | 4-gramas  |
| ---------- | --------- | --------- | --------- | --------- |
| CNN        | 16.75     | 54.33     | 72.42     | 80.37     |
| DailyMail  | 17.03     | 53.78     | 72.14     | 80.28     |
| **XSum**   | **35.76** | **83.45** | **95.50** | **98.49** |

XSum **duplica** los unigramas novedosos respecto de CNN/DM y supera por amplio margen los bigramas (83% vs 54%). Para 4-gramas, **el 98% son construcciones nuevas no presentes en el artículo**. Esto es la definición operacional de summarization verdaderamente abstractive.

### Baselines extractivos: la prueba decisiva

| Dataset    | LEAD R-1  | EXT-ORACLE R-1 |
| ---------- | --------- | -------------- |
| CNN        | 29.15     | 50.38          |
| DailyMail  | 40.68     | 55.12          |
| **XSum**   | **16.30** | **29.79**      |

En XSum, **la mejor oración posible del documento (EXT-ORACLE) sólo alcanza ROUGE-1 = 29.79** —comparable al LEAD-3 de CNN/DM. Cualquier modelo que supere ROUGE-1 ≈ 30 en XSum necesariamente está haciendo abstracción real.

## TConvS2S — el modelo del paper

La segunda contribución es **TConvS2S** (Topic-Aware Convolutional Seq2Seq), una extensión del ConvS2S de Gehring et al. (2017) con condicionamiento por tópicos LDA.

### ConvS2S base

- Encoder y decoder convolucionales 1-D con **Gated Linear Units** (Dauphin et al. 2017) y residual connections.
- **Multi-hop attention**: a diferencia del Bahdanau attention clásico, ConvS2S aplica atención en cada capa del decoder. Para cada palabra del decoder se computa:

$$a^{\ell}_{ij} = \frac{\exp(d^{\ell}_i \cdot z^u_j)}{\sum_{t=1}^{m} \exp(d^{\ell}_i \cdot z^u_t)}, \quad c^{\ell}_i = \sum_{j=1}^{m} a^{\ell}_{ij} (z^u_j + e_j)$$

Los autores eligen CNN sobre RNN por dos razones: campo receptivo jerárquico que crece geométricamente con la profundidad (no sufre vanishing gradients en documentos de 400 tokens) y paralelización en training. Argumento razonable en 2018; pronto superado por Transformers, pero válido en su momento.

### Topic conditioning vía LDA

La hipótesis arquitectónica: para generar un resumen coherente, el modelo necesita saber **de qué trata globalmente el documento**, no sólo qué palabras hay localmente. La atención multi-hop captura dependencias léxicas pero no "tema". **LDA con 512 tópicos** (Blei et al. 2003) entrenado sobre el train split de XSum aporta exactamente eso.

Para cada documento $D$ se computan dos vectores:

- $t_D \in \mathbb{R}^{f'}$: distribución de tópicos del documento completo (vector global).
- $t'_i \in \mathbb{R}^{f'}$: distribución de tópicos de cada palabra $w_i$.

**Inyección en el encoder**: cada token de entrada se representa como

$$e_i = [(x_i + p_i); (t'_i \odot t_D)] \in \mathbb{R}^{f + f'}$$

con producto Hadamard entre el tópico local del token y el tópico global del documento. Palabras alineadas con el tema del documento reciben pesos altos; léxico genérico (preposiciones, artículos) tiende a cero.

**Inyección en el decoder**: cada token generado se representa como

$$g_i = [(x'_i + p'_i); t_D]$$

donde $t_D$ se broadcast a cada posición. Esto fuerza al resumen a "permanecer en el tema" del documento de entrada.

Ejemplos de tópicos aprendidos: T1 (judicial: charge, court, murder, police, arrest), T2 (eclesiástico: church, abuse, bishop, catholic, pope), T4 (político electoral: clinton, trump, vote, election, debate), T6 (sanitario: hospital, patient, nhs, care, health).

## Baselines y resultados

Comparados todos contra `LEAD`, `EXT-ORACLE`, `Seq2Seq`, `PtGen` (See et al. 2017), `PtGen-Covg`, `ConvS2S` y las variantes de `T-ConvS2S`:

| Modelo                                  | ROUGE-1   | ROUGE-2   | ROUGE-L   |
| --------------------------------------- | --------- | --------- | --------- |
| LEAD                                    | 16.30     | 1.60      | 11.95     |
| EXT-ORACLE                              | 29.79     | 8.81      | 22.66     |
| Seq2Seq                                 | 28.42     | 8.77      | 22.48     |
| PtGen (See 2017)                        | 29.70     | 9.21      | 23.24     |
| PtGen-Covg                              | 28.10     | 8.02      | 21.72     |
| ConvS2S                                 | 31.27     | 11.07     | 25.23     |
| **T-ConvS2S** (full)                    | **31.89** | **11.54** | **25.75** |

Lecturas clave:

1. **PtGen-Covg empeora** respecto de PtGen vanilla. El coverage mechanism —diseñado para evitar repeticiones en resúmenes largos de CNN/DM— es contraproducente para single-sentence summaries. Las arquitecturas optimizadas para CNN/DM no transfieren naturalmente.
2. **PtGen supera a EXT-ORACLE** en R-2 y R-L. El modelo abstractive realmente aporta información que el oracle extractive no puede ofrecer.
3. **ConvS2S vanilla ya supera a todos los RNN-based**. La arquitectura convolucional es ventajosa en este task.
4. **T-ConvS2S supera consistentemente a ConvS2S** (+0.62 R-1). Ganancia modesta pero robusta, mantenida en R-2 y R-L.

La evaluación humana —Best-Worst Scaling y QA-based— coloca a T-ConvS2S como el único sistema neural con score positivo y con accuracy de **46.05%** en preguntas factuales contestadas leyendo sólo el resumen (vs PtGen 21.40%, ConvS2S 30.90%, GOLD 97.23%).

## Hallucinations: el problema crítico que el paper documenta

Aunque T-ConvS2S domina las métricas, el paper documenta —sin resolver— el problema más serio de la summarization neuronal abstractive: **alucinaciones**, detalles falsos no soportados por el documento fuente.

Ejemplo del paper (artículo sobre Zac Goldsmith candidato a alcalde de Londres):

| Sistema    | Resumen                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------ |
| PtGen      | UKIP leader **Nigel** Goldsmith has been **elected as the new mayor** of London...               |
| ConvS2S    | London mayoral candidate Zac Goldsmith has been **elected as the new mayor** of London.          |
| T-ConvS2S  | Former London mayoral candidate Zac Goldsmith has been chosen to stand in the mayoral election. |
| GOLD       | Zac Goldsmith will contest the 2016 London mayoral election for the conservatives.              |

PtGen confunde nombre y partido (Nigel ≠ Zac, UKIP ≠ conservatives) y resultado (candidato ≠ ganador). ConvS2S también confunde el resultado. T-ConvS2S es el único factualmente correcto. **ROUGE no penaliza estas alucinaciones siempre que mantengan superposición léxica con el gold**.

Este problema motivará una literatura completa post-2018 sobre faithfulness: **Maynez et al. (2020)** —"On Faithfulness and Factuality in Abstractive Summarization"— reporta que **más del 70% de los resúmenes** generados por BART/PEGASUS sobre XSum contienen alucinaciones extrínsecas. **FactCC** (Kryscinski 2020), **QAGS** (Wang 2020), **QuestEval** y **BARTScore** son métricas que surgen para detectar y mitigar este problema.

## Por qué XSum cambió el campo

Tres cambios estructurales:

### 1. Redefinición de qué cuenta como summarization abstractive

Antes de XSum, "abstractive summarization" era un término aspiracional aplicado a modelos que copiaban casi todo. XSum hizo evidente —vía métricas de novelty y baselines— que CNN/DM no era un benchmark adecuado para esa aspiración. Post-XSum, todo paper serio reporta resultados en XSum además de CNN/DM, justamente para mostrar que su modelo no es un extractor disfrazado.

### 2. Catalizador del paradigma generativo pre-entrenado

Los modelos que dominan XSum no son convolucionales sino Transformers pre-entrenados generativamente:

- **BART** (Lewis et al. 2020): denoising autoencoder. XSum ROUGE-1 = **45.14**.
- **PEGASUS** (Zhang et al. 2020): pre-training *Gap Sentences Generation* específicamente diseñado para summarization. XSum ROUGE-1 = **47.21** (SOTA por largo tiempo).
- **T5** (Raffel et al. 2020): text-to-text framework, reporta XSum entre sus tareas canónicas.
- **GPT-3/4** y descendientes: zero-shot y few-shot competitivos, con alta variabilidad en factualidad.

PEGASUS, en particular, fue diseñado pensando en XSum: su objetivo de pre-entrenamiento (enmascarar oraciones importantes y reconstruirlas) imita exactamente la estructura del task.

### 3. Disparador de la agenda de faithfulness

Las alucinaciones que el paper identifica en T-ConvS2S se vuelven endémicas en BART y PEGASUS sobre XSum precisamente porque la demanda abstractive fuerza al modelo a generar contenido nuevo. Esto motiva toda una sub-área: factuality-aware summarization, hallucination detection, retrieval-augmented summarization.

## Limitaciones

A casi una década del paper, la literatura ha clarificado varias limitaciones:

- **Sesgo de fuente única**: sólo BBC, con estilo periodístico británico y sobre-representación de UK politics. Newsroom (Grusky et al. 2018) lo aborda agregando 38 outlets.
- **Rigidez de target de una oración**: algunos artículos legítimamente necesitan más, y el formato fuerza compresión excesiva que induce alucinación.
- **ROUGE como métrica única**: no detecta alucinaciones, premia overlap léxico aun si la oración es factualmente falsa, y penaliza paráfrasis válidas con vocabulario distinto del gold.
- **LDA con 512 tópicos**: en 2026, con LLMs que internalizan tópicos vía self-attention, pre-computar distribuciones de Dirichlet y concatenarlas como features extra es una arquitectura de transición. La idea —condicionar la generación en una representación global del documento— sobrevive implementada con encoders Transformer.
- **Truncamiento a 400 tokens**: limitación computacional de 2018 sobre M40s; modelos modernos con contexto 32k+ ya no la sufren.
- **Vocabulario fijo de 50k palabras sin BPE/SentencePiece**: OOV frecuentes en nombres propios. PtGen mitiga vía copy mechanism; ConvS2S/T-ConvS2S no.

## Impacto en el ecosistema

- **Datasets sucesores**: Newsroom (Grusky 2018, 38 outlets), Multi-News (Fabbri 2019, multi-doc), Reddit-TIFU (Kim 2019, informal), BigPatent (Sharma 2019, técnico), arXiv/PubMed (Cohan 2018, papers científicos), WikiHow (Koupaee 2018).
- **Modelos optimizados para XSum**: BertSumAbs (Liu & Lapata 2019), BART, PEGASUS, T5, LongT5, LED, BigBird.
- **HuggingFace `datasets.load_dataset("xsum")`** es loader estándar. Es uno de los benchmarks por defecto en Papers With Code y en evaluaciones de modelos comerciales (GPT-4, Claude, Gemini).

## Conexión con la clase 22

El slide 12 del material enumera los datasets canónicos de summarization e incluye explícitamente XSum:

> *"X-Sum. Extreme summarization of news into a short one-sentence summary composed of 225.000 examples."*

El slide 13 formaliza la tarea:

> *"X-Sum. article → one-sentence summary"*

XSum es uno de los **5 datasets canónicos** que la clase discute (CNN/DM, NY Times, XSum, Newsroom y multi-document benchmarks), cada uno representando una dimensión distinta del espectro extractive ↔ abstractive. Su rol triple en la narrativa de la clase: (1) benchmark abstractive de referencia, (2) motivador del pre-training generativo —los slides sobre BART, PEGASUS y T5 toman a XSum como punto de evaluación— y (3) caso de estudio canónico para hallucinations en NLG.

## Cierre

XSum es un paper donde el dataset eclipsa al modelo. La contribución arquitectónica —T-ConvS2S con condicionamiento LDA— fue importante en 2018 pero quedó superada en pocos meses por Transformers pre-entrenados. Lo que persiste es la decisión metodológica: identificar que los benchmarks existentes recompensaban la extracción y construir uno que la castigara. Ese gesto cambió la trayectoria del campo más que cualquier mejora incremental en ROUGE.

El precio de esa exigencia es el problema de las alucinaciones, que XSum hereda como herida abierta y que en 2026 sigue siendo área activa de investigación. La tensión entre forzar abstracción y mantener factualidad es la pregunta central de NLG aplicado, y XSum es el dataset donde esa pregunta se hace más visible.

## Notas y enlaces

**Fundamentos relacionados**:

- [Text summarization](/fundamentos/text-summarization)
- [ROUGE metric](/fundamentos/rouge-metric)

**Papers relacionados**:

- [Pointer-Generator (See 2017)](/papers/pointer-generator-see-2017) — baseline RNN principal del paper.
- [PEGASUS (Zhang 2020)](/papers/pegasus-zhang-2020) — SOTA en XSum por largo tiempo (R-1 = 47.21).
- [BART (Lewis 2020)](/papers/bart-lewis-2020) — denoising seq2seq Transformer (R-1 = 45.14).
- [T5 (Raffel 2020)](/papers/t5-raffel-2020) — text-to-text framework, reporta XSum.

**Clase**:

- [Clase 22 — Summarization y NLG](/clases/clase-22)
