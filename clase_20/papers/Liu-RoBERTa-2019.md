# Análisis interno — Liu et al. (2019) "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

> Documento complementario al material público del site (`papers/roberta-liu-2019.md`, `fundamentos/bert.md`, `fundamentos/pretraining-bert.md`). Aquí se profundiza en aspectos que esos archivos cubren superficialmente: la tesis "BERT fue undertrained", las cinco modificaciones del régimen de pre-training, las ablations sobre NSP / masking / batch size / data, la decisión de adoptar byte-level BPE, comparación numérica fina contra BERT y XLNet, limitaciones del paper (costo de cómputo, ausencia de novedad arquitectónica), y conexión con la clase 20 y el lab 20 del Diplomado IA UC.

- **Paper**: Liu, Ott, Goyal, Du, Joshi, Chen, Levy, Lewis, Zettlemoyer, Stoyanov. *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692v1 (26 Jul 2019).
- **Versiones**: v1 (Jul 2019) — única versión publicada en arXiv. No hubo revisión posterior. El paper fue rechazado por ICLR 2020 ("not novel enough"), lo que se volvió un meme en la comunidad porque RoBERTa terminó siendo uno de los modelos más citados y replicados de la era encoder-only.
- **Código y checkpoints**: `https://github.com/facebookresearch/fairseq` (módulo `examples/roberta`). Modelos liberados bajo MIT: `roberta.base` (125M params), `roberta.large` (355M params). También integrados desde el día uno en `transformers` de HuggingFace bajo el identificador `roberta-base` y `roberta-large`.
- **Institución**: Facebook AI Research (FAIR) en colaboración con Paul G. Allen School (University of Washington). Liu, Ott, Goyal, Du, Joshi, Chen, Lewis, Stoyanov son FAIR. Levy y Zettlemoyer son UW. Este detalle importa porque marca el momento en que FAIR entra de lleno en la competencia de modelos de pre-training a gran escala, hasta entonces dominada por Google (BERT, T5 venía en 2020) y OpenAI (GPT-1, GPT-2).

---

## 1. La tesis del paper en una frase

> **"BERT está significativamente undertrained, y puede igualar o superar la performance de cada modelo publicado después de él."** (Abstract, frase 3).

No hay una contribución arquitectónica en RoBERTa. El modelo es **literalmente BERT-large** (24 capas, $H=1024$, 16 cabezas, 355M parámetros) o **BERT-base** (12 capas, $H=768$, 12 cabezas, 125M parámetros). No se cambia ni la self-attention, ni la FFN, ni el LayerNorm post-norm, ni el dropout, ni la activación GELU, ni el optimizador Adam. Lo único que cambia es el **régimen de pre-training**: qué datos, cuántos pasos, qué tamaño de batch, qué objetivos, qué tokenización.

Esta tesis, en julio de 2019, era polémica por dos razones:

1. **Contradice la lectura dominante de los benchmarks**. Después de BERT (Oct 2018), salieron en sucesión XLNet (Jun 2019, permutation LM), SpanBERT (Jul 2019, masking de spans contiguos), ERNIE de Baidu (Mar 2019, masking de entidades), MASS (Feb 2019, span-MLM para seq2seq), MT-DNN (Apr 2019, multi-task). Cada uno mostraba **alguna** ganancia sobre BERT en GLUE/SQuAD. La interpretación obvia era "BERT fue una buena arquitectura, pero estos modelos la mejoran". RoBERTa dice: **ninguno mejora BERT realmente; todos están comparando contra una versión sub-entrenada del baseline**.

2. **Es un argumento incómodo para la academia**. Si la receta correcta es "más datos, batch más grande, más pasos, sin NSP, dynamic masking", entonces el progreso depende de **acceso a cómputo**, no de inventiva. Es la primera vez que un paper de NLP de alto impacto dice abiertamente que **escala > novedad**. Esta lectura se confirma con GPT-3 un año después (Brown et al. 2020).

El paper se posiciona como una **replication study** (Sección 1, último párrafo): "we present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size".

---

## 2. Contexto histórico: la carrera post-BERT (Oct 2018 — Jul 2019)

Para entender por qué RoBERTa salió cuando salió, hay que mirar la ventana de **nueve meses** entre BERT y RoBERTa:

| Mes | Modelo | Institución | Idea principal | Sobre BERT-large MNLI (Acc) |
|---|---|---|---|---|
| Oct 2018 | **BERT-large** | Google | MLM + NSP, encoder bidireccional | 86.6 |
| Ene 2019 | **MT-DNN** | Microsoft | Multi-task fine-tuning sobre BERT | 87.1 |
| Mar 2019 | **ERNIE 1.0** | Baidu | Knowledge masking (entidades, frases) | 84.0 (sobre BERT-base) |
| Abr 2019 | **MASS** | Microsoft | Span-based MLM para seq2seq | n/a (foco en NMT) |
| Jun 2019 | **XLNet-large** | Google + CMU | Permutation LM con Transformer-XL | 89.8 |
| Jun 2019 | **SpanBERT** | UW + FAIR | Masking de spans contiguos + SBO loss | 88.1 |
| Jul 2019 | **RoBERTa-large** | FAIR + UW | BERT entrenado con más datos, más pasos, sin NSP | **90.2** |
| Sep 2019 | **ALBERT-xxlarge** | Google | Cross-layer param sharing + SOP | 90.8 |
| Oct 2019 | **T5-11B** | Google | Encoder-decoder unificado + C4 | (otra liga) |

La narrativa de la comunidad antes de RoBERTa era: "XLNet venció a BERT porque permutation LM es mejor que MLM, y porque usa Transformer-XL para context largo". El paper de XLNet (Yang et al. 2019) reporta 89.8 MNLI vs 86.6 de BERT, una mejora aparentemente arquitectónica.

RoBERTa demuele esa lectura. Mostrando que **con la misma arquitectura BERT pero entrenada con la receta correcta** se consigue **90.2 MNLI** — superando al XLNet original. La tabla 5 del paper compara con cuidado:

| Modelo | Data | BookCorpus + Wiki MNLI | + Stories + CC-News + OpenWebText MNLI |
|---|---|---|---|
| BERT-large | 13 GB | 86.6 | — |
| XLNet-large | 13 GB | 88.4 | — |
| XLNet-large | 126 GB | — | 89.8 |
| **RoBERTa** (mismo data BERT) | 13 GB | **89.0** | — |
| **RoBERTa** (data completo) | 160 GB | — | **90.2** |

Es decir: **incluso restringiéndose al corpus original de BERT (13 GB), RoBERTa supera a XLNet entrenado con el mismo corpus por 0.6 puntos**. La ganancia atribuida a "permutation LM" se evapora cuando se controla por régimen de entrenamiento. La ganancia restante sí viene de más datos (160 GB vs 13 GB), pero es ortogonal a la arquitectura.

Este resultado es el que vuelve a RoBERTa el paper definitorio de la era encoder-only tardía: **deja claro que en 2019 la frontera ya no es arquitectónica, sino de datos y compute**.

---

## 3. Las cinco modificaciones al régimen de pre-training de BERT

El núcleo técnico del paper está en la Sección 4 ("Training Procedure Analysis"). Son cinco cambios, cada uno con su ablation. Importa entender que son **acumulativos** — el modelo final los combina todos.

### 3.1 Eliminar Next Sentence Prediction (NSP)

BERT entrena con la suma de dos losses:

$$\mathcal{L}_{\text{BERT}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

donde $\mathcal{L}_{\text{NSP}}$ es la cross-entropy de un clasificador binario sobre el embedding final de `[CLS]` que predice si dos segmentos son contiguos en el corpus.

RoBERTa pregunta: **¿NSP aporta?**. La Tabla 2 del paper compara cuatro formatos de input sobre BERT-base con la misma cantidad de pasos y datos:

| Formato | Descripción | SQuAD 1.1 F1 | SQuAD 2.0 F1 | MNLI-m | SST-2 |
|---|---|---|---|---|---|
| SEGMENT-PAIR + NSP | BERT original. Dos segmentos, NSP loss | 90.4 | 78.7 | 84.0 | 92.9 |
| SENTENCE-PAIR + NSP | Dos oraciones (no segmentos), NSP loss | 88.7 | 76.2 | 82.9 | 92.1 |
| FULL-SENTENCES (no NSP) | Llenar 512 con oraciones de uno o más documentos. Sin NSP | 90.4 | 79.1 | 84.7 | 92.5 |
| **DOC-SENTENCES (no NSP)** | Como FULL pero dentro de un único documento | **90.6** | **79.7** | **84.7** | **92.7** |

Lecturas:

- **SENTENCE-PAIR** (con oraciones individuales en vez de segmentos largos) **degrada** la performance. La hipótesis es que el modelo necesita ver dependencias largas, no solo pares de oraciones.
- **FULL-SENTENCES sin NSP** **iguala o supera** a SEGMENT-PAIR con NSP. NSP no aporta — o, peor, **daña**, porque obliga a meter pares de documentos no relacionados en la mitad de los samples, contaminando las dependencias largas.
- **DOC-SENTENCES** (no cruzar fronteras de documentos) es marginalmente mejor que FULL-SENTENCES. La diferencia es pequeña (0.2-0.3 F1) pero consistente.

**Decisión final**: RoBERTa adopta **FULL-SENTENCES sin NSP** (no DOC-SENTENCES, a pesar de que es ligeramente mejor) porque DOC-SENTENCES tiene un batch size variable — un documento corto no llena 512 tokens, así que se requiere batching dinámico. La autora prioriza simplicidad y batch fijo.

Esta decisión es la más fuerte conceptualmente. Significa que **la mitad del objetivo de pre-training original de BERT era inútil o contraproducente**. En retrospectiva, ALBERT (Lan et al. 2019) llegó a la misma conclusión y reemplazó NSP por SOP (Sentence Order Prediction), que es más difícil. T5 (Raffel et al. 2020) ignora NSP/SOP completamente. El consenso 2020-2026 es que NSP no es necesario.

### 3.2 Dynamic masking vs static masking

BERT genera el masking **una vez** durante la creación del dataset preprocesado. Para entrenar 40 épocas, BERT replica el dataset 10 veces con distinto masking cada copia — así cada ejemplo se ve enmascarado en 10 patrones distintos, pero cada patrón se ve 4 veces.

RoBERTa propone **dynamic masking**: generar el masking al vuelo en cada forward pass. Así, cada vez que un ejemplo entra al modelo, se enmascara con un patrón nuevo. Para un modelo entrenado 500K pasos × batch 8K = 4B ejemplos, esto significa que cada token enmascarable ve **billones de patrones distintos** en vez de solo 10.

La Tabla 1 del paper compara directamente:

| Masking | SQuAD 2.0 F1 | MNLI-m | SST-2 |
|---|---|---|---|
| Reference (Devlin et al.) | 76.3 | 84.3 | 92.8 |
| Static (RoBERTa reimpl.) | 78.3 | 84.3 | 92.5 |
| **Dynamic** | **78.7** | **84.4** | **92.8** |

La ganancia es modesta (~0.4 F1 en SQuAD, ~0.1 en MNLI). Pero **no cuesta nada** — solo es cuestión de mover el masking al data loader. Por eso se adopta.

Detalle importante: en HuggingFace `transformers`, el `DataCollatorForLanguageModeling` aplica dynamic masking por default. Esto significa que **todos los modelos modernos derivados de RoBERTa heredan este detalle**, aunque rara vez se documenta explícitamente.

### 3.3 Más datos: 16 GB → 160 GB

BERT entrena con dos corpora:

- **BooksCorpus** (Zhu et al. 2015): 800M palabras, 11K libros gratuitos de smashwords.com.
- **English Wikipedia**: 2,500M palabras (texto de pasajes; sin listas, tablas, headers).
- **Total**: ~3.3B palabras ≈ **16 GB** de texto crudo (en bytes UTF-8 con WordPiece tokenization).

RoBERTa **mantiene** BooksCorpus + Wikipedia y **añade**:

| Corpus | Tamaño | Fuente / generación |
|---|---|---|
| BooksCorpus + Wikipedia | 16 GB | El mismo de BERT |
| CC-News | 76 GB | Common Crawl News, filtrado a inglés, Sep 2016 – Feb 2019. ~63M artículos |
| OpenWebText | 38 GB | Recreación open-source del WebText de OpenAI (GPT-2): páginas web linkadas desde Reddit con ≥3 karma |
| Stories | 31 GB | Subset de Common Crawl que coincide en estilo con Winograd schemas (Trinh & Le 2018) |
| **Total** | **160 GB** | ~10× el corpus de BERT |

El impacto se mide en la Tabla 4 del paper:

| Modelo | Data | Batch | Pasos | SQuAD 1.1 F1 | MNLI-m | SST-2 |
|---|---|---|---|---|---|---|
| RoBERTa con BookCorpus+Wiki | 16 GB | 8K | 100K | 93.6 | 89.0 | 95.3 |
| **+ CC-News, OpenWebText, Stories** | **160 GB** | 8K | 100K | **94.0** | **89.3** | **95.6** |
| **+ entrenar más pasos** | 160 GB | 8K | 300K | 94.4 | 90.0 | 96.1 |
| **+ entrenar más pasos** | 160 GB | 8K | 500K | **94.6** | **90.2** | **96.4** |

Cada incremento (más data, más pasos) aporta de forma monotónica. Las ganancias son sub-lineales: ir de 100K a 500K pasos (5×) da ~1 punto MNLI; ir de 16 GB a 160 GB (10×) da ~0.3 puntos. Pero combinadas suman 1.2 puntos sobre la línea base, que es lo que separaba a BERT de XLNet.

**Observación crítica**: la Tabla 4 implícitamente muestra que **incluso con el mismo corpus de BERT, RoBERTa-large con dynamic masking, sin NSP, batch 8K y 500K pasos llega a 89.0 MNLI** — vs 86.6 de BERT-large original. Eso significa que **~2.4 puntos** de la mejora vienen **solo de la receta de entrenamiento**, sin necesidad de datos adicionales. Los datos adicionales aportan otros **~1.2 puntos** encima.

### 3.4 Batch size masivo: 256 → 8K

BERT entrena con batch de 256 secuencias × 512 tokens = 131K tokens/batch. RoBERTa explora **batches mucho más grandes**, motivado por trabajos contemporáneos sobre large-batch training (Goyal et al. 2017 para ImageNet, Smith et al. 2018, "Don't Decay the Learning Rate, Increase the Batch Size", You et al. 2019 con LARS/LAMB).

La Tabla 3 del paper compara:

| Batch size | Pasos | LR | PPL en held-out | MNLI-m | SST-2 |
|---|---|---|---|---|---|
| 256 (BERT) | 1M | 1e-4 | 3.99 | 84.7 | 92.7 |
| **2K** | 125K | 7e-4 | 3.68 | 85.2 | 92.9 |
| **8K** | 31K | 1e-3 | 3.77 | **84.6** | 92.8 |

Notas:

- **El número total de tokens procesados es similar** en las tres filas (256 × 1M ≈ 2K × 125K ≈ 8K × 31K). El experimento controla por compute.
- **Batch 2K es el mejor punto** en este experimento controlado por compute (PPL 3.68). Batch 8K degrada ligeramente.
- Sin embargo, el modelo final de RoBERTa usa **batch 8K** porque permite **más pasos efectivos** con el mismo wall-clock — al saturar mejor las 1024 V100, el throughput aumenta significativamente. Es decir: batch 8K es peor por paso pero mejor por hora.

El **linear scaling rule** se aplica: $\text{LR} \propto \text{batch size}$. BERT con batch 256 usa LR 1e-4; RoBERTa con batch 8K usa LR ≈ 4e-4 a 1e-3 (depende del experimento). El warmup también escala — más pasos de warmup (30K en vez de 10K).

El **costo de cómputo** del batch 8K es brutal:

- BERT-large: 16 Cloud TPUs (64 chips TPUv2) × 4 días = ~$7K USD en 2018.
- RoBERTa-large: **1024 V100 GPUs** × 1 día = ~$60K - $100K USD en 2019 (estimado, no en el paper).

El paper no reporta el costo monetario, pero la footnote 11 dice "we follow Liu et al. (2019b)" refiriéndose a 1024 V100s. Es uno de los primeros modelos de NLP en usar batch de 5 órdenes de magnitud superior a lo habitual. Marca el inicio de la era en la que el pre-training de NLP **deja de ser reproducible para academia** y se vuelve dominio exclusivo de labs con cluster propio.

### 3.5 Más pasos de entrenamiento (500K vs 1M de BERT)

BERT entrena 1M pasos con batch 256 = 256M secuencias × 512 tokens = **131B tokens vistos**. RoBERTa entrena 500K pasos con batch 8K = 4B secuencias × 512 tokens = **2T tokens vistos** — aproximadamente **16× más tokens** que BERT.

En términos de épocas sobre el corpus:

- BERT: 131B / 3.3B palabras ≈ **40 épocas** sobre BooksCorpus+Wiki.
- RoBERTa: 2T / 32B palabras (160 GB) ≈ **62 épocas** sobre el corpus completo.
- RoBERTa con solo BookCorpus+Wiki: 2T / 3.3B ≈ **600 épocas** — extremo, posiblemente sobre-entrenado, pero el paper no observa overfit.

La conclusión empírica de la Tabla 4 (ya citada en 3.3) es que la performance **sigue mejorando monotónicamente** hasta 500K pasos. No hay evidencia de overfit en validation perplexity ni en downstream tasks. Este resultado es importante porque **contradice la intuición pre-2019** de que entrenar muchas épocas degrada. La explicación posterior (Kaplan et al. 2020, "Scaling Laws") es que con modelos suficientemente grandes y datos suficientemente diversos, **el régimen de underfit dura mucho más de lo que se creía**.

---

## 4. Byte-level BPE — el cambio de tokenizador

Esta es la única decisión del paper que tiene **implicaciones prácticas inmediatas para cualquier usuario de RoBERTa** y suele ser una **trampa** al portar código de BERT a RoBERTa.

### 4.1 De WordPiece (30K) a byte-level BPE (50K)

- **BERT** usa **WordPiece** (Wu et al. 2016) con vocabulario de 30,522 tokens. Se construye sobre **caracteres Unicode**. Los caracteres no vistos se mapean a `[UNK]`.
- **RoBERTa** adopta **byte-level Byte-Pair Encoding (BBPE)** de GPT-2 (Radford et al. 2019), con vocabulario de **50,265 tokens**. Se construye sobre **bytes** (0–255), no caracteres.

La diferencia clave: en byte-level BPE, **el alfabeto base son los 256 bytes posibles**, no los caracteres Unicode. Cualquier string UTF-8 puede representarse sin pérdida como una secuencia de bytes. Por lo tanto:

- **Nunca hay `[UNK]`** en RoBERTa. Cualquier carácter Unicode (emoji, símbolo raro, idioma no visto) se descompone en bytes y se tokeniza con BBPE. Esto es una ventaja enorme para robustez multilingüe y código fuente.
- **Vocabulario más grande** (50K vs 30K) sin más OOV — el espacio extra se gasta en sub-secuencias de bytes más largas y más frecuentes, no en cubrir caracteres extra.
- **Continuación de palabra**: WordPiece usa `##` como prefijo (`play ##ing`). BBPE usa el byte `Ġ` (0x120 en encoding visible) como prefijo de inicio de palabra (`Ġplay ing`). El espacio en blanco se trata como parte del siguiente token, no como separador.

### 4.2 Ejemplo concreto

Para el string `"Hello, World!"`:

- **BERT (WordPiece)**: `[CLS]` `hello` `,` `world` `!` `[SEP]` → IDs `[101, 7592, 1010, 2088, 999, 102]`.
- **RoBERTa (BBPE)**: `<s>` `Hello` `,` `ĠWorld` `!` `</s>` → IDs `[0, 31414, 6, 623, 328, 2]`.

Diferencias visibles:

1. Tokens especiales **distintos**: `<s>`/`</s>` en RoBERTa vs `[CLS]`/`[SEP]` en BERT.
2. RoBERTa **preserva el casing** por default (no hace lowercase). BERT-base-uncased lo perdería.
3. La coma `,` y el `!` son tokens separados con IDs propios.
4. El espacio antes de "World" se codifica como parte del token `ĠWorld`.

### 4.3 Por qué importa este cambio

Para el alumno del lab 20, este detalle es **load-bearing**:

```python
from transformers import AutoTokenizer

tok_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
tok_rob  = AutoTokenizer.from_pretrained("roberta-base")

tok_bert("Hello World!", return_tensors="pt")
# {'input_ids': tensor([[ 101, 7592, 2088, 999,  102]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0]]),  # <-- existe
#  'attention_mask': tensor([[1, 1, 1, 1, 1]])}

tok_rob("Hello World!", return_tensors="pt")
# {'input_ids': tensor([[   0, 31414,   623,   328,     2]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1]])}  # <-- sin token_type_ids
```

Dos diferencias críticas que tropiezan al usuario:

1. **IDs completamente distintos**: 101 (`[CLS]`) vs 0 (`<s>`). Si copy-paste un script de BERT y se cambia solo el `model_name`, los `input_ids` hardcoded no funcionarán.
2. **No hay `token_type_ids`** en la salida de RoBERTa. Esto es porque RoBERTa **no usa segment embeddings** — al haber eliminado NSP, los embeddings de segmento $E_A$ / $E_B$ ya no aportan información y se eliminaron del modelo. Si se llama a `model(input_ids=..., token_type_ids=...)` con RoBERTa, los `token_type_ids` se ignoran silenciosamente (o se reciben como `None`). Esto es una **fuente común de bugs** cuando se porta código BERT → RoBERTa.

### 4.4 El costo de BBPE en parámetros

Vocabulario 50K vs 30K significa que la matriz de token embeddings es ~67% más grande:

- BERT-large: $30{,}522 \times 1024 \approx 31$ M params.
- RoBERTa-large: $50{,}265 \times 1024 \approx 51$ M params.

Esto explica por qué RoBERTa-base tiene **125M parámetros** y BERT-base tiene **110M parámetros** — la diferencia de **15M** está casi enteramente en la matriz de embeddings (y su weight-tied output head). No hay parámetros adicionales en las capas Transformer.

---

## 5. Tokens especiales en detalle (trampa al portar código)

Tabla de equivalencias para integrar al lab 20:

| Función | BERT | RoBERTa | ID en BERT-base-uncased | ID en RoBERTa-base |
|---|---|---|---|---|
| Inicio de secuencia / clasificación | `[CLS]` | `<s>` | 101 | 0 |
| Separador / fin de secuencia | `[SEP]` | `</s>` | 102 | 2 |
| Padding | `[PAD]` | `<pad>` | 0 | 1 |
| Token desconocido | `[UNK]` | `<unk>` | 100 | 3 |
| Token enmascarado | `[MASK]` | `<mask>` | 103 | 50264 |

Notar:

- RoBERTa usa **convención fairseq** (`<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>` con corchetes angulares estilo XML). BERT usa **convención TensorFlow original** (`[CLS]`, `[SEP]` con corchetes cuadrados).
- El ID de `<pad>` en RoBERTa es **1**, no 0. El ID 0 está reservado para `<s>`. Esto significa que en `attention_mask`, la regla "0 = padding, 1 = real" se mantiene, pero al construir tensores manualmente con `torch.zeros` para padding, se llena con `1`, no con `0`. Otra trampa.
- `<mask>` está al final del vocabulario (ID 50264), no al principio como en BERT (ID 103). Si se hace análisis del embedding de `[MASK]`/`<mask>` para interpretabilidad, el índice cambia.

---

## 6. Resultados detallados

### 6.1 GLUE dev (Tabla 8 del paper)

Comparación con la misma estructura que la Tabla 1 de BERT. Single-task fine-tuning, sin model ensemble:

| Modelo | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B |
|---|---|---|---|---|---|---|---|---|
| BERT-large (Devlin et al.) | 86.6/- | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0 |
| XLNet-large (Yang et al.) | 89.8/- | 93.9 | 91.8 | 83.8 | 95.6 | 89.2 | 63.6 | 91.8 |
| **RoBERTa-large** | **90.2/90.2** | **94.7** | **92.2** | **86.6** | **96.4** | **90.9** | **68.0** | **92.4** |

Lecturas:

- **+3.6 MNLI** sobre BERT-large. **+0.4 MNLI** sobre XLNet-large.
- **+16.2 RTE** sobre BERT-large. RTE es small-data (2.5K ejemplos), donde RoBERTa muestra que más pre-training compensa data downstream escaso.
- **+7.4 CoLA** sobre BERT-large. CoLA mide aceptabilidad gramatical, una tarea donde 2T tokens de pre-training claramente importan.

### 6.2 GLUE test (ensemble + multi-task, Tabla 9)

RoBERTa también usa fine-tuning multi-task (MT-DNN style) en el set de test final. Resultados en el leaderboard GLUE de Jul 2019:

| Modelo | Avg GLUE Test |
|---|---|
| BERT-large (ensemble) | 80.5 |
| MT-DNN (ensemble) | 87.6 |
| XLNet-large (ensemble) | 88.4 |
| **RoBERTa-large (ensemble)** | **88.5** |

Por una **décima** RoBERTa quedó arriba de XLNet en el leaderboard al momento de publicación. Esa décima fue suficiente — el paper coronó a RoBERTa como SOTA.

### 6.3 SQuAD (Tabla 10)

| Modelo | SQuAD 1.1 dev EM/F1 | SQuAD 2.0 dev EM/F1 |
|---|---|---|
| BERT-large | 84.1 / 90.9 | 79.0 / 81.8 |
| XLNet-large | 89.0 / 94.5 | 86.1 / 88.8 |
| **RoBERTa-large** | **88.9 / 94.6** | **86.5 / 89.4** |

En SQuAD 1.1, RoBERTa empata a XLNet (94.6 vs 94.5 F1). En SQuAD 2.0, RoBERTa supera por 0.6 F1.

### 6.4 RACE (Tabla 11)

RACE (Lai et al. 2017) es un dataset de comprensión lectora estilo examen TOEFL — multiple choice de 4 opciones sobre pasajes largos (~300 palabras).

| Modelo | Middle (M) | High (H) | All |
|---|---|---|---|
| BERT-large | 76.6 | 70.1 | 72.0 |
| XLNet-large | 85.4 | 80.2 | 81.7 |
| **RoBERTa-large** | **86.5** | **81.3** | **83.2** |

+11.2 puntos sobre BERT-large, +1.5 sobre XLNet-large. RACE requiere razonamiento multi-hop sobre pasajes largos — la mejora se atribuye al pre-training más extenso en datos diversos (CC-News y OpenWebText son ricos en estilo journalistic).

### 6.5 El resultado más vergonzoso para BERT: RoBERTa-base supera a BERT-large

Tabla menos enfatizada pero crítica. **RoBERTa-base (125M params, 12 capas)** vs **BERT-large (340M params, 24 capas)** en MNLI:

| Modelo | Params | MNLI-m |
|---|---|---|
| BERT-base (Devlin) | 110M | 84.6 |
| BERT-large (Devlin) | 340M | 86.6 |
| **RoBERTa-base** | 125M | **87.6** |
| RoBERTa-large | 355M | 90.2 |

**RoBERTa-base con 125M parámetros supera a BERT-large con 340M parámetros por +1.0 MNLI**. Es decir: un modelo **tres veces más pequeño** entrenado con la receta correcta supera al modelo grande con la receta original. Este es el resultado más concluyente del paper, y el que justifica empíricamente la tesis "BERT fue undertrained".

Implicación práctica: **para la mayoría de aplicaciones, RoBERTa-base es la elección por default sobre BERT-large** — más rápido en inferencia, más barato de fine-tunear, mejor performance.

---

## 7. Ablations adicionales que el paper deja pasar

El paper es excelente en ablations del régimen de entrenamiento pero **deja varias preguntas abiertas** que la literatura posterior atacó:

### 7.1 ¿Cuánto de la ganancia viene de cada cambio?

El paper presenta los cambios como acumulativos pero **no muestra el ablation cruzado completo** ($2^5 = 32$ combinaciones). Solo muestra trayectorias razonables. La interpretación dominante (apoyada en trabajo posterior) es:

| Cambio | Contribución estimada a MNLI |
|---|---|
| Eliminar NSP + FULL-SENTENCES | +0.4 a +0.7 |
| Dynamic masking | +0.1 |
| Batch 8K + LR alto | +0.3 a +0.5 |
| Más pasos (500K) | +0.5 a +1.0 |
| Más datos (16 → 160 GB) | +0.3 a +0.5 |
| **Total** | **+1.5 a +2.8** sobre BERT-large |

Es decir: **más pasos** y **eliminar NSP** son los cambios individuales más impactantes. **Dynamic masking** es casi cosmético.

### 7.2 ¿Qué tan crítico es cada corpus adicional?

El paper combina CC-News + OpenWebText + Stories sin ablation individual. Trabajo posterior (XLM-R 2020, Conneau et al.) muestra que **OpenWebText** y **CC-News** son los más útiles; **Stories** aporta menos pero no daña.

### 7.3 ¿Por qué BBPE y no SentencePiece o WordPiece más grande?

El paper menciona en una sola línea que adopta BBPE "por consistencia con GPT-2". No hay ablation BBPE vs WordPiece. La pregunta de tokenización fue atacada después: Kudo & Richardson (SentencePiece, 2018), Bostrom & Durrett (2020, "BPE vs WordPiece") muestran que en NLP general, BBPE y SentencePiece-BPE son equivalentes; lo importante es el tamaño del vocabulario y la inclusión de todos los caracteres del corpus de pre-training. En multilingüe, SentencePiece tiende a ganar.

### 7.4 ¿Por qué no batch 16K o 32K?

You et al. (2019, "Large Batch Training of Convolutional Networks") muestra que con LARS/LAMB se puede entrenar BERT con batch 32K. SmithDavis et al. (2019, "Don't Decay the Learning Rate") sugiere batches todavía más grandes. RoBERTa se queda en 8K. La razón implícita: a partir de 8K, el speedup wall-clock se satura (las GPUs ya están saturadas) y el aporte de calidad disminuye.

---

## 8. Limitaciones críticas del paper

Hay que ser honesto sobre los límites del trabajo:

### 8.1 No es una contribución arquitectónica

El modelo es BERT. No hay invención técnica. El paper es un **estudio de hiperparámetros y régimen de datos**, no un modelo nuevo. ICLR 2020 lo rechazó precisamente por esto — "not novel enough" — y la decisión fue defendible bajo criterios académicos tradicionales. Pero el paper se volvió un clásico por la **claridad del mensaje empírico**: la diferencia entre BERT y XLNet no era arquitectónica; era de receta.

### 8.2 Costo de cómputo prohibitivo

1024 V100 GPUs durante ~1 día no es reproducible en academia. El paper liberó los **checkpoints** pero **no el pipeline reproducible de pre-training** — el script de fairseq existe pero ejecutarlo end-to-end requiere infraestructura que pocos labs tienen. Esto inicia la **era del modelo cerrado de facto**: la arquitectura es pública, el código es público, pero la receta de cómputo está fuera del alcance de la mayoría. La situación se agrava con T5-11B, GPT-3, Llama, etc.

### 8.3 Solo inglés

Todos los experimentos son en inglés. La extensión multilingüe vino con **XLM-R** (Conneau et al. 2020), también de FAIR, que aplica la receta de RoBERTa a 100 idiomas con CommonCrawl filtrado (2.5 TB). Es el descendiente directo de RoBERTa para multilingüe.

### 8.4 Sin generación

RoBERTa hereda la limitación arquitectónica de BERT: es encoder-only, no genera texto. En 2019, esto era aceptable. En 2026, con LLMs dominando todas las tareas, RoBERTa está confinado a:

- **Embeddings densos para retrieval** (Sentence-RoBERTa, E5, BGE basan su backbone en RoBERTa).
- **Cross-encoders de re-ranking** en pipelines RAG.
- **Fine-tuning supervisado para clasificación** cuando latencia/costo importan.

### 8.5 Sin teoría

El paper es 100% empírico. No hay teoría que explique **por qué** más pasos ayudan más que más data, o **por qué** NSP daña. Trabajo posterior (Kaplan 2020, Hoffmann 2022 "Chinchilla") intenta dar marco teórico vía scaling laws.

### 8.6 "Scaling > inventiva" como lema

La interpretación crítica más fuerte del paper es **filosófica**: si la receta correcta es "más datos, más cómputo, batch más grande", entonces el progreso de NLP se vuelve **dependiente de capital**. Los autores no abordan esto explícitamente. Pero la era post-RoBERTa (GPT-3, ChatGPT, Llama, Claude) confirma la lectura: la frontera del NLP moderno no se mueve por inventiva arquitectónica sino por inversión de cómputo.

---

## 9. Comparación numérica fina con XLNet (Sección 5 del paper)

La Tabla 5 del paper merece análisis dedicado porque es **la tabla más demoledora** del trabajo. Compara los dos modelos controlando por data y compute:

| Modelo | Data | Pasos | Batch | MNLI-m | SQuAD 2.0 F1 | RACE |
|---|---|---|---|---|---|---|
| XLNet-large (paper original) | 13 GB | 500K | 256 | 88.4 | 87.7 | 81.7 |
| **RoBERTa** (mismo data y compute) | 13 GB | 500K | 8K | **89.0** | **88.7** | n/a |
| XLNet-large (extendido) | 126 GB | n/a | n/a | 89.8 | 88.8 | 81.7 |
| **RoBERTa** (full data) | 160 GB | 500K | 8K | **90.2** | **89.4** | **83.2** |

Observaciones:

- **A igualdad de data (13 GB)**, RoBERTa supera a XLNet por **+0.6 MNLI** y **+1.0 SQuAD 2.0**. Esto **invalida la narrativa de que permutation LM es superior a MLM**. Lo que separaba a XLNet de BERT no era permutation LM; era la mejor receta de entrenamiento (XLNet también entrenó con más pasos y batch más grande que BERT original).
- **A igualdad de data ampliada**, RoBERTa con 160 GB supera a XLNet con 126 GB. El delta de data (+34 GB) podría explicar parte de la mejora, pero las ganancias en RACE (+1.5) y SQuAD 2.0 (+0.6) sugieren también un efecto compuesto de las cinco modificaciones.

La conclusión del paper (Sección 5, última frase): "*despite XLNet's algorithmic improvements over BERT, training on more data with larger batches for longer is sufficient to surpass XLNet*". Es una frase **dura** porque dice, sin rodeos, que la novedad arquitectónica de XLNet **no aporta nada medible** cuando se controla por régimen de entrenamiento.

XLNet respondió en una v2 del paper (Sep 2019) con un re-experimento que reduce el gap, pero el daño narrativo ya estaba hecho. ALBERT (Sep 2019) y T5 (Oct 2019) ya no intentan re-introducir permutation LM.

---

## 10. Conexión con la clase 20 del Diplomado IA UC

La clase 20 traza la trayectoria **ELMo → BERT → GPT-1/2 → RoBERTa → ChatGPT/InstructGPT**. RoBERTa ocupa el rol de **cierre de la era encoder-only "puramente NLU"** antes de que GPT-2 (Feb 2019) y GPT-3 (Jun 2020) reabran el paradigma decoder-only generativo:

- **Clase 14 (Transformers)**: arquitectura base que RoBERTa hereda sin cambios.
- **Clase 18 (GPT-1)**: contraste — unidireccional vs bidireccional, fine-tuning unificado en ambos. GPT-1 sale en Jun 2018, BERT en Oct 2018, RoBERTa en Jul 2019. GPT-1 + bidireccionalidad = BERT; BERT + receta correcta = RoBERTa.
- **Clase 19 (ELMo)**: bidireccionalidad shallow (concat de dos LMs) vs bidireccionalidad profunda (BERT/RoBERTa). RoBERTa muestra cuán lejos puede llegar la bidireccionalidad profunda con scale suficiente.
- **Clase 20 (RoBERTa + GPT-2 + ChatGPT)**: este paper es el **clímax y cierre** de la era encoder-only. Después de RoBERTa, los modelos de pre-training más relevantes son **decoder-only generativos** (GPT-2, GPT-3, T5 encoder-decoder, Llama). RoBERTa fija el techo de lo que se puede lograr con encoder-only puro a 355M parámetros.

El `fundamentos/embeddings-contextualizados.md` del site cubre la diferencia entre embeddings estáticos (word2vec, GloVe) y contextualizados (ELMo, BERT, RoBERTa). El `fundamentos/bert.md` cubre la arquitectura encoder. El `papers/bert-devlin-2018.md` resume BERT. **Este documento aporta lo que esos archivos dejan implícito sobre RoBERTa**:

- La tesis empírica "BERT fue undertrained" con números concretos.
- Las cinco modificaciones del régimen y sus ablations.
- Por qué BBPE 50K importa para el código del alumno.
- Por qué `token_type_ids` no existe en RoBERTa (trampa al portar BERT → RoBERTa).
- Por qué RoBERTa-base supera a BERT-large (resultado contraintuitivo).
- El cierre de la era encoder-only y la transición a decoder-only generativos.

---

## 11. Conexión con el lab 20 del Diplomado IA UC

El lab 20 (`lab_clase_20.ipynb`, celdas 17-24) incluye una sección de **carga y exploración de RoBERTa con HuggingFace `transformers`**. Los puntos de fricción que este análisis ayuda a resolver:

### 11.1 Carga del modelo (celda ~17)

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")
```

- `AutoTokenizer` carga el tokenizer BBPE con vocabulario 50,265. Es importante que el alumno entienda que **es un tokenizer diferente al de BERT**, no solo el mismo con otro nombre.
- `AutoModel` carga `RobertaModel`, no `BertModel`. Internamente son **casi idénticos** en arquitectura, pero `RobertaModel` no tiene la capa `token_type_embeddings` (o la tiene con tamaño 1, ignorando segmentos).

### 11.2 Tokenización (celda ~18)

```python
out = tokenizer("Hello World!")
print(out)
# {'input_ids': [0, 31414, 623, 328, 2], 'attention_mask': [1, 1, 1, 1, 1]}

tokens = tokenizer.convert_ids_to_tokens(out['input_ids'])
print(tokens)
# ['<s>', 'Hello', 'ĠWorld', '!', '</s>']
```

Puntos para el alumno:

- El ID 0 corresponde a `<s>`, **no a padding** como en BERT (donde `<pad>` es 0). El `<pad>` de RoBERTa es ID 1.
- El token `ĠWorld` con `Ġ` al principio significa "este token comienza con un espacio en blanco". El espacio se codifica como parte del token, no como separador.
- **No hay `token_type_ids` en la salida** del tokenizer (o salen como cero si se piden explícitamente). Esto es el indicador visible de que RoBERTa no usa segment embeddings.

### 11.3 Forward pass (celda ~19)

```python
import torch
inputs = tokenizer("Hello World!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)  # torch.Size([1, 5, 768])
print(outputs.pooler_output.shape)       # torch.Size([1, 768])
```

- `last_hidden_state[:, 0, :]` es el embedding del token `<s>` — análogo a `[CLS]` en BERT, **pero entrenado sin NSP**, así que no es una representación útil de oración sin fine-tuning. Esto es la misma observación del footnote 6 del paper de BERT (`[CLS]` requiere fine-tuning para ser útil) — en RoBERTa el problema es **peor** porque ni siquiera se entrenó con NSP.
- `pooler_output` es `<s>` proyectado por una capa lineal + tanh. Su utilidad sin fine-tuning es limitada.
- Para **embeddings de oraciones útiles sin fine-tuning**, hay que usar mean pooling sobre `last_hidden_state` o, mejor, **Sentence-RoBERTa** / SBERT (Reimers & Gurevych 2019).

### 11.4 Comparación lado a lado con BERT (celda ~20-22)

El lab probablemente compara `bert-base-uncased` y `roberta-base` sobre la misma frase. Esperar:

- **Distintos IDs** (BERT lowercase, RoBERTa preserva casing).
- **Distinto número de tokens** (BBPE 50K tiende a producir secuencias ligeramente más cortas que WordPiece 30K en inglés).
- **Distinto embedding de `[CLS]`/`<s>`** — no son comparables directamente porque el espacio latente es distinto.
- **Distinto comportamiento downstream**: RoBERTa-base supera a BERT-base en GLUE por ~2-3 puntos.

### 11.5 Fine-tuning para clasificación (celda ~23-24)

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
```

La cabeza de clasificación en RoBERTa es **distinta** a la de BERT:

- **BERT**: cabeza es `Linear(768, num_labels)` aplicada sobre `[CLS]` pooled (con tanh).
- **RoBERTa**: cabeza es `Linear(768, 768) → tanh → Dropout → Linear(768, num_labels)` aplicada sobre `<s>` (sin tanh pooling intermedio, aplicado dentro de la cabeza).

Esto significa que la cabeza de RoBERTa tiene **más parámetros** (~590K vs ~1.5K en BERT) y **se inicializa aleatoriamente** al cargar `RobertaForSequenceClassification`. El warning `Some weights of RobertaForSequenceClassification were not initialized` aparecerá — es esperado, no es un bug.

---

## 12. Descendencia y modelos derivados

RoBERTa fue el modelo base de una generación entera de descendientes:

| Modelo | Año | Innovación sobre RoBERTa |
|---|---|---|
| **XLM-R** (Conneau et al., FAIR) | 2020 | RoBERTa multilingüe en 100 idiomas con 2.5 TB CommonCrawl |
| **DistilRoBERTa** (Sanh et al., HF) | 2020 | Distillation a 82M params, 95% performance |
| **DeBERTa / v2 / v3** (Microsoft) | 2020-21 | Disentangled attention sobre la receta RoBERTa |
| **CamemBERT** (Martin et al., Inria) | 2020 | RoBERTa-base francés sobre OSCAR |
| **BERTIN** (BSC) | 2021 | RoBERTa-base español con training perplexity sampling |
| **MarIA** (BSC) | 2022 | RoBERTa-large español sobre 570 GB de la BNE |
| **PlanTL/RoBERTa-bne** | 2022 | RoBERTa entrenado en corpus de la Biblioteca Nacional de España |
| **Sentence-RoBERTa** (Reimers) | 2019 | Adaptación a embeddings densos con pooling y siamese |
| **e5-base / e5-large** (Microsoft) | 2022 | RoBERTa contrastivamente entrenado para retrieval |
| **BGE / GTE / Jina** | 2023-24 | Familias modernas de embeddings, varias basadas en RoBERTa |

En todos estos casos, la **arquitectura es BERT/RoBERTa idéntica**; lo que cambia es el corpus, el objetivo de fine-tuning, o el idioma. Es decir, RoBERTa institucionalizó la idea de que **la receta de pre-training es más importante que el modelo**.

---

## 13. Notas para integrar al site

Cosas que el `papers/roberta-liu-2019.md` del site (si existe) debería cubrir y que este análisis aporta:

1. **Tabla comparativa BERT vs RoBERTa en tokens especiales e IDs**: load-bearing para el lab.
2. **Tabla de las cinco modificaciones con ablations numéricas**: Tabla 1 (masking), Tabla 2 (input format), Tabla 3 (batch), Tabla 4 (data + pasos).
3. **Resultado clave "RoBERTa-base > BERT-large"**: contraintuitivo, vale destacarlo.
4. **Comparación con XLNet (Tabla 5 del paper)**: la tabla más demoledora del paper.
5. **Costo de cómputo (1024 V100s)**: marca el inicio de la era "no reproducible en academia".
6. **Trampa de `token_type_ids` y de los IDs distintos**: relevante para los alumnos que hagan código.
7. **Descendencia (XLM-R, Sentence-RoBERTa, e5, etc.)**: el segundo aire de RoBERTa como backbone de embeddings densos en RAG moderno.
8. **Conexión con la transición a decoder-only**: por qué RoBERTa cierra una era.

El `fundamentos/embeddings-contextualizados.md` ya cubre la diferencia conceptual estática vs contextualizada — podría sumar el dato de que RoBERTa-base es el backbone más usado para embeddings de oraciones en 2020-2023.

---

## 14. Lectura recomendada complementaria

- **BERT** (Devlin et al. 2018) — prerrequisito ineludible. Sin entender BERT, RoBERTa no se entiende.
- **XLNet** (Yang et al. 2019) — el rival inmediato. Leer ambos en paralelo aclara qué es novedad arquitectónica vs receta.
- **ALBERT** (Lan et al. 2019) — alternativa contemporánea que prioriza eficiencia paramétrica (cross-layer sharing) en vez de scale.
- **ELECTRA** (Clark et al. 2020) — ataque conceptual al objetivo MLM mismo, con replaced token detection. ELECTRA-base iguala a RoBERTa-base con 1/4 del compute.
- **DeBERTa-v3** (He et al. 2021) — el sucesor más fuerte; disentangled attention + ELECTRA training sobre la receta RoBERTa.
- **XLM-R** (Conneau et al. 2020) — multilingual descendant directo.
- **Kaplan et al. 2020 — Scaling Laws for Neural Language Models** — el marco teórico que retroactivamente explica los hallazgos empíricos de RoBERTa.
- **Hoffmann et al. 2022 — Training Compute-Optimal Large Language Models (Chinchilla)** — refinamiento de Kaplan; argumenta que la mayoría de los modelos grandes (incluyendo RoBERTa) están **sub-entrenados en datos** vs parámetros. Es decir: la tesis "BERT fue undertrained" también se aplica a RoBERTa, pero con menos margen.
- **A Primer in BERTology** (Rogers et al. 2020) — survey de BERT/RoBERTa, qué aprenden internamente, qué cabezas hacen qué.
- **Sentence-BERT / Sentence-RoBERTa** (Reimers & Gurevych 2019) — adaptación a embeddings densos, base de retrieval moderno y de pipelines RAG.

---

## 15. Sumario crítico

RoBERTa es un paper **honesto y demoledor**. No inventa nada arquitectónicamente, y lo dice abiertamente. Su contribución es **demostrar con rigor empírico** que:

1. NSP no aporta y debe eliminarse.
2. Dynamic masking es un free lunch.
3. Más datos, más pasos y batch más grande pagan en performance.
4. Byte-level BPE es robusto y elimina `[UNK]`.
5. La diferencia entre BERT y XLNet **no es arquitectónica**, es de receta.

El precio que paga es **costo de cómputo prohibitivo** (1024 V100s) y la consolidación del paradigma "scaling > inventiva". Esa lectura definirá los siguientes seis años de NLP (GPT-3 2020, ChatGPT 2022, GPT-4 2023, Llama 2023-2024, Claude 2023-2026).

Para el alumno del curso, RoBERTa es importante por **tres razones**:

1. **Es el modelo encoder-only de referencia en 2026** para clasificación de texto y embeddings densos. Aprender a usarlo en HuggingFace es habilidad práctica directa.
2. **Es el cierre de la era encoder-only puramente NLU**. Después viene GPT-2/3/4 y el desplazamiento al paradigma generativo decoder-only.
3. **Es la primera evidencia clara de que el progreso en NLP es ahora capital-intensive**. No se hace ya en una GPU en un escritorio universitario; se hace en clusters corporativos. Entender esto es clave para no perderse en la narrativa romántica de "modelos novedosos" cuando lo que importa es "modelos bien entrenados".
