# Análisis interno — Devlin et al. (2018) "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"

> Documento complementario al material público del site (`papers/bert-devlin-2018.md`, `fundamentos/bert.md`, `fundamentos/pretraining-bert.md`). Aquí se profundiza en aspectos que esos archivos cubren superficialmente: justificación teórica de MLM, detalles del régimen de masking, comparación numérica fina contra GPT-1, ablations del Apéndice C, decisiones de ingeniería (TPU pods, schedule de longitud, optimizador), limitaciones que la literatura posterior corrigió, y conexión con la clase 20 del Diplomado IA UC.

- **Paper**: Devlin, Chang, Lee, Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805v2 (24 May 2019). NAACL-HLT 2019.
- **Versiones**: v1 (Oct 2018, arXiv) — release inicial. v2 (May 2019, NAACL) — agrega resultados SQuAD 2.0 y ablations extendidas. La v2 es la canónica.
- **Código y checkpoints**: `https://github.com/google-research/bert`. Modelos pre-entrenados (uncased/cased, base/large, multilingual) liberados bajo Apache 2.0.

---

## 1. Contexto histórico: el sprint de transfer learning en NLP de 2018

Para entender el impacto de BERT hay que situarlo en una ventana de **ocho meses** durante 2018 en la que el paradigma de transfer learning de NLP cambió tres veces:

| Mes | Modelo | Idea | Limitación que dejaba abierta |
|---|---|---|---|
| Ene 2018 | **ULMFiT** (Howard & Ruder) | Fine-tuning gradual y discriminativo de un AWD-LSTM pre-entrenado en LM | LSTM, capacidad limitada |
| Feb 2018 | **ELMo** (Peters et al.) | Concatenar embeddings de un BiLSTM forward + backward entrenados independientemente | Bidireccionalidad **superficial**; feature-based |
| Jun 2018 | **GPT-1** (Radford et al.) | Transformer decoder unidireccional + fine-tuning end-to-end | Unidireccional left-to-right |
| Oct 2018 | **BERT** (Devlin et al.) | Transformer encoder + MLM bidireccional **profundo** + fine-tuning | Sin generación; mismatch `[MASK]` |

La pregunta que rondaba a la comunidad después de ELMo y GPT-1 era: **¿se puede tener bidireccionalidad profunda (no superficial como ELMo) y a la vez la simplicidad de fine-tuning end-to-end (como GPT-1)?**

La dificultad técnica era la siguiente. Un Transformer encoder no enmascara causalmente — cada token atiende a todos los demás en todas las capas. Si se intentara entrenarlo con next-token prediction como GPT, la atención bidireccional permitiría a cada token "verse a sí mismo" indirectamente: en una arquitectura multi-capa, el token $t$ en la capa $\ell$ puede atender al token $t+1$ en la capa $\ell-1$, que a su vez atendió al token $t$ original. La predicción colapsa a identidad y el modelo no aprende nada útil (paper, Section 3.1, párrafo "*bidirectional conditioning would allow each word to indirectly see itself*").

ELMo evade el problema entrenando dos LSTMs separados (uno forward, uno backward) y concatenando sus estados finales. Pero esto es bidireccionalidad **shallow**: dentro de cada LSTM, cada token solo ve una dirección. La fusión ocurre únicamente en la capa de output. BERT resuelve el bloqueo con un objetivo distinto: en vez de predecir el siguiente token, predecir tokens **enmascarados** en posiciones aleatorias del input. Como el modelo nunca observa los tokens a predecir, no hay forma de leakage, y se puede usar atención completamente bidireccional sin trampas.

La idea de masking no es original — Taylor (1953) la propuso como **Cloze test** para medir legibilidad, y Fedus et al. (2018, MaskGAN) la usó para generación. La novedad de BERT es aplicarla como objetivo de pre-training masivo y demostrar que escala.

---

## 2. Contribución central (no es el encoder Transformer)

Es importante separar lo que BERT inventa de lo que toma prestado. La arquitectura — bloques Transformer encoder con multi-head self-attention y FFN — viene textualmente de Vaswani et al. (2017). La tokenización WordPiece viene de Wu et al. (2016, Google NMT). La activación GELU viene de Hendrycks & Gimpel (2016). Adam viene de Kingma & Ba (2014). El paradigma pretrain-finetune con LM viene de Dai & Le (2015), ULMFiT (2018) y GPT-1 (2018).

Las contribuciones genuinas son tres:

### 2.1 Masked Language Model (MLM)

Permite bidireccionalidad profunda dentro de **un solo modelo** (no dos LMs concatenados como ELMo). La función de pérdida formal:

$$\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{x \sim \mathcal{D}}\;\mathbb{E}_{M \subset [1,|x|]} \sum_{i \in M} \log P_\theta(x_i \mid x_{\setminus M})$$

donde $M$ es el conjunto de posiciones enmascaradas (15% del input) y $x_{\setminus M}$ es la secuencia con esas posiciones reemplazadas según la regla 80/10/10. Cada predicción usa la representación final $T_i \in \mathbb{R}^H$ del token en la posición $i$, proyectada con la transpuesta de la matriz de token embeddings (weight tying) sobre el vocabulario:

$$P_\theta(x_i \mid x_{\setminus M}) = \text{softmax}(W_{\text{emb}}^\top \cdot \text{gelu}(W_{\text{proj}} T_i + b))$$

Nótese que el loss se calcula solo sobre las 15% de posiciones enmascaradas — no sobre el resto. Esto contrasta con un LM autorregresivo que predice **cada** token. Como observa el Apéndice C.1, MLM converge marginalmente más lento que LTR por esta razón, pero las ganancias de bidireccionalidad superan ampliamente ese costo.

### 2.2 Next Sentence Prediction (NSP)

Objetivo secundario para forzar al modelo a aprender relaciones entre oraciones (relevante para QA, NLI, paraphrasing). Dada una secuencia con dos segmentos $A$ y $B$ separados por `[SEP]`, se predice una etiqueta binaria desde la representación final $C$ del token `[CLS]`:

$$P_\theta(\text{IsNext} \mid x) = \sigma(w_{\text{NSP}}^\top C + b_{\text{NSP}})$$

Durante la generación de datos, el 50% de los pares $(A, B)$ son contiguos en el corpus (`IsNext`) y el 50% son $B$ aleatorio sacado del corpus (`NotNext`). La precisión final del modelo en NSP llega a 97-98% (footnote 5 del paper).

El loss total de pre-training es la suma de ambos (peso 1 a 1, sin tuning):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

NSP terminó siendo una contribución **débil** — RoBERTa (Liu et al. 2019) lo eliminó sin pérdida de performance, y ALBERT (Lan et al. 2019) lo reemplazó por Sentence Order Prediction (SOP) que es más difícil. Pero en el momento del paper, NSP daba 1-2 puntos en QNLI/MNLI/SQuAD según la Tabla 5.

### 2.3 Fine-tuning unificado

Antes de BERT, cada tarea downstream tenía una arquitectura específica encima del backbone pre-entrenado: BiDAF para QA, CRFs para NER, attention-over-attention para NLI. BERT demuestra que una sola arquitectura — encoder + una capa lineal de output — alcanza SOTA en 11 tareas heterogéneas. El paradigma "freezear lo complicado, fine-tunear lo simple" se invierte: **fine-tunear todo, simplificar la cabeza al máximo**.

---

## 3. Arquitectura: detalles que el site no cubre

Los dos tamaños canónicos:

| | $L$ | $H$ | $A$ | $d_{ff}$ | Params | Heads $\times d_k$ |
|---|---|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 3072 | 110M | $12 \times 64 = 768$ |
| BERT-large | 24 | 1024 | 16 | 4096 | 340M | $16 \times 64 = 1024$ |

Convenciones de la arquitectura (varias **no** aparecen en el cuerpo del paper, solo en el Apéndice A.2 y en el código liberado):

- **FFN ratio**: $d_{ff} = 4H$ (Footnote 3 del paper). Esta razón 4:1 hidden-a-FFN se vuelve estándar en toda la literatura Transformer posterior.
- **Atención**: $d_k = d_v = H/A$. En base, $d_k = 64$. En large, $d_k = 64$ también — large escala añadiendo más heads, no haciendo cada head más ancha.
- **Activation**: GELU (Hendrycks & Gimpel 2016), no ReLU. La razón explícita (Apéndice A.2) es seguir a GPT-1.
- **LayerNorm**: **post-LN**, residual original de Vaswani 2017. Cada subcapa computa $\text{LayerNorm}(x + \text{Sublayer}(x))$. Este detalle es relevante: post-LN dificulta entrenar Transformers profundos sin warmup cuidadoso (Xiong et al. 2020 mostraría que pre-LN es más estable). BERT-large necesita los 10K pasos de warmup precisamente porque es post-LN profundo.
- **Position embeddings**: aprendidos, no sinusoidales. Tabla de tamaño $512 \times H$. Esto significa que BERT no extrapola a secuencias más largas que 512 — un techo rígido que la familia (RoBERTa, DistilBERT, etc.) heredó y solo modelos posteriores (Longformer, BigBird, ModernBERT) atacaron.
- **Weight tying**: el output de la cabeza MLM comparte pesos con la matriz de input token embeddings. Ahorra ~23M parámetros en base.
- **Dropout**: 0.1 en todas las capas (attention + residual + embeddings).

Una decisión sutil: **el head del MLM tiene una capa intermedia adicional**. No proyecta directamente $T_i \to V$ con $W_{\text{emb}}^\top$, sino que aplica primero una capa lineal con GELU y LayerNorm, y luego proyecta al vocabulario. Esta "transform" head se reusa rara vez en la literatura posterior, pero está en el código original.

---

## 4. Input representation: WordPiece, segmentos y posiciones

### 4.1 WordPiece y relación con BPE

WordPiece (Wu et al. 2016, originalmente Schuster & Nakajima 2012 para japonés/coreano) es un algoritmo de subword tokenization muy parecido a BPE pero con un criterio distinto de merge:

- **BPE** (Sennrich et al. 2016): en cada iteración, fusiona el par de símbolos más frecuente.
- **WordPiece**: en cada iteración, fusiona el par que maximiza la **likelihood** de un unigram LM sobre el corpus. Equivalente a fusionar el par $(a, b)$ con mayor $\frac{P(ab)}{P(a) P(b)}$, es decir, pointwise mutual information.

En la práctica los vocabularios resultantes son muy similares. La diferencia visible es la convención de marcado:

- **BPE** (GPT, RoBERTa): marca el **inicio** de palabra con un espacio especial (`Ġ`).
- **WordPiece** (BERT): marca la **continuación** de palabra con `##`.

Ejemplo del paper (Figura 2): `playing` → `play ##ing`. El prefijo `##` indica "esto pega al token anterior sin espacio". Vocabulario: 30,000 tokens (uncased) o 28,996 (cased).

### 4.2 Tres embeddings sumados

$$E_i = E^{\text{tok}}(t_i) + E^{\text{seg}}(s_i) + E^{\text{pos}}(i) \in \mathbb{R}^H$$

| Embedding | Vocab size | Aprendido | Comentario |
|---|---|---|---|
| Token | 30,522 | Sí | Compartido con la cabeza MLM (weight tying) |
| Segment | 2 | Sí | $E_A$ y $E_B$. Para single-sentence, todo es $E_A$. |
| Position | 512 | Sí | No sinusoidal (a diferencia de Vaswani 2017) |

Se suman, no se concatenan. Esta suma es matemáticamente equivalente a una concatenación seguida de proyección lineal (con matrices de la forma adecuada), pero más eficiente en memoria. El input final pasa por un LayerNorm y dropout antes de entrar a la primera capa.

### 4.3 Tokens especiales

- `[CLS]`: posición 0 de toda secuencia. Su representación final $C \in \mathbb{R}^H$ se usa para clasificación. **Importante** (footnote 6 del paper): $C$ no es una representación útil de oración sin fine-tuning, porque fue entrenada con el objetivo NSP, que es muy específico. Para usar BERT como encoder de oraciones (Sentence-BERT, 2019), hay que fine-tunear con un objetivo de similitud.
- `[SEP]`: separador entre segmentos. También se coloca al final de la secuencia completa.
- `[MASK]`: usado solo durante MLM. Nunca aparece en fine-tuning ni en inferencia downstream.
- `[PAD]`, `[UNK]`: utilitarios estándar.

---

## 5. Pre-training objectives en profundidad

### 5.1 La regla 80/10/10 — justificación detallada

Para los tokens seleccionados como objetivo de MLM (15% del input):

| Probabilidad | Acción | Ejemplo (`hairy`) |
|---|---|---|
| 80% | Reemplazar por `[MASK]` | `my dog is [MASK]` |
| 10% | Reemplazar por token aleatorio | `my dog is apple` |
| 10% | Mantener sin cambios | `my dog is hairy` |

La justificación (Sección 3.1 y Apéndice A.1) es doble:

1. **Mitigar el mismatch pretrain/finetune**. Durante fine-tuning, `[MASK]` no aparece nunca. Si el modelo se entrenara siempre con `[MASK]` en las posiciones target, aprendería a depender de ese símbolo. Al introducir variabilidad (a veces el target es un token real, a veces uno aleatorio), el modelo no puede asumir que la presencia de `[MASK]` indica "este token va a ser predicho".

2. **Forzar representaciones contextuales para todos los tokens**. Como el Transformer encoder no sabe a priori qué posiciones serán predichas, debe mantener una representación contextual distribucional de **cada** token de input. Si solo se usara `[MASK]`, el modelo podría desarrollar shortcuts que ignoran los tokens no enmascarados.

El Apéndice C.2 incluye una **ablation del régimen de masking** sobre BERT-base, fine-tuneando en MNLI y NER (Tabla 8):

| MASK | SAME | RND | MNLI fine-tune | NER fine-tune | NER feature-based |
|---|---|---|---|---|---|
| 80% | 10% | 10% | **84.2** | 95.4 | **94.9** |
| 100% | 0% | 0% | 84.3 | 94.9 | 94.0 |
| 80% | 0% | 20% | 84.1 | 95.2 | 94.6 |
| 80% | 20% | 0% | 84.4 | 95.2 | 94.7 |
| 0% | 20% | 80% | 83.7 | 94.8 | 94.6 |
| 0% | 0% | 100% | 83.6 | 94.9 | 94.6 |

Hallazgos clave:

- **Fine-tuning es robusto** a casi cualquier régimen razonable. La diferencia entre 80/10/10 y 100% MASK en MNLI es de apenas 0.1 puntos.
- **Feature-based no es robusto**. Cuando se usa BERT como extractor de features (sin fine-tunear), 100% MASK degrada NER de 94.9 a 94.0 — porque el mismatch entre pretraining (lleno de `[MASK]`) y extracción (sin `[MASK]`) ya no se puede corregir.
- Usar **solo RND** es lo peor para fine-tuning (83.6 en MNLI).

La conclusión es que la regla 80/10/10 es una elección razonable pero no mágica. El choice real es "incluir un poco de cada variante" más que las proporciones exactas.

### 5.2 Static vs dynamic masking

Detalle del Apéndice A.2 que el paper menciona en una sola línea pero que fue load-bearing en la literatura posterior: el masking **se aplica en la generación del dataset**, una vez. Es decir, BERT usa **static masking** — un token particular en una posición particular tiene la misma decisión 80/10/10 en cada época.

RoBERTa (Liu et al. 2019) atacó esto con **dynamic masking**: el masking se aplica al vuelo en cada batch, así el mismo input ve patrones de masking distintos en distintas épocas. Esto da ~0.5 puntos en GLUE. Es un detalle de ingeniería que BERT dejó pasar.

### 5.3 NSP — datos, label y crítica

Para generar pares $(A, B)$:

1. Samplear dos spans contiguos de un documento. Llamarlos $A$ y $B$.
2. Con probabilidad 0.5, reemplazar $B$ por un span aleatorio de otro documento → label `NotNext`.
3. Con probabilidad 0.5, mantener $B$ → label `IsNext`.
4. La suma de longitudes $|A| + |B| \le 512$.

Importante: los "spans" no son oraciones gramaticales. El paper aclara (Sección 3.1) que pueden ser arbitrariamente largos. "Sentence" en el paper significa "span contiguo de texto", no "oración linguística".

**Crítica posterior**: el objetivo NSP es demasiado fácil. La negative example viene de un documento aleatorio, así que muchas señales superficiales (vocabulario del dominio, estilo, longitud) bastan para discriminar — sin requerir comprensión coherente entre $A$ y $B$. RoBERTa eliminó NSP. ALBERT lo reemplazó por SOP (Sentence Order Prediction): dado un par $(A, B)$ siempre contiguo, predecir si está en el orden correcto o invertido. SOP es más difícil porque ambos textos vienen del mismo documento y no se puede aprovechar el shift de dominio.

### 5.4 Datos de pre-training

| Corpus | Palabras | Notas |
|---|---|---|
| BooksCorpus (Zhu et al. 2015) | 800M | 11K libros gratis de smashwords.com |
| English Wikipedia | 2,500M | Solo texto pasajes, **sin** listas, tablas, headers |
| **Total** | 3.3B palabras | |

Detalle crucial (Sección 3.1): se usa corpus **document-level**, no shuffled sentence-level como el Billion Word Benchmark de Chelba et al. 2013. Para que NSP funcione y MLM aprenda dependencias largas, se necesita texto contiguo.

Para la versión multilingüe (mBERT), el corpus es Wikipedia de 104 idiomas con upsampling de idiomas de bajo recurso (exponente 0.7). Esto no está en el paper original, sino en el README del repo.

---

## 6. Hyperparams y cómputo de pre-training (Apéndice A.2)

| Parámetro | Valor |
|---|---|
| Optimizer | Adam, $\beta_1 = 0.9$, $\beta_2 = 0.999$ |
| Learning rate | $1 \times 10^{-4}$ |
| LR schedule | Warmup lineal 10K pasos, luego decay lineal |
| Weight decay (L2) | 0.01 |
| Dropout | 0.1 (en todas las capas) |
| Batch size | 256 secuencias |
| Sequence length | 128 (90% de pasos), 512 (10% final) |
| Pasos totales | 1,000,000 |
| Activación | GELU |

El **schedule mixto de longitud** (128 los primeros 900K pasos, 512 los últimos 100K) es una decisión de eficiencia. Como la atención es $O(n^2)$, entrenar con longitud 512 todo el tiempo es 16× más caro. Pero los position embeddings de las posiciones 128-512 necesitan ver datos. La solución pragmática: la mayoría del pre-training es en 128 (rápido, aprende casi todo), y al final 10% en 512 para que las posiciones largas converjan.

256 secuencias × 512 tokens × 1M pasos = **128B tokens procesados**, aproximadamente 40 épocas sobre el corpus de 3.3B palabras.

### Cómputo

| Modelo | Hardware | Tiempo |
|---|---|---|
| BERT-base | 4 Cloud TPUs (16 chips TPUv2) | 4 días |
| BERT-large | 16 Cloud TPUs (64 chips TPUv2) | 4 días |

En 2018, una Cloud TPU costaba aproximadamente $4.50/hora. BERT-large ≈ 64 chips × 96 horas × $1.125/chip ≈ **$7,000 USD** en TPUs reservadas, o ~$60K con preemptible pricing inflado de la nota al pie 13. Hoy (2026) BERT-base se reproduce en horas en una sola GPU H100; BERT-large en un día en un nodo de 8× H100.

---

## 7. Fine-tuning por tarea: detalle de cabezas

El paper repite que las cabezas son "una sola capa lineal", pero los detalles varían:

### 7.1 Sentence-pair classification (MNLI, QQP, QNLI, STS-B, MRPC, RTE)

- Input: `[CLS] sent_A [SEP] sent_B [SEP]`, segmentos $A$/$B$.
- Cabeza: $W \in \mathbb{R}^{K \times H}$ que proyecta $C \to \mathbb{R}^K$.
- Loss: cross-entropy (o regresión para STS-B).
- Parámetros nuevos: $K \times H$. Para MNLI ($K=3$, $H=1024$): 3,072 parámetros.

### 7.2 Single-sentence classification (SST-2, CoLA)

- Input: `[CLS] sent [SEP]`, todo segmento $A$.
- Cabeza idéntica a 7.1.

### 7.3 Question Answering — span extraction (SQuAD v1.1)

- Input: `[CLS] question [SEP] paragraph [SEP]`. Pregunta segmento $A$, párrafo segmento $B$.
- Dos vectores aprendidos $S, E \in \mathbb{R}^H$.
- Para cada token $T_i$ del párrafo, score de inicio: $P_i = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}$. Análogo para end.
- Span score: $S \cdot T_i + E \cdot T_j$, predecir el máximo con $j \ge i$.
- Loss: $-\log P_{\text{start}} - \log P_{\text{end}}$.
- Parámetros nuevos: $2H = 2048$.

### 7.4 SQuAD v2.0 (con preguntas sin respuesta)

Extensión: predecir $[CLS]$ como "no answer". Comparar el span score del mejor par $(i, j)$ con el score de no-answer $s_{\text{null}} = S \cdot C + E \cdot C$. Se predice no-answer cuando $s_{\text{null}} > s_{i,j} + \tau$, con $\tau$ optimizado en dev.

### 7.5 Token tagging (CoNLL-2003 NER)

- Cabeza por token: $W \in \mathbb{R}^{|\text{labels}| \times H}$ sobre cada $T_i$.
- Para subwords, se usa el primer subtoken como input al clasificador (el resto se ignora en el loss).
- **Sin CRF**. El paper explícitamente formula NER como tagging puro, sin transition scores. Esto es notable porque CRFs eran estándar para NER pre-BERT.

### 7.6 Multiple choice (SWAG)

Cuatro inputs paralelos, cada uno `[CLS] context [SEP] choice_i [SEP]`. Score por choice = $w \cdot C_i$ con $w \in \mathbb{R}^H$ aprendido. Softmax sobre los 4 scores.

### 7.7 Hyperparams típicos de fine-tuning

| Param | Rango |
|---|---|
| Batch size | 16, 32 |
| Learning rate | 5e-5, 3e-5, 2e-5 |
| Epochs | 2, 3, 4 |
| Dropout | 0.1 (sin cambio) |

Búsqueda exhaustiva del grid completo por tarea sobre dev set. Para datasets pequeños (<10K ejemplos), BERT-large era a veces inestable — Devlin et al. recurrían a **random restarts** (varios seeds, picking the best on dev). Esto fue identificado por Mosbach et al. 2021 como un bug de fine-tuning de BERT (instabilidad por warmup insuficiente y few-shot).

Fine-tuning total cuesta ~1 hora en una Cloud TPU para cualquier tarea de GLUE. Es **3000× más barato** que el pre-training.

---

## 8. Resultados detallados

### 8.1 GLUE (Tabla 1)

| System | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| BiLSTM+ELMo+Attn | 76.4/76.1 | 64.8 | 79.8 | 90.4 | 36.0 | 73.3 | 84.9 | 56.8 | 71.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| **BERT-base** | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | **79.6** |
| **BERT-large** | 86.7/85.9 | 72.1 | 92.7 | 94.9 | 60.5 | 86.5 | 89.3 | 70.1 | **82.1** |

Comparación clave: **BERT-base (110M) vs OpenAI GPT (117M)**. Tamaños casi idénticos, mismo paradigma fine-tuning, diferencia: BERT es bidireccional con MLM/NSP, GPT es unidireccional con LM. Resultado: **+4.5 puntos en promedio GLUE**. Las ganancias más grandes son en tareas con dependencias bidireccionales claras: CoLA (+6.7, acceptability necesita ver toda la oración), STS-B (+5.8, similitud bidireccional), MRPC (+6.6, paráfrasis), RTE (+10.4, entailment con poco data).

### 8.2 SQuAD v1.1 (Tabla 2)

| System | Dev EM/F1 | Test EM/F1 |
|---|---|---|
| Human | 80.3 / 82.3 | — / 91.2 |
| nlnet (top leaderboard, ensemble) | — | 86.0 / 91.7 |
| BERT-large (Single) | 84.1 / 90.9 | — |
| BERT-large (Ensemble) | 85.8 / 91.8 | — |
| BERT-large (Single + TriviaQA) | 84.2 / 91.1 | 85.1 / 91.8 |
| **BERT-large (Ensemble + TriviaQA)** | 86.2 / 92.2 | **87.4 / 93.2** |

Notar: BERT supera al humano (91.2 F1) por **+2 puntos**. Es el primer modelo de QA en hacerlo en SQuAD 1.1.

### 8.3 SQuAD v2.0 (Tabla 3)

Más difícil porque incluye preguntas sin respuesta. BERT-large single: 81.9 F1 dev, 83.1 F1 test. Mejora **+5.1 F1** sobre el SOTA previo.

### 8.4 SWAG (Tabla 4)

| System | Test |
|---|---|
| ESIM+GloVe | 52.7 |
| ESIM+ELMo | 59.2 |
| OpenAI GPT | 78.0 |
| **BERT-large** | **86.3** |
| Human (expert) | 85.0 |
| Human (5 annotations) | 88.0 |

BERT supera al humano experto (85.0) por +1.3 puntos.

### 8.5 CoNLL-2003 NER (Tabla 7, feature-based)

| Approach | Dev F1 | Test F1 |
|---|---|---|
| ELMo | 95.7 | 92.2 |
| CSE (Akbik 2018) | — | 93.1 |
| BERT-large fine-tune | 96.6 | 92.8 |
| BERT-base fine-tune | 96.4 | 92.4 |
| BERT-base feature-based (concat last 4) | 96.1 | — |
| BERT-base feature-based (sum last 4) | 95.9 | — |
| BERT-base feature-based (last hidden) | 94.9 | — |
| BERT-base feature-based (second-to-last) | 95.6 | — |

Conclusión: feature-based con concat last-4 está a **0.3 F1** del fine-tuning. Confirma que BERT funciona bien en ambos paradigmas. La capa "last hidden" sola no es óptima — la penúltima funciona mejor porque la última está demasiado especializada en MLM.

---

## 9. Ablations clave (Sección 5)

### 9.1 Pre-training tasks (Tabla 5)

| Tasks | MNLI-m | QNLI | MRPC | SST-2 | SQuAD F1 |
|---|---|---|---|---|---|
| BERT-base (MLM + NSP) | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| No NSP (solo MLM) | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| LTR & No NSP (como GPT) | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| + BiLSTM encima del LTR | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

Lecturas:

- **Sin NSP** → cae 3.5 puntos en QNLI, 0.6 en SQuAD. Hit medible, no catastrófico.
- **LTR sin NSP** → cae 10.7 puntos en SQuAD F1, 9.2 en MRPC. La unidireccionalidad **destruye** las tareas token-level.
- Agregar un BiLSTM aleatorio encima del LTR recupera algo de SQuAD (84.9), pero degrada MRPC (75.7) y queda muy por debajo de BERT bidireccional desde el pre-training.

### 9.2 Efecto del tamaño (Tabla 6)

| $L$ | $H$ | $A$ | LM ppl | MNLI-m | MRPC | SST-2 |
|---|---|---|---|---|---|---|
| 3 | 768 | 12 | 5.84 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 3 | 5.24 | 80.6 | 82.2 | 90.7 |
| 6 | 768 | 12 | 4.68 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 3.99 | 84.4 | 86.7 | 92.9 |
| 12 | 1024 | 16 | 3.54 | 85.7 | 86.9 | 93.3 |
| 24 | 1024 | 16 | 3.23 | 86.6 | 87.8 | 93.7 |

Crecimiento monotónico en performance — y, crucialmente, **incluso en MRPC** que tiene solo 3,600 ejemplos de entrenamiento. Antes de BERT se creía que datasets pequeños no se beneficiaban de modelos más grandes (más capacidad → más overfit). BERT demuestra lo contrario: dado pre-training suficiente, escalar el modelo ayuda incluso a tareas con muy poco data downstream. Esta observación abrió la puerta a la era del scaling laws (Kaplan et al. 2020).

### 9.3 Pasos de pre-training (Apéndice C.1, Figura 5)

- BERT-base con 1M pasos vs 500K pasos: +1.0 en MNLI accuracy. Más pre-training ayuda.
- MLM converge más lento que LTR (porque predice 15% de tokens en vez de 100%), pero supera a LTR desde ~200K pasos. El paper concluye que el costo de convergencia es despreciable frente al beneficio de bidireccionalidad.

---

## 10. Limitaciones y crítica posterior

El paper de BERT, leído en 2026, tiene varias decisiones que la literatura posterior corrigió:

### 10.1 NSP es débil

RoBERTa (Liu et al. 2019) lo elimina sin pérdida. ALBERT lo reemplaza por SOP (más difícil). T5 lo ignora. El consenso 2020-2026 es que NSP no es necesario; basta MLM con corpus document-level y secuencias largas.

### 10.2 Mismatch `[MASK]` pretrain/finetune

La regla 80/10/10 lo mitiga pero no lo elimina. ELECTRA (Clark et al. 2020) ataca este problema directamente: en vez de predecir tokens enmascarados, entrena un **discriminador** que detecta tokens reemplazados por un generador pequeño. No hay `[MASK]` en ninguna fase. Resultado: ELECTRA-base iguala a RoBERTa-base con un cuarto del compute.

### 10.3 Static masking

RoBERTa usa dynamic masking — se aplica al vuelo en cada batch. ~0.5 puntos GLUE.

### 10.4 Tasa fija de 15%

Wettig et al. (2023, "Should you mask 15% in MLM?") muestra que con modelos más grandes y secuencias largas, **40% es óptimo**. La elección de 15% es heurística de 2018 y subóptima en escala moderna.

### 10.5 Sin scaling de batch

256 secuencias es pequeño. RoBERTa entrena con batch 8K. Los modelos posteriores (DeBERTa-v3, T5) usan batches todavía más grandes con LR adecuado.

### 10.6 Position embeddings absolutos aprendidos

Techo rígido de 512. RoFormer (RoPE), T5 (relative bias), ALiBi y posteriormente ModernBERT (2024) introducen position encodings extrapolables.

### 10.7 Post-LN

Difícil de entrenar profundo sin warmup cuidadoso. Pre-LN (Xiong et al. 2020) es más estable y se adopta en GPT-2 en adelante.

### 10.8 Sin generación

BERT es exclusivamente representacional. No puede generar texto. La era de los LLMs (GPT-3 en adelante) absorbió el caso de uso "entender + generar" en una sola arquitectura decoder-only.

### 10.9 Fine-tuning inestable en datasets pequeños

Mosbach et al. (2021) muestra que la inestabilidad de BERT-large en MRPC/RTE/CoLA es por warmup insuficiente y se resuelve con tricks (re-init de últimas capas, llr más bajo, más epochs).

---

## 11. Impacto y descendencia

### 11.1 Dominio de benchmarks 2018-2020

BERT lideró GLUE/SuperGLUE hasta que ELECTRA, RoBERTa y T5 lo desplazaron. SQuAD fue saturado (sobre humano) y reemplazado por SQuAD 2.0, luego por benchmarks más difíciles (DROP, QuAC, Natural Questions).

### 11.2 Familia BERT-like

| Modelo | Año | Innovación |
|---|---|---|
| **RoBERTa** (Facebook) | 2019 | No NSP, dynamic masking, batch grande, más data, más pasos |
| **DistilBERT** (HF) | 2019 | Distillation a 66M params, 97% performance |
| **ALBERT** (Google) | 2019 | Cross-layer param sharing, factorización de embeddings, SOP |
| **DeBERTa / v2 / v3** (Microsoft) | 2020-21 | Disentangled attention (content + position separados), ELECTRA-style training |
| **ELECTRA** (Stanford/Google) | 2020 | Replaced token detection, no MLM |
| **XLM-R** (Facebook) | 2019 | Multilingual scaled, 100 idiomas, 2.5TB CommonCrawl |
| **mBERT** (Google) | 2018 | Multilingual original, 104 idiomas Wikipedia |
| **BETO** (CENIA-UC) | 2020 | BERT español, Wikipedia + corpus chileno |
| **BERTIN** (BSC) | 2021 | RoBERTa-base español, training perplexity sampling |
| **ModernBERT** (Answer.AI / LightOn) | 2024 | RoPE, FlashAttention, GeGLU, 8K context, training-time fixes |

### 11.3 Dos nichos donde BERT sigue siendo SOTA

Aunque los LLMs decoder-only dominan generación, BERT-like models siguen siendo state of the art en:

1. **Embeddings densos para retrieval**: Sentence-BERT (Reimers & Gurevych 2019) y descendientes (E5, BGE, GTE, jina-embeddings, nomic-embed). Mucho más eficientes que extraer embeddings de LLMs.
2. **Cross-encoders de re-ranking** en pipelines RAG. Un cross-encoder BERT base re-rankea top-100 con latencia razonable, mientras un LLM generativo sería 50× más caro.

### 11.4 Influencia conceptual

- **Paradigma pretrain-finetune** como default en NLP industrial 2018-2022.
- **Masked modeling** se exportó a otros dominios: BEiT (visión), MAE (visión), wav2vec 2.0 (audio), AlphaFold (proteínas).
- Inspiró la pregunta "¿qué tarea de pre-training es óptima?" — derivó en ELECTRA, BART, T5, etc.

---

## 12. Conexión con la clase 20 del Diplomado IA UC

La clase 20 trata el **Camino 4** del curso: entender texto vía encoders bidireccionales. BERT es la pieza central. Las conexiones a las otras clases del Camino 4 y de la trayectoria total:

- **Clase 14 (Transformers)**: BERT reusa la mitad encoder del Transformer original. Comprender Vaswani 2017 es prerrequisito.
- **Clase 18 (GPT-1)**: contraste explícito — mismo paradigma fine-tuning, distinta direccionalidad. Tabla 1 muestra que el solo cambio a bidireccional + MLM/NSP da +4.5 GLUE.
- **Clase 19 (ELMo)**: contraste con bidireccionalidad shallow. BERT muestra por qué "deep bidirectional" supera a "concat de dos LMs".
- **Clase 21+ (Camino 4 avanzado)**: RoBERTa, DistilBERT, sentence-transformers, RAG con cross-encoders. Todo descendiente directo.

El `fundamentos/bert.md` del site cubre arquitectura, mini-implementación en código del curso (Mini-BERT 952K params) y aplicaciones. El `fundamentos/pretraining-bert.md` cubre el paradigma pretrain-finetune. El `papers/bert-devlin-2018.md` resume el paper. **Este documento aporta el detalle que esos tres dejan implícito**:

- justificación matemática y de leakage de bidireccionalidad
- ablations del Apéndice C (régimen de masking, número de pasos, tamaño)
- comparación numérica fina contra GPT-1 controlando por parámetros
- decisiones de ingeniería del Apéndice A.2 (TPU pods, schedule de longitud)
- limitaciones que RoBERTa, ALBERT, DeBERTa, ELECTRA atacaron
- conexión con scaling laws, Sentence-BERT y RAG moderno

---

## 13. Notas para integrar al site

Cosas que el `papers/bert-devlin-2018.md` actual **no** menciona y conviene agregar (sin duplicar lo que ya tiene):

1. **Tabla del régimen de masking** (Apéndice C.2): muestra que 80/10/10 es robusto pero no único.
2. **Schedule mixto de longitud 128/512** (Apéndice A.2): decisión de ingeniería relevante.
3. **TPU pods y costo**: 4 días en 4 Cloud TPUs (base) / 16 Cloud TPUs (large). $7K USD aproximado.
4. **WordPiece vs BPE**: diferencia técnica (likelihood unigram LM vs frecuencia), convención `##` vs `Ġ`.
5. **Comparación numérica BERT-base vs GPT-1**: 110M vs 117M, +4.5 GLUE atribuible a bidireccionalidad + MLM/NSP.
6. **Feature-based vs fine-tuning** en NER: 96.1 vs 96.4, diferencia de 0.3 F1.
7. **Limitaciones que la literatura corrigió**: NSP eliminado por RoBERTa, dynamic masking, ELECTRA reemplaza MLM, etc.
8. **Conexión RAG moderno**: el segundo aire de BERT como cross-encoder de re-ranking y backbone de Sentence-Transformers.

El `fundamentos/bert.md` ya cubre Mini-BERT del curso, comparación con LLaMA y context tokens — no necesita cambios.

El `fundamentos/pretraining-bert.md` ya cubre el paradigma; podría sumar el dato de que MLM converge marginalmente más lento que LTR pero supera desde ~200K pasos (Apéndice C.1).

---

## 14. Lectura recomendada complementaria

- **The Annotated Transformer** (Rush, 2018) — implementación de Vaswani 2017 paso a paso. Referenciada en footnote 2 del paper.
- **RoBERTa** (Liu et al. 2019) — recetas de pre-training que mejoran BERT sin cambiar la arquitectura.
- **ELECTRA** (Clark et al. 2020) — alternativa a MLM con replaced token detection.
- **A Primer in BERTology** (Rogers et al. 2020) — survey de interpretabilidad y análisis interno de BERT.
- **Should you mask 15% in MLM?** (Wettig et al. 2023) — revisión de la elección de masking rate.
- **ModernBERT** (Warner et al. 2024) — BERT con todos los upgrades modernos (RoPE, FlashAttention, GeGLU, 8K context).
- **Sentence-BERT** (Reimers & Gurevych 2019) — adaptación de BERT para embeddings de oraciones, base de retrieval moderno.
