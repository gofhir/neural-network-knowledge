---
title: "BERT (Bidirectional Encoder Representations from Transformers)"
weight: 295
math: true
---

**BERT** (Bidirectional Encoder Representations from Transformers) es la arquitectura encoder-only introducida por **Devlin et al. (NAACL 2019)** que cambio NLP entre 2018 y 2022. Donde GPT genera texto auto-regresivamente con atencion causal, BERT representa texto bidireccionalmente con un objetivo de pretraining nuevo — Masked Language Modeling (MLM) — y un paradigma de uso distinto: **pretrain masivo + fine-tuning ligero por tarea**. Aunque los decoder-only LLMs (GPT-4, Claude, LLaMA) desplazaron a BERT como protagonista de NLP desde 2022, BERT-like models siguen dominando dos nichos criticos: embeddings densos para busqueda semantica (Sentence-Transformers) y cross-encoders de re-ranking en pipelines RAG. Es el corazon del *Camino 4* del curso.

---

## 1. Apertura: el "otro paradigma" del Transformer

El paper original "Attention is all you need" (Vaswani et al., 2017) introdujo el Transformer como arquitectura encoder-decoder para traduccion. En 2018 dos descendientes divergieron radicalmente:

- **GPT** (Radford et al., junio 2018): mantuvo solo el decoder. Atencion causal. Objetivo: next-token prediction. Pretraining sobre BookCorpus. Uso: generacion via prompting o fine-tuning.
- **BERT** (Devlin et al., octubre 2018): mantuvo solo el encoder. Atencion bidireccional (sin mascara). Objetivo: Masked Language Modeling. Pretraining sobre Wikipedia + BookCorpus. Uso: representacion via fine-tuning con cabeza de tarea.

Durante 4 anos (2018-2022), BERT y sus variantes (RoBERTa, DistilBERT, ELECTRA, ALBERT, DeBERTa, XLM-R) dominaron casi todos los benchmarks de NLP. Hugging Face nacio en parte como hub para distribuir modelos BERT-like fine-tuneados.

A partir de 2022, los decoder-only escalaron a tamanos donde podian hacer las tareas tradicionales de BERT zero-shot via prompting. ChatGPT, GPT-4, Claude y LLaMA-3 hicieron innecesario fine-tunear un encoder para sentimiento o clasificacion. Pero los encoders no desaparecieron: encontraron una segunda vida en sistemas de retrieval, ranking y embeddings — tareas donde la asimetria del decoder lo hace inferior. Hoy, en 2026, todo pipeline RAG serio usa modelos BERT-like en sus etapas de busqueda.

---

## 2. La idea central: bidireccionalidad y MLM

### Bidireccionalidad de la atencion

Un decoder usa atencion causal: el token en posicion $t$ solo puede atender a posiciones $\leq t$. Esto es necesario para generacion auto-regresiva — sin ello, el modelo "veria el futuro" durante el training.

Un encoder elimina la mascara causal. Cada token atiende a TODAS las posiciones, antes y despues:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Sin mascara aditiva $-\infty$ en el triangulo superior. Esto da a cada token una representacion que integra contexto bidireccional — lo que es ideal para clasificacion y representacion, pero rompe la generacion auto-regresiva.

### Masked Language Modeling (MLM)

Si el modelo "ve el futuro", el next-token-prediction trivializa: predecir token $t$ es trivial cuando ya viste token $t$. BERT introduce un objetivo nuevo: enmascarar un porcentaje de tokens (15%) y predecirlos:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \log P(x_i | x_{\setminus M})$$

donde $M$ es el conjunto de posiciones enmascaradas. La distribucion 80/10/10 — 80% de los tokens enmascarados se reemplazan por `[MASK]`, 10% por un token aleatorio, 10% se mantienen — es deliberada: previene que el modelo aprenda solo a "decodificar `[MASK]`" sin extraer representaciones utiles para los tokens no enmascarados.

En PyTorch, MLM se implementa con `ignore_index=-100` en `cross_entropy`: las posiciones no enmascaradas reciben target `-100` y son excluidas del loss. Esto es matematicamente identico al loss masking de SFT (cap 24): solo se penalizan tokens especificos, el resto contribuye 0 al gradiente.

### El token `[CLS]`: resumen de secuencia

BERT prepende `[CLS]` (classification) al inicio de cada secuencia y agrega `[SEP]` (separator) al final. Durante MLM, `[CLS]` no es enmascarado ni predicho — pero participa en la atencion bidireccional con todos los demas tokens, acumulando informacion global.

Despues del pretraining, el vector `[CLS]` en la ultima capa del encoder se usa directamente para tareas de clasificacion: se proyecta con una capa lineal `d_model -> n_classes` y se entrena con cross-entropy. Esta es la convencion del paper original. Para Mini-BERT (cap 45), `d_model=128` y `n_classes=2` — solo 258 parametros adicionales.

---

## 3. Pretraining + fine-tuning: el paradigma BERT

El paper de Devlin formalizo lo que hoy es obvio: separar el aprendizaje de representacion del aprendizaje de tarea.

### Etapa 1: pretraining (caro, una vez)

- Corpus enorme de texto sin etiquetar (Wikipedia + BookCorpus = 3.3B tokens en BERT-base).
- Objetivo: MLM (predicciones de tokens enmascarados) + NSP (Next Sentence Prediction, eliminado en RoBERTa por ser inutil).
- Resultado: encoder con representaciones ricas del lenguaje, almacenadas en los pesos.

En el curso, cap 43 hace MLM pretraining sobre Shakespeare+Quijote durante 3000 iteraciones. La loss baja de $\log(1115) \approx 7.02$ (entropia uniforme sobre el vocab) a 4.96 — el encoder aprendio a predecir tokens enmascarados con razonable precision en un corpus bilingue.

### Etapa 2: fine-tuning (barato, por tarea)

- Dataset etiquetado especifico (clasificacion, NER, QA, etc.).
- Cabeza nueva sobre el encoder pretrained.
- LR mucho menor (5-10x menor que pretraining) para evitar **catastrophic forgetting** — destruir el conocimiento del pretrain con el gradiente de la tarea.
- Pocas iteraciones (cientos a miles, no millones).

En cap 47, fine-tuning a deteccion de idioma EN/ES con `LR=2e-5` (5x menor que el pretrain `1e-4`), 500 iteraciones, 2000 ejemplos. Loss baja de 0.62 a 0.08, accuracy en eval = 0.998. Lo critico: el encoder no olvido lo que aprendio en MLM. Si se reusara para otra tarea, las representaciones lingüisticas seguirian intactas.

Esta asimetria es la esencia: **un encoder pretrained se reusa para decenas de tareas** con solo cambiar la cabeza y fine-tunear con LR bajo. El conocimiento del lenguaje vive en el encoder; el conocimiento de tarea vive en la cabeza.

---

## 4. Arquitectura tecnica

BERT base tiene 12 capas, $d_{\text{model}}=768$, 12 heads, $d_{\text{ff}}=3072$. Total ~110M parametros. Mini-BERT del curso tiene 4 capas, $d_{\text{model}}=128$, 4 heads, $d_{\text{ff}}=512$, total 952K parametros — 100x mas chico pero estructuralmente identico.

Cada bloque BERT (`BERTBlock`) tiene:

1. Multi-Head Attention sin mascara causal (`is_causal=False` o equivalente).
2. Residual + LayerNorm (post-LN, igual que el paper original).
3. FFN con activacion **GELU** (no SwiGLU como LLaMA).
4. Residual + LayerNorm.

La diferencia visible respecto a Mini-LLaMA es la atencion sin mascara y los special tokens. Las decisiones modernas (RMSNorm, RoPE, GQA, SwiGLU) que LLaMA introdujo no se aplican al BERT clasico — aunque variantes modernas como ModernBERT (2024) si las adoptan. Para fidelidad pedagogica al paper de 2018, Mini-BERT usa la formulacion original.

### Special tokens en el BPE

El BPE del curso (cap 30) tiene 1112 tokens. BERT extiende el vocab con tres tokens nuevos:

- `[CLS]` (id 1112): inicio de secuencia, vector de clasificacion.
- `[SEP]` (id 1113): separador de segmentos.
- `[MASK]` (id 1114): placeholder para MLM.

Vocab final: 1115 tokens. La tabla de embeddings se extiende, los embeddings nuevos se inicializan aleatoriamente, y aprenden representaciones durante el pretraining.

---

## 5. BERT vs GPT: la decision arquitectonica

| Dimension | BERT (encoder-only) | GPT (decoder-only) |
|---|---|---|
| Atencion | Bidireccional | Causal |
| Objetivo pretrain | MLM (15% mask) | Next-token prediction |
| Genera texto | No | Si (auto-regresivo) |
| Output canonico | Vector `[CLS]` | Logits sobre vocab |
| Uso tipico | Fine-tuning + cabeza | Prompting o fine-tuning |
| Fortaleza | Representacion / clasificacion | Generacion / razonamiento |

La regla practica: **si la salida es una etiqueta, vector o score → encoder. Si la salida es texto libre → decoder.** Encoder-decoder (T5, BART) cubre el caso de transformacion texto-a-texto pero ha perdido protagonismo frente a decoders grandes via prompting.

---

## 6. La segunda vida: embeddings y RAG

A partir de 2022, casi todas las tareas de NLP de baja-a-media complejidad pueden hacerse con un decoder grande zero-shot. ¿Por que entonces seguir aprendiendo BERT?

### Sentence-Transformers

Reimers & Gurevych (EMNLP 2019) reformularon BERT con un objetivo contrastivo: en lugar de fine-tunear con cross-entropy de clases, entrenar dos encoders identicos (siamese network) para que vectores `[CLS]` de oraciones similares se acerquen y los de oraciones disimiles se alejen.

El resultado: BERT como motor de embeddings densos. Modelos como `all-MiniLM-L6-v2`, `bge-large-en-v1.5`, `e5-large-v2` son la columna vertebral de:

- Buscadores semanticos (retrieval)
- Sistemas FAQ
- Detectores de duplicados
- Recomendadores
- La etapa 1 de pipelines RAG

Para producir embeddings de alta calidad por dolar, los encoders bidireccionales superan a los decoders. Los decoders grandes producen embeddings asimetricos (solo el ultimo token "ve" todo el contexto).

### Cross-encoders en RAG

Un sistema RAG moderno tiene tres etapas:

1. **Retrieval**: buscar top-100 documentos relevantes via embeddings (encoder).
2. **Re-ranking**: re-puntuar con un cross-encoder BERT-like que ve query y documento juntos en una sola secuencia `[CLS] query [SEP] documento [SEP]`.
3. **Generation**: pasar top-5 al LLM grande (decoder) que genera la respuesta.

El cross-encoder es lento (procesa cada par independientemente) pero mucho mas preciso que la similitud de embeddings — porque la atencion bidireccional permite que pregunta y documento se "miren" mutuamente. Modelos canonicos: `cross-encoder/ms-marco-MiniLM-L-12-v2`, `BAAI/bge-reranker-large`, `cohere/rerank-english-v3.0`.

---

## 7. Aplicaciones actuales y limitaciones

### Donde BERT-like sigue siendo SOTA en 2026

- Embeddings densos para retrieval semantico.
- Cross-encoder re-ranking en pipelines RAG.
- Modelos multimodales con encoders por modalidad (CLIP, SigLIP — el encoder de texto es BERT-like).
- Clasificacion a escala donde el costo de un LLM por consulta es prohibitivo.
- Vision Transformers (ViT) — la misma arquitectura encoder aplicada a parches de imagen.

### Donde BERT-like ya no compite

- Generacion de texto libre (siempre fue imposible).
- Tareas open-domain donde el LLM grande zero-shot supera al encoder fine-tuned.
- Razonamiento complejo, codigo, math, agentes — territorio exclusivo de decoders grandes.

### Limitaciones inherentes

- No genera texto: para cualquier tarea generativa requiere un decoder externo.
- Limitado por el contexto del pretraining: BERT-base tiene `max_seq_len=512`, ModernBERT amplia a 8192.
- Asimetrico en eficiencia: barato para representacion, caro para producir output token-by-token.

---

## 8. Lugar en el curso

- **Cap 38**: encoder vs decoder, justificacion de eliminar la mascara causal.
- **Caps 39-41**: positional embeddings aprendidos, special tokens, arquitectura completa Mini-BERT.
- **Caps 42-44**: MLM loss, pretraining con apply_mlm_mask, eval con predict_mask.
- **Caps 45-48**: fine-tuning a deteccion de idioma EN/ES con accuracy 0.998.
- **Cap 49**: comparativa final BERT vs GPT, historia 2018-2026, RAG y Sentence-Transformers.

Caminos relacionados:

- [Foundation Models](/fundamentos/foundation-models) — BERT como uno de los primeros foundation models.
- [Mecanismo de atencion](/fundamentos/mecanismo-atencion) — la atencion bidireccional es la diferencia central.
- [Embeddings distribuidos](/fundamentos/embeddings-distribuidos) — BERT como evolucion de Word2Vec.
- [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) — base de Sentence-Transformers.
- [Loss masking](/fundamentos/loss-masking) — el truco que une SFT y MLM.

El paradigma que BERT introdujo — pretrain masivo + fine-tuning ligero + cabeza por tarea — es la base conceptual de toda la era moderna de NLP. Aunque GPT y sus descendientes lo desplazaron como protagonista, BERT sigue siendo la columna vertebral de busqueda y representacion en 2026.

---

**Referencias:**

- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*.
- Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
- Warner, B. et al. (2024). *ModernBERT: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*.
