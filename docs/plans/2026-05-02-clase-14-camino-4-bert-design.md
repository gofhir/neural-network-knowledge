# Clase 14 — Camino 4: Mini-BERT desde cero (Encoder-only)

**Fecha:** 2026-05-02
**Estado:** diseño aprobado, pendiente plan de implementación
**Contexto previo:** Caminos 1-2.5 completos en main. Mini-GPT + Mini-LLaMA (decoders) + BPE tokenizer. El usuario quiere estudiar el paradigma encoder en detalle, paso a paso.

## Motivación

Mini-GPT y Mini-LLaMA son modelos de generación (decoder-only). El objetivo de Camino 4 es construir el otro paradigma: un **encoder-only** (BERT-style) que entiende texto en lugar de generarlo. Demo final: detección de idioma EN/ES con >95% accuracy. El contraste decoder vs encoder es el contenido pedagógico central.

## Decisiones de diseño

| Dimensión | Decisión | Alternativas rechazadas |
|---|---|---|
| Demo final | Detección idioma EN/ES | Sentimiento (necesita etiquetado), MLM puro (no hay fine-tuning) |
| Tokenizador | Extender BPE 1112 + 3 special tokens → vocab 1115 | BPE nuevo (trabajo duplicado), char-level (pierde bilingualismo) |
| Arquitectura | Mini-BERT fiel al paper (LayerNorm, GELU, learned pos emb, MHA estándar) | Adaptar MiniLLaMA (mezcla confusa), híbrido (menos pedagógico) |
| Profundidad | Máxima granularidad (~12 caps) | Bloques amplios (oculta detalles importantes) |
| Branch | `feat/clase-14-camino-4-bert` desde main | |

## Mapa de capítulos — Fase 9 (caps 38-49)

```
Cap 38  38_encoder_vs_decoder.py        Visualizar causal vs bidireccional
Cap 39  39_positional_embeddings.py     Learned pos emb vs RoPE
Cap 40  40_special_tokens.py            [CLS], [MASK], [SEP] en accion
Cap 41  41_mini_bert.py                 Arquitectura completa, forward pass
Cap 42  42_mlm_loss.py                  MLM masking + 80/10/10 split
Cap 43  43_train_bert.py                MLM pretraining
Cap 44  44_eval_mlm.py                  Fill-in-the-blank interactivo
Cap 45  45_cls_head.py                  [CLS] como clasificador
Cap 46  46_dataset_lang.py              Dataset EN/ES, ventanas 64 tokens
Cap 47  47_finetune_bert.py             Fine-tuning clasificacion
Cap 48  48_eval_bert.py                 Accuracy + attention patterns + PCA
Cap 49  (solo Hugo)                     Comparativa BERT vs GPT, cierre
```

## Arquitectura Mini-BERT

### Diferencias con Mini-LLaMA

| Componente | Mini-LLaMA (decoder) | Mini-BERT (encoder) | Cap donde se explica |
|---|---|---|---|
| Atención | Causal (máscara triangular) | Bidireccional (sin máscara) | Cap 38 |
| Posición | RoPE (rotacional Q/K) | `nn.Embedding(max_seq_len, d_model)` | Cap 39 |
| Normalización | RMSNorm | LayerNorm estándar | Cap 41 |
| FFN | SwiGLU (gate) | GELU estándar | Cap 41 |
| Agrupación | GQA (h_q ≠ h_kv) | MHA estándar (h_q == h_kv) | Cap 41 |
| Objetivo | Next token prediction | MLM | Cap 42 |

### Hiperparámetros

```python
vocab_size   = 1115   # 1112 BPE + [CLS]=1112, [SEP]=1113, [MASK]=1114
max_seq_len  = 128
d_model      = 128
n_heads      = 4      # MHA estándar (no GQA)
n_layers     = 4
d_ff         = 512    # 4 × d_model (convención BERT)
activation   = "gelu"
norm         = "layernorm"  # pre-bloque (más estable que post)
positional   = "learned"    # nn.Embedding(128, 128)
# ≈ 1.1M params
```

### Clases nuevas en `_models.py`

```python
class LearnedPositionalEmbedding(nn.Module):
    """Embeddings de posicion aprendibles (BERT-style, no RoPE)."""

class BERTBlock(nn.Module):
    """Bloque BERT: pre-LayerNorm + MHA sin causal mask + FFN GELU."""

class MiniBERT(nn.Module):
    """Encoder-only: token emb + pos emb + N BERTBlocks."""
    # forward retorna (last_hidden_state, None) para compatibilidad

class MLMHead(nn.Module):
    """Linear d_model → vocab_size para prediccion MLM."""

class ClassificationHead(nn.Module):
    """[CLS] vector → Linear(d_model, n_classes). Para fine-tuning."""
```

## Special tokens

```python
CLS_TOKEN_ID  = 1112   # Agregado al BPETokenizer existente
SEP_TOKEN_ID  = 1113
MASK_TOKEN_ID = 1114

# Formato de input:
# [CLS] tok_1 tok_2 ... tok_N [SEP]
# El vector de salida de [CLS] = representacion agregada de la secuencia
```

Extensión del BPETokenizer: agregar `add_special_tokens()` que inserta los 3 tokens al vocab y extiende `id_to_token`. Compatible hacia atrás — los 1112 tokens originales no cambian.

## MLM Pretraining

### El objetivo

```
Input:  [CLS] To [MASK] or not to [MASK] [SEP]
Target: predecir "be" en pos 2, "be" en pos 6
Loss:   CrossEntropy solo sobre posiciones enmascaradas (mask=1)
```

Simétrico al SFT del cap 24:
- SFT: `mask=1` sobre tokens de RESPUESTA, `mask=0` sobre prompt
- MLM: `mask=1` sobre tokens ENMASCARADOS, `mask=0` sobre el resto

### 80/10/10 split de BERT

De los tokens a enmascarar (15% del input):
- **80%** → reemplazar con `[MASK]`
- **10%** → reemplazar con token aleatorio del vocab
- **10%** → mantener token original (el modelo predice igualmente)

Razón: evitar que el modelo aprenda a "solo atender [MASK]".

### Corpus

Shakespeare + Quijote tokenizado con BPETokenizer extendido (1115 tokens). Mismos archivos del Camino 2.5.

### Hiperparámetros pretraining

```python
mask_prob    = 0.15
batch_size   = 32
lr           = 1e-4
iters        = 3000
mask_split   = (0.80, 0.10, 0.10)
device       = "mps"
```

## Fine-tuning — Detección de idioma

### Dataset (cap 46)

```python
# Ground truth perfecto, cero etiquetado manual
EN_examples: ventanas 64 tokens de shakespeare.txt → label=0
ES_examples: ventanas 64 tokens de quijote.txt    → label=1
# Train: 2000 (1000 EN + 1000 ES)
# Eval:   500 (250 EN + 250 ES)
# Formato: [CLS] tok_1...tok_64 [SEP]
```

### Fine-tuning (cap 47)

```python
# Cargar MiniBERT pretrained
# Agregar ClassificationHead(d_model=128, n_classes=2)
# Loss: CrossEntropy sobre [CLS] output
lr     = 2e-5   # 5× menor que pretraining (convención BERT)
iters  = 500    # Dataset chico, converge rápido
```

### Evaluación (cap 48)

Tres componentes:

1. **Accuracy EN/ES**: esperada >95% (el encoder ya conoce ambos idiomas desde MLM).

2. **Attention patterns**: visualización ASCII de las matrices de atención por capa. Contraste:
   ```
   Decoder (causal):       cada token solo ve tokens anteriores
   Encoder (bidireccional): cada token ve TODOS los demás
   ```

3. **[CLS] embeddings en 2D**: PCA de los vectores [CLS] de frases EN y ES. Antes de fine-tuning: mezcla. Después: clusters separados.

## Cierre Camino 4 (cap 49 — solo Hugo)

Tabla comparativa tripartita:

| | Mini-GPT | Mini-LLaMA | Mini-BERT |
|---|---|---|---|
| Tipo | Decoder | Decoder | Encoder |
| Atención | Causal | Causal + GQA + RoPE | Bidireccional |
| Posición | Aprendida | RoPE | Aprendida |
| Normalización | LayerNorm | RMSNorm | LayerNorm |
| FFN | ReLU | SwiGLU | GELU |
| Objetivo | Next token | Next token | MLM |
| Paper | Vaswani 2017 | LLaMA 2023 | Devlin 2018 |

Contexto histórico: encoders dominaron NLP 2019-2022 (BERT, RoBERTa, DeBERTa). Decoders escalaron desde 2022 (GPT-3, LLaMA). Hoy los encoders sobreviven en embeddings (sentence-transformers) y re-ranking (RAG cross-encoders).

## Tests nuevos

```python
# tests/test_bert.py
test_learned_pos_emb_shape()          # (B, T, d_model)
test_bert_block_no_causal_mask()      # attention es simétrica
test_mlm_masking_80_10_10()           # distribución correcta
test_mini_bert_forward_shape()        # (B, T, d_model)
test_cls_head_output_shape()          # (B, n_classes)
test_bpe_special_tokens_extension()   # vocab 1112 → 1115
```

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| MLM loss no baja (bug en masking) | Test que verifica mask.sum() > 0 por batch |
| Accuracy <90% en detección idioma | Verificar que ventanas de eval no se solapan con train |
| Attention visualization confusa | Usar matshow ASCII simple, no heatmap externo |
| PCA en 2D no separa clusters | Usar primeras 2 componentes PCA de 128 dims, es suficiente |

## Convenciones heredadas

- Pedagogia conversacional: concepto → script → output literal → preguntas
- Scripts `38_*.py` ... `48_*.py` (matchean caps Hugo)
- Capitulos Hugo en español sin tildes
- Output real en caps (no inventado)
- Honestidad sobre resultados
- No Co-Authored-By en commits
