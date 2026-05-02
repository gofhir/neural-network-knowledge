# Clase 14 — Camino 4 (Mini-BERT Encoder-only) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Construir Mini-BERT desde cero — encoder-only bidireccional con MLM pretraining y fine-tuning para detección de idioma EN/ES, produciendo 12 capítulos Hugo (caps 38-49) + 11 scripts ejecutables con outputs literales verificados.

**Architecture:** MiniBERT usa token embeddings + learnable positional embeddings + N BERTBlocks (pre-LayerNorm + MHA sin causal mask + FFN GELU), con MLMHead para pretraining y ClassificationHead ([CLS] → linear) para fine-tuning. Todo agrega al `_models.py` existente. El BPETokenizer existente se extiende con 3 special tokens (vocab 1112 → 1115).

**Tech Stack:** Python 3.12 + PyTorch (MPS) + Hugo. Venv existente en `clase_14/practica/.venv/`. Branch `feat/clase-14-camino-4-bert` desde `main`.

**Design doc:** `docs/plans/2026-05-02-clase-14-camino-4-bert-design.md`

**Working dir:** `/Users/robertoaraneda/projects/personal/courses/ia-uc`

**Verification model:**
- Clases Python → TDD con pytest
- Scripts ejecutables → corren + output en rango esperado
- Capítulos Hugo → output literal en cap, hugo build limpio

---

## Task 0: Branch setup

**Files:** New branch `feat/clase-14-camino-4-bert`

**Step 1: Crear rama**
```bash
git checkout main
git checkout -b feat/clase-14-camino-4-bert
```

**Step 2: Verificar baseline**
```bash
cd clase_14/practica
source .venv/bin/activate
python -m pytest tests/ -v 2>&1 | tail -5
```
Expected: 11/11 PASS.

**Step 3: Crear test file**
```bash
touch clase_14/practica/tests/test_bert.py
```

**Step 4: Commit**
```bash
git add clase_14/practica/tests/test_bert.py
git commit -m "chore: branch Camino 4 — Mini-BERT encoder-only"
```

---

## Task 1: BPETokenizer — extensión con special tokens (TDD)

**Files:**
- Modify: `clase_14/practica/_bpe.py`
- Modify: `clase_14/practica/tests/test_bert.py`

**Step 1: Test failing**
```python
# tests/test_bert.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bpe import BPETokenizer

def test_bpe_special_tokens_extension():
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "shakespeare.txt")
    tok = BPETokenizer()
    tok.train(open(corpus_path).read(), num_merges=10)
    original_size = tok.vocab_size
    tok.add_special_tokens()
    assert tok.vocab_size == original_size + 3
    assert "[CLS]" in tok.vocab
    assert "[SEP]" in tok.vocab
    assert "[MASK]" in tok.vocab
    assert tok.cls_id == tok.vocab["[CLS]"]
    assert tok.sep_id == tok.vocab["[SEP]"]
    assert tok.mask_id == tok.vocab["[MASK]"]

def test_encode_bert_adds_cls_sep():
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "shakespeare.txt")
    tok = BPETokenizer()
    tok.train(open(corpus_path).read(), num_merges=10)
    tok.add_special_tokens()
    ids = tok.encode_bert("hello")
    assert ids[0] == tok.cls_id
    assert ids[-1] == tok.sep_id
    assert len(ids) >= 3  # [CLS] + al menos 1 token + [SEP]
```

**Step 2: Run, expect FAIL**
```bash
cd clase_14/practica && source .venv/bin/activate
python -m pytest tests/test_bert.py -v
```
Expected: FAIL — `AttributeError: 'BPETokenizer' object has no attribute 'add_special_tokens'`

**Step 3: Implementar en `_bpe.py`**

Agregar al final de la clase `BPETokenizer`:
```python
def add_special_tokens(self) -> None:
    """Agrega [CLS], [SEP], [MASK] al vocab. Idempotente."""
    for tok in ["[CLS]", "[SEP]", "[MASK]"]:
        if tok not in self.vocab:
            idx = len(self.vocab)
            self.vocab[tok] = idx
            self.id_to_token[idx] = tok
    self.cls_id  = self.vocab["[CLS]"]
    self.sep_id  = self.vocab["[SEP]"]
    self.mask_id = self.vocab["[MASK]"]

def encode_bert(self, text: str) -> list[int]:
    """Encode con [CLS] al inicio y [SEP] al final."""
    return [self.cls_id] + self.encode(text) + [self.sep_id]
```

**Step 4: Run ALL tests**
```bash
python -m pytest tests/ -v
```
Expected: 13/13 PASS.

**Step 5: Commit**
```bash
git add clase_14/practica/_bpe.py clase_14/practica/tests/test_bert.py
git commit -m "feat(_bpe): add_special_tokens + encode_bert con [CLS][SEP]"
```

---

## Task 2: LearnedPositionalEmbedding (TDD)

**Files:**
- Modify: `clase_14/practica/_models.py`
- Modify: `clase_14/practica/tests/test_bert.py`

**Step 1: Test failing**
```python
def test_learned_pos_emb_shape():
    import torch
    from _models import LearnedPositionalEmbedding
    emb = LearnedPositionalEmbedding(max_seq_len=128, d_model=64)
    x = torch.zeros(2, 10, 64)  # (B=2, T=10, d=64)
    out = emb(x)
    assert out.shape == (2, 10, 64)

def test_learned_pos_emb_different_positions():
    import torch
    from _models import LearnedPositionalEmbedding
    emb = LearnedPositionalEmbedding(max_seq_len=128, d_model=64)
    x = torch.zeros(1, 5, 64)
    out = emb(x)
    # Cada posicion debe dar output distinto (embeddings distintos)
    assert not torch.all(out[0, 0] == out[0, 1])
```

**Step 2: Run, expect FAIL**

**Step 3: Implementar en `_models.py`** (agregar después de la clase `MiniLLaMA`):
```python
class LearnedPositionalEmbedding(nn.Module):
    """Embeddings de posicion aprendibles (BERT-style, no RoPE)."""
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return x + self.embedding(positions)  # broadcast sobre B
```

**Step 4: Run ALL tests** — Expected: 15/15 PASS.

**Step 5: Commit**
```bash
git add clase_14/practica/_models.py clase_14/practica/tests/test_bert.py
git commit -m "feat(_models): LearnedPositionalEmbedding BERT-style"
```

---

## Task 3: BERTBlock — atención bidireccional (TDD)

**Files:**
- Modify: `clase_14/practica/_models.py`
- Modify: `clase_14/practica/tests/test_bert.py`

**Step 1: Test failing**
```python
def test_bert_block_shape():
    import torch
    from _models import BERTBlock
    block = BERTBlock(d_model=64, n_heads=4, d_ff=256)
    x = torch.randn(2, 10, 64)
    out = block(x)
    assert out.shape == (2, 10, 64)

def test_bert_block_bidirectional():
    """El token 0 puede atender al token 9 (bidireccional, no causal)."""
    import torch
    from _models import BERTBlock
    torch.manual_seed(42)
    block = BERTBlock(d_model=64, n_heads=4, d_ff=256)
    x = torch.randn(1, 10, 64)
    # Con atención bidireccional, cambiar el último token cambia la salida del primero
    x1 = x.clone(); x2 = x.clone()
    x2[0, 9] = x2[0, 9] * 10  # modificar token 9
    out1 = block(x1); out2 = block(x2)
    # El token 0 debe ser distinto (vio el token 9)
    assert not torch.allclose(out1[0, 0], out2[0, 0], atol=1e-4)
```

**Step 2: Run, expect FAIL**

**Step 3: Implementar en `_models.py`**:
```python
class BERTBlock(nn.Module):
    """Bloque BERT: pre-LayerNorm + MHA sin causal mask + FFN GELU."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm + residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)  # sin causal mask
        x = x + attn_out
        normed2 = self.norm2(x)
        ff_out = self.ff2(F.gelu(self.ff1(normed2)))
        return x + ff_out
```

**Step 4: Run ALL tests** — Expected: 17/17 PASS.

**Step 5: Commit**
```bash
git add clase_14/practica/_models.py clase_14/practica/tests/test_bert.py
git commit -m "feat(_models): BERTBlock bidireccional con pre-LayerNorm + GELU"
```

---

## Task 4: MiniBERT + MLMHead + ClassificationHead (TDD)

**Files:**
- Modify: `clase_14/practica/_models.py`
- Modify: `clase_14/practica/tests/test_bert.py`

**Step 1: Tests failing**
```python
def test_mini_bert_forward_shape():
    import torch
    from _models import MiniBERT
    model = MiniBERT(vocab_size=1115, max_seq_len=128, d_model=64,
                     n_heads=4, n_layers=2, d_ff=256)
    x = torch.randint(0, 1115, (2, 20))  # (B=2, T=20)
    h = model(x)
    assert h.shape == (2, 20, 64)

def test_mlm_head_shape():
    import torch
    from _models import MiniBERT, MLMHead
    model = MiniBERT(vocab_size=1115, max_seq_len=128, d_model=64,
                     n_heads=4, n_layers=2, d_ff=256)
    head = MLMHead(d_model=64, vocab_size=1115)
    x = torch.randint(0, 1115, (2, 20))
    logits = head(model(x))
    assert logits.shape == (2, 20, 1115)

def test_classification_head_uses_cls():
    import torch
    from _models import MiniBERT, ClassificationHead
    model = MiniBERT(vocab_size=1115, max_seq_len=128, d_model=64,
                     n_heads=4, n_layers=2, d_ff=256)
    head = ClassificationHead(d_model=64, n_classes=2)
    x = torch.randint(0, 1115, (3, 20))
    logits = head(model(x))
    assert logits.shape == (3, 2)
```

**Step 2: Run, expect FAIL**

**Step 3: Implementar en `_models.py`**:
```python
class MiniBERT(nn.Module):
    """Encoder-only: token emb + positional emb aprendido + N BERTBlocks."""
    def __init__(self, vocab_size: int, max_seq_len: int = 128, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 4, d_ff: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = LearnedPositionalEmbedding(max_seq_len, d_model)
        self.blocks    = nn.ModuleList([
            BERTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm      = nn.LayerNorm(d_model)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) — indices de tokens
        h = self.token_emb(x)   # (B, T, d_model)
        h = self.pos_emb(h)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)      # (B, T, d_model)


class MLMHead(nn.Module):
    """Proyecta d_model → vocab_size para prediccion MLM."""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)   # (B, T, vocab_size)


class ClassificationHead(nn.Module):
    """Toma el vector [CLS] (posicion 0) y lo proyecta a n_classes."""
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, d_model) — tomar posicion 0 = [CLS]
        return self.linear(h[:, 0, :])  # (B, n_classes)
```

**Step 4: Run ALL tests** — Expected: 20/20 PASS.

**Step 5: Commit**
```bash
git add clase_14/practica/_models.py clase_14/practica/tests/test_bert.py
git commit -m "feat(_models): MiniBERT + MLMHead + ClassificationHead con tests"
```

---

## Task 5: MLM masking utility (TDD)

**Files:**
- Create: `clase_14/practica/_bert_utils.py`
- Modify: `clase_14/practica/tests/test_bert.py`

**Step 1: Tests failing**
```python
def test_mlm_mask_proportion():
    import torch
    from _bert_utils import apply_mlm_mask
    # ids de 100 tokens normales (no especiales)
    ids = torch.randint(0, 1112, (1, 100))
    masked_ids, labels = apply_mlm_mask(ids.clone(), mask_prob=0.15,
                                         mask_id=1114, vocab_size=1115)
    # labels == -100 para tokens no enmascarados
    n_masked = (labels != -100).sum().item()
    # Esperamos ~15 enmascarados (±5 por aleatoriedad)
    assert 5 <= n_masked <= 30

def test_mlm_labels_minus100_for_unmasked():
    import torch
    from _bert_utils import apply_mlm_mask
    ids = torch.randint(0, 1112, (1, 50))
    _, labels = apply_mlm_mask(ids.clone(), mask_prob=0.15,
                                mask_id=1114, vocab_size=1115)
    # Solo los enmascarados tienen label >= 0
    assert (labels >= 0).sum() <= 50
    assert (labels == -100).sum() + (labels >= 0).sum() == 50

def test_mlm_special_tokens_never_masked():
    """[CLS], [SEP], [MASK] nunca se enmascaran, incluso con mask_prob=1.0."""
    import torch
    from _bert_utils import apply_mlm_mask
    # Secuencia: [CLS] tok tok [SEP]
    ids = torch.tensor([[1112, 100, 200, 1113]])
    _, labels = apply_mlm_mask(ids.clone(), mask_prob=1.0,
                                mask_id=1114, vocab_size=1115,
                                special_ids=(1112, 1113, 1114))
    assert labels[0, 0].item() == -100   # [CLS] nunca enmascarado
    assert labels[0, -1].item() == -100  # [SEP] nunca enmascarado
    # Los tokens normales (pos 1, 2) SI deben estar enmascarados con prob=1.0
    assert labels[0, 1].item() != -100
    assert labels[0, 2].item() != -100
```

**Step 2: Run, expect FAIL**

**Step 3: Crear `_bert_utils.py`**:
```python
"""_bert_utils.py — utilidades para MLM masking y datasets BERT."""
import torch


def apply_mlm_mask(input_ids: torch.Tensor, mask_prob: float = 0.15,
                   mask_id: int = 1114, vocab_size: int = 1115,
                   special_ids: tuple = (1112, 1113, 1114)) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica masking MLM con split 80/10/10 de BERT.

    Returns:
        masked_ids: input_ids con tokens reemplazados
        labels:     original ids donde mask=1, -100 donde no se predice
    """
    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)  # -100 = ignorar en loss

    B, T = input_ids.shape
    for b in range(B):
        for t in range(T):
            tok = input_ids[b, t].item()
            if tok in special_ids:
                continue  # nunca enmascarar [CLS], [SEP], [MASK]
            if torch.rand(1).item() < mask_prob:
                labels[b, t] = tok  # guardar original como target
                r = torch.rand(1).item()
                if r < 0.80:
                    masked_ids[b, t] = mask_id          # [MASK]
                elif r < 0.90:
                    masked_ids[b, t] = torch.randint(0, vocab_size - 3, (1,)).item()
                # else: mantener original (10%)

    return masked_ids, labels
```

**Step 4: Run ALL tests** — Expected: 22/22 PASS.

**Step 5: Commit**
```bash
git add clase_14/practica/_bert_utils.py clase_14/practica/tests/test_bert.py
git commit -m "feat(_bert_utils): apply_mlm_mask con split 80/10/10"
```

---

## Task 6: Cap 38 — script + Hugo (encoder vs decoder visual)

**Files:**
- Create: `clase_14/practica/38_encoder_vs_decoder.py`
- Create: `site/content/clases/clase-14/practica/38-encoder-vs-decoder.md`

**Step 1: Escribir el script**
```python
"""38_encoder_vs_decoder.py - Cap 38: encoder vs decoder.

Visualiza la diferencia entre mascara causal (decoder) y
atencion bidireccional (encoder) sobre la misma frase.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)

T = 6  # longitud de secuencia de ejemplo
frase = ["To", "be", "or", "not", "to", "be"]

# === Mascara causal (decoder) ===
causal = torch.tril(torch.ones(T, T)).bool()
print("=== Mascara CAUSAL (decoder) ===")
print("Cada token solo puede atender tokens anteriores (incluyendose):\n")
header = f"{'':>6}" + "".join(f"{w:>6}" for w in frase)
print(header)
for i, wi in enumerate(frase):
    row = f"{wi:>6}" + "".join("  SI  " if causal[i, j] else "  NO  " for j in range(T))
    print(row)

# === Sin mascara (encoder) ===
print("\n=== Atencion BIDIRECCIONAL (encoder) ===")
print("Cada token puede atender a TODOS los tokens:\n")
print(header)
for i, wi in enumerate(frase):
    row = f"{wi:>6}" + "".join("  SI  " for _ in range(T))
    print(row)

# === Scores de atencion reales (un head aleatorio) ===
print("\n=== Scores de atencion encoder (un head) ===")
print("Muestra como 'be' (pos 1) atiende a todos:\n")
Q = torch.randn(T, 16)  # d_k = 16
K = torch.randn(T, 16)
scores = (Q @ K.T) / (16 ** 0.5)
attn_full = F.softmax(scores, dim=-1)
print("Pesos de atencion del token 'be' sobre todos los tokens:")
for j, wj in enumerate(frase):
    print(f"  be → {wj:>4}: {attn_full[1, j]:.3f}")

scores_causal = scores.masked_fill(~causal, float('-inf'))
attn_causal = F.softmax(scores_causal, dim=-1)
print("\nPesos de atencion del token 'not' (decoder, solo ve hasta 'not'):")
for j, wj in enumerate(frase):
    v = attn_causal[3, j]
    print(f"  not → {wj:>4}: {v:.3f}" + (" (bloqueado)" if v == 0 else ""))
```

**Step 2: Correr y capturar**
```bash
cd clase_14/practica && source .venv/bin/activate
python 38_encoder_vs_decoder.py 2>&1 | tee /tmp/cap38_output.txt
```

**Step 3: Escribir Hugo chapter**

Front matter: `title: "38 - Encoder vs Decoder: la diferencia que lo cambia todo"`, `weight: 380`, `math: true`.

Estructura (~1000 palabras):
1. Apertura — "Todos los modelos que construiste hasta ahora eran decoders. El Mini-GPT y Mini-LLaMA solo leen hacia la izquierda. BERT lee en ambas direcciones. Una sola linea de codigo (`mask=None`) cambia el paradigma."
2. La mascara causal — por que existe y que bloquea
3. Atencion bidireccional — que habilita y que imposibilita (generacion)
4. Embed script completo
5. Output literal de `/tmp/cap38_output.txt`
6. Analisis de las matrices mostradas
7. Por que el encoder NO puede generar texto (explicar matematicamente: sin causal mask, la prediccion del token t usa informacion de t+1, t+2 — circular en autoregresion)
8. Preguntas de verificacion (3)

**Step 4: Hugo build + commit**
```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site && hugo --quiet && echo "OK"
git add clase_14/practica/38_encoder_vs_decoder.py \
        site/content/clases/clase-14/practica/38-encoder-vs-decoder.md
git commit -m "feat+docs(cap38): encoder vs decoder — mascara causal vs bidireccional"
```

---

## Task 7: Cap 39 — Learnable Positional Embeddings

**Files:**
- Create: `clase_14/practica/39_positional_embeddings.py`
- Create: `site/content/clases/clase-14/practica/39-positional-embeddings.md`

**Step 1: Script**
```python
"""39_positional_embeddings.py - Cap 39: learned pos emb vs RoPE.

Muestra como se ven los embeddings de posicion aprendidos
y los compara conceptualmente con RoPE del cap 18.
"""
import torch
import torch.nn as nn
from _models import LearnedPositionalEmbedding

torch.manual_seed(42)

d_model = 128
max_seq_len = 128

emb = LearnedPositionalEmbedding(max_seq_len, d_model)

print("=== Learnable Positional Embeddings ===\n")
print(f"Shape del modulo: nn.Embedding({max_seq_len}, {d_model})")
print(f"Params: {max_seq_len * d_model:,} (uno por posicion × dimension)")
n_params = sum(p.numel() for p in emb.parameters())
print(f"Params totales: {n_params:,}\n")

# Mostrar similitud entre embeddings de posiciones cercanas vs lejanas
weights = emb.embedding.weight.detach()  # (128, 128)

def cos_sim(a, b):
    return (a @ b) / (a.norm() * b.norm())

print("Similaridad coseno entre embeddings de posicion (random init):")
print(f"  pos 0 vs pos 1:  {cos_sim(weights[0], weights[1]):.4f}")
print(f"  pos 0 vs pos 64: {cos_sim(weights[0], weights[64]):.4f}")
print(f"  pos 0 vs pos 127:{cos_sim(weights[0], weights[127]):.4f}")
print("\nNOTA: en random init estos valores son ruido — no tienen significado.")
print("El patron posicional (cercanas mas similares) solo emerge DESPUES del MLM training.")
print("Podemos re-correr este script post-training para ver la diferencia.")

print("\n=== Comparacion con RoPE (cap 18) ===")
print("""
RoPE (Rotary Position Embedding):
  - NO agrega nada a los embeddings de token
  - Rota Q y K en el espacio complejo segun la posicion
  - La similitud posicional emerge del producto punto rotado
  - Ventaja: extrapolacion a secuencias mas largas que el training

Learned Positional Embeddings (BERT):
  - SE SUMA un vector aprendido al embedding de token
  - No hay garantia de extrapolacion
  - Ventaja: mas simple, aprendible de forma directa
  - Limitacion: solo funciona hasta max_seq_len del training
""")

print("=== Forward pass ===")
x = torch.zeros(2, 10, d_model)  # secuencia de zeros
out = emb(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"La diferencia output - input = los embeddings de posicion:")
diff = out - x
for pos in [0, 3, 9]:
    print(f"  pos {pos}: norma = {diff[0, pos].norm():.4f}")
```

**Step 2: Correr y capturar**
```bash
python 39_positional_embeddings.py 2>&1 | tee /tmp/cap39_output.txt
```

**Step 3: Hugo chapter**

Front matter: `title: "39 - Positional Embeddings aprendidos: BERT vs RoPE"`, `weight: 390`, `math: true`.

Estructura (~900 palabras):
1. "En el cap 18 viste RoPE — rotaciones geometricas en Q y K. BERT usa algo mas simple: embeddings aprendidos que se SUMAN al token embedding."
2. Por que BERT eligio learnable en 2018 (RoPE no existia, Vaswani 2017 usaba sin/cos fijo, learnable igualaba experimentalmente)
3. La diferencia fundamental: RoPE no agrega parametros al embedding, learnable agrega max_seq_len × d_model params
4. Script completo + output literal
5. La limitacion de extrapolacion (BERT no puede procesar secuencias mas largas que max_seq_len en training)
6. Tabla comparativa RoPE vs Learned vs Sin/Cos fijo
7. Preguntas de verificacion (3)

**Step 4: Hugo build + commit**
```bash
hugo --quiet && echo "OK"
git add clase_14/practica/39_positional_embeddings.py \
        site/content/clases/clase-14/practica/39-positional-embeddings.md
git commit -m "feat+docs(cap39): positional embeddings aprendidos vs RoPE"
```

---

## Task 8: Cap 40 — Special tokens [CLS], [MASK], [SEP]

**Files:**
- Create: `clase_14/practica/40_special_tokens.py`
- Create: `site/content/clases/clase-14/practica/40-special-tokens.md`

**Step 1: Script**
```python
"""40_special_tokens.py - Cap 40: [CLS], [MASK], [SEP] en accion."""
from pathlib import Path
from _bpe import BPETokenizer

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

print("=== Special tokens BERT ===\n")
print(f"[CLS]  id={tok.cls_id}  — Classification token (inicio de secuencia)")
print(f"[SEP]  id={tok.sep_id}  — Separator token (fin de segmento)")
print(f"[MASK] id={tok.mask_id} — Mask token (reemplaza tokens en MLM)")
print(f"\nVocab size antes: 1112  | despues: {tok.vocab_size}")

sentences = [
    "To be or not to be",
    "En un lugar de la Mancha",
]
print("\n=== encode_bert vs encode regular ===\n")
for s in sentences:
    regular = tok.encode(s)
    bert = tok.encode_bert(s)
    print(f"Texto:   {s!r}")
    print(f"Regular: {regular[:5]}... ({len(regular)} tokens)")
    print(f"BERT:    {bert[:5]}... ({len(bert)} tokens)  ← +2 ([CLS] y [SEP])")
    print(f"Decode:  {tok.decode(bert)!r}\n")

print("=== Rol de cada token ===")
print("""
[CLS] — Classification Token:
  Posicion 0 de CADA input BERT.
  El vector de salida de [CLS] despues de pasar por los N bloques
  representa TODA la secuencia. Es este vector el que va a la
  cabeza de clasificacion en fine-tuning. No tiene contenido
  semantico propio — aprende a ser un "resumen" del input.

[SEP] — Separator Token:
  Indica el fin del input (o separacion entre dos frases en BERT original).
  En nuestro caso de una sola frase: marca el fin.

[MASK] — Mask Token:
  Reemplaza tokens durante pretraining MLM.
  El modelo aprende a predecir el token original dado el contexto.
  NUNCA aparece en fine-tuning — es exclusivo del pretraining.
""")
```

**Step 2: Correr y capturar** → `/tmp/cap40_output.txt`

**Step 3: Hugo chapter** + commit

Front matter: `title: "40 - Special tokens: [CLS], [MASK], [SEP]"`, `weight: 400`, `math: true`.

Contenido: qué hace cada token, por qué [CLS] como representación agregada, el truco del [MASK] en MLM, output literal.

```bash
git commit -m "feat+docs(cap40): special tokens [CLS][MASK][SEP] con roles"
```

---

## Task 9: Cap 41 — Arquitectura Mini-BERT completa

**Files:**
- Create: `clase_14/practica/41_mini_bert.py`
- Create: `site/content/clases/clase-14/practica/41-mini-bert.md`

**Step 1: Script**
```python
"""41_mini_bert.py - Cap 41: forward pass completo de Mini-BERT."""
import torch
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniBERT, get_device

torch.manual_seed(42)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
vocab_size = tok.vocab_size  # 1115

cfg = dict(vocab_size=vocab_size, max_seq_len=128,
           d_model=128, n_heads=4, n_layers=4, d_ff=512)
model = MiniBERT(**cfg).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"MiniBERT: {n_params:,} parametros")
print(f"Comparacion: MiniLLaMA tuvo ~1,072,256 params\n")

# Forward pass de ejemplo
sentences = [
    "To be or not to be",
    "En un lugar de la Mancha",
]
print("=== Forward pass ===\n")
for s in sentences:
    ids = torch.tensor([tok.encode_bert(s)], dtype=torch.long, device=device)
    ids = ids[:, :128]  # truncar a max_seq_len
    h = model(ids)
    cls_vec = h[0, 0]  # vector [CLS]
    print(f"Texto:     {s!r}")
    print(f"Tokens:    {ids.shape[1]} (incluyendo [CLS] y [SEP])")
    print(f"h.shape:   {h.shape}  — (batch=1, seq_len, d_model=128)")
    print(f"[CLS] vec: norma={cls_vec.norm().item():.4f}, primeros 4 dims: {cls_vec[:4].tolist()}")
    print()

print("=== Diferencias con Mini-LLaMA ===")
print("""
Mini-LLaMA                Mini-BERT
-----------               ----------
GQA (h_q=4, h_kv=2)      MHA (n_heads=4)
RoPE en Q y K             Learned pos emb (sumado al token emb)
RMSNorm                   LayerNorm
SwiGLU                    GELU
Causal mask               Sin mascara
max_seq_len=256            max_seq_len=128
Genera: next token         Clasifica: [CLS] vector
""")
```

**Step 2: Correr** → `/tmp/cap41_output.txt`

**Step 3: Hugo chapter** + commit

Front matter: `title: "41 - Arquitectura Mini-BERT completa"`, `weight: 410`, `math: true`.

Tabla comparativa Mini-LLaMA vs Mini-BERT + forward pass explicado + output literal.

```bash
git commit -m "feat+docs(cap41): Mini-BERT forward pass y tabla vs Mini-LLaMA"
```

---

## Task 10: Cap 42 — MLM Loss (el objetivo simétrico al SFT)

**Files:**
- Create: `clase_14/practica/42_mlm_loss.py`
- Create: `site/content/clases/clase-14/practica/42-mlm-loss.md`

**Step 1: Script**
```python
"""42_mlm_loss.py - Cap 42: MLM masking + 80/10/10 split."""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniBERT, MLMHead, get_device
from _bert_utils import apply_mlm_mask

torch.manual_seed(42)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
vocab_size = tok.vocab_size

model = MiniBERT(vocab_size=vocab_size, max_seq_len=128,
                 d_model=128, n_heads=4, n_layers=4, d_ff=512).to(device)
mlm_head = MLMHead(d_model=128, vocab_size=vocab_size).to(device)

sentence = "To be or not to be that is the question"
ids = torch.tensor([tok.encode_bert(sentence)], dtype=torch.long)
print(f"Tokens originales ({ids.shape[1]}):")
print(f"  {[tok.id_to_token[i] for i in ids[0].tolist()]}\n")

masked_ids, labels = apply_mlm_mask(ids.clone(), mask_prob=0.15,
                                     mask_id=tok.mask_id, vocab_size=vocab_size)

print("Despues de MLM masking (15%, split 80/10/10):")
for pos, (orig, masked, label) in enumerate(
        zip(ids[0].tolist(), masked_ids[0].tolist(), labels[0].tolist())):
    if label != -100:
        orig_tok   = tok.id_to_token.get(orig, "?")
        masked_tok = tok.id_to_token.get(masked, "?")
        print(f"  pos {pos:2d}: '{orig_tok}' → '{masked_tok}'  (label={label}, predict='{orig_tok}')")

n_masked = (labels != -100).sum().item()
print(f"\nTokens enmascarados: {n_masked}/{ids.shape[1]} = {n_masked/ids.shape[1]:.1%}")

# Calcular la loss MLM
masked_ids_dev = masked_ids.to(device)
labels_dev     = labels.to(device)
h = model(masked_ids_dev)
logits = mlm_head(h)  # (1, T, vocab_size)

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels_dev.view(-1),
    ignore_index=-100  # ignorar posiciones no enmascaradas
)
print(f"\nMLM loss (modelo random): {loss.item():.4f}")
print(f"Esperado ~log({vocab_size}) = {torch.tensor(vocab_size).float().log().item():.4f}")
print("\nNota: la loss MLM usa ignore_index=-100, igual que SFT usaba loss_mask=0.")
print("Son la misma idea: solo backpropagar donde importa.")
```

**Step 2: Correr** → `/tmp/cap42_output.txt`

**Step 3: Hugo chapter** + commit

Front matter: `title: "42 - MLM Loss: el objetivo simetrico al SFT"`, `weight: 420`, `math: true`.

Clave: mostrar la simetria entre SFT (mask=1 en response) y MLM (ignore_index=-100 en no-enmascarados). Son el mismo principio. Output literal.

```bash
git commit -m "feat+docs(cap42): MLM loss — masking 80/10/10, simetria con SFT cap 24"
```

---

## Task 11: Cap 43 — MLM Pretraining (~30s)

**Files:**
- Create: `clase_14/practica/43_train_bert.py`

**Step 1: Script**
```python
"""43_train_bert.py - Cap 43: MLM pretraining de Mini-BERT."""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniBERT, MLMHead, get_device
from _bert_utils import apply_mlm_mask

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
vocab_size = tok.vocab_size

en = Path("shakespeare.txt").read_text(encoding="utf-8")
es = Path("quijote.txt").read_text(encoding="utf-8")
corpus = en + "\n" + es
data = torch.tensor(tok.encode(corpus), dtype=torch.long)
print(f"Corpus: {len(data):,} tokens")

BLOCK = 64   # longitud de secuencia (sin [CLS][SEP] la ventana real es 62)
BATCH = 32
LR    = 1e-4
ITERS = 3000
WD    = 0.01

model    = MiniBERT(vocab_size=vocab_size, max_seq_len=BLOCK+2,
                    d_model=128, n_heads=4, n_layers=4, d_ff=512).to(device)
mlm_head = MLMHead(d_model=128, vocab_size=vocab_size).to(device)

params = list(model.parameters()) + list(mlm_head.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

n_params = sum(p.numel() for p in params)
print(f"Params: {n_params:,}\n")

def get_batch():
    """Muestrea ventanas aleatorias y las formatea como BERT input."""
    ix = torch.randint(0, len(data) - BLOCK, (BATCH,))
    windows = torch.stack([data[i:i+BLOCK] for i in ix])  # (B, 64)
    # Agregar [CLS] al inicio y [SEP] al final
    cls_col = torch.full((BATCH, 1), tok.cls_id, dtype=torch.long)
    sep_col = torch.full((BATCH, 1), tok.sep_id, dtype=torch.long)
    input_ids = torch.cat([cls_col, windows, sep_col], dim=1)  # (B, 66)
    masked_ids, labels = apply_mlm_mask(
        input_ids.clone(), mask_prob=0.15,
        mask_id=tok.mask_id, vocab_size=vocab_size,
        special_ids=(tok.cls_id, tok.sep_id, tok.mask_id)
    )
    return masked_ids.to(device), labels.to(device)

print(f"MLM pretraining: {ITERS} iters\n")
for it in range(ITERS):
    masked_ids, labels = get_batch()
    h      = model(masked_ids)
    logits = mlm_head(h)
    loss   = F.cross_entropy(logits.view(-1, vocab_size),
                              labels.view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 300 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save({
    "model": model.state_dict(),
    "mlm_head": mlm_head.state_dict(),
    "config": dict(vocab_size=vocab_size, max_seq_len=BLOCK+2,
                   d_model=128, n_heads=4, n_layers=4, d_ff=512),
}, "checkpoints/mini_bert_pretrained.pt")
print("\nSaved -> checkpoints/mini_bert_pretrained.pt")
```

**Step 2: Correr** (el checkpoint `.pt` es gitignored)
```bash
python 43_train_bert.py 2>&1 | tee /tmp/cap43_train.txt
```
Expected: **~3-5 minutos** en MPS (el masking Python loop toma ~10-20ms por batch). Loss baja de ~7.0 (log 1115) a <3.0.

NOTA: Si tarda más de 7 minutos, revisar que `apply_mlm_mask` no esté llamándose con B×T muy grande. Con BATCH=32 y BLOCK=64 → 2112 iteraciones Python por batch × 3000 iters = normal.

**Step 3: Commit script + Hugo chapter**

Front matter: `title: "43 - MLM Pretraining"`, `weight: 430`, `math: true`.

Curva de loss + comparacion con pretrain decoder del cap 31. Output literal.

```bash
git add clase_14/practica/43_train_bert.py \
        site/content/clases/clase-14/practica/43-mlm-pretraining.md
git commit -m "feat+docs(cap43): MLM pretraining Mini-BERT"
```

---

## Task 12: Cap 44 — Eval MLM (fill-in-the-blank interactivo)

**Files:**
- Create: `clase_14/practica/44_eval_mlm.py`
- Create: `site/content/clases/clase-14/practica/44-eval-mlm.md`

**Step 1: Script**
```python
"""44_eval_mlm.py - Cap 44: fill-in-the-blank con Mini-BERT pretrained."""
import torch
from _bpe import BPETokenizer
from _models import MiniBERT, MLMHead, get_device

device = get_device()
tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

ckpt = torch.load("checkpoints/mini_bert_pretrained.pt", map_location=device)
cfg  = ckpt["config"]
model    = MiniBERT(**cfg).to(device)
mlm_head = MLMHead(d_model=cfg["d_model"], vocab_size=cfg["vocab_size"]).to(device)
model.load_state_dict(ckpt["model"])
mlm_head.load_state_dict(ckpt["mlm_head"])
model.eval(); mlm_head.eval()

def predict_mask(left: str, right: str, top_k: int = 5):
    """Predice el token entre left y right.

    IMPORTANTE: NO pasar "[MASK]" como texto — el BPE lo tokenizaria como
    chars individuales '[','M','A','S','K',']'. En su lugar, construimos
    manualmente la secuencia: [CLS] + encode(left) + mask_id + encode(right) + [SEP].
    """
    l_ids = tok.encode(left)
    r_ids = tok.encode(right)
    ids = [tok.cls_id] + l_ids + [tok.mask_id] + r_ids + [tok.sep_id]
    mask_pos = 1 + len(l_ids)  # posicion exacta del mask_id

    x = torch.tensor([ids[:cfg["max_seq_len"]]], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(x)
        logits = mlm_head(h)
    probs = torch.softmax(logits[0, mask_pos], dim=-1)
    top_ids = probs.topk(top_k).indices.tolist()
    top_probs = probs.topk(top_k).values.tolist()
    display = f"{left!r} [MASK] {right!r}"
    print(f"Texto: {display}")
    print(f"Top-{top_k} predicciones:")
    for i, (tid, prob) in enumerate(zip(top_ids, top_probs)):
        tok_str = tok.id_to_token.get(tid, "?")
        print(f"  {i+1}. '{tok_str}' ({prob:.3f})")
    print()

print("=== Fill-in-the-blank con Mini-BERT ===\n")
# Cada ejemplo: (left_context, right_context)
examples = [
    ("To ", " or not to be"),
    ("To be or not to ", ""),
    ("En un ", " de la Mancha"),
    ("The ", " is dead"),
    ("No hay mal que por bien no ", ""),
]
for left, right in examples:
    predict_mask(left, right)
```

**Step 2: Correr** → `/tmp/cap44_mlm.txt`

**Step 3: Hugo chapter** + commit

Front matter: `title: "44 - Eval MLM: fill-in-the-blank"`, `weight: 440`, `math: true`.

Es el capítulo más visual — mostrar las predicciones literales para cada ejemplo. Si "To [MASK] or not to be" → "be" está en top-3, el encoder aprendió algo real.

```bash
git commit -m "feat+docs(cap44): eval MLM — fill-in-the-blank bilingue"
```

---

## Task 13: Cap 45 — [CLS] como clasificador

**Files:**
- Create: `clase_14/practica/45_cls_head.py`
- Create: `site/content/clases/clase-14/practica/45-cls-head.md`

**Step 1: Script**
```python
"""45_cls_head.py - Cap 45: [CLS] como vector clasificador."""
import torch
from _bpe import BPETokenizer
from _models import MiniBERT, ClassificationHead, get_device

device = get_device()
tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

ckpt = torch.load("checkpoints/mini_bert_pretrained.pt", map_location=device)
cfg  = ckpt["config"]
model = MiniBERT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# Cabeza de clasificacion: 128 → 2 (EN=0, ES=1)
cls_head = ClassificationHead(d_model=128, n_classes=2).to(device)

print("=== [CLS] como clasificador ===\n")
print(f"ClassificationHead: Linear(128, 2)")
n_params = sum(p.numel() for p in cls_head.parameters())
print(f"Params de la cabeza: {n_params} (minimos!)\n")

examples = [
    ("To be or not to be", "EN", 0),
    ("The king is dead", "EN", 0),
    ("En un lugar de la Mancha", "ES", 1),
    ("No hay mal que por bien no venga", "ES", 1),
]
print("CLS vectors antes de fine-tuning (clasificacion aleatoria):")
print(f"{'Texto':<40} {'Idioma'} {'Logit EN':>10} {'Logit ES':>10}")
for text, lang, _ in examples:
    ids = torch.tensor([tok.encode_bert(text)[:cfg["max_seq_len"]]],
                       dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids)
        logits = cls_head(h)
    print(f"{text:<40} {lang}     {logits[0,0].item():>10.3f}  {logits[0,1].item():>10.3f}")

print("\nLos logits son aleatorios (cabeza no entrenada) — fine-tuning en cap 47.")
print("\n=== Por que [CLS] y no promedio de todos los tokens? ===")
print("""
BERT podria usar promedio de todos los tokens como representacion.
Usar [CLS] es una decision de diseno:
  1. [CLS] es un token sin contenido propio — aprende libremente a ser 'resumen'
  2. Permite arquitecturas de dos-torres (cross-encoder) eficientes
  3. El promedio puede mezclar señales de tokens no relevantes
  4. En practica: ambos funcionan; [CLS] es el estandar BERT
""")
```

**Step 2: Correr** → `/tmp/cap45_cls.txt`

**Step 3: Hugo chapter** + commit

Front matter: `title: "45 - [CLS] como clasificador"`, `weight: 450`, `math: true`.

```bash
git commit -m "feat+docs(cap45): [CLS] vector — cabeza de clasificacion, por que no promedio"
```

---

## Task 14: Cap 46 — Dataset detección de idioma

**Files:**
- Create: `clase_14/practica/46_dataset_lang.py`
- Output versionado: `clase_14/practica/data/lang_train.jsonl`, `data/lang_eval.jsonl`

**Step 1: Script**
```python
"""46_dataset_lang.py - Cap 46: dataset EN/ES para deteccion de idioma."""
import json, random, torch
from pathlib import Path
from _bpe import BPETokenizer

SEED = 246
random.seed(SEED); torch.manual_seed(SEED)

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

WINDOW = 64  # tokens por ejemplo (sin [CLS][SEP])

en_text = Path("shakespeare.txt").read_text(encoding="utf-8")
es_text = Path("quijote.txt").read_text(encoding="utf-8")
en_tokens = tok.encode(en_text)
es_tokens = tok.encode(es_text)

def sample_windows(tokens, n, label):
    rng = random.Random(SEED + label)
    examples = []
    for _ in range(n):
        start = rng.randint(0, len(tokens) - WINDOW - 1)
        window = tokens[start:start + WINDOW]
        full = [tok.cls_id] + window + [tok.sep_id]
        examples.append({"ids": full, "label": label})
    return examples

Path("data").mkdir(exist_ok=True)

for split, n_each, fout in [
    ("train", 1000, "data/lang_train.jsonl"),
    ("eval",   250, "data/lang_eval.jsonl"),
]:
    examples = sample_windows(en_tokens, n_each, 0) + \
               sample_windows(es_tokens, n_each, 1)
    random.shuffle(examples)
    with open(fout, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[{split}] {len(examples)} ejemplos ({n_each} EN + {n_each} ES) → {fout}")

print("\nEjemplos del train set:")
with open("data/lang_train.jsonl") as f:
    for line in list(f)[:2]:
        ex = json.loads(line)
        decoded = tok.decode(ex["ids"])
        lang = "EN" if ex["label"] == 0 else "ES"
        print(f"  [{lang}] {decoded[:60]!r}...")
```

**Step 2: Correr** y commitear datasets
```bash
python 46_dataset_lang.py 2>&1 | tee /tmp/cap46_dataset.txt
git add clase_14/practica/46_dataset_lang.py \
        clase_14/practica/data/lang_train.jsonl \
        clase_14/practica/data/lang_eval.jsonl
git commit -m "feat(cap46): dataset EN/ES 2500 ejemplos para deteccion idioma"
```

**Step 3: Hugo chapter + commit**

Front matter: `title: "46 - Dataset: deteccion de idioma EN/ES"`, `weight: 460`, `math: true`.

```bash
git add site/content/clases/clase-14/practica/46-dataset-lang.md
git commit -m "docs(cap46): dataset EN/ES — ground truth perfecto sin etiquetado"
```

---

## Task 15: Cap 47 — Fine-tuning (~20s)

**Files:**
- Create: `clase_14/practica/47_finetune_bert.py`

**Step 1: Script**
```python
"""47_finetune_bert.py - Cap 47: fine-tuning BERT para deteccion de idioma."""
import json, torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniBERT, ClassificationHead, get_device

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

ckpt = torch.load("checkpoints/mini_bert_pretrained.pt", map_location=device)
cfg  = ckpt["config"]
model    = MiniBERT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.train()

cls_head = ClassificationHead(d_model=128, n_classes=2).to(device)

# Fine-tuning usa lr mucho menor para no destruir lo aprendido en MLM
LR    = 2e-5  # 5x menor que pretraining
ITERS = 500
BATCH = 32
WD    = 0.01

import random as _random
train_data = [json.loads(l) for l in open("data/lang_train.jsonl")]
params = list(model.parameters()) + list(cls_head.parameters())
opt    = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

print(f"Fine-tuning: {ITERS} iters, LR={LR}\n")
for it in range(ITERS):
    batch = _random.sample(train_data, BATCH)  # muestreo aleatorio sin reemplazo
    max_len = max(len(ex["ids"]) for ex in batch)
    ids_t = torch.zeros(BATCH, max_len, dtype=torch.long, device=device)
    lbl_t = torch.zeros(BATCH, dtype=torch.long, device=device)
    for i, ex in enumerate(batch):
        ids_t[i, :len(ex["ids"])] = torch.tensor(ex["ids"])
        lbl_t[i] = ex["label"]

    h      = model(ids_t)
    logits = cls_head(h)
    loss   = F.cross_entropy(logits, lbl_t)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 50 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

torch.save({
    "model":    model.state_dict(),
    "cls_head": cls_head.state_dict(),
    "config":   cfg,
}, "checkpoints/mini_bert_finetuned.pt")
print("\nSaved -> checkpoints/mini_bert_finetuned.pt")
```

**Step 2: Correr** → `/tmp/cap47_finetune.txt`

**Step 3: Script + Hugo chapter + commit**

Front matter: `title: "47 - Fine-tuning: deteccion de idioma"`, `weight: 470`, `math: true`.

Enfatizar: lr=2e-5 (5× menor) porque no queremos destruir el conocimiento MLM. Paralelo con cap 24 (SFT usa lr 10× menor que pretrain).

```bash
git add clase_14/practica/47_finetune_bert.py \
        site/content/clases/clase-14/practica/47-finetune-bert.md
git commit -m "feat+docs(cap47): fine-tuning BERT deteccion idioma, lr=2e-5"
```

---

## Task 16: Cap 48 — Eval + attention patterns + PCA

**Files:**
- Create: `clase_14/practica/48_eval_bert.py`
- Create: `site/content/clases/clase-14/practica/48-eval-bert.md`

**Step 1: Script**
```python
"""48_eval_bert.py - Cap 48: accuracy + attention patterns + PCA [CLS]."""
import json, torch
import torch.nn.functional as F
from _bpe import BPETokenizer
from _models import MiniBERT, ClassificationHead, get_device

device = get_device()
tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

ckpt = torch.load("checkpoints/mini_bert_finetuned.pt", map_location=device)
cfg  = ckpt["config"]
model    = MiniBERT(**cfg).to(device)
cls_head = ClassificationHead(d_model=128, n_classes=2).to(device)
model.load_state_dict(ckpt["model"])
cls_head.load_state_dict(ckpt["cls_head"])
model.eval(); cls_head.eval()

# === Accuracy en eval set ===
eval_data = [json.loads(l) for l in open("data/lang_eval.jsonl")]
correct = 0
for ex in eval_data:
    ids = torch.tensor([ex["ids"]], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids); logits = cls_head(h)
    pred = logits.argmax(dim=-1).item()
    if pred == ex["label"]: correct += 1
acc = correct / len(eval_data)
print(f"Accuracy EN/ES: {acc:.3f} ({correct}/{len(eval_data)})\n")

# === Attention patterns ===
# Registrar atencion del ultimo bloque
attention_weights = {}
def hook_fn(module, input, output):
    # output: (attn_output, attn_weights) de nn.MultiheadAttention
    # attn_weights shape: (B, T, T) — promedio sobre heads (average_attn_weights=True por defecto)
    # Para ver pesos por head usar average_attn_weights=False en la init de MHA
    if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
        attention_weights["last"] = output[1].detach().cpu()

handle = model.blocks[-1].attn.register_forward_hook(hook_fn)

example_en = "To be or not to be that is the question"
ids_en = torch.tensor([tok.encode_bert(example_en)[:cfg["max_seq_len"]]],
                       dtype=torch.long, device=device)
with torch.no_grad():
    h = model(ids_en)
handle.remove()

tokens_list = [tok.id_to_token.get(i, "?") for i in ids_en[0].tolist()]
attn = attention_weights.get("last")
if attn is not None:
    print("Attention pattern ultimo bloque (fila=[CLS], columnas=todos los tokens):")
    cls_attn = attn[0, 0, :].tolist()  # atencion desde [CLS]
    for i, (tok_str, weight) in enumerate(zip(tokens_list, cls_attn)):
        bar = "█" * int(weight * 40)
        print(f"  {i:2d} {tok_str:>8}: {weight:.3f} {bar}")
else:
    print("(attention weights no disponibles — hook no capturo output)")

# === PCA de [CLS] vectors ===
print("\n=== PCA de embeddings [CLS] (EN vs ES) ===")
en_vecs, es_vecs = [], []
for ex in eval_data[:50]:
    ids = torch.tensor([ex["ids"][:cfg["max_seq_len"]]], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids)
    cls_vec = h[0, 0].cpu()
    if ex["label"] == 0: en_vecs.append(cls_vec)
    else:                es_vecs.append(cls_vec)

all_vecs = torch.stack(en_vecs + es_vecs)
mean = all_vecs.mean(0)
centered = all_vecs - mean
U, S, V = torch.pca_lowrank(centered, q=2)
proj = centered @ V  # (N, 2)
n_en = len(en_vecs)
en_proj = proj[:n_en]; es_proj = proj[n_en:]
print(f"EN centroide: ({en_proj[:, 0].mean():.2f}, {en_proj[:, 1].mean():.2f})")
print(f"ES centroide: ({es_proj[:, 0].mean():.2f}, {es_proj[:, 1].mean():.2f})")
dist = ((en_proj.mean(0) - es_proj.mean(0)).norm()).item()
print(f"Distancia entre centroides: {dist:.3f}")
print("(>2.0 = separacion clara, <1.0 = mezclados)")
```

**Step 2: Correr** → `/tmp/cap48_eval.txt`

**Step 3: Hugo chapter** + commit

Front matter: `title: "48 - Eval: accuracy + attention + PCA [CLS]"`, `weight: 480`, `math: true`.

Tres resultados literales: accuracy, attention pattern (con barras ASCII), PCA centroides. El más visual del camino.

```bash
git add clase_14/practica/48_eval_bert.py \
        site/content/clases/clase-14/practica/48-eval-bert.md
git commit -m "feat+docs(cap48): eval BERT — accuracy + attention pattern + PCA [CLS]"
```

---

## Task 17: Cap 49 — Comparativa BERT vs GPT (solo Hugo)

**File:**
- Create: `site/content/clases/clase-14/practica/49-comparativa-bert-gpt.md`

**Step 1: Escribir capítulo**

Front matter: `title: "49 - Comparativa final: BERT vs GPT — cierre Camino 4"`, `weight: 490`, `math: true`.

Contenido (~1200 palabras, sin script):
1. La tabla tripartita Mini-GPT / Mini-LLaMA / Mini-BERT (todos los parámetros de diseño)
2. Cuándo usar encoder-only vs decoder-only vs encoder-decoder
3. La historia: BERT dominó NLP 2019-2022, decoders escalaron desde 2022, encoders sobreviven hoy en embeddings y re-ranking
4. Sentence-Transformers como aplicación real del encoder
5. Cross-encoders en RAG: la segunda vida de BERT
6. Preguntas finales del Camino 4 (3)
7. Links a Caminos pendientes (5: ViT)

**Step 2: Hugo build + commit**
```bash
git add site/content/clases/clase-14/practica/49-comparativa-bert-gpt.md
git commit -m "docs(cap49): comparativa BERT vs GPT — cierre Camino 4"
```

---

## Task 18: Hub _index.md + glosario + memoria + verificación final

**Files:**
- Modify: `site/content/clases/clase-14/practica/_index.md`
- Create: `site/content/fundamentos/bert.md` (~1500 palabras)
- Modify: memory file

**Step 1: Actualizar hub**

Agregar sección "Camino 4 — Mini-BERT (Encoder-only)" con cards caps 38-49 entre el cierre de Camino 2.5 y "Que viene después".

**Step 2: Glosario `bert.md`**

Entrada profunda (~1500 palabras) que cubra: MLM, [CLS], bidireccionalidad, fine-tuning paradigm, BERT vs GPT, aplicaciones actuales. Con links a caps 38-49.

**Step 3: Verificación final**
```bash
cd clase_14/practica && python -m pytest tests/ -v
```
Expected: ≥20 tests PASS (11 originales + ≥2 BPE specials + ≥7 BERT).

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site && hugo --quiet && echo "OK"
```

**Step 4: Commit final**
```bash
git add site/content/clases/clase-14/practica/_index.md \
        site/content/fundamentos/bert.md
git commit -m "docs(hub+glosario): Camino 4 en hub, bert.md — cierre Camino 4"
```

---

## Resumen de outputs producidos

- 4 clases Python nuevas en `_models.py`: `LearnedPositionalEmbedding`, `BERTBlock`, `MiniBERT`, `MLMHead`, `ClassificationHead`
- 2 métodos nuevos en `_bpe.py`: `add_special_tokens()`, `encode_bert()`
- 1 módulo nuevo: `_bert_utils.py` (apply_mlm_mask)
- ≥7 tests TDD en `tests/test_bert.py`
- 11 scripts ejecutables: `38_*.py` … `48_*.py`
- 12 capítulos Hugo: `38-encoder-vs-decoder.md` … `49-comparativa-bert-gpt.md`
- 2 datasets versionados: `data/lang_train.jsonl` (2000), `data/lang_eval.jsonl` (500)
- 1 entrada glosario: `site/content/fundamentos/bert.md`
- 3 checkpoints gitignored: `mini_bert_pretrained.pt`, `mini_bert_finetuned.pt`
- ~18 commits en branch `feat/clase-14-camino-4-bert`
