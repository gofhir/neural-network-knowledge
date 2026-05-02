# Clase 14 — Camino 2.5 (BPE + SFT + DPO) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Agregar Camino 2.5 (caps 30-37) que resuelve las limitaciones de char-level del Camino 2 implementando BPE tokenización desde cero (~1000 merges), reentrenando Mini-LLaMA con el nuevo vocab, y repitiendo SFT + DPO sobre 4 tareas BPE-naturales (qa, repeat, complete-en, complete-es), con beta sweep en DPO para validar la hipótesis del cap 29.

**Architecture:** `_bpe.py` implementa BPETokenizer + CharTokenizer wrapper. `_models.py` y `_eval.py` se refactorizan para ser tokenizer-agnostic (reciben un objeto tokenizer con `.encode()/.decode()`). Scripts Camino 2 (14-21) se actualizan con 1 línea por script (wrapping char maps en CharTokenizer). Scripts nuevos (30-37) usan BPETokenizer directamente. MiniLLaMA reutilizada con vocab=1000 en vez de 65.

**Tech Stack:** Python 3.12 + PyTorch (MPS) + Hugo. Venv existente en `clase_14/practica/.venv/`. Sin dependencias nuevas.

**Design doc:** `docs/plans/2026-05-02-clase-14-camino-2-5-bpe-design.md`

**Branch:** `feat/clase-14-camino-2.5-bpe` desde HEAD de `feat/clase-14-camino-2-sft-dpo`.

**Working dir:** `/Users/robertoaraneda/projects/personal/courses/ia-uc`

**Verification model (igual que Camino 2):**
- `_bpe.py` helpers → pytest tests reales (TDD)
- Scripts ejecutables → corren sin error + output en rango esperado
- Capítulos Hugo → output literal del script en el cap, hugo build limpio

---

## Task 0: Branch setup

**Files:**
- New branch: `feat/clase-14-camino-2.5-bpe`

**Step 1: Crear rama desde HEAD actual**

```bash
git checkout feat/clase-14-camino-2-sft-dpo
git checkout -b feat/clase-14-camino-2.5-bpe
```

**Step 2: Verificar venv funciona**

```bash
cd clase_14/practica
source .venv/bin/activate
python -c "import torch; print(torch.backends.mps.is_available())"
```
Expected: `True`

**Step 3: Verificar que todos los tests de Camino 2 siguen pasando (baseline)**

```bash
python -m pytest tests/ -v
```
Expected: 6/6 PASS. Si falla alguno, investigar antes de continuar.

**Step 4: Commit**

```bash
git commit --allow-empty -m "chore: branch Camino 2.5 — BPE tokenization desde cero"
```

---

## Task 1: BPETokenizer.train() — TDD

**Files:**
- Create: `clase_14/practica/_bpe.py`
- Create: `clase_14/practica/tests/test_bpe.py`

**Step 1: Escribir test failing**

```python
# tests/test_bpe.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bpe import BPETokenizer

def test_bpe_train_reduces_token_count():
    """Entrenar con merges reduce tokens vs char-level."""
    corpus = "aaabdaaabac"  # pequeño, controlable
    tok = BPETokenizer()
    tok.train(corpus, num_merges=3)
    # Luego de merges sobre este corpus, vocab debe tener > len(set(corpus)) tokens
    assert len(tok.vocab) > len(set(corpus))
```

**Step 2: Run, expect FAIL**

```bash
python -m pytest tests/test_bpe.py::test_bpe_train_reduces_token_count -v
```
Expected: `FAIL — ModuleNotFoundError: No module named '_bpe'`

**Step 3: Implementar `_bpe.py` con `BPETokenizer.train()`**

```python
"""_bpe.py — BPE tokenizer desde cero y CharTokenizer wrapper."""
from __future__ import annotations
import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, corpus: str, num_merges: int) -> None:
        """Entrena BPE sobre corpus. Usa subconjunto de 50k chars para velocidad."""
        # Usar primeros 50k chars — suficiente para aprender ~1000 merges pedagogicamente
        text = corpus[:50_000]

        # Vocab inicial: todos los chars únicos del corpus COMPLETO (no truncado)
        for c in sorted(set(corpus)):
            if c not in self.vocab:
                idx = len(self.vocab)
                self.vocab[c] = idx
                self.id_to_token[idx] = c

        # Tokenizar el texto truncado como lista de chars
        tokens = list(text)

        for _ in range(num_merges):
            # Contar pares consecutivos
            counts = Counter()
            for i in range(len(tokens) - 1):
                counts[(tokens[i], tokens[i + 1])] += 1
            if not counts:
                break

            # Par más frecuente
            a, b = max(counts, key=counts.get)
            new_token = a + b

            # Registrar merge
            self.merges.append((a, b))

            # Agregar al vocab
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.id_to_token[idx] = new_token

            # Aplicar merge al corpus tokenizado
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("implement in Task 2")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("implement in Task 2")

    def save(self, path: str) -> None:
        raise NotImplementedError("implement in Task 3")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        raise NotImplementedError("implement in Task 3")
```

**Step 4: Run test, expect PASS**

```bash
python -m pytest tests/test_bpe.py::test_bpe_train_reduces_token_count -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add clase_14/practica/_bpe.py clase_14/practica/tests/test_bpe.py
git commit -m "feat(_bpe): BPETokenizer.train() con TDD"
```

---

## Task 2: BPETokenizer.encode() y decode()

**Files:**
- Modify: `clase_14/practica/_bpe.py`
- Modify: `clase_14/practica/tests/test_bpe.py`

**Step 1: Agregar tests failing**

```python
def test_bpe_round_trip():
    """encode → decode reproduce el texto original."""
    corpus = open("shakespeare.txt").read()
    tok = BPETokenizer()
    tok.train(corpus, num_merges=100)
    sample = "To be or not to be"
    ids = tok.encode(sample)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == sample

def test_bpe_encode_shorter_than_chars():
    """Con merges suficientes, encode produce menos tokens que chars."""
    corpus = open("shakespeare.txt").read()
    tok = BPETokenizer()
    tok.train(corpus, num_merges=500)
    sample = "the king is dead"
    chars = list(sample)
    ids = tok.encode(sample)
    # Con 500 merges sobre Shakespeare, "the " es un solo merge probable
    assert len(ids) <= len(chars)
```

**Nota:** Los tests requieren estar en `clase_14/practica/`. Correr desde ahí.

**Step 2: Run, expect FAIL** (`encode` lanza `NotImplementedError`)

```bash
python -m pytest tests/test_bpe.py::test_bpe_round_trip tests/test_bpe.py::test_bpe_encode_shorter_than_chars -v
```

**Step 3: Implementar encode y decode**

```python
def encode(self, text: str) -> list[int]:
    """Aplicar merges aprendidos en orden para tokenizar el texto."""
    # Empezar con chars — solo incluir los que están en vocab
    tokens = [c for c in text if c in self.vocab]

    # Aplicar cada merge en orden
    for a, b in self.merges:
        new_token = a + b
        if new_token not in self.vocab:
            continue
        new_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return [self.vocab[t] for t in tokens if t in self.vocab]

def decode(self, ids: list[int]) -> str:
    return "".join(self.id_to_token.get(i, "") for i in ids)
```

**Step 4: Run tests, expect PASS**

```bash
python -m pytest tests/test_bpe.py -v
```
Expected: 3/3 PASS.

**Step 5: Commit**

```bash
git add clase_14/practica/_bpe.py clase_14/practica/tests/test_bpe.py
git commit -m "feat(_bpe): BPETokenizer.encode() decode() con round-trip test"
```

---

## Task 3: BPETokenizer.save() / load() y CharTokenizer

**Files:**
- Modify: `clase_14/practica/_bpe.py`
- Modify: `clase_14/practica/tests/test_bpe.py`

**Step 1: Agregar tests**

```python
def test_bpe_save_load(tmp_path):
    """Guardar y cargar preserva encode/decode idéntico."""
    corpus = open("shakespeare.txt").read()
    tok = BPETokenizer()
    tok.train(corpus, num_merges=50)
    path = str(tmp_path / "tok.json")
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    sample = "hamlet"
    assert tok.encode(sample) == tok2.encode(sample)
    assert tok.decode(tok.encode(sample)) == tok2.decode(tok2.encode(sample))

def test_char_tokenizer_compat():
    """CharTokenizer produce mismo encode que dict directo."""
    from _eval import build_char_maps
    text = open("shakespeare.txt").read()
    c2i, i2c = build_char_maps(text)
    from _bpe import CharTokenizer
    tok = CharTokenizer(c2i, i2c)
    sample = "hello world"
    expected = [c2i[c] for c in sample if c in c2i]
    assert tok.encode(sample) == expected
    assert tok.decode(expected) == sample
```

**Step 2: Run, expect FAIL**

```bash
python -m pytest tests/test_bpe.py::test_bpe_save_load tests/test_bpe.py::test_char_tokenizer_compat -v
```

**Step 3: Implementar save/load y CharTokenizer**

```python
def save(self, path: str) -> None:
    data = {
        "vocab": self.vocab,
        "merges": self.merges,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

@classmethod
def load(cls, path: str) -> "BPETokenizer":
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tok = cls()
    tok.vocab = data["vocab"]
    tok.id_to_token = {int(i): t for t, i in tok.vocab.items()}
    tok.merges = [tuple(m) for m in data["merges"]]
    return tok


class CharTokenizer:
    """Wrapper char-level que expone la interfaz BPETokenizer.
    Permite que los scripts de Camino 2 funcionen sin cambios de lógica.
    """
    def __init__(self, char_to_id: dict, id_to_char: dict):
        self._c2i = char_to_id
        self.id_to_token = id_to_char
        self.vocab_size = len(char_to_id)

    def encode(self, text: str) -> list[int]:
        return [self._c2i[c] for c in text if c in self._c2i]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token.get(i, "") for i in ids)
```

**Step 4: Run ALL tests**

```bash
python -m pytest tests/test_bpe.py -v
```
Expected: 5/5 PASS.

**Step 5: Commit**

```bash
git add clase_14/practica/_bpe.py clase_14/practica/tests/test_bpe.py
git commit -m "feat(_bpe): save/load + CharTokenizer wrapper con tests"
```

---

## Task 4: Refactor `_models.py` generate_with_prompt

**Files:**
- Modify: `clase_14/practica/_models.py`
- Modify: `clase_14/practica/tests/test_models_helpers.py`

**Step 1: Leer `_models.py` para localizar `generate_with_prompt` (línea ~410)**

Confirmar la firma actual:
```python
def generate_with_prompt(model, prompt, char_to_id, id_to_char, max_new_tokens=50,
                        temperature=1.0, top_k=None, device=None, stop_token="\n"):
```

**Step 2: Actualizar el test existente para que use la nueva firma con tokenizer**

Buscar `test_generate_with_prompt_returns_string` en `tests/test_models_helpers.py`.
Actualizar para pasar un `CharTokenizer` en vez de los dicts:

```python
def test_generate_with_prompt_returns_string():
    from _bpe import CharTokenizer
    cfg = dict(vocab_size=65, max_seq_len=64, d_model=64, h_q=2, h_kv=1, n_layers=2, d_ff=128)
    m = MiniLLaMA(**cfg)
    chars = list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ\n.,'?!:;-0123456789")
    chars = chars[:65]
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    tokenizer = CharTokenizer(c2i, i2c)
    out = generate_with_prompt(m, "abc", tokenizer, max_new_tokens=5, temperature=1.0, top_k=10, device="cpu")
    assert isinstance(out, str)
    assert len(out) >= 3
```

**Step 3: Run test, expect FAIL** (firma incompatible)

```bash
python -m pytest tests/test_models_helpers.py::test_generate_with_prompt_returns_string -v
```

**Step 4: Refactorizar `generate_with_prompt` en `_models.py`**

Reemplazar la implementación actual por:

```python
@torch.no_grad()
def generate_with_prompt(model, prompt, tokenizer, max_new_tokens=50,
                        temperature=1.0, top_k=None, device=None, stop_token="\n"):
    """Genera texto condicionado en prompt. tokenizer debe tener .encode() y .decode()."""
    if device is None:
        device = get_device()
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    stop_ids = tokenizer.encode(stop_token) if stop_token else []
    stop_id = stop_ids[0] if len(stop_ids) == 1 else None  # solo si \n es 1 token
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.max_seq_len:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if stop_id is not None and next_id.item() == stop_id:
            break
    return tokenizer.decode(x[0].tolist())
```

**Step 5: Run ALL tests**

```bash
python -m pytest tests/ -v
```
Expected: 11/11 PASS (6 originales + 5 BPE).

Si `test_generate_with_prompt_returns_string` pasa pero algún otro falla, investigar.

**Step 6: Commit**

```bash
git add clase_14/practica/_models.py clase_14/practica/tests/test_models_helpers.py
git commit -m "refactor(_models): generate_with_prompt tokenizer-agnostic"
```

---

## Task 5: Refactor `_eval.py` y actualizar scripts Camino 2

**Files:**
- Modify: `clase_14/practica/_eval.py`
- Modify: `clase_14/practica/14_show_base_no_instructions.py`
- Modify: `clase_14/practica/16_train_sft.py`
- Modify: `clase_14/practica/17_eval_sft.py`
- Modify: `clase_14/practica/19_dpo_loss_derivation.py`
- Modify: `clase_14/practica/21_train_dpo.py`

**Step 1: Refactorizar `_eval.py`**

Cambiar firmas de `eval_exact_match`, `eval_qualitative`, `eval_drift` para recibir `tokenizer` en vez de `char_to_id, id_to_char`:

```python
def eval_exact_match(model, dataset_jsonl, tokenizer,
                     n_per_task=200, max_new_tokens=20, device=None, temperature=0.1):
    examples = load_jsonl(dataset_jsonl)
    by_task = {}
    for ex in examples:
        by_task.setdefault(ex["task"], []).append(ex)
    results = {}
    for task, items in by_task.items():
        sample = items[:n_per_task]
        correct = 0
        for ex in sample:
            full = generate_with_prompt(
                model, ex["prompt"], tokenizer,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=10, device=device, stop_token="\n",
            )
            generated = full[len(ex["prompt"]):].rstrip("\n")
            expected = ex["response"].rstrip("\n")
            if generated == expected:
                correct += 1
        results[task] = correct / len(sample)
    return results

def eval_qualitative(model, prompts, tokenizer,
                     n_samples=3, temperature=0.8, device=None):
    out = {}
    for p in prompts:
        out[p] = [
            generate_with_prompt(model, p, tokenizer,
                                 max_new_tokens=30, temperature=temperature,
                                 top_k=10, device=device, stop_token="\n")
            for _ in range(n_samples)
        ]
    return out

def eval_drift(model, ambiguous_prompts, tokenizer, device=None):
    shakespeare_markers = ["thou", "thee", "thy", "hath", "doth", "ye", "O ", "wilt"]
    drift_count = 0
    total = 0
    for p in ambiguous_prompts:
        for _ in range(5):
            s = generate_with_prompt(model, p, tokenizer,
                                     max_new_tokens=40, temperature=0.8,
                                     top_k=20, device=device, stop_token="\n")
            comp = s[len(p):].lower()
            if any(m.lower() in comp for m in shakespeare_markers):
                drift_count += 1
            total += 1
    return drift_count / total if total else 0.0
```

**Step 2: Actualizar scripts Camino 2 (1 bloque por script)**

En cada script, reemplazar el patrón:
```python
# ANTES
out = generate_with_prompt(model, p, c2i, i2c, ...)
em = eval_exact_match(model, ..., c2i, i2c, ...)
```

Por:
```python
# DESPUES (agregar cerca del top donde se definen c2i, i2c)
from _bpe import CharTokenizer
tokenizer = CharTokenizer(c2i, i2c)
# luego pasar tokenizer en vez de c2i, i2c
out = generate_with_prompt(model, p, tokenizer, ...)
em = eval_exact_match(model, ..., tokenizer, ...)
```

Scripts a actualizar: `14_show_base_no_instructions.py`, `16_train_sft.py`, `17_eval_sft.py`, `19_dpo_loss_derivation.py`, `21_train_dpo.py`.

**Step 3: Verificar que scripts Camino 2 siguen corriendo**

```bash
python 14_show_base_no_instructions.py 2>&1 | head -5
```
Expected: corre sin error (no hace falta ver output completo, solo que no explota).

**Step 4: Run ALL tests**

```bash
python -m pytest tests/ -v
```
Expected: 11/11 PASS. Si `test_build_char_maps_shakespeare` u otro falla, el refactor de `_eval.py` rompió algo — investigar.

**Step 5: Commit**

```bash
git add clase_14/practica/_eval.py \
        clase_14/practica/14_show_base_no_instructions.py \
        clase_14/practica/16_train_sft.py \
        clase_14/practica/17_eval_sft.py \
        clase_14/practica/19_dpo_loss_derivation.py \
        clase_14/practica/21_train_dpo.py
git commit -m "refactor(_eval,scripts): tokenizer-agnostic + CharTokenizer en scripts Camino 2"
```

---

## Task 6: Cap 30 — script `30_build_bpe.py` y dataset BPE

**Files:**
- Create: `clase_14/practica/30_build_bpe.py`
- Output (versionado): `clase_14/practica/data/bpe_tokenizer.json`

**Step 1: Escribir el script**

```python
"""30_build_bpe.py - Cap 30: BPE desde cero.

Entrena un BPETokenizer sobre Shakespeare + Quijote (~1MB bilingue).
Produce data/bpe_tokenizer.json con vocab ~1100 tokens.
"""
from pathlib import Path
from _bpe import BPETokenizer

NUM_MERGES = 1000

print("Cargando corpus bilingue (Shakespeare + Quijote)...")
en = Path("shakespeare.txt").read_text(encoding="utf-8")
es = Path("quijote.txt").read_text(encoding="utf-8")
corpus = en + "\n" + es
print(f"Corpus: {len(corpus):,} chars total (usando primeros 50,000 para entrenamiento)")

tok = BPETokenizer()
print(f"\nEntrenando BPE con {NUM_MERGES} merges...")
tok.train(corpus, num_merges=NUM_MERGES)

print(f"\nVocab size: {tok.vocab_size} tokens")
print(f"Merges aprendidos: {len(tok.merges)}")

# Verificar que \n es un token propio (importante para stop_token)
newline_id = tok.vocab.get("\n")
print(f"\nToken '\\n' en vocab: id={newline_id} {'OK' if newline_id is not None else 'PROBLEMA'}")

# Ejemplos de encoding
examples = [
    "the king is dead",
    "To be or not to be",
    "En un lugar de la Mancha",
    "quien escribio Don Quijote",
]
print("\n=== Ejemplos de tokenizacion ===")
for ex in examples:
    ids = tok.encode(ex)
    tokens = [tok.id_to_token[i] for i in ids]
    print(f"  '{ex}'")
    print(f"    chars: {len(ex)}  tokens: {len(ids)}  ratio: {len(ids)/len(ex):.2f}")
    print(f"    tokens: {tokens}")

# Guardar
Path("data").mkdir(exist_ok=True)
tok.save("data/bpe_tokenizer.json")
print(f"\nSaved -> data/bpe_tokenizer.json")

# Verificar round-trip
tok2 = BPETokenizer.load("data/bpe_tokenizer.json")
sample = "To be or not to be"
assert tok.encode(sample) == tok2.encode(sample), "round-trip fallo"
print("Round-trip verificado.")
```

**Step 2: Correr y capturar output**

```bash
python 30_build_bpe.py 2>&1 | tee /tmp/cap30_output.txt
```

Expected (~30-60s por los 1000 merges):
```
Corpus: ~1,000,000 chars (usando primeros 50,000)
Vocab size: ~1100 tokens
Token '\n' en vocab: id=X OK
=== Ejemplos de tokenizacion ===
  'the king is dead'
    chars: 18  tokens: ~8  ratio: ~0.44
    tokens: ['the', ' king', ' is', ' dead'] (aproximado)
```

Si el vocab size está muy lejos de 1100 o el token `\n` no está, reportar y ajustar.

**Step 3: Verificar el JSON**

```bash
python -c "
import json
d = json.load(open('data/bpe_tokenizer.json'))
print(f'vocab_size={len(d[\"vocab\"])}, merges={len(d[\"merges\"])}')
"
```

**Step 4: Commit**

```bash
git add clase_14/practica/30_build_bpe.py clase_14/practica/data/bpe_tokenizer.json
git commit -m "feat(cap30): BPE tokenizer 1000 merges sobre corpus bilingue"
```

---

## Task 7: Cap 30 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/30-bpe-desde-cero.md`

**Step 1: Escribir capítulo**

Front matter:
```yaml
---
title: "30 - BPE desde cero: el algoritmo que tokeniza GPT"
weight: 300
math: true
---
```

Estructura:
1. **Apertura** — por qué char-level fallaba (referencia honesta al cap 29 + experimento del cap 37 que viene)
2. **El problema de vocabulario** — trade-off chars (vocab=65, tokens largas) vs palabras (vocab enorme, OOV masivo) vs subwords (BPE: lo mejor de ambos)
3. **El algoritmo BPE** — pseudocódigo + explicación de merge frequencies
4. **El script** — `30_build_bpe.py` completo
5. **Output literal** — copiar de `/tmp/cap30_output.txt` EXACTAMENTE
6. **Lectura del output** — qué tokens emergieron, ratio chars/tokens, por qué `the` es 1 token
7. **Verificación que `\n` es token propio** — crítico para que el stop token funcione
8. **Preguntas de verificación** (3)

**Step 2: Hugo build verify**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site && hugo --quiet && echo "OK"
```

**Step 3: Commit**

```bash
git add site/content/clases/clase-14/practica/30-bpe-desde-cero.md
git commit -m "docs(cap30): BPE desde cero — algoritmo, ejemplos, ratio tokens"
```

---

## Task 8: Cap 31 — script `31_pretrain_bpe.py`

**Files:**
- Create: `clase_14/practica/31_pretrain_bpe.py`
- Output (gitignored): `checkpoints/mini_llama_bpe_base.pt`

**Step 1: Escribir el script**

```python
"""31_pretrain_bpe.py - Cap 31: pretrain Mini-LLaMA con vocab BPE.

Carga el BPETokenizer (1100 tokens), entrena Mini-LLaMA sobre
Shakespeare+Quijote tokenizado, guarda mini_llama_bpe_base.pt.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniLLaMA, get_device

torch.manual_seed(1337)
device = get_device()

# Cargar tokenizer BPE
print("Cargando BPETokenizer...")
tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab_size = tok.vocab_size
print(f"vocab_size={vocab_size}")

# Corpus bilingue tokenizado
print("Tokenizando corpus...")
en = Path("shakespeare.txt").read_text(encoding="utf-8")
es = Path("quijote.txt").read_text(encoding="utf-8")
corpus = en + "\n" + es
data = torch.tensor(tok.encode(corpus), dtype=torch.long)
print(f"Tokens totales: {len(data):,}")

# Hyperparams (igual que char-level salvo vocab_size)
BLOCK = 256
BATCH = 32
LR = 3e-4
ITERS = 3000
WD = 0.01

model = MiniLLaMA(vocab_size=vocab_size, max_seq_len=BLOCK,
                  d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)
model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

def get_batch():
    ix = torch.randint(0, len(data) - BLOCK, (BATCH,))
    x = torch.stack([data[i:i+BLOCK] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+BLOCK+1] for i in ix]).to(device)
    return x, y

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

print(f"\nPretrain BPE: {ITERS} iters\n")
for it in range(ITERS):
    x, y = get_batch()
    logits, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 300 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "checkpoints/mini_llama_bpe_base.pt")
print("\nSaved -> checkpoints/mini_llama_bpe_base.pt")

# Sample de generacion
from _models import generate_with_prompt
print("\n=== Sample generacion BPE-base ===")
for prompt in ["To be or not", "En un lugar"]:
    out = generate_with_prompt(model, prompt, tok, max_new_tokens=30,
                               temperature=0.8, top_k=10, device=device,
                               stop_token=None)
    print(f"Prompt: {prompt!r}")
    print(f"Output: {out!r}\n")
```

**Step 2: Correr y capturar**

```bash
python 31_pretrain_bpe.py 2>&1 | tee /tmp/cap31_pretrain.txt
```
Expected: ~30s en MPS. Loss baja de ~7 a <2. Sample genera texto coherente (mezcla EN/ES).

**Step 3: Comparativa perplexity vs char-level (opcional, si queda tiempo)**

```bash
python -c "
import torch, math
from _models import load_pretrained_mini_llama
from _bpe import BPETokenizer, CharTokenizer
from _eval import build_char_maps
from pathlib import Path

# BPE perplexity
tok_bpe = BPETokenizer.load('data/bpe_tokenizer.json')
m_bpe = load_pretrained_mini_llama('checkpoints/mini_llama_bpe_base.pt',
    config=dict(vocab_size=tok_bpe.vocab_size, max_seq_len=256,
                d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))
# [calcular loss sobre holdout] -- simplificado para el plan
print('BPE base loaded OK')

# Char-level perplexity
text = Path('shakespeare.txt').read_text()
c2i, i2c = build_char_maps(text)
m_char = load_pretrained_mini_llama('checkpoints/mini_llama_base.pt')
print('Char base loaded OK')
"
```

**Step 4: Commit**

```bash
git add clase_14/practica/31_pretrain_bpe.py
git commit -m "feat(cap31): pretrain Mini-LLaMA BPE vocab=1100 sobre corpus bilingue"
```

---

## Task 9: Cap 31 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/31-pretrain-bpe.md`

**Step 1: Escribir capítulo**

Front matter: `title: "31 - Pretrain con BPE: nuevo base model"`, `weight: 310`, `math: true`.

Estructura:
1. Diferencia vs pretrain char-level (vocab 65→1100, embedding 8k→140k params)
2. Corpus bilingüe — por qué EN+ES (habilita complete-en y complete-es en SFT)
3. Script completo embebido
4. Output literal de `/tmp/cap31_pretrain.txt`
5. Sample de generación: BPE genera frases más coherentes que char-level (por qué: cada token es una palabra real, modelo aprende contextos de palabras no de chars)
6. Preguntas de verificación

**Step 2: Hugo build + commit**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site && hugo --quiet && echo "OK"
git add site/content/clases/clase-14/practica/31-pretrain-bpe.md
git commit -m "docs(cap31): pretrain BPE bilingue — vocabs, corpus, sample generacion"
```

---

## Task 10: Cap 32 — demo refactor tokenizer-agnostic

**Files:**
- Create: `clase_14/practica/32_tokenizer_refactor_demo.py`

**Step 1: Escribir el script**

```python
"""32_tokenizer_refactor_demo.py - Cap 32: demo que refactor no rompio nada.

Muestra que el mismo generate_with_prompt funciona con CharTokenizer (Camino 2)
y con BPETokenizer (Camino 2.5) — misma funcion, distintos tokenizers.
"""
from _models import load_pretrained_mini_llama, generate_with_prompt, get_device
from _eval import build_char_maps
from _bpe import BPETokenizer, CharTokenizer

device = get_device()
prompt = "INSTR: repeat 'a' three\nRESP: "

print("=== Char-level (Camino 2) ===")
text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)
char_tok = CharTokenizer(c2i, i2c)
model_char = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
out_char = generate_with_prompt(model_char, prompt, char_tok,
                                max_new_tokens=10, temperature=0.1, top_k=5, device=device)
print(f"Prompt: {prompt!r}")
print(f"Output: {out_char[len(prompt):]!r}")
print(f"Tokenizer: CharTokenizer (vocab={char_tok.vocab_size})")

print("\n=== BPE-level (Camino 2.5) ===")
bpe_tok = BPETokenizer.load("data/bpe_tokenizer.json")
from _models import MiniLLaMA
model_bpe = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_base.pt",
    config=dict(vocab_size=bpe_tok.vocab_size, max_seq_len=256,
                d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))
out_bpe = generate_with_prompt(model_bpe, prompt, bpe_tok,
                               max_new_tokens=10, temperature=0.8, top_k=10, device=device)
print(f"Prompt: {prompt!r}")
print(f"Output: {out_bpe[len(prompt):]!r}")
print(f"Tokenizer: BPETokenizer (vocab={bpe_tok.vocab_size})")
print("\nMisma funcion generate_with_prompt, distintos tokenizers. Refactor OK.")
```

**Step 2: Correr y capturar**

```bash
python 32_tokenizer_refactor_demo.py 2>&1 | tee /tmp/cap32_demo.txt
```

**Step 3: Commit + Hugo chapter**

```bash
git add clase_14/practica/32_tokenizer_refactor_demo.py
git commit -m "feat(cap32): demo tokenizer-agnostic refactor"
```

Hugo chapter `32-refactor-tokenizer.md` (weight: 320) — estructura breve (es el cap más corto): interfaz tokenizer, CharTokenizer wrapper, demo output, tests que validan compatibilidad. ~600-800 palabras.

```bash
git add site/content/clases/clase-14/practica/32-refactor-tokenizer.md
git commit -m "docs(cap32): refactor tokenizer-agnostic, CharTokenizer wrapper"
```

---

## Task 11: Cap 33 — script `33_build_sft_bpe.py`

**Files:**
- Create: `clase_14/practica/33_build_sft_bpe.py`
- Output (versionado): `clase_14/practica/data/sft_bpe_dataset.jsonl`, `data/sft_bpe_eval.jsonl`

**Step 1: Escribir el script**

```python
"""33_build_sft_bpe.py - Cap 33: dataset SFT-BPE 4 tareas.

Tareas: qa (bilingue), repeat (word-form), complete-en, complete-es.
Genera 5000 pares (4000 train + 1000 eval) en data/sft_bpe_*.jsonl.
"""
import json, random
from pathlib import Path
from _bpe import BPETokenizer

SFT_BPE_SEED = 142
EVAL_BPE_SEED = 1242

tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab = set(tok.vocab.keys())
en_text = Path("shakespeare.txt").read_text(encoding="utf-8")
es_text = Path("quijote.txt").read_text(encoding="utf-8")

LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
NUM_WORDS = {2: "two", 3: "three", 4: "four"}

def gen_qa(rng):
    qa_facts = [
        # English
        ("Q: who wrote Hamlet?\nA: ", "Shakespeare\n"),
        ("Q: who wrote Macbeth?\nA: ", "Shakespeare\n"),
        ("Q: who wrote Don Quijote?\nA: ", "Cervantes\n"),
        ("Q: what is the capital of France?\nA: ", "Paris\n"),
        ("Q: what is the capital of Spain?\nA: ", "Madrid\n"),
        ("Q: what is the capital of Italy?\nA: ", "Rome\n"),
        ("Q: what is two plus two?\nA: ", "four\n"),
        ("Q: what is three plus three?\nA: ", "six\n"),
        # Spanish
        ("Q: quien escribio Don Quijote?\nA: ", "Cervantes\n"),
        ("Q: quien escribio Hamlet?\nA: ", "Shakespeare\n"),
        ("Q: cual es la capital de Francia?\nA: ", "Paris\n"),
        ("Q: cual es la capital de Espana?\nA: ", "Madrid\n"),
        ("Q: cual es la capital de Italia?\nA: ", "Roma\n"),
        ("Q: cuanto es dos mas dos?\nA: ", "cuatro\n"),
    ]
    p, r = rng.choice(qa_facts)
    return {"prompt": p, "response": r, "task": "qa"}

def gen_repeat(rng):
    c = rng.choice(LOWERCASE)
    n = rng.choice([2, 3, 4])
    return {"prompt": f"INSTR: repeat '{c}' {NUM_WORDS[n]}\nRESP: ",
            "response": f"{c * n}\n", "task": "repeat"}

def extract_complete_lines(text, lang, rng, n, min_len=20, max_len=60):
    """Extraer ventanas de completacion desde el corpus."""
    lines = [l.strip() for l in text.split("\n") if min_len <= len(l.strip()) <= max_len]
    examples = []
    rng.shuffle(lines)
    for line in lines:
        words = line.split()
        if len(words) < 3:
            continue
        target = words[-1]
        context = " ".join(words[:-1])
        prompt = f"{lang}: '{context}'\nNEXT: "
        response = f"{target}\n"
        # Verificar que todos los chars estan en vocab base
        if all(c in tok.vocab for c in prompt + response):
            examples.append({"prompt": prompt, "response": response,
                             "task": f"complete-{lang.lower()}"})
        if len(examples) >= n:
            break
    return examples

def vocab_ok(ex):
    return all(c in tok.vocab for c in ex["prompt"] + ex["response"])

def main():
    Path("data").mkdir(exist_ok=True)
    rng = random.Random(SFT_BPE_SEED)
    eval_rng = random.Random(EVAL_BPE_SEED)

    for split, n_each, n_complete, fout, r in [
        ("train", 1000, 1000, "data/sft_bpe_dataset.jsonl", rng),
        ("eval",  250,  250,  "data/sft_bpe_eval.jsonl",    eval_rng),
    ]:
        examples = []
        for _ in range(n_each): examples.append(gen_qa(r))
        for _ in range(n_each): examples.append(gen_repeat(r))
        examples += extract_complete_lines(en_text, "EN", r, n_complete)
        examples += extract_complete_lines(es_text, "ES", r, n_complete)

        before = len(examples)
        examples = [ex for ex in examples if vocab_ok(ex)]
        print(f"[{split}] kept={len(examples)} dropped={before-len(examples)}")
        by_task = {}
        for ex in examples:
            by_task[ex["task"]] = by_task.get(ex["task"], 0) + 1
        print(f"  dist={by_task}")

        with open(fout, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
```

**Step 2: Correr y verificar**

```bash
python 33_build_sft_bpe.py 2>&1 | tee /tmp/cap33_dataset.txt
wc -l data/sft_bpe_dataset.jsonl data/sft_bpe_eval.jsonl
```

Expected: ~4000 train, ~1000 eval, 0 dropped. Si `complete-EN`/`complete-ES` no alcanza n, ajustar `min_len`/`max_len`.

**Step 3: Commit**

```bash
git add clase_14/practica/33_build_sft_bpe.py \
        clase_14/practica/data/sft_bpe_dataset.jsonl \
        clase_14/practica/data/sft_bpe_eval.jsonl
git commit -m "feat(cap33): dataset SFT-BPE 4 tareas bilingue"
```

---

## Task 12: Cap 33 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/33-dataset-sft-bpe.md`

Front matter: `title: "33 - Dataset SFT-BPE: 4 tareas bilingues"`, `weight: 330`, `math: true`.

Estructura:
1. Las 4 tareas: qa (bilingue), repeat (igual que C2), complete-en, complete-es
2. Por qué complete-* es natural para BPE (palabras como tokens coherentes)
3. Script completo embebido
4. Output literal de counts/distribución
5. Ejemplos concretos de cada tarea (de los JSONL reales)
6. Comparativa con Camino 2 (4 tareas → 4 tareas, pero 2 son nuevas BPE-naturales)

Hugo build + commit mensaje: `"docs(cap33): dataset SFT-BPE 4 tareas bilingues"`

---

## Task 13: Cap 34 — SFT-BPE training + eval

**Files:**
- Create: `clase_14/practica/34_train_sft_bpe.py`
- Output (gitignored): `checkpoints/mini_llama_bpe_sft.pt`

**Step 1: Escribir el script**

```python
"""34_train_sft_bpe.py - Cap 34: SFT con BPE + eval comparativo."""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import load_pretrained_mini_llama, get_device, generate_with_prompt
from _eval import load_jsonl, eval_exact_match, eval_drift

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab_size = tok.vocab_size

BLOCK = 256
BATCH = 32
LR = 1e-4
ITERS = 1500
WD = 0.01

model = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_base.pt", device=device,
    config=dict(vocab_size=vocab_size, max_seq_len=BLOCK,
                d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))
model.train()

examples = load_jsonl("data/sft_bpe_dataset.jsonl")
print(f"Loaded {len(examples)} SFT-BPE examples")

def encode_example(ex):
    prompt_ids = tok.encode(ex["prompt"])
    response_ids = tok.encode(ex["response"])
    full = prompt_ids + response_ids
    if len(full) > BLOCK + 1:
        full = full[:BLOCK + 1]
    P = len(prompt_ids)
    R = len(full) - P
    mask = [0] * (P - 1) + [1] * R
    assert len(mask) == len(full) - 1
    return full, mask

def get_batch():
    batch_inp, batch_tgt, batch_mask = [], [], []
    for _ in range(BATCH):
        ex = examples[torch.randint(0, len(examples), (1,)).item()]
        full, mask = encode_example(ex)
        while len(full) < BLOCK + 1:
            full.append(0)
            mask.append(0)
        full = full[:BLOCK + 1]
        mask = mask[:BLOCK]
        batch_inp.append(full[:-1])
        batch_tgt.append(full[1:])
        batch_mask.append(mask)
    return (torch.tensor(batch_inp, dtype=torch.long, device=device),
            torch.tensor(batch_tgt, dtype=torch.long, device=device),
            torch.tensor(batch_mask, dtype=torch.float, device=device))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

print(f"\nSFT-BPE training: {ITERS} iters\n")
for it in range(ITERS):
    inp, tgt, mask = get_batch()
    logits, _ = model(inp)
    loss_per_tok = F.cross_entropy(logits.reshape(-1, vocab_size),
                                   tgt.reshape(-1), reduction="none")
    loss_per_tok = loss_per_tok.reshape(inp.shape)
    masked_loss = (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)
    opt.zero_grad()
    masked_loss.backward()
    opt.step()
    if it % 100 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {masked_loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "checkpoints/mini_llama_bpe_sft.pt")
print("\nSaved -> checkpoints/mini_llama_bpe_sft.pt")

# Eval comparativo
print("\n=== Eval BPE-Base vs BPE-SFT ===\n")
results = {}
for name, ckpt in [("bpe-base", "checkpoints/mini_llama_bpe_base.pt"),
                   ("bpe-sft",  "checkpoints/mini_llama_bpe_sft.pt")]:
    print(f"--- {name} ---")
    m = load_pretrained_mini_llama(ckpt, device=device,
        config=dict(vocab_size=vocab_size, max_seq_len=BLOCK,
                    d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))
    em = eval_exact_match(m, "data/sft_bpe_eval.jsonl", tok, n_per_task=200, device=device)
    results[name] = em
    print(f"exact_match: {em}\n")

print("=== Tabla BPE-Base vs BPE-SFT vs Char-SFT (referencia cap25) ===")
char_sft = {"qa": 1.0, "repeat": 1.0, "reverse": 0.21, "upper": 0.235}
header = f"{'task':<15}{'bpe-base':<12}{'bpe-sft':<12}{'char-sft':<12}"
print(header)
for task in ["qa", "repeat", "complete-en", "complete-es"]:
    b = results["bpe-base"].get(task, 0.0)
    s = results["bpe-sft"].get(task, 0.0)
    c = char_sft.get(task, "N/A")
    print(f"{task:<15}{b:<12.3f}{s:<12.3f}{str(c):<12}")

print("\n=== Drift BPE-Base vs BPE-SFT ===")
ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]
for name, ckpt in [("bpe-base", "checkpoints/mini_llama_bpe_base.pt"),
                   ("bpe-sft",  "checkpoints/mini_llama_bpe_sft.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device,
        config=dict(vocab_size=vocab_size, max_seq_len=BLOCK,
                    d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))
    drift = eval_drift(m, ambiguous, tok, device=device)
    print(f"{name}: drift = {drift:.3f}")
```

**Step 2: Correr**

```bash
python 34_train_sft_bpe.py 2>&1 | tee /tmp/cap34_sft.txt
```
Expected: ~20-30s. Loss baja de ~3-5 a <1. Tabla muestra BPE-SFT >> BPE-Base en todas las tareas.

**Step 3: Commit**

```bash
git add clase_14/practica/34_train_sft_bpe.py
git commit -m "feat(cap34): SFT-BPE training + eval comparativo 4 columnas"
```

---

## Task 14: Cap 34 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/34-sft-bpe.md`

Front matter: `title: "34 - SFT con BPE: el salto que char-level no pudo dar"`, `weight: 340`, `math: true`.

Puntos clave vs cap 24 (char-level):
- Loss masking idéntico — mecanismo igual, tokenizer distinto
- Tabla literal 4 columnas (bpe-base / bpe-sft / char-sft / diferencia)
- Si complete-en/es llegó a 60-90%: esas tareas NO existían en char-level — son capacidades nuevas que habilita BPE
- Drift analysis
- Qué sigue: DPO sobre el BPE-SFT

Hugo build + commit: `"docs(cap34): SFT-BPE — salto vs char-level, complete-* capacidades nuevas"`

---

## Task 15: Cap 35 — script `35_build_dpo_bpe.py`

**Files:**
- Create: `clase_14/practica/35_build_dpo_bpe.py`
- Output (versionado): `clase_14/practica/data/dpo_bpe_dataset.jsonl`

**Step 1: Escribir el script** (idéntico en estructura a `20_build_dpo_dataset.py` pero con BPETokenizer)

```python
"""35_build_dpo_bpe.py - Cap 35: dataset DPO-BPE.

3000 triples = 1500 base-sampled (del BPE-base) + 1500 cross-task.
"""
import json, random, torch
from pathlib import Path
from _bpe import BPETokenizer
from _models import load_pretrained_mini_llama, generate_with_prompt
from _eval import load_jsonl

DPO_BPE_SEED = 143
torch.manual_seed(DPO_BPE_SEED)
random.seed(DPO_BPE_SEED)

tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab = set(tok.vocab.keys())
vocab_size = tok.vocab_size

print("Cargando BPE base model...")
base_model = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_base.pt",
    config=dict(vocab_size=vocab_size, max_seq_len=256,
                d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

sft = load_jsonl("data/sft_bpe_dataset.jsonl")
rng = random.Random(DPO_BPE_SEED)
rng.shuffle(sft)

triples = []

print("Generando 1500 triples base-sampled...")
for i, ex in enumerate(sft[:1500]):
    rej_full = generate_with_prompt(base_model, ex["prompt"], tok,
                                    max_new_tokens=20, temperature=0.8,
                                    top_k=10, stop_token="\n")
    rejected = rej_full[len(ex["prompt"]):]
    if not rejected.endswith("\n"):
        rejected += "\n"
    if rejected == ex["response"]:
        continue
    triples.append({"prompt": ex["prompt"], "chosen": ex["response"],
                    "rejected": rejected, "source": "base"})
    if (i + 1) % 200 == 0:
        print(f"  base-sampled: {i+1}/1500, aceptados: {len(triples)}", flush=True)

print(f"\nBase-sampled: {len(triples)} triples")

print("Generando 1500 triples cross-task...")
by_task = {}
for ex in sft:
    by_task.setdefault(ex["task"], []).append(ex)

base_count = len(triples)
for ex in sft[1500:3000]:
    other_tasks = [t for t in by_task if t != ex["task"]]
    other_task = rng.choice(other_tasks)
    other_ex = rng.choice(by_task[other_task])
    if other_ex["response"] == ex["response"]:
        continue
    triples.append({"prompt": ex["prompt"], "chosen": ex["response"],
                    "rejected": other_ex["response"], "source": "cross"})

print(f"Cross-task: {len(triples) - base_count} triples")

def vocab_ok(t):
    return all(c in vocab for c in t["prompt"] + t["chosen"] + t["rejected"])

before = len(triples)
triples = [t for t in triples if vocab_ok(t)]
print(f"Filtered by vocab: {before - len(triples)} dropped")

with open("data/dpo_bpe_dataset.jsonl", "w") as f:
    for t in triples:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

by_source = {}
for t in triples:
    by_source[t["source"]] = by_source.get(t["source"], 0) + 1
print(f"\nTotal: {len(triples)}  by_source: {by_source}")
```

**Step 2: Correr (tomará ~7 min — igual que Camino 2)**

```bash
python 35_build_dpo_bpe.py 2>&1 | tee /tmp/cap35_dataset.txt
```

Expected: ~3000 triples, 50/50 base/cross. **Diferencia pedagógica clave**: los rejected base-sampled del BPE-base son más coherentes lingüísticamente que los del char-base (palabras reales vs chars aleatorios), lo que hace DPO más informativo.

**Step 3: Commit**

```bash
git add clase_14/practica/35_build_dpo_bpe.py clase_14/practica/data/dpo_bpe_dataset.jsonl
git commit -m "feat(cap35): dataset DPO-BPE mix base-sampled + cross-task"
```

---

## Task 16: Cap 35 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/35-dataset-dpo-bpe.md`

Front matter: `title: "35 - Dataset DPO-BPE: rejected mas ricos"`, `weight: 350`, `math: true`.

Punto diferencial vs cap 28 (char-level DPO dataset):
- **Los rejected del BPE-base son palabras reales, no gibberish**. Por ejemplo: `prompt: "EN: 'To be or not to'\nNEXT: "`, chosen: `"be"`, rejected base-sampled: `"live"` (una palabra coherente pero incorrecta). Esto es DPO como fue diseñado — preferir la respuesta correcta entre opciones plausibles, no entre correcta y galimatías.
- Mostrar spot-check de 3 ejemplos base-sampled y 3 cross-task de los JSONL reales.

Hugo build + commit: `"docs(cap35): dataset DPO-BPE — rejected linguisticamente ricos"`

---

## Task 17: Cap 36 — DPO-BPE training + beta sweep

**Files:**
- Create: `clase_14/practica/36_train_dpo_bpe.py`
- Output (gitignored): `checkpoints/mini_llama_bpe_dpo_b01.pt`, `checkpoints/mini_llama_bpe_dpo_b05.pt`

**Step 1: Escribir el script**

```python
"""36_train_dpo_bpe.py - Cap 36: DPO-BPE + beta sweep.

Prueba beta=0.1 Y beta=0.5 para validar hipotesis del cap 29
(DPO-char-level se sobre-ajusto con beta=0.1 demasiado bajo).
"""
import torch
from pathlib import Path
from _bpe import BPETokenizer
from _models import load_pretrained_mini_llama, dpo_loss, get_device
from _eval import load_jsonl, eval_exact_match

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab_size = tok.vocab_size
cfg = dict(vocab_size=vocab_size, max_seq_len=256,
           d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

ITERS = 1000
BATCH = 16
LR = 5e-5
WD = 0.01

triples = load_jsonl("data/dpo_bpe_dataset.jsonl")
print(f"Loaded {len(triples)} DPO-BPE triples\n")

def encode(s): return torch.tensor([tok.vocab.get(c, 0) for c in s], dtype=torch.long)

def run_dpo(beta, out_ckpt):
    print(f"=== DPO-BPE beta={beta} ===")
    policy = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_sft.pt", device=device, config=cfg)
    ref    = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_sft.pt", device=device, config=cfg)
    for p in ref.parameters(): p.requires_grad_(False)
    ref.eval(); policy.train()

    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WD)

    for it in range(ITERS):
        losses = []
        for _ in range(BATCH):
            t = triples[torch.randint(0, len(triples), (1,)).item()]
            p_ids = encode(t["prompt"])
            c_ids = encode(t["chosen"])
            r_ids = encode(t["rejected"])
            l = dpo_loss(policy, ref, p_ids, c_ids, r_ids, beta=beta, device=device)
            losses.append(l)
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == ITERS - 1:
            print(f"  iter {it:4d}  loss {loss.item():.4f}", flush=True)

    torch.save(policy.state_dict(), out_ckpt)
    print(f"  Saved -> {out_ckpt}\n")
    return policy

for beta, ckpt in [(0.1, "checkpoints/mini_llama_bpe_dpo_b01.pt"),
                   (0.5, "checkpoints/mini_llama_bpe_dpo_b05.pt")]:
    run_dpo(beta, ckpt)

# Eval comparativa final
print("=== Eval comparativo: BPE-SFT vs DPO-b01 vs DPO-b05 ===\n")
results = {}
for name, ckpt in [("bpe-sft",    "checkpoints/mini_llama_bpe_sft.pt"),
                   ("dpo-b01",    "checkpoints/mini_llama_bpe_dpo_b01.pt"),
                   ("dpo-b05",    "checkpoints/mini_llama_bpe_dpo_b05.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    em = eval_exact_match(m, "data/sft_bpe_eval.jsonl", tok, n_per_task=200, device=device)
    results[name] = em
    print(f"{name}: {em}")

print("\n=== Tabla final ===")
print(f"{'task':<15}{'bpe-sft':<12}{'dpo-b01':<12}{'dpo-b05':<12}")
for task in ["qa", "repeat", "complete-en", "complete-es"]:
    s = results["bpe-sft"].get(task, 0.0)
    d1 = results["dpo-b01"].get(task, 0.0)
    d5 = results["dpo-b05"].get(task, 0.0)
    print(f"{task:<15}{s:<12.3f}{d1:<12.3f}{d5:<12.3f}")
```

**Step 2: Correr (tomará ~14 min — 2 × 7 min para 2 betas)**

```bash
python 36_train_dpo_bpe.py 2>&1 | tee /tmp/cap36_dpo.txt
```

Expected: tabla comparativa SFT vs DPO-b01 vs DPO-b05. Hipótesis a validar: beta=0.5 debería sobre-ajustar menos y dar mejor accuracy que beta=0.1.

**Step 3: Commit**

```bash
git add clase_14/practica/36_train_dpo_bpe.py
git commit -m "feat(cap36): DPO-BPE training con beta sweep 0.1 vs 0.5"
```

---

## Task 18: Cap 36 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/36-dpo-bpe.md`

Front matter: `title: "36 - DPO-BPE: beta sweep y validacion de hipotesis"`, `weight: 360`, `math: true`.

Puntos cruciales:
1. Referenciar explícitamente cap 29 — "las 4 hipótesis del cap 29, hoy probamos la del beta"
2. Setup: policy = SFT-BPE, ref = SFT-BPE, dos corridas (beta 0.1 y 0.5)
3. Tabla literal 3 columnas del output de `/tmp/cap36_dpo.txt`
4. Análisis: si beta=0.5 mejoró vs 0.1 → hipótesis validada. Si ambos degradan → apuntar a cross-task noise o data quality
5. "DPO con rejected lingüísticamente ricos (palabras reales vs chars aleatorios) debería dar señal más limpia"
6. Próximos pasos: cap 37 compara todo

Hugo build + commit: `"docs(cap36): DPO-BPE beta sweep — validacion hipotesis cap 29"`

---

## Task 19: Cap 37 — comparación final char vs BPE

**Files:**
- Create: `clase_14/practica/37_compare_char_vs_bpe.py`

**Step 1: Escribir el script**

```python
"""37_compare_char_vs_bpe.py - Cap 37: tabla maestra char-level vs BPE-level.

Carga los 3 char-level (base/sft/dpo) y los 3 BPE-level (base/sft/dpo).
Eval sobre el subset compartido (qa + repeat que existen en ambos).
"""
import torch, json
from pathlib import Path
from _bpe import BPETokenizer, CharTokenizer
from _models import load_pretrained_mini_llama
from _eval import build_char_maps, eval_exact_match, eval_drift, load_jsonl

device = "cpu"  # eval ligero, cpu es suficiente

# Char-level
text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
char_tok = CharTokenizer(c2i, i2c)
char_cfg = dict(vocab_size=len(c2i), max_seq_len=256,
                d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

# BPE-level
bpe_tok = BPETokenizer.load("data/bpe_tokenizer.json")
bpe_cfg = dict(vocab_size=bpe_tok.vocab_size, max_seq_len=256,
               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

# Shared eval: qa + repeat del eval set BPE (que son iguales en formato a char-level)
shared = [ex for ex in load_jsonl("data/sft_bpe_eval.jsonl")
          if ex["task"] in {"qa", "repeat"}]
with open("/tmp/shared_eval.jsonl", "w") as f:
    for ex in shared:
        f.write(json.dumps(ex) + "\n")

print("=== Tabla maestra: char-level vs BPE-level ===\n")
print(f"{'modelo':<20}{'qa':<10}{'repeat':<10}{'drift':<10}")

ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]

for label, ckpt, tok_obj, cfg in [
    ("char-base",  "checkpoints/mini_llama_base.pt",       char_tok, char_cfg),
    ("char-sft",   "checkpoints/mini_llama_sft.pt",        char_tok, char_cfg),
    ("char-dpo",   "checkpoints/mini_llama_dpo.pt",        char_tok, char_cfg),
    ("bpe-base",   "checkpoints/mini_llama_bpe_base.pt",   bpe_tok,  bpe_cfg),
    ("bpe-sft",    "checkpoints/mini_llama_bpe_sft.pt",    bpe_tok,  bpe_cfg),
    ("bpe-dpo-b05","checkpoints/mini_llama_bpe_dpo_b05.pt",bpe_tok,  bpe_cfg),
]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    em = eval_exact_match(m, "/tmp/shared_eval.jsonl", tok_obj,
                          n_per_task=100, device=device)
    drift = eval_drift(m, ambiguous, tok_obj, device=device)
    qa = em.get("qa", 0.0)
    rep = em.get("repeat", 0.0)
    print(f"{label:<20}{qa:<10.3f}{rep:<10.3f}{drift:<10.3f}")
```

**Step 2: Correr y capturar**

```bash
python 37_compare_char_vs_bpe.py 2>&1 | tee /tmp/cap37_compare.txt
```
Expected: carga 6 modelos, imprime tabla maestra.

**Step 3: Commit**

```bash
git add clase_14/practica/37_compare_char_vs_bpe.py
git commit -m "feat(cap37): tabla maestra char-level vs BPE-level comparativa"
```

---

## Task 20: Cap 37 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/37-comparacion-char-vs-bpe.md`

Front matter: `title: "37 - Comparacion char-level vs BPE: la leccion de la tokenizacion"`, `weight: 370`, `math: true`.

**Es el capítulo más importante de Camino 2.5.** Estructura:
1. Tabla maestra literal de 6 filas (char-base/sft/dpo vs bpe-base/sft/dpo)
2. Lectura: ¿qué mejoró con BPE? ¿qué es igual?
3. Por qué complete-* no existe en char-level (semántica de palabras vs chars)
4. Validación o refutación de hipótesis beta del cap 29
5. La regla práctica: cuándo usar char-level, cuándo BPE, cuándo tiktoken 100k
6. Cierre Camino 2.5 + referencia a Camino 3 (interpretabilidad — ahora con un modelo MEJOR para analizar)

Hugo build + commit: `"docs(cap37): comparacion char vs BPE — la leccion de tokenizacion"`

---

## Task 21: Glossary `bpe.md` + Hub `_index.md`

**Files:**
- Create: `site/content/fundamentos/bpe.md` (~1500 palabras)
- Modify: `site/content/clases/clase-14/practica/_index.md`

**Step 1: Escribir `bpe.md`**

Front matter: `title: "BPE (Byte Pair Encoding)"`, `weight: 290`, `math: true`.

Estructura (~1500 palabras, depth igual a `self-attention.md`):
1. El problema: char-level vs word-level vocab (con math)
2. El algoritmo BPE — merge frequencies, pseudocódigo
3. Ejemplo paso a paso sobre corpus chico ("aaabdaaabac")
4. BPE vs WordPiece (BERT) vs SentencePiece (LLaMA)
5. Por qué vocab size importa para el modelo (embedding params)
6. Implementación PyTorch (snippet del `_bpe.py`)
7. Limitaciones (tokenización subóptima en algunos idiomas, fertility issues)
8. Resumen + Ver también (links a sft, dpo, clase-14 cap 30)

**Step 2: Actualizar `_index.md` del hub**

Agregar sección "Camino 2.5 — BPE addendum" con cards caps 30-37 + descripción. Insertar entre el cierre de Camino 2 y "Que viene despues".

**Step 3: Hugo build + commit**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site && hugo --quiet && echo "OK"
git add site/content/fundamentos/bpe.md site/content/clases/clase-14/practica/_index.md
git commit -m "docs(glosario+hub): bpe.md + Camino 2.5 en hub _index"
```

---

## Task 22: Actualizar memoria del proyecto + verificación final

**Step 1: Actualizar memoria**

Modificar `~/.claude/projects/.../memory/project_clase_14_caminos_pendientes.md`:
- Marcar Camino 2.5 como en-progreso / completado según estado
- Actualizar resultado del beta sweep
- Notar si hipótesis del cap 29 se validó o refutó

**Step 2: Verificación final**

```bash
cd clase_14/practica
source .venv/bin/activate
python -m pytest tests/ -v
```
Expected: 11/11 PASS (5 BPE + 5 helpers + 1 eval).

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site && hugo --quiet && echo "OK"
```
Expected: BUILD OK.

```bash
git log --oneline feat/clase-14-camino-2.5-bpe ^feat/clase-14-camino-2-sft-dpo
```
Mostrar historial de commits del Camino 2.5.

---

## Resumen de outputs producidos

- 1 módulo nuevo: `_bpe.py` (BPETokenizer + CharTokenizer)
- Refactor: `_models.py` + `_eval.py` tokenizer-agnostic
- Compat: 5 scripts Camino 2 actualizados (1 línea c/u)
- 8 scripts ejecutables: `30_build_bpe.py` … `37_compare_char_vs_bpe.py`
- 8 capítulos Hugo: `30-bpe-desde-cero.md` … `37-comparacion-char-vs-bpe.md`
- 5 tests nuevos en `tests/test_bpe.py`
- 4 datasets versionados (tokenizer JSON + 3 JSONL)
- 3 checkpoints gitignored (bpe-base, bpe-sft, bpe-dpo × 2 betas)
- 1 entrada glosario: `bpe.md` (~1500 palabras)
- 1 update hub `_index.md`
- 1 update memoria

Total: ~22 commits en branch `feat/clase-14-camino-2.5-bpe`.
