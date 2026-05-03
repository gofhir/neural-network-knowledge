# Clase 14 — Camino 3 (Interpretabilidad mecanicista) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Construir herramientas de interpretabilidad mecanicista desde cero (hooks, residual stream, logit lens, induction heads, QK/OV, activation patching, SAEs) y aplicarlas a los modelos ya entrenados del curso, produciendo 14 capitulos Hugo (caps 50-63) + 13 scripts ejecutables + modulo `_interp.py` + clase `SparseAutoencoder`.

**Architecture:** `_interp.py` agrega helpers (cache_activations via context manager con forward hooks, logit_lens, patch_activation, qk_circuit, ov_circuit, previous_token_score, induction_score) + clase `SparseAutoencoder`. Sin librerias externas (TransformerLens NO se usa, fiel a la pedagogia "build it yourself" del curso). Reusa modelos ya entrenados: `mini_llama_base.pt`, `mini_llama_sft.pt`, `mini_bert_finetuned.pt`. Solo se agrega un checkpoint nuevo: `sae_mini_llama.pt`.

**Tech Stack:** Python 3.12 + PyTorch (MPS) + Hugo. Venv existente en `clase_14/practica/.venv/`. Branch `feat/clase-14-camino-3-interpretabilidad` desde `main` (ya creado con design doc commiteado).

**Design doc:** `docs/plans/2026-05-03-clase-14-camino-3-interpretabilidad-design.md`

**Working dir:** `/Users/robertoaraneda/projects/personal/courses/ia-uc`

**Verification model:**
- Helpers Python → TDD con pytest (≥7 tests nuevos)
- Scripts ejecutables → corren + output en rango esperado
- Capitulos Hugo → output literal incluido + `hugo --quiet` limpio

**Convenciones del curso (NO desviar):**
- Español sin tildes (a/e/i/o/u sin acentos), excepto ñ.
- Commits sin Co-Authored-By trailer.
- Patron por cap: script ejecutable + capitulo Hugo + commit.
- Honestidad pedagogica: si un patron no emerge a la escala de Mini-LLaMA, documentarlo.

---

## Task 0: Verificar baseline

**Files:** ninguno

**Step 1: Verificar branch y baseline tests**
```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git branch --show-current
# Expected: feat/clase-14-camino-3-interpretabilidad

cd clase_14/practica
source .venv/bin/activate
python -m pytest tests/ -q 2>&1 | tail -3
# Expected: 23 passed
```

**Step 2: Crear test file y modulo vacio**
```bash
touch tests/test_interp.py
touch _interp.py
git add tests/test_interp.py _interp.py
git commit -m "chore: scaffold _interp module + test_interp"
```

---

## Task 1: `cache_activations` helper (TDD)

**Files:**
- Modify: `clase_14/practica/_interp.py`
- Modify: `clase_14/practica/tests/test_interp.py`

**Step 1: Test failing**

```python
# tests/test_interp.py
import torch
from _interp import cache_activations
from _models import MiniGPT

def test_cache_activations_captures_correct_shapes():
    model = MiniGPT(vocab_size=65, d_model=128, n_heads=4, n_layers=4,
                    d_ff=512, max_seq_len=64, dropout=0.0)
    model.eval()
    ids = torch.zeros(1, 8, dtype=torch.long)
    names = ["blocks.0", "blocks.3"]
    with cache_activations(model, names) as cache:
        with torch.no_grad():
            model(ids)
    assert "blocks.0" in cache
    assert "blocks.3" in cache
    assert cache["blocks.0"].shape == (1, 8, 128)
    assert cache["blocks.3"].shape == (1, 8, 128)

def test_cache_activations_cleanup_removes_hooks():
    model = MiniGPT(vocab_size=65, d_model=128, n_heads=4, n_layers=4,
                    d_ff=512, max_seq_len=64, dropout=0.0)
    n_hooks_before = sum(len(m._forward_hooks) for m in model.modules())
    with cache_activations(model, ["blocks.0"]):
        pass
    n_hooks_after = sum(len(m._forward_hooks) for m in model.modules())
    assert n_hooks_after == n_hooks_before
```

**Step 2: Run tests — verify FAIL**
```bash
python -m pytest tests/test_interp.py -v 2>&1 | tail -10
# Expected: FAIL — cache_activations not defined
```

**Step 3: Implement**

```python
# _interp.py
from contextlib import contextmanager
import torch

@contextmanager
def cache_activations(model, names):
    """Context manager that registers forward hooks on submodules by name.
    Returns dict {name: tensor of last forward output}."""
    cache = {}
    handles = []
    name_to_module = dict(model.named_modules())
    for name in names:
        if name not in name_to_module:
            raise KeyError(f"Module '{name}' not found in model")
        def make_hook(n):
            def hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                cache[n] = out.detach()
            return hook
        handles.append(name_to_module[name].register_forward_hook(make_hook(name)))
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()
```

**Step 4: Run tests — verify PASS**
```bash
python -m pytest tests/test_interp.py -v 2>&1 | tail -5
# Expected: 2 passed
```

**Step 5: Commit**
```bash
git add _interp.py tests/test_interp.py
git commit -m "feat(_interp): cache_activations context manager con forward hooks"
```

---

## Task 2: `logit_lens` helper (TDD)

**Files:**
- Modify: `clase_14/practica/_interp.py`
- Modify: `clase_14/practica/tests/test_interp.py`

**Step 1: Test failing**

```python
def test_logit_lens_consistent_with_full_forward():
    """Aplicar logit_lens al residual final debe ser igual al output del modelo."""
    model = MiniGPT(vocab_size=65, d_model=128, n_heads=4, n_layers=4,
                    d_ff=512, max_seq_len=64, dropout=0.0)
    model.eval()
    from _interp import logit_lens
    ids = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        full_logits = model(ids)
    # Cachear el residual stream final (despues del ultimo block + ln)
    with cache_activations(model, ["ln_f"]) as cache:
        with torch.no_grad():
            model(ids)
    final_residual = cache["ln_f"]
    lens_logits = logit_lens(model, final_residual)
    assert torch.allclose(full_logits, lens_logits, atol=1e-5)
```

**Step 2: Run — FAIL**

**Step 3: Implement**

```python
def logit_lens(model, residual):
    """Aplica head al residual stream para proyectar al vocab.
    Para MiniGPT: residual -> head. Para MiniLLaMA: residual -> head.
    Para modelos con norm final, el residual debe ser POST-norm (despues de ln_f)."""
    return model.head(residual)
```

**Step 4: Run — PASS**

**Step 5: Commit**
```bash
git add _interp.py tests/test_interp.py
git commit -m "feat(_interp): logit_lens proyecta residual al vocab"
```

---

## Task 3: `patch_activation` helper (TDD)

**Files:**
- Modify: `clase_14/practica/_interp.py`
- Modify: `clase_14/practica/tests/test_interp.py`

**Step 1: Test failing**

```python
def test_patch_activation_changes_only_target_position():
    """Patchear posicion 3 con zeros debe cambiar logits en posicion >=3
    pero no en posiciones <3 (porque atencion causal solo mira hacia atras)."""
    torch.manual_seed(0)
    model = MiniGPT(vocab_size=65, d_model=128, n_heads=4, n_layers=4,
                    d_ff=512, max_seq_len=64, dropout=0.0)
    model.eval()
    from _interp import patch_activation
    ids = torch.randint(0, 65, (1, 8))
    with torch.no_grad():
        clean_logits = model(ids)
    # Patchear blocks.1 en posicion 3 con un tensor de zeros
    patch = torch.zeros(1, 1, 128)
    patched_logits = patch_activation(model, ids,
                                      {"blocks.1": (3, patch)})
    # Posicion 0,1,2 NO debe cambiar
    assert torch.allclose(clean_logits[0, :3], patched_logits[0, :3], atol=1e-5)
    # Posicion 3 SI debe cambiar
    assert not torch.allclose(clean_logits[0, 3], patched_logits[0, 3], atol=1e-5)
```

**Step 2: Run — FAIL**

**Step 3: Implement**

```python
def patch_activation(model, ids, patch_dict):
    """Forward pass con activaciones reemplazadas en posiciones especificas.
    patch_dict: {module_name: (position_or_slice, replacement_tensor)}.
    replacement_tensor shape: (B, n_positions, d_model)."""
    handles = []
    name_to_module = dict(model.named_modules())
    for name, (positions, replacement) in patch_dict.items():
        def make_patch_hook(positions, replacement):
            def hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                out = out.clone()
                if isinstance(positions, int):
                    out[:, positions:positions+1] = replacement
                else:
                    out[:, positions] = replacement
                return out if not isinstance(output, tuple) else (out, *output[1:])
            return hook
        handles.append(name_to_module[name].register_forward_hook(
            make_patch_hook(positions, replacement)))
    try:
        with torch.no_grad():
            return model(ids)
    finally:
        for h in handles:
            h.remove()
```

**Step 4: Run — PASS**

**Step 5: Commit**
```bash
git add _interp.py tests/test_interp.py
git commit -m "feat(_interp): patch_activation para intervencion causal"
```

---

## Task 4: `qk_circuit` y `ov_circuit` (TDD)

**Files:**
- Modify: `clase_14/practica/_interp.py`
- Modify: `clase_14/practica/tests/test_interp.py`

**Step 1: Test failing**

```python
def test_qk_circuit_shape():
    from _interp import qk_circuit
    W_Q = torch.randn(128, 32)  # d_model, d_head
    W_K = torch.randn(128, 32)
    qk = qk_circuit(W_Q, W_K)
    assert qk.shape == (128, 128)

def test_ov_circuit_shape():
    from _interp import ov_circuit
    W_V = torch.randn(128, 32)
    W_O = torch.randn(32, 128)  # d_head, d_model
    ov = ov_circuit(W_V, W_O)
    assert ov.shape == (128, 128)
```

**Step 2: Run — FAIL**

**Step 3: Implement**

```python
def qk_circuit(W_Q, W_K):
    """QK circuit: W_Q @ W_K^T. Define como una cabeza decide a que atender.
    Shape: (d_model, d_model)."""
    return W_Q @ W_K.T

def ov_circuit(W_V, W_O):
    """OV circuit: W_V @ W_O. Define que informacion mueve una cabeza
    desde la fuente al destino. Shape: (d_model, d_model)."""
    return W_V @ W_O
```

**Step 4: Run — PASS**

**Step 5: Commit**
```bash
git add _interp.py tests/test_interp.py
git commit -m "feat(_interp): qk_circuit + ov_circuit decomposition"
```

---

## Task 5: `previous_token_score` y `induction_score` (TDD)

**Files:**
- Modify: `clase_14/practica/_interp.py`
- Modify: `clase_14/practica/tests/test_interp.py`

**Step 1: Test failing**

```python
def test_previous_token_score_perfect():
    """Un patron de atencion que mira EXACTAMENTE al anterior debe dar score = 1.0."""
    from _interp import previous_token_score
    T = 8
    attn = torch.zeros(T, T)
    for i in range(1, T):
        attn[i, i-1] = 1.0
    score = previous_token_score(attn)
    assert abs(score - 1.0) < 1e-6

def test_previous_token_score_uniform_low():
    """Atencion uniforme da score bajo (~1/T)."""
    from _interp import previous_token_score
    T = 8
    attn = torch.ones(T, T) / T
    score = previous_token_score(attn)
    assert score < 0.2

def test_induction_score_repeated_prompt():
    """En un prompt [A B X Y A], la cabeza ideal de induccion atiende desde A_2 a B."""
    from _interp import induction_score
    T = 5
    # Prompt: A=0, B=1, X=2, Y=3, A=4
    # induction head: en posicion 4, atiende a posicion 1 (B)
    attn = torch.zeros(T, T)
    attn[4, 1] = 1.0  # induction
    # Debe ser mayor que un patron sin induction
    ids = torch.tensor([0, 1, 2, 3, 0])
    score = induction_score(attn, ids)
    assert score > 0.5
```

**Step 2: Run — FAIL**

**Step 3: Implement**

```python
def previous_token_score(attn):
    """Score [0, 1]: cuanto atiende cada posicion i a la i-1.
    attn shape: (T, T). Asume causal (triangular inferior)."""
    T = attn.shape[0]
    if T < 2:
        return 0.0
    diag = torch.tensor([attn[i, i-1].item() for i in range(1, T)])
    return diag.mean().item()

def induction_score(attn, ids):
    """Score de induccion: para cada token repetido en posicion j (con j > i),
    cuanto atiende attn[j] a la posicion i+1 (donde estaba el siguiente token la primera vez).
    Patron: ...A B... A -> B."""
    T = attn.shape[0]
    scores = []
    ids_list = ids.tolist()
    for j in range(2, T):
        tok = ids_list[j]
        # Buscar primera aparicion de tok antes de j
        for i in range(j - 1):
            if ids_list[i] == tok and i + 1 < j:
                scores.append(attn[j, i + 1].item())
                break
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
```

**Step 4: Run — PASS**

**Step 5: Commit**
```bash
git add _interp.py tests/test_interp.py
git commit -m "feat(_interp): previous_token_score + induction_score"
```

---

## Task 6: `SparseAutoencoder` clase (TDD)

**Files:**
- Modify: `clase_14/practica/_interp.py`
- Modify: `clase_14/practica/tests/test_interp.py`

**Step 1: Test failing**

```python
def test_sae_reconstruction_loss_decreases():
    from _interp import SparseAutoencoder
    torch.manual_seed(0)
    sae = SparseAutoencoder(d_model=128, d_features=512, l1_coeff=1e-3)
    x = torch.randn(64, 128)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    initial_loss = None
    for step in range(200):
        opt.zero_grad()
        recon, features = sae(x)
        recon_loss = ((x - recon) ** 2).mean()
        l1_loss = features.abs().mean()
        loss = recon_loss + sae.l1_coeff * l1_loss
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = recon_loss.item()
    final_loss = recon_loss.item()
    assert final_loss < initial_loss * 0.5  # al menos 50% menos
```

**Step 2: Run — FAIL**

**Step 3: Implement**

```python
class SparseAutoencoder(torch.nn.Module):
    """Sparse Autoencoder estilo Bricken et al. 2023.
    encoder: Linear + ReLU. decoder: Linear (sin bias para evitar shrinkage).
    loss = MSE(reconstruction) + l1_coeff * L1(features)."""
    def __init__(self, d_model, d_features, l1_coeff=1e-3):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_features)
        self.decoder = torch.nn.Linear(d_features, d_model, bias=False)
        self.l1_coeff = l1_coeff

    def forward(self, x):
        features = torch.relu(self.encoder(x))
        recon = self.decoder(features)
        return recon, features
```

**Step 4: Run — PASS**

**Step 5: Commit**
```bash
git add _interp.py tests/test_interp.py
git commit -m "feat(_interp): SparseAutoencoder con L1 sparsity"
```

---

## Tasks 7-19: Capitulos 50-62 (script + Hugo)

**Patron comun por cada cap (replicar en cada task 7-19):**

1. Crear `clase_14/practica/{cap_num}_{slug}.py` con script ejecutable
2. Correr el script y capturar output literal: `python {cap_num}_{slug}.py > /tmp/cap{cap_num}.txt 2>&1`
3. Crear `site/content/clases/clase-14/practica/{cap_num}-{slug}.md` con:
   - Front matter: `title`, `weight: {cap_num}*10`, `math: true`
   - Secciones: apertura, concepto, script (literal), output literal, analisis, preguntas (3)
   - Sin tildes (a/e/i/o/u sin acentos)
4. Verificar `cd site && hugo --quiet`
5. Commit: `git add clase_14/practica/{cap_num}_*.py site/content/clases/clase-14/practica/{cap_num}-*.md && git commit -m "feat+docs(cap{cap_num}): {titulo}"`

---

### Task 7: Cap 50 — Forward hooks

**Files:**
- Create: `clase_14/practica/50_forward_hooks.py`
- Create: `site/content/clases/clase-14/practica/50-forward-hooks.md`

**Script:**
```python
"""50_forward_hooks.py - Cap 50: forward hooks y cache de activaciones."""
import torch
from _models import MiniGPT, get_device, CharTokenizer, load_text
from _interp import cache_activations

torch.manual_seed(1337)
device = get_device()

ckpt = torch.load("checkpoints/mini_llama_base.pt", map_location=device, weights_only=False)
cfg = ckpt["config"]
from _models import MiniLLaMA
model = MiniLLaMA(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)
ids = torch.tensor([tok.encode("To be or not to ")], dtype=torch.long, device=device)

names = [f"blocks.{i}" for i in range(cfg["n_layers"])]
print(f"Cacheando activaciones de {len(names)} bloques sobre prompt de {ids.shape[1]} tokens\n")
with cache_activations(model, names) as cache:
    with torch.no_grad():
        model(ids)

print("Activaciones capturadas:")
for name, tensor in cache.items():
    print(f"  {name:>15}: shape={tuple(tensor.shape)}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")

print("\nNorma del residual stream por capa (||x||_2 promedio):")
for name, tensor in cache.items():
    norm = tensor.norm(dim=-1).mean().item()
    print(f"  {name:>15}: {norm:.3f}")
```

**Hugo chapter sections:**
1. Apertura — por que necesitamos hooks (no podemos modificar el modelo)
2. `register_forward_hook` mecanica de PyTorch
3. Context manager para cleanup automatico
4. Script + output literal
5. Analisis: la norma crece o se mantiene a lo largo del modelo?
6. 3 preguntas de verificacion

**Output esperado en rango:** ≥4 capas cacheadas, normas en rango ~1-20.

**Commit:**
```bash
git add clase_14/practica/50_forward_hooks.py site/content/clases/clase-14/practica/50-forward-hooks.md
git commit -m "feat+docs(cap50): forward hooks y cache de activaciones"
```

---

### Task 8: Cap 51 — Residual stream

**Files:**
- Create: `clase_14/practica/51_residual_stream.py`
- Create: `site/content/clases/clase-14/practica/51-residual-stream.md`

**Concepto del cap:** El residual stream es la "autopista" del Transformer. Cada bloque LEE (atencion + FFN) y ESCRIBE (suma) — nunca sobreescribe. Demostrar capturando el residual antes y despues de cada bloque y midiendo cuanto cambio.

**Script idea:** Cachear input y output de cada bloque, computar `delta = output - input` por bloque, mostrar la magnitud relativa `||delta|| / ||input||` — esperar valores pequenos (0.1-0.3), confirmando que los bloques hacen edits incrementales sobre el stream.

**Hugo:** Diagrama ASCII del residual stream + analisis de los deltas.

**Commit:**
```bash
git add clase_14/practica/51_residual_stream.py site/content/clases/clase-14/practica/51-residual-stream.md
git commit -m "feat+docs(cap51): residual stream — autopista del Transformer"
```

---

### Task 9: Cap 52 — Logit lens

**Files:**
- Create: `clase_14/practica/52_logit_lens.py`
- Create: `site/content/clases/clase-14/practica/52-logit-lens.md`

**Concepto:** Aplicar `lm_head` al residual de capas intermedias para ver que predice el modelo "a media procesamiento". Mostrar que las capas tempranas predicen tokens superficiales (caracteres comunes) y las tardias afinan la prediccion.

**Script idea:** Sobre prompt "To be or not to ", cachear el residual tras cada capa, aplicar `logit_lens`, mostrar top-3 predicciones de cada capa para la posicion final. Esperar progresion desde "ruido" (capa 0) a "be" (capa final).

**Commit:**
```bash
git add clase_14/practica/52_logit_lens.py site/content/clases/clase-14/practica/52-logit-lens.md
git commit -m "feat+docs(cap52): logit lens — predicciones capa a capa"
```

---

### Task 10: Cap 53 — Heatmaps de atencion

**Files:**
- Create: `clase_14/practica/53_attention_heatmaps.py`
- Create: `site/content/clases/clase-14/practica/53-attention-heatmaps.md`

**Concepto:** Capturar `attn_weights` de TODAS las cabezas y visualizarlos como ASCII heatmaps. Sobre prompt fijo de Shakespeare, mostrar la matriz `(T, T)` de cada cabeza (n_layers × n_heads = 16 heatmaps).

**Script idea:** Modificar el forward de `GroupedQueryAttention` para retornar attn_weights (o usar hook), generar ASCII art con `█▓▒░ ` segun magnitud.

**Hugo:** Mostrar 4-6 heatmaps representativos en el cap, identificar patrones a ojo (diagonal = self, sub-diagonal = previous token).

**Commit:**
```bash
git add clase_14/practica/53_attention_heatmaps.py site/content/clases/clase-14/practica/53-attention-heatmaps.md
git commit -m "feat+docs(cap53): heatmaps de atencion ASCII por capa/cabeza"
```

---

### Task 11: Cap 54 — Previous-token heads

**Files:**
- Create: `clase_14/practica/54_previous_token_heads.py`
- Create: `site/content/clases/clase-14/practica/54-previous-token-heads.md`

**Concepto:** Identificar las cabezas que copian del token anterior. Computar `previous_token_score` para cada cabeza, ordenar de mayor a menor, mostrar top-3.

**Script idea:** Sobre 50 prompts diferentes de Shakespeare, promediar el score por cabeza. Mostrar tabla ordenada. Honestidad: si Mini-LLaMA no tiene una cabeza claramente >0.5, documentarlo.

**Commit:**
```bash
git add clase_14/practica/54_previous_token_heads.py site/content/clases/clase-14/practica/54-previous-token-heads.md
git commit -m "feat+docs(cap54): previous-token heads — el patron mas simple"
```

---

### Task 12: Cap 55 — Induction heads

**Files:**
- Create: `clase_14/practica/55_induction_heads.py`
- Create: `site/content/clases/clase-14/practica/55-induction-heads.md`

**Concepto:** Diseñar prompts repetidos `[A][B][C]...[X][A][B]` y medir cuanto atiende cada cabeza al token siguiente al match (induccion). Anthropic: estas cabezas son la base del in-context learning.

**Script idea:** Generar 30 prompts con patron `random_seq + random_seq` (la segunda mitad repite la primera), computar `induction_score` por cabeza. Top-3 candidatas a induction head.

**Honestidad:** Mini-LLaMA tiene 4 capas; induction heads emergen tipicamente en capa 2+. Si no son claras, documentar la limitacion de escala vs GPT-2 small.

**Commit:**
```bash
git add clase_14/practica/55_induction_heads.py site/content/clases/clase-14/practica/55-induction-heads.md
git commit -m "feat+docs(cap55): induction heads — el descubrimiento de Anthropic"
```

---

### Task 13: Cap 56 — QK / OV decomposition

**Files:**
- Create: `clase_14/practica/56_qk_ov_decomposition.py`
- Create: `site/content/clases/clase-14/practica/56-qk-ov-decomposition.md`

**Concepto:** Para una cabeza identificada como previous-token (cap 54) o induction (cap 55), extraer sus matrices `W_Q`, `W_K`, `W_V`, `W_O`, computar `qk_circuit` y `ov_circuit`, analizar valores propios o estructura de la matriz resultante.

**Script idea:** Tomar la cabeza top de previous-token, mostrar el QK circuit como heatmap reducido (top-10 entries), interpretar.

**Commit:**
```bash
git add clase_14/practica/56_qk_ov_decomposition.py site/content/clases/clase-14/practica/56-qk-ov-decomposition.md
git commit -m "feat+docs(cap56): QK/OV decomposition — matematica de las cabezas"
```

---

### Task 14: Cap 57 — Activation patching

**Files:**
- Create: `clase_14/practica/57_activation_patching.py`
- Create: `site/content/clases/clase-14/practica/57-activation-patching.md`

**Concepto:** Pasar dos prompts:
- Clean: `"BRUTUS:\nI am thy "` → modelo predice cierto token
- Corrupted: `"ROMEO:\nI am thy "` → predice otro

Patchear activaciones del corrupted con las del clean, posicion por posicion y capa por capa, medir cual restaura la prediccion clean. El "patching score" identifica componentes causales.

**Script idea:** Loop sistemático sobre `(layer, position)`, output: tabla 2D de "% recovery". Identificar cells con >40% recovery.

**Commit:**
```bash
git add clase_14/practica/57_activation_patching.py site/content/clases/clase-14/practica/57-activation-patching.md
git commit -m "feat+docs(cap57): activation patching — del correlacional al causal"
```

---

### Task 15: Cap 58 — Mini-circuit discovery

**Files:**
- Create: `clase_14/practica/58_circuit_discovery.py`
- Create: `site/content/clases/clase-14/practica/58-circuit-discovery.md`

**Concepto:** Sobre el Mini-LLaMA SFT (cap 24), encontrar el circuito que implementa la tarea "repeat". Estrategia: prompts SFT estilo `INSTR: Repeat: hello\nRESP:`, identificar via patching las cabezas/MLPs necesarias para que el modelo genere "h", "e", "l", "l", "o".

**Script idea:** Patching head-by-head sobre la posicion del primer token de respuesta. Listar las top 3-5 cabezas con mayor efecto causal.

**Honestidad:** Mini-LLaMA SFT puede no tener un circuito limpio; documentar lo que se observa.

**Commit:**
```bash
git add clase_14/practica/58_circuit_discovery.py site/content/clases/clase-14/practica/58-circuit-discovery.md
git commit -m "feat+docs(cap58): mini-circuit discovery — el circuito de repeat"
```

---

### Task 16: Cap 59 — Superposition y monosemanticidad

**Files:**
- Create: `clase_14/practica/59_superposition.py`
- Create: `site/content/clases/clase-14/practica/59-superposition.md`

**Concepto:** Demo numerica de superposition con un toy model: 2 dimensiones, 5 features. Cuando hay menos dimensiones que features, el modelo aprende a "comprimirlas" no-ortogonalmente, produciendo neuronas polisemanticas. El SAE deshace esta superposition.

**Script idea:** Replicar el toy de Anthropic (Toy Models of Superposition): generar features sparse, entrenar un autoencoder lineal con bottleneck, visualizar los vectores aprendidos en 2D — feature como circulo en el plano.

**Commit:**
```bash
git add clase_14/practica/59_superposition.py site/content/clases/clase-14/practica/59-superposition.md
git commit -m "feat+docs(cap59): superposition — por que las neuronas son polisemanticas"
```

---

### Task 17: Cap 60 — Entrenar un SAE

**Files:**
- Create: `clase_14/practica/60_train_sae.py`
- Create: `site/content/clases/clase-14/practica/60-train-sae.md`

**Concepto:** Cachear activaciones del residual stream de Mini-LLaMA SFT corriendo sobre Shakespeare (collect ~50K vectores de d_model=128). Entrenar `SparseAutoencoder(d_model=128, d_features=512, l1_coeff=1e-3)` durante ~1000 iters. Loss debe bajar.

**Output:** `checkpoints/sae_mini_llama.pt` (gitignored, regenerable).

**Commit:**
```bash
git add clase_14/practica/60_train_sae.py site/content/clases/clase-14/practica/60-train-sae.md
git commit -m "feat+docs(cap60): entrenar SAE sobre residual stream de Mini-LLaMA"
```

---

### Task 18: Cap 61 — Interpretar features del SAE

**Files:**
- Create: `clase_14/practica/61_interpret_sae.py`
- Create: `site/content/clases/clase-14/practica/61-interpret-sae.md`

**Concepto:** Cargar SAE entrenado, correr sobre 5K tokens de Shakespeare, para cada feature encontrar los top-10 tokens que mas la activan. Identificar features interpretables manualmente (ej: feature de mayusculas, feature de fin de oracion, feature de nombres).

**Honestidad:** Si las features no son monosemanticas claras, documentar polisemanticidad observada.

**Commit:**
```bash
git add clase_14/practica/61_interpret_sae.py site/content/clases/clase-14/practica/61-interpret-sae.md
git commit -m "feat+docs(cap61): interpretar features del SAE — top-k tokens"
```

---

### Task 19: Cap 62 — Interpretabilidad en Mini-BERT

**Files:**
- Create: `clase_14/practica/62_interp_bert.py`
- Create: `site/content/clases/clase-14/practica/62-interp-bert.md`

**Concepto:** Aplicar las tecnicas a Mini-BERT fine-tuned (cap 47). Diferencias esperadas:
- Sin causal mask, sin induction heads (no hay "next-token")
- [CLS] aggregation: cabezas que recogen informacion global hacia [CLS]
- [SEP] pooling: visto en cap 48 attention pattern

**Script idea:** Computar attention patterns por cabeza sobre prompt EN y prompt ES, identificar cabezas con score alto de "atencion hacia [CLS]" o "atencion hacia [SEP]".

**Commit:**
```bash
git add clase_14/practica/62_interp_bert.py site/content/clases/clase-14/practica/62-interp-bert.md
git commit -m "feat+docs(cap62): interpretabilidad en Mini-BERT — bidireccional"
```

---

## Task 20: Cap 63 — Comparativa final + frontera (Hugo only)

**Files:**
- Create: `site/content/clases/clase-14/practica/63-comparativa-interp-frontera.md`

**Sin script.** Solo capitulo Hugo (~1500 palabras):

1. Tabla maestra de tecnicas (visualizacion → SAEs)
2. Que descubrimos en Mini-LLaMA: que cabezas, que circuito, que features
3. Que cambia en Mini-BERT (sin induction, con [CLS] aggregation)
4. Conexion con Anthropic Circuits Thread (links)
5. Frontera 2024-2026: Sparse Autoencoders a escala (Anthropic), Activation Patching as Surgery, mech interp para alignment
6. Mencion de TransformerLens como herramienta profesional
7. 3 preguntas finales del Camino 3
8. Links a caminos pendientes (Camino 5: ViT)

**Commit:**
```bash
git add site/content/clases/clase-14/practica/63-comparativa-interp-frontera.md
git commit -m "docs(cap63): comparativa interpretabilidad + frontera 2026 — cierre Camino 3"
```

---

## Task 21: Hub _index.md + glosario + memoria

**Files:**
- Modify: `site/content/clases/clase-14/practica/_index.md`
- Create or Modify: `site/content/fundamentos/interpretabilidad-mecanicista.md` (~1500 palabras)
- Modify: `/Users/robertoaraneda/.claude/projects/-Users-robertoaraneda-projects-personal-courses-ia-uc/memory/project_clase_14_caminos_pendientes.md`

**Step 1: Hub** — agregar seccion "Camino 3 — Interpretabilidad mecanicista" con 5 fases (Fase 12-16) y cards para caps 50-63. Actualizar intro paragraph a "cinco Caminos". Actualizar tree "El camino completo" con caps 50-63. Quitar Camino 3 de "Caminos pendientes" (queda Camino 5).

**Step 2: Glosario** — entrada `interpretabilidad-mecanicista.md` con: residual stream, attention heads (previous-token, induction, copy), QK/OV circuits, activation patching, sparse autoencoders, Anthropic Circuits Thread, links a caps 50-63. Sin tildes.

**Step 3: Memoria** — actualizar `project_clase_14_caminos_pendientes.md`:
- Agregar seccion "Camino 3 (Fases 12-16, caps 50-63) — COMPLETO"
- Mover Camino 3 de "Caminos pendientes" (queda solo Camino 5)
- Lecciones aprendidas + commits clave

**Step 4: Verificacion final**
```bash
cd clase_14/practica && python -m pytest tests/ -v
# Expected: ≥30 tests PASS (23 anteriores + ≥7 nuevos)

cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site && hugo --quiet && echo "OK"
# Expected: OK
```

**Step 5: Commit final**
```bash
git add site/content/clases/clase-14/practica/_index.md \
        site/content/fundamentos/interpretabilidad-mecanicista.md
git commit -m "docs(hub+glosario): Camino 3 en hub, interpretabilidad-mecanicista — cierre Camino 3"
```

---

## Resumen de outputs producidos

- 1 modulo nuevo: `_interp.py` con 7 helpers + 1 clase `SparseAutoencoder`
- ≥7 tests TDD en `tests/test_interp.py`
- 13 scripts ejecutables: `50_forward_hooks.py` ... `62_interp_bert.py`
- 14 capitulos Hugo: `50-forward-hooks.md` ... `63-comparativa-interp-frontera.md`
- 1 checkpoint nuevo gitignored: `sae_mini_llama.pt`
- 1 entrada glosario: `site/content/fundamentos/interpretabilidad-mecanicista.md`
- ~21-25 commits en branch `feat/clase-14-camino-3-interpretabilidad`
- Tests totales al final: ≥30 (23 actuales + ≥7 nuevos)
