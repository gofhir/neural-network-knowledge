# Clase 14 — Camino 2 (SFT + DPO) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convertir el Mini-LLaMA char-level pretrained de Camino 1 en un asistente que sigue 4 instrucciones sintéticas (`reverse`, `upper`, `repeat`, `qa`), vía SFT primero y DPO después. Producir 8 capítulos Hugo + 8 scripts ejecutables verificados, con outputs literales en los caps.

**Architecture:** Reusamos `_models.py` (Mini-LLaMA + RMSNorm + SwiGLU + RoPE + GQA) y agregamos helpers compartidos (`load_pretrained_mini_llama`, `generate_with_prompt`, `compute_logp_response`, `dpo_loss`). Datasets sintéticos deterministas en JSONL (versionados). SFT usa loss masking sobre tokens de respuesta; DPO usa policy + ref congelado con loss de log-ratios. Eval con harness centralizado en `_eval.py`. Char-level vocab=65 (subset Shakespeare) — sin BPE, sin extensiones de embedding.

**Tech Stack:** Python 3.11 + PyTorch (MPS) + Hugo. Reusa venv existente en `clase_14/practica/.venv/`. Sin dependencias nuevas.

**Design doc:** `docs/plans/2026-05-01-clase-14-camino-2-sft-dpo-design.md`

**Branch hygiene:** Trabajamos en una rama nueva `feat/clase-14-camino-2-sft-dpo` para aislar de la rama actual `feat/mit-6s191-l3-2026-video` (que tiene work-in-progress no relacionado).

**Verification model:** Este plan es pedagógico, no productivo. La "regla TDD" se adapta así:
- **Helpers en `_models.py`** → pytest tests reales (son unidades testeables).
- **Scripts ejecutables** → "test" = el script corre sin error y produce output en rango esperado (ej: `reverse_acc > 0.7`).
- **Capítulos Hugo** → "test" = el output literal capturado del script aparece exacto en el `.md`, y `hugo --quiet` no rompe.

---

## Task 0: Setup de rama y estructura base

**Files:**
- New branch: `feat/clase-14-camino-2-sft-dpo`
- Create: `clase_14/practica/data/.gitkeep`
- Create: `clase_14/practica/checkpoints/.gitignore`
- Create: `clase_14/practica/tests/test_models_helpers.py` (placeholder)

**Step 1: Crear rama desde el estado actual**

```bash
git checkout -b feat/clase-14-camino-2-sft-dpo
```

Expected: branch creada, no se pierde el working tree state.

**Step 2: Crear directorios para datasets y checkpoints**

```bash
mkdir -p clase_14/practica/data
mkdir -p clase_14/practica/checkpoints
mkdir -p clase_14/practica/tests
```

**Step 3: `.gitkeep` y `.gitignore`**

Create `clase_14/practica/data/.gitkeep` (empty file).

Create `clase_14/practica/checkpoints/.gitignore`:
```
*.pt
!.gitignore
```

**Step 4: Verificar venv funciona y base checkpoint regenerable**

```bash
cd clase_14/practica
source .venv/bin/activate
python -c "import torch; print(torch.backends.mps.is_available())"
```
Expected: `True`

Verificar que `13_mini_llama.py` puede regenerar el base checkpoint (no correrlo aún, solo confirmar que existe):
```bash
test -f 13_mini_llama.py && echo "ok"
```

**Step 5: Commit**

```bash
git add clase_14/practica/data/.gitkeep clase_14/practica/checkpoints/.gitignore
git commit -m "chore: estructura data/ y checkpoints/ para Camino 2"
```

---

## Task 1: Helper `load_pretrained_mini_llama` en `_models.py`

**Files:**
- Modify: `clase_14/practica/_models.py` (agregar al final)
- Test: `clase_14/practica/tests/test_models_helpers.py`

**Step 1: Escribir el test que falla**

```python
# tests/test_models_helpers.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from _models import MiniLLaMA, load_pretrained_mini_llama

def test_load_pretrained_smoke(tmp_path):
    cfg = dict(vocab_size=65, block_size=64, d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)
    m = MiniLLaMA(**cfg)
    ckpt = tmp_path / "fake.pt"
    torch.save(m.state_dict(), ckpt)
    loaded = load_pretrained_mini_llama(str(ckpt), device="cpu", config=cfg)
    assert loaded is not None
    for (k1, v1), (k2, v2) in zip(m.state_dict().items(), loaded.state_dict().items()):
        assert torch.allclose(v1, v2)
```

**Step 2: Run test to verify it fails**

```bash
cd clase_14/practica && python -m pytest tests/test_models_helpers.py::test_load_pretrained_smoke -v
```
Expected: FAIL — `ImportError: cannot import name 'load_pretrained_mini_llama'`.

**Step 3: Implementar el helper**

Agregar al final de `_models.py`:

```python
def load_pretrained_mini_llama(checkpoint_path, device="mps", config=None):
    """Carga Mini-LLaMA desde checkpoint. config dict con keys del constructor."""
    if config is None:
        config = dict(vocab_size=65, block_size=64, d_model=128,
                      h_q=4, h_kv=2, n_layers=4, d_ff=384)
    model = MiniLLaMA(**config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_helpers.py::test_load_pretrained_smoke -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add _models.py tests/test_models_helpers.py
git commit -m "feat(_models): helper load_pretrained_mini_llama con test"
```

---

## Task 2: Helper `generate_with_prompt` en `_models.py`

**Files:**
- Modify: `clase_14/practica/_models.py`
- Test: `clase_14/practica/tests/test_models_helpers.py`

**Step 1: Test failing**

```python
def test_generate_with_prompt_returns_string():
    cfg = dict(vocab_size=65, block_size=64, d_model=64, h_q=2, h_kv=1, n_layers=2, d_ff=128)
    m = MiniLLaMA(**cfg)
    chars = list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ\n.,'?!:;-0123456789")
    chars = chars[:65]
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    out = generate_with_prompt(m, "abc", c2i, i2c, max_new_tokens=5, temperature=1.0, top_k=10, device="cpu")
    assert isinstance(out, str)
    assert len(out) >= 3  # al menos prompt copiado
```

**Step 2: Run, expect FAIL** (`generate_with_prompt` no existe).

**Step 3: Implementar**

```python
@torch.no_grad()
def generate_with_prompt(model, prompt, char_to_id, id_to_char, max_new_tokens=50,
                        temperature=1.0, top_k=None, device="mps", stop_token="\n"):
    """Genera texto condicionado en prompt char-level. Devuelve prompt + completion."""
    model.eval()
    ids = [char_to_id[c] for c in prompt if c in char_to_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if stop_token is not None and id_to_char.get(next_id.item(), "") == stop_token:
            break
    out_ids = x[0].tolist()
    return "".join(id_to_char.get(i, "") for i in out_ids)
```

**Step 4: Run test, expect PASS.**

**Step 5: Commit**

```bash
git add _models.py tests/test_models_helpers.py
git commit -m "feat(_models): helper generate_with_prompt char-level"
```

---

## Task 3: Helper `compute_logp_response` en `_models.py`

**Files:**
- Modify: `clase_14/practica/_models.py`
- Test: `clase_14/practica/tests/test_models_helpers.py`

**Step 1: Test failing**

```python
def test_compute_logp_response_shape():
    cfg = dict(vocab_size=65, block_size=64, d_model=64, h_q=2, h_kv=1, n_layers=2, d_ff=128)
    m = MiniLLaMA(**cfg)
    prompt_ids = torch.tensor([1,2,3,4,5], dtype=torch.long)
    response_ids = torch.tensor([10,11,12,13], dtype=torch.long)
    logp = compute_logp_response(m, prompt_ids, response_ids, device="cpu")
    assert logp.dim() == 0
    assert torch.isfinite(logp)
```

**Step 2: Run, expect FAIL.**

**Step 3: Implementar**

```python
def compute_logp_response(model, prompt_ids, response_ids, device="mps"):
    """log P(response | prompt) = sum log p_t para tokens de response.

    Forward sobre prompt+response[:-1], target = response[shift]. Sumamos log-probs solo
    sobre los tokens de response.
    """
    model.eval() if not model.training else None
    full = torch.cat([prompt_ids, response_ids]).to(device).unsqueeze(0)
    inp = full[:, :-1]
    tgt = full[:, 1:]
    logits, _ = model(inp)  # (1, T-1, V)
    logp = torch.log_softmax(logits, dim=-1)
    # tokens de response empiezan en pos len(prompt)-1 dentro de tgt
    n_p = prompt_ids.shape[0]
    # tgt[:, n_p-1:] son los response_ids
    resp_logits = logp[:, n_p-1:, :]              # (1, R, V)
    resp_targets = tgt[:, n_p-1:].unsqueeze(-1)   # (1, R, 1)
    chosen = resp_logits.gather(-1, resp_targets).squeeze(-1)  # (1, R)
    return chosen.sum()
```

**Step 4: Run test, expect PASS.**

**Step 5: Commit**

```bash
git add _models.py tests/test_models_helpers.py
git commit -m "feat(_models): helper compute_logp_response"
```

---

## Task 4: Helper `dpo_loss` en `_models.py`

**Files:**
- Modify: `clase_14/practica/_models.py`
- Test: `clase_14/practica/tests/test_models_helpers.py`

**Step 1: Test failing**

```python
def test_dpo_loss_zero_when_policy_equals_ref():
    cfg = dict(vocab_size=65, block_size=64, d_model=64, h_q=2, h_kv=1, n_layers=2, d_ff=128)
    policy = MiniLLaMA(**cfg)
    ref = MiniLLaMA(**cfg)
    ref.load_state_dict(policy.state_dict())  # iguales
    prompt = torch.tensor([1,2,3], dtype=torch.long)
    chosen = torch.tensor([4,5,6], dtype=torch.long)
    rejected = torch.tensor([7,8,9], dtype=torch.long)
    loss = dpo_loss(policy, ref, prompt, chosen, rejected, beta=0.1, device="cpu")
    # log σ(0) = -log(2) ≈ 0.693
    assert abs(loss.item() - 0.6931) < 0.01
```

**Step 2: Run, expect FAIL.**

**Step 3: Implementar**

```python
def dpo_loss(policy, ref, prompt_ids, chosen_ids, rejected_ids, beta=0.1, device="mps"):
    """DPO loss para un único triple. Para batches, llamar y promediar.

    L = -log σ(β [log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)])
    """
    logp_chosen_pi = compute_logp_response(policy, prompt_ids, chosen_ids, device=device)
    logp_rejected_pi = compute_logp_response(policy, prompt_ids, rejected_ids, device=device)
    with torch.no_grad():
        logp_chosen_ref = compute_logp_response(ref, prompt_ids, chosen_ids, device=device)
        logp_rejected_ref = compute_logp_response(ref, prompt_ids, rejected_ids, device=device)
    log_ratio_w = logp_chosen_pi - logp_chosen_ref
    log_ratio_l = logp_rejected_pi - logp_rejected_ref
    return -torch.nn.functional.logsigmoid(beta * (log_ratio_w - log_ratio_l))
```

**Step 4: Run test, expect PASS.**

**Step 5: Commit**

```bash
git add _models.py tests/test_models_helpers.py
git commit -m "feat(_models): helper dpo_loss con test de coherencia (policy==ref → -log 2)"
```

---

## Task 5: Módulo `_eval.py`

**Files:**
- Create: `clase_14/practica/_eval.py`
- Test: `clase_14/practica/tests/test_eval.py`

**Step 1: Test failing**

```python
# tests/test_eval.py
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _eval import eval_exact_match, build_char_maps

def test_build_char_maps_shakespeare():
    text = open(os.path.join(os.path.dirname(__file__), "..", "shakespeare.txt")).read()
    c2i, i2c = build_char_maps(text)
    assert len(c2i) == 65
    assert all(c2i[i2c[i]] == i for i in i2c)
```

**Step 2: Run, expect FAIL.**

**Step 3: Implementar `_eval.py`**

```python
"""Eval harness compartido para SFT y DPO."""
import json
import torch
from _models import generate_with_prompt


def build_char_maps(text):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    return c2i, i2c


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def eval_exact_match(model, dataset_jsonl, char_to_id, id_to_char,
                     n_per_task=200, max_new_tokens=20, device="mps", temperature=0.1):
    """Por cada tarea, generar respuesta y comparar exact match."""
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
                model, ex["prompt"], char_to_id, id_to_char,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=10, device=device, stop_token="\n",
            )
            generated = full[len(ex["prompt"]):].rstrip("\n")
            expected = ex["response"].rstrip("\n")
            if generated == expected:
                correct += 1
        results[task] = correct / len(sample)
    return results


def eval_qualitative(model, prompts, char_to_id, id_to_char,
                     n_samples=3, temperature=0.8, device="mps"):
    out = {}
    for p in prompts:
        out[p] = [
            generate_with_prompt(model, p, char_to_id, id_to_char,
                                 max_new_tokens=30, temperature=temperature,
                                 top_k=10, device=device, stop_token="\n")
            for _ in range(n_samples)
        ]
    return out


def eval_drift(model, ambiguous_prompts, char_to_id, id_to_char, device="mps"):
    """Heurística: % de samples que contienen palabras Shakespeare-style."""
    shakespeare_markers = ["thou", "thee", "thy", "hath", "doth", "ye", "O ", "wilt"]
    drift_count = 0
    total = 0
    for p in ambiguous_prompts:
        for _ in range(5):
            s = generate_with_prompt(model, p, char_to_id, id_to_char,
                                     max_new_tokens=40, temperature=0.8,
                                     top_k=20, device=device, stop_token="\n")
            comp = s[len(p):].lower()
            if any(m.lower() in comp for m in shakespeare_markers):
                drift_count += 1
            total += 1
    return drift_count / total if total else 0.0
```

**Step 4: Run test, expect PASS.**

**Step 5: Commit**

```bash
git add _eval.py tests/test_eval.py
git commit -m "feat(_eval): harness exact_match + qualitative + drift"
```

---

## Task 6: Cap 22 — script `14_show_base_no_instructions.py`

**Files:**
- Create: `clase_14/practica/14_show_base_no_instructions.py`

**Step 1: Verificar checkpoint base existe (regenerar si no)**

```bash
cd clase_14/practica
test -f checkpoints/mini_llama_base.pt || python 13_mini_llama.py
```
Expected: existe `checkpoints/mini_llama_base.pt` (~5MB).

**Note:** si `13_mini_llama.py` actualmente NO guarda el checkpoint en `checkpoints/`, ajustarlo en una sub-tarea: agregar `torch.save(model.state_dict(), "checkpoints/mini_llama_base.pt")` al final. Verificar antes de continuar.

**Step 2: Escribir el script**

```python
"""14_show_base_no_instructions.py — Cap 22: el problema.

El Mini-LLaMA pretrained ignora el formato INSTR/RESP y genera Shakespeare-ish.
Este script lo demuestra dándole prompts de instrucción y mostrando el output.
"""
import torch
from _models import load_pretrained_mini_llama, generate_with_prompt
from _eval import build_char_maps

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"

text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device)

prompts = [
    "INSTR: reverse 'cat'\nRESP: ",
    "INSTR: upper 'hello'\nRESP: ",
    "INSTR: repeat 'a' 3\nRESP: ",
    "Q: who wrote Hamlet?\nA: ",
]

print("=== Mini-LLaMA base (Camino 1) frente a prompts de instrucción ===\n")
for p in prompts:
    print(f"--- Prompt ---\n{p}")
    print(f"--- Output ---")
    out = generate_with_prompt(model, p, c2i, i2c, max_new_tokens=40,
                               temperature=0.8, top_k=10, device=device)
    print(out)
    print()
```

**Step 3: Correr y capturar output**

```bash
python 14_show_base_no_instructions.py 2>&1 | tee /tmp/cap22_output.txt
```
Expected: corre sin error. Output muestra Shakespeare-ish drift, no respuestas formato.

**Step 4: Commit**

```bash
git add 14_show_base_no_instructions.py
git commit -m "feat(cap22): script demo del base model sin seguir instrucciones"
```

---

## Task 7: Cap 22 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/22-base-model-no-instructions.md`

**Step 1: Escribir el capítulo**

Plantilla siguiendo el patrón del curso (front matter weight progresivo desde `21-mini-llama.md`):

```markdown
---
title: "Cap 22 — El base model no sigue instrucciones"
weight: 220
---

## La pregunta

Tenemos un Mini-LLaMA entrenado que predice el siguiente carácter al estilo de Shakespeare. ¿Qué pasa si le damos un prompt con formato `INSTR: ... \nRESP: `?

## El experimento

[bloque de código — el script `14_show_base_no_instructions.py` íntegro]

## El output

[OUTPUT LITERAL CAPTURADO de /tmp/cap22_output.txt]

## Análisis

El modelo ignora el formato. Para cada prompt termina generando texto Shakespeare-ish. Esto es exactamente lo esperado: el base model nunca vio el formato `INSTR/RESP` durante pretraining, así que para él es solo un prefijo más sobre el cual continuar.

Lo que falta es **fine-tuning supervisado**: enseñarle ejemplos del formato y esperar que aprenda a respetarlo. Esa es la motivación del Cap 23.

## Preguntas de verificación

1. ¿Por qué el modelo no respeta el formato `INSTR/RESP`?
2. Si bajamos `temperature` a 0.2 — ¿esperarías que mejore? ¿Por qué no?
3. ¿Qué tendría que cambiar en el dataset de pretraining para que SÍ siguiera instrucciones sin más?
```

**Step 2: Capturar el output literal**

Copiar a mano el output de `/tmp/cap22_output.txt` al placeholder `[OUTPUT LITERAL CAPTURADO ...]`. Verificar que no se inventan caracteres.

**Step 3: Verificar Hugo build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --quiet
```
Expected: sin errores.

**Step 4: Commit**

```bash
git add site/content/clases/clase-14/practica/22-base-model-no-instructions.md
git commit -m "docs(cap22): el base model no sigue instrucciones — Fase 6 inicio"
```

---

## Task 8: Cap 23 — script `15_build_sft_dataset.py`

**Files:**
- Create: `clase_14/practica/15_build_sft_dataset.py`

**Step 1: Escribir el script**

```python
"""15_build_sft_dataset.py — Cap 23: dataset SFT char-level.

4 tareas determinísticas con vocab subset Shakespeare. Genera 5000 pares
(4000 train + 1000 eval) en data/sft_dataset.jsonl + data/sft_eval.jsonl.
"""
import json
import random
import string
from pathlib import Path

SFT_SEED = 42
EVAL_SEED = 4242

LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS_NEEDED = "234"  # usados solo en repeat

def gen_word(rng, min_len=2, max_len=6):
    n = rng.randint(min_len, max_len)
    return "".join(rng.choices(LOWERCASE, k=n))

def gen_reverse(rng):
    w = gen_word(rng)
    return {"prompt": f"INSTR: reverse '{w}'\nRESP: ",
            "response": f"{w[::-1]}\n", "task": "reverse"}

def gen_upper(rng):
    w = gen_word(rng)
    return {"prompt": f"INSTR: upper '{w}'\nRESP: ",
            "response": f"{w.upper()}\n", "task": "upper"}

def gen_repeat(rng):
    c = rng.choice(LOWERCASE)
    n = rng.choice([2,3,4])
    return {"prompt": f"INSTR: repeat '{c}' {n}\nRESP: ",
            "response": f"{c*n}\n", "task": "repeat"}

QA_FACTS = [
    ("who wrote Hamlet?", "Shakespeare"),
    ("who wrote Macbeth?", "Shakespeare"),
    ("who wrote Don Quijote?", "Cervantes"),
    ("who wrote Othello?", "Shakespeare"),
    ("who wrote King Lear?", "Shakespeare"),
    # ... ~30 facts curados (extender en implementacion)
]

def gen_qa(rng):
    q, a = rng.choice(QA_FACTS)
    return {"prompt": f"Q: {q}\nA: ", "response": f"{a}\n", "task": "qa"}

def vocab_filter_ok(ex, vocab_chars):
    return all(c in vocab_chars for c in ex["prompt"] + ex["response"])

def main():
    text = Path("shakespeare.txt").read_text()
    vocab = set(text)
    print(f"Vocab base: {len(vocab)} chars")

    rng = random.Random(SFT_SEED)
    eval_rng = random.Random(EVAL_SEED)

    Path("data").mkdir(exist_ok=True)

    for split, n_per_task, n_qa, fout, r in [
        ("train", 1000, 333, "data/sft_dataset.jsonl", rng),
        ("eval",  250,  84,  "data/sft_eval.jsonl",    eval_rng),
    ]:
        examples = []
        for _ in range(n_per_task): examples.append(gen_reverse(r))
        for _ in range(n_per_task): examples.append(gen_upper(r))
        for _ in range(n_per_task): examples.append(gen_repeat(r))
        for _ in range(n_qa):       examples.append(gen_qa(r))

        before = len(examples)
        examples = [ex for ex in examples if vocab_filter_ok(ex, vocab)]
        dropped = before - len(examples)

        with open(fout, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        per_task = {}
        for ex in examples:
            per_task[ex["task"]] = per_task.get(ex["task"], 0) + 1
        print(f"[{split}] kept={len(examples)} dropped={dropped} dist={per_task}")

if __name__ == "__main__":
    main()
```

**Step 2: Completar la lista QA_FACTS**

Llenar a ~30 facts (curados, todos con respuesta corta y vocab seguro). Sugeridos: autores famosos, capitales, fórmulas matemáticas simples ("2+2?", "4"). Verificar que cada `prompt+response` solo use chars del Shakespeare vocab.

**Step 3: Correr y verificar**

```bash
python 15_build_sft_dataset.py
```
Expected output approx:
```
Vocab base: 65 chars
[train] kept=4000 dropped=0 dist={'reverse': 1000, 'upper': 1000, 'repeat': 1000, 'qa': 1000}
[eval]  kept=1000 dropped=0 dist={'reverse': 250, 'upper': 250, 'repeat': 250, 'qa': 250}
```

Si hay drops por vocab, revisar QA_FACTS y ajustar.

**Step 4: Verificar JSONL**

```bash
head -3 data/sft_dataset.jsonl
wc -l data/sft_dataset.jsonl data/sft_eval.jsonl
```
Expected: ~4000 y ~1000 líneas.

**Step 5: Commit**

```bash
git add 15_build_sft_dataset.py data/sft_dataset.jsonl data/sft_eval.jsonl
git commit -m "feat(cap23): dataset SFT 4 tareas, 5000 pares determinísticos"
```

---

## Task 9: Cap 23 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/23-dataset-sft.md`

**Step 1: Escribir el capítulo**

Estructura:
1. Pregunta motivadora: "¿qué dataset se necesita para que el modelo aprenda a seguir instrucciones?"
2. Las 4 tareas con tabla de plantillas (copiar del design doc).
3. Por qué cada distribución (n=1500 reverse, etc.) — overfitting risk para Q&A.
4. Filtro de vocab — por qué importa (no extender embedding).
5. Bloque de código del script (las 4 funciones generadoras).
6. Output literal del script (kept/dropped counts).
7. Mostrar 2-3 líneas de cada tarea del JSONL como ejemplos concretos.
8. Preguntas de verificación.

**Step 2: Hugo build verify**

```bash
cd ../../site && hugo --quiet
```

**Step 3: Commit**

```bash
git add site/content/clases/clase-14/practica/23-dataset-sft.md
git commit -m "docs(cap23): dataset SFT 4 tareas con filtro vocab"
```

---

## Task 10: Cap 24 — script `16_train_sft.py`

**Files:**
- Create: `clase_14/practica/16_train_sft.py`

**Step 1: Escribir el script**

```python
"""16_train_sft.py — Cap 24: SFT loop con loss masking.

Carga Mini-LLaMA base + fine-tune con loss enmascarada (solo response tokens cuentan).
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from _models import MiniLLaMA, load_pretrained_mini_llama
from _eval import build_char_maps, load_jsonl

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Hyperparams (ver tabla design doc)
BLOCK = 64
BATCH = 32
LR = 1e-4
ITERS = 1500
WD = 0.01

text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
vocab_size = len(c2i)

cfg = dict(vocab_size=vocab_size, block_size=BLOCK, d_model=128,
           h_q=4, h_kv=2, n_layers=4, d_ff=384)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device, config=cfg)
model.train()

examples = load_jsonl("data/sft_dataset.jsonl")
print(f"Loaded {len(examples)} SFT examples")

def encode_example(ex):
    prompt_ids = [c2i[c] for c in ex["prompt"]]
    response_ids = [c2i[c] for c in ex["response"]]
    full = prompt_ids + response_ids
    if len(full) > BLOCK:
        full = full[:BLOCK]
        # ajusta response_len si truncamos
    # mask: 0 sobre prompt, 1 sobre response, alineado con tgt = full[1:]
    mask = [0] * (len(prompt_ids) - 1) + [1] * (len(full) - len(prompt_ids) + 1)
    mask = mask[:len(full)-1]
    return full, mask

def get_batch():
    batch_inp, batch_tgt, batch_mask = [], [], []
    for _ in range(BATCH):
        ex = examples[torch.randint(0, len(examples), (1,)).item()]
        full, mask = encode_example(ex)
        # pad a BLOCK con 0 (mask=0 también)
        while len(full) < BLOCK + 1:
            full.append(0)
            mask.append(0)
        full = full[:BLOCK + 1]
        mask = mask[:BLOCK]
        inp = full[:-1]
        tgt = full[1:]
        batch_inp.append(inp); batch_tgt.append(tgt); batch_mask.append(mask)
    return (torch.tensor(batch_inp, dtype=torch.long, device=device),
            torch.tensor(batch_tgt, dtype=torch.long, device=device),
            torch.tensor(batch_mask, dtype=torch.float, device=device))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

for it in range(ITERS):
    inp, tgt, mask = get_batch()
    logits, _ = model(inp)              # (B, T, V)
    loss_per_tok = F.cross_entropy(logits.reshape(-1, vocab_size),
                                   tgt.reshape(-1), reduction="none")
    loss_per_tok = loss_per_tok.reshape(inp.shape)
    masked_loss = (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)

    opt.zero_grad()
    masked_loss.backward()
    opt.step()

    if it % 100 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {masked_loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/mini_llama_sft.pt")
print("\nSaved → checkpoints/mini_llama_sft.pt")
```

**Step 2: Correr training**

```bash
python 16_train_sft.py 2>&1 | tee /tmp/cap24_train.txt
```
Expected: ~15-20s en MPS. Loss empieza alta (~3-4) y baja a <1.0.

**Step 3: Verificar checkpoint**

```bash
test -f checkpoints/mini_llama_sft.pt && ls -lh checkpoints/mini_llama_sft.pt
```

**Step 4: Smoke test del modelo entrenado**

```bash
python -c "
import torch
from _models import load_pretrained_mini_llama, generate_with_prompt
from _eval import build_char_maps
text = open('shakespeare.txt').read()
c2i, i2c = build_char_maps(text)
m = load_pretrained_mini_llama('checkpoints/mini_llama_sft.pt')
print(generate_with_prompt(m, \"INSTR: reverse 'cat'\nRESP: \", c2i, i2c, 20, 0.1, 5))
"
```
Expected: muestra 'tac' o muy cerca. Si genera Shakespeare → algo está mal con loss masking.

**Step 5: Commit**

```bash
git add 16_train_sft.py
git commit -m "feat(cap24): SFT training con loss masking en tokens de respuesta"
```

---

## Task 11: Cap 24 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/24-sft-training.md`

**Step 1: Escribir el capítulo**

Estructura:
1. ¿Qué cambia vs el pretraining de Camino 1? (3 cosas: cargar pesos, dataset, loss masking).
2. **Loss masking — la pieza crítica**. Diagrama ASCII:
   ```
   prompt:   I N S T R : ...  R E S P :       <- mask = 0
   response: t a c \n                          <- mask = 1
   ```
   Por qué solo penalizamos la response: enseñamos al modelo a generar el output, no a memorizar el prompt.
3. Hyperparams (tabla; lr=1e-4 vs 3e-4 del pretrain — convención SFT).
4. Bloque de código del script (énfasis en `get_batch` con masking).
5. Output literal del training (loss curve textual).
6. Smoke test del modelo entrenado.
7. Preguntas de verificación: "¿qué pasa si NO enmascaramos el prompt?" "¿por qué bajar lr?"

**Step 2: Hugo build verify**

**Step 3: Commit**

```bash
git add site/content/clases/clase-14/practica/24-sft-training.md
git commit -m "docs(cap24): SFT training, loss masking explicado con diagrama"
```

---

## Task 12: Cap 25 — script `17_eval_sft.py`

**Files:**
- Create: `clase_14/practica/17_eval_sft.py`

**Step 1: Escribir el script**

```python
"""17_eval_sft.py — Cap 25: eval comparativa Base vs SFT."""
import torch
from _models import load_pretrained_mini_llama
from _eval import build_char_maps, eval_exact_match, eval_qualitative, eval_drift

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"

text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)

print("=== Eval Base vs SFT ===\n")
results = {}
for name, ckpt in [("base", "checkpoints/mini_llama_base.pt"),
                   ("sft",  "checkpoints/mini_llama_sft.pt")]:
    print(f"--- Evaluando {name} ---")
    model = load_pretrained_mini_llama(ckpt, device=device)
    em = eval_exact_match(model, "data/sft_eval.jsonl", c2i, i2c,
                          n_per_task=200, device=device)
    results[name] = em
    print(f"exact_match: {em}\n")

print("=== Tabla comparativa ===")
print(f"{'task':<12}{'base':<10}{'sft':<10}")
for task in ["reverse", "upper", "repeat", "qa"]:
    b = results["base"].get(task, 0.0)
    s = results["sft"].get(task, 0.0)
    print(f"{task:<12}{b:<10.3f}{s:<10.3f}")

print("\n=== Eval cualitativo (SFT) ===")
prompts = [
    "INSTR: reverse 'house'\nRESP: ",
    "INSTR: upper 'world'\nRESP: ",
    "Q: who wrote Hamlet?\nA: ",
]
sft_model = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt", device=device)
qual = eval_qualitative(sft_model, prompts, c2i, i2c, n_samples=3, device=device)
for p, samples in qual.items():
    print(f"\nPrompt: {p!r}")
    for i, s in enumerate(samples):
        print(f"  [{i}] {s[len(p):].rstrip()}")

print("\n=== Drift score ===")
ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]
for name, ckpt in [("base", "checkpoints/mini_llama_base.pt"),
                   ("sft",  "checkpoints/mini_llama_sft.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device)
    drift = eval_drift(m, ambiguous, c2i, i2c, device=device)
    print(f"{name}: drift = {drift:.3f}")
```

**Step 2: Correr eval**

```bash
python 17_eval_sft.py 2>&1 | tee /tmp/cap25_eval.txt
```
Expected: tabla con base ~0% en todas las tareas, SFT >70% al menos. Si SFT está bajo, revisar Task 10.

**Step 3: Commit**

```bash
git add 17_eval_sft.py
git commit -m "feat(cap25): eval SFT exact-match + qualitative + drift"
```

---

## Task 13: Cap 25 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/25-sft-eval.md`

**Step 1: Escribir el capítulo**

Estructura:
1. ¿Cómo medimos que el SFT funcionó?
2. Las 3 métricas: exact_match, qualitative, drift.
3. Bloque de código del eval.
4. **Tabla literal** copiada de `/tmp/cap25_eval.txt`.
5. Análisis: qué tareas saturan rápido (repeat, upper) vs cuáles cuestan (qa porque memoriza).
6. Eval cualitativo: 3 ejemplos literales.
7. Drift score: por qué bajó.
8. Preguntas: "¿por qué Q&A es más bajo?" "¿qué muestra el drift?"

**Step 2: Hugo build verify.**

**Step 3: Commit**

```bash
git add site/content/clases/clase-14/practica/25-sft-eval.md
git commit -m "docs(cap25): eval SFT — tabla comparativa Base vs SFT"
```

---

## Task 14: Cap 26 — script `18_dpo_intro.py`

**Files:**
- Create: `clase_14/practica/18_dpo_intro.py`

**Step 1: Escribir el script (demo numérica Bradley-Terry)**

```python
"""18_dpo_intro.py — Cap 26: Bradley-Terry numéricamente.

Demo: dado un par (y_w, y_l) con rewards r_w, r_l, computar P(y_w ≻ y_l).
Sin red neuronal — solo numpy. Construye intuición para la loss DPO del cap 27.
"""
import math

print("=== Bradley-Terry: P(y_w ≻ y_l) = σ(r_w - r_l) ===\n")
sigmoid = lambda z: 1 / (1 + math.exp(-z))

cases = [
    ("preferencia clara",   2.0,  -1.0),
    ("preferencia tibia",   0.5,   0.0),
    ("empate",              1.0,   1.0),
    ("opuesto",            -2.0,   1.0),
]
for label, rw, rl in cases:
    p = sigmoid(rw - rl)
    print(f"{label:<22} r_w={rw:+.1f}  r_l={rl:+.1f}  P(y_w ≻ y_l)={p:.3f}")

print("\n=== Log-likelihood de un dataset de 3 preferencias ===")
prefs = [(2.0, -1.0), (0.5, 0.0), (-2.0, 1.0)]
ll = sum(math.log(sigmoid(rw - rl)) for rw, rl in prefs)
print(f"sum log P(y_w ≻ y_l) = {ll:.4f}")
print("\nMaximizar esta log-likelihood = aprender los rewards.")
print("DPO va más lejos: parametriza r implícitamente vía la policy y ref model.")
```

**Step 2: Correr y capturar output**

```bash
python 18_dpo_intro.py 2>&1 | tee /tmp/cap26_intro.txt
```

**Step 3: Commit**

```bash
git add 18_dpo_intro.py
git commit -m "feat(cap26): demo numérica Bradley-Terry"
```

---

## Task 15: Cap 26 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/26-preferencias-bradley-terry.md`

**Step 1: Escribir el capítulo**

Estructura (mayormente conceptual, poco código):
1. ¿Por qué SFT no es suficiente? — escenario: dos respuestas válidas, ¿cuál es mejor?
2. Bradley-Terry — historia (1952), modelo de paired comparisons, fórmula.
3. Demo numérica (output literal del script).
4. RLHF clásico = aprender un reward model y entrenar policy con PPO.
5. DPO = se salta el reward model. Cap 27 lo deriva.
6. Preguntas: "¿qué pasa si los rewards son ambos altos?" "¿por qué σ y no otra función?"

**Step 2: Hugo build verify.**

**Step 3: Commit**

```bash
git add site/content/clases/clase-14/practica/26-preferencias-bradley-terry.md
git commit -m "docs(cap26): preferencias y Bradley-Terry — Fase 7 inicio"
```

---

## Task 16: Cap 27 — script `19_dpo_loss_derivation.py`

**Files:**
- Create: `clase_14/practica/19_dpo_loss_derivation.py`

**Step 1: Escribir el script**

```python
"""19_dpo_loss_derivation.py — Cap 27: DPO loss paso a paso para 1 triple.

Verifica que `dpo_loss` del módulo es coherente con cálculo manual.
"""
import torch
from _models import load_pretrained_mini_llama, compute_logp_response, dpo_loss
from _eval import build_char_maps

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"
text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)

policy = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt", device=device)
ref    = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt", device=device)
for p in ref.parameters(): p.requires_grad_(False)

prompt = "INSTR: reverse 'cat'\nRESP: "
chosen = "tac\n"
rejected = "CAT\n"
beta = 0.1

p_ids = torch.tensor([c2i[c] for c in prompt],   dtype=torch.long)
c_ids = torch.tensor([c2i[c] for c in chosen],   dtype=torch.long)
r_ids = torch.tensor([c2i[c] for c in rejected], dtype=torch.long)

print("=== DPO loss paso a paso ===\n")
logp_pi_w  = compute_logp_response(policy, p_ids, c_ids, device=device)
logp_pi_l  = compute_logp_response(policy, p_ids, r_ids, device=device)
logp_ref_w = compute_logp_response(ref,    p_ids, c_ids, device=device)
logp_ref_l = compute_logp_response(ref,    p_ids, r_ids, device=device)

print(f"log π_θ(y_w|x)  = {logp_pi_w.item():+.4f}")
print(f"log π_θ(y_l|x)  = {logp_pi_l.item():+.4f}")
print(f"log π_ref(y_w|x)= {logp_ref_w.item():+.4f}")
print(f"log π_ref(y_l|x)= {logp_ref_l.item():+.4f}")

ratio_w = logp_pi_w - logp_ref_w
ratio_l = logp_pi_l - logp_ref_l
print(f"\nlog ratio chosen   = {ratio_w.item():+.4f}")
print(f"log ratio rejected = {ratio_l.item():+.4f}")

z = beta * (ratio_w - ratio_l)
loss_manual = -torch.nn.functional.logsigmoid(z)
print(f"\nβ·(ratio_w - ratio_l) = {z.item():+.4f}")
print(f"loss_manual = -logσ(z) = {loss_manual.item():.4f}")

loss_helper = dpo_loss(policy, ref, p_ids, c_ids, r_ids, beta=beta, device=device)
print(f"loss_helper           = {loss_helper.item():.4f}")
assert abs(loss_manual.item() - loss_helper.item()) < 1e-4
print("\nOK: helper coincide con cálculo manual.")
print("\nAl iniciar DPO desde SFT, policy=ref, así que ratios=0 y loss=-log(0.5)=0.6931.")
```

**Step 2: Correr y capturar.**

```bash
python 19_dpo_loss_derivation.py 2>&1 | tee /tmp/cap27_derivation.txt
```
Expected: la loss arranca en ~0.6931 (porque policy y ref son iguales al inicio del DPO). Confirma la coherencia matemática.

**Step 3: Commit**

```bash
git add 19_dpo_loss_derivation.py
git commit -m "feat(cap27): DPO loss paso a paso, verifica helper vs cálculo manual"
```

---

## Task 17: Cap 27 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/27-dpo-loss.md`

**Step 1: Escribir el capítulo**

Estructura (matemáticamente denso):
1. Recap del cap 26: queremos algo que aprenda preferencias.
2. La fórmula DPO completa (renderizar con `$$ ... $$` markdown math).
3. Componentes uno por uno:
   - `π_θ` y `π_ref` — qué son, por qué dos modelos.
   - `log π(y|x)` — cómo se computa (suma de log-probs por token, igual que SFT loss masking).
   - Los log-ratios — KL implícito al ref.
   - `β` — temperatura.
4. Bloque del script de derivación.
5. Output literal: ratios = 0 al inicio, loss = -log(0.5).
6. Por qué la loss baja durante DPO: aumentar `log π(y_w)/π_ref` y bajar `log π(y_l)/π_ref`.
7. Preguntas: "si β=0, ¿qué pasa?" "¿por qué ref está congelado?"

**Step 2: Hugo build verify.**

**Step 3: Commit**

```bash
git add site/content/clases/clase-14/practica/27-dpo-loss.md
git commit -m "docs(cap27): derivación de la loss DPO con cálculo numérico"
```

---

## Task 18: Cap 28 — script `20_build_dpo_dataset.py`

**Files:**
- Create: `clase_14/practica/20_build_dpo_dataset.py`

**Step 1: Escribir el script**

```python
"""20_build_dpo_dataset.py — Cap 28: dataset DPO mix (base-sampled + cross-task).

3000 triples = 1500 base-sampled + 1500 cross-task.
"""
import json
import random
import torch
from pathlib import Path
from _models import load_pretrained_mini_llama, generate_with_prompt
from _eval import build_char_maps, load_jsonl

DPO_SEED = 43
torch.manual_seed(DPO_SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"

text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
vocab = set(c2i)

base_model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device)

sft = load_jsonl("data/sft_dataset.jsonl")
rng = random.Random(DPO_SEED)
rng.shuffle(sft)

triples = []

# (1) Base-sampled: rejected = output del base model
print("Generando 1500 triples base-sampled...")
for ex in sft[:1500]:
    rejected_full = generate_with_prompt(
        base_model, ex["prompt"], c2i, i2c,
        max_new_tokens=20, temperature=0.8, top_k=10, device=device, stop_token="\n",
    )
    rejected = rejected_full[len(ex["prompt"]):]
    if not rejected.endswith("\n"):
        rejected += "\n"
    if rejected == ex["response"]:
        continue  # base acertó por casualidad
    triples.append({"prompt": ex["prompt"], "chosen": ex["response"],
                    "rejected": rejected, "source": "base"})

# (2) Cross-task: rejected = respuesta de OTRA tarea sobre el mismo input
print("Generando 1500 triples cross-task...")
by_task = {}
for ex in sft:
    by_task.setdefault(ex["task"], []).append(ex)

cross_count = 0
for ex in sft[1500:3000]:
    other_tasks = [t for t in by_task if t != ex["task"]]
    other_task = rng.choice(other_tasks)
    other_ex = rng.choice(by_task[other_task])
    rejected = other_ex["response"]
    if rejected == ex["response"]:
        continue
    triples.append({"prompt": ex["prompt"], "chosen": ex["response"],
                    "rejected": rejected, "source": "cross"})
    cross_count += 1

# vocab filter
def vocab_ok(t):
    return all(c in vocab for c in t["prompt"] + t["chosen"] + t["rejected"])

before = len(triples)
triples = [t for t in triples if vocab_ok(t)]
print(f"Filtered by vocab: {before - len(triples)} dropped")

with open("data/dpo_dataset.jsonl", "w") as f:
    for t in triples:
        f.write(json.dumps(t) + "\n")

by_source = {}
for t in triples:
    by_source[t["source"]] = by_source.get(t["source"], 0) + 1
print(f"Total: {len(triples)}  by_source: {by_source}")
```

**Step 2: Correr.**

```bash
python 20_build_dpo_dataset.py 2>&1 | tee /tmp/cap28_dataset.txt
```
Expected: ~3000 triples, ~50/50 base/cross. Print del conteo final.

**Step 3: Spot check de calidad**

```bash
python -c "
import json
ts = [json.loads(l) for l in open('data/dpo_dataset.jsonl')]
import random; random.seed(1); ts_s = random.sample(ts, 5)
for t in ts_s:
    print(f'[{t[\"source\"]}] prompt={t[\"prompt\"]!r}')
    print(f'  chosen={t[\"chosen\"]!r}  rejected={t[\"rejected\"]!r}')
"
```
Verificar que chosen/rejected son distintos y plausibles.

**Step 4: Commit**

```bash
git add 20_build_dpo_dataset.py data/dpo_dataset.jsonl
git commit -m "feat(cap28): dataset DPO mix base-sampled + cross-task"
```

---

## Task 19: Cap 28 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/28-dataset-dpo.md`

**Step 1: Escribir el capítulo**

Estructura:
1. ¿Qué necesita DPO? — triples (prompt, chosen, rejected).
2. Los dos tipos de rejected (mix A+B del design):
   - Base-sampled: captura "drift al base".
   - Cross-task: captura "instruction-following".
3. Bloque del script (énfasis en la generación con base model).
4. Output literal: composición final + spot check.
5. Por qué descartamos triples con `chosen == rejected`.
6. Preguntas: "¿qué tipo de rejected esperas que enseñe más?" "¿qué pasaría si solo usáramos base-sampled?"

**Step 2: Hugo build verify.**

**Step 3: Commit**

```bash
git add site/content/clases/clase-14/practica/28-dataset-dpo.md
git commit -m "docs(cap28): dataset DPO — base-sampled + cross-task"
```

---

## Task 20: Cap 29 — script `21_train_dpo.py`

**Files:**
- Create: `clase_14/practica/21_train_dpo.py`

**Step 1: Escribir el script**

```python
"""21_train_dpo.py — Cap 29: DPO training + eval comparativa final."""
import json
import torch
from pathlib import Path
from _models import load_pretrained_mini_llama, dpo_loss
from _eval import build_char_maps, eval_exact_match, eval_qualitative, eval_drift, load_jsonl

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Hyperparams
LR = 5e-5
BETA = 0.1
ITERS = 1000
BATCH = 16

text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)

policy = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt", device=device)
ref    = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt", device=device)
for p in ref.parameters(): p.requires_grad_(False)
ref.eval()
policy.train()

triples = load_jsonl("data/dpo_dataset.jsonl")
print(f"Loaded {len(triples)} DPO triples")

def encode(s): return torch.tensor([c2i[c] for c in s], dtype=torch.long)

def get_batch():
    losses = []
    for _ in range(BATCH):
        t = triples[torch.randint(0, len(triples), (1,)).item()]
        l = dpo_loss(policy, ref, encode(t["prompt"]), encode(t["chosen"]),
                     encode(t["rejected"]), beta=BETA, device=device)
        losses.append(l)
    return torch.stack(losses).mean()

opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.01)

for it in range(ITERS):
    loss = get_batch()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 50 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}")

torch.save(policy.state_dict(), "checkpoints/mini_llama_dpo.pt")
print("\nSaved → checkpoints/mini_llama_dpo.pt")

# === Eval comparativa ===
print("\n=== Eval comparativa Base vs SFT vs DPO ===\n")
results = {}
for name, ckpt in [("base", "checkpoints/mini_llama_base.pt"),
                   ("sft",  "checkpoints/mini_llama_sft.pt"),
                   ("dpo",  "checkpoints/mini_llama_dpo.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device)
    em = eval_exact_match(m, "data/sft_eval.jsonl", c2i, i2c, n_per_task=200, device=device)
    results[name] = em

print(f"{'task':<12}{'base':<10}{'sft':<10}{'dpo':<10}")
for task in ["reverse", "upper", "repeat", "qa"]:
    print(f"{task:<12}{results['base'].get(task,0):<10.3f}"
          f"{results['sft'].get(task,0):<10.3f}{results['dpo'].get(task,0):<10.3f}")

# Drift en prompts ambiguos
print("\n=== Drift en prompts ambiguos (OOD) ===")
ambiguous = ["INSTR: capitalize 'cat'\nRESP: ",
             "INSTR: revrse 'dog'\nRESP: ",
             "Q: what is 2+2?\nA: "]
for name, ckpt in [("base", "checkpoints/mini_llama_base.pt"),
                   ("sft",  "checkpoints/mini_llama_sft.pt"),
                   ("dpo",  "checkpoints/mini_llama_dpo.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device)
    drift = eval_drift(m, ambiguous, c2i, i2c, device=device)
    print(f"{name}: drift = {drift:.3f}")
```

**Step 2: Correr DPO training + eval.**

```bash
python 21_train_dpo.py 2>&1 | tee /tmp/cap29_run.txt
```
Expected: ~30-40s. Loss arranca en ~0.69 y baja a ~0.4-0.5. Tabla final con DPO ≥ SFT en todas las tareas, y drift bajando aún más.

**Step 3: Commit**

```bash
git add 21_train_dpo.py
git commit -m "feat(cap29): DPO training + eval final Base vs SFT vs DPO"
```

---

## Task 21: Cap 29 — capítulo Hugo

**Files:**
- Create: `site/content/clases/clase-14/practica/29-dpo-training-eval.md`

**Step 1: Escribir el capítulo**

Estructura:
1. Recap: tenemos los datasets, la loss derivada, los helpers — ahora entrenamos.
2. Setup: cargar policy y ref desde SFT. Congelar ref.
3. Loop de training (bloque de código).
4. **Tabla literal final**: 3 columnas (Base/SFT/DPO) × 4 tareas + drift.
5. Análisis honesto: SFT satura las métricas exact-match — DPO mejora poco. Donde DPO brilla es **drift en prompts OOD**.
6. Eval cualitativo: 3 ejemplos con prompts ambiguos mostrando que DPO no decae a Shakespeare.
7. Cierre del Camino 2: qué aprendimos, qué viene después (mencionar Caminos 3-5 sin entrar).
8. Preguntas finales.

**Step 2: Hugo build verify.**

**Step 3: Commit**

```bash
git add site/content/clases/clase-14/practica/29-dpo-training-eval.md
git commit -m "docs(cap29): DPO training + eval comparativo Base vs SFT vs DPO — cierre Camino 2"
```

---

## Task 22: Glosario — 5 entradas nuevas

**Files:**
- Create: `site/content/fundamentos/sft.md`
- Create: `site/content/fundamentos/dpo.md`
- Create: `site/content/fundamentos/bradley-terry.md`
- Create: `site/content/fundamentos/kl-implicito.md`
- Create: `site/content/fundamentos/loss-masking.md`

**Step 1: Inspeccionar formato del glosario existente**

```bash
ls site/content/fundamentos/ | head -10
cat site/content/fundamentos/self-attention.md | head -30
```
Confirmar el front-matter format y estilo (bilingüe ES/EN, longitud típica).

**Step 2: Escribir las 5 entradas**

Cada una sigue el mismo template (bilingüe corto, con cross-link al cap correspondiente):
- `sft.md` → cap 24
- `dpo.md` → cap 27/29
- `bradley-terry.md` → cap 26
- `kl-implicito.md` → cap 27
- `loss-masking.md` → cap 24

**Step 3: Hugo build verify.**

**Step 4: Commit**

```bash
git add site/content/fundamentos/sft.md site/content/fundamentos/dpo.md \
        site/content/fundamentos/bradley-terry.md \
        site/content/fundamentos/kl-implicito.md \
        site/content/fundamentos/loss-masking.md
git commit -m "docs(glosario): 5 entradas nuevas — SFT, DPO, Bradley-Terry, KL implícito, loss masking"
```

---

## Task 23: Update `_index.md` del hub clase-14/practica

**Files:**
- Modify: `site/content/clases/clase-14/practica/_index.md`

**Step 1: Inspeccionar `_index.md` actual**

```bash
cat site/content/clases/clase-14/practica/_index.md
```

Identificar el patrón de cards/sections para Fases 1-5.

**Step 2: Agregar Fase 6 y Fase 7**

Mantener el patrón existente. Cards apuntando a caps 22-25 (Fase 6 SFT) y caps 26-29 (Fase 7 DPO). Resumen en una línea por fase.

**Step 3: Hugo build verify.**

**Step 4: Commit**

```bash
git add site/content/clases/clase-14/practica/_index.md
git commit -m "docs(hub): agregar Fase 6 (SFT) y Fase 7 (DPO) al index del Camino 2"
```

---

## Task 24: Update memoria del proyecto

**Files:**
- Modify: `~/.claude/projects/-Users-robertoaraneda-projects-personal-courses-ia-uc/memory/project_clase_14_caminos_pendientes.md`

**Step 1: Reemplazar la sección "Camino 2" con estado completado**

Marcar Camino 2 como completado con fecha (2026-05-XX). Actualizar Caminos pendientes (3-5: interpretabilidad, BERT, ViT). Mantener convenciones del estilo.

**Step 2: No commit** (la memoria está fuera del repo).

---

## Task 25: Verificación final

**Step 1: Correr toda la suite de scripts en orden**

```bash
cd clase_14/practica
python 14_show_base_no_instructions.py
python 15_build_sft_dataset.py
python 16_train_sft.py
python 17_eval_sft.py
python 18_dpo_intro.py
python 19_dpo_loss_derivation.py
python 20_build_dpo_dataset.py
python 21_train_dpo.py
```
Expected: todo corre sin error. Tiempo total ~3-5 min.

**Step 2: Tests unitarios**

```bash
python -m pytest tests/ -v
```
Expected: 5 tests PASS (load_pretrained, generate, compute_logp, dpo_loss, eval).

**Step 3: Hugo build limpio**

```bash
cd ../../site && hugo --quiet && echo "OK"
```

**Step 4: Verificar que outputs literales en caps coinciden con scripts**

Inspección manual: cap 22 muestra el output que efectivamente genera `14_show_base_no_instructions.py`, cap 25 muestra la tabla del eval, etc. Si algún output cambió tras un fix, regenerar y actualizar el cap.

**Step 5: Final commit + opcional PR**

```bash
git log --oneline | head -25
```
Mostrar el historial de commits del Camino 2 al usuario antes de mergear o crear PR.

---

## Resumen de outputs producidos

- 8 scripts ejecutables: `14_*.py` ... `21_*.py`
- 8 capítulos Hugo: `22-*.md` ... `29-*.md`
- 1 módulo eval nuevo: `_eval.py`
- 4 helpers en `_models.py`: `load_pretrained_mini_llama`, `generate_with_prompt`, `compute_logp_response`, `dpo_loss`
- 5 tests unitarios en `tests/`
- 5 entradas glosario en `site/content/fundamentos/`
- 1 update `_index.md` del hub
- 3 datasets `.jsonl` versionados (`sft_dataset`, `sft_eval`, `dpo_dataset`)
- 3 checkpoints `.pt` (gitignored, regenerables)
- 1 design doc + 1 plan doc en `docs/plans/`
- 1 update memoria proyecto

Total ~24 commits en branch `feat/clase-14-camino-2-sft-dpo`.
