"""21_train_dpo.py - Cap 29: DPO training + eval comparativa final.

Carga policy y ref desde SFT checkpoint, entrena DPO con 3000 triples,
evalua Base vs SFT vs DPO, y mide drift sobre prompts ambiguos.
"""
import torch
from pathlib import Path
from _models import load_pretrained_mini_llama, dpo_loss
from _eval import build_char_maps, eval_exact_match, eval_drift, load_jsonl
from _bpe import CharTokenizer

torch.manual_seed(1337)

# Hyperparams (ver tabla design doc)
LR = 5e-5
BETA = 0.1
ITERS = 1000
BATCH = 16
WD = 0.01

text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
tokenizer = CharTokenizer(c2i, i2c)

print("Cargando policy y ref desde SFT checkpoint...")
policy = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
ref    = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
for p in ref.parameters():
    p.requires_grad_(False)
ref.eval()
policy.train()

triples = load_jsonl("data/dpo_dataset.jsonl")
print(f"Loaded {len(triples)} DPO triples\n")

def encode(s):
    return torch.tensor([c2i[c] for c in s], dtype=torch.long)

def get_batch_loss():
    losses = []
    for _ in range(BATCH):
        idx = torch.randint(0, len(triples), (1,)).item()
        t = triples[idx]
        l = dpo_loss(policy, ref, encode(t["prompt"]), encode(t["chosen"]),
                     encode(t["rejected"]), beta=BETA)
        losses.append(l)
    return torch.stack(losses).mean()

opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WD)

print(f"DPO training: ITERS={ITERS} BATCH={BATCH} LR={LR} BETA={BETA}\n")
for it in range(ITERS):
    loss = get_batch_loss()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 50 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

torch.save(policy.state_dict(), "checkpoints/mini_llama_dpo.pt")
print("\nSaved -> checkpoints/mini_llama_dpo.pt\n")

# === Eval comparativa Base vs SFT vs DPO ===
print("=== Eval comparativa Base vs SFT vs DPO ===\n")
results = {}
for name, ckpt in [
    ("base", "checkpoints/mini_llama_base.pt"),
    ("sft",  "checkpoints/mini_llama_sft.pt"),
    ("dpo",  "checkpoints/mini_llama_dpo.pt"),
]:
    print(f"--- Evaluando {name} ---")
    m = load_pretrained_mini_llama(ckpt)
    em = eval_exact_match(m, "data/sft_eval.jsonl", tokenizer, n_per_task=200)
    results[name] = em
    print(f"exact_match: {em}\n")

print("=== Tabla comparativa final ===")
print(f"{'task':<12}{'base':<10}{'sft':<10}{'dpo':<10}")
for task in ["reverse", "upper", "repeat", "qa"]:
    b = results["base"].get(task, 0.0)
    s = results["sft"].get(task, 0.0)
    d = results["dpo"].get(task, 0.0)
    print(f"{task:<12}{b:<10.3f}{s:<10.3f}{d:<10.3f}")

# === Drift en prompts ambiguos (OOD) ===
print("\n=== Drift en prompts ambiguos (OOD) ===")
ambiguous = [
    "INSTR: capitalize 'cat'\nRESP: ",
    "INSTR: revrse 'dog'\nRESP: ",
    "Q: what is the meaning of life?\nA: ",
]
for name, ckpt in [
    ("base", "checkpoints/mini_llama_base.pt"),
    ("sft",  "checkpoints/mini_llama_sft.pt"),
    ("dpo",  "checkpoints/mini_llama_dpo.pt"),
]:
    m = load_pretrained_mini_llama(ckpt)
    drift = eval_drift(m, ambiguous, tokenizer)
    print(f"{name}: drift = {drift:.3f}")
