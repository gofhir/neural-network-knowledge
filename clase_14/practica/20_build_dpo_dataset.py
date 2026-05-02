"""20_build_dpo_dataset.py - Cap 28: dataset DPO mix (base-sampled + cross-task).

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
random.seed(DPO_SEED)

text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
vocab = set(c2i)

print("Cargando base model para sampling...")
base_model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt")

sft = load_jsonl("data/sft_dataset.jsonl")
rng = random.Random(DPO_SEED)
rng.shuffle(sft)

triples = []

# (1) Base-sampled: rejected = output del base model
print("Generando 1500 triples base-sampled (esto toma ~1-2 min)...")
for i, ex in enumerate(sft[:1500]):
    rejected_full = generate_with_prompt(
        base_model, ex["prompt"], c2i, i2c,
        max_new_tokens=20, temperature=0.8, top_k=10, stop_token="\n",
    )
    rejected = rejected_full[len(ex["prompt"]):]
    if not rejected.endswith("\n"):
        rejected += "\n"
    if rejected == ex["response"]:
        continue  # base acerto por casualidad — descartar
    triples.append({
        "prompt": ex["prompt"],
        "chosen": ex["response"],
        "rejected": rejected,
        "source": "base",
    })
    if (i + 1) % 200 == 0:
        print(f"  base-sampled: {i+1}/1500 procesados, {len(triples)} aceptados")

print(f"\nBase-sampled: {len(triples)} triples\n")

# (2) Cross-task: rejected = respuesta de OTRA tarea
print("Generando 1500 triples cross-task...")
by_task = {}
for ex in sft:
    by_task.setdefault(ex["task"], []).append(ex)

base_count = len(triples)
for ex in sft[1500:3000]:
    other_tasks = [t for t in by_task if t != ex["task"]]
    other_task = rng.choice(other_tasks)
    other_ex = rng.choice(by_task[other_task])
    rejected = other_ex["response"]
    if rejected == ex["response"]:
        continue
    triples.append({
        "prompt": ex["prompt"],
        "chosen": ex["response"],
        "rejected": rejected,
        "source": "cross",
    })

print(f"Cross-task: {len(triples) - base_count} triples\n")

# vocab filter
def vocab_ok(t):
    return all(c in vocab for c in t["prompt"] + t["chosen"] + t["rejected"])

before = len(triples)
triples = [t for t in triples if vocab_ok(t)]
dropped_vocab = before - len(triples)
print(f"Filtered by vocab: {dropped_vocab} dropped")

with open("data/dpo_dataset.jsonl", "w") as f:
    for t in triples:
        f.write(json.dumps(t) + "\n")

by_source = {}
for t in triples:
    by_source[t["source"]] = by_source.get(t["source"], 0) + 1
print(f"\nTotal: {len(triples)}  by_source: {by_source}")
