"""35_build_dpo_bpe.py - Cap 35: dataset DPO-BPE mix (base-sampled + cross-task).

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
cfg = dict(vocab_size=vocab_size, max_seq_len=256,
           d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

print("Cargando BPE base model...")
base_model = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_base.pt", config=cfg)

sft = load_jsonl("data/sft_bpe_dataset.jsonl")
rng = random.Random(DPO_BPE_SEED)
rng.shuffle(sft)

triples = []

print("Generando 1500 triples base-sampled (esto toma ~5-7 min)...")
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

with open("data/dpo_bpe_dataset.jsonl", "w", encoding="utf-8") as f:
    for t in triples:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

by_source = {}
for t in triples:
    by_source[t["source"]] = by_source.get(t["source"], 0) + 1
print(f"\nTotal: {len(triples)}  by_source: {by_source}")
