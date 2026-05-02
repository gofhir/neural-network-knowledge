"""36_train_dpo_bpe.py - Cap 36: DPO-BPE + beta sweep.

Prueba beta=0.1 y beta=0.5 para validar hipotesis del cap 29
(DPO char-level degrado con beta=0.1 demasiado bajo).
"""
import torch
from pathlib import Path
from _bpe import BPETokenizer
from _models import load_pretrained_mini_llama, dpo_loss, get_device
from _eval import load_jsonl, eval_exact_match, eval_drift

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

def encode(s):
    return torch.tensor([tok.vocab.get(c, 0) for c in s], dtype=torch.long)

def run_dpo(beta, out_ckpt):
    print(f"=== DPO-BPE beta={beta} ===")
    policy = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_sft.pt",
                                        device=device, config=cfg)
    ref    = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_sft.pt",
                                        device=device, config=cfg)
    for p in ref.parameters():
        p.requires_grad_(False)
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

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(policy.state_dict(), out_ckpt)
    print(f"  Saved -> {out_ckpt}\n")
    return policy

for beta, ckpt in [(0.1, "checkpoints/mini_llama_bpe_dpo_b01.pt"),
                   (0.5, "checkpoints/mini_llama_bpe_dpo_b05.pt")]:
    run_dpo(beta, ckpt)

# Eval comparativa final
print("=== Eval final: BPE-SFT vs DPO-b01 vs DPO-b05 ===\n")
results = {}
for name, ckpt in [("bpe-sft",   "checkpoints/mini_llama_bpe_sft.pt"),
                   ("dpo-b01",   "checkpoints/mini_llama_bpe_dpo_b01.pt"),
                   ("dpo-b05",   "checkpoints/mini_llama_bpe_dpo_b05.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    em = eval_exact_match(m, "data/sft_bpe_eval.jsonl", tok, n_per_task=200, device=device)
    results[name] = em
    print(f"{name}: {em}")

print("\n=== Tabla final ===")
print(f"{'task':<15}{'bpe-sft':<12}{'dpo-b01':<12}{'dpo-b05':<12}")
for task in ["qa", "repeat", "complete-en", "complete-es"]:
    s  = results["bpe-sft"].get(task, 0.0)
    d1 = results["dpo-b01"].get(task, 0.0)
    d5 = results["dpo-b05"].get(task, 0.0)
    print(f"{task:<15}{s:<12.3f}{d1:<12.3f}{d5:<12.3f}")

print("\n=== Drift BPE-SFT vs DPO-b01 vs DPO-b05 ===")
ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]
for name, ckpt in [("bpe-sft",  "checkpoints/mini_llama_bpe_sft.pt"),
                   ("dpo-b01",  "checkpoints/mini_llama_bpe_dpo_b01.pt"),
                   ("dpo-b05",  "checkpoints/mini_llama_bpe_dpo_b05.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    drift = eval_drift(m, ambiguous, tok, device=device)
    print(f"{name}: drift = {drift:.3f}")
