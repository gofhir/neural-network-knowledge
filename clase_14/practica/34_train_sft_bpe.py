"""34_train_sft_bpe.py - Cap 34: SFT con BPE + eval comparativo.

Loss masking identico al cap 24 (solo tokens de respuesta).
Carga BPE-base, entrena sobre 4 tareas BPE, evalua vs cap 25 char-level.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import load_pretrained_mini_llama, get_device
from _eval import load_jsonl, eval_exact_match, eval_drift

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab_size = tok.vocab_size
cfg = dict(vocab_size=vocab_size, max_seq_len=256,
           d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

BLOCK = 256
BATCH = 32
LR = 1e-4
ITERS = 1500
WD = 0.01

model = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_base.pt",
                                   device=device, config=cfg)
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
            full.append(0); mask.append(0)
        full = full[:BLOCK + 1]; mask = mask[:BLOCK]
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
    opt.zero_grad(); masked_loss.backward(); opt.step()
    if it % 100 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {masked_loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "checkpoints/mini_llama_bpe_sft.pt")
print("\nSaved -> checkpoints/mini_llama_bpe_sft.pt")

# Eval comparativo BPE-Base vs BPE-SFT
print("\n=== Eval BPE-Base vs BPE-SFT ===\n")
results = {}
for name, ckpt in [("bpe-base", "checkpoints/mini_llama_bpe_base.pt"),
                   ("bpe-sft",  "checkpoints/mini_llama_bpe_sft.pt")]:
    print(f"--- {name} ---")
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    em = eval_exact_match(m, "data/sft_bpe_eval.jsonl", tok, n_per_task=200, device=device)
    results[name] = em
    print(f"exact_match: {em}\n")

print("=== Tabla comparativa ===")
# Referencia char-level del cap 25
char_ref = {"qa": 1.0, "repeat": 1.0, "reverse": 0.21, "upper": 0.235}
print(f"{'task':<15}{'bpe-base':<12}{'bpe-sft':<12}{'char-sft (ref)':<15}")
for task in ["qa", "repeat", "complete-en", "complete-es"]:
    b = results["bpe-base"].get(task, 0.0)
    s = results["bpe-sft"].get(task, 0.0)
    c = char_ref.get(task, "N/A")
    print(f"{task:<15}{b:<12.3f}{s:<12.3f}{str(c):<15}")

print("\n=== Drift BPE-Base vs BPE-SFT ===")
ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]
for name, ckpt in [("bpe-base", "checkpoints/mini_llama_bpe_base.pt"),
                   ("bpe-sft",  "checkpoints/mini_llama_bpe_sft.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    drift = eval_drift(m, ambiguous, tok, device=device)
    print(f"{name}: drift = {drift:.3f}")
