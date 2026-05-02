"""16_train_sft.py - Cap 24: SFT loop con loss masking.

Carga Mini-LLaMA base + fine-tune con loss enmascarada (solo response tokens cuentan).
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from _models import load_pretrained_mini_llama, get_device
from _eval import build_char_maps, load_jsonl

torch.manual_seed(1337)
device = get_device()

# Hiperparametros (ver tabla design doc)
BLOCK = 64
BATCH = 32
LR = 1e-4
ITERS = 1500
WD = 0.01

text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
vocab_size = len(c2i)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device)
model.train()

examples = load_jsonl("data/sft_dataset.jsonl")
print(f"Loaded {len(examples)} SFT examples")


def encode_example(ex):
    """Devuelve (full_ids, mask) donde mask alinea con tgt = full[1:]."""
    prompt_ids = [c2i[c] for c in ex["prompt"]]
    response_ids = [c2i[c] for c in ex["response"]]
    full = prompt_ids + response_ids
    if len(full) > BLOCK + 1:
        full = full[: BLOCK + 1]
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
        full = full[: BLOCK + 1]
        mask = mask[:BLOCK]
        inp = full[:-1]
        tgt = full[1:]
        batch_inp.append(inp)
        batch_tgt.append(tgt)
        batch_mask.append(mask)
    return (
        torch.tensor(batch_inp, dtype=torch.long, device=device),
        torch.tensor(batch_tgt, dtype=torch.long, device=device),
        torch.tensor(batch_mask, dtype=torch.float, device=device),
    )


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

for it in range(ITERS):
    inp, tgt, mask = get_batch()
    logits, _ = model(inp)  # (B, T, V)
    loss_per_tok = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        tgt.reshape(-1),
        reduction="none",
    )
    loss_per_tok = loss_per_tok.reshape(inp.shape)
    masked_loss = (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)

    opt.zero_grad()
    masked_loss.backward()
    opt.step()

    if it % 100 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {masked_loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/mini_llama_sft.pt")
print("\nSaved -> checkpoints/mini_llama_sft.pt")
