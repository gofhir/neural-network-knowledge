"""47_finetune_bert.py - Cap 47: fine-tuning BERT para deteccion de idioma."""
import json, torch, random as _random
import torch.nn.functional as F
from pathlib import Path
from _models import MiniBERT, ClassificationHead, get_device

torch.manual_seed(1337)
device = get_device()

ckpt = torch.load("checkpoints/mini_bert_pretrained.pt", map_location=device, weights_only=False)
cfg  = ckpt["config"]
model    = MiniBERT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.train()

cls_head = ClassificationHead(d_model=cfg["d_model"], n_classes=2).to(device)

LR    = 2e-5  # 5x menor que pretraining (convencion BERT)
ITERS = 500
BATCH = 32
WD    = 0.01

with open("data/lang_train.jsonl") as f:
    train_data = [json.loads(l) for l in f]
params = list(model.parameters()) + list(cls_head.parameters())
opt    = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

print(f"Fine-tuning: {ITERS} iters, LR={LR}\n")
for it in range(ITERS):
    batch = _random.sample(train_data, BATCH)
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

Path("checkpoints").mkdir(exist_ok=True)
torch.save({
    "model":    model.state_dict(),
    "cls_head": cls_head.state_dict(),
    "config":   cfg,
}, "checkpoints/mini_bert_finetuned.pt")
print("\nSaved -> checkpoints/mini_bert_finetuned.pt")
