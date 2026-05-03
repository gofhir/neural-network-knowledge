"""43_train_bert.py - Cap 43: MLM pretraining de Mini-BERT."""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniBERT, MLMHead, get_device
from _bert_utils import apply_mlm_mask

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
vocab_size = tok.vocab_size

en = Path("shakespeare.txt").read_text(encoding="utf-8")
es = Path("quijote.txt").read_text(encoding="utf-8")
corpus = en + "\n" + es
data = torch.tensor(tok.encode(corpus), dtype=torch.long)
print(f"Corpus: {len(data):,} tokens")

BLOCK = 64   # longitud de secuencia (sin [CLS][SEP] la ventana real es 62)
BATCH = 32
LR    = 1e-4
ITERS = 3000
WD    = 0.01

model    = MiniBERT(vocab_size=vocab_size, max_seq_len=BLOCK+2,
                    d_model=128, n_heads=4, n_layers=4, d_ff=512).to(device)
mlm_head = MLMHead(d_model=128, vocab_size=vocab_size).to(device)

params = list(model.parameters()) + list(mlm_head.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

n_params = sum(p.numel() for p in params)
print(f"Params: {n_params:,}\n")

def get_batch():
    """Muestrea ventanas aleatorias y las formatea como BERT input."""
    ix = torch.randint(0, len(data) - BLOCK, (BATCH,))
    windows = torch.stack([data[i:i+BLOCK] for i in ix])  # (B, 64)
    # Agregar [CLS] al inicio y [SEP] al final
    cls_col = torch.full((BATCH, 1), tok.cls_id, dtype=torch.long)
    sep_col = torch.full((BATCH, 1), tok.sep_id, dtype=torch.long)
    input_ids = torch.cat([cls_col, windows, sep_col], dim=1)  # (B, 66)
    masked_ids, labels = apply_mlm_mask(
        input_ids.clone(), mask_prob=0.15,
        mask_id=tok.mask_id, vocab_size=vocab_size,
        special_ids=(tok.cls_id, tok.sep_id, tok.mask_id)
    )
    return masked_ids.to(device), labels.to(device)

print(f"MLM pretraining: {ITERS} iters\n")
for it in range(ITERS):
    masked_ids, labels = get_batch()
    h      = model(masked_ids)
    logits = mlm_head(h)
    loss   = F.cross_entropy(logits.view(-1, vocab_size),
                              labels.view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 300 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save({
    "model": model.state_dict(),
    "mlm_head": mlm_head.state_dict(),
    "config": dict(vocab_size=vocab_size, max_seq_len=BLOCK+2,
                   d_model=128, n_heads=4, n_layers=4, d_ff=512),
}, "checkpoints/mini_bert_pretrained.pt")
print("\nSaved -> checkpoints/mini_bert_pretrained.pt")
