"""31_pretrain_bpe.py - Cap 31: pretrain Mini-LLaMA con vocab BPE.

Carga BPETokenizer (1112 tokens), entrena Mini-LLaMA sobre Shakespeare+Quijote
tokenizado, guarda mini_llama_bpe_base.pt.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniLLaMA, get_device, generate_with_prompt

torch.manual_seed(1337)
device = get_device()

print("Cargando BPETokenizer...")
tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab_size = tok.vocab_size
print(f"vocab_size={vocab_size}")

print("Tokenizando corpus bilingue...")
en = Path("shakespeare.txt").read_text(encoding="utf-8")
es = Path("quijote.txt").read_text(encoding="utf-8")
corpus = en + "\n" + es
data = torch.tensor(tok.encode(corpus), dtype=torch.long)
print(f"Tokens totales: {len(data):,}")

# Hyperparams (igual que char-level salvo vocab_size)
BLOCK = 256
BATCH = 32
LR = 3e-4
ITERS = 3000
WD = 0.01

model = MiniLLaMA(vocab_size=vocab_size, max_seq_len=BLOCK,
                  d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)
model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

def get_batch():
    ix = torch.randint(0, len(data) - BLOCK, (BATCH,))
    x = torch.stack([data[i:i+BLOCK] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+BLOCK+1] for i in ix]).to(device)
    return x, y

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

print(f"\nPretrain BPE: {ITERS} iters\n")
for it in range(ITERS):
    x, y = get_batch()
    logits, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 300 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "checkpoints/mini_llama_bpe_base.pt")
print("\nSaved -> checkpoints/mini_llama_bpe_base.pt")

# Sample de generacion
print("\n=== Sample generacion BPE-base ===")
for prompt in ["To be or not", "En un lugar"]:
    out = generate_with_prompt(model, prompt, tok, max_new_tokens=30,
                               temperature=0.8, top_k=10, device=device,
                               stop_token=None)
    print(f"Prompt: {prompt!r}")
    print(f"Output: {out!r}")
    print()
