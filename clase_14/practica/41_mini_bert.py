"""41_mini_bert.py - Cap 41: forward pass completo de Mini-BERT."""
import torch
from _bpe import BPETokenizer
from _models import MiniBERT, get_device

torch.manual_seed(42)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
vocab_size = tok.vocab_size  # 1115

cfg = dict(vocab_size=vocab_size, max_seq_len=128,
           d_model=128, n_heads=4, n_layers=4, d_ff=512)
model = MiniBERT(**cfg).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"MiniBERT: {n_params:,} parametros")
print(f"Comparacion: MiniLLaMA tuvo ~1,072,256 params\n")

# Forward pass de ejemplo
sentences = [
    "To be or not to be",
    "En un lugar de la Mancha",
]
print("=== Forward pass ===\n")
for s in sentences:
    ids = torch.tensor([tok.encode_bert(s)], dtype=torch.long, device=device)
    ids = ids[:, :128]  # truncar a max_seq_len
    h = model(ids)
    cls_vec = h[0, 0]  # vector [CLS]
    print(f"Texto:     {s!r}")
    print(f"Tokens:    {ids.shape[1]} (incluyendo [CLS] y [SEP])")
    print(f"h.shape:   {h.shape}  — (batch=1, seq_len, d_model=128)")
    print(f"[CLS] vec: norma={cls_vec.norm().item():.4f}, primeros 4 dims: {cls_vec[:4].tolist()}")
    print()

print("=== Diferencias con Mini-LLaMA ===")
print("""
Mini-LLaMA                Mini-BERT
-----------               ----------
GQA (h_q=4, h_kv=2)      MHA (n_heads=4)
RoPE en Q y K             Learned pos emb (sumado al token emb)
RMSNorm                   LayerNorm
SwiGLU                    GELU
Causal mask               Sin mascara
max_seq_len=256            max_seq_len=128
Genera: next token         Clasifica: [CLS] vector
""")
