"""51_residual_stream.py - Cap 51: residual stream como autopista del Transformer."""
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations

torch.manual_seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

prompt = "To be or not to "
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

# Cachear embeddings (input al bloque 0) + output de cada bloque
points = ["tok_emb"] + [f"blocks.{i}" for i in range(4)]
with cache_activations(model, points) as cache:
    with torch.no_grad():
        model(ids)

print("=== El residual stream como autopista ===\n")
print("Cada bloque LEE el stream actual y ESCRIBE una contribucion (delta).")
print("La nueva activacion = activacion previa + delta del bloque.\n")

print(f"{'Bloque':<10} {'||in||':>8} {'||out||':>8} {'||delta||':>10} {'||delta||/||in||':>16} {'cosine(in,out)':>15}")
print("-" * 70)
prev = cache["tok_emb"]
for i in range(4):
    cur = cache[f"blocks.{i}"]
    delta = cur - prev
    norm_in = prev.norm(dim=-1).mean().item()
    norm_out = cur.norm(dim=-1).mean().item()
    norm_delta = delta.norm(dim=-1).mean().item()
    rel = norm_delta / norm_in
    cos = torch.nn.functional.cosine_similarity(prev, cur, dim=-1).mean().item()
    print(f"block.{i:<5} {norm_in:>8.3f} {norm_out:>8.3f} {norm_delta:>10.3f} "
          f"{rel:>16.3f} {cos:>15.3f}")
    prev = cur

print("\nInterpretacion:")
print("  - ||delta|| pequeno relativo a ||in|| -> el bloque modifica el stream con cuidado")
print("  - ||delta|| grande -> el bloque sobreescribe partes del stream")
print("  - cosine cercano a 1 -> el output preserva la direccion del input")
print("  - cosine cercano a 0 -> el bloque rota el stream radicalmente")

print("\n=== Diagrama ===")
print("""
  tok_emb         block_0          block_1          block_2          block_3       norm_final -> head
     |               |                |                |                |
     v               v                v                v                v
  [emb] --------> [+d0] --------> [+d1] --------> [+d2] --------> [+d3] -> [norm]

cada bloque AGREGA un delta al stream sin sobreescribir.
los deltas se acumulan; el head final lee la suma de todas las contribuciones.
""")
