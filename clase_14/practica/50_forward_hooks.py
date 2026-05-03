"""50_forward_hooks.py - Cap 50: forward hooks y cache de activaciones."""
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
print(f"Prompt: {prompt!r}")
print(f"Tokens: {ids.shape[1]} ids = {ids[0].tolist()[:10]}...\n")

names = [f"blocks.{i}" for i in range(4)] + ["norm_final"]
print(f"Cacheando activaciones de {len(names)} puntos:")
with cache_activations(model, names) as cache:
    with torch.no_grad():
        model(ids)

print("\nShapes capturados:")
for name in names:
    t = cache[name]
    print(f"  {name:>15}: shape={tuple(t.shape)}, mean={t.mean():+.4f}, std={t.std():.4f}")

print("\nNorma del residual stream por punto (||x||_2 promedio sobre tokens):")
for name in names:
    norm = cache[name].norm(dim=-1).mean().item()
    print(f"  {name:>15}: {norm:.3f}")

print("\nDelta norma entre bloques consecutivos:")
prev = None
for name in names:
    cur = cache[name].norm(dim=-1).mean().item()
    if prev is not None:
        delta = cur - prev
        sign = "+" if delta >= 0 else ""
        print(f"  {name:>15}: {sign}{delta:.3f}")
    prev = cur
