"""52_logit_lens.py - Cap 52: predicciones capa por capa via logit lens."""
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations, logit_lens

torch.manual_seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

prompt = "To be or not to "
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
T = ids.shape[1]

# Cachear residual en cada punto + norm_final
points = ["tok_emb"] + [f"blocks.{i}" for i in range(4)] + ["norm_final"]
with cache_activations(model, points) as cache:
    with torch.no_grad():
        model(ids)

# Normalizar (con norm_final del modelo) cada residual intermedio antes del head
print(f"Prompt: {prompt!r}")
print(f"Posicion final del stream: {T-1} (proxima prediccion)\n")

print("=== Top-3 predicciones para la posicion final, capa por capa ===\n")
print(f"{'Punto':<13} {'top-1':<22} {'top-2':<22} {'top-3':<22}")
print("-" * 80)
for name in points:
    h = cache[name]
    # Aplicar norm_final del modelo para normalizar antes del head
    h_norm = model.norm_final(h)
    logits = logit_lens(model, h_norm)  # (1, T, vocab)
    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    top = probs.topk(3)
    parts = []
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        ch = tok.decode([idx]).replace("\n", "\\n").replace("\t", "\\t")
        parts.append(f"{ch!r}={p:.3f}")
    print(f"{name:<13} {parts[0]:<22} {parts[1]:<22} {parts[2]:<22}")

print("\n=== Evolucion de la prediccion top-1 vs el target probable 'b' (de 'be') ===\n")
target_id = tok.encode("b")[0]
print(f"target='b' id={target_id}")
print(f"{'Punto':<13} {'P(b)':>8} {'rank de b':>12} {'top-1 actual':<15}")
print("-" * 50)
for name in points:
    h = cache[name]
    h_norm = model.norm_final(h)
    logits = logit_lens(model, h_norm)
    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    p_target = probs[target_id].item()
    rank = (probs > probs[target_id]).sum().item() + 1
    top1_id = last.argmax().item()
    top1_ch = tok.decode([top1_id]).replace("\n", "\\n")
    print(f"{name:<13} {p_target:>8.4f} {rank:>12d} {top1_ch!r:<15}")
