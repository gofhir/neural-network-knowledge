"""55_induction_heads.py - Cap 55: induction heads sobre prompts repetidos."""
import math
import random
import torch
import torch.nn.functional as F
from _models import (load_pretrained_mini_llama, get_device, CharTokenizer,
                     load_text, apply_rope)
from _interp import cache_activations, induction_score

torch.manual_seed(1337)
random.seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))


def compute_attn_weights(x_norm, attn):
    B, T, _ = x_norm.shape
    Q = attn.W_Q(x_norm).view(B, T, attn.h_q, attn.d_k).transpose(1, 2)
    K = attn.W_K(x_norm).view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    Q = apply_rope(Q, attn.rope_cos[:T], attn.rope_sin[:T])
    K = apply_rope(K, attn.rope_cos[:T], attn.rope_sin[:T])
    K_full = K.repeat_interleave(attn.group_size, dim=1)
    scores = Q @ K_full.transpose(-2, -1) / math.sqrt(attn.d_k)
    scores = scores.masked_fill(attn.mask[:, :, :T, :T] == 0, float('-inf'))
    return F.softmax(scores, dim=-1)


# Generar prompts repetidos: secuencia aleatoria + misma secuencia
N_PROMPTS = 30
SEG_LEN = 12
vocab_chars = list(tok.id_to_char.values())
prompts_ids = []
for _ in range(N_PROMPTS):
    seg = [random.choice(vocab_chars) for _ in range(SEG_LEN)]
    full = seg + seg  # repetir
    seq = "".join(full)
    ids = tok.encode(seq)
    if len(ids) >= 2 * SEG_LEN:
        prompts_ids.append(torch.tensor(ids[:2 * SEG_LEN], dtype=torch.long))

print(f"Generados {len(prompts_ids)} prompts repetidos de longitud {2 * SEG_LEN}\n")
print(f"Ejemplo: {tok.decode(prompts_ids[0].tolist())!r}\n")

# Acumular induction scores
n_layers, h_q = 4, 4
sum_scores = torch.zeros(n_layers, h_q)

for ids_t in prompts_ids:
    ids = ids_t.unsqueeze(0).to(device)
    with cache_activations(model, [f"blocks.{i}.norm1" for i in range(n_layers)]) as cache:
        with torch.no_grad():
            model(ids)
    for layer in range(n_layers):
        x_norm = cache[f"blocks.{layer}.norm1"]
        w = compute_attn_weights(x_norm, model.blocks[layer].attn)[0]
        for head in range(h_q):
            sum_scores[layer, head] += induction_score(w[head].cpu(), ids_t)

avg_scores = sum_scores / len(prompts_ids)

print("=== Tabla: induction_score promedio por cabeza ===\n")
print(f"{'cabeza':<18} {'score':>8}")
print("-" * 28)
flat = []
for layer in range(n_layers):
    for head in range(h_q):
        score = avg_scores[layer, head].item()
        flat.append((score, layer, head))
        print(f"block.{layer} head.{head}    {score:>8.3f}")

print("\n=== Top-5 cabezas con mayor induction score ===\n")
flat.sort(reverse=True)
for rank, (score, layer, head) in enumerate(flat[:5], 1):
    print(f"  rank {rank}: block.{layer} head.{head}  score={score:.3f}")

print("\n=== Honestidad: lectura de los resultados ===")
top_score = flat[0][0]
if top_score > 0.5:
    print(f"  Cabeza top con score {top_score:.3f} > 0.5: induction head clara")
elif top_score > 0.3:
    print(f"  Cabeza top con score {top_score:.3f} en [0.3, 0.5]: induction parcial")
else:
    print(f"  Cabeza top con score {top_score:.3f} < 0.3: NO hay induction heads claras")
    print("  Limitacion de escala: Anthropic encontro induction en GPT-2 small (12 capas)")
    print("  Mini-LLaMA tiene 4 capas — posiblemente insuficiente para induction emergente")

# Demo visual de la cabeza top sobre un prompt repetido
print("\n=== Heatmap de la cabeza top-1 sobre prompt repetido ===")
top_score, top_layer, top_head = flat[0]
demo_seg = list("ToBeOrNotToBe")[:SEG_LEN]
demo_full = "".join(demo_seg + demo_seg)
demo_ids_full = tok.encode(demo_full)[:2 * SEG_LEN]
demo_ids = torch.tensor([demo_ids_full], dtype=torch.long, device=device)
with cache_activations(model, [f"blocks.{top_layer}.norm1"]) as cache:
    with torch.no_grad():
        model(demo_ids)
x_norm = cache[f"blocks.{top_layer}.norm1"]
w = compute_attn_weights(x_norm, model.blocks[top_layer].attn)[0, top_head]
demo_chars = [tok.id_to_char[i].replace("\n", "\\n") for i in demo_ids_full]

chars_pal = [' ', '.', '-', '+', '*', '#']
print(f"      " + "".join(f"{c:>3}" for c in demo_chars))
for i in range(w.shape[0]):
    row = ""
    for j in range(w.shape[1]):
        v = w[i, j].item()
        idx = min(int(v * len(chars_pal)), len(chars_pal) - 1) if v > 0 else 0
        row += f"  {chars_pal[idx]}"
    print(f"{demo_chars[i]:>4} {row}")
