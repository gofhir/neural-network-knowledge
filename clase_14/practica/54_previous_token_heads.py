"""54_previous_token_heads.py - Cap 54: identificar cabezas previous-token via score promediado."""
import math
import random
import torch
import torch.nn.functional as F
from _models import (load_pretrained_mini_llama, get_device, CharTokenizer,
                     load_text, apply_rope)
from _interp import cache_activations, previous_token_score

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


# Sample 50 prompts aleatorios de 24 chars desde Shakespeare
N_PROMPTS = 50
WIN = 24
prompts = []
for _ in range(N_PROMPTS):
    start = random.randint(0, len(text) - WIN - 1)
    prompts.append(text[start:start + WIN])

print(f"Promediando previous_token_score sobre {N_PROMPTS} prompts de {WIN} chars\n")

# Acumular scores: shape (n_layers, h_q)
n_layers, h_q = 4, 4
sum_scores = torch.zeros(n_layers, h_q)

for prompt in prompts:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    with cache_activations(model, [f"blocks.{i}.norm1" for i in range(n_layers)]) as cache:
        with torch.no_grad():
            model(ids)
    for layer in range(n_layers):
        x_norm = cache[f"blocks.{layer}.norm1"]
        w = compute_attn_weights(x_norm, model.blocks[layer].attn)[0]  # (h_q, T, T)
        for head in range(h_q):
            sum_scores[layer, head] += previous_token_score(w[head].cpu())

avg_scores = sum_scores / N_PROMPTS

print("=== Tabla: previous_token_score promedio por cabeza ===\n")
print(f"{'cabeza':<18} {'score':>8}")
print("-" * 28)
flat = []
for layer in range(n_layers):
    for head in range(h_q):
        score = avg_scores[layer, head].item()
        flat.append((score, layer, head))
        print(f"block.{layer} head.{head}    {score:>8.3f}")

print("\n=== Top-5 cabezas con mayor previous-token score ===\n")
flat.sort(reverse=True)
for rank, (score, layer, head) in enumerate(flat[:5], 1):
    print(f"  rank {rank}: block.{layer} head.{head}  score={score:.3f}")

print("\n=== Bottom-3 cabezas (menor previous-token score) ===\n")
for rank, (score, layer, head) in enumerate(flat[-3:], 1):
    print(f"  rank -{rank}: block.{layer} head.{head}  score={score:.3f}")

# Demo visual de la mejor cabeza
print("\n=== Heatmap de la cabeza top-1 sobre prompt 'BRUTUS:\\nI am' ===")
top_score, top_layer, top_head = flat[0]
demo_ids = torch.tensor([tok.encode("BRUTUS:\nI am")], dtype=torch.long, device=device)
with cache_activations(model, [f"blocks.{top_layer}.norm1"]) as cache:
    with torch.no_grad():
        model(demo_ids)
x_norm = cache[f"blocks.{top_layer}.norm1"]
w = compute_attn_weights(x_norm, model.blocks[top_layer].attn)[0, top_head]

prompt_chars = ["B", "R", "U", "T", "U", "S", ":", "\\n", "I", " ", "a", "m"]
chars_pal = [' ', '.', '-', '+', '*', '#']
print(f"      " + "".join(f"{c:>3}" for c in prompt_chars))
for i in range(w.shape[0]):
    row = ""
    for j in range(w.shape[1]):
        v = w[i, j].item()
        idx = min(int(v * len(chars_pal)), len(chars_pal) - 1) if v > 0 else 0
        row += f"  {chars_pal[idx]}"
    print(f"{prompt_chars[i]:>4} {row}")
