"""53_attention_heatmaps.py - Cap 53: heatmaps de atencion ASCII por capa/cabeza."""
import math
import torch
import torch.nn.functional as F
from _models import (load_pretrained_mini_llama, get_device, CharTokenizer,
                     load_text, apply_rope)
from _interp import cache_activations

torch.manual_seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

prompt = "BRUTUS:\nI am"
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
T = ids.shape[1]
tokens_visual = [c.replace("\n", "\\n") for c in prompt]


def compute_attn_weights(x_norm, attn):
    """Recomputa attn_weights manualmente dado el input post-norm1."""
    B, T, _ = x_norm.shape
    Q = attn.W_Q(x_norm).view(B, T, attn.h_q, attn.d_k).transpose(1, 2)
    K = attn.W_K(x_norm).view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    cos = attn.rope_cos[:T]
    sin = attn.rope_sin[:T]
    Q = apply_rope(Q, cos, sin)
    K = apply_rope(K, cos, sin)
    K_full = K.repeat_interleave(attn.group_size, dim=1)
    scores = Q @ K_full.transpose(-2, -1) / math.sqrt(attn.d_k)
    mask = attn.mask[:, :, :T, :T]
    scores = scores.masked_fill(mask == 0, float('-inf'))
    return F.softmax(scores, dim=-1)  # (B, h_q, T, T)


def render_heatmap(weights_2d, tokens):
    """Imprime heatmap ASCII de matriz (T, T)."""
    chars = [' ', '.', '-', '+', '*', '#']
    T = weights_2d.shape[0]
    print("        " + "".join(f"{t:>3}" for t in tokens))
    for i in range(T):
        row = ""
        for j in range(T):
            v = weights_2d[i, j].item()
            if v == 0:
                row += "  ."
            else:
                idx = min(int(v * len(chars)), len(chars) - 1)
                row += f"  {chars[idx]}"
        print(f"{tokens[i]:>4} -> {row}")


# Cachear normas de cada bloque para input a la atencion
norms = [f"blocks.{i}.norm1" for i in range(4)]
with cache_activations(model, norms) as cache:
    with torch.no_grad():
        model(ids)

print(f"Prompt: {prompt!r}")
print(f"T = {T} tokens\n")
print(f"Tokens: {tokens_visual}\n")

# Computar attn_weights por capa
all_weights = []
for layer in range(4):
    x_norm = cache[f"blocks.{layer}.norm1"]
    w = compute_attn_weights(x_norm, model.blocks[layer].attn)  # (1, 4, T, T)
    all_weights.append(w[0])

print("=== Heatmaps de atencion (filas = query pos, cols = key pos) ===")
print("Caracteres: ' '=0, '.'=baja, '-'=media-baja, '+'=media, '*'=alta, '#'=muy alta\n")

for layer in range(4):
    for head in range(4):
        print(f"\n--- block.{layer} head.{head} ---")
        render_heatmap(all_weights[layer][head], tokens_visual)

# Resumen de patrones por cabeza
print("\n\n=== Score resumen por cabeza ===")
print(f"{'cabeza':<15} {'self_attn':>10} {'prev_token':>12} {'cls_attn':>10}")
print("-" * 50)
for layer in range(4):
    for head in range(4):
        w = all_weights[layer][head]  # (T, T)
        # self_attn: media de diagonal
        self_a = torch.tensor([w[i, i].item() for i in range(T)]).mean().item()
        # prev_token: media de subdiagonal
        prev_a = torch.tensor([w[i, i - 1].item() for i in range(1, T)]).mean().item()
        # cls_attn (atencion al primer token)
        cls_a = w[:, 0].mean().item()
        print(f"block.{layer} head.{head}  {self_a:>10.3f} {prev_a:>12.3f} {cls_a:>10.3f}")
