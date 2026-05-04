"""58_circuit_discovery.py - Cap 58: head-level patching para identificar circuitos."""
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

clean_prompt = "BRUTUS:\nI am "
corrupted_prompt = "BIANCA:\nI am "
clean_ids = torch.tensor([tok.encode(clean_prompt)], dtype=torch.long, device=device)
corrupted_ids = torch.tensor([tok.encode(corrupted_prompt)], dtype=torch.long, device=device)
T = clean_ids.shape[1]


# Necesitamos cachear los outputs DE CADA CABEZA en cada bloque.
# La salida del bloque es: x = x_pre + attn_out + ffn_out
# attn_out por cabeza (antes de W_O): (B, h_q, T, d_k)
# Vamos a cachear el output del attn (post W_O) usando un hook que captura
# tambien las contribuciones por cabeza via el calculo manual.
def compute_per_head_output(x_norm, attn):
    """Retorna las contribuciones de cada cabeza al residual stream (post-W_O por cabeza).
    Output shape: (B, h_q, T, d_model). Sumarlas == output de attn."""
    B, T, _ = x_norm.shape
    Q = attn.W_Q(x_norm).view(B, T, attn.h_q, attn.d_k).transpose(1, 2)
    K = attn.W_K(x_norm).view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    V = attn.W_V(x_norm).view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    Q = apply_rope(Q, attn.rope_cos[:T], attn.rope_sin[:T])
    K = apply_rope(K, attn.rope_cos[:T], attn.rope_sin[:T])
    K_full = K.repeat_interleave(attn.group_size, dim=1)
    V_full = V.repeat_interleave(attn.group_size, dim=1)
    scores = Q @ K_full.transpose(-2, -1) / math.sqrt(attn.d_k)
    scores = scores.masked_fill(attn.mask[:, :, :T, :T] == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    out_per_head = weights @ V_full  # (B, h_q, T, d_k)

    # Aplicar W_O por cabeza: W_O.weight es (d_model, h_q*d_k).
    # Slice de W_O por cabeza:
    contributions = []
    for h in range(attn.h_q):
        W_O_h = attn.W_O.weight[:, h * attn.d_k:(h + 1) * attn.d_k]  # (d_model, d_k)
        contrib = out_per_head[:, h, :, :] @ W_O_h.T  # (B, T, d_model)
        contributions.append(contrib)
    return torch.stack(contributions, dim=1)  # (B, h_q, T, d_model)


# Cachear normas y per-head outputs del CLEAN
norms = [f"blocks.{i}.norm1" for i in range(4)]
with cache_activations(model, norms) as clean_norms:
    with torch.no_grad():
        clean_logits, _ = model(clean_ids)

clean_per_head = []
for layer in range(4):
    contribs = compute_per_head_output(clean_norms[f"blocks.{layer}.norm1"],
                                       model.blocks[layer].attn)
    clean_per_head.append(contribs)

# Run corrupted
with torch.no_grad():
    corrupted_logits, _ = model(corrupted_ids)

# Target: token con mayor diff
target_id = (clean_logits[0, -1] - corrupted_logits[0, -1]).argmax().item()
diff = (clean_logits[0, -1, target_id] - corrupted_logits[0, -1, target_id]).item()
print(f"Clean:     {clean_prompt!r}")
print(f"Corrupted: {corrupted_prompt!r}")
print(f"Target token: {tok.id_to_char[target_id]!r}, diff (clean - corrupted) = {diff:+.3f}\n")


# Para patchear UNA CABEZA, hacemos: ejecutar corrupted hasta ese bloque,
# capturar contribuciones por cabeza, reemplazar la cabeza target con clean_per_head,
# sumar y continuar.
# Implementacion: hook que reemplaza el output del attn module sumando la diff
# (contribucion clean - contribucion corrupted) para esa cabeza.
def patch_one_head(layer, head):
    """Patchea cabeza especifica del corrupted con la del clean. Retorna logit target."""
    # Computar contribucion corrupted de esa cabeza
    with cache_activations(model, [f"blocks.{layer}.norm1"]) as cor_norms:
        with torch.no_grad():
            model(corrupted_ids)
    cor_contribs = compute_per_head_output(cor_norms[f"blocks.{layer}.norm1"],
                                            model.blocks[layer].attn)
    delta = clean_per_head[layer][:, head] - cor_contribs[:, head]  # (1, T, d_model)

    # Hook: agregar delta al output del attn de ese bloque
    def patch_attn(module, inputs, output):
        return output + delta

    handle = model.blocks[layer].attn.register_forward_hook(patch_attn)
    try:
        with torch.no_grad():
            patched_logits, _ = model(corrupted_ids)
    finally:
        handle.remove()
    return patched_logits[0, -1, target_id].item()


print("=== Head-level patching: % de recovery por (layer, head) ===\n")
print(f"{'cabeza':<18} {'recovery':>10}")
print("-" * 30)
results = []
for layer in range(4):
    for head in range(4):
        patched_target = patch_one_head(layer, head)
        recovery = ((patched_target - corrupted_logits[0, -1, target_id].item()) / diff) * 100
        results.append((recovery, layer, head))
        print(f"block.{layer} head.{head}    {recovery:>+9.1f}%")

print("\n=== Top-3 cabezas con mayor recovery causal ===")
results.sort(key=lambda x: x[0], reverse=True)
for rank, (rec, l, h) in enumerate(results[:3], 1):
    print(f"  rank {rank}: block.{l} head.{h}  recovery={rec:+.1f}%")

print("\n=== Bottom-3 cabezas con menor (o negativo) recovery ===")
for rank, (rec, l, h) in enumerate(results[-3:], 1):
    print(f"  rank -{rank}: block.{l} head.{h}  recovery={rec:+.1f}%")

print("\n=== Lectura del circuito ===")
n_positive = sum(1 for r, _, _ in results if r > 5)
n_strong = sum(1 for r, _, _ in results if r > 20)
print(f"Cabezas con recovery > 5%:  {n_positive}/16")
print(f"Cabezas con recovery > 20%: {n_strong}/16")
if n_strong >= 1:
    print("Hay cabezas con efecto causal claro — circuito identificable")
else:
    print("Ninguna cabeza individual tiene efecto >20% — circuito distribuido")
