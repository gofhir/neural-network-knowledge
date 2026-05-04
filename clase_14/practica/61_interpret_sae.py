"""61_interpret_sae.py - Cap 61: interpretar features del SAE entrenado."""
import random
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations, SparseAutoencoder

torch.manual_seed(1337)
random.seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

# Cargar SAE entrenado
ckpt = torch.load("checkpoints/sae_mini_llama.pt", map_location=device, weights_only=False)
cfg = ckpt["config"]
sae = SparseAutoencoder(d_model=cfg["d_model"], d_features=cfg["d_features"],
                         l1_coeff=cfg["l1_coeff"]).to(device)
sae.load_state_dict(ckpt["sae"])
sae.eval()
print(f"SAE cargado: d_features={cfg['d_features']}, layer={cfg['layer']}\n")

# Recolectar 50 prompts grandes, capturar activaciones del residual stream
# y guardar (token_idx, char_at_token, char_context, feature_activations)
WIN = 64
N_PROMPTS = 50

# Lista de (char, context_str, feature_act_vector)
samples = []
for _ in range(N_PROMPTS):
    start = random.randint(0, len(text) - WIN - 1)
    prompt = text[start:start + WIN]
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    with cache_activations(model, [cfg["target_name"]]) as cache:
        with torch.no_grad():
            model(ids)
    acts = cache[cfg["target_name"]][0]  # (T, d_model)
    with torch.no_grad():
        _, features = sae(acts)  # (T, d_features)
    features = features.cpu()
    for t in range(features.shape[0]):
        token_id = ids[0, t].item()
        ch = tok.id_to_char[token_id]
        ctx_l = max(0, t - 5)
        ctx_r = min(features.shape[0], t + 6)
        context = "".join(tok.id_to_char[ids[0, i].item()] for i in range(ctx_l, ctx_r))
        samples.append((ch, context, features[t]))

print(f"Total samples: {len(samples)} tokens analizados\n")

# Para cada feature, encontrar los top-k tokens que mas la activan
TOP_K = 8
N_FEATURES_INSPECT = 12  # mostrar algunas features
all_features = torch.stack([s[2] for s in samples])  # (N, d_features)
feature_max_act = all_features.max(dim=0).values  # (d_features,)

# Ordenar features por su activacion maxima (las mas "fuertes" primero)
sorted_features = feature_max_act.argsort(descending=True)

print("=== Top-12 features por activacion maxima — top tokens y contexto ===\n")
for fi_idx in range(N_FEATURES_INSPECT):
    fi = sorted_features[fi_idx].item()
    activations_for_feature = all_features[:, fi]
    top_indices = activations_for_feature.topk(TOP_K).indices.tolist()

    print(f"--- Feature #{fi}  (max_act={feature_max_act[fi]:.3f}) ---")
    for rank, idx in enumerate(top_indices):
        ch, ctx, _ = samples[idx]
        act = activations_for_feature[idx].item()
        ch_safe = repr(ch)
        ctx_safe = repr(ctx)
        print(f"  rank {rank+1}: act={act:.2f}  char={ch_safe:>5}  context={ctx_safe}")
    print()

# Estadistica: cuantas features tienen patrones "interpretables" (top-k todos del mismo char)
print("=== Cuantas features son monosemanticas? (top-3 tokens iguales) ===")
n_mono = 0
n_partial = 0
example_mono = []
for fi in range(cfg["d_features"]):
    acts = all_features[:, fi]
    if acts.max().item() < 0.01:
        continue
    top3 = acts.topk(3).indices.tolist()
    chars_top3 = [samples[i][0] for i in top3]
    if len(set(chars_top3)) == 1:
        n_mono += 1
        if len(example_mono) < 5:
            example_mono.append((fi, chars_top3[0], acts.max().item()))
    elif len(set(chars_top3)) == 2:
        n_partial += 1

print(f"  Features con top-3 mismo char (monosemantica fuerte):  {n_mono}/{cfg['d_features']}")
print(f"  Features con top-3 dos chars distintos (parcial):       {n_partial}/{cfg['d_features']}")

if example_mono:
    print(f"\n  Ejemplos de features monosemanticas:")
    for fi, ch, act in example_mono:
        print(f"    feature #{fi}: char={ch!r:>5}  max_act={act:.2f}")
