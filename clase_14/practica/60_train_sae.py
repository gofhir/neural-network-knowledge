"""60_train_sae.py - Cap 60: entrenar SAE sobre residual stream de Mini-LLaMA."""
import random
import torch
from pathlib import Path
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

# === Recolectar activaciones del residual stream ===
LAYER = 2  # capa intermedia
N_PROMPTS = 200
WIN = 64
TARGET_NAME = f"blocks.{LAYER}"

print(f"Recolectando activaciones de {TARGET_NAME} sobre {N_PROMPTS} prompts de {WIN} chars\n")
all_acts = []
for _ in range(N_PROMPTS):
    start = random.randint(0, len(text) - WIN - 1)
    prompt = text[start:start + WIN]
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    with cache_activations(model, [TARGET_NAME]) as cache:
        with torch.no_grad():
            model(ids)
    acts = cache[TARGET_NAME][0].cpu()  # (T, d_model)
    all_acts.append(acts)

X = torch.cat(all_acts, dim=0)  # (N_TOTAL, d_model)
print(f"Total activations recolectadas: shape {tuple(X.shape)}")
print(f"Mean magnitud: {X.norm(dim=-1).mean().item():.3f}")
print(f"Std por dimension: {X.std(dim=0).mean().item():.3f}\n")

# === Entrenar SAE ===
D_MODEL = 128
D_FEATURES = 512  # 4x expansion
L1_COEFF = 0.5    # alto para forzar sparsity real
LR = 3e-4
ITERS = 2000
BATCH = 1024

sae = SparseAutoencoder(d_model=D_MODEL, d_features=D_FEATURES, l1_coeff=L1_COEFF).to(device)
opt = torch.optim.Adam(sae.parameters(), lr=LR)

X_dev = X.to(device)
print(f"=== Entrenando SAE (d_features={D_FEATURES}, lambda={L1_COEFF}, iters={ITERS}) ===\n")
print(f"{'iter':<8} {'recon_loss':>12} {'l1_loss':>10} {'total':>10} {'L0_avg':>10}")
print("-" * 55)

initial_recon = None
for it in range(ITERS):
    idx = torch.randint(0, X_dev.shape[0], (BATCH,))
    x = X_dev[idx]
    recon, features = sae(x)
    recon_loss = ((x - recon) ** 2).mean()
    l1_loss = features.abs().mean()
    loss = recon_loss + L1_COEFF * l1_loss
    opt.zero_grad(); loss.backward(); opt.step()

    if it == 0:
        initial_recon = recon_loss.item()
    if it in [0, 100, 250, 500, 1000, 1499]:
        l0 = (features > 0).float().sum(-1).mean().item()
        print(f"{it:<8d} {recon_loss.item():>12.5f} {l1_loss.item():>10.5f} "
              f"{loss.item():>10.5f} {l0:>10.1f}")

final_recon = recon_loss.item()
print(f"\nReduccion en recon_loss: {initial_recon:.5f} -> {final_recon:.5f} "
      f"({(1 - final_recon/initial_recon)*100:.1f}% menor)")

# === Estadisticas finales del SAE ===
print("\n=== Estadisticas finales del SAE ===")
with torch.no_grad():
    recon_full, features_full = sae(X_dev)
    recon_loss_full = ((X_dev - recon_full) ** 2).mean().item()
    var_explained = 1 - recon_loss_full / X_dev.var().item()
    n_active = (features_full > 0).float().sum(-1)  # L0 por sample
    n_dead = (features_full.sum(0) == 0).sum().item()  # features que nunca se activan

print(f"Recon loss sobre todo el dataset: {recon_loss_full:.5f}")
print(f"Varianza explicada: {var_explained * 100:.1f}%")
print(f"L0 promedio (features activas por sample): {n_active.mean().item():.1f} de {D_FEATURES}")
print(f"L0 mediano: {n_active.median().item():.1f}")
print(f"Features muertas (nunca activas): {n_dead} de {D_FEATURES}")

# === Guardar checkpoint ===
Path("checkpoints").mkdir(exist_ok=True)
ckpt_path = "checkpoints/sae_mini_llama.pt"
torch.save({
    "sae": sae.state_dict(),
    "config": {"d_model": D_MODEL, "d_features": D_FEATURES, "l1_coeff": L1_COEFF,
               "layer": LAYER, "target_name": TARGET_NAME},
    "stats": {"recon_loss": recon_loss_full, "var_explained": var_explained,
              "L0_mean": n_active.mean().item(), "n_dead": n_dead},
}, ckpt_path)
print(f"\nGuardado SAE en {ckpt_path}")
