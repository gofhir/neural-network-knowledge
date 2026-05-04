"""59_superposition.py - Cap 59: superposition demo (Anthropic Toy Models 2022)."""
import math
import torch
import torch.nn as nn

torch.manual_seed(1337)

# Hyperparams del toy model
N_FEATURES = 5         # numero de features "reales"
D_MODEL = 2            # bottleneck — menos dim que features
N_SAMPLES = 8192
SPARSITY = 0.7         # fraccion de features = 0 en cada sample


# Generar datos sparse: cada feature es uniform[0,1] con prob (1-sparsity), 0 con prob sparsity
def generate_data(n_samples, n_features, sparsity):
    mask = torch.rand(n_samples, n_features) > sparsity
    values = torch.rand(n_samples, n_features)
    return values * mask.float()


# Toy autoencoder: x -> proj a 2D -> proj inversa a n_features
class Toy(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_features, d_model) * 0.1)
        # Decoder es transpuesto (tied weights, convencion del paper)

    def forward(self, x):
        h = x @ self.W                # (B, d_model)
        recon = torch.relu(h @ self.W.T)  # (B, n_features), ReLU para non-neg
        return recon, h


# Entrenar
model = Toy(N_FEATURES, D_MODEL)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

print(f"Toy model de superposition: {N_FEATURES} features -> {D_MODEL} dim -> {N_FEATURES}")
print(f"Sparsity={SPARSITY}, samples={N_SAMPLES}\n")

initial_loss = None
for step in range(2000):
    x = generate_data(N_SAMPLES // 8, N_FEATURES, SPARSITY)
    recon, _ = model(x)
    loss = ((x - recon) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step == 0:
        initial_loss = loss.item()
    if step in [0, 100, 500, 1000, 1999]:
        print(f"  step {step:>4d}: loss={loss.item():.5f}")

print()
# Inspeccionar vectores aprendidos
W = model.W.detach()  # (n_features, d_model)
print(f"Matriz aprendida W (shape {tuple(W.shape)}):")
print(W.numpy())

print(f"\nNorma de cada feature vector:")
for i in range(N_FEATURES):
    norm = W[i].norm().item()
    print(f"  feature {i}: ||W_{i}|| = {norm:.3f}")

print(f"\nAngulo de cada feature vector (en grados, vs eje x):")
angles = []
for i in range(N_FEATURES):
    angle_rad = math.atan2(W[i, 1].item(), W[i, 0].item())
    angle_deg = math.degrees(angle_rad)
    angles.append(angle_deg)
    print(f"  feature {i}: angulo={angle_deg:+7.2f}°")

print(f"\nDiferencias angulares (esperamos ~72° si distribucion uniforme):")
sorted_idx = sorted(range(N_FEATURES), key=lambda i: angles[i])
for k in range(N_FEATURES):
    i = sorted_idx[k]
    j = sorted_idx[(k + 1) % N_FEATURES]
    diff = angles[j] - angles[i]
    if diff < 0:
        diff += 360
    print(f"  feature {i:>2} ({angles[i]:+7.2f}°) -> feature {j:>2} ({angles[j]:+7.2f}°): diff={diff:.2f}°")

print(f"\nMatriz de productos punto W @ W^T:")
gram = W @ W.T
for i in range(N_FEATURES):
    row = "  ".join(f"{gram[i, j].item():+.3f}" for j in range(N_FEATURES))
    print(f"  {row}")

print("\n=== Lectura ===")
print("Si los features fueran ortogonales en 2D solo cabrian 2 features.")
print("Pero el modelo aprende representacion NO-ORTOGONAL para empacar 5 features en 2 dims.")
print("Esto es SUPERPOSITION: features comprimidas en angulos distintos del plano.")

# Plot ASCII del plano
print("\n=== Plot ASCII de los vectores feature en 2D ===")
print("(centro en (0,0), cada feature dibujado como linea desde el origen)")
SCALE = 20
GRID = 41
canvas = [[' '] * GRID for _ in range(GRID)]
center = GRID // 2
canvas[center][center] = '+'

for i in range(N_FEATURES):
    x_end = int(W[i, 0].item() * SCALE) + center
    y_end = int(-W[i, 1].item() * SCALE) + center  # invertir Y para display
    # Linea simple desde centro a (x_end, y_end)
    steps = max(abs(x_end - center), abs(y_end - center))
    if steps == 0:
        continue
    for s in range(steps + 1):
        x = center + (x_end - center) * s // steps
        y = center + (y_end - center) * s // steps
        if 0 <= x < GRID and 0 <= y < GRID:
            canvas[y][x] = str(i)

for row in canvas:
    print("  " + "".join(row))
