"""
Ejemplo 3 — Regularizacion L1 y Sparsity

Objetivo: ver como L1 produce pesos EXACTAMENTE en cero (sparsity),
          a diferencia de L2 que solo los hace chicos.

Ejecutar:
  docker run --rm clase8 python -u ejemplos/pytorch/03_regularizacion_l1_sparsity.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

SEPARATOR = "=" * 60

# =============================================
# 1. Diferencia matematica L1 vs L2
# =============================================
print(SEPARATOR)
print("1. DIFERENCIA MATEMATICA: L1 vs L2")
print(SEPARATOR)

weights = torch.tensor([3.0, 0.5, -2.0, 0.01, -0.001, 4.0])

l1_penalty = weights.abs().sum()         # Σ|w|
l2_penalty = (weights ** 2).sum()        # Σw²

print(f"  Pesos: {weights.tolist()}")
print(f"")
print(f"  L1 = Σ|w| = {' + '.join(f'{abs(w):.3f}' for w in weights)} = {l1_penalty.item():.3f}")
print(f"  L2 = Σw²  = {' + '.join(f'{w**2:.3f}' for w in weights)} = {l2_penalty.item():.3f}")
print(f"")
print(f"  Contribucion de CADA peso al penalty:")
print(f"  {'Peso':>8s}  {'|w| (L1)':>8s}  {'w² (L2)':>8s}")
for w in weights:
    print(f"  {w.item():>8.3f}  {abs(w.item()):>8.3f}  {w.item()**2:>8.3f}")
print(f"")
print(f"  L2 penaliza MAS los pesos grandes (4.0² = 16.0)")
print(f"  L1 penaliza IGUAL proporcionalmente (|4.0| = 4.0)")
print(f"")
print(f"  Para pesos CHICOS (0.01, -0.001):")
print(f"    L2: 0.01² = 0.0001  ← casi no penaliza, se queda vivo")
print(f"    L1: |0.01| = 0.01   ← sigue penalizando, lo empuja a 0")

# =============================================
# 2. Crear datos
# =============================================
print(f"\n{SEPARATOR}")
print("2. CREAR DATOS")
print(SEPARATOR)

torch.manual_seed(42)

# Datos con 20 features, pero solo 3 son realmente utiles
n_samples = 200
n_features = 20
n_useful = 3

# Features utiles (determinan la clase)
X_useful = torch.randn(n_samples, n_useful)
# Features de ruido (no sirven)
X_noise = torch.randn(n_samples, n_features - n_useful) * 0.5

X = torch.cat([X_useful, X_noise], dim=1)
# La clase depende SOLO de los primeros 3 features
y = (X_useful.sum(dim=1) > 0).float()

print(f"  {n_samples} muestras, {n_features} features")
print(f"  Solo los primeros {n_useful} features son UTILES")
print(f"  Los otros {n_features - n_useful} son RUIDO")
print(f"  → Queremos que la red descubra cuales son los utiles")

# =============================================
# 3. Red simple
# =============================================

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 1)  # 1 sola capa: 20 pesos

    def forward(self, x):
        return self.fc(x)


# =============================================
# 4. Entrenar con L2
# =============================================
print(f"\n{SEPARATOR}")
print("4. ENTRENAR CON L2 (weight_decay)")
print(SEPARATOR)

torch.manual_seed(42)
model_l2 = SimpleNet()
optimizer_l2 = optim.Adam(model_l2.parameters(), lr=0.01, weight_decay=0.1)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(500):
    pred = model_l2(X).squeeze()
    loss = loss_fn(pred, y)
    optimizer_l2.zero_grad()
    loss.backward()
    optimizer_l2.step()

weights_l2 = model_l2.fc.weight.data.squeeze()
print(f"  Pesos aprendidos con L2:")
print(f"  {'Feature':>10s}  {'Peso':>8s}  {'Tipo':>10s}")
for i, w in enumerate(weights_l2):
    tipo = "UTIL" if i < n_useful else "ruido"
    bar = "#" * int(abs(w.item()) * 10)
    print(f"  Feature {i:2d}:  {w.item():+8.4f}  {tipo:>10s}  {bar}")

n_near_zero_l2 = (weights_l2.abs() < 0.01).sum().item()
print(f"\n  Pesos cerca de 0 (< 0.01): {n_near_zero_l2} de {n_features}")
print(f"  → L2 hace los pesos CHICOS, pero casi ninguno es exactamente 0")

# =============================================
# 5. Entrenar con L1 (manual)
# =============================================
print(f"\n{SEPARATOR}")
print("5. ENTRENAR CON L1 (implementacion manual)")
print(SEPARATOR)

print("""
  L1 NO viene como parametro del optimizador.
  Hay que agregarlo manualmente al loss:

    l1_penalty = sum(p.abs().sum() for p in model.parameters())
    loss = loss_fn(pred, y) + l1_lambda * l1_penalty
""")

torch.manual_seed(42)
model_l1 = SimpleNet()
optimizer_l1 = optim.Adam(model_l1.parameters(), lr=0.01)
l1_lambda = 0.05

for epoch in range(500):
    pred = model_l1(X).squeeze()

    # Loss original
    ce_loss = loss_fn(pred, y)

    # Penalty L1 manual
    l1_penalty = sum(p.abs().sum() for p in model_l1.parameters())

    # Loss total = CE + λ * L1
    loss = ce_loss + l1_lambda * l1_penalty

    optimizer_l1.zero_grad()
    loss.backward()
    optimizer_l1.step()

weights_l1 = model_l1.fc.weight.data.squeeze()
print(f"  Pesos aprendidos con L1:")
print(f"  {'Feature':>10s}  {'Peso':>8s}  {'Tipo':>10s}")
for i, w in enumerate(weights_l1):
    tipo = "UTIL" if i < n_useful else "ruido"
    marker = "  ← CERO!" if abs(w.item()) < 0.01 else ""
    bar = "#" * int(abs(w.item()) * 10)
    print(f"  Feature {i:2d}:  {w.item():+8.4f}  {tipo:>10s}  {bar}{marker}")

n_near_zero_l1 = (weights_l1.abs() < 0.01).sum().item()
print(f"\n  Pesos cerca de 0 (< 0.01): {n_near_zero_l1} de {n_features}")
print(f"  → L1 empuja muchos pesos a EXACTAMENTE 0 (sparsity)")
print(f"  → Los features de ruido fueron 'eliminados' automaticamente")

# =============================================
# 6. Comparar lado a lado
# =============================================
print(f"\n{SEPARATOR}")
print("6. COMPARACION L1 vs L2")
print(SEPARATOR)

print(f"\n  {'Feature':>10s}  {'L2':>8s}  {'L1':>8s}  {'Real':>6s}")
print(f"  {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 6}")
for i in range(n_features):
    tipo = "UTIL" if i < n_useful else "ruido"
    print(f"  Feature {i:2d}:  {weights_l2[i].item():+8.4f}  {weights_l1[i].item():+8.4f}  {tipo:>6s}")

print(f"""
  L2: pesos chicos pero NINGUNO exactamente 0
      Efecto: "todos los features importan un poquito"

  L1: features de ruido en CERO, utiles con pesos grandes
      Efecto: "seleccion automatica de features"

  L1 es como pedirle a la red:
    "Usa la MENOR cantidad de features posible"

  L2 es como pedirle:
    "Usa todos los features, pero sin exagerar ninguno"
""")
