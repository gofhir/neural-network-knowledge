"""
Ejemplo 4 — Comparar TODAS las regularizaciones

Objetivo: entrenar la MISMA red con L1, L2, Dropout y sin nada,
          y comparar overfitting, pesos y accuracy.

Ejecutar:
  docker run --rm clase8 python -u ejemplos/pytorch/04_comparar_regularizaciones.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

SEPARATOR = "=" * 60

# =============================================
# Datos: espiral 2D (dificil de separar linealmente)
# =============================================
torch.manual_seed(42)
np.random.seed(42)

n_points = 150

# Crear espiral de 2 clases
def make_spiral(n, noise=0.5):
    theta = torch.linspace(0, 4 * np.pi, n)
    r = torch.linspace(0.5, 3, n)
    x0 = torch.stack([r * torch.cos(theta) + noise * torch.randn(n),
                       r * torch.sin(theta) + noise * torch.randn(n)], dim=1)
    x1 = torch.stack([r * torch.cos(theta + np.pi) + noise * torch.randn(n),
                       r * torch.sin(theta + np.pi) + noise * torch.randn(n)], dim=1)
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(n), torch.ones(n)])
    return X, y

X_train, y_train = make_spiral(n_points)
X_test, y_test = make_spiral(50, noise=0.5)

print(SEPARATOR)
print("COMPARAR REGULARIZACIONES")
print(SEPARATOR)
print(f"\n  Datos: espiral 2D (dificil)")
print(f"  Train: {len(X_train)} puntos")
print(f"  Test:  {len(X_test)} puntos")


# =============================================
# Red base (grande para poder overfittear)
# =============================================
class BaseNet(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) if use_dropout else None

    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)
        return self.fc3(x)


def train_and_evaluate(model, optimizer, X_train, y_train, X_test, y_test,
                       epochs=300, l1_lambda=0.0, name=""):
    """Entrena y devuelve metricas finales."""
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        pred = model(X_train).squeeze()
        loss = loss_fn(pred, y_train)

        # L1 manual si aplica
        if l1_lambda > 0:
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluar
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).squeeze()
        test_pred = model(X_test).squeeze()
        train_acc = ((train_pred > 0).float() == y_train).float().mean() * 100
        test_acc = ((test_pred > 0).float() == y_test).float().mean() * 100

    # Estadisticas de pesos
    all_weights = torch.cat([p.data.flatten() for p in model.parameters()])
    n_zero = (all_weights.abs() < 0.01).sum().item()
    weight_norm = all_weights.norm().item()

    return {
        'name': name,
        'train_acc': train_acc.item(),
        'test_acc': test_acc.item(),
        'weight_norm': weight_norm,
        'n_zero': n_zero,
        'n_total': all_weights.numel(),
    }


# =============================================
# Entrenar 4 variantes
# =============================================
results = []

# 1. Sin regularizacion
print(f"\n  Entrenando sin regularizacion...")
torch.manual_seed(42)
model = BaseNet(use_dropout=False)
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
results.append(train_and_evaluate(model, opt, X_train, y_train, X_test, y_test, name="Sin nada"))

# 2. Con L2
print(f"  Entrenando con L2 (weight_decay=0.01)...")
torch.manual_seed(42)
model = BaseNet(use_dropout=False)
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
results.append(train_and_evaluate(model, opt, X_train, y_train, X_test, y_test, name="L2 (wd=0.01)"))

# 3. Con L1
print(f"  Entrenando con L1 (lambda=0.001)...")
torch.manual_seed(42)
model = BaseNet(use_dropout=False)
opt = optim.Adam(model.parameters(), lr=0.001)
results.append(train_and_evaluate(model, opt, X_train, y_train, X_test, y_test,
                                   l1_lambda=0.001, name="L1 (λ=0.001)"))

# 4. Con Dropout
print(f"  Entrenando con Dropout (p=0.5)...")
torch.manual_seed(42)
model = BaseNet(use_dropout=True)
opt = optim.Adam(model.parameters(), lr=0.001)
results.append(train_and_evaluate(model, opt, X_train, y_train, X_test, y_test, name="Dropout (0.5)"))

# 5. L2 + Dropout (combinado)
print(f"  Entrenando con L2 + Dropout...")
torch.manual_seed(42)
model = BaseNet(use_dropout=True)
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
results.append(train_and_evaluate(model, opt, X_train, y_train, X_test, y_test, name="L2 + Dropout"))

# =============================================
# Mostrar resultados
# =============================================
print(f"\n{SEPARATOR}")
print("RESULTADOS")
print(SEPARATOR)

print(f"\n  {'Metodo':<16s}  {'Train':>6s}  {'Test':>6s}  {'Overfit':>8s}  {'|W|':>8s}  {'W~0':>6s}")
print(f"  {'─' * 16}  {'─' * 6}  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 6}")

for r in results:
    overfit = r['train_acc'] - r['test_acc']
    overfit_str = f"{overfit:+.1f}%"
    print(f"  {r['name']:<16s}  {r['train_acc']:5.1f}%  {r['test_acc']:5.1f}%  {overfit_str:>8s}"
          f"  {r['weight_norm']:8.1f}  {r['n_zero']:5d}")

print(f"""
  Columnas:
    Train:   accuracy en datos de entrenamiento
    Test:    accuracy en datos NUNCA vistos
    Overfit: diferencia (grande = memorizo, malo)
    |W|:     norma de los pesos (chica = regularizado)
    W~0:     pesos cercanos a 0 (alto en L1 = sparsity)

  Observaciones:
    - Sin nada: train alto, test bajo → OVERFITTING
    - L2: pesos mas chicos (|W| bajo), mejor generalizacion
    - L1: muchos pesos en 0 (sparsity), selecciona features
    - Dropout: train mas bajo (apaga neuronas), test similar a L2
    - L2 + Dropout: combinar ayuda, regulariza por dos caminos
""")

# =============================================
# Como se implementa cada uno
# =============================================
print(f"{SEPARATOR}")
print("COMO IMPLEMENTAR CADA UNO")
print(SEPARATOR)
print(f"""
  # Sin regularizacion
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # L2 (un solo parametro!)
  optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

  # L1 (manual, agregar al loss)
  l1_penalty = sum(p.abs().sum() for p in model.parameters())
  loss = loss_fn(pred, y) + 0.001 * l1_penalty

  # Dropout (capa en la red)
  self.dropout = nn.Dropout(p=0.5)
  x = self.dropout(x)  # en el forward

  # Combinar L2 + Dropout
  optimizer = optim.Adam(..., weight_decay=0.01)  # L2
  self.dropout = nn.Dropout(p=0.5)                # Dropout
""")
