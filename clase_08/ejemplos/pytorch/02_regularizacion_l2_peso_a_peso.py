"""
Ejemplo 2 — Regularizacion L2 (Weight Decay) paso a paso

Objetivo: ver EXACTAMENTE que hace L2 a los pesos de la red.
          Entrenamos dos redes identicas: una sin L2 y otra con L2,
          y comparamos como quedan los pesos.

Ejecutar:
  docker run --rm clase8 python -u ejemplos/pytorch/02_regularizacion_l2_peso_a_peso.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

SEPARATOR = "=" * 60

# =============================================
# 1. Que es L2 matematicamente
# =============================================
print(SEPARATOR)
print("1. QUE ES L2 (Weight Decay)")
print(SEPARATOR)

print("""
  Sin L2:
    Loss = CrossEntropy(prediccion, real)
    → Solo le importa predecir bien

  Con L2:
    Loss = CrossEntropy(prediccion, real) + λ * Σ(w²)
                                            ↑     ↑
                                         lambda  suma de todos
                                                 los pesos al cuadrado
    → Le importa predecir bien Y tener pesos chicos
""")

# Demostrar la formula con numeros
weights = torch.tensor([10.0, -5.0, 3.0, -2.0, 0.5])
l2_penalty = (weights ** 2).sum()
lambda_val = 0.01

print(f"  Ejemplo con pesos: {weights.tolist()}")
print(f"  Σ(w²) = {' + '.join(f'{w**2:.1f}' for w in weights)} = {l2_penalty.item():.1f}")
print(f"  λ * Σ(w²) = {lambda_val} * {l2_penalty.item():.1f} = {lambda_val * l2_penalty.item():.2f}")
print(f"")
print(f"  Si CrossEntropy = 0.5:")
print(f"    Loss sin L2 = 0.5")
print(f"    Loss con L2 = 0.5 + {lambda_val * l2_penalty.item():.2f} = {0.5 + lambda_val * l2_penalty.item():.2f}")
print(f"")
print(f"  Los pesos grandes (10.0, -5.0) 'cuestan' mas que los chicos (0.5)")

# =============================================
# 2. Crear datos sinteticos
# =============================================
print(f"\n{SEPARATOR}")
print("2. CREAR DATOS (clasificacion binaria simple)")
print(SEPARATOR)

torch.manual_seed(42)
np.random.seed(42)

# Dos nubes de puntos en 2D (facil de separar)
n_samples = 100
x_class0 = torch.randn(n_samples, 2) + torch.tensor([-1.0, -1.0])
x_class1 = torch.randn(n_samples, 2) + torch.tensor([1.0, 1.0])
X = torch.cat([x_class0, x_class1])
y = torch.cat([torch.zeros(n_samples), torch.ones(n_samples)])

print(f"  {n_samples * 2} puntos en 2D, 2 clases")
print(f"  Clase 0: centrada en (-1, -1)")
print(f"  Clase 1: centrada en (+1, +1)")

# =============================================
# 3. Red grande (para que pueda hacer overfitting)
# =============================================
print(f"\n{SEPARATOR}")
print("3. RED GRANDE (muchos parametros para pocos datos)")
print(SEPARATOR)


class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Red MUY grande para solo 200 datos 2D
        self.fc1 = nn.Linear(2, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


total_params = sum(p.numel() for p in BigNet().parameters())
print(f"  Arquitectura: 2 → 500 → 500 → 1")
print(f"  Parametros: {total_params:,}")
print(f"  Datos: 200")
print(f"  → {total_params:,} parametros para 200 datos = OVERFITTING seguro!")

# =============================================
# 4. Entrenar SIN regularizacion
# =============================================
print(f"\n{SEPARATOR}")
print("4. ENTRENAR SIN REGULARIZACION (weight_decay=0.0)")
print(SEPARATOR)

torch.manual_seed(42)
model_no_reg = BigNet()
optimizer_no_reg = optim.Adam(model_no_reg.parameters(), lr=0.001, weight_decay=0.0)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(200):
    pred = model_no_reg(X).squeeze()
    loss = loss_fn(pred, y)
    optimizer_no_reg.zero_grad()
    loss.backward()
    optimizer_no_reg.step()

    if (epoch + 1) % 50 == 0:
        accuracy = ((pred > 0).float() == y).float().mean() * 100
        print(f"    Epoca {epoch+1}: Loss={loss.item():.4f}, Acc={accuracy:.1f}%")

# =============================================
# 5. Entrenar CON regularizacion L2
# =============================================
print(f"\n{SEPARATOR}")
print("5. ENTRENAR CON REGULARIZACION L2 (weight_decay=0.01)")
print(SEPARATOR)

torch.manual_seed(42)
model_l2 = BigNet()
optimizer_l2 = optim.Adam(model_l2.parameters(), lr=0.001, weight_decay=0.01)
#                                                          ^^^^^^^^^^^^^^^
#                                                          la UNICA diferencia

for epoch in range(200):
    pred = model_l2(X).squeeze()
    loss = loss_fn(pred, y)
    optimizer_l2.zero_grad()
    loss.backward()
    optimizer_l2.step()

    if (epoch + 1) % 50 == 0:
        accuracy = ((pred > 0).float() == y).float().mean() * 100
        print(f"    Epoca {epoch+1}: Loss={loss.item():.4f}, Acc={accuracy:.1f}%")

# =============================================
# 6. Comparar los PESOS
# =============================================
print(f"\n{SEPARATOR}")
print("6. COMPARAR LOS PESOS")
print(SEPARATOR)

# Pesos de la primera capa
w_no_reg = model_no_reg.fc1.weight.data.flatten()
w_l2 = model_l2.fc1.weight.data.flatten()

print(f"\n  Estadisticas de los pesos de fc1 ({w_no_reg.numel()} pesos):")
print(f"")
print(f"                     Sin L2        Con L2")
print(f"    Media:           {w_no_reg.mean():+.4f}       {w_l2.mean():+.4f}")
print(f"    Std:             {w_no_reg.std():.4f}        {w_l2.std():.4f}")
print(f"    Min:             {w_no_reg.min():+.4f}       {w_l2.min():+.4f}")
print(f"    Max:             {w_no_reg.max():+.4f}       {w_l2.max():+.4f}")
print(f"    Norma L2:        {w_no_reg.norm():.4f}       {w_l2.norm():.4f}")

# Contar pesos "grandes"
threshold = 0.5
n_big_no_reg = (w_no_reg.abs() > threshold).sum().item()
n_big_l2 = (w_l2.abs() > threshold).sum().item()
print(f"    Pesos > {threshold}:      {n_big_no_reg}            {n_big_l2}")

print(f"""
  → Sin L2: los pesos crecen libremente, algunos son MUY grandes
  → Con L2: los pesos se mantienen chicos y distribuidos
  → L2 "penaliza" los pesos grandes, forzando a la red
    a distribuir la importancia entre muchos pesos chicos
""")

# =============================================
# 7. Que pasa con λ muy grande
# =============================================
print(f"{SEPARATOR}")
print("7. QUE PASA CON weight_decay MUY GRANDE")
print(SEPARATOR)

torch.manual_seed(42)
model_extreme = BigNet()
optimizer_extreme = optim.Adam(model_extreme.parameters(), lr=0.001, weight_decay=10.0)

for epoch in range(200):
    pred = model_extreme(X).squeeze()
    loss = loss_fn(pred, y)
    optimizer_extreme.zero_grad()
    loss.backward()
    optimizer_extreme.step()

accuracy = ((model_extreme(X).squeeze() > 0).float() == y).float().mean() * 100
w_extreme = model_extreme.fc1.weight.data.flatten()

print(f"\n  weight_decay = 10.0 (extremo):")
print(f"    Accuracy: {accuracy:.1f}% ← no aprendio NADA!")
print(f"    Max peso: {w_extreme.max():.6f}")
print(f"    Min peso: {w_extreme.min():.6f}")
print(f"    → Todos los pesos son ~0. La penalizacion es TAN fuerte")
print(f"      que la red no puede tener pesos significativos.")

print(f"""
  Valores tipicos de weight_decay:
    0.0:     sin regularizacion
    0.0001:  sutil (comun en practica)
    0.001:   moderada
    0.01:    fuerte
    0.1-1.0: muy fuerte
    10+:     la red no aprende
""")
