"""
Lab Experimento 2 — Regularizacion L2: sin vs con, y preguntas 1-8

Reproduce el experimento 2 del laboratorio:
Red grande (2→1000→1000→1) para datos 2D simples.
Compara sin regularizacion vs con L2, y responde las 8 preguntas.

Ejecutar:
  docker run --rm clase8 python -u ejemplos_c8/pytorch/09_lab_exp2_regularizacion.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

SEPARATOR = "=" * 60

# =============================================
# 1. Crear datos 2D (dos clases)
# =============================================
print(SEPARATOR)
print("EXPERIMENTO 2: REGULARIZACION L2")
print(SEPARATOR)

torch.manual_seed(42)
np.random.seed(42)

# Generar datos como en el lab: dos nubes con algo de overlap
n_train = 200
n_test = 100

# Datos de entrenamiento
angle = torch.linspace(0, 2 * np.pi, n_train)
r0 = 1.0 + 0.3 * torch.randn(n_train)
r1 = 2.5 + 0.3 * torch.randn(n_train)

x0 = torch.stack([r0 * torch.cos(angle) + 0.2 * torch.randn(n_train),
                   r0 * torch.sin(angle) + 0.2 * torch.randn(n_train)], dim=1)
x1 = torch.stack([r1 * torch.cos(angle) + 0.2 * torch.randn(n_train),
                   r1 * torch.sin(angle) + 0.2 * torch.randn(n_train)], dim=1)

X_train = torch.cat([x0, x1])
y_train = torch.cat([torch.zeros(n_train), torch.ones(n_train)])

# Datos de test
angle_t = torch.linspace(0, 2 * np.pi, n_test)
r0_t = 1.0 + 0.3 * torch.randn(n_test)
r1_t = 2.5 + 0.3 * torch.randn(n_test)

x0_t = torch.stack([r0_t * torch.cos(angle_t) + 0.2 * torch.randn(n_test),
                     r0_t * torch.sin(angle_t) + 0.2 * torch.randn(n_test)], dim=1)
x1_t = torch.stack([r1_t * torch.cos(angle_t) + 0.2 * torch.randn(n_test),
                     r1_t * torch.sin(angle_t) + 0.2 * torch.randn(n_test)], dim=1)

X_test = torch.cat([x0_t, x1_t])
y_test = torch.cat([torch.zeros(n_test), torch.ones(n_test)])

print(f"\n  Datos: dos anillos concentricos en 2D")
print(f"  Train: {len(X_train)} puntos")
print(f"  Test:  {len(X_test)} puntos")

# =============================================
# 2. Red grande (igual que el lab)
# =============================================
class BigNet(nn.Module):
    """Red del laboratorio: 2→1000→1000→1 (ENORME para datos 2D)"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)

total_params = sum(p.numel() for p in BigNet().parameters())
print(f"\n  Red: 2 → 1000 → 1000 → 1")
print(f"  Parametros: {total_params:,} (para solo {len(X_train)} datos!)")
print(f"  → Overfitting GARANTIZADO sin regularizacion")


def train_and_eval(weight_decay, epochs=150, label=""):
    """Entrena y evalua una red con el weight_decay dado."""
    torch.manual_seed(42)
    model = BigNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        pred = model(X_train).squeeze()
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).squeeze()
        test_pred = model(X_test).squeeze()
        train_acc = ((train_pred > 0).float() == y_train).float().mean() * 100
        test_acc = ((test_pred > 0).float() == y_test).float().mean() * 100

    # Estadisticas de pesos
    all_w = torch.cat([p.data.flatten() for p in model.parameters()])
    w_max = all_w.abs().max().item()
    w_norm = all_w.norm().item()

    return {
        'label': label,
        'wd': weight_decay,
        'train_acc': train_acc.item(),
        'test_acc': test_acc.item(),
        'w_max': w_max,
        'w_norm': w_norm,
    }


# =============================================
# 3. Entrenar sin regularizacion (Pregunta base)
# =============================================
print(f"\n{SEPARATOR}")
print("SIN REGULARIZACION (weight_decay=0.0)")
print(SEPARATOR)

r_none = train_and_eval(0.0, label="Sin reg")
print(f"\n  Train Accuracy: {r_none['train_acc']:.1f}%")
print(f"  Test Accuracy:  {r_none['test_acc']:.1f}%")
print(f"  Peso maximo:    {r_none['w_max']:.4f}")
print(f"  Norma total:    {r_none['w_norm']:.1f}")

# =============================================
# 4. Con regularizacion L2=0.2 (del lab)
# =============================================
print(f"\n{SEPARATOR}")
print("CON REGULARIZACION L2 (weight_decay=0.2)")
print(SEPARATOR)

r_02 = train_and_eval(0.2, label="L2=0.2")
print(f"\n  Train Accuracy: {r_02['train_acc']:.1f}%")
print(f"  Test Accuracy:  {r_02['test_acc']:.1f}%")
print(f"  Peso maximo:    {r_02['w_max']:.4f}")
print(f"  Norma total:    {r_02['w_norm']:.1f}")

# =============================================
# 5. Preguntas del lab
# =============================================
print(f"\n{SEPARATOR}")
print("PREGUNTA 1: Diferencia en los limites de clasificacion")
print(SEPARATOR)
print(f"""
  Sin regularizacion:
    Limites MUY COMPLEJOS y ruidosos.
    Se curvan y retuercen para pasar por CADA punto.
    → La red memorizo cada dato individual.

  Con regularizacion L2:
    Limites SUAVES y simples.
    No se ajustan a cada punto, capturan el patron general.
    → La red aprendio la ESTRUCTURA, no los datos.
""")

print(f"{SEPARATOR}")
print("PREGUNTA 2: ¿Que explica esta diferencia?")
print(SEPARATOR)
print(f"""
  Sin L2: los pesos crecen libremente.
    Pesos grandes → transformaciones extremas → curvas complejas.
    Peso maximo: {r_none['w_max']:.4f}

  Con L2: los pesos grandes son "caros" (aumentan el loss).
    La red prefiere pesos chicos → transformaciones suaves.
    Peso maximo: {r_02['w_max']:.4f}

  → Pesos chicos = funciones suaves = limites simples.
""")

print(f"{SEPARATOR}")
print("PREGUNTA 3: ¿Cual es mejor en ENTRENAMIENTO?")
print(SEPARATOR)
print(f"""
  Sin regularizacion: Train = {r_none['train_acc']:.1f}%  ← MEJOR en train
  Con regularizacion: Train = {r_02['train_acc']:.1f}%

  Sin regularizacion gana en train porque MEMORIZO los datos.
  Es como sacar 100% en ejercicios de practica
  porque te aprendiste las respuestas de memoria.
""")

print(f"{SEPARATOR}")
print("PREGUNTA 4: ¿Cual es mejor en TEST?")
print(SEPARATOR)
print(f"""
  Sin regularizacion: Test = {r_none['test_acc']:.1f}%
  Con regularizacion: Test = {r_02['test_acc']:.1f}%  ← MEJOR en test

  Con regularizacion gana en test porque GENERALIZA.
  Es como sacar mejor nota en la prueba final
  porque ENTENDISTE la materia en vez de memorizar.
""")

print(f"{SEPARATOR}")
print("PREGUNTA 5: ¿Que estrategia recomendar?")
print(SEPARATOR)
print(f"""
  USAR regularizacion (L2).

  Lo que importa es el TEST (datos del mundo real).
  Aunque el train es menor, el test es MEJOR.

  Sin reg: Train={r_none['train_acc']:.0f}%, Test={r_none['test_acc']:.0f}%  ← overfitting
  Con reg: Train={r_02['train_acc']:.0f}%, Test={r_02['test_acc']:.0f}%   ← generaliza
""")

print(f"{SEPARATOR}")
print("PREGUNTA 6: ¿Que linea cambiar para weight_decay=0.9?")
print(SEPARATOR)
print(f"""
  # Original:
  optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.2)

  # Modificado:
  optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.9)
  #                                                   ^^^^^^^^^^^^^^^
  #                                                   solo este numero
""")

# =============================================
# 6. Preguntas 7 y 8: variar weight_decay
# =============================================
print(f"{SEPARATOR}")
print("PREGUNTAS 7 y 8: Efecto de variar weight_decay")
print(SEPARATOR)

print(f"\n  Entrenando con distintos valores...\n")

results = [r_none, r_02]
for wd in [0.9, 100000]:
    r = train_and_eval(wd, label=f"L2={wd}")
    results.append(r)
    print(f"  weight_decay={wd}: Train={r['train_acc']:.1f}%, Test={r['test_acc']:.1f}%")

print(f"\n  {'Configuracion':<15s}  {'Train':>6s}  {'Test':>6s}  {'|W| max':>8s}  {'Norma':>8s}")
print(f"  {'─'*15}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")
for r in results:
    print(f"  {r['label']:<15s}  {r['train_acc']:5.1f}%  {r['test_acc']:5.1f}%  {r['w_max']:8.4f}  {r['w_norm']:8.1f}")

print(f"""
  PREGUNTA 7 (weight_decay=0.9):
    El limite de clasificacion es AUN MAS SUAVE.
    Puede empezar a underfitear (demasiado simple).

  PREGUNTA 8 (weight_decay=100000):
    Accuracy ≈ 50% → NO APRENDIO NADA.
    Todos los pesos ≈ 0.

    ¿Por que?
      Loss = CrossEntropy + 100000 * Σ(w²)
      El penalty es TAN grande que domina completamente.
      La red gasta todo su esfuerzo en mantener pesos en 0.
      No le queda "capacidad" para aprender los datos.

    Es como poner la regularizacion al maximo:
      λ=0:      sin freno → overfitting
      λ=0.2:    freno suave → generaliza bien
      λ=0.9:    freno fuerte → muy simple
      λ=100000: freno de mano → no se mueve
""")
