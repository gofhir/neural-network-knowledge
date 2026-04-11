"""
Lab Experimento 3 — Tareas Auxiliares y preguntas 1-2

Reproduce el experimento 3 del laboratorio con datos sinteticos:
Red con cabeza principal + cabeza auxiliar, CombinedLoss.

Ejecutar:
  docker run --rm clase8 python -u ejemplos_c8/pytorch/10_lab_exp3_tareas_auxiliares.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

SEPARATOR = "=" * 60

# =============================================
# 1. Contexto: el lab usa CelebA
# =============================================
print(SEPARATOR)
print("EXPERIMENTO 3: TAREAS AUXILIARES")
print(SEPARATOR)

print("""
  En el lab se usa CelebA (caras de celebridades).
  Aqui usamos datos sinteticos para que se pueda correr rapido,
  pero la ARQUITECTURA y el CODIGO son identicos.

  Lab:
    Tarea principal: ¿Sonrie? (Smiling) → binario
    Tarea auxiliar:  ¿Joven? (Young) → binario
    O auxiliar:      Landmarks (coordenadas x,y de ojos, nariz, boca) → regresion

  Aqui:
    Tarea principal: ¿Numero par? → binario
    Tarea auxiliar:  ¿Mayor que 50? → binario
    O auxiliar:      Valor exacto del numero → regresion
""")

# =============================================
# 2. Datos sinteticos
# =============================================
torch.manual_seed(42)

n_samples = 800
n_test = 200

# Features: 20 dimensiones (simulando features de una imagen)
X = torch.randn(n_samples + n_test, 20)
# El "numero real" escondido en los datos (como el contenido de una imagen)
numbers = (X[:, 0] * 30 + 50).clamp(0, 100)  # numero entre 0-100

# Tareas
y_main = (numbers % 2 < 1).float()       # ¿es par? (la principal no es obvia)
y_aux_binary = (numbers > 50).float()     # ¿mayor que 50?
y_aux_regression = numbers / 100.0        # valor normalizado (regresion)

# Split
X_train, X_test = X[:n_samples], X[n_samples:]
y_main_train, y_main_test = y_main[:n_samples], y_main[n_samples:]
y_aux_bin_train, y_aux_bin_test = y_aux_binary[:n_samples], y_aux_binary[n_samples:]
y_aux_reg_train, y_aux_reg_test = y_aux_regression[:n_samples], y_aux_regression[n_samples:]

print(f"  Train: {n_samples}, Test: {n_test}")
print(f"  Features: 20 dimensiones")

# =============================================
# 3. Arquitectura: igual que el lab
# =============================================
print(f"\n{SEPARATOR}")
print("ARQUITECTURA (como en el lab)")
print(SEPARATOR)


class FaceModel(nn.Module):
    """Modelo del lab (adaptado a datos sinteticos)."""
    def __init__(self, auxiliary_task_dim=None):
        super().__init__()
        self.auxiliary_task_dim = auxiliary_task_dim

        # Capas compartidas (en el lab son Conv2d, aqui Linear)
        self.fc1 = nn.Linear(20, 120)
        self.fc2 = nn.Linear(120, 84)

        # Cabeza principal (siempre)
        self.fc3 = nn.Linear(84, 1)

        # Cabeza auxiliar (opcional)
        if auxiliary_task_dim is not None:
            self.fc4 = nn.Linear(84, auxiliary_task_dim)

    def forward(self, x):
        # Capas compartidas
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Dos salidas
        main_task = self.fc3(x)
        if self.auxiliary_task_dim is not None:
            auxiliary_task = self.fc4(x)
            return main_task, auxiliary_task
        else:
            return main_task, None


print("""
                                    ┌→ fc3 (1) → ¿Tarea principal?
  Input(20) → fc1(120) → fc2(84) ─┤
       COMPARTIDAS                  └→ fc4 (N) → ¿Tarea auxiliar?
""")

# =============================================
# 4. CombinedLoss (EXACTO del lab)
# =============================================
print(SEPARATOR)
print("COMBINEDLOSS (codigo del laboratorio)")
print(SEPARATOR)


class CombinedLoss(nn.Module):
    """Igual que en el lab."""
    def __init__(self, auxiliary_task, auxiliary_weight):
        super().__init__()
        self.auxiliary_task = auxiliary_task
        self.aux_weight = auxiliary_weight  # λ

    def forward(self, main_pred, aux_pred, main_labels, aux_labels):
        if aux_labels is None:
            return F.binary_cross_entropy_with_logits(main_pred.squeeze(), main_labels)
        else:
            main_loss = F.binary_cross_entropy_with_logits(main_pred.squeeze(), main_labels)

            if self.auxiliary_task == 'regression':
                aux_loss = F.mse_loss(aux_pred.squeeze(), aux_labels)
            else:
                aux_loss = F.binary_cross_entropy_with_logits(aux_pred.squeeze(), aux_labels)

            return main_loss + self.aux_weight * aux_loss


print("""
  Si no hay tarea auxiliar:
    Loss = BCE(main_pred, main_label)

  Si hay tarea auxiliar binaria:
    Loss = BCE(main_pred, main_label) + λ * BCE(aux_pred, aux_label)

  Si hay tarea auxiliar de regresion (Landmarks):
    Loss = BCE(main_pred, main_label) + λ * MSE(aux_pred, aux_label)

  Nota: la tarea auxiliar puede usar una loss DISTINTA a la principal!
""")

# =============================================
# 5. Entrenar los 3 escenarios del lab
# =============================================
def train_model(auxiliary_task, auxiliary_weight, aux_dim, y_aux_train, y_aux_test, epochs=100):
    torch.manual_seed(42)
    model = FaceModel(auxiliary_task_dim=aux_dim)
    loss_fn = CombinedLoss(auxiliary_task, auxiliary_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        main_pred, aux_pred = model(X_train)
        loss = loss_fn(main_pred, aux_pred, y_main_train,
                       y_aux_train if aux_dim else None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        main_pred, _ = model(X_test)
        acc = ((main_pred.squeeze() > 0).float() == y_main_test).float().mean() * 100
    return acc.item()


print(f"\n{SEPARATOR}")
print("ESCENARIO 1: Solo tarea principal (sin auxiliar)")
print(SEPARATOR)
print("  Como en el lab: primary_task='Smiling', auxiliary_task=None")
acc1 = train_model(None, None, None, None, None)
print(f"  Test Accuracy (tarea principal): {acc1:.1f}%")

print(f"\n{SEPARATOR}")
print("ESCENARIO 2: Principal + auxiliar binaria (λ=0.2)")
print(SEPARATOR)
print("  Como en el lab: primary='Smiling', auxiliary='Young', λ=0.2")
acc2 = train_model('binary', 0.2, 1, y_aux_bin_train, y_aux_bin_test)
print(f"  Test Accuracy (tarea principal): {acc2:.1f}%")

print(f"\n{SEPARATOR}")
print("ESCENARIO 3: Principal + auxiliar regresion (λ=0.1)")
print(SEPARATOR)
print("  Como en el lab: primary='Young', auxiliary='Landmarks', λ=0.1")
acc3 = train_model('regression', 0.1, 1, y_aux_reg_train, y_aux_reg_test)
print(f"  Test Accuracy (tarea principal): {acc3:.1f}%")

# Resumen
print(f"\n{SEPARATOR}")
print("COMPARACION")
print(SEPARATOR)
print(f"""
  {'Escenario':<35s}  {'Test Acc':>8s}
  {'─'*35}  {'─'*8}
  Solo tarea principal                {acc1:6.1f}%
  + auxiliar binaria (λ=0.2)          {acc2:6.1f}%
  + auxiliar regresion (λ=0.1)        {acc3:6.1f}%
""")

# =============================================
# 6. Preguntas del lab
# =============================================
print(SEPARATOR)
print("PREGUNTA 1: aux_loss es 1000x mayor que main_loss")
print(SEPARATOR)
print("""
  Problema:
    main_loss ≈ 0.5   (Cross-Entropy, valores 0-5)
    aux_loss  ≈ 500   (MSE de Landmarks, valores grandes)

  Si usas λ=1:
    Loss = 0.5 + 1 * 500 = 500.5
    → 99.9% del loss viene de la auxiliar
    → La red IGNORA la tarea principal
    → Es como estudiar para la prueba equivocada

  Solucion: λ = escala_main / escala_aux ≈ 0.5/500 = 0.001
    Loss = 0.5 + 0.001 * 500 = 1.0
    → Ambas contribuyen de forma similar
""")

# Demostrar con numeros
print(f"  Demostracion numerica:\n")
main_l = 0.5
aux_l = 500.0
for lam in [1.0, 0.1, 0.01, 0.001]:
    total = main_l + lam * aux_l
    pct_main = main_l / total * 100
    pct_aux = (lam * aux_l) / total * 100
    print(f"    λ={lam:<6.3f}: Loss={total:8.1f}  (main={pct_main:.0f}%, aux={pct_aux:.0f}%)")

print(f"\n  → Con λ=0.001, ambos aportan ~50% cada uno")

print(f"\n{SEPARATOR}")
print("PREGUNTA 2: 1 principal + 4 auxiliares al 50/50")
print(SEPARATOR)
print("""
  Queremos:
    50% para la principal
    50% repartido entre 4 auxiliares (12.5% cada una)

  Si la principal tiene peso implicito = 1:
    Cada auxiliar necesita peso λ = 0.25

  Loss = main_loss + 0.25*aux1 + 0.25*aux2 + 0.25*aux3 + 0.25*aux4

  Verificacion:
    Peso principal:  1.0                    → 1.0/2.0 = 50% ✓
    Peso auxiliares: 0.25*4 = 1.0           → 1.0/2.0 = 50% ✓
    Cada auxiliar:   0.25/2.0 = 12.5%       ✓
""")

# Demostrar
main_loss = torch.tensor(0.5)
aux_losses = [torch.tensor(0.4), torch.tensor(0.6),
              torch.tensor(0.5), torch.tensor(0.3)]
lambda_each = 0.25

total = main_loss + sum(lambda_each * a for a in aux_losses)
pct_main = main_loss / total * 100

print(f"  Ejemplo numerico:")
print(f"    main_loss = {main_loss.item():.1f}")
for i, a in enumerate(aux_losses):
    print(f"    aux{i+1}_loss = {a.item():.1f}  (× {lambda_each} = {lambda_each * a.item():.2f})")
print(f"    total     = {total.item():.2f}")
print(f"    % principal: {pct_main.item():.0f}%")
print(f"    % auxiliares: {100-pct_main.item():.0f}%")

print(f"""
  En codigo:
    loss = main_loss
    for aux_loss in [aux1, aux2, aux3, aux4]:
        loss = loss + 0.25 * aux_loss

  OJO: esto asume que todas las losses tienen ESCALAS SIMILARES.
  Si alguna es 1000x mayor, ajustar su λ individual.
""")
