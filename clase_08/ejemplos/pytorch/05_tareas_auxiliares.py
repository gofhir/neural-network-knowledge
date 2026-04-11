"""
Ejemplo 5 — Tareas Auxiliares paso a paso

Objetivo: entrenar una red con DOS tareas al mismo tiempo.
          La tarea auxiliar ayuda a la principal a aprender
          mejores representaciones internas.

Usamos un ejemplo simple:
  Tarea principal:  clasificar numeros pares vs impares
  Tarea auxiliar:   predecir si el numero es mayor que 50

Ejecutar:
  docker run --rm clase8 python -u ejemplos/pytorch/05_tareas_auxiliares.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

SEPARATOR = "=" * 60

# =============================================
# 1. Que es una tarea auxiliar
# =============================================
print(SEPARATOR)
print("1. QUE ES UNA TAREA AUXILIAR")
print(SEPARATOR)

print("""
  Normalmente una red aprende UNA tarea:
    Imagen → Red → ¿Sonrie? (si/no)

  Con tarea auxiliar, aprende DOS tareas al mismo tiempo:
    Imagen → Capas compartidas ─→ ¿Sonrie?  (tarea principal)
                                └→ ¿Joven?   (tarea auxiliar)

  ¿Por que ayuda?
    Las capas compartidas aprenden features MAS RICAS
    porque tienen que servir para AMBAS tareas.
    Es como estudiar para dos pruebas a la vez:
    aprendes el tema mas profundamente.
""")

# =============================================
# 2. Crear datos
# =============================================
print(f"\n{SEPARATOR}")
print("2. CREAR DATOS")
print(SEPARATOR)

torch.manual_seed(42)

# Numeros del 0 al 99 con algo de ruido
n_samples = 500
numbers = torch.randint(0, 100, (n_samples,)).float()
# Agregar features ruidosos (como si fueran pixeles)
noise = torch.randn(n_samples, 8) * 0.3
X = torch.cat([numbers.unsqueeze(1), noise], dim=1)  # (500, 9)

# Tarea principal: ¿es par?
y_main = (numbers % 2 == 0).float()

# Tarea auxiliar: ¿es mayor que 50?
y_aux = (numbers > 50).float()

# Dividir en train/test
X_train, X_test = X[:400], X[400:]
y_main_train, y_main_test = y_main[:400], y_main[400:]
y_aux_train, y_aux_test = y_aux[:400], y_aux[400:]

print(f"  {n_samples} numeros del 0 al 99")
print(f"  Features: 1 numero real + 8 features de ruido = 9 features")
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"")
print(f"  Tarea principal: ¿es par?       (Ej: 42 → si, 37 → no)")
print(f"  Tarea auxiliar:  ¿es mayor a 50? (Ej: 72 → si, 23 → no)")

# =============================================
# 3. Red con UNA sola tarea (baseline)
# =============================================
print(f"\n{SEPARATOR}")
print("3. RED CON UNA SOLA TAREA (baseline)")
print(SEPARATOR)


class SingleTaskNet(nn.Module):
    """Red que solo predice la tarea principal."""
    def __init__(self):
        super().__init__()
        # Capas compartidas
        self.shared = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Cabeza principal (unica)
        self.head_main = nn.Linear(32, 1)

    def forward(self, x):
        features = self.shared(x)
        main_out = self.head_main(features)
        return main_out


print(f"  Arquitectura:")
print(f"    Entrada(9) → [64 → ReLU → 32 → ReLU] → Salida(1)")
print(f"                  ^^^^^^^^^^^^^^^^^^^^^^^^")
print(f"                  capas compartidas")

# =============================================
# 4. Red con DOS tareas
# =============================================
print(f"\n{SEPARATOR}")
print("4. RED CON DOS TAREAS (principal + auxiliar)")
print(SEPARATOR)


class MultiTaskNet(nn.Module):
    """Red que predice la tarea principal Y la auxiliar."""
    def __init__(self):
        super().__init__()
        # MISMAS capas compartidas
        self.shared = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Cabeza principal: ¿es par?
        self.head_main = nn.Linear(32, 1)
        # Cabeza auxiliar: ¿mayor que 50?
        self.head_aux = nn.Linear(32, 1)

    def forward(self, x):
        features = self.shared(x)                # capas compartidas
        main_out = self.head_main(features)       # prediccion principal
        aux_out = self.head_aux(features)         # prediccion auxiliar
        return main_out, aux_out                  # devuelve las DOS


print(f"  Arquitectura:")
print(f"                                    ┌→ Cabeza principal (¿par?)")
print(f"    Entrada(9) → [64→ReLU→32→ReLU] ─┤")
print(f"                  ^^^^^^^^^^^^^^^^   └→ Cabeza auxiliar (¿>50?)")
print(f"                  COMPARTIDAS")
print(f"")
print(f"  Las capas compartidas aprenden features que sirven para AMBAS tareas.")

# =============================================
# 5. CombinedLoss
# =============================================
print(f"\n{SEPARATOR}")
print("5. COMBINEDLOSS: como sumar los dos losses")
print(SEPARATOR)

print("""
  Cada tarea produce un loss. Se combinan con un peso λ (lambda):

    Loss_total = Loss_principal + λ * Loss_auxiliar

    λ = 0.0: la auxiliar no tiene efecto (es como no tenerla)
    λ = 0.5: la auxiliar tiene la mitad de importancia
    λ = 1.0: ambas tienen la misma importancia
""")


class CombinedLoss(nn.Module):
    """Combina loss principal + auxiliar con un peso lambda."""
    def __init__(self, aux_weight=0.5):
        super().__init__()
        self.aux_weight = aux_weight  # λ
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, main_pred, aux_pred, main_label, aux_label):
        main_loss = self.bce(main_pred.squeeze(), main_label)
        aux_loss = self.bce(aux_pred.squeeze(), aux_label)
        total = main_loss + self.aux_weight * aux_loss
        return total, main_loss.item(), aux_loss.item()


print(f"  Loss_total = BCELoss(par?, real_par) + λ * BCELoss(>50?, real_>50)")

# =============================================
# 6. Entrenar y comparar
# =============================================
print(f"\n{SEPARATOR}")
print("6. ENTRENAR Y COMPARAR")
print(SEPARATOR)


def train_single(epochs=200):
    """Entrena red de una sola tarea."""
    torch.manual_seed(42)
    model = SingleTaskNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        pred = model(X_train).squeeze()
        loss = loss_fn(pred, y_main_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).squeeze()
        acc = ((test_pred > 0).float() == y_main_test).float().mean() * 100
    return acc.item()


def train_multi(aux_weight, epochs=200):
    """Entrena red multitarea."""
    torch.manual_seed(42)
    model = MultiTaskNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    combined_loss = CombinedLoss(aux_weight=aux_weight)

    for epoch in range(epochs):
        model.train()
        main_pred, aux_pred = model(X_train)
        loss, _, _ = combined_loss(main_pred, aux_pred, y_main_train, y_aux_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        main_pred, _ = model(X_test)
        acc = ((main_pred.squeeze() > 0).float() == y_main_test).float().mean() * 100
    return acc.item()


# Entrenar variantes
print(f"\n  Entrenando 5 variantes...\n")

acc_single = train_single()
print(f"  1. Una sola tarea:              Test Acc = {acc_single:.1f}%")

acc_multi_0 = train_multi(aux_weight=0.0)
print(f"  2. Multitarea (λ=0.0, ignorar): Test Acc = {acc_multi_0:.1f}%")

acc_multi_02 = train_multi(aux_weight=0.2)
print(f"  3. Multitarea (λ=0.2):          Test Acc = {acc_multi_02:.1f}%")

acc_multi_05 = train_multi(aux_weight=0.5)
print(f"  4. Multitarea (λ=0.5):          Test Acc = {acc_multi_05:.1f}%")

acc_multi_10 = train_multi(aux_weight=1.0)
print(f"  5. Multitarea (λ=1.0):          Test Acc = {acc_multi_10:.1f}%")

# =============================================
# 7. Paso a paso: que pasa en el forward
# =============================================
print(f"\n{SEPARATOR}")
print("7. PASO A PASO: que pasa en el forward multitarea")
print(SEPARATOR)

torch.manual_seed(42)
model_demo = MultiTaskNet()

# Un solo ejemplo
sample = X_test[0:1]
number = sample[0, 0].item()
is_even = y_main_test[0].item()
is_gt50 = y_aux_test[0].item()

model_demo.eval()
with torch.no_grad():
    main_pred, aux_pred = model_demo(sample)
    main_prob = torch.sigmoid(main_pred).item()
    aux_prob = torch.sigmoid(aux_pred).item()

print(f"""
  Numero: {number:.0f}
  ¿Es par?      Real: {'si' if is_even else 'no'}
  ¿Es mayor 50? Real: {'si' if is_gt50 else 'no'}

  Forward pass (red SIN entrenar):

    sample(9 features)
        ↓
    Capas compartidas: Linear(9→64) → ReLU → Linear(64→32) → ReLU
        ↓
    features (32 numeros): estos features sirven para AMBAS tareas
        ↓                     ↓
    head_main(32→1)       head_aux(32→1)
        ↓                     ↓
    main_logit: {main_pred.item():.4f}     aux_logit: {aux_pred.item():.4f}
        ↓                     ↓
    sigmoid:    {main_prob:.4f}     sigmoid:   {aux_prob:.4f}
        ↓                     ↓
    ¿par?: {'si' if main_prob > 0.5 else 'no'} ({main_prob:.1%})      ¿>50?: {'si' if aux_prob > 0.5 else 'no'} ({aux_prob:.1%})

    (Predicciones sin sentido porque no ha entrenado)

  Loss:
    main_loss = BCELoss({main_prob:.4f}, {is_even:.0f}) = {nn.BCEWithLogitsLoss()(main_pred.squeeze(), torch.tensor(is_even)).item():.4f}
    aux_loss  = BCELoss({aux_prob:.4f}, {is_gt50:.0f}) = {nn.BCEWithLogitsLoss()(aux_pred.squeeze(), torch.tensor(is_gt50)).item():.4f}
    total     = main_loss + λ * aux_loss
""")

# =============================================
# 8. Cuando usar y cuando NO
# =============================================
print(f"{SEPARATOR}")
print("8. CUANDO USAR TAREAS AUXILIARES")
print(SEPARATOR)
print(f"""
  UTIL cuando:
    - La tarea principal tiene pocos datos
    - Hay tareas RELACIONADAS con datos disponibles
    - Las tareas comparten estructura
      (ej: ambas necesitan entender caras)

  NO util cuando:
    - Las tareas NO estan relacionadas
      (ej: detectar sonrisas + predecir el clima)
    - λ esta mal calibrado
      (muy alto: la auxiliar domina y la principal empeora)

  CUIDADO con la escala:
    Si main_loss ≈ 0.5 y aux_loss ≈ 500:
      Loss = 0.5 + λ * 500  → la auxiliar DOMINA!
      Solucion: usar λ muy chico (0.001)

  En la practica (del laboratorio):
    - Tarea principal: detectar sonrisas (Smiling)
    - Tarea auxiliar binaria: detectar juventud (Young), λ=0.2
    - Tarea auxiliar regresion: predecir landmarks faciales, λ=0.1
""")
