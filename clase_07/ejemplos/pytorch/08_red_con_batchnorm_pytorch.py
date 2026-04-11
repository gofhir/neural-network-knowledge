"""
Ejemplo 8 — Red neuronal completa CON BatchNorm (PyTorch)

Objetivo: ver como se usa BatchNorm DENTRO de una red real.
          Comparamos la misma red con y sin BatchNorm para ver
          como afecta las activaciones durante el forward pass.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/pytorch/08_red_con_batchnorm_pytorch.py
"""
import torch
import torch.nn as nn

# =============================================
# Paso 1: Red SIN BatchNorm
# =============================================
print("=" * 55)
print("Paso 1: Red SIN BatchNorm")
print("=" * 55)


class NetworkWithout_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Capa 1
        x = self.fc1(x)
        print(f"    Despues de fc1:  media={x.mean().item():+8.4f}, std={x.std().item():.4f}")
        x = self.relu(x)

        # Capa 2
        x = self.fc2(x)
        print(f"    Despues de fc2:  media={x.mean().item():+8.4f}, std={x.std().item():.4f}")
        x = self.relu(x)

        # Salida
        x = self.fc3(x)
        print(f"    Despues de fc3:  media={x.mean().item():+8.4f}, std={x.std().item():.4f}")
        return x


model_sin_bn = NetworkWithout_BN()

# Datos de entrada: batch de 8 muestras, 10 features
torch.manual_seed(42)
x = torch.randn(8, 10) * 5  # multiplicar por 5 para exagerar la escala

print(f"\nEntrada: {x.shape} (8 muestras, 10 features)")
print(f"Entrada media={x.mean().item():+.4f}, std={x.std().item():.4f}\n")
print("Forward pass:")
output = model_sin_bn(x)

print(f"\n  -> Las medias y std cambian sin control entre capas.")
print(f"     Cada capa recibe datos en escalas impredecibles.\n")

# =============================================
# Paso 2: Red CON BatchNorm
# =============================================
print("=" * 55)
print("Paso 2: Red CON BatchNorm")
print("=" * 55)


class NetworkWith_BN(nn.Module):
    def __init__(self):
        super().__init__()
        # Misma arquitectura, pero con BatchNorm despues de cada capa lineal
        self.fc1 = nn.Linear(10, 64)
        self.bn1 = nn.BatchNorm1d(64)    # normaliza las 64 salidas de fc1
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)    # normaliza las 32 salidas de fc2
        self.fc3 = nn.Linear(32, 2)      # NO ponemos BN en la ultima capa
        self.relu = nn.ReLU()

    def forward(self, x):
        # Capa 1: Linear -> BatchNorm -> ReLU
        x = self.fc1(x)
        print(f"    Despues de fc1:  media={x.mean().item():+8.4f}, std={x.std().item():.4f}")
        x = self.bn1(x)
        print(f"    Despues de bn1:  media={x.mean().item():+8.4f}, std={x.std().item():.4f}  <- normalizado!")
        x = self.relu(x)

        # Capa 2: Linear -> BatchNorm -> ReLU
        x = self.fc2(x)
        print(f"    Despues de fc2:  media={x.mean().item():+8.4f}, std={x.std().item():.4f}")
        x = self.bn2(x)
        print(f"    Despues de bn2:  media={x.mean().item():+8.4f}, std={x.std().item():.4f}  <- normalizado!")
        x = self.relu(x)

        # Salida (sin BN)
        x = self.fc3(x)
        print(f"    Despues de fc3:  media={x.mean().item():+8.4f}, std={x.std().item():.4f}")
        return x


model_con_bn = NetworkWith_BN()

print(f"\nEntrada: {x.shape} (mismos datos)")
print(f"Entrada media={x.mean().item():+.4f}, std={x.std().item():.4f}\n")
print("Forward pass:")
output = model_con_bn(x)

print(f"\n  -> Despues de cada BatchNorm, media vuelve a ~0 y std a ~1.")
print(f"     La siguiente capa siempre recibe datos en una escala estable.\n")

# =============================================
# Paso 3: Que parametros agrego BatchNorm?
# =============================================
print("=" * 55)
print("Paso 3: Parametros de la red")
print("=" * 55)

total_sin_bn = sum(p.numel() for p in model_sin_bn.parameters())
total_con_bn = sum(p.numel() for p in model_con_bn.parameters())

print(f"\n  Sin BatchNorm: {total_sin_bn} parametros")
print(f"  Con BatchNorm: {total_con_bn} parametros")
print(f"  Diferencia:    {total_con_bn - total_sin_bn} parametros extra")
print(f"")

# Desglose de BatchNorm
print(f"  Desglose de bn1 (64 features):")
print(f"    gamma (weight): {model_con_bn.bn1.weight.shape} = {model_con_bn.bn1.weight.numel()} parametros")
print(f"    beta  (bias):   {model_con_bn.bn1.bias.shape} = {model_con_bn.bn1.bias.numel()} parametros")
print(f"    running_mean:   {model_con_bn.bn1.running_mean.shape} (NO es parametro, no se entrena)")
print(f"    running_var:    {model_con_bn.bn1.running_var.shape} (NO es parametro, no se entrena)")
print(f"")
print(f"  -> BatchNorm agrega MUY POCOS parametros (2 por feature: gamma y beta)")
print(f"     pero el beneficio en estabilidad y velocidad de entrenamiento es enorme.")

# =============================================
# Paso 4: Orden de las capas
# =============================================
print(f"\n{'='*55}")
print("Paso 4: Orden de las capas (receta)")
print("=" * 55)
print(f"""
  Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> ReLU -> Linear (salida)
  ──────    ─────────    ────    ──────    ─────────    ────    ──────
  capa      normalizar   activar capa      normalizar   activar capa final

  Reglas:
  1. BatchNorm va DESPUES de la capa lineal/convolucional
  2. BatchNorm va ANTES de la activacion (ReLU)
  3. NUNCA poner BatchNorm en la ultima capa (queremos la prediccion real)
  4. No olvidar model.eval() en inferencia!
""")
