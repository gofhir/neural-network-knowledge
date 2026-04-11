"""
Ejemplo 1 — Normalizacion manual vs BatchNorm (PyTorch)

Objetivo: entender que normalizar es simplemente aplicar la Z-score
          (restar la media, dividir por la desviacion estandar).
          BatchNorm hace exactamente eso, pero automaticamente.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/01_normalizacion_manual_pytorch.py
"""
import torch
import torch.nn as nn

# --- Datos de ejemplo ---
# 4 valores, como si fueran 4 muestras de un solo feature
values = torch.tensor([2.0, 8.0, 4.0, 6.0])
print(f"Valores originales: {values.tolist()}")

# =============================================
# Paso 1: Normalizar A MANO (Z-score)
# =============================================
# Formula: x_norm = (x - media) / desviacion_estandar

mean = values.mean()
std = values.std(correction=0)  # correction=0 = poblacional (dividir por N, no N-1)

print(f"\nPaso 1: Calcular media")
print(f"  media = (2 + 8 + 4 + 6) / 4 = {mean.item():.1f}")

print(f"\nPaso 2: Calcular desviacion estandar")
print(f"  std = {std.item():.4f}")

norm_manual = (values - mean) / std

print(f"\nPaso 3: Normalizar cada valor = (valor - {mean.item():.1f}) / {std.item():.4f}")
for i, (original, normalized) in enumerate(zip(values, norm_manual)):
    print(f"  ({original.item():.1f} - {mean.item():.1f}) / {std.item():.4f} = {normalized.item():.4f}")

print(f"\nResultado manual: {[round(x, 4) for x in norm_manual.tolist()]}")

# =============================================
# Paso 2: Normalizar con BatchNorm de PyTorch
# =============================================
# BatchNorm espera forma [batch, features]
# Tenemos 4 muestras con 1 feature cada una
x = values.unsqueeze(1)  # [4] -> [4, 1]
print(f"\nReformar para BatchNorm: {values.shape} -> {x.shape}")

# affine=False = sin gamma ni beta (normalizacion pura)
# momentum=None = usar estadisticas exactas del batch
bn = nn.BatchNorm1d(num_features=1, momentum=None, affine=False)

norm_bn = bn(x).squeeze(1)  # quitar la dimension extra al final
print(f"Resultado BatchNorm:  {[round(x, 4) for x in norm_bn.tolist()]}")

# =============================================
# Verificar que son iguales
# =============================================
are_equal = torch.allclose(norm_manual, norm_bn)
print(f"\nSon iguales? {are_equal}")
print(f"\nConclusion: BatchNorm hace EXACTAMENTE la Z-score.")
print(f"La diferencia es que lo hace automaticamente dentro de la red,")
print(f"en cada capa, en cada iteracion de entrenamiento.")
