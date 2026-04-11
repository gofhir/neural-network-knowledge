"""
Ejemplo 2 — BatchNorm normaliza por COLUMNA (PyTorch)

Objetivo: ver que BatchNorm calcula media y varianza de cada FEATURE
          (columna) a traves de todas las muestras del batch.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/02_batchnorm_por_columna_pytorch.py
"""
import torch
import torch.nn as nn

# --- Datos: 4 muestras, 3 features ---
# Piensa en una tabla: filas = muestras, columnas = features
x = torch.tensor([
    [2.0, 10.0, 0.5],   # muestra 1
    [8.0, 20.0, 1.5],   # muestra 2
    [4.0, 30.0, 0.8],   # muestra 3
    [6.0, 40.0, 1.2],   # muestra 4
])

print("Datos originales (4 muestras x 3 features):")
print(f"         Feature1  Feature2  Feature3")
for i in range(4):
    print(f"  Muestra {i+1}:  {x[i, 0]:5.1f}     {x[i, 1]:5.1f}     {x[i, 2]:4.1f}")

# =============================================
# Paso 1: Calcular media y std por COLUMNA (a mano)
# =============================================
print(f"\nPaso 1: Estadisticas por COLUMNA (dim=0 = a traves del batch)")
for i in range(3):
    col = x[:, i]
    mean = col.mean()
    std = col.std(correction=0)
    print(f"  Feature {i+1}: valores={col.tolist()}, media={mean.item():.2f}, std={std.item():.2f}")

# =============================================
# Paso 2: Aplicar BatchNorm
# =============================================
bn = nn.BatchNorm1d(num_features=3, momentum=None, affine=False)
result = bn(x)

print(f"\nPaso 2: Despues de BatchNorm")
print(f"         Feature1  Feature2  Feature3")
for i in range(4):
    print(f"  Muestra {i+1}:  {result[i, 0]:6.4f}   {result[i, 1]:6.4f}   {result[i, 2]:7.4f}")

# =============================================
# Paso 3: Verificar media~0 y varianza~1 por columna
# =============================================
print(f"\nPaso 3: Verificar que cada COLUMNA tiene media~0 y varianza~1")
for i in range(3):
    col = result[:, i]
    print(f"  Feature {i+1}: media={col.mean().item():.4f}, varianza={col.var(correction=0).item():.4f}")

print(f"\nConclusion: BatchNorm normaliza VERTICALMENTE.")
print(f"Cada feature (columna) queda con media=0 y varianza=1,")
print(f"sin importar que tan diferentes eran las escalas originales.")
print(f"(Feature 2 iba de 10 a 40, Feature 3 de 0.5 a 1.5)")
