"""
Ejemplo 3 — LayerNorm normaliza por FILA (PyTorch)

Objetivo: ver que LayerNorm calcula media y varianza de cada MUESTRA
          (fila) a traves de todos sus features. Es lo opuesto a BatchNorm.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/03_layernorm_por_fila_pytorch.py
"""
import torch
import torch.nn as nn

# --- Mismos datos que el ejemplo 02 ---
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
# Paso 1: Calcular media y std por FILA (a mano)
# =============================================
print(f"\nPaso 1: Estadisticas por FILA (dim=1 = a traves de features)")
for i in range(4):
    row = x[i]
    mean = row.mean()
    std = row.std(correction=0)
    print(f"  Muestra {i+1}: valores={row.tolist()}, media={mean.item():.2f}, std={std.item():.2f}")

# =============================================
# Paso 2: Normalizar a mano (muestra 1)
# =============================================
row = x[0]  # [2.0, 10.0, 0.5]
mean = row.mean()  # 4.17
std = row.std(correction=0)  # 4.17
print(f"\nPaso 2: Normalizar muestra 1 a mano")
print(f"  Valores: {row.tolist()}")
print(f"  Media:   {mean.item():.4f}")
print(f"  Std:     {std.item():.4f}")
norm_row = (row - mean) / std
print(f"  Normalizado: {[round(v, 4) for v in norm_row.tolist()]}")

# =============================================
# Paso 3: Aplicar LayerNorm
# =============================================
# normalized_shape=3 porque cada muestra tiene 3 features
ln = nn.LayerNorm(normalized_shape=3, elementwise_affine=False)
result = ln(x)

print(f"\nPaso 3: Despues de LayerNorm")
print(f"         Feature1  Feature2  Feature3")
for i in range(4):
    print(f"  Muestra {i+1}:  {result[i, 0]:6.4f}   {result[i, 1]:6.4f}   {result[i, 2]:7.4f}")

# =============================================
# Paso 4: Verificar media~0 y varianza~1 por fila
# =============================================
print(f"\nPaso 4: Verificar que cada FILA tiene media~0 y varianza~1")
for i in range(4):
    row = result[i]
    print(f"  Muestra {i+1}: media={row.mean().item():.4f}, varianza={row.var(correction=0).item():.4f}")

# =============================================
# Comparar con BatchNorm
# =============================================
print(f"\nComparacion:")
print(f"  BatchNorm: normaliza cada COLUMNA (feature) -> media/var por feature")
print(f"  LayerNorm: normaliza cada FILA (muestra)    -> media/var por muestra")
print(f"")
print(f"  BatchNorm necesita multiples muestras (un batch).")
print(f"  LayerNorm solo necesita UNA muestra (no depende del batch).")
