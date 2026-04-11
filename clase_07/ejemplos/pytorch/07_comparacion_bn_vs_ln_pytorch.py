"""
Ejemplo 7 — Comparacion BatchNorm vs LayerNorm (PyTorch)

Objetivo: ver lado a lado como BatchNorm y LayerNorm producen
          resultados diferentes a partir de los MISMOS datos.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/07_comparacion_bn_vs_ln_pytorch.py
"""
import torch
import torch.nn as nn

# --- Mismos datos para ambos ---
x = torch.tensor([
    [2.0, 10.0, 0.5],
    [8.0, 20.0, 1.5],
    [4.0, 30.0, 0.8],
    [6.0, 40.0, 1.2],
])

print("Datos originales (4 muestras x 3 features):")
print(f"         Feature1  Feature2  Feature3")
for i in range(4):
    print(f"  Muestra {i+1}:  {x[i, 0]:5.1f}     {x[i, 1]:5.1f}     {x[i, 2]:4.1f}")

# =============================================
# BatchNorm
# =============================================
bn = nn.BatchNorm1d(3, momentum=None, affine=False)
result_bn = bn(x)

print(f"\n{'='*50}")
print(f"BATCHNORM (normaliza cada COLUMNA)")
print(f"{'='*50}")
print(f"Verificacion por columna (feature):")
for i in range(3):
    col = result_bn[:, i]
    print(f"  Feature {i+1}: media={col.mean().item():+.4f}, var={col.var(correction=0).item():.4f}")

# =============================================
# LayerNorm
# =============================================
ln = nn.LayerNorm(3, elementwise_affine=False)
result_ln = ln(x)

print(f"\n{'='*50}")
print(f"LAYERNORM (normaliza cada FILA)")
print(f"{'='*50}")
print(f"Verificacion por fila (muestra):")
for i in range(4):
    row = result_ln[i]
    print(f"  Muestra {i+1}: media={row.mean().item():+.4f}, var={row.var(correction=0).item():.4f}")

# =============================================
# Comparar las salidas
# =============================================
print(f"\n{'='*50}")
print(f"SALIDAS COMPLETAS (para comparar)")
print(f"{'='*50}")

print(f"\nBatchNorm:")
for i in range(4):
    print(f"  Muestra {i+1}: {[round(v, 4) for v in result_bn[i].tolist()]}")

print(f"\nLayerNorm:")
for i in range(4):
    print(f"  Muestra {i+1}: {[round(v, 4) for v in result_ln[i].tolist()]}")

print(f"\nSon iguales? {torch.allclose(result_bn, result_ln)}")

# =============================================
# Cuando usar cual
# =============================================
print(f"\n{'='*50}")
print(f"CUANDO USAR CUAL")
print(f"{'='*50}")
print(f"  BatchNorm:")
print(f"    - Redes convolucionales (CNNs) para imagenes")
print(f"    - Cuando tienes batches grandes (>= 32)")
print(f"    - Ejemplo: ResNet, VGG, etc.")
print(f"")
print(f"  LayerNorm:")
print(f"    - Transformers (GPT, BERT, etc.)")
print(f"    - Cuando el batch puede ser pequeno")
print(f"    - Cuando trabajas con secuencias de largo variable (NLP)")
print(f"    - Ejemplo: cada capa de atencion en un Transformer usa LayerNorm")
