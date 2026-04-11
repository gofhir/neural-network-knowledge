"""
Ejemplo 4 — Gamma y Beta en PyTorch

Objetivo: entender que despues de normalizar, la red puede RE-ESCALAR
          los valores con dos parametros aprendibles: gamma y beta.
          En PyTorch se llaman .weight (gamma) y .bias (beta).

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/04_gamma_beta_pytorch.py
"""
import torch
import torch.nn as nn

# --- Datos: 4 muestras, 1 feature ---
x = torch.tensor([[2.0], [8.0], [4.0], [6.0]])
print(f"Valores originales: {x.squeeze().tolist()}")

# =============================================
# Paso 1: BatchNorm con gamma=1, beta=0 (default)
# =============================================
# affine=True (default) = incluye gamma y beta
bn = nn.BatchNorm1d(num_features=1, momentum=None)

print(f"\n--- Paso 1: gamma=1, beta=0 (valores iniciales) ---")
print(f"  bn.weight (gamma): {bn.weight.data.tolist()}")
print(f"  bn.bias   (beta):  {bn.bias.data.tolist()}")

result = bn(x)
print(f"  Salida:    {[round(v, 4) for v in result.squeeze().tolist()]}")
print(f"  Media:     {result.mean().item():.4f}")
print(f"  Varianza:  {result.var(correction=0).item():.4f}")
print(f"  -> Con gamma=1 y beta=0, la salida es la normalizacion pura")

# =============================================
# Paso 2: Cambiar gamma=2, beta=5
# =============================================
print(f"\n--- Paso 2: gamma=2, beta=5 (cambiamos manualmente) ---")

# Usamos torch.no_grad() porque estamos modificando parametros
# directamente, no entrenando
with torch.no_grad():
    bn.weight.fill_(2.0)   # gamma = 2
    bn.bias.fill_(5.0)     # beta = 5

print(f"  bn.weight (gamma): {bn.weight.data.tolist()}")
print(f"  bn.bias   (beta):  {bn.bias.data.tolist()}")

result = bn(x)
print(f"  Salida:    {[round(v, 4) for v in result.squeeze().tolist()]}")
print(f"  Media:     {result.mean().item():.4f}  <- se desplazo hacia beta=5")
print(f"  Varianza:  {result.var(correction=0).item():.4f}  <- se estiro por gamma²=4")
print(f"  -> Formula: y = gamma * x_norm + beta = 2 * x_norm + 5")

# =============================================
# Paso 3: Sin gamma ni beta (affine=False)
# =============================================
print(f"\n--- Paso 3: affine=False (sin gamma ni beta) ---")

bn_pure = nn.BatchNorm1d(num_features=1, momentum=None, affine=False)

# Verificar que NO tiene .weight ni .bias
has_weight = hasattr(bn_pure, 'weight') and bn_pure.weight is not None
has_bias = hasattr(bn_pure, 'bias') and bn_pure.bias is not None
print(f"  Tiene weight (gamma)? {has_weight}")
print(f"  Tiene bias (beta)?    {has_bias}")

result = bn_pure(x)
print(f"  Salida:    {[round(v, 4) for v in result.squeeze().tolist()]}")
print(f"  Media:     {result.mean().item():.4f}  <- siempre 0")
print(f"  Varianza:  {result.var(correction=0).item():.4f}  <- siempre 1")
print(f"  -> Sin gamma/beta, la normalizacion es fija (no puede ajustarse)")

# =============================================
# Resumen
# =============================================
print(f"\n--- Resumen ---")
print(f"  gamma (weight) = cuanto ESTIRAR o comprimir (escala)")
print(f"  beta  (bias)   = cuanto DESPLAZAR (centro)")
print(f"  Se inicializan en gamma=1, beta=0 (no cambian nada al inicio)")
print(f"  El optimizador (SGD, Adam) los ajusta durante el entrenamiento")
print(f"  En la practica casi siempre se dejan activados (affine=True)")
