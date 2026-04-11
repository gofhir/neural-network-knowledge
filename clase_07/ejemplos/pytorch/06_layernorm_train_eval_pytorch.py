"""
Ejemplo 6 — LayerNorm: igual en train y eval (PyTorch)

Objetivo: ver que LayerNorm NO cambia entre .train() y .eval().
          A diferencia de BatchNorm, no tiene running stats.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/06_layernorm_train_eval_pytorch.py
"""
import torch
import torch.nn as nn

ln = nn.LayerNorm(normalized_shape=3, elementwise_affine=False)

sample = torch.tensor([[5.0, 10.0, 15.0]])
print(f"Muestra: {sample.squeeze().tolist()}")

# =============================================
# Paso 1: LayerNorm en modo train
# =============================================
ln.train()
result_train = ln(sample)
print(f"\n.train() -> {[round(v, 4) for v in result_train.squeeze().tolist()]}")

# =============================================
# Paso 2: LayerNorm en modo eval
# =============================================
ln.eval()
result_eval = ln(sample)
print(f".eval()  -> {[round(v, 4) for v in result_eval.squeeze().tolist()]}")

# =============================================
# Comparar
# =============================================
are_equal = torch.allclose(result_train, result_eval)
print(f"\nSon iguales? {are_equal}")

# =============================================
# Por que?
# =============================================
print(f"\nPor que son iguales?")
print(f"  LayerNorm normaliza cada muestra usando SUS PROPIOS features.")
print(f"  No necesita estadisticas de otros batches.")
print(f"  No tiene running_mean ni running_var.")
print(f"  Calcula la media y varianza EN EL MOMENTO, siempre igual.")
print(f"")
print(f"  BatchNorm: necesita cambiar entre train/eval (tiene running stats)")
print(f"  LayerNorm: siempre se comporta igual (no tiene running stats)")
print(f"")
print(f"  Por eso LayerNorm es mas simple y es ideal para Transformers,")
print(f"  donde no quieres depender del tamano del batch.")
