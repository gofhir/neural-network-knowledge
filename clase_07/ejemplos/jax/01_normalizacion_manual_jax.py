"""
Ejemplo 1 — Normalizacion manual vs BatchNorm (JAX/Flax)

Objetivo: entender que normalizar es simplemente aplicar la Z-score.
          En JAX se usa Flax (flax.linen) para las capas de redes neuronales.

JAX es FUNCIONAL: no hay objetos con estado como en PyTorch/TF.
Las funciones reciben datos y devuelven resultados, sin modificar nada.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/01_normalizacion_manual_jax.py
"""
import jax
import jax.numpy as jnp

# --- Datos de ejemplo ---
values = jnp.array([2.0, 8.0, 4.0, 6.0])
print(f"Valores originales: {values.tolist()}")

# =============================================
# Paso 1: Normalizar A MANO (Z-score)
# =============================================
mean = jnp.mean(values)
std = jnp.std(values)  # poblacional por defecto en JAX

print(f"\nPaso 1: Calcular media")
print(f"  media = (2 + 8 + 4 + 6) / 4 = {mean:.1f}")

print(f"\nPaso 2: Calcular desviacion estandar")
print(f"  std = {std:.4f}")

norm_manual = (values - mean) / std

print(f"\nPaso 3: Normalizar cada valor = (valor - {mean:.1f}) / {std:.4f}")
for i in range(len(values)):
    print(f"  ({values[i]:.1f} - {mean:.1f}) / {std:.4f} = {norm_manual[i]:.4f}")

print(f"\nResultado manual: {[round(float(x), 4) for x in norm_manual]}")

# =============================================
# Paso 2: Normalizar con Flax BatchNorm
# =============================================
# En JAX/Flax, BatchNorm funciona distinto:
# - No hay "objetos con estado" como en PyTorch/TF
# - Se pasa todo explicitamente (parametros, estado, datos)
import flax.linen as nn

# Definir la capa
bn = nn.BatchNorm(use_running_average=False, use_bias=False, use_scale=False)

# En JAX hay que inicializar los parametros explicitamente
x = values.reshape(4, 1)  # [4, 1]
key = jax.random.PRNGKey(0)
variables = bn.init(key, x)

# Aplicar BatchNorm
norm_bn, updates = bn.apply(variables, x, mutable=['batch_stats'])
norm_bn = norm_bn.squeeze()

print(f"\nResultado BatchNorm (Flax): {[round(float(x), 4) for x in norm_bn]}")

# =============================================
# Verificar
# =============================================
are_equal = jnp.allclose(norm_manual, norm_bn, atol=1e-4)
print(f"\nSon iguales? {are_equal}")

# =============================================
# Comparacion de API
# =============================================
print(f"\nComparacion de API:")
print(f"  PyTorch:      nn.BatchNorm1d(1)           -> bn(x)")
print(f"  TensorFlow:   BatchNormalization()         -> bn(x, training=True)")
print(f"  JAX/Flax:     nn.BatchNorm()               -> bn.apply(vars, x)")
print(f"")
print(f"  JAX es mas explicito: hay que manejar parametros y estado manualmente.")
print(f"  Es mas trabajo, pero da control total sobre todo.")
