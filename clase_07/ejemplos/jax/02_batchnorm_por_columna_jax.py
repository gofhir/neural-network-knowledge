"""
Ejemplo 2 — BatchNorm normaliza por COLUMNA (JAX/Flax)

Objetivo: ver que BatchNorm calcula media y varianza de cada FEATURE
          (columna) a traves de todas las muestras del batch.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/02_batchnorm_por_columna_jax.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

# --- Datos: 4 muestras, 3 features ---
x = jnp.array([
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
print(f"\nPaso 1: Estadisticas por COLUMNA (axis=0 = a traves del batch)")
for i in range(3):
    col = x[:, i]
    mean = jnp.mean(col)
    std = jnp.std(col)
    print(f"  Feature {i+1}: valores={col.tolist()}, media={mean:.2f}, std={std:.2f}")

# =============================================
# Paso 2: Aplicar BatchNorm con Flax
# =============================================
# use_bias=False, use_scale=False = sin gamma ni beta
bn = nn.BatchNorm(use_running_average=False, use_bias=False, use_scale=False)

# Inicializar parametros
key = jax.random.PRNGKey(0)
variables = bn.init(key, x)

# Aplicar
result, _ = bn.apply(variables, x, mutable=['batch_stats'])

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
    print(f"  Feature {i+1}: media={jnp.mean(col):.4f}, varianza={jnp.var(col):.4f}")

# =============================================
# Diferencia de API
# =============================================
print(f"\nDiferencia de API:")
print(f"  PyTorch:     bn(x)                        # estado interno")
print(f"  TensorFlow:  bn(x, training=True)          # training como argumento")
print(f"  JAX/Flax:    bn.apply(vars, x, mutable=..) # todo explicito")
print(f"")
print(f"  En JAX no hay estado interno. Los parametros (variables)")
print(f"  se pasan explicitamente y se devuelven actualizados.")
