"""
Ejemplo 3 — LayerNorm normaliza por FILA (JAX/Flax)

Objetivo: ver que LayerNorm calcula media y varianza de cada MUESTRA
          (fila) a traves de todos sus features.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/03_layernorm_por_fila_jax.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

# --- Mismos datos que el ejemplo 02 ---
x = jnp.array([
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
# Paso 1: Calcular media y std por FILA (a mano)
# =============================================
print(f"\nPaso 1: Estadisticas por FILA (axis=1 = a traves de features)")
for i in range(4):
    row = x[i]
    mean = jnp.mean(row)
    std = jnp.std(row)
    print(f"  Muestra {i+1}: valores={row.tolist()}, media={mean:.2f}, std={std:.2f}")

# =============================================
# Paso 2: Normalizar a mano (muestra 1)
# =============================================
row = x[0]
mean = jnp.mean(row)
std = jnp.std(row)
norm_row = (row - mean) / std
print(f"\nPaso 2: Normalizar muestra 1 a mano")
print(f"  Valores: {row.tolist()}")
print(f"  Media:   {mean:.4f}")
print(f"  Std:     {std:.4f}")
print(f"  Normalizado: {[round(float(v), 4) for v in norm_row]}")

# =============================================
# Paso 3: Aplicar LayerNorm con Flax
# =============================================
# En Flax, LayerNorm es mucho mas simple que BatchNorm
# porque NO tiene estado (no hay running stats)
ln = nn.LayerNorm(use_bias=False, use_scale=False)

# Inicializar (LayerNorm no tiene parametros si use_bias=False, use_scale=False)
key = jax.random.PRNGKey(0)
variables = ln.init(key, x)

# Aplicar (no necesita mutable porque no hay batch_stats)
result = ln.apply(variables, x)

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
    print(f"  Muestra {i+1}: media={jnp.mean(row):.4f}, varianza={jnp.var(row):.4f}")

# =============================================
# Nota: LayerNorm es mas simple en JAX
# =============================================
print(f"\nNota: LayerNorm es MAS SIMPLE que BatchNorm en JAX")
print(f"  BatchNorm: necesita mutable=['batch_stats'] para las running stats")
print(f"  LayerNorm: no tiene estado, solo aplica y listo")
print(f"  Por eso es mas natural para el estilo funcional de JAX.")
