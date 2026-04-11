"""
Ejemplo 4 — Gamma y Beta en JAX/Flax

Objetivo: entender como Flax maneja gamma y beta.
          En Flax se llaman 'scale' (gamma) y 'bias' (beta),
          y se controlan con use_scale y use_bias.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/04_gamma_beta_jax.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

# --- Datos: 4 muestras, 1 feature ---
x = jnp.array([[2.0], [8.0], [4.0], [6.0]])
print(f"Valores originales: {x.squeeze().tolist()}")

# =============================================
# Paso 1: BatchNorm CON gamma y beta (default)
# =============================================
print(f"\n--- Paso 1: use_scale=True, use_bias=True (default) ---")

# use_scale=True (default) = incluye gamma
# use_bias=True (default) = incluye beta
bn = nn.BatchNorm(use_running_average=False)

key = jax.random.PRNGKey(0)
variables = bn.init(key, x)

# Ver los parametros
print(f"  Parametros iniciales:")
print(f"    scale (gamma): {variables['params']['scale'].tolist()}")
print(f"    bias  (beta):  {variables['params']['bias'].tolist()}")

result, _ = bn.apply(variables, x, mutable=['batch_stats'])
print(f"  Salida:    {[round(float(v), 4) for v in result.squeeze()]}")
print(f"  Media:     {jnp.mean(result):.4f}")
print(f"  Varianza:  {jnp.var(result):.4f}")

# =============================================
# Paso 2: Cambiar gamma=2, beta=5
# =============================================
print(f"\n--- Paso 2: scale=2, bias=5 (cambiar manualmente) ---")

# En JAX, los parametros son un diccionario inmutable
# Se modifica creando uno nuevo
from flax.core import freeze, unfreeze
new_params = unfreeze(variables)
new_params['params']['scale'] = jnp.array([2.0])
new_params['params']['bias'] = jnp.array([5.0])
new_variables = freeze(new_params)

print(f"  scale (gamma): {new_variables['params']['scale'].tolist()}")
print(f"  bias  (beta):  {new_variables['params']['bias'].tolist()}")

result, _ = bn.apply(new_variables, x, mutable=['batch_stats'])
print(f"  Salida:    {[round(float(v), 4) for v in result.squeeze()]}")
print(f"  Media:     {jnp.mean(result):.4f}  <- desplazada hacia beta=5")
print(f"  Varianza:  {jnp.var(result):.4f}  <- estirada por gamma²=4")

# =============================================
# Paso 3: Sin gamma ni beta
# =============================================
print(f"\n--- Paso 3: use_scale=False, use_bias=False ---")

bn_pure = nn.BatchNorm(use_running_average=False, use_scale=False, use_bias=False)
variables_pure = bn_pure.init(key, x)

# Verificar que NO tiene scale ni bias
has_params = 'params' in variables_pure and len(variables_pure['params']) > 0
print(f"  Tiene parametros trainable? {has_params}")

result, _ = bn_pure.apply(variables_pure, x, mutable=['batch_stats'])
print(f"  Salida:    {[round(float(v), 4) for v in result.squeeze()]}")
print(f"  Media:     {jnp.mean(result):.4f}  <- siempre 0")
print(f"  Varianza:  {jnp.var(result):.4f}  <- siempre 1")

# =============================================
# Comparacion de nombres
# =============================================
print(f"\n--- Comparacion de nombres ---")
print(f"  Concepto       PyTorch          TensorFlow        JAX/Flax")
print(f"  ─────────      ───────          ──────────        ────────")
print(f"  gamma          .weight          .gamma            params['scale']")
print(f"  beta           .bias            .beta             params['bias']")
print(f"  desactivar     affine=False     scale/center=     use_scale/use_bias=")
print(f"                                  False             False")
print(f"  modificar      torch.no_grad()  .assign()         crear nuevo dict")
print(f"")
print(f"  JAX es inmutable: no se modifican parametros, se crean nuevos.")
print(f"  Esto es el estilo FUNCIONAL puro de JAX.")
