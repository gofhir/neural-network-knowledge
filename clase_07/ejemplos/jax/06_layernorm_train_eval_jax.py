"""
Ejemplo 6 — LayerNorm: igual en train y eval (JAX/Flax)

Objetivo: ver que LayerNorm NO tiene estado, asi que no hay
          diferencia entre "train" y "eval". Esto lo hace
          ideal para el estilo funcional de JAX.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/06_layernorm_train_eval_jax.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

ln = nn.LayerNorm(use_bias=False, use_scale=False)

sample = jnp.array([[5.0, 10.0, 15.0]])
print(f"Muestra: {sample.squeeze().tolist()}")

# Inicializar
key = jax.random.PRNGKey(0)
variables = ln.init(key, sample)

# =============================================
# Paso 1: Aplicar LayerNorm (no hay modo train/eval)
# =============================================
result = ln.apply(variables, sample)
print(f"\nResultado: {[round(float(v), 4) for v in result.squeeze()]}")

# Aplicar de nuevo (identico)
result2 = ln.apply(variables, sample)
print(f"De nuevo:  {[round(float(v), 4) for v in result2.squeeze()]}")

are_equal = jnp.allclose(result, result2)
print(f"\nSon iguales? {are_equal}")

# =============================================
# Por que?
# =============================================
print(f"\nPor que son iguales?")
print(f"  LayerNorm NO tiene estado (no hay batch_stats).")
print(f"  No hay que pasar mutable=[...] ni use_running_average.")
print(f"  Solo: ln.apply(variables, x) y listo.")
print(f"")
print(f"  Comparacion de complejidad en JAX:")
print(f"")
print(f"  BatchNorm:")
print(f"    variables = bn.init(key, x)")
print(f"    result, updates = bn.apply(vars, x,")
print(f"        use_running_average=False,")
print(f"        mutable=['batch_stats'])")
print(f"    batch_stats = updates['batch_stats']  # actualizar estado")
print(f"")
print(f"  LayerNorm:")
print(f"    variables = ln.init(key, x)")
print(f"    result = ln.apply(variables, x)  # eso es todo!")
print(f"")
print(f"  LayerNorm es mucho mas simple y natural en JAX")
print(f"  porque no hay estado mutable que manejar.")
