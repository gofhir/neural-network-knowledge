"""
Ejemplo 7 — Comparacion BatchNorm vs LayerNorm (JAX/Flax)

Objetivo: ver lado a lado como BatchNorm y LayerNorm producen
          resultados diferentes a partir de los MISMOS datos.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/07_comparacion_bn_vs_ln_jax.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

# --- Mismos datos para ambos ---
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

key = jax.random.PRNGKey(0)

# =============================================
# BatchNorm
# =============================================
bn = nn.BatchNorm(use_running_average=False, use_scale=False, use_bias=False)
bn_vars = bn.init(key, x)
result_bn, _ = bn.apply(bn_vars, x, mutable=['batch_stats'])

print(f"\n{'='*50}")
print(f"BATCHNORM (normaliza cada COLUMNA)")
print(f"{'='*50}")
print(f"Verificacion por columna (feature):")
for i in range(3):
    col = result_bn[:, i]
    print(f"  Feature {i+1}: media={jnp.mean(col):+.4f}, var={jnp.var(col):.4f}")

# =============================================
# LayerNorm
# =============================================
ln = nn.LayerNorm(use_scale=False, use_bias=False)
ln_vars = ln.init(key, x)
result_ln = ln.apply(ln_vars, x)

print(f"\n{'='*50}")
print(f"LAYERNORM (normaliza cada FILA)")
print(f"{'='*50}")
print(f"Verificacion por fila (muestra):")
for i in range(4):
    row = result_ln[i]
    print(f"  Muestra {i+1}: media={jnp.mean(row):+.4f}, var={jnp.var(row):.4f}")

# =============================================
# Comparar salidas
# =============================================
print(f"\n{'='*50}")
print(f"SALIDAS COMPLETAS")
print(f"{'='*50}")

print(f"\nBatchNorm:")
for i in range(4):
    print(f"  Muestra {i+1}: {[round(float(v), 4) for v in result_bn[i]]}")

print(f"\nLayerNorm:")
for i in range(4):
    print(f"  Muestra {i+1}: {[round(float(v), 4) for v in result_ln[i]]}")

are_equal = jnp.allclose(result_bn, result_ln, atol=1e-4)
print(f"\nSon iguales? {are_equal}")

# =============================================
# Resumen de los 3 frameworks
# =============================================
print(f"\n{'='*50}")
print(f"RESUMEN: PyTorch vs TensorFlow vs JAX")
print(f"{'='*50}")
print(f"""
  Concepto        PyTorch            TensorFlow          JAX/Flax
  ────────        ───────            ──────────          ────────
  BatchNorm       BatchNorm1d(n)     BatchNormalization  BatchNorm()
  LayerNorm       LayerNorm(n)       LayerNormalization  LayerNorm()
  gamma           .weight            .gamma              params['scale']
  beta            .bias              .beta               params['bias']
  sin gamma/beta  affine=False       scale/center=False  use_scale/bias=False
  train/eval      model.train/eval   training=True/False use_running_average
  estado          interno            interno             externo (tu lo manejas)
  estilo          orientado objetos  orientado objetos   funcional puro
""")
