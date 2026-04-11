"""
Ejemplo 5 — BatchNorm: entrenamiento vs inferencia (JAX/Flax)

Objetivo: en JAX, train/eval se controla con el parametro
          use_running_average=True/False. Los running stats se manejan
          explicitamente como parte del estado (batch_stats).

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/05_batchnorm_train_vs_eval_jax.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze

# =============================================
# Paso 1: Entrenamiento — pasar varios batches
# =============================================
print("--- Paso 1: ENTRENAMIENTO (use_running_average=False) ---")
print("Pasamos 3 batches. En cada uno hay que:")
print("  1. Llamar bn.apply() con mutable=['batch_stats']")
print("  2. Recibir el estado actualizado")
print("  3. Pasarlo al siguiente batch\n")

bn_train = nn.BatchNorm(use_running_average=False)  # para entrenamiento
bn_eval = nn.BatchNorm(use_running_average=True)    # para inferencia
key = jax.random.PRNGKey(0)

# Inicializar con un batch de ejemplo
init_x = jnp.array([[10.0], [12.0], [8.0], [14.0]])
variables = bn_train.init(key, init_x)

# En JAX, los batch_stats son EXTERNOS (no viven dentro del objeto)
batch_stats = variables.get('batch_stats', {})

batches = [
    jnp.array([[10.0], [12.0], [8.0], [14.0]]),
    jnp.array([[20.0], [22.0], [18.0], [24.0]]),
    jnp.array([[15.0], [17.0], [13.0], [19.0]]),
]

for i, batch in enumerate(batches):
    # Aplicar con use_running_average=False (entrenamiento)
    # mutable=['batch_stats'] permite actualizar las running stats
    current_vars = {'params': variables['params'], 'batch_stats': batch_stats}
    result, updates = bn_train.apply(
        current_vars, batch,
        mutable=['batch_stats']
    )

    # Actualizar batch_stats para el siguiente batch
    batch_stats = updates['batch_stats']

    mean_val = float(batch_stats['mean'][0])
    var_val = float(batch_stats['var'][0])
    print(f"  Batch {i+1}: valores={batch.squeeze().tolist()}")
    print(f"    running_mean: {mean_val:.4f}")
    print(f"    running_var:  {var_val:.4f}")
    print()

# =============================================
# Paso 2: Inferencia — una sola muestra
# =============================================
print("--- Paso 2: INFERENCIA (use_running_average=True) ---\n")

sample = jnp.array([[16.0]])

# En inferencia: use_running_average=True, NO mutable
eval_vars = {'params': variables['params'], 'batch_stats': batch_stats}
result = bn_eval.apply(eval_vars, sample)

mean_val = float(batch_stats['mean'][0])
var_val = float(batch_stats['var'][0])
print(f"  Muestra: 16.0")
print(f"  Running mean: {mean_val:.4f}")
print(f"  Running var:  {var_val:.4f}")
print(f"  Resultado: {float(result[0, 0]):.4f}")

# =============================================
# Comparacion de API
# =============================================
print(f"\n--- Comparacion de API ---")
print(f"  PyTorch:      model.train() / model.eval()")
print(f"                Estado INTERNO al modelo")
print(f"")
print(f"  TensorFlow:   bn(x, training=True/False)")
print(f"                Estado INTERNO a la capa")
print(f"")
print(f"  JAX/Flax:     bn.apply(vars, x, use_running_average=True/False)")
print(f"                Estado EXTERNO (lo manejas tu en batch_stats)")
print(f"                Hay que pasar mutable=['batch_stats'] en train")
print(f"")
print(f"  JAX hace todo EXPLICITO: tu decides cuando y como")
print(f"  se actualizan las running stats. Mas trabajo, pero")
print(f"  nunca hay sorpresas de estado oculto.")
