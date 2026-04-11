"""
Ejemplo 4 — Gamma y Beta en TensorFlow

Objetivo: entender que despues de normalizar, la red puede RE-ESCALAR
          los valores con dos parametros aprendibles: gamma y beta.
          En TensorFlow se llaman .gamma y .beta (mas intuitivo que PyTorch).

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/04_gamma_beta_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

# --- Datos: 4 muestras, 1 feature ---
x = tf.constant([[2.0], [8.0], [4.0], [6.0]])
print(f"Valores originales: {tf.squeeze(x).numpy().tolist()}")

# =============================================
# Paso 1: BatchNorm con gamma=1, beta=0 (default)
# =============================================
# scale=True (default) = incluye gamma
# center=True (default) = incluye beta
bn = keras.layers.BatchNormalization(momentum=0.0, epsilon=1e-5)
result = bn(x, training=True)

print(f"\n--- Paso 1: gamma=1, beta=0 (valores iniciales) ---")
print(f"  bn.gamma: {bn.gamma.numpy().tolist()}")
print(f"  bn.beta:  {bn.beta.numpy().tolist()}")
print(f"  Salida:    {[round(v, 4) for v in tf.squeeze(result).numpy().tolist()]}")
print(f"  Media:     {tf.reduce_mean(result).numpy():.4f}")
print(f"  Varianza:  {tf.math.reduce_variance(result).numpy():.4f}")
print(f"  -> Con gamma=1 y beta=0, la salida es la normalizacion pura")

# =============================================
# Paso 2: Cambiar gamma=2, beta=5
# =============================================
print(f"\n--- Paso 2: gamma=2, beta=5 (cambiamos manualmente) ---")

# En TF se usa .assign() para cambiar valores de variables
bn.gamma.assign([2.0])
bn.beta.assign([5.0])

print(f"  bn.gamma: {bn.gamma.numpy().tolist()}")
print(f"  bn.beta:  {bn.beta.numpy().tolist()}")

result = bn(x, training=True)
print(f"  Salida:    {[round(v, 4) for v in tf.squeeze(result).numpy().tolist()]}")
print(f"  Media:     {tf.reduce_mean(result).numpy():.4f}  <- se desplazo hacia beta=5")
print(f"  Varianza:  {tf.math.reduce_variance(result).numpy():.4f}  <- se estiro por gamma²=4")

# =============================================
# Paso 3: Sin gamma ni beta
# =============================================
print(f"\n--- Paso 3: scale=False, center=False (sin gamma ni beta) ---")

bn_pure = keras.layers.BatchNormalization(
    momentum=0.0, scale=False, center=False, epsilon=1e-5
)
result = bn_pure(x, training=True)

print(f"  Tiene gamma? {hasattr(bn_pure, 'gamma') and bn_pure.gamma is not None}")
print(f"  Tiene beta?  {hasattr(bn_pure, 'beta') and bn_pure.beta is not None}")
print(f"  Salida:    {[round(v, 4) for v in tf.squeeze(result).numpy().tolist()]}")
print(f"  Media:     {tf.reduce_mean(result).numpy():.4f}  <- siempre 0")
print(f"  Varianza:  {tf.math.reduce_variance(result).numpy():.4f}  <- siempre 1")

# =============================================
# Comparacion de nombres
# =============================================
print(f"\n--- Comparacion de nombres ---")
print(f"  Concepto       PyTorch          TensorFlow")
print(f"  ─────────      ───────          ──────────")
print(f"  gamma          .weight          .gamma")
print(f"  beta           .bias            .beta")
print(f"  desactivar     affine=False     scale=False, center=False")
print(f"  modificar      torch.no_grad()  .assign()")
print(f"")
print(f"  TF es mas explicito: 'scale' controla gamma, 'center' controla beta.")
print(f"  PyTorch los agrupa en un solo parametro 'affine'.")
