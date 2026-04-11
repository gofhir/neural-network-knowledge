"""
Ejemplo 7 — Comparacion BatchNorm vs LayerNorm (TensorFlow)

Objetivo: ver lado a lado como BatchNormalization y LayerNormalization
          producen resultados diferentes a partir de los MISMOS datos.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/07_comparacion_bn_vs_ln_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

# --- Mismos datos para ambos ---
x = tf.constant([
    [2.0, 10.0, 0.5],
    [8.0, 20.0, 1.5],
    [4.0, 30.0, 0.8],
    [6.0, 40.0, 1.2],
])

print("Datos originales (4 muestras x 3 features):")
print(f"         Feature1  Feature2  Feature3")
for i in range(4):
    print(f"  Muestra {i+1}:  {x[i, 0].numpy():5.1f}     {x[i, 1].numpy():5.1f}     {x[i, 2].numpy():4.1f}")

# =============================================
# BatchNorm
# =============================================
bn = keras.layers.BatchNormalization(
    momentum=0.0, scale=False, center=False, epsilon=1e-5
)
result_bn = bn(x, training=True)

print(f"\n{'='*50}")
print(f"BATCHNORM (normaliza cada COLUMNA)")
print(f"{'='*50}")
print(f"Verificacion por columna (feature):")
for i in range(3):
    col = result_bn[:, i]
    print(f"  Feature {i+1}: media={tf.reduce_mean(col).numpy():+.4f}"
          f", var={tf.math.reduce_variance(col).numpy():.4f}")

# =============================================
# LayerNorm
# =============================================
ln = keras.layers.LayerNormalization(
    scale=False, center=False, epsilon=1e-5
)
result_ln = ln(x)

print(f"\n{'='*50}")
print(f"LAYERNORM (normaliza cada FILA)")
print(f"{'='*50}")
print(f"Verificacion por fila (muestra):")
for i in range(4):
    row = result_ln[i]
    print(f"  Muestra {i+1}: media={tf.reduce_mean(row).numpy():+.4f}"
          f", var={tf.math.reduce_variance(row).numpy():.4f}")

# =============================================
# Comparar las salidas
# =============================================
print(f"\n{'='*50}")
print(f"SALIDAS COMPLETAS (para comparar)")
print(f"{'='*50}")

print(f"\nBatchNorm:")
for i in range(4):
    print(f"  Muestra {i+1}: {[round(v, 4) for v in result_bn[i].numpy().tolist()]}")

print(f"\nLayerNorm:")
for i in range(4):
    print(f"  Muestra {i+1}: {[round(v, 4) for v in result_ln[i].numpy().tolist()]}")

are_equal = tf.reduce_all(tf.abs(result_bn - result_ln) < 1e-4).numpy()
print(f"\nSon iguales? {are_equal}")

# =============================================
# Resumen de API: PyTorch vs TensorFlow
# =============================================
print(f"\n{'='*50}")
print(f"RESUMEN: PyTorch vs TensorFlow")
print(f"{'='*50}")
print(f"")
print(f"  Concepto        PyTorch                    TensorFlow")
print(f"  ────────        ───────                    ──────────")
print(f"  BatchNorm       nn.BatchNorm1d(n)          layers.BatchNormalization()")
print(f"  LayerNorm       nn.LayerNorm(n)            layers.LayerNormalization()")
print(f"  gamma           .weight                    .gamma")
print(f"  beta            .bias                      .beta")
print(f"  sin gamma/beta  affine=False               scale=False, center=False")
print(f"  train/eval      model.train()/eval()       training=True/False")
print(f"  running stats   running_mean/var           moving_mean/variance")
print(f"")
print(f"  La matematica es IDENTICA. Solo cambian los nombres.")
