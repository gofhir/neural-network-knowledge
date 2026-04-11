"""
Ejemplo 1 — Normalizacion manual vs BatchNorm (TensorFlow)

Objetivo: entender que normalizar es simplemente aplicar la Z-score
          (restar la media, dividir por la desviacion estandar).
          BatchNormalization de Keras hace exactamente eso.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/01_normalizacion_manual_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

# --- Datos de ejemplo ---
values = tf.constant([2.0, 8.0, 4.0, 6.0])
print(f"Valores originales: {values.numpy().tolist()}")

# =============================================
# Paso 1: Normalizar A MANO (Z-score)
# =============================================
mean = tf.reduce_mean(values)
std = tf.math.reduce_std(values)  # poblacional por defecto en TF

print(f"\nPaso 1: Calcular media")
print(f"  media = (2 + 8 + 4 + 6) / 4 = {mean.numpy():.1f}")

print(f"\nPaso 2: Calcular desviacion estandar")
print(f"  std = {std.numpy():.4f}")

norm_manual = (values - mean) / std

print(f"\nPaso 3: Normalizar cada valor = (valor - {mean.numpy():.1f}) / {std.numpy():.4f}")
for i in range(len(values)):
    print(f"  ({values[i].numpy():.1f} - {mean.numpy():.1f}) / {std.numpy():.4f} = {norm_manual[i].numpy():.4f}")

print(f"\nResultado manual: {[round(x, 4) for x in norm_manual.numpy().tolist()]}")

# =============================================
# Paso 2: Normalizar con BatchNormalization de Keras
# =============================================
# Reshape a [4, 1] (4 muestras, 1 feature)
x = tf.reshape(values, (4, 1))
print(f"\nReformar para BatchNorm: {values.shape} -> {x.shape}")

# scale=False, center=False = sin gamma ni beta (normalizacion pura)
# momentum=0.0 = usar estadisticas exactas del batch
bn = keras.layers.BatchNormalization(
    scale=False,      # sin gamma   (en PyTorch: affine=False)
    center=False,     # sin beta    (en PyTorch: affine=False)
    momentum=0.0,     # estadisticas exactas (en PyTorch: momentum=None)
    epsilon=1e-5,
)

norm_bn = tf.squeeze(bn(x, training=True))  # training=True para usar stats del batch
print(f"Resultado BatchNorm:  {[round(x, 4) for x in norm_bn.numpy().tolist()]}")

# =============================================
# Verificar que son iguales
# =============================================
are_equal = tf.reduce_all(tf.abs(norm_manual - norm_bn) < 1e-4).numpy()
print(f"\nSon iguales? {are_equal}")

# =============================================
# Diferencia de API
# =============================================
print(f"\nDiferencia clave de API:")
print(f"  PyTorch:     nn.BatchNorm1d(num_features=1, affine=False)")
print(f"  TensorFlow:  layers.BatchNormalization(scale=False, center=False)")
print(f"  Misma matematica, distinto nombre de parametros.")
