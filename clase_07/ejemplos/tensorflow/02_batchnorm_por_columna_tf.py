"""
Ejemplo 2 — BatchNorm normaliza por COLUMNA (TensorFlow)

Objetivo: ver que BatchNormalization calcula media y varianza de cada FEATURE
          (columna) a traves de todas las muestras del batch.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/02_batchnorm_por_columna_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

# --- Datos: 4 muestras, 3 features ---
x = tf.constant([
    [2.0, 10.0, 0.5],   # muestra 1
    [8.0, 20.0, 1.5],   # muestra 2
    [4.0, 30.0, 0.8],   # muestra 3
    [6.0, 40.0, 1.2],   # muestra 4
])

print("Datos originales (4 muestras x 3 features):")
print(f"         Feature1  Feature2  Feature3")
for i in range(4):
    print(f"  Muestra {i+1}:  {x[i, 0].numpy():5.1f}     {x[i, 1].numpy():5.1f}     {x[i, 2].numpy():4.1f}")

# =============================================
# Paso 1: Calcular media y std por COLUMNA (a mano)
# =============================================
print(f"\nPaso 1: Estadisticas por COLUMNA (axis=0 = a traves del batch)")
for i in range(3):
    col = x[:, i]
    mean = tf.reduce_mean(col)
    std = tf.math.reduce_std(col)
    print(f"  Feature {i+1}: valores={col.numpy().tolist()}, media={mean.numpy():.2f}, std={std.numpy():.2f}")

# =============================================
# Paso 2: Aplicar BatchNormalization
# =============================================
# En TF: scale=False y center=False equivalen a affine=False en PyTorch
bn = keras.layers.BatchNormalization(
    momentum=0.0, scale=False, center=False, epsilon=1e-5
)
result = bn(x, training=True)

print(f"\nPaso 2: Despues de BatchNorm")
print(f"         Feature1  Feature2  Feature3")
for i in range(4):
    print(f"  Muestra {i+1}:  {result[i, 0].numpy():6.4f}   {result[i, 1].numpy():6.4f}   {result[i, 2].numpy():7.4f}")

# =============================================
# Paso 3: Verificar media~0 y varianza~1 por columna
# =============================================
print(f"\nPaso 3: Verificar que cada COLUMNA tiene media~0 y varianza~1")
for i in range(3):
    col = result[:, i]
    print(f"  Feature {i+1}: media={tf.reduce_mean(col).numpy():.4f}"
          f", varianza={tf.math.reduce_variance(col).numpy():.4f}")

# =============================================
# Nota sobre la API
# =============================================
print(f"\nDiferencia de API:")
print(f"  PyTorch:     nn.BatchNorm1d(3)           ->  bn(x)")
print(f"  TensorFlow:  layers.BatchNormalization()  ->  bn(x, training=True)")
print(f"  En TF hay que pasar training=True explicitamente.")
