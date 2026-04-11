"""
Ejemplo 3 — LayerNorm normaliza por FILA (TensorFlow)

Objetivo: ver que LayerNormalization calcula media y varianza de cada MUESTRA
          (fila) a traves de todos sus features.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/03_layernorm_por_fila_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

# --- Mismos datos que el ejemplo 02 ---
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
# Paso 1: Calcular media y std por FILA (a mano)
# =============================================
print(f"\nPaso 1: Estadisticas por FILA (axis=1 = a traves de features)")
for i in range(4):
    row = x[i]
    mean = tf.reduce_mean(row)
    std = tf.math.reduce_std(row)
    print(f"  Muestra {i+1}: valores={row.numpy().tolist()}, media={mean.numpy():.2f}, std={std.numpy():.2f}")

# =============================================
# Paso 2: Normalizar a mano (muestra 1)
# =============================================
row = x[0]
mean = tf.reduce_mean(row)
std = tf.math.reduce_std(row)
print(f"\nPaso 2: Normalizar muestra 1 a mano")
print(f"  Valores: {row.numpy().tolist()}")
print(f"  Media:   {mean.numpy():.4f}")
print(f"  Std:     {std.numpy():.4f}")
norm_row = (row - mean) / std
print(f"  Normalizado: {[round(v, 4) for v in norm_row.numpy().tolist()]}")

# =============================================
# Paso 3: Aplicar LayerNormalization
# =============================================
# En TF, LayerNormalization normaliza sobre el ultimo eje por defecto (axis=-1)
# Eso es equivalente a normalized_shape=3 en PyTorch
ln = keras.layers.LayerNormalization(
    scale=False, center=False, epsilon=1e-5
)
result = ln(x)

print(f"\nPaso 3: Despues de LayerNorm")
print(f"         Feature1  Feature2  Feature3")
for i in range(4):
    print(f"  Muestra {i+1}:  {result[i, 0].numpy():6.4f}   {result[i, 1].numpy():6.4f}   {result[i, 2].numpy():7.4f}")

# =============================================
# Paso 4: Verificar media~0 y varianza~1 por fila
# =============================================
print(f"\nPaso 4: Verificar que cada FILA tiene media~0 y varianza~1")
for i in range(4):
    row = result[i]
    print(f"  Muestra {i+1}: media={tf.reduce_mean(row).numpy():.4f}"
          f", varianza={tf.math.reduce_variance(row).numpy():.4f}")

# =============================================
# Diferencia de API
# =============================================
print(f"\nDiferencia de API:")
print(f"  PyTorch:     nn.LayerNorm(normalized_shape=3)")
print(f"  TensorFlow:  layers.LayerNormalization(axis=-1)  # -1 es el default")
print(f"  En TF normaliza sobre el ultimo eje por defecto, que son los features.")
