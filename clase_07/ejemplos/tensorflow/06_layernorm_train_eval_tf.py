"""
Ejemplo 6 — LayerNorm: igual en train y eval (TensorFlow)

Objetivo: ver que LayerNormalization NO cambia entre training=True/False.
          A diferencia de BatchNormalization, no tiene moving stats.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/06_layernorm_train_eval_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

ln = keras.layers.LayerNormalization(scale=False, center=False)

sample = tf.constant([[5.0, 10.0, 15.0]])
print(f"Muestra: {sample.numpy().tolist()[0]}")

# =============================================
# Paso 1: LayerNorm con training=True
# =============================================
result_train = ln(sample, training=True)
print(f"\ntraining=True  -> {[round(v, 4) for v in result_train.numpy().tolist()[0]]}")

# =============================================
# Paso 2: LayerNorm con training=False
# =============================================
result_eval = ln(sample, training=False)
print(f"training=False -> {[round(v, 4) for v in result_eval.numpy().tolist()[0]]}")

# =============================================
# Comparar
# =============================================
are_equal = tf.reduce_all(tf.abs(result_train - result_eval) < 1e-6).numpy()
print(f"\nSon iguales? {are_equal}")

# =============================================
# Por que?
# =============================================
print(f"\nPor que son iguales?")
print(f"  LayerNorm normaliza cada muestra usando SUS PROPIOS features.")
print(f"  No tiene moving_mean ni moving_variance.")
print(f"  El parametro training=True/False NO le afecta.")
print(f"")
print(f"  BatchNorm: training=True/False cambia el comportamiento")
print(f"  LayerNorm: training=True/False no cambia nada")
