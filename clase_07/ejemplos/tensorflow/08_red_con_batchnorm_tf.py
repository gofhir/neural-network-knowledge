"""
Ejemplo 8 — Red neuronal completa CON BatchNorm (TensorFlow)

Objetivo: ver como se usa BatchNormalization DENTRO de una red real.
          Comparamos la misma red con y sin BatchNorm para ver
          como afecta las activaciones durante el forward pass.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/tensorflow/08_red_con_batchnorm_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import numpy as np

# =============================================
# Paso 1: Red SIN BatchNorm
# =============================================
print("=" * 55)
print("Paso 1: Red SIN BatchNorm")
print("=" * 55)

# En TF/Keras podemos construir la red con Sequential
model_sin_bn = keras.Sequential([
    keras.layers.Dense(64, input_shape=(10,)),    # fc1
    keras.layers.ReLU(),
    keras.layers.Dense(32),                        # fc2
    keras.layers.ReLU(),
    keras.layers.Dense(2),                         # fc3 (salida)
])

# Datos de entrada: batch de 8 muestras, 10 features
tf.random.set_seed(42)
x = tf.random.normal((8, 10)) * 5

print(f"\nEntrada: {x.shape} (8 muestras, 10 features)")
print(f"Entrada media={tf.reduce_mean(x).numpy():+.4f}, std={tf.math.reduce_std(x).numpy():.4f}\n")

# Pasar capa por capa para ver las activaciones
print("Forward pass (capa por capa):")
h = x
for layer in model_sin_bn.layers:
    h = layer(h)
    name = layer.name
    print(f"    Despues de {name:10s}: media={tf.reduce_mean(h).numpy():+8.4f}"
          f", std={tf.math.reduce_std(h).numpy():.4f}")

print(f"\n  -> Las medias y std cambian sin control entre capas.\n")

# =============================================
# Paso 2: Red CON BatchNorm
# =============================================
print("=" * 55)
print("Paso 2: Red CON BatchNorm")
print("=" * 55)

# Misma arquitectura pero con BatchNormalization
model_con_bn = keras.Sequential([
    keras.layers.Dense(64, input_shape=(10,)),    # fc1
    keras.layers.BatchNormalization(),             # bn1
    keras.layers.ReLU(),
    keras.layers.Dense(32),                        # fc2
    keras.layers.BatchNormalization(),             # bn2
    keras.layers.ReLU(),
    keras.layers.Dense(2),                         # salida (sin BN)
])

print(f"\nEntrada: {x.shape} (mismos datos)")
print(f"Entrada media={tf.reduce_mean(x).numpy():+.4f}, std={tf.math.reduce_std(x).numpy():.4f}\n")

# Forward capa por capa
print("Forward pass (capa por capa):")
h = x
for layer in model_con_bn.layers:
    h = layer(h, training=True) if isinstance(layer, keras.layers.BatchNormalization) else layer(h)
    name = layer.name
    is_bn = "batch" in name
    marker = "  <- normalizado!" if is_bn else ""
    print(f"    Despues de {name:20s}: media={tf.reduce_mean(h).numpy():+8.4f}"
          f", std={tf.math.reduce_std(h).numpy():.4f}{marker}")

print(f"\n  -> Despues de cada BatchNorm, media vuelve a ~0 y std a ~1.\n")

# =============================================
# Paso 3: Que parametros agrego BatchNorm?
# =============================================
print("=" * 55)
print("Paso 3: Parametros de la red")
print("=" * 55)

# En Keras, .count_params() da el total
# Pero hay que distinguir trainable vs non-trainable
train_sin = sum(p.numpy().size for p in model_sin_bn.trainable_weights)
train_con = sum(p.numpy().size for p in model_con_bn.trainable_weights)
non_train_con = sum(p.numpy().size for p in model_con_bn.non_trainable_weights)

print(f"\n  Sin BatchNorm:")
print(f"    Trainable:     {train_sin}")
print(f"")
print(f"  Con BatchNorm:")
print(f"    Trainable:     {train_con}  (gamma y beta son trainable)")
print(f"    Non-trainable: {non_train_con}  (moving_mean y moving_var)")
print(f"    Diferencia:    {train_con - train_sin} parametros extra trainable")

# Desglose de un BatchNorm
bn_layer = model_con_bn.layers[1]  # primera BatchNorm
print(f"\n  Desglose de '{bn_layer.name}' (64 features):")
for w in bn_layer.trainable_weights:
    print(f"    {w.name:30s} shape={w.shape}  (trainable)")
for w in bn_layer.non_trainable_weights:
    print(f"    {w.name:30s} shape={w.shape}  (non-trainable)")

print(f"\n  -> BatchNorm agrega MUY POCOS parametros (2 trainable por feature)")

# =============================================
# Paso 4: Diferencia de API para construir redes
# =============================================
print(f"\n{'='*55}")
print("Paso 4: Comparacion de API")
print("=" * 55)
print(f"""
  PyTorch (clase con forward):         TensorFlow (Sequential):
  ─────────────────────────            ─────────────────────────
  self.fc1 = nn.Linear(10, 64)        Dense(64)
  self.bn1 = nn.BatchNorm1d(64)       BatchNormalization()
  self.relu = nn.ReLU()               ReLU()
  self.fc2 = nn.Linear(64, 32)        Dense(32)
  self.bn2 = nn.BatchNorm1d(32)       BatchNormalization()

  def forward(self, x):               # TF lo hace automatico
      x = self.relu(self.bn1(self.fc1(x)))  # en Sequential
      x = self.relu(self.bn2(self.fc2(x)))

  Orden: Linear -> BatchNorm -> ReLU -> ... -> Linear (salida)
  NUNCA poner BatchNorm en la ultima capa.
""")
