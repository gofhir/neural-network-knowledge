"""
Ejemplo 9 — Red neuronal con LayerNorm (TensorFlow)

Objetivo: ver como se usa LayerNormalization dentro de una red.
          LayerNorm es el que se usa en Transformers (GPT, BERT, etc.)
          A diferencia de BatchNorm, training=True/False no le afecta.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/tensorflow/09_red_con_layernorm_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

# =============================================
# Paso 1: Construir red con LayerNorm
# =============================================
print("=" * 55)
print("Paso 1: Forward pass con LayerNorm")
print("=" * 55)

model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(10,)),
    keras.layers.LayerNormalization(),        # en vez de BatchNormalization
    keras.layers.ReLU(),
    keras.layers.Dense(32),
    keras.layers.LayerNormalization(),
    keras.layers.ReLU(),
    keras.layers.Dense(2),                    # salida sin normalizar
])

tf.random.set_seed(42)
x = tf.random.normal((8, 10)) * 5

print(f"\nEntrada: {x.shape}")

# Forward capa por capa
h = x
for layer in model.layers:
    h = layer(h)
    name = layer.name
    is_ln = "layer_norm" in name
    marker = "  <- normalizado!" if is_ln else ""
    print(f"  Despues de {name:20s}: media={tf.reduce_mean(h).numpy():+8.4f}"
          f", std={tf.math.reduce_std(h).numpy():.4f}{marker}")

# =============================================
# Paso 2: training=True vs False (no cambia)
# =============================================
print(f"\n{'='*55}")
print("Paso 2: training=True vs False (no cambia)")
print("=" * 55)

sample = tf.random.normal((1, 10))

output_train = model(sample, training=True)
output_eval = model(sample, training=False)

print(f"\n  training=True:  {[round(v, 4) for v in output_train.numpy().tolist()[0]]}")
print(f"  training=False: {[round(v, 4) for v in output_eval.numpy().tolist()[0]]}")

are_equal = tf.reduce_all(tf.abs(output_train - output_eval) < 1e-5).numpy()
print(f"  Son iguales? {are_equal}")
print(f"\n  -> Con LayerNorm, training=True/False da el MISMO resultado.")

# =============================================
# Paso 3: Funciona con batch_size=1
# =============================================
print(f"\n{'='*55}")
print("Paso 3: Funciona con batch_size=1")
print("=" * 55)

single_sample = tf.random.normal((1, 10))

output = model(single_sample)
print(f"\n  LayerNorm con batch=1: {[round(v, 4) for v in output.numpy().tolist()[0]]}")
print(f"  -> Funciona perfecto!")

# BatchNorm falla o da resultados ruidosos con batch=1
print(f"\n  BatchNorm con batch=1:")
bn_model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(10,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(2),
])
try:
    output_bn = bn_model(single_sample, training=True)
    print(f"  -> No da error, pero las estadisticas son ruidosas con 1 muestra")
    print(f"     Los resultados no son confiables")
except Exception as e:
    print(f"  -> ERROR: {e}")

# =============================================
# Paso 4: Parametros
# =============================================
print(f"\n{'='*55}")
print("Paso 4: Parametros de LayerNorm")
print("=" * 55)

ln_layer = model.layers[1]  # primera LayerNormalization
print(f"\n  '{ln_layer.name}' (64 features):")
for w in ln_layer.trainable_weights:
    print(f"    {w.name:30s} shape={w.shape}  (trainable)")
print(f"    Non-trainable weights: {len(ln_layer.non_trainable_weights)}")
print(f"\n  -> Solo gamma y beta. NO tiene moving_mean ni moving_variance.")

# =============================================
# Comparacion de API
# =============================================
print(f"\n{'='*55}")
print("Paso 5: Comparacion de API")
print("=" * 55)
print(f"""
  PyTorch:      nn.LayerNorm(64)
  TensorFlow:   layers.LayerNormalization()

  Diferencia: en PyTorch hay que pasar el numero de features.
  En TF lo infiere automaticamente del input.

  Construccion de la red:
    PyTorch: clase con __init__ + forward (mas control)
    TF:      Sequential([capa, capa, ...]) (mas rapido de escribir)
""")
