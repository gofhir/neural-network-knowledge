"""
Ejemplo 5 — BatchNorm: entrenamiento vs inferencia (TensorFlow)

Objetivo: entender que BatchNormalization se comporta DISTINTO segun
          el parametro training=True/False. En TF no hay model.train()/eval(),
          se pasa directamente en cada llamada.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/05_batchnorm_train_vs_eval_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

bn = keras.layers.BatchNormalization()

# =============================================
# Paso 1: Entrenamiento — pasar varios batches
# =============================================
print("--- Paso 1: ENTRENAMIENTO (training=True) ---")
print("Pasamos 3 batches. En cada uno, BN:")
print("  - Normaliza usando la media/var del batch actual")
print("  - Actualiza el moving_mean y moving_variance acumulados\n")

batches = [
    tf.constant([[10.0], [12.0], [8.0], [14.0]]),
    tf.constant([[20.0], [22.0], [18.0], [24.0]]),
    tf.constant([[15.0], [17.0], [13.0], [19.0]]),
]

for i, batch in enumerate(batches):
    batch_mean = tf.reduce_mean(batch).numpy()
    batch_var = tf.math.reduce_variance(batch).numpy()

    # training=True: usa stats del batch actual, actualiza moving stats
    result = bn(batch, training=True)

    print(f"  Batch {i+1}: valores={tf.squeeze(batch).numpy().tolist()}")
    print(f"    Media del batch:     {batch_mean:.2f}")
    print(f"    Var del batch:       {batch_var:.2f}")
    print(f"    moving_mean:         {bn.moving_mean.numpy()[0]:.4f}")
    print(f"    moving_variance:     {bn.moving_variance.numpy()[0]:.4f}")
    print(f"    Salida normalizada:  {[round(v, 4) for v in tf.squeeze(result).numpy().tolist()]}")
    print()

# =============================================
# Paso 2: Inferencia — una sola muestra
# =============================================
print("--- Paso 2: INFERENCIA (training=False) ---")
print("Ahora usamos el modelo para predecir.\n")

sample = tf.constant([[16.0]])
# training=False: usa moving stats acumuladas
result = bn(sample, training=False)

moving_mean = bn.moving_mean.numpy()[0]
moving_var = bn.moving_variance.numpy()[0]

print(f"  Muestra: {sample.numpy()[0][0]}")
print(f"  Moving mean (acumulada):     {moving_mean:.4f}")
print(f"  Moving variance (acumulada): {moving_var:.4f}")
print(f"  Resultado: {result.numpy()[0][0]:.4f}")

# =============================================
# Paso 3: Que pasa si OLVIDAS training=False?
# =============================================
print(f"\n--- Paso 3: Que pasa si OLVIDAS training=False? ---")

sample1 = tf.constant([[16.0], [16.0]])
sample2 = tf.constant([[16.0], [100.0]])

# "Olvidamos" poner training=False
result1 = bn(sample1, training=True)
result2 = bn(sample2, training=True)

print(f"  Con training=True (MAL para inferencia):")
print(f"    Batch [16, 16]:  resultado para 16 = {result1[0].numpy()[0]:.4f}")
print(f"    Batch [16, 100]: resultado para 16 = {result2[0].numpy()[0]:.4f}")
print(f"  -> El MISMO valor (16) da resultados DISTINTOS!")

# =============================================
# Diferencia clave de API
# =============================================
print(f"\n--- Diferencia clave de API ---")
print(f"  PyTorch:      model.train() / model.eval()")
print(f"                Cambia el modo del MODELO COMPLETO")
print(f"                Afecta TODAS las capas de una vez")
print(f"")
print(f"  TensorFlow:   bn(x, training=True) / bn(x, training=False)")
print(f"                Se pasa en CADA LLAMADA a la capa")
print(f"                Mas explicito, pero mas facil de olvidar")
