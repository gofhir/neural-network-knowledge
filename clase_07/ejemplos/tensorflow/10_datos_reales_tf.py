"""
Ejemplo 10 — Como se ven los datos REALES como tensores (TensorFlow)

Objetivo: ver que imagenes, texto, audio y tablas se convierten
          en arrays de numeros antes de entrar a la red.
          La red nunca ve "una foto" o "una palabra", solo ve numeros.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/tensorflow/10_datos_reales_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import numpy as np

SEPARATOR = "=" * 60

# =============================================
# 1. DATOS TABULARES (lo mas parecido a nuestros ejemplos)
# =============================================
print(SEPARATOR)
print("1. DATOS TABULARES (CSV / Excel)")
print(SEPARATOR)

print("""
  Imagina una tabla de pacientes en un hospital:

  | Edad | Peso(kg) | Presion | Colesterol | → Enfermo? |
  |  25  |   70     |  120    |    180     |     0      |
  |  55  |   95     |  145    |    250     |     1      |
  |  40  |   80     |  130    |    200     |     0      |
""")

x_tabular = tf.constant([
    [25.0, 70.0, 120.0, 180.0],
    [55.0, 95.0, 145.0, 250.0],
    [40.0, 80.0, 130.0, 200.0],
])
y_tabular = tf.constant([0, 1, 0])

print(f"  Tensor de entrada: shape = {x_tabular.shape}")
print(f"    → {x_tabular.shape[0]} pacientes, {x_tabular.shape[1]} features")
print(f"  Tensor:\n{x_tabular.numpy()}")
print(f"\n  Etiquetas: {y_tabular.numpy().tolist()}")

# Pasar por una red
model_tabular = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dense(2),
])
output = model_tabular(x_tabular)
print(f"\n  Salida de la red: {output.shape}")
print(f"    → 3 pacientes, 2 clases (sano/enfermo)")

# =============================================
# 2. IMAGENES (MNIST - digitos escritos a mano)
# =============================================
print(f"\n{SEPARATOR}")
print("2. IMAGENES (MNIST - digitos escritos a mano)")
print(SEPARATOR)

print("""
  Una imagen de MNIST es un digito escrito a mano (28x28 pixeles).
  Cada pixel es un numero entre 0 (negro) y 1 (blanco).
""")

# Descargar MNIST (viene con Keras)
(x_train, y_train), _ = keras.datasets.mnist.load_data()

# Tomar UNA imagen y normalizar a [0, 1]
image = x_train[0].astype('float32') / 255.0
label = y_train[0]

print(f"  Imagen shape: {image.shape}")
print(f"    → {image.shape[0]}x{image.shape[1]} pixeles")
print(f"  Etiqueta: {label} (es el digito '{label}')")
print(f"  Total numeros: {image.size} (28 x 28)")

# Mostrar los pixeles como texto
print(f"\n  Los primeros 10x10 pixeles del digito '{label}':")
print(f"  (0.0 = negro, 1.0 = blanco)")
print()
for row in image[:10, :10]:
    line = ""
    for val in row:
        if val > 0.5:
            line += " ##"
        elif val > 0.1:
            line += " .."
        else:
            line += "   "
    print(f"  {line}")

# Un batch de imagenes
# En TF/Keras, las imagenes van con canal al FINAL: (batch, alto, ancho, canal)
batch_images = x_train[:32].astype('float32') / 255.0
batch_images = batch_images[..., np.newaxis]  # agregar canal: (32, 28, 28, 1)
print(f"\n  Un BATCH de 32 imagenes: shape = {batch_images.shape}")
print(f"    → (batch=32, alto=28, ancho=28, canales=1)")
print(f"    → TF pone los canales AL FINAL (PyTorch los pone al principio)")

# Pasar por una red simple
model_imagen = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),  # (32, 28, 28, 1) → (32, 784)
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),                          # 10 digitos
])
output = model_imagen(batch_images)
print(f"  Salida de la red: {output.shape}")
print(f"    → 32 imagenes, 10 clases (una por digito)")

# =============================================
# 3. TEXTO (tokenizacion + embedding)
# =============================================
print(f"\n{SEPARATOR}")
print("3. TEXTO (tokenizacion + embedding)")
print(SEPARATOR)

print("""
  Las redes no entienden palabras. Hay que convertirlas a numeros.
  Se hace en 2 pasos: tokenizar (palabra → ID) y embedding (ID → vector).
""")

# Paso 1: Vocabulario simple
vocab = {
    "<pad>": 0, "el": 1, "gato": 2, "come": 3,
    "pescado": 4, "perro": 5, "duerme": 6, "mucho": 7,
}
print(f"  Vocabulario: {vocab}")

# Tokenizar frases
sentences = ["el gato come pescado", "el perro duerme mucho"]
print(f"\n  Frases originales:")
tokenized = []
for s in sentences:
    ids = [vocab[word] for word in s.split()]
    tokenized.append(ids)
    print(f"    '{s}' → {ids}")

x_text = tf.constant(tokenized)
print(f"\n  Paso 1 - Tokenizar: shape = {x_text.shape}")
print(f"    → {x_text.shape[0]} frases, {x_text.shape[1]} tokens cada una")

# Paso 2: Embedding
embedding_dim = 8
embedding = keras.layers.Embedding(input_dim=len(vocab), output_dim=embedding_dim)

x_embedded = embedding(x_text)
print(f"\n  Paso 2 - Embedding: shape = {x_embedded.shape}")
print(f"    → {x_embedded.shape[0]} frases, {x_embedded.shape[1]} tokens, {x_embedded.shape[2]} features")

gato_vector = x_embedded[0, 1]  # frase 0, token 1 (gato)
print(f"\n  La palabra 'gato' (ID=2) se convierte en:")
print(f"    {[round(float(v), 4) for v in gato_vector.numpy()]}")
print(f"    → Un vector de {embedding_dim} numeros que REPRESENTA 'gato'")

# Pasar por una red
model_texto = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2),
])
output = model_texto(x_embedded)
print(f"\n  Salida de la red: {output.shape}")
print(f"    → 2 frases, 2 clases")

# =============================================
# 4. AUDIO (onda → espectrograma)
# =============================================
print(f"\n{SEPARATOR}")
print("4. AUDIO (onda sonora → numeros)")
print(SEPARATOR)

print("""
  El sonido es una onda que vibra en el tiempo.
  Se muestrea miles de veces por segundo para convertirla en numeros.
""")

# Generar una nota musical sintetica (La = 440 Hz)
sample_rate = 16000
duration = 0.5
t = np.linspace(0, duration, int(sample_rate * duration), dtype='float32')

frequency = 440.0
waveform = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t)).astype('float32')

print(f"  Nota musical 'La' (440 Hz), 0.5 segundos:")
print(f"  Waveform shape: {waveform.shape}")
print(f"    → {waveform.shape[0]} muestras ({sample_rate} por segundo x {duration}s)")

print(f"\n  Primeros 20 valores de la onda:")
print(f"    {[round(float(v), 4) for v in waveform[:20]]}")

# Calcular espectrograma con TF
waveform_tf = tf.constant(waveform)
spectrogram = tf.signal.stft(waveform_tf, frame_length=512, frame_step=160)
spectrogram = tf.abs(spectrogram)

print(f"\n  Espectrograma shape: {spectrogram.shape}")
print(f"    → {spectrogram.shape[0]} frames de tiempo x {spectrogram.shape[1]} frecuencias")
print(f"    → Se ve como una 'imagen' del sonido")

# Para la red (batch, alto, ancho, canal)
x_audio = spectrogram[tf.newaxis, :, :, tf.newaxis]
print(f"\n  Para la red: shape = {x_audio.shape}")
print(f"    → (batch=1, tiempo={spectrogram.shape[0]}, frecuencias={spectrogram.shape[1]}, canal=1)")

# =============================================
# RESUMEN
# =============================================
print(f"\n{SEPARATOR}")
print("RESUMEN: todo se convierte a tensores")
print(SEPARATOR)
print(f"""
  Tipo de dato      Conversion                Shape tipico
  ──────────        ──────────                ────────────
  Tabla/CSV         Ya son numeros            (batch, features)
  Imagen B/N        Pixeles [0-1]             (batch, 28, 28, 1)  *canales al final en TF
  Imagen color      Pixeles RGB [0-1]         (batch, 224, 224, 3)
  Texto             Tokenizar → Embedding     (batch, tokens, embed_dim)
  Audio             Onda → Espectrograma      (batch, time, freq, 1)

  Diferencia clave con PyTorch:
    PyTorch: canales PRIMERO  → (batch, canales, alto, ancho)
    TF:      canales AL FINAL → (batch, alto, ancho, canales)
""")
