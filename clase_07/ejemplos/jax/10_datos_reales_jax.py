"""
Ejemplo 10 — Como se ven los datos REALES como tensores (JAX)

Objetivo: ver que imagenes, texto, audio y tablas se convierten
          en arrays de numeros antes de entrar a la red.
          JAX usa jax.numpy (jnp) en vez de torch/tf para crear tensores.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/10_datos_reales_jax.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
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

x_tabular = jnp.array([
    [25.0, 70.0, 120.0, 180.0],
    [55.0, 95.0, 145.0, 250.0],
    [40.0, 80.0, 130.0, 200.0],
])
y_tabular = jnp.array([0, 1, 0])

print(f"  Tensor de entrada: shape = {x_tabular.shape}")
print(f"    → {x_tabular.shape[0]} pacientes, {x_tabular.shape[1]} features")
print(f"  Tensor:\n{x_tabular}")

# Pasar por una red en JAX
class TabularNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        return x

key = jax.random.PRNGKey(0)
model_tabular = TabularNet()
params = model_tabular.init(key, x_tabular)
output = model_tabular.apply(params, x_tabular)
print(f"\n  Salida de la red: {output.shape}")
print(f"    → 3 pacientes, 2 clases (sano/enfermo)")

# =============================================
# 2. IMAGENES (generamos un digito sintetico)
# =============================================
print(f"\n{SEPARATOR}")
print("2. IMAGENES (digito sintetico 28x28)")
print(SEPARATOR)

print("""
  Una imagen es una grilla de pixeles. Cada pixel es un numero.
  Generamos un "1" sintetico para ver como se ve.
""")

# Crear un "1" sintetico de 28x28
image = np.zeros((28, 28), dtype='float32')
# Dibujar una linea vertical (el "1")
image[4:24, 13:16] = 1.0   # trazo principal
image[4:7, 10:14] = 0.7    # serifas arriba
image[22:25, 10:18] = 0.7  # base

image_jnp = jnp.array(image)

print(f"  Imagen shape: {image_jnp.shape}")
print(f"    → {image_jnp.shape[0]}x{image_jnp.shape[1]} pixeles")
print(f"  Total numeros: {image_jnp.size} (28 x 28)")

# Mostrar como texto
print(f"\n  El digito '1' sintetico (10x10 pixeles centrales):")
print(f"  (0.0 = negro, 1.0 = blanco)")
print()
for row in image[8:18, 8:18]:
    line = ""
    for val in row:
        if val > 0.5:
            line += " ##"
        elif val > 0.1:
            line += " .."
        else:
            line += "   "
    print(f"  {line}")

# Batch de imagenes
batch_images = jnp.stack([image_jnp] * 32)  # 32 copias
batch_images = batch_images[:, :, :, jnp.newaxis]  # agregar canal
print(f"\n  Un BATCH de 32 imagenes: shape = {batch_images.shape}")
print(f"    → (batch=32, alto=28, ancho=28, canal=1)")

# Pasar por una red
class ImageNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten: (32, 28, 28, 1) → (32, 784)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)   # 10 digitos
        return x

model_imagen = ImageNet()
params = model_imagen.init(key, batch_images)
output = model_imagen.apply(params, batch_images)
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

vocab = {
    "<pad>": 0, "el": 1, "gato": 2, "come": 3,
    "pescado": 4, "perro": 5, "duerme": 6, "mucho": 7,
}
print(f"  Vocabulario: {vocab}")

sentences = ["el gato come pescado", "el perro duerme mucho"]
print(f"\n  Frases originales:")
tokenized = []
for s in sentences:
    ids = [vocab[word] for word in s.split()]
    tokenized.append(ids)
    print(f"    '{s}' → {ids}")

x_text = jnp.array(tokenized)
print(f"\n  Paso 1 - Tokenizar: shape = {x_text.shape}")

# Embedding en JAX/Flax
embedding_dim = 8
embedding = nn.Embed(num_embeddings=len(vocab), features=embedding_dim)
embed_params = embedding.init(key, x_text)

x_embedded = embedding.apply(embed_params, x_text)
print(f"  Paso 2 - Embedding: shape = {x_embedded.shape}")
print(f"    → {x_embedded.shape[0]} frases, {x_embedded.shape[1]} tokens, {x_embedded.shape[2]} features")

gato_vector = x_embedded[0, 1]
print(f"\n  La palabra 'gato' (ID=2) se convierte en:")
print(f"    {[round(float(v), 4) for v in gato_vector]}")
print(f"    → Un vector de {embedding_dim} numeros")

# Pasar por una red
class TextNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        return x

model_texto = TextNet()
params = model_texto.init(key, x_embedded)
output = model_texto.apply(params, x_embedded)
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

sample_rate = 16000
duration = 0.5
t = jnp.linspace(0, duration, int(sample_rate * duration))

frequency = 440.0
waveform = jnp.sin(2 * jnp.pi * frequency * t) + 0.1 * jax.random.normal(key, t.shape)

print(f"  Nota musical 'La' (440 Hz), 0.5 segundos:")
print(f"  Waveform shape: {waveform.shape}")
print(f"    → {waveform.shape[0]} muestras ({sample_rate} por segundo x {duration}s)")

print(f"\n  Primeros 20 valores de la onda:")
print(f"    {[round(float(v), 4) for v in waveform[:20]]}")

# Espectrograma simple con FFT
# JAX no tiene STFT integrado, hacemos una version simple
frame_length = 512
hop_length = 160
n_frames = (len(waveform) - frame_length) // hop_length + 1

# Crear frames y aplicar FFT
frames = jnp.stack([waveform[i*hop_length:i*hop_length+frame_length]
                     for i in range(n_frames)])
window = jnp.hanning(frame_length)
windowed = frames * window
spectrogram = jnp.abs(jnp.fft.rfft(windowed))

print(f"\n  Espectrograma shape: {spectrogram.shape}")
print(f"    → {spectrogram.shape[0]} frames de tiempo x {spectrogram.shape[1]} frecuencias")
print(f"    → Se ve como una 'imagen' del sonido")

x_audio = spectrogram[jnp.newaxis, :, :, jnp.newaxis]
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
  Imagen B/N        Pixeles [0-1]             (batch, 28, 28, 1)
  Texto             Tokenizar → Embedding     (batch, tokens, embed_dim)
  Audio             Onda → Espectrograma      (batch, time, freq, 1)

  Diferencia clave de JAX:
    - Usa jnp.array() en vez de torch.tensor() o tf.constant()
    - Embedding es nn.Embed() (Flax) en vez de nn.Embedding (PyTorch)
    - Todo lo demas es identico: al final son arrays de numeros.
""")
