"""
Ejemplo 10 — Como se ven los datos REALES como tensores (PyTorch)

Objetivo: ver que imagenes, texto, audio y tablas se convierten
          en arrays de numeros antes de entrar a la red.
          La red nunca ve "una foto" o "una palabra", solo ve numeros.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/pytorch/10_datos_reales_pytorch.py
"""
import torch
import torch.nn as nn
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

# Cada fila = un paciente, cada columna = un feature
x_tabular = torch.tensor([
    [25.0, 70.0, 120.0, 180.0],   # paciente 1
    [55.0, 95.0, 145.0, 250.0],   # paciente 2
    [40.0, 80.0, 130.0, 200.0],   # paciente 3
])
y_tabular = torch.tensor([0, 1, 0])  # etiquetas (0=sano, 1=enfermo)

print(f"  Tensor de entrada: shape = {x_tabular.shape}")
print(f"    → {x_tabular.shape[0]} pacientes, {x_tabular.shape[1]} features")
print(f"  Tensor: {x_tabular}")
print(f"\n  Etiquetas: {y_tabular.tolist()}")
print(f"  → Esto es MUY PARECIDO a nuestros ejemplos anteriores!")

# Pasar por una red
model_tabular = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
output = model_tabular(x_tabular)
print(f"\n  Salida de la red: {output.shape}")
print(f"  → 3 pacientes, 2 clases (sano/enfermo)")

# =============================================
# 2. IMAGENES (MNIST - digitos escritos a mano)
# =============================================
print(f"\n{SEPARATOR}")
print("2. IMAGENES (MNIST - digitos escritos a mano)")
print(SEPARATOR)

from torchvision import datasets, transforms

# Descargar MNIST (solo las primeras imagenes)
mnist = datasets.MNIST(
    root='/tmp/data', train=True, download=True,
    transform=transforms.ToTensor()  # convierte imagen a tensor [0-1]
)

# Tomar UNA imagen
image, label = mnist[0]  # primer digito

print(f"""
  Una imagen de MNIST es un digito escrito a mano (28x28 pixeles).
  Cada pixel es un numero entre 0 (negro) y 1 (blanco).
""")
print(f"  Imagen shape: {image.shape}")
print(f"    → {image.shape[0]} canal (blanco/negro)")
print(f"    → {image.shape[1]}x{image.shape[2]} pixeles")
print(f"  Etiqueta: {label} (es el digito '{label}')")
print(f"  Total numeros: {image.numel()} (1 x 28 x 28)")

# Mostrar los pixeles como texto (esquina superior izquierda)
print(f"\n  Los primeros 10x10 pixeles del digito '{label}':")
print(f"  (0.0 = negro, 1.0 = blanco)")
print()
pixels = image[0, :10, :10]  # canal 0, primeras 10 filas, 10 columnas
for row in pixels:
    line = ""
    for val in row:
        if val > 0.5:
            line += " ##"    # pixel blanco (parte del digito)
        elif val > 0.1:
            line += " .."    # pixel gris
        else:
            line += "   "    # pixel negro (fondo)
    print(f"  {line}")

# Un batch de imagenes
batch_images = torch.stack([mnist[i][0] for i in range(32)])
print(f"\n  Un BATCH de 32 imagenes: shape = {batch_images.shape}")
print(f"    → (batch=32, canales=1, alto=28, ancho=28)")

# Pasar por una red simple
model_imagen = nn.Sequential(
    nn.Flatten(),           # (32, 1, 28, 28) → (32, 784)
    nn.Linear(784, 128),    # 784 pixeles → 128 features
    nn.ReLU(),
    nn.Linear(128, 10),     # → 10 clases (digitos 0-9)
)
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

# Paso 1: Vocabulario simple (en la practica se usan tokenizers como BPE)
vocab = {
    "<pad>": 0, "el": 1, "gato": 2, "come": 3,
    "pescado": 4, "perro": 5, "duerme": 6, "mucho": 7,
}
print(f"  Vocabulario: {vocab}")

# Tokenizar frases
sentences = ["el gato come pescado", "el perro duerme mucho"]
print(f"\n  Frases originales:")
for s in sentences:
    print(f"    '{s}'")

# Convertir a IDs
tokenized = []
for s in sentences:
    ids = [vocab[word] for word in s.split()]
    tokenized.append(ids)
    print(f"    '{s}' → {ids}")

# Convertir a tensor
x_text = torch.tensor(tokenized)
print(f"\n  Paso 1 - Tokenizar: shape = {x_text.shape}")
print(f"    → {x_text.shape[0]} frases, {x_text.shape[1]} tokens cada una")
print(f"  Tensor: {x_text}")

# Paso 2: Embedding (cada ID → vector denso)
embedding_dim = 8  # en GPT-3 son 12,288
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)

x_embedded = embedding(x_text)
print(f"\n  Paso 2 - Embedding: shape = {x_embedded.shape}")
print(f"    → {x_embedded.shape[0]} frases, {x_embedded.shape[1]} tokens, {x_embedded.shape[2]} features")

print(f"\n  La palabra 'gato' (ID=2) se convierte en:")
gato_vector = x_embedded[0, 1]  # frase 0, token 1 (gato)
print(f"    {[round(v, 4) for v in gato_vector.tolist()]}")
print(f"    → Un vector de {embedding_dim} numeros que REPRESENTA 'gato'")
print(f"    → Estos numeros se APRENDEN durante el entrenamiento")

# Pasar por una red simple
model_texto = nn.Sequential(
    nn.Flatten(),                          # (2, 4, 8) → (2, 32)
    nn.Linear(4 * embedding_dim, 16),
    nn.ReLU(),
    nn.Linear(16, 2),                      # 2 clases (positivo/negativo)
)
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
sample_rate = 16000  # 16,000 muestras por segundo
duration = 0.5       # medio segundo
t = torch.linspace(0, duration, int(sample_rate * duration))

# Nota La (440 Hz) + un poco de ruido
frequency = 440.0
waveform = torch.sin(2 * np.pi * frequency * t) + 0.1 * torch.randn_like(t)

print(f"  Nota musical 'La' (440 Hz), 0.5 segundos:")
print(f"  Waveform shape: {waveform.shape}")
print(f"    → {waveform.shape[0]} muestras ({sample_rate} por segundo x {duration}s)")

# Mostrar los primeros valores
print(f"\n  Primeros 20 valores de la onda:")
print(f"    {[round(v, 4) for v in waveform[:20].tolist()]}")
print(f"    → Numeros entre -1 y 1, oscilando como una onda")

# Convertir a espectrograma (representacion frecuencia vs tiempo)
# Usamos una FFT simple
n_fft = 512
hop_length = 160

# Calcular espectrograma manualmente con STFT
spectrogram_complex = torch.stft(
    waveform, n_fft=n_fft, hop_length=hop_length,
    return_complex=True, window=torch.hann_window(n_fft)
)
spectrogram = torch.abs(spectrogram_complex)

print(f"\n  Espectrograma shape: {spectrogram.shape}")
print(f"    → {spectrogram.shape[0]} frecuencias x {spectrogram.shape[1]} frames de tiempo")
print(f"    → Se ve como una 'imagen' del sonido")
print(f"    → Se puede procesar con una CNN como si fuera una foto!")

# Para la red, agregamos la dimension de batch y canal
x_audio = spectrogram.unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)
print(f"\n  Para la red: shape = {x_audio.shape}")
print(f"    → (batch=1, canal=1, frecuencias={spectrogram.shape[0]}, tiempo={spectrogram.shape[1]})")

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
  Imagen B/N        Pixeles [0-1]             (batch, 1, 28, 28)
  Imagen color      Pixeles RGB [0-1]         (batch, 3, 224, 224)
  Texto             Tokenizar → Embedding     (batch, tokens, embed_dim)
  Audio             Onda → Espectrograma      (batch, 1, freq, time)

  La red NUNCA ve la imagen, la palabra o el sonido.
  Solo ve arrays de numeros con distinta forma.
  Normalizar (BatchNorm/LayerNorm) funciona igual sin importar
  de donde vinieron los datos.
""")
