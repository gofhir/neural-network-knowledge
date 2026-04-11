"""
Ejemplo 13 — Como se transforma AUDIO en numeros

Objetivo: ver paso a paso como un sonido se convierte
          en un tensor de numeros. Incluye graficos de la onda
          y del espectrograma.

Genera graficos en /app/output/:
  - 08_audio_onda.png
  - 09_audio_muestreo.png
  - 10_audio_espectrograma.png
  - 11_audio_pipeline_completo.png

Ejecutar:
  docker run --rm -v $(pwd)/output:/app/output clase7-pytorch \
    python -u ejemplos/transformaciones/13_transformar_audio.py
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = "/app/output"

# =============================================
# 1. Que es el sonido
# =============================================
print("=" * 60)
print("PASO 1: El sonido es una onda")
print("=" * 60)

print(f"""
  El sonido es VIBRACION del aire.
  Cuando tocas una nota 'La' en un piano, el aire vibra
  440 veces por segundo (440 Hz).

  Un microfono convierte esa vibracion en una senal electrica:
  una onda que sube y baja con el tiempo.
""")

# Generar 3 notas musicales
sample_rate = 16000  # 16,000 mediciones por segundo
duration = 0.05      # 50 milisegundos (para ver la onda clara)
t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

# Nota La (440 Hz), Do (261 Hz), y una mezcla
note_la = np.sin(2 * np.pi * 440 * t)    # La: 440 Hz
note_do = np.sin(2 * np.pi * 261 * t)    # Do: 261 Hz
chord = note_la + note_do                  # acorde (2 notas juntas)

# Graficar las ondas
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t * 1000, note_la, color='steelblue', linewidth=1.5)
axes[0].set_title("Nota La (440 Hz) - vibra 440 veces por segundo", fontsize=12)
axes[0].set_ylabel("Amplitud")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-2.5, 2.5)

axes[1].plot(t * 1000, note_do, color='coral', linewidth=1.5)
axes[1].set_title("Nota Do (261 Hz) - vibra mas lento", fontsize=12)
axes[1].set_ylabel("Amplitud")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-2.5, 2.5)

axes[2].plot(t * 1000, chord, color='forestgreen', linewidth=1.5)
axes[2].set_title("Acorde (La + Do juntos) - la onda se vuelve compleja", fontsize=12)
axes[2].set_ylabel("Amplitud")
axes[2].set_xlabel("Tiempo (milisegundos)")
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-2.5, 2.5)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/08_audio_onda.png", dpi=150)
plt.close()
print(f"  Guardado: {OUTPUT}/08_audio_onda.png")

# =============================================
# 2. Muestreo: onda continua → numeros discretos
# =============================================
print(f"\n{'=' * 60}")
print("PASO 2: Muestreo (onda → array de numeros)")
print("=" * 60)

print(f"""
  La onda del sonido es CONTINUA (infinitos puntos).
  El computador no puede almacenar infinitos puntos.

  Solucion: MUESTREAR (sample) la onda a intervalos regulares.
  - Calidad telefono: 8,000 muestras por segundo (8 kHz)
  - Calidad normal:   16,000 muestras por segundo (16 kHz)
  - Calidad CD:       44,100 muestras por segundo (44.1 kHz)
""")

# Generar 1 segundo de La (440 Hz) + ruido
duration_full = 0.5  # medio segundo
t_full = np.linspace(0, duration_full, int(sample_rate * duration_full), dtype=np.float32)
np.random.seed(42)
waveform = np.sin(2 * np.pi * 440 * t_full) + 0.1 * np.random.randn(len(t_full)).astype(np.float32)

print(f"  Nota La (440 Hz) + ruido, 0.5 segundos:")
print(f"  Sample rate: {sample_rate} Hz (muestras por segundo)")
print(f"  Duracion: {duration_full}s")
print(f"  Total muestras: {len(waveform)}")
print(f"  → Es un array de {len(waveform)} numeros!\n")

print(f"  Primeros 10 valores:")
for i in range(10):
    print(f"    t={t_full[i]*1000:.3f}ms → valor={waveform[i]:+.4f}")

# Grafico mostrando muestreo
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# Onda continua vs muestreada (zoom a 5ms)
t_zoom = t[:80]
wave_zoom = np.sin(2 * np.pi * 440 * t_zoom)

# "Continua" (muchos puntos)
t_continuous = np.linspace(0, t_zoom[-1], 1000)
wave_continuous = np.sin(2 * np.pi * 440 * t_continuous)
ax1.plot(t_continuous * 1000, wave_continuous, 'b-', alpha=0.3, label='Onda real (continua)')
ax1.stem(t_zoom[::4] * 1000, wave_zoom[::4], linefmt='r-', markerfmt='ro', basefmt=' ',
         label=f'Muestreada ({sample_rate//4} Hz)')
ax1.set_title(f"Muestreo: tomar valores a intervalos regulares", fontsize=12)
ax1.set_ylabel("Amplitud")
ax1.set_xlabel("Tiempo (ms)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Array completo
ax2.plot(np.arange(len(waveform)), waveform, linewidth=0.5, color='steelblue')
ax2.set_title(f"La onda completa: un array de {len(waveform)} numeros", fontsize=12)
ax2.set_xlabel(f"Indice de la muestra (0 a {len(waveform)-1})")
ax2.set_ylabel("Valor (-1 a +1)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/09_audio_muestreo.png", dpi=150)
plt.close()
print(f"\n  Guardado: {OUTPUT}/09_audio_muestreo.png")

# =============================================
# 3. Espectrograma: la "imagen" del sonido
# =============================================
print(f"\n{'=' * 60}")
print("PASO 3: Espectrograma (array → 'imagen' del sonido)")
print("=" * 60)

print(f"""
  El array de numeros es UNA dimension: valor vs tiempo.
  Pero un sonido real tiene muchas frecuencias a la vez
  (como un acorde con varias notas).

  El ESPECTROGRAMA descompone el sonido en sus frecuencias:
    - Eje X = tiempo
    - Eje Y = frecuencia (que nota suena)
    - Color = intensidad (que tan fuerte suena)

  Es como una "foto" del sonido. Y como es 2D,
  se puede procesar con una CNN como si fuera una imagen!
""")

# Crear un sonido mas interesante: nota que cambia
# La (440) por 0.25s, luego Do (261) por 0.25s
t1 = np.linspace(0, 0.25, int(sample_rate * 0.25), dtype=np.float32)
t2 = np.linspace(0, 0.25, int(sample_rate * 0.25), dtype=np.float32)
sound = np.concatenate([
    np.sin(2 * np.pi * 440 * t1),      # La
    np.sin(2 * np.pi * 261 * t2),      # Do
])

# Calcular espectrograma con PyTorch
waveform_tensor = torch.from_numpy(sound)
n_fft = 512
hop_length = 160

spectrogram_complex = torch.stft(
    waveform_tensor, n_fft=n_fft, hop_length=hop_length,
    return_complex=True, window=torch.hann_window(n_fft)
)
spectrogram = torch.abs(spectrogram_complex)

# Convertir a dB para mejor visualizacion
spec_db = 20 * torch.log10(spectrogram + 1e-10)

print(f"  Sonido: nota La (0-0.25s) seguida de nota Do (0.25-0.5s)")
print(f"  Waveform shape: {waveform_tensor.shape} → {len(waveform_tensor)} numeros")
print(f"  Espectrograma shape: {spectrogram.shape}")
print(f"    → {spectrogram.shape[0]} frecuencias x {spectrogram.shape[1]} frames de tiempo")

# Grafico del espectrograma
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# La onda
time_axis = np.linspace(0, 0.5, len(sound))
ax1.plot(time_axis, sound, linewidth=0.5, color='steelblue')
ax1.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='Cambio de nota')
ax1.set_title("Onda: nota La (440 Hz) → nota Do (261 Hz)", fontsize=12)
ax1.set_ylabel("Amplitud")
ax1.set_xlabel("Tiempo (segundos)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# El espectrograma
freq_axis = np.linspace(0, sample_rate / 2, spectrogram.shape[0])
time_spec = np.linspace(0, 0.5, spectrogram.shape[1])
im = ax2.pcolormesh(time_spec, freq_axis[:100], spec_db.numpy()[:100, :],
                     shading='gouraud', cmap='magma')
ax2.axhline(y=440, color='cyan', linestyle='--', alpha=0.7, label='440 Hz (La)')
ax2.axhline(y=261, color='lime', linestyle='--', alpha=0.7, label='261 Hz (Do)')
ax2.set_title("Espectrograma: la 'imagen' del sonido", fontsize=12)
ax2.set_ylabel("Frecuencia (Hz)")
ax2.set_xlabel("Tiempo (segundos)")
ax2.legend(loc='upper right')
plt.colorbar(im, ax=ax2, label='Intensidad (dB)')

plt.tight_layout()
plt.savefig(f"{OUTPUT}/10_audio_espectrograma.png", dpi=150)
plt.close()
print(f"\n  Guardado: {OUTPUT}/10_audio_espectrograma.png")
print(f"\n  En el espectrograma se VE claramente:")
print(f"    - Primero suena La (linea brillante a 440 Hz)")
print(f"    - Despues suena Do (linea brillante a 261 Hz)")

# =============================================
# 4. Pipeline completo: audio → red neuronal
# =============================================
print(f"\n{'=' * 60}")
print("PASO 4: Pipeline completo")
print("=" * 60)

# Preparar para la red
x_audio = spectrogram.unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)
print(f"\n  Audio → red neuronal:")
print(f"    1. Onda:           shape = ({len(sound)},)       → {len(sound)} numeros")
print(f"    2. Espectrograma:  shape = {tuple(spectrogram.shape)}   → 'imagen' 2D")
print(f"    3. Agregar dims:   shape = {tuple(x_audio.shape)} → (batch, canal, freq, time)")
print(f"    4. Pasar a la red...")

# Red simple
import torch.nn as nn
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(spectrogram.shape[0] * spectrogram.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 3),  # 3 clases: habla, musica, ruido
)
output = model(x_audio)
print(f"    5. Salida:         shape = {tuple(output.shape)}    → 3 clases")

# Grafico pipeline completo
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5),
                          gridspec_kw={'width_ratios': [3, 3, 3, 2]})

# Onda
axes[0].plot(time_axis[:800], sound[:800], linewidth=0.8, color='steelblue')
axes[0].set_title(f"1. Onda\n({len(sound)} numeros)", fontsize=10)
axes[0].set_xlabel("Tiempo")
axes[0].set_ylabel("Amplitud")

# Espectrograma
axes[1].imshow(spec_db.numpy()[:80, :], aspect='auto', cmap='magma', origin='lower')
axes[1].set_title(f"2. Espectrograma\n{tuple(spectrogram.shape)}", fontsize=10)
axes[1].set_xlabel("Tiempo")
axes[1].set_ylabel("Frecuencia")

# Tensor para la red
axes[2].imshow(spectrogram.numpy()[:80, :], aspect='auto', cmap='viridis', origin='lower')
axes[2].set_title(f"3. Tensor\n{tuple(x_audio.shape)}", fontsize=10)
axes[2].set_xlabel("Tiempo")
axes[2].set_ylabel("Frecuencia")

# Salida
classes = ['Habla', 'Musica', 'Ruido']
probs = torch.softmax(output, dim=1).detach().numpy()[0]
bars = axes[3].barh(classes, probs, color=['steelblue', 'coral', 'gray'])
axes[3].set_title(f"4. Salida\n(3 clases)", fontsize=10)
axes[3].set_xlim(0, 1)

plt.suptitle("Pipeline: Audio → Espectrograma → Red Neuronal → Prediccion", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/11_audio_pipeline_completo.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Guardado: {OUTPUT}/11_audio_pipeline_completo.png")

# =============================================
# RESUMEN
# =============================================
print(f"\n{'=' * 60}")
print("RESUMEN: Audio → Tensor")
print("=" * 60)
print(f"""
  Sonido real (vibracion del aire)
    ↓  microfono + muestreo (16,000 veces/segundo)
  Array 1D de numeros: [0.02, -0.01, 0.05, ...]
    ↓  STFT (Short-Time Fourier Transform)
  Espectrograma 2D: (frecuencias x tiempo)
    ↓  agregar dims de batch y canal
  Tensor shape (1, 1, freq, time)
    ↓  pasar a la red (como si fuera una imagen)
  Salida shape (1, num_clases)

  El espectrograma es literalmente una "foto" del sonido.
  Por eso se puede usar una CNN para clasificar audio,
  exactamente igual que para clasificar imagenes.

  Graficos guardados en {OUTPUT}/
""")
