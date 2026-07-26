---
title: "STFT, espectrograma y MFCC desde cero"
weight: 2
math: true
---

El [camino 01](/clases/clase-35/practica/01-serie-de-fourier-y-dft-desde-cero) construyó la DFT y mostró cómo extraer el **espectro** de una señal —pero un espectro global pierde *cuándo* ocurre cada frecuencia. Este capítulo recupera el tiempo con la **STFT** (aplicar la DFT sobre ventanas deslizantes), construye el **espectrograma**, y termina con los **MFCC** (el banco de filtros Mel + log + DCT). Todo desde cero, sin `librosa`, en las tres representaciones de framework.

> **Lecturas de apoyo:** los fundamentos [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia) y [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel); los papers de [Gabor](/papers/time-frequency-gabor-1946) (incertidumbre) y [Davis-Mermelstein](/papers/mfcc-davis-mermelstein-1980) (MFCC).

---

## 1. Una señal que cambia en el tiempo

Para que la STFT tenga sentido, usemos un **chirp**: una señal cuya frecuencia **aumenta** con el tiempo. Su espectro de Fourier global mostraría "hay muchas frecuencias", pero no *en qué orden* —justo lo que la STFT revela.

```python
import numpy as np
fs = 4000                                     # 4 kHz de sampling
t = np.arange(0, 2, 1/fs)                      # 2 segundos
# frecuencia instantánea que sube de 100 a 800 Hz
x = np.sin(2*np.pi*(100*t + (700/2/2)*t**2))
```

---

## 2. La STFT desde cero

La STFT trocea la señal en **tramas** solapadas, aplica una **ventana** a cada una (para evitar discontinuidades en los bordes) y calcula su DFT. El resultado es una matriz frecuencia × tiempo.

```python
def stft(x, frame_len=256, hop=64):
    window = np.hanning(frame_len)                     # ventana de Hann
    frames = []
    for start in range(0, len(x) - frame_len, hop):
        frame = x[start:start+frame_len] * window      # enventanar
        spectrum = np.fft.rfft(frame)                  # DFT de la trama (solo freq+)
        frames.append(np.abs(spectrum))                # magnitud
    return np.array(frames).T                          # [freq, tiempo]

S = stft(x)
import matplotlib.pyplot as plt
plt.imshow(20*np.log10(S + 1e-8), origin="lower", aspect="auto")
plt.xlabel("Trama (tiempo)"); plt.ylabel("Bin de frecuencia")
plt.title("Espectrograma del chirp")
```

El espectrograma muestra una **línea diagonal** que sube: la frecuencia crece con el tiempo. Eso es lo que Fourier global no podía mostrar —la STFT localiza *cuándo* ocurre cada frecuencia.

{{< concept-alert type="advertencia" >}}
Prueba cambiar `frame_len`. Con `frame_len=1024` (ventana ancha) la diagonal se ve **nítida en frecuencia** pero **borrosa en tiempo**; con `frame_len=64` (ventana angosta), **nítida en tiempo** pero **borrosa en frecuencia**. Es el **principio de incertidumbre de Gabor** en acción: no puedes tener ambas resoluciones a la vez.
{{< /concept-alert >}}

---

## 3. Del espectrograma a los MFCC

Los MFCC agregan la **percepción humana**: comprimen el eje de frecuencias según la escala Mel, toman el log y decorrelacionan con la DCT.

```python
def hz_to_mel(f):  return 2595 * np.log10(1 + f/700)
def mel_to_hz(m):  return 700 * (10**(m/2595) - 1)

def mel_filterbank(n_filters, fft_len, fs):
    """Banco de n_filters triangulares espaciados en la escala Mel."""
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(fs/2), n_filters + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((fft_len + 1) * hz_pts / fs).astype(int)
    fb = np.zeros((n_filters, fft_len//2 + 1))
    for m in range(1, n_filters + 1):
        for k in range(bins[m-1], bins[m]):      # subida del triángulo
            fb[m-1, k] = (k - bins[m-1]) / (bins[m] - bins[m-1])
        for k in range(bins[m], bins[m+1]):      # bajada del triángulo
            fb[m-1, k] = (bins[m+1] - k) / (bins[m+1] - bins[m])
    return fb

def dct(x, n_coef=13):
    """Transformada Coseno Discreta (tipo II), sobre el eje de bandas."""
    M = x.shape[0]
    n = np.arange(M).reshape(-1, 1)
    k = np.arange(n_coef).reshape(1, -1)
    basis = np.cos(np.pi * k * (2*n + 1) / (2*M))   # [M, n_coef]
    return x.T @ basis                              # [tiempo, n_coef]

def mfcc(x, frame_len=256, hop=64, n_filters=26, n_coef=13):
    S = stft(x, frame_len, hop) ** 2                # espectro de potencia [freq, t]
    fb = mel_filterbank(n_filters, frame_len, fs)   # [n_filters, freq]
    mel_energy = fb @ S                             # [n_filters, t]
    log_mel = np.log(mel_energy + 1e-8)             # compresión logarítmica
    return dct(log_mel, n_coef)                     # [tiempo, n_coef]

coeffs = mfcc(x)
print("matriz MFCC:", coeffs.shape)                 # (n_tramas, 13)
```

Cada paso corresponde exactamente al pipeline de la clase: **FFT → filtros Mel → log → DCT**. El resultado es una matriz compacta (tramas × 13 coeficientes) que resume el audio en características perceptualmente relevantes.

---

## 4. El banco de filtros Mel en triple framework

El corazón de los MFCC —aplicar el banco de filtros al espectro— es un producto matricial. Aquí, en los tres frameworks (el banco `fb` se precomputa con NumPy como arriba).

### PyTorch

```python
import torch

def mel_apply_torch(S, fb):
    # S: [freq, t] espectro de potencia ; fb: [n_filters, freq]
    S, fb = torch.tensor(S), torch.tensor(fb)
    return torch.log(fb @ S + 1e-8)          # [n_filters, t] log-mel
```

### TensorFlow

```python
import tensorflow as tf

def mel_apply_tf(S, fb):
    S, fb = tf.constant(S, tf.float32), tf.constant(fb, tf.float32)
    return tf.math.log(tf.matmul(fb, S) + 1e-8)
```

### JAX

```python
import jax.numpy as jnp

def mel_apply_jax(S, fb):
    return jnp.log(jnp.asarray(fb) @ jnp.asarray(S) + 1e-8)
```

Las tres hacen lo mismo: **proyectar el espectro sobre el banco de filtros Mel** (un producto matricial) y tomar el log. Igual que la DFT del camino 01, los MFCC son, en el fondo, álgebra lineal sobre el espectro.

---

## 5. Qué nos llevamos

- La **STFT** recupera el tiempo aplicando la DFT sobre **ventanas** solapadas; su magnitud es el **espectrograma**.
- El **trade-off de la ventana** (nítido en tiempo *o* en frecuencia, nunca ambos) es el **principio de incertidumbre de Gabor**, y se puede *ver* cambiando `frame_len`.
- Los **MFCC** son el espectrograma comprimido por el **banco de filtros Mel**, un **log** y una **DCT** —cada paso motivado por la percepción humana.
- Todo el pipeline es álgebra lineal (productos matriciales), idéntico en NumPy, PyTorch, TensorFlow y JAX.

Con esto, el audio deja de ser una cadena cruda de muestras y se convierte en una **representación 2D** (espectrograma o MFCC) lista para alimentar un modelo —el punto de partida de todo el resto del módulo de [audio](/dominios/audio).

---

**Ver también:** [Clase 35 - Teoría](/clases/clase-35/teoria) · [Clase 35 - Profundización](/clases/clase-35/profundizacion) · [Camino 01: Fourier y DFT](/clases/clase-35/practica/01-serie-de-fourier-y-dft-desde-cero) · [Laboratorio](/laboratorios/lab-35).
