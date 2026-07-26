---
title: "Del WAV al tensor: Mel, SNR y SpecAugment desde cero"
weight: 1
math: true
---

La [teoría de la Clase 37](/clases/clase-37/teoria) recorre cómo el audio pasa del archivo en disco al **tensor** que entra al modelo, y cómo aumentarlo. Este capítulo lo construye **desde cero**, sin `torchaudio` ni `librosa`: partimos de una señal, calculamos su **espectrograma Mel**, y aplicamos las dos augmentations centrales de la clase —**sumar ruido calibrado por SNR** y **SpecAugment**. Reutilizamos la maquinaria de STFT y banco de filtros Mel de la [práctica de la Clase 35](/clases/clase-35/practica/02-stft-espectrograma-y-mfcc-desde-cero).

> **Lecturas de apoyo:** los fundamentos [Representación de audio](/fundamentos/representacion-de-audio) y [Data augmentation de audio](/fundamentos/data-augmentation-de-audio); los papers de [SpecAugment](/papers/specaugment-park-2019) y [GTZAN](/papers/gtzan-tzanetakis-2002).

---

## 1. De la señal al espectrograma Mel

Empecemos con una señal de juguete (dos tonos) y su espectrograma Mel. La STFT y el banco Mel son los del [camino 02 de la Clase 35](/clases/clase-35/practica/02-stft-espectrograma-y-mfcc-desde-cero); aquí los usamos como caja negra.

```python
import numpy as np
sr = 16000                                     # sample rate
t = np.arange(0, 2, 1/sr)
signal = np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*3000*t)   # 440 Hz + 3 kHz

def mel_spectrogram(x, sr, n_fft=512, hop=160, n_mels=64):
    # STFT (magnitud^2) — ver Clase 35, camino 02
    win = np.hanning(n_fft)
    frames = [np.abs(np.fft.rfft(x[i:i+n_fft]*win))**2
              for i in range(0, len(x)-n_fft, hop)]
    S = np.array(frames).T                      # [freq, tiempo]
    # banco de filtros Mel (triangulares en la escala Mel)
    fb = mel_filterbank(n_mels, n_fft, sr)      # [n_mels, freq]  (Clase 35)
    return np.log(fb @ S + 1e-8)                # log-Mel [n_mels, frames]

M = mel_spectrogram(signal, sr)
print("espectrograma Mel:", M.shape)            # (64, ~199)
```

### 1.1 Las dimensiones del tensor

La clase da la fórmula (idéntica al output de una convolución): con $N$ muestras, ventana $W$ y salto $H$,

$$
\text{frames} = \left\lfloor \frac{N - W}{H} \right\rfloor + 1.
$$

```python
N, W, H = len(signal), 512, 160
frames = (N - W) // H + 1
print("frames esperados:", frames, "→ tensor (1, 64, %d)" % frames)
```

{{< concept-alert type="recordar" >}}
El tensor final es $(\text{canales}, \text{n\_mels}, \text{frames})$: una **imagen 2D** por canal (frecuencia × tiempo). Esa es la clave de la clase —*si es una imagen, se le puede tirar una CNN*. La ventana es el **kernel**; el hop, el **stride**.
{{< /concept-alert >}}

---

## 2. Sumar ruido: la perilla es el SNR

La clase insiste: la cantidad de ruido se calibra por **potencia** (SNR en dB), no por amplitud. Para mezclar ruido a un SNR objetivo hay que escalar el ruido por el factor correcto.

```python
def add_noise_snr(signal, noise, snr_db):
    """Suma `noise` a `signal` escalado para lograr el SNR pedido (en dB)."""
    p_signal = np.mean(signal**2)               # potencia = promedio de amplitud^2
    p_noise  = np.mean(noise**2)
    alpha = np.sqrt(p_signal / (p_noise * 10**(snr_db/10)))
    return signal + alpha * noise

noise = np.random.randn(len(signal))            # ruido blanco
for snr in [20, 10, 3]:
    mix = add_noise_snr(signal, noise, snr)
    p_ratio = 10*np.log10(np.mean(signal**2) / np.mean((mix-signal)**2))
    print(f"SNR pedido={snr:2d} dB  →  medido={p_ratio:.1f} dB")   # coincide
```

El SNR medido coincide con el pedido: $+20$ dB el ruido apenas se percibe, $+3$ dB casi tan fuerte como la señal. En entrenamiento, se **sortea** el SNR en un rango (p. ej. 10-20 dB) para que cada época suene distinta:

```python
def augment_noise(signal, noise, snr_range=(10, 20)):
    snr = np.random.uniform(*snr_range)          # distinto cada época
    return add_noise_snr(signal, noise, snr)
```

{{< concept-alert type="advertencia" >}}
El ruido blanco ya regulariza, pero es mejor **ruido real** del entorno de despliegue —que el ruido de train se parezca al de producción. Y ojo: sumar ruido es una invariancia válida para clasificar género o detectar tos, pero puede no serlo para tareas donde el ruido *es* la señal.
{{< /concept-alert >}}

---

## 3. SpecAugment: tapar el espectrograma

En vez de sumar, **tapar**. SpecAugment hace cero bandas de frecuencia (filas) y tramos de tiempo (columnas) del espectrograma —Cutout con estructura, casi gratis.

```python
def spec_augment(S, n_freq_masks=2, n_time_masks=2, F=12, T=20):
    """Enmascara bandas de frecuencia y tramos de tiempo (in-place sobre una copia)."""
    S = S.copy(); n_mels, n_frames = S.shape
    for _ in range(n_freq_masks):
        f = np.random.randint(0, F)              # ancho de la banda
        f0 = np.random.randint(0, n_mels - f)    # inicio
        S[f0:f0+f, :] = 0                        # <- filas a cero
    for _ in range(n_time_masks):
        tt = np.random.randint(0, T)
        t0 = np.random.randint(0, n_frames - tt)
        S[:, t0:t0+tt] = 0                       # <- columnas a cero
    return S

M_aug = spec_augment(M)
print("bandas/tramos tapados:", int((M_aug == 0).sum()), "celdas")
```

Por qué funciona: apaga regiones **contiguas**, forzando al modelo a no depender de una sola banda o instante —si dependía de una, taparla entera lo obliga a buscar evidencia en otra parte. Y como opera sobre el espectrograma (que en producción ya está en la GPU), es dos líneas en el loop de entrenamiento.

---

## 4. El pipeline en triple framework

Una vez que el audio es un tensor, el espectrograma Mel (un producto matricial + log) y SpecAugment (poner a cero por índices) son operaciones tensoriales idénticas. Aquí SpecAugment —el corazón de la augmentation— en los tres frameworks (el banco `fb` se precomputa con NumPy).

### PyTorch

```python
import torch

def spec_augment_torch(S, F=12, T=20):          # S: [n_mels, frames]
    S = S.clone(); n_mels, n_frames = S.shape
    f = torch.randint(0, F, (1,)).item(); f0 = torch.randint(0, n_mels-f, (1,)).item()
    S[f0:f0+f, :] = 0
    tt = torch.randint(0, T, (1,)).item(); t0 = torch.randint(0, n_frames-tt, (1,)).item()
    S[:, t0:t0+tt] = 0
    return S
```

### TensorFlow

```python
import tensorflow as tf

def spec_augment_tf(S, F=12, T=20):             # S: [n_mels, frames]
    n_mels, n_frames = S.shape
    f = tf.random.uniform([], 0, F, tf.int32); f0 = tf.random.uniform([], 0, n_mels-f, tf.int32)
    tt = tf.random.uniform([], 0, T, tf.int32); t0 = tf.random.uniform([], 0, n_frames-tt, tf.int32)
    freq_mask = tf.logical_and(tf.range(n_mels)[:,None] >= f0, tf.range(n_mels)[:,None] < f0+f)
    time_mask = tf.logical_and(tf.range(n_frames)[None,:] >= t0, tf.range(n_frames)[None,:] < t0+tt)
    mask = tf.logical_or(freq_mask, time_mask)
    return tf.where(mask, 0.0, S)                # pone a cero donde la máscara es True
```

### JAX

```python
import jax.numpy as jnp, jax

def spec_augment_jax(S, key, F=12, T=20):
    n_mels, n_frames = S.shape
    k1, k2, k3, k4 = jax.random.split(key, 4)
    f  = jax.random.randint(k1, (), 0, F);  f0 = jax.random.randint(k2, (), 0, n_mels-f)
    tt = jax.random.randint(k3, (), 0, T);  t0 = jax.random.randint(k4, (), 0, n_frames-tt)
    rows = (jnp.arange(n_mels)[:,None] >= f0) & (jnp.arange(n_mels)[:,None] < f0+f)
    cols = (jnp.arange(n_frames)[None,:] >= t0) & (jnp.arange(n_frames)[None,:] < t0+tt)
    return jnp.where(rows | cols, 0.0, S)
```

Las tres hacen lo mismo: construir una **máscara** de las bandas/tramos elegidos y poner esas celdas a cero. Igual que la DFT o los filtros Mel, la augmentation de audio es, en el fondo, manipulación de tensores.

---

## 5. Qué nos llevamos

- El audio se convierte en un tensor 2D (**espectrograma Mel**), con dimensiones calculables como el output de una convolución.
- **Sumar ruido** se calibra por **SNR** (potencia, no amplitud), sorteando el nivel en un rango cada época.
- **SpecAugment** tapa bandas y tramos del espectrograma —Cutout con estructura, casi gratis en la GPU.
- Todo es álgebra de tensores: idéntico en NumPy, PyTorch, TensorFlow y JAX.

En el [camino 02](/clases/clase-37/practica/02-collate-fn-y-batching-de-largo-variable) resolvemos el último eslabón: armar un **batch** con audios de largo variable.

---

**Ver también:** [Clase 37 - Teoría](/clases/clase-37/teoria) · [Clase 37 - Profundización](/clases/clase-37/profundizacion) · [Camino 02: collate_fn](/clases/clase-37/practica/02-collate-fn-y-batching-de-largo-variable) · [Laboratorio](/laboratorios/lab-37).
