---
title: "Serie de Fourier y DFT desde cero"
weight: 1
math: true
---

La [teoría de la Clase 35](/clases/clase-35/teoria) presenta el análisis de Fourier como la herramienta que **descompone una señal en frecuencias**. Este capítulo lo construye desde cero: primero **reconstruimos** señales sumando armónicos (la serie de Fourier), y luego **analizamos** una señal para extraer su espectro (la DFT), verificando que nuestra implementación a mano coincide con `np.fft.fft`. Sin librerías de audio: solo NumPy y las tres representaciones de framework.

> **Lecturas de apoyo:** el fundamento [Análisis de Fourier](/fundamentos/analisis-de-fourier) y el [paper de Cooley-Tukey](/papers/fft-cooley-tukey-1965) (la FFT es la versión $O(N\log N)$ de lo que aquí calculamos en $O(N^2)$).

---

## 1. Reconstruir con armónicos: la serie de Fourier

La serie de Fourier dice que una señal periódica es una suma de sinusoides. Comprobémoslo al revés: **construyamos** una señal cuadrada sumando sus armónicos. La clase nos da los coeficientes: solo armónicos impares, $b_k = 4/(k\pi)$.

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000, endpoint=False)   # 1 segundo
w0 = 2 * np.pi                                 # frecuencia fundamental (1 Hz)

def square_wave(t, n_harmonics):
    """Reconstruye una onda cuadrada con n armónicos impares."""
    y = np.zeros_like(t)
    for k in range(1, 2 * n_harmonics, 2):     # 1, 3, 5, ...
        y += (4 / (k * np.pi)) * np.sin(k * w0 * t)
    return y

for n in [1, 3, 10, 50]:
    plt.plot(t, square_wave(t, n), label=f"{n} armónicos")
plt.legend(); plt.title("Serie de Fourier de una onda cuadrada")
```

Con 1 armónico se obtiene una sinusoide; al sumar más, la señal **converge** a la forma cuadrada (con las oscilaciones de Gibbs en los saltos). El mismo procedimiento, con otros coeficientes, genera la triangular o la diente de sierra —justo lo que anima el [laboratorio](/laboratorios/lab-35).

{{< concept-alert type="clave" >}}
Esta es la **síntesis** (frecuencia → tiempo): dado el espectro (los coeficientes), reconstruimos la señal. El **análisis** (tiempo → frecuencia) es el problema inverso —dada la señal, hallar los coeficientes— y es lo que hace la DFT.
{{< /concept-alert >}}

---

## 2. La DFT como producto matriz-vector

La Transformada Discreta de Fourier de $N$ muestras es $X_k = \sum_{n=0}^{N-1} x_n\, e^{-2\pi i kn/N}$. Esto es, literalmente, **multiplicar la señal por una matriz** $W$ de exponenciales complejas: $X = W x$, con $W_{kn} = e^{-2\pi i kn/N}$.

```python
def dft_matrix(N):
    """La matriz DFT NxN: W[k,n] = exp(-2πi·kn/N)."""
    k = np.arange(N).reshape(-1, 1)   # columna
    n = np.arange(N).reshape(1, -1)   # fila
    return np.exp(-2j * np.pi * k * n / N)

def dft(x):
    return dft_matrix(len(x)) @ x

# Verificación: nuestra DFT == np.fft.fft
x = np.random.randn(64)
assert np.allclose(dft(x), np.fft.fft(x))
print("¡Coincide con np.fft.fft!")
```

Construir la matriz cuesta $O(N^2)$ y multiplicarla también —por eso la DFT directa es $O(N^2)$. La FFT obtiene **exactamente el mismo resultado** en $O(N\log N)$ explotando la estructura de $W$ (los factores se repiten); por eso `np.fft.fft` es lo que se usa en la práctica, pero calcula lo mismo que nuestra matriz.

---

## 3. El espectro de una señal

Apliquemos la DFT a una señal con dos frecuencias conocidas y veamos su espectro (la magnitud de los coeficientes).

```python
fs = 200                                    # frecuencia de muestreo (Hz)
t = np.arange(0, 1, 1/fs)                    # 1 segundo, 200 muestras
x = 3*np.sin(2*np.pi*5*t) + np.sin(2*np.pi*20*t)   # 5 Hz (amp 3) + 20 Hz (amp 1)

X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), d=1/fs)      # eje de frecuencias en Hz
half = len(x) // 2                          # solo frecuencias positivas
plt.stem(freqs[:half], np.abs(X[:half]) * 2/len(x))
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("Magnitud")
```

El espectro muestra **dos picos**: uno en 5 Hz (alto, amplitud 3) y otro en 20 Hz (bajo, amplitud 1) —exactamente las frecuencias y amplitudes que pusimos. La DFT **recuperó** la composición frecuencial de la señal. (Solo miramos la mitad del espectro porque para señales reales la otra mitad es su reflejo conjugado.)

---

## 4. La misma DFT en triple framework

Una vez que la señal es un tensor, la DFT es un producto matriz-vector complejo —idéntico en cualquier framework. Aquí la matriz DFT y su aplicación en los tres.

### PyTorch

```python
import torch

def dft_torch(x):
    N = x.shape[0]
    k = torch.arange(N).view(-1, 1)
    n = torch.arange(N).view(1, -1)
    W = torch.exp(-2j * torch.pi * k * n / N)   # matriz compleja
    return W @ x.to(torch.complex64)

x = torch.randn(64)
assert torch.allclose(dft_torch(x), torch.fft.fft(x), atol=1e-4)
```

### TensorFlow

```python
import tensorflow as tf

def dft_tf(x):
    N = tf.shape(x)[0]
    k = tf.reshape(tf.range(N), (-1, 1))
    n = tf.reshape(tf.range(N), (1, -1))
    ang = -2.0 * np.pi * tf.cast(k * n, tf.float32) / tf.cast(N, tf.float32)
    W = tf.complex(tf.cos(ang), tf.sin(ang))     # e^{iθ} = cosθ + i·sinθ
    return tf.linalg.matvec(W, tf.cast(x, tf.complex64))
```

### JAX

```python
import jax.numpy as jnp

def dft_jax(x):
    N = x.shape[0]
    k = jnp.arange(N).reshape(-1, 1)
    n = jnp.arange(N).reshape(1, -1)
    W = jnp.exp(-2j * jnp.pi * k * n / N)
    return W @ x.astype(jnp.complex64)
```

Las tres construyen la misma matriz de exponenciales complejas y la aplican. La lección: **la Transformada de Fourier es álgebra lineal** —una proyección de la señal sobre una base de sinusoides. Los frameworks solo cambian la sintaxis del producto.

---

## 5. Qué nos llevamos

- La **serie de Fourier** es síntesis (armónicos → señal); la **DFT** es análisis (señal → espectro). Son inversas.
- La **DFT es un producto matriz-vector** $X = Wx$ con exponenciales complejas: $O(N^2)$. La **FFT** calcula lo mismo en $O(N\log N)$.
- El **espectro** (magnitud de la DFT) recupera las frecuencias y amplitudes que componen una señal.
- Es álgebra lineal pura: idéntica en NumPy, PyTorch, TensorFlow y JAX.

En el [camino 02](/clases/clase-35/practica/02-stft-espectrograma-y-mfcc-desde-cero) aplicamos la DFT sobre **ventanas** para recuperar el tiempo (STFT) y construimos los **MFCC**.

---

**Ver también:** [Clase 35 - Teoría](/clases/clase-35/teoria) · [Clase 35 - Profundización](/clases/clase-35/profundizacion) · [Camino 02: STFT y MFCC](/clases/clase-35/practica/02-stft-espectrograma-y-mfcc-desde-cero) · [Laboratorio](/laboratorios/lab-35).
