---
title: "Practica desde 0 - Análisis de Audio"
weight: 30
sidebar:
  open: true
---

La Clase 35 construye la cadena clásica del análisis de audio: **Fourier → digitalización → STFT → MFCC**. Esta práctica la implementa **desde cero**, sin librerías de audio (nada de `librosa`), para que cada operación quepa en la cabeza. El primer camino construye la **serie de Fourier y la DFT** (descomponer una señal en frecuencias); el segundo construye la **STFT, el espectrograma y los MFCC** (recuperar el tiempo y perceptualizar). Cada uno se muestra en **triple framework** (PyTorch, TensorFlow y JAX) —porque una vez que el audio es un tensor de números, la FFT, el enventanado y los bancos de filtros son operaciones tensoriales idénticas en cualquier framework.

## Caminos

{{< cards >}}
  {{< card link="01-serie-de-fourier-y-dft-desde-cero" title="01 - Serie de Fourier y DFT desde cero" subtitle="Reconstruir señales con armónicos, la DFT como matriz, el espectro; verificar contra np.fft (las 3 representaciones)" icon="code" >}}
  {{< card link="02-stft-espectrograma-y-mfcc-desde-cero" title="02 - STFT, espectrograma y MFCC desde cero" subtitle="Enventanado, STFT, banco de filtros Mel y DCT sobre una señal, en PyTorch, TensorFlow y JAX" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 35 - Teoría](/clases/clase-35/teoria) y [Profundización](/clases/clase-35/profundizacion).
- Nociones de números complejos y trigonometría; álgebra lineal básica.
- Python intermedio y NumPy; PyTorch básico. Útil: TensorFlow/Keras y JAX.
- GPU **no necesaria**: todo corre en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - Fourier / DFT | NumPy | PyTorch / TensorFlow / JAX (DFT tensorial) |
| 02 - STFT / MFCC | NumPy + PyTorch | TensorFlow 2.x, JAX |

## El hilo conductor

1. **Serie de Fourier y DFT**: reconstruimos señales (cuadrada, diente de sierra) sumando armónicos, y construimos la **DFT como un producto matriz-vector** para obtener el espectro. Verificamos que coincide con `np.fft.fft` —y por qué la FFT es la versión rápida de exactamente esta cuenta.
2. **STFT, espectrograma y MFCC**: aplicamos la DFT sobre **ventanas** deslizantes para recuperar la información temporal (el espectrograma), y luego construimos el **banco de filtros Mel + log + DCT** para obtener los MFCC —la representación perceptual del audio.

---

**Ver tambien:** [Clase 35 - Teoria](/clases/clase-35/teoria) · [Clase 35 - Profundizacion](/clases/clase-35/profundizacion) · Fundamentos: [Análisis de Fourier](/fundamentos/analisis-de-fourier) · [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia).
