---
title: "Clase 35 - Introducción al Análisis de Audio"
weight: 350
sidebar:
  open: true
---

**Profesor:** Gabriel Sepúlveda (IALab)
**Módulo:** Audio y Video (parte 1 de la serie de audio)

Primera clase del módulo de **Audio y Video**: los **fundamentos de procesamiento de señales** que hacen falta antes de aplicar deep learning al sonido. La clase construye, eslabón por eslabón, toda la cadena clásica del análisis de audio. Parte de la **naturaleza física del sonido** (onda mecánica, el oído, el rango 20 Hz–20 kHz), pasa por el **análisis de Fourier** (descomponer una señal en frecuencias: series, transformada, espectro, la FFT), la **digitalización** (cuantización y muestreo, el teorema de Nyquist-Shannon), el **análisis tiempo-frecuencia** (la STFT, el espectrograma y el trade-off de Gabor), y culmina en los **MFCC** —una representación del audio calibrada a la percepción humana. Es la base sobre la que se apoyará todo el resto del módulo.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 67 diapositivas: naturaleza del sonido, Fourier/FFT, cuantización y sampling, STFT, MFCC" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: serie de Fourier y ortogonalidad, DFT/FFT y complejidad, teorema de muestreo y aliasing, STFT e incertidumbre de Gabor, pipeline MFCC" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="DFT/serie de Fourier y STFT/espectrograma/MFCC desde cero en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-35" title="Laboratorio: FFT, STFT y MFCC" subtitle="FFT, series de Fourier animadas y análisis tiempo-frecuencia sobre audio real con librosa" icon="variable" >}}
  {{< card link="/clases/clase-36" title="Clase siguiente: Introducción al Análisis de Video" subtitle="Movimiento, action recognition, flujo óptico, arquitecturas de video" icon="arrow-right" >}}
  {{< card link="/clases/clase-34" title="Clase anterior: Razonamiento" subtitle="Causalidad, Chain-of-Thought, test-time compute" icon="arrow-left" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/analisis-de-fourier" title="Análisis de Fourier" subtitle="Señal, serie y transformada de Fourier, espectro, armónicos, la FFT" icon="book-open" >}}
  {{< card link="/fundamentos/digitalizacion-de-audio" title="Digitalización de audio" subtitle="Cuantización, muestreo, teorema de Nyquist-Shannon, aliasing" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-tiempo-frecuencia" title="Representación tiempo-frecuencia (STFT)" subtitle="STFT, espectrograma, el trade-off e incertidumbre de Gabor" icon="book-open" >}}
  {{< card link="/fundamentos/mfcc-y-escala-mel" title="MFCC y escala Mel" subtitle="Escala Mel, bancos de filtros, cepstrum, DCT, el pipeline MFCC" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/fft-cooley-tukey-1965" title="FFT (Cooley & Tukey, 1965)" subtitle="El algoritmo que hace Fourier práctico: O(N²) → O(N log N)" icon="document-text" >}}
  {{< card link="/papers/sampling-shannon-1949" title="Teorema de muestreo (Shannon, 1949)" subtitle="Banda W → 2W muestras/s; reconstrucción sinc; aliasing" icon="document-text" >}}
  {{< card link="/papers/time-frequency-gabor-1946" title="Theory of Communication (Gabor, 1946)" subtitle="El principio de incertidumbre tiempo-frecuencia; los logones" icon="document-text" >}}
  {{< card link="/papers/mfcc-davis-mermelstein-1980" title="MFCC (Davis & Mermelstein, 1980)" subtitle="Bancos de filtros Mel + DCT: la representación clásica de la voz" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/audio" title="Dominio: Audio / Voz" subtitle="Línea de tiempo: de MFCC y HMM-GMM a wav2vec, Whisper y los foundation models de audio" icon="globe-alt" >}}
{{< /cards >}}
