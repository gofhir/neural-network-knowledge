---
title: "Lab 35 - Introducción al Análisis de Audio"
weight: 350
sidebar:
  open: true
---

**Profesor:** Gabriel Sepúlveda (IALab)
**Módulo:** Audio y Video (parte 1)
**Notebook origen:** `clase_35/material/Laboratorio/laboratorio_35.ipynb`
**Notebook ejecutado:** [lab35.ipynb](/notebooks/lab35.ipynb) · [HTML](/notebooks-html/lab35.html)

## Encuadre

La contraparte práctica de la [clase 35](/clases/clase-35) y el **"hola mundo" del audio**: cómo se representa y analiza una señal de sonido *antes* de que una red neuronal la toque. No hay deep learning aquí — hay **fundamentos de procesamiento digital de señales**, que son de donde salen los tensores que después alimentan a Whisper, wav2vec2 y los clasificadores de sonido.

El hilo conductor es una **escalera de transformaciones**, donde cada nivel arregla la limitación del anterior:

| Transformación | Qué da | Qué limitación arregla |
|---|---|---|
| **FFT** | las frecuencias de la señal | ver el contenido frecuencial |
| **Serie de Fourier** | señal = suma de sinusoides | *por qué* funciona la FFT |
| **STFT** | cómo cambian las frecuencias en el tiempo | la FFT pierde el *cuándo* |
| **MFCC** | representación perceptual y compacta | la STFT no refleja cómo oímos |

## Resultados consolidados (medidos en el notebook)

| Representación | Shape (audio real) | Lectura |
|---|---|---|
| FFT (sinusoide 5 Hz) | pico en ±5 Hz | valida la herramienta |
| STFT (audio) | `(1025, 173)` | 1025 bins de frecuencia × 173 ventanas |
| MFCC (audio) | `(40, 173)` | 40 coeficientes — ~25× más compacto |
| Sampling rate (librosa) | `22050 Hz` | remuestreo automático por defecto |

### Las lecciones del lab

1. **El dominio de la frecuencia hace explícito lo implícito.** En el tiempo hay que contar ciclos; en frecuencia, "5 Hz" es un pico. Esa es toda la gracia de Fourier.
2. **La forma de la señal determina su espectro.** Cuadrada (salto) → armónicos $1/n$; triangular (pendiente) → $1/n^2$; sierra (asimétrica) → todos los armónicos. Brusquedad ↔ agudos.
3. **El offset vive en 0 Hz.** La componente DC de una señal es su promedio, y aparece en la frecuencia cero (Actividad 1.2).
4. **El espectrograma es una imagen del sonido** → se le aplican CNNs. Es el puente entre audio y visión.
5. **No puedes tener precisión en tiempo Y frecuencia a la vez** — el principio de incertidumbre de Gabor (Actividad 2.1).
6. **Los MFCC comprimen a lo perceptual**: de 1025 bins a 40 coeficientes descorrelacionados que capturan el timbre.

## Bloques del lab

{{< cards >}}
  {{< card link="01-analisis-en-frecuencia" title="Análisis en frecuencia: FFT y Fourier" subtitle="Teorema de muestreo (Nyquist), la FFT y su salida compleja, la serie de Fourier (cuadrada/triangular/sierra), y las actividades 1.1 (código) y 1.2 (la componente DC)" icon="variable" >}}
  {{< card link="02-stft-tiempo-frecuencia" title="Tiempo-frecuencia: la STFT" subtitle="Por qué la FFT pierde el cuándo, la FFT por ventanas, el espectrograma como imagen del sonido, decibeles, y la actividad 2.1 (incertidumbre de Gabor)" icon="adjustments" >}}
  {{< card link="03-escala-mel-mfcc" title="Escala Mel y MFCC" subtitle="La percepción logarítmica del tono, el banco de filtros Mel, el paso cepstral (DCT), y la escalera completa de representaciones (onda → FFT → STFT → Mel → MFCC)" icon="book-open" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/analisis-de-fourier" title="Análisis de Fourier" subtitle="FFT, serie de Fourier, dominio del tiempo vs frecuencia" icon="book-open" >}}
  {{< card link="/fundamentos/digitalizacion-de-audio" title="Digitalización de audio" subtitle="Muestreo, Nyquist-Shannon, aliasing, cuantización" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-de-audio" title="Representación de audio" subtitle="STFT, espectrograma, el sonido como imagen para CNNs" icon="book-open" >}}
  {{< card link="/fundamentos/mfcc-y-escala-mel" title="MFCC y escala Mel" subtitle="Escala perceptual, banco de filtros, coeficientes cepstrales" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-35" title="Clase 35 - Teoría" subtitle="Introducción al análisis de audio: Fourier, digitalización, STFT, MFCC" icon="academic-cap" >}}
  {{< card link="/clases/clase-35/practica" title="Práctica de clase" subtitle="DFT, STFT y MFCC en triple framework" icon="code" >}}
  {{< card link="/dominios/audio" title="Dominio: Audio" subtitle="Línea de tiempo del análisis de audio y sus hitos" icon="globe-alt" >}}
  {{< card link="/laboratorios/lab-34" title="Lab 34 - Razonamiento (anterior)" subtitle="Tool use, LoRA y optimización de prompt en LLMs" icon="arrow-left" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Recorrido celda a celda de las 63 celdas + las 3 actividades resueltas (1.1 señal cuadrada no centrada con código, 1.2 la componente DC, 2.1 el principio de incertidumbre de Gabor). Notebook ejecutado con las animaciones de Fourier y los espectrogramas STFT/MFCC embebidos. Abre el módulo de Audio. Sin papers ni fundamentos nuevos (todos de la clase 35).
