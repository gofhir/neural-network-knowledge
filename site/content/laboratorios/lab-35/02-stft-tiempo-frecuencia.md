---
title: "Tiempo-frecuencia: la STFT"
weight: 2
---

La FFT dice *qué* frecuencias hay, pero no *cuándo*. Como el audio real cambia en el tiempo (una melodía, el habla), necesitamos ver **cómo evolucionan las frecuencias** — para eso está la STFT.

## El problema de la FFT: pierde el *cuándo*

La FFT colapsa toda la duración de una señal en un único espectro. El lab lo demuestra con una señal "escalera": cuatro frecuencias distintas, una en cada cuarto de 8 segundos.

```python
y1 = np.cos( 2*np.pi*10*t[0*dt:1*dt] )   # 10 Hz  (0-2s)
y2 = np.cos( 2*np.pi*25*t[1*dt:2*dt] )   # 25 Hz  (2-4s)
y3 = np.cos( 2*np.pi*50*t[2*dt:3*dt] )   # 50 Hz  (4-6s)
y4 = np.cos( 2*np.pi*100*t[3*dt:4*dt] )  # 100 Hz (6-8s)
y = np.concatenate( (y1, y2, y3, y4) )
```

Una FFT de toda esta señal mostraría cuatro picos (10, 25, 50, 100 Hz) **sin ninguna información de que ocurren en momentos diferentes**. Dos señales con las mismas frecuencias en distinto orden tendrían FFTs casi idénticas. Esa es la limitación.

## La solución: FFT por ventanas

La **STFT (Short-Time Fourier Transform)** hace muchas FFTs sobre ventanas cortas y consecutivas, en vez de una sobre toda la señal:

```python
spec = np.abs( librosa.stft( y ) )
print( spec.shape )   # (1025, 129): 1025 bins de frecuencia × 129 ventanas de tiempo
```

El resultado es una **matriz 2D**: frecuencia × tiempo. Se visualiza como **espectrograma** (X = tiempo, Y = frecuencia, color = energía):

![Espectrograma STFT de la señal escalera: una línea horizontal brillante que salta de 10 a 25 a 50 a 100 Hz cada 2 segundos](/laboratorios/lab-35/stft-sinusoides.png)

La escalera de frecuencias aparece como una línea que **salta hacia arriba cada 2 segundos** — recuperamos el *cuándo* que la FFT perdía.

{{< callout type="info" >}}
**El espectrograma ES la entrada del deep learning de audio.** Esta imagen tiempo×frecuencia es lo que la mayoría de modelos de audio reciben: un espectrograma es una "imagen" del sonido → puedes aplicarle **CNNs** (las mismas del dominio de visión). Whisper, wav2vec2, clasificadores de sonido — casi todos convierten el audio a espectrograma y lo tratan como imagen. Ver [representación de audio](/fundamentos/representacion-de-audio).
{{< /callout >}}

## STFT sobre audio real

Aplicado a un `.wav` real (del dataset UrbanSound8K), con dos detalles importantes:

```python
audio_data, sampling_rate = librosa.load( '100652-3-0-0.wav' )   # remuestrea a 22050 Hz
spec = librosa.stft( audio_data )
spec_db = librosa.amplitude_to_db( np.abs(spec), ref=np.max )     # a DECIBELES
```

![Espectrograma STFT de un audio real en decibeles: patrón complejo de manchas de energía con estructura armónica](/laboratorios/lab-35/stft-audio-real.jpg)

Dos puntos que no son cosméticos:

1. **`librosa.load` remuestrea a 22050 Hz por defecto** (con `resampy`), sin importar la tasa del archivo. La "sampling rate" que ves es la que librosa decidió, no la del `.wav`. Es un uniformizado de resolución, como el resize de imágenes.
2. **La conversión a decibeles es obligatoria para audio real.** La percepción del volumen es logarítmica y el rango dinámico del audio es enorme (un sonido fuerte puede ser 1000× más intenso que uno débil). En escala lineal, los detalles débiles quedan invisibles; en dB ($20\log_{10}$), se hacen visibles. Nota que las sinusoides de juguete se graficaron en lineal, pero el audio real *exige* dB.

Cómo leer el espectrograma: **rayas verticales** = eventos bruscos (golpes, ataques); **líneas horizontales** = tonos sostenidos; **bandas equiespaciadas** = armónicos (timbre).

## Actividad 2.1 — el tamaño de ventana

**Enunciado:** ¿cuál es el efecto sobre la precisión en frecuencia cuando aumentamos el tamaño de la ventana?

**Respuesta:** Al **aumentar** la ventana temporal, **mejora la resolución en frecuencia**: se distinguen frecuencias más cercanas, porque la FFT dispone de más muestras y su resolución es $\Delta f = f_s/N$ (a mayor $N$, menor $\Delta f$). Pero esto **empeora la resolución en tiempo**: al promediar sobre un intervalo mayor, se pierde precisión sobre *cuándo* ocurre cada frecuencia.

Es imposible tener máxima precisión en tiempo y frecuencia simultáneamente — el **principio de incertidumbre de Gabor** (Gabor 1946), análogo al de Heisenberg: $\Delta t \cdot \Delta f \geq \text{constante}$.

> **Ventana grande** → alta precisión en frecuencia, baja en tiempo.
> **Ventana pequeña** → alta precisión en tiempo, baja en frecuencia.

{{< callout type="info" >}}
**La intuición física.** Para saber con precisión qué frecuencia es una nota, necesitas escucharla *un rato* (ventana larga) — un instante infinitesimal no tiene frecuencia definida. Pero mientras la escuchas ese rato, no sabes en qué milisegundo exacto estaba. Precisión en frecuencia y en tiempo se compran una a costa de la otra. Es física, no una limitación de implementación.
{{< /callout >}}
