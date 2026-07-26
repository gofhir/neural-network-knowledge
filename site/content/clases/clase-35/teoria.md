---
title: "Teoría - Introducción al Análisis de Audio"
weight: 10
math: true
---

> **Recorrido de la Clase 35** del Diplomado IA UC (Gabriel Sepúlveda), primera de la serie de **Audio y Video**. Es una clase de **fundamentos de procesamiento de señales**: antes de aplicar deep learning al audio, hay que entender cómo se representa matemáticamente el sonido. La clase construye, paso a paso, toda la cadena clásica: de la **naturaleza física del sonido** al **análisis de Fourier** (cómo descomponer una señal en frecuencias), la **digitalización** (cuantización y muestreo), el **análisis tiempo-frecuencia** (la STFT y el espectrograma) y, finalmente, los **MFCC** (una representación calibrada al oído humano). Es la base sobre la que se apoyará todo el resto del módulo de audio.

---

## 1. La naturaleza del sonido

El sonido es una **onda mecánica** que resulta de la vibración de las partículas del medio en el que se propaga. Su ciclo de vida tiene tres etapas: **generación** (una fuente vibra), **transmisión** (la onda de presión viaja por el medio) y **recepción** (llega a un receptor).

En el oído humano, la **cóclea** es capaz de discriminar entre las distintas frecuencias que componen un sonido, percibiendo entre **20 Hz y 20.000 Hz**. Esto es fundamental porque la frecuencia está directamente relacionada con el **tono**, y la habilidad de distinguir frecuencias nos permite **codificar información de alto nivel** —reconocer una palabra, una voz, una melodía.

{{< concept-alert type="clave" >}}
El experimento que motiva la clase: *¿es posible reconocer una canción reproducida con un solo tono/frecuencia?* La respuesta es no —la información está en **cómo se combinan y suceden las frecuencias**. Todo el análisis de audio consiste en extraer esa estructura frecuencial.
{{< /concept-alert >}}

---

## 2. Análisis de señal en tiempo continuo

Una **señal** es una medición que describe un fenómeno a través del tiempo. Para señales periódicas: **amplitud** $A$ (intensidad/volumen), **período** $T$ y **frecuencia** $f = 1/T$ (tono). La señal más simple es la **sinusoide**, una señal de **frecuencia pura** (una sola frecuencia).

La pregunta central: *¿cómo determinamos las frecuencias presentes en una señal compleja, como una canción?* → el **Análisis de Fourier**, que descompone una señal arbitraria en una **suma de sinusoides puras**, transformándola del **dominio del tiempo** al **dominio de la frecuencia**. Se desarrolla en el fundamento [Análisis de Fourier](/fundamentos/analisis-de-fourier).

### 2.1 La serie de Fourier

Para una señal **periódica** $f(t)$ en $[t_0, t_0+T]$:

$$
\hat f(t) = a_0 + \sum_{k=1}^{\infty}\big(a_k\cos(k\omega_0 t) + b_k\sin(k\omega_0 t)\big), \qquad \omega_0 = \frac{2\pi}{T},
$$

con coeficientes calculados por integración: $a_0 = \frac{1}{T}\int f\,dt$, $a_k = \frac{2}{T}\int f\cos(k\omega_0 t)\,dt$, $b_k = \frac{2}{T}\int f\sin(k\omega_0 t)\,dt$.

El ejemplo de la clase es la **señal cuadrada**, cuya serie tiene $a_0 = a_k = 0$ y solo armónicos impares $b_k = \frac{4}{k\pi}$ ($k=1,3,5,\dots$). Sumar un valor constante a la señal solo modifica $a_0$.

### 2.2 El espectro

El **espectro** representa la magnitud y fase de las componentes en frecuencia de una señal. La componente más lenta es la **frecuencia fundamental** (el tono); las más rápidas son los **armónicos** (el timbre). El análisis de Fourier práctico se hace con la **FFT** (Cooley-Tukey, 1965), que reduce el costo de $O(N^2)$ a $O(N\log N)$. → [paper](/papers/fft-cooley-tukey-1965)

---

## 3. Análisis de señal en tiempo discreto

Para procesar una señal en un computador hay que **digitalizarla** —convertir la señal continua en números discretos. Se hace en dos pasos (fundamento [Digitalización de audio](/fundamentos/digitalizacion-de-audio)):

- **Cuantización.** Discretiza la **amplitud**: restringe los valores continuos a niveles discretos. Con $b$ bits hay $2^b$ niveles; más niveles = más fidelidad pero más datos. (¿Por qué no infinitos niveles? Porque cada nivel cuesta almacenamiento.)
- **Muestreo (sampling).** Discretiza el **tiempo**: toma muestras a una **frecuencia de muestreo** ($f_s$). ¿Con qué frecuencia? Lo responde el **teorema de Nyquist-Shannon**: una señal de banda limitada a $W$ Hz requiere muestrear a $\ge 2W$. Por eso el audio de un CD usa 44,1 kHz (~2×20 kHz). Muestrear menos produce **aliasing**. → [paper](/papers/sampling-shannon-1949)

---

## 4. Análisis tiempo-frecuencia

### 4.1 La limitación de Fourier

El análisis de Fourier dice **qué** frecuencias hay, pero **pierde toda la información temporal**: no sabemos *cuándo* ocurre cada una. Para extraer información de alto nivel necesitamos conocer el **orden** de las frecuencias. Se necesita una herramienta que dé tiempo y frecuencia **simultáneamente**.

### 4.2 La Short-Time Fourier Transform (STFT)

La idea: **aplicar la Transformada de Fourier sobre secciones locales** de la señal. Se desliza una **ventana** temporal y se calcula el espectro en cada posición, obteniendo una representación en el **plano tiempo-frecuencia** (el espectrograma). Se desarrolla en [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia).

{{< concept-alert type="advertencia" >}}
El **trade-off de la ventana**: una ventana **ancha** aumenta la certeza en frecuencia pero **disminuye la certeza en tiempo**; una ventana **angosta** aumenta la certeza en tiempo pero la disminuye en frecuencia. No se puede tener ambas: es el **principio de incertidumbre de Gabor** ($\Delta t \cdot \Delta f \ge$ constante), el análogo del principio de Heisenberg para señales. → [paper](/papers/time-frequency-gabor-1946)
{{< /concept-alert >}}

---

## 5. Mel-Frequency Cepstral Coefficients (MFCC)

Hasta aquí, el eje de frecuencias se trató de forma **lineal**, sin considerar la **percepción humana**. Pero el oído no procesa todas las frecuencias por igual: tiene resolución fina en bajas y gruesa en altas.

### 5.1 La escala Mel

La **escala Mel** (Stevens, Volkmann & Newman, 1937) cuantifica las diferencias tonales según la percepción humana, con referencia 1000 Hz = 1000 mel:

$$
m = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right).
$$

### 5.2 El pipeline MFCC

Los **MFCC** (Davis & Mermelstein, 1980) son un descriptor de audio basado en la escala Mel que da relevancia a las características importantes para el ser humano. Su pipeline: → [paper](/papers/mfcc-davis-mermelstein-1980)

$$
\text{Audio} \to \text{Frames} \to \text{FFT} \to \text{Filtros Mel} \to \log(\cdot) \to \text{DCT} \to \text{MFCC}.
$$

La escala Mel se aplica mediante **bancos de filtros triangulares** (cada coeficiente $\text{coef}_k = \sum_h w_{h,k}\, x_h$), luego el logaritmo (percepción logarítmica de la intensidad) y la **DCT** para decorrelacionar y compactar. El resultado es una **matriz 2D** (coeficientes × tiempo) que representa la evolución temporal del audio. Se desarrolla en [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel).

---

## 6. Cierre

La clase construyó toda la cadena clásica del análisis de audio: **sonido → señal → Fourier/FFT → digitalización → STFT/espectrograma → MFCC**. Cada eslabón resuelve una limitación del anterior: Fourier descompone en frecuencias pero pierde el tiempo; la STFT recupera el tiempo a costa del trade-off de Gabor; los MFCC agregan la percepción humana. Esta base es la que permite, en las clases siguientes, aplicar deep learning al audio —tratando el espectrograma como una imagen o alimentando los MFCC a un modelo. El [laboratorio](/laboratorios/lab-35) implementa toda esta cadena en Python (FFT, series de Fourier animadas, STFT y MFCC sobre un archivo de audio real).

---

**Ver también:** [Clase 35 - Profundización](/clases/clase-35/profundizacion) · [Clase 35 - Práctica](/clases/clase-35/practica) · Fundamentos: [Análisis de Fourier](/fundamentos/analisis-de-fourier) · [Digitalización de audio](/fundamentos/digitalizacion-de-audio) · [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia) · [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel) · [Dominio: Audio](/dominios/audio).
