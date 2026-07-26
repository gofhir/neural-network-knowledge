---
title: "Análisis de Fourier"
weight: 116
math: true
---

El **análisis de Fourier** es la herramienta matemática que permite responder una pregunta aparentemente simple: *¿qué frecuencias componen una señal?* Su idea central —que **cualquier señal se puede descomponer en una suma de sinusoides puras**— es uno de los resultados más fértiles de la matemática aplicada, y el cimiento de todo el procesamiento de audio, imágenes y telecomunicaciones. Este fundamento acompaña a la [Clase 35](/clases/clase-35), la introducción al análisis de audio: recorre la señal y sus parámetros, la serie de Fourier, la transformada, el espectro, y el algoritmo (la **FFT**) que lo hace práctico.

---

## 1. Señal, amplitud, frecuencia

Una **señal** es una medición u observación que describe un fenómeno a través del tiempo. Para una señal **periódica**, tres parámetros la caracterizan:

- **Amplitud** $A$ — la magnitud de la oscilación (presión en Pa, voltaje en V, etc.). En sonido, la amplitud es la **intensidad o volumen**.
- **Período** $T$ — el tiempo que tarda un ciclo completo, en segundos.
- **Frecuencia** $f = 1/T$ — cuántos ciclos por segundo, en hertz (Hz). En sonido, la frecuencia es el **tono**.

La señal más simple es la **sinusoide**, $x(t) = A\sin(2\pi f t + \phi)$: una señal de **frecuencia pura**, porque representa una única frecuencia $f$. El oído humano percibe frecuencias entre **20 Hz y 20.000 Hz**, y la cóclea es capaz de discriminar entre las distintas frecuencias que componen un sonido —una habilidad fundamental, porque nos permite **codificar información de alto nivel** a partir de ellas (reconocer una voz, una melodía, una palabra).

{{< concept-alert type="clave" >}}
Una sinusoide es una frecuencia pura. La pregunta que abre todo el campo es: **¿cómo determinamos qué frecuencias están presentes en una señal compleja** —como una canción, que es una mezcla de muchas? La respuesta es el análisis de Fourier: *descomponerla en su suma de sinusoides*.
{{< /concept-alert >}}

---

## 2. La serie de Fourier

La **serie de Fourier** descompone una señal **periódica** $f(t)$ en una suma (potencialmente infinita) de sinusoides, dentro de un intervalo $[t_0, t_0+T]$:

$$
\hat f(t) = a_0 + \sum_{k=1}^{\infty} \big(a_k \cos(k\omega_0 t) + b_k \sin(k\omega_0 t)\big), \qquad \omega_0 = \frac{2\pi}{T}.
$$

Los **coeficientes** miden "cuánto" de cada frecuencia $k\omega_0$ hay en la señal, y se calculan por integración (proyección de la señal sobre cada sinusoide):

$$
a_0 = \frac{1}{T}\int_{t_0}^{t_0+T} f(t)\, dt, \quad
a_k = \frac{2}{T}\int_{t_0}^{t_0+T} f(t)\cos(k\omega_0 t)\, dt, \quad
b_k = \frac{2}{T}\int_{t_0}^{t_0+T} f(t)\sin(k\omega_0 t)\, dt.
$$

El término $a_0$ es el **valor medio** (componente DC); si a una señal le sumamos una constante, solo cambia $a_0$, no los demás coeficientes.

### 2.1 El ejemplo de la señal cuadrada

El ejemplo canónico de la clase: la serie de Fourier de una **señal cuadrada** impar tiene $a_0 = 0$, $a_k = 0$, y solo armónicos impares:

$$
b_k = \frac{4}{k\pi} \;\; (k = 1, 3, 5, \dots) \quad\Longrightarrow\quad \hat f(t) = \sum_{k=1,3,5,\dots}^{\infty} \frac{4}{k\pi}\sin(k\omega_0 t).
$$

Sumar cada vez más armónicos aproxima mejor la forma cuadrada —una imagen que el [laboratorio de la clase](/laboratorios/lab-35) anima para señales cuadrada, triangular y diente de sierra.

---

## 3. El espectro: fundamental y armónicos

El **espectro** de una señal es su representación en el **espacio de las frecuencias**: describe la magnitud (y fase) de cada componente frecuencial. El análisis de Fourier, en esencia, **transforma la señal desde el dominio del tiempo al dominio de la frecuencia**.

Para una señal periódica, el espectro es una colección de componentes discretas:

- La componente de frecuencia **más baja** se llama **frecuencia fundamental** —determina el tono percibido.
- El resto (múltiplos de la fundamental) son los **armónicos** —determinan el *timbre*, lo que distingue un violín de una flauta tocando la misma nota.

---

## 4. De la serie a la transformada

La serie de Fourier funciona para señales **periódicas**. La **Transformada de Fourier** generaliza la idea a señales **arbitrarias** (no periódicas), convirtiendo la suma discreta de armónicos en una integral sobre un continuo de frecuencias:

$$
X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-2\pi i f t}\, dt.
$$

Para señales **digitales** (muestreadas, ver [digitalización de audio](/fundamentos/digitalizacion-de-audio)), se usa la **Transformada Discreta de Fourier (DFT)**, que opera sobre $N$ muestras:

$$
X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i k n / N}, \qquad k = 0, 1, \dots, N-1.
$$

---

## 5. La FFT: hacer Fourier práctico

Calcular la DFT directamente cuesta $O(N^2)$ operaciones —prohibitivo para señales largas. La **Fast Fourier Transform (FFT)** de Cooley y Tukey (1965) explota la estructura de la DFT con una estrategia de *divide y vencerás* (factorizando $N$) para reducir el costo a $O(N \log N)$. Para $N = 1000$ eso es una mejora de ~100×; para señales de millones de muestras, la diferencia entre lo factible y lo imposible.

{{< concept-alert type="dato" >}}
La FFT es considerada uno de los algoritmos más importantes del siglo XX. Es lo que hace `np.fft.fft` bajo el capó, y sin ella no existirían el audio digital, la compresión de imágenes (JPEG usa una transformada relacionada, la DCT), ni las telecomunicaciones modernas. Ver el [paper de Cooley-Tukey](/papers/fft-cooley-tukey-1965).
{{< /concept-alert >}}

---

## 6. La limitación que abre el siguiente capítulo

El análisis de Fourier tiene un defecto profundo: al transformar al dominio de la frecuencia, **toda la información temporal desaparece**. Sabemos *qué* frecuencias hay, pero no *cuándo* ocurren. Para una señal estacionaria (un tono constante) da igual; pero para una canción o una frase hablada —donde el orden de las frecuencias *es* la información— es un problema grave. La solución (aplicar Fourier sobre ventanas locales de la señal) es la **[representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia)**, el siguiente peldaño del análisis de audio.

---

## 7. Relevancia para salud y señales biomédicas

El análisis de Fourier es ubicuo en el procesamiento de señales clínicas. El análisis espectral de un **electrocardiograma (ECG)** revela componentes de frecuencia asociadas a arritmias; el de un **electroencefalograma (EEG)** separa las bandas (delta, theta, alpha, beta, gamma) que caracterizan estados cerebrales; y el de señales de **audio clínico** (sonidos cardíacos, respiratorios, o la voz) es la base de biomarcadores diagnósticos. En todos estos casos, la FFT es la primera operación del pipeline —y la limitación temporal del Fourier puro es justamente lo que motiva usar espectrogramas para señales biomédicas no estacionarias.

---

## Referencias

- Cooley, J. & Tukey, J. (1965). *An Algorithm for the Machine Calculation of Complex Fourier Series*. Math. of Computation. — [análisis](/papers/fft-cooley-tukey-1965)
- Fundamentos relacionados: [Digitalización de audio](/fundamentos/digitalizacion-de-audio) · [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia) · [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel).
