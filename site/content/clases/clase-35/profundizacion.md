---
title: "Profundización - Análisis de Audio"
weight: 20
math: true
---

> **Desarrollo formal de la Clase 35.** La [teoría](/clases/clase-35/teoria) recorre la cadena del análisis de audio de forma narrativa; aquí se formaliza. Cinco partes: (1) la serie de Fourier y la ortogonalidad; (2) la transformada, la DFT y la complejidad de la FFT; (3) el teorema de muestreo y el aliasing; (4) la STFT y la incertidumbre de Gabor; (5) el pipeline MFCC.

---

## 1. Serie de Fourier y ortogonalidad

La serie de Fourier funciona porque las sinusoides $\{\cos(k\omega_0 t), \sin(k\omega_0 t)\}$ forman una **base ortogonal** sobre un período $T = 2\pi/\omega_0$. La ortogonalidad significa que la integral del producto de dos sinusoides distintas se anula:

$$
\int_{t_0}^{t_0+T} \cos(k\omega_0 t)\cos(l\omega_0 t)\,dt = \begin{cases} T/2 & k = l \neq 0 \\ 0 & k \neq l \end{cases}
$$

y análogamente para senos, con $\int \cos(k\omega_0 t)\sin(l\omega_0 t)\,dt = 0$ para todo $k, l$. Por eso los coeficientes se obtienen **proyectando** la señal sobre cada elemento de la base (multiplicar por la sinusoide e integrar):

$$
a_k = \frac{2}{T}\int_{t_0}^{t_0+T} f(t)\cos(k\omega_0 t)\,dt, \qquad b_k = \frac{2}{T}\int_{t_0}^{t_0+T} f(t)\sin(k\omega_0 t)\,dt.
$$

Es exactamente la misma lógica que la descomposición de un vector en una base ortonormal —solo que en un espacio de funciones de dimensión infinita.

### 1.1 Forma compleja

Usando la identidad de Euler $e^{i\theta} = \cos\theta + i\sin\theta$, la serie se escribe de forma compacta con coeficientes complejos:

$$
f(t) = \sum_{k=-\infty}^{\infty} c_k\, e^{ik\omega_0 t}, \qquad c_k = \frac{1}{T}\int_{t_0}^{t_0+T} f(t)\, e^{-ik\omega_0 t}\,dt.
$$

Esta forma es la que generaliza directamente a la transformada de Fourier.

---

## 2. Transformada de Fourier, DFT y FFT

Para señales **no periódicas**, el período se lleva al infinito ($T \to \infty$) y la suma discreta de armónicos se convierte en una **integral** sobre un continuo de frecuencias:

$$
X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-2\pi i f t}\,dt \qquad\text{(transformada directa)},
$$

con su inversa $x(t) = \int X(f)\,e^{2\pi i f t}\,df$. Para señales **digitales** de $N$ muestras, la **Transformada Discreta de Fourier (DFT)**:

$$
X_k = \sum_{n=0}^{N-1} x_n\, e^{-2\pi i k n / N}, \qquad k = 0,\dots,N-1.
$$

### 2.1 La complejidad y la FFT

Calcular la DFT directamente es un producto matriz-vector: $N$ salidas, cada una una suma de $N$ términos $\Rightarrow O(N^2)$. La **FFT** (Cooley-Tukey) factoriza $N = N_1 N_2$ y reordena la suma como DFTs anidadas más pequeñas. Aplicando la idea recursivamente (radix-2, $N = 2^m$) se obtiene:

$$
T(N) = O(N \log N).
$$

Para $N = 10^6$, esto es $\sim 20$ millones de operaciones en vez de $10^{12}$ —cinco órdenes de magnitud. Sin la FFT, el análisis espectral en tiempo real sería imposible. → [análisis](/papers/fft-cooley-tukey-1965)

---

## 3. Teorema de muestreo y aliasing

Al muestrear una señal continua $x(t)$ a intervalos $T_s = 1/f_s$, obtenemos $x[n] = x(nT_s)$. En el dominio de la frecuencia, el muestreo **replica** el espectro de la señal cada $f_s$ Hz:

$$
X_s(f) = f_s \sum_{k=-\infty}^{\infty} X(f - k f_s).
$$

Si el espectro original está confinado a $|f| \le W$ (banda limitada) y $f_s > 2W$, las réplicas **no se solapan** y la señal se puede recuperar exactamente filtrando con un pasa-bajos ideal. La reconstrucción es una interpolación con funciones **sinc**:

$$
x(t) = \sum_{n=-\infty}^{\infty} x[n]\,\operatorname{sinc}\!\left(\frac{t - nT_s}{T_s}\right), \qquad \operatorname{sinc}(u) = \frac{\sin(\pi u)}{\pi u}.
$$

{{< concept-alert type="advertencia" >}}
Si $f_s < 2W$, las réplicas **se solapan**: las frecuencias por encima de $f_s/2$ (la **frecuencia de Nyquist**) se "pliegan" y aparecen disfrazadas de frecuencias bajas. Es el **aliasing**, y es **irreversible** —una vez que dos frecuencias se confundieron en una muestra, no hay forma de separarlas. Por eso se aplica un **filtro anti-aliasing** *antes* de muestrear. → [análisis](/papers/sampling-shannon-1949)
{{< /concept-alert >}}

---

## 4. STFT e incertidumbre de Gabor

### 4.1 La STFT

La STFT aplica la transformada sobre ventanas locales:

$$
\text{STFT}\{x\}(\tau, f) = \int_{-\infty}^{\infty} x(t)\, w(t-\tau)\, e^{-2\pi i f t}\,dt,
$$

y su magnitud al cuadrado $|\text{STFT}(\tau,f)|^2$ es el **espectrograma**. En versión discreta, se calcula una FFT por cada trama de la señal, desplazando la ventana un **hop** entre tramas.

### 4.2 El principio de incertidumbre

Definiendo las dispersiones temporal $\Delta t$ y frecuencial $\Delta f$ de una señal (como desviaciones estándar de $|x(t)|^2$ y $|X(f)|^2$), Gabor (1946) demostró que su producto está **acotado inferiormente**:

$$
\Delta t \cdot \Delta f \ge \frac{1}{2} \quad (\text{con la convención de Gabor}).
$$

No existe ninguna señal perfectamente localizada en tiempo *y* frecuencia. La cota se **alcanza** (igualdad) con las **señales elementales gaussianas** —una sinusoide modulada por una envolvente gaussiana, los "átomos de Gabor":

$$
g(t) = e^{-\alpha^2(t-t_0)^2}\, e^{2\pi i f_0 t}.
$$

Estos átomos son óptimos: concentran la máxima información posible en una celda del plano tiempo-frecuencia (un "logón"). Elegir el ancho de la ventana STFT es elegir la forma de esa celda —más estrecha en tiempo o más estrecha en frecuencia, nunca ambas. → [análisis](/papers/time-frequency-gabor-1946)

---

## 5. El pipeline MFCC, paso a paso

Los MFCC transforman una trama de audio en un vector compacto de coeficientes perceptuales:

1. **Espectro de potencia.** Tras enventanar (típicamente Hamming) una trama de $N$ muestras, se calcula $|X_k|^2$ con la FFT.

2. **Banco de filtros Mel.** Se definen $M$ filtros triangulares $H_m(k)$ espaciados uniformemente en la **escala Mel**. La conversión Hz→mel:
$$
m(f) = 2595\,\log_{10}\!\left(1 + \frac{f}{700}\right).
$$
La energía de cada banda es la suma ponderada del espectro:
$$
S_m = \sum_{k} H_m(k)\, |X_k|^2, \qquad m = 1, \dots, M.
$$
Como los filtros son estrechos en bajas frecuencias y anchos en altas, esto **comprime** el eje frecuencial imitando la resolución del oído.

3. **Compresión logarítmica.** $\log S_m$ —modela la percepción logarítmica de la intensidad (los decibeles).

4. **DCT (Transformada Coseno Discreta).** Decorrelaciona las log-energías (que están muy correlacionadas entre bandas vecinas) y compacta la información en los primeros coeficientes:
$$
c_n = \sum_{m=1}^{M} \log(S_m)\,\cos\!\left(\frac{\pi n (m - \tfrac12)}{M}\right), \qquad n = 0, 1, \dots, L-1.
$$
Se suelen conservar los primeros $L \approx 13$ coeficientes $c_n$: son los **MFCC** de esa trama.

Repitiendo por cada trama se obtiene la **matriz MFCC** (coeficientes × tiempo). El nombre "cepstral" (anagrama de *spectral*) refleja que se aplica una transformada al **logaritmo del espectro** —el "espectro del espectro"—, lo que separa la **envolvente espectral** (el timbre, la forma del tracto vocal) de la **estructura fina** (el tono). → [análisis](/papers/mfcc-davis-mermelstein-1980)

---

## 6. Síntesis

La cadena completa, en una línea de operadores: una señal continua $x(t)$ se **muestrea** (Nyquist) a $x[n]$, se **transforma** (FFT) a su espectro, se **enventana** (STFT, con el trade-off de Gabor) para recuperar el tiempo, y se **perceptualiza** (filtros Mel + log + DCT) en los MFCC. Cada paso es una elección de representación que descarta lo irrelevante para la tarea y conserva lo esencial —el mismo principio de abstracción que gobierna todo el aprendizaje de representaciones.

---

**Ver también:** [Clase 35 - Teoría](/clases/clase-35/teoria) · [Clase 35 - Práctica](/clases/clase-35/practica) · Fundamentos: [Análisis de Fourier](/fundamentos/analisis-de-fourier) · [Digitalización de audio](/fundamentos/digitalizacion-de-audio) · [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia) · [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel).
