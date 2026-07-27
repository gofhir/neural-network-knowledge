---
title: "Análisis en frecuencia: FFT y Fourier"
weight: 1
---

El audio es una señal continua en el tiempo, pero lo que importa perceptualmente son sus **frecuencias**. Esta primera parte cubre el salto del dominio del tiempo al de la frecuencia — la base de todo el análisis de audio.

## La señal de prueba y el teorema de muestreo

El lab arranca con la señal más simple posible: una sinusoide pura de 5 Hz, muestreada a 512 Hz.

```python
f = 5.0                                   # frecuencia [Hz]
sampling_rate = 512                       # muestras por segundo
t = np.linspace( 0, 1, sampling_rate )
y = np.sin( 2*np.pi*f*t )                 # sinusoide canónica
```

La elección de `sampling_rate = 512` no es arbitraria: obedece al **teorema de muestreo de Nyquist-Shannon** (Shannon 1949, uno de los papers de la clase). Para representar una señal sin distorsión, hay que muestrear a **más del doble** de su frecuencia más alta. La mitad de la tasa de muestreo es la **frecuencia de Nyquist**:

$$f_{\text{Nyquist}} = \frac{512}{2} = 256 \text{ Hz}$$

{{< callout type="info" >}}
**Por qué el audio se muestrea a 44.1 kHz.** El oído humano oye hasta ~20 kHz; para capturarlo sin *aliasing* hay que muestrear a >40 kHz. El estándar de CD (44.1 kHz) sale exactamente de aquí. La `sampling_rate` define qué frecuencias tu señal *puede siquiera representar* — no es un detalle. Ver [digitalización de audio](/fundamentos/digitalizacion-de-audio).
{{< /callout >}}

## La FFT: del tiempo a la frecuencia

La **Transformada de Fourier** responde: *¿de qué frecuencias está compuesta esta señal?* La **FFT** (Fast Fourier Transform) es el algoritmo eficiente para calcularla — el de Cooley-Tukey 1965, que reduce el costo de $O(n^2)$ a $O(n \log n)$.

```python
spectrum = np.fft.fft( y )                                    # salida COMPLEJA
freq_bins = np.fft.fftfreq( spectrum.size, d = 1/sampling_rate )  # eje de frecuencias
plt.plot( freq_bins, spectrum.real )
```

![Espectro de la sinusoide: dos picos simétricos en +5 Hz y -5 Hz](/laboratorios/lab-35/fft-espectro-5hz.png)

Tres puntos clave:

- **La salida es de números complejos.** Cada valor codifica *dos* cosas: la **magnitud** (`np.abs`) = cuánta de esa frecuencia hay, y la **fase** (`np.angle`) = en qué desfase temporal está. Lo correcto para el "espectro de potencia" es graficar `np.abs(spectrum)` (el lab usa `.real` como atajo).
- **Frecuencias positivas y negativas.** Para una señal real el espectro es simétrico: la energía en +5 Hz se refleja en −5 Hz. `fftfreq` genera el eje correcto para el orden (raro) del output.
- **El resultado valida la herramienta**: la sinusoide de 5 Hz produce un pico exactamente en 5 Hz. En el tiempo tenías que *contar ciclos*; en frecuencia, "5 Hz" es un pico explícito. Ver [análisis de Fourier](/fundamentos/analisis-de-fourier).

## La Serie de Fourier: por qué funciona

Si la FFT *analiza* (señal → frecuencias), la Serie de Fourier *sintetiza* (frecuencias → señal). El teorema: cualquier señal periódica es una **suma de sinusoides**:

$$\hat{f}(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\tfrac{2\pi}{T} n t\right) + b_n \sin\left(\tfrac{2\pi}{T} n t\right) \right]$$

- **$a_0/2$** = el término constante (componente **DC** = valor promedio de la señal).
- **$a_n, b_n$** = pesos de cada armónico.

El lab construye tres señales sumando armónicos uno a uno (con animaciones), y sus coeficientes revelan una regla profunda: **la forma de la señal determina qué armónicos tiene y cómo decaen**.

| Señal | DC ($a_0$) | Armónicos | Decaimiento | Suavidad |
|---|---|---|---|---|
| **Cuadrada** | 0 | solo impares | $1/n$ | áspera (fenómeno de Gibbs) |
| **Triangular** | 0 | solo impares | $1/n^2$ | suave (converge rápido) |
| **Diente de sierra** | 1 | **todos** | $1/n$ | áspera, con offset |

{{< callout type="info" >}}
**Brusquedad ↔ agudos.** Cuanto más brusca una señal (saltos, esquinas), más lento decaen sus armónicos y más rica en altas frecuencias es. La cuadrada (salto vertical) decae como $1/n$; la triangular (cambio de pendiente) como $1/n^2$. En audio: los sonidos percusivos y las consonantes tienen espectros anchos por su brusquedad; los tonos puros, espectros concentrados.
{{< /callout >}}

## Actividad 1.1 — señal cuadrada no centrada

**Enunciado:** serie de Fourier de una señal cuadrada impar, mínimo 0 y máximo 2 (no centrada), con animación.

**Clave:** es la cuadrada centrada **desplazada +1**. El offset solo cambia $a_0$; los $b_n$ quedan idénticos. Basta **inicializar `y` en el offset** en vez de en ceros (igual que la diente de sierra hacía `y += 1/2.0`):

```python
y = np.zeros( t.shape )
y += 1.0                                    # término DC: a0/2 = 1 (señal entre 0 y 2)
for i in range( 1, 200, 2 ):                # armónicos impares (igual que la cuadrada)
  y += (4/np.pi) * (1/float(i)) * np.sin( i*2*np.pi*f*t )
  history.append( copy.copy( y ) )
  spectrum = np.abs( np.fft.fft( y ) )
  history_freq.append( copy.copy( spectrum.real ) )
```

La única diferencia con el ejemplo es la línea `y += 1.0`. El espectro resultante es idéntico al de la cuadrada centrada **salvo un pico nuevo en 0 Hz** (el offset).

## Actividad 1.2 — la diferencia de espectros

**Respuesta:** La diferencia está exclusivamente en la **componente de frecuencia cero (DC)**. La señal centrada en cero tiene promedio 0 → $a_0 = 0$ → sin componente en 0 Hz. La señal de la actividad 1.1, desplazada, tiene promedio 1 → $a_0 = 2$ → un **pico adicional en 0 Hz**. El resto del espectro (armónicos impares 5, 15, 25... Hz) es idéntico, porque un offset solo afecta $a_0$, no los $b_n$.

{{< callout type="info" >}}
**La componente DC es el valor promedio.** Restar la media para "centrar en cero" es lo mismo que anular el bin de 0 Hz — el mismo preprocesamiento que en PCA y normalización de features, visto desde el lado frecuencial. Un offset DC en una grabación (sesgo del micrófono) se elimina con un filtro pasa-altos.
{{< /callout >}}
