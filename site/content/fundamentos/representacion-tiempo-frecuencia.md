---
title: "Representación tiempo-frecuencia (STFT)"
weight: 118
math: true
---

El [análisis de Fourier](/fundamentos/analisis-de-fourier) responde *qué* frecuencias componen una señal, pero a un precio: **pierde toda la información de cuándo ocurren**. Para una canción o una frase hablada —donde el orden de las frecuencias en el tiempo *es* la información— eso es inaceptable. La **representación tiempo-frecuencia** resuelve el problema aplicando Fourier sobre **ventanas locales** de la señal, produciendo el objeto central del análisis de audio moderno: el **espectrograma**. Este fundamento acompaña a la [Clase 35](/clases/clase-35) y desarrolla la **Short-Time Fourier Transform (STFT)** y el compromiso fundamental que la gobierna.

---

## 1. La limitación de Fourier

La Transformada de Fourier de una señal completa entrega su espectro global: la lista de frecuencias presentes en **toda** la duración, sin decir en qué momento aparece cada una. Matemáticamente, la información temporal se "integra" y desaparece:

$$
X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-2\pi i f t}\, dt.
$$

Dos señales muy distintas —un do seguido de un mi, versus un mi seguido de un do— tienen **el mismo espectro de Fourier**. Para extraer información de alto nivel necesitamos una herramienta que entregue **tiempo y frecuencia simultáneamente**.

---

## 2. La Short-Time Fourier Transform

La idea de la **STFT** es simple y poderosa: en vez de transformar la señal entera, **aplicar la Transformada de Fourier sobre secciones locales** de la señal. Se desliza una **ventana** temporal a lo largo de la señal y, en cada posición, se calcula el espectro de lo que queda dentro:

$$
\text{STFT}\{x\}(\tau, f) = \int_{-\infty}^{\infty} x(t)\, w(t - \tau)\, e^{-2\pi i f t}\, dt,
$$

donde $w(t-\tau)$ es una **ventana** centrada en el tiempo $\tau$. El resultado es una función de **dos** variables —tiempo $\tau$ y frecuencia $f$— que se visualiza como un **espectrograma**: un mapa 2D donde el eje horizontal es el tiempo, el vertical la frecuencia, y el color la magnitud de cada componente. Dos parámetros la definen: el **tamaño de la ventana** (cuántas muestras abarca cada FFT) y el **hop** (cuánto se desplaza la ventana entre pasos consecutivos).

{{< concept-alert type="clave" >}}
La STFT convierte una señal 1D (amplitud vs tiempo) en una imagen 2D (frecuencia vs tiempo). Esta representación es la que permite aplicar **redes convolucionales** al audio —tratando el espectrograma como una imagen— y es la entrada estándar de casi todos los modelos de audio hasta hoy.
{{< /concept-alert >}}

---

## 3. El compromiso tiempo-frecuencia

El tamaño de la ventana esconde un **trade-off inevitable**:

- **Ventana ancha** → abarca mucha señal en cada FFT. **Aumenta la certeza en frecuencia** (más muestras dan mejor resolución espectral) pero **disminuye la certeza en tiempo** (no se sabe en qué instante *dentro* de la ventana ocurrió cada frecuencia).
- **Ventana angosta** → abarca poca señal. **Aumenta la certeza en tiempo** (se localiza bien el instante) pero **disminuye la certeza en frecuencia** (pocas muestras dan un espectro borroso).

No se puede tener ambas a la vez. Este no es un defecto del método sino una **ley fundamental** de las señales.

{{< concept-alert type="recordar" >}}
**Principio de incertidumbre de Gabor.** Ninguna señal puede estar arbitrariamente localizada en tiempo *y* en frecuencia a la vez: el producto de las dispersiones cumple $\Delta t \cdot \Delta f \ge \text{constante}$. Es el análogo, para señales, del principio de incertidumbre de Heisenberg. Dennis Gabor (1946) lo formalizó y mostró que las **ventanas gaussianas** (moduladas por sinusoides) alcanzan el mínimo de incertidumbre —los "átomos de Gabor". Ver el [análisis del paper](/papers/time-frequency-gabor-1946).
{{< /concept-alert >}}

Elegir el tamaño de ventana es, por tanto, elegir **dónde poner la resolución**: ventanas largas para analizar tonos sostenidos (música), ventanas cortas para eventos transitorios (percusión, consonantes explosivas del habla).

---

## 4. Del espectrograma a las representaciones perceptuales

El espectrograma lineal trata todas las frecuencias por igual, pero el oído humano **no**. El siguiente paso —comprimir el eje de frecuencias según la percepción humana (la [escala Mel y los MFCC](/fundamentos/mfcc-y-escala-mel))— parte siempre de la STFT: primero se calcula el espectrograma, y luego se le aplica el banco de filtros Mel. La STFT es, así, la operación base sobre la que se construyen casi todas las representaciones de audio.

---

## 5. Relevancia para salud y señales biomédicas

Muchas señales biomédicas son **no estacionarias**: su contenido en frecuencia cambia con el tiempo, que es justamente donde Fourier puro falla y la STFT brilla. El análisis tiempo-frecuencia de un **EEG** revela cómo evolucionan las bandas cerebrales durante una crisis epiléptica; el de un **fonocardiograma** localiza en el tiempo los componentes de un soplo cardíaco; el de la **voz** captura la dinámica de los formantes en una patología del habla. En todos, el trade-off de Gabor es una decisión clínica real: una ventana mal elegida puede difuminar un evento transitorio breve (un artefacto epileptiforme) o confundir dos frecuencias cercanas. La representación tiempo-frecuencia correcta puede ser la diferencia entre ver o no ver un patrón diagnóstico.

---

## Referencias

- Gabor, D. (1946). *Theory of Communication*. J. IEE. — el principio de incertidumbre para señales. [análisis](/papers/time-frequency-gabor-1946)
- Fundamentos relacionados: [Análisis de Fourier](/fundamentos/analisis-de-fourier) · [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel) · [Digitalización de audio](/fundamentos/digitalizacion-de-audio).
