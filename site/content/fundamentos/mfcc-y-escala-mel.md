---
title: "MFCC y escala Mel"
weight: 119
math: true
---

Un [espectrograma](/fundamentos/representacion-tiempo-frecuencia) trata todas las frecuencias por igual, pero el **oído humano no lo hace**: distinguimos mucho mejor entre 100 y 200 Hz que entre 10.000 y 10.100 Hz, aunque la distancia en hertz sea la misma. Los **Mel-Frequency Cepstral Coefficients (MFCC)** son el descriptor de audio que incorpora esta percepción humana, y fueron durante tres décadas la representación estándar del reconocimiento de voz. Este fundamento acompaña a la [Clase 35](/clases/clase-35): explica la escala Mel, el pipeline completo de los MFCC y por qué siguen siendo relevantes.

---

## 1. La percepción humana no es lineal

Todo el análisis anterior —Fourier, STFT— trata el eje de frecuencias de forma **lineal y uniforme**, sin considerar cómo percibimos realmente el sonido. Pero nuestro oído tiene un rango limitado (20–20.000 Hz) y, dentro de él, **no procesa todas las frecuencias de la misma forma**: la resolución tonal es **fina en frecuencias bajas** y **gruesa en frecuencias altas**. La pregunta natural: *¿existe una relación lineal entre la escala de frecuencia y nuestra percepción?* La respuesta es no —y la escala Mel lo cuantifica.

---

## 2. La escala Mel

La **escala Mel** (Stevens, Volkmann & Newman, 1937) es una escala perceptual que cuantifica las diferencias tonales según cómo las percibe un humano. Su punto de referencia: **1000 Hz = 1000 mel**. La conversión de frecuencia (Hz) a mel es logarítmica:

$$
m = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right).
$$

La forma logarítmica captura la percepción: a frecuencias bajas, un cambio pequeño en Hz produce un cambio grande en mel (percibimos bien la diferencia); a frecuencias altas, hace falta un cambio grande en Hz para el mismo cambio en mel (percibimos peor). La escala Mel es, en esencia, una **regla de medir el tono calibrada al oído humano**.

{{< concept-alert type="clave" >}}
La escala Mel es el puente entre la **física** de la señal (frecuencias en Hz) y la **percepción** (cómo suenan al oído). Representar el audio en esta escala le da relevancia a las características que **importan para el ser humano** —justo lo que se necesita para tareas como reconocer palabras o voces.
{{< /concept-alert >}}

---

## 3. El pipeline MFCC

Los MFCC (Davis & Mermelstein, 1980) se calculan encadenando varias operaciones, todas motivadas por lo anterior:

$$
\text{Audio} \to \text{Frames} \to \text{FFT} \to \text{Filtros Mel} \to \log(\cdot) \to \text{DCT} \to \text{MFCC}.
$$

1. **Framing.** Se corta el audio en tramas cortas solapadas (típicamente 20–40 ms), dentro de las cuales la señal es aproximadamente estacionaria.
2. **FFT.** Se calcula el espectro de cada trama (magnitud) con la [FFT](/fundamentos/analisis-de-fourier) —es la [STFT](/fundamentos/representacion-tiempo-frecuencia).
3. **Banco de filtros Mel.** Se aplica un conjunto de **filtros triangulares** espaciados según la escala Mel (juntos en bajas frecuencias, separados en altas). Cada filtro suma la energía del espectro en su banda: $\text{coef}_k = \sum_h w_{h,k}\, x_h$. Esto **comprime** el espectro de cientos de bins a unas pocas decenas de bandas perceptuales.
4. **Logaritmo.** Se toma el $\log$ de las energías de cada banda —modelando que también la **intensidad** se percibe de forma logarítmica (decibeles).
5. **DCT (Transformada Coseno Discreta).** Se aplica la DCT a las log-energías. Esto **decorrelaciona** las bandas (que están correlacionadas entre sí) y concentra la información en pocos coeficientes; se suelen conservar los primeros ~13. El resultado son los **coeficientes cepstrales**.

El término "cepstral" (anagrama de "spectral") viene de aplicar una transformada al **logaritmo del espectro**: es el "espectro del espectro", que separa la envolvente espectral (el timbre, la forma del tracto vocal) de la estructura fina (el tono).

---

## 4. El resultado: una matriz tiempo × coeficientes

El resultado final de la codificación MFCC es una **matriz 2D** que representa la **evolución temporal de los coeficientes**: cada columna es una trama en el tiempo, cada fila un coeficiente cepstral. Esta matriz compacta —mucho menor que el espectrograma completo— fue la entrada estándar de los sistemas de reconocimiento de voz (HMM-GMM) durante décadas, y sigue siendo un descriptor de audio muy usado como *feature* de entrada de modelos.

{{< concept-alert type="dato" >}}
Los MFCC dominaron el reconocimiento de voz desde 1980 hasta la llegada del deep learning end-to-end (~2015), cuando modelos como wav2vec empezaron a aprender representaciones directamente del audio crudo o del espectrograma. Aun así, los MFCC siguen vigentes en aplicaciones con pocos datos o recursos limitados, por su eficiencia y su fundamento perceptual. Ver el [dominio de audio](/dominios/audio).
{{< /concept-alert >}}

---

## 5. Relevancia para salud y señales biomédicas

Los MFCC son la base de un campo creciente: los **biomarcadores de voz**. Como comprimen el audio en características perceptualmente relevantes, sirven para detectar patologías a partir de la voz: disfonías y nódulos vocales, signos tempranos de Parkinson (alteraciones en la prosodia y el temblor vocal), depresión (cambios en el ritmo y la energía del habla) o afecciones respiratorias a partir de la tos. También se aplican al análisis de **sonidos cardíacos y pulmonares** (auscultación digital). Su atractivo clínico es doble: son **compactos** (funcionan con datasets médicos pequeños, donde un modelo de audio crudo sobreajustaría) y **interpretables en términos perceptuales**, lo que facilita la validación con expertos humanos.

---

## Referencias

- Stevens, S., Volkmann, J. & Newman, E. (1937). *A Scale for the Measurement of the Psychological Magnitude Pitch*. J. Acoust. Soc. Am. — la escala Mel.
- Davis, S. & Mermelstein, P. (1980). *Comparison of Parametric Representations for Monosyllabic Word Recognition*. IEEE TASSP. — los MFCC. [análisis](/papers/mfcc-davis-mermelstein-1980)
- Fundamentos relacionados: [Análisis de Fourier](/fundamentos/analisis-de-fourier) · [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia) · [Digitalización de audio](/fundamentos/digitalizacion-de-audio).
