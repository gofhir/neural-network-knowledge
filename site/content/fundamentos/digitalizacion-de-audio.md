---
title: "Digitalización de audio"
weight: 117
math: true
---

El sonido es una señal **continua**: una onda de presión que varía suavemente en el tiempo. Pero un computador solo puede almacenar y procesar **números discretos**. La **digitalización** es el puente entre ambos mundos —el proceso que convierte una onda sonora continua en una secuencia finita de números manejables. Se realiza en dos pasos independientes: **cuantización** (discretizar la amplitud) y **muestreo** (discretizar el tiempo). Este fundamento acompaña a la [Clase 35](/clases/clase-35) y explica, además, el teorema que gobierna cuántas muestras necesitamos: el de **Nyquist-Shannon**.

---

## 1. Dos ejes, dos discretizaciones

Una señal analógica varía de forma continua en **dos** ejes: el eje de **amplitud** (el valor de la señal en cada instante) y el eje del **tiempo**. Digitalizar exige discretizar ambos:

- **Cuantización** — discretiza la **amplitud**: restringe el valor continuo de la señal a un conjunto finito de niveles.
- **Muestreo** (*sampling*) — discretiza el **tiempo**: toma el valor de la señal solo en instantes espaciados.

Son operaciones **ortogonales**: una actúa sobre el "cuánto" y la otra sobre el "cuándo".

---

## 2. Cuantización: discretizar la amplitud

La **cuantización** define un conjunto de **niveles** discretos y mapea cada valor de la señal original al nivel más cercano. Con 2 bits hay $2^2 = 4$ niveles; con 3 bits, $2^3 = 8$; en general, con $b$ bits hay $2^b$ niveles.

La señal cuantizada **no representa fielmente** a la original —introduce un **error de cuantización** (la diferencia entre el valor real y el nivel asignado). La forma de mejorar la fidelidad es **agregar más niveles** (más bits): el audio de un CD usa **16 bits** (65.536 niveles), y el audio profesional 24 bits.

{{< concept-alert type="clave" >}}
¿Por qué no usar **infinitos** niveles y recuperar la señal exacta? Porque cada nivel cuesta bits de almacenamiento: más niveles = más precisión pero más datos. La cuantización es, en el fondo, un **compromiso entre fidelidad y tamaño** —la misma tensión que reaparece en toda representación digital.
{{< /concept-alert >}}

El número de bits por muestra se llama **profundidad de bits** (*bit depth*) y determina el **rango dinámico** de la señal (la diferencia entre el sonido más suave y el más fuerte representables).

---

## 3. Muestreo: discretizar el tiempo

Una vez cuantizada, la señal sigue siendo **continua en el tiempo**. El **muestreo** la convierte a discreta tomando muestras a intervalos regulares. La **frecuencia de muestreo** (*sampling rate*, $f_s$) es cuántas muestras se toman por segundo, en Hz. El audio de un CD usa $f_s = 44.100$ Hz (44,1 kHz); el habla telefónica, 8 kHz; el audio profesional, 48 kHz.

La pregunta crítica: **¿con qué frecuencia debo muestrear para no perder información?** Muestrear poco ahorra datos, pero puede distorsionar la señal irreversiblemente. La respuesta es uno de los teoremas más importantes de la ingeniería.

---

## 4. El teorema de muestreo de Nyquist-Shannon

{{< concept-alert type="recordar" >}}
**Teorema de muestreo (Nyquist-Shannon).** Una señal cuyo contenido en frecuencia está limitado a un máximo de $W$ Hz (banda limitada) queda **completamente determinada** por sus muestras si se toman a una tasa de **al menos $2W$ muestras por segundo**. La frecuencia $2W$ se llama **tasa de Nyquist**.
{{< /concept-alert >}}

La consecuencia es directa: como el oído humano percibe hasta ~20 kHz, para capturar todo el audio audible hace falta muestrear a al menos $2 \times 20\,\text{kHz} = 40$ kHz. Los 44,1 kHz del CD dejan un pequeño margen por encima de ese mínimo. Si la señal de banda limitada se muestrea a la tasa de Nyquist o superior, se puede **reconstruir exactamente** la señal continua a partir de las muestras (por interpolación con funciones sinc).

### 4.1 Aliasing: el precio de muestrear poco

Si se muestrea **por debajo** de la tasa de Nyquist, ocurre el **aliasing**: las frecuencias altas (por encima de $f_s/2$, la **frecuencia de Nyquist**) se "pliegan" y aparecen disfrazadas como frecuencias bajas, corrompiendo la señal de forma **irreversible**. Es el mismo efecto que hace que las ruedas de un carro parezcan girar hacia atrás en una película (el frame rate es un muestreo temporal insuficiente). Por eso, antes de muestrear, se aplica un **filtro anti-aliasing** que elimina las frecuencias por encima de $f_s/2$.

---

## 5. Relevancia para salud y señales biomédicas

La digitalización correcta es crítica en instrumentación médica. Un **ECG** tiene contenido relevante hasta ~150 Hz, por lo que se muestrea típicamente a 250-500 Hz para evitar aliasing de artefactos. Un **EEG** exige tasas similares o mayores según las bandas de interés. El **audio clínico** (auscultación digital de sonidos cardíacos o pulmonares) hereda las mismas reglas: elegir mal el *sampling rate* puede hacer que un soplo de alta frecuencia se pliegue y se confunda con un componente de baja frecuencia, con consecuencias diagnósticas. Y la profundidad de bits determina si se captura el rango dinámico completo entre un sonido cardíaco fuerte y un soplo tenue. El teorema de Nyquist-Shannon no es una curiosidad teórica: es una restricción de diseño en cualquier dispositivo que digitalice una señal fisiológica.

---

## Referencias

- Shannon, C. E. (1949). *Communication in the Presence of Noise*. Proc. IRE. — contiene el teorema de muestreo. [análisis](/papers/sampling-shannon-1949)
- Nyquist, H. (1928). *Certain Topics in Telegraph Transmission Theory*. — el precedente.
- Fundamentos relacionados: [Análisis de Fourier](/fundamentos/analisis-de-fourier) · [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia).
