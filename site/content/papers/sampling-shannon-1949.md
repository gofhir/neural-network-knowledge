---
title: "Teorema de muestreo (Shannon, 1949)"
weight: 393
math: true
---

{{< paper-card
    title="Communication in the Presence of Noise"
    authors="Claude E. Shannon (Bell Labs)"
    year="1949"
    venue="Proceedings of the IRE"
    pdf="/papers/sampling-shannon-1949.pdf" >}}
El compañero de ingeniería de *A Mathematical Theory of Communication* (1948). Enuncia y demuestra el **teorema de muestreo** (Theorem 1): una función cuyo espectro no contiene frecuencias superiores a $W$ ciclos por segundo queda *completamente determinada* por sus muestras tomadas cada $1/2W$ segundos, es decir, a una tasa de $2W$ muestras por segundo —la **frecuencia de Nyquist**. Introduce además la representación geométrica de señales (una señal de banda $W$ y duración $T$ es un punto en $2TW$ dimensiones) y deriva la fórmula de **capacidad de canal** $C = W\log_2(1 + P/N)$. Para el laboratorio de la [Clase 35](/clases/clase-35), lo esencial es lo primero: la respuesta exacta a *¿con qué frecuencia debo muestrear para digitalizar sin perder información?*
{{< /paper-card >}}

---

## Contexto: teoría de la información y precedentes del muestreo

Shannon publica este trabajo en el punto más alto de su producción fundacional. En 1948 había definido la información en términos de **entropía** y medido la incertidumbre de una fuente en **bits**; el paper de 1949 es la contraparte orientada al ingeniero de radio, en el lenguaje de señales continuas, ancho de banda en ciclos por segundo y potencia de ruido térmico. El contexto tecnológico es la **modulación por pulsos codificados (PCM)**, entonces emergente en Bell Labs. Shannon menciona que en PCM la voz "debe ser muestreada, comprimida, cuantizada y codificada" —enumerando en 1949 exactamente las dos operaciones que la Clase 35 separa: **muestreo** (discretizar en el tiempo) y **cuantización** (discretizar en amplitud).

Shannon es escrupuloso con los precedentes. Cita a **Harry Nyquist** (1924, 1928), que señaló la importancia del intervalo $1/2W$ segundos en telegrafía —Shannon lo honra llamándolo **intervalo de Nyquist**, por lo que la tasa $2W$ es la *Nyquist rate*—; a **J. M. Whittaker** (1935), que desarrolló la interpolación cardinal por funciones $\operatorname{sinc}$; y a Gabor (1946) y Bennett (1941), que ya apuntaban que $\sim 2TW$ números bastan para especificar una señal de banda $W$ y duración $T$. La contribución de Shannon no fue inventar la fórmula sino enunciarla con **precisión, generalidad y demostración** dentro de la teoría de la comunicación, y mostrar que $2TW$ no es una aproximación sino el número *exacto* de grados de libertad.

## Contribución central

La idea, de enorme consecuencia práctica: **una señal analógica de banda limitada no contiene información "infinita"; contiene exactamente $2W$ números independientes por segundo, y muestrear a esa tasa los captura todos sin pérdida.** Esto convierte el problema de digitalizar una señal continua en un problema con receta exacta:

1. Determina la máxima frecuencia $W$ presente (o fíjala con un filtro pasa-bajos).
2. Muestrea a una tasa $f_s = 2W$ muestras por segundo (o mayor).
3. Reconstruye perfectamente interpolando las muestras con pulsos $\operatorname{sinc}$.

Si muestreas *por debajo* de $2W$, aparece el **aliasing**: las frecuencias altas que el muestreo no puede resolver se "pliegan" sobre el rango bajo y se disfrazan de frecuencias que la señal nunca tuvo, corrompiendo irreversiblemente la reconstrucción.

## El teorema de muestreo

**Enunciado (Theorem 1).** Si $f(t)$ no contiene frecuencias mayores que $W$ cps, queda completamente determinada dando sus ordenadas en una serie de puntos espaciados $1/2W$ segundos. La justificación intuitiva de Shannon es memorable: si $f(t)$ no tiene frecuencias mayores que $W$, no puede cambiar a un valor sustancialmente nuevo en menos de medio ciclo de la frecuencia más alta ($1/2W$); no tiene la agilidad para "esconder" nada entre muestra y muestra.

**Demostración.** Sea $F(\omega)$ el espectro de $f(t)$, nulo fuera de la banda, de modo que

$$f(t) = \frac{1}{2\pi}\int_{-2\pi W}^{+2\pi W} F(\omega)\, e^{i\omega t}\, d\omega.$$

Al evaluar $f$ en los instantes $t = n/2W$, la integral resulta ser —salvo constantes— el $n$-ésimo coeficiente de la serie de Fourier de $F(\omega)$ sobre $[-W, W]$. Entonces: las muestras determinan los coeficientes de Fourier de $F$; estos determinan $F(\omega)$; y $F$ determina $f(t)$. Por transitividad, **las muestras determinan $f(t)$ completamente**, y de forma única: no hay dos señales de banda limitada distintas que compartan el mismo muestreo.

**Reconstrucción por interpolación sinc.** La señal se reconstruye colocando en cada muestra un pulso $\dfrac{\sin 2\pi W t}{2\pi W t}$, la función **sinc**, que vale 1 en $t=0$ y exactamente 0 en los demás puntos de muestreo, y cuyo espectro es constante dentro de $W$ y cero fuera. Con $x_n$ la $n$-ésima muestra:

$$f(t) = \sum_{n=-\infty}^{\infty} x_n\, \frac{\sin \pi(2Wt - n)}{\pi(2Wt - n)}.$$

Es la **fórmula de interpolación cardinal** (Whittaker-Shannon), el trabajo idealizado que hace el filtro de reconstrucción del conversor digital-analógico (DAC).

**Aliasing.** Muestrear a tasa $f_s$ replica el espectro periódicamente cada $f_s$ hertz. Si $f_s \geq 2W$, las copias no se tocan y un pasa-bajos ideal recupera la copia central. Si $f_s < 2W$, las copias adyacentes **se solapan**: una componente de frecuencia $f$ por encima de $f_s/2$ (la *folding frequency*) no desaparece, sino que reaparece disfrazada de $f_{\text{alias}} = |f_s - f|$. El daño es **irreversible**. La defensa es el **filtro anti-aliasing**: un pasa-bajos analógico aplicado *antes* de muestrear, para forzar que la señal cumpla la hipótesis del teorema.

## El resto del paper como contexto: geometría y capacidad de canal

El teorema de muestreo es la **herramienta de conteo** que habilita el resto. Si una señal de banda $W$ y duración $T$ está determinada por $2TW$ muestras, esas muestras son las **coordenadas de un punto en $2TW$ dimensiones**; magnitudes físicas se vuelven distancias ($d^2 = 2WTP$), y la teoría del ruido se vuelve geometría de esferas. Con ese andamiaje, la deducción de la capacidad se convierte en empaquetamiento de esferas y Shannon llega al Theorem 2:

$$C = W \log_2\!\left(1 + \frac{P}{N}\right)\ \text{bits por segundo.}$$

Es la **fórmula de Shannon-Hartley**: el límite máximo de transmisión confiable de un canal de ancho de banda $W$ y relación señal-a-ruido $P/N$, alcanzable con frecuencia de errores arbitrariamente pequeña. Para la Clase 35 es telón de fondo: muestra por qué Shannon necesitaba el teorema de muestreo —era lo que le permitía tratar señales continuas como vectores finitos.

## Limitaciones

- **Banda limitada ideal.** Ninguna señal real de duración finita es exactamente de banda limitada (una señal acotada en el tiempo tiene espectro infinito). En la práctica se fuerza la condición con un filtro anti-aliasing real, de banda de transición finita —por eso se deja margen (44.1 kHz en vez de 40).
- **Muestreo uniforme.** La reconstrucción sinc supone muestras equiespaciadas; el muestreo no uniforme funciona en principio, pero es más frágil y complicado.
- **Interpolador sinc no causal e infinito.** El sinc ideal se extiende de $-\infty$ a $+\infty$ y no es realizable en tiempo real; los DAC reales usan aproximaciones causales de soporte finito.
- **Amplitud continua, no cuantizada.** El teorema trata solo la discretización en el *tiempo* y supone precisión infinita por muestra. La cuantización es un problema distinto, con su propio ruido —la otra mitad de la Clase 35.

## Por qué importa para la Clase 35

La [Clase 35](/clases/clase-35) descompone la [digitalización de audio](/fundamentos/digitalizacion-de-audio) en dos operaciones independientes, y este paper ilumina la frontera entre ellas:

| Operación | Qué discretiza | Pregunta clave | Respuesta |
|---|---|---|---|
| **Muestreo** | El **tiempo** | ¿Con qué frecuencia muestrear? | Nyquist-Shannon: $2W$ |
| **Cuantización** | La **amplitud** | ¿Cuántos bits por muestra? | Compromiso ingenieril (fuera de este paper) |

La lógica de **por qué el CD usa 44.1 kHz** encapsula toda la clase en un número: la banda audible llega hasta $W \approx 20$ kHz; el teorema exige $f_s \geq 2W = 40$ kHz para evitar aliasing; se elige $44.1$ kHz $> 40$ kHz para dar banda de transición al filtro anti-aliasing real. Por debajo de 40 kHz, las frecuencias agudas se plegarían y aparecerían tonos espurios audibles en el rango grave. Otras tasas obedecen la misma lógica: 48 kHz en video profesional, 8 kHz en telefonía (~3.4 kHz de voz).

Las tres ideas a internalizar: (1) **una señal de banda limitada tiene información finita** —$2W$ números por segundo, digitalizable sin pérdida; (2) **la frecuencia de Nyquist $2W$ es un umbral duro**, no un continuo de calidad —muestrear menos corrompe por aliasing; (3) **muestreo y cuantización son problemas distintos**. La misma disciplina rige el registro de señales biomédicas —ECG a 250-1000 Hz, EEG a 256-512 Hz— donde muestrear mal *fabrica* artefactos indistinguibles de fenómenos fisiológicos reales.
