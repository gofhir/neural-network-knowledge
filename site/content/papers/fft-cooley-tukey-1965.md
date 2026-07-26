---
title: "FFT: Fast Fourier Transform (1965)"
weight: 392
math: true
---

{{< paper-card
    title="An Algorithm for the Machine Calculation of Complex Fourier Series"
    authors="James W. Cooley, John W. Tukey (IBM, Bell Labs)"
    year="1965"
    venue="Mathematics of Computation"
    pdf="/papers/fft-cooley-tukey-1965.pdf" >}}
Probablemente el artículo de cuatro páginas más consecuente de la computación científica del siglo XX. Su objeto es engañosamente modesto: calcular la serie de Fourier compleja $X(j) = \sum_{k=0}^{N-1} A(k)\, W^{jk}$, con $W = e^{2\pi i/N}$. El cálculo directo cuesta $O(N^2)$ operaciones; Cooley y Tukey exhiben un algoritmo de **divide y vencerás** que produce el mismo resultado en **menos de $2N\log_2 N$ operaciones**, sin almacenamiento auxiliar. Ese salto de $N^2$ a $N\log N$ convirtió el análisis de Fourier de una elegancia teórica prohibitiva en la infraestructura invisible del procesamiento de señales moderno. Es, literalmente, el motor bajo `np.fft.fft` que usa el laboratorio de la [Clase 35](/clases/clase-35).
{{< /paper-card >}}

---

## Contexto: Fourier existía, pero calcularlo era caro

La teoría estaba lista desde hacía siglo y medio. Joseph Fourier había mostrado (1807, publicado en 1822) que una función periódica se descompone en una suma de senos y cosenos de frecuencias múltiplas: toda señal es superposición de sinusoides. El problema nunca fue la teoría sino la **aritmética**. Evaluar la transformada discreta de Fourier (DFT) por su definición exige, para cada uno de los $N$ valores de salida, una suma de $N$ productos: del orden de $N^2$ multiplicaciones complejas. Para $N = 1000$ —una ventana de audio corta— eso es un millón de operaciones; para las señales sísmicas o de imágenes de los años sesenta, con $N$ de decenas o cientos de miles, el costo $N^2$ volvía el cálculo directamente imposible.

Hay un detalle histórico que el paper no menciona: **Gauss** había descubierto esencialmente el mismo algoritmo hacia 1805 —antes de la publicación de Fourier— interpolando órbitas de asteroides, pero su trabajo quedó enterrado en notación no estándar y sin influencia. La factorización tenía precedentes dispersos (Yates, Good 1958, Gauss), pero fue esta nota de 1965 la que llegó en el momento justo: había computadoras que podían ejecutarla y problemas que la exigían con urgencia. Tukey esbozó la idea en el comité asesor científico del presidente Kennedy, donde se discutía detectar ensayos nucleares soviéticos por análisis espectral de datos sismográficos; Cooley, en IBM, la implementó.

## Contribución: factorizar la matriz densa en matrices dispersas

La contribución no es un teorema nuevo sobre Fourier sino un **algoritmo** que explota la estructura del problema para no repetir trabajo. La matriz de la DFT —cuyas entradas son las potencias $W^{jk}$— es enormemente redundante, porque las potencias de una raíz de la unidad se repiten cíclicamente. Esa redundancia permite **factorizar la matriz densa en un producto de matrices dispersas**, cada una aplicable con muchas menos operaciones.

Si $N = r_1 r_2$, la maniobra central es **reindexar** entrada y salida en un sistema mixto de dos dígitos ($j = j_1 r_1 + j_0$, $k = k_1 r_2 + k_0$). Al sustituir, el exponente $W^{jk}$ se descompone y la suma doble se **anida**: primero sobre un subíndice, luego sobre el otro, pasando por un arreglo intermedio $A_1$. El arreglo $A_1$ tiene $N$ elementos y cada uno cuesta $r_1$ operaciones; obtener $X$ de $A_1$ cuesta $N r_2$. El total es

$$T = N(r_1 + r_2) \quad \text{en vez de} \quad N^2 = N \cdot r_1 r_2.$$

Ya con una sola factorización el costo pasó de un producto $r_1 r_2$ a una suma $r_1 + r_2$. Los factores que "pegan" los subproblemas —potencias de $W$ que corrigen la fase de cada subtransformada— son los célebres **twiddle factors** ("factores de giro").

## El algoritmo: recursión, mariposas y complejidad $O(N\log N)$

El truco de dos pasos se aplica **recursivamente**. Si $N = r_1 r_2 \cdots r_m$, el algoritmo de $m$ pasos requiere $T = N(r_1 + \cdots + r_m)$ operaciones. Conviene usar tantos factores pequeños como sea posible; con todos los factores iguales a $r$ se tiene $m = \log_r N$ pasos y

$$T(r) = r N \log_r N.$$

Ahí aparece explícitamente la ley $N\log N$; para el caso binario $N = 2^m$, la cota es de **menos de $2N\log_2 N$ operaciones**. El paper tabula la eficiencia relativa de distintas bases comparando $r/\log_2 r$: el óptimo es $r = 3$ (1.88), pero $r = 2$ y $r = 4$ (ambas 2.00) ganan apenas ~6% menos y ofrecen ventajas en aritmética binaria. Añade una observación de ingeniería clave: casi nunca hace falta que $N$ sea exactamente una potencia de 2, porque siempre existe un $N$ "altamente compuesto" a pocos por ciento de cualquier número grande.

Para $r = 2$ cada índice se expresa en **binario** y el algoritmo se reescribe como $m$ etapas donde cada operación involucra **solo dos posiciones de memoria** —la célebre operación **"mariposa" (butterfly)**: sumar y restar dos valores, uno multiplicado por un twiddle factor. Como cada mariposa lee y escribe el mismo par de celdas, el cálculo cabe en el arreglo original (*in-place*, sin memoria extra) y admite **cómputo en paralelo**. El precio conocido: el resultado sale permutado en **orden con los bits invertidos** (*bit reversal*), un reordenamiento barato $O(N)$.

## Resultados e impacto

El paper reporta tiempos reales en un **IBM 7094**: para $2^{11}$ puntos, ~0.02 minutos; para $2^{13}$ (8192 puntos), 0.10-0.13 minutos. Al duplicar $N$, el tiempo apenas más que se duplica —comportamiento $N\log N$— en vez de cuadruplicarse. Aterrizando la magnitud del ahorro: para $N = 1000$, el método directo cuesta $\sim 10^6$ operaciones y la FFT $\sim 10^4$, un factor de **cien veces**; para $N = 10^6$, la aceleración es de **cinco órdenes de magnitud** ($10^{12}$ frente a $\sim 2\times10^7$).

La FFT figura de manera recurrente entre los algoritmos más influyentes del siglo XX. Aparece en audio y voz (espectrogramas, STFT, MFCC, MP3), imágenes (la DCT de JPEG es prima cercana), telecomunicaciones (la modulación OFDM de Wi-Fi, LTE y 5G usa IFFT/FFT) y cómputo numérico general (multiplicación rápida de polinomios y enteros, métodos espectrales, convolución rápida). `numpy.fft.fft` y `scipy.fft` son descendientes directos, típicamente vía FFTW.

## Limitaciones

- **Preferencia por $N$ altamente compuesto.** El algoritmo brilla cuando $N$ se factoriza en primos pequeños, idealmente $N = 2^m$. Si $N$ es primo grande, la factorización no ayuda y se cae hacia $O(N^2)$; la práctica habitual es *zero-padding* hasta la siguiente potencia de 2. Bluestein y Rader resolverían después el caso primo.
- **Reordenamiento por inversión de bits.** El resultado sale permutado y requiere un paso de bit-reversal; barato, pero una complicación real que la fórmula esconde.
- **Cota que cuenta operaciones, no arquitectura.** En hardware real el rendimiento depende de acceso a memoria, caché y paralelismo —por eso FFTW *sintoniza* la factorización a la máquina, algo ajeno al análisis de 1965.
- **Presentación densa.** Escrito para especialistas; la derivación por bits es difícil de seguir sin los diagramas de mariposa que la pedagogía posterior tuvo que añadir.

## Por qué importa para la Clase 35

La [Clase 35](/clases/clase-35) recorre una cadena precisa: **naturaleza del sonido → [análisis de Fourier](/fundamentos/analisis-de-fourier) → la FFT → digitalización → STFT → MFCC**. Este paper es el eslabón que hace todo lo demás posible en la práctica. Una señal de audio digital es una secuencia **discreta y finita** de muestras, así que lo que se calcula no es la transformada continua sino la **DFT**, la misma $X(j) = \sum_k A(k) W^{jk}$ del paper. Calcularla de forma ingenua costaría $O(N^2)$; con la FFT cuesta $O(N\log N)$, y por eso se pueden transformar ventanas de audio en tiempo real.

- La **STFT** aplica una FFT a ventanas sucesivas y solapadas para ver cómo evoluciona el espectro en el tiempo: cada columna del **espectrograma** es una FFT.
- Los **MFCC** parten del espectro de magnitud que entrega la FFT, lo pasan por un banco de filtros mel, toman el logaritmo y aplican una transformada coseno. El primer paso de toda la tubería es la FFT.
- El **laboratorio usa `np.fft.fft`** directamente: cada llamada ejecuta, envuelta en NumPy, la recursión de 1965.

En una frase: Fourier dio la teoría (1807-1822), Cooley y Tukey dieron el algoritmo que la volvió práctica (1965), y `np.fft.fft` es su encarnación en el laboratorio de la Clase 35. La misma reducción de $O(N^2)$ a $O(N\log N)$ sostiene el análisis espectral de señales biomédicas —ECG, EEG, sonidos respiratorios— que sin la FFT sería computacionalmente prohibitivo.
