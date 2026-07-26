# An Algorithm for the Machine Calculation of Complex Fourier Series (FFT, Cooley-Tukey) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *An Algorithm for the Machine Calculation of Complex Fourier Series*.
- **Autores:** James W. Cooley (IBM Watson Research Center, Yorktown Heights) y John W. Tukey (Bell Telephone Laboratories, Murray Hill / Princeton University).
- **Venue:** *Mathematics of Computation*, volumen 19, 1965, pp. 297-301.
- **Recepción:** el manuscrito fue recibido el 17 de agosto de 1964. Investigación realizada en parte en Princeton University bajo el patrocinio del Army Research Office (Durham). Los autores agradecen a Richard Garwin por su "rol esencial en la comunicación y el estímulo".
- **Linaje:** el propio paper se declara heredero de una tradición previa. Menciona el método de Yates para calcular las interacciones de un experimento factorial $2^m$, su generalización a $3^m$ por Box et al., y sobre todo el trabajo de **I. J. Good** (1958), quien había dado algoritmos elegantes para factorizar la multiplicación matriz-vector. Cooley y Tukey aplican esas ideas específicamente al cálculo de series de Fourier complejas y las presentan "de una forma bastante distinta".

Este es, probablemente, el artículo de cuatro páginas más consecuente de la computación científica del siglo XX. Su objeto es engañosamente modesto: calcular la **serie de Fourier compleja**

$$X(j) = \sum_{k=0}^{N-1} A(k)\, W^{jk}, \qquad j = 0, 1, \dots, N-1,$$

donde los coeficientes $A(k)$ son complejos y $W$ es la raíz $N$-ésima principal de la unidad, $W = e^{2\pi i/N}$. El paper observa que un cálculo directo con esta fórmula requeriría $N^2$ "operaciones" —donde una operación significa, a lo largo de toda la nota, **una multiplicación compleja seguida de una suma compleja**— y presenta en cambio un algoritmo que produce el mismo resultado en **menos de $2N\log_2 N$ operaciones**, sin necesitar más almacenamiento que el arreglo original $A$.

Ese salto de $N^2$ a $N\log N$ es lo que convirtió el análisis de Fourier de una herramienta teórica elegante pero computacionalmente prohibitiva en la infraestructura invisible del procesamiento de señales moderno. Para la **Clase 35 (Introducción al Análisis de Audio)** este paper importa porque es, literalmente, el motor que hay debajo de `np.fft.fft`: cada espectrograma, cada STFT y cada extracción de MFCC del laboratorio se apoya, en última instancia, en la recursión que Cooley y Tukey formalizaron aquí.

## 2. Contexto histórico: Fourier existía, pero calcularlo era caro

La teoría estaba lista desde hacía siglo y medio. Joseph Fourier había mostrado en 1807 —y publicado en su *Théorie analytique de la chaleur* de 1822— que una función periódica puede descomponerse en una suma de senos y cosenos de frecuencias múltiplas. La idea de que una señal arbitraria es la superposición de componentes sinusoidales es el fundamento conceptual de todo el análisis espectral. El problema nunca fue la teoría, sino la **aritmética**.

Evaluar la transformada discreta de Fourier (DFT) por su definición implica, para cada uno de los $N$ valores de salida, una suma de $N$ productos: en total, del orden de $N^2$ multiplicaciones complejas. Con lápiz y papel, o incluso con las primeras computadoras, esto era ruinoso. Para $N = 1000$ puntos —una ventana de audio corta— hablamos de un millón de operaciones; para las señales sísmicas o de imágenes que interesaban en los años sesenta, con $N$ de decenas o cientos de miles, el costo $N^2$ hacía el cálculo directamente imposible en la práctica. El análisis de Fourier era, en palabras de la época, "demasiado caro para usarlo".

Hay un detalle histórico que el paper no menciona pero que la posteridad rescató: **Carl Friedrich Gauss** había descubierto esencialmente el mismo algoritmo de "divide y vencerás" alrededor de 1805 —antes incluso de la publicación de Fourier— mientras interpolaba las órbitas de los asteroides Pallas y Juno. Su trabajo quedó enterrado en notación no estándar en sus obras completas póstumas y no tuvo influencia. La factorización que Cooley y Tukey redescubren y, sobre todo, **popularizan en la era del computador digital**, tenía por tanto precedentes dispersos (Yates, Good, Gauss), pero fue esta nota de 1965 la que llegó en el momento justo: había computadoras que podían ejecutarla y problemas que la exigían con urgencia.

El contexto inmediato de la publicación tiene también su leyenda. Tukey esbozó la idea en una reunión del comité asesor científico del presidente Kennedy, en la que se discutía cómo detectar ensayos nucleares soviéticos mediante análisis espectral de datos sismográficos —una tarea que exigía transformar series de tiempo enormes. Cooley, en IBM, la implementó. La combinación de un problema nacional de primer orden y un algoritmo que lo volvía tratable disparó la difusión casi inmediata del método.

## 3. Contribución central

La contribución de Cooley y Tukey no es un teorema nuevo sobre Fourier, sino un **algoritmo** que explota la estructura del problema para no repetir trabajo. La observación clave es que la matriz de la DFT —la matriz $N\times N$ cuyas entradas son las potencias $W^{jk}$— no es una matriz cualquiera: es enormemente redundante, porque las potencias de una raíz de la unidad se repiten cíclicamente. Esa redundancia permite **factorizar la matriz densa en un producto de matrices dispersas** (sparse), y cada matriz dispersa se aplica con muchas menos operaciones que la matriz completa.

En términos del propio paper: los métodos de Good, en su generalidad, aplican a problemas donde hay que multiplicar un $N$-vector por una matriz $N\times N$ que puede factorizarse en $m$ matrices dispersas, con $m$ proporcional a $\log N$. Esto convierte un procedimiento de $N^2$ operaciones en uno de $N\log N$ operaciones. La nota de 1965 hace tres cosas concretas:

1. **Deriva el algoritmo de factorización** para el caso de la serie de Fourier compleja, de manera autocontenida y constructiva.
2. **Presta atención explícita a la elección de $N$** —qué factorizaciones son más eficientes— y muestra que $N$ altamente compuesto es lo ideal.
3. **Explota el caso binario** $N = 2^m$: muestra cómo obtener ventajas especiales en un computador binario, tanto en el direccionamiento como en la economía de multiplicaciones, y cómo hacer **todo el cálculo dentro del mismo arreglo** de $N$ posiciones de memoria (algoritmo *in-place*, sin almacenamiento auxiliar).

## 4. El algoritmo: factorización, twiddle factors, recursión y complejidad

### 4.1. La idea de divide y vencerás

Supón que $N$ es compuesto, es decir, $N = r_1 r_2$. La maniobra central es **reindexar** tanto la entrada como la salida en un sistema mixto de dos dígitos. Cooley y Tukey escriben los índices como

$$j = j_1 r_1 + j_0, \qquad k = k_1 r_2 + k_0,$$

con $j_0, k_1 \in \{0,\dots,r_1-1\}$ y $j_1, k_0 \in \{0,\dots,r_2-1\}$. Al sustituir esta descomposición en la suma original, el exponente $W^{jk}$ se descompone y la suma doble sobre $(k_0, k_1)$ se puede **anidar**: primero se suma sobre uno de los subíndices y luego sobre el otro. El paso intermedio define un **arreglo nuevo**,

$$A_1(j_0, k_0) = \sum_{k_1} A(k_1, k_0)\, W^{j_0 k_1 r_2},$$

y el resultado final se escribe como una segunda suma sobre ese arreglo intermedio,

$$X(j_1, j_0) = \sum_{k_0} A_1(j_0, k_0)\, W^{(j_1 r_1 + j_0) k_0}.$$

### 4.2. El ahorro y los twiddle factors

Aquí está el corazón del asunto. El arreglo intermedio $A_1$ tiene $N$ elementos, y cada uno requiere $r_1$ operaciones para calcularse: en total $N r_1$ operaciones. Obtener $X$ a partir de $A_1$ cuesta análogamente $N r_2$ operaciones. Por lo tanto, este algoritmo de dos pasos requiere en total

$$T = N(r_1 + r_2)$$

operaciones, en lugar de las $N^2 = N \cdot r_1 r_2$ del cálculo directo. Ya con una sola factorización el costo pasó de un producto $r_1 r_2$ a una suma $r_1 + r_2$.

Los factores $W^{(j_1 r_1 + j_0)k_0}$ que aparecen en el segundo paso —los que "pegan" o recombinan los resultados de los subproblemas— son lo que la literatura posterior bautizó **twiddle factors** ("factores de giro"). Son las rotaciones en el plano complejo, potencias de $W$, que corrigen la fase de cada subtransformada antes de sumarlas. No son un detalle accesorio: son exactamente lo que permite que dos DFT de tamaño pequeño se combinen para formar una DFT de tamaño mayor.

### 4.3. La recursión y la complejidad $O(N\log N)$

El truco de dos pasos se aplica **recursivamente**. Nada obliga a detenerse en $N = r_1 r_2$: si $N = r_1 r_2 \cdots r_m$, cada aplicación sucesiva del procedimiento —empezando por el subproblema— da un algoritmo de $m$ pasos que requiere

$$T = N(r_1 + r_2 + \cdots + r_m)$$

operaciones. El paper razona entonces sobre cuál es la mejor factorización. Nota que si un factor $r_j = s_j t_j$ con $s_j, t_j > 1$, entonces $s_j + t_j \le r_j$ (salvo el caso $s_j = t_j = 2$, en que hay igualdad): conviene, en general, **usar tantos factores pequeños como sea posible**. En el caso extremo en que todos los factores son iguales a $r$, se tiene $m = \log_r N$ pasos y el costo total es

$$T(r) = r N \log_r N.$$

Ahí aparece explícitamente la ley $N\log N$. Para el caso binario $N = 2^m$, la cota que el paper anuncia en su resumen es de **menos de $2N\log_2 N$ operaciones**. Cooley y Tukey incluso tabulan la eficiencia relativa de distintas bases (radix) comparando el cociente $r/\log_2 r$:

| $r$ | $r/\log_2 r$ |
|---|---|
| 2 | 2.00 |
| 3 | 1.88 |
| 4 | 2.00 |
| 5 | 2.15 |
| 8 | 2.67 |
| 10 | 3.01 |

El valor más bajo (más eficiente) es $r = 3$, pero la ganancia sobre $r = 2$ o $r = 4$ es de apenas ~6%. Como las bases 2 y 4 ofrecen "otras ventajas" en aritmética binaria, son las preferidas en la práctica. El paper añade una observación de ingeniería muy útil: usar factores $r$ de hasta 10 incrementa el número de cómputos en no más de un 50%, de modo que "podemos encontrar valores 'altamente compuestos' de $N$ dentro de un pequeño porcentaje de cualquier número grande dado". Es decir, casi nunca hace falta que $N$ sea exactamente una potencia de 2 para obtener casi todo el beneficio.

### 4.4. El caso binario: representación por bits e inversión de bits

Para $r = 2$, Cooley y Tukey expresan cada índice en **binario**: $k = k_{m-1} 2^{m-1} + \cdots + k_1 2 + k_0$, con cada $k_\nu \in \{0,1\}$ los bits de la representación. El algoritmo se reescribe como una secuencia de $m$ etapas, y en cada etapa

$$A_l(\dots) = A_{l-1}(j_0,\dots,0,\dots) + (-1)^{j_{l-1}}\, A_{l-1}(j_0,\dots,1,\dots)\, W^{(\dots)}$$

involucra **solo dos posiciones de memoria** a la vez: las que difieren en el bit de la posición $2^{m-l}$. Esta es la célebre operación **"mariposa" (butterfly)**: sumar y restar dos valores, uno de ellos multiplicado por un twiddle factor. Como cada mariposa lee y escribe en el mismo par de celdas, el cálculo entero cabe en el arreglo original —de ahí el algoritmo *in-place* sin almacenamiento extra. El paper también nota que la estructura **permite cómputo en paralelo**, pues la operación puede ejecutarse simultáneamente para todos los valores de los índices no involucrados; de hecho menciona que R. E. Miller y S. Winograd, en IBM, diseñaron un circuito de procesamiento múltiple basado en este algoritmo, donde $r = 4$ resultó lo más práctico.

Un precio conocido de esta organización: el arreglo final $A_m$ entrega los $X(j)$ en un **orden con los bits invertidos** (*bit reversal*). El paper lo dice claramente: el índice de cada $X$ debe tener sus bits binarios puestos en orden inverso para hallar su posición en el arreglo. Es un reordenamiento barato ($O(N)$) que las implementaciones modernas resuelven trivialmente, y que en ciertas aplicaciones —cuando la transformada se evalúa dos veces, como al resolver la ecuación en diferencias que el paper usa de ejemplo— puede incluso evitarse.

### 4.5. Evidencia empírica de la época

El paper reporta tiempos reales de un programa en un **IBM 7094** que calculaba sumas de Fourier tridimensionales. Para arreglos de $2^{11}$ puntos el tiempo fue de ~0.02 minutos; para $2^{12}$, entre 0.04 y 0.07 minutos; para $2^{13}$ (8192 puntos), entre 0.10 y 0.13 minutos. Lo notable es la escala del crecimiento: al duplicar $N$, el tiempo apenas más que se duplica —comportamiento $N\log N$—, no se cuadruplica como lo haría un método $N^2$.

### 4.6. La magnitud del ahorro

Vale la pena aterrizar el número. Para $N = 1000$, el método directo cuesta del orden de $N^2 = 10^6$ operaciones. La FFT cuesta del orden de $N\log_2 N \approx 1000 \times 10 \approx 10^4$ operaciones. El cociente es de aproximadamente **cien veces**; y si se compara contra la cota $2N\log_2 N \approx 2\times10^4$, sigue siendo un factor de ~50. Para señales mayores el contraste explota: con $N = 10^6$, $N^2 = 10^{12}$ frente a $N\log_2 N \approx 2\times10^7$, una aceleración de **cinco órdenes de magnitud**. Ese es, en una línea, el motivo por el que el análisis espectral dejó de ser un lujo teórico y se volvió una operación cotidiana.

## 5. Impacto

Es difícil exagerar la influencia de estas cuatro páginas. La FFT figura de manera recurrente en las listas de "los algoritmos más importantes del siglo XX" —la revista *Computing in Science and Engineering* la incluyó entre los diez algoritmos con mayor influencia en el desarrollo y la práctica de la ciencia y la ingeniería. La razón es que la transformada de Fourier aparece en prácticamente toda disciplina que manipule señales, y la FFT es lo que la hace calculable:

- **Audio y voz.** Espectrogramas, STFT, MFCC, ecualizadores, compresión perceptual (MP3, AAC), reconocimiento de voz. Todo el laboratorio de la Clase 35 vive aquí.
- **Imágenes.** Filtrado en el dominio de la frecuencia, compresión (la DCT de JPEG es prima cercana de la FFT), reconstrucción tomográfica.
- **Telecomunicaciones.** La modulación **OFDM** —base de Wi-Fi, LTE, 5G y la TV digital— multiplexa cientos de subportadoras mediante una IFFT en el transmisor y una FFT en el receptor. Sin FFT no habría comunicación inalámbrica moderna de banda ancha.
- **Cómputo numérico general.** Multiplicación rápida de polinomios y de enteros grandes, resolución de ecuaciones en derivadas parciales (métodos espectrales), convolución rápida.

En el ecosistema que usa el curso, `numpy.fft.fft` y `scipy.fft` son descendientes directos de este algoritmo —típicamente vía la biblioteca **FFTW** o implementaciones equivalentes, que generalizan la idea de Cooley-Tukey a factorizaciones mixtas y arbitrarias. Cada vez que el estudiante llama `np.fft.fft(señal)`, está ejecutando la recursión de 1965.

## 6. Limitaciones

- **Preferencia por $N$ altamente compuesto.** El algoritmo brilla cuando $N$ se factoriza en primos pequeños, idealmente $N = 2^m$. Si $N$ es primo grande, la factorización $N = N_1 N_2$ no ayuda y se cae de nuevo hacia el comportamiento $O(N^2)$. El propio paper es consciente de esto y por eso dedica espacio a la elección de $N$; la práctica habitual es hacer *zero-padding* de la señal hasta la siguiente potencia de 2. Algoritmos posteriores (Bluestein, Rader) resolvieron el caso de $N$ primo, pero eso quedó fuera del alcance de esta nota.
- **Reordenamiento por inversión de bits.** El resultado sale permutado y requiere un paso de bit-reversal; es barato, pero es una complicación de implementación real que la fórmula matemática esconde.
- **Cota que cuenta operaciones, no arquitectura.** La cuenta de "operaciones" (una multiplicación más una suma complejas) es una idealización. En hardware real, el rendimiento depende de patrones de acceso a memoria, caché y paralelismo —motivo por el cual bibliotecas como FFTW *sintonizan* la factorización a la máquina concreta, algo que el análisis de 1965 no aborda.
- **Presentación densa.** Escrito por y para especialistas, el paper es notoriamente conciso; la derivación por bits es difícil de seguir sin diagramas de mariposa, que la pedagogía posterior tuvo que añadir.

## 7. Conexión con la Clase 35 (Introducción al Análisis de Audio)

La Clase 35 recorre una cadena conceptual muy precisa: **naturaleza del sonido → análisis de Fourier (series y transformada) → la FFT → digitalización (cuantización y muestreo) → STFT → MFCC**. Este paper es exactamente el eslabón que hace todo lo demás posible en la práctica.

La clase enseña que una señal de audio es una superposición de sinusoides y que la transformada de Fourier revela su "receta" de frecuencias —el espectro. Pero una señal de audio digital es una secuencia **discreta y finita** de muestras (producto del muestreo y la cuantización que también cubre la clase), así que lo que realmente se calcula no es la transformada continua sino la **DFT**, la misma $X(j) = \sum_k A(k) W^{jk}$ del paper. Calcular esa DFT de forma ingenua costaría $O(N^2)$; con la **FFT** de Cooley-Tukey cuesta $O(N\log N)$, y por eso podemos transformar ventanas de audio en tiempo real.

De ahí se despliega el resto de la clase:

- La **STFT** (transformada de Fourier de tiempo corto) no es más que aplicar una FFT a ventanas sucesivas y solapadas de la señal para ver cómo evoluciona el espectro en el tiempo —el **espectrograma**. Cada columna del espectrograma es una FFT.
- Los **MFCC** parten del espectro de magnitud que entrega la FFT, lo pasan por un banco de filtros en escala mel, toman el logaritmo y aplican una transformada coseno. El primer paso de toda la tubería es la FFT.
- El **laboratorio usa `np.fft.fft`** directamente. Cuando el estudiante ejecuta esa llamada, está corriendo, envuelto en NumPy, el algoritmo derivado en estas cuatro páginas de 1965.

En una frase: **Fourier dio la teoría (1807-1822), Cooley y Tukey dieron el algoritmo que la volvió práctica (1965), y `np.fft.fft` es su encarnación en el laboratorio de la Clase 35.**

**Enlaces internos sugeridos:**

- Clase: [/clases/clase-35](/clases/clase-35) — Introducción al Análisis de Audio (Fourier, FFT, STFT, MFCC).
- Fundamento transversal: procesamiento de señales / análisis espectral (series y transformada de Fourier, muestreo, cuantización).

---

**Nota final — relevancia para salud y señales biomédicas.** La FFT es una herramienta omnipresente en el procesamiento de señales clínicas, no una curiosidad de audio. El análisis espectral de la **electrocardiografía (ECG)** permite estudiar la variabilidad de la frecuencia cardíaca y detectar arritmias descomponiendo el ritmo en sus bandas de frecuencia; el de la **electroencefalografía (EEG)** identifica los ritmos cerebrales —delta, theta, alfa, beta, gamma— que caracterizan estados de sueño, vigilia o crisis epilépticas; y las mismas técnicas de espectrograma y MFCC que la clase aplica al audio se usan para analizar sonidos respiratorios, tos o voz como biomarcadores de enfermedad (por ejemplo, en la detección de patologías pulmonares o neurodegenerativas). En todos estos casos, transformar una señal biomédica larga al dominio de la frecuencia sería computacionalmente prohibitivo sin la reducción de $O(N^2)$ a $O(N\log N)$: la FFT de Cooley y Tukey es, silenciosamente, uno de los algoritmos que sostienen el diagnóstico asistido por computador y el monitoreo continuo de pacientes.
