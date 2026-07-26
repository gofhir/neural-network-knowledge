# Communication in the Presence of Noise — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Communication in the Presence of Noise*.
- **Autor:** Claude E. Shannon, Member, IRE.
- **Afiliación:** Bell Telephone Laboratories, Murray Hill, Nueva Jersey.
- **Venue:** *Proceedings of the I.R.E.* (Institute of Radio Engineers, hoy IEEE), volumen 37, enero de 1949, pp. 10-21.
- **Historia editorial:** manuscrito recibido el 23 de julio de 1940; presentado en la *1948 IRE National Convention* (Nueva York, 24 de marzo de 1948) y ante la IRE New York Section (12 de noviembre de 1947). La fecha de 1940 en el manuscrito original delata que las ideas maduraron durante casi una década antes de publicarse.
- **Linaje:** es el compañero de ingeniería de *A Mathematical Theory of Communication* (Shannon, *Bell System Technical Journal*, 1948), la fundación de la teoría de la información. Mientras el paper de 1948 construye la teoría entera (entropía, codificación, capacidad de fuente), el de 1949 la aterriza en el canal físico ruidoso y de banda limitada — y en su segunda sección enuncia, con demostración explícita, el **teorema de muestreo**.

Este paper hace tres cosas que quedaron como pilares permanentes de la ingeniería moderna. Primero, enuncia y demuestra el **teorema de muestreo** (Theorem 1): una función cuyo espectro no contiene frecuencias superiores a $W$ ciclos por segundo queda *completamente determinada* por sus muestras tomadas cada $1/2W$ segundos, es decir, a una tasa de $2W$ muestras por segundo. Segundo, introduce la **representación geométrica de señales**: una señal de banda $W$ y duración $T$ es un punto en un espacio de $2TW$ dimensiones, y toda la teoría del ruido se vuelve geometría de esferas en alta dimensión. Tercero, deriva la célebre **fórmula de capacidad de canal** (Theorem 2), $C = W\log_2\!\left(1 + \tfrac{P}{N}\right)$, que fija el límite máximo de transmisión confiable sobre un canal con ruido gaussiano blanco.

Para la **Clase 35 (Introducción al Análisis de Audio)** lo que importa de este paper es el primer punto. La clase gira en torno a una pregunta operativa: *¿con qué frecuencia debo muestrear una señal continua para digitalizarla sin perder información?* La respuesta es el teorema de muestreo de Nyquist-Shannon, y este es el documento donde Shannon lo formaliza para la comunidad de ingeniería. Los otros dos aportes (geometría de señales y capacidad de canal) los tratamos como contexto: explican *por qué* Shannon necesitaba el teorema de muestreo — era la herramienta que le permitía contar los grados de libertad de una señal y así medir cuánta información cabe en un canal.

## 2. Contexto: Shannon, la teoría de la información y los precedentes del muestreo

### 2.1. El momento histórico (1948-1949)

Shannon publica este trabajo en el punto más alto de su producción fundacional. En 1948 había aparecido *A Mathematical Theory of Communication*, donde definió la información en términos de **entropía**, midió la incertidumbre de una fuente en **bits** (dígitos binarios, término que en este mismo paper atribuye al uso de la base 2 del logaritmo) y demostró los teoremas de codificación de fuente y de canal. El paper de 1949 es la contraparte orientada al ingeniero de radio: reformula esos resultados en el lenguaje de señales continuas, ancho de banda en ciclos por segundo (cps, hoy hertz) y potencia de ruido térmico.

El contexto tecnológico es la telefonía y la telegrafía multiplexadas, y muy especialmente la **modulación por pulsos codificados (PCM)**, entonces una tecnología emergente en Bell Labs. Shannon menciona explícitamente PCM como el caso donde "las distintas funciones de voz deben ser muestreadas, comprimidas, cuantizadas y codificadas". Esa frase es notable para la Clase 35 porque enumera, en 1949, exactamente las dos operaciones que la clase separa: **muestreo** (discretizar en el tiempo) y **cuantización** (discretizar en amplitud).

### 2.2. Precedentes: Nyquist (1924, 1928) y Whittaker (1935)

Shannon es escrupuloso en reconocer que el resultado no es enteramente nuevo. Cita tres linajes:

- **Harry Nyquist** (Bell Labs, 1924 y 1928). En *"Certain factors affecting telegraph speed"* (1924) y *"Certain topics in telegraph transmission theory"* (1928), Nyquist señaló la importancia fundamental del intervalo de tiempo $1/2W$ segundos en telegrafía: es el ritmo máximo de símbolos distinguibles que un canal de banda $W$ puede sostener sin que se solapen. Shannon honra esto llamando a $1/2W$ el **intervalo de Nyquist** correspondiente a la banda $W$. Por eso la tasa $2W$ se conoce hoy como **frecuencia de Nyquist** o *Nyquist rate*.
- **J. M. Whittaker** (1935). En *Interpolatory Function Theory* (Cambridge Tracts, No. 33), Whittaker desarrolló la teoría de interpolación cardinal — la serie de funciones $\operatorname{sinc}$ que reconstruye una función de banda limitada a partir de muestras equiespaciadas. Shannon reconoce que "el Teorema 1 ha sido dado previamente en otras formas por matemáticos", refiriéndose a esta línea.
- **Dennis Gabor** (1946) y **W. R. Bennett** (1941), citados como quienes ya habían apuntado que aproximadamente $2TW$ números bastan para especificar una señal de banda $W$ y duración $T$.

La contribución de Shannon, entonces, no es haber inventado la fórmula desde cero, sino haberla enunciado con **precisión, generalidad y demostración** dentro de la teoría de la comunicación — y haber mostrado que $2TW$ no es una aproximación sino el número *exacto* de grados de libertad de la señal. Como él mismo dice, el teorema, "a pesar de su evidente importancia, parece no haber aparecido explícitamente en la literatura de la teoría de la comunicación". Fue Shannon quien lo instaló ahí para siempre.

## 3. Contribución central

La contribución que nos ocupa se resume en una sola idea de enorme consecuencia práctica: **una señal analógica de banda limitada no contiene información "infinita"; contiene exactamente $2W$ números independientes por segundo, y muestrear a esa tasa los captura todos sin pérdida.** Esto convierte el problema de digitalizar una señal continua — aparentemente imposible sin perder algo — en un problema resuelto y con receta exacta:

1. Determina la máxima frecuencia $W$ presente en la señal (o fíjala con un filtro pasa-bajos).
2. Muestrea a una tasa $f_s = 2W$ muestras por segundo (o mayor).
3. Reconstruye perfectamente la señal original interpolando las muestras con pulsos $\operatorname{sinc}$.

Si en cambio muestreas *por debajo* de $2W$, aparece el **aliasing**: las frecuencias altas que el muestreo no puede resolver se "pliegan" sobre el rango bajo y se disfrazan de frecuencias que la señal original nunca tuvo, corrompiendo irreversiblemente la reconstrucción. Todo el resto del edificio — CDs a 44.1 kHz, audio digital, imagen digital — es una aplicación directa de este único teorema.

## 4. El teorema de muestreo

### 4.1. Enunciado

Shannon lo formula así (Theorem 1, transcrito con su matemática):

> **Teorema 1.** Si una función $f(t)$ no contiene frecuencias mayores que $W$ cps, queda completamente determinada dando sus ordenadas en una serie de puntos espaciados $1/2W$ segundos entre sí.

En términos de tasa de muestreo: se necesitan **$2W$ muestras por segundo** para capturar toda la información de una señal cuyo espectro está confinado a $[0, W]$. Esa tasa $2W$ es la **frecuencia de Nyquist**, el umbral por debajo del cual la reconstrucción deja de ser posible.

La **justificación intuitiva** que da Shannon es memorable por su sencillez: si $f(t)$ no contiene frecuencias mayores que $W$, entonces no puede cambiar a un valor sustancialmente nuevo en un tiempo menor que medio ciclo de la frecuencia más alta, esto es, $1/2W$. La señal simplemente no tiene la agilidad para "esconder" nada entre muestra y muestra si estas están espaciadas $1/2W$. Muestrear más rápido sería redundante; muestrear más lento dejaría espacio para ambigüedad.

### 4.2. La demostración de Shannon

Lo elegante del paper es que Shannon demuestra que el teorema no es aproximado sino *exacto*. El argumento va así. Sea $F(\omega)$ el espectro (transformada de Fourier) de $f(t)$. Por hipótesis, $F(\omega) = 0$ fuera de la banda, de modo que la transformada inversa solo integra sobre $[-2\pi W, 2\pi W]$:

$$f(t) = \frac{1}{2\pi}\int_{-2\pi W}^{+2\pi W} F(\omega)\, e^{i\omega t}\, d\omega.$$

Ahora evaluamos $f$ en los instantes de muestreo $t = \dfrac{n}{2W}$, con $n$ entero:

$$f\!\left(\frac{n}{2W}\right) = \frac{1}{2\pi}\int_{-2\pi W}^{+2\pi W} F(\omega)\, e^{i\omega \frac{n}{2W}}\, d\omega.$$

El paso clave es reconocer que esta integral es, salvo constantes, **el $n$-ésimo coeficiente de la serie de Fourier de la función $F(\omega)$**, tomando el intervalo $-W$ a $+W$ como periodo fundamental. En otras palabras: los valores de las muestras $f(n/2W)$ *son* los coeficientes de Fourier de $F(\omega)$. Y aquí se cierra el círculo lógico:

1. Las muestras determinan los coeficientes de Fourier de $F(\omega)$.
2. Los coeficientes de Fourier determinan $F(\omega)$ (porque $F$ es cero fuera de la banda, y dentro de ella queda fijada por su serie).
3. $F(\omega)$ determina $f(t)$ (una función queda fijada por su espectro).

Por transitividad, **las muestras determinan $f(t)$ completamente**. Shannon lo remata con la unicidad: "hay una y solo una función cuyo espectro está limitado a la banda $W$ y que pasa por los valores dados en los puntos de muestreo separados $1/2W$ segundos". No hay dos señales de banda limitada distintas que compartan el mismo muestreo.

### 4.3. La fórmula de reconstrucción por interpolación sinc

Determinar que la señal *puede* reconstruirse es una cosa; dar la fórmula explícita es otra. Shannon la provee. La función se reconstruye colocando en cada punto de muestreo un pulso de la forma

$$\frac{\sin 2\pi W t}{2\pi W t},$$

que es la función **sinc**. Este pulso tiene dos propiedades que lo hacen el interpolador perfecto:

- Vale **1 en $t = 0$** y **exactamente 0 en todos los demás puntos de muestreo** $t = n/2W$. Esto garantiza que la reconstrucción pase justo por cada valor muestreado sin que los vecinos interfieran en los instantes de muestreo.
- Su **espectro es constante dentro de la banda $W$ y cero fuera** de ella. Es decir, el pulso sinc es él mismo una señal perfectamente de banda limitada, de modo que cualquier suma de ellos también lo es.

Se coloca en cada muestra un pulso sinc cuya amplitud se ajusta al valor de esa muestra, y se suman todos. La suma satisface simultáneamente las dos condiciones (espectro dentro de $W$ y paso por los valores muestreados), y por unicidad *es* la señal original. Formalmente, con $x_n$ la $n$-ésima muestra:

$$f(t) = \sum_{n=-\infty}^{\infty} x_n\, \frac{\sin \pi(2Wt - n)}{\pi(2Wt - n)}.$$

Esta es la **fórmula de interpolación cardinal** (o *Whittaker-Shannon*). Cada término es un sinc centrado en la muestra $n$-ésima y pesado por su amplitud. En la práctica del audio digital, este es el trabajo idealizado que hace el filtro de reconstrucción del conversor digital-analógico (DAC).

Shannon añade una nota de generalidad importante: si la banda no empieza en frecuencia cero sino en un valor más alto (una banda pasante, no pasa-bajos), el teorema sigue valiendo por traslación lineal — físicamente, una modulación de banda lateral única — y el pulso elemental se obtiene modulando el sinc. También observa que las $2TW$ muestras equiespaciadas no son la única representación posible: se podría usar valores y derivadas en muestras alternas, o cualquier conjunto de $2TW$ números independientes; las muestras uniformes son solo la elección más natural.

### 4.4. Aliasing: qué pasa si se muestrea por debajo de $2W$

El teorema tiene una cara oscura que Shannon no bautiza con el nombre moderno "aliasing" pero cuya lógica se desprende de manera directa de su demostración. La reconstrucción perfecta descansa en que $F(\omega)$ sea **cero fuera de la banda $W$**. Muestrear a una tasa $f_s$ replica el espectro de la señal periódicamente cada $f_s$ hertz a lo largo del eje de frecuencia. Si $f_s \geq 2W$, esas copias no se tocan y un filtro pasa-bajos ideal recupera la copia central intacta. Pero si $f_s < 2W$, las copias adyacentes **se solapan**: la cola de alta frecuencia de una réplica invade el rango de baja frecuencia de la vecina.

El efecto visible es que una componente de frecuencia $f$ que supera la mitad de la tasa de muestreo (la frecuencia de Nyquist $f_s/2$) *no desaparece*: se **pliega** (de ahí el término *folding frequency* para $f_s/2$) y reaparece disfrazada de una frecuencia más baja $f_{\text{alias}} = |f_s - f|$ (para $f$ apenas por encima de $f_s/2$). La señal reconstruida contiene entonces tonos que la original nunca tuvo, y el daño es **irreversible**: una vez que dos frecuencias distintas producen exactamente el mismo patrón de muestras, ninguna operación posterior puede separarlas. Es la traducción en frecuencia de la observación intuitiva de Shannon: si la señal *sí* puede cambiar a un valor nuevo en menos de $1/2W$ (porque contiene frecuencias mayores que $W$), entonces las muestras espaciadas $1/2W$ ya no bastan para atraparla, y lo que se pierde no se pierde limpiamente, sino que contamina lo que queda.

La defensa práctica es el **filtro anti-aliasing**: un pasa-bajos analógico que se aplica *antes* de muestrear, para forzar que la señal cumpla la hipótesis del teorema (nada por encima de $W = f_s/2$). Descartar la información por encima de Nyquist con un filtro es una pérdida controlada; dejar que se pliegue por aliasing es una corrupción incontrolada. Esta distinción es central para la Clase 35: no basta con elegir una tasa de muestreo, hay que garantizar de antemano que la señal está efectivamente limitada en banda.

## 5. El resto del paper como contexto: geometría de señales y capacidad de canal

El teorema de muestreo no es un fin en sí mismo dentro del paper; es la **herramienta de conteo** que habilita todo lo demás. Vale la pena reseñarlo brevemente para entender por qué Shannon necesitaba el teorema.

### 5.1. Representación geométrica: la señal como punto en $2TW$ dimensiones

Si una señal de banda $W$ y duración $T$ está determinada por $2TW$ muestras, entonces esas muestras pueden verse como las **coordenadas de un punto en un espacio de $2TW$ dimensiones**. Cada señal posible es exactamente un punto; el espacio entero de señales de banda $W$ y duración $T$ es este hiperespacio. Shannon ilustra la escala con un ejemplo de televisión: una señal de 5 MHz durante una hora vive en un espacio de $2 \times 5\times10^6 \times 3600 = 3.6\times10^{10}$ dimensiones.

En esta geometría, magnitudes físicas se vuelven distancias. Shannon muestra, usando la ortogonalidad de los pulsos sinc, que el **cuadrado de la distancia del origen al punto es $2W$ veces la energía de la señal**, es decir $d^2 = 2WE = 2WTP$, con $P$ la potencia media. Las señales de potencia acotada por $P$ son puntos dentro de una **esfera de radio $\sqrt{2WTP}$**. Pasar una señal por un filtro ideal es proyectar el punto sobre un subespacio. Añadir ruido es desplazar el punto una distancia proporcional al valor RMS del ruido, creando una pequeña **región de incertidumbre** alrededor de cada punto.

### 5.2. Capacidad de canal: la fórmula $C = W\log_2(1+S/N)$

Con la geometría montada, la deducción de la capacidad se vuelve un problema de empaquetamiento de esferas. Con ruido térmico blanco de potencia $N$, cada punto de señal queda rodeado por una "bola de billar" de ruido de radio $\sqrt{2TWN}$; las señales recibidas, de potencia $P+N$, caen sobre una esfera de radio $\sqrt{2TW(P+N)}$. El número máximo de señales distinguibles es, a lo sumo, el volumen de la esfera grande dividido por el de las bolas de ruido. Contando bits, Shannon llega al Theorem 2:

$$C = W \log_2\!\left(\frac{P+N}{N}\right) = W \log_2\!\left(1 + \frac{P}{N}\right)\ \ \text{bits por segundo.}$$

Esta es la **fórmula de Shannon-Hartley**: el límite máximo de transmisión confiable de un canal de ancho de banda $W$ y relación señal-a-ruido $S/N = P/N$. Es "un resultado bastante sorprendente", en palabras de Shannon, porque afirma que se puede transmitir a la tasa $C$ con frecuencia de errores *arbitrariamente pequeña* — no hay que sacrificar tasa por confiabilidad, basta con codificar sobre secuencias suficientemente largas. El paper también extiende el resultado a ruido gaussiano arbitrario (Theorem 3, vía el concepto de *potencia de entropía*), a fuentes discretas (Theorem 4, codificación de fuente) y a fuentes continuas con criterio de fidelidad (Theorem 5, precursor de la teoría tasa-distorsión).

Para la Clase 35 estos resultados son telón de fondo: muestran que el teorema de muestreo es la pieza que le permite a Shannon tratar señales continuas como vectores de números finitos y así aplicarles conteo, geometría y probabilidad. Sin el teorema de muestreo, no hay "$2TW$ dimensiones" y no hay fórmula de capacidad.

## 6. Impacto: la base de todo el mundo digital

Es difícil exagerar el alcance del teorema de muestreo. Es la **condición de posibilidad de toda la tecnología digital de señales**: audio, imagen, video, telecomunicaciones, instrumentación. Cada vez que un fenómeno físico continuo se convierte en una secuencia de números, alguien eligió una tasa de muestreo, y esa elección se justifica con Nyquist-Shannon.

El ejemplo canónico es el **disco compacto (CD)**, que muestrea audio a **44 100 muestras por segundo (44.1 kHz)**. El oído humano percibe hasta aproximadamente 20 kHz. Aplicando el teorema, para capturar contenido hasta $\sim 22$ kHz se necesita muestrear a al menos $2 \times 22 = 44$ kHz. Los 44.1 kHz del CD son precisamente $2W$ con $W \approx 22$ kHz, dejando un pequeño margen por encima de los 20 kHz audibles para que el filtro anti-aliasing tenga banda de transición donde atenuar sin dañar el rango audible. Otras tasas comunes obedecen la misma lógica: 48 kHz en video profesional, 8 kHz en telefonía (que solo necesita cubrir hasta ~3.4 kHz de voz), 96 y 192 kHz en audio de alta resolución.

El teorema también fija el vocabulario permanente del campo: *Nyquist rate* ($2W$), *frecuencia de Nyquist* o *folding frequency* ($f_s/2$), *aliasing*, *filtro anti-aliasing*, *reconstrucción sinc*. Todo ingeniero de señales opera dentro de este marco casi sin nombrarlo, del mismo modo que todo programador usa lógica booleana — otra herencia de Shannon.

## 7. Limitaciones

El teorema es exacto, pero descansa en **idealizaciones** que la práctica solo aproxima:

- **Banda limitada ideal.** El teorema exige que la señal no tenga *ninguna* componente por encima de $W$. Ninguna señal real de duración finita es exactamente de banda limitada (una señal que es cero fuera de un intervalo temporal tiene, por el principio de incertidumbre de Fourier, espectro infinito). Shannon lo reconoce: no es posible cumplir *exactamente* las dos condiciones (espectro dentro de $W$ y tiempo dentro de $T$) a la vez; solo se puede mantener el espectro dentro de $W$ y hacer la cola temporal muy pequeña, o viceversa. En la práctica se fuerza la condición con un filtro anti-aliasing real, que tiene banda de transición finita y no un corte perfecto — por eso se deja margen (los 44.1 kHz en lugar de 40 kHz exactos).
- **Muestreo uniforme.** La fórmula de reconstrucción sinc supone muestras equiespaciadas. Shannon nota que el muestreo no uniforme también funciona en principio, pero "si hay agrupamiento considerable, las muestras deben conocerse con mucha precisión para dar una buena reconstrucción, y el proceso de reconstrucción es más complicado". El muestreo uniforme es el caso robusto y práctico.
- **Interpolador sinc no causal e infinito.** El pulso sinc ideal se extiende de $-\infty$ a $+\infty$ y no es realizable en tiempo real (requeriría conocer todas las muestras futuras). Los DAC reales usan aproximaciones causales de soporte finito, que introducen pequeños errores de reconstrucción.
- **Amplitud continua, no cuantizada.** El teorema de muestreo trata solo la discretización en el *tiempo*. Supone que cada muestra se conoce con precisión infinita. La discretización en *amplitud* — la cuantización — es un problema distinto, con su propio error (ruido de cuantización), que el teorema no aborda. Esta es justamente la otra mitad de la Clase 35.

## 8. Conexión con la Clase 35 (Análisis de Audio)

La Clase 35 descompone la digitalización de una señal en dos operaciones independientes, y este paper ilumina exactamente la frontera entre ellas:

| Operación | Qué discretiza | Pregunta clave | Respuesta |
|---|---|---|---|
| **Muestreo (sampling)** | El **tiempo**: de continuo a instantes discretos | ¿Con qué frecuencia muestrear? | Teorema de Nyquist-Shannon: a $2W$, la frecuencia de Nyquist |
| **Cuantización** | La **amplitud**: de continua a niveles discretos | ¿Con cuántos bits por muestra? | Depende del rango dinámico y el ruido tolerable (fuera de este paper) |

El teorema de muestreo responde de forma **cerrada y exacta** la pregunta central de la clase — *"¿con qué frecuencia debo muestrear?"* — mientras que la cuantización es un compromiso ingenieril sin respuesta única. Shannon menciona ambas en la misma frase (voz que se "muestrea, comprime, cuantiza y codifica" en PCM), lo que confirma que ya en 1949 la separación conceptual estaba clara.

La lógica de **por qué 44.1 kHz** encapsula toda la clase en un número:

1. La banda de interés (audio audible) llega hasta $W \approx 20$ kHz.
2. El teorema exige $f_s \geq 2W = 40$ kHz para evitar aliasing.
3. Se elige $44.1$ kHz $> 40$ kHz para dar margen de transición al filtro anti-aliasing real.
4. Por debajo de $40$ kHz, las frecuencias agudas se plegarían (aliasing) y aparecerían tonos espurios en el rango grave — audibles y molestos.

Las tres ideas que el estudiante de la clase debe internalizar de este paper:

1. **Una señal de banda limitada tiene información finita.** No hay "infinitos detalles" entre muestras; hay exactamente $2W$ números por segundo. Digitalizar sin pérdida es posible y tiene receta.
2. **La frecuencia de Nyquist $2W$ es un umbral duro.** Muestrear a $2W$ o más recupera todo; muestrear menos corrompe irreversiblemente por aliasing. No es un continuo de "más o menos calidad": es un umbral.
3. **Muestreo y cuantización son problemas distintos.** El teorema resuelve el primero exactamente; el segundo es un compromiso. Confundirlos lleva a errores conceptuales frecuentes.

**Enlaces internos:**

- Clase: [/clases/clase-35](/clases/clase-35) — Introducción al Análisis de Audio (muestreo y cuantización).
- Fundamento transversal: análisis de Fourier y representación espectral de señales (base de la demostración del teorema).

## Nota final: relevancia para salud

En el procesamiento de señales biomédicas el teorema de muestreo deja de ser una abstracción y se vuelve una decisión clínica de primer orden. Cada bioseñal tiene su propia banda útil, y elegir mal la tasa de muestreo introduce aliasing que puede *fabricar* artefactos indistinguibles de fenómenos fisiológicos reales — con consecuencias diagnósticas. El **ECG** concentra su información diagnóstica hasta unos 100-150 Hz (el complejo QRS tiene componentes rápidas), por lo que se muestrea típicamente a 250-1000 Hz, muy por encima de $2W$, para no deformar la morfología de las ondas. El **EEG** rara vez supera los 100 Hz de contenido relevante y suele muestrearse a 256-512 Hz; muestrear demasiado lento plegaría el ruido muscular de alta frecuencia sobre las bandas cerebrales de interés (alfa, beta), confundiendo el análisis. El **audio clínico** — auscultación digital de corazón y pulmón, análisis de voz para patología laríngea — hereda directamente las tasas del audio general, filtradas a la banda de interés. En todos los casos la disciplina es la misma que impone Shannon: fijar la banda con un **filtro anti-aliasing** *antes* de muestrear, elegir $f_s \geq 2W$ con margen, y recordar que lo que se pliega por aliasing no se puede recuperar. En un contexto de registro de datos para modelos de aprendizaje automático en salud, esta higiene de muestreo es la primera línea de defensa contra aprender patrones sobre artefactos que nunca estuvieron en el paciente.
