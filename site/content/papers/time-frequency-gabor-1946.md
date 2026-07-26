---
title: "Theory of Communication — incertidumbre tiempo-frecuencia (Gabor, 1946)"
weight: 395
math: true
---

{{< paper-card
    title="Theory of Communication"
    authors="Dennis Gabor"
    year="1946"
    venue="Journal of the IEE"
    pdf="/papers/time-frequency-gabor-1946.pdf" >}}
Este es la **raíz teórica del análisis tiempo-frecuencia**. Gabor ataca la dicotomía histórica de la ingeniería de comunicaciones —describir una señal *o bien* como función del tiempo $s(t)$ *o bien* por su espectro de Fourier $S(f)$— y muestra que ambas son idealizaciones extremas. Propone en su lugar un **plano de información** de dos dimensiones, con tiempo y frecuencia como coordenadas, donde toda señal es un área que se cuantiza en **celdas de tamaño mínimo**, cada una portadora de un dato: un cuanto de información que bautiza **logón**. El piso del tamaño de esas celdas es un **principio de incertidumbre para señales**, $\Delta t \cdot \Delta f \ge \tfrac{1}{2}$, análogo al de Heisenberg. Y las señales que alcanzan la igualdad —las de área mínima— son oscilaciones armónicas moduladas por una envolvente gaussiana: las **señales elementales**, hoy llamadas *átomos de Gabor*. Es el paper detrás del trade-off de la ventana en la STFT que estudia la [Clase 35](/clases/clase-35).
{{< /paper-card >}}

---

## Contexto: comunicación, Fourier y el precio de perder el tiempo

Gabor sitúa su trabajo tras un principio que "amaneció gradualmente" sobre los ingenieros en los años veinte: **transmitir cierta información por unidad de tiempo exige un ancho de banda mínimo**. El linaje va de Carson (1922) a Nyquist y Küpfmüller (1924) hasta Hartley (1928), que generaliza: "la información transmisible es proporcional al producto del rango de frecuencia por el tiempo disponible". Ese producto **frecuencia × tiempo** es el germen del plano de información.

El núcleo es la **crítica a las dos descripciones tradicionales**. La descripción temporal opera con instantes agudos, pero al precio de infinitos datos y trenes de onda de frecuencia rigurosa (duración infinita). El análisis de Fourier,

$$s(t) = \int_{-\infty}^{\infty} S(f)\, e^{2\pi i f t}\, df, \qquad S(f) = \int_{-\infty}^{\infty} s(t)\, e^{-2\pi i f t}\, dt,$$

describe la señal *sub specie aeternitatis*, en un intervalo infinito. El argumento es demoledor: si "frecuencia" se usa en el sentido matemático estricto —que solo aplica a trenes de onda infinitos— entonces "frecuencia cambiante" es una contradicción en los términos. Una oscilación armónica pura es una **línea vertical** en el plano tiempo-frecuencia (frecuencia exacta, época indefinida); una función delta es una **línea horizontal** (época exacta, espectro uniforme). Entre ambos extremos está lo que realmente escuchamos: la música y el habla tienen "un patrón de tiempo tanto como un patrón de frecuencia".

## Contribución central: el plano de información y los logones

La propuesta constructiva es representar las señales en **dos dimensiones, con tiempo y frecuencia como coordenadas** —"diagramas de información"— donde **las áreas son proporcionales al número de datos independientes** que una región puede transmitir. Gabor razona por la física de los instrumentos: cualquier resonador real tiene un **tiempo de decaimiento** y un **ancho de sintonía** cuyo producto es del orden de la unidad, de modo que a cada instrumento le corresponde un **rectángulo característico** en el plano. Un oscilógrafo los tiene anchos y bajos (buena resolución temporal); un banco de lengüetas, angostos y altos (lo inverso). El número de rectángulos en una región es el número de datos independientes obtenibles —de ahí el nombre "diagrama de información".

De aquí sale el concepto estrella: dividir el plano en **celdas de área un medio** ($\Delta t \cdot \Delta f = \tfrac{1}{2}$) y asociar a cada una una **señal elemental** que transmite exactamente un dato numérico, el **logón**. Cualquier señal se expande en logones, y esa expansión **contiene al análisis temporal y al de Fourier como casos extremos** (celdas infinitamente altas y angostas, o lo opuesto). El logón es la unidad atómica común a ambos mundos.

## El principio de incertidumbre para señales

El corazón matemático es la formulación exacta de la relación de incertidumbre, precisada —observa Gabor— solo con la mecánica cuántica de Heisenberg (1927). Para hacerla cuantitativa introduce la **señal compleja** $\psi(t) = s(t) + j\sigma(t)$, donde $\sigma$ es la señal en cuadratura de $s$ (par de transformadas de Hilbert): esto elimina las frecuencias negativas y permite que $\psi$ juegue el papel de la función de onda. Con la función de peso $\psi^*\psi = s^2 + \sigma^2$ define la **duración efectiva** y el **ancho de frecuencia efectivo** como desviaciones r.m.s. ponderadas:

$$\Delta t = \left[\,2\pi\,\overline{(t-\bar t)^2}\,\right]^{1/2}, \qquad \Delta f = \left[\,2\pi\,\overline{(f-\bar f)^2}\,\right]^{1/2}.$$

Multiplicando ambas y aplicando una forma de la **desigualdad de Schwarz** (debida a Weyl y Pauli), el producto queda acotado por debajo:

$$\Delta t \cdot \Delta f \ge \tfrac{1}{2}.$$

Es, en palabras de Gabor, "la identidad matemática que está en la raíz del principio fundamental de la comunicación": la duración r.m.s. de una señal y su ancho de frecuencia r.m.s. definen un **área mínima** en el diagrama, simétrica en $t$ y $f$. (Nota de convención: la constante $\tfrac{1}{2}$ corresponde a la definición particular de Gabor; en otras convenciones habituales de procesamiento de señales el mismo principio aparece como $\Delta t \cdot \Delta f \ge \tfrac{1}{4\pi}$. El contenido físico —el producto está acotado por debajo— es idéntico.)

## Las señales elementales gaussianas

Enunciado el límite, Gabor pregunta cuál es la señal que **satura la desigualdad**. La respuesta, que resalta en cursivas, es notablemente simple: la señal de área mínima $\Delta t \cdot \Delta f = \tfrac{1}{2}$ es una oscilación armónica modulada por un pulso gaussiano. En forma compleja:

$$\psi(t) = e^{-\alpha^2 (t - t_0)^2}\,\operatorname{cis}\!\big(2\pi f_0 t + \varphi\big),$$

con $\alpha$ la agudeza del pulso, $t_0$ la época del pico, $f_0$ la frecuencia moduladora y $\varphi$ la fase. Por simetría, su **espectro es otra gaussiana** centrada en $f_0$, y las agudezas en tiempo y frecuencia son **recíprocas**: estrechar el pulso en el tiempo lo ensancha en frecuencia, exactamente en la proporción que fija la incertidumbre. Los dos casos límite reaparecen como extremos de $\alpha$: con $\alpha = 0$ la envolvente se vuelve infinita y la señal degenera en una sinusoide pura (Fourier); con $\alpha \to \infty$ se estrecha hasta una delta (análisis temporal). Todo el continuo entre ambos vive en una sola familia gaussiana. Gabor demuestra además que cualquier señal se expande en estas señales elementales —aunque, al **no ser ortogonales**, los coeficientes se obtienen por aproximaciones sucesivas, un proceso "más bien inconveniente" pero teóricamente fundamental.

## Legado e impacto

La huella de estas 16 páginas es directa. Expandir una señal en átomos gaussianos desplazados en tiempo y frecuencia es, literalmente, la **transformada de Gabor**: el caso de la STFT con ventana gaussiana. La STFT general —deslizar una ventana arbitraria y transformar cada segmento— es su generalización práctica. El **espectrograma** (módulo al cuadrado de la STFT) es la realización concreta del "diagrama de información" que Gabor dibujó a mano. Las **wavelets** nacen de su observación de que la razón de aspecto $\Delta t/\Delta f$ del rectángulo es arbitraria: usan celdas de aspecto variable, angostas en tiempo para altas frecuencias y anchas para bajas. Y toda técnica tiempo-frecuencia posterior opera bajo el mismo techo $\Delta t \cdot \Delta f \ge \text{const}$: no hay transformada que lo evada, solo maneras distintas de repartir el compromiso.

## Limitaciones

- **Retícula rígida.** La malla de celdas de igual tamaño reparte la resolución de forma uniforme; señales con eventos breves en alta frecuencia y sostenidos en baja piden aspectos variables —la limitación que las wavelets resolverían.
- **Base no ortogonal.** Las señales elementales no son ortogonales, así que la expansión en logones exige aproximaciones sucesivas en vez de proyección directa; Gabor mismo la califica de "inconveniente".
- **El límite es duro, no evitable.** El principio no es un defecto corregible con mejor hardware o aritmética: es una cota matemática sobre lo que una señal *puede* significar. Ninguna técnica lo supera.

## Por qué importa para la Clase 35

La Clase 35 enseña que al calcular una STFT hay que **elegir el largo de la ventana**, y que la elección es un compromiso irreducible:

- **Ventana ancha (larga en tiempo):** muchos ciclos → **buena resolución en frecuencia** ($\Delta f$ pequeño), pero no se sabe *cuándo* ocurrió cada componente → **mala resolución en tiempo** ($\Delta t$ grande).
- **Ventana angosta (corta en tiempo):** localiza el instante → **buena resolución en tiempo**, pero pocos ciclos → **mala resolución en frecuencia**.

**Ese trade-off *es* el principio de incertidumbre de Gabor**, no una limitación de los algoritmos. Cuando la clase dibuja las celdas del espectrograma —anchas y bajas con ventana larga, angostas y altas con ventana corta— está dibujando los rectángulos de área mínima de este paper. Y cuando se recomienda la **ventana gaussiana** como elección "óptima", la razón profunda es la de la sección anterior: la gaussiana es la única forma que *satura* la desigualdad. Gabor demostró en 1946 que el continuo entre "todo tiempo" y "toda frecuencia" existe, que tiene un piso, y que en el piso vive la gaussiana. El detalle del plano tiempo-frecuencia y de la STFT vive en el fundamento [representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia). Entender que ese límite es fundamental —y no un defecto corregible— es lo que permite al ingeniero biomédico elegir conscientemente el compromiso adecuado al analizar señales no estacionarias como el EEG, el ECG o los sonidos cardíacos y pulmonares.
