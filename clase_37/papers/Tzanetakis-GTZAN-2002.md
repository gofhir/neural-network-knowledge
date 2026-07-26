# Musical Genre Classification of Audio Signals (GTZAN) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Musical Genre Classification of Audio Signals*.
- **Autores:** George Tzanetakis (Student Member, IEEE) y Perry Cook (Member, IEEE), ambos de Princeton University (Departamento de Computer Science; Cook con nombramiento conjunto en Music).
- **Venue:** *IEEE Transactions on Speech and Audio Processing*, vol. 10, n.º 5, pp. 293–302, julio de 2002.
- **Recepción:** manuscrito recibido el 28 de noviembre de 2001; revisado el 11 de abril de 2002.
- **Financiamiento:** NSF (Grant 9984087), State of New Jersey Commission on Science and Technology, Intel y la Arial Foundation.
- **Software:** el sistema se distribuyó como parte de **MARSYAS**, un framework libre (GNU Public License) con arquitectura cliente–servidor (servidor C++ para señales y reconocimiento de patrones, cliente Java para la interfaz gráfica).

Este es uno de los papers fundacionales del **music information retrieval (MIR)** moderno. Formaliza la tarea de **clasificación automática de géneros musicales** a partir de la señal de audio cruda y propone tres familias de descriptores —**timbral texture**, **rhythmic content** y **pitch content**— diseñados específicamente para música (no solo tomados prestados del reconocimiento de voz). Sobre un conjunto de datos propio de **diez géneros musicales**, el sistema alcanza **61%** de exactitud en clasificación no-real-time (whole file) y **44%** en real-time por frame, resultados que los autores describen como *comparables al desempeño humano* medido en experimentos con clips cortos.

Más allá de las cifras, la contribución más duradera fue **el conjunto de datos** construido para entrenar y evaluar los clasificadores: 1.000 fragmentos de 30 segundos repartidos en 10 géneros de 100 ejemplos cada uno. Ese conjunto —que la comunidad bautizó después como **GTZAN**, por las iniciales del primer autor— se convirtió en el **benchmark didáctico de facto** para clasificación de género, el "MNIST del audio musical". Es precisamente el dataset del **laboratorio de la Clase 37**.

## 2. Contexto: la explosión de la música digital y el MIR

El paper se escribe en pleno estallido de la distribución digital de música. Tzanetakis y Cook lo dicen sin rodeos: es muy probable que en el futuro cercano *toda la música grabada de la historia humana esté disponible en la Web*, y citan explícitamente la atención legal que había recibido Napster como síntoma de la importancia creciente de la distribución digital. Cuando el catálogo pasa de unos cientos de discos físicos a millones de archivos, **organizar y estructurar automáticamente** ese acervo deja de ser un lujo y se vuelve una necesidad de servicio.

Los **géneros musicales** son las etiquetas categóricas que los humanos crean para describir y organizar ese universo. El paper es honesto sobre su naturaleza: los géneros *no tienen definiciones ni fronteras estrictas*, surgen de una interacción compleja entre público, marketing y factores históricos y culturales. Tanto es así que algunos investigadores propusieron definir un esquema de géneros nuevo, pensado exclusivamente para recuperación de información. Aun así, los miembros de un género comparten características observables ligadas a la **instrumentación**, la **estructura rítmica** y el **contenido armónico (pitch)** de la música. Esas tres dimensiones son, no por casualidad, las que motivan las tres familias de features.

Hasta ese momento las jerarquías de géneros se construían **manualmente** por expertos humanos. La automatización de esa anotación es lo que el trabajo persigue, encuadrándola dentro de un objetivo mayor: un sistema completo de **music information retrieval** sobre señales de audio. El propio paper subraya un beneficio colateral clave: desarrollar features capaces de clasificar género proporciona *un marco para desarrollar y evaluar descriptores de contenido musical* que sirven además para recuperación por similitud, segmentación y *audio thumbnailing*. En otras palabras, la clasificación de género es tanto un fin como un banco de pruebas para representaciones de audio.

El trabajo relacionado que el paper revisa proviene sobre todo del **reconocimiento de voz** y de la discriminación música/voz: los **MFCC** de Davis y Mermelstein, el uso del zero-crossing rate y energía de Saunders, el discriminador multi-feature música/voz de Scheirer y Slaney, la clasificación con HMM de cepstra, y sistemas de recuperación de instrumentos y efectos como el de Wold et al. La observación crítica de Tzanetakis y Cook es que esos descriptores acústicos *no modelan directamente la señal musical* —no capturan, por ejemplo, la estructura rítmica— y por eso son insuficientes para clasificar género. De ahí la necesidad de features nuevas, específicamente musicales.

## 3. Contribución central

La contribución tiene dos caras inseparables:

1. **Tres familias de features de contenido musical.** El timbral texture set se apoya en descriptores ya conocidos de voz y discriminación música/voz, pero las otras dos familias —**rhythmic content** (basada en un *beat histogram* derivado de la transformada wavelet) y **pitch content** (basada en *pitch histograms* de detección multi-pitch)— son **nuevas y diseñadas específicamente** para representar aspectos musicales: ritmo y armonía. Tenerlas separadas permite además distintos tipos de recuperación por similitud (por timbre, por ritmo o por armonía).

2. **Un conjunto de datos de referencia.** Para evaluar las features se recolectó una colección real de audio desde radio, discos compactos y archivos MP3, con esfuerzo deliberado por asegurar variedad de calidades de grabación. El subconjunto de 10 géneros, con 100 clips de 30 s cada uno, es el que la comunidad terminó adoptando bajo el nombre **GTZAN**.

El paper también aporta la distinción operativa entre clasificación **whole-file** (un vector por archivo, no real-time) y **real-time frame-based** (un vector por ventana de análisis), mostrando que solo el timbral texture set es utilizable en tiempo real.

## 4. Método

### 4.1. Extracción de features: ventana de análisis y ventana de textura

Toda la maquinaria opera sobre archivos de audio mono, 16 bits, muestreados a **22.050 Hz**. El análisis de corto plazo trocea la señal en **ventanas de análisis** de **23 ms (512 muestras)**, lo bastante cortas para asumir estacionariedad espectral. Pero la *sensación* de una textura sonora no vive en un solo espectro instantáneo, sino en el patrón de muchos espectros sucesivos. Para capturar esa naturaleza de largo plazo, el sistema introduce la **ventana de textura (texture window)**: en lugar de usar los valores de feature directamente, calcula **medias y varianzas móviles** de las features sobre un número de ventanas de análisis. Conceptualmente equivale a estimar los parámetros de una gaussiana multidimensional móvil, o a una "memoria del pasado" implementada con un buffer circular. La ventana de textura usada es de **1 s (43 ventanas de análisis)**.

### 4.2. Timbral texture features

Se basan en la **STFT** (transformada de Fourier de corto plazo) y se calculan por frame. Sea $M_t[n]$ la magnitud de la transformada de Fourier en el frame $t$ y el bin de frecuencia $n$. Los descriptores son:

- **Spectral centroid** (centroide espectral): el "centro de gravedad" del espectro de magnitud,

$$C_t = \frac{\sum_{n=1}^{N} n\, M_t[n]}{\sum_{n=1}^{N} M_t[n]}$$

Es una medida de la forma espectral; valores altos corresponden a texturas "más brillantes", con más altas frecuencias.

- **Spectral rolloff**: la frecuencia $R_t$ por debajo de la cual se concentra el **85%** de la distribución de magnitud,

$$\sum_{n=1}^{R_t} M_t[n] = 0{,}85 \sum_{n=1}^{N} M_t[n]$$

Otra medida de forma espectral.

- **Spectral flux**: la diferencia cuadrática entre magnitudes normalizadas de espectros sucesivos,

$$F_t = \sum_{n=1}^{N} \big(N_t[n] - N_{t-1}[n]\big)^2$$

donde $N_t$ y $N_{t-1}$ son las magnitudes normalizadas del frame actual y el previo. Mide la cantidad de cambio espectral local.

- **Time domain zero crossings** (cruces por cero): 

$$Z_t = \frac{1}{2}\sum_{n} \big| \operatorname{sign}(x[n]) - \operatorname{sign}(x[n-1]) \big|$$

donde $x[n]$ es la señal en el dominio del tiempo del frame. Es una medida de la *ruidosidad* de la señal.

- **Mel-Frequency Cepstral Coefficients (MFCC)**: features perceptualmente motivadas, también basadas en la STFT. Tras tomar el log de la magnitud del espectro, los bins de la FFT se agrupan y suavizan según la escala perceptual **Mel**, y finalmente se aplica una **transformada coseno discreta (DCT)** para decorrelacionar el vector resultante. Aunque en voz suelen usarse 13 coeficientes, los autores encontraron que **los primeros cinco** dan el mejor desempeño para clasificación de género.

- **Low-energy feature**: única feature basada en la ventana de textura y no en la de análisis. Es el **porcentaje de ventanas de análisis con energía RMS menor que la energía RMS promedio** de la ventana de textura. Música vocal con silencios tendrá low-energy alto; cuerdas continuas, bajo.

El **vector de timbral texture** resultante tiene **19 dimensiones**: medias y varianzas de centroid, rolloff, flux y zero-crossings sobre la ventana de textura (8), low-energy (1) y medias y varianzas de los primeros cinco MFCC (10, excluyendo el coeficiente del componente DC).

### 4.3. Rhythmic content features: el beat histogram

Para representar el ritmo, el sistema construye un **beat histogram (BH)** apoyado en la **transformada wavelet (WT)**. A diferencia de la STFT, que da resolución temporal uniforme para todas las frecuencias, la WT ofrece alta resolución temporal y baja resolución en frecuencia para altas frecuencias, y lo inverso para bajas. La **DWT** se implementa con el algoritmo piramidal (filtros **DAUB4** de Daubechies) y equivale a una descomposición en octavas de la señal.

El pipeline de análisis de beat es: descomponer la señal en bandas de octava con la DWT; extraer la **envolvente temporal** de cada banda mediante **rectificación de onda completa**, **filtrado pasa-bajos** (un filtro de un polo con $\alpha = 0{,}99$) y **downsampling**; **remover la media**; sumar las envolventes; y calcular la **autocorrelación** de la envolvente sumada. Los picos dominantes de la autocorrelación corresponden a las periodicidades de la envolvente. La autocorrelación se **realza** con el método de Tolonen y Karjalainen para reducir el efecto de múltiplos enteros de las periodicidades básicas. Los tres primeros picos en el rango apropiado (bins de **40 a 200 bpm**) se acumulan en el histograma; en vez de sumar 1, se **suma la amplitud del pico**, de modo que las señales muy auto-similares (beat fuerte) producen picos más altos.

Los autores ilustran el BH con "Come Together" de los Beatles (picos en ~80 y ~160 bpm, el beat principal y su primer armónico), y contrastan clásica (Debussy, sin pico dominante claro por la complejidad orquestal), jazz (Dee Dee Bridgewater, picos en 70/140 bpm) y hip-hop (Neneh Cherry, picos muy marcados). Un pequeño estudio (20 excerpts) confirmó que en 18/20 el beat principal cae en el primer o segundo pico del BH. Las **features rítmicas (6 dimensiones)** extraídas son:

- **A0, A1**: amplitud relativa (dividida por la suma de amplitudes) del primer y segundo pico.
- **RA**: razón entre la amplitud del segundo pico y la del primero.
- **P1, P2**: periodo del primer y segundo pico, en bpm.
- **SUM**: suma total del histograma (indicador de la fuerza del beat).

El BH se calcula sobre ventanas de **65.536 muestras (~3 s)** con hop de 32.768 muestras, ventana grande necesaria para captar las repeticiones a nivel de beat y subbeat.

### 4.4. Pitch content features: el pitch histogram

El contenido de pitch se obtiene con el algoritmo de **detección multi-pitch** de Tolonen y Karjalainen: la señal se descompone en dos bandas (por debajo y por encima de 1000 Hz), se extraen envolventes por banda (rectificación de media onda + pasa-bajos), se suman y se computa una **autocorrelación realzada (SACF)**. Los picos prominentes de la SACF son los pitches principales del segmento; los tres dominantes se acumulan en un **pitch histogram (PH)** sobre todo el archivo, con ventana de análisis de 512 muestras (~23 ms).

Las frecuencias de los picos se convierten a notas musicales según el esquema **MIDI**:

$$c = 12 \log_2\!\left(\frac{f}{440}\right) + 69$$

donde $f$ es la frecuencia en Hz y $c$ el bin del histograma (número de nota MIDI). Se generan dos versiones: un histograma **desplegado (UPH)**, que conserva el rango de octavas, y uno **plegado (FPH)**, donde todas las notas se mapean a una sola octava (clase de pitch o *chroma*):

$$c_{\text{folded}} = c_{\text{unfolded}} \bmod 12$$

El FPH captura el contenido armónico (clases de pitch); el UPH, el rango de pitch de la pieza. Finalmente el FPH se remapea a un **círculo de quintas** de modo que bins adyacentes disten una quinta (7 semitonos) en vez de un semitono, lo que expresa mejor las relaciones tonales (tónica–dominante) y mejora la exactitud. Las **features de pitch (5 dimensiones)** son: **FA0** (amplitud del pico máximo del FPH, la clase de pitch dominante), **UP0** (periodo del pico máximo del UPH, rango de octava dominante), **FP0** (periodo del pico máximo del FPH, clase de pitch principal), **IPO1** (intervalo de pitch entre los dos picos más prominentes del FPH, típicamente quinta/cuarta en música simple) y **SUM** (suma del histograma, fuerza de la detección de pitch).

Rítmicas y de pitch se calculan **sobre el archivo completo**; solo el timbral texture set puede computarse en tiempo real.

### 4.5. Clasificadores

Se usaron clasificadores estándar de **reconocimiento estadístico de patrones (SPR)**, cuya idea es estimar la función de densidad de probabilidad de los vectores de feature de cada clase a partir de un conjunto de entrenamiento etiquetado:

- **Gaussian simple (GS)**: cada clase se modela como una única gaussiana multidimensional.
- **Gaussian Mixture Model (GMM)**: cada clase es una **mezcla** de gaussianas; se usan matrices de covarianza diagonales, se inicializan con **k-means** con múltiples arranques aleatorios y se ajustan con el algoritmo **EM**. Se reportan configuraciones con 2 y 3 componentes (GMM(3)).
- **k-Nearest Neighbors (k-NN)**: clasificador no paramétrico que etiqueta cada muestra por mayoría de sus vecinos más cercanos, sin asumir forma funcional para la pdf.

El vector combinado completo tiene **30 dimensiones**: 19 timbrales + 6 rítmicas + 5 de pitch. La evaluación usa **validación cruzada de diez pliegues** (90% entrenamiento, 10% prueba), iterada con particiones aleatorias distintas y promediada (100 iteraciones para la tabla principal). Para el modo real-time, los frames de un mismo archivo nunca se reparten entre entrenamiento y prueba, para evitar exactitud inflada por similitud intra-archivo.

## 5. Experimentos y resultados

El conjunto completo de evaluación abarca una **jerarquía** de 20 géneros musicales más tres categorías de voz (male, female, sports announcing). Para cada género se usaron **100 excerpts representativos de 30 s**, tomados de radio, CD y MP3, en formato 22.050 Hz / 16 bits / mono. El subconjunto central —el que se volvió GTZAN— es el dataset **Genres** con diez clases: **classical, country, disco, hiphop, jazz, rock, blues, reggae, pop, metal**. Se evaluaron además dos jerarquías finas: **classical** (choir, orchestra, piano, string quartet) y **jazz** (bigband, cool, fusion, piano, quartet, swing).

**Resultado principal:** con el vector combinado de 30 dimensiones y clasificación whole-file, se alcanza **61%** de exactitud en los diez géneros; la versión real-time por frame (solo timbral texture, 19 dims, clasificador gaussiano) llega a **44%**. La clasificación de azar sería 10%. Los clasificadores auxiliares dan **86%** en música/voz (azar 50%) y **74%** en el clasificador de voz de tres clases (azar 33%).

**Matriz de confusión.** Las confusiones del sistema son *parecidas a las que cometería un humano*. La música clásica se confunde con jazz en piezas de fuerte ritmo de compositores como Bernstein o Gershwin (un 26% de la clásica fue etiquetada como jazz). El **rock** es el género con **peor exactitud** y el más fácilmente confundido con otros, algo esperable por su naturaleza amplia. En la jerarquía fina, las clases de jazz se confunden mayormente con *fusion* (categoría amplia y de gran variabilidad), y en la clásica la orquesta se confunde con cuarteto de cuerdas.

**Importancia relativa de las features.** Un estudio con clasificador gaussiano muestra que las features **timbrales (STFT + MFCC) son las más informativas**; las de pitch (PHF) y beat (BHF) rinden peor por separado, aunque todas superan el azar y aportan información. Combinarlas mejora, y las mejores features individuales fueron: la suma del beat histogram (BHF.SUM), el periodo del primer pico del pitch histogram plegado (PHF.FP0), la varianza del spectral centroid, y la media del primer coeficiente MFCC. El estudio del **tamaño de ventana de textura** confirma que usarla eleva significativamente la exactitud, con retornos que se saturan pasadas ~40 ventanas de análisis (1 s).

**Comparación con humanos.** El paper cita el experimento de Perrot y Gjerdigen: en un paradigma de elección forzada de diez vías, estudiantes universitarios clasificaron correctamente el **53%** tras oír solo **250 ms**, y el **70%** tras **3 s** (azar 10%); oír más de 3 s no mejoraba. Aunque la comparación directa no es posible (géneros y datasets distintos), los autores concluyen que **el desempeño automático no está lejos del humano**, y que ambos resultados reflejan la naturaleza *difusa* de las fronteras de género.

## 6. Limitaciones y los defectos conocidos del dataset

El propio paper reconoce límites de método: las features rítmicas y de pitch se calculan sobre el archivo completo, lo que solo es válido si el archivo es homogéneo; para archivos con regiones de textura distinta haría falta segmentación previa. Combinar las tres familias no siempre incrementa la exactitud de forma significativa, y los autores sugieren que puede hacer falta **diseñar features específicas por género** para subclasificación fina. También anticipan mejoras futuras: detección de pitch por modelos cocleares, front-ends de filterbank alternativos para el beat, y clasificadores como redes neuronales artificiales.

La limitación más citada, sin embargo, no aparece en el paper de 2002 sino en la **literatura posterior** que auditó el conjunto de datos GTZAN una vez que se volvió estándar. Se documentaron **repeticiones** (el mismo tema o el mismo artista aparece varias veces, a veces en distintos géneros), **errores de etiquetado (mislabels)** y **distorsiones** en varios clips. Estos defectos implican que un modelo puede "hacer trampa" reconociendo artistas o grabaciones específicas en vez de aprender género, y que las cifras de exactitud reportadas sobre GTZAN sin control de estas fugas tienden a estar **optimistamente sesgadas**. Pese a todo, el conjunto sobrevivió: por su tamaño manejable, su formato simple (WAV de 30 s) y su cobertura de diez géneros balanceados, se consolidó como el **benchmark didáctico de facto** —el punto de partida obligado para enseñar clasificación de género— aun cuando para investigación de frontera se prefieran datasets más grandes y limpios. Conviene por eso presentarlo con esa doble cara: excelente para aprender, imperfecto para publicar resultados definitivos.

## 7. Conexión con la Clase 37 y con el laboratorio

La **Clase 37, "Datasets y Herramientas para Audio"** (segunda de cinco del bloque de audio), gira en torno a cómo se representa, se almacena y se clasifica el audio musical. Este paper es la **piedra angular histórica** de esa clase por dos razones. Primero, porque **define el vocabulario de features** que sigue vigente: cuando el laboratorio extrae MFCC, spectral centroid, rolloff, flux y zero-crossing rate de un WAV, está usando exactamente el timbral texture set que Tzanetakis y Cook sistematizaron. Segundo, porque **aporta el dataset del laboratorio**: los "1.000 temas de 30 s en 10 géneros" que menciona la clase *son* GTZAN, el conjunto que este paper construyó.

El flujo del **laboratorio de clasificación de géneros** replica en miniatura el pipeline del paper: se toma cada archivo **WAV** de GTZAN, se extraen features de audio (típicamente con librosa: MFCC, centroide, rolloff, flux, ZCR, agregadas en medias y varianzas —el equivalente a la ventana de textura del paper—), y se alimenta un **clasificador** (desde un k-NN o un GMM como en el original, hasta una red neuronal o un modelo sobre espectrogramas) que predice uno de los diez géneros. Entender este paper le da al estudiante el *porqué* de cada paso del notebook: por qué se promedian las features en el tiempo, por qué los MFCC dominan, por qué el rock será el género peor clasificado, y por qué un ~61% de exactitud ya es un buen resultado —no un fracaso— dado que roza el techo humano en esta tarea intrínsecamente difusa. También explica una lección metodológica valiosa: la importancia de auditar el dataset (repeticiones, mislabels) antes de creer ciegamente en la métrica.

**Enlaces internos:**

- Clase: [/clases/clase-37](/clases/clase-37) — Datasets y Herramientas para Audio.
- Laboratorio: clasificación de géneros musicales sobre GTZAN (WAV → features → modelo).

## Nota final: relevancia para salud

El armazón conceptual de este paper —**extraer descriptores interpretables de una señal acústica y clasificarla con reconocimiento estadístico de patrones**— trasciende la música y es directamente aplicable a dominios clínicos. En salud, el análisis de audio es una frontera activa: la tos y la respiración para detectar patologías respiratorias, la fonación y la prosodia para tamizaje de Parkinson o depresión, los sonidos cardíacos y pulmonares en auscultación digital. En todos estos casos el **MFCC** —la misma feature perceptualmente motivada que aquí domina la clasificación de género— sigue siendo el descriptor de partida, precisamente porque comprime el envolvente espectral en pocos coeficientes robustos e interpretables. El paralelo es exacto: así como el spectral centroid distingue un tema "brillante" de uno "oscuro", los mismos estadísticos de tiempo-frecuencia distinguen una tos húmeda de una seca o una voz disfónica de una sana. La lección de Tzanetakis y Cook —que un vector compacto de features musicales bien elegidas, más un clasificador simple, alcanza desempeño cercano al humano— es el mismo principio que hace atractivos a los modelos de audio interpretables en medicina, donde la trazabilidad de "qué escuchó el modelo" (a diferencia de un embedding opaco) es un requisito, no un lujo.
