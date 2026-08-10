# Convolutional, Long Short-Term Memory, Fully Connected Deep Neural Networks (CLDNN) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** Tara N. Sainath, Oriol Vinyals, Andrew Senior, Haşim Sak.
- **Afiliación:** Google, Inc., New York, NY, USA. Los cuatro con correo `@google.com`; no hay coautores académicos. Es un paper de laboratorio industrial escrito sobre infraestructura de producción.
- **Venue:** ICASSP 2015 (IEEE International Conference on Acoustics, Speech and Signal Processing), Brisbane, Australia. El PDF que tenemos es el preprint de autor compilado con pdfTeX el **10 de octubre de 2014** (metadata del archivo), sin numeración de página de las actas; la paginación habitual que se cita (pp. 4580–4584) **no se puede verificar contra este PDF** y la doy solo como referencia externa.
- **Extensión:** 5 páginas, formato ICASSP de dos columnas. 20 referencias. 9 tablas, 1 figura.

**Qué propone.** Una sola red, entrenada conjuntamente, que apila tres bloques en un orden específico: capas convolucionales → capas LSTM → capas totalmente conectadas. El nombre CLDNN es el acrónimo de ese apilamiento. La tesis es que los tres tipos de red son **complementarios**, no alternativos: la CNN reduce la variación en frecuencia, el LSTM modela la estructura temporal, el DNN mapea a un espacio más separable.

**Cifras ancla, todas verificadas contra las tablas del PDF:**

| Conjunto | Baseline | Modelo | WER baseline | WER modelo | Mejora relativa (calculada) | Ubicación |
|---|---|---|---|---|---|---|
| 200 h limpio | LSTM (init gaussiana) | CLDNN | 18.0 | 17.3 | **3.9%** | Tabla 5 |
| 200 h limpio | LSTM (init uniforme) | CLDNN | 17.7 | 17.0 | **4.0%** | Tabla 6 |
| 200 h limpio | LSTM (init uniforme) | multi-escala CLDNN | 17.7 | 16.8 | **5.1%** | Tabla 7 |
| 2000 h limpio, CE | LSTM | multi-escala CLDNN | 14.6 | 13.8 | **5.5%** | Tabla 8 |
| 2000 h limpio, secuencia | LSTM | CLDNN / multi-escala | 13.7 | 13.1 | **4.4%** | Tabla 8 |
| 2000 h ruidoso, CE | LSTM | multi-escala CLDNN | 20.3 | 19.2 | **5.4%** | Tabla 9 |
| 2000 h ruidoso, secuencia | LSTM | CLDNN / multi-escala | 18.8 | 17.4 | **7.4%** | Tabla 9 |

El abstract resume todo esto como "**4-6% relative improvement in WER over an LSTM, the strongest of the three individual models**". Como se ve en la última columna, esa banda no cubre el mejor resultado del propio paper (7.4% en la condición ruidosa con entrenamiento de secuencia) y su extremo superior depende de redondear 5.5% a "6%". Volveré sobre esto en la Sección 12.

**Lo que hay que retener.** El paper es corto, no introduce ninguna operación nueva y no reporta ni un solo conteo de parámetros ni un tiempo de cómputo. Su valor no está en la novedad técnica sino en haber fijado un **orden canónico de composición** que la comunidad de audio copió durante los siguientes cinco años, y en haber demostrado que ese orden no es arbitrario: se deriva de un argumento sobre dónde una RNN es demasiado superficial.

---

## 2. Contexto: el estado del ASR en 2014-2015

### 2.1. El reemplazo ya había ocurrido

Para 2012, con Hinton et al. en *IEEE Signal Processing Magazine* (referencia [1] del paper, firmada por cuatro grupos industriales incluido Google), el modelo acústico basado en DNN ya había desplazado al GMM en reconocimiento de vocabulario extenso. La arquitectura de producción era estable y aburrida: un MLP de 6 capas de 1024 unidades que recibe un stack de features log-mel con contexto (típicamente ±10 a ±20 frames aplanados en un vector) y emite una posterior sobre miles de estados dependientes de contexto (CD states) de un HMM. En este paper esa configuración es literalmente el baseline DNN: "6 layers, 1,024 hidden units" con "a context of 20 past frames and 5 future frames" (Sección 4.1).

El punto importante para entender el resto: **nada de esto es end-to-end**. El DNN es solo el reemplazo de la función de verosimilitud del GMM dentro de un sistema HMM completo, con árbol de decisión fonético, léxico y modelo de lenguaje n-grama. Los 13,522 targets del paper (Sección 3) son hojas de ese árbol de decisión. Volveré a esto en Limitaciones porque es la fuente de malentendidos más común sobre este paper.

### 2.2. Las dos mejoras que competían

A partir de 2013 aparecieron dos maneras distintas de mejorar sobre el DNN, con argumentos independientes y sin contacto entre sí.

**La rama CNN.** Sainath et al., "Deep Convolutional Neural Networks for LVCSR", ICASSP 2013 (referencia [2], primera autora la misma que aquí), junto con el trabajo paralelo de Abdel-Hamid et al. El argumento: el espectrograma log-mel es una imagen con dos ejes de significado físico distinto, y aplanarlo en un vector para meterlo a un MLP tira a la basura esa estructura. Compartir pesos a lo largo del eje de frecuencia, y hacer max-pooling en ese eje, absorbe una fuente concreta de variabilidad: **el desplazamiento de las formantes entre hablantes**. La ganancia típica reportada en la literatura era de 3-5% relativo sobre el DNN, y en este paper el baseline CNN da 18.0 contra 18.4 del DNN (Tabla 1), o sea 2.2%.

**La rama LSTM.** Graves et al. 2013 en TIMIT y, decisivamente para este paper, Sak, Senior y Beaufays, "Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling", Interspeech 2014 (referencia [3]; Sak y Senior son coautores aquí). El argumento es ortogonal: el contexto que un DNN puede ver está limitado por el tamaño de la ventana que se le aplana en la entrada, y crecerla cuesta parámetros linealmente y no ayuda más allá de cierto punto. Un LSTM tiene memoria de longitud no acotada por construcción y consume **un solo frame a la vez**. La contribución de ingeniería de Sak et al. fue la LSTMP — LSTM con capa de proyección recurrente — que permite tener muchas celdas sin que la matriz recurrente explote, y entrenarla con ASGD distribuido a escala de miles de horas.

### 2.3. Por qué nadie las había combinado

Es la pregunta correcta y hay al menos cinco razones concretas, que no son "no se les ocurrió".

1. **Los contratos de entrada eran incompatibles.** La CNN de 2013 consumía una ventana grande y simétrica (aquí: 20 frames pasados y 5 futuros) y emitía **una** decisión. El LSTM de 2014 consumía **un frame** y su contexto venía de la recurrencia. Concatenar los dos exige decidir qué hace la CNN con el tiempo, y esa decisión no es obvia: si la CNN ya consume 25 frames y el LSTM se desenrolla 20 pasos, el contexto efectivo se multiplica y deja de estar claro qué está modelando quién. El paper lo mide y encuentra el límite: con $l = 20$ el WER **empeora** (Tabla 2), "likely since the LSTM is then unrolled for 20 time steps, so the total context processed by the LSTM is 40".

2. **La forma canónica de combinar modelos era el ensamble, no la arquitectura.** El paper cita explícitamente a Deng y Platt, "Ensemble Deep Learning for Speech Recognition", Interspeech 2014 (referencia [9]) donde "the three models were first trained separately and then the three outputs were combined through a combination layer". Esa era la práctica: entrenar CNN, DNN y RNN por separado, combinar posteriores o lattices. Es más fácil de operar, se paraleliza trivialmente y no requiere que ninguna pieza cambie. La contribución de CLDNN es explícitamente el contraste: "we combine CNNs, LSTMs and [DNNs] into one unified framework that is **trained jointly**".

3. **Entrenar LSTMs a escala era frágil.** La Sección 4.5 del paper es evidencia directa: con inicialización gaussiana de varianza $1/n_{\text{in}}$ el LSTM daba 18.0, y con inicialización uniforme en $[-0.02, 0.02]$ daba 17.7 — una mejora de 1.7% relativo por cambiar solo la inicialización, más grande que la ganancia entera de la CNN sobre el DNN en la Tabla 1. Apilar capas convolucionales debajo de algo tan sensible al condicionamiento era una apuesta razonable de evitar.

4. **Infraestructura.** Todo esto corre sobre DistBelief (referencia [16], Dean et al. NIPS 2012) con ASGD asíncrono sobre un cluster. Meter convoluciones, recurrencia truncada con BPTT y capas densas en un solo grafo entrenado asíncronamente no era un `nn.Sequential`; era trabajo de sistemas.

5. **No había un argumento teórico que dijera en qué orden ponerlas.** Y aquí está la contribución conceptual real del paper, que desarrollo en la sección siguiente.

---

## 3. La tesis de la complementariedad

### 3.1. El argumento del paper no es "apilemos tres cosas"

La lectura superficial —y la que transmite el slide de la clase— es que la CNN, el RNN y el MLP hacen cosas distintas y por lo tanto conviene tenerlas todas. El paper tiene un argumento bastante más específico, y viene de una sola referencia que cita cuatro veces: **Pascanu, Gulcehre, Cho y Bengio, "How to Construct Deep Recurrent Neural Networks", ICLR 2014** (referencia [4]).

La observación de ese paper es que una RNN tiene **tres transiciones separables**, y "profundidad" significa cosas distintas en cada una:

- entrada → estado oculto ($x_t \to h_t$),
- estado oculto → estado oculto ($h_{t-1} \to h_t$),
- estado oculto → salida ($h_t \to y_t$).

Un LSTM estándar es profundo **solo en el tiempo**: las otras dos transiciones son afines de una sola capa. Sainath et al. lo citan literalmente dos veces:

> "One issue with LSTMs is that the temporal modeling is done on the input feature $x_t$ (i.e., log-mel feature). However, higher-level modeling of $x_t$ can help to disentangle underlying factors of variation within the input" (Sección 1).

> "in LSTMs the mapping between $h_t$ and output $y_t$ is also not deep, meaning there is no intermediate nonlinear hidden layer" (Sección 1).

Entonces la CLDNN no es "CNN + LSTM + DNN". Es **un LSTM al que se le rellenaron sus dos transiciones superficiales**: la CNN es la transición entrada→oculto profunda, el DNN es la transición oculto→salida profunda. Esa reformulación explica de una todas las decisiones de diseño del paper: por qué ese orden y no otro, por qué dos capas de cada cosa y no seis, por qué la ganancia satura (Tabla 4: 0→18.0, 1→17.8, 2→17.6, 3→17.6). Una vez que la transición dejó de ser afín, agregar más capas rinde poco.

Vale la pena decirlo explícitamente porque cambia cómo se lee la arquitectura: **el LSTM es el modelo, la CNN y el DNN son adaptadores de sus interfaces.**

### 3.2. El segundo argumento: la receta ya existía en GMM/HMM

El paper da una segunda justificación del orden, distinta y complementaria, y es históricamente la más elegante:

> "state-of-the-art GMM/HMM systems perform speaker adaptation, using techniques such as vocal tract length normalization (VTLN) and feature-space maximum likelihood linear regression (fMLLR), **before** performing temporal modeling via HMMs" (Sección 1).

O sea: **normalizar primero, modelar el tiempo después** era ya el orden canónico de veinte años de ASR estadístico. VTLN estima un factor de warping $\alpha$ por hablante y reescala el eje de frecuencia $f \to \alpha f$ para compensar diferencias de longitud del tracto vocal; fMLLR aplica una transformación afín en el espacio de features. Ambos son normalizaciones que se aplican **antes** del HMM. CLDNN sustituye esa normalización explícita y estimada por EM por una **normalización aprendida y discriminativa**: las capas convolucionales. La cita [5] (Mohamed, Hinton y Penn, ICASSP 2012) refuerza la analogía: "it has been shown that CNNs learn speaker-adapted/discriminatively trained features".

Esta es la clave que hace que CLDNN se sienta inevitable en retrospectiva: **no inventó una jerarquía, recapituló la que ya funcionaba, con módulos entrenables.**

### 3.3. La auditoría: ¿"invarianza a traslación" es lo que dice el paper?

El slide de la clase 39 justifica la CNN con dos propiedades: *span local del filtro* e *invarianza a traslación*. Hay que ser preciso, porque la primera es correcta y la segunda es una sustitución del vocabulario del paper por el del canon de visión por computador.

**Hecho verificable:** la palabra `invariance` (y cualquier variante, `invariant`, `invariancia`) aparece **cero veces** en el texto completo del paper. Lo verifiqué con `grep -icE "invarian"` sobre la extracción completa del PDF: 0 coincidencias. El vocabulario del paper es otro: `variance` (4 veces), `variation` / `variations` (8 veces). Las frases son "reduce frequency variance", "reduce spectral variation", "remove variation in the input".

¿Es la misma afirmación con otras palabras? **Casi, pero no del todo, y la diferencia importa.**

- **Compartir pesos produce equivarianza, no invarianza.** El paper dice "these filters are shared across the entire time-frequency space" (Sección 2.1). Si la entrada se desplaza $k$ bins en frecuencia, la salida de la convolución se desplaza $k$ bins: eso es **equivarianza**. La invarianza local aparece solo donde hay pooling, y en este paper el pooling está en un solo lugar: max-pooling no solapado de tamaño 3, **solo en frecuencia**, y **solo en la primera capa** ("A pooling size of 3 was used for the first layer, and no pooling was done in the second layer", Sección 2.1).
- **"Reducir la variación" es un objetivo, "invarianza a traslación" es un mecanismo.** El paper apunta al objetivo y lo ancla en el argumento VTLN/fMLLR: quiere que la representación sea estable frente a cambios de **hablante**, no frente a traslaciones arbitrarias. Son cosas distintas, y la Sección 3.4 muestra que solo coinciden aproximadamente.
- La formulación del profesor no es incorrecta —el mecanismo es genuinamente el de una CNN— pero es la versión de libro de texto. La versión del paper es más específica: *la CNN es un VTLN aprendido*.

### 3.4. El punto crítico: ¿en qué eje se aplica la invarianza?

Este es el detalle que más se malinterpreta al pasar de visión a audio, y el paper lo resuelve en una sola frase que es fácil leer sin darse cuenta de lo que implica: "**pooling in frequency only is performed**" (Sección 2.1).

**Por qué la invarianza en frecuencia es deseable.** El tracto vocal de un adulto varía típicamente entre unos 13 y 18 cm. La teoría fuente-filtro dice que las frecuencias de resonancia (formantes) escalan aproximadamente de forma **inversamente proporcional** a la longitud del tracto: una persona con tracto más corto tiene todas sus formantes desplazadas hacia arriba por un factor multiplicativo $\alpha$ aproximadamente constante. Eso es exactamente lo que VTLN modela con $f \to \alpha f$.

Ahora bien, un desplazamiento **multiplicativo** en Hz solo se convierte en un desplazamiento **aditivo** —o sea, en una traslación, que es lo que una CNN puede absorber— si el eje es logarítmico. La escala mel es
$$m(f) = 2595 \, \log_{10}\!\left(1 + \frac{f}{700}\right),$$
que es aproximadamente **lineal** para $f \ll 700$ Hz y aproximadamente **logarítmica** para $f \gg 700$ Hz. Por lo tanto:
$$m(\alpha f) - m(f) \;\approx\; 2595 \log_{10} \alpha \quad \text{solo cuando } f \gg 700\ \text{Hz}.$$

**Derivación propia** (el paper no da el rango de frecuencia de su banco de filtros; asumo el típico de Google en esa época, 125–7500 Hz, y los 40 bins que sí especifica). El rango total en mels es $m(7500) - m(125) \approx 2773 - 185 = 2588$ mels, o sea unos **64.7 mels por bin**. Entonces:

| Factor de warping $\alpha$ | $\Delta m$ (mels) | Desplazamiento en bins mel |
|---|---|---|
| 1.05 | 55 | 0.85 |
| 1.10 | 107 | **1.66** |
| 1.20 | 206 | **3.18** |

Una diferencia de tracto vocal del 10% —del orden de la que separa a un hombre adulto de una mujer adulta— produce un desplazamiento de **menos de dos bins mel** en la región logarítmica. Un pooling de tamaño 3 en frecuencia cubre exactamente ese rango. **El "3" del paper no es un número redondo cualquiera: es del orden de magnitud correcto para la física del problema.** Esto es una reconstrucción mía, no una justificación que el paper haga; el paper simplemente cita [11] (Sainath et al., ASRU 2013) donde la estrategia de pooling se había estudiado empíricamente. Pero el número cierra.

Dos matices que conviene no perder:

- La invarianza es **aproximada** y se degrada abajo de ~700 Hz, justo donde vive F1. En la región lineal de la escala mel, el warping VTLN **no** es una traslación sino un escalamiento, y ninguna cantidad de max-pooling lo absorbe. Esa es una limitación estructural del enfoque, no un defecto de implementación.
- Con un solo pooling de tamaño 3 en una red de dos capas convolucionales, la invarianza total es modesta. El paper no está construyendo un pipeline profundo de invarianza al estilo de una CNN de ImageNet con cinco etapas de pooling; está haciendo una normalización superficial y localizada.

**Por qué la invarianza en tiempo sería destructiva.** Aquí hay tres razones, y son acumulativas:

1. **La tarea es una clasificación por frame.** El modelo acústico híbrido debe emitir $P(s \mid x)$ sobre los 13,522 estados CD **para cada frame de 10 ms**, porque esa posterior alimenta la búsqueda de Viterbi en el HMM. Hacer pooling con stride en tiempo diezmaría la tasa de frames y rompería el contrato con el decodificador. (Hacer pooling en tiempo *sin* stride, o sea solapado, no diezma pero sí difumina, que es el punto 2.)
2. **La información discriminativa es la transición, no el estado.** La diferencia entre /b/ y /d/ está esencialmente en la **dirección de la transición de F2** en los primeros ~30 ms tras la explosión; la diferencia entre una oclusiva y una fricativa está en la abruptez del ataque. Un max-pooling temporal de tamaño 3 promedia-por-máximo tres frames de 10 ms, que es precisamente la escala de esos eventos. Se estaría destruyendo la señal que se quiere clasificar.
3. **El tiempo ya tiene dueño en esta arquitectura.** Toda la tesis del paper es que el LSTM se ocupa del eje temporal. Tirar resolución temporal antes de que el LSTM la vea es sabotear la etapa siguiente. Esto es lo que hace que la asignación de ejes sea coherente: **la CNN se queda con la frecuencia, el LSTM se queda con el tiempo.**

Un cuarto matiz que suele confundirse: **la CNN sí convoluciona sobre el tiempo.** Los filtros son "frequency-time" de $9\times9$ y $4\times3$, o sea que el segundo tiene extensión 3 en el eje temporal. Convolucionar sobre el tiempo da **equivarianza** temporal (deseable: es compartir pesos, y ahorra parámetros) y modela dinámica local. Lo que no hay es **pooling** temporal, que es lo que daría invarianza. Convolución sobre un eje ≠ invarianza sobre ese eje, y este paper es un caso limpio donde la distinción tiene consecuencias.

| Eje | Compartición de pesos (equivarianza) | Pooling (invarianza local) | Consecuencia |
|---|---|---|---|
| Frecuencia | Sí | **Sí**, tamaño 3, no solapado, capa 1 | Absorbe warping VTLN |
| Tiempo | Sí | **No** | Preserva resolución de 10 ms para el HMM y para el LSTM |

---

## 4. La arquitectura en detalle

Todo lo que sigue viene de la Sección 2.1 y de la Sección 3 del paper, salvo lo marcado como derivación propia.

### 4.1. Entrada

Cada frame $x_t$ es un vector **log-mel de 40 dimensiones**, calculado cada **10 ms** (Secciones 2.1 y 3). El paper **no especifica** la longitud de la ventana de análisis ni el rango de frecuencias del banco de filtros; solo el hop de 10 ms y la dimensión 40.

La entrada a la red es $[x_{t-l}, \dots, x_{t+r}]$, con $l$ frames de contexto a la izquierda y $r$ a la derecha. Para la CLDNN se fija **$r = 0$** por una razón explícita de producción:

> "In order to ensure that the LSTM does not see more than 5 frames of future context, which would increase the decoding latency, we set $r = 0$ for CLDNNs." (Sección 2.1)

Los "5 frames de futuro" vienen de otra parte: la etiqueta de salida se **retrasa 5 frames**, que es el truco estándar de Sak et al. para darle al LSTM unidireccional un poco de lookahead. Sumado, el sistema tiene 50 ms de latencia de lookahead y ni un milisegundo más — un requisito duro para Voice Search.

El mejor $l$ resulta ser **10** (Tabla 2): con $l=0$ da 17.8, con $l=10$ da 17.6, con $l=20$ da 17.9. Así que la entrada a la CNN en la mejor configuración es una "imagen" de $40 \times 11$ (40 bins de frecuencia × 11 frames de tiempo).

### 4.2. Bloque convolucional

- **2 capas convolucionales**, cada una con **256 mapas de features**.
- Filtro **$9\times9$ frecuencia-tiempo** en la primera capa.
- Filtro **$4\times3$** en la segunda capa. (Nótese: **4×3**, no 4×4. Verificado literalmente en el PDF, línea 83 de la extracción: "followed by a 4x3 filter for the second convolutional layer". Este es el punto de discrepancia principal con el slide de la clase; ver Sección 11.)
- Los filtros se comparten "across the entire time-frequency space", o sea **compartición completa de pesos** en ambos ejes, no la *limited weight sharing* de la literatura de CNN para ASR de 2012.
- **Max-pooling no solapado, solo en frecuencia, tamaño 3, solo tras la primera capa.** Cero pooling en la segunda.

El paper **no especifica el padding**, y esto tiene consecuencias reales: no se pueden reconstruir las formas exactas de los tensores. Peor aún, la Tabla 2 reporta una configuración con $l = 0$, o sea una entrada de **un solo frame en el eje temporal**, a la que se le aplica un filtro de extensión temporal 9. Eso exige padding masivo o alguna convención no documentada. Lo señalo como ambigüedad genuina del paper, no como error de lectura.

### 4.3. La capa lineal de reducción de dimensión

Esta es la pieza que más se malinterpreta y la que el paper justifica con más claridad:

> "The dimension of the last layer of the CNN is large, due to the number of **feature-maps × time × frequency context**. Thus, we add a linear layer to reduce feature dimension, before passing this to the LSTM layer... In [12] we found that adding this linear layer after the CNN layers allows for a reduction in parameters with no loss in accuracy. In our experiments, we found that reducing the dimensionality, such that we have **256 outputs** from the linear layer, was appropriate." (Sección 2.1)

Dos cosas críticas:

1. **Es una capa lineal sobre el tensor aplanado completo**, no una convolución $1\times1$. El paper es explícito: la dimensión grande viene del producto *mapas × tiempo × frecuencia*, y la capa lineal la colapsa a 256. Una convolución $1\times1$ reduciría solo el eje de canales y dejaría intactos los ejes de frecuencia y tiempo. Son operaciones distintas con conteos de parámetros distintos. (El slide de la clase dice "convolución $1\times1$"; ver Sección 11.)
2. **No hay no-linealidad.** El paper dice "linear layer", y la Figura 1 la etiqueta `linear layer / dim red`. Es una factorización de rango bajo pura, exactamente el mismo objeto que la capa de proyección de la LSTMP: no aumenta capacidad expresiva, solo evita que la matriz de entrada del LSTM sea gigante.

**Derivación propia del ahorro.** Asumiendo convoluciones válidas (sin padding) sobre la entrada $40 \times 11$:

| Etapa | Forma de salida | Cómputo |
|---|---|---|
| Entrada | $1 \times 40 \times 11$ | — |
| Conv1 $9\times9$, 256 mapas | $256 \times 32 \times 3$ | $40-9+1=32$; $11-9+1=3$ |
| MaxPool $3\times1$ en frecuencia | $256 \times 10 \times 3$ | $\lfloor 32/3 \rfloor = 10$ |
| Conv2 $4\times3$, 256 mapas | $256 \times 7 \times 1$ | $10-4+1=7$; $3-3+1=1$ |
| Aplanado | **1792** | $256 \times 7 \times 1$ |
| Capa lineal | **256** | matriz $1792 \times 256$ |

Sin la capa lineal, la matriz de entrada del primer LSTM (832 celdas, cuatro compuertas) sería
$$4 \times 832 \times 1792 = 5{,}963{,}776 \ \text{parámetros}.$$
Con ella, es la suma de la lineal y la nueva matriz de entrada:
$$\underbrace{1792 \times 256}_{458{,}752} + \underbrace{4 \times 832 \times 256}_{851{,}968} = 1{,}310{,}720.$$
**Ahorro: ~4.65M de parámetros**, que como se verá en la Sección 4.6 es cerca del 20% del modelo completo. La afirmación del paper de "reduction in parameters with no loss in accuracy" es, cuantitativamente, sustancial. La forma exacta depende del padding que el paper no documenta, así que tómese como orden de magnitud.

### 4.4. Bloque LSTM

Siguiendo la referencia [3] (Sak et al., Interspeech 2014):

- **2 capas LSTM**.
- **832 celdas** por capa.
- **Capa de proyección de 512 unidades** para reducción de dimensionalidad (la arquitectura LSTMP).
- Desenrollado de **20 pasos temporales** para BPTT truncado.
- El paper señala, citando [3], que "adding extra LSTM layers to this configuration was not found to help" (Sección 4.1).

La capa de proyección merece una nota porque es fácil pasarla por alto: en una LSTMP, la salida recurrente que se realimenta no es $h_t \in \mathbb{R}^{832}$ sino $r_t = W_{\text{proj}} h_t \in \mathbb{R}^{512}$. Eso rompe el escalamiento cuadrático de la matriz recurrente: en vez de $4 \times 832 \times 832$ se paga $4 \times 832 \times 512 + 832 \times 512$. Es exactamente el mismo truco de rango bajo que la capa lineal de la Sección 4.3, aplicado al eje recurrente. La CLDNN usa **dos** factorizaciones de rango bajo por las mismas razones, en dos lugares distintos.

### 4.5. Bloque DNN

- **Capas totalmente conectadas de 1,024 unidades ocultas** cada una (Sección 2.1).
- **2 capas** en la configuración final, porque la Tabla 4 muestra que el rendimiento satura ahí.
- Salida softmax sobre **13,522 targets CD** (Sección 3).

### 4.6. Tabla capa por capa

Reconstrucción completa para la configuración final de la Sección 4.4 del paper ($l=10$, $r=0$). **Las columnas de forma y de parámetros son derivación propia**, con las asunciones de padding de la Sección 4.3; el paper no reporta ni formas intermedias ni conteos de parámetros en ningún lugar.

| # | Capa | Hiperparámetros (del paper) | Forma de salida (derivada) | Parámetros (derivados) |
|---|---|---|---|---|
| 0 | Entrada | log-mel 40D, hop 10 ms, $l=10$, $r=0$ | $1 \times 40 \times 11$ | — |
| 1 | Conv 2D | 256 mapas, filtro $9\times9$ (freq × tiempo) | $256 \times 32 \times 3$ | 20,992 |
| 2 | MaxPool | tamaño 3, **solo frecuencia**, no solapado | $256 \times 10 \times 3$ | 0 |
| 3 | Conv 2D | 256 mapas, filtro $4\times3$ | $256 \times 7 \times 1$ | 786,688 |
| 4 | Lineal (dim red) | 256 salidas, sin no-linealidad | $256$ | 458,752 |
| 5 | LSTMP #1 | 832 celdas, proyección 512 | $512$ | ~2.99M |
| 6 | LSTMP #2 | 832 celdas, proyección 512 | $512$ | ~3.84M |
| 7 | FC #1 | 1,024 unidades | $1024$ | 525,312 |
| 8 | FC #2 | 1,024 unidades | $1024$ | 1,049,600 |
| 9 | Softmax | 13,522 targets CD | $13522$ | ~13.86M |
| | **Total** | | | **~23.5M** |

Dos observaciones que salen solo de hacer la cuenta:

- **La capa de salida es el 59% del modelo.** Los 13,522 estados CD × 1,024 unidades = 13.85M parámetros, más que todo el resto junto. Cuando se dice que CLDNN "es una red grande", la mayor parte de ese tamaño no es CLDNN: es el árbol de decisión fonético del sistema HMML. Esto reaparece en Limitaciones.
- **El bloque convolucional es barato en parámetros (~1.27M) pero no en cómputo.** Y el cómputo es el punto sutil de la Sección 13: la CNN se evalúa **una vez por frame de salida** sobre una ventana de 11 frames, así que cada frame de entrada se procesa hasta 11 veces si se implementa ingenuamente.

---

## 5. Las conexiones de salto: qué son, qué aportan y qué no

La Sección 2.2 del paper introduce lo que llama "multi-scale additions", dos flujos adicionales dibujados con línea punteada en la Figura 1. La motivación declarada viene de visión por computador: "Each CNN, LSTM and DNN block captures information about the input representation at different scales [10]", donde [10] es Sermanet y LeCun, *Traffic sign recognition with multi-scale convolutional networks*, IJCNN 2011.

### 5.1. Flujo (1): $x_t$ crudo concatenado a la salida de la CNN, antes del LSTM

**Qué hace.** La CNN consume $[x_{t-10}, \dots, x_t]$ y emite un vector de 256 dimensiones que resume esa ventana larga. Al LSTM se le pasa la **concatenación** de ese vector con el frame crudo $x_t$ (40 dimensiones), o sea 296 dimensiones.

**Por qué.** El argumento del paper es de escalas: el LSTM desenrollado 20 pasos, alimentado por una CNN que ve 11 frames, consume un contexto efectivo de "$20 + l$" frames; el frame crudo $x_t$ es la escala corta, sin procesar, sin contexto. Hay información en el frame actual que la CNN puede haber promediado o normalizado. Además, el paper nota que el trabajo original de LSTM [3] consumía exactamente eso: features de corto plazo sin contexto.

**Qué aporta, medido.** Tabla 7: de 17.0 a **16.8**, o sea **1.2% relativo**. El paper lo reporta como "an additional 1% relative improvement".

**Costo.** Derivación propia: la matriz de entrada del primer LSTM crece de $4 \times 832 \times 256$ a $4 \times 832 \times 296$, o sea **+133,120 parámetros**, un 0.57% del modelo. El paper afirma que "our combination of short and long-term features results in a negligible increase in the number of network parameters"; la cuenta confirma la afirmación.

### 5.2. Flujo (2): salida de la CNN concatenada a la del LSTM, antes del DNN

**Qué hace.** El bloque DNN recibe la concatenación de la salida del LSTM (512) con la salida de la CNN (256), o sea 768 dimensiones. La motivación: "we explore if there is complementarity between modeling the output of the CNN temporally with an LSTM, as well as discriminatively with a DNN".

**Qué aporta, medido.** Tabla 7: de 17.0 a **17.0**. **Nada.** El paper es honesto y directo:

> "Table 7 indicates that this does not yield gains over the CLDNN alone. This indicates that temporal processing of CNN features using the LSTM is sufficient, and more information is not gained by additionally passing CNN features into the DNN." (Sección 4.6)

Y en consecuencia lo **abandona** para todos los experimentos de la Sección 5: "we just include results passing short and long-term features into the CNN, and omit passing the CNN into both the LSTM and DNN, as only the first technique showed gains".

Esto merece énfasis porque se cita mal con frecuencia: **de las dos conexiones de salto del paper, una funciona (marginalmente) y la otra no funciona en absoluto**, y el propio paper la descarta. Cualquier descripción de CLDNN que presente ambas como parte de la arquitectura final está describiendo la Figura 1 (que las muestra ambas, en punteado) y no los resultados.

### 5.3. ¿Es esto un antecedente de las conexiones residuales?

Es una pregunta legítima y la respuesta honesta tiene tres partes.

**Cronología.** El preprint está compilado en octubre de 2014 y el paper es de ICASSP 2015 (abril). *Highway Networks* (Srivastava et al.) es de mayo de 2015; *Deep Residual Learning* (He et al.) es de diciembre de 2015. **CLDNN precede a ambos en publicación.**

**Pero el mecanismo es distinto, y la diferencia no es cosmética.**

| | CLDNN (2015) | ResNet (2015) | DenseNet (2016) |
|---|---|---|---|
| Operación | **Concatenación** | **Suma** | **Concatenación** |
| Dimensiones | Pueden diferir libremente | Deben coincidir (o proyectarse) | Pueden diferir |
| Efecto en el conteo de parámetros | Crece la capa receptora | Cero | Crece la capa receptora |
| Motivación declarada | **Multi-escala** (fusión de información) | **Optimización** (flujo de gradiente, degradación con la profundidad) | Reutilización de features |
| Profundidad involucrada | 2 saltos en una red de 9 capas | Cada bloque, en redes de 50-152 capas | Todos contra todos por bloque |

Por mecanismo, CLDNN es antecedente de **DenseNet**, no de ResNet: concatenación, no suma. Y por motivación no es antecedente de ninguno de los dos: el paper habla de escalas de representación, nunca de gradientes, nunca de degradación con la profundidad, nunca de facilitar la optimización. Es un argumento de **fusión de features**, del linaje Sermanet–LeCun 2011, que a su vez viene de las conexiones salteadas de las redes de los 90.

**Lo que sí es correcto decir.** CLDNN pertenece al momento histórico en que la comunidad estaba descubriendo, por caminos independientes, que **una capa no tiene por qué recibir su entrada solo de la capa inmediatamente anterior**. Ese es el patrón compartido. Atribuirle a CLDNN la intuición residual específica (identidad + residuo, para que el bloque tenga que aprender solo la desviación) es sobreleer el paper. Y hay una ironía útil: la conexión de salto que CLDNN mide y **descarta** (flujo 2) es la que topológicamente más se parece a un salto residual de largo alcance.

---

## 6. Detalles de entrenamiento

Todo de la Sección 3, salvo indicación contraria.

**Features.** Log-mel filterbank de 40 dimensiones, cada 10 ms, para **todos** los modelos incluyendo baselines. Sin deltas, sin CMVN mencionado, sin normalización de hablante explícita — coherente con la tesis de que la CNN hace ese trabajo.

**Contexto asimétrico.** Los baselines CNN y DNN reciben "a context of 20 past frames and 5 future frames" (Sección 4.1): asimétrico y sesgado al pasado. La CLDNN usa $l=10$, $r=0$: **totalmente causal en la entrada**. El único lookahead del sistema es el retraso de 5 frames en la etiqueta de salida. Este es un diseño impuesto por la latencia de decodificación, no por la calidad; el paper lo dice explícitamente.

**Función de pérdida.** Entropía cruzada por frame contra un alineamiento forzado ("Unless otherwise indicated, all neural networks are trained with the cross-entropy criterion"). Los experimentos de la Sección 5 añaden una segunda fase de **entrenamiento de secuencia** siguiendo [17] (Heigold et al., ICASSP 2014) y [20] (Kingsbury, ICASSP 2009), "a strategy which has shown to give consistent gains over CE training".

**Optimización.** **ASGD distribuido** (descenso de gradiente estocástico asíncrono) según [16], o sea DistBelief. El entrenamiento de secuencia usa el ASGD distribuido de [17].

**Inicialización.** Aquí hay dos regímenes distintos y confundirlos hace irreproducibles los números:

- Capas **CNN y DNN**: Glorot-Bengio, siempre.
- Capas **LSTM**: gaussiana con varianza $1/n_{\text{in}}$ en los experimentos iniciales; **uniforme en $[-0.02, 0.02]$** a partir de la Sección 4.5 y en adelante.

El razonamiento de por qué la inicialización gaussiana era mala está en la Sección 4.5 y cita a Pascanu et al. ICML 2013: la gaussiana "produces eigenvalues of the initial recurrent network which are close to zero, thus increasing the chances for vanishing gradients". La inicialización uniforme baja el LSTM de 18.0 a 17.7 y la CLDNN de 17.3 a 17.0 (Tabla 6).

**Learning rate.** "chosen specific to each network, and is chosen to be the largest value such that training remains stable. Learning rates are exponentially decayed." O sea: sintonizado por red, sin valores reportados. Esto es un problema de reproducibilidad que discuto en Limitaciones.

**Unrolling.** 20 pasos, BPTT truncado. La Tabla 3 verifica que 30 pasos **empeora** (18.2 contra 18.0).

**Datos.** Tres conjuntos, todos anonimizados y transcritos a mano, "representative of Google's speech traffic":

| Conjunto | Entrenamiento | Test | Nota |
|---|---|---|---|
| Mediano limpio | 300k enunciados (~200 h) | 30,000 enunciados (20 h) limpios | Todos los experimentos de la Sección 4 |
| Grande limpio | 3M enunciados (2,000 h) | mismo test limpio | Sección 5 |
| Grande ruidoso | 3M enunciados (2,000 h) | test limpio corrompido, condiciones emparejadas | Sección 5 |

El conjunto ruidoso se genera **artificialmente**: "created by artificially corrupting clean utterances using a room simulator, adding varying degrees of noise and reverberation, such that the overall SNR is between 5dB to 30dB. The noise sources are from YouTube and daily life noisy environmental recordings." Es simulación de sala, no grabación real en condiciones ruidosas.

**Advertencia del propio paper.** "It is important to note that the training and test sets used in this paper are different than those in [3], and therefore numbers cannot directly be compared." O sea: los WER de este paper **no** son comparables con los de Sak et al. 2014 aunque el baseline LSTM sea nominalmente el mismo modelo.

---

## 7. Experimentos y resultados

### 7.1. Conjunto de 200 horas (Sección 4)

Todas las tablas de esta sección son sobre 200 h de entrenamiento limpio, evaluación sobre el test limpio, entrenamiento CE únicamente.

**Tabla 1 — Baselines individuales** (inicialización gaussiana en el LSTM):

| Método | WER |
|---|---|
| DNN | 18.4 |
| CNN | 18.0 |
| LSTM | 18.0 |

**Tabla 2 — CNN+LSTM contra DNN+LSTM, variando el contexto izquierdo:**

| Contexto de entrada | Pasos de desenrollado | WER CNN | WER DNN |
|---|---|---|---|
| $l=0, r=0$ | 20 | 17.8 | 18.2 |
| $l=10, r=0$ | 20 | **17.6** | 18.2 |
| $l=20, r=0$ | 20 | 17.9 | 18.5 |

Dos lecturas. Primera: el óptimo de contexto está en 10 y crecerlo a 20 **empeora**, con la explicación del paper de que el contexto total procesado por el LSTM llegaría a 40. Segunda: la CNN gana al DNN en las tres filas, o sea que "the benefits of CNNs over DNNs continue to hold even when combined with LSTMs" — la ganancia de la convolución no queda absorbida por la recurrencia.

**Tabla 3 — Control: ¿es solo contexto extra?**

| Método | WER |
|---|---|
| LSTM, $l=0$, unroll=20 | 18.0 |
| LSTM, $l=10$, unroll=20 | 18.0 |
| LSTM, $l=0$, unroll=30 | 18.2 |

Este es el experimento metodológicamente más importante del paper y es fácil pasarlo por alto. Descarta la explicación aburrida: *la CLDNN no gana porque le den más frames*. Darle al LSTM exactamente el mismo contexto $l=10$ que recibe la CNN no mueve el WER en absoluto (18.0 → 18.0), y darle más recurrencia lo empeora. **La ganancia viene del procesamiento convolucional de esos frames, no de su disponibilidad.**

**Tabla 4 — Cuántas capas DNN después del LSTM:**

| # capas DNN | WER |
|---|---|
| 0 | 18.0 (LSTM) |
| 1 | 17.8 |
| 2 | **17.6** |
| 3 | 17.6 |

**Tabla 5 — La composición completa:**

| Método | WER | Delta absoluto vs LSTM |
|---|---|---|
| LSTM | 18.0 | — |
| CNN+LSTM | 17.6 | −0.4 |
| LSTM+DNN | 17.6 | −0.4 |
| **CLDNN** | **17.3** | **−0.7** |

Cuantificación de la complementariedad: si las dos contribuciones fueran perfectamente aditivas, se esperaría 18.0 − 0.4 − 0.4 = **17.2**. Se observa 17.3. O sea que las ganancias son **~87% aditivas**: se solapan un poco, pero muy poco. Ese es el contenido empírico de la palabra "complementary" en este paper, y es un resultado más fuerte de lo que suele reconocerse. (El paper no hace esta cuenta; es derivación mía sobre sus números.)

**Tabla 6 — Inicialización:**

| Método | WER init gaussiana | WER init uniforme |
|---|---|---|
| LSTM | 18.0 | 17.7 |
| CLDNN | 17.3 | 17.0 |

Este es el control anti-escepticismo del paper, y el argumento está bien planteado: *si el LSTM estuviera mejor inicializado, ¿harían falta las CNN?* Respuesta: sí, la brecha se preserva casi exactamente (0.7 puntos absolutos en ambos regímenes, ~4% relativo en ambos).

**Tabla 7 — Adiciones multi-escala** (ya sobre init uniforme):

| Método | WER |
|---|---|
| LSTM | 17.7 |
| CLDNN, feature de largo plazo al LSTM | 17.0 |
| + feature de corto plazo al LSTM | **16.8** |
| + CNN al LSTM y al DNN | 17.0 |

### 7.2. Conjunto de 2,000 horas (Sección 5)

Aquí solo se reportan tres modelos, y ya sin la conexión de salto (2) que no funcionó. Se añade la columna de entrenamiento de secuencia.

**Tabla 8 — 2,000 h limpio, test limpio:**

| Método | WER-CE | WER-Seq |
|---|---|---|
| LSTM | 14.6 | 13.7 |
| CLDNN | 14.0 | 13.1 |
| multi-escala CLDNN | **13.8** | **13.1** |

**Tabla 9 — 2,000 h ruidoso, test ruidoso:**

| Método | WER-CE | WER-Seq |
|---|---|---|
| LSTM | 20.3 | 18.8 |
| CLDNN | 19.4 | **17.4** |
| multi-escala CLDNN | **19.2** | **17.4** |

### 7.3. Qué es grande y qué es marginal

Todos los relativos son cálculo mío sobre los WER del paper.

| Comparación | Absoluto | Relativo | Veredicto |
|---|---|---|---|
| LSTM → CLDNN, 2000 h ruidoso, secuencia | −1.4 | 7.4% | **Grande.** El mejor resultado del paper, y en la condición más difícil |
| LSTM → CLDNN, 2000 h limpio, CE | −0.6 | 4.1% | **Sólido**, consistente con las 200 h |
| LSTM → CLDNN, 2000 h ruidoso, CE | −0.9 | 4.4% | **Sólido** |
| LSTM → CLDNN, 200 h, cualquier init | −0.7 | ~4% | **Sólido y robusto** a la inicialización |
| CLDNN → multi-escala, 2000 h ruidoso, CE | −0.2 | 1.0% | **Marginal** |
| CLDNN → multi-escala, 200 h | −0.2 | 1.2% | **Marginal** |
| CLDNN → multi-escala, 2000 h limpio, CE | −0.2 | 1.4% | **Marginal** |
| CLDNN → multi-escala, **después de secuencia**, ambos conjuntos | 0.0 | **0%** | **Desaparece** (13.1 vs 13.1; 17.4 vs 17.4) |

La última fila es la más interesante y el paper no la comenta: **el aporte multi-escala se evapora completamente tras el entrenamiento de secuencia**, en las dos condiciones. La lectura razonable es que el entrenamiento de secuencia recupera por su cuenta la información que la conexión de salto de corto plazo estaba aportando. Que una ganancia arquitectónica desaparezca cuando se aplica el criterio de entrenamiento correcto es exactamente el tipo de resultado que en 2015 se reportaba poco y que hoy se consideraría una alerta metodológica.

Simétricamente, el aporte del **núcleo** CLDNN hace lo contrario: crece con el entrenamiento de secuencia en la condición ruidosa (4.4% en CE → 7.4% en secuencia). Esa asimetría es la mejor evidencia del paper de que la contribución central es real y la periférica es ruido de bajo nivel.

---

## 8. Ablations: qué pieza aporta qué

Esta sección separa deliberadamente **lo medido** de **lo atribuido**.

### 8.1. Lo que el paper mide

| Pregunta | Experimento | Resultado | Conclusión soportada |
|---|---|---|---|
| ¿La CNN antes del LSTM ayuda? | Tabla 5, CNN+LSTM vs LSTM | 17.6 vs 18.0 | **Sí**, 2.2% relativo |
| ¿El DNN después del LSTM ayuda? | Tabla 4 / 5 | 17.6 vs 18.0 | **Sí**, 2.2% relativo |
| ¿Se suman? | Tabla 5, CLDNN | 17.3 | **Sí**, ~87% aditivo |
| ¿CNN es mejor que DNN en esa posición? | Tabla 2, 3 filas | 17.8/17.6/17.9 vs 18.2/18.2/18.5 | **Sí**, en las tres |
| ¿Cuánto contexto izquierdo? | Tabla 2 | óptimo en $l=10$ | Existe un óptimo interior |
| ¿Es solo contexto extra? | Tabla 3 | LSTM $l=10$ = 18.0 = LSTM $l=0$ | **No.** Es el procesamiento |
| ¿Es solo mala inicialización del LSTM? | Tabla 6 | brecha idéntica en ambos regímenes | **No** |
| ¿Cuántas capas DNN? | Tabla 4 | satura en 2 | 2 |
| ¿Skip de corto plazo al LSTM? | Tabla 7 | 17.0 → 16.8 | **Sí**, pero ~1% y se anula tras secuencia |
| ¿Skip de CNN al DNN? | Tabla 7 | 17.0 → 17.0 | **No.** Descartada |
| ¿Escala a 2000 h? | Tablas 8, 9 | 4-7% relativo | **Sí** |
| ¿Sobrevive al ruido? | Tabla 9 | mejora más que en limpio | **Sí**, y aumenta |
| ¿Sobrevive al entrenamiento de secuencia? | Tablas 8, 9 | sí para el núcleo, no para multi-escala | Parcialmente |

### 8.2. Lo que el paper NO mide, pese a que suele atribuírsele

Esta lista importa tanto como la anterior.

- **El orden de las capas nunca se ablaciona.** No hay ni un solo experimento con LSTM antes de la CNN, ni con el DNN antes del LSTM, ni con las capas intercaladas. El orden C→L→D se justifica **por argumento** (Pascanu et al., la analogía VTLN/fMLLR→HMM) y se valida **solo indirectamente** por el hecho de que ese orden funciona. Cualquier afirmación del tipo "el paper demostró que este es el mejor orden" es falsa: el paper demostró que **este orden funciona mejor que sus componentes aislados**, que es una afirmación mucho más débil.
- **El número de capas convolucionales nunca se ablaciona.** El paper fija 2 capas de 256 mapas heredándolas de [2] (Sainath et al. 2013). No hay una tabla de 1 vs 2 vs 3 capas convolucionales, ni de 128 vs 256 vs 512 mapas. Contrastar con la Tabla 4, que sí barre el número de capas DNN. **La asimetría es reveladora**: el bloque DNN se sintonizó en este paper, el bloque CNN se importó.
- **La utilidad del pooling nunca se ablaciona.** Este es el hueco más llamativo dado el tema de la clase. No hay ningún experimento *sin* pooling, ni con pooling de tamaño 2 o 5, ni —crucialmente— **con pooling en tiempo**. La decisión de hacer pooling solo en frecuencia se toma por cita a [11] (Sainath et al., ASRU 2013) y por argumento físico, **no por evidencia presentada en este paper**. Es una decisión correcta y bien fundada, pero la evidencia está en otro trabajo.
- **Los tamaños de filtro $9\times9$ y $4\times3$ nunca se ablacionan.** También heredados de [2].
- **La capa lineal de reducción nunca se ablaciona aquí.** El paper dice "In [12] we found that adding this linear layer... allows for a reduction in parameters with no loss in accuracy" — la evidencia está en Sainath et al., Interspeech 2014, no en este paper.
- **No hay comparación con CNN+DNN sin LSTM.** El brief pregunta por esto explícitamente y la respuesta es que **el paper no la hace**. El baseline CNN de la Tabla 1 es "2 convolutional layers with 256 feature maps, and 4 fully connected layers of 1,024 hidden units", o sea que *es* una CNN+DNN sin LSTM — pero con **4** capas densas, no 2, y sin la capa lineal de reducción, y con contexto simétrico ±20/±5 en vez de $l=10, r=0$. Da 18.0, igual que el LSTM. No es una ablación controlada de la CLDNN: es un baseline de una arquitectura distinta. La comparación honesta que se puede extraer es "CNN+DNN da 18.0, CLDNN da 17.3, 3.9% relativo", pero con al menos tres variables confundidas.

**Resumen de la auditoría.** De las once decisiones de diseño de la arquitectura (número y tamaño de filtros, número de capas conv, tamaño y eje del pooling, capa de reducción, número y tamaño de LSTM, proyección, número y tamaño de capas DNN), este paper ablaciona **dos**: el número de capas DNN y el contexto izquierdo. Todo lo demás está heredado de tres papers previos del mismo grupo ([2], [3], [11], [12]). CLDNN es, en el sentido más literal, un paper de **integración**: su contribución es mostrar que tres recetas afinadas por separado se componen sin interferencia destructiva.

---

## 9. Limitaciones

### 9.1. Las que el paper reconoce

- **Latencia.** Reconocida y gestionada explícitamente: $r=0$ se elige para no agregar lookahead más allá del retraso de 5 frames de la etiqueta ("which would increase the decoding latency", Sección 2.1). El límite operacional es ~50 ms.
- **Contexto no comparable con trabajos previos.** "the training and test sets used in this paper are different than those in [3], and therefore numbers cannot directly be compared" (Sección 3).
- **La conexión de salto (2) no funciona**, reconocido y actuado (se descarta en la Sección 5).
- **Saturación con la profundidad.** Tabla 4 (DNN satura en 2 capas) y la nota de [3] de que capas LSTM extra no ayudan.

### 9.2. Las que no reconoce

**No es end-to-end, en ningún sentido moderno.** Esta es la más importante y la que más se distorsiona al citar el paper. La CLDNN es **el modelo acústico de un sistema híbrido DNN/HMM**. Concretamente:

- La salida son 13,522 **estados dependientes de contexto** de un HMM, o sea las hojas de un árbol de decisión fonético construido con métodos GMM.
- La pérdida CE requiere una **etiqueta por frame**, que solo puede venir de un **alineamiento forzado** producido por un sistema previo ya entrenado. El paper no menciona de dónde sale ese alineamiento, porque en 2015 era demasiado obvio para mencionarlo.
- El entrenamiento de secuencia de la Sección 5 opera sobre **lattices** ([20], Kingsbury), que exigen un decodificador, un léxico y un modelo de lenguaje.
- La inferencia requiere búsqueda de Viterbi sobre el HMM con un modelo de lenguaje n-grama.

O sea: hay un pipeline GMM/HMM completo antes y después de esta red. La frase "unified architecture" del abstract se refiere a unificar CNN+LSTM+DNN **entre sí**, no a unificar el reconocedor. Cuando alguien dice "CLDNN es una red end-to-end para audio", está equivocado. La transición real a end-to-end en ASR llega con CTC (que este paper no usa), LAS (2016) y RNN-T (2012 en teoría, en producción de Google en 2019, con Sainath entre los autores).

**Costo de entrenamiento y de inferencia: no reportados.** El paper no da ni un número de parámetros, ni FLOPs, ni tiempo de entrenamiento, ni número de máquinas, ni tiempo real de decodificación. Para un paper cuya decisión de diseño central ($r=0$) está motivada por **latencia**, no reportar latencia es una omisión notable. Y la aritmética de la Sección 4.6 sugiere que el bloque CNN, en una implementación ingenua, se evalúa una vez por frame de salida sobre una ventana de 11 frames: **~11× de trabajo redundante** frente a una implementación que comparta cómputo entre ventanas solapadas.

**Sin medida de dispersión, sin múltiples semillas, sin significancia.** Los deltas que sostienen la mitad del paper son de 0.2 a 0.4 puntos de WER absoluto sobre un test de 30,000 enunciados. No hay intervalos de confianza, ni test de significancia (MAPSSWE o similar, que era práctica estándar en ASR), ni varianza entre semillas. Dado que la Sección 4.5 demuestra que **cambiar solo la inicialización mueve el WER 0.3 puntos**, es difícil argumentar que un delta de 0.2 (el aporte multi-escala) esté por encima del ruido de entrenamiento. La desaparición de ese aporte tras el entrenamiento de secuencia (Sección 7.3) refuerza esa sospecha.

**Learning rates sintonizados por red, no reportados.** "the learning rate is chosen specific to each network, and is chosen to be the largest value such that training remains stable". Cada arquitectura recibió su propio ajuste, sin protocolo documentado y sin valores. Es la fuente de confusión clásica en comparaciones arquitectónicas: no se puede saber cuánta de la diferencia CLDNN-vs-LSTM es arquitectura y cuánta es presupuesto de sintonía.

**La separabilidad tiempo/frecuencia es una asunción, no un resultado.** La arquitectura asigna la frecuencia a la CNN y el tiempo al LSTM, con una zona gris: los filtros convolucionales *también* cubren tiempo ($9$ y $3$ frames respectivamente). Esa asignación descansa en tres supuestos no verificados:

1. Que la variabilidad relevante en frecuencia es **traslacional** — cierto solo aproximadamente, y solo en la región logarítmica de la escala mel (Sección 3.4).
2. Que un mismo kernel es apropiado en todas las bandas de frecuencia. La compartición completa de pesos ("shared across the entire time-frequency space") lo asume; la literatura de *limited weight sharing* de Abdel-Hamid et al. argumentaba lo contrario, porque la estructura espectral de un sonido nasal a 300 Hz y de una fricativa a 6 kHz no es la misma. El paper elige compartición completa sin discutir la alternativa.
3. Que las interacciones tiempo-frecuencia de largo alcance no importan. Un filtro de $9\times9$ seguido de uno de $4\times3$ tiene un campo receptivo acotado; toda dependencia entre una formante baja al inicio de una sílaba y una alta al final queda para el LSTM, que la ve solo a través del cuello de botella de 256 dimensiones.

Y hay un supuesto más profundo que ni este paper ni casi ninguno de su época discute: **el log-mel ya es una decisión irreversible**. Descarta la fase, fija una resolución tiempo-frecuencia, aplica una compresión perceptual no aprendida. La CNN opera sobre lo que sobrevive a esa transformación. Sainath misma atacó esto inmediatamente después con las CLDNN sobre forma de onda cruda (ver Sección 10).

**El conteo de targets domina el modelo.** Con 13,522 targets CD, la capa de salida es ~13.9M de ~23.5M parámetros (derivación de la Sección 4.6). Cualquier comparación de "tamaño" entre CLDNN, LSTM y DNN en este paper está dominada por una constante compartida que no tiene nada que ver con las arquitecturas comparadas.

**Robustez al ruido, evaluada en condiciones emparejadas.** La Tabla 9 entrena con ruido y evalúa con ruido del mismo simulador. No hay evaluación **desemparejada** (entrenar limpio, evaluar ruidoso), que es la que mide robustez de verdad. Lo que la Tabla 9 mide es que CLDNN aprovecha mejor el aumento de datos, no que sea intrínsecamente más robusta.

---

## 10. Impacto y legado

### 10.1. La línea directa: CLDNN en producción en Google

CLDNN no fue un experimento aislado; fue el punto de partida de una familia. Lo que sigue es contexto externo al paper y lo marco como tal.

- **Front-end aprendido.** Sainath, Weiss, Senior, Wilson y Vinyals, *"Learning the Speech Front-end With Raw Waveform CLDNNs"*, Interspeech 2015 — el mismo año. Reemplaza el log-mel por una capa convolucional 1D sobre la **forma de onda cruda** y muestra que la red aprende un banco de filtros con respuesta parecida a mel. Ataca directamente la limitación que señalé al final de la Sección 9.2, y el chasis que usa para hacerlo es precisamente la CLDNN.
- **Multicanal.** La línea de *factored multichannel raw waveform CLDNNs* (Sainath et al., 2016-2017) extiende la idea a arreglos de micrófonos, aprendiendo beamforming y modelo acústico conjuntamente. Es la tecnología detrás del reconocimiento de campo lejano de Google Home.
- **Keyword spotting.** El patrón conv+recurrente se volvió el estándar para detección de palabra clave con presupuesto de cómputo mínimo — el dominio donde la latencia estricta que CLDNN ya respetaba ($r=0$) es un requisito absoluto.
- **Verificación de hablante.** El extractor de embeddings de la línea GE2E de Google usa un stack recurrente con la misma lógica de "features locales primero, agregación temporal después".

El sesgo de selección es real y vale reconocerlo: el paper es de Google, los baselines son de Google, la infraestructura es de Google y el impacto que más se cita es dentro de Google. Eso no lo invalida, pero explica por qué la validación externa independiente llegó por otra vía: la del patrón, no la del modelo.

### 10.2. El patrón CRNN

Lo que se replicó afuera no fue la CLDNN exacta —nadie fuera de Google necesitaba 13,522 estados CD— sino su **esqueleto**: *convoluciones 2D sobre el espectrograma para extraer features locales, con pooling en frecuencia; recurrencia sobre el eje temporal para agregación; capas densas para clasificar*. El nombre que se le puso fue **CRNN**, y colonizó todo el audio no-ASR:

- **Detección de eventos sonoros.** Çakır, Parascandolo, Heittola, Huttunen y Virtanen, *"Convolutional Recurrent Neural Networks for Polyphonic Sound Event Detection"*, IEEE/ACM TASLP 2017. Fue baseline oficial de los desafíos DCASE durante años.
- **Etiquetado de música.** Choi, Fazekas, Sandler y Cho, *"Convolutional Recurrent Neural Networks for Music Classification"*, ICASSP 2017, sobre el Million Song Dataset y MagnaTagATune.
- **Clasificación de escenas acústicas**, detección de actividad de voz, reconocimiento de emociones: la misma plantilla.

**Nota para evitar una confusión frecuente.** El acrónimo CRNN también designa a Shi, Bai y Yao, *"An End-to-End Trainable Neural Network for Image-Based Sequence Recognition"* (TPAMI 2017), la arquitectura CNN+BiLSTM+CTC de reconocimiento de texto en escena — que Roberto ya vio en la clase 21. Son linajes **independientes** que convergieron a la misma topología porque el problema tiene la misma forma: una entrada 2D con un eje "espacial" que conviene comprimir y un eje secuencial que conviene modelar recurrentemente. En texto ese eje espacial es la altura de la imagen; en audio es la frecuencia. **Es el mismo argumento de asignación de ejes de la Sección 3.4.** Que dos comunidades sin contacto llegaran a la misma solución es la mejor evidencia de que la intuición era correcta.

### 10.3. Cuándo dejó de ser estado del arte, y por qué

Dos golpes sucesivos, por razones distintas.

**Primero, el modelo acústico híbrido murió.** Entre 2016 y 2019 el campo migró a modelos verdaderamente end-to-end: LAS (Listen, Attend and Spell) y sobre todo **RNN-T**, que Google llevó a producción en dispositivo en 2019 (Sainath es coautora de ese trabajo también). Eso volvió obsoleta la *función* de CLDNN —emitir posteriores sobre estados CD para un decodificador HMM— independientemente de si su arquitectura era buena. **CLDNN no fue superada; su puesto de trabajo desapareció.**

**Segundo, el LSTM fue reemplazado por atención.** Y aquí es donde el paralelo con Conformer se vuelve importante.

### 10.4. Conformer: la misma tesis, otro operador

**Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", Interspeech 2020.** (El PDF está en esta misma carpeta, `Gulati-Conformer-2020.pdf`; las citas siguientes están verificadas contra él.)

El argumento de apertura de Conformer es, punto por punto, **el mismo argumento de complementariedad de CLDNN**, con self-attention en el lugar del LSTM:

> "self-attention... [is] good at modeling content-based global interactions, while CNNs exploit local features" — abstract, Conformer.

> "we hypothesize that both global and local interactions are important... the combination of self-attention and convolution will achieve the best of both worlds – self-attention learns the global interaction whilst the convolutions efficiently capture the relative-offset-based local correlations" — Sección 1, Conformer.

Compárese con el abstract de CLDNN: "CNNs are good at reducing frequency variations, LSTMs are good at temporal modeling, and DNNs are appropriate for mapping features to a more separable space". **Es literalmente la misma estructura retórica: dos operadores con sesgos inductivos distintos, mejor juntos que por separado.** Cinco años después, con el operador de largo alcance cambiado.

Pero las diferencias son sustantivas y no se reducen a "LSTM → attention":

| | CLDNN (2015) | Conformer (2020) |
|---|---|---|
| Topología | **Pipeline**: bloque conv → bloque recurrente → bloque denso, una vez | **Bloque híbrido repetido**: FFN → MHSA → Conv → FFN, ×16 o ×17 |
| Cuándo se mezclan local y global | **Una sola vez**, y en un orden fijo | **En cada nivel de profundidad**, alternando |
| Operador de largo alcance | LSTM unidireccional, secuencial, $O(T)$ pasos dependientes | Self-attention multi-cabeza, $O(T^2)$ pero paralelizable |
| Eje de la convolución | **2D sobre (frecuencia, tiempo)** | **1D depthwise sobre tiempo**, kernel 32; la frecuencia ya la colapsó el front-end |
| Pooling | Max-pool en frecuencia, tamaño 3 | Sin max-pooling; submuestreo por convolución con stride en el front-end |
| Conexiones de salto | 2, concatenativas, una descartada | Residuales aditivas en cada submódulo, más *half-step* en los FFN |
| Normalización | Ninguna mencionada | LayerNorm + BatchNorm dentro del módulo conv |
| Salida | Posteriores sobre 13,522 estados CD, HMM aparte | RNN-T, end-to-end |
| Escala | ~23.5M (derivado) | 10.3M / 30.7M / **118.8M** (S/M/L, Tabla 1 de Conformer) |
| Resultado ancla | 13.1% WER, Voice Search interno 2000 h | **1.9% / 3.9%** en LibriSpeech test-clean/test-other con LM (2.1%/4.3% sin LM), Conformer(L) |

Cuatro observaciones que hacen que el paralelo valga la pena desarrollar:

1. **La C de CLDNN sobrevivió; la L no.** Conformer conserva un front-end convolucional 2D que submuestrea antes del stack de bloques, exactamente por la razón de CLDNN: la estructura tiempo-frecuencia local existe y una convolución la captura barato. Lo que se murió fue el LSTM. Que **la parte del paper que sobrevivió sea la convolucional** es lo interesante: en 2015 la CNN era la pieza nueva y arriesgada, y el LSTM el caballo de batalla.

2. **Pipeline contra intercalado es la diferencia arquitectónica real.** En CLDNN, el LSTM ve solo lo que la CNN le dejó pasar por un cuello de botella de 256 dimensiones, y el DNN solo lo que el LSTM le dejó. Es una jerarquía estricta, y su rigidez es lo que la Tabla 7 estaba tratando de aflojar con las conexiones de salto (con éxito muy modesto). Conformer resuelve el mismo problema estructuralmente: al alternar mezcla local y global en cada bloque, la información local no tiene que sobrevivir intacta a través de todo el bloque global. **Las conexiones de salto de CLDNN son un parche a un problema que Conformer resuelve por diseño.**

3. **La razón por la que el LSTM cayó no es solo de calidad.** Es de **paralelización**: la recurrencia impone $T$ pasos secuenciales en el forward y en el BPTT, mientras que la self-attention es una sola multiplicación de matrices grande. Cuando el hardware pasó a ser TPUs y GPUs con cientos de TFLOPs y el cuello de botella se volvió la utilización, un operador $O(T^2)$ paralelizable superó a uno $O(T)$ secuencial. Es la misma historia que en NLP con "Attention is All You Need".

4. **Pero la asignación de ejes se mantiene.** Conformer no hace pooling en tiempo tampoco, y su convolución es *depthwise sobre el eje temporal* — la resolución temporal se preserva o se reduce de forma controlada, nunca se colapsa por invarianza. **El insight de la Sección 3.4 de este análisis sobrevivió intacto cinco años y un cambio completo de operador base.**

### 10.5. Una nota sobre el otro camino

Mientras Conformer conservaba la convolución, la otra rama la eliminó del todo: **AST** (Gong, Chung y Glass, *"AST: Audio Spectrogram Transformer"*, Interspeech 2021 — también en esta carpeta) trata el espectrograma como una secuencia de parches y aplica un ViT puro, sin ninguna convolución. Es la refutación más limpia de la tesis de CLDNN: si hay suficientes datos y suficiente pre-entrenamiento (AST inicializa desde ViT/ImageNet), el sesgo inductivo convolucional deja de ser necesario. La tesis de la complementariedad no es una ley: es una afirmación sobre el régimen de datos de 2015.

---

## 11. Conexión con la clase 39

El "Ejemplo 1" de la clase del profesor Sepúlveda es la CLDNN, sin nombrarla. Aquí está el mapeo uno a uno.

| Elemento del slide | Qué dice el paper | Ubicación | Veredicto |
|---|---|---|---|
| Entrada log-mel 40D | "each frame $x_t$ is a 40-dimensional log-mel feature" | Sección 2.1 | **Coincide exactamente** |
| Ventanas de 10-20 ms con 5-10 ms de solape | "computed every 10ms"; la longitud de ventana **no se especifica** | Sección 3 | **Coincide el hop de 10 ms; la ventana es del profesor**, no del paper |
| 2 capas convolucionales | "we use 2 convolutional layers" | Sección 2.1 | **Coincide** |
| 256 filtros por capa | "each with 256 feature maps" | Sección 2.1 | **Coincide exactamente** |
| Kernel $9\times9$ | "a 9x9 frequency-time filter for the first convolutional layer" | Sección 2.1 | **Coincide exactamente** |
| Kernel $4\times4$ | "**a 4x3 filter** for the second convolutional layer" | Sección 2.1 | **NO coincide.** El paper dice $4\times3$ |
| Max-pooling opcional solo en frecuencia | "pooling in frequency only is performed" | Sección 2.1 | **Coincide.** El "opcional" refleja bien que solo la capa 1 tiene pooling |
| Ventanas no solapadas de tamaño 3 | "non-overlapping max pooling... A pooling size of 3 was used for the first layer, and no pooling was done in the second layer" | Sección 2.1 | **Coincide exactamente**, con el matiz de que es solo en la primera capa |
| Convolución $1\times1$ para reducción de dimensión | "we add a **linear layer** to reduce feature dimension... 256 outputs"; la dimensión grande viene de "feature-maps × time × frequency context" | Sección 2.1 | **Coincide el propósito y el 256; NO coincide la operación.** Ver abajo |
| 2 capas LSTM | "we use 2 LSTM layers" | Sección 2.1 | **Coincide** |
| **256 celdas** por LSTM | "each LSTM layer has **832 cells**, and a **512 unit projection layer**" | Sección 2.1 | **NO coincide.** Discrepancia grande |
| 2 capas FC de 1024 unidades | "Each fully connected layer has 1,024 hidden units"; Tabla 4 satura en 2 | Sección 2.1, Tabla 4 | **Coincide exactamente** |
| Softmax | 13,522 targets CD | Sección 3 | **Coincide** en la operación; el número de clases es específico de Voice Search |

### 11.1. Las tres discrepancias, en orden de importancia

**1. Las 256 celdas del LSTM (el paper dice 832 + proyección de 512).** Es la más consecuente. El paper hereda esta configuración de Sak et al. 2014 (referencia [3]), y la capa de proyección no es decorativa: es lo que permite tener 832 celdas sin que la matriz recurrente sea de $832 \times 832$ por compuerta. Un LSTM de 256 celdas sin proyección es una red **sustancialmente más chica** — por mi derivación de la Sección 4.6, el bloque recurrente pasaría de ~6.8M a menos de 1M de parámetros. La hipótesis más probable de dónde salió el 256: es el número que sí aparece dos veces en el paper (256 mapas de features, 256 salidas de la capa lineal) y se contagió a la línea siguiente. Es un error plausible al condensar un párrafo denso en una viñeta.

**2. La convolución $1\times1$ (el paper dice capa lineal).** No es lo mismo y la diferencia es medible:
- Una **convolución $1\times1$** con 256 salidas mapea, en cada posición $(f, t)$, un vector de 256 canales a 256 canales. Preserva los ejes de frecuencia y tiempo. Con mi reconstrucción de formas, la salida sería $256 \times 7 \times 1$, o sea 1792 valores, y no habría reducción de dimensión alguna: el LSTM seguiría recibiendo 1792.
- La **capa lineal del paper** aplana los tres ejes ("feature-maps × time × frequency context") y mapea $1792 \to 256$. Es una reducción real, del orden de 7×, y es todo el punto de la capa.

O sea que la sustitución no solo cambia la operación: **anula la función que motiva la capa**. Dicho eso, la confusión es comprensible: en el vocabulario moderno "capa de reducción de dimensión con $1\times1$" es un idiom estándar (Inception, bottlenecks de ResNet), y el paper de 2015 no lo usa. La corrección precisa para la clase sería: *aplanar y proyectar linealmente a 256*.

**3. El kernel $4\times4$ (el paper dice $4\times3$).** La más menor de las tres. Con mis formas reconstruidas, $4\times3$ sobre una entrada de $10\times3$ (frecuencia × tiempo) consume **exactamente** los 3 frames temporales que quedan y deja el eje temporal en 1. Un $4\times4$ ni siquiera cabría sin padding. Ese detalle sugiere que el $4\times3$ del paper no es arbitrario: está dimensionado para agotar el eje temporal restante. La versión $4\times4$ es probablemente una "simetrización" involuntaria al copiar (el primero es $9\times9$, cuadrado; es fácil asumir que el segundo también).

### 11.2. La justificación conceptual del profesor

El slide justifica la receta diciendo que CNN, RNN y MLP tienen "propiedades complementarias": CNN aprende features locales (span local del filtro + invarianza a traslación), RNN relaciones temporales distantes, MLP el clasificador.

**Lo que coincide.** La estructura del argumento es la del paper, y la palabra "complementary" es literalmente del abstract. La asignación de roles —CNN a features, RNN a tiempo, MLP a clasificación— es correcta y es la del paper.

**Lo que difiere.**

1. **"Invarianza a traslación" no está en el paper** (cero ocurrencias de `invarian*`). El paper dice "reducir la variación en frecuencia". Como argumenté en la Sección 3.3, el mecanismo de la CNN es genuinamente el de compartición de pesos + pooling, así que la afirmación no es falsa, pero borra el matiz que hace interesante a este paper: **la invarianza que se busca es específicamente en el eje de frecuencia, y su justificación es la anatomía del tracto vocal, no un principio general de visión por computador.** Y sin ese matiz se pierde el punto pedagógico más importante de la clase: en audio, la invarianza a traslación en tiempo sería un error.

2. **El paper tiene un tercer argumento que el slide no menciona**, y que es el más profundo: el de Pascanu et al. sobre las tres transiciones de una RNN (Sección 3.1). Bajo esa lectura, la CLDNN no es "tres redes complementarias" sino **un LSTM con sus dos transiciones afines reemplazadas por transiciones profundas**. Esa formulación explica por qué el orden es C→L→D y no otro, cosa que "propiedades complementarias" no explica.

3. **El paper ancla el orden en la historia del ASR** (VTLN/fMLLR antes del HMM), lo que le da a la receta una justificación externa y verificable. El slide presenta el orden como si fuera una consecuencia de las propiedades de cada red, cuando en realidad es una herencia de la arquitectura de sistemas GMM/HMM.

### 11.3. Recomendación concreta para la clase

Los números del slide son la CLDNN con tres modificaciones (kernel 2 simetrizado, LSTM reducido ~3× sin proyección, capa lineal reinterpretada como $1\times1$). Como configuración didáctica es perfectamente razonable —un LSTM de 256 celdas es lo que corre en un Colab—, pero conviene nombrarlo: *"esta es la CLDNN de Sainath et al., ICASSP 2015, con el LSTM reducido para que quepa en la clase"*. Y vale la pena rescatar el número que **sí** hay que defender, que es el pooling de tamaño 3 solo en frecuencia, porque es el que tiene una justificación física derivable (Sección 3.4) y es el que distingue audio de visión.

---

## 12. Erratas, matices y cosas que se citan mal

### 12.1. Erratas del propio paper

| # | Errata | Ubicación exacta | Detalle |
|---|---|---|---|
| 1 | "we combine CNNs, LSTMs and **CNNs** into one unified framework" | Sección 1, párrafo 5 | Debe decir **DNNs**. Es el paper describiendo su propia contribución central y nombrando mal uno de sus tres componentes |
| 2 | "it could be beneficial to **proceed** LSTMs with a few fully connected CNN layers" | Sección 1, párrafo 2 | Debe decir **precede**. Además "fully connected CNN layers" es contradictorio |
| 3 | Glorot y Bengio, AISTATS, **2014** | Referencia [18] | El paper de Glorot-Bengio es de **AISTATS 2010** |
| 4 | "F. Grezl and M. **Karafat**" | Referencia [14] | El apellido es **Karafiát** (y Grézl) |
| 5 | "an 4% relative improvement", "an 4-6% relative reduction" | Sección 1 y Sección 6 | Artículo incorrecto, dos veces |

Las erratas 1 y 2 son sintomáticas de un preprint de 5 páginas escrito contra la fecha límite de ICASSP. No afectan la sustancia, pero conviene conocerlas si alguien cita el texto literalmente.

### 12.2. Discrepancias numéricas entre el texto y las tablas

**Verificadas calculando los relativos sobre los WER de las tablas.**

| Afirmación del texto | Ubicación | Números de la tabla | Relativo real | Veredicto |
|---|---|---|---|---|
| "a **6%** relative reduction in WER over the LSTM after CE training" | Sección 5, párrafo 2 | Tabla 8: 14.6 → 13.8 | **5.48%** | Redondeo **hacia arriba** de 5.5 a 6 |
| "a **5%** relative improvement after sequence training" | Sección 5, párrafo 2 | Tabla 8: 13.7 → 13.1 | **4.38%** | Redondeo **hacia arriba** de 4.4 a 5 |
| "a **4%** relative reduction in WER compared to the LSTM" (CE, ruidoso) | Sección 5, párrafo 3 | Tabla 9: 20.3 → 19.4 | **4.43%** | Correcto (redondeo hacia abajo) |
| "a **7%** relative improvement over the LSTM" (secuencia, ruidoso) | Sección 5, párrafo 3 | Tabla 9: 18.8 → 17.4 | **7.45%** | Correcto |
| "**4-6%** relative improvement in WER over an LSTM" | Abstract y Sección 6 (conclusiones) | Rango real medido: 3.9% a 7.4% | — | **La banda declarada no contiene el mejor resultado del paper** |

Sobre la última fila, que es la más importante: el paper resume su propia contribución como "4-6%", pero:

- El **piso** real es 3.9% (Tabla 5, 200 h, 18.0 → 17.3), ligeramente por debajo de 4.
- El **techo** real es **7.4%** (Tabla 9, ruidoso con entrenamiento de secuencia, 18.8 → 17.4), muy por encima de 6.
- El "6%" del extremo superior de la banda solo aparece redondeando el 5.48% de la Tabla 8 hacia arriba.

O sea: **el abstract subestima el mejor resultado del paper en más de un punto porcentual relativo**, mientras que redondea generosamente otro. La lectura caritativa es que la banda "4-6%" se armó con los resultados de CE (donde el rango es 3.9-5.5%) y no se actualizó tras agregar el entrenamiento de secuencia. Es un caso poco frecuente de un abstract que **infravende** su propio resultado. Si se cita este paper, el número honesto es **"entre 4% y 7% relativo según la condición, con las mayores ganancias en habla ruidosa tras entrenamiento de secuencia"**.

### 12.3. Trampas de lectura de las tablas

**El baseline LSTM cambia de valor a mitad del paper.** Tablas 1 a 5 usan LSTM = **18.0** (inicialización gaussiana). Tablas 6 y 7 usan LSTM = **17.7** (inicialización uniforme). Comparar un número de la Tabla 7 contra uno de la Tabla 5 es comparar regímenes distintos. En particular, el "16.8" de la Tabla 7 **no** se debe comparar contra el "18.0" de la Tabla 1 para calcular una mejora del 6.7%: la comparación válida es contra 17.7, o sea 5.1%.

**Los baselines CNN y DNN nunca se re-evalúan tras el cambio de inicialización.** La Sección 4.5 concluye "with proper weight initialization, the LSTM is better than the CNN or DNN in Table 1", comparando el LSTM mejorado (17.7) contra números de la Tabla 1 (18.0, 18.4) obtenidos en otra fase del estudio. Es técnicamente defendible —el problema de inicialización era específico de las capas recurrentes, y CNN/DNN ya usaban Glorot— pero la comparación mezcla tablas de dos regímenes.

**Tabla 7 lista cuatro filas pero la arquitectura final tiene tres.** La cuarta ("+ CNN to LSTM and DNN layers", 17.0) es un experimento **fallido** que el paper descarta explícitamente. La Figura 1 dibuja ambos flujos punteados, lo que hace fácil creer que ambos son parte del modelo.

### 12.4. Lo que se le atribuye al paper y no está ahí

- **"CLDNN demostró que el orden CNN→LSTM→DNN es el óptimo."** Falso. El orden nunca se ablaciona (Sección 8.2). Se justifica por argumento y se valida contra los componentes aislados, que es otra cosa.
- **"CLDNN introdujo las conexiones residuales en audio."** Falso en dos sentidos: son **concatenativas**, no residuales, y la motivación es multi-escala, no de optimización (Sección 5.3).
- **"CLDNN es un modelo end-to-end."** Falso. Es el modelo acústico de un sistema híbrido DNN/HMM con 13,522 estados CD, alineamiento forzado y decodificador Viterbi (Sección 9.2).
- **"CLDNN demostró que hay que hacer pooling solo en frecuencia."** Falso: lo **hace**, citando trabajo previo ([11]), pero **no lo mide** en este paper. No hay ninguna ablación de pooling.
- **"CLDNN mejora ~5% sobre el LSTM."** Impreciso: entre 3.9% y 7.4% según condición, y la mejora **crece** con ruido y con entrenamiento de secuencia (Sección 7.3).
- **"Las adiciones multi-escala son parte de la CLDNN."** A medias: una de las dos se descarta, y la que sobrevive aporta ~1% en CE y **exactamente 0%** tras el entrenamiento de secuencia (Tablas 8 y 9).
- **"CLDNN usa una convolución $1\times1$ para reducir dimensión."** No: usa una capa lineal sobre el tensor aplanado completo (Sección 4.3).

---

## 13. Cómo se ve hoy

El esqueleto siguiente implementa la CLDNN de la Sección 2.1 del paper con la configuración de la Sección 4.4 ($l=10$, $r=0$, sin las adiciones multi-escala). Los puntos que importan son tres: el pooling **solo en frecuencia**, el manejo de las formas cuando la CNN se evalúa una vez por frame de salida, y la capa lineal que aplana los tres ejes.

```python
import torch
import torch.nn as nn

class CLDNN(nn.Module):
    """Sainath et al., ICASSP 2015. Configuración de la Sección 4.4: l=10, r=0."""

    def __init__(self, n_mels=40, left_ctx=10, n_maps=256,
                 lstm_cells=832, lstm_proj=512, fc_units=1024, n_targets=13522):
        super().__init__()
        self.left_ctx = left_ctx
        win = left_ctx + 1                       # 11 frames: [x_{t-10} ... x_t], r=0

        # --- Bloque C: la "imagen" es (freq, tiempo). Convolución válida, sin padding:
        #     el paper no documenta el padding, y con 'valid' las formas cierran exactas.
        self.conv1 = nn.Conv2d(1, n_maps, kernel_size=(9, 9))          # 9x9 freq-tiempo
        # Pooling SOLO EN FRECUENCIA. El (3, 1) es el corazón del paper: kernel y stride
        # valen 3 en el eje de frecuencia (no solapado) y 1 en tiempo. Poolear en tiempo
        # destruiría la resolución de 10 ms que el HMM y el LSTM necesitan.
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.conv2 = nn.Conv2d(n_maps, n_maps, kernel_size=(4, 3))     # 4x3, NO 4x4
        # Sin pooling en la capa 2 ("no pooling was done in the second layer").

        # Forma tras conv2 con n_mels=40, win=11:
        #   (40,11) -conv 9x9-> (32,3) -pool (3,1)-> (10,3) -conv 4x3-> (7,1)
        f_out = ((n_mels - 8) // 3) - 3          # 40 -> 32 -> 10 -> 7
        t_out = win - 8 - 2                      # 11 -> 3 -> 3 -> 1
        flat = n_maps * f_out * t_out            # 256*7*1 = 1792

        # Capa lineal de reducción: aplana mapas x freq x tiempo y proyecta a 256.
        # NO es una conv 1x1 (esa preservaría los ejes freq/tiempo y no reduciría nada).
        # Sin no-linealidad: es una factorización de rango bajo, ~4.6M de parámetros menos
        # en la matriz de entrada del primer LSTM.
        self.dim_red = nn.Linear(flat, 256)

        # --- Bloque L: LSTMP. proj_size implementa la proyección recurrente de Sak 2014,
        #     que es lo que permite 832 celdas sin una recurrente de 832x832 por compuerta.
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_cells, num_layers=2,
                            proj_size=lstm_proj, batch_first=True)      # unidireccional

        # --- Bloque D: 2 capas (Tabla 4 satura ahí) + softmax sobre estados CD del HMM.
        self.dnn = nn.Sequential(
            nn.Linear(lstm_proj, fc_units), nn.ReLU(),
            nn.Linear(fc_units, fc_units), nn.ReLU(),
            nn.Linear(fc_units, n_targets),          # logits; CE contra alineamiento forzado
        )

    def forward(self, x):                            # x: (B, T, n_mels), hop de 10 ms
        B, T, F = x.shape
        # Ventanas causales solapadas: una por frame de salida. unfold da (B, F, T', win)
        # con T' = T - left_ctx. Cada frame de entrada se reprocesa hasta 11 veces: es el
        # costo que el paper no reporta y que una implementación de producción comparte.
        w = x.transpose(1, 2).unfold(dimension=2, size=self.left_ctx + 1, step=1)
        Tp = w.shape[2]
        w = w.permute(0, 2, 1, 3).reshape(B * Tp, 1, F, self.left_ctx + 1)  # (B*T',1,freq,t)

        h = self.conv2(self.pool1(self.conv1(w)))    # (B*T', 256, 7, 1)
        h = self.dim_red(h.flatten(1))               # (B*T', 256)
        h = h.view(B, Tp, -1)                        # (B, T', 256) -> secuencia para el LSTM

        h, _ = self.lstm(h)                          # (B, T', 512) por la proyección
        return self.dnn(h)                           # (B, T', 13522)
```

**Notas de traducción a 2026.**

- `proj_size` en `nn.LSTM` de PyTorch implementa exactamente la LSTMP de Sak et al.: la salida y el estado realimentado tienen dimensión `proj_size`, no `hidden_size`. Sin ese argumento se estaría implementando un LSTM ordinario de 832 celdas, que es una red distinta y bastante más pesada.
- El `unfold` deja explícito el costo que el paper esconde: la CNN se evalúa `T'` veces sobre ventanas solapadas. Un sistema de producción convoluciona una sola vez sobre la secuencia completa y explota que la convolución ya es equivariante en tiempo — que es, irónicamente, la propiedad que la arquitectura tiene pero que la formulación por ventanas desaprovecha.
- Falta el **retraso de 5 frames de la etiqueta**: en el entrenamiento hay que desplazar los targets, no la entrada. Es lo que da los 50 ms de lookahead sin agregar contexto futuro a la entrada de la CNN.
- Falta el **BPTT truncado a 20 pasos**: en PyTorch se hace segmentando la secuencia y haciendo `detach()` del estado entre segmentos. Con secuencias cortas y `nn.LSTM` completo, el gradiente fluye por toda la secuencia, que no es lo que hace el paper.
- No hay normalización en ningún lado, y es fiel al original: ni BatchNorm ni LayerNorm aparecen en el paper. Una reimplementación moderna casi con seguridad agregaría BatchNorm tras cada convolución y LayerNorm en el stack recurrente, lo que probablemente haría irrelevante toda la discusión de inicialización de la Sección 4.5.
- La salida de 13,522 logits solo tiene sentido dentro de un sistema HMM. Para cualquier tarea moderna de audio —clasificación de eventos, etiquetado de música, keyword spotting— lo que se reemplaza es exactamente eso: un pooling temporal sobre `h` y una cabeza del tamaño del número de clases. Ese cambio de una línea es, literalmente, la transición de CLDNN a CRNN.
