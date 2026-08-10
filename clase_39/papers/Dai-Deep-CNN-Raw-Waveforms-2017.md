# Very Deep Convolutional Neural Networks for Raw Waveforms (familia M3–M34-res) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** Wei Dai\*, Chia Dai\*, Shuhui Qu, Juncheng Li, Samarjit Das. El asterisco indica contribución equitativa de los dos primeros (nota al pie de la primera página).
- **Afiliaciones** (deducidas de los correos que aparecen bajo el título, única fuente en el PDF): `{wdai,chiad}@cs.cmu.edu` → **Carnegie Mellon University**; `shuhuiq@stanford.edu` → **Stanford**; `{billy.li,samarjit.das}@us.bosch.com` → **Bosch Research North America**.
- **Versión analizada:** arXiv:1610.00087v1, marca en el margen "1 Oct 2016"; los metadatos del PDF dan fecha de compilación 3 de octubre de 2016. El PDF **no lleva encabezado ni pie de página de ICASSP**: la referencia habitual "ICASSP 2017" corresponde a la versión publicada, que no es este archivo. Todo lo que se cita abajo sale del preprint.
- **Financiamiento:** contrato FA8702-15-D-0002 con el Software Engineering Institute (centro patrocinado por el Departamento de Defensa de EE.UU.), Sección 6.

El paper hace una sola pregunta y la responde con un barrido limpio: **¿por qué las CNN sobre onda cruda funcionaban mediocre en 2016 — porque la onda cruda es mala entrada, o porque nadie había ido lo suficientemente profundo?** La respuesta es la segunda. Se construye una familia de cinco redes 1D completamente convolucionales, idénticas en filosofía y distintas solo en profundidad, y se mide.

**Cifras ancla (Tabla 2, UrbanSound8K, fold 10 como test, 8 kHz):**

| Modelo | Capas con peso | Capas conv | Parámetros (Tabla 1) | Accuracy test | Tiempo/época (Titan X) |
|---|---|---|---|---|---|
| M3 | 3 | 2 | 0.2M | **56.12%** | 77 s |
| M5 | 5 | 4 | 0.5M | **63.42%** | 63 s |
| M11 | 11 | 10 | 1.8M | **69.07%** | 71 s |
| M18 | 18 | 17 | 3.7M | **71.68%** | 98 s |
| M34-res | 34 | 33 | 4M | **63.47%** | 124 s |

De ahí sale el titular: **M18 supera a M3 por 15.56 puntos absolutos** (71.68 − 56.12), y la profundidad mejora monótonamente hasta 18 capas y después colapsa por sobreajuste.

**Sobre "la comparación contra el baseline MFCC/log-mel", que hay que decir con precisión: el paper no corre ningún baseline de features propio.** No hay una fila "MFCC + la misma red" ni "log-mel + CNN, entrenado por nosotros". La afirmación de paridad se apoya **enteramente** en un número leído de otro paper: la nota al pie 3 dice textualmente que *"Figure 4 in [11] reports ∼68% accuracy using a baseline CNN model"*, donde [11] es Piczak (MLSP 2015), CNN sobre espectrograma log-mel. Y la misma nota al pie reconoce dos diferencias que invalidan la comparación directa: *"we use the 10-th fold as test set, while [11] performs 10-fold evaluation"* y *"we use sound at 8kHz sampling rate while they use the original 44.1kHz"*. Es decir: 71.68% de Dai (un fold, 8 kHz) contra ~68% de Piczak (diez folds, 44.1 kHz). La conclusión del paper —*"To our knowledge, this is the first report of a parity performance between log-mel features and raw time signal for environmental sound recognition"*— es una afirmación fuerte sostenida por una comparación cruzada entre papers con protocolos distintos. El paper lo declara honestamente en la nota al pie; el problema es que la nota al pie no llega al abstract.

---

## 2. Contexto: el debate onda cruda vs features hechos a mano en 2016

### Por qué el pipeline pasaba por MFCC durante décadas

La primera línea del paper lo enmarca así: *"Acoustic modeling is traditionally divided into two parts: (1) designing a feature representation of the audio data, and (2) building a suitable predictive model based on the representation."* Ese corte no era pereza: era la única forma viable de trabajar con audio antes del deep learning, y tenía tres razones sólidas.

**Compresión brutal con pérdida dirigida.** Cuatro segundos de audio a 16 kHz son 64 000 números. Los mismos cuatro segundos en MFCC, con ventanas de 25 ms y salto de 10 ms, son 400 tramas × 13 coeficientes = 5 200 números: una reducción de ~12×, y la mayoría de lo descartado es fase e información fuera del rango perceptualmente relevante.

**Invarianzas gratis.** El espectro de magnitud descarta la fase, con lo cual el resultado es aproximadamente invariante al desplazamiento temporal dentro de la ventana. La escala mel comprime las frecuencias altas imitando la resolución del oído. El logaritmo convierte la ganancia multiplicativa (volumen, distancia al micrófono) en un desplazamiento aditivo. La DCT final decorrelaciona los canales, lo que era indispensable cuando el modelo aguas abajo era un GMM con covarianza diagonal.

**Escala de datos.** Con miles —no millones— de ejemplos etiquetados, un modelo que empieza desde 64 000 muestras crudas tiene demasiada libertad. Los features acotaban el espacio de hipótesis.

El paper resume la objeción moderna en una frase: *"it is often challenging and time-intensive to find the right representation in the so-called 'feature-engineering' process, and the often heuristically designed features might not be optimal for the predictive task."*

### Qué se había intentado antes y por qué quedó a la par

Las referencias que el paper usa para posicionarse son cuatro trabajos de 2014–2015, todos de reconocimiento de voz:

| Trabajo | Ref. en el paper | Qué hizo | Resultado |
|---|---|---|---|
| Tüske, Golik, Schlüter, Ney (INTERSPEECH 2014) | [9] | Modelado acústico con DNN sobre señal temporal cruda para LVCSR | Primera evidencia de que la capa inicial imita transformadas tipo wavelet |
| Golik, Tüske, Schlüter, Ney (INTERSPEECH 2015) | [10] | CNN sobre señal temporal cruda para LVCSR | Competitivo, no superior |
| Hoshen, Weiss, Wilson (ICASSP 2015) | [3] | Modelado acústico desde forma de onda multicanal | Aprende beamforming implícito |
| Sainath, Weiss, Senior, Wilson, Vinyals (INTERSPEECH 2015) | [4] | CLDNN con front-end de forma de onda aprendido | **Iguala** el desempeño de log-mel |

El diagnóstico del paper es de una frase y es el eje de todo: *"These works, however, have mostly considered only less deep networks, such as two convolutional layers [4, 11]."* Nadie había fallado en la onda cruda; nadie la había llevado a profundidad. En Sainath et al. la parte convolucional del CLDNN es de una o dos capas, y el trabajo pesado lo hacen las LSTM que vienen después.

Y hay un contraste que el paper usa con precisión quirúrgica. Piczak [11] entrena una CNN de **dos capas convolucionales sobre espectrograma log-mel** en UrbanSound8K y le va bien. Dai entrena una CNN de **dos capas convolucionales sobre onda cruda** (M3) y obtiene 56.12%. Textual: *"This is in contrast with models using the spectrogram as input, which achieve good performance with just 2 convolutional layers [11], and shows that applying CNN directly on time-series data is challenging."* La misma profundidad, la misma tarea, el mismo dataset: los features hechos a mano compran profundidad efectiva. Cuando la entrada es cruda, hay que pagar esa profundidad con capas.

### La apuesta

*"CNNs have famously achieved performance competitive or even surpassing human-level performance in the visual domains... A common theme among these powerful CNN models is that they are usually very deep, with the number of layers ranging from tens to even over a hundred."*

La analogía es directa: en visión nadie diseña features SIFT/HOG desde 2012, y lo que hizo posible abandonarlos no fue "aplicar una CNN" sino **apilar decenas de capas**. Si en audio se aplicaron CNN de dos capas y quedaron a la par de los features, la hipótesis natural es que falta profundidad, no que la onda cruda sea insuficiente.

La contra evidente —y por eso el paper existe— es que en audio la profundidad es cara de una forma que en imágenes no lo es: la entrada es **32 000 valores en una sola dimensión**. Ahí entran las tres piezas de ingeniería del paper: batch normalization, aprendizaje residual y, sobre todo, **un diseño cuidadoso del submuestreo en las primeras capas**.

---

## 3. La tesis: profundidad, no ingeniería de features

La tesis se puede formular como un experimento controlado: **fijar todo lo demás y mover solo la profundidad.** Las cinco redes comparten:

- la misma entrada (32 000 muestras a 8 kHz = 4 s exactos),
- la misma primera capa (kernel 80, stride 4),
- el mismo kernel 3 en todas las demás convoluciones,
- los mismos cuatro max-pooling de tamaño y stride 4,
- el mismo cabezal (global average pooling → softmax, sin capas densas),
- el mismo esquema de duplicación de canales al reducir resolución,
- el mismo optimizador (Adam), la misma inicialización (Glorot), la misma regularización ($\ell_2$ con coeficiente $10^{-4}$),
- la misma ausencia de dropout y de data augmentation.

Lo único que cambia es **cuántas convoluciones de kernel 3 hay en cada etapa**: 1, 1, 2–3, 4 o 6 bloques residuales. Ese es el diseño experimental, y es lo que le da fuerza al resultado.

**Las dos herencias de visión son explícitas y literales:**

**VGG (Simonyan y Zisserman, ref. [15]).** El paper dice *"we use very small receptive field 3 for all but the first 1D convolutional layers"* con nota al pie *"Small receptive fields were first popularized by [15] for 2D images"*. El razonamiento de VGG traslada exactamente: tres convoluciones de $3\times3$ apiladas cubren un campo receptivo de $7\times7$ con $3\cdot 9 C^2 = 27C^2$ parámetros en vez de $49C^2$, y con dos no-linealidades extra. En 1D el ahorro es menor ($3\cdot 3 C^2 = 9C^2$ contra $7C^2$ — de hecho **apilar sale más caro en parámetros en 1D**), pero se conserva lo importante: más no-linealidades por unidad de campo receptivo, y control fino de la capacidad. La segunda mitad de la herencia VGG es el patrón de canales: *"the reduction of resolution is complemented by a doubling in the number of feature maps"*, con la nota al pie que explica el sesgo inductivo — filtros básicos abajo, especializados arriba.

**ResNet (He et al., ref. [6]).** M34-res no es "una red profunda con atajos": es **la planta de bloques de ResNet-34 copiada dígito por dígito** — 3, 4, 6, 3 bloques residuales por etapa, con dos convoluciones cada uno. Que el nombre sea "M34" y no un número arbitrario es deliberado.

Hay una tercera herencia menos publicitada y más importante para el costo: **GoogLeNet/Inception (ref. [16])**. El paper cita a Szegedy justo para justificar el submuestreo agresivo temprano: *"we aggressively reduce the temporal resolution in the first two layers by 16x with large convolutional and max pooling strides to limit the computation cost in the rest of the network [16]"*. Es el *stem* barato de Inception, trasladado a 1D y llevado al extremo.

Vale la pena notar lo que la tesis **no** dice. No dice que la onda cruda sea mejor que log-mel. Dice `matches` y `competitive`. La afirmación es de **paridad**: los features hechos a mano dejan de ser necesarios, no dejan de ser buenos.

---

## 4. La primera capa: el detalle más importante del paper

### El número 80

`[80/4, C]`: kernel de 80 muestras, stride 4. Todo lo demás en la red usa kernel 3. Esa asimetría es la decisión de diseño central, y el paper la justifica con una regla física, no con búsqueda de hiperparámetros:

$$\frac{80\ \text{muestras}}{8000\ \text{muestras/s}} = 10\ \text{ms}$$

Textual (Sección 2): *"We thus choose our first layer receptive field to cover a 10-millisecond duration, which is similar to the window size for many MFCC computation."*

Y el corolario que casi nadie cita, pero que es la parte generalizable: *"audio sampling rate could affect the receptive field size in the first layer, since a field size of 80 at 8kHz sampling rate is at a different length scale than at 16kHz sampling rate."* **El 80 no es un hiperparámetro; es 10 ms expresados en las unidades de muestreo del dataset.** Cambiar el sample rate obliga a cambiar el 80 (ver Sección 14).

**Matiz sobre "la ventana estándar de MFCC".** En el pipeline MFCC clásico (HTK, Kaldi, `librosa` por defecto) la **ventana** es de 25 ms y el **salto** entre tramas es de 10 ms. Los 10 ms del paper coinciden con el *hop*, no con la ventana. La afirmación sigue siendo defendible —la propia clase 39, en el "Ejemplo 1" del slide, describe log-mel sobre *"overlapped segments of 10-20ms"*, y en reconocimiento de voz las ventanas de 10–20 ms son habituales— pero conviene no repetir "80 muestras es la ventana de MFCC" como si fuera una identidad exacta. Es del **orden de magnitud** de la ventana de MFCC, que es lo que importa para el argumento.

Hay un contraste adicional que el paper no explota. El *hop* de la primera capa es **4 muestras = 0.5 ms**, y tras el max-pooling es de 16 muestras = **2 ms**. Un pipeline MFCC produce una trama cada 10 ms. Es decir, la representación que sale de las dos primeras capas de la familia M está **5 veces más densamente muestreada en el tiempo** que un espectrograma log-mel convencional, aun después del submuestreo "agresivo". Esa redundancia es parte de lo que las capas de kernel 3 tienen que digerir.

### Qué significa "se parece a un banco de filtros"

El paper hace dos afirmaciones distintas que conviene separar.

**La afirmación de diseño (Sección 2):** *"We use a large receptive field in the first convolutional layer to mimic bandpass filters."* Esto es una **intención**: nada en la arquitectura obliga a que la capa 1 sea un banco de filtros pasa-banda. Es una convolución 1D de 80 taps sin restricciones, y podría aprender cualquier respuesta al impulso.

**La afirmación empírica (Sección 4, Figura 2):** *"All of them learn a filter bank of bandpass filter."* El experimento es directo: se toma cada uno de los 64 kernels de la primera capa de M18 ya entrenada, se le aplica **transformada de Fourier** y se grafica la magnitud espectral, con los filtros **ordenados por su frecuencia de activación**. El eje horizontal va de 0 a 4000 Hz (Nyquist a 8 kHz) y el vertical enumera los filtros. En M18 el resultado es una **diagonal limpia y bien poblada**: cada filtro tiene un pico estrecho, los picos cubren el espectro completo sin huecos, y no hay dos filtros redundantes.

Eso es literalmente lo que es un banco de filtros mel: un conjunto de respuestas pasa-banda que particiona el espectro. La diferencia es que aquí **el particionamiento se aprendió**, y su distribución no está forzada a seguir la escala mel.

Por qué importa: es la validación más directa de la tesis del paper. Si se le da a una red la onda cruda y suficiente presión de tarea, **reconstruye por sí sola el primer paso del pipeline que el ingeniero de features habría escrito a mano**. El paper conecta esto con la literatura previa: *"Previous works have shown that the first convolutional layer, when trained on raw waveforms, mimics wavelet transforms [9, 4]"* — así que Dai et al. confirman en sonido ambiental lo que Tüske y Sainath ya habían observado en voz.

**Lo que la observación no dice.** Que la capa aprenda algo *parecido* a un banco de filtros no implica que sea óptima como banco de filtros. Dos años después, **SincNet** (Ravanelli y Bengio, 2018) tomó exactamente esta observación y la llevó a su conclusión lógica: si la capa va a aprender filtros pasa-banda, restrinjámosla a serlo. SincNet parametriza cada filtro de la primera capa con **dos números** (frecuencia de corte inferior y superior de una función sinc en el tiempo), reduciendo esa capa de $80\times C$ pesos libres a $2C$. Esa es la crítica implícita al diseño de Dai: 80 pesos libres por filtro es capacidad desperdiciada si el óptimo vive en una variedad de dimensión 2.

### Por qué 80 y no 8 ni 320: la resolución frecuencial

El paper mide las dos alternativas (Tabla 3) y la explicación es de procesamiento de señales elemental. Un filtro FIR de $N$ taps a frecuencia de muestreo $f_s$ tiene una resolución frecuencial acotada por el ancho del lóbulo principal de su ventana, del orden de

$$\Delta f \approx \frac{f_s}{N}$$

| Variante | $N$ | Duración | $\Delta f$ aprox. a 8 kHz | Consecuencia |
|---|---|---|---|---|
| M-srf | 8 | 1 ms | ~1000 Hz | Solo cuatro "bandas" distinguibles en todo el espectro |
| **M (paper)** | **80** | **10 ms** | **~100 Hz** | ~40 bandas resolubles; comparable a un banco mel de 40 filtros |
| M-lrf | 320 | 40 ms | ~25 Hz | Resolución fina, pero el filtro promedia sobre 40 ms |

Y así se lee la Figura 2 completa. **M18-srf** (centro): *"has much more dispersed bands, and thus lower frequency resolution for subsequent layers"* — las bandas aparecen como manchas anchas y superpuestas concentradas en la zona media, sin diagonal. **M18-lrf** (derecha): *"has fine-grained filters, but does not have sufficient filters in the high frequency range, showing that it cannot effectively respond to local high frequency impulses"* — hay diagonal, pero se satura en frecuencias bajas y deja el rango alto casi vacío. El resumen del paper: *"a small RF popularized by vision models is insufficient to capture the necessary bandpass filter characteristics in the first convolutional layer, while a large RF smooths out local structures and cannot effectively detect local impulse patterns."*

Ese último punto es el **compromiso tiempo-frecuencia** (el principio de incertidumbre de Gabor) apareciendo como hiperparámetro de arquitectura. En UrbanSound8K hay clases que son puro transiente (`gun_shot`, `dog_bark`) y clases que son ruido estacionario de banda ancha (`air_conditioner`, `engine_idling`). Los 10 ms son el punto de equilibrio entre ambas.

### El factor de reducción de la entrada

Combinando kernel, stride y max-pooling en las dos primeras capas:

$$\text{factor} = s_{\text{conv1}} \times k_{\text{pool1}} = 4 \times 4 = 16$$

En cifras exactas, con convoluciones sin padding (que es lo que implementa el código):

$$L_1 = \left\lfloor \frac{32000 - 80}{4} \right\rfloor + 1 = 7981 \quad\longrightarrow\quad L_2 = \left\lfloor \frac{7981}{4} \right\rfloor = 1995$$

$32000/1995 = 16.04$. El paper redondea a *"reduce the temporal resolution in the first two layers by 16x"* y en la Tabla 1 escribe las longitudes idealizadas (8000 y 2000), como si hubiera padding "same". Ver Sección 13 sobre esta discrepancia.

La justificación es puramente de costo: *"to limit the computation cost in the rest of the network"*. Y el paper la valida con un experimento directo: *"When we use stride 1 instead of 4 in the first convolutional layer for M11, we observe a 3.5x increase in training time but a lower test accuracy (67.37%) after 10 hours of training, compared with 68.42% test accuracy reached in 2 hours by M18."* Con stride 1 se paga 3.5× de cómputo y se **pierde** precisión. El submuestreo agresivo no es un mal necesario: es parte de por qué la red funciona.

### El resto de la red: VGG en 1D, y el papel del pooling

Fuera de la primera capa todo es kernel 3. En 1D eso es un campo receptivo que crece **muy** lentamente por sí solo: $L$ convoluciones de kernel 3 con stride 1 dan campo receptivo $2L+1$. Para cubrir los 32 000 muestras de la entrada harían falta **16 000 capas**. Ahí está la razón de ser de los cuatro max-pooling de stride 4: cada uno multiplica por 4 el "salto" (*jump*) del mapa, y por lo tanto multiplica por 4 lo que aporta cada convolución posterior al campo receptivo.

Con las fórmulas estándar $r_l = r_{l-1} + (k_l - 1)\,j_{l-1}$ y $j_l = j_{l-1}\,s_l$, la aritmética completa (calculada capa por capa, sin padding) queda así:

| Red | Campo receptivo antes del GAP | Equivalente en tiempo a 8 kHz | Posiciones que promedia el GAP | Campo receptivo efectivo con GAP |
|---|---|---|---|---|
| M3 | 172 muestras | **21.5 ms** | 498 | 31 980 (todo el clip) |
| M5 | 1 772 muestras | **222 ms** | 30 | 31 468 (todo el clip) |
| M11 | 7 052 muestras | **881 ms** | 25 | 31 628 (todo el clip) |
| M18 | 11 980 muestras | **1.50 s** | 20 | 31 436 (todo el clip) |

Esta tabla es, en mi lectura, la explicación cuantitativa de todo el paper, y volveré sobre ella en la Sección 12.

---

## 5. La familia M en detalle

### Tabla 1 reproducida y auditada

La notación del paper: `[80/4, 256]` es una convolución de campo receptivo 80 con 256 filtros y stride 4; el stride se omite cuando es 1; `[...] × k` son $k$ capas apiladas; los corchetes con **dos filas** son un bloque residual y solo aparecen en M34-res. Todas las convoluciones llevan batch normalization, omitida en la tabla.

| Etapa | M3 | M5 | M11 | M18 | M34-res |
|---|---|---|---|---|---|
| Entrada | 32000×1 | 32000×1 | 32000×1 | 32000×1 | 32000×1 |
| conv1 | `[80/4, 256]` | `[80/4, 128]` | `[80/4, 64]` | `[80/4, 64]` | `[80/4, 48]` |
| pool1 | *Maxpool 4* → Tabla 1: salida 2000×n | ← | ← | ← | ← |
| etapa 1 | `[3, 256]` | `[3, 128]` | `[3, 64] × 2` | `[3, 64] × 4` | `[3,48; 3,48] × 3` |
| pool2 | *Maxpool 4* → Tabla 1: salida 500×n | ← | ← | ← | ← |
| etapa 2 | — | `[3, 256]` | `[3, 128] × 2` | `[3, 128] × 4` | `[3,96; 3,96] × 4` |
| pool3 | *Maxpool 4* → Tabla 1: salida 125×n | ← | ← | ← | ← |
| etapa 3 | — | `[3, 512]` | `[3, 256] × 3` | `[3, 256] × 4` | `[3,192; 3,192] × 6` |
| pool4 | *Maxpool 4* → Tabla 1: salida 32×n | ← | ← | ← | ← |
| etapa 4 | — | — | `[3, 512] × 2` | `[3, 512] × 4` | `[3,384; 3,384] × 3` |
| cabezal | Global average pooling (salida 1×n) → Softmax | ← | ← | ← | ← |

Detalles que se leen mal si uno no cuenta con cuidado:

- **Hay cuatro max-pooling, no cinco.** La última etapa convolucional va **directo** al global average pooling. Eso es coherente con la implementación del laboratorio (`avgPool = nn.AvgPool1d(20)` para M18, sin `pool5`).
- **M3 solo tiene dos etapas.** Sus dos convoluciones (conv1 y una de kernel 3) van seguidas de dos pooling, y de ahí al GAP. Las columnas vacías de M3 y M5 en la Tabla 1 no son omisiones: esas etapas no existen.
- **El conteo de "capas con peso" incluye el softmax.** M3 = 2 convoluciones + 1 capa lineal final = 3. M34-res = 33 convoluciones + 1 lineal = 34. Por eso el abstract dice indistintamente *"the CNN with 3 weight layers"* y, en la introducción, *"networks with 2 convolutional layers"*: hablan del mismo M3.
- **M11 tiene 3 convoluciones en la etapa 3**, no 2. Es la única etapa asimétrica de la familia.

### Conteo de parámetros: verificación independiente

Recalculé los parámetros de cada red desde cero (convoluciones con sesgo, dos parámetros afines por canal de batch norm, capa lineal final a 10 clases):

| Red | Mi cálculo | Tabla 1 | Composición de canales |
|---|---|---|---|
| M3 | **221 194** (0.22M) | 0.2M ✓ | 256 |
| M5 | **559 114** (0.56M) | 0.5M ✓ | 128 → 128 → 256 → 512 |
| M11 | **1 786 442** (1.79M) | 1.8M ✓ | 64 → 64 → 128 → 256 → 512 |
| M18 | **3 683 786** (3.68M) | 3.7M ✓ | 64 → 64 → 128 → 256 → 512 |
| M34-res | **3 978 490** (3.98M) | 4M ✓ | 48 → 48 → 96 → 192 → 384 |

Los cinco cuadran. Un detalle que el cálculo revela: **M34-res usa 48/96/192/384 canales, exactamente el 75% del ancho de ResNet-34** (64/128/256/512). Con el ancho completo la red pesaría ~7M en vez de 4M. El estrechamiento es deliberado, para que M34-res sea comparable en tamaño a M18 y así aislar el efecto de la profundidad del efecto de la capacidad.

### Dónde vive realmente la capacidad

| Red | Parte más pesada | Parámetros | % del total |
|---|---|---|---|
| M3 | la única conv de kernel 3 (256→256) | 197 376 | **89.2%** |
| M18 | etapa 4 completa (4 convs de 512 canales) | 2 758 656 | **74.9%** |

En M3 el 89% de los pesos están en **una sola capa**, la de 256→256 con kernel 3. Esa capa opera sobre una secuencia de longitud 1993 y consume 392 millones de MACs, más que toda la red M5 junta. **M3 es simultáneamente la red más cara de la familia y la peor de la familia** — un detalle que la Tabla 2 confirma pero no comenta (77 s/época contra 63 s de M5). Volveré sobre esto en la Sección 8.

### La auditoría del global average pooling

El paper es tajante sobre esta decisión y le dedica un párrafo entero de la Sección 2:

> *"Most deep convolutional networks for classification use 2 or more fully connected (FC) layers of high dimensions (e.g., 4096 in [15, 5]) for discriminative modeling, leading to a very high number of parameters. We hypothesize that most of the learning occurs in the convolutional layers, and with a sufficiently expressive representation from convolutional layers, no FC layer is necessary."*

Mecánicamente, el GAP colapsa cada mapa de features a **un solo escalar** promediando sobre todo el eje temporal: un tensor $(C, L)$ se convierte en $(C, 1)$. De ahí sale un vector de dimensión $C$ (512 en M5/M11/M18) que alimenta una única capa lineal a 10 clases.

**Por qué la red tiene tan pocos parámetros: la cuenta explícita.** Sin GAP habría que aplanar el mapa completo. En M3, después del último pooling el mapa es $256 \times 500$ (longitud idealizada de la Tabla 1) = 128 000 activaciones. Conectar eso a una capa densa de 1000 unidades cuesta:

$$128\,000 \times 1000 = 128\ \text{millones de parámetros}$$

en **una sola matriz**. Y efectivamente: la Tabla 5 reporta **M3-fc con 129M parámetros** contra los 0.2M de M3. Reproduje la cuenta completa (128M de la primera densa + 1.0M de la segunda + 10K de la salida + 0.22M de las convoluciones) y da 129.2M: cuadra al decimal. Lo mismo con M5-fc: $512 \times 32 \times 1000 = 16.4$M más el resto da 17.95M, y la Tabla 5 dice 18M.

Es decir: **el 99.83% de los parámetros de M3-fc están en el cabezal densificador, y ninguno de ellos sirve.** M3-fc obtiene 46.82% contra 56.12% de M3. Multiplicar los parámetros por 585 **empeora** el resultado en 9.3 puntos.

Hay tres razones por las que el GAP funciona aquí, y solo la primera está en el paper:

1. **Presión sobre las convoluciones.** *"By removing FC layers, the network is forced to learn good representation in the convolutional layers, potentially leading to better generalization."* Si la única operación entre la última convolución y el softmax es un promedio y una proyección lineal, cada canal tiene que convertirse en un detector cuya *tasa de activación media* sea directamente informativa de la clase.
2. **Invarianza temporal total.** Promediar sobre el eje temporal hace la salida invariante a **dónde** ocurrió el evento dentro de los 4 segundos. En sonido ambiental eso es exactamente el sesgo inductivo correcto: un ladrido es un ladrido esté al principio o al final del clip.
3. **Longitud variable.** El paper lo declara como propiedad: *"can be applied to audio of varying lengths"*. Con GAP la arquitectura no tiene ninguna dependencia del largo de la entrada. (Nota de implementación: el código del laboratorio usa `nn.AvgPool1d(30)` con la longitud **hardcodeada**, lo cual rompe justamente esa propiedad. `nn.AdaptiveAvgPool1d(1)` la restituye. Ver Sección 14.)

**Auditoría honesta del argumento.** La Tabla 5 muestra que **las cuatro** variantes con capas densas son peores que sus contrapartes con GAP:

| Modelo | Con GAP | Con FC | Δ | Parámetros GAP → FC |
|---|---|---|---|---|
| M3 | 56.12% | 46.82% | **−9.30** | 0.2M → 129M |
| M5 | 63.42% | 62.76% | −0.66 | 0.5M → 18M |
| M11 | 69.07% | 68.29% | −0.78 | 1.8M → 1.8M (?) |
| M18 | 71.68% | 64.93% | **−6.75** | 3.7M → 8.7M |

Pero dos de los cuatro deltas (−0.66 y −0.78) están **por debajo del ruido de un único fold de test** (ver Sección 10: el error estándar de una accuracy de ~0.7 sobre ~870 muestras es de ±1.5 puntos). La evidencia real a favor del GAP son M3 y M18, donde los deltas son grandes; en los casos intermedios la afirmación *"fully convolutional networks perform comparably or better"* solo puede sostener el "comparably".

---

## 6. M34-res: bloques residuales en 1D

### La adaptación

La Figura 1b define el bloque, y merece leerse con atención porque **no es el bloque de He et al.**:

```
x ──┬── Conv(k=3) → BatchNorm → ReLU → Conv(k=3) → BatchNorm ──┐
    │                                                          + ── BatchNorm → ReLU
    └──────────────── atajo (identidad) ───────────────────────┘
```

Dos convoluciones de kernel 3 por bloque (*"A resblock consists of two convolution layers"*), formulación estándar $\mathcal{F}(x) = \mathcal{H}(x) - x$ explicada en la Sección 2, y **una batch normalization después de la suma**. Esa última BN post-suma no aparece en el bloque original de ResNet (donde el orden es conv-BN-ReLU-conv-BN-suma-ReLU) y el texto no la menciona ni la justifica. Es una desviación silenciosa; en la práctica, normalizar después de la suma reduce el efecto de "camino de identidad limpio" que hace atractivo al bloque residual, porque la señal del atajo ya no llega intacta a la siguiente capa.

La estructura de bloques por etapa es **3, 4, 6, 3** — idéntica a ResNet-34 — con 48, 96, 192 y 384 canales.

### Qué hace el atajo cuando cambia el número de canales

**El paper no lo especifica.** Ni el texto ni la Figura 1b dicen nada sobre la transición 48→96, 96→192 o 192→384, y la figura solo dibuja una flecha de identidad. He et al. ofrecían tres opciones: (A) atajo identidad con relleno de ceros en los canales nuevos, sin parámetros; (B) proyección $1\times1$ solo donde cambia la dimensión; (C) proyección $1\times1$ en todos los atajos.

Mi cálculo de parámetros da **3 978 490 sin ninguna proyección**, que redondea exactamente al "4M" de la Tabla 1. Agregar proyecciones $1\times1$ en las tres transiciones sumaría ~60K parámetros (4.04M, que también redondearía a "4M"), así que la evidencia numérica **no discrimina** entre la opción A y la B. Lo honesto es decir que el detalle es indeterminado a partir de este PDF.

Sí hay algo que la arquitectura resuelve limpiamente y vale la pena notar: **el atajo nunca tiene que cambiar la longitud temporal.** En ResNet-34 el primer bloque de cada etapa lleva stride 2, así que el atajo debe submuestrear además de proyectar. Aquí el submuestreo lo hacen los max-pooling que están **entre** las etapas, fuera de todo bloque residual. Dentro de cada bloque, la longitud solo se reduce en 4 muestras por las dos convoluciones sin padding — lo cual, dicho sea de paso, es un problema en sí mismo: si las convoluciones no llevan padding, $\mathcal{F}(x)$ es **más corto** que $x$ y la suma requiere recortar el atajo. El paper no menciona ni el padding ni el recorte.

### Qué gana sobre M18: nada donde importa

| | M18 | M34-res |
|---|---|---|
| Accuracy de **entrenamiento** | 96.72% | **99.21%** |
| Accuracy de **test** | **71.68%** | 63.47% |
| Brecha train − test | 25.0 pts | **35.7 pts** |
| Tiempo/época | 98 s | 124 s |

El paper es directo: *"M34-res only achieves 63.47% test accuracy. This is due to overfitting. We observe that with residual learning we have no problem optimizing deep networks like M34-res, and M34-res reaches an extremely high training accuracy of 99.21%."* Y añade una observación negativa que raramente se cita: *"We also observe overfitting in a residual variant of M11 network (not shown here) which reaches higher training accuracy but a lower test accuracy (by 0.17%)."* Es decir, **los atajos residuales no ayudaron en ninguna profundidad probada**, ni siquiera a 11 capas.

La conclusión del paper es la correcta: *"We believe that our dataset is too small to train M34-res without further regularization."* Con ~7 900 clips de entrenamiento y sin dropout, sin data augmentation y con $\ell_2$ de solo $10^{-4}$, una red de 33 convoluciones memoriza el conjunto.

**El matiz que el paper no señala.** El argumento clásico de ResNet es que los atajos resuelven un problema de **optimización**: las redes muy profundas sin atajos alcanzan peor error de *entrenamiento*, no solo de test. Aquí ese problema **nunca se manifestó**: M18, sin atajos, ya llegaba a 96.72% de accuracy de entrenamiento. No había degradación que arreglar. Lo único que los atajos compraron fue la capacidad de llevar el entrenamiento de 96.72% a 99.21%, es decir, **más sobreajuste**. La contribución real que hace posible entrenar M18 y M34-res no son los atajos sino la batch normalization, y de eso hay evidencia dura (Tabla 6, ver Sección 9).

---

## 7. Datasets y protocolo experimental

### Qué usa el paper

**Un solo dataset: UrbanSound8K** (Salamon, Jacoby y Bello, ACM Multimedia 2014, ref. [13]). El paper describe: *"UrbanSound8k dataset which contains 10 environmental sounds in urban areas, such as drilling, car horn, and children playing. The dataset consists of 8732 audio clips of 4 seconds or less, totalling 9.7 hours."*

Vale la pena hacer la aritmética de consistencia: $8732 \times 4\ \text{s} = 34\,928\ \text{s} = 9.70\ \text{h}$. Es decir, **el 9.7 horas del paper es la cota superior asumiendo que todos los clips duren exactamente 4 s**, no la duración real del audio (muchos slices son más cortos). Consistente, pero optimista.

Las diez clases (según el `metadata` del dataset, que el laboratorio también lista): `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jack_hammer`, `siren`, `street_music`.

**No hay otros datasets.** Ni ESC-50, ni DCASE, ni AudioSet, ni ningún corpus de voz. La conclusión menciona la esperanza de que las arquitecturas sirvan para reconocimiento de voz (*"hold the promise to improve CNNs for speech recognition and other time-series modeling"*), pero es una proyección, no un resultado. Ver Sección 13.

### Preprocesamiento

| Parámetro | Valor | Cita |
|---|---|---|
| Sample rate | **8 kHz** (remuestreado) | *"For computational speed, the audio waveforms are down-sampled to 8kHz"* |
| Normalización | media 0, varianza 1 | *"standardized to 0 mean and variance 1"* |
| Longitud de entrada | **32 000 muestras = 4.0 s exactos** | Tabla 1: *"Input: 32000x1 time-domain waveform"* |
| Data augmentation | **ninguna** | *"We shuffle the training data but do not perform data augmentation"* |
| Optimizador | Adam | Sección 3 |
| Épocas | 100–400 "hasta convergencia" | Sección 3 |
| Inicialización | Glorot, desde cero, sin preentrenamiento | Sección 3 |
| Regularización | $\ell_2$ con coeficiente $10^{-4}$; **sin dropout** | Secciones 2 y 3 |
| Framework / hardware | TensorFlow, una Titan X | Sección 3 |

El paper **no reporta el learning rate**, ni el batch size, ni el criterio exacto de "convergencia", ni cómo se usó el conjunto de validación para seleccionar el modelo final. Ese hueco importa (ver Sección 10).

Consecuencia acústica de los 8 kHz: la frecuencia de Nyquist queda en **4 kHz**, así que todo el contenido espectral por encima de 4 kHz se descarta. La propia clase 39 advierte lo contrario en el slide "Can We Use Raw Audio Data": *"To avoid loss of info, we need to sample audio data using a high sample rate 15-20KHz (44.1 Khz for music)"*. El paper elige explícitamente el compromiso opuesto —velocidad sobre información— y lo dice.

### El protocolo de folds, y por qué es el punto crítico del laboratorio

**Qué hace el paper:** *"We use the official fold 10 to be our test set, and the rest for training and validation."*

Notar dos cosas de inmediato. Primero, el paper **respeta las particiones oficiales** — no rebaraja. Segundo, el paper **no hace 10-fold cross-validation**: entrena una sola vez y evalúa sobre un solo fold, y por eso la nota al pie 3 tiene que aclarar que su protocolo difiere del de Piczak. El laboratorio de la clase, en cambio, sí plantea el esquema de K-fold completo (K=10), que es lo que el dataset recomienda.

**Por qué el dataset insiste en no rebarajar.** UrbanSound8K no es una colección de 8 732 grabaciones independientes. Es una colección de **slices** extraídos de un número mucho menor de grabaciones de campo originales de Freesound. De una única grabación de diez minutos de un martillo neumático se recortan varios slices de 4 segundos. Esos slices comparten:

- el mismo equipo de grabación y su respuesta en frecuencia,
- el mismo ruido de fondo (el mismo tráfico, el mismo viento, la misma acústica de la calle),
- la misma fuente física concreta (ese martillo específico, ese perro específico),
- a menudo, secciones de audio **literalmente solapadas** o casi idénticas.

Los diez folds oficiales están construidos precisamente para que **todos los slices de una misma grabación original caigan en el mismo fold**. Esa es la única razón por la que existen como archivos preparticionados en vez de dejar que cada quien haga su propio split.

**Qué pasa si se rebaraja: la trampa.** Si se junta todo y se hace un split aleatorio 80/20, con altísima probabilidad el slice #3 de una grabación queda en entrenamiento y el slice #4 de la **misma** grabación queda en test. El modelo no necesita aprender qué es un martillo neumático: le basta con memorizar la firma de ese micrófono, ese ruido de fondo y esa fuente. Al evaluar, reconoce la grabación, no la clase.

El resultado es un accuracy que sube a **más del 90%** y que no significa nada. El laboratorio de la clase 39 lo dice explícitamente, refiriéndose a un práctico anterior:

> *"Si recuerdan de su primer laboratorio, entrenamos un MLP sencillo con este mismo dataset que tuvo resultados de más de 90% en el set de test. Sin embargo, este resultado es poco fiable por una regla básica que fue violada al entrenar ese modelo en ese momento: el set de entrenamiento no debe estar correlacionado al set de test."*

El calibre de la anomalía es fácil de establecer: en 2016–2017 el estado del arte publicado sobre UrbanSound8K con el protocolo correcto rondaba el 70–79%. Un MLP sencillo sobre MFCC dando >90% no es un modelo bueno: es un test set contaminado. **La regla operativa es que un número que supera al estado del arte por veinte puntos con un modelo trivial es una alarma de fuga de datos, no un resultado.**

Hay una segunda razón, menos dramática pero igual de real: los folds oficiales están **balanceados por clase y por condiciones de grabación**, de modo que los resultados de distintos papers sean comparables. Rebarajar destruye esa comparabilidad aunque uno tenga cuidado con las grabaciones.

**Un tercer punto, que el paper implícitamente ilustra:** aun respetando los folds, evaluar sobre **uno solo** tiene un problema distinto — la varianza. Con ~870 clips en un fold, el error estándar de una accuracy de 0.72 es

$$\sqrt{\frac{0.72 \times 0.28}{870}} \approx 0.015$$

es decir **±1.5 puntos de error estándar, ±3 puntos de intervalo de confianza al 95%**. Esa es exactamente la razón por la que el dataset pide reportar el promedio de los diez folds: no solo para evitar la fuga, sino para tener un número con precisión suficiente. El paper de Dai no lo hace, y varias de sus comparaciones menores caen dentro de ese margen (ver Sección 10).

### La trampa que el propio notebook del laboratorio reintroduce

Auditando el código del práctico encontré un caso concreto y verificable de la misma fuga que el notebook enseña a evitar. La clase `AudioDataset` construye la lista de archivos así:

```python
self.audio_paths = glob.glob(audio_paths + '/*' + str(self.folds) + '/*')
```

con `folds` recibiendo una **lista** de Python. Con `train_folds = [2,3,4,5,6,7,8,9,10]`, la interpolación produce:

```
./UrbanSound8K/audio//*[2, 3, 4, 5, 6, 7, 8, 9, 10]/*
```

Los corchetes **no son una lista para `glob`: son una clase de caracteres**. El patrón `*[2, 3, 4, ...]` significa "cualquier cosa seguida de **un** carácter perteneciente al conjunto `{2, ',', ' ', 3, 4, 5, 6, 7, 8, 9, 1, 0}`". Los directorios se llaman `fold1` … `fold10`, y **todos** terminan en un dígito de ese conjunto: `fold1` termina en `1`, `fold10` termina en `0`, `fold7` en `7`.

Lo verifiqué construyendo la estructura de directorios y ejecutando el glob:

```
patrón de train  → fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10   ← ¡los diez!
patrón de test   → fold1
```

**El conjunto de entrenamiento incluye el fold 1 completo, que es exactamente el conjunto de test.** No es una correlación entre slices hermanos: es el test set entero dentro del train set. El accuracy de test que reporte el notebook tal como está es, en el límite, accuracy de entrenamiento.

El arreglo es de una línea:

```python
self.audio_paths = []
for f in self.folds:
    self.audio_paths += glob.glob(os.path.join(audio_paths, f'fold{f}', '*.wav'))
```

La lección meta es más valiosa que el bug: **la fuga de datos casi nunca se ve en las métricas, se ve en el código de carga de datos.** El notebook dedica una celda de markdown entera a explicar por qué no hay que correlacionar train y test, y la implementación que viene tres celdas después lo hace de todos modos.

### Un segundo problema del notebook: el remuestreo

```python
zero_need = 160000 - n
audio_new = F.pad(audio, (zero_need//2, zero_need//2), 'constant', 0)
audio_new = audio_new[:, ::5]     # 160000 → 32000
```

Dos observaciones:

1. **`[::5]` es decimación sin filtro antialias.** Quedarse con una de cada cinco muestras sin aplicar antes un pasabajos en $f_s/10$ **repliega** todo el contenido por encima de la nueva Nyquist sobre las frecuencias bajas. El paper dice *"down-sampled to 8kHz"*, que en cualquier implementación seria (y el notebook instala `resampy` justo para eso) implica filtrado previo. La decimación cruda inyecta aliasing en cada ejemplo del dataset.
2. **El factor 5 es fijo, pero el sample rate de UrbanSound8K no lo es.** Los archivos vienen a la frecuencia original de cada grabación de Freesound (44.1 kHz en la mayoría, pero también 48 kHz, 24 kHz, 22.05 kHz, 16 kHz, 11.025 kHz y 8 kHz). Dividir por 5 uniformemente hace que un archivo a 44.1 kHz quede a 8.82 kHz y uno a 16 kHz quede a 3.2 kHz. **La red ve el mismo evento acústico a escalas de tiempo distintas según el archivo**, lo que rompe la premisa central del paper: que el kernel de 80 corresponde a 10 ms. Con `torchaudio.transforms.Resample(orig_freq=rate, new_freq=8000)` se arregla y además desaparece el aliasing.

---

## 8. Resultados

### Tabla 2: la curva de profundidad

| Modelo | Capas conv | Parámetros | Accuracy test | Δ vs. anterior | Tiempo/época | MACs de las convs (mi cálculo) |
|---|---|---|---|---|---|---|
| M3 | 2 | 0.2M | 56.12% | — | 77 s | **555M** |
| M5 | 4 | 0.5M | 63.42% | **+7.30** | 63 s | 276M |
| M11 | 10 | 1.8M | 69.07% | **+5.65** | 71 s | 215M |
| M18 | 17 | 3.7M | 71.68% | **+2.61** | 98 s | 365M |
| M34-res | 33 | 4M | 63.47% | **−8.21** | 124 s | — |

**Dónde ayuda la profundidad, y dónde satura.** Los incrementos son +7.30, +5.65, +2.61: monótonos pero con **retornos claramente decrecientes**. Duplicar de 2 a 4 capas convolucionales compra 7.3 puntos; pasar de 10 a 17 compra 2.6. La extrapolación natural es que hacia 20–25 capas la curva se aplana, y la caída a M34-res (−8.21) confirma que el régimen cambió: ahí ya no manda la capacidad de representación sino la capacidad de memorización frente a un dataset de 7 900 ejemplos.

**El control de capacidad, y por qué es la parte fuerte del paper.** Un escéptico diría que los deltas se explican por el conteo de parámetros: M18 tiene 18× los parámetros de M3. El paper anticipa la objeción y la mata con la Tabla 4:

| Modelo | Parámetros | Accuracy | Comparación |
|---|---|---|---|
| M3 | 0.2M | 56.12% | referencia |
| **M3-big** (50% más filtros: 384 en conv1) | **0.5M** | **57.55%** | +1.43 con 2.5× parámetros |
| M5 | 0.5M | 63.42% | **+7.30 sobre M3, con los mismos 0.5M que M3-big** |
| **M5-big** (100% más filtros: 256 en conv1) | **2.2M** | **63.30%** | **−0.12** respecto a M5 |
| M11 | 1.8M | 69.07% | **+5.77 sobre M5-big, con menos parámetros** |

Los dos pares son demoledores. **M3-big contra M5**: mismo presupuesto de parámetros (0.5M), 57.55% contra 63.42%. **M5-big contra M11**: M5-big tiene *más* parámetros (2.2M vs 1.8M) y saca 5.77 puntos menos. El paper lo resume: *"The performance increases cannot be simply attributed to the larger number of parameters in the deep models"* y *"shallow models have limited capacity to capture time-series inputs even with a larger model"*.

Ese es el resultado que justifica el título. No es "más grande es mejor": es **más profundo es mejor, a igualdad de tamaño**.

**El costo: la columna Time es más rara de lo que parece.** M3 (77 s) es **más lento** que M5 (63 s) y que M11 (71 s), pese a tener la décima parte de las capas. Calculé los MACs de las convoluciones de cada red y la explicación es clara: M3 pone **256 canales** en la primera capa y una convolución 256→256 sobre una secuencia de 1993 muestras, lo que suma 555M MACs — el doble que M5 (276M) y 2.6× M11 (215M). Las redes profundas usan **menos canales abajo** (64 en M11/M18) y por eso son más baratas donde la secuencia es larga.

La lección de ingeniería, que el paper enuncia de pasada (*"by using an aggressive down-sampling in the initial layers, very deep networks can be economical to train"*), se puede afinar: en 1D sobre secuencias largas, **el costo lo domina el producto (canales × longitud) en las primeras capas, no la profundidad**. M11 tiene 5× las capas de M3 y cuesta 2.6× menos cómputo. Que aun así M11 tarde 71 s contra 77 s (una mejora de solo 8% pese a 2.6× menos MACs) refleja que la profundidad sí cuesta **latencia** por serialización de kernels de GPU, aunque no cueste FLOPs.

### El experimento del stride, con una advertencia

*"When we use stride 1 instead of 4 in the first convolutional layer for M11, we observe a 3.5x increase in training time but a lower test accuracy (67.37%) after 10 hours of training, compared with 68.42% test accuracy reached in 2 hours by M18."*

Lo que se compara: M11 con stride 1 (67.37% tras 10 h) contra M18 con stride 4 (68.42% tras 2 h). **La cifra 68.42% no aparece en ninguna tabla** y no coincide con el 71.68% de M18 en la Tabla 2. Es una corrida truncada a 2 horas para hacer la comparación de tiempo igualada — legítimo, pero significa que el experimento del stride es más flojo de lo que parece: mezcla dos redes distintas (M11 vs M18) y dos presupuestos de tiempo distintos. La conclusión cualitativa (stride 1 cuesta 3.5× y no compra precisión) es plausible; el número exacto no debe citarse como "M11 con stride 1 da 67.37% contra 71.68%".

### Comparación con la literatura de la época

El paper solo tiene **un** punto de comparación externo, y está en la nota al pie 3: Piczak (MLSP 2015), CNN sobre log-mel, *"∼68% accuracy using a baseline CNN model"* leído de su Figura 4, con **10-fold CV a 44.1 kHz**.

| Sistema | Entrada | Protocolo | Accuracy |
|---|---|---|---|
| M18 (este paper) | onda cruda 8 kHz | fold 10 como test, una corrida | **71.68%** |
| CNN baseline de Piczak, según nota al pie 3 | log-mel 44.1 kHz | 10-fold CV | ~68% |

Todo lo demás que se suele poner en esta tabla —el baseline SVM+MFCC del paper original de UrbanSound8K, los números con augmentation de Salamon y Bello (2017), los resultados con mayoría de votos de Piczak— **no está en este PDF**, y no puedo verificarlos contra él. Si se necesitan para el material de clase, hay que sacarlos de sus fuentes primarias y citarlos como tales, no atribuírselos a Dai et al.

---

## 9. Ablations y análisis

### 9.1 Tamaño del campo receptivo de la primera capa (Tabla 3)

Variantes: `-srf` (*small receptive field*) con RF 8; `-lrf` (*large receptive field*) con RF 320. Todo lo demás igual.

| Modelo | RF = 8 | **RF = 80** | RF = 320 |
|---|---|---|---|
| M11 | 64.78% (**−4.29**) | **69.07%** | 65.67% (**−3.40**) |
| M18 | 65.55% (**−6.13**) | **71.68%** | 65.08% (**−6.60**) |

El paper resume: *"the performance degrades significantly by up to 6.6% compared with M11 and M18 with RF 80"* — el 6.6 es el delta de M18-lrf, el peor.

Tres observaciones que el paper no hace:

1. **La sensibilidad crece con la profundidad.** M11 pierde 3.4–4.3 puntos; M18 pierde 6.1–6.6. Una red profunda depende **más** de que la primera capa entregue un banco de filtros bien resuelto, porque todo lo que construye encima parte de esa base. Es un argumento a favor de la tesis del paper: la profundidad no compensa un front-end malo, lo amplifica.
2. **Con RF equivocado, M18 baja al nivel de M11 e incluso de M5.** M18-lrf (65.08%) está más cerca de M5 (63.42%) que de M18 (71.68%). Es decir: **el tamaño del kernel de la primera capa vale tanto como quince capas de profundidad.**
3. **La ablación es de dos puntos, no una curva.** Solo se probaron 8, 80 y 320 — factores de 10 en cada dirección. No hay evidencia de que 80 sea óptimo frente a, digamos, 40 o 160; solo de que es mejor que valores diez veces mayores o menores.

### 9.2 Batch normalization (Tabla 6)

| Modelo | Train (sin BN) | Test (sin BN) | Test **con** BN (Tabla 2) | Δ |
|---|---|---|---|---|
| M11-no-bn | 98.58% | **69.38%** | 69.07% | **+0.31** (¡mejor sin BN!) |
| M18-no-bn | 99.33% | 62.48% | 71.68% | **−9.20** |
| M34-no-bn | **10.96%** | **11.45%** | 63.47% | **−52.02** |

Este es el ablation más informativo del paper, y hay que leerlo con cuidado porque el texto solo cuenta la mitad.

**El resultado sólido: sin BN, M34 simplemente no entrena.** 10.96% de accuracy de entrenamiento tras 159 épocas, contra un azar de 10% con diez clases balanceadas. *"M34-no-bn could not be optimized without BN and performs close to random guess (10%) after 159 epochs of training."* Eso es evidencia dura de que **la batch normalization, no el aprendizaje residual, es lo que hace entrenable a la red profunda.** M34-res tiene atajos residuales *y* BN; quitarle la BN lo mata pese a los atajos.

**El resultado del medio: en M18 la BN vale 9.2 puntos de test sin ser necesaria para optimizar.** M18-no-bn llega a 99.33% de train (optimiza perfectamente) pero solo 62.48% de test. El paper lo interpreta como regularización: *"Note that M18-no-bn results in lower test accuracy, indicating that BN has a regularization effect [12]."* La lectura es razonable: sin BN, M18 memoriza (99.33% train contra el 96.72% que alcanza con BN) y generaliza peor.

**El resultado incómodo que el paper omite: M11-no-bn es mejor que M11.** 69.38% contra 69.07%. La BN **empeora** M11 en 0.31 puntos. El texto dice *"Without BN, both M11-no-bn and M18-no-bn can be optimized to high training accuracy"* y luego salta directo al caso de M18 para el argumento de la regularización. La conclusión honesta a partir de la Tabla 6 completa es: **el valor de la BN aparece con la profundidad** — nulo (o levemente negativo) a 11 capas, grande a 18, existencial a 34. Que es, dicho sea de paso, la conclusión más interesante y perfectamente compatible con la tesis del paper. Nota adicional: los 0.31 puntos de diferencia están muy por debajo del ruido de un solo fold, así que lo correcto es decir "indistinguibles", no "mejor sin BN".

### 9.3 Cabezal convolucional vs capas densas (Tabla 5)

Ya cubierto en la Sección 5. Dos precisiones sobre la lectura del texto:

- El paper afirma *"FC layers do not improve test accuracy, and in the cases of M3-fc and M11-fc the additional FC layers lead to lower test accuracy"*. Pero **las cuatro** variantes son peores, y **M18-fc pierde 6.75 puntos, casi diez veces más que M11-fc (0.78)**. La elección de M3-fc y M11-fc como ejemplos es arbitraria y omite el segundo caso más fuerte a favor de la propia tesis.
- *"increase training time by 2∼95%"*: verificado. M18-fc 100 s vs 98 s (+2.0%), M11-fc 73 vs 71 (+2.8%), M5-fc 66 vs 63 (+4.8%), M3-fc 150 vs 77 (**+94.8%**). El rango cuadra.

### 9.4 El efecto del sample rate: **no hay ablación**

La consigna suele pedir "el efecto del sample rate", y hay que ser explícito: **el paper no hace ningún experimento de sample rate.** Todo se corre a 8 kHz. Lo único que existe sobre el tema es:

- una decisión de diseño (*"For computational speed, the audio waveforms are down-sampled to 8kHz"*),
- un argumento arquitectónico (*"audio sampling rate could affect the receptive field size in the first layer, since a field size of 80 at 8kHz sampling rate is at a different length scale than at 16kHz"*), que es una **regla de escalado**, no una medición,
- y el reconocimiento, en la nota al pie 3, de que la diferencia de sample rate frente a Piczak (8 kHz vs 44.1 kHz) es un factor de confusión en la comparación.

No hay una fila "M18 a 16 kHz" ni "M18 a 44.1 kHz". Es un hueco importante, porque un escéptico podría argumentar exactamente lo contrario de lo que el paper concluye: que la paridad con log-mel se logró **porque** se remuestreó a 8 kHz, donde el espectro es tan pobre que las ventajas de un front-end bien diseñado se reducen.

### 9.5 Los filtros aprendidos (Figura 2)

Ya descrito en la Sección 4. Resumen de la evidencia visual, con el eje horizontal en 0–4000 Hz y el vertical enumerando los ~64 filtros ordenados por frecuencia de activación:

| Panel | Variante | Patrón | Interpretación del paper |
|---|---|---|---|
| Izquierda | M18 (RF 80) | Diagonal nítida, cobertura uniforme de 0 a 4 kHz | *"well-distributed filters"* |
| Centro | M18-srf (RF 8) | Manchas anchas y superpuestas, sin diagonal, concentradas al medio | *"much more dispersed bands, and thus lower frequency resolution"* |
| Derecha | M18-lrf (RF 320) | Diagonal fina pero saturada abajo, casi vacía arriba de ~2.5 kHz | *"does not have sufficient filters in the high frequency range... cannot effectively respond to local high frequency impulses"* |

Es una visualización honesta y bien elegida: explica **por qué** los números de la Tabla 3 caen, no solo que caen.

---

## 10. Limitaciones

### Reconocidas por el paper

- **El dataset es demasiado chico para M34-res.** *"We believe that our dataset is too small to train M34-res without further regularization."* Con ~7 900 clips de entrenamiento, sin augmentation y sin dropout, 33 convoluciones memorizan.
- **La comparación con log-mel usa protocolos distintos.** Nota al pie 3, ya citada: distinto esquema de evaluación y distinto sample rate.
- **El sobreajuste en redes muy profundas es esperable.** *"Overfitting caused by very deep networks is well documented [6]."*

### No reconocidas, y que hay que tener presentes

**Un único fold de test y una única corrida.** Con ~870 clips en el fold de test, el intervalo de confianza al 95% de una accuracy de ~0.70 es de **±3 puntos**. Eso implica que varias conclusiones del paper son estadísticamente vacías: M11 vs M11-fc (0.78 pts), M11 vs M11-no-bn (0.31), M5 vs M5-big (0.12), M34-res vs M5 (0.05). Las conclusiones que sí sobreviven son las de deltas grandes: la curva de profundidad (+7.3, +5.6, +2.6), la caída de M34-res (−8.2), las ablations de RF (−3.4 a −6.6), el colapso sin BN de M34 (−52). Además, cada modelo se entrenó **una vez**: no hay barras de error por semilla, y en una red entrenada con Adam desde inicialización aleatoria la varianza entre semillas sobre un dataset de este tamaño no es despreciable.

**El sample rate de 8 kHz es una elección severa que condiciona todo.** Nyquist en 4 kHz descarta el contenido por encima de esa frecuencia, que es precisamente donde viven los transientes de banda ancha de `gun_shot`, `jack_hammer` y `drilling`. El paper lo justifica por velocidad, pero eso significa que (a) no sabemos si la familia M escala a sample rates realistas, y (b) la comparación con Piczak, que usa 44.1 kHz, tiene el signo del sesgo indeterminado: la onda cruda pierde información, pero también se simplifica el problema para ambos lados.

**La paridad con features hechos a mano depende enteramente del baseline elegido.** El paper no entrena ningún baseline de features. Toma un número (~68%) leído de una figura de otro paper. Basta elegir otro baseline de la literatura para que la afirmación de paridad se caiga o se refuerce, y el paper no da forma de auditar esa elección. Un experimento de una tarde —correr M18 sobre log-mel del mismo audio a 8 kHz, con el mismo fold de test y el mismo optimizador— habría convertido una afirmación cruzada en una comparación controlada. Es la omisión más costosa del trabajo.

**El costo de cómputo por la longitud de la entrada nunca se cuantifica.** El paper reporta tiempo por época en una Titan X, que es una métrica sensible al framework (TensorFlow 2016), al batch size (no reportado) y a la implementación. No hay FLOPs, no hay latencia de inferencia, no hay uso de memoria. Y la afirmación de que las redes son *"economical to train"* es relativa a otras redes sobre onda cruda: cualquiera de estas redes procesa 32 000 valores de entrada donde una CNN sobre log-mel procesaría un mapa de ~$40 \times 400$. La comparación de costo **contra el pipeline de features** —que es la comparación que importa para decidir qué usar— no está.

**Hiperparámetros sin reportar.** No hay learning rate, ni batch size, ni schedule, ni criterio de parada más allá de "100–400 épocas hasta convergencia". La Sección 3 menciona que el resto de folds se usa para *"training and validation"*, pero nunca se describe cómo se usó la validación ni si el modelo reportado se seleccionó por validación o por la mejor época de test. Con un rango de 100 a 400 épocas y sin criterio explícito, la posibilidad de selección de modelo sobre el test set queda abierta.

**Una sola tarea, un solo dominio.** Todo es clasificación de sonido ambiental sobre un dataset. La conclusión proyecta hacia reconocimiento de voz, pero no hay ni un experimento de voz.

**El bloque residual no está especificado.** Padding, recorte del atajo, comportamiento en la transición de canales, y la BN post-suma que aparece en la figura y no en el texto. Reproducir M34-res exactamente desde el paper no es posible.

---

## 11. Impacto y legado

### La línea directa: la primera capa como banco de filtros

La observación de la Figura 2 —que la capa 1 aprende un banco de filtros pasa-banda— disparó una línea de trabajo entera que consiste en **imponer** esa estructura en vez de esperarla:

| Trabajo | Idea | Qué toma de aquí |
|---|---|---|
| **TD-filterbanks** (Zeghidour et al., 2018) | Inicializar la primera capa con filtros tipo Gabor equivalentes a mel, y dejarla aprender desde ahí | Aceptar que la capa 1 *quiere* ser un banco de filtros |
| **SincNet** (Ravanelli y Bengio, 2018) | Parametrizar cada filtro con dos cortes de frecuencia (función sinc), reduciendo la capa a $2C$ parámetros | La misma observación, llevada a restricción dura |
| **LEAF** (Zeghidour et al., 2021) | Front-end totalmente aprendible: filtrado Gabor + pooling + compresión, todo diferenciable | Reemplazar el pipeline log-mel completo, no solo el filtrado |

Todos comparten el diagnóstico de Dai et al. y difieren en la conclusión: si sabemos que el óptimo es un banco de filtros, ¿por qué gastar 80 pesos libres por filtro?

### La familia M como baseline pedagógica

M5 sobrevivió de la forma más práctica posible: es **la arquitectura del tutorial oficial de PyTorch/torchaudio** de clasificación de comandos de voz (`Speech Command Classification with torchaudio`), con la misma primera capa de kernel 80 y stride 4, las mismas cuatro convoluciones y el mismo global average pooling. Cientos de miles de personas escribieron su primer clasificador de audio sobre onda cruda con esta arquitectura sin saber de dónde venía. El laboratorio de la clase 39 es un caso más de la misma herencia.

### El veredicto histórico: la apuesta era correcta, por otra razón

Hoy los modelos grandes de audio **sí** usan encoders convolucionales sobre la onda cruda:

| Modelo | Front-end | Configuración |
|---|---|---|
| **wav2vec 2.0** (Baevski et al., 2020) | 7 convoluciones 1D sobre onda cruda a 16 kHz | kernels (10,3,3,3,3,2,2), strides (5,2,2,2,2,2,2) → reducción **320×**, una trama cada 20 ms |
| **HuBERT** (Hsu et al., 2021) | El mismo encoder convolucional de wav2vec 2.0 | idéntico |
| **EnCodec** (Défossez et al., 2022) | Encoder convolucional con strides (2,4,5,8) | reducción **320×** |
| **SoundStream** (Zeghidour et al., 2021) | Encoder convolucional análogo | idem |

La onda cruda ganó. El pipeline MFCC/log-mel dejó de ser obligatorio. En ese sentido literal, Dai et al. apostaron bien.

**Pero ganó por una razón que este paper no anticipó, y conviene ser preciso sobre cuál.**

**Primero: los encoders que ganaron son superficiales, no profundos.** El encoder de wav2vec 2.0 tiene **siete** capas convolucionales — está entre M5 y M11, muy lejos de M18 o M34. La profundidad de wav2vec 2.0 no está en el encoder de forma de onda: está en el **Transformer de 12 o 24 capas** que viene después y que opera sobre tramas de 20 ms, no sobre muestras. Es decir, la arquitectura ganadora **no** es "una CNN muy profunda sobre la onda"; es "una CNN poco profunda que convierte la onda en tramas, más un modelo profundo sobre tramas". Estructuralmente eso está más cerca del CLDNN de Sainath (que Dai et al. citaban como el estado del arte a superar) que de M18.

**Segundo: lo que compró la victoria fue el preentrenamiento autosupervisado a escala.** Lo que hizo que la onda cruda superara a log-mel no fue apilar capas supervisadas sobre 9.7 horas de audio etiquetado; fue **entrenar sobre 53 600 horas de audio no etiquetado** con un objetivo contrastivo (wav2vec 2.0) o de predicción de unidades enmascaradas (HuBERT). El régimen que Dai et al. identificaron como la limitación de M34-res —*"our dataset is too small"*— resultó ser **el** cuello de botella, y la solución no fue arquitectónica sino de datos y de objetivo.

Hay una ironía útil aquí. El paper concluye que la profundidad es la variable clave y que el sobreajuste de M34-res es un problema de dataset chico. Ambas cosas son ciertas **dentro del régimen supervisado de 2016**. Cuando se levantó la restricción de datos, la variable clave dejó de ser la profundidad del encoder convolucional: ese encoder se estabilizó en siete capas y toda la profundidad migró al modelo de secuencia. La contribución de Dai que sobrevivió no es "34 capas", es **el submuestreo agresivo temprano con un kernel dimensionado en milisegundos** — el 16× de Dai se volvió el 320× de wav2vec 2.0, y la lógica es exactamente la misma: convertir muestras en tramas lo antes posible, barato, y hacer el trabajo pesado sobre las tramas.

**Tercero, y este es el matiz que suele omitirse: en el dominio propio del paper, los features hechos a mano no fueron desplazados.** En clasificación de sonido ambiental y audio tagging, el estado del arte pasó por **espectrogramas log-mel**, no por onda cruda: las CNN de AudioSet (Hershey et al., 2017), PANNs, y luego el **Audio Spectrogram Transformer** (Gong et al., 2021), que aplica un ViT a un espectrograma log-mel tratado como imagen. La onda cruda ganó en **voz** (wav2vec 2.0, HuBERT) y en **códecs y generación** (WaveNet, EnCodec, SoundStream), donde se necesita resolución a nivel de muestra o donde hay decenas de miles de horas de audio no etiquetado. En etiquetado de sonidos generales —el problema que Dai et al. atacaban— log-mel sigue siendo la entrada por defecto.

El veredicto completo, entonces: **la apuesta de la onda cruda fue correcta, pero el mecanismo que la validó (preentrenamiento autosupervisado masivo) y el dominio donde se validó (voz y códecs) no son los que el paper propuso ni el que estudió.**

---

## 12. Conexión con la clase 39 y el laboratorio

### Dos arquitecturas para el mismo problema

La clase plantea la tensión con exactitud en el slide "Can We Use Raw Audio Data": *"Using a convolutional architecture, we need huge filters or a very deep structure, why?"* y ofrece una tercera salida: *"We can increase the receptive field of neurons in intermediate layers using dilated convolution filters."*

El "Ejemplo 2" del slide es la respuesta de la clase; la familia M es la respuesta del paper. Puestas lado a lado:

| | **Ejemplo 2 del slide** | **Familia M (Dai et al.)** |
|---|---|---|
| Entrada | Onda cruda, 15–20K muestras/s | Onda cruda, 8K muestras/s, 32 000 muestras |
| Capas conv | **4**, dilatadas | **2 a 33**, sin dilatación |
| Filtros | 128, 128, 256, 256 | 48–256 en conv1, hasta 512 arriba |
| Kernels | 20, 10, 10, 5 | **80** en conv1, **3** en todo lo demás |
| Reducción temporal | max-pooling *opcional* | **agresiva y obligatoria**: 16× en dos capas, luego ×4 tres veces más |
| Contexto global | **2 capas LSTM de 256D** | **global average pooling** |
| Clasificador | 2 capas FC de 1024 | **ninguna capa densa oculta** |
| Cómo crece el campo receptivo | **dilatación + recurrencia** | **profundidad + stride/pooling** |

Son dos soluciones al mismo problema, y la comparación se hace con aritmética.

### Cómo crece el campo receptivo: las tres estrategias

Con la recurrencia estándar $r_l = r_{l-1} + (k_l - 1)\,d_l\,j_{l-1}$ y $j_l = j_{l-1}\,s_l$, donde $d$ es la dilatación:

**(a) Kernels enormes.** $r = k$. Campo receptivo lineal en el kernel, parámetros lineales en el kernel, costo $O(N k C_{in} C_{out})$ por capa. Para cubrir 1 segundo a 8 kHz haría falta $k = 8000$: 8 000 pesos por filtro. Es lo que el paper hace **solo en la primera capa** ($k=80$) y con una justificación física precisa, no como estrategia general.

**(b) Profundidad con stride/pooling.** Cada convolución de kernel 3 aporta $2 j$ al campo receptivo, donde $j$ es el salto acumulado. Con un pooling de stride 4 entre etapas, $j$ sigue la progresión $4 \to 16 \to 64 \to 256 \to 1024$. El campo receptivo crece **geométricamente con la profundidad**, y el costo total es una **serie geométrica convergente**:

$$\text{costo} \;\propto\; N\left(1 + \tfrac{1}{4} + \tfrac{1}{16} + \tfrac{1}{64} + \cdots\right) < \tfrac{4}{3}N$$

Es decir: **cinco etapas de convoluciones cuestan menos del doble que una sola etapa.** Ese es el truco. En M18 el resultado es un campo receptivo de **11 980 muestras (1.50 s) con 17 convoluciones**.

**(c) Dilatación.** Cada capa aporta $(k-1)\,d$, con $d$ típicamente duplicándose. Sin stride, $j$ se queda en 1 y **la longitud de la secuencia nunca se reduce**: el costo es $O(N)$ **por cada capa**, es decir $O(N L)$ en total. WaveNet lo usa así: *"each 1, 2, 4, ..., 512 block has receptive field of size 1024"*, y apilando tres bloques (30 capas) llega a ~3 000 muestras, lo que el propio paper de WaveNet cifra en *"about 300 milliseconds"* de contexto para TTS.

Aplicando la fórmula al "Ejemplo 2" del slide (kernels 20, 10, 10, 5):

$$r = 1 + 19 d_1 + 9 d_2 + 9 d_3 + 4 d_4$$

| Dilataciones | Campo receptivo | Equivalente a 16 kHz |
|---|---|---|
| 1, 1, 1, 1 (sin dilatar) | 42 muestras | 2.6 ms |
| 1, 2, 4, 8 | 106 muestras | 6.6 ms |
| 1, 10, 100, 1000 | 5 010 muestras | 313 ms |

De ahí la advertencia del slide de que *"dilation factor depends of application"*: con cuatro capas, la dilatación es el **único** grado de libertad que separa 2.6 ms de 313 ms. Y de ahí también las dos LSTM: con cuatro capas convolucionales, aun dilatadas, no se llega al contexto de segundos que requiere clasificar un clip completo. La recurrencia hace ese trabajo.

**La comparación en una línea:** *ambas estrategias hacen crecer el campo receptivo exponencialmente con la profundidad; la diferencia está en el costo y en lo que se conserva.*

| | Profundidad + stride/pooling (familia M) | Dilatación (WaveNet, Ejemplo 2) |
|---|---|---|
| Crecimiento del RF | Geométrico | Geométrico |
| Longitud de la secuencia interna | Se reduce ×4 por etapa | **Se conserva** |
| Costo total de $L$ capas | $O(N)$ — serie geométrica | $O(N L)$ |
| Resolución temporal de la salida | **Se destruye** (queda una trama cada 2 ms, luego cada 128 ms) | **Se preserva** (una salida por muestra de entrada) |
| Adecuado para | Clasificación con pooling global | **Generación muestra a muestra**, segmentación fina, detección de eventos |
| Contexto global | Global average pooling | Requiere apilar bloques o agregar RNN/atención |

Y ahí está el criterio de elección, que es la conclusión que merece llevarse: **WaveNet usa dilatación porque tiene que emitir una muestra por cada muestra de entrada** — no puede permitirse reducir la resolución. **Dai usa stride y pooling porque solo tiene que emitir una etiqueta por cada 4 segundos** — la resolución temporal es exactamente lo que le sobra, y destruirla es lo que le permite ser profundo y barato al mismo tiempo. No es que una técnica sea mejor: son óptimas para tareas distintas.

### La aritmética que explica el paper entero

Vuelvo a la tabla de campos receptivos de la Sección 4, ahora con el punto que importa:

| Red | RF antes del GAP | En tiempo | Posiciones que promedia el GAP | Cobertura total |
|---|---|---|---|---|
| M3 | 172 muestras | **21.5 ms** | 498 | todo el clip |
| M5 | 1 772 | 222 ms | 30 | todo el clip |
| M11 | 7 052 | 881 ms | 25 | todo el clip |
| M18 | 11 980 | **1.50 s** | 20 | todo el clip |

**Las cuatro redes "ven" el clip completo de 4 segundos.** El campo receptivo nominal, contando el promedio global, cubre las 32 000 muestras en las cuatro. Y sin embargo M3 obtiene 56% y M18 obtiene 72%.

La diferencia no es cuánto ven, sino **cómo**. M3 es un promedio de **498 detectores de 21 milisegundos**: puede detectar la presencia de ciertas texturas espectrales muy locales y contar con qué frecuencia aparecen. Es, esencialmente, un **bag-of-features espectral**: no puede representar ninguna estructura que dure más de 21 ms, así que no distingue "el patrón A seguido del patrón B" de "B seguido de A", ni un ritmo de un martillo neumático de un ruido estacionario con el mismo espectro promedio. M18 es un promedio de **20 detectores de un segundo y medio cada uno**: cada uno de esos detectores puede responder a un evento completo con su ataque, su sostén y su decaimiento, o a un patrón rítmico de varios golpes.

Ese es el contenido operativo de la tesis "profundidad, no features": **la profundidad no compra campo receptivo (el pooling global ya lo daba gratis); compra composicionalidad dentro del campo receptivo.**

Y explica el contraste inicial del paper con Piczak. Una CNN de dos capas sobre log-mel tiene, sobre el espectrograma, un campo receptivo de decenas de tramas — es decir, **cientos de milisegundos de audio**, porque cada píxel de entrada ya resume 25 ms. La misma CNN de dos capas sobre onda cruda tiene 21 ms. **Los features hechos a mano regalan varias capas de profundidad efectiva.** Y por eso hay que pagarlas con capas.

### Qué esperar en el laboratorio: M3 contra M18

Las actividades 4 y 5 del práctico preguntan justamente por las diferencias de aprendizaje entre M3 y M18. Predicciones concretas, con su razón:

**1. M3 sube rápido y se estanca temprano; M18 sube lento y sigue subiendo.** M3 tiene dos capas, así que la señal de gradiente llega íntegra y en pocas épocas alcanza su techo. M18 tarda más en arrancar (17 capas de composición que coordinar) pero no se aplana hasta mucho después. Si se entrenan ambas 20 épocas, es probable ver a M3 plana desde la época ~8 y a M18 todavía mejorando en la 20 — lo que significa que **20 épocas subestiman a M18 más que a M3**, y que la brecha real es mayor que la que muestre el gráfico. El paper corrió 100–400 épocas.

**2. La brecha train−test es la diferencia cualitativa más informativa.** M3 tendrá accuracy de entrenamiento **cercana a la de test**: no le alcanza la capacidad ni para memorizar; está en régimen de *underfitting*. M18 tendrá train muy por encima de test — en el paper, 96.72% de entrenamiento contra 71.68% de test, una brecha de 25 puntos. Esa es la respuesta corta a la actividad 5: **M3 falla por falta de capacidad de representación; M18 acierta más y, al mismo tiempo, sobreajusta**. Son dos regímenes distintos, no dos puntos de la misma curva.

**3. M3 no es más rápida.** En el paper M3 tarda 77 s por época contra 63 s de M5 y 71 s de M11. Es contraintuitivo y vale la pena verificarlo en el laboratorio: M3 tiene 256 canales en la primera capa y una convolución 256→256 sobre una secuencia de ~2 000 muestras, que suma **555M MACs, el doble que M5 y 2.6× M11**. La lección: en 1D sobre secuencias largas, el costo lo domina `canales × longitud` en las capas bajas, no la profundidad.

**4. La matriz de confusión debería mostrar un patrón específico.** Las clases que se distinguen por un transiente corto y espectralmente distintivo (`gun_shot`, `dog_bark`, `car_horn`) deberían ser razonables incluso para M3: 21 ms alcanzan para caracterizar un impulso. Las clases de **ruido estacionario de banda ancha** —`air_conditioner`, `engine_idling`, `drilling`, `jack_hammer`— son las que se separan por su estructura temporal de largo alcance (¿es un zumbido continuo o una serie de golpes periódicos?), y son exactamente donde M3 debería colapsar en un bloque de confusión mutua y donde M18 debería ganar la mayor parte de sus 15 puntos. **Si el análisis de la matriz de confusión muestra ese patrón, es la evidencia empírica directa de la tesis del paper.**

**5. Advertencias de implementación del notebook, en orden de gravedad:**

- **El bug de `glob`** documentado en la Sección 7: el fold de test está dentro del entrenamiento. Hay que arreglarlo antes de sacar cualquier conclusión, o todos los números serán accuracy de entrenamiento disfrazada. Es, además, la ilustración perfecta de la lección que el propio notebook enseña.
- **La decimación `[::5]`**: sin filtro antialias y con factor fijo pese a que los archivos de UrbanSound8K tienen sample rates heterogéneos. Reemplazar por `torchaudio.transforms.Resample`.
- **`lr = 0.01` con Adam** es alto para una red de 17 capas. Si M18 se queda pegada en ~10% (azar), no es la arquitectura: bajar a `1e-3`.
- **`nn.AvgPool1d(20)` con la longitud hardcodeada** rompe la propiedad "fully convolutional / longitud variable" que el paper declara. `nn.AdaptiveAvgPool1d(1)` la restituye y hace el código robusto a cambios de sample rate.
- **M34_RES está mencionado en el enunciado del notebook pero no implementado.** Solo hay M3, M5, M11 y M18. Si se quiere reproducir la caída por sobreajuste hay que escribirlo, con la advertencia de que los detalles del bloque residual no están completamente especificados en el paper (Sección 6).

**6. No esperar los números del paper.** El paper reporta 56.12% para M3 sobre el fold 10, a 8 kHz limpios, tras 100–400 épocas. El laboratorio entrena 20 épocas, sobre el fold 1, con audio decimado sin antialias a un sample rate efectivo que varía por archivo. Los números serán distintos; lo que debe reproducirse es el **orden** (M3 < M5 < M11 < M18) y el **patrón de las curvas de train/test**, no las cifras.

---

## 13. Erratas, matices y cosas que se citan mal

### Errores internos del propio paper

**1. La conclusión dice 71.8%; la Tabla 2 dice 71.68%.** Sección 5: *"achieves 71.8% accuracy"*. Redondeado correctamente, 71.68 → **71.7**, no 71.8. Es un error tipográfico menor pero se propaga: hay citas secundarias que reportan "71.8%".

**2. Tabla 5: M11-fc figura con 1.8M parámetros, exactamente lo mismo que M11 sin capas densas.** Es imposible: agregar dos capas densas de 1000 unidades sobre un mapa aplanado de $512 \times 32$ suma ~17.4M parámetros. Verifiqué las otras tres filas y todas cuadran con las longitudes idealizadas de la Tabla 1 (M3-fc: $256\times500\times1000 + \ldots = 129.2$M ✓; M5-fc: $512\times32\times1000 + \ldots = 17.95$M ✓). Con la misma regla, M11-fc debería ser ~18M y M18-fc ~20M; la tabla dice 1.8M y 8.7M. **Al menos la fila de M11-fc es un error de tipeo**, y la de M18-fc tampoco se reconstruye con ninguna regla obvia. Las accuracies de la Tabla 5 no están en duda, solo la columna de parámetros.

**3. Las longitudes de la Tabla 1 son idealizadas, no las reales.** La tabla anota salidas de 2000, 500, 125 y 32 tras cada pooling, y la Figura 1a anota "length: 8000" tras la primera convolución. Con convoluciones **sin padding** —que es lo que implementa el código de referencia— los valores reales son:

| Punto | Tabla 1 / Fig. 1a | Real (M5) | Real (M18) |
|---|---|---|---|
| tras conv1 | 8000 | 7981 | 7981 |
| tras pool1 | 2000 | 1995 | 1995 |
| tras pool2 | 500 | 498 | 496 |
| tras pool3 | 125 | 124 | 122 |
| tras pool4 | 32 | 30 | 28 |
| antes del GAP | 32 | 30 | **20** |

La divergencia crece con la profundidad porque cada convolución de kernel 3 sin padding pierde 2 muestras. En M18 el mapa final mide 20, no 32 — un 37% menos. El paper nunca declara si usa padding; los conteos de parámetros de la Tabla 5 sugieren que **los cálculos se hicieron con las longitudes idealizadas**, aunque la implementación no lleve padding. No afecta ninguna conclusión, pero sí afecta a quien intente reproducir las formas.

**4. El texto sobre las capas densas elige mal sus ejemplos.** *"in the cases of M3-fc and M11-fc the additional FC layers lead to lower test accuracy"*. Las cuatro variantes `-fc` son peores, y M18-fc pierde 6.75 puntos contra los 0.78 de M11-fc. El caso más fuerte a favor de la tesis del paper quedó fuera de la frase.

**5. La Tabla 6 contradice parcialmente el texto sobre batch normalization.** M11-no-bn obtiene 69.38% contra 69.07% de M11 con BN: **la BN no ayuda a 11 capas**. El texto solo comenta el caso de M18 al argumentar el efecto regularizador. La lectura completa es "el beneficio de la BN aparece con la profundidad", que es más interesante y más defendible. (Los 0.31 puntos están dentro del ruido; lo correcto es decir "indistinguible", no "peor con BN".)

**6. El bloque residual de la Figura 1b tiene una BatchNorm después de la suma**, que no está en el bloque de He et al. y que el texto no menciona.

**7. La cifra 68.42% del experimento del stride no aparece en ninguna tabla** y no coincide con el 71.68% de M18. Es una corrida de M18 truncada a 2 horas. La comparación de stride mezcla dos arquitecturas (M11 con stride 1 vs M18 con stride 4) y dos presupuestos de tiempo, así que su fuerza probatoria es menor de lo que sugiere la redacción.

### Cosas que se dicen del paper y no son ciertas

**"Dai et al. evaluaron sobre varios datasets."** **Falso.** Solo UrbanSound8K. La conclusión proyecta hacia reconocimiento de voz (*"hold the promise to improve CNNs for speech recognition"*), pero no hay un solo experimento de voz en el paper.

**"M34-res es el modelo propuesto / el mejor de la familia."** **Falso.** M34-res queda **cuarto** de cinco, con 63.47%, apenas 0.05 puntos por encima de M5 y 8.21 por debajo de M18. El mejor modelo del paper es **M18**. M34-res existe para mostrar dónde deja de funcionar la profundidad, no como propuesta.

**"El paper demuestra que la onda cruda supera a los features log-mel."** **Falso.** El paper dice `matches` y `competitive`, nunca "supera". Y la comparación es contra un número (~68%) de otro paper con otro protocolo de evaluación y otro sample rate, reconocido en la nota al pie 3.

**"El paper hace 10-fold cross-validation."** **Falso.** *"We use the official fold 10 to be our test set, and the rest for training and validation."* Un solo fold, una sola corrida. El propio paper señala esto como la diferencia con Piczak. (Sí respeta las particiones oficiales, que es lo que evita la fuga; simplemente no promedia sobre las diez.)

**"El paper hace un ablation de sample rate."** **Falso.** Todo se corre a 8 kHz. Lo que hay es una *regla de escalado* del kernel de la primera capa con el sample rate, sin medición.

**"M18 tiene 18 capas convolucionales."** **Impreciso.** Tiene **17 convoluciones más la capa lineal del softmax**. Lo mismo para toda la familia: M3 = 2 conv, M5 = 4, M11 = 10, M18 = 17, M34-res = 33.

**"El aprendizaje residual es lo que permite entrenar redes profundas aquí."** **Impreciso, y el paper da la evidencia en contra.** M18, sin atajos residuales, ya alcanzaba 96.72% de accuracy de entrenamiento: no había problema de optimización que resolver. Lo que sí es indispensable es la **batch normalization**: M34-no-bn se queda en 10.96% de entrenamiento tras 159 épocas, *pese a tener los atajos residuales*. La contribución de los atajos, medida, fue permitir llegar a 99.21% de train — es decir, **más sobreajuste**.

**"El kernel de 80 corresponde a la ventana estándar de MFCC."** **Impreciso.** 80 muestras a 8 kHz son 10 ms, que es el **salto** estándar entre tramas de MFCC; la ventana estándar es de 25 ms. El orden de magnitud es correcto y hay pipelines con ventanas de 10–20 ms (la propia clase 39 los describe así), pero la identidad no es exacta.

**"La primera capa es un banco de filtros."** **Impreciso.** La primera capa es una convolución 1D de 80 taps sin restricciones que **aprende** algo que, tras la FFT y ordenado por frecuencia de activación, **se parece** a un banco de filtros pasa-banda (Figura 2). Nada en la arquitectura la obliga. Esa distinción es precisamente lo que motivó SincNet dos años después.

---

## 14. Cómo se ve hoy

### M5 en PyTorch, con las formas anotadas

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class M5(nn.Module):
    """
    Dai et al. 2017, columna M5 de la Tabla 1: 4 convoluciones + softmax.
    Entrada esperada: (B, 1, 32000) = 4 s de audio mono a 8 kHz, media 0 y varianza 1.
    Parametros: 559.114 (la Tabla 1 lo redondea a 0.5M).
    """

    def __init__(self, n_classes: int = 10, n_input: int = 1):
        super().__init__()
        # Capa 1: kernel 80 = 10 ms a 8 kHz. Stride 4. Es la unica capa con kernel grande:
        # aprende el banco de filtros pasa-banda (Fig. 2 del paper).
        self.conv1 = nn.Conv1d(n_input, 128, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(128)
        # Capas 2-4: patron VGG en 1D. Kernel 3, canales que se duplican al reducir resolucion.
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(4)          # el mismo pooling se reusa 4 veces
        # Sin capas densas ocultas: el global average pooling reemplaza al bloque FC.
        # Adaptativo, no AvgPool1d(30): asi la red acepta cualquier largo de entrada,
        # que es la propiedad "fully convolutional" que declara el paper.
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, n_classes)  # 512*10 + 10 = 5.130 parametros en total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x                                         (B,   1, 32000)   4.000 ms
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        #   conv1: (32000-80)//4 + 1 = 7981    ->   (B, 128,  7981)
        #   pool:  7981 // 4        = 1995     ->   (B, 128,  1995)   RF=92 (11.5 ms), salto=16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        #   conv2: 1995 - 2 = 1993             ->   (B, 128,  1993)
        #   pool:  1993 // 4 = 498             ->   (B, 128,   498)   RF=172 (21.5 ms), salto=64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        #   conv3: 498 - 2 = 496               ->   (B, 256,   496)
        #   pool:  496 // 4 = 124              ->   (B, 256,   124)   RF=492 (61 ms), salto=256
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        #   conv4: 124 - 2 = 122               ->   (B, 512,   122)
        #   pool:  122 // 4 = 30               ->   (B, 512,    30)   RF=1772 (222 ms), salto=1024
        x = self.gap(x)
        #   promedio sobre las 30 posiciones   ->   (B, 512,     1)   cobertura efectiva: 31.468 (~todo el clip)
        x = x.flatten(1)
        #                                      ->   (B, 512)
        return self.fc(x)
        #                                      ->   (B,  10)   logits, para nn.CrossEntropyLoss
```

Tres diferencias deliberadas respecto del código del laboratorio, todas menores pero todas correctas:

- **`AdaptiveAvgPool1d(1)` en lugar de `AvgPool1d(30)`.** La versión con longitud fija rompe la propiedad de longitud variable que el paper declara explícitamente (*"can be applied to audio of varying lengths"*) y obliga a recalcular el número a mano cada vez que cambia el sample rate o la duración.
- **`flatten(1)` en lugar de `permute(0, 2, 1)`.** El notebook mantiene una dimensión de tamaño 1 que después obliga a hacer `output.permute(1, 0, 2)` y `output[0]` en el bucle de entrenamiento para que `CrossEntropyLoss` acepte la forma. Aplanar de una vez elimina esa gimnasia.
- **Devolver logits, no probabilidades.** `nn.CrossEntropyLoss` aplica el log-softmax internamente; el `Softmax` de la Tabla 1 es conceptual.

### Qué cambia a 16 kHz o 22.05 kHz

La regla del paper es explícita y es lo único que hay que respetar: **el kernel de la primera capa se dimensiona en milisegundos, no en muestras.** Manteniendo los 10 ms y el factor de reducción de 16× de las dos primeras capas:

| Sample rate | Kernel de conv1 (10 ms) | Stride | Reducción conv1+pool | Nyquist | Entrada de 4 s | Longitud tras pool1 |
|---|---|---|---|---|---|---|
| **8 kHz** (paper) | **80** | **4** | 16× | 4.0 kHz | 32 000 | 1 995 |
| 16 kHz | **160** | 8 | 32× | 8.0 kHz | 64 000 | 1 995 |
| 16 kHz (alternativa) | **160** | 4 | 16× | 8.0 kHz | 64 000 | 3 991 |
| 22.05 kHz | **220** | 11 | 44× | 11.0 kHz | 88 200 | 1 999 |
| 22.05 kHz (alternativa) | **220** | 4 | 16× | 11.0 kHz | 88 200 | 5 499 |

Hay una decisión de diseño en las dos filas de cada sample rate, y es la parte no trivial:

**Opción A — escalar el stride junto con el kernel.** Con kernel 160 y stride 8 a 16 kHz, la longitud tras el primer pooling vuelve a ser 1 995: **el resto de la red queda idéntico, el costo idéntico y las longitudes intermedias idénticas**. Lo que cambia es que el mapa tras las dos primeras capas tiene una trama cada 32 muestras = 2 ms, la misma resolución temporal que a 8 kHz. Es la opción por defecto y la que preserva todos los cálculos de campo receptivo del paper **en segundos**.

**Opción B — mantener el stride en 4.** Se conserva más resolución temporal (una trama cada 1 ms) a costa de duplicar la longitud de todos los mapas intermedios y, por lo tanto, el cómputo. Solo tiene sentido si la tarea necesita esa resolución (detección de eventos, alineamiento) y no basta con clasificación por clip.

Dos consecuencias que no son obvias:

1. **Los filtros de la primera capa cubren más espectro, no más tiempo.** A 16 kHz, un kernel de 160 taps sigue cubriendo 10 ms, pero ahora tiene que representar un rango de 0 a 8 kHz en vez de 0 a 4 kHz. Con el doble de taps la resolución frecuencial ($\Delta f \approx f_s/N$) se mantiene en ~100 Hz, pero hacen falta **el doble de bandas** para cubrir el espectro con la misma densidad. Por eso, al subir el sample rate, conviene **aumentar también el número de filtros de conv1** —de 128 a 192 o 256 en M5— o aceptar una cobertura espectral más rala. El paper no discute esto porque nunca cambia el sample rate.
2. **A 22.05 kHz el kernel de 220 con stride 11 es una elección incómoda** (strides que no son potencias de 2 complican el alineamiento con los pooling de 4). En la práctica es preferible **remuestrear a 16 kHz** —lo que hacen wav2vec 2.0, HuBERT y prácticamente todo el ecosistema de voz— y usar la fila estándar de 160/8. Si el dominio es música, donde el contenido por encima de 8 kHz sí importa, la familia M no es la arquitectura adecuada de todos modos: conviene un front-end tipo log-mel con más bandas, o un encoder con reducción mucho más agresiva al estilo de EnCodec (strides 2, 4, 5, 8 → 320×).
