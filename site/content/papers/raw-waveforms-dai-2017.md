---
title: "Very Deep CNN para formas de onda crudas — la familia M (2017)"
weight: 427
math: true
---

{{< paper-card
    title="Very Deep Convolutional Neural Networks for Raw Waveforms"
    authors="Wei Dai, Chia Dai, Shuhui Qu, Juncheng Li, Samarjit Das (CMU, Stanford, Bosch)"
    year="2017"
    venue="ICASSP 2017 / arXiv:1610.00087"
    pdf="/papers/raw-waveforms-dai-2017.pdf" >}}
El paper hace una sola pregunta y la responde con un barrido limpio: **¿por qué las CNN sobre onda cruda rendían mediocre en 2016 — porque la onda cruda es mala entrada, o porque nadie había ido lo suficientemente profundo?** La respuesta es la segunda. Los autores construyen una familia de cinco redes 1D **completamente convolucionales** —M3, M5, M11, M18 y M34-res— idénticas en filosofía y distintas solo en profundidad, y las miden sobre [UrbanSound8K](/papers/urbansound8k-salamon-2014) con la onda a 8 kHz como única entrada. El resultado es una curva monótona: **56.12% → 63.42% → 69.07% → 71.68%**, es decir **15.56 puntos absolutos que compra la profundidad** entre 2 y 17 capas convolucionales, antes de que M34-res se desplome a 63.47% por puro sobreajuste. Tres piezas de ingeniería lo hacen posible: una primera capa con **kernel de 80 muestras = 10 ms**, dimensionada en milisegundos y no en muestras, que termina aprendiendo por su cuenta un banco de filtros pasa-banda; un **submuestreo agresivo de 16× en las dos primeras capas** que vuelve barata toda la profundidad posterior; y **batch normalization**, que resulta ser —contra la lectura habitual— más determinante que los atajos residuales. El cabezal es un **global average pooling** sin ninguna capa densa oculta, lo que deja a M18 en 3.7M de parámetros. Es el paper del laboratorio de la [Clase 39](/clases/clase-39), y su contribución más duradera no fue "34 capas" sino la receta de convertir muestras en tramas lo antes posible y hacer el trabajo pesado sobre las tramas — exactamente lo que hacen hoy wav2vec 2.0 y EnCodec.
{{< /paper-card >}}

---

## Contexto: onda cruda contra features hechos a mano en 2016

La primera línea del paper enmarca el problema con precisión: *"Acoustic modeling is traditionally divided into two parts: (1) designing a feature representation of the audio data, and (2) building a suitable predictive model based on the representation."* Ese corte en dos no era pereza de ingeniería. Era la única forma viable de trabajar con audio antes del deep learning, y tenía tres razones sólidas.

**Compresión brutal con pérdida dirigida.** Cuatro segundos de audio a 16 kHz son 64 000 números. Los mismos cuatro segundos en MFCC, con ventanas de 25 ms y salto de 10 ms, son 400 tramas × 13 coeficientes = 5 200 números. Una reducción de ~12×, y la mayor parte de lo descartado es fase e información fuera del rango perceptualmente relevante.

**Invarianzas gratis.** El espectro de magnitud descarta la fase, con lo cual el resultado es aproximadamente invariante al desplazamiento temporal dentro de la ventana. La escala mel comprime las frecuencias altas imitando la resolución del oído. El logaritmo convierte la ganancia multiplicativa (volumen, distancia al micrófono) en un desplazamiento aditivo. La DCT final decorrelaciona los canales, algo indispensable cuando el modelo aguas abajo era un GMM con covarianza diagonal. Todo eso está desarrollado en el fundamento de [representación de audio](/fundamentos/representacion-de-audio).

**Escala de datos.** Con miles —no millones— de ejemplos etiquetados, un modelo que arranca desde 64 000 muestras crudas tiene demasiada libertad. Los features acotaban el espacio de hipótesis.

La objeción moderna la resume el propio paper en una frase: *"it is often challenging and time-intensive to find the right representation in the so-called 'feature-engineering' process, and the often heuristically designed features might not be optimal for the predictive task."*

### Qué se había intentado antes y por qué quedó a la par

Las referencias con las que el paper se posiciona son cuatro trabajos de 2014–2015, todos de reconocimiento de voz:

| Trabajo | Qué hizo | Resultado |
|---|---|---|
| Tüske et al. (INTERSPEECH 2014) | DNN sobre señal temporal cruda para LVCSR | Primera evidencia de que la capa inicial imita transformadas tipo wavelet |
| Golik et al. (INTERSPEECH 2015) | CNN sobre señal temporal cruda para LVCSR | Competitivo, no superior |
| Hoshen et al. (ICASSP 2015) | Modelado acústico desde forma de onda multicanal | Aprende beamforming implícito |
| Sainath et al. (INTERSPEECH 2015) | CLDNN con front-end de forma de onda aprendido | **Iguala** el desempeño de log-mel |

El diagnóstico del paper cabe en una línea y es el eje de todo: *"These works, however, have mostly considered only less deep networks, such as two convolutional layers."* Nadie había fallado con la onda cruda; nadie la había llevado a profundidad. En el CLDNN de Sainath la parte convolucional es de una o dos capas, y el trabajo pesado lo hacen las LSTM que vienen después.

Hay un contraste que el paper usa con precisión quirúrgica. Piczak (MLSP 2015) entrena una CNN de **dos capas convolucionales sobre espectrograma log-mel** en UrbanSound8K y le va bien. Dai entrena una CNN de **dos capas convolucionales sobre onda cruda** (M3) y obtiene 56.12%. Textual: *"This is in contrast with models using the spectrogram as input, which achieve good performance with just 2 convolutional layers, and shows that applying CNN directly on time-series data is challenging."* Misma profundidad, misma tarea, mismo dataset. **Los features hechos a mano compran profundidad efectiva; cuando la entrada es cruda hay que pagarla con capas.**

### La apuesta

*"CNNs have famously achieved performance competitive or even surpassing human-level performance in the visual domains... A common theme among these powerful CNN models is that they are usually very deep."*

La analogía es directa. En visión nadie diseña features SIFT/HOG desde 2012, y lo que hizo posible abandonarlos no fue "aplicar una CNN" sino **apilar decenas de capas** (ver [redes convolucionales](/fundamentos/redes-convolucionales)). Si en audio se aplicaron CNN de dos capas y quedaron a la par de los features, la hipótesis natural es que falta profundidad, no que la onda cruda sea insuficiente.

La contra evidente —y por eso el paper existe— es que en audio la profundidad es cara de una forma que en imágenes no lo es: la entrada son **32 000 valores en una sola dimensión**. La respuesta del paper son tres piezas: batch normalization, aprendizaje residual y, sobre todo, un diseño cuidadoso del submuestreo en las primeras capas.

{{< concept-alert type="clave" >}}
La tesis es un experimento controlado, no una arquitectura nueva: **se fija todo y se mueve solo la profundidad**. Las cinco redes comparten entrada (32 000 muestras a 8 kHz), primera capa (kernel 80, stride 4), kernel 3 en todas las demás convoluciones, cuatro max-pooling de tamaño y stride 4, cabezal de global average pooling sin capas densas, optimizador Adam, inicialización Glorot, $\ell_2$ con $10^{-4}$, y ausencia de dropout y de data augmentation. Lo único que cambia es cuántas convoluciones de kernel 3 hay en cada etapa.
{{< /concept-alert >}}

Y conviene notar lo que la tesis **no** dice. No dice que la onda cruda sea mejor que log-mel. Dice `matches` y `competitive`. La afirmación es de **paridad**: los features hechos a mano dejan de ser necesarios, no dejan de ser buenos.

## La primera capa: el detalle central

### El número 80

`[80/4, C]`: kernel de 80 muestras, stride 4. Todo lo demás en la red usa kernel 3. Esa asimetría es la decisión de diseño central del paper, y se justifica con una regla física, no con una búsqueda de hiperparámetros:

$$\frac{80\ \text{muestras}}{8000\ \text{muestras/s}} = 10\ \text{ms}$$

Textual: *"We thus choose our first layer receptive field to cover a 10-millisecond duration, which is similar to the window size for many MFCC computation."* Y el corolario que casi nadie cita, pero que es la parte generalizable: *"audio sampling rate could affect the receptive field size in the first layer, since a field size of 80 at 8kHz sampling rate is at a different length scale than at 16kHz sampling rate."*

**El 80 no es un hiperparámetro: son 10 ms expresados en las unidades de muestreo del dataset.** Cambiar el sample rate obliga a cambiar el 80. A 16 kHz el kernel equivalente es 160; a 22.05 kHz, 220.

{{< concept-alert type="advertencia" >}}
Un matiz que se repite mal: en el pipeline MFCC clásico (HTK, Kaldi, `librosa` por defecto) la **ventana** es de 25 ms y el **salto** entre tramas es de 10 ms. Los 10 ms del paper coinciden con el *hop*, no con la ventana. La afirmación sigue siendo defendible —la propia clase 39 describe log-mel sobre *"overlapped segments of 10-20ms"*, y en voz las ventanas de 10–20 ms son habituales— pero es del **orden de magnitud** de la ventana de MFCC, no una identidad exacta.
{{< /concept-alert >}}

Hay un contraste adicional que el paper no explota. El salto de la primera capa es de **4 muestras = 0.5 ms**, y tras el max-pooling es de 16 muestras = **2 ms**. Un pipeline MFCC produce una trama cada 10 ms. Es decir, la representación que sale de las dos primeras capas de la familia M está **cinco veces más densamente muestreada en el tiempo** que un espectrograma log-mel convencional, aun después del submuestreo "agresivo". Esa redundancia es parte de lo que las capas de kernel 3 tienen que digerir.

### Lo que aprende se parece a un banco de filtros

El paper hace dos afirmaciones distintas que conviene separar.

**La de diseño:** *"We use a large receptive field in the first convolutional layer to mimic bandpass filters."* Esto es una **intención**. Nada en la arquitectura obliga a que la capa 1 sea un banco de filtros pasa-banda: es una convolución 1D de 80 taps sin restricciones y podría aprender cualquier respuesta al impulso.

**La empírica (Figura 2):** *"All of them learn a filter bank of bandpass filter."* El experimento es directo: se toma cada uno de los 64 kernels de la primera capa de M18 ya entrenada, se le aplica **transformada de Fourier** y se grafica la magnitud espectral, con los filtros **ordenados por su frecuencia de activación**. El eje horizontal va de 0 a 4000 Hz (Nyquist a 8 kHz). El resultado es una **diagonal limpia y bien poblada**: cada filtro tiene un pico estrecho, los picos cubren el espectro completo sin huecos, y no hay dos filtros redundantes.

Eso es literalmente lo que es un banco de filtros mel: un conjunto de respuestas pasa-banda que particiona el espectro. La diferencia es que aquí **el particionamiento se aprendió**, y su distribución no está forzada a seguir la escala mel. Es la validación más directa de la tesis del paper: dada la onda cruda y suficiente presión de tarea, la red **reconstruye por sí sola el primer paso del pipeline que el ingeniero de features habría escrito a mano**.

Lo que la observación **no** dice es que ese banco aprendido sea óptimo. Dos años después **SincNet** (Ravanelli y Bengio, 2018) llevó la observación a su conclusión lógica: si la capa va a aprender filtros pasa-banda, restrinjámosla a serlo. SincNet parametriza cada filtro con **dos números** (las frecuencias de corte de una sinc en el tiempo), reduciendo esa capa de $80 \times C$ pesos libres a $2C$. Esa es la crítica implícita al diseño de Dai: 80 pesos libres por filtro es capacidad desperdiciada si el óptimo vive en una variedad de dimensión 2.

### Por qué 80 y no 8 ni 320: la resolución frecuencial

El paper mide las dos alternativas, y la explicación es de procesamiento de señales elemental. Un filtro FIR de $N$ taps a frecuencia de muestreo $f_s$ tiene una resolución frecuencial acotada por el ancho del lóbulo principal de su ventana, del orden de

$$\Delta f \approx \frac{f_s}{N}$$

| Variante | $N$ | Duración | $\Delta f$ a 8 kHz | Consecuencia |
|---|---|---|---|---|
| M-srf | 8 | 1 ms | ~1000 Hz | Solo cuatro "bandas" distinguibles en todo el espectro |
| **M (paper)** | **80** | **10 ms** | **~100 Hz** | ~40 bandas resolubles, comparable a un banco mel de 40 filtros |
| M-lrf | 320 | 40 ms | ~25 Hz | Resolución fina, pero el filtro promedia sobre 40 ms |

Así se lee la Figura 2 completa. **M18-srf** *"has much more dispersed bands, and thus lower frequency resolution for subsequent layers"*: manchas anchas y superpuestas concentradas en la zona media, sin diagonal. **M18-lrf** *"has fine-grained filters, but does not have sufficient filters in the high frequency range"*: hay diagonal, pero se satura en frecuencias bajas y deja el rango alto casi vacío. El resumen del paper: *"a small RF popularized by vision models is insufficient to capture the necessary bandpass filter characteristics in the first convolutional layer, while a large RF smooths out local structures and cannot effectively detect local impulse patterns."*

Ese último punto es el **compromiso tiempo-frecuencia** —el principio de incertidumbre de Gabor— apareciendo como hiperparámetro de arquitectura. En UrbanSound8K hay clases que son puro transiente (`gun_shot`, `dog_bark`) y clases que son ruido estacionario de banda ancha (`air_conditioner`, `engine_idling`). Los 10 ms son el punto de equilibrio entre ambas.

### El factor de reducción de 16×

Combinando kernel, stride y max-pooling en las dos primeras capas, con convoluciones sin padding:

$$L_1 = \left\lfloor \frac{32000 - 80}{4} \right\rfloor + 1 = 7981 \quad\longrightarrow\quad L_2 = \left\lfloor \frac{7981}{4} \right\rfloor = 1995$$

$32000/1995 = 16.04$. El paper lo redondea a *"reduce the temporal resolution in the first two layers by 16x with large convolutional and max pooling strides to limit the computation cost in the rest of the network"*, citando explícitamente a GoogLeNet. Es el *stem* barato de Inception, trasladado a 1D y llevado al extremo.

Y lo valida con un experimento directo: *"When we use stride 1 instead of 4 in the first convolutional layer for M11, we observe a 3.5x increase in training time but a lower test accuracy (67.37%) after 10 hours of training, compared with 68.42% test accuracy reached in 2 hours by M18."* Con stride 1 se paga 3.5× de cómputo y se **pierde** precisión. El submuestreo agresivo no es un mal necesario: es parte de por qué la red funciona.

### El resto de la red: VGG en 1D

Fuera de la primera capa todo es kernel 3, con nota al pie explícita a VGG: *"we use very small receptive field 3 for all but the first 1D convolutional layers"*. El razonamiento traslada con un matiz interesante: en 2D, tres convoluciones de $3\times3$ cubren un campo receptivo de $7\times7$ con $27C^2$ parámetros en vez de $49C^2$ — apilar sale más barato. En 1D la cuenta se invierte ($9C^2$ contra $7C^2$: **apilar sale más caro en parámetros**), pero se conserva lo importante: más no-linealidades por unidad de campo receptivo y control fino de la capacidad.

La segunda mitad de la herencia VGG es el patrón de canales: *"the reduction of resolution is complemented by a doubling in the number of feature maps"* — filtros básicos abajo, especializados arriba.

## La familia M en detalle

La notación del paper: `[80/4, 256]` es una convolución de campo receptivo 80 con 256 filtros y stride 4; el stride se omite cuando es 1; `[...] × k` son $k$ capas apiladas; los corchetes con **dos filas** son un bloque residual y solo aparecen en M34-res. Todas las convoluciones llevan batch normalization, omitida en la tabla.

| Etapa | M3 | M5 | M11 | M18 | M34-res |
|---|---|---|---|---|---|
| Entrada | 32000×1 | 32000×1 | 32000×1 | 32000×1 | 32000×1 |
| conv1 | `[80/4, 256]` | `[80/4, 128]` | `[80/4, 64]` | `[80/4, 64]` | `[80/4, 48]` |
| **pool1** | *Maxpool 4* | ← | ← | ← | ← |
| etapa 1 | `[3, 256]` | `[3, 128]` | `[3, 64] × 2` | `[3, 64] × 4` | `[3,48; 3,48] × 3` |
| **pool2** | *Maxpool 4* | ← | ← | ← | ← |
| etapa 2 | — | `[3, 256]` | `[3, 128] × 2` | `[3, 128] × 4` | `[3,96; 3,96] × 4` |
| **pool3** | — | *Maxpool 4* | ← | ← | ← |
| etapa 3 | — | `[3, 512]` | `[3, 256] × 3` | `[3, 256] × 4` | `[3,192; 3,192] × 6` |
| **pool4** | — | *Maxpool 4* | ← | ← | ← |
| etapa 4 | — | — | `[3, 512] × 2` | `[3, 512] × 4` | `[3,384; 3,384] × 3` |
| cabezal | Global average pooling → Softmax | ← | ← | ← | ← |
| **Capas conv** | 2 | 4 | 10 | 17 | 33 |
| **Parámetros** | **221 194** | **559 114** | **1 786 442** | **3 683 786** | **3 978 490** |
| Tabla 1 del paper | 0.2M | 0.5M | 1.8M | 3.7M | 4M |

Los cinco conteos fueron recalculados desde cero (convoluciones con sesgo, dos parámetros afines por canal de batch norm, capa lineal final a 10 clases) y **los cinco cuadran** con lo que reporta el paper.

Detalles que se leen mal si uno no cuenta con cuidado:

- **Hay cuatro max-pooling, no cinco.** La última etapa convolucional va **directo** al global average pooling.
- **M3 solo tiene dos etapas.** Sus dos convoluciones van seguidas de dos pooling y de ahí al GAP. Las celdas vacías de M3 y M5 no son omisiones: esas etapas no existen.
- **El conteo de "capas con peso" incluye el softmax.** M3 = 2 convoluciones + 1 lineal = 3. M34-res = 33 convoluciones + 1 lineal = 34. Por eso el abstract dice indistintamente *"the CNN with 3 weight layers"* y *"networks with 2 convolutional layers"*: hablan del mismo M3. **M18 tiene 17 convoluciones, no 18.**
- **M11 tiene 3 convoluciones en la etapa 3**, no 2. Es la única etapa asimétrica de la familia.

### M34-res usa el 75% del ancho de ResNet-34

M34-res no es "una red profunda con atajos": es **la planta de bloques de ResNet-34 copiada dígito por dígito** —3, 4, 6, 3 bloques residuales por etapa, con dos convoluciones cada uno—, pero con **48, 96, 192 y 384 canales** en lugar de 64/128/256/512. Es exactamente el 75% del ancho original. Con el ancho completo la red pesaría ~7M en vez de 4M; el estrechamiento es deliberado, para que M34-res sea comparable en tamaño a M18 y así **aislar el efecto de la profundidad del efecto de la capacidad**.

El bloque tiene además una desviación silenciosa respecto de He et al.: la Figura 1b coloca una **batch normalization después de la suma** (conv-BN-ReLU-conv-BN-suma-**BN**-ReLU), que el texto no menciona ni justifica. Normalizar después de la suma reduce el efecto de "camino de identidad limpio" que hace atractivo al bloque residual. El paper tampoco especifica qué hace el atajo cuando cambia el número de canales (48→96→192→384): el conteo de 3 978 490 parámetros cuadra **sin ninguna proyección**, pero agregar proyecciones $1\times1$ sumaría solo ~60K y también redondearía a "4M", así que la evidencia numérica no discrimina. Reproducir M34-res exactamente desde el paper no es posible.

Sí hay algo que la arquitectura resuelve limpiamente: **el atajo nunca tiene que cambiar la longitud temporal**, porque el submuestreo lo hacen los max-pooling que están *entre* las etapas, fuera de todo bloque residual. En ResNet-34, en cambio, el primer bloque de cada etapa lleva stride 2 y el atajo debe submuestrear además de proyectar.

### El global average pooling y por qué mantiene bajo el conteo

El paper le dedica un párrafo entero a esta decisión:

> *"Most deep convolutional networks for classification use 2 or more fully connected (FC) layers of high dimensions (e.g., 4096) for discriminative modeling, leading to a very high number of parameters. We hypothesize that most of the learning occurs in the convolutional layers, and with a sufficiently expressive representation from convolutional layers, no FC layer is necessary."*

Mecánicamente, el GAP colapsa cada mapa de features a **un solo escalar** promediando sobre todo el eje temporal: un tensor $(C, L)$ se convierte en $(C, 1)$. De ahí sale un vector de dimensión $C$ (512 en M5/M11/M18) que alimenta una única capa lineal a 10 clases: **5 130 parámetros en total**.

La cuenta alternativa explica por qué la familia M es tan liviana. Sin GAP habría que aplanar el mapa completo. En M3, tras el último pooling el mapa es $256 \times 500 = 128\,000$ activaciones. Conectar eso a una densa de 1000 unidades cuesta

$$128\,000 \times 1000 = 128\ \text{millones de parámetros}$$

en **una sola matriz**. Y efectivamente el paper reporta M3-fc con **129M** contra los 0.2M de M3. Es decir: **el 99.83% de los parámetros de M3-fc están en el cabezal densificador, y ninguno de ellos sirve** — M3-fc obtiene 46.82% contra 56.12% de M3. Multiplicar los parámetros por 585 empeora el resultado en 9.3 puntos.

Hay tres razones por las que el GAP funciona aquí, y solo la primera está en el paper:

1. **Presión sobre las convoluciones.** *"By removing FC layers, the network is forced to learn good representation in the convolutional layers, potentially leading to better generalization."* Si lo único entre la última convolución y el softmax es un promedio y una proyección lineal, cada canal tiene que convertirse en un detector cuya *tasa de activación media* sea directamente informativa de la clase.
2. **Invarianza temporal total.** Promediar sobre el tiempo hace la salida invariante a **dónde** ocurrió el evento dentro de los 4 segundos. En sonido ambiental ese es exactamente el sesgo inductivo correcto: un ladrido es un ladrido esté al principio o al final del clip.
3. **Longitud variable.** El paper lo declara como propiedad: *"can be applied to audio of varying lengths"*. Con GAP la arquitectura no tiene ninguna dependencia del largo de la entrada — siempre que se implemente con `AdaptiveAvgPool1d(1)` y no con un `AvgPool1d(30)` de longitud hardcodeada, que es lo que rompe la propiedad en la mayoría de las implementaciones didácticas.

## Campo receptivo y costo

Fuera de la primera capa todo es kernel 3, y en 1D eso es un campo receptivo que crece **muy** lentamente por sí solo: $L$ convoluciones de kernel 3 con stride 1 dan campo receptivo $2L+1$. Para cubrir las 32 000 muestras de la entrada harían falta **16 000 capas**. Ahí está la razón de ser de los cuatro max-pooling de stride 4: cada uno multiplica por 4 el *salto* del mapa, y por lo tanto multiplica por 4 lo que aporta cada convolución posterior.

Con las recurrencias estándar $r_l = r_{l-1} + (k_l - 1)\,j_{l-1}$ y $j_l = j_{l-1}\,s_l$, calculadas capa por capa sin padding:

| Red | Campo receptivo antes del GAP | Equivalente a 8 kHz | Posiciones que promedia el GAP | Cobertura efectiva |
|---|---|---|---|---|
| M3 | 172 muestras | **21.5 ms** | 498 | todo el clip |
| M5 | 1 772 muestras | **222 ms** | 30 | todo el clip |
| M11 | 7 052 muestras | **881 ms** | 25 | todo el clip |
| M18 | 11 980 muestras | **1.50 s** | 20 | todo el clip |

Esta tabla es la explicación cuantitativa de todo el paper, y merece leerse despacio. **Las cuatro redes "ven" el clip completo de 4 segundos**: el campo receptivo nominal, contando el promedio global, cubre las 32 000 muestras en las cuatro. Y sin embargo M3 obtiene 56% y M18 obtiene 72%.

La diferencia no es cuánto ven, sino **cómo**. M3 es un promedio de **498 detectores de 21 milisegundos**: puede detectar la presencia de ciertas texturas espectrales muy locales y contar con qué frecuencia aparecen. Es, esencialmente, un **bag-of-features espectral** — no puede representar ninguna estructura que dure más de 21 ms, así que no distingue "el patrón A seguido del patrón B" de "B seguido de A", ni un ritmo de martillo neumático de un ruido estacionario con el mismo espectro promedio. M18 es un promedio de **20 detectores de un segundo y medio cada uno**: cada uno puede responder a un evento completo con su ataque, su sostén y su decaimiento, o a un patrón rítmico de varios golpes.

{{< concept-alert type="clave" >}}
Ese es el contenido operativo de la tesis "profundidad, no features": **la profundidad no compra campo receptivo —el pooling global ya lo daba gratis— sino composicionalidad dentro del campo receptivo.** Y explica el contraste con Piczak: una CNN de dos capas sobre log-mel tiene, sobre el espectrograma, un campo receptivo de decenas de tramas, es decir **cientos de milisegundos de audio**, porque cada píxel de entrada ya resume 25 ms. La misma CNN de dos capas sobre onda cruda tiene 21 ms. Los features hechos a mano regalan varias capas de profundidad efectiva.
{{< /concept-alert >}}

### El costo: M3 es la red más lenta y la peor

La columna de tiempo por época del paper es más rara de lo que parece, y nadie la comenta:

| Red | Capas conv | Parámetros | MACs de las convs | Tiempo/época (Titan X) | Accuracy |
|---|---|---|---|---|---|
| **M3** | 2 | 0.22M | **555M** | **77 s** | **56.12%** |
| M5 | 4 | 0.56M | 276M | 63 s | 63.42% |
| M11 | 10 | 1.79M | **215M** | 71 s | 69.07% |
| M18 | 17 | 3.68M | 365M | 98 s | 71.68% |
| M34-res | 33 | 3.98M | — | 124 s | 63.47% |

**M3 es simultáneamente la red más cara de la familia y la peor de la familia.** Es más lenta que M5 (63 s) y que M11 (71 s), pese a tener la décima parte de las capas. La razón está en dónde vive su capacidad: M3 pone **256 canales** en la primera capa y una convolución 256→256 sobre una secuencia de 1993 muestras. Esa única capa concentra el **89.2%** de los pesos de la red y consume 392M MACs — más que toda la red M5 junta. El total de M3 son 555M MACs, el doble que M5 y 2.6× M11.

Las redes profundas usan **menos canales abajo** (64 en M11 y M18) y por eso son más baratas donde la secuencia es larga. En M18, en cambio, el 74.9% de los parámetros están en la etapa 4, que opera sobre una secuencia de apenas 28 posiciones.

{{< concept-alert type="recordar" >}}
La lección de ingeniería, que el paper enuncia solo de pasada (*"by using an aggressive down-sampling in the initial layers, very deep networks can be economical to train"*), se puede afinar: **en 1D sobre secuencias largas el costo lo domina el producto (canales × longitud) en las primeras capas, no la profundidad.** M11 tiene cinco veces las capas de M3 y cuesta 2.6× menos cómputo. Que aun así M11 tarde 71 s contra 77 s —una mejora de solo 8% pese a 2.6× menos MACs— refleja que la profundidad cuesta **latencia** por serialización de kernels de GPU, aunque no cueste FLOPs.
{{< /concept-alert >}}

## El protocolo de 10 folds y la trampa del data leakage

Esta es, con diferencia, la lección más transferible del paper — y la que el laboratorio de la clase pone en el centro.

**Qué hace el paper:** *"We use the official fold 10 to be our test set, and the rest for training and validation."* Dos cosas de inmediato. Primero, **respeta las particiones oficiales**: no rebaraja. Segundo, **no hace 10-fold cross-validation**: entrena una sola vez y evalúa sobre un solo fold.

### Por qué el dataset insiste en no rebarajar

[UrbanSound8K](/papers/urbansound8k-salamon-2014) no es una colección de 8 732 grabaciones independientes. Es una colección de **slices** extraídos de un número mucho menor de grabaciones de campo originales de Freesound. De una única grabación de diez minutos de un martillo neumático se recortan varios slices de 4 segundos. Esos slices comparten:

- el mismo equipo de grabación y su respuesta en frecuencia,
- el mismo ruido de fondo — el mismo tráfico, el mismo viento, la misma acústica de la calle,
- la misma fuente física concreta: ese martillo específico, ese perro específico,
- a menudo, secciones de audio **literalmente solapadas** o casi idénticas.

Los diez folds oficiales están construidos precisamente para que **todos los slices de una misma grabación original caigan en el mismo fold**. Esa es la única razón por la que existen como archivos preparticionados en vez de dejar que cada quien haga su propio split.

**Qué pasa si se rebaraja.** Si se junta todo y se hace un split aleatorio 80/20, con altísima probabilidad el slice #3 de una grabación queda en entrenamiento y el slice #4 de la **misma** grabación queda en test. El modelo no necesita aprender qué es un martillo neumático: le basta con memorizar la firma de ese micrófono, ese ruido de fondo y esa fuente. Al evaluar, **reconoce la grabación, no la clase.**

El resultado es un accuracy que sube por encima del **90%** y que no significa nada. El calibre de la anomalía es fácil de establecer: en 2016–2017 el estado del arte publicado sobre UrbanSound8K con el protocolo correcto rondaba el **70–79%**.

{{< concept-alert type="advertencia" >}}
**La regla operativa:** un número que supera al estado del arte por veinte puntos con un modelo trivial no es un resultado, es una alarma de fuga de datos. Y el corolario metodológico, que es lo que hay que llevarse: **la fuga casi nunca se ve en las métricas, se ve en el código de carga de datos.** Un notebook puede dedicar una celda de markdown entera a explicar por qué no hay que correlacionar train y test, y la implementación tres celdas después hacerlo de todos modos — por ejemplo, con un `glob` cuyo patrón `*[2, 3, 4, ...]` no interpola una lista de folds sino una **clase de caracteres**, con lo que termina barriendo los diez directorios.
{{< /concept-alert >}}

Hay una segunda razón, menos dramática pero igual de real: los folds oficiales están **balanceados por clase y por condiciones de grabación**, de modo que los resultados de distintos papers sean comparables. Rebarajar destruye esa comparabilidad aunque uno tenga cuidado con las grabaciones.

### El tercer problema: la varianza de un solo fold

Aun respetando los folds, evaluar sobre **uno solo** tiene un problema distinto. Con ~870 clips en un fold, el error estándar de una accuracy de 0.72 es

$$\sqrt{\frac{0.72 \times 0.28}{870}} \approx 0.015$$

es decir **±1.5 puntos de error estándar y ±3 puntos de intervalo de confianza al 95%**. Esa es exactamente la razón por la que el dataset pide reportar el promedio de los diez folds: no solo para evitar la fuga, sino para tener un número con precisión suficiente. Varias de las comparaciones menores del paper caen dentro de ese margen, como se ve en la sección de ablations.

### El resto del protocolo

| Parámetro | Valor |
|---|---|
| Dataset | UrbanSound8K, 8 732 clips, 10 clases, 9.7 h nominales |
| Sample rate | **8 kHz** (remuestreado desde el original) |
| Normalización | media 0, varianza 1 |
| Entrada | **32 000 muestras = 4.0 s exactos** |
| Data augmentation | **ninguna** |
| Optimizador | Adam (learning rate no reportado) |
| Épocas | 100–400, "hasta convergencia" |
| Inicialización | Glorot, desde cero, sin preentrenamiento |
| Regularización | $\ell_2$ con $10^{-4}$; **sin dropout** |
| Framework / hardware | TensorFlow, una Titan X |

Vale la pena hacer la aritmética de consistencia del dataset: $8732 \times 4\ \text{s} = 34\,928\ \text{s} = 9.70\ \text{h}$. Es decir, **las 9.7 horas son la cota superior asumiendo que todos los clips duren exactamente 4 s**, no la duración real del audio. Consistente, pero optimista.

Consecuencia acústica de los 8 kHz: la frecuencia de Nyquist queda en **4 kHz**, así que todo el contenido espectral por encima se descarta. El paper elige explícitamente el compromiso velocidad-sobre-información, y lo dice.

## Resultados

### La curva de profundidad

| Modelo | Capas conv | Parámetros | **Accuracy test** | Δ vs. anterior | Train accuracy |
|---|---|---|---|---|---|
| M3 | 2 | 0.2M | **56.12%** | — | — |
| M5 | 4 | 0.5M | **63.42%** | **+7.30** | — |
| M11 | 10 | 1.8M | **69.07%** | **+5.65** | — |
| **M18** | 17 | 3.7M | **71.68%** | **+2.61** | 96.72% |
| M34-res | 33 | 4M | **63.47%** | **−8.21** | 99.21% |

Los incrementos son +7.30, +5.65, +2.61: monótonos, pero con **retornos claramente decrecientes**. Duplicar de 2 a 4 capas convolucionales compra 7.3 puntos; pasar de 10 a 17 compra 2.6. La extrapolación natural es que hacia 20–25 capas la curva se aplana, y la caída de M34-res confirma que el régimen cambió: ahí ya no manda la capacidad de representación sino la capacidad de memorización frente a un dataset de ~7 900 ejemplos de entrenamiento.

{{< concept-alert type="advertencia" >}}
**La conclusión del paper dice 71.8%; la Tabla 2 dice 71.68%.** Redondeado correctamente, 71.68 → **71.7**, no 71.8. Es un error tipográfico menor, pero se propaga: hay citas secundarias que reportan "71.8%" como si fuera la cifra oficial.
{{< /concept-alert >}}

### El control de capacidad: la parte fuerte del paper

Un escéptico diría que los deltas se explican por el conteo de parámetros: M18 tiene 18× los de M3. El paper anticipa la objeción y la mata:

| Modelo | Parámetros | Accuracy | Lectura |
|---|---|---|---|
| M3 | 0.2M | 56.12% | referencia |
| **M3-big** (384 filtros en conv1) | **0.5M** | **57.55%** | +1.43 con 2.5× parámetros |
| M5 | 0.5M | **63.42%** | **+5.87 sobre M3-big, con los mismos 0.5M** |
| **M5-big** (256 filtros en conv1) | **2.2M** | **63.30%** | **−0.12** respecto a M5 |
| M11 | 1.8M | **69.07%** | **+5.77 sobre M5-big, con menos parámetros** |

Los dos pares son demoledores. **M3-big contra M5**: mismo presupuesto de parámetros, 57.55% contra 63.42%. **M5-big contra M11**: M5-big tiene *más* parámetros (2.2M vs 1.8M) y saca 5.77 puntos menos. El paper lo resume: *"The performance increases cannot be simply attributed to the larger number of parameters in the deep models"* y *"shallow models have limited capacity to capture time-series inputs even with a larger model"*.

Ese es el resultado que justifica el título. No es "más grande es mejor": es **más profundo es mejor, a igualdad de tamaño**.

### M34-res: qué gana sobre M18

| | M18 | M34-res |
|---|---|---|
| Accuracy de **entrenamiento** | 96.72% | **99.21%** |
| Accuracy de **test** | **71.68%** | 63.47% |
| Brecha train − test | 25.0 pts | **35.7 pts** |
| Tiempo/época | 98 s | 124 s |

El paper es directo: *"M34-res only achieves 63.47% test accuracy. This is due to overfitting. We observe that with residual learning we have no problem optimizing deep networks like M34-res, and M34-res reaches an extremely high training accuracy of 99.21%."* Y añade una observación negativa que raramente se cita: *"We also observe overfitting in a residual variant of M11 network (not shown here) which reaches higher training accuracy but a lower test accuracy (by 0.17%)."* Es decir, **los atajos residuales no ayudaron en ninguna profundidad probada**, ni siquiera a 11 capas.

El matiz que el paper no señala es más importante todavía. El argumento clásico de ResNet es que los atajos resuelven un problema de **optimización**: las redes muy profundas sin atajos alcanzan peor error de *entrenamiento*. Aquí ese problema **nunca se manifestó** — M18, sin atajos, ya llegaba a 96.72% de train. No había degradación que arreglar. Lo único que los atajos compraron fue llevar el entrenamiento de 96.72% a 99.21%, es decir, **más sobreajuste**. La pieza que sí hace entrenable a la red profunda es la batch normalization, y de eso hay evidencia dura.

### La comparación externa

El paper tiene **un solo** punto de comparación con la literatura, y está en la nota al pie 3:

| Sistema | Entrada | Protocolo | Accuracy |
|---|---|---|---|
| **M18** (este paper) | onda cruda 8 kHz | fold 10 como test, una corrida | **71.68%** |
| CNN baseline de Piczak (2015) | log-mel 44.1 kHz | 10-fold CV | ~68% (leído de su Figura 4) |

Todo lo demás que se suele poner en esta tabla —el baseline SVM+MFCC del paper original de UrbanSound8K, los números con augmentation de Salamon y Bello (2017), los resultados con mayoría de votos de Piczak— **no está en este PDF** y no debe atribuirse a Dai et al.

## Ablations

### Tamaño del campo receptivo de la primera capa

Variantes `-srf` (RF 8) y `-lrf` (RF 320), todo lo demás igual:

| Modelo | RF = 8 | **RF = 80** | RF = 320 |
|---|---|---|---|
| M11 | 64.78% (**−4.29**) | **69.07%** | 65.67% (**−3.40**) |
| M18 | 65.55% (**−6.13**) | **71.68%** | 65.08% (**−6.60**) |

Tres observaciones que el paper no hace:

1. **La sensibilidad crece con la profundidad.** M11 pierde 3.4–4.3 puntos; M18 pierde 6.1–6.6. Una red profunda depende **más** de que la primera capa entregue un banco de filtros bien resuelto, porque todo lo que construye encima parte de esa base. Es un argumento a favor de la tesis: la profundidad no compensa un front-end malo, lo amplifica.
2. **Con el RF equivocado, M18 baja al nivel de M11 e incluso de M5.** M18-lrf (65.08%) está más cerca de M5 (63.42%) que de M18 (71.68%). Dicho de otro modo: **el tamaño del kernel de la primera capa vale tanto como quince capas de profundidad.**
3. **La ablación es de dos puntos, no una curva.** Solo se probaron 8, 80 y 320 — factores de 10 en cada dirección. No hay evidencia de que 80 sea óptimo frente a 40 o 160; solo de que es mejor que valores diez veces mayores o menores.

### Batch normalization

| Modelo | Train (sin BN) | Test (sin BN) | Test **con** BN | Δ |
|---|---|---|---|---|
| M11-no-bn | 98.58% | **69.38%** | 69.07% | **+0.31** |
| M18-no-bn | 99.33% | 62.48% | 71.68% | **−9.20** |
| M34-no-bn | **10.96%** | **11.45%** | 63.47% | **−52.02** |

Es el ablation más informativo del paper, y hay que leerlo con cuidado porque el texto solo cuenta la mitad.

**El resultado sólido: sin BN, M34 simplemente no entrena.** 10.96% de accuracy de entrenamiento tras 159 épocas, contra un azar de 10% con diez clases balanceadas. Textual: *"M34-no-bn could not be optimized without BN and performs close to random guess (10%) after 159 epochs of training."* Eso es evidencia dura de que **la batch normalization, no el aprendizaje residual, es lo que hace entrenable a la red profunda** — M34-res tiene atajos residuales *y* BN, y quitarle la BN lo mata pese a los atajos.

**El resultado del medio:** en M18 la BN vale 9.2 puntos de test sin ser necesaria para optimizar. M18-no-bn llega a 99.33% de train pero solo 62.48% de test. El paper lo interpreta como regularización: *"Note that M18-no-bn results in lower test accuracy, indicating that BN has a regularization effect."*

{{< concept-alert type="advertencia" >}}
**La anomalía que el paper no comenta: M11 sin batch norm (69.38%) supera a M11 con batch norm (69.07%).** El texto dice *"Without BN, both M11-no-bn and M18-no-bn can be optimized to high training accuracy"* y salta directo al caso de M18 para el argumento de la regularización. La lectura completa de la tabla es más interesante y más defendible: **el beneficio de la BN aparece con la profundidad** — nulo (o levemente negativo) a 11 capas, grande a 18, existencial a 34. Cautela obligatoria: los 0.31 puntos están muy por debajo del ruido de un solo fold, así que lo correcto es decir "indistinguibles", no "mejor sin BN".
{{< /concept-alert >}}

### Cabezal convolucional contra capas densas

| Modelo | Con GAP | Con FC | Δ | Parámetros GAP → FC |
|---|---|---|---|---|
| M3 | 56.12% | 46.82% | **−9.30** | 0.2M → 129M |
| M5 | 63.42% | 62.76% | −0.66 | 0.5M → 18M |
| M11 | 69.07% | 68.29% | −0.78 | 1.8M → **1.8M** ⚠ |
| M18 | 71.68% | 64.93% | **−6.75** | 3.7M → 8.7M |

**Las cuatro** variantes con capas densas son peores que sus contrapartes con GAP. Pero hay que ser honesto con la lectura: dos de los cuatro deltas (−0.66 y −0.78) están **por debajo del ruido de un único fold de test**. La evidencia real a favor del GAP son M3 y M18, donde los deltas son grandes; en los casos intermedios la afirmación *"fully convolutional networks perform comparably or better"* solo puede sostener el "comparably". El paper además elige mal sus ejemplos en el texto —*"in the cases of M3-fc and M11-fc the additional FC layers lead to lower test accuracy"*— dejando fuera a M18-fc, que pierde 6.75 puntos, casi diez veces más que M11-fc.

{{< concept-alert type="advertencia" >}}
**Errata en la tabla de capas densas: M11-fc figura con 1.8M parámetros, exactamente lo mismo que M11 sin capas densas.** Es imposible: agregar dos capas densas de 1000 unidades sobre un mapa aplanado de $512 \times 32$ suma ~17.4M parámetros, de modo que M11-fc debería ser **~18M**. Las otras filas sí se reconstruyen (M3-fc: $256 \times 500 \times 1000 + \ldots = 129.2$M ✓; M5-fc: $512 \times 32 \times 1000 + \ldots = 17.95$M ✓). Las accuracies de esa tabla no están en duda, solo la columna de parámetros.
{{< /concept-alert >}}

Un detalle relacionado: los conteos de la tabla se hicieron con las **longitudes idealizadas** de la Tabla 1 (2000, 500, 125, 32 tras cada pooling). Con convoluciones sin padding —que es lo que implementa el código de referencia— los valores reales son 1995, 496, 122 y 28 en M18, y el mapa antes del GAP mide **20, no 32**. La divergencia crece con la profundidad porque cada convolución de kernel 3 sin padding pierde 2 muestras. No afecta ninguna conclusión, pero sí a quien intente reproducir las formas.

## Limitaciones

### Reconocidas por el paper

- **El dataset es demasiado chico para M34-res.** *"We believe that our dataset is too small to train M34-res without further regularization."* Con ~7 900 clips, sin augmentation, sin dropout y con $\ell_2$ de solo $10^{-4}$, 33 convoluciones memorizan el conjunto.
- **La comparación con log-mel usa protocolos distintos.** Nota al pie 3, con dos diferencias declaradas.
- **El sobreajuste en redes muy profundas es esperable.** *"Overfitting caused by very deep networks is well documented."*

### No reconocidas

**La escala del dataset condiciona toda la conclusión.** 9.7 horas nominales, un solo dataset, una sola tarea, un solo dominio. La conclusión proyecta hacia reconocimiento de voz (*"hold the promise to improve CNNs for speech recognition and other time-series modeling"*), pero **no hay ni un experimento de voz en el paper**.

**Un único fold de test y una única corrida.** Con el intervalo de confianza de ±3 puntos, varias conclusiones son estadísticamente vacías: M11 vs M11-fc (0.78), M11 vs M11-no-bn (0.31), M5 vs M5-big (0.12), M34-res vs M5 (0.05). Las que sí sobreviven son las de deltas grandes: la curva de profundidad (+7.3, +5.6, +2.6), la caída de M34-res (−8.2), las ablations de RF (−3.4 a −6.6), el colapso sin BN de M34 (−52). Además cada modelo se entrenó **una vez**: no hay barras de error por semilla.

**El sample rate de 8 kHz es una elección severa.** Nyquist en 4 kHz descarta el contenido por encima de esa frecuencia, que es precisamente donde viven los transientes de banda ancha de `gun_shot`, `jack_hammer` y `drilling`. No sabemos si la familia M escala a sample rates realistas.

{{< concept-alert type="advertencia" >}}
**La limitación central: el paper no corre ningún baseline propio de MFCC o log-mel.** No hay una fila "MFCC + la misma red" ni "log-mel + CNN, entrenado por nosotros". La afirmación de paridad se apoya **enteramente** en un número leído de una figura de otro paper: la nota al pie 3 dice que *"Figure 4 in [11] reports ∼68% accuracy using a baseline CNN model"*, y la misma nota reconoce las dos diferencias que invalidan la comparación directa — *"we use the 10-th fold as test set, while [11] performs 10-fold evaluation"* y *"we use sound at 8kHz sampling rate while they use the original 44.1kHz"*. Es decir: 71.68% de Dai (un fold, 8 kHz) contra ~68% de Piczak (diez folds, 44.1 kHz). La conclusión —*"the first report of a parity performance between log-mel features and raw time signal"*— es una afirmación fuerte sostenida por una comparación cruzada entre papers con protocolos distintos. **El paper lo declara honestamente en la nota al pie; el problema es que la nota al pie no llega al abstract.**
{{< /concept-alert >}}

Un experimento de una tarde —correr M18 sobre log-mel del mismo audio a 8 kHz, con el mismo fold de test y el mismo optimizador— habría convertido una afirmación cruzada en una comparación controlada. Es la omisión más costosa del trabajo, y deja abierto el argumento contrario: que la paridad con log-mel se logró **porque** se remuestreó a 8 kHz, donde el espectro es tan pobre que las ventajas de un front-end bien diseñado se reducen.

### Dos cosas que se le atribuyen y no hizo

**No hace 10-fold cross-validation.** Usa solo el fold 10 como test, una sola corrida. Sí respeta las particiones oficiales —que es lo que evita la fuga— pero no promedia sobre las diez. El propio paper señala esto como la diferencia con Piczak.

**No hace ablation de sample rate.** Todo se corre a 8 kHz. Lo que existe es una *regla de escalado* del kernel de la primera capa con el sample rate (*"a field size of 80 at 8kHz sampling rate is at a different length scale than at 16kHz"*), que es una regla de diseño sin medición detrás.

Otras dos precisiones frecuentes: **M34-res no es el modelo propuesto ni el mejor de la familia** — queda cuarto de cinco, apenas 0.05 puntos sobre M5, y existe para mostrar dónde deja de funcionar la profundidad. Y **el paper nunca dice que la onda cruda supere a log-mel**: dice `matches` y `competitive`.

**Hiperparámetros sin reportar.** No hay learning rate, ni batch size, ni schedule, ni criterio de parada más allá de "100–400 épocas hasta convergencia". El paper menciona que el resto de folds se usa para *"training and validation"*, pero nunca describe cómo se usó la validación ni si el modelo reportado se seleccionó por validación o por la mejor época de test. Con ese rango de épocas y sin criterio explícito, la posibilidad de selección de modelo sobre el test set queda abierta.

## Por qué importa hoy

### La línea directa: la primera capa como banco de filtros

La observación de la Figura 2 disparó una línea de trabajo entera que consiste en **imponer** la estructura de banco de filtros en vez de esperarla:

| Trabajo | Idea | Qué toma de Dai |
|---|---|---|
| **TD-filterbanks** (Zeghidour et al., 2018) | Inicializar la primera capa con filtros tipo Gabor equivalentes a mel y dejarla aprender desde ahí | Aceptar que la capa 1 *quiere* ser un banco de filtros |
| **SincNet** (Ravanelli y Bengio, 2018) | Parametrizar cada filtro con dos cortes de frecuencia, reduciendo la capa a $2C$ parámetros | La misma observación, llevada a restricción dura |
| **LEAF** (Zeghidour et al., 2021) | Front-end totalmente aprendible: filtrado Gabor + pooling + compresión, todo diferenciable | Reemplazar el pipeline log-mel completo, no solo el filtrado |

Todos comparten el diagnóstico y difieren en la conclusión: si sabemos que el óptimo es un banco de filtros, ¿por qué gastar 80 pesos libres por filtro?

Hay además una supervivencia práctica que vale la pena registrar: **M5 es la arquitectura del tutorial oficial de PyTorch/torchaudio** de clasificación de comandos de voz, con la misma primera capa de kernel 80 y stride 4, las mismas cuatro convoluciones y el mismo global average pooling. Cientos de miles de personas escribieron su primer clasificador de audio sobre onda cruda con esta arquitectura sin saber de dónde venía. El laboratorio de la clase 39 es un caso más de la misma herencia.

### El veredicto histórico: la apuesta era correcta, por otra razón

Hoy los modelos grandes de audio **sí** usan encoders convolucionales sobre la onda cruda:

| Modelo | Front-end | Configuración |
|---|---|---|
| **wav2vec 2.0** (Baevski et al., 2020) | 7 convoluciones 1D sobre onda cruda a 16 kHz | kernels (10,3,3,3,3,2,2), strides (5,2,2,2,2,2,2) → reducción **320×**, una trama cada 20 ms |
| **HuBERT** (Hsu et al., 2021) | El mismo encoder convolucional | idéntico |
| **EnCodec** (Défossez et al., 2022) | Encoder convolucional con strides (2,4,5,8) | reducción **320×** |
| **SoundStream** (Zeghidour et al., 2021) | Encoder convolucional análogo | ídem |

La onda cruda ganó. El pipeline MFCC/log-mel dejó de ser obligatorio. En ese sentido literal, Dai et al. apostaron bien. **Pero ganó por una razón que este paper no anticipó**, y conviene ser preciso sobre cuál.

**Primero: los encoders que ganaron son superficiales, no profundos.** El encoder de wav2vec 2.0 tiene **siete** capas convolucionales — está entre M5 y M11, muy lejos de M18 o M34. La profundidad de wav2vec 2.0 no está en el encoder de forma de onda: está en el **Transformer de 12 o 24 capas** que viene después y que opera sobre tramas de 20 ms, no sobre muestras. La arquitectura ganadora no es "una CNN muy profunda sobre la onda"; es "una CNN poco profunda que convierte la onda en tramas, más un modelo profundo sobre tramas". Estructuralmente eso está más cerca del CLDNN de Sainath —que Dai et al. citaban como el estado del arte a superar— que de M18.

**Segundo: lo que compró la victoria fue el preentrenamiento autosupervisado a escala.** Lo que hizo que la onda cruda superara a log-mel no fue apilar capas supervisadas sobre 9.7 horas de audio etiquetado; fue **entrenar sobre 53 600 horas de audio no etiquetado** con un objetivo contrastivo (wav2vec 2.0) o de predicción de unidades enmascaradas (HuBERT). El régimen que Dai et al. identificaron como la limitación de M34-res —*"our dataset is too small"*— resultó ser **el** cuello de botella, y la solución no fue arquitectónica sino de datos y de objetivo.

Hay una ironía útil aquí. El paper concluye que la profundidad es la variable clave y que el sobreajuste de M34-res es un problema de dataset chico. Ambas cosas son ciertas **dentro del régimen supervisado de 2016**. Cuando se levantó la restricción de datos, la variable clave dejó de ser la profundidad del encoder convolucional: ese encoder se estabilizó en siete capas y toda la profundidad migró al modelo de secuencia.

{{< concept-alert type="clave" >}}
**La contribución de Dai que sobrevivió no es "34 capas": es el submuestreo agresivo temprano con un kernel dimensionado en milisegundos.** El 16× de Dai se volvió el 320× de wav2vec 2.0 y EnCodec, y la lógica es exactamente la misma — convertir muestras en tramas lo antes posible, barato, y hacer el trabajo pesado sobre las tramas.
{{< /concept-alert >}}

**Tercero, el matiz que suele omitirse: en el dominio propio del paper, los features hechos a mano no fueron desplazados.** En clasificación de sonido ambiental y audio tagging, el estado del arte pasó por **espectrogramas log-mel**, no por onda cruda: las CNN de AudioSet ([VGGish](/papers/vggish-hershey-2017)), PANNs, y luego el **Audio Spectrogram Transformer** (Gong et al., 2021), que aplica un ViT a un espectrograma log-mel tratado como imagen. La onda cruda ganó en **voz** (wav2vec 2.0, HuBERT) y en **códecs y generación** ([WaveNet](/papers/wavenet-oord-2016), EnCodec, SoundStream), donde se necesita resolución a nivel de muestra o donde hay decenas de miles de horas de audio no etiquetado. En etiquetado de sonidos generales —el problema que Dai et al. atacaban— log-mel sigue siendo la entrada por defecto. Todo esto está mapeado en el [dominio de audio](/dominios/audio).

El veredicto completo, entonces: **la apuesta de la onda cruda fue correcta, pero el mecanismo que la validó —preentrenamiento autosupervisado masivo— y el dominio donde se validó —voz y códecs— no son los que el paper propuso ni el que estudió.**

## En la clase 39

Este es el paper del laboratorio de la [Clase 39](/clases/clase-39), y la clase lo pone en tensión con una alternativa. El slide "Can We Use Raw Audio Data" plantea el problema exactamente igual que Dai —*"Using a convolutional architecture, we need huge filters or a very deep structure, why?"*— pero ofrece una tercera salida: *"We can increase the receptive field of neurons in intermediate layers using dilated convolution filters."*

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
| Cómo crece el RF | **dilatación + recurrencia** | **profundidad + stride/pooling** |

Son dos soluciones al mismo problema, y la comparación se hace con aritmética. Con la recurrencia general $r_l = r_{l-1} + (k_l - 1)\,d_l\,j_{l-1}$ y $j_l = j_{l-1}\,s_l$, donde $d$ es la [dilatación](/fundamentos/convoluciones-dilatadas):

**(a) Kernels enormes.** $r = k$: campo receptivo lineal en el kernel, parámetros lineales en el kernel, costo $O(N k C_{in} C_{out})$ por capa. Para cubrir 1 segundo a 8 kHz haría falta $k = 8000$, es decir 8 000 pesos por filtro. Es lo que el paper hace **solo en la primera capa** ($k=80$) y con una justificación física precisa, no como estrategia general.

**(b) Profundidad con stride/pooling.** Cada convolución de kernel 3 aporta $2j$ al campo receptivo, donde $j$ es el salto acumulado. Con un pooling de stride 4 entre etapas, $j$ sigue la progresión $4 \to 16 \to 64 \to 256 \to 1024$. El campo receptivo crece **geométricamente con la profundidad**, y el costo total es una **serie geométrica convergente**:

$$\text{costo} \;\propto\; N\left(1 + \tfrac{1}{4} + \tfrac{1}{16} + \tfrac{1}{64} + \cdots\right) < \tfrac{4}{3}N$$

Es decir: **cinco etapas de convoluciones cuestan menos del doble que una sola etapa.** Ese es el truco que hace barata la profundidad. En M18 el resultado es un campo receptivo de **11 980 muestras (1.50 s) con 17 convoluciones**.

**(c) Dilatación.** Cada capa aporta $(k-1)\,d$, con $d$ típicamente duplicándose. Sin stride, $j$ se queda en 1 y **la longitud de la secuencia nunca se reduce**: el costo es $O(N)$ **por cada capa**, es decir $O(NL)$ en total. [WaveNet](/papers/wavenet-oord-2016) lo usa así — *"each 1, 2, 4, ..., 512 block has receptive field of size 1024"*—, y apilando tres bloques (30 capas) llega a ~3 000 muestras, que el propio paper cifra en *"about 300 milliseconds"* de contexto.

Aplicando la fórmula al "Ejemplo 2" del slide, con kernels 20, 10, 10, 5:

$$r = 1 + 19 d_1 + 9 d_2 + 9 d_3 + 4 d_4$$

| Dilataciones | Campo receptivo | Equivalente a 16 kHz |
|---|---|---|
| 1, 1, 1, 1 (sin dilatar) | 42 muestras | 2.6 ms |
| 1, 2, 4, 8 | 106 muestras | 6.6 ms |
| 1, 10, 100, 1000 | 5 010 muestras | 313 ms |

De ahí la advertencia del slide de que *"dilation factor depends of application"*: con cuatro capas, la dilatación es el **único** grado de libertad que separa 2.6 ms de 313 ms. Y de ahí también las dos LSTM del Ejemplo 2 — con cuatro capas convolucionales, aun dilatadas, no se llega al contexto de segundos que requiere clasificar un clip completo, y la recurrencia hace ese trabajo.

La comparación en una línea: *ambas estrategias hacen crecer el campo receptivo exponencialmente con la profundidad; la diferencia está en el costo y en lo que se conserva.*

| | Profundidad + stride/pooling (familia M) | Dilatación (WaveNet, Ejemplo 2) |
|---|---|---|
| Crecimiento del RF | Geométrico | Geométrico |
| Longitud de la secuencia interna | Se reduce ×4 por etapa | **Se conserva** |
| Costo total de $L$ capas | $O(N)$ — serie geométrica | $O(NL)$ |
| Resolución temporal de la salida | **Se destruye** (una trama cada 2 ms, luego cada 128 ms) | **Se preserva** (una salida por muestra de entrada) |
| Adecuado para | Clasificación con pooling global | **Generación muestra a muestra**, segmentación fina, detección de eventos |
| Contexto global | Global average pooling | Requiere apilar bloques o agregar RNN/atención |

{{< concept-alert type="clave" >}}
**El criterio de elección, que es la idea que hay que llevarse:** WaveNet usa dilatación porque tiene que emitir una muestra por cada muestra de entrada — **no puede permitirse destruir resolución temporal**. Dai usa stride y pooling porque solo tiene que emitir una etiqueta cada 4 segundos: la resolución temporal es exactamente lo que le sobra, y **destruirla es lo que le permite ser profundo y barato al mismo tiempo**. No es que una técnica sea mejor que la otra: son óptimas para tareas distintas.

**Regla operativa: si la salida es densa en el tiempo, dilata; si la salida es una etiqueta, submuestrea.**
{{< /concept-alert >}}

Para la [profundización de la clase 39](/clases/clase-39/profundizacion), la aritmética de campos receptivos de la familia M es el ejemplo trabajado de la regla, y el contraste M3-contra-M18 es el experimento que la valida: las clases que se distinguen por un transiente corto y espectralmente distintivo (`gun_shot`, `dog_bark`, `car_horn`) deberían ser razonables incluso para M3, porque 21 ms alcanzan para caracterizar un impulso. Las clases de **ruido estacionario de banda ancha** —`air_conditioner`, `engine_idling`, `drilling`, `jack_hammer`— se separan por su estructura temporal de largo alcance, y son exactamente donde M3 debería colapsar en un bloque de confusión mutua y donde M18 debería ganar la mayor parte de sus 15 puntos. Si la matriz de confusión muestra ese patrón, es la evidencia empírica directa de la tesis del paper.

## Notas y enlaces

**Sobre la versión.** El preprint es arXiv:1610.00087v1, con marca de margen "1 Oct 2016". El PDF no lleva encabezado ni pie de página de ICASSP: la referencia habitual "ICASSP 2017" corresponde a la versión publicada, que no es ese archivo. Todo lo citado aquí sale del preprint. El asterisco en los dos primeros autores indica contribución equitativa; las afiliaciones se deducen de los correos (`cs.cmu.edu`, `stanford.edu`, `us.bosch.com`). El trabajo se financió con el contrato FA8702-15-D-0002 del Software Engineering Institute.

**Cómo reproducirlo bien.** Tres decisiones de implementación que no son obvias y que cambian los resultados:

- Usar `AdaptiveAvgPool1d(1)` en vez de un `AvgPool1d(L)` con la longitud hardcodeada. La versión fija rompe la propiedad de longitud variable que el paper declara explícitamente y obliga a recalcular el número a mano cada vez que cambia el sample rate o la duración.
- Remuestrear con un filtro antialias real (`torchaudio.transforms.Resample`), no con decimación por slicing tipo `[::5]`. Quedarse con una de cada $k$ muestras sin pasabajos previo **repliega** todo el contenido por encima de la nueva Nyquist sobre las frecuencias bajas. Además, los archivos de UrbanSound8K vienen a sample rates heterogéneos (44.1, 48, 24, 22.05, 16, 11.025 y 8 kHz), así que un factor fijo hace que la red vea el mismo evento acústico a escalas de tiempo distintas según el archivo — lo que rompe la premisa central del paper de que el kernel de 80 corresponde a 10 ms.
- Devolver logits, no probabilidades: `nn.CrossEntropyLoss` aplica el log-softmax internamente, y el `Softmax` de la Tabla 1 es conceptual.

**Al cambiar de sample rate**, la regla es que el kernel de la primera capa se dimensiona en milisegundos. Escalando también el stride se preserva todo lo demás:

| Sample rate | Kernel de conv1 (10 ms) | Stride | Reducción conv1+pool | Nyquist | Longitud tras pool1 |
|---|---|---|---|---|---|
| **8 kHz** (paper) | **80** | **4** | 16× | 4.0 kHz | 1 995 |
| 16 kHz | **160** | 8 | 32× | 8.0 kHz | 1 995 |
| 16 kHz (alternativa) | **160** | 4 | 16× | 8.0 kHz | 3 991 |
| 22.05 kHz | **220** | 11 | 44× | 11.0 kHz | 1 999 |

Escalar el stride junto con el kernel deja el resto de la red **idéntico** —mismas longitudes intermedias, mismo costo, mismos campos receptivos medidos en segundos— y es la opción por defecto. Mantener el stride en 4 conserva más resolución temporal a costa de duplicar el cómputo, y solo se justifica si la tarea la necesita (detección de eventos, alineamiento). Una consecuencia menos obvia: al subir el sample rate, el kernel de 10 ms cubre **más espectro, no más tiempo**, así que conviene aumentar también el número de filtros de conv1 o aceptar una cobertura espectral más rala. El paper no discute esto porque nunca cambia el sample rate.

**Enlaces relacionados**

- Clase donde se usa: [Clase 39](/clases/clase-39) y su [profundización](/clases/clase-39/profundizacion)
- La alternativa arquitectónica: [WaveNet (van den Oord et al., 2016)](/papers/wavenet-oord-2016) y el fundamento de [convoluciones dilatadas](/fundamentos/convoluciones-dilatadas)
- El dataset: [UrbanSound8K (Salamon, Jacoby y Bello, 2014)](/papers/urbansound8k-salamon-2014)
- La rama log-mel que no fue desplazada: [VGGish / CNN para AudioSet (Hershey et al., 2017)](/papers/vggish-hershey-2017)
- Fundamentos: [redes convolucionales](/fundamentos/redes-convolucionales), [representación de audio](/fundamentos/representacion-de-audio)
- Panorama: [dominio de audio](/dominios/audio)
