# Deep Learning for Audio Signal Processing (Purwins et al., 2019) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores y afiliaciones** (según la primera página del PDF, nota al pie de la columna izquierda):
  - **Hendrik Purwins** — Department of Architecture, Design & Media Technology, **Aalborg University Copenhagen**. *Nota:* el PDF **no** menciona Sonos; el correo de contacto que da es personal (`hpurwins@gmail.com`). Purwins trabajó en Sonos, pero eso no está en el paper y no conviene citarlo como si lo estuviera.
  - **Bo Li**, **Shuo-yiin Chang** y **Tara Sainath** — **Google**.
  - **Tuomas Virtanen** — **Tampere University** (Finlandia).
  - **Jan Schlüter** — **CNRS LIS, Université de Toulon** y **Austrian Research Institute for Artificial Intelligence (OFAI)**.
  - Marca explícita de contribución igualitaria: "H. Purwins, B. Li, T. Virtanen, and J. Schlüter contributed equally to this paper" (asterisco en la lista de autores). Sainath y Chang no llevan asterisco.
- **Venue exacto:** *IEEE Journal of Selected Topics in Signal Processing (JSTSP)*, **Vol. 13, No. 2, mayo 2019, pp. 206–219**. DOI `10.1109/JSTSP.2019.2908700`. La copia que estamos leyendo es el preprint **arXiv:1905.00078v2, 25 de mayo de 2019** (15 páginas de PDF, 14 de artículo).
- **Fecha de recepción del manuscrito: 11 de octubre de 2018.** Este dato es el más importante de toda la metadata para leer el paper con criterio: **el corte bibliográfico real es octubre de 2018**, no mayo de 2019. Todo lo que aparece "raro" en el survey (que no mencione SpecAugment, que no discuta Transformers, que trate la separación de fuentes como un problema de máscaras tiempo-frecuencia) se explica por esa fecha.
- **Estructura material:** 153 referencias, **2 figuras**, y —dato relevante— **ninguna tabla**. Este survey no tiene tablas comparativas de resultados ni de datasets. No hay una sola cifra de WER, mAP o accuracy en todo el texto. Es un survey **conceptual y bibliográfico**, no un meta-análisis cuantitativo. Cualquier afirmación numérica que alguien le atribuya hay que revisarla dos veces (ver §13).
- **Agradecimiento:** "Great thanks to Duncan Blythe for proof-reading."

### Qué hace y qué lo distingue

El propósito declarado en el abstract es tratar **speech, music y environmental sound processing "side-by-side"**, precisamente para exponer similitudes, diferencias y "potencial de fertilización cruzada entre áreas". Esa es la tesis organizadora y es lo que lo separa de los surveys de la época: en 2018-2019 ya existían revisiones sólidas de *deep learning para ASR*, de *deep learning para MIR* (los autores citan el compendio de Bayle, [99]) y de *sound event detection*, pero cada una vivía en su propia comunidad, con su propio vocabulario para el mismo problema. Purwins et al. se toman el trabajo de construir un vocabulario común: un mismo eje de categorización de tareas (§II-A), un mismo inventario de representaciones (§II-B), un mismo inventario de modelos (§II-C), y recién después bajan a los tres dominios.

El segundo rasgo distintivo es la justificación de por qué el audio merece un survey propio y no es un corolario de visión. Está en la introducción y vale citarla porque es el argumento que el profesor de la clase 39 reproduce cuando muestra el espectrograma como imagen:

> "Raw audio samples form a one-dimensional time series signal, which is fundamentally different from two-dimensional images. Audio signals are commonly transformed into two-dimensional time-frequency representations for processing, **but the two axes, time and frequency, are not homogeneous as horizontal and vertical axes in an image**. Images are instantaneous snapshots of a target and often analyzed as a whole or in patches with little order constraints; however audio signals have to be studied sequentially in chronological order."

Tres afirmaciones empaquetadas ahí: (a) el audio nativo es 1D; (b) el espectrograma **parece** una imagen pero sus ejes **no son intercambiables** —desplazarse en frecuencia significa algo físicamente distinto que desplazarse en tiempo, de modo que la invarianza traslacional del kernel 2D es legítima en un eje y discutible en el otro—; (c) el audio tiene un orden causal obligatorio. De esas tres se derivan casi todas las decisiones de diseño que el survey cataloga después.

El tercer rasgo: el survey es **honestamente inconcluso**. La sección IV-B termina diciendo que no hay preferencia clara entre CNN, RNN y CRNN, y que la razón probable es sociológica: "This may be due to each research group's specialized informal knowledge about how to effectively design and tune a particular architecture type." Es una de las frases más honestas que he leído en un survey de la época.

**Cifras ancla del paper** (las únicas cuantitativas que ofrece, todas en §II-D):

| Dato | Valor según el survey | Ubicación |
|---|---|---|
| ImageNet | 14 millones de imágenes etiquetadas a mano (cifra "2019") | §II-D |
| AudioSet | más de 2 millones de *audio snippets* | §II-D |
| Cuantización a 8 bits del sample | convierte la predicción del siguiente sample en clasificación de **256 clases** | §II-A |
| Referencias | 153 | — |

---

## 2. La estructura del survey

Vale la pena tener el mapa a mano, porque el paper está organizado como una matriz: primero las columnas (métodos transversales), después las filas (dominios).

| Sección | Contenido | Utilidad para la clase 39 |
|---|---|---|
| **I. Introduction** | Tres olas de redes neuronales (perceptrón, backprop, deep learning 2012); por qué el audio es distinto de la imagen | Alta: justifica el slide "Audio vs Image Data" |
| **II-A. Problem Categorization** | Los dos ejes de la Fig. 1: número de etiquetas × tipo de etiqueta. Define *sequence classification*, *multi-label sequence classification*, *sequence regression*, *sequence labeling*, *event detection*, *sequence transduction* | **Máxima**: es la fuente del slide de clasificación single/multiple × global/local |
| **II-B. Audio Features** | MFCC y por qué caen en desuso; log-mel; constant-Q; ventana y resolución; filtros aprendidos; raw waveform | **Máxima** |
| **II-C. Models** | (a) CNN y convolución dilatada; (b) RNN, LSTM, F-LSTM, TF-LSTM, CRNN; (c) seq2seq, CTC, RNN-T, atención, LAS; (d) GAN; (e) funciones de pérdida; (f) modelado de fase | **Máxima** |
| **II-D. Data** | Datasets por dominio; transfer learning; generación y augmentación de datos | Alta: fuente del slide de data augmentation |
| **II-E. Evaluation** | WER, accuracy, AUROC, EER/F-score, SDR/SIR/SAR, MOS, test de Turing | Media |
| **III-A. Analysis** | 1) Speech; 2) Music; 3) Environmental Sounds; 4) Localization and Tracking | Alta |
| **III-B. Synthesis and Transformation** | 1) Source Separation; 2) Audio Enhancement; 3) Generative Models | Media |
| **IV. Discussion and Conclusion** | A) Features; B) Models; C) Data Requirements; D) Computational Complexity; E) Interpretability and Adaptability | **Máxima**: es donde el survey dice lo que realmente piensa |

Las dos figuras:

- **Fig. 1** — la taxonomía de tareas en dos ejes. Media página, ninguna fórmula, y sin embargo es la contribución conceptual más reutilizable del paper.
- **Fig. 2** — cinco formas de procesar contexto temporal: **A** convolución 1D, **B** convolución 1D dilatada, **C** capa recurrente, **D** capa recurrente bidireccional, **E** atención. La leyenda de la figura es sorprendentemente detallada (describe el cálculo del contexto $c_t$ como suma ponderada de los *embeddings* del encoder). Es la figura que ordena todo el zoológico de arquitecturas del survey en un solo eje: *¿cómo hace tu modelo para mirar hacia atrás y hacia adelante en el tiempo?*

---

## 3. Representaciones de audio (§II-B, con cierre en §IV-A)

Esta es la sección más valiosa del survey para la clase, y también donde el survey es notoriamente más preciso que cualquier slide.

### 3.1. El punto de partida conceptual

El survey abre §II-B con la tesis que hace que todo lo demás tenga sentido:

> "Building an appropriate feature representation and designing an appropriate classifier for these features have often been treated as separate problems in audio processing. One drawback of this approach is that the designed features might not be optimal for the classification objective at hand. Deep neural networks can be thought of as performing feature extraction jointly with objective optimization."

Y da la evidencia concreta: Mohamed et al. [10] mostraron que **las activaciones de las capas bajas de una DNN funcionan como features adaptadas al hablante**, mientras que las capas altas hacen discriminación por clase. Es decir: la red reimplementa por su cuenta la etapa que el pipeline clásico hacía a mano (normalización de hablante, tipo VTLN/fMLLR), y lo hace optimizada para la tarea final.

Esto define un continuo, y el survey lo recorre entero, de más "hecho a mano" a menos:

$$\text{MFCC} \;\to\; \text{log-mel} \;\to\; \text{espectro de magnitud completo} \;\to\; \text{filtros aprendidos sobre la onda} \;\to\; \text{onda cruda sin filtro alguno}$$

### 3.2. La onda cruda

Formalmente, la entrada es siempre "a time series of audio samples" (§II-A). El survey aclara en una nota al pie algo que suena obvio pero es metodológicamente importante: "While the audio signal will often be processed into a sequence of features, **we consider this part of the solution, not of the task**." O sea: elegir log-mel ya es una decisión de modelado, no un pre-requisito neutral.

Los dos problemas de la onda cruda que el survey identifica:

1. **Tamaño del campo receptivo** (§II-C-a). "Especially for raw waveform inputs with a high sample rate, reaching a sufficient receptive field size may result in a large number of parameters of the CNN and high computational complexity." A 16 kHz, un contexto de 1 segundo son 16.000 muestras; una CNN con kernels de tamaño razonable necesita o filtros enormes o muchísimas capas. De ahí la dilatación (§4.2).
2. **Invarianza de fase** (§II-C-f). Esta es la observación específica de audio que casi nadie explica bien, y el survey la deja clarísima:

> "When using raw waveform as input representation, for an analysis task, one of the difficulties is that **perceptually and semantically identical sounds may appear at distinct phase shifts**, so using a representation that is invariant to small phase shifts is critical."

Traducción operativa: dos grabaciones del mismo fonema, del mismo golpe de tambor, del mismo motor, son *el mismo objeto perceptual* pero **vectores completamente distintos** en el dominio del tiempo, porque el instante en que se abrió la ventana desplaza la fase de cada componente. El espectrograma de magnitud resuelve esto de un plumazo, porque descarta la fase. Una red sobre onda cruda tiene que **aprender** la invarianza. Las dos soluciones que el survey cataloga:

- **Capas convolucionales que hacen *pooling* en el tiempo** ([20], [21], [23]) — el max-pooling temporal sobre la salida de un filtro es literalmente un detector de energía invariante al desplazamiento dentro de la ventana de pooling; es la reconstrucción aprendida de $|X(f)|$.
- **Capas densas con unidades ocultas grandes, potencialmente sobrecompletas** ([22]), "which are able to capture the same filter shape at a variety of phases" — o sea, replicar el mismo filtro desplazado en fase muchas veces y dejar que la no-linealidad se quede con el que dispare.

Y el corolario: "Raw audio as input representation is often used in synthesis tasks, e.g. when autoregressive models are used [25]" (WaveNet). En síntesis la fase no es un estorbo, es el producto.

### 3.3. Espectrograma de magnitud vía STFT

El survey lo define implícitamente y le dedica su discusión más técnica en §II-B y en §III-B1. Tres puntos que no aparecen en ningún slide:

**(a) El compromiso de la ventana.** "The window size for computing spectra trades temporal resolution (short windows) against frequential resolution (long windows)." Es el principio de incertidumbre de Gabor. Y la observación fina: *sí* se puede usar ventanas más cortas en frecuencias altas (tanto para log-mel como para constant-Q), "but this results in **inhomogeneously blurred spectrograms unsuitable for spatially local models**". Es decir: la solución que suena obvia rompe el supuesto que hace válido usar una CNN 2D. La alternativa que el survey recomienda: calcular espectros con **distintos largos de ventana**, proyectarlos a las **mismas bandas de frecuencia**, y tratarlos como **canales separados** ([16], Schlüter & Böck). Multi-resolución como canales, no como resolución variable.

**(b) Los ejes no son homogéneos, y las bandas no son comparables.** "In contrast to images, value distributions differ significantly between frequency bands. To counter this, **spectrograms can be standardized separately per band**." Este es el consejo práctico más accionable de toda la sección y el que más se olvida: no normalices el espectrograma como si fuera una imagen (media y desviación globales), normalízalo **por banda**. La energía de la banda de 100 Hz y la de 7 kHz difieren en órdenes de magnitud en casi cualquier señal natural.

**(c) Los armónicos rompen la localidad espacial.** "Due to the physics of sound production, there are additional correlations for frequencies that are multiples of the same base frequency (harmonics). To allow a spatially local model (e.g., a CNN) to take these into account, **a third dimension can be added that directly yields the magnitudes of the harmonic series** [14], [15]." Esto es sutil y merece desarrollo: una nota de piano en $f_0 = 220$ Hz enciende bins en 220, 440, 660, 880… Un kernel 2D de $3\times3$ ve, a lo más, tres bins contiguos: **jamás** puede relacionar el fundamental con su tercer armónico. La solución que citan (Lostanlen & Cella, ISMIR 2016; Bittner et al., ISMIR 2017) es apilar como **canales** las versiones del espectrograma desplazadas a $f_0, 2f_0, 3f_0, \dots$ para que el armónico $k$ caiga en el mismo píxel del canal $k$, y ahí sí un kernel local pueda integrarlos. Es el equivalente audio del *dilated* pero en el eje de frecuencia y con espaciado multiplicativo.

### 3.4. Escala mel y log-mel

El survey da la definición operacional en una sola frase densa (§II-B): los MFCC "are magnitude spectra **projected to a reduced set of frequency bands**, converted to **logarithmic magnitudes**, and approximately **whitened and compressed with a discrete cosine transform (DCT)**". Cuatro pasos. El **log-mel** es exactamente eso menos el cuarto.

La justificación del banco mel: "inspired by the human auditory system and physiological findings on speech perception [12]" (Davis & Mermelstein, 1980). El survey no da la fórmula; para el registro, la conversión estándar es

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

que es aproximadamente lineal bajo ~1 kHz y logarítmica arriba. El efecto neto es **reducir dimensionalidad concentrando resolución donde el oído la tiene**: un espectro STFT de 512 bins se comprime a 40–80 bandas.

El logaritmo cumple dos funciones que el survey no separa explícitamente pero conviene tener claras: (i) comprime el rango dinámico enorme del audio a algo que una red pueda digerir con activaciones acotadas; (ii) convierte la **multiplicación** por una función de transferencia de sala/micrófono en una **suma** constante por banda, que la normalización por banda del punto (b) anterior puede después restar. Es decir: el log es lo que hace que la normalización cepstral/por banda sea capaz de cancelar el canal.

### 3.5. MFCC y por qué el survey los declara en desuso

Este es el punto que la consigna pide desarrollar, y es el que más veces se explica mal. La frase exacta del survey (§II-B):

> "With deep learning models, the latter [la DCT] has been shown to be **unnecessary or unwanted, since it removes information and destroys spatial relations**. Omitting it yields the log-mel spectrum, a popular feature across audio domains."

Y el cierre en §IV-A: "Whereas MFCCs are the most common representation in traditional audio signal processing, **log-mel spectrograms are the dominant feature in deep learning**, followed by raw waveforms or complex spectrograms."

**El argumento histórico completo, que el survey comprime en dos líneas:**

El pipeline clásico de ASR era GMM-HMM. Cada estado de trifono tenía asociada una mezcla de gaussianas sobre el vector de features. Y por costo computacional y por cantidad de datos, esas gaussianas se entrenaban con **matriz de covarianza diagonal**: estimar la covarianza plena de un vector de 40 dimensiones son $40\times41/2 = 820$ parámetros por componente, por cada uno de miles de estados. Inviable.

Pero una gaussiana diagonal asume **independencia entre dimensiones**. Y las bandas log-mel están fuertísimamente correlacionadas entre sí: la energía en la banda 12 predice muy bien la de la banda 13. Alimentar un GMM diagonal con log-mel es violar su supuesto central de la peor manera posible.

Ahí entra la DCT. La DCT sobre las log-energías de banda es una aproximación fija (independiente de los datos) a la transformación de Karhunen-Loève del espectro log-mel: **decorrelaciona aproximadamente** las dimensiones. Además concentra la energía en los primeros coeficientes, lo que permite quedarse con 12–13 y descartar el resto. Los MFCC no son "mejores features": son **log-mel torcidos para que quepan en el supuesto de covarianza diagonal de un GMM**. Son un parche de modelado que se disfrazó de feature perceptual durante treinta años.

Cuando el clasificador pasa a ser una red neuronal, **el parche deja de tener sentido y pasa a hacer daño**, por las dos razones que el survey nombra:

1. **"It removes information."** Truncar a 13 coeficientes bota deliberadamente la estructura fina del espectro —justamente la estructura armónica, que vive en los coeficientes cepstrales altos (la "quefrencia" del pitch). Una DNN sí puede aprovecharla; un GMM diagonal no. La DCT truncada es un cuello de botella diseñado para un modelo que ya no usamos.
2. **"It destroys spatial relations."** Y esta es la razón decisiva para la clase 39. Una CNN funciona porque asume que **vecindad en el índice implica vecindad semántica**: los bins log-mel $k$ y $k+1$ son frecuencias adyacentes, y un kernel local sobre ellos captura una estructura espectral real (un formante, un ataque). Los coeficientes cepstrales $c_k$ y $c_{k+1}$ **no son adyacentes en ningún sentido físico**: son proyecciones sobre dos cosenos de frecuencias distintas, cada uno con soporte sobre *todo* el espectro. Convolucionar sobre el eje cepstral es aplicar un kernel local a un vector cuya estructura es global. **La DCT destruye exactamente la propiedad que hace que valga la pena usar una CNN.** Por eso el slide "espectrograma como imagen" solo es coherente con log-mel, nunca con MFCC.

Corolario práctico para quien viene de librosa: `librosa.feature.melspectrogram` seguido de `librosa.power_to_db` es el camino; `librosa.feature.mfcc` es el camino heredado. Si aparece un pipeline moderno con MFCC, casi siempre es inercia de código, no una decisión.

### 3.6. Constant-Q y por qué la música es distinta

El survey le dedica un párrafo preciso (§II-B):

> "For some tasks, it is preferable to use a representation which captures **transpositions as translations**. Transposing a tone consists of scaling the base frequency and overtones by a common factor, which becomes a **shift in a logarithmic frequency scale**. The constant-Q spectrum achieves such a frequency scale with a suitable filter bank [13]."

El razonamiento en detalle: transponer una nota un semitono multiplica $f_0$ y **todos** sus armónicos por $2^{1/12}$. En un eje de frecuencia lineal (STFT) o casi-lineal en la parte baja (mel), esa multiplicación es un **estiramiento no uniforme**: el fundamental se mueve 13 Hz y el quinto armónico se mueve 65 Hz. El patrón armónico **cambia de forma**. En un eje logarítmico, en cambio,

$$\log(2^{1/12} \cdot k f_0) = \log(k f_0) + \tfrac{1}{12}\log 2$$

todos los componentes se desplazan **la misma cantidad**: el patrón se **traslada rígidamente**. Y una CNN es equivariante a traslaciones por construcción. Es decir: **sobre constant-Q, un solo kernel aprendido reconoce un acorde mayor en las doce tonalidades**; sobre STFT o mel habría que aprender doce plantillas distintas. Eso es un factor 12 de eficiencia estadística en un dominio (música) donde los datasets etiquetados son chicos.

El *Q* del nombre es el factor de calidad $Q = f_k / \Delta f_k$, constante por construcción: cada filtro tiene ancho de banda proporcional a su frecuencia central, o sea ancho constante **en semitonos**. La consecuencia práctica: resolución temporal peor en graves (ventanas largas) y mejor en agudos, que es exactamente lo que la música pide.

El survey confirma su uso empírico: Lacoste & Eck [84] entrenaron un MLP sobre excerpts de 200 ms de **constant-Q log-magnitude spectrogram** para onset detection en 2006 y obtuvieron mejores resultados que con STFT; Humphrey & Bello [80] usan constant-Q de magnitud lineal para reconocimiento de acordes; Korzeniowski & Widmer [107] usan espectrogramas de log-frecuencia. Nótese la coherencia: **todas las tareas de armonía usan eje logarítmico**.

### 3.7. El debate onda cruda vs. features: lo que el survey afirma en concreto

Aquí conviene ser quirúrgico, porque es donde más se distorsiona la posición del paper. El veredicto está en §IV-A y es **condicional a la tarea**:

**Para tareas de análisis (ASR, MIR, reconocimiento de sonido ambiente):**

> "Log-mel spectrograms provide a more compact representation, and methods using these features **usually need less data and training to achieve results that are, at the current state of the art, comparable in classification performance** to a setup where raw audio is used."

Léase con cuidado: el survey **no** dice que log-mel gane en precisión. Dice que **empatan en precisión** y que log-mel llega ahí **con menos datos y menos entrenamiento**. La onda cruda "avoids hand-designed features, which should allow to better exploit the improved modeling capability of deep learning models… However, this incurs higher computational costs and data requirements, and **benefits may be hard to realize in practice**."

**Para tareas de síntesis (separación de fuentes, realce, TTS, morphing de timbre):**

> "Using (log-mel) magnitude spectrograms poses the challenge to reconstruct the phase. In that case, **raw waveforms or complex spectrograms are generally preferred** as the input representation."

Ese es el criterio limpio: **el problema de la fase decide**. Si la salida es una etiqueta, tira la fase y usa log-mel. Si la salida es audio, no puedes tirarla.

**Las excepciones que el survey reconoce:** "However, some works report improvements using raw waveforms for analysis tasks [25], [146], [147]" — WaveNet, Ghahremani et al. (Interspeech 2016, modelado acústico desde el dominio de la señal) y Sailor & Patil (aprendizaje no supervisado de bancos de filtros auditivos con RBM convolucionales).

**El camino intermedio, que el survey trata como la línea más prometedora:** "some attempt to find a way in between by designing and/or initializing the first layers of a deep learning system to mimic engineered representations [18], [19], [23], [24]". Aquí está, entre otros, Sainath et al. [24] "Learning the Speech Front-end with Raw Waveform CLDNNs" — la propia coautora del survey construyendo un front-end aprendido cuyas primeras capas **imitan la computación del log-mel** pero con todos los parámetros del filtro aprendidos de los datos. Y en el extremo [25] (WaveNet) "the notion of a filter bank is discarded".

**Evidencia empírica concreta que el survey reporta en el dominio música** (§III-A2) — esto es lo más cercano a un número que ofrece:

| Trabajo | Entrada | Arquitectura | Resultado según el survey |
|---|---|---|---|
| Dieleman & Schrauwen [109] | log-mel de 3 s | CNN con convoluciones 1D cortas (solo sobre tiempo), promediando predicciones | Referencia |
| Dieleman & Schrauwen [109] | **raw samples**, filtro de primera capa dimensionado para igualar un frame de espectrograma | misma CNN | **"achieve worse results"** |
| Lee et al. [111] | **raw samples**, filtros muy cortos (tamaño **2 a 4**) intercalados con max-pooling | CNN "sample-level" | **"matching the performance of log-mel spectrograms"** |

La lectura es instructiva y el survey la deja implícita: **la onda cruda funciona cuando NO intentas imitar el espectrograma**. Dieleman eligió el filtro de primera capa "para calzar con un frame típico de espectrograma" y perdió; Lee eligió filtros de 2 a 4 muestras apilados con pooling —una jerarquía genuinamente aprendida, sin prior de STFT— y empató. Y las preguntas abiertas con las que el survey cierra §IV-A:

> "Are mel spectrograms indeed the best representation for audio analysis? Under what circumstances is it better to use the raw waveform? **Can we do better by exploring the middle ground, a spectrogram with learnable hyperparameters?** If we learn a representation from the raw waveform, does it still generalize between tasks or domains?"

---

## 4. Los modelos (§II-C)

### 4.1. MLP

El survey casi no le dedica espacio como familia propia: "for audio, multiple feedforward, convolutional, and recurrent (e.g. LSTM) layers are usually stacked to increase the modeling capability". El rol funcional del MLP aparece por el lado de la evidencia de Mohamed et al. [10] (capas bajas ≈ features adaptadas al hablante, capas altas ≈ discriminación por clase) y por el lado histórico: las DNN feedforward con millones de parámetros entrenadas sobre miles de horas fueron las que en 2012 "reduce[d] the word error rate dramatically" [3]. En las arquitecturas concretas, el MLP aparece siempre **al final**, como cabeza clasificadora.

### 4.2. CNN, convolución dilatada y WaveNet

La caracterización del survey (§II-C-a), que conviene tener textual porque es la que el slide reproduce:

> "CNNs are based on convolving their input with learnable kernels. In the case of spectral input features, **a 1-d temporal convolution or a 2-d time-frequency convolution** is commonly adopted, whereas **a time-domain 1-d convolution** is applied for raw waveform inputs."

Tres regímenes distintos, no uno. La convolución 1D temporal sobre espectrograma (convolucionar solo en $t$, tratando las frecuencias como canales) es una opción de primera clase que el slide de la clase no menciona y que Dieleman [109] usa. La 2D tiempo-frecuencia es la del "espectrograma como imagen".

Sobre el campo receptivo, el pasaje que la clase copia casi literal:

> "The receptive field (**the number of samples or spectra involved in computing a prediction**) of a CNN is fixed by its architecture. It can be increased by using larger kernels or stacking more layers. Especially for raw waveform inputs with a high sample rate, reaching a sufficient receptive field size may result in a large number of parameters of the CNN and high computational complexity. Alternatively, a **dilated convolution** (also called atrous, or convolution with holes) [25], [27]–[29] can be used, which applies the convolutional filter over an area larger than its filter length **by inserting zeros between filter coefficients**. A stack of dilated convolutions enables networks to obtain very large receptive fields with just a few layers, **while preserving the input resolution as well as computational efficiency**."

Vale la aritmética que el survey no escribe: con $L$ capas de kernel $k$ y dilatación duplicándose $1, 2, 4, \dots, 2^{L-1}$, el campo receptivo crece como

$$R = 1 + (k-1)\sum_{l=0}^{L-1} 2^{l} = 1 + (k-1)(2^{L}-1)$$

es decir **exponencial en la profundidad**, contra el crecimiento **lineal** $R = 1 + L(k-1)$ de la convolución estándar. Con $k=2$ y $L=10$ ya son 1024 muestras; WaveNet apila varios de estos bloques. Y el detalle que el slide de la clase **pierde**: "while preserving the input resolution". La dilatación amplía el contexto **sin hacer downsampling**, que es precisamente lo que la hace usable en tareas de salida densa (sample-por-sample en síntesis, frame-por-frame en detección de eventos). El survey vuelve a este punto en §III-A3: "in order to be able to output an event activity vector at a sufficiently high temporal resolution, the degree of max pooling or stride over time should not be too large — **if a large receptive field is desired, dilated convolution and dilated pooling can be used instead** [114]". La dilatación no es un truco de eficiencia: es lo que resuelve el conflicto entre contexto amplio y resolución de salida fina.

Y la admisión que ninguna clase repite (§II-C-a):

> "**Operational and validated theories on how to determine the optimal CNN architecture** (size of kernels, pooling and feature maps, number of channels and consecutive layers) for a given task **are not available at the time of writing**. Currently therefore, the architecture of a CNN is largely chosen experimentally based on a validation error, which has led to some **rule-of-thumb guidelines**, such as fewer parameters for less data [31], **increasing channel numbers with decreasing sizes of feature maps** in subsequent convolutional layers, considering the necessary size of temporal context, and task-related design."

Ese párrafo es, palabra por palabra, la barra lateral del diagrama del "Ejemplo 1" en el deck de la clase 39. Ver §12 y §13.

**WaveNet** aparece en el survey como caso paradigmático, referenciado como [25] y citado **catorce veces** a lo largo del texto (features, modelos, dilatación, síntesis, vocoder, evaluación). Los rasgos que el survey destaca: (i) descarta la noción de banco de filtros, aprendiendo "a causal regression model of the time-domain waveform samples **without any human prior knowledge**"; (ii) casts la predicción autorregresiva del sample "as a classification problem, the amplitude of the predicted sample being **quantized logarithmically** into distinct classes"; (iii) admite condicionamiento **global** (identidad del hablante) o **variable en el tiempo** ($f_0$, espectros mel); (iv) "WaveNet-based models for speech synthesis outperform state-of-the-art systems by a large margin, but their **training is computationally expensive**", resuelto por Parallel WaveNet [141]; (v) "The WaveNet yields a **higher MOS** than concatenative or parametric methods, which represented the previous state of the art."

### 4.3. RNN / LSTM

La caracterización (§II-C-b):

> "The effective context size that can be modeled by CNNs is limited, even when using dilated convolutions. RNNs follow a different approach for modeling sequences: They compute the output for a time step from both the input at that step and their hidden state at the previous step. This inherently models the temporal dependency in the inputs, and **allows the receptive field to extend indefinitely into the past**. For offline applications, bidirectional RNNs employ a second recurrence in reverse order, **extending the receptive field into the future**."

El argumento de capacidad, que es específico y poco citado: "in contrast to conventional HMMs, with linear growth of the number of recurrent hidden units in RNNs with all-to-all kernels, **the number of representable states grows exponentially, whereas training or inference time grows only quadratically at most** [33]". Es decir: un HMM con $N$ estados necesita $N$ parámetros de estado y una matriz $N\times N$; una RNN con $H$ unidades puede representar del orden de $2^H$ configuraciones de estado con $O(H^2)$ pesos. Ese es el argumento formal de por qué la RNN reemplazó al HMM, y el survey es de los pocos que lo enuncia.

**Variantes específicas de audio que el survey introduce y que casi nadie conoce:**

- **F-LSTM (Frequency LSTM)** [36]: recurre **sobre el eje de frecuencia**, no sobre el tiempo. "Distinctly from CNNs, F-LSTMs capture translational invariance through local filters and recurrent connections. **They do not require pooling operations and are more adaptable to a range of types of input features.**"
- **TF-LSTM (Time-Frequency LSTM)** [37]–[39]: "unrolled across both time and frequency, and may be used to model both spectral and temporal variations". Veredicto del survey: "**TF-LSTMs outperform CNNs on certain tasks** [39], but are **less parallelizable and therefore slower**."

Esa última frase es la que hay que retener: en 2018, la alternativa que ganaba en precisión perdía en paralelismo. Es exactamente el trade-off que el Transformer vino a romper (§10).

### 4.4. CRNN — la receta que la clase presenta como propia

El survey la formula en dos líneas (§II-C-b):

> "Alternatively, RNNs can process the output of a CNN, forming a **Convolutional Recurrent Neural Network (CRNN)**. In this case, **convolutional layers extract local information, and recurrent layers combine it over a longer temporal context.**"

Y la cierra en §IV-B con la comparación completa:

> "Across the domains, CNNs, RNNs and CRNNs are employed successfully, **with no clear preference**. All three can model temporal sequences, and solve sequence classification, sequence labelling and sequence transduction tasks. **CNNs have a fixed receptive field, which limits the temporal context taken into account for a prediction, but at the same time makes it very easy to widen or narrow the context used. RNNs can, theoretically, base their predictions on an unlimited temporal context, but first need to learn to do so**, which may require adaptations to the model (such as LSTM) and **prevents direct control over the context size**. Furthermore, they require processing the input sequentially, making them **slower to train and evaluate on modern hardware** than CNNs. **CRNNs offer a compromise in between, inheriting both CNNs and RNNs advantages and disadvantages.**"

**Contraste con el slide de la clase.** El slide dice:

> "CNNs: good properties to learn local features. Specifically, by exploiting: i) Local span of each filter + ii) Translation invariance of convolutional operator → they discover relevant local features in the input data.
> RNNs: good properties to learn temporal features. Specifically, by exploiting a recurrent operator, they discover relevant distant temporal relations (global) in the input data.
> MLPs: good properties to classify input data. Specifically, by exploiting informative features, they learn good classifiers that map input features to discriminative spaces (class discrimination)."

La correspondencia es directa: "convolutional layers extract local information" → "CNNs learn meaningful local features"; "recurrent layers combine it over a longer temporal context" → "RNNs learn distant and global temporal features"; y para el MLP, el survey lo respalda vía Mohamed et al. [10] ("the activations of the upper layers of DNNs can be thought of as performing class-based discrimination") y vía la descripción de la CLDNN en §III-A1 ("finally passed through a few feedforward layers and an output softmax layer").

**Los cuatro matices que el slide pierde:**

1. **"With no clear preference."** El slide presenta CNN+RNN+MLP como *la* receta consensuada. El survey dice explícitamente que **no hay consenso**, que CNN solo, RNN solo y CRNN funcionan los tres, y que la elección probablemente refleja el conocimiento tácito de cada grupo de investigación más que una superioridad real. En música es todavía más enfático (§III-A2): "neither within nor across tasks is there a consensus on what input representation to use (log-mel spectrogram, constant-Q, raw audio) and what architecture to employ (CNNs or RNNs or both, 2D or 1D convolutions, small square or large rectangular filters)".
2. **El campo receptivo fijo de la CNN es una ventaja, no solo una limitación.** "makes it very easy to widen or narrow the context used" — es controlable por diseño. La RNN tiene contexto teóricamente infinito pero **no controlable**: "prevents direct control over the context size".
3. **La RNN puede tener contexto infinito, pero tiene que aprenderlo.** "can, theoretically, base their predictions on an unlimited temporal context, **but first need to learn to do so**". El slide dice que la RNN "descubre relaciones temporales distantes"; el survey dice que *puede* descubrirlas *si logra aprenderlas*, y que por eso hizo falta inventar la LSTM.
4. **El costo de paralelismo.** El survey nombra la desventaja estructural de la recurrencia —procesamiento secuencial, más lento en hardware moderno— que el slide omite por completo. Es una omisión cara, porque es exactamente el argumento que explica por qué el campo se movió a Transformers y Conformers en los dos años siguientes.

Ejemplo canónico de CRNN en el survey: McFee & Bello [106] para reconocimiento de acordes — "a 2D convolution learning spectrotemporal features, followed by a **1D convolution integrating information across frequencies**, followed by a bidirectional GRU", con **170 clases de acordes** y *side targets* para incorporar relaciones entre ellas.

### 4.5. Sequence-to-sequence (§II-C-c)

El survey plantea el problema de fondo: "due to the large complexity involved in audio processing tasks, conventional systems usually divide the task into series of sub-tasks and solve each task independently. Taking speech recognition as an example… traditional ASR systems comprise separate **acoustic, pronunciation, and language modeling** components that are normally trained independently."

El argumento end-to-end: los sistemas seq2seq "are trained to optimize criteria that are **related to the final evaluation metric** (such as word error rate)"; "are fully neural, and **do not use finite state transducers, a lexicon, or text normalization modules**"; "it does not require **bootstrapping from decision trees or time alignments generated from a separate system**"; y "the process of decoding is also simplified".

El inventario:

| Modelo | Mecanismo | Referencia en el survey |
|---|---|---|
| **CTC** | Introduce un símbolo *blank* para igualar largos e **integra sobre todas las formas de insertar blanks**, optimizando la secuencia de salida en vez de cada etiqueta individual | [48]–[51] |
| **RNN-T** | Extensión de CTC por Graves con un **componente de modelo de lenguaje recurrente separado** | [42] |
| **Atención** | Aprende alineamientos entre entrada y salida **conjuntamente** con la optimización del objetivo | [43], [52], [53] |
| **LAS** | Encoder ≈ modelo acústico, módulo de atención ≈ modelo de alineamiento, decoder ≈ modelo de lenguaje, todo en una red | [43], [54] |

Frase del survey: "Among various sequence-to-sequence models, **listen, attend and spell (LAS) offered improvements over others** [54]." Y en §III-A1: Soltau et al. [45] entrenaron un CTC con **targets a nivel de palabra** que superó a un baseline CD-fonema en captioning de video de YouTube.

### 4.6. GAN y VAE (§II-C-d, §III-B3)

El survey es notoriamente frío con las GAN en audio:

> "**Despite the success of GANs for image synthesis, their use in the audio domain has been limited.** GANs have been used for source separation [56], music instrument transformation [57] and speech enhancement to transform noisy speech input to denoised versions [58]–[61]."

Y en el caso más medido, §III-B2: SEGAN [58] "yields improvements in perceptual speech quality metrics over the noisy data and a traditional enhancement baseline", **pero** — y esto es lo interesante — "In [59], GANs are used to enhance speech represented as log-mel spectra. **When GAN-enhanced speech is used for ASR, no improvement is found compared to enhancement using a simpler regression approach.**" O sea: mejora la métrica perceptual, no mejora la tarea downstream. Ese resultado negativo es de Donahue, Li y Prabhavalkar (ICASSP 2018) — otra vez, coautores del survey reportando contra su propia línea.

Los **VAE** aparecen en dos lugares: en pérdidas (§II-C-e), con el ejemplo de Piano Genie [64] donde "one loss function was customized to encourage the latent variables of a variational autoencoder to remain inside a defined range and another to have changes in the control space be reflected in the generated audio"; y en síntesis blockwise (§III-B3), donde "the sound is often synthesised from a low-dimensional latent representation, from which it needs to be upsampled (e.g. through nearest neighbor or linear interpolation) to the high resolution sound. **Artifacts, induced by the different layer resolutions, can be ameliorated through random phase perturbation in different layers** [140]" (WaveGAN de Donahue et al.). Ese artefacto —el *checkerboard* del upsampling, audible como un zumbido tonal fijo— es un problema específico de audio que rara vez se menciona.

### 4.7. Dos secciones que suelen ignorarse: pérdidas y fase

**Funciones de pérdida (§II-C-e).** El survey hace una observación de fondo que un ingeniero debería tener grabada:

> "Comparing two audio signals by taking the **MSE between the samples in the time domain is not a robust measure**. For example, the loss for **two sinusoidal signals with the same frequency would entirely depend on the difference between their phases.**"

Dos señales perceptualmente idénticas pueden tener MSE máximo en el tiempo. Es el mismo problema de fase de §3.2, ahora del lado de la pérdida. Alternativas que cataloga: MSE entre log-mel (compara envolventes espectrales), MSE entre log-mel **espectrogramas** (agrega estructura temporal), **soft-DTW diferenciable** [62] para tolerar deformaciones temporales no lineales, y **distancia de earth mover / Wasserstein** [63]. Y pérdidas específicas de tarea: en separación de fuentes, "an objective differentiable loss function can be designed based on **psychoacoustic speech intelligibility experiments**" (§II-C-e; en §III-B1 aparece la referencia concreta [123], Kolbæk et al., maximizando una medida STOI de inteligibilidad).

**Modelado de fase (§II-C-f).** El inventario completo de opciones, que es el mejor resumen breve del tema que conozco:

1. **Griffin-Lim** [65] para estimar fase desde la magnitud — "the accuracy of the estimated phase is **insufficient to yield high quality audio**".
2. **Vocoder neural**: entrenar una red (WaveNet) para generar la señal temporal desde log-mel [66] (Tacotron 2).
3. **Espectro complejo como entrada**: magnitud + fase como features [67].
4. **Targets complejos** [68] (complex ratio masking).
5. **Extender toda la red al dominio complejo**: "all operations (convolution, pooling, activation functions) in a DNN may be extended to the complex domain" [69] (Deep Complex Networks).

---

## 5. Las tareas, una por una (§III)

### 5.1. Tabla resumen

Advertencia metodológica: **la columna "benchmark" no viene del survey.** El survey no reporta benchmarks ni resultados numéricos por tarea (§1). Los datasets marcados con ✱ **sí** están nombrados en el texto; los demás los agrego yo como referencia de campo y están marcados como tales.

| Tarea | Planteo (nomenclatura §II-A) | Representación típica según el survey | Arquitectura dominante en 2018 según el survey | Dataset/benchmark |
|---|---|---|---|---|
| **ASR** | *sequence transduction* (audio → palabras) | log-mel; también raw waveform con front-end aprendido | CLDNN → seq2seq (CTC, RNN-T, LAS) | LDC ✱; LibriSpeech, Switchboard (agregados) |
| **Voice activity / endpointing** | *sequence labeling* binario | log-mel | Grid-LSTM, conv. dilatada + gating [95], [143] | — |
| **Speaker / language ID** | *sequence classification* | raw waveform (SincNet [96]), log-mel | CNN/DNN | — |
| **Music tagging** | *multi-label sequence classification* (global) | log-mel 3 s o 29 s; raw samples | CNN 1D, FCN, CNN sample-level | Million Song Dataset ✱; MagnaTagATune (agregado) |
| **Transcripción / notas** | *sequence labeling* o *transduction* | constant-Q / log-frecuencia | CNN, CRNN | MusicNet ✱ |
| **Chord recognition** | *sequence labeling* multiclase | constant-Q, magnitud lineal, contrast-normalized | CNN [80], CNN+chroma [107], CRNN 170 clases [106] | Isophonics Beatles ✱ |
| **Onset detection** | *event detection* (binario por frame) | constant-Q log-magnitud 200 ms; log-mel 15 frames | MLP [84] → BLSTM [100] → CNN [16] | — |
| **Beat / downbeat tracking** | *event detection* | espectrograma | CNN [102] + HMM; RNN [103] + DBN; CRNN [104] | — |
| **Tempo** | *sequence regression* (o clasificación discretizada) | espectrograma, excerpts de **12 s** | CNN directa [108] | — |
| **Segmentación estructural** | *event detection* (fronteras) | espectrograma fuertemente submuestreado, campo receptivo hasta **60 s** | CNN [105] | — |
| **Acoustic scene classification** | *sequence classification* multinomial | log-mel | CNN | DCASE ✱ |
| **Acoustic event detection** | *sequence labeling* polifónico (multi-label por frame) | log-mel | RNN [113]; CNN con conv./pooling dilatados [114] | DCASE ✱ |
| **Audio tagging** | *multi-label sequence classification* (global, sin timing) | log-mel | CNN | AudioSet ✱ (>2M snippets) |
| **Localización / DOA** | *multi-label classification* sobre grilla de direcciones, o *regression* | espectro de fase [115], magnitud [118], GCC entre canales [117] | CNN con kernels que abarcan canales; CRNN [118] | — |
| **Separación de fuentes** | *regression per time step* (máscara T-F) | STFT complejo / magnitud | CNN [121], RNN [122], **deep clustering** [124], **deep attractor network** [125] | wsj0-2mix (agregado) |
| **Realce de voz** | *regression per time step* | STFT | denoising autoencoder [137], CNN [121], RNN [138], SEGAN [58] | — |
| **Síntesis / TTS** | *sequence transduction*, autorregresiva | raw waveform (WaveNet), log-mel como condicionamiento | WaveNet [25], SampleRNN [34], WaveRNN [35], Parallel WaveNet [141] | — |

### 5.2. Reconocimiento de voz (§III-A1)

El arco histórico que el survey traza: GMM-HMM de estados de trifono dominó "for decades", con virtudes reales ("mathematical elegance, which leads to many principled solutions to practical problems such as speaker or task adaptation"); hacia 1990 el entrenamiento discriminativo superó a máxima verosimilitud; se propusieron híbridos con redes [88]–[90] (TDNN de Waibel, RNN de Robinson, el libro de Bourlard & Morgan); y **en 2012 las DNN con millones de parámetros sobre miles de horas bajaron el WER dramáticamente** [3].

Después: "In addition to the great success of deep feedforward and convolutional networks [91], **LSTMs and GRUs have been shown to outperform feedforward DNNs** [92]. Later, a cascade of convolutional, LSTM and feedforward layers, i.e. the **convolutional, long short-term memory deep neural network (CLDNN)** model, was further shown to **outperform LSTM-only models** [93]."

**La descripción de la CLDNN en el survey**, que es literalmente el "Ejemplo 1" de la clase:

> "In CLDNNs, a window of input frames is **first processed by two convolutional layers with max-pooling layers to reduce the frequency variance in the signal**, then **projected down to a lower-dimensional feature space** for the following LSTM layers to model the temporal correlations, and finally **passed through a few feedforward layers and an output softmax layer**."

Los tres roles están nombrados con precisión funcional: la CNN **reduce varianza en frecuencia** (no "aprende features locales" en abstracto: normaliza la variabilidad de altura tonal entre hablantes); la proyección lineal **reduce dimensionalidad** antes de la LSTM; la LSTM **modela correlaciones temporales**; el MLP **discrimina**. Cotejando con el paper original de Sainath et al. (ICASSP 2015), que también está en esta carpeta: 40-dim log-mel calculado cada **10 ms**; **2 capas convolucionales de 256 feature maps**, filtro **9×9** en la primera y **4×3** en la segunda (no 4×4); max-pooling **no solapado, solo en frecuencia, tamaño 3, solo en la primera capa**; una **capa lineal** de reducción a **256** salidas; **2 capas LSTM de 832 celdas con proyección de 512**; capas totalmente conectadas de **1.024** unidades; y la salida retrasada **5 frames** para incorporar contexto futuro sin latencia adicional. Ver §13 para las discrepancias con el slide.

El giro hacia seq2seq: "With the adoption of RNNs for speech modeling, **the conditional independence assumption of the output targets incurred by the traditional HMM-based phone state modeling is no longer necessary**, and the research field shifted towards full sequence-to-sequence models."

Y la nota sobre madurez industrial: "Virtual assistants, such as Google Home, Amazon Alexa and Microsoft Cortana, all adopt voice as the main interaction modality." Más transfer learning entre idiomas: "Transfer learning has been used to boost the performance of ASR systems on **low resource languages with data from rich resource languages** [75]."

### 5.3. Music Information Retrieval (§III-A2)

El survey abre con la diferencia estructural respecto de la voz: "Compared to speech, **music recordings typically contain a wider variety of sound sources of interest**. In many kinds of music, their occurrence follows common constraints in terms of time and frequency, **creating complex dependencies within and between sources**."

Eso es lo que hace la música más difícil que la voz en separación (§III-B1): "in speech it is assumed that the signal is sparse and that different sources are independent from each other. In environmental sounds, independence can usually be assumed. **In music there is a high dependence between simultaneous sources** as well as there are specific temporal dependencies across time, in the waveform as well as regarding long-term structural repetitions." Los instrumentos de un acorde no son fuentes independientes: comparten fundamental, se solapan en armónicos, y están sincronizados rítmicamente. Todo lo que la separación ciega asume, la música lo viola.

La taxonomía MIR completa del survey: análisis de bajo nivel (onset/offset, estimación de $f_0$), análisis rítmico (beat tracking, identificación de compás, downbeat, tempo), análisis armónico (tonalidad, extracción de melodía, acordes), análisis de alto nivel (detección de instrumento, separación, transcripción, segmentación estructural, reconocimiento de artista, género, mood) y comparación de alto nivel (temas repetidos, identificación de covers, similitud, alineamiento con partitura).

**El hallazgo transversal más interesante de esta sección** —y el más fácil de pasar por alto— está en el párrafo de detección de eventos:

> "Comparing approaches, both CNNs with fixed-size temporal context and RNNs with potentially unlimited context are used successfully for event detection. **Interestingly, for the former, it seems critical to blur training targets in time** [16], [84], [105]."

Tres trabajos independientes (Schlüter & Böck en onsets, Lacoste & Eck en onsets, Ullrich et al. en fronteras estructurales) encontraron que hay que **difuminar el target en el tiempo** para que una CNN aprenda detección de eventos. La razón: la anotación humana de un onset tiene jitter de decenas de milisegundos; un target one-hot exacto castiga con pérdida máxima una predicción que está a un frame de distancia y es esencialmente correcta. Difuminar el target (una gaussiana o rectángulo de ±2 frames alrededor del evento) convierte un problema mal condicionado en uno aprendible. Es *label smoothing* en el eje temporal, y es un truco práctico de primera línea que el slide no menciona.

Y el cierre honesto de la sección, ya citado: en música **no hay consenso** ni sobre representación ni sobre arquitectura.

### 5.4. Sonidos ambientales (§III-A3) — la sección más importante para la clase

La clase 39 "mostly focuses on environmental sounds". El survey divide el campo en exactamente tres:

**(a) Acoustic scene classification.** "Aims to label a whole audio recording with a single scene label. Possible scene labels include for example 'home', 'street', 'in car', 'restaurant', etc. The set of scene labels is defined in advance, rendering this a **multinomial classification problem**." Una etiqueta global, exclusiva. En la nomenclatura de §II-A: *sequence classification*.

**(b) Acoustic event detection.** "Aims to estimate the **start and end times** of individual sound events such as footsteps, traffic light acoustic signalling, dogs barking, and assign them an event label." Nomenclatura: *sequence labeling* / *event detection*. La implementación práctica que el survey recomienda: "A simple and efficient way to apply supervised machine learning to do detection is to **predict the activity of each event class in short time segments** using a supervised classifier."

Sobre el contexto: "Usually, the supervised classifier used to do detection will use contextual information, i.e., acoustic features computed from the signal **outside the segment to be classified**. A simple way to do so is to **concatenate acoustic features from multiple context frames** around the target frame, as done in the baseline method for the public DCASE evaluation campaign in 2016 [112]. Alternatively, classifier architectures which model temporal information may be used: for example, recurrent neural networks may be applied to map a sequence of frame-wise acoustic features to a sequence of binary vectors representing event class activities [113]."

Y la advertencia sobre pooling que ya cité: max-pooling o stride temporal agresivo **destruye la resolución de salida** que la tarea necesita; usar dilatación en su lugar [114].

**(c) Tagging.** "Aims to predict the activity of multiple (possibly simultaneous) sound classes, **without temporal information**." Nomenclatura: *multi-label sequence classification*.

**Detección polifónica y el argumento del clasificador multi-etiqueta.** "In both tagging and event detection, multiple event classes can be targeted that can be active simultaneously. In the context of event detection, this is called **polyphonic event detection**. In this approach, the activity of each class can be represented by a **binary vector**… If overlapping classes are permitted, the problem is a **multilabel classification problem**." Y el hallazgo empírico:

> "It has been found out that using a **multilabel classifier to jointly predict the activity of multiple classes at once produces better results**, instead of using single-class classifiers for each class separately. This might be for example due to the multiclass classifier being able to **model the interaction of simultaneously active classes**."

Consecuencia de implementación, que el survey no explicita pero se deduce: la salida es **sigmoide por clase con binary cross-entropy**, no softmax. Un softmax fuerza $\sum_c p_c = 1$, o sea competencia entre clases, que es justamente lo contrario de lo que se quiere en polifonía. El slide de la clase dice, correctamente, "Output: softmax (**or sigmoids**) for class label(s)".

Y el cierre: "Since the analysis of environmental sounds is a **less established research field** in comparison to speech and music, the size and diversity of available datasets… is more limited. Most of the open data has been published in the context of annual DCASE challenges. **Because of the limited size of annotated environmental datasets, data augmentation is a commonly used technique in the field, and it has been found highly effective.**" Ahí está la justificación directa del bloque de augmentation de la clase.

### 5.5. Localización y tracking (§III-A4)

Uso declarado: "can be used as a part of a source separation or speech enhancement system to **separate a source from the estimated source direction**, or in a **diarization** system to estimate the activity of multiple speakers."

Dos formulaciones alternativas, que son un ejemplo perfecto del eje de §II-A:

1. **Clasificación**: "forming a fixed grid of possible directions, and by using **multilabel classification** to predict if there is an active source in a specific direction" [115].
2. **Regresión**: "using regression to predict the directions [116] or spatial coordinates [117] of target sources."

Features de entrada catalogadas: **espectro de fase** [115], **espectro de magnitud** [118] y **correlación cruzada generalizada (GCC) entre canales** [117]. La observación de diseño: "source localization requires the use of **interchannel information**, which can also be learned by a deep neural network with a suitable topology from within-channel features, for example by **convolutional layers where the kernels span multiple channels**" [118]. Es decir: no hace falta calcular GCC a mano si el kernel de la primera capa abarca los canales — la red aprende su propio estimador de diferencia interaural.

Aquí la fase, que en análisis monocanal era un estorbo, es **la señal principal**: la diferencia de fase entre micrófonos es lo que codifica la dirección de llegada.

### 5.6. Separación de fuentes y realce (§III-B1, §III-B2)

Formalización del survey. Mezcla:

$$x_m(n) = \sum_{i=1}^{I} s_{m,i}(n) \tag{1}$$

con $i$ índice de fuente, $I$ número de fuentes, $n$ índice de muestra, $m$ índice de micrófono, y $s_{m,i}(n)$ la **imagen espacial** de la fuente $i$ en el micrófono $m$ (no la fuente "seca": ya incluye la respuesta de la sala hasta ese micrófono).

Enmascaramiento en el dominio tiempo-frecuencia:

$$\hat{S}_{m,i}(f,t) = M_{m,i}(f,t)\, X_m(f,t) \tag{2}$$

**Las tres razones por las que se trabaja en tiempo-frecuencia** (§III-B1), que es el mejor párrafo del survey sobre por qué el espectrograma no es solo una conveniencia:

1. "The structure of natural sound sources **is more prominent in the time-frequency domain**, which allows modeling them more easily than time-domain signals."
2. "**Convolutional mixing** which involves an acoustic transfer function from a source to a microphone **can be approximated as instantaneous mixing in the frequency domain**, simplifying the processing."
3. "Natural sound sources are **sparse in the time-frequency domain**, which facilitates their separation in that domain."

La razón (2) merece énfasis: la propagación en una sala es una **convolución** con la respuesta impulsiva; en el dominio de la frecuencia, y suponiendo ventanas más largas que la respuesta, eso se convierte en una **multiplicación escalar por bin**. Un problema de deconvolución se vuelve un problema de escalado. La razón (3) es la que hace que enmascarar funcione: si dos fuentes rara vez ocupan el mismo bin con energía comparable (hipótesis W-disjoint orthogonality), una máscara binaria puede separarlas casi perfectamente.

Y la elección de la STFT sobre alternativas: "The spectrum $X_m(f,t)$ is typically calculated using the STFT because it can be implemented efficiently using the FFT, and also because the **STFT can be easily inverted**. The use of other time-frequency representations is also possible, such as constant-Q or mel spectrograms. **The use of these has however become less common since they reduce output quality, and deep learning does not require a compact input representation** that they would provide in comparison to the STFT." Nótese la inversión del argumento: mel existía para comprimir; con deep learning la compresión dejó de ser necesaria y su costo (pérdida de calidad, no invertibilidad exacta) pasó a dominar.

Dos categorías de método monocanal: predecir la **máscara** $M_i(f,t)$, o predecir el **espectro de la fuente** $S_i(f,t)$. El target supervisado es "either the oracle mask or the clean signal spectrum [120]"; la máscara oracle "takes either **binary values, or continuous values between 0 and 1**".

Métodos destacados: **deep clustering** [124] — "uses supervised deep learning to estimate **embedding vectors for each time-frequency point**, which are then clustered in an unsupervised manner. **This approach allows separation of sources that were not present in the training set**" (resuelve el *permutation problem* y generaliza a hablantes no vistos); y su extensión, la **deep attractor network** [125], "based on estimating a single attractor vector for each source", con "state-of-the-art results in single-channel source separation".

Multicanal: aplicar los métodos monocanal al espectro de cada canal [126]; agregar **features espaciales** además de las espectrales [127]; o usar la DNN para **estimar los pesos de un beamformer** [128].

**Realce de voz** (§III-B2): "Conventional denoising approaches, such as **Wiener methods, usually assume stationary noise, whereas deep learning approaches can model time-varying noise**." Esa es la ventaja estructural en una línea. Arquitecturas: denoising autoencoders [137], convolucionales [121], recurrentes [138], y GAN [58]–[61] con el resultado negativo ya comentado.

### 5.7. Síntesis (§III-B3)

Los cuatro requisitos que el survey le pone a un modelo generativo de sonido —una lista notablemente bien pensada:

1. **Similaridad**: "similar to sounds from which the model is trained, in terms of typical acoustic features (timbre, pitch content, rhythm)", y reconocible/inteligible.
2. **Originalidad**: "significantly different from sounds in the training set, **instead of simply copying training set sounds**".
3. **Diversidad**.
4. **Controlabilidad**: "condition the sound synthesis, e.g. in speech synthesis on a speaker, a prosodic trajectory, a harmonic schema in music, or physical parameters".

Más un requisito operacional: "training and generation time should be small; **ideally generation should be possible in real-time**."

Dos regímenes: **blockwise** (VAE/GAN desde latente de baja dimensión, con el problema del upsampling y los artefactos) y **autorregresivo** (sample a sample). Dentro del autorregresivo, el survey contrasta: RNN con contexto "infinitamente largo" pero entrenamiento caro, mitigado por **jerarquías temporales** (SampleRNN [34]: "layers of RNNs may be stacked to process the sound on different temporal resolutions, where the activations of one layer depend on the activations of the next layer with coarser resolution") y por **RNN dispersas** (WaveRNN [35], que "folds long sequences into a batch of shorter ones"); versus **convoluciones dilatadas apiladas** (WaveNet).

El pipeline TTS de dos módulos [66] (Tacotron 2): "(1) a neural network is trained from textual input to predict a **sequence of mel spectra**, used as contextual input to (2) a **WaveNet** yielding synthesised speech." Esa es la arquitectura que resolvió el problema de la fase en producción: mel como representación intermedia + vocoder neural.

Y la evaluación de generativos (§III-B3), que es más rica que la de §II-E: **reconocibilidad** objetiva vía clasificador (inception score [140]) o subjetiva por elección forzada; **diversidad** como "the average Euclidean distance between the sounds and their nearest neighbors" sobre log-mel normalizados; **originalidad** como "the average Euclidean distance between a generated sample to their nearest neighbor in the real training set". Esas tres métricas juntas atacan el modo de falla más común de un generativo: memorizar.

---

## 6. Etiquetas fuertes vs. débiles, y el multiple instance learning

**Advertencia de fidelidad, primero.** He revisado el texto completo del PDF (extracción con `pdftotext` y con `pdftotext -layout`, resultados idénticos). Los hechos verificables:

- **La expresión "multiple instance learning" NO aparece en el survey.** Cero ocurrencias.
- **La expresión "weak labeling" / "weakly labeled" aparece exactamente dos veces**: una en §II-A, "the weakly-labelled AudioSet dataset [9]", sin definirla; y otra en el título de la referencia [152], Schlüter, "Learning to pinpoint singing voice from **weakly labeled** examples" (ISMIR 2016), citada en §IV-E a propósito de interpretabilidad.
- **La expresión "strong labels" no aparece nunca.**

Es decir: **el survey trata la sustancia de la distinción a fondo, pero no usa la terminología estándar del campo y no menciona MIL.** Si alguien afirma que el survey "trata en detalle el multiple instance learning", está atribuyéndole algo que no dice. Lo desarrollo igual, porque la sustancia sí está y la correspondencia con el slide es exacta.

### 6.1. Lo que el survey sí dice: los dos ejes de la Fig. 1

La Fig. 1 cruza dos ejes independientes:

**Eje 1 — número de etiquetas a predecir:**

| Caso | Nombre en el survey | Ejemplos del survey |
|---|---|---|
| Una etiqueta global para toda la secuencia | *sequence classification* | idioma, hablante, tonalidad musical, escena acústica |
| Una etiqueta por paso de tiempo | *sequence labeling* / *event detection* | anotación de acordes, detección de actividad vocal, cambios de hablante, onsets de nota |
| Secuencia de etiquetas de largo libre | *sequence transduction* | speech-to-text, transcripción musical, traducción |

**Eje 2 — tipo de cada etiqueta:**

| Caso | Nombre en el survey | Ejemplos |
|---|---|---|
| Una sola clase | (clasificación) | escena acústica, tonalidad |
| Un conjunto de clases | *multi-label* | varios eventos acústicos simultáneos (AudioSet), conjunto de alturas musicales |
| Un valor numérico | *regression* | tempo, distancia a una fuente en movimiento, pitch, el siguiente sample de audio |

Con la nota de que "regression problems can always be discretized and turned into classification problems" (el ejemplo de los 8 bits → 256 clases), y la observación de que "multi-label classification can be **particularly efficient when classes depend on each other**".

### 6.2. La correspondencia exacta con el slide

El slide "Audio Applications: General Sounds — Classification" presenta:

> "**Single global-label vs single local-labels**: Ex.: John is talking vs John says: 'you must know AI'."
> "**Multiple global-labels vs multiple local-labels**: Temporal tagging of audio sequences. Multiple speaker identification. Sound recognition at different levels of granularity."

con dos diagramas: uno con `ID5:John` sobre segmentos sucesivos, y otro con `Kitchen / Window / ID5:John / Door / People / Steps / ID7:Laura / Radio / Bike / Door / Steps` en filas y columnas.

La correspondencia es el **producto cartesiano de los dos ejes de la Fig. 1**:

| Slide | Fig. 1 del survey (eje 1 × eje 2) | Terminología estándar del campo |
|---|---|---|
| Single **global**-label | etiqueta global única × clase única = *sequence classification* | **acoustic scene classification**, clip-level classification |
| Single **local**-labels | etiqueta por paso × clase única = *sequence labeling* | **frame-level classification**, segmentación |
| Multiple **global**-labels | etiqueta global única × conjunto de clases = *multi-label sequence classification* | **audio tagging** con **weak labels** |
| Multiple **local**-labels | etiqueta por paso × conjunto de clases = *sequence labeling* multi-etiqueta | **sound event detection (SED)** polifónica con **strong labels** |

Lo que el slide **omite** del cuadro del survey: (a) el tercer valor del eje 2, la **etiqueta numérica** → toda la familia de regresión (tempo, $f_0$, DOA, separación de fuentes como regresión por frame); (b) el tercer valor del eje 1, **sequence transduction** → ASR, transcripción musical, traducción. El slide dedica una lámina aparte a "Event detection" y otra al bloque de *Speech*, así que las cubre parcialmente por otro lado, pero pierde el hecho de que en el survey son **casillas de la misma matriz**, no categorías paralelas.

### 6.3. Por qué la terminología estándar importa

Lo esencial de la distinción, que el survey deja implícito y conviene decir explícito:

- **Strong labels (etiquetas fuertes)**: la anotación incluye **tiempos de inicio y fin** de cada evento. Permite entrenar SED directamente: cada frame tiene su vector binario de target. Son **caras**: anotar los onsets/offsets de eventos en un clip de 10 segundos toma minutos de trabajo humano experto.
- **Weak labels (etiquetas débiles)**: la anotación dice **qué clases están presentes en el clip**, sin decir cuándo. Es lo que AudioSet ofrece para más de 2 millones de snippets de 10 segundos: barato de anotar (un humano marca casillas), imposible de usar directamente para entrenar un detector por frame.

Y **ahí es donde entra multiple instance learning**, que el survey **no** menciona: la formulación estándar trata cada clip como una **bolsa (*bag*)** de instancias (frames), con la etiqueta de bolsa positiva si **al menos una** instancia es positiva. Se entrena una red que produce predicciones por frame $p_c(t)$ y se agregan a nivel de clip con una función de *pooling* diferenciable —max, mean, o *attention pooling*— para comparar contra la etiqueta débil. El resultado es un detector con resolución temporal entrenado **solo** con etiquetas de clip. En la literatura post-2019 esto es el estándar (PANNs, AST, las tareas 4 de DCASE con datos débilmente etiquetados y sintéticos fuertemente etiquetados).

El survey tiene el punto de contacto en una sola cita, [152], Schlüter (ISMIR 2016), "Learning to **pinpoint** singing voice from weakly labeled examples" — que es exactamente un trabajo de MIL: localizar temporalmente la voz cantada entrenando solo con etiquetas de canción. Pero el survey lo cita en §IV-E como ejemplo de **interpretabilidad** ("investigate which parts of the input a prediction is based on"), no como método de aprendizaje débil. Es una oportunidad perdida del survey y vale registrarla.

**Consecuencia práctica para buscar literatura:** si buscas "single vs multiple labels, global vs local" no encuentras nada. Los términos que hay que usar son:

- `audio tagging` + `weakly labeled` → el caso multi-etiqueta global.
- `sound event detection` + `strongly labeled` → el caso multi-etiqueta local.
- `acoustic scene classification` → el caso mono-etiqueta global.
- `multiple instance learning` + `audio` → el puente entre ambos.
- `DCASE` → el challenge que define las tareas y las métricas oficiales de todo esto.

---

## 7. Data augmentation (§II-D, con refuerzo en §III-A3)

### 7.1. El catálogo completo del survey

El survey introduce el tema con la definición que el slide copia casi textual: "Data augmentation **generates additional training data by manipulating existing examples** to cover a wider range of possible inputs."

Inventario exhaustivo de lo que menciona, con su cita y su dominio de aplicación:

| Técnica | Dominio y cita en el survey | Qué invarianza impone |
|---|---|---|
| **Pitch shifting** (para ASR: *vocal tract length perturbation*, VTLP) | ASR [77]; chord recognition [80]; singing voice detection [81]; instrument recognition [82] | Invarianza al largo del tracto vocal / a la tonalidad |
| **Time stretching** | ASR [78]; singing voice detection [81]; instrument recognition [82] | Invarianza a la velocidad de habla / al tempo |
| **Simulación de sala (reverberación) + multicanal ruidoso** | far-field ASR [79] (Google Home) | Invarianza a la acústica de la sala y a la posición del micrófono |
| **Filtrado espectral** | singing voice detection [81]; instrument recognition [82] | Invarianza a la ecualización / respuesta del micrófono |
| **Combinación lineal de ejemplos con sus etiquetas** (*between-class learning*, análogo de mixup) | environmental sounds [83], Tokozume et al. ICLR 2018 | Regularización del espacio entre clases; "improves generalization" |
| **Mezcla de pistas separadas para sintetizar mezclas** | source separation | Genera pares (mezcla, fuentes) exactos y gratis |
| **Generación sintética completa con parámetros conocidos** | general | "A controlled gradual increase in complexity of the generated data eases understanding, debugging, and improving of machine learning methods" |

Y la advertencia central sobre datos sintéticos, que el slide sí recoge: "**the performance of an algorithm on real data may be poor if trained on generated data only**."

Más el refuerzo específico de dominio en §III-A3: en sonidos ambientales, por la escasez de datos anotados, "data augmentation is a commonly used technique in the field, and it has been **found highly effective**."

### 7.2. Lo que el survey NO menciona (verificado)

- **No menciona "adding noise" como técnica de augmentation** en §II-D. Lo más cercano es la simulación de sala, que produce "multi-channel **noisy** and reverberant speech" [79]. El slide de la clase lista "Add noise" como una de sus tres técnicas principales. Es una técnica legítima y universal (SNR aleatorio con un banco de ruidos tipo MUSAN/DEMAND), pero no viene de este survey.
- **No menciona SpecAugment.** No podía: Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition", es de **abril de 2019**, y el manuscrito del survey se recibió en **octubre de 2018**.
- **No menciona `Scaper`** (Salamon et al., WASPAA 2017), la librería de síntesis y augmentación de *soundscapes* que sí está en la bibliografía de la clase 39, y que es la herramienta que implementa la "generación sintética con parámetros conocidos" del survey. Otra ausencia notable dado el énfasis del survey en sonido ambiental.

### 7.3. Augmentation sobre el espectrograma y la conexión con SpecAugment

El survey menciona **filtrado espectral** ([81], [82]) como augmentation, que ya es una operación sobre la representación tiempo-frecuencia (multiplicar por una envolvente aleatoria suave en frecuencia). Pero no llega a la idea de **enmascarar bloques** del espectrograma.

**SpecAugment** (Park et al., Interspeech 2019) es la pieza que falta y que la clase debería mencionar. Tres operaciones sobre el log-mel, todas triviales de implementar y todas sobre la matriz ya calculada (costo cero de reprocesar audio):

1. **Time warping** — deformación no lineal del eje temporal.
2. **Frequency masking** — poner a cero $f$ bandas mel consecutivas, con $f$ uniforme en $[0, F]$.
3. **Time masking** — poner a cero $t$ frames consecutivos, con $t$ uniforme en $[0, T]$.

Por qué funciona, y por qué es conceptualmente distinto de las técnicas del survey: pitch shift y time stretch generan **variantes plausibles de la señal** (aumentan la cobertura del espacio de entrada). El masking genera señales **imposibles** —ninguna grabación real tiene una banda de frecuencia exactamente muerta durante 20 frames—; lo que hace es forzar a la red a **no depender de ninguna banda ni de ningún instante en particular**, o sea es dropout estructurado sobre la entrada. Es regularización, no cobertura. Y resultó ser una de las mejoras más grandes en ASR end-to-end de esa época, precisamente porque los modelos seq2seq sobreajustaban con facilidad.

### 7.4. El criterio que falta: cuándo la transformación destruye la etiqueta

Este es el punto que ni el survey ni el slide explicitan y que es el más importante en la práctica. La regla general: **una augmentación es válida si y solo si la transformación preserva la etiqueta**, es decir, si la invarianza que impone es una invarianza real de la tarea.

| Transformación | Válida | Destruye la etiqueta |
|---|---|---|
| **Pitch shifting** | ASR (el contenido léxico no depende del pitch — de hecho VTLP simula variación de tracto vocal); detección de voz cantada; clasificación de instrumento **con cuidado** | **Identificación de hablante** (el pitch *es* parte de la identidad); **estimación de $f_0$ / transcripción de notas** (mueve el target); **detección de tonalidad musical**; **clasificación de género musical** si el shift es grande (cambia el timbre percibido y puede sacar la voz de su rango natural); **clasificación de instrumento** si el shift excede el rango real del instrumento |
| **Time stretching** | ASR; clasificación de escena; tagging | **Estimación de tempo** y **beat tracking** (mueve el target directamente); cualquier tarea con target temporal absoluto sin reescalar las anotaciones |
| **Adición de ruido** | casi todo lo de reconocimiento; robustez | **Realce de voz / separación de fuentes** si el ruido se agrega a la referencia limpia en vez de solo a la entrada; **estimación de SNR** |
| **Reverberación simulada** | far-field ASR, reconocimiento robusto | **Estimación de distancia a la fuente**; **localización / DOA** si la simulación no actualiza la geometría; **detección de reverberación** |
| **Filtrado espectral / EQ aleatoria** | tagging, escena, voz cantada | **Identificación de instrumento** con filtros agresivos (el timbre *es* la envolvente espectral); **detección de formantes / clasificación de vocales** |
| **Mixup / between-class** | clasificación multi-etiqueta y mono-etiqueta con targets blandos | **Detección de eventos con targets temporales exactos** (superponer dos clips desalinea los onsets); cualquier tarea donde la mezcla de dos entradas no corresponda a la mezcla de sus etiquetas |
| **Time / frequency masking** | ASR, tagging, escena — regularización general | **Separación de fuentes** (la máscara borra información que el target sí contiene); tareas de reconstrucción donde entrada y salida deben ser consistentes |
| **Inversión temporal** | casi nada en audio | **Todo**: el habla invertida es ininteligible, un ataque de piano invertido suena a órgano. A diferencia del flip horizontal en visión, **no es una simetría del dominio** |

La última fila merece énfasis porque es la diferencia más limpia entre audio e imagen. En visión, el flip horizontal es el augmentation por defecto, gratis y universal. En audio **no hay equivalente**: el eje temporal tiene flecha (causalidad, ataque-decaimiento) y el eje de frecuencia tiene semántica absoluta (no puedes "voltear" el espectro). Las simetrías baratas de visión simplemente no existen aquí, y por eso todas las técnicas del survey son transformaciones **físicas** (pitch, tiempo, sala, canal), no geométricas.

---

## 8. Datos y evaluación (§II-D, §II-E)

### 8.1. Los datasets que el survey nombra

Lista completa, verificada — es **corta**, y esa cortedad es en sí misma un dato:

| Dataset | Dominio | Cómo lo describe el survey |
|---|---|---|
| **ImageNet** [70] | (contraste, visión) | "14 million (2019) hand-labeled images", "a major factor" en el break-through de DL en visión |
| **LDC** [71] (Linguistic Data Consortium) | voz | "For speech recognition, there are **large datasets**, for English in particular" |
| **Million Song Dataset** [72] | música | "for music sequence classification or music similarity" |
| **MusicNet** [73] | música | "addresses **note-by-note sequence labeling**" |
| **Isophonics / Beatles reference annotations** [74] | música | "Datasets for higher-level musical sequence labeling, such as chord, beat, or structural analysis are often **much smaller**" |
| **AudioSet** [9] | sonido ambiental | "more than **2 million audio snippets**", descrito en §II-A como **weakly-labelled** |
| **DCASE** [112] | sonido ambiental | "Most of the open data has been published in the context of annual DCASE challenges" |

Y el diagnóstico transversal (§II-D): "**there is no such a well labeled dataset that can be shared across domains** including speech, music, and environmental sounds."

Reforzado en §IV-C con la que es, a mi juicio, la frase más importante del survey:

> "With the possible exception of speech recognition, in industry, for the most widespread languages, **all tasks in all audio domains face relatively small datasets**, posing a limit on the size and complexity of deep learning models trained on them."

Nótese lo cuidadosamente calificado: la única excepción es ASR, **en industria**, **para los idiomas más difundidos**. Todo lo demás —música, sonido ambiental, ASR académico, ASR de idiomas de bajos recursos— vive en escasez.

### 8.2. La discusión sobre transferencia (§II-D y §IV-C)

El survey plantea el problema por analogía y luego lo deja abierto:

> "In computer vision, a shortage of labeled data for a particular task is offset by the widespread availability of models trained on ImageNet: to distinguish a thousand object categories, these models learned transformations of the raw input images that **form a good starting point for many other vision tasks**. Similarly, in neural language processing, **word prediction models trained on large text corpora** have shown to yield good model initializations for other language processing tasks [148], [149]. **However, no comparable task and dataset — and models pretrained on it — exists for the audio domain.**"

Las referencias [148] y [149] son **ELMo** (Peters et al., NAACL 2018) y **BERT** (Devlin et al., 2018). O sea: el survey **ve** el paradigma de preentrenamiento auto-supervisado funcionando en NLP en tiempo real, y lo señala como lo que le falta al audio.

Las preguntas de investigación que enuncia, literalmente:

1. "What would be an equivalent task for the audio domain?"
2. "Can there be an audio dataset covering speech, music, and environmental sounds, used for transfer learning, solving a great range of audio classification problems?"
3. "How may pre-trained audio recognition models be flexibly adapted to new tasks using a minimal amount of data, i.e. to out-of-vocabulary words, new languages, new musical styles and new acoustic environments?"

Con la reserva escéptica: "It is well possible that this has to be answered **separately for each domain**, rather than across audio domains. Even just within the music domain, while transfer learning might work for global labels like artists and genres, individual tasks like **harmony detection or downbeat detection might be too different** to transfer from one to another."

Y el plan B: "**If transfer learning turns out to be the wrong direction for audio**, research needs to explore other paradigms for learning more complex models from scarce labeled data, such as **semi-supervised learning, active learning, or few-shot learning**."

**Punto de fidelidad importante:** el survey **no menciona el aprendizaje auto-supervisado (self-supervised learning) por su nombre** en ningún lugar. Lista semi-supervisado, activo y few-shot. Que la respuesta histórica haya sido SSL es un dato que el survey no anticipa con ese nombre, aunque su referencia a ELMo/BERT apunta funcionalmente en esa dirección.

### 8.3. Las métricas (§II-E), qué miden y cuándo engañan

El survey da un párrafo compacto. Lo desarrollo con lo que un practicante necesita saber.

**WER (Word Error Rate)** — ASR. Definición del survey: "counts the fraction of word errors after aligning the reference and hypothesis word strings and consists of **insertion, deletion and substitution rates** which are the number of insertions, deletions and substitutions **divided by the number of reference words**."

$$\text{WER} = \frac{S + D + I}{N}$$

con $N$ el número de palabras de la referencia. Alineamiento por distancia de edición (Levenshtein).

*Cuándo engaña:* (a) **no está acotada por 1** — como $I$ no está limitado por $N$, un modelo que alucina puede tener WER > 100%; (b) trata todos los errores como iguales: confundir "no" por "know" cuenta lo mismo que perder una negación que invierte el sentido clínico de una frase; (c) es sensible a normalización de texto (números, puntuación, mayúsculas): dos sistemas pueden diferir 2 puntos de WER solo por cómo escriben "veinte" vs "20"; (d) depende del idioma — para idiomas aglutinantes conviene CER (character error rate).

**Accuracy** — el survey: "Both in music and in acoustic scene classification, **accuracy is a commonly used metric**." *Cuándo engaña:* con clases desbalanceadas. En detección de eventos raros, un modelo que predice "ausente" siempre puede tener 99% de accuracy y ser inútil. Por eso en escena acústica (clases balanceadas por diseño de DCASE) accuracy es razonable, y en detección de eventos no lo es.

**AUROC** — "To evaluate binary classification **without a fixed classification threshold**, the area under the receiver operating characteristic curve is an alternative to accuracy." Mide la probabilidad de que un positivo aleatorio reciba mayor score que un negativo aleatorio. *Cuándo engaña:* es **optimista bajo desbalance severo**. Con 1 positivo por cada 10.000 negativos, un AUROC de 0.99 puede corresponder a una precisión inutilizable, porque el eje de falsos positivos se normaliza por el enorme número de negativos. En ese régimen la métrica correcta es AUPRC / average precision.

**Métricas semánticamente informadas** — "the loss for a chord detection task can be designed to be **smaller if the detected and the actual chord are harmonically closely related**." Es la analogía audio de la jerarquía WordNet en ImageNet: no todos los errores son igual de graves.

**F-score y EER para detección de eventos** — "In event detection, performance is typically measured using **equal error rate or F-score**, where the true positives, false positives and false negatives are calculated **either in fixed-length segments or per event** [84], [85]."

Esa cláusula final es la que importa y hay que desempaquetarla, porque es la fuente número uno de comparaciones inválidas en SED:

- **Segment-based**: se discretiza el tiempo en segmentos fijos (típicamente 1 s o 10 ms) y se compara la actividad de cada clase en cada segmento. Es tolerante a errores de límites.
- **Event-based**: se comparan eventos completos, con una tolerancia de colisión en el onset (típicamente 200 ms) y opcionalmente en el offset. Es mucho más estricto: un evento detectado partido en dos cuenta como un acierto más un falso positivo.

Los mismos sistemas pueden reordenarse completamente entre ambas. La referencia [85] (Mesaros, Heittola & Virtanen, *Applied Sciences* 2016 — Virtanen es coautor del survey) define además el **error rate** de SED en analogía al WER:

$$\text{ER} = \frac{S + D + I}{N}$$

con $N$ el número de eventos de referencia. El survey **no** nombra explícitamente el ER en §II-E, aunque cita el paper que lo define; conviene saber que la métrica oficial de varias tareas DCASE ha sido ER, no F-score.

*Cuándo engañan:* el F-score depende del umbral elegido; el EER solo tiene sentido si el operativo real es simétrico en costos (rara vez lo es: en vigilancia acústica un falso negativo cuesta mucho más que un falso positivo); y ambos, en polifonía, se pueden promediar por clase (macro) o por instancia (micro), con resultados muy distintos cuando las clases están desbalanceadas.

**SDR / SIR / SAR** — separación de fuentes. El survey: "Objective source separation quality is typically measured with metrics such as **signal-to-distortion ratio, signal-to-interference ratio, and signal-to-artifacts ratio** [86]" (Vincent, Gribonval & Févotte, 2006 — el framework `BSS_EVAL`).

La idea del framework: descomponer la señal estimada $\hat{s}$ en cuatro componentes ortogonales por proyección:

$$\hat{s} = s_{\text{target}} + e_{\text{interf}} + e_{\text{noise}} + e_{\text{artif}}$$

y definir razones de energía en dB:

$$\text{SDR} = 10\log_{10}\frac{\|s_{\text{target}}\|^2}{\|e_{\text{interf}} + e_{\text{noise}} + e_{\text{artif}}\|^2}, \quad
\text{SIR} = 10\log_{10}\frac{\|s_{\text{target}}\|^2}{\|e_{\text{interf}}\|^2}, \quad
\text{SAR} = 10\log_{10}\frac{\|s_{\text{target}} + e_{\text{interf}} + e_{\text{noise}}\|^2}{\|e_{\text{artif}}\|^2}$$

- **SIR** mide cuánta **otra fuente** quedó filtrada (fuga).
- **SAR** mide cuánto **artefacto** (distorsión musical, *musical noise*, huecos espectrales) introdujo el algoritmo.
- **SDR** es la métrica global, que combina ambos.

*Cuándo engañan:* (a) hay un **trade-off directo entre SIR y SAR** — una máscara binaria agresiva sube el SIR (elimina la interferencia) y hunde el SAR (deja agujeros audibles); reportar solo SDR oculta de qué lado del trade-off está el sistema; (b) `BSS_EVAL` permite un **filtro de distorsión permitida** al calcular la proyección, lo que hace que la métrica sea invariante a filtrados que el oído sí escucha; (c) **es indefinida o degenerada cuando la fuente está en silencio** — en un fragmento donde el hablante no habla, $\|s_{\text{target}}\|^2 \to 0$ y el SDR se va a $-\infty$, por lo que la evaluación por segmentos con silencios es traicionera; (d) por eso la comunidad migró a **SI-SDR** (scale-invariant SDR, Le Roux et al., ICASSP 2019, contemporáneo del survey), que elimina la invarianza a filtro y deja solo la invarianza a escala. Y (e) **lo que se reporta suele ser SDRi (improvement)**, la diferencia contra la mezcla sin procesar, no el SDR absoluto: confundir ambos hace comparaciones sin sentido.

**MOS (Mean Opinion Score)** — "a **subjective** test for evaluating quality of synthesized audio, in particular speech." Escala típica 1–5, promediada sobre muchos oyentes y muchas muestras. *Cuándo engaña:* **no es comparable entre estudios**. El MOS depende del pool de evaluadores, del set de referencia incluido en el test, de las instrucciones y del hardware de reproducción. Un "MOS 4.2" de un paper y un "MOS 4.1" de otro no se pueden comparar; solo son válidas las comparaciones **dentro del mismo test**, con intervalos de confianza. Y hay efectos de anclaje: incluir grabaciones reales en el test comprime los scores de todo lo demás.

**Turing test** — "asking a human to distinguish between real and synthesized audio examples, **is a hard test for a model**, since passing the Turing test requires that there is **no perceivable difference**."

**Sobre mAP:** la consigna la mencionaba; para el registro, **el survey no menciona mean average precision en ningún lugar**. La mAP se volvió la métrica estándar de AudioSet a partir de los trabajos de Google (Hershey et al. 2017, PANNs, AST), no de este survey. Conviene saberlo porque es la métrica que verás en toda la literatura de audio tagging post-2019: se computa el average precision (área bajo la curva precision-recall) por clase y se promedia sobre las 527 clases de AudioSet. Sobre AudioSet, un mAP de 0.30 ya es un sistema decente y 0.48 es estado del arte, lo que da una idea de lo duro que es el benchmark.

---

## 9. Conclusiones y desafíos abiertos (§IV)

### 9.1. Los desafíos tal como el survey los enumera

**A. Features (§IV-A).** Estado: log-mel domina; raw waveform y espectro complejo lo siguen. Preguntas abiertas, textuales:

1. "Are mel spectrograms indeed the best representation for audio analysis?"
2. "Under what circumstances is it better to use the raw waveform?"
3. "Can we do better by exploring the middle ground, a **spectrogram with learnable hyperparameters**?"
4. "If we learn a representation from the raw waveform, does it still **generalize between tasks or domains**?"

**B. Models (§IV-B).** Nota histórica del survey: en ASR, MIR y sonido ambiental "deep models have replaced **support vector machines** for sequence classification, and **GMM-HMMs** for sequence transduction. In audio enhancement/denoising and source separation, deep learning has solved tasks previously addressed by **non-negative matrix factorization** and **Wiener methods**. In audio synthesis, **concatenative synthesis** has been replaced e.g. by WaveNet, SampleRNN, WaveRNN." Desafío: "**it is an open research question which model is superior in which setting**. From existing literature, this is very hard to answer, since different research groups yield state-of-the-art results with different models."

**C. Data requirements (§IV-C).** Escasez generalizada; ausencia de un "ImageNet del audio" y de modelos preentrenados; las tres preguntas de §8.2; y el plan B (semi-supervisado, activo, few-shot).

**D. Computational complexity (§IV-D).** "State-of-the-art deep neural networks usually require more computation power and more training data" que los enfoques convencionales; CPUs inadecuadas, GPGPU y TPU necesarias. Y: "Applications with strict limits on computational resources, such as **mobile phones or hearing instruments**, require smaller models. While a lot of recent works tackle the simplification, compression or training of neural networks with minimal computational budgets, **it may be worthwhile to explore options for the specific requirements of real-time audio signal processing**."

**E. Interpretability and adaptability (§IV-E).** "The connection between the layer parameters and the actual task is **hard to interpret**." Dos líneas de ataque que cita: relacionar las activaciones de neuronas con la tarea ([16], [151]) e investigar en qué parte de la entrada se basa la predicción ([152], [153] — este último, LIME aplicado a análisis de contenido musical). Y el objetivo: "Further research into understanding how a network or a sub network behaves could help improving the model structure to **address failure cases**."

### 9.2. Qué se resolvió entre 2019 y hoy, y cómo

| Desafío del survey | Estado hoy | Cómo se resolvió |
|---|---|---|
| **Aprender de pocos datos etiquetados** (§IV-C) | **Resuelto**, y no por los caminos que el survey listó | **Self-supervised learning**, no semi-supervisado/activo/few-shot |
| **"No existe un ImageNet del audio" / falta de preentrenados** (§IV-C) | **Resuelto** | AudioSet + modelos preentrenados públicos; VGGish → PANNs → AST → BEATs; wav2vec 2.0 / HuBERT / WavLM para voz |
| **Generación de audio de alta calidad y en tiempo real** (§III-B3, §IV-D) | **Resuelto** | Vocoders GAN, difusión, codecs neuronales y modelos de lenguaje sobre tokens de codec |
| **Modelar contexto largo sin pagar el costo secuencial de la RNN** (§II-C-b, §IV-B) | **Resuelto** | Transformer y Conformer |
| **El problema de la fase en síntesis** (§II-C-f, §IV-A) | **Resuelto en la práctica** | Vocoders neuronales; codecs que operan en el dominio del tiempo |
| **"Qué modelo es superior en qué setting"** (§IV-B) | **Resuelto para ASR, parcial en el resto** | Conformer como respuesta consensuada; en MIR y sonido ambiental sigue habiendo pluralidad |
| **La representación óptima / el "punto medio con hiperparámetros aprendibles"** (§IV-A) | **Parcialmente** | LEAF y front-ends aprendidos existen pero no desplazaron al log-mel; el log-mel sigue siendo el default de facto |
| **Costo computacional y tiempo real / dispositivos** (§IV-D) | **Parcial** | Streaming Conformer, destilación, cuantización, on-device ASR — pero los modelos grandes crecieron más rápido que las optimizaciones |
| **Interpretabilidad** (§IV-E) | **Abierto** | Sigue sin solución satisfactoria; el campo se movió a evaluación de comportamiento y sondas, no a interpretabilidad mecanicista de audio |

**El detalle de los cuatro grandes:**

**(1) SSL resolvió la escasez de etiquetas, por una vía que el survey no nombró.** El survey listó "semi-supervised learning, active learning, or few-shot learning". Lo que ocurrió fue **auto-supervisión sobre audio no etiquetado a escala**:

- **wav2vec 2.0** (Baevski et al., NeurIPS 2020): encoder convolucional sobre **onda cruda** → enmascaramiento de las representaciones latentes → objetivo contrastivo contra latentes cuantizados (Gumbel-softmax sobre un codebook aprendido) → Transformer sobre el contexto. Resultado emblemático: con **10 minutos** de audio etiquetado (48 grabaciones) y 53.000 horas no etiquetadas, alcanza WER competitivo en LibriSpeech. Esto responde de forma frontal a la pregunta 3 del survey ("adapted to new tasks using a minimal amount of data").
- **HuBERT** (Hsu et al., 2021): reemplaza el objetivo contrastivo por **predicción enmascarada de targets discretos** obtenidos por k-means offline sobre MFCC (primera iteración) y luego sobre las propias representaciones del modelo (iteraciones siguientes). Es literalmente BERT sobre audio, con el paso extra de fabricar el vocabulario. Nótese la ironía: el primer paso de HuBERT clusteriza **MFCC**, la feature que el survey declaró en desuso — porque para clustering la decorrelación de la DCT sí ayuda.
- **WavLM** (2021) agrega denoising/mezcla simulada al preentrenamiento y generaliza a tareas no-ASR (diarización, separación).
- **Whisper** (Radford et al., 2022) toma el camino opuesto y también funciona: **680.000 horas de supervisión débil** raspada de la web, encoder-decoder Transformer estándar, log-mel de 80 bandas de entrada. Prueba que la otra salida de la escasez es simplemente conseguir muchos datos ruidosos.

Nótese la simetría con la observación del survey: la respuesta vino de replicar el paradigma de **ELMo/BERT** que el propio survey señaló en §IV-C como lo que le faltaba al audio. El survey vio el hueco con precisión; erró en la lista de candidatos para llenarlo.

**(2) El "ImageNet del audio" existe, y el survey subestimó lo que ya había.** AudioSet, que el survey cita, resultó ser exactamente eso. La cadena de modelos preentrenados:

- **VGGish** (Hershey et al., ICASSP 2017) — CNN entrenada sobre YouTube-100M produciendo embeddings de 128 dimensiones, que se volvió el extractor genérico por defecto. **Es anterior al survey y el survey no lo cita**, mientras afirma que "no comparable task and dataset — and models pretrained on it — exists for the audio domain". Es la afirmación que peor envejeció, y ya era discutible al escribirse. (Ese paper está en esta misma carpeta.)
- **PANNs** (Kong et al., 2020) — CNN14 sobre AudioSet, mAP 0.439, transferencia sistemática a media docena de tareas downstream.
- **AST** (Gong, Chung & Glass, Interspeech 2021) — Transformer puro sobre parches del espectrograma, sin convolución alguna, con inicialización transferida desde **ViT preentrenado en ImageNet**. mAP **0.485** en AudioSet. Es la refutación directa de las tres objeciones del slide sobre Transformers (§10).
- **BEATs** (Chen et al., 2022) — preentrenamiento auto-supervisado con un tokenizador acústico auto-destilado; ~0.486+ mAP.

**(3) Generación de alta calidad y en tiempo real.** El survey dejaba WaveNet como estado del arte con la queja de que "their training is computationally expensive". Lo que pasó: **HiFi-GAN** (2020) y vocoders GAN similares dieron calidad de WaveNet a cientos de veces tiempo real en GPU; **DiffWave/WaveGrad** (2020) trajeron difusión al waveform; **SoundStream** (2021) y **EnCodec** (2022) establecieron los **codecs neuronales** —cuantización residual vectorial que convierte audio en secuencias de tokens discretos—; y sobre esos tokens se montaron modelos de lenguaje: **AudioLM** (2022), **VALL-E** (2023, TTS zero-shot con 3 segundos de referencia), **MusicGen** y **AudioGen** (2023), **Stable Audio** y **AudioLDM** (difusión latente para texto-a-audio). El requisito de "controlabilidad" que el survey listaba se resolvió por una vía que no anticipaba: **condicionamiento por texto libre**.

Y con esto **el problema de la fase quedó resuelto de facto**: nadie invierte espectrogramas con Griffin-Lim en producción. Se predice mel y se sintetiza con vocoder neural, o se trabaja directamente sobre tokens de codec.

**(4) Contexto largo: el Transformer, y su síntesis con la CNN.** El survey identificó el problema exacto (§IV-B): la RNN da contexto ilimitado pero "requires processing the input sequentially, making them slower to train and evaluate on modern hardware"; la CNN es paralelizable pero tiene campo receptivo fijo. El **Transformer** rompe el dilema: contexto global **y** paralelo. Y el **Conformer** (Gulati et al., Interspeech 2020) hace algo aún más interesante para esta clase: **fusiona convolución y self-attention dentro del mismo bloque**, con módulos feed-forward tipo "macaron" a ambos lados. Resultados en LibriSpeech (Tabla 2 del paper de Conformer, verificado): Conformer(L), 118.8M parámetros, **2.1% / 4.3%** de WER en test-clean/test-other **sin modelo de lenguaje**, y **1.9% / 3.9%** con LM. Conformer(S), con solo 10.3M parámetros, logra 2.7/6.3 sin LM.

**Y este es el punto que cierra el círculo con la clase 39:** el Conformer es *la misma receta del slide* —local + global + clasificación— con **self-attention en lugar de la RNN**. La tesis de complementariedad del profesor no envejeció; lo que envejeció es el operador elegido para la parte global. El ablation del propio paper de Conformer lo confirma: quitar el bloque convolucional degrada de 2.1/4.3 a 2.1/4.9 (sin LM), o sea la convolución **sigue aportando** aun teniendo self-attention. No es que el Transformer haya reemplazado a la CNN en audio; es que reemplazó a la RNN.

**Lo que sigue abierto:**

- **Interpretabilidad (§IV-E)** — sin avances estructurales. Es el desafío que menos se movió.
- **La representación óptima (§IV-A)** — el log-mel sigue ganando por inercia y por eficiencia. Los front-ends aprendidos (SincNet, LEAF) funcionan pero no han desplazado al default. La pregunta 3 del survey ("un espectrograma con hiperparámetros aprendibles") tuvo respuestas concretas y ninguna se impuso. La pregunta 4 ("¿generaliza entre tareas y dominios una representación aprendida de la onda cruda?") sí se respondió afirmativamente para voz, vía wav2vec 2.0.
- **Cómputo y tiempo real en dispositivos con restricciones (§IV-D)** — la brecha entre el estado del arte y lo que corre en un audífono o un teléfono es hoy **mayor** que en 2019, no menor, aunque hay más herramientas para cerrarla.
- **Datos para música y sonido ambiental** — la escasez de anotaciones fuertes en MIR (acordes, downbeats) y en SED sigue vigente; la comunidad DCASE sigue dependiendo de datos sintéticos y débilmente etiquetados.
- **Separación de fuentes en música con dependencias fuertes** — el punto que el survey identificó como estructuralmente más difícil sigue siéndolo, aunque Demucs y sus sucesores mejoraron mucho.

**Una que envejeció mal y hay que marcar:** el survey afirma en §III-B1 que "State-of-the-art source separation methods typically **take the route of estimating masking operations in the time-frequency domain**". **Conv-TasNet** (Luo & Mesgarani, IEEE/ACM TASLP 2019, arXiv 2018) invirtió eso en cuestión de meses: encoder-decoder convolucional aprendido en el **dominio del tiempo**, con TCN dilatada, superando las máscaras oracle ideales de tiempo-frecuencia. El survey se cubre parcialmente ("even though there are approaches that operate directly on time-domain signals and use a DNN to learn a suitable representation from it, see e.g. [119]"), pero el peso de la afirmación quedó del lado equivocado.

---

## 10. Lo que el survey decía sobre atención y Transformers

Esta sección requiere el máximo rigor, porque la tentación de leer el survey a la luz de lo que vino después es fuerte. **Auditoría exhaustiva de todas las menciones a atención y a arquitecturas de secuencia en el texto:**

| # | Ubicación | Texto |
|---|---|---|
| 1 | §II-C-c (seq2seq) | "**Attention-based models which learn alignments between the input and output sequences jointly with the target optimization have become increasingly popular** [43], [52], [53]." |
| 2 | §II-C-c | "Among various sequence-to-sequence models, **listen, attend and spell (LAS) offered improvements over others** [54] (see also Fig. 2)." |
| 3 | Fig. 2, panel E + leyenda | "**Attention [52] can be used for sequence transduction.** Encoder and decoder of the network include a recurrent layer respectively as an embedding $h_e$ of the input $x$ and an embedding $h_d$ of output $y$. The context $c_t$ is a weighted sum of the encoder embedding $h_{e,t-2}, h_{e,t-1}, h_{e,t}, h_{e,t+1}$, where the weights are calculated between the decoder embedding $h_{d,t-1}$ and all encoder embeddings respectively… The output $y_t$ is calculated from the previous output $y_{t-1}$, the previous decoder embedding $h_{d,t-1}$ and the context $c_t$, **indicating correlations between input and output positions**." |
| 4 | §III-A1 (Speech) | "The LAS model is a single neural network that includes an encoder which is analogous to a conventional acoustic model, **an attention module that acts as an alignment model**, and a decoder that is analogous to the language model in a conventional system. **Despite the architectural simplicity and empirical performance of such sequence-to-sequence models**, further improvements in both model structure and optimization process have been proposed to outperform conventional models [94]." |
| 5 | Referencias | **[53] = Vaswani et al., "Attention is all you need", NIPS 2017.** Citada **una sola vez**, dentro de la lista `[43], [52], [53]` de la mención #1. |

**Hechos verificados sobre lo que NO está:**

- La palabra **"Transformer" no aparece nunca en el cuerpo del texto** — solo dentro del título de la referencia [149] (BERT).
- La expresión **"self-attention" no aparece nunca**.
- En §IV-B, el inventario de modelos que "are employed successfully, with no clear preference" es **CNN, RNN y CRNN**. Los modelos de atención no están en esa lista.
- El survey **no expresa escepticismo alguno** hacia la atención. No dice que sea inviable, ni que le falten datos, ni que no modele dependencias largas.

**Veredicto honesto: el survey de 2019 no era escéptico, pero tampoco anticipaba el Transformer.** Su posición es precisa y limitada: **la atención, entendida como mecanismo de alineamiento dentro de un encoder-decoder recurrente (estilo Bahdanau/LAS), es una técnica exitosa y en ascenso para transducción de secuencias en ASR.** Nada más y nada menos. Fíjate en la Fig. 2E: el encoder y el decoder son **capas recurrentes**; la atención es el puente entre ellos. Eso es 2015-2016, no 2017. El survey cita a Vaswani pero no lo discute, no describe self-attention, y no contempla eliminar la recurrencia.

**El contraste con el slide de la clase.** El deck (lámina "Audio and Transformers") dice:

> "There have been previous works on audio applications using a Transformer architecture. However, there are 3 relevant problems: (1) In the context of audio, there is still a lack of highly massive audio datasets. (2) Self-attention mechanism operates over a finite sequence of discrete entities. In the context of text, sentence segmentation is trivial, but for audio this is not the case. (3) Transformers are not good to model long dependencies in sequences. As a consequence, **Transformers are not currently very popular for audio applications**."

**Ninguna de esas tres afirmaciones proviene del survey.** El survey no dice nada parecido. Es una posición del profesor, no del paper. Y las tres son cuestionables incluso en 2019, y claramente falsas en 2024:

1. **"Falta de datasets masivos."** Es una extrapolación razonable de §IV-C del survey ("all tasks in all audio domains face relatively small datasets"). Pero el propio survey cita AudioSet con más de 2 millones de snippets, y la respuesta histórica fue precisamente que **no hacen falta etiquetas**: wav2vec 2.0 usa 53.000 horas **sin etiquetar**, Whisper usa 680.000 horas con supervisión débil, LibriLight ofrece 60.000 horas de audiolibros sin transcribir. El audio no etiquetado es de lo más abundante que existe. La objeción confunde "datasets etiquetados" con "datos".

2. **"Self-attention opera sobre una secuencia finita de entidades discretas y el audio no se segmenta trivialmente."** Es la objeción más interesante y la que más claramente fue respondida: **AST** (2021) simplemente parte el espectrograma en **parches de 16×16 con solape**, exactamente como ViT hace con imágenes, y no necesita ninguna segmentación semántica. **wav2vec 2.0** cuantiza latentes aprendidos en un codebook; **HuBERT** clusteriza con k-means; **EnCodec/SoundStream** producen tokens por cuantización vectorial residual. Hay al menos cuatro respuestas distintas y todas funcionan. La premisa de que las "entidades discretas" deban ser semánticamente significativas era falsa: los parches de ViT tampoco son objetos.

3. **"Los Transformers no son buenos para modelar dependencias largas."** Esta es la afirmación más difícil de sostener: es exactamente **lo contrario** de la propiedad definitoria del Transformer. Self-attention conecta cualquier par de posiciones en **un solo salto** ($O(1)$ de longitud de camino), frente a $O(n)$ en una RNN y $O(\log n)$ con convoluciones dilatadas. Lo que el Transformer sí tiene es un **costo cuadrático** $O(n^2)$ en memoria y cómputo, que en audio duele porque las secuencias son larguísimas —y ese es el problema real que motivó el subsampling convolucional de los encoders de ASR, las atenciones locales y las variantes eficientes. Es probable que el slide esté conflacionando "costoso para secuencias largas" con "malo para dependencias largas". Son cosas opuestas.

**Resumiendo el contraste en una tabla:**

| Afirmación | Survey (2019) | Slide de la clase | Realidad post-2020 |
|---|---|---|---|
| La atención como mecanismo | "increasingly popular" para transducción; LAS "offered improvements" | Se discute solo el Transformer completo | Ubicua |
| El Transformer como arquitectura | Citado ([53]) sin discutirlo; ausente del inventario de §IV-B | "not currently very popular for audio" | Arquitectura dominante (Conformer, AST, Whisper, wav2vec 2.0) |
| Datos | "all tasks face relatively small datasets"; propone semi-supervisado/activo/few-shot | Falta de datasets masivos como obstáculo | Resuelto por SSL sobre audio no etiquetado |
| Costo secuencial de la RNN | Identificado explícitamente como desventaja en hardware moderno (§IV-B) | No se menciona | Es la razón por la que ganó el Transformer |
| Dependencias largas | RNN "unlimited context but first need to learn to do so"; CNN dilatada como alternativa | "Transformers are not good to model long dependencies" | Self-attention es *la* solución a dependencias largas; su problema es el costo $O(n^2)$ |

**Conclusión de fidelidad:** el survey **no** era escéptico sobre la atención; era **agnóstico sobre el Transformer**, que en su horizonte todavía era un modelo de traducción automática que aún no había cruzado al audio. Y —esto es lo notable— el survey **nombró la enfermedad sin recetar la cura**: en §IV-B dice que las RNN "require processing the input sequentially, making them slower to train and evaluate on modern hardware than CNNs", y en §II-C-b dice que las TF-LSTM ganan en precisión pero "are less parallelizable and therefore slower". Ese es exactamente el argumento de "Attention is all you need". El survey tenía el diagnóstico completo en la mano y no dio el salto. Es un excelente caso de estudio de cómo se ve un campo justo antes de un cambio de paradigma.

---

## 11. Limitaciones del survey

**1. Fecha de corte.** Manuscrito recibido el **11 de octubre de 2018**. Casi todo lo que un lector de hoy echa en falta cae después: SpecAugment (abr 2019), Conv-TasNet (versión de journal, 2019), wav2vec (2019) y wav2vec 2.0 (2020), Conformer (2020), HiFi-GAN (2020), AST (2021), HuBERT (2021), codecs neuronales (2021-2022), Whisper (2022). No es una crítica: es un requisito de lectura.

**2. Sesgo hacia el ASR y hacia Google.** Tres de los seis autores (Bo Li, Shuo-yiin Chang, Tara Sainath) están en Google y trabajan en reconocimiento de voz. Se nota:

- La sección §III-A1 (Speech) es la más desarrollada y la única con un arco histórico completo.
- La sección §II-C-c (sequence-to-sequence) está escrita **enteramente desde ASR**: CTC, RNN-T, LAS. Ni una palabra sobre seq2seq en música o en sonido ambiental.
- Las referencias con autores del survey son numerosas: Sainath aparece como autor en [18], [24], [39], [41], [75], [79], [91], [93], [94], [95], [131], [143]; Bo Li en [59], [75], [95], [131], [135], [143]; Chang en [95], [143]; Virtanen en [19], [85], [112], [113], [118]; Schlüter en [16], [81], [105], [152]; Purwins en [13].
- La ilustración de madurez industrial cita "Google Home, Amazon Alexa and Microsoft Cortana", y el ejemplo de simulación de sala es literalmente el paper de generación de utterances para Google Home [79].

Esto no invalida el contenido —los autores de Google escriben sobre lo que mejor conocen y lo hacen bien—, pero sí significa que **el lector debe calibrar**: la sección de voz es un estado del arte; las de música y sonido ambiental son buenos mapas con menos profundidad; y el balance ASR/MIR/ESC del survey no refleja el balance de importancia de los tres campos, sino la composición del equipo autoral.

**3. Cobertura muy delgada de datasets.** Siete datasets nombrados en total. Para un survey que declara la escasez de datos como el desafío central, es poco. No aparecen ESC-50, UrbanSound8K, FSD50K (posterior), GTZAN, MagnaTagATune, LibriSpeech, TIMIT, VoxCeleb, Common Voice, Freesound. Tampoco `Scaper`. Quien busque la referencia de datasets debe ir a la clase 37, no aquí.

**4. Ausencia de tablas comparativas y de números.** El survey nunca compara dos métodos con cifras. Eso lo hace envejecer bien como mapa conceptual y mal como referencia de estado del arte. También significa que es **imposible verificar sus afirmaciones comparativas** desde el propio texto: cuando dice "TF-LSTMs outperform CNNs on certain tasks", hay que ir a [39].

**5. Poca cobertura de multimodalidad y de speech translation.** El survey menciona speech translation en una línea [98] y no toca audio-visual (lip reading, audio-visual source separation), que ya era un campo activo en 2018 (Looking to Listen, 2018).

**6. Sin discusión de aspectos de despliegue.** Nada sobre streaming vs offline (más allá de mencionar RNN bidireccionales para "offline applications"), latencia, endpointing (una cita), o la ingeniería de un sistema real. Es un survey de investigación.

**7. Sin discusión ética.** Nada sobre sesgos de reconocimiento por acento, dialecto o género —un tema documentado en ASR ya para 2018—, ni sobre privacidad de la captura de audio siempre-encendida, ni sobre deepfakes de voz (que el survey tiene a mano al describir el "test de Turing" como criterio de éxito de la síntesis, sin comentar la implicancia).

**8. La afirmación sobre modelos preentrenados era discutible al escribirse.** Ya comentada en §9.2: VGGish existía desde 2017 y no se cita.

---

## 12. Conexión con la clase 39

La clase 39, "DL Models for Audio Processing" (Gabriel Sepúlveda, DCC PUC), lista este survey en sus referencias. **La lectura del deck junto al paper confirma que el survey es su fuente estructural**: hay al menos dos pasajes que están parafraseados casi palabra por palabra, y la organización general de la clase sigue la del survey.

### 12.1. Tabla de correspondencia bloque por bloque

| Bloque de la clase | Sección del survey que lo respalda | Relación |
|---|---|---|
| **"Audio Main Applications": general sounds / speech / music** | §III-A1 (Speech), §III-A2 (Music), §III-A3 (Environmental Sounds) | **Reproduce** la partición tripartita, incluso el orden de importancia. El survey la justifica ("side-by-side, in order to point out similarities and differences"); la clase la presenta como natural, sin argumento |
| **"General Sounds — Sound enhancement": noise reduction, reconstruction, source separation, source transformation** | §III-B1 (Source Separation), §III-B2 (Audio Enhancement) | **Reproduce** el listado; **simplifica** al omitir la formulación de máscaras T-F y las tres razones de trabajar en frecuencia |
| **"General Sounds — Synthesis"** | §III-B3 (Generative Models) | **Reproduce** el listado; **omite** los cuatro requisitos (similaridad, originalidad, diversidad, controlabilidad) y la evaluación de generativos |
| **"Classification: single/multiple × global/local"** | §II-A + **Fig. 1** | **Reproduce** el cruce de los dos ejes; **simplifica**: pierde el tercer tipo de etiqueta (valor numérico → regresión) y el tercer número de etiquetas (*sequence transduction*). **No usa la nomenclatura estándar** (weak/strong labeling, tagging/SED) — ver §6 |
| **"Event detection"** (lámina propia) | §II-A (*event detection* como sequence labeling binario) + §III-A3(b) | **Reproduce**; **omite** el punto crítico de la resolución temporal de salida vs. pooling, y el truco de difuminar targets |
| **"Audio vs Image Data": el espectrograma como imagen** | §I (introducción: ejes no homogéneos) + §II-B (normalización por banda, armónicos, ventana) | **Se queda corto**. El slide dice "there are relevant differences between audio and visual data that is important to consider" y no las enumera. El survey enumera **cuatro**: ejes no homogéneos, orden cronológico obligatorio, distribuciones distintas por banda (→ normalizar por banda), correlaciones armónicas no locales (→ tercera dimensión de armónicos). **El survey es mucho más rico aquí** |
| **"Modeling Options": MLP/CNN/RNN/GAN/Transformer/RL/imitación/neuro-simbólico** | §II-C cubre CNN, RNN, seq2seq, GAN (y VAE en §II-C-e y §III-B3) | **Se aparta**: RL, aprendizaje por imitación y modelos neuro-simbólicos **no aparecen en el survey**. Son agregados del profesor, coherentes con el temario del diplomado (clases 31 y 33), no con la literatura de audio |
| **"For audio, most popular models are a combination of MLP + CNN + RNN. Why? They have complementary properties"** | §II-C-b (CRNN) + §IV-B | **Reproduce el contenido, pero endurece la conclusión.** El survey dice explícitamente "**with no clear preference**" entre CNN, RNN y CRNN, y que en música "neither within nor across tasks is there a consensus". La clase presenta la combinación como *la* respuesta |
| **"CNNs learn local features / RNNs learn global temporal / MLPs classify"** | §II-C-b: "convolutional layers extract local information, and recurrent layers combine it over a longer temporal context" + §IV-B + Mohamed et al. [10] | **Reproduce casi textual**. Pierde cuatro matices: el "no clear preference", que el campo receptivo fijo es también una ventaja, que la RNN *debe aprender* a usar su contexto, y el costo de paralelismo de la recurrencia (§4.4) |
| **"Ejemplo 1": 40D log-mel + 2 conv + LSTM + FC** | §III-A1 (descripción de la CLDNN) + Sainath et al. 2015 [93] | **Reproduce** la CLDNN, que es la referencia [93] del survey y cuya autora es coautora del survey. **Con tres imprecisiones numéricas** — ver §13 |
| Barra lateral del diagrama del Ejemplo 1: "chosen experimentally based on validation error / sample frequency and application determine window / fewer parameters for less data / decrease filter size and increase channels for deeper layers" | §II-C-a, párrafo de rules of thumb | **Paráfrasis casi literal** del survey. Es la prueba más directa de que el deck se construyó sobre este paper. **Con una distorsión** — ver §13 |
| **"Can We Use Raw Audio Data": 15-20 kHz (44.1 para música), muchas muestras, filtros enormes o red muy profunda** | §II-C-a (campo receptivo) + §II-B (filtros aprendidos) + §IV-A (veredicto) | **Reproduce** el problema del campo receptivo. **Omite lo esencial**: el problema de la **invarianza de fase** (§II-C-f), el veredicto condicional de §IV-A (log-mel empata en precisión con menos datos; raw/complejo gana cuando hay que reconstruir fase), y la evidencia empírica de Dieleman vs. Lee. Las cifras de sample rate son del profesor: el survey **no da tasas de muestreo** |
| **"Dilated Convolution"** | §II-C-a | **Paráfrasis casi literal**: "reaching a sufficient receptive field size leads to a large number of parameters and a high computational complexity" ← "may result in a large number of parameters of the CNN and high computational complexity"; "enable CNNs to have very large receptive fields with just a few layers depth" ← "enables networks to obtain very large receptive fields with just a few layers". **Omite** "while preserving the input resolution", que es la propiedad clave (§4.2) |
| **"Ejemplo 2": raw audio + 4 conv dilatadas + 2 LSTM + 2 FC** | Sin correspondencia directa | **Se aparta**: no corresponde a ninguna arquitectura concreta del survey. No es WaveNet (que no tiene LSTM ni MLP), no es Dai et al. 2017 (cuyas capas son [80/4,256] → [3,256]…), no es Lee et al. [111]. Es una **receta genérica compuesta** por el profesor, consistente con el espíritu del survey pero sin fuente |
| **"Audio and Transformers": tres problemas, "not currently very popular"** | Sin correspondencia — el survey no dice nada de esto | **Se aparta frontalmente.** Ver §10. El survey trata la atención como "increasingly popular" y cita a Vaswani sin objeciones |
| **"Data Augmentation": modificar pitch, agregar ruido, time stretching; preentrenar en idiomas ricos y adaptar; síntesis con caveat de datos reales** | §II-D | **Paráfrasis casi literal** de tres frases del survey (la definición, el punto de transfer learning entre idiomas [75], [76], y el caveat "performance on real data may be poor if trained on generated data only"). **Omite** cuatro técnicas del catálogo: simulación de sala/reverberación [79], filtrado espectral [81][82], mezcla de fuentes para separación, y combinación lineal de ejemplos con sus etiquetas [83]. **Agrega** "add noise", que el survey no lista como tal. Ver §7 |

### 12.2. Qué leer del survey para profundizar cada bloque

| Si quieres profundizar… | Lee del survey | Y complementa con |
|---|---|---|
| La taxonomía de aplicaciones | §II-A completa + **Fig. 1** | Buscar `audio tagging` / `sound event detection` / `weak labels` / `multiple instance learning` |
| El espectrograma como imagen y sus trampas | §I (últimas 8 líneas) + §II-B completa | Gong et al., AST 2021 — el caso extremo de tratar el espectrograma como imagen (parches ViT) |
| Por qué log-mel y no MFCC | §II-B, párrafo 2 + §IV-A completa | Cualquier tutorial de GMM-HMM para entender de dónde viene la DCT |
| La receta CNN+RNN+MLP | §II-C-b (CRNN) + **§IV-B completa** — es la sección más importante y la que el slide más suaviza | Gulati et al., Conformer 2020 — la misma receta con self-attention |
| El "Ejemplo 1" | §III-A1, párrafo de la CLDNN | Sainath, Vinyals, Senior & Sak, ICASSP 2015 (está en esta carpeta) |
| Raw audio y dilatación | §II-C-a (campo receptivo y dilatación) + **§II-C-f (fase)** + §IV-A (veredicto) + §III-A2 (Dieleman vs. Lee) | van den Oord et al., WaveNet 2016 (en esta carpeta); Dai et al. 2017 (en esta carpeta) |
| El "Ejemplo 2" | §II-C-a + §III-A3 (dilatación para resolución de salida) | Luo & Mesgarani, Conv-TasNet 2019, para ver la TCN dilatada bien hecha |
| Transformers | §II-C-c + **Fig. 2E** — y constatar que el survey no dice lo que el slide dice | Vaswani et al. 2017; Gulati et al. 2020; Gong et al. 2021; Baevski et al. 2020 (todos, salvo Vaswani, en esta carpeta) |
| Data augmentation | §II-D completa + §III-A3 (última frase) | Park et al., SpecAugment 2019; Salamon et al., Scaper 2017 (en esta carpeta) |
| Evaluación | §II-E + §III-B3 (evaluación de generativos) | Mesaros, Heittola & Virtanen 2016 para SED; Le Roux et al. 2019 para SI-SDR |
| Los desafíos abiertos | **§IV completa** — 3 páginas, es lo mejor del paper | §9.2 de este documento |

**Recomendación de lectura mínima si el tiempo es escaso:** §II-A, §II-B y §IV completas. Son unas 5 páginas y contienen el 80% del valor duradero del survey. §III es un catálogo de referencias útil para localizar literatura por tarea, pero envejeció más.

---

## 13. Erratas, matices y cosas que se citan mal

**(a) El survey no menciona "multiple instance learning" ni "weak/strong labeling" como terminología.** Verificado con doble extracción. Trata la sustancia (Fig. 1, §III-A3), pero si alguien afirma que "el survey desarrolla el multiple instance learning", está atribuyéndole algo que no dice. Ver §6.

**(b) El survey no tiene tablas ni reporta cifras de desempeño.** Cualquier "según Purwins et al., el WER de X es Y" es una invención. Las únicas cifras del paper son: 14M imágenes de ImageNet, >2M snippets de AudioSet, 256 clases de la cuantización a 8 bits, y los tamaños de excerpts/campos receptivos de los trabajos de música (200 ms, 15 frames, 3 s, 29 s, 12 s, 60 s).

**(c) Purwins no figura en Sonos en este paper.** La afiliación impresa es Aalborg University Copenhagen. Es un error frecuente porque Purwins trabajó en Sonos en otro período.

**(d) Discrepancia interna sobre la fecha del perceptrón.** El cuerpo dice "the perceptron algorithm [1] in **1957**"; la referencia [1] es Rosenblatt, *Psychological Review*, vol. 65, no. 6, p. 386, **1958**. Ambas fechas circulan (el reporte técnico del Cornell Aeronautical Laboratory es de 1957, el artículo publicado de 1958). Trivial, pero es la clase de cosa que un lector cuidadoso nota.

**(e) El slide del "Ejemplo 1" tiene tres imprecisiones respecto de la CLDNN real.** Cotejando el deck con Sainath et al. (ICASSP 2015), que es la referencia [93] del survey:

| El slide dice | El paper CLDNN dice |
|---|---|
| "9x9 and **4x4** filter sizes" | "a 9x9 frequency-time filter for the first convolutional layer, followed by a **4x3** filter for the second" |
| "Add **1x1 convolution** to reduce dimension. This allows for a reduction in parameters with no loss in accuracy" | "we add a **linear layer** to reduce feature dimension… adding this linear layer after the CNN layers allows for a reduction in parameters with no loss in accuracy" — es una **capa lineal densa**, no una convolución 1×1. La frase "reduction in parameters with no loss in accuracy" es literal del paper |
| "2 LSTM layers. **Cells in LSTMs with 256D**" | "2 LSTM layers, where each LSTM layer has **832 cells**, and a **512 unit projection layer**". El **256** del slide corresponde a la salida de la **capa lineal de reducción**, no al tamaño de las celdas LSTM |
| "40D Log-mel feats for overlapped segments of **10-20ms**, **5-10ms** overlap" | El paper especifica "40-dimensional log-mel filterbank features, **computed every 10ms**" y no da el largo de ventana explícitamente. La configuración estándar de Google en esa época es ventana de **25 ms** con hop de **10 ms**, o sea 15 ms de solape. Los números del slide son una generalización razonable pero no son los de la CLDNN |
| "2 FC layers. Each FC layer has 1.024 hidden units" | "we pass the output of the LSTM to **a few** fully connected DNN layers… Each fully connected layer has **1,024** hidden units". El 1.024 es correcto; el número de capas es variable en el paper (sus experimentos muestran saturación después de dos capas adicionales) |

Además, un matiz que viene del survey y no del paper: el survey describe la CLDNN como "two convolutional layers **with max-pooling layers**" (plural), mientras el paper original aplica pooling **solo en la primera capa** ("A pooling size of 3 was used for the first layer, and **no pooling was done in the second layer**"). El slide, que dice "Optional max-pooling in frequency only. Ex. Non-overlapped windows of size 3", está más cerca del paper que el survey.

**(f) La barra lateral del diagrama del Ejemplo 1 distorsiona sutilmente el rule of thumb del survey.** El survey dice: "**increasing channel numbers with decreasing sizes of feature maps** in subsequent convolutional layers". El slide dice: "**Decrease filter size** and increase number of channels for deeper layers". No son lo mismo: *feature map* es el **mapa de activaciones** (cuya resolución baja por el pooling/stride), no el **filtro**. La regla real es la de VGG/Inception: *a medida que el pooling reduce la resolución espacial, sube el número de canales para mantener aproximadamente constante la capacidad por capa*. Que además convenga reducir el tamaño del kernel en capas profundas es cierto empíricamente y la propia CLDNN lo hace (9×9 → 4×3), pero es otra afirmación. El slide fusionó dos reglas en una.

**(g) El slide de dilatación omite "preserving the input resolution".** El survey dice que la dilatación amplía el campo receptivo "while preserving the input resolution **as well as** computational efficiency". El slide se queda solo con la eficiencia. La preservación de resolución es la propiedad que hace a la dilatación indispensable en detección de eventos y en síntesis (§4.2), y el survey la menciona **dos veces** (§II-C-a y §III-A3).

**(h) El slide agrega "add noise" al catálogo de augmentation del survey.** Técnica correcta y estándar, pero no está en §II-D. Lo que sí está y el slide omite es lo más valioso: simulación de sala/reverberación, filtrado espectral, mezcla de pistas para separación, y combinación lineal de ejemplos con sus etiquetas.

**(i) Las tres objeciones del slide sobre Transformers no vienen del survey.** Ver §10, con el detalle de por qué las tres son cuestionables. En particular la tercera ("Transformers are not good to model long dependencies") es lo contrario de la propiedad definitoria de self-attention, y probablemente confunde el costo $O(n^2)$ con incapacidad de modelar contexto largo.

**(j) Las tasas de muestreo del slide (15-20 kHz, 44.1 kHz para música) no vienen del survey.** El survey nunca menciona tasas de muestreo concretas. Son datos correctos y estándar (16 kHz para voz —Nyquist a 8 kHz, suficiente para inteligibilidad; 44.1 kHz para música), pero no atribuibles a este paper.

**(k) "Los MFCC están obsoletos" es una simplificación del veredicto del survey.** Lo que dice es que la DCT es "unnecessary or unwanted" **con modelos de deep learning**, y que en procesamiento tradicional los MFCC siguen siendo la representación más común. Los MFCC siguen vivos y siendo útiles donde su decorrelación importa: modelos con supuestos de independencia, clustering (HuBERT los usa como target de la primera iteración de k-means), features de baja dimensión para clasificadores clásicos, y sistemas embebidos con presupuesto mínimo.

**(l) "El survey dice que la onda cruda es mejor" es falso; "el survey dice que log-mel es mejor" también.** El veredicto de §IV-A es **condicional a la tarea**: análisis → log-mel (empata en precisión, con menos datos y menos entrenamiento); síntesis y cualquier cosa que deba reconstruir fase → raw waveform o espectro complejo. Citarlo sin la condición es citarlo mal.

**(m) El survey afirma que "no existe un dataset ni modelos preentrenados comparables a ImageNet para audio" (§IV-C) sin citar VGGish**, que era de 2017 y hacía exactamente eso. La afirmación era ya discutible en 2018 y es claramente falsa hoy. Es la sentencia del survey que peor envejeció, junto con la de que la separación de fuentes se hace "típicamente" con máscaras en tiempo-frecuencia (§9.2).
