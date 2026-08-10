# CNN Architectures for Large-Scale Audio Classification (VGGish / YouTube-100M) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** Shawn Hershey, Sourish Chaudhuri, **Daniel P. W. Ellis**, **Jort F. Gemmeke**, **Aren Jansen**, R. Channing Moore, Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, **Malcolm Slaney**, Ron J. Weiss, Kevin Wilson. Todos en **Google, Inc.** (Nueva York y Mountain View).
- **Venue:** ICASSP 2017. Preprint **arXiv:1609.09430v2** (10 de enero de 2017). *Anota bien ese identificador: el notebook del laboratorio lo cita mal, ver Sección 13.*
- **Agradecimientos:** George Toderici y Marvin Ritter (Sección 6). Toderici es coautor de Sports-1M y de "Beyond Short Snippets", los dos papers de video de los que este trabajo toma prestado el esquema de agregación temporal.

Cuatro de los coautores —Ellis, Gemmeke, Jansen, Moore, Plakal— firman también el paper hermano publicado en el **mismo ICASSP 2017**: *Audio Set: An ontology and human-labeled dataset for audio events*. No son dos líneas de investigación paralelas; son las dos mitades de un mismo esfuerzo, y este paper es la mitad de modelado.

**Tesis en una frase:** si se toma el espectrograma log-mel como si fuera una imagen y se le aplican, **casi sin modificar**, las arquitecturas de visión que ya funcionaban en ImageNet (AlexNet, VGG, Inception-V3, ResNet-50), el resultado es excelente; y las representaciones intermedias de esos modelos, entrenados sobre un corpus de escala industrial, transfieren mucho mejor que las features log-mel crudas.

**Cifras ancla (todas verificadas contra el PDF):**

| Resultado | Valor | Ubicación |
|---|---|---|
| Mejor arquitectura (ResNet-50, 5M pasos), sobre 3K etiquetas / 100K videos balanceados | AUC **0.916**, d′ 1.952, mAP **0.182** | Tabla 2 |
| ResNet-50 entrenada mucho más (17M pasos, con reducción de *learning rate*) | AUC **0.926**, d′ **2.041**, mAP **0.212** | Tabla 2, última fila |
| Baseline MLP totalmente conectado | AUC 0.851, d′ 1.471, mAP 0.058 | Tabla 2 |
| AED sobre AudioSet — **entrada log-mel cruda** ($64 \times 20$) | mAP **0.137**, AUC 0.904, d′ 1.846 | Sección 4.4 |
| AED sobre AudioSet — **embeddings** de la mejor ResNet | mAP **0.314**, AUC **0.959**, d′ **2.452** | Sección 4.4 |

**La ganancia por transferir es de $\times 2.29$ en mAP (0.137 → 0.314, +129 % relativo).** Traducido a una métrica más legible, con $\text{EER} = \Phi(-d'/2)$ la tasa de igual error baja de **17.8 % a 11.0 %** (derivación mía a partir de los d′ del paper, no una cifra del texto).

**El resultado más subestimado, y el que más contradice la lectura popular del paper:** la Tabla 4 muestra que **700K videos alcanzan mAP 0.203 y 70M alcanzan 0.206**. Es decir, el 98.5 % del desempeño se obtiene con **1 % de los datos**. El titular "70 millones de videos, 5.24 millones de horas" es real, pero la evidencia interna del propio paper dice que el corpus está enormemente sobredimensionado para esta tarea y esta capacidad de modelo. Ver Sección 7.

---

## 2. Contexto: el problema del dataset en audio en 2016

La primera frase de la introducción es programática: *"Image classification performance has improved greatly with the advent of large datasets such as ImageNet"*. El paper se escribe desde una asimetría concreta y bien documentada.

**El lado de la visión.** Para 2016 ImageNet llevaba siete años disponible: 1.2M imágenes de entrenamiento etiquetadas a mano en 1000 clases para ILSVRC, sobre una base de ~14M imágenes. Esa base es la que hace posible AlexNet, VGG, Inception y ResNet, y —más importante para lo que viene— la que hace posible el **pre-entrenamiento**: en visión, para 2016, ya nadie entrenaba desde cero un modelo para una tarea pequeña.

**El lado del audio.** La Sección 1 enumera los datasets disponibles como quien enumera un problema: **TRECVid**, **ActivityNet**, **Sports-1M**, **TUT/DCASE Acoustic Scenes 2016**. Textualmente: *"which are much smaller than YouTube-100M"*. Para dimensionarlo, DCASE 2016 ASC trabajaba con del orden de mil segmentos de 30 s en 15 clases de escena. No había nada remotamente comparable a ImageNet, y la consecuencia práctica era que la Acoustic Event Detection seguía anclada en **MFCC + GMM / HMM / NMF / SVM** (referencias [8]–[11] del paper), con incursiones recientes de CNN [12] y RNN [13].

**Qué era YouTube-100M.** Sección 2, con todos los números:

| Propiedad | Valor |
|---|---|
| Videos totales | 100 millones |
| Entrenamiento | **70M** |
| Evaluación | 10M |
| Pool de validación | 20M |
| Duración media por video | 4.6 minutos |
| Horas de entrenamiento | **5.24M** (abstract y Sec. 1) / **5.4M** (Sec. 2) — ver la discrepancia en la Sección 13 |
| Vocabulario de etiquetas | **30,871** identificadores del Knowledge Graph |
| Etiquetas por video | ~5 en promedio |
| Ejemplos de 960 ms derivados | ~**20 mil millones** |

La verificación aritmética cierra: $5.24 \times 10^6\,\text{h} \times 3600\,\text{s/h} \div 0.96\,\text{s} \approx 1.96 \times 10^{10}$, los "around 20 billion" de la Sección 4.3.

**Por qué las etiquetas son débiles y ruidosas.** Hay tres capas de degradación, y conviene separarlas porque no son la misma cosa:

1. **Son automáticas, no humanas.** Sección 2: *"The labels are assigned automatically based on a combination of metadata (title, description, comments, etc.), context, and image content for each video."* Nota el detalle: **image content**. Parte de las etiquetas provienen del canal *visual*. El modelo de audio se está entrenando, en parte, para predecir lo que la imagen dice. Eso es un sesgo estructural, no ruido aleatorio.
2. **Son a nivel de video, no de segmento.** Sección 3.1: *"Each frame inherits all the labels of its parent video."* Un video de 4.6 minutos etiquetado "Trumpet" produce ~287 ejemplos de 960 ms, y en la enorme mayoría de ellos no suena una trompeta. Esto es supervisión débil en el sentido técnico de *Multiple Instance Learning*: la etiqueta es una propiedad de la bolsa, no de la instancia.
3. **Muchas no son acústicamente relevantes.** El paper lo dice sin defenderse: *"of the 30K labels, some are clearly acoustically relevant ('Trumpet') and others are less so ('Web Page')"*. Y la Tabla 1 muestra el rango de granularidad y de prior:

   | Prior de la etiqueta | Ejemplos |
   |---|---|
   | 0.1 … 0.2 | Song, Music, Game, Sports, Performance |
   | 0.01 … 0.1 | Singing, Car, Chordophone, Speech |
   | ~$10^{-5}$ | Custom Motorcycle, Retaining Wall |
   | ~$10^{-6}$ | Cormorant, Lecturer |

   No hay jerarquía impuesta: un video etiquetado "Trumpet" suele llevar también "Entertainment", pero nada lo garantiza.

La honestidad del paper en este punto es notable y hay que citarla completa (Sección 1): *"We are not able to quantify how 'weak' the labels are (i.e., what proportion of the segments are uninformative), and for the majority of classes (e.g., 'Computer Hardware', 'Boeing 757', 'Ollie'), it's not clear how to decide relevance."* Y agregan el contrapunto correcto: para clases como "Beach", el **ambiente de fondo es la señal**, así que "uninformativo" no está bien definido.

**La estrategia frente al ruido no es algorítmica, es de escala.** El paper contrasta explícitamente con Kumar y Raj [21], que formalizan el problema como MIL: *"By contrast, we are investigating the limits of training with weak labels for very large datasets. While many of the individual segments will be uninformative about the labels inherited from the parent video, we hope that, given enough training, the net can learn to spot useful cues."* La apuesta es que el ruido de etiqueta que no está correlacionado con el audio actúa como ruido de gradiente y se promedia con suficientes muestras; lo que **sí** está correlacionado (el sesgo del canal visual) no se promedia y queda dentro del modelo.

**La conexión con AudioSet.** AudioSet (Gemmeke et al., ICASSP 2017, referencia [5]) es la respuesta complementaria al mismo problema: en lugar de aceptar el ruido y compensarlo con escala, **paga anotación humana** sobre una ontología de eventos acústicos. Sección 4.4: *"a dataset of over 1 million 10 second excerpts labeled with a vocabulary of acoustic events (whereas not all of the YouTube-100M 30K labels pertain to acoustic events). This comes to about 3000 hours — still only $\approx 0.05\%$ of YouTube-100M."*

La relación entre los dos es exactamente la que uno esperaría de un equipo con presupuesto: **YouTube-100M es el corpus de pre-entrenamiento sucio y masivo; AudioSet es el benchmark limpio y pequeño.** Y este paper es el que demuestra que el primero sirve para el segundo. El detalle que cierra el círculo: los *embeddings* que Google terminó publicando **junto con** AudioSet son los que produce el modelo de este paper. AudioSet no se distribuye como audio (por derechos), se distribuye como embeddings — y esos embeddings son la salida de la red que aquí se describe.

---

## 3. La pregunta del paper

Este no es un paper de arquitectura. Es un paper de **medición**, y conviene ser explícito sobre eso porque cambia cómo se lee cada tabla.

La pregunta declarada en la Sección 1 tiene tres partes:

> *"how popular Deep Neural Network (DNN) architectures compare on video soundtrack classification; how performance varies with different training set and label vocabulary sizes; and whether our trained models can also be useful for AED."*

Y la decisión metodológica clave está en la Sección 1, en una frase que es literalmente el tema de la Clase 39:

> *"Although the distinct meanings of time and frequency axes might argue for audio-specific architectures, this work employs **minimally-altered** image classification networks such as Inception-V3 and ResNet-50."*

**Por qué "no inventar arquitectura" es la elección correcta aquí.** Hay tres razones, y ninguna es pereza:

1. **Aísla la variable de interés.** Si además de escalar los datos se cambiara la arquitectura, no se podría atribuir la mejora a nada en particular. Al congelar la arquitectura en modelos que la comunidad ya calibró exhaustivamente, la única variable libre es la escala. Es un diseño experimental, no un atajo.
2. **Prueba una hipótesis falsable y no trivial.** La hipótesis nula razonable en 2016 era: *"las arquitecturas de visión fracasarán en espectrogramas porque los ejes no son intercambiables"*. Es una predicción concreta con fundamento físico (Sección 9). El paper la somete a prueba de la forma más limpia posible: aplicando las redes tal cual y midiendo. Si hubieran modificado la arquitectura para adaptarla al audio, el resultado no diría nada sobre la hipótesis.
3. **Es la pregunta con mayor valor de opción.** Si la respuesta es sí, entonces todo el aparato de investigación de visión —arquitecturas, inicializaciones, optimizadores, intuiciones— se vuelve importable a audio sin costo. Ese es un multiplicador enorme, y de hecho es lo que ocurrió: cinco años después, AST importa ViT y sus **pesos de ImageNet** directamente (ver Sección 11).

**Lo que hace de esto un experimento controlado.** Los cinco modelos se entrenan con **el mismo dataset (70M videos), el mismo vocabulario (3K etiquetas), la misma entrada ($96 \times 64$ log-mel), la misma pérdida (cross-entropy con sigmoide final), el mismo optimizador (Adam), sin regularización, y se comparan tras el mismo número de pasos (5M mini-batches de 128)**. Sección 4.1 reconoce el punto débil de ese diseño y lo hace explícito: *"Because some networks trained faster than others, comparing after a fixed wall-clock time would give slightly different results but would not change the relative ordering."* Es decir: comparan a **pasos fijos**, no a **cómputo fijo**. La Tabla 2 incluye la columna de horas justamente para que el lector pueda hacer su propia corrección — y la corrección importa: VGG consumió 184 h contra 119 h de ResNet-50 para quedar 0.021 de mAP por debajo.

**Lo único que no controlaron y podían haber controlado:** no hay ninguna arquitectura *específica de audio* en la comparación. No hay CRNN, no hay modelo sobre forma de onda cruda, no hay filtros con receptive field alargado en frecuencia. Así que la conclusión legítima es **"las arquitecturas de visión funcionan bien"**, no **"las arquitecturas de visión son la mejor opción"**. Es una distinción que la literatura posterior borró bastante rápido.

---

## 4. Las arquitecturas comparadas

Todas reciben el mismo tensor de entrada: un parche log-mel de $96 \times 64$ (96 tramas de tiempo × 64 bandas mel), **un solo canal**. Todas terminan en una capa **sigmoide** de 3087 unidades (no softmax, porque cada ejemplo puede llevar múltiples etiquetas — Sección 3.1). Todas usan **batch normalization después de cada capa convolucional**, reemplazando la LRN original donde correspondía. Ninguna usa dropout, weight decay ni ninguna otra regularización: *"In view of the large training set size, we did not use dropout, weight decay, or other common regularization techniques. For the models trained on 7M or more examples, we saw no evidence of overfitting."*

### 4.1. Baseline totalmente conectado (Sección 3.3.1)

Barrido sobre $N \in \{2,3,4,5,6\}$ capas y $M \in \{500, 1000, 2000, 3000, 4000\}$ unidades, ReLU. El ganador: **$N=3$, $M=1000$**, learning rate $3 \times 10^{-5}$, 10 GPUs y 5 parameter servers. **~11.2M pesos y ~11.2M multiplicaciones** (en una red densa ambos números coinciden salvo los sesgos, que es exactamente lo que distingue a un MLP de una CNN: cero reutilización de parámetros).

### 4.2. AlexNet (Sección 3.3.2)

Aquí están las modificaciones más interesantes, porque son las que revelan qué se rompe al cambiar de $224 \times 224 \times 3$ a $96 \times 64 \times 1$:

- **Stride de la primera capa: $4 \to 2 \times 1$.** La AlexNet original arranca con `conv 11×11 stride 4` sobre una entrada de 224 px. Con una entrada de 96 tramas × 64 bandas, un stride de 4 en ambos ejes dejaría un mapa de $24 \times 16$ y colapsaría la resolución de inmediato. El razonamiento del paper es explícito y cuantitativo: *"Because our inputs are $96 \times 64$, we use a stride of $2 \times 1$ so that the number of activations are similar after the initial layer."* Es un stride **anisotrópico**: 2 en tiempo, **1 en frecuencia**. Es decir: aceptan submuestrear el eje temporal pero **no** el de frecuencia. Ese asterisco es la primera admisión implícita de que los ejes no son equivalentes.
- **BatchNorm en vez de LRN.**
- **Capa final: 1000 → 3087 unidades**, con sigmoide.
- **Sin división de filtros entre dispositivos** (el truco de 2012 para caber en dos GTX 580; irrelevante aquí).
- Entrenada con **20 GPUs y 10 parameter servers**.

Costo: la AlexNet original tiene **62.4M pesos y 1.1G multiplicaciones**; la variante de audio, **37.3M pesos y 767M multiplicaciones**. La caída de parámetros viene casi toda de la capa `fc6`, cuyo tensor de entrada es mucho más pequeño.

### 4.3. VGG (Sección 3.3.3)

**El cambio es prácticamente nulo, y esto importa mucho para la Sección 13:**

> *"The only changes we made to VGG (**configuration E**) were to the final layer (3087 units with a sigmoid) as well as the use of batch normalization instead of LRN."*

**Configuration E es VGG-19**: 16 capas convolucionales $3\times3$ en 5 bloques + 3 densas. Original: **144M pesos, 20B multiplicaciones**. Variante de audio: **62M pesos, 2.4B multiplicaciones**.

Vale la pena verificar de dónde salen esos 62M, porque confirma la topología. Con 5 bloques de max-pooling stride 2 sobre $96 \times 64$: $96/32 = 3$, $64/32 = 2$, luego el tensor aplanado es $3 \times 2 \times 512 = 3072$. Entonces las densas aportan $3072 \times 4096 + 4096 \times 4096 + 4096 \times 3087 \approx 12.6 + 16.8 + 12.6 = 42.0$M, y las convolucionales de VGG-19 aportan ~20.0M. Total ~62M. **Cierra exactamente.** (Cálculo mío; el paper solo da el total.)

Detalle experimental que conviene guardar: *"We tried another variant that reduced the initial strides (as we did with AlexNet), but found that not modifying the strides resulted in faster training and better performance."* Y: *"parallelizing beyond 10 GPUs did not help significantly"* → 10 GPUs, 5 parameter servers.

### 4.4. Inception-V3 (Sección 3.3.4)

La cirugía más agresiva:

- **Se eliminan las primeras cuatro capas del *stem*, incluyendo el MaxPool.** El stem de Inception-V3 está diseñado para bajar de $299 \times 299$ a $35 \times 35$; sobre una entrada de $96 \times 64$ ese submuestreo es catastrófico.
- **Se elimina la red auxiliar** (el clasificador intermedio con su pérdida propia).
- **Average Pool final cambiado a $10 \times 6$** para reflejar el tamaño de las activaciones.
- Probaron la alternativa de *conservar* el stem quitando el primer stride 2 y el MaxPool, y **funcionó peor** que truncar el stem.

Costo: original **27M pesos / 5.6B multiplicaciones**; audio **28M pesos / 4.7B multiplicaciones**. Nota la inversión: la variante de audio tiene *más* pesos que la original (por la capa de 3087 salidas) y *menos* cómputo. Entrenada con **40 GPUs y 20 parameter servers** — la más cara del estudio en paralelismo.

### 4.5. ResNet-50 (Sección 3.3.5)

- **Se elimina el stride 2 de la primera convolución $7\times7$**, *"so that the number of activations was not too different in the audio version"*. Nuevamente: preservar resolución.
- **Average Pool final cambiado a $6 \times 4$.**

Costo: original **26M pesos / 3.8B multiplicaciones**; audio **30M pesos / 1.9B multiplicaciones** — la mitad del cómputo con más pesos. **20 GPUs, 10 parameter servers.**

### 4.6. Qué se mantuvo idéntico

Vale la pena listarlo, porque el mensaje del paper vive en esta lista:

- Los tamaños de kernel. Nadie alargó un filtro en frecuencia.
- La topología de bloques (residuales, Inception, apilamiento VGG).
- El esquema de pooling intermedio.
- El número y la disposición de las capas densas.
- La progresión de canales.

**Todo lo que se tocó fue: (i) el stride/pooling de los extremos, para que la resolución no colapse en una entrada 5–10× más chica; (ii) el número de clases de salida; (iii) sigmoide en vez de softmax; (iv) BatchNorm en vez de LRN.** Ninguna de esas cuatro cosas es una adaptación *al audio*: son adaptaciones al **tamaño de la entrada** y al **tipo de problema (multi-etiqueta)**. La adaptación al audio genuina es exactamente cero.

### 4.7. Tabla comparativa consolidada

Tabla 2 del paper (70M videos, 3K etiquetas, evaluada sobre 100K videos balanceados), enriquecida con los costos de las Secciones 3.3.x y con una columna de EER derivada por mí a partir de $\text{EER} = \Phi(-d'/2)$:

| Arquitectura | Pasos | Horas | Pesos (audio) | Multiplicaciones (audio) | Pesos (original) | AUC | d′ | mAP | EER* |
|---|---|---|---|---|---|---|---|---|---|
| Fully Connected (3×1000) | 5M | 35 h | 11.2M | 11.2M | — | 0.851 | 1.471 | 0.058 | 23.1 % |
| AlexNet | 5M | 82 h | 37.3M | 767M | 62.4M | 0.894 | 1.764 | 0.115 | 18.9 % |
| VGG (config. E) | 5M | 184 h | 62M | 2.4B | 144M | 0.911 | 1.909 | 0.161 | 17.0 % |
| Inception-V3 | 5M | 137 h | 28M | 4.7B | 27M | **0.918** | **1.969** | 0.181 | 16.3 % |
| ResNet-50 | 5M | 119 h | 30M | 1.9B | 26M | 0.916 | 1.952 | **0.182** | 16.5 % |
| ResNet-50 (largo) | 17M | 356 h | 30M | 1.9B | 26M | **0.926** | **2.041** | **0.212** | 15.4 % |

\* Columna derivada por mí, no presente en el paper.

**Lecturas que la tabla soporta:**

- **Todas las CNN superan al MLP, y por mucho.** mAP $0.058 \to 0.182$ es un factor $\times 3.1$. La convolución no es un detalle de eficiencia aquí: es la diferencia entre funcionar y no funcionar. La justificación del paper (Sección 4.1) es la esperada: *"their convolutional units can efficiently capture common structures that may occur in different areas of the input array for both images, and, we infer, our audio representation."* Nota el *"we infer"* — no lo demuestran, lo infieren.
- **El ordenamiento replica el de ImageNet.** AlexNet < VGG < Inception ≈ ResNet. Eso es, en sí mismo, el hallazgo: **el ranking arquitectónico de visión se transfiere al audio**. Si el espectrograma fuera un dominio genuinamente ajeno, no habría razón para que el orden se preservara.
- **Los pesos no predicen el desempeño.** VGG tiene **62M** pesos y saca 0.161; ResNet-50 tiene **30M** y saca 0.182 en 65 % del tiempo. La capacidad bruta no es la variable; la topología sí.
- **El cómputo tampoco.** Inception-V3 gasta **4.7B** multiplicaciones contra **1.9B** de ResNet-50 para empatar (0.181 vs 0.182). ResNet-50 es la elección claramente dominante en la frontera de Pareto, y por eso es la que usan en todos los experimentos posteriores.
- **5M pasos no es convergencia.** ResNet-50 pasa de 0.182 a 0.212 (+16 % relativo) con 3.4× más pasos y una reducción de learning rate. Es decir: **toda la Tabla 2 está medida antes de converger**, y los autores lo dicen. Las diferencias entre Inception y ResNet están dentro de ese margen de incertidumbre.

### 4.8. Un resultado lateral que casi nadie cita: la Figura 1

Scatter de d′ por clase (ResNet-50) contra $\log_{10}$ del prior de la clase, sobre un subconjunto aleatorio del 20 % de las 30K etiquetas, con el color codificando el AP. La mediana de d′ (línea roja) **se mantiene plana alrededor de 1.9–2.0 a lo largo de cinco órdenes de magnitud del prior**, de $10^{-6}$ a $10^{-1}$. Lo único que cambia es la **varianza**, que crece para las clases raras.

El paper marca correctamente por qué es raro: *"This is contrary to the usual result where classifier performance improves with increased training data."* Una clase con prior $10^{-6}$ tiene ~20,000 ejemplos positivos en 20 mil millones; una con prior $10^{-1}$ tiene 2 mil millones. Cinco órdenes de magnitud de diferencia en datos por clase, y el mismo d′.

Mi lectura: esto dice que **la capacidad discriminativa está casi enteramente en la representación compartida, no en la cabeza de clasificación**. Una vez que la red aprendió a describir el audio, distinguir "Cormorant" requiere muy pocos ejemplos. Es, de hecho, un argumento a favor de los embeddings antes de que el paper llegue a la Sección 4.4 — y también la razón por la que el mAP global se ve tan bajo: la Sección 3.2 lo explica, *"unlike AUC, [AP] is directly correlated with the prior probability of the class. Because most of our classes have very low priors ($<10^{-4}$), the mAPs we report are typically small, even though the false alarm rates are good."* Un mAP de 0.212 con un prior medio de $10^{-4}$ no es un modelo malo; es un modelo bueno medido con una métrica que castiga los priors bajos.

---

## 5. La representación de entrada

Esta sección es la que más se cita del paper en la práctica, porque define un formato que se volvió un estándar de facto. **Y es también donde hay que ser más cuidadoso con qué dice el paper y qué no.**

### 5.1. Lo que el paper efectivamente especifica (Sección 3.1)

> *"The audio is divided into non-overlapping 960 ms frames. This gave approximately 20 billion examples from the 70M videos. Each frame inherits all the labels of its parent video. The 960 ms frames are decomposed with a short-time Fourier transform applying 25 ms windows every 10 ms. The resulting spectrogram is integrated into 64 mel-spaced frequency bins, and the magnitude of each bin is log-transformed **after adding a small offset to avoid numerical issues**. This gives log-mel spectrogram patches of $96 \times 64$ bins that form the input to all classifiers. During training we fetch mini-batches of 128 input examples by randomly sampling from all patches."*

Eso es **todo** lo que el paper dice sobre el front-end. Nota qué **no** dice:

- **No especifica la frecuencia de muestreo.** El "16 kHz mono" no está en este PDF.
- **No especifica el rango de la escala mel.** El "125 Hz a 7500 Hz" no está en este PDF.
- **No da el valor del offset de estabilización.** Solo dice "a small offset".
- No especifica el tipo de ventana, el tamaño de la FFT, ni si el espectrograma es de magnitud o de potencia.

Todos esos valores provienen del **código publicado** (`vggish_params.py` en el repositorio `tensorflow/models/research/audioset/vggish`), no del paper. Es una distinción que hay que mantener limpia: el paper define la **forma** del tensor; el repositorio define los **valores exactos**.

### 5.2. Los parámetros del pipeline publicado

Con la advertencia anterior en pie, estos son los valores canónicos del preprocesamiento que hoy se llama "el input de VGGish", tal como los fija el código de referencia:

| Parámetro | Valor | Justificación |
|---|---|---|
| Frecuencia de muestreo | 16 kHz, mono | Nyquist a 8 kHz. Estándar de wideband speech. |
| Ventana STFT | 25 ms (400 muestras) | Estándar heredado de ASR. |
| Salto (hop) | 10 ms (160 muestras) | 100 tramas por segundo. Estándar de ASR. |
| Tamaño de FFT | 512 (siguiente potencia de 2 ≥ 400) | → 257 bins de frecuencia. |
| Espectro | **magnitud**, no potencia | El código toma `np.abs(fft)`. |
| Bandas mel | 64 | |
| Rango mel | 125 Hz – 7500 Hz | |
| Compresión | $\log(\text{mel} + 0.01)$ | Offset de estabilización. |
| Ventana de ejemplo | 96 tramas = 960 ms | |
| Salto entre ejemplos | 96 tramas (sin solapamiento) | |

### 5.3. Por qué cada elección, y qué se pierde

**16 kHz.** Descarta todo por sobre 8 kHz. Para voz y para la mayoría de eventos ambientales eso es casi gratis; para música (platillos, aire de instrumentos de viento, brillo percibido) y para clasificación de calidad de grabación es una pérdida real. También es una decisión de costo: a 16 kHz el corpus de 5.24M horas ocupa la mitad que a 32 kHz. Nota que PANNs (2020) entrena a **32 kHz** y reporta que ayuda.

**Ventana de 25 ms, salto de 10 ms.** Es el punto elegido en el compromiso de Gabor: $\Delta t \cdot \Delta f \gtrsim 1/(4\pi)$. Con una ventana de 25 ms la resolución en frecuencia es del orden de $1/0.025 = 40$ Hz. Consecuencia concreta: **una voz masculina con $f_0 \approx 110$ Hz tiene armónicos separados por 110 Hz y se resuelven; una voz femenina o un violín agudo también; pero las diferencias finas de afinación por debajo de 40 Hz de separación se pierden.** En el otro sentido, 25 ms es suficientemente corto para que la señal sea aproximadamente estacionaria dentro de la ventana (la premisa de toda la STFT) pero suficientemente largo para *emborronar* transitorios muy rápidos: un chasquido de 2 ms se distribuye sobre toda la ventana. El solapamiento del 60 % (salto 10 ms sobre ventana 25 ms) es lo que evita que se pierdan eventos entre tramas.

**64 bandas mel entre 125 y 7500 Hz.** Tres decisiones en una:

- **La reducción $257 \to 64$** es un factor 4 de compresión que elimina la estructura armónica fina y conserva la **envolvente espectral** (el timbre). Para clasificación de eventos eso es lo que importa. Para transcripción musical o estimación de pitch, es exactamente lo que se necesitaría preservar, y por eso los modelos de música rara vez usan 64 mel.
- **La escala mel**, $m = 1127 \ln(1 + f/700)$, comprime las altas frecuencias y expande las bajas, imitando la resolución del oído. Consecuencia: las bandas no son equiespaciadas en Hz. Alrededor de 200 Hz una banda cubre decenas de Hz; alrededor de 6 kHz cubre cientos.
- **El recorte inferior en 125 Hz** elimina el DC, la componente de red eléctrica (50/60 Hz) y el ruido de manejo de micrófono, pero también **elimina la fundamental de las voces masculinas graves** (típicamente 85–155 Hz) y buena parte del rango de un bajo eléctrico o un bombo. El modelo tiene que inferir esos objetos a partir de sus armónicos superiores, cosa que hace bien porque el efecto de la "fundamental faltante" es real perceptualmente, pero es una pérdida de información.
- **El recorte superior en 7500 Hz** deja 500 Hz de margen bajo Nyquist para evitar el rolloff del filtro antialiasing.

**El logaritmo con offset.** Dos funciones. Primero, **compresión de rango dinámico**: el rango de energías en audio natural cubre fácilmente 80–100 dB, y una red con activaciones lineales estaría dominada por los frames más fuertes. El log convierte factores multiplicativos en desplazamientos aditivos, que es exactamente el tipo de invarianza que una CNN maneja bien (un cambio de volumen se vuelve un sesgo constante sumado a todo el parche, absorbible por la BatchNorm). Segundo, es el análogo perceptual de la ley de Weber-Fechner: la sonoridad percibida es aproximadamente logarítmica en la intensidad.

El **offset** es indispensable: $\log(0) = -\infty$. Con offset $\epsilon = 0.01$, el piso de la representación queda en $\log(0.01) \approx -4.6$, lo que además define implícitamente un **rango dinámico efectivo**: cualquier energía muy por debajo de 0.01 se aplasta contra ese piso. Es decir, el offset **es** el control de rango dinámico del front-end, disfrazado de truco numérico.

**Lo que el log-mel descarta y no se recupera:**

- **La fase.** La STFT es compleja; el front-end se queda solo con la magnitud. Esto hace la representación no invertible (de ahí Griffin-Lim, WaveNet vocoders, HiFi-GAN). Para clasificación no importa demasiado; para separación de fuentes y síntesis es central.
- **La resolución temporal sub-10 ms.**
- **La resolución en frecuencia sub-banda mel.**
- **Todo lo que está fuera de [125, 7500] Hz.**

**El agrupamiento en 960 ms.** Aquí hay una sutileza aritmética que vale la pena aclarar porque confunde a mucha gente: **96 tramas con salto de 10 ms cubren 960 ms solo si se cuentan los saltos, no las ventanas**. Si tomaras un audio *aislado* de exactamente 0.96 s y lo enmarcaras con ventana 400 / salto 160 sin padding, obtendrías $1 + \lfloor (15360-400)/160 \rfloor = 94$ tramas, no 96. El código funciona al revés: calcula el espectrograma log-mel de **toda** la señal y luego lo corta en bloques de 96 tramas con salto de 96. Sobre una señal larga eso da bloques que consumen $96 \times 160 + 240 = 15600$ muestras ≈ 975 ms de audio con solapamiento de 15 ms entre bloques consecutivos de tramas. La etiqueta "960 ms" refiere al **avance**, no al soporte.

**Por qué 960 ms y no otra cosa.** El paper no lo justifica. Mi lectura: es el número que hace que el parche resultante ($96 \times 64$) tenga un aspect ratio y un tamaño cómodos para una CNN de visión —comparable a las entradas pequeñas de las redes de imágenes— y que además sea aproximadamente la escala temporal de un evento acústico completo (una palabra, un ladrido, un bocinazo). Es la unidad mínima de decisión del sistema, y define su límite de resolución (Sección 10).

**El costo de la elección "non-overlapping".** Sección 3.1: los frames de entrenamiento no se solapan. Eso multiplica por ~1 el número de ejemplos (contra el ~10× que daría un salto de 96 ms) pero garantiza independencia entre ejemplos consecutivos. Con 20 mil millones de ejemplos disponibles, no había ninguna razón para aumentar el dataset con solapamiento.

---

## 6. VGGish como extractor de embeddings

### 6.1. La aclaración terminológica que hay que hacer primero

**La palabra "VGGish" no aparece ni una sola vez en este paper.** Lo verifiqué sobre el texto completo extraído: cero ocurrencias. El nombre lo puso Google después, al liberar el modelo en `tensorflow/models/research/audioset/vggish`.

Y hay una segunda confusión, más grave y muy extendida:

| | "VGG" de la Tabla 2 | "VGGish" publicado |
|---|---|---|
| Configuración VGG | **E** (VGG-19): 16 conv + 3 densas, 5 bloques | Variante de la configuración **A** (VGG-11), **truncada tras el 4º bloque**: 6 conv, 4 bloques |
| Capas de pooling | 5 | 4 |
| Mapa antes de aplanar | $3 \times 2 \times 512 = 3072$ | $6 \times 4 \times 512 = 12288$ |
| Cabeza densa | 4096 → 4096 → **3087** (sigmoide) | 4096 → 4096 → **128** (ReLU) |
| Parámetros | **62M** (dato del paper) | **72.1M** (cálculo mío, Sección 12) |
| Salida | 3087 logits multi-etiqueta | embedding de 128 dimensiones |

Son **arquitecturas distintas**. Citar el AUC 0.911 / mAP 0.161 de la Tabla 2 como "el desempeño de VGGish" es incorrecto. Y más importante todavía:

**Los embeddings del experimento de transferencia de la Sección 4.4 NO son de VGG. Son de ResNet-50.** Textual: *"the second uses the output of the penultimate 'embedding' layer of our **best ResNet model** as inputs"*. La mejor red del paper es ResNet-50, no VGG. Así que la afirmación "los embeddings de VGGish dieron mAP 0.314 en AudioSet" —que circula ampliamente— **atribuye a VGGish un resultado que el paper obtuvo con ResNet-50**. Es el error de citación más frecuente sobre este trabajo (ver Sección 13).

Entonces: ¿qué es VGGish, exactamente? Es el **modelo que Google decidió liberar**: una red de la familia VGG, más chica que la config. E del paper, entrenada con este mismo pipeline y este mismo tipo de corpus, con una cabeza recortada a un cuello de botella de 128 dimensiones. La elección de VGG en vez de ResNet para publicar es razonable desde ingeniería —VGG es trivial de portar entre frameworks, sin bloques residuales ni ramas— pero significa que el checkpoint público **no es** el mejor modelo del paper.

### 6.2. La penúltima capa como embedding, y el respaldo experimental de la Tabla 3

La idea de usar la penúltima capa como representación de propósito general no es original de este paper. Lo que sí es de este paper es la **medición de cuánto cuesta el cuello de botella**, y está en la Tabla 3 (Sección 4.2). Todos los modelos son ResNet-50 entrenadas sobre 70M videos, evaluadas sobre las mismas 400 etiquetas:

| Bottleneck | Etiquetas de entrenamiento | AUC | d′ | mAP |
|---|---|---|---|---|
| no | 30K | — | — | — |
| no | 3K | **0.930** | **2.087** | **0.381** |
| no | 400 | 0.928 | 2.067 | 0.376 |
| sí (128-d) | 30K | 0.925 | 2.035 | 0.369 |
| sí (128-d) | 3K | 0.919 | 1.982 | 0.347 |
| sí (128-d) | 400 | 0.924 | 2.026 | 0.365 |

El cuello de botella de **128 unidades** se inserta justo antes de la capa de salida. Y aquí está el dato clave: el paper **no lo introdujo para producir embeddings**. Lo introdujo por velocidad: *"We introduced the bottleneck layer to speed up the training of the model trained with 30K labels. Without a bottleneck, the larger output layer increased the number of weights from 30M to 80M and significantly reduced training speed."*

La aritmética: la capa de salida sin cuello de botella conecta las **2048** activaciones del Average Pool de ResNet-50 con 30,871 salidas → $2048 \times 30871 \approx 63$M parámetros, que sumados a los ~17M del resto dan los ~80M citados. Con cuello de botella: $2048 \times 128 + 128 \times 30871 \approx 0.26 + 3.95 = 4.2$M. **Un factor 15 de reducción en la capa de salida.** El embedding de 128 dimensiones nació como una optimización de entrenamiento y resultó ser el producto más duradero del paper.

**Cuánto cuesta el cuello de botella.** Comparando filas con las mismas etiquetas: a 400 etiquetas, mAP $0.376 \to 0.365$ (−2.9 % relativo); a 3K, $0.381 \to 0.347$ (−8.9 % relativo). El paper lo reconoce: *"the bottleneck layer is relatively small compared to the 2048 activations coming out of ResNet-50's Average Pool layer and so it is effecting a substantial reduction in information."* Es decir: **comprimir 2048 → 128 cuesta entre 3 % y 9 % de mAP.** Es un precio bajísimo por un factor 16 de compresión, y esa relación es toda la justificación económica del pipeline de embeddings.

### 6.3. PCA y cuantización a 8 bits

Este componente **no está en el paper**. Es parte del pipeline publicado junto con AudioSet, documentado en el repositorio de `audioset` y en el paper de AudioSet [5]. Lo describo como lo que es —el formato de distribución— y marco explícitamente que no lo pude verificar contra este PDF.

El post-procesamiento aplicado a los embeddings antes de publicarlos:

1. **PCA con blanqueo** sobre las 128 dimensiones, ajustada sobre un corpus grande de embeddings. No reduce la dimensionalidad (128 → 128); **decorrelaciona y normaliza la varianza por eje**.
2. **Cuantización uniforme a 8 bits por dimensión**, con recorte previo a un rango fijo.

**Por qué el orden importa.** El embedding crudo sale de una ReLU, así que es no negativo y tiene una distribución muy asimétrica y de varianza muy dispar entre ejes. Cuantizar eso directamente a 8 bits desperdiciaría casi todos los niveles: los ejes de varianza alta se saturarían y los de varianza baja usarían dos o tres niveles. El blanqueo **iguala la varianza de todos los ejes**, y recién entonces una cuantización uniforme de 8 bits es aproximadamente óptima para todos por igual. Es exactamente el mismo razonamiento que hay detrás de la cuantización por canal en inferencia de redes modernas.

**Por qué importa para el almacenamiento.** Los números:

| Formato | Bytes por segundo | AudioSet completo (~2.1M clips × 10 s) |
|---|---|---|
| Audio PCM 16 kHz / 16 bits | 32,000 | ~670 GB |
| Embedding float32 (128-d, 1 Hz) | 512 | ~10.7 GB |
| Embedding uint8 (128-d, 1 Hz) | **128** | **~2.7 GB** |

(Cálculos míos.) La compresión frente al audio crudo es de **250×**, y frente a float32 de **4×**. Ese es el hecho que hace posible distribuir AudioSet: Google **no puede** redistribuir el audio de YouTube por derechos, pero sí puede distribuir un descriptor de 128 bytes por segundo. **El formato del embedding es, literalmente, la razón por la que AudioSet existe como dataset descargable.** Y la contrapartida es la limitación permanente de AudioSet: durante años la comunidad trabajó sobre features precomputados de 1 Hz, sin acceso al audio, lo que impedía cualquier experimentación con el front-end. AST y PANNs solo son posibles cuando la gente empezó a re-descargar el audio por su cuenta.

Una consecuencia técnica que conviene tener presente si trabajas con estos embeddings: **son no negativos (post-ReLU) y están cuantizados**. La distancia coseno entre dos embeddings uint8 sin des-cuantizar no es la misma que entre los float originales, y la no negatividad hace que todos los vectores vivan en el ortante positivo, comprimiendo el rango de ángulos posibles. Si los usas como features para un índice vectorial —algo directamente relevante para el trabajo de *blocking* con bi-encoders— conviene des-cuantizar y, si es posible, revertir el blanqueo antes de calcular similitudes.

### 6.4. Cómo se agregan los embeddings en el tiempo

Sección 3.2: *"We passed each 960 ms frame from each evaluation video through the classifier. We then averaged the classifier output scores across all segments in a video."* Y la justificación en la Sección 1, tomada prestada de video:

> *"We aggregate local classifications to whole-soundtrack decisions by imitating the visual-based video classification of Ng et al. After investigating several more complex models for combining information across time, they found simple averaging of single-frame CNN classification outputs performed nearly as well. By analogy, we apply a classifier to a series of non-overlapping segments, then average all the sets of classifier outputs."*

Es decir: **promedio de scores, no de embeddings, y sin ningún modelo temporal**. Es la decisión más floja del paper y ellos lo saben; la Sección 1 lo dice: *"our labels apply to entire videos without any changes in time, so we have yet to try such recurrent models"*. Volveremos sobre esto en la Sección 10. (Nota metodológica importante: promediar **scores** post-sigmoide no es lo mismo que promediar logits ni que hacer max-pooling. El promedio de scores penaliza los eventos raros y breves: un disparo de 1 s en un video de 4.6 minutos aporta un score alto en 1 de 287 segmentos, y el promedio lo diluye hasta hacerlo indistinguible del ruido. Para detección de eventos esporádicos, max o attention pooling son estrictamente mejores — y es exactamente lo que hicieron PANNs y PSLA años después.)

---

## 7. Los experimentos de escala

Esta es la parte más citada del paper y, paradójicamente, la más malinterpretada. Hay tres ejes, con calidad de evidencia muy desigual.

### 7.1. Eje 1 — cantidad de datos de entrenamiento (Sección 4.3, Tabla 4)

Setup: **ResNet-50**, 3K etiquetas, **16 millones de mini-batches** de 128 (unas 380 horas de entrenamiento) sobre subconjuntos de 70M, 7M, 700K, 70K y 23K videos. Nota que **el número de pasos es constante**; lo que varía es de cuántos videos distintos se muestrean esos pasos. Esto es fundamental y suele pasarse por alto: **no es un experimento de "más datos = más pasos", es un experimento de diversidad a cómputo constante.**

| Videos de entrenamiento | AUC | d′ | mAP | mAP relativo al máximo |
|---|---|---|---|---|
| 70M | **0.923** | **2.019** | **0.206** | 100 % |
| 7M | 0.922 | 2.006 | 0.202 | 98.1 % |
| 700K | 0.921 | 1.997 | 0.203 | 98.5 % |
| 70K | 0.909 | 1.883 | 0.162 | 78.6 % |
| 23K | 0.868 | 1.581 | 0.118 | 57.3 % |

(La última columna es cálculo mío.)

**Esta tabla dice, con claridad, que la escala satura.** De 700K a 70M —un factor **100** de datos— el mAP se mueve de 0.203 a 0.206, es decir **1.5 %**. El AUC se mueve 0.002. Lo que sí importa es el escalón entre 70K y 700K: mAP 0.162 → 0.203, un salto de **25 % relativo** por un factor 10 de datos.

**La conclusión operativa: para esta arquitectura (30M parámetros) y esta tarea, el punto de rendimientos decrecientes está en algún lugar entre 70K y 700K videos** —es decir, entre ~5,400 y ~54,000 horas de audio. Todo lo que hay por encima de eso es margen de seguridad, no ganancia.

**Los dos matices que el paper agrega y que evitan sobreinterpretar la tabla:**

1. **Los modelos chicos sobreajustaron, y eso no se corrigió.** Sección 4.3: *"The 70K and 23K models show worse performance but the validation plots (not included) showed that they likely suffered from overfitting. Regularization techniques (or data augmentation) might have boosted the numbers on these smaller training sets."* Recuerda que **no usaron dropout ni weight decay en ningún modelo**. Así que las dos filas inferiores no miden "qué pasa con menos datos", miden "qué pasa con menos datos y sin regularización". La caída real con un régimen bien regularizado sería menor, lo que **refuerza** aún más la conclusión de saturación.
2. **Ninguno de estos modelos completó una época.** El razonamiento de la Sección 4.3 es delicioso y vale la pena reproducirlo con verificación: con 20 mil millones de ejemplos y ResNet-50 corriendo a **11 mini-batches por segundo con 20 GPUs**, una época tomaría $\frac{2\times10^{10}}{128 \times 11} \approx 1.42\times10^7$ s $\approx$ **23 semanas** — exactamente lo que dice el paper. Pero *"if all videos were equal length and fully randomized, we expect to see at least one frame from each video in only 14 hours"* ($70\times10^6 / (11\times128) \approx 49{,}700$ s ≈ 13.8 h; también cierra). Y la hipótesis: *"even if we cannot get through an entire epoch, 70M videos will provide an advantage over 7M by virtue of the greater diversity of videos underlying the limited number of training patterns consumed."* **La tabla refuta esa hipótesis**: 70M no le gana significativamente a 7M ni a 700K. Los autores lo aceptan en las conclusiones, moderando el enunciado a *"increasing the number of videos up to 7M improves performance"*.

**Por qué esto es importante para el resto de tu trabajo:** este es un resultado de retornos decrecientes de datos medido a escala industrial y publicado por quienes tenían los datos. En un contexto de *record linkage* o MDM, la moraleja estructural es la misma: la curva de datos contra desempeño tiene un codo, y una vez pasado el codo la palanca deja de ser "más ejemplos" y pasa a ser "mejor arquitectura, mejor regularización o mejor calidad de etiqueta". El paper mide el codo y sigue de largo.

### 7.2. Eje 2 — tamaño del vocabulario de etiquetas (Sección 4.2, Tabla 3)

Setup: ResNet-50, 70M videos, 5M mini-batches (≈120 h), entrenando con 30K, 3K o 400 etiquetas, **siempre evaluando sobre las mismas 400**. La hipótesis: entrenar con más categorías podría actuar como regularizador, forzando representaciones intermedias que generalicen mejor incluso para las 400 clases de evaluación.

La tabla ya está reproducida en la Sección 6.2. **Y aquí hay que ser honesto: la evidencia es débil y no monótona.**

- Con cuello de botella: 400 → mAP 0.365; 3K → **0.347**; 30K → 0.369. **El punto intermedio es el peor.** Si la hipótesis fuera correcta, esperaríamos monotonía. No la hay: ni en mAP, ni en AUC (0.924, 0.919, 0.925), ni en d′.
- Sin cuello de botella: 400 → 0.376; 3K → 0.381; falta 30K. **La celda que decidiría el experimento está vacía**, y el paper explica por qué: *"We do not report metrics on the 30K label model without the bottleneck because it would have taken several months to train."*

El paper caracteriza su propio resultado con la palabra justa: *"These results provide **weak support** to the notion that training with a broader set of categories can help to regularize even the 400 class subset."* Las conclusiones repiten la cautela: *"can improve performance, **albeit modestly**"*.

**Mi lectura:** con un rango de ±0.02 de mAP entre condiciones, una sola corrida por celda y sin barras de error, no hay evidencia sólida de nada en este eje. El resultado que la tabla **sí** soporta con firmeza es el otro, el que no era la pregunta: **los modelos sin cuello de botella son consistentemente mejores que los que lo tienen** (0.381 vs 0.347 a 3K; 0.376 vs 0.365 a 400). Eso es una medición limpia del costo de la compresión, y es lo que citamos en la Sección 6.2.

### 7.3. Eje 3 — tamaño del modelo

Este eje **no se barre de forma controlada**, y hay que decirlo. Lo que existe es la Tabla 2, que varía **arquitectura**, no tamaño, y en la que tamaño y topología están confundidos:

| Modelo | Pesos | mAP |
|---|---|---|
| MLP 3×1000 | 11.2M | 0.058 |
| ResNet-50 | 30M | 0.182 |
| Inception-V3 | 28M | 0.181 |
| AlexNet | 37.3M | 0.115 |
| VGG (config. E) | 62M | 0.161 |

**El ordenamiento por parámetros es casi el inverso del ordenamiento por desempeño en el tramo superior.** VGG tiene el doble de pesos que ResNet-50 y saca 0.021 menos de mAP. AlexNet tiene 24 % más pesos que ResNet-50 y saca 0.067 menos. La única lectura defendible es: **a esta escala de datos, la capacidad bruta no es el cuello de botella; la topología sí.**

El único experimento del paper que se parece a una ablación de capacidad limpia es el del cuello de botella (Tabla 3), donde reducir la representación final de 2048 a 128 cuesta 3–9 % de mAP. Es informativo, pero es una ablación de **una capa**, no del modelo.

Un barrido de tamaño de modelo de verdad habría requerido, por ejemplo, ResNet-18 / 50 / 101 / 152 con todo lo demás fijo. **No está.** Y es la ausencia más notable del paper, porque es la variable que la literatura posterior identificó como decisiva: PANNs muestra que CNN14 sobre AudioSet, con la misma escala de datos que AudioSet y no la de YouTube-100M, llega a mAP 0.439.

### 7.4. Síntesis de los tres ejes

| Eje | Rango barrido | Efecto medido | Calidad de la evidencia |
|---|---|---|---|
| Datos de entrenamiento | 23K → 70M videos (×3000) | Grande hasta ~700K, **plano después** | Buena, pero confundida con sobreajuste no regularizado en el extremo bajo |
| Vocabulario de etiquetas | 400 → 30K (×75) | Modesto y **no monótono** | Débil; celda faltante, sin repeticiones |
| Tamaño de modelo | 11M → 62M pesos | Sin relación monótona | **No es un barrido controlado**; confundido con arquitectura |

**Lo que hay que dejar clarísimo, porque es lo que el paper realmente demostró:** más allá del orden de $10^5$ videos (unas $10^4$ horas), agregar datos deja de mover la aguja para una CNN de ~30M de parámetros sobre esta tarea. La escala de 5.24M horas no fue lo que hizo funcionar al modelo; lo que lo hizo funcionar fue **cruzar el umbral mínimo** y usar una arquitectura convolucional decente. El resto del corpus compró margen, robustez a clases raras (Figura 1) y, sobre todo, un titular.

---

## 8. Transferencia a AudioSet

### 8.1. Qué se hizo exactamente

Sección 4.4, completa:

> *"We train two fully-connected models to predict labels for Audio Set. The first model uses $64 \times 20$ log-mel patches and the second uses the output of the penultimate 'embedding' layer of our best ResNet model as inputs. The log-mel baseline achieves a balanced mAP of 0.137 and AUC of 0.904 (equivalent to d-prime of 1.846). The model trained on embeddings achieves mAP / AUC / d-prime of 0.314 / 0.959 / 2.452. This jump in performance reflects the benefit of the larger YouTube-100M training set embodied in the ResNet classifier outputs."*

**Corrección importante respecto de cómo suele describirse este experimento.** Esto **no es fine-tuning ni warm-starting**. El paper no reinicializa ni ajusta la ResNet sobre AudioSet. Lo que hace es:

1. Congelar la ResNet-50 entrenada en YouTube-100M.
2. Extraer la activación de la penúltima capa para cada segmento.
3. Entrenar **un MLP desde cero** sobre esos vectores congelados, con las etiquetas de AudioSet.
4. Comparar contra **otro MLP desde cero** que recibe log-mel crudo.

Es decir: **transferencia por features congeladas (linear/shallow probe)**, no fine-tuning. Y eso hace el resultado *más* fuerte, no menos: si un MLP sobre features congeladas duplica el mAP de un MLP sobre features crudas, la conclusión sobre la calidad de la representación es directa y no está contaminada por la capacidad de la cabeza.

### 8.2. Las cifras

| Entrada del MLP | mAP | AUC | d′ | EER derivado* |
|---|---|---|---|---|
| Log-mel $64 \times 20$ (200 ms de contexto) | 0.137 | 0.904 | 1.846 | 17.8 % |
| Embedding de la penúltima capa de ResNet-50 | **0.314** | **0.959** | **2.452** | **11.0 %** |
| **Ganancia** | **×2.29** | +0.055 | **+0.606** | **−6.8 pts** |

\* Columna derivada por mí vía $\text{EER}=\Phi(-d'/2)$; no está en el paper.

Otra forma de leer el AUC: la "tasa de error" complementaria $1-\text{AUC}$ cae de 0.096 a 0.041, una **reducción del 57 %**. Y el salto en d′ de 0.606 es, en unidades de teoría de detección, la diferencia entre un detector mediocre y uno usable.

### 8.3. El confounder que hay que anotar

**El baseline no es una comparación limpia.** El modelo de log-mel recibe parches de $64 \times 20$: 64 bandas mel × **20 tramas = 200 ms**. El modelo de embeddings recibe un vector que resume **960 ms**. Son casi **5× de diferencia en contexto temporal**.

Parte de la ganancia de 0.137 → 0.314, entonces, no es "el pre-entrenamiento ayuda" sino "ver 960 ms es mejor que ver 200 ms". El paper no separa las dos contribuciones y no ofrece un baseline de log-mel a 960 ms (que habría sido $64 \times 96$, exactamente la entrada de la ResNet). Es una omisión real. Dicho eso, es implausible que 5× de contexto explique por sí solo un factor 2.29 en mAP —una CNN completa entrenada desde cero sobre AudioSet, con contexto pleno, llegaba en esa época a números muy inferiores a 0.314— así que la conclusión cualitativa sobrevive. Pero la magnitud exacta de la ganancia por transferencia está sobreestimada por este diseño.

### 8.4. Por qué este resultado legitimó todo el paradigma

Tres razones, en orden de importancia:

**1. La aritmética del desbalance es brutal, y a favor de la transferencia.** AudioSet son ~3000 horas etiquetadas por humanos. YouTube-100M son 5.24M horas etiquetadas por máquina. La razón es **0.05 %** — el paper la calcula explícitamente. El resultado dice: *5.24M horas de supervisión sucia, comprimidas en un vector de 128 dimensiones por segundo, valen más que 3000 horas de supervisión limpia usadas directamente*. Ese es el argumento fundacional de todo el pre-entrenamiento a gran escala, aplicado a audio y con números.

**2. Es transferencia entre tareas distintas, no solo entre datasets.** YouTube-100M tiene etiquetas de **contexto de video** ("Game", "Web Page", "Boeing 757"); AudioSet tiene etiquetas de **evento acústico**. No es el mismo espacio de etiquetas ni la misma semántica. Que la representación transfiera de todos modos es la evidencia de que la red aprendió algo sobre **el sonido**, no sobre el vocabulario de YouTube. Es exactamente el análogo de que features de ImageNet sirvan para detección de objetos médicos.

**3. Convirtió el modelo en un producto.** Después de este resultado, la pregunta operativa para cualquiera que trabajara en audio dejó de ser *"¿qué arquitectura entreno?"* y pasó a ser *"¿por qué no estoy usando los embeddings de Google?"*. Y como AudioSet se distribuye **como embeddings**, la decisión venía tomada de fábrica. El paper no solo demostró que el pre-entrenamiento funciona: creó la infraestructura que lo hizo el camino de menor resistencia durante media década.

Un dato que cierra el arco y que verifiqué directamente: la línea *"Baseline [15] — CNN+MLP — mAP 0.314"* de la **Tabla 1 del paper de AST (Gong et al., 2021)** es este número. Cuatro años después, el baseline de referencia de la tabla de estado del arte de AudioSet seguía siendo el modelo de esta Sección 4.4.

---

## 9. Las diferencias entre audio e imagen que el paper deja ver

*Esta sección es análisis propio. El paper toca el tema en una sola frase de la Sección 1 —"Although the distinct meanings of time and frequency axes might argue for audio-specific architectures"— y no lo desarrolla. Es exactamente la misma omisión que el slide "Audio vs Image Data" de la Clase 39, que dice "there are relevant differences between audio and visual data that is important to consider" y pasa a la siguiente lámina. Lo que sigue es el desarrollo faltante.*

### 9.1. (a) Los ejes no son intercambiables

En una imagen natural, $x$ e $y$ son **el mismo tipo de cosa**: coordenadas espaciales sobre una superficie. Trasladar un objeto en $x$ o en $y$ produce una imagen que sigue conteniendo el mismo objeto. Esa equivarianza es la premisa de la convolución 2D con pesos compartidos: se justifica compartir el mismo kernel en todas las posiciones porque un borde en la esquina superior izquierda es el mismo evento visual que un borde en el centro.

En un espectrograma:

- **Trasladar en tiempo preserva el objeto.** Un ladrido a los 0.3 s y el mismo ladrido a los 0.7 s son el mismo evento. La equivarianza temporal es **correcta**, y es exactamente la mitad de lo que hace la convolución 2D.
- **Trasladar en frecuencia cambia el objeto.** Un desplazamiento de $\Delta$ bandas transpone el sonido. Una voz masculina desplazada hacia arriba es una voz femenina (o un chirrido); un Do se convierte en un Mi. **El objeto no se conserva.** La equivarianza en frecuencia es **incorrecta**.

Y hay un agravante técnico que casi nunca se menciona: **sobre el eje mel, un desplazamiento constante ni siquiera es una transposición limpia.** La escala es $m = 1127\ln(1+f/700)$: por debajo de ~700 Hz es aproximadamente **lineal** en Hz, por encima es aproximadamente **logarítmica**. Un desplazamiento en frecuencia sí corresponde a una multiplicación por un factor constante (una transposición musical) en el régimen logarítmico; en el régimen lineal de abajo corresponde a una **suma** en Hz, que no es una transposición de nada. Así que las traslaciones en frecuencia sobre un espectrograma mel implementan operaciones **distintas según dónde ocurran**. Para que un desplazamiento fuera transposición en todo el rango habría que usar un eje log puro o una CQT — que es exactamente lo que hace la literatura de música.

**Por qué funciona igual, entonces.** Dos mecanismos, y hay que distinguirlos porque las cinco redes del paper no los usan igual:

1. **La invarianza local es útil aunque la global sea falsa.** El kernel $3\times3$ de la primera capa no está detectando "un Do"; está detectando cosas como *"hay energía que sube en frecuencia con el tiempo"* o *"hay un borde vertical de banda ancha"* (un transitorio). Esos patrones **sí** son aproximadamente equivariantes en frecuencia: un barrido ascendente es un barrido ascendente en cualquier banda. Compartir pesos ahí es correcto y además es lo que permite reconocer una misma clase de sonido pronunciada por hablantes de distinto registro.
2. **La invarianza global se rompe (o no) en la cabeza.** Aquí las arquitecturas divergen de forma medible:
   - **VGG / VGGish aplanan** el mapa final ($6\times4\times512$) preservando la posición. La primera capa densa **puede leer la frecuencia absoluta**: el peso conectado a la banda 3 es distinto del conectado a la banda 60. La equivarianza en frecuencia se rompe exactamente donde debe romperse.
   - **ResNet-50 e Inception-V3 hacen average pooling global** ($6\times4$ y $10\times6$ respectivamente, es decir, sobre **todo** el mapa restante). Eso destruye la información de posición absoluta en **ambos** ejes. Estas redes son, arriba, **totalmente invariantes a transposición**.

   Y sin embargo, la Tabla 2 dice que ResNet-50 e Inception **ganan** (mAP 0.182 y 0.181) contra VGG (0.161). **Empíricamente, para etiquetas de contexto de video, la frecuencia absoluta es prescindible.** Eso tiene sentido: "Car", "Speech", "Guitar" se identifican por textura y envolvente espectral, no por altura absoluta. Sería un desastre para transcripción musical, estimación de $f_0$ o identificación de hablante.

**Conclusión para este punto: es genuinamente sorprendente que compartir pesos en frecuencia no rompa nada, y es todavía más sorprendente que descartar la posición absoluta con average pooling global sea la mejor opción. Ambas cosas dependen críticamente de que la tarea sea etiquetado grueso.** Para tareas sensibles a la altura, la conclusión se invierte, y la literatura lo confirma: las CNN de música casi siempre evitan el pooling global en frecuencia.

### 9.2. (b) La localidad no es simétrica

Un sonido armónico con fundamental $f_0$ tiene energía en $f_0, 2f_0, 3f_0, \dots$ Ese es un patrón **periódico y no local en frecuencia**, y es la firma más informativa que existe para distinguir un tono de un ruido, para separar fuentes y para identificar timbre.

**Un kernel $3\times3$ no lo captura.** Tres bandas mel adyacentes en la región de 1 kHz cubren del orden de 100–150 Hz. Si $f_0 = 200$ Hz, los armónicos están separados por 200 Hz: **el kernel de la primera capa nunca ve dos armónicos simultáneamente.** En imágenes esto no tiene análogo: la textura visual relevante suele ser localmente densa, y cuando hay periodicidad (una reja, un tejido) el patrón está a escala de pocos píxeles.

**Cómo lo resuelve la red igual: por profundidad.** Calculé el campo receptivo de la pila convolucional de VGGish (6 conv $3\times3$ con 4 max-pools de stride 2), usando $\text{RF}_{out} = \text{RF}_{in} + (k-1)\cdot j_{in}$ y $j_{out} = j_{in}\cdot s$:

| Capa | RF | Salto acumulado |
|---|---|---|
| conv1 | 3 | 1 |
| pool1 | 4 | 2 |
| conv2 | 8 | 2 |
| pool2 | 10 | 4 |
| conv3_1 | 18 | 4 |
| conv3_2 | 26 | 4 |
| pool3 | 30 | 8 |
| conv4_1 | 46 | 8 |
| conv4_2 | **62** | 8 |
| pool4 | **70** | 16 |

**El campo receptivo final es de 70 unidades en cada eje.** Como el eje de frecuencia tiene solo 64 bandas, **la última capa convolucional ve la totalidad del espectro**. La estructura armónica global es accesible — pero solo tras seis capas de composición, y la red tiene que **aprenderla** como una conjunción de patrones locales en lugar de recibirla como primitiva.

Ese es el costo real, y es un costo de eficiencia estadística: la red gasta capacidad y datos aprendiendo algo que la física del sonido regala. La literatura posterior lo atacó de frente con **harmonic stacking** (apilar copias del espectrograma desplazadas a $2f, 3f, \dots$ como canales, de modo que un kernel local sí vea armónicos alineados) y con la **CQT**, donde la separación entre armónicos es al menos independiente de $f_0$ en escala log. Que Hershey et al. no necesiten nada de eso confirma que para etiquetado grueso alcanza con la envolvente espectral.

**Veredicto: aquí el hecho de que funcione es moderadamente sorprendente**, y la explicación es que la profundidad compra alcance. Pero es la diferencia estructural donde una arquitectura específica de audio tiene el argumento más sólido.

### 9.3. (c) Las fuentes se superponen aditivamente; los objetos visuales se ocluyen

Esta es, en mi opinión, la diferencia más profunda de las cuatro, y la que tiene consecuencias prácticas más amplias.

**En imágenes, la composición es por oclusión.** Si un objeto A está delante de B, los píxeles de la región compartida son los de A. La información de B en esa región **se pierde por completo**, pero la información de A **se conserva intacta**. La operación es esencialmente un $\max$ en profundidad, o una selección. Consecuencias: un detector puede confiar en que la evidencia local que ve pertenece a un solo objeto, y la segmentación semántica —asignar cada píxel a exactamente una clase— es un problema bien planteado.

**En audio, la composición es por suma.** Si dos fuentes suenan a la vez, la presión acústica en el micrófono es $x(t) = x_1(t) + x_2(t)$. La linealidad se propaga a la STFT, que es lineal: $X(f,t) = X_1(f,t) + X_2(f,t)$. Es al pasar a magnitud donde se complica, porque la suma es compleja y depende de la fase relativa:

$$|X| = \big| |X_1| e^{j\phi_1} + |X_2| e^{j\phi_2} \big| \le |X_1| + |X_2|$$

y luego el log rompe cualquier aditividad residual:

$$\log(|X_1| + |X_2| + \epsilon) \neq \log(|X_1|+\epsilon) + \log(|X_2|+\epsilon)$$

En el régimen habitual, donde una fuente domina ($|X_1| \gg |X_2|$), $\log(|X_1|+|X_2|) \approx \log|X_1|$: **la fuente débil desaparece.** Eso es enmascaramiento, y es la razón física por la que el log-mel es una representación *dispersa* en la que, en cada celda tiempo-frecuencia, típicamente una fuente domina (la hipótesis W-disjoint orthogonality, base de todos los métodos clásicos de separación por máscara binaria).

**Cuatro consecuencias concretas y verificables en este paper:**

1. **La tarea es intrínsecamente multi-etiqueta, y por eso la Sección 3.1 usa sigmoide y no softmax.** *"All models used a final sigmoid layer rather than a softmax layer since each example can have multiple labels."* Eso no es un detalle de implementación: es la superposición aditiva manifestándose en la capa de salida. En clasificación de imágenes de un objeto, softmax es natural. En audio, un segmento de 960 ms **casi siempre** contiene varias fuentes simultáneas, y ninguna oculta a las otras.
2. **La red tiene que hacer separación de fuentes implícita.** Para emitir "Trumpet" con confianza en un segmento donde también hay batería, público y ruido de sala, tiene que aislar la evidencia de la trompeta de un patrón que contiene todo sumado. En visión eso es más fácil: la trompeta ocupa píxeles que son solo de la trompeta.
3. **Mixup es física, no regularización.** En imágenes, promediar dos imágenes con pesos $\lambda$ y $1-\lambda$ produce un artefacto que no existe en el mundo, y su valor es puramente como regularizador. En audio, **sumar dos formas de onda produce un sonido real que un micrófono podría haber capturado, y cuyo conjunto correcto de etiquetas es la unión de los dos.** El aumento de datos por mezcla es una operación **semánticamente exacta**. Ese es el motivo por el cual mixup dio mejoras tan grandes en AudioSet (PANNs, PSLA, AST lo usan todos) mientras que en visión es más marginal. Y es también el fundamento de **Scaper** (Salamon et al., 2017, presente en esta misma carpeta de la clase): sintetizar paisajes sonoros mezclando eventos sobre fondos, con anotación exacta y gratuita. **En visión no existe un Scaper, porque no se pueden componer objetos sin resolver oclusión, sombras e iluminación.**
4. **La aditividad afecta la agregación temporal.** Como discutí en la Sección 6.4, promediar scores sobre un video diluye eventos breves. En imágenes el análogo (promediar sobre crops) es benigno porque los objetos ocupan regiones extensas.

**Veredicto: es aquí donde *menos* sorprende que las CNN de visión funcionen** —una CNN es un detector de patrones locales y no le importa cómo se compuso el patrón— pero es donde **más** cambia todo lo que rodea al modelo: la función de pérdida, el aumento de datos, la métrica y la agregación.

### 9.4. (d) El eje de frecuencia es logarítmico por diseño perceptual

Los píxeles de una imagen viven en un espacio de coordenadas **lineal y físicamente neutro**: la coordenada $x$ es proporcional al ángulo subtendido, sin ninguna deformación introducida por el diseñador. Nadie aplica una transformación no lineal a las coordenadas espaciales antes de alimentar la CNN. (Sí se aplica gamma a la **intensidad**, un análogo débil del log-mel en el eje de valores, pero no a la geometría.)

El espectrograma log-mel aplica **dos** transformaciones no lineales antes de que la red vea nada:

- **En el eje de frecuencia:** el warp mel $m = 1127\ln(1+f/700)$. La coordenada del "píxel" no es la frecuencia física; es una función no lineal de ella, elegida por experimentos psicoacústicos de 1937.
- **En el eje de valores:** el logaritmo de la magnitud, comprimiendo 80–100 dB de rango en un intervalo manejable.

**Lo que esto implica.** El "imagen" que la CNN procesa **no es el dato físico**: es el dato ya filtrado por un modelo del sistema auditivo humano. Hay una cantidad enorme de conocimiento de dominio —resolución crítica del oído, ley de Weber-Fechner, banda útil del habla— incrustada en el preprocesamiento y no en la arquitectura. **La afirmación "usamos redes de visión sin modificar" es cierta a nivel de arquitectura y falsa a nivel de sistema:** la adaptación al audio existe y está toda en el front-end.

Esto también explica por qué las redes sobre forma de onda cruda (Dai et al. 2016, SincNet, wav2vec) tardaron tanto en competir: no están compitiendo contra "una CNN"; están compitiendo contra "una CNN más ochenta años de psicoacústica". Y explica por qué el trabajo posterior que sí mejoró el front-end (más bandas, 32 kHz, per-channel energy normalization, LEAF) obtuvo ganancias reales.

**Segunda implicación, sobre aumento de datos.** El repertorio estándar de visión **no se traslada**:

| Aumento en visión | Traslado a espectrograma |
|---|---|
| Flip horizontal | **Inválido.** Invierte el tiempo. Un sonido reproducido al revés es otro sonido (piensa en el ataque de un piano). |
| Flip vertical | **Inválido.** Invierte el espectro. Absurdo físicamente. |
| Rotación | **Inválida.** Mezcla tiempo con frecuencia; unidades incompatibles. |
| Random resized crop (escala) | **Parcialmente inválido.** Estirar el eje de tiempo es *time stretch* (legítimo, cambia duración); estirar el eje de frecuencia es transposición (cambia la clase para tareas de pitch). **Estirar ambos con el mismo factor —que es lo que hace el crop de visión— no corresponde a ninguna operación acústica.** |
| Traslación | Válida **solo** en tiempo. |
| Jitter de color / brillo | Análogo válido: ganancia global (offset aditivo en log). |
| Cutout / random erasing | **Sí traslada, y muy bien:** es esencialmente **SpecAugment** (Park et al., 2019), pero con la restricción de enmascarar **bandas completas** de tiempo o de frecuencia, no rectángulos arbitrarios — porque el enmascaramiento acústico real ocurre así. |

Que el aumento correcto para espectrogramas (SpecAugment) haya llegado dos años después de este paper y sea una **restricción** de una técnica de visión, y no una importación directa, es la mejor evidencia de que este eje sí es distinto.

**Veredicto: no es sorprendente que la arquitectura funcione, porque la adaptación al audio ya ocurrió antes de la primera capa.**

### 9.5. Síntesis: ¿dónde sorprende y dónde no?

| Diferencia | ¿Sorprende que las redes de visión funcionen? | Por qué |
|---|---|---|
| (a) Ejes no intercambiables | **Sí, bastante** | Compartir pesos en frecuencia es formalmente injustificado. Funciona porque los patrones de bajo nivel sí son equivariantes y porque la tarea no depende de la altura absoluta. El average pooling global de ResNet/Inception, que descarta la frecuencia absoluta, **gana** — eso es el resultado más contraintuitivo. |
| (b) Localidad asimétrica (armónicos) | **Sí, moderadamente** | Kernels $3\times3$ no ven armónicos. La profundidad lo resuelve (RF = 70 > 64 bandas), pero de forma estadísticamente ineficiente. Es donde una arquitectura de audio tiene el mejor caso. |
| (c) Superposición aditiva vs oclusión | **No** | Una CNN es un detector de patrones; le da igual el proceso generativo. Lo que sí cambia es todo lo demás: sigmoide en vez de softmax, mixup como aumento válido, pooling de agregación. |
| (d) Eje logarítmico por diseño | **No** | La adaptación al audio ya está hecha, en el front-end. La red recibe un objeto que ya fue perceptualizado. Pero invalida la mitad del recetario de aumento de visión. |

**Y el punto que unifica todo:** la frase de la Sección 1 —*"the distinct meanings of time and frequency axes might argue for audio-specific architectures"*— es correcta como preocupación teórica. La contribución del paper es mostrar que, **para esta tarea**, el efecto neto es pequeño. El "para esta tarea" es la letra chica que la clase 39 debería incluir en su slide: etiquetado grueso de eventos y contexto, con contexto de ~1 s, donde la envolvente espectral basta. Cambia la tarea a transcripción musical, estimación de $f_0$, verificación de hablante o separación de fuentes, y las cuatro diferencias vuelven a morder.

---

## 10. Limitaciones

**1. Ruido de etiqueta no cuantificado ni cuantificable.** Ya discutido en la Sección 2, pero conviene enumerar las consecuencias: (i) todas las cifras absolutas del paper tienen un techo impuesto por el ruido de etiqueta y no sabemos dónde está ese techo; (ii) el sesgo del canal visual (las etiquetas se derivan en parte de "image content") introduce una correlación sistemática que la escala **no** promedia; (iii) los propios autores reconocen que ni siquiera pueden definir "relevancia" para la mayoría de las clases.

**2. El dataset no es reproducible.** YouTube-100M es interno de Google. Las etiquetas son identificadores del Knowledge Graph, que es propietario y cambiante. **Nadie fuera de Google puede reproducir la Tabla 2, la Tabla 3 ni la Tabla 4.** Lo que la comunidad recibió es el checkpoint, no el experimento. Es un paper cuyo resultado central —"la escala funciona"— es, en sentido estricto, no verificable de forma independiente. La comunidad terminó verificándolo indirectamente: PANNs, entrenando solo sobre AudioSet, superó ampliamente estos números, lo que sugiere que **la escala de YouTube-100M no era la variable crítica** (consistente con la Tabla 4).

**3. Ausencia de modelado temporal explícito.** La única operación temporal por encima de 960 ms es un promedio de scores. La Sección 1 lo reconoce: *"our labels apply to entire videos without any changes in time, so we have yet to try such recurrent models"*, y justifican la elección apoyándose en Ng et al. [20], que encontraron que el promedio simple funcionaba casi tan bien como modelos más complejos **en video visual**. Trasladar esa conclusión a audio no es obvio: en audio, el orden temporal es más discriminativo que en video de acciones (la Clase 38 y el análisis de S3D dan la evidencia opuesta para video: en Kinetics revertir el tiempo no cambia nada). El costo concreto está descrito en la Sección 6.4: **el promedio de scores diluye eventos breves y raros**, que son exactamente los que interesan en detección de eventos acústicos. PANNs y PSLA reemplazaron esto por attention pooling y ganaron.

**4. Resolución temporal de 960 ms como unidad mínima.** El sistema no puede localizar nada más fino que ~1 s. Eventos de interés que duran menos: un disparo (~50 ms de transitorio), un chasquido de dedos, un fonema (~80 ms), el ataque de una nota, un clic de una válvula cardíaca. Todos quedan promediados dentro de su parche. Y como los parches no se solapan en entrenamiento, un evento a caballo entre dos parches se reparte entre ambos. **Este es el límite duro del formato VGGish**, y es la razón principal por la que el mundo del sound event *detection* (con marcas de tiempo, no solo etiquetas) nunca adoptó VGGish como front-end.

**5. Ninguna regularización, en ningún modelo.** Justificado para 70M videos. **No justificado** para las filas de 70K y 23K de la Tabla 4, que los propios autores admiten que sobreajustaron. Esas dos filas, entonces, no miden el efecto de la escala de datos: miden el efecto conjunto de escala y ausencia de regularización.

**6. La evidencia del eje de vocabulario es floja.** Celda faltante (30K sin bottleneck), no monotonía, sin repeticiones, sin barras de error, diferencias del orden de ±0.02 de mAP. El paper lo califica de "weak support", correctamente; la literatura que lo cita a veces no.

**7. Comparación a pasos fijos, no a cómputo fijo.** Reconocido en la Sección 4.1. Con 184 h para VGG contra 119 h para ResNet-50, la Tabla 2 no responde "cuál es mejor con presupuesto X", que suele ser la pregunta real.

**8. Todas las arquitecturas están sin converger.** ResNet-50 gana 16 % de mAP relativo entre 5M y 17M pasos. Las diferencias entre las cuatro CNN (0.115 a 0.182) son grandes, pero las diferencias entre Inception y ResNet (0.181 vs 0.182) están claramente dentro del ruido.

**9. Cero baselines específicos de audio.** No hay CRNN, ni modelo sobre forma de onda, ni filtros con receptive field alargado en frecuencia, ni comparación contra los métodos clásicos (MFCC+GMM) sobre el mismo dataset. La conclusión "las redes de imagen funcionan" está establecida; la conclusión implícita "y son la mejor opción" no está testeada.

**10. El baseline de transferencia usa 200 ms contra 960 ms.** Ya discutido en 8.3. Infla la ganancia atribuida a la transferencia.

**11. Ninguna ablación de qué capa produce el mejor embedding.** Se usa "la penúltima". No se prueba la anterior, ni una concatenación, ni el pooling de varias. Esa pregunta la resolvió la literatura posterior (y la respuesta suele ser que las capas intermedias transfieren mejor para tareas alejadas del pre-entrenamiento).

---

## 11. Impacto y legado

### 11.1. VGGish y YAMNet como los extractores estándar

Google liberó el modelo en el repositorio de AudioSet (`tensorflow/models/research/audioset`) y luego en TensorFlow Hub, y durante aproximadamente 2017–2021 **VGGish fue el default de facto para features de audio**. Lo hizo por razones que tienen poco que ver con su calidad y mucho con la fricción:

- Un tensor de $96\times64$ y una API de una línea (`waveform_to_examples`).
- 128 flotantes por segundo: barato de calcular, barato de guardar, barato de indexar.
- Ports a PyTorch mantenidos por terceros (el laboratorio de la clase usa uno: `tcvrick/audioset-vggish-tensorflow-to-pytorch`).
- **AudioSet se distribuye como embeddings de VGGish.** Quien quisiera usar AudioSet sin re-descargar dos millones de clips de YouTube, usaba VGGish. No era una elección.

**YAMNet** llegó después como el hermano eficiente: misma entrada de $96\times64$ log-mel, mismo formato de salida, pero backbone **MobileNet-v1** (convoluciones separables en profundidad) y una cabeza de **521 clases** de la ontología de AudioSet. Es el que se usa cuando hace falta clasificación directa en vez de embeddings, o cuando hay que correr en el borde. Nota la ironía histórica: **YAMNet es, otra vez, una arquitectura de visión importada sin cambios.** La tesis de Hershey et al. no solo sobrevivió; se volvió el procedimiento operativo estándar de Google para audio.

*(Los detalles de YAMNet provienen de la documentación del repositorio y de TF Hub, no de este paper.)*

### 11.2. AudioSet como benchmark

AudioSet se convirtió en el ImageNet del audio en el sentido que importa: **la tarea sobre la que se mide el progreso**. La progresión de mAP sobre el conjunto completo, con las cifras que pude verificar contra la **Tabla 1 del paper de AST** (Gong et al., 2021), disponible en esta misma carpeta:

| Año | Sistema | Arquitectura | mAP (full AudioSet) |
|---|---|---|---|
| 2017 | **Baseline (este paper / AudioSet)** | CNN embeddings + MLP | **0.314** |
| 2020 | PANNs (Kong et al.) | CNN + attention pooling | 0.439 |
| 2021 | PSLA (single) | CNN + attention | 0.444 |
| 2021 | PSLA (Ensemble-M) | CNN + attention | 0.474 |
| 2021 | **AST (single, weight-averaged)** | Pure attention (ViT/DeiT) | **0.459** |
| 2021 | AST (Ensemble-M) | Pure attention | **0.485** |

Todos esos números están verificados contra la Tabla 1 de `Gong-AST-2021.txt`. Sistemas posteriores (BEATs, Audio-MAE, CAV-MAE) reportan valores en el rango 0.48–0.51; **esos no los pude verificar contra un PDF en esta carpeta y los cito con esa reserva.**

El detalle que más me llama la atención: **la fila 0.314 de este paper sigue estando en la tabla de estado del arte cuatro años después, como el punto de referencia.** Es la definición de un baseline que se volvió canónico.

### 11.3. Qué desplazó a VGGish, y por qué

Cuatro fuerzas, en orden aproximadamente cronológico:

**1. Entrenar directamente sobre AudioSet, con las recetas correctas (PANNs, 2020).** La observación que rompió el paradigma: si la Tabla 4 de este paper dice que 700K videos rinden casi lo mismo que 70M, entonces las ~2M de clips de AudioSet **ya son suficientes** para entrenar una CNN grande desde cero. PANNs lo hizo y llegó a 0.439 contra 0.314 — un salto de 40 % relativo, **sin YouTube-100M**. Lo que aportó no fue escala sino: audio a **32 kHz** (más ancho de banda), **mixup** (que la Sección 9.3 explica por qué es tan potente en audio), **balanced sampling** para las clases raras, y **attention pooling** en vez de promedio. **En retrospectiva, el modelo de Hershey et al. estaba limitado por el front-end, el aumento y la agregación, no por los datos.** Su propia Tabla 4 lo insinuaba.

**2. Attention pooling reemplazando el promedio.** Directamente el punto 3 de la Sección 10. Un evento de 200 ms en un clip de 10 s se recupera si el modelo aprende a ponderar segmentos; se pierde si se promedian scores.

**3. Transformers con pesos de ImageNet (AST, 2021).** Este es el capítulo más irónico del legado. AST descarta la convolución, corta el espectrograma en parches de $16\times16$ y aplica un ViT — **inicializado con pesos de DeiT entrenados en ImageNet**, con una interpolación de los embeddings posicionales para acomodar la forma no cuadrada del espectrograma. La Tabla 2 de AST cuantifica la contribución de esa inicialización: **sin pre-entrenamiento en ImageNet, 0.366 de mAP; con él, 0.459.** Es decir, **+0.093 de mAP puramente por importar pesos de un modelo de imágenes.**

   Léelo en contexto: Hershey et al. mostraron en 2017 que las *arquitecturas* de visión transfieren a audio. AST mostró en 2021 que los *pesos* de visión transfieren a audio. **La tesis de este paper no fue refutada; fue radicalizada.** La analogía espectrograma-como-imagen resultó ser más literal de lo que sus propios autores propusieron.

**4. Auto-supervisión (SSAST, Audio-MAE, BEATs, wav2vec 2.0 / HuBERT).** El golpe final al planteamiento original: si se puede aprender una representación de audio **sin etiquetas**, entonces todo el aparato de 5.24M horas de etiquetas ruidosas del Knowledge Graph deja de ser necesario. Las 5.24M horas siguen siendo útiles — pero como audio, no como pares audio-etiqueta.

**Qué se le puede seguir criticando a VGGish, y qué sigue sirviendo.** Las razones técnicas por las que envejeció: granularidad de 960 ms; embedding **post-ReLU** (no negativo, mal condicionado para métricas de similitud); cuantización a 8 bits en el formato distribuido; banda limitada a 7500 Hz; front-end congelado; entrenamiento sobre etiquetas de contexto de video, no de eventos acústicos; y ninguna de las recetas modernas de aumento. Lo que sigue justificándolo hoy: es **rápido, determinista, corre en CPU, no requiere GPU ni un stack de transformers**, y para un baseline o un sistema de recuperación por similitud sigue siendo perfectamente razonable. Es el equivalente de audio a usar features de ResNet-50 en 2024: nadie escribe un paper con eso, pero mucha gente lo despliega.

### 11.4. El legado conceptual

Más allá del modelo, quedaron tres ideas:

1. **"El espectrograma es una imagen" es una aproximación de trabajo, no una metáfora.** Con márgenes de error que la Sección 9 delimita, pero suficientemente buena para justificar diez años de importación de arquitecturas.
2. **El embedding congelado como producto.** La idea de que la salida útil de un modelo grande no es su predicción sino su representación intermedia, distribuible como un artefacto compacto. En audio, este paper lo instaló.
3. **Las curvas de escala hay que medirlas, no asumirlas.** La Tabla 4 es un ejemplo temprano y honesto de un equipo con acceso a datos ilimitados publicando que sus datos ilimitados no eran necesarios.

---

## 12. Conexión con la Clase 39 y el laboratorio

### 12.1. Qué afirma la clase y qué aporta el paper

El slide "Audio vs Image Data" de `APPS1_C3_T1.pdf` dice, textualmente:

> *"In principle, the 2D time-freq representation (spectrogram) of an audio signal can be interpreted as an image. […] In this way, we can use 2D CNNs to process audio signals. While this is possible, there are relevant differences between audio and visual data that is important to consider."*

Este paper es exactamente las dos cosas: la **validación empírica a gran escala** de la primera afirmación (Tabla 2: cuatro arquitecturas de visión, ninguna adaptada al audio, todas superando ampliamente al MLP y replicando su ranking de ImageNet) y, en su Sección 1, la **fuente de la advertencia** que el slide no desarrolla (*"the distinct meanings of time and frequency axes might argue for audio-specific architectures"*). La Sección 9 de este documento es el desarrollo faltante.

### 12.2. Auditoría del "más de 60 millones de parámetros"

La celda 60 del notebook `Practico_3_Audio_DINTA_alumnos_v3.ipynb` afirma:

> *"Este modelo es bastante más pesado que los modelos M que vimos en la parte anterior en términos de parámetros. Principalmente, esto sucede por la sección final Fully Connected. ¡Ésta acumula por sí sola más de 60 millones de parámetros! Por esto, es esta sección no entrenaremos por más de 2 épocas el modelo."*

Desglosé la arquitectura exacta definida en la celda 59 del notebook. Con entrada $1 \times 96 \times 64$:

| Capa | Salida | Parámetros | Multiplicaciones (MACs) |
|---|---|---|---|
| `Conv2d(1, 64, 3, p=1)` | $64\times96\times64$ | $1\cdot64\cdot9 + 64 = $ **640** | 3.5M |
| `MaxPool(2,2)` | $64\times48\times32$ | 0 | — |
| `Conv2d(64, 128, 3)` | $128\times48\times32$ | **73,856** | 113.2M |
| `MaxPool(2,2)` | $128\times24\times16$ | 0 | — |
| `Conv2d(128, 256, 3)` | $256\times24\times16$ | **295,168** | 113.2M |
| `Conv2d(256, 256, 3)` | $256\times24\times16$ | **590,080** | 226.5M |
| `MaxPool(2,2)` | $256\times12\times8$ | 0 | — |
| `Conv2d(256, 512, 3)` | $512\times12\times8$ | **1,180,160** | 113.2M |
| `Conv2d(512, 512, 3)` | $512\times12\times8$ | **2,359,808** | 226.5M |
| `MaxPool(2,2)` | $512\times6\times4$ | 0 | — |
| **Subtotal convolucional** | flatten → **12,288** | **4,499,712** | **~796M** |
| `Linear(12288, 4096)` | 4096 | **50,335,744** | 50.3M |
| `Linear(4096, 4096)` | 4096 | **16,781,312** | 16.8M |
| `Linear(4096, 128)` | 128 | **524,416** | 0.5M |
| **Subtotal denso** | | **67,641,472** | **~67.6M** |
| **TOTAL** | | **72,141,184** | **~864M** |

**Verificado: la afirmación del notebook es correcta. La sección fully-connected acumula 67.6 millones de parámetros**, el **93.8 %** de los 72.1M totales. El orden de magnitud es exacto y "más de 60 millones" es una descripción conservadora.

**Y hay una inversión que vale la pena señalar, porque es la firma arquitectónica de toda la familia VGG:**

| | Parámetros | Cómputo |
|---|---|---|
| Convolucional | 4.5M (**6.2 %**) | ~796M MACs (**92 %**) |
| Denso | 67.6M (**93.8 %**) | ~67.6M MACs (**8 %**) |

**La parte que tiene los parámetros no es la que hace el trabajo.** Los 67.6M de las densas son casi todos memoria y riesgo de sobreajuste; las 796M de multiplicaciones están en 4.5M de pesos convolucionales que se reutilizan en cada posición del mapa. De ahí el `Linear(12288, 4096)` con sus 50.3M de parámetros solitarios: la primera capa densa, que conecta el mapa aplanado de $512\times6\times4$ con 4096 unidades, **es por sí sola el 70 % del modelo**. Esta es exactamente la patología que Inception y ResNet resolvieron con global average pooling, y es por lo que en la Tabla 2 del paper ResNet-50 (30M pesos) le gana a VGG (62M).

Comparación con la Parte 1 del laboratorio, usando los tamaños que reporta Dai et al. (Tabla 1 de `Dai-Deep-CNN-Raw-Waveforms-2017.txt`):

| Modelo | Parámetros | Accuracy en UrbanSound8K |
|---|---|---|
| M3 | 0.2M | — |
| M5 | 0.5M | — |
| M11 | 1.8M | — |
| M18 | 3.7M | **71.68 %** (Dai et al.) |
| M34-res | 4M | 63.47 % (sobreajuste, según el propio Dai) |
| **VGGish** | **72.1M** | — |

**VGGish tiene ~20 veces más parámetros que el mayor de los modelos M.** Eso es lo que la celda 60 quiere transmitir, y es correcto.

### 12.3. Qué implica hacer fine-tuning de esto con 8732 clips — y qué hace realmente el notebook

**La aritmética del riesgo.** UrbanSound8K tiene 8,732 clips de ≤4 s en 10 clases, repartidos en 10 folds. El notebook entrena con los folds 2–10 (~7,850 clips) y testea con el fold 1. Como cada clip se rellena/recorta a 3 s y produce **3 parches de 960 ms**, el conjunto de entrenamiento efectivo es de ~23,500 ejemplos.

Ajustar **72.1M de parámetros con 23,500 ejemplos** son **~3,070 parámetros por ejemplo**. Sin regularización fuerte, eso es memorización garantizada. Y hay un agravante específico de este dataset: **los 3 parches de un mismo clip son fuertemente correlacionados y comparten etiqueta**, así que el número efectivo de muestras independientes está más cerca de 7,850 que de 23,500.

**Pero el notebook no hace fine-tuning completo. Hace un *probe* lineal.** Leyendo la celda 74:

```python
net = VGGish()
net.load_state_dict(torch.load('./pytorch_vggish.pth'))
for param in net.parameters():
    param.requires_grad = False          # <-- congela TODO
net.fc[-2] = nn.Linear(in_features=4096, out_features=10, bias=True)
```

`fc` es `[Linear, ReLU, Linear, ReLU, Linear(4096,128), ReLU]`, así que `fc[-2]` es `fc[4]`, la capa de embedding. Se la reemplaza por una `Linear(4096, 10)`, que nace con `requires_grad=True`.

**Parámetros entrenables reales: $4096 \times 10 + 10 = $ 40,970.** No 72M, no 67.6M: **cuarenta mil**. Eso es lo que imprimirá `num_trainable_parameters(net)`.

Tres consecuencias que conviene tener claras:

1. **No hay riesgo de sobreajuste catastrófico.** 40,970 parámetros contra ~23,500 ejemplos es un régimen perfectamente sano — es esencialmente una regresión logística multinomial sobre features congeladas.
2. **La advertencia de "no más de 2 épocas" no es por sobreajuste; es por tiempo de cómputo.** El título de la celda 73 dice "1 hora y media". Con batch 32 y ~735 pasos por época, un probe lineal sobre una GPU debería tardar minutos. El costo real está en `__getitem__`: `vggish_input.waveform_to_examples()` corre en **NumPy sobre CPU** (incluye el resampleo con `resampy` de 44.1 kHz a 16 kHz, la STFT y el banco mel) y el `DataLoader` está configurado con **`num_workers: 0`**. Todo el preprocesamiento se hace en el proceso principal, en serie, en cada época. **El cuello de botella del laboratorio es el front-end, no la red.** (Arreglo obvio si quieres experimentar: precomputar los parches una vez y cachearlos, o subir `num_workers`.)
3. **El clasificador no lee el embedding de 128 dimensiones.** Al reemplazar `fc[4]`, la cabeza se conecta a la salida de `fc[2]`, que tiene **4096 dimensiones**. Es un probe sobre una representación 32× más rica que el embedding canónico de VGGish. Funciona mejor, pero no es "usar los embeddings de VGGish" en el sentido en que lo entiende la literatura. Vale la pena saberlo si comparas resultados con papers.

**Un bug que hay que anotar.** Después de reemplazar `fc[4]`, la secuencia `fc` queda como `[..., Linear(4096,10), ReLU(inplace=True)]`. **Los logits pasan por una ReLU antes de entrar a `nn.CrossEntropyLoss`.** Eso significa que:

- Todos los logits son $\ge 0$, así que ninguna clase puede recibir evidencia negativa.
- Para cualquier clase cuya pre-activación sea negativa, la derivada de la ReLU es 0 y **el gradiente no llega a sus pesos**. Esas clases quedan congeladas hasta que algún otro camino las reactive, y como el gradiente está bloqueado, no hay ninguno.
- El modelo solo puede aprender **subiendo** el logit de la clase correcta, nunca bajando el de las incorrectas por debajo de cero.

Es un error de ejecución (`fc[-2]` reemplaza la capa correcta pero deja la ReLU terminal en su lugar) y limita el desempeño alcanzable. La corrección es reemplazar las dos últimas posiciones, por ejemplo `net.fc = nn.Sequential(*list(net.fc)[:4], nn.Linear(4096, 10))`. Junto con `lr=0.01` para Adam —alto incluso para una capa lineal— explica buena parte de cualquier resultado mediocre que observes.

### 12.4. Por qué el fine-tuning de VGGish debería ganarle a una M-net desde cero

**El argumento a favor, en cuatro puntos:**

1. **Asimetría de supervisión.** VGGish trae dentro el equivalente de millones de horas de audio etiquetado. Una M-net entrenada desde cero tiene 7,850 clips. La diferencia es de seis órdenes de magnitud.
2. **Las features de bajo nivel son genéricas y reutilizables.** Los 4.5M de parámetros convolucionales de VGGish codifican detectores de onsets, de pilas armónicas, de texturas de ruido de banda ancha, de barridos frecuenciales. Nada de eso es específico de YouTube: es la gramática básica del sonido. Una taladradora, una sirena y un ladrido se describen con ese mismo vocabulario.
3. **Solapamiento casi perfecto de dominio.** Las 10 clases de UrbanSound8K —`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jack_hammer`, `siren`, `street_music`— son sonidos ambientales urbanos capturados con micrófonos de consumo, exactamente la distribución de la banda sonora de un video de YouTube. Es prácticamente el mejor caso posible para transferencia.
4. **El front-end log-mel ya está resuelto.** La M-net tiene que aprender un banco de filtros desde la forma de onda cruda con 7,850 clips. Dai et al. muestran que sí lo aprende (los kernels de la primera capa se vuelven pasabanda), pero es capacidad y datos gastados en reinventar la STFT. VGGish la recibe gratis y perfeccionada por psicoacústica.

El propio notebook lo constata en su celda de cierre: *"con un modelo preentrenado, en la primera época de entrenamiento ya tenemos un modelo competitivo con haber entrenado desde 0"* — y eso con solo 40,970 parámetros entrenables, un probe lineal, y con la ReLU terminal en contra.

**Cuándo NO ganaría.** Este es el lado más útil de la pregunta:

- **Desajuste de banda.** El front-end recorta a [125, 7500] Hz. Para telefonía a 8 kHz gran parte de la banda queda vacía; para sonidos cardíacos o respiratorios (< 125 Hz), ultrasonido, o vibración industrial, la información **discriminativa está fuera del rango que VGGish puede ver**. Aquí una red sobre forma de onda cruda o un front-end a medida gana de calle. Nota que este es precisamente el escenario médico.
- **Resolución temporal fina.** Cualquier tarea que requiera precisión sub-segundo —detección de onsets, segmentación de fonemas, conteo de eventos rápidos— choca contra la unidad de 960 ms. Una M-net con stride pequeño sobre la forma de onda conserva esa resolución.
- **Invarianzas equivocadas.** VGGish fue entrenada para etiquetar **qué** suena, lo que implica ser **invariante** a quién habla, a la altura absoluta y al canal de grabación. Para identificación de hablante, estimación de $f_0$ o detección de dispositivo, la red fue entrenada explícitamente para descartar la señal que necesitas.
- **Dataset objetivo grande.** Si dispones de decenas de miles de horas etiquetadas en tu dominio, entrenar en dominio gana. Es exactamente lo que demostró PANNs frente a este paper.
- **Desajuste de preprocesamiento.** Si el pipeline alimenta MFCC en vez de log-mel, o normaliza distinto, o cambia la frecuencia de muestreo, los pesos preentrenados operan sobre una distribución de entrada que nunca vieron y el modelo puede quedar por debajo de una red pequeña entrenada correctamente. **La confusión "log-mel vs MFCC" del propio notebook es un ejemplo de cómo se llega a eso por accidente** (Sección 13).
- **Cabeza mal cableada.** El bug de la ReLU terminal es suficiente para que un M18 bien entrenado (71.68 % según Dai) supere a un VGGish mal conectado.
- **Régimen de datos muy pequeño con fine-tuning completo y sin congelar.** Descongelar los 72.1M con unos miles de ejemplos y sin regularización destruye las features preentrenadas en pocos cientos de pasos (*catastrophic forgetting* del pre-entrenamiento). Con datos escasos, el probe congelado que hace el notebook es la decisión correcta, no una simplificación.

---

## 13. Erratas, matices y cosas que se citan mal

### 13.1. La errata del material del curso: arXiv 1610.00087

**Confirmada.** El notebook `Practico_3_Audio_DINTA_alumnos_v3.ipynb` enlaza VGGish así, en la celda 57:

> *"Para esto, ocuparemos el modelo VGGish (https://arxiv.org/pdf/1610.00087.pdf), modelo como cuyo nombre indica es similar en estructura al modelo VGG que usamos en imágenes."*

Verifiqué ambos identificadores contra los PDFs de esta carpeta:

| arXiv | Título real | Autores | Es el paper de… |
|---|---|---|---|
| **1610.00087** | *Very Deep Convolutional Neural Networks for Raw Waveforms* | Wei Dai, Chia Dai, Shuhui Qu, Juncheng Li, Samarjit Das (CMU / Stanford / Bosch) | **Los modelos M3, M5, M11, M18, M34-res** de la Parte 1 del laboratorio |
| **1609.09430** | *CNN Architectures for Large-Scale Audio Classification* | Hershey et al. (Google) | **VGGish** |

La ironía es completa: **1610.00087 es el paper de la Parte 1 del mismo notebook.** La celda 21 lo enlaza correctamente ahí (*"Paper: https://arxiv.org/pdf/1610.00087.pdf"*, en la sección de la familia M). Es un copy-paste dentro del propio documento: el enlace correcto de la Parte 1 se reutilizó en la Parte 2.

**El enlace correcto para VGGish es `https://arxiv.org/abs/1609.09430`.** Queda anotado como errata del material del curso.

### 13.2. "VGGish" no aparece en el paper

Cero ocurrencias en el texto completo. El nombre es posterior y lo puso Google al liberar el modelo. Estrictamente: **este paper no presenta VGGish; presenta el trabajo del que VGGish salió.**

### 13.3. El "VGG" del paper no es VGGish

Desarrollado en la Sección 6.1. El paper usa **configuration E (VGG-19)**, 5 bloques, 62M pesos, salida de 3087 unidades. VGGish es una variante de la **configuration A (VGG-11) truncada tras el 4º bloque**, 6 convoluciones, 72.1M parámetros, salida de 128. Citar los números de la fila "VGG" de la Tabla 2 (AUC 0.911, mAP 0.161) como desempeño de VGGish es incorrecto.

### 13.4. Los embeddings de la Sección 4.4 son de ResNet-50, no de VGG

**Esta es la citación errónea más difundida sobre el paper.** El texto es inequívoco:

> *"the second uses the output of the penultimate 'embedding' layer of our **best ResNet model** as inputs"*

Y la mejor red del paper es ResNet-50 (mAP 0.212 en la Tabla 2), no VGG (0.161). Por lo tanto, **la frase "los embeddings de VGGish alcanzan mAP 0.314 en AudioSet" atribuye a VGGish un resultado obtenido con ResNet-50.** El checkpoint VGGish que Google liberó nunca fue evaluado en el paper.

### 13.5. El experimento de AudioSet no es fine-tuning

Se entrena un MLP **sobre embeddings congelados**, sin tocar la red base. No hay warm-starting, no hay descongelamiento progresivo, no hay ajuste de la ResNet sobre AudioSet. Describirlo como "fine-tuning" cambia el sentido del resultado (y lo debilita: la versión congelada es la afirmación más fuerte).

### 13.6. Inconsistencia interna: 5.24M vs 5.4M horas

- Abstract y Sección 1: *"70M training videos (**5.24 million hours**)"*.
- Sección 2: *"Videos average 4.6 minutes each for a total of **5.4M training hours**"*.

$70\times10^6 \times 4.6/60 = 5.37\times10^6$ h, consistente con 5.4M. Los 5.24M implicarían 4.49 min por video. **Las dos cifras no pueden ser ambas exactas.** Es una inconsistencia menor del paper. El valor más citado en la literatura es 5.24M.

### 13.7. Inconsistencia interna: 405 h vs 356 h

- Sección 4.1: *"We include numbers for ResNet after training for 17 million mini-batches (**405 hours**)"*.
- Tabla 2, última fila: ResNet-50, 17M pasos, **356h**.

Verificado visualmente contra el PDF renderizado. Discrepan.

### 13.8. No confundir la última fila de la Tabla 2 con la primera de la Tabla 4

Son corridas distintas y no se deben mezclar:

| Fuente | Configuración | AUC | d′ | mAP |
|---|---|---|---|---|
| Tabla 2, última fila | ResNet-50, 17M pasos, **con reducción de lr** en el paso 13M | 0.926 | 2.041 | 0.212 |
| Tabla 4, primera fila | ResNet-50, 16M pasos, 70M videos, **sin mención de reducción de lr** | 0.923 | 2.019 | 0.206 |

### 13.9. "El paper demuestra que más datos siempre ayudan"

**Lo contrario.** La Tabla 4 muestra saturación: 700K → 70M videos (factor 100) mueve el mAP de 0.203 a 0.206. Es el hallazgo más contraintuitivo del paper y el que menos se cita. Ver Sección 7.1.

### 13.10. "El paper demuestra que más etiquetas ayudan"

Los autores dicen *"weak support"* y *"albeit modestly"*, y la Tabla 3 **no es monótona** (con cuello de botella: 400 → 0.365, 3K → 0.347, 30K → 0.369). Además falta la celda de 30K sin cuello de botella. No es un resultado que soporte peso.

### 13.11. "VGGish fue preentrenado en AudioSet"

Es como lo describe el notebook (celda 57: *"Este modelo en particular fue preentrenado en el dataset AudioSet"*), y es impreciso. El modelo de este paper se entrenó en **YouTube-100M**; AudioSet fue el objetivo de **evaluación / transferencia** (Sección 4.4). La documentación del repositorio oficial describe el checkpoint publicado como entrenado sobre un corpus grande de YouTube, no sobre AudioSet. La relación es la inversa de la que sugiere el notebook: **AudioSet se distribuye con embeddings calculados por VGGish**, y de ahí nace la asociación. *(La procedencia exacta del checkpoint publicado no es verificable contra este PDF; lo marco como tal.)*

### 13.12. "VGGish usa MFCC"

El notebook lo dice dos veces —celda 60 (*"este modelo trabaja con features MFCC en vez del audio puro"*) y el comentario de la celda 65 (`# Transformamos a MFCC --> 3 x 96 x 64`)— y **es incorrecto**. La Sección 3.1 del paper es explícita: **log-mel spectrogram**, no MFCC. La función que el notebook llama, `vggish_input.waveform_to_examples()`, produce log-mel.

**La diferencia importa conceptualmente, no es una imprecisión de vocabulario.** Un MFCC se obtiene aplicando una **DCT** al log-mel y quedándose con los primeros ~13 coeficientes. Esa DCT:

- **Decorrelaciona** las bandas, que era el requisito para los GMM con matrices de covarianza diagonal de la era pre-deep-learning.
- **Destruye la localidad en frecuencia.** Cada cepstral coefficient es una combinación lineal de **todas** las bandas mel. Un kernel convolucional $3\times3$ sobre MFCCs opera sobre coeficientes cepstrales adyacentes, que no tienen ninguna relación de vecindad significativa.

Es decir: **la DCT elimina exactamente la estructura que hace que una CNN 2D tenga sentido.** Que VGGish use log-mel y no MFCC no es un detalle; es un requisito de la tesis del paper. Un espectrograma es una imagen; un cepstrograma no lo es.

### 13.13. Otros matices del notebook

- **"alrededor de 1 segundo de audio"** (celda 60): son **0.96 s** exactos. La diferencia importa para la aritmética de los 3 parches por clip de 3 s: $\lfloor (3.0 - 0.96)/0.96 \rfloor + 1 = 3$, con 0.12 s descartados.
- **"no entrenaremos por más de 2 épocas"** (celda 60) frente a **`n_epochs = 3`** (celda 74). Discrepan.
- **El relleno puede truncar.** `zero_need = rate*3 - n`; si el clip dura más de 3 s, `zero_need` es negativo y `F.pad` con valores negativos **recorta**. En UrbanSound8K, donde muchos clips duran exactamente 4 s, eso descarta 1 s de audio por clip. Además, `zero_need // 2` en ambos lados deja la señal 1 muestra corta cuando `zero_need` es impar.
- **`num_workers: 0`** con preprocesamiento NumPy en `__getitem__` es el origen real de la hora y media de entrenamiento.

### 13.14. Un matiz sobre la Figura 1

La lectura correcta es "**d′** es aproximadamente constante en el prior", no "el desempeño es constante". El **mAP** de las clases raras es bajísimo, y el propio paper explica por qué en la Sección 3.2: el AP está directamente correlacionado con el prior de la clase. Las dos métricas cuentan historias opuestas sobre las mismas predicciones, y ambas son correctas.

---

## 14. Cómo se ve hoy

Código PyTorch comentado, con las formas anotadas en cada paso. El objetivo es doble: replicar el front-end exacto de VGGish, y exponer los tres puntos donde los valores por defecto de `torchaudio` difieren de la implementación de referencia (que es donde falla el 90 % de las reimplementaciones).

### 14.1. Del WAV al tensor $N \times 96 \times 64$

```python
import torch
import torchaudio
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constantes canónicas de VGGish (vggish_params.py del repo oficial de AudioSet).
# NOTA: el paper (Sec. 3.1) solo especifica 25 ms / 10 ms / 64 bandas mel /
# parches de 96x64 / "un pequeño offset". La frecuencia de muestreo, el rango
# [125, 7500] Hz y el valor 0.01 del offset vienen del CODIGO publicado, no del PDF.
# ---------------------------------------------------------------------------
SAMPLE_RATE   = 16_000
WIN_SECONDS   = 0.025          # ventana STFT
HOP_SECONDS   = 0.010          # salto STFT -> 100 tramas por segundo
N_MELS        = 64
F_MIN, F_MAX  = 125.0, 7500.0
LOG_OFFSET    = 0.01           # log(mel + 0.01); fija el piso en ~-4.6
EXAMPLE_FRAMES = 96            # 96 tramas * 10 ms = 960 ms de avance

WIN_LENGTH = int(round(SAMPLE_RATE * WIN_SECONDS))   # 400 muestras
HOP_LENGTH = int(round(SAMPLE_RATE * HOP_SECONDS))   # 160 muestras
N_FFT      = 2 ** int(torch.ceil(torch.log2(torch.tensor(float(WIN_LENGTH)))))  # 512


# ---------------------------------------------------------------------------
# Los tres defaults de torchaudio que hay que cambiar. Si no los cambias,
# el tensor tendra la forma correcta y valores incorrectos, y el checkpoint
# preentrenado operara sobre una distribucion que nunca vio.
#
#   1. power=1.0   -> VGGish usa MAGNITUD (np.abs), no potencia. torchaudio
#                     usa power=2.0 por defecto.
#   2. norm=None   -> VGGish usa triangulos con pico unitario (estilo HTK).
#                     torchaudio usa norm='slaney' (normalizacion por area).
#   3. center=False-> VGGish enmarca sin padding. torchaudio usa center=True,
#                     lo que agrega n_fft//2 de padding reflejado y desplaza
#                     todas las tramas en 16 ms.
#
# Ademas mel_scale='htk' porque VGGish usa m = 1127*ln(1 + f/700).
# ---------------------------------------------------------------------------
_melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=int(N_FFT),          # 512
    win_length=WIN_LENGTH,     # 400
    hop_length=HOP_LENGTH,     # 160
    f_min=F_MIN,
    f_max=F_MAX,
    n_mels=N_MELS,
    power=1.0,                 # (1) magnitud
    norm=None,                 # (2) triangulos con pico unitario
    mel_scale="htk",           # 1127 * ln(1 + f/700)
    center=False,              # (3) sin padding
    window_fn=torch.hann_window,
)


def wav_to_vggish_input(path: str) -> torch.Tensor:
    """WAV -> (N, 96, 64) float32, listo para VGGish.

    N = numero de ejemplos de 960 ms que caben en el archivo.
    """
    # ---- 1. Cargar y llevar a mono ---------------------------------------
    wav, sr = torchaudio.load(path)              # (C, T_orig)
    wav = wav.mean(dim=0, keepdim=True)          # (1, T_orig)  mezcla a mono

    # ---- 2. Remuestrear a 16 kHz -----------------------------------------
    # Obligatorio: el banco mel esta definido sobre esta tasa. Alimentar
    # 44.1 kHz sin remuestrear desplaza TODO el eje de frecuencia.
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    # wav: (1, T)  con T = duracion_segundos * 16000

    # ---- 3. Espectrograma mel de magnitud ---------------------------------
    mel = _melspec(wav)                          # (1, 64, n_frames)
    # n_frames = 1 + (T - 400) // 160

    # ---- 4. Compresion logaritmica ---------------------------------------
    # log(mel + 0.01): comprime ~80 dB de rango dinamico y convierte los
    # cambios de ganancia (multiplicativos) en offsets aditivos, que la
    # BatchNorm de la red absorbe.
    log_mel = torch.log(mel + LOG_OFFSET)        # (1, 64, n_frames)

    # ---- 5. Reordenar a (tiempo, frecuencia) ------------------------------
    log_mel = log_mel.squeeze(0).transpose(0, 1) # (n_frames, 64)

    # ---- 6. Trocear en ejemplos de 96 tramas, SIN solapamiento ------------
    # Nota sutil: 96 tramas cubren 96*160 + 240 = 15600 muestras (~975 ms).
    # Los "960 ms" del paper son el AVANCE entre ejemplos, no el soporte.
    n_frames = log_mel.shape[0]
    n_examples = 1 + (n_frames - EXAMPLE_FRAMES) // EXAMPLE_FRAMES
    if n_examples < 1:
        # Audio mas corto que 960 ms: rellenar con el piso logaritmico,
        # NO con ceros (0.0 en el dominio log significa mel = 1 - 0.01,
        # es decir energia alta, no silencio).
        pad = EXAMPLE_FRAMES - n_frames
        floor = torch.log(torch.tensor(LOG_OFFSET))
        log_mel = F.pad(log_mel, (0, 0, 0, pad), value=float(floor))
        n_examples = 1

    usable = n_examples * EXAMPLE_FRAMES
    examples = log_mel[:usable].reshape(n_examples, EXAMPLE_FRAMES, N_MELS)
    return examples                               # (N, 96, 64)
```

Sobre el punto 6: rellenar con ceros en el dominio logarítmico es un error frecuente y silencioso. Un cero en log-mel corresponde a $\text{mel} = 1 - \epsilon$, que es **energía apreciable**, no silencio. El piso real es $\log(\epsilon) = \log(0.01) \approx -4.6$. Rellenar con ceros inyecta ruido de banda ancha donde debería haber silencio. (El notebook del laboratorio rellena la **forma de onda** con ceros antes del front-end, que sí es correcto: cero en el dominio del tiempo es silencio de verdad.)

### 14.2. Esqueleto de la red, con formas anotadas

```python
import torch.nn as nn


class VGGish(nn.Module):
    """VGGish: variante truncada de VGG (config. A) para log-mel de 96x64.

    NOTA HISTORICA: esta NO es la red de la fila "VGG" de la Tabla 2 del
    paper de Hershey et al., que es la configuracion E (VGG-19, 5 bloques,
    62M pesos, salida de 3087 unidades). Esta es la red que Google libero
    despues, con 4 bloques y un cuello de botella de 128 dimensiones.
    """

    def __init__(self, n_classes: int | None = None):
        super().__init__()

        # ---- Pila convolucional: 4.5M parametros (6 %), ~796M MACs (92 %) --
        self.features = nn.Sequential(
            # entrada: (B, 1, 96, 64)
            nn.Conv2d(1, 64, 3, stride=1, padding=1),    # (B,  64, 96, 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                   # (B,  64, 48, 32)

            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # (B, 128, 48, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                   # (B, 128, 24, 16)

            nn.Conv2d(128, 256, 3, stride=1, padding=1), # (B, 256, 24, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), # (B, 256, 24, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                   # (B, 256, 12,  8)

            nn.Conv2d(256, 512, 3, stride=1, padding=1), # (B, 512, 12,  8)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), # (B, 512, 12,  8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                   # (B, 512,  6,  4)
        )
        # Campo receptivo de la ultima capa conv: 70 unidades en cada eje.
        # Como el eje de frecuencia tiene solo 64 bandas, la ultima capa
        # ve el espectro COMPLETO -> la estructura armonica global es
        # accesible, pero solo tras seis capas de composicion.

        # ---- Cabeza densa: 67.6M parametros (94 %), ~67.6M MACs (8 %) -----
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 4, 4096),   # 12288 -> 4096 : 50.3M params (70 % del modelo)
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),          #                 16.8M params
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),           #                  0.5M params -> EMBEDDING
            nn.ReLU(inplace=True),          # el embedding es POST-ReLU: no negativo
        )

        # Cabeza de clasificacion opcional, SEPARADA del bloque de embedding.
        # Asi se evita el bug de dejar una ReLU despues de los logits.
        self.classifier = nn.Linear(128, n_classes) if n_classes else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 96, 64)
        x = self.features(x)                       # (B, 512, 6, 4)

        # El checkpoint original viene de TensorFlow, que usa NHWC. El
        # aplanado tiene que respetar ese orden o los 12288 pesos de la
        # primera densa quedan permutados y el modelo produce basura
        # silenciosamente.
        x = x.permute(0, 2, 3, 1).contiguous()     # (B, 6, 4, 512)  -> NHWC
        x = x.flatten(1)                           # (B, 12288)

        emb = self.fc(x)                           # (B, 128)  embedding
        if self.classifier is None:
            return emb
        return self.classifier(emb)                # (B, n_classes)  logits limpios


# ---------------------------------------------------------------------------
# Auditoria de parametros: reproduce el desglose de la Seccion 12.2.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = VGGish()
    conv = sum(p.numel() for p in net.features.parameters())
    dense = sum(p.numel() for p in net.fc.parameters())
    print(f"convolucional : {conv:>12,}  ({conv/(conv+dense):6.1%})")
    print(f"densa         : {dense:>12,}  ({dense/(conv+dense):6.1%})")
    print(f"total         : {conv+dense:>12,}")
    # convolucional :    4,499,712  (  6.2%)
    # densa         :   67,641,472  ( 93.8%)
    # total         :   72,141,184
```

### 14.3. Post-procesamiento de embeddings (formato AudioSet)

Este paso **no está en el paper**; es el formato de distribución de AudioSet. Lo incluyo porque explica cómo se ven los embeddings que la comunidad realmente consume.

```python
import numpy as np

class VGGishPostprocessor:
    """PCA con blanqueo + cuantizacion a 8 bits.

    Orden importante: el embedding sale de una ReLU, asi que es no negativo
    y tiene varianzas muy dispares entre ejes. Cuantizar directo desperdicia
    casi todos los niveles. El blanqueo IGUALA las varianzas y recien
    entonces una cuantizacion uniforme de 8 bits es razonable para todos
    los ejes por igual.

    Resultado: 128 bytes por segundo de audio.
      - PCM 16 kHz/16 bits : 32,000 B/s
      - embedding float32  :    512 B/s
      - embedding uint8    :    128 B/s   -> 250x menos que el audio crudo
    Es la razon por la que AudioSet puede distribuirse: Google no puede
    redistribuir el audio de YouTube, pero si un descriptor de 128 B/s.
    """
    QUANTIZE_MIN, QUANTIZE_MAX = -2.0, +2.0

    def __init__(self, pca_matrix: np.ndarray, pca_means: np.ndarray):
        self.pca_matrix = pca_matrix   # (128, 128)
        self.pca_means = pca_means     # (128, 1)

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        # embeddings: (N, 128) float32, post-ReLU (no negativos)
        pca = np.dot(self.pca_matrix, embeddings.T - self.pca_means).T   # (N, 128)

        clipped = np.clip(pca, self.QUANTIZE_MIN, self.QUANTIZE_MAX)
        scale = 255.0 / (self.QUANTIZE_MAX - self.QUANTIZE_MIN)
        quantized = (clipped - self.QUANTIZE_MIN) * scale
        return quantized.astype(np.uint8)                                # (N, 128)


# Gotcha para busqueda por similitud (relevante si los usas como blocker en
# un indice vectorial): NO calcules coseno sobre los uint8 crudos. Los
# embeddings originales son no negativos (viven en el ortante positivo, lo
# que comprime el rango de angulos posibles) y la cuantizacion introduce un
# sesgo sistematico. Des-cuantiza primero, y si tienes la matriz PCA,
# considera revertir el blanqueo antes de medir distancias.
```

### 14.4. Nota final de reproducibilidad

Esta implementación **no es bit-exacta** respecto del código original. Las diferencias, todas pequeñas, están en: la construcción del banco de filtros mel (el código de referencia calcula los bordes de banda sobre `n_mels + 2` puntos y anula explícitamente el bin de DC, mientras `torchaudio.melscale_fbanks` tiene su propio manejo de bordes); el tratamiento de la ventana de Hann (periódica frente a simétrica); y el redondeo del número de tramas. Para inferencia con el checkpoint preentrenado la diferencia es despreciable, pero **si estás comparando contra números publicados, usa `vggish_input.waveform_to_examples()` del repositorio oficial** — que es, correctamente, lo que hace el notebook del laboratorio.

---

## Apéndice: mapa rápido de dónde está cada cifra

| Cifra | Ubicación en el paper |
|---|---|
| 70M videos de entrenamiento, 5.24M horas, 30,871 etiquetas | Abstract, Sec. 1, Sec. 2 |
| 100M videos totales: 70M train / 10M eval / 20M validación, 4.6 min de promedio | Sec. 2 |
| ~5 etiquetas por video; etiquetas del Knowledge Graph; asignación automática | Sec. 2 |
| Ejemplos de tabla de priors (Song, Cormorant, …) | Tabla 1 |
| 960 ms, STFT 25 ms / 10 ms, 64 bandas mel, log con offset, parches 96×64, batch 128 | Sec. 3.1 |
| Sigmoide (no softmax), cross-entropy, Adam, BatchNorm, sin regularización | Sec. 3.1 |
| Conjuntos de evaluación balanceados: 1M / 100K / 12K videos, ~33 por clase | Sec. 3.2 |
| $d' = \sqrt{2}\,F^{-1}(\text{AUC})$; mAP correlaciona con el prior | Sec. 3.2 y nota al pie 1 |
| MLP: 3 capas × 1000, lr $3\times10^{-5}$, 11.2M pesos, 10 GPUs | Sec. 3.3.1 |
| AlexNet: stride $2\times1$, 3087 salidas, 37.3M pesos / 767M mult., 20 GPUs | Sec. 3.3.2 |
| VGG config. E: solo cambia la capa final; 62M pesos / 2.4B mult., 10 GPUs | Sec. 3.3.3 |
| Inception-V3: stem truncado, sin red auxiliar, AvgPool $10\times6$; 28M / 4.7B, 40 GPUs | Sec. 3.3.4 |
| ResNet-50: sin stride 2 inicial, AvgPool $6\times4$; 30M / 1.9B, 20 GPUs | Sec. 3.3.5 |
| Comparación de arquitecturas (AUC / d′ / mAP / horas) | **Tabla 2** |
| d′ plano frente al prior de clase | **Figura 1** y Sec. 4.1 |
| Tamaño del vocabulario y cuello de botella de 128 unidades; 30M → 80M pesos | **Tabla 3** y Sec. 4.2 |
| Tamaño del conjunto de entrenamiento (23K → 70M) | **Tabla 4** y Sec. 4.3 |
| 11 mini-batches/s, 23 semanas por época, 14 h para ver un frame de cada video | Sec. 4.3 |
| AudioSet: >1M clips de 10 s, ~3000 h, ≈0.05 % de YouTube-100M | Sec. 4.4 |
| Baseline log-mel $64\times20$: 0.137 / 0.904 / 1.846 | Sec. 4.4 |
| Embeddings de ResNet: 0.314 / 0.959 / 2.452 | Sec. 4.4 |
| Ejemplo cualitativo (Trumpet / Piano / Guitar) | **Figura 2** |
| Reconocimiento de la tensión "audio-specific architectures" | Sec. 1 |
| Imposibilidad de cuantificar la debilidad de las etiquetas | Sec. 1 |
| Agregación por promedio, imitando a Ng et al. | Sec. 1 y Sec. 3.2 |
