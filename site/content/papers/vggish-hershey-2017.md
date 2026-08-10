---
title: "VGGish: CNN Architectures for Large-Scale Audio Classification (2017)"
weight: 428
math: true
---

{{< paper-card
    title="CNN Architectures for Large-Scale Audio Classification"
    authors="Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis, Jort F. Gemmeke, Aren Jansen, R. Channing Moore, Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, Malcolm Slaney, Ron J. Weiss, Kevin Wilson (Google)"
    year="2017"
    venue="ICASSP 2017 / arXiv:1609.09430"
    pdf="/papers/vggish-hershey-2017.pdf" >}}
En 2016 la clasificación de imágenes tenía **ImageNet** y el audio no tenía nada equivalente. Google tenía un corpus interno —**YouTube-100M**: 70 millones de videos de entrenamiento, 5.24 millones de horas, 30 871 etiquetas del Knowledge Graph asignadas **automáticamente y a nivel de video**— y decidió hacer con él el experimento más limpio posible: tratar el espectrograma **log-mel** como si fuera una imagen y aplicarle, **casi sin modificar**, las arquitecturas de visión ya calibradas (AlexNet, VGG, Inception-V3, ResNet-50), midiendo qué gana la escala. El resultado es doble. Primero, funcionan: todas las CNN triplican el mAP del baseline totalmente conectado ($0.058 \to 0.182$) y **el ranking de ImageNet se transfiere intacto** (AlexNet < VGG < Inception ≈ ResNet). Segundo, y más duradero: las activaciones de la penúltima capa, usadas como *features congeladas*, llevan el mAP en [AudioSet](/papers/audioset-gemmeke-2017) de **0.137 a 0.314** frente a alimentar log-mel crudo — un factor $\times 2.29$ que legitimó el paradigma del "modelo de audio preentrenado". De este trabajo salió el checkpoint que Google liberó después con el nombre **VGGish**, cuyo formato de entrada ($96 \times 64$ log-mel) y cuyo embedding de 128 dimensiones fueron el estándar de facto del audio durante media década. Y contiene un resultado que casi nadie cita: la Tabla 4 muestra que **700 K videos rinden 98.5 % de lo que rinden 70 M**. Los 5.24 millones de horas del titular no son lo que hizo funcionar al modelo.
{{< /paper-card >}}

---

## Contexto: el audio no tenía un ImageNet

La primera frase de la introducción es programática: *"Image classification performance has improved greatly with the advent of large datasets such as ImageNet"*. El paper se escribe desde una asimetría concreta.

**El lado de la visión.** Para 2016, ImageNet llevaba siete años disponible: 1.2 M imágenes etiquetadas a mano en 1000 clases para ILSVRC, sobre una base de ~14 M. Esa base hizo posibles AlexNet, VGG, Inception y ResNet, y —más importante— hizo posible el **preentrenamiento**: en visión, para 2016, ya nadie entrenaba desde cero un modelo para una tarea pequeña.

**El lado del audio.** La Sección 1 enumera los datasets disponibles como quien enumera un problema —TRECVid, ActivityNet, Sports-1M, TUT/DCASE Acoustic Scenes 2016— y los liquida con una frase: *"which are much smaller than YouTube-100M"*. DCASE 2016 trabajaba con del orden de mil segmentos de 30 s en 15 clases. No había nada comparable a ImageNet, y la consecuencia práctica es que la detección de eventos acústicos seguía anclada en **MFCC + GMM / HMM / NMF / SVM**.

**Qué era YouTube-100M.**

| Propiedad | Valor |
|---|---|
| Videos totales | 100 millones |
| Entrenamiento / evaluación / validación | **70 M** / 10 M / 20 M |
| Duración media por video | 4.6 minutos |
| Horas de entrenamiento | **5.24 M** (abstract) / 5.4 M (Sec. 2) — ver erratas |
| Vocabulario de etiquetas | **30 871** identificadores del Knowledge Graph |
| Etiquetas por video | ~5 en promedio |
| Ejemplos de 960 ms derivados | ~**20 mil millones** |

La aritmética cierra: $5.24\times10^6\,\text{h} \times 3600 \div 0.96 \approx 1.96\times10^{10}$, los "around 20 billion" de la Sección 4.3.

### Por qué las etiquetas son débiles y ruidosas

Hay tres capas de degradación, y conviene separarlas porque no son la misma cosa.

**1. Son automáticas, no humanas.** Sección 2: *"The labels are assigned automatically based on a combination of metadata (title, description, comments, etc.), context, and **image content** for each video."* Parte de las etiquetas provienen del canal **visual**: el modelo de audio se está entrenando, en parte, para predecir lo que dice la imagen. Eso es un sesgo estructural, no ruido aleatorio, y la escala no lo promedia.

**2. Son a nivel de video, no de segmento.** Sección 3.1: *"Each frame inherits all the labels of its parent video."* Un video de 4.6 minutos etiquetado "Trumpet" produce ~287 ejemplos de 960 ms, y en la enorme mayoría de ellos **no suena una trompeta**. Es supervisión débil en el sentido técnico de *Multiple Instance Learning*: la etiqueta es una propiedad de la bolsa, no de la instancia.

**3. Muchas no son acústicamente relevantes.** El paper lo dice sin defenderse: *"of the 30K labels, some are clearly acoustically relevant ('Trumpet') and others are less so ('Web Page')"*. Los priors abarcan cinco órdenes de magnitud, de "Song" y "Music" ($10^{-1}$) a "Cormorant" y "Lecturer" ($10^{-6}$), sin jerarquía impuesta. La honestidad del paper merece citarse: *"We are not able to quantify how 'weak' the labels are […] and for the majority of classes (e.g., 'Computer Hardware', 'Boeing 757', 'Ollie'), it's not clear how to decide relevance."* Con el contrapunto correcto: para una clase como "Beach" el **ambiente de fondo es la señal**, así que "no informativo" ni siquiera está bien definido.

{{< concept-alert type="clave" >}}
La estrategia frente al ruido de etiqueta **no es algorítmica, es de escala**. Frente a Kumar y Raj, que formalizan el problema como MIL, los autores declaran: *"we are investigating the limits of training with weak labels for very large datasets […] we hope that, given enough training, the net can learn to spot useful cues."* La apuesta es que el ruido **no correlacionado** con el audio actúa como ruido de gradiente y se promedia. Lo que **sí** está correlacionado —el sesgo del canal visual— no se promedia y queda dentro del modelo.
{{< /concept-alert >}}

### La relación con AudioSet

[AudioSet](/papers/audioset-gemmeke-2017) (Gemmeke et al., **mismo ICASSP 2017**, con cinco coautores compartidos) es la respuesta complementaria al mismo problema: en lugar de aceptar el ruido y compensarlo con escala, **paga anotación humana** sobre una ontología de eventos acústicos. Sección 4.4: *"a dataset of over 1 million 10 second excerpts […] This comes to about 3000 hours — still only $\approx 0.05\%$ of YouTube-100M."*

No son dos líneas de investigación paralelas, son las dos mitades de un mismo esfuerzo: **YouTube-100M es el corpus de preentrenamiento sucio y masivo; AudioSet es el benchmark limpio y pequeño.** Este paper es el que demuestra que el primero sirve para el segundo. Y el círculo se cierra en un detalle de infraestructura: AudioSet no se distribuye como audio —Google no puede redistribuir video de YouTube por derechos— sino **como embeddings**, y esos embeddings son la salida de la red que aquí se describe.

## La pregunta del paper: medir, no inventar

Este no es un paper de arquitectura. Es un paper de **medición**, y conviene ser explícito porque cambia cómo se lee cada tabla. La pregunta declarada tiene tres partes: *"how popular DNN architectures compare on video soundtrack classification; how performance varies with different training set and label vocabulary sizes; and whether our trained models can also be useful for AED."*

Y la decisión metodológica clave está en una sola frase de la Sección 1, que es literalmente el tema de la [Clase 39](/clases/clase-39):

> *"Although the distinct meanings of time and frequency axes might argue for audio-specific architectures, this work employs **minimally-altered** image classification networks such as Inception-V3 and ResNet-50."*

**Por qué "no inventar arquitectura" es la elección correcta**, y ninguna de las razones es pereza:

1. **Aísla la variable de interés.** Si además de escalar los datos se cambiara la arquitectura, no se podría atribuir la mejora a nada en particular. Al congelar la arquitectura en modelos que la comunidad ya calibró exhaustivamente, la única variable libre es la escala. Es diseño experimental, no atajo.
2. **Prueba una hipótesis falsable y no trivial.** La hipótesis nula razonable en 2016 era *"las arquitecturas de visión fracasarán en espectrogramas porque los ejes no son intercambiables"*: una predicción concreta con fundamento físico. El paper la somete a prueba de la forma más limpia posible, aplicando las redes tal cual. Si las hubieran modificado para adaptarlas al audio, el resultado no diría nada sobre la hipótesis.
3. **Es la pregunta con mayor valor de opción.** Si la respuesta es sí, todo el aparato de investigación de visión —arquitecturas, inicializaciones, optimizadores, intuiciones— se vuelve importable a audio sin costo. Es exactamente lo que ocurrió: cinco años después, [AST](/papers/ast-gong-2021) importa un ViT **con sus pesos de ImageNet**.

**Qué hace de esto un experimento controlado.** Los cinco modelos se entrenan con el mismo dataset, el mismo vocabulario (3 K etiquetas), la misma entrada, la misma pérdida, el mismo optimizador (Adam), **sin regularización de ningún tipo**, y se comparan tras el mismo número de pasos (5 M mini-batches de 128). La Sección 4.1 reconoce el punto débil: comparan a **pasos fijos**, no a **cómputo fijo**. La tabla incluye la columna de horas justamente para que el lector corrija — y la corrección importa: VGG consumió 184 h contra 119 h de ResNet-50 para quedar 0.021 de mAP por debajo.

**Lo único que no controlaron y podían haber controlado:** no hay ninguna arquitectura *específica de audio* en la comparación. Ni CRNN, ni modelo sobre forma de onda cruda, ni filtros con campo receptivo alargado en frecuencia. La conclusión legítima es **"las arquitecturas de visión funcionan bien"**, no **"las arquitecturas de visión son la mejor opción"**. Es una distinción que la literatura posterior borró bastante rápido.

## La representación de entrada: el formato que hoy se llama VGGish

Esta es la parte más citada del paper en la práctica, porque define un formato que se volvió estándar de facto. Y es donde hay que ser más cuidadoso con qué dice el paper y qué no.

### Lo que el paper especifica (Sección 3.1)

> *"The audio is divided into non-overlapping 960 ms frames. […] The 960 ms frames are decomposed with a short-time Fourier transform applying 25 ms windows every 10 ms. The resulting spectrogram is integrated into 64 mel-spaced frequency bins, and the magnitude of each bin is log-transformed after adding a small offset to avoid numerical issues. This gives log-mel spectrogram patches of $96 \times 64$ bins that form the input to all classifiers."*

Eso es **todo**. Nota lo que **no** dice: no especifica la frecuencia de muestreo, ni el rango de la escala mel, ni el valor del offset, ni el tipo de ventana. El "16 kHz mono" y el "125–7500 Hz" **no están en este PDF**: vienen del código publicado (`vggish_params.py`). El paper define la **forma** del tensor; el repositorio define los **valores**.

### Los parámetros canónicos del pipeline

| Parámetro | Valor | Nota |
|---|---|---|
| Frecuencia de muestreo | **16 kHz, mono** | Nyquist a 8 kHz |
| Ventana STFT | **25 ms** (400 muestras) | heredado de ASR |
| Salto (hop) | **10 ms** (160 muestras) | 100 tramas/s |
| Tamaño de FFT | 512 | → 257 bins |
| Espectro | **magnitud**, no potencia | el código toma `np.abs(fft)` |
| Bandas mel | **64** | |
| Rango mel | **125 – 7500 Hz** | |
| Compresión | $\log(\text{mel} + 0.01)$ | offset de estabilización |
| Ventana de ejemplo | **96 tramas = 960 ms** | sin solapamiento |

### Por qué cada elección, y qué se pierde

**16 kHz** descarta todo por sobre 8 kHz. Para voz y eventos ambientales eso es casi gratis; para música (platillos, aire de vientos, brillo percibido) es una pérdida real. También es decisión de costo: a 16 kHz el corpus ocupa la mitad que a 32 kHz. PANNs (2020) entrena a 32 kHz y reporta que ayuda.

**Ventana de 25 ms, salto de 10 ms** es el punto elegido en el compromiso de Gabor, $\Delta t \cdot \Delta f \gtrsim 1/(4\pi)$. Con 25 ms la resolución en frecuencia es del orden de $1/0.025 = 40$ Hz: los armónicos de una voz masculina ($f_0 \approx 110$ Hz) se resuelven, pero las diferencias de afinación por debajo de 40 Hz se pierden. En el otro sentido, 25 ms es corto para que la señal sea aproximadamente estacionaria dentro de la ventana —premisa de toda la STFT— pero suficientemente largo para **emborronar transitorios**: un chasquido de 2 ms se distribuye sobre toda la ventana.

**64 bandas mel entre 125 y 7500 Hz** son tres decisiones en una. La reducción $257 \to 64$ es un factor 4 de compresión que **elimina la estructura armónica fina y conserva la envolvente espectral** (el timbre): para clasificación de eventos eso es lo que importa, y para transcripción musical o estimación de pitch es exactamente lo que habría que preservar. La escala mel, $m = 1127\,\ln(1 + f/700)$, comprime las altas y expande las bajas imitando la resolución del oído (ver [MFCC y escala mel](/fundamentos/mfcc-y-escala-mel)). Y el **recorte inferior en 125 Hz** elimina el DC, la red eléctrica y el ruido de manipulación del micrófono, pero también **elimina la fundamental de las voces masculinas graves** (85–155 Hz) y buena parte del rango de un bajo o un bombo: el modelo tiene que inferirlos por sus armónicos superiores. El recorte superior en 7500 Hz deja margen bajo Nyquist para el rolloff del antialiasing.

**El logaritmo con offset** cumple dos funciones. Primero, **compresión de rango dinámico**: el audio natural cubre 80–100 dB, y el log convierte factores multiplicativos en desplazamientos aditivos —un cambio de volumen se vuelve un sesgo constante que la BatchNorm absorbe. Segundo, es el análogo perceptual de la ley de Weber-Fechner. El **offset** es indispensable ($\log 0 = -\infty$) y hace algo más: con $\epsilon = 0.01$ el piso queda en $\log(0.01) \approx -4.6$, de modo que **el offset es el control de rango dinámico del front-end, disfrazado de truco numérico**.

**Lo que el log-mel descarta y no se recupera:** la **fase** (la representación deja de ser invertible — de ahí Griffin-Lim y los vocoders neuronales), la resolución temporal sub-10 ms, la resolución sub-banda mel y todo lo que está fuera de [125, 7500] Hz.

{{< concept-alert type="advertencia" >}}
**"96 tramas de 10 ms = 960 ms" es cierto solo si cuentas los saltos, no las ventanas.** El código no enmarca un audio aislado de 0.96 s (eso daría 94 tramas): calcula el log-mel de **toda** la señal y luego lo corta en bloques de 96 tramas con salto de 96, de modo que cada bloque consume $96\times160 + 240 = 15\,600$ muestras ≈ 975 ms. La etiqueta "960 ms" refiere al **avance**, no al soporte.
{{< /concept-alert >}}

El paper no justifica por qué 960 ms. La lectura razonable: es el número que da un parche de tamaño y aspect ratio cómodos para una CNN de visión, y que además es aproximadamente la escala de un evento acústico completo (una palabra, un ladrido, un bocinazo). Es la unidad mínima de decisión del sistema y define su límite de resolución.

## Las arquitecturas comparadas

Todas reciben el mismo tensor: un parche log-mel de $96\times64$, **un solo canal**. Todas terminan en una capa **sigmoide** de 3087 unidades (no softmax, porque cada ejemplo lleva múltiples etiquetas). Todas usan **BatchNorm después de cada convolución**, reemplazando la LRN original. Ninguna usa dropout ni weight decay: *"In view of the large training set size, we did not use dropout, weight decay, or other common regularization techniques. For the models trained on 7M or more examples, we saw no evidence of overfitting."*

### Qué hubo que modificar para pasar de RGB a un canal

- **Baseline totalmente conectado.** Barrido sobre $N \in \{2..6\}$ capas y $M \in \{500..4000\}$ unidades; gana $N=3$, $M=1000$. **11.2 M pesos y 11.2 M multiplicaciones** — en una red densa ambos números coinciden, que es justamente lo que la distingue de una CNN: cero reutilización de parámetros.
- **AlexNet.** El stride de la primera capa baja de **4 a $2\times1$**: *"Because our inputs are $96\times64$, we use a stride of $2\times1$ so that the number of activations are similar after the initial layer."* Es un stride **anisotrópico**: 2 en tiempo, **1 en frecuencia**. Aceptan submuestrear el tiempo pero **no** la frecuencia, y esa asimetría es la primera admisión implícita de que los ejes no son equivalentes. Además: BatchNorm en vez de LRN, salida 1000 → 3087, sin la división de filtros entre dispositivos de 2012.
- **VGG.** *"The only changes we made to VGG (**configuration E**) were to the final layer […] as well as the use of batch normalization instead of LRN."* Configuration E es **VGG-19**. Probaron reducir los strides iniciales como en AlexNet y **empeoró**.
- **Inception-V3.** La cirugía más agresiva: se **eliminan las primeras cuatro capas del stem** (incluido el MaxPool), diseñado para bajar de $299\times299$ a $35\times35$ y catastrófico sobre $96\times64$; se elimina la red auxiliar; el AvgPool final pasa a $10\times6$. Conservar el stem quitando el stride inicial funcionó **peor** que truncarlo.
- **ResNet-50.** Se elimina el **stride 2 de la primera convolución $7\times7$**, *"so that the number of activations was not too different in the audio version"*, y el AvgPool final pasa a $6\times4$.

**Qué se mantuvo idéntico:** los tamaños de kernel (nadie alargó un filtro en frecuencia), la topología de bloques, el pooling intermedio, el número de capas densas, la progresión de canales. Todo lo que se tocó fue (i) el stride/pooling de los extremos, para que la resolución no colapse en una entrada 5–10× más chica; (ii) el número de clases; (iii) sigmoide en vez de softmax; (iv) BatchNorm en vez de LRN. **Ninguna de esas cuatro cosas es una adaptación al audio**: son adaptaciones al tamaño de la entrada y al problema multi-etiqueta. La adaptación al audio genuina es exactamente **cero**.

### Resultados (Tabla 2)

70 M videos, 3 K etiquetas, evaluación sobre 100 K videos balanceados. La columna de pesos originales viene de las secciones 3.3.x.

| Arquitectura | Pasos | Horas | Pesos (audio) | Mult. (audio) | Pesos (original) | AUC | d′ | mAP |
|---|---|---|---|---|---|---|---|---|
| Fully Connected (3×1000) | 5 M | 35 h | 11.2 M | 11.2 M | — | 0.851 | 1.471 | 0.058 |
| AlexNet | 5 M | 82 h | 37.3 M | 767 M | 62.4 M | 0.894 | 1.764 | 0.115 |
| VGG (config. E) | 5 M | 184 h | 62 M | 2.4 B | 144 M | 0.911 | 1.909 | 0.161 |
| Inception-V3 | 5 M | 137 h | 28 M | 4.7 B | 27 M | **0.918** | **1.969** | 0.181 |
| ResNet-50 | 5 M | 119 h | 30 M | 1.9 B | 26 M | 0.916 | 1.952 | **0.182** |
| ResNet-50 (largo) | 17 M | 356 h | 30 M | 1.9 B | 26 M | **0.926** | **2.041** | **0.212** |

Cuatro lecturas que la tabla soporta:

1. **Todas las CNN superan al MLP, y por mucho.** $0.058 \to 0.182$ es un factor $\times 3.1$. La convolución no es aquí un detalle de eficiencia: es la diferencia entre funcionar y no funcionar. La justificación del paper es la esperada —*"their convolutional units can efficiently capture common structures that may occur in different areas of the input array"*— con un *"we infer"* que conviene notar: no lo demuestran.
2. **El ordenamiento replica el de ImageNet.** AlexNet < VGG < Inception ≈ ResNet. Eso, en sí mismo, es el hallazgo: **el ranking arquitectónico de visión se transfiere al audio**. Si el espectrograma fuera un dominio genuinamente ajeno, no habría razón para que el orden se preservara.
3. **Ni los pesos ni el cómputo predicen el desempeño.** VGG tiene 62 M pesos y saca 0.161; ResNet-50 tiene 30 M y saca 0.182 en 65 % del tiempo. Inception-V3 gasta 4.7 B multiplicaciones contra 1.9 B de ResNet-50 para empatar. **ResNet-50 domina la frontera de Pareto**, y por eso es la que usan en todo el resto del paper.
4. **5 M pasos no es convergencia.** ResNet-50 pasa de 0.182 a 0.212 (+16 % relativo) con 3.4× más pasos. Es decir, **toda la Tabla 2 está medida antes de converger**, y las diferencias entre Inception y ResNet están dentro de ese margen.

{{< concept-alert type="recordar" >}}
**Un mAP de 0.212 no es un modelo malo.** La Sección 3.2 lo explica: *"unlike AUC, [AP] is directly correlated with the prior probability of the class. Because most of our classes have very low priors ($<10^{-4}$), the mAPs we report are typically small, even though the false alarm rates are good."* La Figura 1 lo confirma: la mediana de d′ por clase **se mantiene plana en 1.9–2.0 a lo largo de cinco órdenes de magnitud del prior** —*"contrary to the usual result where classifier performance improves with increased training data"*—, lo que sugiere que la capacidad discriminativa está casi enteramente en la **representación compartida** y no en la cabeza de clasificación. Es un argumento a favor de los embeddings antes de que el paper llegue a proponerlos.
{{< /concept-alert >}}

## Los experimentos de escala

Tres ejes, con calidad de evidencia muy desigual.

### Eje 1 — cantidad de datos (Tabla 4)

Setup: ResNet-50, 3 K etiquetas, **16 M mini-batches** (unas 380 h) sobre subconjuntos de 70 M, 7 M, 700 K, 70 K y 23 K videos. Nota que **el número de pasos es constante**; lo que varía es de cuántos videos distintos se muestrean esos pasos. No es un experimento de "más datos = más pasos": es un experimento de **diversidad a cómputo constante**.

| Videos de entrenamiento | AUC | d′ | mAP | mAP relativo |
|---|---|---|---|---|
| 70 M | **0.923** | **2.019** | **0.206** | 100 % |
| 7 M | 0.922 | 2.006 | 0.202 | 98.1 % |
| 700 K | 0.921 | 1.997 | **0.203** | **98.5 %** |
| 70 K | 0.909 | 1.883 | 0.162 | 78.6 % |
| 23 K | 0.868 | 1.581 | 0.118 | 57.3 % |

{{< concept-alert type="clave" >}}
**Este es el resultado más subestimado del paper.** De 700 K a 70 M videos —un factor **100** de datos— el mAP se mueve de 0.203 a 0.206: **1.5 %**. El AUC se mueve 0.002. Lo que sí importa es el escalón entre 70 K y 700 K, donde un factor 10 de datos compra 25 % relativo de mAP. Es decir: para esta arquitectura (30 M parámetros) y esta tarea, **el punto de rendimientos decrecientes está entre 70 K y 700 K videos** — entre ~5400 y ~54 000 horas de audio. El titular de "5.24 millones de horas" es real, pero la evidencia interna del propio paper dice que el corpus está enormemente sobredimensionado. **Lo que hizo funcionar al modelo fue cruzar el umbral mínimo y usar una arquitectura convolucional decente**; el resto del corpus compró margen, robustez a clases raras y un titular.
{{< /concept-alert >}}

Dos matices que el paper agrega y que evitan sobreinterpretar la tabla:

- **Los modelos chicos sobreajustaron y eso no se corrigió.** *"The 70K and 23K models […] likely suffered from overfitting. Regularization techniques (or data augmentation) might have boosted the numbers on these smaller training sets."* Recuerda que **no usaron regularización en ningún modelo**. Las dos filas inferiores no miden "qué pasa con menos datos": miden "qué pasa con menos datos **y sin regularización**". Con un régimen bien regularizado la caída sería menor, lo que **refuerza** la conclusión de saturación.
- **Ninguno completó una época.** Con 20 mil millones de ejemplos y ResNet-50 a 11 mini-batches por segundo sobre 20 GPUs, una época tomaría $\tfrac{2\times10^{10}}{128\times11} \approx 1.42\times10^7$ s ≈ **23 semanas** — exactamente lo que dice el paper. Pero *"we expect to see at least one frame from each video in only 14 hours"*. La hipótesis explícita era que *"70M videos will provide an advantage over 7M by virtue of the greater diversity"*. **La tabla la refuta**, y los autores lo aceptan moderando el enunciado en las conclusiones a *"increasing the number of videos up to 7M improves performance"*.

### Eje 2 — tamaño del vocabulario de etiquetas (Tabla 3)

Setup: ResNet-50, 70 M videos, 5 M mini-batches, entrenando con 30 K, 3 K o 400 etiquetas, **siempre evaluando sobre las mismas 400**. La hipótesis: entrenar con más categorías podría actuar como regularizador.

| Bottleneck (128-d) | Etiquetas de entrenamiento | AUC | d′ | mAP |
|---|---|---|---|---|
| no | 3 K | **0.930** | **2.087** | **0.381** |
| no | 400 | 0.928 | 2.067 | 0.376 |
| sí | 30 K | 0.925 | 2.035 | 0.369 |
| sí | 3 K | 0.919 | 1.982 | 0.347 |
| sí | 400 | 0.924 | 2.026 | 0.365 |

**La evidencia es débil y no monótona.** Con cuello de botella: 400 → 0.365, 3 K → **0.347**, 30 K → 0.369; el punto intermedio es el peor, y si la hipótesis fuera correcta esperaríamos monotonía —que no aparece ni en mAP, ni en AUC, ni en d′. Además, la celda que decidiría el experimento —30 K sin bottleneck— **está vacía**: *"it would have taken several months to train."* El paper califica su propio resultado con la palabra justa, *"weak support"*, y las conclusiones repiten *"albeit modestly"*.

Lo que la tabla **sí** soporta con firmeza es el otro resultado, el que no era la pregunta: **los modelos sin cuello de botella son consistentemente mejores** (0.381 vs 0.347 a 3 K; 0.376 vs 0.365 a 400). Esa es una medición limpia del costo de la compresión, y la retomamos abajo.

### Eje 3 — tamaño del modelo

Este eje **no se barre de forma controlada**, y hay que decirlo. Lo único que existe es la Tabla 2, que varía **arquitectura**, no tamaño, con ambas confundidas — y en la que, en el tramo superior, el orden por parámetros es casi el **inverso** del orden por desempeño (VGG 62 M → 0.161; AlexNet 37.3 M → 0.115; ResNet-50 30 M → 0.182; Inception-V3 28 M → 0.181). La única lectura defendible es que **a esta escala de datos la capacidad bruta no es el cuello de botella; la topología sí**. Un barrido de verdad habría requerido ResNet-18 / 50 / 101 / 152 con todo lo demás fijo, y **no está** — es la ausencia más notable del paper, porque es la variable que la literatura posterior identificó como decisiva.

## Transferencia a AudioSet

### Qué se hizo exactamente

> *"We train two fully-connected models to predict labels for Audio Set. The first model uses $64 \times 20$ log-mel patches and the second uses the output of the penultimate 'embedding' layer of our **best ResNet model** as inputs."*

{{< concept-alert type="advertencia" >}}
**Este experimento no es fine-tuning, ni warm-starting.** El paper no reinicializa ni ajusta la ResNet sobre AudioSet. Lo que hace es: (1) congelar la ResNet-50 entrenada en YouTube-100M; (2) extraer la activación de la penúltima capa; (3) entrenar **un MLP desde cero** sobre esos vectores congelados con las etiquetas de AudioSet; (4) compararlo contra **otro MLP desde cero** que recibe log-mel crudo. Es **transferencia por features congeladas** (un *shallow probe*), y eso hace el resultado *más* fuerte, no menos: si un MLP sobre features congeladas duplica el mAP de un MLP sobre features crudas, la conclusión sobre la calidad de la representación es directa y no está contaminada por la capacidad de la cabeza.
{{< /concept-alert >}}

| Entrada del MLP | mAP | AUC | d′ |
|---|---|---|---|
| Log-mel $64\times20$ (200 ms de contexto) | 0.137 | 0.904 | 1.846 |
| Embedding de la penúltima capa de ResNet-50 | **0.314** | **0.959** | **2.452** |
| **Ganancia** | **×2.29** | +0.055 | **+0.606** |

La "tasa de error" complementaria $1-\text{AUC}$ cae de 0.096 a 0.041, una **reducción del 57 %**. Y un salto de 0.606 en d′ es, en unidades de teoría de detección, la diferencia entre un detector mediocre y uno usable.

**El confounder que hay que anotar.** El baseline no es una comparación limpia: el modelo de log-mel recibe **20 tramas = 200 ms**, y el de embeddings resume **960 ms**. Son casi **5× de diferencia en contexto temporal**, y el paper no separa las dos contribuciones ni ofrece un baseline de log-mel a $64\times96$ (que habría sido exactamente la entrada de la ResNet). Es una omisión real. Dicho eso, es implausible que 5× de contexto explique por sí solo un factor 2.29 en mAP, así que la conclusión cualitativa sobrevive con la magnitud exacta sobreestimada.

### Por qué este resultado legitimó todo el paradigma

**La aritmética del desbalance es brutal, y a favor de la transferencia.** AudioSet son ~3000 horas etiquetadas por humanos; YouTube-100M son 5.24 M horas etiquetadas por máquina — una razón de **0.05 %**. El resultado dice: *5.24 M horas de supervisión sucia, comprimidas en 128 dimensiones por segundo, valen más que 3000 horas de supervisión limpia usadas directamente*. Ese es el argumento fundacional del preentrenamiento a gran escala, aplicado a audio y con números (ver [transfer learning](/fundamentos/transfer-learning)). Y **es transferencia entre tareas distintas, no solo entre datasets**: YouTube-100M tiene etiquetas de **contexto de video** ("Game", "Web Page", "Boeing 757") y AudioSet de **evento acústico** — ni el mismo espacio de etiquetas ni la misma semántica. Que la representación transfiera igual es la evidencia de que la red aprendió algo sobre **el sonido**, no sobre el vocabulario de YouTube.

Tercero, y más terrenal: **convirtió el modelo en un producto**. Después de este resultado la pregunta operativa dejó de ser *"¿qué arquitectura entreno?"* y pasó a ser *"¿por qué no estoy usando los embeddings de Google?"*. Un dato que cierra el arco: la línea *"Baseline — CNN+MLP — mAP 0.314"* de la tabla de estado del arte del paper de [AST](/papers/ast-gong-2021) (2021) **es este número**. Cuatro años después, el punto de referencia de AudioSet seguía siendo este modelo.

## VGGish como extractor de embeddings

### Qué es el cuello de botella de 128 dimensiones y de dónde salió

El detalle histórico es delicioso: **el paper no introdujo el cuello de botella para producir embeddings, sino por velocidad**. *"We introduced the bottleneck layer to speed up the training of the model trained with 30K labels. Without a bottleneck, the larger output layer increased the number of weights from 30M to 80M and significantly reduced training speed."* La aritmética: sin cuello de botella, la capa de salida conecta las **2048** activaciones del AvgPool de ResNet-50 con 30 871 salidas → $2048 \times 30871 \approx 63$ M parámetros; con él, $2048\times128 + 128\times30871 \approx 4.2$ M. **Un factor 15 de reducción.** El embedding de 128 dimensiones nació como optimización de entrenamiento y resultó ser el producto más duradero del paper.

**Cuánto cuesta comprimir.** Comparando filas de la Tabla 3 con las mismas etiquetas: a 400, $0.376 \to 0.365$ (−2.9 % relativo); a 3 K, $0.381 \to 0.347$ (−8.9 %). El paper reconoce que el cuello de botella *"is effecting a substantial reduction in information"*. Es decir: **comprimir 2048 → 128 cuesta entre 3 % y 9 % de mAP**, un precio bajísimo por un factor 16 de compresión — y esa relación es toda la justificación económica del pipeline de embeddings.

### PCA y cuantización a 8 bits

Este componente **no está en el paper**: es parte del pipeline publicado junto con AudioSet. El post-procesamiento antes de distribuir los embeddings es (1) **PCA con blanqueo** sobre las 128 dimensiones —no reduce la dimensionalidad, **decorrelaciona y normaliza la varianza por eje**— y (2) **cuantización uniforme a 8 bits** por dimensión, con recorte previo a un rango fijo.

**El orden importa.** El embedding sale de una ReLU, así que es no negativo y con varianzas muy dispares entre ejes; cuantizarlo directo desperdiciaría casi todos los niveles, saturando los ejes de varianza alta y dejando dos o tres niveles a los de varianza baja. El blanqueo **iguala las varianzas**, y recién entonces la cuantización uniforme es aproximadamente óptima para todos. Es el mismo razonamiento de la cuantización por canal en inferencia moderna.

| Formato | Bytes por segundo | AudioSet completo (~2.1 M clips × 10 s) |
|---|---|---|
| Audio PCM 16 kHz / 16 bits | 32 000 | ~670 GB |
| Embedding float32 (128-d, 1 Hz) | 512 | ~10.7 GB |
| Embedding uint8 (128-d, 1 Hz) | **128** | **~2.7 GB** |

**El formato del embedding es, literalmente, la razón por la que AudioSet existe como dataset descargable.** Google no puede redistribuir el audio de YouTube, pero sí un descriptor de 128 bytes por segundo — una compresión de **250×** frente al audio crudo. La contrapartida es la limitación permanente de AudioSet: durante años la comunidad trabajó sobre features precomputadas a 1 Hz, sin acceso al audio, lo que impedía experimentar con el front-end. PANNs y AST solo son posibles cuando la gente empezó a re-descargar el audio por su cuenta.

Una consecuencia práctica si usas estos embeddings en un índice vectorial: **son no negativos (post-ReLU) y están cuantizados**, así que todos los vectores viven en el ortante positivo —lo que comprime el rango de ángulos posibles— y la distancia coseno sobre los uint8 crudos no es la de los float originales. Des-cuantiza primero y, si tienes la matriz PCA, considera revertir el blanqueo antes de medir similitudes.

### Cómo se agregan los embeddings en el tiempo

Sección 3.2: *"We passed each 960 ms frame […] through the classifier. We then averaged the classifier output scores across all segments in a video."* Es decir: **promedio de scores, no de embeddings, y sin ningún modelo temporal**, imitando a Ng et al. en video visual. Es la decisión más floja del paper y los autores lo saben. Promediar **scores post-sigmoide** no es lo mismo que promediar logits ni que hacer max-pooling: **el promedio penaliza los eventos raros y breves**, porque un disparo de 1 s en un video de 4.6 minutos aporta un score alto en 1 de 287 segmentos y el promedio lo diluye hasta hacerlo indistinguible del ruido. Para eventos esporádicos, max o attention pooling son estrictamente mejores — que es exactamente lo que hicieron PANNs y PSLA años después.

## Las diferencias entre audio e imagen que el paper deja ver

El paper toca el tema en **una sola frase** —*"the distinct meanings of time and frequency axes might argue for audio-specific architectures"*— y no lo desarrolla. Es la misma omisión del slide "Audio vs Image Data" de la [Clase 39](/clases/clase-39), que dice *"there are relevant differences between audio and visual data that is important to consider"* y pasa a la lámina siguiente. Lo que sigue es el desarrollo faltante, con el veredicto explícito de en cuáles es sorprendente que las [redes convolucionales](/fundamentos/redes-convolucionales) de visión funcionen igual.

### (a) Los ejes no son intercambiables

En una imagen natural, $x$ e $y$ son **el mismo tipo de cosa**: coordenadas espaciales sobre una superficie. Esa equivarianza es la premisa de la convolución 2D con pesos compartidos — se justifica usar el mismo kernel en todas las posiciones porque un borde en la esquina es el mismo evento visual que un borde en el centro. En un espectrograma, **trasladar en tiempo preserva el objeto** (un ladrido a los 0.3 s y el mismo ladrido a los 0.7 s son el mismo evento: la equivarianza temporal es correcta), pero **trasladar en frecuencia lo cambia**: un desplazamiento de $\Delta$ bandas **transpone** el sonido — una voz masculina desplazada hacia arriba es una voz femenina o un chirrido, un Do se convierte en un Mi. La equivarianza en frecuencia es **incorrecta**.

Y hay un agravante que casi nunca se menciona: **sobre el eje mel, un desplazamiento constante ni siquiera es una transposición limpia.** Como $m = 1127\ln(1+f/700)$ es aproximadamente **lineal** en Hz por debajo de ~700 Hz y **logarítmica** por encima, un mismo desplazamiento implementa una multiplicación por factor constante (transposición musical) arriba y una **suma en Hz** —que no es transposición de nada— abajo. Para que fuera transposición en todo el rango habría que usar un eje log puro o una CQT, que es exactamente lo que hace la literatura de música.

**Por qué funciona igual, entonces.** Dos mecanismos, y las redes del paper no los usan igual:

1. **La invarianza local es útil aunque la global sea falsa.** El kernel $3\times3$ de la primera capa no detecta "un Do": detecta *"hay energía que sube en frecuencia con el tiempo"* o *"hay un borde vertical de banda ancha"* (un transitorio). Esos patrones **sí** son aproximadamente equivariantes en frecuencia, y compartir pesos ahí es lo que permite reconocer una misma clase de sonido en hablantes de distinto registro.
2. **La invarianza global se rompe (o no) en la cabeza.** **VGG y VGGish aplanan** el mapa final preservando la posición, así que la primera capa densa **puede leer la frecuencia absoluta**. **ResNet-50 e Inception-V3 hacen average pooling global** ($6\times4$ y $10\times6$, sobre **todo** el mapa restante), destruyendo la posición absoluta en ambos ejes: arriba son **totalmente invariantes a transposición**. Y sin embargo la Tabla 2 dice que ResNet e Inception **ganan**. **Empíricamente, para etiquetas de contexto de video, la frecuencia absoluta es prescindible** — "Car", "Speech" o "Guitar" se identifican por textura y envolvente espectral, no por altura absoluta. Sería un desastre para transcripción musical, estimación de $f_0$ o identificación de hablante.

**Veredicto: sorprende bastante.** Compartir pesos en frecuencia es formalmente injustificado, y que además *descartar* la posición absoluta con average pooling global sea la mejor opción es el resultado más contraintuitivo del paper. Ambas cosas dependen críticamente de que la tarea sea etiquetado grueso.

### (b) La localidad no es simétrica

Un sonido armónico con fundamental $f_0$ tiene energía en $f_0, 2f_0, 3f_0, \dots$: un patrón **periódico y no local en frecuencia**, y la firma más informativa que existe para distinguir un tono de un ruido, separar fuentes e identificar timbre.

**Un kernel $3\times3$ no lo captura.** Tres bandas mel adyacentes en la región de 1 kHz cubren del orden de 100–150 Hz; si $f_0 = 200$ Hz, los armónicos están separados por 200 Hz y **el kernel de la primera capa nunca ve dos armónicos simultáneamente**. En imágenes esto no tiene análogo: cuando hay periodicidad visual (una reja, un tejido) el patrón está a escala de pocos píxeles.

**La red lo resuelve por profundidad.** Calculando el campo receptivo de la pila convolucional de VGGish (6 conv $3\times3$ con 4 max-pools de stride 2) con $\text{RF}_{out} = \text{RF}_{in} + (k-1)\,j_{in}$, se llega a **RF = 70 unidades por eje** tras `pool4`. Como el eje de frecuencia tiene solo **64 bandas**, la última capa convolucional **ve el espectro completo**: la estructura armónica global es accesible, pero solo tras seis capas de composición, y la red tiene que **aprenderla** como conjunción de patrones locales en lugar de recibirla como primitiva.

Ese es el costo real, y es de **eficiencia estadística**: la red gasta capacidad y datos aprendiendo algo que la física del sonido regala. La literatura posterior lo atacó con **harmonic stacking** (apilar copias del espectrograma desplazadas a $2f, 3f, \dots$ como canales, para que un kernel local sí vea armónicos alineados) y con la CQT. **Veredicto: sorprende moderadamente** — es la diferencia estructural donde una arquitectura específica de audio tiene el argumento más sólido.

### (c) Las fuentes se suman; los objetos visuales se ocluyen

Esta es la diferencia más profunda de las cuatro y la de consecuencias prácticas más amplias.

**En imágenes, la composición es por oclusión.** Si A está delante de B, los píxeles compartidos son los de A: la información de B **se pierde por completo** y la de A **se conserva intacta**. Un detector puede confiar en que la evidencia local que ve pertenece a un solo objeto, y la segmentación semántica —un píxel, una clase— es un problema bien planteado.

**En audio, la composición es por suma.** $x(t) = x_1(t) + x_2(t)$, y la linealidad se propaga a la STFT. Se complica al pasar a magnitud, porque la suma es compleja y depende de la fase relativa:

$$|X| = \big|\,|X_1| e^{j\phi_1} + |X_2| e^{j\phi_2}\,\big| \le |X_1| + |X_2|$$

y el logaritmo rompe cualquier aditividad residual:

$$\log(|X_1| + |X_2| + \epsilon) \neq \log(|X_1|+\epsilon) + \log(|X_2|+\epsilon)$$

En el régimen habitual, donde una fuente domina ($|X_1| \gg |X_2|$), $\log(|X_1|+|X_2|) \approx \log|X_1|$: **la fuente débil desaparece.** Eso es enmascaramiento, y es la razón física por la que el log-mel es una representación *dispersa* en la que, en cada celda tiempo-frecuencia, típicamente una fuente domina.

Cuatro consecuencias concretas y verificables en este paper:

1. **La tarea es intrínsecamente multi-etiqueta, y por eso se usa sigmoide y no softmax.** *"All models used a final sigmoid layer rather than a softmax layer since each example can have multiple labels."* No es un detalle de implementación: es la superposición aditiva manifestándose en la capa de salida. Un segmento de 960 ms **casi siempre** contiene varias fuentes simultáneas y ninguna oculta a las otras.
2. **La red tiene que hacer separación de fuentes implícita.** Para emitir "Trumpet" en un segmento donde también hay batería, público y ruido de sala, tiene que aislar la evidencia de un patrón que contiene todo sumado. En visión la trompeta ocupa píxeles que son solo de la trompeta.
3. **Mixup es física, no regularización.** En imágenes, promediar dos imágenes produce un artefacto que no existe en el mundo, y su valor es puramente regularizador. En audio, **sumar dos formas de onda produce un sonido real que un micrófono podría haber capturado, cuyo conjunto correcto de etiquetas es la unión de los dos**: una operación **semánticamente exacta**. Por eso mixup dio ganancias tan grandes en AudioSet (PANNs, PSLA y AST lo usan todos) mientras que en visión es más marginal, y por eso existe Scaper —sintetizar paisajes sonoros mezclando eventos sobre fondos, con anotación exacta y gratuita— mientras que **en visión no existe un Scaper**: no se pueden componer objetos sin resolver oclusión, sombras e iluminación.
4. **La aditividad afecta la agregación temporal.** Promediar scores sobre un video diluye eventos breves; en imágenes, el análogo (promediar sobre crops) es benigno porque los objetos ocupan regiones extensas.

**Veredicto: aquí es donde *menos* sorprende que funcione.** Una CNN es un detector de patrones locales y no le importa cómo se compuso el patrón. Lo que sí cambia es **todo lo que rodea al modelo**: la pérdida, el aumento de datos, la métrica y la agregación.

### (d) El eje de frecuencia ya viene deformado por diseño perceptual

Los píxeles de una imagen viven en un espacio de coordenadas **lineal y físicamente neutro**: nadie aplica una transformación no lineal a las coordenadas espaciales antes de alimentar la CNN. El espectrograma log-mel aplica **dos** antes de que la red vea nada: el **warp mel** en el eje de frecuencia —la coordenada del "píxel" no es la frecuencia física, sino una función de ella elegida por experimentos psicoacústicos de 1937— y el **logaritmo** en el eje de valores.

{{< concept-alert type="clave" >}}
La "imagen" que la CNN procesa **no es el dato físico**: es el dato ya filtrado por un modelo del sistema auditivo humano. Hay una cantidad enorme de conocimiento de dominio —resolución crítica del oído, ley de Weber-Fechner, banda útil del habla— incrustada en el preprocesamiento y no en la arquitectura. **La afirmación "usamos redes de visión sin modificar" es cierta a nivel de arquitectura y falsa a nivel de sistema: la adaptación al audio existe y está toda en el front-end.**
{{< /concept-alert >}}

Esto explica por qué las redes sobre [forma de onda cruda](/papers/raw-waveforms-dai-2017) tardaron tanto en competir: no compiten contra "una CNN", compiten contra "una CNN más ochenta años de psicoacústica". Y explica por qué el trabajo posterior que sí mejoró el front-end (más bandas, 32 kHz, PCEN, LEAF) obtuvo ganancias reales.

Segunda implicación, sobre aumento de datos: **el repertorio estándar de visión no se traslada.**

| Aumento en visión | Traslado a espectrograma |
|---|---|
| Flip horizontal | **Inválido.** Invierte el tiempo. Un sonido al revés es otro sonido (piensa en el ataque de un piano). |
| Flip vertical | **Inválido.** Invierte el espectro; absurdo físicamente. |
| Rotación | **Inválida.** Mezcla tiempo con frecuencia: unidades incompatibles. |
| Random resized crop | **Parcialmente inválido.** Estirar el tiempo es *time stretch* (legítimo); estirar la frecuencia es transposición. **Estirar ambos con el mismo factor —lo que hace el crop de visión— no corresponde a ninguna operación acústica.** |
| Traslación | Válida **solo** en tiempo. |
| Jitter de brillo | Análogo válido: ganancia global (offset aditivo en log). |
| Cutout / random erasing | **Sí traslada, y muy bien:** es esencialmente **SpecAugment**, pero restringido a enmascarar **bandas completas** de tiempo o de frecuencia, porque el enmascaramiento acústico real ocurre así. |

Que el aumento correcto para espectrogramas (SpecAugment, 2019) haya llegado dos años después de este paper y sea una **restricción** de una técnica de visión, y no una importación directa, es la mejor evidencia de que este eje sí es distinto.

**Veredicto: no sorprende que la arquitectura funcione, porque la adaptación al audio ya ocurrió antes de la primera capa.**

### Síntesis

Sorprende **bastante** en (a) —compartir pesos en frecuencia es formalmente injustificado, y encima el average pooling global, que descarta la frecuencia absoluta, gana— y **moderadamente** en (b), donde la profundidad compra alcance de forma estadísticamente ineficiente. **No sorprende** en (c), porque a una CNN le da igual el proceso generativo, ni en (d), porque la adaptación al audio ya está hecha en el front-end.

**El punto que unifica todo:** la advertencia de la Sección 1 es correcta como preocupación teórica; la contribución del paper es mostrar que **para esta tarea** el efecto neto es pequeño. El "para esta tarea" es la letra chica: etiquetado grueso de eventos y contexto, con ~1 s de contexto, donde la envolvente espectral basta. Cambia la tarea a transcripción musical, estimación de $f_0$, verificación de hablante o separación de fuentes, y las cuatro diferencias vuelven a morder.

## Limitaciones

1. **Ruido de etiqueta no cuantificado ni cuantificable.** Todas las cifras absolutas tienen un techo impuesto por el ruido y no sabemos dónde está. El sesgo del canal visual introduce una correlación sistemática que la escala **no** promedia, y los autores reconocen que ni siquiera pueden definir "relevancia" para la mayoría de las clases.
2. **El dataset no es reproducible.** YouTube-100M es interno de Google y las etiquetas son identificadores del Knowledge Graph, propietario y cambiante. **Nadie fuera de Google puede reproducir las Tablas 2, 3 ni 4:** la comunidad recibió el checkpoint, no el experimento. El resultado central —"la escala funciona"— es, en sentido estricto, no verificable de forma independiente, y terminó verificándose al revés cuando PANNs, entrenando solo sobre AudioSet, superó ampliamente estos números.
3. **Sin modelado temporal explícito.** La única operación por encima de 960 ms es un promedio de scores, justificado por analogía con video visual — una analogía que no es obvia en audio. El costo concreto: **el promedio diluye eventos breves y raros**, exactamente los que interesan en detección de eventos acústicos.
4. **Resolución mínima de 960 ms.** El sistema no puede localizar nada más fino que ~1 s: un disparo (~50 ms de transitorio), un chasquido, un fonema (~80 ms), el ataque de una nota o un clic valvular cardíaco quedan promediados dentro de su parche, y como los parches no se solapan, un evento a caballo entre dos se reparte. **Este es el límite duro del formato**, y la razón por la que el mundo del *sound event detection* con marcas de tiempo nunca adoptó VGGish como front-end.
5. **Ninguna regularización, en ningún modelo.** Justificado para 70 M videos; **no justificado** para las filas de 70 K y 23 K, que los autores admiten que sobreajustaron.
6. **Evidencia floja en el eje de vocabulario**, **comparación a pasos fijos y no a cómputo fijo**, y **todas las arquitecturas sin converger** (+16 % de mAP relativo entre 5 M y 17 M pasos).
7. **Cero baselines específicos de audio** —ni CRNN, ni forma de onda, ni MFCC+GMM sobre el mismo dataset— y **ninguna ablación de qué capa produce el mejor embedding**: se usa "la penúltima", sin probar la anterior ni concatenaciones.

## Por qué importa hoy

### VGGish y YAMNet como extractores estándar

Google liberó el modelo en el repositorio de AudioSet y luego en TensorFlow Hub, y durante aproximadamente **2017–2021 VGGish fue el default de facto para features de audio** — por razones que tienen poco que ver con su calidad y mucho con la fricción: un tensor de $96\times64$ con una API de una línea, 128 flotantes por segundo baratos de calcular, guardar e indexar, ports a PyTorch de terceros, y sobre todo, **AudioSet se distribuye como embeddings de VGGish**. Quien quisiera usar AudioSet sin re-descargar dos millones de clips de YouTube usaba VGGish: no era una elección.

**YAMNet** llegó después como el hermano eficiente: misma entrada $96\times64$ log-mel, mismo formato de salida, pero backbone **MobileNet-v1** y cabeza de **521 clases** de la ontología de AudioSet. La ironía histórica: **YAMNet es, otra vez, una arquitectura de visión importada sin cambios.** La tesis de Hershey et al. no solo sobrevivió; se volvió el procedimiento operativo estándar de Google para audio.

### La sucesión

| Año | Sistema | Arquitectura | mAP (AudioSet completo) |
|---|---|---|---|
| 2017 | **Este paper / baseline de AudioSet** | CNN embeddings + MLP | **0.314** |
| 2020 | PANNs (Kong et al.) | CNN + attention pooling | 0.439 |
| 2021 | PSLA (single / Ensemble-M) | CNN + attention | 0.444 / 0.474 |
| 2021 | **[AST](/papers/ast-gong-2021)** (single / Ensemble-M) | Pure attention (ViT/DeiT) | **0.459 / 0.485** |

(Sistemas posteriores —BEATs, Audio-MAE, CAV-MAE— reportan 0.48–0.51.) El detalle que más llama la atención: **la fila 0.314 de este paper seguía en la tabla de estado del arte cuatro años después, como el punto de referencia**. Es la definición de un baseline canónico.

### Qué desplazó a VGGish, y por qué

1. **Entrenar directamente sobre AudioSet con las recetas correctas (PANNs, 2020).** La observación que rompió el paradigma: si la Tabla 4 de este mismo paper dice que 700 K videos rinden casi lo que 70 M, entonces los ~2 M de clips de AudioSet **ya son suficientes** para entrenar una CNN grande desde cero. PANNs lo hizo y llegó a 0.439 contra 0.314 —un salto de 40 % relativo, **sin YouTube-100M**— aportando no escala sino audio a **32 kHz**, **mixup**, **balanced sampling** y **attention pooling** (un evento de 200 ms en un clip de 10 s se recupera si el modelo aprende a ponderar segmentos; se pierde si se promedian scores). En retrospectiva, **el modelo de Hershey et al. estaba limitado por el front-end, el aumento y la agregación, no por los datos.** Su propia Tabla 4 lo insinuaba.
2. **Transformers con pesos de ImageNet ([AST](/papers/ast-gong-2021), 2021).** El capítulo más irónico del legado: AST descarta la convolución, corta el espectrograma en parches de $16\times16$ y aplica un ViT **inicializado con pesos de DeiT entrenados en ImageNet**. Su ablación cuantifica la contribución: **sin preentrenamiento en ImageNet, 0.366 de mAP; con él, 0.459** — **+0.093 de mAP puramente por importar pesos de un modelo de imágenes**. Hershey et al. mostraron en 2017 que las *arquitecturas* de visión transfieren a audio; AST mostró en 2021 que los *pesos* también. **La tesis de este paper no fue refutada: fue radicalizada.**
3. **Auto-supervisión (SSAST, Audio-MAE, BEATs, wav2vec 2.0 / HuBERT).** El golpe final: si se puede aprender una representación de audio **sin etiquetas**, todo el aparato de 5.24 M horas de etiquetas ruidosas del Knowledge Graph deja de ser necesario. Las horas siguen siendo útiles, pero como audio, no como pares audio-etiqueta.

**Qué se le sigue criticando y qué sigue sirviendo.** Envejeció por: granularidad de 960 ms, embedding post-ReLU (no negativo, mal condicionado para métricas de similitud), cuantización a 8 bits en el formato distribuido, banda limitada a 7500 Hz, front-end congelado, entrenamiento sobre etiquetas de contexto de video en vez de eventos acústicos, y cero recetas modernas de aumento. Lo que sigue justificándolo: es **rápido, determinista y corre en CPU sin GPU ni stack de transformers**, y para un baseline o una recuperación por similitud sigue siendo razonable. Es el equivalente en audio a usar features de ResNet-50 en 2024: nadie escribe un paper con eso, pero mucha gente lo despliega.

Más allá del modelo quedaron tres ideas: que **"el espectrograma es una imagen" es una aproximación de trabajo, no una metáfora**, suficientemente buena para justificar diez años de importación de arquitecturas; que **el embedding congelado es un producto** —la salida útil de un modelo grande no es su predicción sino su representación intermedia, distribuible como artefacto compacto—; y que **las curvas de escala hay que medirlas, no asumirlas**. La Tabla 4 es un ejemplo temprano y honesto de un equipo con datos ilimitados publicando que sus datos ilimitados no eran necesarios.

## Erratas y cosas que se citan mal

### "VGGish" no aparece en el paper

**Cero ocurrencias en el texto completo.** El nombre lo puso Google después, al liberar el modelo. Estrictamente: **este paper no presenta VGGish; presenta el trabajo del que VGGish salió.**

### El "VGG" del paper no es VGGish

Son **arquitecturas distintas**, y confundirlas es el error más extendido después del anterior:

| | "VGG" de la Tabla 2 | "VGGish" publicado |
|---|---|---|
| Configuración | **E** (VGG-19): 16 conv + 3 densas, 5 bloques | Variante de la config. **A** (VGG-11), **truncada tras el 4.º bloque**: 6 conv, 4 bloques |
| Capas de pooling | 5 | 4 |
| Mapa antes de aplanar | $3\times2\times512 = 3072$ | $6\times4\times512 = 12\,288$ |
| Cabeza densa | 4096 → 4096 → **3087** (sigmoide) | 4096 → 4096 → **128** (ReLU) |
| Parámetros | **62 M** | **72.1 M** |
| Salida | 3087 logits multi-etiqueta | embedding de 128 dimensiones |

Citar el AUC 0.911 / mAP 0.161 de la fila "VGG" como "el desempeño de VGGish" es incorrecto. La elección de VGG en vez de ResNet para publicar es razonable desde ingeniería —VGG es trivial de portar entre frameworks, sin bloques residuales ni ramas— pero significa que **el checkpoint público no es el mejor modelo del paper**.

### Los embeddings que dan mAP 0.314 son de ResNet-50, no de VGG

{{< concept-alert type="advertencia" >}}
**Esta es la mala cita más difundida sobre el paper.** El texto es inequívoco: *"the second uses the output of the penultimate 'embedding' layer of our **best ResNet model** as inputs"*. Y la mejor red del paper es **ResNet-50** (mAP 0.212), no VGG (0.161). Por lo tanto, la frase "los embeddings de VGGish alcanzan mAP 0.314 en AudioSet" atribuye a VGGish un resultado obtenido con ResNet-50. **El checkpoint VGGish que Google liberó nunca fue evaluado en el paper.**
{{< /concept-alert >}}

### Inconsistencias internas del propio paper

- **5.24 M vs 5.4 M horas.** Abstract y Sección 1: *"70M training videos (5.24 million hours)"*. Sección 2: *"Videos average 4.6 minutes each for a total of 5.4M training hours"*. Como $70\times10^6 \times 4.6/60 = 5.37\times10^6$, la cifra consistente con los 4.6 minutos es 5.4 M; los 5.24 M implicarían 4.49 min por video. **Las dos no pueden ser ambas exactas.** El valor más citado en la literatura es 5.24 M.
- **405 h vs 356 h.** Sección 4.1: *"after training for 17 million mini-batches (405 hours)"*. Tabla 2, última fila, misma corrida: **356 h**. Discrepan.
- **No confundir la última fila de la Tabla 2 con la primera de la Tabla 4.** Son corridas distintas: ResNet-50 a 17 M pasos **con reducción de learning rate** da 0.926 / 2.041 / 0.212; ResNet-50 a 16 M pasos sobre 70 M videos, sin mención de reducción de lr, da 0.923 / 2.019 / 0.206.

### Dos lecturas invertidas que circulan

- **"El paper demuestra que más datos siempre ayudan."** Demuestra **lo contrario**: la Tabla 4 muestra saturación, con un factor 100 de datos moviendo el mAP de 0.203 a 0.206.
- **"El paper demuestra que más etiquetas ayudan."** Los autores dicen *"weak support"* y *"albeit modestly"*, y la Tabla 3 no es monótona.

Y un matiz sobre la Figura 1: la lectura correcta es "**d′** es aproximadamente constante en el prior", no "el desempeño es constante". El **mAP** de las clases raras es bajísimo, porque el AP está correlacionado con el prior. Las dos métricas cuentan historias opuestas sobre las mismas predicciones, y ambas son correctas.

## En la clase 39 y su laboratorio

El slide "Audio vs Image Data" de la [Clase 39](/clases/clase-39) afirma que *"the 2D time-freq representation (spectrogram) of an audio signal can be interpreted as an image […] While this is possible, there are relevant differences between audio and visual data that is important to consider."* Este paper es exactamente las dos mitades de esa frase: la **validación empírica a gran escala** de la primera parte, y —en su Sección 1— la **fuente de la advertencia** que el slide no desarrolla.

El laboratorio hace **fine-tuning de VGGish sobre [UrbanSound8K](/papers/urbansound8k-salamon-2014)**, en la Parte 2 del mismo notebook cuya Parte 1 entrena la familia M de [Dai et al. sobre forma de onda cruda](/papers/raw-waveforms-dai-2017). Tres cosas conviene tener claras antes de ejecutarlo.

### La auditoría del "60 millones de parámetros"

El notebook afirma que la sección fully-connected *"acumula por sí sola más de 60 millones de parámetros"*. **Verificado, y es correcto.** Desglosando la arquitectura exacta que define el notebook, con entrada $1\times96\times64$:

| Bloque | Parámetros | Multiplicaciones (MACs) |
|---|---|---|
| Pila convolucional (6 conv + 4 max-pool), flatten → 12 288 | **4 499 712** | **~796 M** |
| `Linear(12288, 4096)` | 50 335 744 | 50.3 M |
| `Linear(4096, 4096)` | 16 781 312 | 16.8 M |
| `Linear(4096, 128)` | 524 416 | 0.5 M |
| **Subtotal denso** | **67 641 472** | **~67.6 M** |
| **TOTAL** | **72 141 184** | **~864 M** |

La sección densa suma **67 641 472** de **72 141 184** parámetros totales: el **93.8 %**. "Más de 60 millones" es una descripción conservadora.

{{< concept-alert type="clave" >}}
**La inversión notable: la parte que tiene los parámetros no es la que hace el trabajo.**

| | Parámetros | Cómputo |
|---|---|---|
| Convolucional | 4.5 M (**6.2 %**) | ~796 M MACs (**92 %**) |
| Denso | 67.6 M (**93.8 %**) | ~67.6 M MACs (**8 %**) |

Los 67.6 M de las densas son casi todos memoria y riesgo de sobreajuste; las 796 M de multiplicaciones viven en 4.5 M de pesos convolucionales que se reutilizan en cada posición del mapa. El `Linear(12288, 4096)` es por sí solo el **70 % del modelo**. Esta es exactamente la patología que Inception y ResNet resolvieron con global average pooling — y la razón por la que, en la Tabla 2 del paper, ResNet-50 (30 M pesos) le gana a VGG (62 M).
{{< /concept-alert >}}

Para dimensionarlo contra la Parte 1: el mayor de los modelos M de Dai et al. (M34-res) tiene 4 M de parámetros. **VGGish tiene ~20 veces más parámetros que el mayor de ellos**, y eso es lo que el notebook quiere transmitir.

### El notebook no hace fine-tuning completo: entrena 40 970 parámetros

Leyendo el código de la celda de preparación, el notebook **congela todos los parámetros** (`requires_grad = False` sobre `net.parameters()`) y luego reemplaza la **penúltima capa densa** —la capa de embedding, `fc[-2]`— por una `nn.Linear(4096, 10)` que nace entrenable.

**Parámetros entrenables reales: $4096 \times 10 + 10 = $ 40 970.** No 72 M, no 67.6 M: **cuarenta mil**. Tres consecuencias:

1. **No hay riesgo de sobreajuste catastrófico.** UrbanSound8K aporta ~7850 clips de entrenamiento que, a 3 parches de 960 ms por clip, dan ~23 500 ejemplos. 40 970 parámetros contra eso es un régimen sano: es esencialmente una regresión logística multinomial sobre features congeladas.
2. **La advertencia de "no más de 2 épocas" no es por sobreajuste; es por tiempo de cómputo.** Con batch 32 y ~735 pasos por época, un probe lineal sobre GPU debería tardar minutos. El costo real está en `__getitem__`: `vggish_input.waveform_to_examples()` corre en **NumPy sobre CPU** —incluye el resampleo de 44.1 kHz a 16 kHz, la STFT y el banco mel— y el `DataLoader` usa **`num_workers = 0`**, así que todo el preprocesamiento se hace en el proceso principal, en serie, **en cada época**. **El cuello de botella del laboratorio es el front-end, no la red.** Si quieres experimentar, precomputa los parches una vez y cachéalos, o sube `num_workers`.
3. **El clasificador no lee el embedding de 128 dimensiones.** Al reemplazar la capa de embedding, la cabeza se conecta a la salida anterior, de **4096 dimensiones**: es un probe sobre una representación 32× más rica que el embedding canónico. Funciona mejor, pero no es "usar los embeddings de VGGish" en el sentido en que lo entiende la literatura. Vale la pena saberlo si comparas contra papers.

### Tres erratas del material del curso

{{< concept-alert type="advertencia" >}}
**1. El enlace a VGGish apunta al paper equivocado.** El notebook enlaza *"el modelo VGGish (https://arxiv.org/pdf/1610.00087.pdf)"*, pero **1610.00087 es *Very Deep Convolutional Neural Networks for Raw Waveforms* de Dai et al.** — el paper de los modelos M3/M5/M11/M18/M34 de la **Parte 1 del mismo notebook**, que la celda 21 enlaza correctamente ahí. Es un copy-paste interno. **El identificador correcto para VGGish es `arXiv:1609.09430`.**

**2. "VGGish trabaja con features MFCC" es falso: usa log-mel.** El notebook lo dice dos veces, y la Sección 3.1 del paper es explícita: **log-mel spectrogram**. La función que el propio notebook invoca, `vggish_input.waveform_to_examples()`, produce log-mel. **Y no es una imprecisión cosmética:** un MFCC se obtiene aplicando una **DCT** al log-mel y quedándose con los primeros ~13 coeficientes. Esa DCT decorrelaciona las bandas —lo que hacía falta para los GMM con covarianza diagonal de la era pre-deep-learning— pero **destruye la localidad en frecuencia**: cada coeficiente cepstral es una combinación lineal de **todas** las bandas mel, así que un kernel $3\times3$ sobre MFCCs opera sobre coeficientes adyacentes que no tienen ninguna relación de vecindad significativa. **La DCT elimina exactamente la estructura que hace que una CNN 2D tenga sentido.** Que VGGish use log-mel no es un detalle: es un requisito de la tesis del paper. Un espectrograma es una imagen; un cepstrograma no lo es.

**3. "Preentrenado en AudioSet" es impreciso.** El modelo se entrenó en **YouTube-100M**; AudioSet fue el **destino de la transferencia** (Sección 4.4), no el origen. La relación es la inversa de la que sugiere el notebook: **AudioSet se distribuye con embeddings calculados por VGGish**, y de ahí nace la asociación.
{{< /concept-alert >}}

### Cuándo el probe preentrenado *no* ganaría

UrbanSound8K es casi el mejor caso posible para transferencia: sonidos ambientales urbanos con micrófonos de consumo, exactamente la distribución de la banda sonora de un video de YouTube. El lado útil de la pregunta es el inverso. VGGish pierde frente a una red pequeña entrenada en dominio cuando hay **desajuste de banda** (el front-end recorta a [125, 7500] Hz, así que para sonidos cardíacos o respiratorios por debajo de 125 Hz, ultrasonido o vibración industrial la información discriminativa **está fuera de lo que puede ver** — precisamente el escenario médico); cuando se necesita **resolución temporal fina**, que choca contra la unidad de 960 ms; cuando las **invarianzas son las equivocadas** (fue entrenada para ser invariante a quién habla y a la altura absoluta, justo lo que necesitarías para identificación de hablante o estimación de $f_0$); cuando el **dataset objetivo ya es grande**, como demostró PANNs; y cuando hay **desajuste de preprocesamiento** — que es exactamente a lo que se llega por accidente si uno se cree la confusión "log-mel vs MFCC" del notebook.

## Notas y enlaces

- **PDF:** [CNN Architectures for Large-Scale Audio Classification](/papers/vggish-hershey-2017.pdf) — ICASSP 2017, preprint `arXiv:1609.09430v2` (10 de enero de 2017). **No confundir con `arXiv:1610.00087`**, que es el paper de Dai et al.
- **Paper hermano, mismo ICASSP, cinco coautores compartidos:** [AudioSet: An Ontology and Human-Labeled Dataset for Audio Events](/papers/audioset-gemmeke-2017) — la mitad de datos del mismo esfuerzo.
- **La otra mitad del laboratorio:** [Very Deep Convolutional Neural Networks for Raw Waveforms](/papers/raw-waveforms-dai-2017) (Dai et al., 2017) y el dataset [UrbanSound8K](/papers/urbansound8k-salamon-2014) (Salamon et al., 2014).
- **La radicalización de esta tesis cuatro años después:** [AST: Audio Spectrogram Transformer](/papers/ast-gong-2021) (Gong et al., 2021).
- **Fundamentos relacionados:** [redes convolucionales](/fundamentos/redes-convolucionales), [transfer learning](/fundamentos/transfer-learning), [MFCC y escala mel](/fundamentos/mfcc-y-escala-mel), [clasificación de audio](/fundamentos/clasificacion-de-audio).
- **Recorrido del dominio:** [audio](/dominios/audio).
- **Clase donde se usa:** [Clase 39 — Modelos y arquitecturas para audio](/clases/clase-39).
