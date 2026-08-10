# AST: Audio Spectrogram Transformer (Gong, Chung y Glass, 2021) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** Yuan Gong, Yu-An Chung y James Glass. Los tres en el **MIT Computer Science and Artificial Intelligence Laboratory (CSAIL)**, Cambridge, MA. Es el mismo grupo que había publicado **PSLA** unos meses antes, el modelo CNN+atención que AST viene a desplazar. Vale la pena registrarlo: **el paper se autodestrona**. No es un ataque externo a la línea convolucional, es la auditoría interna de quienes tenían el estado del arte.
- **Venue:** Interspeech 2021. Preprint arXiv:2104.01778v3 (8 jul 2021).
- **Código:** `https://github.com/YuanGongND/ast` (indicado en la nota al pie de la página 1).
- **Financiamiento:** parcialmente apoyado por Signify (Sección 5).

La tesis en una línea: **la CNN no es indispensable para clasificar audio**. AST es "the first convolution-free, purely attention-based model for audio classification" (Abstract). Toma el espectrograma log-Mel, lo parte en parches de $16 \times 16$ **con solape**, los proyecta linealmente, les suma un embedding posicional entrenable, antepone un `[CLS]` y los pasa por un encoder Transformer estándar de 12 capas. Ninguna convolución en el sentido convencional, ningún sesgo inductivo de localidad más allá del que introduce el propio parcheo.

El problema obvio de esa receta es que los Transformers necesitan datos y en audio no los hay. La solución del paper es el aporte que más ha sobrevivido: **transferencia cross-modal desde ImageNet**, adaptando los pesos de un ViT/DeiT entrenado sobre fotos a un modelo que consume espectrogramas, con un mecanismo de *cut-and-interpolate* del embedding posicional que permite pasar de una grilla cuadrada de $24 \times 24$ a una grilla rectangular y de **longitud variable**.

Cifras ancla, todas del paper:

| Benchmark | AST | Estado del arte previo que desplaza | Fuente |
|---|---|---|---|
| **AudioSet full** (mAP, modelo único con *weight averaging*) | **0.459 ± 0.000** | 0.444 (PSLA single, CNN+atención) | Tabla 1 |
| **AudioSet full** (mAP, mejor ensamble) | **0.485** | 0.474 (PSLA Ensemble-M) | Tabla 1 |
| **AudioSet balanced** (mAP, modelo único) | **0.347 ± 0.001** | 0.319 (PSLA single) | Tabla 1 |
| **AudioSet balanced** (mAP, mejor ensamble) | **0.378** | 0.362 (PSLA Ensemble-M) | Tabla 1 |
| **ESC-50** (accuracy, con pre-entrenamiento en audio) | **95.6 ± 0.4** | 94.7 (PANNs) | Tabla 7 |
| **ESC-50** (accuracy, sin datos de audio adicionales) | **88.7 ± 0.7** | 86.5 (Sailor et al., ConvRBM) | Tabla 7 |
| **Speech Commands V2**, 35 clases (accuracy) | **98.11 ± 0.05** | 97.4 (MatchboxNet) / 97.7 (Lin et al., con 200M de audio de YouTube) | Tabla 7 |

Tres observaciones sobre estos números antes de entrar en detalle.

1. Las ganancias en AudioSet son **reales pero modestas en términos absolutos**: +1.5 puntos de mAP sobre PSLA single (0.444 → 0.459), +1.1 en el mejor ensamble. Lo notable no es el tamaño del salto sino que se consigue **eliminando por completo** el componente que todo el campo consideraba obligatorio.
2. Las ganancias en el régimen de **pocos datos** son proporcionalmente mayores: en AudioSet balanced (22k muestras, ~1% del full set) AST pasa de 0.319 a 0.347, un +8.8% relativo. Eso invierte la expectativa: se suponía que el Transformer sería el que sufriera con poco dato.
3. **Una misma arquitectura, sin ningún cambio, cubre entradas de 1 s (Speech Commands), 5 s (ESC-50) y 10 s (AudioSet)**, y contenido que va de habla a sonidos ambientales. Ese es el argumento de "clasificador de audio genérico" que el paper hace en la Sección 3.2 y que la línea convolucional no podía hacer: las CNN requerían *tuning* arquitectónico por tarea.

---

## 2. Contexto: por qué en 2021 todavía dominaban las CNN en audio

### 2.1. La línea VGGish → PANNs → PSLA

La receta canónica de clasificación de audio hacia 2020 tenía tres pasos fijos: (i) convertir la forma de onda en un espectrograma log-Mel, (ii) tratar ese espectrograma **como si fuera una imagen** y pasarlo por una CNN de visión, (iii) agregar sobre el tiempo (típicamente con *global average pooling*) y clasificar.

El paso (ii) es la clave histórica. Desde que Hershey et al. mostraron con **VGGish** que arquitecturas de ImageNet transferidas a espectrogramas funcionaban bien, el campo asumió que el espectrograma es una imagen y que por lo tanto merece un procesador de imágenes. **PANNs** (Kong et al., IEEE/ACM TASLP 2020, ref. [7] del paper) llevó eso a escala: una familia de CNN pre-entrenadas sobre AudioSet completo, con CNN14 como caballo de batalla, que alcanza **0.439 mAP en AudioSet full y 0.278 en balanced** (Tabla 1). **PSLA** (Gong, Chung y Glass, 2021, ref. [8]) — el trabajo previo de los mismos autores — combinó EfficientNet con *attention pooling* y una receta cuidada de pre-entrenamiento, muestreo balanceado, etiquetado y agregación, llegando a **0.444 single / 0.474 ensamble en full** y **0.319 / 0.362 en balanced**.

El sesgo inductivo que justificaba todo esto está enunciado en la introducción del paper: "the inductive biases inherent to CNNs such as **spatial locality** and **translation equivariance** are believed to be helpful". Nótese el "believed" (se cree). El paper está señalando desde la primera página que era una creencia, no un resultado medido.

### 2.2. Los híbridos CNN+atención: qué eran y por qué eran un compromiso

Aquí conviene ser preciso, porque "CNN+attention" cubre al menos tres construcciones distintas que el paper agrupa en la sección *Related Work*:

**(a) Atención como reemplazo del pooling final.** La CNN produce un mapa de features de baja resolución y, en vez de promediarlo sobre el tiempo, se aprende un peso por posición: $z = \sum_t \alpha_t \, h_t$ con $\alpha_t$ producido por una pequeña red. Es lo que hacen PANNs [7], PSLA [8] y el *attention pooling* de Li et al. [10]. Muy barato, pero la atención **solo actúa en la última capa** y sobre una secuencia ya colapsada.

**(b) Transformer apilado sobre la CNN.** Miyazaki et al. (DCASE 2020) [19] y Kong et al. (TASLP 2020) [20]: la CNN actúa como *front-end* que reduce el espectrograma a una secuencia corta de vectores, y encima corre un encoder Transformer completo. El paper lo describe textualmente: "the authors stack a Transformer on top of a CNN".

**(c) Convolución y atención entrelazadas en cada bloque.** El **Conformer** de Gulati et al. (Interspeech 2020) [21], que se volvió el estándar de facto en ASR: cada bloque tiene un módulo de auto-atención *y* un módulo convolucional, con la idea explícita de que la atención capture lo global y la convolución lo local. El paper: "the authors combine a Transformer and a CNN in each model block".

**Por qué las tres son un compromiso.** El argumento estructural es el mismo en los tres casos y AST lo enuncia en la introducción: el Transformer "can capture long-range global context **even in the lowest layers**", mientras que en un híbrido el contexto global solo aparece *después* de que la CNN ya comprimió. Concretamente: un espectrograma de AudioSet es de $1024 \times 128$; una EfficientNet lo reduce por un factor de 32 en cada eje, así que la atención opera sobre un mapa de aproximadamente $32 \times 4$. Eso es barato — 128 posiciones — pero significa que **la decisión de qué información sobrevive hasta la capa de atención ya fue tomada por convoluciones locales de $3\times3$**, capa tras capa, sin acceso a contexto global. Si un evento sonoro se define por la coocurrencia de energía en la banda 10 y en la banda 95 del eje mel, ninguna capa baja de la CNN puede representarlo; para cuando la atención puede verlo, el mapa ya perdió la resolución que lo hacía discriminativo.

Hay además un costo de ingeniería: el híbrido tiene **dos conjuntos de hiperparámetros arquitectónicos** (los de la CNN y los del Transformer) y el punto de corte entre ambos es una decisión de diseño más, que en la práctica se re-sintoniza por tarea.

### 2.3. Qué hacía falta para justificar una arquitectura puramente atencional

La pregunta de la introducción es explícita: "it is unclear whether the reliance on a CNN is necessary, and if neural networks purely based on attention are sufficient". Para responderla afirmativamente hacían falta tres cosas:

1. **Un método para segmentar el audio en tokens.** La auto-atención opera sobre un conjunto finito de entidades. En texto la segmentación viene dada; en audio, no. ViT [11] ya había demostrado la respuesta en imágenes — parches regulares — pero no era obvio que se trasladara a un objeto donde los dos ejes tienen semántica **distinta** (tiempo y frecuencia, no dos ejes espaciales intercambiables).
2. **Una fuente de datos.** ViT solo superaba a las CNN con más de 14 millones de imágenes (el paper lo cita textualmente en la Sección 2.2, referenciando a [11]). AudioSet tiene 2M de clips de 10 s, es decir unas 5.500 horas: mucho audio, pero un orden de magnitud menos ejemplos que JFT-300M o incluso que ImageNet-21k.
3. **Manejo de longitud variable.** Las CNN de audio ya lidiaban mal con esto (padding, *cropping*, arquitecturas por tarea). Una solución que no lo resolviera no sería mejor.

AST responde a (1) con el parcheo solapado, a (2) con la transferencia cross-modal, y a (3) gratis, porque el Transformer es agnóstico a la longitud por construcción — con el matiz del embedding posicional, que es justamente lo que el *cut-and-interpolate* soluciona.

---

## 3. La contribución central: la primera arquitectura de audio libre de convolución

### 3.1. Qué significa exactamente "convolution-free"

Es una afirmación fuerte y el paper tiene la honestidad de matizarla **él mismo**, en el último párrafo de la Sección 2.1:

> "Strictly speaking, the patch embedding layer can be viewed as a single convolution layer with a large kernel and stride size, and the projection layer in each Transformer block is equivalent to $1\times1$ convolution. However, the design is different from conventional CNNs that have multiple layers and small kernel and stride sizes. These Transformer models are usually referred to as convolution-free to distinguish them from CNNs."

Esto es literalmente cierto, y de hecho **toda implementación real de AST implementa el parcheo como un `nn.Conv2d(1, 768, kernel_size=16, stride=10)`**. Extraer parches solapados y proyectarlos linealmente *es* una convolución con kernel $16\times16$ y stride 10. Así que hay que ser preciso sobre qué se está afirmando:

| Propiedad de una CNN convencional | ¿La tiene AST? |
|---|---|
| Múltiples capas convolucionales apiladas | **No.** Una sola operación tipo convolución, en la entrada. |
| Kernels pequeños ($3\times3$) | **No.** Kernel $16\times16$. |
| Stride pequeño (1 o 2), solape alto | Parcialmente: stride 10 con kernel 16, un solape de 6. |
| Jerarquía de campos receptivos que crecen con la profundidad | **No.** Desde la capa 1 el campo receptivo es global. |
| Compartición de pesos con equivarianza traslacional en las capas profundas | **No.** El embedding posicional rompe la equivarianza deliberadamente. |
| *Pooling* espacial progresivo | **No.** La resolución de tokens es constante en las 12 capas. |

La afirmación defendible, entonces, no es "no hay ninguna operación expresable como convolución", sino: **ningún sesgo inductivo convolucional participa en el modelado**. La única operación tipo convolución es la tokenización — un preprocesamiento aprendido, sin no-linealidad después salvo la del bloque Transformer, sin apilamiento y sin jerarquía. Todo el modelado ocurre en auto-atención global. Es la misma convención terminológica que ViT [11] y DeiT [12] establecieron para imágenes, y el paper lo dice explícitamente ("usually referred to as convolution-free to distinguish them from CNNs").

### 3.2. Las tres ventajas que reclama el paper

De la introducción, textualmente resumidas:

1. **Desempeño superior** en AudioSet, ESC-50 y Speech Commands, sobre el estado del arte de cada uno.
2. **Soporte natural de entradas de longitud variable** y aplicabilidad a tareas distintas **sin cambio de arquitectura**: "the models we use for all aforementioned tasks have the same architecture while the input lengths vary from 1 sec. to 10 sec. In contrast, CNN-based models typically require architecture tuning to obtain optimal performance for different tasks."
3. **Arquitectura más simple, menos parámetros y convergencia más rápida** que los híbridos CNN+atención. La convergencia sí está cuantificada: AST necesita **5 épocas** en AudioSet full contra **30** de PSLA (Sección 3.1.2). El "fewer parameters" **no está cuantificado en ninguna parte del paper** y es dudoso en términos absolutos — ver Sección 13.

---

## 4. La arquitectura en detalle

### 4.1. El pipeline completo

Siguiendo la Sección 2.1 y la Figura 1:

```
waveform (16 kHz, t segundos)
  → banco de filtros log-Mel: 128 bandas, ventana Hamming de 25 ms, hop de 10 ms
  → espectrograma  128 × 100t
  → división en N parches de 16×16, stride 10 (solape 6) en AMBAS dimensiones
  → proyección lineal de cada parche (256 valores) a un embedding de 768
  → [CLS] prepended  →  + embedding posicional entrenable (768, uno por posición)
  → encoder Transformer: 12 capas, 12 cabezas, dim 768  (= ViT-Base / DeiT-Base)
  → salida del token [CLS] = representación del espectrograma
  → capa lineal + sigmoide → etiquetas
```

Puntos que merecen comentario porque son específicos del audio y no se deducen de conocer ViT:

**La tasa de 16 kHz y el frame rate de 100 Hz.** El paper no menciona explícitamente la frecuencia de muestreo en el texto, pero la relación "25 ms de ventana cada 10 ms → $100t$ frames" es la parametrización estándar de Kaldi (`torchaudio.compliance.kaldi.fbank`) y fija la resolución temporal de la entrada en **10 ms por frame**. Esto significa que un clip de 10 s produce ~1000 frames: para 2021, una secuencia larga incluso antes de parchear (BERT trabajaba con 512 tokens).

**Por qué log-Mel y no la forma de onda.** El paper no lo discute, lo hereda. Pero es la decisión que hace posible todo lo demás: el espectrograma convierte una señal de $16000 \cdot t$ muestras en una matriz de $128 \times 100t$, una reducción de **12.5×** en el eje temporal, y le da al objeto una estructura 2D con ejes interpretables. Sin ese paso, tokenizar audio para un Transformer sería un problema completamente distinto (es el problema que resuelven wav2vec 2.0 y HuBERT con un *stem* convolucional, que sí es una CNN).

**El eje de frecuencia es mel, no lineal.** 128 bandas mel comprimen las altas frecuencias y expanden las bajas, imitando la resolución perceptual del oído. Consecuencia para el parcheo: un parche de 16 bandas cubre un rango de frecuencias **muy distinto** según dónde esté en el eje. Un parche en las bandas 0–15 abarca quizá 0–250 Hz; uno en las bandas 112–127, varios kilohertz. El modelo tiene que aprender esa no-uniformidad, y lo único que se lo permite es el **embedding posicional**: sin él, dos parches de bandas distintas serían indistinguibles.

**La sigmoide, no softmax.** La cabeza usa "a linear layer with sigmoid activation" porque AudioSet es **multietiqueta** (527 etiquetas, un clip puede contener varias simultáneamente) y la pérdida es *binary cross-entropy* (Sección 3.1.1). Para ESC-50 y Speech Commands, que son multiclase, se usa la formulación estándar correspondiente.

**Solo el encoder.** "Since AST is designed for classification tasks, we only use the encoder of the Transformer." El paper insiste en usar "the original Transformer encoder architecture **without modification**" [18], por dos razones declaradas: (1) está disponible *off-the-shelf* en TensorFlow y PyTorch, y (2) — la razón que importa — **facilita la transferencia**. Cualquier modificación arquitectónica rompería la compatibilidad de pesos con los checkpoints de ViT. Es una decisión de diseño subordinada al truco de la Sección 5.

### 4.2. El número de parches: fórmula, verificación y una discrepancia real

El paper da:

$$N = 12 \left\lceil \frac{100t - 16}{10} \right\rceil$$

Descompongámosla. Con parches de lado $p = 16$ y solape $o = 6$, el **stride** es $s = p - o = 10$. El número de parches a lo largo de un eje de longitud $L$ es el número estándar de ventanas deslizantes:

$$n(L) = \left\lfloor \frac{L - p}{s} \right\rfloor + 1$$

**Eje de frecuencia** ($L = 128$, fijo): $n_f = \lfloor (128-16)/10 \rfloor + 1 = \lfloor 11.2 \rfloor + 1 = 12$. De ahí sale el 12 constante de la fórmula.

**Eje temporal** ($L = 100t$): $n_t = \lfloor (100t-16)/10 \rfloor + 1$. El paper escribe $\lceil (100t-16)/10 \rceil$, que coincide con $\lfloor \cdot \rfloor + 1$ salvo cuando el cociente es entero exacto. Es una simplificación notacional inofensiva.

**Verificación con el ejemplo del paper.** Para $t = 10$ s: $100t = 1000$, $n_t = \lfloor 984/10 \rfloor + 1 = 98 + 1 = 99$, y $N = 12 \times 99 = \mathbf{1188}$.

Pero el paper reporta **tres números distintos para lo mismo**:

- La fórmula da **1188**.
- La Sección 2.2 dice, al explicar el *cut-and-interpolate*: "An AST that takes 10-second audio input has **12 × 100** patches" → **1200**.
- La **Tabla 5** reporta, para la configuración Overlap-6 efectivamente usada: **1212** parches.

¿Cuál es el correcto? Reconstruyendo la Tabla 5 completa con distintas longitudes de entrada, la respuesta es inequívoca. La implementación **rellena (pad) el espectrograma a $T = 1024$ frames**, no a 1000:

| Configuración | Stride | $n_f$ | $n_t$ con $T=1000$ | $N$ con $T=1000$ | $n_t$ con $T=1024$ | $N$ con $T=1024$ | **Tabla 5** |
|---|---|---|---|---|---|---|---|
| No Overlap | 16 | 8 | 62 | 496 | 64 | **512** | 512 |
| Overlap-2 | 14 | 9 | 71 | 639 | 73 | **657** | 657 |
| Overlap-4 | 12 | 10 | 83 | 830 | 85 | **850** | 850 |
| Overlap-6 (usado) | 10 | 12 | 99 | 1188 | 101 | **1212** | 1212 |

**Las cuatro filas coinciden exactamente con $T = 1024$ y ninguna con $T = 1000$.** La Tabla 6 lo confirma de forma independiente: parches de $128\times2$ dan $1 \times (1024/2) = 512$, y parches de $32\times32$ dan $4 \times (1024/32) = 128$, ambos exactamente los valores reportados. Con $T=1000$ ninguno cuadra.

La explicación es prosaica: 1024 es la potencia de 2 inmediatamente superior a 1000, cómoda para el padding y para que la división sin solape sea exacta ($1024/16 = 64$, $128/16 = 8$). La fórmula del paper describe el caso ideal sin padding; la implementación usa el caso práctico. **La fórmula operativa correcta es:**

$$N = \left(\left\lfloor \frac{F - p}{s}\right\rfloor + 1\right)\left(\left\lfloor \frac{T_{\text{pad}} - p}{s}\right\rfloor + 1\right), \qquad F = 128,\ p = 16,\ s = 10,\ T_{\text{pad}} = 1024$$

Es una errata menor pero conviene tenerla clara si se va a implementar o a comparar costos.

### 4.3. Por qué el solape (y qué gana)

Esta es una de las divergencias deliberadas respecto de ViT, que **parte la imagen en parches disjuntos** ($224/16 = 14$, grilla $14\times14$, sin solape). AST usa stride 10 con kernel 16.

**Qué motiva el cambio.** Tres razones, una explícita del paper y dos que se deducen de la estructura del problema:

1. **Está medido y funciona** (Tabla 5): la mAP crece monótonamente con el solape, de 0.336 (sin solape) a 0.347 (solape 6) en balanced, y de 0.451 a 0.459 en full. El paper cita [13] (Tokens-to-Token ViT) como antecedente de la idea en visión.

2. **Elimina el artefacto de la grilla rígida.** En una imagen, partir en una grilla fija es benigno porque los objetos son grandes y redundantes. En un espectrograma, muchos eventos son **estructuras finas y localizadas**: un transitorio percusivo ocupa 2–3 frames (20–30 ms); un armónico ocupa 1–2 bandas mel. Con parches disjuntos, un transitorio que cae justo sobre el borde entre dos parches queda partido en dos mitades incompletas, y ninguna de las dos contiene el patrón. Con solape 6, **cada punto del espectrograma aparece en varios parches con distintos desplazamientos**, así que siempre existe al menos un parche donde el evento está bien centrado. Es, en efecto, una forma barata de recuperar algo de la **equivarianza traslacional** que se perdió al abandonar la convolución.

3. **Aumenta la densidad de tokens sin cambiar la resolución de entrada.** De 512 a 1212 tokens: 2.37× más contexto para la atención con el mismo espectrograma.

**Qué cuesta.** El paper lo señala sin ambigüedad: "increasing the overlap also leads to longer patch sequence inputs to the Transformer, which will **quadratically increase the computational overhead**". Pasar de 512 a 1212 tokens multiplica el costo de la matriz de atención por $(1212/512)^2 = 5.6$. Comprar 1.1 puntos de mAP con 5.6× de cómputo cuadrático es un intercambio malo en producción y el propio paper ofrece la salida: "Even with no patch split overlap, AST can still outperform the previous best system in [8]" — 0.451 sin solape contra 0.444 de PSLA.

**Nota sobre la asimetría que el paper no explota.** El solape se aplica **igual en tiempo y en frecuencia** ("an overlap of 6 in both time and frequency dimension"), pero los dos ejes no son equivalentes: en frecuencia hay 128 bandas fijas y el fenómeno relevante (armónicos) es de estructura amplia; en tiempo hay 1024 frames y el fenómeno relevante (transitorios) es de estructura fina. Un solape asimétrico —mayor en tiempo, menor o nulo en frecuencia— parecería la elección natural y reduciría el conteo de tokens. El paper no lo prueba. Es un hueco del estudio, no un error.

---

## 5. La transferencia cross-modal desde ImageNet: el truco central

### 5.1. La motivación

La Sección 2.2 abre con el diagnóstico: "One disadvantage of the Transformer compared with CNNs is that the Transformer **needs more data to train**. In [11], the authors point out that the Transformer only starts to outperform CNNs when the amount of data is over **14 million** for image classification tasks. However, **audio datasets typically do not have such large amounts of data**."

Esa frase es exactamente la objeción 1 de la clase 39, formulada por los propios autores. La diferencia es que el paper la formula para **resolverla**, no para concluir que el enfoque no sirve.

La observación que habilita la solución está en la misma sección: "images and audio spectrograms have **similar formats**". Ambos son tensores 2D densos con estructura local. Y hay precedente: "Transfer learning from vision tasks to audio tasks has been previously studied in [23, 24, 25, 8], **but only for CNN-based models**". Es decir, la transferencia ImageNet → audio ya era práctica establecida en la línea convolucional (VGGish, ESResNet, el propio PSLA); AST la extiende a arquitecturas atencionales, donde no era obvio que funcionara porque el objeto a transferir es distinto.

El argumento pragmático cierra: "it is computationally expensive to train a state-of-the-art vision model, but many commonly used architectures have off-the-shelf ImageNet-pretrained models for both TensorFlow and PyTorch, making transfer learning much easier".

### 5.2. Los tres ajustes

#### (a) Promedio de los tres canales RGB y normalización de la entrada

**El problema:** la capa de embedding de parches de ViT tiene pesos de forma $[768, 3, 16, 16]$ — tres canales de entrada. AST recibe un espectrograma de **un solo canal**.

**La solución:** "we **average the weights** corresponding to each of the three input channels of the ViT patch embedding layer and use them as the weights of the AST patch embedding layer."

Es decir, $W_{\text{AST}}[o,0,i,j] = \frac{1}{3}\sum_{c=0}^{2} W_{\text{ViT}}[o,c,i,j]$, quedando en forma $[768, 1, 16, 16]$.

**Por qué es la elección correcta**, y el paper lo justifica con precisión: "This is **equivalent to expanding a single-channel spectrogram to 3-channels with the same content**, but is computationally more efficient." La equivalencia es exacta y vale la pena verla escrita. Si se replicara el espectrograma $x$ en los tres canales, la salida de la convolución sería

$$y = \sum_{c=0}^{2} W_c * x = \left(\sum_c W_c\right) * x$$

Promediar en vez de sumar solo introduce un factor $1/3$ en la escala de salida, que es precisamente lo que se quiere: el promedio preserva la **magnitud** típica de las activaciones que las capas siguientes esperan, mientras que la suma las triplicaría. No es una aproximación: es la misma función, con el tercio de FLOPS en la capa de parcheo.

**La normalización.** "We also normalize the input audio spectrogram so that the dataset mean and standard deviation are **0 and 0.5**, respectively."

El 0 es esperable. El **0.5 es lo interesante** y el paper no lo explica. La lectura razonable: los pipelines de ViT/DeiT normalizan las imágenes a media 0.5 y desviación 0.5 (o con las estadísticas de ImageNet), lo que deja los píxeles en un rango aproximado de $[-1, 1]$. Los pesos preentrenados de la primera capa fueron optimizados para activaciones de esa escala. Si el espectrograma llegara con desviación 1, las activaciones de la primera capa serían el doble de grandes de lo que el resto de la red espera, y la LayerNorm posterior lo compensaría solo parcialmente al inicio. Fijar $\sigma = 0.5$ **alinea el rango dinámico de la entrada de audio con el rango dinámico de la entrada de imagen para la que los pesos fueron entrenados**. Esta es interpretación, no cita: el paper solo enuncia el valor.

#### (b) El cut-and-interpolate del embedding posicional

Este es el corazón del paper y el mecanismo que más se ha reutilizado después.

**El problema.** ViT tiene un embedding posicional **entrenable** de forma $[1 + n_{\text{patches}}, 768]$. Ese tensor no es genérico: "it **learns to encode the spatial information** during the ImageNet training". Es decir, después de entrenar, los embeddings posicionales de parches vecinos son similares entre sí y la estructura de la grilla 2D está codificada en ellos. Descartarlos es descartar información real.

Pero la forma no calza. El paper da el ejemplo concreto: un ViT que toma imágenes de $384 \times 384$ con parches de $16\times16$ **sin solape** tiene $24 \times 24 = 576$ parches, y por tanto 576 embeddings posicionales (más el del `[CLS]`). AST necesita una grilla de $12 \times n_t$ donde $n_t$ **depende de la duración del audio** y por tanto cambia entre tareas: ~101 para AudioSet (10 s), ~51 para ESC-50 (5 s), ~11 para Speech Commands (1 s).

**El mecanismo.** El nombre lo dice todo: *cut* en un eje, *interpolate* en el otro.

```
pos_embed de ViT:  [576, 768]
  → reshape a grilla 2D:            [768, 24, 24]        (canales, alto=freq, ancho=tiempo)
  → CUT en el eje de frecuencia:    [768, 12, 24]        (24 → 12: se descartan filas)
  → INTERPOLATE bilineal en tiempo: [768, 12, 101]       (24 → 101: se estira)
  → flatten:                        [1212, 768]
  → el pos_embed del [CLS] se reutiliza tal cual
```

Textualmente: "We therefore **cut the first dimension** and **interpolate the second dimension** of the $24\times24$ ViT positional embedding to $12 \times 100$ and use it as the positional embedding for the AST. We **directly reuse the positional embedding for the [CLS] token**."

**Por qué cortar en un eje e interpolar en el otro, y no lo mismo en ambos.** Esta es la parte que revela que la decisión es sobre **semántica de ejes**, no sobre aritmética:

- El **eje de frecuencia** va de 24 a 12 posiciones. Interpolar hacia abajo mezclaría embeddings de bandas adyacentes, difuminando la distinción entre "grave" y "agudo" que es la señal más discriminativa del espectrograma. Además, la grilla de frecuencia de AST es **fija** (128 bandas siempre): no necesita ser elástica. Cortar preserva 12 embeddings **intactos y mutuamente distinguibles**, que es exactamente lo que hace falta para que el modelo sepa en qué banda está cada parche.
- El **eje temporal** va de 24 a un valor que **cambia con la tarea**. Aquí sí se necesita elasticidad, y la interpolación la da: estirar la estructura aprendida "izquierda-a-derecha" del ViT sobre la longitud que sea. La monotonía relativa se preserva —los embeddings de frames vecinos siguen siendo parecidos— que es la propiedad que importa.

El principio general: **se corta el eje rígido y se interpola el eje elástico**. Es una decisión de modelado, no de conveniencia numérica, y es lo que hace que la misma arquitectura funcione con entradas de 1 a 10 segundos sin retocar nada.

Un detalle que el paper **no especifica**: si el corte de $24 \to 12$ toma las primeras 12 filas o las 12 centrales. La implementación oficial recorta **desde el centro** (lo cual conserva la región de la grilla ViT con estadísticas más regulares, evitando los bordes donde los embeddings posicionales suelen ser atípicos). El texto del paper solo dice "cut the first dimension".

**Por qué esto no es trivial.** Uno podría pensar: "el embedding posicional es aprendible, se reinicializa y listo". El paper mide exactamente eso y la respuesta es no — ver la ablación en la Sección 8.

#### (c) Reemplazo del cabezal

"Since the classification task is essentially different, we **abandon the last classification layer of the ViT and reinitialize a new one** for AST."

Trivial y esperable: 1000 clases de ImageNet no tienen nada que ver con las 527 de AudioSet, las 50 de ESC-50 ni las 35 de Speech Commands. Lo único a notar es que AST además cambia la **activación** de la cabeza (sigmoide para el caso multietiqueta de AudioSet) y la función de pérdida (BCE), cosa que el paper menciona en la Sección 2.1 y en 3.1.1 pero no en la lista de adaptaciones.

### 5.3. ¿Por qué funciona transferir de imágenes naturales a espectrogramas?

Es la pregunta legítima. Las distribuciones son radicalmente distintas: una foto tiene objetos, oclusión, iluminación, perspectiva; un espectrograma tiene rayas horizontales (armónicos sostenidos), rayas verticales (transitorios), manchas difusas (ruido de banda ancha) y trayectorias curvas (barridos de frecuencia, formantes de habla). Un espectrograma no se parece a una foto de un perro.

Las razones por las que la transferencia funciona igual, en orden de importancia:

**1. Lo que se transfiere no es semántica sino estadística de segundo orden.** Los filtros de la primera capa de una red entrenada sobre imágenes naturales convergen invariablemente a un banco de **detectores de bordes orientados, blobs y patrones de frecuencia espacial** — es el resultado clásico de Olshausen y Field sobre estadísticas de escenas naturales. Esos operadores no son "sobre fotos": son sobre **campos 2D con correlaciones locales**. Y un espectrograma es exactamente eso. Un detector de bordes horizontales se convierte en un detector de tono sostenido; uno de bordes verticales, en un detector de onset percusivo; uno de gradiente diagonal, en un detector de barrido de frecuencia. **La transferencia funciona porque la primitiva es geométrica, no semántica.**

**2. Lo que más se transfiere es el bloque Transformer, y ese es casi agnóstico al dominio.** De los ~87M de parámetros, la capa de parcheo tiene $768 \times 3 \times 16 \times 16 \approx 590$k, menos del 1%. El otro 99% son las 12 capas de atención y MLP, que codifican algo mucho más abstracto: **cómo enrutar información entre posiciones**, cómo usar las cabezas para especializarse en distintos patrones de dependencia, cómo mantener el flujo residual estable. Nada de eso es específico de imágenes. Es, esencialmente, una **inicialización bien acondicionada de un Transformer profundo** — el equivalente de partir desde un punto del espacio de pesos donde la optimización ya funciona, en vez de desde ruido gaussiano.

**3. El embedding posicional transfiere una prior geométrica genuina.** La ablación (Tabla 4) demuestra que esto es medible: reinicializarlo cuesta 4.2 puntos de mAP. Lo que se transfiere es "las posiciones vecinas en 2D deben tener representaciones parecidas", que es cierto tanto en fotos como en espectrogramas.

**4. La distribución de destino tiene menos variabilidad, no más.** Los espectrogramas log-Mel son un objeto mucho más restringido que las imágenes naturales: siempre 128 filas, siempre el mismo eje semántico, siempre energías no negativas comprimidas logarítmicamente. Un modelo con capacidad para ImageNet está sobredimensionado para la variabilidad del espectrograma, lo que hace que la inicialización sea generosa más que restrictiva.

**5. El efecto se concentra donde hay poco dato, lo que confirma que es un efecto de regularización.** Comparando la Tabla 2: en balanced (22k muestras) el pre-entrenamiento vale $0.347/0.148 = 2.3\times$; en full (2M) vale $0.459/0.366 = 1.25\times$. El paper lo dice: "The performance improvement of ImageNet pretraining is **more significant when the training data volume is smaller**".

---

## 6. DeiT y la destilación

### 6.1. De qué modelo parte exactamente AST

El paper es específico (Sección 2.2, final): "we use pretrained weights of a **data-efficient image Transformer (DeiT)** [12], which is trained with **CNN knowledge distillation**, **$384\times384$** images, has **87M parameters**, and achieves **85.2% top-1 accuracy on ImageNet 2012**".

Es decir: **DeiT-Base distilled, fine-tuneado a resolución 384** (en la nomenclatura de Touvron et al., `DeiT-B⚗↑384`). Ese es el checkpoint fuente. La elección no es arbitraria: la Tabla 3 la justifica empíricamente contra tres alternativas.

Vale precisar por qué DeiT y no ViT. DeiT (Touvron et al., 2020) fue justamente la respuesta al problema que motiva la Sección 2.2 de AST: ViT necesitaba JFT-300M para superar a las CNN, y DeiT mostró que con una receta de entrenamiento adecuada —augmentación fuerte, regularización, y **destilación desde un profesor CNN**— se podían entrenar Transformers de visión competitivos usando **solo ImageNet-1k**. Es decir, DeiT es el ViT que ya había resuelto el problema de la escasez de datos en su propio dominio. AST hereda ese trabajo.

### 6.2. El token de destilación de DeiT y cómo lo maneja AST

**Qué es en DeiT.** La contribución arquitectónica de DeiT es un token extra, aprendido, análogo al `[CLS]` pero con un objetivo distinto. La secuencia de entrada es `[CLS], [DIST], parche_1, ..., parche_N`. Ambos tokens especiales atraviesan el mismo encoder. Al final:

- La salida de `[CLS]` va a una cabeza que se entrena contra la **etiqueta verdadera** (cross-entropy supervisada).
- La salida de `[DIST]` va a **otra** cabeza que se entrena contra la **predicción de un profesor CNN** (típicamente una RegNet), es decir, contra una etiqueta blanda o dura producida por el profesor.

En inferencia, DeiT promedia las salidas de ambas cabezas. La gracia del diseño es que el token de destilación permite que la señal del profesor influya en la red **a través de la atención**, en todas las capas, en lugar de solo en la pérdida final — de ahí el título del paper, "distillation through attention".

**Cómo lo maneja AST.** Una línea, la última de la Sección 2.2: "During ImageNet training, DeiT has **two [CLS] tokens; we average them as a single [CLS] token** for audio training."

Concretamente: AST toma los dos embeddings de token especiales del checkpoint, calcula su promedio elemento a elemento, y lo usa como el único token `[CLS]` de AST. Lo mismo aplica a sus embeddings posicionales. Y como los dos cabezales de clasificación se descartan de todos modos (ajuste (c) de la Sección 5), no queda ningún rastro de la estructura dual.

**Tres precisiones importantes, porque este punto se cita mal a menudo:**

1. **AST no hace destilación.** No hay profesor CNN, no hay pérdida de destilación, no hay token de destilación en el modelo de audio. La destilación ocurrió enteramente **río arriba**, durante el entrenamiento de DeiT sobre ImageNet. AST simplemente cosecha los pesos resultantes.
2. **La destilación sí importa, pero indirectamente.** La Tabla 3 muestra que "DeiT w/o Distill" (86M, 82.9% ImageNet) da **0.330** en AudioSet balanced, mientras "DeiT w/ Distill" (87M, 85.2%) da **0.347**. Son **1.7 puntos de mAP de diferencia** atribuibles a que el modelo fuente fue destilado. Es una cadena de causalidad curiosa: la señal de un profesor CNN sobre imágenes acaba mejorando un clasificador de audio libre de convoluciones.
3. **Promediar los dos tokens es una heurística, no un resultado.** El paper no la ablaciona. Alternativas razonables (quedarse solo con `[CLS]`, solo con `[DIST]`, o mantener ambos y concatenar las representaciones) no se prueban. Es una de las varias decisiones que el paper toma sin justificación empírica.

---

## 7. Experimentos y resultados

### 7.1. Setup

**AudioSet** (Gemmeke et al., ICASSP 2017): más de 2 millones de clips de 10 s extraídos de YouTube, etiquetados con 527 clases de una ontología de eventos sonoros; multietiqueta y **débilmente etiquetado** (se sabe qué suena en el clip, no cuándo). Splits: **22k (balanced train) / 2M (full train) / 20k (eval)**. Métrica: **mAP** (mean average precision).

El paper es explícito sobre la equidad de la comparación: "we use the **exact same training pipeline with [8]**" — es decir, con PSLA. Esto incluye pre-entrenamiento en ImageNet, muestreo balanceado (solo en full), *mixup* con ratio 0.5, enmascaramiento de espectrograma (SpecAugment, con máscaras de hasta 192 frames en tiempo y 48 bins en frecuencia), promediado de pesos y ensamblado. Y añade: "**[8] also use ImageNet pretraining, so it is a fair comparison**". Esto es importante: la ganancia de AST **no** se debe a que use ImageNet y el baseline no.

Hiperparámetros de AudioSet: batch 12, Adam, BCE. Balanced: lr $5\times10^{-5}$, 25 épocas, lr a la mitad cada 5 épocas después de la 10.ª. Full: lr $1\times10^{-5}$, **5 épocas**, lr a la mitad cada época después de la 2.ª.

### 7.2. AudioSet

**Tabla 1** (mAP; "Architecture" indica el tipo de modelo):

| Modelo | Arquitectura | Balanced mAP | Full mAP |
|---|---|---|---|
| Baseline [15] | CNN + MLP | — | 0.314 |
| PANNs [7] | CNN + Atención | 0.278 | 0.439 |
| PSLA (Single) [8] | CNN + Atención | 0.319 | 0.444 |
| PSLA (Ensemble-S) | CNN + Atención | 0.345 | 0.464 |
| PSLA (Ensemble-M) | CNN + Atención | 0.362 | 0.474 |
| **AST (Single)** | **Atención pura** | **0.347 ± 0.001** | **0.459 ± 0.000** |
| **AST (Ensemble-S)** | **Atención pura** | **0.363** | **0.475** |
| **AST (Ensemble-M)** | **Atención pura** | **0.378** | **0.485** |

Desglose de las variantes (Sección 3.1.2):
- **Single** = modelo único con *weight averaging* sobre todos los checkpoints de época. En full, el modelo de la última época solo da **0.448 ± 0.001**; el promediado de pesos lo sube a **0.459 ± 0.000** — es decir, **1.1 puntos gratis**, sin aumentar el tamaño del modelo.
- **Ensemble-S** = 3 corridas con el mismo setup y distintas semillas, promediando las salidas.
- **Ensemble-M** = los 3 de Ensemble-S más 3 modelos con distintas estrategias de parcheo (en balanced, **11 modelos** con distintas semillas, pesos preentrenados, interpolación posicional y estrategias de parcheo).

**Lectura crítica de las magnitudes:**

- **Ganancia sobre PSLA single en full: +1.5 mAP (0.444 → 0.459), un +3.4% relativo.** Es una ganancia sólida pero **modesta**. Presentarla como un salto grande sería exagerar.
- **Ganancia sobre PSLA single en balanced: +2.8 mAP (0.319 → 0.347), un +8.8% relativo.** Esta sí es sustancial, y es la que más significa conceptualmente: refuta la expectativa de que el Transformer necesitaría más datos. El paper lo dice: "This demonstrates that AST can work better than CNN-attention hybrid models **even when the training set is relatively small**."
- **Ganancia en ensambles: +1.1 mAP (0.474 → 0.485)**, y con **menos modelos**: "we use fewer models (6) for our best ensemble models than [8] (10)".
- **Eficiencia de entrenamiento: la ganancia menos visible pero quizá más práctica.** "AST only needs **5 training epochs**, while in [8], the CNN-attention hybrid model is trained for **30 epochs**." Un factor **6×** en épocas sobre un dataset de 2M clips.

### 7.3. ESC-50

**ESC-50** (Piczak, 2015): 2.000 grabaciones ambientales de 5 s en 50 clases, evaluadas con **validación cruzada estándar de 5 pliegues** (1.600 muestras de entrenamiento por pliegue). Es un dataset **muy pequeño**, lo que lo convierte en el test de estrés del argumento de eficiencia de datos.

Dos regímenes: **-S** (sin datos de audio adicionales, solo pre-entrenamiento en ImageNet) y **-P** (con pre-entrenamiento adicional en AudioSet).

| | ESC-50 (accuracy %) |
|---|---|
| SOTA-S [33] (Sailor et al., ConvRBM) | 86.5 |
| SOTA-P [7] (PANNs) | 94.7 |
| **AST-S** | **88.7 ± 0.7** |
| **AST-P** | **95.6 ± 0.4** |

Ganancias: **+2.2 puntos** en el régimen sin audio adicional, **+0.9** con AudioSet. La segunda es marginal (aunque supera claramente la desviación estándar de 0.4). La primera es la interesante y el paper la subraya: "although ESC-50 has 1,600 training samples for each fold, **AST still works well with such a small amount of data even without AudioSet pretraining**".

Hiperparámetros: batch 48, Adam, 20 épocas, lr $1\times10^{-4}$ (AST-S) o $1\times10^{-5}$ (AST-P), decaimiento por factor 0.85 por época después de la 5.ª. Augmentación: enmascaramiento en frecuencia y tiempo.

### 7.4. Speech Commands V2

**Speech Commands V2** (Warden, 2018): 105.829 grabaciones de 1 s de 35 comandos hablados. Splits: 84.843 / 9.981 / 11.005. Se evalúa la tarea de **35 clases**.

| | Speech Commands V2, 35 clases (accuracy %) |
|---|---|
| SOTA-S [34] (MatchboxNet, CNN separable time-channel 1D) | 97.4 |
| SOTA-P [35] (Lin et al., CNN + 200M de audio de YouTube) | 97.7 |
| **AST-S** | **98.11 ± 0.05** |
| **AST-P** | **97.88 ± 0.03** |

**El resultado más contraintuitivo del paper está aquí: AST-S supera a AST-P.** "We find AudioSet pretraining **unnecessary** for the speech command classification task as AST-S outperforms AST-P." Pre-entrenar en AudioSet **empeora** el resultado en 0.23 puntos (98.11 → 97.88), una diferencia pequeña pero muy por encima de las desviaciones estándar (0.05 y 0.03).

La interpretación razonable, que el paper no desarrolla: AudioSet es abrumadoramente **no-habla** (eventos ambientales, música, ruido), y sus 10 s por clip tienen una estructura temporal completamente distinta de 1 s de una palabra aislada. El pre-entrenamiento en AudioSet especializa el modelo en distinciones tímbricas de banda ancha, no en la estructura de formantes que discrimina "left" de "right". Es un caso de **desajuste de dominio dentro del propio audio**, lo cual es un recordatorio útil: "más pre-entrenamiento" no es monótonamente mejor, y ImageNet resulta ser un punto de partida más **neutro** que AudioSet para tareas de habla.

Ganancia neta sobre el SOTA: **+0.7 puntos** (97.4 → 98.11) en el régimen comparable, y **+0.4** sobre el modelo que usó 200 millones de audios de YouTube. Son **ganancias marginales en términos absolutos** — con 98.11% quedan 208 errores de 11.005 — pero el argumento no es la magnitud sino que se consiguen con **la misma arquitectura sin ningún cambio**, contra un modelo (MatchboxNet) diseñado específicamente para *keyword spotting*.

Hiperparámetros: batch 128, Adam, hasta 20 épocas, lr $2.5\times10^{-4}$, decaimiento 0.85 por época después de la 5.ª, selección del mejor modelo por validación. Augmentación: enmascaramiento en frecuencia y tiempo, ruido aleatorio y *mixup*.

### 7.5. El argumento de generalidad

El cierre de la Sección 3.2 es la síntesis del paper: "while the input audio length varies from 1 sec. (Speech Commands), 5 sec. (ESC-50) to 10 sec. (AudioSet) and content varies from speech to non-speech, **we use a fixed AST architecture for all three benchmarks and achieve SOTA results on all of them**. This indicates the potential for AST use as a **generic audio classifier**."

Es el argumento más fuerte del paper y el que menos depende de cuán grande sea cada ganancia individual.

---

## 8. Ablations

Todas sobre **AudioSet balanced** salvo indicación contraria, con *weight averaging* y **sin** ensambles (Sección 3.1.3), para ahorrar cómputo.

### 8.1. Impacto del pre-entrenamiento en ImageNet (Tabla 2)

**Es la ablación más reveladora del paper.**

| | Balanced Set | Full Set |
|---|---|---|
| **No Pretrain** | **0.148** | **0.366** |
| ImageNet Pretrain (usado) | **0.347** | **0.459** |

- En **balanced**: 0.148 → 0.347. El pre-entrenamiento **multiplica la mAP por 2.34**, o dicho al revés: **sin ImageNet, AST pierde el 57% de su desempeño**. Y 0.148 es un resultado catastrófico — está muy por debajo de PANNs (0.278) y por debajo de cualquier baseline serio.
- En **full**: 0.366 → 0.459, una mejora de **+25% relativo**. Aquí el modelo sin pre-entrenar al menos supera al baseline CNN+MLP de AudioSet (0.314), pero **queda por debajo de PANNs (0.439) y de PSLA (0.444)**.

**Traducción sin adornos: sin transferencia cross-modal, AST no habría sido publicable.** Con 2 millones de clips de entrenamiento, un Transformer puro entrenado desde cero pierde contra las CNN del estado del arte. La contribución arquitectónica y la contribución de transferencia **no son separables**: la primera solo funciona gracias a la segunda.

El paper lo enuncia de forma más diplomática pero igual de clara: "demonstrating that ImageNet pretraining can **greatly reduce the demand for in-domain audio data** for AST".

### 8.2. Qué pesos preentrenados usar (Tabla 3) — modelo fuente y tamaño

| Pesos iniciales | # Params | ImageNet top-1 | AudioSet balanced mAP |
|---|---|---|---|
| ViT Base [11] | 86M | 0.846 | 0.320 |
| ViT Large [11] * | 307M | 0.851 | 0.330 |
| DeiT sin destilación [12] | 86M | 0.829 | 0.330 |
| **DeiT con destilación (usado)** | **87M** | **0.852** | **0.347** |

(*) "Model is trained without patch split overlap due to memory limitation."

**Sobre el tamaño del modelo.** Aquí está el resultado que suele leerse mal. **ViT-Large tiene 3.6× los parámetros de ViT-Base (307M vs 86M) y solo gana 1 punto de mAP** (0.330 vs 0.320) — y ni siquiera es una comparación limpia, porque ViT-Large tuvo que entrenarse **sin solape de parches** por límites de memoria, lo que según la Tabla 5 cuesta ~1.1 puntos. Corrigiendo por eso, ViT-Large probablemente estaría en torno a 0.341, todavía por debajo de DeiT-Base-distilled (0.347) con **3.5× menos parámetros**.

Conclusión: **escalar el modelo no es la palanca; escalar la calidad del checkpoint fuente sí**. Y el hecho de que ViT-Large no cupiera en memoria con solape es en sí mismo un dato sobre los costos de AST (Sección 10).

**Sobre la correlación ImageNet ↔ AudioSet.** El paper afirma: "We find that AST using the weights of the DeiT model with distillation that performs best on ImageNet2012 also performs best on AudioSet." Es cierto para el **ganador**, pero la correlación **no es monótona** en toda la tabla: DeiT sin destilación tiene *peor* accuracy en ImageNet que ViT-Base (0.829 vs 0.846) y sin embargo *mejor* mAP en AudioSet (0.330 vs 0.320). Es decir, lo que transfiere bien no es puramente "accuracy en ImageNet" — la **receta de entrenamiento** de DeiT (augmentación fuerte, regularización) parece transferir independientemente de la accuracy alcanzada. El paper no lo comenta.

### 8.3. Estrategia de adaptación del embedding posicional (Tabla 4)

| | Balanced Set mAP |
|---|---|
| Reinicializar | 0.305 |
| Interpolación por vecino más cercano | 0.346 |
| **Interpolación bilineal (usada)** | **0.347** |

Dos lecturas:

1. **Reinicializar cuesta 4.2 puntos de mAP** (0.347 → 0.305). El paper: "reinitializing the positional embedding does not completely break the pretrained model as the model still performs better than a fully randomly reinitialized model [0.148], but it does lead to a **noticeable performance drop**. This demonstrates the **importance of transferring spatial knowledge**." Descomponiendo el efecto total del pre-entrenamiento: de los 19.9 puntos que aporta ImageNet (0.148 → 0.347), **4.2 vienen específicamente del embedding posicional** y los 15.7 restantes de los pesos del encoder y del parcheo. El embedding posicional representa menos del 1% de los parámetros y aporta el **21% de la ganancia de transferencia**. Ese es el argumento de que el *cut-and-interpolate* es una contribución real y no un detalle de implementación.

2. **El método de interpolación es irrelevante**: 0.346 vs 0.347, dentro del ruido. "Bi-linear interpolation and nearest-neighbor interpolation do not result in a big difference." Lo que importa es **que se preserve la estructura de la grilla**, no cómo exactamente se remuestrea.

### 8.4. Solape entre parches (Tabla 5)

| Configuración | # Parches | Balanced mAP | Full mAP |
|---|---|---|---|
| No Overlap | 512 | 0.336 | 0.451 |
| Overlap-2 | 657 | 0.342 | 0.456 |
| Overlap-4 | 850 | 0.344 | 0.455 |
| **Overlap-6 (usado)** | **1212** | **0.347** | **0.459** |

- Monotonía casi perfecta: la única excepción es Overlap-4 en full (0.455 vs 0.456 de Overlap-2), una inversión de 0.001 dentro del ruido.
- **Ganancia total del solape: +1.1 puntos en balanced (0.336 → 0.347) y +0.8 en full (0.451 → 0.459).**
- **Costo: 2.37× más tokens, es decir 5.6× la matriz de atención.** El paper es explícito sobre lo cuadrático.
- **La observación práctica más útil de toda la tabla:** "Even with no patch split overlap, AST can still outperform the previous best system in [8]" — 0.451 sin solape contra 0.444 de PSLA en full. Es decir, **la tesis del paper (la atención pura basta) se sostiene sin el truco caro.** Para un despliegue real, la configuración sin solape es claramente la razonable.

### 8.5. Forma y tamaño de parche (Tabla 6)

Todos entrenados **sin solape**.

| Forma de parche | # Parches | Sin pre-entrenar | Con pre-entrenar |
|---|---|---|---|
| $128 \times 2$ (rectangular, en orden temporal) | 512 | **0.154** | — |
| **$16 \times 16$ (usado)** | 512 | 0.143 | **0.336** |
| $32 \times 32$ | 128 | 0.139 | — |

Esta ablación es la más honesta del paper y la que revela mejor la naturaleza real de la contribución.

**El punto de comparación es un parcheo alternativo "natural" para audio**: rebanar el espectrograma en columnas de $128 \times 2$, es decir, **el espectro completo en cada instante, en orden temporal estricto**. Eso convierte el audio en una secuencia 1D genuina —como una oración— y elimina la necesidad de razonar sobre una grilla 2D. Es lo que uno haría si viniera de RNN/ASR.

Y **funciona mejor**: 0.154 contra 0.143, un +7.7% relativo, con la misma área de parche (256) y el mismo número de tokens (512). El paper lo reconoce: "using $128\times2$ rectangle patches leads to better performance than using $16\times16$ square patches **when both models are trained from scratch**."

**Pero se descarta**, y la razón es puramente pragmática: "considering there is **no $128\times2$ patch based ImageNet pretrained model**, using $16\times16$ patches is still the current optimal solution."

Ese es un dato conceptualmente importante y hay que leerlo con precisión: **AST elige $16\times16$ no porque sea la mejor tokenización para audio, sino porque es la única compatible con los pesos de ImageNet.** La arquitectura está subordinada al truco de transferencia. Y la aritmética lo confirma: la mejor tokenización sin pre-entrenar da 0.154; la peor tokenización *con* pre-entrenar da 0.336. **El pre-entrenamiento vale más del doble que cualquier decisión de tokenización.**

**Sobre el tamaño:** "smaller size patches lead to better performance" — $32\times32$ da 0.139 contra 0.143 de $16\times16$. Diferencia pequeña, pero consistente con la idea de que más tokens (512 vs 128) dan más resolución a la atención.

### 8.6. Tasa de aprendizaje

**El paper no incluye ninguna ablación de tasa de aprendizaje.** No hay tabla ni discusión al respecto. Lo único que se puede decir con base en el texto son las tasas efectivamente usadas, que sí revelan un patrón:

| Experimento | lr inicial | Schedule |
|---|---|---|
| AudioSet balanced | $5\times10^{-5}$ | mitad cada 5 épocas después de la 10.ª, 25 épocas |
| AudioSet full | $1\times10^{-5}$ | mitad cada época después de la 2.ª, 5 épocas |
| ESC-50, AST-S | $1\times10^{-4}$ | ×0.85 por época después de la 5.ª, 20 épocas |
| ESC-50, AST-P | $1\times10^{-5}$ | ×0.85 por época después de la 5.ª, 20 épocas |
| Speech Commands | $2.5\times10^{-4}$ | ×0.85 por época después de la 5.ª, hasta 20 épocas |

Dos observaciones que sí se sostienen sobre estos números: (i) las tasas son **entre uno y dos órdenes de magnitud menores** que las típicas de una CNN entrenada desde cero, lo cual es lo esperable al hacer *fine-tuning* de un modelo preentrenado grande; (ii) hay una regularidad clara — **cuanto más pre-entrenamiento acumulado, menor la tasa**: AST-P usa 10× menos lr que AST-S en ESC-50, y full AudioSet usa 5× menos que balanced. Afirmar más que eso sería inventar.

---

## 9. La ventaja del campo receptivo global

Este es el argumento arquitectónico del paper, enunciado en la introducción: AST "can capture long-range global context **even in the lowest layers**".

### 9.1. La aritmética del campo receptivo en una CNN

En una pila convolucional, el campo receptivo (RF) de la capa $l$ crece según

$$R_l = R_{l-1} + (k_l - 1)\prod_{i<l} s_i$$

donde $k_l$ es el tamaño de kernel y $s_i$ los strides acumulados. Casos concretos, midiendo en **frames de espectrograma** (10 ms cada uno):

- **Convoluciones $3\times3$, stride 1, sin dilatación:** $R_l = 2l + 1$. Para cubrir los 1024 frames del clip de AudioSet hacen falta **512 capas**. Impracticable.
- **Con *pooling* de factor 2 cada pocas capas** (la receta real de VGG/ResNet/EfficientNet): el RF crece geométricamente. Una EfficientNet-B2 con 5 etapas de reducción llega a un RF nominal de varios cientos de frames en las capas finales. Pero el RF **nominal** no es el **efectivo**: Luo et al. mostraron que la influencia de un píxel de entrada sobre una unidad profunda decae aproximadamente como una gaussiana, así que el RF efectivo es del orden de $\sqrt{\text{RF nominal}}$. En la práctica, las capas medias de una CNN de audio **no ven** el clip completo.
- **Pila dilatada** (la solución de WaveNet y de las TCN): con dilatación que duplica por capa, $R_L = 2^{L+1} - 1$ para kernel 3. Para cubrir 1024 frames hacen falta $L = 9$ capas; para 2048, $L=10$. Es eficiente, pero con dos costos: (i) el RF sigue creciendo **con la profundidad**, así que las capas 1 a 4 siguen siendo miopes; (ii) la dilatación introduce el conocido problema de *gridding*, donde posiciones intermedias nunca se muestrean.

En los tres casos, **la conectividad global es una propiedad emergente de la profundidad**, y las capas bajas —donde se toman las decisiones sobre qué features construir— operan con información estrictamente local.

### 9.2. La aritmética en AST

En AST **no hay aritmética**. En la capa 1, cada token atiende a los 1212 tokens (todos los parches del espectrograma) más el `[CLS]`. La distancia de camino entre dos posiciones cualesquiera es **1 salto**, en **todas** las 12 capas. El campo receptivo es constante e igual al total, desde el primer bloque.

| | Campo receptivo en la capa 1 | Crecimiento con la profundidad | Camino máximo entre dos posiciones |
|---|---|---|---|
| CNN $3\times3$ | 3 frames | lineal ($+2$/capa) | $O(n)$ capas |
| CNN dilatada | 3 frames | exponencial ($\times 2$/capa) | $O(\log n)$ capas |
| CNN + atención al final | 3 frames | lineal/geométrico, y **global solo al final** | $O(1)$ pero solo entre features ya comprimidos |
| **AST** | **todo el espectrograma** | **constante (ya es total)** | **$O(1)$ en cualquier capa** |

### 9.3. Qué implica para sonidos con estructura armónica

Aquí está el caso donde la diferencia es cualitativa, no cuantitativa.

Un sonido tonal —una voz, un violín, un motor— tiene energía concentrada en un fundamental $f_0$ y en armónicos en $2f_0, 3f_0, \dots$. En un espectrograma, eso es un **peine de rayas horizontales** distribuido a lo largo de todo el eje de frecuencia. La identidad tímbrica está en la **relación entre las amplitudes de armónicos muy separados en el eje**: es literalmente lo que distingue un clarinete (armónicos impares dominantes) de una trompeta.

Con 128 bandas mel y parches de 16 bandas, un parche cubre **1/8 del eje de frecuencia**. Un $f_0$ grave y su 6.º armónico pueden caer en parches separados por 60 u 80 bandas mel.

- En una **CNN de $3\times3$**, relacionar la banda 12 con la banda 92 requiere un RF de 80 bandas en el eje de frecuencia, es decir ~40 capas sin *pooling*, o bien esperar hasta que el *pooling* haya reducido el eje de frecuencia lo suficiente — momento en el cual la resolución fina de los armónicos individuales **ya se perdió**. La CNN se ve forzada a un dilema: o ve la relación global sin resolución, o ve la resolución sin la relación global.
- En **AST**, la cabeza de atención de la capa 1 puede aprender un patrón "atiende a los parches cuya posición en frecuencia está en relación armónica con la mía" y **la información llega intacta**, sin haber pasado por ningún *pooling*. Las 12 cabezas permiten que varias de esas relaciones coexistan.

Vale ser justo: el paper **no visualiza mapas de atención ni verifica que AST aprenda efectivamente patrones armónicos**. El argumento es arquitectónico y las cifras lo respaldan indirectamente, pero la evidencia mecanística no está. Trabajos posteriores sí han visualizado atención en modelos tipo AST y encontrado cabezas con estructura de frecuencia, pero eso no es de este paper.

### 9.4. Qué implica para eventos largos

El segundo caso: un evento sonoro cuya identidad depende de correlaciones temporales lejanas. Ejemplos en AudioSet: una sirena (el patrón es la **periodicidad del barrido**, con período de segundos); un tren pasando (el efecto Doppler es una deriva lenta de frecuencia a lo largo de todo el clip); aplausos (la textura es estadística sobre el clip completo, no local).

En un clip de 10 s con 1024 frames, relacionar el segundo 1 con el segundo 9 significa cruzar ~800 frames = 80 parches temporales. En AST eso es una entrada de la matriz de atención. En una CNN es una cadena de decenas de capas o un *pooling* que promedia y destruye el patrón.

Aquí también hay un matiz honesto: **el *global average pooling* de una CNN sí agrega información de todo el clip** — al final. La diferencia no es que la CNN no pueda ver todo el clip, sino **cuándo**: el GAP promedia representaciones que ya fueron construidas localmente. Si la construcción local descartó la información necesaria, el promedio no la recupera. La atención permite que la construcción de features **sea condicional al contexto global desde el principio**. Ese es el argumento preciso, y es exactamente el mismo que S3D/SlowFast hacen en video sobre dónde poner el modelado temporal.

---

## 10. Limitaciones

### 10.1. El costo cuadrático: la aritmética completa

La auto-atención cuesta $O(N^2 d)$ por capa en la longitud de secuencia $N$. Para AST, $N$ crece **linealmente con la duración del audio**, así que **el costo crece cuadráticamente con la duración**.

Con los parámetros del paper ($F = 128$ bandas, $p = 16$, $s = 10$, 100 frames/s, $n_f = 12$):

$$N(t) = 12 \times \left(\left\lfloor \frac{100t - 16}{10} \right\rfloor + 1\right) \approx 120\,t$$

| Duración | Frames | Parches temporales | **$N$ (tokens)** | Entradas de la matriz de atención | Memoria de la matriz (fp32, 12 cabezas, 1 capa) |
|---|---|---|---|---|---|
| 1 s (Speech Commands) | 100 | 9 | 108 | $1.2\times10^4$ | 0.6 MB |
| 5 s (ESC-50) | 500 | 49 | 588 | $3.5\times10^5$ | 17 MB |
| **10 s (AudioSet, con pad a 1024)** | **1024** | **101** | **1212** | **$1.47\times10^6$** | **71 MB** |
| 1 min | 6000 | 599 | 7188 | $5.2\times10^7$ | 2.5 GB |
| **10 min** | **60000** | **5999** | **71988** | **$5.18\times10^9$** | **249 GB** |

**Un clip de 10 minutos genera casi 72.000 parches.** La matriz de atención tiene $5.2\times10^9$ entradas: **20.7 GB por cabeza en fp32**, y con 12 cabezas, **249 GB por capa**. Multiplicado por 12 capas y por lo que exige el backward, está fuera de alcance por varios órdenes de magnitud incluso hoy, con FlashAttention y fp16.

El factor de escala entre 10 s y 10 min es $(71988/1212)^2 \approx \mathbf{3528\times}$, para 60× más audio.

Esta es **la** limitación de AST, y es estructural, no de implementación. El paper la reconoce solo de pasada, en el contexto del solape ("will quadratically increase the computational overhead"), pero no discute el techo de duración que impone. Toda la línea posterior (PaSST con *patchout*, HTS-AT con atención jerárquica tipo Swin, ventanas deslizantes) es esencialmente trabajo para levantar este techo.

**Nótese además** que el problema no es solo el clip largo: es que **el solape lo empeora 5.6×** justo en el eje donde ya duele. Overlap-6 es la decisión que menos escala de todo el diseño.

### 10.2. Memoria durante el entrenamiento

Hay dos evidencias directas en el paper de que la memoria fue una restricción real:

1. **Batch size de 12** en AudioSet (Sección 3.1.1). Doce muestras. Los modelos CNN de audio de la época entrenaban con batches de 64–128 sin dificultad. Un batch de 12 sobre un dataset de 2M implica ~167.000 pasos por época.
2. **ViT-Large tuvo que entrenarse sin solape "due to memory limitation"** (nota de la Tabla 3). El modelo grande literalmente no cupo con la configuración óptima.

El cálculo lo explica: con $N=1212$, guardar las matrices de atención para el backward cuesta ~71 MB por capa por muestra, ~850 MB por muestra en 12 capas, y ~10 GB para un batch de 12 — solo en matrices de atención, sin contar activaciones ni gradientes ni estados del optimizador Adam (que son 2× los parámetros, ~700 MB para 87M en fp32).

### 10.3. Dependencia del pre-entrenamiento en ImageNet

Ya cuantificada en la Sección 8.1, pero conviene enunciarla como limitación explícita: **sin ImageNet, AST no supera a las CNN ni siquiera con 2 millones de clips de entrenamiento** (0.366 vs 0.439 de PANNs y 0.444 de PSLA).

Consecuencias prácticas de esa dependencia:

- **La tokenización queda congelada.** Como se vio en la Tabla 6, $128\times2$ es mejor tokenización que $16\times16$ para audio entrenado desde cero, pero es inutilizable porque no existe checkpoint de ImageNet compatible. **El diseño arquitectónico está subordinado a la disponibilidad de pesos de visión.** Es una restricción rara y poco satisfactoria intelectualmente.
- **El modelo hereda las propiedades del dataset fuente.** Cualquier sesgo, artefacto o particularidad de la distribución de ImageNet-1k entra al modelo de audio por esta puerta. Nadie ha auditado qué significa eso.
- **La receta no es escalable a otros dominios sin un ImageNet equivalente.** La estrategia "toma el mejor checkpoint de visión y adáptalo" solo funciona porque visión resolvió su problema de datos primero.

Esta limitación es la que **SSAST** (el trabajo siguiente del mismo primer autor, AAAI 2022) ataca directamente con pre-entrenamiento autosupervisado sobre audio sin etiquetar, eliminando la dependencia de ImageNet.

### 10.4. Otras

- **No hay análisis de qué aprende el modelo.** Ni un mapa de atención, ni una visualización de embeddings posicionales aprendidos, ni un análisis por clase. Todo el argumento del campo receptivo global es teórico y las cifras lo apoyan solo indirectamente. Es la deuda más grande del paper.
- **El solape es simétrico sin justificación.** Ver Sección 4.3.
- **Solo clasificación.** No hay detección de eventos sonoros (localización temporal), ni separación, ni ASR, ni tareas de secuencia a secuencia. La afirmación de "clasificador de audio genérico" está bien acotada a **clasificación**.
- **Latencia e inferencia no se reportan.** El paper cuantifica la velocidad de **convergencia** (5 épocas vs 30) pero no da FLOPS, latencia ni memoria de inferencia en ninguna parte, lo cual es notable dado que compara contra EfficientNet, una arquitectura diseñada precisamente para eficiencia.

---

## 11. Impacto y legado

### 11.1. La línea directa: SSAST, PaSST, Audio-MAE, BEATs

**SSAST** (Gong et al., AAAI 2022) es la continuación natural y del mismo grupo: aplica **pre-entrenamiento autosupervisado** sobre parches de espectrograma (enmascarar parches y predecirlos, con un objetivo combinado discriminativo y generativo) usando AudioSet y LibriSpeech sin etiquetar. El objetivo declarado es exactamente eliminar la dependencia de ImageNet que la Sección 8.1 dejó al descubierto. Es el eslabón que convierte "necesitamos visión para entrenar audio" en "el audio puede entrenarse a sí mismo".

**PaSST** (Koutini et al., Interspeech 2022) ataca la limitación de la Sección 10.1 con *patchout*: descartar aleatoriamente una fracción de los parches durante el entrenamiento. Reduce el cómputo y actúa como regularizador, permitiendo entrenar con menos memoria y en menos tiempo.

**HTS-AT** introduce jerarquía tipo Swin (atención por ventanas con desplazamiento y reducción progresiva de tokens), recuperando parte de la eficiencia de la pirámide convolucional sin volver a la convolución.

**Audio-MAE** (Huang et al., NeurIPS 2022) traslada el *masked autoencoder* de He et al. al espectrograma, con ratios de enmascaramiento altos y un decoder ligero.

**BEATs** (Chen et al., ICML 2023) cierra el círculo: pre-entrenamiento iterativo donde un tokenizador acústico aprendido produce etiquetas discretas para el modelo enmascarado, y el modelo mejorado re-entrena el tokenizador. Es el que fija el estado del arte moderno en AudioSet.

(Las cifras exactas de SSAST, PaSST, HTS-AT, Audio-MAE y BEATs **no están en este PDF**; las menciono por su rol en la genealogía, no como números verificados.)

### 11.2. La ruta hacia los encoders multimodales actuales

El linaje que va de AST a los modelos de hoy tiene una forma reconocible. Los encoders de audio de los sistemas multimodales contemporáneos son, casi sin excepción, **Transformers sobre representaciones tiempo-frecuencia**:

- **CLAP** y sus variantes (el análogo de CLIP para audio) usan encoders de audio de la familia AST/HTS-AT alineados contra un encoder de texto por contraste. Esa alineación es la que habilita clasificación *zero-shot* de audio y búsqueda texto→audio.
- **Whisper** (OpenAI, 2022) usa un encoder Transformer sobre espectrograma log-Mel de 80 bandas. Con una diferencia importante que conviene registrar: Whisper **sí conserva un *stem* convolucional** (dos capas Conv1D con stride) antes del Transformer, para reducir la longitud de secuencia. No es convolution-free. La convolución sobrevivió como **compresor de secuencia**, que es donde realmente aporta.
- Los LLM con entrada de audio (**Qwen-Audio**, **SALMONN**, **LTU/LTU-AS** — este último también de Yuan Gong) conectan un encoder de audio tipo AST o Whisper a un LLM mediante un proyector. AST y sus descendientes son el ojo del sistema.

El patrón general: **AST estableció que el camino canónico para meter audio en un Transformer es "espectrograma → parches → tokens"**, y esa decisión sobrevivió, aunque los detalles (pre-entrenamiento, jerarquía, stem convolucional para comprimir) cambiaron.

### 11.3. Por qué AST sigue siendo el baseline de referencia

Cuatro razones concretas:

1. **Está en `transformers` de Hugging Face** con checkpoints listos (`ASTForAudioClassification`, `ASTFeatureExtractor`). Bajar un modelo con 0.459 mAP en AudioSet toma tres líneas. Esa accesibilidad es lo que convierte un paper en un baseline.
2. **Es simple de describir y de reimplementar.** El paper insiste deliberadamente en usar el encoder Transformer estándar "without modification". Un baseline con partes móviles propias es un mal baseline; AST no tiene ninguna.
3. **Ocupa una posición conceptual limpia en el espacio de diseño:** atención pura, sin jerarquía, sin trucos de eficiencia, con transferencia cross-modal. Cualquier propuesta nueva puede posicionarse respecto de él quitando o agregando exactamente una cosa.
4. **Las cifras siguen siendo respetables.** 0.459 mAP single en AudioSet no fue superado por márgenes enormes hasta la generación autosupervisada, y sigue siendo un punto de comparación válido.

Y hay un mérito histórico más difícil de medir: AST fue una de las demostraciones que **cerraron el debate sobre si el sesgo inductivo convolucional era necesario**. En 2020 la respuesta obvia era sí; en 2021, entre ViT, DeiT y AST, quedó claro que era **una forma de ahorrar datos**, no un requisito. Con suficientes datos —propios o prestados de otra modalidad— la atención pura basta.

---

## 12. Auditoría de las tres objeciones de la clase 39

La clase 39 del diplomado ("DL Models for Audio Processing", prof. Gabriel Sepúlveda, abril de 2024) incluye una sección "Audio and Transformers" que enumera tres problemas y concluye que "los Transformers no son actualmente muy populares para aplicaciones de audio". AST, publicado casi tres años antes de esa clase, es el contraejemplo documentado. Vale la pena evaluar cada afirmación contra evidencia, sin sobrecorregir.

### 12.1. Objeción 1: "In the context of audio, there is still a lack of highly massive audio datasets"

**Lo que la objeción acierta.** La premisa fáctica era correcta en su momento y el propio AST la enuncia como su principal obstáculo (Sección 2.2): ViT solo superaba a las CNN por encima de **14 millones de imágenes**, y "audio datasets typically do not have such large amounts of data". AudioSet, el dataset etiquetado más grande de eventos sonoros, tiene 2M de clips de 10 s (~5.500 horas). La ablación de la Tabla 2 confirma que la preocupación era legítima: **un AST entrenado desde cero sobre los 2M de AudioSet full alcanza 0.366 mAP, por debajo de PANNs (0.439) y de PSLA (0.444)**. Con los datos de audio disponibles y sin ayuda externa, el Transformer efectivamente pierde. La objeción, en su forma de 2021, tenía base empírica.

**Por qué deja de ser bloqueante.** Hay dos respuestas independientes, y las dos ya estaban publicadas antes de abril de 2024.

**(a) Transferencia cross-modal.** AST elude el problema en lugar de resolverlo: si no hay 14 millones de espectrogramas, se usan los 14 millones de imágenes que sí existen. Las cifras de la Tabla 2 miden exactamente cuánto vale ese préstamo: **0.148 → 0.347 en balanced (×2.34)** y **0.366 → 0.459 en full (+25%)**. La clave conceptual, desarrollada en la Sección 5.3, es que lo que se transfiere no es semántica visual sino (i) estadística local de campos 2D y (ii) una inicialización bien acondicionada de un Transformer profundo. El "dataset masivo" no tiene por qué ser del dominio de destino.

**(b) Pre-entrenamiento autosupervisado.** La respuesta más limpia y la que elimina la dependencia. **SSAST** (Gong et al., AAAI 2022, del mismo primer autor) pre-entrena sobre audio **sin etiquetar** mediante enmascaramiento de parches de espectrograma, y la línea continúa con **Audio-MAE** (2022) y **BEATs** (2023). Aquí la restricción se disuelve por completo: la escasez nunca fue de audio, fue de audio **etiquetado**. Audio sin etiquetar hay ilimitado.

Y a nivel de escala bruta, para 2024 la premisa ya era discutible: **Whisper** (OpenAI, 2022) se entrenó sobre **680.000 horas** de audio con supervisión débil —dos órdenes de magnitud más que AudioSet— y sus versiones posteriores sobre varios millones de horas; **LibriLight** aporta 60.000 horas de habla sin etiquetar para wav2vec 2.0 y HuBERT. (Estas cifras no están en el PDF de AST; son de dominio público y las cito como contexto.)

**Veredicto.** Objeción **históricamente válida, pero superada por dos vías distintas antes de 2022**, ambas publicadas y ampliamente adoptadas. Presentarla en 2024 como razón vigente para no usar Transformers en audio omite la solución que el propio paper que la enuncia ya había implementado.

### 12.2. Objeción 2: "Self-attention operates over a finite sequence of discrete entities. In text, sentence segmentation is trivial, but for audio this is not the case"

**Esta es la objeción que AST desactiva de forma más directa y completa.**

La objeción contiene un supuesto oculto que es el que falla: **que los tokens deben ser unidades semánticamente significativas**. En texto, las palabras lo son, y por eso la analogía sugiere que en audio habría que segmentar en fonemas, notas o eventos — un problema difícil y circular, porque segmentar bien el audio requiere ya entenderlo.

**ViT había refutado ese supuesto en imágenes un año antes, y AST lo traslada al audio.** Una imagen tampoco tiene "palabras": no hay una segmentación trivial en objetos, y segmentar bien es un problema tan difícil como clasificar. ViT resolvió eso ignorando la pregunta: **partir en una grilla regular de $16\times16$ píxeles**. Los parches no respetan bordes de objetos, cortan caras por la mitad, mezclan fondo y figura. Y funciona.

AST hace exactamente lo mismo sobre el espectrograma. Del abstract: "The 2D audio spectrogram is split into a sequence of $16\times16$ patches with overlap, and then linearly projected to a sequence of 1-D patch embeddings."

**El principio que se desprende, y que es la respuesta precisa a la objeción: la tokenización no necesita ser semántica; basta con que sea regular, exhaustiva y complementada con información posicional.** Los tres requisitos reales son:

1. **Cobertura.** Los tokens deben cubrir toda la entrada sin perder información. Parches regulares lo garantizan por construcción.
2. **Regularidad.** El significado de un token debe depender solo de su contenido, no de un proceso de segmentación variable. Una segmentación aprendida introduce un componente frágil río arriba; una grilla fija es determinista.
3. **Posición.** Como el token en sí no dice dónde está, hace falta un embedding posicional. AST lo tiene, entrenable, y la Tabla 4 demuestra que es esencial (reinicializarlo cuesta 4.2 mAP).

Con eso, **la auto-atención construye por sí misma las agrupaciones semánticas** que la segmentación explícita habría tenido que producir. Ese es literalmente su trabajo: aprender qué posiciones se relacionan con cuáles. Exigir una segmentación semántica *antes* de la atención es pedirle al preprocesamiento que resuelva lo que el modelo está diseñado para resolver.

**Y el paper mide esta decisión, no la asume.** La Tabla 6 compara la grilla $16\times16$ contra la alternativa "temporalmente ordenada" de $128\times2$ —que es lo más parecido a una segmentación "natural" del audio como secuencia 1D— y encuentra que la diferencia es de 0.011 mAP a favor de la rectangular sin pre-entrenar, mientras que el pre-entrenamiento vale 0.193. **La elección de tokenización es un efecto de segundo orden.** Ese es el dato que cierra la objeción: no es que AST haya encontrado la segmentación correcta del audio; es que **la segmentación resultó no ser el problema**.

**El resultado transversal que lo confirma.** La misma tokenización y la misma arquitectura, sin ningún cambio, alcanzan el estado del arte en **habla** (Speech Commands, 1 s), **sonidos ambientales** (ESC-50, 5 s) y **eventos sonoros generales** (AudioSet, 10 s). Si la segmentación fuera el problema, cada uno de esos dominios habría exigido una noción distinta de "entidad discreta". No la exigió.

**Veredicto.** Objeción **resuelta**, con evidencia directa y con la ablación que muestra que ni siquiera era el eje que más importaba. AST es el contraejemplo exacto.

### 12.3. Objeción 3: "Transformers are not good to model long dependencies in sequences"

**Es la afirmación más problemática de las tres, porque invierte la motivación original de la auto-atención.**

Vaswani et al. (NIPS 2017), en la Sección 4 de "Attention Is All You Need", justifican la auto-atención con una tabla explícita de tres criterios: complejidad por capa, operaciones secuenciales y **longitud máxima del camino entre dos posiciones cualesquiera**:

| Tipo de capa | Complejidad por capa | Operaciones secuenciales | **Longitud máxima del camino** |
|---|---|---|---|
| Auto-atención | $O(n^2 \cdot d)$ | $O(1)$ | $\mathbf{O(1)}$ |
| Recurrente | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolucional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(n/k)$, o $O(\log_k n)$ con convoluciones dilatadas |

Y el argumento textual: "Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is **the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies**."

La auto-atención fue diseñada **precisamente para modelar dependencias largas mejor que las alternativas**. Con camino $O(1)$, el gradiente entre dos posiciones separadas por 1000 pasos atraviesa **una** operación; en una RNN atraviesa 1000 multiplicaciones matriciales sucesivas, que es exactamente el mecanismo del gradiente que se desvanece y la razón histórica por la que las RNN fallaban en dependencias largas. Afirmar que los Transformers son malos en dependencias largas es afirmar lo contrario de su razón de existir, y de la evidencia que produjo toda la generación de LLM.

**Cuál es el problema real, y por qué la confusión es entendible.** El costo de la auto-atención es $O(n^2 d)$ por capa, contra $O(n d^2)$ de una RNN. Es **cuadrático en la longitud**. Por eso las secuencias largas son caras. Pero eso es **una limitación de costo, no de capacidad**: si la secuencia cabe en memoria, el Transformer modela la dependencia mejor que cualquier alternativa. El error de la objeción es confundir "no puedo permitirme una secuencia larga" con "no modelo bien lo que hay dentro de ella".

En AST esto es cuantificable exactamente (Sección 10.1): un clip de 10 s genera 1212 tokens y entra sin problemas; uno de 10 minutos genera 71.988 y no entra en ningún hardware razonable. **El techo es de cómputo. Dentro de los 1212 tokens, la conectividad es total y el camino es $O(1)$.**

**Dónde la objeción tiene algo de razón, si se la reformula con caridad.** No conviene sobrecorregir. Hay tres fenómenos reales que podrían haberla motivado, y ninguno es "la auto-atención no modela dependencias largas":

1. **Dilución de la atención.** El softmax sobre $n$ claves reparte una masa de probabilidad total de 1. Cuando $n$ crece mucho, es más difícil que una posición se concentre nítidamente en otra, y en la práctica los modelos de contexto muy largo muestran degradación de la recuperación en el medio de la secuencia (el fenómeno de *lost in the middle*). Es un problema de **optimización y de comportamiento del softmax**, no de conectividad estructural.
2. **Extrapolación posicional.** Los embeddings posicionales aprendidos —los que usa AST— no extrapolan a longitudes no vistas en entrenamiento. Es exactamente el problema que motiva el *cut-and-interpolate*, y también toda la línea de RoPE, ALiBi y afines. Es un problema del **esquema de posición**, no de la atención.
3. **Escasez de supervisión de largo alcance.** Aunque la arquitectura permita la dependencia, el modelo solo la aprende si los datos la exhiben y la pérdida la premia. Para 10 s de AudioSet con una sola etiqueta débil por clip, la señal que empuja a aprender dependencias de 8 segundos es tenue. Es un problema de **datos y objetivo**.

Los tres son reales y merecen mención. **Ninguno sostiene la afirmación tal como está escrita en el slide.** La formulación correcta sería: *"la auto-atención tiene el camino más corto posible entre posiciones, lo que la hace la arquitectura mejor equipada para dependencias largas; su limitación es el costo cuadrático en la longitud, más problemas prácticos de extrapolación posicional y de dilución del softmax en contextos muy extensos."*

**Veredicto.** Afirmación **incorrecta como está formulada**, e invertida respecto del argumento fundacional de Vaswani et al. Hay una afirmación cercana y verdadera —sobre el costo cuadrático— que probablemente sea lo que se quiso decir.

### 12.4. Síntesis

| Objeción de la clase 39 | Estado | Evidencia |
|---|---|---|
| 1. Falta de datasets masivos de audio | **Válida en su origen, superada por dos vías** | Tabla 2 (transferencia cross-modal: ×2.34 en balanced, +25% en full); SSAST/Audio-MAE/BEATs (autosupervisión); Whisper (680k h) |
| 2. La auto-atención necesita entidades discretas y el audio no se segmenta trivialmente | **Disuelta** | Parcheo regular $16\times16$ con solape; Tabla 6 muestra que la elección de tokenización es de segundo orden frente al pre-entrenamiento; misma arquitectura para 1 s, 5 s y 10 s, habla y no-habla |
| 3. Los Transformers no modelan bien dependencias largas | **Incorrecta como está formulada** | Vaswani et al. 2017 §4: camino $O(1)$ vs $O(n)$ de una RNN. El problema real es el costo $O(n^2)$, más extrapolación posicional y dilución del softmax |

Y la conclusión del slide —"los Transformers no son actualmente muy populares para aplicaciones de audio"— era, para abril de 2024, difícil de sostener. Whisper (2022) es un Transformer y era el modelo de ASR más usado del mundo; wav2vec 2.0 y HuBERT son Transformers; los encoders de audio de todos los LLM multimodales de 2023–2024 son Transformers; AST y sus descendientes eran el baseline estándar de clasificación de audio desde 2021. Lo que sí es cierto y probablemente esté detrás de la afirmación es que **la convolución no desapareció**: sobrevive como *stem* compresor de secuencia (Whisper, wav2vec 2.0) precisamente para mitigar el costo cuadrático — que es la objeción 3 en su forma correcta.

---

## 13. Erratas, matices y cosas que se citan mal

**1. El número de parches: tres valores distintos en el mismo paper.** Ya desarrollado en la Sección 4.2. La fórmula $N = 12\lceil(100t-16)/10\rceil$ da **1188** para $t=10$; el texto de la Sección 2.2 dice **$12\times100 = 1200$**; la Tabla 5 reporta **1212**. El valor operativo es 1212, y se obtiene solo si el espectrograma se rellena a **1024 frames**, no a 1000. Las cuatro filas de la Tabla 5 y las tres de la Tabla 6 lo confirman sin ambigüedad. La fórmula del paper describe el caso ideal sin padding.

**2. "AST tiene menos parámetros que los híbridos CNN+atención": no está cuantificado y es dudoso en términos absolutos.** La introducción afirma: "comparing with SOTA CNN-attention hybrid models, AST features a simpler architecture with **fewer parameters**, and converges faster during training." La convergencia sí está medida (5 vs 30 épocas). El conteo de parámetros **no aparece en ninguna parte del paper para AST**. Lo único que se sabe es que el checkpoint DeiT fuente tiene **87M** (Sección 2.2, Tabla 3), y AST hereda esencialmente esa cuenta. PSLA se basa en EfficientNet-B2, que es de un orden de magnitud menor. **No se puede verificar la afirmación con la información del paper, y la evidencia disponible sugiere que es falsa en términos absolutos.** La lectura caritativa es que se refiere a los *ensambles* (6 modelos de AST contra 10 de PSLA, dato que sí está en la Sección 3.1.2) o a "simpler architecture" en sentido estructural. Como está escrita, es la afirmación menos sostenida del paper.

**3. "AST usa destilación de conocimiento": no.** AST **no destila nada**. Toma pesos de DeiT, que sí fue destilado desde un profesor CNN durante su entrenamiento en ImageNet. No hay profesor, ni pérdida de destilación, ni token de destilación en el modelo de audio: los dos tokens `[CLS]` de DeiT se **promedian en uno solo** (Sección 2.2). El malentendido es frecuente porque "DeiT w/ Distill" aparece en la Tabla 3.

**4. "Convolution-free" tiene una excepción que el paper declara.** La capa de parcheo *es* una convolución de kernel 16 y stride 10, y el paper lo dice explícitamente ("Strictly speaking, the patch embedding layer can be viewed as a single convolution layer with a large kernel and stride size"). La afirmación defendible es que **ningún sesgo inductivo convolucional participa del modelado**; no que no exista ninguna operación expresable como convolución. Ver Sección 3.1.

**5. Cita cruzada inconsistente en Speech Commands.** El texto de la Sección 3.2 dice: "AST-S model achieves 98.11±0.05, outperforms the SOTA model in **[9]**". La referencia [9] es Rybakov et al., "Streaming keyword spotting on mobile devices". Pero la Tabla 7 identifica el SOTA-S como **[34]** (MatchboxNet, 97.4%). Es una inconsistencia entre texto y tabla. La cifra de comparación correcta es la de la tabla.

**6. La correlación ImageNet ↔ AudioSet no es monótona.** El paper afirma que el modelo que mejor rinde en ImageNet también rinde mejor en AudioSet. Es cierto para el ganador (DeiT distilled: 0.852 → 0.347), pero **falla en el medio de la tabla**: DeiT sin destilación tiene *menor* accuracy en ImageNet que ViT-Base (0.829 vs 0.846) y sin embargo *mayor* mAP en AudioSet (0.330 vs 0.320). Lo que transfiere no es solo la accuracy sino algo de la receta de entrenamiento. Y ViT-Large está handicapeado por haberse entrenado sin solape (nota de la Tabla 3), así que **su 0.330 no es comparable con las demás filas**.

**7. Los 0.485 mAP del abstract son de un ensamble de 6 modelos, no de un modelo.** El modelo único con *weight averaging* da **0.459**; el modelo único en la última época, **0.448**. Comparar 0.485 contra el modelo único de otro trabajo sería incorrecto. El paper es transparente al respecto en la Tabla 1, pero el abstract cita solo la cifra alta.

**8. No hay ablación de tasa de aprendizaje.** El paper reporta las tasas usadas pero no ablaciona ninguna. Cualquier afirmación del tipo "AST es sensible a la tasa de aprendizaje según el paper" es una extrapolación.

**9. Los 128 filtros mel y los 16 kHz.** La frecuencia de muestreo de 16 kHz **no está escrita explícitamente en el paper**; se deduce de la parametrización estándar (25 ms / 10 ms es la configuración canónica de Kaldi, y AudioSet se procesa a 16 kHz en la práctica común). El paper solo especifica "128-dimensional log Mel filterbank features computed with a 25ms Hamming window every 10ms".

**10. La normalización a $\sigma = 0.5$ no está justificada en el paper.** Se enuncia el valor y nada más. La explicación de la Sección 5.1 (alinear el rango dinámico con el de las imágenes normalizadas de ViT) es interpretación razonable, no cita.

**11. Que ESC-50 llegue a 95.6% con AST-P se compara contra PANNs (94.7%), que es CNN.** La ganancia real es de **+0.9 puntos**, marginal. La cifra que vale conceptualmente es la de AST-S (88.7 vs 86.5), porque es la que muestra eficiencia de datos sin pre-entrenamiento en audio.

---

## 14. Cómo se ve hoy

### 14.1. El parcheo del espectrograma y el cut-and-interpolate en PyTorch

Dos piezas: la tokenización (que es literalmente una `Conv2d`) y la adaptación del embedding posicional. El código es una reconstrucción fiel de lo descrito en las Secciones 2.1 y 2.2 del paper.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ---------------------------------------------------------------------------
# 1) Forma de onda -> espectrograma log-Mel -> parches solapados -> tokens
# ---------------------------------------------------------------------------

def waveform_to_fbank(wav, sr=16_000, n_mels=128, target_frames=1024):
    """25 ms de ventana Hamming cada 10 ms -> 100 frames/s, 128 bandas mel.
    Se rellena/recorta a 1024 frames: es el valor que reproduce exactamente
    los conteos de parches de las Tablas 5 y 6 del paper (512/657/850/1212)."""
    fbank = torchaudio.compliance.kaldi.fbank(
        wav, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type="hanning", num_mel_bins=n_mels,
        frame_shift=10.0, frame_length=25.0, dither=0.0,
    )                                            # [T, 128]
    pad = target_frames - fbank.shape[0]
    fbank = F.pad(fbank, (0, 0, 0, pad)) if pad > 0 else fbank[:target_frames]
    # Normalizacion del paper (Sec. 2.2): media 0, desviacion 0.5.
    # El 0.5 alinea el rango dinamico con el de las imagenes normalizadas de ViT.
    return (fbank - fbank.mean()) / (fbank.std() * 2.0)   # [1024, 128]


class SpectrogramPatchEmbed(nn.Module):
    """Parches de 16x16 con solape de 6 (stride 10) en tiempo y frecuencia.
    Extraer parches solapados + proyectar linealmente ES una Conv2d con
    kernel=16 y stride=10. El propio paper lo reconoce ('strictly speaking...')."""
    def __init__(self, embed_dim=768, patch=16, overlap=6, n_mels=128, n_frames=1024):
        super().__init__()
        stride = patch - overlap                                     # 10
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch, stride=stride)
        self.f_dim = (n_mels   - patch) // stride + 1                # 12
        self.t_dim = (n_frames - patch) // stride + 1                # 101
        self.num_patches = self.f_dim * self.t_dim                   # 1212

    def forward(self, fbank):                    # fbank: [B, 1024, 128]
        x = fbank.transpose(1, 2).unsqueeze(1)   # [B, 1, 128(freq), 1024(time)]
        x = self.proj(x)                         # [B, 768, 12, 101]
        return x.flatten(2).transpose(1, 2)      # [B, 1212, 768]


# ---------------------------------------------------------------------------
# 2) Cut-and-interpolate del embedding posicional (Sec. 2.2, Tabla 4)
# ---------------------------------------------------------------------------

def adapt_pos_embed(vit_pos, f_dim, t_dim, grid=24, n_special=2, embed_dim=768):
    """ViT/DeiT 384x384, parches 16x16 sin solape -> grilla 24x24 = 576 posiciones.
    Se CORTA el eje rigido (frecuencia: 24 -> 12) y se INTERPOLA el elastico
    (tiempo: 24 -> t_dim, que cambia con la duracion del clip).
    Cortar en frecuencia preserva embeddings intactos y mutuamente distinguibles;
    interpolar en tiempo estira la estructura aprendida a cualquier longitud.
    vit_pos: [1, n_special + 576, 768]"""
    cls_pos, grid_pos = vit_pos[:, :n_special], vit_pos[:, n_special:]

    # DeiT trae dos tokens especiales ([CLS] y [DIST]); AST los promedia en uno.
    cls_pos = cls_pos.mean(dim=1, keepdim=True)                      # [1, 1, 768]

    g = grid_pos.reshape(1, grid, grid, embed_dim).permute(0, 3, 1, 2)  # [1,768,24,24]

    # --- CUT en frecuencia (recorte centrado, como en la implementacion oficial)
    if f_dim <= grid:
        off = grid // 2 - f_dim // 2
        g = g[:, :, off:off + f_dim, :]
    else:
        g = F.interpolate(g, size=(f_dim, g.shape[-1]), mode="bilinear",
                          align_corners=False)

    # --- INTERPOLATE en tiempo (bilineal; la Tabla 4 muestra que nearest da igual)
    g = F.interpolate(g, size=(f_dim, t_dim), mode="bilinear", align_corners=False)

    g = g.permute(0, 2, 3, 1).reshape(1, f_dim * t_dim, embed_dim)
    return torch.cat([cls_pos, g], dim=1)        # [1, 1 + f_dim*t_dim, 768]
```

Tres cosas que el código hace explícitas y la prosa del paper no:

- **La asimetría del tratamiento de ejes es el núcleo del método.** Un `F.interpolate` en ambos ejes sería más corto de escribir y peor: destruiría la distinguibilidad de las bandas de frecuencia. Corte para el eje rígido, interpolación para el elástico.
- **El promedio de los tokens especiales es una línea** (`cls_pos.mean(dim=1)`), y aplica tanto a los embeddings de token como a sus embeddings posicionales.
- **`num_patches` depende de `n_frames`**, y `adapt_pos_embed` se llama con el `t_dim` que corresponda. Eso es exactamente lo que permite que la misma arquitectura sirva para 1 s, 5 s y 10 s.

Verificación rápida de que los conteos calzan con el paper:

```python
for ov in (0, 2, 4, 6):
    pe = SpectrogramPatchEmbed(overlap=ov)
    print(f"overlap={ov}  stride={16-ov}  f={pe.f_dim}  t={pe.t_dim}  N={pe.num_patches}")
# overlap=0  stride=16  f=8   t=64   N=512     <- Tabla 5: 512
# overlap=2  stride=14  f=9   t=73   N=657     <- Tabla 5: 657
# overlap=4  stride=12  f=10  t=85   N=850     <- Tabla 5: 850
# overlap=6  stride=10  f=12  t=101  N=1212    <- Tabla 5: 1212
```

### 14.2. En la práctica: `transformers` de Hugging Face

Hoy nadie implementa AST desde cero. Está en `transformers` con la API estándar:

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torch, torchaudio

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"   # el 0.4593 = la mAP de la Tabla 1

extractor = ASTFeatureExtractor.from_pretrained(MODEL)
model = ASTForAudioClassification.from_pretrained(MODEL).eval()

wav, sr = torchaudio.load("clip.wav")
if sr != 16_000:
    wav = torchaudio.functional.resample(wav, sr, 16_000)

# El feature extractor hace todo lo de la Seccion 4.1: fbank de 128 bandas,
# 25 ms / 10 ms, pad a max_length frames, normalizacion con mean/std del dataset.
inputs = extractor(wav.squeeze().numpy(), sampling_rate=16_000, return_tensors="pt")
# inputs["input_values"]: [1, 1024, 128]

with torch.no_grad():
    logits = model(**inputs).logits            # [1, 527] para AudioSet

top = logits.sigmoid()[0].topk(5)              # sigmoide: AudioSet es multietiqueta
for score, idx in zip(top.values, top.indices):
    print(f"{model.config.id2label[idx.item()]:35s} {score:.3f}")
```

Notas de uso que ahorran tiempo:

- **El nombre del checkpoint codifica la configuración**: `ast-finetuned-audioset-10-10-0.4593` significa *frequency stride 10, time stride 10, mAP 0.4593* — es decir, la configuración Overlap-6 de la Tabla 5 con el resultado de la Tabla 1 (0.459 redondeado). Existen variantes con otros strides (`12-12`, `14-14`, `16-16`) que corresponden a las demás filas de la Tabla 5 y son **considerablemente más baratas**; si el cómputo importa, `16-16` (sin solape, 512 tokens) da 0.451 y cuesta 5.6× menos en atención.
- **Para clasificación multietiqueta usa `sigmoid`, no `softmax`.** El modelo de AudioSet se entrenó con BCE (Sección 3.1.1) y las 527 salidas son independientes. Aplicar softmax es un error frecuente y produce puntajes sin sentido.
- **Para *fine-tuning* en otra tarea**, se pasa `num_labels`, `ignore_mismatched_sizes=True` y, si la duración cambia, `max_length` en el `ASTConfig`. La reinterpolación del embedding posicional la maneja la librería, pero conviene verificar que `config.max_length` coincida con el número de frames que produce el extractor: si no coinciden, el modelo falla o —peor— silenciosamente rellena con ceros.
- **Tasa de aprendizaje pequeña.** Coherente con los hiperparámetros del paper (Sección 8.6), el rango razonable para *fine-tuning* es $10^{-5}$ a $10^{-4}$, no $10^{-3}$.
- **Memoria.** Con 1212 tokens y 12 capas, el batch máximo en una GPU de 24 GB ronda los 16–32 ejemplos en fp16 con gradient checkpointing. El paper usó batch 12 (Sección 3.1.1), lo cual sigue siendo una guía razonable.
