---
title: "AST: Audio Spectrogram Transformer (2021)"
weight: 434
math: true
---

{{< paper-card
    title="AST: Audio Spectrogram Transformer"
    authors="Yuan Gong, Yu-An Chung, James Glass (MIT CSAIL)"
    year="2021"
    venue="Interspeech 2021 / arXiv:2104.01778"
    pdf="/papers/ast-gong-2021.pdf" >}}
La tesis del paper cabe en una línea: **la CNN no es indispensable para clasificar audio**. AST es "the first convolution-free, purely attention-based model for audio classification": toma el espectrograma log-Mel, lo parte en parches de $16\times16$ **con solape**, los proyecta linealmente, les suma un embedding posicional entrenable, antepone un token `[CLS]` y los pasa por un encoder Transformer estándar de 12 capas — sin ninguna jerarquía convolucional, sin ningún sesgo inductivo de localidad más allá del que introduce el propio parcheo. El problema obvio de esa receta es que los Transformers necesitan datos y en audio no los hay: [ViT](/fundamentos/vision-transformer) solo superaba a las CNN por encima de 14 millones de imágenes. La solución es el aporte que más ha sobrevivido: **transferencia cross-modal desde ImageNet**, adaptando los pesos de un DeiT entrenado sobre fotos a un modelo que consume espectrogramas, mediante un mecanismo de *cut-and-interpolate* del embedding posicional que permite pasar de la grilla cuadrada de $24\times24$ del ViT a una grilla rectangular y **de longitud variable**. Los resultados: **0.485 mAP en [AudioSet](/papers/audioset-gemmeke-2017)** (frente a 0.474 del mejor ensamble previo), **95.6% en [ESC-50](/papers/esc50-piczak-2015)** y **98.11% en [Speech Commands V2](/papers/speech-commands-warden-2018)** — los tres con **una única arquitectura sin ningún cambio**, sobre entradas que van de 1 a 10 segundos y contenido que va de habla a sonidos ambientales. Un detalle biográfico que vale registrar: los mismos tres autores habían publicado PSLA, el modelo CNN+atención que AST viene a desplazar. **El paper se autodestrona.**
{{< /paper-card >}}

---

## Contexto: por qué en 2021 todavía dominaban las CNN en audio

La receta canónica de clasificación de audio hacia 2020 tenía tres pasos fijos: convertir la forma de onda en un espectrograma log-Mel, tratar ese espectrograma **como si fuera una imagen** y pasarlo por una CNN de visión, y agregar sobre el tiempo antes de clasificar.

El segundo paso es la clave histórica. Desde que Hershey et al. mostraron con [VGGish](/papers/vggish-hershey-2017) que arquitecturas de ImageNet transferidas a espectrogramas funcionaban bien, el campo asumió que el espectrograma **es** una imagen y que por lo tanto merece un procesador de imágenes. **PANNs** (Kong et al., 2020) llevó eso a escala: una familia de CNN preentrenadas sobre AudioSet completo, con CNN14 como caballo de batalla, que alcanza 0.439 mAP en AudioSet full y 0.278 en balanced. **PSLA** (Gong, Chung y Glass, 2021) —el trabajo previo de los mismos autores— combinó EfficientNet con *attention pooling* y una receta cuidada de preentrenamiento, muestreo balanceado y agregación, llegando a 0.444 single / 0.474 ensamble en full.

El sesgo inductivo que justificaba todo esto está enunciado en la introducción del paper con una palabra reveladora: "the inductive biases inherent to CNNs such as spatial locality and translation equivariance **are believed to be helpful**". *Believed*. El paper señala desde la primera página que era una creencia, no un resultado medido.

### Los híbridos CNN+atención y por qué eran un compromiso

"CNN+attention" cubre al menos tres construcciones distintas que conviene separar:

| Familia | Qué hace | Ejemplos |
|---|---|---|
| **(a) Atención como reemplazo del pooling final** | La CNN produce un mapa de features y, en vez de promediarlo sobre el tiempo, se aprende un peso por posición: $z=\sum_t \alpha_t h_t$ | PANNs, PSLA |
| **(b) Transformer apilado sobre la CNN** | La CNN actúa como *front-end* que reduce el espectrograma a una secuencia corta; encima corre un encoder Transformer completo | Miyazaki et al. (DCASE 2020), Kong et al. (2020) |
| **(c) Convolución y atención entrelazadas en cada bloque** | Cada bloque tiene un módulo de auto-atención *y* uno convolucional: la atención captura lo global, la convolución lo local | [Conformer](/papers/conformer-gulati-2020) (Gulati et al., 2020) |

El argumento estructural contra las tres es el mismo, y AST lo enuncia en la introducción: el Transformer "can capture long-range global context **even in the lowest layers**", mientras que en un híbrido el contexto global solo aparece *después* de que la CNN ya comprimió.

Concretamente: un espectrograma de AudioSet es de $1024\times128$; una EfficientNet lo reduce por un factor de 32 en cada eje, así que la atención opera sobre un mapa de aproximadamente $32\times4$. Eso es barato —128 posiciones— pero significa que **la decisión de qué información sobrevive hasta la capa de atención ya fue tomada por convoluciones locales de $3\times3$**, capa tras capa, sin acceso a contexto global. Si un evento sonoro se define por la coocurrencia de energía en la banda mel 10 y en la banda 95, ninguna capa baja de la CNN puede representarlo; para cuando la atención puede verlo, el mapa ya perdió la resolución que lo hacía discriminativo.

Hay además un costo de ingeniería: el híbrido tiene **dos conjuntos de hiperparámetros arquitectónicos** —los de la CNN y los del Transformer— y el punto de corte entre ambos es una decisión de diseño más, que en la práctica se re-sintoniza por tarea.

## La contribución: la primera arquitectura de audio libre de convolución

Es una afirmación fuerte, y el paper tiene la honestidad de matizarla **él mismo**:

> "Strictly speaking, the patch embedding layer can be viewed as a single convolution layer with a large kernel and stride size, and the projection layer in each Transformer block is equivalent to $1\times1$ convolution. However, the design is different from conventional CNNs that have multiple layers and small kernel and stride sizes. These Transformer models are usually referred to as convolution-free to distinguish them from CNNs."

Esto es literalmente cierto: **toda implementación real de AST implementa el parcheo como un `nn.Conv2d(1, 768, kernel_size=16, stride=10)`**. Extraer parches solapados y proyectarlos linealmente *es* una convolución. Así que hay que ser preciso sobre qué se está afirmando:

| Propiedad de una CNN convencional | ¿La tiene AST? |
|---|---|
| Múltiples capas convolucionales apiladas | **No.** Una sola operación tipo convolución, en la entrada |
| Kernels pequeños ($3\times3$) | **No.** Kernel $16\times16$ |
| Stride pequeño, solape alto | Parcialmente: stride 10 con kernel 16 |
| Jerarquía de campos receptivos que crecen con la profundidad | **No.** Desde la capa 1 el campo receptivo es global |
| Equivarianza traslacional en las capas profundas | **No.** El embedding posicional la rompe deliberadamente |
| *Pooling* espacial progresivo | **No.** La resolución de tokens es constante en las 12 capas |

{{< concept-alert type="clave" >}}
La afirmación defendible no es "no hay ninguna operación expresable como convolución", sino: **ningún sesgo inductivo convolucional participa del modelado**. La única operación tipo convolución es la tokenización —un preprocesamiento aprendido, sin apilamiento y sin jerarquía—. Todo el modelado ocurre en auto-atención global. Es la misma convención terminológica que ViT y DeiT establecieron para imágenes.
{{< /concept-alert >}}

El paper reclama tres ventajas: (1) desempeño superior en los tres benchmarks; (2) soporte natural de entradas de longitud variable y aplicabilidad a tareas distintas **sin cambio de arquitectura**; y (3) arquitectura más simple, menos parámetros y convergencia más rápida. La convergencia sí está cuantificada —AST necesita **5 épocas** en AudioSet full contra **30** de PSLA— pero lo de "menos parámetros" no aparece medido en ninguna parte (ver [Erratas](#erratas-y-matices)).

## La arquitectura

El pipeline completo, siguiendo la Sección 2.1 y la Figura 1 del paper:

```
waveform (16 kHz, t segundos)
  → banco de filtros log-Mel: 128 bandas, ventana Hamming de 25 ms, hop de 10 ms
  → espectrograma  128 × 100t
  → división en N parches de 16×16, stride 10 (solape 6) en AMBAS dimensiones
  → proyección lineal de cada parche (256 valores) a un embedding de 768
  → [CLS] prepended  →  + embedding posicional entrenable
  → encoder Transformer: 12 capas, 12 cabezas, dim 768  (= ViT-Base / DeiT-Base)
  → salida del token [CLS] = representación del espectrograma
  → capa lineal + sigmoide → etiquetas
```

Cuatro puntos son específicos del audio y no se deducen de conocer ViT:

**El frame rate de 100 Hz.** La relación "25 ms de ventana cada 10 ms $\to 100t$ frames" es la parametrización estándar de Kaldi y fija la resolución temporal de la entrada en **10 ms por frame**. Un clip de 10 s produce ~1000 frames: para 2021, una secuencia larga incluso antes de parchear (BERT trabajaba con 512 tokens).

**Por qué log-Mel y no la forma de onda.** El paper no lo discute, lo hereda. Pero es la decisión que hace posible todo lo demás: el espectrograma convierte una señal de $16000t$ muestras en una matriz de $128\times100t$, una reducción de **12.5×** en el eje temporal, y le da al objeto una estructura 2D con ejes interpretables. Sin ese paso, tokenizar audio para un Transformer sería un problema completamente distinto — es el problema que resuelven wav2vec 2.0 y [HuBERT](/papers/hubert-hsu-2021) con un *stem* convolucional que sí es una CNN.

**El eje de frecuencia es mel, no lineal.** Las 128 bandas mel comprimen las altas frecuencias y expanden las bajas, imitando la resolución perceptual del oído. Consecuencia para el parcheo: un parche de 16 bandas cubre un rango de frecuencias **muy distinto** según dónde esté en el eje — las bandas 0-15 abarcan quizá 0-250 Hz; las bandas 112-127, varios kilohertz. El modelo tiene que aprender esa no-uniformidad, y lo único que se lo permite es el **embedding posicional**: sin él, dos parches de bandas distintas serían indistinguibles.

**Sigmoide, no softmax.** El cabezal usa "a linear layer with sigmoid activation" porque AudioSet es **multietiqueta** (527 clases, un clip puede contener varias simultáneamente) y la pérdida es *binary cross-entropy*.

Y una decisión de diseño que subordina todo lo demás: el paper insiste en usar "the original Transformer encoder architecture **without modification**", porque cualquier modificación arquitectónica rompería la compatibilidad de pesos con los checkpoints de ViT.

### El número de parches

El paper da la fórmula

$$N = 12 \left\lceil \frac{100t - 16}{10} \right\rceil$$

Descompuesta: con parches de lado $p=16$ y solape $o=6$, el stride es $s=p-o=10$, y el número de ventanas deslizantes sobre un eje de longitud $L$ es $n(L)=\lfloor (L-p)/s\rfloor+1$. En **frecuencia** ($L=128$, fijo): $n_f = \lfloor 112/10\rfloor + 1 = 12$ — de ahí sale el 12 constante. En **tiempo** ($L=100t$): $n_t = \lfloor(100t-16)/10\rfloor + 1$.

Para $t=10$ s eso da $N = 12\times 99 = 1188$. Pero el paper reporta **tres números distintos para lo mismo** (1188, 1200 y 1212), y ninguno cuadra con las tablas salvo si el espectrograma se rellena a 1024 frames. El desglose completo está en [Erratas y matices](#erratas-y-matices); la fórmula operativa correcta es:

$$N = \left(\left\lfloor \frac{F - p}{s}\right\rfloor + 1\right)\left(\left\lfloor \frac{T_{\text{pad}} - p}{s}\right\rfloor + 1\right), \qquad F = 128,\ p = 16,\ s = 10,\ T_{\text{pad}} = 1024$$

que produce **1212** tokens para un clip de 10 s.

### Por qué el solape, a diferencia de ViT

Esta es la divergencia deliberada respecto de ViT, que **parte la imagen en parches disjuntos** ($224/16=14$, grilla $14\times14$, stride = kernel). AST usa stride 10 con kernel 16, es decir un solape de 6 en ambos ejes. Tres razones lo justifican:

**1. Está medido y funciona.** La mAP crece monótonamente con el solape: de 0.336 (sin solape) a 0.347 (solape 6) en balanced, y de 0.451 a 0.459 en full.

**2. Elimina el artefacto de la grilla rígida.** En una imagen, partir en una grilla fija es benigno porque los objetos son grandes y redundantes. En un espectrograma, muchos eventos son **estructuras finas y localizadas**: un transitorio percusivo ocupa 2-3 frames (20-30 ms); un armónico ocupa 1-2 bandas mel. Con parches disjuntos, un transitorio que cae justo sobre el borde entre dos parches queda partido en dos mitades incompletas, y ninguna de las dos contiene el patrón. Con solape 6, **cada punto del espectrograma aparece en varios parches con distintos desplazamientos**, así que siempre existe al menos un parche donde el evento está bien centrado. Es, en efecto, una forma barata de recuperar algo de la equivarianza traslacional que se perdió al abandonar la convolución.

**3. Aumenta la densidad de tokens sin cambiar la resolución de entrada:** de 512 a 1212 tokens, 2.37× más contexto para la atención sobre el mismo espectrograma.

**Qué cuesta.** El paper lo señala sin ambigüedad: aumentar el solape "will **quadratically increase the computational overhead**". Pasar de 512 a 1212 tokens multiplica el costo de la matriz de atención por $(1212/512)^2 = 5.6$. Comprar 1.1 puntos de mAP con 5.6× de cómputo cuadrático es un mal intercambio en producción, y el propio paper ofrece la salida: incluso sin solape, AST supera a PSLA.

Un hueco del estudio, no un error: el solape se aplica **igual en tiempo y en frecuencia**, pero los dos ejes no son equivalentes. En frecuencia hay 128 bandas fijas y el fenómeno relevante (armónicos) es de estructura amplia; en tiempo hay 1024 frames y el fenómeno relevante (transitorios) es de estructura fina. Un solape asimétrico —mayor en tiempo, menor o nulo en frecuencia— parecería la elección natural y reduciría el conteo de tokens. El paper no lo prueba.

## La transferencia cross-modal desde ImageNet: el truco central

La Sección 2.2 abre con el diagnóstico: "One disadvantage of the Transformer compared with CNNs is that the Transformer **needs more data to train**. [...] the Transformer only starts to outperform CNNs when the amount of data is over **14 million** for image classification tasks. However, **audio datasets typically do not have such large amounts of data**."

Esa frase es, palabra por palabra, la primera objeción que plantea la [Clase 39](/clases/clase-39). La diferencia es que el paper la formula para **resolverla**.

La observación que habilita la solución está en la misma sección: "images and audio spectrograms have **similar formats**". Ambos son tensores 2D densos con estructura local. Y hay precedente: la transferencia ImageNet → audio ya era práctica establecida en la línea convolucional (VGGish, ESResNet, el propio PSLA), "**but only for CNN-based models**". AST la extiende a arquitecturas atencionales, donde no era obvio que funcionara porque el objeto a transferir es distinto.

El modelo fuente es específico: **DeiT-Base distilled a resolución $384\times384$** — 87M de parámetros, 85.2% top-1 en ImageNet 2012, entrenado con destilación desde un profesor CNN. La elección no es casual: DeiT fue justamente la respuesta al problema que motiva esta sección — ViT necesitaba JFT-300M para superar a las CNN, y DeiT mostró que con augmentación fuerte, regularización y destilación se podían entrenar Transformers de visión competitivos usando **solo ImageNet-1k**. AST hereda ese trabajo.

Adaptarlo requiere tres ajustes.

### (a) El promedio de los canales RGB

**El problema:** la capa de embedding de parches de ViT tiene pesos de forma $[768, 3, 16, 16]$ — tres canales de entrada. AST recibe un espectrograma de **un solo canal**.

**La solución:** "we **average the weights** corresponding to each of the three input channels of the ViT patch embedding layer", es decir $W_{\text{AST}}[o,0,i,j] = \frac{1}{3}\sum_{c=0}^{2} W_{\text{ViT}}[o,c,i,j]$.

El paper justifica por qué es la elección correcta: "This is **equivalent to expanding a single-channel spectrogram to 3-channels with the same content**, but is computationally more efficient." La equivalencia es exacta. Si se replicara el espectrograma $x$ en los tres canales, la salida de la convolución sería

$$y = \sum_{c=0}^{2} W_c * x = \left(\sum_c W_c\right) * x$$

Promediar en vez de sumar solo introduce un factor $1/3$ en la escala, que es precisamente lo deseable: el promedio preserva la **magnitud** típica de las activaciones que las capas siguientes esperan, mientras que la suma las triplicaría. No es una aproximación: es la misma función, con un tercio de los FLOPS en la capa de parcheo.

A esto se suma una normalización de la entrada a **media 0 y desviación 0.5**. El 0 es esperable; el 0.5 el paper no lo explica. La lectura razonable es que los pipelines de ViT/DeiT normalizan las imágenes a media y desviación 0.5, dejando los píxeles en un rango aproximado de $[-1,1]$, y los pesos preentrenados de la primera capa fueron optimizados para activaciones de esa escala. Fijar $\sigma=0.5$ **alinea el rango dinámico de la entrada de audio con el rango dinámico de la entrada de imagen para la que los pesos fueron entrenados**.

### (b) El *cut-and-interpolate* del embedding posicional

Este es el corazón del paper y el mecanismo que más se ha reutilizado después.

**El problema.** ViT tiene un embedding posicional **entrenable** de forma $[1+n_{\text{patches}}, 768]$. Ese tensor no es genérico: "it **learns to encode the spatial information** during the ImageNet training". Después de entrenar, los embeddings de parches vecinos son similares entre sí y la estructura de la grilla 2D está codificada en ellos. Descartarlos es descartar información real.

Pero la forma no calza. Un ViT que toma imágenes de $384\times384$ con parches de $16\times16$ sin solape tiene $24\times24=576$ parches. AST necesita una grilla de $12\times n_t$ donde $n_t$ **depende de la duración del audio**: ~101 para AudioSet (10 s), ~51 para ESC-50 (5 s), ~11 para Speech Commands (1 s).

**El mecanismo.** El nombre lo dice todo: *cut* en un eje, *interpolate* en el otro.

```
pos_embed de ViT:  [576, 768]
  → reshape a grilla 2D:            [768, 24, 24]     (canales, alto=freq, ancho=tiempo)
  → CUT en el eje de frecuencia:    [768, 12, 24]     (24 → 12: se descartan filas)
  → INTERPOLATE bilineal en tiempo: [768, 12, 101]    (24 → 101: se estira)
  → flatten:                        [1212, 768]
  → el pos_embed del [CLS] se reutiliza tal cual
```

**Por qué cortar en un eje e interpolar en el otro, y no lo mismo en ambos.** Esta es la parte que revela que la decisión es sobre **semántica de ejes**, no sobre aritmética:

- El **eje de frecuencia** va de 24 a 12 posiciones. Interpolar hacia abajo mezclaría embeddings de bandas adyacentes, difuminando la distinción entre "grave" y "agudo", que es la señal más discriminativa del espectrograma. Además, la grilla de frecuencia de AST es **fija** (128 bandas siempre): no necesita ser elástica. Cortar preserva 12 embeddings **intactos y mutuamente distinguibles**, que es exactamente lo que hace falta para que el modelo sepa en qué banda está cada parche.
- El **eje temporal** va de 24 a un valor que **cambia con la tarea**. Aquí sí se necesita elasticidad, y la interpolación la da: estirar la estructura aprendida "izquierda-a-derecha" del ViT sobre la longitud que sea. La monotonía relativa se preserva —los embeddings de frames vecinos siguen siendo parecidos—, que es la propiedad que importa.

{{< concept-alert type="clave" >}}
El principio general: **se corta el eje rígido y se interpola el eje elástico.** Es una decisión de modelado, no de conveniencia numérica, y es exactamente lo que permite que la misma arquitectura funcione con entradas de 1 a 10 segundos sin retocar nada.
{{< /concept-alert >}}

Un detalle que el paper no especifica: si el corte de $24\to12$ toma las primeras 12 filas o las 12 centrales. La implementación oficial recorta **desde el centro**, lo que conserva la región de la grilla ViT con estadísticas más regulares y evita los bordes, donde los embeddings posicionales suelen ser atípicos.

Y una nota sobre el token especial: DeiT tiene **dos** tokens especiales durante el entrenamiento en ImageNet (`[CLS]` y `[DIST]`, este último entrenado contra la predicción del profesor CNN). AST los **promedia en uno solo**, tanto los embeddings de token como sus embeddings posicionales. Es una heurística que el paper no ablaciona.

### (c) El reemplazo del cabezal

"Since the classification task is essentially different, we **abandon the last classification layer of the ViT and reinitialize a new one** for AST." Trivial y esperable: las 1000 clases de ImageNet no tienen nada que ver con las 527 de AudioSet, las 50 de ESC-50 ni las 35 de Speech Commands. Lo único adicional a notar es que AST también cambia la **activación** del cabezal (sigmoide para el caso multietiqueta) y la función de pérdida (BCE).

### ¿Por qué funciona transferir de imágenes naturales a espectrogramas?

Es la pregunta legítima. Las distribuciones son radicalmente distintas: una foto tiene objetos, oclusión, iluminación, perspectiva; un espectrograma tiene rayas horizontales (armónicos sostenidos), rayas verticales (transitorios), manchas difusas (ruido de banda ancha) y trayectorias curvas (barridos de frecuencia, formantes de habla). Un espectrograma no se parece a una foto de un perro. Cuatro razones explican que la [transferencia](/fundamentos/transfer-learning) funcione igual, en orden de importancia:

**1. Lo que se transfiere no es semántica, sino estadística de segundo orden.** Los filtros de la primera capa de una red entrenada sobre imágenes naturales convergen invariablemente a un banco de **detectores de bordes orientados, blobs y patrones de frecuencia espacial** — el resultado clásico de Olshausen y Field sobre estadísticas de escenas naturales. Esos operadores no son "sobre fotos": son sobre **campos 2D con correlaciones locales**. Y un espectrograma es exactamente eso. Un detector de bordes horizontales se convierte en un detector de tono sostenido; uno de bordes verticales, en un detector de onset percusivo; uno de gradiente diagonal, en un detector de barrido de frecuencia. **La transferencia funciona porque la primitiva es geométrica, no semántica.**

**2. Lo que más se transfiere es el bloque Transformer, y ese es casi agnóstico al dominio.** De los ~87M de parámetros, la capa de parcheo tiene $768\times3\times16\times16 \approx 590$k, menos del 1%. El otro 99% son las 12 capas de [auto-atención](/fundamentos/self-attention) y MLP, que codifican algo mucho más abstracto: cómo enrutar información entre posiciones, cómo especializar las cabezas en distintos patrones de dependencia, cómo mantener el flujo residual estable. Nada de eso es específico de imágenes. Es, esencialmente, una **inicialización bien acondicionada de un Transformer profundo**.

**3. El embedding posicional transfiere una prior geométrica genuina.** La ablación lo demuestra: reinicializarlo cuesta 4.2 puntos de mAP. Lo que se transfiere es "las posiciones vecinas en 2D deben tener representaciones parecidas", cierto tanto en fotos como en espectrogramas.

**4. La distribución de destino tiene menos variabilidad, no más.** Los espectrogramas log-Mel son un objeto mucho más restringido que las imágenes naturales: siempre 128 filas, siempre el mismo eje semántico, siempre energías no negativas comprimidas logarítmicamente. Un modelo con capacidad para ImageNet está sobredimensionado para la variabilidad del espectrograma, lo que hace que la inicialización sea generosa más que restrictiva.

## Resultados

Todos los experimentos de AudioSet usan "the **exact same training pipeline** with [PSLA]" — preentrenamiento en ImageNet, muestreo balanceado, *mixup*, enmascaramiento de espectrograma, promediado de pesos y ensamblado — y el paper subraya que "[PSLA] also use ImageNet pretraining, **so it is a fair comparison**". La ganancia de AST no viene de que use ImageNet y el baseline no.

### AudioSet

[AudioSet](/papers/audioset-gemmeke-2017): más de 2 millones de clips de 10 s de YouTube, 527 clases, multietiqueta y **débilmente etiquetado**. Splits de 22k (balanced train) / 2M (full train) / 20k (eval). Métrica: mAP.

| Modelo | Arquitectura | Balanced mAP | Full mAP |
|---|---|---|---|
| Baseline (Gemmeke et al.) | CNN + MLP | — | 0.314 |
| PANNs | CNN + Atención | 0.278 | 0.439 |
| PSLA (Single) | CNN + Atención | 0.319 | 0.444 |
| PSLA (Ensemble-S) | CNN + Atención | 0.345 | 0.464 |
| PSLA (Ensemble-M) | CNN + Atención | 0.362 | 0.474 |
| **AST (Single)** | **Atención pura** | **0.347 ± 0.001** | **0.459 ± 0.000** |
| **AST (Ensemble-S)** | **Atención pura** | **0.363** | **0.475** |
| **AST (Ensemble-M)** | **Atención pura** | **0.378** | **0.485** |

Cuatro lecturas, ninguna de las cuales conviene exagerar:

- **La ganancia sobre PSLA single en full es de +1.5 mAP (0.444 → 0.459), un +3.4% relativo.** Sólida pero **modesta**. Lo notable no es el tamaño del salto sino que se consigue **eliminando por completo** el componente que todo el campo consideraba obligatorio.
- **La ganancia en balanced es de +2.8 mAP (0.319 → 0.347), un +8.8% relativo.** Esta sí es sustancial, y es la que más significa conceptualmente: **invierte la expectativa** de que el Transformer sufriría con poco dato. El paper: "AST can work better than CNN-attention hybrid models even when the training set is relatively small".
- **El *weight averaging* aporta 1.1 puntos gratis.** El modelo de la última época en full da 0.448 ± 0.001; promediar los checkpoints de todas las épocas lo sube a 0.459, sin aumentar el tamaño del modelo. Y el mejor ensamble de AST usa **6 modelos** contra los 10 de PSLA.
- **Eficiencia de entrenamiento: 5 épocas contra 30.** Un factor 6× sobre un dataset de 2M de clips. Es la ganancia menos visible y quizá la más práctica.

### ESC-50

[ESC-50](/papers/esc50-piczak-2015): 2.000 grabaciones ambientales de 5 s en 50 clases, con validación cruzada estándar de 5 pliegues (1.600 muestras de entrenamiento por pliegue). Es un dataset **muy pequeño**, lo que lo convierte en el test de estrés del argumento de eficiencia de datos. Los sufijos **-S** y **-P** distinguen los dos regímenes: sin datos de audio adicionales (solo ImageNet) y con preentrenamiento adicional en AudioSet.

| | ESC-50 (accuracy %) |
|---|---|
| SOTA-S (Sailor et al., ConvRBM) | 86.5 |
| SOTA-P (PANNs) | 94.7 |
| **AST-S** | **88.7 ± 0.7** |
| **AST-P** | **95.6 ± 0.4** |

La ganancia con AudioSet (+0.9) es marginal, aunque supera claramente la desviación estándar. La interesante es la otra: **+2.2 puntos sin ningún dato de audio adicional**, con 1.600 muestras de entrenamiento. El paper lo subraya: "AST still works well with such a small amount of data even without AudioSet pretraining".

### Speech Commands V2

[Speech Commands V2](/papers/speech-commands-warden-2018): 105.829 grabaciones de 1 s de 35 comandos hablados, evaluadas sobre las 35 clases.

| | Speech Commands V2, 35 clases (accuracy %) |
|---|---|
| SOTA-S (MatchboxNet, CNN separable 1D) | 97.4 |
| SOTA-P (Lin et al., CNN + 200M de audio de YouTube) | 97.7 |
| **AST-S** | **98.11 ± 0.05** |
| **AST-P** | **97.88 ± 0.03** |

**El resultado más contraintuitivo del paper está aquí: AST-S supera a AST-P.** Preentrenar en AudioSet **empeora** el resultado en 0.23 puntos, una diferencia pequeña pero muy por encima de las desviaciones estándar (0.05 y 0.03). El paper lo reporta y lo declara ("we find AudioSet pretraining unnecessary for the speech command classification task") pero no lo desarrolla.

La interpretación razonable: AudioSet es abrumadoramente **no-habla** —eventos ambientales, música, ruido— y sus 10 s por clip tienen una estructura temporal completamente distinta de 1 s de una palabra aislada. El preentrenamiento en AudioSet especializa el modelo en distinciones tímbricas de banda ancha, no en la estructura de formantes que discrimina "left" de "right". Es un caso de **desajuste de dominio dentro del propio audio**, y un recordatorio útil: "más preentrenamiento" no es monótonamente mejor, e ImageNet resulta ser un punto de partida más **neutro** que AudioSet para tareas de habla.

Las ganancias absolutas son marginales —con 98.11% quedan 208 errores de 11.005— pero el argumento no es la magnitud: se consiguen con **la misma arquitectura sin ningún cambio**, contra un modelo (MatchboxNet) diseñado específicamente para *keyword spotting*.

### El argumento de generalidad

El cierre de la sección experimental es la síntesis del paper: "while the input audio length varies from 1 sec. (Speech Commands), 5 sec. (ESC-50) to 10 sec. (AudioSet) and content varies from speech to non-speech, **we use a fixed AST architecture for all three benchmarks and achieve SOTA results on all of them**. This indicates the potential for AST use as a **generic audio classifier**."

Es el argumento más fuerte del paper, y el que menos depende de cuán grande sea cada ganancia individual. La línea convolucional no podía hacerlo: las CNN requerían *tuning* arquitectónico por tarea.

## Ablations

Todas sobre AudioSet balanced salvo indicación contraria, con *weight averaging* y **sin** ensambles.

### El preentrenamiento en ImageNet

| | Balanced Set | Full Set |
|---|---|---|
| **No Pretrain** | **0.148** | **0.366** |
| ImageNet Pretrain (usado) | **0.347** | **0.459** |

{{< concept-alert type="advertencia" >}}
**La cifra más reveladora del paper.** Sin preentrenamiento de ImageNet, AST cae a **0.148 mAP en balanced** (contra 0.347) y **0.366 en full** (contra 0.459). Con eso **pierde contra las CNN de la época**: PANNs alcanza 0.439 y PSLA 0.444 en full. Traducido sin adornos: con 2 millones de clips de entrenamiento, un Transformer puro entrenado desde cero pierde contra el estado del arte convolucional. **La contribución arquitectónica y la contribución de transferencia no son separables** — la primera solo funciona gracias a la segunda.
{{< /concept-alert >}}

En balanced el preentrenamiento **multiplica la mAP por 2.34**, o dicho al revés: sin ImageNet, AST pierde el 57% de su desempeño, quedando muy por debajo de cualquier baseline serio. En full la mejora es de +25% relativo; ahí el modelo sin preentrenar al menos supera al baseline CNN+MLP de AudioSet (0.314), pero no a PANNs ni a PSLA.

El paper lo enuncia de forma más diplomática pero igual de clara: el preentrenamiento en ImageNet "can **greatly reduce the demand for in-domain audio data**". Y nótese el patrón: el efecto se concentra donde hay poco dato ($2.34\times$ en balanced contra $1.25\times$ en full), lo que confirma que se comporta como un efecto de **regularización**.

### Qué checkpoint fuente usar, y el tamaño del modelo

| Pesos iniciales | # Params | ImageNet top-1 | AudioSet balanced mAP |
|---|---|---|---|
| ViT Base | 86M | 0.846 | 0.320 |
| ViT Large\* | 307M | 0.851 | 0.330 |
| DeiT sin destilación | 86M | 0.829 | 0.330 |
| **DeiT con destilación (usado)** | **87M** | **0.852** | **0.347** |

\* Entrenado sin solape de parches por límites de memoria.

**Escalar el modelo no es la palanca.** ViT-Large tiene 3.6× los parámetros de ViT-Base y solo gana 1 punto de mAP — y ni siquiera es una comparación limpia, porque tuvo que entrenarse **sin solape** por memoria, lo que cuesta ~1.1 puntos. Corrigiendo por eso, ViT-Large estaría en torno a 0.341, todavía por debajo de DeiT-Base-distilled (0.347) con **3.5× menos parámetros**. Lo que sí es palanca es la **calidad del checkpoint fuente**: la destilación de DeiT vale 1.7 puntos de mAP (0.330 → 0.347). Una cadena de causalidad curiosa — la señal de un profesor CNN sobre imágenes acaba mejorando un clasificador de audio libre de convoluciones.

### La estrategia del embedding posicional

| | Balanced Set mAP |
|---|---|
| Reinicializar | 0.305 |
| Interpolación por vecino más cercano | 0.346 |
| **Interpolación bilineal (usada)** | **0.347** |

Dos lecturas. **Reinicializar cuesta 4.2 puntos de mAP.** Descomponiendo el efecto total del preentrenamiento —19.9 puntos, de 0.148 a 0.347— resulta que **4.2 vienen específicamente del embedding posicional** y los 15.7 restantes de los pesos del encoder y del parcheo. El embedding posicional representa menos del 1% de los parámetros y aporta el **21% de la ganancia de transferencia**. Ese es el argumento de que el *cut-and-interpolate* es una contribución real y no un detalle de implementación.

Y **el método de interpolación es irrelevante** (0.346 vs 0.347, dentro del ruido). Lo que importa es **que se preserve la estructura de la grilla**, no cómo exactamente se remuestrea.

### El solape entre parches

| Configuración | # Parches | Balanced mAP | Full mAP |
|---|---|---|---|
| No Overlap | 512 | 0.336 | 0.451 |
| Overlap-2 | 657 | 0.342 | 0.456 |
| Overlap-4 | 850 | 0.344 | 0.455 |
| **Overlap-6 (usado)** | **1212** | **0.347** | **0.459** |

Monotonía casi perfecta (la única inversión, Overlap-4 en full, es de 0.001). La ganancia total del solape es de **+1.1 puntos en balanced y +0.8 en full**, a un costo de **2.37× más tokens, es decir 5.6× la matriz de atención**.

La observación práctica más útil de la tabla es del propio paper: "Even with **no patch split overlap**, AST can still outperform the previous best system" — 0.451 sin solape contra 0.444 de PSLA en full. **La tesis del paper se sostiene sin el truco caro.** Para un despliegue real, la configuración sin solape es claramente la razonable.

### Forma y tamaño del parche

Todos entrenados sin solape:

| Forma de parche | # Parches | Sin preentrenar | Con preentrenar |
|---|---|---|---|
| $128\times2$ (rectangular, en orden temporal) | 512 | **0.154** | — |
| **$16\times16$ (usado)** | 512 | 0.143 | **0.336** |
| $32\times32$ | 128 | 0.139 | — |

Esta es la ablación más honesta del paper. El punto de comparación es una tokenización alternativa **natural para audio**: rebanar el espectrograma en columnas de $128\times2$, es decir el espectro completo en cada instante, en orden temporal estricto. Eso convierte el audio en una secuencia 1D genuina, como una oración. Es lo que uno haría viniendo de RNN/ASR.

Y **funciona mejor**: 0.154 contra 0.143 con la misma área de parche y el mismo número de tokens. El paper lo reconoce, y luego lo descarta por una razón puramente pragmática: "considering there is **no $128\times2$ patch based ImageNet pretrained model**, using $16\times16$ patches is still the current optimal solution."

{{< concept-alert type="recordar" >}}
AST elige parches de $16\times16$ **no porque sean la mejor tokenización para audio, sino porque son la única compatible con los pesos de ImageNet.** La arquitectura está subordinada al truco de transferencia. Y la aritmética lo confirma: la mejor tokenización sin preentrenar da 0.154; la peor tokenización *con* preentrenar da 0.336. **El preentrenamiento vale más del doble que cualquier decisión de tokenización.**
{{< /concept-alert >}}

## La ventaja del campo receptivo global

Este es el argumento arquitectónico del paper: AST "can capture long-range global context **even in the lowest layers**".

En una pila convolucional, el campo receptivo de la capa $l$ crece según

$$R_l = R_{l-1} + (k_l - 1)\prod_{i<l} s_i$$

Con convoluciones $3\times3$ de stride 1 y sin dilatación, $R_l = 2l+1$: para cubrir los 1024 frames de un clip de AudioSet harían falta **512 capas**. Con *pooling* de factor 2 cada pocas capas —la receta real de VGG, ResNet o EfficientNet— el campo receptivo crece geométricamente, pero el campo receptivo **nominal** no es el **efectivo**: la influencia de un píxel de entrada sobre una unidad profunda decae aproximadamente como una gaussiana, así que el efectivo es del orden de la raíz del nominal. Con dilatación exponencial (WaveNet, TCN) bastan 9 capas para cubrir 1024 frames, pero el campo receptivo **sigue creciendo con la profundidad**, así que las primeras capas siguen siendo miopes.

En AST **no hay aritmética que hacer**. En la capa 1, cada token atiende a los 1212 tokens del espectrograma más el `[CLS]`. La distancia entre dos posiciones cualesquiera es **1 salto**, en **todas** las 12 capas.

| | Campo receptivo en la capa 1 | Crecimiento con la profundidad | Camino máximo entre posiciones |
|---|---|---|---|
| CNN $3\times3$ | 3 frames | lineal ($+2$/capa) | $O(n)$ capas |
| CNN dilatada | 3 frames | exponencial ($\times2$/capa) | $O(\log n)$ capas |
| CNN + atención al final | 3 frames | global **solo al final** | $O(1)$, pero entre features ya comprimidos |
| **AST** | **todo el espectrograma** | **constante (ya es total)** | **$O(1)$ en cualquier capa** |

### Dónde la diferencia es cualitativa: la estructura armónica

Un sonido tonal —una voz, un violín, un motor— tiene energía concentrada en un fundamental $f_0$ y en armónicos en $2f_0, 3f_0, \dots$. En un espectrograma eso es un **peine de rayas horizontales distribuido a lo largo de todo el eje de frecuencia**, y la identidad tímbrica está en la **relación entre las amplitudes de armónicos muy separados en el eje**: es literalmente lo que distingue un clarinete (armónicos impares dominantes) de una trompeta.

Con 128 bandas mel y parches de 16 bandas, un parche cubre **1/8 del eje de frecuencia**. Un $f_0$ grave y su sexto armónico pueden caer en parches separados por 60 u 80 bandas mel.

- En una **CNN de $3\times3$**, relacionar la banda 12 con la banda 92 requiere un campo receptivo de 80 bandas en frecuencia: unas 40 capas sin *pooling*, o bien esperar a que el *pooling* haya reducido el eje de frecuencia lo suficiente — momento en el cual la resolución fina de los armónicos individuales **ya se perdió**. La CNN queda ante un dilema: o ve la relación global sin resolución, o ve la resolución sin la relación global.
- En **AST**, una cabeza de atención de la capa 1 puede aprender un patrón del tipo "atiende a los parches cuya posición en frecuencia está en relación armónica con la mía", y **la información llega intacta**, sin haber pasado por ningún *pooling*. Las 12 cabezas permiten que varias de esas relaciones coexistan.

El mismo argumento vale para eventos largos: una sirena (el patrón es la periodicidad del barrido, con período de segundos), un tren pasando (el efecto Doppler es una deriva lenta a lo largo de todo el clip), aplausos (la textura es estadística sobre el clip completo). Relacionar el segundo 1 con el segundo 9 significa cruzar ~800 frames, unos 80 parches temporales: en AST es una entrada de la matriz de atención.

Aquí conviene un matiz honesto: el *global average pooling* de una CNN **sí** agrega información de todo el clip — al final. La diferencia no es que la CNN no pueda ver todo el clip, sino **cuándo**: el GAP promedia representaciones que ya fueron construidas localmente, y si la construcción local descartó la información necesaria, el promedio no la recupera. La atención permite que la construcción de features **sea condicional al contexto global desde el principio**. Ese es el argumento preciso.

Y un matiz más: el paper **no visualiza mapas de atención ni verifica que AST aprenda efectivamente patrones armónicos**. El argumento es arquitectónico y las cifras lo respaldan indirectamente, pero la evidencia mecanística no está en este trabajo.

## Limitaciones

### El costo cuadrático

La auto-atención cuesta $O(N^2 d)$ por capa. Para AST, $N$ crece **linealmente con la duración del audio**, así que **el costo crece cuadráticamente con la duración**. Con los parámetros del paper ($n_f=12$, 100 frames/s, $s=10$):

$$N(t) = 12 \times \left(\left\lfloor \frac{100t - 16}{10} \right\rfloor + 1\right) \approx 120\,t$$

| Duración | Frames | Parches temporales | $N$ (tokens) | Entradas de la matriz de atención | Memoria (fp32, 12 cabezas, 1 capa) |
|---|---|---|---|---|---|
| 1 s (Speech Commands) | 100 | 9 | 108 | $1.2\times10^4$ | 0.6 MB |
| 5 s (ESC-50) | 500 | 49 | 588 | $3.5\times10^5$ | 17 MB |
| **10 s (AudioSet, pad a 1024)** | **1024** | **101** | **1212** | **$1.47\times10^6$** | **71 MB** |
| 1 min | 6000 | 599 | 7188 | $5.2\times10^7$ | 2.5 GB |
| **10 min** | **60000** | **5999** | **71988** | **$5.18\times10^9$** | **249 GB** |

**Un clip de 10 minutos genera casi 72.000 parches.** La matriz de atención tiene $5.2\times10^9$ entradas: 20.7 GB por cabeza en fp32, y con 12 cabezas, **249 GB por capa**. Multiplicado por 12 capas y por lo que exige el backward, está fuera de alcance por varios órdenes de magnitud incluso hoy, con FlashAttention y fp16. El factor de escala entre 10 s y 10 min es $(71988/1212)^2 \approx \mathbf{3528\times}$ para 60× más audio.

Esta es **la** limitación de AST, y es estructural, no de implementación. El paper la reconoce solo de pasada, en el contexto del solape, pero no discute el techo de duración que impone. Toda la línea posterior —PaSST con *patchout*, HTS-AT con atención jerárquica tipo Swin, ventanas deslizantes— es esencialmente trabajo para levantar este techo. Y nótese que el solape lo **empeora 5.6×** justo en el eje donde ya duele: Overlap-6 es la decisión que menos escala de todo el diseño.

### La memoria durante el entrenamiento

Hay dos evidencias directas en el paper de que la memoria fue una restricción real. Primero, el **batch size de 12** en AudioSet — doce muestras, cuando los modelos CNN de audio de la época entrenaban con batches de 64-128 sin dificultad; sobre 2M de clips eso implica ~167.000 pasos por época. Segundo, **ViT-Large tuvo que entrenarse sin solape "due to memory limitation"**: el modelo grande literalmente no cupo con la configuración óptima.

El cálculo lo explica: con $N=1212$, guardar las matrices de atención para el backward cuesta ~71 MB por capa y por muestra, ~850 MB por muestra en 12 capas, y ~10 GB para un batch de 12 — solo en matrices de atención, sin contar activaciones, gradientes ni estados del optimizador Adam.

### La dependencia de ImageNet

Ya cuantificada, pero conviene enunciarla como limitación explícita: **sin ImageNet, AST no supera a las CNN ni siquiera con 2 millones de clips de entrenamiento.** Las consecuencias prácticas son tres:

- **La tokenización queda congelada.** $128\times2$ es mejor tokenización para audio entrenado desde cero, pero es inutilizable porque no existe checkpoint de ImageNet compatible. El diseño arquitectónico está subordinado a la disponibilidad de pesos de visión — una restricción rara y poco satisfactoria.
- **El modelo hereda las propiedades del dataset fuente.** Cualquier sesgo o artefacto de la distribución de ImageNet-1k entra al modelo de audio por esta puerta. Nadie ha auditado qué significa eso.
- **La receta no es escalable a otros dominios sin un ImageNet equivalente.** La estrategia "toma el mejor checkpoint de visión y adáptalo" solo funciona porque visión resolvió su problema de datos primero.

### Otras

- **No hay análisis de qué aprende el modelo.** Ni un mapa de atención, ni una visualización de los embeddings posicionales aprendidos, ni un análisis por clase. Es la deuda más grande del paper.
- **El solape es simétrico sin justificación**, pese a que los dos ejes tienen semántica distinta.
- **Solo clasificación.** No hay detección de eventos sonoros con localización temporal, ni separación, ni ASR. La afirmación de "clasificador de audio genérico" está bien acotada a clasificación.
- **Latencia e inferencia no se reportan.** El paper cuantifica la velocidad de **convergencia** (5 épocas contra 30) pero no da FLOPS, latencia ni memoria de inferencia, lo cual es notable dado que compara contra EfficientNet, una arquitectura diseñada precisamente para eficiencia.

## En la clase 39: la auditoría de las tres objeciones

La [Clase 39](/clases/clase-39) incluye una sección "Audio and Transformers" que enumera tres problemas y concluye que los Transformers "no son actualmente muy populares para aplicaciones de audio". AST, publicado casi tres años antes de esa clase, es el contraejemplo documentado. Vale la pena evaluar cada afirmación contra evidencia, sin sobrecorregir.

### Objeción 1: "faltan datasets de audio masivos"

**Lo que la objeción acierta.** La premisa fáctica era correcta en su momento, y **el propio AST la enuncia como su principal obstáculo**: ViT solo superaba a las CNN por encima de 14 millones de imágenes, y "audio datasets typically do not have such large amounts of data". AudioSet, el dataset etiquetado más grande de eventos sonoros, tiene 2M de clips de 10 s (~5.500 horas). Y la ablación del preentrenamiento confirma que la preocupación era legítima: **un AST entrenado desde cero sobre los 2M de AudioSet full alcanza 0.366 mAP, por debajo de PANNs (0.439) y de PSLA (0.444)**. Con los datos de audio disponibles y sin ayuda externa, el Transformer efectivamente pierde. La objeción, en su forma de 2021, tenía base empírica.

**Lo que la clase no registra** es que la objeción ya tenía dos soluciones desplegadas, ambas publicadas y ampliamente adoptadas, y que **ninguna de las dos consistió en conseguir el dataset etiquetado que faltaba**:

- **Transferencia cross-modal.** AST elude el problema en lugar de resolverlo: si no hay 14 millones de espectrogramas, se usan los 14 millones de imágenes que sí existen. La ablación mide exactamente cuánto vale ese préstamo: 0.148 → 0.347 en balanced ($\times 2.34$) y 0.366 → 0.459 en full (+25%). El "dataset masivo" no tiene por qué ser del dominio de destino.
- **Preentrenamiento autosupervisado.** La respuesta más limpia, y la que elimina la dependencia. **SSAST** (Gong et al., AAAI 2022, del mismo primer autor) preentrena sobre audio **sin etiquetar** mediante enmascaramiento de parches de espectrograma, y la línea continúa con Audio-MAE (2022) y BEATs (2023). Aquí la restricción se disuelve por completo: la escasez nunca fue de audio, fue de audio **etiquetado**, y audio sin etiquetar hay ilimitado.

A nivel de escala bruta, para 2024 la premisa ya era discutible por otro lado: Whisper se entrenó sobre 680.000 horas de audio con supervisión débil —dos órdenes de magnitud más que AudioSet— y LibriLight aporta 60.000 horas de habla sin etiquetar para wav2vec 2.0 y HuBERT.

**Veredicto: válida en su origen, superada por dos vías distintas antes de 2022.**

### Objeción 2: "la self-attention opera sobre entidades discretas y el audio no se segmenta trivialmente"

**Esta es la objeción que AST desactiva de forma más directa y completa.**

La objeción contiene un supuesto oculto que es el que falla: **que los tokens deben ser unidades semánticamente significativas**. En texto, las palabras lo son, y por eso la analogía sugiere que en audio habría que segmentar en fonemas, notas o eventos — un problema difícil y circular, porque segmentar bien el audio requiere ya entenderlo.

[ViT](/fundamentos/vision-transformer) había refutado ese supuesto en imágenes un año antes. Una imagen tampoco tiene "palabras": no hay una segmentación trivial en objetos, y segmentar bien es un problema tan difícil como clasificar. ViT resolvió eso **ignorando la pregunta**: partir en una grilla regular de $16\times16$ píxeles. Los parches no respetan bordes de objetos, cortan caras por la mitad, mezclan fondo y figura. Y funciona. AST hace exactamente lo mismo sobre el espectrograma.

{{< concept-alert type="clave" >}}
El principio que se desprende, y que es la respuesta precisa a la objeción: **la tokenización no necesita ser semántica; basta con que sea regular, exhaustiva y complementada con información posicional.** Los tres requisitos reales son **cobertura** (los tokens cubren toda la entrada sin perder información — una grilla regular lo garantiza por construcción), **regularidad** (el significado de un token depende solo de su contenido, no de un proceso de segmentación variable y frágil río arriba) y **posición** (un embedding posicional que diga dónde está cada token — la ablación demuestra que es esencial).
{{< /concept-alert >}}

Con eso, **la auto-atención construye por sí misma las agrupaciones semánticas** que la segmentación explícita habría tenido que producir. Ese es literalmente su trabajo: aprender qué posiciones se relacionan con cuáles. Exigir una segmentación semántica *antes* de la atención es pedirle al preprocesamiento que resuelva lo que el modelo está diseñado para resolver.

Y el paper **mide** esta decisión en lugar de asumirla. La ablación de forma de parche compara la grilla $16\times16$ contra $128\times2$ —lo más parecido a una segmentación "natural" del audio como secuencia 1D— y encuentra que la diferencia es de 0.011 mAP, mientras que el preentrenamiento vale 0.193. **La elección de tokenización es un efecto de segundo orden.** Ese es el dato que cierra la objeción: no es que AST haya encontrado la segmentación correcta del audio; es que **la segmentación resultó no ser el problema**.

El resultado transversal lo confirma: la misma tokenización y la misma arquitectura, sin ningún cambio, alcanzan el estado del arte en habla (1 s), sonidos ambientales (5 s) y eventos sonoros generales (10 s). Si la segmentación fuera el problema, cada uno de esos dominios habría exigido una noción distinta de "entidad discreta". No la exigió.

**Veredicto: disuelta.**

### Objeción 3: "los Transformers no son buenos para modelar dependencias largas"

**Es la más problemática de las tres, porque invierte la motivación original de la auto-atención.**

Vaswani et al. (2017), en la Sección 4 de "[Attention Is All You Need](/papers/attention-is-all-you-need-vaswani-2017)", justifican la auto-atención con una tabla explícita de tres criterios, y el tercero es exactamente este:

| Tipo de capa | Complejidad por capa | Operaciones secuenciales | **Longitud máxima del camino** |
|---|---|---|---|
| Auto-atención | $O(n^2 \cdot d)$ | $O(1)$ | $\mathbf{O(1)}$ |
| Recurrente | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolucional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(n/k)$, o $O(\log_k n)$ con dilatación |

Y el argumento textual: "One key factor affecting the ability to learn such dependencies is **the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies**."

La auto-atención fue diseñada **precisamente para modelar dependencias largas mejor que las alternativas**. Con camino $O(1)$, el gradiente entre dos posiciones separadas por 1000 pasos atraviesa **una** operación; en una RNN atraviesa 1000 multiplicaciones matriciales sucesivas — que es exactamente el mecanismo del gradiente que se desvanece y la razón histórica por la que las RNN fallaban en dependencias largas.

**Cuál es el problema real, y por qué la confusión es entendible.** El costo de la auto-atención es $O(n^2 d)$ por capa contra $O(nd^2)$ de una RNN: es **cuadrático en la longitud**, y por eso las secuencias largas son caras. Pero eso es **una limitación de costo, no de capacidad**. Si la secuencia cabe en memoria, el Transformer modela la dependencia mejor que cualquier alternativa. El error de la objeción es confundir "no puedo permitirme una secuencia larga" con "no modelo bien lo que hay dentro de ella".

En AST esto es cuantificable exactamente: un clip de 10 s genera 1212 tokens y entra sin problemas; uno de 10 minutos genera 71.988 y no entra en ningún hardware razonable. **El techo es de cómputo. Dentro de los 1212 tokens, la conectividad es total y el camino es $O(1)$.**

**Dónde la objeción tiene algo de razón, si se la reformula con caridad.** Hay tres fenómenos reales que podrían haberla motivado, y ninguno es "la auto-atención no modela dependencias largas":

1. **Dilución de la atención.** El softmax sobre $n$ claves reparte una masa total de 1; cuando $n$ crece mucho, es más difícil que una posición se concentre nítidamente en otra. Es un problema del **comportamiento del softmax**, no de conectividad estructural.
2. **Extrapolación posicional.** Los embeddings posicionales aprendidos —los que usa AST— no extrapolan a longitudes no vistas. Es exactamente el problema que motiva el *cut-and-interpolate*, y también toda la línea de RoPE y ALiBi. Es un problema del **esquema de posición**.
3. **Escasez de supervisión de largo alcance.** Aunque la arquitectura permita la dependencia, el modelo solo la aprende si los datos la exhiben y la pérdida la premia. Con una sola etiqueta débil por clip de 10 s, la señal que empuja a aprender dependencias de 8 segundos es tenue. Es un problema de **datos y objetivo**.

**Veredicto: incorrecta como está formulada.** La versión correcta sería: *"la auto-atención tiene el camino más corto posible entre posiciones, lo que la hace la arquitectura mejor equipada para dependencias largas; su limitación es el costo cuadrático en la longitud, más problemas prácticos de extrapolación posicional y de dilución del softmax en contextos muy extensos."*

### Síntesis

| Objeción de la Clase 39 | Estado | Evidencia |
|---|---|---|
| 1. Falta de datasets masivos de audio | **Válida en su origen, superada por dos vías** | Ablación de preentrenamiento (transferencia cross-modal: $\times2.34$ en balanced, +25% en full); SSAST / Audio-MAE / BEATs (autosupervisión); Whisper (680k h) |
| 2. La auto-atención necesita entidades discretas y el audio no se segmenta trivialmente | **Disuelta** | Parcheo regular $16\times16$ con solape; la ablación de forma de parche muestra que la tokenización es de segundo orden frente al preentrenamiento; misma arquitectura para 1 s, 5 s y 10 s, habla y no-habla |
| 3. Los Transformers no modelan bien dependencias largas | **Incorrecta como está formulada** | Vaswani et al. 2017, Sección 4: camino $O(1)$ frente a $O(n)$ de una RNN. El problema real es el costo $O(n^2)$, más extrapolación posicional y dilución del softmax |

Y sobre la conclusión del slide —que los Transformers no son populares en audio—: para 2024 era difícil de sostener. Whisper es un Transformer y era el modelo de ASR más usado del mundo; wav2vec 2.0 y [HuBERT](/papers/hubert-hsu-2021) son Transformers; los encoders de audio de todos los LLM multimodales de 2023-2024 son Transformers; AST y sus descendientes eran el baseline estándar de clasificación de audio desde 2021.

Lo que sí es cierto, y probablemente esté detrás de la afirmación, es que **la convolución no desapareció**: sobrevive como *stem* compresor de secuencia (Whisper, wav2vec 2.0) y como módulo local dentro del bloque ([Conformer](/papers/conformer-gulati-2020)), precisamente para mitigar el costo cuadrático — que es la objeción 3 en su forma correcta.

## Por qué importa hoy

**La línea directa.** **SSAST** (Gong et al., AAAI 2022) es la continuación natural del mismo grupo: preentrenamiento autosupervisado sobre parches de espectrograma —enmascarar y predecir, con un objetivo combinado discriminativo y generativo— usando AudioSet y LibriSpeech sin etiquetar. El objetivo declarado es exactamente eliminar la dependencia de ImageNet que la ablación dejó al descubierto: es el eslabón que convierte "necesitamos visión para entrenar audio" en "el audio puede entrenarse a sí mismo". **PaSST** (Koutini et al., 2022) ataca el costo cuadrático con *patchout*: descartar aleatoriamente una fracción de los parches durante el entrenamiento, lo que reduce cómputo y regulariza a la vez. **HTS-AT** introduce jerarquía tipo Swin —atención por ventanas con desplazamiento y reducción progresiva de tokens—, recuperando parte de la eficiencia de la pirámide convolucional sin volver a la convolución. **Audio-MAE** (2022) traslada el *masked autoencoder* al espectrograma. Y **BEATs** (2023) cierra el círculo con preentrenamiento iterativo: un tokenizador acústico aprendido produce etiquetas discretas para el modelo enmascarado, y el modelo mejorado re-entrena el tokenizador.

**La ruta hacia los encoders multimodales actuales.** Los encoders de audio de los sistemas multimodales contemporáneos son, casi sin excepción, **Transformers sobre representaciones tiempo-frecuencia**. CLAP y sus variantes —el análogo de CLIP para audio— usan encoders de la familia AST/HTS-AT alineados contra un encoder de texto por contraste, lo que habilita clasificación *zero-shot* y búsqueda texto→audio. Whisper usa un encoder Transformer sobre espectrograma log-Mel de 80 bandas, aunque **conserva un stem convolucional** de dos capas Conv1D con stride para reducir la longitud de secuencia: no es convolution-free, y ese es el dato interesante — la convolución sobrevivió como **compresor de secuencia**, que es donde realmente aporta. Y los LLM con entrada de audio (Qwen-Audio, SALMONN, LTU/LTU-AS, este último también de Yuan Gong) conectan un encoder tipo AST o Whisper a un LLM mediante un proyector.

El patrón general: **AST estableció que el camino canónico para meter audio en un Transformer es "espectrograma → parches → tokens"**, y esa decisión sobrevivió, aunque los detalles —preentrenamiento, jerarquía, stem convolucional— cambiaron.

**Por qué sigue siendo el baseline de referencia.** Cuatro razones concretas. Está en `transformers` de Hugging Face con checkpoints listos (`ASTForAudioClassification`), y bajar un modelo con 0.459 mAP en AudioSet toma tres líneas — esa accesibilidad es lo que convierte un paper en un baseline. Es simple de describir y reimplementar, porque el paper insiste deliberadamente en el encoder Transformer estándar "without modification": un baseline con partes móviles propias es un mal baseline. Ocupa una posición conceptual limpia en el espacio de diseño —atención pura, sin jerarquía, sin trucos de eficiencia, con transferencia cross-modal—, así que cualquier propuesta nueva puede posicionarse respecto de él quitando o agregando exactamente una cosa. Y las cifras siguen siendo respetables: 0.459 mAP single no fue superado por márgenes enormes hasta la generación autosupervisada.

Hay un mérito histórico más difícil de medir. AST fue una de las demostraciones que **cerraron el debate sobre si el sesgo inductivo convolucional era necesario**. En 2020 la respuesta obvia era sí; en 2021, entre ViT, DeiT y AST, quedó claro que era **una forma de ahorrar datos**, no un requisito. Con suficientes datos —propios o prestados de otra modalidad— la atención pura basta.

## Erratas y matices

**1. El paper da tres valores distintos para el número de parches.** La fórmula $N=12\lceil(100t-16)/10\rceil$ da **1188** para $t=10$; el texto de la Sección 2.2 dice **$12\times100=1200$**; y la Tabla 5 reporta **1212**. El valor operativo es 1212, y las filas de las Tablas 5 y 6 solo cuadran si el espectrograma se rellena a **1024 tramas**, no a 1000:

| Configuración | Stride | $n_f$ | $N$ con $T=1000$ | $N$ con $T=1024$ | **Tabla 5** |
|---|---|---|---|---|---|
| No Overlap | 16 | 8 | 496 | **512** | 512 |
| Overlap-2 | 14 | 9 | 639 | **657** | 657 |
| Overlap-4 | 12 | 10 | 830 | **850** | 850 |
| Overlap-6 (usado) | 10 | 12 | 1188 | **1212** | 1212 |

Las cuatro filas coinciden exactamente con $T=1024$ y ninguna con $T=1000$. La Tabla 6 lo confirma de forma independiente: parches de $128\times2$ dan $1\times(1024/2)=512$ y parches de $32\times32$ dan $4\times(1024/32)=128$, ambos exactamente los valores reportados. La explicación es prosaica: 1024 es la potencia de 2 inmediatamente superior a 1000, cómoda para el padding y para que la división sin solape sea exacta ($1024/16=64$, $128/16=8$). **La fórmula del paper describe el caso ideal sin padding; la implementación usa el caso práctico.**

**2. El claim de "menos parámetros" no está cuantificado y probablemente es falso en absoluto.** La introducción afirma que "comparing with SOTA CNN-attention hybrid models, AST features a simpler architecture with **fewer parameters**, and converges faster during training". La convergencia sí está medida (5 épocas contra 30); **el conteo de parámetros de AST no aparece en ninguna parte del paper**. Lo único que se sabe es que el checkpoint DeiT fuente tiene 87M, y AST hereda esencialmente esa cuenta. PSLA se basa en EfficientNet-B2, que es de un orden de magnitud menor. La lectura caritativa es que la afirmación se refiere a los *ensambles* (6 modelos contra 10) o a "simpler architecture" en sentido estructural. Como está escrita, es la afirmación menos sostenida del paper.

**3. AST no destila; solo hereda pesos de un modelo destilado.** No hay profesor CNN, no hay pérdida de destilación, no hay token de destilación en el modelo de audio: los dos tokens especiales de DeiT se **promedian en uno solo** y los dos cabezales se descartan. La destilación ocurrió enteramente **río arriba**, durante el entrenamiento de DeiT sobre ImageNet. El malentendido es frecuente porque "DeiT w/ Distill" aparece en la Tabla 3.

**4. La correlación ImageNet ↔ AudioSet no es monótona, contra lo que afirma el texto.** El paper sostiene que el modelo que mejor rinde en ImageNet también rinde mejor en AudioSet. Es cierto para el ganador (DeiT distilled: 0.852 → 0.347), pero **falla en el medio de la tabla**: DeiT sin destilación tiene *menor* accuracy en ImageNet que ViT-Base (0.829 contra 0.846) y sin embargo *mayor* mAP en AudioSet (0.330 contra 0.320). Lo que transfiere no es solo la accuracy alcanzada, sino algo de la **receta de entrenamiento** de DeiT — augmentación fuerte, regularización. El paper no lo comenta. Y ViT-Large está handicapeado por haberse entrenado sin solape, así que su 0.330 no es estrictamente comparable con las demás filas.

**5. "Convolution-free" tiene una excepción que el propio paper declara.** La capa de parcheo *es* una convolución de kernel 16 y stride 10. La afirmación defendible es que ningún sesgo inductivo convolucional participa del modelado, no que no exista ninguna operación expresable como convolución.

**6. Los 0.485 mAP del abstract son de un ensamble de 6 modelos, no de un modelo único.** El modelo único con *weight averaging* da 0.459; el modelo único en la última época, 0.448. Comparar 0.485 contra el modelo único de otro trabajo sería incorrecto. El paper es transparente al respecto en la Tabla 1, pero el abstract cita solo la cifra alta.

**7. Cita cruzada inconsistente en Speech Commands.** El texto dice que AST-S supera al SOTA de la referencia [9] (Rybakov et al., *streaming keyword spotting*), pero la Tabla 7 identifica el SOTA-S como [34] (MatchboxNet, 97.4%). La cifra de comparación correcta es la de la tabla.

**8. Los 16 kHz no están escritos explícitamente en el paper.** Se deducen de la parametrización estándar: 25 ms / 10 ms es la configuración canónica de Kaldi, y AudioSet se procesa a 16 kHz en la práctica común. El paper solo especifica "128-dimensional log Mel filterbank features computed with a 25ms Hamming window every 10ms".

**9. La normalización a $\sigma=0.5$ no está justificada en el paper.** Se enuncia el valor y nada más. La explicación de alineación del rango dinámico con las imágenes normalizadas de ViT es interpretación razonable, no cita.

**10. No hay ablación de tasa de aprendizaje.** El paper reporta las tasas usadas —de $10^{-5}$ a $2.5\times10^{-4}$ según la tarea, con el patrón claro de que a más preentrenamiento acumulado, menor la tasa— pero no ablaciona ninguna.

## Notas y enlaces

- **Código oficial:** `github.com/YuanGongND/ast`. En producción nadie reimplementa AST: está en `transformers` de Hugging Face como `ASTFeatureExtractor` + `ASTForAudioClassification`.
- **El nombre del checkpoint codifica la configuración.** `MIT/ast-finetuned-audioset-10-10-0.4593` significa *frequency stride 10, time stride 10, mAP 0.4593* — la configuración Overlap-6. Existen variantes `12-12`, `14-14` y `16-16` que corresponden a las demás filas de la tabla de solape y son **considerablemente más baratas**: si el cómputo importa, `16-16` (sin solape, 512 tokens) da 0.451 y cuesta 5.6× menos en atención.
- **Para AudioSet usa `sigmoid`, no `softmax`.** El modelo se entrenó con BCE y las 527 salidas son independientes. Aplicar softmax es un error frecuente y produce puntajes sin sentido.
- **Para fine-tuning**, tasas de $10^{-5}$ a $10^{-4}$, y verificar que `config.max_length` coincida con el número de frames que produce el extractor: si no coinciden, el modelo falla o —peor— rellena con ceros en silencio.
- **En el site:** la [Clase 39](/clases/clase-39) y su [profundización](/clases/clase-39/profundizacion) desarrollan el bloque de Transformers en audio; el fundamento [Vision Transformer](/fundamentos/vision-transformer) cubre el parcheo y el token `[CLS]` de los que AST parte; [Self-Attention](/fundamentos/self-attention) tiene la aritmética del costo cuadrático y la longitud del camino; [Transfer Learning](/fundamentos/transfer-learning) sitúa la transferencia cross-modal dentro del panorama general; y el dominio [Audio](/dominios/audio) ordena cronológicamente la línea que va de [VGGish](/papers/vggish-hershey-2017) y el [Conformer](/papers/conformer-gulati-2020) a AST, [HuBERT](/papers/hubert-hsu-2021) y los encoders multimodales de hoy. Los tres benchmarks tienen ficha propia: [AudioSet](/papers/audioset-gemmeke-2017), [ESC-50](/papers/esc50-piczak-2015) y [Speech Commands](/papers/speech-commands-warden-2018).
