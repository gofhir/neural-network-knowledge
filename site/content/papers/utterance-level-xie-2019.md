---
title: "Utterance-level Aggregation for Speaker Recognition (2019)"
weight: 445
math: true
---

{{< paper-card
    title="Utterance-level Aggregation for Speaker Recognition in the Wild"
    authors="Weidi Xie, Arsha Nagrani, Joon Son Chung, Andrew Zisserman (VGG, University of Oxford; Naver)"
    year="2019"
    venue="ICASSP 2019 / arXiv:1902.10107"
    pdf="/papers/utterance-level-xie-2019.pdf" >}}
El modelo que la [Clase 41](/clases/clase-41) presenta como respuesta al reconocimiento de hablante, y un experimento que aísla una variable con una limpieza poco habitual. La arquitectura combina un **thin ResNet-34** —una ResNet a la que se le recortan los canales, de 3 millones de parámetros contra los 22 de la estándar— con una capa **[NetVLAD](/papers/netvlad-arandjelovic-2016)** o **GhostVLAD** que agrega los descriptores de frame en un único vector de enunciado, todo entrenable de punta a punta sobre [VoxCeleb2](/papers/voxceleb2-chung-2018). El resultado que ordena el paper es la comparación interna: **con el mismo backbone y los mismos datos**, cambiar la agregación de promedio temporal a NetVLAD lleva el EER de **10,48 % a 3,57 %**. La agregación no es un detalle de implementación al final del pipeline — es el componente que decide el rendimiento.
{{< /paper-card >}}

---

## Contexto: el reconocimiento de hablante necesita lo contrario que el de voz

El paper abre con una distinción que vale la pena tener presente porque organiza todo lo demás:

> *"For speaker recognition, the goal is to condense information into a single utterance-level representation, unlike speech recognition where frame-level representations are desired."*

Reconocer **qué se dijo** requiere resolución temporal: cada fonema en su lugar. Reconocer **quién lo dijo** requiere lo opuesto: colapsar todo el enunciado en un vector, descartando el contenido lingüístico y quedándose con lo invariante de la voz. Son objetivos en tensión, y explica por qué la misma arquitectura no sirve para ambos — el punto con el que la [Clase 41](/clases/clase-41) abre su primera parte.

Sobre habla "in the wild" la dificultad se agrava: los enunciados tienen duración variable y contienen **partes irrelevantes** —silencios, ruido, otras voces— que deben filtrarse. Los sistemas previos agregaban con promedio temporal, heredado de la visión, o con capas totalmente conectadas. El problema de ambos, según los autores, es que la agregación **no depende del contenido**: no puede decidir qué partes de la señal importan.

## Método

**Frame level: thin ResNet-34.** Sobre espectrogramas de 257 × T, una ResNet-34 con los canales recortados en cada bloque residual: 3 millones de parámetros contra los 22 de la estándar. La red es totalmente convolucional, así que acepta entradas de longitud arbitraria y produce descriptores de tamaño $1 \times T/32 \times 512$.

**Utterance level: NetVLAD / GhostVLAD.** La capa agrega los $T/32$ descriptores en una matriz $K \times D$:

$$V(k,j) = \sum_{t=1}^{T/32} \frac{e^{\,w_k^\top x_t + b_k}}{\sum_{k'} e^{\,w_{k'}^\top x_t + b_{k'}}}\,\big(x_t(j) - c_k(j)\big)$$

con $\{w_k\}$, $\{b_k\}$ y $\{c_k\}$ entrenables. Después: normalización L2, concatenación y una capa totalmente conectada que reduce a **512 dimensiones**.

**GhostVLAD** agrega clusters *fantasma* —dos, en su implementación— cuyos residuos **se descartan** de la concatenación final. Funcionan como sumidero: los frames ruidosos o irrelevantes reciben la mayor parte de su peso de asignación ahí y su contribución a los clusters reales queda atenuada. Es una forma de descartar información sin decidir explícitamente qué descartar.

**Entrenamiento.** Segmentos fijos de **2,5 s** recortados al azar (ventana Hamming de 25 ms, paso 10 ms, FFT de 512 puntos → 257 × 250), normalizados por frecuencia. **Sin detección de actividad de voz ni eliminación de silencios** — el modelo debe aprender a ignorarlos. Adam con lr inicial de 1e−3, dividido por 10 cada 36 épocas. Además de softmax se prueba **AM-Softmax** (margen angular aditivo, $m = 0{,}4$, $s = 30$).

**Evaluación.** Entrenamiento en VoxCeleb2-dev (5 994 hablantes), prueba en [VoxCeleb1](/papers/voxceleb-nagrani-2017), conjuntos **disjuntos**. La comparación entre dos enunciados es el producto punto de sus descriptores; el umbral se elige sobre la curva ROC y se reporta el **EER**.

## Resultados

La tabla completa sobre el test original de VoxCeleb1:

| Front-end | Pérdida | Agregación | Entrenamiento | EER |
|---|---|---|---|---|
| i-vectors + PLDA | — | — | VoxCeleb1 | 8,80 % |
| VGG-M | Softmax | TAP | VoxCeleb1 | 10,20 % |
| ResNet-34 ([Cai et al.](/papers/x-vectors-snyder-2018)) | A-Softmax + PLDA | LDE | VoxCeleb1 | 4,48 % |
| TDNN (x-vector) | Softmax | ASP | VoxCeleb1 | 3,85 % |
| ResNet-50 ([Chung et al.](/papers/voxceleb2-chung-2018)) | Softmax + Contrastive | TAP | VoxCeleb2 | 4,19 % |
| **Thin ResNet-34 (este)** | Softmax | **TAP** | VoxCeleb2 | **10,48 %** |
| **Thin ResNet-34 (este)** | Softmax | **NetVLAD** | VoxCeleb2 | **3,57 %** |
| Thin ResNet-34 (este) | AM-Softmax | NetVLAD | VoxCeleb2 | 3,32 % |
| **Thin ResNet-34 (este)** | Softmax | **GhostVLAD** | VoxCeleb2 | **3,22 %** |

{{< concept-alert type="clave" >}}
**Las dos filas destacadas comparten backbone, datos, pérdida y protocolo. Lo único que cambia es cómo se agregan los frames, y el EER pasa de 10,48 % a 3,57 %.** Es una diferencia de un factor de 3 producida por el componente que suele considerarse trivial.

La explicación que ofrecen los autores es precisa: las features obtenidas con promedio temporal *"son buenas optimizando la diferencia inter-clase (separar hablantes distintos) pero no reduciendo la variación intra-clase (hacer compactas las features de un mismo hablante)"*. Por eso Chung et al. necesitan una pérdida contrastiva con minería de ejemplos difíciles para que TAP funcione — y por eso llegan a 4,19 % con 26 millones de parámetros donde este modelo llega a 3,22 % con 10.
{{< /concept-alert >}}

En los conjuntos más difíciles la ventaja crece: **VoxCeleb1-E** 3,24 % contra 4,42 %, y **VoxCeleb1-H** —pares del mismo género y nacionalidad— **5,17 % contra 7,33 %**. Cuanto más se eliminan los atajos, más importa la calidad del descriptor.

El barrido de clusters muestra que el método es **robusto**: entre 8 y 14 clusters (más 2 fantasma) el EER se mueve entre 3,22 % y 3,37 %, con las dos pérdidas.

### El efecto de la duración

| Duración del segmento | 2 s | 3 s | 4 s | 5 s | 6 s |
|---|---|---|---|---|---|
| EER | 7,97 % | 5,73 % | 4,70 % | 4,10 % | **3,39 %** |

Más del doble de error con 2 segundos que con 6. La explicación es la misma que motiva GhostVLAD: en habla "in the wild" una porción del audio es ruido, silencio o voces ajenas, y **un segmento corto puede tener mala suerte** y contener sobre todo eso. Al alargarlo, la probabilidad de capturar voz útil del hablante crece.

## Limitaciones

- **Los conjuntos VoxCeleb comparten sesgo**: celebridades entrevistadas. La transferencia a telefonía, habla espontánea o entornos verdaderamente adversos no está evaluada.
- **La longitud importa mucho**, y el paper lo documenta pero no lo resuelve: por debajo de 4 segundos el rendimiento se degrada rápido.
- **GhostVLAD aporta poco sobre NetVLAD** en el test estándar (3,22 % contra 3,57 % con softmax; con AM-Softmax quedan empatados). Su beneficio principal es conceptual.
- **La dimensión intermedia es grande** ($K \times D$ antes de reducir), lo que encarece el entrenamiento.
- **La pérdida con margen angular ayuda con NetVLAD pero no con GhostVLAD**, y el paper no explica por qué.

## Por qué importa para la Clase 41

Es el paper de la segunda mitad de la clase, y la fuente de todo lo que aparece en sus diapositivas: el diagrama de cinco etapas (*Encoding → Projection → Centroid Ownership → Feature Aggregation → Final representation*), el thin ResNet, la fórmula del soft assignment, la cadena `DimReduction → ReLU → L2_norm` que garantiza que el producto punto caiga en $[0,1]$, y el protocolo VoxCeleb2 → VoxCeleb1.

Lo que la clase no destaca —y es lo más instructivo— es la **comparación interna TAP contra NetVLAD**. Presentada así, VLAD parece una elección arquitectónica entre varias razonables. La tabla muestra que es la elección que decide el resultado: el mismo backbone, con el mismo entrenamiento, rinde 10,48 % o 3,57 % según cómo se resuman los frames.

Es también el ejemplo más nítido de un patrón que atraviesa el curso: **una técnica de un dominio trasplantada a otro**. VLAD nació en 2010 para [búsqueda de imágenes](/papers/vlad-jegou-2010), se volvió diferenciable en 2016 para [reconocimiento de lugares](/papers/netvlad-arandjelovic-2016), y en 2019 resulta ser lo que le faltaba al reconocimiento de hablante. La estructura del problema —un conjunto de descriptores locales de cardinalidad variable que hay que resumir en un vector fijo— es la misma; el dominio es incidental.

---

## Tres cosas que el paper dice y su propio código desmiente

El [Lab 41](/laboratorios/lab-41) reimplementa este modelo, carga el checkpoint oficial y **reproduce el resultado**: 3,19 % de EER contra el 3,22 % publicado, sobre la misma lista de 37.720 pares. Con el número replicado, tres afirmaciones del paper resultan verificables — y las tres fallan.

**1. La resolución temporal no es `T/32`, es `T/16`.** La Tabla 1 declara `max pool 3×1, stride (2,2)` → `7 × T/32 × 512`, y el cuerpo del texto escribe `R^{1×T/32×512}` e indexa la suma de VLAD hasta `T/32`. Pero el código oficial de los autores dice:

```python
y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)   # backbone.py del repo
```

`strides=(2,1)`: 2 en frecuencia, **1 en tiempo**. Verificado ejecutando el modelo con tres largos de entrada — la salida es siempre `T/16`. Cada descriptor cubre un paso de **160 ms**, no de 320. El error es invisible porque `MaxPool` no tiene parámetros: cargar los pesos funciona igual con cualquier stride. Y es numéricamente inconsecuente (coseno 0,9993 entre ambas variantes, porque la intra-normalización absorbe el cambio de escala), pero cambia cuál es la unidad de análisis.

**2. La ventana no es Hamming.** El paper dice *"using a hamming window of width 25ms"*; el código llama a `librosa.stft` sin especificar ventana, y el valor por defecto es **Hann**.

**3. El eje de normalización contradice a VoxCeleb.** Este paper normaliza *"by subtracting the mean and dividing by the standard deviation of all frequency components in a single time step"* — es decir, **por frame**. Dos años antes, el mismo grupo escribía en [VoxCeleb](/papers/voxceleb-nagrani-2017): *"Mean and variance normalisation is performed on **every frequency bin** of the spectrum. **This normalisation is crucial**, leading to an almost 10% increase in classification accuracy."* Es el eje contrario, y ninguno de los dos papers menciona el cambio.

La consecuencia práctica de haber invertido el eje: normalizar por frame **elimina toda la información de energía** y **amplifica el silencio 37×** hasta ponerlo en la misma escala que la voz. Lo cual, combinado con que el paper declara explícitamente que *"no voice activity detection (VAD), or automatic silence removal is applied"*, explica por qué [GhostVLAD](/papers/ghostvlad-zhong-2018) aporta 0,35 puntos: los clusters fantasma son el VAD que el preprocesamiento no hace.

### Y lo que hay dentro del checkpoint

| Lo que dice el paper | Lo que hay en los 46 MB de pesos |
|---|---|
| *"trainable discriminative clustering"*, 8 centroides | 8 centroides con **coseno 0,9983** entre sí: el mismo vector |
| Thin ResNet-34 de 3 M de parámetros | **50,2 %** de los canales con filtros encogidos a 10⁻³³ (70,9 % en `block5`) |
| Entrenado en VoxCeleb2, **5.994** hablantes | la capa de clasificación tiene **8.631** salidas (las identidades de VGGFace2) |

Ninguna de las tres impide que el modelo funcione. Pero el «clustering discriminativo» degeneró en *attention pooling* de 8 cabezas, el «thin» ResNet es la mitad de lo que declara, y la cabeza de clasificación —que además nunca se ejecuta en inferencia— arrastra un default heredado de un proyecto de caras. **El resultado es reproducible; la explicación de por qué funciona no describe a este modelo.**
