---
title: "El número 19456"
weight: 1
---

El preprocesamiento del laboratorio cabe en dos funciones y produce dos árboles paralelos de arrays: uno de video, uno de audio. La decisión interesante está en una sola línea, y es un número que parece arbitrario.

```python
data = librosa.load(filename, sr=16000)[0][-19456:]
```

**19456 no es un número mágico.** Es el resultado de resolver hacia atrás la aritmética completa del stream de audio para que produzca exactamente 29 vectores — uno por cada frame de video. Toda la sincronización audiovisual del modelo está horneada en esa constante.

## La cadena que aterriza en 29

Siguiendo la propagación de longitudes a través del frontend y de la ResNet-18 unidimensional:

```
19456 muestras
  ↓ Conv1d(kernel=80, stride=4, padding=38)      ⌊(19456+76−80)/4⌋+1 = 4864
  ↓ layer1  (stride 1)                            4864
  ↓ layer2  (stride 2)                            2432
  ↓ layer3  (stride 2)                            1216
  ↓ layer4  (stride 2)                             608
  ↓ AvgPool1d(kernel=21, padding=1)              ⌊(608+2−21)/21⌋+1 = 29
```

**29.** El mismo número de frames que tiene el clip de video, y la condición sin la cual la concatenación de la celda de evaluación sería imposible.

El paper lo enuncia en una frase que ahora se lee distinto:

> *"The output of the ResNet is divided into 29 frames/windows using average pooling in order to ensure the same frame rate as the video is used."*

La sincronización no se resuelve con interpolación, ni con un módulo de alineamiento, ni con atención cruzada. **Se resuelve eligiendo el largo del recorte de audio para que la aritmética de los strides aterrice justo en 29.**

## El campo receptivo, capa por capa

Propagando campo receptivo y salto entre unidades por toda la cadena:

| Capa | Longitud | Campo receptivo | Salto entre unidades |
|---|---|---|---|
| entrada | 19456 | 0,06 ms | 0,06 ms |
| `conv1` | 4864 | **5,00 ms** | **0,25 ms** |
| `layer1` | 4864 | 7,00 ms | 0,25 ms |
| `layer2` | 2432 | 10,50 ms | 0,50 ms |
| `layer3` | 1216 | 17,50 ms | 1,00 ms |
| `layer4` | 608 | 31,50 ms | 2,00 ms |
| `avgpool` | **29** | **71,50 ms** | **42,00 ms** |

Las dos primeras cifras son el paper traducido a muestras. A 16 kHz, 80 muestras son **5 ms** y 4 muestras son **0,25 ms**:

> *"A temporal kernel of 5ms with a stride of 0.25ms is used in the first convolutional layer in order to extract fine-scale spectral information."*

Y la última fila es la que cierra todo:

```
salto entre frames de audio :  672 muestras = 42,00 ms
frame de video a 25 fps     :  640 muestras = 40,00 ms
29 × 672 = 19488  ≈  19456
```

{{< concept-alert type="clave" >}}
**El recorte de audio es 29 × 672 muestras.** Cada frame de audio avanza 42 ms; cada frame de video, 40 ms. Un 5 % de discrepancia acumulada sobre 1,2 segundos, y los dos flujos quedan alineados sin ningún mecanismo explícito de alineamiento.
{{< /concept-alert >}}

Un matiz que se ve en la tabla: el campo receptivo final es de **71,5 ms** pero el salto es de **42 ms**. Los frames de audio **se solapan**, cada uno cubriendo ~1,7 veces su propio paso. Es el mismo principio del solapamiento de ventanas del STFT que aparece en el [lab 35](/laboratorios/lab-35): sin solape, un fonema que caiga en el borde entre dos ventanas se parte en dos y ninguna lo ve completo.

## Por qué los *últimos* 19456

19456/16000 = **1,216 s**, mientras que un clip de LRW son 29 frames a 25 fps = **1,16 s** = 18.560 muestras. Se piden ~900 muestras más de las que dura el video, y se toman del final.

Es una decisión de alineamiento: la palabra objetivo está en el centro del clip, y anclar por el final deja el desfase en el *pre-roll* —el residuo de la palabra anterior— en vez de cortar la cola de la palabra objetivo. Es una heurística, no un alineamiento medido.

**Riesgo latente:** si el audio de algún MP4 tuviera menos de 19.456 muestras, el slice no falla — devuelve silenciosamente lo que haya. El error aparecería mucho después, como un `RuntimeError` de forma inválida en el `x.view(-1, 29, inputDim)` del stream de audio. En la corrida del laboratorio, los 2500 clips salieron exactos.

## El otro lado: el recorte de la boca

```python
data = extract_opencv(filename)[:, 115:211, 79:175]
```

Los frames de LRW son 256×256 (cara completa). El recorte:

| Eje | Slice | Tamaño | Centro |
|---|---|---|---|
| Filas | `115:211` | 96 px | 163 — **bajo el centro**, la zona de la boca |
| Columnas | `79:175` | 96 px | 127 — centrado |

Está centrado horizontalmente y desplazado hacia abajo, y es **el mismo para todos los videos**: no hay detección de landmarks ni tracking facial. El paper apuesta a que LRW ya viene alineado:

> *"Since the mouth ROIs are already centered, a fixed bounding box of 96 by 96 is used for all videos."*

**Esa apuesta tiene un precio medido.** La Tabla 1 compara el stream visual con ROI fija (**82,0 %**) contra el trabajo de [Stafylakis y Tzimiropoulos](/papers/lipreading-resnet-stafylakis-2017), que calcula la ROI con landmarks faciales trackeados (**83,0 %**). Es la única línea de la tabla que E2E-AVSR no logra superar, y la nota al pie del paper lo reconoce explícitamente. **Un punto entero de accuracy es lo que cuesta no trackear la boca.**

Los 96 píxeles guardados en disco frente a los 88 que consume la red dejan un margen de ±4 px: es el espacio para el *random crop* de la augmentación de entrenamiento. En evaluación se toma el centro.

## La asimetría de normalización

Las dos modalidades se normalizan de forma distinta, y no es inconsistencia:

| Modalidad | Cómo | Dónde | Estadísticos |
|---|---|---|---|
| **Audio** | z-normalización **por clip** | `MyDataset.normalisation` | media y std **de esa muestra** |
| **Video** | z-normalización **global** | `ColorNormalize` | constantes fijas: μ = 0,413621, σ = 0,1700239 |

Es el paper, sección 4.1:

> *"Audio: Each audio segment is z-normalised... **to account for variations in different levels of loudness between the speakers**."*
> *"Video: the frames are transformed to grayscale and are normalized **with respect to the overall mean and variance**."*

La lógica: el volumen absoluto de un hablante es **ruido de molestia** —no aporta nada a saber qué palabra dijo—, así que se elimina por muestra. El brillo absoluto de la cara, en cambio, **sí es señal**: normalizar cada clip por su propio brillo destruiría el contraste relativo entre labios, dientes y piel, que es exactamente lo que el modelo lee.

Las constantes son además una firma del dataset: μ ≈ 105/255 y σ ≈ 43/255 en niveles de gris — una boca de estudio de televisión, bien iluminada y sin extremos. Aplicarlas a video de otra fuente sería un desajuste de dominio silencioso.

## El dataset que resultó

El mini test set distribuido por los profesores contiene **2500 clips**, que resultaron ser **500 palabras × 5 clips exactos** — el vocabulario completo de LRW a un décimo de densidad. No es un subconjunto de clases: es la tarea completa, evaluada sobre menos muestras.

```
video: (29, 96, 96, 3) uint8   ·   audio: (19456,) float32
clips de video con != 29 frames:      0
clips de audio con != 19456 muestras: 0
```

Eso hace que la accuracy del laboratorio sea **directamente comparable** al 98,0 % del paper sobre los 25.000 clips del test set completo. Con la salvedad de que son los 5 *primeros* clips de cada palabra, no 5 sorteados — un detalle que resulta importante al leer el resultado.

---

**Siguiente:** [Los dos streams](02-los-dos-streams) · **Volver al** [índice del lab](/laboratorios/lab-43)
