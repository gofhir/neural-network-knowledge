---
title: "Los dos streams"
weight: 2
---

El modelo son tres redes que se cargan por separado y se conectan en la última celda: un stream de audio, uno de video y un BiGRU de fusión. Entre las tres suman **54.633.500 parámetros**, repartidos de una forma que no es la esperable.

| Componente | Parámetros | Qué es |
|---|---|---|
| Stream de audio | 12.500.340 | ResNet-18 1D + BiGRU |
| Stream de video | 29.025.460 | Conv3D + ResNet-34 2D + BiGRU |
| **BiGRU de fusión** | **13.107.700** | dos capas recurrentes |

**El módulo de fusión —"solo" dos capas recurrentes— pesa más que todo el stream de audio con su ResNet de 18 capas.** Su entrada de 2048 dimensiones multiplicada por 3 compuertas × 512 unidades × 2 direcciones domina el conteo: la primera capa sola tiene 7,9 M de parámetros. Las recurrentes son caras cuando la entrada es ancha.

## El bloque GRU y sus tres roles

Una sola clase se instancia tres veces, con configuraciones que encajan como piezas:

| Instancia | `input_size` | `hidden` | `output_layer` | Salida |
|---|---|---|---|---|
| GRU del stream de audio | 512 | 512 | `False` | `(B, 29, 1024)` |
| GRU del stream de video | 256 | 512 | `False` | `(B, 29, 1024)` |
| **`concat_model`** | **2048** | 512 | `True`, `every_frame=True` | `(B, 29, 500)` |

Los dos streams entregan 1024 cada uno —512 unidades ocultas × 2 direcciones—, se concatenan a 2048, que es exactamente la entrada del fusor.

Sobre el ancho: el paper dice *"a 2-layer BGRU which consists of **1024 cells** in each layer"* y el código usa `hidden_size=512`. **No hay contradicción**: son 512 por dirección, 1024 en total. Es la ambigüedad clásica al leer papers de recurrentes bidireccionales.

**La bidireccionalidad no es un lujo aquí.** Exige conocer el futuro, y es legítima porque LRW es clasificación *offline* de un clip pre-segmentado: los 29 frames están en memoria antes de empezar. En lectura de labios el contexto futuro es además decisivo, porque la forma de la boca al pronunciar una vocal está fuertemente coarticulada con la consonante que viene después. Un modelo unidireccional ve la boca en el frame 10 sin saber hacia dónde va. En un sistema de subtitulado en vivo, nada de esto sería posible.

**Con `output_layer=False`, el `self.fc` se construye pero nunca se ejecuta.** Existe en el `state_dict`, ocupa espacio y no participa del cómputo: es un vestigio del entrenamiento por fases, cuando cada stream necesitaba su propia cabeza de clasificación. Los checkpoints de audio y video contienen clasificadores fantasma — y eso resulta ser apenas la punta de lo que guardan, como se ve en [la arqueología del checkpoint](03-la-arqueologia-del-checkpoint).

## El stream de audio: aprender el banco de filtros

La cadena completa, con entrada `(B, 19456)`:

| Paso | Operación | Salida |
|---|---|---|
| `view` | agregar canal | `(1, 19456)` |
| `frontend1D` | Conv1d(1→64, k=80, s=4) + BN + ReLU | `(64, 4864)` |
| `layer1`–`layer4` | 8 bloques residuales, 64→512 | `(512, 608)` |
| `avgpool` | AvgPool1d(k=21) | **`(512, 29)`** |
| `fc` | Linear(512 → 512) por frame | `(B·29, 512)` |
| `gru` | BiGRU 2 capas | `(B, 29, 1024)` |

Ocho bloques × 2 convoluciones, más el frontend y la `fc` = **18 capas con pesos**.

El `fc` se construye con `num_classes=inputDim`: el nombre es herencia de ImageNet, **aquí no clasifica nada** — es una proyección lineal de features 512→512.

Lo sustantivo está en el frontend, que es lo que ocupa el lugar del pipeline de MFCC:

| | MFCC clásico | `frontend1D` |
|---|---|---|
| Ventana | 40 ms (fija, Hamming) | **5 ms** (aprendida) |
| Paso | 10 ms | **0,25 ms** |
| Banco de filtros | 40 filtros mel, **diseñados a mano** | **64 filtros aprendidos por gradiente** |
| Salida | 13 coeficientes + deltas | 64 canales |

La escala mel es una decisión de 1937 sobre cómo percibe el oído humano, congelada en el pipeline. El frontend no asume nada: aprende sus filtros de 5 ms desde el objetivo de clasificación. El resultado del paper es un empate — **97,7 % contra 97,7 %** — reportado con una honestidad que conviene citar:

> *"we should note that the effort required in order to train the end-to-end system is significantly higher than the 2-layer BGRU used with MFCCs."*

Empatar no suena a victoria, pero el punto es otro: los MFCC dejan de ser un piso obligatorio, y lo aprendido es adaptable al dominio. Es el mismo argumento de [SoundNet](/papers/soundnet-aytar-2016), el otro paper de la clase, que también consume onda cruda con convoluciones 1D.

**Dos detalles silenciosos del `AvgPool1d(kernel_size=21, padding=1)`:** el `stride` no se especifica, y su valor por defecto es `kernel_size` — con `stride=1` la salida serían 590 frames y el reshape reventaría. Y el `padding=1` entra en el promedio (`count_include_pad=True` por defecto), así que la primera y la última ventana promedian un cero artificial. El padding existe solo para que la división dé 29 exacto; el sesgo en los bordes es daño colateral aceptado.

## El stream de video: 3D abajo, 2D arriba

```python
nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
```

Leyendo el kernel como **(tiempo, alto, ancho)**:

| Eje | Kernel | Stride | Padding | Transformación |
|---|---|---|---|---|
| **Tiempo** | 5 | **1** | **2** | **29 → 29** |
| Alto / Ancho | 7 | 2 | 3 | 88 → 44 |

Con stride temporal 1 y padding 2, entran 29 frames y salen 29. El `MaxPool3d` que sigue tiene kernel y stride temporal 1: tampoco toca el tiempo. **La reducción es exclusivamente espacial.**

Lo que sí hace es dar a cada unidad de salida un campo receptivo de **5 frames consecutivos** = **200 ms** a 25 fps, aproximadamente la duración de una sílaba. La capa no aprende cómo *se ve* una boca; aprende cómo **se mueve** una boca en 200 ms.

La diferencia absoluta entre frames consecutivos hace visible qué información hay ahí:

![Diferencias absolutas entre frames consecutivos del clip, mostrando el movimiento concentrado en los labios](/laboratorios/lab-43/movimiento.jpg)

Toda la energía se concentra en **el contorno de los labios y el borde de los dientes**; las mejillas y la barbilla quedan casi negras. Eso es lo que la Conv3D recibe: no una boca, sino la derivada temporal de una boca. Y su kernel de 5 frames abarca cinco de estos paneles a la vez.

> *"A spatiotemporal convolutional layer is capable of capturing the short-term dynamics of the mouth region and is proven to be advantageous, **even when recurrent networks are deployed for back-end**."*

Ese "even when" es el argumento: no basta con poner un BiGRU al final. La recurrente modela la dinámica de la palabra completa, pero opera sobre vectores de 256 números por frame — cuando la información del movimiento rápido ya se perdió. Hay que capturarla **antes**, mientras todavía hay píxeles.

### El reshape que revela la arquitectura real

```python
x = self.frontend3D(x)                       # (B, 64, 29, 22, 22)
x = x.transpose(1, 2).contiguous()
x = x.view(-1, 64, x.size(3), x.size(4))     # (B*29, 64, 22, 22)  ← tiempo → batch
x = self.resnet34(x)
x = x.view(-1, self.frameLen, self.inputDim) # (B, 29, 256)
x = self.gru(x)
```

Ese `view` **colapsa el eje temporal dentro del batch**. Consecuencia: **la ResNet-34 procesa los 29 frames de forma completamente independiente. Nunca ve más de un frame a la vez.**

```
Conv3D (5 frames) → ResNet-34 puramente espacial, aplicada 29 veces → BiGRU (29 frames)
   ↑ dinámica corta        ↑ apariencia, sin tiempo                      ↑ dinámica larga
      ~200 ms                                                               ~1,16 s
```

Es un contraste directo con lo que se ve en la [clase 38](/clases/clase-38): C3D e I3D hacen convolución 3D en *toda* la red. Aquí el 3D vive solo en la primera capa. Ese patrón tiene nombre y validación empírica — es la familia **MC (mixed convolution)** de Tran et al. (2018), *A Closer Look at Spatiotemporal Convolutions for Action Recognition*, cuyo hallazgo es que la convolución temporal rinde donde hay movimiento crudo y se vuelve redundante en las capas altas, donde las features ya son abstractas. El costo también decide: una ResNet-34 íntegramente 3D tendría ~3× los parámetros.

### El AvgPool2d que descarta el 56 % del mapa

```python
self.avgpool = nn.AvgPool2d(2)
```

El mapa que llega a esta capa mide **3×3**. Un pooling de kernel 2 y stride 2 sobre una entrada de 3 produce ⌊(3−2)/2⌋+1 = **1**, correcto — pero la ventana cubre solo `[0:2, 0:2]`. **La última fila y la última columna nunca se leen: de las 9 posiciones espaciales del mapa final, se usan 4.**

La forma canónica sería `nn.AdaptiveAvgPool2d(1)`, que promedia las 9 sin depender de que la aritmética calce; es lo que usa torchvision en todas sus ResNets.

{{< concept-alert type="cuidado" >}}
**Este defecto está congelado en los pesos.** Los checkpoints se entrenaron con esta capa, así que la red aprendió a concentrar información en el cuadrante que sí contribuía al gradiente. Cambiarlo ahora a `AdaptiveAvgPool2d(1)` **empeoraría** el resultado, porque metería en el promedio cinco posiciones que el modelo nunca usó. No es un bug que se pueda arreglar sin reentrenar.
{{< /concept-alert >}}

Y como el recorte de la boca no está centrado verticalmente en el frame, ese cuadrante corresponde aproximadamente al labio superior y la comisura izquierda. Un sesgo espacial arbitrario.

## Dos discrepancias con el paper

**La ResNet es v1, no v2.** El paper dice usar *"the 34-layer **identity mapping** version"*, que es la ResNet v2 de He et al. (2016) con **pre-activación**: `BN → ReLU → Conv → BN → ReLU → Conv`, con la suma limpia al final y el gradiente fluyendo por la identidad sin atravesar ninguna no linealidad. El código implementa **post-activación** — `Conv → BN → ReLU → Conv → BN`, suma, y ReLU *después* —, que es la v1 de 2015. Funciona y los checkpoints corresponden a este código, pero el modelo del laboratorio no es exactamente el que el texto describe.

**La asimetría 512 / 256 no está justificada.** El stream de audio comprime cada frame a 512 números y el de video a 256. El paper solo dice que ambos BiGRU tienen 1024 celdas; la diferencia viene de los dos repositorios de origen. Lo notable es la dirección: **la modalidad de menor dimensionalidad es la que ya tenía menos información útil** — un frame de boca en gris de 88×88 acaba en 256 números, mientras 42 ms de onda acaban en 512. Consistente con que el video acierte 82 % y el audio 97,7 %.

## La fusión, en una línea

```python
inputs = torch.cat((audio_outputs, video_outputs), dim=2)   # (1, 29, 2048)
outputs = concat_model(inputs)                              # (1, 29, 500)
outputs = torch.mean(outputs, 1)                            # (1, 500)
```

`dim=2` es el eje de features, no el temporal: para cada uno de los 29 instantes se pegan los 1024 números del audio con los 1024 del video. Es **fusión tardía a nivel de features**, y solo es posible porque ambos streams entregan exactamente 29 frames — el resultado del [19456](01-el-numero-19456).

Lo que **no** hay: ni pesos de modalidad, ni atención cruzada, ni *gating* explícito. La decisión de cuánto confiar en cada modalidad no se programa — **la aprenden las compuertas del BiGRU de fusión**. Ese es el mecanismo detrás de los +14,1 puntos a −5 dB: cuando el audio es ruido, las compuertas aprenden a dejarlo pasar menos.

### El promedio que el comentario describe mal

El comentario del notebook dice `# average probability among frames`, y no es lo que el código hace. Se promedian **logits**, y el softmax se aplica **después**:

$$\text{softmax}\!\left(\frac{1}{T}\sum_t z_t\right) \neq \frac{1}{T}\sum_t \text{softmax}(z_t)$$

La primera es una **media geométrica** normalizada de las distribuciones — un *product of experts* — y es mucho más severa: un solo frame que asigne probabilidad casi nula a una clase la elimina de la competencia aunque los otros 28 la favorezcan. La media aritmética perdona ese caso. En la práctica el argmax rara vez cambia, pero el comentario describe el promedio que no se está calculando.

Como remate, ese `F.softmax` es **computacionalmente inútil**: softmax es monótona creciente, así que `torch.max` sobre los logits daría idéntico argmax.

---

**Anterior:** [El número 19456](01-el-numero-19456) · **Siguiente:** [La arqueología del checkpoint](03-la-arqueologia-del-checkpoint)
