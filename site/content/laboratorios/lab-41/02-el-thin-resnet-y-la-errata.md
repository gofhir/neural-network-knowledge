---
title: "02 - El Thin ResNet, la errata y el campo receptivo"
weight: 20
math: true
---

> Las 34 capas contadas una por una, una errata en la Tabla 1 de un paper de ICASSP que el propio código de sus autores desmiente, y la medición que cambia lo que significa «descriptor local»: 1,84 segundos, no 160 milisegundos.

---

## 1. Por qué «Thin», con números

El paper es explícito sobre qué modificó:

> *"Compared to the standard ResNet used before by [3], we cut down the number of channels in each residual block, making it a thin ResNet-34. (…) This architecture has only 3 million parameters compared to the standard ResNet-34 (22 million)."*
> — [Xie et al. 2019](/papers/utterance-level-xie-2019)

Verificadas las dos mitades:

| | Parámetros |
|---|---|
| ThinResNet del lab (medido) | **3.690.240** |
| ResNet-34 estándar, backbone sin fc (calculado) | **21.284.672** |
| razón | **5,8×** |

El paper redondea a «3 millones contra 22 millones». Cuadra.

**Y el «34» es literal.** Contando las capas convolucionales del camino principal:

| Bloque | convs camino principal | + shortcut 1×1 | params | % |
|---|---|---|---|---|
| block1 | 1 | 0 | 3.264 | 0,1 % |
| block2 | 6 | 1 | 65.472 | 1,8 % |
| block3 | 9 | 1 | 333.952 | 9,0 % |
| block4 | 9 | 1 | 658.944 | 17,9 % |
| block5 | 9 | 1 | **2.628.608** | **71,2 %** |
| **total** | **34** | 4 | **3.690.240** | |

Treinta y cuatro exactas. Con un detalle que el nombre esconde: **los bloques son de tipo `bottleneck` (1×1 → 3×3 → 1×1), no `BasicBlock` (3×3 → 3×3)**. El ResNet-34 canónico de [He et al.](/papers/resnet-he-2015) usa BasicBlocks con distribución (3,4,6,3); este usa bottlenecks con distribución (2,3,3,3). **Coincide en profundidad con el 34 y en tipo de bloque con el 50**, y usa una fracción de los canales de ambos.

Y el reparto de costo sigue el patrón universal de las ResNets: **el 71 % del backbone vive en `block5`**, el bloque de 512 canales. La profundidad es barata; el ancho es caro.

---

## 2. El bottleneck, y el stride que hay que poner donde estaba

```python
class IdentityBlock2D( nn.Module ):
  def __init__( self, input_channels, kernel_size, filters ):
    filters1, filters2, filters3 = filters
    self.branch1 = nn.Sequential(
        nn.Conv2d( input_channels, filters1, (1, 1), bias = False ),
        nn.BatchNorm2d( filters1 ), nn.ReLU(),
        nn.Conv2d( filters1, filters2, kernel_size, padding = (1, 1), bias = False ),
        nn.BatchNorm2d( filters2 ), nn.ReLU(),
        nn.Conv2d( filters2, filters3, (1, 1), bias = False ),
        nn.BatchNorm2d( filters3 ) )
    self.relu = nn.ReLU()
  def forward( self, x ):
    x = self.branch1( x ) + x
    x = self.relu( x )
    return x
```

**El porqué del bottleneck es económico.** Para 512 canales de entrada y `filters = [256, 256, 512]`: dos convoluciones 3×3 directas de 512→512 costarían `2 × 512 × 512 × 9 = 4,7 M` parámetros. El bottleneck cuesta `512·256 + 256·256·9 + 256·512 = 852 K`. **5,5× más barato**, y la única convolución que mira el vecindario opera en un espacio comprimido a la mitad. Las 1×1 no miran vecinos: son mezclas lineales entre canales, píxel a píxel.

**`bias = False` en las tres convoluciones.** No es descuido: cada una va seguida de `BatchNorm2d`, que resta la media y aporta su propio `β`. Un bias sería **matemáticamente redundante**. La regla se aplica con consistencia en todo el modelo: las dos únicas convoluciones *con* bias (`block_1` y `vlad_conv`) son precisamente las que **no** llevan BatchNorm detrás.

### El stride en la 1×1: la trampa del port

```python
class ConvBlock2D( nn.Module ):
  def __init__( self, input_channels, kernel_size, filters, strides = (2, 2) ):
    self.branch1 = nn.Sequential(
        nn.Conv2d( input_channels, filters1, (1, 1), stride = strides, bias = False ),  # ← acá
        ... )
    self.branch2 = nn.Sequential(          # shortcut proyectivo
        nn.Conv2d( input_channels, filters3, (1, 1), stride = strides, bias = False ),
        nn.BatchNorm2d( filters3 ) )
```

| Variante | Dónde va el stride 2 | Quién la usa |
|---|---|---|
| **ResNet v1 «original»** | en la conv **1×1** | He et al. 2015, Keras Applications, **este lab** |
| ResNet v1.5 | en la conv **3×3** | torchvision, NVIDIA, casi todo PyTorch moderno |

Poner el stride en la 1×1 **descarta información**: lee un píxel de cada dos y nunca mira los que salta. La v1.5 lo mueve a la 3×3, que promedia el vecindario antes de submuestrear, y eso vale ~0,5 puntos de top-1 en ImageNet — razón por la que torchvision la adoptó.

**El lab usa la variante original, y hace bien**, porque los pesos vienen de Keras. Y aquí está el peligro: si alguien «arreglara» esto moviendo el stride a la 3×3, **`load_state_dict` seguiría funcionando sin error** (las formas no cambian) pero el modelo daría resultados sin sentido. Es un bug silencioso perfecto.

### El port de Keras, verificado

Comparado contra `backbone.py` del repo oficial, la estructura es **idéntica** capa por capa: mismo orden (1×1 reduce → BN → ReLU → 3×3 `same` → BN → ReLU → 1×1 increase → BN → add → ReLU), mismo stride en la 1×1, mismo `use_bias=False`, mismo shortcut `1x1_proj` con BN, misma post-activación. El `padding=(1,1)` de PyTorch es el equivalente estricto del `padding='same'` de Keras para kernel 3 con stride 1.

Hay **una** diferencia numérica, y está en un valor por defecto que ninguno de los dos especifica:

```python
# Keras (original):  BatchNormalization(axis=bn_axis, ...)   → epsilon = 1e-3
# PyTorch (el lab):  nn.BatchNorm2d( filters1 )              → eps     = 1e-5
```

Con **38 BatchNorms** encadenadas (34 en el camino principal + 4 en los shortcuts), el efecto se acumula. Medido con los pesos reales del checkpoint:

| | resultado |
|---|---|
| coseno( emb[eps=1e-5] , emb[eps=1e-3] ) | **0,999188** |
| error relativo medio | **3,74 %** |

Es pequeño pero **no es cero, y es sistemático**. El desglose de por qué está en [El checkpoint abierto](04-el-checkpoint-abierto): más de la mitad de los canales son inmunes al eps (numerador exactamente cero) y el efecto viene concentrado en 170 canales vivos con varianza menor que 1e-3.

> Las otras dos diferencias del original —`kernel_initializer='orthogonal'` y `kernel_regularizer=l2(weight_decay)`— solo actúan durante el entrenamiento. Pero ninguna es inocua: la inicialización ortogonal explica la norma **exactamente 1,000** de los centroides fantasma, y el weight decay explica que la mitad del backbone haya quedado encogida a 10⁻³³.

---

## 3. La errata: `T/32` contra `T/16`

Esta es la Tabla 1 del paper, reproducida en el notebook como imagen:

| Módulo | Operación | Salida **declarada** |
|---|---|---|
| entrada | espectrograma | `257 × T × 1` |
| block1 | conv2d 7×7, 64 | `257 × T × 64` |
| block1 | max pool 2×2, stride (2,2) | `128 × T/2 × 64` |
| block2 | [1×1,48 / 3×3,48 / 1×1,96] × 2 | `128 × T/2 × 96` |
| block3 | [1×1,96 / 3×3,96 / 1×1,128] × 3 | `64 × T/4 × 128` |
| block4 | [1×1,128 / 3×3,128 / 1×1,256] × 3 | `32 × T/8 × 256` |
| block5 | [1×1,256 / 3×3,256 / 1×1,512] × 3 | `16 × T/16 × 512` |
| — | **max pool 3×1, stride (2,2)** | **`7 × T/32 × 512`** ← |
| — | conv2d 7×1, 512 | **`1 × T/32 × 512`** ← |

Internamente es consistente: con stride 2 en el eje temporal, `T/16 → T/32`. Y el cuerpo del paper lo repite: *"maps the input spectrogram (R^{257×T×1}) to frame-level descriptors with size R^{1×T/32×512}"*, e indexa la suma de VLAD hasta `T/32`.

**Pero el código del lab dice:**

```python
self.max_pool = nn.MaxPool2d( (3, 1), stride = (2, 1) )
```

`stride=(2,1)`: **2 en frecuencia, 1 en tiempo.** El eje temporal no se toca.

**¿Quién tiene razón?** El código oficial de los propios autores:

```python
y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
```

**`strides=(2,1)`.** El lab es fiel al repo; **la errata está en el paper** — en la Tabla 1 y en el cuerpo del texto.

Verificado ejecutando el modelo con tres largos de entrada:

| Entrada | block5 | tras `max_pool` | ¿T/32 o T/16? |
|---|---|---|---|
| `(1,1,257,250)` | `(1,512,16,16)` | `(1,512,7,16)` | 250/16 ≈ **16 = T/16** |
| `(1,1,257,800)` | `(1,512,16,50)` | `(1,512,7,50)` | 800/16 = **50 = T/16** |
| `(1,1,257,1024)` | `(1,512,16,64)` | `(1,512,7,64)` | 1024/16 = **64 = T/16** |

**La resolución temporal real es T/16.** Y lo que hace el error invisible es que `MaxPool2d` **no tiene parámetros**: cargar el checkpoint funciona igual con cualquier stride, así que nada revienta y nadie lo nota.

### ¿Cuánto importa?

Poniendo el stride de la Tabla 1 y comparando embeddings con los pesos reales:

| | coseno( T/16 , T/32 ) |
|---|---|
| voz A | **0,999279** |
| voz B | **0,999491** |
| descriptores a VLAD | **103** vs **52** |

**Duplicar la cantidad de descriptores cambia el embedding en el tercer decimal**, por la razón que se verifica en [NetVLAD desarmado](03-netvlad-desarmado): la intra-normalización vuelve a VLAD casi invariante al número de descriptores.

{{< concept-alert type="clave" >}}
**La errata es numéricamente inconsecuente y conceptualmente relevante.** Nadie que reprodujera el modelo con `stride=(2,2)` lo notaría en el EER. Pero cambia cuál es tu unidad de análisis: con `T/16` el paso entre descriptores es de **160 ms**; con `T/32`, de 320. Para un clip típico de VoxCeleb1-test (8,2 s, duplicado a 16,4 por el espejado) son **~102 descriptores, no ~51**.

La cadena de propagación del error es: paper → comentario del código del lab (`# output: 7 x T/32 x 512`) → el lector.
{{< /concept-alert >}}

---

## 4. La asimetría del backbone: la frecuencia muere, el tiempo sobrevive

Mirando la columna de salidas de la tabla:

- **frecuencia:** `257 → 128 → 64 → 32 → 16 → 7 → 1` — se colapsa por completo
- **tiempo:** `T → T/2 → T/4 → T/8 → T/16` — se reduce, pero **sobrevive**

Esa asimetría es la decisión arquitectónica central. **El backbone convierte un espectrograma en una secuencia de descriptores de 512-d a 6,25 Hz.** Colapsa la frecuencia porque «qué frecuencias hay» es precisamente la información a extraer; conserva el tiempo porque VLAD necesita un *conjunto* de muestras para estimar estadísticas.

Dos notas sobre `block1`:

- `nn.Conv2d(1, 64, ...)`: **un solo canal de entrada.** El espectrograma entra como imagen en escala de grises. No hay ImageNet, no hay RGB, no hay pesos preentrenados de visión.
- `stride=(1,1)` en la conv 7×7 — un ResNet de imágenes usaría stride 2. Aquí no se submuestrea en la primera capa, cuando la resolución es más barata en canales.

Y la última fila, `conv2d 7×1, 512`, es la que aplana la frecuencia: un kernel que cubre **todo** el eje restante, funcionalmente una capa densa aplicada a cada instante por separado. Cuesta **1.835.520 parámetros** (`512 × 512 × 7 + 512`), la segunda partida más cara del modelo después de la cabeza que nunca se ejecuta.

---

## 5. El campo receptivo: el «descriptor local» ve 1,84 segundos

La tabla invita a leer «un descriptor cada 16 frames» como «cada descriptor representa 160 ms de audio». **Es falso**, y por bastante.

Medido por retropropagación, linealizando el modelo (ReLU → identidad, MaxPool → AvgPool) para obtener la geometría pura:

| Medición | Valor |
|---|---|
| **Campo receptivo temporal** de un descriptor | **184 frames = 1.840 ms** |
| Paso entre descriptores consecutivos | 16 frames = 160 ms |
| **Solape entre vecinos** | **91,3 %** |
| Campo receptivo en frecuencia | **257 de 257 bins** (todo el espectro) |

Cada «descriptor local» ve **1,84 segundos** de audio, y dos consecutivos comparten el 91 % de su entrada. En frecuencia, cada uno ve el espectro completo.

{{< concept-alert type="clave" >}}
**Esto reencuadra toda la discusión sobre VLAD.**

- Los `x_i` que entran a la agregación **no son fonemas ni frames**: son resúmenes de ventanas de casi 2 segundos, fuertemente solapadas. La «localidad» de *Vector of **Locally** Aggregated Descriptors* es relativa — local frente al enunciado completo, no local en sentido acústico.
- Para un clip típico (16,4 s con espejado, ~102 descriptores de 1,84 s cada uno): **cada instante de audio está representado ~11 veces** en el conjunto que VLAD agrega.
- Y explica por qué **el promedio temporal funciona tan mal** (10,48 % de EER contra 3,22 %): los descriptores son tan redundantes que promediarlos destruye casi toda la variabilidad, mientras VLAD conserva *cómo se distribuyen* respecto a los centroides.
{{< /concept-alert >}}

---

**Anterior:** [El dataloader y el eje de la normalización](01-el-dataloader-y-la-normalizacion) · **Siguiente:** [NetVLAD desarmado](03-netvlad-desarmado)
