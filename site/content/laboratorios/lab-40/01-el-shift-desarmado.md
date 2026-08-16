---
title: "01 - El shift desarmado"
weight: 10
math: true
---

> Qué construye realmente el notebook cuando llama a `TSN(..., is_shift=True)`, dónde quedan los 16 módulos, y por qué el nombre de un archivo `.pth` es lo que hace que el checkpoint sea cargable.

---

## 1. El modelo se llama TSN

La primera sorpresa al leer el código es que **no existe una clase `TSM`**. La celda que instancia el modelo hace:

```python
net = TSN(num_class, this_test_segments if is_shift else 1, modality,
          base_model=this_arch,
          consensus_type=args['crop_fusion_type'],
          is_shift=is_shift, shift_div=shift_div, shift_place=shift_place, ...)
```

`TSN`, importada de `ops/models.py`. TSM **no es una arquitectura separada**: es el marco de [TSN](/papers/tsn-wang-2016) con un argumento en `True`. Todo lo demás —el muestreo por segmentos, el consenso por promedio, el dropout de 0,8, la ResNet-50 preentrenada— es infraestructura heredada. La diferencia entre los dos modelos de la clase cabe en un booleano.

La salida de esa celda lo confirma:

```
=> shift: True, shift_div: 8, shift_place: blockres
    Initializing TSN with base model: resnet50.
    TSN Configurations:
        input_modality:     RGB
        num_segments:       8
        consensus_module:   avg
        dropout_ratio:      0.8
=> base model: resnet50
Adding temporal shift...
=> n_segment per stage: [8, 8, 8, 8]
=> Processing stage with 3 blocks residual
=> Using fold div: 8
... (16 veces)
```

Los 16 `Using fold div: 8` son los 16 bloques bottleneck de la ResNet-50: **3 + 4 + 6 + 3**. Contarlos es la verificación de que el desplazamiento se insertó donde corresponde. Si aparecieran 8, se habría activado el `n_round = 2` que la implementación reserva para ResNet-101 y media red estaría sin módulo.

---

## 2. El nombre del archivo es el archivo de configuración

El checkpoint se llama:

```
TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth
```

y **no hay ninguna otra fuente de configuración**. No hay JSON, no hay metadata dentro del `.pth`. La arquitectura se reconstruye parseando el nombre:

```python
def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    return False, None, None
```

Descompuesto, cada fragmento decide algo:

| Fragmento | Significado | Dónde impacta |
|---|---|---|
| `kinetics` | 400 clases | `num_class = 400`, hardcodeado aparte |
| `RGB` | modalidad | `length = 3` canales por frame |
| `resnet50` | backbone | `this_arch`, extraído con `split('TSM_')[1].split('_')[2]` |
| **`shift8`** | `n_div = 8` | `fold = C // 8` **por dirección** |
| **`blockres`** | residual shift | el módulo va dentro de la rama residual |
| `avg` | consenso por promedio | `crop_fusion_type` |
| `segment8` | $T = 8$ | `test_segments` |
| `e100` | 100 épocas | nada |
| **`dense`** | entrenado con muestreo denso | nada — y ahí hay un problema |

Dos observaciones que salen de esta tabla.

{{< concept-alert type="cuidado" >}}
**`shift8` no significa "se desplaza 1/8 del tensor".** El código calcula `fold = C // 8` y mueve **dos** folds —uno hacia el futuro, otro hacia el pasado—, de modo que el total desplazado es **1/4**. El paper es explícito: *"the performance reaches the peak when 1/4 (1/8 for each direction) of the channels are shifted"*. La cifra de 1/8 que aparece en la slide de la clase corresponde al modo **unidireccional**, que es el online.
{{< /concept-alert >}}

**El sufijo `dense` es una inconsistencia del tutorial.** Estos pesos se entrenaron con muestreo denso —clips contiguos estilo I3D—, pero la celda de configuración fija `'dense_sample': False`, es decir muestreo uniforme por segmentos. Se evalúa el modelo con una política de muestreo distinta de la de su entrenamiento. Funciona igual, porque Kinetics perdona mucho, pero es una discrepancia que el notebook no menciona.

---

## 3. Dónde queda el módulo exactamente

La inyección ocurre en `make_temporal_shift`:

```python
def make_block_temporal(stage, this_segment):
    blocks = list(stage.children())
    for i, b in enumerate(blocks):
        if i % n_round == 0:
            blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
    return nn.Sequential(*blocks)
```

`TemporalShift` **envuelve la `conv1`** del bloque bottleneck: la 1×1 que abre la rama residual. Eso es lo que significa `blockres`, y no es un detalle de implementación sino la segunda de las dos correcciones que hacen viable el método. Al estar dentro de la rama, la conexión identidad sigue transportando la activación **sin desplazar**, de modo que la información original permanece accesible. La alternativa —`shift_place='block'`, insertar antes del bloque completo— degrada el modelado espacial, y el paper la descarta con datos.

La aritmética del fold, etapa por etapa, con $C$ = canales de entrada a esa `conv1`:

| Etapa | Bloques | $C$ | `fold = C//8` | Desplazado |
|---|---|---|---|---|
| `layer1[0]` | 1 | 64 | 8 | 16 / 64 |
| `layer1[1:3]` | 2 | 256 | 32 | 64 / 256 |
| `layer2[0]` | 1 | 256 | 32 | 64 / 256 |
| `layer2[1:4]` | 3 | 512 | 64 | 128 / 512 |
| `layer3[0]` | 1 | 512 | 64 | 128 / 512 |
| `layer3[1:6]` | 5 | 1024 | 128 | 256 / 1024 |
| `layer4[0]` | 1 | 1024 | 128 | 256 / 1024 |
| `layer4[1:3]` | 2 | 2048 | 256 | 512 / 2048 |

Todos los anchos son divisibles por 8, así que la proporción es exactamente 1/4 en los 16 módulos.

---

## 4. Las tres líneas que son todo el modelo

```python
@staticmethod
def shift(x, n_segment, fold_div=3, inplace=False):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)
    fold = c // fold_div

    out = torch.zeros_like(x)
    out[:, :-1, :fold]      = x[:, 1:,  :fold]        # futuro  -> presente
    out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # pasado  -> presente
    out[:, :,  2*fold:]     = x[:, :,   2*fold:]      # sin desplazar

    return out.view(nt, c, h, w)
```

Para el frame en el instante $t$, sobre el eje de canales:

```
canales:   [0 ─────── C/8) [C/8 ────── C/4) [C/4 ───────────────────── C)
contenido:   features de     features de           features de t
                t+1              t-1                (sin tocar)
                12.5%            12.5%                  75%
```

Tres detalles que se leen del código y no de la clase:

**El `view` es el único lugar donde existe el tiempo.** El tensor llega como `(N·T, C, H, W)` —el tiempo aplanado dentro del lote— y el módulo lo reinterpreta para saber quién es vecino de quién. Fuera de estas líneas, para la ResNet-50 la entrada es un lote de imágenes independientes.

**El relleno es con ceros.** `torch.zeros_like` implica que el primer frame no recibe pasado y el último no recibe futuro: en ambos, 1/8 de los canales queda anulado, y en cada uno de los 16 módulos. Con $T = 8$ eso afecta al 25 % de los segmentos.

**La versión eficiente está deshabilitada.** El repositorio incluye una clase `InplaceShift` que evitaría copiar el tensor completo, pero está apagada con un `raise NotImplementedError` por errores de orden en ejecución paralela. La consecuencia se discute en la [profundización](/clases/clase-40/profundizacion): en el código que efectivamente corre, el tráfico de memoria es proporcional a $C$ y **no** a la fracción desplazada, de modo que el ahorro de latencia que el *partial shift* promete no se materializa.

---

## 5. La cadena de shapes, de un AVI a 400 logits

| Paso | Salida |
|---|---|
| `ffmpeg -vf scale=-1:331` | 250 JPEGs de 441×331 |
| `_get_test_indices` | 8 índices: `[16, 47, 79, 110, 141, 172, 204, 235]` |
| `_load_image` × 8 | 8 PIL Images de 441×331 |
| `GroupScale(256)` | 8 × 341×256 |
| `GroupCenterCrop(224)` | 8 × 224×224 |
| `Stack(roll=False)` | ndarray `(224, 224, 24)` |
| `ToTorchFormatTensor(div=True)` | tensor `(24, 224, 224)` en [0,1] |
| `GroupNormalize` | mismo shape, media/std de ImageNet |
| `DataLoader` | `(1, 24, 224, 224)` |
| `.view(-1, 3, H, W)` | `(8, 3, 224, 224)` ← el tiempo se vuelve lote |
| `base_model(...)` | `(8, 2048)` |
| `new_fc(...)` | `(8, 400)` ← **400 logits por frame** |
| consenso `.mean(1)` | `(1, 400)` |
| `softmax` | probabilidades |

El paso decisivo es `Stack`: los 8 frames **se apilan sobre el eje de canales** (8 × 3 = 24), no en una dimensión temporal. Para la red son un lote de 8 imágenes RGB. El tiempo no existe como dimensión en ningún tensor de la red — solo dentro de los 16 `view` de los módulos.

Y el consenso opera **sobre predicciones, no sobre features**: `new_fc` emite 400 logits por frame y recién ahí se promedian. Cada frame da su opinión sobre las 400 clases; lo único que impide que sean opiniones independientes es que los features que las produjeron ya venían contaminados por sus vecinos.

{{< concept-alert type="nota" >}}
**Dos flags que en este lab están bien resueltos.** `Stack(roll=...)` y `ToTorchFormatTensor(div=...)` se derivan de `this_arch` en lugar de estar hardcodeados: con ResNet-50 dan `roll=False` (canales en RGB) y `div=True` (rango [0,1] antes de la normalización de ImageNet). Sería `True`/`False` para BNInception, entrenada con Caffe en BGR y [0,255]. Es el contraejemplo del bug que costó 82 puntos en el [Laboratorio 38](/laboratorios/lab-38), donde el preproceso asumía [0,1] y el modelo esperaba [−1,1].
{{< /concept-alert >}}

---

## 6. La carga del checkpoint verifica la arquitectura

```python
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
net.load_state_dict(base_dict)
```

El `split('.')[1:]` decapita el prefijo `module.` que dejó `nn.DataParallel` en el entrenamiento. Pero el detalle interesante es otro: al hacer `blocks[i].conv1 = TemporalShift(b.conv1, ...)`, la ruta del parámetro cambia.

```
sin TSM:  base_model.layer1.0.conv1.weight
con TSM:  base_model.layer1.0.conv1.net.weight
                                   ^^^^
```

El checkpoint contiene **16 claves con `.net.`**, una por módulo. De ahí se sigue algo que el notebook no dice: si `parse_shift_option_from_log_name` hubiera devuelto `is_shift=False`, `load_state_dict` reventaría con 16 claves faltantes y 16 sobrantes.

**El nombre del archivo no es documentación: es lo que hace que el checkpoint sea cargable.** Un contrato frágil —renombrar el `.pth` cambia el modelo— pero que se autoverifica: el `<All keys matched successfully>` es la prueba de que la arquitectura reconstruida coincide con la entrenada.

Ese mensaje tiene un corolario práctico. Si el 100 % de los pesos vino del checkpoint de Kinetics, entonces la descarga de 97,8 MB de pesos ImageNet que dispara `pretrain='imagenet'` en la celda anterior fue **íntegramente desperdiciada**: se sobreescriben todos.

---

## Ver también

- [02 - La varianza intra-clase](02-la-varianza-intra-clase) — qué hace el modelo así construido sobre cinco clips de la misma acción.
- [03 - La ablación del shift](03-la-ablacion-del-shift) — apagar los 16 módulos y medir qué se pierde.
- [Clase 40 - Práctica](/clases/clase-40/practica/01-el-modulo-de-desplazamiento) — el mismo módulo implementado desde cero y verificado contra una convolución temporal.
- [Fundamento: Desplazamiento Temporal](/fundamentos/desplazamiento-temporal) — el mecanismo de forma autónoma.
