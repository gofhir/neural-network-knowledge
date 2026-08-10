---
title: "El ReLU sobre los logits"
weight: 5
---

La Parte 2 cambia todo: convoluciones **2D** sobre espectrogramas en vez de 1D sobre onda cruda, 72 millones de parámetros preentrenados y congelados, y una `collate_fn` que replica etiquetas. Es la contraparte empírica de la primera mitad de la [clase 39](/clases/clase-39) —*el espectrograma es una imagen, usa una CNN 2D*— y del [fundamento de transfer learning](/fundamentos/transfer-learning).

## Dos precisiones sobre VGGish

**El link del notebook está equivocado.** La celda dice *"el modelo VGGish (arxiv.org/pdf/1610.00087)"*, pero ese es el paper de las redes M de la Parte 1. VGGish viene de [Hershey et al. (2017)](/papers/vggish-hershey-2017), y el contexto real importa: fue entrenado sobre **YouTube-100M — 70 millones de videos, 5.24 millones de horas de audio, unos 20 mil millones de ejemplos de 960 ms** con 30 871 etiquetas. Contra los 8732 clips de UrbanSound8K, es otra escala.

**No son MFCC, son log-mel.** El notebook repite "features MFCC", pero `vggish_input.waveform_to_examples` produce un espectrograma log-mel. Del paper:

> *"computing a spectrogram using 64 mel-spaced frequency bins, and the magnitude of each bin is log-transformed. This gives log-mel spectrogram patches of 96 × 64 bins"*

La diferencia no es terminológica. MFCC agrega un paso final —una **DCT** sobre las bandas mel— que decorrelaciona y comprime a ~13-40 coeficientes. VGGish **omite deliberadamente esa DCT**, porque destruye la estructura local en el eje de frecuencia, que es exactamente lo que los kernels de 3×3 de una CNN 2D necesitan para significar algo. Es la misma razón por la que no se aplica una DCT a las imágenes antes de una CNN. El [fundamento de MFCC y escala mel](/fundamentos/mfcc-y-escala-mel) desarrolla el punto.

De ahí salen los números del código: **96 frames de 10 ms** (0.96 s de audio) × **64 bandas mel**, y el `512 * 24` del primer `Linear` viene de que cuatro `MaxPool2d(2)` reducen 96×64 a 6×4 = 24 posiciones por 512 canales.

## El defecto

El clasificador de VGGish está definido así:

```python
self.fc = nn.Sequential(
    nn.Linear(512 * 24, 4096), nn.ReLU(inplace=True),   # fc[0], fc[1]
    nn.Linear(4096, 4096),     nn.ReLU(inplace=True),   # fc[2], fc[3]
    nn.Linear(4096, 128),      nn.ReLU(inplace=True),   # fc[4], fc[5]
)
```

Y el fine-tuning hace:

```python
for param in net.parameters():
    param.requires_grad = False
net.fc[-2] = nn.Linear(in_features=4096, out_features=10, bias=True)
```

`fc[-2]` es el índice 4, así que **reemplaza la capa correcta**. El conteo confirma que solo queda entrenable esa capa: `40 970` parámetros (4096 × 10 + 10) de ~72 M.

El problema es `fc[-1]`. Sigue siendo el `ReLU`, y ahora está aplicado **sobre los logits**.

Tiene todo el sentido en el modelo original —esa salida de 128 dimensiones es un *embedding*, y el `ReLU` forma parte de él— y ninguno cuando la capa pasa a producir logits para `CrossEntropyLoss`. Las consecuencias son dos y ambas son severas:

1. **Ningún logit puede ser negativo.** Todas las clases improbables se colapsan a 0 y **empatan entre sí**. El modelo pierde la capacidad de expresar "esta clase es muy improbable". `CrossEntropyLoss` aplica internamente un `log_softmax`, que espera logits sin restricción de signo: forzarlos a ser no negativos rompe el supuesto de la función de pérdida.
2. **Gradiente cero en la zona negativa.** Una neurona de salida que caiga en preactivación negativa recibe derivada 0 de la `ReLU` y no puede volver. Y como el resto de la red está congelada, sus features de entrada tampoco cambian: la neurona queda **muerta de forma permanente**.

## Lo que se mide

Tres clases con precision, recall y F1 exactamente **0.00**:

```
children_playing   0.00   0.00   0.00   300
drilling           0.00   0.00   0.00   300
gun_shot           0.00   0.00   0.00   105
```

Con `zero_division=0`, precision 0.00 significa que el modelo **nunca predijo esas clases**: cero predicciones emitidas, no predicciones erradas. Son **705 de 2619 parches — el 26.9 % del conjunto de test** — estructuralmente imposibles de acertar.

![Matriz de confusión de VGGish con la configuración original](/laboratorios/lab-39/vggish-matriz.jpg)

Y el diagnóstico sobre los logits del modelo entrenado:

```
logits: mínimo 0.000 | máximo 6.9
fracción de logits exactamente 0 (saturados por la ReLU): 84.9 %
```

**El 84.9 % de los logits vale exactamente cero.** De los diez logits de cada ejemplo, en promedio **8.5 están en cero**: el `argmax` se decide entre ~1.5 candidatos reales y el resto empata. El modelo opera con una fracción mínima de su expresividad.

El corolario es que **el techo del modelo no era 100 % sino 73.1 %**. Con 64.53 % obtenido, sobre las siete clases vivas está acertando el **88.3 %** — bastante mejor de lo que sugiere la cifra global.

## El factorial 2×2

La `ReLU` es el mecanismo, pero el **disparador** podría ser `lr = 0.01`: con pasos diez veces mayores que el default de Adam, algunas neuronas de salida se van a la zona negativa en las primeras iteraciones y ahí la `ReLU` las sella. Son dos hipótesis distintas, y el diseño que las separa es un factorial completo. Cada corrida cuesta 54 segundos con los log-mel cacheados.

| Salida | `lr` | Test | Clases muertas | Logit mínimo | % logits en cero |
|---|---|---|---|---|---|
| `ReLU` | 0.01 | 64.53 % | **3** | 0.000 | **84.9 %** |
| `ReLU` | 0.001 | 71.71 % | 1 (`drilling`) | 0.000 | 71.8 % |
| **`Identity`** | **0.01** | **91.45 %** | **0** | **−38.36** | 0 % |
| `Identity` | 0.001 | 86.52 % | 0 | −20.60 | 0 % |

**La `ReLU` es la causa; el learning rate solo modula.** Eliminarla vale **+26.92 puntos** con `lr = 0.01` y **+14.81** con `lr = 0.001`. Bajar el learning rate sin eliminarla vale +7.18 y **deja todavía una clase muerta**.

El arreglo es una línea:

```python
net.fc[-2] = nn.Linear(in_features=4096, out_features=10, bias=True)
net.fc[-1] = nn.Identity()          # el ReLU final se elimina
```

**Sin la `ReLU`, el logit mínimo llega a −38.36.** Esa es la magnitud del rango dinámico que se estaba truncando: casi cuarenta unidades de logit negativo que el modelo *necesita* para expresar confianza en contra de una clase.

{{< concept-alert type="clave" >}}
**El efecto del learning rate cambia de signo según la salida.** Con `ReLU` conviene `lr = 0.001` (**+7.18**); sin `ReLU` conviene `lr = 0.01` (**+4.93**).

Con una sola capa lineal entrenable y apenas 3 épocas, el learning rate alto converge más rápido — lo único que lo hacía perjudicial era que empujaba neuronas a la zona muerta. Probar solo el learning rate manteniendo la `ReLU` habría llevado a la conclusión "0.001 es mejor", que es falsa una vez corregida la arquitectura.

Es el mismo patrón que aparece en la Parte 1 con [M5 contra M18](/laboratorios/lab-39/04-learning-rate-y-profundidad): variar un factor con el otro fijo puede invertir la conclusión.
{{< /concept-alert >}}

## Qué revela sobre el embedding de AudioSet

Mirando dónde funciona bien la configuración rota, se aprecia la calidad de las features preentrenadas. En las siete clases vivas, VGGish resuelve sin esfuerzo justamente lo que las redes M no lograban representar por falta de contexto temporal:

| Clase | F1 M3 | F1 M5 | **F1 VGGish** |
|---|---|---|---|
| street_music | 0.37 | 0.65 | **0.88** |
| engine_idling | 0.57 | 0.73 | **0.92** |
| car_horn (recall) | 0.56 | 0.61 | **0.95** |
| siren | 0.57 | 0.72 | **0.85** |

`car_horn` —la clase que el *global average pooling* de las redes M diluía sistemáticamente— alcanza recall 0.95. La razón es arquitectónica: VGGish no promedia features sobre el clip completo, sino que **procesa parches de 0.96 s por separado**, de modo que el parche que contiene la bocina no se ve arrastrado por los que contienen ambiente.

Ese es precisamente el tema de la última página.

---

**Siguiente:** [Agregación y transfer learning](/laboratorios/lab-39/06-agregacion-y-transfer-learning) — cómo combinar los tres parches, y cuánto vale realmente traer AudioSet a un dataset de 8732 clips.
