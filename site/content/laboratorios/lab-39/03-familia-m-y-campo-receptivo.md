---
title: "La familia M y el campo receptivo"
weight: 3
---

La primera parte del lab entrena sobre la **onda cruda**, sin espectrograma. Es la tesis de [Dai et al. (2016)](/papers/raw-waveforms-dai-2017), y va contra lo que la [clase 39](/clases/clase-39) presenta en su primera mitad: los modelos sobre waveform de la época usaban ~2 capas convolucionales y perdían feo contra los que usaban log-mel; los autores sostienen que el problema no era la representación cruda sino **la falta de profundidad**.

Con 18 capas llegan a 71.68 % en UrbanSound8K, competitivo con el ~68 % que reportaba una CNN sobre espectrograma. El costo es que necesitan 18 capas donde el espectrograma necesitaba 2, porque las primeras están gastándose en aprender lo que la FFT entrega gratis.

## Los cuatro principios de diseño

**Primera capa con campo receptivo enorme (`k = 80`), el resto con `k = 3`.** En imágenes, VGG popularizó "kernels chicos en todas partes"; acá esa regla se rompe exactamente una vez. Un kernel de 80 muestras a 8 kHz cubre 10 ms, la ventana estándar de un frame de MFCC. Con `stride = 4`, esa capa es **literalmente un banco de 256 filtros FIR aprendidos aplicados con hop de 4 muestras**: una STFT aprendida. El paper observa que los filtros convergen a respuestas de tipo pasabanda — en vez de imponer la escala mel a mano, la red la descubre.

**Downsampling agresivo.** Entre el `stride = 4` de `conv1` y el `MaxPool1d(4)` inmediato, la primera capa reduce el eje temporal por 16. Sin eso, 32 000 muestras con kernels de 3 exigirían cientos de capas para que una neurona viera algo más que un chirrido.

**Fully convolutional: `AvgPool1d(L)` en lugar de capas FC.** El *global average pooling* colapsa todo el eje temporal a un vector de $C$ canales. El clasificador de M3 tiene 2570 parámetros, y el modelo queda invariante a **dónde** ocurre el evento dentro del clip. El paper reporta que las variantes con dos capas FC de 1000 unidades andan igual o peor con muchísimos más parámetros.

**BatchNorm en cada capa**, que es lo que hace entrenable a M18.

## Qué ve realmente cada modelo

Calculando el campo receptivo capa a capa sobre las cuatro arquitecturas —las longitudes coinciden exactamente con los `AvgPool1d(498/30/25/20)` que el código tiene escritos a mano, lo que confirma que están dimensionadas para una entrada de 32 000 muestras:

| Modelo | Capas con peso | Params | Posiciones antes del GAP | **Campo receptivo** | Paper |
|---|---|---|---|---|---|
| M3 | 3 | 0.22 M | 498 | **19.5 ms** | 56.12 % |
| M5 | 5 | 0.56 M | 30 | **200.9 ms** | 63.42 % |
| M11 | 11 | 1.8 M | 25 | **799.5 ms** | 69.07 % |
| M18 | 18 | 3.7 M | 20 | **1358.3 ms** | 71.68 % |

Esta tabla es la respuesta a las Actividades 4 y 5, y conviene leerla con cuidado.

**M3 promedia 498 vectores que vieron 19.5 ms cada uno.** Es un clasificador de *textura espectral instantánea*: puede distinguir el timbre promedio de una sirena del de un ladrido, pero no tiene forma de representar que un martillo neumático es un golpe repetido a cierta cadencia, o que un ladrido tiene envolvente de ataque y decaimiento. Esa información se destruye en el promedio final.

**M18 promedia 20 vectores que vieron 1.36 segundos cada uno.** Ahí sí cabe un patrón rítmico completo, y los +15.56 puntos absolutos que el paper mide entre M3 y M18 son lo que cuesta esa diferencia.

Y es profundidad, no tamaño. El paper prueba `M5-big` con 2.2 M de parámetros —más que M11— y obtiene 63.30 % contra el 69.07 % de M11 con 1.8 M. Más filtros no compran contexto temporal; más capas sí. El techo también está documentado: **M34-res, con 4 M de parámetros, baja a 63.47 %**, peor que M11.

## M3 en la práctica: los tres sumideros

![Curvas de entrenamiento de M3, 20 épocas](/laboratorios/lab-39/m3-curvas.jpg)

M3 entrenado 20 épocas alcanza **56.13 %** en su mejor época (la 19), con 55.55 % en train. Pero el desglose por clase dice mucho más que el número. Reconstruyendo cuántas veces el modelo emitió cada clase a partir de `n_pred = recall × support / precision`:

| Clase | Reales | Predichas | **Ratio** | P | R | F1 |
|---|---|---|---|---|---|---|
| jack_hammer | 120 | 183 | **1.53×** | 0.61 | 0.93 | 0.74 |
| air_conditioner | 100 | 181 | **1.81×** | 0.31 | 0.56 | 0.40 |
| gun_shot | 35 | 65 | **1.86×** | 0.37 | 0.69 | 0.48 |
| drilling | 100 | 120 | 1.20× | 0.60 | 0.72 | 0.65 |
| engine_idling | 96 | 79 | 0.82× | 0.63 | 0.52 | 0.57 |
| dog_bark | 100 | 67 | 0.67× | 0.78 | 0.52 | 0.62 |
| siren | 86 | 58 | 0.67× | 0.71 | 0.48 | 0.57 |
| car_horn | 36 | 23 | 0.64× | 0.87 | 0.56 | 0.68 |
| children_playing | 100 | 58 | 0.58× | 0.64 | 0.37 | 0.47 |
| street_music | 100 | 40 | **0.40×** | 0.65 | 0.26 | 0.37 |

Hay **tres clases sumidero** que absorben casi el doble de ejemplos de los que les corresponden, y clases que el modelo casi no se atreve a emitir.

![Matriz de confusión de M3, mejor época](/laboratorios/lab-39/m3-matriz.jpg)

**Los sumideros son sonidos de banda ancha sin estructura.** Con 19.5 ms de contexto, un zumbido de banda ancha (`air_conditioner`) es el atractor natural de cualquier cosa sin firma clara — de ahí su precision de 0.31, la peor de las diez: es el basurero del clasificador. `gun_shot` es lo mismo en versión impulsiva: en 19.5 ms, un disparo, un golpe de martillo, un portazo y el ataque de un ladrido **son el mismo objeto**, un transitorio de banda ancha.

**El global average pooling diluye los eventos breves**, y eso explica los recalls bajos. Una bocina que suena 0.5 s dentro de un clip de 3.63 s aporta solo ~14 % de las ventanas promediadas; el 86 % restante es ambiente urbano. El evento existe en el tensor, pero se disuelve en el promedio.

Eso explica la firma más llamativa del reporte: **`car_horn` tiene precision 0.87 y recall 0.56**. Cuando el modelo dice "bocina" casi siempre acierta —un tono armónico sostenido es distintivo incluso en 19.5 ms—, pero se le escapa el 44 % porque en esos clips el promedio quedó dominado por el fondo. Lo mismo con `dog_bark` (0.78 / 0.52) y `siren` (0.71 / 0.48): sabe reconocerlas, no logra rescatarlas de la dilución.

**`street_music` con recall 0.26 es el caso extremo.** Distinguir música de ruido callejero exige estructura melódica y rítmica desplegada sobre **segundos**. M3 tiene 19.5 ms y después promedia.

## Lo que corrigen 50 épocas, y lo que no

El curso provee un M3 entrenado 50 épocas con el mismo código, que da **68.61 %**. Comparado con el de 20:

| Clase | F1 @20 | F1 @50 | Δ | Precision @20 → @50 |
|---|---|---|---|---|
| air_conditioner | 0.40 | **0.70** | **+0.30** | 0.31 → **0.62** |
| street_music | 0.37 | 0.64 | +0.27 | 0.65 → 0.77 |
| engine_idling | 0.57 | **0.81** | +0.24 | 0.63 → 0.85 |
| children_playing | 0.47 | 0.67 | +0.20 | 0.64 → 0.64 |
| jack_hammer | 0.74 | 0.82 | +0.08 | 0.61 → **0.90** |
| **car_horn** | 0.68 | 0.64 | **−0.04** | 0.87 → 0.62 |
| **drilling** | 0.65 | 0.57 | **−0.08** | 0.60 → 0.54 |

Los sumideros **se desinflan**: el desbalance de masa predicha se comprime de un rango de 4.6× a uno de 2.2×. Eso es aprendizaje estructural, no memorización — un clasificador deja de usar `air_conditioner` como basurero cuando entiende mejor las clases.

También obliga a matizar la explicación puramente arquitectónica: `street_music` sube de 0.26 a 0.55 de recall **sin cambiar el campo receptivo**. El límite de 19.5 ms marca el techo, pero a 20 épocas M3 todavía no lo había alcanzado.

Y aparece un trade-off que la métrica global esconde: `drilling` cae en ambas métricas mientras `jack_hammer` sube a 0.90 de precision. **El modelo resolvió el par taladro/martillo neumático a costa de `drilling`.**

## M5: el campo receptivo diez veces mayor

![Matriz de confusión de M5, mejor época](/laboratorios/lab-39/m5-matriz.jpg)

M5 alcanza **76.63 %** en la época 19 (74.35 % en train), y su desglose muestra un cambio cualitativo:

| Clase | F1 M3 | F1 M5 | Δ | Ratio M3 → M5 |
|---|---|---|---|---|
| air_conditioner | 0.40 | **0.82** | **+0.42** | 1.81× → **0.75×** |
| street_music | 0.37 | 0.65 | +0.28 | 0.40× → 0.71× |
| dog_bark | 0.62 | **0.86** | +0.24 | 0.67× → 0.88× |
| gun_shot | 0.48 | 0.72 | +0.24 | 1.86× → **1.63×** |
| jack_hammer | 0.74 | **0.93** | +0.19 | 1.53× → 0.98× |
| siren | 0.57 | 0.72 | +0.15 | 0.67× → **1.41×** |
| **car_horn** | 0.68 | 0.70 | **+0.02** | 0.64× → 0.75× |

![Comparación consolidada M3 contra M5](/laboratorios/lab-39/m3-vs-m5.jpg)

**El caso `air_conditioner` es el más espectacular:** pasa de basurero (1.81×, precision 0.31) a ser **la clase de mayor precisión del modelo** (0.96), y a sub-predecirse. Con 19.5 ms un zumbido de banda ancha era indistinguible de cualquier fondo; con 201 ms el modelo puede constatar que es *estacionario*, y esa estacionariedad sostenida resulta ser una firma muy discriminativa.

{{< concept-alert type="clave" >}}
**Los sumideros no desaparecen: se mudan.** En M3 absorbían `air_conditioner`, `jack_hammer` y `gun_shot`; en M5 los sumideros son `gun_shot` (1.63×), `siren` (1.41×) y `drilling` (1.27×). La incertidumbre del clasificador tiene que ir a alguna parte — lo que cambia con el campo receptivo es **en qué clase se deposita**, no que se elimine.
{{< /concept-alert >}}

**Y `car_horn` marca el límite de lo que la profundidad puede arreglar.** Sube apenas 0.02 y su recall se queda en 0.61. El problema no es el contexto de cada ventana sino **la agregación final**: el promedio sobre todo el clip diluye un evento que dura décimas de segundo. Eso no se corrige con más capas; se corrige con un pooling que privilegie el máximo, o con atención temporal — o, como hace VGGish en la Parte 2, **cortando el clip en parches y agregando las predicciones** en lugar de las features.

---

**Siguiente:** [El learning rate decide el orden](/laboratorios/lab-39/04-learning-rate-y-profundidad) — por qué M18, con 1358 ms de contexto, termina perdiendo contra M5.
