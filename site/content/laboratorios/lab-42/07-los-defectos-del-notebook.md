---
title: "Los defectos del notebook"
weight: 7
---

El notebook es una plantilla adaptada, no código escrito para este curso. Se nota en los comentarios en inglés con tono de marketing —*"SOTA"*, *"spatio-temporal pipeline"*, *"tensor memory bloat"*— y en los cuatro imports que no se usan. Ninguno de sus defectos impide que corra, pero **cinco de los nueve rompen la actividad** en cuanto se procesa un segundo video.

## 1. `wget` sin `-O` no sobreescribe: renombra

```python
!wget https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
```

Al ejecutar la celda por segunda vez, la descarga se guarda como `people-detection.mp4.1`. El original queda intacto y el bucle sigue usándolo. En la ejecución de referencia esto se materializó:

```
Saving to: 'people-detection.mp4.1'
```

Es inocuo con el mismo archivo, pero **en la actividad da un análisis equivocado**: si se descarga el segundo video y se corre el pipeline, se estará procesando el primero mientras se cree lo contrario.

```python
!wget -q -O nombre.mp4 <url>    # ← el arreglo
```

## 2. `ffmpeg` sin `-y` bloquea la ejecución

```python
# celda 6
!ffmpeg -i entrada.mp4 -vcodec libx264 salida.mp4          # sin -y
# celda 9
!ffmpeg -y -i entrada.mp4 -vcodec libx264 salida.mp4       # con -y
```

La misma operación, escrita de dos formas distintas en el mismo notebook — la firma más clara de que se armó por copia y pega. Cuando el archivo de salida ya existe, ffmpeg pregunta por `stdin`:

```
File 'compressed_yolo26_tracking_output.mp4' already exists. Overwrite? [y/N]
```

Colab ofrece un campo de texto, así que no es un cuelgue permanente; pero bloquea hasta que alguien responda, lo que rompe cualquier ejecución desatendida y obliga a intervenir tres veces si se procesan tres videos.

## 3. `int(fps)` y los videos que no son de fps entero

```python
fps = int(cap.get(cv2.CAP_PROP_FPS))
```

`people-detection.mp4` tiene **12,0 fps exactos** y no hay problema. De los seis videos usados en la actividad, **cuatro no lo son**:

| Video | fps real | `int()` | `round()` |
|---|---|---|---|
| `people-detection` | 12,000 | 12 ✓ | 12 |
| `person-bicycle-car` | 12,000 | 12 ✓ | 12 |
| `one-by-one-person` | 10,000 | 10 ✓ | 10 |
| `car-detection` | **12,500** | 12 ⚠️ | 13 |
| `bottle-detection` | **29,833** | 29 ⚠️ | 30 |
| `store-aisle` | **59,940** | 59 ⚠️ | 60 |
| `worker-zone` | **59,940** | 59 ⚠️ | 60 |

El video de salida se escribe declarando un frame rate que no corresponde a la duración de sus frames: avanza entre un 1,6 % y un 4 % más rápido. Con material que traiga audio, se desincroniza progresivamente.

```python
fps = round(cap.get(cv2.CAP_PROP_FPS)) or 30    # ← el arreglo
```

## 4. `cv2.VideoWriter` y el codec que ningún navegador reproduce

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
```

`mp4v` es **MPEG-4 Part 2**, no H.264. No es un capricho: las ruedas de PyPI de `opencv-python` se distribuyen **sin el encoder H.264** por el licenciamiento de las patentes AVC, y `VideoWriter_fourcc(*'avc1')` falla en silencio escribiendo un archivo corrupto.

De ahí que exista el paso de `ffmpeg`, que **no es cosmético**: sin él el reproductor de la celda siguiente muestra un rectángulo negro.

Dos trampas relacionadas: si `(width, height)` no coincide **exactamente** con las dimensiones del frame, `write()` no escribe nada y no avisa —el síntoma es un archivo de pocos KB—; y `cv2.VideoCapture` con una ruta inexistente **no lanza excepción**, solo deja `isOpened()` en `False`. Las dos comprobaciones defensivas que el notebook sí incluye están bien puestas.

## 5. Las trayectorias se dibujan desde el torso

```python
x, y, w, h = box
track.append((float(x), float(y)))      # centro de la caja
```

El centro de la caja está a la altura del torso. Cuando alguien se acerca a la cámara, la caja crece y **el centro sube en la imagen** aunque la persona no cambie de posición en el suelo: la trayectoria se curva por un artefacto de proyección.

El punto correcto para trayectorias sobre el plano del suelo es el centro inferior:

```python
track.append((float(x), float(y + h / 2)))    # los pies
```

## 6. Todas las estelas del mismo color

```python
cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)
```

Amarillo fijo para todos los tracks. Con cinco personas en pantalla no se distingue qué estela pertenece a quién — que es justamente lo que hay que ver para diagnosticar un cambio de identidad. Y `results.plot()` colorea las cajas **por clase**, no por identificador, así que tampoco ayuda: todas las personas salen del mismo azul.

```python
from ultralytics.utils.plotting import colors
cv2.polylines(annotated_frame, [points], False, colors(track_id, True), 2)
```

El mismo problema afecta a las máscaras de SAM, que se colorean por índice de detección; el detalle está en [la página anterior](../06-vocabulario-abierto-y-segmentacion).

## 7. `track_history` nunca se purga

```python
track_history = defaultdict(list)
...
if len(track) > 30: track.pop(0)
```

Cada trayectoria se poda a 30 puntos, pero **las entradas de tracks muertos no se borran nunca**. Con 3.000 identificadores creados quedan 3.000 listas vivas. En RAM es despreciable, y hay una forma de darle la vuelta: `len(track_history)` al final del bucle es **el número total de identidades creadas**, que resulta ser la métrica más útil de todo el laboratorio.

## 8. Cuatro imports muertos y dos librerías que nunca se usan

`import os` y `import urllib.request` no aparecen en ninguna celda —probablemente restos de una versión que descargaba el video con `urlretrieve()` en vez de `wget`—. Y la celda de instalación trae dos paquetes que el notebook nunca importa:

```python
!pip install -U ultralytics supervision roboflow opencv-python
```

`supervision` es peso muerto, pero vale la pena saber qué se está dejando pasar: `sv.ByteTrack()` como tracker independiente del detector, `sv.TraceAnnotator` para las estelas, `sv.LineZone` para contar cruces —que permitiría *cuantificar* los cambios de identidad en vez de describirlos a ojo— y `sv.DetectionsSmoother`.

## 9. Los dos estados globales que sobreviven entre celdas

Los dos más difíciles de detectar, porque no producen errores sino números equivocados.

**El contador de identificadores no siempre se reinicia.** En el pipeline de la Actividad 2, las cuatro corridas emiten identificadores consecutivos entre sí: 27-29, 30-36, 37-40, 41-44. El contador de `BaseTrack` es global y no se reinicia entre llamadas si el modelo no se recrea. Consecuencia práctica: cualquier métrica del tipo `ID_máximo − IDs_únicos` cuenta como propios los identificadores de las corridas anteriores. Hay que calcularla sobre el rango observado:

```python
quemados = (max(ids) - min(ids) + 1) - len(set(ids))
```

**`set_classes()` escribe sobre los pesos.** Tras varias corridas en el mismo proceso, la reparametrización de YOLO-World produce un `RuntimeError: Inference tensors do not track version counter`: los tensores creados bajo `torch.inference_mode()` no llevan contador de versión y no pueden usarse como pesos de una convolución en un contexto que sí lo espera. Reiniciar el entorno lo resuelve; instanciar un modelo nuevo por cada conjunto de prompts lo previene.

## El defecto del análisis, no del notebook

Uno propio, que conviene consignar porque es del mismo tipo. Al calcular la cobertura de cada track —detecciones obtenidas sobre muestras posibles— la primera versión de la función infería el paso de submuestreo a partir del hueco entre las **dos primeras** detecciones:

```python
span = (fr[-1] - fr[0]) // max(1, fr[1] - fr[0]) + 1     # ✗
```

Cuando un track empieza con un hueco, ese divisor sale mal y la cobertura supera el 100 %. En los datos aparecieron valores de 168 %, 198 % y hasta **1100 %**. El arreglo es pasar el paso explícitamente:

```python
span = (fr[-1] - fr[0]) // step + 1                      # ✓
```

Y el valor absurdo escondía un dato real: para que la fórmula diera 1100 %, la primera y la segunda detección de ese track tenían que estar separadas por 110 frames. Era el track que sobrevivió una oclusión de 1,5 segundos con siete muestras de margen respecto de `track_buffer`.

---

## Resumen

| # | Defecto | Rompe en | Arreglo |
|---|---|---|---|
| 1 | `wget` sin `-O` | el 2º video de la actividad | `wget -O nombre.mp4` |
| 2 | `ffmpeg` sin `-y` (celda 6) | la 2ª ejecución | agregar `-y` |
| 3 | `int(fps)` | 4 de 6 videos | `round(fps)` |
| 4 | `mp4v` + `VideoWriter` silencioso | archivo vacío sin aviso | verificar tamaño de salida |
| 5 | estela desde el torso | trayectorias curvadas | `y + h/2` |
| 6 | color fijo y por clase | ilegible con varios objetos | `colors(track_id, True)` |
| 7 | `track_history` sin purgar | nada; es aprovechable | `len(track_history)` como métrica |
| 8 | imports y librerías muertos | nada | — |
| 9 | contador global y `set_classes` | métricas y `RuntimeError` | rango observado; modelo nuevo |

---

**Volver al** [índice del laboratorio](../).
