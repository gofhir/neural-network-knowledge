---
title: "Lo que el detector nombra mal"
weight: 5
---

El enunciado de la actividad pide clasificar los errores en dos categorías: los que se refieren a la detección y los que se refieren a la asignación de identificadores. Al correr los tres videos nuevos apareció una tercera, y resultó ser **la más abundante de todas**.

## El fenómeno

`car-detection.mp4` es un video de automóviles. Estas son las clases que recibieron sus nueve tracks:

| Track | Clases recibidas | Objeto real |
|---|---|---|
| `id:1` | `cell phone` ×24, `boat` ×5, `car` ×4, `bus` ×2, `airplane` ×2 | Un auto |
| `id:6` | `cell phone` ×15 (100 %) | Un auto |
| `id:14` | `sink` ×8 (100 %) | Un auto |
| `id:8` | `bus` ×10, `car` ×11, `cell phone` ×7, `truck` ×1 | Un auto |
| `id:16` | `cell phone` ×12, `boat` ×3 | Un auto |

De las 121 detecciones del video, apenas unas **21 dicen `car`**. La clase mayoritaria de un video de autos es **`cell phone`**, y hay tracks enteros etiquetados `sink` de principio a fin.

![Estacionamiento visto desde arriba: el auto es cell phone 0.36 y boat 0.12, el ciclista es tennis racket 0.40](/laboratorios/lab-42/escena-mixta.jpg)

En `person-bicycle-car-detection.mp4` el patrón se repite con más variedad:

| Track | Clases recibidas | Objeto real |
|---|---|---|
| `id:11` | `tennis racket` ×20, `bicycle` ×10, `person` ×8 | Un ciclista |
| `id:41` | `tennis racket` ×21, `bicycle` ×19, `person` ×14 | Un ciclista |
| `id:6` | `car` ×35, `boat` ×8, `cell phone` ×7, `person` ×3, `sports ball` ×1 | Un auto |
| `id:27` | `car` ×56, `boat` ×6, `sports ball` ×3 | Un auto |

Y en el propio video del laboratorio, `people-detection.mp4`, el fenómeno aparece atenuado pero presente: 14 de 403 detecciones (3,5 %) con la clase equivocada, repartidas en 3 de los 9 tracks — `person` que pasa a `dog`, `cat`, `chair` y `bottle`.

![Los frames donde la clase cambia: dog 0.75, dog 0.80, person 0.91, chair 0.23, bottle 0.13, chair 0.49](/laboratorios/lab-42/errores-de-clase.jpg)

## Lo que no es: no es un problema de escala

La explicación cómoda sería que los objetos son demasiado pequeños. Es falsa, y medirlo importa porque cambia por completo el diagnóstico.

![Vista cenital de una carretera: los autos ocupan hasta la mitad del cuadro y se etiquetan cell phone](/laboratorios/lab-42/autos-cenitales.jpg)

Midiendo las cajas emitidas sobre el video de salida:

| Frame | Caja etiquetada `cell phone` | Fracción del cuadro |
|---|---|---|
| 78 | 179 × 247 px | **19,2 %** |
| 207 | 370 × 319 px | **51,2 %** |
| 327 | 190 × 190 px | 15,7 % |

Un objeto que ocupa la mitad del cuadro no es un objeto pequeño. Los autos están **perfectamente localizados**: las cajas se ajustan al vehículo, los tracks duran entre 15 y 37 frames con coberturas de 86 a 100 %, y las estelas de trayectoria son limpias.

**El detector encuentra los autos. Falla al nombrarlos.**

## Lo que sí es: punto de vista, y competencia entre 80 hipótesis

La cámara está montada **cenitalmente**, mirando hacia abajo. Un automóvil visto desde arriba es un rectángulo redondeado, liso y brillante, con una superficie oscura en el centro —el parabrisas— sobre un fondo gris uniforme. Es, geométricamente, la descripción de un teléfono sobre una mesa. El `sink` del track 14 sigue la misma lógica: un objeto blanco ovalado visto desde arriba.

COCO contiene decenas de miles de automóviles, prácticamente todos fotografiados desde el nivel de la calle. La vista cenital está fuera de su distribución de entrenamiento.

Pero la explicación no termina ahí, y la segunda mitad es la interesante. La **Actividad 2** corrió el mismo video con un detector de vocabulario abierto al que se le preguntó por tres conceptos:

```python
yolo_world.set_classes(["person", "bicycle", "car"])
```

![El mismo auto cenital, ahora car 0.85; la misma bicicleta, ahora bicycle 0.89](/laboratorios/lab-42/multiprompt.jpg)

| Objeto | YOLO26 (80 clases) | YOLO-World (3 prompts) |
|---|---|---|
| El auto cenital | `cell phone` 0,36 · `boat` 0,12 · `sports ball` 0,18 | **`car` 0,85** |
| El ciclista | `tennis racket` 0,40 · `bicycle` 0,10 | **`bicycle` 0,89** + **`person` 0,72** |

{{< concept-alert type="clave" >}}
**El mismo píxel, la misma vista cenital, y el auto pasa de `cell phone` a `car` con 0,85 de confianza.**

La evidencia visual siempre bastó para reconocerlo. Lo que fallaba no era la percepción sino la **competencia entre 80 hipótesis**: en un espacio de decisión donde `cell phone` existe, la vista cenital hace que `cell phone` le gane a `car`. Restringido a tres conceptos, `car` gana con holgura.

El error de clasificación no es un fallo de capacidad del modelo. Es un fallo de **cómo se reparte la probabilidad** cuando el objeto se ve desde un ángulo que el entrenamiento no cubrió.
{{< /concept-alert >}}

Esto también reordena lo que se puede decir sobre la evolución YOLO11 → YOLO26 que la clase menciona. El mecanismo **STAL** (*Small-Target-Aware-Labeling*) apunta a la persistencia del seguimiento cuando los objetos se pierden en la distancia, y `car-detection` **no es ese escenario**: los objetos son grandes. Donde sí hay objetos pequeños —las personas de `person-bicycle-car`, de unos 30 px— el detector las reconoce correctamente como `person` con confianzas de 0,90 y 0,51. La escala no es lo que rompe aquí.

## Y sin embargo, la identidad sobrevive

Lo notable es que ninguno de estos errores rompe el seguimiento. El track `id:8` de `car-detection` mantiene **una sola identidad durante 29 frames** mientras su etiqueta oscila entre `bus`, `cell phone`, `car` y `truck`. El `id:3` de `people-detection` va de `dog` a `person` a `cat` sin perder ni un frame de continuidad.

La razón está en dos líneas del código de Ultralytics. La primera es la matriz de costos:

```python
def get_dists(self, tracks, detections):
    dists = matching.iou_distance(tracks, detections)   # solo geometría
    if self.args.fuse_score:
        dists = matching.fuse_score(dists, detections)  # solo puntaje
    return dists
```

No hay ni una referencia a `cls`. **El asociador es class-agnostic**: una detección etiquetada `cat` compite por emparejarse con un track `person` en igualdad de condiciones, porque lo único que importa es dónde está la caja y cuánta confianza tiene.

La segunda es la actualización del track:

```python
self.score = new_track.score
self.cls = new_track.cls        # sobrescritura pura, sin voto ni suavizado
```

No hay voto mayoritario sobre el historial. Si el detector dice `cat` en el frame 73, la caja dice `cat` en el frame 73, aunque los 51 frames anteriores dijeran `person`.

### Por qué es el diseño correcto

Un tracker class-agnostic es robusto ante la inestabilidad del clasificador. La alternativa —exigir que la clase coincida para asociar— rompería el track del `id:8` cada vez que el clasificador dudara, y en `car-detection` eso sería casi todos los frames. Se ganarían etiquetas consistentes a cambio de identidades fragmentadas, que es exactamente lo que importa en seguimiento.

La definición que la clase da de la tarea pone la **preservación de identidad** en primer lugar, y esta decisión de diseño la respeta al costo de la etiqueta.

### Cuándo muerde

Si la aplicación cuenta objetos por categoría —"¿cuántos autos frente a cuántos camiones pasaron?"—, leer `results.boxes.cls` frame a frame da basura. La corrección es votar sobre el historial del track:

```python
from collections import Counter, defaultdict
class_history = defaultdict(list)
# dentro del bucle:
class_history[track_id].append(int(cls))
# al final:
clase_estable = {tid: Counter(v).most_common(1)[0][0] for tid, v in class_history.items()}
```

Con eso, el `id:3` de `people-detection` sería `person` por 51 contra 5, el `id:7` por 69 contra 2 y el `id:16` por 64 contra 7. **Un voto mayoritario corrige el 100 % de los casos de ese video**, porque la clase correcta siempre es la mayoritaria.

En `car-detection` no funcionaría: ahí la clase mayoritaria *es* la equivocada, y ningún posprocesamiento sobre las etiquetas puede recuperar una información que el detector nunca produjo. La corrección, en ese caso, es la de la Actividad 2 — decirle al modelo qué se está buscando.

## El falso positivo que ocupa el cuadro entero

Un caso aparte, visible en el frame 327 de `car-detection`: una caja etiquetada `sink` con confianza 0,18 que cubre el **98,4 %** del cuadro.

No llegó a fundar un track persistente —el mecanismo de confirmación la eliminó—, pero ilustra el régimen en que trabaja el detector con `conf = 0,1`: por debajo de 0,25 hay hipótesis que no guardan ninguna relación con la escena. Los **22 nacimientos abortados** de `people-detection` y los **36** de `person-bicycle-car` son exactamente eso: detecciones que superaron el umbral de creación de tracks y no volvieron a aparecer. El filtro de *unconfirmed* las contuvo a todas antes de que llegaran a la pantalla.

## Resumen de la taxonomía

| Categoría | Peso medido | Dónde se origina |
|---|---|---|
| **Detección** — falsos positivos | 36 nacimientos abortados en un video | `conf = 0,1` + escena fuera de distribución |
| **Detección** — duplicados | hasta 13 pares con IoU > 0,8 | sin NMS no hay supresión |
| **Detección** — falsos negativos | 7 rescates en 9 tracks | objetos entrando y saliendo de cuadro |
| **Clasificación** | 5 de 15 tracks multiclase; clase mayoritaria errónea | punto de vista + competencia entre 80 clases |
| **Asignación** | 5 tracks efímeros; 1 ID switch | política de admisión + asignación global |

> La detección falla mucho y el sistema se recupera. La asignación falla poco y no se recupera. La clasificación falla en silencio: no rompe nada, y por eso pasa desapercibida hasta que alguien lee las etiquetas.

---

**Siguiente:** [Vocabulario abierto y segmentación](../06-vocabulario-abierto-y-segmentacion) — qué pasa cuando se le dice al modelo qué buscar.
