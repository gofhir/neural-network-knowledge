---
title: "El pipeline por dentro"
weight: 1
---

El notebook resuelve el seguimiento en una línea:

```python
results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
```

Detrás de esa línea hay tres decisiones que el laboratorio no menciona y que determinan todo lo que se observa después. Las tres se verifican en el código fuente de Ultralytics.

## El video, medido

Antes del pipeline conviene fijar el material. `people-detection.mp4` viene del repositorio de demos de OpenVINO de Intel, y sus propiedades no son las de un video cualquiera:

| Propiedad | Valor |
|---|---|
| Resolución | 768 × 432 |
| Códec | H.264 (`avc1`) |
| **Frame rate** | **12,0 fps exactos** |
| Frames | 596 |
| Duración | 49,67 s |

Los 12 fps salen de leer el átomo `stts` del contenedor: una sola entrada `(596 muestras, delta = 2000)` con `timescale = 24000`. Sin *drift*, sin fracciones.

Ese número importa más de lo que parece. La asociación por IoU depende de cuánto se mueve un objeto **respecto de su propio tamaño**. Una persona de pie ocupa unos 50 px de ancho a esta resolución; caminando a 1,4 m/s se desplaza

$$\frac{1{,}4\ \text{m/s}}{12\ \text{fps}} = 0{,}117\ \text{m} \approx 10\ \text{px por frame}$$

lo que deja un solapamiento entre frames consecutivos de

$$\text{IoU} = \frac{50-10}{50+10} \approx 0{,}67$$

A 30 fps ese mismo peatón se desplazaría 4 px y el IoU sería 0,85. **A 12 fps el video está 2,5 veces más cerca del punto de ruptura** —que es cuando el desplazamiento iguala el ancho de la caja— de lo que estaría material grabado a velocidad normal. No es un video fácil disfrazado de difícil: es lo contrario.

Dos correcciones a ese cálculo, porque se usa mal con facilidad. La primera es que BoT-SORT no compara la caja anterior contra la detección nueva, sino **la predicción del filtro de Kalman** contra la detección nueva; con un peatón caminando recto el residuo es de uno o dos píxeles y el IoU efectivo sube mucho. El cálculo de arriba es el escenario *sin* modelo de movimiento, el punto de referencia contra el cual medir lo que aporta el filtro. La segunda aparece en la última sección de esta página.

## Sorpresa 1: `conf` baja automáticamente a 0,1

`Model.track()` no es un envoltorio inocuo. Hace esto:

```python
register_tracker(self, persist)
kwargs["conf"] = kwargs.get("conf") or 0.1   # ← ByteTrack-based method needs low confidence predictions
kwargs["batch"] = kwargs.get("batch") or 1
kwargs["mode"] = "track"
return self.predict(source=source, stream=stream, **kwargs)
```

En modo `predict` el umbral de confianza por defecto es **0,25**. En modo `track`, Ultralytics lo **fuerza a 0,1**.

No es un detalle de configuración: es lo que hace posible el algoritmo. La contribución central de [ByteTrack](/papers/bytetrack-zhang-2021) es no descartar las detecciones de puntaje bajo sino usarlas en una segunda ronda de asociación. Si el detector filtrara a 0,25 antes de entregar las cajas, esa segunda ronda quedaría siempre vacía y BoT-SORT degeneraría en [SORT](/papers/sort-bewley-2016).

El efecto se ve en los datos: la confianza mínima observada en la ejecución es **0,107**, y en el frame 400 aparecen simultáneamente tres personas saliendo de cuadro con puntajes de 0,46, **0,11** y 0,32. Con `predict()` y su umbral por defecto, esas tres cajas serían una sola y los tres tracks morirían ahí.

## Sorpresa 2: `persist=True` es lo único que distingue seguir de detectar

El seguimiento no vive dentro del modelo: se implementa con **callbacks**.

- `on_predict_start` crea las instancias del tracker. Con `persist=True` retorna de inmediato si ya existen: `if hasattr(predictor, "trackers") and persist: return`.
- `on_predict_postprocess_end` toma las cajas del frame y llama a `tracker.update(...)`, que devuelve las cajas **con identificador**.

Como el bucle del laboratorio llama a `model.track(frame)` una vez por frame —con un `ndarray` suelto, no con la ruta del video—, cada llamada es un "video nuevo" desde el punto de vista de Ultralytics. Sin `persist=True` el tracker se reinstanciaría en cada iteración y **todos los objetos tendrían identificador 1, 2, 3… en cada frame**, sin ninguna continuidad. Esa palabra clave es la única línea que convierte 596 detecciones independientes en trayectorias.

Corolario práctico, y no menor para la actividad: **al cambiar de video hay que reiniciar el estado**. Si se corre el bucle sobre un segundo video sin recrear el modelo, los identificadores siguen desde donde quedó el anterior y los tracks del primero intentan emparejarse con objetos del segundo durante los primeros frames.

## Sorpresa 3: lo que corre no es BoT-SORT

`tracker="botsort.yaml"` carga esta configuración, que es la del repositorio de Ultralytics:

```yaml
tracker_type: botsort
track_high_thresh: 0.25      # umbral de la primera asociación
track_low_thresh: 0.1        # piso de la segunda asociación
new_track_thresh: 0.25       # puntaje mínimo para nacer como track
track_buffer: 30             # frames que sobrevive un track perdido
match_thresh: 0.8            # umbral de costo en la 1ª asociación
fuse_score: True             # multiplica IoU por el puntaje del detector
gmc_method: sparseOptFlow    # compensación de movimiento de cámara
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: False             # ← re-identificación DESACTIVADA
model: auto
```

{{< concept-alert type="clave" >}}
**`with_reid: False` por defecto.** El paper de BoT-SORT combina tres ingredientes: un filtro de Kalman reparametrizado, compensación de movimiento de cámara, y un modelo de apariencia que se fusiona con el IoU. Con la re-identificación apagada quedan los dos primeros.

Lo que corre es, en esencia, **[ByteTrack](/papers/bytetrack-zhang-2021) más compensación de cámara**. Toda la asociación se decide por **geometría**: dónde predice el filtro que debería estar la caja. Ninguna información visual sobre *quién* es cada persona entra en la decisión.
{{< /concept-alert >}}

Esto da una predicción falsable que conviene tener en mente al mirar el video: dos personas con ropa de colores muy distintos que se crucen deberían intercambiar identidades con la misma facilidad que dos personas idénticas, porque el color no participa de la ecuación. El experimento de activar la apariencia está en [la página 3](../03-el-reid-que-no-puede-ayudar).

## El pipeline de asociación, con los umbrales reales

```
frame t
  │
  ├─ YOLO26 → detecciones con conf ≥ 0,1
  │
  ├─ SPLIT por puntaje
  │     ├─ altas:  score ≥ 0,25
  │     └─ bajas:  0,10 < score < 0,25
  │
  ├─ KALMAN predict   (todos los tracks: activos + perdidos)
  │     BOTrack usa KalmanFilterXYWH — estado 8D (x, y, w, h, ẋ, ẏ, ẇ, ḣ)
  │
  ├─ GMC (sparseOptFlow): homografía cámara t-1→t, corrige los estados predichos
  │
  ├─ ASOCIACIÓN 1: detecciones altas ↔ (tracks activos + perdidos)
  │     costo = 1 − (IoU × score)      ← fuse_score = True
  │     Húngaro con umbral 0,8         → acepta si costo < 0,8
  │
  ├─ ASOCIACIÓN 2: detecciones bajas ↔ tracks aún sin emparejar
  │     costo = 1 − IoU                ← SIN fuse_score
  │     Húngaro con umbral 0,5 (fijo en el código)
  │     Los que siguen sin match → mark_lost()
  │
  ├─ ASOCIACIÓN 3: detecciones sobrantes ↔ tracks "unconfirmed" (nacidos en t−1)
  │     umbral 0,7 (fijo). Los unconfirmed sin match → mark_removed()
  │
  ├─ NACIMIENTOS: detecciones sobrantes con score ≥ 0,25 → track nuevo
  │
  └─ MUERTES: tracks perdidos hace más de track_buffer = 30 frames → removed
```

Cuatro observaciones que salen de leer el código y no la documentación.

**La segunda etapa no fusiona el puntaje, y es deliberado.** `_second_association` usa `iou_distance` puro. Tiene que ser así: esas detecciones tienen puntaje bajo *por definición*, y multiplicar por él las mataría a todas, anulando la contribución de ByteTrack. A cambio el umbral se endurece de 0,8 a 0,5 — se exige más evidencia geométrica porque hay menos evidencia de confianza. Es un intercambio explícito.

**`match_thresh: 0.8` es permisivo, no estricto.** El comentario del YAML habla de "similitud", pero el código compara **costos**: `linear_assignment(dists, thresh=0.8)` rechaza pares con `costo > 0,8`. Como `costo = 1 − IoU`, ese umbral equivale a aceptar cualquier par con **IoU > 0,2**. Es un error de lectura muy común.

**`fuse_score` acopla el detector con el asociador.** Ésta es la segunda corrección al cálculo del principio. La similitud efectiva no es el IoU sino `IoU × score`. Una persona con IoU predicho de 0,66 pero puntaje 0,3 —semiocluida— produce una similitud fusionada de 0,198, es decir costo 0,802, y queda **rechazada**. La misma persona con puntaje 0,9 da costo 0,41 y pasa con holgura. Una oclusión que baja la confianza del detector también rompe la asociación: los dos componentes se penalizan mutuamente.

**`track_buffer: 30` no es un tiempo.** Un track perdido sobrevive 30 frames, que a 12 fps son **2,5 segundos** y a 60 fps son 0,5. Tiene dos caras: tolera oclusiones largas, y durante todo ese lapso el filtro sigue extrapolando en línea recta. Si el objeto cambió de dirección detrás del obstáculo, la caja fantasma queda lejos de donde reaparece. Es el mismo parámetro con signo ambiguo que en la [profundización de la clase](/clases/clase-42/profundizacion) se midió para el `A_max` de [DeepSORT](/papers/deepsort-wojke-2017): el mismo valor que recupera falsos negativos introduce falsos positivos.

**El estado es `(x, y, w, h)`, no `(x, y, a, h)`.** Detalle fino: la clase base `STrack` de ByteTrack usa la parametrización heredada de SORT, con la razón de aspecto como variable de estado. `BOTrack` la sobreescribe con `KalmanFilterXYWH`, que estima ancho y alto directamente. Es una de las contribuciones del paper de BoT-SORT: modelar el aspecto con velocidad propia degrada las estimaciones de tamaño cuando el objeto rota o se deforma.

## Lo que se ve en el video

![Panorama del video anotado: 596 frames, cuatro eventos de personas cruzando una sala](/laboratorios/lab-42/panorama.jpg)

Cámara fija, sala con piso de madera, personas que cruzan ocasionalmente. Medido sobre el video de salida:

| Métrica | Valor |
|---|---|
| Frames con al menos un track | 235 de 596 (**39,4 %**) |
| Detecciones totales | 403 |
| Identidades emitidas | 9 |
| **Personas reales** (conteo manual) | **7** |

Cuatro eventos: f19-75 (una persona), f180-250 (dos), f339-413 (tres), f523-556 (una).

Un apunte sobre la cámara fija: el módulo de compensación de movimiento está activo y estimando una homografía prácticamente igual a la identidad durante todo el video. No hace daño, pero tampoco aporta — su razón de existir es el caso de cámara móvil, que ninguno de los videos de este laboratorio ejercita.

---

**Siguiente:** [Anatomía de un ID switch](../02-anatomia-de-un-id-switch) — de dónde salen las 9 identidades cuando hay 7 personas.
