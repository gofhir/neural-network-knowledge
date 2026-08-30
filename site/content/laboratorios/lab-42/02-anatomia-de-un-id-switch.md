---
title: "Anatomía de un ID switch"
weight: 2
---

El video tiene **7 personas** y el sistema emite **9 identidades**. Una de las dos de más es un track espurio de 2 frames. La otra es un ID switch, y rastrearlo hasta su origen es lo más instructivo del laboratorio: la causa no está donde parece, y por eso ninguno de los cuatro ajustes obvios lo corrige.

## El evento

Instrumentando el bucle para registrar `frame, id, conf, cls, x, y, w, h` en cada detección, el episodio queda así:

| Frame | Persona A (ropa negra) | Persona B (ropa gris) |
|---|---|---|
| 232 | `id:9` conf 0,77 | `id:7` conf 0,75 |
| 234 | `id:9` conf 0,90 | `id:7` conf 0,80 |
| 236 | `id:9` conf 0,79 **+ `id:13` conf 0,43** | `id:7` conf 0,85 |
| **237** | **`id:13`** conf 0,71 ⟵ 💥 | `id:7` conf 0,89 |
| 238 | `id:13` conf 0,69 | `id:7` conf 0,83 |
| 240 | `id:13` conf 0,55 | `id:7` conf 0,57 |

![Los frames 232 a 240: el track 9 desaparece y el 13 ocupa su lugar sobre la misma persona](/laboratorios/lab-42/id-switch.jpg)

La misma persona cambia de identidad entre los frames 236 y 237. Es un error de **asignación de identificadores** en estado puro: la detección es correcta en ambos frames —puntajes de 0,79 y 0,71—, la caja está bien puesta, y sin embargo el hilo se corta.

## Lo que los datos descartan

Antes de explicar qué pasó conviene eliminar las dos hipótesis naturales.

**No fue degradación del seguimiento.** El IoU frame a frame del track 9 en sus últimos seis frames:

```
0,81   0,90   0,90   0,92   0,97   0,96
```

Los dos últimos valores son **los más altos de toda su vida**. Y su área se mantuvo estable —de 17.975 a 17.507 píxeles cuadrados, un −2,6 % en siete frames—, de modo que tampoco se estaba encogiendo contra el borde del cuadro.

> El track murió en su mejor momento. No hubo deriva, no hubo umbral cruzado, no hubo pérdida progresiva.

**No fue un umbral de asociación.** Propagando las velocidades observadas para estimar la predicción de Kalman de cada candidato en el frame 237:

| Track | Predicción para f237 | IoU con la detección real | Costo `1 − IoU × 0,71` |
|---|---|---|---|
| **9** (55 frames de vida) | x≈46,8 y≈188,6 w≈98,8 h≈175,9 | ≈ 0,83 | ≈ **0,413** |
| **13** (2 frames de vida) | x=47,9 y=187,0 w=95,8 h=178,3 | ≈ 0,84 | ≈ **0,406** |

Los dos candidatos difieren **en la tercera cifra decimal**. Ambos pasan holgadamente el umbral de 0,8. El problema no era la aceptación: era el desempate.

*(La estimación es aproximada: no se tiene acceso a las covarianzas internas del filtro, así que la predicción se trata como extrapolación lineal. Lo que importa es el orden de magnitud de la diferencia, no su signo exacto.)*

## La causa: una caja emitida dos veces

Volviendo al frame 236, la línea que lo explica todo:

```
id  9   conf=0.79   x=48.7   y=187.7   w=97.3   h=179.9
id 13   conf=0.43   x=47.9   y=187.0   w=95.8   h=178.3
                                        IoU = 0.976
```

Las dos cajas difieren en **0,8 px en x, 0,7 px en y, 1,5 px de ancho y 1,6 px de alto**, sobre una caja de 97 × 180. No son cajas parecidas: son **la misma caja**, emitida dos veces por el detector con dos confianzas distintas.

Un detalle de implementación permite fechar el nacimiento del track fantasma un frame antes de lo que muestra el registro. `_format_output()` filtra por `is_activated`:

```python
return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)
```

y un track recién creado queda con `is_activated = False` —el estado *unconfirmed*— hasta que empareja por segunda vez. Como `id:13` **aparece** en la salida del frame 236, tuvo que nacer en el **235** y confirmarse en el 236. La duplicación empezó un frame antes, en un frame donde el track fantasma era invisible.

## La secuencia completa

| Frame | Detecciones sobre la persona A | Qué ocurre |
|---|---|---|
| 235 | **2** (duplicado) | El track 9 toma una. La otra queda huérfana con puntaje ≥ 0,25 → **nace el track 13**, invisible |
| 236 | **2** (duplicado, IoU 0,976) | El track 9 toma una en la primera asociación; el 13 toma la otra en `_unconfirmed_association` → **se confirma** → dos tracks vivos sobre una persona |
| 237 | **1** | Ambos están ahora en `tracked_stracks` y entran a la **misma** asignación global. Compiten por una sola detección |

En el frame 237 el algoritmo húngaro tiene que dar esa detección a uno de los dos. Elige el de menor costo: **0,406 contra 0,413**. Gana el track de dos frames.

{{< concept-alert type="clave" >}}
**La matriz de costos no contiene ninguna variable que codifique la antigüedad de un track.** `get_dists()` calcula IoU y puntaje, nada más:

```python
def get_dists(self, tracks, detections):
    dists = matching.iou_distance(tracks, detections)
    if self.args.fuse_score:
        dists = matching.fuse_score(dists, detections)
    return dists
```

Un track con 55 frames de historia y uno con 2 compiten en igualdad absoluta. Cuando sus costos empatan hasta el ruido numérico, la identidad de 55 frames se pierde por azar.
{{< /concept-alert >}}

### Por qué el track joven llegó a competir de igual a igual

La clase describe el estado de SORT como $x = [u, v, s, r, \dot u, \dot v, \dot s]$ y señala que **todas las velocidades se inicializan en cero**. El track 13, nacido dos frames antes, no tiene velocidad estimada: su predicción es "donde estaba en el frame anterior". El track 9 arrastra velocidad acumulada y extrapola.

En el borde del cuadro, donde la parte visible de la persona se recorta contra el marco de la puerta, el centro de la caja **deja de avanzar** aunque la persona siga caminando. El objeto aparenta frenarse. En esa situación particular, la predicción sin movimiento del track recién nacido no es peor que la del track consolidado — y basta con que no sea peor para que la asignación global le entregue la detección.

### La conexión con la *matching cascade*

[DeepSORT](/papers/deepsort-wojke-2017) no resuelve una asignación global única: ordena los tracks por **tiempo desde la última observación** y asigna en pasadas sucesivas, de forma que un track recién creado nunca compite de igual a igual contra uno consolidado. ByteTrack y BoT-SORT **eliminaron la cascada** y volvieron a la asignación global. Es una simplificación deliberada —la cascada tiene sus propias patologías, como la compuerta de Mahalanobis que se autodesactiva, medida en la [profundización de la clase](/clases/clase-42/profundizacion)— pero el precio es exactamente este caso.

## Cuatro intentos de arreglo, y los cuatro fallan

| Configuración | Tiempo | IDs | Quemados | ¿Switch? |
|---|---|---|---|---|
| baseline | 15,4 s | 9 | 22 | SÍ |
| `with_reid: True` | 19,0 s (**+23 %**) | 10 ⚠️ | 13 | SÍ |
| ReID + `proximity_thresh: 0.15` | 17,6 s | 9 | 11 | SÍ |
| `match_thresh: 0.9` | 14,7 s | 9 | 19 | SÍ |

Ninguno lo corrige, y la razón es la misma en los cuatro: **gobiernan si un par es aceptable**, y aquí ambos pares eran perfectamente aceptables. El detalle de por qué la re-identificación no puede ayudar merece su propia página: está en [El ReID que no puede ayudar](../03-el-reid-que-no-puede-ayudar).

## La palanca que sí funciona

El eslabón débil no está en la asociación sino en la **política de admisión** entre la detección y la asociación: el umbral que decide si una caja huérfana puede fundar una identidad.

| Configuración | IDs | Quemados | Duplicados | Switch | Cobertura |
|---|---|---|---|---|---|
| YOLO26 baseline | 9 | 22 | 1 | SÍ | 3 tracks con huecos |
| YOLO26 + `new_track_thresh 0.55` | 8 | 2 | 0 | **NO** | — |
| YOLO11 (con NMS) | 11 | 21 | 2 | SÍ | — |
| **YOLO11 + `new_track_thresh 0.55`** | **7** | **2** | **0** | **NO** | **100 % en las 7** |

**Siete identidades para siete personas, con cobertura perfecta y sin costo computacional** (14,4 s contra 14,7 del baseline).

Y la verificación de que no se perdió nada: los siete tracks cubren los cuatro eventos del video, el track espurio desaparece, y el que estaba partido se fusiona en uno solo que además **llega dos frames más lejos** que la suma de sus partes:

| Track (arreglado) | Rango | Cobertura | Corresponde a (baseline) |
|---|---|---|---|
| `id 1` | f22–75 | 100 % | `id 3` (f19–75, 98,2 %) |
| `id 2` | f182–252 | 100 % | `id 7` (f180–250) |
| **`id 3`** | **f182–242** | **100 %** | **`id 9` + `id 13` ← el switch, reparado** |
| `id 4` | f341–403 | 100 % | `id 15` |
| `id 5` | f343–413 | 100 % | `id 16` (98,6 %) |
| `id 7` | f355–401 | 100 % | `id 21` |
| `id 9` | f524–560 | 100 % | `id 31` |

El mecanismo es directo: los 22 "IDs quemados" del baseline eran, por definición, tracks nacidos con puntaje entre 0,25 y 0,55 que nunca se confirmaron. Subir el umbral por encima de ese rango los elimina de raíz, y con ellos desaparece el competidor que expropió al track 9.

## Una hipótesis intermedia que hubo que descartar

Parecía natural culpar a la arquitectura **NMS-free** de YOLO26: sin supresión de duplicados, nada colapsa dos cajas con IoU 0,976, y un NMS con el umbral estándar de 0,7 las habría fusionado sin dudarlo.

El experimento de control con YOLO11 —que sí tiene NMS— **refutó la hipótesis**: produjo el mismo switch y más identidades espurias (11 contra 9).

El motivo del fallo experimental también quedó claro, y conviene consignarlo. La métrica de duplicados mide solape entre las **salidas del tracker** —estados filtrados de Kalman—, no entre las detecciones crudas. Dos tracks alimentados por detecciones bien separadas pueden converger al mismo estado tras unos frames de filtrado, y ahí el NMS ya no tiene jurisdicción. **El experimento no medía lo que se pretendía medir.**

## La cadena causal

```
POLÍTICA DE ADMISIÓN (new_track_thresh = 0,25, demasiado laxa para esta escena)
   └─ una detección duplicada de puntaje 0,43 sobre un objeto YA seguido funda un track paralelo
         │  nada verifica si esa caja solapa con un track vivo
         ↓
   ASOCIACIÓN (asignación global única, sin matching cascade)
   └─ dos tracks compiten por una detección; costos 0,406 contra 0,413
         │  la matriz de costos no codifica antigüedad
         ↓
   ID SWITCH IRREVERSIBLE — el track de 55 frames muere con IoU 0,96
```

> Un error de detección se propagó dos etapas y se manifestó como un error de asignación. Por eso ningún parámetro del asociador podía corregirlo, y por eso el diagnóstico correcto vale más que el ajuste de parámetros.

## El detector que parpadea, y la identidad que sobrevive

Conviene cerrar con el caso contrario, porque muestra el mismo mecanismo funcionando bien. Entre los frames 70 y 76, la primera persona del video sale por el borde derecho:

| Frame | Detección |
|---|---|
| 70 | `id:3 person 0.87` ✓ |
| 72 | `id:3 person 0.58` ✓ |
| 73 | `id:3 cat 0.49` ✓ |
| **74** | **nada** ✗ — la persona está visible |
| 75 | `id:3 cat 0.44` ✓ |
| **76** | **nada** ✗ — sigue visible |

![Frames 70 a 76: el detector prende y apaga, y el identificador se mantiene](/laboratorios/lab-42/parpadeo.jpg)

El detector prende y apaga en frames alternos, y sin embargo **el identificador se mantiene en 3 durante todo el episodio**. Es `track_buffer` trabajando: en el frame 74 el track pasa a `lost_stracks`, el filtro sigue prediciendo un frame con velocidad constante, y al reaparecer la detección `re_activate()` lo revive conservando `track_id`.

En todo el video hay solo **3 huecos, de un frame cada uno**, sobre 7 tracks con cobertura de 98,2 a 100 %. El filtro de Kalman convierte un detector intermitente en identidades continuas — y, como muestra este mismo laboratorio, sostiene con igual diligencia las 22 identidades que nunca existieron.

---

**Siguiente:** [El ReID que no puede ayudar](../03-el-reid-que-no-puede-ayudar) — por qué activar la apariencia no cambia nada, y qué sí cambia.
