---
title: "Lab 42 - Tracking en video: YOLO26 + BoT-SORT y YOLO-World + SAM 2"
weight: 420
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga (DCC, Pontificia Universidad Católica de Chile)
**Módulo:** Video — seguimiento multi-objeto
**Notebook origen:** `clase_42/material/Laboratorio/Practico_42.ipynb`
**Notebook ejecutado:** [lab42.ipynb](/notebooks/lab42.ipynb) · [HTML](/notebooks-html/lab42.html)

## Encuadre

La contraparte práctica de la [clase 42](/clases/clase-42). Doce celdas, dos pipelines y ninguna línea de algoritmo que escribir: el laboratorio se resuelve llamando a `model.track()` sobre un video de vigilancia y mirando el resultado.

Lo interesante no es el código sino **lo que el resultado permite diagnosticar**. Con el bucle instrumentado para registrar `frame, id, conf, cls, x, y, w, h` en cada detección, el video del enunciado deja de ser una demo y se vuelve un caso: 596 frames, 7 personas reales, **9 identidades emitidas**, y un cambio de identidad que se puede rastrear hasta el frame exacto y la caja exacta que lo causó.

{{< concept-alert type="clave" >}}
**El ID switch no se originó en el asociador.** En el frame 236 el detector emitió **dos cajas con IoU = 0,976** sobre la misma persona —0,8 px de diferencia en $x$ sobre una caja de 97 × 180—. La caja sobrante fundó un track paralelo, y un frame después los dos compitieron por una única detección con costos de **0,406 contra 0,413**. Ganó el track de dos frames sobre el de cincuenta y cinco, que murió con **el IoU más alto de toda su vida (0,96)**.

Cuatro ajustes del asociador fallaron en corregirlo, incluida la re-identificación por apariencia. Un solo parámetro de la **política de admisión de tracks** lo eliminó, y llevó el video a **7 identidades para 7 personas con 100 % de cobertura**, sin costo computacional.
{{< /concept-alert >}}

La tesis del laboratorio, en una línea: **un error de detección se propaga dos etapas y se manifiesta como un error de asignación**, y por eso el esquema de cuatro componentes de la clase —localización, representación, asociación, similaridad— vale más como herramienta de diagnóstico que como descripción.

![Panorama del video anotado: cuatro eventos de personas cruzando una sala](/laboratorios/lab-42/panorama.jpg)

## Resultados consolidados (medidos)

### El video del enunciado: `people-detection.mp4`

| Configuración | IDs | Quemados | Dup. | ID switch | Cobertura |
|---|---|---|---|---|---|
| YOLO26 + BoT-SORT (baseline) | 9 | 22 | 1 | **SÍ** | 3 tracks con huecos |
| `with_reid: True` | 10 ⚠️ | 13 | — | SÍ | — |
| ReID + `proximity_thresh 0.15` | 9 | 11 | — | SÍ | — |
| `match_thresh 0.9` | 9 | 19 | — | SÍ | — |
| YOLO26 + `new_track_thresh 0.55` | 8 | 2 | 0 | **NO** | — |
| **YOLO11 + `new_track_thresh 0.55`** | **7** | **2** | **0** | **NO** | **100 % en las 7** |

7 personas reales, contadas manualmente. La re-identificación cuesta **+23 % de tiempo** y no toca el switch: `proximity_thresh` confina la apariencia a pares con IoU ≥ 0,5, y el episodio ocurrió con IoU ≈ 0,28.

### Actividad 1 — tres videos nuevos con YOLO26

| Configuración | IDs | Quemados | Dup. | Multiclase | Rescates | conf mín |
|---|---|---|---|---|---|---|
| V1 · `car-detection` | 9 | 10 | 11 | 4 | 7 | 0,106 |
| V1 · `new_track_thresh 0.55` | **4** | **2** | **0** | 4 | 3 | 0,103 |
| V2a · `store-aisle` @ 59,94 fps | 8 | 16 | 12 | 0 | 2 | 0,108 |
| V2b · el mismo tramo a ~15 fps | **3** | **5** | **0** | 0 | 1 | 0,108 |
| V3 · `person-bicycle-car` | 15 | 36 | 13 | 5 | 5 | 0,100 |
| V3 · `new_track_thresh 0.55` | **10** | **2** | **1** | 4 | 2 | 0,100 |

Dos resultados que salieron al revés de lo previsto. **Submuestrear de 60 a 15 fps mejoró el seguimiento** —las mismas tres personas con 99-100 % de cobertura, cinco identidades espurias menos y cuatro veces más rápido—, porque el tracker ejecuta una actualización por frame y cuadruplicar los frames cuadruplica las ocasiones de fundar una identidad falsa. Y el umbral de admisión que arregló `people-detection` **cuesta 23 % de la evidencia en V1 y solo 1,5 % en V3**: no es un valor universal.

### Actividad 2 — YOLO-World + SAM 2

| Caso | Prompts | IDs | Quemados | Dup. | Multiclase | conf mín | Llenado |
|---|---|---|---|---|---|---|---|
| 2a · objeto único | `["person"]` | 2 | 1 | 0 | 0 | 0,379 | 0,536 |
| 2b · objeto múltiple | `["person"]` | 7 | **0** | 0 | 0 | 0,296 | 0,526 |
| 2c · multiprompt | `["person","bicycle","car"]` | 4 | **0** | 0 | 0 | 0,306 | 0,570 |
| bonus · objetos idénticos | `["bottle"]` | 4 | **0** | 0 | 0 | **0,696** | **0,761** |

La comparación que ordena todo lo anterior, **misma escena y mismo tracker**, cambiando solo el detector:

| | `person-bicycle-car` · YOLO26 (80 clases) | · YOLO-World (3 prompts) |
|---|---|---|
| Identidades emitidas | 15 | **4** |
| Nacimientos espurios | 36 | **0** |
| Cajas duplicadas | 13 | **0** |
| Tracks con clase inestable | 5 | **0** |
| Clases alucinadas | `boat`, `cell phone`, `sports ball`, `tennis racket` | **ninguna** |
| conf mínima | 0,100 | 0,306 |

### El video de autos cuya clase mayoritaria es `cell phone`

De las 121 detecciones de `car-detection`, unas 21 dicen `car`. Hay tracks enteros etiquetados `sink`. **No es un problema de escala**: las cajas mal etiquetadas miden hasta **370 × 319 px, el 51 % del cuadro**. La cámara es cenital, y un automóvil visto desde arriba es un rectángulo liso y brillante sobre fondo uniforme.

Y la prueba de que tampoco es un problema de percepción: **el mismo auto, en la Actividad 2, es `car` con 0,85**; la misma bicicleta que YOLO26 llama `tennis racket` con 0,40 es `bicycle` con **0,89**. Lo que falla es la competencia entre 80 hipótesis, no la evidencia visual.

### El costo de la segmentación

Ajustando un modelo de dos costos a seis tramos medidos (error < 5 %):

$$\text{frame vacío} \approx 12\ \text{ms} \qquad \text{frame con objetos} \approx 212\ \text{ms}$$

**SAM 2 cuesta unas 17 veces lo que el detector** — y en 1.790 detecciones no produjo ni una máscara vacía ni una pobre. Pero `task_map` selecciona `SAM2Predictor`, el predictor de **imágenes**: el banco de memoria, la atención temporal y la cabeza de oclusión **nunca se ejecutan**. Toda la continuidad temporal del pipeline proviene del filtro de Kalman de BoT-SORT.

## Bloques del lab

{{< cards >}}
  {{< card link="01-el-pipeline-por-dentro" title="El pipeline por dentro" subtitle="Los 12 fps exactos del video y por qué importan, el conf=0.1 que Ultralytics fuerza en modo track, el persist=True que es lo único que distingue seguir de detectar, y el with_reid: False que convierte BoT-SORT en ByteTrack" icon="cube-transparent" >}}
  {{< card link="02-anatomia-de-un-id-switch" title="Anatomía de un ID switch" subtitle="La caja duplicada con IoU 0,976, los costos que difieren en la tercera cifra decimal, el track que muere con su mejor IoU, los cuatro arreglos que fallan y el parámetro que sí funciona" icon="beaker" >}}
  {{< card link="03-el-reid-que-no-puede-ayudar" title="El ReID que no puede ayudar" subtitle="La línea emb_dists[dists_mask] = 1.0 que confina la apariencia a pares con IoU ≥ 0,5, el mismo patrón que la compuerta de Mahalanobis de DeepSORT, y el régimen distinto donde el ReID sí actúa" icon="adjustments" >}}
  {{< card link="04-el-frame-rate-como-dificultad" title="El frame rate como parámetro de dificultad" subtitle="El mismo tramo a 60 y a 15 fps: 8 identidades contra 3, 12 duplicados contra 0, y cuatro veces más rápido. Por qué la predicción teórica sale al revés" icon="trending-down" >}}
  {{< card link="05-lo-que-el-detector-nombra-mal" title="Lo que el detector nombra mal" subtitle="Un video de autos cuya clase mayoritaria es cell phone, la vista cenital como desajuste de distribución, y el asociador class-agnostic que preserva la identidad mientras la etiqueta oscila" icon="photograph" >}}
  {{< card link="06-vocabulario-abierto-y-segmentacion" title="Vocabulario abierto y segmentación" subtitle="La reparametrización de set_classes(), la confianza mínima que nunca baja de 0,30, y SAM 2 ejecutado como si fuera SAM 1 — sin memoria, sin atención temporal, sin cabeza de oclusión" icon="sparkles" >}}
  {{< card link="07-los-defectos-del-notebook" title="Los defectos del notebook" subtitle="Nueve defectos: el wget que renombra, el ffmpeg sin -y, el int(fps) que rompe 4 de 6 videos, las estelas desde el torso, y los dos estados globales que sobreviven entre celdas" icon="exclamation" >}}
{{< /cards >}}

## Clase y fundamentos

{{< cards >}}
  {{< card link="/clases/clase-42" title="Clase 42 - Tracking de Objetos en Video" subtitle="El marco teórico: espacial contra espacio-temporal, offline contra online, SORT y DeepSORT paso a paso, y los modelos integrados" icon="academic-cap" >}}
  {{< card link="/clases/clase-42/profundizacion" title="Profundización de la clase" subtitle="La aritmética de MOTA reconstruida, la compuerta de Mahalanobis que se autodesactiva, y cuándo el húngaro le gana al codicioso" icon="beaker" >}}
  {{< card link="/fundamentos/seguimiento-de-objetos" title="Seguimiento de Objetos" subtitle="SOT contra MOT, tracking-by-detection, y la anatomía común a todos los algoritmos" icon="book-open" >}}
  {{< card link="/fundamentos/filtro-de-kalman" title="Filtro de Kalman" subtitle="Predicción y corrección, y las dos patologías que aparecen en seguimiento" icon="book-open" >}}
  {{< card link="/fundamentos/asignacion-hungara" title="Asignación Húngara" subtitle="Por qué el codicioso no basta, y los cuatro ajustes que exige el caso real" icon="book-open" >}}
  {{< card link="/fundamentos/re-identificacion" title="Re-identificación" subtitle="El descriptor de conjunto abierto que atraviesa las oclusiones" icon="book-open" >}}
  {{< card link="/fundamentos/metricas-de-tracking" title="Métricas de Tracking" subtitle="MOTA, IDF1 y HOTA: qué mide cada una y dónde falla" icon="book-open" >}}
  {{< card link="/fundamentos/deteccion-de-objetos" title="Detección de Objetos" subtitle="IoU, NMS y anchors: la etapa que alimenta el pipeline y acota su rendimiento" icon="book-open" >}}
{{< /cards >}}

## Papers que aparecen en el laboratorio

{{< cards >}}
  {{< card link="/papers/sort-bewley-2016" title="SORT (2016)" subtitle="Bewley et al. — el estado con velocidades inicializadas en cero, que es lo que explica por qué un track de dos frames le gana a uno de cincuenta y cinco" icon="document-text" >}}
  {{< card link="/papers/deepsort-wojke-2017" title="DeepSORT (2017)" subtitle="Wojke et al. — la matching cascade que evitaría este ID switch, y la compuerta que en los experimentos deja λ=0" icon="document-text" >}}
  {{< card link="/papers/bytetrack-zhang-2021" title="ByteTrack (2021)" subtitle="Zhang et al. — la segunda ronda de asociación con detecciones de score bajo, que es la razón por la que Ultralytics fuerza conf=0.1" icon="document-text" >}}
  {{< card link="/papers/oc-sort-cao-2022" title="OC-SORT (2022)" subtitle="Cao et al. — el filtro que se realimenta su error durante la oclusión. En DanceTrack, DeepSORT queda por debajo de SORT" icon="document-text" >}}
  {{< card link="/papers/sam3-meta-2025" title="SAM 3 (2025)" subtitle="Meta — seguir todas las instancias de un concepto dado en lenguaje natural: el paradigma que la Actividad 2 aproxima con YOLO-World + SAM 2" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Lab 40 - TSM](/laboratorios/lab-40) y [Lab 38 - I3D](/laboratorios/lab-38) (la otra mitad del análisis de video) · [Lab 15 - Faster R-CNN](/laboratorios/lab-15) (la etapa de detección que alimenta este pipeline) · Dominio [Video](/dominios/video).
