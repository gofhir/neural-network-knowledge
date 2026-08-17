---
title: "Seguimiento de Objetos"
weight: 132
math: true
---

El **seguimiento de objetos** (*object tracking*) es la tarea de asociar detecciones a lo largo del tiempo preservando la **identidad** de cada objeto. Detectar responde *qué hay y dónde*; seguir agrega *y es el mismo de antes*. Ese "el mismo" es todo el problema: no se observa directamente en ningún frame, hay que inferirlo.

Este fundamento acompaña a la [Clase 42](/clases/clase-42) y consolida el vocabulario del área: la distinción entre razonamiento espacial y espacio-temporal, las variantes de la tarea, el paradigma *tracking-by-detection*, la división online/offline, y la anatomía común a todos los algoritmos.

---

## 1. Detección contra seguimiento

La clase abre con una oposición que conviene fijar antes que nada:

| | Detección de objetos | Seguimiento de objetos |
|---|---|---|
| Razonamiento | **espacial** | **espacio-temporal** |
| Entrada | snapshots aisladas | secuencia ordenada |
| Salida | lista de cajas por frame | lista de **trayectorias** |
| Requiere | localizar y clasificar | además, **preservar identidad**, modelar dinámica y **recuperar la asociación tras oclusión** |

Un detector aplicado frame a frame ya entrega cajas correctas en cada instante. Lo que **no** entrega es el hilo que une la caja del frame $t$ con la del frame $t+1$. Ese hilo —la identidad— es lo que el tracker construye.

{{< concept-alert type="clave" >}}
El seguimiento **no es un problema de percepción sino de asociación**. Si el detector es perfecto, sigue faltando decidir cuál de las $M$ cajas nuevas corresponde a cuál de las $N$ trayectorias activas. Es un problema combinatorio, no visual.
{{< /concept-alert >}}

## 2. Una tarea, muchas variantes

"Seguimiento de objetos" nombra una familia, no un problema único. Los ejes que la clase enumera:

- **Un objeto contra múltiples objetos.** SOT (*single object tracking*) recibe una caja en el primer frame y la sigue; el modelo no necesita detectar nada, solo re-localizar un *template*. MOT (*multiple object tracking*) debe descubrir objetos que entran y salen, y mantener un número variable de identidades. Son literaturas casi disjuntas: [SUTrack](/papers/sutrack-chen-2024) es SOT; [SORT](/papers/sort-bewley-2016) y [DeepSORT](/papers/deepsort-wojke-2017) son MOT.
- **Una cámara contra múltiples cámaras.** Con varias vistas aparece la asociación *entre* cámaras, donde la geometría ya no ayuda (los campos de visión pueden no solaparse) y solo queda la apariencia. Es el escenario que originó [IDF1](/papers/idf1-ristani-2016).
- **Cámara estática contra cámara dinámica.** Con cámara fija, el movimiento en la imagen es el movimiento del objeto. Con cámara móvil, cada píxel se desplaza aunque nada se mueva, y todo modelo de movimiento hereda un sesgo que hay que compensar.

## 3. Por qué es difícil

La clase lista los desafíos y luego insiste, diapositiva tras diapositiva, en que **ocurren todos a la vez**:

- cambios de iluminación,
- variaciones de pose,
- **oclusiones** (parciales, totales, prolongadas),
- variaciones de escala (un peatón a 20 px y a 500 px),
- deformaciones,
- variaciones intra-clase,
- restricciones de tiempo real,
- muchos objetos simultáneos.

De todos, el que estructura la literatura es la **oclusión**: es el único que rompe la continuidad de la observación. Los demás degradan la señal; la oclusión la elimina, y obliga al sistema a sostener una identidad sin evidencia durante $k$ frames. Casi toda la evolución de MOT desde 2016 puede leerse como respuestas sucesivas a esa pregunta.

## 4. Tracking-by-detection

El paradigma dominante desde ~2015 descompone el problema en dos etapas independientes:

1. **Detectar** los objetos en cada frame con un detector entrenado por separado ([Faster R-CNN](/papers/faster-rcnn-ren-2015), YOLO, etc.).
2. **Asociar** las detecciones entre frames para formar trayectorias.

La consecuencia práctica es que la calidad del tracker está acotada por la del detector. [SORT](/papers/sort-bewley-2016) lo demuestra con el experimento más citado del área: manteniendo fijo el algoritmo de seguimiento y cambiando solo el detector (ACF → Faster R-CNN con VGG16), MOTA pasa de **15,1 a 34,0** — más del doble.

{{< concept-alert type="recordar" >}}
En *tracking-by-detection*, **buena parte de lo que parece un problema de seguimiento es un problema de detección**. Antes de sofisticar la asociación, conviene medir cuánto rinde el mismo tracker con un detector mejor.
{{< /concept-alert >}}

El paradigma alternativo, **joint detection and tracking**, entrena una sola red que produce cajas y descriptores de identidad a la vez ([FairMOT](/papers/fairmot-zhang-2020)) o que usa el regresor del detector como modelo de movimiento ([Tracktor](/papers/tracktor-bergmann-2019)).

## 5. Online contra offline

La distinción operativa que ordena los algoritmos:

- **Online tracking**: en el frame $t$ solo se dispone de los frames $\leq t$. La identidad debe emitirse ahora y no puede revisarse. Es el régimen de conducción autónoma, robótica y vigilancia en vivo.
- **Offline (o batch) tracking**: se dispone del video completo. La asociación puede plantearse como una optimización global sobre todo el grafo espacio-temporal —flujo en redes, *min-cost flow*, MHT con ventana completa— y una decisión del frame 10 puede corregirse con evidencia del frame 300.

Offline es estrictamente más informado y por eso, históricamente, más preciso. Online es el que tiene casos de uso masivos. Una tercera categoría, **near-online**, permite mirar unos pocos frames futuros con latencia acotada.

En el marco offline el problema se dibuja naturalmente como un **grafo**: los nodos son detecciones, las aristas candidatas conectan detecciones de frames distintos, y una trayectoria es un camino. La asociación se vuelve entonces "encontrar un conjunto de caminos disjuntos de costo mínimo", y lo que hay que aprender es el **costo de las aristas**: una métrica de distancia entre detecciones. Ahí entran las [redes siamesas](/papers/siamese-networks-koch-2015) y el [triplet loss](/fundamentos/triplet-loss).

## 6. La anatomía de un tracker online

Todo tracker online, por distinto que se vea, tiene los mismos cuatro componentes. La clase los enumera así:

**1. Detección de objetos**
  - 1.1 **Localización en el espacio** — dónde está el objeto (el detector).
  - 1.2 **Representación del objeto** — con qué se lo describe: la caja sola, un histograma, un *embedding* aprendido.

**2. Búsqueda de objetos**
  - 2.1 **Asociación de datos / modelo de movimiento** — dónde *debería* estar en el próximo frame ([filtro de Kalman](/fundamentos/filtro-de-kalman), velocidad constante, flujo óptico).
  - 2.2 **Medida de similaridad** — cuánto se parece lo predicho a cada detección nueva (IoU, [Mahalanobis](/fundamentos/filtro-de-kalman), coseno entre *embeddings*).

A esto se agrega, en la práctica, un quinto componente que ningún diagrama muestra pero que decide la mitad del rendimiento: la **gestión del ciclo de vida** de las trayectorias —cuándo se crea una identidad nueva, cuánto se la mantiene viva sin detecciones, cuándo se la mata—. En SORT ese parámetro es $T_{\text{lost}} = 1$; en DeepSORT, $A_{\max} = 30$. La diferencia entre ambos explica buena parte de la diferencia entre sus métricas.

Con los componentes 2.1 y 2.2 se construye una **matriz de costo** de $N$ trayectorias por $M$ detecciones, y la asignación óptima se resuelve con el [algoritmo húngaro](/fundamentos/asignacion-hungara).

## 7. Los tres errores que se pueden cometer

Cualquier salida de un tracker falla de exactamente tres maneras, y conviene tenerlas separadas porque las [métricas](/fundamentos/metricas-de-tracking) las pesan distinto:

| Error | Qué es | Métrica que lo captura |
|---|---|---|
| **Detección** | falsos negativos y falsos positivos | DetA, FN/FP en MOTA |
| **Asociación** | la identidad salta de un objeto a otro (*ID switch*) o una trayectoria se parte en dos (*fragmentación*) | AssA, IDF1, IDSW |
| **Localización** | la caja está pero mal ajustada | MOTP, LocA |

MOTA cuenta los tres sumando errores de detección y asociación en una sola fracción, y termina dominado por los de detección, que son órdenes de magnitud más numerosos. [HOTA](/papers/hota-luiten-2020) los separa explícitamente en una media geométrica.

## 8. Aplicaciones

La clase abre con un catálogo que vale la pena retener porque explica de dónde vienen los datasets y por qué el campo se financia: análisis deportivo (mapas de calor de un jugador, estadísticas automáticas), conducción autónoma (predicción de trayectorias para planificación), retail sin cajas (Amazon Go), robótica de manipulación, vigilancia, y VFX/postproducción (*motion tracking* para insertar objetos sintéticos). En la clase se agregan además tres dominios donde el tracking se especializa: UAV aéreos, percepción 3D en vehículos, y biología —seguimiento de células en microscopía de lapso temporal.

---

## Ver también

- [Detección de Objetos](/fundamentos/deteccion-de-objetos) — la etapa que alimenta todo el pipeline.
- [Filtro de Kalman](/fundamentos/filtro-de-kalman) — el modelo de movimiento estándar.
- [Asignación Húngara](/fundamentos/asignacion-hungara) — cómo se resuelve la matriz de costo.
- [Re-identificación](/fundamentos/re-identificacion) — la apariencia como puente sobre las oclusiones.
- [Métricas de Tracking](/fundamentos/metricas-de-tracking) — MOTA, IDF1, HOTA y por qué discrepan.
- [Análisis de Video](/fundamentos/analisis-de-video) — el marco general del que esta tarea es una mitad.
- [Metric Learning](/fundamentos/metric-learning) y [Triplet Loss](/fundamentos/triplet-loss) — cómo se aprende la distancia que asocia.
- [Clase 42](/clases/clase-42) — la clase que desarrolla todo esto.
