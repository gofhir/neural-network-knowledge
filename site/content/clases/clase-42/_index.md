---
title: "Clase 42 - Tracking de Objetos en Video"
weight: 420
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga (DCC, Pontificia Universidad Católica de Chile)
**Módulo:** Video — seguimiento multi-objeto

La clase 36 dividió el análisis de video en dos grandes áreas: reconocimiento de acciones y **seguimiento de objetos**. Las clases 38 y 40 desarrollaron la primera. Esta desarrolla la segunda, y su punto de partida es una distinción que conviene fijar antes que nada: detectar es **razonamiento espacial**; seguir es **razonamiento espacio-temporal**. Un detector aplicado frame a frame ya entrega cajas correctas en cada instante; lo que no entrega es el hilo que une la caja del frame $t$ con la del frame $t+1$.

Ese hilo —la identidad— no se observa en ningún frame. Hay que inferirlo. Y por eso el seguimiento no es un problema de percepción sino de **asociación**: aun con un detector perfecto, queda por decidir cuál de las $M$ cajas nuevas corresponde a cuál de las $N$ trayectorias activas.

{{< concept-alert type="clave" >}}
**El hilo de la clase.** La tarea exige tres cosas —preservar identidad, entender dinámicas de movimiento, y recuperar la asociación tras una oclusión— y cada componente de los algoritmos que se presentan ataca exactamente una: el **algoritmo húngaro** para la asignación, el **filtro de Kalman** para el movimiento, los **descriptores de apariencia** para la recuperación.

**Lo que la clase muestra sin decirlo.** [SORT](/papers/sort-bewley-2016) (2016) es cien líneas de código sin un solo parámetro aprendido en su componente de seguimiento, y fue el mejor tracker online de su momento. [DeepSORT](/papers/deepsort-wojke-2017) (2017) le agrega una red de apariencia y reduce los cambios de identidad un 45 %. La clase concluye que es "más robusto", y lo es — pero la métrica con que se rankeaban los benchmarks de la época **casi no lo nota**: esa mejora vale 0,35 puntos de MOTA, mientras el efecto secundario de la misma configuración cuesta 2,28. Medirlo es el eje de la [profundización](profundizacion).
{{< /concept-alert >}}

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Las 95 diapositivas: espacial contra espacio-temporal, los desafíos y la oclusión, offline como problema de grafo, el paréntesis de redes siamesas y triplet, la anatomía de cuatro casillas del tracking online, SORT y DeepSORT paso a paso, y los modelos integrados de 2024-2025" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="La incertidumbre que SORT ya tenía y no usaba, por qué Mahalanobis premia a la trayectoria más incierta y se invierte a los 10 frames, la aritmética de MOTA reconstruida y sus contrafactuales, cuándo el húngaro le gana al codicioso, y los siete años que la clase salta" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="SORT completo desde cero con cuatro ablaciones medidas, y MOTA/IDF1/HOTA implementadas desde su definición — en triple framework, con los cuatro backends coincidiendo hasta cero exacto" icon="code" >}}
  {{< card link="/clases/clase-36" title="Clase relacionada: Introducción al Análisis de Video" subtitle="Donde se define el seguimiento como una de las dos grandes áreas del video, junto al reconocimiento de acciones" icon="eye" >}}
  {{< card link="/clases/clase-40" title="Clase anterior: Reconocimiento de acciones" subtitle="La otra mitad del análisis de video: TSN, TSM y la ruta de la eficiencia" icon="eye" >}}
  {{< card link="/clases/clase-26" title="Relacionada: Meta-aprendizaje" subtitle="Prototypical Networks y metric learning: la misma maquinaria de aprender distancias que aquí resuelve la asociación" icon="sparkles" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/seguimiento-de-objetos" title="Seguimiento de Objetos" subtitle="El vocabulario del área: SOT contra MOT, tracking-by-detection, online contra offline, y la anatomía común a todos los algoritmos" icon="book-open" >}}
  {{< card link="/fundamentos/filtro-de-kalman" title="Filtro de Kalman" subtitle="Predicción y corrección, la ganancia que pondera modelo contra sensor, y las dos patologías que aparecen en tracking" icon="book-open" >}}
  {{< card link="/fundamentos/asignacion-hungara" title="Asignación Húngara" subtitle="Por qué el codicioso no basta, la invariancia por filas y columnas, y los cuatro ajustes que exige el caso real" icon="book-open" >}}
  {{< card link="/fundamentos/metricas-de-tracking" title="Métricas de Tracking" subtitle="MOTA, IDF1 y HOTA: qué mide cada una, dónde falla, y por qué a veces ordenan los mismos sistemas al revés" icon="book-open" >}}
  {{< card link="/fundamentos/re-identificacion" title="Re-identificación" subtitle="El descriptor de conjunto abierto que atraviesa las oclusiones, y por qué la red siamesa colapsa sin tripletas" icon="book-open" >}}
  {{< card link="/fundamentos/deteccion-de-objetos" title="Detección de Objetos" subtitle="IoU, NMS, anchors y mAP: la etapa que alimenta todo el pipeline y que acota su rendimiento" icon="book-open" >}}
  {{< card link="/fundamentos/triplet-loss" title="Triplet Loss" subtitle="La pérdida del paréntesis de la clase, y por qué el colapso deja de ser solución al agregar el ancla" icon="book-open" >}}
  {{< card link="/fundamentos/analisis-de-video" title="Análisis de Video" subtitle="El marco general del que el seguimiento es una de las dos mitades" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

### Los dos algoritmos centrales

{{< cards >}}
  {{< card link="/papers/sort-bewley-2016" title="SORT (2016)" subtitle="Bewley et al. — Kalman más húngaro, sin apariencia, a 260 Hz en un núcleo de CPU. Y el hallazgo que la clase no menciona: cambiar solo el detector mueve MOTA de 15,1 a 34,0" icon="document-text" >}}
  {{< card link="/papers/deepsort-wojke-2017" title="DeepSORT (2017)" subtitle="Wojke et al. — el descriptor de 128-D, la compuerta χ², y la cascada de matching. En los experimentos, λ=0: Mahalanobis nunca entra al costo" icon="document-text" >}}
  {{< card link="/papers/kalman-1960" title="Filtro de Kalman (1960)" subtitle="Kálmán — el estimador recursivo que hizo computable el filtrado óptimo, y que sesenta años después sigue siendo el modelo de movimiento estándar" icon="document-text" >}}
{{< /cards >}}

### Cómo se mide

{{< cards >}}
  {{< card link="/papers/mot16-milan-2016" title="MOT16 (2016)" subtitle="Milan et al. — el benchmark con detecciones públicas y servidor de evaluación, y la decisión de rankear por MOTA que orientó una década" icon="document-text" >}}
  {{< card link="/papers/idf1-ristani-2016" title="IDF1 / DukeMTMC (2016)" subtitle="Ristani et al. — la métrica de identidad nacida en multi-cámara; el dataset fue retirado en 2019 por privacidad" icon="document-text" >}}
  {{< card link="/papers/hota-luiten-2020" title="HOTA (2020)" subtitle="Luiten et al. — MOTA e IDF1 son dos proyecciones sesgadas del mismo espacio; la media geométrica de DetA y AssA los separa" icon="document-text" >}}
{{< /cards >}}

### Los siete años que la clase salta

{{< cards >}}
  {{< card link="/papers/tracktor-bergmann-2019" title="Tracktor (2019)" subtitle="Bergmann et al. — el regresor del detector como modelo de movimiento, sin entrenar nada sobre datos de seguimiento. Es de aquí el diagrama que la clase muestra en DeepSORT" icon="document-text" >}}
  {{< card link="/papers/fairmot-zhang-2020" title="FairMOT (2020)" subtitle="Zhang et al. — por qué fusionar detección y re-ID en una red hace perder a la segunda: anchors, resolución y dimensión del embedding" icon="document-text" >}}
  {{< card link="/papers/bytetrack-zhang-2021" title="ByteTrack (2021)" subtitle="Zhang et al. — asociar también las detecciones de score bajo. El estado del arte de 2021 sin ningún modelo de apariencia" icon="document-text" >}}
  {{< card link="/papers/oc-sort-cao-2022" title="OC-SORT (2022)" subtitle="Cao et al. — el filtro que se realimenta su propio error durante la oclusión, y la trayectoria virtual que lo repara. En DanceTrack, DeepSORT queda por debajo de SORT" icon="document-text" >}}
{{< /cards >}}

### Los modelos integrados

{{< cards >}}
  {{< card link="/papers/sutrack-chen-2024" title="SUTrack (2024)" subtitle="Chen et al. — cinco tareas de seguimiento de un objeto en un Transformer, con soft token type embedding. Es SOT, no MOT" icon="document-text" >}}
  {{< card link="/papers/sam3-meta-2025" title="SAM 3 (2025)" subtitle="Meta — seguir todas las instancias de un concepto dado en lenguaje natural, con la cabeza de presencia que separa reconocer de localizar" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Clase 36 - Introducción al Análisis de Video](/clases/clase-36) · [Clase 38 - CNN para reconocimiento en video](/clases/clase-38) · [Clase 40 - Reconocimiento de acciones](/clases/clase-40) · [Clase 26 - Meta-aprendizaje](/clases/clase-26) (metric learning) · Dominio [Video](/dominios/video).
