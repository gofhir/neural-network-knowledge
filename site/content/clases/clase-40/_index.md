---
title: "Clase 40 - Analítica de Videos: Reconocimiento de acciones"
weight: 400
sidebar:
  open: true
---

**Profesora:** Bianca Del Solar Medrano
**Módulo:** Video — reconocimiento de acciones eficiente

La clase abre con una pregunta de hoja de ruta: *¿cuál es la ruta actual?* La respuesta que da tiene dos ramas —**aumentar el rendimiento** de los modelos y **reducir su tamaño**—, y el resto de la clase transita casi enteramente por la segunda. Ese es el hilo que la distingue de la [Clase 36](/clases/clase-36), que recorrió el panorama del análisis de video, y de la [Clase 38](/clases/clase-38), que se ocupó de cómo heredar pesos preentrenados. Acá la pregunta es otra: **¿se puede modelar el tiempo sin pagar por ello?**

La respuesta llega en dos pasos. **[TSN](/papers/tsn-wang-2016)** elimina el costo del muestreo denso: en vez de procesar frames consecutivos a tasa fija, divide el video en segmentos y toma uno de cada uno, cubriendo la línea de tiempo completa con un presupuesto constante. **[TSM](/papers/tsm-lin-2019)** elimina el costo del modelado temporal: en vez de agregar una convolución 3D, desplaza una fracción de los canales a lo largo del tiempo y deja que la convolución 2D que ya existía haga la mezcla. Cero parámetros, cero FLOPs.

El resultado que ordena la clase está en una sola tabla del paper de TSM: la misma modificación arquitectónica vale **+3,5 puntos en Kinetics y +28,0 en Something-Something**. Esa asimetría dice menos sobre el modelo que sobre los benchmarks, y es la razón por la que la clase cierra con una comparación explícita entre ambos datasets.

{{< concept-alert type="clave" >}}
**Dos precisiones sobre el material.** La slide afirma que "TSM reemplaza 1/8 del mapa de características": eso vale para el modo **unidireccional**; en el bidireccional —el del checkpoint del laboratorio— se desplaza **1/4**, un octavo por cada dirección. Y la lámina titulada "Modelos offline con desplazamiento unidireccional" corresponde a la sección *"Online Models with Uni-directional TSM"* del paper: unidireccional significa **online**, y su sentido es poder procesar un stream en vivo. Ambos puntos se desarrollan en la [teoría](teoria).
{{< /concept-alert >}}

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Las 29 diapositivas: la ruta de la eficiencia, los cuatro enfoques anteriores y sus costos, TSN y el muestreo por segmentos, TSM y el desplazamiento, Kinetics contra Something-Something" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: la descomposición shift + MAC, la aritmética del fold, campo receptivo temporal acumulado, el costo del movimiento de datos y por qué el promedio del consenso es invariante al orden" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Implementar el desplazamiento temporal y verificar su equivalencia con una convolución temporal; el muestreo por segmentos contra el denso — en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-40" title="Laboratorio 40" subtitle="Reconocimiento de acciones con TSM sobre UCF-101, y cuatro experimentos que miden el aporte real del desplazamiento" icon="variable" >}}
  {{< card link="/clases/clase-38" title="Relacionada: CNN para reconocimiento en video" subtitle="La otra respuesta al mismo problema: inflar en vez de desplazar" icon="academic-cap" >}}
  {{< card link="/clases/clase-36" title="Relacionada: Introducción al Análisis de Video" subtitle="El panorama del campo: VOT, action recognition, datasets y enfoques" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/desplazamiento-temporal" title="Desplazamiento Temporal" subtitle="El mecanismo completo: la descomposición de la convolución, partial shift, residual shift y cuándo aporta" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-acciones" title="Reconocimiento de Acciones" subtitle="Tareas, datasets y la evolución de los enfoques de deep learning" icon="book-open" >}}
  {{< card link="/fundamentos/analisis-de-video" title="Análisis de Video" subtitle="Video, movimiento, stream contra sequence, VOT y action recognition" icon="book-open" >}}
  {{< card link="/fundamentos/inflado-de-convoluciones" title="Inflado de Convoluciones" subtitle="La estrategia opuesta: aceptar la convolución 3D y heredar los pesos 2D" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Redes Convolucionales" subtitle="Kernels, bloques residuales y el lugar exacto donde se inserta el módulo" icon="book-open" >}}
  {{< card link="/fundamentos/flujo-optico" title="Flujo óptico" subtitle="Uno de los 'enfoques anteriores' que la clase descarta por su costo de precómputo" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

### Las dos referencias del material

{{< cards >}}
  {{< card link="/papers/tsm-lin-2019" title="TSM (2019)" subtitle="Lin, Gan y Han — el desplazamiento temporal, sus dos correcciones y el modelo del laboratorio" icon="document-text" >}}
  {{< card link="/papers/tsn-wang-2016" title="TSN (2016)" subtitle="Wang et al. — el muestreo por segmentos y el consenso; la baseline sobre la que TSM se construye" icon="document-text" >}}
{{< /cards >}}

### Los enfoques anteriores que la clase enumera

{{< cards >}}
  {{< card link="/papers/c3d-tran-2015" title="C3D (2015)" subtitle="Tran et al. — 'algunos utilizan redes 3D': el costo en memoria y parámetros" icon="document-text" >}}
  {{< card link="/papers/i3d-carreira-2017" title="I3D (2017)" subtitle="Carreira y Zisserman — 'algunos necesitan más fotogramas': los 64 frames consecutivos" icon="document-text" >}}
  {{< card link="/papers/two-stream-simonyan-2014" title="Two-Stream (2014)" subtitle="Simonyan y Zisserman — 'algunos utilizan flujo óptico': los cálculos adicionales" icon="document-text" >}}
  {{< card link="/papers/r2plus1d-tran-2018" title="R(2+1)D (2018)" subtitle="Tran et al. — la vía intermedia: factorizar la convolución 3D en lugar de eliminarla" icon="document-text" >}}
{{< /cards >}}

### Los datasets de la comparación final

{{< cards >}}
  {{< card link="/papers/kinetics-kay-2017" title="Kinetics (2017)" subtitle="Kay et al. — 'no requiere un análisis temporal laborioso': el dataset del checkpoint" icon="document-text" >}}
  {{< card link="/papers/something-something-goyal-2017" title="Something-Something (2017)" subtitle="Goyal et al. — 'requiere modelado temporal detallado': donde TSM gana 28 puntos" icon="document-text" >}}
  {{< card link="/papers/ucf101-soomro-2012" title="UCF-101 (2012)" subtitle="Soomro et al. — los videos sobre los que corre el laboratorio" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Laboratorio 40](/laboratorios/lab-40) · [Clase 38 - CNN para reconocimiento en video](/clases/clase-38) (el inflado, la estrategia opuesta) · [Clase 36 - Introducción al Análisis de Video](/clases/clase-36) (el panorama) · [Clase 28 - Aprendizaje Autosupervisado](/clases/clase-28) (donde se desarrolla el MAE que la clase deja como material extra) · Dominio [Video](/dominios/video).
