---
title: "Clase 38 - CNN para reconocimiento en video"
weight: 380
sidebar:
  open: true
---

**Profesora:** Bianca Del Solar Medrano
**Módulo:** Video — modelos pre-entrenados

La clase recorre la escalera de arquitecturas para comprensión de video —**CNN2D + agrupación temporal → CNN2D + RNN → Two-Stream → C3D → I3D**— pero el subtítulo del material revela el hilo real: **modelos pre-entrenados**. Cada eslabón se lee mejor como respuesta a una pregunta incómoda: *¿de dónde saco los pesos iniciales de un modelo de video, si el único gran corpus etiquetado disponible es de imágenes?* Las arquitecturas 2D heredan ImageNet pero modelan mal el tiempo; las 3D modelan bien el tiempo pero —dice el slide de C3D— "no pueden aprovechar el pre-entrenamiento de ImageNet". **I3D** rompe el dilema con el *punto fijo del video aburrido*: si una imagen repetida $N$ veces es un video válido, entonces repartir los pesos 2D a lo largo del eje temporal y dividirlos por $N$ deja a la red 3D arrancando exactamente donde terminó la 2D.

Complementa la [Clase 36](/clases/clase-36), que cubrió el panorama del análisis de video. Acá el foco es la **mecánica de la transferencia de pesos** y su contabilidad: de dónde salen realmente los 79M de parámetros de C3D contra los 25M de I3D, cómo se lee la tabla de resultados sin confundirla con la famosa, y qué pasó después con las tres desventajas que la clase le deja a I3D.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 29 diapositivas: las cinco arquitecturas, sus ventajas y desventajas, el inflado y las tablas del paper I3D" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: la condición de punto fijo, contabilidad de parámetros y FLOPs, factorización (2+1)D, campo receptivo temporal, asimetría espacio-temporal" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Inflar una CNN 2D a 3D y verificar el punto fijo; bloques (2+1)D — en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-38" title="Laboratorio 38" subtitle="Reconocimiento de acciones con I3D" icon="variable" >}}
  {{< card link="/clases/clase-36" title="Relacionada: Introducción al Análisis de Video" subtitle="El panorama del campo: VOT, action recognition, datasets y enfoques" icon="academic-cap" >}}
  {{< card link="/clases/clase-37" title="Clase anterior: Datasets y Herramientas para Audio" subtitle="El ciclo de vida del dato de audio" icon="arrow-left" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/inflado-de-convoluciones" title="Inflado de Convoluciones" subtitle="El punto fijo del video aburrido, qué se infla y qué no, y el inflado de kernels separables" icon="book-open" >}}
  {{< card link="/fundamentos/analisis-de-video" title="Análisis de Video" subtitle="Video, movimiento, stream vs sequence, VOT y action recognition" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-acciones" title="Reconocimiento de Acciones" subtitle="Tareas, datasets y la evolución de los enfoques de deep learning" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer Learning" subtitle="Feature extraction, fine-tuning y por qué heredar pesos funciona" icon="book-open" >}}
  {{< card link="/fundamentos/flujo-optico" title="Flujo óptico" subtitle="Desplazamiento de píxeles: la entrada de la corriente temporal" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Redes Convolucionales" subtitle="Kernels, filtros y la jerarquía de capas que hace posible el pre-entrenamiento" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

### La escalera de arquitecturas

{{< cards >}}
  {{< card link="/papers/large-scale-video-karpathy-2014" title="Sports-1M / Fusión temporal (2014)" subtitle="Karpathy et al. — el origen de CNN2D + agrupación temporal, y el hallazgo incómodo de que un solo frame casi alcanzaba" icon="document-text" >}}
  {{< card link="/papers/lrcn-donahue-2015" title="LRCN (2015)" subtitle="Donahue et al. — CNN2D + LSTM, la familia (a) de la comparativa" icon="document-text" >}}
  {{< card link="/papers/two-stream-simonyan-2014" title="Two-Stream (2014)" subtitle="Simonyan y Zisserman — apariencia y movimiento en dos corrientes" icon="document-text" >}}
  {{< card link="/papers/two-stream-fusion-feichtenhofer-2016" title="Two-Stream Fusion (2016)" subtitle="Feichtenhofer et al. — dónde y cómo fusionar: la familia (d) 3D-Fused de la tabla" icon="document-text" >}}
  {{< card link="/papers/3d-cnn-ji-2013" title="3D CNN (2010/2013)" subtitle="Ji et al. — el primer 3D CNN para acciones, anterior a AlexNet y con capa hardwired" icon="document-text" >}}
  {{< card link="/papers/c3d-tran-2015" title="C3D (2015)" subtitle="Tran et al. — kernels 3×3×3 homogéneos, y la desventaja de no poder heredar ImageNet" icon="document-text" >}}
  {{< card link="/papers/i3d-carreira-2017" title="I3D (2017)" subtitle="Carreira y Zisserman — el inflado, Kinetics y el punto fijo del video aburrido" icon="document-text" >}}
{{< /cards >}}

### El linaje posterior: qué pasó con las desventajas de I3D

{{< cards >}}
  {{< card link="/papers/s3d-xie-2018" title="S3D (2018)" subtitle="Xie et al. — auditar qué capas necesitan ser 3D: el diseño top-heavy y la separabilidad" icon="document-text" >}}
  {{< card link="/papers/r2plus1d-tran-2018" title="R(2+1)D (2018)" subtitle="Tran et al. — el autor de C3D revisa su propio trabajo: factorizar mejora la optimización" icon="document-text" >}}
  {{< card link="/papers/slowfast-feichtenhofer-2019" title="SlowFast (2019)" subtitle="Feichtenhofer et al. — dos vías por framerate en lugar de por modalidad: adiós al flujo óptico" icon="document-text" >}}
{{< /cards >}}

### Contexto de arquitecturas 2D

{{< cards >}}
  {{< card link="/papers/googlenet-szegedy-2014" title="GoogLeNet / Inception-v1 (2014)" subtitle="Szegedy et al. — el backbone que I3D infla, y la razón real de su bajo conteo de parámetros" icon="document-text" >}}
  {{< card link="/papers/kinetics-kay-2017" title="Kinetics (2017)" subtitle="Kay et al. — el 'ImageNet del video' que hizo prescindible el préstamo de ImageNet" icon="document-text" >}}
{{< /cards >}}
