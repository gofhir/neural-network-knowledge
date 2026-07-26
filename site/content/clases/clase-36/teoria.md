---
title: "Teoría - Introducción al Análisis de Video"
weight: 10
math: true
---

> **Recorrido de la Clase 36** del Diplomado IA UC (Vladimir Araujo, Senior AI Researcher). Segunda clase del módulo de Audio y Video, ahora sobre **video**. La visión por computador maduró entendiendo **imágenes**; el video —una secuencia de imágenes— recibió menos atención pese a ser el formato dominante del mundo real. La clase introduce el campo: qué es un video y por qué el **movimiento** lo cambia todo, las dos grandes áreas (**seguimiento de objetos** y **reconocimiento de acciones**), los **datasets** que definieron el progreso, y la evolución de los **enfoques de deep learning** —del 2D CNN por frame (que ignora el tiempo) a las arquitecturas que sí lo modelan.

---

## 1. De la imagen al video

La visión por computador busca entender imágenes, y creció enormemente con el deep learning (detectar obstáculos, segmentar, extraer contexto de una escena). El **video** empezó a recibir más atención recién después: la mayoría de las aplicaciones se centran en imágenes, con menos foco en secuencias.

El análisis de video **no es nuevo**: ya en **1878** un experimento fotográfico usó múltiples cámaras para capturar 24 imágenes del galope de un caballo —¿levanta las cuatro patas del suelo?—, obteniendo datos de posición y tiempo a partir de los frames.

### 1.1 ¿Qué es un video?

Un **conjunto ordenado de frames** (imágenes), caracterizado por su duración, resolución (constante en la secuencia), color (grises o RGB) y cantidad de frames por segundo. Según el acceso:

- **Video stream** (feed en vivo): solo el frame actual y los anteriores.
- **Video sequence** (longitud fija): acceso completo, del primer al último frame.

### 1.2 ¿Por qué importa?

Aplicaciones útiles (vigilancia, análisis deportivo), múltiples **modalidades** (audio, texto, imágenes), **automatización** (la mayoría de las cámaras solo graban), exploración del mundo (robots, autos autónomos) y **muchos problemas sin resolver**. Se desarrolla en el fundamento [Análisis de Video](/fundamentos/analisis-de-video).

---

## 2. Imagen vs. video: el movimiento

Un video es, por definición, una secuencia de imágenes cuyos frames están relacionados **espacial y temporalmente**. Las diferencias clave:

{{< concept-alert type="clave" >}}
**El movimiento** es la característica que define al video. Es una feature poderosa para entenderlo —*correr* y *trotar* tienen píxeles promedio parecidos pero dinámicas distintas—, y es justamente lo que se pierde al analizar un video frame a frame.
{{< /concept-alert >}}

- **Multimodalidad:** al menos imagen y audio, útiles conjuntamente.
- **Tamaño:** los videos requieren mucho más almacenamiento y procesamiento que las imágenes.

---

## 3. Área 1: seguimiento visual de objetos (VOT)

El **Visual Object Tracking** consiste en **localizar un objeto en todos los frames** de un video, dada solo su ubicación en el **primer frame**. No hace falta saber *qué* es el objeto; se usan solo los frames anteriores (enfoque stream), típicamente para tracking de corto plazo, y puede haber múltiples objetos (MOT).

Sus **desafíos**: carga de cómputo (tiempo real), cambios de **apariencia** (dinámica, iluminación, punto de vista), **interacción** entre objetos (oclusión, similitud) y **movimiento** (estimación de flujo).

### 3.1 Flujo óptico

La **estimación de flujo óptico** es un problema clave: **computar el desplazamiento de píxeles entre dos frames**, tratado como un problema de **correspondencia**. Ayuda a entender el movimiento de los píxeles de un frame a otro; su salida es un **vector de movimiento** entre el frame 1 y el 2. Se desarrolla en el fundamento [Flujo óptico](/fundamentos/flujo-optico); su versión con deep learning es [FlowNet](/papers/flownet-dosovitskiy-2015).

Los datasets de VOT son difíciles de etiquetar (el ground-truth es costoso); el **VOT Challenge** (Kristan, 2020) aporta implementaciones open source y un framework de evaluación, aunque con videos cortos y en poca cantidad.

---

## 4. Área 2: reconocimiento de acciones

Las **acciones humanas** son el contenido principal de los videos. El objetivo: **detectar una acción o evento** en un video —útil para vigilancia inteligente (situaciones anómalas), recuperación de video (buscar por consulta) y navegación de contenido (encontrar un paso de una receta).

### 4.1 Tareas

- **Clasificación de acciones** — predecir la etiqueta de acción del video.
- **Localización/detección** — encontrar la región espacial y/o el intervalo temporal de la acción.

Y variantes complejas: **actividades complejas** (secuencias de acciones hacia una meta: cocinar, lavar), **múltiples acciones** (simultáneas), y **reconocimiento egocéntrico** (first-person, con cámara wearable). Además, la distinción **trimmed** (un solo acto recortado) vs **untrimmed** (con información irrelevante, más realista).

Es una tarea desafiante por *background clutter*, oclusión parcial, cambios de escala, punto de vista, iluminación y apariencia. Se desarrolla en [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).

### 4.2 Los datasets

| Dataset | Aporte |
|---|---|
| **KTH** (Schuldt, 2004) | 6 acciones, 2.391 videos; entorno controlado |
| **[HMDB51](/papers/hmdb-kuehne-2011)** (Kuehne, 2011) | 51 acciones, ~6.849 videos; estabilización |
| **[UCF101](/papers/ucf101-soomro-2012)** (Soomro, 2012) | 101 clases, 13.320 videos de YouTube; 5 grupos |
| **[Kinetics](/papers/kinetics-kay-2017)** (Kay, 2017) | 400 clases, cientos de miles de videos; el "ImageNet del video" |
| **[Something-Something](/papers/something-something-goyal-2017)** (Goyal, 2017) | 174 labels de interacciones ("Poner [algo] sobre [algo]") |
| **[EPIC-KITCHENS](/papers/epic-kitchens-damen-2018)** (Damen, 2018) | Egocéntrico, untrimmed, 55 h, 11,5M frames |

---

## 5. Cómo se usa el deep learning

### 5.1 El punto de partida: 2D CNN por frame

El enfoque simple: **pasar cada frame por una CNN 2D** y predecir en cada frame (opcionalmente fusionando información con una capa intermedia, y usando redes pre-entrenadas en imágenes). Se preprocesa el video como imágenes: una instancia es un conjunto de frames con una etiqueta.

Funciona, pero tiene **limitaciones**:

{{< concept-alert type="advertencia" >}}
El 2D CNN por frame **descarta el sentido temporal**, **descarta el movimiento** de objetos/personas, **no usa información multimodal** y **no es una red especializada** para video. El orden de los frames —la esencia del video— se pierde.
{{< /concept-alert >}}

### 5.2 Introducir la información temporal

El sentido temporal es esencial para detectar bien una acción: un video debe tratarse como una **secuencia**, con **dependencias de largo alcance** deseables. Las familias de arquitecturas que lo logran:

- **2D CNN + RNN** — la solución que propone la clase. Una CNN extrae features por frame, una RNN (apta para secuencias) las procesa. Funciona **mejor** que el 2D CNN solo, pero la RNN **no se puede paralelizar** (es secuencial). Es el enfoque [LRCN](/papers/lrcn-donahue-2015).
- **Two-stream** — apariencia (RGB) + movimiento ([flujo óptico](/fundamentos/flujo-optico)), en [Two-Stream](/papers/two-stream-simonyan-2014).
- **Convoluciones 3D** — features espacio-temporales end-to-end, en [C3D](/papers/c3d-tran-2015).
- **La síntesis** — [I3D](/papers/i3d-carreira-2017) infla una CNN 2D pre-entrenada a 3D y establece "pre-entrenar en Kinetics, transferir".
- **Muestreo esparcido** — [TSN](/papers/tsn-wang-2016) muestrea segmentos distribuidos por el video (la idea del [laboratorio](/laboratorios/lab-36)).

---

## 6. Cierre

La clase trazó el mapa del análisis de video: qué es un video, sus dos áreas (seguimiento y reconocimiento de acciones), sus datasets y la evolución de enfoques de deep learning. El hilo conductor es el **movimiento**: el 2D CNN por frame lo ignora, y cada arquitectura posterior —RNN, flujo óptico, convoluciones 3D— es una forma distinta de recuperarlo. El [laboratorio](/laboratorios/lab-36) aterriza estos conceptos entrenando un clasificador de acciones sobre UCF11 (un backbone ResNet con muestreo de frames), la versión práctica del "2D CNN + agregación temporal" que cierra la clase.

---

**Ver también:** [Clase 36 - Profundización](/clases/clase-36/profundizacion) · [Clase 36 - Práctica](/clases/clase-36/practica) · Fundamentos: [Análisis de Video](/fundamentos/analisis-de-video) · [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) · [Flujo óptico](/fundamentos/flujo-optico) · [Dominio: Video](/dominios/video).
