---
title: "Reconocimiento de Acciones"
weight: 121
math: true
---

El **reconocimiento de acciones** (action recognition) es la tarea central del [análisis de video](/fundamentos/analisis-de-video): dado un video, **identificar qué acción o evento ocurre en él**. Las acciones humanas son el contenido principal de la mayoría de los videos —películas, deportes, vigilancia, tutoriales—, y reconocerlas automáticamente habilita aplicaciones desde la búsqueda de video hasta el monitoreo clínico. Este fundamento acompaña a la [Clase 36](/clases/clase-36): recorre las tareas, los datasets que definieron el campo y la evolución de los enfoques de deep learning, del 2D CNN por frame a las arquitecturas que sí modelan el tiempo.

---

## 1. Acciones, eventos y tareas

Una **acción** es un conjunto de pequeños movimientos hacia un objetivo común; un **evento** puede incluir múltiples acciones (de distintas personas u objetos). Sobre esta base, el campo define varias **tareas** de dificultad creciente:

- **Clasificación de acciones** — predecir la etiqueta de acción de un video (el video ya está recortado a una sola acción).
- **Localización / detección de acciones** — encontrar la **región espacial** y/o el **intervalo temporal** donde ocurre la acción.

Y variantes más complejas y realistas:

- **Actividades complejas** — la vida diaria implica secuencias de acciones hacia una meta (cocinar, lavar).
- **Múltiples acciones** — escenarios reales con acciones simultáneas, de personas u objetos.
- **Reconocimiento egocéntrico** (*first-person*) — analizar video capturado por una cámara *wearable*, útil en robótica y asistencia.

Una distinción práctica clave:

{{< concept-alert type="clave" >}}
**Trimmed vs. untrimmed.** Un video *trimmed* muestra **una sola acción** recortada (el caso fácil, el de UCF101/HMDB). Un video *untrimmed* contiene información adicional posiblemente irrelevante (el caso realista: videos web, cámaras de seguridad, grabación continua). Los datasets modernos ([EPIC-KITCHENS](/papers/epic-kitchens-damen-2018)) apuntan a lo *untrimmed* precisamente porque es más difícil y más parecido al mundo real.
{{< /concept-alert >}}

El reconocimiento es difícil por los mismos problemas que aquejan a toda la visión —*background clutter*, oclusión parcial, cambios de escala, punto de vista, iluminación y apariencia— agravados por la dimensión temporal.

---

## 2. Los datasets que definieron el campo

El progreso del reconocimiento de acciones estuvo marcado por sus benchmarks, cada uno más grande y realista que el anterior:

| Dataset | Año | Escala | Aporte |
|---|---|---|---|
| **KTH** (Schuldt) | 2004 | 6 acciones, 2.391 videos | Primer benchmark, entorno controlado |
| **[HMDB51](/papers/hmdb-kuehne-2011)** (Kuehne) | 2011 | 51 acciones, ~6.849 clips | De películas/web, con estabilización |
| **[UCF101](/papers/ucf101-soomro-2012)** (Soomro) | 2012 | 101 clases, 13.320 videos | YouTube "in the wild", 5 grupos |
| **[Kinetics](/papers/kinetics-kay-2017)** (Kay) | 2017 | 400 clases, ~650.000 videos | El "ImageNet del video" |
| **[Something-Something](/papers/something-something-goyal-2017)** (Goyal) | 2017 | 174 labels, ~220.847 videos | Interacciones temporales, agnóstico al objeto |
| **[EPIC-KITCHENS](/papers/epic-kitchens-damen-2018)** (Damen) | 2018 | 55 h, 11,5M frames | Egocéntrico, untrimmed |

Estos datasets no son intercambiables: **Kinetics** es grande y realista pero muchas de sus clases se reconocen por el **fondo o el objeto** (una piscina implica "nadar") sin entender el movimiento; **Something-Something**, en cambio, usa plantillas agnósticas al objeto ("Poner [algo] sobre [algo]") que **obligan** al modelo a razonar sobre la dinámica temporal —expone la debilidad de los modelos que ignoran el tiempo.

{{< concept-alert type="dato" >}}
La aparición de **Kinetics** fue para el video lo que **ImageNet** fue para las imágenes: un dataset lo suficientemente grande como para **pre-entrenar** modelos que luego se transfieren a datasets pequeños (UCF101, HMDB). Antes de Kinetics, los datasets eran demasiado pequeños para entrenar redes profundas de video sin sobreajustar.
{{< /concept-alert >}}

---

## 3. Cómo se usa el deep learning

### 3.1 El punto de partida: 2D CNN por frame

El enfoque más simple: **pasar cada frame por una CNN 2D** y predecir en cada uno (opcionalmente fusionando información con una capa intermedia, y usando redes pre-entrenadas en imágenes). Para usarlo se preprocesa el video tratándolo como imágenes: una instancia es un conjunto de frames con una etiqueta.

Funciona, pero tiene **limitaciones** graves que la clase enumera:

- **Descarta el sentido temporal** (el orden de los frames).
- **Descarta el movimiento** de objetos o personas.
- **No usa información multimodal**.
- **No es una red especializada** para video.

### 3.2 Añadir la noción temporal

El sentido temporal es esencial para detectar correctamente una acción: un video debe tratarse como una **secuencia**, y es deseable capturar **dependencias de largo alcance**. Varias familias de arquitecturas surgieron para lograrlo:

- **2D CNN + RNN.** La solución que propone la clase: una CNN extrae *features* por frame y una **RNN/LSTM** las procesa como secuencia temporal. Funciona mejor que el 2D CNN solo y captura dependencias largas —pero la RNN es **secuencial y no se puede paralelizar**. Es exactamente el enfoque **[LRCN](/papers/lrcn-donahue-2015)** (Donahue, 2015).
- **Two-stream.** Un stream espacial (apariencia, sobre frames RGB) y un stream temporal (movimiento, sobre **[flujo óptico](/fundamentos/flujo-optico)** precomputado), fusionados al final. Es **[Two-Stream](/papers/two-stream-simonyan-2014)** (Simonyan & Zisserman, 2014).
- **Convoluciones 3D.** Kernels que se extienden también en el tiempo, aprendiendo *features* espacio-temporales end-to-end. Es **[C3D](/papers/c3d-tran-2015)** (Tran, 2015).
- **La síntesis.** **[I3D](/papers/i3d-carreira-2017)** (Carreira & Zisserman, 2017) "infla" una CNN 2D pre-entrenada en ImageNet a 3D, combina two-stream y aprovecha el pre-entrenamiento en Kinetics —estableciendo el paradigma "pre-entrenar en Kinetics, transferir".
- **Muestreo esparcido.** **[TSN](/papers/tsn-wang-2016)** (Wang, 2016) divide el video en segmentos y muestrea un frame por segmento, cubriendo todo el video con poco cómputo —la idea que usa el [laboratorio de la clase](/laboratorios/lab-36). Su defecto es que el consenso por promedio es **invariante al orden**: modela qué ocurre, no en qué secuencia.
- **El tiempo sin pagarlo.** **[TSM](/papers/tsm-lin-2019)** (Lin, Gan y Han, 2019) toma la baseline de TSN y le inserta un [desplazamiento temporal](/fundamentos/desplazamiento-temporal) de 1/4 de los canales dentro de cada bloque residual: obtiene modelado espacio-temporal con **cero parámetros y cero FLOPs adicionales**, y su variante causal habilita reconocimiento en vivo. Cubierto en la [Clase 40](/clases/clase-40).

---

## 4. Relevancia para salud y video clínico

El reconocimiento de acciones tiene aplicaciones clínicas directas. En **cirugía asistida por video**, reconocer las fases de un procedimiento (la secuencia de acciones hacia una meta, como en Something-Something) permite documentación automática, alertas y análisis de habilidad quirúrgica. En **rehabilitación**, clasificar y localizar ejercicios en video da retroalimentación objetiva. En **asistencia a personas mayores**, el reconocimiento egocéntrico (estilo EPIC-KITCHENS) de actividades de la vida diaria y la detección de eventos anómalos (caídas) habilitan el monitoreo en el hogar. En todos, la receta estándar es la misma que en el resto del campo: **pre-entrenar** en un dataset grande (Kinetics) y **transferir** al dominio médico, donde los datos etiquetados son escasos —y donde, a menudo, el **orden temporal** de las acciones (no solo la apariencia de un frame) es lo que importa clínicamente.

---

## Referencias

- Fundamentos relacionados: [Análisis de Video](/fundamentos/analisis-de-video) · [Flujo óptico](/fundamentos/flujo-optico) · [Redes recurrentes](/fundamentos/redes-recurrentes) · [Redes convolucionales](/fundamentos/redes-convolucionales) · [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones) · [Desplazamiento Temporal](/fundamentos/desplazamiento-temporal).
- Clases: [Clase 36 - Introducción al Análisis de Video](/clases/clase-36) · [Clase 38 - CNN para reconocimiento en video](/clases/clase-38) (modelos pre-entrenados y el linaje posterior a I3D: [S3D](/papers/s3d-xie-2018), [R(2+1)D](/papers/r2plus1d-tran-2018), [SlowFast](/papers/slowfast-feichtenhofer-2019)) · [Clase 40 - Reconocimiento de acciones eficiente](/clases/clase-40) (TSN → TSM y la ruta de la eficiencia).
- Laboratorios: [Lab 36](/laboratorios/lab-36) (bag of frames: 4 frames ≥ 8 frames, el pooling ignora el orden) · [Lab 38](/laboratorios/lab-38) (I3D pre-entrenado: la limitación *trimmed* medida —los primeros 2,6 s dan 92,9 % y los últimos 4 s fallan— y el video invertido que no cambia la predicción) · [Lab 40](/laboratorios/lab-40) (TSM: la ablación del desplazamiento mide cuánta temporalidad contiene cada acción — 82,76 puntos en un salto alto contra 0,42 en una guitarra).
- Dominio: [Video](/dominios/video).
