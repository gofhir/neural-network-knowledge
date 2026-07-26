---
title: "UCF101: 101 Human Actions in the Wild (2012)"
weight: 396
math: true
---

{{< paper-card
    title="UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild"
    authors="Khurram Soomro, Amir Roshan Zamir, Mubarak Shah (UCF)"
    year="2012"
    venue="CRCV-TR-12-01 / arXiv:1212.0402"
    pdf="/papers/ucf101-soomro-2012.pdf" >}}
UCF101 no propone un modelo: propone un **dataset** que redefinió el listón de dificultad del [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) durante la primera mitad de la década de 2010. Reúne **101 clases de acciones humanas**, **13 320 clips** y **27 horas** de video descargado de YouTube —videos subidos por usuarios reales ("in the wild"), con movimiento de cámara, fondos abarrotados, oclusión parcial e iluminación variable—. En su publicación era, según los autores, el dataset de acciones más grande y desafiante existente. Aporta además un **baseline** con *bag of words* espacio-temporal que alcanza **44,5 % de exactitud**, y un **protocolo** de validación cruzada de 25 folds *leave-one-group-out*. Es el cuarto y mayor eslabón del linaje UCF (UCF Sports → UCF11 → UCF50 → UCF101), y el benchmark canónico de la [Clase 36](/clases/clase-36).
{{< /paper-card >}}

---

## Contexto: por qué el campo necesitaba un dataset "in the wild"

Hacia 2012 los benchmarks de acción sufrían **dos deficiencias sistemáticas**. La primera era el techo bajo de clases: **KTH** tenía 6, **Weizmann** 9, **UCF Sports** 9, **IXMAS** 11, y ni siquiera el mayor disponible superaba ~50 (**HMDB51** con 51 clases y 6766 clips; **UCF50** con 50 y 6681). Como el número de clases juega un rol crucial al evaluar un método, esos conjuntos decían poco sobre cómo escalaría un clasificador.

La segunda deficiencia eran los **entornos irrealmente controlados**: KTH, Weizmann e IXMAS eran *actor staged* —un actor frente a cámara fija, fondo estático—; HOHA y UCF Sports venían de cine y televisión con encuadre profesional. Nada de eso representa el video que un sistema encuentra en producción, donde la cámara tiembla, el fondo está saturado y la iluminación es la que haya. En un video de YouTube conviven fuentes de variabilidad que **compiten con la señal de la acción**: el flujo óptico (*optical flow*) mezcla el movimiento del sujeto con el paneo y el zoom de la cámara; el fondo abarrotado genera textura distractora; la baja calidad degrada cualquier descriptor de apariencia.

## Composición del dataset

Las 101 clases se reparten en **cinco tipos**: *Human-Object Interaction*, *Body-Motion Only*, *Human-Human Interaction*, *Playing Musical Instruments* y *Sports* (el grupo más numeroso, 50 clases). UCF101 es literalmente una **extensión de UCF50**: hereda sus 50 clases y suma 51 nuevas, fijando 25 grupos por acción y hasta 7 clips por grupo.

La pieza estructural clave es la **organización en grupos**. Los clips de cada clase se dividen en **25 grupos** de 4 a 7 clips, y los del mismo grupo **comparten fondo y actores** (provienen del mismo video fuente o sesión). Si entrenamiento y prueba compartieran clips de un grupo, un clasificador podría reconocer la acción **memorizando el fondo** en vez de aprenderla —una fuga de información que infla la exactitud—. La convención de nombres `v_X_gY_cZ.avi` codifica clase, grupo y clip en el propio archivo, permitiendo implementar el *leave-one-group-out* parseando nombres. Todos los clips son de 25 fps y 320 × 240, con duración media de 7,21 s, guardados como `.avi` con códec DivX.

## Baseline y protocolo

El baseline usa el pipeline dominante de la época: se detectan **esquinas Harris3D** (STIP), se computa un descriptor **HOG/HOF de 162 dimensiones** (apariencia + movimiento) por punto, se agrupan ~100 000 STIP con k-means en un vocabulario de **k = 4000** palabras visuales, cada clip queda como un histograma de 4000 dimensiones, y un **SVM no lineal** con kernel de intersección de histogramas clasifica las 101 acciones. El protocolo recomendado es la **validación cruzada de 25 folds *leave-one-group-out***, que impide la fuga de información al dejar grupos completos fuera.

La exactitud global es **44,5 %**, con fuerte estructura por tipo: *Sports* lidera con **50,54 %** (movimientos distintivos, fondos menos saturados) y *Human-Object Interaction* queda al fondo con **38,52 %** (fondos muy abarrotados, movimiento informativo que ocupa poca fracción del clip). Un 44,5 % sobre 101 clases —contra el ~100 % trivial de KTH— **no es un fracaso: es el objetivo del diseño**. UCF101 fue construido para ser difícil.

## Impacto

UCF101 se convirtió en el **benchmark estándar de reconocimiento de acciones durante buena parte de la década de 2010**. Casi todos los hitos de *deep learning* para video que la Clase 36 recorre reportaron sobre él: las primeras **CNN 2D** cuadro a cuadro, las arquitecturas **2D CNN + RNN**, los **two-stream networks** (apariencia + flujo óptico) y las **CNN 3D** (C3D, I3D), habitualmente junto a [HMDB51](/papers/hmdb-kuehne-2011). Su longevidad hizo que la exactitud se saturara (los mejores modelos superaron el 95 %), lo que motivó datasets aún mayores —Sports-1M, Kinetics, ActivityNet—. Aun así, siguió usándose como *dataset de sanidad* y para pre-entrenamiento/*fine-tuning*.

## Limitaciones

- **Clips cortos y recortados (*trimmed*).** Cada clip contiene una sola acción ya segmentada (7,21 s de media). UCF101 resuelve **clasificación**, no **detección/localización temporal** en video largo sin recortar —el problema realista de "¿qué acción y *cuándo*?"— que atacarían ActivityNet y THUMOS.
- **Sesgo de YouTube.** El corpus proviene de video subido y filtrado manualmente, con sesgos de selección, producción (deportes sobre-representados) y culturales/geográficos. Las 101 clases son un recorte particular, no una muestra representativa.
- **Baja resolución y compresión.** 320 × 240 a 25 fps con DivX es modesto incluso para 2012; se pierden detalles finos de movimiento.
- **Desbalance moderado.** Clips y duración por clase no son uniformes, y los cinco tipos tienen tamaños muy distintos (Sports 50 clases; Human-Human Interaction apenas 5).

## Por qué importa para la Clase 36

La [Clase 36](/clases/clase-36) (Introduction to Video Analysis) cubre definición de video, *object tracking*, *optical flow* y *action recognition*, y sus primeros enfoques de *deep learning* (CNN 2D, CNN 2D + RNN). UCF101 encaja en varios de sus ejes:

- **Tarea.** Es el ejemplo canónico de **clasificación de acciones sobre clips recortados**, la formulación más simple con la que la clase abre el tema.
- **Desafíos.** Movimiento de cámara, fondo abarrotado, iluminación variable y oclusión —los desafíos que la clase enumera— son constitutivos del régimen "in the wild", y el baseline los cuantifica (deportes fácil, interacción humano-objeto difícil).
- **Optical flow.** El descriptor HOF del baseline y los two-stream posteriores hacen del flujo óptico —tema propio de la clase— la señal de movimiento central; UCF101 es donde se midió cuánto aporta el flujo frente a la sola apariencia.

El vínculo más directo: **el laboratorio de la clase usa UCF11**, el eslabón temprano del mismo [linaje UCF](/dominios/video). Leer este paper permite ver el mismo problema a dos escalas —11 clases y 1168 clips en el lab, 101 clases y 13 320 clips en el benchmark—, con idéntica estructura (video de YouTube, grupos con fondo/actor compartidos, clips cortos) y solo cambiando la magnitud y, con ella, la dificultad.
