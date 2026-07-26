---
title: "Kinetics: el ImageNet del video (2017)"
weight: 398
math: true
---

{{< paper-card
    title="The Kinetics Human Action Video Dataset"
    authors="Will Kay et al. (DeepMind)"
    year="2017"
    venue="arXiv:1705.06950"
    pdf="/papers/kinetics-kay-2017.pdf" >}}
**Kinetics** es el dataset a gran escala que reordenó el reconocimiento de acciones en video: **400 clases** de acción humana, **al menos 400 clips por clase** y **306.245 videos** en total, cada clip de **~10 segundos** y extraído de un **video de YouTube distinto**. Su motivación es explícita: replicar en video lo que **ImageNet** hizo en imágenes —un dataset lo bastante grande para **entrenar redes profundas desde cero** y lo bastante difícil para servir de benchmark—. Los datasets previos (HMDB-51, UCF-101) ya no alcanzaban: UCF-101 tenía 13.320 clips pero solo **2.500 videos** (baja variación). Kinetics multiplica la variedad por diseño con **un único clip por video**, y es la infraestructura sobre la que nace [I3D](/papers/i3d-carreira-2017). Es el pilar de datos del [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) que presenta la [Clase 36](/clases/clase-36).
{{< /paper-card >}}

---

## Contexto: por qué UCF-101 y HMDB-51 ya no alcanzaban

Kinetics se presenta como el **sucesor** de los dos benchmarks estándar del área: **HMDB-51** (2011, 51 clases, 6.766 clips) y **UCF-101** (2012, 101 clases, 13.320 clips). El paper reconoce que "sirvieron muy bien a la comunidad", pero que su utilidad "está expirando": no son lo bastante grandes ni variados para entrenar la generación moderna de modelos profundos.

El problema de fondo de UCF-101 no es el conteo, sino la **falta de variación**: sus 13.320 clips provienen de apenas **2.500 videos distintos** —hay 7 clips de un mismo video de la misma persona cepillándose el pelo—, de modo que muchos comparten intérprete, punto de vista, iluminación y fondo. Kinetics lo evita por diseño: **cada clip proviene de un video diferente** —su número de clips totales iguala exactamente su número de videos (306.245)—, lo que multiplica la variedad de intérpretes, poses, velocidades y encuadres. La contracara de usar YouTube es que son videos **amateur** (temblor de cámara, iluminación variable, desorden de fondo); los autores lo presentan como una virtud que hace del dataset un benchmark genuinamente difícil.

## Contribución central: un benchmark de escala para el video profundo

La contribución es el **dataset mismo** como infraestructura, apoyado en tres decisiones de diseño:

1. **Escala suficiente para entrenar desde cero.** Con al menos 400 clips por clase y más de 306.000 videos, Kinetics es "un orden de magnitud más grande" que sus predecesores, y permite por primera vez entrenar arquitecturas masivamente parametrizadas —como los **ConvNets 3D**— sin depender de preentrenamiento en imágenes.
2. **Un clip por video, para maximizar variación.** Menos correlación entre ejemplos significa un benchmark más honesto y modelos que generalizan mejor.
3. **Foco en clasificación, no en localización temporal.** Solo se incluyen clips cortos de ~10 s que contienen la acción; no hay videos sin recortar (*untrimmed*).

El pipeline de construcción es en sí una contribución: candidatos buscados en **YouTube** (emparejando títulos con la lista de acciones, verbos en gerundio "-ing"), posicionamiento temporal con clasificadores de imagen a nivel de fotograma, verificación en **Amazon Mechanical Turk** (al menos **3 de 5** respuestas positivas, sin acceso al audio para forzar clasificación puramente visual) y una limpieza final con de-duplicación por características **Inception-V1** (umbral coseno 0.97) que descartó cerca del 15% adicional de ejemplos aprobados. La anotación es **no exhaustiva** —un clip con varias acciones se lista bajo una sola—, de ahí que la métrica adecuada sea **top-5**, exactamente como en ImageNet.

## Composición y baselines

El dataset final tiene **400 clases**, con entre **400 y 1.150 clips por acción**, divididas en entrenamiento (250–1.000 videos por clase), validación (50 por clase) y prueba (100 por clase). El paper evalúa tres baselines —**ConvNet+LSTM** (29M parámetros), **Two-Stream** (48M) y **3D-ConvNet** (79M)— todos **entrenados desde cero** sobre Kinetics. Los resultados (top-1 / top-5): Two-Stream RGB+Flow alcanza **61.0 / 81.3**, ConvNet+LSTM **57.0 / 79.0** y 3D-ConvNet RGB **56.1 / 79.5**.

El hallazgo que define la historia posterior: el **3D-ConvNet**, rico en parámetros y sin preentrenamiento en ImageNet, rinde pobremente en los datasets pequeños (51.6 en UCF-101, 24.3 en HMDB-51) pero en Kinetics **se acerca mucho** a los demás modelos (56.1 RGB) gracias al conjunto de entrenamiento masivo. Es la observación que motiva directamente a [I3D](/papers/i3d-carreira-2017).

## Impacto: preentrenar y transferir

El impacto se entiende por su tesis fundacional, tomada de ImageNet: **entrenar una red grande en un dataset masivo de clasificación y luego transferirla** a tareas más pequeñas. El paper acompañante —Carreira y Zisserman, [I3D](/papers/i3d-carreira-2017)— demuestra el beneficio de **preentrenar en Kinetics** y transferir las características a UCF-101 y HMDB-51, con saltos grandes de exactitud. El patrón "Kinetics-pretrained" se volvió tan común en [video](/dominios/video) como "ImageNet-pretrained" en imágenes, y prácticamente toda arquitectura de video posterior lo adoptó como estándar de preentrenamiento.

## Limitaciones

- **Sesgo de YouTube.** Videos amateur con temblor, iluminación variable y desorden: realismo, pero una distribución de fuente específica que puede no transferir a video en condiciones controladas.
- **Solo clips recortados.** Pensado para clasificación, no localización temporal: no sirve directamente para detectar *cuándo* ocurre una acción en un video largo.
- **Ruido de etiquetas y anotación no exhaustiva.** La etiqueta es "acción supuesta confirmada por 3 de 5 anotadores", con un clasificador RGB en el posicionamiento (potencial sesgo de selección que los autores consideran menor).
- **Desbalance de categorías.** En 340 de 400 clases no hay dominancia de género, pero el desbalance existe en un subconjunto (p. ej. "shaving beard", "cheerleading") y merece estudio más riguroso.
- **Confusiones de grano fino.** Pares como "long jump" vs. "triple jump" fijan un techo práctico de exactitud top-1.

## Por qué importa para la Clase 36

La [Clase 36](/clases/clase-36) (Introduction to Video Analysis) presenta el [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) y sus datasets, y Kinetics es el punto de inflexión de esa historia:

- **La escala del dato es la palanca.** Kinetics no propone una arquitectura; propone **datos suficientes**, y con ellos desbloquea arquitecturas (ConvNets 3D) que antes no rendían. El benchmark correcto reorganiza el campo.
- **Un clip por video importa.** La variación real —no el conteo de clips— es lo que hace generalizar; la decisión de diseño contra la redundancia de UCF-101 es una lección de curación de datasets.
- **Preentrenar y transferir es el patrón dominante.** El valor de Kinetics no está solo en clasificar sus 400 clases, sino en las **representaciones espacio-temporales** que deja para transferir a cualquier tarea de [video](/dominios/video) posterior.
- **Kinetics e I3D nacen juntos.** Este documento construye el dataset; [I3D](/papers/i3d-carreira-2017) introduce el Inflated 3D ConvNet y confirma que "inflar" arquitecturas 2D de ImageNet a 3D y preentrenarlas en Kinetics es la receta ganadora.

Para el video clínico —ecografía, endoscopía, laparoscopía, análisis de marcha— el mensaje es estructural: la **escala de datos etiquetados es el cuello de botella**, no la arquitectura. Preentrenar un backbone espacio-temporal en Kinetics y ajustarlo a un dominio médico con pocos datos es la estrategia análoga a la que ImageNet volvió estándar para imágenes médicas.
