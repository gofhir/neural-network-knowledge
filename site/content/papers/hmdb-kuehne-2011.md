---
title: "HMDB: A Large Video Database for Human Motion (2011)"
weight: 397
math: true
---

{{< paper-card
    title="HMDB: A Large Video Database for Human Motion Recognition"
    authors="H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, T. Serre"
    year="2011"
    venue="ICCV 2011"
    pdf="/papers/hmdb-kuehne-2011.pdf" >}}
HMDB51 fue, en su momento, la mayor base de datos de acción disponible: **51 categorías** con al menos 101 clips cada una, para un total de **6766 clips** anotados manualmente y extraídos de fuentes muy diversas —películas digitalizadas, el archivo Prelinger, YouTube y Google Videos—. Nace del grupo de neurociencia computacional de Poggio y Serre (MIT/Brown), cuyo modelo del córtex visual es a la vez uno de los baselines y la motivación biológica del trabajo. Cada clip fue validado por dos observadores y anotado con **meta-etiquetas** ricas (parte del cuerpo visible, movimiento de cámara, punto de vista, número de personas, calidad). Donde KTH y Weizmann ya rozaban el techo (100 %), los mejores sistemas sobre HMDB51 apenas rondaban el **23 %** (azar 2 %). Junto a [UCF101](/papers/ucf101-soomro-2012) formó el par de benchmarks estándar del [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) de la [Clase 36](/clases/clase-36).
{{< /paper-card >}}

---

## Contexto: romper el techo de los datasets controlados

Hacia 2011 la visión ya tenía grandes bases de imágenes realistas (ImageNet, PASCAL VOC), pero los datasets de acción iban por detrás. **KTH** (6 acciones), **Weizmann** (9) e **IXMAS** (11) compartían un patrón: un único actor escenificado, sin oclusión, fondo limpio, iluminación uniforme y cámara fija. La consecuencia era la **saturación**: 12 de 21 sistemas superaban 90 % en KTH, y en Weizmann 3 de 16 alcanzaban 100 %. Cuando un benchmark se resuelve casi por completo, deja de discriminar entre métodos.

Datasets más recientes (Hollywood, UCF Sports, UCF YouTube, UCF50) ya eran más difíciles, pero los autores identifican un **sesgo crítico**: en los datasets de deportes, las acciones se distinguen por señales **estáticas de forma o de escena**, sin analizar el movimiento. Lo demuestran con dos experimentos elegantes. Primero, **las poses estáticas bastan en UCF Sports**: un clasificador basado solo en 14 articulaciones por frame alcanza **más del 98 %** (azar 11 %), volviendo innecesaria la cinemática. Segundo, **la escena delata la acción**: el descriptor global de escena **gist** predice la categoría mejor que las features espacio-temporales de nivel medio. En contraste, el experimento análogo sobre HMDB51 (10 categorías comparables) da solo **35 %** con poses estáticas frente a **54 %** con features de movimiento —confirmando que sus categorías se distinguen **por el movimiento, no por la pose estática**—.

## Contribución: HMDB51 y sus meta-anotaciones

Las 51 categorías se agrupan en cinco tipos: acciones faciales generales, faciales con manipulación de objetos, movimientos corporales generales, corporales con interacción de objetos, y corporales para interacción humana. Se generaron pidiendo a estudiantes anotar segmentos con **una única acción no ambigua** (actor de al menos 60 px, mínimo 1 s), partiendo de más de 60 categorías reducidas a 51 reteniendo las de al menos 101 clips.

Lo que distingue a HMDB51 no es solo el tamaño, sino su **capa de meta-información** por clip, un verdadero instrumento de diagnóstico:

- **Parte del cuerpo visible:** cuerpo completo 56,3 %, superior 30,5 %, cabeza 12,3 %, inferior 0,8 %.
- **Movimiento de cámara:** presente en 59,9 % (aproximadamente **dos tercios** de los clips).
- **Punto de vista** (frente 40,8 %, izquierda 22,1 %, derecha 19,0 %, atrás 18,2 %).
- **Calidad del clip:** alta 17,1 %, media 62,1 %, baja 20,8 %.

Estas etiquetas permiten preguntar cuánto cae el desempeño con movimiento de cámara o al bajar la calidad —el análisis que el paper realiza en su evaluación—.

## Composición y estabilización de video

La evaluación usa **tres particiones** train/test (70/30 por categoría) con dos restricciones: **no fuga** —clips del mismo video fuente no aparecen en train y test a la vez— y **balance de meta-etiquetas**. Para asegurar que los tres splits no estén correlacionados, se eligieron de forma codiciosa minimizando la **distancia de Hamming normalizada**. Todo el video se normalizó a 240 px de altura y 30 fps con códec DivX.

Un aporte metodológico central es la **estabilización de video**, que separa el movimiento de la cámara del movimiento del sujeto. El problema es conceptual: un descriptor de flujo óptico no distingue si el patrón que observa proviene del brazo del actor o de un paneo. El procedimiento usa *image stitching*: se emparejan features salientes entre frames vecinos, se estima la transformación geométrica con **RANSAC** y se deforma (warp) todo el clip a un plano de fondo común. Se reporta el desempeño tanto en clips originales como estabilizados, para medir si eliminar el movimiento de cámara ayuda —una pregunta cuya respuesta resultó contraintuitiva—.

## Baselines y hallazgos

Se evalúan dos sistemas clásicos: **HOG/HOF** (bag-of-words espacio-temporal de Laptev sobre esquinas Harris3D, con SVM de kernel RBF $K(u,v)=\exp(-\gamma\lVert u-v\rVert^2)$) y las **features C2**, un modelo jerárquico inspirado en el córtex visual de primates con streams ventral (forma) y dorsal (movimiento) concatenados. Ambos rinden apenas por encima del **20 %** (azar 2 %): HOG/HOF 20,44 % → 21,96 % estabilizado; C2 22,83 % → 23,18 %.

Tres hallazgos importan:

- **La estabilización ayuda solo marginalmente** (+1,5 pts HOG/HOF, +0,35 C2). Aunque el movimiento de cámara *debería* contaminar el cómputo de movimiento, corregirlo no rinde la mejora esperada.
- **La calidad del clip es el factor dominante.** Ni oclusiones ni posición de cámara influyen tanto: de alta a baja calidad, ambos sistemas caen ~**10 puntos**. Una regresión logística confirma que la calidad es el factor de mayor peso por lejos.
- **El movimiento de cámara a veces ayuda a C2** (25,20 % vs. 19,13 %), presumiblemente porque incrementa la respuesta de sus detectores de movimiento S1. Clasificar solo los parámetros de movimiento estimados da 5,29 %: el movimiento de cámara por sí solo no predice la acción.

El análisis forma vs. movimiento muestra que las señales de movimiento solas (C2-Motion 21,96 %) superan a las de forma (C2-Shape 13,40 %): para las acciones complejas de HMDB51 **el movimiento es más poderoso que la forma**.

## Impacto

HMDB51 se volvió, junto a UCF101, el **par de benchmarks estándar del reconocimiento de acciones** en la era pre-deep-learning y la transición a ella. El comentario de los autores —que con ~23 % el dataset "es probablemente un buen lugar para empezar", recordando que Caltech-101 arrancó en ~16 %— resultó profético. Cuando llegaron los **two-stream networks**, **C3D** e **I3D**, HMDB51 y UCF101 fueron los datasets de evaluación por defecto; su tamaño modesto los hizo ideales para medir **transferencia** (I3D mostró el valor del pre-entrenamiento en Kinetics haciendo *fine-tuning* sobre ambos). Las tres particiones balanceadas y la restricción de no-fuga siguieron como protocolo citado por más de una década.

## Limitaciones

- **Sigue lejos de la complejidad real.** 51 categorías y ~7000 clips es grande para su época, pero pequeño frente a Kinetics (cientos de miles).
- **Baselines de baja capacidad.** El ~23 % diagnostica tanto la debilidad de las features clásicas como la dificultad del dataset.
- **La estabilización no rinde lo esperado**, dejando abierto si el stitching no elimina toda la interferencia o si las features no aprovechan la señal limpia.
- **Sesgos residuales.** Menos sesgado que los datasets de YouTube, pero no completamente insesgado.

## Por qué importa para la Clase 36

La [Clase 36](/clases/clase-36) (Introduction to Video Analysis) presenta los datasets fundacionales de reconocimiento de acciones, y HMDB51 es uno de los básicos que cita. Tres ideas conviene internalizar:

1. **Un buen benchmark de video debe forzar el modelado del movimiento, no del contexto.** El experimento de poses estáticas (98 % en UCF Sports vs. 35 % en HMDB51) enseña a desconfiar de datasets que se resuelven con señales de forma o escena.
2. **El movimiento de cámara es un confounder de primer orden en video**, y la estabilización es el intento clásico de controlarlo. Aunque aquí rindió poco, la separación entre movimiento del sujeto y del sensor es central en todo el análisis de video posterior (motion compensation, ego-motion, flujo compensado).
3. **Las meta-anotaciones convierten un dataset en un instrumento de diagnóstico.** Poder desglosar el error por calidad, oclusión y movimiento de cámara es lo que reveló que la calidad del clip —no el movimiento de cámara— era el factor dominante. Sin esas etiquetas, ese hallazgo habría sido invisible.

HMDB51 es el complemento directo de [UCF101](/papers/ucf101-soomro-2012): donde este último aporta escala y variedad "in the wild", HMDB51 aporta un diseño deliberadamente **menos sesgado** y una batería de meta-anotaciones que hacen del reconocimiento de acciones un problema medible por condiciones de adquisición.
