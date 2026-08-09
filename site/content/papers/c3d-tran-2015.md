---
title: "C3D: convoluciones 3D para video (2015)"
weight: 404
math: true
---

{{< paper-card
    title="Learning Spatiotemporal Features with 3D Convolutional Networks"
    authors="Du Tran et al. (FAIR, Dartmouth)"
    year="2015"
    venue="ICCV 2015 / arXiv:1412.0767"
    pdf="/papers/c3d-tran-2015.pdf" >}}
**C3D** (*Convolutional 3D*) propone aprender **features espacio-temporales** de video con redes convolucionales tridimensionales entrenadas a gran escala, en vez de procesar cada cuadro por separado con una 2D CNN. La tesis: un buen descriptor genérico de video —el análogo de lo que ImageNet significó para la imagen estática— debe ser **genérico, compacto, eficiente y simple** (funcionar bien incluso con un clasificador lineal). El hallazgo central es que un kernel de convolución **homogéneo de $3\times 3\times 3$** en todas las capas es la mejor elección, porque la convolución y el pooling 3D **preservan la información temporal** en lugar de colapsarla tras la primera capa. Con un simple SVM lineal, las features C3D **igualan o superan el estado del arte en 6 benchmarks** de análisis de video, alcanzan **52.8% en UCF101 con solo 10 dimensiones** y corren a **313 fps** (91× más rápido que iDT). Es una de las dos grandes vías para dar "sentido temporal" a una red de video que introduce la [Clase 36](/clases/clase-36).
{{< /paper-card >}}

---

## Contexto: el tiempo como dimensión que la 2D CNN pierde

El video tiene estructura **espacial** (dentro de cada cuadro) y **temporal** (a lo largo de los cuadros), y el problema de fondo es modelar ambas a la vez. En 2015 había tres respuestas. Las **features de imagen (2D CNN)** transferidas cuadro a cuadro describen apariencia pero **carecen de modelado del movimiento**: una convolución 2D aplicada sobre múltiples cuadros tratados como canales **también produce una imagen**, es decir la red pierde la información temporal inmediatamente después de la primera convolución. Las **trayectorias densas mejoradas (iDT)** siguen puntos con flujo óptico y descriptores hechos a mano (HOG, HOF, MBH), pero son computacionalmente intensivas e intratables a gran escala. El enfoque **two-stream** (Simonyan y Zisserman, 2014) combina una red RGB con una red de **flujo óptico precomputado**, pero inyecta el movimiento desde afuera y su stream temporal, al ser 2D, también colapsa el tiempo tras la primera capa. C3D promete prescindir de todo eso: aprender apariencia y movimiento **simultáneamente, de extremo a extremo, directamente de los píxeles**.

## Método: la convolución 3D y el kernel temporal óptimo

La diferencia con la 2D ConvNet es geométrica. La convolución 2D desliza un kernel $k\times k$ sobre alto y ancho; la **convolución 3D** desliza un kernel $d\times k\times k$ sobre **tres** dimensiones: alto, ancho **y tiempo**. Esa profundidad temporal $d$ hace que una sola respuesta del filtro dependa de $d$ cuadros consecutivos, capturando cómo cambian los píxeles de un instante al siguiente. La consecuencia decisiva: aplicar convolución 3D sobre un volumen de video **produce otro volumen**, preservando la dimensión temporal en la salida, de modo que la señal temporal se propaga a través de **todas** las capas de la red.

El paper notación clips como $c\times l\times h\times w$ (canales, longitud, alto, ancho) y kernels como $d\times k\times k$. Para decidir cuántos cuadros debe abarcar el kernel en el tiempo, los autores **fijan el campo receptivo espacial en $3\times 3$** (la lección de VGG) y **varían solo $d$**. Un detalle clave: todas las variantes tienen **prácticamente el mismo número de parámetros** —la mayor diferencia entre `depth-1` y `depth-7` es de 51K parámetros, menos del **0.3%** de los 17.5M totales—, así que cualquier diferencia de desempeño se atribuye limpiamente a $d$, no al tamaño del modelo. El resultado sobre UCF101: `depth-3` es el mejor, y `depth-1` (equivalente a convoluciones 2D por cuadro) es **significativamente peor** por falta de modelado del movimiento. Conclusión: **$3\times 3\times 3$ es la mejor elección de kernel**.

La arquitectura **C3D** resultante tiene **8 capas de convolución** ($3\times 3\times 3$, stride 1, filtros de 64 a 512), **5 de max pooling** ($2\times 2\times 2$, excepto pool1 que es $1\times 2\times 2$ para **no fusionar la señal temporal demasiado temprano**), **2 capas fully connected** de 4096 unidades (fc6, fc7) y una softmax. Se entrena sobre **Sports-1M** (1.1M videos deportivos, 487 categorías) con clips de $16\times 112\times 112$. Una vez entrenada, se usa como **extractor de features de propósito general**: se promedian las activaciones de fc6 sobre los clips de 16 cuadros (con 8 de solapamiento), se normaliza con L2 y se obtiene un **descriptor de 4096 dimensiones** que alimenta un SVM lineal, sin fine-tuning por tarea. Mediante deconvolución, los autores observan que C3D **empieza atendiendo la apariencia y luego rastrea el movimiento saliente** —a diferencia del flujo óptico, que se dispara en todos los píxeles en movimiento.

## Resultados

En **Sports-1M**, C3D alcanza 84.4% top-5 desde cero y 85.5% fine-tuneada desde I380K, superando a DeepVideo por ~5 puntos. En **UCF101** (13 320 videos, 101 acciones): C3D de una red + SVM lineal da **82.3%** (solo 4096 dim), tres redes **85.2%**, y combinado con iDT **90.4%**. Con solo RGB, C3D supera a la spatial stream de two-stream por 12.6%, a LRCN por 14.1% y al LSTM composite por 9.4%. En **compacidad** domina el régimen bajo: **52.8% con 10 dimensiones** (más de 20 puntos sobre ImageNet e iDT), 72.6% con 50 y 79.4% con 500. En **ASLAN** (similitud de acciones) logra 78.3% accuracy / 86.5% AUC, +9.6% sobre el estado del arte previo. En **escenas dinámicas** alcanza 98.1% (YUPENN) y 87.7% (Maryland). En **eficiencia** procesa a **313 fps**, 91× más rápido que iDT y 274× más rápido que el flujo óptico GPU de Brox usado por two-stream.

## Limitaciones

- **Muchos parámetros y alto costo de cómputo/memoria.** Del orden de 17.5M parámetros en la variante de estudio; entrenar a $256\times 256$ requiere paralelismo de modelo. Más caro de entrenar que una 2D ConvNet equivalente.
- **Ventana temporal corta y fija.** C3D opera sobre clips de **16 cuadros**; no modela dependencias de largo alcance en un solo pase, y necesita agregación externa (promedio de clips) para razonar sobre videos largos.
- **Dependencia de datos supervisados a gran escala.** El descriptor genérico solo emerge tras entrenar sobre Sports-1M (1.1M videos).
- **Resolución de entrada modesta.** El uso de $112\times 112$ / $128\times 128$ penaliza tareas dominadas por detalle fino de apariencia (ImageNet a resolución completa gana en objetos egocéntricos).

## Por qué importa para la Clase 36

En el marco de la [Clase 36](/clases/clase-36), la pregunta rectora es cómo introducir información temporal más allá de una 2D CNN por cuadro, y C3D encarna **una de las dos vías principales** para ese "sentido temporal", central al [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones):

- **Vía convolución 3D (C3D):** el tiempo es una dimensión más del kernel; el movimiento se aprende **implícitamente y de extremo a extremo** desde los píxeles RGB, sin señales externas. Directa y unificada, pero cara en parámetros y cómputo.
- **Vía 2D CNN + modelado temporal externo:** 2D CNN + RNN (LRCN, LSTM composite) o el **two-stream**, que necesita flujo óptico precomputado. C3D supera a ambas y evita ese flujo costoso (de ahí su ventaja de 91–274× en velocidad).

El desenlace histórico de esta línea es [I3D](/papers/i3d-carreira-2017) (Carreira y Zisserman, 2017), el sucesor directo que corrige las limitaciones de C3D: en lugar de entrenar una 3D ConvNet desde cero, **"infla"** arquitecturas 2D ya preentrenadas en ImageNet y las entrena sobre Kinetics, heredando el poder de los backbones 2D y superando a C3D. C3D es el paso conceptual que **instala la idea de las features espacio-temporales aprendidas**; I3D es la ingeniería que la vuelve competitiva a escala. La [Clase 38](/clases/clase-38) trata este contraste como su tema central, y ahí se muestra un detalle que matiza la comparación de parámetros: de los ~78M de C3D, unos 50M viven en sus dos capas densas `fc6` y `fc7`, no en sus convoluciones 3D —así que la reducción a 25M de I3D se debe sobre todo a heredar la topología de Inception, no al inflado en sí. Tres años después, el propio Du Tran revisa esta arquitectura en [R(2+1)D](/papers/r2plus1d-tran-2018). Para video clínico —endoscopía, laparoscopía, monitoreo de pabellón— la enseñanza es directa: el diagnóstico rara vez vive en un solo cuadro, sino en el movimiento, y un extractor tipo C3D permite un pipeline compacto y de extremo a extremo sin calcular flujo óptico externo.
