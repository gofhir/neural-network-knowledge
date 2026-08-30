---
title: "End-to-End Audiovisual Speech Recognition (2018)"
weight: 459
math: true
---

{{< paper-card
    title="End-to-End Audiovisual Speech Recognition"
    authors="Stavros Petridis, Themos Stafylakis, Pingchuan Ma, Feipeng Cai, Georgios Tzimiropoulos, Maja Pantic (Imperial College London / University of Nottingham)"
    year="2018"
    venue="ICASSP 2018 / arXiv:1802.06424"
    arxiv="1802.06424"
    pdf="/papers/e2e-avsr-petridis-2018.pdf" >}}
El primer modelo de fusión audiovisual que aprende a extraer features **simultáneamente de los píxeles de la boca y de la forma de onda cruda**, sin MFCC ni transformadas intermedias, y clasifica palabras en contexto sobre un dataset grande y no controlado (LRW, 500 palabras de la BBC). Dos flujos —ResNet-34 con frente 3D para el video, ResNet-18 con núcleos 1D para el audio—, cada uno con su BiGRU de dos capas, y una tercera BiGRU que fusiona. El resultado en audio limpio es deliberadamente modesto: **98,0 % contra 97,7 %** del audio solo, apenas +0,3 puntos. Bajo ruido, la historia cambia por completo: **+14,1 puntos a −5 dB**. El paper vale menos por su número principal que por la figura que muestra por qué la fusión existe.
{{< /paper-card >}}

---

## La pregunta

La clase la formula tal como aparece en el paper: *"pero, ¿por qué necesitamos video si tenemos el audio?"*

La respuesta no es que el video aporte información que el audio no tenga. En condiciones limpias, el audio es enormemente superior —97,7 % contra 82,0 %— y agregarle video casi no cambia nada. La respuesta es que **el ruido acústico no afecta al canal visual**, así que cuando el audio se degrada, el video sigue exactamente donde estaba.

## Arquitectura

**Flujo visual.** Entrada de 29 fotogramas de recortes de boca en escala de grises. Una convolución espacio-temporal de 64 núcleos de $5\times 7\times 7$ (tiempo × alto × ancho) captura la dinámica corta, seguida de una **ResNet-34** que reduce la dimensión espacial hasta un vector por paso temporal, y una **BiGRU de 2 capas** con 1024 celdas.

La ResNet se entrena **desde cero**. Los autores lo justifican: los pesos preentrenados *"están optimizados para tareas completamente distintas (por ejemplo, imágenes estáticas a color de ImageNet o CIFAR)"*.

**Flujo de audio.** No hace falta convolución espacio-temporal porque la onda es 1D. Una **ResNet-18** con núcleos unidimensionales, donde la primera capa usa un **núcleo temporal de 5 ms con paso de 0,25 ms** para extraer información espectral fina; las capas siguientes usan núcleos de 3 y capturan características de largo plazo. La salida se divide en **29 ventanas por *average pooling*** para igualar la tasa de fotogramas del video. Luego, otra BiGRU de 2 capas.

**Fusión.** Las salidas de ambas BiGRU se concatenan y entran a una **tercera BiGRU de 2 capas** que modela la dinámica temporal conjunta. Un softmax etiqueta cada instante y la secuencia se decide por la probabilidad promedio más alta.

{{< concept-alert type="clave" >}}
El diseño alinea las dos modalidades **a nivel de tasa de muestreo** antes de fusionarlas: el *average pooling* del flujo de audio existe únicamente para producir 29 ventanas y poder concatenar con los 29 fotogramas. Es fusión **intermedia** — ni concatenar señales crudas ni promediar decisiones finales. Ver [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual).
{{< /concept-alert >}}

## El currículo de entrenamiento

El paper es explícito: *"entrenar directamente end-to-end cada flujo lleva a un rendimiento subóptimo"*. De ahí cinco etapas:

1. Cada flujo con un *back-end* **convolucional temporal**, hasta que no mejore por 5 épocas.
2. Se reemplaza ese back-end por la **BiGRU**, con el frente 3D y la ResNet **congelados**; 5 épocas.
3. Cada flujo completo **end-to-end** (Adam, lote de 36, lr $3\times 10^{-4}$, parada temprana con paciencia 5).
4. Se agrega la BiGRU de fusión con **ambos flujos congelados**; 5 épocas.
5. Toda la red **end-to-end** (lote de 18, lr $10^{-4}$).

Esa complejidad es una de las tres desventajas que el propio paper declara al cerrar.

**Aumentación.** En video, recortes aleatorios y volteo horizontal con probabilidad 50 % aplicado a todos los fotogramas del clip. En audio, **ruido de tipo *babble* de la base NOISEX a un nivel elegido uniformemente entre −5 y 20 dB**, o audio limpio. Esta última decisión es la que hace que la BiGRU de fusión aprenda a **ponderar según la condición** en vez de promediar a ciegas.

## LRW

*Lip Reading in the Wild* (Chung y Zisserman, 2016): segmentos de 1,16 s (29 fotogramas) de programas de la BBC, principalmente noticias y entrevistas. **500 palabras**, más de 1000 hablantes, gran variación de pose e iluminación. Entre 800 y 1000 secuencias por palabra en entrenamiento: **488 766 / 25 000 / 25 000** ejemplos.

Su dificultad no es solo de escala. Las palabras aparecen **en medio de un enunciado**, con coarticulación de las palabras vecinas en los bordes, y el vocabulario incluye deliberadamente pares visualmente casi idénticos como *America* y *American*. Ver [Lectura de Labios](/fundamentos/lectura-de-labios).

## Resultados

| Flujo | Tasa de clasificación |
|---|---|
| A (end-to-end, onda cruda) | 97,7 |
| A (MFCC) | 97,7 |
| V (end-to-end) | 82,0 |
| V (Stafylakis y Tzimiropoulos, 2017)* | **83,0** |
| V (Chung y Zisserman) | 76,2 |
| V (referencia previa) | 61,1 |
| **A + V (end-to-end)** | **98,0** |

\* usa un recorte de boca calculado con puntos faciales seguidos; este trabajo usa una caja fija.

Dos observaciones que el paper hace sin adornarlas:

**La onda cruda empata con MFCC: 97,7 contra 97,7.** Los autores lo llaman *"un resultado significativo dado que la entrada al sistema es solo la forma de onda"* — pero agregan de inmediato que *"el esfuerzo requerido para entrenar el sistema end-to-end es significativamente mayor que el de la BiGRU de 2 capas usada con MFCC"*. En condiciones limpias, ochenta años de procesamiento de señales siguen siendo competitivos con una ResNet-18.

**Su propio flujo visual pierde contra la referencia**, 82,0 contra 83,0, y explican por qué. La contribución no está en el canal visual.

## La figura que justifica el paper

El resultado central no es la tabla sino la curva de exactitud contra relación señal-ruido:

| SNR | mejora de A end-to-end sobre MFCC | mejora de A+V sobre A |
|---|---|---|
| 5 dB | +0,9 | +1,3 |
| 0 dB | +3,5 | +3,9 |
| **−5 dB** | **+7,5** | **+14,1** |

Y en la figura, la línea del flujo **visual solo es horizontal**: ~83 % a −5 dB y ~83 % a 20 dB. El ruido acústico no la toca. A −5 dB, el video solo **supera** al audio solo.

{{< concept-alert type="clave" >}}
Los dos resultados de la tabla son distintos y ambos importan.

**La onda cruda es más robusta al ruido que MFCC** (+7,5 a −5 dB) aunque empaten en limpio. Los MFCC descartan información —fase, estructura fina— que resulta ser justamente la que ayuda a separar habla de ruido. Es el argumento general a favor de aprender la representación en vez de fijarla, y solo se ve al salir de las condiciones de laboratorio.

**La fusión aporta donde la modalidad fuerte falla.** +0,3 puntos en limpio, +14,1 a −5 dB. No es que la fusión funcione mal en condiciones limpias: es que no hay nada que arreglar. La [práctica](/clases/clase-43/practica) de la clase reproduce esa forma desde cero.
{{< /concept-alert >}}

## Limitaciones

Las que el paper declara en su última diapositiva y sección:

- **Limitado a un conjunto fijo de palabras aisladas.** 500 clases con un softmax; no transcribe habla continua.
- **El proceso de entrenamiento es muy complejo.** Cinco etapas con congelamientos y descongelamientos sucesivos.
- **No generaliza bien a variaciones en el largo de la secuencia.** Todo el diseño supone 29 fotogramas.

La salida a las tres es la misma que en [reconocimiento de voz](/fundamentos/reconocimiento-de-voz): reemplazar el softmax por [CTC](/fundamentos/ctc-loss) o *seq2seq* con atención. [LipNet](/papers/lipnet-assael-2016) ya lo había hecho para video en 2016; [AV-HuBERT](/papers/av-hubert-shi-2022) lo cierra en 2022 agregando además preentrenamiento autosupervisado.

## Por qué importa para la Clase 43

Es el segundo paper de la [Clase 43](/clases/clase-43) y su contrapunto exacto. [SoundNet](/papers/soundnet-aytar-2016) usa una modalidad para **enseñarle** a la otra en tiempo de entrenamiento; aquí las dos se usan **juntas en inferencia** para decidir. Son los dos modos de aprovechar la misma propiedad —que imagen y sonido describen la misma escena—, y la clase los presenta seguidos por eso.

Para la trayectoria del diplomado, esta es la clase donde confluyen los dos hilos que venían alternándose: video ([36](/clases/clase-36), [38](/clases/clase-38), [40](/clases/clase-40), [42](/clases/clase-42)) y audio ([35](/clases/clase-35), [37](/clases/clase-37), [39](/clases/clase-39), [41](/clases/clase-41)). Y aparecen componentes de casi todas: ResNet, convoluciones 3D, GRU bidireccionales, onda cruda, aumentación con ruido.

## El paper ejecutado

El [Laboratorio 43](/laboratorios/lab-43) corre este modelo sobre el test set completo de LRW y lo reproduce con **98,84 %** contra el 98,0 % reportado. Tres cosas que la ejecución hace visibles y el texto no menciona:

- **El recorte de audio de 19.456 muestras es 29 × 672**, el largo exacto que hace que la cadena de convoluciones produzca un frame cada 42 ms — la cadencia de 25 fps del video. La sincronización entre modalidades está horneada en la aritmética de los strides, sin ningún módulo de alineamiento.
- **Los checkpoints publicados conservan el backend temporal-convolucional de la fase 1** —el que el paper dice haber "removido"— más un BiLSTM completo de una versión anterior del código. Son 34,9 M de parámetros inertes, el 39 % del archivo: el registro material del entrenamiento por etapas descrito en la sección 4.3.
- **Los 29 errores del sistema son todos vecinos fonológicos**, sin una sola confusión arbitraria. `THERE → THEIR` es irreducible —son homófonos perfectos en inglés británico—, lo que implica que LRW tiene un techo teórico bajo el 100 % por construcción del vocabulario.

---

**Ver también:** [Laboratorio 43](/laboratorios/lab-43) · [SoundNet (2016)](/papers/soundnet-aytar-2016) · [Stafylakis y Tzimiropoulos (2017)](/papers/lipreading-resnet-stafylakis-2017) · [LipNet (2016)](/papers/lipnet-assael-2016) · [AV-HuBERT (2022)](/papers/av-hubert-shi-2022) · [Lectura de Labios](/fundamentos/lectura-de-labios) · [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual) · [Clase 41 - ASR](/clases/clase-41)
