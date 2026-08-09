---
title: "Teoría - CNN para reconocimiento en video"
weight: 10
math: true
---

> **Recorrido de la Clase 38** del Diplomado IA UC (Bianca Del Solar Medrano). La clase revisita la escalera de arquitecturas de video —**CNN2D + agrupación temporal → CNN2D + RNN → Two-Stream → C3D → I3D**— pero con un hilo conductor distinto al de la [Clase 36](/clases/clase-36): el subtítulo de la clase es **"modelos pre-entrenados"**, y esa es la clave que ordena toda la secuencia. Cada arquitectura de la escalera se puede leer como una respuesta a la misma pregunta incómoda: *¿de dónde saco los pesos iniciales para un modelo de video, si el único gran corpus etiquetado que existe es de imágenes?*

{{< concept-alert type="clave" >}}
**El hilo que hay que seguir.** La clase enumera ventajas y desventajas de cinco arquitecturas. Si se leen las desventajas en orden, aparece un patrón: las arquitecturas 2D son fáciles de entrenar porque **heredan ImageNet**, pero modelan mal el tiempo. Las arquitecturas 3D modelan bien el tiempo, pero —dice el slide de C3D— **"no puede aprovechar el pre-entrenamiento de ImageNet"**. I3D no gana por ser más lista con el tiempo: gana porque encuentra la manera de tener las dos cosas.
{{< /concept-alert >}}

---

## 1. ¿Qué es la comprensión de video?

La clase abre con una definición operacional (slide 3): el objetivo es **comprender lo que sucede en un video** — cuatro preguntas concretas.

| Pregunta | Tarea asociada |
|---|---|
| ¿Qué acción se lleva a cabo? | Clasificación de acciones |
| ¿Quién la realizó? | Detección de personas, re-identificación |
| ¿Dónde se lleva a cabo? | Localización espacial |
| ¿En qué parte del video sucede? | Localización temporal |

Vale detenerse en que las cuatro no son la misma dificultad. La escalera de arquitecturas que sigue la clase resuelve fundamentalmente **la primera**: recibe un clip recortado y emite una etiqueta. Las otras tres son problemas de localización, y es la razón por la que datasets posteriores como [AVA](/papers/i3d-carreira-2017) y benchmarks de detección temporal existen aparte. Cuando el slide de I3D dice que "la inferencia no es más rápida que los modelos anteriores", el costo importa precisamente porque en localización hay que evaluar el modelo muchas veces sobre ventanas deslizantes.

## 2. ¿Por qué es útil?

El slide 4 nombra **seguridad, análisis deportivo y conducción autónoma** como campos donde la comprensión de video alimenta la toma de decisiones. Es una lista corta pero honesta sobre la economía del campo: son exactamente los tres dominios donde alguien paga por procesar video en volumen, y por eso son los que financiaron los datasets. El fundamento [Análisis de Video](/fundamentos/analisis-de-video) desarrolla el panorama de aplicaciones y el fundamento [Reconocimiento de Acciones](/fundamentos/reconocimiento-de-acciones) el de las tareas.

## 3. Repaso: cómo funcionan las CNN sobre imágenes

Los slides 5 a 9 son un repaso deliberado de CNN 2D, porque —dice el slide 6— **"los primeros modelos convolucionales para la comprensión de video son una extensión de aquellos utilizados para el reconocimiento de imágenes"**. La frase parece de transición pero es la tesis de la clase: todo lo que viene son extensiones, y el valor de una extensión se mide por cuánto conserva de lo que ya funcionaba.

### 3.1 La jerarquía de capas

El slide 7 describe el argumento clásico: las primeras capas detectan **líneas y curvas**, y las capas profundas se especializan hasta reconocer **formas complejas** como un rostro o la silueta de un animal. Esa jerarquía es lo que hace que el pre-entrenamiento funcione: las capas bajas aprenden detectores de bordes que son útiles para *cualquier* tarea visual, así que reutilizarlas es casi gratis. Se desarrolla en el fundamento [Redes Convolucionales](/fundamentos/redes-convolucionales).

### 3.2 El costo de la entrada

El slide 8 hace la cuenta que motiva todo lo demás. Una imagen de $28 \times 28$ en escala de grises son $784$ neuronas de entrada. A color, con tres canales:

$$28 \times 28 \times 3 = 2352 \text{ neuronas}$$

Y el slide 9 continúa: si la primera convolución aplica 32 filtros y cada uno produce un mapa de $28 \times 28$, la primera capa oculta tiene

$$32 \times 28 \times 28 = 25\,088 \text{ neuronas}$$

Este número es el que hay que retener, porque **el video lo multiplica otra vez**. Un clip de 64 frames —la longitud que usa I3D en entrenamiento— multiplica la entrada por 64 antes de que la red haya hecho nada. Toda la ingeniería de las secciones siguientes es, en el fondo, contabilidad sobre ese factor.

{{< concept-alert type="advertencia" >}}
**Kernel, filtro y la nomenclatura del slide.** El slide 9 dice que se toman grupos de píxeles cercanos y se opera contra una matriz pequeña llamada **kernel**, y que el conjunto de kernels "se llama filtros". La convención más difundida en la literatura es la inversa: un **filtro** es el volumen completo de pesos que produce **un** mapa de salida (y que internamente tiene un kernel 2D por canal de entrada), mientras que hablar de "los filtros" en plural se refiere al banco entero. La cuenta del slide es correcta —32 filtros dan 32 mapas de salida—; solo conviene tener presente la ambigüedad al leer papers, sobre todo cuando se discuten kernels $1\times1$ o convoluciones separables, donde la distinción deja de ser cosmética.
{{< /concept-alert >}}

---

## 4. La dimensión adicional: el módulo temporal

El slide 10 introduce el esquema que estructura la clase entera. Un modelo de video se descompone en dos responsabilidades:

```
                  ┌─────────────── Video model ───────────────┐
   frames  ──────▶│   CNN-2D          ──▶      Temporal Module │──▶ acción
                  └───────────────────────────────────────────┘
                   entender la              entender el cambio de
                   información              información espacial
                   espacial                 a lo largo del tiempo
```

Esta separación es la que hace legible toda la taxonomía posterior: **las cinco arquitecturas de la clase se diferencian solo en qué pone cada una en la casilla "Temporal Module"**, y en si esa casilla está al final o entretejida en toda la red.

| Arquitectura | ¿Qué ocupa la casilla temporal? | ¿Dónde ocurre? |
|---|---|---|
| CNN2D + agrupación temporal | Un promedio (average pooling) | Solo al final |
| CNN2D + RNN | Una LSTM sobre features por frame | Solo al final |
| Two-Stream | Flujo óptico precomputado + promedio de scores | En la entrada y al final |
| C3D | Convoluciones 3D | En toda la red |
| I3D | Convoluciones 3D infladas + dos corrientes | En toda la red |

La metáfora que acompaña el slide —*"estirando los dos extremos de una banda de goma para que se estire"*— apunta a la idea de tomar algo diseñado para 2D y extenderlo a lo largo del tiempo. Es la misma imagen que reaparece, ahora con contenido técnico preciso, en el inflado de I3D.

---

## 5. CNN2D + agrupación temporal

El primer eslabón (slides 11-12) es el más simple imaginable: pasar cada frame por la misma CNN 2D y **promediar** los resultados. La casilla temporal es un `Avg Pooling`.

| Ventajas (slide 12) | Desventajas (slide 12) |
|---|---|
| Fácil de implementar | No aprovecha la información temporal |
| No es computacionalmente costoso | Tiende a tener un rendimiento deficiente |

La primera desventaja merece precisión, porque es más fuerte de lo que suena. El promedio es una operación **conmutativa**: si se barajan los frames del clip, la salida es idéntica. No es que el modelo aproveche "poco" el tiempo — es que es matemáticamente **invariante al orden**. Formalmente, si $f$ es la CNN por frame y el clip es $\{x_1,\dots,x_T\}$:

$$\hat{y} = g\!\left(\frac{1}{T}\sum_{t=1}^{T} f(x_t)\right) = g\!\left(\frac{1}{T}\sum_{t=1}^{T} f(x_{\pi(t)})\right) \quad \text{para toda permutación } \pi$$

La consecuencia práctica es que este modelo no puede distinguir *abrir una puerta* de *cerrar una puerta*, ni *sentarse* de *pararse*: son el mismo conjunto de frames en orden inverso. La demostración empírica de esto está en el [Laboratorio 36](/laboratorios/lab-36), donde muestrear 4 frames rindió igual o mejor que muestrear 8 — señal de que el modelo estaba usando los frames como votos independientes y no como secuencia. La derivación completa está en la [profundización](profundizacion).

La segunda desventaja ("rendimiento deficiente") tiene un asterisco histórico que conviene conocer: el paper que estableció esta familia, [Karpathy et al. (2014)](/papers/large-scale-video-karpathy-2014), encontró que la variante de un solo frame quedaba a apenas **1-2 puntos** de las variantes que sí mezclaban información temporal. Ese resultado incómodo —que el movimiento aportaba tan poco— fue el que motivó a los papers siguientes a introducir el movimiento **explícitamente** en lugar de esperar que la red lo aprendiera sola.

---

## 6. CNN2D + RNN

El segundo eslabón (slides 13-14) reemplaza el promedio por una **red recurrente**: cada frame pasa por la ConvNet, y la secuencia de features alimenta una LSTM que emite la acción al final.

```
   Action
     ▲
   LSTM ─── ··· ──▶ LSTM
     ▲                ▲
  ConvNet          ConvNet
     ▲                ▲
  Image 1   ···    Image K
                        ───▶ tiempo
```

Este es el diagrama de [LRCN (Donahue et al., 2015)](/papers/lrcn-donahue-2015), y es la familia **(a) LSTM** en la comparativa de I3D que veremos en la sección 10.

| Ventajas | Desventajas |
|---|---|
| Fácil de implementar | No captura movimientos finos de bajo nivel ni relaciones temporales largas |
| Extensión natural de los modelos de imágenes | Costoso: hay que desplegar la red por múltiples frames para la retropropagación |
| Modela la información temporal | |

La primera desventaja tiene una razón estructural que vale explicitar: la LSTM opera sobre features **ya agregados globalmente** por la CNN (después del pooling final, un vector por frame). A esa altura la información espacial fina ya se descartó, así que un desplazamiento de pocos píxeles entre frames consecutivos —justamente el movimiento de bajo nivel— es invisible para la recurrencia. La LSTM puede razonar sobre *"había una persona, después había una persona agachada"*, pero no sobre la velocidad del gesto.

La segunda desventaja es el costo de BPTT sobre $K$ frames, que se cubre en [Backpropagation Through Time](/fundamentos/backpropagation-through-time) y en la [Clase 11](/clases/clase-11).

---

## 7. Two-Stream

El tercer eslabón (slides 15-17) toma una decisión distinta: si la red no aprende el movimiento por sí sola, **se lo damos calculado**. Dos corrientes "individuales e iguales" —misma topología, entradas distintas:

- **Spatial stream ConvNet**: recibe un solo frame RGB. Responde *qué* aparece.
- **Temporal stream ConvNet**: recibe un volumen de **flujo óptico** apilado. Responde *cómo* se mueve.

El slide 16 detalla el procedimiento de [Simonyan y Zisserman (2014)](/papers/two-stream-simonyan-2014) con precisión:

1. Selecciona **aleatoriamente un solo fotograma** del video completo y lo pasa por el stream espacial.
2. Calcula un conjunto de **flujo óptico a partir de 10 fotogramas** para el stream temporal.
3. **Promedia las predicciones** de ambas corrientes (fusión tardía de scores).

{{< concept-alert type="clave" >}}
**Por qué el flujo óptico funciona tan bien.** El flujo óptico es una representación donde el movimiento ya está **explícito y desacoplado de la apariencia**: elimina el fondo estático y deja solo el desplazamiento. La red temporal no tiene que aprender a ignorar la textura del muro para ver el brazo moviéndose — el brazo es lo único que queda en la entrada. Es un caso claro de ingeniería de features que le ahorra capacidad al modelo. Ver [Flujo óptico](/fundamentos/flujo-optico).
{{< /concept-alert >}}

| Ventajas | Desventajas |
|---|---|
| Fácil de implementar | Necesita calcular el flujo óptico de cada video |
| Aprovecha información temporal detallada vía flujo óptico | Solo considera la apariencia de **un** fotograma |
| | No captura relaciones temporales largas |

Las tres desventajas tienen destinos distintos en la historia posterior:

- **"Solo considera un fotograma"** se resuelve al año siguiente con [Feichtenhofer et al. (2016)](/papers/two-stream-fusion-feichtenhofer-2016), que fusiona las corrientes en una capa intermedia en lugar de al final. Esa es la familia **(d) 3D-Fused** de la tabla del slide 26.
- **"No captura relaciones largas"** se ataca con [TSN (Wang et al., 2016)](/papers/tsn-wang-2016), que muestrea segmentos a lo largo de todo el video.
- **"Necesita calcular el flujo óptico"** es la más caras de todas y sobrevive hasta [SlowFast (2019)](/papers/slowfast-feichtenhofer-2019), que la elimina reemplazando la separación *apariencia / movimiento* por una separación *framerate lento / framerate rápido*.

Sobre el costo real del flujo óptico: no es un detalle menor de preprocesamiento. Calcularlo para un dataset del tamaño de Kinetics es un trabajo de días de cómputo, y el resultado ocupa **más espacio en disco que los videos originales** en los formatos habituales. Por eso la desventaja aparece primera en la lista de la profesora.

---

## 8. C3D: el enfoque natural y su precio

El cuarto eslabón (slides 18-21) hace lo que el slide llama "un enfoque natural": si la convolución 2D funciona sobre imágenes, usar **filtros espacio-temporales** sobre volúmenes de video. Las 3D ConvNets son "similares a las redes convolucionales estándar, pero con filtros espacio-temporales".

La arquitectura de [C3D (Tran et al., 2015)](/papers/c3d-tran-2015) que muestra el slide 20 es deliberadamente monótona — ocho convoluciones con kernels $3\times3\times3$ y dos capas totalmente conectadas:

```
Conv1a  Conv2a  Conv3a Conv3b  Conv4a Conv4b  Conv5a Conv5b   fc6    fc7
  64  →  128  →  256  256   →  512  512   →  512  512   →   4096 → 4096 → softmax
     Pool1   Pool2       Pool3         Pool4          Pool5
```

| Ventajas | Desventajas |
|---|---|
| Aprovecha información temporal detallada | **No puede aprovechar el pre-entrenamiento de ImageNet** |
| Crea representaciones jerárquicas espacio-temporales | Muchos más parámetros que Conv2D por la dimensión extra |
| | Es más difícil de entrenar |

{{< concept-alert type="clave" >}}
**La desventaja que define la clase.** "No puede aprovechar el pre-entrenamiento de ImageNet" no es una limitación de implementación: es un problema de **forma de los tensores**. Un kernel de ImageNet es una matriz de $k \times k$; un kernel 3D necesita $t \times k \times k$. No hay manera de cargar el primero en el segundo, así que la red 3D arranca desde inicialización aleatoria. Y como además tiene más parámetros y menos datos que ImageNet para entrenarse, las tres desventajas del slide son la misma desventaja vista tres veces: **sin pesos heredados, una red 3D es una red grande entrenada con pocos datos**.
{{< /concept-alert >}}

Esta era una limitación estructural conocida desde mucho antes. El primer 3D CNN para reconocimiento de acciones, [Ji et al. (2010/2013)](/papers/3d-cnn-ji-2013), es anterior a AlexNet y tenía apenas tres capas convolucionales sobre entradas de $60 \times 40$ píxeles y 7 frames — precisamente porque no había de dónde heredar pesos ni con qué datos entrenar algo más grande. Ese paper también revela lo que estaba en juego: para compensar, incluía una capa "hardwired" con flujo óptico y gradientes **precalculados a mano**. La familia 3D pasó una década pagando el peaje de no tener ImageNet.

---

## 9. I3D: inflar en lugar de reinventar

El quinto eslabón (slides 22-24) es el aporte central de la clase. [I3D (Carreira y Zisserman, 2017)](/papers/i3d-carreira-2017) resuelve la tensión sin sacrificar nada: **infla** un modelo 2D exitoso —Inception-v1, y el mismo truco aplica a ResNet— para que opere en 3D, **heredando sus pesos de ImageNet**.

El slide 23 da el argumento en tres pasos, y es una de las cadenas de razonamiento más elegantes del campo:

1. **Una imagen puede convertirse en un video (aburrido)** copiándola repetidamente en una secuencia.
2. Por lo tanto, los modelos 3D pueden ser **pre-entrenados implícitamente en ImageNet**, satisfaciendo el *punto fijo del video aburrido*.
3. Esto se logra, **gracias a la linealidad**, repitiendo los pesos de los filtros 2D $N$ veces a lo largo de la dimensión temporal y **escalándolos dividiendo por $N$**.

{{< concept-alert type="clave" >}}
**El punto fijo del video aburrido.** La condición que se busca es esta: si a la red 3D se le entrega un "video aburrido" —la misma imagen repetida $N$ veces— debe producir **exactamente la misma activación** que producía la red 2D sobre esa imagen. Si eso se cumple, la red inflada *empieza* siendo tan buena como la red de ImageNet, y el entrenamiento en video solo tiene que mejorarla desde ahí. La división por $N$ es lo que hace que el punto fijo se cumpla: sin ella, la suma sobre $N$ copias idénticas daría una activación $N$ veces más grande y toda la red se saturaría. La derivación formal está en la [profundización](profundizacion), y el mecanismo general en el fundamento [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones).
{{< /concept-alert >}}

El slide 22 muestra la arquitectura **Inflated Inception-V1** con sus campos receptivos anotados capa a capa: la entrada de video pasa por `7×7×7 Conv stride 2`, luego `1×3×3 Max-Pool stride 1,2,2` —nótese el stride temporal de 1, deliberado para no colapsar el tiempo demasiado pronto—, y los campos receptivos crecen de `7,11,11` a `59,219,219` y finalmente `99,539,539` (temporal, alto, ancho). El detalle de que los primeros pooling **no reducen la dimensión temporal** es una decisión de diseño, no un descuido: el tiempo tiene menos resolución que el espacio para empezar, así que decimarlo temprano destruye la señal que se quería capturar.

| Ventajas | Desventajas |
|---|---|
| Aprovecha información temporal detallada | Tiene una gran cantidad de parámetros |
| Crea representaciones jerárquicas espacio-temporales | Es computacionalmente costoso |
| **Puede utilizar el pre-entrenamiento de ImageNet** | La inferencia no es más rápida que los modelos anteriores |
| Reduce el número de parámetros | |
| Reduce la complejidad del entrenamiento | |

{{< concept-alert type="advertencia" >}}
**"Reduce el número de parámetros" y "tiene una gran cantidad de parámetros" en la misma tabla.** No es una contradicción del slide, son dos comparaciones distintas. Frente a **C3D** (79M de parámetros), I3D reduce: usa 25M, porque Inception es una topología mucho más eficiente que la pila de VGG que usaba C3D. Frente a un **modelo 2D**, I3D sigue siendo grande y caro. La tabla de resultados del slide 26 hace visibles ambos hechos a la vez, y por eso vale leerla con cuidado.
{{< /concept-alert >}}

---

## 10. Overview: las cinco familias en un cuadro

El slide 25 reproduce la Figura 5 del paper de I3D, que es el mapa completo del campo en 2017:

| | Familia | Entrada | Módulo temporal |
|---|---|---|---|
| (a) | **LSTM** | Imágenes 1 a K | ConvNet por frame + LSTM |
| (b) | **3D-ConvNet** | Imágenes 1 a K | Convoluciones 3D en toda la red |
| (c) | **Two-Stream** | 1 imagen + flujo óptico 1 a N | Dos ConvNets 2D, fusión de scores |
| (d) | **3D-Fused Two-Stream** | 1 imagen + flujo óptico 1 a N | Dos ConvNets 2D + una 3D ConvNet que las fusiona |
| (e) | **Two-Stream 3D-ConvNet** | Imágenes 1 a K + flujo óptico 1 a K | Dos 3D ConvNets infladas, fusión de scores |

La lectura útil de este cuadro es que **(e) no es una idea nueva, es la conjunción de todas las anteriores**: toma las dos corrientes de (c), las convoluciones 3D de (b), y el inflado que permite pre-entrenarlas. El aporte de I3D es de integración, y funciona porque cada pieza resolvía un problema distinto.

## 11. Results: leer la tabla con cuidado

El slide 26 muestra dos tablas del paper. La primera compara el presupuesto de cada arquitectura:

| Método | #Params | Frames (train) | Huella temporal (train) | Frames (test) | Huella temporal (test) |
|---|---|---|---|---|---|
| ConvNet+LSTM | 9M | 25 RGB | 5 s | 50 RGB | 10 s |
| 3D-ConvNet | 79M | 16 RGB | 0.64 s | 240 RGB | 9.6 s |
| Two-Stream | 12M | 1 RGB, 10 flujo | 0.4 s | 25 RGB, 250 flujo | 10 s |
| 3D-Fused | 39M | 5 RGB, 50 flujo | 2 s | 25 RGB, 250 flujo | 10 s |
| Two-Stream I3D | 25M | 64 RGB, 64 flujo | 2.56 s | 250 RGB, 250 flujo | 10 s |

Dos observaciones que la tabla vuelve evidentes. Primero, **I3D tiene un tercio de los parámetros de C3D** (25M contra 79M) — el "reduce el número de parámetros" del slide anterior, cuantificado. Segundo, I3D entrena sobre una huella temporal de **2.56 segundos contra 0.4 del Two-Stream**: es la arquitectura que ve más tiempo de una sola vez, y eso es lo que le permite explotar el flujo óptico mejor que los demás modelos.

La segunda tabla da la precisión por arquitectura y modalidad:

| Arquitectura | UCF-101 RGB | Flow | RGB+Flow | HMDB-51 RGB | Flow | RGB+Flow | Kinetics RGB | Flow | RGB+Flow |
|---|---|---|---|---|---|---|---|---|---|
| (a) LSTM | 81.0 | – | – | 36.0 | – | – | 63.3 | – | – |
| (b) 3D-ConvNet | 51.6 | – | – | 24.3 | – | – | 56.1 | – | – |
| (c) Two-Stream | 83.6 | 85.6 | 91.2 | 43.2 | 56.3 | 58.3 | 62.2 | 52.4 | 65.6 |
| (d) 3D-Fused | 83.2 | 85.8 | 89.3 | 49.2 | 55.5 | 56.8 | – | – | 67.2 |
| (e) **Two-Stream I3D** | **84.5** | **90.6** | **93.4** | **49.8** | **61.9** | **66.4** | **71.1** | **63.4** | **74.2** |

{{< concept-alert type="advertencia" >}}
**Esta tabla no muestra el resultado famoso de I3D.** El número por el que se cita el paper es **98.0% en UCF-101**, y aquí aparece 93.4%. No hay error: esta es la Tabla 2 del paper, que entrena y evalúa **dentro de cada dataset** (con pre-entrenamiento ImageNet, sin Kinetics). El 98.0% viene de la Tabla 4, después de **pre-entrenar en Kinetics** y hacer fine-tuning. La diferencia entre 93.4 y 98.0 **es el valor del pre-entrenamiento en video a gran escala**, y es literalmente el argumento del título del paper. Vale tenerlo claro porque es fácil citar la tabla equivocada.
{{< /concept-alert >}}

El otro dato que salta es la fila **(b) 3D-ConvNet: 51.6% en UCF-101**, la peor de todas por un margen enorme, y muy por debajo de su propio 56.1% en Kinetics —el único caso donde un modelo rinde *mejor* en el dataset difícil que en el fácil. La explicación es exactamente la desventaja del slide de C3D: sin pre-entrenamiento ImageNet, la red 3D necesita muchos datos, y UCF-101 (~13.000 clips) no se los da. Con Kinetics (~240.000) empieza a funcionar. La familia 3D no era mala; estaba desnutrida.

---

## 12. Después de la clase: qué pasó con esas desventajas

La clase cierra en I3D (2017), pero sus tres desventajas —muchos parámetros, costoso, inferencia lenta— fueron la agenda de investigación de los dos años siguientes. Conviene saber cómo terminó, porque el linaje llega hasta las arquitecturas de hoy:

| Desventaja de la clase | Respuesta | Cómo |
|---|---|---|
| I3D tiene muchos parámetros y es costoso | [S3D (Xie et al., 2018)](/papers/s3d-xie-2018) | Audita qué capas necesitan ser 3D: descubre que las **bajas no** (diseño *top-heavy*) y que separar $k\times k\times k$ en $1\times k\times k$ + $k\times1\times1$ mejora precisión **y** velocidad |
| I3D tiene muchos parámetros y es costoso | [R(2+1)D (Tran et al., 2018)](/papers/r2plus1d-tran-2018) | La misma factorización con ResNet, del autor de C3D revisando su propio trabajo; muestra que el beneficio es de **optimización**, no solo de regularización |
| Two-Stream necesita calcular flujo óptico | [SlowFast (Feichtenhofer et al., 2019)](/papers/slowfast-feichtenhofer-2019) | Reemplaza la separación *apariencia/movimiento* por *framerate lento/rápido*: dos vías del mismo tipo, sin flujo óptico precomputado |

Hay una ironía final que vale registrar. El argumento central de I3D es que heredar ImageNet era indispensable para las redes 3D. Dos años después, SlowFast entrena **desde cero** y alcanza el estado del arte. Lo que cambió no fue la arquitectura sino la disponibilidad de datos: con Kinetics maduro, el video ya tenía su propio ImageNet y no necesitaba prestado el de las imágenes. El truco del inflado sigue siendo útil —es la manera estándar de arrancar un modelo de video cuando los datos son escasos, que es la situación de casi cualquier proyecto real— pero dejó de ser la única puerta.

Sobre la implementación concreta del inflado y las factorizaciones, ver la [profundización](profundizacion) y la [práctica desde 0](practica). El [Laboratorio 38](/laboratorios/lab-38) aplica I3D a clasificación de acciones.
