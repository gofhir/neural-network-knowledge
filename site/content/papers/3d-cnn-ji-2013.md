---
title: "3D CNN: la primera convolución espacio-temporal (2010/2013)"
weight: 420
math: true
---

{{< paper-card
    title="3D Convolutional Neural Networks for Human Action Recognition"
    authors="Shuiwang Ji, Wei Xu, Ming Yang, Kai Yu (NEC Laboratories America)"
    year="2010"
    venue="ICML 2010 / IEEE TPAMI 2013"
    pdf="/papers/3d-cnn-ji-2013.pdf" >}}
Este es el **ancestro olvidado** de [C3D](/papers/c3d-tran-2015) y de [I3D](/papers/i3d-carreira-2017): la idea de convolucionar en espacio y tiempo a la vez, formalizada completa **cinco y siete años antes**, en un paper de ICML que la comunidad de visión casi no leyó. Las CNN de 2010 estaban "limitadas a manejar entradas 2D" y, aplicadas cuadro por cuadro, no podían "considerar la información de movimiento codificada en múltiples fotogramas contiguos". Ji et al. extienden la convolución a un kernel $P\times Q\times R$ que se desliza sobre el cubo formado al apilar fotogramas contiguos, para que los features de movimiento se **extraigan de los datos crudos** en vez de diseñarse a mano. Sobre esa operación construyen una red de 7 capas —1 *hardwired*, 3 convolucionales, 2 de submuestreo y 1 de fusión total— que comprime **7 fotogramas de 60×40** en un vector de **128 dimensiones**, con un clasificador lineal encima y **295 458 parámetros**. En video de vigilancia real (TRECVID 2008, aeropuerto de London Gatwick) supera a la 2D CNN equivalente con precisión promedio de **0.5572 contra 0.4805** a FPR 1%, y la ventaja **crece cuando hay pocas muestras positivas**. En KTH alcanza **90.2%**: tercero de seis métodos y dentro del ruido del protocolo. Es un paper de laboratorio industrial —el grupo de Kai Yu en NEC Labs, que ganó ImageNet 2010 y fundaría después el Institute of Deep Learning de Baidu— y la vanguardia de la única facción que en 2010 creía que aprender los features era mejor que diseñarlos. Lo que le faltaba no era la idea: faltaban datos, profundidad y pre-entrenamiento.
{{< /paper-card >}}

---

## Contexto: reconocimiento de acciones antes de AlexNet

Hay que fijar la fecha con brutalidad: ICML 2010 es **dos años antes de [AlexNet](/papers/alexnet-krizhevsky-2012)**. El consenso en visión era que las [redes convolucionales](/fundamentos/redes-convolucionales) eran una curiosidad, y el estado del arte en [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) era lo que los autores llaman "el paradigma convencional de reconocimiento de patrones", en tres etapas separadas: detección de **puntos de interés espacio-temporales** —los **STIP** de Laptev, que extienden Harris al volumen espacio-temporal, o los **cuboides de Dollár et al. (2005)**, que reemplazan Harris3D por filtros separables porque Harris3D detectaba demasiado pocos puntos en video real—; descripción con HOG/HOF/SIFT y cuantización contra un diccionario de *k*-means, la clásica **bag of visual words** con **spatial pyramid matching**; y un **SVM** encima. El paper canónico de la receta es Schüldt et al. (2004), literalmente *"Recognizing human actions: a local SVM approach"*, que además introdujo KTH. La familia competidora era **HMAX** (Serre et al., 2005), jerarquía de Gabor y *max pooling* extendida a video por Jhuang et al. (2007), y contra la cual el paper marca la diferencia filosófica: las CNN son "sistemas completamente entrenables", mientras "todos los módulos de HMAX consisten en conexiones y parámetros hechos a mano".

Su argumento contra los features a mano es específico de la tarea: "rara vez se sabe cuáles features son importantes", y menos aún en acciones humanas, donde "clases de acción distintas pueden verse dramáticamente diferentes en apariencias y patrones de movimiento". ¿Y por qué las CNN no habían llegado al video? Por tres razones de ecosistema: estaban confinadas a 2D (el único antecedente que encuentran, Ning et al. 2005, opera cuadro por cuadro); **no había datos**, mientras el modelo "requiere un gran número de muestras etiquetadas"; y **no había pre-entrenamiento** —tanto que el remedio que proponen a futuro es el pre-entrenamiento *no supervisado* al estilo Ranzato et al. (2007), la línea que se estancó.

---

## La convolución 3D

El punto de partida es la convolución 2D. El valor de la unidad en $(x,y)$ del $j$-ésimo mapa de la $i$-ésima capa es

$$v_{ij}^{xy} = \tanh\!\left(b_{ij} + \sum_{m} \sum_{p=0}^{P_i-1} \sum_{q=0}^{Q_i-1} w_{ijm}^{pq}\, v_{(i-1)m}^{(x+p)(y+q)}\right)$$

con $b_{ij}$ el sesgo del mapa, $m$ indexando los mapas de la capa anterior conectados al actual y $P_i$, $Q_i$ la altura y el ancho del kernel. La extensión 3D agrega un tercer eje de sumatoria: el valor en $(x,y,z)$ es

$$v_{ij}^{xyz} = \tanh\!\left(b_{ij} + \sum_{m} \sum_{p=0}^{P_i-1} \sum_{q=0}^{Q_i-1} \sum_{r=0}^{R_i-1} w_{ijm}^{pqr}\, v_{(i-1)m}^{(x+p)(y+q)(z+r)}\right)$$

donde $R_i$ es el tamaño del kernel en la dimensión temporal. Eso es todo: la ecuación que sostiene C3D, I3D y todos sus descendientes está escrita en 2010.

El kernel tiene extensión $R_i > 1$ en el eje $z$ y sus pesos se **comparten** al deslizarlo: "el mismo kernel 3D se aplica a cubos 3D solapados en el video de entrada para extraer features de movimiento". Un peso positivo en $(p_1,q_1,0)$ y uno negativo en $(p_2,q_2,2)$ implementan un detector de "algo que estaba aquí y dos fotogramas después está allá": una derivada espacio-temporal aprendida. En una CNN 2D aplicada cuadro a cuadro el gradiente nunca ve dos instantes simultáneamente, así que **ningún parámetro puede codificar una velocidad**. El precio del *weight sharing* es que "un kernel convolucional 3D solo puede extraer un tipo de features del cubo": cada kernel es **un único detector**, y de ahí que apliquen varios kernels distintos en la misma ubicación sin compartir pesos entre sí.

---

## La arquitectura, capa por capa

Entrada: **7 fotogramas de 60×40** centrados en el fotograma actual.

| Capa | Operación | Kernel | Salida | Parámetros |
|---|---|---|---|---|
| Entrada | — | — | 7 @ 60×40 | 0 |
| **H1** | *hardwired*, 5 canales | fijo, no entrenable | **33** @ 60×40 | 0 |
| **C2** | conv 3D por canal, **2** juegos | 7×7×3 | 23×2 @ 54×34 | 1480 |
| **S3** | submuestreo | 2×2 | 23×2 @ 27×17 | 92 |
| **C4** | conv 3D por canal y juego, **3** kernels | 7×6×3 | 13×6 @ 21×12 | 3810 |
| **S5** | submuestreo | 3×3 | 13×6 @ 7×4 | 156 |
| **C6** | conv **2D**, conexión completa a los 78 mapas de S5 | 7×4 | 128 @ 1×1 | 289 536 |
| Salida | clasificador lineal | — | 3 unidades | 384 |
| **Total** | | | | **295 458** |

Los cinco canales de H1 son `gray` (los valores de gris de los 7 fotogramas), `gradient-x` y `gradient-y` (gradientes de cada fotograma) y `optflow-x` y `optflow-y` ([flujo óptico](/fundamentos/flujo-optico) entre fotogramas adyacentes): $7+7+7+6+6=33$ mapas. Los canales de flujo tienen **6** y no 7 porque el flujo se define entre pares consecutivos, y esa asimetría se arrastra por toda la red —en C2 el eje temporal da $7-3+1=5$ mapas para gris y gradientes pero $6-3+1=4$ para los flujos, total 23 por juego—. El submuestreo de S3 y S5 es el clásico de LeNet, con coeficiente multiplicativo entrenable y sesgo por mapa, no el max-pooling sin parámetros de hoy.

> **No hay mezcla entre canales** hasta el final: C2 y C4 aplican sus kernels "en cada uno de los canales por separado". Que las cinco vías viajen aisladas de H1 a C6 hace de esta red una arquitectura de **cinco streams con fusión tardía** —apariencia, dos de gradiente espacial, dos de movimiento—, antecedente directo del [two-stream](/papers/two-stream-simonyan-2014) de Simonyan y Zisserman, cuatro años antes. Los autores probaron variantes que combinaban los canales más temprano y reportan que esta rinde mejor: la fusión tardía ganó, igual que ganaría en two-stream.

En C6 el tiempo colapsa. El kernel $7\times4$ coincide con el tamaño espacial de S5, la salida cae a $1\times1$ y cada uno de los 128 mapas queda conectado a **todos** los 78 de S5; el tiempo no desaparece por la convolución sino por la **sumatoria sobre $m$**, que fusiona tiempo y canales de un golpe. Todo se inicializa **al azar** y se entrena con retropropagación en línea al estilo LeCun et al. (1998). Y ahí está la observación de escala que define al paper: **C6 concentra el 98.0% de los parámetros** ($289\,536/295\,458$), mientras las tres capas convolucionales juntas suman 5446 pesos, un 1.8% del modelo. En capacidad, esta red es casi enteramente un clasificador lineal alimentado por un extractor minúsculo.

---

## La capa hardwired: features prescritos en lugar de aprendidos

La justificación cabe en una oración desarmante: "esta capa *hardwired* se usa para codificar nuestro conocimiento previo sobre features, y este esquema usualmente conduce a mejor desempeño en comparación con la inicialización aleatoria".

Es la confesión central del trabajo. Los autores acaban de argumentar que las CNN "automatizan la construcción de features", y sin embargo **la primera capa de su red son features hechos a mano**: gradientes tipo Sobel y flujo óptico precomputado, cero parámetros entrenables. La red aprende recién desde la segunda capa. ¿Por qué no aprender la primera? Porque no tenían con qué. Un `conv1` aprendido converge a detectores de borde y filtros tipo Gabor —exactamente lo que muestran las visualizaciones de AlexNet dos años después—, pero para que eso emerja hacen falta del orden de un millón de imágenes. Ellos tenían parches de 60×40 de cinco días de video de aeropuerto y una red entrenada con SGD en línea, sin ReLU, sin dropout, sin batch normalization y sin GPU mencionada. La capa *hardwired* es una **inyección manual del prior que la red no podía descubrir sola**.

Aquí está el contraste que hace de este paper material obligatorio para la [Clase 38](/clases/clase-38). Siete años después, Carreira y Zisserman resuelven **el mismo problema** —dar a la primera capa un prior visual que los datos de video no alcanzan a proveer— con la respuesta opuesta: en lugar de escribir los filtros a mano, los **heredan** de una red pre-entrenada en ImageNet, inflando cada filtro $N\times N$ a $N\times N\times N$ y replicando los pesos $N$ veces divididos por $N$ (ver [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones)). I3D arranca con un `conv1` que ya sabe de bordes, texturas y colores aprendidos de 1.2 millones de imágenes, y luego lo **sigue afinando**: sus filtros $7\times7\times7$ terminan desarrollando "rica estructura temporal" y dejan de parecerse a los de Inception-v1.

> Ji et al. e I3D resuelven el mismo problema: **Ji con conocimiento humano, I3D con datos.** El prior de Ji es fijo, no se refina y solo contiene lo que sus autores supieron anticipar en 2010; el de I3D es aprendido, se afina y contiene lo que contiene ImageNet.

La versión extendida de TPAMI 2013 agrega una segunda muleta del mismo tipo. El campo receptivo temporal es de 7 fotogramas muestreados cada 2: 13 fotogramas a 25 fps, unos **0.5 segundos**, cuando `ObjectPut` tiene fase de aproximación, de depósito y de retirada. La solución de TPAMI es computar descriptores clásicos de movimiento sobre una ventana mucho más larga y usarlos como **salidas auxiliares** que la red debe predecir además de la clase: un regularizador multi-tarea que fuerza al vector de 128 dimensiones a ser informativo sobre un contexto que la red nunca vio. Es un sustituto artesanal del pre-entrenamiento a gran escala, y desaparece en cuanto aparecen los datos: ni C3D con Sports-1M ni I3D con Kinetics necesitan nada parecido.

---

## Resultados: TRECVID y KTH

**TRECVID 2008 (London Gatwick).** 49 horas de video de aeropuerto, 5 cámaras, 720×576 a 25 fps, tres clases —`CellToEar`, `ObjectPut`, `Pointing`— clasificadas uno-contra-el-resto sobre 235 561 muestras, de las cuales 149 081 (63.3%) son negativas. Un detector de personas y un tracker derivan un *bounding box* por actor, y el cubo se arma extrayendo ese box en la misma posición de 7 fotogramas con **paso 2** (−6, −4, −2, 0, 2, 4, 6), escalados a 60×40. Las líneas base son la **2D CNN** equivalente, para aislar el aporte de la tercera dimensión, y dos variantes de bag-of-words con SIFT densos sobre los **mismos cubos** (codebook de 512 palabras, 8192 dimensiones, SVM lineal). Protocolo: 5 folds, cada uno un día distinto.

| Método | FPR | Precisión | Recall | AUC (×10³) |
|---|---|---|---|---|
| **3D CNN** | 0.1% | **0.7137** | **0.0230** | **0.0129** |
| 2D CNN | 0.1% | 0.6085 | 0.0155 | 0.0092 |
| SPMcube-gray | 0.1% | 0.6056 | 0.0157 | 0.0087 |
| SPMcube-MEHI | 0.1% | 0.6269 | 0.0157 | 0.0081 |
| **3D CNN** | 1% | **0.5572** | **0.1132** | **0.6752** |
| 2D CNN | 1% | 0.4805 | 0.0833 | 0.4844 |
| SPMcube-gray | 1% | 0.4817 | 0.0836 | 0.4855 |
| SPMcube-MEHI | 1% | 0.5020 | 0.0901 | 0.5099 |

El 3D CNN gana en las seis casillas. A FPR 1% la precisión promedio sube de 0.4805 a 0.5572 (**+16.0%** relativo), el recall de 0.0833 a 0.1132 (**+35.9%**) y el AUC parcial de 0.4844 a 0.6752 (**+39.4%**). Los valores absolutos parecen bajísimos porque el **recall se mide a FPR fija y muy baja**: con 149 081 negativos, un FPR de 0.1% permite apenas unos 149 falsos positivos, así que un recall de 0.113 es la operación deliberada en el extremo de alta precisión que exige la vigilancia. Y esos "AUC" son **parciales**, restringidos a esa región de FPR, no el área bajo la ROC completa.

El desglose por clase es lo más interesante. En `CellToEar` y `ObjectPut` el 3D CNN gana "significativamente en todos los casos": a FPR 0.1%, `CellToEar` pasa de 0.4848 (mejor línea base) a **0.6433**, un +32.7%, y de 0.3842 del 2D CNN a 0.6433, un +67.4%. En `Pointing`, en cambio, "el modelo 3D CNN logra un desempeño levemente peor que los otros tres métodos": 0.8230 contra 0.8547. La explicación del propio paper es la frase que hay que retener: "podemos concluir que el modelo 3D CNN es más efectivo cuando el número de muestras positivas es pequeño". `Pointing` tiene 55 127 positivos, 3.94 veces los de `CellToEar`; con esa evidencia, el bag-of-words de 8192 dimensiones alcanza para ganar por un pelo.

**KTH.** 6 clases, 25 sujetos. Para comparar contra HMAX siguen el setup de Jhuang et al. (2007) —cubo de 9 fotogramas y extracción de foreground— pero bajan la resolución a **80×60** frente a los 160×120 de Jhuang. Entrenan con 16 sujetos al azar, testean con los 9 restantes y promedian 5 trials.

| Método | Boxing | Handclapping | Handwaving | Jogging | Running | Walking | **Promedio** |
|---|---|---|---|---|---|---|---|
| **3D CNN** | 90 | 94 | **97** | 84 | 79 | **97** | **90.2** |
| Schüldt et al. (2004) | 97.9 | 59.7 | 73.6 | 60.4 | 54.9 | 83.8 | 71.7 |
| Dollár et al. (2005) | 93 | 77 | 85 | 57 | 85 | 90 | 81.2 |
| Niebles et al. (2008) | **98** | 86 | 93 | 53 | 88 | 82 | 83.3 |
| Jhuang et al. (2007) | 92 | **98** | 92 | **85** | 87 | 96 | 91.7 |
| Schindler y Van Gool (2008) | – | – | – | – | – | – | **92.7** |

**90.2%**: tercero de seis, con +18.5 puntos sobre Schüldt y +9.0 sobre Dollár, pero 1.5 bajo HMAX y 2.5 bajo Schindler. El paper usa la palabra exacta —desempeño **"competitivo"**, no superior— y su defensa es de resolución: "nótese que el modelo HMAX usa features hechos a mano computados a partir de imágenes crudas con resolución 4 veces mayor", $160\times120=19\,200$ píxeles contra $80\times60=4800$. Cerrar 1.5 puntos con un cuarto de los píxeles y con features aprendidos es el argumento real del experimento. Es el mejor de la tabla en `Handwaving` y `Walking` (97) y el peor en `Boxing` (90) y `Running` (79): `Jogging` contra `Running` es la confusión clásica de KTH, porque difieren en velocidad y no en apariencia.

> **La advertencia sobre el protocolo, que conviene tomar más en serio que el propio paper.** La tabla mezcla seis grupos que no usaron el mismo protocolo: Schüldt et al. dividió los 25 sujetos en train/validación/test; Dollár et al. y Niebles et al. reportan **leave-one-subject-out**, que entrena con 24 sujetos en vez de 16 y ve un 50% más de datos; Ji et al. y Jhuang et al. usan 16/9. Se suman decisiones no estandarizadas: evaluar por secuencia o por clip, tratar los cuatro escenarios de grabación juntos o por separado, y usar o no extracción de foreground (Ji et al. sí la usan). Los tres guiones de la fila de Schindler dejan el problema a la vista. La lectura honesta es que **90.2, 91.7 y 92.7 son indistinguibles**. Es el mismo fenómeno que I3D denunciaría siete años después con UCF-101 y HMDB-51: cuando el dataset es diminuto, casi cualquier arquitectura rinde parecido y el benchmark deja de discriminar.

---

## Limitaciones

- **Escala minúscula en todos los ejes.** 7 fotogramas de 60×40, 3 capas convolucionales, 295 458 parámetros. C3D usa 16 fotogramas de 112×112 con 8 capas y del orden de 79M de parámetros, unas **267 veces más**; I3D usa 64 fotogramas de 224×224, **191 veces** el volumen de entrada de Ji, con 25M. Y los pesos están en el lugar equivocado: el 98.0% vive en C6, mientras el extractor espacio-temporal (C2 + C4) tiene 5290.
- **La capa hardwired contradice parcialmente la tesis.** Gradientes y flujo óptico prescritos, cero parámetros entrenables, en un paper cuyo argumento es que los features deben aprenderse.
- **Dependencia de flujo óptico precomputado.** Cómputo externo a la red, un hiperparámetro más y la asimetría de 6 mapas contra 7 que se propaga por toda la arquitectura. I3D **tampoco** la eliminó: su mejor configuración sigue necesitando un stream TV-L1 externo.
- **Ausencia total de pre-entrenamiento.** "Todos los parámetros se inicializan al azar." En las conclusiones apuestan al pre-entrenamiento *no supervisado*; la historia respondió con pre-entrenamiento **supervisado** a gran escala.
- **Ventana temporal de medio segundo.** Las acciones con estructura temporal más larga quedan fuera de alcance. El paso 2, además, ensancha el span pero **submuestrea el movimiento**, degradando el flujo óptico entre fotogramas separados por 80 ms.
- **Datasets pequeños y poco discriminativos.** KTH es de juguete; TRECVID Gatwick es realista pero de **una sola escena** y tres clases, con desbalance 1:15.8: difícil separar "aprendió acciones" de "aprendió el aeropuerto de Gatwick".
- **Maquinaria de 2010.** `tanh` en lugar de ReLU, submuestreo con coeficiente entrenable en lugar de max-pooling, SGD en línea con una muestra por actualización, sin dropout, sin batch normalization, sin GPU.

---

## Por qué importa hoy

La [Clase 38](/clases/clase-38) recorre la escalera CNN2D + pooling temporal → CNN2D + RNN → two-stream → C3D → I3D, y le atribuye a C3D tres desventajas: **no puede aprovechar el pre-entrenamiento de ImageNet**, **tiene muchos más parámetros** y **es más difícil de entrenar**. Este paper prueba que esas tres no son un accidente de la implementación de C3D: son **propiedades estructurales de la familia 3D**, presentes ya en su primer ejemplar, cinco años antes.

El origen del problema es puramente dimensional. Un kernel 2D de $7\times7$ vive en $\mathbb{R}^{49}$; su versión 3D de $7\times7\times3$ vive en $\mathbb{R}^{147}$. No existe ninguna red pre-entrenada cuyos pesos habiten ese espacio, porque todas se entrenaron sobre imágenes: cualquier arquitectura 3D nace **fuera del alcance del zoo de pesos disponibles**. Ji et al. lo compensaron a mano —congelando la primera capa e inyectando contexto de largo alcance como salidas auxiliares—; C3D lo compensó con más datos; I3D lo resolvió cambiando la pregunta: en lugar de buscar pesos 3D que no existen, **los construyó** a partir de pesos 2D con la regla $w^{3D}(t) = \frac{1}{N}w^{2D}$, que garantiza que la red inflada responda a un video constante exactamente como la red 2D respondía a la imagen. Eso es lo único que faltaba, y son tres líneas de código. "Muchos parámetros" y "difícil de entrenar" son la misma cosa: cada kernel 3D cuesta $R$ veces su equivalente 2D, y ese factor hay que amortizarlo con datos. Ji et al. lo pagaron manteniendo la red absurdamente pequeña —3 capas y 5290 pesos en el extractor de movimiento—, que es el techo que impone la escasez.

**¿Por qué quedó enterrado?** Nadie podía ver la ventaja sin los datos: en KTH quedaba tercero y dentro del ruido del protocolo, y en TRECVID ganaba sobre un benchmark que no circulaba. Publicó en **ICML, no en CVPR o ICCV**, así que la comunidad de visión no lo leyó como propio, y llegó dos años antes de AlexNet. Y en 2014 apareció un contraejemplo devastador: cuando por fin hubo un dataset grande, [Karpathy et al.](/papers/large-scale-video-karpathy-2014) reportaron "mejoras significativas frente a líneas base fuertes basadas en features (55.3% a 63.9%), pero solo una mejora sorprendentemente modesta frente a los modelos de un solo fotograma (59.3% a 60.9%)", lo que retrasó la fe en la convolución 3D otro año, hasta que C3D mostró cómo hacerla funcionar. Tran et al. (R(2+1)D, 2018) cierran el arco: las 3D CNN "fueron propuestas, se puede argumentar, primero por Baccouche et al. y por **Ji et al.**", y "más recientemente, se mostró que conducen a resultados fuertes **cuando se entrenan en datasets de gran escala**".

La mejor evidencia del valor del paper es su descendencia: cada elemento que C3D e I3D **quitaron** —la capa *hardwired*, la separación de canales, los features auxiliares, la ventana de medio segundo— era una compensación por la falta de datos, no un error de diseño. Y la lección es directa para cualquier proyecto que aplique modelos espacio-temporales a video propio y limitado, sea clínico, industrial o de vigilancia. Ji et al. hicieron exactamente lo que uno se ve tentado a hacer con una buena idea arquitectónica y pocos datos: encogieron el modelo hasta que cupiera en el dataset, congelaron la primera capa con features de dominio e inyectaron descriptores clásicos como regularizador. Las tres decisiones son razonables, están bien ejecutadas y produjeron un modelo que quedó tercero en un benchmark de juguete. La alternativa que la historia validó fue la opuesta: no encoger el modelo, sino **importar el prior desde otro dominio**. **La ingeniería de features es lo que uno hace cuando no tiene de dónde transferir**, y hoy casi siempre lo hay.

---

## Notas y enlaces

- **Dos versiones.** El trabajo existe como paper de **ICML 2010** (8 páginas) y como versión extendida en **IEEE TPAMI 35(1):221–231, enero 2013**. El PDF enlazado arriba es la versión ICML y todas las cifras de esta página provienen de ese texto; las contribuciones exclusivas de TPAMI se describen a nivel de mecanismo, sin citar números.
- **Genealogía:** este paper → [C3D](/papers/c3d-tran-2015) (2015: quita la capa *hardwired* y la separación de canales, escala a Sports-1M) → [I3D](/papers/i3d-carreira-2017) (2017: resuelve el pre-entrenamiento con el [inflado de convoluciones](/fundamentos/inflado-de-convoluciones)). El recorrido está en [Clase 38](/clases/clase-38) y su [teoría](/clases/clase-38/teoria); la [profundización](/clases/clase-38/profundizacion) desarrolla la matemática.
- **Two-stream viene desde adentro de la convolución 3D.** Los cinco canales aislados de H1 a C6 son un multi-stream con fusión tardía cuatro años antes de [Simonyan y Zisserman](/papers/two-stream-simonyan-2014).
- **El flujo óptico nunca se fue.** Está en Ji et al. 2010 (`optflow-x/y`), en two-stream 2014 y sigue en I3D 2017, cuya mejor configuración necesita un stream TV-L1 externo porque una 3D ConvNet es "puramente feedforward" mientras los algoritmos de [flujo óptico](/fundamentos/flujo-optico) hacen una optimización iterativa que la red no reproduce.
- **Contexto previo:** la [Clase 36](/clases/clase-36) introduce las dos grandes vías para dar sentido temporal a una red de video; los conceptos base están en [redes convolucionales](/fundamentos/redes-convolucionales) y [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).
