# Long-term Recurrent Convolutional Networks (LRCN) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Long-term Recurrent Convolutional Networks for Visual Recognition and Description*.
- **Autores:** Jeff Donahue, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan, Sergio Guadarrama, Kate Saenko, Trevor Darrell. Núcleo del grupo en el **Departamento de Ingeniería Eléctrica y Ciencias de la Computación de UC Berkeley** (con afiliaciones adicionales en el International Computer Science Institute, UT Austin y UMass Lowell).
- **Venue:** *CVPR 2015*; la versión extendida que analizamos aquí es el manuscrito de revista (arXiv:1411.4389v4, 31 mayo 2016). Trevor Darrell dirigía entonces el *Berkeley Vision and Learning Center* (BVLC), hogar de Caffe, framework en el que se liberó la implementación.
- **Sigla:** **LRCN**, *Long-term Recurrent Convolutional Network*.

El paper propone una **clase de arquitecturas recurrentes-convolucionales**, entrenables de extremo a extremo, para tareas visuales que involucran secuencias. La idea central es sencilla y potente: una **CNN extrae un vector de características por cada frame** (o por cada imagen), y esos vectores se entregan como **secuencia temporal a una pila de LSTM** que modela la dinámica en el tiempo. Los autores llaman a estos modelos "doblemente profundos" (*doubly deep*) porque aprenden representaciones composicionales tanto en el **espacio** (las capas de la CNN) como en el **tiempo** (el desenrollado del LSTM).

La contribución clave no es solo la arquitectura, sino su **generalidad**: un mismo esquema CNN+LSTM sirve para tres tareas visuales de naturaleza distinta —**reconocimiento de actividad** (video → etiqueta), **generación de descripciones de imágenes** o *image captioning* (imagen → oración) y **descripción de video** (video → oración)—, simplemente reconfigurando qué extremo del modelo es secuencial. En contraste con métodos previos que asumen una **representación visual fija** o hacen un **promediado temporal simple** sobre ventanas de tamaño fijo, LRCN aprende de forma diferenciable a mapear entradas de longitud variable (videos) a salidas de longitud variable (texto), optimizable con backpropagation.

Para la **Clase 36 (Introduction to Video Analysis)** este paper es directamente el enfoque que la clase propone como solución para inyectar noción temporal: el famoso **"2D CNN + RNN"**. La slide final resume su virtud y su defecto en una línea —"el RNN es apto para procesar secuencias, funciona mejor que la 2D CNN sola, pero el RNN no se puede paralelizar"— y LRCN es exactamente esa arquitectura, con la evidencia experimental que respalda ambas afirmaciones.

## 2. Contexto: la 2D CNN por frame ignora el orden temporal

El reconocimiento y la descripción de imágenes y videos son un desafío fundamental de la visión por computador. Hacia 2014–2015 las **CNN habían dominado el reconocimiento de imágenes estáticas** (AlexNet, VGGNet, GoogLeNet), pero extenderlas a video planteaba un problema estructural: **una imagen es estática, un video es una secuencia**. Un video debería poder procesarse como una entrada de **longitud variable** y, para tareas de descripción, producir también **salidas de longitud variable** (oraciones completas), algo que va más allá de la clásica predicción "uno-contra-todos" de un clasificador.

La investigación en CNN para video había explorado dos extremos del espectro de representación temporal, y el paper los critica explícitamente:

1. **Filtros 3D espacio-temporales** aprendidos sobre datos crudos de la secuencia (las 3D CNN, referencias [1], [2] del paper). Aprenden un ponderado temporal completamente general, pero costoso y difícil de preentrenar sobre datos de imágenes.
2. **Representaciones frame-a-frame agregadas por ventanas fijas** —incorporando *optical flow* instantáneo o modelos de trayectoria— sobre ventanas o segmentos de video de tamaño fijo ([3], [4]). Estos aplican un **pooling temporal simple** (por ejemplo, promediar los *scores* softmax de varios frames).

El problema de fondo con el **pooling / promediado temporal** es que **destruye el orden**. Si un modelo clasifica cada frame de forma independiente con una 2D CNN y luego promedia las probabilidades, el resultado es idéntico sin importar en qué **secuencia** ocurrieron los frames. Como observa la sección de trabajo relacionado, los modelos "superficiales" que codifican características como *bags of words* o *Fisher vectors* **pierden las relaciones temporales**: pueden rastrear cómo cambian características de bajo nivel a lo largo del tiempo, pero no seguir características de nivel más alto ni el orden de los eventos. Levantarse de una silla y sentarse en ella producen frames casi idénticos; solo el **orden** los distingue. Una 2D CNN por frame, seguida de promedio, es ciega a esa diferencia.

La alternativa que propone el paper, inspirada en la misma motivación que dio origen a las CNN profundas, es construir modelos que sean **profundos también en la dimensión temporal**: que tengan **recurrencia temporal de variables latentes**. Las **RNN** son "profundas en el tiempo" —explícitamente al desenrollarse— y forman representaciones composicionales implícitas en el dominio temporal. Su limitación clásica, el **desvanecimiento del gradiente** (que hace difícil propagar la señal de error a través de intervalos temporales largos), se resuelve con **LSTM** (Hochreiter & Schmidhuber, 1997): unidades recurrentes con un estado de celda que se puede mantener sin modificación, actualizar o resetear mediante compuertas aprendidas, habilitando el **aprendizaje de dependencias de largo alcance** (*long-range* / *long-term*).

## 3. Contribución central: CNN+LSTM unificado para tareas visuales secuenciales

La contribución es un **modelo LRCN unificado** que combina un **extractor jerárquico de características visuales** (una CNN) con un **modelo de secuencias** (LSTM) capaz de reconocer y sintetizar dinámica temporal, entrenable de extremo a extremo. Sus rasgos distintivos:

- **Doblemente profundo (espacio + tiempo).** No solo apila capas convolucionales sobre el espacio de píxeles, sino también pasos temporales de recurrencia sobre la secuencia. Para $T$ grande, las últimas predicciones de la RNN se computan mediante una función no lineal muy "profunda" de $T$ capas.
- **Entradas y salidas de longitud variable.** A diferencia de los modelos de ventana fija, LRCN mapea directamente secuencias de longitud arbitraria a salidas de longitud arbitraria, sin necesidad de fijar de antemano cuántos frames o cuántas palabras.
- **Entrenamiento conjunto extremo a extremo.** Los parámetros visuales $V$ (la CNN) y los secuenciales $W$ (la LSTM) se optimizan **juntos** por backpropagation, de modo que el extractor visual aprende a resaltar justo los aspectos relevantes para el problema secuencial. Esto contrasta con enfoques previos donde el modelo visual y el de secuencia se definían u optimizaban por separado.
- **Generalidad.** Un mismo esquema resuelve tres familias de tareas secuenciales según cuál extremo sea variable en el tiempo.

Frente al trabajo previo más cercano en reconocimiento de actividad ([2], [52], que también modelaban dependencias temporales con redes recurrentes), LRCN se diferencia en dos puntos que el propio paper subraya: (1) **integra 2D CNN preentrenables** sobre grandes conjuntos de imágenes, y (2) **combina CNN y LSTM en un único modelo** para permitir el *fine-tuning* de extremo a extremo.

## 4. Método

### 4.1. El núcleo: una CNN por frame, un LSTM sobre la secuencia

Cada entrada visual $x_t$ —una imagen aislada, o un frame de un video— pasa por una transformación de características $\phi_V(\cdot)$ con parámetros $V$, típicamente una CNN, que produce una **representación vectorial de longitud fija** $\phi_V(x_t)$. Las salidas de $\phi_V$ se entregan a un módulo de aprendizaje secuencial recurrente.

En su forma más general, el modelo recurrente tiene parámetros $W$ y mapea una entrada $x_t$ y el estado oculto del paso anterior $h_{t-1}$ a una salida $z_t$ y un estado actualizado $h_t$. Un LSTM básico se rige por:

$$
i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)
$$
$$
f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)
$$
$$
o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)
$$
$$
g_t = \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)
$$

donde $\odot$ es el producto elemento a elemento, $i_t$ es la compuerta de entrada, $f_t$ la de olvido, $o_t$ la de salida, $g_t$ la modulación de entrada y $c_t$ la **celda de memoria**. Como $i_t$ y $f_t$ son sigmoidales (valores en $[0,1]$), funcionan como "perillas" que el LSTM aprende para **olvidar selectivamente** su memoria previa o **considerar** su entrada actual; esto le permite aprender dinámica temporal compleja y de **largo plazo**. Se puede agregar profundidad **apilando** LSTM, usando el estado oculto de la capa $\ell-1$ como entrada de la capa $\ell$.

Para predecir una distribución $P(y_t)$ sobre un conjunto finito de resultados $\mathcal{C}$, la salida $z_t$ del modelo secuencial pasa por una capa lineal $\hat{y}_t = W_z z_t + b_z$ y luego por un **softmax**. El entrenamiento minimiza la **log-verosimilitud negativa** esperada de las salidas verdaderas:

$$
\mathcal{L}(V, W, \mathcal{D}) = -\frac{1}{|\mathcal{D}|} \sum_{(x_t, y_t)_{t=1}^{T} \in \mathcal{D}} \sum_{t=1}^{T} \log P(y_t \mid x_{1:t}, y_{1:t-1}, V, W)
$$

Un detalle de diseño crucial —y que anticipa la tensión de la Clase 36— es que la transformación visual $\phi_V(\cdot)$ es **invariante al tiempo e independiente en cada paso**. Los pesos de la CNN están **atados a través del tiempo** (los mismos para todo frame). Esto tiene una ventaja explícita que el paper enfatiza: hace que la **inferencia y el entrenamiento convolucionales, que son caros, sean paralelizables sobre todos los pasos temporales** de la entrada, aprovechando implementaciones rápidas de CNN cuyo rendimiento depende del procesamiento por lotes independiente. Es decir, **la parte CNN sí se paraleliza**. La parte LSTM, en cambio, no (ver Sección 7).

### 4.2. Las tres configuraciones secuenciales

La elegancia de LRCN está en que las tres tareas visuales estudiadas son instancias de tres clases de aprendizaje secuencial, según dónde viva la longitud variable:

1. **Entrada secuencial, salida estática** $\langle x_1, \dots, x_T \rangle \mapsto y$ (Figura 3, izquierda). Es el **reconocimiento de actividad**: un video de longitud arbitraria $T$ como entrada, pero el objetivo es predecir **una sola etiqueta** de un vocabulario fijo (*running*, *jumping*). Aquí se usa una fusión tardía (*late fusion*): se combinan las predicciones por paso $\langle y_1, \dots, y_T \rangle$ en una única predicción $y$ para la secuencia completa.
2. **Entrada estática, salida secuencial** $x \mapsto \langle y_1, \dots, y_T \rangle$ (Figura 3, centro). Es el **image captioning**: una imagen estática de entrada, pero un espacio de salida mucho más rico —oraciones de cualquier longitud—. Se implementa **duplicando la entrada** $x$ en todos los pasos: $\forall t: x_t := x$.
3. **Entrada y salida secuenciales** $\langle x_1, \dots, x_T \rangle \mapsto \langle y_1, \dots, y_{T'} \rangle$ (Figura 3, derecha). Es la **descripción de video**, donde entrada y salida varían en el tiempo y en general $T \neq T'$ (el número de frames no debe restringir el número de palabras). Se resuelve con un esquema **encoder-decoder** (como en traducción automática): un modelo secuencial codifica la entrada en un vector de longitud fija y otro lo decodifica en la salida.

Esta taxonomía es el corazón conceptual del paper: **el mismo motor CNN+LSTM cubre todas las combinaciones de secuencia en la entrada y/o en la salida**.

## 5. Experimentos

### 5.1. Reconocimiento de actividad (UCF101)

La tarea estrella para la Clase 36. LRCN se evalúa sobre **UCF101**, que contiene **más de 12.000 videos** categorizados en **101 clases de acción humana**, dividido en **tres splits** con algo menos de **8.000 videos de entrenamiento** por split.

Configuración: se entrena con **clips de 16 frames** aunque los videos de UCF101 suelen tener del orden de 100 frames (extraídos a 30 FPS); entrenar sobre clips cortos actúa como aumentación de datos análoga a recortar imágenes. En test se extraen clips de 16 frames con *stride* de 8 y se promedia sobre todos los clips del video. Se consideran dos tipos de entrada: **RGB** y **optical flow** (transformado en una "imagen de flujo" escalando los valores $x$ e $y$ al rango $[-128, +128]$, con un tercer canal para la magnitud del flujo). LRCN predice la clase de actividad en **cada paso temporal** y para obtener una etiqueta única se **promedian las probabilidades softmax** a lo largo de todos los frames.

La base CNN es un híbrido del modelo de referencia **CaffeNet** (variante de AlexNet) y la red de Zeiler & Fergus, **preentrenada sobre los 1,2 millones de imágenes** de ILSVRC-2012 (ImageNet); ese preentrenamiento le da una inicialización fuerte que acelera el entrenamiento y evita el sobreajuste al conjunto de video, comparativamente pequeño. Clasificando *center crops*, la precisión top-1 de la CNN base es **60,2 % (híbrida)** y **57,4 % (CaffeNet)**.

El **baseline clave** es un modelo *single frame*: los $T$ frames se clasifican individualmente por la CNN y el video se clasifica promediando *scores*, **sin ninguna modelización de la secuencia**. Es exactamente la "2D CNN sola" de la clase. LRCN es esa misma CNN, pero con una LSTM encima procesando la secuencia.

Hiperparámetros explorados: número de unidades ocultas del LSTM (256, 512, 1024) y si se usan las características **fc6** o **fc7** como entrada al LSTM. Para flujo, más unidades ayudan (1024 da +1,7 % sobre 256); para RGB, el número de unidades apenas influye (se usan 256). fc6 supera levemente a fc7. Se requirió *dropout* agresivo (**0,9**) para evitar sobreajuste al entrenar de extremo a extremo. Submuestrear frames perjudica; conviene usar **todos** los frames.

### 5.2. Image captioning (Flickr30k, COCO)

Aquí la CNN se invoca **una sola vez** (la entrada es una imagen). En cada paso, la LSTM recibe las características de la imagen y la palabra anterior (codificada como *one-hot* y proyectada por una matriz de *embedding* $W_e \in \mathbb{R}^{d_e \times K}$, que actúa como tabla de búsqueda). El *stack* de LSTM (cada uno con 1000 unidades ocultas) modela la dinámica de la salida —lenguaje natural— y produce una distribución $P(y_t \mid y_{1:t-1}, \phi_V(x))$ sobre el vocabulario, incluyendo el token `<EOS>` que permite generar oraciones de longitud variable.

Se estudiaron tres variantes (Figura 4): una capa (**LRCN1u**), dos capas sin factorizar (**LRCN2u**) y dos capas **factorizadas** (**LRCN2f**, donde las primeras capas quedan "ciegas" a la imagen y solo las superiores fusionan lenguaje y visión). Se evaluó en **recuperación** (imagen↔texto en Flickr30k y COCO) y en **generación** (COCO 2014, con métricas BLEU, METEOR, ROUGE-L y CIDEr-D).

### 5.3. Descripción de video (TACoS multilevel)

Es la instancia de **entrada y salida secuenciales**. Por las limitaciones de los datasets de descripción de video de la época, aquí LRCN no procesa frame a frame de forma incremental, sino que parte de predicciones de actividad, herramienta, objeto y ubicación provistas por un **CRF** sobre el video completo, y usa la LSTM para **generar la oración**. Se evalúa sobre **TACoS multilevel**, con **44.762 pares video/oración** (unos 40.000 para entrenamiento/validación). Se comparan tres arquitecturas: (a) encoder-decoder LSTM con CRF-max, (b) decoder LSTM con CRF-max, y (c) decoder LSTM con **probabilidades del CRF** (que permite a la LSTM aprender incertidumbre en lugar de depender de estimaciones MAP).

## 6. Resultados

**Reconocimiento de actividad (UCF101, promedio de los tres splits, Tabla 1).** LRCN supera **consistente y claramente** al baseline *single frame* con solo la CNN:

| Modelo | RGB | Flow | Promedio ponderado (1/2, 1/2) | Promedio ponderado (1/3, 2/3) |
|---|---|---|---|---|
| Single frame | 67,37 | 74,37 | 75,46 | 80,90 |
| LRCN-fc6 | 68,20 | 77,28 | 78,94 | 82,34 |

LRCN mejora al baseline en **0,83 %** (RGB), **2,91 %** (flujo) y **3,40 %** con el promedio ponderado favoreciendo el flujo. El mensaje es inequívoco: **añadir la LSTM sobre la misma CNN mejora el reconocimiento**, porque la secuencia temporal aporta información que el promedio de frames destruía. En el análisis por clase (Tabla 2), LRCN gana especialmente en acciones donde el **movimiento y su orden** son distintivos (*BoxingPunchingBag* +40,82, *HighJump* +29,73, *JumpRope*), y pierde algo en clases donde basta reconocer objetos estáticos (*Knitting*, *Mixing*); pero las ganancias superan a las pérdidas, elevando la precisión global. También se observa la **complementariedad RGB/flujo**: las actividades identificables por objetos presentes (*Typing* ↔ teclado) las clasifica mejor el modelo RGB, y las definidas por el tipo de movimiento, el modelo de flujo.

En perspectiva de la época, LRCN es **comparable** a otros modelos profundos: la *two-stream* de Simonyan & Zisserman [4] reportaba **87,6 %** en UCF101, mientras la 3D CNN de [3] lograba **65,4 %** (sustancialmente inferior a LRCN). Además, trabajo **contemporáneo/posterior** que combina CNN+LSTM con más capacidad —cuatro LSTM apilados y preentrenamiento sobre el masivo Sports-1M [66]— alcanzó **88,6 %**, confirmando que la receta CNN+LSTM escalaba.

**Image captioning.** En recuperación sobre Flickr30k (Tabla 4), LRCN2f supera consistentemente a los *baselines* fuertes de la época (m-RNN, DeFrag, DeViSE, etc.). Ablaciones importantes: el **LSTM supera claramente a una RNN "vanilla"**, justificando el costo extra de las compuertas incluso para secuencias simples; el *fine-tuning* de la CNN y usar una red más potente (VGGNet en vez de CaffeNet) mejoran sustancialmente. En generación sobre COCO 2014 (Tabla 7), la mejor configuración de LRCN (VGGNet *fine-tuned* + LRCN2f, muestreo con $N=100$, $T=1{,}5$) resultó **4.ª en CIDEr-D (0,934** frente a 0,946 del mejor) y **3.ª en METEOR (0,335** frente a 0,346), competitiva con el estado del arte del *2015 COCO Caption Challenge*.

**Descripción de video (TACoS multilevel, Tabla 9, BLEU-4 en %).** La LSTM supera al enfoque basado en traducción estadística (SMT): las variantes decoder (b) y (c) rinden mejor que el encoder-decoder (a) —probablemente porque no necesitan memorizar la entrada—, y la mejor, (c) con probabilidades del CRF, alcanza **28,8 %**, superando claramente el mejor número previo reportado, **26,9 %** de [48]. Esto demuestra además que LRCN **no está restringido a entradas de redes profundas**: puede integrarse limpiamente con entradas de longitud fija o variable de otros sistemas de visión.

## 7. Limitaciones

- **El RNN es secuencial, no paralelizable (la desventaja de la slide).** Este es el punto que la Clase 36 destaca y que el propio paper hace explícito: *"la inferencia debe ejecutarse secuencialmente"* —computando en orden $h_1 = f_W(x_1, h_0)$, luego $h_2 = f_W(x_2, h_1)$, y así hasta $h_T$—. Mientras la **CNN por frame sí se paraleliza** sobre todos los pasos temporales (por eso se atan sus pesos e independizan por frame), la **recurrencia del LSTM introduce una dependencia estricta paso-a-paso**: $h_t$ no puede calcularse hasta tener $h_{t-1}$. Esa serialización impide aprovechar plenamente el hardware paralelo en la parte temporal y es, precisamente, la motivación que años después llevaría a reemplazar la recurrencia por **atención** (Transformers), que sí procesa toda la secuencia en paralelo. La ventaja (dependencias de largo alcance vía compuertas) y la desventaja (cómputo secuencial) son las dos caras de la misma recurrencia.
- **Costo y datos.** El entrenamiento de extremo a extremo requirió *dropout* muy agresivo (0,9) para no sobreajustar los datasets de video, comparativamente pequeños. La necesidad de preentrenar la CNN sobre ImageNet, y de datasets grandes para explotar la capacidad temporal (Sports-1M en [66]), confirma que el enfoque es **hambriento de datos**.
- **Dinámica temporal limitada en los benchmarks.** Los propios autores reconocen que los datasets de actividad de la época **no tienen dinámica temporal particularmente compleja**; aun así observan mejoras. Esto matiza cuánto de la ganancia proviene realmente del modelado de secuencia versus del mejor uso de la información por frame.
- **La descripción de video no es totalmente end-to-end.** Por limitaciones de datos, esa tarea se apoya en un CRF intermedio en lugar de procesar píxeles de extremo a extremo; trabajo posterior ([69], [70], [71]) cerró esa brecha con modelos *sequence-to-sequence* de video a texto y atención temporal.

## 8. Conexión con la Clase 36 y con el laboratorio

**LRCN es, literalmente, el "2D CNN + RNN" de la Clase 36.** La clase recorre el problema de dotar de noción temporal a modelos de imagen y termina proponiendo esta arquitectura como solución: una 2D CNN que procesa cada frame y una RNN (LSTM) que procesa la secuencia de características. La clase resume su balance en una frase, y LRCN aporta la evidencia de cada término:

- **"Funciona mejor que la 2D CNN sola":** la Tabla 1 lo cuantifica —LRCN supera al baseline *single frame* en hasta 3,40 %—, porque el promedio de frames de una 2D CNN destruye el orden temporal y la LSTM lo recupera.
- **"El RNN es apto para procesar secuencias" (ventaja):** las compuertas del LSTM habilitan dependencias de **largo alcance** (*long-range*), justo lo que un pooling de ventana fija no puede capturar.
- **"El RNN no se puede paralelizar" (desventaja):** el paper afirma que la inferencia recurrente **debe correr secuencialmente**, en contraste con la CNN por frame que sí se paraleliza. Esta serialización es el defecto que la clase señala y el que motivaría el salto hacia arquitecturas basadas en atención.

Respecto del **laboratorio de la clase** (ResNet + componente temporal): LRCN es el molde conceptual. Donde el paper usa CaffeNet/VGGNet como extractor $\phi_V$, el lab usa una **ResNet** preentrenada para obtener un vector de características por frame; y donde el paper apila LSTM para modelar la secuencia, el lab añade su propio bloque temporal sobre esa secuencia de *embeddings*. La lección transferible es doble: (1) **desacoplar** un extractor visual fuerte y preentrenable de un modelo temporal ligero encima; y (2) **atar los pesos de la CNN en el tiempo** para paralelizar la parte cara (la convolución) y reservar la serialización solo para el módulo recurrente.

**Enlaces internos:**

- Clase: [/clases/clase-36](/clases/clase-36) — Introduction to Video Analysis (2D CNN + RNN, arquitecturas temporales).
- Fundamento transversal: [/fundamentos/redes-recurrentes](/fundamentos/redes-recurrentes) — RNN, LSTM, desvanecimiento del gradiente, cómputo secuencial.

## 9. Nota final: relevancia para video clínico

En el dominio clínico, la fortaleza de LRCN —modelar **secuencias temporales largas** donde el **orden** de los eventos es diagnóstico— es exactamente lo que se necesita para el video médico. Una **cirugía** se descompone en **fases ordenadas** (por ejemplo, disección, corte, sutura) cuya identificación depende no del frame aislado sino de la secuencia que lo precede; una **actividad de rehabilitación** progresa a lo largo de repeticiones cuya calidad y evolución solo se aprecian en el tiempo. Un esquema CNN(o ResNet)+LSTM permite extraer características por frame con una red visual preentrenada y luego dejar que la recurrencia con memoria de largo plazo capture la **transición entre fases** o la **progresión del gesto**, exactamente la clase de dependencia temporal que un promedio de frames borraría. La contrapartida práctica es la que la Clase 36 subraya: el módulo recurrente es **secuencial y no paralelizable**, un costo a tener en cuenta para inferencia sobre videos quirúrgicos largos, y una razón por la que los sistemas clínicos modernos a menudo migran hacia alternativas basadas en atención cuando la latencia importa.
