# Temporal Segment Networks: Towards Good Practices for Deep Action Recognition (TSN) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Temporal Segment Networks: Towards Good Practices for Deep Action Recognition*.
- **Autores:** Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang y Luc Van Gool. Afiliaciones: Computer Vision Lab de ETH Zurich (Suiza), Department of Information Engineering de la Chinese University of Hong Kong y Shenzhen Institutes of Advanced Technology (CAS, China).
- **Venue:** *European Conference on Computer Vision (ECCV 2016)*.
- **Año:** 2016. **Preprint:** arXiv:1608.00859v1 (2 ago 2016). Código y modelos: [github.com/yjxiong/temporal-segment-networks](https://github.com/yjxiong/temporal-segment-networks).
- **Linaje:** construye directamente sobre las *two-stream ConvNets* de Simonyan y Zisserman (NIPS 2014) y sobre la arquitectura BN-Inception (Ioffe y Szegedy, ICML 2015). Es uno de los trabajos que consolidó el reconocimiento de acciones basado en aprendizaje profundo como estado del arte frente a las descriptores hechos a mano.

El paper aborda una tensión que en 2016 seguía sin resolverse: mientras las redes convolucionales profundas (ConvNets) dominaban la clasificación de imágenes estáticas, en el **reconocimiento de acciones en video** su ventaja sobre los métodos tradicionales (por ejemplo, *improved dense trajectories*) todavía no era evidente. Los autores identifican dos obstáculos: (1) las arquitecturas dominantes solo modelaban apariencias y movimientos de **corto rango**, sin capturar la **estructura temporal de largo alcance** de las acciones; y (2) los datasets de acciones disponibles eran pequeños, lo que hacía que las redes muy profundas cayeran fácilmente en sobreajuste.

La respuesta es **Temporal Segment Network (TSN)**, un marco a nivel de video que combina una **estrategia de muestreo temporal esparcido** con **supervisión a nivel de video**. En lugar de procesar frames densos o clips cortos, TSN divide cada video en $K$ segmentos de igual duración, muestrea un solo *snippet* corto de cada segmento, los procesa con una red *two-stream* de parámetros compartidos y **agrega** sus predicciones mediante una **función de consenso** para producir una única predicción a nivel de video. Con este esquema, TSN cubre el video completo con un costo computacional drásticamente menor. El segundo aporte es un estudio sistemático de **buenas prácticas** para entrenar ConvNets profundas sobre datos de video limitados: pre-entrenamiento cruzado de modalidades, normalización por lotes parcial (*partial BN*) con dropout y aumento de datos mejorado. TSN alcanza el estado del arte en **HMDB51 (69.4%)** y **UCF101 (94.2%)**.

Para la **Clase 36 (Introduction to Video Analysis)** este paper es la referencia teórica directa del laboratorio: el LAB muestrea un número fijo de frames distribuidos a lo largo del video para clasificar la acción, que es exactamente la intuición de TSN — muestreo esparcido de segmentos que cubre todo el video sin procesar cada frame.

## 2. Contexto: el problema de los frames densos y el rango temporal corto

Para entender TSN hay que entender qué hacían los métodos previos y por qué fallaban. El reconocimiento de acciones tiene dos aspectos complementarios: **apariencia** (cómo se ven los objetos y la escena) y **dinámica** (cómo se mueven en el tiempo). Extraer y combinar ambos es difícil por variaciones de escala, cambios de punto de vista y movimiento de cámara.

La arquitectura dominante era la **two-stream ConvNet** de Simonyan y Zisserman (2014), que separa esos dos aspectos en dos ramas: un *spatial stream* que opera sobre una sola imagen RGB (captura apariencia) y un *temporal stream* que opera sobre una pila de campos de **flujo óptico** de frames consecutivos (captura movimiento de corto plazo). El problema estructural es que **ambas ramas solo ven un instante o una ventana muy breve**: la rama espacial ve un frame, la temporal ve un stack corto de flujo. Las acciones complejas —una zambullida, un salto de altura, una acción deportiva— constan de **múltiples etapas que se despliegan durante un tiempo relativamente largo**, y estas arquitecturas carecen de la capacidad de incorporar esa estructura temporal de largo alcance.

Los intentos previos de modelar rango temporal largo (Ng et al. con LSTM, Varol et al. con *long-term temporal convolutions*, Donahue et al. con LRCN) operaban directamente sobre streams de video continuos y largos. El problema es que dependían de **muestreo temporal denso** con un intervalo de muestreo predefinido. Esto tiene dos consecuencias negativas que el paper señala explícitamente:

1. **Costo computacional excesivo.** Procesar frames densos de secuencias largas es caro, lo que en la práctica obligaba a estos métodos a trabajar con secuencias de longitud fija acotada — típicamente entre **64 y 120 frames**. Su cobertura temporal quedaba limitada.
2. **Redundancia.** La observación clave del paper es que **los frames consecutivos son altamente redundantes**: muestrear densamente produce frames casi idénticos entre sí. Por lo tanto, el muestreo denso es innecesario; aporta poco a cambio de mucho cómputo, y para videos más largos que la longitud máxima corre el riesgo de perder información importante.

A esto se suma el segundo problema, de datos: entrenar ConvNets profundas requiere muchos ejemplos, pero **UCF101 y HMDB51 son pequeños** en tamaño y diversidad. Las redes muy profundas que triunfaban en clasificación de imágenes se enfrentaban a un alto riesgo de sobreajuste al entrenarse desde cero en estos datasets.

Estos dos obstáculos definen las dos preguntas del trabajo: *(1) cómo diseñar un marco eficiente a nivel de video que capture estructura temporal de largo alcance; (2) cómo entrenar ConvNets profundas con pocas muestras.*

## 3. Contribución central

TSN ataca ambos problemas con dos aportes claramente separables:

1. **Muestreo temporal esparcido por segmentos + consenso a nivel de video.** En vez de descorrelacionar la redundancia con muestreo denso, TSN **elimina la redundancia de raíz**: divide el video en $K$ segmentos de igual duración y muestrea **un único snippet corto por segmento**, distribuido uniformemente a lo largo de la dimensión temporal. Cada snippet produce su propia predicción preliminar de clases, y una **función de consenso segmental** agrega estas predicciones en una única predicción a nivel de video. Como el número de segmentos $K$ es fijo para todos los videos, TSN puede **modelar la estructura de largo alcance sobre el video completo** con un costo drásticamente menor, y —crucialmente— habilita el **aprendizaje end-to-end sobre videos largos** dentro de un presupuesto razonable de tiempo y cómputo. Es el primer marco que modela estructura temporal end-to-end sobre el video entero.

2. **Buenas prácticas para entrenar ConvNets profundas en video.** Para desatar el potencial del marco con arquitecturas muy profundas (BN-Inception), el paper explora sistemáticamente: (a) **pre-entrenamiento cruzado de modalidades**, (b) **partial BN con dropout** como regularización, y (c) **aumento de datos mejorado** (recorte de esquinas y *scale-jittering*). Además, estudia empíricamente cuatro modalidades de entrada: RGB único, diferencia de RGB apilada, flujo óptico apilado y flujo óptico *warped* apilado.

## 4. Método

### 4.1. La arquitectura TSN y el muestreo esparcido de segmentos

TSN es, como la two-stream original, una combinación de *spatial stream* ConvNets y *temporal stream* ConvNets. La diferencia estructural es que **no opera sobre frames individuales ni sobre stacks cortos**, sino sobre una secuencia de snippets cortos **muestreados esparcidamente de todo el video**.

Formalmente, dado un video $V$, se lo divide en $K$ segmentos $\{S_1, S_2, \dots, S_K\}$ de igual duración. Luego TSN modela la secuencia de snippets como:

$$\text{TSN}(T_1, T_2, \dots, T_K) = H\big(G(F(T_1; W), F(T_2; W), \dots, F(T_K; W))\big)$$

donde:

- Cada snippet $T_k$ se muestrea **aleatoriamente** de su segmento correspondiente $S_k$. Esta aleatoriedad dentro de cada segmento es una forma implícita de aumento de datos: distintas épocas ven distintos snippets del mismo segmento.
- $F(T_k; W)$ es la función que representa una ConvNet con parámetros $W$ operando sobre el snippet $T_k$ y produciendo *scores* de clase para todas las clases. Es esencial que **todas las ConvNets sobre todos los snippets comparten los parámetros $W$** — no hay $K$ redes distintas, sino una sola red aplicada $K$ veces.
- $G$ es la **función de consenso segmental** que combina las salidas de los múltiples snippets en un consenso de la hipótesis de clase.
- $H$ es la función de predicción que convierte el consenso en la probabilidad de cada clase de acción para todo el video. Se usa **Softmax** para $H$.

La Figura 1 del paper ilustra el flujo: el video se divide en $K$ segmentos, se selecciona un snippet corto aleatorio de cada uno, cada snippet pasa por una Spatial ConvNet y una Temporal ConvNet, los scores de clase de los distintos snippets se fusionan por la función de consenso segmental para dar el consenso segmental (la predicción a nivel de video), y finalmente las predicciones de todas las modalidades se fusionan para producir la predicción final.

### 4.2. La función de consenso y la función de pérdida

La forma de la función de consenso $G$ es, según los autores, una cuestión abierta. En este trabajo usan la forma más simple, donde el score de consenso de la clase $i$ es:

$$G_i = g\big(F_i(T_1), \dots, F_i(T_K)\big)$$

Es decir, el score de consenso de la clase $i$ se infiere de los scores de **esa misma clase** en todos los snippets, mediante una función de agregación $g$. Se evaluaron empíricamente tres candidatos para $g$: **promedio uniforme** (*evenly averaging*), **máximo** (*max pooling*) y **promedio ponderado** (*weighted average*). El promedio uniforme es el que se usa para reportar las precisiones finales.

Combinando el consenso con la pérdida estándar de **entropía cruzada categórica**, la función de pérdida respecto del consenso segmental $G = G(F(T_1; W), \dots, F(T_K; W))$ es:

$$L(y, G) = -\sum_{i=1}^{C} y_i \left( G_i - \log \sum_{j=1}^{C} \exp G_j \right)$$

donde $C$ es el número de clases de acción e $y_i$ la etiqueta *ground-truth* de la clase $i$. En los experimentos, el número de snippets $K$ se fija en **3**, siguiendo trabajos previos de modelado temporal.

### 4.3. Entrenamiento end-to-end: cómo el consenso propaga gradientes al video completo

La clave de TSN es que **se optimizan las pérdidas de las predicciones a nivel de video**, no las de snippet individual como en la two-stream original. TSN es diferenciable (o al menos tiene subgradientes, según la elección de $g$), lo que permite usar los múltiples snippets para optimizar conjuntamente los parámetros $W$ con retropropagación estándar. El gradiente respecto de $W$ es:

$$\frac{\partial L(y, G)}{\partial W} = \frac{\partial L}{\partial G} \sum_{k=1}^{K} \frac{\partial G}{\partial F(T_k)} \frac{\partial F(T_k)}{\partial W}$$

donde $K$ es el número de segmentos. Cuando se usa descenso de gradiente estocástico (SGD), esta ecuación **garantiza que las actualizaciones de parámetros usan el consenso segmental $G$ derivado de todas las predicciones de snippet**. Optimizado así, TSN aprende parámetros **del video completo, no de un snippet corto**. Y como $K$ es fijo para todos los videos, se ensambla una estrategia de muestreo temporal esparcido donde los snippets muestreados contienen solo una pequeña porción de los frames — reduciendo drásticamente el costo de evaluar las ConvNets frente a los métodos de muestreo denso.

Este es el punto que conviene subrayar: el muestreo esparcido no es un truco de inferencia, sino que está **integrado en el entrenamiento**. El gradiente que ajusta la red se computa a partir del consenso de snippets distribuidos por todo el video, de modo que la red aprende a producir predicciones de snippet que, **agregadas**, describen bien la acción completa.

## 5. Buenas prácticas de entrenamiento

TSN provee el marco, pero para alcanzar rendimiento óptimo hay que sortear el sobreajuste por datos limitados. El paper estudia una serie de buenas prácticas.

**Arquitectura de red.** La two-stream original usaba una red relativamente superficial (ClarifaiNet). TSN adopta **BN-Inception** (Inception con Batch Normalization) como bloque de construcción, por su buen balance entre precisión y eficiencia. La rama espacial opera sobre una imagen RGB y la temporal sobre un stack de campos de flujo óptico consecutivos.

**Modalidades de entrada.** Además de RGB (espacial) y flujo óptico (temporal), el paper propone dos modalidades extra: **diferencia de RGB** entre frames consecutivos (que describe el cambio de apariencia y puede corresponder a regiones de movimiento saliente) y **flujo óptico *warped*** (inspirado en *improved dense trajectories*: se estima una matriz de homografía para compensar el movimiento de cámara, suprimiendo el movimiento de fondo y concentrando el flujo en el actor).

**Pre-entrenamiento cruzado de modalidades (*cross-modality pre-training*).** El pre-entrenamiento es una forma efectiva de inicializar ConvNets cuando el dataset objetivo es pequeño. Para la rama espacial (RGB) es natural usar modelos pre-entrenados en **ImageNet**. Para las otras modalidades (flujo óptico, diferencia de RGB), cuyas distribuciones difieren de RGB, se propone una técnica de pre-entrenamiento cruzado: usar los **modelos RGB para inicializar las redes temporales**. Primero se discretiza el flujo óptico al intervalo $[0, 255]$ por una transformación lineal, igualando su rango al de RGB; luego se modifican los pesos de la primera capa convolucional promediando los pesos a través de los canales RGB y **replicando ese promedio** tantas veces como canales tenga la entrada temporal. Esta inicialización funciona muy bien y reduce el sobreajuste.

**Partial BN con dropout.** Batch Normalization acelera la convergencia, pero al estimar media y varianza dentro de cada batch a partir de pocas muestras introduce un sesgo que lleva a sobreajuste durante la transferencia. La solución de TSN es la **normalización por lotes parcial (*partial BN*)**: tras inicializar con el modelo pre-entrenado, se **congelan la media y la varianza de todas las capas BN excepto la primera**. La primera capa se deja libre porque la distribución del flujo óptico difiere de la de RGB, y hay que re-estimar su media y varianza. Adicionalmente se agrega una capa de **dropout** tras el *global pooling* de BN-Inception, con ratio **0.8 para la rama espacial y 0.7 para la temporal**.

**Aumento de datos mejorado.** Además del recorte aleatorio y volteo horizontal de la two-stream original, TSN introduce dos técnicas nuevas: **recorte de esquinas** (*corner cropping*), donde las regiones se extraen solo de las esquinas o el centro para evitar el sesgo implícito hacia la zona central, y **scale-jittering** multiescala. Para este último se fija la imagen (o campo de flujo) en $256 \times 340$, se eligen aleatoriamente ancho y alto del recorte de entre $\{256, 224, 192, 168\}$, y finalmente se redimensiona a $224 \times 224$. Esto combina jittering de escala con jittering de relación de aspecto.

**Inferencia.** Como todos los snippets comparten parámetros, el modelo aprendido puede evaluarse frame por frame como una ConvNet normal. Siguiendo el protocolo de la two-stream original, se muestrean **25 frames RGB o stacks de flujo** del video, se recortan 4 esquinas + 1 centro con sus volteos horizontales, y se fusionan los scores de las 25 muestras y de ambas ramas **antes de la normalización Softmax**. En la fusión espacial-temporal se da más peso a la rama temporal (peso 1 al espacial, 1.5 al temporal; cuando se usa también flujo *warped*, el peso temporal se reparte 1 para flujo normal y 0.5 para *warped*).

## 6. Experimentos y resultados

Los experimentos se realizan sobre dos datasets grandes de reconocimiento de acciones. **UCF101** contiene 101 clases y 13 320 clips de video, evaluado con los tres splits de train/test del protocolo THUMOS13. **HMDB51** reúne 6 766 clips de 51 categorías provenientes de películas y video web, con tres splits y precisión promedio reportada. El entrenamiento usa SGD con mini-batch de 256 y momentum 0.9, inicialización desde ImageNet, y flujo óptico TVL1 (OpenCV con CUDA). El tiempo de entrenamiento sobre UCF101 con 4 GPUs TITANX es de ~2 horas para el TSN espacial y ~9 horas para el temporal.

**Estrategias de entrenamiento (UCF101 split 1, Tabla 1).** Entrenar desde cero da un two-stream de solo **82.9%**, muy por debajo del baseline de la two-stream original (**87.0%**), confirmando la necesidad de estrategias de aprendizaje cuidadosas contra el sobreajuste. Pre-entrenar la rama espacial sube a **90.0%**; agregar pre-entrenamiento cruzado de modalidades sube a **91.5%**; y agregar partial BN con dropout lo lleva a **92.0%**.

**Modalidades de entrada (UCF101 split 1, Tabla 2).** RGB da 84.5% y diferencia de RGB 83.8%, pero su combinación sube a **87.3%** (información complementaria). Flujo óptico (87.2%) y flujo *warped* (86.9%) rinden parecido, y su fusión llega a 87.8%. Combinar flujo óptico + flujo *warped* + RGB da **92.3%**, superando a la combinación de las cuatro modalidades (91.7%): la diferencia de RGB describe patrones de movimiento similares pero inestables, por lo que se descarta de la combinación final. Se la interpreta como una alternativa de movimiento de baja calidad pero alta velocidad.

**Función de consenso (UCF101 split 1, Tabla 3).** Comparando las tres agregaciones, el **promedio** logra el mejor two-stream (**93.5%**), por encima del máximo (91.6%) y del promedio ponderado (92.4%). Por eso el average pooling es la agregación por defecto.

**Arquitectura (UCF101 split 1, Tabla 4).** Entre BN-Inception, GoogLeNet y VGGNet-16, **BN-Inception** logra el mejor two-stream (92.0%), en línea con su desempeño en clasificación de imágenes. Aplicar el marco TSN sobre BN-Inception (**BN-Inception+TSN**) sube el two-stream de 92.0% a **93.5%**.

**Análisis de componentes (Tabla 5).** Agregando componentes uno a uno: two-stream básico 90.0% → cross-modality pre-training 91.5% → partial BN con dropout 92.0% → **Temporal Segment Networks 93.5%**. TSN mejora el rendimiento **aun cuando todas las buenas prácticas ya están aplicadas**, corroborando que modelar la estructura temporal de largo plazo es crucial.

**Estado del arte (Tabla 6).** Ensamblando tres modalidades y todas las técnicas, TSN alcanza **HMDB51 = 69.4%** y **UCF101 = 94.2%**, superando a métodos tradicionales (iDT, MoFAP) y de aprendizaje profundo (C3D, TDD, FST CN, LTC, KVMF). La mejora sobre el mejor método previo es de **3.9% en HMDB51** y **1.1% en UCF101**.

**Visualización.** Con la herramienta DeepDraw (ascenso de gradiente sobre ruido), los autores visualizan por primera vez el conocimiento de clase dentro de modelos de reconocimiento de acciones. Los modelos entrenados solo con información de corto plazo tienden a confundir la escena y los objetos con la acción (en "Diving", la rama espacial de un solo frame busca agua y plataformas, no la persona), mientras que con el modelado de largo plazo de TSN los modelos se concentran en el humano y capturan distintas poses correspondientes a las etapas de la acción.

## 7. Limitaciones

- **Dependencia del flujo óptico.** El mejor rendimiento requiere flujo óptico (y flujo *warped*), cuyo cálculo (TVL1) es costoso y debe precomputarse, lo que complica el despliegue en tiempo real. El propio paper reconoce la diferencia de RGB como alternativa "de baja calidad, alta velocidad", pero inestable.
- **Consenso simple.** La función de consenso definitiva es un promedio uniforme, que trata todos los segmentos por igual y no aprende relaciones temporales entre ellos (orden, causalidad). Es una agregación de *bag-of-segments*: modela qué ocurre en el video, pero no impone estructura sobre el orden en que ocurre. Trabajos posteriores (por ejemplo, redes de relación temporal) abordan justamente esto.
- **$K$ pequeño y fijo.** Se fija $K = 3$ en entrenamiento. Un $K$ pequeño limita la granularidad temporal capturada; el valor óptimo puede depender del dataset y de la duración típica de las acciones.
- **Datasets acotados.** Los resultados se demuestran en UCF101 y HMDB51, datasets pequeños y de clips recortados (una acción por clip). El marco no aborda directamente la localización temporal en videos largos no recortados.

## 8. Conexión con la Clase 36 y con el LAB

La **Clase 36 (Introduction to Video Analysis)** introduce el problema de clasificar la acción de un video, y el LAB implementa la solución más pragmática: **muestrear un número fijo de frames (8) distribuidos a lo largo del video** y clasificar la acción a partir de ellos. Esa decisión de diseño *es* la idea de TSN. Conviene explicitar la correspondencia:

| Idea de TSN | Reflejo en el LAB de la Clase 36 |
|---|---|
| Dividir el video en $K$ segmentos de igual duración | Distribuir $N=8$ posiciones de muestreo a lo largo del video |
| Muestrear un snippet por segmento | Tomar un frame por posición |
| Muestreo esparcido cubre todo el video con poco cómputo | Con solo 8 frames se representa el video completo, evitando procesar cada frame |
| Los frames consecutivos son redundantes → densidad innecesaria | Muestrear pocos frames bien distribuidos basta para clasificar la acción |
| Función de consenso agrega predicciones de snippet | Se agregan/promedian las representaciones de los frames para una decisión a nivel de video |
| Parámetros compartidos entre snippets | La misma red procesa cada frame muestreado |

La lección conceptual que el estudiante debe internalizar es la **eficiencia del muestreo esparcido**: no hace falta ver todos los frames para reconocer una acción, porque la redundancia entre frames consecutivos es enorme. Distribuir unas pocas muestras uniformemente por la línea de tiempo (i) cubre la estructura temporal de largo alcance de la acción, (ii) reduce el costo por uno o dos órdenes de magnitud y (iii) actúa como regularización implícita frente al sobreajuste. Las buenas prácticas de TSN (pre-entrenamiento, partial BN, aumento de datos) son además el "manual de supervivencia" para entrenar redes profundas de video con datos escasos, un problema recurrente en cualquier aplicación práctica de análisis de video.

**Enlaces internos:**

- Clase: [/clases/clase-36](/clases/clase-36) — Introduction to Video Analysis (prof. Vladimir Araujo).
- Fundamento transversal sugerido: reconocimiento de acciones / análisis de video.
- Antecedente arquitectónico: two-stream ConvNets (Simonyan y Zisserman, 2014) — la base que TSN extiende a nivel de video.

## 9. Nota final: relevancia para video clínico

En el dominio médico, el video clínico —endoscopías, ecografías, laparoscopías, registros de monitoreo o de marcha— suele ser **largo y masivamente redundante**: muchos frames consecutivos son casi idénticos, y procesarlos todos con una red profunda es prohibitivamente caro y, a menudo, innecesario. El muestreo temporal esparcido de TSN ofrece un principio directamente transferible: dividir el registro en segmentos, muestrear pocos frames representativos distribuidos por toda la duración del procedimiento y agregar sus predicciones mediante consenso permite clasificar o caracterizar el video completo (por ejemplo, la fase quirúrgica en curso, la presencia de un hallazgo o la calidad del movimiento evaluado) capturando la estructura de largo alcance del evento clínico a una fracción del costo computacional. Sumado a las buenas prácticas frente a datos escasos —pre-entrenamiento por transferencia y regularización agresiva, tan pertinentes cuando los datasets médicos anotados son pequeños—, TSN sigue siendo una base sensata y económica para llevar el análisis de acciones a videos médicos largos sin costo prohibitivo.
