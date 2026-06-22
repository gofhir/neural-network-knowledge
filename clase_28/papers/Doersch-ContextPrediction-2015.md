# Unsupervised Visual Representation Learning by Context Prediction — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Unsupervised Visual Representation Learning by Context Prediction*.
- **Autores:** Carl Doersch (Carnegie Mellon University y UC Berkeley), Abhinav Gupta (Carnegie Mellon University), Alexei A. Efros (UC Berkeley).
- **Venue:** ICCV 2015 (International Conference on Computer Vision).
- **Año:** 2015. **Preprint:** arXiv:1505.05192v3 (16 ene 2016), [arxiv.org/abs/1505.05192](https://arxiv.org/abs/1505.05192).
- **Financiamiento:** Google Graduate Fellowship (CD), ONR MURI N000141010934, Intel research grant, NVidia hardware grant, Amazon Web Services grant.

Este es uno de los papers **fundacionales del aprendizaje autosupervisado moderno en visión**. Su tesis es directa: el *contexto espacial* dentro de una imagen es una fuente de señal supervisora "gratuita y abundante" que puede entrenar una representación visual rica sin una sola etiqueta humana. La idea es trasplantar al dominio de imágenes lo que `word2vec`/skip-gram (Mikolov et al., 2013) había logrado en texto: convertir un problema aparentemente no supervisado (encontrar una buena métrica de similitud) en uno *autosupervisado* (aprender una función desde un dato a su contexto). En texto, predecir las palabras vecinas fuerza a aprender buenos *word embeddings*; el paper propone el análogo visual.

La tarea concreta —el *pretext task*— es el **posicionamiento relativo de parches**: dada una imagen sin etiquetar, se extrae un parche central y uno de sus ocho vecinos (arriba, arriba-derecha, derecha, ..., arriba-izquierda), se presentan ambos parches a una red sin información sobre su ubicación original, y la red debe clasificar cuál de las ocho configuraciones espaciales fue muestreada. La hipótesis subyacente: para resolver bien esta tarea, la red está obligada a reconocer objetos y sus partes, porque "los objetos consisten en múltiples partes que pueden detectarse independientemente y que ocurren en una configuración espacial específica". Si no hay configuración específica, es *stuff* (textura, fondo), no un objeto.

El resultado sorprendente —y el motivo por el que el paper marcó época— es que una representación entrenada con un objetivo que opera sobre *una sola imagen a la vez* (supervisión a nivel de instancia) **generaliza a tareas a nivel de categoría entre imágenes**: transfiere a detección de objetos en PASCAL VOC y permite descubrimiento no supervisado de objetos (gatos, personas, aves) por minería visual. Para la Clase 28 (Aprendizaje Autosupervisado) este paper importa porque es el **origen de toda la familia de *pretext tasks* espaciales**, el antecedente directo de los jigsaw puzzles (Noroozi & Favaro, 2016) y el ejemplo canónico —que la clase presenta literalmente— de cómo un *pretext* bien diseñado debe defenderse de los *atajos* (*shortcuts*) que el modelo intentará explotar.

## 2. Contexto histórico: aprendizaje no supervisado en visión antes de 2015 y el préstamo desde el NLP

Hacia 2015, los métodos de visión por computador habían explotado gracias a datasets de millones de ejemplos *etiquetados* (ImageNet, AlexNet en Krizhevsky et al., 2012). Pero escalar a datasets verdaderamente "de escala Internet" —cientos de miles de millones de imágenes— chocaba con el costo prohibitivo de la anotación humana. El camino natural era el aprendizaje no supervisado, pero, como el paper reconoce con franqueza, "a pesar de varias décadas de esfuerzo sostenido, los métodos no supervisados aún no habían demostrado extraer información útil de grandes colecciones de imágenes reales de tamaño completo". El problema epistemológico de fondo: sin etiquetas, *ni siquiera está claro qué debería representarse*.

El paper repasa las familias previas y explica por qué fracasaban en imágenes naturales de alta resolución:

- **Modelos generativos** (wake-sleep, contrastive divergence, deep Boltzmann machines, VAE/Kingma & Welling): conciben una buena representación como las variables latentes de un modelo generativo. Pero inferir esa estructura latente es intratable salvo para modelos simples, y aunque funcionan en datasets pequeños (dígitos manuscritos), ninguno había probado ser efectivo en imágenes naturales de alta resolución. El argumento del paper: los métodos basados en *reconstrucción* batallan con fenómenos de bajo nivel como las texturas estocásticas, lo que hace difícil incluso *medir* si el modelo está generando bien.
- **Autoencoders (denoising, sparse)**: usan la reconstrucción desde datos ruidosos como *pretext*. El sparse autoencoder de Le (2013) fue el único aplicado a imágenes de tamaño completo, pero requirió **un millón de horas de CPU para descubrir apenas tres objetos**.
- **Embeddings y clustering con features hechos a mano** (bags of visual words, etc.): pierden información de forma y tienden a descubrir clusters de, digamos, follaje, en vez de objetos.
- **Video y coherencia temporal** (Wang & Gupta, 2015, trabajo contemporáneo): la identidad de un objeto persiste aunque su apariencia cambie en el tiempo, un *cue* alternativo de supervisión.

La clave conceptual viene del **dominio del texto**, donde el contexto había demostrado ser una fuente poderosa de señal automática. El modelo *skip-gram* de Mikolov et al. (2013) entrena una red para predecir, desde una sola palabra, las *n* palabras precedentes y sucesoras. Esto convierte un problema no supervisado en uno "autosupervisado": aprender una función desde una palabra a las que la rodean. La predicción de contexto es solo un *pretext* para forzar al modelo a aprender un buen *embedding*, que luego resulta útil en tareas reales como la similitud semántica.

El paper también explica por qué no basta con copiar la receta de texto al pie de la letra: **predecir píxeles es muchísimo más difícil que predecir palabras**, por la enorme variedad de píxeles que puede producir un mismo objeto semántico. Una idea fértil del NLP fue cambiar de una tarea de *predicción* pura a una de *discriminación* (Collobert & Weston, 2008; Okanohara & Tsujii, 2007): discriminar fragmentos reales de texto de fragmentos con una palabra reemplazada al azar. La extensión naíf a 2D —discriminar imágenes reales de imágenes con un parche reemplazado por uno aleatorio del dataset— sería **trivial**, porque bastaría con discriminar estadísticas de color e iluminación de bajo nivel. La solución de Doersch et al.: clasificar entre múltiples *configuraciones* de parches muestreados de la *misma* imagen, que por construcción comparten iluminación y estadísticas de color, forzando un razonamiento de más alto nivel.

## 3. Contribución central

La contribución es un *pretext task* novedoso —**predecir la posición relativa de dos parches dentro de una imagen**— junto con una arquitectura ConvNet para resolverlo y la demostración empírica de que la representación resultante transfiere a tareas reales sin usar etiquetas externas.

El planteamiento formal es una clasificación de 8 clases. Se muestrea un parche central (azul, en la Figura 1) y uno de sus ocho vecinos posibles (rojo); se presenta el par $(P_1, P_2)$ a la red *sin* información sobre la posición original; la red produce una salida softmax sobre las ocho configuraciones espaciales $Y \in \{1, \dots, 8\}$. El azar da 12.5% de acierto. La hipótesis central, repetida a lo largo del paper: "hacerlo bien en esta tarea requiere entender escenas y objetos", de modo que una buena representación para este *pretext* necesariamente extrae objetos y sus partes para razonar sobre su ubicación espacial relativa.

Tres afirmaciones empíricas sostienen la contribución:

1. **La representación captura similitud visual entre imágenes** (vecinos más cercanos), aunque fue entrenada imagen por imagen.
2. **Transfiere a detección de objetos** en PASCAL VOC 2007 dentro del framework R-CNN, dando un boost significativo sobre una ConvNet inicializada al azar — *el mejor resultado conocido hasta entonces en VOC 2007 sin usar etiquetas fuera del dataset*.
3. **Permite descubrimiento/minería visual no supervisada** de objetos en VOC 2011 y en imágenes de Paris Street View.

El segundo gran aporte, menos citado pero igual de influyente, es **metodológico**: la identificación y el combate sistemático de los *atajos* (*trivial solutions* / *shortcut learning*) que un modelo explota para resolver el *pretext* sin aprender lo que uno quiere. Esta sección (§3.1 del paper) se volvió lectura obligada en SSL.

## 4. Método: muestreo, arquitectura siamesa y entrenamiento

### 4.1. Muestreo de parches

Dada una imagen, el primer parche se muestrea uniformemente, sin referencia al contenido. Dada su posición, el segundo se muestrea al azar entre las ocho ubicaciones vecinas (Figura 2). Por eficiencia, los parches se toman de un patrón en cuadrícula (grid), de modo que cada parche puede participar en hasta 8 emparejamientos distintos. Detalles concretos de implementación:

- Las imágenes se redimensionan a entre 150K y 450K píxeles totales, preservando el aspect ratio.
- Los parches se muestrean a resolución **96×96**.
- Se deja un **gap de 48 píxeles** entre parches en la cuadrícula (aproximadamente la mitad del ancho del parche).
- Se aplica **jitter** de −7 a 7 píxeles en cada dirección a la ubicación de cada parche.
- Preprocesamiento: (1) resta de la media, (2) proyección o *dropping* de colores (ver §4.3), (3) *downsampling* aleatorio de algunos parches a tan poco como 100 píxeles totales y luego *upsampling*, para construir robustez a la pixelación.

### 4.2. Arquitectura: red siamesa de fusión tardía con pesos compartidos

La arquitectura (Figura 3) es una **red siamesa** ("late-fusion"): un par de torres estilo AlexNet que procesan cada parche por separado hasta una profundidad análoga a `fc6` de AlexNet, punto en el cual las representaciones se fusionan. Para las capas que procesan un solo parche, **los pesos están atados (compartidos) entre ambos lados de la red** (líneas punteadas en la Figura 3), de modo que se computa exactamente la misma función de *embedding* a nivel de `fc6` para ambos parches. Esto es lo que la Clase 28 describe como "2 redes conv con pesos compartidos".

La pila por torre sigue AlexNet donde es posible: `conv1`(11×11,96,4) → `pool1` → `LRN1` → `conv2`(5×5,384,2) → `pool2` → `LRN2` → `conv3`(3×3,384,1) → `conv4`(3×3,384,1) → `conv5`(3×3,256,1) → `pool5` → `fc6`(4096). Tras `fc6` se concatenan ambas torres y siguen `fc7`(4096) → `fc8`(4096) → `fc9`(8) → softmax. Como solo dos capas (`fc7`, `fc8`) reciben entrada de ambos parches, hay **capacidad limitada para razonamiento conjunto**, lo que obliga a la red a hacer el grueso del razonamiento semántico *por parche, separadamente* — exactamente lo que se busca para que la representación de un parche individual sea útil. Todas las capas conv y fc van seguidas de ReLU, salvo `fc9` que alimenta el softmax.

### 4.3. El problema central: evitar soluciones triviales (*shortcut learning*)

Esta es la sección más influyente metodológicamente. El principio: al diseñar un *pretext*, hay que asegurarse de que la tarea **fuerce** a la red a extraer la información deseada (semántica de alto nivel) *sin* tomar atajos triviales. El paper identifica y combate tres atajos sucesivos:

1. **Continuidad de bordes y texturas.** Patrones de borde o texturas que continúan entre parches adyacentes podrían delatar la respuesta sin entender nada semántico. Mitigación: el **gap** de ~medio ancho de parche entre los dos parches.
2. **Líneas largas que cruzan parches vecinos.** Incluso con gap, una línea recta que atraviesa parches contiguos puede revelar la configuración. Mitigación: el **jitter** aleatorio de hasta 7 píxeles en la posición de cada parche.
3. **Aberración cromática (el atajo más insidioso).** Este fue el hallazgo que más sorprendió a los autores. La aberración cromática surge de que la lente enfoca la luz de distintas longitudes de onda de manera diferente; en muchas cámaras, un canal de color (típicamente el verde) se "encoge" hacia el centro de la imagen respecto a los otros. Resulta que una ConvNet **puede aprender a localizar un parche respecto a la lente misma** detectando la separación entre verde y magenta (rojo + azul). Una vez que la red conoce la *posición absoluta* en la lente, resolver la posición *relativa* se vuelve trivial — la red resuelve el *pretext* sin aprender absolutamente nada de semántica. Mitigaciones, dos estrategias probadas:
   - **Proyección ('projection'):** desplazar verde y magenta hacia el gris. Sea $a = [-1, 2, -1]$ (el eje de color verde-magenta en RGB); se define $B = I - a^T a / (a a^T)$, matriz que sustrae la proyección de un color sobre el eje verde-magenta, y se multiplica cada píxel por $B$.
   - **Color dropping:** descartar al azar 2 de los 3 canales de color por parche, reemplazando los canales caídos con ruido gaussiano (desviación estándar ∼1/100 de la del canal restante).

   Ambas estrategias rinden similar; los resultados cualitativos usan *color-dropping* y para detección se reportan ambas. La §4.2 del paper (un "aside") cuantifica la *aprendibilidad* de la aberración: una red entrenada a predecir las coordenadas absolutas $(x,y)$ de parches logra un RMSE de .255 en el 10% mejor de imágenes (el azar —predecir siempre el centro— da .371); aplicar la proyección sube el error a .321, confirmando que el atajo fue suprimido.

### 4.4. Estabilización del entrenamiento

Con SGD simple, las predicciones de la red **degeneraban a una distribución uniforme** sobre las 8 categorías, con todas las activaciones de `fc6` y `fc7` colapsando a 0: la optimización quedaba atascada en un *saddle point* donde ignoraba la entrada de las capas inferiores (lo que minimizaba la varianza de la salida final), sin poder afinar las features de bajo nivel para escapar. La solución final usa **batch normalization** (Ioffe & Szegedy, 2015) *sin* los parámetros de escala y shift ($\gamma$ y $\beta$), lo que fuerza a las activaciones a variar entre ejemplos, más **momentum alto** (p.ej. .999) que aceleró el aprendizaje. El entrenamiento usó Caffe sobre el set de entrenamiento de ImageNet 2012 (~1.3M imágenes, descartando las etiquetas) y corrió **~4 semanas en una GPU K40**.

## 5. Experimentos y resultados

### 5.1. Vecinos más cercanos (qué se aprende)

Para entender qué considera "similar" la red, se muestrean parches 96×96 al azar y se representan con features `fc6` (usando solo una de las dos torres), buscando vecinos por correlación normalizada. La Figura 4 compara tres features: inicialización aleatoria, AlexNet `fc7` entrenado en ImageNet con etiquetas, y el `fc6` del método propuesto. Los *matches* del método propuesto **capturan información semántica** y, en algunos casos (p.ej. la rueda de un auto), capturan mejor la *pose* que AlexNet. Curiosamente, en algunos casos una ConvNet aleatoria también funciona razonablemente. Este es exactamente el slide "qué se aprende" de la Clase 28, donde se muestran las *nearest sections* recuperadas por la representación autosupervisada.

### 5.2. Detección de objetos en PASCAL VOC 2007 (la tabla clave)

La representación se usa como pre-entrenamiento dentro del pipeline R-CNN (Girshick et al., 2014). Como el algoritmo opera sobre parches 96×96 pero R-CNN sobre propuestas redimensionadas a 227×227, se adapta la arquitectura (Figura 6): se transfieren las capas `conv1`–`pool5`, se convierte `fc6` en una capa convolucional (`conv6`), se añade `conv6b` (kernel 1×1) que reduce a 1024 canales, y un `fc7` final. Se hace *fine-tuning* siguiendo R-CNN, sin *bounding-box regression*. Resultados de mAP en VOC-2007 (Tabla 1 del paper):

| Modelo (AlexNet-style) | mAP |
|---|---|
| Scratch-R-CNN (AlexNet desde cero) | 40.7 |
| Scratch-Ours (nuestra arq. desde cero) | 39.8 |
| **Ours-projection** | **45.7** |
| **Ours-color-dropping** | **46.3** |
| Ours-Yahoo100m (preentrenado en 2M de Flickr 100M) | 44.2 |
| ImageNet-R-CNN (preentrenado con etiquetas ImageNet) | 54.2 |

El pre-entrenamiento autosupervisado da un **boost de ~6% de mAP** sobre la arquitectura desde cero, y supera por más de 5% a un AlexNet entrenado desde cero en PASCAL — dejando al método a unos **8% detrás de R-CNN preentrenado con etiquetas de ImageNet**. Era el mejor resultado conocido en VOC 2007 sin usar etiquetas externas al dataset. El experimento con Yahoo/Flickr 100M (recolectado de forma completamente automática) confirma que el método no depende de los sesgos curatoriales de ImageNet. Con la rescalado de Krähenbühl et al. (2015) se obtiene 51.1 (Ours-rescale) y, escalando a un backbone **VGG-16** de 16 capas (entrenado ~8 semanas en una Titan X), el método alcanza **61.7 mAP** (VGG-Ours-rescale), muy por encima del VGG inicializado con K-means (42.4), evidenciando que casi todo el boost proviene del pre-entrenamiento no supervisado.

> **Nota sobre la Clase 28:** la tabla PASCAL de la clase reporta el posicionamiento relativo en **65.3**. Esa cifra corresponde a evaluaciones/benchmarks posteriores y armonizados de SSL (típicamente clasificación o detección con backbones y protocolos más modernos que los de 2015), no a la mAP de 46.3 que el paper original reporta para AlexNet en VOC-2007. La discrepancia es esperable: la clase usa una tabla comparativa estandarizada entre métodos *pretext* (posicionamiento relativo, jigsaw, colorización, etc.), mientras que el paper de 2015 reporta su número de la época. Lo invariante es el mensaje: el posicionamiento relativo de Doersch et al. es competitivo entre los *pretext* espaciales.

### 5.3. Estimación de geometría (NYUv2)

Para verificar que la representación no es solo "sensible a objetos" sino útil en tareas no basadas en objetos, se hizo *fine-tuning* para estimación de normales de superficie en NYUv2 (Tabla 2). Sorprendentemente, los resultados son **casi equivalentes** a los de un modelo ImageNet completamente etiquetado (Mean 33.2 vs 33.3; el método propuesto incluso iguala o supera en varias métricas), y mejor que entrenar desde cero (38.6) o que el *tracking* no supervisado de Wang & Gupta (34.2). El paper conjetura que la categorización en ImageNet hace relativamente poco por incentivar atención a la geometría, ya que esta es irrelevante una vez identificado el objeto.

### 5.4. Minería visual / descubrimiento de objetos

La idea: si se muestrean dos parches no superpuestos del *mismo* objeto, no solo sus listas de vecinos comparten imágenes, sino que dentro de esas imágenes los vecinos están en aproximadamente la misma configuración espacial; para texturas, en cambio, las configuraciones serían aleatorias. El algoritmo muestrea constelaciones de **cuatro parches adyacentes**, encuentra las 100 imágenes con los *matches* más fuertes, y filtra por **verificación geométrica**. Aplicado a VOC 2011 sin etiquetas ni pre-filtrado, el método descubre objetos —incluyendo aves, torsos y monitores que el trabajo previo (Doersch et al., 2014) no hallaba— y la presencia de aves y torsos (notoriamente deformables) evidencia las invarianzas aprendidas. En la curva purity-coverage sobre un subconjunto de VOC 2007 (Figura 9), el método gana sustancialmente en *coverage* (AUC .87 vs .83 del trabajo previo), aunque pierde algo de pureza. También se aplicó a 15,000 imágenes de Paris Street View, capturando layout de escena y elementos arquitectónicos.

### 5.5. Precisión en la propia tarea de *pretext*

Sobre 500 imágenes de VOC 2007 (256 pares por imagen), la red clasifica las 8 posiciones relativas con **38.4% de acierto** (azar = 12.5%), confirmando que el *pretext* es genuinamente difícil — la precisión humana es similar. En ImageNet (datos de entrenamiento) da 39.5% en train y 40.3% en validación, indicando **poco overfitting**. Restringir los parches a *bounding boxes* de objetos da casi la misma precisión (39.2%; 45.6% solo en autos), lo que revela que la red es sensible a objetos pero **casi igual de sensible al layout del resto de la imagen**.

## 6. Limitaciones reconocidas

- **El gran tema es el *shortcut learning*.** La lección más profunda y honesta del paper es que el modelo explotará *cualquier* atajo de bajo nivel disponible (continuidad de bordes, líneas largas, aberración cromática) en vez de aprender semántica. El descubrimiento accidental de la aberración cromática muestra que el diseñador *no puede anticipar todos los atajos a priori*; hay que descubrirlos empíricamente (vía vecinos más cercanos, vía el regresor de coordenadas absolutas) y neutralizarlos uno a uno. Esto deja una pregunta abierta inquietante: ¿qué atajos *no* se detectaron?
- **Brecha frente a la supervisión.** A pesar del boost, el método queda ~8% de mAP detrás del pre-entrenamiento supervisado en ImageNet (en la era AlexNet). El SSL de 2015 reduce la brecha pero no la cierra.
- **Costo computacional.** ~4 semanas en K40 (AlexNet) y ~8 semanas en Titan X (VGG), un costo considerable para la época.
- **Inestabilidad de optimización.** Sin batch norm sin escala/shift y momentum alto, la red colapsa a predicción uniforme; la solución es algo ad-hoc.
- **Sensibilidad al layout, no solo a objetos.** Que la precisión en *pretext* sea casi igual dentro y fuera de bounding boxes sugiere que la representación mezcla señal de objeto con señal de layout global de la escena — útil, pero no es un detector de objetos puro.
- **Sin máscara de objeto en minería.** El algoritmo de descubrimiento pierde algo de pureza frente al trabajo previo y no determina automáticamente una máscara de objeto.

## 7. Impacto: el origen de los *pretext* espaciales

Este paper es ampliamente reconocido como uno de los **detonantes del aprendizaje autosupervisado moderno en visión**. Su influencia se despliega en varias direcciones:

- **Fundó la familia de *pretext tasks* espaciales.** El posicionamiento relativo de pares fue generalizado casi de inmediato por **Noroozi & Favaro (2016)** a los *jigsaw puzzles*: en vez de dos parches y 8 posiciones, una grilla de 3×3=9 parches barajados cuya permutación hay que predecir. La conexión es directa y el paper de jigsaw cita explícitamente a Doersch et al. como antecedente. De ahí en adelante: rotación (Gidaris et al., 2018), colorización (Zhang et al., 2016), in-painting (Pathak et al., 2016), todos en la misma tradición de "diseñar un *pretext* a partir de la estructura inherente de la imagen".
- **Institucionalizó el combate a los atajos.** La discusión de aberración cromática se volvió el ejemplo de manual de *shortcut learning* en SSL, citado en surveys y cursos (incluida esta Clase 28). Diseñar un *pretext* hoy implica, por defecto, preguntarse "¿qué atajo trivial podría resolver esto?".
- **Antecedió al contrastive learning.** La intuición de que parches del mismo objeto deben acercarse en el espacio de *embedding* —y la arquitectura siamesa de pesos compartidos— prefigura la línea que culmina en SimCLR, MoCo y BYOL (2020), donde la "predicción de contexto" se reemplaza por "invarianza a aumentaciones" pero la maquinaria siamesa permanece.
- **Validó la transferencia instancia → categoría.** Demostrar que un objetivo definido sobre una sola imagen mejora tareas de categoría entre imágenes fue un resultado conceptualmente importante que dio confianza a toda la agenda de SSL.

## 8. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

La Clase 28 usa este trabajo como el **caso de estudio canónico del posicionamiento relativo**, y el mapeo es directo:

- **El *pretext* de posicionamiento relativo.** La clase lo presenta exactamente como en el paper: *Doersch et al.*, **dos redes convolucionales con pesos compartidos** (la red siamesa de fusión tardía de §4.2), que reciben dos parches y deben **predecir cuál de las 8 posiciones relativas** ocupa uno respecto al otro. Es el ejemplo con el que la clase introduce la idea de que un *pretext task* fabrica supervisión a partir de la estructura espacial inherente de la imagen, sin etiquetas humanas — el análogo visual de skip-gram que la clase contrasta con word2vec.

- **La tabla PASCAL.** La clase incluye una tabla comparativa de métodos *pretext* evaluados por transferencia a PASCAL, donde el **posicionamiento relativo aparece con 65.3**. Como se discutió en §5.2, esa cifra proviene de un benchmark armonizado posterior (con backbones y protocolos modernos), no de la mAP de 46.3 del paper original de 2015 con AlexNet en VOC-2007. La lección que la clase transmite con esa tabla es comparativa: dónde se ubica el posicionamiento relativo frente a jigsaw, colorización, rotación, etc. Conviene tener presente la diferencia de protocolo para no confundir el número de la clase con el del paper.

- **El slide "qué se aprende".** La clase muestra los **vecinos más cercanos** (*nearest sections*) recuperados por la representación autosupervisada — exactamente la Figura 4 de §5.1. Es la evidencia cualitativa de que el *embedding* aprendido por el *pretext* captura similitud semántica (parches de ruedas con ruedas, de personas con personas), validando la hipótesis de que resolver el posicionamiento relativo *requiere* reconocer objetos y partes.

- **La gran moraleja didáctica: los atajos.** Para la Clase 28, el valor pedagógico más duradero del paper no es solo el *pretext* sino la **§3.1 sobre soluciones triviales**. El gap, el jitter y —sobre todo— la aberración cromática son el ejemplo perfecto de que *un pretext mal diseñado se resuelve sin aprender nada útil*. Esto conecta con el hilo conductor de la clase sobre por qué el diseño de tareas autosupervisadas es delicado y por qué la comunidad eventualmente migró hacia objetivos contrastivos/de invarianza que son más difíciles de "hackear" con atajos de bajo nivel.

Este análisis enlaza con el fundamento transversal [/fundamentos/aprendizaje-autosupervisado](/fundamentos/aprendizaje-autosupervisado) y con la [Clase 28](/clases/clase-28).
