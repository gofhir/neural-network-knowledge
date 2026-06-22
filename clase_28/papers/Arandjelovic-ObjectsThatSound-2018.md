# Objects that Sound — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Objects that Sound*.
- **Autores:** Relja Arandjelović (DeepMind) y Andrew Zisserman (DeepMind; VGG, Department of Engineering Science, University of Oxford).
- **Venue:** ECCV 2018 (European Conference on Computer Vision).
- **Año:** 2018. **Preprint:** arXiv:1712.06651v2 (25 jul 2018), [arxiv.org/abs/1712.06651](https://arxiv.org/abs/1712.06651).
- **Resultados en video:** playlist de localización en [https://goo.gl/JVsJ7P](https://goo.gl/JVsJ7P).

Este paper persigue dos objetivos, ambos resueltos con la **misma señal de supervisión gratuita**: la *correspondencia audio-visual* (AVC). Primero, redes que **embeben audio e imagen en un espacio común** apto para recuperación cross-modal (consultar con una imagen y recuperar sonidos relacionados, y viceversa). Segundo, una red que, dada una señal de audio, **localiza en la imagen el objeto que produce ese sonido**, sin un solo ejemplo etiquetado. La tesis central es que entrenar desde video sin etiquetar usando solo AVC como función objetivo es suficiente para ambas capacidades: es una forma de **autosupervisión cross-modal** a partir de video.

La idea de la tarea AVC, heredada del trabajo previo de los mismos autores (*Look, Listen and Learn*, ICCV 2017), es elegante por su gratuidad: dado un par formado por un *frame* de video y 1 segundo de audio, la red debe decidir si están en correspondencia o no. Las etiquetas de positivos (par que coincide) y negativos (par desemparejado) se obtienen directamente del propio video — *frame* y audio del mismo instante de un video son un positivo, mientras que *frame* y audio de videos distintos son un negativo. Como las etiquetas se construyen del dato mismo, esto es autosupervisión pura: no hay anotación humana involucrada en el entrenamiento.

Las cuatro contribuciones que el paper enumera explícitamente son: (i) mostrar que se pueden aprender *embeddings* de audio y visión que habilitan recuperación tanto **intra-modal** (audio-a-audio, imagen-a-imagen) como **cross-modal**; (ii) explorar varias arquitecturas para AVC, incluyendo flujos visuales que ingieren un solo *frame*, múltiples *frames*, o un *frame* más flujo óptico multi-frame; (iii) mostrar que el objeto semántico que suena dentro de una imagen puede ser **localizado** (usando solo el sonido, sin información de movimiento ni flujo); y (iv) dar una *cautionary tale* sobre cómo evitar atajos indeseables en la preparación de los datos.

Para la Clase 28 (Aprendizaje Autosupervisado) este paper importa porque materializa el slide "Correspondencia Audio-Visual": modificando levemente una arquitectura de clasificación binaria de correspondencia se obtiene, sin supervisión adicional, un localizador del objeto que suena.

## 2. Contexto histórico: del par contrastivo a *Objects that Sound*

Hacia 2016–2018 hubo una explosión de interés en el aprendizaje cross-modal desde imágenes y audio (SoundNet de Aytar et al. 2016; Harwath et al. 2016; Owens et al. *Ambient sound* 2016). El motor de esa explosión fue un recurso casi ilimitado: videos de YouTube, que entregan simultáneamente un *stream* de imagen y uno de audio sincronizado, y esa información cross-modal sirve para entrenar redes profundas sin etiquetas.

El aprendizaje cross-modal tiene una larga historia en visión, principalmente bajo la forma de **imágenes y texto** (Barnard et al. 2003; DeViSE de Frome et al. 2013). El paper marca por qué audio es un compañero distinto de texto: aunque ambos son secuenciales, **el texto está mucho más cerca de una anotación semántica**. Con texto, el concepto ('un perro') está directamente disponible y el problema se reduce a aterrizar la palabra en una región. Con audio, obtener la semántica es indirecto y se parece más a clasificación de imágenes: el concepto *perro* no está en la señal cruda, sino que requiere algo como una ConvNet para extraerlo. Esta asimetría hace al problema de audio interesante y difícil.

*Objects that Sound* es la **continuación directa de *Look, Listen and Learn* (L3-Net, ICCV 2017)** de los mismos autores. La L3-Net introdujo la tarea AVC y demostró que con ella se aprenden buenas representaciones de audio y de visión. Pero la L3-Net fusiona las dos modalidades por **concatenación** seguida de capas *fully connected*, y solo después calcula el *score* de correspondencia. Ese diseño tiene dos consecuencias que este paper ataca: los *embeddings* de audio y visión **no quedan alineados** en ningún espacio común (inservibles para recuperación cross-modal), y la red nunca fue diseñada ni demostrada para responder *dónde* está el objeto que suena. *Objects that Sound* rediseña la arquitectura para resolver ambas carencias manteniendo intacta la señal de entrenamiento AVC.

El paper también se posiciona frente a la autosupervisión visual de la época —predicción de contexto (Doersch et al. 2015), colorización (Zhang et al. 2016), *jigsaw puzzles* (Noroozi & Favaro 2016), *shuffle and learn* (Misra et al. 2016)— como un caso de la misma familia: inventar una tarea *pretext* cuyas etiquetas salen gratis del dato.

## 3. Contribución central

La contribución no es una nueva función de pérdida ni un nuevo dato, sino **dos arquitecturas de red diseñadas a propósito**, cada una habilitando una funcionalidad nueva a partir de la misma tarea AVC sin etiquetas:

1. **AVE-Net (Audio-Visual Embedding Network):** produce *embeddings* de audio e imagen **directamente alineados** en un espacio común de 128 dimensiones, aptos para recuperación cross-modal. El truco de diseño es forzar que la única información que decide la correspondencia sea la **distancia euclidiana** entre los dos *embeddings* normalizados.

2. **AVOL-Net (Audio-Visual Object Localization Network):** en lugar de un *embedding* global de la imagen, mantiene una rejilla espacial de descriptores locales y calcula la similitud de cada uno con el *embedding* de audio, produciendo un **mapa de localización** que revela en qué parte de la imagen está el objeto que suena. Esto opera sin supervisión alguna, ni de la ubicación del objeto ni de su identidad.

La elegancia del aporte es que ambas redes nacen de **modificar la misma red base** (los mismos *backbones* de visión y audio) y entrenarse con la misma tarea binaria de correspondencia. El paper demuestra que la AVE-Net supera incluso a *baselines* supervisados en recuperación, y que la AVOL-Net localiza objetos sonoros de forma robusta. La cuarta contribución —la *cautionary tale* sobre atajos— es metodológica pero crítica: muestra que un detalle de muestreo aparentemente inocuo puede inflar artificialmente la métrica de la tarea *pretext* mientras degrada las representaciones reales.

## 4. Método

### 4.1. La tarea AVC como objetivo autosupervisado

La entrada es un par (*frame*, 1 s de audio). El positivo se genera muestreando un video al azar, eligiendo un *frame* al azar y tomando 1 segundo de audio con ese *frame* en su punto medio. El negativo toma *frame* y audio de **videos distintos**. La red predice corresponde / no corresponde, con pérdida de entropía cruzada binaria.

Por qué AVC fuerza representaciones semánticas: la única manera de que la red resuelva la tarea es **clasificar conceptos semánticos en ambas modalidades** y luego juzgar si los dos conceptos concuerdan. Además, la red visual ve un **solo *frame***, de modo que no puede hacer trampa explotando información de movimiento — debe entender el contenido estático.

### 4.2. AVE-Net: alineamiento de *embeddings* por distancia euclidiana

La imagen (224×224×3) y el audio (1 s a 48 kHz, convertido a log-espectrograma tratado como imagen en escala de grises de 257×200) pasan por dos *subnetworks* (visión y audio). Cada *subnetwork* produce un *embedding* de **128-D normalizado en L2**. Se calcula la **distancia euclidiana** entre los dos vectores de 128-D, y ese **escalar único** pasa por una FC diminuta que lo escala y desplaza para calibrarlo antes del softmax. El sesgo de esa FC aprende efectivamente el umbral de distancia sobre el cual el par se declara *no correspondiente*.

La clave de diseño es ese **cuello de botella de información**: como lo único que decide la correspondencia es la distancia euclidiana entre los dos *embeddings*, la red está **obligada a alinear ambas modalidades en el mismo espacio**. Más aún, usar la distancia euclidiana durante el entrenamiento hace que las *features* sean "conscientes" de la métrica de distancia, lo que las vuelve aptas para recuperación (técnica heredada de NetVLAD, Arandjelović et al. 2017).

El contraste con L3-Net es directo: L3-Net concatena las *features* y deja que las FC produzcan el *score*, sin nada que fuerce alineamiento. La AVE-Net **mueve las capas FC dentro de cada *subnetwork*** y optimiza las *features* directamente para recuperación. El paper relaciona su entrenamiento con el *metric learning* vía *contrastive loss* (Chopra et al. 2005), pero destaca dos ventajas: (i) es **libre de hiperparámetros** (no requiere ajustar el margen de la *contrastive loss*), y (ii) computa explícitamente la salida corresponde-o-no, haciéndola directamente comparable con L3-Net.

Un detalle sutil del Apéndice B: en su forma cruda, nada impide que la red aprenda *embeddings anti-alineados* (distancia grande = alta similitud). Para estimular el comportamiento deseado (distancia pequeña = similitud alta), basta con **inicializar la FC diminuta con el signo correcto** de pesos, sin necesidad de forzarlo durante el entrenamiento.

### 4.3. AVOL-Net: del *embedding* global al mapa de correspondencia espacial

Aquí está la "pequeña modificación de arquitectura" que el slide de la Clase 28 menciona. En lugar de aprender un *embedding* único de la imagen entera que explique el sonido, el objetivo es **hallar las regiones de la imagen** que explican el sonido, dejando el resto como fondo. El paper lo formula en el marco de **Multiple Instance Learning (MIL)**.

Los cambios concretos respecto de la AVE-Net:

- La *subnetwork* de visión **no hace *pooling* global** de las *features* conv4_2; sigue operando a resolución **14×14**. Las dos FC de visión (fc1, fc2) se convierten en sus equivalentes *fully convolutional* (convoluciones 1×1: conv5, conv6).
- Se **elimina la normalización de *features*** para permitir que las regiones de fondo tengan respuesta baja.
- Se calcula la similitud (producto escalar) entre cada uno de los **14×14 descriptores visuales de 128-D** y el **único descriptor de audio de 128-D**, produciendo un **mapa de similitud de 14×14**.
- Esos *scores* se calibran con una convolución 1×1 diminuta (la fc3 convertida a *fully convolutional*), seguida de una **sigmoide**, que produce un *score* de correspondencia imagen-audio **por cada posición espacial**.
- Un ***max-pooling* sobre todas las posiciones** entrega el *score* de correspondencia final, que se usa para entrenar la tarea AVC con *logistic loss*.

La lógica MIL: para pares correspondientes, el método **incentiva que una región responda alto** (y por tanto localice el objeto); para pares desemparejados, el *score* máximo debe ser bajo, dejando todo el mapa apagado (indicando, correctamente, que no hay objeto que produzca ese sonido). En esencia, **la representación de audio actúa como un filtro que "busca" parches relevantes en la imagen, de forma análoga a un mecanismo de atención** — el paper lo describe como una "atención infinitamente dura". A diferencia de L3-Net, cuyos *heatmaps* se producían examinando neuronas dada solo la imagen (independientes del sonido), aquí la salida **depende del sonido**: cambia el audio, cambia la región resaltada.

### 4.4. Variantes con múltiples *frames* y flujo óptico

El paper explora AVE+MF (25 *frames*, convoluciones 2D→3D) y AVE+OF (un *frame* + 10 *frames* de flujo óptico TV-L1, vía red de dos *streams* estilo Simonyan & Zisserman 2014). Estas variantes suben la *accuracy* de AVC (84.7% y 84.9% vs 81.9%), pero **no mejoran la recuperación**. La explicación es reveladora para la autosupervisión: con movimiento disponible, la red puede resolver AVC explotando correlaciones de bajo nivel (los dedos que se mueven al tocar la guitarra cambian junto con el sonido), lo que **reduce el incentivo para aprender *embeddings* semánticos buenos**. Por eso todos los experimentos principales usan un solo *frame*.

## 5. Experimentos y resultados

### 5.1. Dataset: AudioSet-Instruments

Se usa **AudioSet** (Gemmeke et al. 2017), clips de 10 s de YouTube con énfasis en eventos de audio y etiquetas a nivel de video (ruidosas, organizadas en una ontología). Se filtra a sonidos de **instrumentos musicales, canto y herramientas**, quedando **110 clases de audio**. El dataset resultante tiene **263k / 30k / 4.3k** clips de 10 s en *train* / *val* / *test*. El paper recalca: **no se usa ninguna etiqueta para entrenar** — el dataset se trata como una colección de videos sin etiqueta, y las etiquetas solo sirven para la evaluación cuantitativa.

Los videos son desafiantes: muchos de baja calidad, la fuente sonora no siempre es visible, y el audio a menudo está insertado sobre visuales no relacionados (carátula de álbum, texto con el nombre de la canción, *frame* fijo del músico, o incluso un paisaje).

### 5.2. Recuperación cross-modal e intra-modal

Métrica: **nDCG@30** (normalized discounted cumulative gain, top-30, normalizado a [0,1]), con relevancia definida vía la **distancia de árbol en la ontología de AudioSet** (relevancia = C − d, con C = 20). Se prueban las cuatro combinaciones de consulta/base de datos (imagen/audio × imagen/audio).

En la propia tarea AVC, la AVE-Net logra **81.9%**, batiendo ligeramente a L3-Net (80.8%). Pero el paper insiste en que AVC es solo un *proxy*; lo que importa es la recuperación. Resultados de la Tabla 1 (nDCG@30):

| Método | im-im | im-aud | aud-im | aud-aud |
|---|---|---|---|---|
| Azar | .407 | .407 | .407 | .407 |
| L3-Net | .567 | .418 | .385 | .653 |
| L3-Net + CCA | .578 | .531 | .560 | .649 |
| VGG16-ImageNet (supervisado) | .600 | – | – | – |
| VGG16-ImageNet + L3-Audio CCA | .493 | .458 | .464 | .618 |
| **AVE-Net** | **.604** | **.561** | **.587** | **.665** |

Lecturas clave:

- **Cross-modal (im-aud, aud-im):** la AVE-Net bate a todos los *baselines*. Las *features* crudas de L3-Net dan recuperación cross-modal **a nivel de azar** (.418, .385), confirmando que no están alineadas. Alinearlas con CCA *post hoc* (.531, .560) ayuda mucho, pero entrenar la alineación directamente es mejor.
- **Intra-modal (im-im, aud-aud):** la AVE-Net incluso **supera levemente a VGG16-ImageNet** (.604 vs .600) en imagen-a-imagen, pese a que VGG16 fue entrenada de forma totalmente supervisada en otra tarea, y **aunque la red nunca vio pares de la misma modalidad durante el entrenamiento** — funciona por **transitividad**: la imagen de un violín está cerca del sonido de un violín, que a su vez está cerca de otras imágenes de violines.
- Alinear *features* de ImageNet con audio L3 mediante CCA rinde **peor** que otros métodos: no basta usar redes preentrenadas en ImageNet como extractores de caja negra. Errores cualitativos razonables: confundir una cítara con una guitarra acústica.

### 5.3. Localización del objeto que suena

La AVOL-Net logra en AVC **la misma *accuracy* que la AVE-Net**, lo que es alentador: cambiar al esquema MIL no cuesta capacidad de detectar conceptos semánticos. Cualitativamente (Figs. 5–7) localiza un rango amplio de objetos —teclados, acordeones, tambores, arpas, guitarras, violines, xilófonos, bocas de personas cantando, saxofones— bajo *clutter* significativo y variaciones de iluminación, escala y punto de vista, e incluso **múltiples objetos** (dos violines, dos personas cantando, una orquesta entera). Como es un método no supervisado, a veces enfoca solo partes discriminativas (la interfaz entre las manos y el teclado del piano) más que el objeto completo — lo que conecta con la pregunta filosófica de *qué es el objeto que produce el sonido*.

**Refutación de la hipótesis de saliencia.** La preocupación obvia es que la red simplemente detecte el objeto *saliente* de la imagen, ignorando el sonido. El paper la descarta con un experimento ingenioso de pares **desemparejados** (Fig. 6): dada una imagen de un violín, si se reproduce un sonido de tambores el mapa de localización queda **vacío**; si se reproduce otro violín, resalta el violín. Y de forma decisiva: ante una imagen con **un piano y una flauta**, reproducir sonido de flauta resalta la flauta, y reproducir piano resalta el piano. La red, por tanto, **aprendió a desambiguar múltiples objetos** y mantiene un *embedding* discriminativo para cada uno — la localización depende genuinamente del sonido.

**Evaluación cuantitativa.** Se anotaron 500 clips de validación con la ubicación del instrumento que suena. *Baseline* que siempre predice el centro de la imagen: **57.2%**. AVOL-Net (la moda del *heatmap*): **81.7%**. Esto confirma que la red no se limita a resaltar el objeto saliente central. El paper nota que fue necesario anotar datos propios porque *benchmarks* estándar (PASCAL VOC, COCO, DAVIS, KITTI) no contienen instrumentos musicales, de modo que tampoco existen detectores *off-the-shelf* para anotar automáticamente.

En video (Fig. 7) cada *frame* y su audio se procesan **completamente independientes**, sin información de movimiento ni suavizado temporal, y aun así la localización es estable a lo largo del tiempo y **cambia de objeto** según el audio (por ejemplo, alterna entre habla y guitarra durante una clase de guitarra).

### 5.4. La *cautionary tale*: prevención de atajos

Esta es la cuarta contribución y una lección general para la autosupervisión. Las redes profundas son notorias por hallar atajos sutiles para "hacer trampa" (el caso clásico es la aberración cromática en Doersch et al. 2015). Aquí, el muestreo ingenuo de negativos abre un atajo: como el positivo siempre tiene el audio con punto medio alineado a un *frame* (múltiplo de 0.04 s, a 25 fps), mientras el negativo no tiene esa restricción, **existe una diferencia estadística entre el audio positivo y el negativo**. La red aprende a reconocer audios muestreados en múltiplos de 0.04 s —probablemente explotando artefactos de bajo nivel de la codificación MPEG o del *resampling*— en vez de aprender semántica.

El efecto es contraintuitivo y aleccionador: **sin** prevención del atajo, la AVE-Net logra una *accuracy* artificialmente alta de **87.6%** en AVC, pero su recuperación es **1–2% peor** de forma consistente. **Con** prevención (muestrear también el audio negativo en múltiplos de 0.04 s), la *accuracy* baja a 81.9% pero las representaciones son mejores. Es decir: **mejor desempeño en la tarea *pretext* puede significar peores representaciones reales**. El paper advierte que esto es importante para trabajos futuros donde el alineamiento exacto sea necesario (sincronización audio-visual).

### 5.5. Detalles de implementación

Entrada de imagen 224×224 color; audio resampleado a 48 kHz, log-espectrograma (ventana 0.01 s, solapamiento de media ventana), tratado como imagen 257×200. Aumentación estándar (*crop* aleatorio, *flip* horizontal, *jitter* de brillo/saturación; *jitter* de amplitud para audio). Optimizador **Adam**, *weight decay* 1e-5, *learning rate* por *grid search* con decaimiento de 6% cada 16 *epochs*. Entrenamiento en **16 GPUs** en paralelo con actualizaciones síncronas en **TensorFlow**, lote efectivo de **2048** (128 por *worker*). Cada capa convolucional va seguida de *batch normalization* y ReLU.

## 6. Limitaciones reconocidas

- **Las variantes con movimiento no ayudan a la recuperación** (§4.4): un caso concreto de que el desempeño en la tarea *pretext* no se correlaciona perfectamente con la calidad de las *features*. Es a la vez resultado y limitación: el movimiento, lejos de ayudar, ofrece un atajo de bajo nivel.
- **Localización parcial del objeto.** Al ser no supervisada, la AVOL-Net puede enfocar solo partes discriminativas en vez del objeto completo, y enfrenta la ambigüedad ontológica de *qué* hace el sonido (el cuerpo del piano, las cuerdas, el teclado, los dedos, la orquesta). Gramófonos o radios, que pueden producir sonidos arbitrarios, son casos no resueltos.
- **Dependencia de la calidad de AudioSet.** Los casos de falla se deben mayormente a los problemas del dataset descritos (videos con solo partituras o texto sobre música, fuente sonora no visible).
- **Mono por diseño.** El sistema no usa información de múltiples canales de audio (que podría ayudar a localizar) por razones deliberadas: la calibración del *rig* multi-micrófono es desconocida en videos de YouTube, el número de canales varía, la calidad varía y los métodos multi-micrófono son sensibles a ruido y reverberación. La meta es detectar conceptos semánticos, no localizar "haciendo trampa" con información de micrófonos.
- **Mejora futura propuesta por los propios autores:** dotar a la AVOL-Net de un mecanismo de **atención suave explícita** en lugar del *max-pooling* actual (atención "infinitamente dura").

## 7. Impacto y conexión con la era de la autosupervisión multimodal

*Objects that Sound* es un hito en la línea que va del par contrastivo simple hacia la autosupervisión multimodal moderna. Mostró que una sola señal gratuita —la coocurrencia temporal de imagen y sonido en video— basta no solo para aprender representaciones, sino para habilitar dos capacidades de alto nivel (recuperación y localización) que normalmente exigirían anotación costosa. Trabajos concurrentes y posteriores citados por el propio paper (Senocak et al. 2018; *The Sound of Pixels*, Zhao et al. ECCV 2018; Owens & Efros, *Audio-Visual Scene Analysis*, ECCV 2018) consolidaron esta agenda audio-visual autosupervisada.

La idea de **alinear modalidades en un espacio común mediante un cuello de botella de distancia** anticipa los métodos contrastivos cross-modal que dominarían años después (la familia CLIP en imagen-texto, ConVIRT en imagen-texto clínico). Y el patrón de **localización por correspondencia espacial sin etiquetas de localización** —tratar la representación de una modalidad como filtro de atención sobre la rejilla espacial de la otra— reaparece en *grounding* visual y en segmentación de vocabulario abierto.

## 8. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

La Clase 28 incluye el slide **"Correspondencia Audio-Visual"**, donde se afirma que "modificando un poco la arquitectura podemos saber en qué parte está el objeto que produce el sonido (Arandjelovic y Zisserman 2018)". Este paper es exactamente la fuente de esa afirmación, y permite desempacarla con precisión:

- **El "modificar un poco la arquitectura"** es literal y mínimo. Se parte de la AVE-Net (clasificación binaria de correspondencia con *embedding* global) y se hacen tres cambios quirúrgicos: no hacer *pooling* global y mantener la rejilla 14×14, convertir las FC de visión en convoluciones 1×1, y quitar la normalización para que el fondo pueda apagarse. El resto —*backbones*, tarea AVC, pérdida de correspondencia— queda igual. Esa es la AVOL-Net, y de ese cambio menor **emerge gratis la localización**.

- **Por qué encaja en autosupervisión.** Toda la potencia viene de la tarea *pretext* AVC, cuyas etiquetas se construyen del propio video (mismo instante = positivo, videos distintos = negativo). No hay etiquetas de clase, ni de ubicación, ni de identidad — el caso paradigmático de autosupervisión que la clase estudia, aplicado al setting **multimodal**.

- **La *cautionary tale* es material de clase.** §5.4 ilustra de forma memorable un principio central de la autosupervisión: el desempeño en la tarea *pretext* no es el objetivo, y un atajo de bajo nivel puede subir esa métrica mientras arruina las representaciones. Es el mismo fenómeno que motiva las aumentaciones agresivas en SimCLR/MoCo (también vistos en esta clase) para impedir que la red resuelva el *pretext* por color o por artefactos.

- **Conexión con el paper hermano.** Este trabajo es la continuación directa de *Look, Listen and Learn* (2017), que introdujo la tarea AVC y la L3-Net base. Para entender por qué la AVE-Net rediseña la fusión, conviene leer primero ese paper: ver [/papers/look-listen-learn-arandjelovic-2017](/papers/look-listen-learn-arandjelovic-2017).

Enlaces relacionados: el fundamento transversal de [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado), la [Clase 28](/clases/clase-28) que lo enmarca, y el [dominio multimodal](/dominios/multimodal) donde esta línea de trabajo vive junto a CLIP, ConVIRT y la familia de modelos visión-lenguaje.
