# DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort*.
- **Autores:** Yuxuan Zhang (University of Waterloo / NVIDIA), Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche (NVIDIA / University of Toronto / Vector Institute), Adela Barriuso, Antonio Torralba (MIT), Sanja Fidler (NVIDIA / University of Toronto / Vector Institute).
- **Venue:** CVPR 2021.
- **Preprint:** arXiv:2104.06490v2 (20 abr 2021), [arxiv.org/abs/2104.06490](https://arxiv.org/abs/2104.06490).

La tesis del paper se puede enunciar en una frase: **un GAN que aprendió a sintetizar imágenes realistas necesariamente adquirió conocimiento semántico sobre la estructura de los objetos, y ese conocimiento vive en sus *feature maps* — basta con leerlo**. A partir de esa observación, DatasetGAN convierte un StyleGAN pre-entrenado en una **fábrica de datos etiquetados**: se anotan a mano unas *poquísimas* imágenes generadas por el GAN (16 a 40, según la clase), se entrena un pequeño intérprete sobre las features internas del GAN para predecir la segmentación pixel a pixel, y a partir de ahí el GAN —ahora con una rama extra de etiquetas— produce un número infinito de pares imagen-anotación sintéticos con los que se entrena cualquier arquitectura de visión por computadora.

El problema que ataca es uno de los cuellos de botella más caros del *deep learning* en visión: **etiquetar segmentación es brutalmente laborioso**. El paper cuantifica que anotar una escena compleja de 50 objetos toma entre 30 y 90 minutos, y que crowdsourcear las 10.000 imágenes etiquetadas que usan en sus experimentos al nivel de detalle requerido tomaría más de **3.200 horas (134 días)** — y produciría anotaciones ruidosas, porque etiquetar a ese nivel de detalle exige tanto destreza como paciencia. DatasetGAN reduce ese costo a anotar a mano un puñado de imágenes (en su caso, ~5 horas de trabajo de un solo anotador experto por dataset).

El resultado experimental es contundente: en segmentación de partes, DatasetGAN **supera a todas las líneas base semi-supervisadas por un margen amplio** y queda a la par de métodos totalmente supervisados que, en algunos casos, usan **dos órdenes de magnitud más datos anotados** (hasta 100× más). En la tabla principal, sobre ADE-Car-12, supera al baseline de Transfer-Learning por 20,79 puntos de mIoU y al semi-supervisado por 16,96 puntos.

Para la Clase 29 (Modelos Generativos en Visión) este paper importa de manera directa: la clase lo cita **explícitamente** dentro de "Usos en la industria" como ejemplo de *data augmentation* generativa — la idea de entrenar un clasificador con pocos datos generando datos sintéticos de alta calidad. Es un caso paradigmático de cómo un GAN deja de ser un generador de imágenes bonitas para volverse infraestructura útil aguas abajo.

## 2. Contexto: por qué etiquetar es el cuello de botella y qué intentos previos hubo

Las redes profundas modernas son extremadamente *hambrientas de datos*: se benefician de entrenarse sobre datasets a gran escala, que son justamente los que más cuesta anotar. Para tareas densas — segmentación semántica, segmentación de instancias, keypoints — el costo no es elegir una etiqueta por imagen sino **etiquetar cada píxel**, lo que multiplica el esfuerzo. Datasets como MS-COCO, ADE20K o Cityscapes son tan grandes que ni siquiera es factible que un humano revise cada imagen de entrenamiento individualmente. Este es el problema de fondo.

La comunidad había explorado varias salidas, y el paper se posiciona frente a cada una:

- **Aprendizaje semi-supervisado.** Aprovecha un gran conjunto *no etiquetado* además del pequeño conjunto etiquetado, vía *pseudo-labels* y *consistency regularization*. Funciona bien en clasificación y se ha extendido a segmentación, pero entrena el modelo de segmentación directamente; no explota el modelado *generativo* de las imágenes, como sí hace DatasetGAN.
- **Aprendizaje contrastivo.** Entrena extractores de features con pérdidas auto-supervisadas sobre pares de imágenes o de parches; luego basta un pequeño subconjunto etiquetado para ajustar predictores. Comparte con DatasetGAN la idea de *amortizar* la necesidad de etiquetas vía buenas representaciones, pero usa pérdidas contrastivas en vez de extraer el conocimiento semántico de los *feature maps* de un GAN.
- **Síntesis de datasets con gráficos / GANs previos.** Trabajo anterior generaba escenas 3D con motores gráficos, o usaba *image-to-image translation* para llevar un dataset etiquetado a otro dominio (adaptación de dominio). Estos métodos *asumen la existencia de un gran dominio etiquetado* que trasladar. DatasetGAN solo necesita un puñado de imágenes anotadas a mano y sintetiza un conjunto mucho mayor.

El paper también dialoga con trabajo concurrente que igualmente traduce features de GAN a segmentación (Galeev et al., 2020; el trabajo paralelo de Li et al., 2021 sobre *semantic segmentation with generative models*), del que se diferencia por el **interpretador**: mientras otros usan decodificadores con bloques convolucionales y residuales para proyectar las capas internas de StyleGAN a un mapa de segmentación, DatasetGAN interpreta *directamente* el vector de features desacoplado de cada píxel con un simple *ensemble* de clasificadores MLP, lo que — argumentan — aprovecha mejor el conocimiento semántico de StyleGAN. La afirmación de novedad del paper: es el primer trabajo que usa GANs para sintetizar directamente un gran dataset de imágenes anotadas a alto nivel de detalle.

## 3. Contribución central: el GAN ya "sabe", solo hay que decodificarlo

La intuición que organiza todo el paper es la siguiente. Un modelo generativo entrenado para sintetizar imágenes *altamente* realistas debe haber adquirido conocimiento semántico en su espacio latente de alta dimensión — de lo contrario no podría renderizar de manera coherente las distintas partes de un objeto. En arquitecturas como StyleGAN, el código latente contiene dimensiones **desacopladas** (*disentangled*) que controlan propiedades 3D como el punto de vista y la identidad del objeto; interpolar entre dos códigos latentes produce generaciones realistas, lo que indica que el GAN aprendió a *alinear semántica y geométricamente* los objetos y sus partes.

De ahí la consecuencia operativa: **si un humano provee una etiqueta correspondiente a un código latente, esa etiqueta debería poder propagarse efectivamente a través de todo el espacio latente del GAN**. En vez de etiquetar miles de imágenes independientes, basta etiquetar unas pocas y aprender la función que mapea features internas → etiqueta de píxel; esa función generaliza al resto del espacio latente.

La contribución concreta tiene tres movimientos:

1. **Sintetizar y anotar muy poco.** Se generan pocas imágenes con StyleGAN (16 a 40) y se registran sus *feature maps* latentes. Un anotador humano las etiqueta a alto detalle.
2. **Entrenar un "Style Interpreter".** Un *ensemble* de pequeños MLP se entrena sobre los vectores de features pixel a pixel de StyleGAN para reproducir la etiqueta humana. Esto es DatasetGAN: el método es "extremadamente simple, a la vez que extremadamente potente".
3. **Generar infinitos pares etiquetados.** Una vez entrenado, el intérprete actúa como una **rama de generación de etiquetas** dentro de la arquitectura de StyleGAN. Muestreando códigos latentes $z$ y pasándolos por toda la arquitectura se obtiene un **generador de dataset infinito**: cada imagen sintética viene con su anotación pixel a pixel.

El flujo completo es de cuatro pasos (Figura 1 del paper): (1) anotar a mano un puñado de imágenes sintetizadas por StyleGAN; (2) entrenar el Style Interpreter como rama de etiquetas; (3) generar automáticamente un dataset sintético enorme de pares imagen-anotación; (4) entrenar tu arquitectura favorita sobre el dataset sintético y evaluarla en **imágenes reales**.

## 4. Método: feature maps de StyleGAN, Style Interpreter y ensemble

### 4.1. StyleGAN como motor de "renderizado"

DatasetGAN usa StyleGAN como *backbone* generativo por su calidad de síntesis. Brevemente: el generador mapea un código latente $z \in Z$ (muestreado de una normal) a un código latente intermedio $w \in W$ mediante una red de mapeo; $w$ se transforma luego en $k$ vectores $w_1, \dots, w_k$ vía transformaciones afines aprendidas. Estos códigos transformados se inyectan como información de estilo en $k/2$ bloques de síntesis de forma progresiva. Cada bloque consiste en una capa de *upsampling* (×2) y dos capas convolucionales, y cada convolución va seguida de una capa de **normalización de instancia adaptativa (AdaIN)** controlada por su correspondiente $w_i$. Las features de salida de las $k$ capas AdaIN se denotan $\{S^0, S^1, \dots, S^k\}$.

La metáfora clave del paper: **interpretar StyleGAN como un motor de "renderizado" y sus códigos latentes como atributos de "gráficos" que definen qué renderizar**. La hipótesis derivada: el arreglo aplanado de features que produce un píxel RGB particular contiene información semánticamente significativa para renderizar ese píxel de forma realista — y por tanto, para etiquetarlo.

### 4.2. Style Interpreter

El mecanismo es directo. Se hace *upsampling* de todos los feature maps AdaIN $\{S^0, \dots, S^k\}$ a la resolución de salida más alta y se concatenan en un tensor 3D $S^* = (S^{0,*}, \dots, S^{k,*})$. Así, **cada píxel $i$ tiene su propio vector de features** $S_i^* = (S_i^{0,*}, \dots, S_i^{k,*})$. Estos vectores son de alta dimensionalidad: **5056 dimensiones** por píxel. Sobre cada vector de features se aplica un **MLP de tres capas** (fully-connected → ReLU → BatchNorm, repetido) que predice la etiqueta del píxel. Los pesos se comparten entre todos los píxeles, por simplicidad.

Un punto conceptual importante: el objetivo del entrenamiento es el clasificador de features; la imagen sintetizada solo se usa para *recolectar* la anotación del humano. **No se retropropagan gradientes al backbone de StyleGAN** — el GAN queda congelado.

**Pérdidas según la tarea.** Para segmentación semántica se entrena el clasificador con *cross-entropy*. Para predicción de keypoints se construye un *heatmap* gaussiano por keypoint y se ajusta el valor de calor por píxel (L2). Como los vectores de features son de alta dimensión (5056) y los mapas tienen alta resolución espacial (hasta 1024), no se pueden consumir todos los vectores de un batch: se hace **muestreo aleatorio** de vectores de features de cada imagen, garantizando muestrear al menos una vez de cada región etiquetada.

### 4.3. El ensemble y el denoising por incertidumbre

Para amortizar el efecto del muestreo aleatorio se entrena un **ensemble de $N=10$ clasificadores**. En test, para segmentación se usa **votación por mayoría** por píxel; para keypoints se promedian los $N$ heatmaps.

El ensemble cumple además un segundo rol — **medir incertidumbre para filtrar ruido**. StyleGAN falla ocasionalmente, introduciendo ruido en el dataset sintético. El paper reporta que el score del discriminador de StyleGAN *no* es una medida robusta de fallo; en cambio, usar el desacuerdo del ensemble sí lo es. Siguiendo a Kuo et al. (2018), usan la **divergencia de Jensen-Shannon (JS)** como medida de incertidumbre por píxel, la suman sobre todos los píxeles para obtener la incertidumbre de la imagen, y **filtran el 10% de imágenes más inciertas**. El ablation (Tabla 4) muestra que este *denoising* es importante: subir mIoU de 44,60 (sin filtrar) a 45,64 (filtrando 10%); pero filtrar demasiado reduce diversidad, así que hay un *trade-off*.

### 4.4. DatasetGAN como fábrica de datos

Entrenado el Style Interpreter, se convierte en rama de síntesis de etiquetas sobre el backbone de StyleGAN, formando DatasetGAN. Sintetizar un par imagen-anotación requiere un *forward pass* por StyleGAN, que toma ~9 segundos en promedio. El rendimiento aguas abajo sigue mejorando levemente con cada 10k de imágenes sintetizadas, pero con costo asociado; **usan 10k imágenes para la mayoría de experimentos** (el ablation de Tabla 3 muestra que el rendimiento se satura lentamente: 43,34 con 3k → 45,04 con 20k).

## 5. Experimentos: segmentación de partes, keypoints y aplicación 3D

El esfuerzo de anotación manual fue tiny en términos absolutos. Los datasets de entrenamiento (imágenes generadas por el GAN) fueron anotados por **un único anotador experimentado** con LabelMe: 16 autos (605 polígonos), 16 cabezas (950 polígonos), 30 aves (443 polígonos), 30 gatos (737 polígonos) y 40 dormitorios (1109 polígonos). Como dato curioso, el propio paper nota que **hay más etiquetas en una sola imagen que imágenes en el dataset**.

### 5.1. Segmentación de partes

Se evaluó sobre cinco categorías — Car, Face, Bird, Cat, Bedroom. La red de segmentación es **DeepLab-V3 con backbone ResNet pre-entrenado en ImageNet**, igual para todos los baselines (solo cambian los datos de entrenamiento y el algoritmo). Los baselines:

- **Transfer-Learning (TL):** inicializa con pesos pre-entrenados en segmentación de MS-COCO y hace *finetune* de la última capa sobre el pequeño conjunto anotado.
- **Semi-Supervised** (Mittal et al., 2019): mismo backbone, entrenado sobre las imágenes humanas etiquetadas más las imágenes reales no etiquetadas con que se entrenó StyleGAN.

Resultados (Tabla 1, mIoU): DatasetGAN gana en **todas** las clases por un margen amplio. Sobre **ADE-Car-12**: TL 24,85 / Semi-Sup 28,68 / **Ours 45,64** — supera a los baselines fuera-de-dominio por 20,79 y 16,96 puntos respectivamente, y a las versiones in-domain por 15,93 y 10,82. Sobre Face-34: 45,77 / 48,17 / **53,46**. Importante: DatasetGAN opera en el setting **fuera de dominio** (entrena solo en su dataset sintético y evalúa en imágenes reales), lo que hace los resultados aún más notables.

Comparado con el **fully supervised** (Figura 6 y Tabla 5): DatasetGAN con **25 anotaciones** es comparable al DeepLab-V3 supervisado entrenado con las **2.600 imágenes** completas de ADE-Car-12 — menos del 1% de las etiquetas. En PASCAL-Car-5, evaluando ambos fuera de dominio, supera al baseline por 1,3 puntos, mostrando mejor generalización.

**Ablations adicionales.** Selección de datos a etiquetar (Tabla 6): en régimen de pocos datos, *qué* imágenes etiquetar importa; tanto la selección manual (un experto elige las más representativas) como el *active learning* basado en ensemble + coreset superan a la selección aleatoria. Como AL exige re-etiquetar en cada corrida, el paper adopta la selección manual para el resto.

### 5.2. Detección de keypoints

Para mostrar generalidad, se evalúa keypoints en Car y Bird (prediciendo heatmaps, L2 en vez de cross-entropy). DatasetGAN supera significativamente al baseline de Transfer-Learning con el mismo presupuesto de anotación (Tabla 2): en Car-20, PCK th-15 de 43,54 (TL) → **79,91** (Ours); en CUB-Bird se acerca al fully supervised.

### 5.3. Aplicación 3D: assets animables

Como demostración aguas abajo, el paper reconstruye **assets 3D animables a partir de imágenes monoculares** de autos: usan StyleGAN para generar imágenes multi-vista, el Style Interpreter para generar etiquetas de partes y keypoints, y entrenan una red de *inverse graphics* con *differentiable rendering*. El mapa de partes 3D permite post-procesar el modelo (parabrisas transparentes, luces emisivas, ruedas riggeadas) para que los autos tengan física y se puedan "manejar" virtualmente — el primer resultado de su tipo, habilitado por las etiquetas detalladas que produce el método.

## 6. Limitaciones reconocidas

- **La calidad del GAN limita la calidad de las etiquetas.** Como el método depende de etiquetar imágenes del GAN, cuando el GAN sintetiza mal, la anotación se complica y se degrada. El paper reporta que el anotador "se quejó" al etiquetar aves: las patas sintetizadas eran en su mayoría invisibles, borrosas y poco naturales, lo que hizo que los datasets sintéticos apenas generaran etiquetas de patas — y eso degrada el rendimiento en esa parte específica al evaluar.
- **Errores en partes finas o sin bordes claros.** Los fallos típicos ocurren en estructuras delgadas (arrugas faciales, patas de ave, bigotes de gato) o partes sin frontera visual clara (cuello del gato).
- **Costo de generación.** Cada par imagen-anotación cuesta ~9 s de forward pass; generar datasets grandes tiene costo computacional, con retornos decrecientes más allá de ~10k imágenes.
- **Dependencia de un StyleGAN por categoría.** Cada clase requiere un StyleGAN pre-entrenado específico de la categoría (para aves entrenaron el suyo sobre NABirds, 48k imágenes). El método no es directamente *open-set*: extender DatasetGAN a un conjunto grande y diverso de clases queda como trabajo futuro declarado.
- **Calidad ≠ cantidad de partes.** En clases más difíciles (aves, gatos) el GAN produce menos partes anotables que las imágenes reales, aunque la cantidad de detalle disponible sigue siendo notable.

## 7. Impacto: data augmentation generativa

DatasetGAN reencuadra para qué sirve un GAN: no como fin (generar imágenes) sino como **infraestructura para fabricar datos etiquetados** — *data augmentation generativa* en su forma más ambiciosa, donde lo aumentado no son solo las imágenes sino sus **etiquetas densas**. La conclusión del paper es que es "solo el primer paso" hacia un entrenamiento más eficiente de redes profundas, y abre la puerta a aplicaciones aguas abajo antes imposibles (como la reconstrucción 3D animable).

La lección transferible: cuando un modelo generativo potente ya capturó la estructura de un dominio en sus representaciones internas, etiquetar masivamente desde cero es redundante — conviene *decodificar* lo que el modelo ya sabe a partir de poquísimas anotaciones humanas. Esta idea reaparece después con modelos de difusión y con *foundation models* de segmentación, pero DatasetGAN la cristaliza tempranamente y con un mecanismo desarmantemente simple (un MLP por píxel sobre features de StyleGAN).

## 8. Conexión con la Clase 29 (Modelos Generativos en Visión)

La Clase 29 cita DatasetGAN **explícitamente** en la sección "Usos en la industria", como el ejemplo canónico de **data augmentation generativa**: entrenar un clasificador (o segmentador) con pocos datos, generando datos sintéticos de alta calidad para suplir la escasez de etiquetas. El paper materializa esa idea de punta a punta — del GAN pre-entrenado al dataset sintético etiquetado al modelo final evaluado en datos reales.

La conexión con el resto del temario del curso de generativos es directa:

- Se apoya sobre **StyleGAN** ([Karras et al., 2019](/papers/stylegan-karras-2019)) como backbone — su síntesis de alta calidad y, sobre todo, su **espacio latente desacoplado** son lo que hace que las features internas contengan conocimiento semántico explotable. Sin la calidad y el *disentanglement* de StyleGAN, el Style Interpreter no tendría señal que decodificar.
- Es heredero directo del marco adversarial de **GANs** ([Goodfellow et al., 2014](/papers/goodfellow-gan-2014)): toda la potencia generativa que DatasetGAN reutiliza viene del entrenamiento adversarial que aprendió a renderizar objetos realistas y, con ello, su estructura.
- Encaja en el panorama de los [fundamentos de modelos generativos](/fundamentos/modelos-generativos): ilustra que el valor de un generador no se agota en muestrear imágenes, sino que sus *representaciones intermedias* son por sí mismas un recurso reutilizable.

**Por qué es especialmente relevante para el contexto de Roberto (salud, pocos datos etiquetados).** El escenario clínico es el caso de uso natural de DatasetGAN: etiquetar imágenes médicas (segmentación de lesiones, órganos, estructuras) es carísimo, requiere expertos escasos (radiólogos, patólogos) y produce datasets pequeños. El patrón "entrenar un generador de la modalidad → anotar a mano un puñado de casos → decodificar el conocimiento del generador a etiquetas densas → fabricar un dataset sintético grande" es exactamente la palanca que el método propone. Las mismas limitaciones aplican y se vuelven críticas en salud: la calidad de las etiquetas hereda la del generador, las estructuras finas o de bordes difusos (que abundan en imagen médica) son las más propensas a error, y se necesita un generador por dominio/modalidad. Pero la economía del etiquetado que habilita — pasar de miles de anotaciones a decenas — es precisamente la que más duele en el dominio clínico.

Ver también: [Clase 29 — Modelos Generativos en Visión](/clases/clase-29).
