# U-Net: Convolutional Networks for Biomedical Image Segmentation — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- **Autores:** Olaf Ronneberger, Philipp Fischer, Thomas Brox (Computer Science Department y BIOSS Centre for Biological Signalling Studies, University of Freiburg, Alemania).
- **Venue:** MICCAI 2015 (Medical Image Computing and Computer-Assisted Intervention).
- **Año:** 2015. **Preprint:** arXiv:1505.04597v1 (18 mayo 2015), [arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597).
- **Código y modelos:** implementación basada en Caffe y redes entrenadas disponibles en `http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net`.

Este es un paper de **arquitectura**: propone una red convolucional y una estrategia de entrenamiento para **segmentación semántica** de imágenes biomédicas, es decir, asignar una etiqueta de clase a *cada pixel* de la imagen. La tesis central tiene dos partes entrelazadas. Primera: existe un consenso —que el paper cita en su frase inicial— de que entrenar redes profundas con éxito "requiere muchos miles de muestras anotadas", y en el dominio biomédico esa cantidad de imágenes anotadas "habitualmente está fuera de alcance". Segunda: una arquitectura *encoder-decoder simétrica en forma de U*, combinada con un uso intensivo de **data augmentation**, puede entrenarse de extremo a extremo desde *muy pocas* imágenes y aun así superar al estado del arte previo.

La contribución que volvió a esta red ubicua no es el camino contractivo (un convnet estándar) ni el camino expansivo por sí solos, sino las **skip connections**: conexiones que copian los mapas de características de alta resolución del encoder y los concatenan con los mapas correspondientes del decoder. Esto resuelve la tensión fundamental de la segmentación —contexto vs. localización— que el paper articula explícitamente al criticar los métodos previos.

Para la Clase 29 (Modelos Generativos en Visión) este paper importa por una razón que en 2015 nadie anticipaba: la arquitectura U-Net se convirtió en el **backbone denoiser de los modelos de difusión**. DDPM (Ho et al., 2020) y Stable Diffusion usan una U-Net para predecir el ruido en cada paso del proceso inverso. La clase tiene una sección titulada "Aprendiendo el paso inverso: U-Net" precisamente porque la red que aprende a invertir la difusión —a transformar ruido en imagen— es, estructuralmente, esta misma U. Nace para segmentar tejido neuronal y termina siendo el corazón de los generativos. Ver también [/fundamentos/modelos-de-difusion](/fundamentos/modelos-de-difusion), [/clases/clase-29](/clases/clase-29) y [/papers/ho-ddpm-2020](/papers/ho-ddpm-2020).

## 2. Contexto histórico: segmentación biomédica y el problema de los pocos datos

Hacia 2013–2015 las redes convolucionales profundas ya dominaban el reconocimiento visual: AlexNet (Krizhevsky et al., 2012) había marcado el quiebre entrenando una red de 8 capas y millones de parámetros sobre ImageNet (1 millón de imágenes), y luego vinieron redes aún más profundas como VGG (Simonyan & Zisserman, 2014). Pero el paper subraya que el uso *típico* de un convnet era la **clasificación**: una imagen entra, una etiqueta de clase sale. En muchas tareas visuales —y de modo crítico en procesamiento de imágenes biomédicas— la salida deseada debe incluir **localización**: una etiqueta por pixel. Y el dato duro: en tareas biomédicas, miles de imágenes anotadas suelen ser inalcanzables, porque anotar requiere expertos (patólogos, biólogos) y tiempo.

El antecedente directo que el paper discute en profundidad es **Ciresan et al. (2012)**, que entrenó una red en un esquema de **ventana deslizante** (*sliding-window*): para predecir la etiqueta de cada pixel, se le daba a la red la región local (un *patch*) alrededor de ese pixel como entrada. Este enfoque tenía dos virtudes —podía localizar, y el número de patches de entrenamiento era mucho mayor que el número de imágenes— y de hecho ganó el EM segmentation challenge de ISBI 2012. Pero Ronneberger señala dos defectos fundamentales:

1. **Lentitud y redundancia.** La red debe correrse por separado para cada patch, y los patches vecinos se solapan masivamente, recomputando lo mismo una y otra vez.
2. **El compromiso contexto vs. localización.** Patches grandes requieren más capas de *max-pooling*, lo que *reduce* la precisión de localización; patches pequeños permiten ver solo poco contexto. No se puede tener ambas cosas a la vez con este diseño.

El otro pilar conceptual es la **Fully Convolutional Network (FCN)** de Long, Shelhamer & Darrell (2014). El paper dice construir "sobre una arquitectura más elegante", la FCN. La idea de Long et al. era reemplazar las capas de *pooling* de un convnet contractivo por capas de *upsampling* sucesivas, aumentando así la resolución de la salida; y combinar features de alta resolución del camino contractivo con la salida upsampleada para poder localizar. U-Net toma esta idea y la lleva a su forma simétrica. La diferencia clave que Ronneberger introduce: en el camino de upsampling también hay un **gran número de canales de características**, lo que permite propagar información de contexto hacia las capas de alta resolución. Como consecuencia, el camino expansivo queda "más o menos simétrico" al contractivo, produciendo la **arquitectura en forma de U**.

## 3. Contribución central

La aportación se puede descomponer en tres ideas, de las cuales la primera es la que define la arquitectura:

1. **Encoder-decoder simétrico con skip connections.** La red tiene un *camino contractivo* (encoder, lado izquierdo de la U) que captura **contexto** mediante downsampling sucesivo, y un *camino expansivo* (decoder, lado derecho) que recupera **localización precisa** mediante upsampling sucesivo. La pieza decisiva son las skip connections: en cada nivel del decoder, el mapa de características upsampleado se **concatena con el mapa de características correspondiente del encoder** (recortado por el tema de los bordes, ver §4). Estas conexiones reinyectan detalle espacial de alta resolución —bordes, contornos finos— que el downsampling había destruido. Esto resuelve directamente el dilema contexto/localización de Ciresan: el contexto viene por el camino profundo de la U, el detalle por las skip connections laterales.

2. **Data augmentation elástica para el régimen de pocos datos.** Como hay poquísimos datos de entrenamiento, los autores aplican **deformaciones elásticas** aleatorias a las imágenes anotadas. Esto enseña a la red invarianza a deformaciones sin necesidad de ver esas transformaciones en el corpus anotado. El paper argumenta que esto es *especialmente* pertinente en segmentación biomédica, porque la deformación "solía ser la variación más común en tejido" y se puede simular de forma realista y eficiente.

3. **Weighted loss para separar objetos que se tocan.** Un desafío recurrente en segmentación de células es separar objetos de la misma clase que están en contacto. Para forzar a la red a aprender las finas fronteras de separación entre células adyacentes, los autores introducen una **pérdida ponderada** (un mapa de pesos pixel a pixel) que asigna gran peso a los pixeles de fondo que separan células que se tocan.

El paper enmarca todo esto bajo una restricción de diseño elegante: la red **no tiene capas totalmente conectadas** y usa solo la parte *válida* de cada convolución (sin padding), de modo que el mapa de segmentación solo contiene pixeles para los cuales el contexto completo está disponible en la imagen de entrada. Esto habilita la estrategia overlap-tile (§4).

## 4. Arquitectura y método

### 4.1. El camino contractivo (encoder)

El camino contractivo sigue la arquitectura típica de un convnet. Su bloque repetido consiste en: **dos convoluciones 3×3 sin padding** (*unpadded* / *valid convolutions*), cada una seguida de una **ReLU**, y luego una operación de **max-pooling 2×2 con stride 2** para el downsampling. En cada paso de downsampling se **duplica el número de canales de características**. Así, espacialmente la imagen se encoge mientras la profundidad de canales crece (la firma de un encoder convolucional): de 64 canales arriba a 1024 en el fondo de la U, en el ejemplo de la Figura 1.

### 4.2. El camino expansivo (decoder)

Cada paso del camino expansivo consiste en:

1. Un **upsampling** del mapa de características, seguido de una **convolución 2×2** —la llamada *"up-convolution"* o convolución transpuesta— que **reduce a la mitad** el número de canales de características.
2. Una **concatenación con el mapa de características correspondiente del camino contractivo, recortado** (la skip connection).
3. **Dos convoluciones 3×3**, cada una seguida de una ReLU.

El recorte (*cropping*) del mapa del encoder es *necesario* porque las convoluciones sin padding pierden pixeles de borde en cada paso, de modo que el mapa del encoder es algo más grande que el del decoder y hay que recortarlo para que las dimensiones espaciales cuadren al concatenar. En la **capa final** se usa una **convolución 1×1** para mapear cada vector de características de 64 componentes al número deseado de clases. En total, la red tiene **23 capas convolucionales**.

### 4.3. Estrategia overlap-tile para imágenes arbitrariamente grandes

Como la red usa solo convoluciones válidas, la salida es más chica que la entrada por un ancho de borde constante; el mapa de segmentación solo cubre la región donde el contexto completo está disponible. Esto permite la **estrategia overlap-tile** (Figura 2): para segmentar imágenes arbitrariamente grandes, la imagen se procesa por baldosas (*tiles*) solapadas. Para predecir los pixeles del *borde* de la imagen, el contexto faltante se **extrapola por espejado** (*mirroring*) de la entrada. Esto es crucial para aplicar la red a imágenes grandes, porque de otro modo la resolución estaría limitada por la memoria de la GPU. Para que el teselado sea sin costuras (*seamless*), el tamaño del tile de entrada debe elegirse de modo que todas las operaciones de max-pooling 2×2 se apliquen a una capa con tamaños x e y pares.

### 4.4. Entrenamiento, weighted loss e inicialización

El entrenamiento usa **SGD** (implementación de Caffe). Por las convoluciones sin padding, los autores prefieren **tiles de entrada grandes sobre batches grandes**, reduciendo el batch a *una sola imagen* y compensando con un **momentum alto (0.99)**, de modo que un gran número de muestras vistas previamente determine la actualización actual.

La función de energía es un **soft-max pixel a pixel** sobre el mapa de características final, combinado con **cross-entropy**. El soft-max es $p_k(x) = \exp(a_k(x)) / \sum_{k'} \exp(a_{k'}(x))$, donde $a_k(x)$ es la activación del canal $k$ en la posición de pixel $x$. La pérdida es:

$$E = \sum_{x \in \Omega} w(x) \, \log\big(p_{\ell(x)}(x)\big)$$

donde $\ell(x)$ es la etiqueta verdadera del pixel y, crucialmente, $w(x)$ es un **mapa de pesos** que da más importancia a ciertos pixeles. Ese mapa de pesos se precomputa para (a) compensar la frecuencia desigual de clases y (b) forzar a la red a aprender las pequeñas fronteras de separación entre células que se tocan:

$$w(x) = w_c(x) + w_0 \cdot \exp\!\left(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\right)$$

donde $w_c$ balancea las frecuencias de clase, $d_1$ es la distancia al borde de la célula más cercana y $d_2$ la distancia al borde de la segunda célula más cercana. En sus experimentos usan $w_0 = 10$ y $\sigma \approx 5$ pixeles. El efecto es un "valle" de peso alto justo en la delgada franja de fondo entre dos células adyacentes (Figura 3d), lo que empuja a la red a no fusionar instancias vecinas.

La **inicialización de pesos** se cuida especialmente: en una red profunda con muchas capas y distintos caminos, una mala inicialización haría que partes de la red den activaciones excesivas mientras otras nunca contribuyen. Idealmente cada mapa de características debe tener varianza ~unitaria, lo que se logra muestreando los pesos iniciales de una gaussiana con desviación estándar $\sqrt{2/N}$ (inicialización de He et al., 2015), donde $N$ es el número de nodos entrantes a una neurona (p. ej. para una convolución 3×3 con 64 canales previos, $N = 9 \cdot 64 = 576$).

### 4.5. Data augmentation en detalle

La augmentation es *esencial* en el régimen de pocas muestras. Para imágenes microscópicas se necesita sobre todo invarianza a desplazamiento y rotación, más robustez a deformaciones y variaciones de valor de gris. Las **deformaciones elásticas aleatorias** son "el concepto clave" para entrenar una red de segmentación con muy pocas imágenes anotadas: se generan deformaciones suaves usando vectores de desplazamiento aleatorios sobre una grilla gruesa de 3×3, muestreados de una gaussiana con desviación estándar de 10 pixeles, y los desplazamientos por pixel se interpolan con bicúbica. Las capas de **dropout** al final del camino contractivo realizan augmentation implícita adicional.

## 5. Experimentos

El paper demuestra la U-Net en tres tareas de segmentación.

**EM segmentation (estructuras neuronales en microscopía electrónica).** Datos del EM segmentation challenge iniciado en ISBI 2012: 30 imágenes (512×512) de microscopía electrónica de transmisión de secciones seriadas del cordón nervioso ventral de larva de *Drosophila*, cada una con su mapa de segmentación anotado (células en blanco, membranas en negro). El test es público pero sus mapas son secretos; la evaluación la hacen los organizadores midiendo *warping error*, *Rand error* y *pixel error*. La U-Net (promediada sobre 7 versiones rotadas de la entrada), **sin pre- ni post-procesamiento**, logra un warping error de **0.000353** (el nuevo mejor puntaje de la tabla) y un Rand error de 0.0382. Esto supera significativamente a la sliding-window de Ciresan et al. (warping error 0.000420, Rand error 0.0504). Los únicos algoritmos con mejor Rand error usaban post-procesamiento altamente específico del dataset (uno de ellos sometió 78 soluciones distintas para lograr su resultado).

**ISBI cell tracking challenge 2015 (microscopía de luz transmitida).** En dos datasets 2D:

- **"PhC-U373"**: células de glioblastoma-astrocitoma U373 en microscopía de contraste de fase, con 35 imágenes parcialmente anotadas. La U-Net logra un **IoU promedio de 92%**, contra 83% del segundo mejor algoritmo.
- **"DIC-HeLa"**: células HeLa sobre vidrio plano en microscopía de contraste por interferencia diferencial (DIC), con 20 imágenes parcialmente anotadas. La U-Net logra un **IoU de 77.5%**, contra apenas 46% del segundo mejor.

Estos márgenes —especialmente 77.5% vs. 46% en DIC-HeLa— son enormes para un benchmark de challenge, y se obtienen entrenando desde decenas de imágenes. El paper también destaca la **velocidad**: segmentar una imagen 512×512 toma menos de un segundo en una GPU moderna, y el entrenamiento completo es de apenas ~10 horas en una NVidia Titan (6 GB), gracias a que el overlap-tile elimina la redundancia masiva del sliding-window.

## 6. Limitaciones

El paper es breve y triunfalista en tono, pero se pueden leer límites:

- **Las convoluciones sin padding complican el diseño.** Obligan al recorte de las skip connections, a elegir cuidadosamente el tamaño de tile (para que los pooling caigan en dimensiones pares) y al espejado de bordes en overlap-tile. Variantes posteriores de U-Net suelen usar *same padding* para simplificar, a costa de algún artefacto de borde.
- **Segmentación semántica, no de instancias nativa.** La red produce un mapa de probabilidad por clase; la separación de instancias que se tocan se logra con el truco de la weighted loss sobre las fronteras, no con un mecanismo de instancias propiamente dicho.
- **Batch de tamaño 1 y momentum 0.99.** Es una elección forzada por la memoria (tiles grandes), no necesariamente óptima; hoy se entrenaría distinto.
- **Validación acotada al dominio biomédico microscópico** y a datasets pequeños; el paper no caracteriza el comportamiento en escenarios con abundantes datos ni fuera de microscopía. (Irónicamente, el tiempo demostraría que la arquitectura generaliza muchísimo más allá de lo que el paper se atrevió a reclamar.)

## 7. Impacto: de la segmentación médica al corazón de los generativos

La U-Net se volvió **ubicua**. En su nicho original —segmentación de imágenes médicas— es prácticamente el estándar de facto: resonancia magnética, tomografía, histopatología, segmentación de órganos y lesiones, etc. Para Roberto, que trabaja en salud, este es el punto de contacto más inmediato: cualquier pipeline moderno de segmentación de imagen médica casi seguro tiene una U-Net (o un descendiente como nnU-Net, Attention U-Net, U-Net++ o TransUNet) en su núcleo. La idea de skip connections encoder→decoder migró además a innumerables arquitecturas de visión densa.

Pero el giro que conecta este paper con la **Clase 29** es otro y es profundo. Los **modelos de difusión** —DDPM (Ho, Jain & Abbeel, 2020), y luego Stable Diffusion y compañía— funcionan aprendiendo a *revertir* un proceso que añade ruido gaussiano progresivamente a una imagen. La red que aprende ese paso inverso —la que, dado un dato ruidoso $x_t$ y un paso de tiempo $t$, **predice el ruido** $\epsilon_\theta(x_t, t)$ que hay que quitar— es una **U-Net**. ¿Por qué la U-Net y no cualquier otra red? Porque la tarea de denoising es, estructuralmente, una tarea *imagen-a-imagen densa*: la salida tiene la misma resolución espacial que la entrada y cada pixel de salida depende tanto del contexto global (qué objeto se está formando) como del detalle local de alta resolución (la textura exacta a reconstruir). Esa es *exactamente* la tensión contexto/localización que las skip connections de la U-Net resuelven: el camino contractivo captura el contexto semántico del paso de difusión, y las skip connections reinyectan el detalle fino necesario para predecir el ruido pixel a pixel.

La U-Net de difusión se enriquece respecto a la original de 2015 —se le añaden bloques residuales, *group normalization*, capas de **atención** en las resoluciones bajas, y un *time embedding* que inyecta el paso $t$ en cada bloque— pero el esqueleto es inconfundiblemente la U: encoder que contrae, decoder que expande, skip connections que cosen ambos lados. Por eso la clase rotula la sección "Aprendiendo el paso inverso: U-Net": la red que invierte la difusión es, en su columna vertebral, la misma red que en 2015 segmentaba membranas neuronales de *Drosophila*. Un paper de MICCAI sobre microscopía de células terminó siendo el motor de la generación de imágenes a partir de texto. Ver [/fundamentos/modelos-de-difusion](/fundamentos/modelos-de-difusion) y el análisis de [/papers/ho-ddpm-2020](/papers/ho-ddpm-2020).

## 8. Conexión con la Clase 29 (Modelos Generativos en Visión)

La Clase 29 trata los modelos generativos de visión, con foco en difusión. El recorrido conceptual es: proceso *forward* (añadir ruido gaussiano paso a paso hasta destruir la imagen) y proceso *reverse* (aprender a quitarlo). El paso inverso necesita una red que mapee una imagen ruidosa a una predicción del ruido, manteniendo resolución espacial completa — y esa red es la U-Net de este paper, adaptada. Estudiar a Ronneberger et al. (2015) antes de DDPM permite entender *por qué* el denoiser tiene la forma que tiene, en vez de tomarlo como una caja negra: las skip connections no son un detalle de implementación, son la razón de que la red pueda reconstruir detalle fino mientras razona sobre contexto global. La lección transversal para el dominio de visión del curso es que una buena *primitiva arquitectónica* —aquí, el encoder-decoder simétrico con skips— trasciende la tarea para la que fue inventada: nace en segmentación supervisada con pocos datos y reaparece, una década después, como el componente central del paradigma generativo dominante.
