# Context Encoders: Feature Learning by Inpainting — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Context Encoders: Feature Learning by Inpainting*.
- **Autores:** Deepak Pathak, Philipp Krähenbühl, Jeff Donahue, Trevor Darrell, Alexei A. Efros — todos de University of California, Berkeley.
- **Venue:** CVPR 2016 (Conference on Computer Vision and Pattern Recognition).
- **Preprint:** arXiv:1604.07379v2 (21 nov 2016), [arxiv.org/abs/1604.07379](https://arxiv.org/abs/1604.07379).
- **Código / modelos:** publicados en el sitio del proyecto de los autores (modelos entrenados, resultados de inpainting adicionales).

Este es uno de los papers fundacionales del **aprendizaje autosupervisado en visión por computadora**. Su tesis cabe en una frase: si entrenamos una red convolucional para **rellenar (inpaint) una región faltante de una imagen condicionada en sus alrededores**, la red se ve forzada a "entender el contenido de la imagen completa, además de producir una hipótesis plausible para las partes faltantes". El gesto clave es que ese objetivo —predecir píxeles— **no necesita ninguna etiqueta humana**: la supervisión sale gratis de la propia imagen, recortando una porción y pidiéndole a la red que la reconstruya. Los autores bautizan el modelo **context encoder** por analogía con los autoencoders: una red convolucional encoder-decoder donde el encoder comprime el contexto (la imagen con el agujero) en una representación latente compacta y el decoder produce el contenido faltante.

La aportación tiene dos caras que el paper evalúa por separado. En el lado del **decoder**, los context encoders son —según los autores— "el primer algoritmo de inpainting paramétrico capaz de dar resultados razonables para hole-filling semántico" (es decir, regiones faltantes grandes que los métodos clásicos de inpainting o síntesis de textura no pueden manejar). En el lado del **encoder**, las características aprendidas sin etiquetas transfieren competitivamente a tareas de clasificación, detección y segmentación en PASCAL VOC, situándose a la par de los demás métodos auto/débilmente supervisados contemporáneos.

El segundo gran aporte técnico es la **función de pérdida combinada**: un término de reconstrucción L2 (que captura la estructura global de la región faltante pero produce resultados borrosos al promediar los modos posibles) sumado a un término adversarial estilo GAN (que escoge un modo particular de la distribución y produce predicciones nítidas). Esta combinación —reconstrucción para la coherencia, adversarial para el detalle— es una receta que reaparecería una y otra vez en la literatura posterior de generación de imágenes.

Para la **Clase 28 (Aprendizaje Autosupervisado)** importa porque la clase lo presenta como el ejemplo canónico de **auto-predicción en imágenes**: inpainting como *pretext task*. Y la clase muestra precisamente la tabla de transferencia a PASCAL VOC 2007 que se discute en la sección 7: inpainting/context 56.5%, inicialización aleatoria 53.3%, preentrenamiento ImageNet supervisado 79.9%. Entender este paper es entender por qué predecir píxeles ausentes enseña semántica.

## 2. Contexto histórico: el aprendizaje autosupervisado temprano y las pretext tasks

Hacia 2014-2016 la visión por computadora vivía bajo el dominio del preentrenamiento supervisado en ImageNet. La receta era conocida: entrenar una CNN (típicamente AlexNet) sobre el millón largo de imágenes etiquetadas de ImageNet, y luego reutilizar esas características —que "generalizan muy bien a través de tareas" (Donahue et al., DeCAF, 2014)— como inicialización para detección, segmentación o clasificación en datasets más pequeños. El problema, que el paper enuncia con precisión, es que **seguía abierta la pregunta de si características igual de informativas y generalizables pueden aprenderse de imágenes crudas, sin etiqueta alguna**.

La respuesta a esa pregunta es lo que hoy llamamos **aprendizaje autosupervisado** (self-supervised learning): diseñar una tarea artificial —un *pretext task*— cuya supervisión se deriva automáticamente de la estructura de los datos, de modo que resolverla obligue a la red a aprender representaciones útiles. El paper sitúa varias familias de pretext tasks que estaban emergiendo en paralelo:

- **Señal temporal en video.** La consistencia entre cuadros temporales sirve como supervisión (Goroshin et al., 2015; Ramanathan et al., 2015). Wang & Gupta (2015) rastrean *patches* a través de cuadros de video y usan la coherencia del *tracking* para guiar el entrenamiento.
- **Ego-movimiento.** Agrawal et al. (2015, "Learning to see by moving") y Jayaraman & Grauman (2015) usan el movimiento propio leído de sensores no visuales (odometría) como señal supervisoria para entrenar características visuales.
- **Contexto espacial.** La familia más cercana a este paper. El **Visual Memex** (Malisiewicz & Efros, 2009) modelaba relaciones entre objetos de forma no paramétrica usando contexto, y Doersch et al. (2014) usaron contexto para descubrir objetos sin supervisión —pero ambos dependían de características diseñadas a mano, sin aprendizaje de representación.

El trabajo más directamente comparable y contemporáneo es **Doersch et al. (2015), "Unsupervised visual representation learning by context prediction"**, que entrena una red a predecir la *posición relativa* de dos *patches* vecinos dentro de una imagen ("¿el patch A está arriba o abajo del patch B?"). El paper de context encoders se diferencia de Doersch en algo conceptual y profundo: **Doersch resuelve una tarea discriminativa** (clasificar entre 8 posiciones relativas), mientras que **el context encoder resuelve un problema de predicción puro** (¿qué intensidades de píxel deben ir en el agujero?). Los autores trazan un paralelo lingüístico explícito: en aprendizaje de *word embeddings*, Collobert & Weston (2008) abogan por un enfoque discriminativo, mientras que **word2vec** (Mikolov et al., 2013) lo formula como predicción de palabra dado su contexto. El context encoder es, en este sentido, el "word2vec de los píxeles".

Esa distinción tiene tres consecuencias prácticas que el paper destaca:

1. **Señal supervisoria mucho más rica.** Un context encoder debe predecir alrededor de **15.000 valores reales por ejemplo de entrenamiento**, frente a "1 opción entre 8 alternativas" en Doersch. La densidad de la señal es órdenes de magnitud mayor.
2. **Entrenamiento más rápido.** En parte por esa señal más densa, los context encoders tardan mucho menos en entrenar (14 horas en una Titan X, frente a las 4 semanas que reportaba Doersch).
3. **Más difícil de "hacer trampa".** En Doersch, características de bajo nivel como la aberración cromática resuelven parcialmente la tarea sin entender nada semántico. La predicción basada en contexto no se deja engañar por esos atajos.

## 3. Contribución central: inpainting como pretext task

La idea central es elegante: dada una imagen con una región removida (Figura 1a), entrenar una CNN para que **regrese a los valores de los píxeles faltantes** (Figura 1d). Los autores llaman al modelo *context encoder* porque consta de un encoder que captura el contexto de la imagen en una representación latente compacta y un decoder que usa esa representación para producir el contenido faltante.

El context encoder está "íntimamente relacionado con los autoencoders", con los que comparte la arquitectura encoder-decoder. Pero el paper marca con cuidado las diferencias, porque ahí reside todo el aporte:

- Un **autoencoder** toma una imagen, la pasa por un cuello de botella de baja dimensión, e intenta reconstruirla idéntica. El problema es que esa representación "probablemente solo comprime el contenido de la imagen sin aprender una representación semánticamente significativa" — el autoencoder puede copiar píxeles sin entender nada.
- Un **denoising autoencoder** (Vincent et al., 2008) aborda esto corrompiendo la entrada y pidiéndole a la red deshacer el daño. Pero esa corrupción es típicamente "muy localizada y de bajo nivel, y no requiere mucha información semántica para deshacerse".
- El **context encoder** resuelve una tarea mucho más dura: rellenar áreas faltantes *grandes* de la imagen, donde no puede obtener "pistas" de píxeles cercanos. Esto exige un entendimiento semántico mucho más profundo de la escena y la capacidad de sintetizar características de alto nivel sobre grandes extensiones espaciales. El ejemplo del paper: para inpaint la fachada de la Figura 1a, "una ventana entera tiene que ser conjurada de la nada".

Visto bajo otra luz, un context encoder es **un denoising autoencoder donde la corrupción es espacialmente mucho mayor**, lo bastante grande como para requerir información semántica para deshacerse. Esa es la palanca que convierte una tarea de reconstrucción de píxeles en una tarea de aprendizaje de representación.

El paper subraya además que esta tarea es **inherentemente multimodal**: hay múltiples maneras igualmente plausibles de rellenar una región manteniéndose coherente con el contexto. Esta multimodalidad es exactamente lo que justifica la función de pérdida combinada (sección 5): un único término L2 no puede capturarla.

## 4. Arquitectura: encoder-decoder y la capa channel-wise fully connected

La arquitectura es un *pipeline* encoder-decoder simple en apariencia, pero con una pieza no trivial en el medio.

### 4.1. Encoder

El encoder deriva de **AlexNet** (Krizhevsky et al., 2012). Dada una imagen de entrada de 227×227, usa las primeras cinco capas convolucionales más el *pooling* siguiente (la capa `pool5`) para computar una representación abstracta de **6×6×256 dimensiones**. A diferencia de AlexNet, el modelo no se entrena para clasificación de ImageNet, sino para predicción de contexto "desde cero", con pesos inicializados aleatoriamente.

### 4.2. El problema de la propagación de información y la capa channel-wise fully connected

Aquí está la sutileza arquitectónica más importante del paper. Si el encoder se limitara a capas convolucionales, **no habría forma de que la información se propague directamente de una esquina del mapa de características a otra**. La razón: las capas convolucionales conectan todos los mapas de características entre sí, pero nunca conectan directamente todas las ubicaciones *dentro* de un mapa específico. Tradicionalmente, esa propagación global la hacen las capas *fully-connected* (o de producto interno), donde todas las activaciones se conectan directamente entre sí.

El problema es de presupuesto de parámetros. La dimensión latente es 6×6×256 = **9216** tanto para el encoder como para el decoder (nótese que, a diferencia de un autoencoder, **no se reconstruye la entrada original y por tanto no hace falta un cuello de botella más pequeño**). Conectar totalmente encoder y decoder con una capa fully-connected densa resultaría en una explosión de parámetros: **más de 100 millones**, al punto de hacer inviable el entrenamiento eficiente en las GPU de la época.

La solución es la **capa channel-wise fully connected**. Es, en esencia, una capa fully-connected "con grupos", pensada para propagar información *dentro* de las activaciones de cada mapa de características, pero *no entre* mapas distintos. Si la entrada tiene $m$ mapas de características de tamaño $n \times n$, la capa produce $m$ mapas de tamaño $n \times n$, pero **sin parámetros que conecten mapas diferentes**. El conteo de parámetros se vuelve $mn^4$, frente a los $m^2n^4$ de una fully-connected completa (ignorando el sesgo) — un factor $m$ de ahorro. A esta capa le sigue una convolución de stride 1 para propagar información *a través* de los canales, recuperando así la mezcla entre mapas que la channel-wise omitió.

Esta capa es el corazón del diseño: permite que "cada unidad del decoder razone sobre el contenido entero de la imagen" sin el costo prohibitivo de una fully-connected densa.

### 4.3. Decoder

El decoder genera los píxeles de la imagen a partir de las características del encoder, conectadas vía la channel-wise fully connected. Luego viene una serie de **cinco capas up-convolucionales** (deconvoluciones) con filtros aprendidos, cada una con una activación ReLU. Una up-convolución es simplemente una convolución que produce una imagen de mayor resolución; puede entenderse como *upsampling* seguido de convolución, o como convolución con stride fraccionario. La intuición: la serie de up-convoluciones y no-linealidades constituye un *upsampling no lineal ponderado* de la característica producida por el encoder, hasta alcanzar aproximadamente el tamaño objetivo.

## 5. Función de pérdida: reconstrucción L2 + adversarial GAN

El paper entrena regresando al contenido *ground-truth* de la región removida. Pero como esa tarea es multimodal, los autores **desacoplan la carga en una función de pérdida conjunta** con dos términos que se reparten responsabilidades.

### 5.1. Pérdida de reconstrucción (L2)

Es una distancia L2 enmascarada y normalizada:

$$L_{rec}(x) = \lVert \hat{M} \odot (x - F((1-\hat{M}) \odot x)) \rVert_2^2$$

donde $\hat{M}$ es la máscara binaria de la región removida (1 donde se eliminó un píxel, 0 en los píxeles de entrada), $F$ es el context encoder, y $\odot$ es el producto elemento a elemento. Los autores experimentaron con L1 y L2 y **no encontraron diferencia significativa** entre ambas.

El defecto de esta pérdida es central a la motivación del paper: aunque "alienta al decoder a producir un contorno aproximado del objeto predicho, a menudo falla en capturar cualquier detalle de alta frecuencia" (Figura 1c, resultados borrosos). La explicación es estadística y muy citada: cuando hay múltiples modos posibles para rellenar el agujero, **es más "seguro" para la pérdida L2 predecir la media de la distribución**, porque eso minimiza el error medio por píxel — pero la media de varias soluciones nítidas distintas es una imagen borrosa y promediada. El L2 prefiere una solución borrosa sobre texturas precisas.

### 5.2. Pérdida adversarial (GAN)

Para combatir el desenfoque, los autores añaden una **pérdida adversarial** basada en GAN (Goodfellow et al., 2014; Radford et al., DCGAN, 2016). La GAN aprende conjuntamente un discriminador $D$ que provee gradientes de pérdida al generador $G$; es un juego de dos jugadores donde $D$ intenta distinguir muestras reales de las generadas mientras $G$ intenta confundir a $D$ produciendo muestras que parezcan reales. El objetivo estándar es:

$$\min_G \max_D \; \mathbb{E}_{x \in X}[\log(D(x))] + \mathbb{E}_{z \in Z}[\log(1 - D(G(z)))]$$

Los autores adaptan este marco modelando el generador como el propio context encoder, $G \equiv F$. La pérdida adversarial "trata de hacer que la predicción se vea real, y tiene el efecto de escoger un modo particular de la distribución" — exactamente lo que el L2 no podía hacer.

Hay un detalle de implementación crucial y bien aprendido: los autores **no condicionan el discriminador en el contexto** (la máscara $\hat{M} \odot x$). Probaron GANs condicionales y descubrieron que "no entrenan fácilmente para la tarea de predicción de contexto, porque el discriminador adversarial explota fácilmente la discontinuidad perceptual entre la región generada y el contexto original" para clasificar trivialmente real versus generado, y el proceso no entrena. La fórmula adoptada condiciona solo el generador, no el discriminador:

$$L_{adv} = \max_D \; \mathbb{E}_{x \in X}\left[\log(D(x)) + \log\left(1 - D(F((1-\hat{M}) \odot x))\right)\right]$$

También encontraron que los resultados mejoraban cuando el generador **no se condicionaba en un vector de ruido**. Nótese que este objetivo fuerza a que la *salida entera* del context encoder se vea realista, no solo la región faltante.

### 5.3. Pérdida conjunta

La pérdida total es una combinación lineal:

$$L = \lambda_{rec} L_{rec} + \lambda_{adv} L_{adv}$$

con $\lambda_{rec} = 0{,}999$ y $\lambda_{adv} = 0{,}001$ en los experimentos de inpainting. La asignación de roles es clara: el L2 captura la **estructura global y la coherencia con el contexto**, el adversarial escoge un **modo nítido**. Un punto importante de honestidad: la pérdida adversarial **solo se usó para los experimentos de inpainting**, porque el entrenamiento con AlexNet (la arquitectura usada para el aprendizaje de características) "divergió con la pérdida adversarial conjunta". Por eso los resultados de *feature learning* usan solo reconstrucción.

## 6. Estrategias de enmascaramiento de regiones

Cómo se elige la región a remover resulta importante para la *generalidad* de las características aprendidas. El paper presenta tres estrategias (Figura 3):

- **Región central.** El parche cuadrado en el centro de la imagen. Funciona bien para inpainting, pero la red "aprende características de bajo nivel que se aferran al borde de la máscara central". Esas características no generalizan bien a imágenes sin máscaras, así que no son muy generales.
- **Bloque aleatorio.** Para evitar que la red se aferre al borde constante de la máscara, se aleatoriza: en lugar de una máscara grande fija, se remueven varias máscaras más pequeñas, posiblemente solapadas, que cubren hasta 1/4 de la imagen. Mejora, pero el bloque aleatorio aún tiene bordes nítidos a los que las características convolucionales pueden aferrarse.
- **Región aleatoria.** Para eliminar por completo esos bordes, se remueven formas arbitrarias obtenidas de máscaras aleatorias del dataset PASCAL VOC 2012, deformadas y pegadas en lugares arbitrarios de otras imágenes (no de PASCAL), cubriendo hasta 1/4 de la imagen. Importante: el enmascaramiento se aleatoriza completamente y *no* se busca correlación entre la máscara fuente y la imagen — las formas son solo un medio para evitar que la red aprenda características de bajo nivel correspondientes al borde de la máscara.

En la práctica, las máscaras de **región y bloque aleatorio producen características similarmente generales**, superando significativamente a las de región central. Los autores usan **dropout de región aleatoria para todos los experimentos basados en características**.

## 7. Experimentos y resultados de transferencia

El paper evalúa los dos lados —decoder (inpainting) y encoder (transferencia)— sobre Paris StreetView e ImageNet, sin usar ninguna de sus etiquetas.

### 7.1. Inpainting semántico

Para inpainting se entrena con la pérdida conjunta. Detalles: las imágenes se redimensionan a 128×128, el cuello de botella es de **4000 unidades** (frente a las 100 de DCGAN), se usa una tasa de aprendizaje 10× mayor para el context encoder que para el discriminador, se predice un parche ligeramente mayor que solapa con el contexto (por 7 px) con peso de reconstrucción 10× en esa franja de solape, y se usa el solver ADAM con batch normalization.

Cualitativamente (Figura 6 y 10): **L2 solo** produce resultados bien alineados semánticamente pero borrosos; **adversarial solo** produce resultados nítidos pero no coherentes; **la pérdida conjunta alivia las debilidades de cada uno**. Cuantitativamente, en el dataset Paris StreetView (Tabla 1, imágenes *held-out*):

| Método | Mean L1 Loss | Mean L2 Loss | PSNR (mayor mejor) |
|---|---|---|---|
| NN-inpainting (características HOG) | 19,92% | 6,92% | 12,79 dB |
| NN-inpainting (nuestras características) | 15,10% | 4,30% | 14,70 dB |
| Nuestra reconstrucción (conjunta) | 9,37% | 1,96% | 18,58 dB |

El método paramétrico supera ampliamente al inpainting por vecino más cercano, y usar las características aprendidas dentro de un esquema NN también mejora sobre las características HOG diseñadas a mano. Frente al Content-Aware Fill de Photoshop (basado en PatchMatch), el context encoder funciona mejor en casos semánticos y algo peor en escenarios puramente texturados (Figura 5).

### 7.2. Aprendizaje de características y transferencia a PASCAL VOC

Para consistencia con trabajos previos se usa la arquitectura AlexNet en el encoder (con solo pérdida de reconstrucción, ya que el adversarial divergía). El entrenamiento es rápido: converge en ~100K iteraciones, **14 horas en una Titan X**. Para los experimentos de preentrenamiento se toman los pesos del encoder hasta `pool5` y se reinicializan las capas fully-connected (`fc6`, `fc7`).

La tabla de transferencia (Tabla 2) es la que la **Clase 28 muestra directamente**. Clasificación y detección Fast R-CNN sobre PASCAL VOC 2007 test; segmentación semántica sobre PASCAL VOC 2012 validación (vía FCN):

| Método de preentrenamiento | Supervisión | Tiempo | Clasificación | Detección | Segmentación |
|---|---|---|---|---|---|
| ImageNet | 1000 etiquetas de clase | 3 días | 78,2% | 56,8% | 48,0% |
| Gaussiano aleatorio | inicialización | < 1 min | 53,3% | 43,4% | 19,8% |
| Autoencoder | — | 14 horas | 53,8% | 41,9% | 25,2% |
| Agrawal et al. | ego-movimiento | 10 horas | 52,9% | 41,8% | — |
| Wang et al. | movimiento | 1 semana | 58,7% | 47,4% | — |
| Doersch et al. | contexto relativo | 4 semanas | 55,3% | 46,6% | — |
| **Context encoder (nuestro)** | **contexto** | **14 horas** | **56,5%** | **44,5%** | **30,0%** |

Las lecturas clave: la **inicialización aleatoria queda ~25% por debajo del modelo entrenado en ImageNet** (53,3% vs 78,2% en clasificación), pero no usa etiquetas. Los context encoders **superan significativamente al autoencoder simple** (que solo reconstruye su entrada completa: 53,8%) y a Agrawal et al., y son **competitivos con los métodos auto/débilmente supervisados concurrentes** (Doersch 55,3%, Wang 58,7%) a una fracción del tiempo de cómputo. En segmentación, donde hay menos competidores reportados, el context encoder alcanza el mejor resultado no supervisado de la tabla (30,0%).

> **Nota sobre los números de la Clase 28.** La clase cita "inpainting 56,5 vs random 53,3 vs ImageNet 79,9". Los 56,5 (context encoder) y 53,3 (aleatorio) coinciden exactamente con la columna de clasificación de la Tabla 2; el 79,9 de ImageNet difiere ligeramente del 78,2 del paper (la clase usa probablemente una cifra de ImageNet redondeada de otra fuente o de una corrida con *rescaling*). El mensaje es idéntico: el inpainting cierra buena parte de la brecha entre azar y supervisión completa sin usar una sola etiqueta.

El paper también muestra (Figura 8) que recuperar *vecinos más cercanos* de parches usando solo la característica del contexto produce parches semánticamente similares al original (no visto), superando a HOG y siendo competitivo con AlexNet — evidencia de que la representación captura semántica, no solo apariencia.

## 8. Limitaciones reconocidas

- **La pérdida L2 conservadora pierde detalle.** Es la limitación central que motiva todo el aporte adversarial: el L2 (o L1) prefiere la media borrosa de los modos posibles, perdiendo detalle de alta frecuencia. El adversarial lo corrige, pero introduce su propio costo de estabilidad.
- **El adversarial no converge con AlexNet.** Una limitación honesta y consecuente: los autores no lograron hacer converger la pérdida adversarial con la arquitectura AlexNet, por lo que **los mejores resultados de inpainting (con adversarial) y los de aprendizaje de características (sin adversarial, con AlexNet) usan arquitecturas distintas**. No se obtiene lo mejor de ambos mundos simultáneamente.
- **Peor que la síntesis de textura en regiones texturadas.** Si una región puede rellenarse con texturas de bajo nivel, los métodos clásicos de síntesis de textura suelen superar al context encoder (Figura 5, fila inferior). La fortaleza del método es lo semántico, no lo texturado.
- **El enmascaramiento central induce atajos.** Como se discutió, la región central hace que la red se aferre a los bordes de la máscara, produciendo características que no generalizan — de ahí la necesidad del enmascaramiento de región aleatoria.
- **Pregunta abierta de fondo.** Los autores son explícitos: "no está claro aún si requerir una generación fiel de píxeles es necesario para aprender buenas características visuales". Es decir, no sabemos si la regresión de píxeles es el camino correcto, o solo *un* camino. Esta pregunta orientó buena parte de la investigación posterior en SSL (que migró hacia objetivos contrastivos y luego predictivos en el espacio latente).

## 9. Impacto: uno de los primeros pretext generativos

Los context encoders ocupan un lugar fundacional en la historia del aprendizaje autosupervisado. Fueron **uno de los primeros pretext tasks de naturaleza generativa** —predecir píxeles ausentes— en contraste con los pretext tasks discriminativos contemporáneos (predicción de posición relativa de Doersch, rotación, *jigsaw puzzles*). Establecieron empíricamente que un objetivo de reconstrucción enmascarada, aplicado a regiones suficientemente grandes, induce representaciones semánticas transferibles.

Su huella es doble. Por un lado, en **generación de imágenes**, la receta "reconstrucción + adversarial" para combatir el desenfoque del L2 se volvió estándar y alimentó toda una línea de trabajo en inpainting profundo y traducción de imagen a imagen (pix2pix y descendientes). Por otro, en **aprendizaje autosupervisado**, la idea de *masked prediction* —ocultar parte de la entrada y predecirla desde el resto— es la semilla conceptual directa de los modelos de enmascaramiento que dominarían años después: el *masked language modeling* de BERT en texto y, de forma casi literal en visión, los **Masked Autoencoders (MAE, He et al., 2022)**, que retoman exactamente el gesto de Pathak —enmascarar parches y reconstruirlos con un encoder-decoder— pero a escala de Vision Transformers. El context encoder es, en retrospectiva, el ancestro reconocible de esa familia.

## 10. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

La Clase 28 presenta el aprendizaje autosupervisado como el paradigma donde **la supervisión se extrae de los propios datos**, sin etiquetas humanas, mediante una tarea pretexto. El context encoder es el ejemplo que la clase usa para la categoría de **auto-predicción en imágenes**: inpainting (Pathak 2016).

- **Inpainting como auto-predicción.** El gesto pedagógico que la clase transmite es exactamente la tesis de la sección 3: ocultar una región y reconstruirla *fuerza* a la red a entender el contexto. No hay etiqueta; la "etiqueta" es la propia porción de imagen que escondimos. Resolver bien esa tarea es imposible sin capturar la estructura semántica de la escena, y por eso las características resultantes transfieren.

- **La tabla de fine-tuning en PASCAL VOC 2007.** La clase muestra la comparación inpainting 56,5 vs aleatorio 53,3 vs ImageNet 79,9, que corresponde a la columna de clasificación de la Tabla 2 del paper (sección 7.2). El punto didáctico: el preentrenamiento autosupervisado por inpainting **cierra una porción significativa de la brecha** entre la inicialización aleatoria y la supervisión completa de ImageNet, *sin usar etiquetas*. Es la evidencia cuantitativa de que el pretext task funciona.

- **Conexión con los autoencoders (que la clase introduce antes).** La clase presenta los autoencoders previamente, y el context encoder es la extensión natural y motivada: un autoencoder ordinario puede copiar píxeles a través de su cuello de botella sin aprender semántica; un denoising autoencoder corrompe localmente la entrada; el context encoder lleva esa corrupción al extremo espacial —un agujero grande— de modo que deshacerlo *requiere* semántica. Entender por qué el context encoder no necesita un cuello de botella pequeño (porque no reconstruye la entrada completa, sección 4.2) es ver con precisión en qué se parece y en qué se separa del autoencoder que la clase ya enseñó.

- **El porqué de la pérdida combinada, traducido a la clase.** La intuición transferible es que **una pérdida de reconstrucción conservadora (L2) promedia los futuros posibles y borronea**, y que la pérdida adversarial existe para forzar a la red a comprometerse con un modo nítido. Esta tensión entre "predecir la media segura" y "predecir una muestra realista" reaparece en muchos contextos de modelado generativo y de SSL, y el context encoder es el lugar canónico donde verla por primera vez con claridad.

Enlaces internos del curso: [/fundamentos/aprendizaje-autosupervisado](/fundamentos/aprendizaje-autosupervisado) y [/clases/clase-28](/clases/clase-28).
