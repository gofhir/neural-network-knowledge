# Learning Spatiotemporal Features with 3D Convolutional Networks (C3D) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Learning Spatiotemporal Features with 3D Convolutional Networks*.
- **Autores:** Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, Manohar Paluri.
- **Afiliaciones:** Facebook AI Research y Dartmouth College.
- **Venue:** *IEEE International Conference on Computer Vision* (ICCV 2015).
- **Preprint:** arXiv:1412.0767v4 (7 oct 2015).
- **Código y modelo:** publicados en `http://vlg.cs.dartmouth.edu/c3d`.

El paper propone **C3D** (*Convolutional 3D*), un enfoque simple y efectivo para aprender **features espacio-temporales** de video usando redes convolucionales tridimensionales (3D ConvNets) entrenadas sobre un dataset supervisado a gran escala. La tesis central es que un buen **descriptor genérico de video** —análogo a lo que las features de ImageNet significaron para la imagen estática— debe cumplir cuatro propiedades: ser **genérico** (representar bien tipos muy distintos de video), **compacto** (millones de videos exigen descriptores pequeños para almacenar y recuperar), **eficiente** de computar (miles de videos por minuto en sistemas reales) y **simple** de usar (funcionar bien incluso con un clasificador lineal, sin encoders complicados).

El hallazgo se resume en tres puntos. Primero, las 3D ConvNets son más adecuadas para aprender features espacio-temporales que las 2D ConvNets, porque la convolución y el pooling 3D **preservan la información temporal** en lugar de colapsarla. Segundo, una arquitectura homogénea con kernels pequeños de $3\times 3\times 3$ en todas las capas está entre las mejores arquitecturas para 3D ConvNets. Tercero, las features C3D con un simple clasificador lineal **superan o igualan el estado del arte en 6 benchmarks distintos** de análisis de video (reconocimiento de acciones, etiquetado de similitud de acciones, clasificación de escenas dinámicas y reconocimiento de objetos), a la vez que son extremadamente compactas —alcanzan **52.8% de accuracy en UCF101 con solo 10 dimensiones**— y rápidas de computar.

Para la **Clase 36 (Introduction to Video Analysis)**, C3D representa una de las dos grandes vías para dotar de "sentido temporal" a una red que procesa video. Frente a la 2D CNN que opera cuadro a cuadro e ignora el movimiento, y frente al enfoque *two-stream* que necesita **flujo óptico precomputado** como entrada explícita del movimiento, C3D aprende el movimiento **implícitamente y de extremo a extremo** con convoluciones que se extienden también sobre el eje del tiempo.

## 2. Contexto: el tiempo como dimensión que la 2D CNN pierde

El video es una señal con estructura espacial (dentro de cada cuadro) y temporal (a lo largo de los cuadros). El problema de fondo del análisis de video es cómo modelar ambas simultáneamente. En 2015 había, en esencia, tres respuestas, y el paper las contrasta explícitamente.

**Features de imagen (2D CNN) transferidas al video.** Tras los avances de deep learning en imagen, existían modelos ConvNet preentrenados que extraían features potentes —las activaciones de las últimas capas totalmente conectadas— con excelente desempeño en transfer learning. La tentación natural era aplicarlos cuadro a cuadro y promediar. El problema, que los autores demuestran en sus experimentos, es que estas features basadas en imagen **carecen de modelado del movimiento**: describen apariencia (objetos, escena) pero no la dinámica que distingue "abrir una puerta" de "cerrarla". La razón es estructural: una convolución 2D aplicada sobre una imagen produce una imagen, y aplicada sobre múltiples cuadros tratados como canales **también produce una imagen** —es decir, la 2D ConvNet **pierde la información temporal de la señal inmediatamente después de la primera operación de convolución**. El eje del tiempo se aplasta a un valor por posición espacial y ya no vuelve.

**Trayectorias densas (iDT).** El estado del arte en features hechas a mano era *improved Dense Trajectories* (Wang y Schmid, 2013): puntos muestreados densamente en los cuadros, seguidos mediante **flujo óptico**, con descriptores de bajo nivel (HOG, HOF, MBH) extraídos a lo largo de cada trayectoria. iDT ilustra que la señal temporal se puede tratar de forma distinta a la espacial, pero es **computacionalmente intensivo** y se vuelve intratable a gran escala.

**Two-stream.** Simonyan y Zisserman (2014) obtuvieron los mejores resultados en reconocimiento de acciones combinando dos redes: una **spatial stream** sobre cuadros RGB y una **temporal stream** que recibe como entrada **flujo óptico precomputado** apilado. Aquí está el contraste clave con C3D: el two-stream **inyecta el movimiento desde afuera**, calculando explícitamente el flujo óptico antes de entrenar la red. Además, como su temporal stream usa convoluciones 2D, la información temporal se colapsa de todos modos tras la primera capa; el movimiento sobrevive solo porque ya venía codificado en el flujo de entrada.

La promesa de C3D es prescindir de todo eso: **aprender apariencia y movimiento simultáneamente, de extremo a extremo, directamente de los píxeles**, sin flujo óptico externo, sin encoders complejos, y con un descriptor que sirva genéricamente para muchas tareas sin re-entrenar el modelo en cada una.

## 3. Contribución central

Las contribuciones que el paper enuncia son:

1. **Demostrar empíricamente que las 3D ConvNets profundas son buenas máquinas de aprendizaje de features que modelan apariencia y movimiento al mismo tiempo.** Aunque las 3D ConvNets ya se habían propuesto antes (Ji et al., 2013; Karpathy et al., 2014), este trabajo es —según los autores— el primero en explotarlas en el contexto de datasets supervisados a gran escala y arquitecturas modernas para lograr el mejor desempeño en varios tipos de tarea de video. A diferencia de Ji et al., que segmentaba a los sujetos humanos con un detector antes de alimentar la red, C3D **toma cuadros completos como entrada** y no depende de ningún preprocesamiento, escalando fácilmente.

2. **Encontrar empíricamente que un kernel de convolución de $3\times 3\times 3$ en todas las capas funciona mejor** dentro del conjunto de arquitecturas exploradas.

3. **Proponer las features C3D**: un descriptor de video genérico, compacto, eficiente y conceptualmente simple que, con un modelo lineal, supera o se acerca a los mejores métodos en **4 tareas y 6 benchmarks distintos**.

## 4. Método

### 4.1. Convolución y pooling 3D

La diferencia esencial con la 2D ConvNet es geométrica. La **convolución 2D** desliza un kernel de tamaño $k\times k$ sobre las dos dimensiones espaciales (alto y ancho). La **convolución 3D** desliza un kernel de tamaño $d\times k\times k$ sobre **tres** dimensiones: alto, ancho **y tiempo**. Ese $d$ —la *profundidad temporal* del kernel— es lo que permite que una sola respuesta del filtro dependa de $d$ cuadros consecutivos, capturando cómo cambian los píxeles de un instante al siguiente. En otras palabras, el kernel 3D "ve" un pequeño **volumen espacio-temporal** y responde a patrones de movimiento (un borde que se desplaza, una textura que fluye) tan naturalmente como el kernel 2D responde a patrones espaciales.

La consecuencia decisiva la ilustra el paper: aplicar convolución 3D sobre un volumen de video (múltiples cuadros apilados en el eje temporal) **produce otro volumen**, preservando la dimensión temporal en la salida. Lo mismo vale para el pooling 3D. Por eso la información temporal **se propaga a través de todas las capas** de la red, en vez de morir tras la primera convolución como ocurre en la 2D ConvNet. Esta es la propiedad que hace a la 3D ConvNet "well-suited" para features espacio-temporales.

**Notación.** El paper denota un clip de video como $c\times l\times h\times w$, donde $c$ es el número de canales, $l$ la longitud en cuadros, y $h, w$ el alto y ancho. Un kernel de convolución o pooling 3D se escribe $d\times k\times k$, con $d$ la profundidad temporal y $k$ el tamaño espacial.

### 4.2. El estudio del tamaño de kernel temporal

La pregunta de diseño más importante es: ¿cuántos cuadros debe abarcar el kernel en el tiempo? Para responderla de forma controlada, los autores **fijan el campo receptivo espacial en $3\times 3$** (siguiendo la lección de VGG en imagen: campos receptivos pequeños con redes profundas rinden mejor) y **varían solo la profundidad temporal** $d$.

La red base del estudio tiene 5 capas de convolución (con 64, 128, 256, 256 y 256 filtros respectivamente), cada una seguida inmediatamente de un pooling, más 2 capas totalmente conectadas de 2048 salidas y una capa softmax sobre las 101 acciones de UCF101. Los cuadros se redimensionan a $128\times 171$ (aprox. media resolución), los videos se parten en clips **no solapados de 16 cuadros**, y la entrada es $3\times 16\times 128\times 171$, con recortes aleatorios de $3\times 16\times 112\times 112$ como *jittering*. Todas las convoluciones usan padding apropiado (espacial y temporal) y stride 1, de modo que no cambian el tamaño; todos los pooling son max pooling de $2\times 2\times 2$ **excepto el primero**, que es $1\times 2\times 2$ —deliberadamente sin pooling temporal— **para no fusionar la señal temporal demasiado temprano** y respetar la longitud de 16 cuadros (con factor 2 se puede hacer pooling temporal a lo sumo 4 veces antes de colapsar el eje del tiempo).

Se probaron dos familias. **Profundidad homogénea:** todas las capas con el mismo $d \in \{1, 3, 5, 7\}$, redes llamadas `depth-d`. Nótese que **`depth-1` equivale a aplicar convoluciones 2D sobre cuadros separados** —es la línea base sin tiempo. **Profundidad variable:** creciente (`3-3-5-5-7`) y decreciente (`7-5-5-3-3`).

Un punto metodológico crucial: todas estas redes tienen **prácticamente el mismo número de parámetros**. Dos redes que difieren en profundidad temporal en 2 se distinguen en apenas ~17K parámetros; la mayor diferencia, entre `depth-1` y `depth-7`, es de 51K parámetros —menos del **0.3% de los 17.5 millones** totales. Como la capacidad de aprendizaje es equivalente, cualquier diferencia de desempeño se atribuye limpiamente a la elección de $d$, no al tamaño del modelo.

**Resultado.** Sobre UCF101 test split-1 (clip accuracy), `depth-3` es el mejor entre los homogéneos. `depth-1` (el equivalente 2D) es **significativamente peor** que todos los demás, lo que los autores atribuyen a la **falta de modelado del movimiento**. Entre las redes de profundidad variable, `depth-3` sigue siendo el mejor, aunque la brecha se estrecha. El comportamiento se mantiene al probar campos receptivos espaciales mayores ($5\times 5$) o resolución completa ($240\times 320$). La conclusión: **$3\times 3\times 3$ es la mejor elección de kernel** para 3D ConvNets, y las 3D ConvNets superan consistentemente a las 2D en clasificación de video —lo cual se verifica además sobre un dataset interno a gran escala, I380K.

### 4.3. La arquitectura C3D

Con la lección "$3\times 3\times 3$ homogéneo es mejor" y la premisa "hazla tan profunda como permita la memoria de la GPU", el paper define la arquitectura genérica **C3D**:

- **8 capas de convolución**, todas con kernels $3\times 3\times 3$ y stride $1\times 1\times 1$. El número de filtros crece de 64 (conv1a) a 512 en las capas conv4–conv5.
- **5 capas de max pooling**, todas $2\times 2\times 2$ con stride $2\times 2\times 2$, **excepto pool1** que es $1\times 2\times 2$ con stride $1\times 2\times 2$, de nuevo para preservar la información temporal en la fase temprana.
- **2 capas totalmente conectadas** de **4096** unidades cada una (fc6, fc7).
- Una capa softmax de salida.

Este es el modelo que, de aquí en adelante, se llama simplemente C3D. Su tamaño de entrada operativo es de clips de $16\times 112\times 112$.

### 4.4. Entrenamiento sobre Sports-1M

C3D se entrena sobre **Sports-1M** (Karpathy et al.), el mayor benchmark de clasificación de video disponible entonces: **1.1 millones de videos deportivos** en **487 categorías** —5 veces más categorías y 100 veces más videos que UCF101. De cada video de entrenamiento se extraen aleatoriamente **cinco clips de 2 segundos**, redimensionados a $128\times 171$ y recortados a $16\times 112\times 112$ con jittering espacial y temporal, más flip horizontal con probabilidad 50%. Se entrena con SGD, minibatches de 30, learning rate inicial 0.003 dividido por 2 cada 150K iteraciones, deteniéndose a 1.9M iteraciones (~13 épocas). Se entrena una versión desde cero y otra **fine-tuneada desde un modelo preentrenado en I380K**.

### 4.5. C3D como extractor de features de propósito general

Aquí está el corazón de la propuesta. Una vez entrenada, C3D se usa como **extractor de features** para otras tareas de video sin re-entrenarla. El procedimiento es deliberadamente simple: se parte el video en clips de 16 cuadros con **8 cuadros de solapamiento** entre clips consecutivos, se pasa cada clip por la red para extraer las activaciones de **fc6**, se **promedian** las activaciones de todos los clips y se aplica **normalización L2**. El resultado es un **descriptor de video de 4096 dimensiones**. Ese vector, alimentado a un clasificador lineal (SVM), es todo lo que se necesita: no hay fine-tuning por tarea, no hay Fisher Vectors, no hay VLAD.

**¿Qué aprende C3D?** Usando deconvolución (Zeiler y Fergus), los autores observan que C3D **empieza atendiendo a la apariencia en los primeros cuadros y luego rastrea el movimiento saliente** en los cuadros siguientes. En un ejemplo, el feature se enfoca en la persona completa y luego sigue el movimiento de un salto con garrocha; en otro, se enfoca en los ojos y luego sigue el movimiento de aplicar maquillaje. Esto distingue a C3D de una 2D ConvNet estándar: **atiende selectivamente tanto a movimiento como a apariencia**. La comparación con flujo óptico es reveladora —el flujo óptico se dispara en todos los píxeles en movimiento, mientras que C3D atiende solo al **movimiento saliente**.

## 5. Experimentos y resultados

### 5.1. Clasificación en Sports-1M

C3D entrenada desde cero alcanza **84.4%** de accuracy video top-5, y la fine-tuneada desde I380K, **85.5%**. Ambas superan a las redes de DeepVideo (Karpathy et al., ~80% top-5) por unos 5 puntos. Queda 5.6% por debajo del método de *convolution pooling* sobre clips de 120 cuadros (90.8%), pero esa comparación no es directa: aquel método usa clips **mucho más largos** (120 cuadros) mientras C3D opera sobre clips cortos de 16 cuadros. En la práctica, esquemas de agregación más sofisticados pueden aplicarse **sobre** las features C3D para mejorar el desempeño a nivel de video.

### 5.2. Reconocimiento de acciones en UCF101

**UCF101**: 13,320 videos de 101 categorías de acción humana, con el protocolo de 3 splits. Se extraen features C3D y se entrena un **SVM lineal multiclase**. Resultados:

- **C3D (1 red) + SVM lineal: 82.3%**, con solo 4096 dimensiones.
- **C3D (3 redes) + SVM lineal: 85.2%** (concatenando descriptores L2-normalizados de tres redes; 12,288 dimensiones).
- **C3D (3 redes) + iDT + SVM lineal: 90.4%**.

C3D con 3 redes supera a las baselines de **iDT (+9%)** e **ImageNet (+16.4%)**. En el escenario de solo RGB, C3D supera a las Deep networks de Karpathy et al. por **19.8%** y a la spatial stream de two-stream por **12.6%** (ambas usan AlexNet). Frente a métodos basados en RNN, supera a **LRCN por 14.1%** y al **LSTM composite model por 9.4%**. Notablemente, C3D con solo RGB supera a esos métodos RNN **incluso cuando ellos usan RGB más flujo óptico**, y supera también a la temporal stream de two-stream. Solo al combinarse con iDT (features complementarias basadas en flujo óptico y gradientes de bajo nivel) C3D alcanza a superar al two-stream completo. Que combinar C3D con ImageNet dé solo **0.6%** de mejora, mientras combinar con iDT da mucho más, confirma que **C3D ya captura la apariencia** (por eso ImageNet no aporta) pero es complementario al movimiento de bajo nivel de iDT.

### 5.3. Compacidad

Aplicando PCA para proyectar las features a dimensiones bajas y clasificando con SVM lineal sobre UCF101, C3D domina a ImageNet e iDT en el régimen compacto: con solo **10 dimensiones, 52.8%** (más de 20 puntos sobre el ~32% de ImageNet e iDT); con 50 dim, **72.6%**; con 100 dim, **75.6%**; con 500 dim, **79.4%** (6% mejor que iDT, 11% mejor que ImageNet). Esto es directamente relevante para recuperación a gran escala, donde bajo costo de almacenamiento y recuperación rápida son críticos. La visualización con t-SNE muestra además que las features C3D son **semánticamente más separables** que las de ImageNet, sin ningún fine-tuning —evidencia de buena generalización entre datasets.

### 5.4. Etiquetado de similitud de acciones (ASLAN)

**ASLAN**: 3,631 videos, 432 clases; la tarea es predecir si un par de videos pertenece o no a la misma acción, con el conjunto de test conteniendo acciones "nunca vistas". Se extraen features (prob, fc7, fc6, pool5) por clip, se promedian y normalizan con L2; se computan 12 distancias por par y con 4 tipos de feature se obtiene un vector de 48 dimensiones, clasificado con SVM lineal. C3D logra **78.3% de accuracy y 86.5% de AUC**, superando al estado del arte por **9.6% en accuracy y 11.1% en AUC**, con un método mucho más simple que los competidores (que usan múltiples features hechas a mano, Fisher Vector/VLAD y modelos complejos). C3D cubre la mitad del camino entre el estado del arte previo y el desempeño humano (98.9%). La baseline ImageNet queda 10.8% por debajo de C3D **por falta de modelado del movimiento**.

### 5.5. Escenas dinámicas y objetos

**Escenas** (YUPENN: 420 videos, 14 categorías; Maryland: 130 videos, 13 categorías). Con SVM lineal y promedio simple de features de clip, C3D alcanza **98.1% en YUPENN y 87.7% en Maryland**, superando al mejor método previo por **1.9% y 10%** respectivamente, pese a que aquel usaba codificaciones complejas (FV, LLC, dynamic pooling).

**Objetos** (dataset egocéntrico: 42 objetos cotidianos, grabados en primera persona, con apariencia y movimiento muy distintos a los datos de entrenamiento). C3D obtiene **22.3%**, superando al método comparado por **10.3%** (que usaba matching SIFT-RANSAC con kernel RBF). Es notable porque C3D solo se entrenó en Sports-1M sin ningún fine-tuning. Aquí sí queda **3.4% por debajo de ImageNet**, explicable porque C3D usa menor resolución de entrada ($128\times 128$ vs $256\times 256$) e ImageNet está entrenado sobre 1000 categorías de objetos.

### 5.6. Eficiencia

El análisis de runtime sobre UCF101 muestra que C3D procesa a **313 fps**, siendo **91× más rápido que iDT** y **274× más rápido que la implementación GPU de flujo óptico de Brox** (usada por el two-stream). Mientras iDT y el flujo óptico procesan a menos de 4 fps, C3D corre **mucho más rápido que tiempo real**. Esto materializa la propiedad de "eficiente": la inferencia rápida de las ConvNets hace viable procesar miles de videos por minuto.

Un estudio complementario sobre resolución de entrada (apéndice) muestra el trade-off de tamaño: net-64 tiene 11.1M parámetros (92 min/época), net-128 tiene 17.5M (270 min/época) y net-256 tiene 34.8M (1186 min/época). net-128 supera a net-64 por 3.1% y queda comparable a net-256, ofreciendo el mejor compromiso entre tiempo de entrenamiento, accuracy y memoria.

## 6. Limitaciones

- **Muchos parámetros y alto costo de cómputo/memoria.** La arquitectura C3D es pesada (del orden de 17.5M de parámetros en la variante de estudio, y las convoluciones 3D tienen costo por filtro mayor que las 2D). El estudio de resolución muestra que entrenar a $256\times 256$ requiere **paralelismo de modelo** por el límite de memoria de GPU de la época. Esto lo hace más caro de entrenar que una 2D ConvNet equivalente.
- **Ventana temporal corta y fija.** C3D opera sobre clips de **16 cuadros**; no modela dependencias de largo alcance dentro de un solo pase. Por eso queda por debajo de métodos que usan clips de 120 cuadros en Sports-1M, y necesita agregación externa (promedio de clips, convolution pooling) para razonar sobre videos largos.
- **Dependencia de datos supervisados a gran escala.** El descriptor genérico solo emerge tras entrenar sobre Sports-1M (1.1M videos); sin un dataset de esa escala, aprender las features 3D desde cero es difícil.
- **Resolución de entrada modesta.** El uso de $112\times 112$ / $128\times 128$ penaliza tareas dominadas por detalle fino de apariencia (como el reconocimiento de objetos egocéntricos, donde ImageNet a resolución completa gana).

## 7. Conexión con la Clase 36 y con I3D

En el marco de la **Clase 36 (Introduction to Video Analysis)**, la pregunta rectora es cómo introducir información temporal más allá de una 2D CNN por cuadro. C3D encarna **una de las dos vías principales** para ese "sentido temporal":

- **Vía convolución 3D (C3D):** el tiempo es una dimensión más del kernel. El movimiento se aprende **implícitamente y de extremo a extremo** desde los píxeles RGB, sin señales externas. Es directa y unificada, pero cara en parámetros y cómputo.
- **Vía 2D CNN + modelado temporal externo:** ya sea 2D CNN + RNN (LRCN, LSTM composite) que agrega los cuadros con una recurrencia, o el **two-stream**, que necesita **flujo óptico precomputado** para inyectar el movimiento. C3D se contrasta favorablemente con ambas: supera a LRCN y al LSTM composite en UCF101, y no necesita el costoso flujo óptico del two-stream (de ahí su ventaja de 91–274× en velocidad).

El desenlace histórico de esta línea es **I3D** (Carreira y Zisserman, 2017), que se puede presentar como el sucesor directo que corrige las limitaciones de C3D. En lugar de entrenar una 3D ConvNet desde cero (caro y con pocos datos), I3D **"infla" (inflate)** arquitecturas 2D maduras ya preentrenadas en ImageNet —replicando sus kernels 2D a lo largo del eje temporal para inicializarlos como kernels 3D— y las entrena sobre el gran dataset Kinetics. Así hereda el poder de los backbones 2D preentrenados y lo transporta al dominio espacio-temporal, superando a C3D. C3D es, en este relato, el paso conceptual que instala la idea de las features espacio-temporales aprendidas; I3D es la ingeniería que la vuelve competitiva a escala.

## 8. Nota final: relevancia para video clínico

Para el video médico —endoscopía, laparoscopía, ecografía, monitoreo de pabellón o de UCI— la enseñanza de C3D es directa y valiosa: **el diagnóstico o la clasificación de un procedimiento o evento clínico rara vez vive en un solo cuadro, sino en el movimiento**. Distinguir una fase quirúrgica de otra, detectar el instante de una maniobra, o reconocer un evento fisiológico transitorio depende de cómo cambia la escena en el tiempo, no solo de su apariencia congelada. Un extractor tipo C3D permite construir un pipeline clínico **simple y de extremo a extremo**: preentrenar (o adaptar) una 3D ConvNet, extraer un descriptor espacio-temporal compacto por clip, y montar encima un clasificador liviano —sin necesidad de calcular flujo óptico externo ni de anotar trayectorias a mano. La compacidad (buen desempeño con muy pocas dimensiones) y la eficiencia (inferencia más rápida que tiempo real) son especialmente atractivas para entornos hospitalarios con restricciones de cómputo y almacenamiento, y para tareas de recuperación sobre grandes archivos de video clínico. La contracara a tener presente es la ventana temporal corta de 16 cuadros y el costo de memoria, que en la práctica clínica moderna se mitigan con los descendientes inflados tipo I3D y con esquemas de agregación sobre clips.

---

**Enlaces internos:**

- Clase: [/clases/clase-36](/clases/clase-36) — Introduction to Video Analysis.
- Fundamento transversal: [/fundamentos/redes-convolucionales](/fundamentos/redes-convolucionales) — de la convolución 2D a la 3D.
- Contraste two-stream / flujo óptico: modelado explícito del movimiento vs. aprendizaje implícito de C3D.
- Sucesor: I3D (Carreira y Zisserman, 2017) — inflado de modelos 2D preentrenados al dominio espacio-temporal.
