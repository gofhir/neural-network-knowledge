# 3D Convolutional Neural Networks for Human Action Recognition (Ji, Xu, Yang y Yu) — Análisis interno

> **Nota sobre versiones.** El trabajo existe en dos formas: **ICML 2010** (8 páginas) y una **extensión en IEEE TPAMI 35(1):221–231, enero 2013** (DOI `10.1109/TPAMI.2012.59`). El PDF en `clase_38/papers/Ji-3DCNN-2013.pdf` es la versión ICML, y **todas las cifras de abajo provienen de ese texto**. Las contribuciones exclusivas de TPAMI se tratan en la sección 7 de forma conceptual, sin citar números de esa versión para no inventarlos.

## 1. Metadata y resumen ejecutivo

- **Autores:** **Shuiwang Ji** (Arizona State University), **Wei Xu**, **Ming Yang** y **Kai Yu**, los tres en **NEC Laboratories America, Inc.** (Cupertino, CA). Los agradecimientos aclaran que *"la parte principal de este trabajo se hizo durante la pasantía del primer autor en NEC Laboratories America"*: es un paper de laboratorio industrial.
- **Venue:** ICML 2010 (Haifa), con extensión en TPAMI 2013.
- **Datasets:** TRECVID 2008 development set (vigilancia del aeropuerto de **London Gatwick**) y **KTH**.
- **Cifras clave:** entrada de **7 fotogramas de 60×40**; **295 458** parámetros entrenables; **90.2%** de exactitud promedio en KTH; en TRECVID, precisión promedio de **0.5572** a FPR 1% contra 0.4805 del 2D CNN equivalente.

La afiliación importa: NEC Labs America fue uno de los pocos laboratorios que apostó al deep learning **antes** de que fuera respetable —su rama de Princeton albergó a LeCun, Bottou, Weston y Collobert, y el grupo de Cupertino que dirigía **Kai Yu** ganó la clasificación de ImageNet en 2010, dos años antes de AlexNet—. Kai Yu y Wei Xu fundarían después el Institute of Deep Learning de Baidu.

El contenido: las CNN estaban confinadas a entradas 2D, y los autores proponen **extender la convolución a 3D**, convolucionando un kernel espacio-temporal sobre el cubo formado al apilar fotogramas contiguos, para que los features de movimiento se **extraigan de los datos crudos**. Sobre esa operación construyen una red de 1 capa *hardwired*, 3 convolucionales, 2 de submuestreo y 1 de conexión completa, que comprime 7 fotogramas en un vector de 128 dimensiones con un clasificador lineal encima. Tran et al. (R(2+1)D, 2018) lo reconocen como el origen: las *"3D CNN que usan convoluciones temporales para reconocer acciones humanas en video fueron propuestas, se puede argumentar, primero por Baccouche et al. y por Ji et al."*

Para la **Clase 38** este es el **ancestro olvidado**: la idea central de C3D (2015) e I3D (2017) —convolucionar en espacio y tiempo a la vez— ya está aquí, completa y formalizada, **cinco y siete años antes**. Lo que faltaba no era la idea: faltaban datos y profundidad.

## 2. Contexto: el reconocimiento de acciones alrededor de 2010

Hay que fijar la fecha con brutalidad: ICML 2010 es **dos años antes de AlexNet**. El pipeline dominante era lo que el paper llama *"el paradigma convencional de reconocimiento de patrones"*: (1) detectar **puntos de interés espacio-temporales** —los **STIP** de Laptev, extensión de Harris al volumen espacio-temporal, o los **cuboides de Dollár et al. (2005)**, que reemplazaban Harris3D por filtros separables (Gaussiana en espacio, Gabor en tiempo) porque Harris3D encontraba demasiado pocos puntos en video real—; (2) describir con HOG/HOF/SIFT y cuantizar contra un diccionario de *k*-means, la **bag of visual words**, con **spatial pyramid matching** (Lazebnik et al., 2006) para recuperar algo de estructura espacial; (3) un **SVM** encima. El paper canónico de la receta es `Schüldt et al., 2004`, literalmente *"Recognizing human actions: a local SVM approach"*, que además introdujo KTH. En paralelo estaba **HMAX** (Serre et al., 2005), jerarquía biológicamente inspirada de *template matching* y *max pooling* alternados con un banco de Gabor en su capa S1, extendida a video por `Jhuang et al., 2007`. El paper marca la diferencia filosófica: *"las CNN son sistemas completamente entrenables en los que todos los parámetros se ajustan según los datos, mientras que todos los módulos de HMAX consisten en conexiones y parámetros hechos a mano"*.

¿Por qué las CNN no habían llegado al video? Tres razones. **Estaban "limitadas a manejar entradas 2D"**: el único antecedente que encuentran es `Ning et al., 2005` (embriones en desarrollo), aplicado **cuadro por cuadro**, lo cual *"no considera la información de movimiento codificada en múltiples fotogramas contiguos"*. **No había datos**: los benchmarks eran KTH y colecciones igualmente diminutas, y el modelo *"requiere un gran número de muestras etiquetadas"*. **No había pre-entrenamiento**: inicializar desde ImageNet no era una práctica, y el remedio que proponen a futuro es el pre-entrenamiento **no supervisado** al estilo `Ranzato et al., 2007`.

El argumento contra los features a mano es específico de la tarea —*"clases de acción distintas pueden verse dramáticamente diferentes en apariencias y patrones de movimiento"*, así que *"rara vez se sabe cuáles features son importantes"*— y va acompañado de una crítica de honestidad experimental: los métodos existentes asumen entornos controlados, supuestos que *"rara vez se sostienen en entornos reales"*. De ahí la elección deliberada de TRECVID Gatwick en vez de solo KTH.

## 3. Contribución central

Extender la convolución de 2D a 3D para que los features de movimiento **se aprendan de los datos crudos** en lugar de diseñarse a mano. Cuatro piezas: (1) **la operación**, formalizar la convolución 3D sobre el cubo de fotogramas contiguos, de modo que al compartirse el kernel en el tiempo cada mapa quede conectado a varios fotogramas y así *"capture información de movimiento"*; (2) **la multiplicidad**, mostrar que aplicando múltiples convoluciones distintas en la misma ubicación se extraen múltiples tipos de features; (3) **la arquitectura multicanal**, que genera varios canales desde los fotogramas adyacentes, convoluciona y submuestrea **separadamente en cada canal** y los combina solo al final; (4) **la validación en video real**, con la conclusión de que el 3D CNN *"supera al 2D CNN basado en fotogramas en la mayoría de las tareas"* y que *"las diferencias tienden a ser mayores cuando el número de muestras positivas de entrenamiento es pequeño"*.

Al pasar, el contexto competitivo: con un sistema multi-módulo que incluía el 3D CNN como un módulo participaron en tres tareas de la **TRECVID 2009 Evaluation for Surveillance Event Detection** y *"lograron el mejor desempeño en las tres"*. Los resultados publicados son sobre el **development set de 2008**, para evaluar el módulo aislado.

## 4. La convolución 3D: formulación

El punto de partida es la convolución 2D. El valor de la unidad en $(x,y)$ del $j$-ésimo mapa de la $i$-ésima capa es

$$v_{ij}^{xy} = \tanh\!\left(b_{ij} + \sum_{m} \sum_{p=0}^{P_i-1} \sum_{q=0}^{Q_i-1} w_{ijm}^{pq}\, v_{(i-1)m}^{(x+p)(y+q)}\right)$$

con $\tanh(\cdot)$ la tangente hiperbólica, $b_{ij}$ el sesgo del mapa, $m$ indexando los mapas de la capa $(i-1)$ conectados al actual, $w_{ijm}^{pq}$ el valor en $(p,q)$ del kernel y $P_i$, $Q_i$ su altura y ancho. (Erratum menor: en el original ese kernel aparece como $w_{ijk}^{pq}$, con subíndice $k$ mientras la sumatoria indexa por $m$.)

La extensión 3D agrega un tercer eje de sumatoria. El valor en $(x,y,z)$ es

$$v_{ij}^{xyz} = \tanh\!\left(b_{ij} + \sum_{m} \sum_{p=0}^{P_i-1} \sum_{q=0}^{Q_i-1} \sum_{r=0}^{R_i-1} w_{ijm}^{pqr}\, v_{(i-1)m}^{(x+p)(y+q)(z+r)}\right)$$

donde $R_i$ es el tamaño del kernel a lo largo de la dimensión temporal y $w_{ijm}^{pqr}$ el valor $(p,q,r)$ del kernel conectado al $m$-ésimo mapa previo.

**Por qué captura movimiento.** El kernel tiene extensión $R_i > 1$ en el eje $z$ y sus pesos se **comparten** al deslizarlo sobre cubos 3D solapados. Un peso positivo en $(p_1,q_1,0)$ y uno negativo en $(p_2,q_2,2)$ implementan un detector de "algo que estaba aquí y dos fotogramas después está allá": una derivada espacio-temporal aprendida. En una CNN 2D aplicada cuadro a cuadro el gradiente nunca ve dos instantes simultáneamente, así que ningún parámetro puede codificar una velocidad.

**Por qué un kernel no basta.** El paper explicita una consecuencia del *weight sharing*: *"un kernel convolucional 3D solo puede extraer un tipo de features del cubo de fotogramas, ya que los pesos del kernel se replican a lo largo de todo el cubo"*. Compartir pesos sobre el volumen da invarianza traslacional en espacio y tiempo, pero también fuerza que un kernel sea **un único detector**: si aprendió "movimiento a la derecha de un borde horizontal", no queda grado de libertad para el vertical. De ahí el principio que invocan —*"el número de mapas de features debe incrementarse en las capas tardías"*— implementado con **múltiples kernels distintos en la misma ubicación**.

## 5. La arquitectura completa, capa por capa

Entrada: **7 fotogramas de 60×40 centrados en el fotograma actual**.

| Capa | Operación | Kernel | Salida | Parámetros |
| --- | --- | --- | --- | --- |
| Entrada | — | — | 7 @ 60×40 | 0 |
| **H1** | *hardwired*, 5 canales | fijo, no entrenable | **33** @ 60×40 | 0 |
| **C2** | conv 3D por canal, **2** juegos | 7×7×3 | 23×2 @ 54×34 | 1480 |
| **S3** | submuestreo | 2×2 | 23×2 @ 27×17 | 92 |
| **C4** | conv 3D por canal y juego, **3** kernels | 7×6×3 | 13×6 @ 21×12 | 3810 |
| **S5** | submuestreo | 3×3 | 13×6 @ 7×4 | 156 |
| **C6** | conv **2D**, conexión completa a los 78 mapas de S5 | 7×4 | 128 @ 1×1 | 289 536 |
| Salida | conexión completa (clasificador lineal) | — | 3 unidades | 384 |
| **Total** | | | | **295 458** |

Los números cierran exactos: $1480+92+3810+156+289\,536+384 = 295\,458$.

**H1 y la asimetría del flujo.** Los cinco canales son `gray` (valores de gris de los 7 fotogramas), `gradient-x` y `gradient-y` (gradientes horizontal y vertical de cada fotograma) y `optflow-x` y `optflow-y` (flujo óptico **entre fotogramas adyacentes**). La aritmética del 33 es reveladora: $7+7+7+6+6$. Los canales de flujo tienen **6** mapas y no 7 porque el flujo se define entre pares consecutivos, y esa asimetría se arrastra por toda la red: los mapas temporales por juego son $5+5+5+4+4=23$ en C2 y $3+3+3+2+2=13$ en C4. El submuestreo de S3 y S5 gasta 2 parámetros por mapa (coeficiente multiplicativo entrenable más sesgo): es el submuestreo clásico de LeNet, no max-pool sin parámetros.

**C4 y la separación de canales.** El kernel $7\times6\times3$ se aplica *"en cada uno de los 5 canales en los dos juegos de mapas **por separado**"*: **no hay mezcla entre canales**. Que las cinco vías viajen **aisladas** desde H1 hasta C6 hace de esta red una arquitectura de **cinco streams con fusión tardía** —uno de apariencia, dos de gradiente espacial, dos de movimiento—, antecedente directo del **two-stream** de Simonyan y Zisserman (2014) cuatro años antes, salvo que aquí los streams comparten topología y entrenamiento. El paper reporta haber explorado el eje: *"hemos diseñado y evaluado otras arquitecturas 3D CNN que combinan múltiples canales en distintas etapas, y nuestros resultados muestran que esta arquitectura da el mejor desempeño"*. La fusión tardía ganó, igual que ganaría en two-stream.

**C6 y el colapso temporal.** *"El tamaño de la dimensión temporal ya es relativamente pequeño (3 para gray, gradient-x, gradient-y, y 2 para optflow-x y optflow-y), así que realizamos convolución solo en la dimensión espacial en esta capa."* El kernel $7\times4$ es exactamente el tamaño espacial de S5, así que la salida cae a $1\times1$. El punto conceptual: el tiempo no desaparece por una convolución con $R_i$ igual al remanente temporal, sino por la **sumatoria sobre $m$**. Los 78 mapas de S5 contienen todas las posiciones temporales de todos los canales, así que conectar cada unidad de C6 a los 78 **fusiona globalmente tiempo y canales en un paso**: es fusión total, no convolución, y el único momento en que las cinco vías se hablan.

*Gotcha* aritmético: con un sesgo por mapa de salida, C6 tendría $128\times(78\cdot7\cdot4+1)=279\,680$ parámetros, no 289 536. El número publicado corresponde a $128\times78\times(28+1)$, o sea **un sesgo por conexión entrante**; la capa de salida, en cambio, cuenta $3\times128=384$ **sin sesgos**. Dos convenciones en la misma tabla.

Todo se inicializa **al azar** y se entrena con **retropropagación en línea** al estilo LeCun et al. 1998. Y la observación de escala: **C6 concentra el 98.0% de los parámetros** ($289\,536/295\,458$); las tres capas convolucionales juntas suman 5446 pesos, un 1.8% del modelo. En capacidad, esta red es casi enteramente un clasificador lineal alimentado por un extractor minúsculo.

## 6. La capa hardwired: el prior a mano que I3D elimina

La justificación cabe en una oración desarmante: *"esta capa hardwired se usa para codificar nuestro conocimiento previo sobre features, y este esquema usualmente conduce a mejor desempeño en comparación con la inicialización aleatoria"*.

Es la confesión central del paper. Los autores acaban de argumentar que el problema del estado del arte era depender de features hechos a mano y que las CNN *"automatizan la construcción de features"*, y sin embargo **la primera capa de su red son features hechos a mano**: gradientes tipo Sobel y flujo óptico precomputado, cero parámetros entrenables. ¿Por qué no aprenderla? Porque no tenían con qué. Un `conv1` aprendido converge a detectores de borde y filtros tipo Gabor —exactamente lo que muestran las visualizaciones de AlexNet dos años después— pero para que eso emerja hacen falta del orden de un millón de imágenes. Ellos tenían unos cientos de miles de parches de 60×40 de cinco días de video de un aeropuerto, y una red entrenada con SGD en línea sin ReLU, sin dropout, sin batch normalization y sin GPU mencionada. La capa hardwired es una **inyección manual del prior que la red no podía descubrir sola**.

Aquí está el contraste con I3D, y es el corazón del argumento de la clase. Siete años después Carreira y Zisserman resuelven **el mismo problema** —dar a la primera capa un prior visual que los datos de video no alcanzan a proveer— con la respuesta opuesta: en lugar de escribir los filtros a mano los **heredan** de una red pre-entrenada en ImageNet, inflando cada filtro $N\times N$ a $N\times N\times N$ y replicando los pesos $N$ veces divididos por $N$ (el *boring-video fixed point*). I3D arranca con un `conv1` que ya sabe de bordes, texturas y colores aprendidos de 1.2 millones de imágenes, y luego lo **sigue afinando**: su Figura 4 muestra que los filtros $7\times7\times7$ del stream RGB terminan desarrollando *"rica estructura temporal"*. **Ji et al. e I3D resuelven el mismo problema; Ji con conocimiento humano, I3D con datos.** El prior de Ji es fijo, no se refina y solo contiene lo que sus autores supieron anticipar en 2010; el de I3D es aprendido, se afina y contiene lo que contiene ImageNet.

## 7. Mezcla de salidas y regularización con features auxiliares de largo alcance

> Contribuciones de la **extensión TPAMI 2013**, ausentes del PDF ICML disponible. Se explica el mecanismo; deliberadamente **no se citan cifras** de esa versión.

**Combinación de modelos (mezcla de salidas).** En lugar de un único 3D CNN se entrena un conjunto de arquitecturas con configuraciones distintas —los mismos principios, variando cómo y cuándo se combinan los canales— y las salidas se **mezclan** para la predicción final; el mejor desempeño reportado corresponde a la combinación, no a un modelo individual. Es coherente con dos señales de la versión ICML: las *"otras arquitecturas 3D CNN que combinan múltiples canales en distintas etapas"* que dicen haber evaluado, y que su participación en TRECVID 2009 fue con *"un sistema multi-módulo de detección de eventos que incluye el 3D CNN como un módulo"*. La mezcla formaliza dentro del paper lo que ya hacían en la competencia.

**Features auxiliares de largo alcance como regularizador.** La razón de la muleta se lee en la arquitectura: el campo receptivo temporal es de **7 fotogramas muestreados cada 2**, un span de 13 fotogramas a 25 fps, **unos 0.5 segundos**. Media docena de acciones interesantes no ocurren en medio segundo: `ObjectPut` tiene fase de aproximación, de depósito y de retirada, y medio segundo puede caer entero dentro de cualquiera y verse idéntico a `CellToEar` en el instante equivocado. El 3D CNN es, por construcción, **ciego al contexto más allá de esos 13 fotogramas**. La solución de TPAMI: computar features de movimiento de alto nivel sobre una **ventana temporal mucho más larga** —representaciones tipo bag-of-words sobre descriptores densos y sobre *motion edge history images* (MEHI), la misma clase de features que en ICML servían solo como líneas base, en el linaje de los cuboides espacio-temporales de Dollár— y usarlas como **salidas auxiliares** que la red debe predecir además de la clase. Actúan como **regularizador multi-tarea**: fuerzan a que el vector de 128 dimensiones sea informativo respecto de un contexto que la red nunca vio. La genealogía está en las propias citas de la versión ICML: `Ahmed, Yu, Xu, Gong, Xing (ECCV 2008)`, citado ahí como ejemplo de *"regularización apropiada"*, es literalmente *"...transfer learning from **pseudo-tasks**"*, de los mismos autores de NEC.

El punto para la clase: **esa muleta es un sustituto artesanal del pre-entrenamiento a gran escala**. Inyectaron contexto de largo alcance desde afuera porque no podían (a) alargar la ventana temporal —habría multiplicado memoria y parámetros sin más datos para sostenerlos— ni (b) pre-entrenar el extractor sobre un corpus grande de video. C3D resuelve (a) con 16 fotogramas y Sports-1M; I3D resuelve (a) y (b) juntos con 64 fotogramas y ImageNet + Kinetics. Ninguno necesita bag-of-words auxiliares: **la muleta desaparece cuando aparecen los datos**.

## 8. Experimentos en TRECVID 2008 (London Gatwick)

**El dataset.** **49 horas** de video del aeropuerto de London Gatwick, **5 cámaras**, **720×576 a 25 fps**. Se excluyen los videos de la **cámara 4** porque *"ocurrieron pocos eventos en esa escena"*. Tres clases —`CellToEar`, `ObjectPut`, `Pointing`— clasificadas **uno-contra-el-resto**, con muchos negativos generados de acciones fuera de esas clases, en cinco fechas:

| Fecha | CellToEar | ObjectPut | Pointing | Negative | Total |
| --- | --- | --- | --- | --- | --- |
| 20071101 | 2692 | 1349 | 7845 | 20 056 | 31 942 |
| 20071106 | 1820 | 3075 | 8533 | 22 095 | 35 523 |
| 20071107 | 465 | 3621 | 8708 | 19 604 | 32 398 |
| 20071108 | 4162 | 3582 | 11 561 | 35 898 | 55 203 |
| 20071112 | 4859 | 5728 | 18 480 | 51 428 | 80 495 |
| **Total** | **13 998** | **17 355** | **55 127** | **149 081** | **235 561** |

El **desbalance** opera en dos ejes. Negativos contra positivos: 149 081 de 235 561 (**63.3%**), y como el esquema es uno-contra-el-resto, el negativo efectivo para `CellToEar` es $235\,561-13\,998=221\,563$, razón **1:15.8**. Entre clases positivas, `Pointing` tiene **55 127** muestras, **3.94 veces** las de `CellToEar` y **3.18 veces** las de `ObjectPut` — esa asimetría explica el resultado más interesante del paper. También hay desbalance entre fechas (31 942 contra 80 495, factor 2.5), así que los folds no son equivalentes en tamaño.

**Preprocesamiento.** Como cada fotograma contiene varias personas, aplican un **detector de personas y un tracker guiado por detección** para localizar cabezas y derivar un *bounding box* por actor, que se extrae **en la misma posición** de fotogramas anteriores y posteriores. La dimensión temporal se fija en **7** invocando `Schindler & Van Gool, 2008`: *"5–7 fotogramas son suficientes para lograr un desempeño similar al obtenible con la secuencia completa"*. Se extraen con **paso 2** (**-6, -4, -2, 0, 2, 4, 6**) y cada parche se escala a **60×40**. El paso 2 ensancha el span a 13 fotogramas (0.52 s) pero **submuestrea el movimiento**, lo que degrada el flujo óptico entre fotogramas separados por 80 ms.

**Líneas base.** El **2D CNN** basado en fotogramas, para aislar el aporte de la tercera dimensión, y dos variantes bag-of-words: **SIFT densos** cada 6 píxeles de parches de $7\times7$ y $16\times16$ sobre imágenes de gris o sobre **MEHI** (`Yang et al., 2009`), calculados sobre **los mismos cubos**, cuantizados blandamente con un **codebook de 512 palabras** y agregados con spatial pyramid matching en celdas de $2\times2$ y $3\times4$ — $512\times(4+12)=\mathbf{8192}$ dimensiones — con **SVM lineal uno-contra-todos**. Se denotan **SPMcube-gray** y **SPMcube-MEHI**. *(La extensión TPAMI agrega líneas base adicionales, cuyas tablas no están en el PDF disponible.)* Protocolo: **validación cruzada de 5 folds** donde *"los datos de un solo día se usan como un fold"* —split por fecha, lo que evita fuga temporal—, con **precisión, recall y AUC** a **FPR = 0.1%** y **FPR = 1%**; los AUC están multiplicados por $10^3$.

Promedios sobre las tres clases (columna *Average* de la Tabla 2):

| Método | FPR | Precisión | Recall | AUC (×10³) |
| --- | --- | --- | --- | --- |
| **3D CNN** | 0.1% | **0.7137** | **0.0230** | **0.0129** |
| 2D CNN | 0.1% | 0.6085 | 0.0155 | 0.0092 |
| SPMcube-gray | 0.1% | 0.6056 | 0.0157 | 0.0087 |
| SPMcube-MEHI | 0.1% | 0.6269 | 0.0157 | 0.0081 |
| **3D CNN** | 1% | **0.5572** | **0.1132** | **0.6752** |
| 2D CNN | 1% | 0.4805 | 0.0833 | 0.4844 |
| SPMcube-gray | 1% | 0.4817 | 0.0836 | 0.4855 |
| SPMcube-MEHI | 1% | 0.5020 | 0.0901 | 0.5099 |

El 3D CNN gana en las seis casillas. A FPR 1% la precisión promedio sube de 0.4805 (2D CNN) a 0.5572, **+16.0%** relativo; el recall de 0.0833 a 0.1132, **+35.9%**; el AUC parcial de 0.4844 a 0.6752, **+39.4%**. Contra la mejor línea base a mano (SPMcube-MEHI) la precisión mejora **11.0%**.

Precisión por clase, a ambas FPR:

| Método | CellToEar 0.1% | ObjectPut 0.1% | Pointing 0.1% | CellToEar 1% | ObjectPut 1% | Pointing 1% |
| --- | --- | --- | --- | --- | --- | --- |
| **3D CNN** | **0.6433** | **0.6748** | 0.8230 | **0.4091** | **0.5154** | 0.7470 |
| 2D CNN | 0.3842 | 0.5865 | **0.8547** | 0.3032 | 0.3937 | 0.7446 |
| SPMcube-gray | 0.3576 | 0.6051 | 0.8541 | 0.2607 | 0.4332 | 0.7511 |
| SPMcube-MEHI | 0.4848 | 0.5692 | 0.8268 | 0.3552 | 0.3961 | **0.7546** |

En `CellToEar` y `ObjectPut` el 3D CNN gana *"significativamente en todos los casos"*: a FPR 0.1% `CellToEar` pasa de 0.4848 (mejor línea base) a **0.6433**, **+32.7%**, y **+67.4%** sobre el 2D CNN. En `Pointing`, en cambio, *"el modelo 3D CNN logra un desempeño levemente peor que los otros tres métodos"*: 0.8230 contra 0.8547 del 2D CNN a FPR 0.1%, y recall de 0.0931 contra 0.1020 a FPR 1%. La explicación del paper es la que hay que retener: *"el número de muestras positivas en la clase Pointing es significativamente mayor que las de las otras dos clases. Por lo tanto, podemos concluir que el modelo 3D CNN es más efectivo cuando el número de muestras positivas es pequeño."* En términos modernos: con 295 458 parámetros y la estructura espacio-temporal impuesta por compartir pesos en el cubo, el 3D CNN tiene un **sesgo inductivo fuerte** que paga en régimen de pocos datos; con 55 127 positivos, el bag-of-words de 8192 dimensiones con SVM lineal tiene evidencia suficiente para ganar por un pelo.

Dos advertencias sobre la escala de estas cifras. El **recall se mide a FPR fija y muy baja**: con 149 081 negativos, un FPR de 0.1% permite apenas ~149 falsos positivos, así que un recall de 0.113 es la operación deliberada en el extremo de alta precisión que exige la vigilancia, no un fracaso. Y los "AUC" del orden de $10^{-4}$–$10^{-3}$ son **AUC parciales** restringidas a esa región de FPR, no el área bajo la ROC completa.

## 9. Experimentos en KTH

**Protocolo.** KTH (`Schüldt et al., 2004`) tiene **6 clases** ejecutadas por **25 sujetos**. Para comparar contra HMAX siguen el setup de `Jhuang et al., 2007`: **cubo de 9 fotogramas** y **extracción de foreground**, con la resolución bajada a **80×60** frente a los **160×120** de Jhuang para reducir memoria. La arquitectura es *"similar a la de la Figura 3"* con kernels ajustados a la entrada $80\times60\times9$: las tres capas convolucionales usan **9×7, 7×7 y 6×4**, las dos de submuestreo **3×3**, la salida vuelve a ser un vector de **128 dimensiones** y la capa final tiene **6 unidades**. Entrenamiento con **16 sujetos al azar**, test con los **9 restantes**, promediando **5 trials**.

| Método | Boxing | Handclapping | Handwaving | Jogging | Running | Walking | **Promedio** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **3D CNN** | 90 | 94 | **97** | 84 | 79 | **97** | **90.2** |
| Schüldt et al. (2004) | 97.9 | 59.7 | 73.6 | 60.4 | 54.9 | 83.8 | 71.7 |
| Dollár et al. (2005) | 93 | 77 | 85 | 57 | 85 | 90 | 81.2 |
| Niebles et al. (2008) | **98** | 86 | 93 | 53 | 88 | 82 | 83.3 |
| Jhuang et al. (2007) | 92 | **98** | 92 | **85** | 87 | 96 | 91.7 |
| Schindler & Van Gool (2008) | – | – | – | – | – | – | **92.7** |

El 3D CNN alcanza **90.2%**, tercero de seis: **+18.5 puntos** sobre Schüldt, **+9.0** sobre Dollár, **+6.9** sobre Niebles, pero **1.5** bajo HMAX (91.7) y **2.5** bajo Schindler (92.7). El paper usa la palabra exacta: desempeño **"competitivo"**, no superior. Su advertencia es de resolución: *"nótese que el modelo HMAX usa features hechos a mano computados a partir de imágenes crudas con resolución 4 veces mayor"* — $160\times120=19\,200$ píxeles contra $80\times60=4800$. Cerrar 1.5 puntos con un cuarto de los píxeles, y con features aprendidos en vez de diseñados, es el argumento real del experimento.

Sobre la **comparabilidad de los protocolos** hay que ser más severos de lo que el paper es. La tabla mezcla números de seis grupos que **no usaron el mismo protocolo**: Schüldt et al. dividió los 25 sujetos en conjuntos separados de entrenamiento, validación y test; Dollár et al. y Niebles et al. reportan **leave-one-subject-out**, que entrena con 24 sujetos en vez de 16 y ve un 50% más de datos; Ji et al. y Jhuang et al. usan 16/9. A eso se suman decisiones no estandarizadas —evaluar por secuencia completa o por clip, tratar los cuatro escenarios de grabación (s1–s4, que varían escala, ropa e interior/exterior) juntos o por separado, y usar o no extracción de foreground, que Ji et al. **sí** usan y Dollár y Niebles no—. Las tres celdas con guiones de la fila de Schindler dejan el problema visible: para ese método solo el promedio es comparable. La lectura honesta es que **90.2, 91.7 y 92.7 son indistinguibles**. Es exactamente el fenómeno que I3D denunciaría siete años después con UCF-101 y HMDB-51: cuando el dataset es diminuto, casi cualquier arquitectura rinde parecido y el benchmark deja de discriminar.

Detalle diagnóstico: el 3D CNN es el **mejor** de la tabla en `Handwaving` y `Walking` (97) y el **peor** en `Boxing` (90, contra 98 de Niebles) y `Running` (79, contra 88). `Jogging` vs `Running` es la confusión clásica de KTH —difieren en velocidad, no en apariencia— y 9 fotogramas submuestreados dan poca resolución para distinguir magnitudes de velocidad; `Boxing` es la acción más localizada espacialmente, la más castigada por bajar a 80×60.

## 10. Limitaciones

- **Escala minúscula en todos los ejes.** **7 fotogramas de 60×40** (16 800 valores por canal), **3 capas convolucionales**, **295 458** parámetros. C3D (2015) usa 16 fotogramas de 112×112 —**12 veces** más valores de entrada— con 8 capas convolucionales y ~**79 millones** de parámetros, unas **267 veces** más; I3D (2017) usa 64 fotogramas de 224×224, **191 veces** el volumen de entrada de Ji.
- **Los parámetros están en el lugar equivocado.** El 98.0% de los pesos vive en C6, la capa de fusión total; el extractor espacio-temporal propiamente dicho (C2 + C4) tiene 5290 pesos.
- **La capa hardwired.** La primera capa no se aprende: gradientes y flujo óptico prescritos, cero parámetros entrenables.
- **Dependencia de flujo óptico precomputado.** Cómputo externo a la red, un hiperparámetro más, y la asimetría de 6 mapas contra 7 que se propaga por toda la arquitectura. I3D **tampoco** eliminó esta dependencia: su mejor configuración sigue necesitando un stream TV-L1 precomputado.
- **Ausencia total de pre-entrenamiento.** *"Todos [los parámetros] se inicializan al azar."* En las conclusiones apuestan al pre-entrenamiento **no supervisado** (*"el número de muestras etiquetadas puede reducirse significativamente cuando dicho modelo se pre-entrena usando algoritmos no supervisados"*, Ranzato et al. 2007); la historia respondió con pre-entrenamiento **supervisado** a gran escala.
- **Datasets pequeños y no discriminativos.** KTH es de juguete y sus números no son comparables entre protocolos. TRECVID Gatwick es realista pero de **una sola escena** y **tres clases**, con desbalance 1:15.8, lo que dificulta separar "aprendió acciones" de "aprendió el aeropuerto de Gatwick".
- **Ventana temporal de medio segundo.** 13 fotogramas a 25 fps: las acciones con estructura temporal más larga quedan estructuralmente fuera de alcance.
- **Maquinaria de 2010.** `tanh` en lugar de ReLU, submuestreo con coeficiente entrenable en lugar de max-pooling, SGD en línea, sin dropout, sin batch normalization, sin GPU mencionada.

## 11. Impacto y legado

Tran et al. (R(2+1)D, 2018) escriben que las *"3D CNN que usan convoluciones temporales para reconocer acciones humanas fueron propuestas, se puede argumentar, primero por Baccouche et al. y por **Ji et al.**"*, y en la misma frase completan el arco: *"más recientemente, se mostró que las 3D CNN conducen a resultados fuertes de reconocimiento de acciones **cuando se entrenan en datasets de gran escala**"*, citando C3D. La historia en dos oraciones: la idea es de 2010, el resultado de 2015, y lo que cambió fue la escala.

**Ji et al. 2010 → C3D (Tran et al., ICCV 2015).** C3D le quita a la convolución 3D todo lo artesanal: elimina la **capa hardwired** (entra **RGB crudo**), elimina la **separación en canales** (una torre homogénea en lugar de cinco vías aisladas) y homogeneiza el kernel a $3\times3\times3$ en todas las capas, el equivalente temporal del hallazgo de VGG en 2D. Escala a **8 capas convolucionales**, 16 fotogramas de $112\times112$ y ~79M de parámetros, y sobre todo entrena en **Sports-1M**: **1 millón de videos de YouTube en 487 clases**. Su `fc6` de 4096 dimensiones se vuelve un extractor genérico reutilizable que, con un SVM lineal encima, alcanza **82.3% en UCF-101 y 51.6% en HMDB-51** (cifras de la tabla comparativa de S3D). El vector de 128 dimensiones de Ji et al. quería ser exactamente eso, cinco años antes, sin el corpus que lo hiciera funcionar.

**C3D → I3D (Carreira & Zisserman, CVPR 2017).** Lo que C3D no resolvió es que **se entrena desde cero**: en la comparación de I3D, la variante tipo C3D es *"la única red entrenada desde cero (no hereda ImageNet)"* y tiene *"muchos más parámetros (79M) por la dimensión extra del kernel, lo que la hace más difícil de entrenar"*. I3D rompe el techo con el **inflado** y pre-entrena en **Kinetics-400** (~240 000 videos de entrenamiento): **98.0% en UCF-101 y 80.9% en HMDB-51**, reducciones de error del 63% y 35% sobre el mejor método previo, con **25M** de parámetros, menos que C3D.

**¿Por qué quedó enterrado?** Por razones de ecosistema, ninguna sobre la idea. **Nadie podía ver la ventaja sin los datos**: en KTH quedaba tercero y dentro del ruido del protocolo, y en TRECVID ganaba pero ese no era un benchmark público de facto. **Publicó en ICML, no en CVPR/ICCV**, así que la comunidad de visión no lo leyó como propio, y **llegó dos años antes de AlexNet**, cuando esa comunidad consideraba las CNN una curiosidad. Hubo además un **contraejemplo devastador en 2014**: con Sports-1M ya disponible, Karpathy et al. reportaron *"mejoras significativas frente a líneas base fuertes basadas en features (55.3% a 63.9%), pero solo una mejora **sorprendentemente modesta** frente a los modelos de un solo fotograma (59.3% a 60.9%)"*, lo que retrasó la fe en la convolución 3D hasta que C3D mostró cómo hacerla funcionar. Y **su propia propuesta de rescate era la equivocada**: apostaban al pre-entrenamiento no supervisado, la línea que se estancó.

La mejor evidencia del valor del paper es su descendencia: cada elemento que C3D e I3D **quitaron** —la capa hardwired, la separación de canales, los features auxiliares, la ventana de medio segundo— era una compensación por la falta de datos, no un error de diseño.

## 12. Conexión con la Clase 38

La clase recorre **CNN2D + pooling temporal → CNN2D + RNN → Two-Stream → C3D → I3D** y le atribuye a C3D tres desventajas: **no puede aprovechar el pre-entrenamiento de ImageNet**, **tiene muchos parámetros** y **es más difícil de entrenar**. Este paper prueba que esas tres **no son un accidente de la implementación de C3D**: son propiedades estructurales de la familia 3D, presentes ya en su primer ejemplar, cinco años antes.

**El pre-entrenamiento es el problema estructural**, y su origen es puramente dimensional. Un kernel 2D de $7\times7$ vive en $\mathbb{R}^{49}$; su versión 3D de $7\times7\times3$ vive en $\mathbb{R}^{147}$. No existe ninguna red pre-entrenada cuyos pesos habiten ese espacio, porque todas se entrenaron sobre imágenes: cualquier arquitectura 3D nace **fuera del alcance del zoo de pesos disponibles**. Ji et al. lo compensaron artesanalmente —congelando la primera capa con features prescritos e inyectando contexto de largo alcance como salidas auxiliares—; C3D lo compensó con más datos; I3D lo resolvió cambiando la pregunta: en lugar de buscar pesos 3D que no existen, **los construyó** a partir de pesos 2D con la regla $w^{3D}(t)=\frac{1}{N}w^{2D}$ para $t=1,\dots,N$, que garantiza que la red inflada responda a un video constante exactamente como la red 2D respondía a la imagen. Los "muchos parámetros" y la "dificultad de entrenar" son la misma cosa: cada kernel 3D cuesta $R$ veces su equivalente 2D, y ese factor hay que amortizarlo con datos. Ji et al. lo pagaron manteniendo la red **absurdamente pequeña** —3 capas convolucionales, 5290 pesos en el extractor de movimiento, entrada de 60×40—, que es el techo que impone la escasez. I3D lo dice con precisión: hasta entonces las 3D ConvNets habían sido *"forzosamente poco profundas (hasta 8 capas), porque su alta dimensionalidad de parámetros, combinada con la escasez de datos de video etiquetados, las hacía difíciles de entrenar y parecía excluirlas del pre-entrenamiento ImageNet"*. Ji et al. son el extremo de esa curva: 3 capas, no 8.

Dos observaciones que este paper aporta y que suelen perderse. Primera: su arquitectura **ya es multi-stream** —cinco canales procesados aislados desde H1 hasta C6 y fusionados una sola vez al final—, o sea two-stream con fusión tardía cuatro años antes de Simonyan y Zisserman: en la secuencia de la clase, **Two-Stream no viene después de la convolución 3D, viene desde adentro de ella**. Segunda: la dependencia del **flujo óptico precomputado** no se resolvió nunca en esta línea. Está en Ji et al. 2010 (`optflow-x/y`), en Two-Stream 2014 (pila de 10 fotogramas de flujo) y sigue en I3D 2017, cuya mejor configuración necesita un stream TV-L1 externo porque —según sus propios autores— una 3D ConvNet es *"puramente feedforward"* mientras los algoritmos de flujo son en cierto sentido **recurrentes** y hacen una optimización iterativa que la red no reproduce. Siete años y dos órdenes de magnitud de datos después, el mismo parche sigue ahí.

La conclusión que la clase debería extraer es la que el paper no podía extraer sobre sí mismo: **la convolución espacio-temporal no era una idea que había que inventar, era una idea que había que financiar con datos**. Todo el aparato matemático de la Clase 38 —la ecuación de $v_{ij}^{xyz}$, el kernel $P\times Q\times R$, la fusión de canales— está completo y correcto en 2010. Lo que separa el 90.2% de KTH del 98.0% de UCF-101 no es una ecuación nueva: son ImageNet, Sports-1M y Kinetics, más un truco de inflado que reconcilia la familia 3D con el pre-entrenamiento que le estaba estructuralmente negado.

---

**Nota final — relevancia para dominios con datos escasos.** Ji et al. hicieron exactamente lo que uno se ve tentado a hacer con una buena idea arquitectónica y pocos datos: encogieron el modelo hasta que cupiera en el dataset, congelaron la primera capa con features de dominio hechos a mano e inyectaron descriptores clásicos como regularizador auxiliar. Las tres decisiones son razonables, están bien ejecutadas y **produjeron un modelo que quedó tercero en un benchmark de juguete**. La alternativa que la historia validó fue la opuesta: **importar el prior desde otro dominio**, partiendo de un backbone pre-entrenado en imágenes o en Kinetics y haciendo fine-tuning sobre los datos propios. La lección no es que su arquitectura estuviera mal: es que **la ingeniería de features es lo que uno hace cuando no tiene de dónde transferir**, y hoy casi siempre lo hay.
