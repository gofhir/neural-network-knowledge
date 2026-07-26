# FlowNet: Learning Optical Flow with Convolutional Networks — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *FlowNet: Learning Optical Flow with Convolutional Networks*.
- **Autores:** Philipp Fischer, Alexey Dosovitskiy, Eddy Ilg, Philip Häusser, Caner Hazırbaş, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox. Los tres primeros contribuyeron por igual. Colaboración entre la **Universidad de Friburgo** (grupo de Thomas Brox) y la **Universidad Técnica de Múnich** (grupo de Daniel Cremers).
- **Venue:** *IEEE International Conference on Computer Vision (ICCV 2015)*.
- **Preprint:** arXiv:1504.06852v2 (4 de mayo de 2015).
- **Linaje:** primer trabajo que resuelve la estimación de flujo óptico como una **tarea de aprendizaje supervisado end-to-end** con redes convolucionales. Abrió la línea que continuaría con FlowNet 2.0 (2017) y toda la familia posterior de estimadores de flujo basados en aprendizaje profundo (PWC-Net, RAFT).

El paper ataca un problema que, hasta 2015, había resistido a las redes convolucionales: la **estimación de flujo óptico**, es decir, computar el campo de desplazamiento por píxel entre dos imágenes consecutivas. Las CNN ya dominaban tareas de reconocimiento (clasificación, detección, segmentación), pero el flujo óptico es distinto: no basta con extraer una representación semántica de una imagen, hay que **encontrar correspondencias** entre dos imágenes y localizar cada píxel con precisión subpíxel. Los autores construyen CNN capaces de resolverlo y proponen y comparan **dos arquitecturas**: una genérica (**FlowNetSimple**) y otra con una capa especializada de **correlación** (**FlowNetCorr**).

El segundo obstáculo era el dato. Entrenar una CNN exige ground-truth masivo, pero obtener flujo óptico verdadero de video real es "extremadamente difícil": no existe forma sencilla de medir las correspondencias exactas de todos los píxeles en una escena natural. La solución del paper es generar un **dataset sintético llamado Flying Chairs** —sillas 3D segmentadas superpuestas sobre fondos de Flickr, movidas con transformaciones afines aleatorias—, con lo que se puede fabricar una cantidad arbitraria de pares imagen–flujo con ground-truth denso y exacto. El hallazgo central y sorprendente: redes entrenadas solo sobre estos datos irreales **generalizan muy bien** a datasets realistas como Sintel y KITTI, sin fine-tuning, alcanzando precisión competitiva a **5–10 fps**.

Para la **Clase 36 (Introduction to Video Analysis)**, este paper importa porque el flujo óptico es el mecanismo canónico para **codificar movimiento** entre frames, y FlowNet es la bisagra entre los métodos variacionales clásicos (Horn–Schunck) y el paradigma de deep learning que domina hoy el análisis de video, la base sobre la que se construyen arquitecturas two-stream para reconocimiento de acciones y trackers para seguimiento de objetos.

## 2. Contexto: el problema del flujo óptico y los métodos variacionales clásicos

El **flujo óptico** es el campo vectorial que describe, para cada píxel de la imagen 1, hacia dónde se desplazó en la imagen 2. Si $I_1$ e $I_2$ son dos frames consecutivos, se busca un campo $(u(x,y), v(x,y))$ tal que el píxel en la posición $(x,y)$ de $I_1$ aparezca en $(x+u, y+v)$ en $I_2$. Es, en esencia, un **problema de correspondencia**: emparejar cada punto de una imagen con su contraparte en la otra, y la salida es un vector de movimiento por píxel.

La formulación clásica arranca de la **hipótesis de constancia de brillo**: se asume que la intensidad de un punto no cambia al moverse entre frames, $I_1(x,y) = I_2(x+u, y+v)$. Linealizando por Taylor se obtiene la **ecuación de restricción del flujo óptico**:

$$I_x\, u + I_y\, v + I_t = 0,$$

donde $I_x, I_y$ son los gradientes espaciales e $I_t$ el gradiente temporal. Esta única ecuación tiene dos incógnitas por píxel ($u$ y $v$), de modo que el problema está **subdeterminado** (el célebre *problema de apertura*): a partir de un solo punto no se puede recuperar la componente del movimiento paralela a los bordes.

Desde el trabajo seminal de **Horn y Schunck (1981)**, los **enfoques variacionales** dominaron el campo. Para resolver la indeterminación, añaden un **término de regularización** que impone suavidad al campo: se minimiza una energía que combina el término de datos (constancia de brillo) con un término de suavidad que penaliza gradientes bruscos del flujo. El paper resume que "los enfoques variacionales han dominado la estimación de flujo óptico desde el trabajo de Horn y Schunck", con mejoras sucesivas que integraron robustez, estimación gruesa-a-fina (coarse-to-fine) para grandes desplazamientos y **matching combinatorio** dentro del marco variacional. Trabajos como **DeepMatching / DeepFlow** agregan información de fino a grueso con convoluciones dispersas y max-pooling, y **EpicFlow** interpola matches dispersos a un campo denso respetando los bordes de la imagen. Un rasgo importante que los autores subrayan: todos estos métodos **no aprenden**; sus parámetros se fijan manualmente.

Hubo intentos previos de aplicar aprendizaje al flujo óptico, pero de forma parcial: aprender regularizadores con mezclas de gaussianas, modelar estadísticas locales del flujo, computar componentes principales de un conjunto de campos, o entrenar clasificadores para seleccionar entre estimaciones. También hubo aprendizaje no supervisado de disparidad/movimiento con máquinas de Boltzmann restringidas y autoencoders de sincronía, útiles para reconocimiento de actividad pero **no competitivos con los métodos clásicos** en video realista. En paralelo, existía trabajo sobre matching con redes neuronales (Fischer et al.; Zbontar y LeCun con arquitectura siamesa para similitud de parches), pero eran métodos **basados en parches** que delegaban la agregación espacial al postprocesamiento. La diferencia radical de FlowNet: las redes **predicen directamente campos de flujo completos**, de extremo a extremo, sin postprocesamiento de matching o interpolación hecho a mano.

## 3. Contribución central

La contribución de FlowNet es demostrar que una **CNN entrenada end-to-end** puede predecir directamente el campo de flujo óptico a partir de un par de imágenes, con precisión competitiva y a velocidades cercanas a tiempo real. Esto rompe con décadas de métodos variacionales hechos a mano. Los aportes concretos:

1. **Dos arquitecturas encoder-decoder para flujo.** FlowNetSimple (genérica: apila las dos imágenes y deja que la red descubra cómo extraer el movimiento) y FlowNetCorr (con dos streams idénticos y una **capa de correlación** que provee explícitamente capacidad de matching entre features de ambas imágenes).
2. **La capa de correlación**, un operador sin pesos entrenables que compara parches de los mapas de características de las dos imágenes, incorporando la noción de "matching" directamente en la arquitectura.
3. **El dataset sintético Flying Chairs**, que resuelve el cuello de botella del ground-truth y permite entrenar CNN grandes con ground-truth denso y exacto en cantidades arbitrarias.
4. **La demostración de generalización**: redes entrenadas sobre datos artificiales no realistas transfieren a video real sin fine-tuning, superando incluso a métodos como LDOF y batiendo el estado del arte (DeepFlow, EpicFlow) en el propio test set de Flying Chairs.

Un hallazgo casi contraintuitivo del paper: aunque los autores diseñaron la capa de correlación para *ayudar* a la red con el matching, resultó que **incluso la red genérica (FlowNetSimple), sin ninguna ayuda de matching explícito, aprende a predecir flujo óptico con precisión competitiva**. La red cruda "puede aprender por sí sola".

## 4. Método

### 4.1. Arquitectura general: contracción y expansión

Ambas redes siguen un esquema **encoder-decoder** (contractivo-expansivo). Una **parte contractiva** comprime espacialmente la información mediante convoluciones con stride y pooling, extrayendo features abstractas de alto nivel a costa de resolución. Una **parte expansiva** (refinamiento) recupera la resolución para producir una predicción densa por píxel. Detalles compartidos: nueve capas convolucionales, con **stride 2** (la forma más simple de pooling) en seis de ellas, y una no linealidad **ReLU** tras cada una. **No hay capas totalmente conectadas**, lo que permite procesar imágenes de tamaño arbitrario. Los tamaños de filtro decrecen hacia las capas profundas: $7\times7$ en la primera, $5\times5$ en las dos siguientes y $3\times3$ desde la cuarta; el número de mapas de características aproximadamente se duplica tras cada capa con stride 2.

### 4.2. FlowNetSimple: apilar y procesar

La opción más directa es **apilar las dos imágenes de entrada** una sobre otra (concatenándolas por el eje de canales) y alimentarlas a una red bastante genérica, dejando que ella misma decida cómo procesar el par para extraer la información de movimiento. Esta arquitectura, compuesta solo por capas convolucionales, es **FlowNetSimple** (FlowNetS). El razonamiento es que, si la red es suficientemente grande, en principio *podría* aprender a predecir flujo óptico. La limitación conceptual que los autores admiten es que "nunca podemos estar seguros de que una optimización por gradiente local como SGD lleve a la red a ese punto". De ahí la motivación para diseñar una arquitectura menos genérica pero potencialmente mejor adaptada al problema.

### 4.3. FlowNetCorr: dos streams y la capa de correlación

La alternativa es crear **dos streams de procesamiento separados pero idénticos**, uno por imagen, y combinarlos en una etapa posterior. Así la red primero produce representaciones significativas de cada imagen por separado y luego las combina en un nivel más alto, lo que "se asemeja al enfoque estándar de matching": primero extraer features de parches de ambas imágenes y luego comparar esos vectores de características.

El problema es cómo hacer que la red **encuentre correspondencias** dadas dos representaciones. Para ello se introduce la **capa de correlación**, que realiza comparaciones multiplicativas de parches entre los dos mapas de características. Dados dos mapas multicanal $f_1, f_2$ de ancho $w$, alto $h$ y $c$ canales, la correlación de dos parches centrados en $x_1$ (en el primer mapa) y $x_2$ (en el segundo) se define como:

$$c(x_1, x_2) = \sum_{o \in [-k,k]\times[-k,k]} \langle f_1(x_1 + o),\, f_2(x_2 + o) \rangle,$$

para un parche cuadrado de tamaño $K := 2k + 1$. La operación es **idéntica a un paso de una convolución**, pero en lugar de convolucionar los datos con un filtro aprendido, convoluciona **datos con datos**: compara features de una imagen contra features de la otra. Por eso **no tiene pesos entrenables**. Calcular $c(x_1, x_2)$ implica $c \cdot K^2$ multiplicaciones.

El costo es el problema. Comparar todas las combinaciones de parches supone $w^2 \cdot h^2$ cálculos, un resultado enorme que vuelve intratables los pases forward y backward. Por razones computacionales se **limita el desplazamiento máximo**: dado un desplazamiento máximo $d$, para cada posición $x_1$ se computan correlaciones solo en un vecindario de tamaño $D := 2d + 1$, restringiendo el rango de $x_2$. Se aplican además **strides** $s_1$ y $s_2$ para cuantizar $x_1$ globalmente y $x_2$ dentro del vecindario. El resultado teórico es cuatridimensional (un valor por cada combinación de dos posiciones 2D), pero en la práctica los desplazamientos relativos se organizan en **canales**, dando una salida de tamaño $w \times h \times D^2$. En los experimentos la capa de correlación de FlowNetCorr usa $k=0$, $d=20$, $s_1=1$, $s_2=2$.

### 4.4. Refinamiento: capas upconvolucionales

El pooling es necesario para que el entrenamiento sea factible y para agregar información sobre áreas grandes de la imagen, pero **reduce la resolución**. Para producir predicciones densas por píxel se requiere una forma de **refinar** la representación gruesa. El ingrediente principal son las **capas "upconvolucionales"**, que consisten en un *unpooling* (que expande los mapas de características, lo opuesto al pooling) seguido de una convolución. (El paper aclara que, aunque a estas capas se las suele llamar "deconvolucionales", la operación que realizan es técnicamente una convolución, no una deconvolución.)

El esquema de refinamiento es el aporte de diseño más fino. En cada paso se aplica la upconvolución a los mapas de características y se **concatena** el resultado con: (a) los mapas de características correspondientes de la parte contractiva de la red, y (b) una predicción de flujo más gruesa, upsampleada, si está disponible. De este modo se preservan **tanto la información de alto nivel** que llega de los mapas gruesos **como el detalle fino local** de las capas inferiores. La clave respecto a trabajos previos (Long et al. en segmentación semántica): FlowNet no upconvoluciona solo la predicción gruesa, sino **los mapas de características completos**, lo que transfiere más información de alto nivel a la predicción fina. Cada paso duplica la resolución; se repite **4 veces**, quedando una predicción cuya resolución es todavía 4 veces menor que la entrada. Refinar más allá de ese punto no mejora significativamente frente a un simple **upsampling bilineal** a resolución completa, que se usa como paso final.

### 4.5. Refinamiento variacional opcional (+v)

Como alternativa al upsampling bilineal, el paper ofrece un **esquema variacional** que combina lo mejor de ambos mundos. Se parte de la resolución 4 veces menor y se usa un esquema coarse-to-fine con 20 iteraciones para llevar el campo a resolución completa, más 5 iteraciones finales en la resolución plena. Se computan además los bordes de la imagen y se respetan reemplazando el coeficiente de suavidad por $\alpha = \exp(-\lambda\, b(x,y)^\kappa)$, donde $b(x,y)$ es la fuerza del borde. Este refinamiento variacional, denotado con el sufijo **+v**, es más costoso que el upsampling bilineal pero produce campos suaves y con precisión subpíxel. Como muestran los autores, para movimientos pequeños el refinamiento variacional cambia drásticamente la predicción a mejor, mientras que para movimientos grandes no corrige los errores gruesos pero suaviza el campo, bajando el error.

### 4.6. Entrenamiento y pérdida

La pérdida de entrenamiento es el **endpoint error (EPE)**, la métrica estándar para flujo óptico: la **distancia euclidiana entre el vector de flujo predicho y el ground-truth**, promediada sobre todos los píxeles. Se entrena con una versión modificada de **Caffe**, usando **Adam** ($\beta_1=0.9$, $\beta_2=0.999$) porque converge más rápido que SGD con momento. Como en cierto sentido "cada píxel es una muestra de entrenamiento", se usan mini-batches pequeños de 8 pares. El learning rate arranca en $\lambda = 10^{-4}$ y se divide por 2 cada 100k iteraciones tras las primeras 300k. Con FlowNetCorr se observaron **gradientes explosivos** con $\lambda = 10^{-4}$, así que se arranca con un learning rate muy bajo ($10^{-6}$) que se incrementa hasta $10^{-4}$ en 10k iteraciones antes de seguir el schedule normal.

## 5. Dataset Flying Chairs

El obstáculo del ground-truth es explícito: a diferencia de los enfoques tradicionales, las redes neuronales necesitan datos con ground-truth no solo para ajustar unos pocos parámetros, sino para **aprender la tarea desde cero**, y obtener correspondencias verdaderas de píxeles en escenas del mundo real es difícil. Los datasets existentes son demasiado pequeños:

| Dataset | Pares de frames | Frames con ground-truth | Densidad del ground-truth |
|---|---|---|---|
| Middlebury | 72 | 8 | 100% |
| KITTI | 194 | 194 | ~50% |
| Sintel | 1.041 | 1.041 | 100% |
| **Flying Chairs** | **22.872** | **22.872** | **100%** |

**Middlebury** contiene solo 8 pares con desplazamientos muy pequeños (bajo 10 píxeles). **KITTI** es mayor (194 pares) e incluye desplazamientos grandes, pero solo un tipo muy especial de movimiento (escena rígida vista desde un observador móvil, con ground-truth por láser 3D y por ello disperso). **MPI Sintel** deriva su ground-truth de escenas renderizadas realistas, en versiones *Clean* y *Final* (esta última con motion blur y niebla); con 1.041 pares es el mayor disponible, pero **sigue siendo demasiado pequeño** para entrenar CNN grandes.

La solución es **Flying Chairs**: un dataset sintético construido aplicando **transformaciones afines** a imágenes de fondo de Flickr y a un conjunto público de modelos 3D de sillas renderizadas. Se recuperan 964 imágenes de Flickr ($1024\times768$) de las categorías "city", "landscape" y "mountain", cortadas en 4 cuadrantes para dar fondos de $512\times384$. Sobre ellas se superponen imágenes de sillas (809 tipos, 62 vistas por silla). Para generar movimiento se **muestrean aleatoriamente parámetros de transformación afín** para el fondo y para las sillas; las transformaciones de las sillas son relativas a la del fondo, lo que puede interpretarse como cámara y objetos moviéndose simultáneamente. Con los parámetros se renderiza la segunda imagen, el flujo óptico y las regiones de oclusión. Todos los parámetros (número, tipo, tamaño y posición de sillas; parámetros de transformación) se muestrean al azar, ajustando las distribuciones para que el **histograma de desplazamientos resultante se parezca al de Sintel**. El resultado son **22.872 pares** imagen–flujo. Los autores subrayan que el tamaño se eligió arbitrariamente y podría ser mayor.

Estas imágenes "tienen poco en común con el mundo real", pero se pueden generar en cantidades arbitrarias con propiedades a medida. El **data augmentation** resultó crucial pese al tamaño del dataset: se aplican transformaciones geométricas (traslación, rotación, escala), ruido gaussiano aditivo y cambios de brillo, contraste, gamma y color, todo en GPU y online. Como se busca aumentar también la variedad de campos de flujo, se aplica la misma transformación geométrica fuerte a ambas imágenes del par, más una **transformación relativa menor** entre ellas, adaptando el campo de flujo en consecuencia.

## 6. Resultados

Las redes se evalúan sobre Sintel, KITTI, Middlebury y el propio test set de Flying Chairs, con la métrica de **endpoint error promedio** (en píxeles; menor es mejor). Los hallazgos principales:

- **Generalización sin fine-tuning.** Las redes entrenadas solo sobre los datos no realistas de Flying Chairs "rinden muy bien" sobre datasets de flujo real, superando por ejemplo al conocido método **LDOF**. Esto valida la tesis central del paper: los datos sintéticos artificiales bastan para aprender flujo óptico que transfiere a escenas naturales.
- **Con fine-tuning (+ft) en Sintel** las redes superan al método de tiempo real competidor **EPPM** en Sintel Final y KITTI, siendo el doble de rápidas. FlowNetS+ft+v llega a estar a la par de DeepFlow en Sintel Final.
- **FlowNetC vs FlowNetS.** FlowNetC (con correlación) es mejor que FlowNetS en Sintel Clean y en Flying Chairs, pero la situación se invierte en Sintel Final: FlowNetS generaliza mejor a las condiciones difíciles (motion blur, niebla) que Flying Chairs no incluye. Esto sugiere que **FlowNetC se sobreajusta ligeramente** al tipo de datos de entrenamiento. En KITIT, con fuertes transformaciones proyectivas muy distintas a las vistas en entrenamiento, FlowNetS también supera a FlowNetC.
- **En Flying Chairs, las redes baten el estado del arte**, incluyendo DeepFlow y EpicFlow. Es además el único dataset donde el refinamiento variacional **empeora** los resultados: las redes "ya lo hacen mejor que el refinamiento variacional", lo que indica que con datos de entrenamiento más realistas podrían rendir aún mejor en otros datasets.
- **Velocidad.** Aunque las tasas de error están por debajo del estado del arte absoluto, FlowNet es **el mejor entre los métodos de tiempo real**, prediciendo flujo a hasta 10 pares de imágenes por segundo a resolución completa de Sintel (tiempos por frame de ~0,08 s en GPU para FlowNetS, frente a los 16–65 s en CPU de EpicFlow, DeepFlow o LDOF).
- **Detalle fino.** Cualitativamente, las redes suelen **preservar mejor los detalles finos y objetos pequeños con grandes desplazamientos** que EpicFlow, aunque su EPE sea peor por ruido en regiones de fondo grandes y suaves (parcialmente compensable con refinamiento variacional).

Una ablación confirma el valor de Flying Chairs: una red entrenada solo en Sintel tiene EPE ~1 píxel mayor que la entrenada en Flying Chairs y afinada en Sintel, y quitar el data augmentation eleva el EPE en ~2 píxeles.

## 7. Limitaciones

- **Grandes desplazamientos.** FlowNetC tiene más problemas con desplazamientos muy grandes: para píxeles con desplazamientos de al menos 40 px, FlowNetS+ft logra un error s40+ de 43,3 px frente a 48 px de FlowNetC+ft. La causa es que el **desplazamiento máximo de la capa de correlación** ($d=20$) no permite predecir movimientos muy grandes; ampliar ese rango es posible pero a costa de eficiencia computacional.
- **Precisión absoluta.** Las tasas de error de FlowNet aún están **por debajo del estado del arte** de los métodos clásicos; su ventaja está en la velocidad (tiempo real) y la preservación de detalle, no en el EPE absoluto en datasets realistas.
- **Ruido en el campo.** La salida cruda de las redes es ruidosa y no suave, especialmente en regiones de fondo grandes, lo que penaliza el EPE (métrica que favorece soluciones sobre-suavizadas). Requiere refinamiento variacional para competir en suavidad.
- **Dependencia del refinamiento externo.** Para obtener campos suaves y subpíxel-precisos en datos realistas, todavía se recurre a un **método variacional clásico** de postprocesamiento (+v).
- **Realismo de los datos.** Flying Chairs solo modela movimientos afines de objetos rígidos sintéticos; no incluye motion blur, niebla ni deformaciones no rígidas, lo que limita la generalización a condiciones ausentes del entrenamiento. Los autores anticipan que datos más realistas mejorarían el desempeño (lo que ocurrió con FlowNet 2.0).

## 8. Conexión con la Clase 36 (Introduction to Video Analysis)

El flujo óptico es el mecanismo canónico para **codificar el movimiento** entre frames, y como tal es tema central de la clase. FlowNet es la pieza que traslada ese cómputo del régimen clásico (variacional, hecho a mano) al régimen de aprendizaje profundo, y conviene presentarlo contrastando ambos paradigmas.

| Eje | Variacional clásico (Horn–Schunck y sucesores) | FlowNet (deep learning) |
|---|---|---|
| Cómo obtiene el flujo | Minimiza una energía (datos + suavidad) por optimización | Predicción directa por una CNN entrenada |
| Parámetros | Fijados manualmente, sin aprendizaje | Aprendidos end-to-end desde datos |
| Matching | Descriptores + interpolación hechos a mano | Aprendido, o capa de correlación explícita |
| Velocidad | Segundos por frame (CPU) | ~0,08–0,2 s por frame (GPU), 5–10 fps |
| Dependencia de datos | No requiere ground-truth de entrenamiento | Requiere ground-truth masivo (Flying Chairs) |

Por qué importa el flujo óptico en análisis de video, y dónde encaja FlowNet:

- **Redes two-stream para reconocimiento de acciones.** La arquitectura two-stream (Simonyan & Zisserman) separa un stream espacial (apariencia, sobre frames RGB) de un stream **temporal que consume flujo óptico** apilado como entrada de movimiento. Un estimador de flujo rápido y aprendible como FlowNet es justo lo que permite alimentar ese stream temporal de forma eficiente, y su naturaleza end-to-end abre la puerta a integrarlo dentro de la propia red de reconocimiento.
- **Video Object Tracking (VOT) y segmentación de video.** El flujo denso entre frames propaga máscaras, cajas o etiquetas de un frame al siguiente y da una señal de movimiento robusta para trackers; FlowNet lo entrega en tiempo casi real.
- **Bisagra histórica.** FlowNet es el punto donde el análisis de movimiento en video adopta el paradigma que ya había transformado el reconocimiento, y encabeza la línea FlowNet 2.0 → PWC-Net → RAFT que define el estado del arte actual.

**Enlaces internos:**

- Clase: [/clases/clase-36](/clases/clase-36) — Introduction to Video Analysis (flujo óptico, two-stream, VOT).
- Fundamento transversal: [/fundamentos/redes-convolucionales](/fundamentos/redes-convolucionales) — encoder-decoder, capas convolucionales y de refinamiento.

## Nota final: relevancia para video clínico

La estimación de movimiento por píxel que introduce FlowNet tiene aplicaciones directas en imagenología médica dinámica, donde el "video" es una secuencia clínica y el movimiento es la señal de interés. En **ecocardiografía**, el flujo óptico permite estimar el desplazamiento del miocardio entre frames para cuantificar la deformación (strain) y la función contráctil del corazón, tarea donde el ground-truth denso es prácticamente inobtenible en pacientes reales —exactamente el mismo problema que Flying Chairs resuelve con datos sintéticos, hoy replicado con simuladores de ultrasonido. En **radioterapia y adquisición de imágenes torácicas o abdominales**, la estimación de flujo modela el **movimiento respiratorio** para compensar el desplazamiento de órganos y tumores (gating respiratorio y registro deformable 4D-CT). Y en **análisis de marcha y rehabilitación**, el flujo óptico sobre video captura el movimiento de extremidades para caracterizar patrones patológicos sin marcadores físicos. En todos estos casos la lección de FlowNet es doble: una CNN puede aprender a estimar movimiento denso en tiempo casi real, y —crucial en el dominio clínico donde etiquetar es caro o imposible— un modelo entrenado sobre datos sintéticos bien diseñados puede generalizar a las secuencias reales del paciente.
