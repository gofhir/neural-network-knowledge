# Large-scale Video Classification with Convolutional Neural Networks — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** Andrej Karpathy (Google Research + Stanford), George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar (Google Research), Li Fei-Fei (Stanford).
- **Venue:** CVPR 2014. Sitio: `cs.stanford.edu/people/karpathy/deepvideo`.
- **Dataset presentado:** **Sports-1M** — 1 000 000 de videos de YouTube, 487 clases de deportes.
- **Modelo estrella:** ninguno, y eso es parte del mensaje. Es un **estudio comparativo** de cuatro patrones de conectividad temporal (Single Frame, Early Fusion, Late Fusion, Slow Fusion) más una optimización arquitectónica (**multiresolución fovea + context**).

| Magnitud | Valor |
|---|---|
| Videos / clases | 1 000 000 / 487 (taxonomía jerárquica curada a mano) |
| Videos por clase · multi-etiqueta | 1 000–3 000 · ~5% |
| Split · test set | 70/10/20 · 200 000 videos, 4 000 000 clips |
| Duración media de video | 5 min 36 s |
| Baseline handcrafted (histogramas + red neuronal) | 55.3% Video Hit@1 |
| Single-Frame | **59.3%** / 77.7% Hit@5 |
| Slow Fusion (mejor individual) | **60.9%** / 80.2% Hit@5 |
| Ensamble de las cuatro CNN | 63.9% / 82.4% |
| UCF-101: fine-tune top 3 capas vs. desde cero | **65.4%** vs. 41.3% |
| Tiempo de entrenamiento | ~1 mes en clúster |

El resultado de época no es el máximo, es **la brecha que no apareció**: pasar del modelo que solo ve un fotograma estático (59.3%) al mejor modelo espacio-temporal (60.9%) rinde **1.6 puntos**, mientras que pasar del baseline artesanal a cualquier CNN rinde entre 4 y 8. El abstract lo admite: "*only a surprisingly modest improvement compared to single-frame models*". Ese hallazgo negativo es el legado real del paper y el punto de partida argumentativo de Two-Stream (2014), C3D (2015) e I3D (2017).

Para la **Clase 38** este paper es **el origen histórico del primer eslabón** de la escalera: *CNN2D + agrupación temporal*. Aquí ese enfoque se propone por primera vez a escala, se mide en serio, y se descubre que su techo es incómodamente bajo.

## 2. Contexto: por qué en 2014 el video seguía atrás de las imágenes

En 2014 ya había consenso en imágenes: los features de una red ImageNet, clasificados con un SVM y **sin fine-tuning**, daban estado del arte en muchos datasets. En video, no. Los autores diagnostican tres cuellos de botella distintos, cada uno con solución distinta.

**(a) Ausencia de datasets a escala.** KTH, Weizmann, UCF Sports, IXMAS, Hollywood 2 y UCF-50 tenían "hasta unos pocos miles de clips y hasta unas pocas decenas de clases". Los más grandes eran **CCV (9 317 videos, 20 clases)** y el recién publicado **UCF-101 (13 320 videos, 101 clases)**, empequeñecidos por los datasets de imagen. Y no era pereza: "*los videos son significativamente más difíciles de recolectar, anotar y almacenar*". Como todas las aplicaciones exitosas de CNN en imágenes compartían tener un training set grande, los autores especulan que el estancamiento en video era **parcialmente atribuible a la falta de benchmarks a gran escala**. Es la hipótesis central del paper.

**(b) Costo computacional.** Las CNN de la época tardaban "del orden de semanas" incluso en las GPU más rápidas, y extender la conectividad en el tiempo agrava el problema mecánicamente: si un clip son 10 fotogramas, la primera capa hace 10× el trabajo. Sin poder experimentar, no se encuentra la buena arquitectura.

**(c) Falta de un patrón de conectividad temporal canónico.** En imágenes, convolución 2D con parameter sharing y max pooling era la respuesta probada; en video no había equivalente. Existían trabajos que trataban espacio y tiempo como dimensiones equivalentes (Baccouche et al. 2011; Ji et al. 2013, antecesor directo de C3D) y esquemas no supervisados sobre Convolutional Gated RBM (Taylor et al. 2010) e Independent Subspace Analysis (Le et al. 2011), pero nadie había comparado las alternativas sobre el mismo dataset, backbone y protocolo. El paper considera esas extensiones 3D "*solo una de las generalizaciones posibles*": ese es el hueco que viene a llenar.

## 3. Contribución central

Un aporte por cuello de botella:

1. **Sports-1M.** 1 millón de videos de YouTube, 487 clases de deportes, liberado a la comunidad.
2. **Estudio sistemático de fusión temporal.** Cuatro patrones de conectividad sobre la misma red base, dataset y protocolo. Es el corazón intelectual, con las preguntas planteadas con precisión: *¿qué patrón de conectividad temporal aprovecha mejor el movimiento local? ¿Cuánto mejora el desempeño global?*
3. **Arquitectura multiresolución (fovea + context).** Reduce la dimensionalidad de entrada a la mitad y acelera **2–4×** sin costo en accuracy.

Un cuarto aporte implícito y de gran impacto: la primera demostración de que **pre-entrenar en video a gran escala y transferir** funciona (UCF-101 de 41.3% a 65.4%) — el mismo experimento que I3D repetiría con Kinetics, con resultados mucho mejores.

## 4. El dataset Sports-1M

Sports-1M es un intercambio deliberado de **calidad de etiqueta por escala**.

**Taxonomía.** 487 clases en jerarquía curada manualmente, con nodos internos como *Aquatic Sports*, *Team Sports*, *Winter Sports*, *Ball Sports*, *Combat Sports* y *Sports with Animals*, que se vuelve **fine-grained en las hojas**: **6 tipos de bowling, 7 de fútbol americano y 23 de billar**. Esto explica buena parte del error — la matriz de confusión muestra que la mayoría ocurre entre clases fine-grained, y los cinco pares más confundidos (*deer hunting* vs. *hunting*, *hiking* vs. *backpacking*, *powered paragliding* vs. *paragliding*, *sledding* vs. *toboggan*, *bujinkan* vs. *ninjutsu*) son ambiguos incluso para un humano.

**Etiquetado débil y automático.** Las anotaciones "*se producen automáticamente analizando los metadatos de texto que rodean a los videos*"; no hay anotador humano en el loop. Los autores distinguen **dos niveles de ruido**, distinción más fina de lo que suele citarse:

- **Nivel de video:** la etiqueta puede estar mal porque el algoritmo de predicción de tags falla o porque la descripción no corresponde al contenido.
- **Nivel de fotograma:** incluso con la etiqueta correcta, el video varía enormemente cuadro a cuadro. El ejemplo del paper: un video etiquetado *soccer* puede contener tomas del marcador, entrevistas, presentadores de noticias, el público. Si el modelo consume clips de medio segundo al azar, una fracción no trivial de sus ejemplos es un locutor en un estudio etiquetado "fútbol".

El paper **no reporta tasa de ruido medida** — importante no inventarla. Lo que sí mide es la **contaminación por duplicados**: procesaron el millón de videos con detección de casi-duplicados a nivel de fotograma y encontraron solo **1 755 videos de 1 000 000** con fracción significativa de fotogramas casi-duplicados. Argumentan que la fuga entre splits es improbable porque solo usan hasta **100 clips de medio segundo** por video sobre videos de 5 min 36 s promedio.

**Lo que el paper concluye:** las redes "*parecen aprender bien a pesar del ruido significativo de etiqueta*", incluyendo texto sobreimpreso, efectos, cortes y logos, "*ninguno de los cuales intentamos filtrar explícitamente*" — robustez que anticipa la lógica del pre-entrenamiento débilmente supervisado a escala web.

**Lo que no dice y se volvió evidente:** Sports-1M mide en gran medida **reconocimiento de escena deportiva**, no de acción. Una piscina implica natación; un tatami, artes marciales; un green, golf. Es exactamente el dataset donde un modelo de un solo fotograma debería rendir bien — hay que tenerlo presente al leer la sección 9, porque **parte del hallazgo es una propiedad del dataset, no de las arquitecturas**.

## 5. Las cuatro estrategias de conectividad temporal

El framing es clave: los videos varían mucho en extensión temporal y no se procesan bien con una arquitectura de tamaño fijo, así que el paper **trata cada video como una bolsa de clips cortos de tamaño fijo**. Toda la fusión temporal ocurre *dentro* de un clip; la agregación a nivel de video es un promedio posterior.

### 5.1. Single Frame (línea base)

AlexNet con entrada más pequeña: $170 \times 170 \times 3$ en vez de $224 \times 224 \times 3$. Con $C(d,f,s)$ = capa convolucional de $d$ filtros de tamaño $f \times f$ y stride $s$, $N$ = normalización, $P$ = pooling espacial no solapado de $2 \times 2$:

$$C(96,11,3)\text{-}N\text{-}P\text{-}C(256,5,1)\text{-}N\text{-}P\text{-}C(384,3,1)\text{-}C(384,3,1)\text{-}C(256,3,1)\text{-}P\text{-}FC(4096)\text{-}FC(4096)$$

Normalización con los parámetros de Krizhevsky et al.: $k=2$, $n=5$, $\alpha=10^{-4}$, $\beta=0.5$; softmax denso al final.

**Extensión temporal: $T=1$.** Captura solo apariencia estática y **cero** movimiento. Su rol es diagnóstico: cuantificar cuánto de la clasificación se explica por apariencia pura.

### 5.2. Early Fusion

Combina la ventana temporal completa **de inmediato, a nivel de píxel**, modificando solo los filtros de la primera capa convolucional para que sean de tamaño

$$11 \times 11 \times 3 \times T, \qquad T = 10$$

es decir $11 \times 11$ espacial, 3 canales de color y **10 fotogramas**, "aproximadamente un tercio de segundo" (implica ~30 fps). El resto de la red es idéntico al single-frame.

**Captura:** la conectividad directa y temprana a los píxeles permite "*detectar con precisión la dirección y velocidad del movimiento local*" — el análogo aprendido de un filtro de Gabor espacio-temporal.

**Gotcha:** tras la primera capa **toda la dimensión temporal ha colapsado**; el resto es puramente 2D sobre un mapa que ya integró el tiempo. El movimiento se resume una vez y nunca se vuelve a razonar sobre él.

### 5.3. Late Fusion

El extremo opuesto: **dos redes single-frame con parámetros compartidos** (hasta la última capa convolucional $C(256,3,1)$), aplicadas a dos fotogramas separados **15 fotogramas** (~medio segundo), fusionadas **en la primera capa fully connected**.

**Captura:** "*ninguna de las dos torres single-frame por sí sola puede detectar movimiento alguno, pero la primera capa fully connected puede computar características de movimiento global comparando las salidas de ambas torres*". La palabra clave es **global**: la FC ve dos descripciones de alto nivel separadas medio segundo e infiere cambio agregado, pero perdió la medición de velocidad y dirección local que era la fortaleza de Early Fusion. Compartir parámetros hace viable el esquema, pero impide que una torre se especialice en "antes" y otra en "después".

### 5.4. Slow Fusion

La más elaborada, y la que gana. "*Una mezcla balanceada entre los dos enfoques que fusiona lentamente la información temporal a lo largo de la red, de modo que las capas superiores accedan a información progresivamente más global tanto en la dimensión espacial como en la temporal*": replica en el eje temporal la lógica jerárquica que la CNN ya aplica en el espacial, **extendiendo en el tiempo la conectividad de todas las capas convolucionales**. Con un clip de **10 fotogramas**:

| Capa | Extensión temporal $T$ | Stride temporal | Respuestas en el tiempo |
|---|---|---|---|
| conv1 | 4 | 2 | 4 |
| conv2 | 2 | 2 | 2 |
| conv3 | 2 | 2 | 1 |

Con convolución *valid*: $(10-4)/2+1 = 4$, luego $(4-2)/2+1 = 2$, luego $1$. **La tercera capa convolucional tiene acceso a los 10 fotogramas de entrada.** Las columnas de la Figura 1 **comparten parámetros**, lo que mantiene el conteo de pesos manejable.

**Captura:** conv1 ve movimiento local fino (~0.13 s); conv3 ve la dinámica completa del clip (~0.33 s) pero construida composicionalmente desde primitivas locales. Es, conceptualmente, un C3D en miniatura — un año antes de C3D y tres antes de I3D.

### 5.5. Resumen comparativo

| Estrategia | Dónde fusiona | Extensión temporal | Qué captura | Qué pierde |
|---|---|---|---|---|
| Single Frame | nunca | $T=1$ | apariencia estática | todo el movimiento |
| Early Fusion | primera conv, a nivel de píxel | $T=10$ (~1/3 s) | dirección y velocidad locales, precisas | jerarquía temporal (colapsa en una capa) |
| Late Fusion | primera FC, sobre features altos | 2 fotogramas a 15 de distancia | cambio global entre dos instantes | movimiento local, velocidad, dirección |
| Slow Fusion | todas las conv, progresivamente | $4 \to 2 \to 2$ = 10 fotogramas | jerarquía espacio-temporal local → global | costo; sigue limitada a ~1/3 s |

## 6. La arquitectura multiresolución: fovea + context

**El problema.** El runtime limita la capacidad de experimentar, y las dos alternativas obvias fallan: reducir capas y neuronas "consistentemente baja el desempeño" (coincidiendo con Zeiler y Fergus), y entrenar en baja resolución mejora el tiempo pero "*el detalle de alta frecuencia resultó crítico para lograr buena accuracy*".

**La solución.** Dos streams a dos resoluciones, a partir de un clip de $178 \times 178$:

- **Context stream:** fotogramas **submuestreados a la mitad**, $89 \times 89$ del cuadro completo. Ve toda la escena, borrosa.
- **Fovea stream:** la **región central de $89 \times 89$ a resolución original**. Ve un tercio del área, nítido.

$$\frac{2 \times 89^2}{178^2} = \frac{15\,842}{31\,684} = \frac{1}{2}$$

La dimensionalidad total de entrada se reduce **a la mitad**, y de ahí sale la aceleración. Ambos streams usan la misma red que los modelos de cuadro completo, pero **se elimina la última capa de pooling** para que ambos terminen en $7 \times 7 \times 256$ — el mismo tamaño que produciría la red completa. Las activaciones se **concatenan** y alimentan la primera capa fully connected densa.

**El sesgo de centrado, admitido.** "*Notablemente, este diseño aprovecha el sesgo de cámara presente en muchos videos en línea, ya que el objeto de interés suele ocupar la región central*". La arquitectura funciona porque quienes filman en YouTube centran el sujeto: no es un principio de visión, es una regularidad estadística del corpus. Honestidad admirable y limitación evidente — la fovea fija **no es atención**: no aprende *dónde* mirar, asume que lo interesante está al medio. Vigilancia con múltiples actores o un hallazgo clínico en la periferia rompen el supuesto.

**Aceleración medida.** El paper reporta el rango genérico **2–4×** y estas cifras concretas:

| Modelo | Sin multires | Con multires | Speedup |
|---|---|---|---|
| Single-Frame | 6 clips/s | 21 clips/s | **3.5×** |
| Slow Fusion | 5 clips/s | 10 clips/s | **2×** |

Con la advertencia de que los speedups son "*en parte función de los detalles del particionado del modelo y de nuestra implementación*". El costo en accuracy es nulo y de hecho negativo: Single-Frame 59.3% → Single-Frame + Multires **60.0%**.

En la Figura 3, el **context stream aprende features de color y bajas frecuencias** y el **fovea stream filtros de alta frecuencia en escala de grises**: especialización que nadie programó y que evoca la retina, de ahí el nombre. Las ablaciones confirman que se necesitan ambos (Fovea Only 49.9%, Context Only 56.0%, contra 60.0% combinado), y que el context solo supere holgadamente a la fóvea sola es otra pista de que Sports-1M premia el reconocimiento de escena.

## 7. Detalles de entrenamiento

**Optimización.** **Downpour SGD** (Dean et al., NIPS 2012): SGD **asincrónico** con paralelismo de datos *y* de modelo, réplicas que empujan gradientes a un servidor de parámetros sharded sin sincronizarse, tolerando gradientes obsoletos. **10 a 50 réplicas** por modelo, cada modelo **particionado en 4 a 32 particiones**. Mini-batches de **32**, momento **0.9**, weight decay **0.0005**, learning rate inicial **$10^{-3}$** reducido **a mano** cuando el error de validación deja de mejorar. Sin batch normalization (llega en 2015), sin Adam, sin scheduler automático, y paralelismo asincrónico porque el all-reduce sincrónico sobre GPU aún no era la norma — I3D usaría 32–64 GPUs **síncronas**.

**Data augmentation.** Recorte a la región central, resize a $200 \times 200$, muestreo aleatorio de $170 \times 170$ y flip horizontal con probabilidad 50%. El detalle crítico: "*estos pasos se aplican de manera consistente a todos los fotogramas que forman parte del mismo clip*" — sin esa consistencia, el crop aleatorio por fotograma inyectaría movimiento espurio y destruiría exactamente la señal que el modelo intenta aprender. Se resta **117** a los píxeles crudos (la media del corpus). *Inconsistencia menor del paper:* la sección de modelos habla de $170 \times 170$ para cuadro completo y clip de $178 \times 178$ para multiresolución, mientras el pipeline de augmentation describe $170 \times 170$; no se aclara.

**Escala.** Un mes de entrenamiento; ~**5 clips/s** por réplica en cuadro completo y hasta **20** en multiresolución. Los autores estiman que 5 clips/s es "*aproximadamente 20 veces más lento que lo esperable de una GPU de alta gama*", compensado con 10–50 réplicas, y cifran el dataset de fotogramas muestreados en el orden de **50 millones de ejemplos**, con cada red viendo ~**500 millones de ejemplos** (≈10 épocas efectivas).

**Predicción a nivel de video**, deliberadamente lo más simple posible: (1) muestrear **20 clips al azar** y presentarlos individualmente; (2) propagar cada clip **4 veces** con distintos crops y flips, promediando; (3) **promediar las predicciones de clip** sobre la duración del video. Es **votación por promedio, sin ningún modelo de orden temporal a nivel de video**. Los autores lo reconocen y proponen **RNN** como trabajo futuro para "combinar predicciones de clip en predicciones globales" — literalmente el segundo eslabón de la Clase 38, anunciado en estas conclusiones.

**El baseline artesanal** no es un hombre de paja: **HOG**, **Texton** y **Cuboids** extraídos densamente *y* en puntos de interés dispersos, más features globales (Hue-Saturation, momentos de color, número de caras); cuantización k-means; histogramas con **spatial pyramid encoding** y **soft quantization**; vector de **25 000 dimensiones a nivel de video**; red multicapa con ReLU y softmax. Con una asimetría a su favor que el paper subraya: computa palabras visuales **densamente sobre todo el video**, mientras las CNN solo ven 20 clips al azar.

## 8. Resultados en Sports-1M

Test set de 200 000 videos y 4 000 000 clips. Hit@$k$ = fracción de muestras que contuvieron al menos una etiqueta ground truth en el top $k$.

| Modelo | Clip Hit@1 | Video Hit@1 | Video Hit@5 |
|---|---|---|---|
| Feature Histograms + Neural Net | — | 55.3 | — |
| Single-Frame | 41.1 | 59.3 | 77.7 |
| Single-Frame + Multires | 42.4 | 60.0 | 78.5 |
| Single-Frame Fovea Only | 30.0 | 49.9 | 72.8 |
| Single-Frame Context Only | 38.1 | 56.0 | 77.2 |
| Early Fusion | 38.9 | 57.7 | 76.8 |
| Late Fusion | 40.7 | 59.3 | 78.7 |
| **Slow Fusion** | **41.9** | **60.9** | **80.2** |
| CNN Average (Single+Early+Late+Slow) | 41.4 | **63.9** | **82.4** |

- **Las CNN superan consistente y significativamente al baseline**: 55.3 → 60.9 individual, → 63.9 en ensamble, y con desventaja de protocolo. Es el resultado *positivo*.
- **La variación entre arquitecturas CNN es "sorprendentemente insignificante"** (palabras del paper): 3.2 puntos entre la peor y la mejor variante con movimiento.
- **Early Fusion es la peor variante con movimiento (57.7)**, *por debajo* de Single Frame: el colapso temporal inmediato destruye más apariencia de la que aporta en movimiento.
- **Late Fusion empata exactamente a Single Frame en Video Hit@1** (59.3), mejorando solo en Hit@5. Duplicar el cómputo convolucional para comparar dos instantes no compra nada medible en top-1.
- **El ensamble sube 3 puntos** sobre la mejor individual: las arquitecturas cometen errores **parcialmente decorrelacionados**; hay señal complementaria que ninguna captura sola.
- **Clip Hit@1 (~41%) vs. Video Hit@1 (~60%):** los ~19 puntos son la ganancia pura de promediar 20 clips, y cuantifican el ruido de fotograma del dataset.

**Contribución del movimiento por clase (Tabla 2).** Diferencia de average precision entre Slow Fusion y Single-Frame. Gana el movimiento en *Juggling Club* (+0.12), *Pole Climbing* (+0.10), *Mountain Unicycling* (+0.08), *Tricking* y *Footbag* (+0.07), *Skipping Rope* y *Rope Climbing* (+0.06) — todas **actividades definidas por un patrón cíclico o de equilibrio**, donde un fotograma es genuinamente ambiguo. Pierde en *Short Track Motor Racing*, *Road Racing* y *Jeet Kune Do* (−0.07), *Paintball*, *Freeride*, *Cricket* y *Wrestling* (−0.06) y *Rally Cross* (−0.05). El paper nombra el patrón: "*las redes conscientes del movimiento son más propensas a rendir peor cuando hay movimiento de cámara presente*", con la hipótesis de que "*las CNN tienen dificultad para aprender invarianza completa a través de todos los ángulos y velocidades posibles de traslación y zoom de cámara*".

## 9. El hallazgo incómodo: Single Frame casi empata

$$\text{Single Frame} = 59.3\% \;\longrightarrow\; \text{Slow Fusion} = 60.9\%, \qquad \Delta = 1.6 \text{ puntos}$$

En Hit@5, 77.7 → 80.2 ($\Delta = 2.5$). En Clip Hit@1, 41.1 → 41.9 ($\Delta = 0.8$). Y concediéndole el multires al modelo estático (60.0%), la brecha queda en **0.9 puntos**. Traducido: **añadir toda la información de movimiento del clip, con la mejor de las cuatro estrategias, con un mes de cómputo en clúster sobre un millón de videos, compra menos de dos puntos**. Las conclusiones: "*sorprendentemente, encontramos que un modelo de un solo fotograma ya exhibe desempeño muy fuerte, lo que sugiere que las señales de movimiento local pueden no ser críticamente importantes, incluso para un dataset dinámico como Sports*".

### Las explicaciones que ofrece el paper

**(1) El movimiento local simplemente no es importante para esta tarea.** El contenido semántico de un video deportivo está mayoritariamente en la apariencia: escena, equipamiento, uniformes, geometría del campo. Cuando el paper agrega que "*estos beneficios son sorprendentemente robustos a los detalles de la conectividad de las arquitecturas en el tiempo*", está diciendo que la conectividad temporal casi no es un eje de diseño relevante *para Sports-1M*. Ese paréntesis es lo que las generaciones siguientes atacarían.

**(2) El movimiento de cámara contamina la señal.** La teoría alternativa que los autores prefieren: "*un tratamiento más cuidadoso del movimiento de cámara puede ser necesario, por ejemplo extrayendo features en el sistema de coordenadas local de un punto rastreado*", citando las **dense trajectories** de Wang et al. El movimiento en el plano de imagen es la superposición del movimiento del actor y del observador, y una CNN feedforward sobre píxeles crudos debe aprender invarianza al segundo mientras extrae el primero, con supervisión débil y sin ningún sesgo inductivo. La Tabla 2 lo respalda: todas las clases donde el movimiento perjudica involucran cámara en movimiento. Corregirlo "*requiere cambios significativos a la arquitectura CNN que dejamos para trabajo futuro*".

**(3) La agregación temporal a nivel de video es demasiado pobre.** El paper no la etiqueta como explicación pero la plantea en el mismo párrafo de trabajo futuro: la fusión temporal solo opera dentro de ~1/3 de segundo, y a nivel de video la única agregación es el **promedio de 20 clips al azar**, matemáticamente invariante al orden — un modelo así no puede, en principio, distinguir una acción de su reverso temporal. De ahí la propuesta de "*explorar redes recurrentes*". Súmese el ruido de etiqueta a nivel de fotograma, que diluye la señal de movimiento.

### Por qué se volvió el argumento central de la generación siguiente

Un resultado negativo bien medido, sobre un millón de videos, es un mandato de investigación. Si aprender movimiento *implícitamente* casi no ayuda, quedan dos salidas, y cada una originó un linaje:

- **Dar el movimiento explícitamente, ya calculado.** La respuesta de **Two-Stream ConvNets** (Simonyan y Zisserman, NIPS 2014, el mismo año): se le **entrega flujo óptico pre-computado** en un stream dedicado con su propia red inicializada desde ImageNet, y además **estabilizable** restando el movimiento medio de cámara. El diagnóstico de Karpathy et al. es literalmente su argumento de venta. Two-Stream saltó a la banda de los 88% en UCF-101, un abismo respecto del 65.4% de este paper.
- **Dar a la red la maquinaria arquitectónica adecuada, con suficiente extensión temporal.** El linaje **C3D → I3D**: convoluciones 3D en toda la profundidad, con campos receptivos temporales largos. Slow Fusion es el ancestro de la idea, pero operaba sobre 10 fotogramas con una red poco profunda entrenada desde cero; I3D entrena sobre **snippets de 64 fotogramas** (2.56 s, casi 8× la huella de Slow Fusion) con una Inception-v1 profunda **inicializada desde ImageNet**, y atribuye a esa alta resolución temporal su superior transferibilidad. Y sigue encontrando valioso el stream de flujo óptico: la lección de Karpathy et al. sobrevivió tres generaciones.

Hay una tercera lectura, que la historia validó: **el dataset era el problema**. Ningún estudio de arquitectura puede detectar la utilidad del movimiento en un benchmark donde el movimiento no es discriminativo. Kinetics (2017) se construyó explícitamente sobre **acciones humanas** con clases que exigen razonamiento temporal, y ahí sí las arquitecturas se ordenaron por mérito.

## 10. Transferencia a UCF-101: el resultado que incomoda

Los autores razonan que las CNN aprenden features genéricos abajo (bordes, formas locales) y específicos arriba, y prueban tres puntos de corte sobre la red **Slow Fusion** (la mejor en Sports-1M). UCF-101: 13 320 videos, 101 categorías en 5 grupos — *Human-Object Interaction*, *Body-Motion Only*, *Human-Human Interaction*, *Playing Musical Instruments* y *Sports*. Protocolo: 50 clips por video, mismo esquema de evaluación que Sports, promediado sobre los **3 folds sugeridos**.

- **Fine-tune top layer.** CNN como extractor fijo; clasificador sobre la última capa de 4 096 dimensiones, con dropout muy agresivo: "*tan poco como 10% de probabilidad de mantener cada unidad activa*".
- **Fine-tune top 3 layers.** Se reentrenan también **ambas capas fully connected**, partiendo de la Sports CNN entrenada, con el mismo dropout al 10% de retención.
- **Fine-tune all layers.** Todos los parámetros, incluidas las convolucionales. **Train from scratch:** la red completa solo con UCF-101.

| Modelo | 3-fold Accuracy |
|---|---|
| Soomro et al. (baseline del paper de UCF-101) | 43.9% |
| Feature Histograms + Neural Net | 59.0% |
| Train from scratch | 41.3% |
| Fine-tune top layer | 64.1% |
| **Fine-tune top 3 layers** | **65.4%** |
| Fine-tune all layers | 62.2% |

**La U invertida es el hallazgo.** Congelar demasiado no es óptimo porque "*los features de alto nivel son quizás demasiado específicos de deportes*"; descongelar todo tampoco, "*probablemente debido a overfitting*" (13 320 videos contra decenas de millones de parámetros). El óptimo es reentrenar las dos FC. Y entrenar desde cero "*conduce consistentemente a overfitting masivo y desempeño lamentable*": **41.3%, peor incluso que el baseline de 2012 de los propios autores de UCF-101**.

**Desglose por grupo (mAP, Slow Fusion):**

| Grupo | Desde cero | Fine-tune top 3 | Fine-tune top |
|---|---|---|---|
| Human-Object Interaction | 0.26 | 0.55 | 0.52 |
| Body-Motion Only | 0.32 | 0.57 | 0.52 |
| Human-Human Interaction | 0.40 | 0.68 | 0.65 |
| Playing Musical Instruments | 0.42 | 0.65 | 0.46 |
| Sports | 0.57 | 0.79 | **0.80** |
| **Todos los grupos** | 0.44 | **0.68** | 0.66 |

Buena parte del desempeño se explica por el grupo **Sports** (0.80 mAP), esperable. Más interesante: **la ganancia de "top" a "top 3" viene casi enteramente de las categorías no deportivas** — Sports apenas cae de 0.80 a 0.79 mientras *Playing Musical Instruments* salta de 0.46 a 0.65. Las capas FC de la Sports CNN codifican una representación tan sesgada hacia deportes que reentrenarlas es indispensable para transferir a *tocar la flauta*. Los autores destacan que el desempeño en grupos no deportivos es impresionante considerando que "*la única manera de observar este tipo de fotogramas en los datos de entrenamiento es debido al ruido de etiqueta*": el ruido actuó como diversificación involuntaria del dominio.

**Por qué incomoda.** El paper gana contra los baselines que tiene a mano, pero **no compara contra el estado del arte handcrafted de la época en UCF-101**, y ese silencio es significativo. Las **improved Dense Trajectories** (iDT) con codificación Fisher Vector estaban en la banda alta de los 80: Feichtenhofer et al. (CVPR 2016) tabulan "IDT + higher dimensional FV" (Peng et al., 2014) en **87.9%** en UCF-101 — cifra **externa a este paper**, que no debe atribuírsele, pero que es la que lo contextualiza. La conclusión es dura: una CNN pre-entrenada sobre **un millón de videos** y transferida cuidadosamente alcanza **65.4%**, mientras trayectorias densas con HOG/HOF/MBH, Fisher Vectors y SVM llegaban a **~88%**. Más de 20 puntos a favor de lo artesanal: en 2014 el deep learning ya había arrasado en imágenes, y en video **todavía perdía por goleada**. Y lo que los features artesanales tenían y la CNN no era precisamente **modelado explícito y compensado del movimiento** — las dense trajectories extraen descriptores en el sistema de coordenadas local del punto rastreado, neutralizando el movimiento de cámara. Exactamente el remedio que el propio paper señala y deja pendiente.

**Dos notas al margen.** *Discrepancia de cifras:* el abstract reporta la transferencia como "**63.3% up from 43.9%**", mientras la introducción y la Tabla 3 reportan "**65.4%, up from 41.3%**"; el 63.3% no aparece en ninguna tabla, así que al citar hay que usar los de la Tabla 3. *Advertencia metodológica:* los autores intentaron obtener los IDs de YouTube de UCF-101 para verificar solapamiento con Sports-1M, sin éxito — "*no podemos garantizar que el dataset Sports-1M no tenga solapamiento con UCF-101*".

## 11. Limitaciones

**Reconocidas por los autores:** ningún tratamiento del **movimiento de cámara** (la que ellos señalan como más importante, con la solución identificada y explícitamente diferida); agregación a nivel de video por **promedio simple**, con RNN propuesta a futuro; **cobertura de categorías estrecha** — solo deportes, "*esperamos incorporar categorías más amplias para obtener features más potentes y genéricos*", reconocimiento anticipado del problema que Kinetics resolvería; **ruido de etiqueta no filtrado**; **posible solapamiento Sports-1M / UCF-101**, imposible de verificar; y **speedups dependientes de la implementación**.

**Evidentes en retrospectiva:**

- **La extensión temporal es minúscula.** Máximo 10 fotogramas (~1/3 s), clips de medio segundo. Muchas acciones no se distinguen en ese horizonte; I3D usa 2.56 s y atribuye a eso su ventaja.
- **La red es poco profunda y no hereda ImageNet.** Es una AlexNet entrenada desde cero sobre video. El paper prueba que pre-entrenar en video y transferir funciona, pero nunca prueba lo inverso: **partir de una red pre-entrenada en ImageNet**. Con el diario del lunes, es la palanca más grande que dejó sin tirar — la que Two-Stream usa en ambos streams y la que I3D formaliza con el *boring-video fixed point*.
- **Sin batch normalization, sin residuales, con learning rate schedule manual.**
- **La fóvea fija no es atención**: asume que el sujeto está al centro.
- **El promedio de clips es invariante al orden.** Si $p_i$ es la predicción del clip $i$, $\frac{1}{N}\sum_i p_i$ es invariante a cualquier permutación. Consecuencia formal, no accidente.
- **El estudio de fusión temporal quedó confundido con la calidad del dataset.** "La conectividad temporal casi no importa" es verdadero en Sports-1M y falso en Kinetics. El paper mide honestamente lo que puede medir; el problema es que se citó como verdad general sobre video.

## 12. Impacto y legado

**Por qué define el eslabón "CNN2D + temporal pooling".** Es la primera demostración a escala de que se puede clasificar video ejecutando una CNN de imagen sobre fotogramas y agregando en el tiempo, y —más importante— la primera **medición seria de cuánto rinde eso**. El modelo Single-Frame con votación por promedio de 20 clips **es** CNN2D + temporal pooling, y su número fija el techo del eslabón. Nótese que **Slow Fusion pertenece al mismo linaje como su versión más ambiciosa** y es a la vez el ancestro reconocible de las 3D ConvNets: que ganara solo 1.6 puntos es, en retrospectiva, un artefacto de la poca profundidad, la corta extensión temporal y la falta de pre-entrenamiento de imagen — no una refutación de la convolución 3D, como C3D e I3D demostrarían.

**Qué heredó Two-Stream.** Aparece el mismo año y su motivación es en buena parte el diagnóstico de Karpathy et al.: si el aprendizaje implícito de movimiento no funciona —y menos con cámara en movimiento—, hay que **entregar el movimiento ya calculado**. La cadena causal es directa: *hallazgo negativo → flujo óptico explícito*, y sobrevivió hasta I3D, que en 2017 todavía encontraba valioso el segundo stream con un argumento afín (el flujo óptico es un cómputo iterativo que una red feedforward no reproduce fácilmente).

**Por qué Sports-1M no fue el ImageNet del video.** Es el legado más instructivo, porque es un fracaso. Tenía la escala pero le faltaba todo lo demás:

| Propiedad | Sports-1M | Kinetics |
|---|---|---|
| Etiquetas | automáticas desde metadatos de texto, ruidosas en dos niveles | curadas con verificación humana |
| Dominio | solo deportes | acciones humanas amplias |
| Recorte temporal | videos completos de 5 min 36 s con marcadores, entrevistas, público | clips de ~10 s recortados en la acción |
| Discriminatividad temporal | baja: la escena estática casi basta | alta por diseño |

I3D reporta que su ventaja sobre C3D es grande **aunque C3D se entrenó con más videos** (1M de Sports-1M más un dataset interno) y en ensamble con iDT, atribuyéndolo a la **mejor calidad de Kinetics** y a ser mejor arquitectura. La comunidad concluyó explícitamente que Sports-1M no bastaba, y la lección es que **un ImageNet de video necesita curación y recorte temporal, no solo volumen**. Aun así la contribución fundacional queda en pie: fue el primer dataset de video a escala web, el corpus de pre-entrenamiento de C3D, y estableció que pre-entrenar en video y transferir era viable.

## 13. Conexión con la Clase 38 y el linaje del video

La Clase 38 recorre **CNN2D + temporal pooling → CNN2D + RNN → Two-Stream → C3D → I3D** con foco en el pre-entrenamiento. Este paper es el punto de partida y **anticipa explícitamente los cuatro escalones siguientes**: es CNN2D + temporal pooling (Single-Frame más promedio de 20 clips); *propone* CNN2D + RNN en sus conclusiones; *motiva* Two-Stream con su diagnóstico del movimiento de cámara; *prototipa* la convolución 3D en Slow Fusion, un año antes de C3D; y *deja pendiente* I3D, porque nunca prueba inicializar desde ImageNet.

**Sobre pre-entrenamiento**, la distinción central de la clase: este paper pre-entrena **en video, desde cero**, sin usar ImageNet en ningún momento. Es el experimento complementario y opuesto al de I3D:

| | Karpathy et al. 2014 | I3D 2017 |
|---|---|---|
| Pre-entrenamiento de imagen | ninguno | ImageNet (heredado por inflado 2D→3D) |
| Pre-entrenamiento de video | Sports-1M, 1M videos, etiquetas ruidosas | Kinetics, 240k videos, curado |
| Extensión temporal en entrenamiento | 10 fotogramas (~0.33 s) | 64 fotogramas (2.56 s) |
| Profundidad | ~8 capas estilo AlexNet | Inception-v1 inflada |
| Flujo óptico explícito | no | sí (TV-L1) |
| UCF-101 (3 splits) | **65.4%** | **98.0%** |

Los 32.6 puntos entre esas dos filas son la historia completa de la Clase 38, y el orden de las palancas que los explican es instructivo: **pre-entrenamiento de imagen + profundidad + extensión temporal + movimiento explícito + calidad del dataset**. Este paper tiene solo el volumen de datos, y el volumen solo no alcanzó.

**Sobre las ventajas y desventajas que enumera la profesora para CNN2D + temporal pooling**, este paper es la evidencia primaria de las cuatro:

- **Fácil de implementar** — es AlexNet más un promedio: `mean([net(frame) for frame in clips])`. Sin kernels 3D, sin extractor de flujo óptico externo, sin RNN con estado.
- **No costoso** — 6 clips/s por réplica, hasta 21 con multiresolución, contra 5 y 10 de Slow Fusion y contra las 64 GPUs con que se entrenó I3D.
- **No aprovecha la información temporal** — demostrado formalmente, no solo empíricamente: el promedio de predicciones de clip es invariante a permutaciones.
- **Rinde deficiente** — 59.3% contra 60.9% de la mejor variante espacio-temporal del mismo paper parece poco, pero **65.4% en UCF-101 contra ~88% artesanal de la época y 98.0% de I3D tres años después** muestra la magnitud real del déficit.

La lección de fondo, la que justifica leerlo entero y no solo citarlo: **un resultado negativo, medido con rigor y a la escala correcta, es más productivo para un campo que un resultado positivo marginal**. Karpathy et al. podrían haber titulado "las CNN funcionan en video" con el 63.9% del ensamble contra el 55.3% del baseline. En vez de eso escribieron en el abstract que la mejora sobre el modelo de un solo fotograma era "sorprendentemente modesta", y esa frase organizó los siguientes cinco años de investigación en reconocimiento de acciones.

---

**Nota final — relevancia para pipelines con datos escasos.** La U invertida de la Tabla 3 es el resultado más reutilizable: con dataset objetivo pequeño y dominio de pre-entrenamiento estrecho, **congelar todo desperdicia capacidad de adaptación y descongelar todo produce overfitting masivo**; el óptimo suele estar en reentrenar las capas densas superiores con dropout agresivo, y cuanto más se aleja el dominio objetivo del de pre-entrenamiento, más capas hay que descongelar. Y la lección de Sports-1M frente a Kinetics es la más transferible a cualquier proyecto con datos reales: **un corpus grande con etiquetas ruidosas y sin recorte del evento de interés no sustituye a un corpus más chico pero curado y alineado con la tarea**. La escala sin curación tiene un techo, y este paper es la medición de ese techo.
