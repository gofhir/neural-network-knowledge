---
title: "Sports-1M: Large-scale Video Classification with CNNs (2014)"
weight: 419
math: true
---

{{< paper-card
    title="Large-scale Video Classification with Convolutional Neural Networks"
    authors="Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, Li Fei-Fei (Google Research, Stanford)"
    year="2014"
    venue="CVPR 2014"
    pdf="/papers/large-scale-video-karpathy-2014.pdf" >}}
Este paper no propone un modelo estrella, y eso es parte del mensaje: es **el primer estudio comparativo a escala de cómo conectar una CNN en el tiempo**, junto con el dataset que lo hizo posible. **Sports-1M** son 1.000.000 de videos de YouTube etiquetados automáticamente en **487 clases de deportes**. Sobre él los autores comparan cuatro patrones de conectividad temporal —*Single Frame*, *Early Fusion*, *Late Fusion* y *Slow Fusion*— con la misma red base y el mismo protocolo, más una red **multiresolución fovea + context** que acelera entre 2× y 4× sin costo en precisión. Las CNN superan claramente al baseline artesanal (55,3% → 60,9% Video Hit@1; 63,9% en ensamble), pero el resultado que quedó en la historia es el que *no* apareció: el modelo que solo ve **un fotograma estático** alcanza 59,3%, apenas 1,6 puntos por debajo del mejor modelo espacio-temporal, "*una mejora sorprendentemente modesta*" según el propio abstract. Ese hallazgo negativo, medido sobre un millón de videos y un mes de cómputo en clúster, se volvió el punto de partida argumentativo de las [two-stream ConvNets](/papers/two-stream-simonyan-2014), de [C3D](/papers/c3d-tran-2015) y de [I3D](/papers/i3d-carreira-2017). El paper demuestra además, por primera vez, que pre-entrenar en video a gran escala y transferir funciona: [UCF-101](/papers/ucf101-soomro-2012) pasa de 41,3% desde cero a 65,4% con fine-tuning. Para la [Clase 38](/clases/clase-38) es el origen histórico del primer eslabón de la escalera del video: *CNN2D + agrupación temporal*.
{{< /paper-card >}}

---

## Contexto: por qué el video seguía atrás de las imágenes

En 2014 el consenso en imágenes estaba cerrado: los features de una [AlexNet](/papers/alexnet-krizhevsky-2012) entrenada en ImageNet, clasificados con un SVM y **sin fine-tuning**, daban estado del arte en muchos datasets. En video no ocurría nada parecido, y los autores diagnostican tres cuellos de botella.

**Ausencia de datasets a escala.** KTH, Weizmann, UCF Sports y UCF-50 tenían "hasta unos pocos miles de clips y hasta unas pocas decenas de clases"; los mayores eran CCV (9.317 videos, 20 clases) y el recién publicado [UCF-101](/papers/ucf101-soomro-2012) (13.320 videos, 101 clases). Como todas las aplicaciones exitosas de CNN en imágenes compartían tener un training set grande, los autores especulan que el estancamiento en video era **parcialmente atribuible a la falta de benchmarks a gran escala**: esa es la hipótesis central del trabajo.

**Costo computacional.** Entrenar tardaba "del orden de semanas" incluso en las GPU más rápidas, y extender la conectividad temporal agrava el problema mecánicamente: con clips de 10 fotogramas, la primera capa hace 10× el trabajo.

**Falta de un patrón de conectividad temporal canónico.** En imágenes, convolución 2D con *parameter sharing* y max pooling era la respuesta probada; en video no había equivalente. Existían extensiones que trataban espacio y tiempo como dimensiones equivalentes (Ji et al. 2013, antecesor directo de [C3D](/papers/c3d-tran-2015)), pero el paper las considera "*solo una de las generalizaciones posibles*" y nadie las había comparado sobre el mismo dataset, backbone y protocolo.

---

## Sports-1M: un millón de videos con etiquetas débiles

Sports-1M es un intercambio deliberado de **calidad de etiqueta por escala**.

| Magnitud | Valor |
|---|---|
| Videos / clases | 1.000.000 / 487 |
| Videos por clase · multi-etiqueta | 1.000–3.000 · ~5% |
| Split · test set | 70/10/20 · 200.000 videos, 4.000.000 clips |
| Duración media de video | 5 min 36 s |
| Videos con casi-duplicados detectados | 1.755 de 1.000.000 |

La taxonomía es una jerarquía curada a mano (*Aquatic Sports*, *Team Sports*, *Winter Sports*, *Ball Sports*, *Combat Sports*, *Sports with Animals*) que se vuelve **fine-grained en las hojas**: 6 tipos de bowling, 7 de fútbol americano y 23 de billar. Ahí está buena parte del error: los cinco pares más confundidos (*deer hunting* vs. *hunting*, *hiking* vs. *backpacking*, *powered paragliding* vs. *paragliding*, *sledding* vs. *toboggan*, *bujinkan* vs. *ninjutsu*) son ambiguos incluso para un humano.

Las anotaciones "*se producen automáticamente analizando los metadatos de texto que rodean a los videos*", sin anotador humano en el loop, y los autores distinguen **dos niveles de ruido**. A **nivel de video** la etiqueta puede fallar porque el predictor de tags se equivoca o porque la descripción no corresponde al contenido. A **nivel de fotograma**, incluso con la etiqueta correcta el video varía enormemente cuadro a cuadro: un video *soccer* contiene tomas del marcador, entrevistas, presentadores y público, así que con clips de medio segundo al azar una fracción no trivial de los ejemplos es un locutor en un estudio etiquetado "fútbol".

El paper **no reporta una tasa de ruido medida** —conviene no inventarla al citarlo—; sí mide duplicados, y solo 1.755 videos del millón tienen fracción significativa de fotogramas casi-duplicados. Su conclusión es que las redes "*parecen aprender bien a pesar del ruido significativo de etiqueta*", incluyendo texto sobreimpreso, cortes y logos, "*ninguno de los cuales intentamos filtrar explícitamente*": robustez que anticipa la lógica del pre-entrenamiento débilmente supervisado a escala web.

> Lo que el paper no dice y se volvió evidente después: Sports-1M mide en gran medida **reconocimiento de escena deportiva**, no de acción. Una piscina implica natación; un tatami, artes marciales; un green, golf. Es exactamente el dataset donde un modelo de un solo fotograma debería rendir bien, así que **parte del hallazgo es una propiedad del dataset, no de las arquitecturas**.

---

## Las cuatro estrategias de conectividad temporal

Los videos varían mucho en extensión temporal, así que el paper **trata cada video como una bolsa de clips cortos de tamaño fijo**: toda la fusión temporal ocurre *dentro* de un clip y la agregación a nivel de video es un promedio posterior.

### Single Frame

Una [AlexNet](/papers/alexnet-krizhevsky-2012) con entrada $170 \times 170 \times 3$ en vez de $224 \times 224 \times 3$. Con $C(d,f,s)$ = capa convolucional de $d$ filtros $f \times f$ y stride $s$, $N$ = normalización y $P$ = pooling de $2 \times 2$:

$$C(96,11,3)\text{-}N\text{-}P\text{-}C(256,5,1)\text{-}N\text{-}P\text{-}C(384,3,1)\text{-}C(384,3,1)\text{-}C(256,3,1)\text{-}P\text{-}FC(4096)\text{-}FC(4096)$$

**Extensión temporal $T=1$:** captura apariencia estática (objetos, texturas, escena, pose instantánea) y **cero** movimiento. Su rol es diagnóstico: cuantificar cuánto se explica por apariencia pura.

### Early Fusion

Combina la ventana temporal completa **de inmediato, a nivel de píxel**, cambiando solo los filtros de la primera capa convolucional a

$$11 \times 11 \times 3 \times T, \qquad T = 10$$

es decir $11 \times 11$ espacial, 3 canales de color y 10 fotogramas, "aproximadamente un tercio de segundo" (implica ~30 fps). El resto de la red es idéntico. La conectividad directa a los píxeles permite "*detectar con precisión la dirección y velocidad del movimiento local*": el análogo aprendido de un filtro de Gabor espacio-temporal.

**Gotcha:** tras la primera capa **toda la dimensión temporal ha colapsado**. El resto es puramente 2D sobre un mapa que ya integró el tiempo; el movimiento se resume una vez y nunca se vuelve a razonar sobre él.

### Late Fusion

El extremo opuesto: **dos redes single-frame con parámetros compartidos** hasta la última convolucional $C(256,3,1)$, aplicadas a dos fotogramas separados **15 fotogramas** (~medio segundo) y fusionadas **en la primera capa fully connected**. Ninguna torre por sí sola detecta movimiento, pero la FC "*puede computar características de movimiento global comparando las salidas de ambas torres*". La palabra clave es **global**: infiere cambio agregado entre dos descripciones de alto nivel y pierde la velocidad y dirección locales que eran la fortaleza de Early Fusion.

### Slow Fusion

La propuesta que gana: "*una mezcla balanceada entre los dos enfoques que fusiona lentamente la información temporal a lo largo de la red, de modo que las capas superiores accedan a información progresivamente más global*". Replica en el eje temporal la jerarquía que la CNN ya aplica en el espacial, **extendiendo en el tiempo la conectividad de todas las capas convolucionales**:

| Capa | Extensión temporal $T$ | Stride temporal | Respuestas en el tiempo |
|---|---|---|---|
| conv1 | 4 | 2 | 4 |
| conv2 | 2 | 2 | 2 |
| conv3 | 2 | 2 | 1 |

Con convolución *valid*: $(10-4)/2+1 = 4$, luego $(4-2)/2+1 = 2$, luego $1$. **La tercera capa convolucional ve los 10 fotogramas de entrada**, y las columnas comparten parámetros. conv1 ve movimiento local fino (~0,13 s); conv3 ve la dinámica completa del clip (~0,33 s), construida composicionalmente. Es un [C3D](/papers/c3d-tran-2015) en miniatura, un año antes de C3D y tres antes de [I3D](/papers/i3d-carreira-2017).

| Estrategia | Dónde fusiona | Extensión temporal | Qué pierde |
|---|---|---|---|
| Single Frame | nunca | $T=1$ | todo el movimiento |
| Early Fusion | primera conv, a nivel de píxel | $T=10$ (~1/3 s) | jerarquía temporal (colapsa en una capa) |
| Late Fusion | primera FC, sobre features altos | 2 fotogramas a 15 de distancia | movimiento local, velocidad, dirección |
| Slow Fusion | todas las conv, progresivamente | $4 \to 2 \to 2$ = 10 fotogramas | costo; sigue limitada a ~1/3 s |

---

## La arquitectura multiresolución: fovea y contexto

El runtime era el límite real de la experimentación, y las alternativas obvias fallaban: reducir capas y neuronas "consistentemente baja el desempeño", y entrenar en baja resolución mejora el tiempo pero "*el detalle de alta frecuencia resultó crítico para lograr buena accuracy*". La solución son dos streams sobre un clip de $178 \times 178$: el **context stream** ve fotogramas submuestreados a la mitad ($89 \times 89$ del cuadro completo: toda la escena, borrosa) y el **fovea stream** la región central de $89 \times 89$ a resolución original (un tercio del área, nítido).

$$\frac{2 \times 89^2}{178^2} = \frac{15\,842}{31\,684} = \frac{1}{2}$$

La dimensionalidad de entrada se reduce **a la mitad**, y de ahí sale la aceleración. Se **elimina la última capa de pooling** para que ambos streams terminen en $7 \times 7 \times 256$; las activaciones se concatenan y alimentan la primera FC densa.

| Modelo | Sin multires | Con multires | Speedup |
|---|---|---|---|
| Single-Frame | 6 clips/s | 21 clips/s | 3,5× |
| Slow Fusion | 5 clips/s | 10 clips/s | 2,0× |

El costo en accuracy es nulo y de hecho negativo: Single-Frame sube de 59,3% a **60,0%**. Hay especialización emergente que nadie programó (el context aprende color y bajas frecuencias, el fovea filtros de alta frecuencia en escala de grises) y se necesitan ambos: Fovea Only 49,9% y Context Only 56,0% contra 60,0% combinado. Que el contexto solo supere holgadamente a la fóvea sola es otra pista de que Sports-1M premia el reconocimiento de escena.

> **El sesgo de centrado, admitido:** "*este diseño aprovecha el sesgo de cámara presente en muchos videos en línea, ya que el objeto de interés suele ocupar la región central*". Funciona porque quienes filman en YouTube centran el sujeto: es una regularidad estadística del corpus, no un principio de visión. La fovea fija **no es atención**, no aprende *dónde* mirar; vigilancia con múltiples actores, o un hallazgo clínico en la periferia, rompen el supuesto.

---

## Resultados: el hallazgo que incomodó al campo

Test set de 200.000 videos y 4.000.000 de clips. Hit@$k$ = fracción de muestras que contuvieron al menos una etiqueta ground truth en el top $k$. La predicción a nivel de video es lo más simple posible: muestrear **20 clips al azar**, propagar cada uno 4 veces con distintos crops y flips, y **promediar todas las predicciones**.

| Modelo | Clip Hit@1 | Video Hit@1 | Video Hit@5 |
|---|---|---|---|
| Feature Histograms + Neural Net (baseline artesanal) | — | 55,3 | — |
| Single-Frame | 41,1 | 59,3 | 77,7 |
| Single-Frame + Multires | 42,4 | 60,0 | 78,5 |
| Single-Frame Fovea Only | 30,0 | 49,9 | 72,8 |
| Single-Frame Context Only | 38,1 | 56,0 | 77,2 |
| Early Fusion | 38,9 | 57,7 | 76,8 |
| Late Fusion | 40,7 | 59,3 | 78,7 |
| **Slow Fusion** | **41,9** | **60,9** | **80,2** |
| CNN Average (Single + Early + Late + Slow) | 41,4 | **63,9** | **82,4** |

El baseline no es un hombre de paja: HOG, Texton y Cuboids densos *y* dispersos, cuantización k-means, *spatial pyramid encoding*, un vector de 25.000 dimensiones por video y una red multicapa validada extensivamente; además computa palabras visuales densamente sobre todo el video, mientras las CNN solo ven 20 clips al azar.

- **Las CNN superan consistente y significativamente al baseline**: 55,3 → 60,9 individual, → 63,9 en ensamble, y con desventaja de protocolo. Es el resultado *positivo*.
- **La variación entre arquitecturas CNN es "sorprendentemente insignificante"** (palabras del paper): 3,2 puntos entre la peor y la mejor variante con movimiento.
- **Early Fusion (57,7) queda por debajo de Single Frame**: el colapso temporal inmediato destruye más apariencia de la que aporta en movimiento.
- **Late Fusion empata exactamente a Single Frame en Video Hit@1** (59,3), mejorando solo en Hit@5: duplicar el cómputo convolucional para comparar dos instantes no compra nada medible en top-1.
- **El ensamble sube 3 puntos** sobre la mejor individual: los errores están parcialmente decorrelacionados.
- **Clip Hit@1 (~41%) vs. Video Hit@1 (~60%):** esos ~19 puntos son la ganancia pura de promediar 20 clips, y cuantifican el ruido a nivel de fotograma.

### Por qué Single Frame quedó tan cerca

$$\text{Single Frame} = 59{,}3\% \;\longrightarrow\; \text{Slow Fusion} = 60{,}9\%, \qquad \Delta = 1{,}6 \text{ puntos}$$

En Hit@5, 77,7 → 80,2 ($\Delta = 2{,}5$); en Clip Hit@1, 41,1 → 41,9 ($\Delta = 0{,}8$). Concediéndole el multiresolución al modelo estático (60,0%), la brecha queda en **0,9 puntos**. Traducido: añadir toda la información de movimiento del clip, con la mejor de las cuatro estrategias y un mes de cómputo en clúster sobre un millón de videos, compra menos de dos puntos. Las conclusiones: "*sorprendentemente, encontramos que un modelo de un solo fotograma ya exhibe desempeño muy fuerte, lo que sugiere que las señales de movimiento local pueden no ser críticamente importantes, incluso para un dataset dinámico como Sports*".

**(1) El movimiento local no es importante para esta tarea.** El contenido semántico de un video deportivo está mayoritariamente en la apariencia: escena, equipamiento, uniformes, geometría del campo. Cuando el paper añade que los beneficios son "*sorprendentemente robustos a los detalles de la conectividad de las arquitecturas en el tiempo*", está diciendo que la conectividad temporal casi no es un eje de diseño relevante *para Sports-1M*.

**(2) El movimiento de cámara contamina la señal.** Es la teoría que los autores prefieren: "*un tratamiento más cuidadoso del movimiento de cámara puede ser necesario, por ejemplo extrayendo features en el sistema de coordenadas local de un punto rastreado*", citando las *dense trajectories* de Wang et al. El movimiento en el plano de imagen es la superposición del movimiento del actor y del observador, y una CNN feedforward sobre píxeles crudos debe aprender invarianza al segundo mientras extrae el primero, con supervisión débil y sin sesgo inductivo que la ayude. La evidencia por clase es contundente: la diferencia de average precision entre Slow Fusion y Single-Frame es positiva en *Juggling Club* (+0,12), *Pole Climbing* (+0,10), *Mountain Unicycling* (+0,08), *Tricking* y *Footbag* (+0,07) —actividades definidas por un patrón cíclico o de equilibrio— y negativa en *Short Track Motor Racing*, *Road Racing* y *Jeet Kune Do* (−0,07), *Paintball*, *Cricket* y *Wrestling* (−0,06) y *Rally Cross* (−0,05). El paper nombra el patrón: "*las redes conscientes del movimiento son más propensas a rendir peor cuando hay movimiento de cámara presente*", y deja el remedio para trabajo futuro.

**(3) La agregación temporal a nivel de video es demasiado pobre.** La fusión opera dentro de ~1/3 de segundo y a nivel de video la única agregación es el promedio de 20 clips al azar. Si $p_i$ es la predicción del clip $i$, entonces $\frac{1}{N}\sum_i p_i$ es invariante a cualquier permutación: un modelo así **no puede, en principio, distinguir una acción de su reverso temporal**. Es una consecuencia formal, no un accidente de implementación, y de ahí la propuesta de "*explorar redes recurrentes*", que es el segundo eslabón de la [Clase 38](/clases/clase-38/teoria) anunciado tres años antes.

Hay una cuarta lectura que los autores no hacen y que la historia validó: **el dataset era el problema**. Ningún estudio de arquitectura puede detectar la utilidad del movimiento en un benchmark donde el movimiento no es discriminativo.

---

## Transferencia a UCF-101

Los autores prueban tres puntos de corte sobre Slow Fusion, la mejor red en Sports-1M. [UCF-101](/papers/ucf101-soomro-2012) son 13.320 videos en 101 categorías agrupadas en cinco familias (*Human-Object Interaction*, *Body-Motion Only*, *Human-Human Interaction*, *Playing Musical Instruments*, *Sports*), con 50 clips por video promediados sobre los 3 folds sugeridos.

| Modelo | 3-fold Accuracy |
|---|---|
| Soomro et al. (baseline del paper de UCF-101) | 43,9% |
| Feature Histograms + Neural Net | 59,0% |
| Train from scratch | 41,3% |
| Fine-tune top layer | 64,1% |
| **Fine-tune top 3 layers** | **65,4%** |
| Fine-tune all layers | 62,2% |

**La U invertida es el hallazgo**, y el resultado más reutilizable del paper para cualquier proyecto de [transfer learning](/fundamentos/transfer-learning). Congelar demasiado no es óptimo porque "*los features de alto nivel son quizás demasiado específicos de deportes*"; descongelar todo tampoco, "*probablemente debido a overfitting*" (13.320 videos contra decenas de millones de parámetros). El óptimo es reentrenar las dos capas fully connected con dropout muy agresivo, "*tan poco como 10% de probabilidad de mantener cada unidad activa*". Entrenar desde cero da 41,3%, peor incluso que el baseline de 2012 de los propios autores de UCF-101. En el desglose por grupo el mAP total pasa de 0,44 desde cero a 0,68 con fine-tune top 3, pero **la ganancia de "top" a "top 3" viene casi enteramente de las categorías no deportivas**: *Sports* apenas baja de 0,80 a 0,79 mientras *Playing Musical Instruments* salta de 0,46 a 0,65. Las FC de la Sports CNN están tan sesgadas hacia deportes que reentrenarlas es indispensable para transferir a "tocar la flauta"; y el desempeño no deportivo es notable considerando que "*la única manera de observar este tipo de fotogramas en los datos de entrenamiento es debido al ruido de etiqueta*", que actuó como diversificación involuntaria del dominio.

**La comparación que el paper no hace.** Gana contra los baselines que tiene a mano, pero **no compara contra el estado del arte artesanal de la época en UCF-101**, y ese silencio es significativo. Feichtenhofer et al. (CVPR 2016) tabulan "IDT + higher dimensional FV" (Peng et al., 2014) en **87,9%** —cifra externa a este paper, que no debe atribuírsele, pero que es la que lo contextualiza—. Una CNN pre-entrenada sobre un millón de videos alcanza 65,4% frente a ~88% de trayectorias densas con HOG/HOF/MBH y SVM: más de 20 puntos a favor de lo artesanal. En 2014 el deep learning ya había arrasado en imágenes y en video **todavía perdía por goleada**. Y lo que iDT tenía y la CNN no era exactamente modelado explícito y compensado del movimiento, porque extrae descriptores en el sistema de coordenadas local del punto rastreado: el remedio que el propio paper señala y deja pendiente.

Dos notas al citar: el abstract reporta la transferencia como "63,3% up from 43,9%" mientras la introducción y la Tabla 3 reportan "65,4%, up from 41,3%" —el 63,3% no aparece en ninguna tabla—. Y los autores intentaron obtener los IDs de YouTube de UCF-101 sin éxito: "*no podemos garantizar que el dataset Sports-1M no tenga solapamiento con UCF-101*".

---

## Limitaciones

**Reconocidas por los autores:** ningún tratamiento del movimiento de cámara, que ellos señalan como la limitación más importante, con la solución identificada y explícitamente diferida; agregación a nivel de video por promedio simple; cobertura estrecha, solo deportes, con el deseo de "*incorporar categorías más amplias para obtener features más potentes y genéricos*" —reconocimiento anticipado del problema que [Kinetics](/papers/kinetics-kay-2017) resolvería—; ruido de etiqueta no filtrado; posible solapamiento con UCF-101; y speedups dependientes de la implementación.

**Evidentes en retrospectiva:**

- **La extensión temporal es minúscula.** Máximo 10 fotogramas (~1/3 s), clips de medio segundo. Muchas acciones no se distinguen en ese horizonte; [I3D](/papers/i3d-carreira-2017) usa 64 fotogramas (2,56 s) y atribuye a eso su ventaja.
- **La red es poco profunda y no hereda ImageNet.** El paper prueba que pre-entrenar en video y transferir funciona, pero nunca prueba lo inverso: partir de una red pre-entrenada en ImageNet. Es la palanca más grande que dejó sin tirar, la que [Two-Stream](/papers/two-stream-simonyan-2014) usa en ambos streams y la que I3D formaliza con el *boring-video fixed point*.
- **Sin batch normalization, sin residuales, con learning rate manual.** Entrena con Downpour SGD asincrónico (10 a 50 réplicas por modelo, cada uno en 4 a 32 particiones), mini-batches de 32, momento 0,9, weight decay 0,0005 y learning rate inicial $10^{-3}$ reducido a mano.
- **La fovea fija no es atención:** asume que el sujeto está al centro.
- **El promedio de clips es invariante al orden**, con la consecuencia formal ya descrita.
- **El estudio de fusión temporal quedó confundido con la calidad del dataset.** "La conectividad temporal casi no importa" es verdadero en Sports-1M y falso en Kinetics; el problema es que se citó como verdad general sobre video.

Un detalle que envejeció bien: el augmentation (crop central, resize a $200 \times 200$, muestreo aleatorio de $170 \times 170$, flip al 50%) se aplica "*de manera consistente a todos los fotogramas que forman parte del mismo clip*". Sin esa consistencia, el crop aleatorio por fotograma inyectaría movimiento espurio y destruiría la señal que el modelo intenta aprender.

---

## Por qué importa hoy

**Es el origen del eslabón "CNN2D + agrupación temporal".** Es la primera demostración a escala de que se puede clasificar video ejecutando una CNN de imagen sobre fotogramas y agregando en el tiempo, y la primera medición seria de cuánto rinde eso. El modelo Single-Frame con votación por promedio de 20 clips *es* CNN2D + temporal pooling, y su número fija el techo del eslabón. Cuando la [Clase 38](/clases/clase-38/teoria) enumera sus desventajas —"no aprovecha la información temporal" y "tiende a tener un rendimiento deficiente"— está resumiendo esta tabla de resultados; y las ventajas también, porque es AlexNet más un promedio, sin kernels 3D, sin flujo óptico externo, sin RNN con estado, a 6 clips/s por réplica (21 con multiresolución). La brecha de 1,6 puntos entre Single Frame y Slow Fusion es la **evidencia empírica original** de esa desventaja, y es lo que motivó introducir el movimiento explícitamente.

El [Laboratorio 36](/laboratorios/lab-36) reprodujo el mismo fenómeno a escala de curso: con un ResNet-34 y *average temporal pooling*, muestrear 4 frames rindió igual que muestrear 8, en la mitad del tiempo. Si el modelo ignora el orden, agregar frames no agrega información temporal, solo promedia mejor la apariencia. Es la razón de que la [Clase 36](/clases/clase-36) y el [fundamento de análisis de video](/fundamentos/analisis-de-video) traten el pooling temporal como punto de partida y no como destino.

**Cómo su diagnóstico motivó Two-Stream e I3D.** Un resultado negativo bien medido sobre un millón de videos es un mandato de investigación. Si aprender movimiento *implícitamente* desde píxeles crudos casi no ayuda, quedan dos salidas. La primera es **dar el movimiento ya calculado**: las [two-stream ConvNets](/papers/two-stream-simonyan-2014) aparecen el mismo año y entregan flujo óptico pre-computado en un stream dedicado, inicializado desde ImageNet y estabilizable restando el movimiento medio de cámara; el diagnóstico de Karpathy et al. es literalmente su argumento de venta, y Two-Stream saltó a la banda de los 88% en UCF-101. La segunda es **dar la maquinaria arquitectónica adecuada con suficiente extensión temporal**: el linaje [C3D](/papers/c3d-tran-2015) → [I3D](/papers/i3d-carreira-2017). Slow Fusion es el ancestro reconocible de esa idea, y que solo ganara 1,6 puntos es un artefacto de la poca profundidad, la corta extensión temporal y la falta de pre-entrenamiento de imagen, no una refutación de la convolución 3D.

| | Karpathy et al. 2014 | I3D 2017 |
|---|---|---|
| Pre-entrenamiento de imagen | ninguno | ImageNet (inflado 2D→3D) |
| Pre-entrenamiento de video | Sports-1M, 1M videos, ruidoso | Kinetics, 240k videos, curado |
| Extensión temporal | 10 fotogramas (~0,33 s) | 64 fotogramas (2,56 s) |
| Profundidad | ~8 capas estilo AlexNet | Inception-v1 inflada |
| Flujo óptico explícito | no | sí (TV-L1) |
| UCF-101 (3 splits) | **65,4%** | **98,0%** |

Los 32,6 puntos entre esas filas son la historia completa de la Clase 38, y el orden de las palancas que los explican es instructivo: pre-entrenamiento de imagen, profundidad, extensión temporal, movimiento explícito y calidad del dataset. Este paper tiene solo el volumen de datos, y el volumen solo no alcanzó.

**Por qué Sports-1M no fue el ImageNet del video.** Es el legado más instructivo, precisamente porque es un fracaso parcial: tenía la escala y le faltaba todo lo demás.

| Propiedad | Sports-1M | [Kinetics](/papers/kinetics-kay-2017) |
|---|---|---|
| Etiquetas | automáticas desde metadatos, ruidosas en dos niveles | curadas con verificación humana |
| Dominio | solo deportes | acciones humanas amplias |
| Recorte temporal | videos de 5 min 36 s con marcadores, entrevistas, público | clips de ~10 s recortados en la acción |
| Discriminatividad temporal | baja: la escena estática casi basta | alta por diseño |

I3D reporta que su ventaja sobre C3D es grande **aunque C3D se entrenó con más videos** (el millón de Sports-1M más un dataset interno) y en ensamble con iDT, atribuyéndolo a la mejor calidad de Kinetics además de la arquitectura. La lección es que **un ImageNet de video necesita curación y recorte temporal, no solo volumen**. Aun así la contribución fundacional queda en pie: fue el primer dataset de video a escala web, el corpus de pre-entrenamiento de C3D, y estableció que pre-entrenar en video y transferir era viable.

La lección de fondo: **un resultado negativo, medido con rigor y a la escala correcta, es más productivo para un campo que un resultado positivo marginal**. Los autores podrían haber titulado "las CNN funcionan en video" con el 63,9% del ensamble contra el 55,3% del baseline. En vez de eso escribieron que la mejora sobre el modelo de un solo fotograma era "sorprendentemente modesta", y esa frase organizó los cinco años siguientes de investigación en [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).

---

## Notas y enlaces

- **Clase asociada:** [Clase 38](/clases/clase-38) y su [teoría](/clases/clase-38/teoria), donde este paper abre la escalera *CNN2D + agrupación temporal → CNN2D + RNN → Two-Stream → C3D → I3D*. Anticipa los cuatro escalones siguientes: propone RNN en sus conclusiones, motiva Two-Stream con su diagnóstico del movimiento de cámara, prototipa la convolución 3D en Slow Fusion y deja pendiente I3D al nunca inicializar desde ImageNet.
- **Laboratorio relacionado:** el [Laboratorio 36](/laboratorios/lab-36) reprodujo el hallazgo central —4 frames rindieron prácticamente igual que 8 con average temporal pooling—, prueba directa de que el modelo ignora el orden temporal.
- **Descendencia y benchmarks:** [Two-Stream](/papers/two-stream-simonyan-2014), [C3D](/papers/c3d-tran-2015) (pre-entrenado sobre Sports-1M), [I3D](/papers/i3d-carreira-2017), [UCF-101](/papers/ucf101-soomro-2012) como dataset de transferencia y [Kinetics](/papers/kinetics-kay-2017) como el corpus que sí logró ser el ImageNet del video.
- **Fundamentos del site:** [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones), [análisis de video](/fundamentos/analisis-de-video) y [transfer learning](/fundamentos/transfer-learning). Para este último, la U invertida de UCF-101 es la regla práctica: con dataset objetivo pequeño y dominio de pre-entrenamiento estrecho, congelar todo desperdicia adaptación y descongelar todo produce overfitting; el óptimo suele estar en reentrenar las capas densas superiores con dropout agresivo.
- **Backbone y sitio:** variante de [AlexNet](/papers/alexnet-krizhevsky-2012) con entrada de $170 \times 170$, entrenada desde cero ~1 mes en clúster, viendo ~500 millones de ejemplos (≈10 épocas efectivas sobre ~50 millones de fotogramas). Sitio del proyecto: `cs.stanford.edu/people/karpathy/deepvideo`.
