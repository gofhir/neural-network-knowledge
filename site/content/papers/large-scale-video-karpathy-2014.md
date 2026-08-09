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
Este paper no propone un modelo estrella, y eso es parte del mensaje: es **el primer estudio comparativo a escala de cómo conectar una CNN en el tiempo**, junto con el dataset que lo hizo posible. **Sports-1M** son 1.000.000 de videos de YouTube etiquetados automáticamente en **487 clases de deportes**, y sobre él se comparan cuatro patrones de conectividad temporal —*Single Frame*, *Early Fusion*, *Late Fusion* y *Slow Fusion*— con la misma red base y protocolo, más una red **multiresolución fovea + context** que acelera 2–4× sin costo en precisión. Las CNN superan al baseline artesanal (55,3% → 60,9% Video Hit@1; 63,9% en ensamble), pero el resultado que quedó en la historia es el que *no* apareció: el modelo que solo ve **un fotograma estático** alcanza 59,3%, apenas 1,6 puntos por debajo del mejor modelo espacio-temporal, "*una mejora sorprendentemente modesta*" según el abstract. Ese hallazgo negativo se volvió el punto de partida de las [two-stream ConvNets](/papers/two-stream-simonyan-2014), de [C3D](/papers/c3d-tran-2015) y de [I3D](/papers/i3d-carreira-2017), y el paper demuestra además que pre-entrenar en video y transferir funciona: [UCF-101](/papers/ucf101-soomro-2012) pasa de 41,3% desde cero a 65,4%. Para la [Clase 38](/clases/clase-38) es el origen del primer eslabón de la escalera del video: *CNN2D + agrupación temporal*.
{{< /paper-card >}}

---

## Contexto: por qué el video seguía atrás de las imágenes

En 2014 el consenso en imágenes estaba cerrado: los features de una [AlexNet](/papers/alexnet-krizhevsky-2012) de ImageNet, con un SVM encima y **sin fine-tuning**, daban estado del arte. En video no, y los autores diagnostican tres cuellos.

**Ausencia de datasets a escala.** Los benchmarks tenían "hasta unos pocos miles de clips y hasta unas pocas decenas de clases"; los mayores eran CCV (9.317 videos, 20 clases) y [UCF-101](/papers/ucf101-soomro-2012) (13.320 videos, 101 clases). Como toda aplicación exitosa de CNN en imágenes tenía un training set grande, los autores especulan que el estancamiento era **atribuible en parte a la falta de benchmarks a gran escala**: es la hipótesis central del trabajo.

**Costo computacional.** Entrenar tardaba "del orden de semanas" incluso en las mejores GPU, y con clips de 10 fotogramas la primera capa hace 10× el trabajo.

**Falta de un patrón temporal canónico.** En imágenes la convolución 2D con *parameter sharing* y max pooling era la respuesta probada; en video no había equivalente, y las extensiones 3D existentes eran "*solo una de las generalizaciones posibles*", nunca comparadas sobre el mismo dataset y protocolo.

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

La taxonomía es una jerarquía curada a mano que se vuelve **fine-grained en las hojas** —6 tipos de bowling, 7 de fútbol americano, 23 de billar—, y ahí está buena parte del error: los pares más confundidos (*sledding* vs. *toboggan*) son ambiguos hasta para un humano.

Las anotaciones se producen automáticamente "*analizando los metadatos de texto que rodean a los videos*", y hay **dos niveles de ruido**: a **nivel de video** la etiqueta puede no corresponder al contenido, y a **nivel de fotograma** el video varía cuadro a cuadro —un video *soccer* contiene marcador, entrevistas y público, así que con clips de medio segundo al azar parte de los ejemplos es un locutor etiquetado "fútbol"—. El paper **no reporta una tasa de ruido medida** y concluye que las redes "*parecen aprender bien a pesar del ruido significativo de etiqueta*".

Lo que no dice y se volvió evidente después: Sports-1M mide sobre todo **escena deportiva**, no acción, y es justo el dataset donde un modelo de un solo fotograma debería rendir bien. **Parte del hallazgo central es una propiedad del dataset, no de las arquitecturas.**

---

## Las cuatro estrategias de conectividad temporal

El paper **trata cada video como una bolsa de clips cortos de tamaño fijo**: la fusión temporal ocurre *dentro* del clip y la agregación global es un promedio posterior.

### Single Frame

Una [AlexNet](/papers/alexnet-krizhevsky-2012) con entrada $170 \times 170 \times 3$, donde $C(d,f,s)$ es una convolucional de $d$ filtros $f \times f$ con stride $s$, $N$ normalización y $P$ pooling de $2 \times 2$:

$$C(96,11,3)\text{-}N\text{-}P\text{-}C(256,5,1)\text{-}N\text{-}P\text{-}C(384,3,1)\text{-}C(384,3,1)\text{-}C(256,3,1)\text{-}P\text{-}FC(4096)\text{-}FC(4096)$$

**Extensión temporal $T=1$:** apariencia estática y **cero** movimiento. Rol diagnóstico: cuantificar cuánto se explica por apariencia pura.

### Early Fusion

Combina la ventana completa **de inmediato, a nivel de píxel**, cambiando solo los filtros de la primera capa convolucional a

$$11 \times 11 \times 3 \times T, \qquad T = 10$$

esto es $11 \times 11$ espacial, 3 canales y 10 fotogramas —"aproximadamente un tercio de segundo", lo que implica ~30 fps—, y permite "*detectar con precisión la dirección y velocidad del movimiento local*". **Gotcha:** tras esa capa **la dimensión temporal ha colapsado**; el resto es puramente 2D, sin jerarquía temporal.

### Late Fusion

El extremo opuesto: **dos redes single-frame con parámetros compartidos**, aplicadas a dos fotogramas separados **15 fotogramas** (~medio segundo) y fusionadas **en la primera capa fully connected**. Ninguna torre detecta movimiento por sí sola; la FC lo computa de forma **global** comparando ambas salidas, pero pierde la velocidad y dirección locales de Early Fusion.

### Slow Fusion

La que gana: fusiona el tiempo lentamente **extendiendo la conectividad temporal de todas las convolucionales**, para que las capas altas vean información progresivamente más global. Sobre un clip de 10 fotogramas:

| Capa | Extensión temporal $T$ | Stride temporal | Respuestas en el tiempo |
|---|---|---|---|
| conv1 | 4 | 2 | 4 |
| conv2 | 2 | 2 | 2 |
| conv3 | 2 | 2 | 1 |

Con convolución *valid*: $(10-4)/2+1 = 4$, luego $(4-2)/2+1 = 2$, luego $1$. **La tercera capa convolucional ve los 10 fotogramas de entrada** y las columnas comparten parámetros: conv1 ve movimiento local fino (~0,13 s) y conv3 la dinámica completa del clip (~0,33 s). Es un [C3D](/papers/c3d-tran-2015) en miniatura, tres años antes de [I3D](/papers/i3d-carreira-2017).

---

## La arquitectura multiresolución: fovea y contexto

Reducir capas "consistentemente baja el desempeño" y bajar la resolución tampoco servía, porque "*el detalle de alta frecuencia resultó crítico*". La solución son dos streams sobre un clip de $178 \times 178$: el **context stream** ve el cuadro completo submuestreado a la mitad ($89 \times 89$, borroso) y el **fovea stream** la región central de $89 \times 89$ a resolución original. Ambos terminan en $7 \times 7 \times 256$ y se concatenan antes de la primera FC; la entrada total baja **a la mitad**:

$$\frac{2 \times 89^2}{178^2} = \frac{15\,842}{31\,684} = \frac{1}{2}$$

| Modelo | Sin multires | Con multires | Speedup |
|---|---|---|---|
| Single-Frame | 6 clips/s | 21 clips/s | 3,5× |
| Slow Fusion | 5 clips/s | 10 clips/s | 2,0× |

El costo en accuracy es nulo, incluso negativo: Single-Frame sube de 59,3% a **60,0%**. Y el sesgo está admitido: "*este diseño aprovecha el sesgo de cámara presente en muchos videos en línea, ya que el objeto de interés suele ocupar la región central*". Es una regularidad estadística del corpus, no un principio de visión, y la fovea fija **no es atención**: no aprende *dónde* mirar, así que un hallazgo en la periferia rompe el supuesto.

---

## Resultados: el hallazgo que incomodó al campo

Test set de 200.000 videos y 4.000.000 de clips; Hit@$k$ = fracción de muestras con al menos una etiqueta correcta en el top $k$. La predicción por video es mínima: **20 clips al azar** y promedio de sus predicciones.

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

- **Las CNN superan claramente al baseline** (55,3 → 60,9 individual, → 63,9 en ensamble) y con desventaja de protocolo, porque el baseline recorre el video completo y las CNN solo 20 clips.
- **La variación entre arquitecturas CNN es "sorprendentemente insignificante"**: 3,2 puntos entre la peor y la mejor variante con movimiento.
- **Early Fusion (57,7) queda por debajo de Single Frame**: el colapso temporal inmediato destruye más apariencia de la que aporta en movimiento.
- **Late Fusion empata exactamente a Single Frame en Video Hit@1** (59,3): comparar dos instantes al doble de cómputo no compra nada medible en top-1.
- **Clip Hit@1 (~41%) vs. Video Hit@1 (~60%):** esos ~19 puntos son la ganancia de promediar 20 clips y cuantifican el ruido de fotograma.

### Por qué Single Frame quedó tan cerca

$$\text{Single Frame} = 59{,}3\% \;\longrightarrow\; \text{Slow Fusion} = 60{,}9\%, \qquad \Delta = 1{,}6 \text{ puntos}$$

En Hit@5, 77,7 → 80,2 ($\Delta = 2{,}5$); en Clip Hit@1, 41,1 → 41,9 ($\Delta = 0{,}8$). Con el multiresolución del lado estático (60,0%) la brecha queda en **0,9 puntos**: todo el movimiento del clip, con la mejor de las cuatro estrategias y un mes de cómputo sobre un millón de videos, compra menos de dos puntos. El paper concluye que "*un modelo de un solo fotograma ya exhibe desempeño muy fuerte, lo que sugiere que las señales de movimiento local pueden no ser críticamente importantes, incluso para un dataset dinámico como Sports*".

Hay tres explicaciones en juego, y cada una tuvo descendencia distinta.

**(1) El movimiento local no importa para esta tarea.** La semántica del video deportivo está en la apariencia, y los beneficios son "*sorprendentemente robustos a los detalles de la conectividad de las arquitecturas en el tiempo*".

**(2) El movimiento de cámara contamina la señal.** Es la teoría que los autores prefieren: haría falta "*extraer features en el sistema de coordenadas local de un punto rastreado*", como las *dense trajectories* de Wang et al., porque el movimiento en el plano de imagen superpone el del actor y el del observador. La evidencia por clase lo confirma: la diferencia de average precision entre Slow Fusion y Single-Frame es positiva en *Juggling Club* (+0,12), *Pole Climbing* (+0,10) y *Mountain Unicycling* (+0,08) —actividades cíclicas o de equilibrio— y negativa en *Short Track Motor Racing* (−0,07) o *Wrestling* (−0,06): "*las redes conscientes del movimiento son más propensas a rendir peor cuando hay movimiento de cámara presente*".

**(3) La agregación a nivel de video es demasiado pobre.** Si $p_i$ es la predicción del clip $i$, $\frac{1}{N}\sum_i p_i$ es invariante a cualquier permutación: el modelo **no puede, en principio, distinguir una acción de su reverso temporal**. De ahí la propuesta de "*explorar redes recurrentes*" —el segundo eslabón de la [Clase 38](/clases/clase-38/teoria), anunciado tres años antes—.

---

## Transferencia a UCF-101

Los autores prueban tres puntos de corte sobre Slow Fusion, la mejor red en Sports-1M. [UCF-101](/papers/ucf101-soomro-2012) son 13.320 videos en 101 categorías, promediando sobre los 3 folds sugeridos.

| Modelo | 3-fold Accuracy |
|---|---|
| Soomro et al. (baseline del paper de UCF-101) | 43,9% |
| Feature Histograms + Neural Net | 59,0% |
| Train from scratch | 41,3% |
| Fine-tune top layer | 64,1% |
| **Fine-tune top 3 layers** | **65,4%** |
| Fine-tune all layers | 62,2% |

**La U invertida es el hallazgo**, y el resultado más reutilizable del paper para [transfer learning](/fundamentos/transfer-learning). Congelar demasiado no es óptimo porque "*los features de alto nivel son quizás demasiado específicos de deportes*"; descongelar todo tampoco, "*probablemente debido a overfitting*". El óptimo es reentrenar las dos capas fully connected con dropout muy agresivo, "*tan poco como 10% de probabilidad de mantener cada unidad activa*". Y **la ganancia de "top" a "top 3" viene casi enteramente de las categorías no deportivas**: *Sports* apenas baja de 0,80 a 0,79 mAP mientras *Playing Musical Instruments* salta de 0,46 a 0,65. (Ojo al citar: el abstract dice "63,3% up from 43,9%" y la Tabla 3 dice "65,4%, up from 41,3%".)

**La comparación que el paper no hace.** Gana contra los baselines que tiene a mano, pero **no compara contra el estado del arte artesanal**: Feichtenhofer et al. (CVPR 2016) tabulan "IDT + higher dimensional FV" (Peng et al., 2014) en **87,9%** en UCF-101 —cifra externa a este paper, que no debe atribuírsele—. Más de 20 puntos a favor de lo artesanal: en 2014 el deep learning ya había arrasado en imágenes y en video **todavía perdía por goleada**. Y lo que iDT tenía y la CNN no era exactamente modelado explícito del movimiento de cámara.

---

## Limitaciones

**Reconocidas por los autores:** ningún tratamiento del movimiento de cámara, que señalan como la más importante y dejan diferida; agregación por promedio simple; cobertura estrecha, solo deportes —lo que [Kinetics](/papers/kinetics-kay-2017) resolvería—; ruido no filtrado; y posible solapamiento con UCF-101.

**Evidentes en retrospectiva:**

- **La extensión temporal es minúscula:** máximo 10 fotogramas (~1/3 s), y muchas acciones no se distinguen en ese horizonte. [I3D](/papers/i3d-carreira-2017) usa 64 (2,56 s) y atribuye a eso su ventaja.
- **La red es poco profunda y no hereda ImageNet.** Prueba que pre-entrenar en video y transferir funciona, pero nunca lo inverso: partir de una red pre-entrenada en ImageNet. Es la palanca más grande que dejó sin tirar, la que [Two-Stream](/papers/two-stream-simonyan-2014) usa en ambos streams y la que I3D formaliza inflando pesos 2D a 3D.
- **El estudio de fusión temporal quedó confundido con la calidad del dataset:** "la conectividad temporal casi no importa" es verdadero en Sports-1M y falso en Kinetics, pero se citó como verdad general sobre video.

---

## Por qué importa hoy

**Es el origen del eslabón "CNN2D + agrupación temporal".** El Single-Frame con promedio de 20 clips *es* CNN2D + temporal pooling, y este paper mide su techo por primera vez. Cuando la [Clase 38](/clases/clase-38/teoria) enumera sus desventajas —"no aprovecha la información temporal" y "tiende a tener un rendimiento deficiente"— está resumiendo esta tabla; y sus ventajas también, porque es AlexNet más un promedio: sin kernels 3D, sin flujo óptico, sin RNN. La brecha de 1,6 puntos frente a Slow Fusion es la **evidencia empírica original** de esa desventaja, y lo que motivó introducir el movimiento explícitamente.

El [Laboratorio 36](/laboratorios/lab-36) reprodujo el fenómeno: con un ResNet-34 y *average temporal pooling*, muestrear 4 frames rindió igual que 8, en la mitad del tiempo. Si el modelo ignora el orden, agregar frames solo promedia mejor la apariencia. Por eso la [Clase 36](/clases/clase-36) y el [fundamento de análisis de video](/fundamentos/analisis-de-video) tratan el pooling temporal como punto de partida y no como destino.

**Cómo su diagnóstico motivó Two-Stream e I3D.** Si aprender movimiento *implícitamente* desde píxeles crudos casi no ayuda, quedan dos salidas. **Dar el movimiento ya calculado**: las [two-stream ConvNets](/papers/two-stream-simonyan-2014) aparecen el mismo año con flujo óptico pre-computado en un stream inicializado desde ImageNet, y saltan al 88% en UCF-101. O **dar la maquinaria arquitectónica adecuada con suficiente extensión temporal**: el linaje [C3D](/papers/c3d-tran-2015) → [I3D](/papers/i3d-carreira-2017), del que Slow Fusion es el ancestro; que solo ganara 1,6 puntos es artefacto de la poca profundidad, la corta extensión temporal y la falta de pre-entrenamiento de imagen, no una refutación de la convolución 3D.

| | Karpathy et al. 2014 | I3D 2017 |
|---|---|---|
| Pre-entrenamiento de imagen | ninguno | ImageNet (inflado 2D→3D) |
| Pre-entrenamiento de video | Sports-1M, 1M videos, ruidoso | Kinetics, 240k videos, curado |
| Extensión temporal | 10 fotogramas (~0,33 s) | 64 fotogramas (2,56 s) |
| Profundidad | ~8 capas estilo AlexNet | Inception-v1 inflada |
| Flujo óptico explícito | no | sí (TV-L1) |
| UCF-101 (3 splits) | **65,4%** | **98,0%** |

Los 32,6 puntos entre esas filas son la historia de la Clase 38, y las palancas que los explican son pre-entrenamiento de imagen, profundidad, extensión temporal, movimiento explícito y calidad del dataset. Este paper solo tiene volumen, y el volumen solo no alcanzó.

**Por qué Sports-1M no fue el ImageNet del video.** Es el legado más instructivo, porque es un fracaso parcial: tenía la escala y le faltaba todo lo demás.

| Propiedad | Sports-1M | [Kinetics](/papers/kinetics-kay-2017) |
|---|---|---|
| Etiquetas | automáticas desde metadatos, ruidosas en dos niveles | curadas con verificación humana |
| Dominio | solo deportes | acciones humanas amplias |
| Recorte temporal | videos de 5 min 36 s con marcadores y entrevistas | clips de ~10 s recortados en la acción |
| Discriminatividad temporal | baja: la escena estática casi basta | alta por diseño |

I3D atribuye su ventaja sobre C3D a la calidad de Kinetics **aunque C3D se entrenó con más videos**: **un ImageNet de video necesita curación y recorte temporal, no solo volumen**. La contribución fundacional queda en pie igual, porque fue el primer dataset de video a escala web y el corpus con que se pre-entrenó C3D.

La lección de fondo: **un resultado negativo bien medido es más productivo que un positivo marginal**. Los autores escribieron que la mejora sobre el modelo de un solo fotograma era "sorprendentemente modesta", y esa frase organizó cinco años del [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).

---

## Notas y enlaces

- **Clase y laboratorio:** [Clase 38](/clases/clase-38) y su [teoría](/clases/clase-38/teoria), donde abre la escalera *CNN2D + agrupación temporal → CNN2D + RNN → Two-Stream → C3D → I3D*; el [Laboratorio 36](/laboratorios/lab-36) reprodujo su hallazgo central.
- **Descendencia y benchmarks:** [Two-Stream](/papers/two-stream-simonyan-2014), [C3D](/papers/c3d-tran-2015), [I3D](/papers/i3d-carreira-2017), [UCF-101](/papers/ucf101-soomro-2012), [Kinetics](/papers/kinetics-kay-2017) y el backbone [AlexNet](/papers/alexnet-krizhevsky-2012).
- **Fundamentos:** [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones), [análisis de video](/fundamentos/analisis-de-video) y [transfer learning](/fundamentos/transfer-learning).
