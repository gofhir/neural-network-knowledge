---
title: "Two-Stream: apariencia + movimiento (2014)"
weight: 402
math: true
---

{{< paper-card
    title="Two-Stream Convolutional Networks for Action Recognition in Videos"
    authors="Karen Simonyan, Andrew Zisserman (Oxford VGG)"
    year="2014"
    venue="NeurIPS 2014 / arXiv:1406.2199"
    pdf="/papers/two-stream-simonyan-2014.pdf" >}}
Simonyan y Zisserman resuelven la debilidad que la [Clase 36](/clases/clase-36) nombra —la **2D CNN aplicada fotograma a fotograma descarta el sentido temporal y el movimiento**— sin forzar a la red a aprender el movimiento desde píxeles RGB apilados (enfoque que ya había fracasado en Karpathy et al., quedando ~20% por debajo de las trayectorias artesanales). En su lugar **inyectan el movimiento de forma explícita** mediante dos vías: un **stream espacial** que consume fotogramas RGB individuales (apariencia: escenas y objetos) y un **stream temporal** que consume [flujo óptico](/fundamentos/flujo-optico) denso multi-fotograma (movimiento del observador y de los objetos), fusionados por **late fusion**. La analogía es la *two-streams hypothesis* del córtex visual (vía ventral = "qué", vía dorsal = "cómo se mueve"). El modelo completo alcanza **88.0% en UCF-101 y 59.4% en HMDB-51**, comparable por primera vez al estado del arte artesanal (Improved Dense Trajectories: 87.9% / 61.1%) con una red profunda entrenada sobre datasets pequeños. Es el ancestro directo de [I3D](/papers/i3d-carreira-2017) y de toda la familia two-stream.
{{< /paper-card >}}

---

## Contexto: la 2D CNN por fotograma ignora el movimiento

Comparado con la imagen estática, el video aporta una pista adicional decisiva: la **componente temporal**, y muchas acciones se reconocen de forma fiable solo a partir del movimiento. Hacia 2014 la pregunta era cómo extender al video las ConvNets que ya dominaban la clasificación de imágenes tras AlexNet.

El intento previo dominante (Karpathy et al., CVPR 2014) alimentaba la red con **pilas de fotogramas RGB consecutivos**, esperando que aprendiera *implícitamente* características sensibles al movimiento. El hallazgo fue desalentador: una red que operaba sobre **fotogramas individuales** rendía de forma **similar** a las que recibían pilas de fotogramas. Las características espacio-temporales aprendidas **no capturaban bien el movimiento**, y la representación quedaba 20% por debajo de las trayectorias artesanales. La razón: apilar RGB y confiar en que la red descubra el movimiento es un problema *demasiado difícil* —las primeras capas tendrían que estimar desplazamientos entre fotogramas Y clasificar acciones simultáneamente, con datos escasos. Esta es exactamente la debilidad que la [Clase 36](/clases/clase-36) señala en el [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).

## Método: dos vías y la entrada de flujo óptico

La idea es descomponer el video en sus dos componentes naturales y asignar a cada uno una ConvNet dedicada. El **stream espacial**, al ser una arquitectura de clasificación de imágenes (CNN-M-2048: cinco capas convolucionales + dos totalmente conectadas), se **preentrena sobre ImageNet ILSVRC-2012** —ventaja imposible si apariencia y movimiento estuvieran enredados en una red monolítica.

El corazón del paper es el **stream temporal**. Su entrada se forma **apilando campos de desplazamiento de flujo óptico** entre fotogramas consecutivos, describiendo el movimiento **explícitamente** para que la red no tenga que estimarlo. El flujo denso es un campo vectorial $d_t$ donde $d_t(u,v)$ es el desplazamiento en el punto $(u,v)$ del fotograma $t$ hacia el $t{+}1$; se calcula con el método de Brox et al. (2004). Cada componente horizontal $d^x_t$ y vertical $d^y_t$ es tratable como un canal de imagen. En el **optical flow stacking**, se apilan los canales de $L$ fotogramas consecutivos formando un volumen $I_\tau \in \mathbb{R}^{w\times h\times 2L}$:

$$I_\tau(u,v,2k-1) = d^x_{\tau+k-1}(u,v), \qquad I_\tau(u,v,2k) = d^y_{\tau+k-1}(u,v),$$

con $k=[1;L]$. El paper explora además **trajectory stacking** (muestrear el flujo a lo largo de la trayectoria del movimiento en vez de en posición fija), **flujo bidireccional** y **mean flow subtraction** (restar el vector medio de cada campo, una compensación barata del movimiento global de cámara). Como el stream temporal recibe canales de flujo y no RGB, **no puede preentrenarse en ImageNet** y se entrena desde cero, lo que motiva un esquema de **aprendizaje multitarea** con dos capas softmax (UCF-101 + HMDB-51) que actúa como regularizador. La visualización de los filtros de la primera capa muestra que la red aprende **derivadas espaciales y temporales del flujo**, generalizando de forma aprendible los descriptores artesanales MBH y HOF.

Cada stream produce puntuaciones softmax que se combinan por **fusión tardía**: promedio o un **SVM lineal multiclase** sobre las puntuaciones normalizadas. Una fusión más temprana (capas totalmente conectadas conjuntas) **no fue factible por sobreajuste** en el régimen de datos escasos.

## Resultados

El preentrenamiento en ImageNet es decisivo para el stream espacial (72.7% vs. 52.3% entrenando solo en UCF-101). Para el stream temporal, apilar múltiples flujos es muy beneficioso: single-frame ($L=1$) da 73.9%, mientras $L=10$ con sustracción de media llega a **81.0%**, superando **significativamente al stream espacial**. Un contraste demoledor: la arquitectura "slow fusion" de Karpathy (ConvNet sobre pila de 11 fotogramas RGB) alcanza solo **56.4%** entrenada desde cero, muy por debajo del ~81% del flujo óptico explícito —la información multi-fotograma importa, pero importa igualmente **presentarla de forma apropiada**.

El modelo two-stream completo confirma las conclusiones centrales: los dos streams son **complementarios** —su fusión mejora **6% sobre el temporal solo y 14% sobre el espacial solo**—; la fusión por **SVM supera al promedio**; y el multitarea mejora el stream temporal. La mejor combinación alcanza **88.0% en UCF-101 y 59.4% en HMDB-51** (promedio sobre tres splits), comparable al estado del arte artesanal de la época. El análisis de errores es instructivo: *Hammering* (martillar) es la peor clase, y cada stream falla por su propia razón —el espacial la confunde con *HeadMassage* por la presencia de rostros; el temporal con *BrushingTeeth* por el **patrón de movimiento recurrente** de la mano subiendo y bajando—, ilustrando por qué son complementarios.

## Limitaciones

- **Flujo óptico precomputado y costoso.** Debe calcularse por adelantado (1.5 TB sin comprimir para UCF-101, reducido a 27 GB con reescalado a $[0,255]$ y JPEG) porque hacerlo on-the-fly sería un cuello de botella. Rompe la promesa "end-to-end": hay una etapa de preprocesamiento pesada y no aprendible fuera del grafo de cómputo.
- **Fusión tardía subóptima.** Los streams solo se combinan al final, sobre las puntuaciones softmax; una fusión intermedia sobreajustaba. Feichtenhofer et al. (2016) abordarían esto fusionando mapas convolucionales.
- **Sin modelado temporal de largo alcance.** El stream temporal cubre solo $L=10$ fotogramas (~fracción de segundo); las dependencias largas quedan fuera.
- **Manejo tosco del movimiento de cámara** (solo se resta el desplazamiento medio).

## Por qué importa para la Clase 36

Este paper cierra el arco argumental de la [Clase 36](/clases/clase-36). La clase parte de la 2D CNN por fotograma, señala que **descarta el sentido temporal y el movimiento**, e introduce el [flujo óptico](/fundamentos/flujo-optico) como forma de representarlo explícitamente. Simonyan y Zisserman unen ambas piezas: conservan la 2D CNN como stream espacial (donde es fuerte, la apariencia) y **añaden un segundo stream que consume flujo óptico** para inyectar el movimiento que la 2D CNN sola no puede aprender. El principio organizador —**separar el "qué" del "cómo se mueve"**— estructuró un lustro de investigación en [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones). Su descendiente más importante es [I3D](/papers/i3d-carreira-2017) (Carreira y Zisserman, CVPR 2017), que **infla** las ConvNets 2D a 3D pero **conserva la estructura de dos streams**, demostrando que incluso con convoluciones 3D el stream de flujo óptico seguía aportando ganancia.
