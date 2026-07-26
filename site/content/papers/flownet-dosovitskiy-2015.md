---
title: "FlowNet: flujo óptico con CNN (2015)"
weight: 401
math: true
---

{{< paper-card
    title="FlowNet: Learning Optical Flow with Convolutional Networks"
    authors="Alexey Dosovitskiy et al."
    year="2015"
    venue="ICCV 2015 / arXiv:1504.06852"
    pdf="/papers/flownet-dosovitskiy-2015.pdf" >}}
FlowNet es el **primer trabajo que resuelve la estimación de flujo óptico como una tarea de aprendizaje supervisado end-to-end** con redes convolucionales. Hasta 2015 el flujo óptico —el campo de desplazamiento por píxel entre dos frames consecutivos— había resistido a las CNN: no basta con extraer una representación semántica de una imagen, hay que **encontrar correspondencias** entre dos imágenes con precisión subpíxel. Los autores proponen y comparan **dos arquitecturas encoder-decoder**: una genérica (**FlowNetSimple**, que apila las dos imágenes) y otra con una **capa de correlación** especializada (**FlowNetCorr**). El segundo obstáculo era el dato: obtener flujo verdadero de video real es casi imposible, así que fabrican el dataset sintético **Flying Chairs** (sillas 3D sobre fondos de Flickr, movidas con transformaciones afines), con ground-truth denso y exacto en cantidad arbitraria. El hallazgo central: redes entrenadas solo sobre datos irreales **generalizan** a Sintel y KITTI sin fine-tuning, a 5–10 fps. Es la bisagra entre el flujo clásico y el deep learning que sustenta la [Clase 36](/clases/clase-36).
{{< /paper-card >}}

---

## Contexto: el problema del flujo óptico y los métodos variacionales

El [flujo óptico](/fundamentos/flujo-optico) es el campo vectorial que describe, para cada píxel de la imagen 1, hacia dónde se desplazó en la imagen 2. Si $I_1$ e $I_2$ son dos frames consecutivos, se busca un campo $(u, v)$ tal que el píxel en $(x,y)$ de $I_1$ aparezca en $(x+u, y+v)$ en $I_2$. Es, en esencia, un **problema de correspondencia**. La formulación clásica arranca de la **hipótesis de constancia de brillo**, $I_1(x,y) = I_2(x+u, y+v)$; linealizando por Taylor se obtiene la **ecuación de restricción del flujo óptico**:

$$I_x\, u + I_y\, v + I_t = 0,$$

con $I_x, I_y$ los gradientes espaciales e $I_t$ el temporal. Una ecuación con dos incógnitas por píxel: el problema está **subdeterminado** (el *problema de apertura*). Desde **Horn y Schunck (1981)**, los **enfoques variacionales** dominaron el campo, añadiendo un **término de regularización** que impone suavidad al campo y minimizando una energía que combina datos + suavidad. Trabajos posteriores (DeepMatching/DeepFlow, EpicFlow) refinaron matching y grandes desplazamientos, pero comparten un rasgo clave: **no aprenden**; sus parámetros se fijan manualmente. Hubo intentos parciales de aprendizaje (regularizadores, matching con arquitecturas siamesas tipo Zbontar-LeCun), pero eran métodos **basados en parches** que delegaban la agregación espacial al postprocesamiento. La diferencia radical de FlowNet: las redes **predicen directamente campos de flujo completos**, de extremo a extremo.

## Contribución y método

La contribución es demostrar que una **CNN entrenada end-to-end** puede predecir directamente el campo de flujo con precisión competitiva a velocidades cercanas a tiempo real, rompiendo con décadas de métodos hechos a mano. Ambas redes siguen un esquema **encoder-decoder**: una parte **contractiva** que comprime espacialmente vía convoluciones con stride (extrayendo features abstractas a costa de resolución) y una parte **expansiva** de refinamiento que recupera la resolución para la predicción densa. No hay capas totalmente conectadas, lo que permite entradas de tamaño arbitrario.

**FlowNetSimple** (FlowNetS) es la opción más directa: **apilar las dos imágenes** por el eje de canales y alimentarlas a una red genérica, dejando que ella decida cómo extraer el movimiento. Los autores admiten que "nunca podemos estar seguros de que SGD lleve la red a ese punto", de ahí la motivación por una arquitectura más adaptada.

**FlowNetCorr** (FlowNetC) usa **dos streams idénticos** —uno por imagen— y los combina con una **capa de correlación** que compara parches de los mapas de características. Dados dos mapas $f_1, f_2$, la correlación de parches centrados en $x_1$ y $x_2$ es:

$$c(x_1, x_2) = \sum_{o \in [-k,k]\times[-k,k]} \langle f_1(x_1 + o),\, f_2(x_2 + o) \rangle$$

La operación es idéntica a un paso de convolución, pero convoluciona **datos con datos** en vez de datos con un filtro aprendido: por eso **no tiene pesos entrenables**. Como comparar todos los parches cuesta $w^2 \cdot h^2$, se **limita el desplazamiento máximo** a $d$, dando una salida de $w \times h \times D^2$ con $D = 2d+1$ (en los experimentos $d=20$). El **refinamiento** usa capas *upconvolucionales* (unpooling + convolución) que, en cada paso, concatenan los mapas upconvolucionados con los mapas correspondientes de la parte contractiva y una predicción de flujo más gruesa: así se preservan **tanto la información de alto nivel como el detalle fino local**. La pérdida es el **endpoint error (EPE)** —distancia euclidiana entre flujo predicho y ground-truth—, optimizada con Adam.

Un hallazgo casi contraintuitivo: aunque la capa de correlación se diseñó para *ayudar* con el matching, **incluso FlowNetSimple, sin ayuda explícita, aprende a predecir flujo con precisión competitiva**. La red cruda "puede aprender por sí sola".

## El dataset Flying Chairs

Las CNN necesitan ground-truth masivo para **aprender la tarea desde cero**, y los datasets existentes eran demasiado pequeños: Middlebury (8 pares con ground-truth), KITTI (194, ground-truth disperso por láser), Sintel (1.041, el mayor disponible). La solución es **Flying Chairs**: **22.872 pares** imagen–flujo con ground-truth denso al 100 %, construidos aplicando **transformaciones afines aleatorias** a fondos de Flickr y a modelos 3D de sillas superpuestos. Las transformaciones de las sillas son relativas a la del fondo —interpretable como cámara y objetos moviéndose a la vez— y se ajustan para que el **histograma de desplazamientos se parezca al de Sintel**. Estas imágenes "tienen poco en común con el mundo real", pero se generan en cantidades arbitrarias. El **data augmentation** (transformaciones geométricas, ruido, cambios de brillo/contraste/color) resultó crucial: quitarlo eleva el EPE en ~2 píxeles.

## Resultados

Evaluando con **EPE promedio** (menor es mejor) sobre Sintel, KITTI, Middlebury y el propio test de Flying Chairs:

- **Generalización sin fine-tuning.** Redes entrenadas solo sobre los datos no realistas de Flying Chairs rinden muy bien en flujo real, superando por ejemplo a LDOF: los datos sintéticos bastan para aprender flujo que transfiere.
- **Con fine-tuning (+ft)** superan al método de tiempo real EPPM en Sintel Final y KITTI, siendo el doble de rápidas.
- **FlowNetC vs. FlowNetS.** FlowNetC gana en Sintel Clean y Flying Chairs, pero la situación se invierte en Sintel Final (motion blur, niebla, no presentes en el entrenamiento): FlowNetC se **sobreajusta ligeramente** al tipo de datos.
- **En Flying Chairs baten el estado del arte**, incluyendo DeepFlow y EpicFlow.
- **Velocidad.** Son el mejor método de tiempo real, prediciendo flujo a hasta 10 pares/s a resolución completa de Sintel (~0,08 s por frame en GPU, frente a 16–65 s en CPU de EpicFlow/DeepFlow/LDOF).

## Limitaciones

- **Grandes desplazamientos.** El desplazamiento máximo de la capa de correlación ($d=20$) impide predecir movimientos muy grandes; ampliarlo cuesta cómputo.
- **Precisión absoluta.** El EPE aún está **por debajo del estado del arte** clásico; la ventaja está en velocidad y preservación de detalle, no en el EPE absoluto.
- **Ruido en el campo.** La salida cruda es ruidosa; para campos suaves y subpíxel se recurre todavía a un **refinamiento variacional** clásico de postprocesamiento (+v).
- **Realismo de los datos.** Flying Chairs solo modela movimientos afines de objetos rígidos sintéticos; datos más realistas mejorarían el desempeño (lo que ocurrió con FlowNet 2.0).

## Por qué importa para la Clase 36

El [flujo óptico](/fundamentos/flujo-optico) es el mecanismo canónico para **codificar el movimiento** entre frames, tema central de la [Clase 36](/clases/clase-36). FlowNet es la pieza que traslada ese cómputo del régimen clásico (variacional, hecho a mano) al de aprendizaje profundo. Dónde encaja:

- **Redes two-stream para reconocimiento de acciones.** La arquitectura [two-stream (Simonyan & Zisserman)](/papers/two-stream-simonyan-2014) separa un stream espacial (apariencia RGB) de un stream **temporal que consume flujo óptico** apilado. Un estimador rápido y aprendible como FlowNet es justo lo que alimenta ese stream temporal de forma eficiente, y su naturaleza end-to-end abre la puerta a integrarlo dentro de la propia red de reconocimiento.
- **Tracking y segmentación de video.** El flujo denso propaga máscaras, cajas o etiquetas entre frames y da una señal de movimiento robusta para trackers, en tiempo casi real.
- **Bisagra histórica.** FlowNet es el punto donde el análisis de movimiento adopta el paradigma que ya había transformado el reconocimiento, y encabeza la línea FlowNet 2.0 → PWC-Net → RAFT que define el estado del arte actual.
