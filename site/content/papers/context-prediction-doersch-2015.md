---
title: "Unsupervised Visual Representation Learning by Context Prediction (2015)"
weight: 313
math: true
---

{{< paper-card
    title="Unsupervised Visual Representation Learning by Context Prediction"
    authors="Carl Doersch, Abhinav Gupta, Alexei A. Efros"
    year="2015"
    venue="ICCV 2015"
    pdf="/papers/context-prediction-doersch-2015.pdf"
    arxiv="1505.05192" >}}
Paper fundacional del [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) moderno en visión, usado como caso de estudio canónico en la [Clase 28](/clases/clase-28). Su tesis: el **contexto espacial** dentro de una imagen es señal supervisora gratuita y abundante, igual que el contexto de palabras lo es en `word2vec`. El *pretext task* propuesto es el **posicionamiento relativo de parches**: dada una imagen sin etiquetar, predecir cuál de las **8 posiciones vecinas** ocupa un parche respecto a un parche central. Una **red siamesa** de pesos compartidos resuelve la tarea, y la representación resultante transfiere a detección de objetos en PASCAL VOC sin usar una sola etiqueta humana. Su aporte más duradero, además del *pretext*, es metodológico: la disección de los **atajos** que un modelo explota para "hacer trampa".
{{< /paper-card >}}

---

## Contexto: aprendizaje no supervisado en visión y el préstamo del NLP

Hacia 2015 la visión por computador había explotado gracias a datasets etiquetados a gran escala (ImageNet, AlexNet), pero escalar a colecciones "de escala Internet" chocaba con el costo prohibitivo de la anotación humana. El camino natural era el aprendizaje no supervisado, pero —como reconocen los autores con franqueza— tras décadas de esfuerzo los métodos no supervisados aún no extraían información útil de imágenes naturales de tamaño completo. El problema de fondo es epistemológico: sin etiquetas, *ni siquiera está claro qué debería representarse*.

Las familias previas fracasaban en imágenes reales de alta resolución: los **modelos generativos** (VAE, deep Boltzmann machines) batallan con texturas estocásticas y solo funcionaban en datasets simples como dígitos; los **autoencoders** de Le (2013) requirieron un millón de horas de CPU para descubrir apenas tres objetos; el **clustering sobre features hechos a mano** pierde información de forma y descubre clusters de follaje en vez de objetos.

La clave conceptual llega desde el **texto**. El modelo *skip-gram* de Mikolov et al. (2013) entrena una red para predecir las palabras vecinas de una palabra dada; eso convierte un problema no supervisado en uno *autosupervisado* —aprender una función desde un dato a su contexto— y produce *embeddings* semánticos como subproducto. Doersch et al. proponen el análogo visual. Pero copiar la receta no basta: **predecir píxeles es muchísimo más difícil que predecir palabras** por la enorme variedad de píxeles que produce un mismo objeto. La solución, heredada del NLP, es cambiar predicción por *discriminación*. La extensión naíf —distinguir imágenes reales de imágenes con un parche reemplazado al azar— sería trivial (basta mirar estadísticas de color de bajo nivel), así que se discrimina entre **configuraciones de parches de la misma imagen**, que por construcción comparten iluminación y color y fuerzan razonamiento de alto nivel.

## El pretext task: posicionamiento relativo

El planteamiento formal es una clasificación de **8 clases**. Se muestrea un parche central y uno de sus ocho vecinos (arriba, arriba-derecha, derecha, …, arriba-izquierda); se presenta el par $(P_1, P_2)$ a la red *sin* información sobre su ubicación original, y la red produce un softmax sobre las ocho configuraciones espaciales $Y \in \{1,\dots,8\}$. El azar acierta 12.5%.

La hipótesis central, repetida a lo largo del paper: **"hacerlo bien en esta tarea requiere entender escenas y objetos"**. Los objetos consisten en múltiples partes detectables que ocurren en una configuración espacial específica; si no hay configuración específica, es *stuff* (textura, fondo), no objeto. Por tanto, una buena representación para este *pretext* está obligada a extraer objetos y sus partes para razonar sobre su ubicación relativa.

## Arquitectura: red siamesa de fusión tardía

La arquitectura es una **red siamesa** ("late-fusion"): un par de torres estilo AlexNet que procesan cada parche por separado hasta `fc6`, punto en el cual las representaciones se fusionan. Para las capas que procesan un solo parche, los **pesos están atados (compartidos)** entre ambos lados —exactamente lo que la Clase 28 describe como "2 redes conv con pesos compartidos"—, de modo que se computa la misma función de *embedding* para ambos parches.

La pila por torre sigue AlexNet (`conv1`…`conv5` → `pool5` → `fc6` de 4096), y tras `fc6` se concatenan las dos torres y siguen `fc7` → `fc8` → `fc9`(8) → softmax. Como solo dos capas reciben entrada de ambos parches, hay **capacidad limitada para razonamiento conjunto**: esto obliga a la red a hacer el grueso del trabajo semántico *por parche, separadamente* —justo lo que se busca para que la representación de un parche individual sea útil después.

Detalles de muestreo: imágenes redimensionadas a 150K–450K píxeles, parches de **96×96** tomados de una cuadrícula, con un **gap de 48 píxeles** entre parches y **jitter** aleatorio de ±7 píxeles. El entrenamiento corrió sobre ImageNet 2012 (~1.3M imágenes, descartando las etiquetas) durante **~4 semanas en una GPU K40**. Con SGD simple las predicciones degeneraban a una distribución uniforme (colapso en un *saddle point*); la solución fue **batch normalization sin escala ni shift** más **momentum alto** (~.999).

## El combate a los atajos (la lección de manual)

Esta es la sección más influyente metodológicamente. El principio: un *pretext* debe **forzar** a la red a extraer la información deseada *sin* dejar atajos triviales. El paper identifica y neutraliza tres atajos sucesivos:

1. **Continuidad de bordes y texturas.** Patrones que continúan entre parches adyacentes delatarían la respuesta sin entender nada semántico. Mitigación: el **gap** de ~medio ancho de parche.
2. **Líneas largas que cruzan parches vecinos.** Incluso con gap, una recta que atraviesa parches contiguos revela la configuración. Mitigación: el **jitter** aleatorio de hasta 7 píxeles.
3. **Aberración cromática (el más insidioso).** El hallazgo que más sorprendió a los autores. La lente enfoca distintas longitudes de onda de forma diferente; en muchas cámaras el canal verde se "encoge" hacia el centro respecto a magenta (rojo+azul). Una ConvNet **puede aprender a localizar un parche respecto a la lente misma** detectando esa separación verde-magenta; conociendo la *posición absoluta* en la lente, la posición *relativa* se vuelve trivial —y la red resuelve el *pretext* sin aprender semántica alguna. Dos mitigaciones probadas:
   - **Proyección:** sea $a = [-1, 2, -1]$ el eje verde-magenta; se aplica $B = I - a^T a / (a a^T)$ a cada píxel para sustraer su proyección sobre ese eje.
   - **Color dropping:** descartar al azar 2 de los 3 canales por parche, reemplazándolos con ruido gaussiano de baja varianza.

Ambas estrategias rinden similar. La lección más honesta del paper es que el diseñador *no puede anticipar todos los atajos a priori*: hay que descubrirlos empíricamente y neutralizarlos uno a uno, lo que deja la pregunta inquietante de cuáles *no* se detectaron.

## Transferencia a PASCAL VOC y otros resultados

La representación se usa como pre-entrenamiento dentro del pipeline **R-CNN** sobre PASCAL VOC 2007 (sin *bounding-box regression*). Resultados de mAP del paper original (AlexNet):

| Modelo (AlexNet-style) | mAP |
|---|---|
| Scratch-R-CNN (AlexNet desde cero) | 40.7 |
| Scratch-Ours (arq. propia desde cero) | 39.8 |
| **Ours-projection** | **45.7** |
| **Ours-color-dropping** | **46.3** |
| Ours-Yahoo100m (Flickr 100M) | 44.2 |
| ImageNet-R-CNN (etiquetas ImageNet) | 54.2 |

El pre-entrenamiento autosupervisado da un **boost de ~6% de mAP** sobre la arquitectura desde cero, quedando a ~8% del R-CNN supervisado en ImageNet —el mejor resultado conocido en VOC 2007 sin etiquetas externas en su momento. Escalando a un backbone **VGG-16** el método alcanza **61.7 mAP**, muy por encima de una inicialización por K-means (42.4), lo que evidencia que casi todo el boost viene del pre-entrenamiento no supervisado. El experimento con Flickr 100M (recolectado automáticamente) confirma que el método no depende de los sesgos curatoriales de ImageNet.

Otros resultados clave: en **estimación de normales de superficie** (NYUv2) iguala casi exactamente a un modelo ImageNet etiquetado (Mean 33.2 vs 33.3), mostrando utilidad más allá de tareas de objetos; permite **minería visual no supervisada** de objetos en VOC 2011 y Paris Street View; y en la propia tarea de *pretext* acierta **38.4%** (azar 12.5%), con escaso overfitting.

{{< callout type="warning" >}}
**Nota sobre la cifra 65.3 de la Clase 28.** La tabla comparativa de la clase reporta el posicionamiento relativo en **65.3**, mientras el paper original reporta **~46.3 mAP** (AlexNet, color-dropping, VOC-2007). No es una contradicción: la cifra de la clase proviene de un **benchmark armonizado posterior** que evalúa los distintos *pretext tasks* (posicionamiento relativo, jigsaw, colorización, rotación) con backbones y protocolos modernos estandarizados. El número del paper es el de la época, con AlexNet en 2015. Lo invariante es el mensaje comparativo: el posicionamiento relativo de Doersch et al. es competitivo entre los *pretext* espaciales.
{{< /callout >}}

## Impacto: el origen de los pretext espaciales

Este paper es ampliamente reconocido como uno de los **detonantes del aprendizaje autosupervisado moderno en visión**:

- **Fundó la familia de *pretext tasks* espaciales.** El posicionamiento relativo de pares fue generalizado casi de inmediato por Noroozi & Favaro (2016) a los **jigsaw puzzles**: una grilla de 3×3=9 parches barajados cuya permutación hay que predecir, citando explícitamente a Doersch et al. como antecedente. De ahí siguieron rotación (Gidaris et al., 2018), [colorización](/papers/colorization-zhang-2016) (Zhang et al., 2016) e in-painting (Pathak et al., 2016).
- **Institucionalizó el combate a los atajos.** La discusión de aberración cromática se volvió el ejemplo de manual de *shortcut learning* en SSL. Diseñar un *pretext* hoy implica, por defecto, preguntarse "¿qué atajo trivial podría resolver esto?".
- **Antecedió al contrastive learning.** La intuición de que parches del mismo objeto deben acercarse en el espacio de *embedding*, junto con la arquitectura siamesa de pesos compartidos, prefigura la línea que culmina en SimCLR, MoCo y BYOL (2020): la "predicción de contexto" se reemplaza por "invarianza a aumentaciones", pero la maquinaria siamesa permanece.
- **Validó la transferencia instancia → categoría.** Que un objetivo definido sobre una sola imagen mejore tareas de categoría entre imágenes fue un resultado conceptualmente importante que dio confianza a toda la agenda de SSL.

## Por qué importa para la Clase 28

La [Clase 28](/clases/clase-28) usa este trabajo como **caso de estudio canónico del posicionamiento relativo**, y el mapeo es directo:

- **El *pretext*.** La clase lo presenta exactamente como en el paper: dos redes convolucionales con pesos compartidos que reciben dos parches y predicen cuál de las 8 posiciones relativas ocupa uno respecto al otro —el análogo visual de skip-gram que la clase contrasta con `word2vec`.
- **El slide "qué se aprende".** La clase muestra los **vecinos más cercanos** recuperados por la representación autosupervisada (ruedas con ruedas, personas con personas), la evidencia cualitativa de que resolver el *pretext* requiere reconocer objetos y partes.
- **La gran moraleja.** El valor pedagógico más duradero no es solo el *pretext* sino la sección de atajos: gap, jitter y, sobre todo, la aberración cromática son el ejemplo perfecto de que *un pretext mal diseñado se resuelve sin aprender nada útil*. Esto conecta con el hilo de la clase sobre por qué la comunidad migró hacia objetivos contrastivos/de invarianza, más difíciles de "hackear" con atajos de bajo nivel.

Este análisis enlaza con el fundamento transversal [/fundamentos/aprendizaje-autosupervisado](/fundamentos/aprendizaje-autosupervisado), con la [Clase 28](/clases/clase-28) y con el paper hermano de [colorización (Zhang et al., 2016)](/papers/colorization-zhang-2016).
