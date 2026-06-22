---
title: "Context Encoders: Feature Learning by Inpainting (2016)"
weight: 311
math: true
---

{{< paper-card
    title="Context Encoders: Feature Learning by Inpainting"
    authors="Deepak Pathak, Philipp Krähenbühl, Jeff Donahue, Trevor Darrell, Alexei A. Efros"
    year="2016"
    venue="CVPR 2016"
    pdf="/papers/context-encoders-pathak-2016.pdf"
    arxiv="1604.07379" >}}
Uno de los papers fundacionales del [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) en visión. Su tesis cabe en una frase: si entrenamos una red convolucional para **rellenar (inpaint) una región faltante** de una imagen condicionada en sus alrededores, la red se ve forzada a entender la escena completa para producir una hipótesis plausible de lo que falta. Lo decisivo es que ese objetivo —predecir píxeles— **no necesita etiquetas**: la supervisión sale gratis de la propia imagen. El modelo es un **context encoder**: una red encoder-decoder donde el encoder comprime el contexto (la imagen con el agujero) en un latente compacto y el decoder produce el contenido faltante. Las características que aprende sin etiquetas transfieren competitivamente a clasificación, detección y segmentación en PASCAL VOC. Es el ancestro reconocible de los [Masked Autoencoders (MAE)](/papers/mae-he-2022) de 2022.
{{< /paper-card >}}

---

## Contexto: el aprendizaje autosupervisado temprano

Hacia 2014-2016 la visión vivía bajo el dominio del preentrenamiento supervisado en ImageNet: entrenar una CNN (típicamente AlexNet) sobre un millón largo de imágenes etiquetadas y reutilizar esas características como inicialización para tareas más pequeñas. La pregunta abierta que el paper ataca de frente es: **¿pueden aprenderse características igual de informativas a partir de imágenes crudas, sin etiqueta alguna?**

La respuesta es lo que hoy llamamos [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado): diseñar una tarea artificial —un *pretext task*— cuya supervisión se deriva automáticamente de la estructura de los datos, de modo que resolverla obligue a la red a aprender representaciones útiles. Hacia 2015-2016 emergían varias familias de pretext tasks: señal temporal en video, ego-movimiento (usar odometría como supervisión) y contexto espacial.

El trabajo más comparable es **Doersch et al. (2015)**, que entrena una red a predecir la *posición relativa* de dos parches vecinos ("¿el parche A está arriba o abajo de B?"). El context encoder se diferencia en algo profundo: Doersch resuelve una **tarea discriminativa** (clasificar entre 8 posiciones), mientras que el context encoder resuelve un **problema de predicción puro** (¿qué intensidades de píxel deben ir en el agujero?). Los autores trazan el paralelo lingüístico explícito: como **word2vec** formula el aprendizaje de embeddings como predicción de palabra dado su contexto, el context encoder es el "word2vec de los píxeles". Esto trae tres ventajas: señal supervisoria mucho más rica (predecir ~15.000 valores reales por ejemplo vs. 1 entre 8), entrenamiento más rápido (14 horas vs. 4 semanas) y mayor dificultad de "hacer trampa" con atajos de bajo nivel como la aberración cromática.

## Inpainting como pretext task

La idea central es elegante: dada una imagen con una región removida, entrenar una CNN para que **regrese a los valores de los píxeles faltantes**. El modelo se llama context encoder porque tiene un encoder que captura el contexto en un latente compacto y un decoder que produce el contenido ausente.

Está íntimamente relacionado con los autoencoders, pero las diferencias son todo el aporte:

- Un **autoencoder** pasa la imagen por un cuello de botella e intenta reconstruirla idéntica. Puede copiar píxeles sin aprender nada semántico.
- Un **denoising autoencoder** corrompe la entrada y pide deshacer el daño, pero la corrupción suele ser local y de bajo nivel, sin requerir semántica.
- El **context encoder** rellena áreas faltantes *grandes*, donde no hay pistas en los píxeles cercanos. Para inpaint la fachada de una casa, "una ventana entera tiene que ser conjurada de la nada". Visto así, es un denoising autoencoder donde la corrupción es **espacialmente mucho mayor**, lo bastante grande como para exigir información semántica para deshacerse.

La tarea es además **inherentemente multimodal**: hay múltiples maneras igualmente plausibles de rellenar una región. Esa multimodalidad justifica la pérdida combinada.

## Arquitectura: encoder-decoder y la capa channel-wise FC

El **encoder** deriva de AlexNet: dada una entrada de 227×227, usa las cinco primeras capas convolucionales más `pool5` para producir una representación de **6×6×256 = 9216** dimensiones, entrenada desde cero (no para clasificar ImageNet).

La sutileza arquitectónica central es la **capa channel-wise fully connected**. Si el encoder solo tuviera convoluciones, no habría forma de propagar información directamente de una esquina del mapa de características a otra: las convoluciones conectan mapas entre sí, pero no todas las ubicaciones *dentro* de un mapa. Esa propagación global la suelen hacer las capas fully-connected, pero conectar totalmente encoder y decoder costaría **más de 100 millones de parámetros**, inviable en las GPU de la época.

La solución propaga información *dentro* de cada mapa de características, pero *no entre* mapas distintos. Con $m$ mapas de tamaño $n \times n$, el conteo de parámetros pasa de $m^2 n^4$ (FC completa) a $m n^4$ —un factor $m$ de ahorro—. Luego una convolución de stride 1 recupera la mezcla entre canales. Esta capa permite que "cada unidad del decoder razone sobre el contenido entero de la imagen" sin el costo prohibitivo. Nótese que, a diferencia de un autoencoder, **no se reconstruye la entrada completa, así que no hace falta un cuello de botella pequeño**.

El **decoder** parte de ese latente y aplica **cinco capas up-convolucionales** (deconvoluciones con filtros aprendidos y ReLU), que constituyen un *upsampling no lineal ponderado* hasta el tamaño objetivo.

## Función de pérdida: L2 + adversarial

Como la tarea es multimodal, los autores **desacoplan la carga en una pérdida conjunta** con dos términos que se reparten responsabilidades.

**Pérdida de reconstrucción (L2),** una distancia enmascarada y normalizada:

$$L_{rec}(x) = \lVert \hat{M} \odot (x - F((1-\hat{M}) \odot x)) \rVert_2^2$$

donde $\hat{M}$ es la máscara binaria de la región removida, $F$ es el context encoder y $\odot$ es producto elemento a elemento (L1 y L2 dieron resultados similares). Su defecto es el corazón de la motivación: el L2 captura el contorno aproximado pero produce resultados **borrosos**. La razón es estadística: cuando hay múltiples modos plausibles, lo más "seguro" para el L2 es predecir la **media** de la distribución, y la media de varias soluciones nítidas es una imagen promediada.

**Pérdida adversarial (GAN)** para combatir el desenfoque. Un discriminador $D$ aprende a distinguir muestras reales de generadas mientras el generador (el propio context encoder, $G \equiv F$) intenta confundirlo; esto "escoge un modo particular de la distribución" y produce detalle nítido. Un detalle crucial: los autores **no condicionan el discriminador en el contexto**, porque las GAN condicionales no entrenan —el discriminador explota la discontinuidad perceptual entre la región generada y el contexto para clasificar trivialmente—:

$$L_{adv} = \max_D \; \mathbb{E}_{x \in X}\left[\log(D(x)) + \log\left(1 - D(F((1-\hat{M}) \odot x))\right)\right]$$

**Pérdida conjunta** como combinación lineal, con $\lambda_{rec} = 0{,}999$ y $\lambda_{adv} = 0{,}001$:

$$L = \lambda_{rec} L_{rec} + \lambda_{adv} L_{adv}$$

El reparto de roles es claro: el L2 captura la **estructura global y la coherencia con el contexto**; el adversarial escoge un **modo nítido**. Detalle honesto: el adversarial **solo se usó para inpainting**, porque con AlexNet (la arquitectura del aprendizaje de características) el entrenamiento conjunto divergía; por eso los resultados de transferencia usan solo reconstrucción.

## Estrategias de enmascaramiento

Cómo se elige la región a remover importa para la *generalidad* de las características:

- **Región central.** Buen inpainting, pero la red aprende features de bajo nivel que se aferran al borde de la máscara y no generalizan.
- **Bloque aleatorio.** Varias máscaras menores en posiciones variables (hasta 1/4 de la imagen); mejora, aunque aún tiene bordes nítidos.
- **Región aleatoria.** Formas arbitrarias deformadas y pegadas en lugares arbitrarios para eliminar por completo bordes constantes.

En la práctica, región y bloque aleatorio producen características similarmente generales, superando con holgura a la central. Los autores usan **dropout de región aleatoria** para todos los experimentos basados en características.

## Resultados de transferencia a PASCAL VOC

Para consistencia con trabajos previos se usa AlexNet en el encoder (solo reconstrucción). El entrenamiento es rápido: ~100K iteraciones, **14 horas en una Titan X**. Para preentrenar se toman los pesos del encoder hasta `pool5` y se reinicializan `fc6`/`fc7`. La tabla de transferencia (clasificación y detección sobre VOC 2007, segmentación sobre VOC 2012):

| Preentrenamiento | Supervisión | Tiempo | Clasif. | Detec. | Segm. |
|---|---|---|---|---|---|
| ImageNet | 1000 etiquetas | 3 días | 78,2% | 56,8% | 48,0% |
| Gaussiano aleatorio | inicialización | < 1 min | 53,3% | 43,4% | 19,8% |
| Autoencoder | — | 14 h | 53,8% | 41,9% | 25,2% |
| Wang et al. | movimiento | 1 semana | 58,7% | 47,4% | — |
| Doersch et al. | contexto relativo | 4 semanas | 55,3% | 46,6% | — |
| **Context encoder** | **contexto** | **14 h** | **56,5%** | **44,5%** | **30,0%** |

Lecturas clave: la **inicialización aleatoria queda ~25% por debajo de ImageNet** (53,3% vs. 78,2%), pero sin etiquetas; el context encoder **supera con claridad al autoencoder simple** y es **competitivo con los métodos auto/débilmente supervisados concurrentes** a una fracción del cómputo. En inpainting puro, el método paramétrico supera ampliamente al vecino más cercano (PSNR 18,58 dB con pérdida conjunta vs. 12,79 dB con HOG) y al Content-Aware Fill de Photoshop en casos semánticos.

> **Nota.** La Clase 28 cita "inpainting 56,5 vs. aleatorio 53,3 vs. ImageNet 79,9". Los dos primeros coinciden con la tabla; el 79,9 difiere levemente del 78,2 del paper (cifra redondeada de otra corrida). El mensaje es el mismo: el inpainting cierra buena parte de la brecha entre azar y supervisión completa sin usar una sola etiqueta.

## Limitaciones reconocidas

- **El L2 conservador pierde detalle:** prefiere la media borrosa de los modos. El adversarial lo corrige, pero a costa de estabilidad.
- **El adversarial no converge con AlexNet:** los mejores resultados de inpainting (con adversarial) y los de transferencia (sin él, con AlexNet) usan arquitecturas distintas. No se obtiene lo mejor de ambos mundos a la vez.
- **Peor que la síntesis de textura en regiones texturadas:** la fortaleza del método es lo semántico, no lo texturado.
- **El enmascaramiento central induce atajos** que no generalizan, de ahí la región aleatoria.
- **Pregunta abierta de fondo:** los autores admiten que "no está claro aún si requerir una generación fiel de píxeles es necesario para aprender buenas características". Esta duda orientó el SSL posterior hacia objetivos contrastivos y predictivos en el espacio latente.

## Impacto: ancestro del MAE

Los context encoders fueron **uno de los primeros pretext tasks de naturaleza generativa** —predecir píxeles ausentes— frente a los discriminativos de la época (posición relativa, rotación, jigsaw). Su huella es doble. En **generación de imágenes**, la receta "reconstrucción + adversarial" para combatir el desenfoque del L2 se volvió estándar y alimentó el inpainting profundo y la traducción imagen-a-imagen (pix2pix y descendientes). En **aprendizaje autosupervisado**, la idea de *masked prediction* —ocultar parte de la entrada y predecirla desde el resto— es la semilla conceptual directa del *masked language modeling* de BERT en texto y, de forma casi literal en visión, de los [Masked Autoencoders (MAE, He et al., 2022)](/papers/mae-he-2022), que retoman el gesto exacto de Pathak —enmascarar parches y reconstruirlos con un encoder-decoder— pero a escala de Vision Transformers.

## Por qué importa para la Clase 28

La [Clase 28](/clases/clase-28) presenta el [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) como el paradigma donde la supervisión se extrae de los propios datos mediante una tarea pretexto, y usa el context encoder como ejemplo canónico de **auto-predicción en imágenes**:

- **Inpainting como auto-predicción.** Ocultar una región y reconstruirla fuerza a entender el contexto; la "etiqueta" es la propia porción que escondimos. Por eso las características transfieren.
- **La tabla de fine-tuning en PASCAL VOC.** Evidencia cuantitativa de que el pretext task cierra una porción significativa de la brecha entre azar e ImageNet, sin etiquetas.
- **Conexión con autoencoders.** El context encoder es la extensión natural del autoencoder que la clase introduce antes: lleva la corrupción al extremo espacial para que deshacerla *requiera* semántica.
- **El porqué de la pérdida combinada.** La tensión entre "predecir la media segura" (borrosa) y "predecir una muestra realista" (nítida) reaparece en todo el modelado generativo y SSL; este es el lugar canónico donde verla por primera vez.

## Notas y enlaces

- Preprint: arXiv:1604.07379 — [arxiv.org/abs/1604.07379](https://arxiv.org/abs/1604.07379)
- Afiliación: University of California, Berkeley.
- Paper hermano en color/contexto: [Colorization (Zhang et al., 2016)](/papers/colorization-zhang-2016).
- Descendiente directo en visión: [Masked Autoencoders (He et al., 2022)](/papers/mae-he-2022).
