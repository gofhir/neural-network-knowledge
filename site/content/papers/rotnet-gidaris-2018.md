---
title: "RotNet: Predicting Image Rotations (2018)"
weight: 318
math: true
---

{{< paper-card
    title="Unsupervised Representation Learning by Predicting Image Rotations"
    authors="Spyros Gidaris, Praveer Singh, Nikos Komodakis"
    year="2018"
    venue="ICLR 2018"
    pdf="/papers/rotnet-gidaris-2018.pdf"
    arxiv="1803.07728" >}}
RotNet propone una de las *pretext tasks* más citadas del [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) en visión: entrenar una ConvNet para que reconozca **la rotación 2D que se aplicó a una imagen**, eligiendo entre cuatro posibilidades discretas (0, 90, 180 y 270 grados). Es un problema de clasificación de 4 clases donde la etiqueta es *gratis* —se genera automáticamente al rotar—. La tesis: esta tarea "aparentemente simple" provee una señal de supervisión muy poderosa para aprender características semánticas. El número emblemático es **54.4% mAP en detección PASCAL VOC 2007, a solo 2.4 puntos del supervisado**.
{{< /paper-card >}}

---

## La idea: predecir la rotación como pretexto

El argumento conceptual que sostiene todo el trabajo es elegante: para que una ConvNet pueda decir en cuántos grados fue rotada una imagen, *necesariamente* tiene que haber aprendido primero a reconocer y localizar los objetos presentes, identificar su tipo, sus partes semánticas (ojos, narices, colas, cabezas) y la **orientación canónica** ("up-standing") con que esos objetos suelen aparecer en fotos capturadas por humanos. No hay atajo de bajo nivel que resuelva la rotación; hay que *entender la escena*. La intuición de la Figura 1 del paper es directa: alguien que no conoce los conceptos de los objetos en una imagen no puede reconocer qué rotación se le aplicó.

El paper se distingue cuidadosamente de dos trabajos previos con transformaciones geométricas. ExemplarCNN (Dosovitskiy et al., 2014) entrena para que las representaciones sean *invariantes* a transformaciones; RotNet hace lo opuesto, entrena para *reconocer* la transformación aplicada. Y el método de egomotion (Agrawal et al., 2015) usa una arquitectura siamesa con *dos* cuadros y predice la transformación de cámara por regresión; RotNet recibe una *sola* imagen, sin acceso a la original, y clasifica la rotación que se le aplicó.

## El método

Dado un conjunto de imágenes, se define un conjunto de $K$ transformaciones geométricas discretas. **Es la elección de ese conjunto la que define la dificultad y el valor semántico del pretexto.** La propuesta concreta: rotaciones por múltiplos de 90 grados, con $K = 4$ y $g(X|y) = \text{Rot}(X, (y-1)\cdot 90)$. El objetivo minimiza la *cross-entropy* promediada sobre las cuatro rotaciones de cada imagen:

$$\text{loss}(X_i, \theta) = -\frac{1}{K} \sum_{y=1}^{K} \log\!\big(F^{y}(g(X_i|y)\,|\,\theta)\big)$$

donde $F^{y}(\cdot)$ es la probabilidad predicha para la rotación $y$.

**Las rotaciones se implementan sin interpolación**, con operaciones exactas de *flip* y *transpose* (90° = transpose + flip vertical; 180° = doble flip; 270° = flip vertical + transpose). Esta exactitud es la que garantiza la ausencia de artefactos visuales explotables. Un **truco de entrenamiento** clave: alimentar a la red con las cuatro copias rotadas de una imagen simultáneamente en el mismo mini-batch (en vez de muestrear una sola rotación) estabiliza y mejora el aprendizaje.

## Por qué rotaciones de 90 grados (la sutileza del diseño)

El paper da tres razones de diseño que conviene retener, porque ahí está toda la sutileza del trabajo:

- **Forzar semántica.** Es prácticamente imposible reconocer la rotación sin haber aprendido a reconocer clases de objetos, sus partes y su orientación dominante. La tarea obliga a localizar objetos salientes y relacionar su orientación con la que cada tipo suele tener.
- **Ausencia de artefactos de bajo nivel.** Las rotaciones por múltiplos de 90° (flip/transpose) **no dejan rastros detectables**. Si se usaran escala o aspecto, harían falta rutinas de redimensionado que dejan rastros de bajo nivel que la red explotaría de forma trivial. RotNet *no requiere preprocesamiento especial* para bloquear atajos, a diferencia de muchos competidores.
- **Buena definición (well-posedness).** Como las fotos humanas muestran objetos erguidos, reconocer la rotación está bien definido —sin ambigüedad, salvo objetos perfectamente redondos—. En contraste, la escala varía mucho entre fotos, lo que haría mal definida una tarea basada en escala.

## La simplicidad como virtud

El pretexto tiene el **mismo costo computacional** que el aprendizaje supervisado y una velocidad de convergencia similar —mucho más rápida que los enfoques basados en reconstrucción de imágenes—. Su AlexNet entrena en unos 2 días en una sola GPU Titan X, y puede adoptar trivialmente los esquemas de paralelización del aprendizaje supervisado, lo que lo hace candidato ideal para SSL a escala de internet. La simplicidad, lejos de ser una concesión, es la virtud central: un pretexto trivial de implementar puede competir con los más elaborados *siempre que esté bien diseñado para forzar semántica y bloquear atajos*.

## Evidencia cualitativa: filtros y atención

Dos visualizaciones sustentan el argumento semántico:

- **Mapas de atención** (Figura 3): el modelo entrenado en rotación se enfoca en partes de objetos de alto nivel (ojos, narices, colas, cabezas) y, comparado con un modelo supervisado de reconocimiento, enfoca aproximadamente las mismas regiones. Además son **equivariantes**: las cuatro copias rotadas producen mapas esencialmente iguales —la red se fija en las mismas partes sin importar la rotación.
- **Filtros de primera capa** (Figura 4): los filtros aprendidos por RotNet son mayormente bordes orientados en múltiples orientaciones y frecuencias, con **incluso más variedad que los del modelo supervisado**. Esta comparación lado a lado es la figura que la [Clase 28](/clases/clase-28) retoma en su sección de rotaciones.

## Resultados (números reales)

El protocolo congela los features autosupervisados y entrena clasificadores (lineales o no) encima, midiendo qué tan buenas son las representaciones.

**CIFAR-10 (arquitectura Network-In-Network):**

| Aspecto | Resultado |
|---|---|
| Mejor capa (2.º bloque conv) | 88-89% accuracy de reconocimiento |
| Número óptimo de rotaciones | **4** (89.06%); 8 rot. → 88.51%; 2 rot. (0/180) → 87.46% |
| RotNet + conv vs. NIN supervisado | **91.16% vs. 92.80%** (brecha de solo 1.64 pts) |
| Con fine-tuning | 92.17% (casi iguala al supervisado) |
| Inicialización aleatoria (referencia) | 72.50% |

Supera a Roto-Scat+SVM (82.3%), ExemplarCNN (84.3%) y DCGAN (82.8%). En **semi-supervisado**, RotNet *supera* al supervisado cuando hay menos de 1000 ejemplos etiquetados por categoría, y la ventaja crece a medida que escasean las etiquetas.

**ImageNet (AlexNet, preentrenado sin etiquetas):** con clasificadores no lineales, RotNet logra **50.0%** (Conv4) y **43.8%** (Conv5) top-1, superando a todos los competidores por más de 4 y ~8 puntos respectivamente (referencia supervisada: 59.7%). Con clasificadores lineales lidera en Conv3/Conv4/Conv5 sobre Context, Colorization, Jigsaw, BiGAN, Split-Brain y Counting.

**PASCAL VOC — transfer learning (el resultado más citado):**

| Tarea | RotNet | Supervisado |
|---|---|---|
| **Detección VOC 2007** | **54.4% mAP** | 56.8% (brecha de 2.4 pts) |
| Clasificación VOC 2007 | 70.87% (fc6-8) / 72.97% (FT) | — |
| Segmentación VOC 2012 | 39.1% mIoU (mejor no supervisado) | — |

En detección supera a Jigsaw (53.2%), Context (51.1%), Counting (51.4%) y Colorization (46.9%). En conjunto, RotNet logra estado del arte en *todos* los benchmarks evaluados, estrechando consistentemente la brecha con el supervisado.

## Limitaciones

- **Sesgo de orientación canónica.** El argumento de well-posedness depende de que las imágenes muestren objetos erguidos. La tarea pierde sentido en objetos rotacionalmente simétricos o redondos y en dominios sin "arriba" definido —imágenes aéreas, satelitales, microscopía, escaneos médicos—.
- **No supera al supervisado.** La brecha se estrecha dramáticamente (2.4 pts en detección) pero persiste en el régimen de datos abundantes.
- **Especialización de capas tardías.** Las capas profundas se especializan en el pretexto y *degradan* su utilidad downstream; el mejor feature no es el más profundo, lo que obliga a elegir la capa de extracción con cuidado.
- **Época y arquitecturas.** Los experimentos usan AlexNet y NIN; el paper precede a ResNets profundas y al contrastive learning, que luego cambiarían el panorama del SSL.

## Impacto: anticipa SimCLR, MoCo y BYOL

RotNet se convirtió en un pretexto de referencia del SSL en visión, y su legado es metodológico. La lección —*la simplicidad como virtud, no como concesión*— anticipa la filosofía de los métodos contrastivos posteriores como [SimCLR](/papers/simclr-chen-2020) y MoCo, y de los predictivos como BYOL: pretextos conceptualmente limpios, baratos de computar, escalables a datos masivos, sin preprocesamiento defensivo.

Tres ideas quedaron en el canon del campo: (1) *la elección del pretexto define qué se aprende*, y un buen pretexto es uno que no se puede resolver sin entender la escena; (2) el protocolo de evaluación por *linear/non-linear probing* sobre features congelados, capa por capa, que se volvió estándar; y (3) la observación de que las capas tardías se sobre-especializan en el pretexto, motivando el diseño de proyectores y cabezas desechables en métodos posteriores.

## Conexión con la Clase 28

La [Clase 28](/clases/clase-28) dedica una sección a **"Rotaciones para autosupervisión"** que es esencialmente una exposición de este paper. Dos elementos provienen directamente de aquí: la **comparación de filtros aprendidos** (supervisado vs. predicción de rotación, Figura 4) y la **accuracy en PASCAL VOC 2007** (54.4% mAP en detección, a 2.4 puntos del supervisado), que la clase usa para argumentar que un pretexto simple puede acercarse mucho al techo supervisado.

Para los conceptos transversales, ver el fundamento [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) y el hub de la [Clase 28](/clases/clase-28).

## Notas y enlaces

- arXiv: https://arxiv.org/abs/1803.07728
- Código: https://github.com/gidariss/FeatureLearningRotNet
- Venue: ICLR 2018 (International Conference on Learning Representations).
- Afiliación: University Paris-Est, LIGM — École des Ponts ParisTech.
