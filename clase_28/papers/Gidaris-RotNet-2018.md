# Unsupervised Representation Learning by Predicting Image Rotations (RotNet) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Unsupervised Representation Learning by Predicting Image Rotations*.
- **Autores:** Spyros Gidaris, Praveer Singh, Nikos Komodakis (University Paris-Est, LIGM — École des Ponts ParisTech).
- **Venue:** Publicado como *conference paper* en ICLR 2018 (International Conference on Learning Representations).
- **Año:** 2018. **Preprint:** arXiv:1803.07728v1 (21 mar 2018), [arxiv.org/abs/1803.07728](https://arxiv.org/abs/1803.07728).
- **Código:** [github.com/gidariss/FeatureLearningRotNet](https://github.com/gidariss/FeatureLearningRotNet).

Este paper propone una de las *pretext tasks* más citadas e influyentes del aprendizaje autosupervisado en visión: entrenar una red convolucional (ConvNet) para que reconozca **la rotación 2D que se le aplicó a una imagen**, eligiendo entre cuatro posibilidades discretas —0, 90, 180 y 270 grados—. Es decir, un problema de clasificación de 4 clases donde la etiqueta es *gratis*: se genera automáticamente al rotar la imagen, sin anotación humana. La tesis del paper, que sus autores defienden cualitativa y cuantitativamente, es que esta tarea "aparentemente simple" provee en realidad "una señal de supervisión muy poderosa para el aprendizaje de características semánticas".

El argumento conceptual que sostiene todo el trabajo es elegante: para que una ConvNet pueda decir en cuántos grados fue rotada una imagen, *necesariamente* tiene que haber aprendido primero a reconocer y localizar los objetos presentes, identificar su tipo, sus partes semánticas (ojos, narices, colas, cabezas) y la orientación canónica ("up-standing") con que esos objetos suelen aparecer en fotografías capturadas por humanos. No hay atajo de bajo nivel que resuelva la rotación; hay que *entender la escena*. La intuición, expresada en la Figura 1 del paper, es directa: alguien que no conoce los conceptos de los objetos en una imagen no puede reconocer la rotación que se le aplicó.

Los resultados respaldan la apuesta. Sobre el benchmark de detección de PASCAL VOC 2007, el AlexNet preentrenado de forma no supervisada con RotNet alcanza **54.4% mAP**, apenas **2.4 puntos por debajo** del caso supervisado (56.8%) —cerrando dramáticamente la brecha entre features supervisadas y no supervisadas que hasta entonces parecía infranqueable—. Para el Laboratorio y la Clase 28 esto importa porque RotNet es el ejemplo canónico de la sección "Rotaciones para autosupervisión": muestra que la simplicidad, lejos de ser una limitación, puede ser la virtud central de un buen pretexto.

## 2. Contexto histórico: pretext tasks y la búsqueda de una tarea simple pero semántica

Hacia 2015–2018, las ConvNets habían transformado la visión por computador gracias a su capacidad de aprender características semánticas de alto nivel. Pero ese éxito dependía de cantidades masivas de datos etiquetados manualmente —costoso e imposible de escalar al volumen de imágenes disponibles hoy—. De ahí el creciente interés en el **aprendizaje no supervisado de representaciones**, y dentro de él, un paradigma que el paper nombra explícitamente: el **aprendizaje autosupervisado** (*self-supervised learning*), que define una *pretext task* sin anotación, usando solo la información visual presente en las imágenes o videos, para proveer una señal de supervisión sustituta (*surrogate*) que sirva al aprendizaje de características.

El paper enumera con cuidado el ecosistema de pretextos previos, que constituye el estado del arte contra el que compite:

- **Colorización** (Zhang et al., 2016a; Larsson et al., 2016): entrenar la red para colorear imágenes en escala de grises.
- **Predicción de posición relativa de parches** (Doersch et al., 2015 — *context prediction*; Noroozi & Favaro, 2016 — *jigsaw puzzles*): predecir cómo se ordenan o ubican fragmentos de la imagen.
- **Egomotion** (Agrawal et al., 2015): predecir el auto-movimiento de un vehículo entre dos cuadros consecutivos.
- **Context Encoders / inpainting** (Pathak et al., 2016b): reconstruir regiones faltantes.
- **Conteo** (Noroozi et al., 2017), **Split-Brain autoencoders** (Zhang et al., 2016b), métodos generativos (DCGAN — Radford et al., 2015; BiGAN — Donahue et al., 2016) y de *clustering* (Dosovitskiy et al., 2014 — ExemplarCNN).

El "rationale" común a todos estos pretextos es el mismo: resolverlos debería *forzar* a la ConvNet a aprender características semánticas transferibles a otras tareas de visión. Sin embargo, hasta ese momento ninguna representación autosupervisada lograba igualar a la supervisada, y muchos métodos eran computacionalmente pesados (los basados en reconstrucción), lentos de converger, o requerían *preprocesamiento especial* para evitar que la red explotara artefactos triviales (atajos de bajo nivel como cromática, bordes de compresión o discontinuidades que delatan la respuesta sin entender la escena).

El problema de diseño, entonces, no era inventar *cualquier* pretexto: era encontrar uno que fuera **simple de implementar y barato de computar, pero que a la vez forzara semántica genuina** y *no* dejara atajos explotables. Ese es exactamente el hueco que RotNet llena.

El paper se distingue cuidadosamente de dos trabajos previos que también usan transformaciones geométricas. ExemplarCNN (Dosovitskiy et al., 2014) entrena una red para que sus representaciones sean *invariantes* a transformaciones geométricas y cromáticas; RotNet hace lo opuesto, entrena para *reconocer* la transformación aplicada. Y el método de egomotion (Agrawal et al., 2015) usa una arquitectura siamesa que recibe *dos* cuadros consecutivos y predice la transformación de cámara entre ellos por regresión; RotNet recibe una *sola* imagen, sin acceso a la original, y clasifica la transformación que se le aplicó.

## 3. Contribución central

La contribución es un **nuevo pretexto autosupervisado** —predecir la rotación de la imagen— y una evaluación exhaustiva que demuestra que, pese a (o gracias a) su simplicidad, ofrece una señal de supervisión poderosa. En palabras de los autores, sus aportes son cuatro:

1. Proponen una tarea autosupervisada **muy simple** que a la vez entrega una **señal de supervisión potente** para el aprendizaje semántico.
2. La evalúan exhaustivamente bajo múltiples escenarios (semi-supervisado, transfer learning) y tareas (clasificación en CIFAR-10, ImageNet, Places; clasificación, detección y segmentación en PASCAL).
3. En *todos* esos benchmarks obtienen resultados estado del arte con mejoras dramáticas respecto de los enfoques no supervisados previos.
4. Como consecuencia, **estrechan significativamente la brecha** entre el aprendizaje de características supervisado y no supervisado para varias tareas importantes.

El corazón conceptual de la contribución no es la fórmula sino la *elección del conjunto de transformaciones*. El paper formaliza un marco general donde se define un conjunto de $K$ transformaciones geométricas discretas $G = \{g(\cdot|y)\}_{y=1}^{K}$, y la ConvNet $F(\cdot)$ recibe una imagen transformada $X^{y^*}$ y produce una distribución de probabilidad sobre las $K$ transformaciones posibles. **Es la elección de $G$ la que define la dificultad y el valor semántico del pretexto.** La propuesta concreta: definir $G$ como las rotaciones por múltiplos de 90 grados (0, 90, 180, 270), de modo que $K = 4$ y $g(X|y) = \text{Rot}(X, (y-1)\cdot 90)$.

¿Por qué rotaciones de 90 grados específicamente, y no otra transformación? El paper da tres razones de diseño que conviene retener, porque ahí está la sutileza:

- **Forzar semántica:** es esencialmente imposible reconocer la rotación sin haber aprendido a reconocer clases de objetos y sus partes semánticas. La tarea obliga a localizar objetos salientes, reconocer su orientación y tipo, y relacionar esa orientación con la orientación dominante en que cada tipo de objeto suele aparecer.
- **Ausencia de artefactos de bajo nivel:** las rotaciones por múltiplos de 90 grados se implementan con operaciones de *flip* y *transpose*, que **no dejan artefactos visuales fácilmente detectables**. Esto es crítico: si se usaran, por ejemplo, transformaciones de escala o de aspecto, harían falta rutinas de redimensionado que dejan rastros de bajo nivel que la red aprendería a explotar de forma trivial, sin valor semántico. RotNet *no requiere preprocesamiento especial* para evitar atajos, a diferencia de muchos competidores.
- **Buena definición (well-posedness):** las imágenes capturadas por humanos suelen mostrar los objetos en posición erguida, lo que hace que la tarea de reconocer la rotación esté bien definida —sin ambigüedad, salvo objetos perfectamente redondos—. En contraste, la escala de un objeto varía mucho entre fotos, lo que haría mal definida una tarea basada en escala.

## 4. Método

El objetivo de entrenamiento es directo. Dado un conjunto de $N$ imágenes $D = \{X_i\}_{i=1}^{N}$, se minimiza sobre los parámetros $\theta$ de la red:

$$\min_{\theta} \; \frac{1}{N} \sum_{i=1}^{N} \text{loss}(X_i, \theta)$$

donde la pérdida por imagen promedia la *cross-entropy* sobre las cuatro rotaciones:

$$\text{loss}(X_i, \theta) = -\frac{1}{K} \sum_{y=1}^{K} \log\!\big(F^{y}(g(X_i|y)\,|\,\theta)\big)$$

Es decir, $F^{y}(\cdot)$ es la probabilidad predicha para la rotación $y$, y la red debe maximizar la probabilidad de la rotación correcta para cada una de las cuatro versiones rotadas de cada imagen.

**Implementación de las rotaciones.** El caso de 0 grados es la imagen misma. Las otras tres se obtienen con operaciones exactas, sin interpolación: para 90 grados se transpone y luego se hace flip vertical; para 180 grados, flip vertical seguido de flip horizontal; para 270 grados, flip vertical seguido de transpose. Esta exactitud (sin redimensionar ni interpolar) es la que garantiza la ausencia de artefactos.

**Truco de entrenamiento clave.** En experimentos preliminares, los autores encontraron una mejora significativa al alimentar a la red con **las cuatro copias rotadas de una imagen simultáneamente** en el mismo mini-batch, en vez de muestrear aleatoriamente una sola rotación por imagen. Esto significa que en cada batch la red ve 4 veces más imágenes que el tamaño nominal del batch. Este detalle —presentar las cuatro vistas juntas— estabiliza y mejora el aprendizaje.

**Ventajas de la formulación (sección Discussion).** El pretexto tiene el *mismo costo computacional* que el aprendizaje supervisado y una velocidad de convergencia similar —significativamente más rápida que los enfoques basados en reconstrucción de imágenes—. Su AlexNet entrena en alrededor de 2 días en una sola GPU Titan X. Además puede adoptar trivialmente los esquemas de paralelización eficientes diseñados para aprendizaje supervisado (Goyal et al., 2017), lo que lo convierte en candidato ideal para aprendizaje no supervisado a escala de internet (miles de millones de imágenes). Y, de nuevo, no necesita rutinas especiales de preprocesamiento para evitar features triviales.

**Evidencia cualitativa: mapas de atención y filtros.** El paper aporta dos visualizaciones que sustentan el argumento semántico:

- *Mapas de atención* (Figura 3): calculados a partir de la magnitud de las activaciones en cada celda espacial de una capa convolucional, reflejan dónde pone foco la red. Los mapas del modelo entrenado en rotación se enfocan en partes de objetos de alto nivel (ojos, narices, colas, cabezas) y, comparados con los de un modelo supervisado de reconocimiento de objetos, *enfocan aproximadamente las mismas regiones*. En el apéndice se muestra además que estos mapas son **equivariantes** respecto de la rotación: las cuatro copias rotadas de una imagen producen mapas de atención esencialmente iguales, lo que significa que la red se enfoca en las mismas partes del objeto sin importar la rotación.
- *Filtros de primera capa* (Figura 4): los filtros aprendidos por RotNet (AlexNet) son mayormente filtros de bordes orientados, en múltiples orientaciones y frecuencias. Notablemente, **muestran incluso más variedad que los filtros aprendidos por la tarea supervisada de reconocimiento de objetos**. Esta comparación lado a lado —filtros supervisados vs filtros por rotación— es precisamente la figura que la Clase 28 retoma en su sección de rotaciones.

## 5. Experimentos y resultados (números reales)

Los autores evalúan sobre CIFAR-10, ImageNet, Places205 y PASCAL VOC, en escenarios de transfer learning y semi-supervisado. El protocolo estándar de evaluación congela los features autosupervisados y entrena clasificadores (lineales o no lineales) encima, midiendo qué tan buenas son las representaciones.

**CIFAR-10 (arquitectura Network-In-Network, NIN).** Llaman *RotNet* al modelo entrenado en el pretexto de rotación. Hallazgos:

- *Calidad por profundidad de capa (Tabla 1):* los features del **2º bloque convolucional** logran la mayor accuracy de reconocimiento de objetos (entre 88.26% y 89.06%). Más allá del 2º bloque la accuracy se degrada gradualmente, porque las capas tardías se vuelven cada vez más específicas del pretexto de rotación. Aumentar la profundidad total del RotNet mejora los features de las capas tempranas (su "cabeza" más compleja libera a las capas previas de especializarse en rotación).
- *Número de rotaciones (Tabla 2):* la elección de 4 rotaciones es óptima. Con 4 rotaciones (0/90/180/270) se obtiene **89.06%**; con 8 rotaciones (múltiplos de 45) baja a 88.51% (las transformaciones de 45 grados no son lo bastante distinguibles e introducen artefactos); con 2 rotaciones (0/180) baja a 87.46% (muy pocas clases, menos señal); y con 2 rotaciones (90/270) baja aún más a 85.52% (este modelo nunca "ve" la rotación de 0 grados que sí se usa luego en el entrenamiento de reconocimiento).
- *Comparación con supervisado y no supervisado (Tabla 3):* RotNet + conv alcanza **91.16%**, frente a **92.80%** del NIN totalmente supervisado —una brecha de apenas **1.64 puntos**—. Con fine-tuning sube a 92.17%, casi igualando al supervisado. Supera a Roto-Scat+SVM (82.3%), ExemplarCNN (84.3%), DCGAN (82.8%) y Scattering (84.7%). Como referencia, una inicialización aleatoria congelada da solo 72.50%.
- *Correlación pretexto–tarea (Figura 5a):* a medida que mejora la accuracy en la predicción de rotación, mejora también la accuracy de reconocimiento de objetos sobre esos features —y converge rápido—. Esto confirma que resolver mejor el pretexto produce mejores representaciones semánticas.
- *Semi-supervisado (Figura 5b):* el modelo RotNet **supera al supervisado** cuando hay menos de 1000 ejemplos etiquetados por categoría, y la ventaja crece a medida que decrecen las etiquetas disponibles. Esto demuestra su utilidad práctica cuando los datos anotados escasean.

**ImageNet (arquitectura AlexNet, preentrenado sin etiquetas sobre ImageNet).** Siguiendo protocolos estándar:

- *Clasificadores no lineales (Tabla 4, top-1):* RotNet logra **50.0%** en Conv4 y **43.8%** en Conv5, superando a todos los competidores por margen significativo (más de 4 puntos en Conv4 y ~8 puntos en Conv5 sobre el mejor previo). Como referencia, las etiquetas de ImageNet dan 59.7%.
- *Clasificadores lineales (Tabla 5, regresión logística):* RotNet lidera en Conv3 (38.7%), Conv4 (38.2%) y Conv5 (36.5%), superando a Context, Colorization, Jigsaw, BiGAN, Split-Brain y Counting.

**Places205 (Tabla 6).** Evaluando generalización a clases no vistas durante el preentrenamiento (features de ImageNet, clasificación de 205 escenas con regresión logística), RotNet iguala o supera a los métodos no supervisados previos en la mayoría de las capas.

**PASCAL VOC — transfer learning (Tabla 7).** El resultado más citado del paper:

- *Detección VOC 2007:* **54.4% mAP**, solo **2.4 puntos** bajo el supervisado (56.8%), y superando a Jigsaw (53.2%), Context (51.1%), Counting (51.4%) y Colorization (46.9%).
- *Clasificación VOC 2007:* 70.87% (fc6-8) y 72.97% (fine-tuning completo).
- *Segmentación VOC 2012:* 39.1% mIoU, el mejor entre los no supervisados comparados.

En conjunto, RotNet estado del arte en *todos* los benchmarks evaluados, estrechando consistentemente la brecha con el supervisado.

## 6. Limitaciones

- **Sesgo de orientación canónica.** El argumento de "well-posedness" depende de que las imágenes muestren objetos en posición erguida. La propia tarea pierde sentido en objetos rotacionalmente simétricos o redondos (donde la rotación es ambigua) y en dominios donde no existe una orientación canónica —imágenes aéreas, satelitales, microscopía, escaneos médicos sin "arriba" definido—. Allí el pretexto de rotación puede ser débil o engañoso.
- **No alcanza al supervisado.** Aunque la brecha se estrecha dramáticamente (2.4 puntos en detección VOC), sigue habiendo brecha; RotNet no *supera* al preentrenamiento supervisado en el régimen de datos abundantes.
- **Especialización de capas tardías.** Como muestra la Tabla 1, las capas profundas se especializan en el pretexto y *degradan* su utilidad para tareas downstream; el mejor feature no es el más profundo, lo que obliga a elegir la capa de extracción con cuidado.
- **Arquitecturas y época.** Los experimentos usan AlexNet y NIN, arquitecturas de su tiempo; la comparación con métodos previos es "solo indicativa" porque cada uno usa una arquitectura distinta. El paper precede a ResNets profundas y al contrastive learning, que luego cambiarían el panorama del SSL.
- **Riesgo de atajos residual.** Aunque las rotaciones de 90 grados evitan artefactos de redimensionado, no eliminan todo sesgo posicional: una imagen con cielo arriba y suelo abajo puede a veces resolverse por estadística de color/posición global más que por entender el objeto.

## 7. Impacto: la simplicidad como virtud en el aprendizaje autosupervisado

RotNet se convirtió en uno de los pretextos de referencia del SSL en visión, y su legado principal es metodológico: demostró que **un pretexto trivial de implementar puede competir con los más elaborados, siempre que esté bien diseñado para forzar semántica y bloquear atajos de bajo nivel**. Esa lección —la simplicidad como virtud, no como concesión— anticipa la filosofía de los métodos contrastivos posteriores (SimCLR, MoCo) y de los enfoques predictivos como BYOL: pretextos conceptualmente limpios, baratos de computar, escalables a datos masivos, sin preprocesamiento defensivo.

Tres ideas de RotNet quedaron en el canon del campo: (1) la noción de que *la elección del pretexto define qué se aprende*, y que un buen pretexto es uno que no se puede resolver sin entender la escena; (2) el protocolo de evaluación por *linear/non-linear probing* sobre features congelados, capa por capa, que se volvió estándar para medir representaciones autosupervisadas; y (3) la observación de que las capas tardías se sobre-especializan en el pretexto, motivando el diseño de proyectores y cabezas desechables en métodos posteriores.

## 8. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

La Clase 28 dedica una sección a **"Rotaciones para autosupervisión"** que es, esencialmente, una exposición de este paper. Dos elementos de la clase provienen directamente de aquí:

- La **comparación de filtros aprendidos** (supervisado vs predicción de rotación): la Figura 4 del paper, donde se observa que los filtros de RotNet son bordes orientados en múltiples frecuencias con *más variedad* que los del modelo supervisado, ilustrando que el pretexto induce representaciones de bajo nivel ricas y genéricas.
- La **accuracy en PASCAL VOC 2007**: el número emblemático de 54.4% mAP en detección, a 2.4 puntos del supervisado, que la clase usa para argumentar que un pretexto simple puede acercarse mucho al techo supervisado.

Para profundizar en los conceptos transversales, ver el fundamento [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) y el hub de la [Clase 28](/clases/clase-28).
