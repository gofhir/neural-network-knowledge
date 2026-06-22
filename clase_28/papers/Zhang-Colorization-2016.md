# Colorful Image Colorization — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Colorful Image Colorization*.
- **Autores:** Richard Zhang, Phillip Isola, Alexei A. Efros — todos de la University of California, Berkeley (Berkeley Vision Lab / EECS).
- **Venue:** European Conference on Computer Vision (ECCV) 2016. La versión de referencia es la *camera ready* de ECCV (modelo "v2"); el manuscrito incluye un Apéndice extenso con análisis adicionales generados con un modelo previo ("v1").
- **Preprint:** arXiv:1603.08511, versión v5 del 5 de octubre de 2016, [arxiv.org/abs/1603.08511](https://arxiv.org/abs/1603.08511).
- **Sitio / código:** [richzhang.github.io/colorization](http://richzhang.github.io/colorization/) — modelo, código (Caffe) y demo públicos.

El problema de partida es de gráficos por computador: dada una fotografía en escala de grises, **alucinar una versión a color plausible**. Es un problema severamente subdeterminado (se han perdido dos de las tres dimensiones del color), por lo que los enfoques previos o bien dependían de interacción manual del usuario, o bien producían colorizaciones desaturadas y apagadas. El paper propone un método **totalmente automático** que genera colores vibrantes y realistas, implementado como un único pase *feed-forward* de una CNN en test, entrenada sobre más de un millón de imágenes de color.

La tesis central es abrazar la incertidumbre del problema en lugar de evitarla. Como muchos objetos admiten varias colorizaciones plausibles (una manzana puede ser roja, verde o amarilla, pero no azul), el color es **inherentemente multimodal**. El paper reformula la predicción de color como una **tarea de clasificación** sobre bins de color cuantizados —no como una regresión a un valor continuo— y aplica *class rebalancing* en entrenamiento para amplificar la diversidad de colores raros. La evaluación recurre a un **"Turing test de colorización"** (real vs. fake en Amazon Mechanical Turk): el método engaña a observadores humanos el **32%** de las veces (el límite teórico de la verdad terreno es 50%), muy por encima del trabajo previo.

La segunda contribución, la que hace que el paper pertenezca a la Clase 28 de aprendizaje autosupervisado, es que **la colorización funciona como pretext task** para aprender representaciones. La red, entrenada solo a colorear (sin etiquetas semánticas), aprende características transferibles que alcanzan rendimiento estado del arte en varios benchmarks de feature learning. El paper acuña el término **cross-channel encoder** para describir este mecanismo.

## 2. Contexto histórico: colorización automática y la emergencia del SSL

La colorización es un problema de gráficos con larga tradición. El paper distingue dos linajes. Los métodos **no paramétricos** parten de una o más imágenes de color de referencia (provistas por el usuario o recuperadas automáticamente) y transfieren color desde regiones análogas, en el marco de *Image Analogies* (Hertzmann et al., 2001) y trabajos posteriores. Los métodos **paramétricos** aprenden funciones de predicción a partir de grandes datasets de imágenes de color en entrenamiento, planteándolo como regresión sobre el espacio de color continuo (Cheng et al., 2015; Dahl, 2016) o como clasificación de valores de color cuantizados (Charpiat et al., 2008). El método de Zhang et al. pertenece a esta segunda corriente: clasifica colores, pero con un modelo mayor, más datos, e innovaciones en la pérdida y en el mapeo a la salida continua.

Un punto importante es el **trabajo concurrente**. Larsson et al. (2016) e Iizuka et al. (2016) desarrollaron simultáneamente sistemas similares basados en CNN y datos a gran escala. Las diferencias son instructivas: Zhang usa una pérdida de clasificación con clases raras rebalanceadas; Larsson usa clasificación sin rebalancear; Iizuka usa una pérdida de regresión. Las arquitecturas también difieren —Larsson usa hipercolumnas sobre VGG, Iizuka una arquitectura de dos flujos que fusiona características globales y locales, Zhang una red de un flujo estilo VGG con profundidad añadida y **convoluciones dilatadas**—. Esta triple coincidencia marca un momento en que la colorización con deep learning estaba "en el aire".

El segundo hilo de contexto es el **aprendizaje autosupervisado** (SSL). La idea de aprender representaciones usando los datos crudos como su propia fuente de supervisión se remonta al menos a los autoencoders (Bengio et al., 2013). Hacia 2015–2016 había florecido toda una familia de *pretext tasks* basadas en imputación de datos: predecir la posición relativa de parches (Doersch et al., 2015), aprender de videos y ego-movimiento (Wang & Gupta, 2015; Agrawal et al., 2015), reconstruir regiones ocultas con context encoders / inpainting (Pathak et al., 2016), aprender por feature learning adversarial (Donahue et al., 2016) y resolver puzzles de jigsaw (Noroozi & Favaro, 2016). La colorización encaja en esta línea como una forma particular de imputación: predecir un subconjunto de los canales (el color) a partir de otro (la luminancia). La gran ventaja práctica que comparten todas estas tareas es que **los datos de entrenamiento son prácticamente gratis**: cualquier foto a color sirve como ejemplo, tomando su canal L como entrada y sus canales ab como señal supervisora.

## 3. Contribución central

Las contribuciones se reparten en dos áreas. Primero, **progreso en el problema de gráficos** de la colorización automática: (a) diseñar una función objetivo apropiada que maneja la incertidumbre multimodal del problema y captura una amplia diversidad de colores; (b) introducir un marco novedoso para evaluar algoritmos de colorización (el Turing test), potencialmente aplicable a otras tareas de síntesis de imágenes; y (c) fijar un nuevo récord en la tarea entrenando sobre un millón de fotos a color. Segundo, **introducir la colorización como método competitivo y directo para SSL**, alcanzando resultados estado del arte en varios benchmarks de aprendizaje de representaciones.

El insight técnico que une todo es tratar el color como una **distribución de probabilidad multimodal por píxel** en vez de un único valor. Esto evita el colapso al promedio que sufren las pérdidas de regresión, y es exactamente lo que permite tanto las colorizaciones vibrantes como la riqueza de la representación aprendida.

## 4. Método

### 4.1. Formulación: del espacio Lab y por qué clasificar

Dado el canal de luminancia $X \in \mathbb{R}^{H \times W \times 1}$, el sistema aprende un mapeo $\hat{Y} = \mathcal{F}(X)$ a los dos canales de color asociados $Y \in \mathbb{R}^{H \times W \times 2}$. La tarea se realiza en el espacio de color **CIE Lab**, elegido porque las distancias en ese espacio modelan la distancia perceptual. El canal L (lightness) es la entrada; los canales **a** y **b** (cromaticidad) son la salida a predecir.

El enfoque ingenuo, usado por trabajos previos, es minimizar la **pérdida Euclidiana L2** entre el color predicho y el real:

$$L_2(\hat{Y}, Y) = \frac{1}{2}\sum_{h,w} \lVert Y_{h,w} - \hat{Y}_{h,w}\rVert_2^2$$

El problema fatal de esta pérdida es que no es robusta a la ambigüedad multimodal. Si un objeto admite un conjunto de valores ab distintos, la solución óptima a la pérdida Euclidiana es **el promedio del conjunto**, y promediar colores plausibles produce resultados grisáceos y desaturados. Peor aún: si el conjunto de colorizaciones plausibles es no convexo, la solución óptima cae fuera del conjunto, dando resultados implausibles. Esta es la raíz del aspecto sepia y apagado de los métodos previos.

En su lugar, el paper trata el problema como **clasificación multinomial**. Cuantiza el espacio de salida ab en bins con un grid de tamaño 10 y conserva los $Q = 313$ valores que están *in-gamut* (es decir, que corresponden a colores realizables). La red aprende un mapeo $\hat{Z} = \mathcal{G}(X)$ a una distribución de probabilidad sobre colores posibles $\hat{Z} \in [0,1]^{H \times W \times Q}$ por píxel. La verdad terreno $Y$ se convierte a un vector $Z$ mediante un esquema de **soft-encoding**: en vez de codificar 1-hot al bin más cercano, se buscan los 5 vecinos más cercanos en el espacio de salida y se ponderan proporcionalmente a su distancia con un kernel Gaussiano ($\sigma = 5$). El soft-encoding aceleró el aprendizaje de las relaciones entre elementos del espacio de salida. La pérdida es la entropía cruzada multinomial:

$$L_{cl}(\hat{Z}, Z) = -\sum_{h,w} v(Z_{h,w}) \sum_{q} Z_{h,w,q} \log(\hat{Z}_{h,w,q})$$

donde $v(\cdot)$ es el término de reponderación que introduce el rebalanceo de clases.

### 4.2. Class rebalancing: rescatar los colores raros

La distribución de valores ab en imágenes naturales está fuertemente sesgada hacia valores **bajos (desaturados)**, por la abundancia de fondos como nubes, pavimento, tierra y paredes. Sobre 1.3M de imágenes de ImageNet, el número de píxeles en valores desaturados es órdenes de magnitud mayor que en valores saturados. Sin corregir esto, la pérdida queda dominada por los píxeles desaturados y la red aprende a "jugar a lo seguro" prediciendo grises.

La solución es **reponderar la pérdida de cada píxel según la rareza de su color**, lo que es asintóticamente equivalente a remuestrear el espacio de entrenamiento. Cada píxel se pondera por un factor $w \in \mathbb{R}^Q$ basado en su bin ab más cercano:

$$v(Z_{h,w}) = w_{q^*}, \quad q^* = \arg\max_q Z_{h,w,q}$$

$$w \propto \left((1-\lambda)\tilde{p} + \frac{\lambda}{Q}\right)^{-1}, \quad \mathbb{E}[w] = \sum_q \tilde{p}_q w_q = 1$$

Operativamente: se estima la distribución empírica de colores $p$ en el espacio ab cuantizado sobre todo ImageNet, se suaviza con un kernel Gaussiano ($\sigma = 5$), se mezcla con una distribución uniforme con peso $\lambda = \frac{1}{2}$, se toma el recíproco y se normaliza para que el factor sea 1 en esperanza. El efecto es subir el peso de los colores saturados raros, empujando a la red a explotar toda la diversidad del dataset.

### 4.3. Annealed-mean: de la distribución a un color puntual

Una vez que la red predice una distribución $\hat{Z}$, hace falta una función $\mathcal{H}$ que la mapee a un color puntual $\hat{Y}$. Hay dos extremos malos. Tomar el **modo** de la distribución por píxel produce resultados vibrantes pero espacialmente inconsistentes (parches de color que saltan, como manchas rojas sobre un bus). Tomar la **media** produce resultados espacialmente coherentes pero desaturados, con un tono sepia poco natural —no es sorprendente, porque promediar tras clasificar reintroduce el mismo defecto que la regresión L2—.

El paper interpola entre ambos reajustando la **temperatura $T$** de la distribución softmax y luego tomando la media del resultado, en una operación inspirada en el *simulated annealing* que llaman **annealed-mean**:

$$\mathcal{H}(Z_{h,w}) = \mathbb{E}[f_T(Z_{h,w})], \quad f_T(z) = \frac{\exp(\log(z)/T)}{\sum_q \exp(\log(z_q)/T)}$$

Con $T = 1$ la distribución queda intacta (media simple); bajar $T$ la hace más picuda; $T \to 0$ converge al modo (1-hot). El valor elegido, $T = 0.38$, captura la vibración del modo manteniendo la coherencia espacial de la media. Detalle importante de implementación: el sistema final $\mathcal{F}$ es la composición de la CNN $\mathcal{G}$ y la operación $\mathcal{H}$, que opera por píxel de forma independiente con un único parámetro, por lo que se implementa como parte del pase feed-forward (aunque el sistema no es estrictamente *end-to-end* entrenable a través de $\mathcal{H}$).

### 4.4. Arquitectura

La red es de un solo flujo, estilo VGG, con profundidad añadida. Cada bloque conv agrupa 2 o 3 capas conv+ReLU repetidas, seguidas de **BatchNorm**. No hay capas de pooling: todos los cambios de resolución se logran con downsampling/upsampling espacial entre bloques. La característica clave es el uso de **convoluciones dilatadas** (atrous), cuya dilatación efectiva crece de conv1 a conv5 y luego decrece de conv6 a conv8, ampliando el campo receptivo sin perder resolución. El entrenamiento usó el solver ADAM por unas 450k iteraciones, con inicialización k-means data-dependiente (Krähenbühl et al., 2016).

## 5. Experimentos

### 5.1. Calidad de colorización

Entrenado sobre 1.3M de imágenes de ImageNet y evaluado sobre 10k de validación, el método se compara contra variantes propias (L2, clasificación sin rebalancear) y contra trabajo previo/concurrente, sobre tres métricas complementarias:

- **Realismo perceptual (AMT):** el Turing test real vs. fake. La variante completa (clasificación + rebalanceo) engaña a los participantes el **32.3%** de las veces, frente a 21.2% de la variante L2, 25.2% sin rebalanceo, 18.3% de Dahl (2016) y 27.2% de Larsson et al. La diferencia respecto a todos los métodos comparados es significativa ($p < 0.05$) salvo frente a Larsson ($p = 0.10$). Como control de competencia, en el 10% de los trials se enfrentó la verdad terreno contra la *baseline* Random: los participantes la detectaron como falsa el 87% del tiempo, confirmando que entendían la tarea. Notablemente, en algunos casos los participantes fueron engañados *más* del 50% de las veces, prefiriendo la colorización del modelo sobre la verdad terreno —a menudo porque la foto original tenía mal balance de blancos y el modelo produce una apariencia más prototípica—.
- **Interpretabilidad semántica (clasificación VGG):** se alimenta la imagen recolorizada a una red VGG entrenada en color real para clasificar ImageNet. La precisión cae de 68.3% (color real) a 52.7% (gris), y la recolorización del método completo la recupera a 56.0%. Esto demuestra un uso práctico inmediato: mejorar la clasificación de imágenes en gris simplemente colorizándolas antes de pasarlas a un clasificador estándar, sin reentrenar nada.
- **Exactitud cruda (AuC):** porcentaje de píxeles predichos dentro de una distancia L2 umbralizada de la verdad terreno en ab, barriendo umbrales para integrar el área bajo la curva. Esta métrica está dominada por píxeles desaturados, así que incluso predecir gris puntúa alto; la variante completa con rebalanceo iguala aproximadamente a "predecir gris". Pero bajo la **variante balanceada por clase** del AuC (que repondera inversamente a la probabilidad del color), el método completo supera a todas las variantes y algoritmos comparados, confirmando que el rebalanceo logra su efecto deseado en las regiones perceptualmente interesantes.

El paper es honesto sobre los **modos de falla**: fallos de consistencia de largo alcance, confusiones frecuentes entre rojo y azul, y el tono sepia por defecto en escenas interiores complejas.

### 5.2. Colorización como pretext task de SSL

Aquí la colorización se evalúa como **cross-channel encoder**: como un autoencoder, salvo que la entrada y la salida son canales distintos de la imagen. Para comparar justamente contra otros métodos de SSL, se reentrena una **AlexNet** en la tarea de colorización (método completo, 450k iteraciones) y se mide la calidad de sus características de dos formas.

**Generalización de tarea (ImageNet, clasificadores lineales):** se congelan los pesos pre-entrenados solo en colorización (sin etiquetas semánticas) y se entrenan clasificadores lineales sobre cada capa convolucional. La representación conv1 rinde peor que métodos competidores (por el *handicap* de entrada en gris, que cuesta ~6% constante a lo largo de la red), pero esa brecha se cierra de inmediato en conv2, y desde ahí el método es competitivo con Doersch et al. (2015) y Donahue et al. (2016). Esto indica que resolver colorización fomenta representaciones que **separan linealmente las clases semánticas**.

**Generalización de dataset y tarea (PASCAL):** este es el resultado que la Clase 28 destaca. Se hace fine-tuning de la red en PASCAL para clasificación (VOC 2007), detección (VOC 2007, Fast R-CNN) y segmentación (VOC 2012, FCN), en dos modos: entrada en gris (`Ours (gray)`) y entrada Lab de 3 canales con los pesos ab inicializados en cero (`Ours (color)`).

| Tarea (PASCAL) | Métrica | Ours (gray) | Ours (color) | Mejor previo comparado |
|---|---|---|---|---|
| Clasificación (all layers) | %mAP | **65.9** | 65.6 | Doersch et al. 65.3 |
| Detección (all) | %mAP | 46.1 | 46.9 | Doersch et al. 51.1 |
| Segmentación (all) | %mIU | 35.0 | 35.6 | Donahue et al. 34.9 |

El número **65.9** de clasificación en PASCAL VOC es exactamente el valor que aparece en la tabla de la clase. El método alcanza estado del arte en clasificación y segmentación entre los métodos de SSL probados. En detección queda por debajo de Doersch et al. (51.1) pero por encima de la fuerte baseline k-means (45.6). Como referencia, el pre-entrenamiento supervisado con ImageNet (techo) alcanza 78.9 / 56.8 / 48.0 en las tres tareas, así que todos los métodos de SSL todavía quedan cortos frente a la supervisión semántica plena —pero la colorización los lidera o iguala sin usar una sola etiqueta—.

### 5.3. Análisis adicionales (Apéndice)

El Apéndice añade evidencia importante. Se confirma que la red aprende **distribuciones multimodales genuinas**: para una misma imagen predice colores distintos para el fondo (verde/marrón) y el pájaro de primer plano (rojo/azul). Se descarta que la red explote solo claves de bajo nivel: dada una carta de color Macbeth en gris (donde la luminancia varía mucho) la red no recupera los colores, pero sí distingue dos vegetales casi isoluminantes —prueba de que usa textura y semántica, no solo el valor de lightness—. También se muestra robustez a fotos **legacy en blanco y negro** reales (Ansel Adams, Dorothea Lange, el tilacino extinto de 1936), pese a que sus estadísticas de bajo nivel difieren de las fotos modernas de entrenamiento. Y se demuestra generalización al dataset SUN (LEARCH de Deshpande et al.), engañando al 17.2% vs. 9.8% del estado del arte previo.

## 6. Limitaciones

El propio paper reconoce sus límites. Las colorizaciones fallan en consistencia de largo alcance, confunden recurrentemente rojo y azul, y caen en sepia por defecto en interiores complejos. La métrica AuC cruda está dominada por píxeles desaturados, lo que la hace poco discriminativa salvo en su variante balanceada. El sistema no es estrictamente *end-to-end* a través del annealed-mean. En el frente de SSL, la entrada en escala de grises impone un *handicap* estructural (~6% constante) frente a métodos que ven los tres canales RGB, y la conv1 aprendida es comparativamente pobre. En detección PASCAL queda claramente por debajo de la predicción de contexto de Doersch et al. Finalmente, en la comparación con Larsson et al. la ventaja en el Turing test no es estadísticamente significativa, de modo que el rebalanceo —aunque demostrablemente útil frente a las variantes propias— no establece superioridad concluyente sobre todo el trabajo concurrente.

## 7. Impacto

*Colorful Image Colorization* se convirtió en uno de los trabajos canónicos del aprendizaje autosupervisado en visión. Su valor doble —ser un sistema de colorización de referencia *y* un pretext task competitivo— lo instaló en todas las tablas comparativas de SSL que vinieron después, y la fila "colorización" con su 65.9 en PASCAL VOC quedó como punto de referencia obligado. El concepto de **cross-channel encoder** y la idea más amplia de "auto-predicción de una parte de los datos a partir de otra" anticiparon directamente la era del *contrastive* y *masked* representation learning (SimCLR, MoCo, MAE) que dominaría la visión a partir de 2019–2020. La reformulación regresión → clasificación + rebalanceo de clases raras, y el truco del annealed-mean por temperatura, son lecciones de diseño de pérdidas que trascienden la colorización: aparecen siempre que una tarea es intrínsecamente multimodal y una L2 ingenua colapsaría al promedio. El método se extendió luego a colorización interactiva guiada por el usuario (Zhang et al., 2017) y sirvió de base a innumerables trabajos de restauración de fotos históricas.

## 8. Conexión con la Clase 28

La Clase 28 (Aprendizaje Autosupervisado) presenta este trabajo como el ejemplo paradigmático de **auto-predicción en imágenes**: a partir de imágenes en escala de grises, generar color. La intuición de la clase —"de grises generar color" como tarea que no requiere etiquetas humanas— es exactamente el *cross-channel encoder* de Zhang et al.: el canal L se usa como entrada y los canales ab como su propia señal de supervisión, gratis, sobre cualquier foto a color. La clase exhibe la **tabla de transferencia a PASCAL VOC** con el valor de **colorización 65.9** en clasificación, que es la fila `Ours (gray)` de la Tabla 2 de este paper, situando la colorización junto a otras pretext tasks (context prediction, inpainting, jigsaw) en el panteón de métodos SSL pre-contrastivos. El paper materializa los tres pilares conceptuales que la clase enseña sobre SSL: (1) la señal de supervisión emerge de la estructura de los propios datos; (2) resolver una tarea "auxiliar" plausible obliga a la red a aprender semántica de alto nivel; y (3) esa representación transfiere a tareas posteriores reales.

Ver también: [/fundamentos/aprendizaje-autosupervisado](/fundamentos/aprendizaje-autosupervisado) · [/clases/clase-28](/clases/clase-28).
