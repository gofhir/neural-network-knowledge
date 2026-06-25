# Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*.
- **Autores:** Alec Radford (indico Research, Boston), Luke Metz (indico Research), Soumith Chintala (Facebook AI Research, Nueva York).
- **Venue:** ICLR 2016 (aceptado como conference paper; el PDF circula con el encabezado "Under review as a conference paper at ICLR 2016").
- **Año:** 2015 (preprint) / 2016 (publicación). **Preprint:** arXiv:1511.06434v2 (7 ene 2016), [arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434).
- **Linaje:** descendiente directo de la GAN original (Goodfellow et al., 2014); apoyado en la *all convolutional net* (Springenberg et al., 2014) y en Batch Normalization (Ioffe & Szegedy, 2015).

Este no es un paper que introduzca un *objetivo* nuevo —el juego adversarial minimax ya existía desde Goodfellow et al. (2014)—; es un paper de **ingeniería arquitectónica**. Su tesis es que las GAN, hasta ese momento, "habían sido conocidas por ser inestables de entrenar, produciendo a menudo generadores que generan salidas sin sentido", y que los intentos históricos de escalarlas con CNN "habían fracasado". El aporte central es identificar, tras "exploración extensiva del modelo", **una familia de arquitecturas convolucionales que resultan en entrenamiento estable** a través de distintos datasets y que permite entrenar generadores más profundos y de mayor resolución. A esa clase la bautizan DCGAN (Deep Convolutional GAN).

El paper persigue dos objetivos entrelazados. El primero, **estabilizar** el entrenamiento adversarial con convoluciones. El segundo —y esto es lo que lo hace un *representation learning* paper y no solo un *image synthesis* paper— **demostrar que las representaciones aprendidas son buenas**: que el discriminador, usado como extractor de características, compite con métodos no supervisados consolidados; y que el espacio latente $Z$ del generador tiene estructura semántica navegable (interpolaciones suaves) y *aritmética lineal* del estilo `vector("Rey") − vector("Hombre") + vector("Mujer") ≈ vector("Reina")`, pero sobre caras. Concretamente, los autores listan cuatro contribuciones: (1) el conjunto de restricciones arquitectónicas que estabilizan el entrenamiento; (2) usar los discriminadores entrenados para clasificación de imágenes con rendimiento competitivo frente a otros algoritmos no supervisados; (3) visualizar los filtros aprendidos y mostrar empíricamente que filtros específicos aprenden a dibujar objetos específicos; (4) mostrar las propiedades de aritmética de vectores del generador.

Para la **Clase 29 (Modelos Generativos en Visión)** este paper importa porque es el "cómo se hace una GAN que de verdad funcione con imágenes". La clase introduce el juego adversarial (Goodfellow) y los autoencoders variacionales (VAE); DCGAN es el puente práctico: traduce el objetivo adversarial abstracto a una receta convolucional reproducible y, de paso, muestra que el espacio latente continuo —que la clase asocia sobre todo al VAE— también emerge en una GAN bien construida.

## 2. Contexto histórico: la GAN original y por qué no escalaba a imágenes

En 2014 Goodfellow et al. plantearon la GAN como un juego de dos jugadores: un generador $G$ que mapea ruido $z \sim p_z$ a muestras, y un discriminador $D$ que estima la probabilidad de que una muestra venga de los datos reales y no de $G$. El objetivo —el minimax sobre $V(D,G)$— era elegante y prescindía de la verosimilitud explícita y de funciones de costo heurísticas (como el error cuadrático medio pixel a pixel, que el paper de DCGAN señala como poco atractivo para *representation learning*). Pero la formulación original se implementaba con perceptrones multicapa (MLP) y generaba, en palabras de los propios autores de DCGAN, imágenes "ruidosas e incomprensibles".

El problema no era teórico sino de **estabilidad y arquitectura**. Cuando uno intentaba reemplazar los MLP por las CNN profundas que ya dominaban la visión supervisada (AlexNet, VGG y sucesores), el entrenamiento adversarial colapsaba. El paper enumera la patología más temida: el **mode collapse**, donde "el generador colapsa todas las muestras a un único punto". El equilibrio adversarial es frágil porque $G$ y $D$ se entrenan simultáneamente con objetivos opuestos; si uno gana demasiado rápido, los gradientes del otro se degradan y el aprendizaje se detiene.

La comunidad había buscado rodeos. LAPGAN (Denton et al., 2015) evitaba el problema de escalar una sola red generando la imagen por **etapas** sobre una pirámide laplaciana: en vez de pedirle a una CNN profunda que produjera la imagen final de una vez, encadenaba modelos que iban refinando de baja a alta resolución. Funcionaba, pero el encadenamiento introducía ruido y los objetos salían "tambaleantes" (*wobbly*). Otras líneas —DRAW recurrente (Gregor et al., 2015), redes de deconvolución (Dosovitskiy et al., 2014), el VAE de Kingma & Welling (2013), la difusión de Sohl-Dickstein et al. (2015)— lograban imágenes naturales con grados variables de éxito (el VAE, notoriamente, producía muestras *borrosas*), pero ninguna "aprovechaba los generadores para tareas supervisadas". El hueco que DCGAN se propone llenar es doble: una receta que haga *una sola* CNN adversarial entrenable de extremo a extremo, y la evidencia de que lo que aprende sirve como representación reutilizable.

## 3. Contribución central: las pautas arquitectónicas

La contribución que se cita hasta hoy cabe en cinco viñetas. El paper las titula **"Architecture guidelines for stable Deep Convolutional GANs"** y son el resultado empírico de la exploración extensiva del modelo:

- **Reemplazar todas las capas de *pooling* por convoluciones con *stride*:** convoluciones con paso (*strided*) en el discriminador, y convoluciones de paso fraccionario (*fractional-strided*, también llamadas transpuestas) en el generador. La red **aprende su propio submuestreo espacial** (en el discriminador) y su propio sobremuestreo (en el generador), en lugar de imponer una operación determinista de *max-pooling*. La idea viene de la *all convolutional net* (Springenberg et al., 2014).
- **Usar *batch normalization* en generador y discriminador.** Normaliza la entrada de cada unidad a media cero y varianza unitaria; ataca los problemas de mala inicialización y mejora el flujo de gradiente en redes profundas. El paper lo califica de **crítico** para que generadores profundos empiecen a aprender y para prevenir el *mode collapse*. Con un matiz fino y muy citado: aplicar batchnorm a *todas* las capas causaba oscilación; la solución fue **no aplicarlo a la capa de salida del generador ni a la capa de entrada del discriminador**.
- **Eliminar las capas *fully-connected* ocultas** en arquitecturas profundas. Probaron *global average pooling* (estado del arte en clasificación), que aumentaba la estabilidad pero frenaba la convergencia; el punto medio adoptado fue conectar directamente las características convolucionales más altas a la entrada/salida. La primera capa del generador, que toma el ruido uniforme $Z$, es técnicamente una multiplicación matricial, pero su resultado se **redimensiona a un tensor 4-D** que arranca la pila convolucional; la última capa del discriminador se aplana y va a una única salida sigmoide.
- **Activación ReLU en el generador** en todas las capas excepto la de salida, que usa **Tanh**. Observaron que una activación acotada (Tanh en $[-1,1]$) permite al modelo saturar y cubrir más rápido el espacio de color de la distribución de entrenamiento.
- **Activación LeakyReLU en el discriminador** en todas las capas (pendiente de fuga 0.2), en contraste con la *maxout* del paper GAN original; funcionaba especialmente bien para modelado de mayor resolución.

La lección de diseño es que estas pautas no son trucos aislados sino un **paquete coherente**: convoluciones con stride para que la geometría espacial sea aprendida en vez de impuesta; batchnorm para domar el flujo de gradiente del juego adversarial; ausencia de capas densas para mantener la profundidad sin explotar parámetros; y la pareja ReLU/LeakyReLU + Tanh para acotar y estabilizar las activaciones. Es la receta que convirtió a las GAN de curiosidad inestable en herramienta de ingeniería.

## 4. Método: la arquitectura del generador y el entrenamiento

### 4.1. El generador como pila de convoluciones de paso fraccionario

La Figura 1 del paper describe el generador canónico para LSUN. Una distribución uniforme de **100 dimensiones** ($Z \in \mathbb{R}^{100}$) se proyecta a una representación convolucional de **pequeña extensión espacial pero con muchos mapas de características** (un tensor 4-D tras el reshape). A partir de ahí, una serie de **cuatro convoluciones de paso fraccionario** —que el paper aclara que "en algunos papers recientes se llaman erróneamente deconvoluciones"— convierte progresivamente esa representación de alto nivel en una imagen de **64 × 64** píxeles. Cada convolución transpuesta duplica aproximadamente la resolución espacial mientras reduce la profundidad de canales: la red va literalmente "dibujando" desde un código abstracto hacia píxeles. Notablemente, **no se usan capas fully-connected ni de pooling**.

El discriminador es el espejo: toma la imagen de 64 × 64, la procesa con convoluciones de stride 2 (que reducen resolución y aumentan canales), aplica LeakyReLU y batchnorm, aplana la última capa convolucional y la pasa a una sola salida sigmoide que estima la probabilidad real/falso.

### 4.2. Detalles de entrenamiento adversarial

El paper es deliberadamente explícito con la receta de optimización, lo que facilita reproducirla:

- **Preprocesamiento:** ninguno salvo escalar las imágenes al rango de Tanh, $[-1, 1]$. Sin *data augmentation* en los datasets principales.
- **Inicialización de pesos:** distribución Normal centrada en cero con desviación estándar **0.02**.
- **Optimizador:** **Adam** (Kingma & Ba, 2014), no SGD con momentum como en trabajo previo. La tasa de aprendizaje sugerida de 0.001 era demasiado alta; usaron **0.0002**. Y el término de momentum $\beta_1$ en su valor por defecto de 0.9 causaba oscilación; **reducirlo a 0.5 estabilizó el entrenamiento** —otro detalle pequeño pero hoy estándar.
- **Mini-batch:** SGD con tamaño de lote de **128**.
- **LeakyReLU:** pendiente de fuga 0.2 en todos los modelos.

El conjunto de estas decisiones —junto con las cinco pautas arquitectónicas— es lo que hace que el juego minimax converja de forma robusta. El paper insiste en que estas elecciones fueron empíricas, fruto de mucha prueba y error, no derivadas de teoría.

## 5. Experimentos

### 5.1. Calidad y generalización: LSUN bedrooms, caras, ImageNet

- **LSUN bedrooms** (poco más de 3 millones de ejemplos): el dataset estrella para mostrar escalado a más datos y mayor resolución. Para descartar que la calidad viniera de *memorizar*, muestran muestras tras **una sola pasada** por el dataset (Fig. 2, imitando aprendizaje online) y tras **cinco épocas** (Fig. 3). El argumento: si el modelo memorizara, no podría producir buenas muestras tras una única pasada con tasa de aprendizaje pequeña. Para reforzarlo, aplican un proceso de **deduplicación** basado en un autoencoder denoising 3072-128-3072 con código binarizado (semantic hashing), que detectó y removió ~275.000 casi-duplicados con una tasa de falsos positivos estimada bajo 1 en 100.
- **Faces** (~350.000 recortes de caras de 10K personas, obtenidas de búsquedas web con nombres de dbpedia y un detector OpenCV): el dataset sobre el que se hacen los experimentos más vistosos de espacio latente.
- **ImageNet-1k**: usado como fuente de imágenes naturales para entrenamiento no supervisado, sobre recortes centrales de 32 × 32. Es el modelo cuyo discriminador se reutiliza como extractor de características.

### 5.2. El discriminador como extractor de características (la parte de *representation learning*)

Aquí el paper valida que las GAN aprenden representaciones útiles, no solo imágenes bonitas. La técnica estándar: usar la red como extractor sobre datasets supervisados y entrenar modelos lineales encima.

- **CIFAR-10:** entrenan el DCGAN sobre ImageNet-1k (¡nunca sobre CIFAR-10!), toman las características convolucionales del discriminador de *todas* las capas, hacen max-pooling a una grilla 4 × 4, las concatenan en un vector de 28.672 dimensiones y entrenan un **L2-SVM lineal** encima. Resultado: **82.8% de precisión**, superando todos los métodos basados en K-means (80.6% / 82.0%), aunque por debajo de Exemplar CNN (84.3%). Como el DCGAN nunca vio CIFAR-10, el experimento también demuestra **robustez de dominio** de las características.
- **SVHN** (dígitos de casas, régimen de pocas etiquetas): con solo **1000 ejemplos etiquetados**, el mismo pipeline + L2-SVM alcanza **22.48% de error de test**, estado del arte para ese régimen. Un control importante: una CNN puramente supervisada *con la misma arquitectura* sobre los mismos datos logra solo 28.87%, lo que prueba que el mérito está en las representaciones no supervisadas y no en la mera topología convolucional.

### 5.3. Caminar por el espacio latente y aritmética de vectores

Esta es la sección más influyente conceptualmente.

- **Walking in the latent space (Fig. 4):** interpolan entre puntos aleatorios de $Z$ y observan **transiciones suaves**, donde cada imagen intermedia sigue pareciendo plausiblemente un dormitorio. En una fila, "un cuarto sin ventana se transforma lentamente en un cuarto con una ventana gigante"; en otra, "lo que parece un televisor se transforma en una ventana". La suavidad de las transiciones es evidencia de que el modelo aprendió un *manifold* coherente y no memorizó (que daría transiciones abruptas).
- **Visualizing discriminator features (Fig. 5):** con *guided backpropagation* muestran que los filtros del discriminador se activan ante partes típicas de un dormitorio (camas, ventanas), frente a un baseline de filtros aleatorios que no responden a nada semántico.
- **Forgetting to draw objects (Fig. 6):** ajustan una regresión logística sobre los mapas de características de la segunda capa convolucional más alta para predecir qué activaciones corresponden a "ventana", eliminan esos ~200 mapas, y el generador **deja de dibujar ventanas** (las reemplaza por puertas o espejos), manteniendo la composición de la escena. Evidencia de **desenredo** (*disentanglement*) entre representación de escena y de objeto.
- **Vector arithmetic on faces (Figs. 7 y 8):** el clímax. Inspirados en el `King − Man + Woman ≈ Queen` de word2vec (Mikolov et al., 2013), promedian los vectores $Z$ de tres ejemplares por concepto (un solo ejemplar resultaba inestable) y hacen aritmética: por ejemplo "hombre con lentes − hombre sin lentes + mujer sin lentes" produce una mujer con lentes. También construyen un **vector "turn"** (giro) promediando caras mirando a izquierda vs derecha, y al sumarlo a muestras aleatorias **rotan la pose de la cara** de forma fiable. Es, según los autores, la primera demostración de esta estructura lineal emergiendo de forma puramente no supervisada.

### 5.4. Material suplementario: MNIST condicional

Un DCGAN condicional sobre MNIST evaluado con clasificador de vecino más cercano logra 2.98% / 1.48% de error de test (50K / 10M muestras), igualando o superando técnicas de aumento de datos como InfiMNIST, lo que sugiere que captura bien las distribuciones condicionales.

## 6. Limitaciones reconocidas

El paper es honesto sobre lo que no resolvió:

- **Inestabilidad residual.** "Quedan algunas formas de inestabilidad del modelo": al entrenar por más tiempo, los modelos a veces colapsaban un subconjunto de filtros a un único modo oscilante. El *mode collapse* fue mitigado, no eliminado.
- **Receta empírica, no teoría.** Las pautas surgieron de exploración extensiva; el paper no ofrece una explicación teórica de *por qué* exactamente estabilizan el juego adversarial. Son condiciones necesarias halladas a mano.
- **No iguala el estado del arte supervisado/no supervisado más fuerte.** En CIFAR-10 queda por debajo de Exemplar CNN; los autores reconocen que afinar (*finetune*) el discriminador podría mejorar, pero lo dejan para trabajo futuro.
- **Evaluación cualitativa.** Para juzgar las muestras evitan deliberadamente la verosimilitud (la consideran mala métrica, citando a Theis et al., 2015) y el vecino más cercano en espacio de píxeles (trivialmente engañable). Buena parte de la evidencia es visual, lo que dificulta la comparación cuantitativa rigurosa entre modelos generativos —un problema del campo entero en esa época.
- **Resolución limitada.** Los generadores operan a 64 × 64 (32 × 32 para ImageNet/CIFAR). Escalar a alta resolución seguiría siendo un problema abierto que ocuparía a la comunidad por años.

## 7. Impacto: DCGAN como base de la era de la síntesis de imágenes

DCGAN es uno de los papers más citados en visión generativa, y con razón: **hizo las GAN prácticas para imágenes**. Antes de él, "entrenar una GAN" era un ejercicio frágil de alquimia; después, las cinco pautas se volvieron el *default* sobre el que casi todo el mundo construía. El propio término "arquitectura DCGAN" se convirtió en sinónimo de "GAN convolucional que funciona".

Su descendencia es enorme. La línea de **progresión de resolución y calidad** —Progressive Growing of GANs, **StyleGAN** (Karras et al., 2019) y sus sucesores— hereda directamente el esqueleto convolucional generador/discriminador de DCGAN y lo lleva a fotorrealismo de alta resolución; el espacio latente $\mathcal{Z}$ navegable y semánticamente estructurado que DCGAN demostró es, precisamente, lo que StyleGAN industrializó con su espacio intermedio $\mathcal{W}$ y la edición semántica de atributos. La idea de la **aritmética de vectores en el latente** anticipó toda la literatura de edición de imágenes por manipulación de códigos latentes. Y la noción de **reutilizar el discriminador como extractor de características** prefiguró el uso de modelos generativos para aprendizaje de representaciones, una línea que reaparece en los modelos de difusión y los enfoques auto-supervisados modernos.

Incluso los detalles "pequeños" se volvieron folklore de ingeniería: Adam con $\beta_1 = 0.5$ y lr $= 0.0002$, inicialización Normal con $\sigma = 0.02$, batchnorm en todas las capas salvo salida-de-G/entrada-de-D, Tanh a la salida. Quien implementa hoy una GAN básica suele estar, sin saberlo, copiando DCGAN.

## 8. Conexión con la Clase 29 (Modelos Generativos en Visión)

La Clase 29 organiza los modelos generativos en familias —GAN, VAE, y normalmente difusión. DCGAN encaja en dos lugares:

- **Complementa la sección de GANs como el "cómo se hace que funcione".** La clase presenta el objetivo adversarial de Goodfellow et al. (2014) de forma abstracta (el minimax entre $G$ y $D$). DCGAN es el complemento operativo: traduce ese objetivo a una arquitectura concreta y a una receta de entrenamiento (las cinco pautas + Adam afinado) que de verdad converge sobre imágenes. Sin DCGAN, el juego adversarial es una idea bonita pero inestable; con DCGAN, es una herramienta. Es el eslabón pedagógico entre "qué optimiza una GAN" y "qué tecleo para entrenar una".

- **El espacio latente continuo conecta con el VAE.** La clase introduce el espacio latente continuo y navegable sobre todo en el contexto del **VAE** (donde el *prior* gaussiano y el término KL hacen que el latente sea explícitamente continuo e interpolable). DCGAN muestra que **esa misma propiedad emerge en una GAN** sin imponerla por construcción: las interpolaciones suaves en $Z$ (Fig. 4) y la aritmética de vectores en caras (Figs. 7-8) demuestran que el generador aprendió un manifold semánticamente estructurado. Es un buen punto de discusión en clase: VAE *garantiza* un latente regular a costa de muestras borrosas; GAN/DCGAN *no lo garantiza* pero lo obtiene empíricamente con muestras más nítidas. El contraste ilumina el trade-off central entre las dos familias.

Lecturas relacionadas dentro del curso:

- [Goodfellow et al. (2014), GAN original](/papers/goodfellow-gan-2014) — el objetivo adversarial que DCGAN vuelve práctico.
- [Karras et al. (2019), StyleGAN](/papers/stylegan-karras-2019) — el descendiente que lleva el esqueleto y el latente navegable de DCGAN al fotorrealismo de alta resolución.
- [Fundamento: Modelos Generativos](/fundamentos/modelos-generativos) — el marco conceptual (GAN vs VAE vs difusión) en el que DCGAN se ubica.
- [Clase 29: Modelos Generativos en Visión](/clases/clase-29) — la clase que este análisis acompaña.
