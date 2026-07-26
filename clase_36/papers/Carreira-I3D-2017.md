# Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset (I3D) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*.
- **Autores:** João Carreira y Andrew Zisserman, ambos en **DeepMind**; Zisserman además afiliado al Department of Engineering Science, University of Oxford.
- **Venue:** *IEEE Conference on Computer Vision and Pattern Recognition* (CVPR 2017).
- **Preprint:** arXiv:1705.07750v3 (12 feb 2018), [arxiv.org/abs/1705.07750](https://arxiv.org/abs/1705.07750).
- **Modelo estrella:** **Two-Stream Inflated 3D ConvNet (I3D)**.
- **Dataset acompañante:** Kinetics Human Action Video Dataset (Kay et al., 2017), presentado en un paper hermano.

El trabajo parte de un diagnóstico incómodo del campo: los datasets estándar de clasificación de acciones, **UCF-101 y HMDB-51**, son tan pequeños (del orden de 10 000 videos) que *casi cualquier* arquitectura obtiene desempeños parecidos, lo que hace imposible distinguir qué diseño es realmente bueno. La paucidad de datos había "hecho difícil identificar buenas arquitecturas de video". El paper re-evalúa el estado del arte a la luz de **Kinetics**, un dataset dos órdenes de magnitud más grande —400 clases de acciones humanas, más de 400 clips por clase, cada clip de un video único de YouTube—, y con ello logra dos cosas: (1) ordenar por mérito real las familias de arquitecturas de video, y (2) proponer un modelo nuevo que aprovecha al máximo ese pre-entrenamiento.

La contribución técnica central es **I3D**: en vez de diseñar desde cero una arquitectura espacio-temporal, los autores toman una red de clasificación de imágenes muy profunda y ya probada (**Inception-v1 / GoogLeNet**, pre-entrenada en ImageNet) e **inflan** sus filtros y kernels de pooling de 2D a 3D. Un filtro $N \times N$ se convierte en $N \times N \times N$. Más aún, se pueden **heredar los parámetros pre-entrenados** de ImageNet mediante un truco de "bootstrapping". Sobre esta base se monta una configuración **two-stream** clásica: una red I3D para RGB y otra para flujo óptico, promediando sus predicciones.

El resultado empírico es contundente. Tras pre-entrenar en Kinetics y hacer fine-tuning, I3D lleva el estado del arte a **98.0% en UCF-101 y 80.9% en HMDB-51** (promediados sobre los tres splits estándar), lo que corresponde a reducciones de error de clasificación de **63% y 35%** respecto del mejor modelo previo. Para la **Clase 36 (Introduction to Video Analysis)** este paper es el pivote: sintetiza en una sola arquitectura las tres grandes ideas del *zoo* de video —convolución 3D, arquitectura two-stream y transferencia de pre-entrenamiento— y establece la receta que dominaría los años siguientes: **"pre-entrenar en Kinetics, transferir"**.

## 2. Contexto: el zoo de arquitecturas de video sin un ImageNet propio

Para entender por qué I3D importa hay que ver el paisaje que encontró. Mientras que en imágenes las arquitecturas habían madurado rápido y había líderes claros (de AlexNet a VGG-16 a ResNet), para video **no había una arquitectura ganadora**. El paper identifica los ejes en que las propuestas divergían:

- si los operadores convolucionales usan kernels **2D (basados en imagen)** o **3D (basados en video)**;
- si la entrada es solo **RGB** o incluye además **flujo óptico** pre-computado;
- y, en el caso 2D, cómo se propaga la información entre fotogramas: con capas **recurrentes tipo LSTM** o por **agregación de features** en el tiempo.

Este espacio de diseño había producido varias familias sin un veredicto. El motivo de fondo era la **falta de un "ImageNet de video"**. En imágenes, uno de los beneficios inesperados del desafío ImageNet fue descubrir que redes profundas entrenadas sobre 1000 categorías servían para *otras* tareas y dominios: los features fc7 de una red ImageNet alimentaban clasificación y detección en PASCAL VOC, y mejorar la arquitectura base (de AlexNet a VGG-16) se traducía inmediatamente en mejoras aguas abajo. En video quedaba abierta la pregunta análoga: ¿entrenar una red de clasificación de acciones sobre un dataset suficientemente grande daría un empujón comparable al transferirla a otra tarea o dataset temporal? Los benchmarks pequeños impedían siquiera plantearla en serio.

Kinetics es la respuesta material a esa carencia. Con 400 clases, 400+ ejemplos por clase y unos **240 000 videos de entrenamiento**, y recolectado de videos de YouTube realistas y difíciles, provee por fin la escala para (a) discriminar arquitecturas y (b) servir como fuente de pre-entrenamiento transferible. El título del paper, tomado de un fotograma de la película *Quo Vadis* (1951), es una metáfora deliberada: en un solo fotograma no se sabe si los actores están por besarse o ya lo hicieron —las acciones son ambiguas cuadro a cuadro—, y la pregunta "¿hacia dónde va esto?" se dirige tanto a la escena como al reconocimiento de acciones como campo.

## 3. Contribución central

El paper hace dos aportes entrelazados:

1. **Una evaluación comparativa rigurosa** de cinco familias de arquitecturas de video sobre el mismo backbone (Inception-v1) y bajo el mismo protocolo, variando el dataset de entrenamiento y midiendo cuánto ayuda el pre-entrenamiento. Esto convierte una discusión antes cualitativa en una tabla ordenada.
2. **El modelo I3D**, que combina tres ingredientes que hasta entonces se usaban por separado:
   - **Inflado 2D→3D:** convertir arquitecturas de imagen profundas y probadas en extractores espacio-temporales, sin re-diseñar nada.
   - **Bootstrapping de pesos:** heredar los parámetros ImageNet mediante el *boring-video fixed point*.
   - **Two-stream sobre I3D:** una red I3D RGB más una red I3D de flujo óptico.

La observación clave que habilita todo es simple pero poderosa: las redes de clasificación de imágenes *muy profundas* (Inception, VGG-16, ResNet) pueden **inflarse trivialmente** a extractores espacio-temporales, y sus pesos pre-entrenados proveen una inicialización valiosa. Hasta entonces las 3D ConvNets habían sido forzosamente **poco profundas** (hasta 8 capas), porque su alta dimensionalidad de parámetros, combinada con la escasez de datos de video etiquetados, las hacía difíciles de entrenar y parecía **excluirlas del pre-entrenamiento ImageNet**. I3D rompe ese techo: hereda profundidad y pre-entrenamiento de las redes de imagen.

## 4. Método

### 4.1. Las cinco familias evaluadas

El paper implementa y compara cinco tipos de arquitectura, todas (salvo la 3D-ConvNet) construidas sobre Inception-v1 con batch normalization como backbone común, para poder aislar qué cambio ayuda más:

- **(a) ConvNet + LSTM ("The Old I").** Reutiliza una red de imagen con cambios mínimos: extrae features por fotograma y agrega en el tiempo. Aquí se coloca una **LSTM con 512 unidades ocultas** (con batch normalization) tras la última capa de average pooling de Inception-v1, y una capa fully connected como clasificador. Se entrena con pérdida de entropía cruzada en todos los pasos, pero en test solo se usa la salida del **último** fotograma. La entrada submuestrea 1 de cada 5 fotogramas de un stream de 25 fps. La debilidad conceptual: puede ignorar estructura temporal fina (un modelo puramente de agregación no distingue *abrir* de *cerrar* una puerta).
- **(b) 3D-ConvNet ("The Old II").** Filtros espacio-temporales que crean jerarquías de representación directamente sobre el video. Es una variante tipo **C3D**: 8 capas convolucionales, 5 de pooling, 2 fully connected, entrada de clips de **16 fotogramas** con recortes de $112 \times 112$. Los autores añaden batch normalization y un stride temporal de 2 en el primer pooling. Es la **única red entrenada desde cero** (no hereda ImageNet), y tiene muchos más parámetros (**79M**) por la dimensión extra del kernel, lo que la hace más difícil de entrenar.
- **(c) Two-Stream ("The Old III").** El esquema de Simonyan y Zisserman: promediar las predicciones de un **fotograma RGB único** y una **pila de 10 fotogramas de flujo óptico** externo, ambos pasados por dos réplicas de una ConvNet pre-entrenada en ImageNet. El stream de flujo tiene una capa de entrada adaptada con el doble de canales (el flujo tiene componente horizontal y vertical). Es muy eficiente de entrenar y testear.
- **(d) 3D-Fused Two-Stream.** Una extensión que **fusiona** los streams espacial y de movimiento tras la última capa convolucional mediante una capa convolucional 3D. La implementación toma **5 fotogramas RGB** consecutivos muestreados cada 10, más sus snippets de flujo; los grids de features $5 \times 7 \times 7$ (tiempo, x, y) pasan por una conv 3D de $3 \times 3 \times 3$ con 512 canales, un max-pooling 3D $3 \times 3 \times 3$ y una fully connected. Los pesos nuevos se inicializan con ruido gaussiano.
- **(e) Two-Stream I3D ("The New").** La propuesta, descrita en detalle abajo.

La tabla de la Sección 4.1 del paper resume el número de parámetros y las huellas temporales: ConvNet+LSTM 9M, 3D-ConvNet 79M, Two-Stream 12M, 3D-Fused 39M y **Two-Stream I3D 25M**. En entrenamiento, I3D consume **64 fotogramas RGB y 64 de flujo** (huella de 2.56 s), y en test procesa el video completo, promediando predicciones temporalmente.

### 4.2. Inflado de 2D a 3D

La idea del inflado es negarse a repetir el doloroso proceso de prueba y error que produjo las buenas arquitecturas de imagen, y en su lugar **convertir esas arquitecturas exitosas directamente en 3D ConvNets**. Se parte de una arquitectura 2D y se inflan todos los filtros y kernels de pooling dándoles una **dimensión temporal adicional**. Como los filtros suelen ser cuadrados, simplemente se los vuelve cúbicos:

$$N \times N \;\longrightarrow\; N \times N \times N$$

Un kernel de pooling de $N \times N$ se convierte análogamente en $N \times N \times N$. Con esto, sin re-diseñar la topología, Inception-v1 pasa a operar sobre volúmenes espacio-temporales.

### 4.3. Bootstrapping de pesos: el *boring-video fixed point*

El inflado da la *arquitectura* 3D, pero el gran valor está en **heredar también los parámetros** ImageNet. El mecanismo es elegante. Se observa que una imagen puede convertirse en un video (aburrido, *boring*) copiándola repetidamente $N$ veces a lo largo del eje temporal. Los autores exigen que el modelo 3D respete lo que llaman el **boring-video fixed point**: *las activaciones agrupadas (pooled) sobre un video aburrido deben ser iguales a las de la imagen original de un solo cuadro*.

Gracias a la **linealidad** de la convolución, esto se logra **repitiendo los pesos del filtro 2D $N$ veces** a lo largo de la dimensión temporal y **reescalándolos dividiendo por $N$**:

$$w^{3D}(t) = \frac{1}{N}\, w^{2D}, \qquad t = 1, \dots, N$$

Así la respuesta del filtro convolucional sobre el video aburrido es idéntica a la respuesta 2D sobre la imagen. Y como las salidas de las capas convolucionales para un video aburrido son constantes en el tiempo, las salidas de las no-linealidades puntuales y de las capas de average/max pooling también coinciden con el caso 2D. Por tanto la **respuesta global de la red respeta el punto fijo**, y el modelo 3D queda *implícitamente pre-entrenado en ImageNet*. En la práctica, esto le regala a I3D millones de imágenes etiquetadas de inicialización que ninguna 3D ConvNet entrenada desde cero podía aprovechar.

### 4.4. Ritmo del campo receptivo en espacio, tiempo y profundidad

El punto fijo deja libertad sobre **cómo** inflar los operadores de pooling en el tiempo y qué stride temporal usar; estos son los factores que moldean el tamaño de los campos receptivos. Casi todos los modelos de imagen tratan las dos dimensiones espaciales por igual (mismos kernels y strides), lo cual es natural. Pero un campo receptivo **simétrico no es necesariamente óptimo al incluir el tiempo**: depende de la tasa de fotogramas y las dimensiones de imagen. Si crece demasiado rápido en el tiempo relativo al espacio, puede **fundir bordes de objetos distintos** y romper la detección temprana de features; si crece demasiado lento, puede no capturar bien la dinámica de la escena.

Con videos procesados a **25 fps**, los autores encontraron útil **no hacer pooling temporal en las dos primeras capas de max-pooling** (usando kernels $1 \times 3 \times 3$ y stride 1 en tiempo), manteniendo kernels y strides simétricos en las demás. La capa final de average pooling usa un kernel $2 \times 7 \times 7$. El modelo se entrena con **snippets de 64 fotogramas** y se testea sobre videos completos promediando predicciones en el tiempo. La arquitectura resultante es la **Inflated Inception-v1**.

### 4.5. Dos streams 3D: por qué el flujo óptico sigue siendo útil

Aunque una 3D ConvNet *debería* poder aprender features de movimiento directamente del RGB, sigue siendo un cómputo **puramente feedforward**, mientras que los algoritmos de flujo óptico son en cierto sentido **recurrentes** (realizan optimización iterativa de los campos de flujo). Quizás por esta falta de recurrencia, los autores encontraron experimentalmente que **seguía siendo valioso** tener la configuración two-stream: una red I3D entrenada sobre RGB y otra sobre flujo, que carga información de movimiento optimizada y suave. Las dos redes se **entrenan por separado** y sus predicciones se **promedian en test**. El flujo óptico se computó con un algoritmo **TV-L1**.

### 4.6. Detalles de implementación

Todos los modelos salvo la 3D-ConvNet tipo C3D usan Inception-v1 pre-entrenada en ImageNet como base. Cada capa convolucional va seguida de batch normalization y ReLU (excepto las últimas que producen los class scores). El entrenamiento usa SGD con momento 0.9, paralelización síncrona sobre **32 GPUs** para la mayoría de los modelos y **64 GPUs** para las 3D ConvNets (que reciben muchos fotogramas y necesitan más GPUs para armar batches grandes). Se entrenó en Kinetics por **110 000 pasos**, con reducción del learning rate 10× al saturarse la pérdida de validación, y hasta 5000 pasos en UCF-101/HMDB-51 con 16 GPUs. Todo se implementó en **TensorFlow**. La aumentación de datos incluye recorte aleatorio espacial (redimensionar el lado menor a 256 y recortar $224 \times 224$) y temporal, además de volteo horizontal aleatorio consistente por video; los videos cortos se repiten en bucle para satisfacer la interfaz de entrada de cada modelo.

## 5. Experimentos

### 5.1. El dataset Kinetics

Kinetics se enfoca en **acciones humanas** (no actividades ni eventos), con clases que cubren: acciones de persona singular (dibujar, beber, reír, golpear); persona-persona (abrazar, besar, dar la mano); y persona-objeto (abrir regalos, cortar el césped, lavar platos). Algunas acciones son de grano fino y exigen razonamiento temporal (distintos tipos de nado). Tiene 400 clases, 400+ clips por clase de videos únicos, **240 000 videos de entrenamiento**, clips de ~10 s (sin videos sin recortar) y un test de 100 clips por clase.

### 5.2. Comparación de arquitecturas (Tabla 2)

Entrenando y testeando *dentro* de cada dataset, la Tabla 2 arroja varias observaciones. Primero, **los modelos I3D ganan en todos los datasets** con cualquier modalidad (RGB, flujo o RGB+flujo). Two-Stream I3D alcanza en UCF-101 **93.4%** (RGB+flujo), en HMDB-51 **66.4%** y en Kinetics **74.2%**. Esto es notable dado su gran número de parámetros y lo pequeños que son UCF-101 y HMDB-51: **muestra que los beneficios del pre-entrenamiento ImageNet se extienden a las 3D ConvNets**.

Segundo, el desempeño de todos los modelos es mucho **más bajo en Kinetics que en UCF-101**, indicando la diferencia de dificultad; y es más alto que en HMDB-51, que fue construido deliberadamente difícil (muchos clips comparten la misma escena con acciones distintas). Tercero, el ranking de arquitecturas es **mayormente consistente** entre datasets.

Sobre el rol del flujo óptico: las arquitecturas two-stream son superiores en todos los datasets, pero el valor **relativo** de RGB y flujo varía mucho. El flujo solo aporta un poco más que RGB en UCF-101, mucho más en HMDB-51, y **sustancialmente menos en Kinetics**. La inspección visual sugiere que Kinetics tiene **mucho más movimiento de cámara**, lo que dificulta el trabajo del stream de movimiento. Aun así, **I3D exprime más el stream de flujo** que los demás modelos, probablemente por su campo receptivo temporal mucho más largo (**64 fotogramas vs. 10** durante el entrenamiento) y su maquinaria de extracción temporal más integrada.

### 5.3. El valor del pre-entrenamiento ImageNet (Tabla 3)

La Tabla 3 compara entrenar en Kinetics **desde ImageNet** vs. **desde cero**. El pre-entrenamiento ImageNet **sigue ayudando en todos los casos**, y algo más notoriamente para los streams RGB, como cabría esperar. Con ImageNet+Kinetics, RGB-I3D alcanza **71.1%** (Top-1) / 89.3% (Top-5) y Two-Stream I3D **74.2%** / 91.3% en Kinetics.

### 5.4. Transferencia a UCF-101 y HMDB-51 (Tabla 4)

Aquí está el corazón del argumento. Se evalúa la generalizabilidad de las redes entrenadas en Kinetics de dos formas: **(Fixed)** congelar la red y entrenar solo un clasificador softmax nuevo sobre sus features, y **(Full-FT)** hacer fine-tuning end-to-end. Ambas se comparan con **(Original)**: entrenar directamente en UCF-101/HMDB-51.

El resultado claro es que **todas las arquitecturas se benefician del pre-entrenamiento en el video adicional de Kinetics, pero algunas mucho más que otras** —notablemente la I3D-ConvNet y la 3D-ConvNet (esta última partiendo de una base mucho más baja). Para I3D, entrenar solo las últimas capas tras Kinetics (**Fixed**) ya rinde **mucho mejor** que entrenar directamente en los datasets pequeños. Con ImageNet+Kinetics y Full-FT, Two-Stream I3D llega a **98.0% en UCF-101 y 81.2% en HMDB-51** (split 1).

La explicación que ofrecen los autores para la **transferibilidad superior** de los features de I3D es su **alta resolución temporal**: se entrenan sobre snippets de 64 fotogramas a 25 fps y procesan todos los fotogramas en test, lo que les permite capturar estructura temporal de grano fino. Dicho al revés: los métodos con entradas de video más *ralas* se benefician menos del dataset grande porque, desde su perspectiva, **los videos no difieren tanto de las imágenes de ImageNet**. La diferencia sobre la C3D se explica además porque I3D es mucho más profunda con muchos menos parámetros, arranca en caliente desde ImageNet, se entrena sobre videos 4× más largos y opera a 2× la resolución espacial.

### 5.5. Comparación con el estado del arte (Tabla 5)

Promediando sobre los tres splits estándar, el mejor método previo era el de Feichtenhofer et al. (ST-ResNet + IDT), con 94.6% en UCF-101 y 70.3% en HMDB-51. **Cualquiera** de los modelos RGB-I3D o Flow-I3D solo, pre-entrenados en Kinetics, ya supera todo lo publicado. La arquitectura combinada Two-Stream I3D con **ImageNet+Kinetics** amplía la ventaja a **98.0% en UCF-101 y 80.9% en HMDB-51**, correspondiente a reducciones de error de clasificación de **63% y 35%** respecto del mejor modelo previo. La brecha con las 3D ConvNets previas (C3D) es aún mayor, pese a que C3D se entrenó con más videos (1M de Sports-1M más un dataset interno) y en ensamble combinado con IDT — atribuible a la mejor calidad de Kinetics y a que I3D es simplemente **una mejor arquitectura**.

Un detalle interpretable de la Figura 4: al visualizar los 64 filtros conv1 ($7 \times 7 \times 7$), los filtros de la red RGB-I3D desarrollan **rica estructura temporal** y ya no se parecen a los originales de Inception-v1, mientras que —curiosamente— los del stream de flujo permanecen más cercanos a los filtros ImageNet originales.

## 6. Limitaciones

- **Costo de cómputo.** La 3D-ConvNet y la propia I3D consumen muchos fotogramas y requieren batches grandes: los modelos 3D se entrenaron sobre **64 GPUs**. I3D es cara de entrenar y de evaluar en comparación con los enfoques 2D+LSTM.
- **Sigue dependiendo del flujo óptico.** Pese a que una 3D ConvNet debería aprender movimiento del RGB, la mejor configuración **todavía necesita** un stream de flujo óptico externo pre-computado (con TV-L1), lo que añade un paso costoso fuera de la red y contradice parcialmente la promesa de "aprender todo end-to-end". Los autores dejan como pregunta abierta integrar alguna forma de **estabilización de movimiento** en la arquitectura.
- **Exploración de arquitecturas incompleta.** Reconocen no haber explorado exhaustivamente el espacio: no usaron *action tubes*, mecanismos de atención sobre los actores, ni las detecciones enlazadas en el tiempo que otros trabajos proponían para localizar espacio-temporalmente al actor.
- **Transferencia probada solo en una tarea afín.** Se demostró transferencia de Kinetics a UCF-101/HMDB-51, que es la *misma* tarea (clasificación de acciones, aunque con clases distintas). Queda por ver si Kinetics ayuda en **otras** tareas de video —segmentación semántica de video, detección de objetos en video o cómputo de flujo óptico—.

## 7. Conexión con la Clase 36 y el linaje del video

Este paper es exactamente el nudo que la Clase 36 (Introduction to Video Analysis) usa para amarrar el tema. La clase recorre el *zoo* de enfoques de deep learning para reconocimiento de acciones y la aparición de Kinetics como el dataset que faltaba; **I3D es la síntesis** de tres líneas que la clase presenta por separado:

1. **Convolución 3D (C3D):** I3D adopta los filtros espacio-temporales que crean jerarquías directamente sobre el video, pero resuelve el defecto histórico de la familia —ser forzosamente poco profunda y entrenada desde cero— vía el inflado.
2. **Two-Stream (RGB + flujo óptico):** I3D conserva la intuición de Simonyan-Zisserman de que un stream de movimiento explícito aporta señal que el RGB no captura fácilmente, y muestra que sigue ayudando incluso sobre una 3D ConvNet profunda.
3. **Pre-entrenamiento y transferencia:** la lección de ImageNet trasladada al video. El *boring-video fixed point* hereda pesos de imagen, y Kinetics provee el pre-entrenamiento de video que se transfiere a los benchmarks pequeños.

La contribución de época no es solo el número: es haber establecido el **paradigma metodológico** que rige el reconocimiento de acciones posterior — *pre-entrenar en Kinetics y transferir*. Al re-evaluar las cinco familias sobre un backbone común y un dataset grande, el paper también le dio al campo el orden que le faltaba: por fin se podía decir, con evidencia, qué arquitectura era mejor. En términos de la clase, I3D cierra la evolución "ConvNet+LSTM → C3D → Two-Stream → 3D-Fused → I3D" y abre la era de las arquitecturas de video pre-entrenadas a gran escala.

---

**Nota final — relevancia para video clínico.** La receta que I3D estandarizó es precisamente la que hace viable el análisis de video médico, donde los datos etiquetados son escasos y caros: rara vez se dispone de cientos de miles de videos anotados de endoscopías, ecocardiografías, estudios de marcha o gestos quirúrgicos. La estrategia de **pre-entrenar en Kinetics y hacer fine-tuning** sobre el dataset clínico pequeño —o incluso congelar la red y entrenar solo un clasificador (el régimen *Fixed* de la Tabla 4)— transfiere la maquinaria de extracción de features espacio-temporales aprendida sobre acciones humanas cotidianas a la tarea médica, recuperando gran parte del desempeño que sería imposible obtener entrenando desde cero con pocos ejemplos. Así como en imágenes médicas el backbone pre-entrenado en ImageNet se volvió el punto de partida por defecto, en video clínico el I3D pre-entrenado en Kinetics se transformó en la línea base natural, y la lección de fondo del paper —que los modelos de video se pre-entrenan mejor sobre videos que sobre imágenes— es directamente aplicable a cualquier pipeline que deba clasificar o localizar eventos dinámicos en registros clínicos con datos limitados.
