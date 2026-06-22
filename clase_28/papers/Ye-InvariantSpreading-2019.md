# Unsupervised Embedding Learning via Invariant and Spreading Instance Feature — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Unsupervised Embedding Learning via Invariant and Spreading Instance Feature*.
- **Autores:** Mang Ye y Pong C. Yuen (Hong Kong Baptist University); Xu Zhang y Shih-Fu Chang (Columbia University). El emparejamiento de instituciones no es casual: Shih-Fu Chang y Xu Zhang venían de trabajar en *spread-out local feature descriptors* (Zhang et al., ICCV 2017), y esa idea de "esparcir" los descriptores es la semilla directa de este paper.
- **Venue:** CVPR 2019 (IEEE/CVF Conference on Computer Vision and Pattern Recognition).
- **Año:** 2019. **Preprint:** arXiv:1904.03436v1 (6 abr 2019), [arxiv.org/abs/1904.03436](https://arxiv.org/abs/1904.03436).
- **Código:** [github.com/mangye16/Unsupervised_Embedding_Learning](https://github.com/mangye16/Unsupervised_Embedding_Learning), implementado en PyTorch.

El paper aborda el problema del **aprendizaje de *embeddings* no supervisado** (*unsupervised embedding learning*, también llamado *unsupervised metric learning*): aprender una función $f_\theta(\cdot)$ que mapee imágenes a un espacio de baja dimensión donde la similitud entre vectores refleje la similitud visual o de categoría, **sin etiquetas humanas**. La distinción que el paper hace desde la primera página es fina pero crucial: el *general unsupervised feature learning* (autoencoders, GANs, pretextos) aprende una buena representación "intermedia" que luego se afina con datos etiquetados para una tarea destino; pero esa representación intermedia "puede no preservar la similitud visual", y su rendimiento se desploma en tareas basadas en similitud directa como la búsqueda por vecino más cercano (kNN). El aprendizaje de *embeddings* exige esa propiedad de similitud directamente en el espacio aprendido.

La tesis central, formulada como una analogía con el aprendizaje supervisado por categorías, es que un buen *embedding* debe satisfacer dos propiedades observadas en el caso supervisado: **(1) concentración positiva** (*positive concentrated*) — los rasgos de muestras de la misma categoría están cerca entre sí; y **(2) separación negativa** (*negative separated*) — los rasgos de categorías distintas están tan separados como sea posible. Sin etiquetas, el paper *aproxima* estas dos propiedades usando **supervisión a nivel de instancia** (*instance-wise supervision*): cada imagen es su propia "clase". El positivo se construye con **aumentación de datos** (la misma instancia, transformada, debe dar un rasgo **invariante**), y los negativos se construyen tratando las **otras instancias del mismo mini-batch** como negativos aproximados, forzando una propiedad de **dispersión** (*spread-out*).

Para el curso (Clase 28, Aprendizaje Autosupervisado) este paper es el **puente conceptual hacia SimCLR/MoCo**. La diapositiva que introduce el aprendizaje contrastivo lo cita literalmente: "La representación de una imagen debe ser más cercana a ella misma transformada (positivo) que a otra imagen distinta (negativo)" (Ye et al., 2019). Es, en una frase, el ADN del aprendizaje contrastivo moderno, formulado un año antes de que SimCLR (Chen et al., 2020) y MoCo (He et al., 2020) lo escalaran a ImageNet.

## 2. Contexto histórico: de la discriminación de instancias al contrastive learning

Entender por qué este paper es un antecedente directo de SimCLR/MoCo requiere situarlo en la cadena de ideas que lo precede.

**Exemplar CNN (Dosovitskiy et al., 2014/2016).** La idea fundacional: tratar **cada imagen como una clase distinta**. Se entrena un clasificador con tantas salidas como imágenes hay en el dataset, y se exige que las versiones aumentadas de una imagen se clasifiquen en su "clase" original. El problema de ingeniería es severo: la matriz de pesos del clasificador $W = [w_1, \dots, w_n]^T \in \mathbb{R}^{n \times d}$ crece linealmente con el número de imágenes (millones de columnas para datasets grandes), y los pesos $w_i$ "previenen la comparación explícita sobre los rasgos", limitando eficiencia y discriminabilidad.

**Non-Parametric Instance Discrimination / NCE (Wu et al., CVPR 2018).** El antecedente más cercano y la línea base más fuerte del paper. Wu et al. eliminan los pesos del clasificador y, en su lugar, montan un ***memory bank*** que almacena el rasgo $v_i$ calculado para cada instancia en el paso anterior. La probabilidad de reconocer $x_j$ como la instancia $i$ se calcula con un softmax sobre las similitudes coseno $v_i^T f_j / \tau$, donde $\tau$ es la **temperatura** que controla la concentración de la distribución. El problema que Ye et al. identifican: el rasgo memorizado $v_i$ **solo se actualiza una vez por época** (en la iteración que toma $x_i$ como entrada), mientras la red se actualiza en cada iteración. Comparar el rasgo en tiempo real $f_i$ con un $v_i$ desactualizado "entorpece el entrenamiento". El *memory bank* sigue siendo ineficiente.

**El callejón sin salida del optimizado directo.** La idea obvia para mejorar la eficiencia sería optimizar directamente sobre el rasgo $f_i$, reemplazando $w_i$ o $v_i$ por $f_i$. Pero el paper muestra que esto es inviable por dos razones: (1) como $f_i^T f_i = 1$ (rasgos $\ell_2$-normalizados), el rasgo y su "pseudo-peso clasificador" (él mismo) están siempre perfectamente alineados, así que optimizar la red **no aporta ninguna propiedad de concentración positiva**; y (2) es impráctico calcular los rasgos de todas las $n$ muestras sobre la marcha para el denominador del softmax. La solución de Ye et al. resuelve ambos problemas a la vez, y es justamente esa solución la que SimCLR reconocería después como el núcleo del contrastive learning.

**Lo que vino después.** SimCLR (Chen et al., 2020) escaló exactamente esta receta —positivo por aumentación, negativos del mismo batch, softmax sobre similitudes coseno con temperatura— añadiendo batches enormes (4096+), una cabeza de proyección no lineal, aumentaciones más agresivas (con *color jitter* fuerte y *blur*) y la pérdida NT-Xent. MoCo (He et al., 2020) atacó la necesidad de muchos negativos con una cola/diccionario y un *momentum encoder*, recuperando la idea del *memory bank* pero manteniéndola consistente. Visto en retrospectiva, este paper de 2019 **formaliza la intuición invariante+spreading** que esos dos trabajos llevaron a escala industrial.

## 3. Contribución central

Las contribuciones que el paper enumera son tres, pero conviene leerlas como una sola idea con sus consecuencias:

1. **Un método de *softmax embedding* basado en el rasgo de instancia "real".** En vez de optimizar sobre pesos del clasificador (Exemplar) o sobre rasgos memorizados (NCE), el método optimiza **directamente los productos internos de los rasgos de instancia reales** sobre la función softmax. Esto produce ganancias significativas de velocidad de aprendizaje y de precisión.

2. **La demostración de que ambas propiedades —invariancia a la aumentación y dispersión de instancias— son importantes** para el aprendizaje de *embeddings* no supervisado a nivel de instancia, y que ambas ayudan a capturar la similitud visual aparente entre muestras y a generalizar a categorías de prueba no vistas.

3. **Estado del arte** sobre otros métodos no supervisados en experimentos comprehensivos de clasificación de imágenes y aprendizaje de *embeddings*.

La idea de diseño que une todo, y que es la aportación conceptual para el curso, es la **dualidad invariante + spreading**:

- **Invariante (positivo).** Las dos vistas aumentadas $x_i$ y $\hat{x}_i = T(x_i)$ de la misma instancia deben producir rasgos cercanos. Maximizar $\exp(f_i^T \hat{f}_i / \tau)$ —dado que los rasgos están $\ell_2$-normalizados— equivale a aumentar la similitud coseno entre $f_i$ y $\hat{f}_i$, resultando en un rasgo **invariante a la aumentación de datos**. Esta es la propiedad que aproxima la *concentración positiva* del caso supervisado.

- **Spreading (negativo).** Como los datos no etiquetados suelen estar muy desbalanceados, el número de negativos de cada instancia es mucho mayor que el de positivos. Por tanto, **un pequeño lote de instancias muestreadas al azar puede tratarse aproximadamente como negativos** de cada instancia. Minimizar $\exp(f_k^T \hat{f}_i / \tau)$ para $k \neq i$ asegura que $\hat{f}_i$ se separe de las otras instancias del lote; considerando todas las instancias del batch, estas se ven forzadas a separarse entre sí, dando la propiedad de **dispersión** (*spread-out*). Esta aproxima la *separación negativa* supervisada.

El paper es honesto sobre el supuesto: tratar todas las demás instancias del batch como negativos "puede no siempre cumplirse, y cada batch puede contener algunos falsos negativos". Pero la evidencia experimental muestra que la propiedad de dispersión mejora efectivamente la discriminabilidad pese a ese ruido. Es exactamente el mismo trade-off que SimCLR aceptaría después: con batches grandes habrá colisiones de clase ocasionales, pero el beneficio neto de los negativos abundantes domina.

## 4. Método: softmax sobre el rasgo de instancia real

### 4.1. Formulación

Sea $X = \{x_1, \dots, x_n\}$ el conjunto de imágenes no etiquetadas, y $f_\theta(\cdot)$ la red de *embedding* que mapea $x_i$ a $f_i \in \mathbb{R}^d$, con todos los rasgos $\ell_2$-normalizados ($\|f_i\|_2 = 1$). En cada iteración se muestrean $m$ instancias del dataset. A cada una se le aplica una aumentación aleatoria $T(\cdot)$, produciendo $\hat{x}_i = T(x_i)$ con rasgo $\hat{f}_i$.

En vez de plantear el aprendizaje del rasgo como una **clasificación multiclase** (con $n$ clases, una por instancia — el enfoque de Exemplar/NCE), el paper lo resuelve como un problema de **clasificación binaria vía estimación de máxima verosimilitud (MLE)**. Para la instancia $x_i$: la muestra aumentada $\hat{x}_i$ *debe* clasificarse en la instancia $i$, y las otras instancias $x_j$ ($j \neq i$) *no* deben clasificarse en la instancia $i$.

La probabilidad de que $\hat{x}_i$ sea reconocida como la instancia $i$ es:

$$P(i \mid \hat{x}_i) = \frac{\exp(f_i^T \hat{f}_i / \tau)}{\sum_{k=1}^{m} \exp(f_k^T \hat{f}_i / \tau)}$$

La probabilidad de que $x_j$ ($j \neq i$) sea reconocida como la instancia $i$ es:

$$P(i \mid x_j) = \frac{\exp(f_i^T f_j / \tau)}{\sum_{k=1}^{m} \exp(f_k^T f_j / \tau)}, \quad j \neq i$$

y la de *no* ser reconocida es $1 - P(i \mid x_j)$.

Asumiendo independencia entre las instancias, la probabilidad conjunta de que $\hat{x}_i$ se reconozca como $i$ **y** que las $x_j$ ($j \neq i$) no se clasifiquen como $i$ es:

$$P_i = P(i \mid \hat{x}_i) \prod_{j \neq i} \big(1 - P(i \mid x_j)\big)$$

La pérdida es la suma de la log-verosimilitud negativa sobre todas las instancias del batch:

$$J = -\sum_i \log P(i \mid \hat{x}_i) - \sum_i \sum_{j \neq i} \log\big(1 - P(i \mid x_j)\big)$$

El primer término empuja la invariancia (alinea cada instancia con su versión aumentada); el segundo empuja la dispersión (separa las instancias entre sí).

### 4.2. Análisis del fundamento (por qué funciona)

El paper dedica una sección ("Rationale Analysis") a mostrar por qué minimizar la pérdida logra ambas propiedades. Maximizar $P(i \mid \hat{x}_i)$ requiere maximizar $\exp(f_i^T \hat{f}_i / \tau)$ —invariancia— y minimizar $\exp(f_k^T \hat{f}_i / \tau)$ para $k \neq i$ —dispersión—. Un detalle técnico esclarecedor aparece al reescribir el segundo término: el denominador de $P(i \mid x_j)$ incluye $\exp(f_j^T f_j / \tau) = \exp(1/\tau)$. Como $\tau$ es pequeño (0.1 en los experimentos), ese término domina el denominador, así que minimizar $P(i \mid x_j)$ se reduce esencialmente a minimizar $\exp(f_i^T f_j / \tau)$, que separa $f_j$ de $f_i$ — reforzando aún más la dispersión.

La función softmax cumple, además, un papel de ***hard negative mining* implícito**: aprovecha las relaciones entre todas las instancias muestreadas y pondera más los negativos difíciles (los más parecidos), sin necesidad de una estrategia explícita de muestreo de tripletas. El paper atribuye a esta naturaleza de *hard mining* del softmax la superioridad del método frente a la pérdida *triplet*.

### 4.3. Entrenamiento con red siamesa

El método se implementa con una **red siamesa** de dos ramas que **comparten pesos**. En cada iteración, $m$ instancias seleccionadas al azar entran por la primera rama y sus versiones aumentadas por la segunda; ambas pasan por el backbone CNN, una capa FC y normalización $\ell_2$ hasta el espacio de *embedding* de baja dimensión. La aumentación se aplica también en la primera rama, para enriquecer las muestras. Para cada muestra, hay **un positivo aumentado** y **$2N - 2$ negativos** (donde $N$ es el tamaño de batch), todos provenientes del propio mini-batch — **sin memory bank**. Esta es la diferencia operativa clave con NCE: los negativos son rasgos *frescos*, calculados en la misma pasada, no rasgos memorizados y desactualizados. El paper nota que teóricamente podría usarse una red multi-rama con varias aumentaciones por instancia (anticipando, de nuevo, la dirección de SimCLR).

## 5. Experimentos

El paper evalúa en dos protocolos. En el primero, **categorías de prueba vistas** (*seen testing categories*): entrenamiento y prueba comparten las mismas categorías — el protocolo clásico de *unsupervised feature learning*. En el segundo, **categorías de prueba no vistas** (*unseen testing categories*): las categorías de prueba no se solapan con las de entrenamiento — el protocolo de *supervised embedding learning*, más exigente, que revela la calidad de los rasgos sobre categorías nuevas.

**Categorías vistas — CIFAR-10 y STL-10.** Con ResNet18, *embedding* de 128 dimensiones, $\tau = 0.1$, y cuatro aumentaciones de PyTorch (RandomResizedCrop, RandomGrayscale, ColorJitter, RandomHorizontalFlip), evaluado con un clasificador kNN ponderado (k=200, similitud coseno). En **CIFAR-10**, el método alcanza **83.6 %** de precisión kNN, superando a Exemplar (74.5 %) por 9.1 puntos, a NPSoftmax (80.8 %) por 2.8, a NCE (80.4 %) por 3.2, y a DeepCluster y *triplet* por márgenes amplios. En **STL-10**, con 5K imágenes de entrenamiento logra 74.1 % kNN / 69.5 % lineal; con 105K imágenes sube a 81.6 % kNN / 77.9 % lineal — evidencia de que el método se beneficia de más datos no etiquetados.

**Eficiencia.** Este es uno de los resultados más vistosos: el método alcanza 60 % de precisión kNN en CIFAR-10 en **solo 2 épocas**, mientras NCE necesita 25 y Exemplar 45 para el mismo nivel. La velocidad se debe a optimizar directamente sobre el rasgo de instancia en vez de sobre pesos del clasificador o un *memory bank* desactualizado.

**Categorías no vistas — CUB200, Stanford Online Product, Car196.** Con backbone Inception-V1 preentrenado en ImageNet y *embedding* de 128-d, evaluado con Recall@K y NMI sobre la similitud coseno. El método es el claro ganador entre los no supervisados (p. ej., en Car196 R@1 = 41.3 % vs. 37.5 % de NCE y 35.5 % de MOM; en Product R@1 = 48.9 %). Notablemente, en CUB200 resulta **competitivo incluso con algunos métodos supervisados**. Los métodos *instance-wise* (NCE, Exemplar, este) superan a los *non-instance-wise* (DeepCluster, MOM) en generalización a categorías no vistas.

**Entrenamiento desde cero.** Sobre Stanford Online Product con ResNet18 **sin preentrenamiento**, el método sigue siendo el ganador (R@1 = 39.7 %). Aquí MOM (Iscen et al., 2018), que depende de minar etiquetas sobre variedades, **fracasa**: los rasgos de una red inicializada al azar no dan información fiable para minar etiquetas. Este resultado subraya una ventaja estructural del enfoque por instancia: no depende de una inicialización de calidad.

**Estudios de ablación.** Dos ablaciones cierran el argumento de las dos propiedades:

- *Invariancia (aumentación).* Quitar la aumentación por completo derrumba la precisión de 83.6 % a **37.4 %** — sin aumentación la red no crea ninguna concentración positiva y separa erróneamente imágenes visualmente similares. Quitando operaciones una a una, **RandomResizedCrop es la que más aporta** (cae a 56.2 %), seguida de ColorJitter (75.7 %); un orden de importancia que SimCLR confirmaría más tarde.
- *Dispersión (elección de negativos).* Usar solo el 50 % de negativos *difíciles* (los más similares al query) mantiene la precisión casi intacta (83.2 %), mientras usar solo el 50 % *fáciles* la desploma a 57.5 %. La separación de los negativos difíciles es lo que mejora la discriminabilidad — corroborando el *hard mining* implícito del softmax.

**Comprensión del *embedding*.** Las distribuciones de similitud coseno entre un query y sus 5 vecinos más cercanos positivos/negativos muestran que el método separa positivos y negativos mejor que NCE, Exemplar y una red aleatoria, y preserva la mejor propiedad de dispersión. Más interesante aún: el rasgo aprendido separa bien **otros atributos** (p. ej., "animales vs. artefactos", "animal de forma grande vs. pequeña") no usados como etiqueta, lo que evidencia la capacidad de generalización del *embedding*.

## 6. Limitaciones

- **El supuesto de negativos es ruidoso.** Tratar todas las demás instancias del batch como negativos introduce **falsos negativos** (instancias de la misma clase semántica tratadas como negativos). El paper lo reconoce y argumenta empíricamente que el beneficio domina, pero el problema de los falsos negativos persiste y motivaría líneas posteriores (p. ej., métodos no contrastivos como BYOL/SimSiam que prescinden de negativos explícitos).
- **Dependencia del tamaño de batch.** Como los negativos vienen del mini-batch, la cantidad y diversidad de negativos está acotada por el batch. El paper usa batches modestos (128 en CIFAR/STL, 64 en fine-grained) y no explora el régimen de batches grandes; sería SimCLR quien mostraría que escalar el batch a miles mejora sustancialmente — y MoCo quien desacoplaría el número de negativos del batch con una cola.
- **Sensibilidad a las aumentaciones.** La ablación deja claro que el rendimiento depende fuertemente de la familia de aumentaciones (sin ellas, colapsa). El paper no realiza una búsqueda exhaustiva del espacio de aumentaciones; la elección de "qué transformaciones definen la invariancia" quedaría como una pregunta central de diseño en el contrastive learning posterior.
- **Backbone y escala.** Los experimentos están en datasets pequeños/medianos (CIFAR, STL, datasets fine-grained) y con backbones modestos (ResNet18, Inception-V1). No hay validación en ImageNet a gran escala con transfer a detección/segmentación — el banco de pruebas que consagraría a MoCo y SimCLR.

## 7. Impacto: la formalización que SimCLR escaló

El valor histórico de este paper no está en una tabla de resultados (que datasets pequeños hacen modesta en retrospectiva), sino en haber **destilado y nombrado** la intuición que se convertiría en el contrastive learning canónico. Tres ideas que aquí ya están completas:

1. **Positivo por aumentación de la misma instancia, sin etiquetas.** La invariancia a aumentaciones como señal de supervisión propia.
2. **Negativos del mismo mini-batch, sin memory bank.** Optimizar sobre rasgos frescos y reales, no memorizados — exactamente el esquema que SimCLR adoptaría (y que MoCo refinaría con su cola para hacerlo escalable sin batches gigantes).
3. **Softmax sobre similitudes coseno con temperatura $\tau$.** La forma funcional de la pérdida —invariante en el numerador, spreading en el denominador— es la columna vertebral de NT-Xent/InfoNCE.

La frase que la Clase 28 cita —que la representación de una imagen debe estar más cerca de sí misma transformada que de otra imagen distinta— es precisamente la propiedad **invariante + spreading** que este paper formaliza. Por eso ocupa el lugar de **antecedente directo** en la narrativa del curso: es el eslabón entre la discriminación de instancias de 2018 (Wu et al., con su memory bank) y la explosión del contrastive learning de 2020 (SimCLR, MoCo). Quien entiende este paper entiende *por qué* SimCLR funciona, antes de ver el truco de escala que lo hizo famoso.

## 8. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

La Clase 28 introduce el aprendizaje contrastivo con la idea-eje de que un buen *embedding* debe acercar una imagen a su versión transformada (positivo) y alejarla de otras imágenes (negativos) — y cita a Ye et al. (2019) justo ahí. Este paper es el referente que **fundamenta esa frase** con una formulación matemática completa: la pérdida con su numerador invariante y su denominador de dispersión, la red siamesa de pesos compartidos, el rol de la temperatura $\tau$, y la evidencia ablativa de que ambas propiedades son necesarias.

En la progresión conceptual del curso, este paper es el **puente** entre la discriminación de instancias con *memory bank* (Wu et al., 2018) y los métodos que la clase desarrolla después — SimCLR (negativos del batch a gran escala, cabeza de proyección, NT-Xent) y MoCo (momentum encoder y cola de negativos). Conviene leerlo junto con el fundamento de [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo), que sistematiza la familia de pérdidas (InfoNCE, NT-Xent, triplet) y el rol de los positivos/negativos; y con el fundamento de [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado), que ubica el contrastive learning dentro del panorama más amplio de pretextos (rotación, jigsaw, inpainting, colorización) que este mismo paper repasa en su sección de trabajo relacionado. El desarrollo completo de estos métodos está en [la Clase 28](/clases/clase-28).

Para un estudiante con trasfondo en *record linkage* y *patient matching*, hay además una lectura transversal directa: la dualidad invariante+spreading es exactamente la geometría que se busca en un *embedding* de pares de registros — acercar las variantes/aumentaciones de una misma entidad (mismo paciente con datos ruidosos o tipográficamente distintos) y separar entidades diferentes. El *hard negative mining* implícito del softmax y la advertencia sobre falsos negativos del batch son, también, problemas que reaparecen tal cual al aprender *embeddings* de similitud sin etiquetas en datos clínicos.
