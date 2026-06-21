# Semi-Supervised Classification with Graph Convolutional Networks — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Semi-Supervised Classification with Graph Convolutional Networks*.
- **Autores:** Thomas N. Kipf (University of Amsterdam) y Max Welling (University of Amsterdam y Canadian Institute for Advanced Research, CIFAR).
- **Venue:** *International Conference on Learning Representations* (ICLR) 2017, publicado como *conference paper*.
- **Año:** 2017. **Preprint:** arXiv:1609.02907v4 (22 feb 2017; primera versión sep 2016), [arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907).
- **Código:** [github.com/tkipf/gcn](https://github.com/tkipf/gcn), implementación en TensorFlow con multiplicaciones dispersas-densas.

Este es uno de los papers fundacionales del aprendizaje sobre grafos. Su tesis es deceptivamente simple: se puede hacer clasificación semi-supervisada de nodos —documentos en una red de citas, entidades en un grafo de conocimiento— con una red neuronal que opera *directamente* sobre el grafo, condicionando cada capa sobre la matriz de adyacencia. La pieza central es una regla de propagación capa-a-capa, $H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$, que el paper deriva no por intuición sino como una **aproximación de primer orden de las convoluciones espectrales sobre grafos**. El modelo escala linealmente en el número de aristas del grafo y aprende representaciones ocultas que codifican simultáneamente la estructura local del grafo y los rasgos (features) de los nodos.

El argumento de fondo es que los métodos previos de aprendizaje semi-supervisado sobre grafos sufren de una limitación conceptual: suponen que las aristas codifican *mera similitud* entre nodos (el supuesto de suavidad del Laplaciano), lo que restringe la capacidad de modelado porque "las aristas no necesariamente codifican similitud de nodos, sino que pueden contener información adicional". Kipf y Welling proponen abandonar la regularización explícita basada en grafos en la función de pérdida y en cambio **codificar la estructura del grafo dentro del propio modelo** $f(X, A)$, entrenando solo sobre los nodos etiquetados pero dejando que el gradiente de la pérdida supervisada se distribuya por el grafo hacia los nodos sin etiqueta.

Para la Clase 27 (Redes Neuronales de Grafos) esto importa porque GCN es el modelo de referencia que la clase usa como punto de entrada al *message passing*: la fórmula del paper, una vez expandida en su forma vectorial por nodo, es exactamente "agregar los features de los vecinos normalizados por grado, transformarlos con una matriz aprendida y pasarlos por una no-linealidad". Es la GNN más citada de la historia y la base sobre la que se construyeron GraphSAGE, GAT, GIN y prácticamente todo lo que vino después.

## 2. Contexto histórico: clasificación semi-supervisada en grafos antes de GCN

El problema que ataca el paper —clasificar nodos de un grafo cuando solo una pequeña fracción tiene etiqueta— se venía formulando como *graph-based semi-supervised learning*. La idea clásica es **suavizar la información de etiquetas a lo largo del grafo** mediante alguna forma de regularización explícita. El paper resume esa tradición con la ecuación que pone en su introducción:

$$\mathcal{L} = \mathcal{L}_0 + \lambda \mathcal{L}_{\text{reg}}, \quad \text{con} \quad \mathcal{L}_{\text{reg}} = \sum_{i,j} A_{ij}\, \lVert f(X_i) - f(X_j)\rVert^2 = f(X)^\top \Delta f(X).$$

Aquí $\mathcal{L}_0$ es la pérdida supervisada sobre la parte etiquetada, $\Delta = D - A$ es el **Laplaciano del grafo no normalizado** ($A$ la matriz de adyacencia, $D_{ii} = \sum_j A_{ij}$ la matriz de grados), y el término $\mathcal{L}_{\text{reg}}$ penaliza que nodos conectados reciban predicciones distintas. Esta formulación —que cubre *label propagation* (Zhu et al., 2003), *learning with local and global consistency* (Zhou et al., 2004), *manifold regularization* (Belkin et al., 2006) y *deep semi-supervised embedding* (Weston et al., 2012)— descansa sobre el supuesto de que **nodos conectados tienden a compartir etiqueta**. El paper lo critica explícitamente: ese supuesto puede limitar la capacidad de modelado, porque una arista podría señalar una relación que *no* es de similitud (un documento puede citar a otro para refutarlo).

La segunda familia previa, más reciente al momento del paper, eran los **métodos de embeddings de grafos** inspirados en el modelo skip-gram de word2vec (Mikolov et al., 2013). **DeepWalk** (Perozzi et al., 2014) aprende embeddings de nodos prediciendo su vecindario local a partir de caminatas aleatorias (*random walks*) sobre el grafo; **LINE** (Tang et al., 2015) y **node2vec** (Grover & Leskovec, 2016) lo extienden con esquemas de caminata o búsqueda en anchura más sofisticados. El problema que Kipf y Welling les señalan es de *ingeniería del pipeline*: todos requieren un proceso multi-etapa —generación de caminatas, luego optimización del embedding, luego un clasificador— donde cada etapa se optimiza por separado, lo que es difícil de afinar conjuntamente. **Planetoid** (Yang et al., 2016) mitigaba esto inyectando información de etiquetas durante el aprendizaje del embedding, y es de hecho el baseline más fuerte y la fuente de los splits experimentales que el paper adopta.

La tercera raíz es la **teoría espectral de grafos** y las redes neuronales sobre grafos previas. Las primeras GNN (Gori et al., 2005; Scarselli et al., 2009) eran redes recurrentes que aplicaban mapas de contracción repetidamente hasta alcanzar un punto fijo estable; Li et al. (2016) modernizaron eso con prácticas de entrenamiento de RNN (las *Gated Graph Neural Networks*, que la Clase 27 contrasta con GCN). Por otra parte, las **convoluciones espectrales** sobre grafos fueron introducidas por Bruna et al. (2014) y aceleradas por **Defferrard et al. (2016)** con filtros localizados rápidos basados en polinomios de Chebyshev (ChebNet). GCN se posiciona precisamente como una **simplificación radical de ChebNet** que sacrifica generalidad espectral a cambio de escalabilidad y desempeño en grafos grandes con distribuciones de grado muy dispersas.

## 3. Contribución central: de la convolución espectral a la regla de propagación

La aportación teórica del paper es mostrar que su regla de propagación, sorprendentemente simple, **se deriva como una aproximación de primer orden de las convoluciones espectrales localizadas sobre grafos** (Hammond et al., 2011; Defferrard et al., 2016). El recorrido tiene cuatro pasos y vale la pena seguirlo porque es lo que distingue a GCN de un agregador de vecinos inventado *ad hoc*.

**Paso 1 — convolución espectral.** Una convolución sobre grafos se define como la multiplicación de una señal $x \in \mathbb{R}^N$ (un escalar por nodo) con un filtro $g_\theta = \text{diag}(\theta)$ en el dominio de Fourier del grafo: $g_\theta \star x = U g_\theta U^\top x$, donde $U$ es la matriz de autovectores del **Laplaciano normalizado** $L = I_N - D^{-\frac{1}{2}} A D^{-\frac{1}{2}} = U \Lambda U^\top$, y $U^\top x$ es la transformada de Fourier de la señal sobre el grafo. El problema: multiplicar por $U$ cuesta $O(N^2)$ y, peor, calcular la descomposición espectral de $L$ es prohibitivo para grafos grandes.

**Paso 2 — aproximación de Chebyshev (ChebNet).** Hammond et al. (2011) mostraron que $g_\theta(\Lambda)$ se puede aproximar bien por una expansión truncada en **polinomios de Chebyshev** $T_k$ hasta orden $K$: $g_{\theta'}(\Lambda) \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{\Lambda})$, con $\tilde{\Lambda} = \frac{2}{\lambda_{\max}}\Lambda - I_N$ y los polinomios definidos recursivamente como $T_k(x) = 2x\,T_{k-1}(x) - T_{k-2}(x)$, $T_0 = 1$, $T_1 = x$. Sustituyendo, la convolución queda $g_{\theta'} \star x \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{L})\, x$, con $\tilde{L} = \frac{2}{\lambda_{\max}} L - I_N$. La clave es que esta expresión es **$K$-localizada**: como es un polinomio de grado $K$ en el Laplaciano, depende solo de nodos a distancia máxima $K$ del nodo central (el vecindario de orden $K$), y evaluarla cuesta $O(|E|)$, lineal en las aristas. Defferrard et al. (2016) usaron exactamente esto para definir su CNN sobre grafos.

**Paso 3 — modelo lineal de primer orden ($K=1$).** Kipf y Welling dan el salto: limitan la convolución a $K=1$, es decir, una función *lineal* respecto al Laplaciano. La intuición declarada es que apilando muchas de estas capas lineales-por-capa se recupera una clase rica de filtros convolucionales sin estar atado a la parametrización explícita de Chebyshev, y que esto puede aliviar el sobreajuste a estructuras de vecindario local en grafos con distribuciones de grado muy anchas (redes sociales, redes de citas, grafos de conocimiento). Aproximando además $\lambda_{\max} \approx 2$ (confiando en que los parámetros de la red se adapten a ese cambio de escala durante el entrenamiento), la convolución se reduce a dos parámetros libres:

$$g_{\theta'} \star x \approx \theta'_0 x - \theta'_1 D^{-\frac{1}{2}} A D^{-\frac{1}{2}} x.$$

**Paso 4 — un solo parámetro y el *renormalization trick*.** Para limitar aún más el número de parámetros (combatir sobreajuste y reducir operaciones por capa), se impone $\theta = \theta'_0 = -\theta'_1$, quedando:

$$g_\theta \star x \approx \theta \left(I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}\right) x.$$

Aquí surge el problema numérico que motiva el truco más conocido del paper. La matriz $I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$ tiene autovalores en el rango $[0, 2]$; aplicarla repetidamente —como ocurre al apilar capas en una red profunda— puede provocar **inestabilidades numéricas y gradientes que explotan o se desvanecen**. El **truco de renormalización** consiste en reemplazar ese operador por una versión equivalente pero estable:

$$I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \;\longrightarrow\; \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}, \quad \text{con} \quad \tilde{A} = A + I_N, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}.$$

Es decir: se añaden **auto-conexiones** ($\tilde{A} = A + I_N$, una arista de cada nodo consigo mismo, que es lo que aporta el término identidad) y se vuelve a normalizar simétricamente con la matriz de grados $\tilde{D}$ del grafo *aumentado*. Generalizando de una señal escalar a una matriz de features $X \in \mathbb{R}^{N \times C}$ con $C$ canales de entrada y $F$ filtros, la operación de convolución de una capa queda:

$$Z = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X \Theta,$$

con $\Theta \in \mathbb{R}^{C \times F}$ la matriz de parámetros del filtro. Esta operación tiene complejidad $O(|E| F C)$, porque $\tilde{A} X$ se implementa eficientemente como producto de una matriz dispersa por una densa. Apilando estas capas con una no-linealidad $\sigma$ se obtiene la regla de propagación del título de la sección 2 del paper.

## 4. Método en detalle: GCN de dos capas para clasificación de nodos

Con la capa de convolución definida, el modelo concreto para clasificación semi-supervisada de nodos es una **GCN de dos capas**. Se precalcula una sola vez, en un paso de preprocesamiento, la matriz normalizada $\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$, y el modelo forward toma la forma compacta:

$$Z = f(X, A) = \text{softmax}\!\left(\hat{A}\;\text{ReLU}\!\left(\hat{A} X W^{(0)}\right) W^{(1)}\right).$$

Las dimensiones son el corazón de la comprensión. $X \in \mathbb{R}^{N \times C}$ es la matriz de features de entrada ($N$ nodos, $C$ canales —en las redes de citas, vectores *bag-of-words* dispersos del documento). $W^{(0)} \in \mathbb{R}^{C \times H}$ es la matriz de pesos entrada-a-oculta, con $H$ feature maps en la capa oculta (en los experimentos, $H = 16$ para las redes de citas y $64$ para NELL). $W^{(1)} \in \mathbb{R}^{H \times F}$ es la matriz oculta-a-salida, con $F$ igual al número de clases. La softmax se aplica fila por fila, produciendo una distribución sobre clases para *cada* nodo. La pérdida es la **entropía cruzada evaluada solo sobre los nodos etiquetados**:

$$\mathcal{L} = -\sum_{l \in \mathcal{Y}_L} \sum_{f=1}^{F} Y_{lf} \ln Z_{lf},$$

donde $\mathcal{Y}_L$ es el conjunto de índices de nodos con etiqueta. Los pesos $W^{(0)}$ y $W^{(1)}$ se entrenan por descenso de gradiente; la estocasticidad se introduce solo vía *dropout*, no por mini-batches.

**Por qué esto es semi-supervisado.** Aquí está la elegancia del diseño. La pérdida solo "ve" los pocos nodos etiquetados (en Cora, 20 etiquetas por clase, ~5% de los nodos). Pero cada capa multiplica por $\hat{A}$, lo que mezcla los features de cada nodo con los de sus vecinos. En una red de dos capas, la predicción de un nodo etiquetado depende de su vecindario de **segundo orden**, que casi con seguridad incluye nodos *sin* etiqueta. Por lo tanto, cuando el gradiente de la pérdida supervisada fluye hacia atrás, **se distribuye por el grafo y actualiza representaciones de nodos sin etiqueta**. El modelo aprende de la estructura del grafo y de los features de *todos* los nodos (etiquetados o no), pero solo necesita la señal de supervisión de unos pocos. Esto es lo que reemplaza a la regularización explícita del Laplaciano de la ecuación clásica: la estructura del grafo no entra como término de penalización en la pérdida, sino que está *horneada en la arquitectura* a través de $\hat{A}$.

El entrenamiento usa **descenso de gradiente por lotes completos (full-batch)**: cada iteración procesa el dataset entero. Esto es viable mientras el grafo quepa en memoria; usando representación dispersa para $A$, el requerimiento de memoria es $O(|E|)$, lineal en las aristas. La complejidad de evaluar el forward es $O(|E| C H F)$. Los autores dejan explícitamente las extensiones con mini-batch SGD para trabajo futuro.

**Interpretación Weisfeiler-Lehman (Apéndice A).** El paper ofrece una segunda lectura de la regla de propagación: es una generalización diferenciable y parametrizada del **algoritmo Weisfeiler-Lehman de 1 dimensión** (WL-1), el clásico test de isomorfismo de grafos. WL-1 actualiza el "color" de cada nodo aplicando una función hash sobre los colores de sus vecinos; si se reemplaza el hash por una capa neuronal diferenciable $h_i^{(l+1)} = \sigma\!\left(\sum_{j \in \mathcal{N}_i} \frac{1}{c_{ij}} h_j^{(l)} W^{(l)}\right)$ y se elige la constante de normalización $c_{ij} = \sqrt{d_i d_j}$ (con $d_i = |\mathcal{N}_i|$ el grado del nodo), **se recupera exactamente la regla de propagación de GCN en forma vectorial por nodo**. Esta es precisamente la forma que la Clase 27 presenta como *message passing*. Como demostración, una GCN de 3 capas con pesos aleatorios y sin entrenar produce, sobre la red del club de karate de Zachary, embeddings de nodos comparables a los de DeepWalk (que usa un entrenamiento no supervisado mucho más caro): la estructura del grafo por sí sola, atravesada por la normalización correcta, ya es un extractor de features potente.

## 5. Experimentos: datasets, resultados y baselines

El paper evalúa GCN en cuatro escenarios: clasificación semi-supervisada de documentos en redes de citas, clasificación de entidades en un grafo de conocimiento, una ablación de los modelos de propagación y un análisis de tiempo de ejecución en grafos aleatorios. Sigue de cerca el protocolo experimental de Yang et al. (2016).

**Datasets.** Tres redes de citas y un grafo de conocimiento (estadísticas del paper, tomadas de Yang et al. 2016):

| Dataset | Tipo | Nodos | Aristas | Clases | Features | Label rate |
|---|---|---|---|---|---|---|
| Citeseer | Red de citas | 3 327 | 4 732 | 6 | 3 703 | 0.036 |
| Cora | Red de citas | 2 708 | 5 429 | 7 | 1 433 | 0.052 |
| Pubmed | Red de citas | 19 717 | 44 338 | 3 | 500 | 0.003 |
| NELL | Grafo de conocimiento | 65 755 | 266 144 | 210 | 5 414 | 0.001 |

En las redes de citas los nodos son documentos con features *bag-of-words* dispersas, y las aristas son enlaces de cita tratados como no dirigidos; se usan solo 20 etiquetas por clase para entrenar pero *todos* los vectores de features. **NELL** es un grafo bipartito extraído de un grafo de conocimiento (Carlson et al., 2010): para cada triple entidad-relación-entidad $(e_1, r, e_2)$ se asignan nodos de relación separados, resultando en 55 864 nodos de relación y 9 891 nodos de entidad, con vectores de features dispersos de 61 278 dimensiones; aquí el escenario es extremo, con **una sola etiqueta por clase** (label rate 0.001).

**Protocolo.** GCN de dos capas, evaluación sobre un test de 1 000 nodos, conjunto de validación de 500 para afinar hiperparámetros (sin usar sus etiquetas para entrenar). Optimizador Adam (lr 0.01), máximo 200 épocas, *early stopping* con ventana de 10, inicialización Glorot, dropout 0.5 y regularización L2 de $5 \cdot 10^{-4}$ en la primera capa (para citas), $H=16$ unidades ocultas. Para NELL: dropout 0.1, L2 de $1 \cdot 10^{-5}$, $H=64$.

**Resultados (accuracy en %, con tiempo wall-clock entre paréntesis).** Tabla 2 del paper:

| Método | Citeseer | Cora | Pubmed | NELL |
|---|---|---|---|---|
| ManiReg | 60.1 | 59.5 | 70.7 | 21.8 |
| SemiEmb | 59.6 | 59.0 | 71.1 | 26.7 |
| LP (label propagation) | 45.3 | 68.0 | 63.0 | 26.5 |
| DeepWalk | 43.2 | 67.2 | 65.3 | 58.1 |
| ICA | 69.1 | 75.1 | 73.9 | 23.1 |
| Planetoid* | 64.7 (26s) | 75.7 (13s) | 77.2 (25s) | 61.9 (185s) |
| **GCN (este paper)** | **70.3 (7s)** | **81.5 (4s)** | **79.0 (38s)** | **66.0 (48s)** |
| GCN (splits aleatorios) | 67.9 ± 0.5 | 80.1 ± 0.5 | 78.9 ± 0.7 | 58.4 ± 1.7 |

GCN supera a todos los baselines en los cuatro datasets, y por un margen amplio en Cora (81.5 vs. 75.7 de Planetoid) y NELL (66.0 vs. 61.9). Además es notablemente más rápido: 4 s en Cora frente a los 13 s de Planetoid, y 48 s en NELL frente a 185 s. La fila de *splits aleatorios* (media ± error estándar sobre 10 particiones) muestra que el desempeño es robusto y no un artefacto de la partición particular de Yang et al.

**Ablación del modelo de propagación (Tabla 3).** El experimento más instructivo compara variantes de la capa de propagación, manteniendo todo lo demás igual. El filtro de Chebyshev completo ($K=3$: 79.5 en Cora; $K=2$: 81.2) no mejora al truco de renormalización (81.5). El modelo de primer orden de dos parámetros (Eq. 6) da 80.0 en Cora; el de un solo parámetro $(I_N + D^{-1/2}AD^{-1/2})X\Theta$ da 79.2; y el **truco de renormalización $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X\Theta$ da el mejor resultado en los tres datasets** (70.3 / 81.5 / 79.0). Un perceptrón multicapa que ignora el grafo por completo ($X\Theta$) se desploma a 46.5 / 55.1 / 71.4, cuantificando cuánto aporta la estructura del grafo. Esto valida empíricamente que la simplificación —menos parámetros, más estabilidad— no solo es más barata sino *mejor*.

**Tiempo por época (grafos aleatorios).** En grafos sintéticos con $2N$ aristas, el tiempo de entrenamiento por época crece linealmente con el número de aristas, confirmando la complejidad $O(|E|)$; en GPU el modelo escala hasta millones de aristas antes de quedarse sin memoria (marcado con un asterisco de *out-of-memory*), punto a partir del cual el entrenamiento en CPU sigue siendo viable.

## 6. Limitaciones reconocidas

El paper es honesto sobre tres límites, todos los cuales motivaron líneas de investigación posteriores enteras.

**Naturaleza transductiva (no inductiva).** GCN aprende sobre un grafo fijo y completo: precalcula $\hat{A}$ una vez con todos los nodos presentes. No puede generar embeddings para nodos nuevos que aparezcan después del entrenamiento sin reentrenar, porque no aprende una función de agregación reutilizable independiente del grafo concreto. Esta es precisamente la limitación que **GraphSAGE** (Hamilton et al., 2017) atacaría poco después, aprendiendo funciones de agregación sobre vecindarios muestreados para generalizar a nodos no vistos —el contraste *transductivo vs. inductivo* que la Clase 27 destaca.

**Requerimiento de memoria (full-batch).** Con descenso de gradiente por lotes completos, la memoria crece linealmente con el tamaño del dataset, y para grafos grandes que no caben en GPU hay que recurrir a CPU. El paper anticipa que el mini-batch SGD aliviaría esto, pero advierte que generar mini-batches debe tener en cuenta el número de capas: para una GCN de $K$ capas hay que almacenar el vecindario de orden $K$ de cada nodo, lo que explota en grafos densos. Esta tensión motivó el muestreo de vecindario (GraphSAGE) y métodos de muestreo por capa posteriores.

**Sobre-suavizado (*over-smoothing*) con muchas capas.** Aunque el paper no usa el término moderno "over-smoothing", lo documenta experimentalmente en el Apéndice B: los mejores resultados se obtienen con modelos de **2 o 3 capas**, y a partir de ~7 capas el entrenamiento se vuelve difícil sin conexiones residuales, porque el tamaño del contexto efectivo de cada nodo crece con su vecindario de orden $K$ en cada capa adicional —es decir, las representaciones de todos los nodos tienden a colapsar hacia un valor común a medida que se promedian vecindarios cada vez más grandes. Los autores añaden conexiones residuales estilo ResNet ($H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)}) + H^{(l)}$) para mitigarlo, anticipando todo un subcampo dedicado a hacer GNN profundas.

**Otras limitaciones.** El modelo solo soporta grafos no dirigidos y no maneja naturalmente features de arista (aunque NELL muestra que se pueden codificar relaciones dirigidas convirtiéndolas en nodos bipartitos). Y el truco de renormalización asume **igual importancia entre la auto-conexión y las aristas a los vecinos**; el paper sugiere un parámetro de compensación aprendible $\tilde{A} = A + \lambda I_N$ como extensión —idea que prefigura los pesos de atención de **GAT** (Velickovic et al., 2018), donde la importancia de cada vecino se aprende en vez de fijarse por grado.

## 7. Impacto

GCN se convirtió en el paper de redes neuronales de grafos **más citado de la historia** y en la base de casi todo lo que vino después. Su contribución no fue solo un modelo con buen desempeño, sino una *plantilla mental*: la idea de que una capa de GNN es "agregar mensajes de los vecinos, transformar, no-linealidad", expresable como una multiplicación matricial dispersa, abrió la puerta al marco de *message passing neural networks* (Gilmer et al., 2017) bajo el que se unifican casi todas las GNN. GraphSAGE generalizó la agregación al caso inductivo; GAT reemplazó la normalización fija por grado por atención aprendida; GIN (Xu et al., 2019) analizó la expresividad de GCN a través de su conexión con Weisfeiler-Lehman (que el propio paper ya había señalado en su Apéndice A). El truco de renormalización con auto-conexiones se volvió un estándar de facto. La accesibilidad de la formulación —y la implementación pública en TensorFlow— hizo que GCN fuera el modelo con el que la comunidad aprendió a pensar sobre grafos.

## 8. Conexión con la Clase 27 (Redes Neuronales de Grafos)

La Clase 27 presenta GCN explícitamente como modelo fundacional (slide "Kipf & Welling 2017") y lo usa como puerta de entrada al *message passing*, que es el hilo conductor de toda la sesión. Mapeo directo entre el paper y la clase:

- **La fórmula de la clase es la del paper, en forma vectorial por nodo.** La clase escribe la actualización como $h_t = \sigma\!\left(\frac{1}{\#\text{neigh}+1}\, W_t (h_{t-1} + h')\right)$, es decir, el nuevo estado de un nodo se obtiene promediando su propio estado anterior $h_{t-1}$ con los de sus vecinos $h'$, normalizando por el número de vecinos más uno (el "+1" es la auto-conexión, exactamente el $\tilde{A} = A + I_N$ del paper), transformando con la matriz aprendida $W_t$ y aplicando una no-linealidad. Esto es precisamente la forma Weisfeiler-Lehman del Apéndice A del paper, $h_i^{(l+1)} = \sigma\!\left(\sum_{j \in \mathcal{N}_i} \frac{1}{c_{ij}} h_j^{(l)} W^{(l)}\right)$. La clase usa una normalización por grado simétrica simplificada (dividir por $\#\text{neigh}+1$) que es la versión intuitiva del término $\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$ del paper; la diferencia es solo si la normalización es por grado del nodo receptor o por la media geométrica $\sqrt{d_i d_j}$ de ambos extremos de la arista.

- **El mensaje es el feature del vecino normalizado por grado.** La clase explica GCN como un caso de *message passing* donde el "mensaje" que viaja por cada arista es simplemente el vector de features del vecino, escalado por la normalización de grado, y la "agregación" es una suma (que con la normalización se vuelve un promedio). No hay función de mensaje aprendida aparte: el peso $W$ se aplica tras agregar. Esto es lo que hace a GCN el message passing *más simple posible* y, por eso, el punto de partida pedagógico ideal.

- **Contraste con GGNN y GraphSAGE.** La clase posiciona a GCN dentro de un espectro de variantes de message passing. **GGNN** (Gated Graph Neural Networks, Li et al. 2016 —citado en el related work del paper como el trabajo que modernizó las GNN recurrentes) usa una **GRU como función de actualización**: en vez de promediar y aplicar $W$, agrega los mensajes y los combina con el estado anterior mediante las compuertas de una GRU, lo que permite muchas iteraciones de propagación sin sobre-suavizar tan rápido. **GraphSAGE** (Hamilton et al. 2017) introduce el **muestreo de vecindario** (en vez de agregar *todos* los vecinos, muestrea un subconjunto de tamaño fijo) y agregadores alternativos (mean, LSTM, pooling), lo que lo hace inductivo y escalable a grafos enormes —resolviendo justo las limitaciones transductiva y de memoria que el paper de GCN reconocía. Así, la clase usa la tríada **GCN (promedio normalizado por grado) → GGNN (update con GRU) → GraphSAGE (sampling + agregadores)** para mostrar que todas son instancias del mismo esqueleto de message passing que difieren en *cómo agregan* y *cómo actualizan*, siendo GCN la forma canónica y mínima de la que parte todo el resto.
