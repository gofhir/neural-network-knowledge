# Inductive Representation Learning on Large Graphs (GraphSAGE) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Inductive Representation Learning on Large Graphs*. El método propuesto se llama **GraphSAGE** (de **SA**mple and aggre**GAtE**).
- **Autores:** William L. Hamilton, Rex Ying y Jure Leskovec (Department of Computer Science, Stanford University). Los dos primeros autores hicieron contribuciones equivalentes.
- **Venue:** *31st Conference on Neural Information Processing Systems* (**NeurIPS / NIPS 2017**), Long Beach, CA, USA.
- **Año:** 2017. **Preprint:** arXiv:1706.02216 (la versión consultada es la v4, 10 de septiembre de 2018, posterior a la *camera-ready*).
- **Código y datos:** [snap.stanford.edu/graphsage](http://snap.stanford.edu/graphsage/). Datasets Reddit y PPI públicos; Web of Science licenciado por Thomson Reuters.

GraphSAGE responde a una pregunta concreta y de enorme valor industrial: ¿cómo generamos un *embedding* para un nodo que **no estaba presente durante el entrenamiento**? Hasta 2017, casi todos los métodos de *node embedding* exitosos (DeepWalk, node2vec, LINE, factorizaciones espectrales) eran **transductivos**: optimizaban directamente un vector por nodo sobre un grafo único y fijo, de modo que un nodo nuevo —un post recién publicado en Reddit, un video subido a YouTube, una proteína de un organismo no estudiado— no tenía representación sin reentrenar. El propio GCN de Kipf & Welling (2016), pese a usar convoluciones sobre el grafo, había sido aplicado solo en el régimen transductivo y, en su formulación exacta, requería conocer el **Laplaciano del grafo completo** durante el entrenamiento.

La tesis central del paper es que en vez de aprender *un embedding por nodo* hay que aprender **una función** que, dado un nodo y sus features, *genera* el embedding agregando información de su vecindario local. Como la función —no el embedding— es lo que se aprende, el modelo es **inductivo**: una vez entrenado, se aplica a nodos o grafos completamente nuevos sin reentrenar. El paper demuestra esto en tres benchmarks (citas Web of Science, Reddit, e interacciones proteína-proteína), mejorando el F1 de clasificación en un **51% en promedio** frente a usar solo features, y superando consistentemente a una *baseline* transductiva fuerte (DeepWalk) que además es ~100× más lenta al inferir sobre nodos no vistos. Los agregadores entrenables nuevos aportan un **7.4%** de ganancia promedio sobre un agregador estilo GCN.

Para la Clase 27 (Redes Neuronales de Grafos) esto importa porque GraphSAGE es el paso de bisagra entre el GCN transductivo y full-batch y la generación de GNN escalables e inductivas que se usan en producción. La clase lo presenta directamente con la fórmula $h^t = \sigma(W^t \cdot \text{concat}(h^{t-1}, h'))$ y el mecanismo de muestreo de vecinos para eficiencia.

## 2. Contexto histórico: el problema transductivo y la necesidad de inductividad

Un **embedding de nodo** comprime la información de alta dimensión del vecindario de un nodo en un vector denso de baja dimensión, que luego alimenta tareas *downstream*: clasificación de nodos, *clustering*, predicción de enlaces. La familia dominante hacia 2016–2017 aprendía esos embeddings con objetivos basados en *random walks* y factorización de matrices (DeepWalk, node2vec, LINE, GraRep, SDNE), emparentados con el *spectral clustering*, el *multidimensional scaling* y PageRank.

El problema estructural de toda esta familia es que **entrena un vector por cada nodo individual** mediante un *embedding look-up*. Tres consecuencias:

1. **Es inherentemente transductiva.** Las predicciones se hacen sobre los nodos de un único grafo fijo. Un nodo nuevo simplemente no tiene fila en la tabla de embeddings. Adaptar estos métodos al régimen inductivo exige rondas adicionales y caras de descenso de gradiente antes de poder predecir sobre el nodo nuevo (el paper mide que DeepWalk en modo "online" es 100–500× más lento al inferir por esto mismo).
2. **El espacio de embeddings no se alinea entre grafos.** Para muchos de estos objetivos (DeepWalk, node2vec, LINE, SDNE) la función de pérdida es invariante a transformaciones ortogonales: $ZQ^\top QZ^\top = ZZ^\top$ para cualquier matriz ortogonal $Q$. El espacio entero puede rotar arbitrariamente durante el entrenamiento. Entrenado sobre un grafo A, los embeddings de un grafo B caen en un espacio rotado al azar respecto a A, y un clasificador entrenado en A produce salidas esencialmente aleatorias sobre B. Esto hace imposible **generalizar entre grafos** sin un procedimiento de alineación explícito (el paper dedica el Apéndice D a este punto, y es la razón de no poder aplicar DeepWalk al benchmark multi-grafo PPI).

El régimen **inductivo** es más difícil que el transductivo precisamente porque generalizar a nodos no vistos exige "alinear" subgrafos recién observados con la estructura de embeddings ya optimizada; el modelo debe aprender a reconocer propiedades estructurales del vecindario que revelan tanto el rol local del nodo como su posición global. Y es justamente la capacidad que necesitan los sistemas de ML de producción de alto *throughput*, que operan sobre grafos en evolución y encuentran nodos nuevos sin parar (posts de Reddit, usuarios y videos de YouTube). La inductividad también habilita la **generalización entre grafos** de la misma forma: entrenar un generador de embeddings sobre grafos de interacción proteína-proteína de un organismo modelo y producir embeddings sobre datos de organismos nuevos con el modelo ya entrenado.

El GCN de Kipf & Welling (2016) había introducido convoluciones sobre grafos como metodología de embedding prometedora, pero hasta ese momento solo se había aplicado en el régimen **transductivo con grafos fijos**, y su algoritmo exacto requiere conocer el Laplaciano completo del grafo en entrenamiento. GraphSAGE se posiciona explícitamente como la **extensión del marco GCN al régimen inductivo**, generalizándolo para usar **funciones de agregación entrenables** (más allá de la convolución simple). Una excepción previa es Planetoid-I (Yang et al., 2016), inductivo y basado en embeddings, pero que no usa la estructura del grafo durante la inferencia (la usa solo como regularización en entrenamiento) — a diferencia de GraphSAGE, que sí explota el vecindario al inferir.

## 3. Contribución central: SAmple and aggreGatE

La idea central tiene tres componentes que, juntos, producen un modelo inductivo y escalable:

**(a) Aprender funciones de agregación, no embeddings por nodo.** En vez de entrenar un vector distinto por nodo, GraphSAGE entrena un conjunto de **funciones agregadoras** que aprenden a combinar información de features del vecindario local de un nodo. Cada función agrega información desde un número distinto de saltos (*hops*) o profundidad de búsqueda. Al aprovechar las **features de los nodos** (atributos de texto, perfil, grados), el modelo aprende simultáneamente la estructura topológica del vecindario *y* la distribución de features en él. En inferencia, se aplican las funciones agregadoras aprendidas para generar embeddings de nodos completamente nuevos. Como las features son el insumo (no una identidad memorizada), el modelo se transfiere a nodos y grafos nuevos. Si un grafo carece de features ricas, se pueden usar features estructurales presentes en todo grafo (grados de nodo).

**(b) Muestreo de vecindario de tamaño fijo para escalar.** En vez de usar el vecindario completo de cada nodo —cuyo tamaño y costo son impredecibles, en el peor caso $O(|V|)$ por *batch*—, GraphSAGE muestrea uniformemente un **conjunto de vecinos de tamaño fijo** en cada iteración. Esto fija el costo de espacio y tiempo por *batch* en $O(\prod_{i=1}^{K} S_i)$, con $S_i$ y $K$ constantes elegidas por el usuario. En la práctica, $K=2$ y $S_1 \cdot S_2 \le 500$ ya dan alto rendimiento.

**(c) Agregadores entrenables.** Como los vecinos de un nodo **no tienen orden natural**, la función agregadora idealmente debe ser **simétrica** (invariante a permutaciones de su entrada) y a la vez entrenable y expresiva. El paper examina tres:
- **Mean** (media elemento a elemento): casi equivalente a la regla de propagación convolucional del GCN.
- **LSTM**: mayor capacidad expresiva, pero *no* es simétrica (procesa secuencialmente); se adapta aplicándola a una **permutación aleatoria** de los vecinos.
- **Pooling** (max-pooling): cada vector de vecino pasa por una red *fully-connected* y luego se aplica un *max* elemento a elemento; simétrica y entrenable.

**(d) Pérdida no supervisada basada en grafo, o supervisada.** Siguiendo el trabajo previo en embeddings, GraphSAGE puede entrenarse sin supervisión específica de tarea con una pérdida basada en *random walks* que empuja a nodos cercanos a tener representaciones similares y a nodos dispares a tener representaciones distintas. Alternativamente, la pérdida no supervisada se reemplaza o complementa por un objetivo supervisado (cross-entropy de clasificación).

## 4. El método en detalle

### 4.1. Algoritmo de *forward* (generación de embeddings)

El Algoritmo 1 asume parámetros ya entrenados: $K$ funciones agregadoras $\text{AGGREGATE}_k$, matrices de pesos $W_k$ (que propagan información entre "profundidades de búsqueda"), una no linealidad $\sigma$ y una función de vecindario $N: v \to 2^V$. La intuición: en cada iteración $k$ (cada salto), los nodos agregan información de sus vecinos inmediatos, y al iterar van incorporando información de regiones cada vez más lejanas del grafo.

El bucle, para cada profundidad $k = 1 \ldots K$ y cada nodo $v$:

1. **Caso base:** $h^0_v \leftarrow x_v$, las features de entrada.
2. **Agregar el vecindario:** se agregan las representaciones de los vecinos del paso anterior en un solo vector:
   $$h^k_{N(v)} \leftarrow \text{AGGREGATE}_k\big(\{\, h^{k-1}_u, \forall u \in N(v) \,\}\big).$$
3. **Concatenar con la propia y proyectar:** se concatena la representación actual del nodo $h^{k-1}_v$ con el vector agregado del vecindario $h^k_{N(v)}$, se pasa por una capa lineal con no linealidad:
   $$h^k_v \leftarrow \sigma\big(W^k \cdot \text{CONCAT}(h^{k-1}_v,\, h^k_{N(v)})\big).$$
   Esta es exactamente la forma que la Clase 27 escribe como $h^t = \sigma(W^t \cdot \text{concat}(h^{t-1}, h'))$, donde $h'$ es el vector agregado del vecindario (la clase lo deja "libre", p. ej. el promedio).
4. **Normalizar L2:** $h^k_v \leftarrow h^k_v / \lVert h^k_v \rVert_2$, lo que evita que las normas crezcan sin control entre capas.
5. **Salida:** $z_v \equiv h^K_v$, la representación final tras $K$ saltos.

La **concatenación** del paso 3 es importante. Funciona como una especie de *skip connection* entre profundidades de búsqueda: la representación previa del nodo se preserva junto a la información del vecindario, y el paper reporta que esto produce ganancias significativas de rendimiento. (El agregador "convolucional" estilo GCN es justamente la variante que *no* concatena, sino que mezcla el nodo y sus vecinos en una sola media: $h^k_v \leftarrow \sigma(W \cdot \text{MEAN}(\{h^{k-1}_v\} \cup \{h^{k-1}_u, \forall u \in N(v)\}))$.)

### 4.2. Dimensiones y muestreo

El vecindario $N(v)$ se redefine como un **muestreo uniforme de tamaño fijo** del conjunto $\{u \in V : (u,v) \in E\}$, con muestras distintas e independientes en cada iteración $k$. En el régimen *minibatch* (Algoritmo 2 del Apéndice A), el proceso de muestreo es conceptualmente **inverso** al bucle de agregación: se parte de los nodos objetivo del *batch* $B$ ("capa $K$"), se muestrean sus vecinos ("capa $K-1$"), y así sucesivamente, calculando solo las representaciones estrictamente necesarias para satisfacer la recursión. Con $K=2$ y tamaños $S_1, S_2$, cada nodo objetivo muestrea $S_2$ vecinos inmediatos y $S_1 \cdot S_2$ vecinos a 2 saltos. En los experimentos se fijó la dimensión de salida $h^k$ en 256 en toda profundidad, con $S_1=25$ y $S_2=10$.

### 4.3. Aprendizaje de parámetros

La **pérdida no supervisada basada en grafo** sobre las representaciones de salida $z_u$ es:
$$J_G(z_u) = -\log\big(\sigma(z_u^\top z_v)\big) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)}\big[\log\big(\sigma(-z_u^\top z_{v_n})\big)\big],$$
donde $v$ es un nodo que co-ocurre cerca de $u$ en un *random walk* de longitud fija, $\sigma$ es la sigmoide, $P_n$ es la distribución de *negative sampling* y $Q$ el número de negativos. Lo crucial frente a métodos previos: las representaciones $z_u$ que entran en la pérdida se **generan a partir de las features del vecindario local**, no de un look-up por nodo. Los parámetros ($W_k$ y los de los agregadores) se ajustan con SGD y *backpropagation*. En entrenamiento se corrieron 50 *random walks* de longitud 5 por nodo y 20 negativos.

### 4.4. Relación con el test de Weisfeiler-Lehman

GraphSAGE está conceptualmente inspirado en el test de isomorfismo de **Weisfeiler-Lehman** (WL, "refinamiento ingenuo de vértices"). Si en el Algoritmo 1 se fija $K=|V|$, las matrices de pesos a la identidad y se usa una función de *hash* como agregador (sin no linealidad), el algoritmo *es* una instancia del test WL. GraphSAGE es entonces una **aproximación continua del test WL**, reemplazando el *hash* por agregadores neuronales entrenables. Esta conexión da contexto teórico al diseño: el algoritmo aprende estructura topológica del vecindario.

## 5. Experimentos

Tres benchmarks, todos **inductivos** (predicción sobre nodos no vistos en entrenamiento; en PPI, además, sobre grafos enteros no vistos). *Baselines*: clasificador aleatorio, regresión logística sobre features (ignora el grafo), DeepWalk (representante de factorización), y la concatenación de features crudas con embeddings de DeepWalk. Variantes de GraphSAGE: GCN, mean, LSTM y pool, cada una en versión no supervisada y supervisada. Se usó ReLU, $K=2$, $S_1=25$, $S_2=10$, Adam, *batch* 512, implementación en TensorFlow.

**Citation (Web of Science).** Grafo de citas no dirigido del Web of Science Core Collection de Thomson Reuters: todos los papers de seis campos de biología, años 2000–2005, 302.424 nodos, grado medio 9.15. Tarea: predecir el campo (6 etiquetas). Entrenamiento con datos 2000–2004, prueba sobre 2005 (grafo en evolución). Features: grados de nodo y embeddings de los *abstracts* (enfoque de Arora et al., word2vec de 300 dimensiones vía GenSim).

**Reddit.** Posts de septiembre 2014. Se conectan dos posts si un mismo usuario comenta en ambos; la etiqueta es el *subreddit* (50 comunidades grandes). 232.965 posts, grado medio 492. Entrenamiento con los primeros 20 días, prueba con el resto. Features: GloVe CommonCrawl de 300 dim (promedio del título, promedio de los comentarios, *score* y número de comentarios).

**PPI (protein-protein interaction).** Generalización **entre grafos**: clasificar roles de proteínas (funciones celulares de gene ontology, 121 etiquetas) en grafos de interacción proteína-proteína, cada grafo un tejido humano distinto. Grafo promedio: 2373 nodos, grado medio 28.8. Entrenamiento sobre 20 grafos, prueba sobre 2 grafos **enteramente no vistos** (2 más para validación). Esta tarea requiere aprender *roles* de nodo, no estructura de comunidad, y el 42% de los nodos no tiene features no nulas, lo que vuelve crítica la información del vecindario. DeepWalk no puede aplicarse aquí por el problema de invarianza ortogonal entre grafos disjuntos.

**Resultados (F1 micro-promediado, Tabla 1):**

| Método | Citation Unsup. | Citation Sup. | Reddit Unsup. | Reddit Sup. | PPI Unsup. | PPI Sup. |
|---|---|---|---|---|---|---|
| Random | 0.206 | 0.206 | 0.043 | 0.042 | 0.396 | 0.396 |
| Raw features | 0.575 | 0.575 | 0.585 | 0.585 | 0.422 | 0.422 |
| DeepWalk | 0.565 | 0.565 | 0.324 | 0.324 | — | — |
| DeepWalk + features | 0.701 | 0.701 | 0.691 | 0.691 | — | — |
| GraphSAGE-GCN | 0.742 | 0.772 | 0.908 | 0.930 | 0.465 | 0.500 |
| GraphSAGE-mean | 0.778 | 0.820 | 0.897 | 0.950 | 0.486 | 0.598 |
| GraphSAGE-LSTM | 0.788 | 0.832 | 0.907 | 0.954 | 0.482 | 0.612 |
| GraphSAGE-pool | 0.798 | 0.839 | 0.892 | 0.948 | 0.502 | 0.600 |

GraphSAGE supera a todas las *baselines* por un margen amplio, y los agregadores neuronales entrenables (mean/LSTM/pool) ganan sobre el estilo GCN. La variante no supervisada GraphSAGE-pool supera a "DeepWalk + features" en 13.8% (citas) y 29.1% (Reddit); la supervisada en 19.7% y 37.2%. Llamativamente, el agregador LSTM rinde fuerte pese a estar diseñado para secuencias y no para conjuntos. El rendimiento no supervisado es razonablemente competitivo con el supervisado, lo que indica que el marco produce embeddings útiles sin *fine-tuning* específico de tarea.

**Tiempos y sensibilidad (Sección 4.3).** Los tiempos de entrenamiento son comparables (GraphSAGE-LSTM el más lento), pero DeepWalk es **100–500× más lento al inferir** porque necesita nuevas rondas de *random walks* y SGD para embeber nodos nuevos. Pasar de $K=1$ a $K=2$ da +10–15% de F1; ir más allá de $K=2$ aporta apenas 0–5% pero multiplica el costo por 10–100×. Muestrear vecindarios grandes da retornos decrecientes (Figura 2.B): pese a la varianza extra del submuestreo, GraphSAGE mantiene la precisión con gran mejora de *runtime*. Un test de Wilcoxon sobre las 6 configuraciones muestra que mean/LSTM/pool baten significativamente al GCN ($p=0.02$); LSTM y pool no difieren entre sí, pero pool es ~2× más rápido que LSTM, dándole una ligera ventaja global.

## 6. Análisis teórico

El paper prueba (Teorema 1) que GraphSAGE puede aprender estructura del grafo pese a basarse en features. Como caso de estudio toma el **coeficiente de clustering** de un nodo (proporción de triángulos cerrados en su vecindario de 1 salto). Si las features de los nodos son todas distintas ($\lVert x_v - x_{v'} \rVert_2 > C$ para todo par) y el modelo es suficientemente dimensional, entonces para todo $\epsilon > 0$ existe un ajuste de parámetros tal que tras $K=4$ iteraciones $|z_v - c_v| < \epsilon$ para todo nodo, donde $c_v$ es el coeficiente de clustering. La intuición de la prueba: con features únicas el modelo puede mapear nodos a vectores indicadores e identificar vecindarios. La prueba se apoya en propiedades del agregador **pooling**, lo que también da indicio de por qué GraphSAGE-pool supera a GCN y mean.

## 7. Limitaciones

- **Varianza del muestreo.** Submuestrear el vecindario introduce varianza en las representaciones y los gradientes; el paper lo asume como compromiso aceptable a cambio del *runtime* fijo, y muestra empíricamente que la precisión se mantiene, pero la varianza existe.
- **El vecindario fijo pierde información.** Tomar un tamaño fijo de vecinos (y, en el preprocesado, hacer *downsampling* de aristas para que ningún nodo supere grado 128) descarta parte de la estructura de nodos de alto grado. El muestreo es **uniforme**, no informado; el propio paper marca explorar *samplers* no uniformes (y aprenderlos) como dirección de trabajo futuro.
- **Profundidad limitada en la práctica.** Más allá de $K=2$ los retornos son marginales y el costo explota, así que GraphSAGE captura sobre todo vecindarios de 2 saltos.
- **Grafos no dirigidos / unimodales.** Las extensiones a grafos dirigidos o multimodales quedan como trabajo futuro.

## 8. Impacto

GraphSAGE se volvió una de las piezas fundacionales de las GNN inductivas y escalables. Su descendiente directo más célebre es **PinSage** (Ying et al., 2018), el sistema de recomendación de Pinterest construido sobre el mismo principio de *sample-and-aggregate* y llevado a un grafo de miles de millones de nodos — el primer despliegue de una GNN a escala web real, y un ejemplo canónico de por qué la inductividad importa en producción. El marco también encaja en la formulación general de *message passing* (MPNN) y es una de las arquitecturas de referencia en librerías como PyTorch Geometric y DGL. Trabajo de seguimiento (Chen & Zhu, 2017) mejoró los números de PPI optimizando hiperparámetros y agregando *dropout*, *layer normalization* y nuevos esquemas de muestreo, lo que muestra que el marco GraphSAGE siguió siendo base de mejoras posteriores.

## 9. Conexión con la Clase 27 (Redes Neuronales de Grafos)

La Clase 27 presenta GraphSAGE en la *slide* atribuida a Hamilton 2017 con exactamente la mecánica de este paper, y conviene mapear pieza por pieza:

- **La fórmula de la clase es el paso 3 del Algoritmo 1.** La clase escribe $h^t = \sigma(W^t \cdot \text{concat}(h^{t-1}, h'))$ más normalización, dejando $h'$ "libre" (por ejemplo, el promedio del vecindario). Aquí $h'$ es el $h^k_{N(v)}$ del paper —el resultado del agregador— y "promedio" es el agregador **mean**. La concatenación de $h^{t-1}$ (la representación propia previa) con $h'$ y la posterior **normalización L2** son tal cual los pasos 3 y 4 del algoritmo. Que la clase deje $h'$ libre refleja el diseño modular del paper: el agregador es un *placeholder* intercambiable (mean, LSTM, pool).

- **El muestreo de vecinos para eficiencia** que la clase menciona es la sección 3.1 / Apéndice A: vecindario de tamaño fijo muestreado uniformemente, costo por *batch* acotado en $O(\prod_i S_i)$, con $K=2$ y $S_1 \cdot S_2 \le 500$ como receta práctica. Es el ingrediente que vuelve a GraphSAGE escalable a grafos grandes y la diferencia clave frente al GCN full-batch.

- **Contraste con GCN.** La clase contrasta GraphSAGE con el GCN (transductivo, *full-batch*). El paper lo formaliza: el GCN exacto necesita el Laplaciano del grafo completo en entrenamiento y se aplicó solo en el régimen transductivo, mientras GraphSAGE *generaliza* el GCN al régimen inductivo con agregadores entrenables. De hecho, GraphSAGE-GCN (la variante "convolucional" sin concatenación, que mezcla nodo y vecinos en una media) es precisamente la versión inductiva del GCN, y en la Tabla 1 queda por debajo de las variantes con concatenación y agregadores neuronales: la lección es que **concatenar la representación propia y usar agregadores entrenables ayuda**.

- **Contraste con GGNN.** La clase también lo sitúa frente a las *Gated Graph Neural Networks* (Li et al., 2015, referencia [21] del paper). Las GGNN usan unidades recurrentes con compuertas (estilo GRU) para propagar mensajes muchos pasos y suelen orientarse a clasificar grafos o subgrafos enteros; GraphSAGE, en cambio, agrega un número pequeño y fijo de saltos ($K\approx2$) con agregadores simétricos y se enfoca en generar representaciones **por nodo** de forma inductiva y escalable. La diferencia de profundidad y de objetivo (nodo vs. grafo, propagación recurrente larga vs. agregación de pocos saltos con muestreo) es la que la clase usa para ubicar a cada arquitectura en el espacio de diseño de las GNN.

En síntesis para el curso: GraphSAGE es el modelo que convierte la convolución sobre grafos de un procedimiento transductivo y *full-batch* (GCN) en una **función inductiva, muestreada y escalable** —*sample, aggregate, concatenate, project, normalize*— y por eso es la base conceptual de las GNN industriales que la clase presenta a continuación.
