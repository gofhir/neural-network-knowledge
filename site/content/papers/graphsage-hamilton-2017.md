---
title: "GraphSAGE: Inductive Representation Learning (2017)"
weight: 302
math: true
---

{{< paper-card
    title="Inductive Representation Learning on Large Graphs"
    authors="William L. Hamilton, Rex Ying, Jure Leskovec"
    year="2017"
    venue="NeurIPS 2017"
    pdf="/papers/graphsage-hamilton-2017.pdf"
    arxiv="1706.02216" >}}
Paper de Stanford que introdujo **GraphSAGE** (de **SA**mple and aggre**GAtE**), la pieza que convirtió la convolución sobre grafos de un procedimiento transductivo y *full-batch* en una **función inductiva, muestreada y escalable**. En vez de aprender un vector por nodo, aprende una **función agregadora** que genera el *embedding* de cualquier nodo —incluso uno no visto en entrenamiento— combinando las features de su vecindario local. Demuestra el método en tres benchmarks inductivos (citas Web of Science, Reddit, interacciones proteína-proteína), mejorando el F1 en un 51% promedio sobre usar solo features y superando a una baseline transductiva (DeepWalk) que además es 100-500× más lenta al inferir sobre nodos nuevos. Es el ancestro directo de [PinSage](/papers/pinsage-ying-2018) y la base conceptual de las GNN industriales que ve la [Clase 27](/clases/clase-27).
{{< /paper-card >}}

---

## Contexto: el problema transductivo y la necesidad de inductividad

Un **embedding de nodo** comprime la información del vecindario de un nodo en un vector denso de baja dimensión, que luego alimenta tareas *downstream*: clasificación de nodos, *clustering*, predicción de enlaces. Hacia 2016-2017, la familia dominante de métodos —DeepWalk, node2vec, LINE, SDNE, factorizaciones espectrales— aprendía esos vectores con objetivos basados en *random walks* y factorización de matrices, emparentados con el *spectral clustering* y PageRank.

El problema estructural de toda esa familia es que **entrena un vector por cada nodo individual** mediante un *embedding look-up*. De ahí se siguen dos consecuencias graves:

1. **Es inherentemente transductiva.** Las predicciones se hacen sobre los nodos de un único grafo fijo. Un nodo nuevo —un post recién publicado en Reddit, un video subido a YouTube, una proteína de un organismo no estudiado— simplemente no tiene fila en la tabla de embeddings. Adaptar estos métodos al régimen inductivo exige rondas adicionales y caras de descenso de gradiente antes de poder predecir sobre el nodo nuevo: el paper mide que DeepWalk en modo "online" es 100-500× más lento al inferir por esta razón.
2. **El espacio de embeddings no se alinea entre grafos.** Para muchos de estos objetivos la función de pérdida es invariante a rotaciones ortogonales ($ZQ^\top QZ^\top = ZZ^\top$ para cualquier $Q$ ortogonal). El espacio entero puede rotar arbitrariamente durante el entrenamiento, de modo que los embeddings de un grafo B caen en un espacio rotado al azar respecto a uno entrenado sobre A. Eso vuelve imposible **generalizar entre grafos** sin un procedimiento de alineación explícito.

El régimen **inductivo** es más difícil precisamente porque generalizar a nodos no vistos exige reconocer propiedades estructurales del vecindario que revelan el rol local del nodo y su posición global. Y es justo la capacidad que necesitan los sistemas de ML de producción de alto *throughput*, que operan sobre grafos en evolución y encuentran nodos nuevos sin parar.

El [GCN de Kipf & Welling (2016)](/papers/gcn-kipf-2017) había introducido convoluciones sobre grafos como metodología prometedora, pero solo se había aplicado en el régimen **transductivo con grafos fijos**, y su algoritmo exacto requiere conocer el **Laplaciano del grafo completo** en entrenamiento. GraphSAGE se posiciona explícitamente como la **extensión del marco GCN al régimen inductivo**, generalizándolo para usar **funciones de agregación entrenables** más allá de la convolución simple.

## Contribución central: sample and aggregate

La tesis del paper es que en vez de aprender *un embedding por nodo* hay que aprender **una función** que, dado un nodo y sus features, *genera* el embedding agregando información de su vecindario local. Como lo que se aprende es la función y no el embedding, el modelo es **inductivo**: una vez entrenado, se aplica a nodos o grafos completamente nuevos sin reentrenar. La idea tiene tres componentes:

**(a) Aprender funciones de agregación, no embeddings por nodo.** GraphSAGE entrena un conjunto de **funciones agregadoras** que aprenden a combinar features del vecindario local. Cada función agrega desde un número distinto de saltos (*hops*). Al aprovechar las **features de los nodos** (texto, perfil, grados), el modelo aprende a la vez la estructura topológica del vecindario *y* la distribución de features en él. Como las features son el insumo —no una identidad memorizada—, el modelo se transfiere a nodos y grafos nuevos. Si un grafo carece de features ricas, se pueden usar features estructurales presentes en cualquier grafo (grados de nodo).

**(b) Muestreo de vecindario de tamaño fijo para escalar.** En vez de usar el vecindario completo de cada nodo —cuyo costo en el peor caso es $O(|V|)$ por *batch*—, GraphSAGE muestrea uniformemente un **conjunto de vecinos de tamaño fijo** en cada iteración. Esto fija el costo de espacio y tiempo por *batch* en $O(\prod_{i=1}^{K} S_i)$, con $S_i$ y $K$ constantes elegidas por el usuario. En la práctica, $K=2$ y $S_1 \cdot S_2 \le 500$ ya dan alto rendimiento. Es el ingrediente que vuelve a GraphSAGE escalable y la diferencia clave frente al GCN *full-batch*.

**(c) Pérdida no supervisada por random walks, o supervisada.** GraphSAGE puede entrenarse sin supervisión específica de tarea con una pérdida basada en *random walks* que empuja a nodos cercanos a tener representaciones similares y a nodos dispares a tenerlas distintas. Alternativamente, esa pérdida se reemplaza o complementa por un objetivo supervisado (cross-entropy).

## El método en detalle

### Forward: generación de embeddings

El algoritmo de *forward* asume parámetros ya entrenados: $K$ funciones $\text{AGGREGATE}_k$, matrices de pesos $W^k$, una no linealidad $\sigma$ y una función de vecindario $N(v)$. La intuición: en cada iteración $k$ (cada salto) los nodos agregan información de sus vecinos inmediatos, y al iterar incorporan regiones cada vez más lejanas del grafo. Para cada profundidad $k=1\ldots K$ y cada nodo $v$:

1. **Caso base:** $h^0_v \leftarrow x_v$, las features de entrada.
2. **Agregar el vecindario:** $h^k_{N(v)} \leftarrow \text{AGGREGATE}_k\big(\{\, h^{k-1}_u, \forall u \in N(v) \,\}\big)$.
3. **Concatenar con la propia y proyectar:** $h^k_v \leftarrow \sigma\big(W^k \cdot \text{CONCAT}(h^{k-1}_v,\, h^k_{N(v)})\big)$.
4. **Normalizar L2:** $h^k_v \leftarrow h^k_v / \lVert h^k_v \rVert_2$, lo que evita que las normas crezcan sin control entre capas.
5. **Salida:** $z_v \equiv h^K_v$, la representación final tras $K$ saltos.

Esta es exactamente la fórmula que la Clase 27 escribe como $h^t = \sigma(W^t \cdot \text{concat}(h^{t-1}, h'))$, donde $h'$ es el vector agregado del vecindario. La **concatenación** del paso 3 funciona como una *skip connection* entre profundidades de búsqueda —preserva la representación previa del nodo junto a la información del vecindario— y el paper reporta que aporta ganancias significativas. El proceso encaja en la formulación general de [paso de mensajes](/fundamentos/message-passing): agregar mensajes de los vecinos, actualizar el estado del nodo.

### Los tres agregadores

Como los vecinos de un nodo **no tienen orden natural**, el agregador idealmente debe ser **simétrico** (invariante a permutaciones) y a la vez entrenable y expresivo. El paper examina tres:

- **Mean** (media elemento a elemento): casi equivalente a la regla de propagación convolucional del GCN. La variante "GCN" del paper es la que *no* concatena, sino que mezcla nodo y vecinos en una sola media: $h^k_v \leftarrow \sigma(W \cdot \text{MEAN}(\{h^{k-1}_v\} \cup \{h^{k-1}_u\}))$ —es decir, la versión inductiva del GCN.
- **LSTM**: mayor capacidad expresiva, pero *no* es simétrica (procesa secuencialmente); se adapta aplicándola a una **permutación aleatoria** de los vecinos.
- **Pooling** (max-pooling): cada vector de vecino pasa por una red *fully-connected* y luego se aplica un *max* elemento a elemento; simétrica y entrenable.

### Pérdida no supervisada por random walks

La pérdida no supervisada basada en grafo sobre las representaciones de salida $z_u$ es:

$$J_G(z_u) = -\log\big(\sigma(z_u^\top z_v)\big) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)}\big[\log\big(\sigma(-z_u^\top z_{v_n})\big)\big],$$

donde $v$ es un nodo que co-ocurre cerca de $u$ en un *random walk* de longitud fija, $\sigma$ es la sigmoide, $P_n$ la distribución de *negative sampling* y $Q$ el número de negativos. Lo crucial frente a métodos previos: las representaciones $z_u$ se **generan a partir de las features del vecindario local**, no de un look-up por nodo. Los parámetros ($W^k$ y los de los agregadores) se ajustan con SGD y *backpropagation*. En entrenamiento se corrieron 50 *random walks* de longitud 5 por nodo y 20 negativos.

GraphSAGE también tiene un anclaje teórico: está inspirado en el test de isomorfismo de **Weisfeiler-Lehman**. Con $K=|V|$, pesos a la identidad y un *hash* como agregador, el algoritmo *es* una instancia del test WL; GraphSAGE es entonces su **aproximación continua** con agregadores neuronales.

## Experimentos: tres benchmarks inductivos

Las tres tareas son **inductivas** (predicción sobre nodos no vistos; en PPI, además, sobre grafos enteros no vistos). Las baselines: clasificador aleatorio, regresión logística sobre features, DeepWalk, y la concatenación de features crudas con embeddings de DeepWalk. Configuración: ReLU, $K=2$, $S_1=25$, $S_2=10$, dimensión 256, Adam, *batch* 512.

- **Citation (Web of Science).** Grafo de citas de seis campos de biología, 302.424 nodos, grado medio 9.15. Tarea: predecir el campo (6 etiquetas). Entrenamiento 2000-2004, prueba 2005 (grafo en evolución).
- **Reddit.** 232.965 posts de septiembre 2014, conectados si un mismo usuario comenta en ambos; la etiqueta es el *subreddit* (50 comunidades). Grado medio 492. Entrenamiento con los primeros 20 días.
- **PPI (protein-protein interaction).** Generalización **entre grafos**: clasificar roles de proteínas (121 etiquetas de gene ontology), cada grafo un tejido humano distinto. Entrenamiento sobre 20 grafos, prueba sobre 2 grafos **enteramente no vistos**. El 42% de los nodos no tiene features no nulas, lo que vuelve crítica la información del vecindario. DeepWalk no puede aplicarse aquí por la invarianza ortogonal entre grafos disjuntos.

**Resultados (F1 micro-promediado):**

| Método | Citation Sup. | Reddit Sup. | PPI Sup. |
|---|---|---|---|
| Raw features | 0.575 | 0.585 | 0.422 |
| DeepWalk + features | 0.701 | 0.691 | — |
| GraphSAGE-GCN | 0.772 | 0.930 | 0.500 |
| GraphSAGE-mean | 0.820 | 0.950 | 0.598 |
| GraphSAGE-LSTM | 0.832 | 0.954 | 0.612 |
| GraphSAGE-pool | **0.839** | 0.948 | 0.600 |

GraphSAGE supera a todas las baselines por un margen amplio, y los agregadores neuronales entrenables (mean/LSTM/pool) ganan sobre el estilo GCN en un 7.4% promedio. La variante supervisada GraphSAGE-pool supera a "DeepWalk + features" en 19.7% (citas) y 37.2% (Reddit). El rendimiento no supervisado es razonablemente competitivo con el supervisado, lo que indica que el marco produce embeddings útiles sin *fine-tuning* específico de tarea.

**Tiempos y sensibilidad.** Los tiempos de entrenamiento son comparables (LSTM el más lento), pero DeepWalk es **100-500× más lento al inferir** sobre nodos nuevos. Pasar de $K=1$ a $K=2$ da +10-15% de F1; ir más allá de $K=2$ aporta apenas 0-5% pero multiplica el costo por 10-100×. Muestrear vecindarios grandes da retornos decrecientes. Un test de Wilcoxon muestra que mean/LSTM/pool baten al GCN significativamente ($p=0.02$); pool es ~2× más rápido que LSTM, lo que le da una ligera ventaja global.

## Limitaciones

- **Varianza del muestreo.** Submuestrear el vecindario introduce varianza en las representaciones y los gradientes. El paper lo asume como compromiso aceptable a cambio del *runtime* fijo, y muestra empíricamente que la precisión se mantiene, pero la varianza existe y crece con submuestreos agresivos.
- **El vecindario fijo pierde información.** Tomar un tamaño fijo de vecinos (y, en preprocesado, hacer *downsampling* de aristas para que ningún nodo supere grado 128) descarta estructura de nodos de alto grado. El muestreo es **uniforme**, no informado; el propio paper marca explorar *samplers* no uniformes y aprenderlos como trabajo futuro.
- **Profundidad limitada en la práctica.** Más allá de $K=2$ los retornos son marginales y el costo explota, así que GraphSAGE captura sobre todo vecindarios de 2 saltos.
- **Grafos no dirigidos / unimodales.** Las extensiones a grafos dirigidos o multimodales quedan como trabajo futuro.

## Impacto

GraphSAGE se volvió una de las piezas fundacionales de las **GNN inductivas y escalables**. Su descendiente directo más célebre es [PinSage](/papers/pinsage-ying-2018) (Ying et al., 2018), el sistema de recomendación de Pinterest construido sobre el mismo principio de *sample-and-aggregate* y llevado a un grafo de miles de millones de nodos: el primer despliegue de una GNN a escala web real, y el ejemplo canónico de por qué la inductividad importa en producción. El mismo principio de generar embeddings desde features —en vez de memorizarlos por entidad— conecta con las arquitecturas de recuperación tipo [Two-Tower](/papers/two-tower-yi-2019), donde un encoder genera la representación de ítems frescos sin reentrenar. El marco encaja en la formulación general de [message passing](/fundamentos/message-passing) (MPNN) y es una de las arquitecturas de referencia en librerías como PyTorch Geometric y DGL.

## Por qué importa para la Clase 27

La [Clase 27](/clases/clase-27) (Redes Neuronales de Grafos) presenta GraphSAGE con exactamente la mecánica de este paper, y conviene mapear pieza por pieza:

- **La fórmula de la clase es el paso de actualización del algoritmo.** $h^t = \sigma(W^t \cdot \text{concat}(h^{t-1}, h'))$ más normalización, dejando $h'$ "libre" (por ejemplo, el promedio del vecindario). Aquí $h'$ es el resultado del agregador y "promedio" es el agregador **mean**; que la clase lo deje libre refleja el diseño modular del paper, donde el agregador es un *placeholder* intercambiable (mean, LSTM, pool).
- **El muestreo de vecinos para eficiencia** es el ingrediente que vuelve a GraphSAGE escalable a grafos grandes y la diferencia clave frente al [GCN](/papers/gcn-kipf-2017) *full-batch*: vecindario de tamaño fijo, costo por *batch* acotado, $K=2$ y $S_1 \cdot S_2 \le 500$ como receta práctica.
- **Contraste con GCN.** El GCN exacto necesita el Laplaciano del grafo completo y es transductivo; GraphSAGE lo *generaliza* al régimen inductivo con agregadores entrenables. La lección de la Tabla 1 es que **concatenar la representación propia y usar agregadores entrenables ayuda** frente a la convolución simple.

GraphSAGE es, en síntesis para el curso, el modelo que vuelve [inductivas y escalables](/fundamentos/redes-neuronales-de-grafos) las GNN —*sample, aggregate, concatenate, project, normalize*— y por eso es la base de las GNN industriales sobre datos [estructurados](/dominios/estructurados) que la clase presenta a continuación.

## Notas y enlaces

- Código y datasets: [snap.stanford.edu/graphsage](http://snap.stanford.edu/graphsage/)
- Preprint arXiv:1706.02216 (versión consultada: v4, 10 de septiembre de 2018)
- Análisis interno extendido: `clase_27/papers/Hamilton-GraphSAGE-2017.md`
