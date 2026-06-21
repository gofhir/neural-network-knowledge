# Graph Attention Networks (GAT) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Graph Attention Networks*.
- **Autores:** Petar Veličković (University of Cambridge), Guillem Cucurull (Centre de Visió per Computador, UAB), Arantxa Casanova (CVC, UAB), Adriana Romero (Montréal Institute for Learning Algorithms, MILA), Pietro Liò (University of Cambridge) y Yoshua Bengio (MILA). Los tres primeros figuran como coautores con contribución igual; el trabajo de Cucurull y Casanova se realizó durante su paso por MILA.
- **Venue:** *International Conference on Learning Representations (ICLR) 2018*.
- **Año / preprint:** publicado en 2018; preprint arXiv:1710.10903v3 (4 feb 2018), [arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903).
- **Código:** implementación oficial de referencia en [github.com/PetarV-/GAT](https://github.com/PetarV-/GAT) (TensorFlow). Los autores agradecen al equipo de TensorFlow (Abadi et al., 2015).

**Nota sobre su lugar en la Clase 27.** Este es un paper **canónico** del campo de las redes neuronales de grafos (GNN), pero **no aparece citado explícitamente en las slides de la Clase 27** (Redes Neuronales de Grafos). Se incorpora aquí porque es la pieza que *completa el panorama* que la clase construye alrededor de "cómo combinar mensajes de los vecinos": donde la clase presenta GCN (promedio ponderado por grado) y GraphSAGE (agregadores fijos: mean, pool, LSTM), GAT introduce la tercera opción —**aprender los pesos de combinación con atención**— que se volvió tan influyente que hoy es difícil enseñar GNN sin mencionarla. Tratarlo como lectura complementaria es lo correcto: no es material literal del profesor, pero es el eslabón conceptual que vuelve aprendible la "función conmutativa de combinación" de la clase.

La tesis del paper es directa. Las arquitecturas de grafos previas —tanto las **espectrales** (Bruna et al., 2014; Defferrard et al., 2016; Kipf & Welling, 2017) como muchas **no-espectrales**— sufren de dos limitaciones acopladas: (a) los métodos espectrales aprenden filtros que **dependen de la base propia del Laplaciano del grafo**, que a su vez depende de la estructura, de modo que un modelo entrenado sobre un grafo no se transfiere a otro de estructura distinta (son intrínsecamente *transductivos*); y (b) los métodos que sí son aplicables a estructuras variables suelen **ponderar a los vecinos de forma fija o estructural** (por ejemplo, normalizando por el grado), sin permitir que la importancia de un vecino dependa del *contenido* de sus features. GAT resuelve ambas de un golpe con una idea simple: aplicar **capas de self-attention enmascaradas** (*masked self-attentional layers*) en las que cada nodo atiende sobre sus vecinos, calculando coeficientes de atención que indican cuánto importa cada vecino, y combinando linealmente las features de los vecinos con esos pesos. La operación no requiere ninguna operación matricial costosa (inversiones, eigendecomposiciones), no necesita conocer la estructura completa del grafo de antemano, y por tanto se aplica de forma natural a problemas **inductivos** —incluyendo grafos completamente nuevos en test—. Los autores validan GAT en cuatro benchmarks consolidados, igualando o superando el estado del arte: Cora, Citeseer y Pubmed (transductivos) y un dataset de interacción proteína-proteína / PPI (inductivo).

Es importante subrayar el contexto temporal. El paper se sube a arXiv en octubre de 2017, **apenas meses después de "Attention is all you need"** (Vaswani et al., 2017, junio 2017), y antes de que los Transformers dominaran el campo. GAT cita ese trabajo como inspiración del multi-head attention, pero su mecanismo de atención base sigue de cerca a Bahdanau et al. (2015), el paper seminal de atención para traducción. En retrospectiva, GAT es el momento en que el mecanismo de atención cruza desde las secuencias hacia los grafos arbitrarios —y, como veremos en la §8, esa conexión resultó ser de doble vía: un Transformer puede leerse como un GAT sobre el grafo completo.

## 2. Contexto histórico: del Laplaciano al peso aprendido por arista

Para entender por qué GAT importa hay que reconstruir el estado del arte de las GNN hacia 2017, que el propio paper organiza en su introducción.

**Las CNN y el problema del dominio irregular.** Las redes convolucionales triunfaron en datos con estructura de *rejilla* (imágenes, audio, traducción con convoluciones): reutilizan filtros locales con parámetros aprendibles aplicándolos a todas las posiciones del input. Pero muchos dominios interesantes —mallas 3D, redes sociales, redes de telecomunicaciones, redes biológicas, conectomas cerebrales— viven en un dominio *irregular* que se representa como grafo, donde no hay una noción canónica de "vecino de arriba" o "píxel a la derecha". El reto: generalizar la convolución a grafos manteniendo el reparto de pesos y la localidad.

**La línea espectral.** Un primer ataque define la convolución en el **dominio de Fourier del grafo**, vía la eigendescomposición del Laplaciano del grafo (Bruna et al., 2014). El problema es doble: el cómputo es intenso y los filtros resultantes no están localizados espacialmente. Henaff et al. (2015) parametrizan los filtros espectrales con coeficientes suaves para localizarlos; Defferrard et al. (2016) los aproximan con una **expansión de Chebyshev** del Laplaciano, eliminando la necesidad de calcular eigenvectores y produciendo filtros localizados; y Kipf & Welling (2017) —el **GCN** que la Clase 27 sí presenta— simplifican todo restringiendo los filtros a operar sobre el vecindario de 1 salto. Pero el defecto compartido es estructural y no se elimina: **en todos los enfoques espectrales los filtros aprendidos dependen de la base propia del Laplaciano, que depende de la estructura del grafo**. Un modelo entrenado sobre un grafo específico no puede aplicarse directamente a un grafo de estructura diferente. Esto los ata al régimen *transductivo*.

**La línea no-espectral.** La alternativa define convoluciones directamente sobre el grafo, operando sobre grupos de vecinos espacialmente cercanos. El reto aquí es definir un operador que funcione con vecindarios de tamaño variable y conserve el reparto de pesos. Las soluciones previas eran incómodas: Duvenaud et al. (2015) aprenden **una matriz de pesos distinta por cada grado de nodo**; Atwood & Towsley (2016) usan potencias de una matriz de transición; Niepert et al. (2016) extraen y normalizan vecindarios de tamaño fijo. Monti et al. (2016) presentan **MoNet**, un marco espacial que unifica varias arquitecturas tipo CNN sobre grafos y variedades. Y, crucialmente, Hamilton et al. (2017) introducen **GraphSAGE** —que la Clase 27 también presenta—, un método *inductivo* que muestrea un vecindario de tamaño fijo por nodo y aplica un **agregador** sobre él (mean, pooling, o alimentar las features a una LSTM). GraphSAGE obtuvo resultados fuertes en benchmarks inductivos de gran escala, pero arrastra dos compromisos que GAT critica explícitamente (§5): el muestreo de tamaño fijo impide acceder a la totalidad del vecindario en inferencia, y el agregador LSTM —su variante más potente— asume un *orden secuencial* de los nodos que un vecindario no tiene (los autores lo parchean alimentando órdenes aleatorios).

**El hueco que GAT llena.** Mirando esas dos líneas se ve el vacío con nitidez. Los métodos espectrales aprenden pesos pero dependen del Laplaciano (no inductivos). Los no-espectrales son inductivos pero ponderan a los vecinos de forma fija o estructural —GCN normaliza por el grado, GraphSAGE promedia o agrupa— **sin dejar que la importancia de un vecino dependa del contenido de sus features**. Faltaba un mecanismo de **pesos aprendidos, dependientes del contenido, por arista, y compatible con el régimen inductivo**. Mientras tanto, los mecanismos de atención se habían vuelto estándar de facto en tareas secuenciales (Bahdanau et al., 2015; Gehring et al., 2016): permiten lidiar con inputs de tamaño variable, enfocándose en las partes más relevantes. Cuando la atención calcula la representación de una sola secuencia se llama *self-attention* o intra-atención; Vaswani et al. (2017) habían mostrado que self-attention sola basta para construir un modelo SOTA de traducción. GAT toma exactamente ese mecanismo y lo lleva al vecindario de un grafo.

## 3. Contribución central

La contribución de GAT es el **graph attentional layer**: una capa que computa la nueva representación de cada nodo atendiendo sobre sus vecinos mediante self-attention, con pesos de atención aprendidos y dependientes de las features. El paper resalta tres propiedades atractivas de esta arquitectura:

1. **Es eficiente**, porque la operación es paralelizable across los pares nodo-vecino (no hay dependencias secuenciales ni operaciones matriciales globales).
2. **Maneja vecindarios de grado arbitrario**, asignando pesos arbitrarios (aprendidos) a los distintos vecinos.
3. **Es directamente aplicable a problemas inductivos**, incluyendo tareas donde el modelo debe generalizar a grafos completamente nuevos.

El corazón es el cálculo del coeficiente de atención normalizado entre el nodo $i$ y su vecino $j$:

$$\alpha_{ij} = \frac{\exp\!\big(\text{LeakyReLU}(\vec{a}^{\,T}[W\vec{h}_i \,\Vert\, W\vec{h}_j])\big)}{\sum_{k\in\mathcal{N}_i}\exp\!\big(\text{LeakyReLU}(\vec{a}^{\,T}[W\vec{h}_i \,\Vert\, W\vec{h}_k])\big)}$$

donde $W$ es una transformación lineal compartida, $\Vert$ es concatenación, $\vec{a}$ es un vector de pesos del mecanismo de atención, y la normalización softmax es sobre el vecindario $\mathcal{N}_i$. Tres rasgos lo distinguen de todo lo anterior: el peso $\alpha_{ij}$ se **aprende por arista** y depende del contenido de ambos extremos; el cómputo **no requiere la matriz de adyacencia completa ni el Laplaciano** —solo saber quiénes son los vecinos de $i$—; y al ser un mecanismo *compartido* aplicado a todas las aristas, transfiere a grafos nuevos. El paper también introduce **multi-head attention** para estabilizar el aprendizaje, siguiendo a Vaswani et al. (2017).

Los autores son cuidadosos en situar su trabajo respecto a vecinos teóricos: GAT puede reformularse como una **instancia particular de MoNet** (Monti et al., 2016) —fijando la función de pseudo-coordenadas a la concatenación de features y la función de peso a un softmax de un MLP—, con la diferencia clave de que GAT usa **features de los nodos** para las similitudes, no propiedades estructurales (lo que evitaría tener que conocer la estructura de antemano). Y el compartir un cómputo neuronal across aristas recuerda a las *relational networks* (Santoro et al., 2017) y a VAIN (Hoshen, 2017).

## 4. El método en detalle

### 4.1. El graph attentional layer

El input de la capa es un conjunto de features de nodos $h = \{\vec{h}_1, \dots, \vec{h}_N\}$, con $\vec{h}_i \in \mathbb{R}^F$ ($N$ nodos, $F$ features por nodo). La salida es un nuevo conjunto $h' = \{\vec{h}'_1, \dots, \vec{h}'_N\}$ con $\vec{h}'_i \in \mathbb{R}^{F'}$ (posiblemente otra cardinalidad). Los pasos:

**Paso 1 — transformación lineal compartida.** Para ganar poder expresivo se requiere al menos una transformación lineal aprendible. Se aplica una matriz de pesos compartida $W \in \mathbb{R}^{F' \times F}$ a cada nodo. Es "compartida" en el sentido convolucional: los mismos parámetros para todos los nodos, lo que preserva el *weight sharing*.

**Paso 2 — coeficientes de atención sin normalizar.** Un mecanismo de atención compartido $a: \mathbb{R}^{F'} \times \mathbb{R}^{F'} \to \mathbb{R}$ computa
$$e_{ij} = a(W\vec{h}_i, W\vec{h}_j),$$
que indica la importancia de las features del nodo $j$ para el nodo $i$. En su formulación más general, esto permitiría que cada nodo atienda sobre cualquier otro, descartando toda información estructural.

**Paso 3 — masked attention.** Aquí entra la estructura del grafo: en vez de atender sobre todos los nodos, GAT solo computa $e_{ij}$ para $j \in \mathcal{N}_i$, el vecindario de $i$ en el grafo. En todos los experimentos $\mathcal{N}_i$ son exactamente los **vecinos de primer orden de $i$ (incluyéndose a sí mismo)**. Esta *masked attention* es la inyección de la topología: la atención se restringe a las aristas que existen. Una consecuencia elegante: el grafo **no necesita ser no-dirigido** —basta con omitir el cómputo de $\alpha_{ij}$ si la arista $j \to i$ no existe.

**Paso 4 — normalización.** Para que los coeficientes sean comparables across nodos, se normalizan con softmax sobre el vecindario (Ecuación 2 del paper):
$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}.$$

**El mecanismo de atención concreto.** En los experimentos, $a$ es una **red feedforward de una sola capa**, parametrizada por un vector de pesos $\vec{a} \in \mathbb{R}^{2F'}$, con no-linealidad **LeakyReLU** (pendiente negativa $\alpha = 0.2$). Desplegando todo se obtiene la Ecuación 3 (la fórmula completa de la §3). El operando de LeakyReLU es $\vec{a}^{\,T}[W\vec{h}_i \,\Vert\, W\vec{h}_j]$: se concatenan las proyecciones de los dos nodos, se proyecta sobre $\vec{a}$ para obtener un escalar, y se pasa por LeakyReLU antes del softmax.

**Paso 5 — combinación.** Una vez obtenidos los $\alpha_{ij}$ normalizados, sirven como pesos de una combinación lineal de las features transformadas de los vecinos, seguida (opcionalmente) de una no-linealidad $\sigma$ (Ecuación 4):
$$\vec{h}'_i = \sigma\!\left(\sum_{j\in\mathcal{N}_i} \alpha_{ij}\, W\vec{h}_j\right).$$

### 4.2. Multi-head attention

Para estabilizar el aprendizaje de la self-attention, GAT extiende el mecanismo a **multi-head attention**, como Vaswani et al. (2017). Se ejecutan $K$ mecanismos de atención independientes (cada uno con su propio $\vec{a}^k$ y su propia $W^k$), y sus salidas se **concatenan** (Ecuación 5):
$$\vec{h}'_i = \big\Vert_{k=1}^{K} \sigma\!\left(\sum_{j\in\mathcal{N}_i} \alpha_{ij}^{k}\, W^k\vec{h}_j\right),$$
de modo que la salida tiene $K F'$ features por nodo en vez de $F'$.

**El caso de la capa final.** Cuando el multi-head attention se aplica en la **capa de predicción** (la última), concatenar deja de tener sentido (cambiaría la dimensión de salida). En su lugar, GAT **promedia** las $K$ cabezas y retrasa la no-linealidad final (softmax o sigmoide logística) hasta después del promedio (Ecuación 6):
$$\vec{h}'_i = \sigma\!\left(\frac{1}{K}\sum_{k=1}^{K}\sum_{j\in\mathcal{N}_i} \alpha_{ij}^{k}\, W^k\vec{h}_j\right).$$

### 4.3. Complejidad y comparación con el trabajo previo

El paper dedica una subsección (2.2) a comparar la capa con los métodos previos:

- **Eficiencia computacional.** La operación de la capa self-attentional se **paraleliza across todas las aristas**, y el cómputo de las features de salida across todos los nodos. No hay eigendescomposiciones ni operaciones matriciales costosas. La complejidad temporal de una sola cabeza de GAT que computa $F'$ features es $O(|V|FF' + |E|F')$, donde $|V|$ y $|E|$ son el número de nodos y aristas. Esto está **a la par con GCN**. El multi-head multiplica el almacenamiento y los parámetros por un factor $K$, pero las cabezas son independientes y paralelizables.
- **Capacidad de modelo.** A diferencia de GCN, GAT asigna (implícitamente) **importancias distintas a nodos del mismo vecindario**, lo que el paper describe como "un salto en capacidad de modelo". Además, analizar los pesos de atención aprendidos abre la puerta a la **interpretabilidad** (como ocurrió en traducción con Bahdanau et al., 2015).
- **No depende de la estructura global.** El mecanismo se aplica de forma compartida a todas las aristas, así que no requiere acceso previo a la estructura global del grafo ni a las features de todos los nodos. De ahí dos implicaciones: el grafo puede ser dirigido, y la técnica es **directamente aplicable a aprendizaje inductivo**, incluso a grafos no vistos en entrenamiento.
- **Frente a GraphSAGE.** GAT trabaja con la **totalidad del vecindario** (a costa de un footprint computacional variable, todavía comparable a GCN) y **no asume ningún orden** dentro de él —resolviendo los dos compromisos de GraphSAGE (muestreo de tamaño fijo y agregador LSTM dependiente del orden).
- **Versión esparsa.** Los autores produjeron una versión de la capa que aprovecha operaciones de matrices esparsas, reduciendo la complejidad de almacenamiento a lineal en nodos y aristas. La limitación: el framework que usaron solo soporta multiplicación esparsa para tensores de rango 2, lo que restringe el *batching* (especialmente con múltiples grafos).

## 5. Experimentos

GAT se evalúa en **cuatro benchmarks consolidados**, tres transductivos y uno inductivo (Tabla 1 del paper):

**Datasets transductivos — redes de citas (Sen et al., 2008), siguiendo el setup de Yang et al. (2016).** Los nodos son documentos, las aristas son citas (no-dirigidas), las features son una representación *bag-of-words*, y cada nodo tiene una etiqueta de clase. Solo se usan **20 nodos por clase** para entrenar (aunque, honrando el régimen transductivo, el algoritmo tiene acceso a los vectores de features de *todos* los nodos); 1000 nodos de test y 500 de validación (los mismos de Kipf & Welling, 2017).
- **Cora:** 2708 nodos, 5429 aristas, 7 clases, 1433 features/nodo.
- **Citeseer:** 3327 nodos, 4732 aristas, 6 clases, 3703 features/nodo.
- **Pubmed:** 19717 nodos, 44338 aristas, 3 clases, 500 features/nodo.

**Dataset inductivo — PPI (protein-protein interaction; Zitnik & Leskovec, 2017).** 24 grafos correspondientes a distintos tejidos humanos: 20 para entrenar, 2 para validar, 2 para test. **Crucialmente, los grafos de test permanecen completamente no observados durante el entrenamiento** (esto es lo que lo hace inductivo). En promedio 2372 nodos por grafo; 50 features por nodo (gene sets posicionales, motif gene sets, firmas inmunológicas); **121 etiquetas por nodo en formato multietiqueta** (un nodo puede tener varias). Se usó el preprocesado de Hamilton et al. (2017).

**Configuración de la arquitectura.**
- *Transductivo:* GAT de **dos capas**. Primera capa: $K=8$ cabezas de atención computando $F'=8$ features cada una (64 features en total), seguida de ELU. Segunda capa (clasificación): una sola cabeza que computa $C$ features (= número de clases), seguida de softmax. Regularización agresiva por el pequeño tamaño de entrenamiento: $L_2$ con $\lambda = 0.0005$ y **dropout $p = 0.6$** en los inputs de ambas capas *y en los coeficientes de atención normalizados* (lo que significa que en cada iteración cada nodo ve un vecindario muestreado estocásticamente). Pubmed, por su training set aún menor (60 ejemplos), requirió $K=8$ cabezas de salida y $\lambda = 0.001$.
- *Inductivo:* GAT de **tres capas**. Las dos primeras: $K=4$ cabezas con $F'=256$ features (1024 en total) + ELU. Capa final (multietiqueta): $K=6$ cabezas computando 121 features cada una, **promediadas** y seguidas de sigmoide logística. Sin $L_2$ ni dropout (el training set es grande), pero **con skip connections** (He et al., 2016) sobre la capa atencional intermedia. Batch de 2 grafos. Glorot init, Adam, cross-entropy, early stopping con paciencia de 100 épocas.

Para aislar el aporte de la atención, en el setting inductivo entrenan además un **Const-GAT**: la misma arquitectura pero con un mecanismo de atención *constante* $a(x,y)=1$ —que asigna el mismo peso a cada vecino, esencialmente un operador inductivo tipo GCN.

**Resultados (Tablas 2 y 3).** En transductivo se reporta accuracy media (±desviación) sobre 100 corridas:

| Método | Cora | Citeseer | Pubmed |
|---|---|---|---|
| GCN (Kipf & Welling, 2017) | 81.5% | 70.3% | 79.0% |
| MoNet (Monti et al., 2016) | 81.7 ± 0.5% | — | 78.8 ± 0.3% |
| **GAT (ours)** | **83.0 ± 0.7%** | **72.5 ± 0.7%** | **79.0 ± 0.3%** |

En inductivo (PPI), micro-F1 promediado sobre 10 corridas:

| Método | PPI |
|---|---|
| GraphSAGE-LSTM (Hamilton et al., 2017) | 0.612 |
| GraphSAGE* (mejor variante reajustada por los autores) | 0.768 |
| Const-GAT (ours) | 0.934 ± 0.006 |
| **GAT (ours)** | **0.973 ± 0.002** |

Las lecturas que el paper extrae: GAT mejora a GCN por **1.5% en Cora y 1.6% en Citeseer**, sugiriendo que asignar pesos distintos a vecinos del mismo vecindario ayuda. En PPI la mejora es dramática: **+20.5% respecto al mejor GraphSAGE** que pudieron obtener, demostrando el potencial inductivo y el valor de observar el vecindario completo; y **+3.9% respecto a Const-GAT** —la misma arquitectura con atención constante—, lo que demuestra *directamente* que la ganancia viene del mecanismo de atención, no de la arquitectura ni del mayor número de parámetros. Finalmente, una visualización **t-SNE** (Maaten & Hinton, 2008) de las representaciones de la primera capa de un GAT preentrenado en Cora muestra clustering discernible que corresponde a las siete clases, confirmando poder discriminativo.

## 6. Limitaciones

El paper es honesto con sus límites, varios en la subsección de comparaciones y otros en las conclusiones:

- **Batching restringido por la implementación esparsa.** La versión esparsa solo soportaba multiplicación de matrices esparsas para tensores de rango 2 en el framework usado, lo que limita el batching, especialmente con datasets de múltiples grafos. Los autores lo nombran como dirección importante de trabajo futuro.
- **GPUs no siempre ayudan en el régimen esparso.** Dependiendo de la regularidad de la estructura del grafo, las GPU pueden no ofrecer ventajas mayores frente a CPUs en estos escenarios esparsos.
- **El campo receptivo está acotado por la profundidad.** Como en GCN y modelos similares, el tamaño del "receptive field" del modelo está limitado por la profundidad de la red. Sugieren que las skip connections (He et al., 2016) podrían extenderla.
- **Cómputo redundante en paralelización distribuida.** Paralelizar across todas las aristas, sobre todo de forma distribuida, puede implicar mucho cómputo redundante porque los vecindarios suelen solaparse fuertemente en los grafos de interés.
- **Solo clasificación de nodos.** En las conclusiones, los autores listan como trabajo futuro extender el método a **clasificación de grafos** (no solo de nodos), incorporar **features de aristas** (que indicarían relaciones entre nodos), manejar batches más grandes, y aprovechar la atención para un análisis serio de interpretabilidad —que en el paper queda apenas esbozado (la visualización de coeficientes de Cora se deja "para trabajo futuro" porque requiere conocimiento de dominio).

## 7. Conexión con la Clase 27 (Redes Neuronales de Grafos)

GAT no está en las slides, pero **completa el panorama** que la Clase 27 construye. La clase organiza las GNN alrededor de la idea de *message passing*: cada nodo agrega ("combina") mensajes de sus vecinos mediante una **función conmutativa** (invariante a permutaciones, porque un vecindario no tiene orden), y actualiza su estado. La pregunta pedagógica central es *cómo combinar esos mensajes*, y la clase presenta dos respuestas:

- **GCN** (Kipf & Welling, 2017): combina con un **promedio ponderado por el grado** de los nodos —un peso *fijo y estructural*, $1/\sqrt{d_i d_j}$, que no depende del contenido de los vecinos.
- **GraphSAGE** (Hamilton et al., 2017): combina con **agregadores fijos** elegidos de antemano —mean, pooling, o LSTM— igualmente independientes del contenido relativo de cada par nodo-vecino.

**GAT es la tercera respuesta: aprender los pesos de combinación con atención.** En vez de fijar el peso de cada vecino por su grado (GCN) o por un agregador predefinido (GraphSAGE), GAT calcula $\alpha_{ij}$ —un peso por arista, aprendido, dependiente de las features de ambos extremos— y combina con él. Dicho en el vocabulario de la clase: GAT toma la "función conmutativa de combinación" y la **vuelve aprendible**. La invariancia a permutaciones se preserva (el softmax sobre el vecindario y la suma ponderada no dependen del orden), pero los pesos dejan de ser un dato estructural y pasan a ser un parámetro del modelo que se ajusta al contenido. Const-GAT, el ablation del paper, es precisamente el puente conceptual: con atención constante, GAT *colapsa a un operador tipo GCN inductivo*; la diferencia de 3.9% en PPI es la medida empírica de cuánto vale aprender los pesos en lugar de fijarlos.

La conexión se extiende a otras dos clases del diplomado:

- **Mecanismo de atención (Clase 15).** El $\alpha_{ij}$ de GAT es literalmente el mecanismo de atención de Bahdanau et al. (2015) que la Clase 15 introduce, trasladado de los pares query-key de una secuencia a los pares nodo-vecino de un grafo. El softmax que normaliza la importancia de cada vecino es el mismo softmax que normaliza la atención sobre las palabras de una frase fuente.
- **Transformers (Clase 14).** El multi-head attention de GAT viene directamente de Vaswani et al. (2017). Y aquí está la idea profunda que vuelve a GAT un puente y no un destino: **un Transformer es, esencialmente, un GAT sobre el grafo completo**. Si se toma una secuencia de tokens, se la representa como un grafo donde *todos* los nodos están conectados con *todos* (vecindario $\mathcal{N}_i$ = todos los demás tokens), y se aplica self-attention sobre ese grafo, se recupera la atención del Transformer. GAT es ese mismo mecanismo pero con *masked attention* que restringe el vecindario a las aristas que existen. Visto al revés: la atención del Transformer es el caso degenerado de GAT en el que el grafo es un *clique* completo. Esta dualidad —que el paper anticipa al citar a Vaswani meses después de su publicación— es la razón por la que GAT terminó siendo tan influyente: ancla las GNN y los Transformers en un mismo marco, y abre la línea de los *Graph Transformers* que dominaría la investigación posterior.

En suma, para la Clase 27 GAT es la pieza que responde la pregunta que GCN y GraphSAGE dejan abierta —¿y si los pesos de combinación se aprendieran?— y que, al responderla con atención, conecta la teoría de grafos del módulo con el corazón atencional del resto del diplomado.
