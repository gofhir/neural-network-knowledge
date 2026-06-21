# Modeling Relational Data with Graph Convolutional Networks (R-GCN) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Modeling Relational Data with Graph Convolutional Networks*.
- **Autores:** Michael Schlichtkrull (University of Amsterdam), Thomas N. Kipf (University of Amsterdam) — ambos con contribución igual—, Peter Bloem (VU Amsterdam), Rianne van den Berg (University of Amsterdam), Ivan Titov (University of Amsterdam) y Max Welling (University of Amsterdam, CIFAR).
- **Venue:** *Extended Semantic Web Conference* (ESWC) 2018 (best student research paper). El manuscrito que analizamos es la versión arXiv.
- **Año / preprint:** publicado en 2018; arXiv:1703.06103v4 (26 oct 2017), [arxiv.org/abs/1703.06103](https://arxiv.org/abs/1703.06103).
- **Linaje:** Kipf y Welling firman tanto el GCN original (ICLR 2017) como el variational graph auto-encoder (2016); R-GCN es la fusión deliberada de esas dos líneas para datos **multi-relacionales**.

Este paper resuelve un problema concreto y muy citado: cómo aplicar las redes convolucionales de grafos (GCN) a **grafos de conocimiento**, donde no hay un solo tipo de arista sino *muchos* (cientos o miles de relaciones distintas: `educated_at`, `lived_in`, `country`, `award`...). El GCN estándar agrega información de los vecinos con una **única** matriz de pesos compartida por todas las aristas, lo que es razonable en un grafo homogéneo (una red de citas, una malla de píxeles) pero pierde toda la semántica cuando la *etiqueta* de la arista es justamente lo que importa. R-GCN —*Relational* GCN— introduce **pesos por tipo de relación** ($W_r$ distinto para cada relación $r$ y para cada dirección), de modo que el mensaje que un vecino envía depende de *cómo* está conectado, no solo de *que* está conectado.

La tesis tiene dos mitades. La primera es de modelado: las dos tareas canónicas de completitud de bases de conocimiento —**link prediction** (recuperar tripletas faltantes sujeto-predicado-objeto) y **entity classification** (asignar tipos/atributos a entidades)— pueden atacarse con un mismo encoder relacional que acumula evidencia sobre la vecindad del grafo. La segunda es de ingeniería: como los pesos por relación hacen explotar el número de parámetros con el número de relaciones, el paper introduce **dos esquemas de regularización** —*basis decomposition* y *block-diagonal decomposition*— que comparten parámetros entre relaciones o imponen sparsidad. Para link prediction, el modelo se plantea como un **autoencoder**: un encoder R-GCN que produce embeddings de entidades, seguido de un decoder **DistMult** que puntúa tripletas. El resultado más llamativo del paper es una mejora del **29.8%** sobre el baseline de solo-decoder (DistMult puro) en FB15k-237, el dataset más difícil y limpio.

Para la Clase 27 (Redes Neuronales de Grafos) este paper importa porque es **la aplicación estrella** del marco de message passing al mundo real de los knowledge graphs, y porque concreta dos ideas que la clase desarrolla en abstracto: que el cálculo del mensaje puede depender del *tipo* de arista (no solo del nodo origen y destino), y que una matriz de pesos por tipo de relación es la realización natural de esa idea —el mismo espíritu que las matrices $E_k$ por tipo de arista de las Gated Graph Neural Networks.

## 2. Contexto: grafos de conocimiento, datos relacionales y los límites del GCN homogéneo

Las bases de conocimiento (DBpedia, Wikidata, Yago, Freebase) organizan hechos como **tripletas** de la forma `(sujeto, predicado, objeto)` —por ejemplo `(Mikhail Baryshnikov, educated_at, Vaganova Academy)`. El paper introduce la terminología que usaremos en toda la clase: *Baryshnikov* y *Vaganova Academy* son **entidades**, y `educated_at` es una **relación**. Además, las entidades llevan **tipos** (`person`, `university`, `ballet_dancer`). La representación natural de una base de conocimiento es entonces un **multigrafo dirigido y etiquetado**: nodos = entidades, aristas etiquetadas = relaciones. La Figura 1 del paper —que la Clase 27 reproduce casi literalmente en su slide de Schlichtkrull 2018— muestra exactamente este fragmento alrededor de Baryshnikov, con `educated_at`, `award`, `citizen_of`/`country` y la entidad U.S.A.

El problema motivador es que **incluso las bases de conocimiento más grandes están incompletas**, y esa incompletitud daña aplicaciones río abajo (question answering, information retrieval). Predecir la información faltante es el objeto del *statistical relational learning* (SRL). El paper considera dos tareas fundamentales de SRL:

- **Link prediction:** recuperar tripletas faltantes. En la Figura 1, inferir que `(Mikhail Baryshnikov, lived_in, Russia)` debería pertenecer al grafo (la arista roja faltante).
- **Entity classification:** recuperar atributos/tipos faltantes de las entidades. En la misma figura, inferir que Vaganova Academy tiene el tipo `university`, o que Baryshnikov es `person`.

La intuición que el paper explota es que **buena parte de la información faltante reside en la estructura de vecindad del grafo**: saber que Baryshnikov fue educado en la Vaganova Academy implica a la vez que es una `person` y que probablemente vivió en Rusia. Un modelo que propague información a través de las aristas relacionales puede capturar esas dependencias.

¿Por qué no usar GCN directamente? Porque el GCN de Kipf y Welling (2017), igual que los grafos de moléculas de Duvenaud et al. (2015), opera sobre **vecindarios locales con una sola matriz de pesos** $W$, implícitamente asumiendo un único tipo de arista. La regla de actualización del GCN homogéneo se inscribe en el **marco de message passing** de Gilmer et al. (2017), que la Clase 27 toma como columna vertebral:

$$h_i^{(l+1)} = \sigma\!\left( \sum_{m \in \mathcal{M}_i} g_m(h_i^{(l)}, h_j^{(l)}) \right)$$

donde $h_i^{(l)} \in \mathbb{R}^{d^{(l)}}$ es el estado oculto del nodo $v_i$ en la capa $l$, $\mathcal{M}_i$ es el conjunto de mensajes entrantes (típicamente las aristas entrantes), $g_m(\cdot,\cdot)$ es una función de mensaje —a menudo una transformación lineal $g_m(h_i,h_j)=Wh_j$ con una *única* $W$— y $\sigma$ es una no linealidad como ReLU. El punto crítico: con una sola $W$, las relaciones `educated_at`, `lived_in` y `country` se mezclan indistinguiblemente. La semántica relacional, que es justamente lo que hace útil a un knowledge graph, se borra. R-GCN nace para reparar esa pérdida.

## 3. Contribución central: pesos por tipo de relación y su regularización

### 3.1. La extensión relacional del message passing

La contribución de modelado de R-GCN es deliberadamente simple: **introducir transformaciones específicas por relación**. En vez de una sola $W$, hay una $W_r$ por cada relación $r$, y la suma sobre mensajes se descompone en una suma anidada sobre relaciones y, dentro de cada relación, sobre los vecinos bajo esa relación. La regla de propagación (ecuación 2 del paper) es:

$$h_i^{(l+1)} = \sigma\!\left( \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}}\, W_r^{(l)} h_j^{(l)} \;+\; W_0^{(l)} h_i^{(l)} \right)$$

Las piezas, una por una:

- $\mathcal{N}_i^r$ es el conjunto de índices de vecinos del nodo $i$ **bajo la relación $r$**. Esto es lo que diferencia a R-GCN del GCN: los vecinos se agrupan por tipo de relación antes de transformarlos.
- $W_r^{(l)}$ es la **matriz de pesos específica de la relación $r$** en la capa $l$. Es el corazón del modelo: el mensaje que un vecino $j$ envía a $i$ se transforma según la relación que los une. El paper enfatiza que estas transformaciones dependen del **tipo y la dirección** de la arista.
- $c_{i,r}$ es una **constante de normalización** específica del problema, que puede aprenderse o fijarse de antemano (la elección por defecto es $c_{i,r} = |\mathcal{N}_i^r|$, el grado del nodo bajo esa relación).
- $W_0^{(l)} h_i^{(l)}$ es la **self-connection** (auto-conexión): asegura que la representación de un nodo en la capa $l+1$ también dependa de su propia representación en la capa $l$. En la práctica se implementa añadiendo a cada nodo una arista de un tipo de relación especial (self-loop).

**La dirección importa.** El paper aclara en una nota al pie que $\mathcal{R}$ contiene las relaciones tanto en su dirección canónica (`born_in`) como en su dirección inversa (`born_in_inv`). Es decir, para una arista `(s, r, o)` se modela tanto el mensaje de $o$ hacia $s$ como el de $s$ hacia $o$, con matrices distintas. La Figura 2 del paper lo dibuja: un nodo rojo recibe activaciones de sus vecinos azules, transformadas *por separado* para cada tipo de relación y para las aristas entrantes (`rel_N (in)`) y salientes (`rel_N (out)`), más el self-loop; todo se acumula en una suma normalizada y pasa por ReLU. Esta actualización por nodo se computa en paralelo, con parámetros compartidos en todo el grafo, y en la práctica se implementa eficientemente con multiplicaciones de matrices dispersas. Apilar $L$ capas permite capturar dependencias a $L$ pasos relacionales de distancia.

### 3.2. El problema de la explosión de parámetros

Aquí aparece el problema central que el resto del paper resuelve. Una $W_r$ por relación significa que el número de parámetros **crece linealmente con el número de relaciones**. En grafos de conocimiento realistas eso es catastrófico: FB15k tiene 1.345 relaciones, y cada $W_r$ es una matriz densa $d^{(l+1)} \times d^{(l)}$. El paper advierte dos consecuencias: **overfitting sobre relaciones raras** (las que aparecen en pocas tripletas no tienen datos para ajustar bien su $W_r$) y **modelos de tamaño descomunal**. Sin regularización, R-GCN sería impracticable en bases de conocimiento reales.

### 3.3. Dos soluciones: basis decomposition y block-diagonal decomposition

El paper propone dos métodos de regularización de los pesos de las capas R-GCN.

**(a) Basis decomposition (descomposición en base).** Cada $W_r^{(l)}$ se define como una combinación lineal de un conjunto compartido de $B$ **transformaciones base** $V_b^{(l)}$:

$$W_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} V_b^{(l)}$$

Las matrices base $V_b^{(l)} \in \mathbb{R}^{d^{(l+1)} \times d^{(l)}}$ son **compartidas por todas las relaciones**; lo único que depende de $r$ son los **coeficientes escalares** $a_{rb}^{(l)}$. Esto es una forma de **weight sharing efectivo entre tipos de relación**: solo hay que aprender $B$ matrices (con $B$ pequeño) más un vector de $B$ coeficientes por relación, en lugar de una matriz densa por relación. El beneficio para las relaciones raras es directo: como las actualizaciones de gradiente de las bases se comparten entre relaciones frecuentes y raras, las raras "heredan" estructura aprendida de las frecuentes, lo que **alivia el overfitting**.

**(b) Block-diagonal decomposition (descomposición en bloques).** Cada $W_r^{(l)}$ se define como la suma directa (block-diagonal) de un conjunto de matrices de baja dimensión:

$$W_r^{(l)} = \bigoplus_{b=1}^{B} Q_{br}^{(l)}, \qquad Q_{br}^{(l)} \in \mathbb{R}^{(d^{(l+1)}/B) \times (d^{(l)}/B)}$$

Es decir, $W_r^{(l)} = \mathrm{diag}(Q_{1r}^{(l)}, \dots, Q_{Br}^{(l)})$ es **block-diagonal**: solo los bloques de la diagonal son no nulos. Esto impone una **restricción de sparsidad** sobre la matriz de cada relación. La intuición que codifica: las características latentes pueden agruparse en conjuntos de variables más fuertemente acopladas *dentro* de cada grupo que *entre* grupos. Aquí, a diferencia de la basis decomposition, los parámetros no se comparten entre relaciones, pero cada relación tiene muchos menos parámetros.

Ambas descomposiciones reducen el número de parámetros para datos altamente multi-relacionales. El paper las posiciona como complementarias: la basis decomposition es weight sharing entre relaciones (mejor para relaciones raras), la block decomposition es una restricción de sparsidad por relación.

### 3.4. El setup encoder-decoder para link prediction

Para link prediction, R-GCN se enmarca como un **graph autoencoder** (heredando del variational graph auto-encoder de Kipf y Welling, 2016):

- **Encoder:** una R-GCN que mapea cada entidad $v_i$ a un vector real $e_i = h_i^{(L)} \in \mathbb{R}^d$ (la representación de la última capa). Esto es lo que distingue el trabajo de la mayoría de los métodos previos de factorización, que optimizan un vector $e_i$ por entidad *directamente* en entrenamiento, sin encoder. El encoder acumula evidencia sobre la vecindad relacional en múltiples pasos.
- **Decoder:** una función de scoring que reconstruye las aristas a partir de las representaciones de los vértices. El paper usa **DistMult** (Yang et al., 2014), donde cada relación $r$ se asocia a una matriz diagonal $R_r \in \mathbb{R}^{d \times d}$ y una tripleta se puntúa como:

$$f(s, r, o) = e_s^\top R_r\, e_o$$

DistMult es uno de los métodos de factorización más simples y efectivos en benchmarks estándar; la elección del decoder es ortogonal a la del encoder, y el paper señala que cualquier función de scoring (ComplEx, HolE...) podría incorporarse en el mismo marco.

El modelo se entrena con **negative sampling**: por cada tripleta observada se muestrean $\omega$ tripletas negativas, corrompiendo aleatoriamente el sujeto o el objeto. La pérdida es una cross-entropy (loss logística) que empuja a puntuar las tripletas observadas más alto que las corrompidas:

$$\mathcal{L} = -\frac{1}{(1+\omega)|\hat{\mathcal{E}}|} \sum_{(s,r,o,y)\in\mathcal{T}} \Big( y \log \ell\big(f(s,r,o)\big) + (1-y)\log\big(1 - \ell\big(f(s,r,o)\big)\big) \Big)$$

donde $\mathcal{T}$ es el conjunto de tripletas reales y corrompidas, $\ell$ es la sigmoide logística e $y$ indica si la tripleta es positiva ($y=1$) o negativa ($y=0$).

Para **entity classification** el setup es más simple: se apilan capas R-GCN (ecuación 2) con un softmax por nodo en la salida de la última capa, y se minimiza la cross-entropy sobre los nodos etiquetados (ignorando los no etiquetados). El input de la primera capa, si no hay features, es un **vector one-hot único por nodo**.

## 4. Experimentos y resultados

### 4.1. Entity classification

Se evalúa sobre cuatro datasets en formato RDF (Ristoski, de Vries y Paulheim 2016): **AIFB, MUTAG, BGS y AM**. Las escalas varían enormemente —de 8.285 entidades y 45 relaciones (AIFB) a 1.666.764 entidades y 133 relaciones (AM)—, con entre 146 y 1.000 entidades etiquetadas y de 2 a 11 clases. Un detalle metodológico importante: se eliminan las relaciones usadas para crear las etiquetas (p.ej. `employs`/`affiliation` en AIFB, `isMutagenic` en MUTAG) para evitar fugas.

El R-GCN usado es un modelo de **2 capas con 16 unidades ocultas** (10 para AM), **basis decomposition**, entrenado con Adam por 50 épocas, learning rate 0.01, y normalización $c_{i,r} = |\mathcal{N}_i^r|$. Los baselines son RDF2Vec, kernels Weisfeiler-Lehman (WL) y extractores de features hechos a mano (Feat).

Resultados (accuracy, promedio sobre 10 corridas):

| Modelo | AIFB | MUTAG | BGS | AM |
|---|---|---|---|---|
| Feat | 55.55 | 77.94 | 72.41 | 66.66 |
| WL | 80.55 | 80.88 | 86.20 | 87.37 |
| RDF2Vec | 88.88 | 67.20 | 87.24 | 88.33 |
| **R-GCN** | **95.83** | 73.23 | 83.10 | **89.29** |

R-GCN logra **estado del arte en AIFB y AM**, pero queda por detrás en **MUTAG y BGS**. El paper ofrece una explicación honesta: MUTAG es un dataset de grafos moleculares convertido a RDF y BGS de tipos de roca con descripciones jerárquicas; en ambos, las entidades etiquetadas están conectadas solo a través de **nodos hub de alto grado** que codifican una feature. El paper conjetura que la **constante de normalización fija** es parcialmente culpable, ya que es problemática para nodos de grado muy alto, y sugiere reemplazarla por un **mecanismo de atención** (pesos $a_{ij,r}$ dependientes de los datos, con $\sum_{j,r} a_{ij,r}=1$) como dirección futura —una anticipación directa de los Graph Attention Networks.

### 4.2. Link prediction

Se evalúa sobre **FB15k** (subconjunto de Freebase, 14.951 entidades, 1.345 relaciones), **WN18** (subconjunto de WordNet, 40.943 entidades, 18 relaciones) y **FB15k-237** (14.541 entidades, 237 relaciones). El paper destaca un defecto serio de FB15k y WN18 detectado por Toutanova y Chen (2015): contienen **pares de tripletas inversas** $t=(e_1, r, e_2)$ y $t'=(e_2, r^{-1}, e_1)$ con $t$ en train y $t'$ en test, lo que reduce gran parte de la tarea a **memorización**. Por eso FB15k-237, con esos pares removidos, es el dataset primario de evaluación. El baseline `LinkFeat` (un clasificador lineal sobre features de relaciones observadas) explota justamente esa fuga y domina en FB15k/WN18, pero **fracasa en generalizar en FB15k-237**.

Configuración: para FB15k/WN18, basis decomposition con 2 funciones base y una sola capa de codificación de 200 dimensiones; para FB15k-237, **block decomposition** con 2 capas, bloques $5\times5$ y embeddings de 500 dimensiones. Se regulariza con **edge dropout** (0.2 para self-loops, 0.4 para el resto), lo que hace el objetivo similar al de un denoising autoencoder, más $\ell_2$ sobre el decoder. La normalización que mejor funcionó fue $c_{i,r}=c_i=\sum_r |\mathcal{N}_i^r|$ (aplicada *a través de* los tipos de relación). Se reporta también **R-GCN+**, un ensemble entre R-GCN y un DistMult entrenado por separado: $f^{\text{R-GCN+}} = \alpha f^{\text{R-GCN}} + (1-\alpha) f^{\text{DistMult}}$ con $\alpha=0.4$.

Resultados en **FB15k-237** (el dataset clave), MRR y Hits@n:

| Modelo | MRR (filt.) | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| DistMult | 0.191 | 0.106 | 0.207 | 0.376 |
| **R-GCN** | **0.248** | 0.153 | 0.258 | 0.414 |
| R-GCN+ | 0.249 | 0.151 | 0.264 | 0.417 |
| ComplEx | 0.201 | 0.112 | 0.213 | 0.388 |

Aquí R-GCN **supera al baseline DistMult por un 29.8%** en MRR filtrado, el número titular del paper. La lección: cuando la información se elimina de la memorización (FB15k-237) y debe inferirse de la estructura, **el encoder relacional aporta valor real**. R-GCN y R-GCN+ rinden parecido en este dataset (la información local no se solapa con la de un decoder puro), confirmando la predicción del paper.

En **FB15k y WN18**, R-GCN y R-GCN+ superan a DistMult pero, como todos los demás, quedan por debajo de LinkFeat (que explota las inversas). Un hallazgo interesante: R-GCN+ supera a ComplEx en FB15k aunque su decoder (DistMult) **no modela asimetría de relaciones** —lo que sugiere combinar el encoder R-GCN con un decoder ComplEx como trabajo futuro. La Figura 4 ofrece el análisis diagnóstico más revelador: R-GCN supera a DistMult **en nodos de alto grado**, donde abunda el contexto, lo que motiva el ensemble (los dos modelos son complementarios).

## 5. Limitaciones reconocidas

- **Constante de normalización fija.** El propio paper la señala como culpable del bajo rendimiento en MUTAG/BGS y de la fragilidad en nodos de alto grado; propone reemplazarla por atención (anticipando GAT).
- **Explosión de parámetros como riesgo latente.** Aunque las descomposiciones lo mitigan, el modelo sigue siendo sensible al número de relaciones, y la elección entre basis y block decomposition es un hiperparámetro no trivial que cambia por dataset.
- **Escalabilidad.** R-GCN opera en full-batch (gradiente sobre el grafo entero); el paper reconoce que para grafos muy grandes habría que explorar técnicas de subsampling como las de GraphSAGE (Hamilton, Ying y Leskovec 2017).
- **Decoder simple.** DistMult no modela relaciones asimétricas; el paper deja abierto incorporar decoders más expresivos (ComplEx).
- **Transformaciones lineales en los mensajes.** El mensaje es una transformación lineal $W_r h_j$; el paper menciona que podrían usarse funciones más flexibles (MLPs) a costa de eficiencia, pero lo deja para trabajo futuro.
- **Featureless.** En los experimentos las entidades entran como one-hot, sin features de nodo; integrar features sería directo y beneficioso, pero no se explora.

## 6. Impacto: GNN para grafos de conocimiento

R-GCN es uno de los trabajos fundacionales de la intersección entre GNN y knowledge graphs. Su aporte duradero es haber mostrado que **el marco de message passing/GCN se puede aplicar a datos relacionales** introduciendo pesos por tipo de arista, y que esa estructura *enriquece* a los modelos de embeddings de KG (DistMult, ComplEx, TransE) que hasta entonces optimizaban un vector por entidad sin tener en cuenta la vecindad. El patrón **encoder relacional + decoder de factorización** se volvió una receta estándar para *KG embeddings con estructura*. Las dos técnicas de regularización (basis y block decomposition) se citan constantemente como la forma canónica de domar la explosión de parámetros en GNN relacionales, y la conjetura sobre atención prefigura directamente los Graph Attention Networks. R-GCN también dejó la familia de "GNN relacionales" (CompGCN y sucesores) y es la base de aplicaciones en extracción de relaciones, recomendación y razonamiento sobre grafos heterogéneos.

## 7. Conexión con la Clase 27 (Redes Neuronales de Grafos)

La Clase 27 presenta R-GCN como **la aplicación concreta** del marco teórico de GNN que desarrolla, y usa exactamente el ejemplo de la Figura 1 del paper: el grafo de **Mikhail Baryshnikov**, donde aparecen las relaciones `educated_at` (hacia la **Vaganova Academy**), `award` (hacia el Vilcek prize), `citizen_of`/`country` (hacia **Rusia**/U.S.A.) y el tipo `ballet_dancer`. La clase plantea sobre ese grafo las dos tareas que estructuran el paper: **link prediction** de relaciones faltantes (inferir que `(Baryshnikov, lived_in, Russia)` debe existir) y **entity classification** (inferir que Vaganova Academy es `university`). Este es el puente narrativo entre la abstracción del message passing y un caso de uso tangible.

La conexión conceptual más profunda es con la noción de **tipo de arista en el cálculo del mensaje** que la clase introduce. La clase explica que, en su forma más general, la función de mensaje $f$ debe poder depender no solo del nodo origen y el nodo destino, sino también del **tipo de arista** que los conecta —es decir, $f(\text{origen}, \text{tipo}, \text{destino})$. R-GCN es la **realización canónica** de esa idea: la transformación $W_r$ depende explícitamente de la relación $r$ (el tipo) y de su dirección. Donde el GCN homogéneo de la clase usa una sola $W$, R-GCN usa una $W_r$ por tipo, lo que conecta directamente con el diagnóstico de la sección 2 de este análisis.

Esa misma idea aparece en la clase a través de las **Gated Graph Neural Networks (GGNN)**, que asocian una matriz $E_k$ a cada tipo de arista $k$ para transformar los mensajes antes de agregarlos en la unidad recurrente. Las $E_k$ de las GGNN y las $W_r$ de R-GCN son la *misma* decisión de diseño —pesos por tipo de arista— en dos linajes distintos (GRU recurrente vs. convolución espectral/espacial). Entender R-GCN a la luz de las GGNN ayuda al estudiante a ver que "pesos por tipo de relación" no es un truco aislado del paper de Schlichtkrull, sino el patrón recurrente con que las GNN incorporan **heterogeneidad de aristas**. La diferencia clave que la clase puede subrayar: R-GCN, al multiplicar relaciones por matrices densas, *necesita* la regularización (basis/block decomposition) que las GGNN —con su número típicamente pequeño de tipos de arista— no requieren; es el precio de operar sobre knowledge graphs con cientos o miles de relaciones.
