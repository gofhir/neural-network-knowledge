---
title: "R-GCN: Modeling Relational Data with GCNs (2018)"
weight: 306
math: true
---

{{< paper-card
    title="Modeling Relational Data with Graph Convolutional Networks"
    authors="Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling"
    year="2018"
    venue="ESWC 2018"
    pdf="/papers/rgcn-schlichtkrull-2018.pdf"
    arxiv="1703.06103" >}}
R-GCN lleva las [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos) al mundo de los **grafos de conocimiento**, donde no hay un solo tipo de arista sino cientos o miles de relaciones distintas. La idea central es simple y poderosa: en vez de una sola matriz de pesos $W$ compartida por todas las aristas (como en el GCN homogéneo), R-GCN usa **una matriz $W_r$ por tipo de relación**, de modo que el mensaje que un vecino envía depende de *cómo* está conectado y no solo de *que* está conectado. Para domar la explosión de parámetros propone dos regularizaciones (basis y block-diagonal decomposition), y para link prediction se enmarca como un **autoencoder**: encoder R-GCN + decoder DistMult. Su resultado titular es una mejora del **29.8%** sobre DistMult puro en FB15k-237, el benchmark más limpio.
{{< /paper-card >}}

---

## Contexto: grafos de conocimiento y los límites del GCN homogéneo

Las bases de conocimiento (DBpedia, Wikidata, Yago, Freebase) almacenan hechos como **tripletas** `(sujeto, predicado, objeto)` —por ejemplo `(Mikhail Baryshnikov, educated_at, Vaganova Academy)`. Aquí *Baryshnikov* y *Vaganova Academy* son **entidades**, `educated_at` es una **relación**, y las entidades llevan **tipos** (`person`, `university`, `ballet_dancer`). La representación natural es entonces un **multigrafo dirigido y etiquetado**: nodos = entidades, aristas etiquetadas = relaciones.

El problema motivador es que **incluso las bases de conocimiento más grandes están incompletas**, lo que daña aplicaciones río abajo (question answering, recuperación de información). El paper ataca las dos tareas canónicas del *statistical relational learning*:

- **Link prediction:** recuperar tripletas faltantes —inferir que `(Baryshnikov, lived_in, Russia)` debería pertenecer al grafo.
- **Entity classification:** recuperar tipos/atributos faltantes de las entidades —inferir que Vaganova Academy es `university`.

La intuición que el modelo explota es que **buena parte de la información faltante reside en la estructura de vecindad del grafo**: saber dónde estudió alguien condiciona qué tipo de entidad es y dónde probablemente vivió.

¿Por qué no usar [GCN](/papers/gcn-kipf-2017) directamente? Porque el GCN de Kipf y Welling (2017) agrega los vecinos con una **única** matriz de pesos $W$, asumiendo implícitamente un solo tipo de arista. Eso es razonable en un grafo homogéneo (red de citas, malla de píxeles), pero con una sola $W$ las relaciones `educated_at`, `lived_in` y `country` se mezclan indistinguiblemente. La semántica relacional —justo lo que hace útil a un grafo de conocimiento— se borra. R-GCN nace para reparar esa pérdida dentro del marco de [message passing](/fundamentos/message-passing).

## Contribución central: pesos por tipo de relación

La contribución de modelado es deliberadamente simple: **introducir transformaciones específicas por relación**. En vez de una sola $W$, hay una $W_r$ por relación $r$ (y por dirección), y la agregación se descompone en una suma anidada sobre relaciones y, dentro de cada una, sobre los vecinos bajo esa relación. La regla de propagación es:

$$h_i^{(l+1)} = \sigma\!\left( \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}}\, W_r^{(l)} h_j^{(l)} \;+\; W_0^{(l)} h_i^{(l)} \right)$$

Pieza por pieza:

- $\mathcal{N}_i^r$ son los vecinos del nodo $i$ **bajo la relación $r$**. Agrupar los vecinos por tipo de relación antes de transformarlos es lo que diferencia a R-GCN del GCN.
- $W_r^{(l)}$ es la **matriz de pesos específica de la relación $r$**: el corazón del modelo. El mensaje de un vecino se transforma según la relación que lo une al nodo, y según su **dirección**.
- $c_{i,r}$ es una **constante de normalización** (por defecto $c_{i,r} = |\mathcal{N}_i^r|$, el grado bajo esa relación).
- $W_0^{(l)} h_i^{(l)}$ es la **self-connection** (auto-conexión), que se implementa como una arista de tipo especial (self-loop) y asegura que el nodo conserve su propia representación.

**La dirección importa.** El conjunto $\mathcal{R}$ incluye cada relación en su dirección canónica (`born_in`) y en su inversa (`born_in_inv`), con matrices distintas. Apilar $L$ capas captura dependencias a $L$ pasos relacionales de distancia, y en la práctica se computa con multiplicaciones de matrices dispersas.

### El problema: explosión de parámetros

Una $W_r$ por relación significa que el número de parámetros **crece linealmente con el número de relaciones**. En grafos reales esto es catastrófico: FB15k tiene 1.345 relaciones, y cada $W_r$ es una matriz densa. Sin regularización aparecen dos males: **overfitting sobre relaciones raras** (que tienen pocas tripletas para ajustar su $W_r$) y **modelos de tamaño descomunal**.

### Dos regularizaciones

**(a) Basis decomposition.** Cada $W_r^{(l)}$ se escribe como combinación lineal de $B$ **transformaciones base** compartidas:

$$W_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} V_b^{(l)}$$

Las bases $V_b^{(l)}$ son **comunes a todas las relaciones**; lo único que depende de $r$ son los **coeficientes escalares** $a_{rb}^{(l)}$. Es weight sharing efectivo entre tipos de relación: las relaciones raras "heredan" estructura aprendida de las frecuentes, lo que **alivia el overfitting**.

**(b) Block-diagonal decomposition.** Cada $W_r^{(l)}$ es la suma directa de matrices pequeñas:

$$W_r^{(l)} = \bigoplus_{b=1}^{B} Q_{br}^{(l)}, \qquad Q_{br}^{(l)} \in \mathbb{R}^{(d^{(l+1)}/B) \times (d^{(l)}/B)}$$

Es decir, $W_r^{(l)}$ es **block-diagonal**: solo los bloques de la diagonal son no nulos. Impone **sparsidad** por relación, codificando que las características latentes se agrupan en conjuntos más acoplados entre sí. Aquí los parámetros no se comparten entre relaciones, pero cada una tiene muchos menos.

### Encoder R-GCN + decoder DistMult

Para link prediction, el modelo se plantea como un **graph autoencoder** (heredando del variational graph auto-encoder de Kipf y Welling, 2016):

- **Encoder:** una R-GCN que mapea cada entidad a un vector $e_i = h_i^{(L)} \in \mathbb{R}^d$. Esto distingue al modelo de los métodos de factorización clásicos, que optimizan un vector por entidad *directamente*, sin encoder: aquí el encoder acumula evidencia sobre la vecindad relacional en varios pasos.
- **Decoder:** **DistMult** (Yang et al., 2014), que asocia a cada relación una matriz diagonal $R_r$ y puntúa una tripleta como:

$$f(s, r, o) = e_s^\top R_r\, e_o$$

DistMult es simple y efectivo; la elección del decoder es ortogonal a la del encoder, así que podría sustituirse por ComplEx o HolE. El entrenamiento usa **negative sampling**: por cada tripleta observada se muestrean $\omega$ negativas corrompiendo sujeto u objeto, y se minimiza una cross-entropy logística que empuja a puntuar las tripletas reales más alto que las corrompidas.

Para **entity classification** el setup es más directo: se apilan capas R-GCN con un softmax por nodo en la salida y se minimiza la cross-entropy sobre los nodos etiquetados. Si no hay features de nodo, la entrada es un **vector one-hot por nodo**.

## Experimentos y resultados

### Entity classification

Sobre cuatro datasets RDF de escalas muy dispares —**AIFB** (8.285 entidades, 45 relaciones), **MUTAG**, **BGS** y **AM** (1.666.764 entidades, 133 relaciones)—, con un R-GCN de **2 capas, 16 unidades ocultas y basis decomposition**. Un detalle metodológico clave: se eliminan las relaciones que filtran las etiquetas (`employs` en AIFB, `isMutagenic` en MUTAG). Accuracy promedio sobre 10 corridas:

| Modelo | AIFB | MUTAG | BGS | AM |
|---|---|---|---|---|
| Feat | 55.55 | 77.94 | 72.41 | 66.66 |
| WL | 80.55 | 80.88 | 86.20 | 87.37 |
| RDF2Vec | 88.88 | 67.20 | 87.24 | 88.33 |
| **R-GCN** | **95.83** | 73.23 | 83.10 | **89.29** |

R-GCN logra **estado del arte en AIFB y AM**, pero queda atrás en **MUTAG y BGS**. La explicación honesta del paper: en esos datasets las entidades etiquetadas se conectan solo a través de **nodos hub de alto grado**, y la **constante de normalización fija** es problemática para grados muy altos. El paper sugiere reemplazarla por un **mecanismo de atención** —anticipando directamente los Graph Attention Networks.

### Link prediction

Sobre **FB15k** (14.951 entidades, 1.345 relaciones), **WN18** (40.943 entidades, 18 relaciones) y **FB15k-237** (14.541 entidades, 237 relaciones). El paper destaca un defecto serio de FB15k/WN18: contienen **pares de tripletas inversas** repartidas entre train y test, lo que reduce la tarea a **memorización**. Por eso FB15k-237 —con esos pares removidos— es el benchmark primario. La configuración para FB15k-237 fue **block decomposition** con 2 capas, bloques $5\times5$ y embeddings de 500 dimensiones, regularizado con **edge dropout** (estilo denoising autoencoder).

Resultados en **FB15k-237** (MRR filtrado y Hits@n):

| Modelo | MRR (filt.) | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| DistMult | 0.191 | 0.106 | 0.207 | 0.376 |
| **R-GCN** | **0.248** | 0.153 | 0.258 | 0.414 |
| R-GCN+ | 0.249 | 0.151 | 0.264 | 0.417 |
| ComplEx | 0.201 | 0.112 | 0.213 | 0.388 |

Aquí R-GCN **supera a DistMult por un 29.8%** en MRR filtrado, el número titular del paper. La lección: cuando la información debe inferirse de la estructura y no de la memorización, **el encoder relacional aporta valor real**. En FB15k/WN18 R-GCN supera a DistMult pero queda por debajo del baseline `LinkFeat`, que explota justamente las inversas. El análisis diagnóstico más revelador: R-GCN supera a DistMult **en nodos de alto grado**, donde abunda el contexto, lo que motiva el ensemble R-GCN+ (combinación lineal de R-GCN y un DistMult entrenado por separado).

## Limitaciones reconocidas

- **Constante de normalización fija.** Señalada como culpable del bajo rendimiento en MUTAG/BGS y de la fragilidad en nodos de alto grado; el paper propone reemplazarla por atención (anticipando GAT).
- **Sensibilidad al número de relaciones.** Aunque las descomposiciones lo mitigan, elegir entre basis y block decomposition es un hiperparámetro no trivial que cambia por dataset.
- **Escalabilidad.** R-GCN opera en full-batch; para grafos enormes habría que explorar subsampling al estilo GraphSAGE.
- **Decoder simple.** DistMult no modela relaciones asimétricas; queda abierto incorporar decoders más expresivos (ComplEx).
- **Mensajes lineales y featureless.** El mensaje es una transformación lineal $W_r h_j$ y las entidades entran como one-hot, sin features de nodo; ambas extensiones se dejan para trabajo futuro.

## Impacto

R-GCN es uno de los trabajos fundacionales de la intersección entre GNN y grafos de conocimiento. Su aporte duradero fue mostrar que **el marco de message passing/GCN se puede aplicar a datos relacionales** introduciendo pesos por tipo de arista, y que esa estructura *enriquece* a los modelos de embeddings de KG (DistMult, ComplEx, TransE). El patrón **encoder relacional + decoder de factorización** se volvió una receta estándar, las dos regularizaciones (basis y block) son la forma canónica de domar la explosión de parámetros en GNN relacionales, y la conjetura sobre atención prefigura los Graph Attention Networks. R-GCN dejó tras de sí toda la familia de GNN relacionales (CompGCN y sucesores), con aplicaciones en extracción de relaciones, recomendación y razonamiento sobre [grafos heterogéneos](/dominios/estructurados).

## Por qué importa para la Clase 27

La [Clase 27](/clases/clase-27) presenta R-GCN como **la aplicación concreta** del marco de [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos), usando exactamente el grafo de **Mikhail Baryshnikov** para plantear link prediction y entity classification sobre un caso tangible. La conexión conceptual más profunda es con la idea de que la función de mensaje debe poder depender del **tipo de arista**, no solo de los nodos origen y destino: R-GCN es la **realización canónica** de esa idea, donde el GCN homogéneo usa una sola $W$ y R-GCN una $W_r$ por tipo. Es la misma decisión de diseño que las matrices $E_k$ por tipo de arista de las [Gated Graph Neural Networks](/papers/ggnn-li-2015) —pesos por tipo de relación— en dos linajes distintos. La diferencia que conviene subrayar: R-GCN, al multiplicar cientos de relaciones por matrices densas, *necesita* la regularización (basis/block decomposition) que las GGNN, con sus pocos tipos de arista, no requieren. Es el precio de operar sobre grafos de conocimiento reales.

## Notas y enlaces

- arXiv: [1703.06103](https://arxiv.org/abs/1703.06103) (v4, 26 oct 2017).
- Venue: Extended Semantic Web Conference (ESWC) 2018 — best student research paper.
- Linaje: Kipf y Welling firman el GCN original (ICLR 2017) y el variational graph auto-encoder (2016); R-GCN fusiona ambas líneas para datos multi-relacionales.
