---
title: "GAT: Graph Attention Networks (2018)"
weight: 304
math: true
---

{{< paper-card
    title="Graph Attention Networks"
    authors="Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio"
    year="2018"
    venue="ICLR 2018"
    pdf="/papers/gat-velickovic-2018.pdf"
    arxiv="1710.10903" >}}
Paper canónico de las **redes neuronales de grafos** que introduce la atención por arista en las GNN. Donde [GCN](/papers/gcn-kipf-2017) pondera a cada vecino con un peso fijo y estructural ($1/\sqrt{d_i d_j}$), GAT **aprende** ese peso con un mecanismo de atención que depende del contenido de las features: $\alpha_{ij} = \mathrm{softmax}_j\big(\mathrm{LeakyReLU}(\vec{a}^{\,T}[W\vec{h}_i \,\Vert\, W\vec{h}_j])\big)$. No requiere eigendescomposiciones ni la matriz de adyacencia completa, así que es **inductivo** (generaliza a grafos no vistos). Iguala o supera el estado del arte en Cora, Citeseer, Pubmed y PPI. Es la pieza que tiende el puente entre GNN y Transformers: **un Transformer es un GAT sobre el grafo completo**.
{{< /paper-card >}}

> **Nota sobre la Clase 27.** GAT es un paper canónico de las GNN, pero **no aparece citado explícitamente en las slides de la [Clase 27](/clases/clase-27)**. Se incluye aquí como lectura complementaria porque completa el panorama que la clase construye: donde la clase presenta GCN (promedio ponderado por grado) y GraphSAGE (agregadores fijos), GAT introduce la tercera opción —aprender los pesos de combinación con atención— que hoy es difícil de omitir al enseñar [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos).

---

## Contexto

Las redes convolucionales triunfaron en datos con estructura de **rejilla** (imágenes, audio): reutilizan filtros locales con pesos compartidos sobre todas las posiciones. Pero muchos dominios viven en un dominio **irregular** que se representa como grafo —redes sociales, redes biológicas, mallas 3D, conectomas— donde no existe una noción canónica de "píxel a la derecha". El reto de las GNN es generalizar la convolución a grafos conservando el reparto de pesos y la localidad.

Hacia 2017 el campo tenía dos grandes líneas, ambas con un defecto.

La **línea espectral** define la convolución en el dominio de Fourier del grafo, vía la eigendescomposición del Laplaciano (Bruna et al., 2014; Defferrard et al., 2016 con expansiones de Chebyshev; Kipf & Welling, 2017 con [GCN](/papers/gcn-kipf-2017)). El defecto es estructural: los filtros aprendidos **dependen de la base propia del Laplaciano**, que depende de la estructura del grafo. Un modelo entrenado sobre un grafo no se transfiere a otro de estructura distinta —son intrínsecamente **transductivos**.

La **línea no-espectral** define convoluciones directamente sobre el grafo. GraphSAGE (Hamilton et al., 2017) es **inductivo** —muestrea un vecindario de tamaño fijo y aplica un agregador (mean, pooling o LSTM)— pero arrastra dos compromisos: el muestreo de tamaño fijo impide ver el vecindario completo en inferencia, y el agregador LSTM asume un **orden secuencial** que un vecindario no tiene.

El hueco que GAT llena se ve con nitidez. Los métodos espectrales aprenden pesos pero dependen del Laplaciano (no inductivos). Los no-espectrales son inductivos pero ponderan a los vecinos de forma **fija o estructural** —GCN normaliza por el grado, GraphSAGE promedia o agrupa— sin dejar que la importancia de un vecino dependa del contenido de sus features. Faltaba un mecanismo de **pesos aprendidos, dependientes del contenido, por arista, y compatible con el régimen inductivo**.

Mientras tanto, la atención (Bahdanau et al., 2015; Gehring et al., 2016) se había vuelto estándar de facto en tareas secuenciales: permite lidiar con inputs de tamaño variable, enfocándose en las partes más relevantes. Cuando la atención calcula la representación de una sola secuencia se llama *self-attention*; Vaswani et al. (2017) habían mostrado que self-attention sola basta para construir un modelo de traducción de estado del arte. GAT toma exactamente ese mecanismo —apenas meses después de *Attention is all you need*, y antes de que los Transformers dominaran el campo— y lo lleva al vecindario de un grafo. En retrospectiva, GAT es el momento en que la atención cruza desde las secuencias hacia los grafos arbitrarios.

## Contribución: el graph attentional layer

El corazón de GAT es una capa que computa la nueva representación de cada nodo atendiendo sobre sus vecinos con self-attention. Tres propiedades la hacen atractiva: es **eficiente** (paralelizable a través de los pares nodo-vecino), maneja **vecindarios de grado arbitrario** y es directamente aplicable a problemas **inductivos**.

El coeficiente de atención normalizado entre el nodo $i$ y su vecino $j$ es:

$$\alpha_{ij} = \frac{\exp\!\big(\mathrm{LeakyReLU}(\vec{a}^{\,T}[W\vec{h}_i \,\Vert\, W\vec{h}_j])\big)}{\sum_{k\in\mathcal{N}_i}\exp\!\big(\mathrm{LeakyReLU}(\vec{a}^{\,T}[W\vec{h}_i \,\Vert\, W\vec{h}_k])\big)}$$

donde $W \in \mathbb{R}^{F' \times F}$ es una transformación lineal **compartida** entre todos los nodos, $\Vert$ es concatenación, $\vec{a} \in \mathbb{R}^{2F'}$ es el vector de pesos del mecanismo de atención, y la normalización softmax es sobre el vecindario $\mathcal{N}_i$. Tres rasgos lo distinguen de todo lo previo: el peso $\alpha_{ij}$ se **aprende por arista** y depende del contenido de ambos extremos; el cómputo **no requiere la matriz de adyacencia ni el Laplaciano** —solo saber quiénes son los vecinos de $i$—; y al ser un mecanismo compartido aplicado a todas las aristas, transfiere a grafos nuevos.

GAT puede reformularse como una instancia particular de MoNet (Monti et al., 2016), con la diferencia clave de que usa **features de los nodos** para las similitudes, no propiedades estructurales —lo que evita conocer la estructura de antemano.

## El método en detalle

La capa recibe features $h = \{\vec{h}_1, \dots, \vec{h}_N\}$ con $\vec{h}_i \in \mathbb{R}^F$ y produce $h' = \{\vec{h}'_1, \dots, \vec{h}'_N\}$ con $\vec{h}'_i \in \mathbb{R}^{F'}$. Los pasos:

1. **Transformación lineal compartida.** Se aplica $W$ a cada nodo, preservando el *weight sharing* convolucional.
2. **Coeficientes sin normalizar.** Un mecanismo $a$ compartido computa $e_{ij} = a(W\vec{h}_i, W\vec{h}_j)$, la importancia de las features de $j$ para $i$.
3. **Masked attention.** Aquí entra la topología: en vez de atender sobre todos los nodos, GAT solo computa $e_{ij}$ para $j \in \mathcal{N}_i$, los **vecinos de primer orden** de $i$ (incluyéndose a sí mismo). La atención se restringe a las aristas que existen. Consecuencia elegante: el grafo **no necesita ser no-dirigido** —basta omitir $\alpha_{ij}$ si la arista $j \to i$ no existe.
4. **Normalización.** Softmax sobre el vecindario, para que los coeficientes sean comparables entre nodos.
5. **Combinación.** Los $\alpha_{ij}$ ponderan una combinación lineal de las features transformadas, con no-linealidad $\sigma$:

$$\vec{h}'_i = \sigma\!\left(\sum_{j\in\mathcal{N}_i} \alpha_{ij}\, W\vec{h}_j\right).$$

El mecanismo $a$ concreto es una red feedforward de **una sola capa** con activación **LeakyReLU** (pendiente negativa 0.2): se concatenan las proyecciones de los dos nodos, se proyecta sobre $\vec{a}$ para obtener un escalar y se pasa por LeakyReLU antes del softmax.

### Multi-head attention

Para estabilizar el aprendizaje, GAT ejecuta $K$ mecanismos de atención independientes (cada uno con su propio $\vec{a}^k$ y $W^k$) y **concatena** sus salidas:

$$\vec{h}'_i = \big\Vert_{k=1}^{K} \sigma\!\left(\sum_{j\in\mathcal{N}_i} \alpha_{ij}^{k}\, W^k\vec{h}_j\right),$$

de modo que la salida tiene $K F'$ features. En la **capa final de predicción**, concatenar cambiaría la dimensión de salida, así que GAT **promedia** las $K$ cabezas y retrasa la no-linealidad final (softmax o sigmoide) hasta después del promedio.

### Complejidad

La complejidad de una sola cabeza que computa $F'$ features es $O(|V|FF' + |E|F')$, **a la par con GCN**. No hay eigendescomposiciones. El multi-head multiplica parámetros por $K$, pero las cabezas son independientes y paralelizables. Existe una versión esparsa que reduce el almacenamiento a lineal en nodos y aristas, limitada por que el framework solo soportaba multiplicación esparsa para tensores de rango 2.

## Experimentos

GAT se evalúa en cuatro benchmarks consolidados, tres transductivos y uno inductivo.

**Redes de citas (transductivo).** Nodos = documentos, aristas = citas, features = bag-of-words. Solo **20 nodos por clase** para entrenar:

| Dataset | Nodos | Aristas | Clases | Features |
|---|---|---|---|---|
| Cora | 2.708 | 5.429 | 7 | 1.433 |
| Citeseer | 3.327 | 4.732 | 6 | 3.703 |
| Pubmed | 19.717 | 44.338 | 3 | 500 |

**PPI — protein-protein interaction (inductivo).** 24 grafos de tejidos humanos (20 train / 2 validación / 2 test); los grafos de test permanecen **completamente no observados** durante el entrenamiento. ~2.372 nodos por grafo, 50 features, **121 etiquetas multietiqueta** por nodo.

**Arquitectura.** Transductivo: GAT de dos capas, primera con $K=8$ cabezas de $F'=8$ features (64 en total) + ELU, segunda de clasificación con una cabeza + softmax; regularización agresiva ($L_2$ con $\lambda=0.0005$, **dropout 0.6** sobre inputs y sobre los coeficientes de atención). Inductivo: tres capas, las dos primeras con $K=4$ cabezas de 256 features (1024 en total), capa final con $K=6$ cabezas promediadas + sigmoide, **skip connections** y sin dropout.

**Resultados transductivos** (accuracy media sobre 100 corridas):

| Método | Cora | Citeseer | Pubmed |
|---|---|---|---|
| GCN (Kipf & Welling, 2017) | 81.5% | 70.3% | 79.0% |
| MoNet (Monti et al., 2016) | 81.7 ± 0.5% | — | 78.8 ± 0.3% |
| **GAT** | **83.0 ± 0.7%** | **72.5 ± 0.7%** | **79.0 ± 0.3%** |

**Resultados inductivos** (micro-F1 sobre PPI, 10 corridas):

| Método | PPI |
|---|---|
| GraphSAGE-LSTM (Hamilton et al., 2017) | 0.612 |
| GraphSAGE* (mejor variante reajustada) | 0.768 |
| Const-GAT (atención constante) | 0.934 ± 0.006 |
| **GAT** | **0.973 ± 0.002** |

Las lecturas clave: GAT mejora a GCN en **1.5% (Cora)** y **1.6% (Citeseer)**, lo que sugiere que asignar pesos distintos a vecinos del mismo vecindario ayuda. En PPI la mejora es dramática: **+20.5% sobre el mejor GraphSAGE** que los autores pudieron reajustar —demostrando el potencial inductivo y el valor de observar el vecindario completo— y **+3.9% sobre Const-GAT**, la misma arquitectura pero con atención constante $a(x,y)=1$ (que asigna el mismo peso a cada vecino, esencialmente un operador inductivo tipo GCN). Esa última diferencia es la prueba *directa* de que la ganancia viene del mecanismo de atención, no de la arquitectura ni del mayor número de parámetros. Una visualización t-SNE de las representaciones de la primera capa de un GAT preentrenado en Cora muestra clustering discernible que corresponde a las siete clases, confirmando su poder discriminativo.

## Limitaciones

- **Batching restringido** por la implementación esparsa (solo tensores de rango 2), especialmente con datasets de múltiples grafos.
- **GPUs no siempre ayudan** en el régimen esparso, según la regularidad de la estructura del grafo.
- **Campo receptivo acotado por la profundidad**, como en GCN; las skip connections lo extienden parcialmente.
- **Cómputo redundante** al paralelizar de forma distribuida, porque los vecindarios se solapan fuertemente.
- **Solo clasificación de nodos.** Como trabajo futuro, los autores listan extender el método a clasificación de grafos (no solo de nodos), incorporar **features de aristas** (que indicarían relaciones entre nodos), manejar batches más grandes y aprovechar la atención para un análisis serio de interpretabilidad —que en el paper queda apenas esbozado, porque visualizar los coeficientes de Cora requiere conocimiento de dominio que los autores no abordan.

## Impacto: el puente entre GNN y Transformers

GAT responde la pregunta que GCN y GraphSAGE dejan abierta —*¿y si los pesos de combinación se aprendieran?*— y la responde con atención. En el vocabulario de las GNN, GAT toma la **función conmutativa de combinación** (invariante a permutaciones, porque un vecindario no tiene orden) y la vuelve **aprendible**: el softmax sobre el vecindario y la suma ponderada preservan la invariancia, pero los pesos dejan de ser un dato estructural y pasan a ser parámetros del modelo ajustados al contenido. Const-GAT es el puente conceptual: con atención constante, GAT colapsa a un operador tipo GCN inductivo.

La conexión más profunda es la que vuelve a GAT un puente y no un destino: **un Transformer es, esencialmente, un GAT sobre el grafo completo**. El $\alpha_{ij}$ de GAT es el [mecanismo de atención](/fundamentos/mecanismo-atencion) de Bahdanau trasladado de los pares query-key de una secuencia a los pares nodo-vecino de un grafo; el multi-head viene directamente del [Transformer](/papers/attention-is-all-you-need-vaswani-2017). Si se representa una secuencia de tokens como un grafo donde *todos* los nodos están conectados con *todos* ($\mathcal{N}_i$ = todos los demás tokens) y se aplica self-attention, se recupera la atención del Transformer. Visto al revés: la atención del Transformer es el caso degenerado de GAT en el que el grafo es un *clique* completo, y GAT es ese mismo mecanismo con *masked attention* que restringe el vecindario a las aristas que existen. Esta dualidad —anticipada al citar a Vaswani meses después de su publicación— ancla las GNN y los Transformers en un mismo marco y abre la línea de los **Graph Transformers** que dominaría la investigación posterior.

## Notas y enlaces

- Preprint: [arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903) (v3, feb. 2018).
- Código oficial (TensorFlow): [github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Inspiración directa: Bahdanau et al. (2015) para la atención base; Vaswani et al. (2017) para el multi-head.
