---
title: "GCN: Graph Convolutional Networks (2017)"
weight: 300
math: true
---

{{< paper-card
    title="Semi-Supervised Classification with Graph Convolutional Networks"
    authors="Thomas N. Kipf, Max Welling"
    year="2017"
    venue="ICLR 2017"
    pdf="/papers/gcn-kipf-2017.pdf"
    arxiv="1609.02907" >}}
El paper fundacional de las [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos). Su tesis es deceptivamente simple: se puede clasificar nodos de un grafo —documentos en una red de citas, entidades en un grafo de conocimiento— con una red neuronal que opera directamente sobre el grafo, condicionando cada capa sobre la matriz de adyacencia. La pieza central es una regla de propagación capa-a-capa, $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)})$, que el paper no inventa por intuición sino que **deriva como aproximación de primer orden de las convoluciones espectrales sobre grafos**. El modelo escala linealmente en el número de aristas y aprende representaciones que codifican a la vez la estructura local del grafo y los features de los nodos. Es la GNN más citada de la historia y la base de GraphSAGE, GAT y GIN.
{{< /paper-card >}}

---

## Contexto: clasificación semi-supervisada en grafos

El problema es clasificar nodos de un grafo cuando solo una fracción pequeña tiene etiqueta. La tradición clásica lo resolvía **suavizando las etiquetas a lo largo del grafo** mediante regularización explícita:

$$\mathcal{L} = \mathcal{L}_0 + \lambda \mathcal{L}_{\text{reg}}, \quad \mathcal{L}_{\text{reg}} = \sum_{i,j} A_{ij}\, \lVert f(X_i) - f(X_j)\rVert^2 = f(X)^\top \Delta f(X),$$

donde $\mathcal{L}_0$ es la pérdida supervisada sobre los nodos etiquetados y $\Delta = D - A$ es el Laplaciano del grafo ($A$ la adyacencia, $D$ la matriz diagonal de grados). El término $\mathcal{L}_{\text{reg}}$ penaliza que nodos conectados reciban predicciones distintas. Esta familia —label propagation (Zhu et al., 2003), manifold regularization (Belkin et al., 2006), deep semi-supervised embedding (Weston et al., 2012)— descansa en el supuesto de que **nodos conectados comparten etiqueta**. Kipf y Welling lo critican: una arista puede señalar una relación que no es de similitud (un documento puede citar a otro para refutarlo), así que ese supuesto limita la capacidad de modelado.

Una segunda familia eran los **embeddings de grafos** inspirados en word2vec: DeepWalk (Perozzi et al., 2014), LINE (Tang et al., 2015) y node2vec (Grover & Leskovec, 2016) aprenden embeddings prediciendo el vecindario a partir de caminatas aleatorias. Su problema es de pipeline: son multi-etapa (generar caminatas, optimizar el embedding, entrenar un clasificador), donde cada etapa se afina por separado y es difícil optimizarlas en conjunto. Planetoid (Yang et al., 2016) mitigaba esto inyectando etiquetas durante el aprendizaje del embedding, y es el baseline más fuerte y la fuente de los splits experimentales que GCN adopta.

La tercera raíz es la **teoría espectral de grafos**. Las convoluciones espectrales fueron introducidas por Bruna et al. (2014) y aceleradas por **Defferrard et al. (2016)** con filtros localizados basados en polinomios de Chebyshev (ChebNet). GCN se posiciona como una **simplificación radical de ChebNet** que sacrifica generalidad espectral a cambio de escalabilidad y mejor desempeño en grafos grandes y dispersos.

## De la convolución espectral a la regla de propagación

La aportación teórica es mostrar que la regla de propagación, sorprendentemente simple, se deriva como aproximación de primer orden de las convoluciones espectrales localizadas. El recorrido tiene cuatro pasos.

**1 — Convolución espectral.** Una convolución sobre grafos se define como la multiplicación de una señal $x \in \mathbb{R}^N$ con un filtro $g_\theta = \text{diag}(\theta)$ en el dominio de Fourier del grafo: $g_\theta \star x = U g_\theta U^\top x$, donde $U$ son los autovectores del Laplaciano normalizado $L = I_N - D^{-1/2} A D^{-1/2} = U \Lambda U^\top$. El problema: multiplicar por $U$ cuesta $O(N^2)$ y la descomposición espectral de $L$ es prohibitiva en grafos grandes.

**2 — Aproximación de Chebyshev (ChebNet).** Hammond et al. (2011) mostraron que $g_\theta(\Lambda)$ se aproxima por una expansión truncada en polinomios de Chebyshev $T_k$ hasta orden $K$: $g_{\theta'} \star x \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{L})\, x$, con la recursión $T_k(x) = 2x\,T_{k-1}(x) - T_{k-2}(x)$. La clave es que esta expresión es **$K$-localizada**: como es un polinomio de grado $K$ en el Laplaciano, depende solo de nodos a distancia máxima $K$, y evaluarla cuesta $O(|E|)$, lineal en las aristas.

**3 — Modelo lineal de primer orden ($K=1$).** Kipf y Welling limitan la convolución a $K=1$, una función lineal respecto al Laplaciano. La intuición: apilando muchas de estas capas se recupera una clase rica de filtros sin atarse a la parametrización de Chebyshev, y se alivia el sobreajuste a estructuras locales en grafos con distribuciones de grado muy anchas. Aproximando $\lambda_{\max} \approx 2$, la convolución queda con dos parámetros: $g_{\theta'} \star x \approx \theta'_0 x - \theta'_1 D^{-1/2} A D^{-1/2} x$.

**4 — Un parámetro y el renormalization trick.** Para reducir parámetros se impone $\theta = \theta'_0 = -\theta'_1$:

$$g_\theta \star x \approx \theta \left(I_N + D^{-1/2} A D^{-1/2}\right) x.$$

Aquí surge el problema que motiva el truco más famoso del paper. La matriz $I_N + D^{-1/2} A D^{-1/2}$ tiene autovalores en $[0, 2]$; aplicarla repetidamente al apilar capas profundas provoca **inestabilidades numéricas y gradientes que explotan o se desvanecen**. El **truco de renormalización** la reemplaza por una versión equivalente pero estable:

$$I_N + D^{-1/2} A D^{-1/2} \;\longrightarrow\; \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}, \quad \tilde{A} = A + I_N, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}.$$

Es decir: se añaden **auto-conexiones** ($\tilde{A} = A + I_N$, una arista de cada nodo consigo mismo) y se renormaliza simétricamente con la matriz de grados $\tilde{D}$ del grafo aumentado. Generalizando de una señal escalar a una matriz de features $X \in \mathbb{R}^{N \times C}$ con $F$ filtros, una capa queda $Z = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X \Theta$, con complejidad $O(|E| F C)$ porque $\tilde{A} X$ se implementa como producto disperso-denso.

## GCN de dos capas para clasificación de nodos

Con la capa definida, el modelo concreto es una **GCN de dos capas**. Se precalcula una sola vez la matriz normalizada $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ y el forward toma la forma compacta:

$$Z = f(X, A) = \text{softmax}\!\left(\hat{A}\;\text{ReLU}\!\left(\hat{A} X W^{(0)}\right) W^{(1)}\right).$$

Las dimensiones son el corazón de la comprensión: $X \in \mathbb{R}^{N \times C}$ son los features de entrada ($N$ nodos, $C$ canales —en las redes de citas, vectores bag-of-words dispersos del documento); $W^{(0)} \in \mathbb{R}^{C \times H}$ proyecta a la capa oculta ($H=16$ en las redes de citas, $64$ en NELL); $W^{(1)} \in \mathbb{R}^{H \times F}$ proyecta a las $F$ clases. La softmax se aplica por fila, dando una distribución sobre clases para cada nodo. La pérdida es la **entropía cruzada solo sobre los nodos etiquetados** $\mathcal{Y}_L$:

$$\mathcal{L} = -\sum_{l \in \mathcal{Y}_L} \sum_{f=1}^{F} Y_{lf} \ln Z_{lf}.$$

**Por qué es semi-supervisado.** La pérdida solo "ve" los pocos nodos etiquetados (en Cora, 20 por clase, ~5% de los nodos). Pero cada capa multiplica por $\hat{A}$, que mezcla cada nodo con sus vecinos. En una red de dos capas, la predicción de un nodo etiquetado depende de su vecindario de **segundo orden**, que incluye nodos sin etiqueta. Al hacer backprop, el gradiente de la pérdida supervisada **se distribuye por el grafo y actualiza representaciones de nodos sin etiqueta**. La estructura del grafo no entra como penalización en la pérdida (como en la ecuación clásica) sino que está horneada en la arquitectura vía $\hat{A}$. El entrenamiento es full-batch: cada iteración procesa el dataset entero, con memoria $O(|E|)$ usando representación dispersa.

**Lectura Weisfeiler-Lehman.** El paper ofrece una segunda lectura: la regla es una generalización diferenciable del algoritmo Weisfeiler-Lehman de 1 dimensión (WL-1), el clásico test de isomorfismo de grafos. Reemplazando su hash por una capa neuronal $h_i^{(l+1)} = \sigma(\sum_{j \in \mathcal{N}_i} \tfrac{1}{c_{ij}} h_j^{(l)} W^{(l)})$ y eligiendo $c_{ij} = \sqrt{d_i d_j}$, se recupera exactamente la regla de GCN en forma vectorial por nodo —que es precisamente la forma de [message passing](/fundamentos/message-passing). Como demostración, una GCN de 3 capas con pesos aleatorios y sin entrenar produce, sobre la red del club de karate de Zachary, embeddings comparables a los de DeepWalk: la estructura del grafo más la normalización correcta ya es un extractor de features potente.

## Experimentos

GCN sigue el protocolo de Yang et al. (2016): tres redes de citas y un grafo de conocimiento. Estadísticas de los datasets:

| Dataset | Tipo | Nodos | Aristas | Clases | Features | Label rate |
|---|---|---|---|---|---|---|
| Citeseer | Red de citas | 3 327 | 4 732 | 6 | 3 703 | 0.036 |
| Cora | Red de citas | 2 708 | 5 429 | 7 | 1 433 | 0.052 |
| Pubmed | Red de citas | 19 717 | 44 338 | 3 | 500 | 0.003 |
| NELL | Grafo de conocimiento | 65 755 | 266 144 | 210 | 5 414 | 0.001 |

En las redes de citas los nodos son documentos con features bag-of-words dispersas, y las aristas son citas tratadas como no dirigidas; se usan solo 20 etiquetas por clase para entrenar pero todos los vectores de features. NELL es extremo: **una sola etiqueta por clase**. El protocolo usa una GCN de dos capas, test de 1 000 nodos, validación de 500, optimizador Adam (lr 0.01), máximo 200 épocas, early stopping, dropout 0.5 y L2 de $5 \cdot 10^{-4}$.

**Resultados (accuracy en %, tiempo wall-clock entre paréntesis):**

| Método | Citeseer | Cora | Pubmed | NELL |
|---|---|---|---|---|
| ManiReg | 60.1 | 59.5 | 70.7 | 21.8 |
| DeepWalk | 43.2 | 67.2 | 65.3 | 58.1 |
| ICA | 69.1 | 75.1 | 73.9 | 23.1 |
| Planetoid | 64.7 (26s) | 75.7 (13s) | 77.2 (25s) | 61.9 (185s) |
| **GCN (este paper)** | **70.3 (7s)** | **81.5 (4s)** | **79.0 (38s)** | **66.0 (48s)** |
| GCN (splits aleatorios) | 67.9 ± 0.5 | 80.1 ± 0.5 | 78.9 ± 0.7 | 58.4 ± 1.7 |

GCN supera a todos los baselines, con margen amplio en Cora (81.5 vs. 75.7 de Planetoid) y NELL (66.0 vs. 61.9), y es mucho más rápido (4 s vs. 13 s en Cora; 48 s vs. 185 s en NELL). La fila de splits aleatorios (media ± error estándar sobre 10 particiones) confirma que el desempeño es robusto y no un artefacto de la partición de Yang et al.

**Ablación de la propagación.** El experimento más instructivo compara variantes de la capa. El filtro de Chebyshev completo ($K=3$: 79.5 en Cora; $K=2$: 81.2) no mejora al truco de renormalización (81.5). El modelo de dos parámetros da 80.0; el de un solo parámetro $(I_N + D^{-1/2}AD^{-1/2})X\Theta$ da 79.2; y el **truco de renormalización gana en los tres datasets** (70.3 / 81.5 / 79.0). Un MLP que ignora el grafo ($X\Theta$) se desploma a 46.5 / 55.1 / 71.4, cuantificando cuánto aporta la estructura. La simplificación —menos parámetros, más estabilidad— no solo es más barata sino mejor. En grafos sintéticos, el tiempo por época crece linealmente con las aristas, confirmando la complejidad $O(|E|)$.

## Limitaciones

El paper es honesto sobre tres límites, cada uno de los cuales abrió una línea de investigación entera.

**Naturaleza transductiva.** GCN aprende sobre un grafo fijo y completo: precalcula $\hat{A}$ una vez con todos los nodos presentes. No puede generar embeddings para nodos nuevos sin reentrenar, porque no aprende una función de agregación reutilizable. Esta es la limitación que **GraphSAGE** (Hamilton et al., 2017) atacaría aprendiendo funciones de agregación sobre vecindarios muestreados —el contraste transductivo vs. inductivo.

**Memoria full-batch.** Con descenso por lotes completos, la memoria crece linealmente con el dataset y los grafos grandes deben recurrir a CPU. El mini-batch SGD aliviaría esto, pero generar mini-batches debe almacenar el vecindario de orden $K$ de cada nodo, lo que explota en grafos densos —tensión que motivó el muestreo de vecindario.

**Over-smoothing con muchas capas.** Aunque el paper no usa el término moderno, lo documenta: los mejores resultados se obtienen con **2 o 3 capas**, y desde ~7 capas el entrenamiento se vuelve difícil sin conexiones residuales, porque las representaciones de todos los nodos tienden a colapsar hacia un valor común al promediar vecindarios cada vez más grandes. Los autores añaden conexiones residuales estilo ResNet para mitigarlo, anticipando todo un subcampo de GNN profundas. Además, el truco de renormalización asume igual importancia entre la auto-conexión y los vecinos; el paper sugiere un parámetro aprendible $\tilde{A} = A + \lambda I_N$ —idea que prefigura los pesos de atención de GAT.

## Impacto

GCN se convirtió en el paper de redes neuronales de grafos **más citado de la historia** y en la base de casi todo lo que vino después. Su contribución no fue solo un modelo con buen desempeño, sino una plantilla mental: la idea de que una capa de GNN es "agregar mensajes de los vecinos, transformar, no-linealidad", expresable como una multiplicación matricial dispersa, abrió la puerta al marco de message passing neural networks (Gilmer et al., 2017) que unifica casi todas las GNN. GraphSAGE generalizó la agregación al caso inductivo; GAT reemplazó la normalización fija por grado por atención aprendida; GIN (Xu et al., 2019) analizó la expresividad de GCN vía su conexión con Weisfeiler-Lehman, que el propio paper ya señalaba. El truco de renormalización con auto-conexiones se volvió estándar de facto, y la implementación pública en TensorFlow hizo de GCN el modelo con el que la comunidad aprendió a pensar sobre grafos.

## Enlaces

- [Clase 27 — Redes Neuronales de Grafos](/clases/clase-27)
- [Fundamento: Redes Neuronales de Grafos](/fundamentos/redes-neuronales-de-grafos)
- [Fundamento: Message Passing](/fundamentos/message-passing)
- [Dominio: Datos Estructurados](/dominios/estructurados)
- [Paper: GGNN — Gated Graph Neural Networks (2015)](/papers/ggnn-li-2015)
- [Paper: GAT — Graph Attention Networks (2018)](/papers/gat-velickovic-2018)
- Código original: [github.com/tkipf/gcn](https://github.com/tkipf/gcn)
