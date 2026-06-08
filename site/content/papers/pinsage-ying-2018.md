---
title: "PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
weight: 252
math: true
---

{{< paper-card
    title="Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
    authors="Ying, He, Chen, Eksombatchai, Hamilton, Leskovec"
    year="2018"
    venue="KDD 2018"
    pdf="/papers/pinsage-ying-2018.pdf"
    arxiv="1806.01973" >}}
**PinSage** es el sistema de recomendación canónico de **Pinterest**: la primera red convolucional sobre grafos (**GCN**) desplegada en producción a escala web, sobre un grafo de **3 mil millones de nodos** y **18 mil millones de aristas** (pins y boards). Reemplaza el costoso Laplaciano completo por **convoluciones localizadas vía caminatas aleatorias**, agrega un **importance pooling** que pondera vecinos por Personalized PageRank (+46%) y un **entrenamiento por currículo** con *hard negatives* crecientes (+12%). Resultado: **+40% en ranking offline**, ~**60% de preferencia humana** y **30–100% de mejora en engagement** en tests A/B. Es la mayor aplicación de embeddings profundos de grafo de su época.
{{< /paper-card >}}

---

## Contexto

Hacia 2018, las redes convolucionales sobre grafos (GCN) dominaban los *benchmarks* de aprendizaje sobre grafos, pero esas ganancias no llegaban a producción. El cuello de botella era de escala: todos los recomendadores GCN existentes requerían operar sobre el **Laplaciano completo del grafo** durante el entrenamiento, algo inviable con miles de millones de nodos y estructura cambiante. Métodos como node2vec o DeepWalk tampoco servían: son no supervisados, ignoran features de nodo y su número de parámetros crece linealmente con el grafo.

PinSage parte de **GraphSAGE** (Hamilton, Ying, Leskovec, 2017), la variante inductiva de las GCN que muestrea vecindarios. Lo reingenieriza para producción: elimina el requisito de tener el grafo completo en memoria GPU, sustituyendo el muestreo por **caminatas aleatorias** de baja latencia dentro de una arquitectura productor-consumidor. El escenario es Pinterest: más de **2 mil millones de pins** (marcadores visuales) organizados en más de **1 mil millones de boards**, conectados por **18 mil millones de aristas**. El grafo resultante es ~10.000× más grande que las aplicaciones típicas de GCN de la época.

## Ideas principales

### Convolución localizada

El núcleo es la operación `convolve`. Para un nodo $u$ con vecindario $\mathcal{N}(u)$:

1. Agregar vecinos transformados con una media ponderada simétrica $\gamma$:
$$\mathbf{n}_u = \gamma\big(\{\mathrm{ReLU}(\mathbf{Q}\mathbf{h}_v + \mathbf{q}) \mid v \in \mathcal{N}(u)\},\ \boldsymbol{\alpha}\big)$$
2. Concatenar con la representación propia y transformar:
$$\mathbf{z}_u^{\text{new}} = \mathrm{ReLU}\big(\mathbf{W}\cdot \mathrm{concat}(\mathbf{z}_u, \mathbf{n}_u) + \mathbf{w}\big)$$
3. Normalizar a norma unitaria: $\mathbf{z}_u^{\text{new}} / \lVert \mathbf{z}_u^{\text{new}} \rVert_2$.

La **concatenación** (en vez de la media de Kipf-Welling) da ganancias claras, y la normalización L2 estabiliza el entrenamiento y acelera la búsqueda aproximada de vecinos. Los parámetros se **comparten entre nodos** (difieren solo entre capas), por lo que la complejidad paramétrica es **independiente del tamaño del grafo**. En producción se apilan $K=2$ capas.

### Vecindarios e importance pooling

En vez de vecindarios de $k$-saltos completos, PinSage define $\mathcal{N}(u)$ como los $T$ nodos más influyentes sobre $u$: simula caminatas aleatorias desde $u$ y toma los top-$T$ por **conteo de visitas normalizado por $L_1$**. En el límite, estos conteos aproximan los scores de **Personalized PageRank** respecto de $u$. Esos mismos conteos son los pesos $\boldsymbol{\alpha}$ de la media ponderada: eso es el **importance pooling**, que aporta una ganancia de **46%** en métricas offline. Ventaja adicional: un número fijo $T$ de vecinos acota la huella de memoria.

### Pérdida max-margin y hard negatives

Entrenamiento supervisado con pares $(q,i)\in\mathcal{L}$ relacionados, usando pérdida de margen máximo:
$$J_\mathcal{G}(\mathbf{z}_q,\mathbf{z}_i) = \mathbb{E}_{n_k \sim P_n(q)}\ \max\{0,\ \mathbf{z}_q\cdot\mathbf{z}_{n_k} - \mathbf{z}_q\cdot\mathbf{z}_i + \Delta\}$$

Se comparten **500 negativos** por minibatch. Pero 500 negativos aleatorios de un catálogo de 2 mil millones dan resolución de solo 1/500, demasiado fácil. Se agregan **hard negatives**: ítems rankeados en posiciones **2000–5000** por Personalized PageRank respecto de $q$, lo bastante parecidos para forzar discriminación fina. Véase [/fundamentos/aprendizaje-contrastivo](/fundamentos/aprendizaje-contrastivo).

### Entrenamiento por currículo

Usar hard negatives desde el inicio duplica las épocas para converger. En cambio: **época 1 sin hard negatives** (el modelo halla rápido una zona de baja pérdida) y en la **época $n$ se agregan $n-1$** hard negatives por ítem. Ganancia del **12%**.

### Escalabilidad de sistemas

- **Productor-consumidor:** el grafo y las features (miles de millones de nodos) viven en CPU; por re-indexado se arma un subgrafo $G'$ por minibatch que se carga a GPU sin comunicación CPU-GPU durante `convolve`. El productor (CPU) prepara la iteración $n{+}1$ mientras el consumidor (GPU) corre la $n$, **reduciendo el tiempo casi a la mitad**.
- **Inferencia MapReduce:** evita recomputar vecindarios solapados; cada embedding se calcula **una sola vez**. Embeddings de **3 mil millones de ítems en menos de 24 horas**.
- **Servicio:** KNN aproximado vía **LSH** + operador Weak AND sobre embeddings precalculados.

## Resultados experimentales

Setup: $K=2$, dimensión oculta $m=2048$, embedding $d=1024$; features por pin = visual VGG-16 (4.096 dim) + texto Word2Vec (256 dim) + log-grado. Entrenamiento sobre **7,5 mil millones de ejemplos** (1,2 mil millones de pares positivos), 16 GPU Tesla K80, inferencia en cluster de 378 nodos AWS.

**Evaluación offline** — hit-rate (top K=500 sobre 5M pins) y MRR escalado:

| Método | Hit-rate | MRR |
|---|---|---|
| Visual | 17% | 0,23 |
| Annotation | 14% | 0,19 |
| Combined | 27% | 0,37 |
| max-pooling | 39% | 0,37 |
| mean-pooling | 41% | 0,51 |
| mean-pooling-xent | 29% | 0,35 |
| mean-pooling-hard | 46% | 0,56 |
| **PinSage** | **67%** | **0,59** |

PinSage logra **+40% absoluto (150% relativo)** en hit-rate y **+22% absoluto (60% relativo)** en MRR sobre el mejor baseline. La distribución de similitud coseno de sus embeddings es la más dispersa (kurtosis **0,43** vs 1,20 visual y 2,49 annotation), dando más "resolución" y menos colisiones LSH. Sobre métricas de ranking, ver [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics).

**Estudios de usuario** — cabeza a cabeza:

| Comparación | Win | Lose | Draw | Fracción de victorias |
|---|---|---|---|---|
| PinSage vs Visual | 28,4% | 21,9% | 49,7% | 56,5% |
| PinSage vs Annotation | 36,9% | 14,0% | 49,1% | 72,5% |
| PinSage vs Combined | 22,6% | 15,1% | 57,5% | 60,0% |
| PinSage vs Pixie | 32,5% | 19,6% | 46,4% | 62,4% |

Entre los casos con opinión, ~**60%** prefieren PinSage.

**Test A/B de producción** (homefeed, métrica *repin rate*): **10–30% de mejora** sobre Annotation y Visual.

**Runtime:** batch 2048 es el más eficiente (48,8 h). El vecindario $T$ tiene retornos decrecientes: $T{=}10$ → 60%/0,51; $T{=}20$ → 63%/0,54; $T{=}50$ → 67%/0,59. Entrenar sobre subgrafo de 300M ítems reduce runtime **6×** sin perder hit-rate (modelo **inductivo**).

## Limitaciones reconocibles

- **No modela usuarios explícitamente:** genera embeddings de ítems; la personalización homefeed es por proximidad a pins recientes, sin embedding de usuario aprendido. Es esencialmente ítem-ítem.
- **Solo los pins tienen features** (no los boards), lo que obliga a un número par de capas; depende de features visuales/textuales de buena calidad.
- **Profundidad limitada** ($K=2$); mayor profundidad agravaría el solapamiento de vecindarios.
- **Negativos sesgados por Pixie:** los pares etiquetados vienen de engagement donde Pixie ya recomendaba, por lo que Pixie no aparece en la comparación offline.
- **Costo de infraestructura** (18 TB, 500 GB RAM, 16 K80, 378 nodos AWS) fuera del alcance de la mayoría.

## Por qué importa hoy

PinSage es el sistema de recomendación **GNN a escala web canónico** y el primer despliegue de embeddings profundos de grafo a escala de miles de millones de nodos en producción. Su receta —muestreo por caminatas aleatorias, importance pooling y currículo con hard negatives— se volvió estándar de facto en GNN industriales. Validó empíricamente que las GNN dan ganancias reales de engagement, no solo de benchmark, y prefiguró los sistemas de recomendación multimodales modernos al fusionar contenido (imagen+texto) con estructura de grafo. Ver [/fundamentos/recommender-systems](/fundamentos/recommender-systems).

## Conexión con la Clase 25

La [/clases/clase-25](/clases/clase-25) es un *case study* de recomendación multimodal en Pinterest, donde cada pin combina imagen y texto. PinSage es **el** recomendador de Pinterest a escala web, por lo que ancla la clase:

- **Pins como nodos:** los pins multimodales son nodos de un grafo bipartito pin-board del que PinSage genera embeddings.
- **Multimodalidad:** las features de entrada (visual 4096 + texto 256 + grado) muestran cómo fusionar imagen y texto antes de la convolución; combinar modalidades supera ~60% a cada una sola.
- **Recomendación por vecindad:** similitud en espacio de embeddings vía KNN aproximado con LSH sobre embeddings normalizados.
- **Estructura colaborativa:** los boards capturan curación colectiva, puente entre filtrado colaborativo y contenido.
- **Escala real:** contraste entre prototipo académico y sistema industrial (3B nodos, 18B aristas, A/B con repin rate).

## Notas y enlaces

- **PDF:** [/papers/pinsage-ying-2018.pdf](/papers/pinsage-ying-2018.pdf) · **arXiv:** [1806.01973](https://arxiv.org/abs/1806.01973) · **DOI:** [10.1145/3219819.3219890](https://doi.org/10.1145/3219819.3219890)
- **Base inductiva:** GraphSAGE (Hamilton, Ying, Leskovec, NIPS 2017).
- **Relacionado en el curso:** [/fundamentos/recommender-systems](/fundamentos/recommender-systems), [/fundamentos/aprendizaje-contrastivo](/fundamentos/aprendizaje-contrastivo), [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics), [/clases/clase-25](/clases/clase-25).
