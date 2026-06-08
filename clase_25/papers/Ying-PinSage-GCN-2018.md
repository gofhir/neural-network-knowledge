# Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSage)

**Autores:** Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, Jure Leskovec (Pinterest + Stanford University)
**Venue:** KDD 2018 (The 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Londres)
**arXiv:** 1806.01973
**DOI:** 10.1145/3219819.3219890

---

## 1. Contexto

A mediados de la década de 2010, las redes neuronales convolucionales sobre grafos (Graph Convolutional Networks, GCN) se habían establecido como el estado del arte en numerosos *benchmarks* de aprendizaje sobre grafos: clasificación de nodos, predicción de enlaces y completitud de matrices para sistemas de recomendación (por ejemplo, el benchmark MovieLens de Monti et al. y Berg et al.). La idea central de una GCN es aprender a agregar iterativamente información de características desde vecindarios locales del grafo usando redes neuronales: una sola "convolución" transforma y agrega información del vecindario de un salto (one-hop) de un nodo, y apilando varias convoluciones la información se propaga por regiones cada vez más lejanas del grafo. A diferencia de modelos puramente basados en contenido (como las RNN), las GCN aprovechan simultáneamente la información de contenido y la estructura del grafo.

Sin embargo, esos avances en *benchmarks* no se habían traducido a ganancias en entornos de producción reales. El obstáculo era de escala: todos los sistemas de recomendación basados en GCN existentes requerían operar sobre el **Laplaciano completo del grafo** durante el entrenamiento, supuesto inviable cuando el grafo tiene miles de millones de nodos y una estructura en evolución constante. Métodos clásicos de embeddings de grafo como node2vec y DeepWalk tampoco servían: son no supervisados, no incorporan características de nodo, y aprenden embeddings directamente por nodo, de modo que el número de parámetros crece linealmente con el tamaño del grafo, algo prohibitivo a esta escala.

El trabajo se ancla en **GraphSAGE** (Hamilton, Ying, Leskovec, NIPS 2017), la variante inductiva de las GCN que muestrea vecindarios en lugar de operar sobre el grafo entero. PinSage es, en esencia, una reingeniería de GraphSAGE para producción a escala web: elimina la restricción de almacenar el grafo completo en memoria GPU, sustituyendo el muestreo de vecindarios por **caminatas aleatorias** (random walks) de baja latencia dentro de una arquitectura productor-consumidor.

El escenario concreto es Pinterest, la mayor colección de imágenes curadas por usuarios del mundo: más de **2 mil millones de pins** (marcadores visuales de contenido en línea) organizados en más de **1 mil millones de boards** (tableros temáticos), conectados por más de **18 mil millones de aristas** (la pertenencia de cada pin a sus boards). PinSage se entrena sobre **7,5 mil millones de ejemplos** en un grafo de **3 mil millones de nodos** (pins + boards) y 18 mil millones de aristas, un grafo unas **10.000 veces más grande** que las aplicaciones típicas de GCN de la época. Es, según los autores, la mayor aplicación de embeddings profundos de grafo realizada hasta entonces.

## 2. Contribución central

PinSage es un *framework* GCN basado en caminatas aleatorias, altamente escalable, desplegado en producción en Pinterest. Su aporte se divide en dos familias de innovaciones:

**Innovaciones de escalabilidad:**

1. **Convoluciones al vuelo (on-the-fly):** en lugar de multiplicar matrices de características por potencias del Laplaciano completo, PinSage realiza convoluciones localizadas muestreando el vecindario de cada nodo y construyendo dinámicamente un grafo de cómputo. Esto elimina la necesidad de operar sobre el grafo entero.
2. **Construcción de minibatches productor-consumidor:** un productor ligado a CPU (con gran memoria) muestrea vecindarios y obtiene características; un consumidor ligado a GPU (TensorFlow) ejecuta el SGD sobre grafos de cómputo predefinidos, maximizando la utilización de GPU.
3. **Inferencia MapReduce eficiente:** una vez entrenado el modelo, un pipeline MapReduce distribuye el modelo para generar embeddings de miles de millones de nodos sin recomputaciones redundantes.

**Innovaciones algorítmicas / de calidad:**

4. **Convoluciones vía caminatas aleatorias:** se muestrea el grafo de cómputo con caminatas aleatorias cortas, lo que además asigna a cada nodo un *score de importancia*.
5. **Importance pooling:** la agregación del vecindario se pondera según medidas de similitud por caminata aleatoria, lo que produce una **ganancia de 46%** en métricas offline.
6. **Entrenamiento por currículo (curriculum training):** se alimentan ejemplos cada vez más difíciles (hard negatives crecientes), con una **ganancia de 12%**.

El resultado global: más de **40% de mejora** sobre el mejor baseline en métricas de ranking offline, preferencia humana de ~**60%** en evaluaciones cabeza a cabeza, y mejoras de **30% a 100%** en engagement en tests A/B de producción.

## 3. Método

### 3.1 Planteamiento del problema

Pinterest se modela como un **grafo bipartito** con dos conjuntos disjuntos de nodos: `I` (pins/ítems) y `C` (boards/contextos o colecciones). Cada pin `u ∈ I` lleva atributos de valor real `x_u ∈ R^d`: en Pinterest, características visuales y textuales ricas. El objetivo es generar embeddings de alta calidad de los pins que sirvan para recomendación, por ejemplo vía búsqueda de vecinos más cercanos (related-pin recommendation) o como features de un sistema de re-ranking aguas abajo. Por generalidad, el algoritmo se describe sobre el conjunto de nodos `V = I ∪ C` sin distinguir pins de boards salvo cuando es estrictamente necesario.

### 3.2 Arquitectura del modelo — convolución localizada

El núcleo computacional es la operación `convolve` (Algoritmo 1), que aprende a agregar información del vecindario `N(u)` de un nodo:

1. Se transforman las representaciones `z_v` de los vecinos `v ∈ N(u)` mediante una red densa con ReLU y se aplica una función de agregación/pooling simétrica `γ` (media ponderada), obteniendo `n_u = γ({ReLU(Q h_v + q) | v ∈ N(u)}, α)`, una representación del vecindario.
2. Se concatena `n_u` con la representación actual `z_u` del propio nodo y se transforma por otra capa densa: `z_u^new = ReLU(W · concat(z_u, n_u) + w)`. Los autores observan ganancias significativas usando **concatenación** en lugar de la media de Kipf-Welling.
3. Se **normaliza** a norma L2 unitaria: `z_u^new = z_u^new / ||z_u^new||_2`. Esto estabiliza el entrenamiento y hace más eficiente la búsqueda aproximada de vecinos.

**Vecindarios basados en importancia.** En vez de tomar vecindarios de k-saltos completos, PinSage define `N(u)` como los `T` nodos que más influyen sobre `u`. Para ello simula caminatas aleatorias desde `u` y calcula el **conteo de visitas normalizado por L1**; el vecindario son los top `T` nodos con mayor conteo. En el límite de simulaciones infinitas estos conteos aproximan los scores de **Personalized PageRank** respecto de `u`. Las ventajas: (a) un número fijo de vecinos controla la huella de memoria durante el entrenamiento; (b) los pesos de la media (`α` = conteos de visita normalizados) ponderan la importancia de cada vecino — esto es el **importance pooling**.

**Apilamiento de convoluciones.** Se apilan `K` capas de convolución; la entrada de la capa `k` depende de la salida de la capa `k−1`, y la "capa 0" son las features de entrada. Los parámetros (`Q, q, W, w`) se comparten entre nodos pero difieren entre capas. El Algoritmo 2 (minibatch) primero muestrea los vecindarios de los nodos del minibatch `M` (descendiendo de capa `K` a `1`), luego aplica `K` iteraciones convolucionales y finalmente pasa la representación final por una red densa (`G_1, G_2, g`) para generar el embedding de salida `z_u`. **Clave de escalabilidad:** como los parámetros se comparten, la complejidad paramétrica es independiente del tamaño del grafo.

### 3.3 Entrenamiento del modelo

**Función de pérdida — max-margin ranking.** Se entrena de forma supervisada con pares etiquetados `(q, i) ∈ L` que se asumen relacionados. La pérdida para un par es:

`J_G(z_q, z_i) = E_{n_k ~ P_n(q)} max{0, z_q · z_{n_k} − z_q · z_i + Δ}`

Busca que el producto interno del par positivo supere al de los negativos por al menos un margen `Δ`. `P_n(q)` es la distribución de negativos.

**Entrenamiento multi-GPU con minibatches grandes.** Forward/backward en estilo *multi-tower* sobre múltiples GPU; los gradientes se agregan y se aplica un paso de SGD síncrono. Tamaños de batch de **512 a 4096**. Se usa *warmup* gradual del learning rate (regla de escalado lineal de Goyal et al.) seguido de decaimiento exponencial.

**Construcción productor-consumidor de minibatches.** La lista de adyacencia y la matriz de features de miles de millones de nodos viven en **memoria CPU** (demasiado grandes para GPU). Mediante una técnica de **re-indexado** se crea un subgrafo `G' = (V', E')` con solo los nodos y vecindarios involucrados en el minibatch actual, junto con una matriz de features pequeña. Estos se cargan a GPU al inicio de cada iteración, de modo que **no hay comunicación CPU-GPU durante el paso convolve**. El productor (CPU: extracción de features, re-indexado, muestreo de negativos) ejecuta la iteración `n+1` en paralelo con el consumidor (GPU) ejecutando la iteración `n`, lo que **reduce el tiempo de entrenamiento casi a la mitad**.

**Muestreo de ítems negativos.** Por eficiencia se comparten **500 negativos** por minibatch entre todos los ejemplos (empíricamente igual de bueno que muestrear por nodo). Pero 500 negativos aleatorios de un catálogo de 2 mil millones dan una "resolución" de solo 1 entre 500, demasiado fácil: el modelo debe poder identificar 1 ítem entre ~2 millones. Por eso se agregan **hard negatives**: ítems algo relacionados con `q` pero no tanto como el positivo, obtenidos rankeando por Personalized PageRank respecto de `q` y muestreando aleatoriamente los rankeados en posiciones **2000–5000**. Esto fuerza al modelo a discriminar a granularidad fina.

**Entrenamiento por currículo.** Usar hard negatives desde el inicio duplica las épocas necesarias para converger. La solución (Bengio et al. 2009): en la **época 1 no hay hard negatives** (el modelo encuentra rápido una zona de baja pérdida), y en la **época `n` se agregan `n−1`** hard negatives por ítem. Ganancia del **12%**.

### 3.4 Embeddings de nodos vía MapReduce

Aplicar el Algoritmo 2 directamente para todos los ítems generaría **recomputaciones masivas** por el solapamiento de vecindarios de K-saltos. El pipeline MapReduce sobre el grafo bipartito pin-board tiene dos jobs por capa: (1) proyectar todos los pins al espacio latente de baja dimensión (agregación, línea 1 del Alg. 1); (2) unir esas representaciones con los IDs de los boards donde aparecen y calcular el embedding de cada board agrupando (pooling) sus vecinos muestreados. Cada vector latente **se computa solo una vez**. Se itera el proceso (dos jobs más para los embeddings de segunda capa de pins). Como solo los pins tienen features, se requiere un **número par de capas convolucionales**.

### 3.5 Búsqueda eficiente de vecinos más cercanos

En servicio, dado un pin de consulta `q`, se recomiendan sus K vecinos más cercanos en el espacio de embeddings. El KNN aproximado se obtiene con **Locality Sensitive Hashing (LSH)** y recuperación de dos niveles vía el operador **Weak AND**. Como el modelo se entrena offline y los embeddings se precalculan vía MapReduce y se guardan en base de datos, el servicio es online.

## 4. Experimentos

**Datos y setup.** Pares positivos `(q, i)` definidos por engagement histórico: usuario interactúa con `i` inmediatamente después de `q`. En total: **1,2 mil millones de pares positivos** + 500 negativos por batch + 6 hard negatives por pin = **7,5 mil millones de ejemplos de entrenamiento**. Se entrena sobre un subgrafo (20% de boards y los pins que tocan, 70% de los ejemplos etiquetados; 10% para tuning), y se infiere sobre el grafo completo (2 mil millones de pins). Datasets de entrenamiento/evaluación ~**18 TB**; embeddings de salida ~**4 TB**.

**Features.** Por pin se concatenan: embeddings visuales de **4.096 dim** (6ª capa FC de una VGG-16), embeddings de anotaciones textuales de **256 dim** (modelo Word2Vec sobre anotaciones), y el **log del grado** del nodo.

**Baselines:** Visual, Annotation, Combined (MLP de 2 capas sobre visual+texto), y **Pixie** (random-walk en producción de Pinterest). Variantes de ablación: max-pooling, mean-pooling, mean-pooling-xent (cross-entropy de GraphSAGE), mean-pooling-hard, y PinSage completo (importance pooling). Hiperparámetros: `K=2`, dimensión oculta `m=2048`, embedding `d=1024`.

**Recursos.** TensorFlow, 1 máquina con 32 cores y **16 GPU Tesla K80**, 500 GB de RAM (con Linux HugePages de 2 MB). Inferencia MapReduce en cluster Hadoop2 con **378 nodos d2.8xlarge de AWS**.

### 4.1 Evaluación offline (Tabla 1)

Hit-rate (fracción de queries donde el positivo `i` está entre los top K=500 NN sobre una muestra de 5 millones de pins) y MRR escalado (factor 100):

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

PinSage logra **67% hit-rate** y **0,59 MRR**: +40% absoluto (**150% relativo**) en hit-rate y +22% absoluto (**60% relativo**) en MRR sobre el mejor baseline. Combinar visual+texto supera en ~60% a cualquiera por separado.

**Distribución de similitud.** Los embeddings de PinSage tienen la distribución de similitud coseno **más dispersa** (kurtosis **0,43**, frente a 2,49 de annotation y 1,20 de visual), lo que da mayor "resolución" para distinguir ítems y reduce las colisiones de LSH en servicio.

### 4.2 Estudios de usuario (Tabla 2)

Comparación cabeza a cabeza (qué imagen es más relevante a la consulta):

| Comparación | Win | Lose | Draw | Fracción de victorias |
|---|---|---|---|---|
| PinSage vs Visual | 28,4% | 21,9% | 49,7% | 56,5% |
| PinSage vs Annotation | 36,9% | 14,0% | 49,1% | 72,5% |
| PinSage vs Combined | 22,6% | 15,1% | 57,5% | 60,0% |
| PinSage vs Pixie | 32,5% | 19,6% | 46,4% | 62,4% |

Entre los casos con opinión, ~**60%** prefieren PinSage. Cualitativamente, los embeddings visuales confunden semántica (plantas con comida, tala de árboles con fotos de guerra); Pixie acierta la categoría pero no los ítems más relevantes; PinSage combina similitud visual y temática. El t-SNE de 1000 ítems muestra agrupamiento coherente por contenido y tema.

### 4.3 Test A/B de producción

Tarea homefeed, métrica **repin rate** (porcentaje de recomendaciones que el usuario guarda en un board, acción de alto valor). PinSage logra **10–30% de mejora** en repin rate sobre las recomendaciones por Annotation y Visual.

### 4.4 Análisis de runtime (Tablas 3 y 4)

Por ser **inductivo**, PinSage computa embeddings de ítems no vistos en entrenamiento. Entrenar sobre un subgrafo de **300 millones de ítems** logra el mejor hit-rate y **reduce el runtime 6×** frente al grafo completo. El **batch size 2048** es el más eficiente (48,8 h totales). El tamaño de vecindario `T` muestra retornos decrecientes: `T=10` → 60%/0,51 (20 h); `T=20` → 63%/0,54 (33 h); `T=50` → 67%/0,59 (78 h). La inferencia para **3 mil millones de ítems se completa en menos de 24 horas** gracias a MapReduce.

## 5. Limitaciones reconocibles

- **No modela usuarios explícitamente:** PinSage genera embeddings de ítems; la personalización en homefeed se hace por proximidad a pins recientes del usuario, sin un embedding de usuario aprendido. Es esencialmente recomendación ítem-ítem.
- **Dependencia de features de calidad:** asume features visuales (VGG-16) y textuales (Word2Vec) ya buenas; solo los pins tienen features (no los boards), lo que obliga a un número par de capas.
- **Profundidad limitada:** `K=2` en producción; profundidades mayores no se exploran a fondo y agravarían el solapamiento de vecindarios.
- **Negativos sesgados por Pixie:** los pares etiquetados provienen de engagement donde Pixie ya era el recomendador, por lo que el baseline Pixie no aparece en la comparación offline (solo en estudios de usuario).
- **Costo de infraestructura:** 18 TB de datos, 500 GB de RAM, 16 K80, cluster de 378 nodos AWS — fuera del alcance de la mayoría de las organizaciones.
- **Grafo bipartito específico:** la formulación pin-board no traslada trivialmente a grafos heterogéneos más ricos (multi-relacionales, con atributos en aristas).

## 6. Impacto y legado

PinSage es el sistema de recomendación **canónico de GNN a escala web** y el primer despliegue de embeddings profundos de grafo a escala de miles de millones de nodos en producción. Estableció el patrón de muestreo por caminatas aleatorias + importance pooling + entrenamiento por currículo con hard negatives, hoy estándar de facto. Inspiró líneas posteriores en GNN escalables (sampling de vecindarios, GNN industriales), y validó empíricamente que las GNN aportan ganancias reales de engagement, no solo de benchmark. La combinación de contenido multimodal (imagen+texto) con estructura de grafo prefigura los sistemas de recomendación multimodales modernos. Junto a GraphSAGE (su base inductiva), PinSage es referencia obligada en cualquier curso de GNN aplicadas.

## 7. Conexión con la Clase 25

La Clase 25 es un *case study* de recomendación multimodal en Pinterest, donde cada pin combina imagen y texto. PinSage es **el** sistema de recomendación de Pinterest a escala web, por lo que es el ancla perfecta de la clase:

- **Pins como nodos del grafo:** la clase trata los pins como ítems multimodales; PinSage los modela como nodos de un grafo bipartito pin-board y genera sus embeddings.
- **Multimodalidad:** las features de entrada (visual VGG-16 4096-dim + texto Word2Vec 256-dim + grado) muestran cómo se fusiona imagen y texto antes de la convolución de grafo, mostrando empíricamente que combinar modalidades supera a cada una sola (~60%).
- **Recomendación por vecindad:** la clase enfatiza recomendación por similitud en espacio de embeddings; PinSage implementa esto con KNN aproximado vía LSH sobre embeddings normalizados.
- **Estructura social/colaborativa vía grafo:** los boards capturan la curación colectiva de usuarios, complementando el contenido puro — el puente entre filtrado colaborativo y contenido que la clase discute.
- **Escala real de producción:** la clase contrasta prototipos académicos con sistemas industriales; PinSage es el ejemplo máximo (3B nodos, 18B aristas, tests A/B con repin rate).

Para el dominio del curso (FHIR/MDM de Roberto), PinSage también ilustra el patrón **blocker + scorer**: las caminatas aleatorias + LSH actúan como recuperación de candidatos (blocking) y el embedding como score de relevancia — análogo a la arquitectura bi-encoder como blocker que se usa en *patient matching*.
