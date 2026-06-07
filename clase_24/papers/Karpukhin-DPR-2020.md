# Dense Passage Retrieval for Open-Domain Question Answering

> Análisis técnico exhaustivo para el Diplomado IA UC (PUC Chile) — Clase 24.
> Audiencia: practitioner de ML con foco en retrieval/matching. Las cifras de este documento están extraídas literalmente del texto del paper.

## 1. Metadata

| Campo | Detalle |
|---|---|
| Título | Dense Passage Retrieval for Open-Domain Question Answering |
| Autores | Vladimir Karpukhin*, Barlas Oğuz*, Sewon Min†, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen‡, Wen-tau Yih |
| Afiliaciones | Facebook AI; † University of Washington; ‡ Princeton University |
| Aporte igual (*) | Karpukhin y Oğuz contribuyeron por igual |
| Venue | EMNLP 2020 |
| arXiv | 2004.04906v3 [cs.CL], 30 Sep 2020 |
| Código y modelos | https://github.com/facebookresearch/DPR |
| Sigla del método | DPR (Dense Passage Retriever) |

El paper es notable por su simplicidad: no introduce una arquitectura nueva (usa BERT base estándar y un esquema dual-encoder que data de Bromley et al. 1994), sino que demuestra que con el *régimen de entrenamiento correcto* un retriever denso supera ampliamente a BM25 usando muy pocos pares pregunta–passage. Tres de los autores (Karpukhin, Lewis, Yih) son también coautores del paper de RAG (Lewis et al. 2020b), que construye directamente sobre DPR.

## 2. Contexto: open-domain QA = retriever + reader

El **open-domain question answering** (Voorhees, 1999) responde preguntas factoides usando una colección grande de documentos. Ejemplos del paper: *"Who first voiced Meg on Family Guy?"* o *"Where was the 8th Dalai Lama born?"*. El paper asume el setting **extractivo**: la respuesta es un span (subsecuencia de tokens) que aparece textualmente en uno o más passages del corpus.

Los sistemas tempranos de QA (Ferrucci 2012 — Watson de IBM; Moldovan et al. 2003) eran complicados y multicomponente. El avance de los modelos de reading comprehension permitió simplificar todo a un framework de dos etapas (Chen et al. 2017, DrQA):

1. **Retriever** $R: (q, C) \to C_F$ — toma una pregunta $q$ y un corpus $C$, y devuelve un subconjunto filtrado mucho más pequeño $C_F \subset C$ con $|C_F| = k \ll |C|$.
2. **Reader** — un modelo de machine reading que examina los $k$ passages recuperados e identifica el span de respuesta.

Formalmente, dada una colección de $D$ documentos $d_1, \dots, d_D$, se dividen en passages de longitud fija para formar el corpus $C = \{p_1, p_2, \dots, p_M\}$, donde cada passage $p_i$ es una secuencia de tokens $w_1^{(i)}, w_2^{(i)}, \dots, w_{|p_i|}^{(i)}$. La tarea es encontrar el span $w_s^{(i)}, w_{s+1}^{(i)}, \dots, w_e^{(i)}$ que responde la pregunta. El corpus puede ir de millones de documentos (Wikipedia) a miles de millones (la Web), por lo que el retriever eficiente es indispensable.

El **retriever tradicional era TF-IDF o BM25** (Robertson y Zaragoza, 2009): empareja keywords eficientemente con un índice invertido y puede verse como una representación de pregunta y contexto en vectores **sparse de alta dimensión** (con pesos). Su limitación es estructural: el matching es **léxico**, sobre tokens. No captura sinónimos ni paráfrasis — el problema de *vocabulary mismatch*.

El ejemplo canónico del paper: la pregunta *"Who is the bad guy in lord of the rings?"* se responde desde el contexto *"Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy."* Un sistema basado en términos tiene dificultad para recuperar ese contexto, porque "bad guy" y "villain" no comparten tokens. Un retriever denso, en cambio, mapea ambos a vectores cercanos y recupera el contexto correcto.

Otra ventaja conceptual de las representaciones densas: son **aprendibles**. Se puede ajustar la función de embedding para obtener una representación específica de la tarea, algo imposible con BM25 (que es un esquema fijo de pesos). Y con estructuras de datos en memoria e índices apropiados, la búsqueda densa se hace eficiente vía **maximum inner product search (MIPS)** (Shrivastava y Li 2014; Guo et al. 2016).

El obstáculo histórico era una creencia: que aprender una buena representación densa requiere *muchos* pares etiquetados pregunta–contexto. Por eso los métodos densos nunca habían superado a TF-IDF/BM25 en open-domain QA antes de **ORQA** (Lee et al. 2019), que introdujo el costoso *inverse cloze task* (ICT) como pre-entrenamiento adicional. Pero ORQA tenía dos debilidades: (1) el pre-entrenamiento ICT es computacionalmente intensivo y no es claro que las oraciones regulares sean buenos sustitutos de preguntas; (2) el encoder de contexto no se afina con pares pregunta–respuesta, dejando representaciones potencialmente subóptimas.

La pregunta central del paper: *¿podemos entrenar un mejor modelo de embedding denso usando solo pares de preguntas y passages, sin pre-entrenamiento adicional?*

## 3. Idea central: reemplazar BM25 por un retriever denso

DPR responde que sí. La solución, tras una serie de ablations cuidadosos, es "sorprendentemente simple": un **bi-encoder (dual-encoder)** basado en BERT que mapea preguntas y passages a vectores densos, optimizado para **maximizar el producto interno** entre la pregunta y su passage relevante, comparando *todos* los pares pregunta–passage dentro de un batch.

El resultado es contundente. DPR supera a BM25 por amplio margen (en el abstract: 9%–19% absoluto en top-20 retrieval accuracy a lo largo de varios datasets). En el setting Natural Questions:

- **Top-5 accuracy**: 65.2% (DPR) vs. 42.9% (BM25).
- **End-to-end QA exact match**: 41.5% (DPR) vs. 33.3% (ORQA).

Y lo logra entrenando con relativamente pocos ejemplos. La Figura 1 muestra que **DPR entrenado con solo 1,000 ejemplos ya supera a BM25** en Natural Questions, desmintiendo la creencia de que se necesitan grandes cantidades de pares etiquetados.

Las dos contribuciones declaradas:

1. Con el setup de entrenamiento adecuado, **simplemente afinar los encoders de pregunta y passage sobre pares existentes basta para superar ampliamente a BM25**; el pre-entrenamiento adicional (ICT de ORQA) puede no ser necesario.
2. En el contexto de open-domain QA, **mayor precisión de retrieval se traduce en mayor accuracy end-to-end de QA** — verifican empíricamente que la cadena retriever→reader propaga las mejoras.

## 4. Arquitectura del bi-encoder

DPR usa dos encoders separados:

- $E_P(\cdot)$ — encoder de passage, mapea cualquier texto a un vector real de $d$ dimensiones. Se aplica a los $M$ passages para construir el índice.
- $E_Q(\cdot)$ — encoder de pregunta, mapea la pregunta de entrada a un vector de $d$ dimensiones.

La **similitud** entre pregunta y passage es el producto punto de sus vectores:

$$\mathrm{sim}(q, p) = E_Q(q)^\top E_P(p) \tag{1}$$

En la implementación, ambos encoders son **dos redes BERT independientes** (base, uncased), tomando la representación del token `[CLS]` como salida, por lo que $d = 768$.

### Por qué dos torres y no un cross-encoder

Esta es la decisión arquitectónica clave, y la que más le importa a quien trabaja en matching. El paper es explícito: existen formas más expresivas de medir similitud entre pregunta y passage — por ejemplo, **redes con múltiples capas de cross-attention** (un cross-encoder, donde pregunta y passage se concatenan y se procesan juntos). Pero:

> la función de similitud necesita ser **descomponible** para que las representaciones de la colección de passages puedan precomputarse.

Esta es la restricción que define toda la arquitectura. Un cross-encoder calcula una función conjunta no factorizable $f(q, p)$: para puntuar un par hay que pasar pregunta y passage juntos por la red. Eso impide indexar offline — habría que ejecutar el modelo $M$ veces (una por passage) en cada query, lo que con 21 millones de passages es inviable en tiempo real.

El bi-encoder, en cambio, **factoriza** el cómputo: $E_P(p)$ no depende de $q$, así que todos los vectores de passage se precomputan una vez y se indexan. En tiempo de query solo se computa $E_Q(q)$ y se busca el vecino más cercano por producto interno. El passage encoder corre **offline**; el question encoder corre **online**. Esa asimetría temporal es la razón de ser de las dos torres.

El paper observa que la mayoría de las funciones de similitud descomponibles son transformaciones de la distancia euclidiana (L2): el coseno equivale al producto interno para vectores unitarios, y la distancia de Mahalanobis equivale a L2 en un espacio transformado. Como el ablation (Sección 5.2, Apéndice B) encuentra que otras funciones de similitud rinden de forma comparable, eligen la más simple — el **producto interno** — y concentran el esfuerzo en aprender mejores encoders. La filosofía del paper: simplicidad arquitectónica, sofisticación en el entrenamiento.

## 5. Entrenamiento

Entrenar los encoders para que el producto punto (Ec. 1) sea una buena función de ranking es esencialmente un problema de **metric learning** (Kulis, 2013): construir un espacio vectorial donde los pares relevantes pregunta–passage tengan menor distancia (mayor similitud) que los irrelevantes.

### La loss: negative log-likelihood del positive

Los datos de entrenamiento son $m$ instancias:

$$D = \{\langle q_i, p_i^+, p_{i,1}^-, \cdots, p_{i,n}^- \rangle\}_{i=1}^{m}$$

Cada instancia tiene una pregunta $q_i$, un passage relevante (positivo) $p_i^+$, y $n$ passages irrelevantes (negativos) $p_{i,j}^-$. Se optimiza la **negative log-likelihood del passage positivo**:

$$L(q_i, p_i^+, p_{i,1}^-, \cdots, p_{i,n}^-) = -\log \frac{e^{\mathrm{sim}(q_i, p_i^+)}}{e^{\mathrm{sim}(q_i, p_i^+)} + \sum_{j=1}^{n} e^{\mathrm{sim}(q_i, p_{i,j}^-)}} \tag{2}$$

Esto es exactamente un softmax sobre los scores de similitud con la cross-entropy concentrada en el positivo — un objetivo contrastivo. Maximizar el numerador empuja el vector de la pregunta hacia su passage positivo; el denominador lo aleja de los negativos.

### El problema de los negativos

En retrieval, los positivos suelen estar disponibles explícitamente, pero los negativos hay que seleccionarlos de un pool gigantesco. La selección de negativos "a menudo se pasa por alto pero puede ser decisiva". Se consideran tres tipos:

1. **Random** — cualquier passage aleatorio del corpus.
2. **BM25** — top passages devueltos por BM25 que *no* contienen la respuesta pero coinciden con la mayoría de los tokens de la pregunta. Son negativos "duros" (hard negatives): léxicamente parecidos pero incorrectos.
3. **Gold** — passages positivos *de otras preguntas* en el set de entrenamiento.

El mejor modelo usa **gold passages del mismo mini-batch + un passage negativo de BM25**.

### In-batch negatives: el truco de eficiencia

Es la pieza central del esquema. Supóngase un mini-batch de $B$ preguntas, cada una con su passage relevante. Sean $Q$ y $P$ las matrices $(B \times d)$ de embeddings de preguntas y passages del batch. Entonces:

$$S = Q P^\top$$

es una matriz $(B \times B)$ de scores de similitud, donde cada fila corresponde a una pregunta emparejada con los $B$ passages del batch. De este modo se reutiliza el cómputo y se entrena efectivamente sobre $B^2$ pares $(q_i, p_j)$ por batch. Cualquier par $(q_i, p_j)$ es **positivo cuando $i = j$ y negativo en caso contrario**. Esto genera $B$ instancias de entrenamiento por batch, cada una con $B-1$ passages negativos.

La elegancia: los embeddings de passage ya se computaron para usarse como positivos de *sus* preguntas; reusarlos como negativos de las *demás* preguntas no cuesta nada extra. Un solo producto matricial $QP^\top$ produce todos los scores. Se incrementa enormemente el número de ejemplos de entrenamiento sin costo computacional adicional, y por eso **la accuracy mejora consistentemente al crecer el batch size**. El truco viene del setting full-batch (Yih et al. 2011) y se popularizó en mini-batch (Henderson et al. 2017; Gillick et al. 2019) para entrenar dual-encoders.

El modelo principal de los experimentos usa **batch size 128 + un negativo BM25 adicional por pregunta**, entrenado hasta 40 epochs (datasets grandes: NQ, TriviaQA, SQuAD) o 100 epochs (datasets pequeños: TREC, WQ), con learning rate $10^{-5}$ usando Adam, scheduling lineal con warm-up y dropout 0.1.

Para quien trabaja en matching, conviene notar la combinación final: **gold in-batch negatives (gratis, abundantes, mayormente "fáciles") + 1 hard negative de BM25 (caro de obtener pero informativo)**. Los in-batch dan volumen; el hard negative de BM25 fuerza al modelo a discriminar contra distractores léxicamente plausibles, que son justo donde un retriever ingenuo falla.

## 6. Búsqueda eficiente: FAISS para MIPS

En inferencia, $E_P$ se aplica a todos los passages y se indexan con **FAISS** (Johnson et al. 2017) offline — una librería open-source de búsqueda de similitud y clustering de vectores densos que escala a miles de millones de vectores. Dada una pregunta $q$ en tiempo de ejecución, se deriva su embedding $v_q = E_Q(q)$ y se recuperan los top $k$ passages cuyos embeddings están más cerca de $v_q$.

El corpus es el dump de Wikipedia en inglés del 20 de diciembre de 2018 (siguiendo a Lee et al. 2019), pre-procesado con el código de DrQA para extraer texto limpio (se eliminan tablas, infoboxes, listas y páginas de desambiguación), y luego dividido en **bloques disjuntos de 100 palabras** como unidades de retrieval. Esto da **21,015,324 passages** (≈21M). Cada passage se prepende con el título del artículo de Wikipedia más un token `[SEP]`. Los autores notaron que passages de longitud fija rinden mejor que párrafos naturales, y que el solapamiento entre passages no aporta ventaja.

### Eficiencia en tiempo de ejecución (Sección 5.4)

Perfilado en un servidor Intel Xeon E5-2698 v4 @ 2.20GHz con 512GB de memoria, índice FAISS en memoria (HNSW en CPU, 512 vecinos por nodo, profundidad de construcción 200, profundidad de búsqueda 128):

| Métrica | DPR (FAISS) | BM25 (Lucene/Java) |
|---|---|---|
| Throughput de query | **995.0 preguntas/s** (top-100) | 23.7 preguntas/s por thread de CPU |
| Construcción del índice | 8.5 h (FAISS, 1 servidor) + 8.8 h calcular embeddings (8 GPUs) | ~30 min (índice invertido) |

El trade-off es claro: **DPR es ~42× más rápido en query**, pero la **construcción del índice es mucho más cara** (computar 21M embeddings en GPUs y construir el HNSW). El embedding es paralelizable; la indexación invertida de Lucene es trivialmente barata. Para un sistema de producción esto importa: el costo de indexación es un costo único amortizable, pero la re-indexación tras re-entrenar el passage encoder es onerosa — razón por la cual el joint-training del Apéndice D *congela* el passage encoder.

## 7. Resultados

### Retrieval (top-k accuracy, Tabla 2)

Top-20 y Top-100 accuracy en los test sets (porcentaje de las top 20/100 passages recuperadas que contienen la respuesta). "Single" = DPR entrenado por dataset; "Multi" = entrenado combinando todos los datasets excepto SQuAD.

| Train | Retriever | NQ (20) | TriviaQA (20) | WQ (20) | TREC (20) | SQuAD (20) |
|---|---|---|---|---|---|---|
| None | BM25 | 59.1 | 66.9 | 55.0 | 70.9 | 68.8 |
| Single | DPR | **78.4** | **79.4** | **73.2** | 79.8 | 63.2 |
| Single | BM25+DPR | 76.6 | 79.8 | 71.0 | **85.2** | 71.5 |
| Multi | DPR | 79.4 | 78.8 | 75.0 | **89.1** | 51.6 |
| Multi | BM25+DPR | 78.0 | 79.9 | 74.7 | 88.5 | 66.2 |

| Train | Retriever | NQ (100) | TriviaQA (100) | WQ (100) | TREC (100) | SQuAD (100) |
|---|---|---|---|---|---|---|
| None | BM25 | 73.7 | 76.7 | 71.1 | 84.1 | 80.0 |
| Single | DPR | 85.4 | 85.0 | 81.4 | 89.1 | 77.2 |
| Multi | DPR | 86.0 | 84.7 | 82.9 | 93.9 | 67.6 |

Lecturas clave:

- DPR supera a BM25 en **todos los datasets excepto SQuAD**. La brecha es mayor cuando $k$ es pequeño (78.4% vs. 59.1% en top-20 NQ): DPR ordena mejor los primeros resultados.
- **TREC** (el dataset más pequeño) se beneficia mucho del entrenamiento multi-dataset (79.8 → 89.1 en top-20). NQ y WQ mejoran modestamente; TriviaQA se degrada levemente.
- **SQuAD es la excepción.** DPR rinde peor que BM25. Dos razones conjeturadas: (1) los anotadores escribieron las preguntas *después* de ver el passage, generando alto solapamiento léxico que favorece a BM25; (2) los datos provienen de solo 500+ artículos de Wikipedia, sesgando fuertemente la distribución. En el setting Multi, SQuAD se excluye del entrenamiento, lo que explica su caída adicional (51.6 en top-20).
- El **híbrido BM25+DPR** (rerank de la unión de los top-2000 de cada uno usando $\mathrm{BM25}(q,p) + \lambda \cdot \mathrm{sim}(q,p)$ con $\lambda = 1.1$) ayuda en algunos casos, sobre todo donde BM25 es competitivo (TREC, SQuAD).

### End-to-end QA (Exact Match, Tabla 4)

El reader procesa hasta los top-100 passages recuperados; el span de la mejor passage es la respuesta final. EM tras normalización menor.

| Train | Modelo | NQ | TriviaQA | WQ | TREC | SQuAD |
|---|---|---|---|---|---|---|
| Single | ORQA (Lee et al. 2019) | 33.3 | 45.0 | 36.4 | 30.1 | 20.2 |
| Single | REALM_News (Guu et al. 2020) | 40.4 | — | 40.7 | 46.8 | — |
| Single | BM25 | 32.6 | 52.4 | 29.9 | 24.9 | **38.1** |
| Single | DPR | **41.5** | 56.8 | 34.6 | 25.9 | 29.8 |
| Single | BM25+DPR | 39.0 | 57.0 | 35.2 | 28.0 | 36.7 |
| Multi | DPR | 41.5 | 56.8 | 42.4 | 49.4 | 24.1 |
| Multi | BM25+DPR | 38.8 | **57.9** | 41.1 | **50.6** | 35.8 |

Lecturas clave:

- **Mayor accuracy de retrieval ⇒ mejor QA final**, en todos los casos excepto SQuAD: las respuestas extraídas de passages de DPR son más probablemente correctas que las de BM25. Esto valida la segunda contribución del paper.
- DPR establece **nuevo estado del arte en 4 de los 5 datasets**, con diferencias de 1% a 12% absoluto en EM.
- DPR **supera a ORQA y a REALM** (desarrollado concurrentemente) en NQ y TriviaQA, pese a que ambos usan pre-entrenamiento adicional y entrenamiento end-to-end costoso. Los autores conjeturan que el pre-entrenamiento adicional solo ayuda cuando los sets de entrenamiento objetivo son pequeños.
- El **reader procesa más passages** que ORQA (hasta 100 vs. 5), pero caben todos en un batch en una sola GPU de 32GB, manteniendo latencia ≈20ms (similar al caso de una passage). Con $k=50$ óptimo para NQ; $k=10$ da solo pérdida marginal (40.8 vs. 41.5 EM).
- **Pipeline vs. joint training**: una ablation con retriever y reader entrenados conjuntamente (siguiendo a Lee et al. 2019) da 39.8 EM en NQ, *peor* que el pipeline. Entrenar retriever y reader por separado, con un diseño más simple, aprovecha mejor la supervisión disponible.

## 8. Ablations (Sección 5.2, Tabla 3)

La Tabla 3 compara esquemas de entrenamiento por top-k retrieval accuracy en el dev set de NQ. `#N` = número de negativos, `IB` = in-batch.

| Tipo | #N | IB | Top-5 | Top-20 | Top-100 |
|---|---|---|---|---|---|
| Random | 7 | no | 47.0 | 64.3 | 77.8 |
| BM25 | 7 | no | 50.0 | 63.3 | 74.8 |
| Gold | 7 | no | 42.6 | 63.1 | 78.3 |
| Gold | 7 | sí | 51.1 | 69.1 | 80.8 |
| Gold | 31 | sí | 52.1 | 70.8 | 82.1 |
| Gold | 127 | sí | 55.8 | 73.0 | 83.1 |
| G.+BM25(1) | 31+32 | sí | 65.0 | 77.3 | 84.4 |
| G.+BM25(2) | 31+64 | sí | 64.5 | 76.4 | 84.0 |
| G.+BM25(1) | 127+128 | sí | **65.8** | **78.0** | **84.9** |

Hallazgos:

- **Tipo de negativos sin in-batch (bloque superior)**: random vs. BM25 vs. gold *no* importa mucho cuando $k \geq 20$ en el setting estándar 1-of-N.
- **In-batch negatives (bloque medio)**: con la misma configuración (7 gold negatives), in-batch mejora sustancialmente (Top-5: 42.6 → 51.1). La diferencia clave es si los gold negatives vienen del mismo batch o de todo el training set. In-batch produce más pares y más ejemplos de entrenamiento.
- **Batch size**: la accuracy mejora consistentemente al crecer el batch (Gold 7 → 31 → 127 in-batch: Top-5 51.1 → 52.1 → 55.8).
- **Hard negatives de BM25 (bloque inferior)**: agregar **un solo negativo BM25** mejora sustancialmente (Top-5 salta a 65.0–65.8). **Agregar dos no ayuda más** (G.+BM25(2): 64.5). Estos negativos BM25 sirven como negativos para *todas* las preguntas del batch.

Otros ablations:

- **Sample efficiency (Figura 1)**: DPR con 1,000 ejemplos ya supera a BM25 en NQ; más ejemplos (1k → 59k) mejoran consistentemente.
- **Gold vs. distant supervision (Apéndice A, Tabla 5)**: usar la passage top de BM25 que contiene la respuesta (distant supervision) en lugar del gold context degrada solo ~1 punto (top-20: 78.1 vs. 77.1). Esto es importante: para datasets que solo dan pares pregunta–respuesta (TREC, WQ, TriviaQA), la supervisión distante es suficiente.
- **Funciones de similitud y loss (Apéndice B, Tabla 6)**: L2 rinde comparable al producto punto, y ambos superan al coseno. La triplet loss no cambia mucho los resultados frente a NLL. Confirma la elección del producto punto + NLL por simplicidad.
- **Generalización cross-dataset**: DPR entrenado solo en NQ y aplicado directamente a WQ y TREC pierde 3–5 puntos en top-20 (69.9/86.3 vs. 75.0/89.1 fine-tuned), pero sigue superando claramente a BM25 (55.0/70.9). El espacio de embedding aprendido transfiere razonablemente.

## 9. Limitaciones

- **Requiere entrenamiento supervisado** con pares pregunta–passage. Aunque "pocos" (1k bastan para superar a BM25), no es zero-shot como BM25, que funciona out-of-the-box sobre cualquier corpus sin entrenar.
- **Indexación costosa**: 8.8h de cómputo de embeddings en 8 GPUs + 8.5h de construcción del índice FAISS, vs. ~30 min de Lucene. La **re-indexación tras re-entrenar el passage encoder** es tan cara que el joint-training (Apéndice D) lo evita congelando el passage encoder.
- **BM25 gana en matching léxico exacto raro.** El análisis cualitativo (Apéndice C, Tabla 7) lo ilustra: para *"Who plays Thoros of Myr in Game of Thrones?"*, la frase saliente "Thoros of Myr" es crítica y rara, y DPR no logra capturarla, mientras BM25 acierta. En cambio, para *"What is the body of water between England and Ireland?"*, DPR empareja "body of water" con vecinos semánticos como "sea" y "channel" sin solapamiento léxico, y acierta donde BM25 falla. Son **complementarios**: DPR para variación léxica/semántica, BM25 para keywords selectivos y entidades raras.
- **SQuAD** muestra que DPR puede ser peor cuando el dataset tiene alto solapamiento léxico pregunta–passage y distribución sesgada.
- El **reader sigue usando cross-attention** (no descomponible) sobre los top-k recuperados: DPR resuelve solo el cuello de botella del retrieval, no el del reranking fino.

## 10. Impacto

DPR se volvió un componente fundacional del retrieval neuronal moderno:

- **RAG** (Lewis et al. 2020b — coautores compartidos con DPR): combina DPR con modelos generativos como BART (Lewis et al. 2020a) y T5 (Raffel et al. 2019). El paper ya lo anticipa: "DPR puede combinarse con modelos de generación... logrando buen rendimiento en open-domain QA y otras tareas knowledge-intensive". DPR es el *retriever* sobre el que se construye toda la familia retrieval-augmented.
- **FiD / Fusion-in-Decoder** (Izacard y Grave 2020): leveraging passage retrieval con modelos generativos.
- **ANCE** (Xiong et al. 2020a): extiende la idea de hard negatives usando el modelo de retrieval de la iteración previa para descubrir nuevos negativos en cada iteración, *partiendo del modelo DPR entrenado* y mejorándolo.
- **ColBERT** (Khattab y Zaharia 2020): trabajo concurrente que, en lugar del dual-encoder, introduce un operador de *late interaction* sobre los encoders BERT — un punto medio entre bi-encoder y cross-encoder.
- Consolidó la **arquitectura retriever–reader** (y luego retriever–generator) como el patrón estándar de QA y de sistemas knowledge-intensive, y validó empíricamente que **mayor recall de retrieval propaga a mayor accuracy end-to-end**.

En términos generales, DPR es la prueba de concepto definitiva de que un **bi-encoder entrenado con in-batch negatives + hard negatives** es un retriever de primera categoría a escala de decenas de millones de ítems. Ese patrón trasciende QA: es el mismo que se usa hoy en búsqueda semántica, recomendación, deduplicación y entity matching.

## 11. Conexión con la Clase 24

El PDF de la clase cubre **IR-based Factoid QA** (slides 13–16): el pipeline clásico de *question processing → passage retrieval → answer processing*. DPR es la **versión moderna y neuronal de la etapa de passage retrieval** de ese pipeline.

- **Question processing** clásico: extraer keywords, tipo de respuesta esperada, query expansion. En DPR, todo esto se reemplaza por el question encoder $E_Q(q)$ — una sola pasada de BERT que produce el vector de query. No hay reglas ni expansión manual: la "comprensión" de la pregunta está latente en el embedding.
- **Passage retrieval** clásico: TF-IDF/BM25 sobre índice invertido, matching de keywords. DPR lo reemplaza por MIPS sobre embeddings densos vía FAISS. El cambio es de *matching léxico sparse* a *matching semántico denso*. La clase enseña BM25 como el de facto; DPR muestra cómo el retrieval neuronal lo supera resolviendo el vocabulary mismatch que la clase identifica como su limitación.
- **Answer processing** clásico: extracción de spans, ranking de candidatos. En DPR esto es el reader neuronal con cross-attention que asigna span scores y passage selection scores (Ecuaciones 3–5 del paper). El reader es la contraparte moderna del "answer processing" de la clase.

En síntesis: la clase enseña el esqueleto conceptual (retriever + reader, las tres etapas); DPR moderniza la etapa de retrieval reemplazando el componente sparse por uno denso aprendido, y demuestra que esa modernización mejora todo el pipeline.

## 12. Conexión con el trabajo de Roberto

DPR es directamente relevante para el **patient matching con bi-encoders**. La arquitectura es la misma idea: dos torres que mapean entidades a un espacio vectorial común, donde la cercanía implica relevancia/coincidencia.

- **Bi-encoder como blocker.** En patient matching, el embedding bi-encoder funciona como *blocker*: reduce el espacio de candidatos de millones a un top-k manejable, exactamente como DPR reduce 21M passages a top-20/100 para el reader. La analogía es precisa: DPR no resuelve el matching fino (eso lo hace el reader con cross-attention), solo el recall a escala. En el stack de MDM, el bi-encoder es el blocker y el scorer fino (el GBM/XGBoost) es el análogo del reader.
- **In-batch negatives** son aplicables tal cual: si se entrena un bi-encoder de pacientes con pares positivos (registros que son la misma persona), los demás del batch sirven gratis como negativos. El $QP^\top$ del paper es el mismo truco.
- **Hard negatives de BM25.** El hallazgo más transferible: un solo hard negative (un registro léxicamente parecido pero de otra persona — mismo nombre, distinta fecha de nacimiento) mejora dramáticamente la discriminación, mientras que más negativos fáciles aportan poco. En patient matching, los hard negatives serían near-duplicates que NO son la misma persona — justo los casos donde un blocker ingenuo falla.
- **Complementariedad densa/sparse.** El análisis cualitativo de DPR (gana en variación semántica, pierde en tokens raros salientes) refleja el dilema clásico de matching: los embeddings capturan variantes de escritura y abreviaturas, pero un identificador exacto raro (un RUT, un número de ficha) lo maneja mejor un match léxico/determinístico. El híbrido BM25+DPR es el argumento a favor de un pipeline de matching híbrido (determinístico + probabilístico/embedding), que es exactamente la arquitectura de tiers del MDM.
- **Costo de re-indexación.** La lección de congelar el passage encoder para no re-indexar (Apéndice D) aplica a cualquier sistema con índice vectorial de embeddings de pacientes: re-entrenar el encoder obliga a re-embeber todo el corpus, un costo operacional a considerar.

## 13. Notas y enlaces

- **Paper**: Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020. arXiv:2004.04906v3.
- **Código y modelos**: https://github.com/facebookresearch/DPR
- **Datasets**: Natural Questions (Kwiatkowski et al. 2019), TriviaQA (Joshi et al. 2017), WebQuestions (Berant et al. 2013), CuratedTREC (Baudiš y Šedivỳ 2015), SQuAD v1.1 (Rajpurkar et al. 2016).
- **Tamaños de train (preguntas usadas para entrenar DPR, Tabla 1)**: NQ 58,880; TriviaQA 60,413; WQ 2,474; TREC 1,125; SQuAD 70,096.
- **Antecedentes directos**: dual-encoder / red siamesa (Bromley et al. 1994); DSSM para Web search (Huang et al. 2013); dense entity retrieval (Gillick et al. 2019); ORQA con ICT (Lee et al. 2019).
- **Infraestructura**: BERT base uncased ($d=768$); FAISS (Johnson et al. 2017) con índice HNSW; MIPS (Shrivastava y Li 2014; Guo et al. 2016).
- **Descendientes**: RAG (Lewis et al. 2020b), FiD (Izacard y Grave 2020), ANCE (Xiong et al. 2020a), ColBERT (Khattab y Zaharia 2020), REALM (Guu et al. 2020).
- **Cifras emblemáticas**: Top-5 NQ 65.2% vs. 42.9% BM25; EM end-to-end NQ 41.5% vs. 33.3% ORQA; throughput 995 q/s (DPR) vs. 23.7 q/s/thread (BM25); 21,015,324 passages; mejora absoluta de retrieval 9%–19% en top-20.
