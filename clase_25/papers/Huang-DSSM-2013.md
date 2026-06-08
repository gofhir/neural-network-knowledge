# Learning Deep Structured Semantic Models for Web Search using Clickthrough Data (DSSM)

**Autores:** Po-Sen Huang (University of Illinois at Urbana-Champaign), Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, Larry Heck (Microsoft Research)
**Venue:** CIKM 2013 (San Francisco, octubre-noviembre 2013)
**Análisis interno exhaustivo — Clase 25 (RecSys multimodal / two-tower)**

---

## 1. Contexto

A comienzos de la década de 2010, los motores de búsqueda web recuperaban documentos principalmente por coincidencia de palabras clave: comparar los términos de la consulta con los términos del documento. El problema central que motiva este paper es la *discrepancia lingüística* (language discrepancy) entre consultas y documentos: un mismo concepto se expresa con vocabularios y estilos distintos en uno y otro lado. Un usuario escribe "ccra" y el documento relevante se titula "canada revenue agency website"; escribe "bfpo" y el documento es un artículo de Wikipedia sobre códigos postales del Reino Unido. La coincidencia léxica falla en estos casos, aunque la relación semántica sea evidente.

Los **modelos semánticos latentes** surgieron para atacar exactamente este problema. El representante clásico es **LSA (Latent Semantic Analysis)** [Deerwester et al., 1990]: mediante la descomposición en valores singulares (SVD) de una matriz documento-término, se mapea cada documento (o consulta) a un vector concepto de baja dimensión $\hat{D} = A^T D$, donde $A$ es la matriz de proyección. La relevancia se mide como la similaridad coseno entre los vectores concepto. La idea clave es que términos que aparecen en contextos similares quedan agrupados en el mismo *cluster* semántico, de modo que consulta y documento pueden tener alta similaridad aunque no compartan ningún término literal.

Sobre LSA se construyeron extensiones probabilísticas: **PLSA (Probabilistic LSA)** [Hofmann, 1999] y **LDA (Latent Dirichlet Allocation)** [Blei et al., 2003]. El problema señalado por los autores es que estos modelos de tópicos se entrenan de forma *no supervisada*, con una función objetivo que solo está débilmente acoplada con la métrica de evaluación del retrieval. El resultado: su rendimiento real en búsqueda web no estaba a la altura de lo esperado.

Aquí entran en juego los **clickthrough data** (datos de clics). Un log de clics es una lista de consultas con los documentos que los usuarios clicaron tras esa consulta. La hipótesis razonable: si un usuario clica un documento tras buscar algo, ese documento es al menos parcialmente relevante. Estos datos constituyen una señal de supervisión barata y masiva. El trabajo previo de Gao et al. (2010, 2011) explotó clickthrough con dos familias de modelos: los **Bi-Lingual Topic Models (BLTM)** —generativos, que exigen que consulta y documento clicado compartan distribución de tópicos— y los **Discriminative Projection Models (DPM)** —entrenados con el algoritmo S2Net siguiendo el paradigma de learning-to-rank por pares. Ambos superaron a LSA/PLSA, pero arrastraban dos defectos: BLTM se entrenaba maximizando log-verosimilitud (subóptimo para ranking), y DPM involucraba multiplicaciones de matrices cuyo tamaño crece con el vocabulario, obligando a podar agresivamente el vocabulario (lo que degrada el rendimiento).

La segunda línea de investigación previa fue el **deep learning para semántica**: Salakhutdinov y Hinton (2007) extendieron LSA con autoencoders profundos (semantic hashing, SH). Demostraron que se puede extraer estructura semántica jerárquica con redes profundas, pero su enfoque seguía siendo *no supervisado* (optimizado para reconstruir el documento, no para distinguir documentos relevantes de irrelevantes) y enfrentaba el mismo cuello de botella de escalabilidad: limitaban el vocabulario a las 2000 palabras más frecuentes.

El DSSM nace en la intersección de ambas líneas: traer el deep learning supervisado al espacio de modelos semánticos latentes, entrenado directamente sobre clickthrough con un criterio acoplado al ranking, y resolver el problema de vocabulario con una técnica nueva (word hashing).

## 2. Contribución

La contribución del paper se articula en tres ejes que los propios autores enumeran en las conclusiones:

1. **Entrenamiento discriminativo sobre clickthrough orientado a ranking.** A diferencia de LSA/PLSA/autoencoders no supervisados, todos los parámetros del modelo se optimizan apuntando directamente al objetivo de ranking de documentos: maximizar la verosimilitud condicional del documento clicado dada la consulta.

2. **Extensión no lineal profunda de los modelos semánticos lineales.** En lugar de una única proyección lineal (LSA), se usa una red neuronal profunda (DNN) con múltiples capas de proyección no lineal, lo que aumenta la capacidad de modelar estructuras semánticas más sofisticadas.

3. **Word hashing por letter n-grams.** Una técnica de reducción de dimensionalidad basada en n-gramas de letras que permite escalar el entrenamiento a vocabularios muy grandes (cientos de miles de palabras), algo esencial para búsqueda web real y que era el cuello de botella de los métodos previos.

La arquitectura conceptual es la que hoy llamamos **dual-encoder / two-tower**: dos "torres" (la red que procesa la consulta y la que procesa el documento) proyectan sus entradas a un *espacio semántico común de baja dimensión*, y la relevancia se calcula como una distancia (coseno) en ese espacio. Este es, precisamente, el ancestro directo de la arquitectura de la Clase 25.

## 3. Método

### 3.1 DNN para computar características semánticas

La entrada a la DNN es un vector de términos de alta dimensión: los recuentos crudos (bag-of-words sin normalizar) de los términos de una consulta o documento. La salida es un vector concepto en un espacio semántico de baja dimensión.

Denotando $x$ el vector de términos de entrada, $y$ el vector de salida, $l_i$ ($i = 1,\dots,N-1$) las capas ocultas intermedias, $W_i$ la i-ésima matriz de pesos y $b_i$ el i-ésimo sesgo:

$$l_1 = W_1 x$$
$$l_i = f(W_i l_{i-1} + b_i), \quad i = 2, \dots, N-1$$
$$y = f(W_N l_{N-1} + b_N)$$

La función de activación en capas ocultas y salida es la **tangente hiperbólica**:

$$f(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$$

La relevancia semántica entre consulta $Q$ y documento $D$ se mide por **similaridad coseno** de sus vectores concepto $y_Q$, $y_D$:

$$R(Q, D) = \text{cosine}(y_Q, y_D) = \frac{y_Q^T y_D}{\lVert y_Q \rVert \lVert y_D \rVert}$$

Dada la consulta, los documentos se ordenan por su puntaje de relevancia semántica.

### 3.2 Word hashing

El tamaño del vector de términos equivale al tamaño del vocabulario de indexación, que en búsqueda web real es del orden de millones. Usar eso como capa de entrada haría inmanejable el entrenamiento. El **word hashing** resuelve esto:

- Dada una palabra (p. ej. *good*), se agregan marcas de inicio y fin: `#good#`.
- Se descompone en n-gramas de letras (trigramas de letras): `#go`, `goo`, `ood`, `od#`.
- La palabra se representa como un vector de n-gramas de letras.

Esto es, en la práctica, la primera capa de la DNN ($W_1$): una **transformación lineal fija (no adaptativa, no se entrena)** que proyecta el vector de términos al espacio de n-gramas de letras. La Tabla 1 cuantifica el beneficio:

| Vocabulario | Letter-Bigram (tokens / colisiones) | Letter-Trigram (tokens / colisiones) |
|---|---|---|
| 40k palabras | 1107 / 18 | 10306 / 2 |
| 500k palabras | 1607 / 1192 | 30621 / 22 |

Para el vocabulario de 500k, los trigramas de letras producen un vector de 30 621 dimensiones: una reducción de 16× con una tasa de colisión despreciable de 0,0044 % (22/500 000). El número de palabras inglesas es ilimitado, pero el número de n-gramas de letras es finito y limitado.

Ventajas adicionales del word hashing:
- **Robustez a out-of-vocabulary (OOV):** una palabra no vista en entrenamiento es problemática con representaciones por palabra, pero no con n-gramas de letras (los n-gramas seguramente sí fueron vistos). El único riesgo es la colisión menor cuantificada arriba.
- **Variaciones morfológicas** de una misma palabra quedan mapeadas a puntos cercanos en el espacio de n-gramas de letras.

### 3.3 Aprendizaje del DSSM (entrenamiento discriminativo)

Inspirados en el entrenamiento discriminativo de speech/language processing, los autores aprenden los parámetros $\{W_i, b_i\}$ maximizando la verosimilitud condicional de los documentos clicados.

Primero se transforma el puntaje de relevancia en una **probabilidad posterior** vía softmax:

$$P(D \mid Q) = \frac{\exp(\gamma R(Q, D))}{\sum_{D' \in \mathbf{D}} \exp(\gamma R(Q, D'))}$$

donde $\gamma$ es un factor de suavizado (smoothing) del softmax, fijado empíricamente en un conjunto held-out. $\mathbf{D}$ es el conjunto de documentos candidatos. Idealmente $\mathbf{D}$ debería contener todos los documentos posibles; en la práctica, para cada par (consulta, documento clicado) $(Q, D^+)$ se aproxima $\mathbf{D}$ incluyendo $D^+$ y **cuatro documentos no clicados elegidos al azar** $\{D_j^-; j=1,\dots,4\}$. Los autores notan que en su estudio piloto no observaron diferencia significativa entre distintas estrategias de muestreo de negativos.

El entrenamiento minimiza la función de pérdida (verosimilitud negativa):

$$L(\Lambda) = -\log \prod_{(Q, D^+)} P(D^+ \mid Q)$$

donde $\Lambda = \{W_i, b_i\}$. Como $L(\Lambda)$ es diferenciable, el modelo se entrena con descenso de gradiente. El apéndice deriva el gradiente completo (regla de actualización $\Lambda_t = \Lambda_{t-1} - \epsilon_t \frac{\partial L(\Lambda)}{\partial \Lambda}$, retropropagación con productos de Hadamard); notablemente, $W_1$ (word hashing) es fijo y no requiere entrenamiento.

### 3.4 Detalles de implementación

- Arquitectura: **3 capas ocultas**. La primera es la capa de word hashing (~30k nodos). Las dos siguientes tienen **300 nodos** cada una; la capa de salida tiene **128 nodos** (dimensión del espacio semántico común).
- Inicialización de pesos: distribución uniforme en $[-\sqrt{6/(fanin+fanout)}, \sqrt{6/(fanin+fanout)}]$ (inicialización tipo Glorot).
- No observaron mejora con pre-entrenamiento capa a capa.
- Optimización: **SGD por mini-batches** de 1024 muestras; converge en ~20 épocas.

## 4. Experimentos

### Datos y metodología

- **Conjunto de evaluación:** 16 510 consultas en inglés muestreadas de un año de logs de un motor comercial; ~15 documentos (URLs) por consulta. Cada par consulta-título tiene una etiqueta de relevancia humana en escala 0-4.
- **Entrenamiento:** ~100 millones de pares consulta-título extraídos de URLs populares con clics ricos. Solo se usa el campo *título* del documento para ranking. El objetivo de investigación es aprender de URLs populares (con clics) y aplicar a URLs de cola/nuevas (sin clics).
- **Métrica:** NDCG (Normalized Discounted Cumulative Gain) en truncamientos 1, 3 y 10. Validación cruzada 2-fold. Test de significancia: paired t-test, p < 0,05.

### Resultados (Tabla 2)

| # | Modelo | NDCG@1 | NDCG@3 | NDCG@10 |
|---|---|---|---|---|
| 1 | TF-IDF | 0.319 | 0.382 | 0.462 |
| 2 | BM25 | 0.308 | 0.373 | 0.455 |
| 3 | WTM (word translation) | 0.332 | 0.400 | 0.478 |
| 4 | LSA | 0.298 | 0.372 | 0.455 |
| 5 | PLSA | 0.295 | 0.371 | 0.456 |
| 6 | DAE (deep autoencoder / SH) | 0.310 | 0.377 | 0.459 |
| 7 | BLTM-PR | 0.337 | 0.403 | 0.480 |
| 8 | DPM | 0.329 | 0.401 | 0.479 |
| 9 | DNN (sin word hashing) | 0.342 | 0.410 | 0.486 |
| 10 | L-WH linear | 0.357 | 0.422 | 0.495 |
| 11 | L-WH non-linear | 0.357 | 0.421 | 0.494 |
| **12** | **L-WH DNN (mejor modelo)** | **0.362** | **0.425** | **0.498** |

Hallazgos principales:
- El mejor DSSM (L-WH DNN, fila 12) supera a todos los competidores por un margen estadísticamente significativo. Respecto al mejor baseline previo, la mejora es de **2,5-4,3 % en NDCG@1**.
- **La supervisión sobre clickthrough es esencial:** DNN (fila 9) y DAE (fila 6) usan el mismo vocabulario de 40k y la misma arquitectura profunda, pero DNN es supervisado y DAE no; DNN supera a DAE por 3,2 puntos en NDCG@1.
- **El word hashing habilita vocabularios grandes:** la fila 12 (500k palabras con word hashing) supera a la fila 9 (40k palabras), aun teniendo *menos* parámetros libres (la capa de word hashing es de solo ~30k nodos).
- **La profundidad ayuda:** pasar de 1 a 3 capas no lineales sube NDCG en 0,4-0,5 puntos (significativo). Entre modelos shallow lineal vs no lineal de una capa (filas 10 vs 11) no hay diferencia significativa.

### Análisis de errores y visualización (Apéndice II)

Sobre 16 412 consultas únicas: L-WH DNN gana a TF-IDF en 1985 consultas (suma de diferencias NDCG@1 = 1332,3) y pierde en 1077 (suma = 630,61). Las victorias del DSSM provienen de coincidencias a nivel semántico más que léxico (ej.: "ccra" → "canada revenue agency", "met art" → "metropolitan museum of art"). Las derrotas tienden a casos donde la coincidencia léxica literal sí era la correcta (ej.: "hey arnold" → "hey arnold the movie"). La visualización de activaciones de nodos de salida muestra clusters semánticamente coherentes (automotive/wheels/cars/auto; chevrolet/toyota/chevy; etc.).

## 5. Limitaciones

- **Modelo de bolsa de palabras (BoW):** la entrada es un vector de recuentos de términos sin orden. No hay modelado de secuencia ni de posición; se pierde sintaxis y contexto local (eso llegará después con CLSM/CNN-DSSM y luego con Transformers).
- **Solo título del documento:** por el diseño experimental (URLs de cola sin clics), solo se usa el campo título, no el cuerpo completo del documento.
- **Negativos aleatorios:** se aproxima el softmax con 4 negativos muestreados al azar. No hay hard-negative mining (que sería central en trabajos posteriores).
- **Colisiones de word hashing:** aunque mínimas, existen; dos palabras distintas pueden compartir representación de n-gramas.
- **Datos y código propietarios:** el dataset proviene de logs de un motor comercial no público; no hay arxiv ni reproducción abierta directa.
- **Evaluación monolingüe (inglés)** y sobre un solo motor comercial.

## 6. Impacto

DSSM es ampliamente reconocido como la **raíz del paradigma two-tower / dual-encoder** en retrieval y recomendación neuronal. Su patrón —dos codificadores que proyectan a un espacio común y comparan por similaridad coseno, entrenados discriminativamente con softmax sobre positivos y negativos— se convirtió en el esqueleto de:

- **Variantes inmediatas en MSR:** CLSM/CDSSM (convolucional, agregando ventanas de contexto), y posteriormente versiones con LSTM.
- **Sentence/text embeddings densos** y el retrieval denso moderno (dense passage retrieval).
- **Recomendadores two-tower a escala industrial:** el modelo de muestreo de YouTube (Yi et al., 2019) y las arquitecturas de candidate generation de Google/YouTube heredan directamente esta estructura.
- **Contrastive learning multimodal:** la idea de dos torres a espacio común con InfoNCE/softmax sobre negativos es el corazón de CLIP y de los recomendadores multimodales actuales.

El softmax con negativos de DSSM es, conceptualmente, un precursor de la pérdida contrastiva / sampled softmax que domina el entrenamiento de embeddings hoy. La técnica de word hashing por n-gramas de letras anticipa además las representaciones subword (BPE, WordPiece, fastText) por su robustez a OOV.

## 7. Conexión con la Clase 25

La Clase 25 (sistemas de recomendación multimodales) usa una arquitectura de **dos torres** que proyectan ítems y usuarios/consultas a un **espacio común**, comparándolos por **distancia/similaridad**. DSSM es el antecesor directo de ese case study:

- **Dos torres → espacio común:** en DSSM, la torre de la consulta y la torre del documento son DNNs que mapean a un mismo espacio de 128 dimensiones. En la Clase 25, las torres mapean usuario/contexto e ítem (con features multimodales) al mismo espacio.
- **Similaridad como relevancia:** DSSM define $R(Q,D)=\text{cosine}(y_Q,y_D)$; la Clase 25 usa exactamente este principio de scoring por producto interno/coseno en el espacio latente, lo que permite retrieval eficiente (ANN) a escala.
- **Entrenamiento con softmax sobre negativos:** el $P(D|Q)$ con un positivo y negativos muestreados es la semilla de la pérdida contrastiva / sampled softmax que usan los recomendadores two-tower modernos.
- **Señal de supervisión implícita:** clickthrough en DSSM ≈ interacciones (clics, compras, vistas) en recomendación. Es feedback implícito barato y masivo.

En síntesis: la Clase 25 toma este esqueleto de 2013, reemplaza el bag-of-words por encoders multimodales (texto, imagen, etc.) y escala con muestreo de negativos y búsqueda aproximada de vecinos, pero la columna vertebral conceptual —dos torres, un espacio, una similaridad— es la de DSSM.

## Referencias clave citadas en el paper

- Deerwester et al. (1990) — Indexing by latent semantic analysis (LSA).
- Hofmann (1999) — Probabilistic latent semantic indexing (PLSA).
- Blei, Ng, Jordan (2003) — Latent Dirichlet Allocation (LDA).
- Gao, He, Nie (2010) — Clickthrough-based translation models for web search.
- Gao, Toutanova, Yih (2011) — Clickthrough-based latent semantic models (BLTM, DPM).
- Salakhutdinov, Hinton (2007) — Semantic hashing (deep autoencoder).
- Burges et al. (2005) — Learning to rank using gradient descent.
- Järvelin, Kekäläinen (2000) — IR evaluation methods (NDCG).
- Yih, Toutanova, Platt, Meek (2011) — Learning discriminative projections (S2Net).
