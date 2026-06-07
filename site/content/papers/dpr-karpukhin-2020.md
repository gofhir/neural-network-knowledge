---
title: "DPR (Dense Passage Retrieval for Open-Domain QA)"
weight: 118
math: true
---

{{< paper-card
    title="Dense Passage Retrieval for Open-Domain Question Answering"
    authors="Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih"
    year="2020"
    venue="EMNLP 2020 (arXiv 2004.04906)"
    pdf="/papers/dpr-karpukhin-2020.pdf"
    arxiv="2004.04906" >}}
DPR demuestra que el retrieval para open-domain QA no necesita seguir atado a BM25. Un **bi-encoder** de dos torres BERT que mapea preguntas y passages a vectores densos, entrenado para **maximizar el producto interno** entre cada pregunta y su passage relevante, supera a BM25 por 9-19 puntos absolutos en top-20 retrieval accuracy. La receta no es arquitectonica sino de entrenamiento: **in-batch negatives** (reusar los passages del batch como negativos gratis) mas **un hard negative de BM25** por pregunta. Con apenas 1.000 ejemplos ya supera a BM25, y la mayor precision de retrieval se propaga a mejor exact match end-to-end (41.5% vs. 33.3% de ORQA en Natural Questions). El indice se construye offline con **FAISS** sobre 21 millones de passages de Wikipedia.
{{< /paper-card >}}

---

## El problema

El **open-domain question answering** responde preguntas factoides (*"Who first voiced Meg on Family Guy?"*) usando una coleccion grande de documentos, sin que se entregue de antemano el pasaje donde esta la respuesta. Desde DrQA (Chen et al. 2017) el patron dominante es un pipeline de dos etapas:

1. **Retriever** $R: (q, C) \to C_F$ — toma la pregunta $q$ y un corpus $C$ de millones de passages, y devuelve un subconjunto pequeno $C_F$ con $|C_F| = k \ll |C|$.
2. **Reader** — un modelo de comprension lectora que examina esos $k$ passages e identifica el span de respuesta.

Formalmente, los documentos se dividen en passages de longitud fija $C = \{p_1, \dots, p_M\}$ y la tarea extractiva es encontrar el span $w_s^{(i)}, \dots, w_e^{(i)}$ que responde la pregunta. Con corpus de decenas de millones de items, el retriever eficiente es indispensable: el reader nunca podria leer todo.

Durante anos ese retriever fue **TF-IDF o BM25** (Robertson y Zaragoza, 2009): empareja keywords con un indice invertido y se puede ver como una representacion de pregunta y contexto en vectores **sparse de alta dimension** con pesos. Su limitacion es estructural: el matching es **lexico**, sobre tokens. No captura sinonimos ni parafrasis — el problema de *vocabulary mismatch*.

El ejemplo canonico del paper: la pregunta *"Who is the bad guy in lord of the rings?"* se responde desde *"Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy."* Un sistema basado en terminos tiene dificultad para recuperar ese contexto, porque "bad guy" y "villain" no comparten tokens. Un retriever denso, en cambio, mapea ambos a vectores cercanos y recupera el contexto correcto.

Las representaciones densas tienen otra ventaja conceptual: son **aprendibles**. Se puede ajustar la funcion de embedding a la tarea, algo imposible con BM25 (un esquema fijo de pesos). Y con indices apropiados, la busqueda densa se hace eficiente via **maximum inner product search (MIPS)**.

El obstaculo historico era una creencia: que aprender una buena representacion densa requiere *muchos* pares etiquetados pregunta–contexto. Por eso los metodos densos nunca habian superado a BM25 en open-domain QA antes de **ORQA** (Lee et al. 2019), que introdujo el costoso *inverse cloze task* (ICT) como pre-entrenamiento adicional. La pregunta central de DPR: *¿podemos entrenar un mejor modelo de embedding denso usando solo pares de preguntas y passages, sin pre-entrenamiento adicional?*

---

## Idea central — reemplazar BM25 por un retriever denso

DPR responde que si. La solucion, tras una serie de ablations cuidadosos, es "sorprendentemente simple": un **bi-encoder (dual-encoder)** basado en BERT que mapea preguntas y passages a vectores densos, optimizado para **maximizar el producto interno** entre la pregunta y su passage relevante, comparando *todos* los pares pregunta–passage dentro de un batch.

El resultado es contundente. DPR supera a BM25 por amplio margen (9-19 puntos absolutos en top-20 retrieval accuracy a lo largo de varios datasets). En Natural Questions:

- **Top-5 accuracy**: 65.2% (DPR) vs. 42.9% (BM25).
- **End-to-end QA exact match**: 41.5% (DPR) vs. 33.3% (ORQA).

Y lo logra con relativamente pocos ejemplos: **DPR entrenado con solo 1.000 ejemplos ya supera a BM25** en NQ, desmintiendo la creencia de que se necesitan grandes cantidades de pares etiquetados.

Las dos contribuciones declaradas:

1. Con el setup de entrenamiento adecuado, **afinar los encoders de pregunta y passage sobre pares existentes basta para superar ampliamente a BM25**; el pre-entrenamiento adicional (ICT de ORQA) puede no ser necesario.
2. En open-domain QA, **mayor precision de retrieval se traduce en mayor accuracy end-to-end** — la cadena retriever→reader propaga las mejoras.

---

## Arquitectura del bi-encoder

DPR usa dos encoders separados:

- $E_P(\cdot)$ — encoder de passage, mapea cualquier texto a un vector real de $d$ dimensiones. Se aplica a los $M$ passages para construir el indice.
- $E_Q(\cdot)$ — encoder de pregunta, mapea la pregunta de entrada a un vector de $d$ dimensiones.

La **similitud** entre pregunta y passage es el producto punto de sus vectores:

$$\mathrm{sim}(q, p) = E_Q(q)^\top E_P(p)$$

Ambos encoders son **dos redes BERT independientes** (base, uncased), tomando la representacion del token `[CLS]` como salida, por lo que $d = 768$.

### Por que dos torres y no un cross-encoder

Esta es la decision arquitectonica clave. Existen formas mas expresivas de medir similitud — por ejemplo, un **cross-encoder** donde pregunta y passage se concatenan y pasan juntos por capas de cross-attention. Pero el paper es explicito:

> la funcion de similitud necesita ser **descomponible** para que las representaciones de la coleccion de passages puedan precomputarse.

Un cross-encoder calcula una funcion conjunta no factorizable $f(q, p)$: para puntuar un par hay que pasar pregunta y passage juntos por la red. Eso impide indexar offline — habria que ejecutar el modelo $M$ veces (una por passage) en cada query, lo que con 21 millones de passages es inviable en tiempo real.

El bi-encoder, en cambio, **factoriza** el computo: $E_P(p)$ no depende de $q$, asi que todos los vectores de passage se precomputan una vez y se indexan. En tiempo de query solo se computa $E_Q(q)$ y se busca el vecino mas cercano por producto interno. El passage encoder corre **offline**; el question encoder corre **online**. Esa asimetria temporal es la razon de ser de las dos torres.

El ablation encuentra que otras funciones de similitud (L2, coseno) rinden de forma comparable, asi que eligen la mas simple — el **producto interno** — y concentran el esfuerzo en aprender mejores encoders. La filosofia del paper: simplicidad arquitectonica, sofisticacion en el entrenamiento.

---

## Entrenamiento

Entrenar los encoders para que el producto punto sea una buena funcion de ranking es esencialmente un problema de **metric learning**: construir un espacio vectorial donde los pares relevantes pregunta–passage tengan mayor similitud que los irrelevantes.

### La loss: negative log-likelihood del positivo

Los datos de entrenamiento son $m$ instancias, cada una con una pregunta $q_i$, un passage relevante (positivo) $p_i^+$, y $n$ passages irrelevantes (negativos) $p_{i,j}^-$. Se optimiza la **negative log-likelihood del passage positivo**:

$$L(q_i, p_i^+, p_{i,1}^-, \cdots, p_{i,n}^-) = -\log \frac{e^{\mathrm{sim}(q_i, p_i^+)}}{e^{\mathrm{sim}(q_i, p_i^+)} + \sum_{j=1}^{n} e^{\mathrm{sim}(q_i, p_{i,j}^-)}}$$

Es exactamente un softmax sobre los scores de similitud con la cross-entropy concentrada en el positivo — un objetivo **contrastivo**. Maximizar el numerador empuja el vector de la pregunta hacia su passage positivo; el denominador lo aleja de los negativos.

### El problema de los negativos

En retrieval los positivos suelen estar disponibles, pero los negativos hay que seleccionarlos de un pool gigantesco — una eleccion "a menudo pasada por alto pero que puede ser decisiva". Se consideran tres tipos:

1. **Random** — cualquier passage aleatorio del corpus.
2. **BM25** — top passages devueltos por BM25 que *no* contienen la respuesta pero coinciden con la mayoria de los tokens de la pregunta. Son **hard negatives**: lexicamente parecidos pero incorrectos.
3. **Gold** — passages positivos *de otras preguntas* del set de entrenamiento.

El mejor modelo usa **gold passages del mismo mini-batch + un passage negativo de BM25**.

### In-batch negatives: el truco de eficiencia

Es la pieza central. Dado un mini-batch de $B$ preguntas, cada una con su passage relevante, sean $Q$ y $P$ las matrices $(B \times d)$ de embeddings de preguntas y passages del batch. Entonces

$$S = Q P^\top$$

es una matriz $(B \times B)$ de scores, donde la fila $i$ es la pregunta $i$ emparejada con los $B$ passages del batch. Cualquier par $(q_i, p_j)$ es **positivo cuando $i = j$ y negativo en caso contrario**. Esto genera $B$ instancias de entrenamiento por batch, cada una con $B-1$ passages negativos.

La elegancia: los embeddings de passage ya se computaron para servir de positivos de *sus* preguntas; reusarlos como negativos de las *demas* preguntas no cuesta nada extra. Un solo producto matricial $QP^\top$ produce todos los scores. Por eso **la accuracy mejora consistentemente al crecer el batch size**.

El modelo principal usa **batch size 128 + un negativo BM25 adicional por pregunta**, entrenado hasta 40 epochs (datasets grandes) o 100 epochs (pequenos), con learning rate $10^{-5}$ usando Adam, scheduling lineal con warm-up y dropout 0.1.

La combinacion final lo dice todo: **gold in-batch negatives (gratis, abundantes, mayormente "faciles") + 1 hard negative de BM25 (caro pero informativo)**. Los in-batch dan volumen; el hard negative de BM25 fuerza al modelo a discriminar contra distractores lexicamente plausibles, que son justo donde un retriever ingenuo falla.

---

## Busqueda eficiente — FAISS para MIPS

En inferencia, $E_P$ se aplica a todos los passages y se indexan con **FAISS** (Johnson et al. 2017) offline — una libreria de busqueda de similitud que escala a miles de millones de vectores. Dada una pregunta $q$ en tiempo de ejecucion, se deriva su embedding $v_q = E_Q(q)$ y se recuperan los top $k$ passages cuyos embeddings estan mas cerca de $v_q$.

El corpus es el dump de Wikipedia en ingles del 20 de diciembre de 2018, dividido en **bloques disjuntos de 100 palabras** como unidades de retrieval. Esto da **21.015.324 passages** (≈21M). Cada passage se prepende con el titulo del articulo mas un token `[SEP]`. Los passages de longitud fija rinden mejor que parrafos naturales, y el solapamiento entre passages no aporta ventaja.

El perfilado de eficiencia (servidor con indice FAISS HNSW en CPU) muestra el trade-off:

| Metrica | DPR (FAISS) | BM25 (Lucene) |
|---|---|---|
| Throughput de query | **995.0 preguntas/s** (top-100) | 23.7 preguntas/s por thread |
| Construccion del indice | 8.5 h FAISS + 8.8 h embeddings (8 GPUs) | ~30 min (indice invertido) |

**DPR es ~42× mas rapido en query, pero la construccion del indice es mucho mas cara** (computar 21M embeddings en GPUs y construir el HNSW). El costo de indexacion es unico y amortizable, pero re-indexar tras re-entrenar el passage encoder es oneroso — razon por la cual el joint-training del paper *congela* el passage encoder.

---

## Resultados

### Retrieval (top-k accuracy)

Porcentaje de las top 20/100 passages recuperadas que contienen la respuesta. "Single" = DPR por dataset; "Multi" = entrenado combinando todos excepto SQuAD.

| Train | Retriever | NQ (20) | TriviaQA (20) | WQ (20) | TREC (20) | SQuAD (20) |
|---|---|---|---|---|---|---|
| None | BM25 | 59.1 | 66.9 | 55.0 | 70.9 | 68.8 |
| Single | DPR | **78.4** | **79.4** | **73.2** | 79.8 | 63.2 |
| Single | BM25+DPR | 76.6 | 79.8 | 71.0 | **85.2** | 71.5 |
| Multi | DPR | 79.4 | 78.8 | 75.0 | **89.1** | 51.6 |

Lecturas clave:

- DPR supera a BM25 en **todos los datasets excepto SQuAD**. La brecha es mayor cuando $k$ es pequeno (78.4% vs. 59.1% en top-20 NQ): DPR ordena mejor los primeros resultados.
- **TREC** (el mas pequeno) se beneficia mucho del entrenamiento multi-dataset (79.8 → 89.1 en top-20).
- **SQuAD es la excepcion.** DPR rinde peor que BM25. Dos razones conjeturadas: las preguntas se escribieron *despues* de ver el passage (alto solapamiento lexico que favorece a BM25), y los datos provienen de solo 500+ articulos, sesgando la distribucion.
- El **hibrido BM25+DPR** (rerank de la union de top-2000 de cada uno con $\mathrm{BM25}(q,p) + \lambda \cdot \mathrm{sim}(q,p)$, $\lambda = 1.1$) ayuda donde BM25 es competitivo (TREC, SQuAD).

### End-to-end QA (Exact Match)

El reader procesa hasta los top-100 passages recuperados.

| Train | Modelo | NQ | TriviaQA | WQ | TREC | SQuAD |
|---|---|---|---|---|---|---|
| Single | ORQA (Lee 2019) | 33.3 | 45.0 | 36.4 | 30.1 | 20.2 |
| Single | BM25 | 32.6 | 52.4 | 29.9 | 24.9 | **38.1** |
| Single | DPR | **41.5** | 56.8 | 34.6 | 25.9 | 29.8 |
| Multi | DPR | 41.5 | 56.8 | 42.4 | 49.4 | 24.1 |
| Multi | BM25+DPR | 38.8 | **57.9** | 41.1 | **50.6** | 35.8 |

- **Mayor accuracy de retrieval ⇒ mejor QA final**, en todos los casos excepto SQuAD — esto valida la segunda contribucion.
- DPR establece **nuevo estado del arte en 4 de los 5 datasets** y **supera a ORQA y REALM** en NQ y TriviaQA, pese a que ambos usan pre-entrenamiento adicional costoso.
- El **pipeline supera al joint training**: una ablation con retriever y reader entrenados conjuntamente da 39.8 EM en NQ, peor que entrenarlos por separado.

---

## Ablations

Esquemas de entrenamiento por top-k retrieval accuracy en el dev set de NQ. `IB` = in-batch.

| Tipo | #N | IB | Top-5 | Top-20 | Top-100 |
|---|---|---|---|---|---|
| Random | 7 | no | 47.0 | 64.3 | 77.8 |
| BM25 | 7 | no | 50.0 | 63.3 | 74.8 |
| Gold | 7 | no | 42.6 | 63.1 | 78.3 |
| Gold | 7 | si | 51.1 | 69.1 | 80.8 |
| Gold | 127 | si | 55.8 | 73.0 | 83.1 |
| G.+BM25(1) | 127+128 | si | **65.8** | **78.0** | **84.9** |

Hallazgos:

- **Tipo de negativos sin in-batch**: random vs. BM25 vs. gold *no* importa mucho cuando $k \geq 20$ en el setting 1-of-N.
- **In-batch negatives**: con la misma configuracion (7 gold), in-batch mejora sustancialmente (Top-5: 42.6 → 51.1).
- **Batch size**: la accuracy crece consistentemente al aumentar el batch (Gold 7 → 127 in-batch: Top-5 51.1 → 55.8).
- **Hard negatives de BM25**: agregar **un solo negativo BM25** dispara el Top-5 a 65.8. **Agregar dos no ayuda mas.**

Otros: con **1.000 ejemplos** DPR ya supera a BM25 en NQ; la **supervision distante** (top passage de BM25 que contiene la respuesta) degrada solo ~1 punto frente al gold; y DPR entrenado solo en NQ **transfiere** a WQ/TREC perdiendo 3-5 puntos pero superando claramente a BM25.

---

## Por que importa hoy

DPR se volvio un componente fundacional del **retrieval neuronal** moderno:

- **RAG** (Lewis et al. 2020 — coautores compartidos con DPR): combina DPR con generadores como BART y T5. DPR es el *retriever* sobre el que se construye toda la familia retrieval-augmented. El paper ya lo anticipa.
- **FiD / Fusion-in-Decoder** (Izacard y Grave 2020): aprovecha el retrieval de passages con modelos generativos.
- **ANCE** (Xiong et al. 2020): extiende los hard negatives usando el modelo de la iteracion previa para descubrir nuevos negativos, partiendo del modelo DPR.
- **ColBERT** (Khattab y Zaharia 2020): introduce *late interaction*, un punto medio entre bi-encoder y cross-encoder.

DPR es la prueba de concepto definitiva de que un **bi-encoder entrenado con in-batch negatives + hard negatives** es un retriever de primera categoria a escala de decenas de millones de items. Ese patron trasciende QA: es el mismo que hoy usan la busqueda semantica, la recomendacion, la deduplicacion y el entity matching. Tambien validó empiricamente que **mayor recall de retrieval propaga a mayor accuracy end-to-end** — la justificacion conceptual de toda la cadena retrieval-augmented que domina los sistemas knowledge-intensive actuales.

Las **limitaciones** marcan donde el sparse sigue vivo: para tokens raros y salientes (nombres propios poco frecuentes) BM25 gana, mientras DPR brilla en variacion lexica/semantica. Por eso el hibrido densa+sparse es complementario, no redundante.

---

## Conexion con la clase 24

La clase 24 cubre **IR-based Factoid QA**: el pipeline clasico de *question processing → passage retrieval → answer processing*. DPR es la **version moderna y neuronal de la etapa de passage retrieval** de ese pipeline.

- **Question processing** clasico (extraer keywords, tipo de respuesta, query expansion) se reemplaza por el question encoder $E_Q(q)$ — una sola pasada de BERT que produce el vector de query. La "comprension" de la pregunta queda latente en el embedding, sin reglas ni expansion manual.
- **Passage retrieval** clasico (TF-IDF/BM25 sobre indice invertido) se reemplaza por MIPS sobre embeddings densos via FAISS. El cambio es de *matching lexico sparse* a *matching semantico denso*. La clase ensena BM25 como el de facto; DPR muestra como el retrieval neuronal lo supera resolviendo el vocabulary mismatch que la clase identifica como su limitacion.
- **Answer processing** clasico (extraccion y ranking de spans) es el reader neuronal con cross-attention que asigna span scores y passage selection scores.

En sintesis: la clase ensena el esqueleto conceptual (retriever + reader, las tres etapas); DPR moderniza la etapa de retrieval reemplazando el componente sparse por uno denso aprendido, y demuestra que esa modernizacion mejora todo el pipeline.

---

## Notas y enlaces

- Paper original: Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020. arXiv:2004.04906.
- Codigo y modelos: [facebookresearch/DPR](https://github.com/facebookresearch/DPR).
- Datasets: Natural Questions, TriviaQA, WebQuestions, CuratedTREC, SQuAD v1.1.
- Antecedentes directos: dual-encoder / red siamesa (Bromley et al. 1994); DSSM (Huang et al. 2013); dense entity retrieval (Gillick et al. 2019); ORQA con ICT (Lee et al. 2019).
- Cifras emblematicas: Top-5 NQ 65.2% vs. 42.9% BM25; EM end-to-end NQ 41.5% vs. 33.3% ORQA; 995 q/s vs. 23.7 q/s/thread; 21.015.324 passages; mejora absoluta de retrieval 9-19% en top-20.

Ver fundamentos: [Dense Retrieval](/fundamentos/dense-retrieval) - [Question Answering](/fundamentos/question-answering) - [Aprendizaje Contrastivo](/fundamentos/aprendizaje-contrastivo) - [BERT](/fundamentos/bert).

Ver papers: [MS MARCO (Nguyen 2016)](/papers/ms-marco-nguyen-2016) - [BERT (Devlin 2018)](/papers/bert-devlin-2018) - [Stanford Attentive Reader (Chen 2016)](/papers/stanford-attentive-reader-chen-2016).

Ver clase: [Clase 24 — Question Answering](/clases/clase-24).
