---
title: "Sistemas de Recomendación"
weight: 140
math: true
---

Un **sistema de recomendación** (recommender system) es un modelo cuyo objetivo es, dado un universo enorme de ítems —productos, videos, canciones, publicaciones, prendas de ropa— y un usuario, **predecir qué ítems le resultarán relevantes** y ordenarlos para mostrárselos. Es uno de los problemas de machine learning con mayor impacto económico directo: la mayor parte del consumo en Netflix, YouTube, Amazon, Spotify o Pinterest está mediado por recomendaciones. A diferencia de la búsqueda, donde el usuario expresa una intención explícita mediante una consulta, aquí el sistema debe **inferir la intención** a partir del comportamiento pasado y de las características del usuario y de los ítems. El reto fundamental es que la información disponible es **escasa y sesgada**: un usuario interactúa solo con una fracción minúscula del catálogo, y lo que no observamos no significa necesariamente desinterés. Este fundamento recorre la familia de técnicas que aborda este problema, desde la factorización de matrices clásica hasta los recomendadores neuronales multimodales del [Case Study de la Clase 25](/clases/clase-25).

---

## 1. La pregunta que responde: la matriz usuario-ítem y la sparsity

El punto de partida formal es una **matriz de interacciones** $R \in \mathbb{R}^{m \times n}$, con $m$ usuarios y $n$ ítems. La entrada $r_{ui}$ codifica la interacción del usuario $u$ con el ítem $i$: una calificación de 1 a 5 estrellas, un clic, una compra, los segundos vistos, o simplemente un 1 si hubo interacción. El problema de recomendación se reduce a **completar las entradas faltantes** de esta matriz y luego rankear los ítems no observados por su valor predicho.

La dificultad central es la **sparsity** (dispersión extrema). En el dataset Netflix, ~480.000 usuarios y ~17.770 películas generan una matriz con más de 8.500 millones de celdas, de las cuales solo ~100 millones están observadas: **más del 98,8% está vacío**. En catálogos industriales con millones de ítems, la densidad cae por debajo del 0,01%. No podemos tratar esto como una regresión ordinaria sobre features, porque la mayoría de los pares usuario-ítem nunca se observan, y los que se observan **no son una muestra aleatoria**: están sesgados por lo que el sistema mostró antes y por la popularidad.

{{< concept-alert type="clave" >}}
Recomendar **no es predecir un rating con bajo error**: es **ordenar correctamente** un catálogo gigante para cada usuario, usando observaciones extremadamente escasas y sesgadas. Toda la disciplina gira en torno a generalizar desde poquísimas señales y a priorizar el **top-K** —los pocos ítems que efectivamente se muestran— por sobre la calidad de la predicción en el resto de la matriz.
{{< /concept-alert >}}

---

## 2. Feedback explícito vs. implícito

Las señales que alimentan al sistema se dividen en dos grandes tipos, con propiedades estadísticas muy distintas.

El **feedback explícito** es una valoración que el usuario emite de forma deliberada: estrellas, pulgar arriba/abajo, una nota numérica. Es de alta calidad y bidireccional (informa de gusto y disgusto), pero **raro y sesgado**: la gente califica poco, y tiende a hacerlo cuando ama u odia algo, no en los casos intermedios.

El **feedback implícito** es la huella del comportamiento: clics, vistas, compras, tiempo de permanencia, scroll, agregar al carrito. Es **abundante y barato**, y constituye hoy la fuente dominante en producción. Pero plantea un problema profundo: solo observamos **señales positivas**. Si un usuario no vio un video, ¿es porque no le interesa, o porque nunca se lo mostramos? El feedback implícito **no tiene negativos verdaderos**, solo positivos y ausencias ambiguas.

| Aspecto | Feedback explícito | Feedback implícito |
|---|---|---|
| Ejemplo | Rating 1-5 estrellas | Clic, vista, compra |
| Volumen | Escaso | Masivo |
| Señal negativa | Disponible (rating bajo) | Ausente (solo positivos) |
| Sesgo | Selección (califica quien quiere) | Exposición (popularidad, lo mostrado) |
| Tarea natural | Regresión / predecir rating | Ranking / clasificación 1-clase |

Este giro —de "predecir cuánto gustará" a "ordenar bajo positivos implícitos"— motiva el aprendizaje de ranking (sección 6) y obliga a tratar las ausencias como **negativos débiles muestreados**, no como ceros reales.

---

## 3. Content-based vs. Collaborative Filtering vs. Híbrido

Hay dos filosofías para predecir relevancia, más su combinación.

El **filtrado basado en contenido** (content-based) recomienda ítems **similares a los que el usuario ya consumió**, usando atributos de los ítems: género, texto, etiquetas, features visuales. Construye un perfil del usuario en el espacio de atributos y busca ítems cercanos. Ventaja: funciona para ítems nuevos (basta con sus atributos) y no necesita otros usuarios. Desventaja: queda **encerrado en la burbuja** del usuario (over-specialization), no descubre intereses laterales, y depende de tener buenos atributos.

El **filtrado colaborativo** (collaborative filtering, CF) ignora los atributos y se apoya solo en el **patrón colectivo de interacciones**: "usuarios que se parecieron a ti en el pasado consumieron esto". Es la técnica más potente cuando hay suficientes datos, porque captura señales latentes que ningún atributo describe explícitamente (por ejemplo, un "estilo" difícil de etiquetar). Su talón de Aquiles es el **cold start**: no sabe qué hacer con usuarios o ítems sin historial (sección 10).

Los sistemas **híbridos** combinan ambos: CF para el grueso de usuarios e ítems con historial, content-based para cubrir el cold start y aportar diversidad. La mayoría de los recomendadores modernos —incluido el de la Clase 25— son híbridos: usan colaborativo para los factores latentes y contenido (texto, imágenes) para enriquecer las representaciones.

---

## 4. Collaborative Filtering por vecindad (neighborhood methods)

La forma más antigua e intuitiva de CF es la basada en **vecindad**. No aprende parámetros: calcula similitudes directamente sobre la matriz.

En el enfoque **user-based**, para predecir $r_{ui}$ se buscan usuarios parecidos a $u$ que sí calificaron $i$, y se promedia su valoración ponderada por similitud. En el enfoque **item-based** —el que escaló Amazon— se buscan los ítems más parecidos a los que $u$ ya valoró. La predicción item-based típica es:

$$
\hat{r}_{ui} = \frac{\sum_{j \in N(i;u)} s_{ij}\, r_{uj}}{\sum_{j \in N(i;u)} |s_{ij}|},
$$

donde $N(i;u)$ son los ítems vecinos de $i$ que $u$ calificó y $s_{ij}$ es la similitud entre ítems. Las dos medidas clásicas de similitud son el **coseno** y la **correlación de Pearson** (coseno centrado por la media, que corrige el sesgo de usuarios "duros" o "blandos"):

$$
s^{\cos}_{ij} = \frac{\sum_u r_{ui} r_{uj}}{\sqrt{\sum_u r_{ui}^2}\sqrt{\sum_u r_{uj}^2}}, \qquad
s^{\text{Pearson}}_{ij} = \frac{\sum_u (r_{ui}-\bar r_i)(r_{uj}-\bar r_j)}{\sqrt{\sum_u (r_{ui}-\bar r_i)^2}\sqrt{\sum_u (r_{uj}-\bar r_j)^2}}.
$$

El enfoque item-based domina en producción porque las similitudes ítem-ítem son **más estables en el tiempo** que las usuario-usuario y se pueden precomputar. Su límite es que **no generaliza**: solo conecta ítems que comparten usuarios. Si dos películas excelentes nunca fueron vistas por la misma persona, su similitud es cero. Esto motiva el salto a factores latentes.

---

## 5. Matrix Factorization: factores latentes

La **factorización de matrices** (MF) es el avance que dominó la era post-Netflix Prize. La idea: aprender, para cada usuario y cada ítem, un **vector latente** de dimensión $k$ (típicamente 20-200), de modo que la afinidad se prediga por su producto punto.

$$
\hat{r}_{ui} = q_i^\top p_u, \qquad p_u, q_i \in \mathbb{R}^k.
$$

Cada dimensión latente captura un factor no observado —"cuán de ciencia ficción", "cuán orientado a niños"— que se infiere automáticamente de los datos, no se etiqueta a mano. A diferencia de la vecindad, MF **sí generaliza**: dos ítems que ningún usuario compartió pueden quedar cercanos en el espacio latente si sus patrones globales coinciden. En la práctica se añaden **sesgos** (biases) que absorben las tendencias marginales —algunos usuarios califican alto, algunas películas son universalmente populares—:

$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^\top p_u.
$$

Los parámetros se aprenden minimizando el error sobre las entradas **observadas**, con regularización $L_2$ para evitar sobreajuste en la sparsity:

$$
\min_{p,q,b} \sum_{(u,i)\in\mathcal{K}} \left( r_{ui} - \mu - b_u - b_i - q_i^\top p_u \right)^2 + \lambda\left(\lVert p_u\rVert^2 + \lVert q_i\rVert^2 + b_u^2 + b_i^2\right).
$$

Las dos rutas de optimización son **SGD** (descenso de gradiente estocástico, simple y rápido) y **ALS** (alternating least squares, que fija un lado y resuelve el otro en forma cerrada, paralelizable y preferido para feedback implícito). El artículo canónico de [Koren, Bell y Volinsky (2009)](/papers/matrix-factorization-koren-2009) sistematizó esta familia, incluyendo extensiones temporales y la fusión con vecindad. MF sigue siendo un **baseline durísimo de superar** y la base conceptual de casi todo lo que vino después: los "embeddings" de usuarios e ítems de los modelos neuronales son los descendientes directos de $p_u$ y $q_i$.

---

## 6. Aprender a rankear: BPR y feedback implícito

Cuando el feedback es implícito, minimizar el error cuadrático sobre ratings carece de sentido: no hay ratings, y tratar todas las ausencias como ceros sesga el modelo hacia la inactividad. La solución es cambiar la función objetivo de **predecir un valor** a **ordenar pares**.

**BPR** (Bayesian Personalized Ranking, [Rendle et al. 2009](/papers/bpr-rendle-2009)) formula el problema de manera *pairwise*: para cada usuario $u$, un ítem positivo $i$ (con el que interactuó) debe rankear **por encima** de un ítem $j$ no observado. El objetivo no compara contra un valor objetivo, sino que maximiza la probabilidad de que el orden $i \succ_u j$ sea correcto:

$$
\max \sum_{(u,i,j)} \ln \sigma\!\left(\hat{x}_{ui} - \hat{x}_{uj}\right) - \lambda \lVert \Theta \rVert^2,
$$

donde $\hat{x}_{ui}$ es la puntuación del modelo (por ejemplo, la de MF) y $\sigma$ es la sigmoide. El gradiente solo empuja a que el positivo supere al negativo muestreado, **sin forzar valores absolutos**. BPR es agnóstico al modelo de scoring subyacente y se convirtió en la pérdida estándar para recomendación con feedback implícito. Conceptualmente, es la versión recomendadora del aprendizaje contrastivo y del [triplet loss](/fundamentos/triplet-loss): "acerca el positivo, aleja el negativo". El muestreo de negativos —cómo elegir $j$— resulta crítico: muestrear por popularidad o usar *hard negatives* mejora notablemente la señal.

---

## 7. Deep recommender systems

A partir de ~2016, las redes neuronales reemplazaron el producto punto por interacciones aprendidas y absorbieron features ricas.

**Neural Collaborative Filtering** ([He et al. 2017](/papers/neural-collaborative-filtering-he-2017)) reemplaza el producto punto $q_i^\top p_u$ por un **MLP** que aprende una función de interacción arbitraria entre los embeddings de usuario e ítem, argumentando que el producto punto es una restricción lineal innecesaria. (Trabajo posterior matizó esto: un producto punto bien regularizado suele igualar o superar al MLP, una lección saludable sobre no descartar baselines simples).

**Wide & Deep** ([Cheng et al. 2016](/papers/wide-and-deep-cheng-2016)) combina dos vías: una **wide** (regresión lineal sobre cruces de features, que memoriza co-ocurrencias específicas) y una **deep** (embeddings + MLP, que generaliza a combinaciones no vistas). El sistema, desplegado en Google Play, captura simultáneamente memorización y generalización.

**DeepFM** ([Guo et al. 2017](/papers/deepfm-guo-2017)) refina la vía wide reemplazando los cruces manuales por una **máquina de factorización** (FM) que aprende automáticamente las interacciones de segundo orden entre features, compartiendo embeddings con la parte profunda. Elimina la ingeniería manual de cruces de Wide & Deep.

| Modelo | Núcleo | Aporte clave |
|---|---|---|
| MF | $q_i^\top p_u$ | factores latentes, baseline fuerte |
| NCF | MLP sobre embeddings | interacción no lineal aprendida |
| Wide & Deep | lineal + MLP | memorización + generalización |
| DeepFM | FM + MLP | cruces de 2.º orden automáticos |

---

## 8. Retrieval a gran escala: two-tower y candidate generation

Con catálogos de millones de ítems es **imposible puntuar cada par usuario-ítem** con un modelo pesado en tiempo de respuesta. La industria adopta una arquitectura en **dos etapas**: primero *candidate generation* (retrieval), que reduce millones a unos cientos de candidatos baratos; luego *ranking*, un modelo costoso que ordena finamente ese conjunto pequeño.

El retrieval se resuelve con la arquitectura **two-tower** (ver [el fundamento dedicado](/fundamentos/two-tower-retrieval)): una torre codifica al usuario/consulta y otra al ítem, en un **espacio de embeddings compartido**, de modo que la afinidad sea un producto punto. La clave operativa es que los embeddings de ítems se **precomputan offline** y se indexan; en línea solo se calcula el embedding del usuario y se hace una búsqueda de **vecinos más cercanos** (ANN), recuperando el top-K en milisegundos.

Esta idea nació en **DSSM** ([Huang et al. 2013](/papers/dssm-huang-2013)) para emparejar consultas y documentos en búsqueda web, se popularizó como generador de candidatos en **YouTube** ([Covington et al. 2016](/papers/youtube-dnn-covington-2016)) —que modela la recomendación como una clasificación softmax sobre millones de videos— y se formalizó para retrieval con muestreo corregido por sesgo en el **Two-Tower** de [Yi et al. (2019)](/papers/two-tower-yi-2019). La torre de usuario y la separación retrieval/ranking son hoy el patrón arquitectónico dominante en recomendación a escala.

---

## 9. Recomendación con contenido multimodal

Cuando un ítem tiene **contenido rico** —texto, imágenes— podemos incorporarlo a sus representaciones, lo que es central para moda, arte, productos visuales y el [Case Study de la Clase 25](/clases/clase-25), que combina imágenes y texto.

**VBPR** (Visual Bayesian Personalized Ranking, [He y McAuley 2016](/papers/vbpr-he-2016)) extiende BPR incorporando **features visuales** extraídas con una CNN preentrenada: el embedding del ítem se descompone en una parte latente (colaborativa) y una parte visual proyectada desde la imagen. Esto mejora drásticamente el cold start de ítems (una prenda nueva sin interacciones todavía tiene su imagen) y captura la dimensión estética que el colaborativo puro no ve.

**PinSage** ([Ying et al. 2018](/papers/pinsage-ying-2018)) lleva esto a escala web con una **red neuronal de grafos** (GCN) sobre el grafo bipartito pin-tablero de Pinterest: cada ítem agrega información de sus vecinos en el grafo, fusionando contenido visual/textual con la estructura colaborativa. Es uno de los primeros despliegues de GNN a escala de miles de millones de nodos.

El patrón general del recsys multimodal: codificar texto e imagen con encoders preentrenados, **fusionar** esas señales con los factores colaborativos, y entrenar con una pérdida de ranking tipo BPR o contrastiva. Es exactamente la receta del Case Study de la Clase 25.

---

## 10. El problema del cold start

El **cold start** (arranque en frío) es la debilidad estructural del filtrado colaborativo: sin historial de interacciones, no hay nada que factorizar. Tiene tres variantes:

- **Usuario nuevo**: nadie con quien compararlo. Se mitiga con onboarding (preguntar preferencias), features demográficas, o recomendar lo popular hasta acumular señal.
- **Ítem nuevo**: ningún usuario lo tocó aún. Aquí brilla el **contenido**: VBPR usa la imagen, los modelos de texto usan la descripción, de modo que el ítem ya tiene embedding antes de su primera interacción.
- **Sistema nuevo**: sin datos en absoluto; obliga a partir de content-based o reglas.

El cold start es la principal razón por la que los sistemas industriales son **híbridos**: el contenido cubre el vacío que el colaborativo no puede llenar, y la **exploración** (mostrar ítems inciertos, al estilo bandits) acelera la recolección de señal para que el colaborativo tome el relevo.

---

## 11. Evaluación: del RMSE al ranking

La métrica importa tanto como el modelo, y su elección refleja la tarea real. En la era del rating explícito reinaba el **RMSE** (raíz del error cuadrático medio), la métrica del Netflix Prize:

$$
\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u,i)\in\mathcal{T}} (\hat{r}_{ui} - r_{ui})^2}.
$$

El problema: el RMSE penaliza errores en toda la matriz por igual, pero al usuario **solo le mostramos el top-K**. Un modelo puede tener excelente RMSE y aun así ordenar mal los pocos ítems que importan. Por eso, con feedback implícito y objetivo de ranking, la evaluación migró a **métricas de top-K**: Precision@K, Recall@K, MAP, MRR y, sobre todo, **NDCG** (Normalized Discounted Cumulative Gain), que premia colocar lo relevante en las primeras posiciones con un descuento logarítmico por rango. El detalle formal de estas métricas está en el [fundamento de métricas de ranking](/fundamentos/ranking-metrics). En producción, además, ninguna métrica offline reemplaza al **A/B test**: el juez final es el comportamiento real (engagement, retención, ingresos), porque el sistema **influye en los datos que recoge** y el sesgo de exposición distorsiona cualquier evaluación puramente offline.

---

## Para profundizar

- [Matrix Factorization Techniques for Recommender Systems (Koren et al. 2009)](/papers/matrix-factorization-koren-2009) — el artículo canónico de factores latentes, sesgos y la era post-Netflix Prize.
- [BPR: Bayesian Personalized Ranking (Rendle et al. 2009)](/papers/bpr-rendle-2009) — la pérdida pairwise estándar para feedback implícito.
- [Neural Collaborative Filtering (He et al. 2017)](/papers/neural-collaborative-filtering-he-2017) — interacción usuario-ítem aprendida por un MLP.
- [Wide & Deep Learning (Cheng et al. 2016)](/papers/wide-and-deep-cheng-2016) — memorización lineal + generalización profunda.
- [DeepFM (Guo et al. 2017)](/papers/deepfm-guo-2017) — máquinas de factorización fusionadas con redes profundas.
- [DSSM (Huang et al. 2013)](/papers/dssm-huang-2013) — el origen de las torres de embeddings para matching.
- [Deep Neural Networks for YouTube Recommendations (Covington et al. 2016)](/papers/youtube-dnn-covington-2016) — candidate generation + ranking a escala industrial.
- [Sampling-Bias-Corrected Two-Tower (Yi et al. 2019)](/papers/two-tower-yi-2019) — retrieval con corrección de sesgo de muestreo.
- [VBPR: Visual Bayesian Personalized Ranking (He y McAuley 2016)](/papers/vbpr-he-2016) — features visuales para cold start de ítems.
- [PinSage (Ying et al. 2018)](/papers/pinsage-ying-2018) — GNN sobre grafos de contenido a escala web.

**Fundamentos relacionados:** [Two-Tower Retrieval](/fundamentos/two-tower-retrieval) · [Métricas de Ranking](/fundamentos/ranking-metrics) · [Triplet Loss](/fundamentos/triplet-loss) · [Metric Learning](/fundamentos/metric-learning) · [Case Study Clase 25](/clases/clase-25)
