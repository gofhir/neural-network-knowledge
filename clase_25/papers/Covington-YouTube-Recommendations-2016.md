# Deep Neural Networks for YouTube Recommendations (Covington, Adams, Sargin — RecSys 2016)

**Análisis interno exhaustivo** · Clase 25 — Case Study: Recomendación usando Imágenes y Texto · Diplomado IA UC

---

## 1. Contexto histórico

Cuando este paper se publica en RecSys 2016, el estado del arte industrial en recomendación seguía dominado por **matrix factorization** (factorización de matrices) y filtrado colaborativo clásico. El propio paper lo reconoce explícitamente: "In contrast to vast amount of research in matrix factorization methods [19], there is relatively little work using deep neural networks for recommendation systems". Es decir, el deep learning ya había arrasado en visión (AlexNet 2012) y empezaba a transformar NLP y traducción, pero **recomendación seguía siendo un nicho mayormente clásico**. Los pocos trabajos previos con redes neuronales eran: recomendación de noticias [17], de citas [8], de ratings de reviews [20], filtrado colaborativo como red profunda [22] o autoencoders [18], modelado cross-domain de usuarios [5] y recomendación de música basada en contenido por Burges et al. [21].

El sistema descrito es el recomendador de YouTube, que el paper presenta como "one of the largest scale and most sophisticated industrial recommendation systems in existence", sirviendo a **más de mil millones de usuarios**. El predecesor directo era un enfoque de **matrix factorization entrenado bajo rank loss** (referencia [23], el método WSABIE de Weston, Bengio, Usunier), y las primeras iteraciones de la red neuronal "mimicked this factorization behavior with shallow networks that only embedded the user's previous watches". Por eso el paper se posiciona a sí mismo como **"a non-linear generalization of factorization techniques"** — no un reemplazo conceptual radical, sino una generalización profunda.

El contexto tecnológico también es clave: el sistema corre sobre **Google Brain**, recién open-sourced como **TensorFlow** (referencia [1], 2015). Los modelos aprenden "approximately one billion parameters" y se entrenan sobre "hundreds of billions of examples". Este es un paper de ingeniería de producción tanto como de ML.

El paper define tres desafíos que dan forma a todo el diseño:

- **Escala (Scale):** muchos algoritmos que funcionan en problemas pequeños fallan a la escala de YouTube; se requieren algoritmos de entrenamiento distribuido especializados y sistemas de serving eficientes.
- **Frescura (Freshness):** el corpus es dinámico ("many hours of video are uploaded per second"); el sistema debe modelar contenido recién subido y las últimas acciones del usuario. Esto se entiende como un balance exploración/explotación.
- **Ruido (Noise):** el comportamiento histórico es difícil de predecir por sparsity y factores externos no observables. Rara vez se obtiene ground truth de satisfacción; se modela **implicit feedback** ruidoso. Los metadatos están mal estructurados, sin ontología bien definida.

## 2. Contribución central

La contribución central es **doble**:

1. **Demostrar que el deep learning supera a matrix factorization en recomendación industrial a escala masiva**, formulando explícitamente la recomendación bajo la **dicotomía clásica de recuperación de información en dos etapas (two-stage)**: un modelo profundo de **candidate generation** seguido de un modelo profundo separado de **ranking**.

2. **Aportar lecciones prácticas de diseño, iteración y mantenimiento** de un sistema con impacto masivo de cara al usuario — buena parte del valor del paper está en estas "trampas" y trucos que no aparecen en papers puramente académicos: el truco de "example age", la selección del problema sustituto (surrogate), el rollback de historial, el ejemplo "taylor swift", la regresión logística ponderada por watch time, etc.

La arquitectura conceptual unificadora es el **funnel** (embudo, Figura 2): el corpus de **millones** de videos se reduce a **cientos** de candidatos (candidate generation), que luego ranking reduce a **decenas** que se presentan al usuario.

## 3. Arquitectura y método

### 3.1 El esquema two-stage (embudo)

El sistema consta de **dos redes neuronales**:

- **Candidate generation:** toma eventos del historial de actividad del usuario y recupera un subconjunto pequeño (cientos) del corpus grande. Estos candidatos deben ser "generally relevant to the user with **high precision**". Solo provee personalización amplia vía **filtrado colaborativo**, con features gruesos: IDs de videos vistos, tokens de búsqueda, demografía.
- **Ranking:** asigna un score a cada video según una función objetivo deseada usando "a rich set of features describing the video and user", logrando **high recall** en el ranking fino. Los videos de mayor score se presentan ordenados.

La ventaja del two-stage es que permite recomendar desde un corpus enorme (millones) garantizando que los pocos videos finales son personalizados y atractivos. Además **permite mezclar (blending) candidatos de otras fuentes** cuyos scores no son directamente comparables (referencia [3], el sistema de recomendación de YouTube de 2010).

Metodológicamente importante: el equipo usa métricas offline (precision, recall, ranking loss) para guiar iteraciones, pero la determinación final de efectividad se hace vía **A/B testing en experimentos en vivo**, midiendo click-through rate, watch time y engagement. El paper advierte: "live A/B results are not always correlated with offline experiments" — una de las lecciones transversales más citadas del paper.

### 3.2 Candidate generation: recomendación como clasificación extrema multiclase

El modelo plantea la recomendación como **clasificación extrema multiclase**: predecir el video específico $w_t$ visto en el tiempo $t$ entre millones de videos $i$ (clases) del corpus $V$, dado un usuario $U$ y contexto $C$:

$$P(w_t = i \mid U, C) = \frac{e^{v_i u}}{\sum_{j \in V} e^{v_j u}}$$

donde $u \in \mathbb{R}^N$ es el embedding de alta dimensión del par (usuario, contexto) y $v_j \in \mathbb{R}^N$ son embeddings de cada video candidato. Un embedding es "simply a mapping of sparse entities (individual videos, users etc.) into a dense vector in $\mathbb{R}^N$". La tarea de la red profunda es **aprender los embeddings de usuario $u$ como función del historial y contexto** del usuario, útiles para discriminar videos con un clasificador softmax.

Usan **implicit feedback** (watches), no explicit (thumbs up/down, encuestas), porque hay órdenes de magnitud más historial implícito, lo que permite recomendaciones "deep in the tail" donde el feedback explícito es extremadamente escaso. Un usuario que **completa** un video es un ejemplo positivo.

### 3.3 Softmax eficiente sobre millones de clases: negative sampling

Entrenar un softmax con millones de clases es inviable directamente. La solución: **candidate sampling** — muestrear clases negativas de la distribución de fondo y corregir vía **importance weighting** (referencia [10], Jean et al.). Para cada ejemplo se minimiza la **cross-entropy** para la etiqueta verdadera y las negativas muestreadas. En la práctica se muestrean **"several thousand negatives"**, lo que da **"more than 100 times speedup over traditional softmax"**.

Probaron **hierarchical softmax** [15] como alternativa popular, pero **no lograron accuracy comparable**: en hierarchical softmax recorrer cada nodo del árbol implica discriminar entre conjuntos de clases frecuentemente no relacionadas, lo que hace el problema más difícil y degrada el desempeño.

### 3.4 Serving por nearest neighbor

En serving hay que computar las $N$ clases (videos) más probables bajo una latencia estricta de **decenas de milisegundos**, lo que exige un esquema de scoring **sublineal en el número de clases**. Sistemas previos en YouTube usaban hashing [24]. El insight clave: **las likelihoods calibradas del softmax no se necesitan en serving**; entonces el problema se reduce a una **búsqueda de vecino más cercano (nearest neighbor) en el espacio de producto punto**, para lo cual sirven bibliotecas de propósito general [12]. Notablemente, "A/B results were not particularly sensitive to the choice of nearest neighbor search algorithm".

Este es el punto **arquitectónicamente más relevante para la Clase 25**: el modelo aprende un embedding de usuario $u$ y un conjunto de embeddings de video $v_j$, y la inferencia es **distancia / producto punto mínimo en el espacio compartido** — exactamente la lógica de la representación de pin y la inferencia por mínima distancia del case study.

### 3.5 Arquitectura del modelo: embeddings promediados (CBOW-like)

Inspirados en los modelos de lenguaje **continuous bag of words (CBOW)** [14] (Mikolov et al.), aprenden embeddings de alta dimensión para cada video de un vocabulario fijo y los alimentan a una red feedforward. El historial de vistas es una secuencia de largo variable de video IDs sparse, mapeada a vectores densos vía embeddings.

Como la red requiere inputs densos de tamaño fijo, hay que colapsar el bag de vistas. Probaron varias estrategias (sum, component-wise max) y **promediar (averaging) los embeddings funcionó mejor**. Crítico: los embeddings se aprenden **conjuntamente** con todos los demás parámetros vía backpropagation normal por descenso de gradiente. Los features se concatenan en una primera capa ancha, seguida de varias capas fully connected de **ReLU** [6] (Figura 3).

### 3.6 Señales heterogéneas

La ventaja de la red profunda sobre matrix factorization es que **se pueden agregar features continuos y categóricos arbitrarios** fácilmente:

- **Historial de búsqueda:** cada query se tokeniza en unigramas y bigramas, cada token se embebe; promediados, representan un historial de búsqueda denso resumido.
- **Demografía:** importante para dar **priors** que hagan razonables las recomendaciones a usuarios nuevos. Región geográfica y dispositivo se embeben y concatenan. Género, estado de login y edad entran directo como valores reales normalizados a $[0,1]$.

### 3.7 El truco de "Example Age"

Este es uno de los aportes más originales del paper. Como muchas horas de video se suben cada segundo, recomendar contenido **fresco** es vital. Los usuarios prefieren contenido fresco (aunque no a expensas de la relevancia), y existe un fenómeno secundario crítico de **bootstrapping y propagación de contenido viral** [11].

El problema: los sistemas de ML tienen un **sesgo implícito hacia el pasado** porque se entrenan para predecir comportamiento futuro a partir de ejemplos históricos. La distribución de popularidad de videos es altamente **no estacionaria**, pero la distribución multinomial que produce el recomendador reflejará el **promedio de likelihood de la ventana de entrenamiento** (de varias semanas). 

La solución: alimentar la **edad del ejemplo de entrenamiento ("example age") como feature** durante el entrenamiento. En **serving, este feature se fija en cero (o ligeramente negativo)** para reflejar que el modelo predice al final mismo de la ventana de entrenamiento. La Figura 4 muestra que el modelo con example age representa con precisión el tiempo de subida y la popularidad dependiente del tiempo (un pico agudo justo tras la subida), mientras que sin el feature el modelo predeciría aproximadamente el promedio de la ventana.

### 3.8 Selección de etiqueta y contexto (surrogate problem)

El paper enfatiza que recomendación implica resolver un **problema sustituto (surrogate)** y transferir el resultado a un contexto particular. La elección del surrogate tiene "an outsized importance on performance in A/B testing but is very difficult to measure with offline experiments". Lecciones concretas:

- **Generar ejemplos desde TODAS las vistas de YouTube** (incluso videos embebidos en otros sitios), no solo desde las recomendaciones producidas. Si no, el contenido nuevo no podría aflorar y el recomendador estaría sesgado a la explotación.
- **Número fijo de ejemplos de entrenamiento por usuario**, pesando a los usuarios por igual en la función de pérdida. Esto evita que una cohorte pequeña de usuarios muy activos domine la pérdida.
- **Withhold information from the classifier (ocultar información):** ejemplo "taylor swift" — si el usuario acaba de buscar "taylor swift", un clasificador que predice el siguiente video visto predecirá los videos de la página de resultados de esa búsqueda; reproducir la última página de búsqueda como recomendaciones de home funciona muy mal. Solución: descartar información de secuencia y representar queries como un **bag of tokens desordenado**, de modo que el clasificador no sea consciente directamente del origen de la etiqueta.
- **Predecir la PRÓXIMA vista, no una vista aleatoria retenida (Figura 5).** Los patrones de consumo son **asimétricos** (series episódicas se ven secuencialmente; los usuarios descubren artistas empezando por lo más popular). Muchos sistemas de filtrado colaborativo retienen un ítem aleatorio y lo predicen desde el resto (5a) — esto **filtra información futura** e ignora la asimetría. En cambio, se hace **"rollback" del historial**: se elige una vista aleatoria y solo se ingresan acciones **anteriores** a la etiqueta retenida (5b). En 5b, example age se expresa como $t_{\max} - t_N$.

### 3.9 Ranking

El ranking **especializa y calibra** las predicciones de candidatos para la interfaz particular usando datos de impresión. Tiene acceso a muchos más features porque solo se scorean unos cientos de videos (no millones). Es crucial también para hacer ensembling de fuentes de candidatos cuyos scores no son comparables.

Arquitectura similar a candidate generation, pero con **regresión logística** (Figura 7). El objetivo de ranking se ajusta constantemente vía A/B, pero es generalmente una función simple del **expected watch time per impression**. Rankear por click-through rate promueve "clickbait" (videos engañosos que el usuario no completa); el watch time captura mejor el engagement [13, 25].

**Representación de features:** taxonomía categórica vs continua/ordinal; univalent (un valor, ej. el video ID de la impresión) vs multivalent (conjunto, ej. los últimos N video IDs vistos); query (computados una vez por request) vs impression (por cada ítem scoreado). Usan **cientos de features**, repartidos ~mitad y mitad. Pese a la promesa del deep learning de eliminar feature engineering, "we still expend considerable engineering resources transforming user and video data into useful features". El desafío principal es representar la secuencia temporal de acciones del usuario y su relación con la impresión scoreada.

Señales más importantes: la **interacción previa del usuario con el ítem mismo y con ítems similares** (matching la experiencia en ranking de ads [7]). Ejemplos: cuántos videos vio el usuario de ese canal; cuándo fue la última vez que vio un video de ese tema. Es crucial **propagar información de candidate generation a ranking como features** (qué fuentes nominaron al candidato, qué scores asignaron). Features de **frecuencia de impresiones pasadas** son clave para introducir "churn" (que requests sucesivos no devuelvan listas idénticas): si se recomendó un video y no se vio, el modelo lo demota en la siguiente carga.

**Embeddings de categóricos:** cada espacio de ID (vocabulario) tiene su embedding aprendido con dimensión que crece "approximately proportional to the logarithm of the number of unique values". Vocabularios de cardinalidad muy alta se truncan al top N por frecuencia en impresiones clickeadas; out-of-vocabulary → embedding cero. **Embeddings compartidos:** existe un único embedding global de video IDs que muchos features distintos usan, aunque cada feature se alimenta por separado para que las capas superiores aprendan representaciones especializadas. Compartir embeddings mejora generalización, acelera entrenamiento y reduce memoria. Dato notable: la mayoría abrumadora de parámetros está en estos embeddings de alta cardinalidad — "one million IDs embedded in a 32 dimensional space have 7 times more parameters than fully connected layers 2048 units wide".

**Normalización de continuos:** las redes neuronales son sensibles a la escala/distribución de inputs [9]. Un feature continuo $x$ con distribución $f$ se transforma a $\tilde{x}$ escalándolo para que quede equidistribuido en $[0,1)$ usando la CDF: $\tilde{x} = \int_{-\infty}^{x} df$, aproximada por interpolación lineal sobre cuantiles computados en una sola pasada. Además se ingresan las potencias $\tilde{x}^2$ y $\sqrt{\tilde{x}}$, dando más poder expresivo (funciones super- y sublineales). Estas potencias mejoraron accuracy offline.

### 3.10 Modelando expected watch time: weighted logistic regression

Objetivo: predecir el **expected watch time** dados ejemplos positivos (impresión clickeada) o negativos (no clickeada). Los positivos se anotan con el tiempo que el usuario vio el video. Técnica: **weighted logistic regression** (desarrollada para este propósito). El modelo se entrena con regresión logística bajo cross-entropy, pero las **impresiones positivas (clickeadas) se ponderan por el watch time observado**; las negativas reciben peso unitario.

Así, las odds aprendidas por la logística son:

$$\text{odds} = \frac{\sum T_i}{N - k}$$

donde $N$ es el número de ejemplos de entrenamiento, $k$ el número de impresiones positivas, y $T_i$ el watch time de la $i$-ésima impresión. Asumiendo que la fracción de positivos es pequeña (cierto en su caso), las odds aprendidas son aproximadamente $E[T](1 + P)$, con $P$ la probabilidad de click y $E[T]$ el watch time esperado. Como $P$ es pequeño, el producto es cercano a $E[T]$. En inferencia usan la **función exponencial $e^x$** como activación final para producir estas odds que estiman expected watch time.

## 4. Experimentos y resultados clave

### 4.1 Candidate generation: features y profundidad (Figura 6)

Configuración experimental: vocabulario de **1M videos** y **1M search tokens**, embebidos con **256 floats cada uno**, bag máximo de **50 vistas recientes** y **50 búsquedas recientes**. La capa softmax emite una multinomial sobre las mismas 1M clases de video con dimensión 256 (un embedding de video de salida separado). Entrenados hasta convergencia sobre todos los usuarios (varias épocas). Estructura "tower": la base es la más ancha y cada capa oculta sucesiva **divide a la mitad** el número de unidades:

- **Depth 0:** capa lineal que transforma la concatenación a dim 256 (equivalente al sistema predecesor, factorización lineal).
- **Depth 1:** 256 ReLU
- **Depth 2:** 512 → 256 ReLU
- **Depth 3:** 1024 → 512 → 256 ReLU
- **Depth 4:** 2048 → 1024 → 512 → 256 ReLU

Resultado (Figura 6, métrica **Holdout MAP %**): tanto agregar **features** (más allá de embeddings de video: búsquedas, example age, all features) como agregar **profundidad** mejoran monótonamente el MAP de holdout. La curva "All Features" con depth 3-4 alcanza ~13% MAP frente a ~6-7% de "Watches Only" en depth 0. Las capas de profundidad agregan expresividad para modelar la interacción de estos features adicionales.

### 4.2 Ranking: capas ocultas (Tabla 1)

Métrica: **weighted, per-user loss** sobre next-day holdout. Se scorean una impresión positiva (clickeada) y una negativa de la misma página; si la negativa recibe mayor score que la positiva, se considera el watch time de la positiva como "mispredicted watch time". La pérdida es el total de watch time mispredicho como fracción del watch time total sobre los pares de holdout.

| Hidden layers | weighted, per-user loss |
|---|---|
| None | 41.6% |
| 256 ReLU | 36.9% |
| 512 ReLU | 36.7% |
| 1024 ReLU | 35.8% |
| 512 → 256 ReLU | 35.2% |
| 1024 → 512 ReLU | 34.7% |
| 1024 → 512 → 256 ReLU | **34.6%** |

Más ancho y más profundo mejora, con el trade-off de tiempo de CPU en serving. La config **1024 → 512 → 256** dio los mejores resultados dentro del presupuesto de CPU. Dos ablations sobre esa config:

- Alimentar solo los continuos normalizados **sin sus potencias** → **+0.2%** de loss (peor).
- Pesar positivos y negativos por igual (en vez de ponderar por watch time) → **+4.1%** de loss (dramáticamente peor). Esto valida el núcleo de la weighted logistic regression.

## 5. Limitaciones

- **Feature engineering manual sigue siendo necesario:** pese a la promesa del deep learning, "we still expend considerable engineering resources transforming user and video data". El paper es honesto en que el deep learning no eliminó el trabajo de features en este dominio.
- **Métricas offline poco correlacionadas con A/B en vivo:** una limitación metodológica central reconocida; la elección del surrogate problem casi solo puede evaluarse en vivo.
- **Trucos específicos del dominio:** withhold information, rollback de historial, example age — son ingeniería del problema, no resultados de principios generales fácilmente transferibles.
- **Infraestructura de serving up-to-the-second:** "an engineering feat onto itself outside the scope of this paper" — el sistema asume infraestructura masiva no descrita.
- **Ausencia de detalles cuantitativos del A/B en producción:** los números reportados son holdout offline (MAP, weighted per-user loss); las mejoras "dramáticas" de watch time en A/B se mencionan cualitativamente.
- **Sin código ni dataset público:** no es reproducible directamente (esto es esperable en un paper industrial de Google).
- **Hierarchical softmax descartado empíricamente** sin un análisis profundo del por qué más allá de la observación de discriminación entre clases no relacionadas.

## 6. Impacto y legado

Este paper se convirtió en **uno de los papers de recsys industrial más influyentes de la década**. Estableció varios patrones que se volvieron canónicos:

1. **El patrón two-stage (retrieval + ranking)** es hoy el estándar de facto en sistemas de recomendación a gran escala (YouTube, Pinterest, Meta, TikTok, etc.).
2. **Recuperación por embeddings + nearest neighbor** (el embedding de usuario y el de ítem en un espacio compartido, recuperando por producto punto) es el germen directo de la familia de modelos **two-tower** que dominaría retrieval en la segunda mitad de la década (y que conecta con FAISS, ScaNN, etc.).
3. **Optimizar watch time / engagement en lugar de CTR** se volvió doctrina en ranking, junto con la conciencia del problema del "clickbait".
4. **El truco de example age** y la conciencia del sesgo hacia el pasado / no estacionariedad influyó en el modelado de freshness en muchos sistemas.
5. **Las "lecciones prácticas"** (surrogate problem, withhold information, A/B vs offline, pesar usuarios por igual) son citadas como sabiduría operacional.

## 7. Conexión con la Clase 25 (Case Study multimodal de Pinterest)

La Clase 25 cita este paper explícitamente como **"Inspiration"**, y la conexión es estructural, no superficial:

- **Representación en un espacio de embeddings compartido + inferencia por distancia mínima.** El case study modela $r_{ij} = h(g(u_i), f(x_j, c_j))$: una representación de usuario $g(u_i)$ y una representación de pin $f(x_j, c_j)$, comparadas para recomendar. Esto es exactamente la idea de candidate generation de Covington: aprender un embedding de usuario $u$ y embeddings de ítem $v_j$ en $\mathbb{R}^N$, y recuperar por **producto punto / nearest neighbor**. El "serving por nearest neighbor en el espacio de producto punto" de la Sección 3.1 es el ancestro directo de la inferencia por mínima distancia del case study.

- **Metric learning.** Mientras Covington usa softmax muestreado sobre millones de clases como objetivo, ambos comparten la idea de **aprender un espacio donde la cercanía implica relevancia**. El case study de Pinterest lo lleva a un esquema explícito de metric learning, pero la motivación (recuperar por distancia a escala) es la misma.

- **Fusión de señales heterogéneas → concatenación → capas FC.** Covington concatena embeddings de vistas promediados + búsquedas promediadas + features demográficos + example age en una primera capa ancha, seguida de ReLUs. El case study multimodal hace lo análogo pero **multimodal**: la imagen del pin pasa por una **CNN**, el comentario por **BERT**, y ambos se **concatenan → FC → representación de pin**. La estrategia de "embeber cada fuente y concatenar antes de las capas densas" es heredada directamente de la arquitectura de Figura 3.

- **Promediado de embeddings de historial.** El truco CBOW de promediar embeddings de las vistas del usuario para obtener un vector de tamaño fijo es la base de cómo el case study representa el historial/contexto del usuario $g(u_i)$ a partir de sus pins/interacciones previas.

- **Two-stage en producción.** Aunque el case study se centra en la arquitectura de representación multimodal, opera en el mismo régimen de "millones de ítems, recuperar por embedding, rankear por score" que Covington formalizó. Pinterest tiene millones de pins; recuperar candidatos por embedding multimodal y luego rankear es la aplicación natural del embudo.

- **Métricas de ranking.** El case study evalúa con métricas de ranking; Covington fija el precedente de evaluar candidate generation con MAP de holdout y ranking con pérdidas pareadas ponderadas, además del énfasis en A/B en vivo.

En síntesis: **Covington 2016 aporta el esqueleto conceptual** (aprende embeddings de usuario e ítem, recupera por distancia, fusiona señales heterogéneas por concatenación + FC, optimiza engagement), y la **Clase 25 lo extiende al caso multimodal** reemplazando los embeddings de ID por **representaciones aprendidas de imagen (CNN) y texto (BERT)** de cada pin. El paper es, literalmente, la inspiración arquitectónica del case study.

## 8. Referencias internas del paper relevantes

- [10] Jean et al. — sampling de vocabulario objetivo grande (negative sampling / importance weighting).
- [12] Liu et al. — algoritmos prácticos de approximate nearest neighbor (serving).
- [14] Mikolov et al. — CBOW / word2vec (inspiración del averaging de embeddings).
- [15] Morin & Bengio — hierarchical softmax (alternativa descartada).
- [23] Weston et al. (WSABIE) — el predecesor de matrix factorization bajo rank loss.
- [7] He et al. — practical lessons predicting clicks on ads at Facebook (inspiración de features de interacción).
- [13, 25] — justificación de watch time sobre CTR.
