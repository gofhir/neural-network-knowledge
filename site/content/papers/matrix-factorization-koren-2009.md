---
title: "Matrix Factorization Techniques for Recommender Systems"
weight: 253
math: true
---

{{< paper-card
    title="Matrix Factorization Techniques for Recommender Systems"
    authors="Koren, Bell, Volinsky"
    year="2009"
    venue="IEEE Computer 2009"
    pdf="/papers/matrix-factorization-koren-2009.pdf" >}}
Escrito por el equipo BellKor en la cúspide del **Netflix Prize**, este es el artículo canónico que consolidó la **factorización de matrices** como la metodología dominante del filtrado colaborativo. Caracteriza usuarios e ítems mediante vectores de factores latentes $q_i, p_u \in \mathbb{R}^f$ cuya interacción $q_i^T p_u$ predice la calificación, y muestra cómo extender ese núcleo con sesgos, feedback implícito, dinámica temporal y niveles de confianza. Es el ancestro directo de los embeddings y del scoring por similitud que usan los recomendadores neuronales y multimodales de hoy.
{{< /paper-card >}}

---

## Contexto

A fines de los 2000, los [sistemas de recomendación](/fundamentos/recommender-systems) se organizaban en dos estrategias. El **content filtering** construye perfiles explícitos de usuario y producto (género de una película, actores, demografía del usuario) y los empareja; su ejemplo célebre es el Music Genome Project de Pandora, donde analistas humanos puntúan cientos de "genes" musicales. Su debilidad es que exige información externa costosa de recopilar. El **collaborative filtering** se apoya solo en el comportamiento pasado —calificaciones, transacciones— sin perfiles explícitos. Es *domain free* y suele ser más preciso, pero sufre el **cold start**: no maneja productos ni usuarios nuevos sin historial.

Dentro del filtrado colaborativo conviven los **métodos de vecindad** (estimar la preferencia desde ítems o usuarios "vecinos") y los **modelos de factores latentes** (explicar las calificaciones con 20 a 100 factores inferidos de los patrones). El artículo defiende la tesis, validada por el **Netflix Prize**, de que la factorización de matrices —una realización de los factores latentes— **supera a los vecinos más cercanos** y, además, admite incorporar información adicional. El contexto es la competencia que Netflix lanzó en 2006: 1 millón de dólares por mejorar 10% el RMSE, un dataset de más de 100 millones de calificaciones (~500.000 usuarios, más de 17.000 películas, escala 1-5), y más de 48.000 equipos participantes. Los autores son el equipo **BellKor**, primer lugar en 2007.

## Ideas principales

### Predicción como producto punto en un espacio latente

Cada ítem se asocia a $q_i \in \mathbb{R}^f$ y cada usuario a $p_u \in \mathbb{R}^f$. Los elementos de $q_i$ miden cuánto posee el ítem cada factor; los de $p_u$, cuánto le interesan al usuario ítems altos en ese factor. La calificación se aproxima por su producto punto:

$$\hat{r}_{ui} = q_i^T p_u$$

El modelo se emparenta con la **SVD**, pero la SVD clásica está indefinida con la matriz incompleta y dispersa de calificaciones, y rellenar los huecos por **imputación** es caro y distorsiona. La solución del artículo es **modelar solo las calificaciones observadas** y evitar el sobreajuste con regularización.

### Función de costo regularizada

Sobre el conjunto $\kappa$ de pares $(u,i)$ con calificación conocida:

$$\min_{q^*, p^*} \sum_{(u,i)\in\kappa} (r_{ui} - q_i^T p_u)^2 + \lambda(\|q_i\|^2 + \|p_u\|^2)$$

La constante $\lambda$ penaliza la magnitud de los parámetros para generalizar a calificaciones futuras y se fija por validación cruzada.

### Aprendizaje: SGD vs. ALS

Hay dos formas de minimizar el costo. El **descenso de gradiente estocástico** ([SGD](/fundamentos/optimizadores)), popularizado por Simon Funk, recorre las calificaciones, computa el error $e_{ui} = r_{ui} - q_i^T p_u$ y actualiza en dirección opuesta al gradiente con tasa $\gamma$:

$$q_i \leftarrow q_i + \gamma \,(e_{ui}\, p_u - \lambda\, q_i)$$
$$p_u \leftarrow p_u + \gamma \,(e_{ui}\, q_i - \lambda\, p_u)$$

Los **mínimos cuadrados alternados (ALS)** explotan que, aunque el problema conjunto no es convexo, fijar una incógnita lo vuelve cuadrático y resoluble de forma exacta; ALS alterna entre recomputar los $q_i$ y los $p_u$. ALS conviene cuando se puede paralelizar masivamente (cada factor es independiente) y cuando los datos son implícitos (no dispersos), donde recorrer caso por caso sería impráctico.

### Sesgos (biases)

Mucha variación viene de efectos ajenos a la interacción: usuarios que califican alto, ítems populares. Se modelan con sesgos aditivos:

$$\hat{r}_{ui} = \mu + b_i + b_u + q_i^T p_u$$

con media global $\mu$, sesgo de ítem $b_i$ y de usuario $b_u$. Ejemplo del artículo: si $\mu = 3.7$, *Titanic* tiende a $+0.5$ y Joe (crítico) a $-0.3$, la estimación de primer orden es $3.9$ estrellas. La función de costo penaliza también $b_u^2$ y $b_i^2$.

### Feedback implícito y atributos

Para mitigar el cold start se suman señales. Con $N(u)$ el conjunto de ítems preferidos implícitamente y un segundo vector de ítem $x_i$, más atributos demográficos $A(u)$ con vectores $y_a$, la representación de usuario se enriquece:

$$\hat{r}_{ui} = \mu + b_i + b_u + q_i^T \left[ p_u + |N(u)|^{-0.5}\!\!\sum_{i\in N(u)} x_i + \sum_{a\in A(u)} y_a \right]$$

### Dinámica temporal

La percepción y el gusto cambian. Se hacen funciones del tiempo el sesgo de ítem $b_i(t)$, el de usuario $b_u(t)$ y las preferencias $p_u(t)$, mientras los factores de ítem $q_i$ se mantienen **estáticos** (los ítems no cambian de naturaleza):

$$\hat{r}_{ui}(t) = \mu + b_i(t) + b_u(t) + q_i^T p_u(t)$$

### Niveles de confianza

No toda observación pesa igual (publicidad masiva, usuarios adversarios, feedback implícito binario). Se adjunta un score $c_{ui}$ que pondera el término de error:

$$\min_{p^*,q^*,b^*} \sum_{(u,i)\in\kappa} c_{ui}(r_{ui} - \mu - b_u - b_i - p_u^T q_i)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2)$$

## Resultados experimentales

Las entradas ganadoras del equipo combinaron **más de 100 conjuntos de predictores**, la mayoría modelos de factorización. Al factorizar la matriz de Netflix, los **dos primeros factores resultan interpretables** (Figura 3 del artículo): el eje 1 separa comedias chabacanas y terror para público masculino/adolescente (*Half Baked*, *Freddy vs. Jason*) de dramas con protagonistas femeninas fuertes (*Sophie's Choice*, *Moonstruck*); el eje 2 separa cine independiente y peculiar (*Punch-Drunk Love*) de filmes formulaicos (*Armageddon*). *The Wizard of Oz* queda al centro, gustando a todos.

Sobre la complejidad del modelo (Figura 4), el artículo reporta que la precisión **mejora al aumentar la dimensionalidad** $f$ y al refinar el modelo (sesgos → feedback implícito → componentes temporales), siendo los **efectos temporales particularmente importantes**. Como referencias, el sistema propio de Netflix logra **RMSE = 0,9514** sobre ese dataset y el grand prize exigía **RMSE = 0,8563** (la meta del 10%).

## Limitaciones reconocibles

- **Cold start.** El feedback implícito y la demografía lo mitigan, pero no lo resuelven; los autores admiten que ahí el content filtering es superior.
- **No convexidad.** El problema conjunto no es convexo; SGD no garantiza óptimo global.
- **Interacción lineal.** Modelar la interacción como producto punto captura solo relaciones bilineales en el espacio latente; sin no linealidades ni interacciones de orden superior.
- **Métrica única.** Se optimiza RMSE (predicción de calificación), no ranking top-N ni diversidad/novedad.
- **Ensemble poco desplegable.** El resultado ganador requirió más de 100 predictores combinados, impráctico en producción.

## Por qué importa hoy

Este artículo convirtió un conjunto de trucos de competencia en un **marco unificado y didáctico**, y dejó ideas que perduran. Los vectores $q_i$ y $p_u$ son, en esencia, **embeddings** aprendidos por descenso de gradiente: la misma idea que subyace a word2vec, a las capas de embedding neuronales y a los two-tower / dual-encoder de los recomendadores actuales. El scoring $q_i^T p_u$ es el ancestro del retrieval por similitud de embeddings (ANN, FAISS) que usan YouTube y Spotify. El modelado aditivo de sesgos anticipa Wide & Deep; la formulación de confianza y feedback implícito es la base del trabajo posterior sobre datos implícitos; y la dinámica temporal prefigura los recomendadores secuenciales (GRU4Rec, SASRec). Sigue siendo un **baseline obligatorio** en cualquier benchmark moderno.

## Conexión con la Clase 25

La [Clase 25](/clases/clase-25) es un Case Study de **recomendación multimodal**, y este artículo es su punto de partida histórico. La transición clave es **de factores latentes a embeddings neuronales**: donde Koren et al. infieren $q_i$ y $p_u$ puramente del patrón de calificaciones, un recomendador multimodal deriva esos vectores de la imagen del producto (CNN/ViT), su texto (transformers) o su audio. El producto punto sigue ahí; lo que cambia es la arquitectura que genera los vectores. Además, el **cold start** —la limitación que el artículo solo mitiga— es exactamente lo que la información multimodal resuelve: un producto nuevo no tiene calificaciones, pero sí foto y descripción de las que extraer su embedding inicial. La Clase 25 puede leerse como "¿qué pasa cuando reemplazamos los factores inferidos solo de ratings por embeddings multimodales profundos, conservando la columna vertebral de scoring por similitud que este artículo estableció en 2009?".

## Notas y enlaces

- **Cita:** Y. Koren, R. Bell y C. Volinsky, "Matrix Factorization Techniques for Recommender Systems", *IEEE Computer*, vol. 42, n.º 8, pp. 42-49, agosto 2009.
- **PDF:** [/papers/matrix-factorization-koren-2009.pdf](/papers/matrix-factorization-koren-2009.pdf)
- **Fundamentos relacionados:** [sistemas de recomendación](/fundamentos/recommender-systems), [optimizadores](/fundamentos/optimizadores) (SGD).
- **Clase:** [Clase 25 — Case Study de recsys multimodal](/clases/clase-25).
- Trabajos clave citados en el artículo: Funk (Netflix Update, 2006), Salakhutdinov & Mnih (Probabilistic Matrix Factorization, NIPS 2007), Hu, Koren & Volinsky (Collaborative Filtering for Implicit Feedback Datasets, ICDM 2008), Koren (Collaborative Filtering with Temporal Dynamics, KDD 2009).
