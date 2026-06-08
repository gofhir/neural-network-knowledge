---
title: "Deep Neural Networks for YouTube Recommendations"
weight: 250
math: true
---

{{< paper-card
    title="Deep Neural Networks for YouTube Recommendations"
    authors="Covington, Adams, Sargin"
    year="2016"
    venue="RecSys 2016"
    pdf="/papers/youtube-dnn-covington-2016.pdf" >}}
El paper que llevó el **deep learning a la recomendación industrial a escala masiva** (mil millones de usuarios). Formaliza el patrón **two-stage**: una red de **candidate generation** que reduce millones de videos a cientos vía clasificación extrema multiclase, y una red de **ranking** que los ordena por **expected watch time**. El truco clave: aprender un **embedding de usuario** y embeddings de video en un espacio compartido, y recuperar por **vecino más cercano en el espacio de producto punto**. Es la inspiración arquitectónica directa del case study multimodal de la [Clase 25](/clases/clase-25).
{{< /paper-card >}}

---

## Contexto

En 2016 la recomendación industrial seguía dominada por **matrix factorization** y filtrado colaborativo clásico; el deep learning ya había transformado visión y NLP pero, como reconoce el paper, "there is relatively little work using deep neural networks for recommendation systems". El predecesor de este sistema era una factorización de matrices entrenada bajo rank loss (WSABIE), y las primeras redes neuronales del equipo "mimicked this factorization behavior with shallow networks". Por eso el modelo se presenta como **"a non-linear generalization of factorization techniques"**.

El sistema corre sobre Google Brain / TensorFlow, aprende **~mil millones de parámetros** y se entrena sobre **cientos de miles de millones de ejemplos**. Tres desafíos moldean todo el diseño: **escala** (millones de videos, mil millones de usuarios), **frescura** (horas de video subidas por segundo) y **ruido** (se modela *implicit feedback*, no ground truth de satisfacción).

Ver fundamentos: [/fundamentos/recommender-systems](/fundamentos/recommender-systems), [/fundamentos/two-tower-retrieval](/fundamentos/two-tower-retrieval), [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics).

## Ideas principales

### El embudo two-stage

El sistema son **dos redes neuronales** en un embudo: **candidate generation** reduce el corpus de **millones** a **cientos** de candidatos (personalización amplia, alta precisión), y **ranking** los reduce a **decenas** presentables (representación fina, alto recall). El two-stage permite recomendar desde un corpus enorme garantizando que los pocos finales son personalizados, y permite **mezclar candidatos de otras fuentes** cuyos scores no son comparables.

### Candidate generation: clasificación extrema multiclase

Se plantea recomendar como predecir el video $w_t$ visto en el tiempo $t$ entre millones de clases, dado usuario $U$ y contexto $C$:

$$P(w_t = i \mid U, C) = \frac{e^{v_i u}}{\sum_{j \in V} e^{v_j u}}$$

donde $u \in \mathbb{R}^N$ es el embedding del par (usuario, contexto) y $v_j \in \mathbb{R}^N$ los embeddings de cada video. La red aprende $u$ como **función del historial y contexto** del usuario. Se usa **implicit feedback** (completar un video = positivo) por su abundancia, lo que permite recomendaciones "deep in the tail".

### Softmax eficiente: negative sampling

Entrenar softmax sobre millones de clases es inviable directo. Se muestrean **varios miles de negativos** de la distribución de fondo (*candidate sampling*) corrigiendo vía *importance weighting*, minimizando cross-entropy para la etiqueta verdadera y los negativos. Esto da **más de 100× de speedup** sobre softmax tradicional. Probaron **hierarchical softmax** pero no lograron accuracy comparable.

### Serving por nearest neighbor

En serving hay que computar los top-$N$ videos bajo latencia de **decenas de milisegundos**, lo que exige scoring **sublineal**. El insight: las likelihoods calibradas del softmax **no se necesitan** en serving, así que el problema se reduce a **búsqueda de vecino más cercano en el espacio de producto punto** entre $u$ y los $v_j$. Los A/B "were not particularly sensitive to the choice of nearest neighbor search algorithm". Este es el germen directo de los modelos [two-tower](/fundamentos/two-tower-retrieval).

### Embeddings promediados (estilo CBOW)

Inspirados en **continuous bag of words** (word2vec), embeben cada video del vocabulario y representan el historial de vistas (secuencia de largo variable) **promediando** los embeddings — averaging funcionó mejor que sum o component-wise max. Los embeddings se aprenden **conjuntamente** vía backprop. Los features se concatenan en una primera capa ancha seguida de varias capas **ReLU**.

### Señales heterogéneas y el truco de "Example Age"

Se agregan features arbitrarios: búsqueda (tokenizada en unigramas/bigramas, embebida y promediada), demografía (región y dispositivo embebidos; género/edad/login normalizados a $[0,1]$). Para combatir el sesgo hacia el pasado y la no estacionariedad de la popularidad, se alimenta la **edad del ejemplo de entrenamiento** como feature; en **serving se fija en 0** (o ligeramente negativo). Esto permite modelar el pico de popularidad justo tras la subida de un video, en vez de predecir el promedio de la ventana.

### Selección del problema sustituto (surrogate)

Lecciones de gran impacto en A/B (difíciles de medir offline): generar ejemplos desde **todas** las vistas (no solo desde recomendaciones propias), **número fijo de ejemplos por usuario** (para que usuarios muy activos no dominen la pérdida), **ocultar información** al clasificador (ejemplo "taylor swift": representar la búsqueda como bag of tokens desordenado para no reproducir la página de resultados), y **predecir la próxima vista, no una vista aleatoria retenida** — esto respeta la asimetría del consumo y evita filtrar información futura ("rollback" del historial).

### Ranking: weighted logistic regression sobre watch time

Ranking especializa los candidatos para la interfaz usando cientos de features (interacción previa del usuario con el ítem/canal, frecuencia de impresiones para introducir "churn", features propagados desde candidate generation). Categóricos van a embeddings compartidos; continuos se normalizan vía CDF a $[0,1)$ y se agregan potencias $\tilde{x}^2$ y $\sqrt{\tilde{x}}$.

El objetivo es **expected watch time**, no CTR (rankear por CTR promueve "clickbait"). Se usa **weighted logistic regression**: positivos ponderados por el watch time observado, negativos con peso unitario. Las odds aprendidas son:

$$\text{odds} = \frac{\sum T_i}{N - k} \approx E[T](1 + P) \approx E[T]$$

con $N$ ejemplos, $k$ positivos, $T_i$ el watch time, $P$ pequeña la probabilidad de click. En inferencia la activación final es $e^x$.

## Resultados experimentales

**Candidate generation (Holdout MAP %, vocabulario 1M videos / 1M tokens, embeddings de 256, bag de 50 vistas + 50 búsquedas):** tanto agregar features como agregar profundidad mejoran monótonamente el MAP. Estructura "tower" (cada capa divide a la mitad):

| Profundidad | Configuración |
|---|---|
| 0 | lineal → 256 (≈ factorización, igual al predecesor) |
| 1 | 256 ReLU |
| 2 | 512 → 256 ReLU |
| 3 | 1024 → 512 → 256 ReLU |
| 4 | 2048 → 1024 → 512 → 256 ReLU |

"All Features" con profundidad alta alcanza ~13% MAP frente a ~6-7% de "Watches Only" en profundidad 0.

**Ranking (weighted, per-user loss sobre next-day holdout; menor es mejor):**

| Hidden layers | weighted, per-user loss |
|---|---|
| None | 41.6% |
| 256 ReLU | 36.9% |
| 512 ReLU | 36.7% |
| 1024 ReLU | 35.8% |
| 512 → 256 ReLU | 35.2% |
| 1024 → 512 ReLU | 34.7% |
| **1024 → 512 → 256 ReLU** | **34.6%** |

Ablations sobre la config 1024→512→256: quitar las potencias de los continuos **+0.2%** de loss; pesar positivos y negativos por igual (sin ponderar por watch time) **+4.1%** de loss — validación directa de la weighted logistic regression.

## Limitaciones reconocibles

- **El feature engineering manual sigue siendo necesario:** "we still expend considerable engineering resources transforming user and video data".
- **Las métricas offline no siempre correlacionan con el A/B en vivo**, lo que vuelve difícil evaluar la elección del surrogate problem.
- Varios aportes (example age, withhold information, rollback) son **trucos específicos del dominio**, no principios generales fácilmente transferibles.
- La infraestructura de serving up-to-the-second queda "outside the scope of this paper".
- Sin código ni dataset público; las mejoras "dramáticas" de watch time en A/B se reportan cualitativamente.

## Por qué importa hoy

Este paper fijó patrones hoy canónicos en recomendación a gran escala: el **embudo retrieval + ranking** (estándar en YouTube, Pinterest, Meta, TikTok), la **recuperación por embeddings + nearest neighbor** como ancestro directo de los modelos **two-tower** (FAISS, ScaNN), el énfasis en **optimizar engagement/watch time sobre CTR**, y un cuerpo de **sabiduría operacional** (surrogate problem, A/B vs offline, pesar usuarios por igual, manejo de freshness) que se sigue citando como referencia práctica.

## Conexión con la Clase 25

La [Clase 25](/clases/clase-25) cita este paper explícitamente como **"Inspiration"**, y la conexión es estructural:

- **Espacio de embeddings compartido + inferencia por distancia mínima.** El case study modela $r_{ij} = h(g(u_i), f(x_j, c_j))$: una representación de usuario $g(u_i)$ y una de pin $f(x_j, c_j)$ comparadas para recomendar. Es exactamente la idea de candidate generation: aprender $u$ y $v_j$ en $\mathbb{R}^N$ y recuperar por **producto punto / vecino más cercano**.
- **Fusión de señales por concatenación → FC.** Covington concatena vistas promediadas + búsquedas + demografía + example age en una capa ancha seguida de ReLUs. El case study lo hace **multimodal**: imagen del pin por **CNN**, comentario por **BERT**, ambos **concatenados → FC → representación de pin**. La estrategia "embeber cada fuente y concatenar antes de las densas" viene de aquí.
- **Promediado de embeddings de historial** (estilo CBOW) es la base de cómo se representa el contexto del usuario $g(u_i)$.
- **Metric learning y métricas de ranking:** ambos aprenden un espacio donde cercanía = relevancia y evalúan con métricas de ranking ([/fundamentos/ranking-metrics](/fundamentos/ranking-metrics)).

En síntesis, Covington 2016 aporta el **esqueleto conceptual** (embeddings de usuario e ítem, recuperación por distancia, fusión de señales heterogéneas, optimizar engagement) y la Clase 25 lo **extiende al caso multimodal**, reemplazando embeddings de ID por representaciones aprendidas de imagen y texto.

## Notas y enlaces

- **PDF:** [/papers/youtube-dnn-covington-2016.pdf](/papers/youtube-dnn-covington-2016.pdf)
- **Venue:** RecSys 2016 (ACM, Boston). DOI: 10.1145/2959100.2959190.
- **Fundamentos relacionados:** [recommender-systems](/fundamentos/recommender-systems) · [two-tower-retrieval](/fundamentos/two-tower-retrieval) · [ranking-metrics](/fundamentos/ranking-metrics)
- **Clase:** [Clase 25 — Recomendación usando Imágenes y Texto](/clases/clase-25)
- **Linaje técnico:** word2vec/CBOW (Mikolov 2013) → averaging de embeddings; importance sampling de vocabulario grande (Jean et al. 2014) → negative sampling; approximate nearest neighbor (Liu et al. 2004) → serving.
