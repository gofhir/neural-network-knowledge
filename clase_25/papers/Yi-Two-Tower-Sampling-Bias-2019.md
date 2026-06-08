# Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations

**Autores:** Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, Ed Chi (Google, Inc.)
**Venue:** RecSys '19 (Thirteenth ACM Conference on Recommender Systems), Copenhague, septiembre 2019.
**DOI:** 10.1145/3298689.3346996

---

## 1. Contexto: retrieval a gran escala

Los sistemas de recomendación industriales conectan miles de millones de usuarios con catálogos (corpus) de millones a miles de millones de ítems, bajo requisitos estrictos de latencia. La práctica dominante, descrita por el paper, es tratar la recomendación como un problema de **retrieval-and-ranking en dos fases**: un modelo de *retrieval* (también llamado *candidate generation* o *nomination*) escalable selecciona primero una pequeña fracción de ítems relevantes desde el corpus completo, y luego un modelo de *ranking* más pesado los reordena según uno o varios objetivos (clics, ratings, engagement). Este paper se concentra exclusivamente en la fase de **retrieval**.

Dado un *triplet* `{user, context, item}`, la receta estándar para construir un modelo de retrieval escalable consiste en: (1) aprender representaciones (embeddings) para la consulta `{user, context}` y para el `{item}` por separado, y (2) usar una función de scoring simple — típicamente el **producto punto** — entre ambas representaciones. El contexto captura variables dinámicas (hora del día, dispositivo). Separar las torres es lo que permite, en inferencia, **precalcular los embeddings de todos los ítems** y resolver el retrieval como una búsqueda de máximo producto interno (MIPS) en tiempo sublineal.

El aprendizaje de estas representaciones es difícil por dos razones que el paper enfatiza: (1) el corpus puede ser extremadamente grande; (2) los datos de feedback son **muy esparsos** para la mayoría de los ítems, lo que produce predicciones de alta varianza para el *long-tail* y agrava el problema de *cold-start*. La distribución de ítems sigue una **ley de potencias** (power-law): unos pocos ítems populares acumulan la mayoría de las interacciones.

Históricamente, inspirado por el Netflix Prize, el enfoque dominante fue la **factorización matricial (MF)** con features de contenido (SVDFeature, Factorization Machines). El límite de MF es que solo captura interacciones de segundo orden (bilineales), insuficiente para una colección heterogénea de features (IDs esparsos, features densos, texto, imagen). De ahí el giro hacia DNNs, que codifican estados de usuario complejos y features de contenido de ítems en un espacio de embeddings de baja dimensión.

### Sampled softmax e in-batch negatives

El retrieval puede formularse como **clasificación multiclase con recompensas continuas**: dado `x`, predecir cuál ítem `y` de entre `M` candidatos. La probabilidad natural es un softmax sobre los `M` ítems (Ec. 1). Pero cuando `M` es del orden de millones, calcular la función de partición (el denominador, la suma sobre todos los ítems) es inviable. La técnica clásica de **sampled softmax** muestrea un subconjunto de clases negativas desde un vocabulario fijo (típicamente con una distribución unigram o uniforme como proxy).

El problema: el sampled softmax clásico **no es aplicable cuando la etiqueta (el ítem) tiene features de contenido**, porque la torre del ítem es una red profunda con parámetros compartidos — muestrear y entrenar sobre muchos negativos arbitrarios es ineficiente, ya que cada negativo requiere un *forward pass* completo de la torre. La solución que adopta el paper es usar **in-batch negatives**: para cada par `(x_i, y_i)` del minibatch, los demás ítems `y_j` del mismo batch sirven como negativos para `x_i`. Esto reutiliza los embeddings ya calculados de la torre de ítems — no hay forward passes extra. El softmax resultante se calcula solo sobre el batch (Ec. 3, **batch softmax**).

## 2. Contribución central: corrección del sesgo de muestreo

El batch softmax tiene un **sesgo de muestreo (sampling bias)** severo. Como los ítems in-batch se muestrean según la distribución de tráfico — power-law —, los **ítems populares aparecen como negativos con altísima probabilidad** y, por tanto, son **sobre-penalizados**: el modelo aprende a empujarlos hacia abajo más de lo que correspondería bajo el softmax completo. Esto distorsiona el aprendizaje, especialmente con distribuciones muy sesgadas, y degrada la calidad del retrieval.

La contribución del paper es **corregir este sesgo mediante la corrección log-Q**, importada del sampled softmax adaptativo de Bengio & Sénécal. Cada logit se corrige restando el logaritmo de la probabilidad de muestreo del ítem:

```
s^c(x_i, y_j) = s(x_i, y_j) − log(p_j)
```

donde `p_j` es la probabilidad de que el ítem `j` aparezca en un batch aleatorio. Intuitivamente, esto **descuenta la ventaja artificial de los ítems frecuentes** como negativos: si un ítem aparece mucho, su logit se reduce proporcionalmente, recuperando el comportamiento del softmax completo. Esta corrección es un estimador insesgado del softmax full bajo muestreo por importancia.

El reto práctico — y la verdadera novedad técnica — es que en datos **streaming** no existe un vocabulario fijo de ítems ni una distribución conocida: el catálogo de YouTube cambia constantemente (uploads de videos frescos) y la distribución de popularidad se desplaza con el tiempo. Por eso `p_j` debe **estimarse online**, de forma distribuida, sin vocabulario fijo, y adaptándose a cambios de distribución. La segunda contribución es precisamente un **algoritmo de estimación de frecuencia en streaming**.

## 3. Método

### 3.1 Two-tower DNN

El modelo aprende dos funciones de embedding parametrizadas:

```
u : X × R^d → R^k   (torre de query: user + context)
v : Y × R^d → R^k   (torre de ítem/candidato)
```

ambas redes neuronales profundas que comparten el parámetro `θ`. El score es el producto interno de los dos embeddings:

```
s(x, y) = ⟨u(x, θ), v(y, θ)⟩
```

El two-tower DNN se generaliza desde la red de clasificación multiclase (un MLP): la torre derecha de un MLP clásico es una sola capa de embeddings de ítem; aquí se reemplaza por una **torre profunda completa**, lo que permite modelar etiquetas con estructura o features de contenido. La diferencia clave frente a Neural Collaborative Filtering (NCF, He et al. 2017): NCF **concatena** los embeddings de usuario e ítem y los pasa por una red conjunta — lo que impide precalcular ítems e inferir en tiempo sublineal — y usa pérdida point-wise. El two-tower mantiene las torres **separadas hasta el producto punto final**, habilitando MIPS, y usa una pérdida softmax multiclase con modelado explícito de frecuencia.

### 3.2 Batch softmax y pérdida

El dataset es `T = {(x_i, y_i, r_i)}`, donde `r_i ∈ R` es una **recompensa** asociada a cada par. En clasificación pura `r_i = 1`; en recomendación `r_i` captura grados de engagement (p. ej., tiempo de visualización: `r_i = 0` para un clic con poco watch time, `r_i = 1` si se vio el video completo). La pérdida es la **log-verosimilitud ponderada por recompensa** (Ec. 2/4):

```
L_B(θ) = − (1/B) Σ_i  r_i · log( P^c_B(y_i | x_i; θ) )
```

donde `P^c_B` es el softmax in-batch **corregido** con `s^c`. El entrenamiento es SGD: `θ ← θ − γ·∇L_B(θ)` (Ec. 5). Crucialmente, `L_B` **no requiere un conjunto fijo de queries ni candidatos**, por lo que se aplica directamente a datos streaming cuya distribución cambia.

### 3.3 Estimación de frecuencia en streaming (hashing + streaming)

El núcleo técnico. En vez de estimar la probabilidad `p` directamente, el algoritmo estima `δ`, el **número promedio de pasos (global steps) entre dos apariciones consecutivas** de un ítem. Relación: si un ítem se muestrea cada 50 pasos, `p = 0.02 = 1/δ`. Usar el *global step* (número de batches consumidos, sincronizado entre workers vía parameter servers) tiene dos ventajas: (1) sincroniza implícitamente la estimación entre múltiples workers distribuidos; (2) permite estimar `δ` con una simple **media móvil**, adaptativa al cambio de distribución.

Como no es práctico mantener un vocabulario fijo, se usan **arreglos de hash**. El **Algoritmo 2** mantiene dos arreglos `A` y `B` de tamaño `H` y una función de hash `h` con rango `[H]`:
- `A[h(y)]`: último paso en que se muestreó `y`.
- `B[h(y)]`: estimación actual de `δ` para `y`.

Cuando `y` aparece en el paso `t`:

```
B[h(y)] ← (1 − α)·B[h(y)] + α·(t − A[h(y)])     (Ec. 6)
A[h(y)] ← t
```

En inferencia: `p̂ = 1 / B[h(y)]`. La actualización es exactamente **SGD con learning rate fijo α para aprender la media** de la variable aleatoria `Δ` (gap entre apariciones).

**Análisis teórico (Proposición 4.1).** Para `δ_i = (1−α)·δ_{i−1} + α·Δ_i`:
- **Sesgo:** `E(δ_t) − δ = (1−α)^t·δ_0 − (1−α)^{t−1}·δ`, que → 0 cuando `t → ∞`. Una inicialización ideal `δ_0 = δ/(1−α)` da estimación **insesgada en cada paso**.
- **Varianza:** acotada por `(1−α)^{2t}·(δ_0−δ)^2 + α·E[(Δ_1−δ)^2]`. El learning rate `α` actúa en dos direcciones: un `α` alto hace decaer más rápido el término de error de inicialización (más adaptativo a cambios de distribución), pero un `α` bajo reduce el término de varianza residual (que no decae con el tiempo). Trade-off adaptabilidad vs. varianza.

**Actualizaciones distribuidas.** Los arreglos `A`, `B` y el global step viven en los parameter servers; cada worker hace fetch/update asíncrono junto con el SGD de la red. La estimación funciona en entrenamiento distribuido asíncrono.

**Multiple hashing (Algoritmo 3).** Inspirado en el **count-min sketch**, para mitigar la **sobre-estimación de frecuencia por colisiones de hash**. Con colisiones, un bucket `B` puede representar la unión de varios ítems, lo que produce un gap aparente menor (sobre-estima la frecuencia). Se usan `m` funciones de hash independientes con arreglos propios y, en inferencia, se toma el **máximo** de las `m` estimaciones de `δ`: `p̂ = 1 / max_i{B_i[h(y)]}`. Tomar el máximo de los gaps corrige la sub-estimación del gap (= sobre-estimación de frecuencia) inducida por colisiones.

### 3.4 Normalización y temperatura

Dos detalles empíricos que mejoran la calidad:
- **L2 normalization** de ambos embeddings: `u ← u/‖u‖₂`, `v ← v/‖v‖₂`. Mejora la entrenabilidad y la estabilidad. Tras normalizar, el producto punto se vuelve similitud coseno.
- **Temperatura τ**: `s(x, y) = ⟨u(x), v(y)⟩ / τ`. Afila las predicciones. Es un hiperparámetro que se ajusta para maximizar recall/precision; el paper muestra que **su efecto es notable y debe tunearse con cuidado** cuando se aplica normalización.

## 4. Sistema de retrieval para YouTube

El paper aterriza el framework en un producto real: recomendaciones de YouTube condicionadas a un **seed video** (el video que el usuario está viendo). El sistema tiene dos etapas (nomination/retrieval + ranking); aquí se construye un **nominador** adicional.

- **Arquitectura (Figura 2):** torre de query (features del seed video + historial de watch del usuario como *bag-of-words* promediado) y torre de candidato (features del video candidato). DNNs de tres capas `[1024, 512, 128]` para ambas torres, ReLU, L2 normalization, in-batch softmax arriba.
- **Label de entrenamiento:** clics como positivos, con recompensa `r_i` que refleja engagement (0 a 1 según watch time).
- **Features de video:** categóricos (Video Id, Channel Id) y densos (views, likes). Embeddings **compartidos** entre seed, candidato e historial para el mismo tipo de ID. Hash buckets para entidades out-of-vocabulary (clave para capturar videos frescos).
- **Sequential training:** los datos llegan organizados por días; el trainer los consume **secuencialmente del más antiguo al más reciente**, y al alcanzar el día actual espera nuevos datos. Esto modela el stream y permite que la estimación de frecuencia (Ec. 6) se adapte a la distribución cambiante.
- **Indexing y serving:** pipeline de tres etapas (generación de candidatos, inferencia de embeddings con la torre derecha, indexado). El índice se construye con técnicas de **tree + quantization para MIPS aproximado** (maximum-inner-product-search). El SavedModel de serving une la torre de query con el modelo de índice.

## 5. Experimentos

### 5.1 Simulación de la estimación de frecuencia

Con `M=1000` ítems, distribución power-law `q_i ∝ i²`, conmutada en el paso 10000 a `q_i ∝ (M−1−i)²` para probar adaptabilidad. Métrica: distancia L1 reescalada entre `p̂` y `p` real.
- **Efecto de α** (Fig. 4): las tres curvas convergen a un nivel de error (de colisiones + varianza). `α` alto = más adaptativo al cambio de distribución, pero mayor varianza final — confirma la Proposición 4.1.
- **Efecto de multiple hashing** (Fig. 5): con `m = 1, 2, 4` y mismo número total de buckets, **más funciones de hash reducen el error** incluso a igualdad de parámetros.

### 5.2 Wikipedia Page Retrieval

Predicción de enlaces intra-sitio: dado una página fuente, recuperar las páginas destino. Grafo inglés: **5.3M páginas, 430M enlaces, 510K n-grams de título, 403.4K categorías**. Two-tower con dos capas ReLU `[512, 128]`, embeddings compartidos. Batch size 1024, Adagrad lr 0.01, 10M steps. Estimación: `m=1, H=40M, α=0.01`.

Baselines: **plain-sfx** (batch softmax sin corrección), **correct-sfx** (con corrección), y **mse-gramian** (MSE con regularización Gramian, Krichene et al. 2019). Métrica: **Recall@K** contra el corpus completo de 5.3M páginas.

| Método | R@10 | R@50 | R@100 | R@300 |
|---|---|---|---|---|
| mse-gramian | 0.0432 | 0.1326 | 0.2027 | 0.3530 |
| plain-sfx τ=0.07 | 0.0643 | 0.2423 | 0.3746 | 0.5991 |
| **correct-sfx τ=0.07** | **0.1065** | 0.3079 | 0.4664 | 0.7234 |
| **correct-sfx τ=0.05** | 0.0987 | **0.3202** | **0.4835** | **0.7413** |

`correct-sfx` supera a `plain-sfx` por **un amplio margen** en cada temperatura (p. ej. R@10 casi se duplica: 0.0643 → 0.1065). El softmax batch supera a mse-gramian. El efecto de `τ` es notable: τ=0.14 degrada el rendimiento, lo que confirma la necesidad de tunearla con normalización.

### 5.3 YouTube (offline + live)

Datos: miles de millones de clics diarios. DNN `[1024, 512, 128]`, Adagrad lr 0.2, batch 8192, `H=50M, m=1, α=0.01`. Índice de ~10M videos reconstruido cada pocas horas (cubre >90% de los ejemplos de entrenamiento). Sequential training con catch-up de 15 días; resultados promediados sobre 7 días para neutralizar el patrón semanal.

**Offline** (Recall@K, `r_i = 1` para todos los clics):

| Método | R@5 | R@10 | R@30 | R@50 |
|---|---|---|---|---|
| mse-gramian | 0.0554 | 0.0768 | 0.1149 | 0.1338 |
| plain-sfx τ=0.05 | 0.2069 | 0.2728 | 0.3964 | 0.4586 |
| **correct-sfx τ=0.05** | **0.2150** | **0.2960** | **0.4537** | **0.5322** |

`correct-sfx` gana consistentemente; el batch softmax aplasta a mse-gramian.

**Live (A/B test):** grupo control = sistema de producción; tratamiento = producción + candidatos del nuevo retrieval neuronal. Recompensa entrenada para reflejar engagement real.

| Método | Mejora en engagement |
|---|---|
| plain-sfx τ=0.05 | +0.20% |
| **correct-sfx τ=0.05** | **+0.37%** |

La corrección de sesgo **casi duplica la mejora de engagement** en producción, demostrando que el modelado explícito de frecuencia importa en el régimen de corpus realmente grande.

## 6. Limitaciones

- **No publica arquitecturas ni datos crudos completos de YouTube** (datos propietarios); las cifras live son mejoras relativas pequeñas (+0.37%), aunque significativas a escala YouTube.
- **Sesgo del logger sin tratar a fondo:** in-batch negatives heredan la distribución del tráfico; la corrección log-Q corrige la *frecuencia de muestreo* pero no el sesgo de selección del sistema de producción que generó el log (feedback loop). El paper no aborda *false negatives* (un ítem in-batch que de hecho le gustaría al usuario tratado como negativo).
- **Colisiones de hash:** mitigadas con count-min sketch, pero no eliminadas; el error residual de simulación proviene en parte de colisiones.
- **τ y α requieren tuning cuidadoso**, sin receta cerrada; la sensibilidad a τ es alta.
- **Métrica de recompensa offline simplificada** (`r_i = 1`) porque definir una métrica offline para recompensa continua "no es obvio" — brecha entre evaluación offline y objetivo real.
- El **MIPS aproximado** (quantization) introduce error de recuperación que el paper deliberadamente no analiza ("glosamos los detalles").

## 7. Impacto

Este paper se volvió la **referencia canónica del two-tower / dual-encoder para candidate generation** a escala industrial. Estandarizó tres prácticas hoy ubicuas: (1) **in-batch negatives + corrección log-Q** como receta de entrenamiento de retrievers; (2) **L2-normalization + temperatura** sobre el producto punto; (3) **serving vía MIPS** sobre embeddings de ítems precalculados. Su linaje conecta hacia atrás con DSSM (Huang et al. 2013) y el modelo de candidate generation de YouTube (Covington et al. 2016), y hacia adelante con DPR (retrieval denso para QA), los embeddings de TensorFlow Recommenders (TFRS implementa directamente este modelo y su pérdida) y prácticamente todo retrieval semántico moderno basado en embeddings. La idea de estimar la distribución de muestreo en streaming para corregir el sesgo del softmax batch sigue siendo el estado del arte para entrenar retrievers sobre catálogos dinámicos.

## 8. Conexión con la Clase 25

La Clase 25 es un *case study* de un recsys **multimodal** cuya arquitectura es, en esencia, exactamente este two-tower. En el case study, **una torre representa el pin** combinando una CNN (imagen) y un BERT (texto) que se fusionan en una capa FC para producir un embedding del ítem — es decir, la **torre de candidato/ítem `v(y)`** del Algoritmo de Yi et al., solo que con features de contenido multimodales en vez de IDs+densos de YouTube. La inferencia del case study compara la representación del pin contra la del usuario por **mínima distancia**, lo que es matemáticamente equivalente al **máximo producto punto** de este paper: con embeddings L2-normalizados, minimizar distancia euclidiana ⟺ maximizar similitud coseno ⟺ maximizar `⟨u(x), v(y)⟩`. Por tanto:

- El **scoring por mínima distancia ≈ dot product** del case study es precisamente `s(x,y) = ⟨u(x), v(y)⟩` formalizado aquí.
- El **retrieval del pin más cercano al usuario** se implementa con el mismo **MIPS aproximado** que describe el sistema de YouTube.
- El **entrenamiento contrastivo con in-batch negatives** y los problemas de sesgo por popularidad que enfrentaría el case study son exactamente los que este paper resuelve con la corrección log-Q y la estimación de frecuencia en streaming.

Este paper aporta entonces la **formalización rigurosa** de la arquitectura que la Clase 25 presenta de forma aplicada: por qué separar torres habilita inferencia escalable, por qué los in-batch negatives sesgan el aprendizaje hacia el long-tail, y cómo corregirlo. Es el puente teórico entre el case study multimodal y la familia de retrieval neuronal (ver también DSSM 2013 como ancestro del dual-encoder).
