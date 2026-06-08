# Wide & Deep Learning for Recommender Systems — Análisis interno

**Autores:** Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, Hemal Shah (Google Inc.)
**Venue:** DLRS@RecSys 2016 (1st Workshop on Deep Learning for Recommender Systems)
**arXiv:** 1606.07792 (24 jun 2016)
**slug:** wide-and-deep-cheng-2016

---

## 1. Contexto

El paper nace de un problema operativo concreto y de enorme escala: el sistema de recomendación de **Google Play**, una tienda de aplicaciones móviles con más de mil millones de usuarios activos y más de un millón de apps. El equipo plantea el sistema de recomendación como un caso particular de *search ranking*: la *query* es un conjunto de información de usuario y de contexto, y la salida es una lista ordenada de ítems (apps). Dada una query, la tarea es recuperar ítems relevantes de la base de datos y luego rankearlos según un objetivo de negocio —en este caso, la probabilidad de adquisición/instalación de la app.

El eje conceptual del paper es la tensión entre dos objetivos que tradicionalmente se atacaban con familias de modelos distintas:

- **Memorización (memorization):** aprender la co-ocurrencia frecuente de ítems o features y explotar la correlación directamente disponible en los datos históricos. Las recomendaciones basadas en memorización tienden a ser más *tópicas* y directamente relevantes a ítems sobre los que el usuario ya actuó. En la práctica, la memorización se logra muy bien con **modelos lineales generalizados** (regresión logística) entrenados sobre features dispersas binarizadas (one-hot), y enriquecidos con **transformaciones cross-product**. El ejemplo canónico del paper: la feature binaria `user_installed_app=netflix` vale 1 si el usuario instaló Netflix; el cross-product `AND(user_installed_app=netflix, impression_app=pandora)` vale 1 si el usuario instaló Netflix y luego se le mostró Pandora. Estas reglas son **efectivas e interpretables**, pero requieren *feature engineering* manual y, crucialmente, **no generalizan a pares query-ítem que no aparecieron en el entrenamiento**.

- **Generalización (generalization):** se basa en la *transitividad* de la correlación y explora combinaciones de features nuevas o raras. Mejora la **diversidad** de las recomendaciones. Los modelos basados en *embeddings* —factorization machines o redes neuronales profundas— generalizan a pares query-ítem no vistos aprendiendo un **vector denso de baja dimensión** para cada feature de query e ítem, con mucho menos esfuerzo de ingeniería de features.

El problema observado por Google es que los **embeddings densos sobre-generalizan** (over-generalize) cuando la matriz query-ítem subyacente es **dispersa y de alto rango** (sparse and high-rank): usuarios con preferencias muy específicas o ítems de nicho con apelación estrecha. En estos casos *no debería* haber interacción entre la mayoría de los pares query-ítem, pero los embeddings densos producen predicciones distintas de cero para *todos* los pares, generando recomendaciones menos relevantes. Los modelos lineales con cross-product, en cambio, pueden memorizar estas "reglas de excepción" con muchísimos menos parámetros.

La motivación del paper es entonces: **¿cómo obtener lo mejor de ambos mundos en un solo modelo?**

## 2. Contribución

Las contribuciones declaradas en el paper son tres:

1. **El framework Wide & Deep:** entrenamiento *conjunto* (joint training) de una red feed-forward con embeddings (componente *deep*) y un modelo lineal con transformaciones de features (componente *wide*), para sistemas de recomendación genéricos con entradas dispersas.
2. **Implementación y evaluación en producción** del recomendador Wide & Deep en Google Play, con experimentos online (A/B testing) a escala de mil millones de usuarios.
3. **Liberación open-source** de la implementación junto con una API de alto nivel en **TensorFlow** (el tutorial Wide & Deep de tensorflow.org).

La idea es deliberadamente simple ("While the idea is simple…"), pero el aporte está en demostrar que el *joint training* mejora significativamente la tasa de adquisición de apps en producción, satisfaciendo además los requisitos de velocidad de entrenamiento y de *serving*.

## 3. Método

### 3.1 Componente Wide

El componente wide es un modelo lineal generalizado de la forma:

$$ y = \mathbf{w}^T \mathbf{x} + b $$

donde $\mathbf{x} = [x_1, x_2, \dots, x_d]$ es el vector de $d$ features, $\mathbf{w} = [w_1, \dots, w_d]$ son los parámetros y $b$ el sesgo. El conjunto de features incluye features crudas y **features transformadas**. La transformación más importante es la **cross-product transformation**:

$$ \phi_k(\mathbf{x}) = \prod_{i=1}^{d} x_i^{c_{ki}}, \quad c_{ki} \in \{0,1\} $$

donde $c_{ki}$ es una variable booleana que vale 1 si la $i$-ésima feature forma parte de la $k$-ésima transformación $\phi_k$, y 0 en caso contrario. Para features binarias, un cross-product como `AND(gender=female, language=en)` vale 1 si y solo si todas las features constituyentes valen 1. Esto **captura interacciones entre features binarias y añade no-linealidad** al modelo lineal generalizado. Es el mecanismo de memorización.

### 3.2 Componente Deep

El componente deep es una red feed-forward. Para **features categóricas**, las entradas originales son strings (por ejemplo `language=en`). Cada feature categórica dispersa y de alta dimensión se convierte primero en un **vector embedding** denso de baja dimensión, real-valued. La dimensionalidad de los embeddings está en el orden de $O(10)$ a $O(100)$. Los embeddings se inicializan aleatoriamente y sus valores se entrenan para minimizar la función de pérdida final durante el entrenamiento del modelo.

Estos vectores densos se alimentan a las capas ocultas en el forward pass. Cada capa oculta computa:

$$ a^{(l+1)} = f\big(W^{(l)} a^{(l)} + b^{(l)}\big) $$

donde $l$ es el número de capa, $f$ es la activación (típicamente ReLU), y $a^{(l)}$, $b^{(l)}$, $W^{(l)}$ son las activaciones, sesgo y pesos de la capa $l$. Es el mecanismo de generalización.

### 3.3 Joint Training (entrenamiento conjunto)

Los componentes wide y deep se combinan mediante una **suma ponderada de sus log-odds de salida**, que se alimenta a una **función de pérdida logística común** para el entrenamiento conjunto. Para regresión logística, la predicción del modelo combinado es:

$$ P(Y=1\mid\mathbf{x}) = \sigma\!\Big( \mathbf{w}_{wide}^T [\mathbf{x}, \phi(\mathbf{x})] + \mathbf{w}_{deep}^T a^{(l_f)} + b \Big) $$

donde $Y$ es la etiqueta binaria, $\sigma(\cdot)$ es la sigmoide, $\phi(\mathbf{x})$ son los cross-products de las features originales, $b$ el sesgo, $\mathbf{w}_{wide}$ los pesos del modelo wide, y $\mathbf{w}_{deep}$ los pesos aplicados sobre las activaciones finales $a^{(l_f)}$.

El paper enfatiza la **distinción entre joint training y ensemble**:

- En un **ensemble**, los modelos individuales se entrenan por separado *sin conocerse entre sí*, y sus predicciones se combinan solo en inferencia. Esto implica que cada modelo individual debe ser *más grande* (más features y transformaciones) para alcanzar precisión razonable.
- En **joint training**, todos los parámetros se optimizan simultáneamente, tomando en cuenta tanto la parte wide como la deep *y* los pesos de su suma al momento del entrenamiento. Esto permite que la parte wide solo necesite **complementar las debilidades** de la parte deep con un *pequeño* número de cross-products, en lugar de ser un modelo wide de tamaño completo.

El joint training se realiza retropropagando los gradientes desde la salida hacia ambas partes simultáneamente con optimización estocástica por mini-batches. En los experimentos usaron **dos optimizadores distintos**: **FTRL (Follow-the-regularized-leader)** con regularización $L_1$ para la parte wide, y **AdaGrad** para la parte deep.

### 3.4 Implementación del sistema

El pipeline tiene tres etapas: generación de datos, entrenamiento del modelo y serving.

- **Data Generation:** cada ejemplo corresponde a una *impresión*. La etiqueta es **app acquisition** (1 si la app impresa fue instalada, 0 si no). Se generan *vocabularios* (tablas que mapean strings categóricos a IDs enteros) para features que ocurren más de un mínimo de veces. Las **features continuas** se normalizan a $[0,1]$ mediante su función de distribución acumulada $P(X \le x)$, dividida en $n_q$ cuantiles; el valor normalizado para el $i$-ésimo cuantil es $\frac{i-1}{n_q-1}$.
- **Model Training:** la estructura usada (Figura 4) tiene el componente wide consistente en el cross-product de *user installed apps* × *impression apps*. Para la parte deep, se aprende un **embedding de 32 dimensiones** por cada feature categórica. Todos los embeddings se **concatenan junto con las features densas (continuas)**, resultando en un vector denso de **~1200 dimensiones**. Ese vector pasa por **3 capas ReLU** (1024 → 512 → 256) y finalmente la unidad logística de salida. Los modelos se entrenan sobre **más de 500 mil millones de ejemplos**. Para evitar re-entrenar desde cero cada vez que llegan datos nuevos, implementaron **warm-starting**: inicializan el modelo nuevo con los embeddings y pesos lineales del modelo previo.
- **Model Serving:** para cumplir latencias del orden de 10 ms, optimizaron con **paralelismo multithreading**, corriendo batches más pequeños en paralelo en lugar de scorear todos los candidatos en un solo batch.

## 4. Experimentos

### 4.1 App Acquisitions (A/B test online)

Experimentos *live* en framework de A/B testing durante **3 semanas**. Grupo de control: 1% de usuarios con el modelo de ranking previo (regresión logística wide-only altamente optimizada, con cross-products ricos). Grupo experimental: 1% de usuarios con el modelo Wide & Deep entrenado con el *mismo* conjunto de features. Resultados (Tabla 1):

| Modelo | AUC offline | Ganancia online de adquisición |
|---|---|---|
| Wide (control) | 0.726 | 0% |
| Deep | 0.722 | +2.9% |
| Wide & Deep | 0.728 | **+3.9%** |

El modelo Wide & Deep mejoró la tasa de adquisición de apps en la landing principal de la tienda en **+3.9% relativo al control** (estadísticamente significativo), y **+1% sobre el modelo deep-only** (también significativo).

Observación clave del paper: **el AUC offline de Wide & Deep es solo ligeramente superior (0.728 vs 0.726/0.722), pero el impacto online es mucho más significativo**. Posible explicación: en los datasets offline las impresiones y etiquetas están *fijas*, mientras que el sistema online puede generar recomendaciones exploratorias nuevas combinando generalización con memorización, y aprender de las nuevas respuestas de usuarios. Es una lección importante sobre la brecha entre métricas offline y métricas de negocio online.

### 4.2 Serving Performance

A tráfico pico, los servidores scorean **más de 10 millones de apps por segundo**. Con single-threading, scorear todos los candidatos en un solo batch tomaba 31 ms. El multithreading (dividir en batches pequeños) redujo la latencia *client-side* a **14 ms** (incluyendo overhead de serving). Tabla 2:

| Batch size | Threads | Latencia (ms) |
|---|---|---|
| 200 | 1 | 31 |
| 100 | 2 | 17 |
| 50 | 4 | 14 |

## 5. Limitaciones

- **Mejora offline marginal:** el salto de AUC (0.726 → 0.728) es minúsculo; toda la justificación descansa en el A/B online, que no es reproducible fuera de Google.
- **Feature engineering del wide sigue siendo manual:** aunque el deep reduce la ingeniería, los cross-products del wide deben elegirse a mano (en producción, `user_installed_app × impression_app`). El framework no aprende *qué* interacciones cruzar; eso es exactamente lo que DeepFM/DCN automatizarían después.
- **Solo cruza pares manuales de bajo orden** en el wide; las interacciones de orden superior dependen enteramente del MLP, que es una caja negra menos interpretable.
- **Dos optimizadores heterogéneos** (FTRL+L1 para wide, AdaGrad para deep) añaden complejidad de tuning.
- **Escala industrial específica:** 500 mil millones de ejemplos, warm-starting y serving multithreaded son requisitos de Google; la transferibilidad a entornos pequeños no se discute.
- **Solo objetivo binario** (instalación sí/no); no modela ranking multi-objetivo ni señales de engagement post-instalación.

## 6. Impacto

Wide & Deep se convirtió en un **patrón de referencia industrial** y la base conceptual de una familia de modelos de *deep recommendation / CTR prediction*:

- **DeepFM (Guo et al., 2017):** reemplaza el componente wide (que requiere cross-products manuales) por una **Factorization Machine** que aprende automáticamente las interacciones de segundo orden, compartiendo embeddings con la parte deep. Elimina el feature engineering manual del wide.
- **Deep & Cross Network — DCN (Wang et al., 2017):** introduce una *cross network* que aprende explícitamente interacciones de features de orden creciente de forma automática, sustituyendo los cross-products manuales.
- Otros descendientes: **xDeepFM**, **Deep Interest Network (DIN)**, **AutoInt**, etc.
- La **API `tf.estimator.DNNLinearCombinedClassifier`** de TensorFlow popularizó el patrón en la industria.

La contribución más duradera no es la arquitectura puntual sino el **principio de combinar memorización + generalización mediante entrenamiento conjunto**, y la articulación clara de por qué los embeddings densos sobre-generalizan en matrices dispersas de alto rango.

## 7. Conexión con la Clase 25 (RecSys multimodal)

La Clase 25 trata cómo **representar y combinar tipos de datos heterogéneos**. Wide & Deep es el ejemplo canónico de varios principios centrales de la clase:

1. **Embeddings para categóricos dispersos:** cada feature categórica de alta cardinalidad (string como `language=en`, `user_installed_app=netflix`) se mapea a un vector denso entrenable de baja dimensión (32 en el caso de Google Play). Es exactamente la técnica de representar variables categóricas que la clase enseña: pasar de one-hot disperso a embeddings densos aprendidos end-to-end.

2. **Tratamiento de features continuas:** se normalizan a $[0,1]$ por cuantiles de su CDF, ejemplo de cómo se preprocesan los datos continuos antes de combinarlos.

3. **Concatenación de representaciones heterogéneas:** el paso central de la Figura 4 es **concatenar todos los embeddings de categóricos + las features continuas normalizadas** en un único vector denso (~1200 dim) que alimenta el MLP. Esta concatenación es justamente el mecanismo de "combinar representaciones de tipos de datos distintos" que discute la clase.

4. **Combinación de paradigmas de modelado:** la suma ponderada de log-odds wide + deep es una forma temprana de *fusión* (late fusion a nivel de logit) de dos sub-modelos que procesan la información de manera distinta — un antecedente conceptual de las fusiones multimodales más sofisticadas.

5. **Memorización vs generalización** como trade-off de representación: enlaza con la discusión de cuándo conviene una representación dispersa exacta (cross-products) versus una densa generalizable (embeddings).

Para el curso, este paper es el puente entre los modelos lineales clásicos de recomendación y las arquitecturas neuronales modernas (DeepFM, DCN), y un ejemplo limpio de ingeniería de representación de datos tabulares heterogéneos a escala industrial.
