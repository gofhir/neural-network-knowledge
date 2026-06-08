# VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback

**Autores:** Ruining He, Julian McAuley (UC San Diego)
**Venue:** AAAI 2016
**arXiv:** 1510.01784 (6 oct 2015)
**Análisis interno para Clase 25 — Recomendación multimodal con features visuales**

---

## 1. Contexto

Los sistemas de recomendación (Recommender Systems, RS) modernos entregan sugerencias personalizadas aprendiendo desde feedback histórico y "destilando" (teasing apart) las dimensiones latentes que codifican tanto las propiedades de los ítems como las preferencias de los usuarios hacia ellos. Estos sistemas son centrales para ayudar a las personas a descubrir ítems de interés en corpus enormes: desde películas y música, hasta artículos de investigación, noticias, libros, tags e incluso otros usuarios.

El feedback que alimenta estos modelos viene en dos formas. El **feedback explícito** son las calificaciones con estrellas (star ratings), donde el usuario declara su preferencia de forma directa. El **feedback implícito** es mucho más abundante pero más ambiguo: historiales de compra, marcadores (bookmarks), logs de navegación, patrones de búsqueda, actividad del mouse. El paper se ubica de lleno en el escenario de feedback implícito, que es el caso realista en plataformas de comercio electrónico donde rara vez se piden calificaciones explícitas.

Para modelar feedback en datasets grandes y reales, las aproximaciones de **Matrix Factorization (MF)** se han propuesto para descubrir las dimensiones latentes más relevantes, tanto en settings explícitos como implícitos (Koren, Hu, Pan, Rendle). A pesar de su gran éxito, sufren de un problema crítico: **cold start** (arranque en frío), provocado por la dispersión (sparsity) de los datasets reales. Un ítem con muy pocas observaciones asociadas (o ninguna) no tiene suficiente señal para que MF estime sus factores latentes de forma confiable.

El gancho conceptual del paper es directo y muy intuitivo: **uno no compraría una polera (t-shirt) en Amazon sin ver la imagen del producto**. La apariencia visual es una señal de primer orden en la decisión de compra de productos como ropa, calzado o accesorios; sin embargo, es justamente la fuente de información que los RS tradicionales ignoran. Mientras que abundantes trabajos previos habían incorporado señales auxiliares (side information) — texto, ubicación física, estación del año, temperatura, taxonomías, demografía — nadie había incorporado la apariencia visual de los ítems directamente al predictor de preferencias y, de paso, descubierto las "dimensiones visuales" relevantes para la opinión de las personas.

El momento histórico es clave: en 2015-2016 las **Deep CNN** ya habían demostrado un poder descriptivo enorme en detección de objetos, anotación de estilo fotográfico y categorización de calidad estética. Y, de manera decisiva para este trabajo, los estudios de **transfer learning** (Donahue/DeCAF 2014, Razavian "CNN features off-the-shelf" 2014) habían mostrado que una CNN entrenada en ImageNet podía extraer features genéricas que superaban al estado del arte en datasets y tareas completamente distintas. Esto significaba que las features visuales estaban **disponibles "out-of-the-box"** sin ingeniería manual costosa. VBPR explota exactamente esa observación.

---

## 2. Contribución

El paper propone **VBPR (Visual Bayesian Personalized Ranking)**, un modelo de factorización escalable que incorpora señales visuales al predictor de opiniones. Las contribuciones principales, declaradas explícitamente, son tres:

1. **Una aproximación de Matrix Factorization que incorpora señales visuales** al predictor de opiniones, escalando a datasets grandes (millones de acciones de usuario).
2. **Derivación y análisis de un procedimiento de entrenamiento basado en BPR** (Bayesian Personalized Ranking, Rendle et al. 2009) adecuado para descubrir factores visuales.
3. **Experimentos sobre datasets reales grandes y novedosos** (incluyendo el nuevo dataset de Tradesy.com), además de visualizaciones del espacio de "rating visual" descubierto.

La idea unificadora es **particionar las dimensiones del rating en dos grupos**: factores latentes (no visuales, como en MF clásico) y **factores visuales** derivados de las features CNN. La señal visual sirve de "señal auxiliar" precisamente en las situaciones donde MF falla — los ítems fríos — porque un ítem nuevo, aunque tenga cero interacciones, sí tiene una imagen y, por lo tanto, sí tiene features visuales que el modelo puede usar de inmediato.

Lo que distingue a VBPR de trabajos previos de visión aplicada a moda (Simo-Serra fashionability, Jagadeesh street fashion, Kalantidis "getting the look", McAuley IBR "styles and substitutes") es que esos trabajos hacen **recuperación visual** (visual retrieval): encuentran ítems estilísticamente similares a una imagen de consulta, pero **no se personalizan** según el feedback histórico del usuario, ni consideran factores no visuales. VBPR combina ambos mundos: la señal visual **y** el feedback histórico del usuario, que es lo esencial para resolver el ranking personalizado de una clase (one-class personalized ranking).

---

## 3. Método

### 3.1 Formulación del problema

Sean $\mathcal{U}$ el conjunto de usuarios e $\mathcal{I}$ el conjunto de ítems. Cada usuario $u$ tiene asociado un conjunto de ítems positivos $\mathcal{I}_u^+$ sobre los que expresó feedback positivo (implícito). Además, **hay una sola imagen disponible por cada ítem** $i \in \mathcal{I}$. El objetivo es generar, para cada usuario $u$, un **ranking personalizado** de los ítems que aún no ha consumido, es decir $\mathcal{I} \setminus \mathcal{I}_u^+$.

### 3.2 Predictor de preferencias

El punto de partida es el predictor de MF clásico (Koren & Bell 2011):

$$\hat{x}_{u,i} = \alpha + \beta_u + \beta_i + \gamma_u^T \gamma_i$$

donde $\alpha$ es el offset global, $\beta_u$ y $\beta_i$ son los sesgos (bias) de usuario e ítem, y $\gamma_u, \gamma_i \in \mathbb{R}^K$ son los **factores latentes** ($K$-dimensionales) de usuario e ítem. El producto interno $\gamma_u^T \gamma_i$ codifica la "compatibilidad" entre las preferencias latentes del usuario y las propiedades del producto.

El problema, ya mencionado, es la existencia de ítems "fríos" sobre los que hay muy pocas observaciones para estimar sus dimensiones latentes. La propuesta es **particionar las dimensiones** en factores visuales y factores latentes (no visuales):

$$\hat{x}_{u,i} = \alpha + \beta_u + \beta_i + \gamma_u^T \gamma_i + \theta_u^T \theta_i$$

donde $\theta_u, \theta_i \in \mathbb{R}^D$ son los **factores visuales** $D$-dimensionales recién introducidos. El producto $\theta_u^T \theta_i$ modela la interacción visual: hasta qué punto el usuario $u$ se siente atraído por cada una de las $D$ dimensiones visuales. (Nótese que $K$ sigue denotando las dimensiones latentes.)

### 3.3 El embedding visual: el corazón del método

Una forma ingenua de implementar lo anterior sería usar directamente las features de la Deep CNN $f_i$ como $\theta_i$. Pero esto presenta un problema serio: **la alta dimensionalidad** — las features usadas tienen **4096 dimensiones**. Aprender un factor de usuario de 4096-d por usuario sería intratable y propenso al overfitting. La reducción de dimensionalidad tipo PCA sería una opción, pero con la desventaja de perder gran parte del poder expresivo de las features originales para explicar el comportamiento.

En su lugar, VBPR propone **aprender un kernel de embedding** que transforma linealmente las features de alta dimensión a un espacio de "rating visual" de mucha menor dimensión (digamos 20 o así):

$$\theta_i = E f_i$$

Aquí $E$ es una matriz $D \times F$ que embebe el espacio de features de la Deep CNN ($F$-dimensional, $F = 4096$) en el espacio visual ($D$-dimensional). Los valores numéricos de las dimensiones proyectadas se interpretan como el grado en que un ítem exhibe una determinada faceta de rating visual. Este embedding es **eficiente**: **todos los ítems comparten la misma matriz** $E$, lo que reduce dramáticamente la cantidad de parámetros a aprender. La señal visual del usuario ($\theta_u$) sí se aprende por usuario, pero solo en el espacio reducido $D$.

Adicionalmente se introduce un término de **sesgo visual** $\beta'$, cuyo producto interno con $f_i$ modela la opinión global de los usuarios hacia la apariencia visual de un ítem (independiente del usuario). El predictor final es:

$$\hat{x}_{u,i} = \alpha + \beta_u + \beta_i + \gamma_u^T \gamma_i + \theta_u^T (E f_i) + \beta'^T f_i$$

Los dos últimos términos son las dos contribuciones visuales: el término personalizado $\theta_u^T(E f_i)$ y el sesgo visual global $\beta'^T f_i$.

### 3.4 Entrenamiento con BPR

BPR es un framework de optimización de ranking **pairwise** (por pares) que usa ascenso de gradiente estocástico (SGA). El conjunto de entrenamiento $D_S$ consiste en triples $(u, i, j)$:

$$D_S = \{(u,i,j) \mid u \in \mathcal{U} \wedge i \in \mathcal{I}_u^+ \wedge j \in \mathcal{I} \setminus \mathcal{I}_u^+\}$$

es decir, un usuario $u$, un ítem positivo $i$ y un ítem no observado $j$. El criterio de optimización BPR-OPT maximiza:

$$\sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) - \lambda_\Theta \|\Theta\|^2$$

donde $\sigma$ es la sigmoide logística, $\lambda_\Theta$ es el hiperparámetro de regularización, y $\hat{x}_{uij} = \hat{x}_{u,i} - \hat{x}_{u,j}$ es la diferencia de scores entre el ítem positivo y el negativo. La idea pairwise es **más débil pero más realista** que las point-wise: no se asume que el feedback no observado sea negativo, solo que el ítem positivo $i$ debe ser "más preferible" que el no observado $j$. (Nótese que $\alpha$ y $\beta_u$ se cancelan en la diferencia $\hat{x}_{u,i} - \hat{x}_{u,j}$ y se eliminan del conjunto de parámetros.)

La regla de actualización por triple muestreado es:

$$\Theta \leftarrow \Theta + \eta \cdot \left(\sigma(-\hat{x}_{uij}) \frac{\partial \hat{x}_{uij}}{\partial \Theta} - \lambda_\Theta \Theta\right)$$

Hay ahora **dos conjuntos de parámetros**: (a) los no visuales (actualizados como en BPR-MF) y (b) los visuales recién introducidos. Las actualizaciones visuales explícitas son:

$$\theta_u \leftarrow \theta_u + \eta \cdot (\sigma(-\hat{x}_{uij}) E(f_i - f_j) - \lambda_\Theta \theta_u)$$
$$\beta' \leftarrow \beta' + \eta \cdot (\sigma(-\hat{x}_{uij})(f_i - f_j) - \lambda_\beta \beta')$$
$$E \leftarrow E + \eta \cdot (\sigma(-\hat{x}_{uij}) \theta_u (f_i - f_j)^T - \lambda_E E)$$

Se introduce un hiperparámetro adicional $\lambda_E$ para regularizar la matriz de embedding $E$.

### 3.5 Escalabilidad

La eficiencia de BPR-MF subyacente hace que VBPR sea igualmente escalable. BPR-MF requiere $O(K)$ por triple. En VBPR: actualizar $\theta_u$ toma $O(D \times F)$, $\beta'$ toma $O(F)$, y $E$ toma $O(D \times F)$. La complejidad total por triple es $O(K + D)$ (donde $F$ está fijo en 4096), es decir, **lineal en el número de dimensiones**. Además, las features visuales de las Deep CNN son **sparse** (por las ReLU), lo que reduce significativamente el tiempo de cómputo en la práctica.

---

## 4. Experimentos

### 4.1 Datasets

Cuatro datasets reales (estadísticas tras preprocesamiento), todos donde se descarta a usuarios con $|\mathcal{I}_u^+| < 5$:

| Dataset | #usuarios | #ítems | #feedback |
|---|---|---|---|
| Amazon Women | 99.748 | 331.173 | 854.211 |
| Amazon Men | 34.212 | 100.654 | 260.352 |
| Amazon Phones | 113.900 | 192.085 | 964.477 |
| Tradesy.com | 19.823 | 166.526 | 410.186 |
| **Total** | **267.683** | **790.438** | **2.489.226** |

Los tres datasets de Amazon (Women's y Men's Clothing, más Cell Phones & Accessories) provienen de McAuley et al. 2015; se usan historiales de reseñas como feedback implícito y una imagen por ítem. **Tradesy.com** es un dataset nuevo introducido en el paper: una comunidad de venta de ropa de segunda mano, donde se combinan historiales de compra y "thumbs-up" como feedback positivo. Tradesy es inherentemente cold start por la naturaleza "one-off" (de una sola vez) de las transacciones de segunda mano — cada ítem es prácticamente único.

### 4.2 Features visuales

Para cada ítem se extraen features $f_i$ con el **modelo de referencia de Caffe** (Jia et al. 2014), que implementa la arquitectura CNN de **Krizhevsky, Sutskever & Hinton 2012 (AlexNet)**: 5 capas convolucionales seguidas de 3 capas fully-connected, pre-entrenada sobre 1,2 millones de imágenes de **ImageNet (ILSVRC2010)**. Se toma la salida de la **segunda capa fully-connected (FC7)**, obteniendo un vector de features visuales de **$F = 4096$ dimensiones**.

### 4.3 Metodología de evaluación

Split train/validación/test: para cada usuario se selecciona un ítem aleatorio para validación ($\mathcal{V}_u$) y otro para test ($\mathcal{T}_u$); el resto es entrenamiento. La métrica es **AUC** (Area Under the ROC Curve):

$$\text{AUC} = \frac{1}{|\mathcal{U}|} \sum_u \frac{1}{|E(u)|} \sum_{(i,j) \in E(u)} \delta(\hat{x}_{u,i} > \hat{x}_{u,j})$$

que mide la fracción de pares en que el ítem de test se rankea por sobre un ítem no observado. Se reporta el desempeño en test para los hiperparámetros que mejor funcionaron en validación.

### 4.4 Baselines

- **RAND** (random), **MP** (Most Popular, no personalizado).
- **MM-MF**: MF pairwise de Gantner et al. 2011, optimizado con pérdida hinge.
- **BPR-MF**: el pairwise de Rendle et al. 2009, estado del arte para ranking personalizado con feedback implícito.
- **IBR** (Image-Based Recommendation, McAuley 2015): baseline content-based que aprende un espacio visual y recupera ítems estilísticamente similares por vecino más cercano (no usa feedback, usa grafos de relaciones entre ítems).
- **WRMF** (Hu, Koren, Volinsky 2008): método point-wise, para comparación.

Para comparación justa se usa el **mismo número total de dimensiones** en todos los métodos MF. En VBPR las dimensiones visuales y no visuales se fijan en un split **50/50** por simplicidad.

### 4.5 Resultados (AUC en test, #factores = 20)

| Dataset | Setting | RAND | MP | IBR | MM-MF | BPR-MF | **VBPR** | mej. vs best | mej. vs BPR-MF |
|---|---|---|---|---|---|---|---|---|---|
| Amazon Women | All | 0,4997 | 0,5772 | 0,7163 | 0,7127 | 0,7020 | **0,7834** | 9,4% | 11,6% |
| Amazon Women | Cold | 0,5031 | 0,3159 | 0,6673 | 0,5489 | 0,5281 | **0,6813** | 2,1% | 29,0% |
| Amazon Men | All | 0,4992 | 0,5726 | 0,7185 | 0,7179 | 0,7100 | **0,7841** | 9,1% | 10,4% |
| Amazon Men | Cold | 0,4986 | 0,3214 | 0,6787 | 0,5666 | 0,5512 | **0,6898** | 1,6% | 25,1% |
| Amazon Phones | All | 0,5063 | 0,7163 | 0,7397 | 0,7956 | 0,7918 | **0,8052** | 1,2% | 1,7% |
| Amazon Phones | Cold | 0,5014 | 0,3393 | 0,6319 | 0,5570 | 0,5346 | 0,6056 | −4,2% | 13,3% |
| Tradesy.com | All | 0,5003 | 0,5085 | N/A | 0,6097 | 0,6198 | **0,7829** | 26,3% | 26,3% |
| Tradesy.com | Cold | 0,4972 | 0,3721 | N/A | 0,5172 | 0,5241 | **0,7594** | 44,9% | 44,9% |

Los ítems cold start representan ~60% del test para los datasets de Amazon y ~80% para Tradesy.com. Hallazgos principales:

1. Sobre BPR-MF, VBPR mejora en promedio **más de 12% en all items y más de 28% en cold start**. Esto demuestra el beneficio significativo de incorporar features CNN al ranking.
2. **IBR supera a BPR-MF y MM-MF en cold start** (donde MF puro no puede aprender factores significativos), pero **pierde frente a MF en warm start** porque no se entrena con feedback histórico.
3. Al combinar las fortalezas de MF y content-based, **VBPR supera a todos los baselines en la mayoría de los casos**.
4. Las mejoras son **particularmente grandes en Tradesy.com** (+44,9% en cold start), por ser inherentemente un dataset cold start.
5. Las features visuales aportan **más en ropa que en celulares** — en celulares lo visual juega un rol menor (aunque aún significativo); de hecho en Amazon Phones cold start VBPR queda −4,2% bajo el mejor baseline (IBR), el único caso negativo.
6. Los métodos basados en popularidad (MP) son particularmente inefectivos aquí, porque los ítems fríos son inherentemente "impopulares".

Finalmente, los métodos pairwise superan a los point-wise: VBPR le gana a WRMF en promedio **14,3% en all items y 20,3% en cold start**.

### 4.6 Sensibilidad, eficiencia y visualización

- **Sensibilidad** (Figura 2): MM-MF, BPR-MF y VBPR mejoran al aumentar el número de factores, mostrando la capacidad de los métodos pairwise de evitar overfitting.
- **Eficiencia** (Figura 3): VBPR tarda más en converger que MM-MF/BPR-MF, pero solo requiere **~3,5 horas** para converger en el dataset más grande (Women's Clothing) en un desktop estándar (4 cores físicos, 32GB RAM).
- **Visualización del espacio visual** (Figura 4): con t-SNE se proyecta el espacio visual de 10-D aprendido para Amazon Women. Dos observaciones: (1) aunque las features vienen de una CNN pre-entrenada en otro dataset, el embedding logra aprender una "transición visual" (loosely) a través de subcategorías, confirmando el poder expresivo de las features; (2) VBPR no solo aprende la taxonomía oculta, sino que descubre las dimensiones visuales subyacentes más relevantes y mapea ítems y usuarios al espacio descubierto.

---

## 5. Limitaciones

- **Una sola imagen por ítem.** El modelo usa exactamente una imagen por producto; productos con múltiples vistas o cuya apariencia depende del ángulo no se aprovechan plenamente.
- **Features CNN congeladas (frozen).** Las features se extraen de una AlexNet pre-entrenada en ImageNet y no se hace fine-tuning de la CNN para la tarea de recomendación; solo se aprende la matriz de embedding $E$ por encima. Esto es eficiente, pero el espacio de features está atado a lo que ImageNet (clasificación de objetos) considera relevante, que no siempre coincide con lo que importa estéticamente para una decisión de compra.
- **Lo visual no manda en todas las categorías.** En celulares las features visuales aportan poco; de hecho VBPR queda por debajo de IBR en Phones cold start. El método es más útil donde la apariencia domina la decisión (ropa, calzado).
- **Split 50/50 fijo entre dimensiones visuales y latentes.** Los propios autores indican que afinar este split podría mejorar el desempeño; se dejó fijo por simplicidad.
- **Sin dinámica temporal.** No modela la deriva de las modas en el tiempo (declarado como trabajo futuro).
- **Solo feedback implícito.** El estudio se limita a implícito; la extensión a feedback explícito queda como trabajo futuro.
- **Dependencia de AUC.** AUC valora el ordenamiento global pero es menos sensible al top de la lista (top-k), que es lo que el usuario realmente ve.

---

## 6. Impacto

VBPR fue el trabajo seminal que abrió la línea de **recomendación visual personalizada** combinando factorización + features de CNN + ranking pairwise. Su receta — extraer features CNN pre-entrenadas (transfer learning), proyectarlas con una matriz de embedding compartida a un espacio de rating visual de baja dimensión, y entrenar con BPR — se convirtió en una plantilla replicada y extendida ampliamente.

Los propios autores (He & McAuley) lo extendieron en trabajos posteriores muy citados: **Sherlock** (incorporando jerarquías de categorías a la señal visual), **Monomer**, y especialmente la línea de **dinámica visual temporal** que modela cómo evolucionan las modas (anticipada aquí como trabajo futuro). El uso de features visuales para mitigar cold start, junto con la idea de descubrir "dimensiones visuales interpretables", influyó en sistemas de recomendación de moda y e-commerce. La introducción del **dataset Tradesy.com** también aportó un benchmark inherentemente cold start a la comunidad.

Conceptualmente, VBPR es un puente entre dos comunidades — visión por computador (CNN, transfer learning) y sistemas de recomendación (MF, BPR) — y demostró que las features visuales genéricas "off-the-shelf" tienen poder predictivo real sobre el comportamiento de compra, no solo sobre clasificación de objetos.

---

## 7. Conexión con la Clase 25

La Clase 25 trata exactamente lo que hace este paper: **recomendación multimodal usando features visuales extraídas con CNN + feedback implícito**. Es el paper conceptualmente más cercano al case study de la clase.

- **Features CNN 4096-d = el dataset Pinterest de la clase.** VBPR usa la salida FC7 de AlexNet (4096-d) como representación de cada ítem. El case study de la clase trabaja con features visuales pre-extraídas con CNN de la misma dimensionalidad sobre imágenes (Pinterest). La lección es idéntica: **no se entrena la CNN; se usan features pre-computadas y se aprende un embedding por encima** — transfer learning puro. Esto explica por qué en clase las imágenes vienen como vectores y no como píxeles.
- **Ranking pairwise (BPR) sobre feedback implícito.** La clase recomienda ítems mediante ranking a partir de feedback implícito (pins, likes), no de calificaciones explícitas. VBPR formaliza por qué pairwise (preferir lo positivo sobre lo no observado) es más adecuado que tratar lo no observado como negativo duro.
- **Cold start resuelto vía la señal visual.** El argumento central de la clase — que las features visuales permiten recomendar ítems nuevos sin historial — es exactamente el resultado estrella de VBPR (+44,9% AUC en cold start en Tradesy). Un ítem sin interacciones igual tiene imagen, y por lo tanto tiene factores visuales utilizables de inmediato.
- **El embedding compartido $E$.** La idea de proyectar 4096-d a un espacio de ~20-D con una matriz compartida por todos los ítems es la pieza didáctica que la clase reutiliza: reduce parámetros, evita overfitting y produce un espacio visual interpretable.

Cross-links sugeridos: BPR de Rendle 2009 (predictor pairwise subyacente), fundamentos de redes convolucionales (de dónde salen las features), transfer learning (por qué funcionan features de ImageNet en otra tarea) y recommender systems (marco general de MF y feedback implícito).
