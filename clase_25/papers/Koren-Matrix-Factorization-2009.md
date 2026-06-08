# Matrix Factorization Techniques for Recommender Systems

**Autores:** Yehuda Koren (Yahoo Research), Robert Bell y Chris Volinsky (AT&T Labs—Research)
**Publicación:** IEEE Computer, vol. 42, n.º 8, agosto 2009, pp. 42-49
**arXiv:** ninguno

---

## 1. Contexto: el Netflix Prize y dos familias de sistemas de recomendación

El artículo se escribe en el punto más alto de la competencia **Netflix Prize**, anunciada en 2006 por la empresa de alquiler de DVD Netflix con el objetivo de mejorar en un 10% el RMSE de su propio sistema de recomendación (Cinematch). El premio de 1 millón de dólares, junto con la liberación de un conjunto de datos varios órdenes de magnitud más grande que cualquier dataset público anterior (más de 100 millones de calificaciones, ~500.000 clientes anónimos, más de 17.000 películas, escala de 1 a 5 estrellas), desató una actividad febril en el campo: según el sitio de la competencia, más de 48.000 equipos de 182 países descargaron los datos. Los autores de este artículo forman el equipo **BellKor**, que tomó el primer lugar en 2007 (8,43% mejor que Netflix), ganó el Progress Prize 2007, y luego —aliado con Big Chaos— el Progress Prize 2008 con 9,46%. Al momento de escribir el artículo seguían en primer lugar acercándose al 10%.

El artículo abre con una taxonomía clara de los sistemas de recomendación, organizándolos en dos grandes estrategias:

**Content filtering (filtrado basado en contenido).** Crea un perfil explícito para cada usuario y cada producto. Un perfil de película puede incluir género, actores, popularidad en taquilla; un perfil de usuario, datos demográficos o respuestas a un cuestionario. El sistema asocia usuarios con productos cuyos perfiles coinciden. El ejemplo canónico que citan es el **Music Genome Project** de Pandora.com, donde analistas musicales entrenados puntúan cada canción según cientos de "genes" (características musicales distintas). La debilidad: requiere recopilar información externa que muchas veces no está disponible ni es fácil de obtener.

**Collaborative filtering (filtrado colaborativo).** Se apoya **únicamente** en el comportamiento pasado del usuario —transacciones previas, calificaciones de productos— sin construir perfiles explícitos. El término fue acuñado por los desarrolladores de **Tapestry**, el primer sistema de recomendación (Goldberg et al., 1992). Su gran atractivo es que es *domain free* (independiente del dominio) y captura aspectos de los datos difíciles de perfilar con contenido. Es generalmente más preciso que el contenido, pero sufre el **cold start problem**: no puede tratar productos ni usuarios nuevos sin historial. En ese aspecto puntual, el filtrado por contenido es superior.

Dentro del filtrado colaborativo, los autores distinguen dos subáreas:

- **Métodos de vecindad (neighborhood methods).** Calculan relaciones entre ítems o entre usuarios. El enfoque orientado a ítems estima la preferencia de un usuario por un ítem a partir de las calificaciones que el mismo usuario dio a ítems "vecinos" (los que tienden a recibir calificaciones similares). El ejemplo es *Saving Private Ryan*, cuyos vecinos serían películas bélicas, de Spielberg o de Tom Hanks. El enfoque orientado a usuarios (ilustrado en la Figura 1 con "Joe") identifica usuarios afines que se complementan entre sí.
- **Modelos de factores latentes (latent factor models).** Explican las calificaciones caracterizando ítems y usuarios mediante, digamos, 20 a 100 factores inferidos de los patrones de calificación. Estos factores son una alternativa computarizada a los "genes" musicales creados por humanos: para películas pueden medir dimensiones obvias (comedia vs. drama, cantidad de acción, orientación infantil), dimensiones menos definidas (profundidad de personajes, rareza) o dimensiones completamente ininterpretables. La Figura 2 ilustra el caso de dos dimensiones (orientación femenina-masculina y serio-escapista), donde la calificación predicha relativa a la media equivale al producto punto de las posiciones de usuario y película.

La tesis central del artículo, anunciada desde el subtítulo: **los modelos de factorización de matrices son superiores a las técnicas clásicas de vecinos más cercanos** para producir recomendaciones, y además permiten incorporar información adicional como feedback implícito, efectos temporales y niveles de confianza.

## 2. Contribución central: factores latentes vía factorización de matrices

La idea unificadora del artículo es que algunas de las realizaciones más exitosas de los modelos de factores latentes se basan en **factorización de matrices**. En su forma básica, la factorización caracteriza tanto ítems como usuarios mediante vectores de factores inferidos directamente de los patrones de calificación. Una alta correspondencia entre los factores de un ítem y los de un usuario conduce a una recomendación. Estos métodos se popularizaron por combinar **buena escalabilidad con precisión predictiva**, ofreciendo además gran flexibilidad para modelar situaciones reales.

Los autores subrayan que los datos de los sistemas de recomendación se organizan típicamente en una matriz usuario × ítem. El dato más conveniente es el **explicit feedback** (feedback explícito): entrada directa del usuario sobre su interés, como las estrellas de Netflix o los pulgares arriba/abajo de TiVo. A este lo llaman *ratings* (calificaciones). El feedback explícito conforma usualmente una **matriz dispersa** (sparse), pues cualquier usuario individual ha calificado solo un pequeño porcentaje de ítems posibles. Cuando no hay feedback explícito, se puede inferir preferencia desde **implicit feedback** (feedback implícito): historial de compras, navegación, patrones de búsqueda o incluso movimientos del mouse. El feedback implícito denota presencia/ausencia de un evento y suele representarse por una **matriz densamente poblada**.

## 3. Método

### 3.1 Modelo básico de factorización

Cada ítem $i$ se asocia a un vector $q_i \in \mathbb{R}^f$ y cada usuario $u$ a un vector $p_u \in \mathbb{R}^f$, en un espacio latente conjunto de dimensión $f$. Los elementos de $q_i$ miden cuánto posee el ítem cada factor (positivo o negativo); los de $p_u$ miden el interés del usuario en ítems altos en cada factor. El producto punto $q_i^T p_u$ captura la interacción usuario-ítem y aproxima la calificación:

$$\hat{r}_{ui} = q_i^T p_u \tag{1}$$

El desafío mayor es **computar el mapeo** de cada ítem y usuario a sus vectores de factores. Una vez aprendido, predecir cualquier calificación es trivial mediante la Ecuación 1.

### 3.2 Relación con SVD y el problema de los valores faltantes

El modelo está íntimamente relacionado con la **descomposición en valores singulares (SVD)**, técnica establecida para identificar factores semánticos latentes en recuperación de información. Pero aplicar SVD al filtrado colaborativo exige factorizar la matriz usuario-ítem, lo que plantea dificultades por la **alta proporción de valores faltantes** debida a la dispersión: la SVD convencional **está indefinida** cuando el conocimiento de la matriz es incompleto. Además, atender descuidadamente solo a las pocas entradas conocidas es muy propenso al **sobreajuste**.

Sistemas anteriores recurrían a **imputación** (rellenar las calificaciones faltantes para densificar la matriz, Sarwar et al. 2000), pero la imputación es muy costosa (aumenta drásticamente el volumen de datos) y una imputación inexacta puede distorsionar considerablemente los datos. Por eso, trabajos más recientes (Funk; Koren; Paterek; Takács et al.) propusieron **modelar directamente solo las calificaciones observadas**, evitando el sobreajuste mediante **regularización**.

### 3.3 Función de costo regularizada

El sistema aprende minimizando el error cuadrático regularizado sobre el conjunto $\kappa$ de pares $(u,i)$ con $r_{ui}$ conocida (el conjunto de entrenamiento):

$$\min_{q^*, p^*} \sum_{(u,i)\in\kappa} (r_{ui} - q_i^T p_u)^2 + \lambda(\|q_i\|^2 + \|p_u\|^2) \tag{2}$$

La constante $\lambda$ controla el grado de regularización (penaliza la magnitud de los parámetros aprendidos para generalizar a calificaciones futuras) y se determina usualmente por **validación cruzada**. Los autores notan que la "Probabilistic Matrix Factorization" de Salakhutdinov y Mnih ofrece un fundamento probabilístico para esta regularización.

### 3.4 Algoritmos de aprendizaje: SGD vs. ALS

La Ecuación 2 se minimiza por dos vías:

**Descenso de gradiente estocástico (SGD).** Simon Funk popularizó esta optimización (en su famoso post de blog de 2006). El algoritmo recorre todas las calificaciones del conjunto de entrenamiento; para cada caso computa el error de predicción $e_{ui} \overset{\text{def}}{=} r_{ui} - q_i^T p_u$ y modifica los parámetros en la dirección opuesta al gradiente, con magnitud proporcional a la tasa $\gamma$:

$$q_i \leftarrow q_i + \gamma \cdot (e_{ui}\cdot p_u - \lambda \cdot q_i)$$
$$p_u \leftarrow p_u + \gamma \cdot (e_{ui}\cdot q_i - \lambda \cdot p_u)$$

Combina facilidad de implementación con tiempo de ejecución relativamente rápido.

**Mínimos cuadrados alternados (ALS).** Como $q_i$ y $p_u$ son ambos incógnitas, la Ecuación 2 **no es convexa**. Pero si se fija una de las incógnitas, el problema se vuelve cuadrático y se resuelve óptimamente. ALS alterna: fija todos los $p_u$ y recomputa los $q_i$ resolviendo un problema de mínimos cuadrados, y viceversa; cada paso decrece la Ecuación 2 hasta converger. ALS es preferible en dos casos: (1) cuando se puede **paralelizar masivamente** (cada $q_i$ se computa independiente de los demás factores de ítem, e igual con los $p_u$); (2) en sistemas centrados en **datos implícitos**, donde el conjunto de entrenamiento no es disperso y recorrer cada caso individual —como hace SGD— sería impráctico.

### 3.5 Sesgos (biases)

Gran parte de la variación observada en las calificaciones se debe a efectos asociados a usuarios o ítems **independientes de cualquier interacción**: algunos usuarios califican sistemáticamente más alto, algunos ítems reciben calificaciones más altas. Una aproximación de primer orden del sesgo es:

$$b_{ui} = \mu + b_i + b_u \tag{3}$$

donde $\mu$ es la media global, y $b_u$, $b_i$ las desviaciones observadas de usuario e ítem. El ejemplo del artículo: si $\mu = 3.7$ estrellas, *Titanic* tiende a 0,5 sobre la media, y Joe (usuario crítico) tiende a 0,3 bajo la media, la estimación de primer orden para Joe-Titanic es $3.7 + 0.5 - 0.3 = 3.9$ estrellas. Los sesgos extienden la predicción:

$$\hat{r}_{ui} = \mu + b_i + b_u + q_i^T p_u \tag{4}$$

descomponiendo la calificación en cuatro componentes (media global, sesgo de ítem, sesgo de usuario, interacción usuario-ítem) para que cada uno explique solo lo que le corresponde. La función de costo se vuelve:

$$\min_{p^*,q^*,b^*} \sum_{(u,i)\in\kappa} (r_{ui} - \mu - b_u - b_i - p_u^T q_i)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2) \tag{5}$$

Como los sesgos capturan mucho de la señal observada, su modelado preciso es vital.

### 3.6 Fuentes de entrada adicionales (feedback implícito y atributos)

Para mitigar el cold start, se incorpora feedback implícito. Con feedback booleano, $N(u)$ es el conjunto de ítems para los que el usuario $u$ expresó una preferencia implícita; cada ítem $i$ recibe un **segundo** vector de factores $x_i \in \mathbb{R}^f$, y el usuario se perfila por $\sum_{i\in N(u)} x_i$, normalizado convenientemente por $|N(u)|^{-0.5}$. Atributos conocidos del usuario (demografía: género, edad, código postal, ingreso) en el conjunto $A(u)$ aportan otro vector $y_a \in \mathbb{R}^f$ por atributo. El modelo integra todas las señales con representación de usuario enriquecida:

$$\hat{r}_{ui} = \mu + b_i + b_u + q_i^T \left[ p_u + |N(u)|^{-0.5}\sum_{i\in N(u)} x_i + \sum_{a\in A(u)} y_a \right] \tag{6}$$

### 3.7 Dinámica temporal

Los modelos anteriores son estáticos, pero la percepción y popularidad de productos, y las inclinaciones de los clientes, cambian con el tiempo. La factorización se presta bien a modelar efectos temporales descomponiendo en términos que varían en el tiempo: sesgos de ítem $b_i(t)$ (una película entra y sale de moda), sesgos de usuario $b_u(t)$ (un usuario re-escala su criterio; deriva natural, calificación relativa a otras recientes, cambio de la identidad del calificador dentro de un hogar) y preferencias de usuario $p_u(t)$ (un fan de thrillers psicológicos se vuelve fan de dramas criminales un año después). Notablemente, los factores de ítem $q_i$ se mantienen **estáticos**, porque a diferencia de los humanos, los ítems no cambian de naturaleza. La regla de predicción dinámica:

$$\hat{r}_{ui}(t) = \mu + b_i(t) + b_u(t) + q_i^T p_u(t) \tag{7}$$

### 3.8 Niveles de confianza variables

No todas las calificaciones observadas merecen el mismo peso. Publicidad masiva puede inflar votos; usuarios adversarios pueden manipular calificaciones; y en sistemas de feedback implícito, el nivel de preferencia exacto es difícil de cuantificar (se trabaja con una representación binaria cruda: "probablemente le gusta" / "probablemente no le interesa"). Conviene adjuntar **scores de confianza** $c_{ui}$, que pueden provenir de valores numéricos de frecuencia (cuánto tiempo vio un programa, con qué frecuencia compró un ítem). La función de costo se pondera:

$$\min_{p^*,q^*,b^*} \sum_{(u,i)\in\kappa} c_{ui}(r_{ui} - \mu - b_u - b_i - p_u^T q_i)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2) \tag{8}$$

Para una aplicación real con estos esquemas, los autores remiten a "Collaborative Filtering for Implicit Feedback Datasets" (Hu, Koren, Volinsky, ICDM 08).

## 4. Experimentos sobre los datos del Netflix Prize

Las entradas ganadoras del equipo consistieron en **más de 100 conjuntos de predictores**, la mayoría modelos de factorización con variantes de los métodos descritos. Discusiones con otros equipos punteros y publicaciones en el foro confirmaron que estos eran los métodos más populares y exitosos para predecir calificaciones.

**Interpretabilidad de los factores (Figura 3).** Al factorizar la matriz usuario-película de Netflix, los dos primeros factores revelan estructura semántica clara: el primer factor (eje x) opone comedias chabacanas y terror para público masculino/adolescente (*Half Baked*, *Freddy vs. Jason*) contra dramas/comedias serias con protagonistas femeninas fuertes (*Sophie's Choice*, *Moonstruck*); el segundo (eje y) separa cine independiente, aclamado y peculiar (*Punch-Drunk Love*, *I Heart Huckabees*) de filmes convencionales formulaicos (*Armageddon*, *Runaway Bride*). Hay intersecciones interesantes: *Kill Bill* y *Natural Born Killers* (indie + chabacano, temas violentos artísticos); *The Sound of Music* (drama femenino + masivo); y *The Wizard of Oz* justo en el centro, gustando a todos. Casos como *Annie Hall* y *Citizen Kane* aparecen contiguos (clásicos célebres de directores famosos) y solo el **tercer** factor termina separándolos.

**RMSE y complejidad del modelo (Figura 4).** Los autores compararon implementaciones evolutivas: factorización pura, + sesgos, + feedback implícito, y dos variantes con componentes temporales. Hallazgos:

- La precisión **mejora al aumentar la dimensionalidad** $f$ (número de factores) en cada familia de modelo.
- Los **modelos más refinados** (con más conjuntos distintos de parámetros) son más precisos.
- Los **componentes temporales son particularmente importantes**: hay efectos temporales significativos en los datos.
- Cifras de referencia citadas en la Figura 4: el sistema propio de Netflix logra **RMSE = 0,9514** sobre el mismo dataset, y el grand prize exigía **RMSE = 0,8563** (la mejora del 10%). Las curvas de los modelos de factorización van desde ~0,91 (plain, 40-180 factores) hasta ~0,876 (temporal v.2, hasta 1.500 factores y decenas de miles de millones de parámetros).

## 5. Limitaciones reconocibles

- **Cold start.** Aunque el feedback implícito y los atributos demográficos lo mitigan, el filtrado colaborativo sigue sin manejar bien productos y usuarios completamente nuevos; los propios autores admiten que en ese punto el content filtering es superior.
- **No convexidad.** La Ecuación 2 no es convexa; SGD no garantiza óptimo global (de ahí el valor de ALS por bloques, que sí resuelve subproblemas exactamente).
- **Modelo lineal de interacción.** La interacción se modela como producto punto (bilineal). Solo captura relaciones lineales en el espacio latente; no hay no linealidades ni interacciones de orden superior entre factores.
- **Sensibilidad a hiperparámetros.** $\lambda$ y $\gamma$ requieren validación cruzada cuidadosa; el sobreajuste es un riesgo constante dada la dispersión.
- **Escala como única métrica reportada.** El artículo optimiza RMSE (predicción de calificación), no métricas de ranking top-N ni diversidad/novedad, que son lo que importa en producción.
- **Esfuerzo de ingeniería del ensemble.** El resultado ganador requirió más de 100 predictores combinados, algo poco práctico de desplegar (lección que Netflix mismo reconocería después).

## 6. Impacto y legado

Este artículo es el **texto canónico** que cristalizó la factorización de matrices como metodología dominante en filtrado colaborativo. Su síntesis —escrita por los protagonistas del Netflix Prize— convirtió un conjunto de trucos de competencia (el "Funk SVD" de Simon Funk, las variantes de Paterek, Takács, Salakhutdinov-Mnih) en un marco unificado y didáctico. Establece de forma duradera varias ideas:

1. **Embeddings aprendidos por descenso de gradiente.** $q_i$ y $p_u$ son, en esencia, *embeddings* de ítems y usuarios. La idea de representar entidades discretas como vectores densos de baja dimensión, aprendidos minimizando un error de reconstrucción, es exactamente la que subyace a word2vec, a las capas de embedding de las redes neuronales modernas, y a los two-tower / dual-encoder de los recomendadores neuronales actuales.
2. **Predecir como producto punto en espacio latente.** El scoring $q_i^T p_u$ es el ancestro directo del retrieval por similitud de embeddings (ANN, FAISS) que usan hoy YouTube, Spotify y los buscadores semánticos.
3. **Modelado aditivo de sesgos.** Separar media global + sesgo de ítem + sesgo de usuario + interacción anticipa el diseño de modelos como Wide & Deep y los términos de bias en arquitecturas neuronales.
4. **Feedback implícito y confianza ponderada.** La formulación de $c_{ui}$ y $N(u)$ es la base del trabajo posterior sobre datos implícitos (BPR, modelos de logged feedback, recsys de producción que casi nunca tienen ratings explícitos).
5. **Dinámica temporal.** Anticipa los recomendadores secuenciales y session-based (GRU4Rec, SASRec, transformers para recomendación).

La factorización de matrices entrega un modelo **compacto, eficiente en memoria y relativamente fácil de aprender**, que integra naturalmente múltiples formas de feedback, dinámica temporal y niveles de confianza. Sigue siendo un baseline obligatorio en cualquier benchmark de recomendación moderno.

## 7. Conexión con la Clase 25 (recsys multimodal)

La Clase 25 es un Case Study de un **sistema de recomendación multimodal**, donde las señales no son solo calificaciones sino también imágenes, texto, audio y metadatos procesados por redes neuronales. Este artículo es el **punto de partida histórico** del que parte toda esa línea:

- **De factores latentes a embeddings neuronales.** Los $q_i$ y $p_u$ aprendidos aquí por SGD son la versión "shallow" de los embeddings que hoy producen encoders profundos. Donde Koren et al. inferían factores puramente del patrón de calificaciones, un recomendador multimodal **inicializa o reemplaza** esos factores con representaciones extraídas de la imagen del producto (CNN/ViT), su descripción (BERT/transformers) o su audio. El producto punto $q_i^T p_u$ sigue ahí, pero ahora $q_i$ proviene de modalidades múltiples.
- **El cold start como motivación del multimodal.** La gran limitación que el artículo solo mitiga parcialmente —ítems y usuarios nuevos sin historial— es **exactamente** lo que la información multimodal resuelve: un producto nuevo no tiene calificaciones, pero sí tiene foto y texto, de los cuales se puede derivar su embedding inicial. El multimodal es la respuesta directa al cold start que Koren et al. dejan abierto.
- **Continuidad del marco de entrenamiento.** Las ideas de regularización, función de costo sobre observados, ponderación por confianza y feedback implícito siguen vigentes en el entrenamiento de recomendadores neuronales; lo que cambia es la arquitectura que genera los vectores, no la filosofía de optimización.

En síntesis, la Clase 25 puede leerse como "¿qué pasa cuando reemplazamos los factores latentes inferidos solo de ratings por embeddings multimodales profundos, manteniendo la columna vertebral de scoring por similitud que este artículo estableció en 2009?".
