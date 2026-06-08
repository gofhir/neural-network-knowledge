# DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

**Autores:** Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He
**Venue:** IJCAI 2017 (arXiv:1703.04247v1, 13 de marzo de 2017)
**Afiliaciones:** Harbin Institute of Technology (Shenzhen) y Noah's Ark Research Lab, Huawei

---

## 1. Contexto: la predicción de CTR y el cuello de botella del feature engineering

La predicción de la tasa de clics (**click-through rate**, CTR) es una de las tareas centrales de cualquier sistema de recomendación o de publicidad en línea. El objetivo es estimar la probabilidad de que un usuario haga clic sobre un ítem recomendado en un contexto dado. Cuando el sistema busca maximizar el número de clics, los ítems se rankean directamente por su CTR estimado; cuando además importa el ingreso (como en publicidad), el ranking se ajusta a `CTR × bid` (el beneficio que recibe el sistema si el ítem es clicado). En ambos casos, estimar bien el CTR es lo que mueve la aguja del negocio: los autores reportan que la facturación diaria del App Store de su "Company" está en el orden de los millones de dólares, de modo que incluso un alza de pocos puntos porcentuales en CTR se traduce en millones de dólares anuales adicionales.

El paper parte de una observación empírica clave: detrás del comportamiento de clic hay **interacciones de features** sofisticadas, y modelarlas bien es lo que distingue un buen predictor de uno mediocre. Los autores ilustran esto con dos ejemplos extraídos de un mercado de apps real:

- A la hora de las comidas la gente tiende a descargar apps de delivery de comida → la interacción de **orden 2** entre `categoría de la app` y `time-stamp` es señal de CTR.
- Los adolescentes varones prefieren juegos de disparos y RPG → la interacción de **orden 3** entre `categoría de app`, `género` y `edad` es otra señal.

El problema es que solo algunas de estas interacciones son comprensibles a priori y diseñables por expertos. La mayoría están escondidas en los datos y no se pueden anticipar (el clásico ejemplo de minería de reglas de asociación "pañales y cerveza" no lo descubrió ningún experto, lo descubrieron los datos). Aun para las interacciones obvias, es inviable que un experto las enumere exhaustivamente cuando hay miles o millones de features.

El estado del arte previo se reparte entre tres familias, cada una con un sesgo:

1. **Modelos lineales generalizados** (p. ej. FTRL de McMahan et al., 2013). Simples y eficientes, pero por construcción **no aprenden interacciones**; la práctica habitual es inyectar manualmente productos cruzados de features en el vector de entrada, lo que no generaliza a interacciones de orden alto ni a las que aparecen rara vez en el entrenamiento.

2. **Factorization Machines (FM)** (Rendle, 2010). Modelan las interacciones de orden 2 como producto interno de vectores latentes asociados a cada feature. Su gran ventaja sobre los modelos lineales con cruces manuales es que pueden estimar el peso de la interacción entre features `i` y `j` aunque ese par nunca haya co-ocurrido, porque el vector latente `V_i` se entrena cada vez que `i` aparece (con cualquier otra feature). En principio FM puede modelar órdenes altos, pero en la práctica se queda en orden 2 por costo computacional.

3. **Modelos profundos** que extienden CNN/RNN (sesgados a interacciones entre features vecinas o a datos secuenciales), **FNN** (Zhang et al., 2016: una red feed-forward inicializada con un FM pre-entrenado) y **PNN** (Qu et al., 2016: introduce una capa de producto entre el embedding y la primera capa densa). El problema, ya señalado por Cheng et al. (2016), es que FNN y PNN —como los modelos profundos en general— capturan poco las interacciones de **orden bajo**, que también son esenciales para CTR.

4. **Wide & Deep** (Cheng et al., 2016, Google). El antecedente directo. Combina un modelo lineal ("wide", que memoriza co-ocurrencias) con un modelo profundo ("deep", que generaliza), logrando modelar simultáneamente órdenes bajos y altos. **Pero tiene dos entradas distintas**: el "deep part" recibe embeddings, mientras que el "wide part" sigue dependiendo de **feature engineering experto** (por ejemplo, el producto cruzado entre las apps instaladas por el usuario y las apps mostradas). Ese vector de entrada del lado wide puede ser enorme porque incluye los cruces diseñados a mano, lo que también incrementa la complejidad.

La tesis del paper es directa: los modelos existentes están **sesgados a orden bajo o alto, o dependen de feature engineering**. DeepFM demuestra que se puede aprender interacciones de **todos los órdenes** de forma **end-to-end** y **sin ingeniería manual** más allá de las features crudas.

## 2. Contribución

Las contribuciones que los autores enumeran son tres:

1. **Una nueva arquitectura, DeepFM**, que integra un FM y una DNN. Modela interacciones de orden bajo como FM y de orden alto como la DNN. A diferencia de Wide & Deep, se entrena de punta a punta sin ningún feature engineering.

2. **Eficiencia por compartición de entrada y de embeddings.** A diferencia de Wide & Deep, la parte wide (FM) y la parte deep (DNN) comparten la **misma entrada** y el **mismo vector de embedding**. En Wide & Deep el vector de entrada del lado wide puede ser gigantesco porque incluye los cruces de features diseñados a mano; eso eleva mucho la complejidad. DeepFM lo evita.

3. **Evaluación empírica** sobre datos de benchmark (Criteo) y datos comerciales (Company), mostrando mejora consistente sobre los modelos existentes.

La tabla comparativa del paper (Tabla 1) resume la posición de DeepFM frente a las alternativas según cuatro propiedades:

| Modelo | Sin pre-entrenamiento | Features orden alto | Features orden bajo | Sin feature engineering |
|---|:---:|:---:|:---:|:---:|
| FNN | ✗ | ✓ | ✗ | ✓ |
| PNN | ✓ | ✓ | ✗ | ✓ |
| Wide & Deep | ✓ | ✓ | ✓ | ✗ |
| **DeepFM** | ✓ | ✓ | ✓ | ✓ |

DeepFM es el **único** que satisface las cuatro: no necesita pre-entrenamiento, captura orden alto y bajo, y no requiere feature engineering.

## 3. Método

### 3.1 Planteo del problema

El dataset de entrenamiento son `n` instancias `(χ, y)`, donde `χ` es un registro de `m` campos (fields) que típicamente involucra un par usuario-ítem, e `y ∈ {0,1}` es el clic (1 = clic, 0 = no). Los campos pueden ser **categóricos** (género, ubicación) o **continuos** (edad). Cada campo categórico se representa como un vector one-hot, y cada campo continuo como el valor mismo o como one-hot tras discretizar. Concatenando todos los campos se obtiene `x = [x_field1, x_field2, ..., x_fieldm]`, un vector `d`-dimensional normalmente **altísimamente disperso y de muy alta dimensión** (en un app store de mil millones de usuarios, el campo de user ID ya tiene mil millones de dimensiones). La tarea es construir `ŷ = CTR_model(x)`.

### 3.2 La arquitectura DeepFM

DeepFM consta de dos componentes que **comparten la misma entrada**: el **componente FM** y el **componente deep**. Para cada feature `i` hay:

- un escalar `w_i` que pondera su importancia de **orden 1**, y
- un vector latente `V_i ∈ R^k` que mide su impacto en interacciones con otras features.

El vector `V_i` cumple **doble función**: se alimenta al componente FM para modelar interacciones de orden 2, y se alimenta al componente deep para modelar interacciones de orden alto. Todos los parámetros (`w_i`, `V_i` y los pesos de la red `W^(l)`, `b^(l)`) se entrenan **conjuntamente** para la predicción combinada:

$$\hat{y} = \text{sigmoid}(y_{FM} + y_{DNN})$$  (Ec. 1)

donde `ŷ ∈ (0,1)` es el CTR predicho, `y_FM` la salida del componente FM e `y_DNN` la del componente deep.

### 3.3 Componente FM

El componente FM es exactamente una factorization machine (Rendle, 2010). Su gran virtud frente a enfoques previos es que captura las interacciones de orden 2 **mucho mejor cuando los datos son dispersos**: en los métodos clásicos, el parámetro de la interacción `(i, j)` solo se puede entrenar si `i` y `j` co-ocurren en algún registro; en FM se mide vía el producto interno `⟨V_i, V_j⟩`, de modo que `V_i` se entrena cada vez que `i` aparece con cualquier feature. Así, interacciones que nunca o casi nunca aparecen en el entrenamiento se aprenden mejor.

La salida del FM es la suma de una unidad de adición (orden 1) y unidades de producto interno (orden 2):

$$y_{FM} = \langle w, x \rangle + \sum_{j_1=1}^{d} \sum_{j_2=j_1+1}^{d} \langle V_i, V_j \rangle \, x_{j_1} \cdot x_{j_2}$$  (Ec. 2)

con `w ∈ R^d` y `V_i ∈ R^k`. El término `⟨w, x⟩` refleja la importancia de las features de orden 1; las unidades de producto interno representan las interacciones de orden 2.

### 3.4 Componente deep

Es una red feed-forward que aprende interacciones de orden alto. La entrada de CTR es muy distinta a la de imágenes o audio (densas y continuas): es **dispersa, de alta dimensión, mixta categórica-continua y agrupada en campos**. Por eso se necesita una **capa de embedding** que comprima la entrada a un vector real denso de baja dimensión antes de pasarla a la primera capa oculta; de lo contrario la red sería inentrenable.

Dos rasgos interesantes de esa subred entrada→embedding:

1. Aunque los vectores de entrada de los distintos campos tengan **largos diferentes**, sus embeddings son del **mismo tamaño `k`**.
2. Los vectores latentes `V` del FM funcionan ahora como **pesos de la red** que aprenden a comprimir cada campo a su embedding. A diferencia de FNN (donde `V` se pre-entrena con un FM y solo sirve de inicialización), aquí el FM es **parte de la arquitectura global** y se entrena conjuntamente. Esto elimina la necesidad de pre-entrenar.

Denotando la salida de la capa de embedding como:

$$a^{(0)} = [e_1, e_2, ..., e_m]$$  (Ec. 3)

donde `e_i` es el embedding del campo `i` y `m` el número de campos, el forward de la red es:

$$a^{(l+1)} = \sigma(W^{(l)} a^{(l)} + b^{(l)})$$  (Ec. 4)

con `l` la profundidad de capa y `σ` la activación. Finalmente:

$$y_{DNN} = \sigma(W^{|H|+1} \cdot a^{H} + b^{|H|+1})$$

donde `|H|` es el número de capas ocultas.

### 3.5 La clave: embeddings compartidos

El punto central es que FM y deep **comparten el mismo embedding de features**, lo que aporta dos beneficios: (1) se aprenden interacciones de orden bajo **y** alto desde las features crudas; (2) no hace falta feature engineering experto de la entrada, como sí lo requiere Wide & Deep. La compartición influye —vía backpropagation— en la representación de features tanto por las interacciones de orden bajo como por las de orden alto, modelando esa representación con mayor precisión. Los autores señalan que una extensión natural de Wide & Deep es reemplazar la LR del lado wide por un FM (lo evalúan como "FM & DNN"); esa variante se parece a DeepFM, **pero no comparte el embedding**, y ahí está la diferencia que DeepFM explota.

### 3.6 Relación con FNN y PNN

- **FNN**: red inicializada por FM pre-entrenado. Dos limitaciones: los parámetros de embedding pueden quedar sobre-influidos por el FM, y el pre-entrenamiento añade overhead. Además FNN **solo** captura orden alto.
- **PNN**: introduce una capa de producto entre embedding y primera capa densa. Tiene tres variantes (IPNN con producto interno, OPNN con producto externo, PNN* con ambos). El producto externo resulta **menos confiable** (su cómputo aproximado pierde información). El producto interno es más confiable pero costoso, porque la salida de la capa de producto se conecta a **todas** las neuronas de la primera capa oculta. En DeepFM, la salida de la capa de producto (el FM) se conecta **solo a la neurona de salida final**. Como FNN, todos los PNN ignoran el orden bajo.

## 4. Experimentos

### 4.1 Datasets

- **Criteo**: 45 millones de registros de clic, 13 features continuas y 26 categóricas. Split aleatorio 90% train / 10% test.
- **Company (comercial)**: 7 días consecutivos de registros de clic del game center del App Store de la Company para entrenar, y el día siguiente para test. Cerca de **mil millones de registros**. Incluye features de app (identificación, categoría), de usuario (apps descargadas) y de contexto (hora de operación).

### 4.2 Métricas y modelos

Métricas: **AUC** (área bajo ROC) y **LogLoss** (entropía cruzada). Se comparan 9 modelos: LR, FM, FNN, PNN (tres variantes), Wide & Deep y DeepFM. Para Wide & Deep se evalúan dos variantes: la original con LR en el lado wide ("LR & DNN") y una con FM en el lado wide ("FM & DNN"), esta última para eliminar el esfuerzo de feature engineering. Hiperparámetros (Criteo, siguiendo a Qu et al. 2016): dropout 0.5, estructura 400-400-400, optimizador Adam, activación tanh para IPNN y relu para el resto; DeepFM usa el mismo setting; LR usa FTRL, FM usa Adam con dimensión latente 10.

### 4.3 Efectividad (Tabla 2)

| Modelo | Company AUC | Company LogLoss | Criteo AUC | Criteo LogLoss |
|---|---|---|---|---|
| LR | 0.8640 | 0.02648 | 0.7686 | 0.47762 |
| FM | 0.8678 | 0.02633 | 0.7892 | 0.46077 |
| FNN | 0.8683 | 0.02629 | 0.7963 | 0.45738 |
| IPNN | 0.8664 | 0.02637 | 0.7972 | 0.45323 |
| OPNN | 0.8658 | 0.02641 | 0.7982 | 0.45256 |
| PNN* | 0.8672 | 0.02636 | 0.7987 | 0.45214 |
| LR & DNN | 0.8673 | 0.02634 | 0.7981 | 0.46772 |
| FM & DNN | 0.8661 | 0.02640 | 0.7850 | 0.45382 |
| **DeepFM** | **0.8715** | **0.02618** | **0.8007** | **0.45083** |

Tres conclusiones que extraen los autores:

1. **Aprender interacciones mejora el CTR.** LR (el único que no considera interacciones) es el peor. DeepFM supera a LR en 0.86% y 4.18% de AUC (1.15% y 5.60% de LogLoss) en Company y Criteo respectivamente.
2. **Aprender orden alto y bajo simultáneamente y bien mejora el CTR.** DeepFM supera a los que solo aprenden orden bajo (FM) o solo alto (FNN, IPNN, OPNN, PNN*). Frente al segundo mejor modelo, DeepFM gana más de 0.37% y 0.25% de AUC (0.42% y 0.29% de LogLoss).
3. **Compartir el mismo embedding ayuda.** DeepFM supera a los modelos que aprenden orden alto y bajo con embeddings **separados** (LR & DNN, FM & DNN), por más de 0.48% y 0.33% de AUC (0.61% y 0.66% de LogLoss).

Los autores enfatizan que una mejora pequeña en AUC offline suele traducirse en una mejora grande en CTR online: citando a Cheng et al. (2016), Wide & Deep mejoró AUC en 0.275% offline pero el CTR online subió 3.9%.

### 4.4 Eficiencia

Se mide como `tiempo de entrenamiento del modelo profundo / tiempo de LR`, en CPU y GPU. Observaciones: el pre-entrenamiento de FNN lo vuelve menos eficiente; IPNN y PNN* son costosos por las operaciones de producto interno; **DeepFM es de los más eficientes en ambos tests**.

### 4.5 Estudio de hiperparámetros (en Company)

- **Activación**: relu es mejor que tanh para todos los modelos profundos salvo IPNN (posiblemente porque relu induce esparsidad).
- **Dropout**: el óptimo cae entre 0.6 y 0.9; algo de aleatoriedad refuerza la robustez.
- **Neuronas por capa**: más neuronas no siempre ayuda (OPNN empeora de 400 a 800); 200 o 400 es buena elección. DeepFM se mantiene estable entre 400 y 800.
- **Número de capas ocultas**: mejora al inicio, pero degrada si se sigue aumentando (overfitting).
- **Forma de la red**: la forma "constant" (mismo número de neuronas por capa) es empíricamente mejor que increasing, decreasing y diamond.

## 5. Limitaciones reconocibles

- El componente FM modela explícitamente solo hasta **orden 2**; las interacciones de orden alto quedan delegadas a la DNN, que las captura de forma **implícita y no controlada** (no hay garantía de que aprenda las interacciones útiles). Los propios autores plantean como trabajo futuro introducir, por ejemplo, capas de **pooling** para reforzar el aprendizaje de las interacciones de orden alto más útiles.
- Los resultados sobre el dataset comercial **no son reproducibles** públicamente (es propietario), y las mejoras de AUC son fracciones de punto porcentual; el argumento de impacto descansa en la afirmación (citada de Wide & Deep) de que se amplifican online.
- La dimensión del embedding `k` es compartida y uniforme entre todos los campos, lo que puede no ser óptimo para campos de cardinalidad muy distinta.
- El trabajo se entrena en una sola GPU; escalar a clusters para problemas de gran escala queda explícitamente como dirección futura.

## 6. Impacto

DeepFM se volvió una de las arquitecturas de referencia para CTR prediction y un componente estándar en pipelines industriales de recomendación y advertising. Su receta —**FM + DNN con embeddings compartidos, entrenados end-to-end**— inspiró toda una familia de modelos posteriores (xDeepFM con CIN para interacciones explícitas de orden alto, DCN/Deep & Cross Network con cross layers, AutoInt con atención, etc.). El mensaje central —que combinar interacciones explícitas de orden bajo con interacciones implícitas de orden alto, **sin feature engineering**, supera a memorizar cruces a mano— se consolidó como buena práctica de la industria.

## 7. Conexión con la Clase 25 (recsys multimodal)

La Clase 25 trata sobre **combinar representaciones de features heterogéneas** en recomendación. DeepFM es un caso paradigmático y temprano de esa idea por tres motivos:

1. **Embeddings de categóricos como puente.** El problema de fondo de DeepFM —entradas dispersas, de alta dimensión, mixtas categórica-continua y agrupadas en campos— es exactamente el que enfrentan los sistemas multimodales: cada modalidad/campo tiene su propia naturaleza y dimensionalidad, y la capa de embedding las proyecta a un **espacio denso común de dimensión `k`**. Que campos de largos distintos terminen en embeddings del mismo tamaño es el mecanismo básico que permite "mezclar" representaciones heterogéneas.

2. **Compartir representaciones entre componentes.** La lección de que **compartir el embedding** entre el componente FM (orden bajo, explícito) y el componente deep (orden alto, implícito) mejora la representación —porque ambos caminos la moldean vía backpropagation— anticipa el principio multimodal de aprender representaciones conjuntas en lugar de procesar cada vista por separado y concatenar al final.

3. **Fusión de señales de distinto orden/naturaleza.** La combinación final `ŷ = sigmoid(y_FM + y_DNN)` es una **fusión tardía** de dos vistas complementarias de los mismos datos. En recsys multimodal, esa misma plantilla se generaliza a fusionar vistas de distintas modalidades (texto, imagen, señales tabulares), donde DeepFM aporta la intuición de cómo combinar una rama que captura estructura explícita con otra que captura estructura implícita.
