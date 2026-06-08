---
title: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
weight: 261
math: true
---

{{< paper-card
    title="DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
    authors="Guo, Tang, Ye, Li, He"
    year="2017"
    venue="IJCAI 2017"
    pdf="/papers/deepfm-guo-2017.pdf"
    arxiv="1703.04247" >}}
DeepFM predice la **tasa de clics (CTR)** combinando una **Factorization Machine** (que modela interacciones de features de **orden 1 y 2**) con una **red profunda** (que modela interacciones de **orden alto**), donde **ambos componentes comparten el mismo embedding de features**. A diferencia de Wide & Deep, se entrena **end-to-end sin ningún feature engineering** más allá de las features crudas, y supera consistentemente a LR, FM, FNN, PNN y Wide & Deep en AUC y LogLoss sobre Criteo y datos comerciales.
{{< /paper-card >}}

---

## Contexto

Predecir el CTR —la probabilidad de que un usuario haga clic sobre un ítem— es el motor de los sistemas de recomendación y de la publicidad en línea: los ítems se rankean por CTR estimado (o por `CTR × bid` cuando importa el ingreso). La clave está en **modelar las interacciones de features** detrás del comportamiento de clic. Los autores lo ilustran con datos reales de un mercado de apps: a la hora de comer la gente descarga apps de delivery (interacción de orden 2 entre `categoría` y `hora`), y los adolescentes varones prefieren juegos de disparos/RPG (interacción de orden 3 entre `categoría`, `género` y `edad`).

El problema es que solo algunas interacciones son obvias y diseñables por expertos; la mayoría están escondidas en los datos (como la clásica regla "pañales y cerveza"). El estado del arte previo tenía un sesgo claro: los modelos lineales (FTRL) **no aprenden interacciones** salvo que se les inyecten cruces a mano; las **Factorization Machines** capturan bien el orden 2 incluso con datos dispersos, pero en la práctica se quedan ahí; y los modelos profundos como **FNN** (red inicializada por un FM pre-entrenado) y **PNN** (con capa de producto) capturan poco el **orden bajo**. El antecedente directo, **[Wide & Deep](/papers/wide-and-deep-cheng-2016)** de Google, combina orden bajo y alto, pero su lado "wide" todavía **depende de feature engineering experto** y usa una entrada distinta del lado "deep". DeepFM demuestra que se puede aprender interacciones de **todos los órdenes, end-to-end y sin ingeniería manual**.

## Ideas principales

El registro de entrada `χ` tiene `m` campos (categóricos como género/ubicación, continuos como edad), convertidos a un vector `x` **altamente disperso y de alta dimensión** (el campo de user ID puede tener mil millones de dimensiones). DeepFM tiene dos componentes que **comparten la misma entrada y el mismo embedding**: para cada feature `i` hay un escalar `w_i` (importancia de orden 1) y un vector latente `V_i ∈ R^k` que alimenta **a la vez** al FM (orden 2) y a la red profunda (orden alto).

### Componente FM, componente deep, embeddings compartidos y predicción

La predicción combinada es una **fusión tardía** de ambos componentes pasada por una sigmoide:

$$\hat{y} = \text{sigmoid}(y_{FM} + y_{DNN})$$

**Componente FM.** Es una Factorization Machine: suma un término de orden 1 y los productos internos de los vectores latentes para el orden 2:

$$y_{FM} = \langle w, x \rangle + \sum_{j_1=1}^{d} \sum_{j_2=j_1+1}^{d} \langle V_i, V_j \rangle \, x_{j_1} \cdot x_{j_2}$$

Su virtud es que captura el orden 2 incluso con datos dispersos: el peso de la interacción `(i,j)` se mide vía `⟨V_i, V_j⟩`, así que `V_i` se entrena cada vez que `i` aparece con cualquier feature —aunque el par `(i,j)` nunca haya co-ocurrido.

**Componente deep.** Una red feed-forward que aprende el orden alto. Como la entrada de CTR es dispersa, mixta y agrupada en campos, una **capa de embedding** comprime cada campo a un vector denso de tamaño uniforme `k` (campos de largos distintos producen embeddings del mismo tamaño). La salida de embedding `a^{(0)} = [e_1, ..., e_m]` se propaga por:

$$a^{(l+1)} = \sigma(W^{(l)} a^{(l)} + b^{(l)})$$

**Embeddings compartidos.** Los vectores latentes `V` del FM **son** los pesos de la capa de embedding de la red. A diferencia de FNN —donde `V` solo inicializa la red tras un pre-entrenamiento FM—, aquí el FM es parte de la arquitectura y se entrena **conjuntamente**, sin pre-entrenamiento. Compartir el embedding hace que la representación se moldee (vía backpropagation) tanto por las interacciones de orden bajo como las de orden alto, modelándola con mayor precisión. La tabla del paper resume que DeepFM es el **único** modelo que cumple las cuatro propiedades a la vez: sin pre-entrenamiento, orden alto, orden bajo y sin feature engineering.

## Resultados experimentales

Se evalúa sobre **Criteo** (45M de registros, 13 features continuas + 26 categóricas, split 90/10) y un dataset **comercial** (Company, ~mil millones de registros del App Store), con métricas **AUC** y **LogLoss**, contra LR, FM, FNN, PNN (IPNN/OPNN/PNN*) y Wide & Deep (variantes LR & DNN y FM & DNN).

| Modelo | Company AUC | Company LogLoss | Criteo AUC | Criteo LogLoss |
|---|---|---|---|---|
| LR | 0.8640 | 0.02648 | 0.7686 | 0.47762 |
| FM | 0.8678 | 0.02633 | 0.7892 | 0.46077 |
| FNN | 0.8683 | 0.02629 | 0.7963 | 0.45738 |
| PNN* | 0.8672 | 0.02636 | 0.7987 | 0.45214 |
| LR & DNN | 0.8673 | 0.02634 | 0.7981 | 0.46772 |
| **DeepFM** | **0.8715** | **0.02618** | **0.8007** | **0.45083** |

DeepFM obtiene el mejor AUC y LogLoss en ambos datasets. Frente a LR (sin interacciones) gana 0.86% y 4.18% de AUC en Company y Criteo; frente al segundo mejor modelo, más de 0.37% y 0.25% de AUC; y frente a los modelos que usan embeddings **separados** (LR & DNN, FM & DNN), más de 0.48% y 0.33% de AUC, lo que evidencia el valor de **compartir el embedding**. En eficiencia (tiempo relativo a LR, en CPU y GPU) DeepFM está entre los más rápidos, sin el overhead del pre-entrenamiento de FNN ni el costo de los productos internos de PNN. Los autores recuerdan que una mejora pequeña de AUC offline se amplifica online: citando a Wide & Deep, 0.275% de AUC offline rindió 3.9% de CTR online.

## Limitaciones reconocibles

- El FM modela explícitamente solo hasta **orden 2**; el orden alto queda delegado a la DNN, que lo aprende de forma **implícita y no controlada**. Los autores proponen como trabajo futuro reforzarlo (p. ej. con capas de pooling).
- El dataset comercial es **propietario y no reproducible**, y las mejoras de AUC son fracciones de punto; el argumento de impacto descansa en la amplificación online citada de otro trabajo.
- La dimensión de embedding `k` es **uniforme** para todos los campos, lo que puede no ser óptimo ante cardinalidades muy distintas.
- El entrenamiento es en una sola GPU; escalar a clusters queda como dirección futura.

## Por qué importa hoy

DeepFM se volvió una arquitectura de referencia para CTR prediction y un patrón estándar en pipelines industriales. Su receta —**FM + DNN con embeddings compartidos, end-to-end**— inspiró toda una familia posterior (xDeepFM, DCN/Deep & Cross, AutoInt). El mensaje de fondo —combinar interacciones explícitas de orden bajo con interacciones implícitas de orden alto **sin feature engineering** supera a memorizar cruces a mano— quedó consolidado como buena práctica.

## Conexión con la Clase 25

La [Clase 25](/clases/clase-25) trata sobre **combinar representaciones de features heterogéneas** en recomendación multimodal, y DeepFM es un caso temprano y nítido de esa idea. La capa de embedding que proyecta campos categóricos dispersos a un **espacio denso común de tamaño `k`** es el mecanismo básico que permite mezclar features de naturaleza y dimensionalidad distintas —el mismo desafío que enfrentan los [sistemas de recomendación](/fundamentos/recommender-systems) multimodales al unir texto, imagen y señales tabulares (ver también [representación de datos](/fundamentos/representacion-datos)). La lección de **compartir la representación** entre un componente explícito (FM) y uno implícito (deep), de modo que ambos la moldeen vía backpropagation, anticipa el principio multimodal de aprender representaciones conjuntas en vez de procesar cada vista por separado. Y la fusión final `ŷ = sigmoid(y_FM + y_DNN)` es la plantilla de **fusión tardía** que se generaliza a combinar vistas de distintas modalidades.

## Notas y enlaces

- **PDF:** [/papers/deepfm-guo-2017.pdf](/papers/deepfm-guo-2017.pdf)
- **arXiv:** [1703.04247](https://arxiv.org/abs/1703.04247)
- **Antecedente directo:** [Wide & Deep (Cheng et al., 2016)](/papers/wide-and-deep-cheng-2016)
- **Fundamentos relacionados:** [recommender-systems](/fundamentos/recommender-systems), [representación de datos](/fundamentos/representacion-datos)
- **Clase:** [Clase 25 — Recsys multimodal](/clases/clase-25)
