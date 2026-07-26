---
title: "TSN: Temporal Segment Networks (2016)"
weight: 406
math: true
---

{{< paper-card
    title="Temporal Segment Networks: Towards Good Practices for Deep Action Recognition"
    authors="Limin Wang et al. (CUHK, ETH)"
    year="2016"
    venue="ECCV 2016 / arXiv:1608.00859"
    pdf="/papers/tsn-wang-2016.pdf" >}}
**TSN (Temporal Segment Network)** es un marco a nivel de video para reconocimiento de acciones que resuelve dos problemas de las [two-stream ConvNets](/papers/two-stream-simonyan-2014): su incapacidad de capturar **estructura temporal de largo alcance** y el sobreajuste al entrenar redes profundas con datasets pequeños. La idea central es un **muestreo temporal esparcido**: en vez de procesar frames densos o clips cortos, TSN divide cada video en $K$ segmentos de igual duración, muestrea **un solo snippet corto por segmento** distribuido a lo largo de toda la línea de tiempo, lo procesa con una red de parámetros compartidos y **agrega las predicciones mediante una función de consenso** para producir una única salida a nivel de video. Como los frames consecutivos son enormemente redundantes, unas pocas muestras bien distribuidas cubren el video completo a una fracción del cómputo. TSN alcanzó el estado del arte en **HMDB51 (69.4%)** y **UCF101 (94.2%)**. Es la referencia teórica directa del laboratorio de la [Clase 36](/clases/clase-36), donde se muestrean 8 frames distribuidos para clasificar la acción.
{{< /paper-card >}}

---

## Contexto: frames densos y rango temporal corto

En 2016 las ConvNets profundas dominaban la clasificación de imágenes, pero en **reconocimiento de acciones en video** su ventaja sobre los descriptores hechos a mano (por ejemplo, *improved dense trajectories*) todavía no era clara. La arquitectura dominante era la [two-stream ConvNet](/papers/two-stream-simonyan-2014) de Simonyan y Zisserman: una rama *espacial* que ve una imagen RGB (apariencia) y una rama *temporal* que ve una pila de campos de **flujo óptico** de frames consecutivos (movimiento de corto plazo). El problema estructural es que **ambas ramas solo ven un instante o una ventana muy breve**, mientras que las acciones complejas —una zambullida, un salto de altura— constan de múltiples etapas que se despliegan durante un tiempo relativamente largo.

Los intentos previos de modelar rango temporal largo (LSTM, *long-term temporal convolutions*, LRCN) dependían de **muestreo temporal denso** con un intervalo predefinido, lo que traía dos problemas. Primero, **costo excesivo**: procesar secuencias largas densamente es caro, obligando a trabajar con longitudes fijas acotadas (típicamente 64–120 frames) que limitan la cobertura. Segundo, **redundancia**: los frames consecutivos son casi idénticos, así que el muestreo denso aporta poco a cambio de mucho cómputo. A esto se suma que **UCF101 y HMDB51 son datasets pequeños**, con alto riesgo de sobreajuste al entrenar redes profundas desde cero.

## Método: muestreo esparcido y consenso a nivel de video

TSN combina una *spatial* ConvNet y una *temporal* ConvNet, pero **no opera sobre frames individuales ni sobre stacks cortos**, sino sobre una secuencia de snippets muestreados esparcidamente de todo el video. Dado un video $V$, se lo divide en $K$ segmentos $\{S_1, \dots, S_K\}$ de igual duración y se modela como:

$$\text{TSN}(T_1, \dots, T_K) = H\big(G(F(T_1; W), F(T_2; W), \dots, F(T_K; W))\big)$$

donde cada snippet $T_k$ se muestrea **aleatoriamente** de su segmento $S_k$ (una forma implícita de aumento de datos: distintas épocas ven distintos snippets); $F(T_k; W)$ es la ConvNet con parámetros $W$ que produce *scores* de clase; $G$ es la **función de consenso segmental** que combina las salidas de los múltiples snippets; y $H$ es un **Softmax** que convierte el consenso en probabilidades. Es esencial que **todas las ConvNets comparten los parámetros $W$** — no hay $K$ redes, sino una sola aplicada $K$ veces.

El score de consenso de la clase $i$ se infiere de los scores de esa misma clase en todos los snippets, $G_i = g(F_i(T_1), \dots, F_i(T_K))$. Se evaluaron tres agregaciones para $g$: promedio uniforme, máximo y promedio ponderado; el **promedio uniforme** es el que rinde mejor. Combinando el consenso con entropía cruzada categórica, la pérdida es:

$$L(y, G) = -\sum_{i=1}^{C} y_i \left( G_i - \log \sum_{j=1}^{C} \exp G_j \right)$$

El punto clave es que **se optimizan las pérdidas a nivel de video, no de snippet individual** como en la two-stream original. El gradiente respecto de $W$,

$$\frac{\partial L(y, G)}{\partial W} = \frac{\partial L}{\partial G} \sum_{k=1}^{K} \frac{\partial G}{\partial F(T_k)} \frac{\partial F(T_k)}{\partial W},$$

garantiza que cada actualización de parámetros usa el consenso $G$ derivado de todas las predicciones de snippet. Así el muestreo esparcido **no es un truco de inferencia**, sino que está integrado en el entrenamiento: la red aprende a producir predicciones de snippet que, agregadas, describen bien la acción completa. En los experimentos $K$ se fija en **3**.

Para desatar el potencial del marco con redes profundas (BN-Inception) sin caer en sobreajuste, el paper sistematiza varias **buenas prácticas**: *pre-entrenamiento cruzado de modalidades* (usar modelos RGB de ImageNet para inicializar las redes de flujo óptico, promediando y replicando los pesos de la primera capa); *partial BN con dropout* (congelar media y varianza de todas las capas BatchNorm salvo la primera, más dropout 0.8 espacial / 0.7 temporal); y *aumento de datos mejorado* (recorte de esquinas y *scale-jittering* multiescala).

## Resultados

Sobre UCF101 split 1, un análisis de componentes muestra la contribución de cada pieza: two-stream básico **90.0%** → cross-modality pre-training **91.5%** → partial BN con dropout **92.0%** → **Temporal Segment Networks 93.5%**. TSN mejora el rendimiento **aun cuando todas las buenas prácticas ya están aplicadas**, corroborando que modelar la estructura temporal de largo plazo es crucial. Entrenar desde cero, en cambio, da solo **82.9%**, por debajo del baseline de la two-stream original (87.0%), lo que confirma la necesidad de estas estrategias contra el sobreajuste.

Ensamblando tres modalidades (RGB + flujo óptico + flujo *warped*) y todas las técnicas, TSN alcanza **HMDB51 = 69.4%** y **UCF101 = 94.2%**, superando tanto a métodos tradicionales (iDT, MoFAP) como de aprendizaje profundo (C3D, TDD, LTC), con una mejora de **3.9%** en HMDB51 y **1.1%** en UCF101 sobre el mejor método previo. Con la herramienta DeepDraw, los autores visualizan además que los modelos de corto plazo confunden escena y objetos con la acción (en "Diving" buscan agua y plataformas), mientras que con el modelado de largo plazo de TSN se concentran en el humano y capturan distintas poses de las etapas de la acción.

## Limitaciones

- **Dependencia del flujo óptico.** El mejor rendimiento requiere flujo óptico (y flujo *warped*), cuyo cálculo (TVL1) es costoso y debe precomputarse, lo que complica el despliegue en tiempo real.
- **Consenso simple.** La función de consenso definitiva es un promedio uniforme que trata todos los segmentos por igual y **no aprende relaciones temporales** (orden, causalidad). Es una agregación tipo *bag-of-segments*: modela qué ocurre, no en qué orden.
- **$K$ pequeño y fijo.** Se fija $K=3$ en entrenamiento; un $K$ pequeño limita la granularidad temporal y su valor óptimo depende del dataset.
- **Datasets acotados.** Los resultados se demuestran en clips recortados (una acción por clip); el marco no aborda la localización temporal en videos largos no recortados.

## Por qué importa para la Clase 36

La [Clase 36](/clases/clase-36) (Introduction to Video Analysis) introduce el problema de clasificar la acción de un video, y su [laboratorio](/laboratorios/lab-36) implementa la solución más pragmática: **muestrear un número fijo de frames (8) distribuidos a lo largo del video** y clasificar a partir de ellos. Esa decisión de diseño *es* la idea de TSN:

| Idea de TSN | Reflejo en el LAB de la Clase 36 |
|---|---|
| Dividir el video en $K$ segmentos de igual duración | Distribuir $N=8$ posiciones de muestreo a lo largo del video |
| Muestrear un snippet por segmento | Tomar un frame por posición |
| Muestreo esparcido cubre todo el video con poco cómputo | Con 8 frames se representa el video completo, sin procesar cada frame |
| Frames consecutivos redundantes → densidad innecesaria | Pocos frames bien distribuidos bastan para clasificar la acción |
| Función de consenso agrega predicciones de snippet | Se promedian las representaciones de los frames para decidir a nivel de video |
| Parámetros compartidos entre snippets | La misma red procesa cada frame muestreado |

La lección conceptual es la **eficiencia del muestreo esparcido**: no hace falta ver todos los frames para reconocer una acción, porque la redundancia entre frames consecutivos es enorme. Distribuir unas pocas muestras uniformemente por la línea de tiempo cubre la estructura temporal de largo alcance, reduce el costo por uno o dos órdenes de magnitud y actúa como regularización implícita. TSN construye directamente sobre las [two-stream ConvNets](/papers/two-stream-simonyan-2014) llevándolas del nivel de clip al nivel de video, y sus buenas prácticas de entrenamiento con datos escasos son el conector natural con el [fundamento de reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).
