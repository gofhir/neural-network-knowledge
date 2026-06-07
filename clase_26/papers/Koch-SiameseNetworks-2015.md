# Siamese Neural Networks for One-shot Image Recognition — Análisis interno exhaustivo

## 1. Metadata y resumen ejecutivo

**Título:** Siamese Neural Networks for One-shot Image Recognition
**Autores:** Gregory Koch, Richard Zemel, Ruslan Salakhutdinov
**Afiliación:** Department of Computer Science, University of Toronto, Ontario, Canadá.
**Venue:** ICML 2015 Deep Learning Workshop (Proceedings of the 32nd International Conference on Machine Learning, Lille, Francia, 2015. JMLR: W&CP volume 37).
**Dominio:** reconocimiento de imágenes / one-shot learning / aprendizaje de métricas profundas (deep metric learning).
**Dataset principal:** Omniglot (50 alfabetos, 105×105 imágenes binarias dibujadas a mano por 20 escritores). Experimento secundario: MNIST 10-versus-1.

Este es uno de los papers fundacionales del deep metric learning moderno y del few-shot learning. La idea es engañosamente simple pero profundamente influyente: en vez de entrenar un clasificador que mapea una imagen a una de N clases fijas, se entrena una red que aprende a **comparar dos imágenes** y decidir si pertenecen a la misma clase o a clases distintas. Esa tarea de comparación se llama **verificación** (verification). Una vez aprendida una buena función de similitud sobre un conjunto de clases conocidas (el background set), esa función se reutiliza —sin reentrenar— para clasificar ejemplos de clases completamente nuevas que la red nunca vio durante el entrenamiento. Esto es el corazón del one-shot learning: clasificar correctamente cuando solo se dispone de **un único ejemplo** de cada clase nueva.

La arquitectura concreta es una red convolucional **siamesa**: dos torres gemelas que comparten exactamente los mismos pesos, cada una procesa una de las dos imágenes de entrada, y al tope se computa la distancia $L_1$ componente a componente entre los dos embeddings, ponderada por parámetros aprendidos $\alpha_j$ y pasada por una sigmoide para producir una probabilidad de "misma clase".

**Resultados clave (números reales del paper):**
- En la tarea **20-way within-alphabet one-shot** de Omniglot, la red convolucional siamesa alcanza **92.0%** de accuracy, contra **95.5%** de humanos y **95.2%** de HBPL (Hierarchical Bayesian Program Learning, el estado del arte de Lake et al.). Supera a todos los demás baselines: Affine model 81.8%, Hierarchical Deep 65.2%, Deep Boltzmann Machine 62.0%, Simple Stroke 35.2%, 1-Nearest Neighbor 21.7%. Una red siamesa **no convolucional** (fully-connected) logra solo 58.3%, lo que evidencia la importancia de la arquitectura convolucional.
- En la tarea de verificación (Tabla 1), con 150k ejemplos de entrenamiento y distorsiones afines ×8, alcanza **93.42%** de accuracy.
- En MNIST 10-versus-1 one-shot (transfiriendo features aprendidos en Omniglot, **sin ningún fine-tuning** en MNIST), logra **70.3%** contra 26.5% de 1-NN.

La gran contribución conceptual: demostrar que una red discriminativa, sin priors generativos sobre el dominio (a diferencia de HBPL que modela explícitamente el proceso de dibujo con trazos), puede acercarse al rendimiento humano y al estado del arte en one-shot, simplemente aprendiendo features genéricos transferibles a clases desconocidas.

## 2. Contexto: el problema de one-shot recognition y los antecedentes siamese

**El problema.** El aprendizaje supervisado clásico requiere muchos ejemplos por clase y un conjunto fijo de clases conocido de antemano. Cuando aparece una clase nueva, el modelo debe reentrenarse, lo que es caro o imposible si hay pocos datos, o inviable en escenarios de predicción online (los autores mencionan web retrieval). Los humanos, en cambio, exhiben una capacidad notable: ante un estímulo nuevo entienden el concepto rápidamente y reconocen variaciones futuras del mismo (Lake et al., 2011).

El **one-shot learning** es la restricción extrema: se observa **un solo ejemplo** de cada clase posible antes de hacer una predicción sobre una instancia de test. Hay que distinguirlo del **zero-shot learning** (Palatucci et al., 2009), donde el modelo no puede ver ningún ejemplo de las clases objetivo y debe apoyarse en descripciones semánticas auxiliares.

**Antecedentes en one-shot.** La obra seminal data de principios de los 2000 con Fei-Fei et al., que desarrollaron un marco bayesiano variacional para clasificación one-shot, bajo la premisa de que las clases ya aprendidas pueden apalancarse para pronosticar clases futuras (Fe-Fei et al., 2003; Fei-Fei et al., 2006). Más cercano al paper, Lake et al. abordaron one-shot desde la ciencia cognitiva con **Hierarchical Bayesian Program Learning (HBPL)**, modelando generativamente el proceso de dibujar caracteres, descomponiendo la imagen en piezas pequeñas (trazos) y buscando una explicación estructural de los píxeles observados. La debilidad de HBPL: la inferencia es difícil porque el espacio de parámetros conjunto es enorme, llevando a un problema de integración intratable. HBPL es potente pero incorpora mucho conocimiento de dominio (sabe que los caracteres se dibujan con trazos).

**Antecedentes siamese.** Las redes siamesas fueron introducidas a inicios de los 1990 por **Bromley y LeCun** para resolver la **verificación de firmas** como un problema de matching de imágenes (Bromley et al., 1993, "Signature verification using a siamese time delay neural network"). Una red siamesa consiste en redes gemelas que aceptan entradas distintas pero están unidas por una **función de energía** en el tope, que computa una métrica entre la representación de más alto nivel de cada lado. Los parámetros de las gemelas están **atados** (tied weights).

Posteriormente, Chopra, Hadsell y LeCun (2005) usaron una **función de energía contrastiva** con términos duales para *disminuir* la energía de pares iguales y *aumentar* la de pares distintos, aplicado a verificación de rostros. La diferencia clave con el presente paper: LeCun et al. **aprendían directamente la métrica de similitud**, implícitamente definida por la pérdida de energía. Koch et al., en cambio, **fijan la métrica** (distancia $L_1$ ponderada + sigmoide), siguiendo el enfoque de **DeepFace** de Facebook (Taigman et al., 2014). Esto convierte el problema en una clasificación binaria limpia con objetivo cross-entropy.

**Por qué aprender una métrica de similitud en vez de un clasificador.** Un clasificador softmax sobre N clases tiene un número de salidas fijo: añadir una clase requiere modificar la arquitectura y reentrenar. Una métrica de similitud aprendida es **agnóstica al número de clases**: aprende qué hace que dos cosas sean "iguales" o "distintas" a nivel de features, y eso se transfiere a clases nunca vistas. La hipótesis central del paper es: *si una red es buena en verificación (decir si dos imágenes son de la misma clase), debería generalizar bien a one-shot classification*. Los autores razonan que si los features aprendidos bastan para confirmar o negar la identidad de caracteres de un conjunto de alfabetos, deberían bastar para otros alfabetos, siempre que la red haya sido expuesta a suficiente variedad de alfabetos para fomentar la varianza entre los features aprendidos.

## 3. La idea central: dos torres gemelas que comparten pesos

La estrategia general (Figura 2 del paper) tiene dos pasos:

1. **Entrenar** un modelo para discriminar entre una colección de pares *same/different* (misma clase / clases distintas).
2. **Generalizar** para evaluar categorías nuevas, basándose en los mapeos de features aprendidos, vía verificación.

La red siamesa toma **dos** entradas $x_1$ y $x_2$. Cada una pasa por una de las dos torres gemelas. La propiedad esencial es el **weight tying** (atado de pesos): ambas torres computan **exactamente la misma función** $f_\theta$. Esto tiene dos consecuencias formales:

- **Garantía de consistencia local.** Dos imágenes extremadamente similares no pueden ser mapeadas a ubicaciones muy distintas en el espacio de features, porque cada torre computa la misma función. Si $x_1 \approx x_2$, entonces $f_\theta(x_1) \approx f_\theta(x_2)$ por continuidad de la red.
- **Simetría.** La red es simétrica: presentar $(x_1, x_2)$ a las gemelas produce la misma métrica que presentar $(x_2, x_1)$ a las gemelas opuestas. Esto es deseable porque la relación "ser de la misma clase" es simétrica: $\text{sim}(x_1, x_2) = \text{sim}(x_2, x_1)$.

Cada torre mapea su entrada a un **embedding** (vector de features de alto nivel) $h_1 = f_\theta(x_1)$ y $h_2 = f_\theta(x_2)$. En el tope, una **capa de distancia** computa una métrica entre $h_1$ y $h_2$, y una **sigmoide** produce $p \in [0,1]$, la probabilidad de que $x_1$ y $x_2$ sean de la misma clase. La tarea de entrenamiento es así una verificación binaria.

La elegancia del diseño: el conocimiento del dominio no se inyecta a mano (como en HBPL con sus trazos), sino que **emerge** de aprender qué features hacen que dos imágenes sean comparables como iguales o distintas. Esos features, una vez aprendidos sobre 30-40 alfabetos, capturan invariancias generales de la escritura manuscrita (curvas, intersecciones, terminaciones de trazo) que se transfieren a alfabetos nunca vistos.

## 4. La arquitectura concreta: la convnet siamesa y la distancia L1 ponderada

El modelo estándar es una red convolucional siamesa con $L$ capas, cada una con $N_l$ unidades. $h_{1,l}$ denota el vector oculto en la capa $l$ para la primera gemela, $h_{2,l}$ para la segunda. Se usan unidades **ReLU** en las primeras $L-2$ capas y unidades sigmoidales en las restantes.

**Capas convolucionales.** El modelo es una secuencia de capas convolucionales, cada una con un solo canal, filtros de tamaño variable y **stride fijo de 1**. El número de filtros convolucionales se especifica como múltiplo de 16 para optimizar rendimiento. Cada capa aplica ReLU sobre los feature maps de salida, opcionalmente seguido de **max-pooling** con filtro y stride de 2. El $k$-ésimo feature map en cada capa toma la forma:

$$a^{(k)}_{1,m} = \text{max-pool}\!\left(\max\!\left(0,\; W^{(k)}_{l-1,l} \star h_{1,(l-1)} + b_l\right),\, 2\right)$$

$$a^{(k)}_{2,m} = \text{max-pool}\!\left(\max\!\left(0,\; W^{(k)}_{l-1,l} \star h_{2,(l-1)} + b_l\right),\, 2\right)$$

donde $W_{l-1,l}$ es el tensor tridimensional de los feature maps de la capa $l$, y $\star$ es la convolución **válida** (valid convolution): devuelve solo las unidades de salida resultantes del solapamiento completo entre el filtro y los feature maps de entrada (sin padding). Nótese que **ambas ecuaciones comparten los mismos $W^{(k)}_{l-1,l}$ y $b_l$**: ahí está materializado el weight tying.

**Cabeza de distancia.** Las unidades de la última capa convolucional se aplanan en un único vector. Sigue una capa fully-connected, y luego una capa que computa la **métrica de distancia inducida** entre las dos gemelas, alimentando a una única unidad sigmoidal de salida. La predicción es:

$$p = \sigma\!\left(\sum_j \alpha_j \,\bigl|\,h^{(j)}_{1,L-1} - h^{(j)}_{2,L-1}\,\bigr|\right)$$

donde $\sigma$ es la sigmoide. Desglose:

- $h^{(j)}_{1,L-1}$ y $h^{(j)}_{2,L-1}$ son la $j$-ésima componente del embedding de cada gemela en la capa $(L-1)$ (la fully-connected de 4096 unidades en el modelo más grande).
- $\bigl|h^{(j)}_1 - h^{(j)}_2\bigr|$ es la **distancia $L_1$ componente a componente** (valor absoluto de la diferencia, dimensión a dimensión). Esto produce un vector de la misma dimensión que los embeddings, no un escalar: cada componente mide cuánto difieren las gemelas en esa dimensión de feature.
- Los $\alpha_j$ son **parámetros adicionales aprendidos** durante el entrenamiento, que ponderan la importancia de cada componente de la distancia. Algunas dimensiones de feature serán más discriminativas que otras para decidir "misma clase", y los $\alpha_j$ aprenden ese peso.
- La suma ponderada $\sum_j \alpha_j |\cdots|$ colapsa el vector de distancias a un escalar, y la sigmoide lo mapea a $[0,1]$.

Esta capa final induce una métrica sobre el espacio de features de la capa $(L-1)$ y puntúa la similitud entre los dos vectores. Es la $L$-ésima capa fully-connected, la que une las dos gemelas.

**El modelo más grande** (Figura 4, el que dio el mejor resultado en verificación): la gemela se une justo después de la capa fully-connected de **4096 unidades**, donde se computa la distancia $L_1$ componente a componente. La arquitectura concreta usa varias capas convolucionales con filtros de tamaño decreciente (el espacio de búsqueda fue 3×3 a 20×20) y número de filtros creciente (16 a 256, en múltiplos de 16), seguidas de la FC de 4096.

Detalle a notar: la elección de $L_1$ (Manhattan) en lugar de $L_2$ (euclidiana) es deliberada. La $L_1$ componente a componente preserva información por dimensión antes de ponderar, mientras que $L_2$ ya colapsaría a una distancia escalar. Combinar $|h_1 - h_2|$ con la sigmoide hace que el objetivo cross-entropy sea la elección natural, porque la salida ya está en $[0,1]$.

## 5. La función de pérdida: cross-entropy regularizada sobre pares

Sea $M$ el tamaño del minibatch, $i$ el índice del minibatch. Sea $y(x^{(i)}_1, x^{(i)}_2)$ un vector de largo $M$ con las etiquetas del minibatch, donde:

$$y(x^{(i)}_1, x^{(i)}_2) = \begin{cases} 1 & \text{si } x_1, x_2 \text{ son de la misma clase de carácter} \\ 0 & \text{en caso contrario} \end{cases}$$

El objetivo es **cross-entropy binaria regularizada**:

$$\mathcal{L}(x^{(i)}_1, x^{(i)}_2) = y(x^{(i)}_1, x^{(i)}_2)\,\log p(x^{(i)}_1, x^{(i)}_2) + \bigl(1 - y(x^{(i)}_1, x^{(i)}_2)\bigr)\,\log\bigl(1 - p(x^{(i)}_1, x^{(i)}_2)\bigr) + \boldsymbol{\lambda}^T |\mathbf{w}|^2$$

El primer bloque es la log-verosimilitud binaria estándar (log loss): cuando la etiqueta es 1, premia que $p \to 1$; cuando es 0, premia que $p \to 0$. El término $\boldsymbol{\lambda}^T|\mathbf{w}|^2$ es **regularización $L_2$** sobre los pesos, con $\boldsymbol{\lambda}$ un vector de penalizaciones definidas **por capa** (layer-wise): cada capa puede tener su propio coeficiente de regularización $\lambda_j$.

**Estrategia de muestreo de pares.** Esta es una decisión crítica y a menudo subestimada del paradigma siamés. El entrenamiento opera sobre **pares**, no sobre ejemplos individuales. Los autores construyeron tres conjuntos de entrenamiento de tamaños 30.000, 90.000 y 150.000 ejemplos, muestreando aleatoriamente pares *same* y *different*. Apartaron el 60% del total para entrenamiento: 30 alfabetos de 50 y 12 escritores de 20.

Un punto fino del balanceo: fijaron un **número uniforme de ejemplos de entrenamiento por alfabeto**, de modo que cada alfabeto reciba representación equitativa durante la optimización (aunque esto no garantiza balance a nivel de clases de carácter individuales dentro de cada alfabeto). Esto previene que alfabetos grandes (hasta 40 caracteres) dominen sobre alfabetos pequeños (15 caracteres).

El muestreo de pares determina implícitamente la dificultad del problema: hay muchos más pares posibles *different* que *same* (combinatoria de clases), por lo que muestrear pares balanceados o ponderados es necesario para que la señal de "misma clase" no se diluya. Este es precisamente el problema que años después motivó técnicas como **hard negative mining** en triplet loss.

## 6. De verificación a one-shot classification

Una vez optimizada la red siamesa para dominar la verificación, se la usa directamente en one-shot, **sin reentrenar**. El procedimiento de inferencia:

Supóngase una imagen de test $\mathbf{x}$ (un vector columna) que queremos clasificar en una de $C$ categorías. Se nos dan también $C$ imágenes $\{x_c\}_{c=1}^{C}$, una por cada categoría posible (el **support set**, un ejemplo por clase). Se consulta la red usando cada par $(\mathbf{x}, x_c)$ como entrada para $c = 1, \dots, C$. Cada consulta produce $p(c)$, la probabilidad de que $\mathbf{x}$ y $x_c$ sean de la misma clase. Luego se predice la clase con máxima similitud:

$$C^* = \arg\max_c \; p(c)$$

Es decir: comparar la query contra cada ejemplo del support set, y elegir la clase del par con mayor probabilidad de "misma clase". Esto convierte una red de verificación binaria en un clasificador multiclase **sin parámetros adicionales y sin reentrenamiento**, definido enteramente por la métrica aprendida y el support set.

**Eficiencia.** Los autores notan que esto se puede procesar eficientemente apilando $C$ copias de $\mathbf{x}$ en una matriz $X$ y los $x_c^T$ en filas de otra matriz $X_C$, de modo que basta **un único forward pass** con minibatch de tamaño $C$ usando $(X, X_C)$ como entrada. Esto es importante: la clasificación de una query es una sola pasada batched, no $C$ pasadas secuenciales.

## 7. Detalles de optimización

**Backpropagation con pesos atados.** El objetivo se combina con backprop estándar, donde el gradiente es **aditivo a través de las gemelas** debido a los pesos atados. Es decir, como ambas torres comparten $\mathbf{w}$, el gradiente total respecto de un peso es la suma de las contribuciones de ambos caminos. Tamaño de minibatch fijo en **128**.

**Regla de actualización layer-wise.** Con learning rate $\eta_j$, momentum $\mu_j$ y regularización $L_2$ con peso $\lambda_j$ definidos por capa, la actualización en la época $T$:

$$w^{(T)}_{kj}(x^{(i)}_1, x^{(i)}_2) = w^{(T)}_{kj} + \Delta w^{(T)}_{kj}(x^{(i)}_1, x^{(i)}_2) + 2\lambda_j |w_{kj}|$$

$$\Delta w^{(T)}_{kj}(x^{(i)}_1, x^{(i)}_2) = -\eta_j \nabla w^{(T)}_{kj} + \mu_j \Delta w^{(T-1)}_{kj}$$

donde $\nabla w_{kj}$ es la derivada parcial respecto del peso entre la $j$-ésima neurona de una capa y la $k$-ésima de la capa siguiente. El término $\mu_j \Delta w^{(T-1)}_{kj}$ es el momentum clásico (acumula el incremento previo), y $2\lambda_j|w_{kj}|$ es el decaimiento de pesos por regularización. Nótese que cada **capa** tiene su propio $\eta_j$, $\mu_j$, $\lambda_j$: el modelo tiene libertad para optimizar a distinta velocidad cada nivel de abstracción.

**Inicialización de pesos.**
- Pesos convolucionales: distribución normal de media cero y desviación estándar $10^{-2}$.
- Biases (todas las capas): normal con media $0.5$ y desviación estándar $10^{-2}$. (El bias positivo inicial favorece que las ReLU estén activas al inicio.)
- Pesos fully-connected: normal de media cero pero desviación estándar mucho mayor, $2\times10^{-1}$.

**Schedule de learning rate.** Aunque cada capa tiene su propio learning rate, todos decaen **uniformemente un 1% por época**: $\eta^{(T)}_j = 0.99\, \eta^{(T-1)}_j$. El annealing ayudó a converger a mínimos locales sin quedar atascado en la superficie de error. El **momentum** arranca en 0.5 en todas las capas y crece linealmente cada época hasta alcanzar el valor $\mu_j$ específico de la capa.

**Criterio de parada.** Se entrena cada red por un máximo de **200 épocas**, monitoreando el error de one-shot validation sobre un conjunto de **320 tareas one-shot** generadas aleatoriamente desde alfabetos y escritores del conjunto de validación. Si el error de validación no decrece por 20 épocas, se detiene y se usan los parámetros del mejor epoch según el error de one-shot validation (early stopping). Punto importante: el criterio de parada se basa en **proxy de la tarea objetivo** (one-shot), no solo en el error de verificación; en la práctica esto fue al menos tan efectivo como el error de verificación.

**Búsqueda bayesiana de hiperparámetros.** Se usó la versión beta de **Whetlab**, un framework de optimización bayesiana, para seleccionar hiperparámetros. Rangos buscados:
- learning rate layer-wise: $\eta_j \in [10^{-4}, 10^{-1}]$
- momentum layer-wise: $\mu_j \in [0, 1]$
- regularización $L_2$ layer-wise: $\lambda_j \in [0, 0.1]$
- tamaño de filtros convolucionales: de 3×3 a 20×20
- número de filtros por capa: 16 a 256 (múltiplos de 16)
- unidades en capas fully-connected: 128 a 4096 (múltiplos de 16)

El optimizador se configuró para **maximizar la accuracy de one-shot validation**, y la puntuación de cada iteración de Whetlab fue el valor más alto de esa métrica encontrado en cualquier época.

**Data augmentation con transformaciones afines.** Se aumentó el set de entrenamiento con pequeñas distorsiones afines (Figura 5). Para cada par de imágenes $(x_1, x_2)$ se generó un par de transformaciones afines $(T_1, T_2)$ para producir $x'_1 = T_1(x_1)$, $x'_2 = T_2(x_2)$, determinadas estocásticamente por una distribución uniforme multidimensional. Una transformación arbitraria $T$ se parametriza como:

$$T = (\theta, \rho_x, \rho_y, s_x, s_y, t_x, t_y)$$

con rangos: $\theta \in [-10.0, 10.0]$ (rotación en grados), $\rho_x, \rho_y \in [-0.3, 0.3]$ (shear/cizalla), $s_x, s_y \in [0.8, 1.2]$ (escala), $t_x, t_y \in [-2, 2]$ (traslación). **Cada componente se incluye con probabilidad 0.5**. Se añadieron **8 transformaciones** por ejemplo de entrenamiento, de modo que los datasets aumentados pasaron de 30k/90k/150k a **270.000 / 810.000 / 1.350.000** ejemplos efectivos.

## 8. Experimentos en Omniglot

**El dataset Omniglot.** Recolectado por Brenden Lake y colaboradores en el MIT vía Amazon Mechanical Turk, como benchmark estándar para aprendizaje desde pocos ejemplos en reconocimiento de caracteres manuscritos (Lake et al., 2011). Contiene **50 alfabetos**, desde lenguas internacionales bien establecidas (latín, coreano) hasta dialectos locales menos conocidos, e incluso conjuntos de caracteres ficticios (Aurek-Besh, Klingon). Cada carácter es una imagen binaria de **105×105**, dibujada a mano en un canvas online. Las trayectorias de los trazos se recolectaron junto a las imágenes compuestas, así que es posible incorporar información temporal y estructural. El número de letras por alfabeto varía considerablemente, de ~15 a más de 40 caracteres. Cada carácter fue producido **una sola vez** por cada uno de **20 escritores** (drawers).

Lake dividió los datos en un **background set de 40 alfabetos** (para desarrollo del modelo: aprender hiperparámetros y mapeos de features) y un **evaluation set de 10 alfabetos** (usado solo para medir el rendimiento de one-shot). Esta separación es estricta: las clases de evaluación nunca se ven en entrenamiento. A Omniglot se le llama el "MNIST transpuesto": muchísimas clases, pocos ejemplos por clase (lo opuesto a MNIST).

**Verificación (Tabla 1).** Seis corridas variando tamaño (30k/90k/150k) y distorsiones (sí/no):

| Configuración | Test accuracy |
|---|---|
| 30k sin distorsiones | 90.61 |
| 30k afines ×8 | 91.90 |
| 90k sin distorsiones | 91.54 |
| 90k afines ×8 | 93.15 |
| 150k sin distorsiones | 91.63 |
| 150k afines ×8 | **93.42** |

Dos lecciones: (1) más datos ayudan, pero con retornos decrecientes (91.54 → 91.63 de 90k a 150k sin distorsiones); (2) las distorsiones afines aportan consistentemente ~1.3-1.8 puntos, más que duplicar los datos crudos. Los filtros de primera capa aprendidos (Figura 7) asumen roles diferenciados: algunos detectan features point-wise muy pequeños, otros funcionan como detectores de bordes de mayor escala.

**Protocolo 20-way one-shot.** Lake desarrolló una tarea de clasificación **within-alphabet** de 20 vías. Se elige un alfabeto del evaluation set, y 20 caracteres uniformemente al azar. Se seleccionan **dos** escritores del pool de evaluación. Cada carácter producido por el **primer escritor** es una imagen de test, comparada individualmente contra los **20 caracteres del segundo escritor**, con el objetivo de predecir la clase del test. El que sea *within-alphabet* lo hace difícil: los 20 distractores son del mismo alfabeto, comparten estilo visual. El proceso se repite dos veces por alfabeto, dando 40 trials por alfabeto × 10 alfabetos = **400 trials one-shot** en total, de los que se calcula la accuracy.

**Resultados one-shot (Tabla 2):**

| Método | Test accuracy |
|---|---|
| Humans | 95.5 |
| Hierarchical Bayesian Program Learning (HBPL) | 95.2 |
| Affine model | 81.8 |
| Hierarchical Deep | 65.2 |
| Deep Boltzmann Machine | 62.0 |
| Simple Stroke | 35.2 |
| 1-Nearest Neighbor | 21.7 |
| Siamese Neural Net (no convolucional) | 58.3 |
| **Convolutional Siamese Net** | **92.0** |

Lectura: con **92.0%**, el método convolucional es más fuerte que cualquier modelo excepto HBPL (95.2%), que a su vez está apenas por debajo de los humanos (95.5%). La brecha entre la siamesa convolucional (92.0%) y la fully-connected (58.3%) confirma que la convolución —con su connectividad local, compartición de parámetros e invariancia a traslación— es esencial para este resultado. El salto sobre 1-NN (21.7%) es dramático: comparar píxeles crudos en el espacio original es casi inútil; comparar en el espacio de features aprendido es casi tan bueno como un humano.

La ventaja conceptual sobre HBPL: la red siamesa **no incorpora ningún conocimiento previo sobre caracteres o trazos** (no sabe nada del proceso generativo de dibujo), mientras HBPL modela explícitamente los trazos. Que un modelo discriminativo agnóstico al dominio llegue a 92% es el argumento de fondo del paper.

**MNIST 10-versus-1 (Tabla 3).** Para probar transferibilidad cross-domain, trataron los 10 dígitos de MNIST como un "alfabeto" y evaluaron one-shot 10-way, **sin ningún fine-tuning** en MNIST. Imágenes de 28×28 upsampleadas a 35×35, dadas a una versión reducida del modelo entrenado en Omniglot a 35×35 (downsampleadas por factor 3). 400 trials.

| Método | Test accuracy |
|---|---|
| 1-Nearest Neighbor | 26.5 |
| Convolutional Siamese Net | 70.3 |

El 1-NN se comporta similar a Omniglot, mientras la red cae más (de 92% a 70.3%) por el cambio de dominio, pero **aún generaliza razonablemente** a un dataset jamás visto sin entrenar en él. Es una demostración temprana de transfer learning de features de similitud entre dominios.

## 9. Por qué importa

Este paper estableció el **deep metric learning** como el enfoque dominante para one-shot y few-shot learning. Su contribución no es una técnica matemática nueva (la siamesa existía desde 1993, la cross-entropy es estándar), sino la **demostración empírica** de que:

1. Aprender una función de similitud sobre clases conocidas produce features que **transfieren a clases desconocidas** sin reentrenar.
2. Un enfoque **discriminativo y agnóstico al dominio** puede competir con métodos generativos cargados de priors de dominio (HBPL) y acercarse al rendimiento humano.
3. La verificación es una tarea proxy poderosa para el one-shot: optimizar "¿son iguales estas dos?" induce features que sirven para "¿cuál de estas $C$ es?".

Es el **antecedente directo** de toda una línea de trabajo: Matching Networks (Vinyals et al., 2016), Prototypical Networks (Snell et al., 2017), Relation Networks, y de la verificación facial industrial (FaceNet, Schroff et al., 2015, que reemplazó la siamesa de pares por triplets). El embedding-then-compare se volvió el patrón canónico de retrieval, búsqueda por similitud y few-shot.

## 10. Limitaciones

1. **Entrenamiento por pares, no episódico.** El paper entrena sobre pares *same/different* muestreados independientemente, no sobre **episodios** que simulen exactamente la tarea de test (N-way K-shot). Matching Networks y Prototypical Networks corrigieron esto con el principio "train as you test": construir episodios de entrenamiento idénticos en estructura a los de evaluación, lo que mejora la generalización. La siamesa aquí solo aproxima la tarea one-shot vía el criterio de parada (las 320 tareas de validación), no vía el objetivo de entrenamiento.

2. **No usa el contexto del support set completo.** En inferencia, la query se compara con cada ejemplo del support set **independientemente** ($p(c)$ se computa par a par). El modelo nunca ve los $C$ candidatos en conjunto, así que no puede razonar comparativamente ("este se parece más al carácter 3 que al 7 porque 3 y 7 son distintos entre sí"). Matching Networks introdujo atención sobre todo el support set; Prototypical Networks promedia los embeddings por clase para formar prototipos.

3. **La métrica es fija una vez entrenada.** El espacio de embeddings y los pesos $\alpha_j$ quedan congelados tras el entrenamiento. No hay adaptación a la distribución específica del support set en test time (no hay fine-tuning few-shot ni meta-aprendizaje del optimizador, como sí haría MAML años después). La métrica $L_1$ ponderada también es una elección de diseño fija, no aprendida desde cero como en los enfoques de energía de LeCun.

4. **Dependencia del muestreo de pares y del balanceo.** El rendimiento depende de cómo se muestrean los pares *same/different* y de la representación uniforme por alfabeto. No hay hard negative mining, así que muchos pares *different* fáciles aportan poca señal de gradiente.

5. **Costo del augmentation y la búsqueda de hiperparámetros.** Llegar a 92% requirió data augmentation afín ×8 y búsqueda bayesiana extensa de hiperparámetros layer-wise (learning rate, momentum, regularización, tamaños de filtro, todo por capa). Es un modelo cuidadosamente tuneado, no un resultado out-of-the-box.

## 11. Legado

- **Triplet loss y FaceNet (Schroff et al., 2015).** En vez de pares, FaceNet usa **triplets** (anchor, positivo, negativo) y una pérdida que fuerza $\|f(a) - f(p)\|^2 + \text{margin} < \|f(a) - f(n)\|^2$. Resuelve el problema de calibración de la decisión binaria de la siamesa y permite hard/semi-hard negative mining. FaceNet llevó la verificación facial a niveles de producción.
- **Matching Networks (Vinyals et al., 2016).** Entrenamiento episódico end-to-end con atención sobre el support set, y la noción de que la arquitectura de entrenamiento debe espejar la de test.
- **Prototypical Networks (Snell et al., 2017).** Cada clase se representa por el **centroide** (prototipo) de los embeddings de su support set; la clasificación es por distancia euclidiana al prototipo más cercano. Simple, robusto, y conecta directamente con la siamesa: es metric learning con un agregado por clase.
- **Todo el metric learning moderno.** Contrastive loss, triplet loss, InfoNCE, SimCLR/MoCo (aprendizaje contrastivo auto-supervisado), y los **bi-encoders / dual-encoders** para retrieval semántico (sentence embeddings, dense retrieval) descienden conceptualmente de la idea siamesa: dos torres que comparten pesos, embeddings comparables por una métrica.
- **Aplicaciones de verificación.** Verificación facial, verificación de firmas (el origen mismo en Bromley-LeCun 1993), re-identificación de personas, detección de duplicados, y record linkage.

## 12. Conexión con la Clase 26 y relevancia para salud (FHIR / MDM)

**Conexión con la Clase 26 (métodos no-paramétricos, redes siamesas).** La red siamesa de Koch es el puente perfecto entre los métodos no-paramétricos clásicos (k-NN: clasificar por vecindad en un espacio de features) y el deep learning. El one-shot por argmax de similitud es, en esencia, **un 1-NN sobre un espacio de embeddings aprendido**: $C^* = \arg\max_c p(c)$ es vecino más cercano según una métrica aprendida en vez de la euclidiana sobre píxeles crudos. La diferencia entre el 21.7% de 1-NN crudo y el 92.0% de la siamesa es exactamente el valor de aprender la métrica. La clase trata las redes siamesas como el caso donde el "espacio" sobre el que opera el método no-paramétrico se aprende discriminativamente.

**Relevancia directa para tu trabajo en FHIR / patient matching / MDM.** Este es, conceptualmente, **el mismo paradigma que el record linkage y el patient matching**. El problema de MDM (Master Data Management) en salud es: dados dos registros de paciente —posiblemente de sistemas distintos, con nombres tipeados de forma diferente, fechas de nacimiento con errores, RUT con dígitos transpuestos— decidir si son **la misma entidad** (misma persona) o personas distintas. Eso es **exactamente la tarea de verificación** del paper: $p(\text{misma clase} \mid x_1, x_2)$, salvo que aquí $x_1, x_2$ son registros de paciente en vez de imágenes de caracteres.

El mapeo es casi uno a uno con tu arquitectura MDM (bi-encoder como blocker + GBM como scorer, según tu memoria del proyecto FALP):

- **Las dos torres gemelas que comparten pesos** = tu **bi-encoder**: cada registro de paciente se mapea a un embedding mediante el mismo encoder. El weight tying es lo que garantiza que dos registros similares caigan cerca en el espacio de embeddings —exactamente la propiedad que necesitas para que el bi-encoder funcione como **blocker** (recuperar candidatos por vecindad ANN antes de scorear).
- **La distancia $L_1$ ponderada + sigmoide** = la cabeza de scoring que decide "match / no-match". En tu caso, el scorer principal es un GBM (XGBoost ONNX) que toma features de comparación de campos; pero el principio es idéntico al $\sum_j \alpha_j |h_1^{(j)} - h_2^{(j)}|$: una combinación ponderada de distancias componente a componente, donde los pesos aprenden qué campos/dimensiones son más discriminativos para la identidad. El GBM es simplemente un scorer más expresivo (no lineal) que la sigmoide lineal sobre $L_1$.
- **El muestreo de pares same/different** = tu generación de pares de entrenamiento de match/no-match, con el mismo desafío de desbalance (hay muchísimos más pares no-match que match) que el paper enfrenta con su balanceo por alfabeto. Es el mismo problema que en MDM se ataca con blocking + muestreo de hard negatives (pares casi-iguales pero distintos: dos hermanos, un padre y un hijo con mismo apellido).
- **El one-shot por argmax** = la decisión de linkage: ante un registro entrante, compararlo contra los candidatos del golden record y elegir (o crear) la entidad de máxima probabilidad de match. El truco de batching ($X$, $X_C$ en un solo forward pass) es directamente aplicable a scorear un registro contra todos sus candidatos de bloque en una pasada.
- **La transferencia cross-domain (Omniglot → MNIST sin fine-tuning, 70.3%)** es la promesa —y la advertencia— para tu paper sobre **retornos decrecientes del ML en MDM LATAM**: features de similitud aprendidos en un dominio transfieren parcialmente a otro, pero con degradación. Un bi-encoder entrenado en una población no transfiere perfectamente a otra con distintas convenciones de nombres; el paper cuantifica esa caída (92% → 70%) en un setting limpio, lo que respalda tu argumento de que más sofisticación ML rinde menos de lo esperado cuando cambia la distribución de datos.

La limitación del paper —**la métrica fija y la verificación par a par sin contexto del set completo**— también es directamente tu limitación de diseño en MDM: el scorer par a par no razona sobre el cluster de candidatos en conjunto (no resuelve transitividad: si A=B y B=C, entonces A=C), lo que en record linkage se resuelve con clustering/resolución de entidades sobre el grafo de matches, análogo a cómo Matching/Prototypical Networks agregaron contexto del support set. Y la observación de que **la calidad del blocking domina** (1-NN crudo da 21.7%) refuerza que en MDM el blocker (tu bi-encoder) y el scorer (tu GBM) son piezas complementarias: el embedding aprendido hace el trabajo pesado de poner candidatos cerca, y el scorer afina la decisión final.
