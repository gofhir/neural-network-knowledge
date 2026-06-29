# Supermasks in Superposition (SupSup) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Supermasks in Superposition*.
- **Autores:** Mitchell Wortsman∗ (University of Washington), Vivek Ramanujan∗ (Allen Institute for AI), Rosanne Liu (ML Collective), Aniruddha Kembhavi (Allen Institute for AI, también UW), Mohammad Rastegari (UW), Jason Yosinski (ML Collective), Ali Farhadi (UW). (∗ contribución igual.)
- **Venue:** 34th Conference on Neural Information Processing Systems (**NeurIPS 2020**), Vancouver, Canadá.
- **Año:** 2020. **Preprint:** arXiv:2006.14769v3 (22 oct 2020), [arxiv.org/abs/2006.14769](https://arxiv.org/abs/2006.14769).
- **Código:** [github.com/RAIVNLab/supsup](https://github.com/RAIVNLab/supsup).

SupSup (Supermasks in Superposition) es un modelo de **aprendizaje continuo** capaz de aprender *miles* de tareas en secuencia **sin olvido catastrófico**. Su punto de partida es una observación tan simple como potente: el olvido catastrófico no puede ocurrir si los pesos de la red **permanecen fijos y aleatorios**. SupSup explota esto manteniendo una red base inicializada al azar y nunca entrenada, y para cada tarea aprende únicamente una **supermáscara** (*supermask*) —una subred binaria que selecciona qué conexiones de esa red aleatoria quedan activas— que logra buen desempeño. Como los pesos jamás se tocan, las tareas previas no se degradan por construcción.

El modelo se apoya en dos ideas centrales. La primera: el **poder expresivo de subredes con pesos aleatorios**, observado por Zhou et al. [57] y Ramanujan et al. [39] —dentro de una red aleatoria existen subredes que ya resuelven tareas complejas sin entrenar los pesos, solo eligiendo la máscara correcta. La segunda, y la novedad de mayor calado: la **inferencia de la identidad de la tarea (task ID) como un problema de optimización**. Cuando no se sabe a qué tarea pertenecen los datos en inferencia, SupSup superpone todas las máscaras aprendidas —cada una ponderada por un coeficiente αᵢ— y usa gradientes para encontrar la combinación convexa que **minimiza la entropía de la salida**, recuperando así la máscara correcta. En la práctica, **un solo paso de gradiente** suele bastar para identificar la tarea correcta, incluso entre 2500 tareas.

Para la Clase 32 (Olvido Catastrófico) el paper importa porque la clase presenta SupSup como la **versión generalizada de Piggyback** (Mallya et al., 2018) que funciona *aunque no se sepa el task ID en inferencia*, usando un criterio de incerteza/confianza en la predicción para decidir qué máscara aplicar. Donde Piggyback exige que alguien le diga qué tarea está resolviendo, SupSup lo *deduce* del propio comportamiento de la red.

## 2. Contexto histórico: de la lottery ticket hypothesis a las supermáscaras, y de Piggyback a SupSup

El olvido catastrófico —el fenómeno por el cual entrenar una red en una tarea nueva degrada brutalmente su desempeño en las anteriores— está documentado desde McCloskey & Cohen [33] y French [10], y sigue siendo el obstáculo central del aprendizaje continuo [23]. Las familias clásicas de soluciones que el paper revisa son tres: (1) **regularización** (EWC [23] con la matriz de información de Fisher, SI [54]), que penaliza mover parámetros importantes para tareas previas pero solo atenúa el olvido sin eliminarlo; (2) **replay / ejemplares / modelos generativos** (iCaRL [40], GEM [27], replay generativo [45]), que recapturan datos pasados a costa de memoria o de generadores que también olvidan; y (3) **componentes específicos por tarea** (PNN [42], DEN [53], PackNet [31], BatchEnsemble [51], PSP [4]), donde distintas partes de la red sirven a distintas tareas.

SupSup pertenece a esta tercera familia, pero su linaje teórico viene de un lugar distinto: la **lottery ticket hypothesis** (Frankle & Carbin [8]). Esa hipótesis sostuvo que dentro de una red densa hay subredes ("billetes de lotería") que, entrenadas en aislamiento, alcanzan la precisión de la red completa. El giro radical lo dieron Zhou et al. [57] y Ramanujan et al. [39]: existen **supermáscaras** sobre una red **aleatoria y nunca entrenada** tales que la subred resultante ya resuelve la tarea —el conocimiento no está en los pesos, sino en *qué subconjunto de pesos aleatorios se elige*. El número de subredes posibles es combinatorio en el número de parámetros, y ese espacio gigantesco es lo bastante rico como para contener soluciones a tareas complejas. Ramanujan et al. [39] mostraron además que encontrar una supermáscara cuesta aproximadamente lo mismo que entrenar pesos y compite en precisión.

El antecedente directo en aprendizaje continuo es **Piggyback** (Mallya et al. [30]): para cada tarea nueva aprende una máscara binaria aplicada a una red *preentrenada en ImageNet*. SupSup lo generaliza en dos ejes. Primero, reemplaza la red preentrenada por una red **aleatoria fija** —lo que elimina la necesidad de almacenar los pesos (basta guardar la semilla aleatoria). Segundo, y más importante, Piggyback y todos los métodos de su categoría (incluidos BatchE [51] y PSP [4]) **están limitados al escenario en que el task ID se da tanto en entrenamiento como en inferencia**; SupSup rompe esa limitación e infiere el task ID cuando no se provee. La diferencia es la columna vertebral conceptual de la Clase 32.

## 3. Contribución central

La contribución de SupSup es un modelo de aprendizaje continuo que (a) **no puede sufrir olvido catastrófico** porque la red base nunca se entrena, solo se enmascara, y (b) **funciona en los cuatro escenarios de aprendizaje continuo**, incluidos aquellos donde no se conoce la identidad de la tarea —algo que ningún método previo de su familia lograba. Concretamente:

1. **Taxonomía de escenarios.** El paper propone una taxonomía de tres letras (GG, GNs, GNu, NNs) que ordena el campo según si el task ID se da en entrenamiento, si se da en inferencia, y si las etiquetas se comparten entre tareas (sección 2).
2. **Escenario GG (task ID dado en train y test):** SupSup es una extensión natural de Piggyback [30]; con un backbone aleatorio y control de sparsity supera baselines recientes en SplitImageNet usando menos almacenamiento y tiempo.
3. **Escenario GN (task ID dado en train, no en test):** SupSup infiere la tarea minimizando la entropía sobre una **superposición** de máscaras, escalando a **2500 permutaciones de MNIST** sin olvido, con la inferencia resuelta en una sola computación de gradiente.
4. **Escenario NNs (sin task ID nunca):** SupSup detecta cuándo está incierto sobre datos nuevos y **asigna una supermáscara nueva** automáticamente, infiriendo así los límites entre tareas.
5. **Almacenamiento en memoria constante (HopSupSup):** el conjunto creciente de máscaras puede guardarse implícitamente como **atractores de una red de Hopfield** de tamaño fijo.
6. **Neuronas superfluas:** añadir neuronas extra a la capa de salida mejora notablemente la inferencia de tarea.

La novedad clave, repetida a lo largo del texto, es **inferir el task ID sin que nadie lo provea**, framing la inferencia como una optimización sobre coeficientes de mezcla de máscaras.

## 4. Método

### 4.1. Preliminares: supermáscaras sobre pesos aleatorios fijos

En una clasificación estándar de ℓ vías, una entrada x se mapea a una distribución p = f(x, W). SupSup escribe la salida como **p = f(x, W ⊙ M)**, donde M es una máscara binaria y ⊙ es el producto elemento a elemento. W se mantiene **congelado en su inicialización**: los bias son 0 y cada peso es ±c con igual probabilidad, donde c es la desviación estándar de la distribución Kaiming normal de esa capa (inicialización *signed Kaiming constant* de [39]). Solo M se aprende. Para hallar las máscaras se usa el algoritmo **Edge-Popup** [39]: a cada conexión se le asocia un "score" real, en el forward se quedan activas las conexiones con score más alto (las top-k por presupuesto de sparsity), y aunque la máscara es binaria y no diferenciable, el gradiente se propaga a los scores mediante un *straight-through estimator*. Así se entrena la máscara a costo comparable al de entrenar pesos, dejando W intacto.

### 4.2. Escenario GG: task ID conocido en entrenamiento e inferencia

Cuando el task ID se conoce siempre, SupSup aprende una máscara binaria Mⁱ por tarea i (los únicos parámetros que se aprenden, pues W queda fijo). En inferencia para la tarea i se computa p = f(x, W ⊙ Mⁱ). Cada máscara nueva se inicializa al azar o, mejor aún, con una **media corriente** de las máscaras aprendidas hasta el momento —un truco de "Transfer" que funciona muy bien en la práctica. Este es el régimen de Piggyback, y aquí SupSup gana sobre todo por eficiencia de almacenamiento: como W es aleatorio, basta guardar la semilla, y las máscaras sparse se almacenan en formato `scipy.sparse.csc` con enteros de 16 bits.

### 4.3. Inferencia de tarea por entropía y superposición (GNs, GNu)

Es el corazón conceptual. Llegan datos de una tarea j desconocida y queremos recuperar la máscara Mʲ. Se asocia a cada una de las k máscaras un coeficiente αᵢ ∈ [0,1], inicializado en 1/k —interpretable como la "creencia" de que Mⁱ es la máscara correcta. La salida se computa con una **superposición ponderada** de todas las máscaras:

> p(α) = f( x, W ⊙ (Σᵢ αᵢ Mⁱ) )

La intuición [19]: la máscara correcta Mʲ debe producir una salida **confiada, de baja entropía**, mientras que las máscaras equivocadas producen salidas inciertas. Por tanto se buscan los coeficientes α que **minimizan la entropía H de p(α)**. En vez de probar las k máscaras una por una (k forward passes), se quiere un método de tiempo sublineal que lleve α a una esquina del símplex (todo 0 salvo un único 1, cuyo índice es la tarea inferida). El paper ofrece dos algoritmos:

- **One-Shot:** la tarea se infiere con **un solo gradiente**, tomando `argmaxᵢ (−∂H(p(α))/∂αᵢ)` —la coordenada en que la entropía decrece más rápido. Equivale a un paso del algoritmo de Frank-Wolfe [7], o a un paso de descenso de gradiente con re-normalización softmax y step size η → ∞. Por defecto x es una *sola imagen*, no un batch.
- **Binary:** a modo de búsqueda binaria, en log k pasos descarta en cada iteración la mitad de las tareas (las de menor decremento de entropía), poniendo sus αᵢ a 0 y re-normalizando. Útil cuando las tareas son muy parecidas (p.ej. rotaciones de 10°) y conviene usar un batch completo.

Una vez inferida la tarea, se aplica su máscara como en GG para obtener las clases. En GNs las etiquetas se comparten (el modelo solo predice la etiqueta compartida); en GNu, estrictamente más difícil, debe predecir además a qué tarea pertenecen los datos.

### 4.4. Escenario NNs: sin task ID ni en entrenamiento ni en inferencia

La inferencia de tarea habilita el caso más extremo. Si al recibir datos nuevos SupSup está **incierto** sobre la identidad de la tarea, es probable que esos datos no pertenezcan a ninguna tarea vista, así que se **asigna una máscara nueva** y se incrementa k. Operacionalmente: se computa ν = softmax(−∇α H(p(α))); si ν es aproximadamente uniforme (formalmente, si k·maxᵢ νᵢ < 1 + ε) se crea una máscara nueva; si no, se usa la máscara arg maxᵢ νᵢ. Así SupSup infiere por sí solo los **límites entre tareas** mientras aprende.

### 4.5. Extensiones: HopSupSup y neuronas superfluas

- **HopSupSup (más allá de la dependencia lineal de memoria):** almacenar k máscaras cuesta memoria lineal en k. HopSupSup las guarda implícitamente como **atractores en una red de Hopfield** [20] de tamaño fijo (usando la regla de aprendizaje de Storkey [46]). En inferencia, "Hopfield Recovery" desciende sobre la energía de Hopfield combinada con la entropía de salida hasta converger —en menos de ~30 pasos— a la máscara correcta. Para mantener la red de Hopfield de tamaño manejable, aquí se enmascaran las *salidas de cada capa* en vez de los pesos.
- **Neuronas superfluas (s-neurons):** SupSup hace clasificación de ℓk vías sin tener ℓk neuronas de salida (reutiliza las neuronas de la capa final para la tarea inferida). El paper observa que añadir neuronas extra {ℓ+1,...,s} ayuda mucho a la inferencia: la cross-entropy las empuja a valores negativos, y un objetivo alternativo **G = logsumexp(p)** (con gradientes enmascarados para las clases reales) puede usarse en lugar de la entropía, con la ventaja de que su gradiente sobre las s-neurons replica exactamente el gradiente de la pérdida supervisada de entrenamiento.

## 5. Experimentos

**Escenario GG — SplitCIFAR100 y SplitImageNet.** Siguiendo a Wen et al. [51], SplitCIFAR100 parte CIFAR100 en 20 problemas de 5 vías y SplitImageNet parte ImageNet en 100 tareas de 10 vías; se usan ResNet-18 (canales reducidos) y ResNet-50 respectivamente, con Edge-Popup [39] para las máscaras. En SplitImageNet SupSup se aproxima al *upper bound* (88.68% vs. 92.55%) con muchísimos menos bytes que los baselines, y en SplitCIFAR100 supera a métodos de tamaño similar, beneficiándose del truco de Transfer. El overhead de cómputo del producto por máscara binaria es mínimo: ~1% del tiempo de un forward pass de ResNet-50 en una 1080 Ti, menos que el overhead de BatchE.

**Escenarios GNs/GNu — PermutedMNIST, RotatedMNIST, SplitMNIST.** Con arquitecturas FC 1024-1024 y LeNet 300-100, SupSup aprende **2500 permutaciones de MNIST** —efectivamente una clasificación de 25.000 vías— infiriendo la tarea con el algoritmo One-Shot usando una **sola imagen** (Figura 4). Crucialmente, **SupSup en GNu supera a PSP y BatchE operando en el escenario estrictamente más fácil GG** (que sí reciben el task ID). Tras aprender 250 permutaciones, SupSup logra 94.91% de precisión, mientras que Online EWC y SI alcanzan apenas 33.88% y 29.31% tras solo 10 permutaciones [49]. En RotatedMNIST distingue rotaciones que difieren en solo 10° usando el algoritmo Binary con batch completo. Para HopSupSup en SplitMNIST, la máscara recuperada converge a la correcta en <30 pasos de gradiente, con precisión media de 97.43%.

**Escenario NNs — PermutedMNIST.** Sin acceso alguno al task ID (ni siquiera en entrenamiento), decidiendo cada 100 batches si asignar máscara nueva (ε = 2⁻³), SupSup aprende miles de tareas y alcanza desempeño comparable al de GNu, con una pequeña caída final al imponerse un presupuesto de 2500 máscaras.

## 6. Limitaciones reconocidas

El propio paper es explícito: la inferencia de task ID **falla cuando los modelos están mal calibrados** —cuando una máscara equivocada produce una salida *confiada* (baja entropía) sobre datos que no le corresponden, el criterio de mínima entropía la elige por error. Esto limita la aplicación a problemas no uniformes y más difíciles que las permutaciones/rotaciones de MNIST. Los autores señalan como trabajo futuro modelos mejor calibrados [14] y objetivos alternativos a la entropía —auto-supervisión [16], modelos basados en energía [13]— para escalar GN y NNs a problemas grandes. Otros límites implícitos: HopSupSup escala cuadráticamente en el tamaño de la máscara (de ahí que enmascare salidas, no pesos), y la inferencia por superposición agrega cómputo de gradiente en test time respecto a un simple forward.

## 7. Impacto

SupSup popularizó la idea de que **el olvido catastrófico se evita estructuralmente congelando los pesos** y aprendiendo solo máscaras, conectando la lottery ticket hypothesis con el aprendizaje continuo. Su aporte más citado es el **mecanismo de inferencia de tarea por minimización de entropía sobre una superposición de máscaras**, que rompió la barrera de los métodos basados en máscaras (Piggyback, PackNet, BatchE, PSP) atados a conocer el task ID. La taxonomía GG/GNs/GNu/NNs ofreció además un vocabulario común para comparar métodos de aprendizaje continuo bajo supuestos distintos. En lo práctico demostró escalabilidad a miles de tareas con almacenamiento subir-lineal (vía Hopfield) y costo de inferencia bajo, posicionándolo como referente del enfoque de "componentes específicos por tarea".

## 8. Conexión con la Clase 32 (Olvido Catastrófico)

La Clase 32 presenta SupSup en la slide homónima como la **generalización de Piggyback** que cierra la limitación más incómoda de este: Piggyback (ver [/papers/piggyback-mallya-2018](/papers/piggyback-mallya-2018)) aprende una máscara binaria por tarea sobre una red fija, pero **necesita que se le diga qué tarea está resolviendo** tanto en entrenamiento como en inferencia (escenario GG). SupSup conserva la idea —máscara binaria por tarea sobre pesos fijos— pero la lleva al caso en que **no se conoce el task ID en inferencia**, resolviéndolo con un **criterio de incerteza/confianza en la predicción**: superpone todas las máscaras, ajusta los coeficientes αᵢ con gradiente y se queda con la máscara que produce la salida de **menor entropía** (la más confiada). Ese es exactamente el mecanismo que la clase resalta como el salto cualitativo respecto a Piggyback.

Para el hilo conductor del módulo —ver [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo) y el hub de [/clases/clase-32](/clases/clase-32)— SupSup ilustra una estrategia *arquitectónica* contra el olvido catastrófico distinta de la regularización (EWC, SI) y del replay: en vez de proteger los pesos importantes o reensayar datos pasados, **se renuncia por completo a entrenar los pesos** y se confía en que una red aleatoria suficientemente grande ya contiene, vía supermáscaras, soluciones a todas las tareas. El olvido se vuelve imposible por diseño, y el problema se traslada de "no degradar lo aprendido" a "recuperar la máscara correcta en inferencia" —que es donde entra la elegante reducción a un problema de optimización por entropía.
