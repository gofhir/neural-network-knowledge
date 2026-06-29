---
title: "SupSup: Supermasks in Superposition (2020)"
weight: 363
math: true
---

{{< paper-card
    title="Supermasks in Superposition"
    authors="Mitchell Wortsman, Vivek Ramanujan, Rosanne Liu, Aniruddha Kembhavi, Mohammad Rastegari, Jason Yosinski, Ali Farhadi"
    year="2020"
    venue="NeurIPS 2020"
    pdf="/papers/supsup-wortsman-2020.pdf"
    arxiv="2006.14769" >}}
SupSup (*Supermasks in Superposition*) es un modelo de [aprendizaje continuo](/fundamentos/aprendizaje-continuo) capaz de aprender **miles de tareas en secuencia sin olvido catastrófico**. La idea de partida es simple y potente: el olvido no puede ocurrir si los pesos permanecen **fijos y aleatorios**. SupSup mantiene una red base inicializada al azar y nunca entrenada, y para cada tarea aprende solo una **supermáscara** —una subred binaria que decide qué conexiones de esa red aleatoria quedan activas—. Su novedad de mayor calado: funciona **sin task ID en inferencia**, infiriendo la tarea como un problema de optimización que minimiza la **entropía** de la salida sobre una superposición de máscaras. Es presentado en la [Clase 32](/clases/clase-32) como la generalización de [Piggyback](/papers/piggyback-mallya-2018).
{{< /paper-card >}}

---

## Contexto

El **olvido catastrófico** —entrenar una red en una tarea nueva degrada brutalmente su desempeño en las anteriores— sigue siendo el obstáculo central del aprendizaje continuo. Las familias clásicas de solución son tres: **regularización** (EWC con la matriz de Fisher, SI), que solo atenúa el olvido; **replay** (iCaRL, GEM, replay generativo), que recaptura datos pasados a costa de memoria; y **componentes específicos por tarea** (PackNet, BatchEnsemble, PSP), donde distintas partes de la red sirven a distintas tareas.

SupSup pertenece a esta tercera familia, pero su linaje teórico viene de otro lugar: la **lottery ticket hypothesis** (Frankle & Carbin, 2018), que sostiene que dentro de una red densa hay subredes ("billetes de lotería") que alcanzan la precisión de la red completa. El giro radical lo dieron Zhou et al. y Ramanujan et al.: existen **supermáscaras** sobre una red **aleatoria y nunca entrenada** tales que la subred resultante ya resuelve la tarea. El conocimiento no está en los pesos, sino en *qué subconjunto de pesos aleatorios se elige*. El número de subredes posibles es combinatorio en la cantidad de parámetros, y ese espacio gigante es lo bastante rico como para contener soluciones a tareas complejas —encontrar la máscara cuesta aproximadamente lo mismo que entrenar pesos.

## De Piggyback a SupSup

El antecedente directo es [**Piggyback**](/papers/piggyback-mallya-2018) (Mallya et al., 2018): para cada tarea nueva aprende una máscara binaria aplicada a una red *preentrenada en ImageNet*. SupSup lo generaliza en dos ejes:

1. **Reemplaza la red preentrenada por una red aleatoria fija**, lo que elimina la necesidad de almacenar los pesos: basta guardar la semilla aleatoria.
2. **Infiere el task ID cuando no se provee.** Piggyback y todos los métodos de su categoría (BatchEnsemble, PSP) están limitados al escenario en que el task ID se da tanto en entrenamiento como en inferencia. SupSup rompe esa limitación. Donde Piggyback exige que alguien le diga qué tarea está resolviendo, SupSup lo *deduce* del propio comportamiento de la red.

Esta diferencia es la columna vertebral conceptual de la Clase 32.

## Método

### Supermáscaras sobre pesos aleatorios fijos

En una clasificación estándar, una entrada $x$ se mapea a una distribución $p = f(x, W)$. SupSup escribe la salida como:

$$p = f(x,\; W \odot M)$$

donde $M$ es una máscara **binaria** y $\odot$ es el producto elemento a elemento. $W$ se mantiene **congelado en su inicialización**: bias en 0 y cada peso $\pm c$ con igual probabilidad (inicialización *signed Kaiming constant*). Solo $M$ se aprende, con el algoritmo **Edge-Popup**: cada conexión recibe un "score" real, en el forward quedan activas las conexiones con score más alto (top-$k$ por presupuesto de sparsity), y aunque la máscara no es diferenciable el gradiente se propaga a los scores mediante un *straight-through estimator*. Así se entrena la máscara dejando $W$ intacto, y el olvido se vuelve imposible por construcción.

### Inferir la tarea minimizando la entropía

Es el corazón conceptual. Llegan datos de una tarea $j$ desconocida y queremos recuperar su máscara $M^j$ entre las $k$ aprendidas. Se asocia a cada máscara un coeficiente $\alpha_i \in [0,1]$, inicializado en $1/k$ —interpretable como la "creencia" de que $M^i$ es la correcta—. La salida se computa con una **superposición ponderada** de todas las máscaras:

$$p(\alpha) = f\!\left( x,\; W \odot \textstyle\sum_i \alpha_i M^i \right)$$

La intuición: la máscara correcta produce una salida **confiada, de baja entropía**, mientras que las equivocadas producen salidas inciertas. Por tanto, se buscan los coeficientes $\alpha$ que **minimizan la entropía $H$ de $p(\alpha)$**, llevándolos a una esquina del símplex (todo 0 salvo un único 1, cuyo índice es la tarea inferida). El paper ofrece dos algoritmos:

- **One-Shot:** la tarea se infiere con **un solo paso de gradiente**, tomando $\arg\max_i\,(-\partial H / \partial \alpha_i)$ —la coordenada donde la entropía decrece más rápido—. Equivale a un paso de Frank-Wolfe. Por defecto basta *una sola imagen*.
- **Binary:** a modo de búsqueda binaria, en $\log k$ pasos descarta en cada iteración la mitad de las tareas (las de menor decremento de entropía). Útil cuando las tareas son muy parecidas (p. ej. rotaciones de 10°).

En la práctica un solo paso de gradiente suele bastar para identificar la tarea correcta **incluso entre 2500 tareas**.

### Cuando no hay task ID en ningún momento

La inferencia de tarea habilita el caso más extremo: ni en entrenamiento ni en inferencia se conoce el task ID. La clave es un **criterio de incerteza**: si al recibir datos nuevos SupSup está incierto sobre la identidad de la tarea, es probable que esos datos no pertenezcan a ninguna tarea vista, así que **se asigna una máscara nueva** y se incrementa $k$. Operacionalmente se computa $\nu = \mathrm{softmax}(-\nabla_\alpha H)$; si $\nu$ es aproximadamente uniforme (formalmente $k\cdot\max_i \nu_i < 1 + \epsilon$) se crea una máscara nueva; si no, se usa la máscara $\arg\max_i \nu_i$. Así SupSup infiere por sí solo los **límites entre tareas** mientras aprende.

### Extensiones

- **HopSupSup (memoria constante):** almacenar $k$ máscaras cuesta memoria lineal en $k$. HopSupSup las guarda implícitamente como **atractores de una red de Hopfield** de tamaño fijo; en inferencia, un descenso sobre la energía de Hopfield combinada con la entropía converge a la máscara correcta en menos de ~30 pasos.
- **Neuronas superfluas:** añadir neuronas extra a la capa de salida mejora notablemente la inferencia de tarea; un objetivo alternativo $G = \mathrm{logsumexp}(p)$ puede sustituir a la entropía.

## Taxonomía de escenarios

El paper propone un vocabulario común de cuatro escenarios según si el task ID se da en entrenamiento, si se da en inferencia y si las etiquetas se comparten:

| Escenario | Task ID en train | Task ID en test | Qué hace SupSup |
|---|---|---|---|
| **GG** | Sí | Sí | Extensión natural de Piggyback con backbone aleatorio |
| **GNs** | Sí | No (etiquetas compartidas) | Infiere la tarea por mínima entropía |
| **GNu** | Sí | No (debe predecir la tarea) | Caso estrictamente más difícil |
| **NNs** | No | No | Asigna máscaras nuevas por criterio de incerteza |

## Resultados

- **GG — SplitImageNet / SplitCIFAR100:** con ResNet-50, SupSup se aproxima al *upper bound* (88.68% vs. 92.55%) usando muchísimos menos bytes que los baselines, porque $W$ es aleatorio (basta la semilla) y las máscaras sparse se guardan comprimidas. El overhead del producto por máscara binaria es de ~1% del tiempo de un forward pass.
- **GNs/GNu — PermutedMNIST:** SupSup aprende **2500 permutaciones de MNIST** —efectivamente una clasificación de 25.000 vías— infiriendo la tarea con One-Shot usando una sola imagen. Crucialmente, **SupSup en el escenario difícil GNu supera a PSP y BatchEnsemble operando en el escenario fácil GG**. Tras 250 permutaciones logra 94.91% de precisión, mientras que Online EWC y SI caen a 33.88% y 29.31% tras solo 10 permutaciones.
- **NNs:** sin acceso alguno al task ID, SupSup aprende miles de tareas y alcanza desempeño comparable al de GNu.

## Limitaciones

El propio paper es explícito: la inferencia de task ID **falla cuando los modelos están mal calibrados** —si una máscara equivocada produce una salida *confiada* (baja entropía) sobre datos que no le corresponden, el criterio de mínima entropía la elige por error—. Esto limita la aplicación a problemas más difíciles que las permutaciones/rotaciones de MNIST. Como trabajo futuro se proponen modelos mejor calibrados y objetivos alternativos a la entropía (auto-supervisión, modelos basados en energía). Otros límites: HopSupSup escala cuadráticamente en el tamaño de la máscara, y la superposición agrega cómputo de gradiente en test time.

## Por qué importa para la Clase 32

La [Clase 32](/clases/clase-32) presenta SupSup como la **generalización de [Piggyback](/papers/piggyback-mallya-2018)** que cierra su limitación más incómoda. Piggyback aprende una máscara binaria por tarea sobre una red fija, pero necesita que se le diga qué tarea está resolviendo. SupSup conserva la idea —máscara binaria por tarea sobre pesos fijos— pero la lleva al caso en que **no se conoce el task ID en inferencia**, resolviéndolo con un **criterio de incerteza/confianza en la predicción**: superpone todas las máscaras, ajusta los coeficientes $\alpha_i$ con gradiente y se queda con la máscara de **menor entropía** (la más confiada).

Para el hilo conductor del módulo (ver [aprendizaje continuo](/fundamentos/aprendizaje-continuo)), SupSup ilustra una estrategia *arquitectónica* contra el olvido distinta de la regularización y del replay: en vez de proteger pesos importantes o reensayar datos pasados, **se renuncia por completo a entrenar los pesos** y se confía en que una red aleatoria suficientemente grande ya contiene, vía supermáscaras, soluciones a todas las tareas. El olvido se vuelve imposible por diseño, y el problema se traslada de "no degradar lo aprendido" a "recuperar la máscara correcta en inferencia" —una elegante reducción a optimización por entropía.

## Notas y enlaces

- Preprint: [arxiv.org/abs/2006.14769](https://arxiv.org/abs/2006.14769) (NeurIPS 2020, Vancouver).
- Código: [github.com/RAIVNLab/supsup](https://github.com/RAIVNLab/supsup).
- Afiliaciones: University of Washington, Allen Institute for AI, ML Collective.
