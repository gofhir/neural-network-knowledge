# Learning to Prompt for Continual Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Learning to Prompt for Continual Learning* (abreviado **L2P**).
- **Autores:** Zifeng Wang (Northeastern University; trabajo realizado durante una pasantía en Google Cloud AI Research), Zizhao Zhang, Chen-Yu Lee, Ruoxi Sun, Xiaoqi Ren, Tomas Pfister (Google Cloud AI), Han Zhang, Guolong Su, Vincent Perot (Google Research) y Jennifer Dy (Northeastern University).
- **Venue:** CVPR 2022 (Conference on Computer Vision and Pattern Recognition).
- **Año:** 2022. **Preprint:** arXiv:2112.08654v2 (21 mar 2022), [arxiv.org/abs/2112.08654](https://arxiv.org/abs/2112.08654).
- **Código:** [github.com/google-research/l2p](https://github.com/google-research/l2p) (implementación oficial en JAX/Flax).

L2P es un paper bisagra: marca el punto donde el **aprendizaje continuo** (continual learning) deja de pensarse como un problema de *cómo proteger los pesos de un modelo que se reentrena* y empieza a pensarse como un problema de *cómo recuperar y componer conocimiento almacenado fuera de los pesos*. Su tesis, en una frase: si partimos de un Transformer pre-entrenado potente y lo dejamos **congelado**, podemos almacenar el conocimiento específico de cada tarea en un pequeño conjunto de vectores aprendibles —los *prompts*— en vez de modificar el backbone o guardar ejemplos del pasado.

El problema que ataca es el **olvido catastrófico** (catastrophic forgetting): cuando un modelo entrena secuencialmente sobre tareas no estacionarias, al ajustar sus pesos para la tarea actual destruye el conocimiento de las tareas previas. El paradigma dominante hasta ese momento atacaba el olvido de dos formas, ambas con limitaciones críticas. Los **métodos basados en rehearsal** (repaso) guardan un *buffer* de ejemplos de tareas pasadas para reentrenar sobre ellos —pero se degradan cuando el buffer es pequeño y son inaplicables cuando la privacidad de los datos prohíbe almacenarlos. Los **métodos que asumen identidad de tarea conocida en test** (task-incremental) adjuntan módulos específicos por tarea —pero conocer la identidad de la tarea en inferencia es una suposición poco realista.

L2P propone un tercer camino: un **pool de prompts** (memoria key-value de vectores aprendibles), un mecanismo de **selección instance-wise** basado en query-key que escoge qué prompts usar para cada entrada *sin conocer la identidad de la tarea*, y la concatenación de esos prompts como tokens al inicio de la secuencia que entra al Transformer congelado. Solo se entrenan los prompts, sus claves y el clasificador (~0.1% de parámetros extra). El resultado es contundente: L2P supera a los métodos rehearsal incluso **sin buffer de memoria**, y funciona en el escenario class-incremental sin task ID, e incluso en el escenario task-agnostic (sin fronteras de tarea).

Para la Clase 32 esto importa porque **L2P es el método con que cierra la clase**: representa el estado del arte moderno del aprendizaje continuo, donde la sinergia entre Transformers pre-entrenados y prompting reformula un problema que durante décadas se atacaba modificando pesos.

## 2. Contexto histórico: olvido catastrófico en la era de los Transformers pre-entrenados

El olvido catastrófico se documenta desde McCloskey & Cohen (1989): las redes neuronales conexionistas, al aprender una tarea nueva, sobrescriben las representaciones de las anteriores. Durante décadas el aprendizaje continuo consolidó **tres familias** de soluciones, que el paper resume con precisión:

1. **Métodos basados en regularización** (EWC, LwF, Synaptic Intelligence, Memory Aware Synapses). Limitan la plasticidad del modelo penalizando cambios en los parámetros importantes para tareas previas —típicamente bajando la tasa de aprendizaje efectiva en esos pesos. Atacan el olvido sin guardar ejemplos, pero no rinden bien en escenarios desafiantes ni con datasets complejos.

2. **Métodos basados en rehearsal/repaso** (ER, GDumb, BiC, DER++, Co²L, iCaRL). Construyen un *data buffer* con muestras de tareas viejas y las mezclan con la tarea actual al entrenar. Idea simple pero efectiva: alcanzaban el SOTA en muchos benchmarks. Sus dos talones de Aquiles: la performance se deteriora con buffers pequeños, y el almacenamiento de datos es inviable cuando la privacidad importa (escenarios clínicos, financieros).

3. **Métodos basados en arquitectura** (Progressive Networks, PackNet, SupSup, DualNet, Learn-to-Grow). Asignan componentes separados —subredes, máscaras o ramas expandidas— a cada tarea. La mayoría requiere identidad de tarea en test para condicionar la red, lo que los excluye de los escenarios class-incremental y task-agnostic; y a menudo añaden tantos parámetros como el modelo completo.

El cambio de era que habilita L2P es doble. Primero, la **disponibilidad de Transformers pre-entrenados potentes** (ViT-B/16 pre-entrenado en ImageNet se vuelve "un activo común" en visión), que ofrecen representaciones genéricas tan buenas que ya no hace falta reajustar el backbone. Segundo, la maduración del **prompt-based learning** en NLP. Técnicas como Prompt Tuning (Lester et al., 2021), Prefix Tuning (Li & Liang, 2021) y AutoPrompt (Shin et al., 2020) mostraron que, en vez de adaptar los pesos de un modelo de lenguaje congelado a una tarea, basta diseñar *prompts* —tokens templados o aprendibles, prepended a la entrada— que "instruyen" al modelo para resolver la tarea condicionalmente. Un prompt captura conocimiento específico de tarea con muchísimos menos parámetros que Adapters o LoRA.

La conexión conceptual del paper con la **teoría de Sistemas de Aprendizaje Complementarios (CLS)** es elegante: el cerebro logra aprendizaje continuo gracias a dos sistemas —el hipocampo (aprendizaje rápido, memoria episódica) y el neocórtex (memoria de largo plazo). El paper plantea explícitamente que el *pool de prompts* hace de aprendizaje rápido y el *backbone congelado* de memoria de largo plazo. El abstract resume la ambición: construir "un sistema de memoria más sucinto sin acceder a la identidad de tarea en test". Y un dato llamativo para dimensionar la economía del método: su mayor espacio de prompts es **más pequeño que una sola imagen de 224×224**.

## 3. El obstáculo: por qué prompting no se aplica trivialmente al aprendizaje continuo

Antes de presentar su solución, el paper explica con cuidado por qué el prompting de NLP no se traslada directamente —es la motivación que justifica todo el diseño posterior. Hay dos formas ingenuas de usar prompts en aprendizaje continuo, y ambas fallan:

1. **Un prompt distinto por tarea.** Si entrenamos prompts independientes para cada tarea, en test seguimos necesitando saber a qué tarea pertenece la entrada para escoger el prompt correcto. Vuelve el requisito de identidad de tarea que queríamos eliminar. Además, prompts independientes impiden compartir conocimiento entre tareas similares.

2. **Un único prompt compartido para todas las tareas.** Habilita compartir conocimiento, pero como ese prompt se reentrena secuencialmente, sufre el **olvido catastrófico** igual que cualquier parámetro reajustado (el paper lo confirma empíricamente en su ablación, Sección 5.4).

El objetivo ideal que el paper articula: aprender un sistema que **comparta conocimiento cuando las tareas son similares y lo mantenga independiente en caso contrario**. Esa tensión —compartir vs. aislar conocimiento, decidida automáticamente y por instancia— es exactamente lo que el pool de prompts con selección query-key resuelve.

## 4. Contribución central: el pool de prompts

L2P introduce un **espacio de memoria key-value llamado prompt pool**, optimizado conjuntamente con la pérdida supervisada, que decopla explícitamente conocimiento task-invariant (compartido) y task-specific (aislado). El pool se define como:

$$P = \{P_1, P_2, \cdots, P_M\}, \quad M = \text{número total de prompts}$$

donde cada $P_j \in \mathbb{R}^{L_p \times D}$ es un prompt individual con longitud de token $L_p$ y la misma dimensión de embedding $D$ que las features de la entrada. Para una entrada $x$ con embedding $x_e = f_e(x)$, se seleccionan $N$ índices $\{s_i\}_{i=1}^N$ del pool y se adapta la entrada concatenando los prompts elegidos al inicio:

$$x_p = [P_{s_1}; \cdots; P_{s_N}; x_e], \quad 1 \le N \le M$$

donde `;` es concatenación a lo largo de la dimensión de tokens. La idea clave es que **los prompts son libres de componerse**: pueden codificar conjuntamente conocimiento (features visuales, información de tarea) que el modelo procesa. Esto habilita un esquema de compartición de conocimiento de grano fino a nivel de instancia: entradas similares tienden a compartir más prompts, y viceversa.

Las tres contribuciones que el paper enumera: (1) un marco novedoso de aprendizaje continuo basado en prompts, con un mecanismo —el pool de prompts como espacio de memoria— que sirve de "instrucciones parametrizadas" para un modelo pre-entrenado, aplicable al exigente escenario task-agnostic; (2) experimentos comprehensivos en escenarios class-, domain- y task-incremental donde L2P supera consistentemente al SOTA previo, incluso **sin buffer de rehearsal**; (3) ser, hasta donde los autores saben, **los primeros en introducir prompting en el campo del aprendizaje continuo**.

## 5. Método: selección query-key, pérdida de matching y predicción

### 5.1. Backbone congelado y la query como extractor de features

El backbone es un ViT $f = f_r \circ f_e$ (sin la cabeza de clasificación), con $f_e$ la capa de embedding y $f_r$ el stack de capas de self-attention. La imagen se reformatea en una secuencia de parches; el primer token es el `[class]` token. **El backbone se mantiene congelado** durante todo el aprendizaje continuo, preservando su generalidad.

La query se obtiene usando el propio modelo pre-entrenado **como extractor de features congelado**: la función de query $q(x) = f(x)[0,:]$ toma el vector de features correspondiente al token `[class]`. Crucialmente, $q$ es **determinista respecto a las tareas y no tiene parámetros aprendibles** —es lo que permite que el mecanismo funcione sin conocer la identidad de la tarea. (El paper nota que otros extractores, como una ConvNet, también son viables.)

### 5.2. Selección instance-wise vía query-key

Cada prompt $P_i$ se asocia, como *valor*, a una **clave aprendible** $k_i \in \mathbb{R}^{D_k}$:

$$\{(k_1, P_1), (k_2, P_2), \cdots, (k_M, P_M)\}, \quad K = \{k_i\}_{i=1}^M$$

La idea es dejar que **la propia instancia decida qué prompts elegir** mediante el matching query-key. Definiendo $\gamma$ como función que puntúa el match entre query y clave (los autores encuentran que la **distancia coseno** funciona bien), para una entrada $x$ se buscan las top-$N$ claves resolviendo:

$$K_x = \underset{\{s_i\}_{i=1}^N \subseteq [1,M]}{\arg\min} \sum_{i=1}^N \gamma(q(x), k_{s_i})$$

Este diseño key-value **decopla el aprendizaje del mecanismo de query del aprendizaje de los prompts** —algo que la ablación demuestra ser crítico. Y como la consulta es **instance-wise**, todo el marco es task-agnostic: funciona sin fronteras claras de tarea en entrenamiento ni identidad de tarea en test.

### 5.3. Diversificación opcional de la selección

Aunque L2P no necesita información de fronteras de tarea, en la práctica las transiciones suelen ser discretas y las fronteras se conocen en entrenamiento. Para tareas muy diversas, el paper añade un mecanismo *opcional*: una **tabla de frecuencia de prompts** $H_t = [h_1, \cdots, h_M]$ que registra cuán frecuentemente se ha seleccionado cada prompt hasta la tarea $t-1$. La selección se modifica a:

$$K_x = \underset{\{s_i\}_{i=1}^N \subseteq [1,M]}{\arg\min} \sum_{i=1}^N \gamma(q(x), k_{s_i}) \cdot h_{s_i}$$

donde $h_{s_i}$ penaliza los prompts usados con frecuencia, fomentando una selección diversificada que reduce la compartición innecesaria de conocimiento entre tareas no relacionadas. Esta ecuación solo se aplica en entrenamiento; en test se usa la selección estándar.

### 5.4. Objetivo de optimización y predicción

Tras seleccionar los $N$ prompts, el embedding adaptado $x_p$ se alimenta al resto del modelo congelado $f_r$ y al clasificador $g_\phi$. La pérdida end-to-end minimiza:

$$\min_{P, K, \phi} \; \mathcal{L}\big(g_\phi(f_r^{avg}(x_p)), y\big) + \lambda \sum_{K_x} \gamma(q(x), k_{s_i})$$

con dos términos. El **primero** es la cross-entropy softmax estándar de clasificación. El **segundo** es una *pérdida surrogate de matching* que tira las claves seleccionadas más cerca de las features de la query correspondientes —es lo que entrena las claves para que el mecanismo de selección aprenda a apuntar a los prompts correctos. $\lambda$ pesa ambos términos (los autores reportan $\lambda = 0.5$, poco sensible).

La predicción usa $f_r^{avg} = \text{AvgPool}(f_r(x_p)[0:N L_p, :])$: es decir, se **promedian los vectores ocultos de salida correspondientes a las $N \cdot L_p$ posiciones de prompt** antes de pasar por el clasificador. Notablemente, solo se actualizan $P$, $K$ y $\phi$ —el backbone permanece intacto.

El Algoritmo 1 del paper detalla el bucle: para cada tarea $t$, cada época y cada mini-batch, por muestra se calcula la query, se buscan las top-$N$ claves, se seleccionan los prompts asociados, se prepende al embedding, se calcula la pérdida por muestra, y al final del batch se actualizan claves, prompts y clasificador por descenso de gradiente (Adam).

## 6. Experimentos

L2P se evalúa sobre un ViT-B/16 pre-entrenado en cuatro benchmarks que cubren los tres escenarios:

- **Split CIFAR-100** (class-incremental): CIFAR-100 dividido en 10 tareas de 10 clases disjuntas cada una. Tareas con cierta similitud (comparten superclases).
- **5-datasets** (class-incremental): secuencia de CIFAR-10, MNIST, Fashion-MNIST, SVHN y notMNIST. Tareas muy diversas, donde el modelo es susceptible al olvido.
- **CORe50** (domain-incremental): 50 objetos en 11 dominios distintos; mismas clases, distinta distribución de entrada por dominio.
- **Gaussian scheduled CIFAR-100** (task-agnostic): la distribución de datos cambia gradualmente sin fronteras explícitas de tarea.

Las métricas son **Average Accuracy** (mayor es mejor) y **Forgetting** (menor es mejor). Se compara contra baselines (fine-tuning secuencial, EWC, LwF) y métodos rehearsal SOTA (ER, GDumb, BiC, DER++, Co²L) y arquitectura (SupSup, DualNet). Se reportan dos variantes propias: **L2P** (sin buffer) y **L2P-R** (L2P con buffer, para comparación justa con métodos rehearsal). Hiperparámetros: $M=10, N=5, L_p=5$ para CIFAR-100 y CORe50; $M=20, N=4, L_p=5$ para 5-datasets. Los prompts añaden solo 46.080 y 92.160 parámetros (0.05% y 0.11% de incremento).

**Resultados class-incremental (Tabla 1).** Sin buffer, L2P logra **83.83%** de accuracy y solo **7.63%** de forgetting en Split CIFAR-100 —frente a 60.69%/27.77% de LwF y 47.01%/33.27% de EWC. Más impactante aún: L2P **sin buffer supera a casi todos los métodos rehearsal con buffer pequeño**. Con buffer, L2P-R alcanza 84.21–86.31% según el tamaño, contra ~67–83% de los rehearsal. En 5-datasets, L2P sin buffer logra 81.14% accuracy y 4.64% forgetting, superando ampliamente a los rehearsal. El upper-bound (entrenamiento i.i.d. sobre todas las tareas) es 90.85%, así que L2P cierra gran parte de la brecha.

**Vs. métodos de arquitectura (Tabla 2).** Medido por distancia al upper-bound de su backbone, L2P tiene un *gap* de 7.02 puntos, frente a 40.27 de DualNet y 52.07 de SupSup —una mejora enorme.

**Domain-incremental, CORe50 (Tabla 3).** L2P logra 78.33% sin buffer (vs. 75.45% de LwF) y L2P-R llega a 81.07%, el mejor de todos.

**Task-agnostic, Gaussian CIFAR-100 (Tabla 4).** El escenario más desafiante. L2P **sin buffer alcanza 88.34%**, superando incluso a métodos rehearsal con buffer (DER++ 85.24%). Los autores hipotetizan que la transición suave de tareas ayuda a L2P a consolidar conocimiento en los prompts.

**Ablaciones (Tabla 5, en 5-datasets).** Cada componente importa: quitar el pool de prompts (usar un solo prompt) hunde la accuracy a 51.96% con 26.60% de forgetting —confirma que un prompt único sufre olvido severo. Quitar la clave aprendible (usar la media de prompts como clave) baja a 58.33%. Quitar la diversificación baja a 62.26%. El modelo completo: 81.14%. Los **histogramas de selección** muestran el comportamiento esperado: en Split CIFAR-100 las tareas comparten prompts (alta similitud intra-tarea), mientras en 5-datasets cada tarea favorece prompts específicos.

**Hiperparámetros (Figura 4).** Un $L_p$ muy pequeño perjudica; uno excesivo causa underfitting del conocimiento. Aumentar el tamaño del pool $M$ ayuda más en tareas diversas (5-datasets) que en similares (CIFAR-100).

## 7. Limitaciones reconocidas

El paper es explícito en su sección B sobre los límites:

- **Demostrado solo en visión.** Aunque el método no asume modalidad alguna, los experimentos son todos de clasificación de imágenes. Generalizar a otras modalidades queda como trabajo futuro.
- **Requiere un modelo pre-entrenado basado en secuencias.** L2P asume un Transformer pre-entrenado. Cómo generalizar el marco a otras arquitecturas de visión (p.ej. ConvNets) es una dirección abierta y no trivial, porque el mecanismo de prepender tokens es nativo de los modelos de secuencia.
- **El benchmark task-agnostic es sintético.** El Gaussian scheduled CIFAR-100, pese a ser el escenario más cercano al mundo real, es artificial. El paper pide benchmarks más complejos para evaluar de verdad el aprendizaje continuo task-agnostic.
- **Sesgos heredados (sección A).** Como L2P toma un modelo pre-entrenado congelado, cualquier sesgo o problema de fairness del backbone se arrastra al proceso continuo. Los autores recomiendan auditar el modelo base y testear robustez frente a ataques adversariales, sobre todo en aplicaciones de seguridad crítica.

## 8. Impacto: el nacimiento de la era prompt-based del aprendizaje continuo

L2P **inauguró una línea de investigación completa**. Demostrar que un backbone congelado + prompts aprendibles podía batir a métodos rehearsal *sin guardar datos* reformuló el campo y disparó una familia de métodos prompt-based que se volvió dominante:

- **DualPrompt** (Wang et al., ECCV 2022), de los mismos autores, separa los prompts en *G-Prompts* (general, conocimiento compartido) y *E-Prompts* (expert, conocimiento específico de tarea), insertándolos en distintas capas del Transformer —refinando la dicotomía task-invariant/task-specific que L2P introdujo.
- **CODA-Prompt** (Smith et al., CVPR 2023) reemplaza la selección discreta top-$N$ por una **combinación ponderada y diferenciable** de componentes de prompt vía atención, eliminando la no-diferenciabilidad del argmin de L2P y mejorando la plasticidad.
- La línea conecta con el auge del **parameter-efficient fine-tuning** (PEFT): L2P es, en esencia, PEFT aplicado al setting continuo, donde la eficiencia de parámetros (~0.1%) no es solo ahorro de cómputo sino el mecanismo mismo que evita el olvido (al no tocar el backbone).

El método sigue siendo una referencia obligada y un baseline estándar en cualquier paper de aprendizaje continuo con modelos pre-entrenados posterior a 2022.

## 9. Conexión con la Clase 32 (Olvido Catastrófico)

La Clase 32 recorre el problema del olvido catastrófico y sus soluciones clásicas —regularización (EWC, SI), rehearsal (replay, iCaRL), arquitectura (Progressive Networks, PackNet)— y **cierra con L2P como el método moderno** que sintetiza dos hilos del curso: el prompting (visto en la Clase 20, donde aprendimos cómo los prompts aprendibles instruyen a modelos de lenguaje congelados) y los Transformers (la arquitectura backbone). L2P es la respuesta de la era de los modelos pre-entrenados a un problema que durante décadas se atacó modificando pesos.

El flujo que las slides de L2P presentan es exactamente el de la Sección 5:

1. **Query**: la entrada pasa por el backbone congelado, que extrae un vector de features ($q(x)$ del token `[class]`).
2. **Selección query-key**: ese vector se compara (distancia coseno) contra las claves aprendibles del pool y se eligen los top-$N$ prompts —*instance-wise*, sin saber la tarea.
3. **Prompts → tokens**: los prompts seleccionados se concatenan al inicio de la secuencia de embeddings.
4. **Predicción**: el Transformer congelado procesa la secuencia extendida, se promedian los tokens de prompt y el clasificador predice.

Y los resultados que la clase destaca son los tres titulares del paper: **mejor exactitud** (supera al SOTA en los cuatro benchmarks), **menor olvido** (forgetting de un dígito frente a decenas en los baselines) y **sin task ID** (funciona en class-incremental y hasta en task-agnostic). El contraste pedagógico con los métodos clásicos es la lección central: donde EWC penaliza el cambio de pesos y el replay guarda ejemplos, L2P simplemente *no toca el backbone* y guarda el conocimiento en prompts diminutos recuperables por contenido. Esto profundiza lo visto en el fundamento de [aprendizaje continuo](/fundamentos/aprendizaje-continuo) y se integra en la [Clase 32](/clases/clase-32) como el broche moderno del tema.
