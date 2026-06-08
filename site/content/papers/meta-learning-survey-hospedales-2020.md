---
title: "Meta-Learning Survey (Hospedales)"
weight: 266
math: true
---

{{< paper-card
    title="Meta-Learning in Neural Networks: A Survey"
    authors="Timothy Hospedales, Antreas Antoniou, Paul Micaelli, Amos Storkey"
    year="2020"
    venue="IEEE TPAMI 2021"
    pdf="/papers/meta-learning-survey-hospedales-2020.pdf"
    arxiv="2004.05439" >}}
{{< /paper-card >}}

## Por qué este survey es la referencia canónica

Hacia 2020 el meta-aprendizaje en redes neuronales había explotado tras MAML (Finn et al., 2017), Matching Networks (Vinyals et al., 2016) y Prototypical Networks (Snell et al., 2017), pero el campo carecía de un vocabulario unificado: distintas comunidades usaban "meta-learning" de formas incompatibles (selección de algoritmo en data mining, AutoML, el *learning-to-learn* de Thrun y Pratt). El survey de Hospedales et al. se volvió la referencia canónica por tres razones.

Primero, llega en el momento exacto en que el campo necesitaba ordenarse. Segundo, propone una **definición formal unificada** del meta-learning como **optimización de dos niveles** (*bilevel optimization*) que reconcilia las visiones dispersas. Tercero, ofrece una **taxonomía de tres ejes** (meta-representation, meta-optimizer, meta-objective) que funciona como un espacio de diseño: cualquier método existente o futuro se ubica como un punto en ese espacio, y cualquier combinación no explorada de los ejes sugiere un método potencialmente nuevo.

Como es un survey, su valor no está en un resultado experimental sino en el **mapa conceptual**. Es el documento que un practicante lee para orientarse: entender qué se ha hecho, cómo se relaciona con campos vecinos (transfer learning, AutoML, optimización de hiperparámetros) y dónde están las fronteras abiertas. La tesis central se resume así: el deep learning convencional aprende un modelo *desde cero* con un algoritmo de aprendizaje *fijo y diseñado a mano*; el meta-learning, en cambio, **aprende el algoritmo de aprendizaje mismo** a partir de la experiencia de múltiples episodios. Es la siguiente capa de *joint learning*: el deep learning unió aprendizaje de features y de modelo; el meta-learning aspira a unir features, modelo y **algoritmo**.

El campo no es nuevo. El survey traza su genealogía hasta Schmidhuber (1987) con métodos *self-referential* (redes que reciben sus propios pesos y predicen actualizaciones), Bengio et al. (1990) con reglas de aprendizaje biológicamente plausibles, y Thrun y Pratt (1998), que definieron *learning to learn* como lo que ocurre cuando el rendimiento de un aprendiz mejora *con el número de tareas vistas* (no solo con más datos de una sola tarea). Esta perspectiva ve el meta-learning como herramienta para gestionar el teorema **"no free lunch"** de Wolpert: buscar el *inductive bias* mejor adaptado a una familia de problemas.

## La definición formal: meta-learning como bilevel optimization

En aprendizaje supervisado clásico, dado un dataset $\mathcal{D}=\{(x_1,y_1),\dots,(x_N,y_N)\}$, se entrena un modelo $\hat{y}=f_\theta(x)$ resolviendo:

$$\theta^* = \arg\min_\theta \mathcal{L}(\mathcal{D};\theta,\omega)$$

La clave es el **condicionamiento sobre $\omega$**: $\omega$ codifica los supuestos sobre *cómo aprender* (optimizador, clase de funciones para $f$, regularización, inicialización). La asunción convencional es doble: (1) la optimización se hace *desde cero* para cada problema, y (2) $\omega$ está *pre-especificado*. El meta-learning ataca la segunda asunción: en lugar de fijar $\omega$ a mano, lo **aprende**.

El survey formaliza el meta-entrenamiento como un problema de **optimización de dos niveles**, un concepto que viene de la teoría de Stackelberg en economía. Las dos ecuaciones centrales del paper son:

$$\omega^* = \arg\min_\omega \sum_{i=1}^M \mathcal{L}^{meta}\!\left(\theta^{*(i)}(\omega),\,\omega,\,\mathcal{D}^{val(i)}_{source}\right)$$

sujeto a:

$$\theta^{*(i)}(\omega) = \arg\min_\theta \mathcal{L}^{task}\!\left(\theta,\,\omega,\,\mathcal{D}^{train(i)}_{source}\right)$$

Los componentes:

- **$\theta$ (base learner)**: parámetros del modelo que resuelve la tarea concreta (por ejemplo, un clasificador de imágenes). Se optimizan en el **nivel interno (inner loop)** sobre el train de cada tarea.
- **$\omega$ (meta-knowledge)**: el "cómo aprender". Es lo que se comparte entre tareas y lo que el meta-learning optimiza en el **nivel externo (outer loop)**. Puede ser una inicialización, un optimizador, hiperparámetros, una métrica, una pérdida, una arquitectura.
- **$\mathcal{L}^{task}$ (objetivo interno)**: la pérdida de la tarea, por ejemplo cross-entropy sobre el support set.
- **$\mathcal{L}^{meta}$ (objetivo externo)**: la pérdida meta, evaluada sobre el conjunto de validación tras adaptar el modelo. Mide si $\omega$ produjo modelos que generalizan bien.

La estructura crítica es la **asimetría líder-seguidor (leader-follower)**: el nivel interno está *condicionado* a la estrategia de aprendizaje $\omega$ que define el nivel externo, pero **no puede cambiar $\omega$** durante su entrenamiento. $\omega$ es el líder de Stackelberg, $\theta$ es el seguidor.

El survey es honesto sobre el alcance: la imagen bilevel es "estrictamente precisa solo para los métodos basados en optimización" (como MAML), pero sirve para visualizar la mecánica del meta-learning en general. Para los métodos que sintetizan modelos en una pasada feed-forward, se habla de modelos **amortizados** (*amortized*): el costo de aprender una tarea nueva se reduce a una pasada por una función $g_\omega(\cdot)$, porque la optimización iterativa "ya se pagó" durante el meta-entrenamiento de $\omega$.

## Las dos vistas: task-distribution y bilevel

El survey presenta el meta-learning desde dos perspectivas complementarias.

**Vista de distribución de tareas (task-distribution view).** El objetivo es aprender un algoritmo de aprendizaje de propósito general que **generalice a través de tareas**, idealmente de modo que cada tarea nueva se aprenda mejor que la anterior. Se define una tarea de forma laxa como un par $\mathcal{T}=\{\mathcal{D},\mathcal{L}\}$ y se evalúa $\omega$ sobre una distribución $p(\mathcal{T})$:

$$\min_\omega \mathbb{E}_{\mathcal{T}\sim p(\mathcal{T})}\, \mathcal{L}(\mathcal{D};\omega)$$

En la práctica se accede a $M$ **tareas fuente (source tasks)** para el **meta-training**, cada una con datos de entrenamiento y validación. En la jerga del few-shot estos se llaman **support set** y **query set**. Luego se tienen $Q$ **tareas objetivo (target tasks)** para el **meta-testing**, donde se usa el meta-conocimiento aprendido $\omega^*$ para entrenar el modelo base en cada tarea nunca vista. Esta vista trae las analogías de **meta-underfitting** y **meta-overfitting**: este último es el problema por el cual el meta-conocimiento aprendido sobre las tareas fuente *no generaliza* a las objetivo, común cuando hay pocas tareas fuente.

**Vista bilevel.** La vista de distribución describe el *flujo* pero no dice *cómo* resolver el paso de meta-training. La vista bilevel responde el "cómo": castea el meta-training como la optimización jerárquica de dos niveles de la sección anterior.

El survey enfatiza además que la distribución sobre tareas **no es una condición necesaria**. En el **caso de tarea única** ($M=Q=1$) se parte el train set para obtener validación y se aprende $\omega$ sobre varios episodios con distintos splits train-val. Esto cubre, por ejemplo, la optimización de hiperparámetros de tarea única.

## Posicionamiento vs transfer learning, AutoML, HPO, continual learning

Una de las contribuciones más útiles del survey es desambiguar el meta-learning de campos con los que se confunde. El criterio discriminante recurrente es: **¿hay un meta-objetivo explícito optimizado end-to-end?** Si no lo hay, no es meta-learning en el sentido del paper.

- **Transfer Learning (TL)**: usa experiencia de una tarea fuente para mejorar el aprendizaje en una objetivo, vía transferencia de parámetros + fine-tuning. La diferencia: en TL el prior se extrae por aprendizaje *vanilla* sobre la fuente, *sin* meta-objetivo. En meta-learning, el prior se define por una optimización externa que *evalúa el beneficio del prior cuando se aprende una tarea nueva*.
- **Domain Adaptation (DA) y Domain Generalization (DG)**: atacan el *domain-shift* (mismo objetivo, distinta distribución de entrada). DA y DG vanilla no usan meta-objetivo, pero el meta-learning *puede usarse* para hacer ambos simulando el shift entre train y validación.
- **Continual Learning (CL)**: aprende sobre una secuencia de tareas no estacionaria sin olvidar las viejas. Comparte la noción de distribución de tareas, pero la mayoría de métodos de CL *no resuelven explícitamente* un meta-objetivo.
- **Multi-Task Learning (MTL)**: aprende conjuntamente varias tareas conocidas vía regularización por compartir parámetros. Es optimización de un solo nivel, sin meta-objetivo, y apunta a un *número fijo de tareas conocidas* (no a tareas futuras no vistas).
- **Hyperparameter Optimization (HO)**: cae *dentro* del meta-learning cuando define un meta-objetivo entrenado end-to-end (HO basado en gradiente, NAS). El survey *excluye* random search y Bayesian Optimization.
- **Hierarchical Bayesian Models (HBM)**: dan una vista de *modelado* (no algorítmica) del meta-learning, con un prior $p(\theta\mid\omega)$ y un prior sobre $\omega$. MAML puede leerse así.
- **AutoML**: paraguas amplio para automatizar partes del pipeline. El survey concluye que **el meta-learning puede verse como una especialización de AutoML**.

## La taxonomía de 3 ejes (meta-representation / meta-optimizer / meta-objective)

Las taxonomías previas dividían el meta-learning en tres familias: **optimization-based** (el inner loop se resuelve como optimización, ej. MAML), **model-based / black-box** (el inner learning está envuelto en la pasada feed-forward de un modelo, ej. RNNs, memory-augmented networks) y **metric-based / non-parametric** (comparación de puntos en el nivel interno, ej. Matching, Prototypical, Relation). El survey argumenta que esa división *no expone todas las facetas de interés* y propone una descomposición en **tres ejes independientes** que forman un espacio de diseño. La representación del modelo base $\theta$ **no** se incluye, porque depende de la aplicación.

**Eje 1 — Meta-Representation ("What?"): qué se meta-aprende.** Es la elección de $\omega$. El catálogo incluye: *Parameter Initialization* (MAML; $\omega=\theta_0$), *Optimizer* (aprender el paso de actualización, ej. "learning to learn by gradient descent"), *Feed-Forward Models / black-box / amortized* (una red mapea el support set a los parámetros del clasificador; incluye memory-augmented networks e hypernetworks), *Embedding Functions / metric learning* (redes de embedding para comparación por similitud), *Losses and Auxiliary Tasks*, *Architectures / NAS* (ej. DARTS), *Attention Modules*, *Modules*, *Hyperparameters*, *Data Augmentation*, *Sample Weights / Curriculum*, y representaciones **transductivas** donde $\omega$ literalmente son datos (*dataset distillation*, labels, parámetros de simulador para sim2real). El paper añade tres distinciones transversales: transductivas vs no, simbólicas vs sub-simbólicas, y grado de amortización.

**Eje 2 — Meta-Optimizer ("How?"): cómo se optimiza el nivel externo.**

- **Gradient**: descenso sobre $\omega$ por regla de la cadena, $\frac{d\mathcal{L}^{meta}}{d\omega} = \frac{d\mathcal{L}^{meta}}{d\theta}\cdot\frac{d\theta}{d\omega}$. Lo más eficiente, pero enfrenta diferenciar a través de muchos pasos internos (gradientes de segundo orden, *implicit differentiation*), degradación del gradiente y operaciones no diferenciables.
- **Reinforcement Learning**: cuando el base learner o el meta-objetivo son no diferenciables, se estima $\nabla_\omega\mathcal{L}^{meta}$ vía policy gradient. Alivia la diferenciabilidad pero es de alta varianza y muy costoso.
- **Evolution (EA)**: optimiza cualquier modelo sin restricción de diferenciabilidad, no sufre degradación de gradiente y es paralelizable, pero el tamaño de población crece con los parámetros y su ajuste es inferior al gradiente para modelos grandes.

**Eje 3 — Meta-Objective ("Why?"): para qué se meta-aprende.** Determinado por $\mathcal{L}^{meta}$, la distribución $p(\mathcal{T})$ y el flujo de datos entre niveles. Opciones de diseño: *many vs few-shot*, *fast adaptation vs asymptotic performance* (según si la validación se computa al final o como suma tras cada paso interno), *multi vs single-task*, *online vs offline*, y objetivos como simular domain-shift, robustez a label noise o compresión. La Tabla 1 del paper cruza meta-representation × meta-optimizer y colorea por meta-objetivo, ubicando cada paper relevante en una celda.

## Dónde caen MAML, Matching/Prototypical y memory-augmented en la taxonomía

El punto pedagógico clave: tres métodos que parecen muy distintos son **tres elecciones del mismo eje 1 (meta-representation)** combinadas con el mismo eje 2 (gradient) y el mismo eje 3 (few-shot). El survey transforma una lista heterogénea en un espacio ordenado.

- **MAML (Finn et al., 2017)** — taxonomía previa: *optimization-based*. En tres ejes: meta-representation = **Parameter Initialization** ($\omega=\theta_0$); meta-optimizer = **Gradient** (diferenciando a través de los updates del base learner, con gradientes de segundo orden); meta-objective = **few-shot** *sample-efficient*. La idea: aprender una inicialización tal que pocos pasos internos produzcan un clasificador que rinde bien en validación. Amortización *limitada*.

- **Matching y Prototypical Networks** — taxonomía previa: *metric-based / non-parametric*. En tres ejes: meta-representation = **Embedding Functions (Metric)** ($\omega$ es la red de embedding); meta-optimizer = **Gradient**; meta-objective = **few-shot**. El survey las reconcilia mostrando que metric learning es un **caso especial de feed-forward model**: el embedding del support genera implícitamente los "pesos" (prototipos / similitudes) que interpretan el query, equivalente a una hypernetwork que sintetiza un clasificador lineal.

- **Memory-Augmented Neural Networks (MANN)** — taxonomía previa: *model-based / black-box*. En tres ejes: meta-representation = **Feed-Forward Model / Black-Box** (con memoria explícita); meta-optimizer = **Gradient**; meta-objective = **few-shot**. Ventaja: optimización más simple sin gradientes de segundo orden. Desventaja observada: generalizan peor a tareas fuera de distribución y son asintóticamente más débiles que los métodos basados en optimización.

## Challenges abiertos

- **Distribuciones de tareas diversas y multi-modales**: muchos éxitos han sido en familias *estrechas*. MAML asume implícitamente que $p(\mathcal{T})$ es uni-modal y que un solo $\omega$ sirve para todas. Las distribuciones reales (imágenes médicas vs satelitales vs cotidianas) son multi-modales, con gradientes en conflicto entre tareas.
- **Meta-generalización**: generalizar *a través de tareas*. Dos sub-retos: (i) de meta-train a tareas meta-test nuevas de la misma $p(\mathcal{T})$, agravado por el bajo número de tareas de meta-training (falla conocida como **memorisation**, cuando cada tarea se resuelve sin adaptarse al support); (ii) a tareas de una distribución *distinta* (domain-shift a nivel meta), inevitable al pasar de ImageNet a imágenes médicas.
- **Task families**: muchos frameworks requieren familias de tareas que no siempre están disponibles; el meta-learning no supervisado y de tarea única podrían aliviarlo.
- **Costo de cómputo y many-shot**: la implementación naive de bilevel es cara en *tiempo* (cada paso externo requiere varios internos) y *memoria* (reverse-mode differentiation almacena los estados intermedios), de ahí el foco en few-shot. Para extenderlo a many-shot hay *implicit differentiation*, *forward-mode differentiation*, métodos online (con *short-horizon bias*) y closed-form solvers. Para el deployment, los feed-forward models tienen ventaja decisiva: resuelven tareas nuevas en una pasada sin backpropagation.
- **Benchmarks**: los existentes son demasiado estrechos para testear genuino *learning-to-learn*; se necesitan benchmarks más amplios y diversos (Meta-Dataset, Meta-World).

## Conexión con la Clase 26

Este survey es la **fuente de las fórmulas formales de la Clase 26**. Las ecuaciones de bilevel optimization, la distinción support/query, los conceptos de meta-train/meta-test y la separación base learner $\theta$ / meta-knowledge $\omega$ que vertebran la clase provienen directamente de aquí. La clase presenta MAML, Matching/Prototypical y memory-augmented networks; el survey provee el marco que los unifica como puntos del mismo espacio de diseño de tres ejes.

Su distinción entre "hay meta-objetivo explícito" vs "no lo hay" es además la prueba de fuego para decidir, en cualquier proyecto de datos escasos, si lo que se necesita es transfer learning, domain generalization vanilla o meta-learning genuino. En dominios donde los datos son escasos por construcción (enfermedades raras, cohortes pequeñas, generalización entre centros con distintos equipos y protocolos), el meta-learning es el paradigma diseñado exactamente para ese régimen, y su sección de *social good* recoge aplicaciones médicas concretas (detección de cáncer de mama con MAML + currículum, segmentación de lesiones de piel con labels ruidosos).

Una limitación de alcance: la versión arXiv v2 es de noviembre de 2020, por lo que queda fuera la ola posterior del meta-learning implícito en los Large Language Models (el *in-context learning* como forma de learning-to-learn) y los foundation models que cambiaron el cálculo costo-beneficio del few-shot clásico.

## Notas y enlaces

- Publicado en IEEE TPAMI 2021; preprint en arXiv:2004.05439 (v2, noviembre 2020). Es un survey, no un benchmark: no provee comparaciones experimentales cabeza a cabeza; para números concretos hay que ir a los papers individuales.
- La Figura 1 (taxonomía) y la Tabla 1 son citadas en cientos de papers posteriores como referencia de posicionamiento, lo que convirtió al survey en infraestructura conceptual del campo.
- El propio paper reconoce que la formulación bilevel es estrictamente precisa solo para métodos optimizer-based, y que sus tres ejes tienen solapamientos (el step size puede ser hiperparámetro u optimizador; metric learning es caso especial de feed-forward model).

Ver fundamentos: [Meta-aprendizaje](/fundamentos/meta-aprendizaje) - [Optimización binivel](/fundamentos/optimizacion-binivel) - [Few-Shot Learning](/fundamentos/few-shot-learning) - [Transfer Learning](/fundamentos/transfer-learning).

Ver papers: [MAML (Finn 2017)](/papers/maml-finn-2017) - [Prototypical Networks (Snell 2017)](/papers/prototypical-networks-snell-2017) - [Matching Networks (Vinyals 2016)](/papers/matching-networks-vinyals-2016) - [MANN (Santoro 2016)](/papers/mann-santoro-2016).

Ver clase: [Clase 26 -- Meta-aprendizaje](/clases/clase-26).
