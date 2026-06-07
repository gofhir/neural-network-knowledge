# Meta-Learning in Neural Networks: A Survey (Hospedales et al., 2020/2021)

## 1. Metadata y resumen ejecutivo

- **Título**: "Meta-Learning in Neural Networks: A Survey"
- **Autores**: Timothy Hospedales (Samsung AI Centre, Cambridge / University of Edinburgh), Antreas Antoniou, Paul Micaelli, Amos Storkey (University of Edinburgh)
- **Publicación**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021. Preprint en arXiv:2004.05439 (v2, 7 nov 2020).
- **Tipo**: survey / artículo de revisión. No introduce un método nuevo, sino un **marco conceptual y una taxonomía**.

Este survey es la **referencia canónica** del meta-learning contemporáneo en redes neuronales por tres razones. Primero, llega en el momento exacto (2020) en que el campo había explotado tras MAML (Finn et al., 2017), Matching Networks (Vinyals et al., 2016) y Prototypical Networks (Snell et al., 2017), pero carecía de un vocabulario unificado: distintas comunidades usaban "meta-learning" de formas incompatibles. Segundo, propone una **definición formal unificada** (meta-learning como optimización de dos niveles / bilevel optimization) que reconcilia las visiones dispersas. Tercero, ofrece una **taxonomía de tres ejes** (meta-representation, meta-optimizer, meta-objective) que funciona como un espacio de diseño: cualquier método existente o futuro se puede ubicar como un punto en ese espacio, y cualquier combinación no explorada de los ejes sugiere un método potencialmente nuevo.

El valor del paper, dado que es un survey, no está en un resultado experimental sino en el **mapa conceptual**. Es el documento que un practicante lee para orientarse: entender qué se ha hecho, cómo se relaciona con campos vecinos (transfer learning, AutoML, HPO), y dónde están las fronteras abiertas. Para Roberto, como ML practitioner en salud, el survey provee precisamente la estructura mental para razonar sobre escenarios donde los datos son escasos por construcción (enfermedades raras, cohortes oncológicas pequeñas, generalización entre centros clínicos).

La tesis central del paper se puede resumir así: el deep learning convencional aprende un modelo *desde cero* con un algoritmo de aprendizaje *fijo y diseñado a mano*. El meta-learning, en cambio, **aprende el algoritmo de aprendizaje mismo** a partir de la experiencia de múltiples episodios de aprendizaje. Es la siguiente capa de "joint learning": el deep learning unió aprendizaje de features y de modelo; el meta-learning aspira a unir features, modelo y **algoritmo**.

El survey enmarca el meta-learning como una respuesta a las críticas conocidas al deep learning contemporáneo (Marcus): los grandes éxitos han ocurrido donde hay enormes cantidades de datos y cómputo, lo que excluye aplicaciones donde los datos son intrínsecamente raros o caros, o donde los recursos de cómputo no están disponibles. El meta-learning ofrece un paradigma alternativo orientado a eficiencia de datos y de cómputo, mejor alineado con el aprendizaje humano y animal — donde las estrategias de aprendizaje mejoran tanto en la escala de una vida como en la escala evolutiva. El paper distingue además dos escenarios donde el meta-learning ha probado utilidad: **multi-task** (se extrae conocimiento task-agnostic de una familia de tareas para mejorar el aprendizaje de tareas nuevas de esa familia) y **single-task** (un solo problema se resuelve repetidamente y se mejora a lo largo de múltiples episodios). El alcance que el survey se autoimpone es deliberadamente estrecho frente a usos históricos del término: se enfoca en *aprendizaje de algoritmo mediante optimización end-to-end de una función objetivo explícitamente definida* (como cross-entropy), excluyendo la mera selección de algoritmo basada en features del dataset (que se confunde con AutoML).

### Contexto histórico

El survey traza la genealogía del campo. El meta-learning y el learning-to-learn aparecen por primera vez en 1987 con J. Schmidhuber, quien introdujo métodos de **self-referential learning**: redes que reciben como input sus propios pesos y predicen actualizaciones para esos pesos, aprendidas con algoritmos evolutivos. Bengio et al. (1990) propusieron meta-aprender reglas de aprendizaje biológicamente plausibles. Thrun y Pratt (1998) definieron operacionalmente "learning to learn" como lo que ocurre cuando el rendimiento de un aprendiz al resolver tareas de una familia *mejora con respecto al número de tareas vistas* (en contraste con el ML convencional, donde el rendimiento mejora con más datos de una sola tarea). Esta perspectiva ve el meta-learning como herramienta para gestionar el teorema **"no free lunch"** de Wolpert: buscar el inductive bias mejor adaptado a un problema o familia de problemas. Las propuestas de entrenar sistemas de meta-learning con descenso de gradiente y backpropagation datan de 1991, con extensiones en 2001 (Hochreiter et al., "Learning to Learn Using Gradient Descent"). El meta-learning en RL apareció en 1995.

## 2. La definición formal unificada: meta-learning como bilevel optimization

### Punto de partida: aprendizaje supervisado convencional

En ML supervisado clásico, dado un dataset $\mathcal{D}=\{(x_1,y_1),\dots,(x_N,y_N)\}$, se entrena un modelo predictivo $\hat{y}=f_\theta(x)$ parametrizado por $\theta$ resolviendo:

$$\theta^* = \arg\min_\theta \mathcal{L}(\mathcal{D};\theta,\omega)$$

Aquí $\mathcal{L}$ es la función de pérdida y la clave es el **condicionamiento sobre $\omega$**: $\omega$ codifica los supuestos sobre *cómo aprender* (la elección de optimizador, la clase de funciones para $f$, la regularización, la inicialización). La asunción convencional es doble: (1) esta optimización se realiza *desde cero* para cada problema $\mathcal{D}$, y (2) $\omega$ está *pre-especificado*. El meta-learning ataca exactamente esa segunda asunción: en lugar de fijar $\omega$ a mano, lo **aprende**.

### La formulación bilevel

El survey propone formalizar el meta-entrenamiento como un problema de **optimización de dos niveles** (bilevel optimization), un concepto que viene de la teoría de Stackelberg en economía y de la programación matemática jerárquica. Las dos ecuaciones centrales del paper son:

$$\omega^* = \arg\min_\omega \sum_{i=1}^M \mathcal{L}^{meta}\!\left(\theta^{*(i)}(\omega),\,\omega,\,\mathcal{D}^{val(i)}_{source}\right)$$

sujeto a:

$$\theta^{*(i)}(\omega) = \arg\min_\theta \mathcal{L}^{task}\!\left(\theta,\,\omega,\,\mathcal{D}^{train(i)}_{source}\right)$$

Desglose riguroso de los componentes:

- **$\theta$ (base learner)**: los parámetros del modelo base que resuelve la tarea concreta (por ejemplo, los pesos de un clasificador de imágenes). Se optimizan en el **nivel interno (inner loop)** sobre el conjunto de entrenamiento de cada tarea.
- **$\omega$ (meta-knowledge / across-task knowledge)**: el "cómo aprender". Es lo que se comparte entre tareas y lo que el meta-learning optimiza en el **nivel externo (outer loop)**. Puede ser una inicialización, un optimizador, hiperparámetros, una métrica, una función de pérdida, una arquitectura, etc. (la taxonomía de la Sección 5 enumera todas las opciones).
- **$\mathcal{L}^{task}$ (objetivo interno)**: la pérdida de la tarea, por ejemplo cross-entropy sobre el support set.
- **$\mathcal{L}^{meta}$ (objetivo externo)**: la pérdida meta, típicamente evaluada sobre el conjunto de validación tras haber adaptado el modelo. Mide si $\omega$ produjo modelos $\theta^{*(i)}(\omega)$ que generalizan bien.

La estructura crítica es la **asimetría líder-seguidor (leader-follower)**: el nivel interno (Ec. interna) está *condicionado* a la estrategia de aprendizaje $\omega$ que define el nivel externo, pero **no puede cambiar $\omega$** durante su entrenamiento. $\omega$ es el líder (Stackelberg), $\theta$ es el seguidor. El nivel externo aprende $\omega$ tal que produzca modelos que rindan bien sobre sus conjuntos de validación *después* de entrenar.

El survey es honesto sobre el alcance de esta vista: argumenta que la imagen bilevel es "estrictamente precisa solo para los métodos basados en optimización" (como MAML), pero que sirve como herramienta para *visualizar la mecánica* del meta-learning en general, incluso para métodos feed-forward.

### La vista feed-forward (toy example)

Para los métodos que sintetizan modelos en una pasada feed-forward (en lugar de optimización iterativa), el survey instancia el objetivo abstracto con un ejemplo de meta-aprendizaje de regresión lineal:

$$\min_\omega \mathbb{E}_{\substack{\mathcal{T}\sim p(\mathcal{T}) \\ (\mathcal{D}^{tr},\mathcal{D}^{val})\in\mathcal{T}}} \sum_{(x,y)\in\mathcal{D}^{val}} \left[\left(x^T g_\omega(\mathcal{D}^{tr}) - y\right)^2\right]$$

Aquí $g_\omega$ es una función que **embebe** el train set $\mathcal{D}^{tr}$ en un vector de pesos de regresión. "Aprender a aprender" significa entrenar $g_\omega$ para mapear un conjunto de entrenamiento a un vector de pesos. Estos modelos se llaman **amortizados** (amortized): el costo de aprender una tarea nueva se reduce a una pasada feed-forward por $g_\omega(\cdot)$, porque la optimización iterativa "ya se pagó" durante el meta-entrenamiento de $\omega$.

## 3. Las dos vistas: task-distribution y bilevel optimization

El survey presenta el meta-learning desde dos perspectivas complementarias.

### Vista de distribución de tareas (task-distribution view)

Aquí el objetivo es aprender un algoritmo de aprendizaje de propósito general que **generalice a través de tareas**, idealmente de modo que cada tarea nueva se aprenda mejor que la anterior. Se define una tarea de forma laxa como un par dataset + función de pérdida $\mathcal{T}=\{\mathcal{D},\mathcal{L}\}$, y se evalúa $\omega$ sobre una distribución de tareas $p(\mathcal{T})$:

$$\min_\omega \mathbb{E}_{\mathcal{T}\sim p(\mathcal{T})}\, \mathcal{L}(\mathcal{D};\omega)$$

En la práctica, se asume acceso a un conjunto de $M$ **tareas fuente (source tasks)** muestreadas de $p(\mathcal{T})$ para el **meta-training**:

$$\mathcal{D}_{source} = \{(\mathcal{D}^{train}_{source},\mathcal{D}^{val}_{source})^{(i)}\}_{i=1}^M$$

donde cada tarea tiene datos de entrenamiento y de validación. En la jerga del few-shot, estos se llaman respectivamente **support set** y **query set**. El paso de meta-training se escribe como $\omega^*=\arg\max_\omega \log p(\omega\mid\mathcal{D}_{source})$.

Luego se tiene un conjunto de $Q$ **tareas objetivo (target tasks)** para el **meta-testing**, $\mathcal{D}_{target}=\{(\mathcal{D}^{train}_{target},\mathcal{D}^{test}_{target})^{(i)}\}_{i=1}^Q$. En la etapa de meta-test se usa el meta-conocimiento aprendido $\omega^*$ para entrenar el modelo base en cada tarea objetivo nunca vista:

$$\theta^{*(i)} = \arg\max_\theta \log p(\theta\mid\omega^*,\mathcal{D}^{train(i)}_{target})$$

El contraste con el aprendizaje convencional es nítido: aprender sobre el train set de una tarea objetivo ahora **se beneficia del meta-conocimiento $\omega^*$** sobre qué algoritmo usar. Ese $\omega^*$ puede ser una estimación de los parámetros iniciales (MAML), un modelo de aprendizaje completo, o una estrategia de optimización entera. La precisión del meta-learner se evalúa por el rendimiento de $\theta^{*(i)}$ sobre el split de test de cada tarea objetivo $\mathcal{D}^{test(i)}_{target}$.

Esta vista trae consigo analogías directas del underfitting/overfitting clásico: **meta-underfitting** y **meta-overfitting**. El meta-overfitting es el problema por el cual el meta-conocimiento aprendido sobre las tareas fuente *no generaliza* a las tareas objetivo. Es relativamente común, sobre todo cuando hay pocas tareas fuente; se interpreta como aprender un inductive bias $\omega$ que restringe el espacio de hipótesis de $\theta$ demasiado ajustado a las soluciones de las tareas fuente.

### Vista bilevel

La vista de distribución de tareas describe el *flujo* pero no dice *cómo* resolver el paso de meta-training. La vista bilevel (Sección 2 de este análisis) es la que responde el "cómo": casteando el meta-training como la optimización jerárquica de dos niveles.

### Caso de tarea única

El survey enfatiza que la noción de distribución sobre tareas **no es una condición necesaria** para el meta-learning. Con un solo train y test dataset ($M=Q=1$), se puede partir el train set para obtener datos de validación: $\mathcal{D}_{source}=(\mathcal{D}^{train}_{source},\mathcal{D}^{val}_{source})$ para meta-training, y $\mathcal{D}_{target}=(\mathcal{D}^{train}_{source}\cup\mathcal{D}^{val}_{source},\mathcal{D}^{test}_{target})$ para meta-testing. Se aprende $\omega$ sobre varios episodios con distintos splits train-val. Esto cubre, por ejemplo, la optimización de hiperparámetros de tarea única.

## 4. Posicionamiento frente a campos vecinos

Una de las contribuciones más útiles del survey es desambiguar el meta-learning de campos con los que se confunde. El criterio discriminante recurrente es: **¿hay un meta-objetivo explícito optimizado end-to-end?** Si no lo hay, no es meta-learning en el sentido del paper.

- **Transfer Learning (TL)**: usa experiencia de una tarea fuente para mejorar el aprendizaje en una tarea objetivo, típicamente vía transferencia de parámetros + fine-tuning opcional. La diferencia: en TL el prior se extrae por aprendizaje *vanilla* sobre la fuente, *sin* meta-objetivo. En meta-learning, el prior se define por una optimización externa que *evalúa el beneficio del prior cuando se aprende una tarea nueva* (como MAML). Además, meta-learning maneja una gama mucho más amplia de meta-representaciones, no solo parámetros del modelo.

- **Domain Adaptation (DA) y Domain Generalization (DG)**: ambos atacan el *domain-shift* (mismo objetivo, distinta distribución de entrada). DA adapta usando datos escasos/no etiquetados del target; DG busca robustez sin adaptación. De nuevo, DA y DG vanilla no usan meta-objetivo. Pero el meta-learning *puede usarse* para hacer DA y DG (Sección 5.8). **Este punto es de máxima relevancia para salud multi-centro** (ver Sección 11).

- **Continual Learning (CL)**: aprende sobre una secuencia de tareas de distribución potencialmente no estacionaria, acelerando el aprendizaje de tareas nuevas sin olvidar las viejas. Comparte con meta-learning la noción de distribución de tareas, pero la mayoría de métodos de CL *no resuelven explícitamente* un meta-objetivo. El meta-learning provee un marco para avanzar CL definiendo meta-objetivos que codifican el rendimiento continual.

- **Multi-Task Learning (MTL)**: aprende conjuntamente varias tareas relacionadas para beneficiarse de la regularización por compartir parámetros. Es optimización de un solo nivel, sin meta-objetivo. Además, MTL busca resolver *un número fijo de tareas conocidas*, mientras que el meta-learning apunta a *tareas futuras no vistas*.

- **Hyperparameter Optimization (HO)**: cae *dentro* del meta-learning cuando define un meta-objetivo entrenado end-to-end con redes neuronales (HO basado en gradiente, NAS). Pero el survey *excluye* random search y Bayesian Optimization, que rara vez se consideran meta-learning.

- **Hierarchical Bayesian Models (HBM)**: proveen una vista de *modelado* (no algorítmica) del meta-learning, con un prior $p(\theta\mid\omega)$ y un prior sobre $\omega$. MAML puede entenderse a través del lente de HBM (Grant et al.). El modelo completo es $\left[\prod_{i=1}^M p(\mathcal{D}_i\mid\theta_i)p(\theta_i\mid\omega)\right]p(\omega)$.

- **AutoML**: paraguas amplio para automatizar partes del pipeline de ML (preparación de datos, selección de algoritmo, tuning, búsqueda de arquitectura). Usa muchas heurísticas fuera del meta-learning, pero a veces emplea optimización end-to-end de un meta-objetivo. El survey concluye que **el meta-learning puede verse como una especialización de AutoML**.

## 5. La taxonomía de tres ejes (contribución central)

Las taxonomías previas dividían el meta-learning en tres familias: **optimization-based** (el inner loop se resuelve literalmente como optimización, ej. MAML), **model-based / black-box** (el inner learning está envuelto en la pasada feed-forward de un modelo, ej. RNNs, memory-augmented networks) y **metric-based / non-parametric** (aprendizaje no paramétrico en el nivel interno por comparación de puntos, ej. Siamese, Matching, Prototypical, Relation, Graph networks).

El survey argumenta que esa división *no expone todas las facetas de interés*. Propone en cambio una descomposición a lo largo de **tres ejes independientes**, que juntos forman un espacio de diseño. Nota importante: la representación del modelo base $\theta$ **no** se incluye en la taxonomía, porque se determina específicamente según la aplicación.

### Eje 1 — Meta-Representation ("What?"): qué se meta-aprende

Es la elección del meta-conocimiento $\omega$. El catálogo del paper:

- **Parameter Initialization**: $\omega$ es la inicialización de pesos. **MAML** es el ejemplo canónico: una buena inicialización está a pocos pasos de gradiente de la solución de cualquier tarea $\mathcal{T}\sim p(\mathcal{T})$. Reto: el outer loop debe resolver tantos parámetros como el inner (hasta cientos de millones). De ahí líneas de trabajo que aíslan subconjuntos a meta-aprender (por subespacio, por capa, separando escala/shift) y variantes con mixturas de múltiples condiciones iniciales.

- **Optimizer**: $\omega$ define el optimizador interno, una función que toma estados de optimización ($\theta$ y $\nabla_\theta\mathcal{L}^{task}$) y produce el paso de actualización. Va desde un step size fijo, a matrices de pre-condicionamiento, hasta optimizadores no lineales completos (ej. "Learning to learn by gradient descent by gradient descent"). Se puede fusionar con la inicialización (aprender ambos conjuntamente).

- **Feed-Forward Models (FFM / black-box / amortized)**: $\omega$ es una red que mapea directamente el support set a los parámetros del clasificador, $\theta=g_\omega(\mathcal{D}^{train})$, sin optimización iterativa. Conectan con **hypernetworks** (generan los pesos de otra red). Pueden entenderse como inferencia amortizada en modelos probabilísticos:
  $$q_\omega(y\mid x,\mathcal{D}^{tr}) = \int p(y\mid x,\theta)\, q_\omega(\theta\mid\mathcal{D}^{tr})\, d\theta$$
  Las **memory-augmented neural networks** caen en esta categoría.

- **Embedding Functions (Metric Learning)**: $\omega$ es una red de embedding que transforma inputs en una representación apta para reconocimiento por comparación de similitud (coseno o euclídea) entre query y support. Prototypical, Matching, Relation, Graph nets. El survey muestra que son un **caso especial de FFM**: cuando los logits se basan en el producto interno de embeddings $g_\omega^T(x_q)g_\omega(x_s)$, el support genera "pesos" para interpretar el query.

- **Losses and Auxiliary Tasks**: $\omega$ es la función de pérdida interna $\mathcal{L}^{task}_\omega(\cdot)$, una pequeña red que produce un escalar. Beneficios: pérdidas más fáciles de optimizar (menos mínimos locales), aprendizaje más rápido, o mínimos que corresponden a modelos robustos a domain-shift. También aproximaciones diferenciables a pérdidas no diferenciables (ej. área bajo curva precision-recall).

- **Architectures (NAS)**: $\omega$ especifica la arquitectura. Enfoques evolutivos (topología de celdas LSTM), RL (generar descripciones de CNNs, Zoph & Le), y gradiente (**DARTS**: softmax sobre todas las capas posibles de un bloque, ponderadas por coeficientes meta-aprendidos; en meta-test se discretiza quedándose con los coeficientes más altos).

- **Attention Modules**: como comparadores en meta-learners métricos, para prevenir catastrophic forgetting, etc.

- **Modules**: el conocimiento task-agnostic define un conjunto de módulos re-componibles por tarea.

- **Hyperparameters**: regularización, task-relatedness, sparsity. Solapa con el optimizer (ej. step size).

- **Data Augmentation**: $\omega$ define la estrategia de augmentation, optimizada para maximizar validación. Como las operaciones suelen ser no diferenciables, requiere RL, estimadores de gradiente discreto o evolución.

- **Minibatch Selection, Sample Weights, Curriculum**: $\omega$ como probabilidad de selección de instancias o pesos por muestra (útil para label noise, outliers, desbalance de clases), o como política de teaching que define un currículum.

- **Datasets, Labels, Environments**: representaciones **transductivas** donde $\omega$ literalmente son datos. *Dataset distillation* aprende las imágenes de soporte; también se pueden aprender labels (semi-supervisado) o parámetros de un simulador (**sim2real**).

El survey hace tres distinciones transversales sobre las meta-representaciones: (a) **transductivas vs no** (cuando $\omega$ escala con el tamaño del dataset, limitando escalabilidad); (b) **simbólicas vs sub-simbólicas** (código legible vs redes neuronales; las simbólicas generalizan mejor a través de familias de tareas pero requieren RL/evolución por ser no diferenciables); (c) grado de **amortización** (desde fine-tuning sin amortización, pasando por MAML con amortización limitada, hasta FFMs puros completamente amortizados, con híbridos semi-amortizados en el medio).

### Eje 2 — Meta-Optimizer ("How?"): cómo se optimiza el nivel externo

- **Gradient**: descenso de gradiente sobre $\omega$, requiriendo $\frac{d\mathcal{L}^{meta}}{d\omega} = \frac{d\mathcal{L}^{meta}}{d\theta}\cdot\frac{d\theta}{d\omega}$ por regla de la cadena. Es lo más eficiente cuando hay gradientes analíticos, pero enfrenta tres retos: (i) diferenciar a través de muchos pasos del inner loop (de ahí implicit differentiation, gradientes de segundo orden); (ii) degradación del gradiente que empeora con el número de pasos internos; (iii) operaciones discretas/no diferenciables.

- **Reinforcement Learning**: cuando el base learner o el meta-objetivo son no diferenciables, se estima $\nabla_\omega\mathcal{L}^{meta}$ vía policy gradient. Alivia el requisito de diferenciabilidad pero es extremadamente costoso: estimaciones de alta varianza requieren muchos pasos externos, cada uno costoso por envolver la optimización de la tarea.

- **Evolution (EA)**: optimiza cualquier modelo y meta-objetivo sin restricción de diferenciabilidad, no sufre degradación de gradiente, es altamente paralelizable y evita mínimos locales por mantener poblaciones diversas. Desventajas: el tamaño de población crece rápido con el número de parámetros, sensibilidad a la estrategia de mutación, y capacidad de ajuste generalmente inferior al gradiente para modelos grandes (CNNs). Más común en RL (modelos pequeños, inner loops largos no diferenciables) y en representaciones simbólicas.

### Eje 3 — Meta-Objective ("Why?"): para qué se meta-aprende

Determinado por la elección de $\mathcal{L}^{meta}$, la distribución de tareas $p(\mathcal{T})$ y el flujo de datos entre niveles. Opciones de diseño de episodio:

- **Many vs Few-Shot**: episodios con muchos o pocos ejemplos por tarea.
- **Fast Adaptation vs Asymptotic Performance**: si la pérdida de validación se computa al *final* del episodio interno, se favorece el rendimiento final; si se computa como *suma tras cada paso interno*, se favorece la *velocidad* de aprendizaje.
- **Multi vs Single-Task**: episodios de tareas aleatorias de $p(\mathcal{T})$, o todos de la misma tarea subyacente. Multi-task requiere familia de tareas y amortiza el costo de meta-training entre múltiples targets; single-task debe mejorar el rendimiento asintótico o ser online.
- **Online vs Offline**: meta-optimización dentro de un solo episodio (base $\theta$ y learner $\omega$ co-evolucionan) vs el pipeline clásico de outer-loop separado.
- **Otros factores**: simular domain-shift, compresión/cuantización, label noise, o un validation set adversarial entre train y val para meta-optimizar la robustez correspondiente.

La Tabla 1 del paper cruza meta-representation × meta-optimizer y colorea por meta-objetivo (eficiencia de muestras, velocidad de aprendizaje, rendimiento asintótico, cross-domain), ubicando cada paper relevante en una celda.

## 6. Aplicaciones revisadas

- **Few-Shot Classification (Computer Vision)**: la aplicación más común. Inner y outer loss son cross-entropy sobre train y val. Se han usado modelos optimizer-centric (MAML), black-box y metric learning. Extensiones a object detection, landmark prediction (pose), segmentación few-shot, generación de imágenes/video y density estimation (PixelCNN few-shot). **Benchmarks**: miniImageNet, Tiered-ImageNet, SlimageNet, Omniglot y Meta-Dataset. Problema reconocido: falta de diversidad ($p(\mathcal{T})$ estrecha) hace que el rendimiento no refleje el mundo real; de ahí Meta-Dataset y el CVPR cross-domain few-shot challenge (generalizar de ImageNet a imágenes médicas, satelitales, agrícolas).

- **Meta-RL y Robótica**: el RL sufre extrema ineficiencia de muestras por recompensas dispersas, la necesidad de exploración y la alta varianza de los optimizadores. Pero abunda en familias de tareas naturales (locomoción a distintas posiciones, navegación en distintos entornos, conducir distintos autos, competir con distintos agentes, lidiar con handicaps como fallas en extremidades del robot). La intuición es clara: el meta-conocimiento de, por ejemplo, el layout de un laberinto es transferible a todas las tareas que requieran navegar ese laberinto. Meta-representaciones exploradas: condiciones iniciales, hiperparámetros, step directions/sizes, modelos black-box recurrentes/convolucionales que embeben la experiencia de un entorno para sintetizar una política. Particular del RL: la **exploration policy** como $\omega$ — modelar la adquisición de la estrategia de exploración o la función de curiosidad como problema de meta-learning ("aprender a explorar"). Por la dificultad de optimización (la política aprendida suele estar lejos del óptimo incluso en episodios de entrenamiento), meta-RL se usa más para mejorar rendimiento *asintótico* (no solo eficiencia de muestras), con mucho trabajo en aprender pérdidas/recompensas más densas y suaves que la recompensa dispersa verdadera. Dicotomía relevante: los métodos off-policy (más eficientes en muestras) han sido más difíciles de extender a meta-RL, por lo que muchos métodos meta-RL se construyen sobre RL on-policy, limitando su rendimiento absoluto. Benchmarks: ALE/Atari, Sonic, CoinRun/ProcGen, Meta-World (50 tareas de control continuo, cuya evaluación sugiere que los métodos actuales luchan con distribuciones amplias y shifts meta-train/meta-test), PHYRE.

- **NAS (Neural Architecture Search)**: HPO donde $\omega$ es la arquitectura. El inner loop entrena redes con la arquitectura especificada; el outer loop busca arquitecturas con buen rendimiento de validación. Analizado por search space / search strategy / performance estimation (que corresponden a hipótesis de $\omega$, meta-optimizer y meta-objetivo). Retos: evaluar el inner loop es carísimo (entrenar una red completa hasta el final), lo que fuerza aproximaciones como sub-sampling del train set, terminación temprana del inner loop, y descenso intercalado sobre $\omega$ y $\theta$ (como en DARTS, que es online meta-learning); además el search space es amplio, difícil de definir y no trivialmente diferenciable, lo que lleva a búsqueda a nivel de celda, RL, estimadores de gradiente discreto y evolución. NASbenches provee mediciones pre-computadas para reproducibilidad. Tópico abierto: si las arquitecturas descubiertas generalizan a problemas nuevos, y meta-entrenar arquitecturas iniciales o priors de arquitectura fáciles de adaptar (análogo a MAML).

- **Domain Generalization / Adaptation**: simular domain-shift entre train y val para meta-aprender regularizadores, pérdidas o noise augmentation que maximicen robustez al shift train-test. Benchmarks: PACS, Visual Decathlon, Meta-Dataset.

- **Bayesian Meta-Learning**: formaliza el meta-learning vía modelado jerárquico bayesiano y usa inferencia (no optimización directa) para aprender. Provee **medidas de incertidumbre** sobre $\omega$ y sobre las predicciones — crítico para aplicaciones safety-critical, exploración en RL y active learning. Ejemplos: Neural Statistician (extiende VAEs para modelar variables de tarea), Neural Processes (meta-learner bayesiano feed-forward inspirado en Gaussian Processes), recasting de MAML como hierarchical empirical Bayes (Grant et al.), Bayesian MAML (ensemble con posteriors no gaussianos sobre $\theta$) y Probabilistic MAML (modela la incertidumbre en $\omega$ con estimación MAP de $\theta$).

- **Unsupervised Meta-Learning**: el aprendizaje no supervisado interactúa con el meta-learning según se ubique en inner/outer loop y en meta-train/meta-test. Dos familias: (a) *unsupervised learning of a supervised learner* — construir tareas fuente sintéticas sin supervisión (clustering, augmentation que preserva clase) para definir el meta-objetivo, evitando necesitar muchas tareas fuente etiquetadas; (b) *supervised learning of an unsupervised learner* — meta-entrenar reglas o pérdidas no supervisadas tal que el rendimiento supervisado downstream se optimice (incluye *amortized clustering*, que amortiza la computación iterativa de clustering en una sola pasada feed-forward).

- **Adversarial Defense, Label Noise, Active Learning, Continual Learning, Sim2Real, Network Compression, Communications, Recommendation (cold-start), Language/Speech, Abstract Reasoning**: el survey recorre todas estas, mostrando la amplitud del paradigma. En **adversarial defense** se entrena el algoritmo de aprendizaje para robustez definiendo un meta-loss en términos del rendimiento bajo ataque adversarial. En **active learning** se mapea el diseño del algoritmo a una tarea de aprendizaje: el inner loop es el aprendizaje supervisado convencional, $\omega$ es la query policy que selecciona los mejores datapoints no etiquetados, y el meta-objetivo es el rendimiento de validación tras anotar y aprender iterativamente. En **recommendation** se ataca el cold-start (usuarios/items nuevos con poca historia) con modelos black-box (item cold-start) y métodos basados en gradiente (user cold-start). Destaca una subsección de **Meta-learning for Social Good** con aplicaciones médicas, especialmente relevante dada la escasez global de patólogos: predicción de toxicidad de moléculas one-shot (LSTM + graph neural network), MAML adaptado a detección de cáncer de mama débilmente supervisada con el orden de las tareas seleccionado por currículum, MAML + denoising autoencoders para visual question answering médico, y ponderación de muestras (extendida a pixel-wise) para segmentación de lesiones de piel con labels ruidosos.

## 7. Challenges abiertos

- **Distribuciones de tareas diversas y multi-modales**: muchos éxitos han sido en familias *estrechas*. Frameworks como MAML asumen implícitamente que $p(\mathcal{T})$ es *uni-modal* y que un solo $\omega$ sirve para todas. Pero las distribuciones reales son multi-modales (imágenes médicas vs satelitales vs cotidianas). Distintas tareas requieren distintas estrategias, algo difícil hoy, en parte por gradientes en conflicto entre tareas.

- **Meta-generalización**: el reto de generalización *a través de tareas*, análogo a generalizar a través de instancias. Dos sub-retos: (i) generalizar de meta-train a tareas meta-test nuevas de la misma $p(\mathcal{T})$ — agravado porque el número de tareas de meta-training suele ser bajo; falla conocida como **memorisation** (cada tarea de meta-train se resuelve sin adaptación al support set); (ii) generalizar a tareas meta-test de una distribución *distinta* (meta-nivel del domain-shift) — inevitable en aplicaciones prácticas como pasar de ImageNet a imágenes médicas.

- **Task families**: muchos frameworks requieren familias de tareas para meta-training, que a veces no están disponibles. Unsupervised y single-task meta-learning podrían aliviar este requisito.

- **Costo de cómputo y many-shot**: la implementación naive de bilevel es cara en *tiempo* (cada paso externo requiere varios internos) y *memoria* (reverse-mode differentiation almacena los estados internos intermedios). Por eso gran parte del meta-learning se ha enfocado en el régimen few-shot. Para extenderlo a many-shot, las soluciones incluyen: **implicit differentiation** de $\omega$ (escala a grandes dimensiones pero da gradientes aproximados y requiere que la pérdida de tarea sea función de $\omega$), **forward-mode differentiation** (exacto y sin esas restricciones, pero escala mal con dim($\omega$)), métodos **online** que alternan pasos inner/outer (baratos pero con *short-horizon bias*), gradient preconditioning, truncation, shortcuts o inversión del inner loop, aprender una inicialización que minimice la longitud de la trayectoria de descenso, y closed-form solvers en el inner loop. La **degradación del gradiente** también es un reto en many-shot, con soluciones como warp layers o promediado de gradientes. Para el meta-test, los **FFMs tienen ventaja decisiva** porque resuelven tareas nuevas en una pasada feed-forward sin backpropagation, lo que los hace ideales para deployment en dispositivos móviles cuyos frameworks embebidos típicamente carecen de soporte para entrenamiento por backprop.

- **Benchmarks**: necesidad de benchmarks más amplios y diversos; los existentes son demasiado estrechos para testear genuino learning-to-learn (la evaluación de Meta-World sugiere que los métodos actuales luchan con distribuciones amplias y shifts meta-train/meta-test).

## 8. Cómo el survey ubica los métodos de la Clase 26

Los tres métodos que conviene anclar en la taxonomía:

- **MAML (Finn et al., 2017)**: en la taxonomía previa es **optimization-based**. En la taxonomía de tres ejes es: meta-representation = **Parameter Initialization** ($\omega=\theta_0$); meta-optimizer = **Gradient** (diferenciando a través de los updates del modelo base, requiriendo gradientes de segundo orden); meta-objective = sample-efficient **few-shot**. La idea: aprender una inicialización tal que pocos pasos internos produzcan un clasificador que rinde bien en validación. En términos de amortización, MAML provee amortización *limitada* (ajusta una condición inicial para permitir fine-tuning de pocos pasos).

- **Matching Networks y Prototypical Networks**: en la taxonomía previa son **metric-based / non-parametric**. En los tres ejes: meta-representation = **Embedding Functions (Metric)** ($\omega$ es la red de embedding); meta-optimizer = **Gradient**; meta-objective = **few-shot**. El survey las reconcilia mostrando que metric learning es un **caso especial de FFM**: el embedding del support genera implícitamente los "pesos" (prototipos / similitudes) que interpretan el query, equivalente a una hypernetwork que sintetiza un clasificador lineal. Prototypical en particular hace clasificación no paramétrica por distancia euclídea a centroides de clase.

- **Memory-Augmented Neural Networks**: en la taxonomía previa son **model-based / black-box** (un buffer de almacenamiento explícito). En los tres ejes: meta-representation = **Feed-Forward Model / Black-Box** (con memoria explícita); meta-optimizer = **Gradient**; meta-objective = few-shot. Tienen la habilidad de recordar datos viejos y asimilar datos nuevos rápido. Ventaja: optimización más simple sin gradientes de segundo orden. Desventaja observada: generalizan peor a tareas out-of-distribution que los métodos basados en optimización, y son asintóticamente más débiles (cuesta embeber un train set grande en un modelo base rico).

El punto pedagógico clave para la Clase 26: estos tres métodos, que parecen muy distintos, son **tres elecciones del mismo eje 1 (meta-representation)** combinadas con el mismo eje 2 (gradient) y el mismo eje 3 (few-shot). El survey transforma una lista heterogénea en un espacio ordenado.

## 9. Por qué importa

El survey importa porque **dio al campo un vocabulario común**. Antes de él, "meta-learning" significaba cosas distintas según la comunidad (algoritmo selection en data mining, AutoML, learning-to-learn de Thrun). El paper acota el término a "aprendizaje de algoritmo mediante optimización end-to-end de un meta-objetivo explícitamente definido", y esa definición se volvió estándar.

La **figura de la taxonomía** (Fig. 1) y la Tabla 1 son citadas en cientos de papers posteriores como referencia de posicionamiento: un autor que propone un método nuevo lo ubica diciendo "nuestra meta-representation es X, nuestro meta-optimizer es Y, nuestro meta-objective es Z". Esto convirtió al survey en infraestructura conceptual del campo. La descomposición en tres ejes también funciona como **generador de ideas**: combinaciones no exploradas de los ejes sugieren métodos nuevos (el "design-space" que el paper explícitamente promueve).

## 10. Limitaciones del survey

- **Cobertura temporal hasta 2020**: la versión arXiv v2 es de noviembre 2020. Queda *fuera* toda la ola posterior: el meta-learning implícito en los Large Language Models (in-context learning como forma de learning-to-learn), prompt-based few-shot a gran escala (GPT-3 es contemporáneo pero su lectura como meta-learner se desarrolla después), y los avances en foundation models que cambiaron el cálculo costo-beneficio del few-shot clásico.

- **Es un survey, no un benchmark empírico**: no provee comparaciones experimentales cabeza a cabeza ni reproduce resultados. Para números concretos hay que ir a los papers individuales.

- **Sesgo hacia la vista bilevel**: el propio paper admite que la formulación bilevel es estrictamente precisa solo para métodos optimizer-based; forzar los métodos feed-forward/métricos a ese molde es una simplificación pedagógica.

- **La taxonomía tiene solapamientos**: los ejes no son perfectamente ortogonales (step size puede ser hiperparámetro u optimizador; metric learning es caso especial de FFM). El survey reconoce estos solapamientos pero no los resuelve del todo.

## 11. Conexión con la Clase 26 y relevancia para salud

Este survey es la **fuente de las fórmulas formales de la Clase 26**. Las ecuaciones de bilevel optimization ($\omega^*=\arg\min_\omega\sum_i\mathcal{L}^{meta}(\theta^{*(i)}(\omega),\omega,\mathcal{D}^{val}_i)$ sujeto a $\theta^{*(i)}(\omega)=\arg\min_\theta\mathcal{L}^{task}(\theta,\omega,\mathcal{D}^{train}_i)$), la distinción support/query, los conceptos de meta-train/meta-test, y la separación base learner $\theta$ / meta-knowledge $\omega$ que vertebran la clase provienen directamente de aquí. La clase presenta MAML, Matching/Prototypical y memory-augmented; el survey provee el marco que los unifica como puntos del mismo espacio de diseño.

**Relevancia para salud y oncología (FALP)**:

- **Few-shot por construcción**: en oncología hay enfermedades raras, subtipos histológicos poco frecuentes y cohortes pequeñas donde nunca habrá millones de ejemplos. El meta-learning es el paradigma diseñado exactamente para ese régimen de datos escasos, como ilustra la propia subsección de "social good" del paper (detección de cáncer de mama con MAML + currículum, segmentación de lesiones de piel con labels ruidosos).

- **Domain generalization multi-centro**: el reto número uno del ML clínico desplegable es el *domain-shift* entre hospitales — distintos equipos de imagen, distintos protocolos de adquisición, distintas poblaciones de pacientes. El survey muestra (Sección 5.8) cómo el meta-learning puede *meta-optimizar para robustez al domain-shift* simulando ese shift entre train y validation durante el meta-training. Un modelo entrenado en datos de FALP que deba generalizar a otro centro de la red, o a datos de un equipo nuevo, es precisamente el escenario de DG que el paper formaliza.

- **Label noise**: las anotaciones clínicas (extraídas de registros, crowd-sourced, o derivadas de códigos) son ruidosas. El eje de *sample weights* (down-weighting de muestras ruidosas como $\omega$) es directamente aplicable.

- **Conexión con el trabajo de Roberto en FHIR/MDM**: el patient matching y el master data management enfrentan un problema análogo al meta-learning multi-dominio — generalizar reglas de matching a través de fuentes de datos heterogéneas (distintos centros con distintas convenciones de captura). La vista de "aprender cómo aprender a hacer matching robusto al shift entre fuentes" es un encuadre útil, aunque el GBM/embedding-blocker actual no sea meta-learning en sentido estricto (no hay meta-objetivo bilevel explícito, lo que lo ubica más cerca del transfer learning según la propia taxonomía del survey).

En síntesis: el survey de Hospedales et al. es el mapa que permite navegar el meta-learning con un vocabulario preciso, y su distinción entre "hay meta-objetivo explícito" vs "no lo hay" es la prueba de fuego para decidir, en cualquier proyecto clínico de datos escasos, si lo que se necesita es transfer learning, domain generalization vanilla, o meta-learning genuino.
