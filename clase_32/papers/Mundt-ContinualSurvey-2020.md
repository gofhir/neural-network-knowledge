---
title: "A Wholistic View of Continual Learning — Análisis interno"
linkTitle: "Survey de Continual Learning (Mundt et al.)"
weight: 60
---

# A Wholistic View of Continual Learning with Deep Neural Networks — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *A Wholistic View of Continual Learning with Deep Neural Networks: Forgotten Lessons and the Bridge to Active and Open World Learning*.
- **Autores:** Martin Mundt (Goethe University Frankfurt / TU Darmstadt), Yongwon Hong (Yonsei University), Iuliia Pliushch (Goethe University) y Visvanathan Ramesh (Goethe University).
- **Venue:** *Neural Networks* (Elsevier), versión final aceptada — [doi.org/10.1016/j.neunet.2023.01.014](https://doi.org/10.1016/j.neunet.2023.01.014).
- **Año:** 2020 (preprint) / 2023 (versión publicada). **Preprint:** arXiv:2009.01797v3 (23 ene 2023).

> **NOTA IMPORTANTE sobre la procedencia del archivo.** Este archivo lleva el nombre `Masana-CILSurvey-2020` porque en la Clase 32 se cita el *survey* de class-incremental learning de Masana et al. (*"Class-incremental learning: survey and performance evaluation on image classification"*, IEEE TPAMI, arXiv:2010.15277). Sin embargo, el PDF y el texto extraído que acompañan a este archivo corresponden **en realidad** al survey de Mundt et al., arXiv:2009.01797 (el identificador arXiv que la bibliografía de la clase asoció a "Masana 2020" pertenece a Mundt et al., no a Masana et al.). Para no inventar contenido, este análisis está **fundado exclusivamente en el paper que efectivamente tenemos a mano** (Mundt et al.). Felizmente ambos son surveys de *continual learning* con la misma taxonomía de familias (regularización / rehearsal-memoria / arquitectura) y los mismos métodos canónicos (EWC, SI, LwF, iCaRL, GEM, BiC), de modo que el análisis cumple el mismo rol pedagógico para la Clase 32: dar el panorama completo del campo que la clase organiza. Donde la consigna pedía conceptos específicos del survey de Masana (p.ej. *task-recency bias* nombrado así, el protocolo de evaluación unificado de iCaRL/EEIL/LUCIR/etc.), se aclara que esos detalles pertenecen al survey de Masana y aquí se discuten en la medida en que Mundt et al. los toca.

Este no es un paper de un *algoritmo* nuevo: es un **survey crítico** sobre el aprendizaje continuo (continual learning, CL) con redes neuronales profundas. Su tesis no es proponer un mejor método contra el olvido catastrófico, sino **denunciar el estrechamiento del campo**: la comunidad de continual learning, según los autores, redujo el problema a "monitorear el olvido catastrófico sobre versiones secuencializadas de benchmarks clásicos" (MNIST, CIFAR, ImageNet partidos en tareas disjuntas), bajo una **suposición de mundo cerrado** (closed world) que asume que el modelo solo verá datos de la misma distribución de entrenamiento. El paper argumenta que ese marco ignora "lecciones olvidadas" de dos campos vecinos —**active learning** (qué datos consultar/etiquetar) y **open set recognition** (cómo rechazar lo desconocido)— y propone una **visión consolidada** que los une bajo el paraguas del aprendizaje continuo, validándola con evidencia empírica sobre extracción de *exemplars*/core sets, consultas activas, robustez a corrupciones y orden de tareas.

Para la **Clase 32 (Olvido Catastrófico)** este survey importa por su primera mitad: ofrece la **taxonomía canónica** de los métodos contra el olvido —regularización, rehearsal y arquitectura— con los nombres y referencias que la clase organiza (EWC, SI, MAS, LwF, iCaRL, GEM, BiC, PathNet, HAT, Piggyback, Progressive Networks). Entender este paper es entender *cómo está estructurado todo el campo* y *por qué la evaluación importa tanto como el algoritmo*.

## 2. Contexto: la fragmentación de métodos y evaluaciones en aprendizaje continuo

El olvido catastrófico (*catastrophic interference / catastrophic forgetting*, McCloskey y Cohen 1989; Ratcliff 1990) es el efecto particular de los modelos que actualizan sus parámetros *greedily* según la población de datos presente: una red que itera sus pesos con gradientes estocásticos, al recibir datos que desplazan la distribución, guía sus representaciones unidireccionalmente hacia la tarea actual y **sobrescribe** lo aprendido. El aprendizaje continuo es el paradigma que estudia cómo aprender de datos que llegan en el tiempo *sin* acumularlos todos, preservando el conocimiento previo.

El diagnóstico central del paper no es la falta de algoritmos, sino la **fragmentación y el reduccionismo de las prácticas de evaluación**. Tres problemas concretos:

1. **El mundo cerrado omnipresente.** Casi todos los benchmarks de CL evalúan sobre datos garantizadamente de la misma distribución que el entrenamiento. Pero "es un hecho conocido desde hace décadas que las redes neuronales son erróneamente *overconfident*" (Matan et al., 1990) ante datos desconocidos o corruptos. Los sistemas de CL "se rompen inmediatamente" ante datos no vistos o corrupciones menores en despliegue (Hendrycks y Dietterich, 2019). El olvido catastrófico se ataca con esmero mientras se ignora la robustez en el mundo abierto.
2. **Datasets secuencializados sin entender su naturaleza.** El grueso de los trabajos de class-incremental learning (Li y Hoiem 2016; Kirkpatrick et al. 2017; Rebuffi et al. 2017; Lopez-Paz y Ranzato 2017) toma benchmarks de clasificación (MNIST, CIFAR, ImageNet), parte las clases en conjuntos disjuntos y los muestra en secuencia (Figura 3 del paper). Preguntas sobre **el efecto del orden de tareas** o **el solapamiento entre tareas** se pasan por alto rutinariamente "en favor de retener comparabilidad sobre un benchmark".
3. **Métricas heredadas del aprendizaje aislado.** El protocolo de evaluación habitual (Figura 2) sigue inspirado en el aprendizaje en aislamiento: se extraen métricas de una matriz de confusión convencional (degradación de la accuracy de la primera tarea, *forward/backward transfer*, *amount of forgetting*, consumo de memoria, *task boundaries*, *prediction oracle*). Esto desatiende la relevancia del orden/permutación de tareas, la elección de tareas y de datos, y cualquier forma de robustez en el mundo abierto.

El paper sitúa el aprendizaje continuo dentro de una constelación de paradigmas vecinos que define cuidadosamente: *lifelong machine learning* (Thrun 1996; Chen y Liu 2017, con sus cinco pilares: aprendizaje continuo, acumulación/mantenimiento de una *knowledge base*, uso del conocimiento pasado, descubrir tareas nuevas, aprender mientras se trabaja), *transfer learning*, *multi-task learning*, *online learning*, *few-shot learning*, *curriculum learning* y *open world learning*. La tesis transversal: **CL debería definirse como un superconjunto** de estos paradigmas, no como un nicho que solo combate el olvido.

## 3. Contribución central

La contribución de Mundt et al. es triple:

1. **Una taxonomía visual unificada (Figura 4)** que organiza, en un solo diagrama, los métodos neuronales de los **tres campos** —continual learning, active learning y open set recognition— mostrando sus nodos y subnodos. Para CL la taxonomía replica y refina la categorización estándar (regularización / rehearsal / arquitectura), añadiendo una cuarta categoría de **enfoques combinados** (donde vive iCaRL).
2. **Un argumento de "lecciones olvidadas"**: que active learning (cómo *consultar* qué datos incluir) y open set recognition (cómo *rechazar* lo desconocido) ya resolvieron parcialmente problemas que el CL profundo está reencontrando a ciegas, y que **open set recognition es la interfaz natural** entre active learning y continual learning. La conjetura: solo combinando los tres se obtiene un sistema robusto en el mundo abierto.
3. **Evidencia empírica de respaldo** (Sección 4): no una evaluación masiva tipo *leaderboard* de todos los métodos de CIL, sino un conjunto de experimentos sobre MNIST/CIFAR10/CIFAR100/AudioMNIST que aíslan cuatro factores que las comparaciones habituales descuidan —**selección de exemplars/core sets**, **consultas activas**, **robustez a corrupciones** y **orden de tareas/currículo**— usando como vehículo un framework variacional Bayesiano propio (extensión de Mundt et al. 2019, 2022). Los autores recalcan que *no* proponen ese framework como solución universal: lo usan para *ilustrar* la importancia de los puntos de vista del paper.

A diferencia del survey de Masana et al. —que sí construye un **protocolo de evaluación único y comparable** para correr cabeza a cabeza los métodos de class-incremental learning (EWC, LwF, iCaRL, EEIL, BiC, LUCIR…) y aislar el *task-recency bias*— Mundt et al. es **más crítico-conceptual que comparativo**: prioriza reconectar campos por sobre rankear algoritmos.

## 4. Método: la taxonomía de familias contra el olvido

Esta es la sección de mayor valor para la Clase 32. El paper categoriza los métodos de CL en tres familias (más una combinada), cada una subdividida.

### 4.1. Regularización

Buscan equilibrar entre **proteger** representaciones ya aprendidas y dar **flexibilidad** para codificar lo nuevo —el dilema estabilidad-plasticidad de la neurociencia. Dos subgrupos:

- **Estructural** (protege parámetros directamente): **EWC** (Kirkpatrick et al. 2017) estima la importancia de cada parámetro vía **información de Fisher** y penaliza los cambios sobre los más específicos de tareas pasadas. **Synaptic Intelligence (SI)** (Zenke et al. 2017) y **Memory Aware Synapses (MAS)** (Aljundi et al. 2018) equipan cada parámetro ("sinapsis") con medidas de importancia acumuladas durante el entrenamiento. **RWalk** (Chaudhry et al. 2018) generaliza EWC+SI combinando Fisher y la trayectoria de optimización en una variedad de Riemann. **IMM** (Lee et al. 2017) matchea los momentos de las posteriors de las tareas; **UCL/UCB** (Ahn et al. 2019; Ebrahimi et al. 2020) usan incertidumbre Bayesiana para regularizar online.
- **Funcional** (preserva la *salida* del modelo, vía destilación): **LwF** (Learning without Forgetting, Li y Hoiem 2016) usa la destilación de conocimiento (Hinton et al. 2014) — guarda las predicciones (*soft targets*) del modelo viejo sobre los datos nuevos y regulariza para preservarlas, aun cuando sean "nonsensical" porque las clases nuevas todavía no se predicen bien. **EBLL** (Rannen et al. 2017) lleva la destilación a reconstrucciones de autoencoder. El paper observa que la destilación "rara vez se usa en aislamiento": casi siempre aparece combinada con otros mecanismos.

### 4.2. Rehearsal (memoria / repaso)

Preservan información **replayando** datos de tareas vistas. Almacenar y reproducir *todo* resolvería el problema trivialmente pero a costo de memoria intolerable; el núcleo del enfoque es hallar un **subconjunto** que aproxime bien la distribución observada —la **selección de exemplars** o construcción de un **core set**. Inspiración biológica explícita: *complementary learning systems* (hipocampo-neocórtex), ciclos sueño-vigilia. Dos subgrupos:

- **Exemplar rehearsal** (memoria episódica de datos reales): **GeppNet** (sistema dual de memoria corto/largo plazo), **SER** (selección de experiencias sorprendentes vs. recompensantes), **GEM** (Gradient Episodic Memory, Lopez-Paz y Ranzato 2017) que replaya con la restricción de que los gradientes nuevos no entren en conflicto con los de tareas previas, y su versión eficiente **A-GEM** (Chaudhry et al. 2019). **BiC** (Bias Correction, Wu et al. 2019) repasa exemplars *y además corrige el sesgo de la capa de clasificación*.
- **Generative rehearsal** (los datos repasados se *generan*): desde el *pseudo-rehearsal* de Robins (1995) y las redes reverberantes (Ans y Rousset 1997), hasta **Deep Generative Replay (DGR)** (Shin et al. 2017) que entrena una GAN separada para generar muestras de tareas pasadas, **RfF** (van de Ven y Tolias 2018) con un único modelo que clasifica y genera, e **ILCAN** (que repasa *embeddings* en vez de píxeles).

### 4.3. Arquitectura

Mitigan el olvido **modificando la arquitectura**; son casi por definición complementarios a los demás. Dos subgrupos:

- **Capacidad fija** (enrutamiento de información específico por tarea sobre una red sobre-parametrizada): *activation sharpening* (French 1992), **PathNet** (Fernando et al. 2017, congela pathways útiles vía algoritmo genético), **Piggyback** (Mallya et al. 2018) y **HAT** (Hard Attention to the Task, Serra et al. 2018) que aprenden **máscaras binarias** para *gatear* la propagación por la red, y la variante Bayesiana UCB-P.
- **Crecimiento dinámico** (añaden capacidad explícitamente, inspirado en la *neurogénesis*): desde *dynamic node creation* (Ash 1989) hasta **Progressive Neural Networks (PNN)** (Rusu et al. 2016), ExpertGate, **DEN** (Dynamically Expandable Networks, Yoon et al. 2018), NDL, RCL y Learn-to-Grow (que castean el crecimiento como meta-aprendizaje).

### 4.4. Combinados

Métodos que mezclan familias. El ejemplo arquetípico —y el más citado— es **iCaRL** (Rebuffi et al. 2017), que **acopla regularización por destilación con rehearsal de exemplars seleccionados por un procedimiento *greedy* de herding** (Welling 2009): los exemplars se eligen uno a uno de modo que cada adición aproxime mejor la media del *embedding* de la clase. **VCL** (Nguyen et al. 2018) funde memoria episódica con regularización de parámetros desde inferencia Bayesiana aproximada; **FearNet** critica la dependencia de iCaRL de la *cantidad* de datos almacenados y añade rehearsal generativo. El paper resalta que las combinaciones "crecen muy rápidamente" — y argumenta que combinar no solo es ventajoso sino "concebiblemente una necesidad".

## 5. Resultados empíricos: qué gana y bajo qué condiciones

Los experimentos (Sección 4) no buscan coronar un método sino **mostrar que factores rutinariamente ignorados cambian las conclusiones**. Datasets: MNIST, CIFAR10, CIFAR100 y AudioMNIST.

- **El balanceo de mini-batches es esencial.** En *split* MNIST/CIFAR10 (clases introducidas de a pares, reteniendo pocos exemplars de las viejas), comparar trabajos de core set *solo porque usaron un tamaño de core set similar* puede ser "comparar peras con manzanas": hay **brechas de más del 5%** según cómo se muestrea el mini-batch (Figura 9). La conclusión metodológica es exactamente la del survey de Masana: **sin un protocolo unificado, los números no son comparables.**
- **La estrategia de selección de exemplars manda.** Comparan seis estrategias de construcción de core set (Figura 10): *random*, *greedy k-center* (como en VCL), *input k-means*, *latent k-means*, **latent herding** (la adaptación al espacio latente del herding de iCaRl/BiC) y su propuesta *latent EVT*. Hallazgos: el muestreo aleatorio tiene varianza enorme; el k-center *greedy* falla en datos reales (CIFAR10) porque optimiza una cobertura por distancias máximas sin replicar la densidad ni manejar outliers; el **herding** parte muy bien pero su brecha crece con core sets grandes porque "elige muestras cada vez más redundantes" (su objetivo de aproximar la media no busca diversidad). Esto matiza el rol del herding —pieza central de iCaRL/BiC—: es bueno, pero no domina en todos los regímenes.
- **El bias correction y el data augmentation importan, pero la robustez es el talón de Aquiles.** Al inyectar **corrupciones de imagen** naturales, *todas* las estrategias de core set salvo la propuesta colapsan (Tablas 1 y 2): los métodos que asumen un *pool* limpio y completo incluyen datos corruptos/no representativos en el core set y degradan. El mensaje: las comparaciones de mundo cerrado esconden fragilidad.
- **El orden de tareas y el currículo mueven la aguja ~10%.** En CIFAR100 (y análogamente en AudioMNIST con generative replay), la diferencia de accuracy entre **distintos órdenes de tareas** llega a **~10%** (Figura 14). Es decir: el *mismo* método con el *mismo* presupuesto de memoria puede verse mucho mejor o peor según el orden — un grado de libertad que casi ningún benchmark reporta. El paper esboza currículos ("más inliers / tareas similares primero" vs. variedad) como variable de diseño legítima.

En suma: **el herding y el bias correction son herramientas valiosas pero no universalmente ganadoras; el balanceo de batch, el data augmentation, la robustez a corrupciones y el orden de tareas son factores de primer orden** que cualquier evaluación honesta debe controlar.

## 6. Limitaciones reconocidas

- **No es la evaluación comparativa masiva que la Clase 32 podría querer.** A diferencia del survey de Masana et al. —que corre decenas de métodos de class-incremental bajo un protocolo único y reporta tablas comparables— aquí la evidencia empírica está **canalizada por un único framework** (el VAE Bayesiano propio de los autores). Los propios autores admiten que *no* proponen ese framework como solución universal o única, sino como ilustración.
- **Alcance auto-limitado.** El paper se declara explícitamente una *visión panorámica* ("bird's-eye view") y crítica, que renuncia a las "elaboraciones largas sobre matices algorítmicos y detalles matemáticos no esenciales". Para las matemáticas finas de cada método (la pérdida de EWC, el objetivo de iCaRL) hay que ir a los papers originales.
- **Sesgo hacia la agenda open-world.** Buena parte de la energía va a *open set recognition* y *active learning*; el lector que solo busca "el ranking de métodos contra el olvido" encontrará la taxonomía completa pero no el veredicto cuantitativo cabeza a cabeza.
- **Escala de los experimentos.** MNIST/CIFAR/AudioMNIST son benchmarks pequeños/medianos; las conclusiones sobre core sets y orden de tareas no se validan a escala ImageNet, donde el survey de Masana sí opera.

## 7. Impacto: guía de referencia del campo

Como survey crítico publicado en *Neural Networks*, su impacto es de **encuadre conceptual** más que de algoritmo: ofrece (a) la taxonomía visual de tres campos que se cita como mapa del territorio, y (b) el argumento, hoy ampliamente aceptado, de que la evaluación de mundo cerrado del CL es insuficiente y que la robustez open-world, la selección de datos (active) y el orden de tareas deben formar parte del protocolo. Junto con los surveys hermanos (Parisi et al. 2019; De Lange et al. 2021; Lesort et al. 2020; y el de **Masana et al. 2020** sobre class-incremental learning específicamente), constituye la **literatura de referencia que organiza el campo** que la Clase 32 estudia.

## 8. Conexión con la Clase 32 (Olvido Catastrófico)

Este survey es la **referencia que da el panorama completo** de los métodos que la clase organiza en tres familias:

- **Regularización** → la clase cubre [EWC](/papers/ewc-kirkpatrick-2017) (penalización por Fisher) y SI/MAS/LwF como su núcleo. El paper los ubica como *structural* (EWC/SI/MAS) y *functional* (LwF, vía destilación), exactamente el corte que la clase usa.
- **Rehearsal / memoria** → [iCaRL](/papers/icarl-rebuffi-2017) es el método combinado arquetípico (destilación + herding de exemplars). La discusión empírica del herding (§5) aterriza *por qué* iCaRL funciona y dónde se queda corto (redundancia con core sets grandes, fragilidad ante corrupciones).
- **Arquitectura** → Progressive Networks, PathNet, HAT y Piggyback como crecimiento dinámico / capacidad fija.

El puente didáctico clave es el **mensaje metodológico**: la clase no debe quedarse en "qué algoritmo es mejor" sino en "**cómo se evalúa el olvido**". Mundt et al. demuestra empíricamente que el balanceo de batch (~5% de brecha), la estrategia de selección de exemplars, la robustez a corrupciones y el orden de tareas (~10% de brecha) cambian el veredicto — la misma lección que motiva el protocolo unificado del survey de Masana. Para profundizar en el marco conceptual del campo, ver el [fundamento de aprendizaje continuo](/fundamentos/aprendizaje-continuo); para el recorrido completo de la clase, ver [Clase 32](/clases/clase-32).
