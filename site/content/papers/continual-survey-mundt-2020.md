---
title: "A Wholistic View of Continual Learning (survey, 2020)"
weight: 367
math: true
---

{{< paper-card
    title="A Wholistic View of Continual Learning with Deep Neural Networks: Forgotten Lessons and the Bridge to Active and Open World Learning"
    authors="Martin Mundt, Yongwon Hong, Iuliia Pliushch, Visvanathan Ramesh"
    year="2020"
    venue="Neural Networks 2023 / arXiv"
    pdf="/papers/continual-survey-mundt-2020.pdf"
    arxiv="2009.01797" >}}
Survey crítico del aprendizaje continuo (continual learning, CL) que es la **referencia panorámica** de la [Clase 32](/clases/clase-32). No propone un algoritmo nuevo: ofrece la **taxonomía canónica** de las familias de métodos contra el olvido catastrófico (regularización, rehearsal/memoria, arquitectura) y denuncia el **estrechamiento del campo** —la fijación en monitorear el olvido sobre benchmarks clásicos partidos en tareas, bajo una suposición de mundo cerrado. Su tesis de las "lecciones olvidadas" tiende un puente hacia el **active learning** (qué datos consultar) y el **open set / open world recognition** (cómo rechazar lo desconocido), y lo respalda con experimentos sobre selección de exemplars/core sets, robustez a corrupciones y orden de tareas.
{{< /paper-card >}}

> **Nota sobre la procedencia.** El slide de la Clase 32 asoció el arXiv **2009.01797** al "survey de Masana 2020" sobre class-incremental learning, pero ese identificador corresponde **en realidad** a Mundt et al. (el survey de Masana et al. es arXiv:2010.15277). Felizmente ambos son surveys de continual learning con la misma taxonomía de familias y los mismos métodos canónicos (EWC, SI, LwF, iCaRL, GEM, BiC), de modo que este paper cumple el mismo rol pedagógico: dar el mapa completo del campo que la clase organiza.

---

## Contexto

El **olvido catastrófico** (*catastrophic forgetting*, McCloskey y Cohen 1989) es el efecto por el cual una red que actualiza sus pesos con gradiente estocástico, al recibir datos que desplazan la distribución, guía sus representaciones hacia la tarea actual y **sobrescribe** lo aprendido. El aprendizaje continuo estudia cómo aprender de datos que llegan en el tiempo *sin* acumularlos todos, preservando el conocimiento previo.

El diagnóstico central de Mundt et al. no es la falta de algoritmos, sino la **fragmentación y el reduccionismo de la evaluación**. Tres problemas concretos:

1. **El mundo cerrado omnipresente.** Casi todos los benchmarks de CL evalúan sobre datos garantizadamente de la misma distribución que el entrenamiento. Pero es sabido desde hace décadas que las redes neuronales son erróneamente *overconfident* ante datos desconocidos o corruptos (Matan et al. 1990), y "se rompen inmediatamente" ante corrupciones menores en despliegue (Hendrycks y Dietterich 2019). Se ataca el olvido con esmero mientras se ignora la robustez en el mundo abierto.
2. **Datasets secuencializados sin entender su naturaleza.** El grueso de los trabajos toma benchmarks de clasificación (MNIST, CIFAR, ImageNet), parte las clases en conjuntos disjuntos y los muestra en secuencia. Preguntas sobre el **efecto del orden de tareas** o el **solapamiento entre tareas** se pasan por alto "en favor de retener comparabilidad sobre un benchmark".
3. **Métricas heredadas del aprendizaje aislado.** El protocolo habitual extrae métricas de una matriz de confusión convencional (*forward/backward transfer*, *amount of forgetting*, consumo de memoria, *task boundaries*) y desatiende el orden de tareas, la elección de datos y cualquier forma de robustez open-world.

El paper sitúa el CL en una constelación de paradigmas vecinos —*lifelong learning*, *transfer learning*, *multi-task*, *online*, *few-shot*, *curriculum* y *open world learning*— y defiende que el aprendizaje continuo **debería definirse como un superconjunto** de ellos, no como un nicho que solo combate el olvido.

## Contribución central

La aportación es triple:

1. **Una taxonomía visual unificada** que organiza en un solo diagrama los métodos neuronales de los **tres campos** —continual learning, active learning y open set recognition—. Para CL refina la categorización estándar (regularización / rehearsal / arquitectura) y añade una cuarta categoría de **enfoques combinados** (donde vive iCaRL).
2. **El argumento de las "lecciones olvidadas":** el active learning (cómo *consultar* qué datos incluir) y el open set recognition (cómo *rechazar* lo desconocido) ya resolvieron parcialmente problemas que el CL profundo reencuentra a ciegas; el open set recognition es la **interfaz natural** entre active y continual learning. La conjetura: solo combinando los tres se obtiene un sistema robusto en el mundo abierto.
3. **Evidencia empírica de respaldo:** no un *leaderboard* masivo, sino experimentos sobre MNIST/CIFAR10/CIFAR100/AudioMNIST que aíslan cuatro factores que las comparaciones habituales descuidan —selección de exemplars/core sets, consultas activas, robustez a corrupciones y orden de tareas— usando como vehículo un framework variacional Bayesiano propio (que los autores explícitamente *no* proponen como solución universal, solo como ilustración).

A diferencia del survey de Masana et al. —que sí construye un **protocolo de evaluación único y comparable** para correr cabeza a cabeza los métodos de class-incremental learning—, Mundt et al. es **más crítico-conceptual que comparativo**: prioriza reconectar campos por sobre rankear algoritmos.

## La taxonomía de familias contra el olvido

Esta es la sección de mayor valor para la clase. El paper categoriza los métodos en tres familias (más una combinada).

### Regularización

Equilibran entre **proteger** lo aprendido y dar **flexibilidad** para lo nuevo —el dilema estabilidad-plasticidad. Dos subgrupos:

- **Estructural** (protege parámetros directamente): [**EWC**](/papers/ewc-kirkpatrick-2017) (Kirkpatrick et al. 2017) estima la importancia de cada parámetro vía **información de Fisher** y penaliza cambios sobre los más específicos de tareas pasadas. **Synaptic Intelligence (SI)** (Zenke et al. 2017) y **Memory Aware Synapses (MAS)** (Aljundi et al. 2018) acumulan medidas de importancia por "sinapsis". **RWalk** generaliza EWC+SI sobre una variedad de Riemann; **IMM** matchea momentos de las posteriors; **UCL/UCB** usan incertidumbre Bayesiana.
- **Funcional** (preserva la *salida* del modelo vía destilación): **LwF** (Learning without Forgetting, Li y Hoiem 2016) guarda los *soft targets* del modelo viejo sobre los datos nuevos y regulariza para preservarlos. **EBLL** lleva la destilación a reconstrucciones de autoencoder. El paper observa que la destilación "rara vez se usa en aislamiento": casi siempre aparece combinada.

### Rehearsal (memoria / repaso)

Preservan información **replayando** datos de tareas vistas. Almacenar y reproducir *todo* resolvería el problema pero a costo de memoria intolerable; el núcleo es hallar un **subconjunto** que aproxime la distribución observada —la **selección de exemplars** o construcción de un **core set**. Inspiración biológica: *complementary learning systems* (hipocampo-neocórtex). Dos subgrupos:

- **Exemplar rehearsal** (memoria episódica de datos reales): **GEM** (Gradient Episodic Memory, Lopez-Paz y Ranzato 2017) replaya con la restricción de que los gradientes nuevos no entren en conflicto con los previos, y su versión eficiente **A-GEM**. **BiC** (Bias Correction, Wu et al. 2019) repasa exemplars *y además* corrige el sesgo de la capa de clasificación.
- **Generative rehearsal** (los datos repasados se *generan*): desde el *pseudo-rehearsal* de Robins (1995) hasta **Deep Generative Replay** (Shin et al. 2017), que entrena una GAN para generar muestras de tareas pasadas, e **ILCAN** (repasa *embeddings* en vez de píxeles).

### Arquitectura

Mitigan el olvido **modificando la arquitectura**; son casi por definición complementarios a las otras familias. Dos subgrupos:

- **Capacidad fija** (enrutamiento por tarea sobre una red sobre-parametrizada): **PathNet** (Fernando et al. 2017, congela pathways útiles), **Piggyback** y **HAT** (Hard Attention to the Task, Serra et al. 2018), que aprenden **máscaras binarias** para gatear la propagación.
- **Crecimiento dinámico** (añaden capacidad, inspirado en la *neurogénesis*): **Progressive Neural Networks** (Rusu et al. 2016), **DEN** (Dynamically Expandable Networks) y Learn-to-Grow.

### Combinados

Mezclan familias. El ejemplo arquetípico es [**iCaRL**](/papers/icarl-rebuffi-2017) (Rebuffi et al. 2017), que **acopla destilación con rehearsal de exemplars** seleccionados por un procedimiento *greedy* de **herding** (Welling 2009): cada exemplar se elige de modo que su adición aproxime mejor la media del *embedding* de la clase. **VCL** funde memoria episódica con regularización Bayesiana. El paper argumenta que combinar no solo es ventajoso sino "concebiblemente una necesidad".

## Experimentos: qué gana y bajo qué condiciones

Los experimentos no buscan coronar un método, sino **mostrar que factores rutinariamente ignorados cambian las conclusiones** (datasets: MNIST, CIFAR10, CIFAR100, AudioMNIST).

- **El balanceo de mini-batches es esencial.** En *split* MNIST/CIFAR10 hay **brechas de más del 5%** según cómo se muestrea el mini-batch. Comparar trabajos de core set solo porque usaron tamaños similares puede ser "comparar peras con manzanas": **sin un protocolo unificado, los números no son comparables** (la misma lección que el survey de Masana).
- **La estrategia de selección de exemplars manda.** Comparan seis estrategias de core set (*random*, *greedy k-center*, *input/latent k-means*, **latent herding**, *latent EVT*). El muestreo aleatorio tiene varianza enorme; el k-center *greedy* falla en datos reales (CIFAR10) porque optimiza cobertura por distancias máximas sin replicar la densidad; el **herding** parte muy bien pero su brecha crece con core sets grandes porque "elige muestras cada vez más redundantes" (su objetivo de aproximar la media no busca diversidad). Esto matiza el rol del herding —pieza central de iCaRL/BiC—: es bueno, pero no domina en todos los regímenes.
- **La robustez es el talón de Aquiles.** Al inyectar **corrupciones de imagen**, *todas* las estrategias de core set salvo la propuesta colapsan: los métodos que asumen un *pool* limpio incluyen datos corruptos en el core set y degradan. Las comparaciones de mundo cerrado esconden fragilidad.
- **El orden de tareas mueve la aguja ~10%.** En CIFAR100 la diferencia de accuracy entre **distintos órdenes de tareas** llega a **~10%**: el mismo método con el mismo presupuesto de memoria puede verse mucho mejor o peor según el orden —un grado de libertad que casi ningún benchmark reporta. El paper esboza currículos ("inliers / tareas similares primero" vs. variedad) como variable de diseño legítima.

En suma: **el herding y el bias correction son herramientas valiosas pero no universalmente ganadoras; el balanceo de batch, el data augmentation, la robustez a corrupciones y el orden de tareas son factores de primer orden** que cualquier evaluación honesta debe controlar.

## Limitaciones reconocibles

- **No es la evaluación comparativa masiva** que un curso podría esperar: la evidencia está canalizada por un único framework (el VAE Bayesiano propio de los autores), no por decenas de métodos bajo protocolo común.
- **Alcance auto-limitado:** se declara una *visión panorámica* ("bird's-eye view") crítica y renuncia a las matemáticas finas de cada método (la pérdida de EWC, el objetivo de iCaRL) —para ello hay que ir a los papers originales.
- **Sesgo hacia la agenda open-world:** buena parte de la energía va a open set recognition y active learning; quien solo busca "el ranking de métodos contra el olvido" encontrará la taxonomía completa pero no el veredicto cuantitativo cabeza a cabeza.
- **Escala de los experimentos:** MNIST/CIFAR/AudioMNIST son benchmarks pequeños/medianos; las conclusiones sobre core sets y orden no se validan a escala ImageNet.

## Por qué importa para la Clase 32

Este survey es la **referencia que da el panorama completo** de los métodos que la [Clase 32](/clases/clase-32) organiza en tres familias:

- **Regularización** → la clase cubre [EWC](/papers/ewc-kirkpatrick-2017) (penalización por Fisher) y SI/MAS/LwF. El paper los ubica como *structural* (EWC/SI/MAS) y *functional* (LwF), exactamente el corte que la clase usa.
- **Rehearsal / memoria** → [iCaRL](/papers/icarl-rebuffi-2017) es el método combinado arquetípico (destilación + herding). La discusión empírica del herding aterriza *por qué* funciona y dónde se queda corto (redundancia con core sets grandes, fragilidad ante corrupciones).
- **Arquitectura** → Progressive Networks, PathNet, HAT y Piggyback.

El puente didáctico clave es el **mensaje metodológico**: la clase no debe quedarse en "qué algoritmo es mejor" sino en "**cómo se evalúa el olvido**". Mundt et al. demuestra que el balanceo de batch (~5%), la estrategia de selección de exemplars, la robustez a corrupciones y el orden de tareas (~10%) cambian el veredicto. Para el marco conceptual del campo, ver el [fundamento de aprendizaje continuo](/fundamentos/aprendizaje-continuo).
