# iCaRL: Incremental Classifier and Representation Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *iCaRL: Incremental Classifier and Representation Learning*.
- **Autores:** Sylvestre-Alvise Rebuffi (University of Oxford / IST Austria), Alexander Kolesnikov, Georg Sperl y Christoph H. Lampert (IST Austria).
- **Venue:** CVPR 2017 (Conference on Computer Vision and Pattern Recognition).
- **Preprint:** arXiv:1611.07725v2 (14 abr 2017), [arxiv.org/abs/1611.07725](https://arxiv.org/abs/1611.07725).
- **Código:** [github.com/srebuffi/iCaRL](http://www.github.com/srebuffi/iCaRL) (implementaciones en Theano para iCIFAR-100 y TensorFlow para iILSVRC).

> **NOTA sobre su lugar en la Clase 32.** iCaRL **no aparece citado en las slides** de la clase de olvido catastrófico, pero es —sin exageración— **EL método clásico de aprendizaje incremental por clases** (*class-incremental learning*). Es el trabajo que en 2017 definió el problema de forma rigurosa, propuso el protocolo de evaluación que todavía se usa (entrenar en lotes de clases, medir *average incremental accuracy*) y se convirtió en el **baseline de facto** contra el que se comparan casi todos los métodos posteriores. Lo incluimos porque es la pieza que **une las dos grandes familias que la Clase 32 enseña por separado** —memoria/rehearsal y regularización/distillation— en un solo sistema coherente, y porque es el sucesor natural de LwF (Li & Hoiem, 2016) llevado al escenario multi-clase verdadero.

iCaRL es una estrategia de entrenamiento que permite aprender clasificadores **y** una representación de características de forma **simultánea e incremental**, recibiendo las clases en lotes sucesivos y sin tener nunca acceso a todos los datos de todas las clases a la vez. Su tesis central: para lograr aprendizaje incremental por clases real sobre arquitecturas profundas, no basta una sola idea; hacen falta **tres componentes que se combinan**: (1) un conjunto de *exemplars* (ejemplos representativos almacenados) por clase, seleccionados por *herding* y con un presupuesto de memoria fijo; (2) una función de pérdida que mezcla un término de **clasificación** con uno de **destilación** (estilo LwF) al incorporar clases nuevas; y (3) una regla de clasificación por **nearest-mean-of-exemplars** (NME) que reemplaza al softmax y resulta robusta frente al desbalance y a los cambios de la representación. La validación empírica en CIFAR-100 e ImageNet muestra que iCaRL aprende durante periodos largos donde finetuning, representación fija y LwF se degradan rápidamente.

## 2. Contexto histórico: el problema del aprendizaje incremental por clases

Los sistemas de visión naturales son inherentemente incrementales: un niño que visita el zoológico aprende animales nuevos sin olvidar la mascota que tiene en casa. Los sistemas artificiales de reconocimiento de objetos, en cambio, casi siempre se entrenan en modo *batch*: todas las clases se conocen de antemano y todos sus datos están disponibles al mismo tiempo. El paper define formalmente el escenario que rompe ese supuesto —*class-incremental learning*— con **tres propiedades exigibles** a cualquier algoritmo que merezca el nombre:

1. Debe ser entrenable desde un **flujo de datos** en el que ejemplos de distintas clases aparecen en distintos momentos.
2. Debe proveer **en todo momento** un clasificador multi-clase competitivo para todas las clases vistas hasta ahí.
3. Sus requisitos de cómputo y su huella de memoria deben permanecer **acotados, o crecer muy lentamente**, respecto al número de clases vistas.

La tercera condición es la que da filo al problema: descarta la solución trivial de "guardar todos los ejemplos y reentrenar un clasificador multi-clase desde cero cada vez que llegan datos nuevos". El paper observa que, pese al enorme progreso en clasificación de imágenes, en 2017 **no existía un solo algoritmo satisfactorio** de aprendizaje incremental por clases. La mayoría de las técnicas multi-clase violaban (i) o (ii) —solo manejan un número fijo de clases o necesitan todos los datos simultáneamente. El intento ingenuo de entrenar con SGD sobre el flujo incremental degrada rápidamente la *accuracy*, un efecto conocido desde McCloskey & Cohen (1989) como **olvido catastrófico** (*catastrophic forgetting* / *catastrophic interference*). Las pocas técnicas que sí cumplían las tres propiedades estaban limitadas a **representaciones de características fijas** y no podían extenderse a arquitecturas profundas que aprenden clasificador y representación a la vez.

El paper organiza el trabajo previo en dos ejes. Con **representación fija**, el clasificador *nearest-class-mean* (NCM) de Mensink et al. (2012) representa cada clase por el vector promedio de sus *features* —calculable de forma incremental sin almacenar todos los datos— pero no puede aprender una representación no lineal junto con el clasificador. Con **aprendizaje de representación**, las dos estrategias clásicas identificadas por la era conexionista para combatir el olvido son: **congelar/crecer** la red (preservar pesos viejos y añadir capacidad nueva, como los Progressive Networks de Rusu et al. o el árbol de Xiao et al. —pero esto viola la propiedad iii porque crece sin límite) y el **rehearsal** (re-estimular la red con datos antiguos además de los recientes). iCaRL adopta rehearsal y le añade destilación.

## 3. Contribución central

La contribución de iCaRL es **combinar tres componentes** que, juntos, satisfacen las tres propiedades del aprendizaje incremental por clases sobre redes profundas. Por separado ninguno basta; el aporte es la integración:

1. **Clasificación por nearest-mean-of-exemplars (NME).** Se abandona el clasificador softmax. La predicción se hace comparando el *feature* de la imagen con el prototipo (vector medio) de cada clase, calculado sobre sus exemplars almacenados.
2. **Selección priorizada de exemplars por *herding*.** Por cada clase se guarda un subconjunto pequeño de imágenes, elegidas iterativamente para que su media de *features* aproxime lo mejor posible la media de la clase completa.
3. **Aprendizaje de representación con destilación + clasificación.** Al llegar clases nuevas, la red se actualiza minimizando una pérdida que combina un término de **clasificación** (para las clases nuevas) y un término de **destilación** (para preservar el conocimiento de las clases viejas), al estilo de Learning without Forgetting.

La novedad arquitectónica es **desacoplar el rol de la red del rol del clasificador**. La red profunda se usa únicamente como **extractor de características** $\varphi: \mathcal{X} \to \mathbb{R}^d$; sus salidas sigmoidales sirven para *aprender la representación*, no para clasificar. La clasificación efectiva la hace la regla NME sobre los prototipos. Esto resuelve un problema sutil que se explica abajo: en un clasificador lineal estándar los vectores de peso $w_y$ están **desacoplados** de $\varphi$, de modo que cuando $\varphi$ cambia (al aprender clases nuevas) los $w_y$ quedan obsoletos y la red "olvida" de forma incontrolada. Los prototipos NME, en cambio, se recalculan automáticamente al cambiar la representación, haciendo al clasificador **robusto frente a los cambios de $\varphi$**.

Conviene subrayar *por qué* esa robustez importa tanto en el escenario incremental. En una red estándar la regla de predicción es $y^* = \arg\max_y g_y(x)$, y como $\arg\max_y g_y(x) = \arg\max_y w_y^\top \varphi(x)$, es equivalente a un clasificador lineal con mapa no lineal $\varphi$ y pesos $w_y$. El problema: cada vez que la representación $\varphi$ se ajusta para acomodar clases nuevas, *todos* los $w_1,\dots,w_t$ deberían actualizarse en consecuencia; si no se hace —y en el flujo incremental no hay forma de hacerlo bien sin todos los datos viejos— las salidas de la red cambian de manera incontrolada, lo que se observa exactamente como olvido catastrófico. La regla NME elimina esa fuente de inestabilidad de raíz: como el prototipo $\mu_y$ es función directa de $\varphi$, al moverse $\varphi$ los prototipos se mueven con ella de forma coherente, sin pesos rezagados. Por eso iCaRL puede permitirse cambiar la representación agresivamente entre lotes sin que el clasificador se desmorone.

## 4. Método

iCaRL opera con dos operaciones principales —clasificar y entrenar incrementalmente— apoyadas en rutinas de gestión de exemplars.

### 4.1. Clasificación: nearest-mean-of-exemplars

Para predecir la etiqueta de una imagen $x$, iCaRL calcula el prototipo de cada clase observada como la media de los *features* de sus exemplars:

$$\mu_y = \frac{1}{|P_y|} \sum_{p \in P_y} \varphi(p)$$

y asigna la clase cuyo prototipo está más cerca:

$$y^* = \arg\min_{y=1,\dots,t} \| \varphi(x) - \mu_y \|$$

Todos los vectores de características están **L2-normalizados** (y los resultados de cualquier operación sobre ellos se re-normalizan). Como se trabaja con vectores normalizados, la regla equivale a $y^* = \arg\max_y \mu_y^\top \varphi(x)$: se puede leer como una clasificación con vector de peso, pero uno que **no está desacoplado** de la representación, sino que cambia consistentemente con ella. El prototipo se inspira en NCM, con la diferencia clave de que iCaRL no puede recalcular la media verdadera de la clase (eso exigiría almacenar todos los datos), así que la **aproxima** con la media sobre un número flexible de exemplars cuidadosamente escogidos.

### 4.2. Aprendizaje de representación

Cuando llegan datos $X^s, \dots, X^t$ de clases nuevas, iCaRL actualiza $\varphi$ y los exemplars (Algoritmo 3). El procedimiento:

1. **Construye un conjunto de entrenamiento aumentado** $D$ que une los ejemplos nuevos con los exemplars almacenados de las clases viejas. (Es importante guardar exemplars como **imágenes**, no como *features*, porque estas quedan obsoletas al cambiar $\varphi$.)
2. **Almacena las salidas de la red pre-actualización** $q_i^y = g_y(x_i)$ para todas las clases viejas y todos los ejemplos de $D$ —estos son los *soft targets* de la destilación.
3. **Minimiza una pérdida que combina clasificación y destilación**:

$$\ell(\Theta) = -\sum_{(x_i,y_i)\in D}\left[\sum_{y=s}^{t} \delta_{y=y_i}\log g_y(x_i) + \delta_{y\neq y_i}\log(1-g_y(x_i)) + \sum_{y=1}^{s-1} q_i^y \log g_y(x_i) + (1-q_i^y)\log(1-g_y(x_i))\right]$$

El primer bloque (clases $s,\dots,t$) es el **término de clasificación**: empuja a la red a emitir el indicador de clase correcto para las clases nuevas. El segundo bloque (clases $1,\dots,s-1$) es el **término de destilación**: empuja a reproducir los *scores* sigmoidales que la red emitía antes de la actualización, preservando la información discriminativa previa. Las salidas usan **cross-entropy binaria** por nodo sigmoidal (no softmax), lo que —observa el paper— permite tasas de aprendizaje grandes. La diferencia con LwF (Li & Hoiem, 2016) está en el paso de predicción: LwF fue diseñado para multi-*task* (multi-dataset, cada clasificador evaluado solo en su dataset), mientras iCaRL ataca el caso multi-clase verdadero donde un solo clasificador debe elegir entre todas las clases vistas.

### 4.3. Gestión de exemplars: herding y reducción

iCaRL respeta un **presupuesto de memoria fijo** $K$: el número total de imágenes-exemplar nunca lo excede. Con $t$ clases vistas, asigna $m = K/t$ exemplars por clase (todas tratadas por igual). Dos rutinas lo gestionan:

- **Construcción por *herding* (Algoritmo 4).** Los exemplars $p_1,\dots,p_m$ se seleccionan **iterativamente**: en cada paso se añade el ejemplo que hace que la media de *features* sobre los exemplars elegidos **mejor aproxime la media de la clase completa**. El resultado es una **lista priorizada** —el orden importa, los primeros son más representativos. *Herding* (Welling, 2009) logra buena aproximación con menos muestras que el submuestreo aleatorio.
- **Reducción (Algoritmo 5).** Cuando llegan clases nuevas y baja $m$, reducir un conjunto de $m'$ a $m$ exemplars es trivial: se **descartan los últimos** y se conservan $p_1,\dots,p_m$. Esto funciona precisamente porque la construcción es priorizada: cualquier prefijo de la lista sigue siendo una buena aproximación de la media. La estrategia de remoción es **independiente de los datos** (no necesita recalcular la media de clase, que ya no está disponible), lo cual encaja exactamente con el escenario incremental: la selección por clase se hace una sola vez —cuando la clase se ve por primera vez— y luego solo se llama a la reducción, que no requiere acceso a datos antiguos.

## 5. Experimentos

El paper propone un **protocolo de evaluación** que se volvió estándar: arreglar un orden aleatorio fijo de clases, entrenar en lotes sucesivos y, tras cada lote, evaluar sobre las clases vistas hasta ahí. El resumen en un número es la **average incremental accuracy** (promedio de las *accuracies* tras cada lote).

- **iCIFAR-100:** CIFAR-100 entrenado en lotes de 2, 5, 10, 20 o 50 clases. ResNet de 32 capas (Theano), $K = 2000$ exemplars, 70 épocas por paso. Promediado sobre 10 órdenes de clases distintos.
- **iILSVRC:** ImageNet ILSVRC 2012 en dos variantes —*small* (100 clases en lotes de 10) y *full* (1000 clases en lotes de 100). ResNet de 18 capas (TensorFlow), $K = 20000$. Medida: top-5 accuracy.

**Comparación con baselines.** iCaRL se compara contra *finetuning* (entrena sin ninguna medida anti-olvido), *fixed representation* (congela la representación tras el primer lote y solo entrena pesos de clases nuevas) y **LwF.MC** (la versión multi-clase de Learning without Forgetting: usa destilación pero **sin exemplars**, y clasifica con las salidas de la red). Resultado central (Figura 2): **iCaRL supera claramente a todos**, y la ventaja crece cuanto más incremental es el escenario (lotes más pequeños). LwF.MC es siempre segundo, salvo en iILSVRC-full donde conviene la representación fija. *Finetuning* es siempre el peor, confirmando que el olvido catastrófico es el problema dominante. Como referencia, la misma red entrenada en *batch* con todos los datos alcanza 68.6% de *accuracy* multi-clase.

**Matrices de confusión (Figura 3) — el hallazgo más ilustrativo.** Revelan el *sesgo* de cada método: iCaRL produce una matriz **homogénea** sobre todas las clases (sin sesgo hacia clases tempranas o tardías —no sufre olvido catastrófico); LwF.MC tiende a predecir clases de **lotes recientes**; la representación fija prefiere clases del **primer lote**; y *finetuning* predice **exclusivamente** clases del último lote —"simplemente olvidó que las clases anteriores existen".

**Análisis diferencial (Sección 4.2, Tabla 1).** Tres híbridos aíslan el aporte de cada componente. *hybrid1* aprende la representación como iCaRL pero clasifica con las salidas de la red; *hybrid2* usa exemplars para clasificar pero sin destilación; *hybrid3* usa exemplars solo en el aprendizaje de representación. Los híbridos caen entre iCaRL y LwF.MC, **confirmando que los tres componentes contribuyen**. Hallazgos finos: la regla mean-of-exemplars es más ventajosa en lotes pequeños (más actualizaciones de representación); para lotes muy pequeños la destilación puede incluso **perjudicar**; y comparar *hybrid3* con LwF.MC muestra la efectividad de los exemplars para frenar el olvido. La Tabla 1b compara NME contra recalcular el NCM verdadero tras cada actualización (que exigiría guardar todos los datos): la diferencia es **mínima**, confirmando que el *herding* identifica exemplars representativos. La Figura 4 muestra que todos los métodos mejoran con presupuesto $K$ mayor, y que con $\geq 1000$ prototipos NME iguala a NCM.

## 6. Limitaciones reconocidas

- **Necesita almacenar exemplars.** El componente que más aporta —los exemplars— es también el que rompe la pureza del "no guardar datos". iCaRL acota la memoria a $K$ imágenes, pero requiere conservar datos crudos, lo que es problemático cuando hay restricciones de **privacidad** (el paper menciona explícitamente este escenario como trabajo futuro, sugiriendo codificar las clases viejas vía autoencoder en vez de imágenes).
- **Brecha frente al entrenamiento batch.** El propio paper admite que el aprendizaje incremental por clases "está lejos de resolverse": el rendimiento de iCaRL sigue por debajo del entrenamiento *batch* con todos los datos disponibles (p. ej. ~64% vs 68.6% en iCIFAR-100 con lotes de 10).
- **Destilación puede perjudicar en lotes muy pequeños.** El análisis diferencial muestra que con lotes de 2 clases la destilación llega a bajar la *accuracy* frente a usar solo prototipos.
- **Número finito de clases.** Bajo un presupuesto $K$ fijo, eventualmente $m = K/t$ se acerca a su mínimo y solo un número finito de clases puede aprenderse, salvo que se permita crecer los recursos.

## 7. Impacto

iCaRL es el **baseline de facto del aprendizaje incremental por clases**. Su contribución duradera es doble: (1) formalizó el problema con las tres propiedades y aportó el **protocolo de evaluación** (lotes de clases + *average incremental accuracy* + matrices de confusión como diagnóstico de sesgo) que la literatura posterior adoptó; y (2) demostró que la **combinación de memoria (rehearsal) + regularización (destilación) + un clasificador robusto a la deriva de la representación (NME)** supera a cualquiera de esas ideas por separado. La hipótesis que el paper deja planteada —que combinar parámetros de red con exemplars almacenados es sorprendentemente potente, y que "muchas miles de imágenes (comprimidas) caben en una memoria comparable al tamaño de las redes actuales"— anticipó toda la línea de métodos de *rehearsal* y *replay* que dominó el aprendizaje continuo en los años siguientes.

## 8. Conexión con la Clase 32 (Olvido catastrófico)

La Clase 32 enseña las grandes familias de soluciones al olvido catastrófico —típicamente separadas en **regularización** (penalizar el cambio de pesos/salidas importantes: EWC, LwF, destilación), **memoria/rehearsal** (almacenar o regenerar datos antiguos) y **arquitecturas dinámicas** (crecer la red). iCaRL es valioso para la clase precisamente porque **no escoge una familia: las fusiona**.

- **Familia memoria + familia regularización en un solo método.** El *exemplar set* + rehearsal es memoria; la pérdida de **destilación** es regularización funcional (penaliza que las salidas de las clases viejas se desvíen de lo que la red predecía antes). iCaRL muestra empíricamente —vía los híbridos— que ambas contribuyen y que su combinación es lo que vence al olvido. Es el ejemplo canónico para discutir en clase por qué "memoria *o* regularización" es un falso dilema.

- **Sucesor de LwF en el escenario class-incremental.** LwF (Li & Hoiem, 2016) introdujo la destilación para aprendizaje continuo, pero en un marco multi-*task*. iCaRL toma esa misma destilación, la lleva al caso **multi-clase verdadero** (un único clasificador sobre todas las clases) y le añade los dos ingredientes que LwF no tenía: exemplars y la regla NME. El experimento LwF.MC del paper es, de hecho, "LwF sin exemplars", y queda sistemáticamente por debajo de iCaRL —lo que cuantifica *cuánto* aporta sumar memoria a la regularización.

- **El mecanismo del olvido, hecho visible.** Las matrices de confusión de la Figura 3 son material didáctico directo: muestran que *finetuning* colapsa hacia el último lote (olvido catastrófico puro), que congelar la representación sesga hacia el primer lote (rigidez), y que iCaRL logra el balance —la tensión *estabilidad–plasticidad* que la clase discute en abstracto, hecha gráfico.

- **El presupuesto de memoria como perilla de diseño.** La Clase 32 plantea que el aprendizaje continuo siempre negocia recursos contra rendimiento. iCaRL hace ese trade-off explícito y medible: el parámetro $K$ (memoria total de exemplars) es una perilla que el practicante ajusta según su aplicación. La Figura 4 cuantifica que más memoria siempre ayuda, pero con retornos decrecientes —un punto directamente conectable con las discusiones del curso sobre cuándo invertir en memoria/replay vale la pena frente a métodos puramente regularizadores que no guardan datos.

- **Herding como puente a la familia de métricas.** La selección de exemplars por *herding* —elegir un subconjunto cuya media de *features* aproxime la media de la clase— emparenta a iCaRL con las ideas de prototipos y aprendizaje por métrica (NCM, Prototypical Networks). Para un estudiante que viene de las clases de meta-aprendizaje, ver el mismo concepto de "clasificar por cercanía al prototipo de clase" reaparecer aquí, ahora al servicio de combatir el olvido, refuerza que las técnicas del curso no viven en compartimentos estancos.

Enlaces internos del curso: fundamento transversal de [aprendizaje continuo](/fundamentos/aprendizaje-continuo), la [Clase 32](/clases/clase-32) sobre olvido catastrófico, y el paper predecesor [LwF (Li & Hoiem, 2016)](/papers/lwf-li-2016).
