# Overcoming Catastrophic Forgetting in Neural Networks (EWC) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Overcoming catastrophic forgetting in neural networks*.
- **Autores:** James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis y Dharshan Kumaran (todos de **DeepMind**, Londres); Claudia Clopath (Bioengineering, Imperial College London); y Raia Hadsell (DeepMind).
- **Venue:** *Proceedings of the National Academy of Sciences* (PNAS), 2017. **Preprint:** arXiv:1612.00796v2 (25 ene 2017), [arxiv.org/abs/1612.00796](https://arxiv.org/abs/1612.00796).
- **Filiación dominante:** DeepMind — el mismo grupo que produjo el DQN de Atari (Mnih et al., 2015), lo que explica el uso de Atari como banco de pruebas y la mezcla de perspectivas de neurociencia (Clopath) y deep RL.

Este paper introduce **Elastic Weight Consolidation (EWC)**, el método que abrió la era moderna del aprendizaje continuo basado en **regularización**. La tesis es simple y poderosa: el olvido catastrófico —la pérdida abrupta de lo aprendido en una tarea A cuando una red se entrena después en una tarea B— no es una propiedad inevitable de los modelos conexionistas, sino un problema que se puede mitigar **ralentizando selectivamente el aprendizaje sobre los pesos que importan para las tareas anteriores**. La inspiración es neurocientífica: en el cerebro mamífero, la *consolidación sináptica* protege el conocimiento previo volviendo menos plásticas ciertas sinapsis. EWC implementa el análogo artificial de ese mecanismo mediante una penalización cuadrática sobre los pesos, ponderada por su importancia, donde la importancia se estima con la **diagonal de la matriz de información de Fisher**.

Para la Clase 32 (Olvido Catastrófico / Aprendizaje Continuo) este paper es central: la clase presenta EWC explícitamente como **el primer método de la familia de regularización** (slide "Elastic Weight Consolidation (EWC)", marcado como *Regularización*). Es el punto de partida conceptual frente a las otras dos familias —*replay/rehearsal* (iCaRL, GEM) y *arquitecturas dinámicas* (Progressive Networks, PackNet/PiggyBack)— y el ancestro directo de Synaptic Intelligence (Zenke et al., 2017) y de la formulación online de EWC.

## 2. Contexto histórico: el olvido catastrófico como barrera

El **olvido catastrófico** (catastrophic forgetting, también *catastrophic interference*) fue documentado a fines de los años 80 y 90 por McCloskey y Cohen (1989), Ratcliff (1990) y French (1999). El fenómeno es el siguiente: una red neuronal entrenada secuencialmente en una serie de tareas pierde de forma abrupta —no gradual— el desempeño sobre las tareas previas a medida que aprende las nuevas. La causa mecánica es directa: los pesos de la red que eran importantes para la tarea A son sobreescritos por el descenso de gradiente que optimiza la tarea B. Dado que la red comparte un único conjunto de parámetros entre todas las tareas, y que el gradiente de B no "sabe" nada de A, nada impide que B destruya la solución de A.

El paper sitúa esto como un obstáculo fundamental hacia la **inteligencia artificial general**: un agente inteligente debe exhibir *aprendizaje continual* —aprender tareas consecutivas sin olvidar cómo ejecutar las anteriores—, en un mundo donde las tareas no vienen etiquetadas, pueden cambiar de forma impredecible y pueden no recurrir durante largos intervalos. El estado del arte hacia 2016 evitaba el problema en vez de resolverlo: el paradigma de **multitask learning** entrelaza (interleaves) los datos de todas las tareas durante el entrenamiento, de modo que los pesos se optimizan conjuntamente y el olvido nunca ocurre. Pero esto exige tener todos los datos simultáneamente disponibles. Cuando las tareas llegan en secuencia, la única alternativa multitask es almacenar todas las experiencias en un sistema de memoria episódica y *reproducirlas* (replay) durante el entrenamiento —lo que el paper llama *system-level consolidation*—, una estrategia impráctica porque la memoria necesaria crece de forma proporcional al número de tareas.

La **inspiración neurocientífica** es el corazón conceptual del trabajo. En marcado contraste con las redes artificiales, los humanos y otros animales aprenden de forma continua. La evidencia reciente (Cichon y Gan, 2015; Hayashi-Takagi et al., 2015; Yang et al., 2009, 2014) sugiere que el neocórtex mamífero evita el olvido catastrófico protegiendo el conocimiento previo en sus circuitos. Cuando un ratón adquiere una habilidad nueva, una proporción de sinapsis excitatorias se *fortalece*, lo que se manifiesta como un aumento del volumen de las espinas dendríticas individuales. Críticamente, estas espinas dendríticas agrandadas **persisten** pese al aprendizaje posterior de otras tareas, lo que explica la retención de desempeño meses después; y cuando se "borran" selectivamente esas espinas (con técnicas optogenéticas), la habilidad correspondiente se olvida. Esto constituye evidencia causal de que la protección de sinapsis fortalecidas es crítica para la retención. Junto con modelos neurobiológicos (Fusi et al., 2005; Benna y Fusi, 2016), estos hallazgos apuntan a un proceso de **consolidación sináptica específica de tarea**: el conocimiento sobre cómo ejecutar una tarea adquirida se codifica de forma durable en una proporción de sinapsis que se vuelven menos plásticas y, por tanto, estables a lo largo del tiempo. EWC es la traducción algorítmica de ese principio.

## 3. Contribución central

La contribución de EWC es **una pérdida de regularización** que, al entrenar la tarea B, penaliza el desplazamiento de cada peso respecto a su valor óptimo en la tarea A, con una penalización *individualizada por parámetro* proporcional a su importancia para A. La analogía mecánica que da nombre al método es el **resorte (spring) elástico**: cada peso queda anclado a su valor previo θ\*_A por un resorte cuya rigidez no es uniforme, sino mayor para los pesos que más importan al desempeño de A. De ahí el nombre *elastic weight consolidation*: consolidación (volver estables los pesos importantes) implementada de forma elástica (un anclaje cuadrático suave, no una congelación dura).

El insight que hace viable este enfoque es la **sobreparametrización**: una red profunda tiene tantos parámetros que muchas configuraciones distintas de θ logran el mismo desempeño (Nielsen, 1989; Sussmann, 1992). Esto hace probable que exista una solución para la tarea B, θ\*_B, *cercana* a la solución previamente hallada para A, θ\*_A. EWC explota esa cercanía: en vez de dejar que el gradiente de B vaya a cualquier mínimo de B (destruyendo A), lo restringe a quedarse en una región de bajo error para A, centrada en θ\*_A. La Figura 1 del paper ilustra esto esquemáticamente: el gradiente de B solo (flecha azul) minimiza B pero destruye A; una restricción uniforme con el mismo coeficiente para todos los pesos (flecha verde, equivalente a regularización L2) es demasiado severa y solo permite recordar A a costa de no aprender B; EWC (flecha roja) encuentra una solución para B sin incurrir en pérdida significativa sobre A, precisamente porque calcula explícitamente cuán importante es cada peso para A.

## 4. Método: de Bayes a la penalización de Fisher

### 4.1. Justificación bayesiana

El paper justifica EWC desde una perspectiva probabilística. Optimizar los parámetros equivale a hallar sus valores más probables dados los datos D, vía la regla de Bayes:

> log p(θ|D) = log p(D|θ) + log p(θ) − log p(D)

donde log p(D|θ) es simplemente el negativo de la función de pérdida, −L(θ). Suponiendo que los datos se dividen en dos partes independientes —los de la tarea A (D_A) y los de la tarea B (D_B)— se puede reordenar la regla de Bayes:

> log p(θ|D) = log p(D_B|θ) + log p(θ|D_A) − log p(D_B)

Esta reformulación es el núcleo del argumento. El lado izquierdo sigue siendo el *posterior* de los parámetros dado todo el dataset; pero el lado derecho solo depende de la función de pérdida de la tarea B, log p(D_B|θ), más el **posterior de la tarea A**, log p(θ|D_A). Toda la información sobre la tarea A —incluyendo qué parámetros fueron importantes para ella— quedó absorbida en ese posterior p(θ|D_A). La idea elegante: **el posterior de la tarea A se convierte en el prior para aprender la tarea B**. Aprender continualmente es, formalmente, encadenar inferencias bayesianas donde la creencia acumulada sobre los parámetros se va templando (tempering) tarea a tarea.

### 4.2. Aproximación de Laplace y matriz de Fisher

El posterior verdadero p(θ|D_A) es intratable. Siguiendo la **aproximación de Laplace** de MacKay (1992), EWC lo aproxima por una **Gaussiana** centrada en los parámetros óptimos θ\*_A, con una precisión diagonal dada por la **diagonal de la matriz de información de Fisher** F. La elección de Fisher se justifica por tres propiedades clave (Pascanu y Bengio, 2013): (a) es equivalente a la segunda derivada de la pérdida cerca de un mínimo (es decir, aproxima la curvatura/Hessiano), de modo que un peso con Fisher alto es uno donde la pérdida sube rápido si se lo mueve; (b) se puede calcular **solo con derivadas de primer orden**, lo que la hace barata incluso para modelos grandes (no requiere computar el Hessiano explícito); y (c) está garantizada como semidefinida positiva, lo que asegura que la penalización es convexa y bien comportada. El método es análogo a *expectation propagation*, donde cada subtarea se ve como un factor del posterior (Eskin et al., 2004).

### 4.3. La función objetivo

Con esta aproximación, la función que EWC minimiza al entrenar la tarea B es:

> **L(θ) = L_B(θ) + Σ_i (λ/2) · F_i · (θ_i − θ\*_{A,i})²**

donde:
- **L_B(θ)** es la pérdida de la tarea B *sola* (la nueva tarea).
- El sumatorio recorre cada parámetro individual i de la red.
- **F_i** es el i-ésimo término diagonal de la matriz de Fisher de la tarea A: la importancia del peso i para A.
- **(θ_i − θ\*_{A,i})²** es el desplazamiento cuadrático del peso respecto a su valor óptimo en A.
- **λ** es un hiperparámetro escalar que fija cuán importante es la tarea vieja respecto a la nueva (la rigidez global de los resortes).

La interpretación mecánica es transparente: cada parámetro es jalado de vuelta hacia su valor anterior θ\*_{A,i} con una fuerza proporcional a F_i. Pesos con F_i alto (cruciales para A) quedan casi congelados; pesos con F_i bajo (irrelevantes para A) quedan libres para reaprender en B. Esto es exactamente la "ralentización selectiva del aprendizaje" que el resumen promete. Nótese que λ/2 es la constante del resorte cuadrático y que la penalización es una suma desacoplada por parámetro: la aproximación diagonal de Fisher ignora las correlaciones entre pesos, lo que es lo que la vuelve computacionalmente barata pero también introduce su principal debilidad teórica (ver §6).

### 4.4. Múltiples tareas

Al pasar a una tercera tarea C, EWC intenta mantener los parámetros cercanos a los óptimos de A *y* de B. Esto se puede imponer con dos penalizaciones separadas (una hacia θ\*_A, otra hacia θ\*_B), o —observando que la suma de dos penalizaciones cuadráticas es a su vez una penalización cuadrática— consolidarlas en un solo término. Esta observación es la que mantiene el costo acotado al crecer el número de tareas, aunque en la versión original cada nueva tarea añade un término de anclaje (lo que motivó la posterior *online EWC*, que mantiene una sola Gaussiana acumulada).

## 5. Experimentos

### 5.1. MNIST permutado (aprendizaje supervisado)

El primer banco de pruebas es una secuencia de tareas construidas sobre MNIST según un esquema usado en la literatura de continual learning (Srivastava et al., 2013; Goodfellow et al., 2014): **permuted MNIST**. Para cada tarea se genera una permutación aleatoria fija de los píxeles de entrada, que se aplica a todas las imágenes. Cada tarea tiene exactamente la misma dificultad que el MNIST original, pero requiere una solución distinta, porque la red ve los píxeles barajados de forma diferente. Se entrena una red totalmente conectada (ReLU) en secuencia sobre estas tareas; tras un entrenamiento fijo en cada tarea, no se permite más entrenamiento sobre ese dataset.

Los resultados (Figura 2) son la demostración fundacional del método:
- **SGD plano** (azul) sufre olvido catastrófico: al cambiar de la tarea A a la B, el desempeño en B sube empinadamente mientras el de A cae en picada, y el olvido empeora con cada tarea añadida.
- **Regularización L2** (verde, restricción cuadrática uniforme) protege A —el desempeño de A se degrada mucho menos— pero **no logra aprender B**, porque la restricción uniforme protege todos los pesos por igual y deja poca capacidad libre para B. Es la flecha verde de la Figura 1.
- **EWC** (rojo) es el único capaz de mantener un alto desempeño en las tareas viejas *y* conservar la capacidad de aprender las nuevas. Frente a SGD con *dropout* (la mejor línea base previa, que solo escalaba hasta dos permutaciones), EWC escala a un número grande de tareas con solo un crecimiento modesto de la tasa de error.

Un experimento secundario revelador (Figura 2C) mide el **solapamiento entre las matrices de Fisher** de pares de tareas en función de la profundidad de la red, para responder si EWC asigna partes completamente separadas de la red a cada tarea o si comparte representaciones. Hallazgo: cuando dos tareas son muy similares (solo unos pocos píxeles permutados), dependen de conjuntos de pesos similares en toda la red; cuando son más disímiles, la red empieza a asignar capacidad separada para cada una. Aun así, incluso con permutaciones grandes, las capas más cercanas a la salida se reutilizan para ambas tareas —reflejo de que las permutaciones cambian el dominio de entrada pero no el de salida (las etiquetas de clase son compartidas). EWC, por tanto, comparte representación donde las tareas comparten estructura, y subdivide donde no.

### 5.2. Atari 2600 secuencial (aprendizaje por refuerzo)

El segundo banco de pruebas es mucho más exigente: **aprendizaje por refuerzo profundo** con **DQN** (Mnih et al., 2015) sobre el conjunto Atari 2600 (Bellemare et al., 2013). Cada experimento consiste en diez juegos elegidos al azar entre los que DQN juega a nivel humano o superior. El agente se expone a las experiencias de cada juego por periodos extendidos; el orden de presentación es aleatorio y permite volver a juegos previos. A intervalos regulares se mide el puntaje del agente en los diez juegos sin permitir entrenamiento (test).

A diferencia de los enfoques previos de continual learning en RL —que añadían capacidad a la red (Progressive Networks, Rusu et al., 2016) o entrenaban redes separadas y luego las destilaban en una sola (Policy Distillation, Actor-Mimic)—, EWC usa **una sola red con recursos fijos** y un overhead computacional mínimo. El sistema añade dos componentes a un agente tipo Double DQN (van Hasselt et al., 2016):
1. **Un módulo de reconocimiento de tarea**, necesario porque en RL las etiquetas de tarea no se dan explícitamente. El contexto de tarea se trata como variable latente de un *Hidden Markov Model*: cada tarea se asocia a un modelo generativo de las observaciones, y se permite añadir modelos generativos nuevos si explican los datos recientes mejor que los existentes (procedimiento inspirado en el *forget me not process*, Milan et al., 2016). El modelo es bayesiano y no paramétrico: puede crecer en función de los datos observados.
2. **La penalización EWC**, calculando la diagonal de Fisher en cada cambio de tarea (100 minibatches del replay buffer) y añadiendo un término de anclaje con factor de escala λ = 400, pero **solo** a los juegos que ya hubieran acumulado al menos 20 millones de frames.

El agente mantiene además buffers de memoria de corto plazo separados por tarea (experience replay off-policy). El sistema opera así con memoria en dos escalas: a corto plazo el experience replay descorrelaciona las experiencias dentro de una tarea; a largo plazo EWC consolida el know-how entre tareas. Resultado (Figura 3): con descenso de gradiente plano el agente nunca aprende a jugar más de un juego y el puntaje humano-normalizado total queda por debajo de 1; con EWC, el agente sí aprende a jugar múltiples juegos. Proveer la etiqueta de tarea verdadera (en vez de inferirla con el HMM) solo aporta una mejora modesta, validando el módulo de reconocimiento.

### 5.3. Validación de la diagonal de Fisher

Un experimento de control (Figura 3C) prueba empíricamente la calidad de la estimación de importancia. Se entrena un agente en un solo juego (Breakout) y se perturban sus pesos con ruido gaussiano de distintas covarianzas, midiendo el efecto en el puntaje. El agente es siempre **más robusto** a perturbaciones moldeadas por el inverso de la diagonal de Fisher (azul, que imita los cambios que EWC permitiría) que a perturbaciones uniformes (negro) —lo que valida que la diagonal de Fisher es un buen estimador de la importancia de cada parámetro. Sin embargo, perturbar en el *espacio nulo* de Fisher (naranja, pesos que Fisher estima irrelevantes) tiene el mismo efecto que perturbar en el espacio inverso de Fisher, cuando en teoría no debería tener efecto alguno. Esto sugiere que el método es **sobreconfiado** sobre qué parámetros son irrelevantes: subestima la incertidumbre de los parámetros, lo que conecta directamente con sus limitaciones.

## 6. Limitaciones reconocidas

- **Aproximación diagonal de Fisher.** EWC ignora las correlaciones entre pesos al usar solo la diagonal de F (un posterior gaussiano factorizado). Esto es lo que vuelve el método lineal en el número de parámetros y de ejemplos —en contraste con ELLA (Eaton y Ruvolo, 2013), que invierte matrices de dimensión igual al número de parámetros y solo se aplicó a regresiones lineales/logísticas—, pero sacrifica la captura de interacciones entre pesos.
- **Estimación puntual de la varianza (Laplace).** El paper reconoce explícitamente que usar una estimación puntual de la varianza del posterior, como en la aproximación de Laplace, constituye una "debilidad significativa". El experimento de la Figura 3C lo confirma: el método **subestima la incertidumbre de los parámetros**, siendo sobreconfiado sobre qué pesos son prescindibles. Los autores sugieren que se podría mejorar con redes neuronales bayesianas (Blundell et al., 2015).
- **Crecimiento de los términos de penalización.** En la formulación original, cada tarea nueva añade un anclaje, lo que en principio hace crecer el número de términos de regularización (mitigado en parte por la observación de que las penalizaciones cuadráticas se pueden sumar en una). En secuencias largas esto motivó la posterior *online EWC*.
- **Brecha con redes separadas.** En Atari, EWC permite aprender muchos juegos en secuencia sin olvido catastrófico, pero **no alcanza el puntaje** que se obtendría entrenando diez DQNs separados. Hay un costo de compartir capacidad fija entre tareas.

## 7. Impacto

EWC es, sin discusión, **el método de regularización fundacional del aprendizaje continuo moderno**. Definió toda una familia de enfoques —los métodos basados en *penalización de pesos* o *regularización*— que estiman una medida de importancia por parámetro y penalizan su cambio. Su descendencia directa incluye **Synaptic Intelligence** (Zenke et al., 2017), que calcula la importancia de forma *online* a lo largo de la trayectoria de entrenamiento (en vez de en puntos discretos de cambio de tarea como EWC); **Memory Aware Synapses** (MAS); la *online EWC* de Schwarz et al. (2018); y un sinnúmero de variantes que refinan la aproximación de la curvatura. Junto con las otras dos grandes familias —*replay/rehearsal* (iCaRL de Rebuffi et al., 2017; GEM de Lopez-Paz y Ranzato, 2017) y *arquitecturas dinámicas/aislamiento de parámetros* (Progressive Networks de Rusu et al., 2016; PackNet; PiggyBack de Mallya et al., 2018; HAT de Serra et al., 2018)— EWC define la taxonomía estándar del campo (ver van de Ven y Tolias, 2019, sobre los tres escenarios; Masana et al., 2020, survey de class-incremental learning).

Más allá de la técnica, el paper aportó una contribución conceptual de doble sentido: no solo construyó un algoritmo a partir de observaciones neurobiológicas, sino que demostró que las teorías neurobiológicas de consolidación sináptica **escalan** a sistemas de aprendizaje a gran escala, ofreciendo evidencia *prima facie* de que estos principios podrían ser fundamentales para el aprendizaje y la memoria en el cerebro. El vínculo bidireccional IA↔neurociencia es marca registrada del DeepMind de esa época.

## 8. Conexión con la Clase 32 (Olvido Catastrófico / Aprendizaje Continuo)

La Clase 32 presenta EWC como el **primer método de la familia de regularización** del aprendizaje continuo, en el slide "Elastic Weight Consolidation (EWC)" marcado con la etiqueta *Regularización*. Es el contrapunto fundacional a las familias de *replay* y de *arquitecturas dinámicas* que la clase también cubre. Lo que el estudiante debe internalizar de este paper es la mecánica completa de la idea: (1) el olvido catastrófico ocurre porque el gradiente de la tarea nueva sobreescribe los pesos importantes de la vieja; (2) la solución no es congelar todo (eso es L2 uniforme, que mata el aprendizaje nuevo) ni añadir capacidad (eso es la familia de arquitecturas), sino **penalizar selectivamente el cambio de los pesos importantes**; (3) la importancia se mide con la diagonal de Fisher, barata de calcular y justificada como aproximación de la curvatura; (4) todo el esquema tiene una lectura bayesiana limpia —el posterior de A es el prior de B—; y (5) sus límites (Fisher diagonal, subestimación de incertidumbre, crecimiento de términos) son justamente lo que motivó la siguiente generación de métodos.

Recursos relacionados en el sitio del curso:

- Fundamento transversal: [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo) — la familia de regularización, replay y arquitecturas dinámicas; los tres escenarios de continual learning.
- Clase: [/clases/clase-32](/clases/clase-32) — Olvido Catastrófico / Aprendizaje Continuo.
- Paper hermano (regularización, importancia online): [/papers/synaptic-intelligence-zenke-2017](/papers/synaptic-intelligence-zenke-2017) — Synaptic Intelligence, la evolución directa de EWC que estima la importancia a lo largo de la trayectoria de entrenamiento en vez de en puntos de cambio de tarea.
