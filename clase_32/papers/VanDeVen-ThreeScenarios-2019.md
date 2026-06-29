# Three scenarios for continual learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Three scenarios for continual learning*.
- **Autores:** Gido M. van de Ven (Center for Neuroscience and Artificial Intelligence, Baylor College of Medicine, Houston; Computational and Biological Learning Lab, University of Cambridge) y Andreas S. Tolias (Baylor College of Medicine; Department of Electrical and Computer Engineering, Rice University, Houston).
- **Venue:** Reporte técnico / preprint. Versión extendida de un trabajo presentado en el *NeurIPS Continual Learning workshop* (2018).
- **Año:** 2019. **Preprint:** arXiv:1904.07734v1 (15 abr 2019), [arxiv.org/abs/1904.07734](https://arxiv.org/abs/1904.07734).
- **Código:** [github.com/GMvandeVen/continual-learning](https://github.com/GMvandeVen/continual-learning) — implementaciones PyTorch de todos los métodos comparados, documentadas y fáciles de adaptar.
- **Financiamiento:** IBRO-ISN Research Fellowship, programa *Lifelong Learning Machines* (L2M) de DARPA, e IARPA vía DoI/IBC.

> **NOTA sobre su rol en la Clase 32.** Este paper **no aparece citado explícitamente en las diapositivas** de la clase. Sin embargo, **es la fuente canónica de la taxonomía** que estructura toda la sección de escenarios de la clase: la distinción **Task-Incremental / Domain-Incremental / Class-Incremental** que organiza la sección 3 de la Clase 32 proviene directamente de este reporte (y de su antecesor van de Ven & Tolias 2018). Cuando la clase habla de "los tres escenarios del aprendizaje continuo", está reproduciendo el esquema definido aquí. Por eso lo incluimos como lectura interna pese a no estar en la bibliografía explícita: es el armazón conceptual del bloque.

El paper no propone un algoritmo nuevo. Su aporte es **metodológico y taxonómico**: poner orden en un campo —el aprendizaje continuo (*continual learning*) o aprendizaje a lo largo de la vida (*lifelong learning*)— que hacia 2019 estaba lleno de métodos que reclamaban ser "estado del arte" pero se evaluaban bajo protocolos experimentales incomparables entre sí. La tesis central es simple y poderosa: **el factor que más determina la dificultad de un problema de aprendizaje continuo no es el dataset, sino qué información sobre la identidad de la tarea está disponible en el momento de test**. A partir de esa única pregunta —¿se entrega el task-ID en test?, y si no, ¿hay que inferirlo?— los autores derivan tres escenarios de dificultad creciente, demuestran que cualquier secuencia de tareas bien definidas puede plantearse según cualquiera de los tres, y comparan sistemáticamente los métodos existentes en cada uno.

El hallazgo empírico que da fuerza al paper: los métodos de **regularización** (EWC, Online EWC, SI), celebrados en la literatura, **colapsan por completo en el escenario más difícil (Class-Incremental)**, cayendo al nivel del azar incluso en tareas tan simples como clasificar dígitos MNIST. Solo los métodos basados en **replay** (repetición de experiencias previas) logran desempeño aceptable cuando hay que inferir la tarea. Este resultado reorientó buena parte de la investigación posterior hacia el replay.

## 2. Contexto histórico: un campo sin taxonomía común

El olvido catastrófico —la tendencia de las redes neuronales estándar a perder lo aprendido sobre tareas previas al entrenarse en una nueva— es el obstáculo central del aprendizaje continuo. Entre 2016 y 2019 proliferaron métodos para mitigarlo: Elastic Weight Consolidation (Kirkpatrick et al., 2017), Synaptic Intelligence (Zenke et al., 2017), Learning without Forgetting (Li & Hoiem, 2017), iCaRL (Rebuffi et al., 2017), Deep Generative Replay (Shin et al., 2017), entre muchos otros.

El problema que el paper diagnostica con precisión es que **muchos de estos métodos reclamaban "estado del arte" simultáneamente** porque cada uno se evaluaba bajo un protocolo distinto. Peor aún: métodos que brillaban en un setup fallaban dramáticamente en otro. El ejemplo que los autores destacan es EWC, que rinde muy bien en Kirkpatrick et al. (2017) y Zenke et al. (2017) pero "fracasa dramáticamente" en Kemker et al. (2018) y Kamra et al. (2017). La causa de esta aparente contradicción no era el método en sí, sino que **se estaba evaluando en escenarios de dificultad radicalmente distinta sin que nadie lo nombrara explícitamente**.

Existía un intento previo de estructurar el campo: la distinción **"single-headed" vs "multi-headed"** (Farquhar & Gal, 2018; Chaudhry et al., 2018), que se refiere a la arquitectura de la capa de salida —una cabeza de salida separada por tarea (multi-headed) versus una capa de salida compartida (single-headed)—. El paper de van de Ven & Tolias se relaciona con esa distinción pero la considera insuficiente por dos razones que detallaremos en la sección 4. La taxonomía de los tres escenarios fue introducida primero por los mismos autores en un paper anterior (van de Ven & Tolias, 2018, "Generative replay with feedback connections") y ya había sido adoptada por varios estudios (Hsu et al., 2018; Zeno et al., 2019; Lee et al., 2019); este reporte la trata en profundidad.

## 3. Contribución central: los tres escenarios

La contribución principal es una **taxonomía de tres escenarios de aprendizaje continuo de dificultad creciente**, distinguidos por una sola pregunta operativa sobre las condiciones de evaluación.

| Escenario | Qué se requiere en test |
|---|---|
| **Task-IL** (Task-Incremental Learning) | Resolver las tareas vistas, **con el task-ID provisto** |
| **Domain-IL** (Domain-Incremental Learning) | Resolver la tarea presente, **sin task-ID** (pero no hay que inferirlo) |
| **Class-IL** (Class-Incremental Learning) | Resolver las tareas vistas **e inferir el task-ID** |

### 3.1. Task-Incremental Learning (Task-IL)

El modelo **siempre sabe qué tarea debe resolver** en el momento de test. Es el escenario más fácil. Como la identidad de la tarea está siempre disponible, es posible entrenar modelos con **componentes específicos por tarea**. La arquitectura típica usa una **capa de salida "multi-cabeza"** (*multi-headed*): cada tarea tiene sus propias unidades de salida, mientras el resto de la red puede compartirse. Ejemplo en split MNIST: "dado que esta es la tarea X, ¿es la primera o la segunda clase?" (p.ej. ¿0 o 1?).

### 3.2. Domain-Incremental Learning (Domain-IL)

El task-ID **no está disponible** en test, pero el modelo **solo necesita resolver la tarea actual; no se le exige identificar cuál es**. La estructura de las tareas es siempre la misma, pero la distribución del input cambia. Ejemplo del mundo real que dan los autores: un agente que debe sobrevivir en distintos entornos sin necesidad de identificar explícitamente en cuál se encuentra. En split MNIST: "tarea desconocida, ¿es una primera o una segunda clase?" (¿está en {0,2,4,6,8} o en {1,3,5,7,9}?). En permuted MNIST este es el escenario más natural: "¿qué dígito es?" sin saber qué permutación se aplicó.

### 3.3. Class-Incremental Learning (Class-IL)

El escenario **más difícil**. El modelo debe **resolver cada tarea vista hasta ahora Y además inferir con qué tarea está siendo confrontado**. Es el problema clásico y de máxima relevancia práctica de aprender incrementalmente nuevas clases de objetos. En split MNIST: "tarea desconocida, ¿qué dígito es?" — elección entre las 10 clases 0-9, habiéndolas aprendido de a dos. En permuted MNIST: "¿qué dígito y qué permutación?". La arquitectura usa una unidad de salida por clase, con todas las unidades de las clases vistas hasta el momento activas simultáneamente.

### 3.4. Frontera con el esquema single-/multi-headed

La taxonomía de los tres escenarios se diferencia del esquema single-/multi-headed en dos puntos clave:

1. El esquema single-/multi-headed está atado a la **arquitectura de la capa de salida**, mientras que los escenarios reflejan más generalmente **las condiciones bajo las cuales se evalúa el modelo**. Multi-headed es la forma más común de usar el task-ID, pero no la única: el task-ID puede usarse también en las capas ocultas (como en XdG/Context-dependent Gating).
2. Los escenarios **extienden** el esquema al reconocer que, cuando el task-ID no se provee, hay una distinción adicional según si la red está obligada o no a inferirlo — y esa distinción (Domain-IL vs Class-IL) genera una diferencia sustancial de dificultad, como demuestran los experimentos.

Una condición fronteriza importante: los tres escenarios **asumen límites de tarea claros y bien definidos** durante el entrenamiento. Si las transiciones entre tareas son graduales o continuas (sin fronteras), la taxonomía deja de aplicar y el problema se vuelve mucho más difícil y menos estructurado.

## 4. Método: protocolo experimental y familias de estrategias

### 4.1. Dos protocolos de tareas, tres escenarios cada uno

Para demostrar que cualquier protocolo puede plantearse según los tres escenarios, los autores usan dos protocolos clásicos sobre MNIST:

- **Split MNIST:** el dataset se parte en **cinco tareas** de clasificación binaria (0/1, 2/3, 4/5, 6/7, 8/9). Imágenes 28×28 en escala de grises sin preprocesar.
- **Permuted MNIST:** secuencia de **diez tareas**, cada una clasifica los diez dígitos pero con una **permutación aleatoria distinta** aplicada a los píxeles (imágenes zero-padded a 32×32 = 1024 píxeles). Es naturalmente Domain-IL pero puede plantearse en los tres escenarios.

### 4.2. Las cuatro familias de estrategias contra el olvido

El paper organiza los métodos en una taxonomía de estrategias, paralela a la de escenarios:

- **Componentes específicos por tarea** (sección 3.1 del paper): definir una sub-red distinta por tarea. Ejemplos: **XdG** (Context-dependent Gating; Masse et al., 2018), que apaga aleatoriamente un porcentaje de unidades por tarea. Por diseño, **solo sirve en Task-IL** porque requiere el task-ID para seleccionar los componentes.
- **Optimización regularizada** (sección 3.2): usar toda la red en ejecución pero penalizar cambios a parámetros importantes para tareas previas. **EWC** estima la importancia vía la diagonal de la matriz de Información de Fisher; **Online EWC** (Schwarz et al., 2018) reduce EWC a un único término cuadrático con suma acumulada de Fisher (su costo no crece con el número de tareas); **SI** (Synaptic Intelligence) estima la importancia online integrando la contribución de cada parámetro al cambio de la pérdida.
- **Modificación de los datos de entrenamiento — replay** (sección 3.3): complementar los datos de cada tarea nueva con "pseudo-datos" representativos de tareas previas. **LwF** (Learning without Forgetting) reetiqueta los inputs de la tarea actual con *soft targets* del modelo previo (una forma de destilación). **DGR** (Deep Generative Replay) entrena un modelo generativo separado (un VAE) que genera muestras de tareas previas, etiquetadas con *hard targets*. **DGR+distill** combina ambos: muestras generadas con *soft targets*.
- **Uso de exemplars** (sección 3.4): almacenar datos reales de tareas previas. **iCaRL** usa una red extractora de features y clasifica con una regla de *nearest-class-mean* sobre los exemplars almacenados, protegiendo el extractor vía replay y destilación. Por su diseño con destilación binaria, **solo se aplica en Class-IL**.

Como baselines: **None** (fine-tuning secuencial, cota inferior) y **Offline** (joint training con todos los datos, cota superior). Todos los métodos usan la misma arquitectura MLP (2 capas ocultas de 400 nodos para split, 1000 para permuted), optimizador Adam, y cada experimento se repite 20 veces.

## 5. Resultados: el colapso de la regularización en Class-IL

Los resultados (Tablas 4 y 5 del paper) muestran una **diferencia nítida de dificultad entre los tres escenarios**, y son el corazón empírico del trabajo.

**Split MNIST (precisión media sobre todas las tareas):**

| Método | Task-IL | Domain-IL | Class-IL |
|---|---|---|---|
| None (cota inferior) | 87.2% | 59.2% | 19.9% |
| Offline (cota superior) | 99.7% | 98.4% | 97.9% |
| EWC | 98.6% | 64.0% | **20.0%** |
| Online EWC | 99.1% | 64.3% | **20.0%** |
| SI | 99.1% | 65.4% | **20.0%** |
| LwF | 99.6% | 71.5% | 23.9% |
| DGR | 99.5% | 95.7% | **90.8%** |
| DGR+distill | 99.6% | 96.8% | **91.8%** |
| iCaRL (budget 2000) | — | — | **94.6%** |

Lecturas clave:

- **En Task-IL todos los métodos funcionan bien** (>98%). El escenario fácil no discrimina entre métodos.
- **En Class-IL los métodos de regularización colapsan al nivel del azar** (~20% = 1/5, que es lo que se obtiene adivinando entre las dos clases de la última tarea sobre 10 clases). EWC, Online EWC y SI rinden esencialmente igual que la cota inferior None. Este es el resultado más citado del paper.
- **Solo el replay rescata el Class-IL:** DGR, DGR+distill e iCaRL superan el 90%. El replay de experiencias previas parece **necesario** para resolver este escenario.
- **Hallazgo llamativo:** incluso LwF —que solo "repite" inputs de la tarea actual reetiquetados— previene el olvido mejor que cualquier método de regularización en Domain-IL. Sugiere que el replay, aun imperfecto, es más efectivo que penalizar parámetros.

**Permuted MNIST:** patrón análogo. Todos los métodos salvo LwF rinden bien en Task-IL y Domain-IL (la diferencia entre ambos es pequeña aquí, porque el task-ID solo se usó en la capa de salida; al combinarlos con XdG —usando task-ID en capas ocultas— Task-IL mejora claramente, confirmando que es más fácil que Domain-IL). Pero en **Class-IL la regularización vuelve a fallar** (EWC 25%, SI 29%, Online EWC 34%) y solo el replay alcanza buen desempeño (DGR 92%, DGR+distill 96%, iCaRL 95%). LwF falla en permuted MNIST porque las permutaciones aleatorias dejan los inputs de tareas distintas no correlacionados.

Un matiz metodológico honesto: los autores obtuvieron EWC competitivo en Task-IL (contra reportes previos que lo daban por fallido) porque exploraron un rango de hiperparámetros mucho más amplio —valores varios órdenes de magnitud mayores que los típicos—, debido a que en split MNIST las tareas individuales son tan fáciles que la Información de Fisher resultante es muy pequeña.

## 6. Discusión, limitaciones e impacto

**Conclusión central del paper:** para el escenario class-incremental —cuando hay que inferir la identidad de la tarea— **solo los métodos basados en replay producen resultados aceptables hoy**, y los métodos de regularización (EWC, SI) fracasan por completo incluso en MNIST. Para los escenarios más desafiantes y "etológicamente relevantes" donde el task-ID no se provee, **el replay podría ser una herramienta inevitable**.

**Limitaciones reconocidas por los autores:**

- Las imágenes de MNIST son relativamente fáciles de generar; queda como pregunta abierta si el generative replay seguirá siendo tan exitoso con distribuciones de input más complejas. Como mitigantes señalan que (a) la calidad de los modelos generativos mejora rápidamente, y (b) el buen desempeño de LwF sugiere que aun replays imperfectos ayudan.
- Discuten críticamente la fijación de hiperparámetros en aprendizaje continuo (Apéndice D): el método estándar de grid search sobre datos de validación de todas las tareas **viola el principio de visitar cada tarea solo una vez**, y advierten contra hiperparámetros influyentes que dan ventajas injustas.

**Impacto.** La taxonomía Task-IL / Domain-IL / Class-IL se convirtió en el **estándar de facto del campo** para describir y comparar configuraciones de aprendizaje continuo. Cualquier paper moderno de continual learning especifica bajo qué escenario evalúa. El resultado del colapso de la regularización en Class-IL reorientó la investigación hacia el replay y motivó trabajos posteriores de los propios autores (replay generativo basado en cerebro, *Brain-inspired replay*, Nature Communications 2020) y de toda la comunidad. El código abierto y reproducible reforzó esa adopción.

## 7. Conexión con la Clase 32 (Olvido Catastrófico)

La **sección 3 de la Clase 32 dedica su exposición a los escenarios Task / Domain / Class Incremental** — y **esta es la fuente directa de esa taxonomía**. Mapeo concreto:

- **El armazón conceptual de la clase es este paper.** Cuando la clase pregunta "¿se conoce la tarea en test?" para clasificar un problema de aprendizaje continuo, está aplicando la pregunta operativa que van de Ven & Tolias formalizaron. Las tres categorías —cabezas separadas con task-ID conocido (Task-IL), misma estructura con distinta distribución sin task-ID (Domain-IL), distinguir todas las clases e inferir la tarea (Class-IL)— son textualmente las del paper.
- **El "porqué" del replay en la clase viene de aquí.** Si la clase argumenta que la regularización (EWC/SI) no basta para escenarios incrementales de clase y que el replay es necesario, el respaldo empírico es la Tabla 4 de este paper: el colapso al ~20% de los métodos de regularización en Class-IL frente al >90% del replay. Es la evidencia que justifica por qué el campo —y la clase— prioriza el replay para el caso más realista.
- **Vincular con un caso clínico/de producción.** En un sistema real (p.ej. un clasificador médico que va incorporando nuevas patologías a lo largo del tiempo), el escenario relevante suele ser **Class-IL**: en inferencia no se sabe a qué "tarea" pertenece el caso y hay que distinguir entre todas las clases acumuladas. Este paper dice, sin ambigüedad, que en ese régimen la regularización no alcanza y se requiere repetir/almacenar ejemplos previos — una guía de diseño directa.

**Enlaces internos del site:**

- Fundamento transversal: [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo)
- Clase: [/clases/clase-32](/clases/clase-32)
