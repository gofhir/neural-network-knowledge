---
title: "Three Scenarios for Continual Learning (2019)"
weight: 361
math: true
---

{{< paper-card
    title="Three scenarios for continual learning"
    authors="Gido M. van de Ven, Andreas S. Tolias"
    year="2019"
    venue="NeurIPS Workshop / arXiv"
    pdf="/papers/three-scenarios-van-de-ven-2019.pdf"
    arxiv="1904.07734" >}}
Reporte técnico que pone orden en el campo del **aprendizaje continuo** definiendo una taxonomía de **tres escenarios** de dificultad creciente, distinguidos por una sola pregunta: ¿qué se sabe sobre la identidad de la tarea en el momento de test? De ahí salen **Task-IL** (se conoce el task-ID), **Domain-IL** (misma tarea, distinta distribución, sin ID) y **Class-IL** (hay que distinguir todas las clases e inferir la tarea, el más difícil). Su hallazgo empírico más citado: los métodos de regularización (EWC, SI) **colapsan al nivel del azar en Class-IL**, mientras solo el *replay* logra desempeño aceptable. No propone algoritmo nuevo; su aporte es **metodológico y taxonómico**, y se volvió el estándar de facto del área. No está citado en las diapositivas de la [Clase 32](/clases/clase-32), pero **es la fuente canónica de su sección de escenarios**.
{{< /paper-card >}}

---

## Contexto: un campo sin taxonomía común

El **olvido catastrófico** —la tendencia de una red neuronal a perder lo aprendido sobre tareas previas al entrenarse en una nueva— es el obstáculo central del [aprendizaje continuo](/fundamentos/aprendizaje-continuo). Entre 2016 y 2019 proliferaron métodos para mitigarlo: [EWC](/papers/ewc-kirkpatrick-2017) (Kirkpatrick et al., 2017), Synaptic Intelligence (Zenke et al., 2017), Learning without Forgetting (Li & Hoiem, 2017), [iCaRL](/papers/icarl-rebuffi-2017) (Rebuffi et al., 2017), Deep Generative Replay (Shin et al., 2017), entre muchos otros.

El problema que el paper diagnostica con precisión es que **muchos de estos métodos reclamaban "estado del arte" simultáneamente** porque cada uno se evaluaba bajo un protocolo distinto. Peor aún: un método que brillaba en un setup fallaba dramáticamente en otro. El caso emblemático es EWC, que rinde muy bien en Kirkpatrick et al. (2017) pero "fracasa dramáticamente" en Kemker et al. (2018). La causa no era el método en sí, sino que **se estaba evaluando en escenarios de dificultad radicalmente distinta sin que nadie lo nombrara**.

Existía un intento previo de estructurar el campo: la distinción **single-headed vs multi-headed**, referida a la arquitectura de la capa de salida —una cabeza por tarea (multi-headed) versus una capa compartida (single-headed)—. Van de Ven & Tolias la consideran insuficiente y la extienden con su taxonomía de tres escenarios, introducida primero en un paper anterior de los mismos autores (2018) y profundizada aquí.

## Contribución central: los tres escenarios

La tesis es simple y poderosa: **el factor que más determina la dificultad de un problema de aprendizaje continuo no es el dataset, sino qué información sobre la identidad de la tarea está disponible en test**. A partir de esa única pregunta —¿se entrega el task-ID?, y si no, ¿hay que inferirlo?— se derivan tres escenarios.

| Escenario | Qué se requiere en test |
|---|---|
| **Task-IL** (Task-Incremental Learning) | Resolver las tareas vistas, **con el task-ID provisto** |
| **Domain-IL** (Domain-Incremental Learning) | Resolver la tarea presente, **sin task-ID** (pero no hay que inferirlo) |
| **Class-IL** (Class-Incremental Learning) | Resolver las tareas vistas **e inferir el task-ID** |

**Task-IL.** El modelo **siempre sabe qué tarea debe resolver** en test. Es el escenario más fácil: como la identidad de la tarea está disponible, se pueden usar **componentes específicos por tarea**. La arquitectura típica es una capa de salida **multi-cabeza** (cada tarea con sus propias unidades de salida, el resto de la red compartido). En split MNIST: "dado que esta es la tarea X, ¿es la primera o la segunda clase?" (p. ej. ¿0 o 1?).

**Domain-IL.** El task-ID **no está disponible**, pero el modelo **solo necesita resolver la tarea actual; no se le exige identificar cuál es**. La estructura de las tareas es siempre la misma, pero la distribución del input cambia. Ejemplo real de los autores: un agente que debe sobrevivir en distintos entornos sin identificar explícitamente en cuál está. En split MNIST: "tarea desconocida, ¿es una primera o una segunda clase?" (¿está en {0,2,4,6,8} o en {1,3,5,7,9}?).

**Class-IL.** El escenario **más difícil**. El modelo debe **resolver cada tarea vista Y además inferir con qué tarea está siendo confrontado**. Es el problema clásico y de máxima relevancia práctica de aprender incrementalmente nuevas clases de objetos. En split MNIST: "tarea desconocida, ¿qué dígito es?" —elección entre las 10 clases 0-9, habiéndolas aprendido de a dos—. La arquitectura usa una unidad de salida por clase, con todas las clases vistas activas simultáneamente.

**Frontera con single-/multi-headed.** Los escenarios se diferencian del esquema clásico en dos puntos: (1) el esquema single-/multi-headed está atado a la **arquitectura de la capa de salida**, mientras los escenarios reflejan más generalmente **las condiciones bajo las que se evalúa el modelo** (el task-ID puede usarse también en capas ocultas, como en Context-dependent Gating); (2) los escenarios reconocen que, cuando el task-ID no se provee, hay una distinción adicional según si la red debe inferirlo o no —y esa distinción (Domain-IL vs Class-IL) genera una diferencia sustancial de dificultad—. Una condición fronteriza: los tres escenarios **asumen límites de tarea claros**; si las transiciones son graduales, la taxonomía deja de aplicar.

## Protocolo experimental

Para demostrar que cualquier protocolo puede plantearse según los tres escenarios, los autores usan dos clásicos sobre MNIST:

- **Split MNIST:** el dataset se parte en **cinco tareas** de clasificación binaria (0/1, 2/3, 4/5, 6/7, 8/9).
- **Permuted MNIST:** **diez tareas**, cada una clasifica los diez dígitos pero con una **permutación aleatoria distinta** de los píxeles. Es naturalmente Domain-IL pero puede plantearse en los tres escenarios.

Los métodos se organizan en cuatro familias de estrategias contra el olvido:

- **Componentes específicos por tarea:** definir una sub-red por tarea. Ejemplo: **XdG** (Context-dependent Gating), que apaga unidades por tarea. Solo sirve en Task-IL, porque requiere el task-ID.
- **Optimización regularizada:** usar toda la red pero penalizar cambios a parámetros importantes para tareas previas. **EWC** estima la importancia vía la diagonal de la matriz de Información de Fisher; **Online EWC** la reduce a un único término acumulado; **SI** (Synaptic Intelligence) la estima online integrando la contribución de cada parámetro a la pérdida.
- **Replay:** complementar los datos de cada tarea nueva con "pseudo-datos" de tareas previas. **LwF** reetiqueta los inputs actuales con *soft targets* del modelo previo (destilación). **DGR** (Deep Generative Replay) entrena un VAE que genera muestras de tareas previas. **DGR+distill** combina ambos.
- **Exemplars:** almacenar datos reales previos. **iCaRL** usa un extractor de features con regla *nearest-class-mean*; por su diseño con destilación binaria, solo se aplica en Class-IL.

Baselines: **None** (fine-tuning secuencial, cota inferior) y **Offline** (joint training, cota superior). Arquitectura MLP común, optimizador Adam, cada experimento repetido 20 veces.

## Resultados: el colapso de la regularización en Class-IL

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
- **En Class-IL la regularización colapsa al nivel del azar** (~20% = 1/5, lo que se obtiene adivinando entre las dos clases de la última tarea sobre 10). EWC, Online EWC y SI rinden esencialmente igual que la cota inferior None. Este es el resultado más citado del paper.
- **Solo el replay rescata el Class-IL:** DGR, DGR+distill e iCaRL superan el 90%. El replay de experiencias previas parece **necesario** para resolver este escenario.
- Incluso LwF —que solo "repite" inputs de la tarea actual reetiquetados— previene el olvido mejor que cualquier método de regularización en Domain-IL: el replay imperfecto resulta más efectivo que penalizar parámetros.

**Permuted MNIST** muestra el mismo patrón: en Class-IL la regularización vuelve a fallar (EWC 25%, SI 29%, Online EWC 34%) y solo el replay alcanza buen desempeño (DGR 92%, DGR+distill 96%, iCaRL 95%). LwF falla aquí porque las permutaciones dejan los inputs de tareas distintas no correlacionados.

## Impacto y conexión con la Clase 32

**Conclusión central:** para el escenario class-incremental —cuando hay que inferir la identidad de la tarea— **solo los métodos basados en replay producen resultados aceptables**, y la regularización fracasa por completo incluso en MNIST.

La taxonomía Task-IL / Domain-IL / Class-IL se convirtió en el **estándar de facto del campo**: cualquier paper moderno de continual learning especifica bajo qué escenario evalúa. El colapso de la regularización en Class-IL reorientó la investigación hacia el replay y motivó trabajos posteriores de los propios autores (*Brain-inspired replay*, Nature Communications 2020) y de toda la comunidad. El código abierto y reproducible reforzó la adopción.

La **[Clase 32](/clases/clase-32)** dedica su sección de escenarios a esta taxonomía, aunque no la cite explícitamente: **este reporte es su fuente directa**. Cuando la clase pregunta "¿se conoce la tarea en test?" para clasificar un problema de aprendizaje continuo, aplica la pregunta operativa que van de Ven & Tolias formalizaron. Y el "porqué" del replay en la clase viene de aquí: el respaldo de que la regularización no basta y el replay es necesario para el caso más realista es la tabla del colapso al ~20% frente al >90% del replay.

En un sistema real —por ejemplo un clasificador médico que incorpora nuevas patologías a lo largo del tiempo— el escenario relevante suele ser **Class-IL**: en inferencia no se sabe a qué "tarea" pertenece el caso y hay que distinguir entre todas las clases acumuladas. Este paper dice, sin ambigüedad, que en ese régimen la regularización no alcanza y se requiere repetir o almacenar ejemplos previos; una guía de diseño directa.

## Notas y enlaces

- Preprint: [arxiv.org/abs/1904.07734](https://arxiv.org/abs/1904.07734) (15 abr 2019). Versión extendida de un trabajo del NeurIPS Continual Learning workshop (2018).
- Código (PyTorch, todos los métodos comparados): [github.com/GMvandeVen/continual-learning](https://github.com/GMvandeVen/continual-learning).
- Afiliaciones: Baylor College of Medicine (Houston), University of Cambridge, Rice University.
- Fundamento transversal: [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo).
