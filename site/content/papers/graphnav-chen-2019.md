---
title: "GraphNav: Visual Navigation with Graph Localization Networks (2019)"
weight: 307
math: true
---

{{< paper-card
    title="A Behavioral Approach to Visual Navigation with Graph Localization Networks"
    authors="Kevin Chen, Juan Pablo de Vicente, Gabriel Sepúlveda, Fei Xia, Álvaro Soto, Marynel Vázquez, Silvio Savarese"
    year="2019"
    venue="RSS 2019"
    pdf="/papers/graphnav-chen-2019.pdf"
    arxiv="1903.00445" >}}
Trabajo Stanford–PUC Chile–Yale que el ayudante **Felipe del Río** presenta en la [Clase 27](/clases/clase-27) como el ejemplo que saca a las [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos) de sus dominios canónicos (moléculas, knowledge graphs, redes sociales) hacia la **robótica y la visión por computador**. GraphNav modela el interior de un edificio como un **grafo dirigido** —nodos = lugares, aristas = comportamientos de movimiento `move-left / move-right / move-forward`— y usa una **Graph Localization Network** (GNN + CNN) para localizar al robot dentro de ese grafo a partir de su cámara. La idea clave es **separar** la localización (¿en qué arista estoy?) de la **política conductual** (¿cómo me muevo?), en vez de mezclarlas en una red end-to-end. Validado en el simulador Gibson sobre Stanford 2D-3D-S, supera a baselines de deep learning en entornos vistos y no vistos.
{{< /paper-card >}}

---

## Contexto: navegación visual en interiores

Llevar un robot de un punto a otro en un interior usando solo visión tiene tres familias clásicas de soluciones, que el paper contrasta:

**(a) Mapas métricos y SLAM clásico.** La aproximación dominante en robótica separa el mapeo y la localización (SLAM, grids de ocupación) de la planificación de trayectorias. Reconstruye una representación métrica precisa del mundo y localiza el robot con exactitud de centímetros. El costo: requiere localización precisa, es frágil ante cambios del entorno (muebles que se mueven, desorden que se acumula) y entrega **más precisión geométrica de la que la navegación realmente necesita**.

**(b) End-to-end con deep RL.** La oleada de deep learning propuso mapear directamente de los sensores a las acciones, sin mapa explícito. El trabajo pionero de Zhu et al. (2017) usa reinforcement learning para conducir el robot a una meta a partir de imágenes; aprende priors de escena (cocina vs. dormitorio) pero, por ser **reactivo**, generaliza mal a entornos nuevos y a trayectorias largas. Variantes posteriores añaden tareas auxiliares (profundidad, cierre de bucle) o emulan el ciclo bayesiano de SLAM con redes, pero siguen dependiendo de *ego-motion* ground truth o de representaciones métricas.

**(c) Representaciones topológicas.** El antecedente más cercano es la *Semi-Parametric Topological Memory* de Savinov et al. (2018), un grafo donde las aristas asocian memorias visuales de lugares vecinos y la localización se hace por **vecinos más cercanos**. GraphNav se diferencia en que la localización no es nearest-neighbor sino una **inferencia relacional aprendida con una GNN**.

La tesis del paper es **conductual** (*behavioral*) e inspirada en psicología cognitiva: los humanos navegan ambientes cambiantes sin reconstruir un mapa métrico, apoyándose en una **descripción topológica** del entorno (los "mapas cognitivos" de Tolman, 1948). GraphNav lleva esa intuición al robot: dado un **mapa topológico**, un **plan** (cómo ir de A a B en ese mapa) y la **observación visual actual**, el robot debe navegar usando solo eso. El entorno se vuelve un grafo dirigido, navegar de A a B se vuelve **recorrido de grafo** (graph traversal) y un plan es simplemente el camino más corto traducido a una secuencia de comportamientos.

## Contribución central: localización vs. política

La aportación conceptual es **descomponer** la navegación en dos subproblemas que el deep RL end-to-end mezcla en una sola red:

1. **Localización (¿dónde estoy en el grafo?)** — resuelta por la **Graph Localization Network (GLN)**. La pregunta no es "¿cuáles son mis coordenadas métricas?" sino "¿en qué **arista** del grafo estoy ahora mismo?". La localización solo necesita ser **gruesa** (coarse), relativa al grafo, no precisa en metros.

2. **Comportamiento (¿cómo me muevo?)** — resuelto por un conjunto de **behavior networks**, una red por cada comportamiento primitivo. El espacio de acciones no son velocidades de bajo nivel sino **comportamientos semánticos de alto nivel**: `{find door, corridor follow, turn left, turn right, straight into room}`. En el diagrama de la clase, estos son los `move-left / move-right / move-forward`.

Esta separación es el "enfoque conductual" del título, heredado de la arquitectura de subsunción de Brooks (1986). Su elegancia: una vez localizado el agente en una arista, **es trivial determinar qué comportamiento ejecutar** —es el primer comportamiento del plan desde esa posición— y un módulo de *behavior selection* enciende la red correspondiente. Localización y selección se repiten en **cada timestep (5 Hz)** para garantizar transiciones suaves.

Por qué una [GNN](/fundamentos/redes-neuronales-de-grafos) y no otra arquitectura: las GNN capturan **sesgos inductivos relacionales** (Battaglia et al., 2018). La conectividad del grafo es exactamente la estructura del problema, y la GNN la respeta por construcción. Localizar es **clasificar aristas** sobre un grafo, justo donde las GNN brillan. Los autores afirman ser, hasta donde saben, los **primeros en usar GNN para localización robótica** y los primeros en plantear la navegación como recorrido de grafo aprendido.

## Método: visión → localización → acción

### El grafo de comportamiento

El entorno es un **grafo dirigido**: nodos = lugares (oficina, puerta, tramo de pasillo), aristas = comportamientos que llevan del nodo fuente al destino. Para evitar ambigüedades, **nodos y aristas tienen orientación**: un pasillo se modela con **dos conjuntos de nodos/aristas, uno por sentido** (si no, "seguir el pasillo" no especifica hacia dónde). Cada sala tiene un nodo de sala y un **nodo de puerta** (orientado hacia la salida) para transiciones suaves de entrada y salida. La regla general: **poner un nodo en cualquier punto de transición**. Las anotaciones se hicieron **manualmente** con una herramienta GUI de "graph drawer" sobre los mapas métricos del dataset.

### La GNN: bloques GN y propagación

GraphNav usa el formalismo de **graph network blocks** de Battaglia et al. Un grafo es una tupla $G = (u, V, E)$: un **feature global** $u$, features de nodo $V$ y tuplas de arista $E$, todos de dimensión $D = 512$. Un GN block actualiza el grafo en tres pasos secuenciales:

1. **Aristas:** $e'_k \leftarrow \phi^e(e_k, v_{r_k}, v_{s_k}, u)$ — cada arista se actualiza según sí misma, sus dos nodos extremos y el global.
2. **Nodos:** se agregan las aristas entrantes (suma) y $v'_i \leftarrow \phi^v(\bar{e}'_i, v_i, u)$.
3. **Global:** $u' \leftarrow \phi^u(\bar{e}', \bar{v}', u)$.

Las funciones $\phi$ son **MLP**; las de agregación $\rho$ son **sumas elementwise** (simétricas, agnósticas a la permutación). La GLN apila **dos GN blocks**: así la información visual (inyectada en el global) y la estructura de conectividad se mezclan, refinando cada nodo/arista con sus vecinos, hasta emitir un **logit por arista**.

Para alimentar la GNN, el mapa se traduce a features aprendibles vía una **tabla de embeddings**: cada nodo es uno de tres tipos (`room`, `hallway`, `open space`), cada arista uno de cinco comportamientos, y el global cambia en cada timestep según la visión. Esto materializa la idea de **representación inicial de nodos** de la Clase 27: los nodos no llegan con features arbitrarios, sino con embeddings aprendidos según su tipo, que la GNN propaga y refina.

### Pipeline completo

**GLN.** El robot mantiene un *stack* de las $C = 20$ imágenes de **profundidad** más recientes (320×240, recortadas a 3,5 m; se usa depth y no RGB para generalizar a escenas de distinta apariencia). El stack pasa por una **CNN** de 7 capas que produce el vector de 512-D usado como **feature global** de la GNN. En paralelo, se hace **subgraph cropping**: como el grafo del edificio puede ser enorme y es improbable saltar de un extremo a otro, se recorta una región local centrada en la última ubicación predicha (3 aristas adelante, 2 atrás). La localización se trata como **clasificación de aristas** —se clasifica en cuál de las $m_s$ aristas del subgrafo está el robot— entrenada con softmax cross-entropy. Se elige la arista (y no el nodo) porque está mejor definida en todo instante y lleva información de nodo fuente *y* destino.

**Behavior networks.** Una vez localizado, el primer comportamiento del plan determina qué red usar. Cada red toma la profundidad y emite **velocidades de control** $[v_p, v_\theta]$. Hay dos arquitecturas: una **CNN reactiva** para `corridor follow` y `find door`, y una **CNN-LSTM** para `turn left/right` y `straight`, donde la memoria recurrente ayuda. Se entrenan por **behavioral cloning** (pérdida MSE) y cada una por separado, solo con los frames anotados para ese comportamiento.

**Particle filter (GraphNavPF).** Una variante combina la GLN con un **filtro de partículas** que suaviza las predicciones en el tiempo (modelo de movimiento simple: probabilidad 0,8 de quedarse, resto repartido entre vecinos; modelo de medición proporcional a la salida de la GLN). Es la versión que mejores resultados obtiene.

## Experimentos

**Entorno.** Todo se entrena y testea en el simulador **Gibson** (física PyBullet) sobre el dataset **Stanford 2D-3D-S**: mallas de edificios universitarios reales, con plantas complejas y mucho desorden. El agente es un **Turtlebot** controlado vía ROS, con cámara de profundidad de 150° de FOV, comandos a 5 Hz y velocidad tope 0,5 m/s. Detalle de rigor: **las colisiones son fatales** y cuentan como fallo, a diferencia de varios trabajos previos que las ignoraban. No se entrega *ego-motion* ground truth. Implementación en **PyTorch** (Adam, lr 1e-4, batch 32).

**Dataset.** 2.371 trayectorias (promedio 423,56 frames) recolectadas corriendo el ROS Navigation Stack con localización ground truth, grabando RGB, profundidad, semántica y odometría. Se **inyectó ruido** en las velocidades para enseñar al agente a recuperarse de mal posicionamiento. Se usan cinco áreas: **1, 5, 6 para entrenamiento**, **3 para validación**, **4 para test**. Solo hay tres edificios distintos (áreas 1, 3, 6 son partes del mismo) — diversidad limitada, clave para las limitaciones. Las tareas se dividen por número de nodos del camino: I (1–10), II (11–20), III (>20).

**Métricas.** **Success rate (SR):** éxito si el robot sigue el plan hasta el destino sin desviarse ni chocar. **Plan completion (PC):** fracción de nodos del plan alcanzados. Además, métricas por comportamiento y por dificultad.

**Baselines.** *PhaseNet* (LSTM que decide cuándo cambiar de comportamiento), *BehavRNN* (seq2seq que clasifica el comportamiento en cada timestep) y *GTL (Ground Truth Location)*, que localiza con anotación perfecta en tiempo real y funciona como **techo** del rendimiento de las behavior networks.

**Resultados.** GraphNav supera a PhaseNet y BehavRNN en **todas** las áreas. PhaseNet sigue pasillos a ciegas pero pierde la mayoría de los giros (success rates de giro típicamente <50%). Ejemplos concretos: en el área 1 (seen) el giro a la izquierda sube de **42,9% (PhaseNet) a 79,0% (GraphNav)**; en plan completion del área 3 (unseen), PhaseNet logra 55,4% mientras **GraphNavPF llega a 77,7%**; el SR total del área 5 (seen) pasa de **30,3% (GraphNav) a 60,6% (GraphNavPF)**. El **GTL** confirma que las behavior networks son robustas (success rates por comportamiento generalmente sobre 80–90%). La precisión de localización de la GLN es alta en train (**89,8%**) pero cae fuerte en validación (52,3%) y test (**31,1%**) — la firma del overfitting por falta de diversidad de entornos. Un hallazgo cualitativo elegante: al acercarse a una puerta, la GLN **reparte la probabilidad entre el giro izquierdo y el derecho**, lo que equivale a predecir que el agente está en el nodo de puerta y dispara la transición de comportamiento.

## Limitaciones

El propio paper las enumera con honestidad:

- **Mapas y comportamientos manuales:** las anotaciones topológicas se hicieron a mano y los cinco comportamientos están predefinidos; automatizar la creación de mapas y descubrir comportamientos data-driven queda como trabajo futuro.
- **Margen en success rate:** los números absolutos en entornos no vistos son modestos (SR total de GraphNav en área 4 unseen: 16,0%). El reto clave es el **timing de las transiciones** entre comportamientos.
- **Poca diversidad de entrenamiento:** solo tres edificios; el gap train/test en localización (89,8% → 31,1%) lo evidencia.
- **Espacios abiertos grandes:** incluso el GTL falla ahí; el robot se desorienta, agravado por el límite de 3,5 m de la profundidad.
- **Solo simulación:** no hay experimentos en robots reales; el *sim-to-real transfer* queda pendiente.

## Impacto: GNN en robótica

GraphNav es uno de los primeros trabajos en aplicar [GNN](/fundamentos/redes-neuronales-de-grafos) a **localización robótica** y en plantear la navegación como **recorrido de grafo aprendido**, sacando a las GNN de sus dominios canónicos hacia la robótica encarnada. Su valor de archivo es doble: **técnico** —demuestra que combinar arquitecturas robóticas clásicas (separar mapeo/localización de planificación/ejecución) con deep learning produce sistemas más robustos y generalizables que el end-to-end puro— y de **infraestructura** —entrega a la comunidad un dataset anotado, una especificación de mapas topológicos y un testbed de benchmarking sobre Gibson controlable por ROS. Se inscribe en la corriente que ve los **sesgos inductivos relacionales** como puente entre el aprendizaje profundo y el razonamiento estructurado, y anticipa el interés posterior en navegación topológica y memorias de grafos. Para el lector chileno es, además, un hito de la colaboración Stanford–PUC en deep learning aplicado a robótica.

## Por qué importa para la Clase 27

La [Clase 27](/clases/clase-27) ("Redes Neuronales de Grafos") usa GraphNav como **aplicación** de GNN fuera de sus dominios típicos:

- **El grafo como estructura del problema.** La conectividad del entorno (qué lugar lleva a qué otro y con qué movimiento) es literalmente el grafo, y la GNN la respeta por construcción. El slide con `kitchen / hall / office` unidos por `move-left / move-right / move-forward` es la abstracción central del paper hecha diagrama: nodos = lugares, aristas = acciones.
- **Representación inicial de nodos.** Cada nodo arranca con un embedding aprendido según su tipo (`room` / `hallway` / `open space`), cada arista según su comportamiento, y un feature global que inyecta la visión. Los dos GN blocks propagan y actualizan esas representaciones — un ejemplo concreto de cómo la teoría de la "representación inicial de nodos" se instancia en un sistema real.
- **Localización vs. política.** Más allá de las GNN, ilustra la idea de descomponer un problema en una parte relacional (que resuelve el grafo) y otra de percepción-acción, en vez de una sola caja negra end-to-end.

Relacionado: la [GCN de Kipf & Welling (2017)](/papers/gcn-kipf-2017) aporta el formalismo base de convolución en grafos; en dominios, este trabajo cruza lo [estructurado](/dominios/estructurados) (datos en forma de grafo) con la [robótica](/dominios/robotica) (percepción-acción encarnada).

## Notas y enlaces

- Sitio del proyecto: [graphnav.stanford.edu](https://graphnav.stanford.edu)
- arXiv:1903.00445v1 (1 mar 2019), cs.CV.
- Financiamiento: Toyota Research Institute; Fondecyt grant 1181739, Conicyt, Chile.
- Presentado por el ayudante **Felipe del Río** en la Clase 27.
