---
title: "MPNN: Neural Message Passing for Quantum Chemistry (2017)"
weight: 303
math: true
---

{{< paper-card
    title="Neural Message Passing for Quantum Chemistry"
    authors="Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl"
    year="2017"
    venue="ICML 2017"
    pdf="/papers/mpnn-gilmer-2017.pdf"
    arxiv="1704.01212" >}}
El paper de Google Brain/DeepMind que le dio nombre al **message passing** y que vertebra la [Clase 27](/clases/clase-27). Hace dos cosas a la vez: es un **paper de síntesis** que demuestra que al menos ocho modelos de [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos) publicados entre 2013 y 2017 son casos particulares de un mismo marco abstracto —las *Message Passing Neural Networks* (MPNN)—; y es un **paper empírico** que, una vez establecido ese marco, explora variantes nuevas y alcanza estado del arte en QM9, logrando precisión química en 11 de 13 propiedades moleculares. El framework de tres funciones (mensaje, actualización, readout) se convirtió en el lenguaje común de todo el subcampo.
{{< /paper-card >}}

---

## Contexto: DFT es exacto pero lentísimo

Predecir las propiedades de una molécula es, en principio, un problema resuelto por la física: la mecánica cuántica las determina. El problema es que las ecuaciones de Schrödinger "son demasiado difíciles de resolver exactamente". Por eso el campo desarrolló una jerarquía de aproximaciones, siendo la **Teoría del Funcional de la Densidad (DFT)** la herramienta de trabajo dominante.

DFT tiene dos defectos que el paper cuantifica con cifras memorables. Es **lentísima** —escala como O(N_e³) con el número de electrones— y arrastra errores sistemáticos respecto a la solución exacta. El número clave: correr DFT sobre una sola molécula de 9 átomos pesados de QM9 toma cerca de **una hora** en un núcleo de un Xeon E5-2660; para 17 átomos pesados, hasta **8 horas**. En contraste, la inferencia de las redes neuronales es **300.000 veces más rápida** (la Figura 1 lo ilustra como DFT ≈ 10³ s frente a MPNN ≈ 10⁻² s). Ese es el premio: aprender una función rápida que imite el cálculo costoso de DFT.

El paper enmarca el momento como análogo a la visión por computador *antes* de las CNN. Ya había redes neuronales aplicadas a química (Duvenaud et al. 2015, entre otros), pero el ML del área giraba en torno a la **ingeniería manual de features** (Coulomb Matrix, Bag of Bonds). La apuesta: como AlexNet desplazó a los SVM sobre features hechos a mano, llegó el momento de que modelos que aprenden sus propias features del grafo molecular desplacen a la ingeniería manual.

¿Por qué grafos? Porque la simetría correcta de un sistema atómico es la **invariancia al isomorfismo de grafos**: reordenar los átomos no debe cambiar la predicción. Una molécula es naturalmente un grafo —nodos = átomos (tipo, número atómico, hibridación), aristas = enlaces (single/double/triple/aromatic o distancia espacial)—. Construir esa invariancia *en la arquitectura* es el *inductive bias* adecuado.

El estado del arte en 2017 era un **zoo fragmentado**: las *Convolutional Molecular Fingerprints* (Duvenaud et al. 2015), las *Gated Graph Neural Networks* (GG-NN, Li et al. 2016), las *Interaction Networks* (Battaglia et al. 2016), las *Molecular Graph Convolutions* (Kearnes et al. 2016), las *Deep Tensor NN* (Schütt et al. 2017) y los métodos espectrales basados en el Laplaciano (Bruna et al. 2013; Defferrard et al. 2016; Kipf & Welling 2016, los GCN). Cada uno con su notación y su motivación. Faltaba el marco que dijera "todos estos son la misma cosa con piezas distintas".

## El framework MPNN: el marco que unifica las GNN

La contribución conceptual es definir un **marco abstracto para aprendizaje supervisado sobre grafos** que captura las comunalidades de los modelos anteriores. El paso adelante de un MPNN sobre un grafo no dirigido, con features de nodo x_v y de arista e_vw, tiene **dos fases**:

1. **Message passing** (paso de mensajes), que corre durante **T pasos temporales**.
2. **Readout** (lectura), que produce la predicción a nivel de grafo.

Todo se especifica mediante **tres funciones diferenciables aprendidas**:

- **Función de mensaje M_t** — calcula el mensaje que cada vecino envía.
- **Función de actualización U_t** — combina los mensajes recibidos con el estado actual del nodo.
- **Función de readout R** — colapsa los estados finales de todos los nodos en una predicción a nivel de grafo.

Durante el message passing, los estados ocultos $h_v^t$ de cada nodo se actualizan según los mensajes $m_v^{t+1}$:

$$m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw}) \qquad (1)$$

$$h_v^{t+1} = U_t(h_v^t, m_v^{t+1}) \qquad (2)$$

donde $N(v)$ son los vecinos de $v$. La fase de readout calcula el vector del grafo entero:

$$\hat{y} = R(\{h_v^T \mid v \in G\}) \qquad (3)$$

Las cuatro etapas que enseña la clase se leen directamente de estas ecuaciones: **cálculo del mensaje** = evaluar $M_t$ para cada par $(v,w)$; **traspaso** = enviarlo por la arista; **combinación/agregación** = la suma $\sum_{w \in N(v)}$ sobre los vecinos; **actualización** = aplicar $U_t$. Este es exactamente el [mecanismo de message passing](/fundamentos/message-passing) que vertebra la clase.

Una restricción de diseño es **no negociable**: $R$ debe ser **invariante a permutaciones** de los estados de los nodos. Si $R$ dependiera del orden, el MPNN entero dejaría de ser invariante al isomorfismo de grafos y perdería la simetría que es la razón misma de usar grafos. Por eso $R$ suele ser una suma ($\sum_v$) o un mecanismo tipo conjunto (set2set) — operaciones que ignoran el orden.

## GGNN, GCN y otros como casos particulares

El insight pedagógico —y la razón por la que este paper define el vocabulario de la clase— es que **especificar $M_t$, $U_t$ y $R$ recupera cada modelo anterior**. El paper toma ocho arquitecturas publicadas y las reescribe en términos de las tres funciones:

- **[GG-NN](/papers/ggnn-li-2015) (Li et al. 2016):** $M_t = A_{e_{vw}} h_w^t$, con $A_e$ una matriz aprendida por etiqueta de arista (asume tipos discretos); $U_t = \text{GRU}(h_v^t, m_v^{t+1})$ con *weight tying* (la misma actualización en cada paso $t$); y un readout con compuertas $\sum_v \sigma(i(h_v^T, h_v^0)) \odot j(h_v^T)$. Es el baseline fuerte del que parten.
- **[GCN](/papers/gcn-kipf-2017) y métodos del Laplaciano (Kipf & Welling 2016, etc.):** generalizan la convolución a un grafo arbitrario. Resultan en $M_t = C_{vw}^t h_w^t$ con matrices parametrizadas por los autovectores del Laplaciano; $U_t = \sigma(m_v^{t+1})$. El caso de Kipf & Welling da el conocido $c_{vw} = (\deg(v)\deg(w))^{-1/2} A_{vw}$. Un apéndice deriva en detalle que los GCN equivalen a "tomar cierto promedio ponderado de los nodos vecinos en cada paso temporal" — lo que cierra el argumento de que *incluso los métodos espectrales* caben en el marco espacial.
- **Convolutional Molecular Fingerprints (Duvenaud et al. 2015):** $M = \text{concat}(h_w, e_{vw})$; $U_t = \sigma(H_t^{\deg(v)} m_v^{t+1})$. El paper *critica* este esquema: al sumar por separado sobre nodos y aristas, "es incapaz de identificar correlaciones entre estados de aristas y estados de nodos".
- **Deep Tensor NN (Schütt et al. 2017):** $U_t = h_v^t + m_v^{t+1}$ (actualización residual).

Así, una colección de arquitecturas dispares se vuelve una *familia parametrizada*, y queda claro qué grados de libertad explorar: qué $M$, qué $U$, qué $R$.

## Variantes propias

Partiendo de GG-NN, exploran funciones de mensaje, readout y representación de entrada. Su implementación opera sobre **grafos dirigidos** (canales separados para aristas entrantes y salientes), usa *weight tying* por paso y una **GRU** como función de actualización.

**Funciones de mensaje:**

- **Edge Network** (su propuesta principal): $M(h_v, h_w, e_{vw}) = A(e_{vw}) h_w$, donde $A(e_{vw})$ es una **red neuronal que mapea el vector de arista a una matriz $d \times d$**. Permite features de arista vectoriales (distancias continuas), no solo etiquetas discretas.
- **Pair Message:** $m_{wv} = f(h_w^t, h_v^t, e_{vw})$, dependiente de ambos nodos. Teóricamente más expresiva, pero entrenó peor que el edge network.

**Elementos virtuales** para capturar interacciones de largo alcance: *virtual edges* (aristas virtuales entre nodos no conectados) y *master node* (un nodo latente conectado a todos, como "espacio de borrador global").

**Funciones de readout:** además del readout con compuertas de GG-NN, **set2set** (Vinyals et al. 2015), diseñado para operar sobre conjuntos con más expresividad que una simple suma, y que produce un embedding del grafo **invariante al orden** de las tuplas — la pieza que respeta la invariancia exigida por el marco.

**Multiple Towers:** contribución de eficiencia. Un paso de message passing en grafo denso cuesta $O(n^2 d^2)$. La idea: partir el embedding de $d$ dimensiones en $k$ copias de $d/k$, propagar cada una por separado y luego mezclarlas con una red compartida que **preserva la invariancia a permutaciones**. El costo baja a $O(n^2 d^2 / k)$; con $k=8$ se observa 2× de speedup.

## Experimentos: QM9

**QM9 (Ramakrishnan et al. 2014)** consta de ~134k moléculas orgánicas (130.462 usadas) de H, C, O, N, F con hasta 9 átomos pesados, cada una con **13 propiedades** aproximadas por DFT — 13 tareas de regresión. Incluye la geometría 3D, lo que permite estudiar el caso *con* información espacial y el caso *solo topología*.

El criterio de éxito se mide como **ratio de error** = MAE del modelo / **chemical accuracy** del target (el error objetivo fijado por la comunidad química). Un ratio < 1 significa que se alcanzó precisión química. El setup: búsqueda aleatoria de hiperparámetros (50 trials por modelo/target), ADAM, 3 millones de pasos, $3 \le T \le 8$. Hallazgo importante: **entrenar un modelo por target supera consistentemente a entrenar uno solo para los 13** (hasta 40% de mejora).

**Resultados.** El mejor MPNN —**edge network + set2set + hidrógenos explícitos**, llamado *enn-s2s*— y su ensemble (*enn-s2s-ens5*) se comparan contra el estado del arte de Faber et al. (2017): cinco representaciones hechas a mano y dos MPNN baseline. **El nuevo MPNN logra estado del arte en los 13 targets y precisión química en 11 de 13.**

| Modelo | Ratio de error promedio |
|---|---|
| CM (Coulomb Matrix) | 53.97 |
| GC (Kearnes et al. 2016) | 2.59 |
| GG-NN (Li et al. 2016) | 1.36 |
| HDAD | 1.35 |
| **enn-s2s** | **0.68** |
| **enn-s2s-ens5** | **0.52** |

Solo *gap* (1.60) y *ZPVE* (1.27) quedan sobre 1 sin ensemble; con ensemble bajan a 1.23 y 1.10.

**Sin información espacial (solo topología):** lo que importa es *capturar interacciones de largo alcance*. Partiendo de GG-NN sobre el grafo disperso (ratio 3.47), agregar virtual edges baja a 2.90, master node a 2.62 y set2set a 2.57 — alcanzando precisión química en 5 de 13 targets *sin coordenadas atómicas*. **Towers:** GG-NN + towers + set2set supera al baseline en 12 de 13 targets. **Eficiencia de datos:** el edge network iguala con 11k muestras al mejor baseline entrenado con 110k. Y la entrada es decisiva: sin distancia 2.57, con distancia 0.98, con distancia + hidrógenos explícitos 0.68.

## Limitaciones reconocidas

- **Generalización a grafos más grandes:** el reto central declarado. Es difícil con información espacial porque la distribución de distancias por pares depende del número de átomos y el grafo se vuelve totalmente conectado. Sugieren un **mecanismo de atención** sobre los mensajes entrantes como dirección futura.
- **Escalabilidad:** un paso es $O(n^2 d^2)$ en grafo denso; towers ayuda, pero "se necesitarán mejoras adicionales".
- **El pair message no rindió:** pese a ser más expresivo, lo abandonaron.
- **Techo del benchmark:** el propio éxito es un límite. QM9 a 9 átomos pesados queda casi resuelto; el trabajo futuro debe ir a moléculas mayores o etiquetas más precisas que la propia DFT.

## Impacto: el paper que definió "message passing"

El impacto trasciende sus números en QM9. Su contribución más duradera es **terminológica y conceptual**: instaló "message passing" como *la* forma canónica de describir cualquier red neuronal sobre grafos. Después de 2017, prácticamente toda GNN nueva —GraphSAGE, GAT, GIN, los Graph Networks de Battaglia et al. 2018— se describe especificando su función de mensaje, su agregador y su update; es decir, *en los términos de este paper*. El framework $M_t / U_t / R$ se volvió el lenguaje común del subcampo, como "encoder-decoder" o "atención" en NLP.

Esto fue posible por el movimiento de síntesis: al demostrar que ocho arquitecturas eran instancias de un mismo esquema, el paper convirtió un catálogo de modelos en una *teoría* del diseño de GNN. La librería PyTorch Geometric organiza su clase base de capas convolucionales (`MessagePassing`) directamente alrededor de los tres métodos `message`, `aggregate`, `update` — una implementación literal de este marco.

## Por qué importa para la Clase 27

La [Clase 27](/clases/clase-27) ("Redes Neuronales de Grafos") no está meramente relacionada con este paper: **enseña su mecanismo como contenido central**.

- **El mecanismo de cuatro fases ES el framework MPNN.** Cuando la clase describe message passing como "cálculo del mensaje → traspaso → combinación/agregación → actualización", está describiendo la secuencia $M_t$ → envío por la arista → $\sum_{w \in N(v)}$ → $U_t$ de las ecuaciones (1)-(2). El readout para predicción a nivel de grafo es la ecuación (3).
- **MPNN como aplicación destacada (química cuántica).** La diapositiva que cita "Gilmer 2017" y contrasta el método lento (DFT) con la GNN rápida es la Figura 1 de este paper: una MPNN que *predice* el resultado de un cálculo DFT costoso. El "300.000× más rápido" es la cifra de la nota al pie.
- **Invariancia a permutación como principio de diseño.** La clase insiste en que las operaciones sobre [grafos](/fundamentos/redes-neuronales-de-grafos) deben ser invariantes/equivariantes a permutaciones; este paper lo formula como el requisito explícito sobre $R$.
- **Mapa mental del zoo de GNN.** Permite ver [GCN](/papers/gcn-kipf-2017), GAT, GraphSAGE y [GG-NN](/papers/ggnn-li-2015) no como una lista inconexa sino como puntos en un mismo espacio de diseño.
- **Aristas con features vectoriales.** El *edge network* $M(h_v, h_w, e_{vw}) = A(e_{vw}) \cdot h_w$ es el ejemplo de cómo incorporar información rica de arista más allá de la adyacencia binaria. Es un caso natural de [datos estructurados](/dominios/estructurados) llevado a aprendizaje profundo.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/1704.01212 — *Proceedings of the 34th ICML* (2017), Sídney, PMLR 70.
- Implementación canónica del marco: la clase base `MessagePassing` de PyTorch Geometric (`message` / `aggregate` / `update`).
- Dataset QM9: Ramakrishnan et al. (2014); benchmark de referencia para predicción de propiedades moleculares.
