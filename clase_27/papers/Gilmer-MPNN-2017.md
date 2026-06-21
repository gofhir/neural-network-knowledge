# Neural Message Passing for Quantum Chemistry — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Neural Message Passing for Quantum Chemistry*.
- **Autores:** Justin Gilmer (Google Brain), Samuel S. Schoenholz (Google Brain), Patrick F. Riley (Google), Oriol Vinyals (Google DeepMind), George E. Dahl (Google Brain).
- **Venue:** *Proceedings of the 34th International Conference on Machine Learning* (ICML 2017), Sídney, Australia, PMLR 70.
- **Año:** 2017. **Preprint:** arXiv:1704.01212v2 (12 jun 2017), [arxiv.org/abs/1704.01212](https://arxiv.org/abs/1704.01212).
- **Correspondencia:** Justin Gilmer (gilmer@google.com), George E. Dahl (gdahl@google.com).

Este paper hace dos cosas a la vez, y ahí radica su importancia histórica. Primero, es un **paper de síntesis**: toma "al menos ocho ejemplos notables" de modelos de redes neuronales sobre grafos publicados entre 2013 y 2017 —que hasta entonces se presentaban como arquitecturas dispares, cada una con su propia notación y motivación— y demuestra que todos son **casos particulares de un mismo marco abstracto** al que llaman *Message Passing Neural Networks* (MPNN). Segundo, es un **paper empírico**: una vez establecido el marco unificador, explora variaciones novedosas dentro de él y obtiene estado del arte en QM9, un benchmark importante de predicción de propiedades moleculares cuánticas, alcanzando precisión química ("chemical accuracy") en 11 de 13 propiedades.

La tesis del paper, declarada en el abstract con una franqueza poco habitual, es que el campo ya tenía suficientes *variantes* de modelos invariantes a las simetrías moleculares; lo que faltaba era **encontrar una variante particularmente efectiva de este enfoque general y aplicarla a benchmarks de predicción química hasta resolverlos o alcanzar los límites del enfoque**. En sus propias palabras: los resultados que obtienen son "lo bastante fuertes" como para creer que el trabajo futuro debería enfocarse en *moléculas más grandes* o en *etiquetas de ground truth más precisas* — es decir, el problema de QM9 a 9 átomos pesados queda esencialmente cerrado.

Para la Clase 27 (Redes Neuronales de Grafos) este paper importa por una razón estructural, no solo histórica: **el mecanismo central que la clase enseña —"message passing" en cuatro fases: cálculo del mensaje, traspaso, combinación/agregación y actualización— es literalmente el framework de este paper.** Cuando la clase escribe en una diapositiva "Gilmer 2017, química cuántica: el método tradicional (DFT) es lento, la GNN es rápida", está citando exactamente la Figura 1 de este artículo. Es, en sentido estricto, el paper que le dio nombre al mecanismo que vertebra toda la clase.

## 2. Contexto histórico: predicción de propiedades moleculares y el zoo de GNN sin marco unificador

### 2.1. El problema de fondo: DFT es exacto pero lentísimo

La predicción de propiedades de moléculas es, en principio, un problema resuelto por la física: la mecánica cuántica nos permite calcular las propiedades de una molécula. El problema es que las ecuaciones que dicta la física "son demasiado difíciles de resolver exactamente". Por eso los científicos desarrollaron una jerarquía de aproximaciones a la mecánica cuántica con distintos compromisos de velocidad y precisión: la **Teoría del Funcional de la Densidad (DFT)** con diversos funcionales (Becke 1993; Hohenberg & Kohn 1964), la aproximación GW (Hedin 1965), Quantum Monte-Carlo (Ceperley & Alder 1986).

DFT es la herramienta de trabajo del campo, pero tiene dos defectos que el paper cuantifica con números concretos y memorables. Es **demasiado lenta** —escala como O(N_e³) donde N_e es el número de electrones— y además exhibe errores sistemáticos y aleatorios respecto a las soluciones exactas de la ecuación de Schrödinger. La cifra clave: correr DFT sobre una sola molécula de 9 átomos pesados de QM9 toma alrededor de **una hora** en un núcleo de un Xeon E5-2660 (2.2 GHz) usando Gaussian G09; para una molécula de 17 átomos pesados, hasta **8 horas**. Por contraste, el paper señala en una nota al pie que el tiempo de inferencia de las redes neuronales discutidas es **300.000 veces más rápido** (la Figura 1 lo ilustra como DFT ≈ 10³ segundos vs MPNN ≈ 10⁻² segundos). Ese es el premio: aprender una función rápida que imite el cálculo costoso de DFT.

El paper enmarca el momento como análogo al estado de la visión por computador *antes* de la adopción masiva de las CNN: las redes neuronales ya se habían aplicado a la química (Merkwirth & Lengauer 2005; Micheli 2009; Lusci et al. 2013; Duvenaud et al. 2015) pero no se habían adoptado ampliamente, en parte por falta de evidencia empírica de que arquitecturas con el *inductive bias* apropiado pudieran tener éxito. La mayor parte del ML aplicado a química giraba en torno a la **ingeniería manual de features** (Coulomb Matrix, Bag of Bonds, etc.). El paper apuesta a que, como pasó con AlexNet desplazando a los SVM sobre features hechos a mano, llegó el momento de que modelos que aprenden sus propias features directamente del grafo molecular desplacen a la ingeniería manual.

### 2.2. La simetría correcta: invariancia al isomorfismo de grafos

¿Por qué grafos? Porque las simetrías de los sistemas atómicos sugieren que las redes adecuadas son las que operan sobre datos estructurados como grafo y son **invariantes al isomorfismo de grafos** — es decir, reordenar los átomos no debe cambiar la predicción. Una molécula es naturalmente un grafo: nodos = átomos (con features x_v: tipo de átomo, número atómico, hibridación, etc.), aristas = enlaces (con features e_vw: tipo de enlace single/double/triple/aromatic, o distancia espacial). Construir la invariancia *en la arquitectura* (en lugar de aprenderla por *data augmentation*, como hacían Rupp et al. 2012) es el inductive bias correcto.

### 2.3. El zoo de variantes sin marco

El estado del arte en 2017 era un conjunto fragmentado de modelos cercanos pero presentados de forma independiente: las *Convolutional Networks for Molecular Fingerprints* de Duvenaud et al. (2015), las *Gated Graph Neural Networks* (GG-NN) de Li et al. (2016), las *Interaction Networks* de Battaglia et al. (2016), las *Molecular Graph Convolutions* de Kearnes et al. (2016), las *Deep Tensor Neural Networks* de Schütt et al. (2017), y los métodos espectrales basados en el Laplaciano del grafo de Bruna et al. (2013), Defferrard et al. (2016) y Kipf & Welling (2016) — los GCN. Cada uno con su notación, su motivación, su dominio. Faltaba el marco que dijera "todos estos son la misma cosa con piezas distintas".

## 3. Contribución central: el framework MPNN que unifica las GNN

La contribución conceptual es definir un **framework abstracto para aprendizaje supervisado sobre grafos** que captura las comunalidades de los modelos anteriores. El paso adelante (*forward pass*) de un MPNN sobre un grafo no dirigido G con features de nodo x_v y features de arista e_vw tiene **dos fases**:

1. **Fase de message passing** (paso de mensajes), que corre durante **T pasos temporales**.
2. **Fase de readout** (lectura), que produce la predicción a nivel de grafo.

El marco se especifica enteramente mediante **tres funciones diferenciables aprendidas**:

- **Función de mensaje M_t** — calcula el mensaje que cada vecino envía.
- **Función de actualización del vértice U_t** — combina los mensajes recibidos con el estado actual del nodo.
- **Función de readout R** — colapsa los estados finales de todos los nodos en una predicción a nivel de grafo.

El insight pedagógico clave —y la razón por la que este paper define el vocabulario de toda la clase— es que **especificar M_t, U_t y R recupera cada modelo anterior como caso particular**. El paper dedica la Sección 2 a hacer exactamente esto, tomando ocho modelos publicados y reescribiéndolos en términos de las tres funciones. Esto convierte una colección de arquitecturas en una *familia parametrizada*, y deja claro qué grados de libertad quedan por explorar.

Una restricción de diseño es no negociable: **R debe ser invariante a permutaciones** de los estados de los nodos. Si R no fuera invariante al reordenamiento de los nodos, el MPNN entero no sería invariante al isomorfismo de grafos, y perdería la simetría que es la razón misma para usar grafos. Por eso R suele ser una suma (∑_v) o un mecanismo tipo conjunto (set2set) — operaciones que ignoran el orden.

## 4. Método en detalle

### 4.1. Las ecuaciones del marco

Durante la fase de message passing, los estados ocultos h_v^t de cada nodo se actualizan según mensajes m_v^{t+1}:

$$m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw}) \qquad (1)$$

$$h_v^{t+1} = U_t(h_v^t, m_v^{t+1}) \qquad (2)$$

donde N(v) son los vecinos de v en el grafo. La fase de readout calcula el vector de features de todo el grafo:

$$\hat{y} = R(\{h_v^T \mid v \in G\}) \qquad (3)$$

Las cuatro etapas que la clase enseña se leen directamente de estas ecuaciones: **cálculo del mensaje** = evaluar M_t para cada par (v,w); **traspaso** = enviar M_t a lo largo de la arista; **combinación/agregación** = la suma ∑_{w∈N(v)} sobre los vecinos; **actualización** = aplicar U_t. El paper también nota que se pueden aprender features de arista introduciendo estados ocultos h_{evw}^t para todas las aristas y actualizándolos análogamente a (1) y (2) — de los MPNN existentes, solo Kearnes et al. (2016) usó esa idea.

### 4.2. Cómo los modelos previos son instancias del marco

El paper reescribe cada modelo. Algunos ejemplos cruciales:

- **GG-NN (Li et al. 2016):** M_t(h_v^t, h_w^t, e_vw) = A_{e_vw} h_w^t, donde A_e es una matriz aprendida por etiqueta de arista (asume tipos discretos de arista); U_t = GRU(h_v^t, m_v^{t+1}) con *weight tying* (la misma función de actualización en cada paso t); y un readout R con compuertas de la forma ∑_v σ(i(h_v^T, h_v^0)) ⊙ j(h_v^T). Este es el modelo del que parten los autores como baseline fuerte.
- **Convolutional Molecular Fingerprints (Duvenaud et al. 2015):** M = concat(h_w, e_vw); U_t = σ(H_t^{deg(v)} m_v^{t+1}) con una matriz aprendida por grado de vértice; R con skip connections a todos los estados previos. El paper *critica* este esquema: como el mensaje resulta ser ∑(h_w, e_vw), que suma por separado sobre nodos y aristas conectados, el modelo "es incapaz de identificar correlaciones entre estados de aristas y estados de nodos".
- **Interaction Networks (Battaglia et al. 2016):** M y U son redes neuronales sobre concatenaciones; U toma (h_v, x_v, m_v) donde x_v es una influencia externa al vértice; R = f(∑_v h_v^T). El modelo original solo se definió para T=1.
- **Deep Tensor NN (Schütt et al. 2017):** M_t = tanh(W^fc((W^cf h_w^t + b1) ⊙ (W^df e_vw + b2))); U_t(h_v^t, m_v^{t+1}) = h_v^t + m_v^{t+1} (actualización residual); R suma salidas de una NN por nodo.
- **Métodos basados en el Laplaciano / GCN (Bruna et al. 2013; Defferrard et al. 2016; Kipf & Welling 2016):** generalizan la convolución a un grafo arbitrario con matriz de adyacencia real. Resultan en M_t(h_v^t, h_w^t) = C_vw^t h_w^t con matrices parametrizadas por los autovectores del Laplaciano del grafo; U_t = σ(m_v^{t+1}). El caso de Kipf & Welling da el conocido c_vw = (deg(v)deg(w))^{-1/2} A_vw. El Apéndice 10.1 deriva con todo detalle —incluyendo un tensor de rango 4 L̃ y la regla de propagación H^{l+1} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^l W^l)— que los GCN equivalen a "tomar un cierto promedio ponderado de los nodos vecinos en cada paso temporal". Esta derivación es lo que cierra el argumento de que *incluso los métodos espectrales* caben en el marco espacial de message passing.

### 4.3. Variantes propias que prueban

Partiendo de GG-NN como baseline, exploran funciones de mensaje, funciones de readout, representación de entrada e hiperparámetros. Usan d para la dimensión del estado oculto de cada nodo y n para el número de nodos. Su implementación opera sobre **grafos dirigidos** con canales de mensaje separados para aristas entrantes y salientes (m_v = concat de m_v^in y m_v^out); tratar un grafo no dirigido como dirigido duplica el tamaño del canal a 2d. Los estados iniciales h_v^0 se fijan a los features de átomo x_v, paddeados a dimensión d. Todos los experimentos usan *weight tying* por paso temporal y una **GRU** como función de actualización (como en GG-NN).

**Funciones de mensaje M (Sección 5.1):**

- **Matrix Multiplication:** la de GG-NN, M = A_{e_vw} h_w (requiere etiquetas discretas de arista).
- **Edge Network:** su propuesta principal. M(h_v, h_w, e_vw) = A(e_vw) h_w, donde A(e_vw) es una **red neuronal que mapea el vector de arista e_vw a una matriz d×d**. Esto permite features de arista con valores vectoriales (distancias continuas), no solo etiquetas discretas.
- **Pair Message:** inspirada en Battaglia et al. (2016), m_wv = f(h_w^t, h_v^t, e_vw) — el mensaje depende de *ambos* nodos fuente y destino. En teoría usa el canal de mensaje más eficientemente, pero en la práctica entrenó peor que el edge network.

**Elementos virtuales del grafo (Sección 5.2):** dos formas de cambiar cómo viajan los mensajes para capturar interacciones de largo alcance:
- **Virtual edges (aristas virtuales):** un tipo de arista "virtual" separado para pares de nodos no conectados; se implementa como preprocesamiento de datos y permite que la información viaje largas distancias en la fase de propagación.
- **Master node (nodo maestro):** un nodo latente conectado a *todos* los nodos del grafo con un tipo de arista especial, que sirve de "espacio de borrador global" del que cada nodo lee y al que escribe en cada paso. Puede tener dimensión propia d_master y pesos propios (otra GRU). Complejidad O(|E|d² + n·d_master²).

**Funciones de readout R (Sección 5.3):**
- El readout con compuertas de GG-NN (ecuación 4).
- **set2set (Vinyals et al. 2015):** diseñado específicamente para operar sobre conjuntos, con más poder expresivo que sumar los estados finales. Proyecta linealmente cada tupla (h_v^T, x_v), y tras M pasos de cómputo produce un embedding a nivel de grafo q_t* **invariante al orden** de las tuplas, que se pasa por una NN para la salida. Esta es la pieza que respeta la invariancia a permutación exigida por el marco.

**Multiple Towers (Sección 5.4):** la contribución de eficiencia. Un paso de message passing en grafo denso cuesta O(n²d²) multiplicaciones. La idea: partir el embedding de d dimensiones en k embeddings de d/k dimensiones, correr la propagación en cada copia por separado, y luego mezclarlas con una red g compartida (ecuación 5) que **preserva la invariancia a permutaciones**. Con multiplicación matricial el costo baja a O(n²d²/k); para k=8, n=9, d=200 se observa 2× de speedup en inferencia. Esto entrena modelos con representaciones de nodo más grandes sin un aumento correspondiente en cómputo o memoria — la tercera contribución clave del abstract.

### 4.4. Representación de entrada (Sección 6)

Los features de átomo (Tabla 1) incluyen: tipo de átomo (H, C, N, O, F one-hot), número atómico, aceptor/donante de electrones, aromaticidad, hibridación (sp/sp²/sp³), y número de hidrógenos. Experimentan con hacer los **hidrógenos nodos explícitos** (grafos de hasta 29 nodos en vez de hasta 9 átomos pesados), lo que ralentiza el entrenamiento ~10× pero ayuda en varios targets. Tres representaciones de arista según el modelo: **chemical graph** (tipos de enlace discretos: single/double/triple/aromatic), **distance bins** (distancias binadas en 14 símbolos para la función de multiplicación matricial), y **raw distance feature** (vector de 5 dimensiones: distancia euclídea + one-hot del tipo de enlace, para el edge network).

## 5. Experimentos: QM9

### 5.1. El dataset y el criterio de éxito

**QM9 (Ramakrishnan et al. 2014)** consta de ~134k (130.462 usadas) moléculas orgánicas tipo fármaco compuestas de H, C, O, N, F con hasta 9 átomos pesados, cada una con **13 propiedades** aproximadas por simulación DFT, dando 13 tareas de regresión. QM9 incluye también la geometría 3D completa de la conformación de baja energía, lo que permite estudiar tanto el caso *con* información espacial como el caso *solo topología* (sin coordenadas). Las 13 propiedades se agrupan en cuatro categorías: energías de atomización (U0, U, H, G), vibraciones fundamentales (ω1, ZPVE), estados de los electrones (HOMO, LUMO, gap Δε), y distribución espacial de electrones (⟨R²⟩, momento dipolar μ, polarizabilidad α; más Cv).

Hay dos niveles de error de referencia: el **"DFT error"** (error promedio estimado de la propia aproximación DFT respecto a la naturaleza) y la **"chemical accuracy"** (error objetivo establecido por la comunidad química; los estimados de ambos para los 13 targets vienen de Faber et al. 2017, Tabla 5). Las tablas reportan el **ratio de error** = MAE del modelo / chemical accuracy del target; un ratio < 1 significa que se alcanzó precisión química.

### 5.2. Setup de entrenamiento

Búsqueda aleatoria de hiperparámetros con 50 trials por combinación modelo/target. T restringido a 3 ≤ T ≤ 8 (en la práctica cualquier T ≥ 3 funciona); M (pasos de set2set) en 1 ≤ M ≤ 12. ADAM, batch 20, 3 millones de pasos (~540 épocas), learning rate inicial uniforme entre 1e-5 y 5e-4 con decaimiento lineal. 10.000 muestras para validación, 10.000 para test, resto para entrenamiento; early stopping en validación. Se minimiza MSE pero se evalúa MAE. Hallazgo importante: **entrenar un modelo por target supera consistentemente a entrenar uno solo para los 13** (mejora de hasta 40% en algunos casos).

### 5.3. Resultados reales

El mejor MPNN —**edge network + set2set + hidrógenos explícitos**, denotado *enn-s2s*— y su ensemble de los 5 mejores modelos (*enn-s2s-ens5*) se comparan en la Tabla 2 contra el estado del arte de Faber et al. (2017): cinco representaciones hechas a mano (BAML, BOB, CM, ECFP4, HDAD) y dos MPNN baseline (GC de Kearnes et al. 2016; GG-NN de Li et al. 2016). **El nuevo MPNN logra estado del arte en los 13 targets y precisión química en 11 de 13.** El ratio de error promedio: enn-s2s = **0.68**, enn-s2s-ens5 = **0.52**, frente a HDAD = 1.35, GG-NN = 1.36, GC = 2.59 y CM = 53.97. Solo gap (1.60) y ZPVE (1.27) quedan por encima de 1 sin ensemble; con ensemble bajan a 1.23 y 1.10.

**Entrenamiento sin información espacial (solo topología).** Aquí el resultado central es que *capturar interacciones de largo alcance es lo que importa*. Partiendo de GG-NN sobre el grafo disperso (ratio promedio 3.47), agregar virtual edges baja a 2.90, master node a 2.62, y cambiar el readout a set2set a 2.57 (Tabla 3 y Tabla 7). El modelo con set2set alcanza precisión química en 5 de 13 targets *sin coordenadas atómicas en absoluto* — la segunda contribución del abstract. (Nota: las Tablas 3 y 4 usan un feature de carga parcial que es salida de DFT y no estaría disponible en aplicación real; los números de estado del arte de la Tabla 2 *no* lo usan.)

**Towers.** GG-NN + towers + set2set supera al GG-NN + set2set baseline en **12 de 13 targets** tanto en entrenamiento conjunto (1.75 vs 1.92) como individual (1.37 vs 1.53), Tablas 4 y 8. Los autores conjeturan que el beneficio es que "se parece a entrenar un ensemble de modelos". Combinar towers con el edge network, sin embargo, *no* mejoró el rendimiento (posiblemente porque dificulta el entrenamiento).

**Eficiencia de datos (Tabla 6).** El edge network + set2set es sorprendentemente eficiente en datos: en R² y Omega iguala o supera con **11k muestras** al mejor baseline entrenado con 110k. El edge network supera consistentemente al pair message (Tabla 9: promedio conjunto 1.53 vs 3.98). Y la información de entrada es decisiva (Tabla 10): sin distancia 2.57, con distancia 0.98, con distancia + hidrógenos explícitos 0.68.

## 6. Limitaciones reconocidas

El paper es explícito sobre dónde *no* llega, y lo enmarca como agenda futura:

- **Generalización a grafos más grandes.** El reto central declarado: diseñar MPNN que generalicen a grafos más grandes que los del entrenamiento. Es "particularmente desafiante" con información espacial, por dos razones: (1) la distribución de distancias por pares depende fuertemente del número de átomos; (2) las formas más exitosas de usar info espacial crean un grafo *totalmente conectado* donde el número de mensajes entrantes también depende del número de nodos. Para esto último sugieren un **mecanismo de atención** sobre los vectores de mensaje entrantes como dirección interesante.
- **Escalabilidad.** Un paso de message passing es O(n²d²) en grafo denso; towers ayuda pero "se necesitarán mejoras adicionales para escalar a grafos mucho más grandes".
- **El pair message no rindió.** Pese a ser teóricamente más expresivo, entrenó peor que el edge network y lo abandonaron.
- **Techo del benchmark.** El propio éxito es un límite: QM9 a 9 átomos pesados queda casi resuelto, por lo que el trabajo futuro debería ir a moléculas mayores (p.ej. GDB-17, Ruddigkeit et al. 2012) o etiquetas más precisas que la propia DFT.

## 7. Impacto: el marco que definió el término "message passing"

El impacto de este paper trasciende sus resultados en QM9. Su contribución más duradera es **terminológica y conceptual**: instaló "message passing" como *la* forma canónica de describir cualquier red neuronal sobre grafos. Después de 2017, prácticamente toda GNN nueva —GraphSAGE, GAT, GIN, los Graph Networks de Battaglia et al. 2018— se describe especificando su función de mensaje, su agregador y su update; es decir, *en los términos de este paper*. El framework M_t / U_t / R se volvió el lenguaje común del subcampo, de la misma manera que "encoder-decoder" o "atención" lo son en NLP.

Esto fue posible precisamente por el movimiento de síntesis: al demostrar que ocho arquitecturas dispares eran instancias de un mismo esquema, el paper convirtió un catálogo de modelos en una *teoría* del diseño de GNN, donde las elecciones de M, U y R son los grados de libertad. Cualquier investigador podía entonces situar su modelo en ese espacio y razonar sobre qué piezas variar. La librería PyTorch Geometric, de hecho, organiza su clase base de capas convolucionales (`MessagePassing`) directamente alrededor de los tres métodos `message`, `aggregate`, `update` — una implementación literal de este marco.

## 8. Conexión con la Clase 27 (Redes Neuronales de Grafos)

La clase y este paper no están meramente relacionados: la clase **enseña el mecanismo de este paper como su contenido central**, y lo presenta como aplicación destacada.

- **El mecanismo de cuatro fases ES el framework MPNN.** Cuando la Clase 27 describe message passing como "cálculo del mensaje → traspaso → combinación/agregación → actualización", está describiendo exactamente la secuencia M_t (cálculo) → envío por la arista (traspaso) → ∑_{w∈N(v)} (combinación, ecuación 1) → U_t (actualización, ecuación 2) de este artículo. La fase de readout R de la clase, para predicción a nivel de grafo, es la ecuación 3. El estudiante que entiende este paper entiende el corazón conceptual de toda la clase, no un apéndice de ella.
- **MPNN como aplicación destacada (química cuántica).** La diapositiva que cita "Gilmer 2017" y contrasta el método tradicional lento (DFT) contra la GNN rápida es, literalmente, la Figura 1 de este paper: una MPNN que *predice* el resultado de un cálculo DFT computacionalmente caro. El número de la clase —"la GNN es mucho más rápida"— es la cifra de 300.000× más rápido de la nota al pie, o el 10³ s vs 10⁻² s de la figura. Es el ejemplo canónico de por qué las GNN importan en ciencia.
- **Invariancia a permutación como principio de diseño.** La clase insiste en que las operaciones sobre grafos deben ser invariantes/equivariantes a permutaciones de nodos; este paper lo formula como el requisito explícito sobre R y lo materializa en sus elecciones (suma, set2set, la red de mezcla de towers que preserva la invariancia). El paper es el ejemplo concreto de *por qué* ese principio no es opcional.
- **Unificación de arquitecturas como mapa mental.** La clase típicamente presenta varias GNN (GCN, GAT, GraphSAGE...). Este paper le da al estudiante la herramienta para no verlas como una lista inconexa sino como puntos en un mismo espacio de diseño (qué M, qué U, qué R) — exactamente el aporte de síntesis que hace navegable el zoo de GNN.
- **Aristas con features vectoriales y el edge network.** La clase, al tratar grafos con atributos de arista (distancias, tipos de enlace), conecta con la innovación principal del paper: el *edge network* M(h_v, h_w, e_vw) = A(e_vw)·h_w, que aprende a transformar un vector de arista en una matriz de paso de mensaje. Es el ejemplo de cómo incorporar información rica de arista, más allá de la simple adyacencia binaria de un GCN básico.
