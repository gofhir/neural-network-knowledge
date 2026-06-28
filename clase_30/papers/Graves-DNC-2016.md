# Hybrid computing using a neural network with dynamic external memory (DNC) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Hybrid computing using a neural network with dynamic external memory*.
- **Autores:** Alex Graves\* y Greg Wayne\* (contribución igualitaria), Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu y Demis Hassabis. Todos en **Google DeepMind** (Londres).
- **Venue:** *Nature*, vol. 538 (2016), artículo doi:10.1038/nature20101. Recibido el 5 de enero de 2016, aceptado el 19 de septiembre, publicado online el 12 de octubre de 2016.
- **Modelo presentado:** *Differentiable Neural Computer* (DNC).

**Nota sobre el estatus de este paper en la Clase 30.** Este es un paper **canónico** de la línea de modelos con memoria externa, pero **no aparece citado en las slides de la clase**, que se centran en la familia *Memory Networks* (Weston et al.) y sus variantes end-to-end. Lo incorporamos igualmente porque es **fundamental e ineludible** para entender el tema: el DNC es el **sucesor directo del Neural Turing Machine** (NTM, Graves, Wayne & Danihelka 2014) y representa la cumbre de la otra gran línea de memoria externa diferenciable —la línea "computador diferenciable", paralela y complementaria a la línea Memory Networks que sí discute la clase. Que esté publicado en *Nature* (algo rarísimo para un paper de arquitectura de redes neuronales) habla de su impacto y ambición: posicionar a las redes con memoria externa como un puente entre el procesamiento neuronal y la computación simbólica clásica.

En una frase: el DNC es **una red neuronal acoplada a una matriz de memoria externa direccionable**, donde un controlador recurrente aprende —por descenso de gradiente, de extremo a extremo— a leer y escribir en esa memoria como si fuera la RAM de un computador, pero **descubriendo desde los datos** los algoritmos de acceso. Frente al NTM, el DNC añade tres mecanismos nuevos que resuelven defectos concretos del predecesor: **asignación dinámica de memoria**, **enlaces temporales** entre escrituras, y **modos de lectura** mejorados (contenido + temporal hacia adelante/atrás).

## 2. Contexto histórico: por qué las redes neuronales necesitan memoria externa

El argumento de apertura del paper es una analogía con la arquitectura von Neumann. Los computadores modernos **separan cómputo y memoria**: un procesador trae operandos desde y hacia una memoria direccionable. Esto da dos beneficios cruciales: (1) almacenamiento extensible para escribir información nueva, y (2) la capacidad de tratar los contenidos de la memoria como **variables**. Las variables son críticas para la *generalidad algorítmica*: para aplicar el mismo procedimiento a un dato u otro, basta con cambiar la dirección de la que se lee.

Las redes neuronales, en contraste, **mezclan recursos de cómputo y memoria** en los pesos y en las activaciones de las neuronas. El paper lo llama un "pasivo importante" (*major liability*): a medida que las demandas de memoria de una tarea crecen, estas redes no pueden asignar almacenamiento nuevo dinámicamente, ni aprender con facilidad algoritmos que actúen con independencia de los valores concretos de las variables de la tarea. Científicos cognitivos y neurocientíficos (Gallistel & King; Marcus; Hinton) habían argumentado durante años que las redes neuronales son limitadas para representar variables y estructuras de datos, y para almacenar información en escalas de tiempo largas sin interferencia.

La propuesta es combinar lo mejor de ambos mundos: dotar a una red neuronal de **acceso de lectura-escritura a una memoria externa**. El acceso es deliberadamente **focalizado** (*narrowly focused*), lo que minimiza la interferencia entre memoranda y habilita almacenamiento de largo plazo. Y como todo el sistema es diferenciable, se entrena de extremo a extremo con descenso de gradiente: la red **aprende cómo operar y organizar la memoria de manera dirigida a un objetivo**.

El antecedente inmediato es el **Neural Turing Machine** (NTM, 2014), que ya tenía esta estructura de controlador + matriz de memoria con acceso de lectura-escritura. El DNC se distingue del NTM —y de marcos contemporáneos como Memory Networks y Pointer Networks— en que la memoria puede escribirse selectivamente además de leerse, permitiendo **modificación iterativa** del contenido.

## 3. Las limitaciones del NTM que el DNC viene a resolver

El paper dedica una sección de Métodos a comparar explícitamente con el NTM, y ahí está la motivación central. El NTM combinaba **direccionamiento por contenido** con **direccionamiento por ubicación**: este último permitía a la red iterar por las posiciones de memoria en orden de sus índices (la posición *n*, luego *n+1*, etc.), almacenando secuencias temporales en **bloques contiguos** de memoria. Suena razonable, pero tiene tres defectos graves:

1. **No previene el solapamiento de bloques.** El NTM no tiene mecanismo que garantice que los bloques de memoria asignados no se solapen e interfieran entre sí —un problema básico de gestión de memoria que cualquier sistema operativo resuelve. La asignación dinámica del DNC entrega **una ubicación libre individual a la vez**, sin importar el índice, por lo que no requiere bloques contiguos y la interferencia deja de ser un problema.

2. **No puede liberar memoria.** El NTM no tiene forma de liberar posiciones ya escritas y, por tanto, no puede reutilizar memoria al procesar secuencias largas. El DNC lo resuelve con las *free gates* (puertas de liberación) para la desasignación.

3. **El orden temporal se pierde con los saltos.** En el NTM la información secuencial se preserva **sólo mientras la red siga iterando por posiciones consecutivas**; en cuanto la cabeza de escritura salta a otra parte de la memoria (vía direccionamiento por contenido), el orden de las escrituras de antes y después del salto **no puede recuperarse**. La *temporal link matrix* del DNC no sufre este problema porque registra explícitamente el orden en que se hicieron las escrituras, independientemente de dónde caigan en la matriz.

Estos tres defectos motivan exactamente las tres contribuciones del DNC.

## 4. Contribución central: el Differentiable Neural Computer

El DNC es una red neuronal (el **controlador**) acoplada a una **matriz de memoria externa** $M \in \mathbb{R}^{N \times W}$ de $N$ posiciones ("locations") de ancho $W$. Si la memoria es la RAM del DNC, el controlador es un "CPU diferenciable" cuyas operaciones se aprenden por descenso de gradiente. El comportamiento de la red es **independiente del tamaño de la memoria** mientras ésta no se llene —razón por la cual la memoria se considera "externa".

En lugar de direcciones únicas, el DNC usa **mecanismos de atención diferenciable** que definen *distribuciones* sobre las $N$ filas de $M$. Estas distribuciones se llaman **weightings** (ponderaciones) y representan el grado en que cada posición participa de una lectura o escritura. La lectura es una suma ponderada: $r = \sum_{i=1}^{N} M[i,\cdot]\, w^r[i]$. La escritura primero **borra** con un vector de borrado $e$ y luego **añade** con un vector de escritura $v$: $M[i,j] \leftarrow M[i,j](1 - w^w[i]e[j]) + w^w[i]v[j]$. Las unidades que calculan y aplican las ponderaciones son las **cabezas de lectura y de escritura** (read/write heads).

El controlador recibe en cada paso $t$ una entrada $x_t$ y los $R$ vectores de lectura del paso anterior, y emite (a) una salida $y_t$ y (b) un **vector de interfaz** $\xi_t$ que parametriza toda su interacción con la memoria. En el paper el controlador es una variante profunda de **LSTM**, pero podría ser cualquier red (incluso feedforward). El vector de interfaz se subdivide en: claves y fuerzas de lectura, clave/fuerza/borrado/escritura de la cabeza de escritura, $R$ *free gates*, una *allocation gate*, una *write gate*, y $R$ *read modes*. La salida final combina la salida del controlador con los vectores de lectura más recientes, lo que permite condicionar las decisiones en la memoria que se acaba de leer (sin crear un ciclo en el grafo de cómputo).

Las cabezas usan **tres formas de atención diferenciable**, que son las tres contribuciones del DNC sobre el NTM.

### 4.1. Atención por contenido (heredada, refinada)

El *content lookup* compara una **clave** emitida por el controlador con el contenido de cada posición según una medida de similitud (aquí, **similitud coseno**), modulada por una **fuerza de clave** $\beta \in [1,\infty)$:

$$C(M, k, \beta)[i] = \frac{\exp\{D(k, M[i,\cdot])\,\beta\}}{\sum_j \exp\{D(k, M[j,\cdot])\,\beta\}}$$

Esto sirve para **recuerdo asociativo** (lecturas) o para modificar un vector existente (escritura). Un punto fino y poderoso: una clave que coincide **parcialmente** con el contenido puede igualmente atender fuertemente a esa posición, habilitando una forma de **pattern completion** —el valor recuperado puede incluir información que no estaba en la clave. Como el contenido de una dirección puede codificar referencias a otras direcciones, el *key-value retrieval* da un mecanismo rico para **navegar estructuras de datos asociativas** en la memoria.

### 4.2. Asignación dinámica de memoria (nuevo — resuelve los defectos 1 y 2 del NTM)

El DNC implementa un análogo diferenciable del esquema clásico de *free list* (lista de libres) de la gestión de memoria. Se mantiene un **vector de uso** $u_t \in [0,1]^N$, donde el uso de cada posición es un número entre 0 y 1. Cada escritura **incrementa** el uso de una posición (hasta un máximo de 1); cada lectura puede **decrementarlo** mediante las *free gates*, que el controlador emite (una por cabeza de lectura) para decidir si las posiciones recién leídas pueden liberarse:

$$\psi_t = \prod_{i=1}^{R}\left(1 - f^i_t\, w^{r,i}_{t-1}\right), \qquad u_t = (u_{t-1} + w^w_{t-1} - u_{t-1}\odot w^w_{t-1})\odot \psi_t$$

A partir del uso se ordena la **free list** $\phi_t$ (índices en orden ascendente de uso) y se construye la **ponderación de asignación** $a_t$, que entrega a la cabeza de escritura posiciones no usadas:

$$a_t[\phi_t[j]] = (1 - u_t[\phi_t[j]])\prod_{i=1}^{j-1} u_t[\phi_t[i]]$$

(El ordenamiento introduce discontinuidades en el gradiente que los autores simplemente ignoran, sin perjuicio observable para el aprendizaje.) Tres propiedades clave:

- El mecanismo es **independiente del tamaño y del contenido** de la memoria. Esto implica que un DNC puede **entrenarse con una memoria pequeña y luego escalarse a una más grande sin reentrenar** (Extended Data Fig. 2). En principio, habilita memoria externa *ilimitada*: bastaría añadir posiciones cada vez que el uso mínimo supere un umbral.
- Provee posiciones libres **una a una**, sin necesidad de bloques contiguos → resuelve la interferencia.
- Permite **reasignar** memoria que ya no se necesita → resuelve la incapacidad del NTM para liberar.

Los autores notan un paralelo neurocientífico: la modificación de memoria del DNC es rápida y *one-shot*, parecida a la potenciación de largo plazo de las sinapsis del hipocampo (CA3/CA1); y la asignación basada en uso con ponderaciones dispersas evoca el rol del giro dentado en aumentar la dispersión representacional y la capacidad de memoria.

### 4.3. Enlaces temporales (nuevo — resuelve el defecto 3 del NTM)

Un segundo mecanismo de atención registra las **transiciones entre posiciones escritas consecutivamente** en una matriz $L \in [0,1]^{N\times N}$. $L[i,j]$ se acerca a 1 si $i$ fue la posición escrita **inmediatamente después** de $j$. Para cualquier ponderación $w$, la operación $Lw$ **desplaza el foco hacia adelante** a las posiciones escritas después de las enfatizadas en $w$, mientras que $L^\top w$ lo desplaza **hacia atrás**. Esto le da al DNC la capacidad nativa de **recuperar secuencias en el orden en que las escribió, aun cuando las escrituras consecutivas no hayan ocurrido en pasos de tiempo adyacentes** —exactamente lo que el NTM no podía.

La matriz se mantiene con un **vector de precedencia** $p_t$ (cuánto fue cada posición la última escrita) y la recurrencia:

$$L_t[i,j] = (1 - w^w_t[i] - w^w_t[j])\, L_{t-1}[i,j] + w^w_t[i]\, p_{t-1}[j]$$

Cada escritura borra los enlaces viejos a/desde esa posición y añade enlaces nuevos desde la última escrita. La diagonal es siempre 0 (no tiene sentido seguir una transición de una posición a sí misma). De $L_t$ se derivan la ponderación **forward** $f^i_t = L_t\, w^{r,i}_{t-1}$ y la **backward** $b^i_t = L_t^\top\, w^{r,i}_{t-1}$ para cada cabeza de lectura $i$. El paper conecta esto con el *temporal context model* del recuerdo libre humano (mayor probabilidad de recordar ítems en el orden de presentación, un fenómeno dependiente del hipocampo).

**Costo y dispersión.** La matriz $L$ es $N\times N$, con costo $O(N^2)$ en memoria y cómputo. Como en la práctica es muy dispersa, se aproxima con costo $O(N\log N)$ en cómputo y $O(N)$ en memoria conservando sólo los $K$ valores más altos por fila/columna; con $K=8$ basta sin importar el tamaño de la memoria (Extended Data Fig. 4 muestra que no hay diferencia sistemática entre la versión densa y $K=5$).

### 4.4. Modos de lectura mejorados (nuevo)

Cada cabeza de lectura recibe un **read mode** $\pi^i_t \in S_3$ (una distribución sobre 3 opciones) que **interpola** entre la ponderación backward, la de contenido, y la forward:

$$w^{r,i}_t = \pi^i_t[1]\, b^i_t + \pi^i_t[2]\, c^{r,i}_t + \pi^i_t[3]\, f^i_t$$

Si domina $\pi^i_t[2]$, la lectura es por **contenido** (clave); si domina $\pi^i_t[3]$, la cabeza **itera hacia adelante** en orden de escritura ignorando la clave; si domina $\pi^i_t[1]$, **itera hacia atrás**. La cabeza de escritura, por su parte, interpola entre la ponderación de contenido $c^w_t$ y la de asignación $a_t$ mediante la *allocation gate* $g^a_t$ y la *write gate* $g^w_t$:

$$w^w_t = g^w_t\left[g^a_t\, a_t + (1 - g^a_t)\, c^w_t\right]$$

Si la *write gate* es 0, no se escribe nada (protege la memoria de modificaciones innecesarias). El diseño de las tres atenciones está motivado por consideraciones computacionales: **contenido** forma estructuras asociativas, **enlaces temporales** permiten recuperación secuencial, y **asignación** provee posiciones libres a la cabeza de escritura.

## 5. Experimentos

Los autores eligen tareas sintéticas de pequeña escala porque son fáciles de generar e interpretar; memorias de hasta 512 posiciones bastaron. Todas usan 20 redes con inicialización aleatoria, búsqueda en grilla de hiperparámetros, Downpour SGD distribuido y RMSProp.

### 5.1. Question answering: bAbI

El **bAbI** (Weston et al.) tiene 20 tipos de preguntas sintéticas que emulan razonamiento textual: historias cortas seguidas de preguntas cuya respuesta se infiere ("John está en el patio. John recogió la pelota. ¿Dónde está la pelota?" → "patio"; o deducción con distractores: "Las ovejas temen a los lobos. Gertrude es una oveja… ¿A qué le teme Gertrude?" → "lobos"). Un **único DNC entrenado conjuntamente** en los 20 tipos (10.000 instancias cada uno) logró **3,8% de error medio** con fallo (>5% error) en sólo 2 tipos, frente al 7,5% de error medio y 6 tareas falladas del mejor resultado conjunto previo (MemN2N). El DNC superó claramente a LSTM y NTM. A diferencia de trabajos previos, la entrada fueron **tokens de palabra individuales** (léxico de 156 palabras + 3 símbolos, vector one-hot de tamaño 159), **sin embeddings de oración ni preprocesamiento**, lo que produce secuencias mucho más largas y estresa más la memoria de largo alcance —y es más cercano al lenguaje natural real.

### 5.2. Razonamiento sobre grafos

Aunque bAbI viene en lenguaje natural, su conocimiento proposicional equivale a **restricciones sobre un grafo** (nodos y aristas etiquetados). Los autores pasan entonces a grafos explícitos, donde cada vector de entrada codifica una **tripla** (nodo origen, etiqueta de arista, nodo destino) con etiquetas numéricas 0–999 (cada dígito en one-hot). Tres consultas, todas con **aprendizaje por currículo** de complejidad creciente:

- **Traversal (recorrido):** seguir un camino desde un nodo inicial por una secuencia de aristas (random walk) y reportar el nodo de llegada. La red debe inferir el destino de cada tripla y recordarlo como origen implícito de la siguiente.
- **Shortest path (camino más corto):** dados nodo inicial y final, devolver la secuencia de triplas de un camino mínimo (hasta longitud 5 —más difícil que el máximo de 2 de bAbI). Incluye una fase de planificación de 10 pasos sin entrada y se trata como **predicción estructurada** (estilo DAgger), porque las decisiones de la red determinan el camino y, por ende, las decisiones futuras.
- **Inference (inferencia):** se predefinen 400 etiquetas de "relación" que abrevian secuencias de 2–5 aristas conectadas. La red recibe (nodo inicial, etiqueta de relación, _) y debe devolver el nodo final. Como **las secuencias de relación nunca se le presentan**, debe inferirlas sólo de las señales de error y hacer un recorrido implícito durante la fase de planificación.

**Generalización a grafos reales.** Tras entrenar sólo con grafos aleatorios, se probó **sin reentrenar** en dos grafos reales: el **mapa del metro de Londres** (estaciones de la Zona 1) y un **árbol genealógico** inventado. En el metro: recorridos aleatorios de 7 pasos con **98,8% de exactitud** y caminos más cortos de 4 pasos con 55,3%. En el árbol: relaciones de 4 pasos (p. ej., "tío abuelo materno") con **81,8%** de exactitud promedio. La visualización (Fig. 3) muestra que el DNC **escribe cada tripla en una posición separada** durante la definición del grafo, y luego usa la cabeza de lectura 1 (enlaces temporales hacia adelante) para recuperar instrucciones en orden y la cabeza 2 (contenido) para encontrar estaciones a lo largo del camino. El contraste con el baseline es brutal: el mejor LSTM tras búsqueda extensa de hiperparámetros **no completó ni el primer nivel** del currículo (37% de exactitud tras casi 2 millones de ejemplos), mientras el DNC alcanzó 98,8% tras ~1 millón.

### 5.3. Block puzzle: Mini-SHRDLU (aprendizaje por refuerzo)

Para investigar **planificación lógica**, los autores crearon Mini-SHRDLU, inspirado en el SHRDLU de Winograd: bloques numerados en una grilla $3\times3$ (hasta 6 bloques), donde un agente mueve el bloque superior de una columna a otra. A diferencia de los experimentos supervisados, aquí se usó **aprendizaje por refuerzo** (policy gradient con red de política + red de valor, ambas DNC). Se presentan varias metas posibles —cada una un conjunto de restricciones de adyacencia ("bloque 6 bajo 2; bloque 4 a la izquierda de 1; …"), transmitidas una restricción por paso— y luego se elige al azar una meta a satisfacer. El DNC **escribió las instrucciones iterativamente en posiciones de memoria** y luego ejecutó la meta elegida.

El hallazgo más notable: en 800 episodios, las primeras 5 acciones podían **decodificarse de la memoria justo después de escribir la meta**, muchos pasos antes de ejecutarla (89% de exactitud vs. 17% del baseline de frecuencias de acción). Esto indica que el DNC **había escrito su plan en memoria antes de actuar** —aprendió a planificar. Un t-SNE de los contenidos de memoria muestra que cada etiqueta de meta queda **codificada geométricamente**. De nuevo, sólo el DNC completó el currículo de aprendizaje; el LSTM no.

### 5.4. Validación de los mecanismos nuevos (Extended Data)

- **Asignación dinámica (Ext. Fig. 1):** en un copy problem con memoria de 10 posiciones insuficiente para las 50 entradas, el DNC reusó las mismas posiciones —la *free gate* activa en fase de lectura (desasigna) y la *allocation gate* en fase de escritura (reusa).
- **Escalado de memoria (Ext. Fig. 2):** un DNC entrenado en traversal con 256 posiciones se probó variando el número de posiciones; explota toda la memoria disponible sin importar con cuánta se entrenó → confirma que memoria y procesamiento son independientes.
- **Dispersión de enlaces (Ext. Fig. 4):** sin la matriz de enlaces, el copy problem no se resuelve (la secuencia debe recuperarse en orden); $K=5$ rinde igual que la versión densa.

## 6. Limitaciones reconocidas

- **Escala pequeña.** Los autores son explícitos: los experimentos son tareas sintéticas de pequeña escala (memorias de hasta 512 posiciones). Para datos del mundo real harían falta **miles o millones de posiciones**, punto en el cual la memoria podría almacenar más información de la que cabe en los pesos del controlador.
- **Costo cuadrático de la matriz de enlaces.** $O(N^2)$ en su forma exacta; la aproximación dispersa lo mitiga pero introduce un hiperparámetro $K$ y la suposición de dispersión.
- **Discontinuidades del gradiente.** El ordenamiento de la *free list* induce discontinuidades que se ignoran al calcular el gradiente.
- **Costo computacional del entrenamiento.** Requiere currículo cuidadoso (esencial salvo en bAbI), entrenamiento distribuido y BPTT con *gradient clipping* a $[-10,10]$ —no es un modelo fácil de entrenar ni barato.
- **Sintético vs. natural.** Aunque bAbI con tokens de palabra acerca el modelo al lenguaje natural, sigue siendo un dataset programáticamente generado de vocabulario limitado; queda como trabajo futuro la aplicación a datos naturalistas (los autores apuntan a one-shot learning, comprensión de escenas, procesamiento de lenguaje y mapeo cognitivo).

## 7. Impacto y lugar en la historia

El DNC es **la cumbre de la línea de memoria externa diferenciable**. Demostró por primera vez, de forma convincente y publicada en *Nature*, que una red neuronal puede aprender **razonamiento algorítmico y sobre grafos** —encontrar caminos, inferir relaciones, planificar— manipulando una memoria externa que opera como variables direccionables, todo aprendido desde los datos sin programación explícita. La transferencia *zero-shot* de grafos aleatorios al metro de Londres y a un árbol genealógico es la demostración emblemática de que el modelo aprendió un **procedimiento general** y no memorizó instancias.

Históricamente, el DNC cierra el arco NTM (2014) → DNC (2016) de "computadores neuronales diferenciables". En perspectiva, esta línea resultó costosa y difícil de entrenar a escala, y la comunidad pivotó hacia los **Transformers** (2017), cuya atención sobre toda la secuencia ofreció una forma de "memoria" más simple, paralelizable y escalable que terminó dominando. Pero las ideas del DNC —memoria direccionable por contenido, *key-value retrieval*, atención como mecanismo de acceso— resuenan directamente en la atención de los Transformers y en los sistemas modernos de **memoria/retrieval aumentada** (RAG, memorias de agentes). El DNC es el punto en que la metáfora "red neuronal como computador con RAM" se llevó tan lejos como pudo de manera completamente diferenciable.

## 8. Conexión con la Clase 30 (Modelos con memoria externa)

La Clase 30 se organiza, en las slides, alrededor de la línea **Memory Networks** (Weston et al.) y sus versiones end-to-end: redes que almacenan hechos en una memoria y los consultan por atención para responder preguntas. El DNC representa la **otra gran línea** del mismo problema —la línea "computador diferenciable" NTM→DNC— y por eso lo incorporamos aunque no esté citado: **completa el mapa** del tema. Las dos líneas comparten la idea raíz (separar cómputo de una memoria externa accedida por atención diferenciable), pero difieren en énfasis: Memory Networks se centra en **leer** una memoria poblada con hechos para QA, mientras que el DNC añade **escritura iterativa, asignación/liberación dinámica y orden temporal**, acercándose más a un modelo de cómputo general.

Conviene fijar el linaje para la clase:

- **NTM (2014)** → primer controlador + memoria de lectura-escritura diferenciable; direccionamiento por contenido + ubicación. Ver el análisis en [`/papers/ntm-graves-2014`]({{< relref "papers/ntm-graves-2014" >}}).
- **DNC (2016, este paper)** → NTM mejorado con asignación dinámica, enlaces temporales y modos de lectura; razonamiento sobre grafos en *Nature*.
- **MANN (Santoro et al. 2016)** → adapta la idea de memoria externa al **meta-aprendizaje** few-shot, conectando esta clase con la Clase 26. Ver [`/papers/mann-santoro-2016`]({{< relref "papers/mann-santoro-2016" >}}).

Para los fundamentos transversales del mecanismo (memoria como matriz externa, ponderaciones, atención por contenido vs. ubicación, lectura/escritura diferenciable), ver [`/fundamentos/memory-augmented-networks`]({{< relref "fundamentos/memory-augmented-networks" >}}). El hub de la clase está en [`/clases/clase-30`]({{< relref "clases/clase-30" >}}).
