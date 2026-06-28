---
title: "Differentiable Neural Computer (2016)"
weight: 346
math: true
---

{{< paper-card
    title="Hybrid computing using a neural network with dynamic external memory"
    authors="Alex Graves, Greg Wayne, Malcolm Reynolds, et al."
    year="2016"
    venue="Nature 2016"
    pdf="/papers/dnc-graves-2016.pdf" >}}
Paper de Google DeepMind publicado en **Nature** que presenta el **Differentiable Neural Computer (DNC)**, sucesor directo del [Neural Turing Machine](/papers/ntm-graves-2014). Una red neuronal (el controlador) acoplada a una **matriz de memoria externa** que aprende, por descenso de gradiente y de extremo a extremo, a leer y escribir como si la memoria fuera la RAM de un computador. Frente al NTM, añade tres mecanismos: **asignación dinámica de memoria** (usage/free gates), **matriz de enlaces temporales** que registra el orden de las escrituras, y **modos de lectura** que combinan contenido y orden temporal. Demostró razonamiento sobre **grafos** —metro de Londres, árbol genealógico— y QA con **bAbI**, transfiriendo sin reentrenar desde grafos aleatorios. Es la cumbre de la línea de memoria externa diferenciable.
{{< /paper-card >}}

---

> **Nota sobre la Clase 30.** Este es un paper **canónico** pero **no aparece citado en las slides de la clase**, que se organizan alrededor de la familia *Memory Networks* (Weston et al.) y sus variantes end-to-end. Lo incorporamos porque es **ineludible** para entender el tema: el DNC es el sucesor directo del [NTM](/papers/ntm-graves-2014) y representa la otra gran línea de memoria externa diferenciable —la línea "computador diferenciable"—, complementaria a la línea Memory Networks. Que esté publicado en *Nature* (algo rarísimo para un paper de arquitectura de redes) habla de su impacto y ambición.

## Contexto: por qué las redes neuronales necesitan memoria externa

El argumento de apertura es una analogía con la arquitectura von Neumann. Los computadores modernos **separan cómputo y memoria**: un procesador trae operandos desde y hacia una memoria direccionable. Esto da dos beneficios cruciales: almacenamiento extensible para escribir información nueva, y la capacidad de tratar los contenidos de la memoria como **variables**. Las variables habilitan la generalidad algorítmica: para aplicar el mismo procedimiento a un dato u otro, basta cambiar la dirección de la que se lee.

Las redes neuronales, en cambio, **mezclan cómputo y memoria** en los pesos y las activaciones. El paper lo llama un "pasivo importante": cuando la demanda de memoria de una tarea crece, estas redes no pueden asignar almacenamiento nuevo dinámicamente ni aprender con facilidad algoritmos independientes de los valores concretos de las variables. La propuesta: dotar a una red de **acceso de lectura-escritura a una memoria externa**, deliberadamente **focalizado** para minimizar la interferencia entre memoranda y habilitar almacenamiento de largo plazo. Como todo el sistema es diferenciable, se entrena de extremo a extremo: la red **aprende cómo operar y organizar la memoria** dirigida a un objetivo.

El antecedente inmediato es el **Neural Turing Machine** (NTM, 2014), que ya tenía la estructura de controlador + matriz de memoria. El DNC se distingue del NTM —y de marcos contemporáneos como Memory Networks o Pointer Networks— en que la memoria se escribe selectivamente además de leerse, permitiendo **modificación iterativa** del contenido.

## Las limitaciones del NTM que el DNC resuelve

El NTM combinaba direccionamiento por contenido con direccionamiento por ubicación; este último permitía iterar por posiciones consecutivas, almacenando secuencias en **bloques contiguos**. Tres defectos graves:

1. **No previene el solapamiento de bloques.** No hay mecanismo que garantice que los bloques asignados no se solapen e interfieran —un problema básico de gestión de memoria.
2. **No puede liberar memoria.** No hay forma de liberar posiciones ya escritas, así que no puede reutilizar memoria en secuencias largas.
3. **El orden temporal se pierde con los saltos.** El orden secuencial se preserva sólo mientras la red itere por posiciones consecutivas; en cuanto la cabeza de escritura salta (vía contenido), el orden de las escrituras de antes y después del salto **no puede recuperarse**.

Estos tres defectos motivan exactamente las tres contribuciones del DNC.

## El Differentiable Neural Computer

El DNC es una red neuronal (el **controlador**) acoplada a una matriz de memoria externa $M \in \mathbb{R}^{N \times W}$ de $N$ posiciones de ancho $W$. Si la memoria es la RAM, el controlador es un "CPU diferenciable" cuyas operaciones se aprenden por descenso de gradiente. Su comportamiento es **independiente del tamaño de la memoria** mientras ésta no se llene.

En vez de direcciones únicas, el DNC usa **atención diferenciable** que define *distribuciones* sobre las $N$ filas de $M$. Esas distribuciones son las **weightings** (ponderaciones). La lectura es una suma ponderada $r = \sum_i M[i,\cdot]\, w^r[i]$; la escritura primero **borra** con un vector $e$ y luego **añade** con un vector $v$. Las unidades que calculan y aplican las ponderaciones son las **cabezas de lectura y escritura**.

El controlador (en el paper, un LSTM profundo) recibe en cada paso $t$ la entrada $x_t$ y los vectores de lectura previos, y emite una salida $y_t$ y un **vector de interfaz** $\xi_t$ que parametriza toda su interacción con la memoria: claves y fuerzas de lectura, parámetros de escritura, *free gates*, *allocation gate*, *write gate* y *read modes*. Las cabezas usan **tres formas de atención**, que son las tres contribuciones del DNC.

### Atención por contenido (heredada, refinada)

El *content lookup* compara una **clave** $k$ del controlador con el contenido de cada posición por **similitud coseno**, modulada por una fuerza $\beta \in [1,\infty)$:

$$C(M, k, \beta)[i] = \frac{\exp\{D(k, M[i,\cdot])\,\beta\}}{\sum_j \exp\{D(k, M[j,\cdot])\,\beta\}}$$

Sirve para **recuerdo asociativo**. Un punto fino: una clave que coincide **parcialmente** puede igual atender fuerte a esa posición (*pattern completion*), y como el contenido puede codificar referencias a otras direcciones, el *key-value retrieval* permite **navegar estructuras de datos asociativas**.

### Asignación dinámica de memoria (nuevo — resuelve defectos 1 y 2)

Análogo diferenciable de la *free list* clásica. Se mantiene un **vector de uso** $u_t \in [0,1]^N$: cada escritura **incrementa** el uso de una posición; cada lectura puede **decrementarlo** vía las *free gates* $f^i_t$ (una por cabeza), que deciden si las posiciones recién leídas se liberan:

$$\psi_t = \prod_{i=1}^{R}\left(1 - f^i_t\, w^{r,i}_{t-1}\right), \qquad u_t = (u_{t-1} + w^w_{t-1} - u_{t-1}\odot w^w_{t-1})\odot \psi_t$$

A partir del uso se ordena la *free list* y se construye la **ponderación de asignación** $a_t$, que entrega posiciones no usadas a la cabeza de escritura. Tres propiedades clave: es **independiente del tamaño y del contenido** de la memoria —un DNC se puede entrenar con poca memoria y escalarse sin reentrenar—; provee posiciones libres **una a una**, sin bloques contiguos (resuelve la interferencia); y permite **reasignar** memoria que ya no se necesita (resuelve la incapacidad de liberar del NTM). Los autores notan un paralelo con la potenciación de largo plazo del hipocampo y el rol del giro dentado en la dispersión representacional.

### Enlaces temporales (nuevo — resuelve defecto 3)

Una matriz $L \in [0,1]^{N\times N}$ registra las **transiciones entre posiciones escritas consecutivamente**: $L[i,j]\to 1$ si $i$ fue escrita **inmediatamente después** de $j$. Para cualquier ponderación $w$, $Lw$ **desplaza el foco hacia adelante** y $L^\top w$ **hacia atrás**, dándole al DNC la capacidad de **recuperar secuencias en el orden en que las escribió, aunque las escrituras no hayan ocurrido en pasos adyacentes** —justo lo que el NTM no podía. Se mantiene con un vector de precedencia $p_t$:

$$L_t[i,j] = (1 - w^w_t[i] - w^w_t[j])\, L_{t-1}[i,j] + w^w_t[i]\, p_{t-1}[j]$$

La matriz es $N\times N$ ($O(N^2)$), pero como en la práctica es muy dispersa se aproxima a $O(N\log N)$ en cómputo conservando sólo los $K$ valores más altos por fila/columna; con $K=8$ basta sin importar el tamaño de la memoria.

### Modos de lectura mejorados (nuevo)

Cada cabeza de lectura recibe un **read mode** $\pi^i_t \in S_3$ que **interpola** entre la ponderación backward, la de contenido y la forward:

$$w^{r,i}_t = \pi^i_t[1]\, b^i_t + \pi^i_t[2]\, c^{r,i}_t + \pi^i_t[3]\, f^i_t$$

Según qué componente domine, la cabeza lee por **contenido**, **itera hacia adelante** en orden de escritura, o **itera hacia atrás**. La cabeza de escritura interpola entre contenido y asignación vía la *allocation gate* y la *write gate* (si ésta es 0, no se escribe nada).

## Experimentos

**bAbI (question answering).** Los 20 tipos de preguntas sintéticas de Weston et al. emulan razonamiento textual. Un **único DNC** entrenado conjuntamente en los 20 tipos logró **3,8% de error medio** (fallo en sólo 2 tipos), frente al 7,5% y 6 tareas falladas del mejor resultado previo conjunto (MemN2N). La entrada fueron **tokens de palabra individuales** (sin embeddings de oración ni preprocesamiento), lo que estresa más la memoria de largo alcance.

**Razonamiento sobre grafos.** Cada vector de entrada codifica una **tripla** (nodo origen, etiqueta de arista, nodo destino). Tres consultas con aprendizaje por currículo: *traversal* (seguir un camino), *shortest path* (devolver el camino mínimo, tratado como predicción estructurada estilo DAgger) e *inference* (predecir el nodo final de una relación abreviada nunca presentada). Lo emblemático: tras entrenar **sólo con grafos aleatorios**, se probó **sin reentrenar** en dos grafos reales. En el **metro de Londres** (Zona 1): recorridos de 7 pasos con **98,8% de exactitud**. En un **árbol genealógico** inventado: relaciones de 4 pasos (p. ej. "tío abuelo materno") con **81,8%**. La visualización muestra que el DNC escribe cada tripla en una posición separada y luego usa los enlaces temporales para recuperar instrucciones en orden y la atención por contenido para encontrar estaciones. El contraste es brutal: el mejor LSTM no completó ni el primer nivel del currículo (37% tras casi 2 millones de ejemplos).

**Mini-SHRDLU (refuerzo).** Block puzzle resuelto con policy gradient (política y valor, ambas DNC). Hallazgo notable: las primeras 5 acciones podían **decodificarse de la memoria justo después de escribir la meta**, muchos pasos antes de ejecutarla (89% vs. 17% del baseline). El DNC **escribió su plan en memoria antes de actuar** —aprendió a planificar.

## Limitaciones reconocidas

- **Escala pequeña.** Tareas sintéticas con memorias de hasta 512 posiciones; el mundo real exigiría miles o millones.
- **Costo cuadrático de la matriz de enlaces.** $O(N^2)$ exacto; la aproximación dispersa lo mitiga pero introduce el hiperparámetro $K$.
- **Discontinuidades del gradiente.** El ordenamiento de la *free list* induce discontinuidades que se ignoran.
- **Entrenamiento costoso.** Requiere currículo cuidadoso, entrenamiento distribuido y BPTT con *gradient clipping*.
- **Sintético vs. natural.** Aun bAbI con tokens de palabra sigue siendo un dataset generado programáticamente.

## Impacto y lugar en la historia

El DNC es **la cumbre de la línea de memoria externa diferenciable**. Demostró, de forma convincente y publicada en *Nature*, que una red neuronal puede aprender **razonamiento algorítmico y sobre grafos** —encontrar caminos, inferir relaciones, planificar— manipulando una memoria externa que opera como variables direccionables, todo desde los datos sin programación explícita. La transferencia *zero-shot* de grafos aleatorios al metro de Londres es la demostración emblemática de que aprendió un **procedimiento general** y no memorizó instancias.

Históricamente, el DNC cierra el arco **NTM (2014) → DNC (2016)** de "computadores neuronales diferenciables". Esta línea resultó costosa y difícil de entrenar a escala, y la comunidad pivotó hacia los **Transformers** (2017), cuya atención sobre toda la secuencia ofreció una forma de "memoria" más simple, paralelizable y escalable. Pero las ideas del DNC —memoria direccionable por contenido, *key-value retrieval*, atención como mecanismo de acceso— resuenan directamente en los Transformers y en los sistemas modernos de **retrieval aumentado** (RAG, memorias de agentes).

## Conexión con la Clase 30

La [Clase 30](/clases/clase-30) se organiza en torno a la línea **Memory Networks** (Weston et al.) y sus versiones end-to-end. El DNC representa la **otra gran línea** del mismo problema —la línea "computador diferenciable" NTM→DNC—, y por eso lo incorporamos aunque no esté citado: **completa el mapa**. Ambas comparten la idea raíz (separar cómputo de una memoria externa accedida por atención diferenciable), pero el DNC añade **escritura iterativa, asignación/liberación dinámica y orden temporal**, acercándose a un modelo de cómputo general. El linaje:

- **NTM (2014)** → primer controlador + memoria de lectura-escritura diferenciable. Ver [`/papers/ntm-graves-2014`](/papers/ntm-graves-2014).
- **DNC (2016, este paper)** → NTM mejorado con asignación dinámica, enlaces temporales y modos de lectura; razonamiento sobre grafos en *Nature*.
- **MANN (Santoro et al. 2016)** → adapta la memoria externa al **meta-aprendizaje** few-shot. Ver [`/papers/mann-santoro-2016`](/papers/mann-santoro-2016).

Para los fundamentos transversales del mecanismo, ver [memory-augmented networks](/fundamentos/memory-augmented-networks) y [redes de memoria](/fundamentos/redes-de-memoria). El hub de la clase está en [`/clases/clase-30`](/clases/clase-30).
