# The Graph Neural Network Model — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *The Graph Neural Network Model*.
- **Autores:** Franco Scarselli y Marco Gori (Fellow, IEEE) — University of Siena, Italia; Ah Chung Tsoi — Hong Kong Baptist University; Markus Hagenbuchner — University of Wollongong, Australia; Gabriele Monfardini — University of Siena.
- **Venue:** *IEEE Transactions on Neural Networks*, vol. 20, no. 1, pp. 61–80, enero 2009. DOI: 10.1109/TNN.2008.2005605.
- **Historia editorial:** recibido en mayo de 2007, revisado en enero y mayo de 2008, aceptado en junio de 2008, primera publicación en diciembre de 2008. El trabajo nació de una estancia de Scarselli en Wollongong financiada por el Australian Research Council.

Este es **el paper fundacional del campo**: es el artículo que **acuña el término "graph neural network" (GNN)**. La frase exacta del texto es elocuente: *"We will call this novel neural network model a graph neural network (GNN)"*. Aunque las slides de la Clase 27 no lo citan explícitamente —presentan las GNN modernas (paso de mensajes, GCN, GAT, etc.) ya como un campo maduro—, **todo lo que la clase enseña desciende, directa o indirectamente, de este artículo**. Es un paper canónico cuya lectura da la perspectiva histórica de *de dónde viene la idea de que una red neuronal puede operar sobre un grafo arbitrario*.

La tesis central es deceptivamente simple y, vista desde hoy, profética. Muchos dominios —visión por computador, química molecular, biología molecular, reconocimiento de patrones, minería de datos, ingeniería de software, procesamiento de lenguaje natural— representan sus datos naturalmente como grafos. Hasta 2009, el machine learning lidiaba con grafos mediante un **preprocesamiento** que "aplastaba" (*squashes*, palabra del paper) el grafo a un vector de reales, perdiendo en el proceso la dependencia topológica de la información en cada nodo, de forma impredecible y dependiente del algoritmo de preprocesamiento. Scarselli et al. proponen en cambio **procesar el grafo directamente**: aprender una función `τ(G, n)` que mapea un grafo `G` y uno de sus nodos `n` a un espacio euclidiano `m`-dimensional, sin aplanar nada. El modelo *unifica* dos líneas previas —las recursive neural networks para datos estructurados y los modelos de random walk tipo PageRank— en un solo marco, y los extiende a la clase de grafos prácticamente más útil: **acíclicos, cíclicos, dirigidos y no dirigidos**.

El mecanismo es un **proceso de difusión de información**: cada nodo del grafo tiene un *estado*, los nodos actualizan su estado intercambiando información con sus vecinos, y esa actualización se itera hasta alcanzar un **equilibrio estable** (un punto fijo). La pieza teórica que hace que el modelo sea bien definido —que el equilibrio exista y sea único— es exigir que la función de transición sea una **contracción**, invocando el teorema de punto fijo de Banach.

## 2. Contexto histórico: de las recursive neural networks a los grafos generales

Para apreciar la ambición del paper hay que entender el estado del arte que lo precede. A fines de los 1990 y principios de los 2000, la línea dominante para "aprender de datos estructurados" eran las **recursive neural networks** (Sperduti & Starita 1997; Frasconi, Gori & Sperduti 1998, el clásico "general framework for adaptive processing of data structures"). Esas redes procesan estructuras de datos —árboles, secuencias— recorriéndolas según un orden inducido por la estructura: el estado de un nodo se computa a partir de los estados de sus hijos, en un barrido de las hojas hacia la raíz. Eso funciona porque el cómputo es **acíclico**: hay un orden parcial bien definido, y por tanto cada estado se calcula una sola vez. El paper resume estas restricciones con precisión: el dominio de entrada se limita a *directed acyclic graphs (DAG)*, las entradas de la función se limitan al nodo y a sus *hijos*, y existe un *supersource* (un nodo raíz desde el cual todos los demás son alcanzables) que se usa típicamente para producir la salida en tareas graph-focused.

Las recursive neural networks ya se habían aplicado con éxito a clasificación de términos lógicos, clasificación de compuestos químicos, reconocimiento de logos, scoring de páginas web y localización de caras. También se relacionaban con los **kernels para grafos** de las support vector machines (diffusion kernels, marginalized kernels, convolution kernels): ambos enfoques codifican el grafo en una representación interna, pero en las recursive networks la codificación se *aprende*, mientras que en SVM la diseña el usuario.

La segunda línea es la de las **cadenas de Markov y random walks**, que modelan procesos donde las conexiones causales entre eventos se representan como un grafo. Su aplicación más célebre es el ranking de páginas web: **PageRank** (Brin & Page 1998) y el algoritmo de autoridad de Kleinberg (1999). Estos métodos tienen la virtud de capturar la conectividad global del grafo mediante relajación a un estado estacionario, pero —salvo extensiones— *no aprenden*: la matriz de transición está dada, no estimada de ejemplos.

La ambición de Scarselli et al. es procesar grafos **generales** directamente, sin preprocesamiento y sin las tres restricciones de las recursive networks. El problema técnico que aparece de inmediato es que, en cuanto se admiten **ciclos**, el estado de un nodo puede depender de sí mismo (a través de un camino cerrado), y ya no existe un orden de cómputo que evalúe cada estado una sola vez. La solución del paper —tomada prestada de las cellular neural networks de Chua & Yang y de las Hopfield networks— es definir el estado **implícitamente, como solución de un sistema de ecuaciones de punto fijo**, y computarlo por relajación iterativa. La diferencia con esos modelos previos es que las GNN admiten clases de grafos más generales (incluyendo enlaces no dirigidos) y un mecanismo de difusión más general.

## 3. Contribución central: estado por nodo, transición local, punto fijo de Banach

La idea intuitiva: en un grafo, los nodos representan **objetos o conceptos** y las aristas sus **relaciones**. Cada concepto se define por sus propias características y por los conceptos relacionados. Por tanto, a cada nodo `n` se le adjunta un **estado** `x_n ∈ ℝ^s` que contiene una representación del concepto y que se construye con la información de la *vecindad* de `n`. De ese estado se produce una **salida** `o_n` (la decisión sobre el concepto).

Formalmente se definen dos funciones paramétricas:

- la **función de transición local** `f_w`, que expresa la dependencia de un nodo respecto de su vecindad;
- la **función de salida local** `g_w`, que describe cómo se produce la salida.

En la forma del paper (ecuación 1):

```
x_n = f_w( l_n, l_co[n], x_ne[n], l_ne[n] )
o_n = g_w( x_n, l_n )
```

donde `l_n` es la etiqueta (*label*) del nodo `n`, `l_co[n]` las etiquetas de sus aristas, `x_ne[n]` los estados de sus vecinos y `l_ne[n]` las etiquetas de los nodos vecinos. Las etiquetas son vectores de reales que codifican atributos: en una imagen segmentada, el label de un nodo puede ser área, perímetro y color medio de una región, y el de una arista la distancia y el ángulo entre regiones. **Los mismos parámetros `w` se comparten en todos los nodos** (weight sharing), lo que es exactamente el principio de las GNN modernas. El paper observa (Remark 3) que en general podrían usarse funciones distintas por tipo de nodo, pero por simplicidad analiza el modelo compartido.

Apilando todos los estados, salidas y etiquetas en vectores globales, las ecuaciones se reescriben de forma compacta como un sistema (ecuación 2):

```
x = F_w(x, l)        o = G_w(x, l_N)
```

donde `F_w` es la **función de transición global** (la versión apilada de las `f_w`) y `G_w` la de salida global. La pregunta clave es: **¿este sistema tiene solución, y es única?** Aquí entra la contribución teórica del paper.

**El teorema de punto fijo de Banach** da una condición suficiente: si `F_w` es una **contracción** respecto del estado —es decir, existe `μ`, `0 ≤ μ < 1`, tal que `‖F_w(x,l) − F_w(y,l)‖ ≤ μ‖x − y‖` para todo `x, y`— entonces (2) tiene una **solución única**, y además el esquema iterativo

```
x(t+1) = F_w(x(t), l)
```

(que es el método iterativo de Jacobi para resolver ecuaciones no lineales) **converge exponencialmente rápido a esa solución desde cualquier estado inicial** `x(0)`. Esta es la pieza conceptual que hace que el modelo esté bien definido a pesar de los ciclos: el estado deja de depender del orden de cómputo y queda definido como el atractor único del proceso de difusión.

El cómputo de (5) se interpreta como una **red de codificación** (*encoding network*): cada nodo del grafo se reemplaza por una unidad que computa `f_w`, las unidades se conectan según la conectividad del grafo, y otra unidad computa `g_w` para la salida. Cuando `f_w` y `g_w` son redes feedforward (FNN), la encoding network resulta ser una **red neuronal recurrente**, cuyas conexiones se dividen en internas (la arquitectura de la FNN) y externas (las aristas del grafo procesado). El paper también muestra el *unfolding* de esa red recurrente en el tiempo, donde cada capa corresponde a un instante y contiene una copia de todas las unidades (Fig. 3) — exactamente la estructura sobre la que opera backpropagation.

## 4. El método en detalle: aprendizaje vía Almeida–Pineda

El aprendizaje estima `w` minimizando una función de costo cuadrática (ecuación 6) sobre los nodos supervisados. En tareas *graph-focused* solo un nodo especial porta el target; en tareas *node-focused* la supervisión puede darse en cualquier nodo. El paper menciona (Remark 4) que el costo puede incluir un término de penalización —por ejemplo, de suavizado para mejorar la generalización.

El algoritmo es **descenso de gradiente** con tres pasos por iteración:

1. **Forward.** Los estados se actualizan iterando (5) hasta aproximar el punto fijo `x(T) ≈ x`. La hipótesis de contracción garantiza la convergencia; en la práctica la iteración se detiene cuando `‖x(t) − x(t−1)‖` cae bajo un umbral.
2. **Backward.** Se computa el gradiente `∂e_w/∂w`.
3. **Update.** Se actualizan los pesos según ese gradiente.

El punto técnicamente delicado es el paso 2. La encoding network es recurrente, así que la opción "obvia" sería **backpropagation through time (BPTT)**: desplegar la red en el tiempo y retropropagar por el grafo desplegado. Pero BPTT exige **almacenar el estado de cada instancia de cada unidad en cada paso temporal**, lo que para grafos grandes (por ejemplo, una porción significativa de la web) demanda una cantidad de memoria considerable.

La elegancia del paper es notar que, como (5) **ya alcanzó un punto estable** antes de calcular el gradiente, se cumple `x(t) = x(t−1) = x` para todo `t`. Eso permite usar el **algoritmo de Almeida–Pineda** (Almeida 1987; Pineda 1987) —"backpropagation through the fixed point"—, que retropropaga *a través del punto fijo* en vez de a través de toda la trayectoria temporal, almacenando solo el estado estable `x`. El paper lo formaliza con dos teoremas:

- **Teorema 1 (Diferenciabilidad).** Si `F_w` y `G_w` son continuamente diferenciables respecto de `x` y `w`, entonces `φ_w` (la función global aprendida) es continuamente diferenciable respecto de `w`. La demostración aplica el teorema de la función implícita a `T(x,l,w) = x − F_w(x,l)`: como `F_w` es contracción, el jacobiano `I − ∂F/∂x` es invertible (su determinante no es nulo). El paper subraya un punto profundo: **esta diferenciabilidad no vale para sistemas dinámicos generales** —donde un pequeño cambio en los parámetros puede saltar de un punto fijo a otro— y se debe *precisamente* a la suposición de contracción.

- **Teorema 2 (Backpropagation).** Define una sucesión `z(t)` que converge exponencialmente a un vector `z`, independiente del estado inicial, y muestra que el gradiente `∂e_w/∂w` se descompone en dos términos (ecuación 8): la contribución de la función de **salida** `g_w` (que backpropagation calcula al propagar por la capa de `g`) más la contribución de la función de **transición** `f_w` (que se obtiene iterando hasta el punto fijo del sistema adjunto). El algoritmo se detalla en el pseudocódigo de la Tabla I con tres procedimientos: `FORWARD` (computa estados), `BACKWARD` (computa el gradiente) y `MAIN` (itera ambos minimizando el error). En el simulador de los autores, los pesos se actualizan con **resilient backpropagation (RPROP)** de Riedmiller & Braun.

El paper advierte (con honestidad) que la encoding network solo es *aparentemente* similar a una red estática: el número de capas se determina dinámicamente y los pesos se comparten según la topología del grafo, de modo que técnicas de segundo orden, *pruning* y *growing* diseñadas para redes estáticas no se aplican directamente.

### Garantizar la contracción: dos implementaciones

`g_w` no necesita cumplir ninguna restricción (es una FNN multicapa). El peso recae en `f_w`, cuya implementación debe garantizar que `F_w` sea contracción. El paper propone dos modelos sobre la *forma no posicional* (ecuación 3, donde `f_w` se suma sobre los vecinos y es insensible al número y posición de los hijos):

1. **GNN lineal (no posicional).** Se implementa `h_w` como `x_n = Σ_{u∈ne[n]} A_{n,u} x_u + b_n`, donde la matriz `A_{n,u}` (de dimensión `s×s`) y el vector `b_n` son producidos por dos FNN llamadas *transition network* y *forcing network* (ecuaciones 12–14). La matriz `A` se escala por un factor `μ/(s·|ne[n]|)` y se fuerza que su norma-1 sea acotada, lo que **garantiza por construcción que `F_w` es contracción para cualquier `w`**. Esta forma captura el random walk como caso particular cuando `f_w` es lineal.

2. **GNN no lineal (no posicional).** `f_w` es una FNN multicapa (universal approximator), más expresiva, pero **no todos los parámetros son admisibles**: hay que asegurar la contracción de otra manera. La solución es añadir al costo un **término de penalización** `β·L(‖∂F_w/∂x‖)` que penaliza cuando la norma del jacobiano de la transición excede un umbral `μ` (la constante de contracción deseada), y es 0 en otro caso. En los experimentos usan `Σ_j L(‖A^j‖)` (con `A^j` la `j`-ésima columna del jacobiano) como aproximación diferenciable de `‖∂F_w/∂x‖`. Es decir: **se aprende libremente la transición, pero se la empuja a comportarse como contracción mediante regularización del jacobiano.**

El análisis de complejidad (Sección III, Tabla II) muestra que el costo por época es **lineal** respecto del tamaño del grafo (número de aristas) y de las FNN, salvo términos cuadráticos en la dimensión `s` del estado en los modelos posicional y no lineal. Empíricamente bastan **5–15 iteraciones** para aproximar el punto fijo, el número de iteraciones backward es pequeño, y el término costoso del jacobiano rara vez se activa gracias a la penalización (`r̄` típicamente 0, ocasionalmente 1–5).

## 5. Experimentos

El modelo se implementó en Matlab (el *GNN Toolbox*, descargable) y se evaluó en un Power Mac G5 a 2 GHz. Se probaron las formas lineal y no lineal (ambas no posicionales, con FNN de tres capas y activaciones sigmoidales), promediando los resultados sobre cinco corridas. Tres problemas:

- **Subgraph matching.** Encontrar los nodos de un subgrafo `S` dentro de un grafo mayor `G` (target = 1 si el nodo pertenece a una copia isomorfa de `S`, −1 si no). Dataset de 600 grafos aleatorios conexos divididos en train/validación/test, con un subgrafo `S` insertado en cada uno, labels enteros en un rango y ruido gaussiano (σ = 0.25) añadido. La comparación clave: una **FNN que solo usa el label del nodo** (sin topología) frente a las GNN. Resultado (Tabla III): las **GNN siempre superan a la FNN**, confirmando que explotan label y topología simultáneamente. El error medio de la FNN (22.8) es ~50% mayor que el de las GNN (12.7 no lineal, 13.5 lineal). El modelo no lineal supera levemente al lineal. La ventaja topológica decrece cuando crece `|S|` (hay que difundir más información). Se usó además para validar empíricamente la complejidad: el tiempo crece linealmente con nodos/aristas/neuronas y cuadráticamente con `s` (Fig. 5).

- **Mutagenesis.** Dataset de relational learning / inductive logic programming con 230 compuestos nitroaromáticos; predicción binaria de mutagenicidad. Cada molécula es un grafo (nodos = átomos, aristas = enlaces, ~26 nodos de media) con **un único nodo supervisado**. Se usaron features atom-bond, químicas y estructurales precodificadas (sin los functional groups, porque la GNN recupera por sí sola las subestructuras relevantes). Evaluación con validación cruzada de diez pliegues. Resultado destacado: las **GNN logran la mejor precisión reportada en la literatura de la época sobre la parte "regression-unfriendly" (Tabla V) y sobre el dataset completo (Tabla VI)**, y resultados cercanos al estado del arte en la parte "regression-friendly" (Tabla IV). Curiosamente, donde otros métodos rinden peor (la parte unfriendly), las GNN rinden mejor.

- **Web page ranking.** Aprender una versión de PageRank *modificada por el contenido*: la GNN ajusta la medida de autoridad según el tópico de la página. Grafo web sintético de 5000 nodos, con solo 50 nodos supervisados en train y 50 en validación. A cada página se le asocia un label booleano bidimensional de pertenencia a dos tópicos, y el target combina PageRank con ese contenido. Se usó **solo el modelo lineal**, naturalmente adecuado para aproximar la dinámica lineal de PageRank. La salida de la GNN sigue muy de cerca la función objetivo (Fig. 7), y notablemente **no hay sobreajuste pese a entrenar con 50 páginas de un grafo de 5000** (Fig. 8): error de train y validación se mantienen cercanos y aún descendiendo tras 2400 épocas.

## 6. Limitaciones

El propio diseño que hace al modelo elegante es también su principal limitación, y la historia posterior lo confirmaría:

- **El requisito de contracción limita la expresividad.** Forzar que `F_w` sea contracción (`μ < 1`) significa que la influencia entre nodos *decae* a medida que se propaga. Información que debería viajar lejos en el grafo se atenúa exponencialmente. El paper mismo lo deja entrever en subgraph matching: la ventaja topológica decae cuando hay que difundir más información. En términos modernos, la contracción impide una propagación arbitrariamente lejana de la señal.
- **Costo de iterar hasta el punto fijo.** Cada paso forward repite la difusión hasta converger (5–15 iteraciones típicas, pero dependientes del problema), y el aprendizaje repite forward + backward por muchas épocas. Es más caro que un número *fijo* de pasos.
- **Dominio estático.** El paper asume explícitamente que el grafo no cambia en el tiempo; en las conclusiones señala como trabajo futuro extender las GNN a dominios dinámicos (evolución de la web, redes sociales) y a casos donde las relaciones deben inferirse.
- **Aprendizaje atado al gradiente.** Diseñar algoritmos de aprendizaje no basados en gradiente "no es obvio" y queda como investigación futura; las técnicas estándar para redes estáticas no transfieren.

## 7. Impacto: el origen del campo

Quince años después, es difícil exagerar la importancia de este artículo. Acuñó el nombre del campo y estableció el patrón fundamental que todas las GNN modernas heredan: **un estado/embedding por nodo, una función de transición local con parámetros compartidos, y la actualización iterativa por agregación de la vecindad**. Las contribuciones que perduran son conceptuales: la idea de difusión de información sobre el grafo, el weight sharing entre nodos, la unificación de recursive networks y random walks, y la noción de que el grafo se procesa *como tal* sin aplanarlo.

Lo que la historia revisó fue la *mecánica* del estado. El sucesor directo, las **Gated Graph Neural Networks (Li et al., 2015)**, reemplazaron explícitamente el punto fijo contractivo de Scarselli por un **número fijo de pasos de propagación** con celdas **GRU** como actualización recurrente: ya no se itera hasta converger ni se exige contracción, lo que libera expresividad y simplifica el entrenamiento (BPTT estándar sobre pasos acotados). Poco después, las **Graph Convolutional Networks (Kipf & Welling, 2017)** y el marco general de **Message Passing Neural Networks (Gilmer et al., 2017)** completaron la transición a la formulación moderna: capas apiladas, cada una un paso de paso de mensajes, sin punto fijo. El paper de Scarselli es el ancestro común de toda esa descendencia.

## 8. Conexión con la Clase 27

Para la Clase 27 (Redes Neuronales de Grafos), este paper es **el ancestro de absolutamente todo lo que la clase presenta**. Conviene hacer explícitas las correspondencias:

- El mecanismo de "**repetir N veces** la propagación de mensajes" que la clase enseña (paso de mensajes: cada nodo agrega información de sus vecinos, y se repite un número fijo de capas/pasos) **es la versión moderna y simplificada de la idea original de Scarselli**. Donde la clase fija `N` pasos, Scarselli iteraba `F_w` *hasta el punto fijo* —idealmente infinitos pasos, en la práctica hasta converger—. La diferencia es precisamente esa: **pasos fijos (clase) vs. iteración hasta punto fijo (Scarselli 2009)**.

- La función de transición local `f_w` con **parámetros compartidos entre todos los nodos** es, conceptualmente, la *función de agregación/actualización* de mensajes de la clase. El estado `x_n` es el *embedding del nodo*. La función de salida `g_w` es la *cabeza de lectura* (readout) sobre los embeddings.

- La restricción de **contracción** —que en Scarselli era central— **desaparece en las formulaciones modernas**. Las GNN de la clase no la necesitan porque no buscan un punto fijo: aplican un número finito de capas. Esto explica por qué las GNN modernas son más expresivas y por qué pudieron escalar: se quitaron la camisa de fuerza del teorema de Banach.

- La transición concreta la marca **GGNN (Li et al., 2015)**, que la clase puede mencionar como el puente: reemplaza la iteración a punto fijo por **pasos fijos + GRU**, y de ahí en adelante el campo es el que la clase presenta (GCN, GraphSAGE, GAT, MPNN).

En síntesis: leer este paper es entender *por qué* las GNN modernas hacen lo que hacen. Cuando la clase dice "propagamos mensajes N veces", está ejecutando —en forma truncada, sin punto fijo y sin exigencia de contracción— exactamente el proceso de difusión que Scarselli, Gori, Tsoi, Hagenbuchner y Monfardini formalizaron en 2009 y al que le dieron nombre.
