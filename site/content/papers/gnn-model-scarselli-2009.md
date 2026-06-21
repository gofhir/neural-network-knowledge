---
title: "The Graph Neural Network Model (2009)"
weight: 310
math: true
---

{{< paper-card
    title="The Graph Neural Network Model"
    authors="Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, Gabriele Monfardini"
    year="2009"
    venue="IEEE Transactions on Neural Networks"
    pdf="/papers/gnn-model-scarselli-2009.pdf" >}}
El **paper fundacional** que acuñó el término *graph neural network* (GNN). Su frase literal —*"We will call this novel neural network model a graph neural network"*— bautizó un campo entero. Propone procesar un grafo **directamente**, sin aplanarlo a un vector: a cada nodo se le asigna un **estado** que se actualiza intercambiando información con sus vecinos mediante una **función de transición local con parámetros compartidos**, iterada **hasta un punto fijo**. Para que ese punto fijo exista y sea único, exige que la transición sea una **contracción** (teorema de Banach), y entrena con **Almeida–Pineda** (backprop a través del punto fijo). Aunque las slides de la [Clase 27](/clases/clase-27) no lo citan, todo lo que enseñan desciende de aquí: es el origen histórico del paso de mensajes.
{{< /paper-card >}}

---

> **Nota sobre su rol.** Este artículo no aparece citado en las slides de la [Clase 27](/clases/clase-27), que presentan las [GNN](/fundamentos/redes-neuronales-de-grafos) modernas ([message passing](/fundamentos/message-passing), GCN, GAT) ya como campo maduro. Lo incluimos por ser **canónico**: es el documento que da nombre al campo y la perspectiva histórica de *de dónde viene* la idea de que una red neuronal pueda operar sobre un grafo arbitrario.

## Contexto: de las recursive networks a los grafos generales

Muchos dominios —visión, química molecular, biología, minería de datos, lenguaje natural— representan sus datos como [grafos](/dominios/estructurados). Hasta 2009 el machine learning los trataba con un **preprocesamiento** que "aplastaba" (*squashes*, palabra del paper) el grafo a un vector de reales, perdiendo de forma impredecible la dependencia topológica de cada nodo. Scarselli et al. proponen lo contrario: aprender una función $\tau(G, n)$ que mapea un grafo $G$ y uno de sus nodos $n$ a $\mathbb{R}^m$ **sin aplanar nada**.

El estado del arte previo eran dos líneas. Primero, las **recursive neural networks** (Sperduti & Starita 1997; Frasconi, Gori & Sperduti 1998): procesan estructuras recorriéndolas en un orden inducido por ellas —el estado de un nodo se computa desde los estados de sus hijos, de las hojas hacia la raíz—. Esto funciona porque el cómputo es **acíclico**, pero impone tres restricciones fuertes: el dominio se limita a *directed acyclic graphs* (DAG), las entradas se limitan al nodo y sus **hijos**, y se requiere un *supersource* (nodo raíz desde el que todos son alcanzables). Segundo, los modelos de **random walk** tipo **PageRank** (Brin & Page 1998) y el algoritmo de autoridad de Kleinberg (1999): capturan la conectividad global por relajación a un estado estacionario, pero —salvo extensiones— **no aprenden**, la matriz de transición está dada.

La ambición del paper es procesar grafos **generales** (acíclicos, cíclicos, dirigidos y no dirigidos) directamente y sin esas restricciones, **unificando** ambas líneas. El obstáculo técnico aparece de inmediato: al admitir **ciclos**, el estado de un nodo puede depender de sí mismo a través de un camino cerrado, y ya no existe un orden que evalúe cada estado una sola vez. La solución, tomada de las cellular neural networks y las redes de Hopfield, es definir el estado **implícitamente, como solución de un sistema de punto fijo**, y computarlo por relajación iterativa.

## El modelo: estado por nodo, transición local, punto fijo de Banach

Los nodos representan **objetos o conceptos** y las aristas sus **relaciones**. A cada nodo $n$ se le adjunta un **estado** $x_n \in \mathbb{R}^s$ construido con la información de su vecindad, del que se produce una **salida** $o_n$. Se definen dos funciones paramétricas: la **función de transición local** $f_w$ (dependencia del nodo respecto de sus vecinos) y la **función de salida local** $g_w$:

$$x_n = f_w\big(l_n,\; l_{\text{co}[n]},\; x_{\text{ne}[n]},\; l_{\text{ne}[n]}\big), \qquad o_n = g_w\big(x_n,\; l_n\big)$$

donde $l_n$ es la etiqueta del nodo, $l_{\text{co}[n]}$ las de sus aristas, $x_{\text{ne}[n]}$ los estados de sus vecinos y $l_{\text{ne}[n]}$ las etiquetas de los vecinos. Las etiquetas son vectores de atributos (en una imagen segmentada: área, perímetro, color medio de una región). El punto clave —y el principio de las GNN modernas— es que **los mismos parámetros $w$ se comparten en todos los nodos** (*weight sharing*).

Apilando todos los estados y etiquetas en vectores globales, el sistema se escribe de forma compacta:

$$x = F_w(x, l), \qquad o = G_w(x, l_N)$$

La pregunta crítica es: ¿este sistema tiene solución, y es única? Aquí está la contribución teórica. **El teorema de punto fijo de Banach** da una condición suficiente: si $F_w$ es una **contracción** respecto del estado —existe $\mu$, con $0 \le \mu < 1$, tal que $\lVert F_w(x,l) - F_w(y,l)\rVert \le \mu\lVert x-y\rVert$— entonces el sistema tiene **solución única** y el esquema iterativo

$$x(t+1) = F_w\big(x(t), l\big)$$

(el método de Jacobi para ecuaciones no lineales) **converge exponencialmente rápido** a esa solución desde cualquier $x(0)$. Esta es la pieza que hace al modelo bien definido pese a los ciclos: el estado deja de depender del orden de cómputo y queda definido como el **atractor único** del proceso de difusión.

El cómputo se interpreta como una **red de codificación** (*encoding network*): cada nodo se reemplaza por una unidad que computa $f_w$, conectadas según la topología del grafo, más una unidad $g_w$ para la salida. Cuando $f_w$ y $g_w$ son redes feedforward, la encoding network resulta ser una **red neuronal recurrente** sobre la que opera backpropagation tras desplegarla en el tiempo.

## Entrenamiento: Almeida–Pineda y la garantía de contracción

El aprendizaje estima $w$ minimizando un costo cuadrático sobre los nodos supervisados, por **descenso de gradiente** en tres pasos por iteración: **(1) Forward** —iterar el punto fijo hasta $x(T) \approx x$, deteniéndose cuando $\lVert x(t) - x(t-1)\rVert$ cae bajo un umbral—; **(2) Backward** —computar $\partial e_w / \partial w$—; **(3) Update**.

El paso delicado es el backward. La opción obvia, **backpropagation through time** (desplegar y retropropagar por toda la trayectoria), exige **almacenar el estado de cada unidad en cada paso temporal**, prohibitivo para grafos grandes. La elegancia del paper es notar que, como el forward **ya alcanzó el punto estable**, se cumple $x(t) = x(t-1) = x$ para todo $t$. Eso habilita el **algoritmo de Almeida–Pineda** (Almeida 1987; Pineda 1987) —*backpropagation through the fixed point*—, que retropropaga **a través del punto fijo** en vez de toda la trayectoria, almacenando solo el estado estable $x$. Dos teoremas lo formalizan: el de **diferenciabilidad** (vía teorema de la función implícita; nota profunda: esto **no vale para sistemas dinámicos generales**, donde un cambio en los parámetros puede saltar de un punto fijo a otro, y se debe *precisamente* a la contracción) y el de **backpropagation** (el gradiente se descompone en la contribución de la salida $g_w$ más la de la transición $f_w$, esta última iterando hasta el punto fijo del sistema adjunto). El simulador de los autores actualiza los pesos con **resilient backpropagation (RPROP)**.

Para garantizar la contracción, el peso recae en $f_w$ ($g_w$ no necesita restricción). El paper propone dos implementaciones:

- **GNN lineal:** $f_w$ se arma con dos FNN (*transition* y *forcing network*) que producen una matriz $A_{n,u}$ y un vector $b_n$; $A$ se escala de modo que su norma quede acotada, lo que **garantiza por construcción que $F_w$ es contracción para cualquier $w$**. Captura el random walk como caso particular.
- **GNN no lineal:** $f_w$ es una FNN multicapa (aproximador universal), más expresiva, pero no todos los parámetros son admisibles. Se añade al costo un **término de penalización del jacobiano** $\beta\, L(\lVert \partial F_w / \partial x\rVert)$ que castiga superar el umbral $\mu$. Es decir: **se aprende libremente la transición, pero se la empuja a comportarse como contracción mediante regularización.**

El costo por época es **lineal** en el tamaño del grafo. Empíricamente bastan **5–15 iteraciones** para aproximar el punto fijo.

## Experimentos

Implementado en Matlab (*GNN Toolbox*), evaluado en tres problemas con FNN de tres capas y activaciones sigmoidales, promediando cinco corridas:

- **Subgraph matching.** Localizar los nodos de un subgrafo $S$ dentro de un grafo mayor. Sobre 600 grafos aleatorios con ruido gaussiano, la comparación clave es una **FNN que solo usa el label del nodo** (sin topología) frente a las GNN. Las **GNN siempre ganan**: su error medio (12.7 no lineal, 13.5 lineal) es ~50% menor que el de la FNN (22.8), confirmando que explotan label y topología simultáneamente. La ventaja topológica decrece cuando crece $|S|$ (hay que difundir más información). Sirvió también para validar la complejidad lineal.
- **Mutagenesis.** 230 compuestos nitroaromáticos, predicción binaria de mutagenicidad; cada molécula es un grafo (átomos = nodos, enlaces = aristas) con **un único nodo supervisado**. Validación cruzada de diez pliegues. Las GNN logran **la mejor precisión reportada en la literatura de la época** sobre la parte "regression-unfriendly" y sobre el dataset completo —curiosamente, rinden mejor justo donde otros métodos rinden peor.
- **Web page ranking.** Aprender un PageRank **modificado por contenido** sobre un grafo web sintético de 5000 nodos con **solo 50 nodos supervisados**. Usando el modelo lineal, la salida sigue de cerca la función objetivo y, notablemente, **no hay sobreajuste** pese a entrenar con 50 de 5000 páginas.

## Limitaciones

El diseño elegante es también la limitación principal, y la historia lo confirmó:

- **La contracción limita la expresividad y el alcance.** Forzar $\mu < 1$ significa que la influencia entre nodos **decae exponencialmente** al propagarse: información que debería viajar lejos en el grafo se atenúa. En términos modernos, la contracción impide una propagación arbitrariamente lejana de la señal.
- **Costo del punto fijo.** Cada forward repite la difusión hasta converger (5–15 iteraciones), y el entrenamiento repite forward + backward por muchas épocas. Es más caro que un número *fijo* de pasos.
- **Dominio estático.** Se asume que el grafo no cambia en el tiempo.
- **Aprendizaje atado al gradiente.** Las técnicas para redes estáticas (pruning, growing, segundo orden) no transfieren directamente.

## Impacto: el origen del campo

Quince años después, cuesta exagerar su importancia. Acuñó el nombre del campo y estableció el patrón que toda [GNN](/fundamentos/redes-neuronales-de-grafos) moderna hereda: **un embedding por nodo, una función de transición local con parámetros compartidos, y la actualización iterativa por agregación de la vecindad**. Lo que perdura es conceptual: difusión de información sobre el grafo, weight sharing, unificación de recursive networks y random walks, y procesar el grafo *como tal* sin aplanarlo.

Lo que la historia revisó fue la **mecánica del estado**. El sucesor directo, las **[Gated Graph Neural Networks (Li et al., 2015)](/papers/ggnn-li-2015)**, reemplazaron el punto fijo contractivo por un **número fijo de pasos de propagación** con celdas **GRU** como actualización: ya no se itera hasta converger ni se exige contracción, lo que libera expresividad y simplifica el entrenamiento (BPTT estándar sobre pasos acotados). Poco después, **GCN (Kipf & Welling, 2017)** y el marco de **[Message Passing Neural Networks](/fundamentos/message-passing) (Gilmer et al., 2017)** completaron la transición a la formulación moderna: capas apiladas, cada una un paso de mensajes, sin punto fijo.

## Conexión con la Clase 27

Para la [Clase 27](/clases/clase-27) (Redes Neuronales de Grafos), este paper es el **ancestro de todo lo que la clase presenta**:

- El mecanismo de "**repetir $N$ veces** la propagación de mensajes" que enseña la clase es la versión moderna y simplificada de la idea de Scarselli. La diferencia exacta: **pasos fijos (clase) vs. iteración hasta punto fijo (Scarselli 2009)**.
- La función de transición $f_w$ con **parámetros compartidos** es la función de agregación/actualización del [message passing](/fundamentos/message-passing); el estado $x_n$ es el *embedding del nodo*; $g_w$ es la cabeza de lectura (*readout*).
- La restricción de **contracción** —central en Scarselli— **desaparece** en las formulaciones modernas, que no buscan un punto fijo. Eso explica por qué las GNN modernas son más expresivas y por qué pudieron escalar: se quitaron la camisa de fuerza de Banach.
- El puente lo marca **[GGNN (Li et al., 2015)](/papers/ggnn-li-2015)**: reemplaza la iteración a punto fijo por **pasos fijos + GRU**.

Leer este paper es entender *por qué* las GNN modernas hacen lo que hacen. Cuando la clase dice "propagamos mensajes $N$ veces", está ejecutando —truncado, sin punto fijo y sin contracción— el mismo proceso de difusión que Scarselli, Gori, Tsoi, Hagenbuchner y Monfardini formalizaron en 2009 y al que le dieron nombre.

## Notas y enlaces

- *IEEE Transactions on Neural Networks*, vol. 20, no. 1, pp. 61–80, enero 2009. DOI: 10.1109/TNN.2008.2005605.
- Afiliaciones: University of Siena (Italia), Hong Kong Baptist University, University of Wollongong (Australia).
- Sucesor directo en el sitio: [Gated Graph Neural Networks (Li et al., 2015)](/papers/ggnn-li-2015).
