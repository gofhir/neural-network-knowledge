---
title: "GGNN: Gated Graph Sequence Neural Networks (2015)"
weight: 301
math: true
---

{{< paper-card
    title="Gated Graph Sequence Neural Networks"
    authors="Yujia Li, Daniel Tarlow, Marc Brockschmidt, Richard Zemel"
    year="2016"
    venue="ICLR 2016"
    pdf="/papers/ggnn-li-2015.pdf"
    arxiv="1511.05493" >}}
Paper bisagra en la historia de las [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos). Toma el GNN clásico de Scarselli (2009) y lo moderniza con dos cambios: reemplaza la iteración hasta **punto fijo** por una recurrencia desenrollada un número **fijo** de pasos con una **GRU** como actualización, entrenada con backpropagation through time (BPTT); y lo extiende para producir **secuencias** de salidas (GGS-NN). Es el primer modelo que lleva el mecanismo de compuertas al message passing y un ancestro directo del [marco moderno](/fundamentos/message-passing). Su aplicación estrella —inferir invariantes de la memoria de un programa— iguala el estado del arte (89.96% vs 89.11%) **sin ingeniería de características manual**.
{{< /paper-card >}}

---

## Contexto: el GNN de Scarselli y la tiranía del punto fijo

Hacia 2015 las [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos) ya existían —el GNN clásico de Gori et al. (2005) y Scarselli et al. (2009) llevaba casi una década— pero, en palabras del propio paper, "no parecen estar en uso generalizado en la comunidad ICLR". Una de las metas explícitas de los autores es, literalmente, **publicitar los GNN** como una variante de red neuronal útil. Para entender qué reparan, hay que entender qué andaba mal.

El **GNN original** opera sobre un grafo $G=(V,E)$: cada nodo $v$ tiene un vector $h_v \in \mathbb{R}^D$ que se actualiza con una recurrencia **iterada hasta convergencia**. No corre un número predefinido de pasos: se repite hasta que las representaciones alcanzan un **punto fijo**. El aprendizaje usa el algoritmo de Almeida-Pineda: se propaga hasta converger y recién entonces se computan gradientes desde la solución convergida. Esto tiene una ventaja elegante —no hace falta almacenar estados intermedios— pero un precio severo, que es el corazón del argumento del paper.

Para que el punto fijo **exista y sea único**, el paso de propagación debe ser un **mapa de contracción**: una función con $\rho < 1$ tal que $\lVert T(h) - T(h') \rVert < \rho \lVert h - h' \rVert$. Cuando la actualización es una red neuronal, esto se fuerza penalizando la norma del Jacobiano. La consecuencia, demostrada formalmente en el apéndice del paper, es demoledora: la influencia de un nodo sobre otro a distancia $\delta$ **decae como $\rho^\delta$**, exponencialmente con la distancia en el grafo. El constreñimiento de contracción **impide modelar dependencias de largo alcance**. Peor aún: como el punto fijo es independiente de la inicialización, no tiene sentido inyectar información del problema (etiquetas de nodo) como estado inicial.

Ese es el contexto exacto: GNN diferenciables pero atados a un esquema de convergencia que (i) limita su expresividad por la contracción y (ii) los vuelve poco prácticos con las herramientas modernas de deep learning. El paper se sitúa junto a trabajos contemporáneos sobre grafos —*spectral networks* (Bruna 2013), *molecular fingerprints* (Duvenaud 2015)— pero a diferencia de todos, que producen una salida única, ninguno aborda la producción de **secuencias**.

## La contribución: de punto fijo a pasos fijos con compuertas

La modificación central, en palabras del paper, es **usar Gated Recurrent Units (Cho et al., 2014) y desenrollar la recurrencia un número fijo $T$ de pasos, con BPTT para los gradientes**. Tres consecuencias se desprenden de este único cambio.

**Primero, se elimina el constreñimiento de contracción.** Como ya no se busca un punto fijo —solo se corren $T$ pasos y se para—, la propagación no necesita ser una contracción. La información puede recorrer $T$ saltos sin el decaimiento exponencial forzado. El costo es de memoria: BPTT exige guardar los estados intermedios de los $T$ pasos, algo que Almeida-Pineda evitaba. Los autores aceptan explícitamente el *trade-off*.

**Segundo, ahora sí tiene sentido inicializar.** Como el resultado ya no es independiente de la inicialización, se introducen las **anotaciones de nodo** $x_v$, vectores que codifican etiquetas del problema. El ejemplo canónico es alcanzabilidad: para predecir si $t$ es alcanzable desde $s$ se da $x_s=[1,0]^\top$, $x_t=[0,1]^\top$ y $x_v=[0,0]^\top$ al resto. El estado inicial copia la anotación y rellena con ceros: $h_v^{(1)}=[x_v^\top,\,0]^\top$.

**Tercero, la actualización es una GRU.** El paper experimentó primero con una recurrencia estilo RNN vanilla, pero halló la GRU más efectiva. La analogía que trazan los propios autores: pasar del GNN clásico a GG-NN es como pasar de las *Recursive Neural Networks* a las *Tree LSTM*, o de *belief propagation* hasta convergencia a *truncated BP* entrenado para dar buen resultado tras un número fijo de iteraciones. En los tres casos el objetivo es mejorar la propagación de largo alcance con compuertas en lugar de iteración hasta el equilibrio.

### Mensajes, agregación y la GRU

El [message passing](/fundamentos/message-passing) de GG-NN se ejecuta para $t=1\dots T$. El paso de mensajes agrega información de los vecinos con parámetros que **dependen del tipo de arista y de la dirección**:

$$a_v^{(t)} = A_{v:}^\top\,\big[h_1^{(t-1)\top}\dots h_{|V|}^{(t-1)\top}\big]^\top + b.$$

La matriz $A$ es dispersa (su patrón de ceros son las aristas reales) y tiene *parameter tying*: los bloques se comparten según tipo y dirección. Se descompone en $A=[A^{(\text{out})}, A^{(\text{in})}]$, y si una arista de tipo $B$ tiene su reversa $B'$, ambas son parámetros **distintos** —el modelo distingue la dirección. En la notación de la Clase 27, la matriz de bloques por tipo de arista $k$ es lo que la diapositiva llama $E_k$: el mensaje por una arista de tipo $k$ es $m = h \cdot E_k$ (más sesgo), y $a_v$ es la **suma agregada** de todos los mensajes entrantes/salientes de $v$.

Una vez agregados los mensajes, el estado se actualiza con la mecánica completa de una GRU —compuerta de actualización $z$, compuerta de reset $r$, estado candidato $\tilde{h}$ y mezcla final—, que la clase condensa como $h_t=\mathrm{GRU}(h_{t-1}, h')$ con $h'=a_v^{(t)}$. Las compuertas deciden cuánto del pasado conservar y cuánto del mensaje nuevo incorporar a lo largo de los $T$ pasos: ese es el mecanismo que combate el decaimiento del GNN clásico. Para salidas a nivel de grafo se usa un *readout* con atención suave, $h_G=\tanh\big(\sum_v \sigma(i(h_v^{(T)},x_v))\odot\tanh(j(h_v^{(T)},x_v))\big)$, donde $\sigma(i(\cdot))$ decide qué nodos importan.

### Extensión a secuencias (GGS-NN)

La contribución que da nombre al paper: encadenar varias GG-NN para emitir $o^{(1)}\dots o^{(K)}$ —caminos, enumeraciones de nodos o fórmulas lógicas. En el paso $k$ se usan dos GG-NN: $\mathcal{F}_o^{(k)}$ predice la salida y $\mathcal{F}_\mathcal{X}^{(k)}$ predice las anotaciones del paso siguiente (el estado que se arrastra). Hay dos regímenes de entrenamiento: con **anotaciones observadas** (se especifican las $\mathcal{X}^{(k)}$ intermedias, descomponiendo la secuencia en predicciones de un paso) o con **anotaciones latentes** (solo se da $\mathcal{X}^{(1)}$ y los objetivos, retropropagando por toda la secuencia).

## Experimentos

**bAbI (Weston et al., 2015).** Suite de 20 tareas de razonamiento básico. Los autores transforman bAbI a grafos con la opción `--symbolic`: cada entidad es un nodo, cada relación una arista etiquetada, y los argumentos de la pregunta son anotaciones iniciales. En tareas de salida única (deducción, inducción, conteo) GG-NN alcanza **100% con solo 50 ejemplos** y **menos de 600 parámetros**, mientras los baselines RNN/LSTM (5k–30k parámetros) o necesitan muchos más ejemplos o fallan directamente (33–50% incluso con 950 ejemplos). En la tarea 19 *Path Finding* —la más difícil de bAbI— GGS-NN logra 99.0% con 250 ejemplos frente a 24.7%/28.2% de RNN/LSTM con 950.

**Algoritmos de grafo.** Dos tareas nuevas de salida secuencial: **Shortest Path** y **Eulerian Circuit**. GGS-NN logra **100% con 50 ejemplos** en ambas, mientras RNN/LSTM fracasan estrepitosamente (9.7%/10.5% en shortest path; 0.3%/0.1% en Eulerian). La razón: las secuencias llegan a ~80 tokens, exigen memoria de largo alcance, y **la salida no aparece en el mismo orden que la entrada** —de hecho permutar las aristas no cambia la salida. GGS-NN maneja bien estos datos "estáticos" porque la estructura de grafo es explícita; los RNN/LSTM no.

**Verificación de programas (la aplicación real).** Es la motivación del trabajo. Dado un programa C, un paso clave de la verificación automática es inferir **invariantes** que describan las estructuras de datos del *heap*. Los autores representan la memoria como grafo dirigido (nodos = direcciones, aristas etiquetadas = punteros como `next`, `left`, `right`) y entrenan un GGS-NN para predecir una fórmula en *separation logic* del tipo $\exists x_1\dots x_n.\,a_1*\dots*a_m$, con átomos $\mathrm{ls}(x,y)$ (lista), $\mathrm{tree}(x)$ o $\mathrm{none}(x)$. La generación de la fórmula se descompone en una secuencia de decisiones, cada una una GG-NN: clasificación a nivel de grafo (¿hace falta otro cuantificador?), selección de nodo (¿cuál cuantificar?) y actualización de anotaciones. Sobre ~160.000 combinaciones fórmula/grafo, el GGS-NN logra **89.96% de accuracy sin ingeniería de características**, superando al método previo de Brockschmidt et al. (2015) que lograba 89.11% **con** ingeniería manual extensa. Los invariantes hallados se usaron exitosamente en un *theorem prover* para probar la corrección de programas reales (Traverse, Concat, Copy, Insert, Remove).

## Limitaciones reconocidas

- **Pérdida de orden temporal y relaciones de orden superior.** La transformación de bAbI a grafo descarta el orden temporal y no maneja con facilidad relaciones ternarias o superiores.
- **Mayor costo de memoria.** BPTT sobre $T$ pasos exige almacenar los estados intermedios, a diferencia de Almeida-Pineda. Es el precio directo de eliminar la contracción.
- **Dependencia de entradas estructuradas/simbólicas.** Usar la forma simbólica de bAbI es cómodo; manejar lenguaje natural crudo queda como desafío abierto.
- **La pregunta llega después de todos los hechos.** El modelo consume todos los hechos antes de la pregunta, obligándolo a derivar y almacenar *todas* las consecuencias en las representaciones de nodo.
- **Datos estáticos vs. temporales.** GGS-NN brilla en datos sin naturaleza secuencial intrínseca, pero aplicarlos a datos genuinamente temporales que RNN/LSTM manejan bien queda como trabajo futuro.

## Impacto: el gating llega al message passing

La importancia histórica de GGNN supera a la de cualquiera de sus benchmarks. Fue **el primer modelo que introdujo compuertas (gating) en las redes neuronales de grafos**, demostrando que se podía abandonar la iteración hasta punto fijo del GNN clásico a favor de una recurrencia de profundidad fija entrenada con BPTT —exactamente el patrón que domina las GNN modernas. Probó que las GNN eran prácticas y competitivas con las herramientas estándar de deep learning.

Conceptualmente, GGNN es un **ancestro directo del [message passing](/fundamentos/message-passing) moderno**. Apenas dos años después, Gilmer et al. (2017) abstraen el patrón común a GGNN, [GCN](/papers/gcn-kipf-2017) y otros bajo el nombre *Message Passing Neural Networks*, con sus tres fases de mensaje, actualización y *readout* —y citan a GGNN como un caso particular donde la actualización es una GRU. La idea de anotaciones de nodo, de matrices de mensaje por tipo de arista ($E_k$) y de *readout* con atención suave se volvió vocabulario estándar. Y su aplicación a código inauguró una línea fértil que culmina en [Programs-as-Graphs](/papers/programs-as-graphs-allamanis-2018) (Allamanis et al., 2018), que extiende directamente las ideas de GGNN.

## Conexión con la Clase 27

La [Clase 27](/clases/clase-27) presenta GGNN como el **primer modelo concreto** tras exponer el mecanismo genérico de [message passing](/fundamentos/message-passing), y la elección no es casual: es la encarnación más limpia de ese mecanismo con compuertas.

- **El cómputo de mensajes $m = h \cdot E_k$.** La diapositiva muestra que el mensaje por una arista de tipo $k$ se obtiene multiplicando el estado del nodo por una matriz aprendible específica del tipo de arista $E_k$ —exactamente la matriz de bloques $A_{v:}$ del paper, con *parameter tying* entre aristas del mismo tipo.
- **La actualización $h_t = \mathrm{GRU}(h_{t-1}, h')$.** La clase resume las cuatro ecuaciones GRU del paper en esta única expresión: el estado previo y el mensaje agregado entran a una GRU que decide cuánto conservar y cuánto incorporar. Ese es el rasgo distintivo de GGNN: actualización **recurrente y con compuertas**.
- **El contraste con GCN y GraphSAGE.** Aquí está el valor pedagógico de presentar los tres juntos. [GCN](/papers/gcn-kipf-2017) (Kipf & Welling, 2017) normaliza por el grado de los nodos y aplica una transformación lineal por capa, sin compuertas ni tipos de arista; típicamente *shallow*. **GraphSAGE** (Hamilton 2017) introduce muestreo de vecinos y agregadores genéricos para escalar y para *inductive learning*. **GGNN** es recurrente (reusa los mismos pesos en $T$ pasos), usa compuertas GRU y **distingue tipos de arista** vía $E_k$ —ideal para grafos con aristas etiquetadas como el *heap* de un programa o un [grafo estructurado](/dominios/estructurados). El eje que la clase fija: GCN normaliza por grado, GraphSAGE muestrea y agrega, GGNN recurre y compuerta.

## Notas y enlaces

- Paper original que GGNN moderniza: [The Graph Neural Network Model (Scarselli 2009)](/papers/gnn-model-scarselli-2009).
- arXiv: [1511.05493](https://arxiv.org/abs/1511.05493) (v1 nov 2015, ICLR 2016).
- Autores: Yujia Li y Richard Zemel (University of Toronto); Marc Brockschmidt y Daniel Tarlow (Microsoft Research). El trabajo principal se realizó durante una pasantía de Li en Microsoft Research.
