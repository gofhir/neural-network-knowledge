---
title: "Redes Neuronales de Grafos (GNN)"
weight: 100
math: true
---

Una **red neuronal de grafos** (Graph Neural Network, GNN) es una familia de modelos diseñada para aprender directamente sobre datos cuya estructura es **relacional**: objetos conectados por relaciones, es decir, un **grafo**. Donde una CNN explota la regularidad de una rejilla de píxeles y una RNN la de una secuencia de tokens, una GNN explota la topología arbitraria de un grafo —una molécula, una red social, un grafo de conocimiento, el grafo de llamadas de un programa— sin aplastarlo antes a un vector de tamaño fijo. La idea organizadora de todo el campo es el **paso de mensajes** (*message passing*): cada nodo actualiza su representación intercambiando información con sus vecinos, repetidamente, de modo que tras varias iteraciones cada nodo "sabe" no solo de sí mismo sino del subgrafo que lo rodea. Este fundamento recorre qué es una GNN y por qué, el mecanismo que las unifica, las tareas que resuelven, la familia de modelos que define el campo, sus problemas conocidos y sus aplicaciones, y lo conecta con el resto del curso. Es el fundamento núcleo de la [Clase 27](/clases/clase-27).

---

## 1. Qué es una GNN y por qué los grafos importan

Muchísimos datos del mundo real **no viven en una rejilla ni en una secuencia**, sino en un grafo: átomos unidos por enlaces, usuarios que siguen a otros usuarios, papers que citan papers, entidades de una base de conocimiento unidas por relaciones, funciones que se llaman entre sí. Hasta finales de los 2000, el machine learning lidiaba con estos datos mediante un **preprocesamiento que "aplastaba" el grafo a un vector** de reales (degree counts, features hechas a mano), perdiendo en el proceso la dependencia topológica de la información de manera impredecible y dependiente del algoritmo. La apuesta fundacional de [Scarselli et al. (2009)](/papers/gnn-model-scarselli-2009) —el paper que acuña el término *graph neural network*— fue **procesar el grafo directamente**, aprendiendo una función que mapea un grafo y uno de sus nodos a un espacio vectorial sin aplanar nada de antemano.

### Notación de grafos

Formalmente, un grafo es $G = (V, E)$, con $V$ el conjunto de **nodos** (vértices) y $E \subseteq V \times V$ el conjunto de **aristas** (relaciones). Cada nodo $v$ puede traer un vector de **features** $x_v$ (tipo de átomo, texto de un documento, perfil de usuario) y cada arista un vector $e_{vw}$ (tipo de enlace, peso, distancia). La estructura se codifica en la **matriz de adyacencia** $A \in \{0,1\}^{|V|\times|V|}$, donde $A_{vw}=1$ si existe la arista $(v,w)$. El **grado** de un nodo es $d_v = \sum_w A_{vw}$, el número de vecinos, recogidos en la matriz diagonal de grados $D$. Denotamos por $\mathcal{N}(v) = \{w : (w,v) \in E\}$ al **vecindario** de $v$, el conjunto de nodos que le envían información. La cantidad de aristas $|E|$ es típicamente mucho menor que $|V|^2$ (los grafos reales son **dispersos**), lo que permite representar $A$ como matriz dispersa y obtener algoritmos cuya complejidad escala con $|E|$, no con $|V|^2$.

Un grafo es **no dirigido** si $A$ es simétrica (la relación es recíproca: dos amigos, dos átomos enlazados) y **dirigido** si no lo es (una cita, un hipervínculo, una llamada a función). En grafos dirigidos, un nodo que solo tiene aristas salientes —*out-only*, como el primer documento de una cadena de citas— nunca recibiría mensajes y su representación quedaría congelada en sus features iniciales; por eso es práctica común **añadir aristas inversas** (un canal de mensaje separado para entrantes y salientes, como hace [GGNN](/papers/ggnn-li-2015)) o **auto-conexiones** (una arista de cada nodo consigo mismo), para que la información pueda fluir en ambos sentidos y cada nodo conserve también su propio estado. Cuando las aristas además tienen **tipo** —cientos de relaciones distintas en un grafo de conocimiento— el grafo es un *multigrafo dirigido y etiquetado*, el caso que aborda [R-GCN](/papers/rgcn-schlichtkrull-2018).

### Invarianza a permutación: por qué CNN y RNN no sirven

La propiedad que define el problema —y que descarta a las arquitecturas anteriores— es la **invarianza a permutación**. Un grafo no tiene un orden canónico de sus nodos: si renumeramos los átomos de una molécula, sigue siendo la misma molécula y sus propiedades no cambian. Cualquier modelo correcto debe ser **invariante (o equivariante) a permutaciones de los nodos**: reordenar las filas y columnas de $A$ no debe alterar la predicción. Una CNN asume una rejilla con vecindades fijas ("el píxel de arriba", "el de la derecha"), que un grafo no tiene; una RNN asume un orden lineal, que tampoco existe entre los vecinos de un nodo. Aplicar una de ellas exigiría imponer un orden arbitrario y romper la simetría. Las GNN, en cambio, construyen la invarianza **dentro de la arquitectura**, usando operaciones de agregación que ignoran el orden (sumas, promedios, máximos sobre conjuntos de vecinos).

{{< concept-alert type="clave" >}}
Una GNN es a los grafos lo que una CNN es a las imágenes: una arquitectura cuyo *inductive bias* está hecho a la medida de la estructura del dato. La diferencia esencial es que el grafo **no tiene orden ni vecindad fija**, así que toda la disciplina gira en torno a operaciones **invariantes a permutación** —agregar información de vecinos sin que el resultado dependa de cómo numeramos los nodos.
{{< /concept-alert >}}

---

## 2. El mecanismo central: message passing

Casi todas las GNN modernas son instancias de un mismo esqueleto, el **paso de mensajes** (*message passing*), formalizado como marco unificador por [Gilmer et al. (2017)](/papers/mpnn-gilmer-2017), que demostró que al menos ocho arquitecturas publicadas entre 2013 y 2017 —hasta entonces presentadas como modelos dispares— eran casos particulares de un mismo esquema abstracto. La intuición es la de un proceso de difusión: la información de cada nodo "se contagia" a sus vecinos en cada capa, de modo que tras varias rondas cada representación integra evidencia de un vecindario cada vez más amplio. Su funcionamiento, en cada capa $t$, tiene cuatro fases —el detalle completo está en el [fundamento de message passing](/fundamentos/message-passing):

1. **Cálculo del mensaje.** Para cada arista, una función aprendida $M_t$ produce el mensaje que un vecino $w$ envía al nodo $v$, a partir de sus estados y la feature de la arista: $M_t(h_v^t, h_w^t, e_{vw})$.
2. **Agregación conmutativa.** El nodo $v$ combina los mensajes de **todos** sus vecinos con una operación invariante al orden —suma, promedio o máximo—: $m_v^{t+1} = \sum_{w \in \mathcal{N}(v)} M_t(h_v^t, h_w^t, e_{vw})$.
3. **Actualización.** Una función aprendida $U_t$ fusiona el mensaje agregado con el estado previo del nodo: $h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$.
4. **Repetir $N$ veces.** Se itera la propagación $N$ capas (o $T$ pasos temporales).

La agregación **debe ser conmutativa** (suma, media, max) precisamente para preservar la invarianza a permutación de la sección 1: el conjunto de vecinos no tiene orden, así que la operación que los combina no puede depender de cómo los enumeremos. Esto descarta operaciones sensibles al orden como una RNN sobre los vecinos —salvo que se las fuerce a serlo, como hace GraphSAGE alimentando su agregador LSTM con permutaciones aleatorias. Las distintas GNN se distinguen, en buena medida, por *qué* eligen para estas tres funciones: GCN usa una media normalizada por grado sin función de mensaje aprendida; GGNN usa una GRU como actualización; GAT pondera el mensaje de cada vecino con un coeficiente de atención; GIN suma y aplica un MLP.

Una consecuencia clave es el crecimiento del **campo receptivo** (*receptive field*): tras una capa, cada nodo ve a sus vecinos directos (1 salto); tras dos capas, a los vecinos de sus vecinos (2 saltos); tras $N$ capas, a todo su vecindario de orden $N$. El número de capas controla, por tanto, **cuán lejos en el grafo puede mirar cada nodo** —análogo a cómo apilar convoluciones agranda el campo receptivo de una CNN. Esto tiene un doble filo: pocas capas limitan el alcance de la información (no captura dependencias lejanas), pero demasiadas hacen que los vecindarios de orden $N$ se solapen tanto que los nodos se vuelven indistinguibles (el *over-smoothing* de la sección 5). La elección típica de **2 o 3 capas** es el compromiso empírico que equilibra ambas presiones.

---

## 3. Las tareas que resuelven las GNN

Una vez que el message passing produce un embedding $h_v$ por nodo, distintas tareas se montan encima.

- **Clasificación de nodos.** Asignar una etiqueta a cada nodo: clasificar un documento por su tema, una entidad por su tipo, un usuario como fraudulento. La predicción se lee directamente del embedding del nodo: $\hat{y}_v = \text{softmax}(W h_v^N)$. Es el escenario original de [GCN](/papers/gcn-kipf-2017), típicamente **semi-supervisado**: solo una fracción pequeña de los nodos tiene etiqueta (en Cora, ~20 por clase, ~5% del grafo), pero como cada capa mezcla features de nodos vecinos —etiquetados o no— el gradiente de la pérdida supervisada se distribuye por el grafo y actualiza también las representaciones de los nodos sin etiqueta. La estructura del grafo entra entonces *horneada en la arquitectura*, no como un término de regularización en la pérdida.
- **Clasificación / regresión de grafo.** Predecir una propiedad del grafo entero: si una molécula es tóxica, su energía cuántica, su solubilidad. Requiere un paso de **readout** (o *pooling*) que colapse todos los embeddings de nodo en un único vector invariante a permutación: $h_G = R(\{h_v^N : v \in V\})$, con $R$ una suma, un promedio, o un mecanismo más expresivo tipo *set2set*; luego $\hat{y} = \text{MLP}(h_G)$. Es la tarea de [MPNN](/papers/mpnn-gilmer-2017) en química.
- **Predicción de aristas (*link prediction*).** Estimar si una arista que no observamos debería existir: completar un grafo de conocimiento, recomendar una amistad, predecir una interacción proteína-proteína. Se puntúa un par de nodos con una función de scoring sobre sus embeddings, $s(u,v) = h_u^\top R\, h_v$ (un decoder bilineal tipo DistMult, como en [R-GCN](/papers/rgcn-schlichtkrull-2018)).
- **Selección de nodos.** Elegir uno o varios nodos del grafo según un criterio aprendido: por ejemplo, en navegación robótica seleccionar el siguiente waypoint, o en verificación de programas elegir sobre qué variable razonar a continuación. Se implementa puntuando cada nodo y aplicando una softmax (o un *argmax*) sobre el grafo entero.

Un mismo grafo puede sostener varias de estas tareas a la vez, y la diferencia entre ellas está casi enteramente en la **cabeza** (*head*) que se monta sobre los embeddings $h_v$, no en el cuerpo de message passing, que es compartido. Esto convierte a la GNN en un **extractor de representaciones de propósito general** para datos relacionales, igual que un encoder preentrenado lo es para texto o imágenes.

---

## 4. La familia de modelos

El campo se entiende mejor como un **espacio de diseño**: todas las GNN comparten el esqueleto de message passing y difieren en *cómo calculan el mensaje*, *cómo agregan* y *cómo actualizan*. La siguiente tabla recorre los modelos canónicos en orden histórico.

| Modelo | Año / paper | Idea distintiva | Mensaje / actualización |
|---|---|---|---|
| **GNN original** | [Scarselli 2009](/papers/gnn-model-scarselli-2009) | Difusión hasta **punto fijo** (contracción, teorema de Banach) | Itera una función de transición contractiva hasta el equilibrio |
| **GGNN** | [Li 2015](/papers/ggnn-li-2015) | Recurrencia desenrollada $T$ pasos fijos, entrenada con BPTT | Mensaje $m = E_k h$ (matriz $E_k$ por tipo de arista); update con **GRU** |
| **GCN** | [Kipf 2017](/papers/gcn-kipf-2017) | Aproximación de 1.er orden de la convolución **espectral** | Promedio de vecinos normalizado por grado $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ + auto-conexión |
| **GraphSAGE** | [Hamilton 2017](/papers/graphsage-hamilton-2017) | **Inductivo**: aprende funciones agregadoras + muestreo de vecinos | Agregadores mean / pool / LSTM sobre un vecindario muestreado; concat con $h$ |
| **GAT** | [Veličković 2018](/papers/gat-velickovic-2018) | Pondera vecinos con **atención** aprendida por contenido | Pesos $\alpha_{ij}$ de self-attention enmascarada sobre el vecindario |
| **MPNN** | [Gilmer 2017](/papers/mpnn-gilmer-2017) | **Marco unificador**: $M_t$, $U_t$, $R$ recuperan todos los anteriores | *Edge network* $M(h_v,h_w,e_{vw}) = A(e_{vw})\,h_w$; update GRU; readout set2set |
| **R-GCN** | [Schlichtkrull 2018](/papers/rgcn-schlichtkrull-2018) | Pesos **por tipo de relación** para grafos de conocimiento | Matriz $W_r$ distinta por relación $r$ y dirección; basis decomposition |
| **GIN** | [Xu 2019](/papers/gin-xu-2019) | **Máxima expresividad**: agregador inyectivo (suma + MLP) | $h_v = \text{MLP}\big((1+\epsilon)\,h_v + \sum_{w} h_w\big)$ |

El recorrido cuenta una historia coherente, que vale la pena seguir porque ilumina las decisiones de diseño.

[Scarselli (2009)](/papers/gnn-model-scarselli-2009) partía de un proceso de **difusión iterado hasta un punto fijo**: cada nodo actualizaba su estado intercambiando información con sus vecinos hasta alcanzar un equilibrio estable, cuya existencia y unicidad garantizaba exigiendo que la función de transición fuera una **contracción** (teorema de punto fijo de Banach). Elegante en teoría, pero costoso y delicado de entrenar: iterar hasta convergencia es lento y la restricción de contracción limita la expresividad.

[GGNN (Li 2015)](/papers/ggnn-li-2015) lo modernizó con dos ideas de las RNN: **desenrollar la recurrencia un número fijo de pasos** $T$ (en vez de iterar hasta el punto fijo) y usar una **GRU** como función de actualización, entrenada con *backpropagation through time*. El mensaje es $m = E_k h$, con una matriz $E_k$ por tipo de arista. Es el representante de la familia **recurrente/gated**, y su aplicación motivadora —verificación de programas, inferir fórmulas lógicas sobre el heap— ya anticipaba las salidas estructuradas.

[GCN (Kipf 2017)](/papers/gcn-kipf-2017) —la GNN más citada de la historia— derivó la capa más simple posible **como aproximación de primer orden de la convolución espectral** sobre grafos, reduciéndola a la regla $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)})$. En palabras: "promediar los features de los vecinos normalizados por grado, transformar con una matriz aprendida y aplicar una no-linealidad". El *truco de renormalización* —añadir auto-conexiones $\tilde{A} = A + I$ y renormalizar— estabiliza el apilamiento de capas y se volvió estándar de facto. No hay función de mensaje aprendida aparte: por eso es el message passing mínimo y el punto de partida pedagógico.

[GraphSAGE (Hamilton 2017)](/papers/graphsage-hamilton-2017) la volvió **inductiva y escalable**: en vez de aprender un embedding por nodo, aprende **funciones de agregación** (mean, pooling, LSTM) sobre un **vecindario muestreado** de tamaño fijo, de modo que el modelo entrenado se aplica a nodos y grafos nuevos sin reentrenar. El nombre lo dice: *SAmple and aggreGatE*.

[GAT (Veličković 2018)](/papers/gat-velickovic-2018) reemplazó la normalización fija por grado por **pesos de atención aprendidos por contenido**: en vez de promediar a todos los vecinos por igual, calcula un coeficiente $\alpha_{ij}$ que mide cuánto importa cada vecino según sus features, con self-attention enmascarada al vecindario. Aplicable a problemas inductivos y sin operaciones matriciales costosas.

[MPNN (Gilmer 2017)](/papers/mpnn-gilmer-2017) demostró que **todos ellos son instancias de un mismo marco** especificado por tres funciones ($M_t$, $U_t$, $R$), convirtiendo un zoo de arquitecturas dispares en una familia parametrizada; su *edge network* aprende a mapear un vector de arista a una matriz de paso de mensaje.

[R-GCN (Schlichtkrull 2018)](/papers/rgcn-schlichtkrull-2018) generalizó GCN a grafos **multi-relacionales** (grafos de conocimiento) con una matriz de pesos $W_r$ distinta por tipo de relación y dirección, controlando la explosión de parámetros con *basis decomposition*.

Y [GIN (Xu 2019)](/papers/gin-xu-2019) cerró el círculo teórico: mostró *qué tan poderosas* pueden ser estas redes (el techo es 1-WL) y diseñó la arquitectura más simple que alcanza ese límite, usando una agregación **inyectiva** (suma seguida de MLP).

---

## 5. Problemas conocidos

Las GNN tienen patologías propias, bien caracterizadas, que delimitan dónde funcionan y dónde no.

**Over-smoothing (sobre-suavizado).** Al apilar muchas capas, las representaciones de todos los nodos tienden a **colapsar hacia un valor común**: cada capa promedia vecindarios cada vez más grandes, y en el límite todos los nodos ven (casi) todo el grafo y se vuelven indistinguibles. Esto explica por qué las GNN, a diferencia de las redes profundas en visión, suelen rendir mejor con **solo 2 o 3 capas**; ya el paper de GCN documentaba la caída de rendimiento a partir de ~7 capas sin conexiones residuales.

**Over-squashing (sobre-compresión).** Cuando la información debe viajar entre nodos lejanos, debe atravesar cuellos de botella del grafo, donde **mensajes de un vecindario exponencialmente grande se comprimen en un vector de tamaño fijo**, perdiéndose. Es el reverso del over-smoothing: limita la capacidad de capturar dependencias de **largo alcance**. Las estrategias clásicas para mitigarlo —ya exploradas por [MPNN](/papers/mpnn-gilmer-2017) en química— son introducir **aristas virtuales** entre nodos no conectados o un **nodo maestro** conectado a todo el grafo, que sirve de pizarra global por la que la información salta sin atravesar la topología; ambas, sin embargo, cuestan en cómputo y diluyen el *inductive bias* del grafo.

**Transductivo vs. inductivo.** Un modelo **transductivo** (como el GCN original) aprende sobre un grafo fijo y completo, precalculando la adyacencia una vez; no puede generar embeddings para nodos nuevos sin reentrenar. Un modelo **inductivo** (como [GraphSAGE](/papers/graphsage-hamilton-2017)) aprende una **función de agregación reutilizable**, aplicable a nodos o grafos completamente nuevos —imprescindible en producción, donde llegan posts, usuarios y videos nuevos sin parar.

**Escalabilidad.** El entrenamiento *full-batch* (procesar el grafo entero en cada paso, como el GCN original) no cabe en memoria para grafos con miles de millones de aristas, y para una GNN de $K$ capas habría que almacenar el vecindario de orden $K$ de cada nodo, que en grafos densos explota. La solución dominante es el **muestreo de vecindario** de GraphSAGE: en vez de agregar todos los vecinos, muestrear un subconjunto de tamaño fijo $S_i$ por capa, fijando así el costo por *batch* en $O(\prod_i S_i)$ independientemente del tamaño del grafo. En la práctica, $K=2$ con $S_1 \cdot S_2 \le 500$ ya rinde bien, y es lo que hizo viable desplegar GNN sobre los grafos de miles de millones de nodos de la industria.

**Expresividad limitada (1-WL).** [GIN](/papers/gin-xu-2019) demostró que las GNN de message passing son **a lo más tan poderosas como el test de isomorfismo de Weisfeiler-Lehman de 1 dimensión** (1-WL): hay pares de grafos no isomorfos que *ninguna* GNN de esta familia puede distinguir. Alcanzar ese techo exige que la agregación sea **inyectiva** —y por eso la suma es estrictamente más poderosa que el promedio, y el promedio más que el máximo. El detalle está en el [fundamento de expresividad de GNN](/fundamentos/expresividad-gnn).

---

## 6. Aplicaciones

Las GNN no son un ejercicio teórico: están desplegadas en producción y en ciencia.

- **Química y descubrimiento de fármacos.** [MPNN](/papers/mpnn-gilmer-2017) predice propiedades cuánticas de moléculas (energías de atomización, momentos dipolares, gaps HOMO-LUMO) ~300.000× más rápido que la simulación DFT —que tarda alrededor de una hora por molécula pequeña—, alcanzando *chemical accuracy* en 11 de 13 propiedades del benchmark QM9. La molécula es el grafo: átomos como nodos (con tipo, hibridación, número atómico), enlaces como aristas (single/double/triple/aromatic, o distancias). Es el ejemplo canónico de por qué las GNN importan en ciencia: aprender una función rápida que imite un cálculo físico costoso.
- **Grafos de conocimiento.** [R-GCN](/papers/rgcn-schlichtkrull-2018) completa bases de conocimiento (Wikidata, Freebase, DBpedia) mediante *link prediction* (recuperar tripletas sujeto-relación-objeto faltantes) y *entity classification* (asignar tipos a entidades), modelando cientos de tipos de relación con pesos por relación. El modelo se plantea como un autoencoder: un encoder R-GCN produce embeddings de entidad y un decoder DistMult puntúa las tripletas, con mejoras de ~30% sobre el decoder solo en FB15k-237.
- **Recomendación a escala web.** PinSage (Pinterest) aplica una GNN al grafo bipartito pin–tablero para fusionar contenido visual con la señal colaborativa, sobre miles de millones de nodos; es un descendiente directo de [GraphSAGE](/papers/graphsage-hamilton-2017). Ver el [fundamento de sistemas de recomendación](/fundamentos/recommender-systems).
- **Código y programas.** Allamanis et al. representan el programa como grafo enriquecido —el árbol de sintaxis más aristas de flujo de datos, orden de ejecución y uso de variables— y entrenan una GNN para detectar bugs sutiles como el mal uso de variables (*VarMisuse*: usar la variable equivocada de entre las que están en scope), aprovechando que las dependencias del código son intrínsecamente relacionales y que un modelo secuencial sobre el texto las perdería. Es el mismo espíritu del GGNN aplicado a verificación de programas.
- **Navegación robótica.** GraphNav y métodos afines usan GNN sobre un grafo topológico del entorno (nodos = lugares, aristas = transiciones posibles) para planificar trayectorias, seleccionando el siguiente nodo objetivo a partir de la estructura espacial aprendida en vez de un mapa métrico denso.

---

## 7. Conexión con el curso

Las GNN no llegan aisladas: conectan hacia atrás con varias piezas centrales del curso.

**Atención (Clase 15) y Transformers (Clase 14).** La conexión más profunda del campo: un **Transformer es, en esencia, un GAT sobre el grafo completo**. La self-attention de un Transformer hace que cada token atienda a *todos* los demás con pesos aprendidos por contenido —exactamente lo que hace [GAT](/papers/gat-velickovic-2018), salvo que GAT **enmascara la atención al vecindario del grafo** en vez de dejarla actuar sobre la secuencia entera. Visto al revés: una secuencia es un grafo completamente conectado (todo token con todo token), de modo que el Transformer es el caso particular de una GNN con atención sobre un grafo denso, mientras que GAT es la generalización a un grafo disperso arbitrario. La invarianza a permutación lo confirma: la self-attention es invariante al orden de los tokens, y por eso el Transformer necesita inyectar *positional encodings* para recuperar el orden de la secuencia —exactamente la información estructural que en una GNN ya está dada por las aristas. Por eso GAT, publicado pocos meses después de *Attention is all you need*, es el puente entre las dos ideas: el mecanismo de atención cruza de las secuencias a los grafos arbitrarios, y la lección es de doble vía.

**Recomendación (Clase 25).** PinSage y los recomendadores basados en grafos llevan el message passing al grafo usuario–ítem, fusionando el filtrado colaborativo con el contenido de los nodos. La [Clase 25](/clases/clase-25) trata este patrón en detalle.

**Embeddings.** Los $h_v$ que produce una GNN son embeddings de nodo, primos de los embeddings de palabra y de usuario/ítem vistos antes en el curso: vectores densos aprendidos donde la geometría codifica relaciones.

Todo esto cae bajo el [dominio de datos estructurados](/dominios/estructurados), donde el grafo es la estructura nativa del problema.

---

## 8. Resumen

1. Una **GNN** aprende directamente sobre datos **relacionales** (grafos), sin aplastarlos antes a un vector; su *inductive bias* está hecho a la medida del grafo, como el de una CNN para imágenes.
2. La propiedad que define el problema es la **invarianza a permutación**: no hay orden canónico de nodos, así que CNN y RNN no sirven directamente y toda operación debe ser conmutativa.
3. El mecanismo unificador es el **message passing**: calcular mensajes, agregarlos de forma conmutativa, actualizar el estado, y repetir $N$ veces; el número de capas fija el **campo receptivo**.
4. Las tareas son **clasificación de nodos**, **clasificación de grafo** (con readout/pooling), **predicción de aristas** y **selección de nodos**.
5. La familia va de **Scarselli (punto fijo)** → **GGNN (GRU)** → **GCN (promedio espectral)** → **GraphSAGE (sampling inductivo)** → **GAT (atención)** → **MPNN (marco unificador)** → **R-GCN (relaciones)** → **GIN (expresividad)**.
6. Sus patologías son **over-smoothing**, **over-squashing**, la tensión **transductivo/inductivo**, la **escalabilidad** (resuelta con sampling) y la **expresividad limitada por 1-WL**.
7. Están desplegadas en **química, grafos de conocimiento, recomendación, análisis de código y robótica**.
8. Conectan con el curso vía **atención y Transformers** (un Transformer es un GAT sobre grafo completo) y la **recomendación** de la Clase 25.

---

## Para profundizar

- [The Graph Neural Network Model (Scarselli et al. 2009)](/papers/gnn-model-scarselli-2009) — el paper fundacional que acuña el término GNN; difusión hasta punto fijo.
- [Gated Graph Sequence Neural Networks (Li et al. 2015)](/papers/ggnn-li-2015) — recurrencia desenrollada con GRU; la familia gated del message passing.
- [Semi-Supervised Classification with GCN (Kipf y Welling 2017)](/papers/gcn-kipf-2017) — la GNN más citada; el truco de renormalización y la conexión con Weisfeiler-Lehman.
- [Inductive Representation Learning on Large Graphs (Hamilton et al. 2017)](/papers/graphsage-hamilton-2017) — GraphSAGE: agregadores aprendidos, muestreo de vecinos e inductividad.
- [Graph Attention Networks (Veličković et al. 2018)](/papers/gat-velickovic-2018) — pesos de vecino aprendidos por atención; el puente con los Transformers.
- [Neural Message Passing for Quantum Chemistry (Gilmer et al. 2017)](/papers/mpnn-gilmer-2017) — el marco unificador $M_t$/$U_t$/$R$ y el estado del arte en QM9.
- [Modeling Relational Data with GCN (Schlichtkrull et al. 2018)](/papers/rgcn-schlichtkrull-2018) — R-GCN: pesos por relación para grafos de conocimiento.
- [How Powerful are Graph Neural Networks? (Xu et al. 2019)](/papers/gin-xu-2019) — GIN: la cota de expresividad 1-WL y por qué la suma gana.

**Fundamentos relacionados:** [Message Passing](/fundamentos/message-passing) · [Expresividad de GNN](/fundamentos/expresividad-gnn) · [Sistemas de Recomendación](/fundamentos/recommender-systems) · [Clase 27](/clases/clase-27) · [Dominio: Datos Estructurados](/dominios/estructurados)
