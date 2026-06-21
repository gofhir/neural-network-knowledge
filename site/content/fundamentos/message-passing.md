---
title: "Message Passing en GNN"
weight: 101
math: true
---

El **paso de mensajes** (*message passing*) es el mecanismo que hace funcionar a casi todas las redes neuronales de grafos modernas. Mientras el [fundamento de GNN](/fundamentos/redes-neuronales-de-grafos) recorre la *familia de modelos* —qué es una GNN, qué tareas resuelve, qué arquitecturas existen—, este fundamento se concentra en el **mecanismo en sí**: cómo, en detalle, un nodo recolecta información de sus vecinos, la combina y se actualiza, qué ecuaciones lo gobiernan, cómo se implementa eficientemente con álgebra de matrices, y por qué ciertas decisiones de diseño (la agregación conmutativa, el número de capas) no son negociables. La pieza central es el marco **MPNN** (*Message Passing Neural Network*) que [Gilmer et al. (2017)](/papers/mpnn-gilmer-2017) formuló para demostrar que ocho arquitecturas dispares de GNN eran, en realidad, instancias de un mismo esquema gobernado por tres funciones aprendidas. Ese marco es el que vertebra la [Clase 27](/clases/clase-27) y el que estudiaremos aquí hasta el último engranaje.

---

## 1. El framework MPNN: tres funciones, dos fases

La contribución conceptual de Gilmer fue notar que el *forward pass* de cualquier GNN sobre un grafo no dirigido $G$ —con features de nodo $x_v$ y features de arista $e_{vw}$— puede describirse con **dos fases** y **tres funciones diferenciables aprendidas**. Las dos fases son: una **fase de message passing** que corre durante $T$ pasos temporales (las "capas"), y una **fase de readout** que produce la predicción a nivel de grafo.

Cada nodo $v$ mantiene un **estado oculto** $h_v^t$, inicializado en sus features ($h_v^0 = x_v$). En cada paso $t$, el estado se actualiza en dos movimientos. Primero se calcula el **mensaje agregado** que llega a $v$ desde sus vecinos $\mathcal{N}(v)$:

$$
m_v^{t+1} = \sum_{w \in \mathcal{N}(v)} M_t\big(h_v^t,\, h_w^t,\, e_{vw}\big) \qquad (1)
$$

y luego se actualiza el estado del nodo combinando ese mensaje con el estado previo:

$$
h_v^{t+1} = U_t\big(h_v^t,\, m_v^{t+1}\big) \qquad (2)
$$

Tras $T$ pasos, una **función de readout** colapsa todos los estados finales en un vector de grafo y, de ahí, en la predicción:

$$
\hat{y} = R\big(\{\, h_v^T \mid v \in G \,\}\big) \qquad (3)
$$

Las tres funciones aprendidas son, entonces:

- **Función de mensaje $M_t(h_v, h_w, e_{vw})$** — calcula qué le dice un vecino $w$ al nodo $v$, en función de ambos estados y la arista que los conecta.
- **Función de actualización $U_t(h_v, m_v)$** — fusiona los mensajes recibidos con el estado actual del nodo.
- **Función de readout $R(\{h_v\})$** — agrega los estados de todos los nodos en una representación del grafo entero (solo necesaria para tareas a nivel de grafo).

{{< concept-alert type="clave" >}}
Todo el diseño de una GNN se reduce a **tres decisiones**: qué función de mensaje $M_t$, qué función de actualización $U_t$ y qué readout $R$. Especificar estas tres piezas **recupera cualquier arquitectura concreta** —GCN, GAT, GraphSAGE, GGNN— como un caso particular. Por eso Gilmer no propuso "otra GNN" sino el lenguaje común con el que se describen todas: el subcampo entero pasó a hablar en términos de *message / aggregate / update*.
{{< /concept-alert >}}

El número de pasos $T$ (equivalentemente, el número de capas) y la dimensión $d$ del estado oculto son los hiperparámetros estructurales. En los experimentos de QM9 de Gilmer, cualquier $T \ge 3$ funcionaba bien, con *weight tying* (la misma $M_t$ y $U_t$ reutilizadas en cada paso, al estilo de una recurrencia).

---

## 2. Las cuatro etapas del mecanismo

La [Clase 27](/clases/clase-27) descompone el message passing en **cuatro etapas**, que se leen directamente de las ecuaciones (1)–(2). Verlas separadas ayuda a entender qué grado de libertad aporta cada una.

**Etapa 1 — Cálculo del mensaje.** Para cada arista $(w, v)$ se evalúa $M_t(h_v^t, h_w^t, e_{vw})$. El mensaje puede depender de tres cosas: el estado del **nodo origen** $h_w$ (qué tiene para decir el vecino), el estado del **nodo destino** $h_v$ (a quién se lo dice), y el **tipo o feature de la arista** $e_{vw}$ (por qué canal viaja). El caso más simple ignora origen-destino y arista y solo copia el estado del vecino; el más rico —el *edge network* de Gilmer— transforma el vector de arista en una matriz $A(e_{vw})$ y calcula $M = A(e_{vw})\,h_w$, permitiendo aristas con valores continuos (distancias) en vez de tipos discretos.

**Etapa 2 — Traspaso.** Cada mensaje viaja por su arista desde el origen hasta el destino. En la implementación es la fase de *scatter/gather*: se recolectan, para cada nodo $v$, todos los mensajes entrantes. Conceptualmente es trivial, pero es donde reside el costo: hay un mensaje por arista, no por nodo.

**Etapa 3 — Combinación (agregación).** El nodo $v$ funde todos los mensajes recibidos en uno solo, con una **función conmutativa** —invariante a permutación—: típicamente **suma**, **media** o **máximo**:

$$
m_v^{t+1} = \bigoplus_{w \in \mathcal{N}(v)} M_t(h_v^t, h_w^t, e_{vw}), \qquad \bigoplus \in \{\textstyle\sum,\ \mathrm{mean},\ \max\}.
$$

La elección del agregador no es cosmética: determina **qué puede distinguir** la red. La suma conserva el conteo y la multiplicidad (cuántos vecinos hay y de qué tipo); la media normaliza y pierde la magnitud del vecindario; el máximo solo retiene el rasgo dominante. Esto tiene consecuencias formales de expresividad —la suma es estrictamente más poderosa que la media, y la media más que el máximo— que se desarrollan en el [fundamento de expresividad de GNN](/fundamentos/expresividad-gnn). Lo que aquí importa es por qué la operación **debe ser conmutativa**: el conjunto de vecinos no tiene orden (sección 5).

**Etapa 4 — Actualización del nodo.** Finalmente $U_t$ combina el **estado anterior** $h_v^t$ con el **mensaje agregado recibido** $m_v^{t+1}$ para producir el nuevo estado $h_v^{t+1}$. Aquí se decide cuánto del pasado conserva el nodo y cuánto incorpora del entorno: puede ser una suma residual ($h_v + m_v$), una transformación con no-linealidad ($\sigma(W m_v)$), o una **GRU** que regula con compuertas qué información retener. La actualización es lo que evita que el nodo "olvide" su propia identidad al promediarse con los vecinos.

---

## 3. Implementación matricial: la adyacencia como operador de propagación

Iterar sobre nodos y aristas con bucles sería inviable. En la práctica, **una capa de message passing es una multiplicación de matrices**, y entender esa equivalencia es clave para implementarla y para razonar sobre su costo.

Apilemos los estados de los $n = |V|$ nodos en una matriz $H \in \mathbb{R}^{n \times d}$ (una fila por nodo). Sea $A \in \{0,1\}^{n \times n}$ la matriz de adyacencia. Considérese el producto:

$$
(A H)_v = \sum_{w} A_{vw}\, h_w = \sum_{w \in \mathcal{N}(v)} h_w.
$$

Es decir, **$A H$ envía a cada nodo la suma de los estados de sus vecinos** en una sola operación. La matriz de adyacencia actúa como un **operador de propagación**: multiplicar por $A$ ejecuta simultáneamente las etapas de traspaso (etapa 2) y de combinación por suma (etapa 3) para *todos* los nodos a la vez. La etapa de actualización con una transformación lineal aprendida $W$ y no-linealidad $\sigma$ se monta encima:

$$
H^{t+1} = \sigma\big(A H^t W\big).
$$

Esta es, en esencia, una capa de GNN. La suma cruda tiene un problema: los nodos de alto grado acumulan magnitudes enormes y desestabilizan el entrenamiento, mientras que los nodos aislados se desvanecen. La solución es **normalizar por el grado**. Con la matriz diagonal de grados $D$ (donde $D_{vv} = d_v$), la normalización simétrica de [GCN](/papers/gcn-kipf-2017) reemplaza $A$ por:

$$
\hat{A} = \tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}, \qquad \tilde{A} = A + I,
$$

donde $\tilde{A} = A + I$ añade **auto-conexiones** (cada nodo es vecino de sí mismo, para que la actualización conserve su estado), y $\tilde{D}^{-1/2}(\cdot)\tilde{D}^{-1/2}$ pondera cada mensaje por $1/\sqrt{d_v d_w}$. El resultado es la célebre regla de propagación:

$$
H^{t+1} = \sigma\big(\hat{A}\, H^t\, W^t\big).
$$

{{< concept-alert type="info" >}}
La identidad **$A H$ = "enviar y sumar los mensajes de los vecinos"** es la bisagra entre la visión local (nodo a nodo) y la implementación real (álgebra lineal). Como los grafos reales son **dispersos** ($|E| \ll n^2$), $A$ se almacena como matriz dispersa y el producto $A H$ cuesta $O(|E| \cdot d)$ en vez de $O(n^2 d)$: la eficiencia de las GNN viene precisamente de explotar la dispersión de la adyacencia.
{{< /concept-alert >}}

Sobre grafos densos o totalmente conectados, en cambio, el costo sube a $O(n^2 d^2)$ por paso; Gilmer propuso el truco de *multiple towers* —partir el embedding de $d$ en $k$ bloques de $d/k$ y propagar cada uno por separado— para bajarlo a $O(n^2 d^2 / k)$ sin perder la invarianza a permutación.

---

## 4. Variantes de las funciones según el modelo

La gracia del marco MPNN es que cada GNN famosa se obtiene **eligiendo $M_t$, $U_t$ y el agregador**. La siguiente tabla traduce los modelos canónicos al lenguaje de message passing; obsérvese que la columna de mensaje y la de actualización son, literalmente, las decisiones de diseño.

| Modelo | Mensaje $M_t$ | Agregación | Actualización $U_t$ | Idea |
|---|---|---|---|---|
| **[GCN](/papers/gcn-kipf-2017)** | $\frac{1}{\sqrt{d_v d_w}}\, h_w$ | suma normalizada | $\sigma(W\, m_v)$ | promedio espectral por grado |
| **[GGNN](/papers/ggnn-li-2015)** | $E_{k}\, h_w$ (matriz por tipo de arista) | suma | $\mathrm{GRU}(h_v, m_v)$ | recurrencia con compuertas |
| **GraphSAGE** | $h_w$ (sobre vecinos muestreados) | mean / pool / LSTM | $\sigma(W\cdot[\,h_v \,\|\, m_v\,])$ | agregadores aprendidos, inductivo |
| **[GAT](/papers/gat-velickovic-2018)** | $\alpha_{vw}\, W h_w$ | suma ponderada | $\sigma(m_v)$ | atención aprendida como peso |
| **R-GCN** | $W_r\, h_w$ (matriz por relación $r$) | suma por relación + propia | $\sigma(m_v)$ | un canal por tipo de relación |
| **MPNN** ([Gilmer](/papers/mpnn-gilmer-2017)) | $A(e_{vw})\, h_w$ (*edge network*) | suma | $\mathrm{GRU}(h_v, m_v)$ | arista → matriz, readout set2set |

Las diferencias cuentan una historia. **GCN** usa un peso fijo, determinado solo por la topología (los grados): el mensaje no se aprende, solo se normaliza. **GGNN** convierte la actualización en una **GRU**, tratando los $T$ pasos como una recurrencia desenrollada que decide con compuertas cuánto del nuevo mensaje incorpora. **GraphSAGE** generaliza la agregación a una **función aprendida** (mean, max-pooling, o incluso una LSTM) y concatena el estado propio con el agregado, lo que la vuelve **inductiva**: aprende cómo agregar, no embeddings fijos. **GAT** reemplaza el peso fijo de GCN por un **peso de atención aprendido** $\alpha_{vw}$ que depende del contenido de los dos nodos —la combinación deja de ser un promedio ciego y se vuelve una suma ponderada por relevancia. **R-GCN** introduce una **matriz de peso distinta por tipo de relación** $W_r$, indispensable en grafos de conocimiento con cientos de relaciones. Y el **edge network** de MPNN lleva esto al extremo: una red neuronal mapea el vector de arista $e_{vw}$ a la matriz $A(e_{vw})$ que transforma el mensaje, admitiendo features de arista continuas.

Todas comparten el esqueleto de las ecuaciones (1)–(3); difieren solo en con qué llenan las tres ranuras.

---

## 5. Invarianza y equivarianza a permutación

¿Por qué la agregación (etapa 3) **debe** ser conmutativa? Porque los vecinos de un nodo forman un **conjunto, no una secuencia**: no existe "el primer vecino" ni "el segundo". Si numeramos los átomos de una molécula de una forma o de otra, sigue siendo la misma molécula, y la predicción no puede cambiar. Formalmente, exigimos que el modelo sea **equivariante a permutaciones** en las capas intermedias e **invariante a permutaciones** en el readout.

Sea $P$ una matriz de permutación que reordena los nodos. **Equivarianza** significa que permutar la entrada permuta la salida de la misma manera: si $f$ es una capa de message passing, entonces

$$
f(P A P^\top,\, P H) = P\, f(A, H).
$$

Cada nodo obtiene el mismo embedding sin importar cómo se numeró el grafo —su estado "viaja" con él. **Invarianza** es la condición más fuerte que pide el readout: la representación del grafo entero no debe cambiar en absoluto al reordenar nodos,

$$
R(P H) = R(H).
$$

Una suma $\sum_v h_v$, una media o un máximo cumplen esto trivialmente porque ignoran el orden de los sumandos. Una concatenación o una RNN sobre los nodos **no** lo cumplen, y por eso están prohibidas como agregador o readout. Gilmer lo enuncia como un requisito de diseño no negociable: si $R$ no fuera invariante a permutación, el MPNN entero perdería la invarianza al isomorfismo de grafos, que es la razón misma para usar grafos en vez de aplanarlos a un vector. La cadena lógica es directa: **agregación conmutativa → capa equivariante → readout invariante → modelo invariante al isomorfismo.**

---

## 6. Receptive field, profundidad y sus patologías

Cada capa de message passing extiende el horizonte de un nodo en **un salto**. Tras la primera capa, $h_v^1$ depende de $v$ y de sus vecinos directos ($\mathcal{N}(v)$, 1 salto). Tras la segunda, $h_v^2$ depende de los vecinos de sus vecinos, porque los $h_w^1$ ya incorporaron *su* vecindario. En general:

$$
\text{tras } N \text{ capas, } h_v^N \text{ depende del subgrafo de nodos a } \le N \text{ saltos de } v.
$$

Ese subgrafo es el **campo receptivo** (*receptive field*) del nodo, exactamente análogo a cómo apilar convoluciones agranda el campo receptivo de una CNN. La consecuencia práctica: el **número de capas controla el alcance** de la información. Si una tarea exige razonar sobre dependencias a 5 saltos, se necesitan al menos 5 capas.

Pero aumentar la profundidad no sale gratis, y dos patologías lo limitan:

- **Over-smoothing (sobre-suavizado).** Cada capa promedia vecindarios cada vez más grandes; en el límite, todos los nodos ven (casi) todo el grafo y sus representaciones **colapsan hacia un vector común**, volviéndose indistinguibles. Por eso las GNN suelen rendir mejor con solo **2 o 3 capas**, a diferencia de las redes profundas de visión. Es un efecto del propio operador de propagación $\hat{A}$: aplicarlo repetidamente converge al autovector dominante, borrando las diferencias entre nodos.
- **Over-squashing (sobre-compresión).** Para que la información viaje entre nodos lejanos debe atravesar **cuellos de botella** del grafo, donde mensajes de un vecindario que crece exponencialmente con el número de saltos se comprimen en un vector de tamaño fijo. La señal de largo alcance se pierde. Es el reverso del over-smoothing: no es que todo se promedie, sino que lo lejano no llega.

Hay una tensión de fondo: tareas de largo alcance piden muchas capas, pero muchas capas inducen over-smoothing y over-squashing. Los paliativos —conexiones residuales, *jumping knowledge*, virtual nodes que conectan todo el grafo, o reescritura de aristas (*rewiring*)— buscan agrandar el alcance sin pagar el colapso.

---

## 7. Conexión: self-attention es message passing sobre un grafo completo

La conexión más profunda de este mecanismo con el resto del curso es que **la self-attention de los Transformers (Clase 14) es, exactamente, message passing sobre un grafo completamente conectado.** No es una analogía suelta: es una identidad estructural.

En un Transformer, cada token atiende a *todos* los demás tokens de la secuencia. Pensemos en la secuencia como un grafo donde cada token es un nodo y existe una arista entre cada par de nodos —un grafo **completo**. La operación de self-attention,

$$
h_v' = \sum_{w} \alpha_{vw}\, (W_V h_w), \qquad \alpha_{vw} = \mathrm{softmax}_w\!\left(\frac{(W_Q h_v)^\top (W_K h_w)}{\sqrt{d_k}}\right),
$$

calca el message passing: el **mensaje** de $w$ a $v$ es el valor proyectado $W_V h_w$; el **peso** $\alpha_{vw}$ es la atención aprendida por contenido; la **agregación** es la suma ponderada (conmutativa, por eso el Transformer es invariante a permutación y necesita *positional encodings* para reintroducir el orden). Esto es precisamente lo que hace [GAT](/papers/gat-velickovic-2018) —pesos de atención sobre los vecinos— con una sola diferencia: **GAT enmascara la atención al vecindario del grafo**, mientras el Transformer atiende a la secuencia entera. Una secuencia es un grafo completo; el grafo arbitrario es la generalización.

Por eso GAT, publicado pocos meses después de *Attention is all you need*, es el puente entre los dos mundos: lleva el [mecanismo de atención](/fundamentos/mecanismo-atencion) de las secuencias a los grafos arbitrarios. Visto al revés, un Transformer es una GNN que ha olvidado la estructura —ha decidido que todo conecta con todo y deja que la atención aprenda qué relaciones importan. Entender message passing es, entonces, entender la self-attention desde su forma más general.

---

## 8. Resumen

1. El **message passing** es el mecanismo que unifica casi todas las GNN, formalizado por el marco **MPNN** de [Gilmer et al. (2017)](/papers/mpnn-gilmer-2017) en dos fases (message passing + readout) y tres funciones aprendidas.
2. Las tres funciones son el **mensaje $M_t(h_v, h_w, e_{vw})$**, la **actualización $U_t(h_v, m_v)$** y el **readout $R(\{h_v\})$**; elegirlas recupera cualquier arquitectura concreta como caso particular.
3. El mecanismo tiene **cuatro etapas**: cálculo del mensaje (puede depender de origen, destino y arista), traspaso, combinación con una función **conmutativa** (suma/media/máximo), y actualización con el estado previo.
4. Una capa es, en la práctica, una **multiplicación de matrices**: $A H$ envía y suma los mensajes de los vecinos; la **normalización por grado** $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ estabiliza, y la dispersión de $A$ da eficiencia $O(|E| d)$.
5. Las GNN famosas son **elecciones de las tres ranuras**: GCN (promedio por grado), GGNN (update GRU), GraphSAGE (agregadores), GAT (atención como peso), R-GCN (mensaje por relación), MPNN (edge network).
6. La agregación **debe ser conmutativa** para garantizar **equivarianza** en las capas e **invarianza** en el readout, y con ello la invarianza al isomorfismo de grafos.
7. $N$ capas dan un **campo receptivo** de $N$ saltos; aumentar la profundidad choca con **over-smoothing** (colapso de representaciones) y **over-squashing** (compresión de la señal de largo alcance).
8. La **self-attention de los Transformers es message passing sobre un grafo completo**: GAT es el puente, y un Transformer es una GNN que conecta todo con todo.

---

## Para profundizar

- [Neural Message Passing for Quantum Chemistry (Gilmer et al. 2017)](/papers/mpnn-gilmer-2017) — el marco $M_t$/$U_t$/$R$ que define el vocabulario del mecanismo y alcanza precisión química en QM9.
- [Gated Graph Sequence Neural Networks (Li et al. 2015)](/papers/ggnn-li-2015) — la actualización con GRU; la recurrencia desenrollada del message passing.
- [Semi-Supervised Classification with GCN (Kipf y Welling 2017)](/papers/gcn-kipf-2017) — la regla de propagación matricial $\hat{A} H W$ y la normalización por grado.
- [Graph Attention Networks (Veličković et al. 2018)](/papers/gat-velickovic-2018) — la atención como peso de combinación; el puente con los Transformers.

**Fundamentos relacionados:** [Redes Neuronales de Grafos](/fundamentos/redes-neuronales-de-grafos) · [Expresividad de GNN](/fundamentos/expresividad-gnn) · [Mecanismo de Atención](/fundamentos/mecanismo-atencion) · [Clase 27](/clases/clase-27)
