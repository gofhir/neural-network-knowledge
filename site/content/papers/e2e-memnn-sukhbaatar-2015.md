---
title: "End-To-End Memory Networks (2015)"
weight: 342
math: true
---

{{< paper-card
    title="End-To-End Memory Networks"
    authors="Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus"
    year="2015"
    venue="NeurIPS 2015"
    pdf="/papers/e2e-memnn-sukhbaatar-2015.pdf"
    arxiv="1503.08895" >}}
MemN2N convierte la [Memory Network](/papers/memory-networks-weston-2014) original en una arquitectura **entrenable extremo a extremo y sin supervisión de los hops**: reemplaza la selección dura (hard max) de la memoria por una **atención softmax continua (soft attention)** y retropropaga el error a través de múltiples accesos a memoria hasta la entrada. Usa embeddings de entrada/salida (matrices A, B, C, W) y apila varios hops con atado de pesos. Compite con las Memory Networks fuertemente supervisadas en bAbI usando mucha menos supervisión, y mejora a RNN/LSTM en modelado de lenguaje (Penn Treebank, Text8). Es un **precursor directo de la self-attention de los Transformers**: producto interno + softmax + suma ponderada = query/key/value.
{{< /paper-card >}}

---

## Contexto

Hacia 2014–2015 resurgieron los modelos con **almacenamiento explícito y atención**: Memory Networks (Weston et al.), Neural Turing Machines (Graves et al., 2014) y RNNsearch (Bahdanau et al., 2015). Todos atacan dos desafíos de fondo de la IA: dar **múltiples pasos de cómputo** para responder una pregunta, y modelar **dependencias de largo plazo** en secuencias. El contraste con RNN/LSTM es la clave: en esas redes la memoria es el estado oculto, latente e inestable a lo largo de escalas temporales largas. MemN2N opta en cambio por una **memoria global** con funciones de lectura y escritura compartidas.

El antecedente que el paper busca destronar es la **Memory Network de Weston, Chopra & Bordes (2015)**: poderosa, pero con una limitación práctica decisiva. No era fácil de entrenar por backpropagation y **requería supervisión en cada capa**. En el dataset de QA había que indicarle explícitamente *cuáles* oraciones de soporte (supporting facts) eran relevantes para cada pregunta. Esa anotación rara vez existe en tareas reales (modelado de lenguaje, QA realista). MemN2N nace para quitar esa muleta: al volver continua la lectura de memoria, **se puede entrenar end-to-end desde pares entrada–salida**.

Frente a la Neural Turing Machine —que accede por contenido *y* por dirección— MemN2N solo permite acceso por contenido (el acceso por dirección llega de forma limitada vía features temporales) y es más simple: escribe cada memoria secuencialmente, sin operaciones como el *sharpening*. Frente a RNNsearch, su "memoria" es análoga al mecanismo de atención de Bahdanau, pero con dos diferencias decisivas: atiende sobre **muchas** memorias (no una sola oración) y hace **varios hops** antes de emitir una salida.

## Contribución central

MemN2N toma una versión *continua* de la Memory Network y la hace entrenable extremo a extremo sin supervisión de los hops. Cuatro ideas la componen:

1. **Lectura por soft attention.** En lugar del hard max (seleccionar la memoria de mayor score), una distribución **softmax** sobre las memorias, computada por producto interno entre la consulta embebida y cada memoria. La respuesta es una suma ponderada de los vectores de salida. Todo queda diferenciable.
2. **Embeddings de entrada/salida (matrices A, B, C, W).** Cada elemento de memoria tiene una representación de *entrada* (matriz A) para casar con la consulta y una de *salida* (matriz C) para construir la respuesta; la consulta se embebe con B y la predicción final con W.
3. **Múltiples hops apilados con weight tying.** Las capas de memoria se apilan (típicamente K = 3), cada una refina el estado interno *u*, y se comparten pesos entre capas (esquemas *adjacent* y *layer-wise*) para regularizar y reducir parámetros.
4. **Aplicabilidad dual.** El mismo armazón sirve para **QA** (bAbI) y para **modelado de lenguaje** (Penn Treebank, Text8): no es un truco específico de tarea.

La idea que une todo: como toda la cadena de entrada a salida es suave, **el error se retropropaga a través de los múltiples accesos a memoria hasta la entrada**, eliminando la necesidad de etiquetas de soporte intermedias. El modelo "deduce por sí mismo, en entrenamiento y en test, qué oraciones son relevantes y cuáles son distractores".

## Método: una capa de memoria

El modelo recibe entradas discretas $x_1, \dots, x_n$ que almacena en memoria, una consulta $q$ y produce una respuesta $a$. Cada $x_i$ se embebe con una matriz $A$ en un **vector de memoria** $m_i$ de dimensión $d$. La consulta $q$ se embebe con $B$ en el **estado interno** $u$. El *match* entre $u$ y cada memoria se computa por producto interno seguido de softmax:

$$p_i = \mathrm{Softmax}(u^T m_i), \qquad \mathrm{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Así $p$ es un **vector de probabilidad sobre las entradas** — el peso de atención asignado a cada memoria. Cada $x_i$ tiene además un **vector de salida** $c_i$ (vía la matriz $C$), y el vector respuesta $o$ es la suma ponderada:

$$o = \sum_i p_i\, c_i$$

La predicción final pasa la suma de $o$ y $u$ por una matriz $W$ y un softmax:

$$\hat{a} = \mathrm{Softmax}(W(o + u))$$

Las matrices $A$, $B$, $C$ y $W$ se aprenden conjuntamente minimizando **cross-entropy** entre $\hat{a}$ y la etiqueta $a$, vía SGD.

### Múltiples hops y weight tying

Para $K$ hops, las capas se apilan: la entrada a cada capa por encima de la primera es la suma de la salida y la entrada de la capa previa, $u^{k+1} = u^k + o^k$. El modelo "hace varios pasos de cómputo antes de producir una salida al mundo exterior". Hay dos esquemas de atado de pesos:

- **Adjacent (adyacente):** el embedding de salida de una capa es el de entrada de la de arriba ($A^{k+1} = C^k$), con $W^T = C^K$ y $B = A^1$. Es el defecto en QA.
- **Layer-wise (estilo RNN):** entrada y salida iguales en todas las capas, con un mapeo lineal $H$ en la actualización ($u^{k+1} = H u^k + o^k$). Así MemN2N **se puede ver como una RNN** donde se separan salidas internas (consultar memoria) y externas (predecir). Es el esquema usado en modelado de lenguaje.

En esencia, MemN2N es la Memory Network original salvo que las operaciones **hard max se reemplazan por el ponderado continuo del softmax**.

### Codificaciones para QA

Dos representaciones de oración: **bag-of-words** (suma de embeddings de palabra, ignora el orden) y **Position Encoding (PE)**, $m_i = \sum_j l_j \cdot A x_{ij}$ con un vector $l_j$ estructurado que hace que el orden de palabras afecte a $m_i$. El **Temporal Encoding** suma una matriz aprendida $T_A(i)$ a cada memoria para codificar contexto temporal (saber que Sam está en el dormitorio *después* de la cocina); las oraciones se indexan en orden inverso. Una técnica de entrenamiento clave es el **Linear Start (LS)**: arrancar sin los softmax intermedios (modelo casi lineal) y reinsertarlos cuando la validación se estanca, lo que ayuda a evitar mínimos locales (la tarea 16 baja de 53.6% a 1.6% de error).

## Experimentos

**bAbI (QA sintético, 20 tareas).** Conjunto de afirmaciones + una pregunta de respuesta de una palabra; solo un subconjunto de afirmaciones es relevante y el resto son distractores. La diferencia con Weston et al.: ese subconjunto de soporte **ya no se entrega al modelo**. Configuración por defecto K = 3 hops, atado adjacent. El mejor MemN2N llega a **12.6%** de error medio (1k, PE+LS+RN, joint) frente a 6.7% del MemNN fuertemente supervisado y 51.3% del LSTM; en 10k, 4.2% vs 3.2% vs 36.4%. Es decir, **se acerca al modelo supervisado pese a usar mucha menos supervisión y supera con holgura a las líneas base débiles**. Hallazgo transversal: **más hops mejoran** (1 hop = 25.8% error medio; 2 hops = 15.6%; 3 hops = 13.3%), y el modelo aprende a concentrar la atención en las oraciones de soporte correctas sin que se le indiquen.

**Modelado de lenguaje (Penn Treebank, Text8).** A nivel de palabra: las $N$ palabras previas se embeben en celdas de memoria, sin pregunta ($q$ fijo a 0.1), con atado layer-wise y ReLU sobre la mitad de las unidades. La "secuencia" sobre la que la red es recurrente **no está en el texto, sino en los hops de memoria**. MemN2N logra **menor perplejidad** en ambos: 111 en Penn Treebank (7 hops) vs 129/115 de RNN/LSTM, y 147 en Text8 vs 154 del LSTM, con ~1.5× los parámetros de una RNN (la LSTM tiene ~4×). De nuevo, más hops mejoran; algunos hops se concentran en palabras recientes y otros tienen atención amplia, alternándose como un n-grama suavizado combinado con un cache que no decae exponencialmente.

## Limitaciones reconocidas

- **No iguala al supervisado.** MemN2N aún no alcanza exactamente a las Memory Networks fuertemente supervisadas, y ambos fallan en varias tareas bAbI 1k (razonamiento posicional, path finding, inducción básica sin no-linealidad).
- **Escalabilidad de la atención suave.** El softmax pondera *todas* las memorias, así que las búsquedas suaves pueden no escalar con memorias muy grandes; los autores proponen atención multiescala o hashing.
- **Acceso solo por contenido.** A diferencia de la NTM, el acceso por dirección llega solo de forma limitada vía features temporales.
- **Varianza por inicialización.** Mitigada con 10 reentrenamientos y selección del mejor — señal de un paisaje de optimización difícil.

## El puente hacia la atención de los Transformers

El aporte de mayor alcance es haber aislado y vuelto entrenable, sin supervisión intermedia, el patrón **"computar pesos de atención por producto interno + softmax, leer por suma ponderada, repetir"** — exactamente el núcleo de la atención de los Transformers (Vaswani et al., 2017):

- El **producto interno $u^T m_i$ seguido de softmax** es el ancestro directo del *scaled dot-product attention*. El estado interno $u$ es la **query**; los $m_i$ son las **keys**; los vectores de salida $c_i$ son los **values**; y la suma ponderada $o = \sum_i p_i c_i$ es la salida de una capa de atención.
- Los **múltiples hops apilados** prefiguran el **apilar capas de self-attention**: la self-attention puede leerse como múltiples hops de soft attention donde la "memoria" son las representaciones de todos los tokens de la secuencia.
- La separación **embedding de entrada (A) / salida (C)** anticipa la distinción **key / value**.

MemN2N demostró que la soft attention multi-hop es entrenable end-to-end y mejora con la profundidad de cómputo — dos lecciones que los Transformers llevarían al extremo.

## Por qué importa para la Clase 30

La [Clase 30](/clases/clase-30) ("Modelos con memoria externa") dedica varias slides a MemN2N con el ejemplo "¿Quién dirigió El Origen?" sobre una base de conocimiento de **tripletas** (sujeto–relación–objeto):

- **La base de tripletas** = el conjunto $\{x_i\}$ escrito a memoria. Cada hecho ("El Origen — dirigida_por — Christopher Nolan") se embebe en $m_i$ (matriz A) y $c_i$ (matriz C); la pregunta en $u$ (matriz B).
- **La lectura** = el producto interno + softmax asigna alta probabilidad a la tripleta relevante, y la suma ponderada recupera el objeto correcto. Es la **soft attention** que la clase contrasta con el lookup duro de una base de datos clásica.
- **Sin supervisión de los hops:** el modelo aprende solo a atender el hecho correcto entre distractores.
- **Múltiples hops** = preguntas que encadenan hechos (razonamiento multi-paso): un hop localiza la entidad intermedia, el siguiente el objeto final.

## Notas y enlaces

- Fundamento transversal: [redes de memoria](/fundamentos/redes-de-memoria) y [self-attention](/fundamentos/self-attention).
- Antecedente: [Memory Networks (Weston et al., 2014)](/papers/memory-networks-weston-2014).
- Sucesor (generaliza entrada/salida → clave/valor): [Key-Value Memory Networks (Miller et al., 2016)](/papers/key-value-memnn-miller-2016).
- Origen de las tareas de QA: [bAbI (Weston et al., 2015)](/papers/babi-weston-2015).
- Código: [github.com/facebook/MemNN](https://github.com/facebook/MemNN). Preprint: arXiv:1503.08895.
