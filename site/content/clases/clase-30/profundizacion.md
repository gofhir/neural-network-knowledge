---
title: "Profundizacion - Modelos con memoria externa"
weight: 20
math: true
---

> Esta pagina complementa la [teoria de la clase 30](/clases/clase-30/teoria) con las derivaciones formales de los modelos con memoria externa. Seis partes. **Parte I** aisla el mecanismo comun a toda la familia: la lectura como atencion suave (direccionamiento por contenido). **Parte II** formaliza la End-to-End Memory Network hop a hop y muestra su equivalencia con la self-attention de los Transformers. **Parte III** desarrolla la Key-Value MemNN y por que separar clave de valor agrega flexibilidad. **Parte IV** deriva la celda por bloque del Recurrent Entity Network, incluyendo por que la normalizacion implementa olvido. **Parte V** desarma las cinco operaciones diferenciables de la Neural Turing Machine. **Parte VI** describe brevemente las extensiones del Differentiable Neural Computer: asignacion dinamica y matriz de enlaces temporales.

---

## Parte I — La lectura como atencion suave (el mecanismo comun)

### I.1 El problema del lookup diferenciable

Toda la familia de modelos con memoria externa comparte un mismo nucleo. Tenemos una **memoria** representada como un conjunto de vectores $\{m_1, \dots, m_N\}$ (las filas de una matriz $M \in \mathbb{R}^{N\times d}$) y queremos **leer** de ella en respuesta a una **consulta** (query) $u \in \mathbb{R}^d$. En un computador clasico, el acceso es por *direccion exacta*: elegimos un indice $i^\star$ y devolvemos $m_{i^\star}$. Pero la operacion $i^\star = \arg\max_i \text{score}(u, m_i)$ es una funcion escalonada — no tiene gradiente util respecto de $u$ ni de los $m_i$, asi que no se puede entrenar la red que produce $u$ por retropropagacion.

La solucion, comun a toda la familia, es **relajar el lookup duro a un promedio ponderado suave**. En lugar de elegir una sola memoria, se asigna a cada una un peso $p_i \in [0,1]$ con $\sum_i p_i = 1$ y se devuelve la combinacion convexa.

### I.2 Derivacion del direccionamiento por contenido

El peso de cada memoria mide su **afinidad** con la consulta. La medida natural es el producto interno $u^\top m_i$, normalizado a una distribucion de probabilidad por el softmax:

$$
p_i = \mathrm{Softmax}(u^\top m_i) = \frac{\exp(u^\top m_i)}{\sum_{j=1}^{N}\exp(u^\top m_j)}
$$

La salida de la lectura es la suma de los vectores de memoria ponderada por esos pesos:

$$
\boxed{\; o = \sum_{i=1}^{N} p_i\, c_i \;}
$$

donde, en el caso mas general, $c_i$ es un **vector de salida** asociado a la memoria $i$ (que no tiene por que coincidir con el $m_i$ usado para casar). Cuando $c_i = m_i$, la lectura es literalmente un promedio de las filas de $M$.

{{< concept-alert type="clave" >}}
Esto es **direccionamiento por contenido** (content-based addressing): no accedemos a una direccion fisica, sino a las memorias *cuyo contenido se parece* a la consulta. La operacion entera $o = \sum_i \mathrm{Softmax}(u^\top m_i)\, c_i$ es suave y diferenciable tanto respecto de $u$ como de $\{m_i\}$ y $\{c_i\}$, asi que el gradiente fluye por la lectura hasta la red que produjo la consulta. Esa diferenciabilidad es la innovacion que destrabo todo el campo.
{{< /concept-alert >}}

### I.3 La fuerza de clave $\beta$ controla la nitidez

Un grado de libertad extra es escalar el producto interno por una **fuerza de clave** $\beta \ge 0$ antes del softmax:

$$
p_i = \mathrm{Softmax}\big(\beta\, K[u, m_i]\big), \qquad K[u,m] = \frac{u^\top m}{\lVert u\rVert\,\lVert m\rVert}
$$

donde $K$ es la similitud coseno (usada por la NTM y el DNC). El parametro $\beta$ interpola entre dos regimenes: con $\beta \to 0$ los pesos tienden a la distribucion uniforme ($p_i \to 1/N$, lectura difusa sobre toda la memoria); con $\beta \to \infty$ el softmax colapsa al $\arg\max$ ($p_{i^\star}\to 1$, lectura aguda casi-discreta). Permitir que el controlador *emita* $\beta$ deja que el modelo decida, en cada paso, cuan concentrada debe ser su lectura. Este unico mecanismo — query, scores por producto interno, softmax, suma ponderada de valores — reaparece, con variantes, en las cinco arquitecturas siguientes.

---

## Parte II — End-to-End Memory Networks (MemN2N): formalizacion

### II.1 Los tres embeddings y un hop

La [End-to-End Memory Network](/papers/e2e-memnn-sukhbaatar-2015) toma un conjunto de entradas discretas $x_1, \dots, x_n$ (los hechos/oraciones), una consulta $q$ y produce una respuesta. Define **tres matrices de embedding** de tamaño $d\times V$ (con $V$ el vocabulario):

- $A$ — *input memory*: embebe cada $x_i$ en un vector de memoria de entrada $m_i = A x_i$.
- $C$ — *output memory*: embebe cada $x_i$ en un vector de salida $c_i = C x_i$.
- $B$ — *query*: embebe la consulta en el estado interno $u = B q$.

Un hop ejecuta exactamente el mecanismo de la Parte I:

$$
p_i = \mathrm{Softmax}\big((Bq)^\top (A x_i)\big) = \mathrm{Softmax}(u^\top m_i), \qquad
o = \sum_i p_i\, (C x_i) = \sum_i p_i\, c_i
$$

### II.2 El apilamiento de hops

Para razonamiento multi-paso se apilan $K$ hops. El estado interno se actualiza sumando la lectura del hop al estado de entrada:

$$
\boxed{\; u^{k+1} = u^{k} + o^{k}, \qquad o^k = \sum_i \mathrm{Softmax}\big((u^k)^\top A^k x_i\big)\, C^k x_i \;}
$$

Cada hop tiene en principio sus propias matrices $A^k, C^k$, y la prediccion final se computa sobre el estado de la cima:

$$
\hat{a} = \mathrm{Softmax}\big(W u^{K+1}\big) = \mathrm{Softmax}\big(W(o^K + u^K)\big)
$$

El primer hop localiza una entidad intermedia; el segundo, partiendo de un $u^2$ ya enriquecido, recupera el objeto final. La red **aprende sola** a concentrar la atencion en los hechos de soporte correctos, sin etiquetas intermedias — esa es la diferencia con la Memory Network original, que exigia supervision en cada capa.

### II.3 Weight tying: adjacent y layer-wise

Para regularizar y reducir parametros se atan pesos entre hops, con dos esquemas:

| Esquema | Restricciones | Uso tipico |
|---|---|---|
| **Adjacent** | $A^{k+1} = C^k$, $W^\top = C^K$, $B = A^1$ | QA (bAbI) |
| **Layer-wise (estilo RNN)** | $A^1=\dots=A^K$, $C^1=\dots=C^K$, mas un mapeo lineal $H$ | Modelado de lenguaje |

En el esquema *adjacent*, la memoria de salida de un hop se reutiliza como memoria de entrada del siguiente: la lectura del hop $k$ "se convierte" en lo que casa el hop $k+1$. En el *layer-wise* la actualizacion lleva una transformacion lineal extra, $u^{k+1} = H u^k + o^k$, y el modelo se lee como una RNN donde la recurrencia no esta en el texto sino **en los hops de memoria**.

### II.4 Codificaciones de posicion y temporal

El embedding bag-of-words $m_i = \sum_j A x_{ij}$ pierde el orden de las palabras. El **position encoding** lo recupera con una mascara multiplicativa $l_j$:

$$
m_i = \sum_j l_j \odot A x_{ij}, \qquad l_{kj} = (1 - j/J) - \frac{k}{d}(1 - 2j/J)
$$

con $J$ el numero de palabras de la oracion y $d$ la dimension del embedding; $\odot$ es producto elemento a elemento. Para tareas con noción de tiempo se suma un **temporal encoding** aprendido: $m_i = \sum_j A x_{ij} + T_A(i)$, donde $T_A(i)$ es la fila $i$ de una matriz especial (analogamente $T_C$ en la salida). Las oraciones se indexan en orden inverso, de modo que su indice refleja la distancia a la pregunta.

### II.5 La equivalencia con la self-attention

Aqui esta el puente conceptual de mayor alcance del modulo. Comparemos un hop de MemN2N con el *scaled dot-product attention* del Transformer:

$$
\text{Attention}(Q,K,V) = \mathrm{Softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

El mapeo es exacto, termino a termino:

| MemN2N | Self-attention | Rol |
|---|---|---|
| $u = Bq$ | $Q$ | query |
| $m_i = A x_i$ | $K$ (keys) | con que se casa |
| $c_i = C x_i$ | $V$ (values) | que se devuelve |
| $p_i = \mathrm{Softmax}(u^\top m_i)$ | $\mathrm{Softmax}(QK^\top/\sqrt{d_k})$ | pesos |
| $o = \sum_i p_i c_i$ | suma ponderada de $V$ | salida |

{{< concept-alert type="recordar" >}}
La separacion **input memory $A$** (con que se casa) frente a **output memory $C$** (que se devuelve) prefigura literalmente la distincion **key / value**. Y apilar $K$ hops de MemN2N es el antecedente de apilar capas de self-attention. La self-attention de un Transformer puede leerse como multiples hops de atencion suave donde la "memoria" son las representaciones de todos los tokens de la secuencia. MemN2N (2015) formalizo el patron dos años antes que *Attention Is All You Need* (2017). La unica pieza nueva del Transformer es el factor $1/\sqrt{d_k}$, que estabiliza los gradientes cuando $d_k$ es grande.
{{< /concept-alert >}}

---

## Parte III — Key-Value Memory Networks

### III.1 Separar la representacion de busqueda de la de retorno

La [Key-Value MemNN](/papers/key-value-memnn-miller-2016) parte de una observacion practica: *la representacion con la que buscas no tiene por que ser la representacion que devuelves*. Cada slot de memoria se desdobla en un par $(k_i, v_i)$ — una **clave** que se usa para direccionar y un **valor** que se usa para leer. El acceso pasa por *feature maps* diseñables $\Phi_K$, $\Phi_V$, $\Phi_X$ que extraen rasgos antes del embedding.

### III.2 Key addressing y value reading

El **direccionamiento por clave** compara la pregunta con cada clave, ambas embebidas por una matriz $A \in \mathbb{R}^{d\times D}$:

$$
p_i = \mathrm{Softmax}\big(A\Phi_X(x)^\top \, A\Phi_K(k_i)\big)
$$

La **lectura del valor** devuelve la suma ponderada de los valores embebidos:

$$
o = \sum_i p_i\, A\Phi_V(v_i)
$$

Es el mismo mecanismo de la Parte I, pero con la asimetria clave/valor explicita: los pesos $p_i$ se calculan sobre $\Phi_K(k_i)$ y la salida $o$ se construye sobre $\Phi_V(v_i)$.

### III.3 Actualizacion del query entre hops

Tras la primera lectura, el query se **actualiza** incorporando la evidencia recuperada, antes del siguiente hop:

$$
\boxed{\; q_{j+1} = R_j\,(q_j + o) \;}
$$

con $R_j \in \mathbb{R}^{d\times d}$ una matriz distinta por hop. El direccionamiento del hop siguiente usa el query actualizado:

$$
p_i = \mathrm{Softmax}\big(q_{j+1}^\top A\Phi_K(k_i)\big)
$$

Tras $H$ hops, la prediccion final escoge entre los candidatos $y_i$:

$$
\hat{a} = \arg\max_{i=1,\dots,C}\, \mathrm{Softmax}\big(q_{H+1}^\top B\Phi_Y(y_i)\big)
$$

donde $B$ puede atarse a $A$. Se entrena de punta a punta minimizando entropia cruzada, aprendiendo $A$, $B$ y $R_1,\dots,R_H$.

### III.4 Por que separar clave de valor da flexibilidad

La asimetria permite *codificar conocimiento previo* que un slot unico no puede expresar. El ejemplo canonico es la representacion **window-level**: la clave es una ventana entera de $W$ palabras — mas probable de casar con la pregunta — mientras que el valor es solo la **palabra central** de la ventana — la entidad, mas probable de ser la respuesta:

$$
k_i = \Phi_K(\text{ventana de } W \text{ palabras}), \qquad v_i = \Phi_V(\text{palabra central})
$$

En una MemN2N, donde clave y valor son el mismo vector, esa asimetria es imposible. Formalmente, **KV-MemNN generaliza estrictamente a MemN2N**: fijando $k_i = v_i$ para todo $i$ y $\Phi_K = \Phi_V = \text{identidad}$, se recupera exactamente la arquitectura de Sukhbaatar et al. Esta idea — desacoplar la representacion de busqueda de la de retorno — es la misma que hoy estructura los almacenes vectoriales y los recuperadores densos de RAG: se indexa por un embedding y se devuelve el pasaje original.

---

## Parte IV — Recurrent Entity Networks (EntNet)

### IV.1 Una memoria estructurada por entidades

El [Recurrent Entity Network](/papers/entity-networks-henaff-2017) cambia la filosofia: en vez de almacenar el texto crudo y razonar al final, mantiene un **estado del mundo** que se actualiza frase a frase, *online* y en una sola pasada. La memoria se divide en $m$ **bloques** $h_1, \dots, h_m$, cada uno con una **clave** asociada $w_j$. Idealmente cada bloque rastrea una entidad (una persona, un objeto, un lugar). Es una bateria de RNN con compuertas que **comparten parametros** — el *weight tying* refleja que las leyes del mundo son las mismas para todas las entidades.

El input se codifica con una mascara multiplicativa aprendida: $s_t = \sum_i f_i \odot e_i$, donde los $\{f_i\}$ son vectores de mascara compartidos en cada paso (degeneran en bag-of-words si todos valen 1).

### IV.2 La celda por bloque: gate, candidato, update

Cada bloque $j$ se actualiza con cuatro ecuaciones por paso temporal $t$:

$$
\textbf{(1) Gate:}\quad g_j = \sigma\big(s_t^\top h_j + s_t^\top w_j\big)
$$
$$
\textbf{(2) Candidato:}\quad \tilde{h}_j = \phi\big(U h_j + V w_j + W s_t\big)
$$
$$
\textbf{(3) Update:}\quad h_j \leftarrow h_j + g_j \odot \tilde{h}_j
$$
$$
\textbf{(4) Normalizacion:}\quad h_j \leftarrow \frac{h_j}{\lVert h_j\rVert}
$$

La pieza mas instructiva es el **gate**, con sus dos terminos:

- $s_t^\top h_j$ — termino de **contenido**: abre la compuerta para bloques cuyo *contenido actual* coincide con la entrada (direccionamiento por contenido).
- $s_t^\top w_j$ — termino de **ubicacion/clave**: abre la compuerta para bloques cuya *clave* coincide con la entrada (direccionamiento por clave).

La combinacion permite encontrar la entidad correcta tanto por su nombre (clave) como por lo que ya sabemos de ella (contenido). En "Mary went to the garden" se reactiva el bloque de "ball" — aunque la palabra "ball" no aparece — porque la informacion de Mary quedo grabada en su contenido en el paso anterior, y el termino $s_t^\top h_j$ lo detecta.

Las matrices $U, V, W$ son compartidas entre todos los bloques (de ahi el weight tying); $\phi$ es una PReLU o la identidad. Si $g_j \approx 0$ el bloque queda intacto: solo las entidades relevantes se modifican, y **varios bloques pueden actualizarse en paralelo** porque cada gate es independiente — no hay softmax que los acople, a diferencia de la NTM.

### IV.3 Por que la normalizacion implementa olvido

La ecuacion (4) proyecta cada memoria a la **esfera unitaria** $\lVert h_j\rVert = 1$. La consecuencia es sutil pero central: si toda la informacion vive en la *direccion* (fase) del vector, sumar cualquier vector $\tilde{h}_j$ que no sea colineal con $h_j$ **rota** $h_j$ y reduce su similitud coseno con el estado anterior.

Sea $h' = h + \delta$ con $\lVert h\rVert = 1$ y luego renormalizamos a $\hat{h}' = h'/\lVert h'\rVert$. La similitud coseno entre el estado nuevo y el viejo es

$$
\cos(\hat{h}', h) = \frac{h^\top(h+\delta)}{\lVert h+\delta\rVert} = \frac{1 + h^\top\delta}{\lVert h+\delta\rVert}
$$

Salvo que $\delta$ apunte exactamente en la direccion de $h$, esta cantidad es $< 1$: cada update *aleja* la memoria de su valor previo. Asi, **a medida que se agrega informacion nueva, la antigua se olvida gradualmente** — sin necesidad de una compuerta de olvido explicita como la de la LSTM. La esfera tiene capacidad acotada (todo vive en la fase), y escribir cosas nuevas necesariamente desplaza lo viejo. Es una forma elegante y barata de evitar que la memoria sature.

### IV.4 La capa de salida

Para responder se aplica una atencion de un hop sobre los bloques:

$$
p_j = \mathrm{Softmax}(q^\top h_j), \qquad u = \sum_j p_j h_j, \qquad y = R\,\phi(q + H u)
$$

que es, literalmente, una MemN2N de un solo hop con una no-linealidad $\phi$ añadida. El EntNet fue el primer modelo en resolver las 20 tareas bAbI (error medio 0.5 % en el regimen de 10k), porque convierte el razonamiento secuencial costoso (en el momento de la pregunta) en una simple lectura sobre un estado del mundo ya mantenido.

---

## Parte V — Neural Turing Machine (NTM)

### V.1 Controlador, memoria y cabezas

La [Neural Turing Machine](/papers/ntm-graves-2014) es el modelo seminal de la memoria externa diferenciable. Acopla un **controlador** (LSTM o feedforward) a una **matriz de memoria** $M_t \in \mathbb{R}^{N\times M}$ ($N$ ubicaciones de ancho $M$), accedida por **cabezas** de lectura y escritura. La analogia es explicita: el controlador es la CPU, la matriz es la RAM, las activaciones ocultas son los registros. La clave del invento es que el acceso es *borroso* (blurry): cada cabeza interactua con *todas* las ubicaciones, ponderadas por un vector $w_t$ normalizado ($\sum_i w_t(i) = 1$, $w_t(i)\in[0,1]$). Ese foco continuo es lo que vuelve toda la maquina diferenciable.

### V.2 Lectura

La lectura es la combinacion convexa de las filas de la memoria:

$$
r_t = \sum_{i=1}^{N} w_t(i)\, M_t(i)
$$

Identica al mecanismo de la Parte I, ahora sobre una memoria persistente y editable.

### V.3 Escritura: borrar y añadir (estilo compuertas LSTM)

Inspirada en las compuertas *forget* e *input* de la LSTM, cada escritura se descompone en dos pasos. Primero un **borrado** con un vector $e_t \in (0,1)^M$:

$$
\tilde{M}_t(i) = M_{t-1}(i)\,\big[\,\mathbf{1} - w_t(i)\, e_t\,\big]
$$

Un elemento se pone a cero solo si *tanto* el peso de la ubicacion *como* el elemento de borrado valen 1; si cualquiera es 0, queda intacto. Luego una **adicion** con un vector $a_t \in \mathbb{R}^M$:

$$
M_t(i) = \tilde{M}_t(i) + w_t(i)\, a_t
$$

Como $e_t$ y $a_t$ tienen $M$ componentes independientes, la red tiene control fino sobre *que elementos* de cada ubicacion modifica. Borrar es la compuerta de olvido; añadir, la de entrada — pero operando sobre una RAM externa direccionable en vez del estado de celda interno.

### V.4 Direccionamiento: contenido + ubicacion

El vector de pesos $w_t$ surge de combinar **dos mecanismos**. El **direccionamiento por contenido** ya lo vimos en la Parte I.3: la cabeza emite una clave $k_t$ y una fuerza $\beta_t$, y

$$
w^c_t(i) = \mathrm{Softmax}\big(\beta_t\, K[k_t, M_t(i)]\big)
$$

con $K$ la similitud coseno. El **direccionamiento por ubicacion** habilita iteracion y saltos, en tres pasos encadenados:

1. **Interpolacion.** Una compuerta escalar $g_t \in (0,1)$ mezcla el peso por contenido actual con el peso del paso anterior:
$$
w^g_t = g_t\, w^c_t + (1 - g_t)\, w_{t-1}
$$
Con $g_t = 0$ se ignora el contenido y se reusa el peso previo; con $g_t = 1$ se usa solo el contenido.

2. **Shift (convolucion circular).** La cabeza emite una distribucion $s_t$ sobre desplazamientos enteros permitidos (p. ej. $\{-1,0,+1\}$), y se rota el peso:
$$
\tilde{w}_t(i) = \sum_{j=0}^{N-1} w^g_t(j)\, s_t(i - j) \pmod N
$$
Esto permite avanzar a una ubicacion *adyacente* (iterar por un array).

3. **Sharpening.** La convolucion difumina; para reafilar, la cabeza emite $\gamma_t \ge 1$ y renormaliza:
$$
w_t(i) = \frac{\tilde{w}_t(i)^{\gamma_t}}{\sum_j \tilde{w}_t(j)^{\gamma_t}}
$$

### V.5 Por que toda la maquina es diferenciable

La pregunta central: ¿por que esto se puede entrenar por gradiente si la metafora es un computador discreto? Porque **cada operacion reemplazo una eleccion discreta por una mezcla continua**, y toda mezcla continua tiene jacobiano. Recapitulando:

| Operacion discreta (computador) | Relajacion continua (NTM) | Por que es diferenciable |
|---|---|---|
| leer una celda $M(i^\star)$ | $r_t = \sum_i w_t(i) M_t(i)$ | combinacion convexa, lineal en $w$ y $M$ |
| escribir/borrar una celda | $M_t = M_{t-1}(1 - w e) + w a$ | producto y suma de tensores |
| seleccionar por igualdad exacta | softmax de similitud coseno con $\beta$ | softmax es $C^\infty$ |
| saltar a la direccion $i \pm k$ | convolucion circular con $s_t$ | suma ponderada de shifts |
| decidir entre contenido y ubicacion | interpolacion con compuerta $g_t$ | combinacion convexa |

Ningun $\arg\max$, ningun indice entero, ninguna ramificacion dura sobrevive en el camino forward: todo es producto, suma, softmax y sigmoide. El gradiente fluye desde la perdida, por la lectura, por los pesos $w_t$, hasta los vectores de interfaz ($k_t, \beta_t, g_t, s_t, \gamma_t, e_t, a_t$) que emite el controlador, y de ahi a sus parametros. Notablemente, el numero de parametros **no crece con $N$** (el tamaño de la memoria), a diferencia de una LSTM cuyos parametros crecen cuadraticamente con las unidades ocultas — por eso la NTM aprende algoritmos que *generalizan a secuencias mas largas* que las de entrenamiento.

---

## Parte VI — Differentiable Neural Computer (DNC)

### VI.1 Los tres defectos del NTM y sus parches

El [Differentiable Neural Computer](/papers/dnc-graves-2016) es el sucesor directo del NTM. Mantiene el esqueleto (controlador + matriz $M \in \mathbb{R}^{N\times W}$ + cabezas) y la atencion por contenido de la Parte V.4, pero corrige tres defectos concretos del NTM con dos mecanismos nuevos:

1. El NTM no previene que bloques de memoria se solapen → **asignacion dinamica** entrega ubicaciones libres una a una.
2. El NTM no puede liberar memoria → las *free gates* la desasignan.
3. El NTM pierde el orden temporal cuando la cabeza salta → la **matriz de enlaces temporales** lo registra explicitamente.

### VI.2 Asignacion dinamica de memoria

Se mantiene un **vector de uso** $u_t \in [0,1]^N$. Cada lectura puede decrementar el uso de las posiciones leidas (via *free gates* $f^i_t$), y cada escritura lo incrementa:

$$
\psi_t = \prod_{i=1}^{R}\big(1 - f^i_t\, w^{r,i}_{t-1}\big), \qquad
u_t = \big(u_{t-1} + w^w_{t-1} - u_{t-1}\odot w^w_{t-1}\big)\odot \psi_t
$$

El factor $\psi_t$ (*retention vector*) preserva las posiciones que las cabezas de lectura *no* quieren liberar. A partir del uso se ordena la **free list** $\phi_t$ (indices en orden ascendente de uso) y se construye la **ponderacion de asignacion** $a_t$, que privilegia la posicion menos usada:

$$
a_t[\phi_t[j]] = \big(1 - u_t[\phi_t[j]]\big)\prod_{i=1}^{j-1} u_t[\phi_t[i]]
$$

El producto acumulado garantiza que casi toda la masa va a la primera posicion libre. El mecanismo es **independiente del tamaño y del contenido** de la memoria: un DNC entrenado con memoria pequeña se escala a una mas grande sin reentrenar. (El ordenamiento introduce discontinuidades en el gradiente, que los autores simplemente ignoran sin perjuicio observable.)

### VI.3 La matriz de enlaces temporales

Para recordar *el orden* en que se escribio, el DNC mantiene una matriz $L_t \in [0,1]^{N\times N}$ donde $L_t[i,j]\to 1$ si $i$ se escribio inmediatamente despues de $j$. Con un **vector de precedencia** $p_t$ (cuanto fue cada posicion la ultima escrita):

$$
L_t[i,j] = \big(1 - w^w_t[i] - w^w_t[j]\big)\, L_{t-1}[i,j] + w^w_t[i]\, p_{t-1}[j], \qquad L_t[i,i] = 0
$$

Cada escritura borra los enlaces viejos a/desde esa posicion y añade enlaces desde la ultima escrita. La potencia esta en como se usa: para cualquier ponderacion de lectura, $f^i_t = L_t\, w^{r,i}_{t-1}$ **desplaza el foco hacia adelante** (a lo escrito despues) y $b^i_t = L_t^\top\, w^{r,i}_{t-1}$ **hacia atras**. Esto permite recuperar una secuencia en el orden en que se escribio *aunque las escrituras no hayan caido en posiciones contiguas* — exactamente lo que el NTM no podia.

### VI.4 Modos de lectura

Cada cabeza de lectura recibe un **read mode** $\pi^i_t \in S_3$ (distribucion sobre 3 opciones) que interpola entre las tres ponderaciones:

$$
w^{r,i}_t = \pi^i_t[1]\, b^i_t + \pi^i_t[2]\, c^{r,i}_t + \pi^i_t[3]\, f^i_t
$$

Si domina $\pi^i_t[2]$, la lectura es por contenido (clave); si domina $\pi^i_t[3]$, itera hacia adelante en orden de escritura; si domina $\pi^i_t[1]$, hacia atras. La cabeza de escritura interpola entre contenido y asignacion via la *allocation gate* $g^a_t$ y la *write gate* $g^w_t$:

$$
w^w_t = g^w_t\big[\,g^a_t\, a_t + (1 - g^a_t)\, c^w_t\,\big]
$$

Con $g^w_t = 0$ no se escribe nada (protege la memoria). Estos tres modos — contenido para estructuras asociativas, enlaces temporales para recuperacion secuencial, asignacion para posiciones libres — son lo que le permite al DNC aprender razonamiento sobre grafos (caminos en el metro de Londres, arboles genealogicos) que el NTM no alcanzaba.

{{< concept-alert type="recordar" >}}
El arco NTM (2014) → DNC (2016) llevo la metafora "red neuronal como computador con RAM" tan lejos como pudo de forma completamente diferenciable. Resulto costoso y dificil de entrenar a escala, y la comunidad pivoto hacia los **Transformers** (2017), cuya atencion sobre toda la secuencia ofrecio una "memoria" de solo-lectura mas simple y paralelizable. Pero las ideas — direccionamiento por contenido, *key-value retrieval*, atencion como mecanismo de acceso — resuenan directamente en los sistemas modernos de *retrieval-augmented generation* y en las memorias de agentes.
{{< /concept-alert >}}

---

## Sintesis: un mecanismo, cinco variantes

| Modelo | Direccionamiento | Escritura | Aporte distintivo |
|---|---|---|---|
| **MemN2N** | contenido (softmax) | no persistente (deposita hechos) | hops apilados, end-to-end sin supervision; ancestro de la self-attention |
| **KV-MemNN** | contenido sobre clave | no persistente | clave $\neq$ valor: desacopla busqueda de retorno |
| **EntNet** | contenido + clave por bloque | gate independiente por bloque | estado del mundo por entidades; normalizacion = olvido |
| **NTM** | contenido + ubicacion (shift) | borrar + añadir | RAM diferenciable; aprende algoritmos |
| **DNC** | contenido + enlaces temporales | borrar + añadir, con asignacion/liberacion | gestion dinamica de memoria y orden de escritura |

Todos comparten el nucleo de la Parte I — query, scores por producto interno, softmax, suma ponderada de valores — y difieren en *que* se almacena, *como* se escribe y *que* informacion auxiliar (posicion, tiempo, uso) se mantiene. Para el marco transversal del campo (memoria como matriz externa, ponderaciones, direccionamiento por contenido frente a ubicacion, lectura/escritura diferenciable), ver el fundamento [/fundamentos/memory-augmented-networks](/fundamentos/memory-augmented-networks). El puente hacia la atencion moderna esta en [/fundamentos/self-attention](/fundamentos/self-attention). La teoria de la clase completa vive en [/clases/clase-30](/clases/clase-30).
