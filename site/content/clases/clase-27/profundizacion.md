---
title: "Profundizacion - Redes Neuronales de Grafos"
weight: 20
math: true
---

> Esta pagina complementa la [teoria de la clase 27](/clases/clase-27/teoria) con derivaciones formales. Seis bloques: **Parte I** formaliza el *message passing* (estado por nodo, funciones Message/Aggregate/Update) y su forma matricial $H^{(l+1)}=\sigma(\hat{A}H^{(l)}W^{(l)})$. **Parte II** deriva GCN desde las convoluciones espectrales —Laplaciano, Fourier en grafos, Chebyshev, truncamiento de primer orden y el *renormalization trick*—. **Parte III** desarrolla la actualizacion GRU de GGNN y por que reemplaza la iteracion a punto fijo de Scarselli. **Parte IV** formaliza GAT y su equivalencia con la atencion de los Transformers. **Parte V** prueba la expresividad: test 1-WL, la cota, GIN con suma inyectiva y por que suma $>$ promedio $>$ maximo. **Parte VI** trata el *pooling/readout* y el *over-smoothing*.

---

## Parte I — Formalizacion del message passing

### I.1 El estado por nodo y la recurrencia local

Un grafo es $G=(V,E)$ con nodos $v\in V$ y aristas $(u,v)\in E$. Cada nodo lleva un **estado oculto** $h_v^{(l)}\in\mathbb{R}^{d_l}$ en la capa $l$, inicializado con los features de entrada $h_v^{(0)}=x_v$. Una capa de GNN —un **paso de message passing**— actualiza simultaneamente todos los estados combinando, para cada nodo, la informacion que llega de su vecindario $\mathcal{N}(v)=\{u:(u,v)\in E\}$.

El esquema generico (Gilmer et al., 2017) se descompone en tres funciones:

$$
m_v^{(l)} = \underbrace{\text{Aggregate}^{(l)}\Big(\big\{\,\text{Message}^{(l)}(h_v^{(l)}, h_u^{(l)}, e_{uv}) : u\in\mathcal{N}(v)\,\big\}\Big)}_{\text{mensaje agregado del vecindario}}
$$
$$
h_v^{(l+1)} = \text{Update}^{(l)}\big(h_v^{(l)},\, m_v^{(l)}\big)
$$

donde:

- **Message** construye el mensaje que viaja por la arista $(u,v)$, eventualmente condicionado por el feature de arista $e_{uv}$. En GCN es simplemente $h_u^{(l)}$ escalado por la normalizacion de grado; en GGNN es $h_u^{(l)}E_k$ (matriz por tipo de arista $k$); en GAT es $\alpha_{vu}Wh_u^{(l)}$.
- **Aggregate** colapsa el **multiconjunto** de mensajes en un solo vector. Es la pieza critica: debe ser **invariante a permutaciones** (un vecindario no tiene orden), de modo que solo puede ser suma, promedio, maximo, o una combinacion atencional de estos. Su eleccion gobierna la expresividad (Parte V).
- **Update** funde el estado anterior $h_v^{(l)}$ con el mensaje agregado $m_v^{(l)}$: una transformacion lineal mas no-linealidad (GCN), una GRU (GGNN), o concatenacion (GraphSAGE).

Tras $T$ capas, $h_v^{(T)}$ codifica el **subarbol enraizado de profundidad $T$** alrededor de $v$: el estado depende de todos los nodos a distancia $\le T$ saltos. Este es el analogo grafico del *receptive field* de una CNN, y explica por que el numero de capas $T$ controla el radio de informacion que cada nodo "ve".

{{< concept-alert type="clave" >}}
El **campo receptivo** crece un salto por capa. Con $T$ capas, la prediccion de un nodo depende de su vecindario de orden $T$. Esto es a la vez la fortaleza (capturar contexto estructural) y el origen del *over-smoothing* (Parte VI): si $T$ es grande, el vecindario de orden $T$ cubre casi todo el grafo y los estados convergen a un mismo valor.
{{< /concept-alert >}}

### I.2 La forma matricial

Apilar las operaciones por nodo en una matriz de estados $H^{(l)}\in\mathbb{R}^{N\times d_l}$ (fila $v$ = $h_v^{(l)\top}$, con $N=|V|$) permite escribir la propagacion como **producto de matrices**. El caso GCN, que es el esqueleto canonico, toma la forma

$$
H^{(l+1)} = \sigma\!\big(\hat{A}\,H^{(l)}\,W^{(l)}\big),
$$

con $W^{(l)}\in\mathbb{R}^{d_l\times d_{l+1}}$ la matriz de pesos aprendida y $\sigma$ una no-linealidad (ReLU). Lo notable esta en $\hat{A}$: el producto $\hat{A}H^{(l)}$ realiza la **agregacion sobre vecinos en una sola multiplicacion matricial dispersa** —la fila $v$ de $\hat{A}H^{(l)}$ es exactamente $\sum_u \hat{A}_{vu}h_u^{(l)}$, la suma ponderada de los estados de los vecinos—.

### I.3 Por que $\hat{A}$ y no $A$ a secas

Usar la adyacencia cruda $A$ tiene dos defectos. **Primero**, $A H$ promedia los vecinos pero *omite al propio nodo*: el estado de $v$ no entra en su actualizacion. Se corrige con auto-conexiones, $\tilde{A}=A+I$ (cada nodo es vecino de si mismo). **Segundo**, sin normalizar, multiplicar repetidamente por $\tilde{A}$ cambia la escala de los features de forma descontrolada: nodos de alto grado acumulan magnitudes enormes y el espectro de $\tilde{A}$ tiene autovalores $>1$, lo que hace explotar las activaciones al apilar capas. La solucion es **normalizar** por el grado. Dos opciones:

$$
\hat{A}_{\text{rw}}=\tilde{D}^{-1}\tilde{A}\quad\text{(random-walk, promedio simple)}, \qquad
\hat{A}_{\text{sym}}=\tilde{D}^{-1/2}\tilde{A}\,\tilde{D}^{-1/2}\quad\text{(simetrica)},
$$

con $\tilde{D}_{vv}=\sum_u\tilde{A}_{vu}$ la matriz diagonal de grados del grafo aumentado. La version simetrica $\hat{A}_{\text{sym}}$ es la de GCN: el peso de la arista $(u,v)$ es $1/\sqrt{d_u d_v}$, la **media geometrica** de los grados de ambos extremos. La Parte II muestra que esta eleccion no es un truco *ad hoc* sino que **cae de la teoria espectral**.

---

## Parte II — GCN desde las convoluciones espectrales

GCN no fue inventado como "promedio de vecinos"; Kipf y Welling lo derivaron como una **aproximacion de primer orden de las convoluciones espectrales sobre grafos**. Reconstruimos esa derivacion porque es lo que distingue a GCN de un agregador inventado a mano. Ver [GCN (Kipf & Welling, 2017)](/papers/gcn-kipf-2017).

### II.1 El Laplaciano del grafo

Sea $A$ la adyacencia y $D=\text{diag}(d_1,\dots,d_N)$ la matriz de grados. El **Laplaciano no normalizado** es

$$
L = D - A.
$$

Es simetrico y semidefinido positivo: para toda señal $x\in\mathbb{R}^N$ (un escalar por nodo),

$$
x^\top L x = \tfrac12\sum_{(u,v)\in E}(x_u - x_v)^2 \;\ge\; 0.
$$

Esta forma cuadratica mide la **no-suavidad** de $x$ sobre el grafo: vale cero cuando $x$ es constante en cada componente conexa y crece cuanto mas difieren nodos adyacentes. El Laplaciano **normalizado simetrico** es

$$
L_{\text{sym}} = D^{-1/2}LD^{-1/2} = I - D^{-1/2}AD^{-1/2},
$$

con autovalores en $[0,2]$.

### II.2 Fourier en grafos

Como $L_{\text{sym}}$ es real simetrico, admite descomposicion espectral

$$
L_{\text{sym}} = U\Lambda U^\top,\qquad \Lambda=\text{diag}(\lambda_1,\dots,\lambda_N),\quad U=[u_1,\dots,u_N],
$$

con autovectores ortonormales $u_i$ y autovalores $0=\lambda_1\le\dots\le\lambda_N\le 2$. La analogia con Fourier es exacta: los autovectores de $L_{\text{sym}}$ son las **"frecuencias" del grafo** (los de $\lambda$ pequeño varian suavemente sobre el grafo; los de $\lambda$ grande oscilan), y la **transformada de Fourier en grafos** de una señal $x$ es $\hat{x}=U^\top x$, su proyeccion sobre esa base. Un **filtro espectral** $g_\theta=\text{diag}(\theta)$ actua multiplicando cada frecuencia:

$$
g_\theta \star x = U\,g_\theta(\Lambda)\,U^\top x.
$$

El problema es de costo: formar $U$ exige una eigendescomposicion $O(N^3)$, y cada filtrado cuesta $O(N^2)$. Inviable para grafos grandes.

### II.3 Aproximacion de Chebyshev (ChebNet)

Defferrard et al. (2016) evitan la eigendescomposicion aproximando el filtro $g_\theta(\Lambda)$ por un **polinomio truncado en polinomios de Chebyshev** $T_k$ de grado $\le K$:

$$
g_{\theta'}(\Lambda)\approx \sum_{k=0}^{K}\theta'_k\,T_k(\tilde\Lambda),\qquad \tilde\Lambda=\frac{2}{\lambda_{\max}}\Lambda - I,
$$

con la recurrencia $T_0(x)=1$, $T_1(x)=x$, $T_k(x)=2x\,T_{k-1}(x)-T_{k-2}(x)$. Como $U T_k(\tilde\Lambda)U^\top = T_k(\tilde L)$, el filtrado se evalua **directamente sobre el Laplaciano sin formar $U$**:

$$
g_{\theta'}\star x \approx \sum_{k=0}^{K}\theta'_k\,T_k(\tilde L)\,x,\qquad \tilde L = \frac{2}{\lambda_{\max}}L_{\text{sym}} - I.
$$

La gracia: $T_k(\tilde L)$ es un polinomio de grado $K$ en $L_{\text{sym}}$, asi que es **$K$-localizado** —solo mezcla nodos a distancia $\le K$ saltos— y se evalua en $O(K|E|)$, lineal en las aristas.

### II.4 Truncamiento a primer orden y renormalization trick

Kipf y Welling dan el salto: fijan $K=1$ (filtro **lineal** en el Laplaciano) y aproximan $\lambda_{\max}\approx 2$, confiando en que la red se adapte a la escala. Con esto $\tilde L \approx L_{\text{sym}} - I = -D^{-1/2}AD^{-1/2}$, y la convolucion queda con dos parametros:

$$
g_{\theta'}\star x \approx \theta'_0\, x - \theta'_1\, D^{-1/2}AD^{-1/2}\,x.
$$

Para reducir parametros y sobreajuste, atan $\theta=\theta'_0=-\theta'_1$:

$$
g_\theta\star x \approx \theta\big(I + D^{-1/2}AD^{-1/2}\big)x.
$$

Aqui aparece el problema numerico. El operador $I + D^{-1/2}AD^{-1/2}$ tiene autovalores en $[0,2]$; aplicarlo repetidamente (al apilar capas) amplifica las componentes de autovalor $2$ y produce **inestabilidad / explosion de gradientes**. El **renormalization trick** lo reemplaza por una version equivalente pero estable, que reintroduce las auto-conexiones *antes* de normalizar:

$$
\boxed{\;I + D^{-1/2}AD^{-1/2}\;\longrightarrow\;\tilde{D}^{-1/2}\tilde{A}\,\tilde{D}^{-1/2},\qquad \tilde{A}=A+I,\;\;\tilde{D}_{vv}=\textstyle\sum_u\tilde{A}_{vu}.\;}
$$

Generalizando de una señal escalar $x$ a una matriz de features $X\in\mathbb{R}^{N\times C}$ con una matriz de filtros $\Theta\in\mathbb{R}^{C\times F}$, una capa es $Z=\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X\Theta$. Apilando con $\sigma$ se obtiene exactamente la forma matricial de la Parte I, y la entrada $(u,v)$ de $\hat{A}=\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ es $1/\sqrt{\tilde d_u \tilde d_v}$: **la normalizacion por media geometrica de grados no es arbitraria, es la simplificacion espectral**.

| Variante de propagacion | Operador | Cora (ablacion del paper) |
| --- | --- | --- |
| Chebyshev $K=3$ | $\sum_{k=0}^{3}\theta_k T_k(\tilde L)$ | 79.5 |
| Chebyshev $K=2$ | $\sum_{k=0}^{2}\theta_k T_k(\tilde L)$ | 81.2 |
| 1er orden, 2 parametros | $\theta_0 X - \theta_1 D^{-1/2}AD^{-1/2}X$ | 80.0 |
| 1 parametro | $(I+D^{-1/2}AD^{-1/2})X\Theta$ | 79.2 |
| **Renormalization trick** | $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X\Theta$ | **81.5** |

La simplificacion mas agresiva resulta tambien la mejor: menos parametros, mas estabilidad, mayor accuracy.

---

## Parte III — GGNN y la actualizacion GRU

[GGNN (Li et al., 2015)](/papers/ggnn-li-2015) es el primer modelo concreto de la clase, y su rasgo distintivo es usar una **GRU como funcion Update**. Para entender por que, hay que ver primero el modelo que viene a reparar.

### III.1 La iteracion a punto fijo de Scarselli y por que limita

El [GNN original de Scarselli et al. (2009)](/papers/gnn-model-scarselli-2009) no apila un numero fijo de capas: itera la transicion $h_v^{(t)}=F_w(\cdot)$ **hasta un punto fijo** $h^\star=F_w(h^\star)$. Para que ese punto fijo *exista y sea unico*, se invoca el **teorema de punto fijo de Banach**: $F_w$ debe ser una **contraccion**, es decir existe $\mu<1$ tal que

$$
\lVert F_w(h)-F_w(h')\rVert \le \mu\,\lVert h-h'\rVert\quad\forall h,h'.
$$

El aprendizaje usa el algoritmo de **Almeida-Pineda** ("backprop a traves del punto fijo"), que retropropaga solo a traves del estado convergido sin almacenar la trayectoria. Elegante, pero el precio es severo. La contraccion implica que la influencia de un nodo sobre otro a distancia $\delta$ **decae como $\mu^\delta$**: en el caso no lineal, por la regla de la cadena,

$$
\left\lVert\frac{\partial h^{(t)}}{\partial h^{(1)}}\right\rVert \le \mu^{\,t-1}\xrightarrow[t\to\infty]{}0.
$$

La señal se atenua exponencialmente con la distancia en el grafo: **la contraccion impide modelar dependencias de largo alcance**. Ademas, como el punto fijo es independiente de la inicializacion, no se pueden inyectar etiquetas de nodo como entrada.

### III.2 Pasos fijos + GRU: liberar la expresividad

GGNN abandona el punto fijo: ejecuta un numero **fijo** $T$ de pasos y para, entrenando con backpropagation through time (BPTT). Como ya no se busca un punto fijo, **no hace falta que $F_w$ sea contraccion** —se libera la expresividad, al costo de almacenar los $T$ estados intermedios—. La funcion Update pasa a ser una **GRU completa**. En el paso $t$, primero se agrega el mensaje (suma de mensajes por tipo de arista $k$, con matriz $E_k$):

$$
a_v^{(t)} = \sum_{u\in\mathcal{N}(v)} h_u^{(t-1)} E_{k(u,v)} + b,
$$

y luego la GRU decide cuanto del pasado conservar y cuanto del mensaje incorporar:

$$
z_v^{(t)} = \sigma\!\big(W^z a_v^{(t)} + U^z h_v^{(t-1)}\big)\qquad\text{(compuerta de actualizacion)}
$$
$$
r_v^{(t)} = \sigma\!\big(W^r a_v^{(t)} + U^r h_v^{(t-1)}\big)\qquad\text{(compuerta de reset)}
$$
$$
\tilde{h}_v^{(t)} = \tanh\!\big(W a_v^{(t)} + U\,(r_v^{(t)}\odot h_v^{(t-1)})\big)\qquad\text{(estado candidato)}
$$
$$
h_v^{(t)} = (1-z_v^{(t)})\odot h_v^{(t-1)} + z_v^{(t)}\odot \tilde{h}_v^{(t)}\qquad\text{(estado nuevo)}
$$

con $\sigma$ la sigmoide logistica y $\odot$ el producto de Hadamard. La clase condensa esto como $h_t=\text{GRU}(h_{t-1},h')$ con $h'=a_v^{(t)}$.

{{< concept-alert type="recordar" >}}
La compuerta de actualizacion $z_v^{(t)}$ es la clave anti-decaimiento. Cuando $z_v^{(t)}\to 0$, el estado se **copia** ($h_v^{(t)}\approx h_v^{(t-1)}$): la informacion se preserva intacta a traves de muchos pasos, en vez de atenuarse como $\mu^\delta$. Es la misma idea de las *skip connections* que mantiene los gradientes vivos en redes profundas, y lo que permite a GGNN propagar señal a largo alcance donde el GNN contractivo de Scarselli fallaba.
{{< /concept-alert >}}

---

## Parte IV — GAT: atencion en grafos

GCN fija el peso de cada vecino por su grado ($1/\sqrt{d_u d_v}$, una constante estructural). [GAT (Velickovic et al., 2018)](/papers/gat-velickovic-2018) lo **aprende** y lo hace **depender del contenido** de los features.

### IV.1 El coeficiente de atencion

Sea $W\in\mathbb{R}^{F'\times F}$ una proyeccion lineal compartida y $\vec{a}\in\mathbb{R}^{2F'}$ el vector del mecanismo de atencion. El coeficiente sin normalizar entre $i$ y un vecino $j$ es

$$
e_{ij} = \text{LeakyReLU}\big(\vec{a}^{\top}[\,W\vec{h}_i \,\Vert\, W\vec{h}_j\,]\big),
$$

donde $\Vert$ es concatenacion. La **atencion enmascarada** restringe el calculo a $j\in\mathcal{N}_i$ (solo aristas que existen: ahi entra la topologia), y la normalizacion softmax sobre el vecindario da

$$
\alpha_{ij} = \frac{\exp\!\big(\text{LeakyReLU}(\vec{a}^{\top}[W\vec{h}_i\Vert W\vec{h}_j])\big)}{\sum_{k\in\mathcal{N}_i}\exp\!\big(\text{LeakyReLU}(\vec{a}^{\top}[W\vec{h}_i\Vert W\vec{h}_k])\big)}.
$$

La nueva representacion es la combinacion lineal de los vecinos con esos pesos:

$$
\vec{h}'_i = \sigma\!\Big(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\,W\vec{h}_j\Big).
$$

### IV.2 Comparacion con la normalizacion fija de GCN

| | GCN | GAT |
| --- | --- | --- |
| Peso del vecino | $1/\sqrt{d_i d_j}$ (fijo, estructural) | $\alpha_{ij}$ (aprendido, depende del contenido) |
| Requiere | matriz $\hat{A}$ global | solo conocer $\mathcal{N}_i$ |
| Regimen | transductivo | inductivo (generaliza a grafos nuevos) |

El ablation *Const-GAT* del paper —GAT con atencion constante $a(x,y)=1$, que colapsa a un operador tipo GCN inductivo— mide la ganancia pura de aprender los pesos: $+3.9$ puntos micro-F1 en PPI ($0.973$ vs $0.934$). La importancia de un vecino deja de ser un dato estructural y pasa a ser un parametro.

### IV.3 Multi-head

Para estabilizar el aprendizaje se corren $K$ cabezas independientes (cada una con su $\vec{a}^k$, $W^k$) y se **concatenan** en capas intermedias:

$$
\vec{h}'_i = \big\Vert_{k=1}^{K}\;\sigma\!\Big(\sum_{j\in\mathcal{N}_i}\alpha_{ij}^{k}\,W^k\vec{h}_j\Big),
$$

mientras que en la capa final se **promedian** (concatenar cambiaria la dimension de salida): $\vec{h}'_i=\sigma\big(\tfrac{1}{K}\sum_{k}\sum_{j}\alpha_{ij}^{k}W^k\vec{h}_j\big)$.

### IV.4 La conexion: self-attention de Transformers = GAT sobre el grafo completo

La equivalencia es exacta y profunda. Toma una secuencia de tokens y representala como un grafo donde **todos los nodos estan conectados con todos** ($\mathcal{N}_i = V\setminus\{i\}$, un grafo completo o *clique*). Aplicar GAT sobre ese grafo recupera el self-attention del Transformer: el softmax sobre el vecindario es el softmax de la atencion sobre todos los tokens, y la suma ponderada es el value-weighting. Visto al reves:

$$
\underbrace{\text{Transformer self-attention}}_{\text{atencion sobre todos los tokens}} \;=\; \underbrace{\text{GAT}}_{\text{atencion sobre vecinos}}\;\text{con grafo} = \text{clique completo}.
$$

La unica diferencia es la **mascara**: GAT enmascara para atender solo a las aristas que existen, mientras el Transformer (sin mascara causal) atiende a todo. Esta dualidad ancla las GNN y los Transformers en un mismo marco. Ver [Self-Attention](/fundamentos/self-attention).

---

## Parte V — Expresividad: 1-WL, GIN y la jerarquia de agregadores

¿Que grafos puede *distinguir* una GNN de message passing? La respuesta —el resultado teorico central del campo— la dan [GIN (Xu et al., 2019)](/papers/gin-xu-2019) en lenguaje combinatorio y [Barcelo et al. (2020)](/papers/logical-expressiveness-barcelo-2020) en lenguaje logico.

### V.1 El test de Weisfeiler-Lehman (1-WL)

El test 1-WL es un algoritmo clasico de isomorfismo de grafos que **colorea iterativamente** los nodos. Paso a paso:

1. **Inicializacion.** Cada nodo recibe un color inicial $c_v^{(0)}$ (su etiqueta, o una constante si el grafo no esta etiquetado).
2. **Refinamiento.** En cada ronda, el nuevo color combina el color propio con el **multiconjunto** de colores de los vecinos, via una funcion hash *inyectiva*:
$$
c_v^{(t)} = \text{Hash}\Big(c_v^{(t-1)},\;\{\!\!\{\,c_u^{(t-1)} : u\in\mathcal{N}(v)\,\}\!\!\}\Big),
$$
donde $\{\!\!\{\cdot\}\!\!\}$ denota multiconjunto.
3. **Decision.** Tras estabilizar, dos grafos son "posiblemente isomorfos" si tienen el mismo histograma de colores; si difieren, son **no isomorfos** con certeza.

La estructura es **identica** a una capa de GNN: agregar el multiconjunto de vecinos y combinar con el estado propio. La unica diferencia es que WL usa un hash inyectivo perfecto y la GNN usa funciones aprendidas.

### V.2 La cota: las GNN de message passing estan acotadas por 1-WL

**Lema (cota superior).** Si una GNN $\mathcal{A}$ mapea dos grafos $G_1, G_2$ a embeddings distintos, entonces 1-WL tambien los declara no isomorfos. **Consecuencia:** ninguna GNN basada en agregacion de vecinos puede distinguir mas pares de grafos que 1-WL.

La demostracion es por induccion: si dos nodos reciben el mismo color WL en cada ronda, reciben tambien la misma feature GNN, porque las mismas Aggregate/Combine aplicadas a la misma entrada producen la misma salida. 1-WL es un **techo** para toda la familia.

**Teorema (alcanzar la cota).** Una GNN *iguala* el poder de 1-WL si y solo si, con suficientes capas, sus funciones Aggregate sobre el multiconjunto y su Update son **inyectivas** (y el readout de grafo tambien). La intuicion: tras $k$ capas, el estado de un nodo representa un subarbol de altura $k$; si la agregacion captura el multiconjunto completo de vecinos, captura recursivamente esos subarboles y por tanto iguala a WL.

### V.3 GIN: suma inyectiva sobre multiconjuntos

¿Como construir un Aggregate inyectivo? El **Lema de deep multisets**: si el universo de features es contable, existe $f$ tal que $h(X)=\sum_{x\in X}f(x)$ es **unico para cada multiconjunto** $X$, y toda funcion de multiconjunto se descompone como $g(X)=\varphi\big(\sum_{x\in X}f(x)\big)$. La **suma** es inyectiva sobre multiconjuntos; el promedio y el maximo no lo son.

Para incorporar tambien el nodo central preservando inyectividad, el **Corolario** da: para casi todo $\epsilon$ (todos los irracionales), $h(c,X)=(1+\epsilon)f(c)+\sum_{x\in X}f(x)$ es unico para cada par $(c,X)$. De aqui la regla de **GIN**:

$$
\boxed{\;h_v^{(k)} = \text{MLP}^{(k)}\!\Big((1+\epsilon^{(k)})\cdot h_v^{(k-1)} + \sum_{u\in\mathcal{N}(v)} h_u^{(k-1)}\Big)\;}
$$

Cada pieza justificada por la teoria: la **suma** es el agregador inyectivo; el termino $(1+\epsilon)h_v$ mezcla el nodo central sin colision; el **MLP** modela las funciones universales $f,\varphi$ (un MLP de $\ge 2$ capas, porque un perceptron de 1 capa no aproxima funciones de multiconjunto arbitrarias). $\epsilon$ puede aprenderse (GIN-$\epsilon$) o fijarse a $0$ (GIN-0).

### V.4 Por que suma $>$ promedio $>$ maximo, formalmente

Los tres son invariantes a permutaciones, pero capturan aspectos *distintos* del multiconjunto, ordenados por inyectividad:

| Agregador | Que captura | Inyectivo sobre multiconjuntos |
| --- | --- | --- |
| **Suma** $\sum_{u}h_u$ | el multiconjunto completo (elementos *y* multiplicidades) | si |
| **Promedio** $\frac{1}{\lvert\mathcal{N}\rvert}\sum_u h_u$ | la distribucion / proporcion de tipos | no |
| **Maximo** $\max_u h_u$ | solo el conjunto subyacente de elementos distintos | no |

Las pruebas son por contraejemplo:

- **Promedio.** Para $X_1=(S,m)$ y $X_2=(S,k\cdot m)$ (mismo conjunto, multiplicidades escaladas por $k$), el promedio da el **mismo** embedding: $\frac{1}{\lvert X_1\rvert}\sum=\frac{1}{\lvert X_2\rvert}\sum$. No distingue $\{$verde, rojo$\}$ de $\{$verde, verde, rojo, rojo$\}$ porque $\tfrac12(h_g+h_r)=\tfrac14(h_g+h_g+h_r+h_r)$.
- **Maximo.** El max-pooling colapsa el multiconjunto a su **conjunto** de elementos distintos: $\max(h_g,h_r)$ es identico para $\{$verde, rojo$\}$ y $\{$verde, rojo, rojo$\}$. Ignora por completo las multiplicidades.
- **Suma.** Distingue ambos: $2f(a)\ne 3f(a)$ (un nodo con dos vecinos vs uno con tres, en un grafo no etiquetado donde promedio y maximo devuelven siempre $f(a)$ y no capturan *ninguna* estructura).

Esto explica empiricamente por que GCN (promedio) y GraphSAGE (maximo) son estrictamente menos expresivos que GIN, y por que en REDDIT —grafos sin etiquetas de nodo, donde toda la señal es estructural— las GNN de promedio no superan al azar mientras GIN domina.

### V.5 La cara logica: FOC2 y logica modal graduada (Barcelo)

Barcelo et al. miden la expresividad contra la **logica de primer orden con dos variables y cuantificadores de conteo** ($\text{FOC}_2$, que permite $\exists^{\ge N}x$). El resultado clasico de Cai-Furer-Immerman conecta ambos mundos: **dos nodos reciben el mismo color 1-WL si y solo si los clasifican igual todas las formulas $\text{FOC}_2$** —WL y $\text{FOC}_2$ son dos caras de la misma moneda discriminativa—.

El teorema fino de Barcelo: una GNN de aggregate-combine (AC-GNN) captura exactamente la **logica modal graduada** (un fragmento estricto de $\text{FOC}_2$ donde toda subformula esta "guardada" por la arista: no se puede decir "existe algun nodo $y$", solo "existe un *vecino* $y$"). La limitacion es la **localidad**: la AC-GNN solo ve su vecindario. Para capturar todo $\text{FOC}_2$ basta añadir un **readout global** por capa (que lee/suma sobre *todos* los nodos), obteniendo las ACR-GNN —un cómputo no-local que rompe la frontera de la localidad—.

---

## Parte VI — Pooling, readout y over-smoothing

### VI.1 Readout para clasificacion de grafo

Las Partes I-V producen embeddings *por nodo* $h_v^{(T)}$. Para clasificar el **grafo entero** hay que colapsarlos en un solo vector $h_G$ con una funcion de **readout** que, como Aggregate, debe ser **invariante a permutaciones** de los nodos:

$$
h_G = \text{Readout}\big(\{\!\!\{\,h_v^{(T)} : v\in V\,\}\!\!\}\big)\in\Big\{\textstyle\sum_v h_v^{(T)},\;\tfrac{1}{N}\sum_v h_v^{(T)},\;\max_v h_v^{(T)}\Big\}.
$$

La invarianza a permutacion es **obligatoria**: relabelar los nodos de un grafo no debe cambiar la prediccion, y suma/promedio/maximo son simetricas por construccion. Por el mismo argumento de la Parte V, la **suma** es el readout mas expresivo. GIN refina esto con un readout estilo *Jumping Knowledge*: concatena la suma de cada capa para no perder la estructura local de las iteraciones tempranas:

$$
h_G = \text{Concat}\Big(\textstyle\sum_{v}h_v^{(k)} \;\Big|\; k=0,1,\dots,K\Big).
$$

El **pooling jerarquico** (DiffPool, etc.) va mas alla: agrupa nodos en super-nodos por capas, construyendo un grafo cada vez mas pequeño hasta un solo nodo, analogo al pooling espacial de las CNN.

### VI.2 Over-smoothing: por que apilar muchas capas colapsa los embeddings

Una capa GCN multiplica por $\hat{A}=\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$. Ignorando las no-linealidades y los pesos, tras $T$ capas el estado es $\hat{A}^T X$. El analisis espectral revela el problema. $\hat{A}$ tiene autovalores $1=\mu_1\ge\mu_2\ge\dots\ge\mu_N\ge -1$, con autovalor dominante $\mu_1=1$ asociado a un autovector $\propto \tilde{D}^{1/2}\mathbf{1}$ (proporcional a la raiz de los grados). Al elevar a la potencia $T$:

$$
\hat{A}^T = \sum_i \mu_i^T\, u_i u_i^\top \xrightarrow[T\to\infty]{} u_1 u_1^\top,
$$

porque $\lvert\mu_i\rvert<1$ para $i\ge 2$ hace que $\mu_i^T\to 0$ exponencialmente, mientras $\mu_1^T=1$ sobrevive. **Todas las componentes de alta frecuencia (estructura discriminativa) se aniquilan y solo sobrevive la componente constante por grado.** En consecuencia, los embeddings de todos los nodos de una componente conexa convergen a un mismo vector: pierden poder discriminativo. Esto es el **over-smoothing**, y por eso GCN funciona mejor con $2$-$3$ capas, degradandose a partir de $\sim 7$.

{{< concept-alert type="recordar" >}}
El over-smoothing es el reverso oscuro del campo receptivo de la Parte I. Mas capas = mas contexto, pero tambien mas promediado. La velocidad del colapso la fija el **gap espectral** $1-\lvert\mu_2\rvert$: cuanto mayor el segundo autovalor, mas lento el over-smoothing. Las mitigaciones —conexiones residuales $H^{(l+1)}=\sigma(\hat{A}H^{(l)}W^{(l)})+H^{(l)}$, *jumping knowledge*, las compuertas GRU de GGNN (Parte III)— preservan informacion de capas tempranas y frenan la convergencia al autovector dominante.
{{< /concept-alert >}}

---

## Sintesis matematica

| Concepto | Ecuacion central |
| --- | --- |
| Message passing (forma matricial) | $H^{(l+1)}=\sigma(\hat{A}H^{(l)}W^{(l)})$ |
| Renormalization trick (GCN) | $\hat{A}=\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2},\;\tilde{A}=A+I$ |
| Convolucion espectral | $g_\theta\star x = U g_\theta(\Lambda)U^\top x$ |
| Update GGNN | $h_v^{(t)}=(1-z)\odot h_v^{(t-1)}+z\odot\tilde{h}_v^{(t)}$ (GRU) |
| Atencion GAT | $\alpha_{ij}=\text{softmax}_j(\text{LeakyReLU}(\vec{a}^\top[W\vec{h}_i\Vert W\vec{h}_j]))$ |
| GIN (suma inyectiva) | $h_v^{(k)}=\text{MLP}((1+\epsilon)h_v^{(k-1)}+\sum_u h_u^{(k-1)})$ |
| Over-smoothing | $\hat{A}^T\to u_1 u_1^\top$ cuando $T\to\infty$ |

El hilo conductor: **agregar un multiconjunto de vecinos** (invariante a permutacion) es la operacion atomica, y *como* se agrega lo decide todo —su normalizacion sale de la teoria espectral (GCN), su update puede tener compuertas (GGNN) o atencion aprendida (GAT, = Transformer sobre clique), y su inyectividad fija el poder expresivo (GIN $=$ 1-WL $=$ FOC2). Apilar muchas capas extiende el contexto pero arrastra al over-smoothing.

---

**Ver tambien:** [Teoria de la clase 27](/clases/clase-27/teoria) · [Practica desde 0](/clases/clase-27/practica) · Fundamentos: [Redes Neuronales de Grafos](/fundamentos/redes-neuronales-de-grafos) · [Message Passing](/fundamentos/message-passing) · [Expresividad de GNN](/fundamentos/expresividad-gnn) · Papers: [GCN](/papers/gcn-kipf-2017) · [GGNN](/papers/ggnn-li-2015) · [GAT](/papers/gat-velickovic-2018) · [GIN](/papers/gin-xu-2019) · [GNN Model (Scarselli)](/papers/gnn-model-scarselli-2009) · [Logical Expressiveness (Barcelo)](/papers/logical-expressiveness-barcelo-2020).
