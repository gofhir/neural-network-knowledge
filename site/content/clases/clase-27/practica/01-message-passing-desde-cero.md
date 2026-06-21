---
title: "Message Passing desde cero"
weight: 1
math: true
---

Una Red Neuronal de Grafos no es, en su núcleo, ningún misterio nuevo: es una multiplicación de matrices repetida unas pocas veces. La [teoría de la clase](/clases/clase-27/teoria) y el [fundamento de message passing](/fundamentos/message-passing) muestran que toda la familia de GNN —GCN, GraphSAGE, GAT, R-GCN— comparte una misma maquinaria de **paso de mensajes**, y que esa maquinaria, cuando se escribe sobre la matriz de adyacencia densa, colapsa a la operación

$$
H' = \hat{A}\,H\,W.
$$

En esta página la construimos **desde cero**: sin `torch_geometric`, sin DGL, sin ninguna librería de grafos. Solo arrays densos y álgebra lineal. Usamos el grafo de juguete de 7 nodos $A$–$G$ de la clase, lo que nos permite imprimir cada matriz completa y *verla* propagar la información paso a paso. Lo escribimos tres veces —PyTorch, TensorFlow y JAX— para que el mismo cómputo quede expuesto en los tres dialectos del deep learning numérico.

El objetivo no es eficiencia (los grafos reales se almacenan dispersos, ver el [fundamento](/fundamentos/message-passing)), sino *transparencia*: con la adyacencia densa cada etapa del message passing es una línea de álgebra que se puede inspeccionar.

---

## 1. Las cuatro etapas, y por qué colapsan en `Â · H · W`

El [fundamento de message passing](/fundamentos/message-passing) descompone una capa de GNN en cuatro etapas. Conviene tenerlas a la vista antes de escribir código, porque cada una corresponde a un trozo de la fórmula matricial:

1. **Mensaje.** Cada nodo $w$ prepara un mensaje a partir de su estado actual $h_w$. En la versión más simple (GCN) el mensaje *es* el propio estado: $m_w = h_w$.
2. **Traspaso (transmisión).** Los mensajes viajan por las aristas hacia los nodos vecinos. Aquí entra la topología: quién recibe de quién lo dicta la matriz de adyacencia.
3. **Agregación conmutativa.** Cada nodo combina los mensajes que recibe con una operación **que no depende del orden** —típicamente una **suma**—. Esto es esencial: el conjunto de vecinos no tiene orden canónico, así que la agregación debe ser invariante a permutaciones.
4. **Update (actualización).** El nodo actualiza su estado mezclando lo agregado con una transformación lineal aprendida $W$ y una no-linealidad $\sigma$.

La observación central —y la razón por la que una capa de GNN cabe en una línea— es que las etapas 2 y 3 son *exactamente* lo que hace el producto matriz-por-matriz. Si apilamos los estados de los $n$ nodos en una matriz $H \in \mathbb{R}^{n\times d}$ (una fila por nodo) y $A$ es la adyacencia, entonces

$$
(A H)_v = \sum_{w} A_{vw}\, h_w = \sum_{w\in\mathcal{N}(v)} h_w,
$$

es decir, $AH$ **envía a cada nodo la suma de los estados de sus vecinos**, para todos los nodos a la vez. Montando encima la transformación aprendida $W$ y la activación $\sigma$ obtenemos la capa completa:

$$
H' = \sigma\big(\hat{A}\,H\,W\big).
$$

La matriz $\hat{A}$ es la adyacencia **normalizada con auto-conexiones**, que estabiliza la propagación (sección 3). Remitimos al [fundamento de message passing](/fundamentos/message-passing) para la derivación completa de la equivalencia entre las cuatro etapas locales y esta forma matricial.

---

## 2. El grafo de juguete: los 7 nodos A–G

Trabajamos con el grafo de la clase, $V=\{A,B,C,D,E,F,G\}$. Para que las matrices sean simétricas y la normalización quede limpia y verificable, lo tratamos como **no dirigido**: cada relación se cuenta en ambos sentidos (recordemos que un grafo no dirigido equivale a uno dirigido con dos aristas por relación, ver la [teoría](/clases/clase-27/teoria)). Las aristas:

$$
E = \{A\!-\!C,\; C\!-\!F,\; F\!-\!G,\; G\!-\!E,\; E\!-\!A,\; E\!-\!B,\; B\!-\!A,\; D\!-\!C\}.
$$

Indexamos los nodos en orden $A,B,C,D,E,F,G \to 0,1,2,3,4,5,6$. La matriz de adyacencia simétrica $A_{ij}=1 \iff$ hay arista entre $i$ y $j$ queda:

| | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| **A** | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| **B** | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| **C** | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| **D** | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| **E** | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| **F** | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **G** | 0 | 0 | 0 | 0 | 1 | 1 | 0 |

{{< concept-alert type="recordar" >}}
La [teoría de la clase](/clases/clase-27/teoria#14-matriz-de-adyacencia) usa la convención **dirigida** $A_{ij}=1 \iff$ existe la arista de $j$ hacia $i$ (las columnas son orígenes, las filas destinos), donde el nodo $D$ es *OUT-only* y nunca recibe mensajes. Aquí, al volver el grafo no dirigido, la adyacencia es **simétrica** ($A = A^\top$) y todos los nodos reciben y emiten. Es la versión natural para una capa GCN sobre un grafo de juguete, y hace que la normalización simétrica $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ esté bien definida.
{{< /concept-alert >}}

### Self-loops: por qué $\tilde{A} = A + I$

Antes de propagar añadimos **auto-conexiones**: cada nodo se vuelve vecino de sí mismo, $\tilde{A} = A + I$. Sin esto, al hacer $AH$ un nodo recibe los estados de sus vecinos pero **olvida el suyo propio**, lo que descarta información en cada capa. El self-loop conserva el estado del nodo dentro de la agregación.

### Normalización por grado

La suma cruda $\tilde{A}H$ tiene un problema: los nodos de alto grado acumulan magnitudes grandes y los de bajo grado se desvanecen. Normalizamos por el grado. Hay dos opciones estándar (ambas en el [fundamento](/fundamentos/message-passing)):

- **Random-walk (por filas):** $\hat{A} = \tilde{D}^{-1}\tilde{A}$, que **promedia** los vecinos. Cada fila suma 1.
- **Simétrica (GCN):** $\hat{A} = \tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}$, que pondera cada mensaje por $1/\sqrt{d_v d_w}$.

donde $\tilde{D}$ es la matriz diagonal de grados de $\tilde{A}$ (incluyendo el self-loop). Usaremos la **simétrica**, la regla de [Kipf y Welling (GCN)](/papers/gcn-kipf-2017), porque es la canónica; dejamos la random-walk comentada para que se pueda comparar.

---

## 3. Implementación

El plan es idéntico en los tres frameworks:

1. Construir $A$ (densa, simétrica) y añadir self-loops: $\tilde{A} = A + I$.
2. Calcular grados $\tilde{d}_i = \sum_j \tilde{A}_{ij}$ y la normalización simétrica $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$.
3. Inicializar features $H^{(0)} \in \mathbb{R}^{7\times d}$. Para *ver* la propagación usamos features one-hot ($d=7$): cada nodo arranca con un "color" único, y observamos cómo se mezclan.
4. Aplicar 2–3 capas $H' = \sigma(\hat{A}\,H\,W)$ y observar cómo crece el receptive field.

Las dimensiones, fijas en todos los frameworks: $\hat{A}$ es $7\times 7$, $H$ es $7\times d_{\text{in}}$, $W$ es $d_{\text{in}}\times d_{\text{out}}$. El producto $\hat{A}H$ da $7\times d_{\text{in}}$, y $(\hat{A}H)W$ da $7\times d_{\text{out}}$. Cuadra.

### PyTorch

```python
import torch
import torch.nn as nn

# --- 1. Grafo: lista de aristas no dirigidas (índices 0..6 = A..G) ---
#  A=0 B=1 C=2 D=3 E=4 F=5 G=6
edges = [(0, 2), (2, 5), (5, 6), (6, 4), (4, 0), (4, 1), (1, 0), (3, 2)]
N = 7

A = torch.zeros(N, N)
for i, j in edges:
    A[i, j] = 1.0
    A[j, i] = 1.0          # no dirigido => simétrica

# --- 2. Self-loops + normalización simétrica  Â = D̃^{-1/2} (A+I) D̃^{-1/2} ---
def normalize_adjacency(A):
    A_tilde = A + torch.eye(A.shape[0])          # añadir self-loops: Ã = A + I
    deg = A_tilde.sum(dim=1)                      # grado de cada nodo (fila)
    d_inv_sqrt = torch.pow(deg, -0.5)            # D̃^{-1/2} (vector diagonal)
    # Â_ij = d_inv_sqrt[i] * Ã_ij * d_inv_sqrt[j]  (escalado fila y columna)
    A_hat = d_inv_sqrt.unsqueeze(1) * A_tilde * d_inv_sqrt.unsqueeze(0)
    return A_hat                                  # (N, N), simétrica

A_hat = normalize_adjacency(A)
print("filas de Â suman ~", A_hat.sum(dim=1))    # ~1 con random-walk; <=1 con simétrica

# --- 3. Capa de message passing: H' = σ(Â · H · W) ---
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)   # la W aprendible

    def forward(self, A_hat, H):
        # Â (N,N) @ H (N,in) -> (N,in) : envía+suma mensajes de vecinos
        agg = A_hat @ H
        # @ W : transformación lineal aprendida -> (N,out)
        return self.W(agg)

# --- 4. Apilar capas y observar la propagación ---
torch.manual_seed(0)
H0 = torch.eye(N)                                # features one-hot: un "color" por nodo (d=7)

layer1 = GraphConv(N, 8)
layer2 = GraphConv(8, 8)
layer3 = GraphConv(8, 4)

H1 = torch.relu(layer1(A_hat, H0))               # receptive field: vecinos a 1 salto
H2 = torch.relu(layer2(A_hat, H1))               # a 2 saltos
H3 = layer3(A_hat, H2)                           # a 3 saltos (sin σ en la última)

print("H0", H0.shape, "-> H1", H1.shape, "-> H2", H2.shape, "-> H3", H3.shape)
# H0 (7,7) -> H1 (7,8) -> H2 (7,8) -> H3 (7,4)
```

Para *ver* la propagación sin pesos aprendidos —solo el operador $\hat{A}$ actuando sobre las features— basta con propagar las one-hot sin la $W$:

```python
# Propagación pura (sin W ni σ): cuántos saltos alcanza cada nodo
X = torch.eye(N)
for t in range(1, 4):
    X = A_hat @ X
    # la fila i muestra la "masa" que el nodo i ha recibido de cada nodo del grafo
    reached = (X.abs() > 1e-6).sum(dim=1)        # nº de nodos que ya influyen en i
    print(f"t={t}: nodos en el receptive field por nodo = {reached.tolist()}")
# t=1: cada nodo ve a sus vecinos directos (+ él mismo)
# t=2: ve a los vecinos de sus vecinos
# t=3: el grafo (7 nodos, diámetro pequeño) queda casi totalmente conectado
```

La fila $i$ de $\hat{A}^t X$ tiene entradas no nulas exactamente en los nodos que están a $\le t$ saltos de $i$: ese conjunto es el **receptive field** del nodo $i$ tras $t$ capas. Como el grafo A–G es pequeño y bien conectado, a $t=3$ casi todos los nodos se "ven" entre sí.

### TensorFlow

Mismo cómputo con `tf.Variable` y `tf.matmul`. La $W$ se declara como variable entrenable; el resto es álgebra densa idéntica.

```python
import tensorflow as tf

# --- 1. Grafo ---
edges = [(0, 2), (2, 5), (5, 6), (6, 4), (4, 0), (4, 1), (1, 0), (3, 2)]
N = 7

A = tf.zeros((N, N))
idx = []
for i, j in edges:
    idx += [[i, j], [j, i]]                       # ambos sentidos => simétrica
A = tf.tensor_scatter_nd_update(A, idx, tf.ones(len(idx)))

# --- 2. Self-loops + normalización simétrica ---
def normalize_adjacency(A):
    A_tilde = A + tf.eye(tf.shape(A)[0])          # Ã = A + I
    deg = tf.reduce_sum(A_tilde, axis=1)          # grados
    d_inv_sqrt = tf.pow(deg, -0.5)                # D̃^{-1/2}
    # escalado fila (unsqueeze a columna) y columna (unsqueeze a fila)
    A_hat = (d_inv_sqrt[:, None] * A_tilde) * d_inv_sqrt[None, :]
    return A_hat

A_hat = normalize_adjacency(A)
print("filas de Â suman ~", tf.reduce_sum(A_hat, axis=1).numpy())

# --- 3. Capa de message passing: H' = σ(Â · H · W) ---
class GraphConv(tf.Module):
    def __init__(self, in_dim, out_dim, seed=0):
        super().__init__()
        init = tf.initializers.GlorotUniform(seed=seed)
        self.W = tf.Variable(init((in_dim, out_dim)), name="W")   # W aprendible

    def __call__(self, A_hat, H):
        agg = tf.matmul(A_hat, H)                 # Â · H : (N, in)
        return tf.matmul(agg, self.W)             # · W   : (N, out)

# --- 4. Apilar capas ---
H0 = tf.eye(N)                                    # one-hot por nodo
layer1 = GraphConv(N, 8, seed=1)
layer2 = GraphConv(8, 8, seed=2)
layer3 = GraphConv(8, 4, seed=3)

H1 = tf.nn.relu(layer1(A_hat, H0))                # 1 salto
H2 = tf.nn.relu(layer2(A_hat, H1))                # 2 saltos
H3 = layer3(A_hat, H2)                            # 3 saltos (sin σ final)

print("H0", H0.shape, "-> H1", H1.shape, "-> H2", H2.shape, "-> H3", H3.shape)

# --- Propagación pura (sin W ni σ): receptive field creciente ---
X = tf.eye(N)
for t in range(1, 4):
    X = tf.matmul(A_hat, X)
    reached = tf.reduce_sum(tf.cast(tf.abs(X) > 1e-6, tf.int32), axis=1)
    print(f"t={t}: receptive field por nodo = {reached.numpy().tolist()}")
```

### JAX

En JAX el cómputo es una **función pura** con los parámetros pasados explícitamente: no hay estado mutable ni `self.W` escondido. Esto hace trivial aplicar `jit` y, llegado el caso, `grad`. Inicializamos los pesos con `jax.random` y los pasamos como una lista.

```python
import jax
import jax.numpy as jnp

# --- 1. Grafo ---
edges = [(0, 2), (2, 5), (5, 6), (6, 4), (4, 0), (4, 1), (1, 0), (3, 2)]
N = 7

A = jnp.zeros((N, N))
for i, j in edges:
    A = A.at[i, j].set(1.0).at[j, i].set(1.0)     # ambos sentidos => simétrica

# --- 2. Self-loops + normalización simétrica (función pura) ---
def normalize_adjacency(A):
    A_tilde = A + jnp.eye(A.shape[0])             # Ã = A + I
    deg = A_tilde.sum(axis=1)                     # grados
    d_inv_sqrt = jnp.power(deg, -0.5)             # D̃^{-1/2}
    A_hat = d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]
    return A_hat

A_hat = normalize_adjacency(A)
print("filas de Â suman ~", A_hat.sum(axis=1))

# --- 3. Inicializar pesos como PARÁMETROS EXPLÍCITOS (lista de matrices W) ---
def init_params(key, dims):
    # dims = [in, h1, h2, out] -> una W por capa
    params = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        key, sub = jax.random.split(key)
        # init tipo Glorot: escala ~ sqrt(2/(d_in+d_out))
        scale = jnp.sqrt(2.0 / (d_in + d_out))
        params.append(jax.random.normal(sub, (d_in, d_out)) * scale)
    return params

# --- 4. Forward: capas H' = σ(Â · H · W) como composición pura ---
def gnn_forward(params, A_hat, H):
    *hidden, W_last = params
    for W in hidden:
        H = jax.nn.relu(jnp.matmul(jnp.matmul(A_hat, H), W))   # σ(Â H W)
    H = jnp.matmul(jnp.matmul(A_hat, H), W_last)               # última capa sin σ
    return H

gnn_forward = jax.jit(gnn_forward)               # compila el grafo de cómputo

key = jax.random.PRNGKey(0)
params = init_params(key, dims=[N, 8, 8, 4])     # 3 capas
H0 = jnp.eye(N)                                  # one-hot por nodo
H_out = gnn_forward(params, A_hat, H0)
print("H0", H0.shape, "-> H_out", H_out.shape)   # (7,7) -> (7,4)

# --- Propagación pura (sin W ni σ): receptive field ---
X = jnp.eye(N)
for t in range(1, 4):
    X = jnp.matmul(A_hat, X)
    reached = (jnp.abs(X) > 1e-6).sum(axis=1)
    print(f"t={t}: receptive field por nodo = {reached.tolist()}")
```

{{< concept-alert type="clave" >}}
Los tres bloques computan **exactamente lo mismo**: $H' = \sigma(\hat{A}\,H\,W)$ apilada en capas. Lo que cambia es cómo cada framework expresa los pesos y la composición. **PyTorch** los esconde en `nn.Module` (`self.W`); **TensorFlow** usa `tf.Variable` dentro de un `tf.Module` y `tf.matmul`; **JAX** los hace **parámetros explícitos** que viajan como argumento de una función pura, lo que vuelve el forward inmediatamente `jit`-able y `grad`-able. El operador central, `A_hat @ H @ W`, es idéntico carácter por carácter en los tres.
{{< /concept-alert >}}

---

## 4. Cómo evolucionan los embeddings: el receptive field crece

La salida que más enseña es la de la **propagación pura** (sin $W$, sin $\sigma$): $\hat{A}^t X$. Con $X = I$ (one-hot), la fila $i$ de $\hat{A}^t$ tiene entradas no nulas exactamente en los nodos alcanzables desde $i$ en $\le t$ saltos. Para el grafo A–G:

- **$t=1$ (una capa):** cada nodo mezcla su estado con el de sus **vecinos directos**. El nodo $A$ (conectado a $B$, $C$, $E$) recibe de esos tres más de sí mismo; el nodo $D$ (solo conectado a $C$) recibe solo de $C$ y de sí mismo.
- **$t=2$ (dos capas):** la información llega a **vecinos de vecinos**. Ahora $D$ —vía $C$— ya "siente" a $A$ y a $F$, aunque no tenga arista directa con ellos.
- **$t=3$ (tres capas):** en este grafo pequeño (diámetro reducido), casi todos los nodos se influyen mutuamente. El embedding de cada nodo codifica información de prácticamente todo el grafo.

Esta es la lección estructural del message passing: **$t$ capas $=$ receptive field de $t$ saltos**. Es el análogo en grafos de apilar convoluciones en una CNN para ampliar el campo receptivo. Y anticipa una patología: con demasiadas capas, todos los nodos terminan viendo casi lo mismo y sus embeddings convergen —el *over-smoothing*— detallado en el [fundamento de message passing](/fundamentos/message-passing#6-receptive-field-profundidad-y-sus-patologias).

---

## 5. Invarianza a permutación y receptive field

Dos propiedades cierran lo que construimos, y ambas son consecuencia directa de que la agregación sea una **suma**:

**Invarianza/equivarianza a permutación.** Un grafo no tiene orden canónico de sus nodos: numerar $A$ como 0 o como 6 es arbitrario. La capa $\hat{A}HW$ es **equivariante a permutación**: si reordenamos los nodos con una matriz de permutación $P$ (aplicando $PAP^\top$ a la adyacencia y $PH$ a las features), la salida se reordena de la misma forma, $P H'$, pero los *embeddings de cada nodo no cambian*. Esto se cumple precisamente porque la suma $\sum_{w\in\mathcal{N}(v)} h_w$ no depende del orden de los sumandos —por eso la etapa 3 **debe** ser conmutativa—. Si después agregamos todos los nodos a un único vector de grafo (con otra suma o media), obtenemos **invarianza** total: el mismo grafo dibujado de dos maneras produce la misma predicción. La formalización está en el [fundamento de message passing](/fundamentos/message-passing#5-invarianza-y-equivarianza-a-permutacion).

**Receptive field.** Cada capa expande en un salto el conjunto de nodos que influyen en un embedding dado. Con $t$ capas, el embedding de un nodo resume su vecindario de radio $t$. Elegir la profundidad de una GNN es, entonces, elegir **cuánto contexto del grafo** quiere ver cada nodo —ni tan poco que ignore estructura relevante, ni tanto que caiga en over-smoothing—.

Lo que acabamos de implementar en tres frameworks —self-loops, normalización por grado, $\hat{A}HW$ apilado— es exactamente una **GCN** ([Kipf y Welling, 2017](/papers/gcn-kipf-2017)) y un caso particular del marco general de [Gilmer et al. (MPNN, 2017)](/papers/mpnn-gilmer-2017), donde la función de mensaje es la identidad, la agregación es suma normalizada y el update es lineal + ReLU. Cambiar esas tres funciones —mensaje, agregación, update— genera toda la familia de GNN; la maquinaria de propagación matricial permanece igual.

---

**Ver también:** [Clase 27](/clases/clase-27) · [Teoría - Clase 27](/clases/clase-27/teoria) · Fundamentos: [Message passing](/fundamentos/message-passing) · Papers: [MPNN - Gilmer et al. 2017](/papers/mpnn-gilmer-2017) · [GCN - Kipf y Welling 2017](/papers/gcn-kipf-2017).
