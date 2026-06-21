---
title: "GCN desde cero (clasificación de nodos)"
weight: 2
math: true
---

La [teoría de la clase](/clases/clase-27) describe la GNN como un esquema general de *message passing*: cada nodo recibe mensajes de sus vecinos, los agrega, y actualiza su estado. La **Graph Convolutional Network** de Kipf y Welling (2017) es la encarnación más limpia y más citada de esa idea. Es lo bastante simple como para implementarla desde cero en una tarde, y lo bastante potente como para que, con apenas un par de etiquetas, clasifique correctamente casi todos los nodos de un grafo. Eso es lo que vamos a construir aquí, en **tres frameworks** —PyTorch, TensorFlow y JAX— sobre un grafo de juguete clásico: el **club de karate de Zachary**.

La promesa de este capítulo es concreta: vas a ver una red de 2 capas aprender a separar dos facciones de un club social a partir de **solo 2 nodos etiquetados** entre 34. No es magia; es que la estructura del grafo —quién es amigo de quién— propaga la señal de esas dos etiquetas a todo el resto. Ese fenómeno, la **clasificación semi-supervisada de nodos**, es el caso de uso estrella de las GCN y la mejor forma de internalizar por qué la convolución sobre grafos funciona.

Para la matemática completa del operador y su deducción espectral, remitimos al [fundamento de redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos) y al [análisis del paper de Kipf y Welling](/papers/gcn-kipf-2017). Aquí nos concentramos en construir el operador con las manos y verlo entrenar.

---

## 1. La regla de propagación en una ecuación

Toda la GCN cabe en una línea. La capa $l$ transforma las representaciones de los nodos $H^{(l)}$ en las de la capa siguiente $H^{(l+1)}$ mediante:

$$
H^{(l+1)} = \sigma\!\left( \tilde{D}^{-1/2}\, \tilde{A}\, \tilde{D}^{-1/2}\, H^{(l)}\, W^{(l)} \right). \tag{1}
$$

Descompongamos cada símbolo, porque cada uno carga una decisión de diseño:

- $A \in \mathbb{R}^{N \times N}$ es la **matriz de adyacencia** del grafo: $A_{ij}=1$ si los nodos $i$ y $j$ están conectados. $N$ es el número de nodos.
- $\tilde{A} = A + I$ añade **auto-conexiones** (la identidad $I$). Sin esto, al agregar los vecinos un nodo *perdería su propia información*: solo vería a los demás, no a sí mismo. El truco $A+I$ garantiza que cada nodo siempre se incluya en su propia agregación.
- $\tilde{D}$ es la matriz de grados de $\tilde{A}$: $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ (cuántos vecinos tiene el nodo $i$, contándose a sí mismo). Es diagonal.
- $\tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}$ es la **normalización simétrica**, conocida como el *renormalization trick*. Reescala cada arista por $1/\sqrt{\tilde{d}_i \tilde{d}_j}$, de modo que los nodos con muchos vecinos no dominen la suma. Llamaremos a esta matriz $\hat{A}$ y la calcularemos **una sola vez**: no depende de los pesos, solo del grafo.
- $H^{(l)} \in \mathbb{R}^{N \times F_l}$ son las representaciones de los $N$ nodos en la capa $l$, cada una de dimensión $F_l$. La entrada es $H^{(0)} = X$, la matriz de *features* de los nodos.
- $W^{(l)} \in \mathbb{R}^{F_l \times F_{l+1}}$ es la matriz de **pesos entrenables** de la capa: proyecta de $F_l$ a $F_{l+1}$ dimensiones. Es lo único que se aprende.
- $\sigma$ es una no linealidad (ReLU en capas ocultas; en la última capa, un softmax para clasificar).

### 1.1 Verificación de dimensiones

Antes de programar, comprobemos que las formas encajan. Es el chequeo que más errores evita:

| Objeto | Forma | Comentario |
|---|---|---|
| $\hat{A}$ | $N \times N$ | matriz de propagación, fija |
| $H^{(l)}$ | $N \times F_l$ | un vector por nodo |
| $W^{(l)}$ | $F_l \times F_{l+1}$ | proyección de features |
| $H^{(l)} W^{(l)}$ | $N \times F_{l+1}$ | primero transforma cada nodo |
| $\hat{A}\,(H^{(l)} W^{(l)})$ | $N \times F_{l+1}$ | luego mezcla con los vecinos |

El producto $\hat{A} H^{(l)} W^{(l)}$ es asociativo, pero el **orden importa para la eficiencia**: conviene calcular primero $H^{(l)} W^{(l)}$ (transformación lineal por nodo, una matriz $N \times F_{l+1}$ relativamente angosta) y solo después multiplicar por $\hat{A}$ (la mezcla con vecinos). El otro orden, $\hat{A} H^{(l)}$ primero, da el mismo resultado pero suele ser más caro. En el código verás `A_hat @ (X @ W)`.

### 1.2 Por qué dos capas

Cada capa GCN propaga información **un salto** (un *hop*) en el grafo: tras una capa, cada nodo "ve" a sus vecinos directos; tras dos capas, ve a los vecinos de sus vecinos (radio 2). Kipf y Welling encontraron que **dos capas** es el punto dulce para la mayoría de los benchmarks de clasificación de nodos: suficiente para que la señal se propague a una vecindad útil, sin caer en el **over-smoothing** —el fenómeno por el cual, con demasiadas capas, todos los nodos convergen a representaciones casi idénticas y se vuelven indistinguibles. Más profundidad no ayuda y suele perjudicar. La arquitectura canónica es entonces:

$$
Z = \operatorname{softmax}\!\Big( \hat{A}\; \operatorname{ReLU}\!\big( \hat{A}\, X\, W^{(0)} \big)\, W^{(1)} \Big). \tag{2}
$$

Dos multiplicaciones por $\hat{A}$ (dos hops), una ReLU en medio, un softmax al final. Eso es toda la red.

### 1.3 Qué significa "semi-supervisado"

El escenario que ataca la GCN es el de **pocas etiquetas**. Tenemos $N$ nodos, pero solo un puñado tiene etiqueta conocida (el conjunto $\mathcal{Y}_L$, los nodos *labeled*). El resto no. La red ve **las features de todos los nodos y la estructura completa del grafo** durante el forward —porque $\hat{A}$ y $X$ incluyen a todos—, pero la pérdida solo se calcula sobre los nodos etiquetados:

$$
\mathcal{L} = -\sum_{i \in \mathcal{Y}_L} \sum_{c} Y_{ic} \log Z_{ic}. \tag{3}
$$

Esta es la clave conceptual: aunque solo penalicemos los errores en los nodos etiquetados, **el gradiente que esos nodos generan fluye, vía $\hat{A}$, hacia los pesos que también afectan a los nodos sin etiqueta**. Y en inferencia, la predicción de un nodo sin etiqueta depende de su vecindad, que tarde o temprano conecta con algún nodo etiquetado. La estructura del grafo es el canal por el que la supervisión escasa se difunde a todo el conjunto.

---

## 2. El grafo de juguete: club de karate de Zachary

Usaremos el **club de karate de Zachary** (Zachary, 1977), el "hola mundo" del análisis de redes sociales. Es un grafo de **34 nodos** (los miembros de un club universitario de karate) y 78 aristas (relaciones de amistad fuera del club). Durante el estudio, el club se fracturó en dos facciones por un conflicto entre el instructor (nodo 0, "Mr. Hi") y el presidente (nodo 33, "Officer"). Cada miembro terminó en una de las dos facciones: esas son nuestras **2 clases**.

El experimento semi-supervisado clásico: etiquetamos **solo 2 nodos** —el instructor (clase 0) y el presidente (clase 1), los dos "líderes"— y dejamos que la GCN infiera la facción de los otros 32 a partir de la estructura de amistades. Es exactamente el régimen de la Ecuación (3) con $|\mathcal{Y}_L| = 2$.

```python
import numpy as np
import networkx as nx

# Karate Club: 34 nodos, 2 facciones
G = nx.karate_club_graph()
N = G.number_of_nodes()                      # 34
nodes = sorted(G.nodes())
A = nx.to_numpy_array(G, nodelist=nodes)     # (34, 34) adyacencia binaria simetrica

# Etiqueta = faccion. 'Mr. Hi' -> 0, 'Officer' -> 1
labels = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in nodes])

# Semi-supervisado: solo 2 nodos etiquetados (instructor=0, presidente=33)
train_mask = np.zeros(N, dtype=bool)
train_mask[[0, 33]] = True

print(f"N={N}  aristas={G.number_of_edges()}  etiquetados={train_mask.sum()}")
# N=34  aristas=78  etiquetados=2
```

### 2.1 Features de los nodos: la matriz identidad

¿Qué *features* le damos a cada nodo? En el club de karate no hay atributos naturales (edad, género, etc.) en el dataset estándar. La elección canónica de Kipf y Welling para este caso es usar la **matriz identidad** $X = I_N$: cada nodo se representa con un vector *one-hot* de dimensión $N$. Esto equivale a decir "cada nodo es su propia feature, sin información de contenido". La red debe aprender a clasificar **únicamente a partir de la estructura del grafo** —que es justo lo que queremos demostrar.

```python
X = np.eye(N, dtype=np.float32)   # (34, 34): one-hot por nodo, sin features de contenido
```

Con features identidad, la primera capa $\hat{A} X W^{(0)} = \hat{A} W^{(0)}$ aprende, en efecto, un embedding por nodo modulado por la vecindad. Toda la señal viene de $\hat{A}$.

---

## 3. El operador $\hat{A}$: el *renormalization trick* (común a los 3 frameworks)

Antes de tocar ningún framework, calculamos $\hat{A} = \tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}$ **una sola vez** en NumPy puro. No depende de los pesos ni cambia durante el entrenamiento; es geometría fija del grafo. Lo hacemos portable para reusarlo en PyTorch, TF y JAX.

```python
def normalize_adjacency(A):
    """Renormalization trick de Kipf & Welling: D~^-1/2 (A + I) D~^-1/2.
    A: (N, N) adyacencia binaria.  Devuelve A_hat: (N, N) normalizada simetrica.
    """
    N = A.shape[0]
    A_tilde = A + np.eye(N)               # 1) auto-conexiones: A~ = A + I
    deg = A_tilde.sum(axis=1)             # 2) grados de A~ (vector de N)
    d_inv_sqrt = np.power(deg, -0.5)      # 3) D~^-1/2 como VECTOR (no matriz)
    # 4) escalar fila i y columna j por 1/sqrt(d_i) y 1/sqrt(d_j) via broadcasting
    A_hat = A_tilde * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return A_hat.astype(np.float32)

A_hat = normalize_adjacency(A)
assert A_hat.shape == (N, N)                       # (34, 34)
assert np.allclose(A_hat, A_hat.T)                 # simetrica
assert np.isfinite(A_hat).all()                    # sin inf/nan
```

Dos detalles de implementación que evitan errores reales:

1. **$\tilde{D}^{-1/2}$ como vector, no como matriz diagonal.** La forma "de libro" es $D^{-1/2} \tilde{A} D^{-1/2}$ con $D^{-1/2}$ una matriz diagonal de $N \times N$. Pero construir esa diagonal con `np.diag(1/np.sqrt(deg))` y multiplicar matrices puede generar `inf` intermedios (si algún grado fuera 0) o desperdiciar memoria. Escalar por *broadcasting* con el vector `d_inv_sqrt[:, None] * d_inv_sqrt[None, :]` es matemáticamente idéntico, numéricamente estable y mucho más eficiente. Como $\tilde{A}$ siempre tiene la diagonal en 1 (por el $+I$), ningún grado es 0 y no hay división por cero.
2. **La simetría es un invariante que conviene chequear.** Si tu grafo es no dirigido, $\hat{A}$ debe ser simétrica. El `assert` lo verifica; si falla, hay un bug en cómo construiste $A$.

Esta matriz $\hat{A}$ de $34 \times 34$ es el corazón de la GCN. El resto es una transformación lineal entrenable envuelta en dos productos por $\hat{A}$.

---

## 4. Sección 1: PyTorch

PyTorch es el framework dominante en investigación de GNN (PyTorch Geometric es la librería de referencia). Su estilo *define-by-run* hace que la capa GCN se exprese como un `nn.Module` natural.

### 4.1 La capa GCN

Una capa GCN es, literalmente, "multiplica por $\hat{A}$, luego por $W$". La transformación entrenable es una `nn.Linear` **sin bias** (el bias se puede añadir, pero la formulación canónica de la Ecuación (1) no lo incluye en el operador de propagación):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class GCNLayer(nn.Module):
    """Una capa: H' = A_hat @ (H @ W).  Equivalente a la Ecuacion (1) sin sigma."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)  # W: (F_in, F_out)

    def forward(self, A_hat, H):
        # H: (N, F_in) -> H @ W: (N, F_out) -> A_hat @ (.): (N, F_out)
        return A_hat @ self.linear(H)   # primero transforma por nodo, luego mezcla vecinos
```

El orden `A_hat @ self.linear(H)` aplica primero $H W$ (proyección por nodo) y después la mezcla por $\hat{A}$, como discutimos en 1.1.

### 4.2 El modelo de 2 capas

```python
class GCN(nn.Module):
    """GCN de 2 capas para clasificacion de nodos (Ecuacion 2)."""
    def __init__(self, in_features, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.gc1 = GCNLayer(in_features, hidden)       # F -> hidden
        self.gc2 = GCNLayer(hidden, num_classes)       # hidden -> C
        self.dropout = dropout

    def forward(self, A_hat, X):
        H = F.relu(self.gc1(A_hat, X))                 # capa 1 + ReLU: (N, hidden)
        H = F.dropout(H, self.dropout, training=self.training)
        Z = self.gc2(A_hat, H)                         # capa 2 (logits): (N, C)
        return Z                                       # softmax lo hace cross_entropy
```

No aplicamos `softmax` explícito: `F.cross_entropy` lo hace internamente (espera logits). El `dropout` entre las dos capas es la regularización que Kipf y Welling usan; con solo 2 etiquetas, regularizar importa.

### 4.3 El loop de entrenamiento con máscara

Aquí está la pieza semi-supervisada: el forward usa **todos** los nodos, pero la pérdida solo se evalúa sobre los etiquetados, indexando los logits con `train_mask`.

```python
# Tensores (una sola vez)
A_hat_t = torch.tensor(A_hat)                  # (34, 34)
X_t = torch.tensor(X)                          # (34, 34) identidad
y_t = torch.tensor(labels, dtype=torch.long)   # (34,)
mask = torch.tensor(train_mask)                # (34,) bool, solo 2 True

model = GCN(in_features=N, hidden=16, num_classes=2, dropout=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(101):
    model.train()
    optimizer.zero_grad()
    logits = model(A_hat_t, X_t)               # (34, 2) -- forward sobre TODOS los nodos
    # Perdida SOLO sobre nodos etiquetados (Ecuacion 3)
    loss = F.cross_entropy(logits[mask], y_t[mask])
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(A_hat_t, X_t).argmax(dim=1)
            acc = (pred == y_t).float().mean()  # accuracy sobre los 34 nodos
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}  acc {acc.item():.3f}")
```

Salida típica (las cifras varían levemente con la semilla, pero la tendencia es robusta):

```
epoch   0  loss 0.6885  acc 0.559
epoch  20  loss 0.3794  acc 0.971
epoch  40  loss 0.0496  acc 0.971
epoch  60  loss 0.0166  acc 0.971
epoch  80  loss 0.0052  acc 0.971
epoch 100  loss 0.0038  acc 0.971
```

**Lee ese resultado con atención.** Empezamos clasificando bien apenas el 56% de los nodos (azar). Tras 20 épocas, con la pérdida calculada sobre **solo 2 nodos etiquetados**, la red clasifica correctamente el **97%** de los 34 nodos (33 de 34). No vio las etiquetas de los otros 32; las dedujo de la estructura. Esto es la difusión de la señal de la que hablábamos: el gradiente de los 2 nodos líderes, propagado dos hops por $\hat{A}$, basta para colorear el grafo entero.

---

## 5. Sección 2: TensorFlow

TensorFlow 2.x con `tf.GradientTape` expresa lo mismo con un estilo de gradientes explícito. Para que el paralelo con la matemática sea nítido, escribimos el forward a bajo nivel con `tf.Variable` para los pesos —es la versión más transparente del operador. (La variante con `tf.keras.layers.Layer` se menciona al final.)

### 5.1 Pesos y forward

```python
import tensorflow as tf
tf.random.set_seed(0)

A_hat_tf = tf.constant(A_hat)                       # (34, 34)
X_tf = tf.constant(X)                               # (34, 34)
y_tf = tf.constant(labels, dtype=tf.int32)          # (34,)
mask_tf = tf.constant(train_mask)                   # (34,) bool

HIDDEN, NUM_CLASSES = 16, 2
# Inicializacion Glorot, sin bias (igual que la nn.Linear de PyTorch)
init = tf.keras.initializers.GlorotUniform(seed=0)
W0 = tf.Variable(init((N, HIDDEN)),       name="W0")   # (34, 16)
W1 = tf.Variable(init((HIDDEN, NUM_CLASSES)), name="W1")   # (16, 2)

def gcn_forward(A_hat, X, training=False, dropout=0.5):
    H = tf.nn.relu(A_hat @ (X @ W0))                # capa 1 + ReLU: (N, 16)
    if training:
        H = tf.nn.dropout(H, rate=dropout)
    Z = A_hat @ (H @ W1)                            # capa 2 (logits): (N, 2)
    return Z
```

El patrón `A_hat @ (X @ W0)` es idéntico al de PyTorch: proyección por nodo y luego mezcla por $\hat{A}$.

### 5.2 El loop con `tf.GradientTape`

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
params = [W0, W1]

for epoch in range(101):
    with tf.GradientTape() as tape:
        logits = gcn_forward(A_hat_tf, X_tf, training=True)        # (34, 2), TODOS los nodos
        # Enmascarar: perdida solo sobre etiquetados (Ecuacion 3)
        logits_l = tf.boolean_mask(logits, mask_tf)                # (2, 2)
        labels_l = tf.boolean_mask(y_tf, mask_tf)                  # (2,)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels_l, logits_l)
        )
        # weight decay 5e-4 (equivalente al de PyTorch)
        loss += 5e-4 * tf.add_n([tf.nn.l2_loss(w) for w in params])

    grads = tape.gradient(loss, params)
    optimizer.apply_gradients(zip(grads, params))

    if epoch % 20 == 0:
        logits_eval = gcn_forward(A_hat_tf, X_tf, training=False)
        pred = tf.argmax(logits_eval, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(pred == y_tf, tf.float32))
        print(f"epoch {epoch:3d}  loss {float(loss):.4f}  acc {float(acc):.3f}")
```

El enmascarado se hace con `tf.boolean_mask` en vez de la indexación booleana de PyTorch (`logits[mask]`), pero el efecto es el mismo: la cross-entropy ve solo las 2 filas etiquetadas. El `weight_decay` de PyTorch (que el `Adam` aplica automáticamente) se traduce aquí en un término L2 explícito sumado a la pérdida. La curva de accuracy reproduce la de PyTorch: ~0.50 al inicio, ~0.97 tras ~20 épocas.

---

## 6. Sección 3: JAX

JAX abraza las **funciones puras + transformaciones**. El modelo no tiene estado: el forward es una función de `(params, A_hat, X)`. Los gradientes salen de componer `jax.grad`, y `optax` lleva el estado del optimizador por fuera. El operador GCN, al ser pura álgebra matricial, queda especialmente limpio.

### 6.1 Parámetros y forward puro

```python
import jax
import jax.numpy as jnp
import optax

A_hat_j = jnp.array(A_hat)                          # (34, 34)
X_j = jnp.array(X)                                  # (34, 34)
y_j = jnp.array(labels)                             # (34,)
mask_j = jnp.array(train_mask)                      # (34,) bool

def init_params(key, in_features, hidden, num_classes):
    k1, k2 = jax.random.split(key)
    # Inicializacion Glorot/Xavier escalada
    glorot = lambda k, shape: jax.random.normal(k, shape) * jnp.sqrt(2.0 / sum(shape))
    return {
        "W0": glorot(k1, (in_features, hidden)),    # (34, 16)
        "W1": glorot(k2, (hidden, num_classes)),    # (16, 2)
    }

def gcn_forward(params, A_hat, X):
    """Forward PURO. Sin dropout para mantener la funcion determinista
    (el dropout requeriria pasar una PRNGKey; lo omitimos por claridad)."""
    H = jax.nn.relu(A_hat @ (X @ params["W0"]))     # capa 1 + ReLU: (N, 16)
    Z = A_hat @ (H @ params["W1"])                  # capa 2 (logits): (N, 2)
    return Z
```

### 6.2 Pérdida enmascarada y `value_and_grad`

```python
def loss_fn(params, A_hat, X, y, mask):
    logits = gcn_forward(params, A_hat, X)          # (34, 2), TODOS los nodos
    # cross-entropy por nodo
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)  # (34,)
    # Promediar SOLO sobre nodos etiquetados (Ecuacion 3): la mascara como peso
    return jnp.sum(ce * mask) / jnp.sum(mask)

key = jax.random.PRNGKey(0)
params = init_params(key, in_features=N, hidden=16, num_classes=2)
optimizer = optax.adamw(learning_rate=0.01, weight_decay=5e-4)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, A_hat_j, X_j, y_j, mask_j)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for epoch in range(101):
    params, opt_state, loss = train_step(params, opt_state)
    if epoch % 20 == 0:
        pred = jnp.argmax(gcn_forward(params, A_hat_j, X_j), axis=1)
        acc = jnp.mean(pred == y_j)
        print(f"epoch {epoch:3d}  loss {float(loss):.4f}  acc {float(acc):.3f}")
```

Dos idiomas JAX que vale la pena notar:

- **Máscara como peso, no como índice.** En vez de indexar `logits[mask]` (que produciría una forma dinámica incompatible con `jit`), calculamos la cross-entropy de los 34 nodos y la promediamos ponderada por la máscara binaria: `jnp.sum(ce * mask) / jnp.sum(mask)`. Matemáticamente idéntico a la Ecuación (3), pero con forma estática —lo que `jax.jit` exige.
- **`@jax.jit` compila todo el paso** a XLA. La primera llamada compila (lenta); las siguientes vuelan. El `weight_decay` lo aporta `optax.adamw` (AdamW = Adam con weight decay desacoplado, el equivalente correcto del `weight_decay` de PyTorch).

La salida vuelve a reproducir la curva: accuracy ~0.62 al inicio, ~0.97 tras 50 épocas. Los tres frameworks convergen al mismo resultado porque implementan **exactamente la misma matemática**.

---

## 7. Lo que acabamos de ver: la estructura propaga la señal

Detengámonos en el resultado, porque es el corazón pedagógico de este capítulo. Entrenamos con la pérdida calculada sobre **2 de 34 nodos** (un 6% de etiquetas) y obtuvimos **97% de accuracy sobre el grafo completo**. ¿Cómo?

La respuesta es que la GCN no clasifica cada nodo de forma aislada: lo clasifica **en función de su vecindad**. Tras dos capas, la representación de cada nodo es una mezcla (ponderada por $\hat{A}$) de las features de los nodos a distancia $\le 2$. Como el grafo del club está bien separado en dos comunidades densamente conectadas internamente y débilmente entre sí, los nodos cercanos al instructor terminan con representaciones parecidas a la del instructor, y análogamente para el presidente. La frontera de decisión que la red aprende sobre los 2 nodos etiquetados se traslada, vía la conectividad, a los 32 restantes.

Esto es un ejemplo de **regularización por estructura**: el supuesto implícito de la GCN es que *nodos conectados tienden a compartir etiqueta* (homofilia). Ese sesgo inductivo —codificado en $\hat{A}$— es lo que permite generalizar desde poquísimas etiquetas. Es también la razón por la que las GCN brillan en grafos homofílicos (redes de citas, redes sociales) y sufren en grafos heterofílicos (donde los vecinos tienden a ser de clases distintas), un punto que el [fundamento de redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos#5-problemas-conocidos) desarrolla.

| Época | Accuracy (34 nodos) | Qué pasó |
|---:|---:|---|
| 0 | ~0.50–0.56 | pesos aleatorios, clasifica al azar |
| 20 | ~0.97 | la señal de 2 etiquetas ya se difundió por el grafo |
| 100 | ~0.97 | converge; el único error suele ser un nodo en la frontera entre facciones |

El nodo que típicamente queda mal clasificado es uno en la frontera entre las dos comunidades —un miembro con amistades en ambos bandos—, exactamente el caso ambiguo que también costó clasificar en el estudio sociológico original de Zachary.

---

## 8. Comparación lado a lado de los tres frameworks

Las tres implementaciones son **isomorfas**: misma $\hat{A}$ precalculada, mismas dos capas, misma ReLU, misma cross-entropy enmascarada. Cambia el andamiaje, no la matemática.

| Concepto | PyTorch | TensorFlow | JAX |
|---|---|---|---|
| Definición del modelo | `class GCN(nn.Module)` + `forward` | `tf.Variable` + función `gcn_forward` | `dict` de params + función pura |
| Capa GCN | `A_hat @ self.linear(H)` | `A_hat @ (X @ W0)` | `A_hat @ (X @ params["W0"])` |
| Pesos | `nn.Linear(..., bias=False)` | `tf.Variable(GlorotUniform(...))` | `dict` con init Glorot manual |
| Forward sobre todos los nodos | `model(A_hat, X)` | `gcn_forward(A_hat, X)` | `gcn_forward(params, A_hat, X)` |
| Máscara semi-supervisada | `logits[mask]` (indexación) | `tf.boolean_mask(logits, mask)` | `ce * mask` (peso, forma estática) |
| Cross-entropy | `F.cross_entropy` (logits) | `sparse_softmax_cross_entropy_with_logits` | `optax.softmax_cross_entropy_with_integer_labels` |
| Diferenciación | `loss.backward()` | `tf.GradientTape` | `jax.value_and_grad` |
| Weight decay | arg `weight_decay` de Adam | término L2 explícito | `optax.adamw` |
| Optimizador | `torch.optim.Adam` | `keras.optimizers.Adam` | `optax.adamw` |
| Compilación | `torch.compile` (opcional) | `@tf.function` (opcional) | `@jax.jit` |

La lectura práctica:

- **PyTorch** es el más usado en investigación de GNN; PyTorch Geometric y DGL viven aquí. La capa como `nn.Module` es la forma idiomática.
- **TensorFlow** brilla en producción y deployment (TF Serving, TFLite). Spektral es su librería de GNN.
- **JAX** destaca cuando quieres `vmap` sobre muchos grafos pequeños o `jit` agresivo; Jraph es su librería de GNN. La máscara como peso (en vez de índice) es el ajuste mental clave para que todo compile.

---

## 9. Gotchas

Cosas que rompen el código —a veces en silencio— si las pasas por alto.

**Transversales (los tres):**

1. **Olvidar las auto-conexiones ($A+I$).** Si normalizas $A$ directamente sin sumar $I$, cada nodo pierde su propia feature en la agregación y la red aprende mucho peor. El $+I$ no es opcional: es el *renormalization trick*.
2. **Normalización asimétrica.** Usar $\tilde{D}^{-1}\tilde{A}$ (normalización por filas, *random walk*) en vez de $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ (simétrica) cambia el operador. La GCN de Kipf usa la **simétrica**; verifica con `assert np.allclose(A_hat, A_hat.T)`.
3. **Recalcular $\hat{A}$ en cada época.** $\hat{A}$ no depende de los pesos. Calcularla dentro del loop de entrenamiento es desperdicio puro. Hazla una vez, fuera del loop.
4. **Aplicar softmax y luego cross-entropy.** Las tres funciones de pérdida usadas aquí (`F.cross_entropy`, `sparse_softmax_cross_entropy_with_logits`, `softmax_cross_entropy_with_integer_labels`) esperan **logits crudos** y aplican el log-softmax internamente. Si aplicas un `softmax` antes, normalizas dos veces y los gradientes se aplanan.
5. **Demasiadas capas.** Apilar 4, 8 o más capas GCN no mejora: provoca *over-smoothing* y la accuracy cae. Dos capas es el estándar; tres como mucho en grafos grandes.

**PyTorch:**

6. **`model.eval()` para la evaluación.** Si evalúas sin pasar a modo `eval`, el `dropout` sigue activo y la accuracy de test fluctúa. Usa `model.eval()` + `torch.no_grad()`.

**TensorFlow:**

7. **`tf.boolean_mask` vs indexación.** TF no soporta `logits[mask]` como PyTorch. Usa `tf.boolean_mask(logits, mask)`. Y recuerda desactivar el dropout (`training=False`) en evaluación.

**JAX:**

8. **Máscara como índice rompe `jit`.** `logits[mask]` produce una forma dinámica que `@jax.jit` no puede compilar. Usa la máscara como **peso** en el promedio: `jnp.sum(ce * mask) / jnp.sum(mask)`.
9. **`adam` vs `adamw` para el weight decay.** `optax.adam` no aplica weight decay; necesitas `optax.adamw` (o sumar el término L2 a mano) para igualar el `weight_decay` de PyTorch.

---

## 10. Cómo seguir

1. **Cambia el número de nodos etiquetados.** Prueba con 4, 6, 10 etiquetas y observa cómo sube la accuracy. Con 1 sola etiqueta por clase ya se ve el efecto; el experimento es sorprendentemente robusto.
2. **Quita el *renormalization trick*.** Reemplaza $\hat{A}$ por $\tilde{A}$ sin normalizar (o por $A$ sin auto-conexiones) y mide la degradación. Es la mejor forma de internalizar para qué sirve cada pieza.
3. **Añade una tercera capa** y observa el *over-smoothing*: la accuracy debería empeorar, no mejorar.
4. **Pasa a un dataset real:** Cora (2708 nodos, 7 clases, papers de ML con features de bag-of-words). El código es **idéntico**; solo cambian $A$, $X$ y el número de clases. Es el benchmark del paper original.
5. **Implementa GraphSAGE o GAT** sobre el mismo esqueleto: cambian la regla de agregación (muestreo de vecinos en SAGE, atención en GAT) pero el loop semi-supervisado es el mismo.

---

## 11. Cross-links

- [Teoría — Clase 27: Redes Neuronales de Grafos](/clases/clase-27): el mecanismo de message passing, las tareas (clasificación de nodos, de grafos, predicción de aristas) y la familia de modelos (GCN, GraphSAGE, GGNN).
- [Fundamento: Redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos): la deducción del operador, invarianza a permutación, over-smoothing y la conexión espectral.
- [Paper: GCN (Kipf & Welling, 2017)](/papers/gcn-kipf-2017): el paper canónico que implementamos aquí, con la deducción desde la convolución espectral y los resultados en Cora, Citeseer y Pubmed.

---

**Ver también:** [Profundización — Clase 27](/clases/clase-27/profundizacion) · [Teoría — Clase 27](/clases/clase-27/teoria).
