---
title: "GAT desde cero (atención en grafos)"
weight: 3
math: true
---

En el camino de las [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos), una capa GCN actualiza cada nodo promediando a sus vecinos con un peso **fijo y estructural**: el coeficiente $1/\sqrt{d_i d_j}$ depende solo de los grados de los nodos, no de lo que cada vecino *contiene*. Eso es elegante y barato, pero rígido: un vecino ruidoso pesa lo mismo que un vecino informativo, simplemente porque tienen el mismo grado. **GAT** (Graph Attention Network, [Veličković et al., 2018](/papers/gat-velickovic-2018)) reemplaza ese peso fijo por uno **aprendido**: la red decide, mirando las features de cada par $(i, j)$, cuánto debe atender el nodo $i$ a su vecino $j$. Es el [mecanismo de atención](/fundamentos/mecanismo-atencion) trasplantado del mundo de las secuencias al mundo de los grafos.

Vamos a implementar una capa GAT **desde cero** sobre un grafo de juguete, en los tres frameworks, con el cálculo de coeficientes por pares, la **atención enmascarada** (la pieza que distingue GAT de la self-attention de un Transformer), el softmax por filas y la agregación multi-head. Al final cerraremos con la conexión que vale por todo el capítulo: la self-attention de los Transformers que vimos en la [Clase 14](/clases/clase-14) es, literalmente, **un GAT sobre el grafo completo**.

---

## 1. Por qué atención: del peso fijo al peso aprendido

Recordemos la actualización de un nodo en una GNN como *paso de mensajes*: cada nodo $i$ recoge mensajes de su vecindario $\mathcal{N}(i)$ y los combina. Lo único que cambia entre arquitecturas es **cómo se ponderan esos mensajes**.

| Arquitectura | Peso del vecino $j$ sobre $i$ | ¿Depende de las features? |
|---|---|---|
| GCN (Kipf-Welling) | $1/\sqrt{d_i\,d_j}$ — fijo, estructural | No |
| GraphSAGE-mean | $1/|\mathcal{N}(i)|$ — promedio uniforme | No |
| **GAT** | $\alpha_{ij}$ — **aprendido por atención** | **Sí** |

La pregunta que GAT responde es: *si dos vecinos tienen el mismo grado pero uno es mucho más relevante para la tarea, ¿por qué deberían pesar lo mismo?* La respuesta es dejar que un pequeño mecanismo de atención —apenas un vector de parámetros— calcule un peso $\alpha_{ij}$ que **dependa del contenido** de $h_i$ y $h_j$, y que se entrene end-to-end junto con el resto de la red.

{{< concept-alert type="clave" >}}
La diferencia es exactamente la misma que separa un *pooling promedio* de un *pooling con atención*. GCN promedia con pesos congelados por la topología; GAT aprende a quién mirar. Y como el peso lo produce una función de las features (no de la identidad del nodo), GAT es **inductivo**: la misma capa funciona sobre un grafo nuevo nunca visto, sin reentrenar. Eso es lo que GCN, atado a la matriz de adyacencia normalizada del grafo de entrenamiento, no puede hacer de forma directa.
{{< /concept-alert >}}

---

## 2. El mecanismo, paso a paso

Una capa GAT transforma features de entrada $h_i \in \mathbb{R}^{F}$ en features de salida $h_i' \in \mathbb{R}^{F'}$ en cuatro pasos. Fijemos la notación: $N$ nodos, $W \in \mathbb{R}^{F \times F'}$ una proyección lineal compartida, y $\vec{a} \in \mathbb{R}^{2F'}$ el **vector de atención**.

**Paso 1 — Proyección lineal.** Toda feature pasa por la misma $W$:

$$
z_i = W^\top h_i \in \mathbb{R}^{F'}
$$

**Paso 2 — Score de atención por par (sin normalizar).** Para cada par conectado $(i, j)$ se mide cuánto debe atender $i$ a $j$. GAT lo hace concatenando las dos proyecciones, multiplicando por $\vec{a}$ y pasando por una no linealidad **LeakyReLU**:

$$
e_{ij} = \mathrm{LeakyReLU}\!\big(\vec{a}^{\,\top}\,[\,z_i \,\Vert\, z_j\,]\big)
$$

donde $\Vert$ es la concatenación. Un truco de implementación que usaremos en los tres frameworks: como $\vec{a}$ se aplica a un vector concatenado, podemos **partirlo en dos mitades** $\vec{a} = [\vec{a}_{\text{src}} \,\Vert\, \vec{a}_{\text{dst}}]$ y reescribir

$$
\vec{a}^{\,\top}[z_i \Vert z_j] = \vec{a}_{\text{src}}^{\,\top} z_i + \vec{a}_{\text{dst}}^{\,\top} z_j
$$

Esto evita materializar la matriz $N \times N \times 2F'$ de todas las concatenaciones: calculamos dos vectores de tamaño $N$ y los sumamos por *broadcasting*. La fila es el término del nodo $i$ (source), la columna el del vecino $j$ (destination).

**Paso 3 — Atención enmascarada + softmax.** Aquí está la diferencia con un Transformer puro. No queremos que $i$ atienda a *todos* los nodos, sino **solo a sus vecinos** $\mathcal{N}(i)$. Usamos la matriz de adyacencia $A$ como **máscara**: ponemos $-\infty$ en los scores de pares no conectados *antes* del softmax, de modo que su peso post-softmax sea exactamente $0$:

$$
\alpha_{ij} = \mathrm{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}, \qquad e_{ij} \leftarrow -\infty \ \text{si} \ (i,j) \notin E
$$

El softmax se toma **por filas** (sobre el eje de los vecinos $j$), así que cada fila $\alpha_{i\cdot}$ suma 1 y solo tiene masa sobre los vecinos reales.

**Paso 4 — Agregación.** La nueva feature es la combinación convexa de los mensajes de los vecinos, ponderada por la atención, seguida de una no linealidad $\sigma$:

$$
h_i' = \sigma\!\Big(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, z_j\Big) = \sigma\!\Big(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, W^\top h_j\Big)
$$

**Multi-head.** Como en los Transformers, GAT usa $K$ cabezas de atención independientes (cada una con su propia $W^{(k)}$ y $\vec{a}^{(k)}$) y las combina. En capas ocultas se **concatenan**; en la capa de salida se **promedian**:

$$
h_i'^{\text{(oculta)}} = \big\Vert_{k=1}^{K}\, \sigma\Big(\textstyle\sum_j \alpha_{ij}^{(k)} W^{(k)\top} h_j\Big), \qquad
h_i'^{\text{(salida)}} = \sigma\Big(\tfrac{1}{K}\textstyle\sum_{k}\sum_j \alpha_{ij}^{(k)} W^{(k)\top} h_j\Big)
$$

Múltiples cabezas estabilizan el aprendizaje y dejan que cada una capture un tipo distinto de relación de vecindad.

{{< concept-alert type="advertencia" >}}
El orden importa: **LeakyReLU antes del softmax, máscara antes del softmax.** Si enmascaras *después* del softmax (poniendo a cero los pesos de no-vecinos y renormalizando), el resultado numérico es parecido pero el gradiente se contamina: nodos no conectados reciben gradiente a través del softmax. La forma correcta —y la del paper— es inyectar $-\infty$ en los logits, de modo que esas posiciones no participen en absoluto del softmax ni de su backward.
{{< /concept-alert >}}

---

## 3. El grafo de juguete

Trabajamos sobre un grafo de 5 nodos, lo bastante chico para inspeccionar a mano cada coeficiente de atención. Incluimos **self-loops** (la diagonal de $A$ en 1) para que cada nodo se atienda también a sí mismo —práctica estándar en GAT, equivalente al "residual" del message passing.

```
   0 --- 1
   |  \  |
   |   \ |
   2 --- 3 --- 4
```

```python
import numpy as np

# Adyacencia con self-loops (diagonal en 1). Simétrica = grafo no dirigido.
# Aristas: 0-1, 0-2, 0-3, 1-3, 2-3, 3-4
A = np.array([
    [1, 1, 1, 1, 0],
    [1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1],
], dtype=np.float32)                       # (N=5, N=5)

# Features de entrada: 3 dimensiones por nodo (didácticas, valores arbitrarios)
H = np.arange(5 * 3, dtype=np.float32).reshape(5, 3) / 10.0   # (N=5, F_in=3)
```

Verificación de dimensiones que mantendremos en los tres frameworks: con `F_in=3`, `F_out=4` y `K=2` cabezas en modo concat, la salida será `(N, K*F_out) = (5, 8)`. La matriz de atención de cada cabeza es `(N, N) = (5, 5)`, y cada una de sus filas debe sumar 1 con ceros exactos donde no hay arista.

---

## 4. Implementación en PyTorch

La capa es un `nn.Module` con tres tensores de parámetros apilados por cabeza: `W` de forma `(K, F_in, F_out)`, y los dos medios-vectores de atención `a_src`, `a_dst` de forma `(K, F_out, 1)`. El forward recibe las features y la adyacencia, y devuelve las features nuevas más los pesos de atención (para inspeccionarlos).

```python
import torch
import torch.nn as nn

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1, concat=True, slope=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat                      # True: concatena cabezas; False: promedia
        # Un W por cabeza:                  (K, F_in, F_out)
        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        # Vector de atención partido en dos mitades: a = [a_src || a_dst]
        self.a_src = nn.Parameter(torch.empty(n_heads, out_features, 1))   # (K, F_out, 1)
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_features, 1))   # (K, F_out, 1)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky = nn.LeakyReLU(slope)

    def forward(self, h, adj):
        # h:   (N, F_in)      features de entrada
        # adj: (N, N)         1 donde hay arista (incl. self-loops), 0 si no
        N = h.size(0)

        # Paso 1: proyección lineal, una por cabeza -> (K, N, F_out)
        Wh = torch.einsum("nf,kfo->kno", h, self.W)

        # Paso 2: scores por par. a^T[z_i || z_j] = a_src^T z_i + a_dst^T z_j
        s_i = Wh @ self.a_src                       # (K, N, 1)  término "source" (fila i)
        s_j = Wh @ self.a_dst                       # (K, N, 1)  término "destination" (col j)
        # broadcasting: (K, N, 1) + (K, 1, N) -> (K, N, N); e[k,i,j]
        scores = self.leaky(s_i + s_j.transpose(1, 2))         # (K, N, N)

        # Paso 3: máscara con adyacencia (-inf donde NO hay arista) + softmax por filas
        mask = adj.unsqueeze(0) > 0                            # (1, N, N) -> broadcast a (K,N,N)
        scores = scores.masked_fill(~mask, float("-inf"))      # no-vecinos -> -inf
        alpha = torch.softmax(scores, dim=-1)                  # (K, N, N), filas suman 1

        # Paso 4: agregación ponderada -> (K, N, F_out)
        h_out = alpha @ Wh
        if self.concat:                              # capa oculta: concatena las K cabezas
            out = h_out.permute(1, 0, 2).reshape(N, self.n_heads * self.out_features)  # (N, K*F_out)
        else:                                        # capa de salida: promedia las cabezas
            out = h_out.mean(dim=0)                                                    # (N, F_out)
        return out, alpha
```

Probémosla sobre el grafo de juguete:

```python
torch.manual_seed(0)
A_t = torch.tensor(A)
H_t = torch.tensor(H)

layer = GATLayer(in_features=3, out_features=4, n_heads=2, concat=True)
out, alpha = layer(H_t, A_t)

print("salida:", out.shape)            # torch.Size([5, 8])  = (N, K*F_out)
print("atención:", alpha.shape)        # torch.Size([2, 5, 5]) = (K, N, N)
print("filas suman 1:", alpha.sum(-1)) # todas 1.0
print("cero fuera de aristas:", torch.allclose(alpha[:, A_t == 0], torch.zeros(())))  # True
```

```
salida: torch.Size([5, 8])
atención: torch.Size([2, 5, 5])
filas suman 1: tensor([[1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1.]], ...)
cero fuera de aristas: True
```

El `masked_fill(~mask, -inf)` es el corazón de la atención enmascarada: convierte los logits de los no-vecinos en $-\infty$, y el softmax los manda a exactamente 0. La verificación `cero fuera de aristas: True` confirma que ningún nodo atiende a quien no es su vecino.

---

## 5. Implementación en TensorFlow

Estructuralmente idéntica. Usamos `tf.einsum` para la proyección, `tf.nn.leaky_relu`, y `tf.where` para inyectar la máscara (TF no tiene `masked_fill`, así que escribimos el `where` a mano con un tensor lleno de un valor muy negativo).

```python
import tensorflow as tf

class GATLayer(tf.keras.layers.Layer):
    def __init__(self, out_features, n_heads=1, concat=True, slope=0.2):
        super().__init__()
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.slope = slope

    def build(self, input_shape):
        in_f = int(input_shape[-1])
        init = tf.keras.initializers.GlorotUniform()
        self.W     = self.add_weight(shape=(self.n_heads, in_f, self.out_features),
                                     initializer=init, name="W")
        self.a_src = self.add_weight(shape=(self.n_heads, self.out_features, 1),
                                     initializer=init, name="a_src")
        self.a_dst = self.add_weight(shape=(self.n_heads, self.out_features, 1),
                                     initializer=init, name="a_dst")

    def call(self, h, adj):
        # h: (N, F_in)   adj: (N, N)
        N = tf.shape(h)[0]

        # Paso 1: proyección por cabeza -> (K, N, F_out)
        Wh = tf.einsum("nf,kfo->kno", h, self.W)

        # Paso 2: scores por par (a_src^T z_i + a_dst^T z_j)
        s_i = tf.matmul(Wh, self.a_src)                          # (K, N, 1)
        s_j = tf.matmul(Wh, self.a_dst)                          # (K, N, 1)
        scores = tf.nn.leaky_relu(s_i + tf.transpose(s_j, (0, 2, 1)), self.slope)  # (K, N, N)

        # Paso 3: máscara (-inf efectivo donde no hay arista) + softmax por filas
        mask = tf.expand_dims(adj, 0) > 0                        # (1, N, N)
        neg_inf = tf.fill(tf.shape(scores), tf.constant(-1e9, tf.float32))
        scores = tf.where(mask, scores, neg_inf)                 # no-vecinos -> -1e9
        alpha = tf.nn.softmax(scores, axis=-1)                   # (K, N, N)

        # Paso 4: agregación -> (K, N, F_out)
        h_out = tf.matmul(alpha, Wh)
        if self.concat:
            out = tf.reshape(tf.transpose(h_out, (1, 0, 2)),
                             (N, self.n_heads * self.out_features))   # (N, K*F_out)
        else:
            out = tf.reduce_mean(h_out, axis=0)                       # (N, F_out)
        return out, alpha
```

```python
tf.random.set_seed(0)
layer = GATLayer(out_features=4, n_heads=2, concat=True)
out, alpha = layer(tf.constant(H), tf.constant(A))

print("salida:", out.shape)                       # (5, 8)
print("atención:", alpha.shape)                   # (2, 5, 5)
print("filas suman 1:", tf.reduce_sum(alpha, -1).numpy()[0])   # [1. 1. 1. 1. 1.]
print("cero fuera de aristas:",
      np.allclose(alpha.numpy()[:, A == 0], 0))    # True
```

Dos detalles frente a PyTorch: usamos `-1e9` en lugar de `-inf` literal (más seguro numéricamente bajo `tf.where`, y el softmax lo trata como cero efectivo), y los pesos se crean en `build()` con `add_weight(shape=..., initializer=..., name=...)`, la firma de Keras 3.

---

## 6. Implementación en JAX

En JAX el modelo no tiene estado: los parámetros viven en un diccionario y la capa es una **función pura** `(params, h, adj) -> (out, alpha)`. Esto la hace trivialmente diferenciable, `jit`-eable y `vmap`-eable. Usamos `jax.nn.leaky_relu` y `jax.nn.softmax`.

```python
import jax
import jax.numpy as jnp
from jax import random

def init_gat_params(key, in_f, out_f, n_heads):
    k1, k2, k3 = random.split(key, 3)
    # Inicialización tipo Glorot/Xavier (fan_in + fan_out sobre los dos ejes de feature)
    glorot = lambda k, shp: random.normal(k, shp) * jnp.sqrt(2.0 / (shp[-2] + shp[-1]))
    return {
        "W":     glorot(k1, (n_heads, in_f, out_f)),     # (K, F_in, F_out)
        "a_src": glorot(k2, (n_heads, out_f, 1)),        # (K, F_out, 1)
        "a_dst": glorot(k3, (n_heads, out_f, 1)),        # (K, F_out, 1)
    }

def gat_layer(params, h, adj, concat=True, slope=0.2):
    # h: (N, F_in)   adj: (N, N)   -> función PURA, sin estado
    N = h.shape[0]

    # Paso 1: proyección por cabeza -> (K, N, F_out)
    Wh = jnp.einsum("nf,kfo->kno", h, params["W"])

    # Paso 2: scores por par (a_src^T z_i + a_dst^T z_j)
    s_i = Wh @ params["a_src"]                                # (K, N, 1)
    s_j = Wh @ params["a_dst"]                                # (K, N, 1)
    scores = jax.nn.leaky_relu(s_i + jnp.transpose(s_j, (0, 2, 1)), slope)  # (K, N, N)

    # Paso 3: máscara (-inf donde no hay arista) + softmax por filas
    mask = adj[None, :, :] > 0                               # (1, N, N)
    scores = jnp.where(mask, scores, -jnp.inf)               # no-vecinos -> -inf
    alpha = jax.nn.softmax(scores, axis=-1)                  # (K, N, N)

    # Paso 4: agregación -> (K, N, F_out)
    h_out = alpha @ Wh
    if concat:
        out = jnp.transpose(h_out, (1, 0, 2)).reshape(N, -1)  # (N, K*F_out)
    else:
        out = jnp.mean(h_out, axis=0)                         # (N, F_out)
    return out, alpha
```

```python
params = init_gat_params(random.PRNGKey(0), in_f=3, out_f=4, n_heads=2)
out, alpha = gat_layer(params, jnp.array(H), jnp.array(A), concat=True)

print("salida:", out.shape)                  # (5, 8)
print("atención:", alpha.shape)              # (2, 5, 5)
print("filas suman 1:", alpha.sum(-1)[0])    # [1. 1. 1. 1. 1.]
print("cero fuera de aristas:",
      np.allclose(np.array(alpha)[:, A == 0], 0))   # True
```

Como la capa es pura, conseguir gradientes respecto a los parámetros es solo envolverla en una pérdida y aplicar `jax.grad`; compilar todo el forward es un `@jax.jit` arriba de la función. Nada de estado escondido, nada de tapes: la atención en grafos se siente nativa.

{{< concept-alert type="recordar" >}}
Cuidado con `-jnp.inf` cuando un nodo pueda quedar **sin vecinos** (fila de adyacencia toda en cero): el softmax de una fila de puros $-\infty$ da `NaN`. En nuestro grafo cada nodo tiene self-loop, así que nunca ocurre. En grafos reales, garantiza siempre los self-loops (suma la identidad a $A$) o usa un valor finito grande como `-1e9` —el mismo truco que en TensorFlow— para que la fila siga siendo normalizable.
{{< /concept-alert >}}

---

## 7. Interpretabilidad: leer los pesos de atención aprendidos

La gran ventaja didáctica de GAT es que $\alpha$ es **inspeccionable**: para cada nodo podemos ver a qué vecinos les dio más peso. A diferencia de los coeficientes fijos de GCN, estos los aprendió la red. Tomemos la cabeza 0 del modelo PyTorch y miremos la fila del nodo 3 (el más conectado: vecinos 0, 1, 2, 3, 4):

```python
import torch

a0 = alpha[0]                         # cabeza 0: (N, N)
for i in range(5):
    vecinos = (A_t[i] > 0).nonzero().flatten().tolist()
    pesos = {j: round(a0[i, j].item(), 3) for j in vecinos}
    print(f"nodo {i} atiende a {pesos}")
```

```
nodo 0 atiende a {0: 0.211, 1: 0.231, 2: 0.262, 3: 0.296}
nodo 1 atiende a {0: 0.303, 1: 0.310, 3: 0.387}
nodo 2 atiende a {0: 0.311, 2: 0.327, 3: 0.362}
nodo 3 atiende a {0: 0.187, 1: 0.192, 2: 0.196, 3: 0.201, 4: 0.224}
nodo 4 atiende a {3: 0.494, 4: 0.506}
```

Con pesos aleatorios iniciales la distribución es casi uniforme (todos los vecinos pesan parecido), pero **la estructura ya es correcta**: cada fila solo reparte masa entre los vecinos reales y suma 1. Al entrenar sobre una tarea (clasificación de nodos, por ejemplo), estos coeficientes se desbalancean y revelan qué vecinos resultaron informativos —es la lectura de interpretabilidad que GAT habilita y GCN no. Una forma habitual de visualizarlo es dibujar el grafo con el grosor de cada arista proporcional a $\alpha_{ij}$.

{{< concept-alert type="clave" >}}
Que la atención sea interpretable **no la vuelve automáticamente una explicación causal** —misma cautela que con la atención en NLP. Un peso alto $\alpha_{ij}$ dice "este vecino contribuyó mucho a la representación de $i$", no "este vecino *causa* la etiqueta de $i$". Es una pista de inspección valiosa, no una prueba.
{{< /concept-alert >}}

---

## 8. La conexión clave: un Transformer es un GAT sobre el grafo completo

Aquí cierra el arco. Mira otra vez el cálculo de la atención GAT y compáralo con la self-attention que vimos en la [Clase 14](/clases/clase-14):

| | GAT (esta página) | Self-attention (Transformer) |
|---|---|---|
| Entidades | nodos del grafo | tokens de la secuencia |
| Proyección | $z_i = W^\top h_i$ | $q_i, k_j, v_j$ desde $h$ |
| Score por par | $e_{ij} = \mathrm{LeakyReLU}(\vec{a}^\top[z_i \Vert z_j])$ | $e_{ij} = q_i^\top k_j / \sqrt{d}$ |
| Normalización | $\mathrm{softmax}_j$ por filas | $\mathrm{softmax}_j$ por filas |
| **A quién se atiende** | **solo a los vecinos** (máscara = $A$) | **a todos los tokens** (sin máscara) |
| Agregación | $\sum_j \alpha_{ij}\, z_j$ | $\sum_j \alpha_{ij}\, v_j$ |
| Multi-head | sí | sí |

Las dos arquitecturas son **el mismo mecanismo**: proyectar, puntuar por pares, softmax por filas, combinar; ambas con múltiples cabezas. La **única** diferencia estructural es la máscara de adyacencia. GAT enmascara a los no-vecinos; el Transformer no enmascara nada (o enmascara solo el futuro, en el caso causal). Dicho de otro modo:

> **Un Transformer es un GAT donde el grafo es completo: todos los tokens son vecinos de todos.**

Comprobémoslo numéricamente. Si pasamos a nuestra capa GAT una adyacencia toda-en-uno —el grafo completo— el resultado es self-attention pura, sin un solo cero:

```python
A_full = torch.ones(5, 5)               # grafo completo: todos atienden a todos
out_full, alpha_full = layer(H_t, A_full)

print("atención sobre grafo completo, fila del nodo 0:")
print(alpha_full[0, 0])                 # SIN ceros: el nodo 0 atiende a los 5 nodos
print("filas suman 1:", alpha_full[0].sum(-1))
```

```
atención sobre grafo completo, fila del nodo 0:
tensor([0.067, 0.105, 0.165, 0.258, 0.405], ...)   # masa sobre TODOS los nodos
filas suman 1: tensor([1., 1., 1., 1., 1.], ...)
```

Compáralo con la fila del paso 7 (`nodo 0 atiende a {0,1,2,3}` con un cero en la posición 4): la **misma capa, el mismo código**, solo cambió la máscara. Cuando $A$ codifica un grafo disperso obtenemos una GNN inductiva; cuando $A$ es la matriz de unos obtenemos un Transformer. La self-attention no es una idea distinta de la atención en grafos —es su **caso límite**, el del grafo totalmente conectado. Por eso a los Transformers se les llama a veces "GNN sobre grafos completos", y por eso técnicas de un mundo (atención dispersa, máscaras estructuradas) migran con naturalidad al otro.

{{< concept-alert type="clave" >}}
Esta equivalencia tiene una consecuencia práctica enorme: cualquier *sesgo estructural* que quieras inyectar en un Transformer se reduce a **diseñar la máscara**. Atención causal (un token solo ve el pasado), atención local por ventana, atención dispersa para secuencias largas, o atención guiada por un grafo de conocimiento: todas son la misma capa con un patrón de adyacencia distinto. GAT y self-attention son dos nombres para la misma maquinaria; lo único que negocias es la topología del grafo sobre el que corre.
{{< /concept-alert >}}

---

## 9. Comparación lado a lado de los tres frameworks

| Concepto | PyTorch | TensorFlow | JAX |
|---|---|---|---|
| Parámetros | `nn.Parameter` en `__init__` | `add_weight` en `build()` | dict en `init_gat_params` |
| Proyección por cabeza | `torch.einsum("nf,kfo->kno", ...)` | `tf.einsum("nf,kfo->kno", ...)` | `jnp.einsum("nf,kfo->kno", ...)` |
| LeakyReLU | `nn.LeakyReLU(slope)` | `tf.nn.leaky_relu(x, slope)` | `jax.nn.leaky_relu(x, slope)` |
| Máscara de adyacencia | `scores.masked_fill(~mask, -inf)` | `tf.where(mask, scores, -1e9)` | `jnp.where(mask, scores, -jnp.inf)` |
| Softmax por filas | `torch.softmax(s, dim=-1)` | `tf.nn.softmax(s, axis=-1)` | `jax.nn.softmax(s, axis=-1)` |
| Concat de cabezas | `permute(1,0,2).reshape(N, -1)` | `transpose+reshape` | `transpose+reshape` |
| Estado del modelo | mutable (`nn.Module`) | mutable (`Layer`) | sin estado (función pura) |

La lectura: el algoritmo es **el mismo** en los tres; lo que cambia es dónde viven los parámetros (atributos del módulo vs. diccionario explícito) y la sintaxis del enmascarado. JAX hace más visible que una capa GAT no es más que una función matemática de `(params, h, adj)`; PyTorch y TF la envuelven en objetos con estado. Cualquiera de las tres produce la misma matriz de atención sobre el mismo grafo.

---

## 10. Cómo seguir

1. **Apila dos capas GAT** (oculta con `concat=True`, salida con `concat=False` + softmax) y entrena clasificación de nodos sobre Cora; compara la accuracy con una GCN de la misma profundidad.
2. **Visualiza la atención entrenada**: dibuja el grafo con el grosor de arista $\propto \alpha_{ij}$ y observa qué vecinos pesan más tras entrenar.
3. **Quita los self-loops** y comprueba que un nodo aislado produce `NaN` en JAX con `-jnp.inf`; arréglalo con `-1e9`. Es el gotcha de la sección 6 en vivo.
4. **Pasa al grafo completo** (`A = ones`) y verifica que recuperas exactamente la self-attention de la [Clase 14](/clases/clase-14): añade el escalado $1/\sqrt{d}$ y el producto punto $q^\top k$ en vez del `LeakyReLU(a^T[..])` y tendrás un bloque de Transformer.
5. **Implementa GATv2** (Brody et al., 2021), que corrige la "atención estática" de GAT moviendo la $W$ dentro del LeakyReLU: $e_{ij} = \vec{a}^\top \mathrm{LeakyReLU}(W[h_i \Vert h_j])$. Un cambio de una línea con efecto medible.

---

## 11. Cross-links

- [Clase 27 - Redes neuronales de grafos](/clases/clase-27): la clase que enmarca GCN, GraphSAGE y GAT dentro del paso de mensajes.
- [Fundamento: Redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos): el paso de mensajes, las tareas de grafo y la familia completa de modelos GNN.
- [Fundamento: Mecanismo de atención](/fundamentos/mecanismo-atencion): la atención query-key-value que GAT trasplanta al grafo y que el Transformer usa sobre el grafo completo.
- [Paper GAT (Veličković et al., 2018)](/papers/gat-velickovic-2018): el paper canónico que implementamos aquí, con los resultados en Cora, Citeseer, Pubmed y PPI.
- [Clase 14 - Transformers](/clases/clase-14): la self-attention que esta página revela como un GAT sobre el grafo completo.

---

**Ver también:** [Teoría - Clase 27](/clases/clase-27/teoria) · [Profundización - Clase 27](/clases/clase-27/profundizacion).
