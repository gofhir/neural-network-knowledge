---
title: "Explorando grafos con PyTorch Geometric"
weight: 1
---

Antes de entrenar nada, hay que entender **cómo se representa un grafo en código**. Toda la sección gira en torno a una sola clase —`Data`— y a un dataset de juguete —el club de karate de Zachary— que sirve para fijar conceptos que reaparecen en cada experimento posterior.

## La instalación no es trivial: el baile de versiones

```python
torch_ver = torch.__version__.split('+')[0]              # '2.11.0'
cuda_tag  = 'cpu' if torch.version.cuda is None else 'cu' + ''.join(torch.version.cuda.split('.'))
index_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
```

PyTorch Geometric no es PyTorch puro. Sus primitivas centrales (`torch-scatter`, `torch-sparse`, `torch-cluster`) son **extensiones compiladas en C++/CUDA**, atadas a una versión exacta de la ABI de PyTorch y del toolkit CUDA. Un wheel compilado para `torch 2.1 + cu118` corriendo sobre `torch 2.11 + cu128` produce `undefined symbol` o un segfault al importar. Por eso el código **detecta** la versión instalada y construye dinámicamente la URL del índice de wheels precompilados que coincide.

{{< callout type="info" >}}
**La operación *scatter* es el corazón de toda GNN.** Cuando un nodo agrega mensajes de sus vecinos, internamente hay un tensor de mensajes (uno por arista) que hay que sumar **agrupados por nodo destino**: eso es `scatter_add(messages, dst_index)`. Hacerlo eficiente sobre grafos dispersos en GPU es lo que justifica la extensión compilada `torch-scatter`. Reaparece, conceptualmente, en cada `GCNConv`.
{{< /callout >}}

`python-louvain` se instala para la detección de comunidades que usa el `ClusterLoader` (ver [trucos de producción](trucos-practicos)).

## El dataset Karate Club

El **Zachary's Karate Club** (1977) es el "hello world" de las redes sociales. Wayne Zachary observó durante tres años un club de karate de 34 miembros y registró una arista entre dos personas cuando interactuaban **fuera** de las clases. Durante el estudio, un conflicto entre el instructor (*Mr. Hi*, nodo 0) y el administrador (*Officer*, nodo 33) **partió el club en dos facciones** — y la estructura del grafo de amistades **predice** a qué bando se fue cada persona. Es el caso canónico de "la topología contiene la señal".

![Karate Club original — diagrama del paper de Zachary 1977](/images/lab27-c11.png)

## El objeto `Data`: cómo PyG guarda un grafo

```python
from torch_geometric.datasets import KarateClub
dataset = KarateClub()
data = dataset[0]   # único grafo
```

Un grafo vive en un objeto **`Data`** con estos campos clave:

| Atributo | Qué es | En Karate Club |
|---|---|---|
| `x` | features de nodos `[num_nodes, num_features]` | `[34, 34]` (one-hot) |
| `edge_index` | aristas `[2, num_edges]` (formato COO) | `[2, 156]` |
| `y` | labels de nodos | `[34]`, 4 clases |

El bloque diagnóstico imprime estadísticas que **anticipan tres de las seis actividades**:

```text
Number of nodes: 34
Number of edges: 156
Average node degree: 4.59
Contains isolated nodes: False
Contains self-loops: False
Is undirected: True
```

`has_isolated_nodes()`, `has_self_loops()` e `is_undirected()` no son adornos: el self-loop es el "truco" de las [actividades 2 y 3](actividades), y el grado promedio es la [actividad 4](actividades).

## `edge_index`: el formato COO

```python
print(edge_index.t()[10:30,:])
# tensor([[ 1,  2], [ 1,  3], [ 1,  7], ...])
```

`edge_index` tiene shape `[2, num_edges]`: la fila 0 son los nodos **origen**, la fila 1 los **destino**. Es el formato **COO (coordinate)** de una matriz de adyacencia dispersa. `.t()` lo transpone a `[num_edges, 2]` para leer cada arista como un par `(src, dst)`.

{{< callout type="warning" >}}
**Hint clave para la [Actividad 4](actividades).** Como el grafo es **no dirigido**, PyG almacena cada amistad **dos veces**: `(i,j)` y `(j,i)`. Por eso `num_edges = 156` aunque haya solo 78 amistades. Esto hace que `data.num_edges / data.num_nodes` dé el **grado promedio correcto sin dividir por 2**: la suma de grados de todos los nodos es exactamente `num_edges` bajo esta convención.
{{< /callout >}}

## Features one-hot: nodos sin información propia

```python
features = data.x   # matriz identidad 34×34
# tensor([[1., 0., 0., ...], [0., 1., 0., ...], ...])
```

Los features son **one-hot** (la matriz identidad): el nodo *i* tiene un 1 en la posición *i* y ceros en el resto. Esto es **deliberado y conceptualmente central**: significa que los nodos **no traen información propia útil** — cada uno es solo "su identidad", y todos son ortogonales entre sí. Toda señal que la GNN extraiga tendrá que venir de la **estructura** (quién está conectado con quién), no de los atributos. Esta es la clave de la [Actividad 1](actividades).

## Visualización del grafo

```python
from torch_geometric.utils import to_networkx
G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)
```

`to_networkx()` convierte el `Data` a un grafo de **NetworkX**, y `nx.spring_layout` lo dibuja con un **layout de resortes** (force-directed): trata las aristas como muelles y los nodos como cargas que se repelen, hasta el equilibrio. Por eso los nodos conectados aparecen cerca.

![Grafo Karate Club coloreado por facción](/images/lab27-c22.png)

El "ordenamiento" que se ve **lo impone el algoritmo de layout** (que sí mira las aristas), no un orden intrínseco de los datos. Mantené esta distinción presente: la [Actividad 1](actividades) pregunta por un fenómeno distinto pero emparentado — el ordenamiento que produce la **GCN sin entrenar**.

---

Con la representación clara —`Data`, COO, features one-hot, grado promedio— ya podemos construir la primera red. Sigue en [clasificación de nodos](clasificacion-nodos).
