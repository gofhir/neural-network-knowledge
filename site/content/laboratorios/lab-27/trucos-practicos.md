---
title: "Trucos de producción"
weight: 4
---

Sección breve "de bolsillo" con tres recetas para llevar las GNN del notebook al mundo real: entrenar grafos que no caben en GPU, construir tus propios datasets, y el catálogo de capas listas para combinar.

## Grafos enormes → Cluster-GCN

El problema: una GCN procesa el grafo **entero** en cada forward pass (todos los nodos, todas las aristas). Si tiene millones de nodos, no cabe en GPU. Y no podés simplemente "tomar un batch de nodos" porque al sacar un nodo perdés sus aristas y rompés el message passing.

**Cluster-GCN (Chiang et al., 2019)** resuelve esto con una idea simple:

```python
from torch_geometric.loader import ClusterData, ClusterLoader

cluster_data = ClusterData(data, num_parts=128)              # 1. particiona en 128 subgrafos
train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)   # 2. batch = 32 subgrafos
# Computing METIS partitioning... Done!
```

1. **Particiona** el grafo en muchos subgrafos densos con un algoritmo de partición (METIS / Louvain — por eso se instaló `python-louvain`). El truco: agrupar nodos **muy conectados entre sí**, de modo que cada subgrafo conserve la mayoría de sus aristas internas.
2. Cada **minibatch** es un puñado de esos subgrafos. Como cada uno es chico, cabe en GPU; como es internamente denso, se pierden **pocas** aristas (solo las que cruzan particiones).

Una vez creado, el `ClusterLoader` se usa **igual que un DataLoader normal** — toda la complejidad queda encapsulada. El lab lo demuestra sobre **PubMed** (19.717 nodos).

{{< callout type="info" >}}
Es la misma filosofía del *blocking* en record linkage: cuando el espacio completo no es tratable, lo **particionás en bloques manejables** que preservan la mayoría de los pares relevantes y descartás los cruces improbables. Cluster-GCN bloquea por conectividad; un MDM bloquea por claves de similitud.
{{< /callout >}}

## Datos propios → construir tu propio `Data`

El template del notebook usa `...` (placeholders) y **no es ejecutable tal cual** —`torch.tensor([...])` lanza `Could not infer dtype of ellipsis`, porque `...` es el objeto `Ellipsis` de Python, no datos. El patrón funcional, con un grafo mínimo de 4 nodos:

```python
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# 4 nodos, cada uno con 2 features
nodes_features = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[0.5,0.5]], dtype=torch.float)  # [4, 2]

# aristas COO [2, num_edges]: fila 0 = origen, fila 1 = destino
# no dirigido 0-1, 1-2, 2-3 → cada arista en ambas direcciones
edges = torch.tensor([[0,1,1,2,2,3],
                      [1,0,2,1,3,2]], dtype=torch.long)   # [2, 6]

y = torch.tensor([0,1,1,0], dtype=torch.long)            # una clase por nodo

data = Data(x=nodes_features, edge_index=edges, y=y)
data.train_mask = torch.tensor([True, True, False, False])   # atributos a medida
data.test_mask  = torch.tensor([False, False, True, True])

print(data)
# Data(x=[4, 2], edge_index=[2, 6], y=[4], train_mask=[4], test_mask=[4])
```

Reglas para que no falle:

1. **`x`** debe ser `float` (entra a capas lineales). **`edge_index`** debe ser `torch.long` (son índices). **`y`** suele ser `long` para clasificación.
2. **`edge_index` es `[2, num_edges]`**, no `[num_edges, 2]`. Fila 0 = origen, fila 1 = destino.
3. Si el grafo es **no dirigido**, cada arista va **dos veces** (`i→j` y `j→i`).

Para **muchos grafos** (como MUTAG), se crea una lista y se la pasa al `DataLoader`:

```python
data_list = [Data(...), Data(...), ...]   # un Data por grafo
loader = DataLoader(data_list, batch_size=32)
```

## Arquitecturas a medida → el catálogo de capas de PyG

PyG es un **lego de capas**. Las arquitecturas vistas en clase ya están implementadas y listas para combinar:

| Capa en PyG | Paper de la clase |
|---|---|
| `GCNConv` | [Kipf & Welling 2017](/papers/gcn-kipf-2017) |
| `SAGEConv` | [GraphSAGE — Hamilton 2017](/papers/graphsage-hamilton-2017) |
| `GatedGraphConv` | [GGNN — Li 2015](/papers/ggnn-li-2015) |
| `GATConv` / `TransformerConv` | [GAT — Veličković 2018](/papers/gat-velickovic-2018) |

El mensaje pedagógico: la teoría de la clase ya está implementada; cambiar de arquitectura es cambiar una línea — exactamente como pasar de `GCNConv` a `GraphConv` en [clasificación de grafos](clasificacion-grafos).

---

Con las herramientas de producción cubiertas, solo queda resolver y justificar las seis actividades de cierre. Sigue en [actividades](actividades).
