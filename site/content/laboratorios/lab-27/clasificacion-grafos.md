---
title: "Clasificación de grafos: pooling, batching y expresividad"
weight: 3
---

Hasta ahora clasificábamos **nodos** dentro de un grafo. Ahora cambia el paradigma: clasificar **grafos enteros**. Cada ejemplo es un grafo completo y queremos una etiqueta por grafo. Esto introduce dos conceptos nuevos —el **batching de grafos** y el **readout**— y cierra con la sección más profunda del lab: por qué el agregador importa.

## El dataset MUTAG

**MUTAG** son 188 moléculas representadas como grafos: nodos = átomos (one-hot del tipo: C, N, O…, 7 tipos), aristas = enlaces químicos, y **una etiqueta por grafo** (2 clases).

```python
dataset = TUDataset(root='data/TUDataset', name='MUTAG')
# Number of graphs: 188 | Number of features: 7 | Number of classes: 2
```

{{< callout type="warning" >}}
**Errata del enunciado.** El notebook dice que MUTAG clasifica moléculas "según si inhiben o no la replicación del VIH". Es **incorrecto**: MUTAG (Debnath et al., 1991) clasifica **mutagenicidad** de compuestos nitroaromáticos sobre *Salmonella typhimurium* (test de Ames). El dataset del VIH es otro (HIV/MoleculeNet). En la entrega conviene describirlo como "predicción de mutagenicidad".
{{< /callout >}}

`TUDataset` es la colección de benchmarks de grafos de TU Dortmund.

## El batching de grafos: un solo grafo gigante disconexo

Aquí hay un concepto **no obvio**. ¿Cómo se arma un *batch* de grafos de tamaños distintos? Una imagen siempre es 28×28; las moléculas tienen distinto número de átomos.

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**La solución de PyG es elegante:** apila los grafos del batch en **un solo grafo gigante disconexo** —concatena todos los nodos y aristas sin conectarlos entre grafos— y agrega un vector **`batch`** donde `batch[i]` = índice del grafo al que pertenece el nodo `i`. Como los subgrafos no comparten aristas, el message passing **nunca mezcla** información entre moléculas distintas. Así "procesar N grafos" se convierte en "procesar 1 grafo disperso", aprovechando al máximo la GPU.

El split usa `dataset.shuffle()` + slicing: 150 grafos de train, 38 de test.

## El readout: de embeddings por nodo a un vector por grafo

```python
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)             # [batch_size, hidden] ← readout
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)
```

El paso que **no existía** en clasificación de nodos es el **readout**: `global_mean_pool(x, batch)`. Tras 3 capas de message passing tenemos un embedding **por átomo**, pero queremos un vector **por molécula**. `global_mean_pool` **promedia los embeddings de todos los nodos de cada grafo** (usando `batch` para agruparlos), produciendo `[batch_size, hidden_channels]`. Recién ahí el `Linear` clasifica cada molécula.

El pooling debe ser **permutation-invariant** (mean, sum, max): el resultado no puede depender del orden arbitrario en que se listen los átomos. `mean` cumple.

{{< callout type="info" >}}
**Los tres niveles de tarea en GNN:**
- **Nivel nodo** (Karate, Cora): salida por nodo, sin readout.
- **Nivel grafo** (MUTAG): readout que colapsa nodos → un vector por grafo.
- **Nivel arista** (link prediction, no en este lab): predice si dos nodos deberían estar conectados — conceptualmente, el problema de *record linkage*: ¿existe una arista "es la misma persona" entre dos registros?
{{< /callout >}}

Con 3 capas + pooling + 64 hidden y solo 150 grafos de train:

```text
Test Acc: 0.7105
```

## Morris et al. (2018): por qué el agregador importa

Esta sección "opcional" es la **más profunda del lab**, y conecta con los papers GIN (Xu 2019) y Morris. Compará las dos ecuaciones:

**GCN (Kipf):**
$$
\mathbf{h}_i^{(t+1)} = \mathbf{W}^{(t+1)} \sum_{j \in \mathcal{N}(i)\cup\{i\}} \tfrac{1}{c_{i,j}}\,\mathbf{h}_j^{(t)}
$$

**GraphConv (Morris):**
$$
\mathbf{h}_i^{(t+1)} = \mathbf{W}_A^{(t+1)}\,\mathbf{h}_i^{(t)} + \mathbf{W}_B^{(t+1)} \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(t)}
$$

Dos cambios cruciales:

1. **Dos matrices separadas** ($\mathbf{W}_A$ para uno mismo, $\mathbf{W}_B$ para los vecinos) en vez de mezclar el self-loop dentro de la misma suma. Desacopla "qué soy yo" de "qué son mis vecinos".
2. **Quita la normalización $\frac{1}{c_{i,j}}$** y usa **suma** en vez de promedio.

En PyG, cambiar de capa es cambiar una línea:

```python
from torch_geometric.nn import GraphConv   # en vez de GCNConv
self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
```

```text
Test Acc: 0.8158
```

| Capa | Agregación | Test Acc |
|---|---|---|
| GCNConv | promedio normalizado | 71.1% |
| GraphConv | suma + skip-connection | **81.6%** |

**+10.5 puntos.** ¿Por qué? El **promedio pierde la información del grado**. Un nodo con 2 vecinos idénticos y otro con 5 vecinos idénticos producen **el mismo promedio** → la GCN no los distingue. La **suma** sí ($2\mathbf{h} \neq 5\mathbf{h}$). Para tareas donde la estructura/conteo importa (contar enlaces en una molécula), **sum > mean**.

{{< callout type="info" >}}
**Expresividad y Weisfeiler-Lehman.** Este es el argumento de **GIN** (Xu et al. 2019): la agregación por suma es **máximamente expresiva**, equivalente en poder al test de isomorfismo de grafos de Weisfeiler-Lehman. La intuición operativa: si tu agregador descarta información (como el promedio descarta el grado), tu red **no puede** aprender funciones que dependan de ella, por más que la entrenes. El paper de [Barceló et al. (PUC, 2020)](/papers/logical-expressiveness-barcelo-2020) formaliza qué propiedades lógicas puede o no expresar una GNN. Ver también el fundamento de [expresividad de GNN](/fundamentos/expresividad-gnn).
{{< /callout >}}

---

Cerrados los dos niveles de tarea (nodo y grafo), la última sección reúne los **trucos para llevar las GNN al mundo real**. Sigue en [trucos de producción](trucos-practicos).
