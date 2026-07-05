---
title: "Lab 27 - Redes Neuronales de Grafos con PyTorch Geometric"
weight: 270
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga
**Fecha:** Junio 2026
**Notebook origen:** `clase_27/material/Laboratorio/Laboratorio_27.ipynb`
**Notebook ejecutado:** [lab27.ipynb](/notebooks/lab27.ipynb) · [HTML](/notebooks-html/lab27.html)

## Encuadre

La contraparte práctica de la [clase 27](/clases/clase-27): pasar de la teoría del **message passing** a un pipeline funcional con [**PyTorch Geometric**](https://pytorch-geometric.readthedocs.io/) (PyG), la librería estándar para GNN. El lab recorre los **tres niveles de tarea** en grafos y cierra con seis actividades conceptuales.

- **Nivel nodo** — clasificar nodos *dentro* de un grafo: el club de karate de Zachary (juguete) y **Cora** (red de citas de papers).
- **Nivel grafo** — clasificar grafos *enteros*: **MUTAG**, moléculas etiquetadas por mutagenicidad.
- **Trucos de producción** — Cluster-GCN para grafos que no caben en GPU, y cómo construir tus propios `Data`/`DataLoader`.

La tesis del lab, demostrada con un experimento controlado, es simple y contundente: **la estructura relacional es información, y las GNN la convierten en accuracy**.

| Pieza | Implementación en el lab |
|---|---|
| Formato de grafo | `torch_geometric.data.Data` (`x`, `edge_index` en COO, `y`) |
| Capa GCN | `GCNConv` (Kipf & Welling 2017) — agregación normalizada + self-loops |
| Capa Morris | `GraphConv` (Morris et al. 2018) — suma sin normalizar + skip-connection |
| Readout (nivel grafo) | `global_mean_pool` con vector `batch` |
| Datasets | `KarateClub`, `Planetoid` (Cora/PubMed), `TUDataset` (MUTAG) |
| Escalado | `ClusterData` + `ClusterLoader` (Cluster-GCN, Chiang et al. 2019) |

## Resultados consolidados

Notebook ejecutado en GPU (torch 2.11 + CUDA 12.8). Los números clave:

### El experimento central — ¿aporta el grafo? (Cora, misma arquitectura)

| Modelo | Usa el grafo | Test Accuracy |
|---|---|---|
| **MLP** (solo features bag-of-words) | ❌ | **57.4%** |
| **GCN** (features + `edge_index`) | ✅ | **81.3%** |

→ **+24 puntos** absolutos cambiando una sola línea (`Linear` → `GCNConv`), con los mismos features, mismos hiperparámetros y mismo seed. Toda la ganancia es atribuible a *dejar que el modelo mire las citas*.

### Expresividad del agregador (MUTAG, clasificación de grafos)

| Capa | Agregación | Test Accuracy |
|---|---|---|
| **GCNConv** (Kipf) | promedio normalizado | **71.1%** |
| **GraphConv** (Morris) | suma + skip-connection | **81.6%** |

→ **+10.5 puntos** porque la **suma preserva el grado** (cuántos vecinos tiene cada nodo), información que el promedio descarta. Es el argumento teórico de GIN/Weisfeiler-Lehman hecho número.

### Otros hitos medidos

- **Karate Club:** la GCN agrupa nodos por comunidad **antes de entrenar** (pesos aleatorios) — el *inductive bias* estructural en acción.
- **Cluster-GCN:** particiona PubMed (19.717 nodos) en 128 subgrafos con METIS para entrenar por minibatches.

## Bloques del lab

{{< cards >}}
  {{< card link="exploracion-grafos" title="Explorando grafos con PyG" subtitle="El objeto Data, edge_index en COO, features one-hot, grado promedio, Karate Club y su visualización" icon="academic-cap" >}}
  {{< card link="clasificacion-nodos" title="Clasificación de nodos" subtitle="GCN en Karate (bias inductivo sin entrenar) + el experimento MLP vs GCN en Cora (+24 pts)" icon="beaker" >}}
  {{< card link="clasificacion-grafos" title="Clasificación de grafos" subtitle="MUTAG, batching de grafos disconexos, global_mean_pool, y la mejora de Morris (+10.5 pts)" icon="beaker" >}}
  {{< card link="trucos-practicos" title="Trucos de producción" subtitle="Cluster-GCN para grafos enormes, construir tus propios Data/DataLoader, catálogo de capas PyG" icon="cog" >}}
  {{< card link="actividades" title="Actividades (1-6) resueltas" subtitle="Bias inductivo, self-loops, nodos aislados, grado promedio, dimensiones de GCNConv, campo receptivo" icon="document-text" >}}
{{< /cards >}}

## Papers relacionados

{{< cards >}}
  {{< card link="/papers/gcn-kipf-2017" title="GCN (2017)" subtitle="Kipf & Welling — la GCNConv del lab: agregación normalizada simétrica con self-loops" icon="document-text" >}}
  {{< card link="/papers/mpnn-gilmer-2017" title="MPNN (2017)" subtitle="Gilmer et al. — el framework de message passing que generaliza toda GNN" icon="document-text" >}}
  {{< card link="/papers/graphsage-hamilton-2017" title="GraphSAGE (2017)" subtitle="Hamilton et al. — SAGEConv, una de las capas alternativas del catálogo PyG" icon="document-text" >}}
  {{< card link="/papers/gat-velickovic-2018" title="GAT (2018)" subtitle="Veličković et al. — atención sobre grafos, base de TransformerConv" icon="document-text" >}}
  {{< card link="/papers/ggnn-li-2015" title="GGNN (2015)" subtitle="Li et al. — GatedGraphConv, la GNN con GRU del catálogo PyG" icon="document-text" >}}
  {{< card link="/papers/logical-expressiveness-barcelo-2020" title="Expresividad lógica (2020)" subtitle="Barceló et al. (PUC) — qué puede y no puede distinguir una GNN" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/redes-neuronales-de-grafos" title="Redes Neuronales de Grafos" subtitle="La familia completa: notación, tareas, modelos, problemas y aplicaciones" icon="book-open" >}}
  {{< card link="/fundamentos/message-passing" title="Message Passing" subtitle="El mecanismo: las 4 etapas, forma matricial, invarianza a permutación, campo receptivo" icon="book-open" >}}
  {{< card link="/fundamentos/expresividad-gnn" title="Expresividad de GNN" subtitle="Weisfeiler-Lehman, por qué sum > mean, los límites de lo que una GNN distingue" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-27" title="Clase 27 - Teoría" subtitle="GNN: message passing, GCN/GGNN/GraphSAGE, R-GCN, GraphNav, MPNN, expresividad" icon="academic-cap" >}}
  {{< card link="/clases/clase-27/profundizacion" title="Profundización" subtitle="Math: forma matricial, GCN espectral (Laplaciano/Chebyshev), GAT, WL/FOC2, over-smoothing" icon="beaker" >}}
  {{< card link="/dominios/estructurados" title="Dominio: Datos estructurados" subtitle="GNN sobre grafos, la familia relacional" icon="globe-alt" >}}
  {{< card link="/laboratorios/lab-26" title="Lab 26 - Meta-aprendizaje (anterior)" subtitle="MAML y Prototypical Networks few-shot" icon="academic-cap" >}}
  {{< card link="/laboratorios/lab-28" title="Lab 28 - Aprendizaje Autosupervisado: UDA (siguiente)" subtitle="Semi-supervisión por consistencia sobre IMDB con 20 etiquetas" icon="arrow-right" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. 5 páginas temáticas que recorren los tres niveles de tarea en grafos (nodo, grafo, producción) y las 6 actividades resueltas y verificadas contra el notebook ejecutado. Resultados medidos en GPU: MLP 57.4% vs GCN 81.3% en Cora (+24 pts por usar el grafo); GCNConv 71.1% vs GraphConv 81.6% en MUTAG (+10.5 pts por sumar en vez de promediar). Notebook ejecutado con torch 2.11 + CUDA 12.8.
