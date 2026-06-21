---
title: "Practica desde 0 - Redes Neuronales de Grafos"
weight: 30
sidebar:
  open: true
---

La clase 27 cubre las **Redes Neuronales de Grafos (GNN)**: cómo aprender sobre datos con estructura relacional propagando información entre nodos vecinos. Esta práctica implementa el mecanismo y dos modelos emblemáticos en **mínima escala**, sin librerías especializadas (nada de `torch_geometric` ni `DGL`), usando solo la **matriz de adyacencia densa** sobre grafos de juguete — para entender por dentro qué hace cada pieza, no solo leer los papers. Partimos del **message passing genérico** ($H' = \hat{A}HW$), seguimos con la **GCN** de Kipf & Welling para clasificación semi-supervisada de nodos, y cerramos con la **atención en grafos (GAT)**, que conecta directamente con los Transformers. Cada camino replica el mismo modelo en **triple framework** (PyTorch, TensorFlow y JAX) para ver cómo cada uno expresa las mismas ideas.

## Caminos

{{< cards >}}
  {{< card link="01-message-passing-desde-cero" title="01 - Message Passing desde cero" subtitle="Las 4 etapas y la propagación matricial Â·H·W en PyTorch, TensorFlow y JAX" icon="code" >}}
  {{< card link="02-gcn-desde-cero" title="02 - GCN desde cero" subtitle="GCN de 2 capas para clasificación semi-supervisada (Karate Club) en triple framework" icon="code" >}}
  {{< card link="03-gat-desde-cero" title="03 - GAT desde cero" subtitle="Atención en grafos y la conexión Transformer = GAT sobre grafo completo" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 14 - Transformer desde 0](../../clase-14/practica): la self-attention es message passing sobre un grafo completo; el camino 03 (GAT) lo hace explícito.
- [Clase 15 - Mecanismo de atención](../../clase-15): la base de la atención que GAT lleva a los grafos.
- Python intermedio y NumPy; PyTorch básico (tensores, `nn.Module`, autograd, training loop). Útil pero no obligatorio: nociones de TensorFlow/Keras y JAX.
- GPU **no necesaria**: los grafos de juguete (7-34 nodos) corren en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - Message Passing | PyTorch 2.x | TensorFlow 2.x, JAX |
| 02 - GCN | PyTorch 2.x | TensorFlow 2.x, JAX + optax |
| 03 - GAT | PyTorch 2.x | TensorFlow 2.x, JAX |

Versiones de referencia: `torch>=2.2`, `tensorflow>=2.15`, `jax>=0.4`, `optax>=0.2`.

## El hilo conductor

Los tres caminos son la misma idea a creciente sofisticación de la "combinación de mensajes":

1. **Message passing** agrega los vecinos con un peso **fijo** (normalización por grado): $\hat{A}HW$.
2. **GCN** es exactamente eso, derivado desde convoluciones espectrales, aplicado a clasificar nodos con pocos labels.
3. **GAT** reemplaza el peso fijo por uno **aprendido con atención** ($\alpha_{ij}$) — y cuando el grafo es completo, eso es un Transformer.

---

**Ver tambien:** [Clase 27 - Teoria](../teoria) · [Clase 27 - Profundizacion](../profundizacion) · Fundamentos: [Redes Neuronales de Grafos](/fundamentos/redes-neuronales-de-grafos) · [Message Passing](/fundamentos/message-passing) · [Expresividad de las GNN](/fundamentos/expresividad-gnn).
