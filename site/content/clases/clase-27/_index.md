---
title: "Clase 27 - Redes Neuronales de Grafos"
weight: 270
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga
**Curso 3:** Relacional, GANs, RL, Meta-Learning, Razonamiento y Memoria

Primera clase del bloque relacional. Aborda las **Redes Neuronales de Grafos (GNN)**: una familia de modelos para aprender sobre datos con **estructura relacional** —moléculas, redes sociales, grafos de conocimiento, programas, mapas de navegación— donde las CNN (rejillas) y las RNN (secuencias) no encajan. La idea central es el **message passing**: cada nodo actualiza su representación combinando mensajes de sus vecinos, repetido $N$ veces, de modo que su **campo receptivo** crece y termina capturando la estructura del grafo. La clase construye el mecanismo desde la notación de grafos (nodos, aristas, matriz de adyacencia) hasta los modelos canónicos —**GGNN, GCN, GraphSAGE**— y sus aplicaciones reales —**R-GCN** (grafos de conocimiento), **GraphNav** (navegación robótica), **MPNN** (química cuántica) y **detección de bugs en código**—, cerrando con la pregunta teórica de qué pueden y no pueden distinguir las GNN.

La clase se apoya en el [Mecanismo de atención (Clase 15)](/clases/clase-15) y los [Transformers (Clase 14)](/clases/clase-14) —un Transformer es, formalmente, una GNN de atención (GAT) sobre un grafo completo— y conecta con la [Recomendación (Clase 25)](/clases/clase-25), donde PinSage/GraphSAGE llevan las GNN a escala industrial.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 49 diapositivas: notación de grafos, message passing en 4 etapas, receptive field, objetivos, GGNN/GCN/GraphSAGE y aplicaciones" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: forma matricial, GCN desde convoluciones espectrales (Laplaciano/Chebyshev), GRU de GGNN, atención de GAT, expresividad WL/FOC2, over-smoothing" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Message passing, GCN y GAT desde cero en triple framework (PyTorch, TensorFlow, JAX)" icon="code" >}}
  {{< card link="/clases/clase-26" title="Clase anterior: Meta-aprendizaje" subtitle="Aprender a aprender, few-shot, MAML" icon="arrow-left" >}}
  {{< card link="/clases/clase-15" title="Base: Mecanismo de atencion" subtitle="La atención que GAT lleva a los grafos" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/redes-neuronales-de-grafos" title="Redes Neuronales de Grafos" subtitle="La familia completa: notación, tareas, modelos, problemas y aplicaciones" icon="book-open" >}}
  {{< card link="/fundamentos/message-passing" title="Message Passing" subtitle="El mecanismo en detalle: framework MPNN, las 4 etapas, forma matricial, invarianza a permutación" icon="book-open" >}}
  {{< card link="/fundamentos/expresividad-gnn" title="Expresividad de las GNN" subtitle="Qué pueden distinguir: test Weisfeiler-Lehman, GIN, caracterización lógica FOC2" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Mecanismo de Atencion" subtitle="Base de GAT; un Transformer es un GAT sobre grafo completo" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Redes Convolucionales" subtitle="La GCN generaliza la convolución de rejillas a grafos arbitrarios" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/ggnn-li-2015" title="GGNN (2015)" subtitle="Li et al. — Gated Graph Sequence NN: pasos fijos + GRU" icon="document-text" >}}
  {{< card link="/papers/gcn-kipf-2017" title="GCN (2017)" subtitle="Kipf & Welling — convolución espectral simplificada, el paper más citado de GNN" icon="document-text" >}}
  {{< card link="/papers/graphsage-hamilton-2017" title="GraphSAGE (2017)" subtitle="Hamilton et al. — inductivo, sample-and-aggregate a gran escala" icon="document-text" >}}
  {{< card link="/papers/mpnn-gilmer-2017" title="MPNN (2017)" subtitle="Gilmer et al. — el framework que unifica las GNN (química cuántica)" icon="document-text" >}}
  {{< card link="/papers/rgcn-schlichtkrull-2018" title="R-GCN (2018)" subtitle="Schlichtkrull et al. — grafos de conocimiento, pesos por relación" icon="document-text" >}}
  {{< card link="/papers/graphnav-chen-2019" title="GraphNav (2019)" subtitle="Chen et al. — navegación visual con grafo topológico (presentado por Felipe del Río)" icon="document-text" >}}
  {{< card link="/papers/programs-as-graphs-allamanis-2018" title="Programs as Graphs (2018)" subtitle="Allamanis et al. — detección de bugs representando código como grafo" icon="document-text" >}}
  {{< card link="/papers/logical-expressiveness-barcelo-2020" title="Logical Expressiveness (2020)" subtitle="Barceló, Pérez et al. (PUC Chile) — la expresividad de las GNN en lógica FOC2" icon="document-text" >}}
{{< /cards >}}

## Papers canónicos (complementarios)

{{< cards >}}
  {{< card link="/papers/gnn-model-scarselli-2009" title="The Graph Neural Network Model (2009)" subtitle="Scarselli et al. — el paper que acuñó el término GNN (punto fijo por contracción)" icon="document-text" >}}
  {{< card link="/papers/gat-velickovic-2018" title="GAT (2018)" subtitle="Veličković et al. — atención en grafos, el puente hacia los Transformers" icon="document-text" >}}
  {{< card link="/papers/gin-xu-2019" title="GIN (2019)" subtitle="Xu et al. — How Powerful are GNNs?: el límite Weisfeiler-Lehman" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/estructurados" title="Dominio: Datos estructurados" subtitle="Línea de tiempo: del ML tabular clásico a las GNN y los Transformers tabulares" icon="globe-alt" >}}
{{< /cards >}}
