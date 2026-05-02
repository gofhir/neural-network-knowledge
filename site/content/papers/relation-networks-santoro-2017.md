---
title: "Relation Networks (Razonamiento Relacional)"
weight: 297
math: true
---

{{< paper-card
    title="A simple neural network module for relational reasoning"
    authors="Santoro, Raposo, Barrett, Malinowski, Pascanu, Battaglia, Lillicrap"
    year="2017"
    venue="NeurIPS 2017"
    pdf="/papers/relation-networks-santoro-2017.pdf"
    arxiv="1706.01427" >}}
Introduce las **Relation Networks (RN)** -- un modulo neural simple, plug-and-play, disenado especificamente para razonamiento relacional. Calcula relaciones entre **todos** los pares de objetos en un conjunto, con una MLP compartida, y agrega los resultados via suma. Logra rendimiento super-humano en CLEVR (95.5%) y resuelve 18/20 tareas de bAbI. Es un precursor conceptual directo de self-attention y de los Graph Neural Networks modernos.
{{< /paper-card >}}

---

## Contexto

El **razonamiento relacional** -- inferir relaciones entre entidades y propiedades -- es una capacidad cognitiva central. Ejemplos: comparar distancias entre arboles para escoger destino, encadenar pistas en una novela policial, o decidir si dos objetos en una escena 3D son del mismo tamano.

Los enfoques simbolicos manejan relaciones de forma explicita (logica, algebra) pero sufren del symbol grounding problem. Los modelos neurales puros (CNN, MLP) generalizan desde datos crudos pero fallan en problemas con estructura relacional rica y datos escasos. Soluciones intermedias previas como **Memory Networks** y **soft attention** proveen inspiracion pero no estan disenadas explicitamente para relaciones par-a-par.

Santoro et al. (DeepMind, 2017) proponen un modulo dedicado, simple y diferenciable que se conecta sobre arquitecturas estandar (CNN, LSTM) para inyectar bias relacional explicito.

---

## Ideas principales

### 1. Modulo plug-and-play

Una RN no procesa pixeles ni tokens directamente: opera sobre un **conjunto de objetos** $O = \{o_1, \dots, o_n\}$, $o_i \in \mathbb{R}^m$. Estos objetos pueden ser celdas de un feature map de CNN, estados finales de LSTM por oracion, o filas de una matriz de state descriptions. La RN se acopla aguas abajo de cualquier extractor de features.

### 2. Forma funcional

$$RN(O) = f_\phi\left(\sum_{i,j} g_\theta(o_i, o_j)\right)$$

- $g_\theta$: MLP que codifica la **relacion** entre el par $(o_i, o_j)$. Compartida para todos los pares.
- $f_\phi$: MLP que **agrega** la representacion sumada y produce la salida final.
- $\sum_{i,j}$: suma sobre todos los pares (incluyendo $i=j$ y orden $(i,j) \neq (j,i)$).

### 3. Considera todos los pares

A diferencia de soft attention -- que **pondera** posiciones con $\alpha_{ij}$ aprendido -- la RN evalua $g_\theta$ en **todos** los pares con peso uniforme. Es la red la que decide internamente, dentro de $g_\theta$, si la relacion es relevante o no.

### 4. Condicionamiento por pregunta

Cuando hay un input adicional como una pregunta $q$ (codificada por LSTM), la RN se condiciona:

$$RN(O, q) = f_\phi\left(\sum_{i,j} g_\theta(o_i, o_j, q)\right)$$

Asi, las relaciones relevantes dependen del contexto.

### 5. Tres propiedades clave

- **Aprende a inferir relaciones** -- no se le dice cuales pares importan.
- **Es eficiente en datos** -- una sola $g_\theta$ entrena con $n^2$ pares por sample, no con $n^2$ MLPs distintas.
- **Permutation invariant** -- la suma garantiza que el orden de los objetos no afecta el output.

---

## Aplicaciones

### CLEVR (visual QA, 3D rendered)

Pipeline: CNN procesa imagen $\to$ feature map $d \times d \times k$ $\to$ cada celda taggeada con coordenada $(x,y)$ es un "objeto". LSTM codifica la pregunta. RN combina pares condicionando con $q$. MLP final produce respuesta.

- **CNN+LSTM+RN: 95.5%** (super-humano, humano = 92.6%).
- CNN+LSTM solo: 52.3%. Stacked Attention: 76.6%.
- En version "from state descriptions" (sin pixeles): **96.4%**.

### bAbI (text QA)

Cada oracion del support set se codifica con LSTM como un objeto. RN razona sobre los objetos.

- **18/20 tareas** con accuracy >95%.
- Comparable a Sparse DNC (19/20), supera a Memory Networks (14/20) y EntNet (16/20).

### Sort-of-CLEVR (sintetico)

Dataset disenado para separar preguntas relacionales de no-relacionales sobre 6 objetos 2D.

- RN: 94% en relacionales, 99% en no-relacionales.
- CNN+MLP: 63% en relacionales (falla), 99% en no-relacionales.

Demuestra que el bias relacional explicito **es necesario** para razonar sobre relaciones, incluso con suficiente capacidad.

### Sistemas fisicos dinamicos

Bolas conectadas con resortes en MuJoCo. RN infiere conexiones (93% accuracy) y cuenta sistemas conectados (95%) desde trayectorias.

---

## Conexion con self-attention

La similitud estructural con el Transformer (Vaswani et al. 2017, mismo ano) es notable:

- **Self-attention**: $\text{output}_i = \sum_j \alpha_{ij} (W^V x_j)$, con $\alpha_{ij} = \text{softmax}(QK^T/\sqrt{d_k})_{ij}$. Pondera con peso aprendido via $Q,K$.
- **RN**: $\sum_j g_\theta(x_i, x_j)$. Agrega con peso uniforme y MLP bivariada fija.

Self-attention puede leerse como una RN donde $g_\theta(x_i, x_j) = \alpha_{ij} W^V x_j$ con $\alpha$ aprendido por inner-product. Ambas comparten el **bias relacional** -- considerar todos los pares $(i,j)$ -- y la **permutation invariance** (modulo positional encoding). RN es un precursor conceptual directo de self-attention.

---

## Por que importa

- **Formaliza el bias relacional** como un building block reutilizable, separable del resto de la red.
- **Influencia directa en GNNs**: RN es message-passing en un grafo completo con funcion de mensaje aprendible -- la base de Graph Neural Networks modernos.
- **Conexion a Transformers**: Misma forma sum-over-pairs, distinto esquema de pesos. Ayuda a entender por que self-attention escala el principio.
- **Demuestra induccion de objetos**: la CNN aguas arriba aprende, sin supervision explicita, a producir representaciones object-like utiles para la RN.

---

## Limitaciones

- **Complejidad $O(n^2)$** en pares -- intratable para $n$ grande.
- **Solo binario**: $g_\theta$ ve dos objetos a la vez. Relaciones de mayor aridad (ternarias, etc.) no son representadas explicitamente.
- **Suma uniforme**: sin attention, todas las relaciones contribuyen igual al pooling. Self-attention es una generalizacion estricta.
- **Requiere objetos discretos**: en pixeles, depende de que el grid de la CNN haga las veces de "objetos".

---

## Notas y enlaces

- Codigo: implementaciones reproducidas por la comunidad estan disponibles en GitHub.
- Follow-ups directos:
  - **Vaswani et al. 2017** "Attention Is All You Need" -- self-attention generaliza el bias.
  - **Zaheer et al. 2017** "Deep Sets" -- formaliza la equivalencia $\sum + \text{MLP}$ como representacion universal de funciones permutation-invariant.
  - **Battaglia et al. 2018** "Relational Inductive Biases" -- enmarca RN, GNN e Interaction Networks bajo un framework unificado.

Ver fundamentos: [Self-Attention](/fundamentos/self-attention) - [Transformer](/fundamentos/transformer) - [Clase 14](/clases/clase-14).
