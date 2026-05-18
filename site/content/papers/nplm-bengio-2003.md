---
title: "NPLM - A Neural Probabilistic Language Model"
weight: 246
math: true
---

{{< paper-card
    title="A Neural Probabilistic Language Model"
    authors="Bengio, Ducharme, Vincent, Jauvin"
    year="2003"
    venue="JMLR vol. 3, pp. 1137-1155"
    pdf="/papers/nplm-bengio-2003.pdf" >}}
El **paper fundacional del paradigma de embeddings aprendidos**. Propone aprender simultaneamente (1) una representacion vectorial densa para cada palabra y (2) la funcion de probabilidad de secuencias expresada en terminos de esos vectores. Resuelve la maldicion de dimensionalidad de los n-gramas via generalizacion semantica: si "dog" y "cat" tienen vectores similares y se vio "The cat is walking", entonces "A dog was running" tambien recibe alta probabilidad. Sin este paper no hay Word2Vec, BERT ni GPT.
{{< /paper-card >}}

---

## Contexto

En 2002 los LMs dominantes eran **n-gramas con suavizado Kneser-Ney** (Chen & Goodman 1998). Su debilidad central: dos problemas combinados.

1. **Curse of dimensionality**: para $|V| = 17.000$ y $n = 10$, hay $|V|^{n-1} \approx 10^{42}$ contextos posibles. Solo una fraccion infima aparece en cualquier corpus.
2. **Sin similitud entre palabras**: "The cat is walking in the bedroom" vs "A dog was running in a room" -- semanticamente equivalentes, pero un n-grama trata ambas como totalmente independientes.

Antecedentes: Hinton 1986 propuso *distributed representations* como idea filosofica; Elman 1990 entreno SRNs con representaciones implicitas; Miikkulainen & Dyer 1991 usaron NN para LM a pequena escala. Bengio escala estas ideas a contexto multi-palabra con datasets reales (Brown 800k, AP News 14M).

---

## Ideas principales

### 1. Asociar a cada palabra un vector denso

Matriz $C \in \mathbb{R}^{|V| \times m}$ de **lookup**. Cada palabra $w_i$ es $C(w_i) \in \mathbb{R}^m$ con $m = 30, 60, 100$. Mucho menor que $|V|$ -- resuelve el problema de dimensionalidad.

### 2. Red neuronal feedforward sobre contexto concatenado

Para predecir $w_t$ dado contexto $w_{t-n+1:t-1}$:

$$x = [C(w_{t-1}); C(w_{t-2}); \ldots; C(w_{t-n+1})] \in \mathbb{R}^{(n-1)m}$$

$$y = b + Wx + U \tanh(d + Hx)$$

$$\hat{P}(w_t = i \mid w_{<t}) = \frac{e^{y_i}}{\sum_j e^{y_j}}$$

Donde $H \in \mathbb{R}^{h \times (n-1)m}$ son los pesos de la capa hidden (con $h = 50, 60$ unidades tanh), $U \in \mathbb{R}^{|V| \times h}$ proyecta a output, y $W \in \mathbb{R}^{|V| \times (n-1)m}$ son **skip connections directas** opcionales.

### 3. Aprender embeddings y pesos simultaneamente

Backprop end-to-end optimiza $\theta = (b, d, W, U, H, C)$ con cross-entropy. La matriz $C$ es **compartida** entre todas las posiciones del contexto -- esto es **parameter sharing across positions**, base de RNNs, LSTMs y Transformers posteriores.

### 4. Generalizacion via geometria

Si dos palabras tienen vectores cercanos, la probabilidad asignada por la red es similar -- propiedad heredada de la **suavidad** de la red neuronal. Por eso una sola observacion de "The cat is walking" aumenta la probabilidad de un **numero exponencial de oraciones vecinas** en el espacio de embeddings.

### 5. Mixture con interpolated trigram

El paper reporta que combinar el NPLM con un trigram suavizado mediante interpolacion lineal ($\alpha = 0.5$) reduce la perplejidad. Los dos modelos capturan informacion complementaria.

### 6. Asynchronous SGD pre-HogWild

La seccion 3 detalla paralelizacion con **actualizaciones lockless** en memoria compartida:

> *"Sometimes, part of an update on the parameter vector by one of the processors is lost, being overwritten by the update of another processor, and this introduces a bit of noise. However, this noise seems to be very small."*

Esto es asynchronous SGD ANTES de HogWild! (Niu 2011) por casi una decada.

---

## Resultados experimentales

| Dataset | Modelo | Perplejidad test |
|---|---|---|
| Brown (800k tokens) | Trigram interpolated | 343 |
| Brown | 5-gram Kneser-Ney | 321 |
| Brown | **NPLM** | **268** |
| Brown | NPLM mixture (NPLM + trigram) | **252** |
| AP News (14M tokens) | Trigram | 137 |
| AP News | **NPLM** | **109** |
| AP News | NPLM mixture | **104** |

**Reduccion de perplejidad sobre KN5**: 20-25%, enorme para 2003. Costo: **3 semanas en cluster** para Brown.

---

## Limitaciones reconocidas

1. **Costo computacional**: 3 semanas para Brown (~1M palabras) -- inviable para escalar.
2. **Softmax sobre $|V|$**: cuello de botella dominante. Resuelto despues con hierarchical softmax (Morin & Bengio 2005) y negative sampling (Mikolov 2013).
3. **Vocabulario fijo**: OOV se mapean a `<unk>`.
4. **Contexto markoviano**: ventana fija $n-1$, no captura dependencias largas. RNN-LM (Mikolov 2010) lo soluciona.
5. **Sin bidireccionalidad**: solo contexto izquierdo. ELMo y BERT corregiran esto.

---

## Por que importa hoy

NPLM 2003 establece **tres principios** que sobreviven en cada modelo foundation moderno:

1. **Embeddings densos aprendidos** como representacion primaria de tokens.
2. **End-to-end learning** de representacion y modelo.
3. **Parameter sharing** entre posiciones del contexto.

Bengio recibio el **Premio Turing 2018** junto con Hinton y LeCun por su trabajo fundacional en deep learning -- y NPLM es uno de los papers mas citados de esa contribucion. La slide 21 de la Clase 18 reproduce el diagrama de la Figura 1 de este paper (sin atribucion explicita) bajo el titulo "Neural Probabilistic Language Models".

Toda la genealogia [[NPLM]] -> [[RNN-LM]] -> [[Word2Vec]] -> [[ELMo]] -> [[BERT]] -> GPT hereda dos cosas: embeddings densos aprendidos y autosupervision sobre tokens contextuales.

---

## Notas y enlaces

- **Codigo**: el paper publico el codigo en C/C++. Hoy se reimplementa en pocas lineas con PyTorch / Flax.
- **Comparacion con sucesores**: ver [Mikolov 2010 RNN-LM](/papers/rnn-lm-mikolov-2010), [Word2Vec Efficient 2013](/papers/word2vec-efficient-mikolov-2013).
- **Clase asociada**: [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
- **Fundamento relacionado**: [Embeddings distribuidos](/fundamentos/embeddings-distribuidos), [Modelos de lenguaje](/fundamentos/modelos-de-lenguaje).
- **Cita BibTeX**:

```bibtex
@article{bengio2003neural,
  title={A neural probabilistic language model},
  author={Bengio, Yoshua and Ducharme, R{\'e}jean and Vincent, Pascal and Jauvin, Christian},
  journal={Journal of Machine Learning Research},
  volume={3},
  pages={1137--1155},
  year={2003}
}
```
