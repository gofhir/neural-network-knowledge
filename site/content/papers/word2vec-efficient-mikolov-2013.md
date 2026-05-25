---
title: "Word2Vec - Efficient Estimation of Word Representations"
weight: 248
math: true
---

{{< paper-card
    title="Efficient Estimation of Word Representations in Vector Space"
    authors="Mikolov, Chen, Corrado, Dean"
    year="2013"
    venue="ICLR 2013 Workshop"
    pdf="/papers/word2vec-efficient-mikolov-2013.pdf"
    arxiv="1301.3781" >}}
Introduce **CBoW** y **Skip-gram**: dos arquitecturas drasticamente simplificadas (sin hidden layer) para aprender word embeddings desde corpora masivos. Su tesis central -- "modelos simples entrenados con mucha data superan a sistemas complejos entrenados con poca data" -- se demostraria correcta y definiria toda la era de foundation models. Introduce ademas la **tarea de analogias** que se vuelve benchmark estandar.
{{< /paper-card >}}

---

## Contexto

Pre-2013, los modelos neuronales de embeddings (NPLM de Bengio, Collobert-Weston, HLBL de Mnih-Hinton) eran **demasiado costosos para escalar a corpora gigantes**. Mikolov observa que la mayor parte del computo de NPLM va a (i) la capa hidden con tanh y (ii) el softmax sobre $|V|$. Si lo que interesan son los embeddings (no el LM), se puede sacrificar la capa hidden a cambio de **escalar a 10-100x mas datos**.

---

## Ideas principales

### 1. Continuous Bag-of-Words (CBoW)

Dado el contexto $\{w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c}\}$, predecir $w_t$.

$$\mathbf{h} = \frac{1}{2c} \sum_{-c \le j \le c, j \ne 0} \mathbf{v}_{w_{t+j}}$$

$$P(w_t = k \mid \text{ctx}) = \frac{\exp(\mathbf{v}'_k \cdot \mathbf{h})}{\sum_i \exp(\mathbf{v}'_i \cdot \mathbf{h})}$$

Promedio de embeddings de contexto -> softmax. **Sin orden** (es "bag of words" pero con embeddings continuos).

### 2. Continuous Skip-gram

Tarea inversa: dado $w_t$, predecir cada palabra del contexto $w_{t+j}$.

$$P(w_{t+j} = k \mid w_t) = \frac{\exp(\mathbf{v}'_k \cdot \mathbf{v}_{w_t})}{\sum_i \exp(\mathbf{v}'_i \cdot \mathbf{v}_{w_t})}$$

Objetivo a maximizar sobre el corpus:

$$\mathcal{L}_{\text{SG}} = \frac{1}{T} \sum_t \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} \mid w_t)$$

**Truco**: ventana variable. En cada paso muestrear $R \sim \text{Uniform}\{1, \ldots, c\}$ y solo predecir las $R$ palabras a cada lado. Pondera implicitamente palabras cercanas.

### 3. Dos matrices de embeddings

Cada palabra $w$ tiene **dos** vectores: $\mathbf{v}_w$ (input) y $\mathbf{v}'_w$ (output). Al final del entrenamiento, $\mathbf{v}_w$ se exporta como "el embedding"; $\mathbf{v}'_w$ se descarta o se promedia.

### 4. Tarea de evaluacion: word analogies

El paper introduce el benchmark **questions-words** ($\sim$20k preguntas):

- **Sintacticas** ($\sim$8k): plurales (`apple:apples :: car:?`), comparativos, verbo presente/pasado, etc.
- **Semanticas** ($\sim$9k): capitales (`Athens:Greece :: Oslo:?`), monedas, familia, etc.

Metrica: dada `a:b :: c:?`, computar $\mathbf{x} = \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$ y devolver la palabra mas cercana por cosine similarity.

### 5. CBoW vs Skip-gram

| Tarea | CBoW | Skip-gram |
|---|---|---|
| Velocidad | Rapida | ~5x mas lenta |
| Sintaxis | Mejor | Peor |
| Semantica | Peor | Mejor |
| Palabras raras | Peor (suma diluye) | Mejor (cada palabra es su propio target) |

Skip-gram dominaria la era post-2013.

---

## Resultados experimentales

### Comparacion con modelos previos (analogias)

| Modelo | Dim | Train words | Sem. acc | Syn. acc | Total |
|---|---|---|---|---|---|
| Collobert-Weston | 50 | 660M | 9.3 | 12.3 | 11.0 |
| Mnih HLBL | 100 | 37M | 1.6 | 8.5 | 5.4 |
| Mikolov NNLM | 100 | 6B | 23.2 | 53.0 | 39.8 |
| **CBoW** | 300 | 783M | 15.5 | 53.1 | 36.1 |
| **Skip-gram** | 300 | 783M | **50.0** | **55.9** | **53.3** |

Skip-gram con 783M palabras supera a NNLM con 6B en analogias semanticas -- en una fraccion del tiempo de entrenamiento.

### Speedup vs NNLM

- NNLM 783M words: ~10 horas en 14 CPUs.
- **CBoW 783M words: 40 minutos en 1 CPU.**
- **Skip-gram 783M words: 40 minutos en 1 CPU.**

Esto explica la explosion en adopcion: cualquier laboratorio podia entrenar embeddings de SOTA en su laptop.

### Tabla de dimensionalidad (Tabla 1 del paper)

| Dim | Train words | Total accuracy |
|---|---|---|
| 50 | 24M | 12.7 |
| 100 | 24M | 18.7 |
| 300 | 24M | 21.0 |
| 600 | 24M | 21.0 |

**Conclusion clave**: la calidad escala con **datos**, no con dimensiones. A partir de 300 dim, mas dimensiones no ayudan si no se aumenta el corpus.

---

## Limitaciones

1. **No es un LM**: solo aprende embeddings, no la distribucion completa $P(w_t \mid w_{<t})$.
2. **No orden**: CBoW promedia, Skip-gram trata cada palabra de la ventana por separado.
3. **No frases**: "New York" se descompone. El paper companion (Distributed Representations) resuelve esto.
4. **No subwords**: morfologia compleja no se captura. FastText (2016) lo soluciona.
5. **Embeddings no contextuales**: un unico vector por palabra. ELMo y BERT lo arreglaran.
6. **Softmax exacto sigue siendo el bottleneck**: resuelto en el paper companion con negative sampling.

---

## Por que importa hoy

Word2Vec es **el paper que democratizo los word embeddings**. Los embeddings preentrenados `GoogleNews-vectors-negative300.bin` (3M palabras, 300d, 1.6 GB) se descargaron millones de veces y fueron el "ImageNet de NLP" durante 4 anos. La idea de aprender embeddings simples a escala masiva se generalizo a:

- **Recsys**: prod2vec, item2vec.
- **Grafos**: DeepWalk, node2vec.
- **Biologia**: gene2vec, protein embeddings.
- **Codigo**: code2vec.

La filosofia "modelos simples + mucha data" guio toda la era de pre-training masivo que culmina en GPT-3, Llama y Claude.

---

## Notas y enlaces

- **Codigo**: https://code.google.com/p/word2vec/ (luego migrado a GitHub).
- **Wrapper Python**: [Gensim](https://radimrehurek.com/gensim/).
- **Continuation**: [Word2Vec Distributed Representations](/papers/word2vec-distributed-mikolov-2013) introduce negative sampling, subsampling, phrases.
- **Analisis teorico**: [Levy & Goldberg 2014](/papers/sgns-implicit-mf-levy-goldberg-2014) demuestra que SGNS factoriza PMI shifted.
- **Clase asociada**: [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
- **Laboratorio asociado**: [Lab 18 - Word Embeddings en accion](/laboratorios/lab-18) (Skip-gram entrenado sobre Google News).
- **Fundamentos relacionados**: [Word2Vec](/fundamentos/word2vec), [Embeddings distribuidos](/fundamentos/embeddings-distribuidos).
- **Cita BibTeX**:

```bibtex
@inproceedings{mikolov2013efficient,
  title={Efficient estimation of word representations in vector space},
  author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  booktitle={International Conference on Learning Representations (Workshop)},
  year={2013}
}
```
