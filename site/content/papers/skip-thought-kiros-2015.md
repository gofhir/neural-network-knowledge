---
title: "Skip-Thought Vectors"
weight: 252
math: true
---

{{< paper-card
    title="Skip-Thought Vectors"
    authors="Kiros, Zhu, Salakhutdinov, Zemel, Torralba, Urtasun, Fidler"
    year="2015"
    venue="NeurIPS 2015"
    pdf="/papers/skip-thought-kiros-2015.pdf" >}}
El **primer modelo no-supervisado de sentence embeddings transferibles**. Generaliza la idea de Word2Vec Skip-gram del nivel palabra al nivel oracion: dada una oracion, predecir las **oraciones adyacentes** en un corpus continuo. La oracion se codifica con un GRU; los decoders predicen las oraciones siguiente y anterior. Es el ancestro directo de InferSent, USE y Sentence-BERT.
{{< /paper-card >}}

---

## Contexto

En 2015, representar oraciones era un problema abierto. Las opciones eran:

- **Composicion de word embeddings** (sum/mean): simple pero pobre, ignora orden y sintaxis.
- **Modelos supervisados** (CNN, RNN, Tree-LSTM): requieren labels, no transferibles entre tareas.
- **Paragraph Vectors / Doc2Vec** (Le-Mikolov 2014): un vector por documento, **pero re-entrenamiento en test time** (no es un encoder verdadero).

Pregunta del paper: *"Is there a task and a corresponding loss that will allow us to learn highly generic sentence representations?"*

Respuesta: **generalizar Skip-gram a oraciones**.

---

## Ideas principales

### 1. Analogia exacta con Skip-gram

| Skip-gram (Word2Vec) | Skip-Thought |
|---|---|
| Unidad: palabra $w_t$ | Unidad: oracion $s_i$ |
| Predecir: palabras del contexto $w_{t \pm j}$ | Predecir: oraciones adyacentes $s_{i \pm 1}$ |
| Encoder: lookup | Encoder: GRU |
| Decoder: softmax sobre $\|V\|$ | Decoder: GRU palabra-por-palabra |
| Resultado: word embeddings | Resultado: **sentence embeddings** |

### 2. Encoder GRU

Procesar la oracion $s_i = (w_i^1, \ldots, w_i^N)$ con un GRU estandar:

$$\mathbf{r}^t = \sigma(\mathbf{W}_r \mathbf{x}^t + \mathbf{U}_r \mathbf{h}^{t-1})$$

$$\mathbf{z}^t = \sigma(\mathbf{W}_z \mathbf{x}^t + \mathbf{U}_z \mathbf{h}^{t-1})$$

$$\bar{\mathbf{h}}^t = \tanh(\mathbf{W} \mathbf{x}^t + \mathbf{U}(\mathbf{r}^t \odot \mathbf{h}^{t-1}))$$

$$\mathbf{h}^t = (1 - \mathbf{z}^t) \odot \mathbf{h}^{t-1} + \mathbf{z}^t \odot \bar{\mathbf{h}}^t$$

El sentence embedding es $\mathbf{h}_i = \mathbf{h}^N$ (estado final).

### 3. Conditional GRU decoder

Dos decoders (uno para $s_{i+1}$, otro para $s_{i-1}$) con **conditional GRU**: el embedding $\mathbf{h}_i$ se inyecta en cada gate del decoder via matrices $\mathbf{C}_r, \mathbf{C}_z, \mathbf{C}$:

$$\mathbf{r}^t = \sigma(\mathbf{W}_r^d \mathbf{x}^{t-1} + \mathbf{U}_r^d \mathbf{h}^{t-1} + \mathbf{C}_r \mathbf{h}_i)$$

(similar para $\mathbf{z}^t$ y $\bar{\mathbf{h}}^t$). Los decoders tienen parametros separados pero **comparten la matriz de vocabulario $V$**.

### 4. Objetivo

$$\mathcal{L} = \sum_t \log P(w_{i+1}^t \mid w_{i+1}^{<t}, \mathbf{h}_i) + \sum_t \log P(w_{i-1}^t \mid w_{i-1}^{<t}, \mathbf{h}_i)$$

### 5. Vocabulary expansion -- el truco crucial

El corpus de entrenamiento (BookCorpus) tiene vocab ~20k. Pero en test queremos encodear oraciones con palabras nuevas. Solucion: aprender una **regresion lineal** $\mathbf{W}_{\text{exp}}: \mathbb{R}^{300} \to \mathbb{R}^{620}$ que mapea Word2Vec preentrenado (cobertura amplia ~3M) a los embeddings del encoder de Skip-Thought.

Vocabulario efectivo expandido de 20k a **930.911 palabras** (~46x).

### 6. Variantes

- **uni-skip**: encoder unidireccional, 2400 dim.
- **bi-skip**: bidireccional, 1200 + 1200 = 2400 dim concatenadas.
- **combine-skip**: concat(uni, bi) = 4800 dim. **Ganador empirico**.

---

## Resultados experimentales

### Semantic Relatedness (SICK)

| Metodo | Pearson $r$ | Spearman $\rho$ | MSE |
|---|---|---|---|
| Mean word vectors | 0.758 | 0.674 | 0.456 |
| Tree-LSTM (supervisado, requiere parser) | **0.868** | **0.808** | **0.253** |
| **combine-skip** | 0.858 | 0.792 | 0.269 |
| **combine-skip + COCO features** | 0.866 | 0.800 | 0.256 |

Skip-Thought (sin etiquetas) compite con Tree-LSTM supervisado.

### Paraphrase Detection (MSR Paraphrase Corpus)

| Metodo | Acc | F1 |
|---|---|---|
| TF-KLD (Ji-Eisenstein, supervisado) | **80.4** | **86.0** |
| combine-skip | 73.0 | 82.0 |
| combine-skip + features | 75.8 | 83.0 |

### Image-sentence ranking, sentiment, sujetividad

Competitivo con metodos supervisados de la era en MR, CR, SUBJ, MPQA, TREC.

### Corpus de entrenamiento: BookCorpus

| Estadistica | Valor |
|---|---|
| # libros | 11.038 |
| # oraciones | 74M |
| # palabras | ~1B |
| # palabras unicas | 1.3M |
| Generos | 16 (Romance, Fantasy, Sci-Fi, etc.) |

Mismo corpus que despues usarian **BERT** y **GPT-1**.

---

## Limitaciones

1. **Costo de entrenamiento**: ~2 semanas en GPU. Inviable para investigadores sin recursos.
2. **Vocabulary expansion** es un hack. BPE/WordPiece (BERT 2018) lo resuelve de raiz.
3. **Polisemia y composicionalidad fina**: el modelo falla en distinguir "tricks on a motorcycle" vs "tricking a person on a motorcycle".
4. **Solo ingles**: requiere corpus narrativo, dificil de portar a low-resource languages.
5. **Encoder secuencial**: lento, limitado en dependencias largas.

---

## Por que importa hoy

Skip-Thought establece **el patron** que dominaria sentence embeddings durante 5 anos:

1. **Encoder universal aprendido sin supervision** -> InferSent (2017), USE (2018), SBERT (2019), SimCSE (2021), Sentence-T5 (2021).
2. **Autosupervision a nivel de oracion** -> Next Sentence Prediction (BERT), Sentence Order Prediction (ALBERT), objetivos contrastivos (SimCSE).

La idea evoluciono pero el principio fundamental -- *"oraciones adyacentes son semanticamente similares; entrenar para distinguirlas produce embeddings utiles"* -- persiste en todo el campo moderno de sentence/document retrieval (RAG, vector DBs).

---

## Notas y enlaces

- **Codigo**: https://github.com/ryankiros/skip-thoughts (Theano original).
- **Sucesores**: InferSent (Conneau 2017), Universal Sentence Encoder (Cer 2018), [Sentence-BERT](https://www.sbert.net) (Reimers 2019).
- **Predecesores conceptuales**: [Word2Vec Distributed](/papers/word2vec-distributed-mikolov-2013).
- **Clase asociada**: [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
- **Fundamentos relacionados**: [Skip-Thought](/fundamentos/skip-thought), [Redes recurrentes](/fundamentos/redes-recurrentes), [LSTM/GRU](/fundamentos/lstm-gru).
- **Cita BibTeX**:

```bibtex
@inproceedings{kiros2015skip,
  title={Skip-thought vectors},
  author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Russ R and Zemel, Richard and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3294--3302},
  year={2015}
}
```
