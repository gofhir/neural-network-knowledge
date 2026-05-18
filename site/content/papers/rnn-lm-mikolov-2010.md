---
title: "RNN-LM - Recurrent Neural Network Based Language Model"
weight: 247
math: true
---

{{< paper-card
    title="Recurrent Neural Network Based Language Model"
    authors="Mikolov, Karafiat, Burget, Cernocky, Khudanpur"
    year="2010"
    venue="INTERSPEECH 2010, Makuhari"
    pdf="/papers/rnn-lm-mikolov-2010.pdf" >}}
El **primer paper de Tomas Mikolov** sobre LMs neurales. Reemplaza la ventana fija del NPLM por una **Simple Recurrent Network (Elman)** que codifica contexto en un estado oculto recurrente. Resultado: ~50% reduccion de perplejidad y ~18% reduccion de WER en Wall Street Journal vs Kneser-Ney 5-gram. Demuele el mito de que "los n-gramas con suavizado son insuperables" y abre la era neuronal del NLP.
{{< /paper-card >}}

---

## Contexto

En 2010 los LMs dominantes en speech recognition y MT eran **5-gram Kneser-Ney modificado** (Chen & Goodman 1998). NPLM (Bengio 2003) habia mostrado mejora pero a costo prohibitivo (3 semanas para 1M palabras). Mikolov critica el NPLM por su **contexto fijo**:

> *"Bengio's approach uses a feedforward network with fixed length context that needs to be specified ad hoc... It is well known that humans can exploit longer context with great success."*

La solucion: una **RNN simple (Elman)** que mantiene un estado oculto $s(t)$ actualizado en cada palabra. El contexto efectivo es teoricamente ilimitado.

---

## Ideas principales

### 1. Simple Recurrent Network (SRN / Elman)

Tres capas: input $x$, hidden $s$, output $y$.

$$x(t) = w(t) \oplus s(t-1)$$

(concatenacion del one-hot $w(t)$ con el estado previo $s(t-1)$)

$$s_j(t) = \sigma\left( \sum_i x_i(t) u_{ji} \right)$$

$$y_k(t) = \text{softmax}\left( \sum_j s_j(t) v_{kj} \right)$$

Cada paso: el estado $s(t)$ codifica todo el contexto anterior comprimido en $H$ dimensiones (tipicamente 30-500).

### 2. Truncated BPTT $\tau = 1$

Backprop a un solo paso atras inicialmente. Simplifica entrenamiento pero limita dependencias largas. En trabajos posteriores (2011, 2012) Mikolov usa BPTT mas profundo y obtiene mejoras.

### 3. Dynamic learning durante test

Una idea radical: **continuar entrenando con $\alpha = 0.1$ fijo mientras se procesan datos de test**. Permite al modelo adaptarse online a nuevos nombres propios y vocabulario especifico.

> *"If a new person-name occurs repeatedly in the test set, it will repeatedly get a very small probability... we refer to such model as dynamic."*

Es el ancestro conceptual del **in-context learning** y la adaptacion continua en LLMs modernos.

### 4. Rare words como token unico

Para acelerar el softmax: palabras con frecuencia < threshold se consolidan en `<rare>` con probabilidad uniforme entre ellas. Predecesor de `<UNK>` y BPE.

### 5. Mixture con KN5

Interpolacion lineal RNN + KN5 con peso 0.75 / 0.25. Los dos modelos son **complementarios** -- KN captura n-gramas frecuentes especificos; RNN generaliza.

---

## Resultados experimentales

### Wall Street Journal (WSJ)

| Modelo (trained on 6.4M words) | Perplejidad | WER |
|---|---|---|
| KN5 baseline | 221 | 13.5 |
| RNN 250/5 | 173 | 12.3 |
| KN5 + RNN 250/5 (mixture) | **156** | **11.7** |
| 3xRNN static + KN5 | 143 | 11.3 |
| **3xRNN dynamic + KN5** | **121** | **11.1** |

Reduccion de perplejidad de **45%** y WER de **18%** vs baseline.

### Comparacion vs SOTA con menos datos

| Modelo | Train words | EVAL WER (WSJ) |
|---|---|---|
| KN5 baseline | 37M | 17.2 |
| Discriminative LM (Xu) | 37M | 16.9 |
| Joint LM (Filimonov) | 70M | 16.7 |
| **Static 3xRNN + KN5** | **6.4M** | **15.5** |

RNN con **6.4M** palabras supera a un baseline KN entrenado con **70M** palabras -- ruptura formal del paradigma "mas datos = mejor LM".

### Speedup vs NPLM

> *"It takes around 6 hours for our basic implementation to train RNN model based on Brown corpus (800K words, 100 hidden units), while Bengio reports 113 days for basic implementation."*

**6 horas vs 113 dias** -- la recurrencia y la ausencia de capa de embedding explicita simplifican drasticamente.

---

## Limitaciones

1. **Truncated BPTT $\tau = 1$**: dependencias mas alla de unos pasos se diluyen rapido.
2. **Vanishing gradients**: SRN sufre del problema clasico (Bengio 1994). LSTM (Sundermeyer 2012) lo mitigaria.
3. **Sin embedding layer explicita**: el word vector es one-hot. Word2Vec corregira esto.
4. **Softmax sobre $|V|$**: bottleneck. Resuelto en Mikolov 2011 con hierarchical softmax.

---

## Por que importa hoy

Este paper marca el **cambio de paradigma** en LM. La frase clave del paper:

> *"Obtained results are breaking myth that language modeling is just about counting n-grams, and that the only reasonable way how to improve results is by acquiring new training data."*

Tres anos despues, Mikolov publica **Word2Vec** (2013) aplicando lecciones aprendidas aqui:
- Embeddings densos aprendidos (aunque W2V los hace explicitos).
- Autosupervision sobre contexto.
- Speed via simplificaciones drasticas.

La slide 22 de la Clase 18 muestra el diagrama del RNN-LM (figura 1 de este paper) presentandolo como alternativa al feedforward del NPLM.

---

## Notas y enlaces

- **Codigo**: rnnlm-toolkit en C++, publicado en https://www.fit.vutbr.cz/~imikolov/rnnlm/.
- **Sucesores**: [Word2Vec Efficient](/papers/word2vec-efficient-mikolov-2013), [Word2Vec Distributed](/papers/word2vec-distributed-mikolov-2013), [ELMo](/papers/elmo-peters-2018).
- **Clase asociada**: [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
- **Fundamentos relacionados**: [Redes recurrentes](/fundamentos/redes-recurrentes), [Modelos de lenguaje](/fundamentos/modelos-de-lenguaje).
- **Cita BibTeX**:

```bibtex
@inproceedings{mikolov2010rnnlm,
  title={Recurrent neural network based language model},
  author={Mikolov, Tomas and Karafi{\'a}t, Martin and Burget, Luk{\'a}s and Cernock{\`y}, Jan and Khudanpur, Sanjeev},
  booktitle={Interspeech},
  pages={1045--1048},
  year={2010}
}
```
