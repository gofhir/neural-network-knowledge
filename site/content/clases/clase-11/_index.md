---
title: "Clase 11 - Redes Recurrentes (RNNs)"
weight: 70
sidebar:
  open: true
---

**Profesor:** Alvaro Soto
**Fecha:** 2026-04-15

Redes neuronales recurrentes para procesamiento de secuencias: vanilla RNN, configuraciones (one-to-many, many-to-many, encoder-decoder), bidireccionales, BPTT, vanishing/exploding gradient, LSTM y GRU.

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 46 diapositivas de la clase" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="BPTT, LSTM math, vanishing gradient analysis" icon="beaker" >}}
  {{< card link="/fundamentos/redes-recurrentes" title="Fundamento: RNNs" subtitle="Definicion, configuraciones, aplicaciones" icon="book-open" >}}
  {{< card link="/fundamentos/lstm-gru" title="Fundamento: LSTM y GRU" subtitle="Compuertas, ecuaciones, comparacion" icon="book-open" >}}
  {{< card link="/fundamentos/backpropagation-through-time" title="Fundamento: BPTT" subtitle="Algoritmo, vanishing/exploding, soluciones" icon="book-open" >}}
  {{< card link="/videos/mit-6s191-rnn" title="Video MIT 6.S191 (2020)" subtitle="Ava Soleimany - mismo material en formato video" icon="film" >}}
  {{< card link="/videos/mit-6s191-l2-2026" title="Video MIT 6.S191 (2026)" subtitle="Ava Amini - extiende a Transformers" icon="film" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/lstm-hochreiter-1997" title="LSTM (1997)" subtitle="Hochreiter & Schmidhuber" icon="document-text" >}}
  {{< card link="/papers/gru-cho-2014" title="GRU + Encoder-Decoder (2014)" subtitle="Cho et al." icon="document-text" >}}
  {{< card link="/papers/difficulty-training-rnns-pascanu-2013" title="Difficulty Training RNNs (2013)" subtitle="Pascanu, Mikolov, Bengio" icon="document-text" >}}
  {{< card link="/papers/seq2seq-sutskever-2014" title="Seq2Seq (2014)" subtitle="Sutskever, Vinyals, Le" icon="document-text" >}}
  {{< card link="/papers/show-and-tell-vinyals-2015" title="Show and Tell (2015)" subtitle="Vinyals et al." icon="document-text" >}}
{{< /cards >}}
