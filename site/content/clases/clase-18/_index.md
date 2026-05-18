---
title: "Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought"
weight: 140
sidebar:
  open: true
---

**Profesor:** Pablo Messina
**Fecha:** 2026-05-14

Segunda clase del bloque de NLP. La clase introduce el **modelo de lenguaje (LM) probabilistico** como objeto matematico central, contrasta dos paradigmas para representarlo -- discreto (n-gramas) vs continuo (embeddings distribuidos) -- y termina con tres modelos canonicos de la era pre-Transformer: Word2Vec, GloVe y Skip-Thought. Es el puente conceptual entre la representacion clasica (Bag-of-Words de la Clase 16) y los modelos contextuales modernos (ELMo / GPT / BERT, Clase 19).

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 41 diapositivas: LM, representaciones, W2V/GloVe/SkipThought" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Suavizado KN, perplejidad, NPLM, negative sampling, derivacion GloVe, conditional GRU" icon="beaker" >}}
  {{< card link="/clases/clase-16" title="Clase anterior: Intro NLP" subtitle="Bag-of-Words, tokenizacion, Ley de Zipf" icon="arrow-left" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/modelos-de-lenguaje" title="Modelos de lenguaje" subtitle="LM probabilistico, regla cadena, n-gramas, perplejidad" icon="book-open" >}}
  {{< card link="/fundamentos/word2vec" title="Word2Vec" subtitle="CBoW, Skip-gram, negative sampling, subsampling" icon="book-open" >}}
  {{< card link="/fundamentos/glove" title="GloVe" subtitle="Factorizacion de log-co-ocurrencia global" icon="book-open" >}}
  {{< card link="/fundamentos/skip-thought" title="Skip-Thought y Sentence Embeddings" subtitle="Sentence encoders desde Skip-Thought hasta SBERT" icon="book-open" >}}
  {{< card link="/fundamentos/embeddings-distribuidos" title="Embeddings distribuidos" subtitle="One-hot vs denso, hipotesis distribucional" icon="book-open" >}}
  {{< card link="/fundamentos/redes-recurrentes" title="Redes recurrentes" subtitle="RNN como arquitectura para LMs" icon="book-open" >}}
  {{< card link="/fundamentos/bag-of-words" title="Bag of Words" subtitle="Representacion discreta clasica (Clase 16)" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/nplm-bengio-2003" title="NPLM (2003)" subtitle="Bengio - Neural Probabilistic LM, fundacional" icon="document-text" >}}
  {{< card link="/papers/rnn-lm-mikolov-2010" title="RNN-LM (2010)" subtitle="Mikolov - antecesor directo de Word2Vec" icon="document-text" >}}
  {{< card link="/papers/word2vec-efficient-mikolov-2013" title="Word2Vec Efficient (2013)" subtitle="Mikolov - CBoW + Skip-gram" icon="document-text" >}}
  {{< card link="/papers/word2vec-distributed-mikolov-2013" title="Word2Vec Distributed (2013)" subtitle="Mikolov - negative sampling, phrases" icon="document-text" >}}
  {{< card link="/papers/glove-pennington-2014" title="GloVe (2014)" subtitle="Pennington - factorizacion global de co-ocurrencia" icon="document-text" >}}
  {{< card link="/papers/sgns-implicit-mf-levy-goldberg-2014" title="SGNS as Implicit MF (2014)" subtitle="Levy & Goldberg - SGNS factoriza PMI shifted" icon="document-text" >}}
  {{< card link="/papers/skip-thought-kiros-2015" title="Skip-Thought (2015)" subtitle="Kiros - sentence embeddings autosupervisados" icon="document-text" >}}
  {{< card link="/papers/analogies-explained-allen-hospedales-2019" title="Analogies Explained (2019)" subtitle="Allen & Hospedales - prueba rigurosa de las analogias" icon="document-text" >}}
{{< /cards >}}
