---
title: "Clase 16 - Introduccion a NLP"
weight: 120
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga
**Fecha:** 2026-04-30

Primera clase del bloque de Procesamiento de Lenguaje Natural (NLP). Recorrido conceptual: que es NLP, por que el lenguaje es dificil (ambiguedad, multimodalidad, common sense, grounding), regularidades estadisticas (Ley de Zipf, Ley de Heaps), aplicaciones canonicas (POS tagging, parsing, NER, coreference, sentiment, NMT) y tecnicas clasicas (stop-words, stemming, lematizacion, Bag of Words, n-grams). Cierra con el ecosistema de herramientas (spaCy, NLTK, Transformers).

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 37 diapositivas de la clase" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Derivacion de Zipf y Heaps, TF-IDF, Porter stemmer y limites de BoW" icon="beaker" >}}
  {{< card link="/laboratorios/lab-16" title="Laboratorio 16" subtitle="Práctico: NLTK + spaCy + NLLB-200 + VADER + Bag of Words sobre SMS spam" icon="academic-cap" >}}
  {{< card link="/fundamentos/bag-of-words" title="Fundamento: Bag of Words" subtitle="La representación clásica de texto como vector" icon="book-open" >}}
  {{< card link="/fundamentos/tokenizacion-clasica" title="Fundamento: Tokenización clásica" subtitle="Sentence + word tokenization, NLTK + spaCy" icon="book-open" >}}
  {{< card link="/fundamentos/sentiment-analysis" title="Fundamento: Sentiment Analysis" subtitle="Rule-based vs neural, translate-then-analyze" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-datos" title="Fundamento: Representacion de Datos" subtitle="Como pasar de texto a vectores" icon="book-open" >}}
  {{< card link="/fundamentos/redes-recurrentes" title="Fundamento: RNNs" subtitle="Modelos secuenciales para lenguaje" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

- Zipf 1949 -- "Human Behaviour and the Principle of Least Effort"
- Heaps 1978 -- "Information Retrieval: Computational and Theoretical Aspects"
- Salton & Buckley 1988 -- "Term-weighting approaches in automatic text retrieval" (TF-IDF)
- [Porter 1980 -- An algorithm for suffix stripping](/papers/porter-stemmer-1980) (Porter Stemmer)
- [Miller et al. 1990/1995 -- WordNet: An Online Lexical Database](/papers/wordnet-miller-1995)
- [Bird & Loper 2006 -- NLTK: The Natural Language Toolkit](/papers/nltk-bird-loper-2006)
- [Kiss & Strunk 2006 -- Punkt: Unsupervised Multilingual Sentence Boundary Detection](/papers/punkt-kiss-strunk-2006)
- [Gimpel et al. 2011 -- Part-of-Speech Tagging for Twitter](/papers/twitter-pos-gimpel-2011)
- [Hutto & Gilbert 2014 -- VADER: A Parsimonious Rule-based Model for Sentiment Analysis](/papers/vader-hutto-gilbert-2014)
- [NLLB Team 2022 -- No Language Left Behind](/papers/nllb-team-2022) (traducción multilingüe)
- Mikolov et al. 2013 -- "Efficient Estimation of Word Representations in Vector Space" (anuncio Clase 17)
- Russell & Norvig -- "Artificial Intelligence: A Modern Approach" (texto base)
