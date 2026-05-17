---
title: "Lab 16 - Introducción a NLP: NLTK + spaCy + NLLB + VADER + Bag of Words"
weight: 160
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga
**Fecha:** Mayo 2026
**Notebook origen:** `clase_16/material/Laboratorio/Practico_16.ipynb`

## Encuadre

Laboratorio organizado en **siete bloques** que recorren ~30 años de NLP comprimidos en un notebook. Desde tokenización clásica con NLTK hasta traducción multilingüe neural con NLLB-200, pasando por sentiment analysis con reglas y un clasificador completo Bag of Words. Cierra con 10 actividades evaluadas (3 sobre NLTK/spaCy, 1 sobre integración NLLB+VADER, 6 sobre BoW + n-grams).

Línea histórica que recorre el lab sin decirlo explícitamente:

```
1990s ─────── 2000s ────── 2010s ────── 2014 ────── 2017 ────── 2022 ──── hoy
  │             │             │           │            │           │
  NLTK         spaCy        word2vec    VADER       Transformer   NLLB-200
  reglas    pipelines      embeddings  reglas+lex  attention     200 idiomas
  +stats    industriales                                          + MoE
```

Para la teoría detrás de cada técnica ver la [clase 16](/clases/clase-16/).

## Resultados consolidados

Ejecutado end-to-end en Colab free tier (CPU + GPU T4 cuando disponible). **106 celdas** recorridas, **10 actividades** completadas, **7 papers fundacionales** descargados y analizados.

| Bloque | Celdas | Técnica principal | Output característico |
| --- | --- | --- | --- |
| 1 | 5-31 | NLTK clásico (tokenización, FreqDist, stemming) | Vocab Moby Dick: 17,200 tipos / 260,819 tokens |
| 2 | 32-42 | spaCy (POS, NER, dependency parsing) | 3 entidades NER detectadas: Apple→ORG, U.K.→GPE, $1 billion→MONEY |
| 3 | 43-52 | Actividades 1-3 (multiple choice) | Identificar opciones correctas vs trampas |
| 4 | 53-62 | NLLB-200 distilled 600M | Traducción ES↔EN funcional |
| 5 | 63-71 | VADER + Actividad 4 (NLLB+VADER) | F1 = 0.96 nativo en tweets |
| 6 | 72-92 | Bag of Words + n-grams + Naive Bayes | accuracy ~0.86 con GaussianNB |
| 7 | 93-105 | Actividades 5-10 (BoW conceptual) | Cierre evaluado del lab |

## Recursos del lab — Parte 1 (NLTK clásico)

{{< cards >}}
  {{< card link="nltk-tokenizacion" title="Tokenización con NLTK" subtitle="sent_tokenize (Punkt), word_tokenize (Treebank), TweetTokenizer" icon="academic-cap" >}}
  {{< card link="nltk-estadisticas" title="Estadísticas de texto" subtitle="FreqDist, Ley de Zipf, Ley de Heaps, dispersion plots" icon="academic-cap" >}}
  {{< card link="nltk-normalizacion" title="Normalización: stop-words y stemming" subtitle="NLTK stopwords español, Porter/Snowball/Lancaster, WordNetLemmatizer" icon="academic-cap" >}}
{{< /cards >}}

## Recursos del lab — Parte 2 (spaCy industrial)

{{< cards >}}
  {{< card link="spacy-pipeline" title="spaCy: POS, NER, Dependency Parsing" subtitle="spacy.load + displacy + iteración por Doc/Span/Token" icon="academic-cap" >}}
{{< /cards >}}

## Recursos del lab — Parte 3 (Actividades 1-3)

{{< cards >}}
  {{< card link="actividades-1-3" title="Actividades 1-3 (multiple choice)" subtitle="Tweets EN, correos formales ES, stop-words Wikipedia ES" icon="academic-cap" >}}
{{< /cards >}}

## Recursos del lab — Parte 4 (Modelos modernos)

{{< cards >}}
  {{< card link="nllb-traduccion" title="NLLB-200 traducción multilingüe" subtitle="Transformer MoE distilled 600M, FLORES-200, 200 idiomas" icon="academic-cap" >}}
  {{< card link="vader-sentiment" title="VADER + translate-then-analyze" subtitle="Lexicón 7500 entradas + 5 reglas heurísticas + Actividad 4" icon="academic-cap" >}}
{{< /cards >}}

## Recursos del lab — Parte 5 (BoW + clasificación)

{{< cards >}}
  {{< card link="bow-clasificacion" title="Bag of Words + N-grams + Naive Bayes" subtitle="Pipeline completo: SMS spam classification con sklearn" icon="academic-cap" >}}
  {{< card link="actividades-finales" title="Actividades 5-10 (BoW conceptual)" subtitle="Orden, n-grams, configuraciones, recomendación de técnicas avanzadas" icon="academic-cap" >}}
{{< /cards >}}

## Papers fundacionales

{{< cards >}}
  {{< card link="/papers/porter-stemmer-1980" title="Porter (1980) — An algorithm for suffix stripping" subtitle="El stemmer más usado de la historia, ~15k citas" icon="document" >}}
  {{< card link="/papers/wordnet-miller-1995" title="Miller et al. (1990/1995) — WordNet" subtitle="El recurso léxico más influyente del NLP, ~50k citas combinadas" icon="document" >}}
  {{< card link="/papers/nltk-bird-loper-2006" title="Bird & Loper (2006) — NLTK" subtitle="El toolkit pedagógico Python de NLP, ~6700 citas" icon="document" >}}
  {{< card link="/papers/punkt-kiss-strunk-2006" title="Kiss & Strunk (2006) — Punkt" subtitle="Sentence tokenization no supervisada multilingüe, 11 idiomas, ~1700 citas" icon="document" >}}
  {{< card link="/papers/twitter-pos-gimpel-2011" title="Gimpel et al. (2011) — Twitter POS Tagger" subtitle="Tagset de 25 etiquetas Twitter-específicas, base del TweetTokenizer NLTK" icon="document" >}}
  {{< card link="/papers/vader-hutto-gilbert-2014" title="Hutto & Gilbert (2014) — VADER" subtitle="Sentiment analysis rule-based, F1=0.96 en tweets, ~9k citas" icon="document" >}}
  {{< card link="/papers/nllb-team-2022" title="NLLB Team (2022) — No Language Left Behind" subtitle="Transformer MoE para 200 idiomas, +44% BLEU sobre SOTA" icon="document" >}}
{{< /cards >}}

## Fundamentos transversales

{{< cards >}}
  {{< card link="/fundamentos/tokenizacion-clasica" title="Tokenización clásica" subtitle="Los dos niveles (oración + palabra), tokenizadores comparados" icon="document-text" >}}
  {{< card link="/fundamentos/bag-of-words" title="Bag of Words" subtitle="Representación vectorial clásica, n-grams, TF-IDF" icon="document-text" >}}
  {{< card link="/fundamentos/sentiment-analysis" title="Sentiment Analysis" subtitle="Rule-based vs neural, translate-then-analyze, casos clínicos" icon="document-text" >}}
{{< /cards >}}

## Cross-links

- Teoría correspondiente: [Clase 16 — Introducción a NLP](/clases/clase-16/).
- Dominio relacionado: [Dominio texto](/dominios/texto/) — línea histórica completa del NLP.
- Lab anterior: [Lab 15 — Faster R-CNN](/laboratorios/lab-15/) (visión).
- Lab siguiente: pendiente en clase 17+.

---

> **Estado:** lab completo recorrido celda a celda. 10 actividades resueltas. 7 papers fundacionales descargados con análisis exhaustivo. Práctica local en [clase_16/practica/](https://github.com/) con diseño + plan de 35 tasks pendiente de ejecución (NLP clínico sobre MEDDOCAN + Cantemist + PharmaCoNER + Quijote).
