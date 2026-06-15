---
title: "Clase 25 - Recomendación con Imágenes y Texto"
weight: 250
sidebar:
  open: true
---

**Profesores:** Julio Hurtado & Felipe del Río
**Institución:** Computer Science Department, PUC

Case Study del bloque de aplicaciones avanzadas. En lugar de presentar una técnica nueva, la clase **modela un problema real de punta a punta** siguiendo un framework de preguntas por etapa: definir el problema, mirar los datos, elegir el modelo, representar cada tipo de dato y medir el éxito. El problema elegido es un **sistema de recomendación multimodal** estilo Pinterest: recomendar *pins* (una imagen $x_j$ + un comentario $c_j$) a un usuario $u_i$ según sus interacciones previas, computando una relevancia

$$r_{ij} = h\big(g(u_i),\, f(x_j, c_j)\big)$$

donde $f$ representa el pin (CNN para la imagen + BERT para el texto → concatenación → capa densa = *pin representation*), $g$ representa al usuario como el conjunto de sus pins, y $h$ mide relevancia por **distancia en un espacio de embeddings aprendido** (metric learning). La arquitectura es, en el fondo, un **two-tower**: se entrena como clasificación de usuario (cross-entropy) para que los pins de un mismo usuario queden cercanos, y en inferencia se recomienda por mínima distancia.

La clase integra casi todo el curso: [CNNs](/fundamentos/redes-convolucionales) y [transfer learning](/fundamentos/transfer-learning) para imágenes, [BERT](/fundamentos/bert) y [Transformers](/clases/clase-14) para texto, [metric learning / triplet loss](/fundamentos/triplet-loss) para el espacio de embeddings, y cierra con las **métricas de ranking** (Precision@k, MAP, MRR, nDCG) que evalúan listas recomendadas.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 56 diapositivas: framework de preguntas, problem/data/model, representación de datos, métricas" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: formalización de r_ij, metric learning, two-tower y log-Q correction, invarianza a permutación, derivación de nDCG" icon="beaker" >}}
  {{< card link="/laboratorios/lab-25" title="Laboratorio 25: Recomendación multimodal" subtitle="AlexNet + BERT, proxy task, recomendación por similitud y dos bugs de métrica (nDCG 0.02 a 0.86)" icon="beaker" >}}
  {{< card link="/clases/clase-22" title="Clase previa integrada: Summarization" subtitle="T5, BERTSum, decoding, ROUGE" icon="arrow-left" >}}
  {{< card link="/clases/clase-20" title="Base: ELMo, BERT, GPT, ChatGPT" subtitle="BERT como encoder de texto del pin" icon="academic-cap" >}}
  {{< card link="/clases/clase-14" title="Base: Transformers" subtitle="Encoder para texto, sets y secuencias" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/recommender-systems" title="Sistemas de Recomendación" subtitle="Collaborative filtering, matrix factorization, deep recsys, cold start, evaluación" icon="book-open" >}}
  {{< card link="/fundamentos/ranking-metrics" title="Métricas de Ranking" subtitle="Precision@k, Recall@k, MAP, MRR, DCG/nDCG con math y ejemplos de la clase" icon="book-open" >}}
  {{< card link="/fundamentos/two-tower-retrieval" title="Two-Tower Retrieval" subtitle="Dual encoder, in-batch negatives, candidate generation, ANN serving" icon="book-open" >}}
  {{< card link="/fundamentos/triplet-loss" title="Triplet Loss y Metric Learning" subtitle="El espacio donde items co-preferidos quedan cerca" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-contrastivo" title="Aprendizaje Contrastivo" subtitle="Acercar positivos, alejar negativos (CLIP)" icon="book-open" >}}
  {{< card link="/fundamentos/bert" title="BERT (Encoder-only)" subtitle="Encoder del texto del pin" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Redes Convolucionales" subtitle="Encoder de la imagen (features 4096-d)" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer Learning" subtitle="Pretrain de CNN y BERT, freeze + fine-tune" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-datos" title="Representación de Datos" subtitle="Discretos, continuos, sets, secuencias, combinación" icon="book-open" >}}
{{< /cards >}}

## Papers citados en la clase

{{< cards >}}
  {{< card link="/papers/youtube-dnn-covington-2016" title="YouTube DNN (2016)" subtitle="Covington et al. — la inspiración: candidate generation + ranking de dos etapas" icon="document-text" >}}
  {{< card link="/papers/pinterest-dataset-2017" title="Pinterest Dataset (2017)" subtitle="Gómez et al. — el dataset usado: 70.200 pins, 117 usuarios, features CNN 4096-d" icon="document-text" >}}
{{< /cards >}}

## Papers canónicos del campo (complementarios)

{{< cards >}}
  {{< card link="/papers/matrix-factorization-koren-2009" title="Matrix Factorization (2009)" subtitle="Koren et al. — factores latentes, la era del Netflix Prize" icon="document-text" >}}
  {{< card link="/papers/bpr-rendle-2009" title="BPR (2009)" subtitle="Rendle et al. — ranking pairwise desde feedback implícito" icon="document-text" >}}
  {{< card link="/papers/dssm-huang-2013" title="DSSM (2013)" subtitle="Huang et al. — el ancestro del two-tower / dual encoder" icon="document-text" >}}
  {{< card link="/papers/wide-and-deep-cheng-2016" title="Wide & Deep (2016)" subtitle="Cheng et al. — memorización + generalización conjuntas" icon="document-text" >}}
  {{< card link="/papers/vbpr-he-2016" title="VBPR (2016)" subtitle="He & McAuley — recomendación visual con features CNN + BPR" icon="document-text" >}}
  {{< card link="/papers/neural-collaborative-filtering-he-2017" title="Neural CF (2017)" subtitle="He et al. — de matrix factorization a embeddings neuronales" icon="document-text" >}}
  {{< card link="/papers/deepfm-guo-2017" title="DeepFM (2017)" subtitle="Guo et al. — interacciones de features de orden bajo y alto" icon="document-text" >}}
  {{< card link="/papers/pinsage-ying-2018" title="PinSage (2018)" subtitle="Ying et al. — GNN web-scale, el recsys real de Pinterest" icon="document-text" >}}
  {{< card link="/papers/two-tower-yi-2019" title="Two-Tower (2019)" subtitle="Yi et al. — retrieval a gran escala con log-Q correction" icon="document-text" >}}
  {{< card link="/papers/ndcg-jarvelin-2002" title="DCG / nDCG (2002)" subtitle="Järvelin & Kekäläinen — la métrica de ranking de facto" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/recomendacion" title="Dominio: Recomendación" subtitle="Línea de tiempo: de GroupLens 1994 a la recomendación generativa con LLMs" icon="globe-alt" >}}
{{< /cards >}}
