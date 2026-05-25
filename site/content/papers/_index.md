---
title: Papers
type: docs
weight: 40
sidebar:
  open: true
---

Análisis exhaustivos de los papers fundamentales referenciados en el diplomado, con enlaces a los PDFs originales y BibTeX para citar. Cada análisis cubre contexto histórico, contribución central, resultados experimentales, limitaciones y conexión con la clase o laboratorio donde aparece.

Organizados temáticamente. Para un recorrido cronológico, mira la línea de tiempo de cada **[dominio](/dominios)**.

## Fundamentos de entrenamiento y optimización

{{< cards >}}
  {{< card link="backpropagation-rumelhart-1986" title="Rumelhart 1986 - Backpropagation" subtitle="El algoritmo que destrabó las redes neuronales profundas" icon="adjustments" >}}
  {{< card link="dropout-srivastava-2014" title="Srivastava 2014 - Dropout" subtitle="Regularización por desactivación aleatoria de neuronas" icon="adjustments" >}}
  {{< card link="batch-norm-ioffe-2015" title="Ioffe & Szegedy 2015 - Batch Normalization" subtitle="Normaliza activaciones, acelera entrenamiento 10x" icon="adjustments" >}}
  {{< card link="adam-kingma-2015" title="Kingma 2015 - Adam" subtitle="Optimizador adaptativo con momentum y segundo momento" icon="trending-down" >}}
  {{< card link="saddle-points-dauphin-2014" title="Dauphin 2014 - Saddle Points" subtitle="El problema no son mínimos locales, son puntos silla" icon="trending-down" >}}
  {{< card link="loss-landscape-li-2018" title="Li 2018 - Loss Landscape Visualization" subtitle="Visualización 3D de la superficie de pérdida" icon="eye" >}}
  {{< card link="lookahead-zhang-2019" title="Zhang 2019 - Lookahead Optimizer" subtitle="Optimizador wrapper con fast/slow weights" icon="trending-down" >}}
  {{< card link="sgdr-loshchilov-2017" title="Loshchilov 2017 - SGDR (Cosine Annealing)" subtitle="Learning rate con warm restarts" icon="trending-down" >}}
  {{< card link="cyclical-lr-smith-2017" title="Smith 2017 - Cyclical Learning Rates" subtitle="LR cíclico, range test" icon="trending-down" >}}
  {{< card link="super-convergence-smith-2018" title="Smith 2018 - Super-Convergence" subtitle="OneCycle policy, entrenamiento 10x más rápido" icon="trending-down" >}}
  {{< card link="large-minibatch-sgd-goyal-2017" title="Goyal 2017 - Large Minibatch SGD" subtitle="Linear scaling rule, warmup, ImageNet en 1 hora" icon="trending-down" >}}
  {{< card link="mixup-zhang-2017" title="Zhang 2017 - Mixup" subtitle="Augmentation por combinación lineal de samples" icon="chip" >}}
{{< /cards >}}

## CNNs y visión por computadora

{{< cards >}}
  {{< card link="alexnet-krizhevsky-2012" title="Krizhevsky 2012 - AlexNet" subtitle="El paper que arrancó el deep learning moderno (ImageNet 2012)" icon="photograph" >}}
  {{< card link="vggnet-simonyan-2014" title="Simonyan 2014 - VGGNet" subtitle="Profundidad uniforme con filtros 3x3" icon="photograph" >}}
  {{< card link="googlenet-szegedy-2014" title="Szegedy 2014 - GoogLeNet" subtitle="Inception modules, eficiencia computacional" icon="photograph" >}}
  {{< card link="resnet-he-2015" title="He 2015 - ResNet" subtitle="Conexiones residuales, redes de 100+ capas" icon="photograph" >}}
  {{< card link="transferable-features-yosinski-2014" title="Yosinski 2014 - Transferable Features" subtitle="¿Qué tan transferibles son los features de CNNs?" icon="chip" >}}
{{< /cards >}}

## Detección de objetos y segmentación

{{< cards >}}
  {{< card link="faster-rcnn-ren-2015" title="Ren 2015 - Faster R-CNN" subtitle="Detector end-to-end con Region Proposal Network" icon="cube-transparent" >}}
  {{< card link="fpn-lin-2017" title="Lin 2017 - Feature Pyramid Networks" subtitle="Pirámide multi-escala con top-down + lateral" icon="cube-transparent" >}}
  {{< card link="mask-rcnn-he-2017" title="He 2017 - Mask R-CNN" subtitle="Segmentación de instancias, RoIAlign" icon="cube-transparent" >}}
  {{< card link="coco-lin-2014" title="Lin 2014 - Microsoft COCO" subtitle="El dataset estándar de detección y segmentación" icon="cube-transparent" >}}
{{< /cards >}}

## Pose estimation y modelado humano

{{< cards >}}
  {{< card link="openpose-cao-2017" title="Cao 2017 - OpenPose" subtitle="Bottom-up multi-persona en tiempo real, Part Affinity Fields" icon="user-group" >}}
  {{< card link="pifpaf-kreiss-2019" title="Kreiss 2019 - PifPaf" subtitle="Composite fields para pose en baja resolución y oclusión" icon="user-group" >}}
  {{< card link="vitpose-xu-2022" title="Xu 2022 - ViTPose" subtitle="Vision Transformer aplicado a pose estimation" icon="user-group" >}}
  {{< card link="blazepose-bazarevsky-2020" title="Bazarevsky 2020 - BlazePose" subtitle="Mobile single-person, MediaPipe Pose, 33 keypoints" icon="user-group" >}}
  {{< card link="densepose-guler-2018" title="Güler 2018 - DensePose" subtitle="Mapping denso de pixels al manifold UV del cuerpo" icon="user-group" >}}
  {{< card link="facenet-schroff-2015" title="Schroff 2015 - FaceNet" subtitle="Triplet loss, embeddings faciales de 128D" icon="user-group" >}}
  {{< card link="smpl-loper-2015" title="Loper 2015 - SMPL" subtitle="Skinned Multi-Person Linear Model, body mesh parametrizado" icon="user-group" >}}
{{< /cards >}}

## RNNs y secuencias

{{< cards >}}
  {{< card link="lstm-hochreiter-1997" title="Hochreiter 1997 - LSTM" subtitle="Long Short-Term Memory, gates para gradient flow" icon="refresh" >}}
  {{< card link="gru-cho-2014" title="Cho 2014 - GRU y RNN Encoder-Decoder" subtitle="Gated Recurrent Unit, primer encoder-decoder neuronal" icon="refresh" >}}
  {{< card link="difficulty-training-rnns-pascanu-2013" title="Pascanu 2013 - Difficulty Training RNNs" subtitle="Vanishing/exploding gradients, gradient clipping" icon="refresh" >}}
{{< /cards >}}

## Seq2Seq y mecanismo de atención

{{< cards >}}
  {{< card link="seq2seq-sutskever-2014" title="Sutskever 2014 - Seq2Seq" subtitle="Sequence to Sequence Learning, NMT con LSTMs" icon="sparkles" >}}
  {{< card link="bahdanau-attention-2015" title="Bahdanau 2015 - Attention (NMT)" subtitle="Soft attention que destrabó traducción de oraciones largas" icon="sparkles" >}}
  {{< card link="show-and-tell-vinyals-2015" title="Vinyals 2015 - Show and Tell" subtitle="Image captioning con CNN + LSTM encoder-decoder" icon="sparkles" >}}
  {{< card link="show-attend-tell-xu-2015" title="Xu 2015 - Show, Attend and Tell" subtitle="Captioning con visual attention" icon="sparkles" >}}
  {{< card link="pointer-generator-see-2017" title="See 2017 - Pointer-Generator" subtitle="Summarization híbrido: copia + generación" icon="sparkles" >}}
  {{< card link="bottom-up-attention-anderson-2018" title="Anderson 2018 - Bottom-Up/Top-Down Attention" subtitle="VQA y captioning con object features" icon="sparkles" >}}
{{< /cards >}}

## NLP clásico

{{< cards >}}
  {{< card link="porter-stemmer-1980" title="Porter 1980 - Porter Stemmer" subtitle="Stemming por reglas heurísticas en 5 fases" icon="document-text" >}}
  {{< card link="wordnet-miller-1995" title="Miller 1995 - WordNet" subtitle="Base de datos léxica con synsets y relaciones semánticas" icon="document-text" >}}
  {{< card link="nltk-bird-loper-2006" title="Bird & Loper 2006 - NLTK" subtitle="Natural Language Toolkit, plataforma pedagógica NLP" icon="document-text" >}}
  {{< card link="punkt-kiss-strunk-2006" title="Kiss & Strunk 2006 - Punkt" subtitle="Sentence boundary detection unsupervised multilingüe" icon="document-text" >}}
  {{< card link="twitter-pos-gimpel-2011" title="Gimpel 2011 - Twitter POS" subtitle="POS tagging adaptado a redes sociales (25 tags)" icon="document-text" >}}
  {{< card link="vader-hutto-gilbert-2014" title="Hutto 2014 - VADER" subtitle="Sentiment analysis rule-based para social media" icon="document-text" >}}
  {{< card link="nllb-team-2022" title="NLLB Team 2022 - No Language Left Behind" subtitle="MT para 202 lenguajes, FLORES-200, mixture-of-experts" icon="document-text" >}}
{{< /cards >}}

## Modelos de lenguaje y embeddings

{{< cards >}}
  {{< card link="nplm-bengio-2003" title="Bengio 2003 - NPLM" subtitle="Neural Probabilistic Language Model, primer LM neuronal" icon="document-text" >}}
  {{< card link="rnn-lm-mikolov-2010" title="Mikolov 2010 - RNN-LM" subtitle="LM con RNNs, mejor perplexity que n-gramas" icon="document-text" >}}
  {{< card link="word2vec-efficient-mikolov-2013" title="Mikolov 2013 ICLR - Word2Vec" subtitle="CBOW y Skip-gram, embeddings a escala masiva" icon="document-text" >}}
  {{< card link="word2vec-distributed-mikolov-2013" title="Mikolov 2013 NeurIPS - Word2Vec Distributed" subtitle="Negative sampling, subsampling, phrases" icon="document-text" >}}
  {{< card link="glove-pennington-2014" title="Pennington 2014 - GloVe" subtitle="Global Vectors, factorización de co-ocurrencias" icon="document-text" >}}
  {{< card link="skip-thought-kiros-2015" title="Kiros 2015 - Skip-Thought Vectors" subtitle="Sentence embeddings vía predicción de oraciones vecinas" icon="document-text" >}}
  {{< card link="sgns-implicit-mf-levy-goldberg-2014" title="Levy-Goldberg 2014 NeurIPS - SGNS as Implicit MF" subtitle="SGNS = factorización implícita de PMI shifted" icon="document-text" >}}
  {{< card link="linguistic-regularities-levy-goldberg-2014" title="Levy-Goldberg 2014 CoNLL - 3CosMul" subtitle="Fórmula multiplicativa para analogías" icon="document-text" >}}
  {{< card link="contrastive-analogies-ri-lee-verma-2023" title="Ri-Lee-Verma 2023 - Contrastive Analogies" subtitle="Teorema 1: analogías son líneas paralelas con factor ζ" icon="document-text" >}}
  {{< card link="analogies-explained-allen-hospedales-2019" title="Allen-Hospedales 2019 - Analogies Explained" subtitle="Análisis teórico de por qué funcionan los embeddings" icon="document-text" >}}
{{< /cards >}}

## Transformers, BERT y modelos contextualizados

{{< cards >}}
  {{< card link="attention-is-all-you-need-vaswani-2017" title="Vaswani 2017 - Attention Is All You Need" subtitle="El Transformer original, self-attention reemplaza recurrencia" icon="cube-transparent" >}}
  {{< card link="elmo-peters-2018" title="Peters 2018 - ELMo" subtitle="Deep contextualized word representations, primer embedding contextual" icon="document-text" >}}
  {{< card link="bert-devlin-2018" title="Devlin 2018 - BERT" subtitle="Pre-training bidirectional con MLM y NSP" icon="document-text" >}}
  {{< card link="vit-dosovitskiy-2021" title="Dosovitskiy 2021 - Vision Transformer" subtitle="Patches como tokens, Transformer aplicado a visión" icon="photograph" >}}
{{< /cards >}}

## Familia GPT y modelos generativos

{{< cards >}}
  {{< card link="gpt-1-radford-2018" title="Radford 2018 - GPT-1" subtitle="Improving Language Understanding by Generative Pre-Training" icon="document-text" >}}
  {{< card link="gpt-2-radford-2019" title="Radford 2019 - GPT-2" subtitle="Language Models are Unsupervised Multitask Learners (1.5B params)" icon="document-text" >}}
  {{< card link="gpt-3-brown-2020" title="Brown 2020 - GPT-3" subtitle="Language Models are Few-Shot Learners (175B params)" icon="document-text" >}}
  {{< card link="instructgpt-ouyang-2022" title="Ouyang 2022 - InstructGPT" subtitle="RLHF: alineamiento con preferencias humanas" icon="document-text" >}}
{{< /cards >}}

## Multimodal y razonamiento

{{< cards >}}
  {{< card link="clip-radford-2021" title="Radford 2021 - CLIP" subtitle="Contrastive Language-Image Pre-training, zero-shot vision" icon="academic-cap" >}}
  {{< card link="relation-networks-santoro-2017" title="Santoro 2017 - Relation Networks" subtitle="Razonamiento relacional como módulo entrenable" icon="academic-cap" >}}
{{< /cards >}}

## Foundation models y MLOps

{{< cards >}}
  {{< card link="foundation-models-bommasani-2021" title="Bommasani 2021 - Foundation Models" subtitle="Survey: oportunidades y riesgos de los modelos fundacionales" icon="academic-cap" >}}
  {{< card link="hidden-technical-debt-sculley-2015" title="Sculley 2015 - Hidden Technical Debt in ML" subtitle="El paper que originó MLOps: CACE, glue code, pipeline jungles" icon="cube-transparent" >}}
  {{< card link="challenges-deploying-ml-paleyes-2022" title="Paleyes 2022 - Challenges in Deploying ML" subtitle="Survey ACM CSUR con case studies industriales" icon="cube-transparent" >}}
  {{< card link="mlops-overview-kreuzberger-2023" title="Kreuzberger 2023 - MLOps Overview" subtitle="9 principios + 9 componentes + 7 roles + arquitectura formal" icon="cube-transparent" >}}
{{< /cards >}}
