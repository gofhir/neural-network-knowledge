---
title: "Clase 26 - Meta-aprendizaje"
weight: 260
sidebar:
  open: true
---

**Profesor:** Pablo Messina
**Fecha:** 2026-06 (Topicos de Profundizacion)

Clase de profundizacion sobre **meta-aprendizaje** (*meta-learning*): el arte de **aprender a aprender**. Donde el deep learning clasico necesita muchos datos por tarea, el meta-aprendizaje entrena un modelo para que se **adapte rapido a tareas nuevas con pocos ejemplos**, reutilizando experiencia previa. La clase parte de la intuicion (¿como clasificamos a Braque vs Cezanne con solo 6 pinturas?), formaliza el problema como optimizacion bi-nivel y su relacion con few-shot learning (N-way K-shot), y recorre cinco algoritmos emblematicos en tres familias: **MAML** (optimization-based), **MANN** (memoria), y los metodos no-parametricos **Siamese / Matching / Prototypical Networks** (metric-based). Cierra con aplicaciones, con foco fuerte en **medicina** (segmentacion multi-centro, MICCAI 2025 best paper, denoising de fMRI).

La clase se apoya en la [Clase 13 (Transfer learning)](/clases/clase-13) — MAML lleva la inicializacion preentrenada al extremo de "inicializacion adaptable" — y en la [Clase 14 (Transformers)](/clases/clase-14) — Matching Networks y MANN son, matematicamente, atencion key-value sobre una coleccion, la misma idea que potencia el *in-context learning* de los LLMs.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 59 diapositivas: intuicion, formalizacion, few-shot, MAML, MANN, metodos no-parametricos, aplicaciones" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: optimizacion bi-nivel, meta-gradiente de segundo orden, FOMAML/Reptile, Bregman en Prototypical, memoria LRUA" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Episodios N-way K-shot, Prototypical y MAML en triple framework, Siamese y Matching Networks" icon="code" >}}
  {{< card link="/clases/clase-27" title="Clase siguiente: Redes Neuronales de Grafos" subtitle="GNN, message passing, GCN/GAT" icon="arrow-right" >}}
  {{< card link="/clases/clase-25" title="Clase anterior: Recomendacion con imagenes y texto" subtitle="Sistemas recomendadores multimodales" icon="arrow-left" >}}
  {{< card link="/clases/clase-13" title="Base: Transfer learning" subtitle="Reutilizar conocimiento previo via fine-tuning" icon="academic-cap" >}}
{{< /cards >}}

## Laboratorio

{{< cards >}}
  {{< card link="/laboratorios/lab-26" title="Lab 26 - Meta-aprendizaje: MAML y Prototypical Networks" subtitle="La contraparte practica: MAML (optimization-based) vs Prototypical (metric-based) sobre Omniglot y Mini-ImageNet, las 7 actividades resueltas" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/meta-aprendizaje" title="Meta-aprendizaje" subtitle="Aprender a aprender: bilevel optimization, las tres familias de metodos, in-context learning" icon="book-open" >}}
  {{< card link="/fundamentos/few-shot-learning" title="Few-shot Learning" subtitle="El planteamiento N-way K-shot, support/query, one-shot y zero-shot, benchmarks" icon="book-open" >}}
  {{< card link="/fundamentos/optimizacion-binivel" title="Optimizacion bi-nivel" subtitle="El marco matematico de MAML: inner/outer loop, meta-gradiente, Hessiano, FOMAML/Reptile/iMAML" icon="book-open" >}}
  {{< card link="/fundamentos/metric-learning" title="Metric Learning" subtitle="Aprender un espacio de embeddings donde la distancia codifica similitud" icon="book-open" >}}
  {{< card link="/fundamentos/memory-augmented-networks" title="Memory-Augmented Networks" subtitle="Memoria externa direccionable: NTM, MANN/LRUA, el puente a la atencion de Transformers" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer Learning" subtitle="El campo vecino: reutilizar features preentrenadas" icon="book-open" >}}
  {{< card link="/fundamentos/in-context-learning" title="In-Context Learning" subtitle="Meta-aprendizaje implicito en los LLMs" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/maml-finn-2017" title="MAML (2017)" subtitle="Finn et al. -- aprender una inicializacion adaptable, el algoritmo estrella optimization-based" icon="document-text" >}}
  {{< card link="/papers/mann-santoro-2016" title="MANN (2016)" subtitle="Santoro et al. -- one-shot learning con memoria externa (LRUA)" icon="document-text" >}}
  {{< card link="/papers/matching-networks-vinyals-2016" title="Matching Networks (2016)" subtitle="Vinyals et al. -- atencion sobre el support, creo el protocolo episodico y miniImageNet" icon="document-text" >}}
  {{< card link="/papers/prototypical-networks-snell-2017" title="Prototypical Networks (2017)" subtitle="Snell et al. -- prototipos como centroides, baseline de facto del metric-based" icon="document-text" >}}
  {{< card link="/papers/siamese-networks-koch-2015" title="Siamese Networks (2015)" subtitle="Koch et al. -- torres gemelas para one-shot, antecedente del deep metric learning" icon="document-text" >}}
  {{< card link="/papers/omniglot-lake-2015" title="Omniglot / BPL (2015)" subtitle="Lake et al. -- el benchmark fundacional del few-shot, concept learning a nivel humano" icon="document-text" >}}
  {{< card link="/papers/meta-learning-survey-hospedales-2020" title="Meta-Learning Survey (2020)" subtitle="Hospedales et al. -- la taxonomia canonica de 3 ejes del campo" icon="document-text" >}}
{{< /cards >}}

## Aplicaciones en medicina

{{< cards >}}
  {{< card link="/papers/meta-disentanglement-liu-2021" title="Meta + Disentanglement (2021)" subtitle="Liu et al. (MICCAI Oral) -- segmentacion con generalizacion de dominio multi-centro" icon="document-text" >}}
  {{< card link="/papers/metaseg-vyas-2025" title="MetaSeg (2025)" subtitle="Vyas et al. (MICCAI Best Paper) -- INR meta-aprendido, Dice de U-Net con 90% menos parametros" icon="document-text" >}}
  {{< card link="/papers/fmri-denoising-heo-2025" title="fMRI Denoising (2025)" subtitle="Heo et al. (MICCAI) -- domain adaptation con criteria shift entre centros" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/vision" title="Dominio: Vision" subtitle="Era Meta-aprendizaje y Few-shot (2015-2025): Omniglot, Siamese, Matching, MANN, MAML, Prototypical, meta-learning medico" icon="globe-alt" >}}
{{< /cards >}}
