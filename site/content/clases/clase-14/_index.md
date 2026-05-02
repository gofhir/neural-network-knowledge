---
title: "Clase 14 - Transformers"
weight: 100
sidebar:
  open: true
---

**Profesor:** Felipe del Rio
**Fecha:** 2026-04-29

El Transformer (Vaswani et al. 2017) reemplazo las RNNs como arquitectura dominante en NLP y se extendio a vision (ViT), multi-modalidad (CLIP) y casi todo deep learning moderno. La clase recorre la motivacion, la mecanica de self-attention y multi-head attention, la arquitectura encoder-decoder, positional encodings, y como BERT, ViT y CLIP construyen sobre la misma idea base.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 111 diapositivas de la clase" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math detallado de scaled dot-product, positional encoding, CLIP y Relation Networks" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Construir el Transformer paso a paso en PyTorch: embeddings, dot product, cross-entropy, gradient descent, mini Word2Vec" icon="code" >}}
  {{< card link="/clases/clase-13" title="Clase anterior: Seq2Seq + Attention" subtitle="Bahdanau attention como precursor del Transformer" icon="arrow-left" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/self-attention" title="Self-Attention" subtitle="Q/K/V, scaled dot-product, multi-head -- corazon del Transformer" icon="book-open" >}}
  {{< card link="/fundamentos/transformer" title="Arquitectura Transformer" subtitle="Encoder-decoder, FFN, layer norm, residuals, masked y cross attention" icon="book-open" >}}
  {{< card link="/fundamentos/positional-encoding" title="Positional Encoding" subtitle="Sinusoidal, aprendido, RoPE, ALiBi" icon="book-open" >}}
  {{< card link="/fundamentos/embeddings-distribuidos" title="Embeddings Distribuidos" subtitle="Capa embedding, espacios semanticos, W2V/GloVe, tied embeddings" icon="book-open" >}}
  {{< card link="/fundamentos/pretraining-bert" title="Pre-training BERT" subtitle="Masked LM, NSP, fine-tuning, RoBERTa/ALBERT/DeBERTa" icon="book-open" >}}
  {{< card link="/fundamentos/vision-transformer" title="Vision Transformer" subtitle="Patches 16x16, [class] token, trade-off datos vs inductive bias" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-contrastivo" title="Aprendizaje Contrastivo (CLIP)" subtitle="InfoNCE simetrico, zero-shot, multimodal" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Mecanismo de Atencion" subtitle="Cross-attention en Bahdanau, precursor de self-attention" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/attention-is-all-you-need-vaswani-2017" title="Attention Is All You Need (2017)" subtitle="Vaswani et al. -- el paper fundacional del Transformer" icon="document-text" >}}
  {{< card link="/papers/bert-devlin-2018" title="BERT (2018)" subtitle="Devlin et al. -- pre-training bidireccional masivo para NLP" icon="document-text" >}}
  {{< card link="/papers/vit-dosovitskiy-2021" title="Vision Transformer (2021)" subtitle="Dosovitskiy et al. -- imagenes como secuencias de patches" icon="document-text" >}}
  {{< card link="/papers/clip-radford-2021" title="CLIP (2021)" subtitle="Radford et al. -- vision y lenguaje via contrastive learning" icon="document-text" >}}
  {{< card link="/papers/relation-networks-santoro-2017" title="Relation Networks (2017)" subtitle="Santoro et al. -- razonamiento relacional como modulo" icon="document-text" >}}
{{< /cards >}}

## Investigacion ampliada

{{< cards >}}
  {{< card link="wiki" title="Wiki de investigacion" subtitle="Dossier integrado: arquitecturas, papers, codigo en 3 frameworks, evolucion historica" icon="sparkles" >}}
{{< /cards >}}
