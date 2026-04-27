---
title: "Clase 13 - Seq2Seq y Attention"
weight: 90
sidebar:
  open: true
---

**Profesor:** Gabriel Sepulveda
**Fecha:** 2026-04-22

Modelos Sequence-to-Sequence (Seq2Seq) usando RNNs con mecanismo de atencion. Encoder-decoder architecture, context vector, autoregressive decoding, Bahdanau additive attention, soft vs hard attention, y aplicaciones en traduccion automatica, summarization e image captioning.

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 40 diapositivas de la clase" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math detallado de Bahdanau attention, variantes y conexion al Transformer" icon="beaker" >}}
  {{< card link="/fundamentos/seq2seq" title="Fundamento: Seq2Seq" subtitle="Encoder-decoder, teacher forcing, beam search" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Fundamento: Attention" subtitle="Bahdanau/Luong/scaled dot-product, soft vs hard" icon="book-open" >}}
  {{< card link="/videos/mit-6s191-l2-2026" title="Video MIT 6.S191 (2026)" subtitle="Ava Amini - self-attention, multi-head, Transformer" icon="film" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/seq2seq-sutskever-2014" title="Seq2Seq (2014)" subtitle="Sutskever, Vinyals, Le" icon="document-text" >}}
  {{< card link="/papers/bahdanau-attention-2015" title="Bahdanau Attention (2015)" subtitle="Bahdanau, Cho, Bengio" icon="document-text" >}}
  {{< card link="/papers/show-attend-tell-xu-2015" title="Show, Attend and Tell (2015)" subtitle="Xu, Ba, Kiros, Cho, et al." icon="document-text" >}}
  {{< card link="/papers/pointer-generator-see-2017" title="Pointer-Generator (2017)" subtitle="See, Liu, Manning" icon="document-text" >}}
  {{< card link="/papers/bottom-up-attention-anderson-2018" title="Bottom-Up Attention (2018)" subtitle="Anderson, He, Buehler, et al." icon="document-text" >}}
{{< /cards >}}
