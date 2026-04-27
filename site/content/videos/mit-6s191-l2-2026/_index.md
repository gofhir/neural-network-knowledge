---
title: "MIT 6.S191 (2026): RNNs, Transformers y Attention"
weight: 20
sidebar:
  open: true
---

**Curso:** MIT 6.S191 - Introduction to Deep Learning (2026)
**Instructora:** Ava Amini
**Lecture:** 2 - Recurrent Neural Networks, Transformers, and Attention
**Fecha:** 5 de enero de 2026
**Slides oficiales:** [PDF local (4.4 MB)](/videos/mit-6s191-l2-2026/slides.pdf) - [Original MIT](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf)

{{< youtube d02VkQ9MP44 >}}

## Resumen

La edicion **2026** del segundo lecture de MIT 6.S191 fue significativamente reestructurada respecto a 2020 para reflejar la realidad post-LLM. Ya no es solo sobre RNNs: el lecture cubre toda la cadena conceptual del modelado de secuencias en una sola hora, desde la motivacion (audio, video, texto, secuencias biologicas) hasta el **Transformer** completo de Vaswani et al. (2017).

Ava Amini construye el material en tres bloques. Primero, las **RNNs vanilla**: por que las arquitecturas feed-forward fallan con secuencias, los cuatro requisitos clave (longitud variable, dependencias largas, orden, parameter sharing), BPTT y el problema de vanishing/exploding gradients. Segundo, las **soluciones arquitectonicas**: gradient clipping, LSTM y GRU, y la motivacion para abandonar la recurrencia debido al cuello de botella secuencial. Tercero, los **Transformers**: self-attention con queries/keys/values, scaled dot-product, multi-head attention, positional encoding sinusoidal, conexiones residuales, layer norm, y aplicaciones transversales (NLP con BERT/GPT, vision con ViT, biologia con AlphaFold).

Para nuestro curso UC, este lecture **complementa** las clases [11](/clases/clase-11/) (RNNs, Alvaro Soto) y [13](/clases/clase-13/) (Seq2Seq + Attention, Gabriel Sepulveda), aportando intuiciones visuales, demos en codigo y un ritmo pedagogico distinto al del curso. Ademas conecta directamente con la siguiente generacion arquitectonica (Transformers/LLMs), tema que en el programa UC aparece de forma mas distribuida.

## Diferencia con el video 2020

El video [`mit-6s191-rnn`](/videos/mit-6s191-rnn/) cubre el **mismo lecture** pero en su version 2020, que termina en LSTM/GRU y solo alude al Transformer al final. Este video 2026 extiende a self-attention y arquitectura Transformer completa - son recursos complementarios:

- **Empezar con el video 2020** si vienes sin base de RNN: el ritmo es mas pausado y la motivacion mas detallada.
- **Saltar directo a este 2026** si ya manejas RNN/LSTM y quieres entender Transformers en el mismo estilo pedagogico de MIT 6.S191.

{{< cards >}}
  {{< card link="notas" title="Notas" subtitle="Recorrido tematico de las 83 diapositivas" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Papers seminales y derivaciones de Transformer" icon="beaker" >}}
  {{< card link="glosario" title="Glosario" subtitle="Terminos RNN + Transformers (44 entradas)" icon="book-open" >}}
  {{< card link="/videos/mit-6s191-rnn" title="Video 2020 (RNN)" subtitle="Version anterior del mismo lecture" icon="film" >}}
  {{< card link="/clases/clase-11" title="Clase 11 (curso UC)" subtitle="RNN - material complementario" icon="academic-cap" >}}
  {{< card link="/clases/clase-13" title="Clase 13 (curso UC)" subtitle="Seq2Seq + Attention" icon="academic-cap" >}}
{{< /cards >}}

---

> Material adaptado de **MIT 6.S191 (2026) Lecture 2: Recurrent Neural Networks, Transformers, and Attention**, Ava Amini, 5 de enero de 2026.
> [Video](https://www.youtube.com/watch?v=d02VkQ9MP44) - [Slides oficiales](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf) - [Sitio del curso](https://introtodeeplearning.com/).
> Notas en espanol como elaboracion independiente. Sin afiliacion oficial con MIT.
