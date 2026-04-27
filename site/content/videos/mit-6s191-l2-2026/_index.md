---
title: "MIT 6.S191 (2026): Deep Sequence Modeling"
weight: 5
sidebar:
  open: true
---

**Curso:** MIT 6.S191 - Introduction to Deep Learning (2026)
**Instructora:** Ava Amini
**Lecture:** 2 - Deep Sequence Modeling
**Fecha:** 5 de enero de 2026
**Slides oficiales:** [PDF local (4.4 MB)](/videos/mit-6s191-l2-2026/slides.pdf) - [Original MIT](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf)

{{< youtube d02VkQ9MP44 >}}

## Atribucion

Material original de **MIT 6.S191** ([introtodeeplearning.com](https://introtodeeplearning.com)), Ava Amini, 5 de enero de 2026, distribuido bajo licencia del curso. Las notas en espanol son una elaboracion independiente, sin afiliacion oficial con MIT.

## Resumen

El segundo lecture del curso MIT 6.S191 edicion 2026 (titulado **Deep Sequence Modeling**) cubre toda la cadena conceptual del modelado de secuencias en una sola hora: motivacion (audio, video, texto, secuencias biologicas), construccion del **RNN vanilla** desde el perceptron, **Backpropagation Through Time**, problema de **vanishing/exploding gradients**, gating como solucion (LSTM/GRU mencionado brevemente), aplicaciones reales y el **pivot explicito a self-attention** que culmina con la arquitectura Transformer y sus aplicaciones modernas (BERT, GPT, AlphaFold, ViT).

El lecture mantiene el armazon clasico del 2020 (motivacion -> RNN -> BPTT -> vanishing -> aplicaciones) hasta la slide 60 y a partir de ahi pivota explicitamente hacia attention. **LSTMs y GRUs ocupan una sola slide** (la 54) y se mencionan apenas como teaser: el enfasis pedagogico se desplazo del gating a la atencion, reflejando como evoluciono el campo entre 2020 y 2026. La presentacion del scaled dot-product attention ocupa nueve slides (70-78) con un build progresivo paso a paso.

Para nuestro curso UC, este lecture **complementa** las clases [11](/clases/clase-11/) (RNNs, Alvaro Soto) y [13](/clases/clase-13/) (Seq2Seq + Attention, Gabriel Sepulveda), aportando intuiciones visuales y un ritmo pedagogico distinto al del curso.

## Estructura del lecture (83 slides en 12 bloques)

| Bloque | Slides | Tema |
|---|---|---|
| 1 | 1-8 | Motivacion: pelota -> secuencias en el mundo -> 4 tipos de mapping |
| 2 | 9-23 | Construccion del RNN: perceptron -> recurrence relation -> 3 ecuaciones |
| 3 | 24-29 | Computational graph, BPTT graph, codigo TF/PyTorch |
| 4 | 30-40 | Criterios de diseno + predict next word + embeddings |
| 5 | 41-44 | Backpropagation Through Time mecanico |
| 6 | 45-53 | Vanishing/exploding gradients y long-term dependencies |
| 7 | 54 | Gating como solucion (LSTM/GRU teaser - una sola slide) |
| 8 | 55-58 | Aplicaciones de RNN: musica y sentiment |
| 9 | 59-64 | Limitaciones de RNN y pivot explicito a self-attention |
| 10 | 65-69 | Self-attention conceptual: la analogia YouTube/search |
| 11 | 70-78 | Self-attention tecnico: los 4 pasos hasta el head completo |
| 12 | 79-83 | Multi-head, aplicaciones modernas (BERT/GPT/AlphaFold/ViT) y cierre |

## Diferencia con el video 2020

El video [`mit-6s191-rnn`](/videos/mit-6s191-rnn/) cubre el **mismo lecture** en su version 2020, que dedica una porcion sustancial a LSTM/GRU (forget/input/output gates desagregadas, ecuaciones, intuicion del cell state) y solo alude a attention al final. Este video 2026 colapsa LSTM/GRU a una sola slide y dedica nueve slides al scaled dot-product attention. Son recursos complementarios:

- **Empezar con el video 2020** si vienes sin base de RNN: el ritmo es mas pausado, las naive approaches (fixed window, bag of words) estan desarrolladas, y la motivacion para LSTM se cubre con detalle.
- **Saltar directo a este 2026** si ya manejas RNN/LSTM y quieres entender self-attention y Transformer en el mismo estilo pedagogico de MIT 6.S191.

## Recursos

{{< cards >}}
  {{< card link="notas" title="Notas" subtitle="Recorrido tematico de las 83 diapositivas en 12 bloques" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="BPTT, scaled dot-product, multi-head, papers seminales" icon="beaker" >}}
  {{< card link="glosario" title="Glosario" subtitle="80 terminos clave RNN + Attention + Transformer" icon="book-open" >}}
  {{< card link="slides.pdf" title="Slides PDF" subtitle="83 paginas, copia oficial MIT" icon="document" >}}
{{< /cards >}}

## Cross-links del curso

{{< cards >}}
  {{< card link="/clases/clase-11" title="Clase 11 - Redes Recurrentes" subtitle="RNN/LSTM/GRU - Alvaro Soto" icon="academic-cap" >}}
  {{< card link="/clases/clase-13" title="Clase 13 - Seq2Seq y Attention" subtitle="Encoder-decoder + Bahdanau attention - Gabriel Sepulveda" icon="academic-cap" >}}
  {{< card link="/fundamentos/redes-recurrentes" title="Fundamento: RNNs" subtitle="Definicion, configuraciones, aplicaciones" icon="book-open" >}}
  {{< card link="/fundamentos/lstm-gru" title="Fundamento: LSTM y GRU" subtitle="Compuertas, ecuaciones, comparacion" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Fundamento: Attention" subtitle="Bahdanau/Luong/scaled dot-product" icon="book-open" >}}
  {{< card link="/videos/mit-6s191-rnn" title="Video MIT 6.S191 (2020)" subtitle="Mismo curso, lecture 2 version 2020 - Ava Soleimany" icon="film" >}}
{{< /cards >}}

---

> Material adaptado de **MIT 6.S191 (2026) Lecture 2: Deep Sequence Modeling**, Ava Amini, 5 de enero de 2026.
> [Video](https://www.youtube.com/watch?v=d02VkQ9MP44) - [Slides oficiales](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf) - [Sitio del curso](https://introtodeeplearning.com/).
> Notas en espanol como elaboracion independiente. Sin afiliacion oficial con MIT.
