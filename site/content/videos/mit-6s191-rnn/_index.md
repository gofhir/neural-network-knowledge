---
title: "MIT 6.S191 (2020): Deep Sequence Modeling"
weight: 10
sidebar:
  open: true
---

**Curso:** MIT 6.S191 - Introduction to Deep Learning (2020)
**Instructora:** Ava Soleimany
**Lecture:** 2 - Deep Sequence Modeling / Recurrent Neural Networks
**Duracion:** ~45 min
**Video:** [YouTube](https://www.youtube.com/watch?v=SEnXr6v2ifU)
**Slides oficiales:** [PDF local (9 MB)](/videos/mit-6s191-rnn/slides.pdf) - [Original MIT](http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf)

{{< youtube SEnXr6v2ifU >}}

## Resumen

El segundo lecture del curso MIT 6.S191 (edicion 2020) cubre las **redes neuronales recurrentes** desde la motivacion hasta el mecanismo de atencion. Ava Soleimany construye el material en tres bloques: primero, por que las arquitecturas feed-forward fallan con datos secuenciales y como una RNN soluciona los cuatro requisitos clave (longitud variable, dependencias largas, orden, parameter sharing). Segundo, el problema del vanishing/exploding gradient que motiva las arquitecturas con compuertas (LSTM, GRU). Tercero, aplicaciones reales -- generacion de musica, clasificacion de sentimiento, traduccion automatica con encoder-decoder -- y la introduccion a attention como solucion al cuello de botella del context vector.

Para nuestro curso UC, este lecture **complementa** las clases 11 (RNNs, Alvaro Soto) y 13 (Seq2Seq y Attention): aporta intuiciones visuales, demos en codigo y un ritmo pedagogico distinto al enfoque mas formal del curso.

{{< cards >}}
  {{< card link="notas" title="Notas" subtitle="Recorrido tematico de las 93 diapositivas" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Papers, comparacion con clase 11, conceptos no cubiertos" icon="beaker" >}}
  {{< card link="glosario" title="Glosario" subtitle="Terminos clave en espanol e ingles" icon="book-open" >}}
  {{< card link="/clases/clase-11" title="Clase 11 (curso UC)" subtitle="Material complementario de Alvaro Soto" icon="academic-cap" >}}
{{< /cards >}}

---

> Material adaptado de **MIT 6.S191 (2020) Lecture 2: Deep Sequence Modeling**, Alexander Amini & Ava Soleimany.
> [Video](https://www.youtube.com/watch?v=SEnXr6v2ifU) - [Slides oficiales](http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L2.pdf) - [Sitio del curso](http://introtodeeplearning.com/2020/).
> Notas en espanol con investigacion complementaria. Sin afiliacion oficial con MIT.
