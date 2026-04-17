---
title: "GRU y RNN Encoder-Decoder"
weight: 130
math: true
---

{{< paper-card
    title="Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
    authors="Cho, van Merrienboer, Gulcehre, Bahdanau, Bougares, Schwenk, Bengio"
    year="2014"
    venue="EMNLP 2014"
    pdf="/papers/gru-cho-2014.pdf"
    arxiv="1406.1078" >}}
Introduce dos contribuciones fundamentales: (1) la arquitectura **RNN Encoder-Decoder** que aprende a mapear secuencias de longitud variable a secuencias de longitud variable, y (2) la **Gated Recurrent Unit (GRU)** -- una alternativa mas simple y eficiente a LSTM con solo dos compuertas (reset y update) y un solo estado oculto.
{{< /paper-card >}}

---

## Contexto

En 2014, los sistemas de traduccion automatica estaban dominados por **Statistical Machine Translation (SMT)** basado en frases (phrase-based SMT). Las redes neuronales se usaban como **componentes auxiliares** (rescoring de hipotesis, language modeling) pero no como traductores end-to-end. Este paper -- junto con Sutskever et al. 2014 publicado meses despues -- inicio la era de la traduccion neural pura (NMT).

---

## Ideas principales

### 1. RNN Encoder-Decoder

Dos RNNs entrenadas conjuntamente:

- **Encoder**: lee la secuencia fuente $x_1, \ldots, x_T$ y produce un **vector de contexto** $c$ (ultimo estado oculto).
- **Decoder**: genera la secuencia objetivo $y_1, \ldots, y_{T'}$ condicionado en $c$ y los outputs previos.

$$h_t^{\text{dec}} = f(h_{t-1}^{\text{dec}}, y_{t-1}, c)$$

$$P(y_t \mid y_{t-1}, \ldots, y_1, c) = g(h_t^{\text{dec}}, y_{t-1}, c)$$

Entrenamiento por **maxima verosimilitud condicional** sobre pares (fuente, objetivo). Este patron es la base directa de Seq2Seq, atencion (Bahdanau 2014) y Transformer.

### 2. Gated Recurrent Unit (GRU)

Motivado por LSTM pero buscando algo mas simple. Dos compuertas:

$$
\begin{aligned}
r_t &= \sigma(W_r x_t + U_r h_{t-1}) \quad \text{(reset)} \\
z_t &= \sigma(W_z x_t + U_z h_{t-1}) \quad \text{(update)} \\
\tilde{h}_t &= \tanh(W x_t + U(r_t \odot h_{t-1})) \\
h_t &= z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
\end{aligned}
$$

- **Reset gate $r_t$**: cuando es 0, ignora el estado previo al calcular el candidato (permite "olvidar" informacion irrelevante).
- **Update gate $z_t$**: interpola entre conservar $h_{t-1}$ y adoptar el candidato $\tilde{h}_t$. Funciona como combinacion de input + forget gate de LSTM.

Ventajas vs LSTM:

- **Menos parametros** (3 conjuntos de pesos vs 4).
- **Sin distincion entre cell state y hidden state** -- un solo $h_t$.
- ~25% **mas rapido** en entrenamiento e inferencia.

### 3. Aprende representaciones semanticas continuas

El analisis cualitativo muestra que el encoder aprende representaciones de frases que **agrupan por significado**: paraphrases distintas terminan cerca en el espacio latente. Esto presagiaba word embeddings y sentence embeddings densos modernos.

---

## Resultados

Tarea: **WMT'14 English to French** (348M palabras de entrenamiento).

| Modelo | BLEU dev | BLEU test |
|---|---|---|
| Baseline SMT | 30.64 | 33.30 |
| Baseline + RNN encoder-decoder rescoring | 31.20 | 33.87 |
| Baseline + CSLM + RNN | 31.48 | 34.64 |
| Baseline + CSLM + RNN + word penalty | **31.50** | **34.54** |

El RNN no reemplazo aun a SMT (eso vendria con Sutskever 2014 y Bahdanau 2014), pero **mejoro el sistema cuando se usaba como feature de scoring**.

---

## Por que importa hoy

- **GRU es la alternativa estandar a LSTM** en aplicaciones donde la velocidad importa: streaming ASR, modelos en edge, prototipado rapido.
- La arquitectura **encoder-decoder** se generalizo a cualquier tarea seq2seq: traduccion, resumen, dialogo, code generation, vision-language.
- El paper introdujo **6 de los autores que luego co-fundaron Mila y Element AI**, incluyendo Bahdanau (atencion), Cho (NLP en NYU) y Bengio (Turing Award 2018).
- Junto con Sutskever 2014, marca el inicio de la **era neural** en NLP.

---

## Notas y enlaces

- La GRU se introduce en la Seccion 2.3 ("Hidden Unit that Adaptively Remembers and Forgets").
- Comparaciones LSTM vs GRU: ver Chung et al. 2014 "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (mismo grupo).
- El follow-up directo es **Bahdanau, Cho, Bengio 2014** "Neural Machine Translation by Jointly Learning to Align and Translate" -- introduce **atencion** sobre este encoder-decoder, eliminando el cuello de botella del vector $c$ unico.

Ver fundamentos: [LSTM y GRU](/fundamentos/lstm-gru) · [Redes Recurrentes](/fundamentos/redes-recurrentes).
