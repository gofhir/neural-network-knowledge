---
title: "Seq2Seq (Sequence to Sequence Learning)"
weight: 150
math: true
---

{{< paper-card
    title="Sequence to Sequence Learning with Neural Networks"
    authors="Sutskever, Vinyals, Le"
    year="2014"
    venue="NeurIPS 2014"
    pdf="/papers/seq2seq-sutskever-2014.pdf"
    arxiv="1409.3215" >}}
Demuestra que dos LSTMs profundas pueden traducir oraciones de lenguaje natural **end-to-end**, sin pipeline de SMT, alcanzando BLEU 34.8 en WMT'14 EN→FR -- superando por primera vez a un sistema estadistico tradicional. Introduce el truco de **invertir la secuencia fuente** para mejorar el aprendizaje y establece el blueprint del Seq2Seq usado durante toda la era pre-Transformer.
{{< /paper-card >}}

---

## Contexto

Antes de 2014, la traduccion automatica neural se limitaba a usar redes como **componentes auxiliares** dentro de pipelines de SMT (Cho et al. 2014 era contemporaneo). Las redes feedforward requerian entradas de tamano fijo, lo que las hacia inadecuadas para mapear oraciones de longitud variable a oraciones de longitud variable. Este paper de Google demostro que las **LSTMs profundas** pueden hacerlo end-to-end, marcando el cambio de paradigma a Neural Machine Translation (NMT).

---

## Ideas principales

### 1. Arquitectura Encoder-Decoder con LSTMs profundas

Dos LSTMs separadas:

- **Encoder LSTM**: lee la oracion fuente token por token, sin producir output. Su estado final $v$ es un **vector de tamano fijo** que codifica toda la oracion.
- **Decoder LSTM**: inicializado con $v$, genera la oracion objetivo token por token, hasta producir `<EOS>`.

$$p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_{t=1}^{T'} p(y_t \mid v, y_1, \ldots, y_{t-1})$$

```
Source: A B C <EOS>
        ↓ ↓ ↓  ↓
Encoder LSTM ───→ v (8000-dim)
                  ↓
                  Decoder LSTM
                  ↓ ↓ ↓ ↓
Target:           W X Y Z <EOS>
```

### 2. Profundidad importa

Usaron LSTMs de **4 capas** con 1000 celdas por capa. Cada capa adicional **mejoro perplexity en ~10%**. LSTMs poco profundas (1-2 capas) no alcanzan el mismo nivel.

### 3. Reversion de la secuencia fuente

El truco mas inesperado: **invertir el orden** de las palabras de la oracion fuente al alimentarlas al encoder. Es decir, en lugar de `A B C → W X Y`, alimentar `C B A → W X Y`.

Por que funciona: introduce **dependencias de corto plazo** entre cada palabra fuente y su correspondiente palabra objetivo (la primera palabra fuente queda mas cerca temporalmente de la primera palabra objetivo). Reduce el "minimum time lag" promedio que el LSTM debe atravesar, facilitando que el optimizador establezca la conexion entrada-salida.

Resultado: BLEU subio de 25.9 a **30.6** solo con esta inversion. Perplexity bajo de 5.8 a 4.7.

### 4. Vocabulario limitado + UNK

Vocabulario de 160K palabras (fuente) y 80K (objetivo). Palabras fuera de vocabulario se reemplazan por `<UNK>`. Esta limitacion es uno de los problemas que motivo despues los modelos por **subword units** (BPE, WordPiece).

### 5. Beam search

En inferencia, en lugar de elegir greedy el token mas probable en cada paso, se mantiene un beam de las $B$ hipotesis parciales mas probables (ellos usan $B = 12$). Mejora consistentemente la calidad.

---

## Resultados experimentales

Tarea: **WMT'14 English-French**, 12M oraciones de entrenamiento.

| Modelo | BLEU |
|---|---|
| Baseline SMT (Durrani et al. 2014) | 33.30 |
| Single forward LSTM (5) | 26.17 |
| Single reversed LSTM (5) | 30.59 |
| Ensemble of 5 reversed LSTMs | **34.81** |
| LSTM rescoring del 1000-best del baseline | 36.5 |
| Best WMT'14 (Durrani et al.) | 37.0 |

Notable:

- 384M parametros (32M encoder + 32M decoder + 1B word embeddings), entrenado en **8 GPUs durante 10 dias**.
- LSTM funciono bien incluso en **oraciones largas** (>35 palabras), refutando la asuncion de la epoca.
- Aprendio representaciones que son **invariantes a voz pasiva/activa** y sensibles al orden de palabras.

---

## Por que importa hoy

- Es el **paper fundacional** de Neural Machine Translation. Lo que vino despues (Bahdanau 2014 con atencion, Vaswani 2017 con Transformer) son evoluciones directas de este blueprint.
- El patron **encoder-decoder** se generalizo a virtualmente cualquier tarea seq2seq: resumen, dialogo, generacion de codigo, vision-language, audio-to-text.
- El **cuello de botella** del vector $v$ unico (toda la oracion comprimida en un vector fijo) motivo el desarrollo del **mecanismo de atencion** (Bahdanau, Cho, Bengio 2014).
- Los **trucos practicos** -- gradient clipping (norma 5), inicializacion uniforme [-0.08, 0.08], batches por longitud, lr scheduling -- siguen siendo estandar.
- Sutskever fue luego co-fundador de OpenAI; este trabajo establecio su reputacion en deep learning aplicado.

---

## Notas y enlaces

- El detalle del **reversal trick** esta en la Seccion 3.3. La justificacion intuitiva (short-term dependencies) es debatida -- algunos creen que es solo un artefacto del optimizador.
- La **representacion de oraciones aprendida** se visualiza en la Seccion 3.7 con PCA: oraciones con significado similar quedan cercanas, invariantes a voz activa/pasiva.
- Follow-up directo: **Bahdanau, Cho, Bengio 2014** "Neural Machine Translation by Jointly Learning to Align and Translate" -- elimina el cuello de botella con atencion.
- Reemplazo definitivo: **Vaswani et al. 2017** "Attention is All You Need" -- el Transformer.

Ver fundamentos: [Redes Recurrentes](/fundamentos/redes-recurrentes) · [LSTM y GRU](/fundamentos/lstm-gru).
