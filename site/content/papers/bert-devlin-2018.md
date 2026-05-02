---
title: "BERT (Pre-training Bidirectional Transformers)"
weight: 290
math: true
---

{{< paper-card
    title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    authors="Devlin, Chang, Lee, Toutanova"
    year="2019"
    venue="NAACL 2019"
    pdf="/papers/bert-devlin-2018.pdf"
    arxiv="1810.04805" >}}
Define el paradigma **pretrain-finetune** moderno para NLP. BERT es un encoder Transformer entrenado sobre texto sin etiquetar con dos objetivos auto-supervisados -- **Masked Language Model** (MLM) y **Next Sentence Prediction** (NSP) -- y luego ajustado con una capa minima por tarea. Avanzo el estado del arte en 11 benchmarks (GLUE, SQuAD 1.1/2.0, SWAG, NER) y se volvio la base de toda una familia de modelos posteriores.
{{< /paper-card >}}

---

## Contexto

A mediados de 2018 el pre-entrenamiento de representaciones para NLP coexistia en tres familias:

- **Word2Vec / GloVe** (Mikolov 2013, Pennington 2014): vectores fijos por token, sin contexto.
- **ELMo** (Peters 2018): features contextuales obtenidas concatenando un LSTM left-to-right y otro right-to-left **entrenados independientemente**. Bidireccional solo en superficie -- cada direccion no ve a la otra durante el pre-entrenamiento. Enfoque **feature-based**: los pesos pre-entrenados se congelan.
- **OpenAI GPT** (Radford 2018): Transformer decoder estrictamente **left-to-right**. Enfoque **fine-tuning**: todos los pesos se ajustan a la tarea downstream. La unidireccionalidad es subobtima para tareas sentence-level (entailment, sentiment) y especialmente para tareas token-level (QA, NER) donde el contexto derecho es informativo.

BERT mejora sobre cada uno: introduce un objetivo (**MLM**) que permite **bidireccionalidad profunda y conjunta** dentro de un Transformer encoder, con el formato fine-tuning de GPT y mas datos (BooksCorpus + Wikipedia, 3.3B palabras).

---

## Ideas principales

### 1. Encoder Transformer bidireccional

Apila $L$ bloques Transformer encoder (Vaswani 2017): self-attention multi-cabeza + FFN + residual + LayerNorm. Cada token atiende a **todos** los demas en cada capa -- nada se enmascara causalmente.

- **BERT-base**: $L=12$, $H=768$, $A=12$, $\sim$110M parametros (mismo tamano que GPT, para comparacion directa).
- **BERT-large**: $L=24$, $H=1024$, $A=16$, $\sim$340M parametros.

### 2. WordPiece tokenization

Vocabulario de 30k tokens construido por subwords WordPiece (Wu 2016). Permite cubrir morfologia sin OOV (palabras desconocidas se descomponen en piezas, marcadas con `##`).

### 3. Tres embeddings sumados

$$E_i = E^{token}_i + E^{segment}_i + E^{position}_i$$

- **Token embedding**: vector aprendido por pieza WordPiece.
- **Segment embedding**: $E_A$ o $E_B$ -- indica a cual de las dos oraciones pertenece el token (para tareas con pares).
- **Position embedding**: aprendido (no sinusoidal), hasta 512 posiciones.

Tokens especiales: `[CLS]` al inicio (su salida final $C \in \mathbb{R}^H$ se usa como representacion agregada para clasificacion), `[SEP]` separa oraciones, `[MASK]` marca tokens enmascarados durante MLM.

### 4. Masked Language Model (MLM, regla 80/10/10)

Se enmascara aleatoriamente el **15% de los tokens WordPiece**. Para los seleccionados:

- **80%** se reemplazan por `[MASK]`
- **10%** se reemplazan por un token aleatorio
- **10%** se dejan sin cambios

El modelo predice los tokens originales con cross-entropy. La razon de la mezcla: durante fine-tuning no existe `[MASK]`, asi que entrenar siempre con `[MASK]` crea un mismatch pretrain/finetune; el ruido aleatorio y el "dejarlo igual" obligan al modelo a mantener una representacion contextual de **todo** token.

### 5. Next Sentence Prediction (NSP)

Pares de oraciones $(A, B)$: 50% son contiguas en el corpus (`IsNext`), 50% B es aleatoria (`NotNext`). Clasificacion binaria desde $C$. Pensado para tareas que dependen de relacion entre oraciones (NLI, QA).

### 6. Fine-tuning para tareas downstream

Misma arquitectura, mismos pesos pre-entrenados; se agrega **una capa de salida** y se entrena end-to-end:

- **Single sentence classification** (SST-2, CoLA): softmax sobre $C$.
- **Sentence pair classification** (MNLI, QQP, MRPC): softmax sobre $C$ con segmentos A/B.
- **Question answering** (SQuAD): vectores aprendidos $S, E \in \mathbb{R}^H$ que predicen inicio y fin del span via dot-product con cada $T_i$.
- **Token tagging** (NER): clasificador por token sobre $T_i$.

Hiperparametros tipicos: batch 16/32, lr Adam $\in \{5\text{e-}5, 3\text{e-}5, 2\text{e-}5\}$, epochs 2-4. Fine-tuning es barato: $\sim$1 hora en una TPU.

---

## Resultados

### GLUE (Tabla 1)

| Sistema | Promedio |
|---|---|
| Pre-OpenAI SOTA | 74.0 |
| OpenAI GPT | 75.1 |
| **BERT-base** | **79.6** |
| **BERT-large** | **82.1** |

BERT-base supera a GPT por **+4.5** puntos manteniendo el mismo numero de parametros. La unica diferencia significativa es bidireccionalidad + MLM/NSP.

### SQuAD 1.1 (Tabla 2)

- BERT-large (single, con TriviaQA): **F1 91.8 / EM 85.1**
- Ensemble: **F1 93.2** -- supera al humano (91.2) y al top leaderboard (91.7).

### SQuAD 2.0 (Tabla 3)

- BERT-large: **F1 83.1** (+5.1 sobre el mejor previo).

### SWAG (Tabla 4)

Sentencia + 4 continuaciones, sentido comun: BERT-large 86.3 (+27.1 sobre ESIM+ELMo, +8.3 sobre GPT).

### NER CoNLL-2003 (Tabla 7)

BERT-large fine-tune: **F1 92.8** (state-of-the-art). Feature-based con suma ponderada de las ultimas 4 capas: F1 96.1 en dev, casi a la par con fine-tuning.

---

## Por que importa

BERT cristalizo el paradigma **pretrain-finetune** que domina NLP desde 2018. Cualquier laboratorio con un GPU mediano podia tomar un checkpoint pre-entrenado y obtener resultados competitivos en su tarea con horas de fine-tuning -- democratizo el acceso al estado del arte.

Familia directa de descendientes:

- **RoBERTa** (Liu 2019): mismo BERT pero entrenado mas tiempo, con mas datos, batch mayor, **sin NSP**, masking dinamico. SOTA sobre BERT.
- **ALBERT** (Lan 2019): factoriza embeddings y comparte parametros entre capas; introduce SOP en lugar de NSP.
- **DistilBERT** (Sanh 2019): destilacion -- 40% menos parametros, 60% mas rapido, 97% del rendimiento.
- **DeBERTa** (He 2020): disentangled attention (separa contenido y posicion) + enhanced mask decoder.
- **ELECTRA** (Clark 2020): replaced-token detection -- todas las posiciones aportan senal, no solo el 15% enmascarado.

Ademas, BERT inspiro adaptaciones a otros dominios: **BioBERT**, **SciBERT**, **ClinicalBERT**, **CodeBERT**, **mBERT** multilingue.

---

## Limitaciones

- **NSP cuestionado**: RoBERTa demuestra que removerlo no degrada (y a veces mejora). La correlacion con downstream era debil; SOP de ALBERT es una alternativa mejor.
- **No genera texto**: encoder-only y bidireccional -- BERT no es un LM autoregresivo. Para generacion se necesita GPT, T5 o BART.
- **Mismatch `[MASK]`**: el truco 80/10/10 mitiga pero no elimina la asimetria pretrain/finetune.
- **Cuadratico en longitud**: self-attention $O(n^2)$ limita las secuencias a 512 tokens. Longformer, BigBird, Reformer extendieron despues.
- **Sesgos del corpus**: BookCorpus + Wikipedia inglesa -- representacion sesgada de demografia, dialectos, dominios. Estudios posteriores documentaron sesgos de genero, raza y profesion en los embeddings.
- **Costo de pre-entrenamiento**: 4 dias en 16 TPUs para BERT-large; reproducirlo desde cero esta fuera del alcance de la mayoria.

---

## Notas y enlaces

- El paper es legible en una sentada: 16 paginas (Secciones 1-3 son la propuesta, 4-5 experimentos y ablations, apendices con masking procedure y hiperparametros).
- **Figura 1**: pretrain vs fine-tune (misma arquitectura, distintas cabezas).
- **Figura 2**: input representation = token + segment + position.
- **Figura 3**: BERT vs GPT vs ELMo lado a lado -- ilustra "deeply bidirectional" vs "shallowly bidirectional".
- Codigo y checkpoints originales: [github.com/google-research/bert](https://github.com/google-research/bert).
- Implementaciones modernas: [HuggingFace Transformers](https://huggingface.co/docs/transformers) (`BertModel`, `BertForSequenceClassification`, etc.).

Ver fundamentos: [Pre-training y BERT](/fundamentos/pretraining-bert) - [Transformer](/fundamentos/transformer) - [Transfer Learning](/fundamentos/transfer-learning) - [Clase 14](/clases/clase-14).
