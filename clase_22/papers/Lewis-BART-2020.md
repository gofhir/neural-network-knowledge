---
title: "BART: Denoising Sequence-to-Sequence Pre-training (Lewis et al., ACL 2020)"
slug: bart-lewis-2020
authors:
  - Mike Lewis
  - Yinhan Liu
  - Naman Goyal
  - Marjan Ghazvininejad
  - Abdelrahman Mohamed
  - Omer Levy
  - Ves Stoyanov
  - Luke Zettlemoyer
year: 2020
venue: ACL 2020 (arXiv:1910.13461, octubre 2019)
arxiv: "https://arxiv.org/abs/1910.13461"
pdf: "Lewis-BART-2020.pdf"
clase: 22
tags:
  - bart
  - denoising-autoencoder
  - seq2seq
  - summarization
  - encoder-decoder
  - text-infilling
  - pretraining
  - facebook-ai
---

# BART: Denoising Sequence-to-Sequence Pre-training para Generación, Traducción y Comprensión

## Resumen ejecutivo

**BART** (Bidirectional and Auto-Regressive Transformers) es un modelo pre-entrenado encoder-decoder presentado por Lewis et al. en ACL 2020. Su contribución conceptual central es **unificar dentro de un mismo objetivo el espíritu de BERT (encoder bidireccional) y de GPT (decoder autoregresivo)** mediante un esquema simple: **corromper texto con una función de ruido arbitraria y aprender a reconstruir el documento original**. La elegancia del diseño está en su flexibilidad — a diferencia de BERT (donde el ruido se limita a token masking) o de MASS (donde se reemplaza un span contiguo), BART permite **cualquier transformación**, incluso transformaciones que cambian la longitud del texto.

El paper aporta tres ejes empíricos:

1. **Ablation sistemática** de cinco funciones de ruido (token masking, token deletion, text infilling, sentence permutation, document rotation) bajo condiciones controladas. **Text infilling** — donde spans de longitud variable (Poisson, $\lambda=3$) son reemplazados por **un único** `[MASK]` — emerge como el objetivo más robusto.
2. **State-of-the-art en summarization abstractiva**: ROUGE-1 = 44.16 en CNN/DailyMail, ROUGE-1 = 45.14 en XSum (con ganancia de 6 puntos ROUGE sobre BERTSUMEXTABS, el mejor previo).
3. **Paridad con RoBERTa en tareas discriminativas** (GLUE, SQuAD), demostrando que la arquitectura encoder-decoder no penaliza performance en comprensión cuando se usa el decoder como cabecera adicional.

BART-large tiene aproximadamente **400M parámetros** (10% más que BERT-large por el cross-attention), se pre-entrena sobre 160GB del corpus de RoBERTa (news, books, stories, web text), 500k steps con batch size 8000, y usa el BPE tokenizer de GPT-2. El objetivo final combina **text infilling (30% de tokens) + sentence permutation**.

El impacto industrial es significativo: **`facebook/bart-large-cnn` es el modelo por defecto del pipeline `summarization` de HuggingFace** y sigue siendo en 2025-2026 una de las herramientas estándar para resumen automático en producción. La descendencia incluye mBART (multilingüe, 50+ idiomas), PLBART (lenguajes de programación), Pegasus (concurrente, summarization especializado) y la familia T0/FLAN que extiende el denoising seq2seq hacia instruction tuning.

---

## 1. Contexto histórico

### 1.1 La explosión del self-supervised pretraining (2018-2019)

Entre 2018 y 2019, el NLP atravesó su "momento ImageNet": una sucesión de objetivos de pre-entrenamiento self-supervised que reescribieron el state-of-the-art trimestre a trimestre.

| Modelo | Fecha | Arquitectura | Objetivo | Restricción |
|---|---|---|---|---|
| ELMo | feb 2018 | BiLSTM | Bidireccional shallow (concat L→R y R→L) | No interacción profunda izq-der |
| GPT-1 | jun 2018 | Decoder | LM causal izquierda → derecha | Sin contexto derecho |
| BERT | oct 2018 | Encoder | MLM 15% + NSP | Mal generador (predicciones independientes) |
| GPT-2 | feb 2019 | Decoder | LM causal escalado | Solo izquierda |
| XLNet | jun 2019 | Encoder + perm. | Permutation LM autoregresivo | Complejo, two-stream attention |
| RoBERTa | jul 2019 | Encoder | MLM con mejor receta (datos, batch, sin NSP) | Solo comprensión |
| SpanBERT | jul 2019 | Encoder | MLM con spans (geometric clamped) | Solo comprensión |
| MASS | jun 2019 | Encoder-decoder | Span enmascarado al encoder, predicción al decoder | Discriminative limitado |
| UniLM | mayo 2019 | Encoder con masks | MLM + LM unificado vía atención enmascarada | Predicciones condicionalmente independientes |
| T5 | oct 2019 | Encoder-decoder | Span corruption con sentinels separados + multitarea supervisada | Conceptualmente más complejo |
| **BART** | **oct 2019** | **Encoder-decoder** | **Denoising autoencoder con ruido arbitrario** | — |

BART y T5 son **contemporáneos** (ambos en octubre de 2019, arXiv:1910.13461 vs arXiv:1910.10683 con 18 días de diferencia). Comparten la apuesta por la arquitectura encoder-decoder pero llegan a ella por caminos distintos:

- **T5**: parte de la hipótesis de "todo es texto-a-texto" y propone un framework unificado. Pre-entrena con span corruption + multitarea supervisada masiva (C4, 750GB).
- **BART**: parte de la hipótesis de que **el ruido óptimo es una pregunta empírica abierta** y construye un framework que permite experimentar con cualquier ruido, dejando que los datos elijan el ganador.

### 1.2 La pregunta abierta que motiva el paper

Los autores se preguntan explícitamente: **¿qué función de ruido es óptima para downstream generation?** En 2019 ya existía evidencia de que cada objetivo era "bueno en lo suyo":

- BERT (MLM 15%): excelente en clasificación y SQuAD, pobre en generación.
- GPT (LM causal): bueno en generación, débil en QA span-prediction.
- XLNet (permutation LM): mejora ambos pero con costo arquitectónico alto.
- MASS (span seq2seq): bueno en MT, pero discriminative penalizado.

BART propone un **terreno común** donde estos objetivos se pueden comparar bajo idéntica arquitectura y datos. El trabajo aporta entonces no solo un modelo, sino **un protocolo de evaluación honesto** de pretraining objectives.

---

## 2. Arquitectura BART

### 2.1 Encoder-decoder Transformer estándar

BART implementa la arquitectura seq2seq de Vaswani et al. (2017) sin modificaciones estructurales mayores:

- **Encoder**: Transformer bidireccional (igual a BERT). Atiende a todos los tokens del input corrupto.
- **Decoder**: Transformer autoregresivo con dos tipos de atención por capa:
  1. **Self-attention causal** (igual a GPT): genera token a token de izquierda a derecha.
  2. **Cross-attention**: cada capa del decoder atiende a la última hidden layer del encoder. Esto es lo que distingue a BART de un decoder-only puro.

Las diferencias respecto a BERT son menores pero deliberadas:

1. **GeLU en vez de ReLU** (siguiendo GPT): activación más suave que mejora gradient flow.
2. **Inicialización $\mathcal{N}(0, 0.02)$** (siguiendo GPT).
3. **Sin FFN final antes de la word prediction** (BERT tiene una capa adicional, BART no).
4. **Sin Next Sentence Prediction** (NSP): replicando la decisión de RoBERTa.

### 2.2 Tamaños y cómputo

| Variante | Encoder layers | Decoder layers | Hidden | Params (~) |
|---|---|---|---|---|
| BART-base | 6 | 6 | 768 | 140M |
| BART-large | 12 | 12 | 1024 | **400M** |

Para referencia comparativa:

- BERT-large: 24 encoder layers, ~340M params.
- T5-base: 12+12, ~220M (FFN más pequeña).
- T5-large: 24+24, ~770M.
- T5-11B: variante gigante.

BART-large tiene **aproximadamente 10% más parámetros que BERT-large equivalente** según los autores, debido al cross-attention adicional en cada capa del decoder.

### 2.3 Comparación esquemática con BERT y GPT (Figura 1 del paper)

La Figura 1 del paper presenta una comparación visual elegante:

- **(a) BERT**: encoder bidireccional. Predice tokens enmascarados **independientemente** ($P(x_i) = P(x_i \mid x_{\setminus i})$ por separado para cada $i$). No sirve para generación porque no hay decodificación autoregresiva.
- **(b) GPT**: decoder autoregresivo. Predice $P(x_t \mid x_{<t})$ condicionado **solo en contexto izquierdo**. No aprende interacciones bidireccionales.
- **(c) BART**: combina ambos. El encoder bidireccional ingiere el documento corrupto; el decoder autoregresivo genera el documento original. **El input al encoder no necesita estar alineado con el output del decoder** — esto permite ruidos que cambian longitud (deletion, text infilling, sentence permutation).

Este último punto es crítico: en BERT, si enmascaras `[MASK]`, el modelo predice un token en esa misma posición. En BART con text infilling, un solo `[MASK]` puede expandirse a cero, uno, dos o más tokens, porque el decoder genera longitud libre.

### 2.4 Objetivo de pre-entrenamiento

BART optimiza la **negative log-likelihood del documento original** dado el documento corrupto:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{|x|} \log P_\theta(x_t \mid x_{<t}, g(x))
$$

donde $x$ es el documento original, $g(\cdot)$ es la función de ruido aplicada y $x_{<t}$ son los tokens ya generados por el decoder. En el caso extremo donde $g(x)$ destruye toda la información del input, BART se reduce a un language model puro. En el caso extremo donde $g$ es la identidad, se reduce a un autoencoder trivial. La gracia está en el medio.

---

## 3. Funciones de ruido evaluadas

El paper estudia cinco transformaciones de ruido (Figura 2), que pueden combinarse:

### 3.1 Token Masking (estilo BERT)

Tokens individuales se reemplazan por `[MASK]`. Es la implementación canónica de Devlin et al. (2019).

```
A B C D E . → A [MASK] C [MASK] E .
```

El modelo aprende: dado contexto bidireccional, ¿qué tokens fueron ocultados?

### 3.2 Token Deletion

Tokens son **borrados**, no enmascarados. El modelo no recibe el placeholder; debe descubrir **qué falta y dónde**.

```
A B C D E . → A C E .
```

A diferencia de masking, deletion fuerza al modelo a **razonar sobre la posición de las ausencias** además del contenido. Empíricamente, el paper observa que deletion supera a masking en tareas de generación (ver Tabla 1: token deletion vs token masking en XSum, ConvAI2, CNN/DM en BART-base).

### 3.3 Text Infilling (la contribución más original)

Spans de longitud variable se reemplazan por **un solo** `[MASK]`. Las longitudes se muestrean de una Poisson con $\lambda = 3$:

$$
L \sim \text{Poisson}(3), \qquad P(L=k) = \frac{3^k e^{-3}}{k!}
$$

Esto da una distribución con media 3, varianza 3, y soporte en $\{0, 1, 2, \ldots\}$. Los spans de longitud 0 corresponden a **inserción** de un `[MASK]`: el modelo debe aprender que ese `[MASK]` no corresponde a nada y eliminarlo en la reconstrucción.

```
A B C D E . → A [MASK] D [MASK] E .
```

Aquí el primer `[MASK]` reemplaza el span `B C` (longitud 2) y el segundo es una inserción (longitud 0).

**Distinción crítica con SpanBERT y T5**:

- **SpanBERT** (Joshi et al., 2019): spans con longitud de **geometric clamped distribution**, reemplazados por **una secuencia de `[MASK]` de la misma longitud que el span**. El modelo conoce de antemano cuántos tokens predecir.
- **T5** (Raffel et al., 2019): spans reemplazados por **sentinel tokens distintos** (`<extra_id_0>`, `<extra_id_1>`, ...). Cada sentinel es un placeholder identificable.
- **BART text infilling**: **un solo `[MASK]` opaco**. El modelo debe **inferir cuántos tokens van** en cada hueco.

Esta diferencia es sutil pero importante: text infilling de BART obliga a **modelar la distribución sobre longitudes de span**, lo cual es relevante para tareas generativas donde la longitud del output no está predeterminada (resumen, traducción, respuesta libre).

### 3.4 Sentence Permutation

El documento se segmenta en oraciones (heurística: por puntos), y las oraciones se mezclan aleatoriamente.

```
"A. B C. D E." → "D E. A. B C."
```

El modelo debe **restaurar el orden discursivo original**. Esto incentiva la captura de coherencia inter-oracional, contexto narrativo y estructura discursiva.

### 3.5 Document Rotation

Se elige un token uniformemente al azar y el documento se **rota** circularmente para empezar en ese token.

```
"A B C . D E ." → "C . D E . A B"
```

El modelo aprende a identificar el **inicio del documento**.

### 3.6 Composabilidad

Estas cinco transformaciones son **composables**: pueden combinarse en cualquier orden. El paper explora combinaciones, en particular **text infilling + sentence permutation**, que termina siendo la receta del modelo final.

---

## 4. Fine-tuning para tareas downstream

BART admite cuatro estrategias de fine-tuning según el tipo de tarea.

### 4.1 Sequence Classification (GLUE, MNLI, RTE)

El **mismo input** se alimenta al encoder y al decoder (input replicado). La representación del **último token del decoder** alimenta un clasificador lineal multi-clase.

```
Encoder input:  [s] tokens... [/s]
Decoder input:  [s] tokens... [/s] [class]
Predicción:     hidden_state([class]) → Linear → softmax
```

El truco está en añadir un token adicional al final del decoder input para que su representación pueda atender (vía self-attention causal) a **toda la secuencia procesada**. Esto difiere de BERT, que usa `[CLS]` al inicio.

### 4.2 Token Classification (SQuAD endpoint)

Para tareas como predicción de span endpoints en SQuAD, el documento completo se alimenta a encoder y decoder, y se usa el **top hidden state del decoder para cada token** como su representación.

### 4.3 Sequence Generation (XSum, CNN/DM, ELI5, ConvAI2)

Esta es la modalidad natural de BART: encoder recibe el input source, decoder genera el output autoregresivamente. **No requiere modificaciones arquitectónicas** porque el pre-training mismo es un objetivo seq2seq.

Hiperparámetros típicos de generación:

- **Label smoothing** $\epsilon = 0.1$ (Pereyra et al., 2017).
- **Beam search** con beam size = 5.
- **Trigram blocking**: se eliminan trigramas duplicados en el beam.
- **min-len, max-len, length penalty** tuneados en validación (siguiendo Fan et al., 2017).

### 4.4 Machine Translation (bridge architecture)

Esta es la propuesta más exótica del paper. Para traducir, por ejemplo, rumano → inglés, se hace lo siguiente:

1. **BART se mantiene como decoder pre-entrenado en inglés** (encoder + decoder).
2. Se reemplaza la **capa de embeddings del encoder de BART** por un nuevo encoder pequeño (6 layers Transformer) inicializado aleatoriamente.
3. Este nuevo encoder traduce rumano → un "ruido en inglés" que BART puede de-noise hacia inglés limpio.
4. Entrenamiento en dos pasos:
   - **Paso 1**: congelar la mayoría de los parámetros de BART. Solo entrenar el nuevo encoder + positional embeddings de BART + matriz de proyección de input self-attention de la primera capa del encoder de BART.
   - **Paso 2**: descongelar todo, entrenar end-to-end por unas pocas iteraciones.

Este esquema se llama **bridge architecture**: el encoder externo es el "puente" entre el idioma fuente y el "inglés ruidoso" que el modelo pre-entrenado entiende. El paper reporta +1.1 BLEU sobre baseline en WMT16 RO-EN (37.96 vs 36.80).

**Limitación honesta**: el método requiere **back-translation data** para funcionar bien y es propenso a overfitting. No escala fácilmente a múltiples pares de idiomas (cada par requiere su propio encoder bridge). Es por esto que mBART (Liu et al., 2020) abandona este enfoque y pre-entrena multilingual desde cero.

---

## 5. Pre-training corpus y configuración

### 5.1 Datos

BART-large se entrena sobre el **corpus de RoBERTa**: 160GB en total, combinando:

- News (CC-News).
- Books (Toronto BookCorpus).
- Stories (Stories from Trinh & Le, 2018).
- Web text (OpenWebText, Common Crawl filtrado).

### 5.2 Hiperparámetros

- **Batch size**: 8000.
- **Steps**: 500,000.
- **Tokenizer**: byte-pair encoding de GPT-2 (~50K vocab).
- **Masking ratio**: 30% de tokens enmascarados (más alto que el 15% de BERT, justificado por el span infilling: enmascarar spans de 3 tokens en promedio implica que cada span "consume" más tokens y se necesita mayor ratio para ver suficiente señal).
- **Sentence permutation**: aplicada a **todas** las oraciones.
- **Dropout**: deshabilitado en el último 10% de los training steps (helper para "settle" el modelo en datos limpios).

### 5.3 Objetivo final

Después de las ablations (Sección 6), el objetivo elegido para BART-large es:

$$
\text{Final objective} = \text{Text Infilling (30\% tokens, Poisson } \lambda=3\text{)} + \text{Sentence Permutation (all sentences)}
$$

---

## 6. Ablation de funciones de ruido (Sección 4 del paper)

Esta es la sección más informativa del trabajo. Los autores comparan **bajo idéntica arquitectura, datos y procedimiento de fine-tuning** seis objetivos previos y seis variantes de BART, todos en modelo base (6+6 layers, hidden 768) entrenados 1M steps sobre books+Wikipedia.

### 6.1 Baselines re-implementados

1. **Language Model** (GPT-style): decoder-only causal, sin cross-attention.
2. **Permuted LM** (XLNet-style): 1/6 de tokens predichos autoregresivamente en orden permutado. Sin relative positional embeddings ni segment recurrence (simplificación).
3. **Masked LM** (BERT-style): 15% tokens enmascarados, predicción independiente.
4. **Multitask Masked LM** (UniLM-style): MLM con varias máscaras de atención (1/6 L→R, 1/6 R→L, 1/3 sin máscara, 1/3 prefijo + L→R).
5. **Masked Seq-to-Seq** (MASS-style): span de 50% tokens enmascarado, seq2seq predice el span.

### 6.2 Tabla 1: resultados (SQuAD F1, MNLI Acc, ELI5/XSum/ConvAI2/CNN-DM PPL)

| Objetivo | SQuAD F1 | MNLI Acc | ELI5 PPL | XSum PPL | ConvAI2 PPL | CNN/DM PPL |
|---|---:|---:|---:|---:|---:|---:|
| BERT Base (publicado) | 88.5 | 84.3 | — | — | — | — |
| Masked LM | 90.0 | 83.5 | 24.77 | 7.87 | 12.59 | 7.06 |
| Masked Seq2Seq | 87.0 | 82.1 | 23.40 | 6.80 | 11.43 | 6.19 |
| Language Model | 76.7 | 80.1 | **21.40** | 7.00 | 11.51 | 6.56 |
| Permuted LM | 89.1 | 83.7 | 24.03 | 7.69 | 12.23 | 6.96 |
| Multitask Masked LM | 89.2 | 82.4 | 23.73 | 7.50 | 12.39 | 6.74 |
| **BART** w/ Token Masking | 90.4 | 84.1 | 25.05 | 7.08 | 11.73 | 6.10 |
| **BART** w/ Token Deletion | 90.4 | 84.1 | 24.61 | 6.90 | 11.46 | 5.87 |
| **BART** w/ Text Infilling | **90.8** | 84.0 | 24.26 | **6.61** | **11.05** | 5.83 |
| **BART** w/ Document Rotation | 77.2 | 75.3 | 53.69 | 17.14 | 19.87 | 10.59 |
| **BART** w/ Sentence Shuffling | 85.4 | 81.5 | 41.87 | 10.93 | 16.67 | 7.89 |
| **BART** w/ Text Infilling + Sentence Shuffling | **90.8** | 83.8 | 24.17 | 6.62 | 11.12 | **5.41** |

(Menor PPL es mejor. Mayor F1/Acc es mejor.)

### 6.3 Hallazgos clave

**Hallazgo 1 — La performance varía mucho por tarea**.
ELI5 (long-form QA, output débilmente condicionado por input) favorece al language model puro (21.40 PPL). SQuAD (extractive QA, input crítico) favorece a modelos con encoder bidireccional. **No existe un único objetivo dominante**.

**Hallazgo 2 — Token masking o deletion es crucial**.
Document rotation y sentence shuffling **aisladas** dan pésimos resultados (SQuAD 77.2 y 85.4 respectivamente). La señal de aprendizaje está en **reconstruir tokens, no posiciones globales**. Las transformaciones de gran escala (orden de oraciones, inicio del documento) **no son suficientes por sí solas**.

**Hallazgo 3 — Pre-training left-to-right ayuda a generación**.
Masked LM y Permuted LM son los peores en generación (XSum, CNN/DM PPL más alto entre los métodos con masking). Esto se debe a que **no incluyen pre-training autoregresivo izquierda-derecha**, que es la modalidad nativa de generación. BART y MASS sí lo hacen y obtienen mejores PPL generativos.

**Hallazgo 4 — Encoders bidireccionales son cruciales para SQuAD**.
El Language Model puro obtiene 76.7 F1 en SQuAD vs 90.4 de BART con token masking. **El contexto futuro es esencial para QA extractive**. Pero BART logra performance comparable a BERT en SQuAD con **solo la mitad de capas bidireccionales** (12 vs 24), gracias a la sinergia encoder-decoder.

**Hallazgo 5 — El objetivo no es el único factor**.
El Permuted LM (re-implementado) obtiene 89.1 F1 en SQuAD vs 89.0 reportado por XLNet original. Pero XLNet usa además relative position embeddings y segment recurrence; cuando esos se omiten (como en la re-implementación), parte de la ganancia se pierde. **Hay que separar contribución del objetivo de contribución arquitectónica**.

**Hallazgo 6 — Text infilling es el ganador robusto**.
En 4 de las 6 tareas evaluadas, BART con text infilling (solo o combinado con sentence shuffling) gana o empata. Es el único objetivo que es **simultáneamente bueno en clasificación y generación**.

**Hallazgo 7 — Sentence shuffling agrega valor en summarization larga**.
Text infilling + Sentence shuffling baja PPL de CNN/DM de 5.83 a 5.41 (mejor resultado de la tabla). En XSum no mejora (6.61 vs 6.62, prácticamente empate). Los autores hipotetizan que la mejora viene del razonamiento de orden discursivo, importante en resúmenes largos.

---

## 7. Resultados a gran escala (BART-large)

### 7.1 Tareas discriminativas (Tabla 2 del paper)

| Modelo | SQuAD 1.1 EM/F1 | SQuAD 2.0 EM/F1 | MNLI m/mm | SST | QQP | QNLI | STS-B | RTE | MRPC | CoLA |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT | 84.1/90.9 | 79.0/81.8 | 86.6/— | 93.2 | 91.3 | 92.3 | 90.0 | 70.4 | 88.0 | 60.6 |
| UniLM | — | 80.5/83.4 | 87.0/85.9 | 94.5 | — | 92.7 | — | 70.9 | — | 61.1 |
| XLNet | **89.0**/94.5 | 86.1/88.8 | **89.8**/— | 95.6 | 91.8 | 93.9 | 91.8 | 83.8 | 89.2 | 63.6 |
| RoBERTa | 88.9/**94.6** | **86.5**/**89.4** | **90.2/90.2** | 96.4 | 92.2 | 94.7 | **92.4** | 86.6 | **90.9** | **68.0** |
| **BART** | 88.8/**94.6** | 86.1/89.2 | 89.9/90.1 | **96.6** | **92.5** | **94.9** | 91.2 | **87.0** | 90.4 | 62.8 |

BART obtiene performance **comparable a RoBERTa y XLNet en todas las tareas discriminativas**. La diferencia es típicamente menor a 1 punto. Esto valida la tesis central: **el pre-training de BART no sacrifica comprensión por generación**.

### 7.2 Summarization (Tabla 3 del paper)

| Modelo | CNN/DM R1 | R2 | RL | XSum R1 | R2 | RL |
|---|---:|---:|---:|---:|---:|---:|
| Lead-3 baseline | 40.42 | 17.62 | 36.67 | 16.30 | 1.60 | 11.95 |
| PTGEN (See et al., 2017) | 36.44 | 15.66 | 33.42 | 29.70 | 9.21 | 23.24 |
| PTGEN+COV | 39.53 | 17.28 | 36.38 | 28.10 | 8.02 | 21.72 |
| UniLM | 43.33 | 20.21 | 40.51 | — | — | — |
| BERTSUMABS (Liu & Lapata, 2019) | 41.72 | 19.39 | 38.76 | 38.76 | 16.33 | 31.15 |
| BERTSUMEXTABS | 42.13 | 19.60 | 39.18 | 38.81 | 16.50 | 31.27 |
| **BART** | **44.16** | **21.28** | **40.90** | **45.14** | **22.27** | **37.25** |

En CNN/DM (más extractivo), BART supera a BERTSUMEXTABS por ~2 puntos ROUGE-1. En XSum (más abstractivo), la ganancia es de **~6.3 puntos ROUGE-1** (45.14 vs 38.81). Esta es la ganancia más espectacular del paper.

**¿Por qué XSum se beneficia más?** XSum tiene resúmenes muy cortos (una oración) y altamente abstractivos — las palabras del resumen rara vez aparecen en el documento fuente. Modelos extractivos (Lead-3) obtienen apenas R1=16.30. Esto requiere **paráfrasis genuina, conocimiento del mundo y compresión semántica** — exactamente lo que el pre-training denoising entrena.

### 7.3 Diálogo (Tabla 4 — ConvAI2)

| Modelo | Valid F1 | Valid PPL |
|---|---:|---:|
| Seq2Seq + Attention | 16.02 | 35.07 |
| Best System (anterior) | 19.09 | 17.51 |
| **BART** | **20.72** | **11.85** |

### 7.4 ELI5 (Tabla 5 — long-form abstractive QA)

| Modelo | R1 | R2 | RL |
|---|---:|---:|---:|
| Best Extractive | 23.5 | 3.1 | 17.5 |
| Language Model | 27.8 | 4.7 | 23.1 |
| Seq2Seq | 28.3 | 5.1 | 22.8 |
| Seq2Seq Multitask | 28.9 | 5.4 | 23.1 |
| **BART** | **30.6** | **6.2** | **24.3** |

Aquí BART mejora +1.2 ROUGE-L sobre Seq2Seq Multitask. Es notable que en la ablation base, BART perdía con language model puro en ELI5 PPL; pero en escala grande con datos completos, BART recupera ventaja. Esto sugiere que **el beneficio del encoder bidireccional aparece con suficiente capacidad y datos**.

### 7.5 Traducción (Tabla 6 — WMT16 RO-EN con back-translation)

| Modelo | BLEU |
|---|---:|
| Baseline Transformer-large | 36.80 |
| Fixed BART (paso 1, solo encoder bridge entrenado) | 36.29 |
| **Tuned BART** (paso 2, todo fine-tuned) | **37.96** |

Ganancia de +1.16 BLEU. Modesta pero significativa considerando que BART solo está pre-entrenado en inglés (no en rumano).

---

## 8. Análisis cualitativo (Sección 6 del paper)

La Tabla 7 muestra resúmenes generados por BART-XSum sobre artículos de WikiNews **publicados después del corpus de pre-training** (evitando data leakage). Los autores observan:

1. **Fluidez gramatical**: output fluido, sin artifacts de generación.
2. **Alta abstractividad**: pocas frases copiadas literalmente del input.
3. **Precisión factual general**: integra evidencia distribuida + conocimiento de mundo.
4. **Inferencias no triviales**: en un ejemplo sobre peces protegiendo arrecifes en Fiji, el modelo infiere correctamente la conexión causa-efecto.
5. **Conocimiento de mundo**: completa nombres (Boris Johnson como Primer Ministro), infiere que PG&E opera en California, integra contexto geográfico.
6. **Alucinación ocasional**: en el ejemplo de Fiji, BART afirma que el trabajo "fue publicado en Science" — **claim no soportado por el texto fuente**. Esta es la primera observación pública de **hallucination en summarization neural** que se vuelve crítica años después con LLMs.

---

## 9. Comparación detallada BART vs T5

Dado que ambos modelos son contemporáneos y conceptualmente paralelos, conviene una comparación lado a lado.

| Dimensión | BART (Lewis 2020) | T5 (Raffel 2020) |
|---|---|---|
| **Arquitectura** | Encoder-decoder Transformer estándar | Encoder-decoder con modificaciones (relative position bias) |
| **Tamaños** | base (140M), large (400M) | small, base, large, 3B, 11B |
| **Pre-training data** | 160GB (news+books+stories+web) | 750GB C4 (Common Crawl limpiado) |
| **Objetivo** | Text infilling (Poisson λ=3) + sentence permutation, ratio 30% | Span corruption con **sentinels** + multitarea supervisada |
| **Sentinels** | Un solo `[MASK]` opaco | `<extra_id_0>`, `<extra_id_1>`, ... diferenciados |
| **Tokenizer** | BPE de GPT-2 (~50K) | SentencePiece (32K) |
| **Fine-tuning** | Per-task heads y modalidad específica | Texto-a-texto unificado (prefix instruction) |
| **CNN/DM R1** | 44.16 | 43.52 (T5-base), 44.66 (T5-11B) |
| **XSum R1** | 45.14 | (no reportado directamente en T5 paper) |
| **GLUE avg** | ~88.4 | 89.7 (T5-large) |
| **Filosofía** | Denoising puro + per-task fine-tuning | Multi-task supervised + prompt prefix |

**¿Cuál usar?**

- **BART** si la prioridad es **summarization rápido** con un modelo pequeño (`facebook/bart-large-cnn` corre en GPU consumer).
- **T5** si se quiere **multi-task con prompts** o escalar a 11B parámetros.
- **HuggingFace `pipeline("summarization")`** usa **BART por defecto** (`facebook/bart-large-cnn`). Esto es resultado de un balance entre calidad y tamaño/velocidad.

Curiosamente, las performances son **muy similares en summarization a tamaños comparables**, lo cual sugiere que el espacio de "denoising seq2seq objectives" ha convergido hacia un óptimo similar.

---

## 10. Limitaciones reconocibles

Aunque el paper no incluye una sección explícita de limitaciones (era común omitirla en 2019), una lectura crítica revela:

1. **Costo computacional vs BERT**: BART requiere encoder + decoder. Para tareas puramente discriminativas, RoBERTa es más eficiente en cómputo a igual capacidad.

2. **Memoria de cross-attention**: cada capa del decoder calcula atención sobre todos los tokens del encoder, lo cual escala $O(L_{enc} \cdot L_{dec})$ en memoria — más caro que GPT puro.

3. **Bridge architecture no escala**: el esquema de translation requiere un encoder bridge por cada par de idiomas. mBART resuelve esto pero a costa de pre-entrenar multilingüe desde cero.

4. **No instruction-tuned**: BART en su forma original no entiende prompts del estilo "Resume esto:" — el fine-tuning per-task es necesario. FLAN-T5 y T0 luego mostrarían que el instruction tuning explícito mejora generalización.

5. **Hallucination en summarization**: los propios autores observan en Tabla 7 que BART inventa información (ej. "publicado en Science"). Este es un problema endémico de modelos generativos pre-entrenados sin grounding.

6. **Documento length**: positional embeddings learned, max 1024 tokens. Documentos largos requieren chunking o variantes como LongBART.

7. **El nombre "BART"** es un retro-acrónimo construido (Bidirectional and Auto-Regressive Transformers). Los autores admiten implícitamente que la motivación principal fue **una etiqueta memorable** que evocara a BERT y GPT.

---

## 11. Impacto e influencia posterior

### 11.1 Descendencia directa

- **mBART** (Liu et al., 2020): versión multilingüe entrenada en 25 idiomas (luego expandida a 50+). Usa el mismo objetivo de text infilling + sentence permutation pero sobre Common Crawl multilingüe. Es la base de muchos sistemas de traducción zero-shot.

- **PLBART** (Ahmad et al., 2021): aplicado a lenguajes de programación. Entrenado sobre código de GitHub. Usado en code summarization, code generation, bug fixing.

- **DistilBART**: variante destilada para deployment.

### 11.2 Modelos concurrentes/competidores

- **Pegasus** (Zhang et al., 2020, ICML): paper concurrente especializado en summarization. Propone **Gap Sentence Generation (GSG)**: enmascarar oraciones enteras "importantes" (medidas por ROUGE contra el resto del documento) y predecirlas. Pegasus supera a BART en XSum (R1=47.21 vs 45.14) pero requiere objetivo más especializado.

- **ProphetNet** (Qi et al., 2020): predice **n tokens futuros simultáneamente** en vez de uno. Mejora summarization marginalmente.

### 11.3 Influencia en instruction tuning

La idea de "denoising sequence-to-sequence" sobrevive en:

- **T0** (Sanh et al., 2021): T5 fine-tuned en multi-task con prompt naturales.
- **FLAN** (Wei et al., 2021): instruction tuning de T5 y PaLM.
- **InstructGPT** (Ouyang et al., 2022): instruction tuning + RLHF en GPT-3. Aunque conceptualmente decoder-only, hereda la idea de "aprende a transformar texto noisy/instructed en texto limpio/útil".

### 11.4 Adopción industrial

- **HuggingFace summarization pipeline**: el default es `sshleifer/distilbart-cnn-12-6` o `facebook/bart-large-cnn`. Esto significa que **decenas de miles de aplicaciones** que usan `pipeline("summarization")` están corriendo BART por defecto.

- **API comerciales**: muchas APIs de summarization en 2020-2022 (Cohere, AssemblyAI, etc.) usaron variantes de BART antes de migrar a LLMs decoder-only.

- **AWS Bedrock, Azure ML**: ofrecen BART como modelo gestionado.

### 11.5 Citaciones

A enero de 2026, el paper acumula **>15,000 citaciones según Google Scholar**, ubicándolo entre los papers más influyentes de pre-training NLP junto con BERT (>100K), GPT-2 (>10K), T5 (>20K) y RoBERTa (>30K).

---

## 12. Conexión con la Clase 22 (IA UC)

La Clase 22 del curso aborda **Text Generation** desde la óptica de **modelos seq2seq encoder-decoder**. La estructura de slides cubre:

- **Slide 28-35**: arquitectura encoder-decoder, problema de generación abstractiva.
- **Slide 36-41**: T5 como ejemplo canónico de pre-trained seq2seq.
- **Slide 42-50**: estrategias de decoding (greedy, beam search, top-k, top-p/nucleus, temperature).
- **Slide 51-60**: aplicaciones (summarization, translation, dialog).

**¿Por qué BART encaja aquí?**

1. **Es la alternativa práctica a T5 en HuggingFace**: si el lab usa `pipeline("summarization")`, está usando BART implícitamente. Comprender BART es comprender qué hay debajo del default.

2. **Misma familia conceptual que T5**: ambos son encoder-decoder denoising. La intuición transferida del paper de T5 (Raffel) aplica directamente a BART. Conocer las diferencias (sentinels vs `[MASK]` opaco, multi-task vs single-task pretraining) **enriquece la comprensión del espacio de diseño**.

3. **Decoding strategies aplican igual**: la sección de la clase sobre beam search, length penalty, trigram blocking, sampling temperature aplica a BART tal cual. De hecho, los hiperparámetros de generación de BART en el paper original (beam=5, trigram blocking, label smoothing 0.1) son **el setup canónico** que la clase enseña.

4. **Hallucination y limitaciones**: la observación del paper sobre alucinaciones en summarization es **un puente directo** hacia la discusión de la clase sobre confiabilidad, factualidad y el problema de generación libre.

5. **Conexión histórica**: BART completa el cuadro 2018-2019 (BERT → GPT-2 → XLNet → RoBERTa → T5 → BART) que culmina en la era de los foundation models. Sin BART, el árbol genealógico que llega a InstructGPT/FLAN/LLMs queda incompleto.

**Lectura sugerida en paralelo a la clase**:

- Sección 2 del paper (modelo) → mapea directamente al diagrama encoder-decoder de la slide 28.
- Sección 4 (ablation de ruidos) → permite discutir críticamente qué se pre-entrena cuando se hace pretraining.
- Sección 5 (resultados grandes) → contextualiza los benchmarks ROUGE/BLEU que la clase introduce.
- Sección 6 (análisis cualitativo) → motiva la discusión de hallucination y factualidad en gen.

---

## 13. Conclusión

BART logra algo conceptualmente elegante: **unifica el pre-training de comprensión y generación bajo un solo objetivo (denoising autoencoding) y un solo modelo (encoder-decoder Transformer)**, eligiendo experimentalmente la función de ruido óptima en vez de postular una a priori. El resultado es un modelo que:

- **Iguala a RoBERTa en comprensión** (GLUE, SQuAD).
- **Establece state-of-the-art en summarization** (CNN/DM, XSum) con +6 ROUGE en XSum.
- **Es competitivo en QA abstractiva, diálogo y traducción**.
- **Se convierte en el default de HuggingFace** para summarization pipeline, asegurando un legado industrial duradero.

Su contribución metodológica más importante no es el modelo en sí, sino **la ablation sistemática de funciones de ruido**, que demuestra empíricamente:

1. **Text infilling con `[MASK]` opaco y Poisson(λ=3)** es el ruido más versátil.
2. **Token deletion supera a masking** en generación.
3. **Sentence permutation ayuda** en summarization de documentos largos.
4. **Document rotation aislado es inútil** — la señal está en los tokens, no en el orden global.

En el contexto histórico, BART representa el **último gran paper de la era pre-LLM "discriminativo + generativo separados"**. Pocos meses después, GPT-3 (Brown et al., junio 2020) introduciría el paradigma de **scaling decoder-only + in-context learning** que dominaría el campo hasta hoy. Pero el legado de BART persiste: cada vez que alguien llama a `pipeline("summarization")` o usa mBART para traducción multilingüe, está ejecutando una descendiente directa de las ideas de Lewis, Liu, Goyal y colaboradores.

Para el curso IA UC, BART ofrece un ejemplo **canónico, reproducible y didácticamente generoso** de pre-training seq2seq. Junto con T5, son los dos pilares conceptuales sobre los que se apoya cualquier discusión seria de text generation en NLP moderno.

---

## Referencias clave

- **Paper original**: Lewis, Liu, Goyal et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL 2020. [arXiv:1910.13461](https://arxiv.org/abs/1910.13461).
- **T5 (concurrente)**: Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR. arXiv:1910.10683.
- **RoBERTa (corpus base)**: Liu et al. (2019). arXiv:1907.11692.
- **SpanBERT (inspiración del span infilling)**: Joshi et al. (2019). arXiv:1907.10529.
- **MASS (precursor seq2seq)**: Song et al. (2019). ICML.
- **mBART (descendiente multilingüe)**: Liu et al. (2020). arXiv:2001.08210.
- **Pegasus (competidor en summarization)**: Zhang et al. (2020). ICML. arXiv:1912.08777.
- **XSum (benchmark crítico)**: Narayan, Cohen, Lapata (2018). EMNLP. arXiv:1808.08745.
- **CNN/DailyMail**: Hermann et al. (2015). NeurIPS.
- **BERTSUMABS / BERTSUMEXTABS (baseline summarization)**: Liu & Lapata (2019). arXiv:1908.08345.
- **HuggingFace model card**: `facebook/bart-large-cnn` — default del summarization pipeline.

