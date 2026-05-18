---
title: "GPT-1 (Improving Language Understanding by Generative Pre-Training)"
weight: 295
math: true
---

{{< paper-card
    title="Improving Language Understanding by Generative Pre-Training"
    authors="Radford, Narasimhan, Salimans, Sutskever"
    year="2018"
    venue="OpenAI Technical Report"
    pdf="/papers/gpt-1-radford-2018.pdf"
    arxiv="" >}}
GPT-1 introduce el paradigma **generative pre-training + discriminative fine-tuning** con un Transformer **decoder-only** de 117M parámetros. Primero se entrena un modelo de lenguaje autoregresivo sobre BookCorpus (~800M palabras) y luego se fine-tunea end-to-end por tarea, reformulando cada input como una secuencia única de tokens con delimitadores especiales. Alcanza estado del arte en 9 de 12 datasets de NLU (SNLI, MNLI, SciTail, QNLI, Story Cloze, RACE, CoLA, STS-B, QQP) y eleva el promedio GLUE de 68.9 a 72.8. Establece la línea genealógica que pasa por GPT-2, GPT-3 y ChatGPT.
{{< /paper-card >}}

---

## Contexto

A mediados de 2018 el "transfer learning" en NLP todavía era casi sinónimo de **embeddings de palabras pre-entrenados** (Word2Vec, GloVe, fastText) inyectados en arquitecturas supervisadas entrenadas desde cero por tarea. Cada problema tenía su modelo: ESIM para NLI, BiDAF para QA extractivo, Tree-LSTM para sentiment. Anotar datasets supervisados era caro y los corpora etiquetados rara vez superaban unos cientos de miles de ejemplos.

La pregunta abierta era si podía transferirse **más que palabras** -- sintaxis, semántica composicional, coreferencia, sentido común -- desde texto no anotado. Varios trabajos atacaron esa pregunta en paralelo:

- **CoVe** (McCann et al., 2017): encoder LSTM bidireccional entrenado en traducción inglés-alemán, usado como features. Limitado por la escasez de pares paralelos.
- **Semi-supervised Sequence Learning** (Dai & Le, 2015): pre-entrena un LSTM con LM y fine-tunea. Antecedente directo de GPT-1 en espíritu pero con LSTM (long-range pobre) y resultados modestos.
- **ULMFiT** (Howard & Ruder, ACL 2018): AWD-LSTM pre-entrenado en WikiText-103, learning rates discriminativos y slanted triangular schedule. Mostró transferencia efectiva en clasificación de texto.
- **ELMo** (Peters et al., NAACL 2018): BiLM de dos LSTMs (forward + backward) entrenado en 1B Word Benchmark; representaciones contextuales **como features**, sin fine-tunear el LM.

ELMo y GPT-1 son contemporáneos conceptualmente rivales:

| Eje | ELMo (feb 2018) | GPT-1 (jun 2018) |
|---|---|---|
| Arquitectura | BiLSTM 2 capas | Transformer decoder 12 capas |
| Direccionalidad | Bidireccional (concat fwd+bwd) | Unidireccional (causal) |
| Uso downstream | Features fijos + cabeza task-specific | Fine-tuning end-to-end del LM |
| Long-range | Limitado por recurrencia | Self-attention $O(n^2)$ directa |
| Tamaño | ~94M | 117M |

El [Transformer](/fundamentos/transformer) (Vaswani et al., NeurIPS 2017) ya tenía ocho meses. Su impacto inicial fue en traducción seq2seq y todavía no era claro si era una arquitectura de propósito general. Liu et al. (ICLR 2018, "Generating Wikipedia by summarizing long sequences") habían introducido un decoder-only Transformer para resumen abstractivo y GPT-1 cita ese trabajo como inspiración. OpenAI apostó por tres decisiones que retrospectivamente parecen obvias:

1. **Transformer en vez de LSTM**: long-range dependencies sin cuello de botella secuencial.
2. **Decoder-only causal en vez de encoder-decoder o bidireccional**: porque LM autoregresivo es naturalmente decoder; evita combinar dos modelos como ELMo.
3. **Fine-tuning end-to-end** del modelo entero, no extracción de features.

El abstract es explícito sobre las dos preguntas no resueltas en 2018:

> "First, it is unclear what type of optimization objectives are most effective at learning text representations that are useful for transfer. (...) Second, there is no consensus on the most effective way to transfer these learned representations to the target task."

GPT-1 apuesta por **language modeling puro** (objetivo) y **fine-tuning end-to-end con input transformations** (transferencia). Esa combinación es la base sobre la que se construirá toda la familia decoder-only que dominará desde 2020.

---

## Ideas principales

### 1. Decoder-only Transformer

Stack de 12 bloques decoder Transformer con **masked self-attention causal**:

- $L=12$ capas, $d_{\text{model}}=768$, $A=12$ heads ($d_k=d_v=64$).
- FFN interno 3072 (= $4 \times 768$) con activación **GELU** (Hendrycks & Gimpel, 2016).
- **Position embeddings aprendidos** (no sinusoidales como en Vaswani 2017).
- Ventana de contexto **512 tokens**.
- Tokenización **BPE** con 40k merges sobre texto pre-procesado con `ftfy` + spaCy.
- Dropout 0.1 (residual, embeddings, attention), inicialización $\mathcal{N}(0, 0.02)$.
- **~117M parámetros** -- casi idéntico en tamaño a [BERT-base](/papers/bert-devlin-2018) (110M), lo que hace que las comparaciones sean particularmente limpias.

La arquitectura cabe en tres ecuaciones:

$$
\begin{aligned}
h_0 &= U W_e + W_p \\
h_l &= \text{transformer\_block}(h_{l-1}), \quad l=1,\dots,12 \\
P(u) &= \text{softmax}(h_n W_e^\top)
\end{aligned}
$$

con $W_e$ compartido entre input y output (**weight tying**) y $W_p$ aprendido. La máscara causal se aplica antes del softmax:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V, \quad M_{ij}=\begin{cases}0 & j\le i \\ -\infty & j > i\end{cases}
$$

La elección decoder-only frente a encoder-decoder se justifica internamente: el objetivo de pre-training es LM autoregresivo, no hay fuente distinta de la salida, y las tareas downstream se reformulan como secuencias únicas. [BERT](/papers/bert-devlin-2018), cuatro meses después, elegirá lo opuesto -- encoder-only con MLM -- y ganará momentáneamente en understanding benchmarks. La apuesta de OpenAI se validará cuando GPT-2/3 muestren que la generación condicional unifica todas las tareas.

### 2. Pre-training autoregresivo en BookCorpus

Dado un corpus $\mathcal{U} = \{u_1, \dots, u_n\}$, se maximiza:

$$
L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)
$$

Único objetivo: no hay multi-task, no hay objetivos auxiliares, no hay masked LM. Solo predicción next-token con máscara causal.

**Corpus**: **BookCorpus** (Zhu et al., 2015), ~7,000 libros únicos no publicados (~800M palabras, ~5GB). El paper enfatiza la elección frente al **1B Word Benchmark** que usaba ELMo: BookCorpus contiene "long stretches of contiguous text", lo que permite aprender dependencias largas (anáfora, coherencia de párrafo, arcos narrativos). El modelo alcanza perplejidad de 18.4 a nivel de token.

**Hiperparámetros**: Adam, lr máx $2.5\times10^{-4}$, warmup lineal 2000 pasos + cosine annealing, batch 64 secuencias de 512 tokens, **100 épocas**, weight decay 0.01 estilo AdamW (Loshchilov & Hutter, 2017). Eso da $\sim$2.4M updates de gradiente -- comparable a BERT-base pero mucho menos que GPT-2 (40GB WebText) o GPT-3 (~570GB filtrados). La receta **Adam + warmup + cosine** se convertirá en estándar para entrenar Transformers profundos.

### 3. Fine-tuning con input transformations

Para una tarea supervisada con secuencia $x^1,\dots,x^m$ y etiqueta $y$, se toma la activación del **último token** en la última capa ($h_l^m$) y se proyecta linealmente:

$$
P(y \mid x^1, \dots, x^m) = \text{softmax}(h_l^m W_y)
$$

Los **únicos parámetros nuevos** son $W_y$ y los embeddings de los tokens delimitadores especiales. Todo el resto se inicializa desde el pre-trained y se sigue ajustando con gradientes.

El ingrediente práctico es convertir cualquier tarea estructurada en **una secuencia única** vía tokens delimitadores `<s>` (start), `<e>` (extract, el último token cuya activación se usa) y `$` (delimiter). El paper describe cuatro patrones (Figura 1):

1. **Classification** (SST-2, CoLA): `<s> texto <e>` → linear sobre $h_l^m$.
2. **Textual entailment** (SNLI, MNLI, QNLI, RTE, SciTail): `<s> premisa $ hipótesis <e>` → linear sobre 3 clases.
3. **Similarity** (MRPC, QQP, STS-B): se procesan dos órdenes (`<s> t_1 $ t_2 <e>` y `<s> t_2 $ t_1 <e>`) y las representaciones finales se **suman elemento a elemento** antes del linear, porque la similitud es simétrica.
4. **Multiple choice** (RACE, Story Cloze): para cada opción $a_k$ se construye `<s> contexto $ pregunta $ a_k <e>`, se obtiene un escalar $s_k$, y se aplica softmax sobre las $N$ opciones.

Esta estrategia, que en el paper se llama **traversal-style approach** (Rocktäschel et al., 2015), **prefigura el formato de prompt** que dominará la era GPT-3+. La diferencia es que GPT-1 ajusta los parámetros de los delimitadores vía fine-tuning, mientras que GPT-3 lo hace zero/few-shot vía in-context learning.

### 4. Auxiliary LM loss durante fine-tuning

GPT-1 introduce un truco: combinar el objetivo supervisado con el LM como regularizador:

$$
L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C}), \qquad \lambda = 0.5
$$

donde $L_2$ es el cross-entropy de clasificación y $L_1$ es el LM aplicado a la misma secuencia. La justificación es doble:

1. **Mejor generalización**: el LM auxiliar evita que el fine-tuning destruya las representaciones lingüísticas aprendidas en pre-training.
2. **Convergencia acelerada**: bastan **3 épocas** para fine-tunear casi todas las tareas, con lr $6.25\times 10^{-5}$, batch 32, warmup 0.2% y dropout 0.1 en la cabeza.

Esta idea será descartada en GPT-2/GPT-3 (que trabajan sin gradientes en el fine-tuning) y reemplazada por el "prefix LM" implícito del in-context learning.

---

## Resultados experimentales

### Natural Language Inference (Tabla 2)

| Método | MNLI-m | MNLI-mm | SNLI | SciTail | QNLI | RTE |
|---|---|---|---|---|---|---|
| ESIM + ELMo (5x ensemble) | -- | -- | 89.3 | -- | -- | -- |
| CAFE (5x ensemble) | 80.2 | 79.0 | 89.3 | -- | -- | -- |
| Multi-task BiLSTM + Attn | 72.2 | 72.1 | -- | -- | 82.1 | **61.7** |
| **GPT-1 (single)** | **82.1** | **81.4** | **89.9** | **88.3** | **88.1** | 56.0 |

Mejoras absolutas: +1.5 MNLI-m, +5 SciTail, +5.8 QNLI. RTE (2,490 ejemplos) es la única excepción: con tan pocos datos, multi-task BiLSTM con attention todavía gana. El paper conjetura que multi-task fine-tuning ayudaría.

### Question Answering y commonsense (Tabla 3)

| Método | Story Cloze | RACE-m | RACE-h | RACE |
|---|---|---|---|---|
| BiAttention MRU (9x) | -- | 60.2 | 50.3 | 53.3 |
| **GPT-1** | **86.5** | **62.9** | **57.4** | **59.0** |

**+8.9 absoluto en Story Cloze** y **+5.7 en RACE**. Son los gains más espectaculares: tareas que requieren razonar sobre múltiples oraciones y contextos largos son donde la capacidad long-range del Transformer brilla más.

### Similarity, Classification, GLUE (Tabla 4)

| Método | CoLA (mc) | SST-2 | MRPC (F1) | STS-B (pc) | QQP (F1) | GLUE |
|---|---|---|---|---|---|---|
| Multi-task BiLSTM + ELMo + Attn | 18.9 | 91.6 | 83.5 | 72.8 | 63.3 | 68.9 |
| **GPT-1** | **45.4** | 91.3 | 82.3 | **82.0** | **70.3** | **72.8** |

**CoLA pasa de 35.0 a 45.4** (correlación Matthews). CoLA mide aceptabilidad gramatical: que GPT-1 casi duplique la métrica sugiere que el pre-training capturó conocimiento sintáctico implícito que ningún modelo supervisado puro había extraído. Promedio GLUE: 72.8 vs 68.9 previo.

**Resumen global**: SOTA en **9 de 12 datasets**, funciona tanto en datasets pequeños (STS-B, ~5.7k) como grandes (SNLI, ~550k).

### Análisis 1: capas transferidas

Fine-tuning usando solo las primeras $k$ capas pre-entrenadas (las $k+1,\dots,12$ se reinicializan) muestra mejora **monotónica** en MultiNLI y RACE: con $k=0$ (solo embeddings) ya hay ganancia sobre random; transferir las 12 capas vs solo embeddings da hasta **+9 absoluto** en MultiNLI. **Cada capa aporta funcionalidad útil**, no hay una capa privilegiada. Este resultado se replicará después en los probing classifiers de Tenney et al. (2019) sobre BERT.

### Análisis 2: zero-shot behaviors emergentes

Este es probablemente **el experimento más profético del paper**. Los autores diseñan heurísticas zero-shot que usan el LM pre-entrenado **sin ningún fine-tuning** y miden su accuracy a lo largo del pre-training:

| Tarea | Heurística zero-shot |
|---|---|
| Sentiment (SST-2) | Agregar "very" → comparar $P(\text{positive})$ vs $P(\text{negative})$ |
| Linguistic acceptability (CoLA) | Average log-prob por token → threshold |
| Multiple choice QA (RACE) | Average log-prob condicional por respuesta → argmax |
| Winograd Schema | Reemplazar pronombre por cada referente, scorear, argmax |

A medida que avanza el pre-training, **la accuracy de estas heurísticas sube monotónicamente** en todas las tareas. Un LSTM equivalente muestra mucha más varianza, sugiriendo que **el inductive bias del Transformer ayuda específicamente a la transferencia zero-shot**.

Implicación profunda: el LM, simplemente al optimizar predicción next-token sobre texto natural, está aprendiendo a hacer sentiment, parsing, sentido común. **Esto es exactamente la tesis de GPT-3** ("Language Models are Few-Shot Learners", 2020). GPT-1 ya tenía la observación en 2018, pero el modelo era demasiado pequeño para que las accuracies zero-shot fueran competitivas con SOTA, así que el paper la presenta como "análisis" y no como contribución principal.

### Ablations (Tabla 5)

| Variante | Avg Score | Delta |
|---|---|---|
| Transformer + pre-training + aux LM (full) | 74.7 | -- |
| Transformer + pre-training (sin aux LM) | 75.0 | +0.3 |
| LSTM + pre-training + aux LM | 69.1 | **-5.6** |
| Transformer sin pre-training | 59.9 | **-14.8** |

Es lo más cercano a una ecuación de Lavoisier del transfer learning en NLP de 2018: **pre-training >> arquitectura >> auxiliar LM**. El aux LM es marginal en promedio pero ayuda en datasets grandes; el Transformer es crítico (LSTM con la misma receta pierde 5.6 puntos); pero el pre-training mismo es el ingrediente dominante (~15 puntos).

---

## Limitaciones reconocibles

**Reconocidas en el paper**:
- RTE underperforms (56% vs 61.7%) -- multi-task fine-tuning probablemente ayudaría, pero no lo exploran.
- Solo inglés, solo tareas NLU clásicas.

**Estructurales en retrospectiva**:

1. **Unidireccionalidad**: solo ve contexto izquierdo. Para NLI o QA donde el contexto bidireccional ayuda, esto es subóptimo. [BERT](/papers/bert-devlin-2018) (4 meses después) explotará esta debilidad con MLM y dominará benchmarks NLU. La comunidad concluirá apresuradamente que "decoder-only es inferior" -- conclusión que GPT-3 desmentirá en 2020.
2. **Necesita fine-tuning por tarea**: cada tarea requiere su propio dataset etiquetado y su propio modelo final. No escala a "miles de tareas". GPT-2 (2019) atacará esto con zero-shot competitivo; GPT-3 (2020) con few-shot in-context learning.
3. **Input transformations ad-hoc**: los delimitadores `<s>`, `<e>`, `$` son una solución de ingeniería. GPT-2/3 usarán prompts en lenguaje natural, más flexibles y sin tokens especiales.
4. **Tamaño y datos limitados**: 117M parámetros, 800M palabras. GPT-3 es **1500x más grande** (175B) y entrenado en ~750x más texto.
5. **BookCorpus sesgado a ficción narrativa**: nada de código, papers, diálogos o instrucciones. Limita generalización.
6. **No hay alignment**: el LM aprende lo que está en BookCorpus, sin esfuerzo de hacerlo útil/seguro para humanos. InstructGPT (2022) atacará esto con RLHF.
7. **Auxiliar LM con $\lambda=0.5$** es un kludge con balancing fino; BERT, RoBERTa y T5 lo descartarán.
8. **BPE tradicional**: no maneja bien lenguajes no-latinos, código, emojis. GPT-2 introducirá **byte-level BPE** sobre bytes UTF-8.

---

## Por qué importa hoy

GPT-1 es **el momento fundacional del paradigma decoder-only** que culmina en ChatGPT. Su publicación informal -- technical report en el blog de OpenAI, sin venue formal -- limitó su impacto académico inicial frente a BERT (NAACL 2019), que se llevó la atención durante 2019. Pero la línea genealógica es directa y vertebra la **Clase 20** del curso:

| Hito | Año | Salto sobre el anterior |
|---|---|---|
| **GPT-1** (Radford et al.) | 2018 | Decoder-only + pre-training + fine-tuning |
| BERT (Devlin et al.) | 2018 | Encoder bidireccional + MLM. Domina NLU |
| GPT-2 (Radford et al.) | 2019 | 1.5B params, WebText, zero-shot competitivo |
| GPT-3 (Brown et al.) | 2020 | 175B params, few-shot in-context learning |
| InstructGPT (Ouyang et al.) | 2022 | RLHF, alignment |
| ChatGPT | 2022 | Chat interface, deployment masivo |
| GPT-4 | 2023 | Multimodal, capacidades emergentes |

Comparación directa GPT-1 vs BERT-base (mismo año, mismo tamaño):

| Aspecto | GPT-1 (jun 2018) | BERT-base (oct 2018) |
|---|---|---|
| Arquitectura | Decoder (causal mask) | Encoder (bidireccional) |
| Capas / Dim / Heads | 12 / 768 / 12 | 12 / 768 / 12 |
| Parámetros | 117M | 110M |
| Objetivo pre-training | Forward LM (next-token) | MLM + NSP |
| Datos | BookCorpus (800M) | BookCorpus + Wikipedia (3.3B) |
| GLUE | 72.8 | 78.3 |
| Direccionalidad | Unidireccional | Bidireccional |
| Tokenización | BPE 40k | WordPiece 30k |

Con arquitecturas casi idénticas, la decisión de **direccionalidad y objetivo de pre-training** define el comportamiento downstream. BERT gana en understanding; GPT-1 sienta las bases para la generación. La narrativa del curso es que esta aparente derrota se revierte en 2020+ cuando la escala revela que **decoder-only generative pre-training escala mejor** que encoder MLM -- porque un decoder es naturalmente generativo y la generación subsume clasificación, QA, traducción y resumen sin modificaciones arquitectónicas.

Cuatro principios de GPT-1 siguen vigentes en 2026:

1. **Generative pre-training como ruta hacia generalismo**: la idea de que un modelo entrenado para predecir tokens aprende muchas habilidades emergentes ha sido validada en cada escalamiento posterior.
2. **Decoder-only + causal LM**: arquitectura dominante para LLMs grandes. Encoder-only sobrevive en búsqueda/embedding, pero los LLMs generativos son todos decoder-only.
3. **[Transfer learning](/fundamentos/transfer-learning) end-to-end**: el modelo entero se sigue ajustando, no se congelan capas. Se mantiene incluso en LoRA y otros métodos PEFT.
4. **Reformular tareas como secuencias**: input transformations son el ancestro directo del prompting moderno y el chat templating.

Y la **Figura 2 derecha** -- la aparición de comportamientos zero-shot a lo largo del pre-training -- es una de las primeras evidencias empíricas de que **las capacidades emergen del pre-training puro**. Esta hipótesis es la base de la "scaling hypothesis" que justifica el gasto computacional de la era post-2020.

Lo que el paper **no** anticipó: in-context learning con ejemplos few-shot en el prompt, chain-of-thought, RLHF y alignment, multimodalidad (GPT-4V, Gemini), y la persistente debilidad de la familia en razonamiento simbólico estable (matemática, lógica formal, planning).

---

## Notas y enlaces

- El paper tiene 12 páginas y es legible en una sentada: Sección 1-2 (introducción y framework), Sección 3 (arquitectura y experimentos), Sección 5 (análisis -- la parte más profética).
- **Figura 1**: las cuatro input transformations lado a lado (classification, entailment, similarity, multiple choice).
- **Figura 2**: izquierda = effect of number of layers transferred; derecha = zero-shot behaviors emergentes a lo largo del pre-training.
- **Tabla 5**: ablation studies -- la "ecuación de Lavoisier" del transfer learning de 2018.
- Código original de OpenAI: [github.com/openai/finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm) (TensorFlow 1.x, bastante legible).
- Implementaciones modernas: HuggingFace `OpenAIGPTModel` y `OpenAIGPTLMHeadModel`.

Ver fundamentos: [GPT family](/fundamentos/gpt-family) - [Transformer](/fundamentos/transformer) - [Transfer Learning](/fundamentos/transfer-learning) - [Pre-training y BERT](/fundamentos/pretraining-bert) - [BPE](/fundamentos/bpe) - [Clase 20](/clases/clase-20). Papers relacionados: [Attention is All You Need](/papers/attention-is-all-you-need-vaswani-2017) - [BERT](/papers/bert-devlin-2018).
