---
title: "BART (Denoising Sequence-to-Sequence Pre-training)"
weight: 114
math: true
---

{{< paper-card
    title="BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
    authors="Lewis, Liu, Goyal, Ghazvininejad, Mohamed, Levy, Stoyanov, Zettlemoyer"
    year="2020"
    venue="ACL 2020"
    pdf="/papers/bart-lewis-2020.pdf"
    arxiv="1910.13461" >}}
Denoising autoencoder encoder-decoder Transformer. Pre-entrena con **text infilling** (spans Poisson($\lambda=3$) reemplazados por un solo `[MASK]` opaco) + **sentence permutation**. Iguala a RoBERTa en GLUE/SQuAD y establece state-of-the-art en summarization abstractiva (CNN/DM ROUGE-1 = 44.16, XSum ROUGE-1 = 45.14, +6 ROUGE sobre BERTSUMEXTABS). Es el modelo por defecto del `pipeline("summarization")` de HuggingFace via `facebook/bart-large-cnn`.
{{< /paper-card >}}

---

## El problema

Para mediados de 2019 el NLP self-supervised tenia ya un zoologico de objetivos: MLM de BERT, LM causal de GPT, permutation LM de XLNet, span seq2seq de MASS, MLM-mejorado de RoBERTa, span corruption de T5. Cada uno era bueno en lo suyo -- BERT en clasificacion, GPT en generacion, XLNet en ambos pero arquitectonicamente caro -- y nadie habia comparado los **objetivos de ruido en si** bajo arquitectura y datos identicos.

La pregunta abierta de los autores: **que funcion de ruido $g(\cdot)$ es optima para downstream generation?** BART aporta dos cosas: (i) una arquitectura flexible donde **cualquier** transformacion del input es admisible, incluso las que cambian la longitud; y (ii) una ablation sistematica de cinco ruidos que **deja a los datos elegir el ganador**. El resultado -- text infilling -- emerge como el ruido mas versatil para una gama amplia de tareas.

BART es contemporaneo exacto de T5 (ambos en arXiv en octubre 2019, con 18 dias de diferencia). Ambos apuestan por encoder-decoder, pero llegan por caminos distintos: T5 desde "todo es texto-a-texto + multitarea masiva supervisada"; BART desde "denoising puro y per-task fine-tuning".

---

## Arquitectura

BART es un **encoder-decoder Transformer estandar** sin modificaciones estructurales mayores:

- **Encoder**: Transformer bidireccional, igual a BERT. Atiende a todos los tokens del input corrupto $g(x)$.
- **Decoder**: Transformer autoregresivo con dos atenciones por capa -- self-attention causal (igual a GPT) y **cross-attention** a la ultima hidden layer del encoder. Genera token a token de izquierda a derecha.

Diferencias menores respecto a BERT (heredadas de GPT/RoBERTa): activacion **GeLU**, inicializacion $\mathcal{N}(0, 0.02)$, sin FFN final antes de la word prediction, sin Next Sentence Prediction.

**Tamanos**:

| Variante | Encoder | Decoder | Hidden | Params |
|---|---|---|---|---|
| BART-base | 6 layers | 6 layers | 768 | ~140M |
| BART-large | 12 layers | 12 layers | 1024 | **~400M** |

BART-large tiene aproximadamente 10% mas parametros que BERT-large por el cross-attention adicional.

**Objetivo de pre-entrenamiento**: negative log-likelihood del documento original $x$ dado el documento corrupto $g(x)$:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{|x|} \log P_\theta(x_t \mid x_{<t}, g(x))
$$

En el limite donde $g$ destruye todo, BART degenera a un language model puro. En el limite donde $g$ es identidad, a un autoencoder trivial. La gracia esta en el medio.

**Distincion clave con BERT**: en BERT, un `[MASK]` predice un token en esa **misma posicion**. En BART, el decoder puede expandir un solo `[MASK]` a cero, uno o varios tokens, porque la longitud del output es libre. Esto habilita ruidos que cambian longitud.

---

## Funciones de ruido evaluadas

El paper estudia cinco transformaciones componibles:

### Token Masking (estilo BERT)

Tokens individuales se reemplazan por `[MASK]`:

```
A B C D E . → A [MASK] C [MASK] E .
```

### Token Deletion

Tokens se **borran**, no se enmascaran. El modelo no recibe placeholder -- debe descubrir **que falta y donde**:

```
A B C D E . → A C E .
```

Empiricamente supera a masking en generacion porque obliga a razonar sobre la posicion de las ausencias.

### Text Infilling (la contribucion mas original)

Spans de longitud variable se reemplazan por **un solo** `[MASK]`. Las longitudes se muestrean de una Poisson con $\lambda = 3$:

$$
L \sim \text{Poisson}(3), \qquad P(L=k) = \frac{3^k e^{-3}}{k!}
$$

Los spans de longitud 0 son **inserciones**: un `[MASK]` que el modelo debe eliminar:

```
A B C D E . → A [MASK] D [MASK] E .
```

Distincion sutil pero importante:

- **SpanBERT**: span enmascarado por una secuencia de `[MASK]` de la **misma longitud** que el span. El modelo conoce cuantos tokens predecir.
- **T5**: spans reemplazados por **sentinels distinguibles** (`<extra_id_0>`, `<extra_id_1>`, ...).
- **BART**: **un solo `[MASK]` opaco**. El modelo debe **inferir cuantos tokens van** en cada hueco. Esto obliga a modelar la distribucion sobre longitudes -- justo lo que importa en summarization, MT y respuesta libre.

### Sentence Permutation

Las oraciones del documento se mezclan aleatoriamente. El modelo restaura el orden discursivo original. Incentiva captura de coherencia inter-oracional.

### Document Rotation

El documento se rota circularmente para empezar en un token elegido al azar. El modelo aprende a identificar el inicio.

Las cinco transformaciones son **composables**, y el paper explora combinaciones. La receta final del modelo combina text infilling + sentence permutation.

---

## Ablation: que ruido gana?

Bajo arquitectura, datos y procedimiento de fine-tuning identicos, se comparan baselines re-implementados (LM, Permuted LM, MLM, Multitask MLM, Masked Seq2Seq) contra cinco variantes de BART. Todos en modelo base, 1M steps, sobre books+Wikipedia.

| Objetivo | SQuAD F1 | MNLI Acc | XSum PPL | CNN/DM PPL |
|---|---:|---:|---:|---:|
| Masked LM | 90.0 | 83.5 | 7.87 | 7.06 |
| Language Model | 76.7 | 80.1 | 7.00 | 6.56 |
| Permuted LM | 89.1 | 83.7 | 7.69 | 6.96 |
| Masked Seq2Seq | 87.0 | 82.1 | 6.80 | 6.19 |
| **BART** w/ Token Masking | 90.4 | 84.1 | 7.08 | 6.10 |
| **BART** w/ Token Deletion | 90.4 | 84.1 | 6.90 | 5.87 |
| **BART** w/ Text Infilling | **90.8** | 84.0 | **6.61** | 5.83 |
| **BART** w/ Document Rotation | 77.2 | 75.3 | 17.14 | 10.59 |
| **BART** w/ Sentence Shuffling | 85.4 | 81.5 | 10.93 | 7.89 |
| **BART** w/ Infilling + Shuffling | 90.8 | 83.8 | 6.62 | **5.41** |

(Menor PPL es mejor. Mayor F1/Acc es mejor.)

Hallazgos:

1. **Text infilling es el ganador robusto**: unico objetivo simultaneamente bueno en clasificacion y generacion.
2. **Token deletion > token masking** en generacion: razonar sobre la posicion ausente es senal mas dificil.
3. **Document rotation y sentence shuffling aislados son inutiles**: la senal de aprendizaje esta en reconstruir tokens, no en posiciones globales.
4. **Encoders bidireccionales son cruciales para SQuAD**: LM puro cae a 76.7 F1 vs 90.4 de BART.
5. **Pre-training left-to-right ayuda a generacion**: MLM puro queda atras en XSum/CNN-DM porque no entrena el decoder autoregresivo.

El objetivo final de BART-large es **Text Infilling (30% tokens, Poisson $\lambda=3$) + Sentence Permutation (todas las oraciones)**.

---

## Fine-tuning

BART admite cuatro modalidades segun la tarea:

- **Sequence classification** (GLUE, MNLI, RTE): el mismo input se replica a encoder y decoder, y se anade un token adicional al final del decoder; su hidden state final alimenta un clasificador lineal. Difiere de BERT (que usa `[CLS]` al inicio) porque BART quiere que ese token atienda via self-attention causal a **toda** la secuencia procesada.
- **Token classification** (SQuAD endpoints): documento completo a encoder y decoder; se usa el top hidden state del decoder por token.
- **Sequence generation** (summarization, dialog, QA abstractivo): modalidad natural -- encoder recibe source, decoder genera target. Sin modificaciones arquitectonicas. Hiperparametros tipicos: label smoothing $\epsilon=0.1$, beam search con beam = 5, trigram blocking, length penalty tuneado.
- **Machine translation (bridge architecture)**: se reemplaza la capa de embeddings del encoder por un nuevo encoder pequeno (6 layers) que traduce idioma fuente $\to$ "ingles ruidoso" que BART de-noisifica. Entrenamiento en dos fases (congelado y descongelado). +1.16 BLEU en WMT16 RO-EN. Honestidad: el esquema no escala a multiples idiomas (cada par requiere su propio bridge); mBART lo abandona luego.

---

## Pre-training corpus

BART-large se entrena sobre el **corpus de RoBERTa**: 160GB combinando CC-News, BookCorpus, Stories y OpenWebText. Batch 8000, 500k steps, BPE de GPT-2 (~50K vocab). Masking ratio: **30%** (mayor que el 15% de BERT porque cada span "consume" mas tokens). Dropout deshabilitado en el ultimo 10% de los pasos.

---

## Resultados a gran escala

### Tareas discriminativas (GLUE + SQuAD)

| Modelo | SQuAD 1.1 F1 | SQuAD 2.0 F1 | MNLI m | SST | QQP | QNLI | RTE |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT-large | 90.9 | 81.8 | 86.6 | 93.2 | 91.3 | 92.3 | 70.4 |
| XLNet | 94.5 | 88.8 | **89.8** | 95.6 | 91.8 | 93.9 | 83.8 |
| RoBERTa | **94.6** | **89.4** | **90.2** | 96.4 | 92.2 | 94.7 | 86.6 |
| **BART** | **94.6** | 89.2 | 89.9 | **96.6** | **92.5** | **94.9** | **87.0** |

BART iguala o supera a RoBERTa y XLNet en casi todas las tareas discriminativas, validando que la arquitectura encoder-decoder **no penaliza comprension**.

### Summarization

| Modelo | CNN/DM R1 | R2 | RL | XSum R1 | R2 | RL |
|---|---:|---:|---:|---:|---:|---:|
| Lead-3 baseline | 40.42 | 17.62 | 36.67 | 16.30 | 1.60 | 11.95 |
| BERTSUMEXTABS | 42.13 | 19.60 | 39.18 | 38.81 | 16.50 | 31.27 |
| **BART** | **44.16** | **21.28** | **40.90** | **45.14** | **22.27** | **37.25** |

En XSum la ganancia es de **+6.3 ROUGE-1** sobre BERTSUMEXTABS. XSum es altamente abstractivo (resumenes de una oracion, paginas que rara vez se solapan con el fuente) -- justo lo que el denoising entrena: parafrasis genuina, compresion semantica, conocimiento de mundo.

### Otros

- **ConvAI2** (dialog): F1 = 20.72, PPL = 11.85 (best previo: 19.09 / 17.51).
- **ELI5** (long-form QA): ROUGE-L = 24.3 (+1.2 sobre Seq2Seq multitask).
- **WMT16 RO-EN**: 37.96 BLEU (+1.16 sobre baseline Transformer-large).

---

## Analisis cualitativo

Sobre articulos de WikiNews **posteriores al corpus de pre-training** (sin data leakage), los autores observan:

- **Fluidez gramatical** y output sin artifacts.
- **Alta abstractividad**: pocas frases copiadas literalmente.
- **Inferencias no triviales** e integracion de conocimiento de mundo (ej. completar nombres, inferir contextos geograficos).
- **Hallucination ocasional**: BART inventa que un trabajo "fue publicado en Science" sin soporte en el texto fuente. Esta es **una de las primeras observaciones publicas de alucinacion en summarization neural** -- un problema endemico que se volveria critico anos despues con LLMs.

---

## Limitaciones

- **Costo encoder + decoder**: para tareas puramente discriminativas, RoBERTa es mas eficiente en computo a igual capacidad.
- **Memoria de cross-attention**: $O(L_{enc} \cdot L_{dec})$ por capa del decoder -- mas caro que GPT puro.
- **Bridge architecture no escala**: el esquema de MT requiere un encoder bridge por par de idiomas; mBART lo resuelve pero pre-entrenando multilingue desde cero.
- **No instruction-tuned**: BART original no entiende prompts como "Resume esto:". Necesita fine-tuning per-task. FLAN-T5 y T0 demostraran luego que el instruction tuning explicito mejora generalizacion.
- **Hallucination**: los propios autores documentan inventos factuales en summarization.
- **Maximo 1024 tokens**: positional embeddings learned; documentos largos requieren chunking o variantes como LongBART.

---

## BART vs T5

| Dimension | BART | T5 |
|---|---|---|
| **Arquitectura** | Encoder-decoder estandar | Encoder-decoder + relative position bias |
| **Tamanos** | base (140M), large (400M) | small, base, large, 3B, 11B |
| **Datos pre-training** | 160GB (corpus RoBERTa) | 750GB (C4) |
| **Objetivo** | Text infilling Poisson($\lambda=3$) + sentence permutation, 30% ratio | Span corruption con **sentinels** + multitarea supervisada |
| **Placeholder de span** | Un solo `[MASK]` opaco | Sentinels distinguibles (`<extra_id_i>`) |
| **Tokenizer** | BPE GPT-2 (~50K) | SentencePiece (32K) |
| **Fine-tuning** | Per-task heads | Texto-a-texto unificado (prefix prompt) |
| **CNN/DM R1** | 44.16 | 43.52 (base) / 44.66 (11B) |
| **Default HF summarization** | **Si** (`facebook/bart-large-cnn`) | No |
| **Filosofia** | Denoising puro, per-task | Multi-task supervisado + prompt |

A tamanos comparables, las performances en summarization son muy similares -- el espacio de "denoising seq2seq" parece haber convergido a un optimo similar. La decision suele ser practica: **BART** para summarization rapido en GPU consumer; **T5** para multi-task con prompts o para escalar a 11B.

---

## Impacto

- **HuggingFace default**: `pipeline("summarization")` carga `facebook/bart-large-cnn` o `sshleifer/distilbart-cnn-12-6` por defecto. Decenas de miles de aplicaciones en produccion estan ejecutando BART implicitamente.
- **mBART** (Liu et al., 2020): version multilingue entrenada en 25+ idiomas con el mismo objetivo de text infilling + sentence permutation. Base de muchos sistemas de MT zero-shot.
- **PLBART** (Ahmad et al., 2021): aplicado a lenguajes de programacion sobre codigo de GitHub -- code summarization, generation, bug fixing.
- **DistilBART**: variante destilada para deployment.
- **Pegasus** (Zhang et al., 2020): paper concurrente que especializa el denoising para summarization via Gap Sentence Generation; supera a BART en XSum (47.21 vs 45.14) pero con objetivo mas especializado.
- **ProphetNet** (Qi et al., 2020): predice n tokens futuros simultaneamente, mejora marginal sobre BART.

A enero de 2026 el paper acumula **>15,000 citaciones**, entre los papers mas influyentes de pre-training NLP. La idea de "denoising seq2seq" sobrevive en T0, FLAN e indirectamente en el spirit del instruction tuning.

---

## Conexion con la Clase 22

La Clase 22 cubre Text Generation desde seq2seq encoder-decoder. La presentacion en slides 33-41 usa **T5** como ejemplo canonico de pre-trained seq2seq -- pero BART es la alternativa empiricamente equivalente y, en la practica, **mas frecuente en produccion** porque es el default del pipeline de HuggingFace.

Por que BART encaja aqui:

1. **Es lo que esta corriendo cuando se llama `pipeline("summarization")`**. Comprender BART es comprender que hay debajo del default.
2. **Misma familia conceptual que T5**: la intuicion del paper de Raffel aplica directamente; las diferencias (sentinels distinguibles vs `[MASK]` opaco, multi-task vs single-task) enriquecen el espacio de diseno.
3. **Mismas decoding strategies**: beam search con beam = 5, trigram blocking, length penalty, label smoothing $\epsilon = 0.1$ -- los hiperparametros canonicos de la clase son **literalmente** los que el paper original de BART usa.
4. **Hallucination**: la observacion del paper sobre invenciones factuales en Tabla 7 es un puente directo a la discusion de confiabilidad y factualidad en gen libre que aparece transversalmente en la clase y en los labs.
5. **Completa el cuadro historico**: BERT $\to$ GPT-2 $\to$ XLNet $\to$ RoBERTa $\to$ T5 $\to$ BART cierra la era pre-LLM "discriminativo + generativo separados" justo antes de GPT-3 (junio 2020) y el paradigma de scaling decoder-only + in-context learning.

---

## Notas y enlaces

Ver fundamentos: [T5 y encoder-decoder](/fundamentos/t5-encoder-decoder) - [Text summarization](/fundamentos/text-summarization) - [Decoding strategies](/fundamentos/decoding-strategies) - [Transformer](/fundamentos/transformer) - [BERT](/fundamentos/bert) - [Familia GPT](/fundamentos/gpt-family).

Papers relacionados: [T5 (Raffel et al. 2020)](/papers/t5-raffel-2020) - [Pegasus (Zhang et al. 2020)](/papers/pegasus-zhang-2020) - [BERT (Devlin et al. 2018)](/papers/bert-devlin-2018) - [Attention Is All You Need (Vaswani et al. 2017)](/papers/attention-is-all-you-need-vaswani-2017).

Clase: [Clase 22 - Text Generation](/clases/clase-22).

- Codigo y checkpoints originales: [github.com/facebookresearch/fairseq/tree/main/examples/bart](https://github.com/facebookresearch/fairseq/tree/main/examples/bart).
- Modelo de produccion default: [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) en HuggingFace.
- Lectura cruzada recomendada: leer las secciones 4 (ablation de ruidos) y 6 (analisis cualitativo) del paper junto con T5 (Raffel et al.) para contrastar filosofias de pre-training.
