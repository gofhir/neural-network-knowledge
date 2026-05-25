---
title: "PEGASUS (Gap-Sentence Generation)"
weight: 115
math: true
---

{{< paper-card
    title="PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization"
    authors="Zhang, Zhao, Saleh, Liu"
    year="2020"
    venue="ICML 2020"
    pdf="/papers/pegasus-zhang-2020.pdf"
    arxiv="1912.08777" >}}
Propone un **objetivo de pre-entrenamiento alineado con summarization**: enmascarar oraciones completas seleccionadas por su importancia (top-$m$ por ROUGE-1 F-score contra el resto del documento) y forzar al decoder a regenerarlas como secuencia. Con 568M parametros y un corpus mixto C4 + HugeNews, **alcanza SOTA en los 12 benchmarks** de summarization evaluados al momento (XSum, CNN/DailyMail, NEWSROOM, Multi-News, Gigaword, WikiHow, Reddit TIFU, BIGPATENT, arXiv, PubMed, AESLC, BillSum). Demuestra ademas que **1000 ejemplos de fine-tuning bastan** para superar el SOTA previo en seis datasets y obtiene **paridad humana** en XSum, CNN/DailyMail y Reddit TIFU.
{{< /paper-card >}}

---

## El problema

A finales de 2019 el panorama de modelos sequence-to-sequence pre-entrenados estaba dominado por T5, BART, MASS y UniLM. Todos comparten una arquitectura encoder-decoder Transformer y un objetivo de denoising self-supervised, pero **ninguno disena su pre-training pensando en summarization**:

- **T5** (Raffel 2019) enmascara *spans* aleatorios cortos entre tokens sentinel.
- **BART** (Lewis 2019) combina varias noising functions (text infilling, sentence permutation, document rotation).
- **MASS** (Song 2019) enmascara un unico fragmento contiguo y lo regenera.
- **UniLM** (Dong 2019) entrena LM unidireccional, bidireccional y seq-to-seq juntos.

La hipotesis de PEGASUS es directa: **si el objetivo de pre-entrenamiento se parece a la tarea downstream, el transfer learning sera mas eficiente**. En particular, si se enmascaran oraciones enteras importantes y el modelo debe regenerarlas a partir del resto, esta aprendiendo exactamente la dinamica que necesita en summarization -- identificar lo importante, condensarlo, reescribirlo.

Como antecedente directo conviene mencionar dos trabajos. **Khandelwal et al. (2019)** habian probado summarization con un Transformer pre-entrenado sobre Wikipedia y obtenian apenas ROUGE-2 = 13.1 en CNN/DailyMail con 3000 ejemplos -- PEGASUS llega al mismo regimen con 1000 ejemplos y obtiene ROUGE-2 = 19.35. **Radford et al. (2018b)** habian mostrado que GPT-2 podia resumir zero-shot al promptear con "TL;DR" (ROUGE-2 = 8.27 en CNN/DM), prueba de concepto de que language models grandes podian resumir sin supervision pero lejos del SOTA. PEGASUS demuestra que un objetivo **disenado** para summarization llega mucho mas lejos que un objetivo generico de language modeling, incluso con orden de magnitud menos parametros.

La inspiracion inmediata reconocida en el paper viene de dos lineas. Por un lado, los trabajos de masking de spans contiguos como **SpanBERT** (Joshi 2019) y **T5** (Raffel 2019), que mostraron que enmascarar spans en lugar de tokens individuales mejoraba el aprendizaje de fenomenos linguisticos largos. Por otro, **SummaRunner** (Nallapati 2017), un extractive summarizer basado en RNN que seleccionaba oraciones secuencialmente con greedy -- el mismo algoritmo que PEGASUS adopta para la variante Principal-Seq de su seleccion. La originalidad esta en combinar estas dos ideas: tomar la seleccion por importancia del extractive summarization clasico y convertirla en senal de pre-entrenamiento para un modelo abstractive.

---

## Gap Sentence Generation (GSG)

### Formalizacion

Sea un documento $D = \{s_1, s_2, \ldots, s_n\}$ una secuencia ordenada de $n$ oraciones. GSG procede en cuatro pasos:

1. **Seleccion**: elegir un subconjunto $G \subset D$ de tamano $m = \lfloor r \cdot n \rfloor$, donde $r$ es el **Gap Sentence Ratio (GSR)**, tipicamente $r \in [0.15, 0.45]$.
2. **Input al encoder**: reemplazar cada $s_i \in G$ por el token `[MASK1]` en su posicion original:
   $$X = \big( [\text{MASK1}] \text{ si } s_i \in G \text{ else } s_i \big)_{i=1}^{n}$$
3. **Target del decoder**: concatenar las oraciones de $G$ en orden, separadas por un token de separacion.
4. **Loss** cross-entropy autoregresiva estandar:
   $$\mathcal{L}_{\text{GSG}} = -\sum_{t=1}^{|Y|} \log P_\theta(y_t \mid y_{<t}, X)$$

A diferencia de T5 (reconstruye spans con sentinels) y BART (reconstruye toda la entrada), **PEGASUS solo genera las oraciones enmascaradas**, no la entrada completa. El target se parece estructuralmente a un resumen: una secuencia corta de oraciones que sintetizan el contenido del documento.

---

## Seleccion de gap sentences

El paper compara tres familias de estrategias:

### Random-$m$

Se eligen $m$ oraciones uniformemente al azar. Baseline natural.

### Lead-$m$

Las primeras $m$ oraciones del documento. Aprovecha el **lead bias** del newsroom (piramide invertida: lo importante va al inicio). Baseline fuerte en CNN/DailyMail.

### Principal-$m$

Se asigna a cada oracion un score de importancia:

$$\text{score}(s_i) = \text{rouge}\big(s_i,\; D \setminus \{s_i\}\big)$$

y se toman las top-$m$. La intuicion: las oraciones que mas solapan lexicamente con el resto del documento son las mas "centrales", las que mejor resumen el contenido distribuido.

Sobre esta base hay cuatro variantes segun dos ejes:

- **Independent (Ind)** -- cada oracion se puntua una sola vez contra todo el documento.
- **Sequential (Seq)** -- seleccion greedy, evitando redundancia entre las oraciones elegidas.
- **Orig** -- ROUGE con n-gramas con multiplicidad.
- **Uniq** -- n-gramas tratados como conjuntos.

En las ablaciones sobre C4 (Figura 4a), **Ind-Orig** resulta la mejor estrategia en promedio. Es la que se usa en el modelo final, con ruido uniforme de 20% sobre los scores para regularizacion estocastica.

**Resultado de ablacion clave**: **Principal-Ind > Lead > Random** consistentemente, y la brecha es mayor en datasets non-news donde la heuristica de lead bias falla.

### Hyperparametros de GSG

- **GSR optimo**: depende del dataset (XSum 15-30%, WikiHow 45%) pero **siempre por debajo de 50%**: enmascarar demasiado pierde el contexto que guia la generacion.
- **Longitudes**: $L_{\text{input}} = 512$ tokens, $L_{\text{target}} = 256$ tokens en pre-entrenamiento. En fine-tuning sube a 1024/256 para datasets con documentos largos.

---

## Arquitectura

PEGASUS no propone una arquitectura nueva. Es un **encoder-decoder Transformer estandar** al estilo Vaswani 2017, con positional encodings sinusoidales y SentencePiece Unigram (Kudo 2018) como tokenizer, vocabulario 96k.

| Modelo | Layers $L$ | Hidden $H$ | FFN $F$ | Heads $A$ | Params |
|--------|-----------|------------|---------|-----------|--------|
| PEGASUS-base  | 12 | 768  | 3072 | 12 | 223M |
| PEGASUS-large | 16 | 1024 | 4096 | 16 | 568M |

$L$ es el numero de bloques **en cada uno** del encoder y del decoder. PEGASUS-large es comparable a BART-large (406M) y T5-base (220M) en orden de magnitud.

---

## Pre-training corpus

Dos corpora masivos:

- **C4** (Raffel 2019): version limpia del Common Crawl, **350M paginas web, 750 GB**. Cobertura general.
- **HugeNews**: dataset **nuevo del paper, 1.5B articulos** curados de Common Crawl entre 2013-2019 (3.8 TB de texto), filtrado por whitelist de dominios de news. **No fue liberado publicamente** (solo los checkpoints), lo que ha generado criticas de reproducibilidad.

Optimizer Adafactor (Shazeer & Stern 2018), 500k-1.5M steps, batch 8192 en PEGASUS-large, square root LR decay, dropout 0.1.

---

## Resultados

### Tabla principal (ROUGE-1 / ROUGE-2 / ROUGE-L F1)

| Dataset | TransformerBASE | PEGASUS-base | SOTA previo | PEGASUS-large (HugeNews) |
|---------|----------------|-------------|---------------|--------------------------|
| **XSum** | 30.83/10.83/24.41 | 39.79/16.58/31.70 | 45.14/22.27/37.25 (BART) | **47.21/24.56/39.25** |
| **CNN/DailyMail** | 38.27/15.03/35.48 | 41.79/18.81/38.93 | 44.16/21.28/40.90 (BART) | **44.17/21.47/41.11** |
| NEWSROOM | 40.28/27.93/36.52 | 42.38/30.06/38.52 | 39.91/28.38/36.87 | **45.15/33.51/41.33** |
| Multi-News | 34.36/5.42/15.75 | 42.24/13.27/21.44 | 43.47/14.89/17.41 | **47.52/18.72/24.91** |
| BIGPATENT | 42.98/20.51/31.87 | 43.55/20.43/31.80 | 37.52/10.63/22.79 | **53.41/32.89/42.07** |
| arXiv | 35.63/7.95/20.00 | 34.81/10.16/22.50 | 41.59/14.26/23.55 | **44.67/17.18/25.73** |
| PubMed | 33.94/7.43/19.02 | 39.98/15.15/25.23 | 40.59/15.59/23.59 | **45.09/19.56/27.42** |
| WikiHow | 32.48/10.53/23.86 | 36.58/15.64/30.01 | 28.53/9.23/26.54 | **41.35/18.51/33.42** |

PEGASUS-large gana en **los 12 benchmarks** simultaneamente. Observaciones:

- **HugeNews gana en news** (XSum, CNN/DM, NEWSROOM, Multi-News). En XSum el salto frente a BART es notable: +2.07 ROUGE-1.
- **C4 gana en non-news** (WikiHow, Reddit TIFU, BIGPATENT, arXiv, PubMed, AESLC). El corpus generico transfiere mejor fuera de noticias.
- **PEGASUS-base ya supera el SOTA previo** en siete de doce datasets pese a tener menos parametros que los modelos comparados.

### Low-resource shine (Figura 6)

El resultado mas impactante para la practica:

- **Zero-shot** (sin fine-tuning): ROUGE-2 = 13.28 en CNN/DailyMail, ~60% mejor que GPT-2 con prompt "TL;DR".
- **10 ejemplos**: 15.84 R2 en CNN/DM.
- **100 ejemplos**: ya supera el SOTA previo en BIGPATENT, Reddit TIFU, BillSum.
- **1000 ejemplos**: supera el SOTA previo en seis datasets (Multi-News, WikiHow, Reddit TIFU, BIGPATENT, AESLC, BillSum).
- **1000 ejemplos ~ Transformer-base con dataset completo** en datasets medianos.

La intuicion: GSG ya enseno al modelo a generar texto en formato resumen; el fine-tuning solo le indica el estilo y longitud especificos del dataset target.

---

## Ablations clave

Ablaciones realizadas sobre PEGASUS-base + C4 con 500k steps:

1. **Estrategia de seleccion**: **Principal-Ind-Orig > Lead > Random** consistentemente. La brecha es mayor en datasets non-news.
2. **GSR**: optimo en 15-45%, **siempre <50%**. Enmascarar demasiado destruye el contexto.
3. **GSG vs MLM**: GSG-solo > MLM+GSG > MLM-solo. MLM+GSG converge mas rapido pero se estanca, por eso PEGASUS-large no incluye MLM en su objetivo final.
4. **Corpus**: **HugeNews mejor en news, C4 mejor en non-news**. Confirma que el dominio del pre-training corpus condiciona transferencia.
5. **Vocabulario**: Unigram 96k > BPE 32k > Unigram 32k. La granularidad fina ayuda en summarization donde se generan palabras raras.

---

## Human evaluation

En Amazon Mechanical Turk, 3 jueces por ejemplo, escala Likert 1-5:

| Dataset | Humano | PEGASUS-large (HugeNews) | $p$-value vs humano |
|---------|--------|-------------------------|---------------------|
| XSum | 3.0 | 3.1 | 0.7 (no diferente) |
| CNN/DailyMail | 3.1 | 3.6 | 0.007 (**mejor que humano**) |
| Reddit TIFU | 3.2 | 3.1 | 0.3 (no diferente) |

**Paridad humana en XSum y Reddit TIFU**, y evaluacion **superior al resumen humano de referencia en CNN/DailyMail** ($p<0.01$). En low-resource, **100 ejemplos** ya bastan para alcanzar paridad humana en XSum y CNN/DailyMail.

### Test-set overlap

Los autores verifican que las mejoras no vienen de memorizacion: HugeNews y C4 son crawls de la web y podrian contener test examples. Filtrando ejemplos con similaridad >0.8 entre test target y corpus de pre-training, **ROUGE cambia en menos de 1%**. Las ganancias no se explican por memorizacion.

---

## Limitaciones

1. **Alucinaciones factuales**: PEGASUS produce resumenes fluidos pero ocasionalmente confunde entidades nombradas o numeros (el Appendix muestra un caso confundiendo "California" con "north carolina"). ROUGE no captura factualidad.
2. **Sesgo extractivo residual**: aunque GSG empuja hacia abstractive, el modelo reusa frases del documento; problematico en XSum.
3. **Solo ingles**: PEGASUS original es monolingue.
4. **Sin instruction tuning**: cada dataset requiere fine-tuning especifico; no hay una "instruccion" universal como en FLAN-T5.
5. **HugeNews no es publico**: la mejor variante se entreno con un corpus privado, comprometiendo la reproducibilidad.
6. **Documentos largos**: limitado a 1024 tokens en fine-tuning. Excluye papers cientificos completos, libros, contratos largos.
7. **ROUGE como criterio y metrica**: Principal selecciona oraciones por ROUGE-1 F-score y luego se evalua con ROUGE; la circularidad metodologica puede sesgar hacia outputs con alto solapamiento lexico, no necesariamente mejores semanticamente.
8. **Sesgo del corpus de news**: HugeNews sobre-representa lenguaje periodistico; el modelo tiende a producir resumenes con estructura de lead noticioso, inapropiada para literatura o conversacion.

---

## Sucesores e impacto

PEGASUS ha sido extraordinariamente influyente:

- **PEGASUS-X** (Phang et al., 2022): extension a documentos largos hasta 16k tokens con staggered block-local attention. Misma idea base de GSG pero arquitectura para long-context.
- **mPEGASUS**: variantes multilingues (XLSum, WikiLingua).
- **FLAN-PEGASUS**: instruction tuning sobre PEGASUS pre-entrenado.
- **GSG como meta-pattern**: "elegir un subconjunto importante del input y hacer al modelo regenerarlo" se ha aplicado a dialogue summarization, code summarization y meeting summarization.

En el ecosistema HuggingFace los checkpoints siguen siendo baseline obligado: `google/pegasus-cnn_dailymail`, `google/pegasus-xsum`, `google/pegasus-multi_news`, `google/pegasus-pubmed`, `google/pegasus-arxiv`, `google/pegasus-billsum`, `google/pegasus-large`.

```python
from transformers import pipeline
summarizer = pipeline("summarization", model="google/pegasus-xsum")
summarizer(documento)
```

---

## T5 vs BART vs PEGASUS

| Aspecto | T5 (2019) | BART (2019) | PEGASUS (2020) |
|---------|-----------|-------------|-----------------|
| Arquitectura | Encoder-decoder | Encoder-decoder | Encoder-decoder |
| Tamano "large" | 220M-11B | 406M | 568M |
| Objetivo | Span corruption (spans cortos) | Multiple noising (text infilling, permutation, rotation) | **GSG** sobre oraciones completas seleccionadas por importancia |
| Corpus | C4 (750 GB) | 160 GB news/books/web | C4 + HugeNews (3.8 TB) |
| Target | Spans entre sentinels | Toda la entrada original | **Solo oraciones enmascaradas** (formato de resumen) |
| Optimizado para | Text-to-text general | Generacion + comprension general | **Summarization especifico** |
| XSum (R1/R2/RL) | -- | 45.14/22.27/37.25 | **47.21/24.56/39.25** |
| CNN/DM (R1/R2/RL) | 43.52/21.55/40.69 | 44.16/21.28/40.90 | **44.17/21.47/41.11** |
| Instruction-tunable | Si (corazon de FLAN-T5) | No nativo | No nativo |
| Multilingue | mT5 disponible | mBART disponible | Solo ingles |

**Conclusion cualitativa**:

- En **summarization extrema** (XSum), PEGASUS gana claramente porque GSG fue disenado exactamente para ese regimen.
- En **summarization de news con resumenes mas largos** (CNN/DailyMail), los tres modelos estan casi empatados; PEGASUS gana por margen estrecho.
- En **tareas no-summarization** (clasificacion, QA, parsing), T5 y BART son mejores porque su pre-training objective es mas general.
- Para **produccion**, T5 ofrece mas flexibilidad multitask y multilingue; BART es buen all-rounder; PEGASUS es la opcion mas fuerte cuando la tarea es claramente summarization en ingles.

---

## Conexion con la clase 22

La clase 22 cubre summarization moderno con foco en T5 como ejemplo cannon de encoder-decoder pre-entrenado. PEGASUS **no aparece explicitamente en el PDF de la clase** pero es **el modelo summarization-specific** que complementa a T5 (general-purpose) en el toolkit moderno: cuando la tarea es claramente summarization en ingles, PEGASUS suele ser la primera eleccion.

Para casos clinicos y healthcare (dominio de Roberto), la combinacion habitual ha sido:

- **PEGASUS-pubmed**: pre-entrenado en C4 y fine-tuned en PubMed; estandar de hecho para summarization de abstracts medicos.
- **BioBART / ClinicalBART**: variantes de BART en dominio biomedico.
- **T5/FLAN-T5**: cuando se necesita flexibilidad (resumir, traducir, clasificar) en el mismo modelo.

La leccion operativa de PEGASUS -- **alinear el objetivo de pre-entrenamiento con la tarea downstream** -- es trasladable a otros dominios: un modelo para extraccion de entidades clinicas podria pre-entrenarse enmascarando entidades en lugar de spans aleatorios; un modelo para CDA-to-FHIR conversion podria entrenarse con noising functions especificas de XML/JSON estructurado.

El segundo takeaway practico es que **el regimen low-resource cambia la economia de la summarization**: con 100-1000 ejemplos curados de calidad, un PEGASUS-large fine-tuned puede alcanzar paridad con modelos entrenados sobre datasets de cientos de miles. Para casos FHIR donde recolectar pares (nota clinica, resumen estructurado) es costoso, esto representa meses de anotacion evitados.

---

## Notas y enlaces

- **Figura 1** del paper: ilustracion de GSG combinado con MLM sobre un mismo ejemplo.
- **Figura 4** (ablaciones): comparacion de estrategias de seleccion y GSR.
- **Figura 6** (low-resource): curva de ROUGE vs numero de ejemplos de fine-tuning, donde se ve la "rodilla" en ~1000 ejemplos.
- **Figura 7**: analisis de overlap test-set vs corpus de pre-training para descartar memorizacion.
- Codigo y checkpoints: [github.com/google-research/pegasus](https://github.com/google-research/pegasus).
- Implementacion HuggingFace: `PegasusTokenizer`, `PegasusForConditionalGeneration`.

Ver fundamentos: [Text Summarization](/fundamentos/text-summarization) - [T5 Encoder-Decoder](/fundamentos/t5-encoder-decoder) - [Decoding Strategies](/fundamentos/decoding-strategies) - [ROUGE](/fundamentos/rouge-metric).

Ver papers: [T5 (Raffel 2020)](/papers/t5-raffel-2020) - [BART (Lewis 2020)](/papers/bart-lewis-2020) - [BERTSum (Liu 2019)](/papers/bertsum-liu-2019) - [XSum (Narayan 2018)](/papers/xsum-narayan-2018).

Ver clase: [Clase 22 -- Summarization](/clases/clase-22).
