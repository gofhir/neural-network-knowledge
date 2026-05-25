---
title: "T5 (Text-to-Text Transfer Transformer)"
weight: 110
math: true
---

{{< paper-card
    title="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
    authors="Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li, Liu"
    year="2020"
    venue="JMLR 21 (2020)"
    pdf="/papers/t5-raffel-2020.pdf"
    arxiv="1910.10683" >}}
Propone un **framework text-to-text unificado** donde toda tarea NLP -- traduccion, clasificacion, regresion, QA, summarization -- se reformula como mapear texto de entrada (con un prefijo identificador) a texto de salida, con la misma arquitectura encoder-decoder, la misma loss y los mismos hiperparametros. Sobre esa base ejecuta un estudio empirico sistematico de arquitecturas, objetivos, datasets y escalado; introduce el objetivo de **span corruption** y el corpus **C4** (~750 GB filtrados de Common Crawl); y escala a **11B parametros** alcanzando state-of-the-art en 18 de 24 benchmarks en 2020 (GLUE 90.3, SuperGLUE 88.9, SQuAD EM 91.26, CNN/DM ROUGE-1 43.52).
{{< /paper-card >}}

---

## El problema

A inicios de 2020 el campo del transfer learning para NLP estaba fragmentado. Entre 2018 y 2019 habian aparecido ELMo, GPT-1, BERT, GPT-2, XLNet, RoBERTa, ALBERT, SpanBERT y MASS, cada uno con su propia combinacion de arquitectura (encoder-only, decoder-only, encoder-decoder), objetivo no supervisado (MLM, NSP, permutation LM, span masking), dataset (Wikipedia, BookCorpus, OpenWebText, CC-News) y tricks de optimizacion.

Esta fragmentacion impedia responder preguntas basicas: que del exito de BERT viene de su arquitectura, que de su objetivo MLM, que de su dataset, que de su escala? Es mejor encoder-only para clasificacion y encoder-decoder para generacion, o un solo modelo puede dominar ambos regimenes? El campo carecia de un **framework unificado** que permitiera comparaciones controladas.

T5 propone tres contribuciones entrelazadas para resolver esto: (i) un **framework text-to-text** que reduce toda tarea NLP a la misma estructura input -> output; (ii) un **estudio empirico exhaustivo** que varia un factor a la vez sobre ese framework (arquitectura, objetivo, dataset, mixing, escalado); (iii) una **receta final escalada a 11B** que valida los hallazgos del estudio.

---

## El framework text-to-text

La idea central se resume en la Figura 1 del paper: el modelo recibe textos de entrada con un **prefijo identificador** y produce textos de salida. Ejemplos:

| Tarea | Input | Output |
|---|---|---|
| Traduccion EN->DE | `translate English to German: That is good.` | `Das ist gut.` |
| Aceptabilidad (CoLA) | `cola sentence: The course is jumping well.` | `not acceptable` |
| Similitud (STS-B) | `stsb sentence1: ... sentence2: ...` | `3.8` |
| Summarization (CNN/DM) | `summarize: state authorities dispatched emergency crews...` | `six people hospitalized after a storm in attala county.` |

Bajo este formato, **una sola funcion de perdida** -- cross-entropy de maxima verosimilitud con teacher forcing -- entrena al modelo en todas las tareas:

$$
\mathcal{L}(\theta) = -\sum_{(x, y) \in \mathcal{D}} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{<t}, x)
$$

donde $x$ es el input con prefijo e $y$ es la secuencia objetivo (etiqueta como string, traduccion, resumen, etc.). Casos especiales: STS-B (regresion continua 1-5) se discretiza redondeando al multiplo de 0.2 y se emite como string `"2.6"`. Si el modelo emite algo fuera de las etiquetas validas (ej. `"hamburger"` para entailment), cuenta como error -- en la practica nunca ocurrio.

A diferencia de la era posterior de prompt engineering, los prefijos T5 son **tokens fijos casi opacos** que el modelo aprende a asociar con tareas durante fine-tuning, no instrucciones en lenguaje natural. Los autores reportan que el wording exacto del prefijo tiene impacto limitado y no hicieron busqueda extensiva. Esto contrasta con FLAN/T0 posteriores, donde la instruccion verbal explicita ("Resume el siguiente articulo") cobra protagonismo.

---

## Span-corruption pretraining

El objetivo de pre-entrenamiento es la contribucion tecnica mas distintiva. T5 abandona el MLM token-a-token de BERT en favor de un **denoising por spans**:

1. Se selecciona el **15% de los tokens** para corrupcion.
2. Los tokens corruptos contiguos se agrupan en spans y se reemplazan en el input por un **sentinel unico** (`<X>`, `<Y>`, `<Z>`...). Cada sentinel es un token especial del vocabulario.
3. El target es la concatenacion de los spans removidos, cada uno precedido por su sentinel, mas un sentinel final que marca fin de secuencia.

Ejemplo canonico del paper:

```
Original: Thank you for inviting me to your party last week.
Input  : Thank you <X> me to your party <Y> week.
Target : <X> for inviting <Y> last <Z>
```

Aqui `for inviting` se colapsa bajo `<X>`, `last` bajo `<Y>`, y `<Z>` cierra el target.

**Ventajas respecto del MLM de BERT**:

- **Target mas corto**: en MLM se reconstruye toda la secuencia; aqui solo los spans corruptos. Reduce FLOPs del decoder.
- **Coherencia con encoder-decoder generativo**: encaja naturalmente con autorregresion sobre los spans faltantes.
- **Inductive bias hacia composicionalidad**: predecir spans de varios tokens fuerza a modelar frases y no solo palabras aisladas.

**Ablations clave**: el paper compara prefix-LM, BERT-style MLM, deshuffling, MASS-style, drop-corrupted-spans; varia la corruption rate (10%, **15%**, 25%, 50%); y la longitud media de span (2, **3**, 5, 10). El sweet spot es **15% / span length 3**. La diferencia entre variantes razonables del denoising es pequena, asi que la eleccion se justifica por **eficiencia computacional**.

El span corruption se mantuvo como objetivo de referencia en BART (corrupciones diversas), PEGASUS (gap sentence generation) y UL2 (mezcla de denoisings).

---

## Arquitectura T5

T5 es un **encoder-decoder Transformer estandar** (Vaswani et al. 2017) con tres modificaciones menores:

1. **LayerNorm simplificado**: solo reescala las activaciones, omite el sesgo aditivo. Es esencialmente **RMSNorm avant la lettre**, popularizado luego por LLaMA.
2. **Pre-norm**: el LayerNorm se aplica antes de cada subcomponente, fuera del residual path. Estabiliza redes profundas.
3. **Relative position bias** simplificado: en vez de embeddings sinusoidales o absolutas, T5 usa un **escalar relativo** que se suma al logit de atencion. Cada head tiene su propio embedding, pero los embeddings se **comparten entre capas**. Se usan 32 buckets con rangos logaritmicos hasta offset 128.

Configuracion por variante:

| Modelo | $d_{\text{model}}$ | $d_{\text{ff}}$ | heads | layers (enc+dec) | Parametros |
|---|---|---|---|---|---|
| T5-Small | 512 | 2048 | 8 | 6+6 | ~60 M |
| T5-Base | 768 | 3072 | 12 | 12+12 | ~220 M |
| T5-Large | 1024 | 4096 | 16 | 24+24 | ~770 M |
| T5-3B | 1024 | 16 384 | 32 | 24+24 | ~2.8 B |
| T5-11B | 1024 | 65 536 | 128 | 24+24 | ~11 B |

T5-3B y T5-11B escalan **$d_{\text{ff}}$** masivamente manteniendo $d_{\text{model}}=1024$, porque las TPU son mas eficientes en matmuls grandes de feed-forward que en aumentar la dimension residual.

Otros detalles: optimizador **AdaFactor** (memory-efficient, factoriza estadisticas), dropout 0.1, vocabulario compartido SentencePiece de 32k entrenado sobre mezcla EN/DE/FR/RO para soportar traduccion, schedule inverse-square-root con 10k pasos de warmup, batch packing de 65k tokens por step. T5 no introduce el scaling moderno (sin attention scaling factor explicito ni embeddings tied del estilo posterior).

---

## C4 -- Colossal Clean Crawled Corpus

Pre-entrenar a la escala que T5 explora requiere un dataset masivo. Common Crawl produce ~20 TB de texto crudo cada mes, pero la mayor parte es ruido. Los autores aplican una bateria de heuristicas sobre el dump de abril 2019:

1. Solo lineas terminadas en puntuacion final (., !, ?, comilla).
2. Paginas con menos de 3 oraciones se descartan; lineas con menos de 5 palabras se descartan.
3. Filtro de palabras ofensivas (LDNOOBW).
4. Eliminacion de lineas con "Javascript" (warnings de habilitar JS).
5. Descarte de placeholder "Lorem ipsum".
6. Descarte de paginas que contengan llaves `{` (anti-codigo).
7. Limpieza de markers tipo `[1]`, `[citation needed]`.
8. Eliminacion de boilerplate legal ("terms of use", "privacy policy").
9. **Deduplicacion** por spans de 3 oraciones repetidos.
10. Filtro de idioma: solo paginas detectadas como ingles con probabilidad >= 0.99.

El resultado: **~750 GB de texto limpio en ingles**, ordenes de magnitud mas grande que Wikipedia (~16 GB), Wiki+TBC (~20 GB), WebText (~40 GB) o RealNews (~35 GB).

Ablations sobre dataset confirman dos hechos: (i) C4 sin filtrar pierde en todas las tareas -- los filtros importan; (ii) datasets in-domain mas pequenos a veces ganan en benchmarks del mismo dominio (Wiki+TBC mejora MultiRC), pero a costa de generalidad. Otro experimento muestra que repetir hasta 64 veces es inocuo, pero repetir >1000 veces genera **memorizacion** y degrada downstream (GLUE 76.34 vs baseline 83.28).

**Limitaciones de C4 reconocidas posteriormente** (Dodge et al. 2021, "Documenting C4"): el filtro de bad words sobre-filtra contenido LGBT+ y minorias; el requisito de puntuacion final excluye poesia y formatos web modernos; el filtro de llaves elimina blogs tecnicos y LaTeX; el threshold de idioma 0.99 excluye code-switching. Estos sesgos heredados se propagan a todos los modelos pre-entrenados sobre C4.

---

## Comparacion de arquitecturas

El paper explora tres estructuras bajo el framework text-to-text:

1. **Encoder-decoder estandar** (baseline T5): encoder con fully-visible mask, decoder causal con cross-attention.
2. **Decoder-only LM**: pila Transformer causal alimentada con `input target` concatenado (estilo GPT).
3. **Prefix-LM**: como decoder-only pero con fully-visible mask sobre el prefijo (input) y causal sobre el target.

Normalizando por parametros $P$ y FLOPs $M$:

| Arquitectura | Objetivo | Params | Cost | GLUE | CNNDM | SQuAD | SGLUE |
|---|---|---|---|---|---|---|---|
| **Encoder-decoder** | Denoising | 2P | M | **83.28** | **19.24** | **80.88** | **71.36** |
| Enc-dec shared params | Denoising | P | M | 82.81 | 18.78 | 80.63 | 70.73 |
| Language model | Denoising | P | M | 74.70 | 17.93 | 61.14 | 55.02 |
| Prefix LM | Denoising | P | M | 81.82 | 18.61 | 78.94 | 68.11 |
| Encoder-decoder | LM | 2P | M | 79.56 | 18.59 | 76.02 | 64.29 |

**Conclusiones**: encoder-decoder con denoising gana consistentemente; compartir parametros entre encoder y decoder cuesta poco (importa mas la estructura computacional que el total de parametros); denoising > LM causal en NLU; prefix-LM queda en un punto intermedio razonable.

---

## Multi-task fine-tuning

Cuando se entrena un solo modelo sobre multiples tareas, como elegir la proporcion de cada una? El paper explora tres estrategias de mixing: **examples-proportional** con cap $K$, **temperature-scaled** (tasas elevadas a $1/T$), y **equal mixing**. Resultado: equal mixing degrada significativamente (GLUE 76.13 vs 83.28); examples-proportional con cap razonable se acerca al baseline pero **no iguala** a pre-train + fine-tune.

La estrategia ganadora es **multi-task pre-training + fine-tuning** por tarea: mezclar el objetivo no supervisado con todas las tareas supervisadas durante pre-training, luego fine-tunear individualmente en cada downstream. Resultados comparables al baseline (GLUE 83.11) con la ventaja practica de monitorear desempeno downstream durante pre-training. El experimento leave-one-out muestra que la task interference no es destructiva.

Tareas en el mix supervisado: **CNN/DM** (summarization), **GLUE** y **SuperGLUE** (NLU), **SQuAD** (QA generativo), **WMT** EN-DE/FR/RO (traduccion). Multi-task pre-training reduce la varianza de fine-tuning y mejora low-resource tasks como CB y COPA.

---

## Resultados: state-of-the-art en 2020

La receta final (Seccion 3.7) combina: span corruption (15%, span length 3), 1 trillon de tokens de pre-training (32x mas que el baseline), multi-task pre-training con examples-proportional, fine-tuning individual por tarea, beam search (width 4, length penalty $\alpha=0.6$) para outputs largos.

| Modelo | GLUE | SuperGLUE | SQuAD EM | SQuAD F1 | WMT EnDe | CNN/DM R-1 | CNN/DM R-2 | CNN/DM R-L |
|---|---|---|---|---|---|---|---|---|
| Previous best | 89.4 | 84.6 | 90.1 | 95.5 | 33.8 | 43.47 | 20.30 | 40.63 |
| T5-Small (60M) | 77.4 | 63.3 | 79.10 | 87.24 | 26.7 | 41.12 | 19.56 | 38.35 |
| T5-Base (220M) | 82.7 | 76.2 | 85.44 | 92.08 | 30.9 | 42.05 | 20.34 | 39.40 |
| T5-Large (770M) | 86.4 | 82.3 | 86.66 | 93.79 | 32.0 | 42.50 | 20.68 | 39.75 |
| T5-3B (2.8B) | 88.5 | 86.4 | 88.53 | 94.95 | 31.8 | 42.72 | 21.02 | 39.94 |
| **T5-11B** | **90.3** | **88.9** | **91.26** | **96.22** | 32.1 | **43.52** | **21.55** | **40.69** |

**Highlights**:

- **GLUE 90.3** -- supera ALBERT (89.4) con un solo modelo (sin ensemble).
- **SuperGLUE 88.9** -- gana al SOTA previo (84.6) por margen amplio, casi al nivel humano (89.8).
- **SQuAD EM 91.26 / F1 96.22** -- gana ALBERT.
- **CNN/DM ROUGE-1 = 43.52, R-2 = 21.55, R-L = 40.69** -- state-of-the-art en **abstractive summarization** y referencia obligatoria del campo.
- **WMT**: NO logra SOTA. Los autores lo atribuyen a pre-training monolingue; los metodos SOTA usan backtranslation y datos multi-source.

T5-11B logra SOTA en **18 de 24 tareas**. Los 6 perdidos son traducciones (limitadas por monolinguismo) y common-sense low-resource (COPA, WSC) donde humanos llegan a 100%.

**Scaling laws empiricas del paper**: el estudio sistematico de escalado (Tabla 13) compara como gastar 4x mas computo. El veredicto: aumentar tamano del modelo da el mayor bump; pasos x4 y batch x4 son aproximadamente equivalentes; tamano y pasos son complementarios (2x size x 2x steps ~ 4x size x 1x steps); ensembling de 4 modelos gana en generacion pero no en clasificacion. Se confirma la "bitter lesson" de Sutton: el escalado vence a los trucos algoritmicos.

---

## Variantes

T5 sembro una progenie extensa:

- **mT5** (Xue et al. 2021): T5 multilingue entrenado sobre **mC4** (101 idiomas). Aplicable a tareas multilingues y cross-lingual transfer. A diferencia de T5 monolingue, **requiere fine-tuning supervisado** porque solo se pre-entreno con span corruption, sin multi-task pre-training.
- **UMT5** (Chung et al. 2023): version universal con mejoras en tokenizacion y task generalization.
- **ByT5** (Xue et al. 2022): variante **byte-level**, sin tokenizacion; util para low-resource languages y robustez a errores ortograficos.
- **FLAN-T5** (Chung et al. 2022, "Scaling Instruction-Finetuned Language Models"): T5 con **instruction tuning** sobre cientos de tareas con prompts en lenguaje natural. Convierte T5 de modelo task-specialized a general-purpose instruction follower.
- **T0** (Sanh et al. 2022): T5 entrenado con prompted multi-task supervised training explicitamente para **zero-shot generalization**.

Otros derivados del paradigma text-to-text: **CodeT5** (programacion), **Pix2Seq** (object detection como secuencia), **PEGASUS** (summarization-specific pretraining), **BART** (corrupciones mas variadas).

---

## Limitaciones

A pesar del exito, T5 tiene limitaciones reconocidas explicitamente en la Seccion 4.2:

1. **Pre-training English-only**: limita traduccion multilingue. Resuelto por mT5.
2. **Costo de inferencia de 11B**: impractico en produccion. Los autores abogan por distilacion, parameter sharing y mixture-of-experts.
3. **Falta de capacidades zero/few-shot estilo GPT-3**: T5-11B **requiere fine-tuning por tarea**. GPT-3 175B (publicado meses despues) demostro que prompting in-context puede sustituir el fine-tuning. T5 es fundamentalmente task-specialized via fine-tuning, no general-purpose few-shot learner.
4. **Prefijos rigidos**: `"summarize:"`, `"cola sentence:"` son tokens fijos. FLAN/T0 introducirian instrucciones en lenguaje natural.
5. **Fixed context window**: el relative position bias discrimina hasta offset 128 y comparte embeddings mas alla; el input efectivo es 512 tokens. Para documentos largos hay que truncar o usar chunk-based summarization.
6. **Eficiencia del span corruption**: aprender denoising sobre 1T tokens es caro. ELECTRA (Clark et al. 2020) propone objetivos mas eficientes (replaced-token detection).

---

## Por que importa hoy

T5 sigue vigente media decada despues por varias razones:

1. **El paradigma encoder-decoder no murio**. Aunque la era LLM se inclino hacia decoder-only (GPT-3, LLaMA, Claude), los encoder-decoders siguen siendo competitivos para tareas seq2seq: summarization, traduccion, paraphrasing, structured generation. FLAN-T5 11B sigue siendo baseline en muchos papers actuales.

2. **C4 se convirtio en estandar de facto**. El pipeline reproducible de filtrado de Common Crawl que T5 inauguro marco el camino para datasets como The Pile, RefinedWeb, FineWeb, RedPajama. La conciencia sobre filtrado de datos web nace, en gran medida, con C4.

3. **Foundation para Flan-T5 y mT5**. La era de instruction tuning (FLAN, T0) se construye **encima** de T5 base. Sin T5 no hay FLAN-T5, que sigue siendo workhorse para tareas multilingues y de generacion controlada en 2026.

4. **Framework text-to-text como lingua franca**. La idea "todo es texto a texto" articulada por T5 es ahora **el paradigma por defecto** de la era LLM. Chat completions, tool calling, structured output -- todo se reduce a sequence-to-sequence con prompts. T5 lo nombro y lo demostro.

5. **Metodologia rigurosa**. El "coordinate ascent" sobre el espacio de diseno (variar un factor a la vez), reportar varianza inter-run (10 baselines desde inicializaciones distintas), separar validation/test en todas las tablas excepto la final -- estas son practicas de honestidad metodologica que el campo deberia replicar mas seguido.

6. **C4 + 11B abrieron la barrera del trillon de tokens**. Antes de T5 los modelos publicos veian 100-200B tokens; T5 mostro que 1T era posible y deseable. Esta cifra se volveria estandar.

---

## Conexion con la clase 22

La clase 22 del curso IA UC, dictada por Felipe del Rio R., distingue dos paradigmas de summarization:

- **Extractive**: TextRank, LexRank, BertSumExt. Selecciona oraciones del documento original.
- **Abstractive**: genera un resumen nuevo, posiblemente con palabras ausentes del input.

**T5 es el paper estrella del modulo Abstractive Model**. Los slides 36-41 lo cubren directamente: framework text-to-text, prefijo `summarize:`, arquitectura encoder-decoder, span corruption, resultados en CNN/DM, y la familia mT5/UMT5/FLAN-T5/ByT5.

El ejemplo canonico que aparece tanto en la Figure 1 del paper como en los slides:

```
Input : "summarize: state authorities dispatched emergency crews tuesday to
         survey the damage after an onslaught of severe weather in mississippi..."
Output: "six people hospitalized after a storm in attala county."
```

Los numeros ROUGE de T5-11B en CNN/DM (R-1 = 43.52, R-2 = 21.55, R-L = 40.69) son la referencia moderna obligatoria para abstractive summarization. PEGASUS y BART tambien compiten en este benchmark; T5 ofrece la ventaja del **framework unificado**: el mismo modelo que resume CNN/DM puede traducir EN->DE y resolver SQuAD sin cambiar arquitectura.

Para la practica del curso, T5 es notable porque: (i) esta disponible en HuggingFace (`t5-small`...`t5-11b`, `flan-t5-*`); (ii) el prefijo `summarize:` funciona out-of-the-box; (iii) mT5 y FLAN-T5 habilitan summarization multilingue y zero-shot con instrucciones en lenguaje natural. El slide 41 anota explicitamente que **mT5 y UMT5 requieren fine-tuning supervisado** porque solo se pre-entrenaron con span corruption no supervisado.

---

## Notas y enlaces

- Paper extenso (67 paginas con anexos). La lectura util se concentra en Secciones 2-3 (framework, estudio sistematico) y 4 (resultados SOTA + outlook). El Appendix C provee ejemplos de resumenes generados, no cherry-picked.
- **Figuras clave**: Figure 1 (framework text-to-text), Figure 2 (span corruption), Figure 6 (curva de loss vs repeticion).
- **Tablas clave**: Tabla 2 (comparacion de arquitecturas), Tablas 4-7 (ablations de objetivo), Tabla 13 (estudio de escalado), Tabla 14 (resultados finales SOTA).
- Codigo y checkpoints: [github.com/google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer); pesos en HuggingFace.
- Decoding recomendado para summarization: beam search width 4, length penalty $\alpha=0.6$. Max input 512 tokens; max output tipicamente 100-200 en CNN/DM.

Ver fundamentos: [T5 encoder-decoder](/fundamentos/t5-encoder-decoder) - [Text summarization](/fundamentos/text-summarization) - [Decoding strategies](/fundamentos/decoding-strategies) - [Transformer](/fundamentos/transformer) - [BERT](/fundamentos/bert) - [GPT family](/fundamentos/gpt-family).

Ver papers relacionados: [BART](/papers/bart-lewis-2020) - [PEGASUS](/papers/pegasus-zhang-2020) - [BERT](/papers/bert-devlin-2018) - [Attention Is All You Need](/papers/attention-is-all-you-need-vaswani-2017).

Ver [Clase 22 -- Summarization](/clases/clase-22).
