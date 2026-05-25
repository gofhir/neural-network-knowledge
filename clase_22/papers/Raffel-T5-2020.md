---
title: "T5 — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
slug: t5-raffel-2020
authors:
  - Colin Raffel
  - Noam Shazeer
  - Adam Roberts
  - Katherine Lee
  - Sharan Narang
  - Michael Matena
  - Yanqi Zhou
  - Wei Li
  - Peter J. Liu
year: 2020
venue: "Journal of Machine Learning Research 21 (140), pp. 1-67"
arxiv: "1910.10683"
url: "https://arxiv.org/abs/1910.10683"
pdf: "Raffel-T5-2020.pdf"
tags:
  - nlp
  - transfer-learning
  - transformer
  - encoder-decoder
  - summarization
  - text-to-text
  - c4
  - span-corruption
  - multi-task-learning
  - scaling
clase: 22
clase_titulo: "Summarization (Felipe del Río R.)"
date: 2020-06-01
---

## Resumen ejecutivo

El paper de Raffel et al. (2020), conocido universalmente como **T5** ("Text-to-Text Transfer Transformer"), es una de las contribuciones más influyentes y exhaustivas en la historia del *transfer learning* para NLP. Su tesis central es deceptivamente simple: **todo problema de procesamiento de lenguaje natural puede formularse como una transformación de texto en texto**. Traducción, clasificación, regresión, *question answering*, *summarization* y todas las tareas tradicionales pasan a compartir la misma arquitectura encoder-decoder, la misma función de pérdida (cross-entropy de máxima verosimilitud), el mismo procedimiento de *decoding* y el mismo conjunto de hiperparámetros. Lo único que cambia entre tareas es el *prefijo* que se concatena al input —`"summarize:"`, `"translate English to German:"`, `"cola sentence:"`, etc.

Sobre esta base unificada, los autores realizan un estudio empírico sistemático que abarca arquitecturas (encoder-decoder, decoder-only, prefix-LM), objetivos no supervisados (LM causal, MLM tipo BERT, deshuffling, *span corruption*), datasets de pre-entrenamiento, estrategias de transferencia, *multi-task learning* y escalado. Como subproducto generan **C4** (Colossal Clean Crawled Corpus), un dataset de ~750 GB de texto inglés filtrado de Common Crawl que se convirtió en estándar de facto para pre-entrenamiento. Su recomendación final combina un objetivo de **span corruption** (15% de tokens enmascarados en *spans* contiguos de longitud media 3) sobre C4 con multi-task pre-training y *fine-tuning* posterior. Las variantes T5-Small (60M), Base (220M), Large (770M), 3B y 11B logran *state-of-the-art* en 18 de 24 *benchmarks*, alcanzando GLUE 90.3, SuperGLUE 88.9, SQuAD EM 91.26 y, para *abstractive summarization* en CNN/DM, ROUGE-1 = 43.52, ROUGE-2 = 21.55 y ROUGE-L = 40.69 con T5-11B.

## Contexto histórico: la era post-BERT y la fragmentación del paradigma

Cuando T5 fue sometido a JMLR (enero de 2020), el campo del *transfer learning* para NLP atravesaba un crecimiento explosivo y caótico. Entre 2018 y 2019 habían aparecido, en rápida sucesión, ELMo (Peters et al., 2018), GPT (Radford et al., 2018), BERT (Devlin et al., 2018), GPT-2 (Radford et al., 2019), XLNet (Yang et al., 2019), RoBERTa (Liu et al., 2019), ALBERT (Lan et al., 2019), SpanBERT (Joshi et al., 2019) y MASS (Song et al., 2019), entre otros. Cada uno proponía variaciones sobre el tema del *pre-training* no supervisado: cambios de arquitectura (encoder-only vs decoder-only vs encoder-decoder), nuevos objetivos (MLM, NSP, *permutation language modeling*, *span masking*), datasets distintos (Wikipedia + BookCorpus, OpenWebText, CC-News, Reddit) y trucos de optimización.

Esta fragmentación dificultaba responder preguntas básicas: ¿el éxito de BERT viene de su arquitectura, de su objetivo MLM, de su dataset, de su escala, o de la combinación? ¿Es mejor encoder-only para clasificación y encoder-decoder para generación, o un solo modelo puede dominar ambos regímenes? ¿Qué tamaño de modelo y cuántos pasos de entrenamiento son razonables? Los autores de T5 reconocen explícitamente que su objetivo "no es proponer nuevos métodos sino proveer una perspectiva comprensiva sobre dónde está el campo", combinando **survey, exploración empírica y comparación rigurosa** bajo un único *framework* unificado.

El paper también se sitúa en el comienzo de la **era del scaling**. Las leyes empíricas que Kaplan et al. (2020) formalizarían meses después ya se intuían: Hestnes et al. (2017) y Shazeer et al. (2017, 2018) habían demostrado que entrenar modelos más grandes con más datos era una estrategia robusta. T5 lleva esta intuición al extremo: sus variantes más grandes (3B y 11B) son **órdenes de magnitud mayores** que cualquier modelo público de NLP en 2019, y solo son viables gracias al acceso a TPU Pods (1024 chips TPU v3 conectados por interconexión de alta velocidad) y a la librería Mesh TensorFlow (Shazeer et al., 2018) para paralelismo de modelo y datos.

La metodología del paper merece destacarse por su rigor: en vez de comparar configuraciones holísticamente (cambiar varias cosas a la vez), los autores adoptan un **coordinate ascent** sobre el espacio de diseño. Fijan un *baseline* razonable (BERT-base-sized encoder-decoder, denoising objective, C4, 2³⁵ tokens de pre-training) y modifican **un factor a la vez**: arquitectura → objetivo → dataset → estrategia de transferencia → escalado. Reconocen explícitamente que esto puede omitir interacciones de segundo orden (e.g., un objetivo dado podría funcionar mejor solo en modelos más grandes), pero el método logra un balance entre cobertura y costo computacional razonable. Para cuantificar la variabilidad inter-run, entrenan el baseline 10 veces desde inicializaciones distintas (Tabla 1 del paper) y reportan la desviación estándar; la mayoría de tareas tienen σ < 1% de la métrica, con excepciones en *low-resource* tasks como CB, CoLA y COPA.

Tres tradiciones previas inspiran el approach text-to-text:

1. **Natural Language Decathlon** (McCann et al., 2018) — todas las tareas de NLP como *question answering*.
2. **Language Models are Multitask Learners** (Radford et al., 2019, GPT-2) — *zero-shot* prompting de un LM causal.
3. **Span extraction** unificado (Keskar et al., 2019).

T5 toma de estas ideas la unificación de formato, pero con dos diferencias críticas: usa **prefijos cortos** en vez de un formato question-answer obligatorio, y soporta tareas verdaderamente generativas (traducción, resumen) donde no se puede enumerar todas las salidas posibles.

## El framework text-to-text

La idea central del paper se resume en una sola figura (Figure 1 del PDF): el modelo T5 recibe textos de entrada con un *prefijo* identificador y produce textos de salida.

Ejemplos del paper:

| Tarea | Input | Output |
|---|---|---|
| Traducción EN→DE | `translate English to German: That is good.` | `Das ist gut.` |
| Aceptabilidad gramatical (CoLA) | `cola sentence: The course is jumping well.` | `not acceptable` |
| Similitud semántica (STS-B) | `stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.` | `3.8` |
| Resumen (CNN/DM) | `summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught of severe weather in mississippi...` | `six people hospitalized after a storm in attala county.` |

Bajo este formato, **una sola función de pérdida** —cross-entropy de máxima verosimilitud con *teacher forcing*— entrena al modelo en todas las tareas:

$$
\mathcal{L}(\theta) = -\sum_{(x, y) \in \mathcal{D}} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{<t}, x)
$$

donde $x$ es el input con prefijo y $y$ es la secuencia objetivo (la etiqueta como string, la traducción, el resumen, etc.). La elegancia del enfoque está en que tareas estructuralmente distintas se reducen a un solo problema de *autoregressive sequence-to-sequence*.

Algunos casos requieren artificios:

- **STS-B** es regresión continua entre 1 y 5. Los autores la convierten en clasificación de 21 clases redondeando al múltiplo más cercano de 0.2 y emitiendo la salida como string (`"2.6"`). En *test time* parsean el string a *float*; si el modelo emite algo no parseable, se cuenta como error.
- **Winograd / WNLI / WSC / DPR** se reformulan como predicción del nombre referido por un pronombre ambiguo (no como clasificación binaria).
- **Clasificación**: si el modelo emite una palabra fuera del conjunto de etiquetas válidas (ej. `"hamburger"` para entailment), siempre se cuenta como error, aunque en la práctica esto nunca se observó.

Los autores reportan que el *exact wording* del prefijo es un hiperparámetro con impacto limitado; no realizaron búsqueda extensiva sobre ello. Esta observación es importante: a diferencia de la actual era de *prompt engineering* —donde la formulación exacta del prompt puede mover métricas decenas de puntos— en T5 los prefijos son etiquetas casi opacas que el modelo aprende a asociar con tareas durante fine-tuning, no enunciados en lenguaje natural que el modelo deba "entender". Esta es una de las diferencias filosóficas fundamentales con respecto a la era posterior de *instruction tuning* (FLAN, T0), donde las instrucciones son verbalizaciones explícitas como "Resume el siguiente artículo" o "Determina si la siguiente oración es gramaticalmente aceptable".

**Comparación con frameworks unificadores previos**:

- **Natural Language Decathlon (decaNLP)** de McCann et al. (2018): unifica todas las tareas como *question answering*. Por ejemplo, sentiment analysis se formula como "Is this review positive or negative?". T5 lo simplifica usando prefijos cortos en vez de QA explícito, lo que ahorra tokens y permite tratar la traducción y la generación como tareas naturalmente seq2seq sin disfraz.
- **GPT-2 / TL;DR prompting** (Radford et al., 2019): unifica via *language modeling* causal, alimentando documentos seguidos del literal "TL;DR:" para inducir resúmenes. T5 separa explícitamente input y output via encoder-decoder, lo que evita que el decoder vea contaminado el "contexto" con tokens del target durante training.
- **Span extraction unificado** de Keskar et al. (2019): formula muchas tareas como extracción de spans dentro del input. Esto no permite tareas verdaderamente generativas como traducción o *abstractive summarization*, donde el output puede contener palabras no presentes en el input. T5 evita esta limitación al generar autoregresivamente.

## Arquitectura T5

T5 utiliza un **encoder-decoder Transformer estándar** (Vaswani et al., 2017) con tres modificaciones menores:

1. **LayerNorm simplificado**: en vez de la formulación tradicional con escala y sesgo, T5 usa una versión donde solo se reescalan las activaciones y se omite el sesgo aditivo. Esto es esencialmente **RMSNorm avant la lettre**, popularizado luego por modelos como LLaMA.
2. **Pre-norm**: el LayerNorm se aplica antes de cada subcomponente (auto-atención y feed-forward), fuera del *residual path*. Esto estabiliza el entrenamiento de redes profundas.
3. **Relative position bias** simplificado: en vez de las embeddings posicionales sinusoidales del Transformer original o las absolutas aprendidas de BERT/GPT, T5 usa embeddings de posición *relativas* (Shaw et al., 2018; Huang et al., 2018a) reducidas a un **escalar** que se suma al *logit* de atención. Cada *attention head* dentro de una capa usa su propio embedding, pero los embeddings se **comparten entre capas**. Se usan 32 embeddings, con rangos que crecen logarítmicamente hasta un *offset* de 128 (más allá del cual todas las posiciones colapsan al mismo embedding).

La configuración por variante (Tabla resumen del paper, Sección 3.6 y 3.7):

| Modelo | $d_{\text{model}}$ | $d_{\text{ff}}$ | $d_{\text{kv}}$ | heads | layers (enc + dec) | Parámetros |
|---|---|---|---|---|---|---|
| **T5-Small** | 512 | 2048 | 64 | 8 | 6 + 6 | ~60 M |
| **T5-Base** | 768 | 3072 | 64 | 12 | 12 + 12 | ~220 M |
| **T5-Large** | 1024 | 4096 | 64 | 16 | 24 + 24 | ~770 M |
| **T5-3B** | 1024 | 16 384 | 128 | 32 | 24 + 24 | ~2.8 B |
| **T5-11B** | 1024 | 65 536 | 128 | 128 | 24 + 24 | ~11 B |

El detalle interesante de T5-3B y T5-11B es que **escalan $d_{\text{ff}}$** masivamente manteniendo $d_{\text{model}} = 1024$. Los autores justifican esto porque los aceleradores TPU son más eficientes en *matmuls* densos grandes como los de las feed-forward, en lugar de aumentar la dimensión residual.

Dropout = 0.1 en todas partes. Optimizador **AdaFactor** (Shazeer & Stern, 2018), elegido por su menor *memory footprint* respecto de Adam (almacena estadísticas factorizadas en vez de matrices completas). Vocabulario compartido encoder-decoder con **SentencePiece** (Kudo & Richardson, 2018) de 32 000 *wordpieces*, entrenado sobre una mezcla 10:1:1:1 de inglés/alemán/francés/rumano para soportar las tareas de traducción downstream.

**Schedule de learning rate**: durante pre-training se usa el clásico "inverse square root":

$$
\eta_t = \frac{1}{\sqrt{\max(t, k)}}
$$

con $k = 10^4$ pasos de warmup. Esto da $\eta = 0.01$ durante los primeros 10⁴ pasos y luego decae como $1/\sqrt{t}$. Durante fine-tuning se usa learning rate constante de 0.001. La elección del *inverse square root* sobre el *triangular schedule* (Howard & Ruder, 2018) —que tiene mejor desempeño marginal— responde a una restricción práctica: el inverse square root no requiere conocer de antemano el número total de pasos, lo que es conveniente cuando se varían los pasos en distintos experimentos del estudio sistemático.

**Batch packing**: el paper menciona que múltiples secuencias se "empaquetan" en cada entrada del batch para que cada batch contenga aproximadamente 2¹⁶ = 65 536 tokens. Esto permite usar el throughput de las TPU eficientemente incluso cuando las secuencias individuales son cortas, y es una técnica de ingeniería común pero raramente discutida en papers de modelos.

**Compute total del baseline**: 524 288 pasos × 2¹⁶ tokens/batch ≈ 2³⁵ ≈ 34B tokens vistos durante pre-training. Esto es considerablemente menos que BERT (137B tokens) o RoBERTa (2.2T tokens). Los autores justifican esta elección por presupuesto computacional, sabiendo que en la etapa final (Sección 3.7) podrán escalar a 1T tokens cuando ya conozcan la receta óptima.

## El objetivo de span corruption

El objetivo de pre-entrenamiento es la contribución técnica más distintiva del paper. T5 abandona el MLM token-a-token de BERT en favor de un **denoising por spans**.

**Procedimiento** (Figure 2 del paper):

1. Se toma una secuencia tokenizada.
2. Se seleccionan aleatoriamente **15% de los tokens** para corrupción.
3. Los tokens corruptos contiguos se agrupan en **spans** y se reemplazan en el input por un único **sentinel token** único (`<X>`, `<Y>`, `<Z>`, ...). Cada sentinel es un token especial añadido al vocabulario.
4. El **target** consiste en la concatenación de los spans removidos, cada uno precedido por el sentinel que lo reemplazó en el input, más un sentinel final que marca el fin de secuencia.

**Ejemplo canónico del paper**:

```
Original: Thank you for inviting me to your party last week.
Input  : Thank you <X> me to your party <Y> week.
Target : <X> for inviting <Y> last <Z>
```

Aquí "for" e "inviting" son tokens consecutivos seleccionados y se colapsan bajo `<X>`; "last" se colapsa bajo `<Y>`; `<Z>` marca el final del target.

**Ventajas respecto del MLM de BERT**:

- **Target más corto**: en MLM el target reconstruye la secuencia completa; en span corruption solo contiene los tokens corruptos. Esto reduce el costo computacional del decoder, que ya no necesita atender sobre secuencias largas.
- **Coherencia natural con encoder-decoder generativo**: el encoder ve el input corrupto, el decoder genera autorregresivamente los spans faltantes, en lugar de una arquitectura encoder-only que predice cada posición independientemente.
- **Mayor dificultad implícita**: predecir spans de varios tokens consecutivos requiere modelar dependencias a más largo plazo que predecir tokens individuales rodeados de contexto intacto.

**Ablations clave** (Sección 3.3 del paper):

- **Approaches dispares** (Tabla 4): prefix-LM, BERT-style MLM, deshuffling. Ganadores: BERT-style y prefix-LM dan resultados similares; deshuffling pierde claramente.
- **Variantes BERT-style** (Tabla 5): MASS-style (predecir secuencia completa), replace corrupted spans (la versión T5 *baseline*), drop corrupted tokens. Todas similares; los autores prefieren *replace spans* por la combinación de simplicidad y target corto.
- **Corruption rate** (Tabla 6): 10%, **15%**, 25%, 50%. El 15% es el *sweet spot*; 50% degrada notablemente.
- **Average span length** (Tabla 7): 2, **3**, 5, 10. La longitud 3 da los mejores resultados generales; spans largos (10) degradan moderadamente.

**Conclusión del estudio**: la diferencia entre variantes razonables del objetivo de *denoising* es pequeña; la elección debe hacerse principalmente por **eficiencia computacional**. El span corruption con 15% y span length media 3 (estilo SpanBERT, Joshi et al., 2019) gana por eficiencia.

**Por qué span corruption tiene sentido teóricamente**:

1. **Inductive bias hacia composicionalidad**: predecir un span contiguo de 3 tokens force al modelo a aprender representaciones que capten *frases*, no solo palabras aisladas. Esto se alinea naturalmente con cómo el lenguaje encapsula significado: las unidades semánticamente atómicas suelen ser *n*-gramas (named entities, expresiones idiomáticas, frases verbales) y no palabras.
2. **Mejor approximación a tareas downstream**: muchas tareas downstream involucran generar texto coherente más largo que un token (resumen, traducción, QA generativo). Predicción de spans entrena directamente esta capacidad.
3. **Reducción de "trivial cases"**: en MLM token-a-token, predecir el token enmascarado a veces es trivial cuando el contexto local lo determina (artículos, preposiciones). En span corruption, predecir 3 tokens consecutivos a partir de contexto roto requiere razonamiento sobre estructura de mayor escala.
4. **Eficiencia de target**: si 15% de tokens se enmascaran y se agrupan en spans de longitud media 3, el target tiene aproximadamente 5% del tamaño del input (15% × 3 tokens-por-span + sentinels). Comparado con MASS (predicción de la secuencia completa) o BERT-style (mismo length que input), esto reduce sustancialmente el FLOPs del decoder.

El span corruption se ha mantenido como objetivo de referencia en modelos posteriores: BART (Lewis et al., 2020) usa una variante con corrupciones más diversas; PEGASUS (Zhang et al., 2020) generaliza la idea con *gap sentence generation* (enmascarar oraciones completas); UL2 (Tay et al., 2022) propone una mezcla de varias formas de denoising.

## C4 — Colossal Clean Crawled Corpus

Pre-entrenar a la escala que T5 explora requiere un dataset masivo. Common Crawl produce ~20 TB de texto extraído de la web cada mes, pero la mayor parte es ruido: *menus*, *boilerplate*, mensajes de error, duplicados, *gibberish*, código fuente, contenido ofensivo.

Los autores aplican una **batería de heurísticas** sobre el dump de abril de 2019:

1. **Retención por puntuación**: solo se conservan líneas terminadas en signo de puntuación final (., !, ?, comilla).
2. **Filtros de longitud**: páginas con menos de 3 oraciones se descartan; solo se retienen líneas con al menos 5 palabras.
3. **Filtro de palabras ofensivas**: se descarta cualquier página que contenga palabras de la lista "List of Dirty, Naughty, Obscene or Otherwise Bad Words" (LDNOOBW).
4. **Anti-JavaScript**: se eliminan líneas con la palabra "Javascript" (típicamente warnings de "habilitar JS").
5. **Lorem ipsum**: descarte por presencia de placeholder.
6. **Anti-código**: descarte de páginas que contengan llaves `{` (heurística para detectar código).
7. **Limpieza Wikipedia**: eliminación de markers `[1]`, `[citation needed]`.
8. **Boilerplate legal**: eliminación de líneas con "terms of use", "privacy policy", "cookie policy".
9. **Deduplicación**: para spans de tres oraciones que aparecen más de una vez en el dataset, se conserva solo una copia.
10. **Filtro de idioma**: solo páginas detectadas como inglés por `langdetect` con probabilidad ≥ 0.99.

El resultado: **~750 GB de texto limpio en inglés**. C4 es órdenes de magnitud más grande que los datasets típicos previos (Wikipedia ~16 GB; Wikipedia + Toronto Books ~20 GB; WebText ~40 GB; RealNews ~35 GB).

**Ablations sobre dataset** (Tabla 8): comparando pre-entrenar sobre C4, C4 sin filtrar, RealNews-like, WebText-like, Wikipedia, y Wikipedia + TBC. C4 sin filtrar pierde en todas las tareas; los datasets más pequeños y específicos (Wikipedia + TBC, WebText-like, RealNews-like) a veces ganan en *benchmarks* del mismo dominio (Wikipedia + TBC mejora MultiRC porque TBC viene de libros, similares a los textos de MultiRC). Esto confirma que **pre-training en datos in-domain ayuda**, pero a costa de generalidad.

**Tabla 9**: efecto de repetir datos. Con datasets pequeños (artificialmente truncados a $2^{23}$ tokens, repetidos 4096 veces) se observa **memorización**: el *training loss* baja drásticamente y el desempeño downstream se degrada (GLUE 76.34 vs baseline 83.28). Esto motiva usar datasets grandes como C4 que no se repiten durante el pre-training de $2^{35}$ tokens.

| Tokens únicos | Repeticiones | GLUE | CNNDM | SQuAD | SGLUE |
|---|---|---|---|---|---|
| Full C4 | 0 | **83.28** | **19.24** | **80.88** | **71.36** |
| 2²⁹ | 64 | 82.87 | 19.19 | 80.97 | 72.03 |
| 2²⁷ | 256 | 82.62 | 19.20 | 79.78 | 69.97 |
| 2²⁵ | 1024 | 79.55 | 18.57 | 76.27 | 64.76 |
| 2²³ | 4096 | 76.34 | 18.33 | 70.92 | 59.29 |

Curiosamente, repetir hasta 64 veces es prácticamente inocuo (resultados al nivel del baseline). El daño aparece claramente cuando se repite >1000 veces, posiblemente porque el modelo empieza a memorizar el dataset entero. La Figure 6 del paper muestra la curva de *training loss* divergente: para el dataset más pequeño (2²³), la loss cae a ~0.1 muy rápido, signo claro de memorización.

**Limitaciones explícitas de los filtros de C4** reconocidas posteriormente:

- El filtro de "bad words" tiende a **sobre-filtrar** contenido relacionado con LGBT+ y otros grupos minoritarios cuyas comunidades online usan términos que también aparecen en la lista LDNOOBW (Dodge et al., 2021, "Documenting the English Colossal Clean Crawled Corpus").
- El requisito de puntuación final excluye texto creativo, poesía y formatos web modernos.
- El filtro de llaves `{` elimina indiscriminadamente cualquier página con código, lo que termina afectando blogs técnicos, tutoriales y matemáticas (LaTeX).
- El filtro de idioma con threshold 0.99 puede excluir textos válidos con code-switching.

Estos sesgos heredados de C4 se propagan a todos los modelos pre-entrenados sobre él, incluyendo T5, mT5 y FLAN-T5. La documentación posterior de C4 (Dodge et al., 2021) reveló también que C4 incluye texto patentado, contenido de noticias y blogs de origen no claro, lo que ha sido fuente de discusión legal sobre datasets de entrenamiento de LLMs.

## Comparación de arquitecturas

La Sección 3.2 explora **tres estructuras** bajo el framework text-to-text:

1. **Encoder-decoder estándar** (T5 baseline): encoder con *fully-visible* mask sobre el input; decoder con *causal* mask y atención cruzada al encoder.
2. **Language model decoder-only**: una sola pila Transformer con *causal* mask, alimentada con la concatenación `input target` (al estilo GPT).
3. **Prefix-LM**: como decoder-only, pero usando *fully-visible* mask sobre la porción de prefijo (input) y *causal* mask sobre el target. Equivalente a un encoder-decoder con parámetros compartidos y atención cruzada reemplazada por atención completa sobre input + target.

Para comparación justa, los autores normalizan por dos métricas: número de parámetros $P$ y costo computacional $M$ (FLOPs). Una encoder-decoder con $L$ capas en cada stack tiene $2L$ capas totales y $2P$ parámetros, pero el costo es solo $M$ porque cada stack solo se aplica a una porción de la secuencia.

**Resultados** (Tabla 2):

| Arquitectura | Objetivo | Params | Cost | GLUE | CNNDM | SQuAD | SGLUE |
|---|---|---|---|---|---|---|---|
| **Encoder-decoder** | Denoising | 2P | M | **83.28** | **19.24** | **80.88** | **71.36** |
| Enc-dec, shared params | Denoising | P | M | 82.81 | 18.78 | 80.63 | 70.73 |
| Enc-dec, L/2 capas | Denoising | P | M/2 | 80.88 | 18.97 | 77.59 | 68.42 |
| Language model | Denoising | P | M | 74.70 | 17.93 | 61.14 | 55.02 |
| Prefix LM | Denoising | P | M | 81.82 | 18.61 | 78.94 | 68.11 |
| Encoder-decoder | LM | 2P | M | 79.56 | 18.59 | 76.02 | 64.29 |

**Conclusiones**:

- **Encoder-decoder con denoising gana** en todas las tareas.
- **Compartir parámetros** entre encoder y decoder (mismos pesos) solo pierde marginalmente, lo que sugiere que la cantidad de parámetros importa menos que la **estructura computacional**.
- **Denoising > LM causal** consistentemente en tareas de comprensión (GLUE, SGLUE, SQuAD).
- El **prefix-LM** queda en un punto intermedio interesante: no llega a encoder-decoder, pero supera al decoder-only puro.

## Estrategias de entrenamiento: multi-task y fine-tuning

### Multi-task sampling

Cuando se entrena un solo modelo sobre múltiples tareas simultáneamente, ¿cómo elegir la proporción de ejemplos de cada tarea? El paper explora tres estrategias (Sección 3.5):

1. **Examples-proportional**: probabilidad de muestreo de la tarea $m$:
   $$
   r_m = \frac{\min(e_m, K)}{\sum_n \min(e_n, K)}
   $$
   donde $e_m$ es el tamaño del dataset y $K$ un *cap* artificial. Sin el cap, el unsupervised objective (cuyo dataset es órdenes de magnitud más grande) dominaría el batch.
2. **Temperature-scaled**: las tasas $r_m$ se elevan a $1/T$ y se renormalizan. Cuando $T = 1$ es proportional; cuando $T \to \infty$ tiende a equal mixing. Inspirado en mBERT.
3. **Equal mixing**: cada tarea con peso uniforme.

**Resultados** (Tabla 11): la estrategia equal sufre notablemente (GLUE 76.13 vs baseline 83.28). Examples-proportional con $K = 2^{19}$ logra GLUE 81.42 — todavía por debajo del baseline pre-train + fine-tune (83.28). Multi-task puro **no iguala** a pre-train + fine-tune.

### Pre-train + multi-task + fine-tune

La estrategia ganadora (Sección 3.5.3 y Tabla 12): **multi-task pre-training** (mezcla de objetivo no supervisado + todas las tareas supervisadas) seguido de **fine-tuning** en cada tarea. Resultados comparables al baseline puro (GLUE 83.11 vs 83.28), con la ventaja práctica de poder monitorear desempeño downstream durante el pre-training.

Curiosamente, el experimento **leave-one-out** (pre-entrenar sobre todas las tareas excepto la objetivo, luego fine-tunear sobre esta última) solo pierde marginalmente, lo que sugiere que la *task interference* en multi-task pre-training no es destructiva.

### Métodos alternativos de fine-tuning

Tabla 10 compara:

- **All parameters** (baseline T5): mejor desempeño general.
- **Adapter layers** (Houlsby et al., 2019) con dimensión interna $d \in \{32, 128, 512, 2048\}$: degradación moderada; adapters más anchos ayudan a tareas grandes (GLUE).
- **Gradual unfreezing** (Howard & Ruder, 2018): degradación pequeña pero gana en velocidad.

La conclusión es pragmática: actualizar todos los parámetros sigue siendo lo mejor cuando hay recursos.

## Estudio sistemático de scaling

La Sección 3.6 plantea la pregunta operacional clave: **dado 4× más cómputo, ¿cómo gastarlo?**

**Opciones exploradas** (Tabla 13):

| Estrategia | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
|---|---|---|---|---|---|---|---|
| Baseline | 83.28 | 19.24 | 80.88 | 71.36 | 26.98 | 39.82 | 27.65 |
| 1× size, 4× steps | 85.33 | 19.33 | 82.45 | 74.72 | 27.08 | 40.66 | 27.93 |
| 1× size, 4× batch | 84.60 | 19.42 | 82.52 | 74.64 | 27.07 | 40.60 | 27.84 |
| **2× size, 2× steps** | **86.18** | **19.66** | **84.18** | **77.18** | **27.52** | **41.03** | **28.19** |
| **4× size, 1× steps** | **85.91** | 19.73 | **83.86** | **78.04** | 27.47 | 40.71 | 28.10 |
| 4× ensemble | 84.77 | **20.10** | 83.09 | 71.74 | **28.05** | 40.53 | **28.57** |

**Insights**:

- Aumentar **tamaño del modelo** da el mayor *bump*.
- **Pasos × 4** vs **batch × 4** son aproximadamente equivalentes.
- **Tamaño y pasos son complementarios**: 2× tamaño × 2× pasos ≈ 4× tamaño × 1× pasos.
- **Ensembling** de 4 modelos independientes gana en tareas generativas (CNN/DM, traducción) pero no en clasificación.

La "**bitter lesson**" de Sutton (2019) se confirma: el escalado vence sistemáticamente a los trucos algorítmicos.

## Resultados finales: state-of-the-art

La Sección 3.7 combina todos los hallazgos:

- **Objetivo**: span corruption (15%, span length 3).
- **Entrenamiento**: 1 millón de pasos × batch 2048 secuencias × 512 tokens = **~1 trillón de tokens** pre-training (32× más que el baseline original).
- **Multi-task pre-training**: examples-proportional con caps específicos.
- **Tamaños**: Small (60M), Base (220M), Large (770M), 3B (2.8B), **11B**.
- **Fine-tuning individual** por tarea (no concatenado).
- **Beam search** (width 4, length penalty $\alpha = 0.6$) para tareas con outputs largos (WMT, CNN/DM).

**Tabla 14 — extracto de resultados clave**:

| Modelo | GLUE | SuperGLUE | SQuAD EM | SQuAD F1 | WMT EnDe | WMT EnFr | WMT EnRo | CNN/DM R-1 | CNN/DM R-2 | CNN/DM R-L |
|---|---|---|---|---|---|---|---|---|---|---|
| Previous best | 89.4 | 84.6 | 90.1 | 95.5 | 33.8 | 43.8 | 38.5 | 43.47 | 20.30 | 40.63 |
| T5-Small (60M) | 77.4 | 63.3 | 79.10 | 87.24 | 26.7 | 36.0 | 26.8 | 41.12 | 19.56 | 38.35 |
| T5-Base (220M) | 82.7 | 76.2 | 85.44 | 92.08 | 30.9 | 41.2 | 28.0 | 42.05 | 20.34 | 39.40 |
| T5-Large (770M) | 86.4 | 82.3 | 86.66 | 93.79 | 32.0 | 41.5 | 28.1 | 42.50 | 20.68 | 39.75 |
| T5-3B (2.8B) | 88.5 | 86.4 | 88.53 | 94.95 | 31.8 | 42.6 | 28.2 | 42.72 | 21.02 | 39.94 |
| **T5-11B** | **90.3** | **88.9** | **91.26** | **96.22** | 32.1 | 43.4 | 28.1 | **43.52** | **21.55** | **40.69** |

**Highlights**:

- **GLUE 90.3** — supera el estado del arte previo (89.4, ALBERT con ensemble de 6-17 modelos).
- **SuperGLUE 88.9** — gana al estado del arte (84.6) por margen amplio. Casi alcanza performance humano (89.8).
- **SQuAD EM 91.26 / F1 96.22** — gana ALBERT por un punto en EM.
- **CNN/DM ROUGE-1 = 43.52, ROUGE-2 = 21.55, ROUGE-L = 40.69** — *state-of-the-art* en *abstractive summarization* (estos son los números canónicos que se citan en todo el campo).
- **WMT**: NO logra SOTA. Los autores atribuyen esto a que solo pre-entrenan en inglés; los métodos SOTA usan *backtranslation* (Edunov et al., 2018) y datos multi-source.

**Análisis detallado de CNN/DM**: T5-11B mejora considerablemente sobre el estado del arte previo en ROUGE-2 (de 20.30 a 21.55, un +1.25 absoluto). Los autores citan dos cautelas importantes:

1. **ROUGE no es perfecto**: trabajos previos (Paulus et al., 2017) han mostrado que mejorar ROUGE no necesariamente produce resúmenes más coherentes. Los modelos abstractivos entrenados con maximum likelihood tienden a producir resúmenes repetitivos.
2. **CNN/DM es susceptible a métodos extractivos**: Liu (2019) mostró que approaches puramente extractivos pueden competir con abstractivos en este benchmark. Esto sugiere que CNN/DM no captura completamente el desafío de la *abstractive summarization*; podría haber un componente de "data leakage" donde oraciones del artículo aparecen literalmente en el *gold summary*.

A pesar de estas cautelas, los autores reportan que **los resúmenes generados por T5 son coherentes y correctos** (Appendix C del paper, no incluido en la lectura pero referenciado). El modelo demuestra capacidad de paráfrasis, compresión y selección de información relevante.

**Análisis de SuperGLUE**: SuperGLUE fue diseñado específicamente para ser "más allá del estado del arte actual pero resoluble por hablantes nativos de inglés con educación universitaria". El humano referencia es 89.8. T5-11B alcanza 88.9 — virtualmente igualando humanos en promedio. En tareas de *reading comprehension* (MultiRC, ReCoRD) **supera el desempeño humano**, lo que los autores sugieren puede indicar **sesgo de las métricas a favor de salidas tipo-máquina**. Por otro lado, en COPA y WSC los humanos alcanzan 100% y T5-11B se queda atrás (94.8 y 93.8 respectivamente), lo que confirma que tareas de *common sense* con datasets pequeños siguen siendo difíciles.

**Tabla 14 — métricas adicionales no destacadas previamente**:

- **CoLA Matthew's correlation**: 71.6 (T5-11B) vs 69.2 previous best.
- **SST-2**: 97.5 vs 97.1.
- **MNLI-matched**: 92.2 vs 91.3.
- **STS-B Pearson**: 93.1 vs 92.7.
- **BoolQ**: 91.2 vs 87.1.

El patrón general es de **mejoras significativas pero moderadas** en clasificación (de 1 a 4 puntos) y mejoras más grandes en NLU compleja (SuperGLUE, RTE: de 88.2 a 92.5).

T5-11B logra SOTA en **18 de las 24 tareas** evaluadas. Las 6 tareas perdidas son principalmente traducciones (limitadas por el pre-training monolingüe) y *low-resource* en *common sense* (COPA, WSC), donde humanos alcanzan 100%.

**Tabla 15** desambigua el aporte del *scaling* vs el *non-scaling*: comparando baseline (220M, 34B tokens), baseline-1T (220M, 1T tokens) y T5-Base (220M, 1T tokens + span corruption + multi-task pre-training), se observa que la mayor parte del *bump* en T5-Base viene de los cambios de receta, no solo de más cómputo. Esto valida los hallazgos del estudio sistemático.

## Limitaciones reconocibles

A pesar del éxito, T5 tiene limitaciones que los autores explícitamente reconocen en la Sección 4.2 (Outlook):

1. **Pre-training English-only**: limita performance en traducción multilingüe. La extensión natural es mT5 (Xue et al., 2021) entrenado sobre mC4 (101 idiomas).
2. **Costo de inferencia de 11B**: el modelo más grande es impráctico para deployment en producción. Los autores abogan por distilación (Hinton et al., 2015), *parameter sharing* (ALBERT-style) y *conditional computation* (mixture-of-experts).
3. **Eficiencia del span corruption**: aprender denoising de 1 trillón de tokens es computacionalmente caro. Trabajos posteriores como ELECTRA (Clark et al., 2020) demuestran objetivos más eficientes.
4. **Falta de capacidades zero-shot/few-shot al estilo GPT-3**: T5-11B requiere *fine-tuning* por tarea, mientras que GPT-3 175B (Brown et al., 2020, publicado meses después de T5) demostró que prompting in-context puede sustituir el fine-tuning. T5 es fundamentalmente un modelo *task-specialized* via fine-tuning, no un *general-purpose few-shot learner*.
5. **Prefijos rígidos**: `"summarize:"`, `"cola sentence:"`, etc. son tokens fijos. Trabajos posteriores (FLAN, T0) introducen *natural language instructions* mucho más flexibles.
6. **No formaliza similitud entre tareas**: el éxito del pre-training in-domain (Wikipedia + TBC mejora MultiRC) sugiere que entender qué dominio elegir es importante, pero el paper no provee una noción formal.

## Legado

T5 sembró una progenie extensa:

- **mT5** (Xue et al., 2021): T5 multilingüe entrenado sobre mC4 (101 idiomas). Aplicable a tareas multilingües, traducción y *cross-lingual transfer*. La Slide 41 de la clase 22 del curso menciona explícitamente que mT5 (y UMT5) **requieren fine-tuning supervisado** porque solo se pre-entrenaron con el span corruption no supervisado, sin multi-task pre-training. Esto los distingue de T5 monolingüe, que puede usarse directamente con prefijos.
- **UMT5** (Chung et al., 2023): versión universal multilingüe con mejoras en *tokenization* y *task generalization*.
- **ByT5** (Xue et al., 2022): variante *byte-level*, sin tokenización; útil para *low-resource languages* y robustez a errores ortográficos.
- **FLAN-T5** (Wei et al., 2022; Chung et al., 2022, "Scaling Instruction-Finetuned Language Models"): T5 con *instruction tuning* sobre cientos de tareas con prompts en lenguaje natural. Convierte T5 de modelo *task-specialized* a *general-purpose instruction follower*.
- **T0** (Sanh et al., 2022): T5 entrenado con *prompted multi-task supervised training* explícitamente para *zero-shot generalization*.
- **PEGASUS** (Zhang et al., 2020): inspirado en el span corruption de T5 pero con un objetivo específico para *summarization* (*gap sentence generation*).
- **BART** (Lewis et al., 2020): contemporáneo de T5, también encoder-decoder con denoising, pero con corrupciones más variadas (token masking, deletion, infilling, sentence permutation, document rotation).
- **Pix2Seq** (Chen et al., 2022): aplica el paradigma text-to-text a visión computacional, formulando *object detection* como secuencia.
- **CodeT5** (Wang et al., 2021): T5 adaptado a programación, con vocabulario que incluye tokens de código.

El **C4 dataset** se convirtió en estándar de facto y sigue siendo usado en modelos posteriores (incluyendo derivados como el dataset RedPajama). El reproducible pipeline de filtrado de Common Crawl que T5 inauguró marcó el camino para datasets como The Pile, RefinedWeb, FineWeb.

## Conexión con la clase 22 (Summarization)

La clase 22 del curso IA UC, dictada por Felipe del Río R., distingue entre dos paradigmas de *summarization*:

- **Extractive**: seleccionar oraciones del documento original (TextRank, LexRank, BertSumExt).
- **Abstractive**: generar un resumen nuevo, posiblemente con palabras no presentes en el original.

**T5 es el paradigma abstractive del curso**. El ejemplo canónico que aparece en el material de la clase (y en la Figure 1 del paper) muestra:

```
Input : "summarize: state authorities dispatched emergency crews tuesday to
         survey the damage after an onslaught of severe weather in mississippi…"
Output: "six people hospitalized after a storm in attala county."
```

Los números ROUGE de T5-11B en CNN/DM (R-1 = 43.52, R-2 = 21.55, R-L = 40.69) son los que típicamente se citan como baseline moderno para *abstractive summarization* y son punto de comparación obligatorio. PEGASUS y BART también compiten en este *benchmark*; T5 ofrece la ventaja del *framework unificado*: el mismo modelo que resume CNN/DM puede traducir EN→DE y resolver SQuAD sin cambiar arquitectura.

Para la práctica del curso, T5 es notable porque:

1. **Está disponible en Hugging Face** (`t5-small`, `t5-base`, `t5-large`, `t5-3b`, `t5-11b`, plus `flan-t5-*`), permitiendo *fine-tuning* en datasets propios con relativa facilidad.
2. **El prefijo `summarize:`** funciona out-of-the-box para resumen sin entrenamiento adicional (aunque mejor con fine-tuning en dominio específico).
3. **mT5 y FLAN-T5** permiten resumen multilingüe y zero-shot summarization con instrucciones en lenguaje natural, respectivamente.

**Detalles de inferencia para summarization con T5**:

- Decoding: **beam search** con beam width 4 y *length penalty* $\alpha = 0.6$ es lo recomendado por los autores.
- *Max length* del input: 512 tokens (limitación del *positional encoding* relativo, que solo discrimina hasta offset 128 y comparte los embeddings para offsets mayores). Para documentos más largos hay que truncar o usar *chunk-based summarization*.
- *Max length* del output: típicamente 100–200 tokens en CNN/DM. Forzar longitud mínima evita resúmenes triviales de una palabra.
- Trade-off con *length penalty*: $\alpha > 0$ favorece outputs más largos; $\alpha < 1$ comprime; $\alpha = 0.6$ es un punto intermedio comúnmente usado en MT y summarization.

**Comparación cualitativa T5 vs métodos extractivos**:

- TextRank/LexRank: rápidos, sin training, pero limitados a copiar oraciones. No pueden generar paráfrasis ni comprimir información de múltiples oraciones.
- BertSumExt (Liu, 2019): clasifica oraciones del input como "incluir/excluir". Mejor que TextRank pero sigue siendo extractivo.
- T5 (abstractive): genera resumen palabra por palabra, puede parafrasear, comprimir y reorganizar. A cambio, riesgo de *hallucination* (introducir hechos no presentes en el input).
- PEGASUS: similar a T5 en arquitectura encoder-decoder pero con objetivo de pre-training específicamente diseñado para summarization (*gap sentence generation*: enmascarar oraciones completas durante pre-training). Tiende a superar a T5 en CNN/DM cuando se controla por tamaño.

## Observaciones de ingeniería y reproducibilidad

Algunos detalles del paper que merecen atención por su relevancia práctica:

1. **Open source**: los autores liberan código (`google-research/text-to-text-transfer-transformer`), pesos pre-entrenados de todas las variantes y el dataset C4 a través de TensorFlow Datasets. Esta apertura es fundamental para que T5 se convierta en *workhorse* del campo.
2. **Implementación en Mesh TensorFlow** (Shazeer et al., 2018): paraleliza el modelo a través de un *mesh* lógico de TPUs, permitiendo entrenar T5-11B con paralelismo de tensores y de datos combinados. Esta tecnología luego inspirará approaches como GSPMD y PaLM.
3. **Reportes de varianza inter-run** (Tabla 1): el paper explícitamente entrena 10 baselines desde inicializaciones distintas para reportar desviación estándar. Esta práctica de **honestidad metodológica** es relativamente rara en deep learning y merece aplaudirse.
4. **No-cherry-picking en summaries**: el Appendix C provee ejemplos de resúmenes generados por T5 explícitamente no seleccionados; esta práctica permite evaluar capacidades cualitativas sin sesgo del autor.
5. **Reportes en validation set** vs test set: la mayoría de las tablas reportan en validation (Tabla 1 y Tablas 2-13) para evitar *test set leakage* durante el estudio. Solo la Tabla 14 final (resultados SOTA) reporta en test. Esta separación es buena práctica científica.

## Tabla resumen de hiperparámetros y receta final

| Componente | Valor recomendado | Justificación |
|---|---|---|
| Arquitectura | Encoder-decoder | Mejor que decoder-only y prefix-LM en todas las tareas (Tabla 2) |
| Objetivo | Span corruption | Eficiente y comparable a otros denoisings (Tablas 4-7) |
| Corruption rate | 15% | Sweet spot (Tabla 6) |
| Avg span length | 3 | Mejor que 2, 5, 10 (Tabla 7) |
| Dataset | C4 (filtrado) | Grande y diverso (Tabla 8) |
| Pre-train tokens | ~1 T (1 millón pasos × 2¹¹ × 512) | Más es mejor (Tabla 13) |
| Multi-task | Pre-train + fine-tune | Comparable a baseline, monitorea downstream (Tabla 12) |
| Mixing | Examples-proportional con cap | Mejor que equal o temperature (Tabla 11) |
| Optimizer pre-train | AdaFactor + inverse sqrt LR | Memory-efficient (Sec 3.1) |
| Optimizer fine-tune | AdaFactor + LR constante 0.001 | Estable (Sec 3.1) |
| Decoder | Beam search w=4, α=0.6 | Para tareas de output largo (Sec 3.7) |
| Dropout | 0.1 | Estándar (Sec 3.1) |
| Vocab | SentencePiece 32K | Multi-idioma para WMT (Sec 3.1.3) |
| Position | Relative position bias compartido | Más eficiente y mejor que absoluto (Sec 2.1) |

## Referencias clave citadas en el paper

- **Transformer** (Vaswani et al., 2017) — arquitectura base.
- **BERT** (Devlin et al., 2018) — comparador principal, fuente del MLM denoising.
- **GPT-2** (Radford et al., 2019) — *zero-shot* via prompting, inspiración para text-to-text.
- **SpanBERT** (Joshi et al., 2019) — *span masking*, antecedente del span corruption de T5.
- **MASS** (Song et al., 2019) — *masked sequence-to-sequence pre-training*.
- **XLNet** (Yang et al., 2019) — *permutation language modeling*, comparador.
- **RoBERTa** (Liu et al., 2019c) — escalado del pre-training de BERT.
- **ALBERT** (Lan et al., 2019) — *parameter sharing*, comparador en GLUE/SQuAD.
- **AdaFactor** (Shazeer & Stern, 2018) — optimizador usado.
- **SentencePiece** (Kudo & Richardson, 2018) — tokenización.
- **Mesh TensorFlow** (Shazeer et al., 2018) — paralelismo de modelo en TPU.
- **CNN/Daily Mail** (Hermann et al., 2015; See et al., 2017) — dataset de summarization.
- **GLUE / SuperGLUE** (Wang et al., 2018; 2019b) — benchmarks de NLU.
- **SQuAD** (Rajpurkar et al., 2016) — QA extractivo (reformulado generativo en T5).
- **Common Crawl** — fuente bruta de C4.
- **Bitter Lesson** (Sutton, 2019) — justificación filosófica para el escalado.

T5 cierra una era. Antes de T5 el campo discutía qué arquitectura y qué objetivo era mejor; después de T5 (y especialmente después de GPT-3 en mid-2020) la discusión se desplaza al escalado, al *instruction-tuning* y al RLHF. La elegancia del *framework text-to-text* es que **sigue vigente** — los actuales sistemas instruccionales (FLAN-T5, instruction-tuned LLaMA, ChatGPT) son herederos directos del paradigma "todo es texto a texto" que T5 articuló con tanta claridad.
