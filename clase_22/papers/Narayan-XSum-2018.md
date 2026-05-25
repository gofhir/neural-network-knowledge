---
title: "XSum — Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization"
slug: xsum-narayan-2018
authors:
  - Shashi Narayan
  - Shay B. Cohen
  - Mirella Lapata
year: 2018
venue: "Proceedings of EMNLP 2018, Brussels, Belgium"
arxiv: "1808.08745"
url: "https://arxiv.org/abs/1808.08745"
pdf: "Narayan-XSum-2018.pdf"
tags:
  - nlp
  - summarization
  - abstractive-summarization
  - extreme-summarization
  - dataset
  - bbc
  - convolutional-seq2seq
  - topic-modeling
  - lda
  - rouge
clase: 22
order: 6
---

# XSum — Don't Give Me the Details, Just the Summary!

**Topic-Aware Convolutional Neural Networks for Extreme Summarization** — Shashi Narayan, Shay B. Cohen y Mirella Lapata (University of Edinburgh, EMNLP 2018).

## Resumen ejecutivo

El paper introduce **XSum** (eXtreme Summarization), un dataset y una tarea de single-document summarization deliberadamente diseñada para que las estrategias extractivas no funcionen. Cada uno de los **226.711 artículos** de la BBC viene acompañado de un **resumen profesional de exactamente una oración** redactado por periodistas, que responde a la pregunta "¿de qué trata este artículo?". A diferencia de CNN/DailyMail (CNN/DM) o NY Times, donde las primeras oraciones del documento ya constituyen un buen resumen y un baseline LEAD-3 alcanza ROUGE-1 ≈ 40, en XSum el baseline LEAD-1 sólo alcanza ROUGE-1 = 16.30 y el oracle extractivo (EXT-ORACLE) llega apenas a 29.79. La razón es estructural: en XSum el 35.76% de los unigramas y el 83.45% de los bigramas del resumen son **novel** (no aparecen en el documento), confirmando que la tarea exige paráfrasis, fusión, abstracción e inferencia, no extracción.

El segundo aporte es **T-ConvS2S** (Topic-Aware Convolutional Sequence-to-Sequence), una extensión del ConvS2S de Gehring et al. (2017) que enriquece tanto el encoder como el decoder con un vector de distribución de tópicos obtenido por LDA (512 tópicos). T-ConvS2S supera a todos los baselines, incluyendo el pointer-generator de See et al. (2017) y el EXT-ORACLE, alcanzando ROUGE-1 = 31.89, ROUGE-2 = 11.54 y ROUGE-L = 25.75. Más importante aún: la evaluación humana basada en *Best-Worst Scaling* y en *Question Answering* coloca a T-ConvS2S como el sistema neural mejor rankeado, con sólo el gold humano superándolo de forma estadísticamente significativa.

El impacto del paper es doble. Por un lado, XSum se convierte en el benchmark canónico para evaluar capacidad abstractiva real, desplazando el rol que CNN/DM había monopolizado. Por otro, T-ConvS2S documenta que modelos puramente convolucionales (sin RNNs ni atención global) pueden modelar dependencias largas y abstracción, anticipando el dominio posterior de arquitecturas no-recurrentes (Transformer, BART, PEGASUS, T5) en la tarea.

## Contexto y motivación

### 2018: el callejón sin salida de CNN/DailyMail

Hacia mediados de 2018 el campo de summarization neuronal vivía una paradoja incómoda. Desde Rush et al. (2015), Nallapati et al. (2016) y See et al. (2017), una sucesión de arquitecturas encoder-decoder con atención y mecanismos de copia (pointer-generator) había sido evaluada principalmente en CNN/DailyMail (Hermann et al., 2015) y, secundariamente, en NY Times (Sandhaus, 2008). El consenso reportado en papers era que los modelos abstractivos "iban superando" a los extractivos. Pero el ranking ocultaba un hecho incómodo: el baseline **LEAD-3** —tomar las tres primeras oraciones del artículo— alcanzaba ROUGE-1 ≈ 40.68 en DailyMail y ≈ 29.15 en CNN, niveles cercanos o superiores a los modelos abstractivos del estado del arte. Es decir, los modelos abstractivos en CNN/DM no eran realmente abstractivos: aprendían a comportarse como extractores sofisticados.

El paper aporta evidencia cuantitativa de esta sospecha en la **Tabla 2**: el porcentaje de bigramas novedosos en los resúmenes de referencia de CNN es 54.33% y de DailyMail 53.78%, pero los modelos neurales entrenados sobre esos datasets (See et al., 2017) generan menos del 5% de bigramas novedosos. El gold es relativamente abstractivo, pero los modelos colapsan a copiar oraciones del lead.

### El insight: si el benchmark no penaliza, el modelo no aprende

Narayan, Cohen y Lapata articulan el problema como un *task design*: si la tarea de evaluación recompensa la extracción, ningún modelo aprenderá a abstractar. El campo necesita un dataset donde:

1. La sola posición no resuelva la tarea (LEAD-k debe ser malo).
2. La selección de oraciones tampoco (EXT-ORACLE debe estar lejos del techo).
3. El gold humano debe contener información reformulada, no copiada.
4. La tarea debe ser **single-sentence**, eliminando la posibilidad de "concatenar oraciones importantes".

### El paradigma de los headlines abstractivos

La inspiración no llega del aire: Rush et al. (2015) ya habían propuesto generar headlines en estilo Gigaword, pero los headlines son demasiado cortos (≈ 8 tokens) y telegráficos. Lo que XSum captura es un *sweet spot*: la oración introductoria de un artículo BBC —marcada en HTML como `<storybody_introduction>`— escrita por el autor profesional del artículo. Esta oración no es un titular sensacionalista (cuyo objetivo es atraer clics) sino un *abstract* compacto. Tiene en promedio 23.26 palabras y una sola oración (versus 3.59 oraciones en CNN, 3.86 en DailyMail).

## Diseño del dataset XSum

### Fuente y recolección

Los autores recolectan **226.711 artículos** del archivo Wayback Machine de BBC.co.uk cubriendo aproximadamente una década (2010–2017). La identificación del resumen es trivial gracias a una convención editorial de la BBC: cada artículo comienza con una oración introductoria con clase HTML `story-body__introduction`, escrita profesionalmente, típicamente por el autor del artículo. Esa oración se separa del cuerpo y se usa como **target**; el resto del artículo es el **source**.

El método es elegante por simplicidad: no requiere alineamiento heurístico (como en CNN/DM, donde los highlights fueron usados como resúmenes), no exige anotación manual, y aprovecha una práctica editorial consistente.

### Cobertura temática

La cobertura es deliberadamente amplia: News, Politics, Sports, Weather, Business, Technology, Science, Health, Family, Education, Entertainment y Arts. Esta diversidad temática es importante porque permite entrenar modelos generalistas y evita el sesgo de dominio que aqueja a benchmarks acotados a un solo género (por ejemplo, papers científicos en arXiv o transcripciones legales).

### Splits

Los autores dividen los datos aleatoriamente sobre identificadores únicos de URL, obteniendo:

- **Train**: 204.045 documentos (90%).
- **Validation**: 11.332 documentos (5%).
- **Test**: 11.334 documentos (5%).

El split por URL evita data leakage (no hay overlapping de versiones del mismo artículo entre splits).

## Análisis estadístico

### Longitud

La Tabla 1 del paper compara XSum con tres benchmarks contemporáneos:

| Dataset    | # docs (train/val/test)        | doc len (words) | summary len (words) | summary len (sentences) | vocab doc |
| ---------- | ------------------------------ | --------------- | ------------------- | ----------------------- | --------- |
| CNN        | 90.266 / 1.220 / 1.093         | 760.50          | 45.70               | 3.59                    | 343.516   |
| DailyMail  | 196.961 / 12.148 / 10.397      | 653.33          | 54.65               | 3.86                    | 563.663   |
| NY Times   | 589.284 / 32.736 / 32.739      | 800.04          | 45.54               | 2.44                    | 1.399.358 |
| **XSum**   | **204.045 / 11.332 / 11.334**  | **431.07**      | **23.26**           | **1.00**                | **399.147** |

Lecturas clave:

- **Compresión extrema**: el ratio source/target en XSum es ≈ 18.5× (431 / 23), comparable a CNN (≈ 16.6×) pero el target es **mucho más corto en términos absolutos**.
- **Una sola oración** (sentences = 1.00 por diseño) — esto es lo que da el nombre "extreme summarization".
- El documento promedio es **más corto** (431 palabras) que los demás benchmarks, lo cual no es accidental: artículos BBC tienden a ser más concisos que reportajes largos de NY Times.
- El vocabulario es comparable al de CNN/DM, asegurando suficiente diversidad léxica.

### Novelty: el indicador crítico

La métrica más reveladora del paper es el porcentaje de **n-gramas novedosos** en el gold summary, es decir, n-gramas que aparecen en el resumen pero no en el documento (Tabla 2):

| Dataset    | unigramas | bigramas  | trigramas | 4-gramas  |
| ---------- | --------- | --------- | --------- | --------- |
| CNN        | 16.75     | 54.33     | 72.42     | 80.37     |
| DailyMail  | 17.03     | 53.78     | 72.14     | 80.28     |
| NY Times   | 22.64     | 55.59     | 71.93     | 80.16     |
| **XSum**   | **35.76** | **83.45** | **95.50** | **98.49** |

XSum **duplica** el porcentaje de unigramas novedosos respecto de CNN/DM y supera por amplio margen el porcentaje de bigramas (83.45% vs 53–55%). Para 4-gramas, 98.49% son novedosos: prácticamente cualquier secuencia de 4 palabras del resumen es una construcción nueva no presente en el artículo. Esto es la definición operacional de "summarization verdaderamente abstractiva".

### Baselines extractivos: la prueba decisiva

La misma Tabla 2 reporta el rendimiento del baseline LEAD (primera oración para XSum, primeras 3 para CNN, primeras 4 para DailyMail, primeras 100 palabras para NY Times) y del EXT-ORACLE (la única oración del documento que maximiza ROUGE contra el gold):

| Dataset    | LEAD R1 | LEAD R2 | LEAD RL | EXT-ORACLE R1 | EXT-ORACLE R2 | EXT-ORACLE RL |
| ---------- | ------- | ------- | ------- | ------------- | ------------- | ------------- |
| CNN        | 29.15   | 11.13   | 25.95   | 50.38         | 28.55         | 46.58         |
| DailyMail  | 40.68   | 18.36   | 37.25   | 55.12         | 30.55         | 51.24         |
| NY Times   | 31.85   | 15.86   | 23.75   | 52.08         | 31.59         | 46.72         |
| **XSum**   | **16.30** | **1.61** | **11.95** | **29.79**     | **8.81**      | **22.65**     |

Interpretación:

- **CNN/DM/NYT**: LEAD ya alcanza ROUGE-1 entre 29 y 41. EXT-ORACLE entre 50 y 55. El techo extractivo es muy alto.
- **XSum**: LEAD apenas alcanza 16.30 — un baseline muy bajo. Y crucialmente, el oracle extractivo (la mejor oración posible del documento) sólo llega a 29.79. **El techo extractivo en XSum es comparable al LEAD de CNN/DM**. Esto significa que cualquier modelo que supere ROUGE-1 ≈ 30 en XSum necesariamente está haciendo abstracción real.

## Topic-Aware ConvS2S (T-ConvS2S): el modelo del paper

### Por qué convolucional y no recurrente

Los autores eligen extender el ConvS2S de Gehring et al. (2017) en lugar de basarse en RNNs (la práctica común en 2018 vía See et al. y Nallapati et al.) por dos razones:

1. **Captura jerárquica de dependencias largas**: las CNNs apiladas construyen un campo receptivo que crece geométricamente. Para documentos de 400 tokens (el máximo usado en el paper), una pila de 10–15 capas con kernel 3 alcanza cualquier posición. Las RNNs requieren propagar por toda la secuencia, lo que las hace vulnerables a vanishing gradients.
2. **Paralelización**: ConvS2S se entrena mucho más rápido que LSTMs/GRUs, una ventaja práctica para datasets de 200k ejemplos.

(Este argumento será absorbido y superado por Transformers a los pocos años, pero en 2018 era una posición razonable y competitiva.)

### Arquitectura ConvS2S base

El backbone es el de Gehring et al. (2017b):

- **Encoder convolucional**: capas de convolución 1-D con GLU (Gated Linear Units, Dauphin et al. 2017) y residual connections (He et al. 2016).
- **Decoder convolucional**: también con GLU, autoregresivo (mask futuros), con multi-hop attention sobre cada capa del encoder.
- **Multi-hop attention**: a diferencia del Bahdanau attention que opera una sola vez, ConvS2S aplica atención en cada capa del decoder, permitiendo "recordar" qué se atendió previamente.

Para cada palabra del decoder se computa:

$$a^{\ell}_{ij} = \frac{\exp(d^{\ell}_i \cdot z^u_j)}{\sum_{t=1}^{m} \exp(d^{\ell}_i \cdot z^u_t)}$$

donde $d^{\ell}_i = W^{\ell}_d h^{\ell}_i + b^{\ell}_d + g_i$ combina el estado del decoder con el embedding del token previo $g_i$. El contexto se obtiene como:

$$c^{\ell}_i = \sum_{j=1}^{m} a^{\ell}_{ij} (z^u_j + e_j)$$

donde la suma incluye no sólo la salida del encoder sino también el embedding de entrada $e_j$, sumando información posicional refinada.

### Topic conditioning: la contribución arquitectónica

Sobre esta base, los autores agregan **condicionamiento por tópicos**. La hipótesis es: para generar un resumen coherente, el modelo necesita saber **de qué trata globalmente el documento**, no sólo qué palabras hay localmente. La atención multi-hop captura dependencias léxicas pero no necesariamente "tema". LDA (Blei et al. 2003) ofrece exactamente eso: una distribución global sobre tópicos latentes.

#### Topic features

Sean:

- $t_D \in \mathbb{R}^{f'}$ la distribución de tópicos del **documento completo** $D$, calculada por LDA. Vector global.
- $t'_i \in \mathbb{R}^{f'}$ la distribución de tópicos de la **palabra individual** $w_i$ en el documento.

Ambos provienen de un modelo LDA entrenado sobre la porción de training de XSum con **512 tópicos** (los autores exploraron varias configuraciones en validación). La Tabla 3 muestra ejemplos de tópicos aprendidos:

- T1: charge, court, murder, police, arrest, guilty, sentence, boy, bail, space, crown, trial → tópico judicial/criminal.
- T2: church, abuse, bishop, child, catholic, gay, pope, school, christian, priest, cardinal → tópico religioso/escándalos eclesiásticos.
- T4: clinton, party, trump, climate, poll, vote, plaid, election, debate, change, candidate, campaign → tópico político electoral.
- T6: hospital, patient, trust, nhs, people, care, health, service, staff, report, review, system, child → tópico sanitario.

Estos tópicos no son entrenados conjuntamente con la red: LDA pasa por una vez sobre el corpus, y los vectores resultantes se inyectan como input adicional. Esta separación tiene la ventaja de mantener la arquitectura convolucional eficiente sin necesidad de modelar temas conjuntamente.

#### Inyección en el encoder

Cada token de entrada se representa como:

$$e_i = [(x_i + p_i); (t'_i \otimes t_D)] \in \mathbb{R}^{f + f'}$$

donde:

- $x_i$ es el word embedding.
- $p_i$ es el position embedding.
- $t'_i \otimes t_D$ es el producto punto-a-punto entre la distribución de tópicos del token y la del documento (no producto exterior — Hadamard). El símbolo $\otimes$ en el paper denota element-wise.

Esta concatenación enriquece el contexto del token con su **relevancia tópica local-global**: si una palabra tiene alta probabilidad en los mismos tópicos que el documento entero, ese producto es alto; si es léxico genérico (preposiciones, artículos), tiende a cero.

#### Inyección en el decoder

Similarmente, cada token generado se representa como:

$$g_i = [(x'_i + p'_i); t_D] \in \mathbb{R}^{f + f'}$$

donde $t_D$ se broadcast a cada posición del decoder. **Nota crítica**: el decoder no usa $t'$ de los tokens ya generados (sería caro computar LDA on-the-fly), sólo el vector global del documento de entrada. Esto fuerza al resumen a "permanecer en el tema" del documento.

#### Probabilidad de salida

La probabilidad de generar $y_{i+1}$ es:

$$p(y_{i+1} | y_1, \ldots, y_i, D, t_D, t') = \mathrm{softmax}(W_o h^L_i + b_o) \in \mathbb{R}^T$$

con $T$ el tamaño del vocabulario target (50.000 palabras según los autores). La pérdida es cross-entropy autoregresivo estándar.

### Variantes evaluadas

Los autores ablacionan cuatro variantes de T-ConvS2S (Tabla 4):

1. **T-ConvS2S (enc$_{t'}$)**: sólo encoder usa el tópico de palabra.
2. **T-ConvS2S (enc$_{t'}$, dec$_{t_D}$)**: encoder usa $t'$, decoder usa $t_D$.
3. **T-ConvS2S (enc$_{(t', t_D)}$)**: encoder usa ambos.
4. **T-ConvS2S (enc$_{(t', t_D)}$, dec$_{t_D}$)**: ambos en ambos lados. **Variante recomendada y usada en el paper.**

## Setup experimental

### Preprocesamiento

- No se anonimizan entidades nombradas (a diferencia de See et al. 2017 en CNN/DM, donde personas y lugares se reemplazaban por placeholders).
- Documentos se truncan a 400 tokens, resúmenes a 90 tokens.
- Texto en minúsculas (lowercase).
- Vocabulario de 50.000 palabras (cubre training).

### Hyperparámetros

- ConvS2S y T-ConvS2S: 512 hidden states, 512-dim word/position embeddings.
- Nesterov SGD, momentum 0.99, gradient renormalization si la norma excede 0.1 (Pascanu et al. 2013).
- Learning rate inicial 0.10, reducido por orden de magnitud cuando la perplexity de validación deja de mejorar, hasta caer bajo $10^{-4}$.
- Dropout 0.2 en embeddings, decoder outputs y inputs de los bloques convolucionales.
- Batch size: 32 sentences en una sola NVIDIA M40.
- Beam search con beam size 10 en inferencia.

### Sistemas comparados

- **RANDOM**: una oración aleatoria del documento.
- **LEAD**: primera oración del documento (excluyendo la introductory que es el target).
- **EXT-ORACLE**: la oración del documento con mayor ROUGE contra el gold (upper bound extractivo).
- **Seq2Seq**: encoder-decoder RNN con atención (See et al. 2017 settings).
- **PtGen**: pointer-generator (See et al. 2017), permite copiar del source.
- **PtGen-Covg**: PtGen con coverage mechanism.
- **ConvS2S**: vanilla Gehring et al. (2017b) sin topic conditioning.
- **T-ConvS2S**: el modelo propuesto.

Todos los modelos RNN usan los mejores hiperparámetros reportados por See et al. para CNN/DM. Esto es importante: los autores no debilitan los baselines para hacer ver mejor su modelo.

## Resultados cuantitativos

### Evaluación automática (Tabla 4)

| Modelo                                         | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ---------------------------------------------- | ------- | ------- | ------- |
| Random                                         | 15.16   | 1.78    | 11.27   |
| LEAD                                           | 16.30   | 1.60    | 11.95   |
| EXT-ORACLE                                     | 29.79   | 8.81    | 22.66   |
| Seq2Seq                                        | 28.42   | 8.77    | 22.48   |
| PtGen                                          | 29.70   | 9.21    | 23.24   |
| PtGen-Covg                                     | 28.10   | 8.02    | 21.72   |
| ConvS2S                                        | 31.27   | 11.07   | 25.23   |
| T-ConvS2S (enc$_{t'}$)                         | 31.71   | 11.38   | 25.56   |
| T-ConvS2S (enc$_{t'}$, dec$_{t_D}$)            | 31.71   | 11.34   | 25.61   |
| T-ConvS2S (enc$_{(t', t_D)}$)                  | 31.61   | 11.30   | 25.51   |
| **T-ConvS2S (enc$_{(t', t_D)}$, dec$_{t_D}$)** | **31.89** | **11.54** | **25.75** |

Lecturas clave:

1. **PtGen-Covg empeora respecto de PtGen** (28.10 vs 29.70). El coverage mechanism, útil para evitar repetición en resúmenes largos de CNN/DM, es contraproducente para resúmenes de una sola oración. Esto confirma que arquitecturas optimizadas para CNN/DM no transfieren naturalmente.
2. **PtGen supera a EXT-ORACLE** (29.70 vs 29.79 en R1, pero claramente en R2 y RL: 9.21 vs 8.81; 23.24 vs 22.66). Esto valida el dataset: el modelo abstractivo realmente está aportando información nueva que el oracle extractivo no puede ofrecer.
3. **ConvS2S vanilla ya supera a todos los RNN-based** (31.27 vs 29.70 mejor RNN). La arquitectura convolucional es ventajosa para el task.
4. **T-ConvS2S supera consistentemente a ConvS2S**. La ganancia absoluta es modesta (+0.62 R1) pero estadísticamente robusta y se mantiene en R2/RL.
5. La mejor configuración usa tópicos en ambos extremos (encoder $t'$ + $t_D$, decoder $t_D$).

### Análisis de novelty (Tabla 5)

¿Los modelos efectivamente generan n-gramas novedosos?

| Modelo       | unigramas | bigramas | trigramas | 4-gramas |
| ------------ | --------- | -------- | --------- | -------- |
| LEAD         | 0.00      | 0.00     | 0.00      | 0.00     |
| EXT-ORACLE   | 0.00      | 0.00     | 0.00      | 0.00     |
| PtGen        | 27.40     | 73.33    | 90.43     | 96.04    |
| ConvS2S      | 31.26     | 79.50    | 94.28     | 98.10    |
| T-ConvS2S    | 30.73     | 79.18    | 94.10     | 98.03    |
| GOLD         | 35.76     | 83.45    | 95.50     | 98.49    |

Observaciones:

- LEAD y EXT-ORACLE generan 0% novedad por definición (todo viene del documento).
- PtGen entrenado en XSum genera un 27% de unigramas novedosos — muy por encima del 1% que See et al. (2017) reportaban en CNN/DM. **Esto demuestra que PtGen sí puede ser abstractivo cuando el dataset lo exige.**
- ConvS2S y T-ConvS2S casi alcanzan el nivel de novedad del gold humano (31% y 30% vs 35%).
- Las longitudes de los resúmenes generados son comparables: PtGen 22.57 palabras, ConvS2S 20.07, T-ConvS2S 20.22, GOLD 23.26 — el modelo no inflará artificialmente la novedad acortando.

### Evaluación humana (Tabla 7)

Dos estudios crowd-sourced en Amazon Mechanical Turk:

**Best-Worst Scaling** (50 documentos, 3 anotadores por par, todos los pares entre 5 sistemas: EXT-ORACLE, PtGen, ConvS2S, T-ConvS2S, GOLD). Score = % veces best − % veces worst, rango [−1, 1].

| Modelo      | Score   | QA accuracy |
| ----------- | ------- | ----------- |
| EXT-ORACLE  | −0.121  | 15.70       |
| PtGen       | −0.218  | 21.40       |
| ConvS2S     | −0.130  | 30.90       |
| T-ConvS2S   | **+0.037** | **46.05**   |
| GOLD        | +0.431  | 97.23       |

T-ConvS2S es el único sistema con score positivo (más elegido como best que como worst). Diferencia significativa contra ConvS2S y PtGen (ANOVA + Tukey HSD, $p < 0.01$). GOLD significativamente mejor que todos.

**QA-based evaluation**: dos preguntas factuales por documento (escritas leyendo el gold). Anotadores las contestan leyendo sólo el resumen generado. Score = accuracy.

- **T-ConvS2S: 46.05%** — anotadores pueden contestar casi la mitad de las preguntas.
- ConvS2S: 30.90%.
- PtGen: 21.40%.
- EXT-ORACLE: 15.70% — incluso peor que PtGen, porque seleccionar la oración con mayor ROUGE-overlap no garantiza que contenga las entidades clave.
- GOLD: 97.23% — los anotadores recuperan casi toda la información factual del summary humano.

T-ConvS2S es significativamente mejor que ConvS2S y PtGen y comparable a GOLD en algunos casos, mostrando que el topic conditioning ayuda a **seleccionar las entidades correctas**, no sólo a generar texto fluído.

## Análisis cualitativo: ejemplos del paper

La Tabla 6 muestra tres ejemplos representativos donde puede contrastarse la calidad de cada sistema. Reproduzco el más ilustrativo:

**Documento (resumido)**: artículo sobre Zac Goldsmith siendo elegido candidato conservador para alcalde de Londres en 2016.

| Sistema | Resumen generado | ROUGE [1, 2, L] |
| --- | --- | --- |
| EXT-ORACLE | Caroline Pidgeon is the Lib Dem candidate, Sian Berry will contest the election for the Greens and UKIP has chosen its culture spokesman Peter Whittle. | [34.1, 20.5, 34.1] |
| PtGen | UKIP leader Nigel Goldsmith has been elected as the new mayor of London to elect a new conservative MP. | [45.7, 6.1, 28.6] |
| ConvS2S | London mayoral candidate Zac Goldsmith has been elected as the new mayor of London. | [53.2, 21.4, 26.7] |
| T-ConvS2S | Former London mayoral candidate Zac Goldsmith has been chosen to stand in the London mayoral election. | [50.0, 26.7, 37.5] |
| GOLD | Zac Goldsmith will contest the 2016 London mayoral election for the conservatives, it has been announced. | — |

Análisis:

- **PtGen alucina**: "UKIP leader Nigel Goldsmith" es factualmente incorrecto (Goldsmith no era UKIP ni se llama Nigel). Además dice "has been elected as the new mayor" — Goldsmith fue *candidato*, perdió la elección. Dos errores graves.
- **ConvS2S también alucina parcialmente**: dice "has been elected as the new mayor of London" — falso.
- **T-ConvS2S es el único factualmente correcto**: "chosen to stand in the London mayoral election" — correcto, fue elegido para ser candidato.
- **EXT-ORACLE** seleccionó una oración del documento que tiene alta superposición léxica con el gold pero **no responde a la pregunta** "¿quién será candidato conservador?". Habla de los rivales en lugar del protagonista.

Este ejemplo encapsula la crítica del paper a ROUGE como métrica única: ConvS2S obtiene la R1 más alta (53.2) **pero su resumen es factualmente falso**. T-ConvS2S, con R1 ligeramente menor (50.0), es el único que pasaría una evaluación periodística.

## Hallucinations: el problema persistente

El paper documenta —pero no resuelve— el problema más serio de la summarization neuronal abstractiva: **hallucinations** (generación de hechos falsos no soportados por el documento fuente).

Los ejemplos de la Tabla 6 muestran:

- PtGen confunde nombres (Nigel ≠ Zac, UKIP ≠ conservatives).
- ConvS2S confunde resultado (candidato ≠ ganador).
- T-ConvS2S confunde detalles secundarios (en el segundo ejemplo dice "Sunderland manager" cuando debiera decir "Sunderland boss" — error menor, pero existe).

Los autores son honestos: el modelo no tiene mecanismo explícito para verificar que las entidades del resumen aparezcan o sean consistentes con el documento. ROUGE no penaliza alucinaciones siempre que mantengan superposición de palabras.

Este problema motivará una literatura completa post-2018:

- **Cao et al. (2018)**: faithfulness-aware summarization con grafos de hechos.
- **Maynez et al. (2020)**: "On Faithfulness and Factuality in Abstractive Summarization" — el paper canónico sobre hallucinations en XSum, distingue extrínseca vs intrínseca, mostrando que >70% de los resúmenes de BART/PEGASUS en XSum contienen alguna alucinación.
- **FactCC** (Kryscinski et al. 2020), **QAGS** (Wang et al. 2020), **MFMA** (Lee et al. 2022): métricas y entrenamientos específicos para factualidad.

## Por qué XSum cambió el campo

El impacto del paper trasciende sus números. Tres cambios estructurales:

### 1. Redefinición de qué cuenta como "summarization"

Antes de XSum, "abstractive summarization" en la práctica era un término aspiracional aplicado a modelos que copiaban casi todo. XSum hizo evidente —vía métricas de novelty y baselines— que CNN/DM no era un buen benchmark para esa aspiración. Post-XSum, todo paper serio reporta resultados en XSum además de CNN/DM, justamente para mostrar que su modelo no es sólo un extractor disfrazado.

### 2. Catalizador del paradigma generativo pre-entrenado

Los modelos posteriores que dominan XSum no son convolucionales sino Transformers pre-entrenados generativamente:

- **BART** (Lewis et al. 2020): denoising autoencoder pre-entrenado, reporta XSum ROUGE-1 = 45.14, ROUGE-2 = 22.27, ROUGE-L = 37.25.
- **PEGASUS** (Zhang et al. 2020): pre-entrenamiento *Gap Sentences Generation* específicamente diseñado para summarization, alcanza XSum ROUGE-1 = 47.21.
- **T5** (Raffel et al. 2020): text-to-text framework, reporta XSum como una de sus tareas.
- **GPT-3/4** y descendientes: usados zero-shot, alcanzan resultados competitivos pero con alta variabilidad en factualidad.

La existencia de XSum permite comparar estos modelos en una tarea donde la abstracción es necesaria, no opcional.

### 3. Disparador de la agenda de faithfulness

Los hallucinations que el paper identifica en T-ConvS2S se vuelven endémicos en BART y PEGASUS sobre XSum: Maynez et al. (2020) reportan que aprox. 70% de los resúmenes generados por estos modelos contienen alucinaciones extrínsecas. XSum, por su demanda abstractiva, hace casi inevitable que los modelos inventen. Esto motiva toda una sub-área: factuality-aware summarization, hallucination detection, retrieval-augmented summarization.

## Limitaciones reconocibles

A casi una década del paper, podemos enumerar limitaciones que el tiempo y la literatura posterior han clarificado:

### Sesgo de fuente única

XSum es **sólo BBC**. Esto introduce:

- **Estilo periodístico británico** específico (lenguaje, ortografía, registro).
- **Diversidad temática limitada por la línea editorial de la BBC** (sobre-representación de UK politics, sub-representación de eventos en Asia/África/Latinoamérica).
- **Convenciones uniformes** en cómo se redacta la introductory sentence — entrenar en XSum y evaluar en CNN puede no transferir bien.

Trabajos posteriores como **Newsroom** (Grusky et al. 2018) abordan esto agregando 38 fuentes distintas.

### Rigidez del target de una oración

El diseño "exactly one sentence" tiene la ventaja didáctica de evitar ambigüedad, pero:

- Algunos artículos legítimamente necesitan más de una oración para resumirse.
- Modelos optimizados para una oración pueden no generalizar a otros formatos (TL;DR, abstracts).
- Cuando un artículo es muy complejo, una sola oración fuerza alucinación por compresión excesiva.

### ROUGE como métrica

Los autores explícitamente reconocen las limitaciones de ROUGE (Schluter 2017): no detecta hallucinations, premia overlap léxico aunque la oración sea factualmente incorrecta, y en XSum —donde el gold tiene 35% de unigramas novedosos— ROUGE penaliza paráfrasis válidas que usen vocabulario distinto del gold.

La evaluación QA es una mejora pero costosa.

### LDA: una elección de 2018

El uso de **LDA con 512 tópicos** se ve hoy un poco anticuado. En 2026, con LLMs que internalizan representaciones de tópico vía embeddings densos y self-attention, la idea de pre-computar distribuciones de Dirichlet y concatenarlas como features extra es una arquitectura de transición. La idea misma —condicionar la generación en una representación global del documento— sobrevive, pero implementada con encoders Transformer.

### Tamaño del documento truncado a 400 tokens

Los autores truncan los artículos a 400 tokens. Esto significa que para artículos largos (algunos en BBC son mucho más extensos), el modelo nunca ve el final. En 2018, con CNN sobre M40, 400 tokens era un límite computacional razonable; en 2026, modelos con contexto de 32k+ no tienen esa restricción.

### Vocabulario fijo de 50k palabras

OOV (out-of-vocabulary) tokens —nombres de personas, lugares, eventos— son frecuentes en news. Los autores no anonimizan ni usan BPE/SentencePiece (que llegan después). PtGen mitiga parcialmente vía copy mechanism, pero ConvS2S/T-ConvS2S no tienen esa capacidad.

## Sucesores e impacto en el ecosistema

### Otros datasets de summarization post-XSum

XSum estimuló la creación de benchmarks complementarios:

- **Newsroom** (Grusky et al. 2018): 1.3M resúmenes de 38 outlets, con métricas de "estilo" (extractive, abstractive, mixed). Permite estudiar el efecto del estilo editorial.
- **Multi-News** (Fabbri et al. 2019): multi-document summarization (varios artículos sobre un mismo evento → un resumen).
- **Reddit-TIFU** (Kim et al. 2019): user-generated content, informal, "TL;DR" como target.
- **BigPatent** (Sharma et al. 2019): patentes (highly technical, multi-paragraph abstracts).
- **arXiv / PubMed** (Cohan et al. 2018): papers científicos, resúmenes largos.
- **WikiHow** (Koupaee & Wang 2018): how-to articles.

Cada uno explora una dimensión que XSum no cubría: multi-doc, dominio técnico, informalidad, longitud.

### Modelos optimizados para XSum

- **BertSum / BertSumAbs** (Liu & Lapata 2019): primer encoder Transformer pre-entrenado (BERT) sobre summarization.
- **BART** (Lewis et al. 2020): seq2seq Transformer con denoising pre-training, lidera XSum por meses.
- **PEGASUS** (Zhang et al. 2020): pre-training específicamente diseñado para summarization, con objetivo *Gap Sentences Generation*. SOTA en XSum por largo tiempo.
- **T5** (Raffel et al. 2020): unifica summarization con otras tareas en formato text-to-text.
- **LongT5, LED, BigBird**: variantes con atención esparsa para documentos largos.

### Investigación en faithfulness y factualidad

- Maynez et al. (2020), Pagnoni et al. (2021), Goyal & Durrett (2021), Cao et al. (2022).
- Métricas: FactCC, DAE, QAGS, QuestEval, BARTScore, FActScore.

### Integración en HuggingFace y herramientas

El dataset `xsum` está disponible en `datasets` de HuggingFace, con loaders estandarizados. Es uno de los benchmarks por defecto para evaluar cualquier modelo de summarization. Aparece en leaderboards de Papers With Code y en evaluaciones de modelos comerciales (GPT-4, Claude, Gemini).

## Conexión con la clase 22

La clase 22 del curso IA UC presenta **summarization** como una de las tareas paradigmáticas de NLG. El slide 12 enumera los datasets canónicos del campo y XSum aparece explícitamente:

> "X-Sum. Extreme summarization of news into a short one-sentence summary composed of 225.000 examples."

Y el slide 13 formaliza la tarea:

> "X-Sum. article → one-sentence summary"

XSum es uno de los **5 datasets canónicos** que la clase discute: CNN/DM, NY Times, XSum, Newsroom y multi-document benchmarks. Cada uno representa una dimensión distinta del espectro extractive ↔ abstractive.

El rol de XSum en la narrativa de la clase es triple:

1. **Como benchmark abstractivo de referencia**: muestra qué significa exigirle a un modelo summarization "real", no extractive.
2. **Como motivador del pre-training generativo**: los modelos T5, BART y PEGASUS que la clase presenta en slides 33+ están explícitamente diseñados pensando en XSum (o se evalúan principalmente allí). PEGASUS-XSum es un punto de referencia mencionado repetidamente.
3. **Como caso de estudio para hallucinations**: la clase aborda la limitación de ROUGE y la necesidad de métricas factuales. Los ejemplos de hallucinations en XSum son el caso canónico que la literatura usa.

La arquitectura T-ConvS2S no se enseña hoy (es un punto histórico), pero el **dataset** XSum sigue siendo central y prácticamente todo paper de summarization moderno (incluyendo los slides finales de la clase 22 sobre LLM-based summarization) reporta números sobre él.

## Cierre

XSum es uno de esos papers donde el dataset eclipsa al modelo. La contribución arquitectónica —T-ConvS2S con condicionamiento LDA— fue importante en 2018 pero quedó superada en pocos meses por Transformers pre-entrenados. Lo que persiste, una década después, es la decisión metodológica: identificar que los benchmarks existentes recompensaban la extracción y construir uno que la castigara. Ese gesto cambió la trayectoria del campo más que cualquier mejora incremental en ROUGE.

El paper es un ejemplo de cómo el diseño de evaluación es a veces más valioso que el diseño de arquitectura. Si los modelos optimizan para una métrica, y la métrica se calcula contra un benchmark, entonces el benchmark **define implícitamente la tarea**. XSum redefinió la tarea para que summarization significara abstracción genuina —paráfrasis, fusión, inferencia, generación de palabras nuevas— en lugar de "selección sofisticada de la primera oración". Todos los modelos generativos modernos de summarization (BART, PEGASUS, T5, Flan-T5, GPT-4, Claude, Gemini) son evaluados sobre XSum precisamente porque la tarea allí no admite atajos.

El precio de esa exigencia es el problema de las alucinaciones, que XSum hereda como herida abierta y que aún en 2026 sigue siendo área activa de investigación. Pero esa tensión —entre forzar abstracción y mantener factualidad— es la pregunta central de NLG aplicado, y XSum es el dataset donde esa pregunta se hace más visible.

## Referencias clave del paper

- **Gehring et al. (2017a, 2017b)** — Convolutional Sequence to Sequence Learning. Backbone arquitectónico de T-ConvS2S.
- **See et al. (2017)** — Pointer-Generator Networks. Baseline RNN principal, dataset CNN/DM como punto de comparación.
- **Hermann et al. (2015)** — Teaching Machines to Read and Comprehend. Dataset CNN/DM original.
- **Blei et al. (2003)** — Latent Dirichlet Allocation. Modelo de tópicos para topic conditioning.
- **Lin & Hovy (2003)** — ROUGE metric.
- **Sandhaus (2008)** — NY Times Annotated Corpus.
- **Grusky et al. (2018)** — Newsroom dataset, trabajo contemporáneo.
- **Nallapati et al. (2016, 2017)** — RNN-based abstractive summarization, SummaRuNNer.
- **Dauphin et al. (2017)** — Gated Linear Units.
- **He et al. (2016)** — Residual connections.
- **Sukhbaatar et al. (2015)** — End-to-end memory networks (base teórica del multi-hop attention).
- **Schluter (2017)** — Limits of ROUGE evaluation, justifica evaluación humana.

## Datos de identificación

- **Título**: Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization
- **Autores**: Shashi Narayan, Shay B. Cohen, Mirella Lapata
- **Afiliación**: Institute for Language, Cognition and Computation, School of Informatics, University of Edinburgh
- **Venue**: EMNLP 2018, Brussels, Belgium
- **arXiv**: 1808.08745 (27 Aug 2018)
- **Código y dataset**: https://github.com/shashiongithub/XSum
- **HuggingFace**: `datasets.load_dataset("xsum")`
