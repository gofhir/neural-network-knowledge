# Análisis interno — Cui, Che, Liu, Qin, Yang et al. (2019/2021) "Pre-Training with Whole Word Masking for Chinese BERT"

> Documento complementario al material público del site (`papers/`, `fundamentos/`). Aquí se profundiza en el origen, la mecánica formal y el impacto cross-lingual de Whole Word Masking (WWM) — una técnica que el paper introduce casi de pasada para chino pero que terminó adoptada por Google (`bert-large-cased-whole-word-masking`), por BETO (CENIA-UC, español) y por la gran mayoría de modelos BERT-like en idiomas con morfología flexiva. También cubrimos MacBERT, la extensión que el equipo de HIT/iFLYTEK propuso en la versión TASLP 2021 (`MLM-as-correction`), y la conexión directa con el lab 20 del Diplomado IA UC.

- **Paper original**: Cui, Che, Liu, Qin, Yang, Wang, Hu. *Pre-Training with Whole Word Masking for Chinese BERT*. arXiv:1906.08101 (19 Jun 2019, v1).
- **Versión TASLP**: arXiv:1906.08101v3 (25 Nov 2021). Publicada en *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, Nov 2021. Esta versión extiende el v1 con: (a) introducción de **MacBERT**, (b) comparación numérica contra RoBERTa-wwm, ELECTRA-Chinese y RBT, (c) ablations de la estrategia de masking en CMRC 2018 / DRCD / SIGHAN-15.
- **Autores y afiliaciones**: Yiming Cui (HIT + iFLYTEK), Wanxiang Che (HIT), Ting Liu (HIT), Bing Qin (HIT), Ziqing Yang (iFLYTEK). El centro académico es el **Research Center for Social Computing and Information Retrieval** (HIT-SCIR), Harbin; el centro industrial es el **State Key Laboratory of Cognitive Intelligence, iFLYTEK Research**, Beijing/Hebei.
- **Código y checkpoints**: `https://github.com/ymcui/Chinese-BERT-wwm`. Modelos: BERT-wwm, BERT-wwm-ext, RoBERTa-wwm-ext, RoBERTa-wwm-ext-large, MacBERT-base, MacBERT-large, RBT3/4/6/L3. Apache 2.0.
- **PDF local**: `clase_20/papers/Cui-WWM-2019.pdf`.

---

## 1. Contexto histórico: el problema del masking subword en BERT original

Para entender por qué WWM importa hay que situarlo entre **dos olas** de optimización de BERT:

| Mes | Modelo | Innovación sobre masking |
|---|---|---|
| Oct 2018 | **BERT** (Devlin et al.) | MLM con 15% de tokens enmascarados, regla 80/10/10, selección uniformemente aleatoria a nivel **subword** |
| Apr 2019 | **ERNIE 1.0** (Sun et al., Baidu) | Masking a nivel de entidad + frase (en chino) |
| May 2019 | **BERT-wwm en inglés** (Google, no publicación formal) | Google libera `bert-large-cased-whole-word-masking` |
| Jun 2019 | **BERT-wwm en chino** (Cui et al., este paper, v1) | Adapta WWM al chino usando segmentador LTP |
| Jul 2019 | **SpanBERT** (Joshi et al.) | Span-based masking con geometric span lengths |
| Jul 2019 | **RoBERTa** (Liu et al.) | Sin NSP, dynamic masking, batches grandes |
| Mar 2020 | **BETO** (Cañete, CENIA-UC) | Adopta WWM por default para español |
| Nov 2021 | **MacBERT** (Cui et al., TASLP) | `MLM as correction`: reemplazar `[MASK]` por sinónimos |

El BERT original (Devlin et al. 2018) tiene una decisión que en retrospectiva parece arbitraria: el masking se aplica **a nivel de WordPiece**, no a nivel de palabra. Concretamente, el procedimiento es:

1. Tokenizar el input con WordPiece (palabras infrequentes se parten en subwords con prefijo `##`).
2. Sobre la lista de tokens resultante (incluyendo subwords), seleccionar uniformemente al azar el 15% de las posiciones.
3. Aplicar la regla 80/10/10 sobre las posiciones seleccionadas.

Esto significa que para una palabra que se parte en varios subwords, **solo algunos de sus subwords pueden quedar enmascarados, mientras los otros quedan visibles**. Ejemplo del propio paper de Cui et al. (Tabla II adaptada):

```
Oración:        we use a language model to predict the probability of the next word.
WordPiece:      we use a language model to pre ##di ##ct the pro ##ba ##bility of the next word .
BERT original:  we use a language [M] to [M] ##di ##ct the pro [M] ##bility of the next word .
```

Nótese que el modelo BERT original ve `pre ##di ##ct` con solo `pre` enmascarado, y `pro [M] ##bility` con solo `##ba` enmascarado. Para predecir `##ba`, el modelo casi no necesita semántica: basta ver `pro` + `##bility` y la respuesta es obvia (los subwords `##ba` y `##bility` colocan de forma casi determinista después de `pro` en este vocabulario). De forma análoga, predecir `pre` viendo `##di ##ct` es trivial porque solo hay un puñado de palabras en el vocabulario que terminan en `##di ##ct`. **El modelo está aprendiendo morfología local de WordPiece, no semántica contextual**.

El problema es especialmente serio en **chino**, donde la tokenización BERT original funciona a nivel de **carácter individual** (cada han-zi es un token). Una palabra china típica tiene 2-4 caracteres. Por ejemplo, 哈尔滨 (Hā'ěrbīn, Harbin) son tres caracteres. Si BERT enmascara solo 哈, el modelo puede predecirlo trivialmente viendo 尔滨 — porque la única palabra del vocabulario que termina en 尔滨 es Harbin. La predicción no requiere comprender contexto, solo memorizar combinaciones de caracteres.

El paper formula este problema en la Sección II.A: *"the whole word masking (wwm) for optimizing the original masking in the MLM task. In this setting, instead of randomly selecting WordPiece tokens to mask, we always mask all of the tokens corresponding to a whole word at once. This explicitly forces the model to recover the whole word in the MLM pre-training task instead of just recovering WordPiece tokens, which is much more challenging."*

La idea no es exclusivamente china — Google adoptó WWM también en inglés con `bert-large-cased-whole-word-masking`, liberado en mayo 2019 (un mes antes del paper de Cui et al.). Pero Google nunca publicó un paper describiéndolo: lo liberó como una nota en el README del repo `google-research/bert`. **El paper de Cui et al. es el primer documento académico que describe formalmente WWM** y mide su impacto cuantitativamente. Esta es la razón por la que el WWM se cita típicamente como "Cui et al. 2019" aunque la receta es paralela a la de Google.

El contexto industrial también importa. iFLYTEK es la principal empresa china de tecnologías de voz y NLP (~50,000 empleados). HIT (Harbin Institute of Technology) tiene uno de los grupos de NLP más fuertes de Asia. El paper es el primero de una serie larga: Cui et al. publicaron también la versión MacBERT (TASLP 2021), Chinese-LLaMA-Alpaca (2023), y han mantenido el repo `ymcui/Chinese-BERT-wwm` como referencia para la comunidad china desde 2019.

---

## 2. Idea central de Whole Word Masking — definición formal

### 2.1 Definición

Sea $w$ una palabra del input que se tokeniza en $k$ subwords $w = (t_1, t_2, \ldots, t_k)$. En **BERT original**, el procedimiento de selección es:

$$\text{seleccionar}(t_i) \overset{\text{iid}}{\sim} \text{Bernoulli}(0.15), \quad \forall i \in \{1, \ldots, k\}$$

Es decir, cada subword se selecciona independientemente con probabilidad 15%. La cantidad de subwords enmascarados de una palabra de $k$ subwords sigue distribución $\text{Binomial}(k, 0.15)$.

En **WWM**, la selección se hace a nivel de palabra y se propaga:

$$\text{seleccionar}(w) \sim \text{Bernoulli}(0.15)$$

$$\text{seleccionar}(t_i) = \text{seleccionar}(w), \quad \forall i \in \{1, \ldots, k\}$$

Es decir, **todos los subwords de la palabra se enmascaran juntos o todos quedan visibles juntos**. La distribución de la cantidad de subwords enmascarados por palabra ya no es binomial sino un Bernoulli escalado: con probabilidad 15% se enmascaran los $k$ subwords, con probabilidad 85% se enmascaran 0.

### 2.2 Ejemplo en inglés

```
Oración:           the playing children laughed loudly
WordPiece:         the play ##ing child ##ren laughed loud ##ly
                    │   │   │     │    │    │      │   │
                   t1  t2  t3    t4   t5   t6     t7  t8

BERT original (selecciona t3 y t7):
                   the play [M]   child ##ren laughed [M] ##ly

WWM (selecciona la palabra "playing" → t2,t3 ; y "loudly" → t7,t8):
                   the [M]  [M]   child ##ren laughed [M] [M]
```

Para predecir `##ing` desde `play`, BERT original tiene una tarea casi trivial: la mayoría de las palabras inglesas con `play` como prefijo terminan en `##ing`, `##ed`, `##er`, `##s`, `##ful`. El modelo aprende un sufijado morfológico. WWM en cambio le pide predecir `play ##ing` juntos a partir de `the [M] [M] child ##ren laughed [M] [M]` — eso requiere entender que el sujeto es `children`, que el verbo está en presente progresivo, y elegir un verbo que combine semánticamente con `children laughed`.

### 2.3 Ejemplo en chino

El paper presenta el siguiente ejemplo (Tabla II, adaptado):

```
Oración:                  使用语言模型来预测下一个词的概率。
                          ("usar un modelo de lenguaje para predecir la probabilidad de la siguiente palabra")

Segmentación CWS (LTP):   语言 | 模型 | 来 | 预测 | 下 | 一个 | 词 | 的 | 概率 | 。
                          ("idioma" | "modelo" | "para" | "predecir" | "siguiente" | "una" | "palabra" | "de" | "probabilidad" | ".")

Tokenizer BERT (char):    语 言 模 型 来 预 测 下 一 个 词 的 概 率 。

BERT original masking:    语 言 [M] 型 来 [M] 测 下 一 个 词 的 概 率 。
                          (enmascara 模 de 模型, y 预 de 预测 — caracteres sueltos)

WWM masking:              语 言 [M] [M] 来 [M] [M] 下 一 个 词 的 概 率 。
                          (enmascara la palabra completa 模型 y la palabra completa 预测)
```

Para predecir 模 viendo 型 al lado, BERT original tiene una tarea memorística trivial: en el vocabulario chino, 模 + 型 forma 模型 (modelo) y casi no hay otras combinaciones con 型 a la derecha que sean palabras válidas. El modelo aprende **co-ocurrencias de caracteres**, no semántica.

Con WWM, predecir 模型 entero (sin ver ninguno de sus dos caracteres) requiere usar el contexto: 语言 [palabra] 来 预测 — "lenguaje [palabra] para predecir" — y deducir que la palabra más probable es modelo, gramática, sistema, etc. Eso sí es semántica.

### 2.4 Identificación de límites de palabra

WWM requiere saber qué subwords pertenecen a qué palabra. Hay tres regímenes según el idioma:

| Idioma | Heurística para límites de palabra |
|---|---|
| Inglés, español, francés, alemán | **Whitespace tokenization** previo al WordPiece. Un grupo de subwords que comparten una pre-tokenización por espacios es "una palabra". Los subwords `##xxx` (continuación) se agregan al subword anterior sin `##`. |
| Chino, japonés | **Sin whitespace** entre palabras. Se necesita un segmentador externo. Cui et al. usan **LTP** (Language Technology Platform, Che et al. 2010, también de HIT). LTP devuelve listas de palabras; cada carácter chino de una palabra se trata como subword "agrupado". |
| Coreano | Whitespace separa palabras (eojeol), pero la morfología es flexiva. Algunos modelos coreanos usan un analizador morfológico (MeCab-ko, Khaiii) además del whitespace. |
| Tailandés, lao, jemer | Sin whitespace entre palabras. Se usan segmentadores específicos (PyThaiNLP). |

**Notar**: WWM solo afecta la **selección de tokens para masking**. El tokenizer del modelo sigue siendo WordPiece (en inglés/chino BERT) o BPE (en RoBERTa, GPT). No hay un nuevo vocabulario ni una nueva arquitectura. La complejidad adicional está en el **preprocessing**: necesitas una lista de límites de palabra antes de generar las máscaras.

En el código de Google (`create_pretraining_data.py`), la implementación es:

```python
# Para inglés/whitespace languages:
# 1. Pre-tokenize por whitespace y puntuación: ["The", "playing", "children", "."]
# 2. Por cada palabra, aplica WordPiece: ["The"], ["play", "##ing"], ["child", "##ren"], ["."]
# 3. Cada lista interna es un "whole word".
# 4. Selecciona el 15% de whole words y enmascara TODOS sus subwords juntos.
```

Para chino, Cui et al. modifican el paso 1 reemplazando la pre-tokenización por whitespace con LTP:

```python
# Para chino:
# 1. Pre-tokenize con LTP: ["语言", "模型", "来", "预测", ...]
# 2. Cada palabra china se trata como una lista de caracteres: [["语","言"], ["模","型"], ["来"], ["预","测"], ...]
# 3. Cada lista interna es un "whole word".
# 4. Selecciona el 15% de whole words y enmascara TODOS sus caracteres juntos.
```

---

## 3. Comparación detallada con BERT original

### 3.1 Distribución de tokens enmascarados

Bajo BERT original, la fracción de tokens enmascarados por documento es exactamente 15% (la varianza es despreciable porque los documentos son largos). Bajo WWM, la fracción **fluctúa** con la longitud promedio de palabra del corpus:

- Si la palabra promedio tiene $\bar{k}$ subwords, una selección Bernoulli(0.15) a nivel de palabra produce una fracción esperada de subwords enmascarados igual a $0.15 \times \bar{k} / \bar{k} = 0.15$ — la **misma media**, pero con varianza mayor.
- Sin embargo, los subwords enmascarados ya no son independientes: están agrupados en clusters del tamaño de las palabras. La "información geográfica" del masking cambia.

En inglés, $\bar{k} \approx 1.3$ subwords/palabra con un vocabulario de 30K WordPiece. En chino, $\bar{k} \approx 1.6$ caracteres/palabra (por la segmentación CWS). En alemán, $\bar{k} \approx 2.0$ por las palabras compuestas largas. En turco, $\bar{k} \approx 2.5-3.0$ por la aglutinación.

### 3.2 Información que el modelo "no ve"

Esta es la clave del por qué WWM importa. Considerar dos escenarios para una palabra $w = (t_1, t_2, t_3)$ donde $t_2$ es seleccionado:

| Régimen | Input que ve el modelo | Información disponible para predecir $t_2$ |
|---|---|---|
| BERT original | `... t_1 [MASK] t_3 ...` | $t_1$, $t_3$, contexto global. Predicción tiende a usar **morfología local** ($t_1 + t_3$ casi determina $t_2$). |
| WWM | `... [MASK] [MASK] [MASK] ...` | Solo contexto global. Predicción debe usar **semántica frasal**. |

**El cambio es cualitativo**: BERT original aprende a completar morfología subword (un problema *cerrado* con vocabulario pequeño localmente); WWM aprende a completar palabras enteras (un problema *abierto* con vocabulario del orden de decenas de miles).

### 3.3 Por qué WWM fuerza al modelo a usar contexto sintáctico

Hay una intuición teórica clara. Para predecir un subword aislado de una palabra parcial, la entropía condicional es muy baja:

$$H(t_i \mid t_{1,\ldots,i-1}, t_{i+1,\ldots,k}) \ll H(t_i \mid \text{contexto global})$$

Por ejemplo, en "play [MASK] ##ren" → ##ing children..., el subword `##ing` tiene casi entropía cero dado el sufijo. La información del contexto global aporta poco al masking subword.

Para predecir una palabra entera dado solo contexto global, la entropía es alta:

$$H(w \mid \text{contexto sin } w) \approx H(w \mid \text{frase completa con hueco})$$

Y esa entropía solo se puede reducir aprendiendo sintaxis (qué clases de palabra encajan en ese hueco) y semántica (qué significado encaja). El gradiente que llega al modelo es **más informativo** porque la tarea es **más difícil**.

Es la misma intuición que llevó a SpanBERT (Joshi et al. 2019) a enmascarar spans contiguos de varias palabras, y a ELECTRA (Clark et al. 2020) a usar tokens reemplazados (no enmascarados) para forzar al modelo a usar contexto. WWM es el primer paso en esa dirección.

### 3.4 Impacto cuantitativo en chino

La Tabla V del paper (CMRC 2018 y DRCD, machine reading comprehension):

| Modelo | CMRC 2018 dev EM/F1 | CMRC 2018 test EM/F1 | CMRC 2018 challenge EM/F1 | DRCD dev EM/F1 | DRCD test EM/F1 |
|---|---|---|---|---|---|
| BERT (Devlin original) | 65.5 / 84.5 | 70.0 / 87.0 | 18.6 / 43.3 | 83.1 / 89.9 | 82.2 / 89.2 |
| **BERT-wwm** | 66.3 / 85.6 | 70.5 / 87.4 | 21.0 / 47.0 | 84.3 / 90.5 | 82.8 / 89.7 |
| BERT-wwm-ext | 67.1 / 85.7 | 71.4 / 87.7 | 24.0 / 47.3 | 85.0 / 91.2 | 83.6 / 90.4 |
| RoBERTa-wwm-ext | 67.4 / 87.2 | 72.6 / 89.4 | 26.2 / 51.0 | 86.6 / 92.5 | 85.6 / 92.0 |
| MacBERT-base | 68.5 / 87.9 | 73.2 / 89.5 | 30.2 / 54.0 | 89.4 / 94.3 | 89.5 / 93.8 |
| RoBERTa-wwm-ext-large | 68.5 / 88.4 | 74.2 / 90.6 | 31.5 / 60.1 | 89.6 / 94.8 | 89.6 / 94.5 |
| MacBERT-large | **70.7 / 88.9** | **74.8 / 90.7** | **31.9 / 60.2** | **91.2 / 95.6** | **91.7 / 95.6** |

Lecturas:

- **BERT → BERT-wwm**: +0.8 EM, +1.1 F1 en CMRC 2018 dev. Ganancia modesta pero consistente con el mismo corpus.
- **BERT-wwm → BERT-wwm-ext**: +0.8 EM, solo cambia el corpus de 0.4B a 5.4B palabras (más datos).
- **El "challenge set" de CMRC 2018** (preguntas que requieren razonamiento profundo) muestra ganancias mucho más grandes: BERT 18.6 → MacBERT-large 31.9 EM (**+13.3 puntos absolutos**). Es donde la diferencia entre "predecir subwords por morfología" y "predecir palabras por semántica" se ve más.

---

## 4. Aplicación al chino: detalles de implementación

### 4.1 Corpus

El paper usa dos corpora:

| Corpus | Tamaño | Notas |
|---|---|---|
| Chinese Wikipedia dump (25 mar 2019) | 0.4B palabras (~13M docs) | Simplified + Traditional sin conversión |
| In-house "extended" (modelos `-ext`) | 5.4B palabras | Encyclopedia + news + Q&A web |

Para WWM se usa **LTP** ([Che, Li, Liu 2010](https://github.com/HIT-SCIR/ltp)) — Language Technology Platform, también producto de HIT. LTP es un segmentador chino basado en CRF + DNN. Cui et al. mencionan en la Sección V.A: *"In order to identify the boundary of Chinese words for whole word masking, we use LTP for Chinese word segmentation."*

Detalle pragmático: el **vocabulario** del tokenizer no cambia. Sigue siendo el vocabulario de 21,128 tokens del Chinese BERT original (Devlin), que es character-level (cada han-zi es un token, más unos pocos subwords latinos y números). LTP solo se usa para **agrupar caracteres en palabras durante el preprocessing del masking**. Esto es importante porque permite:

- Reutilizar el checkpoint pre-entrenado de Devlin (BERT-base-chinese)
- Hacer fine-tuning con el tokenizer normal sin cambios
- Cargar el modelo en HuggingFace sin tokenizer custom

### 4.2 Hiperparámetros (Tabla III del paper)

| | BERT (Devlin) | BERT-wwm | RoBERTa-wwm | RBT | ELECTRA | MacBERT |
|---|---|---|---|---|---|---|
| Word # corpus | 0.4B | 5.4B | 5.4B | 5.4B | 5.4B | 5.4B |
| Vocab size | 21,128 | 21,128 | 21,128 | 21,128 | 21,128 | 21,128 |
| Activation | GeLU | GeLU | GeLU | GeLU | GeLU | GeLU |
| Optimizer | AdamW | LAMB | AdamW | AdamW | AdamW | LAMB |
| Training steps (base/large) | ? | 2M | 1M / 2M | 1M | 1M / 2M | 1M / 2M |
| Initial checkpoint (base) | random | BERT | BERT | RoBERTa | random | BERT |

Notar que para BERT-wwm, RoBERTa-wwm y MacBERT-base, el checkpoint inicial es el **BERT-base-chinese de Google** — no entrenan desde cero. Esto es una decisión pragmática para ahorrar cómputo (~3-4 días de TPU v3-8). Los modelos `-large` sí entrenan desde cero porque no había un checkpoint chino large disponible.

Sequence length: 512 tokens (no usan el schedule 128→512 de Devlin; argumentan que para tareas de reading comprehension la pre-training a 128 deja al modelo poco adaptado a contextos largos).

Hardware: *"a single Google Cloud TPU v3-8 (equals to a single TPU) or TPU Pod v3-32 (equals to 4 TPUs)"*. Costo aproximado para MacBERT-large (2M steps, batch 512): ~$10-15K USD en TPU reservada.

### 4.3 Tabla completa de resultados en 10 datasets chinos

El paper evalúa 10 datasets cubriendo MRC, single-sentence classification y sentence-pair classification:

| Dataset | Task | Train / Dev / Test | Métricas |
|---|---|---|---|
| CMRC 2018 | MRC span extraction (simplified) | 10K / 3.2K / 4.9K | EM, F1 |
| DRCD | MRC span extraction (traditional) | 27K / 3.5K / 3.5K | EM, F1 |
| CJRC | MRC con yes/no + no-answer (legal) | 10K / 3.2K / 3.2K | EM, F1 |
| ChnSentiCorp | Sentiment binario | 9.6K / 1.2K / 1.2K | Accuracy |
| THUCNews | News classification (10 dominios) | 50K / 5K / 10K | Accuracy |
| TNEWS | News title classification (15 clases) | 53.3K / 10K / 10K | Accuracy |
| XNLI (Chinese) | Natural language inference | 392K / 2.5K / 5K | Accuracy |
| LCQMC | Question matching | 240K / 8.8K / 12.5K | Accuracy |
| BQ Corpus | Bank question matching | 100K / 10K / 10K | Accuracy |
| OCNLI | Original Chinese NLI | 56K / 3K / 3K | Accuracy |

Resultados resumidos en sentence-pair classification (Tabla VIII, adaptada):

| Modelo | XNLI test | LCQMC test | BQ test | OCNLI dev |
|---|---|---|---|---|
| BERT | 77.8 | 86.9 | 84.8 | 74.6 |
| BERT-wwm | 78.2 | 87.0 | 85.2 | 74.6 |
| BERT-wwm-ext | 78.7 | 87.1 | 85.3 | 76.0 |
| RoBERTa-wwm-ext | 78.8 | 86.4 | 85.0 | 76.5 |
| MacBERT-base | 79.3 | 87.0 | 85.2 | 77.0 |
| RoBERTa-wwm-ext-large | 81.2 | 87.0 | 85.8 | 78.5 |
| MacBERT-large | **81.3** | **87.6** | 85.6 | **79.0** |

La ganancia BERT → BERT-wwm en XNLI es de +0.4 puntos. Modesta pero significativa dado que XNLI es un dataset grande (392K train) donde el efecto del pre-training se dilute. En CMRC 2018 challenge set (donde el dataset es chico y la dificultad alta), las ganancias son mucho mayores (+13 puntos como vimos antes).

---

## 5. MacBERT (versión TASLP 2021): correcciones sobre WWM

La versión TASLP 2021 del paper introduce **MacBERT** (MLM as Correction BERT), una extensión de WWM que ataca dos problemas adicionales:

### 5.1 N-gram masking

Adicionalmente a WWM, MacBERT enmascara **n-gramas** de hasta 4 palabras con probabilidades 40% (unigram), 30% (bigram), 20% (trigram), 10% (4-gram). Esto es similar a SpanBERT (Joshi et al. 2019) que usa una distribución geométrica de longitudes de span con $p = 0.2$.

La motivación: incluso WWM solo enmascara una palabra a la vez. Predecir una palabra dado todo el resto del contexto puede ser fácil si el contexto es muy específico (por ejemplo, predecir 模型 viendo "neural ___" en un corpus de ML). Enmascarar bigrams o trigrams enteros aumenta la dificultad y obliga al modelo a usar más contexto distal.

### 5.2 MLM as Correction (Mac)

Esta es la innovación conceptual más importante de la versión TASLP. En BERT original (y BERT-wwm), los tokens seleccionados para masking se reemplazan con la regla 80/10/10:

- 80% `[MASK]` (token artificial que nunca aparece en fine-tuning → discrepancia)
- 10% token aleatorio (rompe la naturalidad del texto)
- 10% mantener (señal débil)

MacBERT propone:

- 80% **sinónimo** del token original (obtenido de Synonyms toolkit basado en word2vec)
- 10% token aleatorio
- 10% mantener

El modelo nunca ve `[MASK]`. En vez de "predice qué iba aquí cuando ves el token fantasma", el modelo aprende "**corrige** este token: si parece raro en el contexto, propón uno mejor". Es similar al paradigma de ELECTRA pero más simple (no requiere un generador separado).

Ejemplo del paper (Tabla II, fila +++ Mac):

```
Original:  we use a language model to predict the probability of the next word.
Mac:       we use a text  system to ca##lc##ulate the po##si##bility of the next word.
```

`language` → `text`, `model` → `system`, `predict` → `calculate`, `probability` → `possibility`. Todos sinónimos plausibles que rompen la oración original pero mantienen la coherencia gramatical y semántica suficiente para que el modelo entrene "limpieza" en vez de "completado de hueco".

### 5.3 Sentence Order Prediction (SOP)

MacBERT reemplaza NSP por SOP (siguiendo ALBERT, Lan et al. 2019):

- Positivos: par $(A, B)$ contiguo en el orden correcto.
- Negativos: par $(A, B)$ contiguo en el **orden invertido** $(B, A)$.

Es más difícil que NSP porque ambos segmentos vienen del mismo documento (sin shift de dominio), y requiere entender flujo temporal/lógico real entre oraciones.

### 5.4 Ablations de MacBERT (Tabla X)

| Sistema | CMRC 2018 EM/F1 | DRCD EM/F1 | XNLI | OCNLI | AVG |
|---|---|---|---|---|---|
| MacBERT-large | 74.8 / 90.7 | 91.7 / 95.6 | 81.3 | — | **87.18** |
| SOP → NSP | 74.5 / 90.6 | 91.5 / 95.5 | 81.2 | — | 87.00 |
| w/o SOP | 74.4 / 90.6 | 91.0 / 95.4 | 81.1 | — | 86.89 |
| w/o Mac | 74.2 / 90.1 | 91.2 / 95.4 | 81.2 | — | 86.88 |
| w/o NM | 74.0 / 89.8 | 90.9 / 95.1 | 81.3 | — | 86.89 |
| RoBERTa-large (baseline) | 74.2 / 90.6 | 89.6 / 94.5 | 81.2 | — | 86.79 |

Las contribuciones de cada componente son aproximadamente equivalentes (~0.3 puntos AVG cada uno). N-gram masking y Mac son las dos más efectivas. SOP aporta menos que NSP-removal pero ayuda marginalmente.

### 5.5 Investigación sobre el régimen de masking (Sección VII.B y Figura 2)

El paper hace un experimento crucial: entrena MacBERT desde 1M a 2M pasos con cuatro variantes de masking:

| Variante | 80% replacement strategy |
|---|---|
| MacBERT | similar words (sinónimos) |
| Random Replace | random words from vocabulary |
| Partial Mask | original BERT (80% `[MASK]`, 10% random, 10% same) |
| All Mask | 90% `[MASK]`, 10% same |

Resultados en CMRC 2018:

- **MacBERT (similar words)** > **Random Replace** > **Partial Mask** > **All Mask**.
- *"Random words rather than the artificial token [MASK] could improve the de-noising ability of the pre-trained model."*
- La fluidez del texto reemplazado importa: sinónimos > random words > `[MASK]` artificial.

Esta es una validación empírica fuerte de que el problema central del MLM original es el **mismatch [MASK] pretrain/finetune**. ELECTRA lo resuelve eliminando `[MASK]` con un discriminador. MacBERT lo resuelve eliminando `[MASK]` con sinónimos. Ambas direcciones funcionan.

---

## 6. Generalización a otros idiomas: el segundo aire de WWM

WWM es una de esas ideas que se propagan rápido en la comunidad porque son **fáciles de implementar y aportan ganancias consistentes**. La adopción cross-lingual fue casi inmediata.

### 6.1 Inglés: Google adoptó WWM

En mayo 2019 (un mes antes del paper de Cui et al.), Google liberó `bert-large-cased-whole-word-masking` y `bert-large-uncased-whole-word-masking` en `google-research/bert`. El README dice:

> *"Whole Word Masking is a recent update to BERT. The training is identical -- we still predict each masked WordPiece token independently. The improvement comes from the fact that the original prediction task was too 'easy' for words that had been split into multiple WordPieces."*

Google nunca publicó un paper formal sobre esto, pero los modelos están disponibles en HuggingFace. Para SQuAD v1.1, las ganancias reportadas son:

| Modelo | Dev EM | Dev F1 |
|---|---|---|
| BERT-large-cased (original) | 84.1 | 90.9 |
| BERT-large-cased-wwm | 86.5 | 92.8 |

**+2.4 EM, +1.9 F1** solo por cambiar la estrategia de masking. Mismo modelo, mismo corpus, mismo número de pasos.

### 6.2 Español: BETO usó WWM por default

[BETO](https://github.com/dccuchile/beto) (Cañete, Chaperon, Fuentes, Ho, Kang, Pérez, 2020) es el primer BERT específicamente entrenado en español, desarrollado en el CENIA-UC (Centro Nacional de Inteligencia Artificial, Universidad de Chile). El paper original (PMLDC 2020) reporta:

> *"We trained BETO using the Whole Word Masking technique."*

La decisión es deliberada y se justifica por la **morfología flexiva del español**, donde una palabra puede partirse en muchos subwords. Sin WWM, el modelo aprendería sufijos morfológicos (-ando, -ería, -mente, -ción) en vez de semántica.

Los checkpoints públicos son:

- `dccuchile/bert-base-spanish-wwm-cased` — el que usa el lab 20
- `dccuchile/bert-base-spanish-wwm-uncased` — versión uncased

El sufijo `wwm` en el nombre del modelo HuggingFace es literal: indica que fue pre-entrenado con Whole Word Masking. **Sin WWM se llamaría simplemente `bert-base-spanish-cased`**, y los autores explicitamente decidieron incluir el sufijo para enfatizar la diferencia.

Resultados de BETO vs mBERT en POS, NER, MLDoc, PAWS-X, XNLI:

| Modelo | POS | NER | MLDoc | PAWS-X | XNLI |
|---|---|---|---|---|---|
| mBERT (multilingual) | 97.10 | 87.38 | 95.70 | 81.20 | 78.50 |
| **BETO-wwm-cased** | **98.97** | **88.43** | **96.12** | **89.05** | **82.01** |

BETO supera a mBERT por márgenes claros en todas las tareas, principalmente porque (a) ve más datos en español, (b) usa WWM, (c) tiene un tokenizer optimizado para español.

### 6.3 Por qué WWM importa más en idiomas con morfología flexiva

El argumento clave: **la fracción de palabras que se parte en múltiples subwords** depende de la morfología del idioma y del tamaño del vocabulario del tokenizer.

| Idioma | Tipología | Fracción de palabras multi-subword (BPE 30K) |
|---|---|---|
| Inglés | Analítico moderado | ~15-25% |
| Chino | Aislante (sin morfología) | ~0% palabras, pero 60-70% si tratamos cada char como subword |
| **Español** | **Flexivo (verbos: 50+ formas)** | **~35-45%** |
| Alemán | Flexivo + compuestos | ~50-60% |
| Ruso | Flexivo (6 casos, 3 géneros) | ~55-70% |
| Turco | Aglutinante | ~75-85% |
| Finlandés | Aglutinante | ~80-90% |

Para idiomas con morfología pesada, una alta fracción de palabras se parte. Sin WWM, el modelo aprende a completar morfología local en lugar de semántica. **Este es el grupo de idiomas donde WWM tiene mayor impacto**.

Ejemplo concreto en español: la palabra "comerían" (3ª persona plural condicional de "comer") se tokeniza típicamente como:

```
"comerían" → "com" + "##er" + "##ían"   (3 subwords)
```

Sin WWM:

- BERT puede enmascarar solo `##er`: `com [MASK] ##ían`
- Para predecir `##er` desde `com` + `##ían`, basta saber que entre la raíz `com` y la terminación `##ían` casi siempre va `##er` (no `##ar` ni `##ir`). El modelo no aprende semántica: aprende que el patrón `com X ##ían` casi determina `X = ##er`. Es una conjugación regular.

Con WWM:

- BERT enmascara los tres subwords juntos: `[MASK] [MASK] [MASK]`
- Para predecir `comerían`, el modelo debe usar el contexto: *"si tuvieran hambre [MASK] [MASK] [MASK] el pastel completo"*. Solo se puede deducir que es un verbo en condicional plural conjugado con sujeto plural, y debe encajar semánticamente con "pastel" — eso sugiere `comerían`, no `dormirían` ni `correrían`.

El gradiente que recibe el modelo en WWM es **incomparablemente más rico** que en BERT original para idiomas con conjugación.

### 6.4 Otros modelos que adoptaron WWM

| Modelo | Idioma | Autor / Año |
|---|---|---|
| `bert-large-(un)cased-whole-word-masking` | Inglés | Google, 2019 |
| `bert-base-japanese-whole-word-masking` | Japonés | Tohoku NLP, 2019 |
| **Chinese-BERT-wwm** (este paper) | Chino | Cui et al., 2019 |
| **BETO-wwm** | Español | CENIA-UC, 2020 |
| AlBERTo | Italiano | Polignano et al., 2019 (no WWM explícito) |
| CamemBERT | Francés | Martin et al., 2020 (WWM via SentencePiece subword regularization) |
| RuBERT-wwm | Ruso | DeepPavlov, 2020 |
| HerBERT | Polaco | Mroczkowski et al., 2021 (WWM con segmentador morfológico KRNNT) |
| FinBERT | Finés | Virtanen et al., 2019 |

Para finés (idioma aglutinante extremo, palabras pueden tener 10+ morfemas), Virtanen et al. argumentan que sin WWM el modelo aprende solo morfología y nunca semántica. FinBERT con WWM alcanza el state of the art en finés.

---

## 7. Detalles de implementación

### 7.1 Modificación en `create_pretraining_data.py`

El código original de Google para BERT (sin WWM) selecciona tokens uniformemente al azar. Para WWM, la modificación es:

```python
# Pseudocódigo simplificado del create_pretraining_data.py con WWM

def create_masked_lm_predictions_whole_word(tokens, words):
    """
    tokens: lista de subwords (con prefijos ## en BERT inglés)
    words: lista de listas, donde cada inner-list son los índices de subwords de una palabra
            Ejemplo: words = [[0], [1,2], [3,4], [5]]   # The play##ing child##ren .
    """
    # 1. Calcula cuántas palabras enmascarar
    num_to_predict = int(len(tokens) * 0.15)
    
    # 2. Baraja la lista de palabras y selecciona las primeras N
    shuffled_words = random.shuffle(words)
    masked_word_indices = []
    cnt = 0
    for word_indices in shuffled_words:
        if cnt + len(word_indices) > num_to_predict:
            break
        masked_word_indices.extend(word_indices)
        cnt += len(word_indices)
    
    # 3. Aplica regla 80/10/10 a cada subword de las palabras seleccionadas
    for idx in masked_word_indices:
        r = random.random()
        if r < 0.8:
            tokens[idx] = "[MASK]"
        elif r < 0.9:
            tokens[idx] = random_token_from_vocab()
        # else: keep original
    
    return tokens, masked_word_indices
```

Detalle: la regla 80/10/10 se aplica **subword por subword dentro de la palabra**, no a la palabra entera. Eso significa que en una palabra `play##ing`, el subword `play` puede ir a `[MASK]` mientras `##ing` puede mantenerse como `##ing` original (con probabilidad 10% cada uno). Esto introduce algo de variabilidad incluso dentro de la palabra enmascarada — el modelo no siempre ve `[MASK] [MASK]` sino a veces `[MASK] ##ing` o `play [MASK]` por la regla 80/10/10. La "whole word" se refiere a **selección conjunta de la palabra como candidata a masking**, no a aplicación uniforme de `[MASK]` a todos sus subwords.

### 7.2 Anclas vs followers en WWM

En la implementación de Chinese BERT-wwm, hay una distinción técnica:

- El **primer subword** de una palabra (que en BERT inglés no tiene prefijo `##`) es el "ancla". Si se selecciona, toda la palabra se enmascara.
- Los **subwords siguientes** con prefijo `##` son "followers". Se enmascaran junto con su ancla.

En chino, esta distinción no aplica directamente (no hay prefijo `##`), pero LTP devuelve listas de caracteres que forman palabras, y cada lista es tratada como un grupo "ancla + followers" implícito.

En el código del repo Chinese-BERT-wwm, hay un script `create_pretraining_data_wwm.py` modificado que toma como input un archivo donde las palabras están pre-segmentadas con marcadores especiales, por ejemplo:

```
原始文本: 使用语言模型来预测下一个词的概率。
分词后:   使用 语言 模型 来 预测 下 一个 词 的 概率 。
带标记:   使 ##用 语 ##言 模 ##型 来 预 ##测 下 一 ##个 词 的 概 ##率 。
```

El marcador `##` en chino es **artificial** — no es parte de la tokenización WordPiece (que en chino es character-level y no usa `##`). El script de Cui et al. introduce `##` como marcador interno para indicar "este carácter es un follower del anterior dentro de la misma palabra". Después de la selección de masking, los `##` se remueven y el input final pasa al tokenizer normal.

### 7.3 Balance con la regla 80/10/10

Hay una sutileza adicional. Si una palabra tiene $k$ subwords, y todos ellos se seleccionan como masking target (porque la palabra fue seleccionada), entonces:

- Cada subword tiene probabilidad 80% de ser `[MASK]`, 10% de ser random, 10% de mantenerse.
- La probabilidad de que **todos** los $k$ subwords sean `[MASK]` es $0.8^k$. Para $k=2$ es 64%, para $k=3$ es 51%, para $k=4$ es 41%.
- Eso significa que en algunos casos, una palabra WWM-seleccionada termina mostrando algunos de sus subwords originales o reemplazos random.

Esta variabilidad es **intencional**: mantiene la regla 80/10/10 original de BERT (que mitiga el mismatch `[MASK]`) y combina con WWM (que mitiga el masking demasiado fácil). Los dos efectos son ortogonales.

---

## 8. Limitaciones

### 8.1 WWM no resuelve span-based masking

WWM enmascara una palabra a la vez. Pero hay muchos fenómenos lingüísticos donde el span relevante es **multi-palabra**: named entities ("New York City"), idioms ("kick the bucket"), multi-word expressions ("a pesar de"). Para predecir "City" dado "New York [MASK]", el modelo todavía tiene una tarea fácil.

**SpanBERT** (Joshi et al. 2019, contemporáneo a WWM) ataca este problema enmascarando spans contiguos de longitud variable (sampleada de una distribución geométrica con $p=0.2$, longitud máxima 10). MacBERT incorpora n-gram masking hasta 4-gramas que es un paso intermedio. Pero ni WWM ni MacBERT son **entity-aware**.

### 8.2 WWM no resuelve named-entity boundary masking

Si un named entity tiene varias palabras (como "Roberto Araneda" o "Universidad de Chile"), WWM enmascara cada palabra independientemente. Para predecir "Araneda" dado "Roberto [MASK]", el modelo puede usar el contexto del apellido pero no aprende la **co-ocurrencia entidad-completa**.

**ERNIE 1.0** (Sun et al., Baidu, 2019) ataca esto con tres niveles de masking:

- Basic-level (subword/character): como BERT original
- Phrase-level: spans contiguos de palabras
- **Entity-level**: spans que corresponden a entidades nombradas (detectadas con NER previa)

ERNIE muestra ganancias sobre BERT-wwm en tareas chinas que dependen de entidades (relation extraction, entity typing).

### 8.3 WWM sigue siendo subword-aware, no entity-aware

La distinción es importante para tareas de knowledge-intensive NLP:

- **Subword-aware** (BERT original): el modelo razona sobre tokens individuales del vocabulario WordPiece.
- **Word-aware** (BERT-wwm): el modelo razona sobre palabras enteras.
- **Entity-aware** (ERNIE, K-BERT, KEPLER): el modelo razona sobre entidades del conocimiento estructurado.
- **Knowledge-aware** (T5+entity embeddings, ERNIE 3.0): el modelo razona sobre relaciones de un knowledge graph.

WWM es un paso intermedio, no el final. Para RAG moderno (2024-2025) la tendencia es delegar el conocimiento estructurado al retrieval (vector DB + ranking) y dejar al language model enfocado en razonamiento. WWM sigue siendo útil porque mejora las representaciones internas que alimentan el retriever (sentence-transformers, embeddings).

### 8.4 Sensibilidad al segmentador (en chino)

En chino, WWM depende de la calidad del segmentador (LTP en este paper). Si el segmentador comete errores — por ejemplo, partir 哈尔滨 (Harbin) en 哈尔 + 滨 — entonces WWM enmascara incorrectamente. LTP tiene una precisión de ~97% en CWS, pero el 3% restante introduce ruido en el pre-training.

Alternativas:

- **Jieba**: segmentador chino ampliamente usado, basado en HMM + diccionario.
- **THULAC**: Tsinghua University Lexical Analyzer for Chinese.
- **ICTCLAS**: Institute of Computing Technology, Chinese Lexical Analysis System.

Cui et al. eligen LTP probablemente porque es del mismo grupo HIT-SCIR (sesgo institucional + dogfooding). Reportan que la elección del segmentador afecta el resultado pero no de forma catastrófica.

### 8.5 No resuelve el mismatch `[MASK]` (eso es lo que ataca MacBERT y ELECTRA)

WWM mantiene la regla 80/10/10 con `[MASK]` artificial. Solo cambia **qué se selecciona** para masking, no **con qué se reemplaza**. El mismatch pretrain/finetune sigue ahí.

MacBERT lo ataca reemplazando `[MASK]` por sinónimos. ELECTRA lo ataca eliminando `[MASK]` y usando un discriminador. Ambos enfoques son ortogonales a WWM y se pueden combinar (de hecho MacBERT combina WWM + N-gram masking + Mac).

---

## 9. Conexión con la clase 20 del Diplomado IA UC

La clase 20 trata el **Camino 4** del curso: encoders bidireccionales y su evolución desde ELMo (2018) a BERT (2018) a las variantes 2019-2021 (RoBERTa, ALBERT, ELECTRA, BETO). WWM es uno de los "**pequeños cambios que importan**" que la comunidad introdujo entre BERT (octubre 2018) y BETO (marzo 2020).

La narrativa del Camino 4 puede leerse como una secuencia de **ablations sobre BERT**:

| Cambio | Modelo de referencia | Ganancia en GLUE / equivalente |
|---|---|---|
| Sin NSP | RoBERTa | +0.5 |
| Dynamic masking | RoBERTa | +0.4 |
| Batch grande (8K) | RoBERTa | +1.0 |
| Más data + más pasos | RoBERTa | +2.0 |
| **WWM** | BERT-wwm / BETO | **+0.5-2.0 dependiendo de idioma** |
| Span-based masking | SpanBERT | +1.5 (en tareas span-extraction) |
| `[MASK]` reemplazado | ELECTRA / MacBERT | +1.0 |
| Param sharing + SOP | ALBERT | +0.5 (con menos params) |
| Disentangled attention | DeBERTa | +1.5 |

Cada cambio individualmente es pequeño, pero **compuestos** llevan a modelos como DeBERTa-v3 (2021) que superan a BERT-base por 5-7 puntos en GLUE con tamaños comparables.

WWM ilustra un meta-punto pedagógico relevante para el curso: **las ganancias en NLP de 2018-2020 no vinieron principalmente de arquitecturas radicalmente nuevas**, sino de iterar cuidadosamente sobre detalles del objetivo de pre-training, el régimen de masking, la escala de datos y batches, y el tokenizer. La arquitectura Transformer encoder de Vaswani 2017 + Devlin 2018 sigue siendo la misma; lo que cambió fue todo el resto.

Para el estudiante del Diplomado, esto importa porque:

1. Sugiere que **fine-tuning + cambios pequeños** pueden ser más rentables que diseñar arquitecturas nuevas.
2. Justifica la importancia de leer **papers de ablations** (RoBERTa, este paper, MacBERT) en lugar de solo papers fundacionales (BERT, GPT).
3. Conecta con la **historia industrial**: las empresas (Google, Facebook, iFLYTEK, Baidu) lideraron estos cambios incrementales porque tenían el cómputo para hacer ablations a escala.

---

## 10. Conexión con el lab 20

El lab 20 del Diplomado IA UC (Hugging Face: BERT y derivados en español) carga en sus celdas 25-26 el modelo:

```python
from transformers import AutoTokenizer, AutoModel

model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

El sufijo `wwm` en el nombre del modelo **es exactamente lo que describe este paper**. BETO usa Whole Word Masking para el pre-training, y eso lo distingue de un BERT estándar entrenado en español sin WWM.

Una demostración pedagógica útil para el lab sería comparar BETO con un modelo BERT en español **sin** WWM (por ejemplo, `mrm8488/bert-base-spanish-uncased`, que es una réplica sin WWM). Tokenizar las mismas oraciones con ambos modelos y aplicar masking artificial:

```python
# Ejemplo de comparación pedagógica

text = "Los estudiantes comerían empanadas si tuvieran hambre"

# Tokenización con BETO
toks_beto = tokenizer.tokenize(text)
# ['los', 'estudiantes', 'come', '##rían', 'em', '##pan', '##adas', 'si', 'tuvieran', 'hambre']

# Sin WWM, BERT enmascararía aleatoriamente subwords sueltos:
# ['los', 'estudiantes', 'come', '[MASK]', 'em', '##pan', '##adas', 'si', '[MASK]', 'hambre']
# Predecir '##rían' viendo 'come' es trivial (morfología local)

# Con WWM, BERT enmascararía palabras enteras:
# ['los', 'estudiantes', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', 'si', 'tuvieran', 'hambre']
# Predecir 'comerían empanadas' requiere semántica completa
```

Esta comparación visualizable es el tipo de ejercicio que el lab puede agregar para mostrar **por qué BETO usa WWM y no BERT vanilla**. Sin esa motivación, el sufijo `wwm` del nombre del checkpoint queda sin explicación — es un detalle técnico oculto que la mayoría de los estudiantes nunca cuestiona.

Otra extensión posible: usar `pipeline("fill-mask", model="dccuchile/bert-base-spanish-wwm-cased")` con `top_k=5` sobre oraciones donde se enmascara una palabra que se parte en varios subwords (como "convertibilidad", "constitucionalmente"). El modelo debería proponer palabras semánticamente coherentes — y esa coherencia es precisamente lo que WWM optimiza.

Tercera extensión: comparar BETO (WWM) con XLM-RoBERTa (sin WWM, modelo multilingual). Para tareas en español, BETO suele ganar por margen pequeño pero consistente, parte del cual es atribuible a WWM.

---

## 11. Notas para integrar al site

Cosas que el `papers/cui-wwm-2019.md` del site **debe** cubrir (sin duplicar este documento interno):

1. **Definición ejecutiva de WWM**: 2-3 párrafos con un ejemplo visual en español.
2. **Tabla de modelos que adoptaron WWM**: Chinese-BERT-wwm, BERT-large-wwm de Google, BETO-wwm, FinBERT, RuBERT-wwm. Útil para que el estudiante vea que es una técnica universal.
3. **Por qué BETO usa WWM**: argumento de morfología flexiva del español, con el ejemplo "comerían" → 3 subwords.
4. **Tabla de resultados chinos**: solo los 3-4 datasets más representativos (CMRC 2018, DRCD, XNLI-zh) y los modelos clave (BERT, BERT-wwm, RoBERTa-wwm, MacBERT).
5. **Conexión con MacBERT**: 1 párrafo mencionando que la versión TASLP 2021 introduce `MLM-as-correction` con sinónimos en lugar de `[MASK]`.
6. **Limitaciones**: WWM no resuelve span ni entity masking (eso es SpanBERT / ERNIE).

Cosas que ya cubren otros materiales del site y no hace falta repetir:

- `fundamentos/bert.md`: arquitectura BERT, MLM, NSP. Ya cubre la regla 80/10/10 estándar.
- `papers/devlin-bert-2018.md`: paper original BERT.
- `papers/beto-canete-2020.md` (si existe): debe cross-linkear a este paper para explicar el sufijo `wwm`.

---

## 12. Lectura recomendada complementaria

- **Devlin et al. 2018 (BERT)** — paper base. Cui et al. modifican solo la estrategia de masking.
- **Sun et al. 2019 (ERNIE 1.0)** — masking de entidades y frases en chino. Predecesor de Cui et al. pero con tres niveles en vez de uno.
- **Joshi et al. 2019 (SpanBERT)** — span-based masking. Generalización de WWM a spans contiguos arbitrarios.
- **Liu et al. 2019 (RoBERTa)** — ablations sobre BERT (dynamic masking, no NSP, batch grande). Combinable con WWM (RoBERTa-wwm).
- **Lan et al. 2019 (ALBERT)** — introduce SOP que MacBERT adopta.
- **Clark et al. 2020 (ELECTRA)** — alternativa más fuerte al mismatch `[MASK]`. MacBERT es un punto intermedio entre BERT-wwm y ELECTRA.
- **Cañete et al. 2020 (BETO)** — adopción de WWM al español. Modelo usado en el lab 20.
- **Cui et al. 2021 (versión TASLP de este paper)** — incluye MacBERT y comparaciones extendidas.
- **Wettig et al. 2023 (Should you mask 15% in MLM?)** — revisita la elección del ratio de masking. Concluye que con WWM + span masking, 40% es mejor que 15% para modelos grandes.
- **Levine et al. 2021 (PMI-masking)** — selección de spans para masking basada en pointwise mutual information. Mencionado por MacBERT como una dirección futura.
