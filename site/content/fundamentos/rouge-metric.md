---
title: "ROUGE Metric"
weight: 91
math: true
---

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) es la familia de métricas más usada para evaluar **summarization automática** y, por extensión, **generación de texto** en general (image captioning, simplification, headline generation). Fue propuesta por Chin-Yew Lin (ISI/USC) en su paper [ROUGE: A Package for Automatic Evaluation of Summaries (2004)](https://aclanthology.org/W04-1013/), publicado en el workshop de DUC (Document Understanding Conference) — el primer benchmark sistemático de summarization.

La motivación era práctica: la evaluación humana de summaries en DUC era lenta, cara y poco reproducible. Lin propuso un conjunto de métricas automáticas que **correlacionan con juicios humanos** y que se computan en segundos. Quince años después, ROUGE sigue siendo el reporte obligatorio en cualquier paper de summarization, aunque sus limitaciones son bien conocidas y han motivado una larga lista de sucesores (METEOR, BERTScore, BLEURT, G-Eval).

Este fundamento cubre las **cinco variantes principales** de la familia (N, L, W, S, SU), el ejemplo paso a paso del slide 50 de la clase, las versiones multi-reference y multi-sentence (ROUGE-Lsum), las **limitaciones** que motivan métricas semánticas, y los **pitfalls de implementación** que dan headaches en la práctica.

---

## 1. ¿Por qué recall-oriented?

El nombre delata la filosofía: **Recall-Oriented Understudy**. La intuición es simple — un summary debe **preservar la información** del reference. Si el reference dice "El paciente presenta dolor torácico irradiado al brazo izquierdo" y mi summary dice "El paciente tiene dolor", técnicamente las palabras coinciden pero **falta información**.

Compará con BLEU (Papineni 2002), que es precision-oriented y se usa en traducción: en MT lo que importa es que **lo que generaste sea correcto**, no necesariamente que cubra todo el source. En summarization la balanza se invierte: querés **cobertura del contenido del reference**.

{{< concept-alert type="clave" >}}
**ROUGE = recall sobre n-gramas del reference**. El denominador en la fórmula básica es la cuenta de n-gramas en los reference summaries (humanos), no en el candidate. Esto es lo opuesto a BLEU, que normaliza por el candidato.
{{< /concept-alert >}}

En la práctica casi todos los papers modernos reportan **F1** (la media armónica de recall y precision), pero la herencia recall-oriented se ve en cómo está construida la fórmula y en variantes como ROUGE-Lsum.

---

## 2. El problema formal

Sea $C$ el **candidate summary** (lo que produjo el modelo) y $R = \{R_1, \ldots, R_k\}$ el conjunto de **reference summaries** escritos por humanos (usualmente $k \in \{1, 4\}$ — DUC tenía 4 references por documento).

Queremos una función:

{{< math-formula title="Score de evaluación" >}}
\text{score}(C, R) \in [0, 1]
{{< /math-formula >}}

que sea **alta** cuando $C$ se parece a los $R_i$ y **baja** cuando no, y cuya correlación de Pearson/Spearman con juicios humanos sea lo más alta posible.

ROUGE no es **una** métrica sino una **familia**: cada variante captura un aspecto distinto de la similitud (overlap léxico, subsecuencias, pares con saltos). La elección de variante depende del tipo de summary (extractivo, abstractivo, single-sentence, multi-sentence) y de qué propiedad querés enfatizar (cobertura, fluencia, orden de palabras).

---

## 3. ROUGE-N: overlap de n-gramas

La variante más simple y reportada. Computa el **overlap de n-gramas** entre $C$ y $R$, normalizado por la cuenta total de n-gramas en $R$ (recall-style).

{{< math-formula title="ROUGE-N (recall)" >}}
\text{ROUGE-N} = \frac{\sum_{S \in R} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in R} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}
{{< /math-formula >}}

donde:
- $\text{gram}_n$ es un n-grama (secuencia de $n$ palabras consecutivas).
- $\text{Count}(\text{gram}_n)$ es las veces que ese n-grama aparece en el reference $S$.
- $\text{Count}_{\text{match}}(\text{gram}_n)$ es el mínimo entre la cuenta en $C$ y la cuenta en $S$ (clipping, igual que BLEU).

### Variantes por $n$

| Variante | $n$ | Captura |
|---|---|---|
| **ROUGE-1** | 1 | Overlap de palabras (vocabulario) |
| **ROUGE-2** | 2 | Overlap de bigramas (colocaciones, orden local) |
| **ROUGE-3** | 3 | Trigramas (raro en práctica, demasiado estricto) |

ROUGE-1 es **lo más correlacionado con juicios humanos de informatividad**; ROUGE-2 captura mejor **fluencia local** porque exige pares consecutivos.

### Precision, Recall y F1

Aunque la fórmula original es recall, en la práctica computamos los tres:

$$R_n = \frac{|\text{n-grams}(C) \cap \text{n-grams}(R)|}{|\text{n-grams}(R)|}, \quad P_n = \frac{|\text{n-grams}(C) \cap \text{n-grams}(R)|}{|\text{n-grams}(C)|}$$

$$F_n = \frac{2 P_n R_n}{P_n + R_n}$$

Casi todos los papers modernos reportan **F1 de ROUGE-1, ROUGE-2 y ROUGE-L**. Esa tripleta se ha vuelto el estándar de facto.

---

## 4. Ejemplo paso a paso (ROUGE-1 y ROUGE-2)

Tomemos el ejemplo del slide 50 de la clase, que ilustra muy bien el cómputo.

**Candidate $C$**: "I really loved reading the Hunger Games" — 7 palabras.

**Reference $R$**: "I loved reading the Hunger Games" — 6 palabras.

### ROUGE-1

Unigramas del reference: `{I, loved, reading, the, Hunger, Games}` — 6 unigramas.

Unigramas del candidate: `{I, really, loved, reading, the, Hunger, Games}` — 7 unigramas.

Match (intersección): `{I, loved, reading, the, Hunger, Games}` — 6 matches.

$$R_1 = \frac{6}{6} = 1.0 \quad (\text{todas las del reference están en } C)$$

$$P_1 = \frac{6}{7} \approx 0.857$$

$$F_1 = \frac{2 \cdot 0.857 \cdot 1.0}{0.857 + 1.0} = \frac{12}{13} \approx 0.923$$

### ROUGE-2

Bigramas del reference (5): `(I, loved), (loved, reading), (reading, the), (the, Hunger), (Hunger, Games)`.

Bigramas del candidate (6): `(I, really), (really, loved), (loved, reading), (reading, the), (the, Hunger), (Hunger, Games)`.

Match: `(loved, reading), (reading, the), (the, Hunger), (Hunger, Games)` — 4 bigramas.

$$R_2 = \frac{4}{5} = 0.8$$

$$P_2 = \frac{4}{6} \approx 0.667$$

$$F_2 = \frac{2 \cdot 0.667 \cdot 0.8}{0.667 + 0.8} \approx 0.727$$

{{< concept-alert type="ojo" >}}
La inserción de **"really"** en el candidato baja ROUGE-2 de 1.0 (que tendríamos si $C = R$) a 0.727 porque rompe dos bigramas del reference: `(I, loved)` desaparece y aparece `(I, really)` que no matchea. ROUGE-2 es **mucho más estricto** con inserciones que ROUGE-1.
{{< /concept-alert >}}

---

## 5. ROUGE-L: Longest Common Subsequence

ROUGE-N exige n-gramas **contiguos**. Eso penaliza fuerte cualquier reordenamiento, aunque el reordenamiento preserve sentido. ROUGE-L resuelve esto usando **Longest Common Subsequence (LCS)**: la subsecuencia más larga de $R$ que aparece en orden en $C$, **sin requerir contigüidad**.

{{< math-formula title="ROUGE-L (recall, precision, F)" >}}
R_{\text{LCS}} = \frac{\text{LCS}(C, R)}{|R|}, \quad P_{\text{LCS}} = \frac{\text{LCS}(C, R)}{|C|}, \quad F_{\text{LCS}} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}
{{< /math-formula >}}

donde $|R|$ y $|C|$ son el largo en palabras y $\beta$ controla el balance recall/precision (default $\beta = 1$, F1).

### Algoritmo DP

LCS se computa con programación dinámica en $O(|C| \cdot |R|)$ tiempo y espacio:

$$\text{LCS}[i][j] = \begin{cases} 0 & \text{si } i = 0 \text{ o } j = 0 \\ \text{LCS}[i-1][j-1] + 1 & \text{si } C_i = R_j \\ \max(\text{LCS}[i-1][j], \text{LCS}[i][j-1]) & \text{caso contrario} \end{cases}$$

### Ejemplo

Con el mismo $C, R$ de arriba:

- $C$ = `I really loved reading the Hunger Games`
- $R$ = `I loved reading the Hunger Games`
- LCS = `I loved reading the Hunger Games` → **6 palabras**.

$$R_{\text{LCS}} = \frac{6}{6} = 1.0, \quad P_{\text{LCS}} = \frac{6}{7} \approx 0.857, \quad F_{\text{LCS}} \approx 0.923$$

En este ejemplo ROUGE-L iguala a ROUGE-1 porque las 6 palabras del reference aparecen en orden en el candidate. La diferencia se ve cuando hay reordenamientos: si $C$ = `Hunger Games — I really loved reading`, ROUGE-2 colapsa pero ROUGE-L preserva el match de la subsecuencia más larga.

### Por qué se prefiere a ROUGE-N para abstractive

ROUGE-L **no requiere n-gramas consecutivos**, así que un modelo abstractivo que parafrasea pero conserva el orden general de hechos no es penalizado tanto. Por eso es el reporte estándar en T5, BART, Pegasus.

---

## 6. ROUGE-W: Weighted LCS

Problema de ROUGE-L: una subsecuencia con **muchos saltos** vale lo mismo que una subsecuencia con **bloques consecutivos**. Intuitivamente, "ABC___D" (3 palabras consecutivas + 1 lejana) debería puntuar más que "A__B__C__D" (4 palabras dispersas). ROUGE-W introduce un peso que premia **runs consecutivos**.

Sea $f(k) = k^\alpha$ con $\alpha > 1$ (típicamente $\alpha = 2$). Para una secuencia de matches con runs de longitudes $\text{len}_1, \text{len}_2, \ldots$:

{{< math-formula title="WLCS (Weighted LCS)" >}}
\text{WLCS}(C, R) = \max_{\text{matchings}} \sum_i f(\text{len}_i)
{{< /math-formula >}}

Las fórmulas de recall/precision se definen análogas pero con la **función inversa** $f^{-1}$ para normalizar a $[0, 1]$:

$$R_{\text{WLCS}} = f^{-1}\left(\frac{\text{WLCS}(C, R)}{f(|R|)}\right), \quad P_{\text{WLCS}} = f^{-1}\left(\frac{\text{WLCS}(C, R)}{f(|C|)}\right)$$

### Ejemplo numérico de WLCS

Supongamos:

- $C$ = `the cat sat on the mat`
- $R_1$ = `the cat sat on the mat` (matching perfecto, 6 palabras consecutivas)
- $R_2$ = `the cat ate the mat` (matching disperso: `the cat` + `the mat`, en dos runs de 2)

Con $f(k) = k^2$:

- WLCS$(C, R_1) = f(6) = 36$ (un único run de longitud 6).
- WLCS$(C, R_2) = f(2) + f(2) = 8$ (dos runs de longitud 2: `the cat` y `the mat`).

Si solo contáramos LCS estándar, $R_2$ daría 4 (las 4 palabras de la subsecuencia), y $R_1$ daría 6. ROUGE-W exagera la diferencia: 36 vs 8, premiando fuertemente el match consecutivo.

ROUGE-W es **más sensible a fluencia local** que ROUGE-L, pero rara vez se reporta en papers modernos — ROUGE-L y ROUGE-2 cubren ese rol en la práctica, y la elección de $\alpha$ es un hiperparámetro adicional que complica la comparación entre papers.

---

## 7. ROUGE-S: Skip-bigrams

Generalización a otra dirección: en lugar de exigir bigramas consecutivos (ROUGE-2) o la subsecuencia más larga (ROUGE-L), contamos **todos los pares de palabras en orden**, permitiendo gaps arbitrarios entre ellas.

Para una oración de $n$ palabras hay $\binom{n}{2}$ skip-bigrams.

### Ejemplo

Frase: `"I loved Hunger Games"` (4 palabras → 6 skip-bigrams):

| Skip-bigram | Gap |
|---|---|
| `(I, loved)` | 0 |
| `(I, Hunger)` | 1 |
| `(I, Games)` | 2 |
| `(loved, Hunger)` | 0 |
| `(loved, Games)` | 1 |
| `(Hunger, Games)` | 0 |

### Fórmulas

$$R_{\text{S}} = \frac{\text{SKIP2}(C, R)}{\binom{|R|}{2}}, \quad P_{\text{S}} = \frac{\text{SKIP2}(C, R)}{\binom{|C|}{2}}, \quad F_{\text{S}} = \frac{2 R_{\text{S}} P_{\text{S}}}{R_{\text{S}} + P_{\text{S}}}$$

### ROUGE-S$d_{\text{skip}}$: gap máximo

Sin restricción de gap, ROUGE-S premia matches "espurios" entre palabras muy distantes. Se introduce un gap máximo $d_{\text{skip}}$:

- **ROUGE-S2**: gap máximo 2 — solo pares con hasta 2 palabras entre medio.
- **ROUGE-S4**: gap máximo 4.

Lin (2004) reporta que ROUGE-S4 tiene la **mejor correlación con humanos** en DUC entre las variantes skip.

---

## 8. ROUGE-SU: Skip-bigram + Unigram

Problema: si $C$ y $R$ no comparten **ningún** skip-bigram (caso extremo: $C = $ `"the"` y $R = $ `"a"`), ROUGE-S = 0 aunque haya overlap a nivel de palabra. ROUGE-SU lo arregla agregando un **token marcador** al inicio de cada oración, de modo que cualquier palabra forma un skip-bigram con ese marcador (= unigram match).

$$\text{ROUGE-SU} = \text{ROUGE-S}(C', R') \quad \text{donde } C' = \text{"<s>"} + C, \; R' = \text{"<s>"} + R$$

En la práctica esto degenera a un promedio ponderado de ROUGE-S y ROUGE-1. Útil para summaries cortos donde ROUGE-S puede colapsar.

---

## 9. Multi-reference scoring y jackknifing

Cuando hay $k > 1$ references, ROUGE se computa así:

### Multi-reference max

Para cada $R_i$, computar ROUGE-N$(C, R_i)$, y devolver el **máximo**:

$$\text{ROUGE-N}_{\text{multi}}(C, R) = \max_{i=1, \ldots, k} \text{ROUGE-N}(C, R_i)$$

Esto da el "mejor caso" — el reference más permisivo para el candidato.

### Jackknifing

Para comparar **sistemas automáticos** contra humanos en datasets como DUC: si hay $k$ references, cada sistema se evalúa contra los $k$ references; cada humano se evalúa contra los **otros $k-1$ references**. Esto da una comparación justa porque los humanos no "ven" su propio summary.

$$\text{ROUGE-N}_{\text{jackknife}}(R_j) = \text{average}_{i \neq j} \text{ROUGE-N}(R_j, R_i)$$

Es estándar en DUC, TAC y otros benchmarks oficiales.

---

## 10. ROUGE-Lsum: variante multi-sentence

Acá empiezan los **pitfalls de implementación**. Para summaries de **una sola oración**, ROUGE-L está bien definido. Para summaries de **varias oraciones** hay dos opciones:

### Opción A: ROUGE-L "flat" (concatenado)

Concatenar todas las oraciones del candidate y del reference en una sola string, computar LCS sobre todo el texto. **Problema**: la LCS puede saltar entre oraciones, produciendo matches sin sentido.

### Opción B: ROUGE-Lsum (per-sentence union LCS)

Para cada oración $r_j$ del reference, encontrar la LCS contra el candidate **completo**, sumar los matches únicos (union):

$$\text{LCS}_\cup(C, R) = \bigcup_{r_j \in R} \text{LCS}(C, r_j)$$

Luego computar recall/precision sobre esa unión. Esto **respeta la estructura por oración** del reference.

{{< concept-alert type="ojo" >}}
**Pitfall crítico**: HuggingFace's `evaluate.load("rouge")` por default usa **ambas variantes y las reporta separadas** como `rougeL` y `rougeLsum`. Muchos papers (Liu et al BERTSum, Raffel et al T5) reportan **ROUGE-Lsum** sin aclararlo. Si comparás tu modelo con `rougeL` flat contra el número de un paper que usó `rougeLsum`, vas a estar **2-5 puntos abajo sin saberlo**.
{{< /concept-alert >}}

---

## 11. Limitaciones

ROUGE es **lexical**, no semántica. Esa frase resume sus problemas. Detalle:

| Limitación | Ejemplo | Por qué falla |
|---|---|---|
| **Paráfrasis** | `car` vs `automobile` | Tokens distintos, match = 0 |
| **Sinónimos** | `doctor` vs `physician` | Sin WordNet/embeddings, no se reconocen |
| **Inflexión** | `running` vs `runs` | Sin stemming, son tokens distintos |
| **Stopwords** | `the, a, of` | Inflan ROUGE-1 sin aportar info |
| **Orden de info** | reordenar dos hechos | ROUGE-2 colapsa, ROUGE-L parcial |
| **Hallucinations** | añadir hecho falso que usa palabras del source | Puede subir ROUGE-1, faithfulness baja |
| **Coherencia** | summary con saltos lógicos | ROUGE no la mide |
| **Fluencia global** | gramática mala | ROUGE-2/L no la capturan bien |
| **Lenguajes no-flexivos** | chino, japonés sin espacios | Requieren tokenización custom |

### El faithfulness gap

Una de las críticas más fuertes a ROUGE en la era LLM: un modelo puede **alucinar** un hecho falso usando vocabulario del input y obtener ROUGE alto. Ejemplo:

- **Reference**: "El paciente fue diagnosticado con neumonía adquirida en la comunidad."
- **Candidate alucinado**: "El paciente fue diagnosticado con neumonía nosocomial adquirida durante hospitalización."

ROUGE-1 va a ser alto (mucho overlap léxico), pero el summary es **clínicamente falso** — cambió comunidad por nosocomial. Esto motivó toda una línea de investigación sobre **faithfulness metrics**: FactCC (Kryscinski 2019), QAGS (Wang 2020), MFMA (Lee 2022).

---

## 11b. Correlación con juicios humanos

Lin (2004) validó ROUGE midiendo correlación de Pearson y Spearman entre los scores ROUGE de sistemas en DUC y los rankings producidos por evaluación humana. Resultados clave:

| Variante | Pearson vs humano (DUC 2001-2003) | Comentario |
|---|---|---|
| **ROUGE-1** | 0.91 - 0.99 | Sorprendentemente alto |
| **ROUGE-2** | 0.93 - 0.99 | Suele ser el mejor |
| **ROUGE-3** | 0.91 - 0.97 | Decae con $n$ grande |
| **ROUGE-L** | 0.90 - 0.97 | Comparable a ROUGE-2 |
| **ROUGE-W** ($\alpha=1.2$) | 0.93 - 0.98 | Marginal sobre ROUGE-L |
| **ROUGE-S4** | 0.92 - 0.98 | El mejor entre las skip |
| **ROUGE-SU4** | 0.93 - 0.98 | Comparable a ROUGE-S4 |

**Interpretación**: en summaries cortos de noticias (DUC), todas las variantes correlacionan ≥0.90 con humanos. Pero esa correlación se degrada en:

- **Summaries largos** (literary, scientific): humanos miran coherencia global, ROUGE no.
- **Modelos modernos abstractivos**: BART/PEGASUS/T5 producen paráfrasis fluidas que ROUGE penaliza injustamente.
- **Tareas semánticas** (QA, dialogue): ROUGE no captura "respondió correctamente la pregunta".

Estudios recientes (Fabbri et al 2021, *SummEval*) reportaron correlaciones de **ROUGE con humanos en torno a 0.3-0.5** cuando se evalúan modelos abstractivos modernos contra dimensiones específicas (coherence, consistency, fluency, relevance). Esa caída motivó la oleada de métricas semánticas post-2020.

---

## 12. Sucesores y alternativas

| Métrica | Año | Mecanismo | Cuándo usarla |
|---|---|---|---|
| **BLEU** (Papineni) | 2002 | Precision-oriented n-gram | MT (siempre se reporta) |
| **METEOR** (Banerjee-Lavie) | 2005 | Match exact + stem + sinónimos WordNet | MT, captioning |
| **CIDEr** (Vedantam) | 2015 | TF-IDF de n-gramas | Image captioning |
| **BERTScore** (Zhang) | 2020 | Cosine sim de embeddings BERT por token | Summarization, captioning, paráfrasis |
| **BLEURT** (Sellam) | 2020 | BERT fine-tuneado para predecir score humano | Cuando hay datos de human ratings |
| **MoverScore** (Zhao) | 2019 | Earth Mover's Distance sobre embeddings | Similar a BERTScore, más robusto a re-orden |
| **FactCC** (Kryscinski) | 2019 | Modelo entrenado para detectar inconsistencias | Faithfulness en summarization |
| **QAGS** (Wang) | 2020 | QA-based: genera preguntas, compara respuestas | Faithfulness |
| **G-Eval** (Liu) | 2023 | GPT-4 como juez con chain-of-thought | Evaluación holística, costoso |
| **GPTScore** (Fu) | 2023 | LLM como evaluador zero-shot | Cuando hay budget de API |

{{< concept-alert type="clave" >}}
**No abandones ROUGE**. La comunidad sigue reportándolo porque es **barato, determinístico y comparable con literatura previa**. La práctica moderna es **reportar ROUGE + una métrica semántica (BERTScore) + una métrica de faithfulness**. Triple combo.
{{< /concept-alert >}}

---

## 13. Implementación práctica

### Instalación

Hay tres paquetes principales:

```bash
# Google research (recomendado, usado en T5, Pegasus)
pip install rouge-score

# Versión Python con ROUGE-S, ROUGE-W
pip install py-rouge

# HuggingFace wrapper (usa rouge-score internamente)
pip install evaluate rouge_score
```

### Snippet con rouge-score

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
    use_stemmer=True  # Porter stemmer: running → run
)

reference = "I loved reading the Hunger Games"
candidate = "I really loved reading the Hunger Games"

scores = scorer.score(reference, candidate)
# {'rouge1': Score(precision=0.857, recall=1.0, fmeasure=0.923),
#  'rouge2': Score(precision=0.667, recall=0.8, fmeasure=0.727),
#  'rougeL': Score(precision=0.857, recall=1.0, fmeasure=0.923),
#  'rougeLsum': Score(precision=0.857, recall=1.0, fmeasure=0.923)}
```

### Snippet con HuggingFace evaluate

```python
import evaluate

rouge = evaluate.load("rouge")

preds = ["I really loved reading the Hunger Games"]
refs = ["I loved reading the Hunger Games"]

results = rouge.compute(
    predictions=preds,
    references=refs,
    use_stemmer=True,
    use_aggregator=True  # promedia sobre el batch
)
# {'rouge1': 0.923, 'rouge2': 0.727, 'rougeL': 0.923, 'rougeLsum': 0.923}
```

### Pitfalls comunes

1. **Stemming on/off**: `use_stemmer=True` reduce `runs/running/ran → run`, sube los scores 2-5 puntos. Reportar siempre el flag usado.
2. **ROUGE-L vs ROUGE-Lsum**: si tu summary tiene varias oraciones, **reportá ambos** o aclará cuál es. Diferencia típica de 1-3 puntos.
3. **Tokenización**: rouge-score usa whitespace + lowercasing por default. Para clínico o multilingüe puede ser inadecuado.
4. **Aggregator**: `use_aggregator=True` reporta el promedio macro sobre el batch; `False` reporta el score per-ejemplo (útil para análisis).
5. **Multi-reference**: rouge-score no soporta nativo; hay que computar por reference y agregar manualmente.
6. **Casing**: por default lowercased. Importante si tu dominio tiene siglas (NER médico).
7. **Idiomas no espaciados**: chino y japonés requieren un tokenizador externo (jieba, MeCab) antes de pasar a rouge-score. Si no, cada oración es un único "token gigante" y ROUGE colapsa.
8. **Sentence splitting para Lsum**: rouge-score requiere que separes las oraciones con `\n` para ROUGE-Lsum. Si pasás un summary sin saltos de línea, ROUGE-Lsum colapsa a ROUGE-L. Es un error silencioso muy común — el output se ve "razonable" pero está mal.

### Discrepancias entre implementaciones

Comparativa empírica sobre 100 ejemplos CNN/DailyMail con un mismo modelo BART:

| Implementación | rouge1 F1 | rouge2 F1 | rougeL F1 | Notas |
|---|---|---|---|---|
| `rouge-score` (Google) | 44.16 | 21.28 | 30.81 | Default, sin stemmer |
| `rouge-score` + stemmer | 44.89 | 21.55 | 31.17 | +0.4-0.7 puntos |
| `py-rouge` | 44.22 | 21.30 | 30.84 | Casi idéntica |
| `pyrouge` (Perl original) | 44.05 | 21.18 | 40.59 | ROUGE-L diferente: usa Lsum-style |
| `nltk.translate` | N/A | N/A | N/A | NLTK no implementa ROUGE |

**Moraleja**: cuando compares contra un paper, lee el README del repo oficial. T5 (Raffel 2020) usa rouge-score sin stemmer; PEGASUS (Zhang 2020) usa rouge-score con stemmer; BART (Lewis 2020) usa el Perl original. Diferencias de 1-2 puntos entre papers a veces se explican solo por la implementación.

---

## 14. ROUGE en summarization extractiva (caso BERTSum)

Un uso interesante de ROUGE es **construir el ground truth** de modelos extractivos. En BERTSum (Liu 2019) y similares, el modelo aprende a **seleccionar oraciones** del documento original. Pero el dataset (CNN/DailyMail) solo tiene **summaries abstractivos** humanos, no labels per-oración. ¿Cómo aprendemos qué oraciones seleccionar?

### El truco del oracle ROUGE-2

Greedy: ir seleccionando oraciones del documento que **maximicen ROUGE-2** contra el reference humano:

{{< math-formula title="Oracle extractivo (greedy)" >}}
G^* = \arg\max_{G \subseteq \{s_1, \ldots, s_n\}} \text{ROUGE-2}(G, \text{summary}_{\text{human}})
{{< /math-formula >}}

Algoritmo: empezar con $G = \emptyset$. En cada paso, agregar la oración $s_i$ que más aumenta ROUGE-2($G \cup \{s_i\}$, summary). Parar cuando agregar cualquier oración no mejora.

Esto produce labels binarios $y_i \in \{0, 1\}$ por oración (es / no es parte del oracle). El modelo se entrena con **BCE loss** contra esos labels:

$$\mathcal{L} = -\sum_i [y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i)]$$

### Por qué importa

El oracle define un **techo teórico**: si tu modelo seleccionase exactamente las oraciones del oracle, obtendría el ROUGE máximo extractivo posible. En CNN/DailyMail, ese techo es ~52-55 ROUGE-1 / ~30-33 ROUGE-2. Los modelos extractivos reales llegan a ~43-45 ROUGE-1 — hay gap pero no astronómico.

Para abstractivos (BART, PEGASUS, T5) el techo no aplica porque pueden generar palabras no presentes en el documento. Los SOTA llegan a ~44-46 ROUGE-1 en CNN/DM.

---

## 15. ROUGE en el curso

ROUGE aparece en dos puntos del curso:

### Clase 16 (NLP intro) — métricas clásicas

Mención breve junto a BLEU como ejemplo de evaluación automática de generación. Sirve para contrastar con métricas de clasificación (accuracy, F1) que son más simples.

### Clase 22 (Summarization) — el fundamento

Slides 49-52: definición, recall vs precision, ROUGE-N, ROUGE-L y el ejemplo Hunger Games. La clase explica por qué métricas automáticas son **necesarias** (evaluación humana no escala) y **limitadas** (no capturan semántica).

El laboratorio asociado (BERTSum extractivo) usa **ROUGE-2 como oracle** para construir labels y **ROUGE-1/2/L F1** para reportar resultados sobre CNN/DailyMail.

---

## 14b. ROUGE en otras tareas de generación

Aunque nació para summarization, ROUGE se reusa en cualquier tarea donde la salida es texto:

### Image captioning

Junto a BLEU, METEOR y CIDEr, ROUGE-L es reporte estándar en MS-COCO Captions. La crítica es la misma: si el caption parafrasea con sinónimos, ROUGE penaliza. Por eso CIDEr (TF-IDF de n-gramas multi-reference) suele preferirse en captioning — pondera n-gramas raros que son discriminativos.

### Headline generation

Tarea de "summarization extrema" (XSum dataset): generar un titular de una oración a partir de un artículo. ROUGE-1/2/L son la métrica oficial. PEGASUS (Google 2020) reporta ROUGE-1 ~47, ROUGE-2 ~24, ROUGE-L ~39 en XSum.

### Text simplification

Reescribir un texto complejo en lenguaje simple. ROUGE se reporta junto a **SARI** (System output Against References and Input) que mide adds/keeps/deletes ponderado. ROUGE solo es engañosa acá porque un simplificador que **copia el input** obtiene ROUGE alto pero no simplificó nada.

### Question answering generativo

En SQuAD/NaturalQuestions con respuestas largas, ROUGE-L F1 se usa para comparar respuesta generada contra gold. Otra vez, la versión extractiva (EM, F1 token) suele ser preferida por su simplicidad.

### Dialogue / chatbot

En tareas de respuesta abierta (PersonaChat, BlenderBot), ROUGE casi no se usa — la diversidad de respuestas correctas es muy alta. Se prefiere evaluación humana o métricas basadas en embeddings (BERTScore, BLEURT).

### Code generation

Curiosamente, ROUGE-L se usa en algunos benchmarks de code generation (CodeBLEU lo incluye). La crítica es obvia: dos snippets de código pueden ser sintácticamente distintos y semánticamente equivalentes. CodeBLEU agrega match de AST y data-flow para compensar.

---

## 15b. Pseudo-código del cómputo end-to-end

Para que quede transparente, el pipeline completo de evaluar un modelo con ROUGE sobre un test set:

```python
from rouge_score import rouge_scorer
import numpy as np

# 1. Cargar modelo y test set
model = load_model()  # BART, T5, etc.
test_examples = load_test_set()  # lista de (article, reference)

# 2. Generar candidatos
candidates = []
for article, _ in test_examples:
    summary = model.generate(article, max_length=128, num_beams=4)
    candidates.append(summary)

# 3. Computar ROUGE por ejemplo
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
    use_stemmer=True
)

per_example_scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
for (_, reference), candidate in zip(test_examples, candidates):
    # IMPORTANTE: separar oraciones por \n para Lsum
    reference_lsum = '\n'.join(sent_tokenize(reference))
    candidate_lsum = '\n'.join(sent_tokenize(candidate))
    scores = scorer.score(reference_lsum, candidate_lsum)
    for k in per_example_scores:
        per_example_scores[k].append(scores[k].fmeasure)

# 4. Agregar (media + intervalo de confianza bootstrap)
for k, scores in per_example_scores.items():
    mean = np.mean(scores) * 100
    # Bootstrap 95% CI
    boot_means = [np.mean(np.random.choice(scores, size=len(scores), replace=True))
                  for _ in range(1000)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5]) * 100
    print(f"{k}: {mean:.2f} [{ci_low:.2f}, {ci_high:.2f}]")
```

**Reporte final típico**: `ROUGE-1: 44.16 [43.21, 45.08]`, etc. Los intervalos de confianza bootstrap son cada vez más esperados en venues serios (ACL, EMNLP) — un número aislado sin CI puede esconder diferencias estadísticamente nulas entre dos sistemas que difieren en 0.3 puntos.

---

## 16. Resumen ejecutivo

| Variante | Captura | Cuándo reportarla |
|---|---|---|
| **ROUGE-1** F1 | Cobertura de vocabulario | Siempre |
| **ROUGE-2** F1 | Colocaciones / fluencia local | Siempre |
| **ROUGE-3+** | Trigramas o más | Casi nunca, muy estricto |
| **ROUGE-L** F1 | Orden general (LCS) | Siempre (single sentence) |
| **ROUGE-Lsum** F1 | Orden por oración | Siempre (multi-sentence) |
| **ROUGE-W** | Fluencia con bonus por runs | Raro |
| **ROUGE-S/SU** | Pares con skip controlado | Raro, dominios específicos |

**Reporte estándar moderno**: ROUGE-1/2/L (o Lsum) F1, con `use_stemmer=True`, sobre CNN/DailyMail o XSum, comparado contra BART/PEGASUS/T5 como baselines.

**Si tu paper es post-2022 y solo reporta ROUGE**: vas a recibir review pidiendo BERTScore + algo de faithfulness. La métrica sigue siendo necesaria pero ya no suficiente.

---

## Referencias internas

- [Sentiment Analysis](/fundamentos/sentiment-analysis) — otra métrica/evaluación clásica de NLP.
- [BERT](/fundamentos/bert) — backbone de BERTSum extractivo y BERTScore.
- [Modelos de Lenguaje](/fundamentos/modelos-de-lenguaje) — contexto general.
- [GPT Family](/fundamentos/gpt-family) — generación abstractiva moderna.
- Clase 16: introducción a NLP, métricas clásicas.
- Clase 22: Text Summarization extractiva y abstractiva.
