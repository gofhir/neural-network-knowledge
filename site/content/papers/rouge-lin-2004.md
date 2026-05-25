---
title: "ROUGE (Recall-Oriented Understudy for Gisting Evaluation)"
weight: 113
math: true
---

{{< paper-card
    title="ROUGE: A Package for Automatic Evaluation of Summaries"
    authors="Chin-Yew Lin"
    year="2004"
    venue="ACL Workshop on Text Summarization Branches Out"
    pdf="/papers/rouge-lin-2004.pdf"
    arxiv="" >}}
Define una familia de metricas para evaluar resumenes automaticos contra resumenes de referencia humanos: **ROUGE-N** (n-gram overlap), **ROUGE-L** (longest common subsequence), **ROUGE-W** (weighted LCS) y **ROUGE-S / ROUGE-SU** (skip-bigram). Validada contra juicios humanos en DUC 2001-2003 via Pearson, Spearman y Kendall con bootstrap. Veinte anos despues sigue siendo la metrica de facto reportada en summarization, parafraseo, simplificacion y dialogo abstractivo.
{{< /paper-card >}}

---

## El problema

Antes de 2004 la evaluacion de sistemas de summarization se hacia **enteramente con jueces humanos**. La conferencia de referencia era el **Document Understanding Conference (DUC)** del NIST, donde cada ronda requeria estimadamente **3.000 horas-trabajador** solo para una evaluacion basica de cobertura de contenido y calidad linguistica. Cada iteracion sobre un algoritmo significaba semanas de espera y presupuesto que pocos grupos podian sostener.

La metrica analoga en traduccion automatica, **BLEU** (Papineni 2002), ya era estandar pero tiene una orientacion incompatible: es **precision-oriented**. BLEU pregunta "que fraccion de los n-gramas del candidato aparece en la referencia", lo que en MT tiene sentido porque importa que el sistema no invente palabras. En summarization la pregunta dual es mas critica: **"que fraccion del contenido del resumen humano aparece en el resumen automatico"**, es decir **recall-oriented**. Si el sistema omite informacion clave, debe penalizarse.

De alli el nombre que Lin elige deliberadamente como contrapunto: **R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation.

---

## ROUGE-N

ROUGE-N mide el recall de n-gramas: que fraccion de los n-gramas presentes en las referencias aparecen tambien en el candidato.

$$
\text{ROUGE-N} = \frac{\displaystyle\sum_{S \in R} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\displaystyle\sum_{S \in R} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}
$$

donde $n$ es la longitud del n-grama, $\text{Count}(\text{gram}_n)$ cuenta n-gramas en la referencia, y $\text{Count}_{\text{match}}$ es el numero **clipped** de co-ocurrencias entre candidato y referencia (el clipping evita inflar el numerador con repeticiones del candidato).

La clave conceptual es que **el denominador suma sobre las referencias**, no sobre el candidato. Esa eleccion convierte la metrica en recall: el sistema es castigado cuando omite contenido que los humanos consideraron importante.

Las dos variantes mas reportadas son:

- **ROUGE-1**: fluidez lexica basica, sensible a vocabulario y palabras de contenido.
- **ROUGE-2**: coherencia local de bigramas, mas exigente con el ordenamiento.

ROUGE-3 y superiores se degradan rapido en resumenes cortos (no hay suficientes 3-gramas) y casi no se reportan en la practica.

---

## Ejemplo paso a paso

Consideremos el caso canonico del curso (slides 49-52):

- **Referencia**: `I really loved reading the Hunger Games` (7 unigramas)
- **Candidato**: `I loved reading the Hunger Games` (6 unigramas)

### ROUGE-1

Unigramas que coinciden: `I`, `loved`, `reading`, `the`, `Hunger`, `Games` = **6 matches**.

$$
R_1 = \frac{6}{7} \approx 0.857 \qquad P_1 = \frac{6}{6} = 1.000
$$

$$
F_1 = \frac{2 \cdot 0.857 \cdot 1.000}{0.857 + 1.000} \approx 0.923
$$

(El planteo simplificado del slide ignora `really` en el numerador y usa $R = 6/6$, $P = 6/7$, $F = 12/13 \approx 0.923$, que da el mismo F-score por simetria.)

### ROUGE-2

Bigramas de la referencia (6): `I really`, `really loved`, `loved reading`, `reading the`, `the Hunger`, `Hunger Games`.
Bigramas del candidato (5): `I loved`, `loved reading`, `reading the`, `the Hunger`, `Hunger Games`.

Matches: `loved reading`, `reading the`, `the Hunger`, `Hunger Games` = **4 matches**.

$$
R_2 = \frac{4}{6} \approx 0.667 \qquad P_2 = \frac{4}{5} = 0.800
$$

ROUGE-2 baja respecto de ROUGE-1 porque al cambiar `really enjoyed` por `loved` se rompen dos bigramas adyacentes, no solo un unigrama. Esa sensibilidad al orden es lo que hace que ROUGE-2 sea preferida sobre ROUGE-1 cuando hay suficiente texto.

---

## ROUGE-L

ROUGE-N exige **contiguidad estricta** de n-gramas, lo que penaliza paraphrases legitimos donde se preserva el orden pero se intercalan palabras. ROUGE-L resuelve esto con la **longest common subsequence (LCS)**: la subsecuencia mas larga que aparece en ambos textos preservando el orden pero sin exigir adyacencia.

Sean $X$ (referencia) de longitud $m$, $Y$ (candidato) de longitud $n$, y $\text{LCS}(X,Y)$ la longitud de la LCS. Entonces:

$$
R_{\text{LCS}} = \frac{\text{LCS}(X, Y)}{m} \qquad P_{\text{LCS}} = \frac{\text{LCS}(X, Y)}{n}
$$

$$
F_{\text{LCS}} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}
$$

En DUC, Lin reporta $\beta \to \infty$ por defecto, lo que reduce $F_{\text{LCS}}$ esencialmente a recall. La LCS se computa con DP estandar en $O(mn)$.

**Ejemplo** con la misma referencia y candidato anteriores: LCS = `I reading the Hunger Games`, longitud 5 (saltando `really loved` en la referencia y `loved` en el candidato).

$$
R_{\text{LCS}} = \frac{5}{7} \approx 0.714 \qquad P_{\text{LCS}} = \frac{5}{6} \approx 0.833
$$

Con $\beta = 1$: $F_{\text{LCS}} \approx 0.769$.

### Limitacion clave de LCS

LCS cuenta **una sola** subsecuencia, la mas larga. Si el candidato preserva fragmentos del orden original pero los rearma (ej: `"the gunman police killed"` versus `"police killed the gunman"`), LCS escoge solo uno de los fragmentos y no premia la presencia simultanea de varios. Esa limitacion motiva las dos variantes siguientes.

---

## ROUGE-W

ROUGE-W (Weighted LCS) corrige otro sesgo: LCS trata igual a una subsecuencia **consecutiva** y otra **dispersa**. Considerar:

- $X$: `[A B C D E F G]`
- $Y_1$: `[A B C D H I K]` (match consecutivo `A B C D`)
- $Y_2$: `[A H B K C I D]` (match disperso, igual longitud 4)

Ambos tienen LCS = 4, mismo ROUGE-L. Pero $Y_1$ es claramente mejor porque preserva contiguidad.

ROUGE-W introduce una funcion de peso $f(k)$ donde $k$ es la longitud del bloque consecutivo, con la propiedad clave de **superaditividad**:

$$
f(x + y) > f(x) + f(y) \quad \forall x, y > 0
$$

Lin usa $f(k) = k^{1.2}$ por defecto. El DP se extiende manteniendo una tabla auxiliar $w(i,j)$ con la longitud del bloque consecutivo que termina en $(i,j)$. Al extender un bloque, el score crece en $f(k+1) - f(k)$ —mayor que $f(1)$— lo que premia la extension sobre comenzar un bloque nuevo. La normalizacion a $[0,1]$ requiere que $f$ tenga inversa cerrada $f^{-1}$.

En la practica, ROUGE-W se reporta poco porque agrega un hiperparametro ($\alpha$) y porque las implementaciones populares no la priorizan.

---

## ROUGE-S y ROUGE-SU

Un **skip-bigram** es cualquier par ordenado de palabras $(w_i, w_j)$ con $i < j$ en la misma oracion, permitiendo gaps arbitrarios. Para una oracion de $m$ palabras hay $\binom{m}{2}$ skip-bigrams.

$$
R_{\text{skip2}} = \frac{\text{SKIP2}(X, Y)}{\binom{m}{2}} \qquad P_{\text{skip2}} = \frac{\text{SKIP2}(X, Y)}{\binom{n}{2}}
$$

donde $\text{SKIP2}(X,Y)$ es el numero de skip-bigrams compartidos.

Para controlar ruido, Lin permite un **maximum skip distance** $d_{\text{skip}}$:

- $d_{\text{skip}} = 0$: equivalente a ROUGE-2.
- $d_{\text{skip}} = 4$: hasta 4 palabras de distancia (ROUGE-S4).
- $d_{\text{skip}} = \infty$: sin restriccion (ROUGE-S*).

**Problema de ROUGE-S puro**: si el candidato es la inversa exacta de la referencia (`"gunman the killed police"` vs `"police killed the gunman"`), todos los unigramas coinciden pero **ningun skip-bigram en orden** —el score colapsa a 0, lo que es indeseable porque al menos el vocabulario es correcto.

**ROUGE-SU** resuelve esto agregando unigramas como unidad de cuenta adicional. Operacionalmente equivale a prepender un marcador `BOS` (begin-of-sentence) y aplicar ROUGE-S; cada unigrama $w_i$ forma un skip-bigram $(\text{BOS}, w_i)$. La consecuencia: ROUGE-SU **nunca da cero** si hay al menos un unigrama compartido, lo que la hace mas robusta como metrica de fallback. **ROUGE-SU4** es la variante mas reportada.

---

## Validacion en DUC 2001-2003

Lin evalua 17 variantes de ROUGE contra juicios humanos del DUC en tres tareas:

- **Single-document, ~100 palabras** (DUC 2001, 2002).
- **Very short summaries, ~10 palabras tipo headline** (DUC 2003).
- **Multi-document, 10/50/100/200/400 palabras**.

La metrica de validacion es la **correlacion de Pearson** entre el score ROUGE promedio de cada sistema y el score humano del mismo sistema, con intervalos de confianza al 95% via **bootstrap resampling**.

### Resultados clave (single-document)

| Metrica | DUC 2001 (1 ref) | DUC 2001 (3 refs) | DUC 2002 (1 ref) | DUC 2002 (2 refs) |
| --- | ---: | ---: | ---: | ---: |
| ROUGE-1 | 0.76 | 0.80 | 0.98 | 0.98 |
| **ROUGE-2** | 0.84 | 0.87 | 0.99 | 0.99 |
| **ROUGE-L** | 0.83 | 0.86 | 0.99 | 0.99 |
| **ROUGE-W-1.2** | 0.85 | 0.87 | 0.99 | 0.99 |
| **ROUGE-S4** | 0.84 | 0.87 | 0.99 | 0.99 |
| ROUGE-SU4 | 0.84 | 0.87 | 0.99 | 0.99 |

### Resultados (very short summaries, DUC 2003)

| Metrica | CASE (4 refs) | STEM (4 refs) | STOP (4 refs) |
| --- | ---: | ---: | ---: |
| **ROUGE-1** | 0.95 | 0.95 | 0.90 |
| ROUGE-2 | 0.76 | 0.75 | 0.77 |
| **ROUGE-L** | 0.96 | 0.96 | 0.96 |
| **ROUGE-SU4** | 0.96 | 0.95 | 0.97 |

### Conclusiones operacionales del paper

1. **Single-doc**: ROUGE-2, ROUGE-L, ROUGE-W, ROUGE-S/SU4 estan todas en el mismo top.
2. **Very short summaries**: ROUGE-1, ROUGE-L y ROUGE-SU4 dominan; ROUGE-2 degrada (pocos bigramas).
3. **Multi-doc**: correlaciones bajan a 0.70-0.85 por menor numero de muestras; ROUGE-1/2/SU4 con stopword removal son razonables; ROUGE-L y ROUGE-W no funcionan bien aca.
4. **Stopword removal** generalmente ayuda.
5. **Multiples referencias** generalmente ayudan, especialmente combinadas con **jackknifing**: se computa el mejor score sobre cada subconjunto de $M-1$ referencias y se promedia.

---

## Limitaciones

Veinte anos despues, las debilidades de ROUGE son material de docenas de papers criticos:

- **Overlap lexico, no semantico**: "automobile" vs "car" recibe cero credito. Castiga sistemicamente a sistemas abstractivos versus extractivos.
- **Dominancia de stopwords**: ROUGE-1 se infla con `the`, `a`, `of`. La recomendacion de stopword removal de Lin se ignora habitualmente.
- **No mide fluidez ni gramaticalidad**: un resumen agramatical con las palabras correctas puede tener ROUGE alto.
- **No detecta alucinaciones (faithfulness gap)**: si el candidato dice "30 anos" donde la fuente dice "60", ROUGE no penaliza la falsedad si la cifra no esta en las referencias. Critico en la era LLM.
- **Sesgo hacia extraccion**: los humanos tambien copian frases del documento, lo que ancla la metrica al lexico fuente.
- **Idiomas no flexivos**: chino, japones y tailandes requieren tokenizacion custom; decisiones de segmentacion cambian el score 2-5 puntos.
- **Single-reference como norma practica**: CNN/DailyMail, XSum y la mayoria de benchmarks tienen una sola referencia, lo que reduce el upper bound y desfavorece resumenes legitimamente distintos.
- **No mide informatividad**: asume que la referencia es la mejor representacion; no premia capturar lo realmente importante.

---

## Sucesores

La busqueda de mejores metricas es un subcampo activo desde 2005:

| Metrica | Anho | Idea |
| --- | --- | --- |
| **METEOR** | 2005 | Exact + stemming + sinonimos (WordNet) + paraphrasing; penaliza fragmentation |
| **MoverScore** | 2019 | Earth Mover's Distance sobre embeddings contextualizados |
| **BERTScore** | 2020 | Similitud coseno entre embeddings BERT; precision/recall/F1 |
| **BLEURT** | 2020 | Regresor sobre BERT entrenado en ratings humanos WMT |
| **FactCC / SummaC** | 2020-22 | NLI a nivel oracion para detectar contradicciones |
| **QAEval** | 2021 | Genera preguntas sobre el resumen, las responde con el documento |
| **G-Eval** | 2023 | LLM como juez (GPT-4 chain-of-thought sobre coherencia/fluidez/relevancia) |

A pesar de toda esta proliferacion, **ROUGE-1, ROUGE-2 y ROUGE-L siguen siendo las metricas obligatorias**. Razones: reproducibilidad (deterministico, sin modelos), costo ($O(mn)$ vs llamadas a LLM), historico (comparabilidad con miles de papers previos) y conservadurismo de los reviewers. Las metricas modernas se reportan como **complemento**, raramente como reemplazo.

---

## Implementacion practica

### Instalacion

```bash
pip install rouge-score
```

`rouge-score` es la re-implementacion pura en Python de Google (2020), referencia moderna que reemplazo a `pyrouge` (wrapper del Perl original de ISI/USC).

### Uso directo

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

ref = "I really loved reading the Hunger Games"
cand = "I loved reading the Hunger Games"

scores = scorer.score(ref, cand)
# {'rouge1': Score(precision=1.0, recall=0.857, fmeasure=0.923),
#  'rouge2': Score(precision=0.8, recall=0.667, fmeasure=0.727),
#  'rougeL': Score(precision=1.0, recall=0.857, fmeasure=0.923),
#  'rougeLsum': Score(precision=1.0, recall=0.857, fmeasure=0.923)}
```

### Via HuggingFace

```python
import evaluate
rouge = evaluate.load("rouge")
result = rouge.compute(predictions=[cand], references=[ref])
```

### ROUGE-L vs ROUGE-Lsum

En implementaciones modernas:

- **ROUGE-L** se computa a nivel oracion y se promedia.
- **ROUGE-Lsum** aplica el union LCS a nivel de resumen completo (Lin 2004, seccion 3.2).

Para resumenes largos los dos scores difieren significativamente. Papers de summarization deben reportar ambos para evitar ambiguedad.

### Reproducibilidad

Pequenas diferencias en tokenizacion, lowercasing y handling de stopwords pueden mover el score 1-2 puntos, lo que ha generado debates sobre como reportar ROUGE de forma comparable. La recomendacion actual es citar la version exacta del paquete y los parametros usados.

---

## Por que importa hoy

A pesar de la proliferacion de metricas neurales modernas, ROUGE sigue siendo:

1. **La metrica de facto en summarization**. Todo paper de CNN/DailyMail, XSum, PubMed, arXiv-summarization reporta ROUGE-1/2/L como minimo.
2. **Reward signal para entrenamiento**. **BERTSum** (Liu & Lapata, 2019) usa **ROUGE como criterio de construccion del oracle target** para entrenamiento supervisado de modelos extractivos: el oracle es el subconjunto de oraciones que maximiza ROUGE contra el resumen de referencia.
3. **Loss auxiliar en summarization abstractiva**. PEGASUS, BART, T5 reportan ROUGE durante entrenamiento como sanity check y para selection de checkpoint.
4. **Benchmark universal**. Leaderboards de Papers With Code, GEM y similares reportan ROUGE por compatibilidad con literatura previa.
5. **Educacion**. Su simplicidad matematica la hace ideal para ensenar evaluacion automatica de NLP en cursos universitarios y MOOCs.

---

## Conexion con la clase 22

La clase 22 del curso aborda **evaluacion de modelos NLP generativos** con foco en summarization. Los slides 49-52 cubren explicitamente ROUGE:

- **Slide 49**: motivacion recall-oriented vs precision-oriented (BLEU).
- **Slide 50**: definicion ROUGE-1 y ROUGE-2 con el ejemplo `I really loved reading the Hunger Games`.
- **Slide 51**: ROUGE-L con LCS sobre la misma referencia.
- **Slide 52**: tabla comparativa de variantes y cuando usar cada una.

ROUGE aparece tambien en otros puntos de la clase:

- **Slide 29** (BERTSum extractivo): el modelo se entrena con un oracle construido a partir de ROUGE-1/ROUGE-2 contra la referencia. Sin ROUGE no hay BERTSum.
- **Tabla de resultados CNN/DailyMail**: PEGASUS, BART y T5 se comparan en ROUGE-1/2/L como columnas estandar.
- **Discusion de faithfulness**: ROUGE no detecta alucinaciones, motivando la introduccion de FactCC y SummaC en la segunda mitad de la clase.

Como fundamento transversal, ROUGE conecta con:

- **N-gramas** (modelos de lenguaje, Clase 12): la unidad basica de ROUGE-N.
- **Programacion dinamica**: estructura algoritmica de LCS.
- **Estadistica inferencial**: jackknifing, bootstrap resampling, correlacion de Pearson.
- **Embeddings contextualizados** (Clase 20): los sucesores semanticos BERTScore, BLEURT.

---

## Notas y enlaces

**Fundamentos transversales:**

- [Fundamento: ROUGE como metrica de evaluacion](/fundamentos/rouge-metric)
- [Fundamento: Text summarization extractiva y abstractiva](/fundamentos/text-summarization)
- [Fundamento: Embeddings contextualizados (BERT, ELMo)](/fundamentos/embeddings-contextualizados)

**Papers relacionados:**

- [BERTSum (Liu 2019)](/papers/bertsum-liu-2019) — usa ROUGE para construir el oracle de entrenamiento.
- [T5 (Raffel et al. 2020)](/papers/t5-raffel-2020) — reporta ROUGE-1/2/L como metrica primaria en summarization.
- [BART (Lewis et al. 2020)](/papers/bart-lewis-2020) — modelo seq2seq evaluado en CNN/DailyMail con ROUGE.
- [PEGASUS (Zhang et al. 2020)](/papers/pegasus-zhang-2020) — pre-entrenamiento con gap-sentences seleccionadas via ROUGE.
- [BERT (Devlin et al. 2018)](/papers/bert-devlin-2018) — base de BERTScore, sucesor semantico de ROUGE.

**Clase del curso:**

- [Clase 22: Summarization](/clases/clase-22)

**Referencias del paper:**

- Papineni et al. 2002 — BLEU, precedente directo.
- Lin & Hovy 2003 (HLT-NAACL) — preliminar de Lin que motiva ROUGE.
- Banerjee & Lavie 2005 — METEOR, primer sucesor importante.
- Zhang et al. 2020 — BERTScore, sucesor semantico moderno.
- Kryscinski et al. 2019 — FactCC, critica de faithfulness.
- Liu et al. 2023 — G-Eval, LLM-as-a-judge.
