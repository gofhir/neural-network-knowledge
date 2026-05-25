---
title: "ROUGE: A Package for Automatic Evaluation of Summaries"
slug: rouge-lin-2004
authors:
  - Chin-Yew Lin
year: 2004
venue: "Text Summarization Branches Out — ACL Workshop"
type: paper
clase: 22
tags:
  - summarization
  - evaluation
  - rouge
  - n-gram
  - lcs
  - skip-bigram
  - duc
  - nlp-metrics
pdf: Lin-ROUGE-2004.pdf
date: 2026-05-25
---

# ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004)

## Resumen ejecutivo

El paper **"ROUGE: A Package for Automatic Evaluation of Summaries"** de Chin-Yew Lin (Information Sciences Institute, USC), presentado en el workshop *Text Summarization Branches Out* de ACL 2004, introduce la métrica más influyente y duradera del campo de evaluación automática de resúmenes (text summarization). El acrónimo **ROUGE** significa *Recall-Oriented Understudy for Gisting Evaluation*, contrapuesto deliberadamente al ya establecido BLEU (Bilingual Evaluation Understudy, Papineni 2002) de traducción automática.

El aporte central de Lin es operacionalizar la intuición de que un resumen automático es bueno si **comparte muchos n-gramas (o subsecuencias, o pares de palabras)** con uno o varios resúmenes de referencia escritos por humanos. A partir de esa idea define una **familia de cuatro variantes** —ROUGE-N, ROUGE-L, ROUGE-W y ROUGE-S, esta última extendida como ROUGE-SU— que cubren respectivamente: overlap de n-gramas, longest common subsequence, weighted LCS y skip-bigram co-occurrence. Cada métrica viene con un F-score (precision + recall) parametrizable por un coeficiente $\beta$, y un esquema de manejo de **múltiples referencias** vía jackknifing.

La validación experimental compara las 17 variantes de ROUGE contra **juicios humanos del Document Understanding Conference (DUC) 2001, 2002 y 2003** usando coeficientes de correlación de Pearson, Spearman y Kendall, con intervalos de confianza vía bootstrap resampling. Las conclusiones empíricas: (i) ROUGE-2, ROUGE-L, ROUGE-W y ROUGE-S correlacionan muy bien en single-document summarization; (ii) ROUGE-1, ROUGE-L y ROUGE-SU4/9 son excelentes para very-short summaries (headlines); (iii) en multi-document la correlación es menor pero mejora con remoción de stopwords; (iv) usar múltiples referencias mejora la correlación.

Veinte años después, **ROUGE sigue siendo la métrica de facto** reportada en prácticamente todo paper de summarization, parafraseo, simplificación de texto y diálogo abstractivo, sobreviviendo incluso la llegada de métricas semánticas modernas como BERTScore, BLEURT y G-Eval. Su éxito tiene tanto que ver con su simplicidad matemática como con la inexistencia de un sucesor consensuado.

---

## 1. Contexto histórico

### Pre-2004: la evaluación de resúmenes era manual

Antes de 2004, la evaluación de sistemas de summarization se hacía **enteramente con jueces humanos**. La conferencia de referencia del campo era el **DUC (Document Understanding Conference)** organizada por el NIST de EE.UU. desde 2001. En DUC se evaluaban dimensiones como:

- **Content coverage**: ¿el resumen captura las unidades de información clave del documento fuente?
- **Coherence**: ¿las oraciones fluyen lógicamente?
- **Conciseness**: ¿el resumen es breve sin redundancias?
- **Grammaticality**: ¿la sintaxis es correcta?
- **Readability**: ¿se puede leer cómodamente?

Lin cita en la introducción que una evaluación simple manual sobre unas pocas preguntas de calidad lingüística y cobertura de contenido en DUC **requería más de 3,000 horas de esfuerzo humano**. Esto hacía imposible iterar rápido sobre algoritmos de summarization: cada cambio significativo en un modelo requería re-evaluación humana.

### BLEU (Papineni 2002): el precedente desde MT

En traducción automática, Papineni et al. (2001/2002) habían publicado **BLEU (Bilingual Evaluation Understudy)**, una métrica basada en n-gram overlap entre la traducción candidata y traducciones de referencia humanas. BLEU se popularizó rápidamente porque (i) era barato computarla y (ii) correlacionaba razonablemente con juicios humanos. Pero BLEU tiene una orientación clave: es **precision-oriented** —pregunta "¿qué fracción de n-gramas del candidato aparece en la referencia?"— porque en MT lo importante es que el candidato no introduzca palabras inventadas.

En summarization la pregunta inversa es más importante: **"¿qué fracción del contenido del resumen humano aparece en el resumen automático?"**. Es decir, **recall-oriented**. Si un resumen automático omite información clave de la referencia, está fallando. De ahí el contraste explícito que Lin enfatiza desde el título: **Recall-Oriented Understudy**.

### Antecedentes inmediatos

- **Saggion et al. (2002)** propusieron tres métricas content-based para evaluación de resúmenes: cosine similarity, unit overlap (uni/bigram) y longest common subsequence. Pero no validaron correlación con juicios humanos.
- **Lin & Hovy (2003)** mostraron en HLT-NAACL que adaptar BLEU a summarization (n-gram co-occurrence) producía métricas que correlacionaban con humanos.

El paper de 2004 sistematiza y expande este trabajo en un **paquete unificado, libre, replicable** —el ROUGE evaluation package— que se vuelve estándar del campo.

---

## 2. El problema formal

Sea:

- $C$ = candidate summary, un texto generado por un sistema automático.
- $R = \{R_1, R_2, \ldots, R_k\}$ = conjunto de **reference summaries** escritos por humanos para el mismo documento (o cluster de documentos).

Queremos definir una función $\text{score}(C, R) \in [0, 1]$ tal que:

1. **Correlacione con evaluación humana**: si humanos juzgan $C_a$ mejor que $C_b$, el score debería reflejarlo.
2. **Sea barato de computar**: idealmente $O(|C| \cdot |R|)$ o mejor.
3. **Sea determinístico y reproducible**: no requiere modelos entrenados.
4. **Maneje múltiples referencias**: porque distintos humanos producen resúmenes válidos diferentes.

La **asunción de fondo** —admitidamente fuerte— es que si $C$ comparte muchas unidades léxicas (palabras, secuencias, pares) con $R$, probablemente captura información similar y es por tanto un buen resumen. Esta asunción se rompe en parafraseos, sinónimos y reordenamientos —limitación que la comunidad termina pagando caro 15+ años después— pero a 2004 es operacionalmente razonable.

---

## 3. ROUGE-N: N-gram co-occurrence

### Definición matemática

ROUGE-N mide el **recall de n-gramas**: qué fracción de los n-gramas presentes en las referencias aparecen también en el candidato. Formalmente:

$$
\text{ROUGE-N} = \frac{\displaystyle\sum_{S \in \{R\}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\displaystyle\sum_{S \in \{R\}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}
$$

donde:

- $n$ es la longitud del n-grama ($n=1$ para ROUGE-1, $n=2$ para ROUGE-2, etc.).
- $\text{Count}(\text{gram}_n)$ es el número de veces que un n-grama aparece en la referencia $S$.
- $\text{Count}_{\text{match}}(\text{gram}_n)$ es el número máximo de co-ocurrencias entre candidato y referencia (clipping para evitar inflar con repeticiones).

### Por qué es recall-oriented

El **denominador suma sobre las referencias**, no sobre el candidato. Esto es la diferencia clave con BLEU. Interpretación: "del total de n-gramas que los humanos creen importantes (las referencias), ¿cuántos están en mi resumen automático?". Si el sistema omite contenido, el numerador cae pero el denominador no, y la métrica baja correctamente.

### Múltiples referencias: dos esquemas

Lin discute dos formas de combinar múltiples referencias:

**Esquema 1 (formula original)**: el denominador y el numerador suman sobre todas las referencias. Esto da más peso a n-gramas que aparecen en muchas referencias —favoreciendo consenso— porque cuentan múltiples veces.

**Esquema 2 (pairwise + max)**: computar ROUGE-N entre candidato y cada referencia por separado, tomar el máximo:

$$
\text{ROUGE-N}_{\text{multi}} = \arg\max_i \text{ROUGE-N}(R_i, C)
$$

Para estabilidad estadística, Lin **adopta jackknifing**: dado $M$ referencias, computa el mejor score sobre $M$ subconjuntos de $M-1$ referencias y promedia. Esto permite además **comparar sistemas con humanos**, porque cada referencia humana puede ser evaluada contra las $M-1$ restantes.

### Ejemplo paso a paso (ROUGE-1)

Reference: `"the gunman killed the police officer"` — 6 unigramas.
Candidate: `"police shot the gunman"` — 4 unigramas.

Unigramas de la referencia: {the, gunman, killed, the, police, officer}, con cuentas: the(2), gunman(1), killed(1), police(1), officer(1). Total = 6.

Matches en candidato: the(1, clip a 1 del candidato), gunman(1), police(1). Total matches = 3.

$$
\text{ROUGE-1} = \frac{3}{6} = 0.50
$$

### Ejemplo (ROUGE-2)

Bigramas de la referencia: {the gunman, gunman killed, killed the, the police, police officer} = 5 bigramas.
Bigramas del candidato: {police shot, shot the, the gunman} = 3 bigramas.

Matches: `the gunman` aparece en ambos → 1 match.

$$
\text{ROUGE-2} = \frac{1}{5} = 0.20
$$

### Caso del slide 49-52 del curso

Reference: `"I really enjoyed reading the Hunger Games"` — 7 unigramas.
Candidate: `"I loved reading the Hunger Games"` — 6 unigramas.

Unigramas: I, reading, the, Hunger, Games coinciden (5 matches). Total referencia: 7.

$$
\text{ROUGE-1 recall} = \frac{5}{7} \approx 0.714
$$
$$
\text{ROUGE-1 precision} = \frac{5}{6} \approx 0.833
$$

Para ROUGE-2 bigramas:
Referencia: {I really, really enjoyed, enjoyed reading, reading the, the Hunger, Hunger Games} = 6 bigramas.
Candidato: {I loved, loved reading, reading the, the Hunger, Hunger Games} = 5 bigramas.

Matches: {reading the, the Hunger, Hunger Games} = 3.

$$
\text{ROUGE-2 recall} = \frac{3}{6} = 0.5
$$

---

## 4. ROUGE-L: Longest Common Subsequence

### Motivación

ROUGE-N exige **contigüidad estricta** de n-gramas. Pero un buen resumen puede preservar el orden de las palabras importantes sin que aparezcan adyacentes. Ejemplo: si la referencia dice "police killed the gunman" y el candidato dice "police killed gunman", `the gunman` no es bigrama exacto pero la subsecuencia `police killed gunman` se preserva en orden.

La **longest common subsequence (LCS)** captura precisamente esto. Una secuencia $Z = [z_1, \ldots, z_n]$ es subsecuencia de $X = [x_1, \ldots, x_m]$ si existen índices estrictamente crecientes $[i_1, \ldots, i_k]$ tales que $x_{i_j} = z_j$ para todo $j$. La LCS no exige contigüidad, solo **preservación del orden**.

### Definición formal

Sean $X$ (referencia) de longitud $m$ y $Y$ (candidato) de longitud $n$. Sea $\text{LCS}(X,Y)$ la longitud de la subsecuencia común más larga.

**Recall y precision LCS-based**:

$$
R_{\text{LCS}} = \frac{\text{LCS}(X, Y)}{m}
$$

$$
P_{\text{LCS}} = \frac{\text{LCS}(X, Y)}{n}
$$

**F-measure** (definición de ROUGE-L):

$$
F_{\text{LCS}} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}
$$

donde $\beta = P_{\text{LCS}}/R_{\text{LCS}}$ en el punto donde $\partial F_{\text{LCS}}/\partial R_{\text{LCS}} = \partial F_{\text{LCS}}/\partial P_{\text{LCS}}$.

**En DUC**, Lin reporta que $\beta$ se setea a un número muy grande ($\beta \to \infty$), efectivamente reduciendo $F_{\text{LCS}} \to R_{\text{LCS}}$. Es decir, **ROUGE-L estándar en DUC es esencialmente recall**.

### Algoritmo de programación dinámica

LCS se computa con DP estándar en $O(m \cdot n)$ tiempo y espacio. La recursión:

$$
c(i, j) = \begin{cases}
0 & \text{si } i = 0 \text{ o } j = 0 \\
c(i-1, j-1) + 1 & \text{si } x_i = y_j \\
\max(c(i-1, j), c(i, j-1)) & \text{en otro caso}
\end{cases}
$$

con $\text{LCS}(X, Y) = c(m, n)$.

### Ejemplo del slide 52

Reference $X$: `"I really enjoyed reading the Hunger Games"` (7 palabras, indexadas 1-7).
Candidate $Y$: `"I loved reading the Hunger Games"` (6 palabras).

LCS posible: `I reading the Hunger Games` —se preserva el orden, longitud 5.

$$
R_{\text{LCS}} = \frac{5}{7} \approx 0.714
$$

$$
P_{\text{LCS}} = \frac{5}{6} \approx 0.833
$$

Con $\beta = 1$:

$$
F_{\text{LCS}} = \frac{2 \cdot 0.714 \cdot 0.833}{0.714 + 0.833} \approx 0.769
$$

### Comparación didáctica del paper

Lin usa estos ejemplos para mostrar la utilidad de ROUGE-L:

- **S1** (ref): `"police killed the gunman"`
- **S2**: `"police kill the gunman"` (verbo cambiado, mismo orden)
- **S3**: `"the gunman kill police"` (orden invertido)

Para ROUGE-2: S2 y S3 tienen ambos el bigrama `the gunman` y reciben el mismo score. **ROUGE-2 no distingue significados opuestos**.

Para ROUGE-L con $\beta = 1$:
- S2 vs S1: LCS = `police the gunman`, longitud 3. $F = 3/4 = 0.75$.
- S3 vs S1: LCS = `the gunman` (o `police` solo), longitud 2. $F = 2/4 = 0.50$.

ROUGE-L sí distingue: S2 (significado preservado, verbo cambia) recibe más score que S3 (significado invertido).

### Limitación de LCS

LCS solo cuenta **una** subsecuencia, la más larga. Si el candidato es S4 = `"the gunman police killed"` (palabras correctas, orden mezclado), LCS escoge `the gunman` o `police killed` pero no ambos. ROUGE-L da el mismo score que a S3 (que es peor semánticamente). ROUGE-2 acá funciona mejor porque captura los dos bigramas `the gunman` y `police killed`.

Esto motiva ROUGE-W (consecutividad) y ROUGE-S (skip-bigrams) como alternativas.

### LCS summary-level

Para resúmenes multi-oración, Lin define el **union LCS**. Dada referencia con $u$ oraciones (total $m$ palabras) y candidato con $v$ oraciones (total $n$ palabras):

$$
R_{\text{LCS}} = \frac{\sum_{i=1}^{u} |\text{LCS}_{\cup}(r_i, C)|}{m}
$$

$$
P_{\text{LCS}} = \frac{\sum_{i=1}^{u} |\text{LCS}_{\cup}(r_i, C)|}{n}
$$

donde $\text{LCS}_{\cup}(r_i, C)$ es la unión de las LCS entre la oración de referencia $r_i$ y cada oración candidata. Ejemplo: si $r_i = w_1 w_2 w_3 w_4 w_5$ y el candidato tiene dos oraciones $c_1 = w_1 w_2 w_6 w_7 w_8$ (LCS = $w_1 w_2$) y $c_2 = w_1 w_3 w_8 w_9 w_5$ (LCS = $w_1 w_3 w_5$), el union LCS es $\{w_1, w_2, w_3, w_5\}$ con longitud 4. Por tanto $|\text{LCS}_{\cup}|/|r_i| = 4/5$.

Esto evita doble-conteo de palabras compartidas pero distribuye crédito entre oraciones candidatas relevantes.

### ROUGE-Lsum (variante moderna)

En implementaciones modernas (rouge-score Python package, HuggingFace evaluate), **ROUGE-L** se computa a nivel oración y se promedia, mientras que **ROUGE-Lsum** aplica el union summary-level descrito arriba. La distinción es importante porque para resúmenes largos los scores difieren significativamente.

---

## 5. ROUGE-W: Weighted Longest Common Subsequence

### Motivación

LCS tiene un sesgo problemático: trata igual a subsecuencias **consecutivas** y **dispersas**. Ejemplo del paper:

- $X$ (ref): `[A B C D E F G]`
- $Y_1$: `[A B C D H I K]` — match consecutivo `A B C D`
- $Y_2$: `[A H B K C I D]` — match disperso `A B C D` con gaps

Ambos tienen LCS = 4 (`A B C D`), por tanto el mismo ROUGE-L. Pero $Y_1$ es claramente mejor porque preserva contigüidad.

### Definición

ROUGE-W introduce una **función de peso** $f(k)$ donde $k$ es la longitud de matches consecutivos. La restricción clave:

$$
f(x + y) > f(x) + f(y) \quad \forall x, y > 0
$$

Es decir, $f$ es **superadditiva**: premia más a un bloque consecutivo de longitud $x+y$ que a dos bloques separados de longitudes $x$ e $y$.

Ejemplos de $f$ válidas:
- **Polinomial**: $f(k) = k^\alpha$ con $\alpha > 1$. Lin usa $\alpha = 1.2$ por default.
- **Lineal con penalty**: $f(k) = ak - b$ con $a, b > 0$.

Para que ROUGE-W sea **normalizable a $[0,1]$**, $f$ debe tener inversa de forma cerrada $f^{-1}$. Por ejemplo $f(k) = k^2$ tiene $f^{-1}(k) = k^{1/2}$.

### Algoritmo

Lin extiende el DP de LCS manteniendo una tabla auxiliar $w(i,j)$ que guarda **la longitud del bloque consecutivo terminando en $(i,j)$**:

```
For (i = 1; i <= m; i++)
  For (j = 1; j <= n; j++)
    If x_i = y_j:
      k = w(i-1, j-1)
      c(i,j) = c(i-1, j-1) + f(k+1) - f(k)
      w(i,j) = k + 1
    Else:
      c(i,j) = max(c(i-1,j), c(i,j-1))
      w(i,j) = 0
```

Cuando se extiende un bloque de longitud $k$ a $k+1$, el score crece en $f(k+1) - f(k)$, que con $f$ superadditiva es mayor que $f(1)$. Por tanto extender es mejor que empezar un bloque nuevo.

### F-measure normalizada

$$
R_{\text{WLCS}} = f^{-1}\!\left(\frac{\text{WLCS}(X, Y)}{f(m)}\right)
$$

$$
P_{\text{WLCS}} = f^{-1}\!\left(\frac{\text{WLCS}(X, Y)}{f(n)}\right)
$$

$$
F_{\text{WLCS}} = \frac{(1 + \beta^2) R_{\text{WLCS}} P_{\text{WLCS}}}{R_{\text{WLCS}} + \beta^2 P_{\text{WLCS}}}
$$

La inversa $f^{-1}$ "deshace" la escala superadditiva para que el score final viva en $[0,1]$.

### Ejemplo numérico

Con $f(k) = k^2$, $|X| = 7$, $|Y| = 7$:

- $Y_1$ (consecutivo `ABCD`): WLCS = $f(4) = 16$. $R = \sqrt{16/49} = 4/7 \approx 0.571$.
- $Y_2$ (disperso, 4 matches de longitud 1): WLCS = $4 \cdot f(1) = 4$. $R = \sqrt{4/49} = 2/7 \approx 0.286$.

ROUGE-W premia $Y_1$ por $2\times$ sobre $Y_2$, capturando la diferencia que ROUGE-L pasaba por alto.

---

## 6. ROUGE-S: Skip-bigram co-occurrence

### Definición

Un **skip-bigram** es cualquier par de palabras en su orden original en la oración, **permitiendo gaps arbitrarios**. Es decir, todas las combinaciones $(w_i, w_j)$ con $i < j$ en la misma oración.

Para una oración de $m$ palabras hay $\binom{m}{2}$ skip-bigrams. Ejemplo: `"police killed the gunman"` (4 palabras, $\binom{4}{2} = 6$ skip-bigrams):

- (police, killed)
- (police, the)
- (police, gunman)
- (killed, the)
- (killed, gunman)
- (the, gunman)

### F-measure

Sea $\text{SKIP2}(X, Y)$ el número de skip-bigrams compartidos entre $X$ (longitud $m$) e $Y$ (longitud $n$):

$$
R_{\text{skip2}} = \frac{\text{SKIP2}(X, Y)}{\binom{m}{2}}
$$

$$
P_{\text{skip2}} = \frac{\text{SKIP2}(X, Y)}{\binom{n}{2}}
$$

$$
F_{\text{skip2}} = \frac{(1 + \beta^2) R_{\text{skip2}} P_{\text{skip2}}}{R_{\text{skip2}} + \beta^2 P_{\text{skip2}}}
$$

### Ejemplo del paper

Reference S1: `"police killed the gunman"`. Skip-bigrams S1: las 6 listadas arriba.

- **S2** `"police kill the gunman"`: skip-bigrams compartidos con S1 = {(police, the), (police, gunman), (the, gunman)} = 3 matches. $F_{\text{skip2}} = 3/6 = 0.5$.
- **S3** `"the gunman kill police"`: skip-bigrams: {(the, gunman), (the, kill), (the, police), (gunman, kill), (gunman, police), (kill, police)}. Comparte solo (the, gunman) con S1. $F = 1/6 \approx 0.167$.
- **S4** `"the gunman police killed"`: skip-bigrams: {(the, gunman), (the, police), (the, killed), (gunman, police), (gunman, killed), (police, killed)}. Comparte {(the, gunman), (police, killed)} = 2. $F = 2/6 \approx 0.333$.

Ranking ROUGE-S: S2 > S4 > S3, **lo cual coincide con el ranking intuitivo** (S2 cambia solo el tiempo verbal; S4 reordena pero mantiene contenido; S3 invierte completamente). Tanto ROUGE-2 como ROUGE-L fallaban en alguna de estas distinciones.

### Maximum skip distance $d_{\text{skip}}$

Sin restricción, skip-bigrams pueden incluir matches espurios como `the the` o `of in` si el documento es muy largo. Lin permite limitar el gap máximo entre palabras a $d_{\text{skip}}$:

- $d_{\text{skip}} = 0$: equivalente a bigram overlap (los dos elementos deben ser adyacentes).
- $d_{\text{skip}} = 4$: hasta 4 palabras de distancia.
- $d_{\text{skip}} = \infty$ (denotado `*`): sin restricción.

En la práctica el paquete ROUGE reporta variantes como **ROUGE-S4, ROUGE-S9, ROUGE-S\***.

### Comparación con LCS

Skip-bigram cuenta **todos los pares en orden**, no solo la subsecuencia más larga. Es más fino que LCS en este aspecto. Pero por la misma razón puede ser más ruidoso con stopwords.

---

## 7. ROUGE-SU: Skip-bigram + unigram

### Problema que resuelve

Considere S5: `"gunman the killed police"` —reverso exacto de S1 `"police killed the gunman"`. Comparten **todos los unigramas** pero **ningún skip-bigram en orden**, por tanto $\text{ROUGE-S}(S5, S1) = 0$.

Esto es indeseable: S5 al menos contiene las palabras correctas (vocabulario), aunque mal ordenadas. Una métrica que dé 0 no distingue S5 de un resumen totalmente irrelevante.

### Definición

ROUGE-SU agrega **unigramas como unidad de cuenta adicional**. Conceptualmente:

$$
\text{ROUGE-SU} = \text{ROUGE-S} \text{ aplicado al texto con un "begin-of-sentence" (BOS) marker prepended}
$$

Al prepender BOS, cada palabra $w_i$ forma un skip-bigram (BOS, $w_i$) con el marcador. Esto efectivamente cuenta cada unigrama como un skip-bigram contra BOS. El efecto: el denominador incluye $m$ unigramas + $\binom{m}{2}$ skip-bigrams, y el numerador suma matches de ambos tipos.

ROUGE-SU es **más robusto** que ROUGE-S puro porque siempre da crédito por palabras correctas, incluso si su orden es completamente roto.

---

## 8. Manejo de múltiples referencias y jackknifing

Como adelantamos, dadas $M$ referencias humanas, Lin computa ROUGE de la siguiente manera:

1. Para cada subconjunto de $M-1$ referencias, computar el mejor score del candidato contra esas referencias.
2. Promediar los $M$ scores resultantes (uno por cada referencia excluida).

Esto se llama **jackknifing**. Sus ventajas:

- **Estabilidad**: reduce varianza cuando $M$ es pequeño (típicamente $M = 2$ o $M = 4$ en DUC).
- **Comparabilidad humano-sistema**: cada referencia humana puede ser evaluada como candidato contra las $M-1$ restantes, generando un **upper bound** humano que se compara directamente con sistemas automáticos.
- **Intervalos de confianza**: combinado con bootstrap resampling, permite estimar incertidumbre estadística.

---

## 9. Validación experimental: DUC 2001-2003

### Dataset y configuración

Lin evalúa en datos del DUC 2001, 2002 y 2003, cubriendo:

- **Single document, 100 palabras**: 12 sistemas DUC 2001, 14 sistemas DUC 2002.
- **Single document, very short (~10 palabras, headline-like)**: 14 sistemas DUC 2003.
- **Multi-document, 10/50/100/200/400 palabras**: distintos sistemas y números de muestras.

Cada sistema fue evaluado por humanos con Summary Evaluation Environment (SEE) de USC/ISI, dando un **content coverage score** por resumen. El promedio sobre todos los resúmenes da el score humano del sistema.

### Métrica de validación

Lin computa **Pearson, Spearman y Kendall correlations** entre el score promedio de cada sistema según ROUGE y el score humano. Reporta primariamente Pearson, con intervalos de confianza al 95% vía bootstrap resampling (Davison & Hinkley 1997).

El **valor crítico para Pearson** al 95% con 8 grados de libertad es 0.632; con 16 grados de libertad es 0.468.

### Configuraciones evaluadas

17 variantes de ROUGE: ROUGE-N con $N = 1, \ldots, 9$; ROUGE-L; ROUGE-W con $\alpha = 1.2$; ROUGE-S con $d_{\text{skip}} \in \{1, 4, 9, *\}$; ROUGE-SU con $d_{\text{skip}} \in \{1, 4, 9, *\}$.

Tres variantes de preprocesamiento:
- **CASE**: texto original, sin modificaciones.
- **STEM**: Porter stemmer aplicado.
- **STOP**: stopwords removidas.

### Resultados clave (Tabla 1, single-doc 100 palabras)

| Métrica | DUC 2001 (1 ref) | DUC 2001 (3 refs) | DUC 2002 (1 ref) | DUC 2002 (2 refs) |
|---------|-----:|-----:|-----:|-----:|
| ROUGE-1 | 0.76 | 0.80 | 0.98 | 0.98 |
| **ROUGE-2** | 0.84 | 0.87 | 0.99 | 0.99 |
| ROUGE-3 | 0.82 | 0.86 | 0.99 | 0.99 |
| **ROUGE-L** | 0.83 | 0.86 | 0.99 | 0.99 |
| ROUGE-S* | 0.74 | 0.78 | 0.98 | 0.98 |
| **ROUGE-S4** | 0.84 | 0.87 | 0.99 | 0.99 |
| ROUGE-SU4 | 0.84 | 0.87 | 0.99 | 0.99 |
| **ROUGE-W-1.2** | 0.85 | 0.87 | 0.99 | 0.99 |

Observaciones de Lin:
- Stemming y stopword removal no cambian mucho las correlaciones acá.
- ROUGE-2 es la mejor variante de ROUGE-N (ROUGE-3+ degradan).
- ROUGE-L, ROUGE-W, ROUGE-S/SU4 son todas competitivas.
- Múltiples referencias ayudan marginalmente.

### Resultados (Tabla 2, very short summaries DUC 2003)

| Métrica | CASE (4 refs) | STEM (4 refs) | STOP (4 refs) |
|---------|-----:|-----:|-----:|
| **ROUGE-1** | 0.95 | 0.95 | 0.90 |
| ROUGE-2 | 0.76 | 0.75 | 0.77 |
| **ROUGE-L** | 0.96 | 0.96 | 0.96 |
| ROUGE-W-1.2 | 0.96 | 0.96 | 0.96 |
| **ROUGE-SU4** | 0.96 | 0.95 | 0.97 |

Observaciones:
- Para resúmenes muy cortos (~10 palabras), **ROUGE-1, ROUGE-L y ROUGE-SU4** son superiores.
- ROUGE-N con $N \geq 2$ degrada significativamente (resúmenes cortos tienen pocos bigramas).
- Stopword removal ayuda excepto para ROUGE-1.

### Resultados multi-document (Tabla 3)

Aquí las correlaciones bajan: rara vez superan 0.85. Lin atribuye esto al **menor número de muestras** (~30 por sistema en multi-doc vs 100+ en single-doc), que produce inestabilidad estadística.

Hallazgos:
- ROUGE-1, ROUGE-2, ROUGE-S4/9, ROUGE-SU4/9 con stopword removal superan 0.70 consistentemente.
- ROUGE-L y ROUGE-W **no funcionan bien** en multi-document (probablemente porque los resúmenes son largos y la LCS captura solo una secuencia, perdiendo info).
- Múltiples referencias ayudan más que en single-doc.

### Conclusiones operacionales del paper

1. ROUGE-2, ROUGE-L, ROUGE-W, ROUGE-S → **single-doc summarization**.
2. ROUGE-1, ROUGE-L, ROUGE-W, ROUGE-SU4/9 → **very short summaries** (headlines).
3. Multi-doc es difícil; ROUGE-1, ROUGE-2, ROUGE-S4/9, ROUGE-SU4/9 con stopword removal son razonables.
4. **Stopword removal generalmente ayuda**.
5. **Múltiples referencias generalmente ayudan**.

---

## 10. Variantes prácticas del campo y la "trinidad" reportada

En la práctica académica post-2004 emergió un consenso de reporting:

- **ROUGE-1**: fluidez léxica básica, sensible a vocabulario.
- **ROUGE-2**: coherencia local de bigramas.
- **ROUGE-L** (y ROUGE-Lsum): orden global vía LCS.

Esta "trinidad" se reporta en virtualmente todo paper de summarization desde 2010 en adelante.

ROUGE-W y ROUGE-S/SU, a pesar de tener buenas propiedades teóricas, se reportan con mucho menor frecuencia. Posibles razones:
- Más complejos de explicar y reproducir.
- ROUGE-W requiere elegir $\alpha$ (parámetro adicional).
- Las implementaciones populares (`pyrouge`, `rouge-score`) priorizan la trinidad por compatibilidad.

### Implementaciones canónicas

- **ROUGE-1.5.5 (Perl)**: implementación original de Lin, distribuida por ISI/USC.
- **pyrouge**: wrapper Python de la versión Perl.
- **rouge-score** (Google, 2020): re-implementación pura en Python, la **referencia moderna**.
- **HuggingFace evaluate / datasets**: usa rouge-score por debajo.

Pequeñas diferencias en tokenización y handling de stopwords pueden cambiar los scores en 1-2 puntos, lo cual ha generado debates sobre reproducibilidad.

---

## 11. Limitaciones de ROUGE

Veinte años después, las limitaciones de ROUGE están bien documentadas y son material de docenas de papers críticos:

### 11.1 Overlap léxico, no semántico

ROUGE compara palabras tal cual. No captura paráfrasis ni sinónimos. Un resumen que diga "automobile" cuando la referencia dice "car" recibe **cero crédito** para esa palabra. Esto desfavorece sistemáticamente a sistemas abstractivos (que reescriben) versus extractivos (que copian).

### 11.2 Dominancia de stopwords

ROUGE-1 puede inflarse por palabras función como "the", "a", "of". Por eso Lin mismo recomienda stopword removal en muchos casos, pero en la práctica los benchmarks reportan ROUGE con stopwords —porque la implementación original lo hace así.

### 11.3 Sin fluidez ni gramaticalidad

Un resumen con las palabras correctas pero en orden gramatical incorrecto puede tener ROUGE alto. La métrica no penaliza ungrammaticality. ROUGE-L y ROUGE-W mitigan parcialmente vía word order, pero solo en sentido débil.

### 11.4 Faithfulness gap (hallucinations)

ROUGE no detecta **alucinaciones**: contenido en el candidato que no está en la fuente o que la contradice. Si el candidato dice "la víctima tenía 30 años" pero la fuente dice 60, y la palabra "30" no aparece en ninguna referencia, ROUGE no penaliza específicamente la alucinación. Este problema se vuelve crítico en la era de LLMs (2020+) que inventan datos.

### 11.5 Sesgo hacia resúmenes extractivos

Sistemas que copian segmentos del documento original tienden a obtener ROUGE alto porque la referencia humana muchas veces también copia frases del documento. Esto distorsiona la comparación frente a sistemas abstractivos creativos.

### 11.6 Idiomas no-flexivos

Para chino, japonés, tailandés y otros idiomas sin separación de palabras por espacio, ROUGE requiere **tokenización custom**. Las decisiones de tokenización (palabras vs caracteres vs subwords) afectan dramáticamente el score, haciendo difícil comparar entre estudios.

### 11.7 Single-reference suele ser la norma práctica

Aunque Lin recomienda múltiples referencias, en la práctica datasets como CNN/DailyMail o XSum tienen **una sola referencia por documento**. Esto reduce el upper bound y desfavorece sistemas que generan resúmenes legítimos diferentes al gold.

### 11.8 No mide informatividad

ROUGE asume que la referencia es la mejor representación del contenido. No mide si el resumen captura información **importante**. Un resumen que copie las palabras más comunes del documento puede tener ROUGE decente sin transmitir nada útil.

---

## 12. Métricas sucesoras y competidoras

La búsqueda de mejores métricas ha sido un sub-campo activo de NLP desde 2005:

### 12.1 METEOR (Banerjee & Lavie, 2005)

Combina exact match, **stemming**, **synonyms (WordNet)** y **paráfrasis** para computar precision/recall harmonic mean. Penaliza fragmentation (matches no alineados). Mejor correlación humana que BLEU en MT, pero adopción limitada en summarization.

### 12.2 BERTScore (Zhang et al., 2020)

Usa **embeddings contextualizados de BERT** para computar similitud coseno entre tokens del candidato y referencia. Captura paráfrasis y sinónimos automáticamente. Reporta precision, recall y F1.

$$
R_{\text{BERT}} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} x_i^T \hat{x}_j
$$

donde $x_i, \hat{x}_j$ son embeddings BERT normalizados. Es la métrica semántica más adoptada post-2020.

### 12.3 BLEURT (Sellam et al., 2020)

Entrena un **regresor sobre BERT** para predecir scores humanos. Pretraining sintético con perturbaciones de texto + fine-tuning sobre WMT human ratings. Aprende a evaluar, en lugar de definir reglas.

### 12.4 MoverScore (Zhao et al., 2019)

Usa **Earth Mover's Distance** sobre embeddings contextualizados. Generaliza BERTScore permitiendo matches uno-a-muchos con pesos. Más robusto a reordenamientos.

### 12.5 QAEval, FactCC, SummaC (2020-2022)

Métricas específicas para **faithfulness/factuality**:
- **QAEval**: genera preguntas sobre el resumen, las responde usando el documento fuente, compara respuestas.
- **FactCC**: clasificador entrenado en pares (claim, document) para predecir entailment.
- **SummaC**: usa NLI a nivel oración para detectar contradicciones.

### 12.6 G-Eval (Liu et al., 2023)

Usa **GPT-4 como juez**: prompt al modelo con criterios de evaluación (coherencia, consistencia, fluidez, relevancia) y el resumen, obteniene un score numérico vía chain-of-thought. Correlaciona muy alto con humanos pero introduce sesgos del LLM evaluador.

### 12.7 Realidad práctica

A pesar de toda esta proliferación, **ROUGE-1, ROUGE-2 y ROUGE-L siguen siendo las métricas obligatorias** en práctica. Reasons:
- Reproducibilidad: determinístico, sin modelos.
- Costo: $O(mn)$ vs llamadas a LLM.
- Histórico: comparabilidad con miles de papers previos.
- Conservadurismo: los reviewers piden ROUGE.

Las métricas modernas se reportan como **complemento**, raramente como reemplazo.

---

## 13. Conexión con la clase 22 del curso IA UC

La clase 22 del curso aborda **evaluación de modelos de NLP generativos**, con foco en métricas para summarization, traducción y diálogo. Los slides 49-52 reproducen casos canónicos de ROUGE:

- **Slide 49**: motivación recall-oriented vs precision-oriented (BLEU).
- **Slide 50**: definición ROUGE-1, ROUGE-2 con ejemplo "I really loved reading the Hunger Games".
- **Slide 51**: ROUGE-L con LCS sobre la misma referencia.
- **Slide 52**: tabla comparativa de las variantes y discusión de cuándo usar cada una.

Para los ejercicios prácticos del curso, los estudiantes implementan ROUGE-1 y ROUGE-L manualmente, comparan contra `rouge-score` de Python, y discuten qué métrica reportar en un proyecto de summarization con CNN/DailyMail o un dataset propio.

Como **fundamento transversal**, ROUGE conecta con:
- **N-gramas** (Clase 12, Modelos de Lenguaje): la unidad básica de ROUGE-N.
- **BLEU** (clases de traducción): contrapunto precision-oriented.
- **Programación dinámica** (estructura algorítmica de LCS).
- **Estadística inferencial**: jackknifing, bootstrap, correlación de Pearson.
- **Embeddings contextualizados** (Clase 20, BERT/ELMo): sucesores semánticos.

---

## 14. Lecciones de diseño del paper

Más allá de las fórmulas, el paper de Lin enseña principios de diseño metodológico:

1. **Validar la métrica contra ground truth humano**: cualquier métrica nueva debe demostrar correlación con juicios humanos antes de adoptarse. Saggion 2002 falló en esto y por eso ROUGE eclipsó propuestas previas.

2. **Bootstrap y jackknifing para significancia**: reportar correlaciones sin intervalos de confianza es subóptimo. Lin establece el estándar de bootstrap resampling.

3. **Múltiples variantes, un paquete unificado**: ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S no compiten —se complementan. Distribuir todas en un solo paquete (1.5.5 Perl) facilitó adopción masiva.

4. **Hyperparameters explícitos**: $\beta$ para balance precision/recall, $d_{\text{skip}}$ para skip-bigrams, $\alpha$ para ROUGE-W. Documentar y publicar defaults razonables.

5. **Ablations cruzadas**: CASE × STEM × STOP × single/multi reference produce 18 datapoints por métrica por tarea. Exhaustivo, replicable.

---

## 15. Críticas y replies históricos

### Crítica: "ROUGE solo mide n-gram overlap, no contenido"

Reply de Lin (en el propio paper): no afirmamos que ROUGE mida "contenido" en sentido semántico; afirmamos que **correlaciona empíricamente** con juicios humanos de coverage. La validación es empírica, no teórica.

### Crítica: "Los benchmarks de summarization están saturados"

A partir de ~2018 los modelos neuronales (PEGASUS, BART, T5) empezaron a obtener ROUGE comparables a humanos en CNN/DailyMail. Esto se interpretó como "ROUGE está saturado" pero papers como Kryściński et al. (2019) mostraron que esos modelos **alucinan mucho** —ROUGE alto sin faithfulness alta.

### Crítica: "ROUGE penaliza abstracción"

Sí. Es una limitación inherente al overlap léxico. La comunidad lo sabe pero sigue reportándola por convención.

### Reply institucional: ROUGE como leaderboard universal

DUC, TAC, y posteriormente shared tasks de Workshop on Neural Generation and Translation (WNGT) y GEM consolidaron ROUGE como métrica de leaderboard. Su disponibilidad libre y su independencia de modelos entrenados son virtudes operacionales irremplazables.

---

## 16. Lectura recomendada en orden

1. **Papineni et al. 2002 (BLEU)** — precedente directo, contrapunto precision-oriented.
2. **Lin & Hovy 2003 (HLT-NAACL)** — paper preliminar de Lin que motiva ROUGE.
3. **Este paper (Lin 2004)** — la definición canónica.
4. **Banerjee & Lavie 2005 (METEOR)** — primer sucesor importante.
5. **Zhang et al. 2020 (BERTScore)** — sucesor semántico moderno.
6. **Kryściński et al. 2019 (FactCC)** — crítica de faithfulness.
7. **Liu et al. 2023 (G-Eval)** — LLM-as-a-judge.

---

## 17. Cierre

ROUGE (Lin 2004) es uno de esos papers de "workshop" que termina definiendo un campo entero. Sin teoremas profundos, sin arquitecturas complejas: solo una operacionalización cuidadosa, validada empíricamente, distribuida como software libre con buena documentación. Sus fórmulas caben en una página; su impacto se mide en decenas de miles de citas y en cada paper de summarization publicado entre 2005 y 2026.

Para el curso IA UC, ROUGE es una pieza pedagógica ideal: muestra cómo una métrica simple, bien validada, puede generar 20 años de uso —y también cómo sus limitaciones léxicas motivan toda una agenda de investigación en métricas semánticas modernas. Entender ROUGE es entender por qué la evaluación automática de NLP sigue siendo un problema abierto en 2026, incluso con LLMs de billones de parámetros disponibles.

---

## Referencias clave citadas en el paper

- Cormen, Leiserson, Rivest. 1989. *Introduction to Algorithms*. MIT Press. (LCS DP)
- Davison & Hinkley. 1997. *Bootstrap Methods and Their Application*. Cambridge UP.
- Lin & Hovy. 2003. Automatic evaluation of summaries using n-gram co-occurrence statistics. HLT-NAACL.
- Lin & Och. 2004. Automatic evaluation of machine translation quality using LCS and skip-bigram statistics. ACL.
- Mani. 2001. *Automatic Summarization*. John Benjamins.
- Melamed. 1995. Automatic evaluation and uniform filter cascades. WVLC3.
- Over & Yen. 2003. An introduction to DUC 2003. NIST.
- Papineni et al. 2001. BLEU. IBM Research Report RC22176.
- Saggion, Radev, Teufel, Lam. 2002. Meta-evaluation of summaries. COLING.
- Van Rijsbergen. 1979. *Information Retrieval*. Butterworths.
