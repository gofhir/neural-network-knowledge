# Levy & Goldberg 2014 — Neural Word Embedding as Implicit Matrix Factorization

| Campo | Valor |
|---|---|
| **Autores** | Omer Levy, Yoav Goldberg |
| **Afiliación** | Bar-Ilan University |
| **Venue** | NeurIPS 2014 |
| **Pdf** | `Levy-Goldberg-SGNS-MF-2014.pdf` (9 páginas) |
| **Citaciones** | >3.000 |
| **URL** | https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization |

> *"We analyze skip-gram with negative-sampling (SGNS) and show that it is implicitly factorizing a word-context matrix, whose cells are the pointwise mutual information (PMI) of the respective word and context pairs, shifted by a global constant."*

Este es **el paper que une las dos tradiciones** de word embeddings:

- **Métodos basados en conteo** (LSA, PPMI) — tradición de los 80s y 90s en distributional semantics.
- **Métodos basados en predicción neuronal** (Word2Vec, SGNS) — tradición de Bengio 2003 + Mikolov 2013.

Levy & Goldberg demuestran formalmente que **SGNS es una factorización implícita de la matriz PMI shifted**. Esto cierra una brecha conceptual de dos décadas y motiva un puente entre ambos paradigmas.

---

## 1. Contexto

### 1.1 Las dos tradiciones de word embeddings

**Tradición "count-based"** (Harris 1954, Church & Hanks 1990, Deerwester 1990):
- Construir matriz $M$ con celdas $M_{ij}$ = alguna medida de asociación entre palabra $i$ y contexto $j$.
- Medidas estándar: count, log-count, PMI, PPMI, TF-IDF.
- Reducir dimensionalidad con SVD para obtener vectores densos.

**Tradición "predict-based"** (Bengio 2003, Collobert 2008, Mikolov 2013):
- Entrenar una red neuronal a predecir palabras desde contexto (o viceversa).
- Los embeddings emergen como subproducto.

**Pregunta abierta pre-2014**: ¿son estas dos tradiciones equivalentes? ¿O hay algo fundamentalmente diferente en lo que cada una captura?

Baroni, Dinu & Kruszewski (2014) ya habían comparado empíricamente y concluyeron que predict-based **gana sistemáticamente** sobre count-based en una variedad de tareas (su paper se titula provocativamente *"Don't count, predict!"*).

Pero **¿por qué**? La respuesta no estaba clara — Word2Vec se presentaba con motivación operacional, no teórica. Hasta este paper.

### 1.2 Notación del paper

- $V_W$, $V_C$: vocabularios de palabras y contextos. En SGNS estándar $V_W = V_C$.
- $D$: colección de pares (palabra, contexto) observados en el corpus.
- $\#(w, c)$: número de veces que el par $(w, c)$ aparece en $D$.
- $\#(w) = \sum_{c'} \#(w, c')$, $\#(c) = \sum_{w'} \#(w', c)$: marginales.
- $|D| = \sum_{w, c} \#(w, c)$: total de pares.
- $\vec{w} \in \mathbb{R}^d$: embedding de palabra (filas de matriz $W$).
- $\vec{c} \in \mathbb{R}^d$: embedding de contexto (filas de matriz $C$).
- $\sigma(x) = 1/(1 + e^{-x})$.

---

## 2. Recordatorio del objetivo SGNS

El objetivo SGNS de Mikolov 2013, para un par observado $(w, c)$:

$$
\log \sigma(\vec{w} \cdot \vec{c}) + k \cdot \mathbb{E}_{c_N \sim P_D} \left[ \log \sigma(-\vec{w} \cdot \vec{c}_N) \right]
$$

donde $k$ = número de negativos, $P_D(c) = \#(c) / |D|$ = distribución unigrama empírica (en la fórmula del paper se omite la elevación a $3/4$ por simplicidad analítica).

**Objetivo global** sobre todos los pares:
$$
\ell = \sum_{w \in V_W} \sum_{c \in V_C} \#(w, c) \left( \log \sigma(\vec{w} \cdot \vec{c}) + k \cdot \mathbb{E}_{c_N \sim P_D} [\log \sigma(-\vec{w} \cdot \vec{c}_N)] \right). \quad (2)
$$

---

## 3. La demostración central — SGNS factoriza PMI shifted

### 3.1 Sé qué matriz factoriza SGNS

SGNS aprende dos matrices $W$ y $C$ con $W \cdot C^\top = M \in \mathbb{R}^{|V_W| \times |V_C|}$. La pregunta: ¿qué es $M_{ij} = \vec{w}_i \cdot \vec{c}_j$?

### 3.2 Derivación

**Paso 1: reescribir el objetivo (ecuación 3 del paper).**

Expandiendo la expectativa:
$$
\mathbb{E}_{c_N \sim P_D} [\log \sigma(-\vec{w} \cdot \vec{c}_N)] = \sum_{c_N \in V_C} \frac{\#(c_N)}{|D|} \log \sigma(-\vec{w} \cdot \vec{c}_N).
$$

Separando el término $c_N = c$ del resto:
$$
= \frac{\#(c)}{|D|} \log \sigma(-\vec{w} \cdot \vec{c}) + \sum_{c_N \neq c} \frac{\#(c_N)}{|D|} \log \sigma(-\vec{w} \cdot \vec{c}_N). \quad (4)
$$

**Paso 2: objetivo local por par $(w, c)$.**

Combinando (3) y (4), el objetivo que depende **solo de $(w, c)$ con $\vec{w} \cdot \vec{c}$ fijo** es:
$$
\ell(w, c) = \#(w, c) \log \sigma(\vec{w} \cdot \vec{c}) + k \cdot \#(w) \cdot \frac{\#(c)}{|D|} \log \sigma(-\vec{w} \cdot \vec{c}). \quad (5)
$$

**Paso 3: maximizar como función de $x = \vec{w} \cdot \vec{c}$.**

Derivada respecto a $x$:
$$
\frac{\partial \ell}{\partial x} = \#(w, c) \cdot \sigma(-x) - k \cdot \#(w) \cdot \frac{\#(c)}{|D|} \cdot \sigma(x) = 0.
$$

Resolver: definir $y = e^x$, llega a una ecuación cuadrática en $y$:
$$
e^{2x} - \left( \frac{\#(w, c)}{k \cdot \#(w) \cdot \#(c)/|D|} - 1 \right) e^x - \frac{\#(w, c)}{k \cdot \#(w) \cdot \#(c)/|D|} = 0.
$$

La solución es:
$$
e^x = \frac{\#(w, c) \cdot |D|}{\#(w) \cdot \#(c) \cdot k}.
$$

Tomando log:
$$
\boxed{\vec{w} \cdot \vec{c} = \log \left( \frac{\#(w, c) \cdot |D|}{\#(w) \cdot \#(c)} \right) - \log k.} \quad (6)
$$

**Paso 4: identificar PMI.**

El término $\log \frac{\#(w, c) \cdot |D|}{\#(w) \cdot \#(c)}$ es exactamente la definición empírica de **pointwise mutual information**:
$$
\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w) \cdot P(c)} = \log \frac{\#(w, c) \cdot |D|}{\#(w) \cdot \#(c)}.
$$

Por lo tanto:
$$
\boxed{\vec{w} \cdot \vec{c} = \text{PMI}(w, c) - \log k.} \quad (7)
$$

### 3.3 Conclusión

**SGNS está factorizando la matriz**:
$$
M_{ij}^{\text{SGNS}} = \text{PMI}(w_i, c_j) - \log k.
$$

i.e., **la matriz PMI shifted por una constante $\log k$**.

Para $k = 1$: SGNS factoriza la PMI matrix directamente.
Para $k > 1$: SGNS factoriza una versión shifted (todas las celdas reducidas por $\log k$).

### 3.4 NCE también factoriza una matriz

El paper deriva análogamente que NCE (Noise Contrastive Estimation, el predecesor teórico de SGNS) factoriza:

$$
M_{ij}^{\text{NCE}} = \log P(w_i \mid c_j) - \log k = \log \frac{\#(w, c)}{\#(c)} - \log k. \quad (8)
$$

i.e., la **log-conditional probability shifted**. Diferente a PMI — explica empíricamente por qué SGNS funciona mejor que NCE en tareas downstream.

---

## 4. Implicaciones

### 4.1 Factorización ponderada

SGNS no factoriza $M^{\text{PMI}} - \log k$ con SVD (que es factorización con error L2 uniforme). SGNS factoriza con un **peso** dado por $\#(w, c)$ — pares observados muchas veces tienen más peso, pares raros menos.

Esto se ve claramente en la ecuación (5): la loss para el par $(w, c)$ es proporcional a $\#(w, c)$. Por lo tanto pares frecuentes contribuyen más al gradiente.

**Es equivalente a una factorización matricial ponderada** estilo Koren et al. 2009 (Matrix factorization for recommender systems).

### 4.2 SGNS vs SVD

| Aspecto | SGNS | SVD truncado de PMI |
|---|---|---|
| Loss | Sigmoide ponderada | $L_2$ uniforme |
| Distingue observado/no-observado | Sí | No |
| Maneja celdas $-\infty$ (PMI con conteo 0) | Sí (las ignora) | No (necesita imputación) |
| Manejo de palabras raras | Down-weight (contribuyen poco) | Same weight |
| Hyperparameter tuning | Sí | Mínimo |
| Escalable a corpora grandes | Sí (streaming) | Difícil ($O(V^2)$ mem) |

### 4.3 Por qué SGNS funciona mejor en analogías

El paper conjetura (sección 6): SGNS rinde mejor en analogías porque su **factorización ponderada da más importancia a pares frecuentes**, que son los más "limpios" estadísticamente. SVD uniformizado se ve afectado por celdas raras o ausentes.

---

## 5. Aplicaciones derivadas

### 5.1 Shifted PPMI (SPPMI)

Sabiendo que la matriz óptima es $\text{PMI} - \log k$, Levy & Goldberg proponen un nuevo método **basado solo en conteos**:

$$
\text{SPPMI}_k(w, c) = \max(\text{PMI}(w, c) - \log k, 0). \quad (12)
$$

i.e., **positive Shifted PMI** — recortar valores negativos a cero para mantener la matriz sparse.

**Resultado**: SPPMI sin redución de dimensionalidad (representación sparse, ~150k dim) **iguala o supera SGNS** en tareas de word similarity. Sin redes neuronales, sin entrenamiento iterativo — solo conteos + PMI + shift + truncamiento.

### 5.2 SVD sobre SPPMI

Otro alternativo: aplicar SVD al SPPMI matrix para obtener vectores densos. El paper demuestra empíricamente:

**Tabla 1** — % de desviación del óptimo en optimizar la loss SGNS:

| Method | PMI-$\log k$ | SPPMI | SVD $d=100$ | SVD $d=500$ | SGNS $d=100$ | SGNS $d=500$ |
|---|---|---|---|---|---|---|
| $k=1$ | 0% | 0.00009% | 26.1% | 25.2% | 31.4% | 29.4% |
| $k=5$ | 0% | 0.00004% | 95.8% | 95.1% | 39.3% | 36.0% |
| $k=15$ | 0% | 0.00002% | 266% | 266% | 7.80% | 6.37% |

**Lectura**: SPPMI es **casi óptimo** en optimizar la loss SGNS. SVD truncado en dim baja es **muy malo**, especialmente cuando $k$ aumenta. SGNS se aproxima al óptimo cuando $d$ es grande.

### 5.3 Tabla 2 — Comparación en tareas downstream

| Repr | $k$ | WS353 corr | MEN corr | Mixed analogies acc | Synt. analogies acc |
|---|---|---|---|---|---|
| SPPMI | 5 | **0.691** | **0.735** | 0.655 | 0.466 |
| SPPMI | 1 | 0.605 | 0.688 | 0.567 | 0.353 |
| SVD ($d=1000$) | 1 | 0.652 | — | 0.644 | — |
| SVD ($d=1000$) | 5 | 0.661 | 0.708 | 0.471 | 0.448 |
| SGNS ($d=1000$) | 1 | 0.633 | 0.690 | 0.619 | 0.59 |
| **SGNS** ($d=1000$) | 5 | **0.666** | 0.716 | **0.616** | **0.619** |
| SGNS ($d=1000$) | 15 | 0.644 | 0.694 | 0.540 | **0.627** |

**Observaciones**:
- En **word similarity** (WS353, MEN): SPPMI gana o empata.
- En **analogías sintácticas**: SGNS domina claramente.
- SGNS prefiere $k$ grande para sintaxis; SPPMI prefiere $k = 5$ moderado.

### 5.4 ¿Por qué SGNS gana en analogías sintácticas?

Conjetura del paper (sección 6): las analogías sintácticas (`good:better :: smart:smarter`) dependen de **palabras de función** ("the", "a", "many"). Estas palabras son muy frecuentes y la **weighted factorization de SGNS las favorece**. SVD/SPPMI las tratan uniformemente y pierden información.

---

## 6. Limitaciones y notas

### 6.1 Aproximación (no exactitud)

La derivación asume que **$d$ es suficientemente grande para reconstrucción perfecta**. En la práctica $d \ll |V|$ (típicamente $d = 300$), por lo que SGNS hace una **factorización aproximada**. El argumento del paper es que la dirección del gradiente sigue siendo correcta — converge hacia el óptimo aunque no lo alcance.

### 6.2 El paper ignora el exponente 3/4

Por simplicidad analítica, las derivaciones usan $P_n(w) = U(w)$ (unigrama puro), no $U(w)^{3/4}$ como Mikolov 2013. La nota al pie 1 reconoce que con $U(w)^{3/4}$ la PMI se generaliza:
$$
\text{PMI}_{3/4}(w, c) = \log \frac{\#(w, c)}{\#(w) \cdot \#(c)^{3/4}/Z}.
$$
Y SGNS factoriza $\text{PMI}_{3/4} - \log k$. Los resultados cualitativos no cambian.

### 6.3 Solo SGNS, no SG con hierarchical softmax

El análisis se aplica específicamente a SGNS (negative sampling). Hierarchical softmax tiene una estructura distinta y el paper no lo cubre — quedan preguntas abiertas sobre qué factoriza HS.

---

## 7. Impacto y legado

### 7.1 Unificación conceptual

Antes de Levy & Goldberg, había dos comunidades:

1. **Count-based** (linguistas computacionales, NLP clásico) — usaban PMI, LSA, COALS, HAL.
2. **Predict-based** (deep learning) — usaban Word2Vec, GloVe.

Este paper demuestra que **ambas comunidades están haciendo lo mismo** desde perspectivas diferentes. SGNS es count-based en disfraz neuronal.

### 7.2 Influencia en GloVe

GloVe (Pennington 2014, publicado meses antes) deriva una factorización **explícita** de log-co-ocurrencia. Levy & Goldberg sale meses después y muestra que SGNS hace algo **implícito y análogo**. Los dos papers convergen en la misma conclusión: word embeddings = factorización de matriz de co-ocurrencia.

### 7.3 Línea de trabajo posterior

- **Arora et al. 2016** (RAND-WALK): justificación teórica de PMI shifted desde primeros principios usando modelos generativos.
- **Hashimoto et al. 2016**: extensión a embeddings de oraciones y documentos.
- **Allen & Hospedales 2019** (Analogies Explained): usa Levy & Goldberg como punto de partida para explicar matemáticamente las analogías.
- **Mu & Viswanath 2018**: propiedades geométricas (anisotropía) de embeddings interpretadas vía PMI.

### 7.4 Levy & Goldberg como autores

Este paper inicia una serie de trabajos influyentes de los dos autores:
- **Levy, Goldberg, Dagan 2015** (*"Improving distributional similarity with lessons learned from word embeddings"*) — sistematiza qué hiperparámetros importan.
- **Goldberg 2017** (*"Neural Network Methods for Natural Language Processing"*) — libro que se vuelve el estándar.
- **Levy 2018**: doctorado completo en distributional semantics.

Ambos se convierten en figuras centrales del análisis empírico-teórico de embeddings.

---

## 8. Conexión con la clase 18

La clase 18 **no menciona explícitamente** este paper, pero implícitamente lo usa cuando:

- Slide 26 muestra la composicionalidad aditiva `Beijing - China + Russia ≈ Moscow`. La justificación matemática (sección 5 de `Mikolov-Word2Vec-DistributedRepresentations-2013.md`) se basa en la interpretación log-bilinear: $\mathbf{v}_w \cdot \mathbf{u}_c \approx \log P(c \mid w)$. Levy & Goldberg refinan esta interpretación: SGNS factoriza PMI, no log-probability — lo cual cambia la geometría sutilmente.

- Slide 28 menciona "se aprende todo automáticamente" y "no hay que estar contando cuántas veces aparece cada n-grama". Levy & Goldberg demuestran que **sí estás contando, solo que de forma implícita** — la información de conteos vive en los gradientes.

---

## 9. Cita BibTeX

```bibtex
@inproceedings{levy2014neural,
  title={Neural word embedding as implicit matrix factorization},
  author={Levy, Omer and Goldberg, Yoav},
  booktitle={Advances in Neural Information Processing Systems},
  volume={27},
  year={2014},
  url={https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization}
}
```

---

## 10. Frase para recordar

> *"Word2Vec is PMI in disguise."* — el punchline del paper. Esta equivalencia teórica conecta 50 años de distributional semantics con la era neuronal, y motiva que muchas heurísticas modernas (negative sampling, subsampling, exponente 3/4) tengan correlatos en la literatura pre-neural.

---

## 11. Apéndice — la matriz PMI en código

```python
import numpy as np
from collections import Counter

def build_pmi_matrix(corpus_tokens, vocab, window=5, k=5):
    """Construir matriz PMI - log k = matriz factorizada implícitamente por SGNS."""
    # Conteos de co-ocurrencia
    cooc = Counter()
    word_counts = Counter()
    context_counts = Counter()
    total_pairs = 0

    for i, w in enumerate(corpus_tokens):
        if w not in vocab:
            continue
        word_counts[w] += 1
        for j in range(max(0, i - window), min(len(corpus_tokens), i + window + 1)):
            if j == i: continue
            c = corpus_tokens[j]
            if c not in vocab: continue
            cooc[(w, c)] += 1
            context_counts[c] += 1
            total_pairs += 1

    # PMI shifted
    V = list(vocab)
    idx = {w: i for i, w in enumerate(V)}
    M = np.full((len(V), len(V)), -np.inf)
    log_k = np.log(k)
    for (w, c), count in cooc.items():
        pmi = np.log(count * total_pairs / (word_counts[w] * context_counts[c]))
        M[idx[w], idx[c]] = pmi - log_k

    return M, V

def sppmi(M):
    """Shifted Positive PMI: aplicar max(·, 0)."""
    return np.maximum(M, 0)

def svd_embeddings(M_sparse, d=300):
    """SVD truncado para obtener embeddings densos."""
    from scipy.sparse.linalg import svds
    U, S, Vt = svds(M_sparse, k=d)
    # Embeddings simétricos según Levy & Goldberg
    W = U * np.sqrt(S)
    C = Vt.T * np.sqrt(S)
    return W, C
```

Este pipeline de 4 funciones, sin redes neuronales, **rivaliza con Word2Vec en word similarity** según Tabla 2 del paper. Es la demostración práctica de que SGNS y SVD-of-PMI **son hermanas**.

---

## 12. Tres lecciones del paper que cambian cómo pensar word embeddings

1. **No hay magia neural**: Word2Vec no aprende algo cualitativamente distinto de SVD sobre PMI. Aprende algo cuantitativamente mejor por la factorización ponderada.

2. **PMI es la "moneda" central**: cualquier word embedding razonable es alguna proyección de baja-rango de una matriz tipo PMI. Pero **lo que importa son los pesos**.

3. **Lo simple gana cuando se entiende**: SPPMI (cuatro líneas de código) iguala SGNS (cientos de líneas de código optimizado) en muchas tareas, una vez que sabemos qué matriz factorizar.
