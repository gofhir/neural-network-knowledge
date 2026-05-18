# Pennington, Socher & Manning 2014 — GloVe: Global Vectors for Word Representation

| Campo | Valor |
|---|---|
| **Autores** | Jeffrey Pennington, Richard Socher, Christopher D. Manning |
| **Afiliación** | Stanford NLP Group |
| **Venue** | EMNLP 2014, Doha, Qatar |
| **ACL Anthology** | D14-1162 |
| **Pdf** | `Pennington-GloVe-2014.pdf` (12 páginas) |
| **Citaciones** | >40.000 |
| **URL** | https://aclanthology.org/D14-1162/ |
| **Código** | https://github.com/stanfordnlp/GloVe |
| **Embeddings preentrenados** | https://nlp.stanford.edu/projects/glove/ |

> *"GloVe — for Global Vectors — because the global corpus statistics are captured directly by the model."*

GloVe es la respuesta de Stanford NLP a Word2Vec. Publicado un año después de Mikolov, **unifica las dos tradiciones** que hasta ese momento estaban en competencia:

- **Métodos basados en conteos globales** (LSA, HAL, COALS, HPCA, PPMI): aprovechan estadística global pero rinden mal en analogías.
- **Métodos basados en ventanas locales** (Word2Vec, vLBL): excelentes en analogías pero ignoran información global.

GloVe entrena embeddings **directamente sobre la matriz de co-ocurrencia global** con una pérdida cuadrática ponderada cuya derivación es elegante y produce embeddings competitivos o superiores a Skip-gram.

---

## 1. Contexto

### 1.1 Las dos tradiciones pre-2014

#### Familia 1: factorización de matrices

**LSA** (Deerwester 1990): matriz palabra-documento, SVD, retener top-k componentes. Captura temas pero no sintaxis.

**HAL** (Lund & Burgess 1996): matriz palabra-palabra con ventana móvil. Problema citado en este paper: *"the most frequent words contribute a disproportionate amount to the similarity measure: the number of times two words co-occur with `the` or `and` will have a large effect on their similarity despite conveying relatively little about their semantic relatedness."*

**COALS** (Rohde 2006): normaliza por entropía o correlación antes de aplicar SVD.

**PPMI** (Bullinaria & Levy 2007): transformar matriz por positive pointwise mutual information, luego SVD. Estado del arte en distributional semantics pre-2013.

**HPCA / Hellinger PCA** (Lebret & Collobert 2014): transformación raíz cuadrada + PCA.

#### Familia 2: ventanas locales

**Bengio 2003 NPLM**: entrena embeddings como subproducto de un LM.

**Collobert & Weston 2008**: desacopla embeddings del LM, usa ranking loss.

**Word2Vec** (Mikolov 2013a/b): CBoW y Skip-gram con softmax aproximado.

**vLBL / ivLBL** (Mnih & Kavukcuoglu 2013): variantes log-bilineales.

### 1.2 Diagnóstico de Pennington et al.

Pennington identifica que ambas familias tienen el mismo objetivo profundo — explotar estadísticas de co-ocurrencia — pero **lo hacen mal por razones opuestas**:

- LSA/HAL usan la matriz global pero pesan mal las celdas (palabras frecuentes dominan).
- Word2Vec usa ventanas locales y nunca ve la estadística agregada.

La pregunta del paper: **¿se puede entrenar directamente sobre la matriz de co-ocurrencia con una función de pérdida que evite los problemas de LSA?**

---

## 2. La idea central — análisis del ratio de co-ocurrencia

Esta es la sección más original del paper (sección 3) y vale la pena seguir el razonamiento paso a paso.

### 2.1 Notación

- $X \in \mathbb{N}^{|V| \times |V|}$: matriz de co-ocurrencia. $X_{ij}$ = número de veces que la palabra $j$ aparece en el contexto de $i$ (con ventana, e.g., $\pm 10$).
- $X_i = \sum_k X_{ik}$: total de contextos de $i$.
- $P_{ij} = P(j \mid i) = X_{ij}/X_i$: probabilidad condicional.

### 2.2 La observación clave — el ratio $P_{ik}/P_{jk}$

**Tabla 1 del paper** con corpus de 6B tokens:

| Probabilidad y ratio | $k=$ solid | $k=$ gas | $k=$ water | $k=$ fashion |
|---|---|---|---|---|
| $P(k \mid \text{ice})$ | $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-5}$ |
| $P(k \mid \text{steam})$ | $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$ | $1.8 \times 10^{-5}$ |
| **Ratio $P(k\|\text{ice})/P(k\|\text{steam})$** | **8.9** | **0.085** | **1.36** | **0.96** |

**Lectura**:
- Para $k =$ "solid" (relacionado con ice, no steam): ratio grande (8.9).
- Para $k =$ "gas" (relacionado con steam, no ice): ratio chico (0.085).
- Para $k =$ "water" (relacionado con ambos): ratio cerca de 1 (1.36).
- Para $k =$ "fashion" (irrelevante para ambos): ratio cerca de 1 (0.96).

**Conclusión central**: el ratio **distingue mejor** que las probabilidades absolutas. Las palabras irrelevantes (ruido) se cancelan en el ratio.

### 2.3 De la observación al modelo

El paper postula que el modelo $F$ debe satisfacer:

$$
F(\mathbf{w}_i, \mathbf{w}_j, \tilde{\mathbf{w}}_k) = \frac{P_{ik}}{P_{jk}}. \quad (1)
$$

Donde $\mathbf{w}$ son los embeddings de "palabras" y $\tilde{\mathbf{w}}$ son los embeddings de "contextos" (dos matrices distintas, similar a Word2Vec input/output).

#### Paso 1: forma vectorial

La operación natural en espacios vectoriales es la diferencia. Restringen $F$ a depender solo de la diferencia $\mathbf{w}_i - \mathbf{w}_j$:

$$
F(\mathbf{w}_i - \mathbf{w}_j, \tilde{\mathbf{w}}_k) = \frac{P_{ik}}{P_{jk}}. \quad (2)
$$

Justificación: si los embeddings codifican analogías como `vec(king) - vec(queen) ≈ vec(man) - vec(woman)`, la información relevante vive en las diferencias.

#### Paso 2: hacer el argumento escalar

El lado derecho es escalar. Para que el lado izquierdo también lo sea, toman el producto punto:

$$
F\left( (\mathbf{w}_i - \mathbf{w}_j)^T \tilde{\mathbf{w}}_k \right) = \frac{P_{ik}}{P_{jk}}. \quad (3)
$$

Justificación: *"prevents $F$ from mixing the vector dimensions in undesirable ways"*. Una red neuronal arbitraria podría mezclar dimensiones y destruir la estructura lineal.

#### Paso 3: simetría palabra ↔ contexto

La distinción entre "palabra" y "contexto" es **arbitraria** — podemos intercambiar $\mathbf{w} \leftrightarrow \tilde{\mathbf{w}}$ y $X \leftrightarrow X^T$. Para que la fórmula final sea invariante a este intercambio, imponen que $F$ sea un **homomorfismo entre grupos**:

$$
F\left( (\mathbf{w}_i - \mathbf{w}_j)^T \tilde{\mathbf{w}}_k \right) = \frac{F(\mathbf{w}_i^T \tilde{\mathbf{w}}_k)}{F(\mathbf{w}_j^T \tilde{\mathbf{w}}_k)}. \quad (4)
$$

i.e., $F: (\mathbb{R}, +) \to (\mathbb{R}_{>0}, \times)$ con $F(a-b) = F(a)/F(b)$. La única solución es $F = \exp$.

Sustituyendo:
$$
\exp(\mathbf{w}_i^T \tilde{\mathbf{w}}_k) = P_{ik} = \frac{X_{ik}}{X_i}.
$$

Tomando log:
$$
\mathbf{w}_i^T \tilde{\mathbf{w}}_k = \log X_{ik} - \log X_i. \quad (6)
$$

#### Paso 4: absorber $\log X_i$ en un bias

$\log X_i$ depende solo de $i$ → se absorbe como bias $b_i$. Por simetría agregan $\tilde{b}_k$:

$$
\mathbf{w}_i^T \tilde{\mathbf{w}}_k + b_i + \tilde{b}_k = \log X_{ik}. \quad (7)
$$

Esta es la **predicción** del modelo. La pérdida es el error cuadrático entre predicción y verdad, ponderada por una función $f(X_{ik})$.

#### Paso 5: pérdida ponderada

$$
\boxed{\mathcal{J} = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( \mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2.} \quad (8)
$$

Esta es **la fórmula central de GloVe** (la que aparece en slide 36 de la clase).

---

## 3. La función de peso $f(X_{ij})$

### 3.1 Por qué necesitamos $f$

Tres problemas si usáramos $f \equiv 1$:

1. **Divergencia**: cuando $X_{ij} = 0$, $\log X_{ij} = -\infty$. Hay que excluir o suavizar los ceros.
2. **Co-ocurrencias raras dominan**: si una palabra rara co-ocurre con otra una vez, contribuye con el mismo peso que un par muy frecuente.
3. **Co-ocurrencias muy frecuentes dominan**: $(\text{the}, \text{is})$ co-ocurre $10^6$ veces. Sin peso, domina la loss.

### 3.2 Desiderata para $f$

El paper enuncia 3 condiciones:

1. $f(0) = 0$. Más estrictamente: $\lim_{x \to 0} f(x) \log^2 x$ es finito (para que el término $(X_{ij} = 0)$ no contribuya).
2. $f(x)$ no-decreciente (co-ocurrencias raras no deben dominar).
3. $f(x)$ acotada para $x$ grande (co-ocurrencias muy frecuentes tampoco deben dominar).

### 3.3 La elección concreta

$$
f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{si } x < x_{\max}, \\ 1 & \text{si } x \geq x_{\max}. \end{cases}
$$

**Hiperparámetros**:
- $x_{\max} = 100$ (el modelo es **débilmente sensible** a este valor).
- $\alpha = 3/4$ — exactamente el mismo exponente que en negative sampling de Word2Vec.

El paper observa: *"It is interesting that a similar fractional power scaling was found to give the best performance in Mikolov et al. (2013a)."* — sugiere que el exponente 3/4 es una propiedad robusta de las estadísticas Zipf de los corpus naturales, no un artefacto de un algoritmo específico.

### 3.4 Gráfica

```
f(x)
 1 │              ┌────────────
   │            ╱
0.5│         ╱
   │      ╱
 0 │___╱
   └───┴──────────── x
       x_max=100
```

Penaliza co-ocurrencias raras (rampa creciente) y satura para evitar que frecuencias enormes dominen.

---

## 4. Relación con Skip-gram

Sección 3.1 del paper hace algo muy elegante: **derivar Skip-gram desde GloVe** y mostrar que es un caso particular subóptimo.

Si Skip-gram se interpreta como un objetivo global (no streaming), corresponde a:

$$
\mathcal{J}_{\text{SG-global}} = -\sum_{i \in \text{corpus}, j \in \text{ctx}(i)} \log Q_{ij}, \quad Q_{ij} = \frac{\exp(\mathbf{w}_i^T \tilde{\mathbf{w}}_j)}{\sum_k \exp(\mathbf{w}_i^T \tilde{\mathbf{w}}_k)}.
$$

Agregando términos con el mismo $i$ y $j$ (porque $\log Q_{ij}$ depende solo del par):

$$
\mathcal{J}_{\text{SG-global}} = -\sum_{i=1}^{|V|} \sum_{j=1}^{|V|} X_{ij} \log Q_{ij} = \sum_i X_i \cdot H(P_i, Q_i),
$$

donde $H(P, Q)$ es la cross-entropy entre las distribuciones empírica $P_i$ y modelada $Q_i$.

**Crítica del paper**: cross-entropy con $Q$ normalizada vía softmax exacto es costosa. Si reemplazamos cross-entropy por least squares de los **logaritmos**, evitamos normalizar y obtenemos:

$$
\hat{\mathcal{J}} = \sum_{i,j} X_i \cdot (\log P_{ij} - \log Q_{ij})^2 = \sum_{i,j} X_i \cdot (\mathbf{w}_i^T \tilde{\mathbf{w}}_j - \log X_{ij})^2.
$$

Falta la libertad en el peso. Reemplazando $X_i$ por $f(X_{ij})$ general → ecuación (8) de GloVe.

**Conclusión**: GloVe = Skip-gram con (i) cross-entropy reemplazada por least squares de logaritmos, (ii) factor de peso $X_i$ generalizado a $f(X_{ij})$. Ambos cambios mejoran la calidad y eficiencia.

---

## 5. Complejidad

Sección 3.2: el cómputo escala con $|X|_{nnz}$ = número de entradas no-cero de la matriz $X$.

- En el peor caso, $|X|_{nnz} = O(|V|^2)$ — para $|V| = 400k$, eso son $1.6 \times 10^{11}$ entradas, inviable.
- En la práctica, las co-ocurrencias siguen una **ley de potencia**: $X_{ij} \sim k/r_{ij}^\alpha$, donde $r_{ij}$ es el rank del par por frecuencia y $\alpha \approx 1.25$.
- El paper deriva (ecuación 22): $|X|_{nnz} = O(|V| \cdot |C|^{1/\alpha}) = O(|V|^{0.8})$ donde $|C|$ es el tamaño del corpus.

Resultado: **GloVe escala como $O(|C|^{0.8})$**, mejor que el peor caso $O(|V|^2)$ y comparable o mejor que Skip-gram que escala como $O(|C|)$.

En la práctica: para Wikipedia (6B tokens) la matriz se construye en una pasada y ocupa decenas de GB en formato denso, decenas de MB en formato sparse.

---

## 6. Experimentos

### 6.1 Word analogies

**Tabla 2 del paper** (selección):

| Modelo | Dim | Size | Sem. | Syn. | Total |
|---|---|---|---|---|---|
| ivLBL | 100 | 1.5B | 55.9 | 50.1 | 53.2 |
| HPCA | 100 | 1.6B | 4.2 | 16.4 | 10.8 |
| **GloVe** | **100** | **1.6B** | **67.5** | **54.3** | **60.3** |
| SG (word2vec) | 300 | 1B | 61 | 61 | 61 |
| CBOW | 300 | 1.6B | 16.1 | 52.6 | 36.1 |
| vLBL | 300 | 1.5B | 54.2 | 64.8 | 60.0 |
| ivLBL | 300 | 1.5B | 65.2 | 63.0 | 64.0 |
| **GloVe** | **300** | **1.6B** | **80.8** | **61.5** | **70.3** |
| SVD-L | 300 | 6B | 56.6 | 63.0 | 60.1 |
| CBOW | 300 | 6B | 63.6 | 67.4 | 65.7 |
| SG | 300 | 6B | 73.0 | 66.0 | 69.1 |
| **GloVe** | **300** | **6B** | **77.4** | **67.0** | **71.7** |
| CBOW | 1000 | 6B | 57.3 | 68.9 | 63.7 |
| SG | 1000 | 6B | 66.1 | 65.1 | 65.6 |
| SVD-L | 300 | 42B | 38.4 | 58.2 | 49.2 |
| **GloVe** | **300** | **42B** | **81.9** | **69.3** | **75.0** |

GloVe domina en casi todas las configuraciones, especialmente en **analogías semánticas**.

### 6.2 Word similarity

Evaluado en WordSim-353, MC, RG, SCWS, RW. GloVe gana o empata con Skip-gram en todos.

### 6.3 NER (CoNLL-2003)

Embeddings como features para un clasificador CRF de NER. GloVe da F1 + 1 punto sobre Skip-gram.

### 6.4 Ablations

- **Dimensión**: ganancia rápida hasta 200, plateau después.
- **Tamaño de ventana**: ventanas chicas (5) favorecen sintaxis; ventanas grandes (10) favorecen semántica.
- **Ventana simétrica vs asimétrica**: símetrica (palabras a ambos lados) es ligeramente mejor.
- **Pesos de ventana**: el paper usa $1/d$ donde $d$ es la distancia a la palabra central. Esto da más peso a contextos cercanos.

---

## 7. Embeddings preentrenados publicados

Stanford publicó embeddings entrenados en distintos corpus, descargables desde https://nlp.stanford.edu/projects/glove/:

| Nombre | Corpus | Vocab | Dim | Tamaño |
|---|---|---|---|---|
| `glove.6B` | Wikipedia 2014 + Gigaword 5 (6B tokens) | 400k | 50/100/200/300 | 822 MB |
| `glove.42B.300d` | Common Crawl uncased (42B) | 1.9M | 300 | 1.75 GB |
| `glove.840B.300d` | Common Crawl cased (840B) | 2.2M | 300 | 2.03 GB |
| `glove.twitter.27B` | Twitter (27B tokens) | 1.2M | 25/50/100/200 | 1.42 GB |

Estos archivos se usaron como **embeddings de inicialización** en miles de modelos de NLP entre 2014 y 2018.

---

## 8. Limitaciones

1. **Memoria**: la matriz $X$ puede ser TB para corpus muy grandes. Stanford solucionó esto con un construct-and-stream en C.
2. **Sin manejo de OOV**: igual que Word2Vec, una palabra no en el vocab no tiene embedding. Sin solución hasta FastText.
3. **Embedding no contextual**: igual que Word2Vec, un único vector por palabra. La polisemia se promedia.
4. **Sin información subword**: igual problema que Word2Vec con morfología.
5. **Ventana fija**: no captura dependencias largas. Skip-Thought, ELMo, BERT lo resuelven.
6. **Asume estacionariedad**: el modelo asume que $P(j \mid i)$ es estable en el corpus. Para corpus multi-dominio (e.g., literatura científica + tweets), un único embedding promedia ambos.

---

## 9. Impacto y legado

### 9.1 Adopción

GloVe se convirtió en el segundo "estándar de oro" de embeddings preentrenados junto con Word2Vec. En la era 2014-2018:

- **Diccionario común**: muchos artículos reportaban resultados con ambos para robustez.
- **Inicialización de redes**: word embeddings de Glove o Word2Vec eran la primera capa de cualquier RNN/CNN para NLP.
- **NER, parsing, sentiment**: la mayoría de SOTA usaban GloVe.

### 9.2 Insights teóricos que sobreviven

1. **Factorización implícita de log-co-ocurrencia**: la idea de que los embeddings densos son una factorización de baja-rango de matrices basadas en co-ocurrencia. Levy & Goldberg 2014 hicieron explícito que SGNS también factoriza una versión shiftada de PMI — ver `Levy-Goldberg-SGNS-MF-2014.md`.

2. **Loss cuadrática log-bilineal**: el patrón "$\mathbf{w}^T \mathbf{u} \approx \log f(\text{count})$" reaparece en muchos modelos:
   - **Item embeddings** en recomendadores (factorización de matrices binarizadas con frecuencia logarítmica).
   - **Embeddings de grafos** (TransE: $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$).
   - **Embeddings de tabla** (TabNet, etc.).

3. **Función $f$ con saturación**: la idea de pesar diferencialmente raros vs frecuentes apareció en **focal loss** (Lin 2017), **importance weighting**, **curriculum learning**.

### 9.3 Críticas

- **Sin garantía teórica de convergencia rigurosa**: la derivación motivacional es informal; la convergencia se demuestra empíricamente.
- **Memoria intensiva**: hasta el día de hoy, entrenar GloVe en Common Crawl requiere terabytes.
- **Eclipsado por contextual embeddings**: a partir de ELMo (2018) y BERT (2018), los embeddings no-contextuales perdieron relevancia para tareas downstream serias. GloVe sobrevive como baseline, en proyectos low-resource, y en aplicaciones donde se necesita un único vector por palabra (e.g., diccionarios bilingües).

---

## 10. Conexión con la clase 18

Slides 35-36 cubren GloVe:

- **Slide 35**: portada del paper.
- **Slide 36**: fórmula central
  $$\mathcal{J} = \sum_{i,j=1}^{V} f(X_{ij}) (\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$
  + descripción "se busca aprender word embeddings cuyo producto punto aproxime el nivel de co-ocurrencia empírico de las respectivas palabras".

Lo que el slide NO menciona y vale la pena entender:
- La **derivación** desde el ratio $P_{ik}/P_{jk}$ (5 pasos de la sección 2 de este análisis).
- La **función de peso** $f$ con $\alpha = 3/4$.
- La **conexión con Skip-gram** vía cross-entropy → least squares.

---

## 11. Cita BibTeX

```bibtex
@inproceedings{pennington-etal-2014-glove,
    title = "{G}lo{V}e: Global Vectors for Word Representation",
    author = "Pennington, Jeffrey  and Socher, Richard  and Manning, Christopher",
    booktitle = "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = oct,
    year = "2014",
    address = "Doha, Qatar",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D14-1162",
    doi = "10.3115/v1/D14-1162",
    pages = "1532--1543",
}
```

---

## 12. Frase para recordar

> *"Global word-word co-occurrence counts, locally efficient regression."* — GloVe rompe la dicotomía global/local mostrando que se puede aprovechar lo mejor de los dos mundos: estadística completa del corpus, optimización por SGD sobre celdas no-cero.

---

## 13. Notas técnicas

- **Embedding final** = $\mathbf{w} + \tilde{\mathbf{w}}$, no solo $\mathbf{w}$. El paper recomienda promediar ambos por simetría.
- **Inicialización**: uniforme en $[-0.5/d, 0.5/d]$, biases en cero.
- **Optimizer**: AdaGrad con learning rate inicial 0.05.
- **Iteraciones**: 50 epochs típicamente.
- **Implementación referencia**: https://github.com/stanfordnlp/GloVe (C optimizado).
- **Wrapper Python**: `glove-python-binary`, `pyglove`, o vía Gensim para cargar vectores preentrenados.
