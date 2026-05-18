# Mikolov et al. 2013 — Distributed Representations of Words and Phrases and their Compositionality

| Campo | Valor |
|---|---|
| **Autores** | Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean |
| **Afiliación** | Google Inc., Mountain View |
| **Venue** | NeurIPS 2013 |
| **Fecha** | Octubre 2013 |
| **Pdf** | `Mikolov-Word2Vec-DistributedRepresentations-2013.pdf` (9 páginas) |
| **Citaciones** | >50.000 |
| **URL** | https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html |

> *"We describe several extensions [to Skip-gram] that improve both the quality of the vectors and the training speed."*

Este es el segundo paper de Word2Vec — extensiones al modelo Skip-gram del primer paper (`Mikolov-Word2Vec-Efficient-2013.md`). Introduce las cuatro innovaciones técnicas que **hicieron a Word2Vec realmente práctico a escala**:

1. **Negative sampling** — alternativa simple y rápida al softmax exacto.
2. **Subsampling de palabras frecuentes** — speedup 2-10× + mejores embeddings para palabras raras.
3. **Hierarchical softmax con árbol Huffman** — refinamiento del softmax jerárquico.
4. **Phrase embeddings** — manejar "New York", "Air Canada" como tokens únicos.

---

## 1. Contexto y motivación

El paper anterior (Mikolov 2013a, ICLR Workshop) ya había mostrado que Skip-gram con vectores 300-dim entrenado en 1B palabras superaba a NNLM. Pero quedaban tres problemas:

1. **Velocidad**: el softmax sobre $|V| = 10^6$ era el cuello de botella incluso después de eliminar la capa hidden.
2. **Palabras frecuentes** ("the", "of", "and") dominaban el gradiente sin aportar información semántica.
3. **Composicionalidad limitada**: `vec("New") + vec("York") ≠ vec("New_York")`. Frases idiomáticas no se podían representar.

Este paper resuelve los tres con técnicas pragmáticas — no propone una nueva arquitectura, sino refinamientos que **escalaron Word2Vec a 30B palabras y vocabulario de 700k+**.

---

## 2. Contribuciones técnicas

### 2.1 Negative Sampling (NEG) — la contribución más influyente

#### 2.1.1 El problema

El softmax exacto del Skip-gram:
$$
P(w_O \mid w_I) = \frac{\exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_I})}.
$$

El gradiente $\nabla \log P(w_O \mid w_I)$ requiere computar el denominador completo: $O(|V| \cdot N)$ por ejemplo. Para $|V| = 10^6$, $N = 300$: 300M flops por par (palabra, contexto). Inviable a escala.

#### 2.1.2 Noise Contrastive Estimation (NCE), el predecesor teórico

Gutmann & Hyvärinen (2010) propusieron NCE: en lugar de modelar $P(w | \text{ctx})$ explícitamente, **entrenar un clasificador binario** que distingue muestras reales (de los datos) de muestras de ruido (de una distribución conocida $P_n$). Mnih & Teh (2012) lo aplicaron a LMs.

NCE tiene la propiedad teórica de que **maximizar el objetivo NCE → maximizar log-softmax**. Es decir, NCE es una aproximación legítima al LM.

#### 2.1.3 Simplificación de Mikolov: Negative Sampling

Mikolov razona: si lo único que importan son los embeddings (no la calibración del LM), se puede **simplificar NCE** preservando solo lo esencial. El objetivo NEG es:

$$
\mathcal{L}_{\text{NEG}}(w_O, w_I) = \log \sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I}) \right].
$$

**Interpretación**:
- $\sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})$: probabilidad de que el par $(w_I, w_O)$ sea "real". Maximizar → producto punto alto.
- $\sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I})$ con $w_i$ ruido: probabilidad de que el par sea "falso". Maximizar → producto punto bajo.

Es **clasificación binaria** con $k$ negativos por positivo.

**Diferencia con NCE**:
- NCE necesita conocer $P_n(w)$ analíticamente (la incorpora a la log-prob).
- NEG solo necesita **muestrear** de $P_n(w)$ — no usa la probabilidad numérica.
- NEG **no aproxima el log-softmax**. Es un objetivo distinto, pero los embeddings resultantes son de calidad comparable o mejor para tareas downstream.

#### 2.1.4 Elección de $P_n$ — el famoso exponente 3/4

Mikolov probó:
- $P_n = \text{Uniform}(V)$: pésimo, las palabras frecuentes nunca aparecen como negativos.
- $P_n = U(w)$ (unigrama): mejor, pero las palabras muy frecuentes saturan.
- $P_n \propto U(w)^{3/4}$: **mejor**, comprime la distribución unigrama (palabras frecuentes pierden masa relativa, palabras raras ganan).

Cita textual del paper (sección 2.2):

> *"We investigated a number of choices for $P_n(w)$ and found that the unigram distribution $U(w)$ raised to the 3/4rd power (i.e., $U(w)^{3/4}/Z$) outperformed significantly the unigram and the uniform distributions, for both NCE and NEG on every task we tried including language modeling (not reported here)."*

El exponente 3/4 es **puramente empírico** — no hay justificación teórica conocida. Es uno de los "trucos mágicos" más citados de Word2Vec.

#### 2.1.5 Valor de $k$

- Datasets pequeños (1B palabras): $k = 5$ a $20$.
- Datasets grandes (>10B palabras): $k = 2$ a $5$ (suficiente porque cada par positivo se ve muchas veces).

#### 2.1.6 Costo computacional

NEG reduce el cómputo de $O(|V| \cdot N)$ por ejemplo a $O((k+1) \cdot N)$. Con $k=5$, $N=300$, $|V|=10^6$: **de 300M a 1800 ops** — speedup de ~$10^5$. Es lo que permitió entrenar en 30B palabras en un día.

### 2.2 Subsampling de palabras frecuentes

#### 2.2.1 Problema

En corpus reales, las palabras siguen una **distribución de Zipf**: las top 100 palabras representan ~50% del total. Aparecen tantas veces que dominan el gradiente.

Pero las co-ocurrencias `(the, France)` son poco informativas (the co-ocurre con todo), mientras que `(France, Paris)` es altamente informativa.

#### 2.2.2 Fórmula de subsampling

Cada ocurrencia de la palabra $w_i$ se **descarta** con probabilidad:

$$
P_{\text{discard}}(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}},
$$

donde $f(w_i)$ es la frecuencia relativa de $w_i$ en el corpus y $t$ es un umbral, típicamente $t = 10^{-5}$.

**Interpretación**:
- Si $f(w_i) \leq t$: $P_{\text{discard}} \leq 0$ → la palabra siempre se conserva.
- Si $f(w_i) \gg t$ (palabra muy frecuente): $P_{\text{discard}} \to 1$ → casi siempre se descarta.

Para $t = 10^{-5}$ y "the" con $f = 0.07$: $P_{\text{discard}} = 1 - \sqrt{10^{-5}/0.07} \approx 0.988$ — se descarta el 98.8% de las ocurrencias de "the".

#### 2.2.3 Resultados

**Tabla 1** del paper:

| Método | Tiempo [min] | Sintáctico [%] | Semántico [%] | Total [%] |
|---|---|---|---|---|
| NEG-5 (sin subsampling) | 38 | 63 | 54 | 59 |
| NEG-15 | 97 | 63 | 58 | 61 |
| HS-Huffman | 41 | 53 | 40 | 47 |
| NCE-5 | 38 | 60 | 45 | 53 |
| **Con $t=10^{-5}$ subsampling:** | | | | |
| NEG-5 | **14** | 61 | **58** | **60** |
| NEG-15 | 36 | 61 | **61** | **61** |
| HS-Huffman | **21** | 52 | **59** | 55 |

Observaciones del paper:
- Subsampling **da 2-10× speedup** (38 → 14 min para NEG-5).
- Subsampling **mejora la accuracy** en tareas semánticas (54 → 58 para NEG-5), porque libera al modelo de aprender redundantemente "the" en todos los contextos.

### 2.3 Hierarchical Softmax con árbol de Huffman

#### 2.3.1 Estructura

Construir un árbol binario donde **las hojas son las $|V|$ palabras** y los **nodos internos tienen vectores $\mathbf{v}'_n$**. La probabilidad de una palabra es el **producto de probabilidades binarias** a lo largo del camino desde la raíz:

$$
P(w \mid w_I) = \prod_{j=1}^{L(w)-1} \sigma\left( [\![n(w, j+1) = \text{ch}(n(w, j))]\!] \cdot \mathbf{v}'_{n(w,j)} \cdot \mathbf{v}_{w_I} \right)
$$

donde:
- $n(w, j)$ es el $j$-ésimo nodo en el camino desde la raíz a $w$.
- $\text{ch}(n)$ es el hijo "predeterminado" de $n$ (e.g., izquierdo).
- $[\![\cdot]\!]$ es 1 si la condición es true, -1 si false.
- $L(w)$ es la profundidad del camino.

**Propiedades**:
- $\sum_w P(w \mid w_I) = 1$ exacto (es un softmax exacto, solo factorizado).
- Cómputo de $P(w \mid w_I)$ y su gradiente: $O(L(w) \cdot N) \approx O(\log |V| \cdot N)$.
- **Una sola representación $\mathbf{v}_w$ por palabra**, más una representación $\mathbf{v}'_n$ por **nodo interno** (no por palabra). Esto es diferente del softmax exacto y de NEG, que tienen 2 representaciones por palabra.

#### 2.3.2 ¿Por qué Huffman?

Un árbol de Huffman asigna **caminos cortos a palabras frecuentes**. Esto es óptimo en el sentido de información: minimiza la longitud esperada del camino dado el unigrama.

**Resultado práctico**: una palabra como "the" tiene camino de ~5 nodos (frecuencia altísima), una palabra como "supernova" tiene camino de ~25 nodos. El cómputo promedio sigue siendo $O(\log |V|)$ pero está **sesgado a las palabras frecuentes**.

#### 2.3.3 Trade-offs HS vs NEG

| | Hierarchical Softmax | Negative Sampling |
|---|---|---|
| Naturaleza | Softmax exacto factorizado | Aproximación binaria |
| Costo | $O(\log V \cdot N)$ | $O(k \cdot N)$ |
| Parámetros | $|V| \cdot N$ + $(|V|-1) \cdot N$ (nodos) | $2 |V| \cdot N$ |
| Mejor para | Palabras raras | Palabras frecuentes, frases |
| Implementación | Más compleja (construir árbol) | Trivial |

El paper concluye (sección 3): **NEG-15 con subsampling es el ganador para analogías sintácticas y semánticas**, pero **HS con subsampling gana en analogías de frases** (NEG-15: 42%; HS: 47% en phrase analogies).

### 2.4 Phrase Embeddings

#### 2.4.1 Motivación

"Boston Globe" es un periódico — su significado no es composición de "Boston" + "Globe". Tratarla como token único permite a Skip-gram aprender un embedding específico para la frase.

Ejemplos del paper:
- `vec("Montreal Canadiens") - vec("Montreal") + vec("Toronto")` → `vec("Toronto Maple Leafs")`
- `vec("Air Canada") - vec("Canada") + vec("France")` → `vec("Air France")`

#### 2.4.2 Detección de frases

**Score bigrama** (sección 4, ecuación 6):
$$
\text{score}(w_i, w_j) = \frac{\text{count}(w_i w_j) - \delta}{\text{count}(w_i) \cdot \text{count}(w_j)}.
$$

- Numerador: bigramas **frecuentes**.
- Denominador: penaliza pares donde las palabras individuales también son frecuentes.
- $\delta$: descuento para evitar formar frases con palabras raras.

Pasos del proceso:
1. Calcular scores de todos los bigramas.
2. Reemplazar bigramas por encima de umbral por tokens únicos (`New_York` → `New_York_(token)`).
3. Repetir 2-4 pasadas con umbral decreciente para formar frases más largas: `San_Jose_Mercury_News`, `New_York_Times`.

#### 2.4.3 Resultados de phrase analogies

**Tabla 3** del paper:

| Método | Dim | Sin subsampling [%] | Con $t=10^{-5}$ subsampling [%] |
|---|---|---|---|
| NEG-5 | 300 | 24 | 27 |
| NEG-15 | 300 | 27 | 42 |
| HS-Huffman | 300 | 19 | **47** |

Con corpus de 30B palabras y modelo HS de 1000-dim: **72% accuracy** en phrase analogies.

#### 2.4.4 Composicionalidad aditiva — el "AND" semántico

Sección 5 del paper introduce un fenómeno sorprendente: `vec(Russia) + vec(river)` está cerca de `vec(Volga River)`. La explicación del paper:

> *"The word vectors are in a linear relationship with the inputs to the softmax nonlinearity... the values are related logarithmically to the probabilities computed by the output layer, so the sum of two word vectors is related to the product of the two context distributions. The product works here as the AND function: words that are assigned high probabilities by both word vectors will have high probability."*

**Formalización**: si $\mathbf{v}_w \cdot \mathbf{v}'_c \approx \log P(c \mid w)$ (aproximación válida para Skip-gram), entonces:
$$
(\mathbf{v}_{w_1} + \mathbf{v}_{w_2}) \cdot \mathbf{v}'_c \approx \log P(c \mid w_1) + \log P(c \mid w_2) = \log [P(c \mid w_1) \cdot P(c \mid w_2)].
$$

Las palabras $c$ que tienen alta probabilidad bajo **ambos** contextos $w_1$ y $w_2$ son las que rankean más alto en la suma. Esto es el "AND semántico".

Ejemplos espectaculares de la Tabla 5:
- `Czech + currency` → `koruna` (correcta), `Check crown`, `Polish zolty`.
- `Vietnam + capital` → `Hanoi`.
- `German + airlines` → `airline Lufthansa`, `carrier Lufthansa`, `flag carrier Lufthansa`.
- `Russian + river` → `Moscow`, `Volga River`, `upriver`.

---

## 3. Setup experimental

- **Corpus principal**: 1B palabras de noticias internas de Google (mismo del primer paper).
- **Corpus extendido**: ~33B palabras (también interno).
- **Vocab**: palabras con frecuencia ≥ 5 → 692k palabras.
- **Phrase vocab**: hasta ~3M tras 2-4 iteraciones de detección.
- **Dim**: 300 para experimentos principales, 1000 para el modelo final de phrases.
- **Ventana**: 5 para palabras, **toda la oración** para frases.

---

## 4. Comparación con otros modelos preentrenados

**Tabla 6** del paper compara nearest neighbors de palabras raras (Redmond, Havel, ninjutsu, graffiti, capitulate) en 4 modelos:

| Modelo (training time) | Para "Havel" |
|---|---|
| Collobert (50d, 2 meses) | plauen, dzerzhinsky, osterreich |
| Turian (200d, semanas) | Jewell, Arzu, Ovitz |
| Mnih (100d, 7 días) | Pontiff, Pinochet, Rodionov |
| **Skip-Phrase (1000d, 1 día)** | **Vaclav Havel, president Vaclav Havel, Velvet Revolution** |

Skip-Phrase domina en orden de magnitud — los otros modelos ni siquiera tienen "Havel" en un cluster reconocible, mientras que Skip-Phrase lo asocia correctamente con la Revolución de Terciopelo y su título presidencial.

---

## 5. Limitaciones del paper

1. **Detección de frases es greedy**: la heurística bigrama → token es simple y pierde frases con sintaxis flexible ("The New York Times" vs "New York Times").
2. **No comparación con métodos basados en conteos**: GloVe (2014) cubrirá este gap.
3. **No análisis teórico**: Word2Vec se presenta empíricamente. Levy & Goldberg 2014 (ver `Levy-Goldberg-SGNS-MF-2014.md`) suplirán el análisis formal.
4. **No reproducibilidad del corpus**: el dataset de Google News no es público. Los embeddings sí (`GoogleNews-vectors-negative300.bin`).
5. **Polisemia no atacada**: "apple" sigue siendo un único vector que mezcla fruta y empresa.
6. **No subwords**: "running", "ran", "runs" no comparten información morfológica.

---

## 6. Impacto y legado

### 6.1 Adopción industrial

- Los embeddings publicados (`GoogleNews-vectors-negative300.bin`, 1.6 GB) se usaron en **decenas de miles de proyectos**.
- Gensim implementó el algoritmo en Python y se convirtió en estándar.
- spaCy, FastText, AllenNLP y prácticamente todas las librerías NLP de la era 2014-2018 incluyeron loaders para estos embeddings.

### 6.2 Contribuciones que sobreviven

Tres ideas de este paper aparecen en arquitecturas modernas:

1. **Negative sampling** evolucionó a **InfoNCE** (van den Oord 2018), que es la pérdida central de:
   - **SimCLR** (Chen 2020) en visión.
   - **CLIP** (Radford 2021) en multimodal.
   - **Sentence-BERT** (Reimers 2019) en NLP.
   - **DPR** (Karpukhin 2020) en retrieval.

2. **Subsampling** se generaliza a **importance sampling** y **curriculum learning**: pesar ejemplos según su informatividad.

3. **Detección de phrases** → **subword tokenization** (BPE, WordPiece, Unigram) que es estándar en todo Transformer-LM.

### 6.3 Crítica posterior

- **Drozd 2016** (*"Word Embeddings, Analogies, and Machine Learning: Beyond king − man + woman = queen"*): mostraron que las analogías de Mikolov son **sesgadas** por el `arg max` con exclusión de las palabras de la query.
- **Bolukbasi 2016** (*"Man is to Computer Programmer as Woman is to Homemaker?"*): demostraron sesgos de género profundamente codificados en los embeddings preentrenados.
- **Mu & Viswanath 2018**: encontraron que los embeddings son **anisotrópicos** (concentrados en un cono estrecho), lo que degrada cosine similarity. Proponen post-processing por PCA-removal.

---

## 7. Conexión con la clase 18

Slide 32 (`Word2Vec`) menciona los dos algoritmos CBoW y Skip-gram y la "motivación: modelos más simples pueden escalar a datasets más grandes" — el espíritu central de este paper.

Slides 33-34 muestran los diagramas pero **no mencionan negative sampling**, hierarchical softmax, subsampling ni phrases — todos los temas centrales de este paper. Estos son los puntos que la `profundizacion.md` debe llenar.

El slide 28 menciona "autosupervisado" y "no estamos limitados a n-gramas" — son las consecuencias prácticas de las técnicas de este paper.

---

## 8. Cita BibTeX

```bibtex
@inproceedings{mikolov2013distributed,
  title={Distributed representations of words and phrases and their compositionality},
  author={Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg S and Dean, Jeff},
  booktitle={Advances in Neural Information Processing Systems},
  volume={26},
  year={2013},
  url={https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html}
}
```

---

## 9. Frase para recordar

> *"Negative sampling is the killer feature."* — la opinión común de la comunidad. Casi todo el contrastive learning moderno deriva de la formulación de NEG en este paper: distinguir señal de ruido vía clasificación binaria con sampling. Es el ancestro directo de InfoNCE, CLIP y Sentence-BERT.

---

## 10. Errata y notas técnicas

- El paper dice "$U(w)^{3/4}$" — en código de Word2Vec, esto se implementa con una **tabla precomputada**: muestrear de $U^{3/4}$ se reduce a indexar una tabla de tamaño $10^8$ donde cada palabra aparece proporcional a $U(w)^{3/4}$.
- La fórmula de subsampling tiene una versión alternativa en el código fuente: $P_{\text{keep}}(w) = \left(\sqrt{f(w)/t} + 1\right) \cdot \frac{t}{f(w)}$. Las dos fórmulas son **diferentes** pero ambas se usan. Levy 2015 documenta esta discrepancia.
- En código, el `dyn_window` (ventana variable) muestrea $R \sim \text{Uniform}\{1, \dots, c\}$ por par. Esto **pondera implícitamente las palabras cercanas más** — equivalente a una ventana con peso decreciente.
