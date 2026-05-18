# Mikolov et al. 2013 — Efficient Estimation of Word Representations in Vector Space

| Campo | Valor |
|---|---|
| **Autores** | Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean |
| **Afiliación** | Google Inc., Mountain View |
| **Venue** | ICLR 2013 Workshop (preprint arXiv:1301.3781) |
| **Fecha** | 16 enero 2013 (v1) / 7 septiembre 2013 (v3) |
| **Pdf** | `Mikolov-Word2Vec-Efficient-2013.pdf` (12 páginas) |
| **Citaciones** | >40.000 (uno de los papers de NLP más citados) |
| **URL** | https://arxiv.org/abs/1301.3781 |

> *"We propose two novel model architectures for computing continuous vector representations of words from very large data sets."*

Este es el primer paper de Word2Vec — el que introduce **CBoW** y **Skip-gram** como arquitecturas; el complementario (NeurIPS 2013, ver `Mikolov-Word2Vec-DistributedRepresentations-2013.md`) introduce negative sampling, subsampling y phrases. Juntos son la **piedra angular del NLP moderno**.

---

## 1. Contexto histórico (2010-2013)

Antes de Word2Vec, el panorama de representaciones de palabras tenía tres grandes líneas:

### 1.1 Modelos basados en conteos (LSA, HAL, PPMI)

**Latent Semantic Analysis** (Deerwester 1990): construir matriz palabra-documento, aplicar SVD, retener las top-k componentes. Bueno para temas, pobre para sintaxis y analogías.

**Hyperspace Analogue to Language** (Lund & Burgess 1996): matriz palabra-palabra con ventana móvil, también con descomposición. Captura más información local.

**Positive Pointwise Mutual Information** (Bullinaria & Levy 2007): construir matriz PPMI y aplicarle SVD. Era el SOTA en *distributional semantics* en 2012.

Limitación común: escalan mal con vocabularios grandes (SVD cuadrático en $|V|$). En 2013, ningún método de conteos podía entrenarse en >1B palabras.

### 1.2 Modelos basados en predicción (NPLM y derivados)

**Bengio 2003 — Neural Probabilistic Language Model**: el primer LM neuronal exitoso, con embeddings $C$ aprendidos como subproducto. Ver análisis separado en `Bengio-NPLM-2003.md`. Limitación: el softmax sobre $|V|$ hace que entrenar en >100M palabras sea inviable sin optimizaciones agresivas.

**Collobert & Weston 2008** (*"SENNA"*): proponen una arquitectura unificada para POS, NER, chunking, SRL con embeddings compartidos. Usaban **ranking loss** (margin-based) en vez de softmax: maximizar `score(real)` vs `score(corrupto)`. Fue el predecesor directo en filosofía de Word2Vec — entrenaron embeddings en Wikipedia con ranking en lugar de softmax exacto. Limitación: lento.

**Mnih & Hinton 2007/2009 — Hierarchical Log-Bilinear (HLBL)**: árbol binario aprendido para reducir softmax a $\log_2 |V|$. Es el predecesor del hierarchical softmax que usa Word2Vec.

**Schwenk 2007**: aplicó NNLM a speech recognition con éxito limitado por el costo computacional.

**Mikolov 2010 — RNNLM**: ver análisis separado en `Mikolov-RNN-LM-2010.md`. El propio Mikolov había estado trabajando en RNNLMs entrenables en cientos de millones de palabras, pero seguían siendo costosos.

### 1.3 Motivación del paper

Mikolov observó que en NNLM y RNNLM, **la mayor parte del cómputo se gastaba en la capa softmax y en las no-linealidades** (tanh), no en aprender los embeddings. Si lo que interesaba eran los embeddings (no el LM), se podía sacrificar la capacidad del modelo a cambio de **escalar a corpus 10-100× más grandes**, lo que en la práctica daba mejores embeddings.

La hipótesis central: *"simple models trained on huge amounts of data outperform complex systems trained on less data."* — una afirmación que se demostró cierta y se generalizó después en la era de los foundation models.

---

## 2. Contribución central

Dos nuevas arquitecturas para aprender word embeddings:

1. **Continuous Bag-of-Words (CBoW)**: predecir la palabra central dado el contexto.
2. **Continuous Skip-gram**: predecir el contexto dada la palabra central.

Ambas **eliminan la capa hidden** del NNLM clásico, dejando un modelo log-bilineal cuyo entrenamiento es órdenes de magnitud más rápido.

El paper también introduce:
- **Una tarea de evaluación nueva**: word analogies sintácticas y semánticas (5 categorías sintácticas + 9 semánticas, ~20k preguntas).
- **Comparaciones cuantitativas** vs NNLM (Bengio), RNNLM (Mikolov), Collobert-Weston, HLBL.

---

## 3. Arquitecturas en detalle

### 3.1 Notación

- $V$ = vocabulario, $|V|$ típicamente $10^5$ a $10^7$.
- $N$ = dimensión del embedding ($N = 50, 100, 300, 1000$ en los experimentos).
- $C$ = ventana de contexto (5 o 10).

Cada palabra tiene **dos representaciones**:
- $\mathbf{v}_w \in \mathbb{R}^N$ — *input vector*. Es lo que se exporta como "el word embedding".
- $\mathbf{v}'_w \in \mathbb{R}^N$ — *output vector*. Se descarta tras el entrenamiento o se promedia con $\mathbf{v}_w$.

### 3.2 Continuous Bag-of-Words (CBoW)

**Idea:** dado un contexto $\{w_{t-C}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+C}\}$, predecir $w_t$.

**Pipeline:**
1. Cada $w_i$ del contexto se mapea a $\mathbf{v}_{w_i}$.
2. Los embeddings se **promedian** (la suma del paper es equivalente al promedio salvo por escala):
   $$\mathbf{h} = \frac{1}{2C} \sum_{-C \leq j \leq C, j \neq 0} \mathbf{v}_{w_{t+j}}.$$
3. Se proyecta al vocabulario:
   $$P(w_t = k \mid \text{ctx}) = \frac{\exp(\mathbf{v}'_k \cdot \mathbf{h})}{\sum_{i=1}^{|V|} \exp(\mathbf{v}'_i \cdot \mathbf{h})}.$$
4. Loss: $-\log P(w_t \mid \text{ctx})$.

**¿Por qué "bag of words"?** Porque la suma/promedio es **invariante a permutación** del contexto. CBoW no tiene noción de orden de las palabras del contexto — es exactamente la "bag" del BoW clásico (clase 16) pero con embeddings continuos en vez de conteos.

**Costo por ejemplo**: $O(N + N \cdot |V|)$ — el segundo término domina.

### 3.3 Continuous Skip-gram

**Idea inversa:** dada $w_t$, predecir cada una de las palabras del contexto.

**Pipeline:**
1. $\mathbf{v}_{w_t}$ es el único input.
2. Para cada $j$ con $-C \leq j \leq C, j \neq 0$:
   $$P(w_{t+j} = k \mid w_t) = \frac{\exp(\mathbf{v}'_k \cdot \mathbf{v}_{w_t})}{\sum_{i=1}^{|V|} \exp(\mathbf{v}'_i \cdot \mathbf{v}_{w_t})}.$$
3. Loss: $-\sum_j \log P(w_{t+j} \mid w_t)$.

**Truco práctico para regularizar implícitamente:** el paper usa **ventana variable**. En cada step, se muestrea $R \sim \text{Uniform}\{1, \dots, C\}$ y se predicen solo las $R$ palabras a izquierda y $R$ a derecha. Esto pondera más las palabras cercanas (que aparecen en todos los $R$) que las lejanas — análogo a una ventana con peso decreciente.

**Costo por ejemplo**: $O(2C \cdot N \cdot |V|)$ — peor que CBoW por un factor $2C$, pero el paper lo justifica diciendo que cada palabra recibe más actualizaciones por iteración.

### 3.4 ¿Por qué eliminar el hidden layer es legítimo?

El NNLM clásico tiene:
- Capa input → embedding → concat de $n-1$ embeddings → $\mathbf{x}$ con dimensión $(n-1)N$.
- Capa hidden con tanh: $\mathbf{h} = \tanh(\mathbf{W}\mathbf{x} + \mathbf{b})$, dimensión $H$.
- Capa output: softmax sobre $|V|$.

Costo por ejemplo: $O((n-1)N \cdot H + H \cdot |V|)$, dominado por $H \cdot |V|$ con $H = 500$ y $|V| = 10^6$ → $5 \times 10^8$ ops.

Word2Vec elimina la capa hidden y reduce la "proyección" a la suma de embeddings (sin parámetros propios). Costo: $O(N \cdot |V|)$, con $N = 100$ y $|V| = 10^6$ → $10^8$ ops — 5× menos. Combinado con hierarchical softmax o negative sampling (segundo paper), se reduce a $O(N \cdot \log|V|)$ o $O(N \cdot K)$ con $K \approx 10$ — efectivamente $10^4$ ops, **5000× menos**.

Mikolov razona: la capa hidden capturaba "no-linealidades del LM" pero no era esencial para aprender similitud distribucional. Empíricamente, los embeddings resultantes son **mejores que los del NNLM** porque se entrenan con mucho más data.

---

## 4. Tarea de evaluación: word analogies

El paper introduce **el dataset de analogías** que se convirtió en estándar:

- **Sintácticas** (~8.000 preguntas, 5 categorías): plurales (`apple:apples :: car:?`), comparativos (`big:bigger :: small:?`), superlativos, verbos en presente/pasado, adjetivos→adverbios.
- **Semánticas** (~9.000 preguntas, 9 categorías): capitales (`Athens:Greece :: Oslo:?`), monedas (`Algeria:dinar :: USA:?`), familia (`brother:sister :: grandson:?`), género de ciudades, etc.

**Métrica:** dada `a:b :: c:?`, computar $\mathbf{x} = \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$ y devolver la palabra del vocabulario más cercana en **cosine similarity** (excluyendo $a, b, c$). Accuracy = % de respuestas exactas.

El test set está en `code.google.com/p/word2vec/source/browse/trunk/questions-words.txt`. Existe también el HuggingFace dataset `word2vec-google-news-300-analogies`.

**Crítica posterior** (Linzen 2016, Drozd 2016): el `arg max` excluyendo las palabras de la query infla artificialmente el accuracy — las analogías "se encuentran" porque las palabras de la query están descartadas. Pero el dataset sigue siendo el benchmark estándar de la era pre-BERT.

---

## 5. Experimentos clave

### 5.1 Setup

- **Corpus pequeño**: subset de Google News, ~6B palabras.
- **Corpus grande**: Google News completo, ~50B palabras (no se publicó el dataset; sí los embeddings).
- **Vocab**: 1M palabras más frecuentes.
- **Implementación**: C single-threaded inicialmente, luego paralelizado con **HogWild!** (lockless SGD, Niu 2011).

### 5.2 Resultados principales

**Tabla 1** del paper compara dimensionalidad y data:

| Dim | Train words | Sem. acc | Syn. acc | Tot. acc |
|---|---|---|---|---|
| 50 | 24M | 13.4 | 12.3 | 12.7 |
| 100 | 24M | 19.4 | 18.1 | 18.7 |
| 300 | 24M | 23.2 | 19.1 | 21.0 |
| 600 | 24M | 24.0 | 18.5 | 21.0 |

Observación crucial: a partir de 300 dimensiones, **más dimensiones no ayudan si no se aumenta el data**. Esto motiva la conclusión que se ha vuelto canónica: la **calidad de embeddings escala con data, no con parámetros**.

**Tabla 2** compara con NNLM (Bengio), RNNLM, Collobert-Weston:

| Modelo | Vector dim | Train words | Sem. acc | Syn. acc | Total |
|---|---|---|---|---|---|
| Collobert-Weston | 50 | 660M | 9.3 | 12.3 | 11.0 |
| Turian | 200 | 37M | 1.4 | 2.6 | 2.1 |
| Mnih | 100 | 37M | 1.6 | 8.5 | 5.4 |
| Mikolov NNLM | 100 | 6B | 23.2 | 53.0 | 39.8 |
| **CBoW** | 300 | 783M | 15.5 | 53.1 | 36.1 |
| **Skip-gram** | 300 | 783M | **50.0** | 55.9 | **53.3** |

Skip-gram con 783M palabras (subset) ya supera a NNLM con 6B en semántica. Con 6B palabras y dim 300, Skip-gram alcanza ~65% total.

### 5.3 Tiempo de entrenamiento

CBoW en 783M palabras: **40 minutos en 1 CPU** con 100 dim.
Skip-gram en 783M palabras: **40 minutos en 1 CPU** con 100 dim.
NNLM en 783M palabras: **>10 horas en 14 CPUs**.

Estas cifras explican por qué Word2Vec **explotó en adopción**: cualquier laboratorio podía entrenar embeddings de calidad estado del arte en su laptop.

### 5.4 Microsoft Research Sentence Completion Challenge

El paper también evalúa en MSR SCC (Zweig & Burges 2011), que pide elegir la palabra correcta entre 5 candidatos para completar una oración. Combinando Skip-gram con un RNNLM, alcanzan **58.9% accuracy**, superando el SOTA previo (55.4%).

---

## 6. Detalles de implementación útiles

### 6.1 Inicialización

Embeddings input: uniforme en $[-0.5/N, 0.5/N]$. Embeddings output: **cero**. Esta asimetría es deliberada — los gradientes iniciales fluyen mejor cuando solo un lado de la dot product está inicializado.

### 6.2 Optimizer

SGD con learning rate decreciente lineal: $\eta_t = \eta_0 \cdot \max(1 - t/T, 10^{-4})$, con $\eta_0 = 0.025$ para Skip-gram y $0.05$ para CBoW.

### 6.3 HogWild!

Múltiples threads actualizan los mismos embeddings sin locks. Funciona porque las actualizaciones son **dispersas** (cada ejemplo toca pocas filas) y los conflictos son raros. Reportan ~3-5× speedup con 12 threads.

### 6.4 Parameter sharing entre input y output

Ningún parameter sharing — input y output son matrices distintas. Esto es diferente del NPLM de Bengio, que **sí** comparte la matriz de embeddings con la output projection.

---

## 7. Limitaciones reconocidas

El paper es honesto sobre lo que **no** hace:

1. **No modelo de lenguaje**: si quieres $P(w_t | w_{<t})$, este modelo no te sirve directamente. Solo word embeddings.
2. **No orden de palabras**: CBoW promedia el contexto sin posiciones; Skip-gram trata cada palabra de la ventana por separado.
3. **No frases**: "New York" → "New" y "York" por separado. El segundo paper (NeurIPS 2013) resuelve esto.
4. **Sin información subword**: una palabra rara con morfología compleja como "antidisestablishmentarianism" se mapea a un único vector aprendido con pocas ocurrencias. FastText (Bojanowski 2016) resuelve esto.
5. **Embeddings no contextuales**: un único vector por palabra. La polisemia se promedia.
6. **Softmax exacto sigue siendo el cuello de botella** en este paper. Se resuelve en el siguiente con hierarchical softmax / negative sampling.

---

## 8. Impacto y legado

### 8.1 Adopción masiva

- **`code.google.com/p/word2vec`** (luego migrado a GitHub): código C original, accesible.
- **Gensim** (Řehůřek): re-implementación Python optimizada, gold standard de la comunidad.
- **GoogleNews-vectors-negative300.bin** (3M palabras, 300d): los embeddings preentrenados publicados por Mikolov se descargaron **millones de veces** y fueron el "ImageNet de NLP" durante 3-4 años.

### 8.2 Sucesores conceptuales directos

| Año | Modelo | Innovación sobre W2V |
|---|---|---|
| 2014 | **GloVe** | Factorización matricial global de co-ocurrencia |
| 2014 | **Doc2Vec / Paragraph Vectors** (Le & Mikolov) | Extensión a documentos |
| 2015 | **Skip-Thought Vectors** | Extensión a oraciones |
| 2016 | **FastText** | Subwords (n-gramas de caracteres) |
| 2018 | **ELMo** | Embeddings contextuales con biLSTM |
| 2018 | **BERT** | Embeddings contextuales con Transformer |

### 8.3 Impacto fuera de NLP

La idea de aprender embeddings densos con un objetivo predictivo simple migró a:

- **Recsys**: prod2vec, item2vec (Barkan 2016).
- **Grafos**: DeepWalk (Perozzi 2014), node2vec (Grover 2016) — aplican Skip-gram a paseos aleatorios sobre grafos.
- **Biología**: gene2vec, protein embeddings.
- **Code**: code2vec.

### 8.4 Insight unificador

Lo más importante de Word2Vec quizás no es la arquitectura sino el **principio**: *"entrenar predicciones simples sobre contexto a escala masiva → representaciones útiles"*. Este principio es **literalmente el mismo** que el de GPT y BERT, salvo escala y arquitectura. Word2Vec es la primera demostración convincente de que la autosupervisión a escala produce representaciones de calidad sin supervisión humana.

---

## 9. Conexión con la clase 18

Slides 31-34 son una recapitulación de este paper:

- Slide 31: portada y abstract.
- Slide 32: motivación ("modelos simples escalan mejor", "feature learning", **CBoW + Skip-gram**).
- Slide 33: diagrama CBoW (predicción central desde contexto).
- Slide 34: diagrama Skip-gram (predicción de contexto desde central).

La fórmula que aparece en slide 34 corresponde exactamente a la ecuación (2) de este paper: el softmax $P(w_O | w_I) = \exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) / \sum_w \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_I})$.

Las matrices $C$ y $H$ que el slide menciona ($C \in \mathbb{R}^{|V| \times m}$, $H \in \mathbb{R}^{m \times |V|}$) corresponden a las matrices de input embeddings y output embeddings respectivamente, mismo tamaño que en el paper.

**Lo que el slide NO menciona** (y se cubre en el paper companion):
- Negative sampling (slide implícito al hablar de "feature learning" pero sin detalle).
- Hierarchical softmax.
- Subsampling.
- Phrases.

Esos están en `Mikolov-Word2Vec-DistributedRepresentations-2013.md`.

---

## 10. Referencias clave del paper

Las referencias que cita y que vale la pena conocer:

1. **Bengio 2003 — NNLM**: el predecesor neuronal directo. → ver `Bengio-NPLM-2003.md`.
2. **Bengio 2007 — Schwenk Continuous space LMs**: aplica NNLM a speech.
3. **Collobert & Weston 2008**: arquitectura unificada con ranking loss.
4. **Mnih & Hinton 2009**: HLBL, hierarchical softmax.
5. **Mikolov 2010**: RNNLM, antecesor inmediato → ver `Mikolov-RNN-LM-2010.md`.
6. **Mikolov 2012**: tesis doctoral con todo el background.
7. **Turian 2010**: word representations as semi-supervised learning features.

---

## 11. Cita BibTeX

```bibtex
@inproceedings{mikolov2013efficient,
  title={Efficient estimation of word representations in vector space},
  author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  booktitle={International Conference on Learning Representations (Workshop)},
  year={2013},
  url={https://arxiv.org/abs/1301.3781}
}
```

---

## 12. Frase para recordar

> *"Logical algebra on word vectors"* — la frase que se hizo viral en 2013 cuando Mikolov mostró que `vec(King) - vec(Man) + vec(Woman) ≈ vec(Queen)`. Aunque la analogía es **estadística** (no lógica) y tiene limitaciones, capturó la imaginación de la comunidad y aceleró la adopción de embeddings densos en todo NLP.
