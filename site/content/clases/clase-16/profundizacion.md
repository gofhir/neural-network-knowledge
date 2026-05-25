---
title: "Profundizacion - Estadistica del Lenguaje, TF-IDF y Limites de BoW"
weight: 20
math: true
---

> Este documento profundiza los fundamentos matematicos detras de la Clase 16.
> Cubre la derivacion de las leyes de Zipf y Heaps desde principios de minimo esfuerzo,
> la formalizacion del TF-IDF como medida de informacion,
> el algoritmo del Porter stemmer en sus cinco fases,
> y un analisis riguroso de los limites de BoW que motiva los embeddings de la [Clase 18](/clases/clase-18) (Word2Vec, GloVe, Skip-Thought).

---

# Parte I: Estadistica del Lenguaje

---

## 1. Ley de Zipf: Derivacion y Justificacion

### 1.1 Forma funcional

Sea $f(k)$ la frecuencia (numero de ocurrencias) de la palabra de rango $k$ en un corpus. La Ley de Zipf afirma:

$$f(k) = \frac{C}{k^s}$$

con $C$ constante de normalizacion y exponente $s \approx 1$. Para un corpus de $N$ tokens y vocabulario de tamano $V$:

$$\sum_{k=1}^{V} f(k) = N \quad \Rightarrow \quad C = \frac{N}{H_V^{(s)}}$$

donde $H_V^{(s)} = \sum_{k=1}^{V} 1/k^s$ es el numero armonico generalizado. Para $s = 1$ y $V$ grande, $H_V^{(1)} \approx \ln V + \gamma$ (con $\gamma$ constante de Euler-Mascheroni $\approx 0.577$).

### 1.2 Forma log-log

Tomando logaritmo:

$$\log f(k) = \log C - s \log k$$

En un grafico log-log, Zipf es una **recta de pendiente $-s$**. Esta es la verificacion empirica estandar: ajustar regresion lineal sobre $(\log k, \log f(k))$.

### 1.3 Principio de minimo esfuerzo (Zipf 1949)

Zipf justifico la ley desde un **trade-off** entre dos esfuerzos:

- **Esfuerzo del hablante**: prefiere vocabulario pequeno (pocas palabras polisemicas reutilizadas).
- **Esfuerzo del oyente**: prefiere vocabulario grande (cada palabra con significado univoco).

El equilibrio entre ambos produce una distribucion de tipo potencia. Mandelbrot (1953) refino la formula:

$$f(k) = \frac{C}{(k + b)^s}$$

con $b \geq 0$ corrigiendo el ajuste para palabras de rango bajo.

### 1.4 Conexion con entropia

La distribucion Zipfiana implica que la **entropia de Shannon** del lenguaje converge:

$$H = -\sum_{k=1}^{V} p(k) \log p(k), \quad p(k) = \frac{1}{k^s H_V^{(s)}}$$

Para $s = 1$, $H \sim \log \log V$ -- crece muy lentamente con el vocabulario. Esto es lo que hace que **comprimir texto** funcione tan bien (Huffman, gzip): la mayor parte de los bits se concentran en las pocas palabras frecuentes.

### 1.5 Implicancia para NLP

- **Stop-words** (zona izquierda) son ruido para clasificacion → eliminar.
- **Rare-words** (cola larga) son features muy informativas pero **dispersas**: aparecen en pocos documentos. TF-IDF las pondera al alza (proximo bloque).
- El **largo de la cola** es indomable: sin importar cuan grande sea el corpus, siempre habra palabras nuevas (justificacion teorica para subword tokenization: [BPE, WordPiece](/fundamentos/bpe) -- se cubren en [Clase 20](/clases/clase-20)).

---

## 2. Ley de Heaps (Herdan)

### 2.1 Forma funcional

Si $n$ es el numero total de tokens y $V_R(n)$ el numero de tipos (palabras unicas) tras procesar $n$ tokens, entonces:

$$V_R(n) = K \cdot n^\beta, \quad \beta \in (0, 1)$$

Tipicamente $K \in [10, 100]$ y $\beta \in [0.4, 0.6]$ para corpus en lenguajes naturales.

### 2.2 Derivacion desde Zipf

Heaps no es independiente: **se deriva de Zipf**. Esquema de prueba:

Supongamos que en un corpus muy grande la frecuencia de la palabra de rango $k$ es $f(k) = C/k^s$ con $s = 1 + \epsilon$. La probabilidad de aparicion de cada palabra es $p(k) = 1/(k \cdot H_V)$.

El numero esperado de **tipos distintos** observados tras $n$ tokens, asumiendo $V$ infinito, se aproxima por:

$$E[V_R(n)] \;\approx\; \int_{1}^{\infty} \left(1 - (1 - p(k))^n\right) dk$$

Para $n$ grande y $s = 1 + \epsilon$ pequeno:

$$E[V_R(n)] \;\sim\; n^{1/s} = n^{1/(1+\epsilon)} \;\approx\; n^{1 - \epsilon}$$

Identificando $\beta = 1/s$, obtenemos $\beta < 1$ siempre que $s > 1$. La conexion **Zipf $\Leftrightarrow$ Heaps** es robusta y se verifica empiricamente.

### 2.3 Implicancia practica

- El vocabulario crece **sublinealmente**: duplicar el corpus no duplica el vocabulario, lo aumenta por factor $2^\beta \approx 1.3-1.5$.
- Justifica el uso de **vocabularios fijos truncados** (top-K palabras) en modelos clasicos.
- Justifica que los modelos modernos usen **subword units** -- ya que los tipos crecen sin limite, es mejor descomponer palabras en piezas reusables.

---

# Parte II: TF-IDF

> Esta seccion no aparece en las slides pero es la **extension natural** de BoW y sigue siendo la baseline numero uno en information retrieval.

---

## 3. Term Frequency - Inverse Document Frequency

### 3.1 Motivacion

BoW puro asigna a cada palabra un peso igual a su **conteo crudo**. Problema: las palabras comunes (`the`, `de`) dominan el vector aun tras eliminar stop-words. Las palabras realmente discriminativas (especificas de un documento) pesan poco.

**Idea de Salton & Buckley (1988)**: ponderar cada termino por dos factores:

- **TF** (term frequency): cuan frecuente es el termino **en este documento**.
- **IDF** (inverse document frequency): cuan **raro** es el termino **en el corpus** entero.

### 3.2 Definicion clasica

Sea $t$ un termino, $d$ un documento, $D = \{d_1, \ldots, d_N\}$ el corpus.

**Term frequency**:

$$\text{tf}(t, d) = \text{count}(t, d)$$

(o variantes: $1 + \log \text{count}$, frecuencia normalizada por largo del documento, etc.)

**Inverse document frequency**:

$$\text{idf}(t, D) = \log \frac{N}{|\{d \in D : t \in d\}|} = \log \frac{N}{\text{df}(t)}$$

donde $\text{df}(t)$ es el **document frequency** (numero de documentos que contienen $t$).

**TF-IDF**:

$$\text{tf-idf}(t, d, D) = \text{tf}(t, d) \cdot \text{idf}(t, D)$$

### 3.3 Variantes habituales

- **Smoothed IDF** (evita division por cero):

$$\text{idf}(t) = \log \frac{1 + N}{1 + \text{df}(t)} + 1$$

- **Sublinear TF** (compresion logaritmica):

$$\text{tf}(t, d) = 1 + \log \text{count}(t, d)$$

- **Normalizacion L2**: tras computar el vector tf-idf, normalizar a norma 1 para que documentos largos y cortos sean comparables por **similitud coseno**.

### 3.4 Justificacion teorica

IDF puede verse como una estimacion de **informacion mutua puntual** (PMI). Si modelamos la probabilidad de que un documento contenga $t$ como $p(t) = \text{df}(t) / N$:

$$\text{idf}(t) = -\log p(t)$$

Es decir, IDF es la **auto-informacion de Shannon** del evento "el termino $t$ aparece en un documento". Terminos raros tienen mucha informacion; terminos comunes, poca.

### 3.5 Ejemplo

Corpus de $N = 1000$ documentos. Termino *"banco"* aparece en 800 docs ($\text{df} = 800$), termino *"hipoteca"* en 50 docs.

- $\text{idf}(\text{banco}) = \log(1000/800) = \log(1.25) \approx 0.22$
- $\text{idf}(\text{hipoteca}) = \log(1000/50) = \log(20) \approx 3.0$

En un documento que menciona ambos terminos 3 veces:

- $\text{tf-idf}(\text{banco}) = 3 \cdot 0.22 = 0.66$
- $\text{tf-idf}(\text{hipoteca}) = 3 \cdot 3.0 = 9.0$

*Hipoteca* domina el vector, como queremos.

### 3.6 BM25: la generalizacion moderna

Robertson & Zaragoza (2009) propusieron **BM25** (Best Matching 25), refinamiento de TF-IDF que sigue siendo el **estandar de oro** en search engines (Elasticsearch, Lucene):

$$\text{BM25}(t, d) = \text{idf}(t) \cdot \frac{\text{tf}(t, d) \cdot (k_1 + 1)}{\text{tf}(t, d) + k_1 \cdot (1 - b + b \cdot |d|/\overline{|d|})}$$

con $k_1 \in [1.2, 2.0]$, $b \in [0, 1]$ (tipico $b = 0.75$). Saturacion de TF y normalizacion por largo de documento.

{{< concept-alert type="clave" >}}
TF-IDF y BM25 son **anteriores a las redes neuronales** y, sorprendentemente, siguen siendo competitivos como **baseline o componente** en sistemas modernos de retrieval (ej. retrieval-augmented generation con LLMs combina BM25 + vector search).
{{< /concept-alert >}}

---

# Parte III: Algoritmo del Porter Stemmer

---

## 4. Reglas del Porter Stemmer (1980)

El algoritmo de Porter es un **sistema de reescritura por reglas** sobre sufijos del ingles. Define un concepto clave: la **medida** $m$ de una palabra.

### 4.1 La medida $m$

Toda palabra se ve como una alternancia de **vocales (V)** y **consonantes (C)**:

$$[C](VC)^m[V]$$

donde $[\cdot]$ denota presencia opcional. La medida $m$ es el numero de pares VC consecutivos.

Ejemplos:

- `tree` → `(tr)(ee)` → $m = 0$ (no hay VC completo despues de C inicial).
- `trouble` → `(tr)(ou)(bl)(e)` → $m = 1$.
- `private` → `(pr)(i)(v)(a)(t)(e)` → $m = 2$.
- `oaten` → `(oa)(t)(e)(n)` → $m = 2$.

### 4.2 Las cinco fases

Porter aplica reglas en **5 pasos secuenciales**, cada uno reescribiendo sufijos comunes con condiciones sobre $m$.

**Paso 1a (plurales y -s)**:

| Sufijo | Reemplazo | Ejemplo |
|---|---|---|
| `sses` | `ss` | `caresses` → `caress` |
| `ies` | `i` | `ponies` → `poni` |
| `ss` | `ss` | `caress` → `caress` |
| `s` | `` | `cats` → `cat` |

**Paso 1b (-ed, -ing)**: Si la palabra cumple `(m > 0) EED → EE`:

| Condicion | Sufijo | Reemplazo | Ejemplo |
|---|---|---|---|
| $m > 0$ | `eed` | `ee` | `feed` → `feed` ($m = 0$, no aplica), `agreed` → `agree` |
| `*v* ed` | `ed` | `` | `plastered` → `plaster` |
| `*v* ing` | `ing` | `` | `motoring` → `motor` |

(`*v*` denota "contiene una vocal en el stem").

Tras eliminar `ed`/`ing`, post-proceso:

| Patron | Accion | Ejemplo |
|---|---|---|
| `at` | `→ ate` | `conflat` → `conflate` |
| `bl` | `→ ble` | `troubl` → `trouble` |
| `iz` | `→ ize` | `siz` → `size` |
| consonante doble final (no l, s, z) | eliminar uno | `hopp` → `hop` |
| `(m=1) *o` | agregar `e` | `fail` → `fail`, `hop` → `hope` |

**Paso 1c**: `(*v*) y → i`: `happy → happi`, `sky → sky` (no aplica, no contiene vocal en stem).

**Paso 2 (sufijos -tional, -enci, -izer, etc.)**: requiere $m > 0$.

| Sufijo | Reemplazo | Ejemplo |
|---|---|---|
| `ational` | `ate` | `relational` → `relate` |
| `tional` | `tion` | `conditional` → `condition` |
| `enci` | `ence` | `valenci` → `valence` |
| `izer` | `ize` | `digitizer` → `digitize` |
| `alli` | `al` | `radicalli` → `radical` |
| `ousness` | `ous` | `callousness` → `callous` |
| `ization` | `ize` | `realization` → `realize` |

**Paso 3 (sufijos -icate, -ative, etc.)**: requiere $m > 0$.

| Sufijo | Reemplazo | Ejemplo |
|---|---|---|
| `icate` | `ic` | `triplicate` → `triplic` |
| `ative` | `` | `formative` → `form` |
| `alize` | `al` | `formalize` → `formal` |
| `iciti` | `ic` | `electriciti` → `electric` |
| `ical` | `ic` | `electrical` → `electric` |
| `ful` | `` | `hopeful` → `hope` |

**Paso 4 (sufijos largos)**: requiere $m > 1$.

| Sufijo | Accion | Ejemplo |
|---|---|---|
| `al`, `ance`, `ence`, `er`, `ic`, `able`, `ible`, `ant`, `ement`, `ment`, `ent`, `ou`, `ism`, `ate`, `iti`, `ous`, `ive`, `ize` | eliminar | `revival` → `reviv`, `homologou` → `homolog` |

**Paso 5a**: `(m > 1) e → ` o `(m = 1, no *o) e → `: `probate → probat`, `rate → rate`.

**Paso 5b**: `(m > 1, *d *L) → eliminar uno`: `controll → control`.

### 4.3 Resultado final

La cadena pasa por las cinco fases, cada una aplicando a lo mas una regla. El output es el **stem** -- no necesariamente una palabra valida, pero suficiente para colapsar variantes morfologicas.

### 4.4 Variantes y limitaciones

- **Snowball (Porter2, 2001)**: el mismo Porter publico una version mejorada con framework para multiples idiomas.
- **Lancaster**: mas agresivo, raices mas cortas, mayor over-stemming.
- **Limitaciones**: solo morfologia regular del ingles; no maneja excepciones (`went → went` deberia ser `go`); falla en palabras compuestas.

---

# Parte IV: Limites de BoW y Motivacion para Embeddings

---

## 5. Que pierde BoW

### 5.1 Orden de palabras

*"el perro mordio al hombre"* y *"el hombre mordio al perro"* tienen el mismo vector BoW. La diferencia semantica es total. n-Grams mitigan parcialmente pero no escalan.

### 5.2 Sinonimia

*"automovil"* y *"coche"* son ortogonales en BoW: el espacio coseno distance entre ambos vectores one-hot es siempre $\sqrt{2}$, igual que la distancia entre *"coche"* y *"banano"*. BoW **no captura similitud semantica**.

### 5.3 Polisemia

*"banco"* (institucion financiera) y *"banco"* (asiento de plaza) reciben el mismo vector. El sentido no se distingue sin contexto. Modelos contextuales (BERT, ELMo) en la [Clase 20](/clases/clase-20) resuelven esto.

### 5.4 Dimensionalidad y dispersion

Vocabularios reales: $|V| \in [10^4, 10^6]$. BoW produce vectores **muy dispersos** (sparse): casi todos los componentes son cero. Aunque scipy maneja esto con sparse matrices, los modelos lineales sufren cuando el numero de features supera el de ejemplos -- requiere regularizacion fuerte.

### 5.5 Composicion no trivial

El significado de una frase no es **suma** de los significados de sus palabras. *"not good"* no es la suma de *"not"* + *"good"*. BoW es lineal y no captura composicion.

### 5.6 Necesidad de embeddings densos

Lo que necesitamos: una representacion **densa, de baja dimension** ($d \approx 100-1000$) donde:

- Palabras semanticamente similares estan **cerca** en el espacio.
- Operaciones algebraicas tengan sentido (`king - man + woman ≈ queen`).
- Se pueda **componer** vectores para frases.

Esto es exactamente lo que entregaran **[Word2Vec](/papers/word2vec-efficient-mikolov-2013)** (Mikolov 2013) y **[GloVe](/papers/glove-pennington-2014)** (Pennington 2014) en la [Clase 18](/clases/clase-18).

```mermaid
graph LR
    BOW[BoW / TF-IDF] -->|sparse, sin semantica| LIM[Limites]
    LIM --> EMB[Embeddings densos<br/>W2V, GloVe]
    EMB --> CTX[Embeddings contextuales<br/>ELMo, BERT, GPT]

    style BOW fill:#94a3b8,color:#000
    style EMB fill:#fbbf24,color:#000
    style CTX fill:#a78bfa,color:#fff
```

---

# Parte V: Hilo Conductor -- Clasico vs Moderno

---

## 6. Cuando usar cada cosa

> *"Don't use a cannon to kill a fly"* -- Confucio

| Tarea | Recomendacion |
|---|---|
| Filtro de spam corporativo | TF-IDF + logistic regression |
| Clasificacion de tickets de soporte (5 clases, 10K ejemplos) | TF-IDF + SVM lineal |
| Sentiment analysis general (cualquier dominio) | DistilBERT fine-tuned |
| Information retrieval / search | BM25 + reranker neural |
| Question answering open-domain | Retrieval (BM25) + generador (LLM) |
| Traduccion entre lenguas mayoritarias | Transformer pretrained (mBART, NLLB) |
| Traduccion lenguas low-resource | Transfer learning + back-translation |
| Resumen de documento corto | T5 / BART fine-tuned |
| Conversacion abierta | LLM grande (GPT-4, Claude) |

**Heuristica**: empezar siempre con baseline clasico (TF-IDF + clasificador lineal). Si funciona suficientemente bien, **detenerse**. Si no, escalar gradualmente: word embeddings → Transformer pequeno → Transformer grande → LLM.

---

## 7. Resumen Ejecutivo

1. **Zipf** ($f(k) \propto 1/k^s$, $s \approx 1$) y **Heaps** ($V \sim n^\beta$, $\beta \in (0,1)$) son leyes empiricas universales del lenguaje, conectadas analiticamente.
2. La forma de Zipf justifica las **tres zonas** del vocabulario (stop / key / rare) y motiva tanto la eliminacion de stop-words como la ponderacion IDF.
3. **TF-IDF** combina conteo local con rareza global; IDF es esencialmente la **auto-informacion** de Shannon. **BM25** es su refinamiento moderno.
4. **Porter stemmer** es un sistema de reescritura por reglas en 5 fases, basado en la medida $m$ del stem.
5. **Stemming vs lematizacion**: stemming es rapido pero produce strings invalidos; lematizacion es lento pero produce lemas validos.
6. **BoW** pierde orden, sinonimia, polisemia y composicion. n-Grams mitigan parcialmente pero no escalan.
7. La motivacion para **word embeddings** ([Clase 18](/clases/clase-18)) es construir representaciones **densas, de baja dimension**, donde la geometria capture semantica.
8. Heuristica practica: **empezar simple**. Las tecnicas clasicas son interpretables, baratas y frecuentemente suficientes.

---

## Referencias

- Zipf, G. K. (1949). *Human Behaviour and the Principle of Least Effort*. Addison-Wesley.
- Mandelbrot, B. (1953). An informational theory of the statistical structure of language. *Communication Theory*.
- Heaps, H. S. (1978). *Information Retrieval: Computational and Theoretical Aspects*. Academic Press.
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*.
- Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130-137.
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR*.
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

Volver a [Teoria](teoria) | Hub de la [Clase 16](/clases/clase-16).
