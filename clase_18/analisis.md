# Clase 18 — Modelos de lenguaje, Word2Vec, GloVe y Skip-Thought

Análisis exhaustivo del PDF `Clase18.pdf` (Pablo Messina, 41 slides). Documento de estudio interno antes de integrar al site. Cubre las 3 secciones del PDF y rellena gaps no explicitados en las slides pero necesarios para completitud (suavizado, perplejidad, negative sampling, hierarchical softmax, derivación de GloVe).

**Convenciones de notación:**

- $V$ = vocabulario, $|V|$ = tamaño del vocabulario.
- $w_t$ = palabra en posición $t$. $w_{i:j} = (w_i, w_{i+1}, \dots, w_j)$.
- $\mathbf{v}_w \in \mathbb{R}^m$ = embedding de la palabra $w$ (a veces llamado *input embedding*). $m$ es la dimensión del embedding (típicamente 50, 100, 200, 300).
- $\mathbf{u}_w \in \mathbb{R}^m$ = embedding *output* de $w$ (Word2Vec usa dos matrices distintas; GloVe usa $\tilde{w}$ con notación parecida).
- $C \in \mathbb{R}^{|V| \times m}$ matriz de input embeddings (cada fila = un $\mathbf{v}_w$).
- $H \in \mathbb{R}^{m \times |V|}$ matriz de output embeddings (cada columna = un $\mathbf{u}_w$).
- $\sigma(x) = 1/(1+e^{-x})$ sigmoide. $\text{softmax}(\mathbf{z})_i = e^{z_i}/\sum_j e^{z_j}$.

---

## Parte I — Modelos de lenguaje

### 1.1 Definición probabilística

Un **modelo de lenguaje (LM)** es una distribución de probabilidad sobre secuencias de tokens. Dado un vocabulario $V$ y una secuencia $w_{1:T} = (w_1, \dots, w_T)$ con $w_t \in V$:

$$
P_\theta : V^* \to [0,1], \quad \sum_{w_{1:T} \in V^T} P_\theta(w_{1:T}) = 1 \text{ para cada } T.
$$

Las slides usan ejemplos como `P(Hola) = 0.1`, `P(Hola, cómo estás?) = 0.05`, `P(supernova flor barroco saltar hola chao) ≈ 10⁻¹¹`. La observación pedagógica es la correcta: **secuencias bien formadas y semánticamente coherentes reciben más masa que ruido**, sin que el modelo distinga "gramática" de "significado" explícitamente — solo cuenta lo que vio en el corpus.

**¿Por qué probabilidades sobre secuencias y no sobre oraciones aisladas?** Porque la composicionalidad del lenguaje permite que **el mismo modelo P(w | contexto) sirva para múltiples tareas** vía la regla de la cadena. Eso es lo que hace al LM una herramienta unificadora — desde n-gramas hasta GPT-4, el objetivo formal es el mismo.

### 1.2 Probabilidad condicional y regla de la cadena

La regla de la cadena de probabilidades, aplicada a secuencias de tokens, es:

$$
P(w_1, w_2, \dots, w_T) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1, w_2) \cdots P(w_T \mid w_{1:T-1}) = \prod_{t=1}^{T} P(w_t \mid w_{1:t-1}).
$$

**Implicación operativa central:** modelar la distribución conjunta $P(w_{1:T})$ se reduce a modelar **una única función condicional** $P(w_t \mid w_{1:t-1})$. Toda la maquinaria de LMs modernos (RNN-LM, Transformer-LM, GPT) parametriza exactamente esta función.

Ejemplo numérico del PDF:
```
P(hola cómo estás) = P(hola) · P(cómo | hola) · P(estás | hola cómo)
```

### 1.3 Aplicaciones del LM

#### 1.3.1 Generación de lenguaje (NLG) — decoding

Dado un prefijo $X$, generar $w_0, w_1, \dots$ maximizando localmente $P(w_t \mid X, w_{0:t-1})$. Las slides muestran **greedy decoding**:

$$
w_t = \arg\max_{w \in V} P(w \mid X, w_{0:t-1}).
$$

Greedy es subóptimo (no maximiza $P(w_{0:T} \mid X)$ globalmente). Alternativas estándar:

| Método | Idea | Trade-off |
|---|---|---|
| **Greedy** | $\arg\max$ por paso | Rápido, repetitivo, puede colapsar |
| **Beam search** ($k=B$) | Mantener $B$ hipótesis parciales con mayor log-prob acumulada | Más diverso, $O(B \cdot |V|)$ por paso; sigue siendo "modo único" en LM neuronal |
| **Sampling** ($\sim P$) | Muestrear de la distribución completa | Diverso pero ruidoso |
| **Top-k sampling** | Muestrear del top-$k$ más probable | Compromiso |
| **Top-p / nucleus** (Holtzman 2020) | Muestrear del menor conjunto cuya masa acumulada $\geq p$ | Adaptativo al pico de la distribución |
| **Temperature** ($T$) | Escalar logits: $P_T(w) \propto \exp(\text{logit}(w)/T)$ | $T \to 0$ ≈ greedy, $T \to \infty$ ≈ uniforme |

El PDF solo presenta greedy, pero **para una implementación seria los embeddings se evalúan en escenarios con sampling o beam** (perplejidad no captura calidad de generación).

#### 1.3.2 Machine Translation

MT como LM condicional: $P(Y \mid X) = \prod_t P(y_t \mid X, y_{<t})$. Este es el corazón de seq2seq (Sutskever 2014) y luego del Transformer encoder-decoder. La slide 10 lo plantea como decoding sobre el LM condicionado a $X$ — exactamente el approach moderno.

#### 1.3.3 Otras aplicaciones

| Aplicación | Cómo usa el LM |
|---|---|
| **Spelling correction** | Reranking de candidatos por $P(\text{candidato} \mid \text{contexto})$ |
| **Document summarization** | LM condicional sobre el documento fuente |
| **Question answering** | LM extractivo (span) o generativo |
| **Sentence completion** | NLG directo |
| **Speech recognition** | $\arg\max_{w} P(w) \cdot P(\text{audio} \mid w)$ — Bayes, LM como prior |
| **Information retrieval** | Query likelihood: $P(q \mid d)$ |
| **Code completion** | LM sobre tokens de código |

### 1.4 Modelos de N-gramas

Aproximación Markoviana al LM: truncar el contexto a las últimas $N-1$ palabras. Un **modelo de n-gramas de orden N** asume:

$$
P(w_t \mid w_{1:t-1}) \approx P(w_t \mid w_{t-N+1:t-1}).
$$

(El PDF usa una convención ligeramente distinta — llama "N" al número de palabras de contexto, no al tamaño del n-grama. Voy a usar la convención estándar: un trigrama = 3 tokens = 2 de contexto + 1 de target.)

#### 1.4.1 Estimación MLE

Maximum likelihood estimator por conteos:

$$
P_{\text{MLE}}(w_t \mid w_{t-N+1:t-1}) = \frac{\text{count}(w_{t-N+1:t})}{\text{count}(w_{t-N+1:t-1})}.
$$

**Ejemplo trabajado** (slide 19): corpus
```
S1: the cat sat on the mat
S2: the dog sat on the cat
S3: the cat caught the mouse
```
$\text{count}(\text{"the"}) = 6$, $\text{count}(\text{"the cat"}) = 3$, luego $P(\text{cat} \mid \text{the}) = 3/6 = 0.5$.

#### 1.4.2 Problema de sparsity y suavizado

**El problema crítico no mencionado en el PDF:** la mayoría de los n-gramas posibles **nunca se observan**, lo que da $P = 0$ y rompe la regla de la cadena (un solo factor cero anula toda la oración). Esto se conoce como *zero-frequency problem* y motiva el suavizado.

**Laplace (add-one) smoothing:**
$$
P_{\text{Lap}}(w_t \mid w_{t-N+1:t-1}) = \frac{\text{count}(w_{t-N+1:t}) + 1}{\text{count}(w_{t-N+1:t-1}) + |V|}.
$$
Simple pero sesgado: suaviza demasiado para vocabularios grandes.

**Add-k smoothing:** reemplaza el "+1" por "+k" con $k < 1$ optimizado en validación.

**Backoff** (Katz 1987): si un trigrama no se vio, retroceder a bigrama; si bigrama no se vio, retroceder a unigrama, con factores de descuento.

**Kneser-Ney** (Kneser & Ney 1995, mejorado por Chen & Goodman 1998): el **estándar de oro** para LMs n-gram. Usa *continuation probability* $P_{\text{cont}}(w)$ = cuán diverso es el conjunto de contextos en los que $w$ aparece, no su frecuencia absoluta. Esto corrige el problema clásico: la palabra "Francisco" puede ser frecuente, pero casi siempre después de "San" — su $P_{\text{cont}}$ es baja porque su contexto es restringido.

$$
P_{\text{KN}}(w \mid h) = \frac{\max(\text{count}(h, w) - d, 0)}{\sum_{w'} \text{count}(h, w')} + \lambda(h) \cdot P_{\text{cont}}(w),
$$

donde $d \in (0,1)$ es un descuento absoluto, $\lambda(h)$ es masa redistribuida, y $P_{\text{cont}}(w) = |\{h': \text{count}(h', w) > 0\}| / |\{(h', w'): \text{count}(h', w') > 0\}|$ es la fracción de bigramas distintos terminados en $w$.

Kneser-Ney **modificado** (con tres descuentos $d_1, d_2, d_{3+}$ según el conteo) fue el SOTA para LMs estadísticos durante una década, hasta el NPLM de Bengio en 2003 y los LMs neuronales modernos.

#### 1.4.3 Perplejidad — la métrica fundamental que no aparece en el PDF

La **perplejidad** (PPL) es la métrica estándar para evaluar LMs. Sobre un conjunto de test de $T$ tokens:

$$
\text{PPL}(w_{1:T}) = P(w_{1:T})^{-1/T} = \exp\left( -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_{1:t-1}) \right).
$$

Interpretación: PPL es la **media geométrica inversa** de la probabilidad asignada a cada token; equivalente al exponencial de la **cross-entropy promedio**. Un LM con PPL = $k$ "duda entre $k$ palabras igualmente probables" en cada paso.

| Modelo | PPL en WikiText-103 (referencia histórica) |
|---|---|
| 5-gram Kneser-Ney | ~80 |
| LSTM | ~50 |
| Transformer-XL | ~18 |
| GPT-3 (zero-shot) | ~10-15 |

PPL conecta directamente con la **cross-entropy loss** de entrenamiento: minimizar cross-entropy es minimizar log-PPL. Por eso entrenar LMs neuronales con softmax + CE optimiza implícitamente PPL.

#### 1.4.4 Limitaciones n-grama (slide 20)

1. **Representación por IDs**: no hay similitud semántica entre palabras.
2. **N pequeño**: el contexto efectivo de un n-grama rara vez supera 5 (Kneser-Ney 5-gram). Más allá, todos los n-gramas son únicos.
3. **No generaliza** a combinaciones nunca vistas: "Me gusta comer manzanas" recibe $P = 0$ si solo se vio "Me gusta comer naranjas".
4. **Tamaño del modelo escala con el vocabulario**: $|V|^N$ entradas posibles → tablas hash gigantes (Google 5-gram dataset = 30 GB para inglés web).
5. **No captura dependencias largas**: "El gato que vimos ayer en el parque corre" — el verbo "corre" depende de "gato", a 6 tokens.

Estas limitaciones motivan la transición a **representaciones distribuidas**.

---

## Parte II — De representaciones discretas a distribuidas

### 2.1 One-hot vs embeddings — geometría del espacio

**One-hot encoding:** cada palabra $w_i$ es el vector $\mathbf{e}_i \in \{0,1\}^{|V|}$ con un 1 en la posición $i$ y ceros en el resto.

- Producto punto $\mathbf{e}_i \cdot \mathbf{e}_j = \delta_{ij}$ — todas las palabras son **mutuamente ortogonales**. "perro" y "gato" están a la misma distancia que "perro" y "supernova".
- Distancia $\|\mathbf{e}_i - \mathbf{e}_j\|_2 = \sqrt{2}$ para $i \neq j$ — la geometría no codifica significado.
- Dimensionalidad $|V|$ — típicamente $10^4$ a $10^6$.

**Distributed (dense) embeddings:** $\mathbf{v}_w \in \mathbb{R}^m$ con $m \ll |V|$, **aprendidos** de modo que palabras semánticamente relacionadas terminan cerca en el espacio.

- $\mathbf{v}_{\text{perro}} \cdot \mathbf{v}_{\text{gato}} > \mathbf{v}_{\text{perro}} \cdot \mathbf{v}_{\text{supernova}}$ típicamente.
- La similitud se mide con **cosine similarity**: $\cos(\mathbf{v}, \mathbf{u}) = \mathbf{v} \cdot \mathbf{u} / (\|\mathbf{v}\| \|\mathbf{u}\|)$.
- $m$ es de orden 50-1000.

El nombre "**distribuida**" viene de Hinton 1986: la representación de un concepto se **distribuye** entre varias dimensiones; cada dimensión no codifica un concepto interpretable sino una característica latente compartida entre muchas palabras. Es lo opuesto a una representación "localista" (one-hot).

### 2.2 Hipótesis distribucional (Firth, Harris)

Las slides motivan los embeddings con la frase "cercanía semántica" pero la base teórica es la **hipótesis distribucional**:

> "You shall know a word by the company it keeps." — J.R. Firth, 1957.

Formalmente (Harris 1954): si dos palabras aparecen en contextos similares, tienen significado similar. Esta hipótesis es **la justificación filosófica** de Word2Vec, GloVe, BERT y todos los modelos basados en contexto. Toda la era de embeddings densos descansa sobre ella.

### 2.3 Bengio 2003 — Neural Probabilistic Language Model

El paper *A Neural Probabilistic Language Model* (Bengio, Ducharme, Vincent, Jauvin — JMLR 2003) es el origen del paradigma de embeddings aprendidos. La slide 21 lo reproduce sin citarlo explícitamente — vale la pena entenderlo a fondo porque es la **arquitectura padre de Word2Vec y todos los LMs neuronales**.

**Setup:** predecir $w_t$ dadas las $n-1$ palabras previas $w_{t-n+1:t-1}$.

**Arquitectura:**
1. **Tabla de embeddings** $C \in \mathbb{R}^{|V| \times m}$. Cada palabra del contexto se mapea a su fila correspondiente: $C(w_i) \in \mathbb{R}^m$.
2. **Concatenación**: $\mathbf{x} = [C(w_{t-n+1}); \dots; C(w_{t-1})] \in \mathbb{R}^{(n-1)m}$.
3. **MLP**: $\mathbf{h} = \tanh(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$.
4. **Output**: $\mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2 \in \mathbb{R}^{|V|}$, y $P(w_t = i \mid w_{t-n+1:t-1}) = \text{softmax}(\mathbf{y})_i$.
5. **Loss**: cross-entropy negativa.

**Innovación central:** la matriz $C$ se **comparte entre todas las posiciones del contexto y se aprende junto con los pesos del MLP**. Después de entrenar el LM, $C$ es de bonus un conjunto de word embeddings.

**Por qué importa:**
- Generaliza a n-gramas no vistos vía similitud en el espacio de embeddings (resuelve la limitación 3 de la sección 1.4.4).
- El número de parámetros crece $O(|V| m + (n-1)m h + h |V|)$, **mucho mejor que $O(|V|^N)$** de tablas n-gram.
- Sentó la base para Word2Vec (que es esencialmente este modelo simplificado).

**Costo:** el softmax sobre $|V|$ es el cuello de botella ($O(|V| h)$ por ejemplo). Word2Vec atacará exactamente este problema.

### 2.4 RNN-LM (Mikolov 2010)

La slide 22 muestra un **RNN-LM** estilo Mikolov 2010 (*Recurrent Neural Network Based Language Model*). Reemplaza la ventana fija del NPLM por un estado oculto recurrente:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{v}_{w_t} + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h),
$$
$$
P(w_{t+1} = i \mid w_{1:t}) = \text{softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)_i.
$$

**Ventaja:** contexto teóricamente ilimitado. **Realidad:** vanishing gradients limitan dependencias efectivas a unas 10-20 palabras (mitigado con LSTM/GRU — ver fundamentos `lstm-gru.md` del site, ya integrados desde clase 12).

La slide 23 muestra un **Transformer multimodal** (CXR-Mate-RRG24) como teaser hacia el Transformer-LM moderno y conexión con el lab clínico de la clase. Conceptualmente es el mismo problema —parametrizar $P(w_t \mid w_{<t})$— pero con self-attention en vez de recurrencia.

### 2.5 Composicionalidad aditiva y analogías

La famosa observación de Mikolov 2013:

$$
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}.
$$

Equivalentemente: el vector $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}}$ codifica un "concepto" de realeza-sin-género, y aplicarlo a "woman" rinde "queen". Slides 25-26 lo ilustran con `Beijing - China + Russia ≈ Moscow`.

**Por qué funciona** (Allen & Hospedales, ICML 2019): bajo ciertas condiciones sobre la distribución de co-ocurrencias y la pérdida usada (PMI-style), las analogías corresponden a **líneas paralelas** en el espacio de embeddings. La intuición es:
- Para Word2Vec/GloVe, $\mathbf{v}_w \cdot \mathbf{u}_c \approx \log P(c \mid w) - \log Z$.
- Si la relación "rey:reina :: hombre:mujer" se manifiesta como una traslación constante en log-PMI, los embeddings la heredan.

**Limitación práctica:** las analogías funcionan en **promedio** pero no son confiables individualmente. Estudios posteriores (Linzen 2016) mostraron que `arg max` sobre el vocabulario excluyendo las palabras de entrada infla artificialmente el accuracy del benchmark de analogías de Mikolov.

### 2.6 Generalización por similitud

Slide 27: si el modelo aprendió `Me gusta comer naranjas de postre` y conoce que `manzana ≈ naranja` (porque aparecen en contextos similares), puede asignar masa razonable a `Me gusta comer manzanas de postre` aunque nunca la haya visto. Esto es el **soft sharing** que un n-grama no puede hacer.

### 2.7 Aprendizaje autosupervisado

La slide 28 marca "autosupervisado" como ventaja. Vale la pena precisar el concepto: en autosupervisión, **las etiquetas se construyen automáticamente desde el texto crudo**. Para LM, el target $w_t$ se obtiene "quitando" la palabra correcta del contexto y prediciéndola — sin anotación humana. Es la razón por la que Word2Vec/BERT/GPT pueden escalar a corpus de TB: no requieren labels.

Este principio es la línea de continuidad desde Word2Vec (2013) hasta los modelos foundation actuales — todos son LMs autosupervisados, sólo cambia la arquitectura.

---

## Parte III — Word2Vec (Mikolov et al., 2013)

Dos papers complementarios:
1. **Efficient Estimation of Word Representations in Vector Space** (arXiv:1301.3781) — introduce CBoW y Skip-gram.
2. **Distributed Representations of Words and Phrases and their Compositionality** (NeurIPS 2013) — introduce negative sampling, subsampling, phrase embeddings.

**Idea central** (slide 32): si solo queremos word embeddings (no un LM completo), podemos simplificar la arquitectura para escalar a corpus mucho más grandes. "Abandonan el LM" significa: en vez de modelar $P(w_t \mid w_{<t})$ con softmax exacto, modelan **predicciones locales más simples** dentro de una ventana.

### 3.1 Continuous Bag-of-Words (CBoW)

**Tarea:** dado el contexto $\{w_{t-c}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+c}\}$ (ventana de tamaño $c$), predecir la palabra central $w_t$.

**Arquitectura:**

1. Cada $w_i$ del contexto se mapea a $\mathbf{v}_{w_i} \in \mathbb{R}^m$ via $C$.
2. Suman (o promedian) los embeddings de contexto: $\mathbf{h} = \sum_{i \in \text{ctx}(t)} \mathbf{v}_{w_i}$.
3. Proyectan vía $H \in \mathbb{R}^{m \times |V|}$ y softmax:
$$
P(w_t = j \mid \text{ctx}) = \frac{\exp(\mathbf{u}_j \cdot \mathbf{h})}{\sum_{k=1}^{|V|} \exp(\mathbf{u}_k \cdot \mathbf{h})}.
$$
4. Loss: cross-entropy.

Slide 33 lo grafica con la convención de "bag" — no importa el orden de las palabras del contexto, solo su suma.

**Por qué "bag":** la suma es invariante a permutaciones. No hay positional encoding (eso vendrá con el Transformer en 2017).

### 3.2 Skip-gram

**Tarea inversa:** dada $w_t$, predecir cada palabra del contexto $w_{t+j}$ con $j \in \{-c, \dots, -1, 1, \dots, c\}$.

**Arquitectura** (slide 34):
$$
P(w_{t+j} = k \mid w_t) = \frac{\exp(\mathbf{u}_k \cdot \mathbf{v}_{w_t})}{\sum_{i=1}^{|V|} \exp(\mathbf{u}_i \cdot \mathbf{v}_{w_t})}.
$$

**Objetivo a maximizar** sobre el corpus de $T$ tokens:
$$
\mathcal{L}_{\text{SG}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-c \leq j \leq c \\ j \neq 0}} \log P(w_{t+j} \mid w_t).
$$

**Comparación CBoW vs Skip-gram** (del paper original):

| | CBoW | Skip-gram |
|---|---|---|
| Tarea | Predecir palabra central desde contexto | Predecir contexto desde palabra central |
| Tiempo de entrenamiento | Rápido | ~5× más lento |
| Calidad en analogías sintácticas | Mejor | Peor |
| Calidad en analogías semánticas | Peor | Mejor |
| Palabras raras | Peor (la suma diluye) | Mejor (cada palabra es su propio target) |

**Regla práctica:** CBoW para corpus pequeños o cuando interesan palabras frecuentes; Skip-gram cuando interesa cobertura semántica y palabras raras. La era post-2013 usó casi exclusivamente Skip-gram con negative sampling (SGNS).

### 3.3 El cuello de botella: softmax sobre $|V|$

El término $\sum_{i=1}^{|V|} \exp(\mathbf{u}_i \cdot \mathbf{v}_{w_t})$ del denominador requiere $O(|V| \cdot m)$ por ejemplo. Para $|V| = 10^6$ y $m = 300$, eso son $3 \times 10^8$ multiplicaciones por **cada** (palabra, contexto) — inviable.

Dos soluciones del paper de 2013:

### 3.4 Negative sampling (SGNS)

Reemplazan el softmax por una **clasificación binaria**: distinguir el par real $(w_t, w_{t+j})$ de pares "falsos" $(w_t, w_{\text{neg}})$ donde $w_{\text{neg}}$ se muestrea de una distribución de ruido $P_n$.

Para cada par positivo $(w, c)$ con $w$ palabra y $c$ palabra de contexto:
$$
\mathcal{L}_{\text{SGNS}}(w, c) = \log \sigma(\mathbf{u}_c \cdot \mathbf{v}_w) + \sum_{i=1}^{K} \mathbb{E}_{w_i \sim P_n} \left[ \log \sigma(-\mathbf{u}_{w_i} \cdot \mathbf{v}_w) \right].
$$

- Primer término: el par real debe tener producto punto alto (sigmoide → 1).
- Segundo término: cada uno de los $K$ negativos debe tener producto punto bajo.
- $K$ típico: 5-20 para datasets pequeños, 2-5 para datasets grandes.

**Distribución de ruido:**
$$
P_n(w) \propto U(w)^{3/4},
$$
donde $U(w)$ es la frecuencia unigrama. El exponente $3/4$ es **empírico** — comprime la distribución (palabras raras suben en probabilidad relativa, comunes bajan). El paper original encontró que $3/4$ funcionaba mejor que $1.0$ o el uniforme en analogías.

**Costo:** $O((K+1) \cdot m)$ por ejemplo — independiente de $|V|$. Esto es lo que permitió entrenar Skip-gram en 6B tokens con $|V| = 700k$ en un día.

**Interpretación teórica** (Levy & Goldberg, NeurIPS 2014): SGNS es equivalente a una factorización implícita de la matriz **PMI shiftada**:
$$
\mathbf{v}_w \cdot \mathbf{u}_c \approx \text{PMI}(w, c) - \log K,
$$
donde $\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w) P(c)}$. Este resultado conecta Word2Vec con tradiciones previas de **distributional semantics** y prepara el terreno para GloVe.

### 3.5 Hierarchical softmax (HS)

Alternativa a negative sampling: organizar $|V|$ palabras en un **árbol binario** (idealmente un árbol de Huffman para que palabras frecuentes tengan caminos cortos). Cada palabra $w$ se identifica con un camino $\pi(w)$ desde la raíz; en cada nodo interno se decide ir izquierda o derecha con una sigmoide:

$$
P(w \mid \text{ctx}) = \prod_{n \in \pi(w)} \sigma\left( [\![n \text{ es izquierdo}]\!] \cdot \mathbf{u}_n \cdot \mathbf{h} \right).
$$

- Cada nodo interno tiene su propio vector $\mathbf{u}_n$.
- Costo por ejemplo: $O(\log_2 |V| \cdot m)$ — para $|V| = 10^6$ son ~20 productos punto.
- HS es **exacto** (suma a 1 sobre $V$), a diferencia de SGNS que es una aproximación.

**Trade-off:** SGNS gana en velocidad pura y en calidad sobre tareas con palabras frecuentes; HS gana en palabras raras (porque cada palabra rara tiene su propio camino del árbol). En la práctica moderna SGNS dominó.

### 3.6 Subsampling de palabras frecuentes

Las palabras como "the", "of", "and" aparecen tantas veces que dominan el gradiente sin aportar señal semántica. Mikolov 2013 (NeurIPS) introduce **subsampling**: cada ocurrencia de $w_i$ se descarta con probabilidad

$$
P_{\text{discard}}(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}},
$$

donde $f(w_i)$ es la frecuencia relativa de $w_i$ y $t \approx 10^{-5}$ es un umbral. Palabras con $f(w_i) < t$ siempre se conservan; las muy frecuentes se conservan con probabilidad cada vez menor.

Efecto reportado: ~2× speedup + mejora en calidad de embeddings de palabras raras.

### 3.7 Phrase embeddings

El segundo paper de Mikolov 2013 propone detectar **frases multipalabra frecuentes** ("New_York", "Air_Canada") y tratarlas como tokens únicos. Se usan dos pasadas con un *pointwise mutual information* descontado:

$$
\text{score}(w_i, w_j) = \frac{\text{count}(w_i w_j) - \delta}{\text{count}(w_i) \cdot \text{count}(w_j)},
$$

y se unen pares por encima de un threshold. Permite analogías tipo `Air_Canada - Canada + France ≈ Air_France`.

### 3.8 Código — Skip-gram con negative sampling

#### 3.8.1 PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, emb_dim)
        self.out_emb = nn.Embedding(vocab_size, emb_dim)
        # Inicialización del paper original
        nn.init.uniform_(self.in_emb.weight, -0.5 / emb_dim, 0.5 / emb_dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, context, negatives):
        # center:    [B]      palabra central
        # context:   [B]      palabra de contexto real (positivo)
        # negatives: [B, K]   K palabras negativas
        v_c = self.in_emb(center)          # [B, D]
        u_p = self.out_emb(context)        # [B, D]
        u_n = self.out_emb(negatives)      # [B, K, D]

        # Positivo: log σ(u_p · v_c)
        pos_score = (u_p * v_c).sum(dim=-1)             # [B]
        pos_loss = F.logsigmoid(pos_score)

        # Negativo: Σ_k log σ(-u_n · v_c)
        neg_score = torch.bmm(u_n, v_c.unsqueeze(-1)).squeeze(-1)  # [B, K]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)

        return -(pos_loss + neg_loss).mean()


def sample_negatives(unigram_probs: torch.Tensor, batch_size: int, K: int):
    # unigram_probs ya está elevado a 3/4 y normalizado
    return torch.multinomial(unigram_probs, batch_size * K, replacement=True).view(batch_size, K)
```

Detalles a tener en cuenta:
- Dos matrices distintas (`in_emb`, `out_emb`). Al final del entrenamiento se usa `in_emb` como los "word vectors"; `out_emb` se descarta o se promedia con `in_emb`.
- `torch.bmm` para batched dot product entre $K$ negativos y el embedding central.
- La inicialización del paper original (uniform con escala $1/D$) es importante — la inicialización gaussiana estándar de PyTorch da peores resultados en este modelo.

#### 3.8.2 TensorFlow / Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SkipGramNS(keras.Model):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_emb = layers.Embedding(
            vocab_size, emb_dim,
            embeddings_initializer=keras.initializers.RandomUniform(-0.5/emb_dim, 0.5/emb_dim),
        )
        self.out_emb = layers.Embedding(
            vocab_size, emb_dim,
            embeddings_initializer="zeros",
        )

    def call(self, inputs):
        center, context, negatives = inputs  # [B], [B], [B, K]
        v_c = self.in_emb(center)             # [B, D]
        u_p = self.out_emb(context)           # [B, D]
        u_n = self.out_emb(negatives)         # [B, K, D]

        pos_score = tf.reduce_sum(u_p * v_c, axis=-1)                         # [B]
        neg_score = tf.einsum("bkd,bd->bk", u_n, v_c)                          # [B, K]

        pos_loss = tf.math.log_sigmoid(pos_score)
        neg_loss = tf.reduce_sum(tf.math.log_sigmoid(-neg_score), axis=-1)

        return -tf.reduce_mean(pos_loss + neg_loss)


# Sampler nativo de TF para negative sampling — más eficiente que multinomial
sampled, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes=context_batch[:, None],   # [B, 1]
    num_true=1,
    num_sampled=K * batch_size,
    unique=False,
    range_max=vocab_size,
)
# Reshape a [B, K]
negatives = tf.reshape(sampled, (batch_size, K))
```

TensorFlow ofrece `tf.random.log_uniform_candidate_sampler` y `tf.nn.sampled_softmax_loss` / `tf.nn.nce_loss` como abstracciones de alto nivel. Internamente NCE = Negative Contrastive Estimation (Gutmann & Hyvärinen 2010) es el predecesor teórico de SGNS — el paper de Mikolov 2013 lo cita.

#### 3.8.3 JAX (con Flax + Optax)

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

class SkipGramNS(nn.Module):
    vocab_size: int
    emb_dim: int

    def setup(self):
        self.in_emb = nn.Embed(self.vocab_size, self.emb_dim,
                               embedding_init=nn.initializers.uniform(scale=1.0 / self.emb_dim))
        self.out_emb = nn.Embed(self.vocab_size, self.emb_dim,
                                embedding_init=nn.initializers.zeros)

    def __call__(self, center, context, negatives):
        v_c = self.in_emb(center)              # [B, D]
        u_p = self.out_emb(context)            # [B, D]
        u_n = self.out_emb(negatives)          # [B, K, D]

        pos_score = jnp.sum(u_p * v_c, axis=-1)
        neg_score = jnp.einsum("bkd,bd->bk", u_n, v_c)

        pos_loss = jax.nn.log_sigmoid(pos_score)
        neg_loss = jnp.sum(jax.nn.log_sigmoid(-neg_score), axis=-1)

        return -jnp.mean(pos_loss + neg_loss)


@jax.jit
def train_step(params, opt_state, batch, key):
    center, context = batch
    key, subkey = jax.random.split(key)
    # Sampling de negativos vectorizado
    negatives = jax.random.categorical(
        subkey, jnp.log(unigram_probs_pow_3_4), shape=(center.shape[0], K)
    )

    def loss_fn(p):
        return model.apply(p, center, context, negatives)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, key
```

Idiosincrasia JAX:
- Toda la función debe ser pura — el sampler se inyecta como argumento via `jax.random.PRNGKey`.
- `jax.jit` compila el `train_step` y se ejecuta a velocidad de XLA — para Word2Vec en GPU/TPU es comparable o más rápido que las implementaciones C-optimized del paper original.
- `optax` reemplaza el optimizer state de PyTorch/TF.

---

## Parte IV — GloVe (Pennington, Socher, Manning, 2014)

**GloVe = Global Vectors for Word Representation** (ACL D14-1162). Idea pivotal: **combinar lo mejor de los dos mundos** previos a 2014:

- Métodos basados en **conteos globales** (LSA, HAL, COALS): aprovechan información del corpus completo pero el desempeño en analogías es modesto.
- Métodos basados en **ventanas locales** (Word2Vec): excelente en analogías pero ignoran la información global de co-ocurrencia.

GloVe aprende embeddings que **aproximan la log-probabilidad de co-ocurrencia global** directamente.

### 4.1 Matriz de co-ocurrencia

Definir $X \in \mathbb{R}^{|V| \times |V|}$ donde $X_{ij}$ = número de veces que la palabra $j$ aparece en el contexto de $i$ (con ventana fija, e.g. $\pm 10$).

Algunas definiciones derivadas:
- $X_i = \sum_k X_{ik}$ = total de ocurrencias del contexto de $i$.
- $P_{ij} = P(j \mid i) = X_{ij} / X_i$ = probabilidad condicional de ver $j$ dado $i$.

### 4.2 Derivación de la función de costo

El paper desarrolla una motivación formal — vale la pena seguirla porque revela la conexión con PMI.

**Paso 1:** observan que las razones $P_{ik} / P_{jk}$ codifican mejor las relaciones semánticas que probabilidades individuales:

| $k$ | $P(k \mid \text{ice})$ | $P(k \mid \text{steam})$ | Razón |
|---|---|---|---|
| solid | $1.9 \times 10^{-4}$ | $2.2 \times 10^{-5}$ | 8.9 |
| gas | $6.6 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | 0.085 |
| water | $3.0 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | 1.36 |
| fashion | $1.7 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | 0.96 |

La razón distingue "solid" (ice-only) y "gas" (steam-only) muy claramente; palabras genéricas como "water" o irrelevantes como "fashion" caen cerca de 1.

**Paso 2:** postulan que la función a aprender es $F(\mathbf{w}_i, \mathbf{w}_j, \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$.

**Paso 3:** imponen restricciones de simetría e invarianza:
1. La operación natural en el espacio vectorial es la diferencia: $F((\mathbf{w}_i - \mathbf{w}_j), \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$.
2. El argumento debe ser escalar para que el lado derecho sea escalar: $F((\mathbf{w}_i - \mathbf{w}_j)^T \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$.
3. Para que la simetría de roles (palabra-contexto intercambiables) se preserve, $F$ debe ser un homomorfismo entre $(\mathbb{R}, +)$ y $(\mathbb{R}_{>0}, \times)$ — i.e., $F = \exp$.

Esto da:
$$
\mathbf{w}_i^T \tilde{\mathbf{w}}_k = \log P_{ik} = \log X_{ik} - \log X_i.
$$

**Paso 4:** absorber $\log X_i$ en un bias $b_i$ y agregar bias $\tilde{b}_k$ para mantener simetría:
$$
\mathbf{w}_i^T \tilde{\mathbf{w}}_k + b_i + \tilde{b}_k = \log X_{ik}.
$$

**Paso 5:** definir la pérdida como **least squares ponderada**:
$$
\mathcal{J} = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( \mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2.
$$

Esta es la fórmula que aparece en la slide 36.

### 4.3 La función de peso $f$

$f(X_{ij})$ es crucial. Debe satisfacer:

1. $f(0) = 0$ (los ceros no contribuyen — y son la mayoría de las entradas de $X$).
2. $f$ no-decreciente — co-ocurrencias raras no deben dominar.
3. $f(x)$ acotada para $x$ grande — co-ocurrencias muy frecuentes (e.g., con "the") no deben dominar.

Pennington et al. proponen:
$$
f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{si } x < x_{\max}, \\ 1 & \text{si } x \geq x_{\max}. \end{cases}
$$

con $x_{\max} = 100$ y $\alpha = 3/4$ — sí, el mismo $3/4$ que en negative sampling de Word2Vec. Aparece como una constante de escala "buena" en distributional semantics.

### 4.4 Comparación Word2Vec vs GloVe

| Aspecto | Word2Vec (SGNS) | GloVe |
|---|---|---|
| Naturaleza | Predicción local | Factorización de matriz global |
| Datos consumidos | Ventanas individuales (streaming) | Matriz $X$ pre-computada |
| Loss | Log-binary classification | Squared error ponderado |
| Memoria | $O(|V| m)$ | $O(|X|_{nnz})$ — matriz puede ser TB |
| Embedding final | $\mathbf{v}_w$ (input embedding) | $\mathbf{w} + \tilde{\mathbf{w}}$ (suma de ambos) |
| Hiperparámetros clave | ventana, K negativos, subsampling | $x_{\max}$, $\alpha$, ventana |
| Calidad | Similar en analogías y similaridad | Similar |

La práctica industrial post-2014 dividió el campo: GloVe se popularizó por Stanford y los embeddings preentrenados publicados (`glove.6B.300d`, `glove.840B.300d`). Word2Vec se mantuvo en `gensim`.

### 4.5 Código — GloVe loss

#### 4.5.1 PyTorch

```python
import torch
import torch.nn as nn

class GloVe(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, x_max: float = 100.0, alpha: float = 0.75):
        super().__init__()
        self.w = nn.Embedding(vocab_size, emb_dim)
        self.w_tilde = nn.Embedding(vocab_size, emb_dim)
        self.b = nn.Embedding(vocab_size, 1)
        self.b_tilde = nn.Embedding(vocab_size, 1)
        nn.init.uniform_(self.w.weight, -0.5/emb_dim, 0.5/emb_dim)
        nn.init.uniform_(self.w_tilde.weight, -0.5/emb_dim, 0.5/emb_dim)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b_tilde.weight)
        self.x_max = x_max
        self.alpha = alpha

    def f_weight(self, x):
        return torch.where(x < self.x_max, (x / self.x_max) ** self.alpha, torch.ones_like(x))

    def forward(self, i_idx, j_idx, x_ij):
        # i_idx, j_idx: [B]
        # x_ij:        [B]  conteo de co-ocurrencia
        w_i = self.w(i_idx)               # [B, D]
        w_j = self.w_tilde(j_idx)         # [B, D]
        b_i = self.b(i_idx).squeeze(-1)   # [B]
        b_j = self.b_tilde(j_idx).squeeze(-1)

        dot = (w_i * w_j).sum(dim=-1)
        diff = dot + b_i + b_j - torch.log(x_ij)
        weight = self.f_weight(x_ij)
        return (weight * diff.pow(2)).mean()

    def get_embeddings(self):
        return self.w.weight + self.w_tilde.weight
```

#### 4.5.2 TensorFlow

```python
import tensorflow as tf
from tensorflow.keras import layers

class GloVe(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, x_max=100.0, alpha=0.75):
        super().__init__()
        init = tf.keras.initializers.RandomUniform(-0.5/emb_dim, 0.5/emb_dim)
        self.w = layers.Embedding(vocab_size, emb_dim, embeddings_initializer=init)
        self.w_tilde = layers.Embedding(vocab_size, emb_dim, embeddings_initializer=init)
        self.b = layers.Embedding(vocab_size, 1, embeddings_initializer="zeros")
        self.b_tilde = layers.Embedding(vocab_size, 1, embeddings_initializer="zeros")
        self.x_max, self.alpha = x_max, alpha

    def f_weight(self, x):
        return tf.where(x < self.x_max, (x / self.x_max) ** self.alpha, tf.ones_like(x))

    def call(self, inputs):
        i_idx, j_idx, x_ij = inputs
        w_i = self.w(i_idx)
        w_j = self.w_tilde(j_idx)
        b_i = tf.squeeze(self.b(i_idx), axis=-1)
        b_j = tf.squeeze(self.b_tilde(j_idx), axis=-1)
        dot = tf.reduce_sum(w_i * w_j, axis=-1)
        diff = dot + b_i + b_j - tf.math.log(x_ij)
        weight = self.f_weight(x_ij)
        return tf.reduce_mean(weight * tf.square(diff))
```

#### 4.5.3 JAX

```python
import jax.numpy as jnp
import flax.linen as nn

class GloVe(nn.Module):
    vocab_size: int
    emb_dim: int
    x_max: float = 100.0
    alpha: float = 0.75

    def setup(self):
        init = nn.initializers.uniform(scale=1.0 / self.emb_dim)
        self.w = nn.Embed(self.vocab_size, self.emb_dim, embedding_init=init)
        self.w_tilde = nn.Embed(self.vocab_size, self.emb_dim, embedding_init=init)
        self.b = nn.Embed(self.vocab_size, 1, embedding_init=nn.initializers.zeros)
        self.b_tilde = nn.Embed(self.vocab_size, 1, embedding_init=nn.initializers.zeros)

    def f_weight(self, x):
        return jnp.where(x < self.x_max, (x / self.x_max) ** self.alpha, 1.0)

    def __call__(self, i_idx, j_idx, x_ij):
        w_i = self.w(i_idx)
        w_j = self.w_tilde(j_idx)
        b_i = jnp.squeeze(self.b(i_idx), axis=-1)
        b_j = jnp.squeeze(self.b_tilde(j_idx), axis=-1)
        dot = jnp.sum(w_i * w_j, axis=-1)
        diff = dot + b_i + b_j - jnp.log(x_ij)
        weight = self.f_weight(x_ij)
        return jnp.mean(weight * diff ** 2)
```

**Truco práctico:** la matriz $X$ se construye en un único pass por el corpus con un `Counter` o `scipy.sparse.coo_matrix`. Para corpus grandes (Wikipedia), se usa `glove-python-binary` o el código original C de Stanford que hace streaming-by-window.

---

## Parte V — Skip-Thought Vectors (Kiros et al., 2015)

**Skip-Thought Vectors** (NeurIPS 2015) extiende la idea de Word2Vec del nivel de palabra al **nivel de oración**. La motivación es directa por analogía:

| | Word2Vec Skip-gram | Skip-Thought |
|---|---|---|
| Unidad | palabra | oración |
| Target | palabras de contexto | oraciones adyacentes |
| Encoder | embedding lookup | RNN |
| Producto | word embeddings | **sentence embeddings** |

**Tarea:** dada una oración $s_i$, predecir las oraciones $s_{i-1}$ y $s_{i+1}$.

### 5.1 Arquitectura

Tres componentes:

1. **Encoder GRU** (notación del paper: $h_t$) procesa $s_i$ palabra por palabra. El estado oculto final $\mathbf{h}_i = h_T$ es el **sentence embedding** de $s_i$.
2. **Decoder GRU previo** ($\mathbf{h}_{i-1}^{\text{dec}}$): genera $s_{i-1}$ palabra por palabra, condicionado en $\mathbf{h}_i$.
3. **Decoder GRU siguiente** ($\mathbf{h}_{i+1}^{\text{dec}}$): genera $s_{i+1}$, también condicionado en $\mathbf{h}_i$.

Los dos decoders no comparten pesos pero comparten la matriz de salida (proyección a vocabulario).

**Objetivo:**
$$
\mathcal{L} = \sum_{i} \log P(s_{i+1} \mid s_i) + \log P(s_{i-1} \mid s_i),
$$
donde cada $\log P(s \mid s_i) = \sum_t \log P(w_t \mid w_{<t}, \mathbf{h}_i)$ usa cross-entropy estándar (softmax sobre $|V|$, con $|V| \sim 20k$ palabras del BookCorpus).

### 5.2 Conditional GRU — cómo se condiciona el decoder

El paper introduce un **conditional GRU** donde el embedding de la oración $\mathbf{h}_i$ se inyecta en los **reset gate, update gate, y candidate state** del decoder:

$$
\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1}^{\text{dec}} + \mathbf{C}_r \mathbf{h}_i),
$$
$$
\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1}^{\text{dec}} + \mathbf{C}_z \mathbf{h}_i),
$$
$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \mathbf{x}_t + \mathbf{U}(\mathbf{r}_t \odot \mathbf{h}_{t-1}^{\text{dec}}) + \mathbf{C} \mathbf{h}_i),
$$
$$
\mathbf{h}_t^{\text{dec}} = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1}^{\text{dec}} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t.
$$

Las matrices $\mathbf{C}_r, \mathbf{C}_z, \mathbf{C}$ son nuevas — proyectan el sentence embedding a cada gate.

### 5.3 Vocabulary expansion — el truco para vocabulario abierto

El BookCorpus tiene un vocabulario fijo de ~20k. Pero downstream queremos embeddings para palabras fuera del corpus. **Solución:** entrenar una **regresión lineal** $\mathbf{W}_\text{exp}: \mathbb{R}^{300} \to \mathbb{R}^{620}$ que mapea Word2Vec pretraído (cobertura amplia) a los embeddings del encoder de Skip-Thought. Para palabras nuevas:
1. Tomar su Word2Vec.
2. Proyectar con $\mathbf{W}_\text{exp}$.
3. Usar como input del encoder de Skip-Thought.

Esto extendió el vocabulario efectivo de ~20k a ~1M palabras sin re-entrenar.

### 5.4 Aplicaciones downstream

Slide 39 lista las tareas evaluadas en el paper:

- **Semantic relatedness** (SICK dataset): predecir el score humano de similitud entre dos oraciones. Skip-Thought obtuvo $r = 0.858$ — competitivo con métodos supervisados de la época.
- **Paraphrase detection** (Microsoft Research Paraphrase Corpus).
- **Classification**: sentiment (MR, CR), subjectivity (SUBJ), opinion polarity (MPQA), TREC question types.

**Por qué importa:** Skip-Thought es el **primer modelo no-supervisado** que aprende sentence embeddings transferibles. Es el antecesor directo de InferSent (Conneau 2017, supervisado en NLI), Universal Sentence Encoder (Google 2018), y Sentence-BERT (Reimers 2019).

### 5.5 Limitaciones

1. **Entrenamiento muy costoso**: ~2 semanas en GPU para el paper original.
2. **Vocabulary fijo + truco de expansión** es awkward — modernamente se usa BPE o WordPiece para vocabulario abierto.
3. **Sentence embedding es un vector único** (no contextual) — BERT lo supera porque produce embeddings contextuales por token.
4. **Sin atención**: limitado a oraciones cortas. Frases largas pierden información en la compresión a $\mathbf{h}_T$.

Skip-Thought fue rápidamente superado (2-3 años) pero la idea de **autosupervisar a nivel de oración** persistió: SimCSE, Sentence-T5, gtr-t5 son herederos conceptuales.

### 5.6 Código — Skip-Thought encoder

#### 5.6.1 PyTorch

```python
import torch
import torch.nn as nn

class SkipThoughtEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 620, hidden_dim: int = 2400):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, token_ids, lengths):
        # token_ids: [B, T] con padding
        emb = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_T = self.gru(packed)
        return h_T.squeeze(0)  # [B, hidden_dim]


class ConditionalGRUCell(nn.Module):
    """GRU cell con condicionamiento de la oración fuente."""
    def __init__(self, input_dim, hidden_dim, cond_dim):
        super().__init__()
        self.W_r = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.C_r = nn.Linear(cond_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.C_z = nn.Linear(cond_dim, hidden_dim)
        self.W = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.C = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x_t, h_prev, h_cond):
        r = torch.sigmoid(self.W_r(x_t) + self.U_r(h_prev) + self.C_r(h_cond))
        z = torch.sigmoid(self.W_z(x_t) + self.U_z(h_prev) + self.C_z(h_cond))
        h_tilde = torch.tanh(self.W(x_t) + self.U(r * h_prev) + self.C(h_cond))
        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new


class SkipThought(nn.Module):
    def __init__(self, vocab_size, emb_dim=620, hidden_dim=2400):
        super().__init__()
        self.encoder = SkipThoughtEncoder(vocab_size, emb_dim, hidden_dim)
        self.decoder_prev = ConditionalGRUCell(emb_dim, hidden_dim, hidden_dim)
        self.decoder_next = ConditionalGRUCell(emb_dim, hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.embedding = self.encoder.embedding  # tied embeddings

    def decode(self, decoder, target_ids, h_cond):
        B, T = target_ids.shape
        h = torch.zeros(B, h_cond.shape[-1], device=h_cond.device)
        logits = []
        emb = self.embedding(target_ids)
        for t in range(T):
            h = decoder(emb[:, t], h, h_cond)
            logits.append(self.output_proj(h))
        return torch.stack(logits, dim=1)  # [B, T, V]
```

#### 5.6.2 JAX (encoder solamente, por brevedad)

```python
import flax.linen as nn
import jax.numpy as jnp

class SkipThoughtEncoder(nn.Module):
    vocab_size: int
    emb_dim: int = 620
    hidden_dim: int = 2400

    @nn.compact
    def __call__(self, token_ids):
        emb = nn.Embed(self.vocab_size, self.emb_dim)(token_ids)
        gru_cell = nn.GRUCell(features=self.hidden_dim)
        h_init = gru_cell.initialize_carry(jax.random.PRNGKey(0), (token_ids.shape[0],))
        # Scan sobre la dimensión temporal
        h_final, _ = nn.scan(
            lambda mdl, c, x: mdl(c, x),
            in_axes=1, out_axes=1, variable_broadcast="params", split_rngs={"params": False},
        )(gru_cell, h_init, emb)
        return h_final
```

JAX usa `nn.scan` para reemplazar el loop temporal — esto permite compilación XLA del encoder completo. Para Skip-Thought completo con dos decoders condicionales, la implementación más natural en JAX es definir el `ConditionalGRUCell` como un `nn.Module` que se llama dentro de un `nn.scan` análogo.

---

## Parte VI — Conexiones, evaluación y legado

### 6.1 Evaluación de word embeddings

El PDF no entra en evaluación pero es parte indispensable del entendimiento. Dos familias:

**Intrínsecas:**
- **Word similarity**: dataset de pares de palabras con scores humanos (WordSim-353, SimLex-999). Métrica: correlación de Spearman entre $\cos(\mathbf{v}_w, \mathbf{v}_{w'})$ y score humano.
- **Word analogies**: dataset de Mikolov (`a:b :: c:?`). Métrica: accuracy del top-1 sobre $V \setminus \{a, b, c\}$.
- **Categorization** (BLESS, AP): clustering de palabras por categorías semánticas.

**Extrínsecas:**
- POS tagging, NER, sentiment, parsing — usar los embeddings como features de entrada a un modelo downstream y medir su efecto.

Encuentra empírico (Schnabel 2015): los rankings de embeddings cambian según la métrica. No hay "mejor embedding universal".

### 6.2 Sucesores históricos inmediatos

| Año | Modelo | Innovación sobre Word2Vec/GloVe |
|---|---|---|
| 2016 | **FastText** (Bojanowski, FAIR) | Embeddings a nivel de **subword n-grama** → maneja OOV y morfología |
| 2017 | **InferSent** (Conneau, FAIR) | Sentence embeddings supervisados con SNLI |
| 2018 | **ELMo** (Peters, AI2) | Embeddings **contextuales** vía biLM con LSTM |
| 2018 | **ULMFiT** (Howard & Ruder) | Transfer learning con LM como pre-training task |
| 2018 | **BERT** (Devlin, Google) | Transformer bidireccional + masked LM |

### 6.3 Limitaciones fundamentales de Word2Vec/GloVe/Skip-Thought

1. **Embeddings no contextuales**: la palabra "banco" tiene un único vector independiente de si aparece en "banco de peces" o "banco financiero". ELMo/BERT lo resuelven.
2. **Polisemia ignorada**: estudios mostraron que un single embedding promedia los sentidos.
3. **Sesgos sociales codificados**: Bolukbasi 2016 (*Man is to Computer Programmer as Woman is to Homemaker?*) mostró que las analogías reproducen estereotipos del corpus de entrenamiento.
4. **Anisotropía**: Mu & Viswanath 2018 mostraron que los embeddings se concentran en un cono estrecho del espacio, lo que degrada la cosine similarity. Soluciones post-hoc: PCA-removal de las top componentes.
5. **Out-of-vocabulary**: cualquier palabra no vista en entrenamiento no tiene embedding. FastText resuelve esto con subwords.

### 6.4 ¿Por qué esta clase importa en 2026?

A pesar de que GPT-4 y Llama-3 superaron a Word2Vec por órdenes de magnitud, los conceptos de la clase 18 siguen siendo fundamentales:

- **Toda capa de embedding** en un Transformer moderno (incluyendo GPT) es **conceptualmente el mismo `nn.Embedding`** que Bengio 2003.
- **El objetivo autosupervisado** de "predecir tokens dado contexto" es el mismo que en Word2Vec, escalado a billones de tokens y arquitectura Transformer.
- **Negative sampling y noise contrastive estimation** son la base de InfoNCE (CLIP, SimCLR, Sentence-BERT) — todo el aprendizaje contrastivo moderno.
- **Cosine similarity y dot product** sobre embeddings son la operación primaria de retrieval (RAG, vector DBs como Pinecone/Weaviate).
- **GloVe-style factorización** reaparece como matrix-factorization regularizers en sistemas de recomendación.

Cuando un ingeniero levanta una RAG moderna con Sentence-BERT, está usando código y conceptos cuya genealogía directa son Word2Vec → Skip-Thought → InferSent → SBERT. Esta clase es la base.

### 6.5 Conexión con el lab de la clase

La slide 23 (Transformer multimodal CXR-Mate-RRG24) y la slide 4 (que menciona "Trabajo práctico") sugieren que el lab integra estos conceptos a un caso clínico. Pendiente analizar el lab en sesión separada con `feedback_lab_walkthrough_strategy`.

### 6.6 Conexión con la clase 19

Clase 19 (ELMo, GPT, BERT) parte exactamente donde termina ésta: con la observación de que embeddings no contextuales son insuficientes. ELMo introduce embeddings contextuales con biLSTM, GPT/BERT con Transformer. La transición es la **historia central de NLP entre 2013 y 2018**.

---

## Apéndice A — Hiperparámetros típicos

| Modelo | Vocab | Emb dim | Ventana | Negativos / x_max | Subsampling t | Epochs | Corpus |
|---|---|---|---|---|---|---|---|
| Word2Vec Skip-gram | 1M | 300 | 10 | 5 negativos | $10^{-5}$ | 3-5 | Google News 100B |
| Word2Vec CBoW | 1M | 300 | 5 | 5-10 negativos | $10^{-5}$ | 3-5 | Google News 100B |
| GloVe 6B | 400k | 300 | 10 | $x_{\max}=100, \alpha=3/4$ | — | 50 | Wikipedia 2014 + Gigaword 5 |
| GloVe 840B | 2.2M | 300 | 10 | $x_{\max}=100, \alpha=3/4$ | — | 100 | Common Crawl 840B |
| Skip-Thought | 20k (encoder) / 1M (extended) | 620 emb / 2400 hidden | — | — | — | 1 epoch | BookCorpus 70M oraciones |

## Apéndice B — Embeddings preentrenados disponibles

```bash
# Word2Vec original (Google News)
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

# GloVe
wget https://nlp.stanford.edu/data/glove.6B.zip       # 400k vocab, varios dims
wget https://nlp.stanford.edu/data/glove.840B.300d.zip  # 2.2M vocab, 300d

# FastText
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
```

En Python:
```python
import gensim.downloader as api
model = api.load("word2vec-google-news-300")  # 1.6 GB
print(model.most_similar("computer"))
print(model.similarity("king", "queen"))
print(model.most_similar(positive=["king", "woman"], negative=["man"]))
# → "queen" alto en el ranking
```

## Apéndice C — Papers a descargar para `clase_18/papers/`

| Orden | Paper | URL | Prioridad |
|---|---|---|---|
| 1 | Mikolov 2013 — Efficient Estimation | https://arxiv.org/pdf/1301.3781 | crítico |
| 2 | Mikolov 2013 — Distributed Representations | https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf | crítico |
| 3 | Pennington 2014 — GloVe | https://aclanthology.org/D14-1162.pdf | crítico |
| 4 | Kiros 2015 — Skip-Thought Vectors | https://papers.nips.cc/paper/2015/file/f442d33fa06832082290ad8544a8da27-Paper.pdf | crítico |
| 5 | Bengio 2003 — NPLM | https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf | crítico (fundacional) |
| 6 | Levy & Goldberg 2014 — SGNS implicit MF | https://papers.nips.cc/paper/2014/file/feab05aa91085b7a8012516bc3533958-Paper.pdf | alta (teoría) |
| 7 | Allen & Hospedales 2019 — Analogies Explained | https://arxiv.org/pdf/1901.09813 | media |
| 8 | Mikolov 2010 — RNN-LM | https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf | media (RNN-LM citado en slide 22) |

---

## Referencias adicionales

- Jurafsky & Martin, *Speech and Language Processing* (3rd ed., draft), capítulos 3 (N-grams), 6 (Vector Semantics), 7 (Neural LMs): https://web.stanford.edu/~jurafsky/slp3/
- Goldberg, Y. (2017). *Neural Network Methods for Natural Language Processing*. Morgan & Claypool. Capítulos 10-11.
- Manning, C., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval*. Capítulo 6 (vector space model — antecesor histórico de embeddings).
- Lecture notes Stanford CS224N (Manning): https://web.stanford.edu/class/cs224n/
