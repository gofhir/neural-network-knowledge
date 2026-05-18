# Bengio, Ducharme, Vincent & Jauvin 2003 — A Neural Probabilistic Language Model

| Campo | Valor |
|---|---|
| **Autores** | Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin |
| **Afiliación** | Université de Montréal, DIRO, Centre de Recherche Mathématiques |
| **Venue** | JMLR (Journal of Machine Learning Research), vol. 3, pp. 1137-1155 |
| **Fecha** | Sometido abril 2002, publicado febrero 2003 |
| **Pdf** | `Bengio-NPLM-2003.pdf` (19 páginas) |
| **Citaciones** | >15.000 |
| **URL** | https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.html |

> *"We propose to fight the curse of dimensionality by learning a distributed representation for words which allows each training sentence to inform the model about an exponential number of semantically neighboring sentences."*

Este es **el paper fundacional del paradigma de embeddings aprendidos**. Aunque sus diagramas aparecen (sin atribución) en la slide 21 de la clase 18, no se cita explícitamente. Sin Bengio 2003 no hay Word2Vec, no hay BERT, no hay GPT. Es la idea inicial de que **las palabras pueden representarse como vectores densos aprendidos junto con un modelo probabilístico de lenguaje**.

---

## 1. Contexto histórico (1998-2002)

### 1.1 Estado del arte pre-NPLM

En 2002, los modelos de lenguaje dominantes eran:

- **Trigramas con suavizado Kneser-Ney** (Chen & Goodman 1998). SOTA en perplejidad y aplicaciones (speech, MT).
- **Clase-based models** (Brown 1992): agrupar palabras en clases, modelar transiciones entre clases.
- **Maximum entropy LMs** (Rosenfeld 1996): combinar features lingüísticas con modelos log-lineares.
- **LSA-LM** (Bellegarda 1997): combinar trigramas con scoring vía LSA para contexto largo.

**Limitaciones comunes**:
1. **Curse of dimensionality**: para vocabulario $|V| = 17.000$, hay $17.000^{10} = 10^{42}$ posibles 10-gramas. Solo una fracción ínfima aparece en cualquier corpus.
2. **Sin similitud entre palabras**: si `the cat is walking in the bedroom` se vio en training, `a dog was running in a room` recibe probabilidad cero o casi cero, **aunque sea semánticamente equivalente**.
3. **N-gramas no escalan a contexto largo**: la frecuencia de un n-grama decae exponencialmente con $n$.

### 1.2 Antecedentes en representación distribuida

- **Hinton 1986** — *Learning distributed representations of concepts*. Propone la idea filosófica de que conceptos se representan distribuidos sobre múltiples unidades neuronales.
- **Elman 1990** — Simple Recurrent Networks aprenden representaciones implícitas de palabras vía contexto.
- **Miikkulainen & Dyer 1991** — primer uso explícito de redes neuronales para LM con embeddings, pero pequeña escala.
- **Schmidhuber 1996** — predicción de caracteres con NN para compresión.
- **Xu & Rudnicky 2000** — NN-LM con una sola palabra de input. Sin hidden layer, captura solo unigramas/bigramas.
- **Paccanaro & Hinton 2000** — Linear Relational Embeddings para datos simbólicos.

Bengio toma estas ideas — distributed representations + neural networks for LMs — y las **escala a contexto multi-palabra con hidden layers, en datasets reales (millones de palabras)**.

---

## 2. Contribución central — las 3 ideas en una frase

El paper resume su idea en 3 pasos (sección 1.1 del paper):

1. **Asociar a cada palabra del vocabulario un vector distribuido de features** $C(i) \in \mathbb{R}^m$ (un real-valued vector con $m \approx 30, 60, 100$).
2. **Expresar la función de probabilidad conjunta** de secuencias en términos de los vectores de feature de las palabras.
3. **Aprender simultáneamente los vectores de feature y los parámetros de la función de probabilidad**.

**La idea genial**: $m \ll |V|$. Para $|V| = 17.000$ y $m = 100$, los embeddings ocupan $17.000 \times 100 = 1.7M$ parámetros — comparable a un trigram model pero generalizando **infinitamente mejor**.

**Mecanismo de generalización** (sección 1.1):

> *"If we knew that `dog` and `cat` played similar roles (semantically and syntactically), and similarly for (`the`, `a`), (`bedroom`, `room`), (`is`, `was`), (`running`, `walking`), we could naturally generalize (i.e. transfer probability mass) from `The cat is walking in the bedroom` to `A dog was running in a room`."*

> *"In the proposed model, it will so generalize because 'similar' words are expected to have a similar feature vector, and because the probability function is a smooth function of these feature values, a small change in the features will induce a small change in the probability."*

Esta es **la justificación pedagógica que sobrevive 20 años después**: word embeddings funcionan porque palabras semánticamente similares ocupan regiones similares del espacio, y la red neuronal es una función suave de los embeddings.

---

## 3. Arquitectura formal

### 3.1 Setup

- Entrenamiento: secuencia $w_1, w_2, \dots, w_T$ con $w_t \in V$.
- Objetivo: aprender $\hat{f}(w_t, w_{t-1}, \dots, w_{t-n+1}) = \hat{P}(w_t \mid w_{t-n+1:t-1})$.
- Restricción: $\sum_{i=1}^{|V|} f(i, w_{t-1}, \dots, w_{t-n+1}) = 1$ (normalización).
- Métrica: **perplejidad** = $\exp(-\frac{1}{T} \sum \log \hat{P}(w_t \mid w_{<t}))$.

### 3.2 Descomposición en dos partes

$$
f(i, w_{t-1}, \dots, w_{t-n+1}) = g(i, C(w_{t-1}), \dots, C(w_{t-n+1}))
$$

donde:
- $C: V \to \mathbb{R}^m$ es una matriz de **lookup** ($|V| \times m$). $C(w)$ devuelve el embedding de $w$.
- $g$ es una red neuronal que, dados los embeddings del contexto, produce la distribución sobre el próximo token.

### 3.3 Red neuronal $g$ — feedforward de una capa hidden

**Ecuación 1 del paper:**

$$
y = b + Wx + U \tanh(d + Hx) \quad (1)
$$

donde:
- $x = [C(w_{t-1}); C(w_{t-2}); \dots; C(w_{t-n+1})]$ — concatenación de embeddings, $\in \mathbb{R}^{(n-1)m}$.
- $H \in \mathbb{R}^{h \times (n-1)m}$ — pesos de input a hidden.
- $d \in \mathbb{R}^h$ — bias de hidden.
- $\tanh$ aplicado elementwise sobre la hidden layer ($h$ unidades).
- $U \in \mathbb{R}^{|V| \times h}$ — pesos de hidden a output.
- $W \in \mathbb{R}^{|V| \times (n-1)m}$ — **conexiones directas** de embeddings al output (skip connections). Pueden ser $W = 0$ para desactivarlas.
- $b \in \mathbb{R}^{|V|}$ — biases del output.
- $y \in \mathbb{R}^{|V|}$ — log-probabilidades no normalizadas.

**Output normalizado**:
$$
\hat{P}(w_t = i \mid w_{<t}) = \frac{e^{y_i}}{\sum_{j=1}^{|V|} e^{y_j}}.
$$

### 3.4 Parámetros

Conjunto completo: $\theta = (b, d, W, U, H, C)$.

Número total de parámetros: $|V|(1 + nm + h) + h(1 + (n-1)m)$.

**Análisis** (ejemplo de Brown corpus en el paper): $|V| = 17.964$, $h = 60$, $n = 6$, $m = 100$:
- $C$: $17.964 \times 100 \approx 1.8M$ params (embeddings).
- $W$: $17.964 \times 500 \approx 9M$ params (direct skip).
- $U$: $17.964 \times 60 \approx 1.1M$ params (hidden-to-output).
- $H$: $60 \times 500 = 30k$ params.
- Total: **~12M parámetros**.

Esto es **mucho** para 2002. Bengio observa: *"Training such large models (with millions of parameters) within a reasonable time is itself a significant challenge."*

### 3.5 Diagrama

Reproduce literalmente el diagrama del paper (figura 1):

```
                  i-th output = P(w_t = i | context)
                       ↑
                  ┌─────────┐
                  │ softmax │  ← most computation here
                  └─────────┘
                       ↑
                  ┌─────────┐
                  │  tanh   │
                  └─────────┘
                       ↑
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────────┐    ┌────────┐    ┌────────┐
   │C(w_t-n+1)│  │C(w_t-2)│ ...│C(w_t-1)│   ← Table look-up in C
   └────────┘    └────────┘    └────────┘     (shared params)
        ↑              ↑              ↑
   index w_t-n+1  index w_t-2    index w_t-1
```

Notar:
- **Matriz $C$ compartida**: la misma matriz $C$ se usa para todas las posiciones del contexto. Esto es **clave** — los embeddings se aprenden una vez por palabra, no por posición.
- **Skip connection** (linea verde punteada): $Wx$ se suma directamente al output, en paralelo a la rama tanh.

---

## 4. Entrenamiento

### 4.1 Loss y SGD

Maximizar la log-likelihood penalizada:
$$
L = \frac{1}{T} \sum_t \log f(w_t, w_{t-1}, \dots, w_{t-n+1}; \theta) - R(\theta)
$$

donde $R(\theta)$ es **weight decay** sobre $W$, $H$, $U$ (no sobre $C$ ni biases).

**SGD update**:
$$
\theta \leftarrow \theta + \epsilon \cdot \frac{\partial \log \hat{P}(w_t \mid w_{<t})}{\partial \theta}
$$

con learning rate $\epsilon = 10^{-3}$ aproximadamente.

### 4.2 Optimización clave: gradientes dispersos en $C$

Cuando se procesa el ejemplo $(w_{t-n+1}, \dots, w_t)$:
- Solo las **$n-1$ filas de $C$** correspondientes al contexto reciben gradiente.
- Las demás filas no se tocan.

Esto reduce drásticamente el cómputo. Bengio lo enfatiza: *"a large fraction of the parameters needs not be updated or visited after each example."*

### 4.3 Mixture con interpolated trigram

Al final del paper (sección 4), Bengio reporta que **combinar el NPLM con un trigram suavizado** mediante interpolación lineal mejora la perplejidad:
$$
\hat{P}_{\text{mix}} = \alpha \hat{P}_{\text{NN}} + (1 - \alpha) \hat{P}_{\text{trigram}}, \quad \alpha = 0.5.
$$

Esto sugiere que los dos modelos capturan **información complementaria**: el NN generaliza mejor (palabras similares), el trigram captura n-gramas frecuentes específicos.

---

## 5. Paralelización — un capítulo aparte

La sección 3 del paper se dedica enteramente a la **infraestructura de cómputo**. En 2002 esto era no-trivial.

### 5.1 Data-parallel async SGD

Múltiples CPUs comparten memoria. Cada CPU procesa un subset de los datos y actualiza $\theta$ en memoria compartida **sin locks**. Bengio observa:

> *"Sometimes, part of an update on the parameter vector by one of the processors is lost, being overwritten by the update of another processor, and this introduces a bit of noise in the parameter updates. However, this noise seems to be very small and did not apparently slow down training."*

**Esto es asynchronous SGD ANTES de HogWild! (Niu 2011) por casi una década.** Bengio descubrió empíricamente lo que después se formalizaría: que el ruido de las actualizaciones perdidas no degrada significativamente la convergencia.

### 5.2 Parameter-parallel para softmax

Distribuyen la matriz de output ($U$, $W$, $b$) entre varios nodos de un cluster. Cada nodo computa una porción del softmax. Comunican solo el factor de normalización y los gradientes sobre la hidden y los embeddings. Speedup casi-lineal con el número de nodos.

Esta arquitectura es **conceptualmente la misma** que la paralelización moderna de Transformers (tensor parallel + data parallel + pipeline parallel).

---

## 6. Experimentos

### 6.1 Datasets

| Dataset | $|V|$ | Train tokens | Validation | Test |
|---|---|---|---|---|
| **Brown Corpus** | 16.383 (después de filtrar) | 800k | 200k | 181k |
| **AP News** (1995-1996) | 17.964 (con $|V|_{\text{full}} = 148k$) | 14M | 1M | 1M |

### 6.2 Resultados clave en Brown Corpus

| Modelo | $n$ | $h$ | $m$ | Perplejidad test |
|---|---|---|---|---|
| Trigram interpolated | 3 | — | — | 343 |
| 5-gram Kneser-Ney | 5 | — | — | 321 |
| Class-based 5-gram | 5 | — | — | 312 |
| **NPLM** | 5 | 50 | 30 | 268 |
| **NPLM with skip** | 5 | 50 | 30 | 276 |
| **NPLM mixture** (NPLM + trigram) | 5 | 50 | 30 | **252** |

**Lectura**: NPLM **reduce perplejidad ~20-25%** sobre Kneser-Ney 5-gram, el SOTA estadístico. Mixture mejora aún más.

### 6.3 AP News

| Modelo | Perplejidad |
|---|---|
| Trigram | 137 |
| **NPLM** | **109** |
| **NPLM mixture** | **104** |

**24% reducción** en perplejidad sobre trigram — enorme en términos absolutos.

### 6.4 Tiempo de cómputo

Brown corpus, NPLM con $h=50$, $m=30$, $n=5$, ~10M parámetros: ~3 semanas en cluster paralelo. Comparado con segundos para un trigram. Este costo era **lo que limitaba la adopción**: Word2Vec (2013) atacará exactamente este problema, dejando los embeddings aprendidos pero eliminando el LM completo del objetivo.

---

## 7. Aporte conceptual perdurable

### 7.1 Embeddings como representación primaria de palabras

Antes de NPLM, las palabras eran IDs o miembros de una clase discreta. NPLM demuestra que **un vector real de dimensión moderada puede capturar suficiente información semántica para ser útil**.

### 7.2 Aprendizaje conjunto de representación y modelo

NPLM enseña que las representaciones no deben ser fijas ni pre-computadas — deben **aprenderse junto con la tarea**. Esto es **el principio central del deep learning moderno**: end-to-end learning.

### 7.3 Matriz de embeddings compartida

La matriz $C$ se aplica a **todas las posiciones del contexto**. Esta idea de *parameter sharing across positions* es la base de:
- Embedding layers en RNN, LSTM, GRU.
- Word embeddings en CNN para texto.
- Token embeddings en Transformer (BERT, GPT).

### 7.4 SGD async sin locks

Como mencioné, NPLM hace en 2002 lo que HogWild! popularizaría en 2011. Es un caso de adelanto técnico que pasó desapercibido en su momento.

---

## 8. Limitaciones reconocidas

El paper es honesto sobre limitaciones (sección 5):

1. **Costo computacional**: 3 semanas en cluster para Brown (1M words). Inviable para AP News completo sin más recursos.
2. **Softmax sobre $|V|$**: el bottleneck dominante. *"main computational bottleneck"* — la solución vendrá con hierarchical softmax (Morin & Bengio 2005) y negative sampling (Mikolov 2013).
3. **Vocabulario fijo**: palabras OOV se mapean a un símbolo especial.
4. **Sin información subword**.
5. **No bidireccional**: NPLM solo ve contexto anterior. ELMo (2018) y BERT (2018) corregirán esto.
6. **Modelo Markoviano**: ventana fija de $n-1$ palabras, no captura dependencias muy largas. RNN-LM (Mikolov 2010) la elimina.

---

## 9. Impacto y legado

### 9.1 Hijos directos

| Año | Modelo | Continuidad con NPLM |
|---|---|---|
| 2005 | **Morin & Bengio — Hierarchical NPLM** | Mismo modelo, hierarchical softmax |
| 2008 | **Collobert & Weston — SENNA** | Ranking loss, embeddings preentrenados, comunicación implícita |
| 2010 | **Mikolov — RNN-LM** | NPLM + recurrencia (no markoviano) |
| 2013 | **Word2Vec** | NPLM sin hidden layer, solo embeddings importan |
| 2014 | **GloVe** | Misma filosofía (embeddings densos), diferente loss |
| 2018 | **ELMo** | NPLM con LSTM bidireccional |
| 2018 | **GPT** | NPLM con Transformer y escala masiva |
| 2018 | **BERT** | NPLM bidireccional + masked LM |

Toda esta genealogía hereda de Bengio 2003 dos cosas: (i) embeddings densos aprendidos, (ii) cross-entropy sobre vocabulario para predecir tokens contextuales.

### 9.2 Insight unificador

Lo que Bengio 2003 captura en una frase y que ha guiado al campo desde entonces:

> *"Learning simultaneously (1) a distributed representation for each word along with (2) the probability function for word sequences, expressed in terms of these representations."*

Es **literalmente** el paradigma de pre-training de modelos foundation actuales — solo cambia la arquitectura (de feedforward a Transformer) y la escala (de millones a trillones de tokens).

### 9.3 Reconocimiento tardío

NPLM no recibió mucha atención inmediatamente — su costo computacional lo hacía poco práctico. La comunidad NLP siguió usando n-gramas suavizados durante varios años. Solo con la era deep learning (2012+) y Word2Vec (2013) se reconoció universalmente que NPLM había marcado el camino.

Yoshua Bengio recibió el **Premio Turing 2018** junto con Geoffrey Hinton y Yann LeCun por su trabajo fundacional en deep learning. NPLM es uno de los papers más representativos de esa contribución.

---

## 10. Conexión con la clase 18

**Slide 21** de la clase 18 muestra el diagrama del NPLM (Word Embedding → Feedforward NN → softmax sobre vocab), bajo el título *"A.k.a. Neural Probabilistic Language Models"*. Es **literalmente** la arquitectura de este paper. La clase no lo cita explícitamente pero la imagen es una reproducción directa de la figura 1 de Bengio 2003.

Slide 21 explicita los **2 objetivos** del NPLM:
1. *Objetivo 1*: aprender una representación continua distribuida para cada palabra → word embedding.
2. *Objetivo 2*: aprender los pesos de una red neuronal que prediga la probabilidad de una palabra w dado un contexto h(w) → $P(w | h(w))$.

Exactamente los puntos 1 y 2 de la sección 1.1 de Bengio 2003.

**Slide 22** transiciona a RNN-LM (Mikolov 2010), que es el sucesor inmediato de NPLM — reemplaza la ventana fija por recurrencia.

---

## 11. Cita BibTeX

```bibtex
@article{bengio2003neural,
  title={A neural probabilistic language model},
  author={Bengio, Yoshua and Ducharme, R{\'e}jean and Vincent, Pascal and Jauvin, Christian},
  journal={Journal of Machine Learning Research},
  volume={3},
  pages={1137--1155},
  year={2003},
  url={https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf}
}
```

---

## 12. Frase para recordar

> *"Distributed representations are the antidote to the curse of dimensionality."* — Bengio 2003 es donde nace formalmente la idea de que los embeddings densos resuelven el problema combinatorio de los n-gramas. Es la fundación de todo NLP moderno.

---

## 13. Notas técnicas y curiosidades

- **Tanh, no ReLU**: ReLU (Glorot 2011) aún no se había popularizado. Tanh tiene problemas de vanishing gradients pero fue suficiente para una sola capa hidden.
- **No dropout**: el dropout (Hinton 2012) tampoco se había inventado. Bengio usa **weight decay** como regularizador.
- **Direct skip connections**: la matriz $W$ que conecta inputs directamente al output es opcional. El paper reporta que **no ayuda** en sus experimentos (Brown), pero la incluye en el modelo general.
- **Hidden layer chica**: $h = 50$ o $60$ — irrisorio comparado con los Transformers modernos ($h = 4096$ en GPT-3). Pero suficiente para mostrar la idea.
- **Implementation language**: C/C++. Python aún no era el lenguaje dominante de ML. PyTorch / TF / JAX vendrían 10+ años después.
- **GPUs no usadas**: el paper habla de clusters de CPUs Intel. CUDA salió en 2007.

---

## 14. Comparación rápida con sucesores

| | Bengio 2003 (NPLM) | Mikolov 2013 (Word2Vec) | Devlin 2018 (BERT-base) | Brown 2020 (GPT-3) |
|---|---|---|---|---|
| Embedding dim | 30-100 | 300 | 768 | 12.288 |
| Hidden | 50-60 | 0 (eliminada) | 768 × 12 layers | 12.288 × 96 layers |
| Contexto | $n=5$ | $n=10$ (ventana) | 512 tokens | 2048-4096 tokens |
| Vocab | 17k | 1M | 30k (WordPiece) | 50k (BPE) |
| Params | ~12M | ~300M (vocab × dim) | 110M | 175B |
| Training tokens | 1M (Brown) | 6B (Google News) | 3.3B (Wikipedia + BookCorpus) | 300B+ |
| Hardware | CPU cluster | CPU/laptop | 16 TPU v3 × 4 días | ~10.000 GPUs |
| Tarea | $P(w_t \| w_{<t})$ | Word context prediction | Masked LM + NSP | $P(w_t \| w_{<t})$ |
| Output útil | Embeddings + LM | Embeddings | Embeddings contextuales | Generación |

**Conclusión**: el formato base — embeddings + NN + softmax sobre vocab — sobrevive intacto desde Bengio 2003. Solo cambian las dimensiones, la arquitectura del NN, y la escala.
