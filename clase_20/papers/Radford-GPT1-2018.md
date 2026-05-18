# Improving Language Understanding by Generative Pre-Training (GPT-1)

**Autores:** Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI)
**Año:** 2018 (technical report, OpenAI Blog + preprint, sin venue formal)
**PDF analizado:** `clase_20/papers/Radford-GPT1-2018.pdf`
**Análisis para:** Diplomado IA UC — Clase 20 (de Transformers a ChatGPT)

---

## 1. Contexto histórico: el NLP de 2017–2018

Para entender la importancia real de GPT-1 hay que reconstruir el estado del arte que tenía OpenAI delante en la primera mitad de 2018. Era un momento bisagra: en menos de doce meses la comunidad pasó de "embeddings pre-entrenados de palabras" a "modelos pre-entrenados de lenguaje", y GPT-1 es exactamente el artefacto que cierra esa transición desde el lado decoder-only.

### 1.1. El paradigma dominante: embeddings pre-entrenados de palabras

Hasta finales de 2017 el "transfer learning" en NLP se reducía, en la práctica, a inicializar la capa de embeddings de un modelo supervisado con vectores pre-entrenados:

- **Word2Vec** (Mikolov et al., 2013, ref. [39] del paper): skip-gram y CBOW sobre Google News. Vectores estáticos de 300 dimensiones.
- **GloVe** (Pennington et al., 2014, ref. [42]): factorización de matriz de coocurrencias global. Mismo tamaño, mejor desempeño en analogías.
- **fastText** (Bojanowski et al., 2017): subword embeddings vía n-gramas de caracteres.

El resto del modelo (LSTM bidireccional con attention, BiLSTM + CRF, etc.) se entrenaba **desde cero** para cada tarea, sobre datasets supervisados que típicamente tenían entre 1k y 500k ejemplos. Esto generaba dos cuellos de botella:

1. **Escasez de datos etiquetados**: anotar SNLI o MultiNLI cuesta cientos de miles de dólares; para dominios especializados (médico, legal) directamente no existían corpora de ese tamaño.
2. **Especialización arquitectónica**: cada paper proponía una arquitectura distinta por tarea (ESIM para NLI, BiDAF para QA, Tree-LSTM para sentiment), con poca transferencia entre ellas.

La pregunta de fondo era: ¿se puede transferir **más que palabras** desde texto no anotado? Es decir, ¿se puede capturar sintaxis, semántica composicional, coreferencia, sentido común, todo en parámetros pre-entrenados?

### 1.2. La primera ola de pre-training contextual (2017–Feb 2018)

Varias líneas de trabajo apuntaron a esa pregunta en paralelo:

- **CoVe** (McCann et al., 2017, NeurIPS, ref. [38]): entrenan un encoder LSTM bidireccional para traducción inglés→alemán, y usan sus estados ocultos como features adicionales en tareas downstream. Punto débil: depende de pares paralelos, que son escasos.
- **Semi-supervised Sequence Learning** (Dai & Le, 2015, ref. [13]): pre-entrenan un LSTM con objetivo de modelado de lenguaje y luego fine-tunean. Es el antecedente directo de GPT-1 en espíritu, pero con LSTM (capacidad limitada para long-range dependencies) y resultados modestos.
- **ULMFiT** (Howard & Ruder, 2018, ACL, ref. [21]): AWD-LSTM pre-entrenado en WikiText-103, fine-tuning con learning rates discriminativos y slanted triangular schedules. Demostró transferencia efectiva en clasificación de texto.
- **ELMo** (Peters et al., NAACL 2018, ref. [44]): BiLM de dos LSTMs (forward + backward) entrenado sobre 1B Word Benchmark. La novedad fue **contextual word representations**: la representación de cada token depende de su contexto en la oración. ELMo se usaba como features (capa adicional, sin fine-tunear el LM completo).

ELMo y GPT-1 son contemporáneos y conceptualmente rivales: ambos atacan el mismo problema (representaciones contextuales transferibles) pero desde ángulos opuestos:

| Eje | ELMo (Feb 2018) | GPT-1 (Jun 2018) |
|---|---|---|
| Arquitectura | BiLSTM (2 capas) | Transformer decoder (12 capas) |
| Direccionalidad | Bidireccional (concatena fwd+bwd LMs) | Unidireccional (forward LM) |
| Uso downstream | Features fijos + arquitectura task-specific | Fine-tuning end-to-end del LM completo |
| Capacidad de long-range | Limitada por recurrencia | Self-attention O(n²) directa |
| Tamaño | ~94M parámetros | 117M parámetros |

### 1.3. El Transformer existía desde hace ocho meses

"Attention is All You Need" (Vaswani et al., NeurIPS 2017, ref. [62]) había aparecido en junio de 2017. Su impacto inicial fue en traducción automática (encoder-decoder), y a comienzos de 2018 todavía no estaba claro si era una arquitectura de propósito general o un truco específico de seq2seq. Existían algunas extensiones:

- Liu et al., "Generating Wikipedia by summarizing long sequences" (ICLR 2018, ref. [34]): introducen un **decoder-only Transformer** para resumen abstractivo de documentos largos. GPT-1 cita este trabajo como inspiración directa para la elección de arquitectura.
- Kitaev & Klein (ACL 2018, ref. [29]): constituency parsing con self-attentive encoder.

OpenAI tomó tres apuestas que retrospectivamente parecen obvias pero en su momento no lo eran:

1. **Transformer en vez de LSTM**: justificado por la capacidad de modelar long-range dependencies sin el cuello de botella secuencial.
2. **Decoder-only (causal masking) en vez de encoder-decoder o bidireccional**: porque el objetivo de pre-training es language modeling autoregresivo, y un decoder con masked self-attention es la forma natural de implementarlo. Además, evita la complicación de combinar dos modelos (fwd + bwd) como ELMo.
3. **Fine-tuning end-to-end del modelo entero**, no extracción de features: la mejor manera de usar la representación es seguir adaptándola al downstream.

### 1.4. La pregunta abierta que el paper plantea

El abstract y la introducción son explícitos: hay dos preguntas no resueltas en 2018:

> "First, it is unclear what type of optimization objectives are most effective at learning text representations that are useful for transfer. (...) Second, there is no consensus on the most effective way to transfer these learned representations to the target task."

Sobre la primera pregunta, GPT-1 apuesta por **language modeling puro** (predicción next-token, autoregresivo). Sobre la segunda, apuesta por **fine-tuning end-to-end con cambios mínimos a la arquitectura**, mediante input transformations.

Esta combinación, simple en retrospectiva, es la contribución central del paper y la base sobre la que se construirán GPT-2, GPT-3, ChatGPT y toda la familia decoder-only que dominará desde 2020.

---

## 2. Contribución central

GPT-1 establece un framework de dos etapas:

1. **Generative pre-training**: maximizar la log-verosimilitud de un modelo de lenguaje autoregresivo (forward LM) sobre un corpus grande de texto no etiquetado.
2. **Discriminative fine-tuning**: adaptar los parámetros pre-entrenados a una tarea supervisada combinando dos objetivos —tarea + LM auxiliar— y usando **input transformations** task-specific para convertir entradas estructuradas en una secuencia única de tokens.

### 2.1. Objetivo de pre-training

Dado un corpus de tokens $\mathcal{U} = \{u_1, \dots, u_n\}$, se maximiza:

$$
L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)
$$

donde $k$ es el tamaño de la ventana de contexto y $\Theta$ son los parámetros de la red. Este es el objetivo clásico de language modeling causal (forward LM). El paper enfatiza:

- **Único objetivo**: no hay multi-task pre-training, ni objetivos auxiliares, ni masked LM. Solo predicción next-token.
- **Causal masking**: la atención de cada posición $i$ solo puede ver posiciones $< i$. Esto se implementa con una máscara triangular inferior en la matriz de scores de attention antes del softmax.

La arquitectura se describe en tres ecuaciones compactas:

$$
\begin{aligned}
h_0 &= U W_e + W_p \\
h_l &= \text{transformer\_block}(h_{l-1}) \quad \forall l \in [1, n] \\
P(u) &= \text{softmax}(h_n W_e^T)
\end{aligned}
$$

donde $U = (u_{-k}, \dots, u_{-1})$ es la ventana de contexto, $W_e \in \mathbb{R}^{|V| \times d}$ es la matriz de embeddings de tokens (compartida entre input y output, **weight tying**), y $W_p \in \mathbb{R}^{k \times d}$ es la matriz de embeddings posicionales **aprendidos** (no sinusoidales como en el Transformer original).

### 2.2. Objetivo de fine-tuning

Dado un dataset etiquetado $\mathcal{C}$ donde cada ejemplo es una secuencia $x^1, \dots, x^m$ con etiqueta $y$, se pasa la secuencia por el modelo pre-entrenado, se toma la activación del **último token** ($h_l^m$) en la última capa, y se la proyecta linealmente:

$$
P(y \mid x^1, \dots, x^m) = \text{softmax}(h_l^m W_y)
$$

El objetivo supervisado puro es:

$$
L_2(\mathcal{C}) = \sum_{(x, y)} \log P(y \mid x^1, \dots, x^m)
$$

Pero GPT-1 introduce un truco importante: combinar este objetivo con el LM auxiliar:

$$
L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C})
$$

con $\lambda = 0.5$ en los experimentos. La justificación es doble:

1. **Mejor generalización**: el LM auxiliar actúa como regularizador, evitando que el fine-tuning destruya las representaciones lingüísticas aprendidas en pre-training.
2. **Convergencia acelerada**: pocas épocas (3) bastan para fine-tunear, en parte porque el modelo no tiene que "reaprender" desde cero el comportamiento de lenguaje.

Esta idea (auxiliar LM loss durante fine-tuning) sería más tarde descartada en GPT-2/GPT-3 (porque trabajan zero/few-shot, sin gradientes en el fine-tuning) y reemplazada por el "Prefix LM" implícito que aparece cuando se hace in-context learning.

Los **únicos parámetros nuevos** durante fine-tuning son $W_y$ (matriz de proyección a logits de clases) y los embeddings de los tokens delimitadores especiales (`<s>`, `<e>`, `$`). Todo el resto del modelo se inicializa desde el pre-trained y se sigue ajustando con gradientes.

### 2.3. Input transformations: el ingrediente práctico

Una de las contribuciones más subestimadas del paper es la idea de **convertir cualquier tarea supervisada en una secuencia de tokens unificada**, en lugar de modificar la arquitectura del modelo. Esto se hace agregando tokens delimitadores especiales:

- `<s>` (Start): marca el inicio de la secuencia.
- `<e>` (Extract): marca el final, su representación final se usa para la clasificación.
- `$` (Delim): separa segmentos dentro de la secuencia (por ejemplo premisa de hipótesis).

El paper detalla cuatro tipos de transformaciones (Figura 1 del paper):

1. **Classification** (texto único): `<s> texto <e>` → linear sobre la activación de `<e>`.
2. **Textual entailment**: `<s> premisa $ hipótesis <e>` → linear sobre `<e>`.
3. **Similarity**: dos órdenes (`<s> texto1 $ texto2 <e>` y `<s> texto2 $ texto1 <e>`) procesados independientemente; las representaciones finales se **suman elemento a elemento** antes de la proyección linear (porque la similitud es simétrica y no hay orden natural).
4. **Multiple choice** (QA, sentido común): para cada opción $a_k$ se construye `<s> contexto $ pregunta $ respuesta_k <e>`, se pasan las $N$ secuencias independientemente, se obtiene un score por opción, y se aplica softmax sobre las $N$ opciones.

Esta estrategia tiene un nombre técnico: **traversal-style approach** (Rocktäschel et al., 2015, ref. [52]). La aplicación a fine-tuning con un Transformer pre-entrenado es la innovación de GPT-1.

Cabe notar que este enfoque **prefigura el formato de "prompt"** que dominará la era GPT-3+: tomar una tarea estructurada y convertirla en una secuencia de tokens es exactamente lo que hace un prompt template. La diferencia es que GPT-1 aprende los parámetros de las representaciones de los delimitadores vía fine-tuning, mientras que GPT-3 lo hace zero-shot vía in-context learning.

---

## 3. Arquitectura

### 3.1. Hiperparámetros y dimensiones

| Parámetro | Valor |
|---|---|
| Capas (decoder blocks) | 12 |
| Dimensión del modelo ($d_{\text{model}}$) | 768 |
| Heads de atención | 12 |
| Dimensión por head ($d_k = d_v$) | 64 (= 768/12) |
| FFN interno | 3072 (= 4 × 768) |
| Activación FFN | GELU (Hendrycks & Gimpel, 2016) |
| Position embeddings | Aprendidos (matriz $W_p$) |
| Ventana de contexto | 512 tokens |
| Vocabulario BPE | 40,000 merges |
| Dropout | 0.1 (residual, embeddings, attention) |
| Parámetros totales | ~117M |

Notar que **12 capas × 768 dim ≈ 117M parámetros** es casi exactamente el tamaño de BERT-base (110M, 12 capas × 768 dim), lo que hace que las comparaciones entre ambos sean particularmente limpias en términos de capacidad. La diferencia fundamental es estructural: GPT-1 es decoder-only con masked self-attention causal, BERT es encoder-only con self-attention bidireccional y objetivo MLM.

### 3.2. Bloque Transformer (decoder-only)

Cada bloque sigue la estructura **pre-norm** (LayerNorm antes del sub-layer, aunque el paper la describe en orden post-norm en algunas figuras; el código publicado de OpenAI es pre-norm):

$$
\begin{aligned}
\tilde{h}_l &= h_{l-1} + \text{MaskedMHA}(\text{LN}(h_{l-1})) \\
h_l &= \tilde{h}_l + \text{FFN}(\text{LN}(\tilde{h}_l))
\end{aligned}
$$

Componentes:

1. **Masked Multi-Head Self-Attention**: idéntico al del Transformer encoder excepto por la máscara causal $M \in \{0, -\infty\}^{n \times n}$ aplicada antes del softmax:

   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V
   $$

   donde $M_{ij} = -\infty$ si $j > i$, y $0$ en otro caso. Esto garantiza que la posición $i$ solo atiende a $j \leq i$.

2. **Position-wise FFN**: dos capas lineales con GELU en el medio:

   $$
   \text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
   $$

   con $W_1 \in \mathbb{R}^{768 \times 3072}$ y $W_2 \in \mathbb{R}^{3072 \times 768}$.

3. **LayerNorm**: aplicada antes de cada sub-layer.

### 3.3. Detalles de tokenización

GPT-1 usa **Byte-Pair Encoding (BPE)** (Sennrich et al., 2015, ref. [53]) con 40k merges. La justificación de BPE en NLP era:

- Vocabulario fijo (no OOV en runtime).
- Trade-off entre granularidad de carácter (sin OOV pero secuencias largas) y de palabra (corto pero OOV en cola larga).
- Las palabras frecuentes quedan como tokens únicos; las raras se descomponen en subwords.

El texto se pre-procesa con `ftfy` (corrige encoding Unicode roto) y se tokeniza con spaCy antes de aplicar BPE. Este detalle de ingeniería es importante porque BookCorpus tiene mucho texto OCR-eado con errores de encoding.

### 3.4. Por qué decoder-only y no encoder-decoder

El paper no lo discute explícitamente, pero la lógica es:

1. **Objetivo de pre-training es LM autoregresivo**, que es naturalmente decoder-only.
2. **No hay fuente de entrada distinta de la salida** durante pre-training (a diferencia de traducción), así que un encoder separado sería redundante.
3. **Las tareas downstream se reformulan como secuencias únicas** vía input transformations, así que la arquitectura encoder-decoder no aporta nada.
4. **Coherencia entre pre-training y fine-tuning**: el modelo siempre ve la misma cosa, una secuencia causal de tokens.

Esta elección tendrá consecuencias profundas. BERT (octubre 2018) elegirá lo opuesto —encoder-only con MLM— y ganará en understanding benchmarks. Pero la apuesta decoder-only de GPT-1 se validará cuando se descubra (GPT-2/3) que la generación condicional es la forma natural de unificar todas las tareas, incluyendo las de understanding.

---

## 4. Pre-training: corpus y procedimiento

### 4.1. BookCorpus

El corpus de pre-training es **BookCorpus** (Zhu et al., 2015, ref. [71]):

- ~7,000 libros únicos no publicados, de géneros variados (aventura, fantasía, romance).
- ~800M palabras (~5GB de texto).
- Característica clave: **contiene long stretches of contiguous text**, lo que permite que el modelo aprenda dependencias largas (referencias anafóricas, coherencia de párrafo, arcos narrativos, etc.).

El paper compara explícitamente con **1B Word Benchmark** (usado por ELMo), que es de tamaño similar pero **shuffled a nivel de oración**, destruyendo la estructura de largo alcance. GPT-1 alcanza perplexity de 18.4 a nivel de token en BookCorpus, un número bajo que sugiere que el modelo está modelando bien la distribución.

El uso de BookCorpus tiene también implicaciones de licencia (el corpus fue construido scrapeando libros de SmashWords, no comerciales pero con copyright variado) que volverán como tema legal con GPT-2 y posteriores.

### 4.2. Hiperparámetros de entrenamiento

| Hiperparámetro | Valor |
|---|---|
| Optimizador | Adam (Kingma & Ba, 2014) |
| Learning rate máximo | 2.5e-4 |
| Schedule | Warmup linear (2000 steps) → cosine annealing a 0 |
| Batch size | 64 secuencias |
| Longitud de secuencia | 512 tokens |
| Épocas | 100 |
| Weight decay | L2 modificado ($w = 0.01$) sobre pesos no-bias y no-gain (Loshchilov & Hutter, 2017) |
| Inicialización | $\mathcal{N}(0, 0.02)$ |
| Dropout | 0.1 |

100 épocas sobre BookCorpus a batch 64 × 512 tokens da aproximadamente $100 \times 800\text{M} / (64 \times 512) \approx 2.4\text{M}$ updates. Comparativamente, BERT-base entrena ~1M updates sobre 16GB de texto (BookCorpus + Wikipedia). GPT-2 entrenará sobre 40GB de WebText. GPT-3 sobre ~570GB filtrados. La escalada es de varios órdenes de magnitud.

El uso de **Adam + cosine schedule + warmup** se convertirá en la receta estándar para entrenar Transformers. La inicialización $\mathcal{N}(0, 0.02)$ es relativamente pequeña, lo que ayuda con la estabilidad de pre-norm Transformers profundos.

---

## 5. Fine-tuning por tarea

### 5.1. Hiperparámetros de fine-tuning

| Hiperparámetro | Valor |
|---|---|
| Learning rate | 6.25e-5 |
| Batch size | 32 |
| Épocas | 3 (típicamente suficiente) |
| Warmup | 0.2% del total de pasos |
| Dropout en clasificador | 0.1 |
| $\lambda$ (auxiliar LM weight) | 0.5 |

La velocidad de convergencia (3 épocas) es notable y refleja que el modelo ya tiene representaciones útiles; solo necesita ajustes finos para adaptarlas a la tarea.

### 5.2. Tipos de transformaciones (detalle)

**Classification (single text)**: Stanford Sentiment Treebank-2 (SST-2), CoLA.
- Formato: `<s> texto <e>`
- Predicción: $h_l^m$ (último token) → linear de $W_y \in \mathbb{R}^{768 \times C}$ → softmax sobre $C$ clases.

**Textual entailment**: SNLI, MultiNLI, QNLI, RTE, SciTail.
- Formato: `<s> premisa $ hipótesis <e>`
- Etiquetas: {entailment, contradiction, neutral}.
- Predicción: $h_l^m$ → linear → softmax sobre 3 clases (o 2 para datasets binarios).

**Similarity**: MRPC, QQP, STS-B.
- Problema: la similitud es simétrica, pero el modelo causal sí tiene orden.
- Solución: procesar las dos órdenes y sumar:

  $$
  h^{\text{sim}} = h_l^{m}(\langle s \rangle, t_1, \$, t_2, \langle e \rangle) + h_l^{m}(\langle s \rangle, t_2, \$, t_1, \langle e \rangle)
  $$

  Luego linear sobre $h^{\text{sim}}$.
- STS-B (regresión continua de 0 a 5): la cabeza es linear sin softmax.

**Multiple Choice**: RACE, Story Cloze, COPA-style.
- Para una pregunta con $N$ alternativas $\{a_1, \dots, a_N\}$:
  - Se construyen $N$ secuencias `<s> contexto $ pregunta $ a_k <e>`.
  - Cada una se pasa independientemente por el Transformer, obteniendo un escalar $s_k = h_l^m W_y$ (con $W_y \in \mathbb{R}^{768 \times 1}$).
  - $P(a_k \mid \text{contexto, pregunta}) = \text{softmax}(s_1, \dots, s_N)_k$.
- Entrenamiento con cross-entropy sobre la opción correcta.

Esta formulación de multiple choice es elegante: en vez de modificar la arquitectura para aceptar $N$ entradas, se ejecuta el mismo modelo $N$ veces y se compara. Es exactamente el enfoque de **likelihood scoring** que GPT-3 usará para zero/few-shot multiple choice.

### 5.3. La tabla 1 del paper: cobertura de tareas

| Tarea | Datasets |
|---|---|
| Natural language inference | SNLI, MultiNLI, Question NLI (QNLI), RTE, SciTail |
| Question answering | RACE, Story Cloze |
| Sentence similarity | MRPC, QQP, STS-B |
| Classification | SST-2, CoLA |

Total: 12 datasets, cubriendo 4 grandes tipos de tareas. GLUE (Wang et al., 2018, ref. [64]) acababa de ser introducido como benchmark unificado y GPT-1 fue uno de los primeros en reportarse contra él (logrando 72.8 vs 68.9 del estado del arte previo).

---

## 6. Experimentos: resultados

### 6.1. Natural Language Inference (Tabla 2 del paper)

| Método | MNLI-m | MNLI-mm | SNLI | SciTail | QNLI | RTE |
|---|---|---|---|---|---|---|
| ESIM + ELMo (5x ensemble) | — | — | 89.3 | — | — | — |
| CAFE (5x) | 80.2 | 79.0 | 89.3 | — | — | — |
| CAFE (single) | 78.7 | 77.9 | 88.5 | 83.3 | — | — |
| Multi-task BiLSTM + Attn | 72.2 | 72.1 | — | — | 82.1 | **61.7** |
| **GPT-1 (single model)** | **82.1** | **81.4** | **89.9** | **88.3** | **88.1** | 56.0 |

Mejoras absolutas notables: +1.5% en MNLI-m, +5% en SciTail, +5.8% en QNLI, +0.6% en SNLI. RTE (2490 ejemplos) es la única excepción: con tan pocos datos, multi-task biLSTM con attention todavía gana. El paper reconoce: "Given the strong performance of our approach on larger NLI datasets, it is likely our model will benefit from multi-task training as well but we have not explored this currently."

### 6.2. Question Answering y Commonsense (Tabla 3)

| Método | Story Cloze | RACE-m | RACE-h | RACE |
|---|---|---|---|---|
| Hidden Coherence Model | 77.6 | — | — | — |
| BiAttention MRU (9x) | — | 60.2 | 50.3 | 53.3 |
| **GPT-1** | **86.5** | **62.9** | **57.4** | **59.0** |

+8.9% absoluto en Story Cloze, +5.7% absoluto en RACE. Estos son los gains más espectaculares del paper: tareas que requieren razonamiento sobre múltiples oraciones y contextos largos son exactamente donde la capacidad de long-range del Transformer brilla.

### 6.3. Similarity y Classification (Tabla 4)

| Método | CoLA (mc) | SST-2 (acc) | MRPC (F1) | STS-B (pc) | QQP (F1) | GLUE |
|---|---|---|---|---|---|---|
| Sparse byte mLSTM | — | 93.2 | — | — | — | — |
| Multi-task BiLSTM + ELMo + Attn | 18.9 | 91.6 | 83.5 | 72.8 | 63.3 | 68.9 |
| **GPT-1** | **45.4** | 91.3 | 82.3 | **82.0** | **70.3** | **72.8** |

Resultado especialmente notable: **CoLA pasa de 35.0 a 45.4 (correlación Matthews)**. CoLA mide aceptabilidad gramatical: si una oración es sintácticamente válida en inglés. Que GPT-1 mejore tanto sugiere que el pre-training capturó conocimiento gramatical implícito que ningún modelo supervisado puro había logrado extraer.

### 6.4. Resumen global

- **SOTA en 9 de 12 datasets**.
- GLUE score: 72.8 (vs 68.9 anterior).
- Funciona bien tanto en datasets pequeños (STS-B, ~5.7k ejemplos) como grandes (SNLI, ~550k).
- La mejora más grande en absolute terms es **Story Cloze (+8.9%)**, una tarea de sentido común narrativo.

---

## 7. Análisis: las dos figuras clave

El paper dedica una sección entera (Section 5) a entender **por qué** el método funciona. Hay dos análisis muy citados.

### 7.1. Impact of number of layers transferred (Figura 2 izquierda)

El experimento: tomar el modelo pre-entrenado, fine-tunear usando solo las primeras $k$ capas transferidas (las capas $k+1, \dots, 12$ se reinicializan aleatoriamente), variando $k \in \{0, 1, \dots, 12\}$.

**Resultados** (sobre MultiNLI y RACE):

- Con $k=0$ (solo embeddings transferidos), la accuracy ya sube sobre el baseline random.
- Cada capa adicional aporta accuracy incremental, **monotónicamente**.
- Transferir todas las 12 capas vs solo embeddings da hasta **+9% absoluto** en MultiNLI.

**Implicación**: cada capa del Transformer contiene "funcionalidad útil" para tareas downstream. No hay una capa privilegiada (como podría sugerir ELMo, que aprende una combinación lineal). El conocimiento está distribuido a lo largo de la profundidad. Este resultado se replicará en análisis posteriores de BERT (probing classifiers de Tenney et al., 2019).

### 7.2. Zero-shot behaviors (Figura 2 derecha)

Este es probablemente **el experimento más profético del paper**. La hipótesis: el modelo de lenguaje, durante pre-training, aprende a hacer muchas tareas implícitamente, porque hacerlo bien le ayuda a predecir el siguiente token. Para validar esto, los autores diseñan **heurísticas zero-shot** que usan el LM pre-entrenado **sin ningún fine-tuning** y miden su accuracy a lo largo del pre-training.

Las heurísticas zero-shot diseñadas (tabla informal a partir del paper):

| Tarea | Heurística zero-shot |
|---|---|
| Sentiment (SST-2) | Append "very" → comparar $P(\text{positive})$ vs $P(\text{negative})$ |
| Linguistic acceptability (CoLA) | Average token log-probability → threshold |
| Question answering (RACE) | Para cada respuesta, average log-prob condicionada en contexto + pregunta → argmax |
| Winograd Schema (DPRD) | Reemplazar el pronombre por cada referente, scorear el resto, argmax |

**Resultados** (Figura 2 derecha):

- A medida que avanza el pre-training (eje X = LM updates), la accuracy de estas heurísticas **sube monotónicamente** en todas las tareas medidas.
- Sentiment, Winograd, CoLA y QA muestran trayectorias suaves de mejora.
- Un LSTM equivalente (entrenado con el mismo objetivo y mismos datos) muestra **mucha más varianza** en su comportamiento zero-shot, sugiriendo que **el inductive bias del Transformer ayuda específicamente a este tipo de transferencia**.

**Implicación profunda**: el LM, simplemente al optimizar por predicción next-token sobre texto natural, está aprendiendo a hacer sentiment analysis, parsing, sentido común, etc. **Esto es exactamente la tesis de GPT-3** ("Language Models are Few-Shot Learners", 2020). GPT-1 ya tenía esta observación en 2018, pero el modelo era demasiado pequeño para que las accuracies zero-shot fueran competitivas con SOTA, así que el paper la presenta como "análisis" y no como "principal contribución".

### 7.3. Ablation studies (Tabla 5)

Tres ablations:

1. **Sin auxiliar LM durante fine-tuning**: ayuda en datasets grandes (NLI, QQP), no en pequeños. Avg score baja de 74.7 a 75.0 (curiosamente sube un poco; la utilidad del aux LM es marginal en promedio pero importante en algunos casos).
2. **Transformer → LSTM (2048 unidades, 1 capa) con mismo framework**: avg score cae 5.6 puntos (74.7 → 69.1). El LSTM solo gana en MRPC. **Conclusión: la arquitectura Transformer es crítica, no es solo el pre-training**.
3. **Sin pre-training (Transformer entrenado from scratch)**: avg score cae 14.8 puntos (74.7 → 59.9). **Conclusión: el pre-training es la fuente principal de la mejora**.

Resumen cuantitativo del aporte de cada ingrediente:

| Variante | Avg Score | Delta vs full |
|---|---|---|
| Transformer + pre-training + aux LM (full) | 74.7 | — |
| Transformer + pre-training (no aux LM) | 75.0 | +0.3 |
| LSTM + pre-training + aux LM | 69.1 | -5.6 |
| Transformer sin pre-training | 59.9 | -14.8 |

Esta tabla es lo más cercano a una "ecuación de Lavoisier" del transfer learning en NLP de 2018: **pre-training >> arquitectura >> auxiliar LM**.

---

## 8. Limitaciones reconocidas (y otras detectadas)

### 8.1. Limitaciones explícitas del paper

El paper es relativamente humilde y reconoce:

- **RTE underperforms**: 56% vs 61.7% de Multi-task BiLSTM + Attn. Los autores hipotetizan que multi-task training ayudaría.
- **No exploran multi-task fine-tuning**, aunque la intuición sugiere que ayudaría.
- **Solo evalúan en inglés** y solo en tareas de NLU clásicas.

### 8.2. Limitaciones estructurales (en retrospectiva)

Desde la perspectiva de 2026, las limitaciones obvias de GPT-1 son:

1. **Unidireccionalidad**: el modelo solo ve contexto izquierdo. Para tareas como NLI o QA donde el contexto bidireccional ayuda, esto es subóptimo. **BERT (4 meses después)** explotará esta debilidad usando MLM (Masked Language Modeling) que sí permite atender en ambas direcciones. BERT-large ganará en casi todos los benchmarks que GPT-1 reportó. La comunidad concluirá apresuradamente que "decoder-only es inferior", una conclusión que GPT-3 desmentirá en 2020.

2. **Necesita fine-tuning por tarea**: cada tarea requiere su propio fine-tuning, su propio dataset etiquetado, su propio modelo final almacenado. Esto no escala a "miles de tareas". GPT-2 (2019) atacará esto mostrando zero-shot competitivo en algunas tareas. GPT-3 (2020) lo perfeccionará con few-shot in-context learning.

3. **Input transformations ad-hoc**: los delimitadores `<s>`, `<e>`, `$` son una solución de ingeniería específica. No hay un formato natural. GPT-2/3 usarán **prompts en lenguaje natural** ("Translate English to French: cheese =>"), que son más flexibles y emergen sin necesidad de tokens especiales.

4. **Tamaño limitado**: 117M parámetros, 800M palabras de training. Esto era estado del arte en 2018 pero pequeño comparado con lo que vendrá. GPT-3 es 1500× más grande (175B params) y entrenado en ~750× más texto.

5. **BookCorpus es un dominio sesgado**: ficción narrativa principalmente. No hay código, ni papers científicos, ni diálogos, ni instrucciones. Esto limita la generalización del modelo.

6. **No hay alignment**: el LM aprende lo que está en BookCorpus, sin ningún esfuerzo de hacerlo útil para humanos. InstructGPT (2022) atacará esto con RLHF, dando origen a ChatGPT.

7. **Auxiliar LM loss durante fine-tuning** (con $\lambda=0.5$) es un kludge que requiere balancing fino. Trabajos posteriores (BERT, RoBERTa, T5) lo abandonarán porque el pre-training es suficientemente robusto.

8. **Vocabulario BPE en byte-pair de texto**: no maneja bien lenguajes no-latinos, código, emojis. GPT-2 introducirá **byte-level BPE** sobre bytes UTF-8, eliminando los problemas de OOV de Unicode.

---

## 9. Impacto y legado

### 9.1. Impacto inmediato (2018–2019)

GPT-1 fue publicado en junio de 2018 como technical report en el blog de OpenAI. No fue a ningún venue formal (no a NeurIPS, no a ACL). Esto es importante: la publicación informal limitó su impacto académico inicial.

**Octubre 2018: BERT** (Devlin et al., NAACL 2019). Misma idea (pre-training + fine-tuning) pero con MLM bidireccional. BERT eclipsa rápidamente a GPT-1 en benchmarks NLU. La comunidad académica gravita hacia BERT y sus derivados (RoBERTa, ALBERT, ELECTRA, DistilBERT) durante 2019–2020.

**Febrero 2019: GPT-2**. OpenAI dobla la apuesta decoder-only: 1.5B parámetros, 40GB de WebText, sin fine-tuning. La controversia de "demasiado peligroso para liberar" pone a OpenAI en titulares. GPT-2 demuestra zero-shot competitivo en varias tareas, validando la hipótesis emergente de GPT-1.

**Mayo 2020: GPT-3**. 175B parámetros, ~570GB de texto. Few-shot in-context learning. Cambia el paradigma: ya no hay fine-tuning, hay prompting. La industria entera pivota hacia LLMs decoder-only.

**Noviembre 2022: ChatGPT**. GPT-3.5 + RLHF (InstructGPT) en una interfaz de chat. Cambia el mundo.

### 9.2. Legado conceptual

GPT-1 estableció cuatro principios que siguen vigentes en 2026:

1. **Generative pre-training como ruta hacia generalismo**: la idea de que un modelo entrenado para predecir tokens aprende muchas habilidades emergentes ha sido validada en cada escalamiento posterior.
2. **Decoder-only + causal LM**: la arquitectura dominante para LLMs grandes. Encoder-only (BERT family) sobrevive en aplicaciones de búsqueda/embedding, pero los LLMs generativos son todos decoder-only.
3. **Transfer learning end-to-end**: el modelo entero se sigue ajustando, no se congelan capas. Esto se mantiene incluso en LoRA y otros métodos de PEFT.
4. **Reformular tareas como secuencias**: la idea de input transformations es el ancestro directo del prompting moderno y el chat templating.

Adicionalmente, el experimento de zero-shot behaviors (Figura 2 derecha) es **una de las primeras evidencias empíricas de que las capacidades emergen del pre-training puro**. Esta hipótesis es la base de la "scaling hypothesis" que justifica todo el gasto computacional de la era post-2020.

### 9.3. Lo que el paper no anticipó

A pesar de su visión, GPT-1 no anticipó:

- **In-context learning**: la idea de que el prompt mismo puede contener ejemplos few-shot.
- **Chain-of-thought prompting**: que pedir razonamiento paso a paso mejora performance.
- **RLHF y alignment**: que el LM puro necesita ajuste para ser útil/seguro.
- **Multimodalidad**: GPT-4 con imágenes, Gemini con audio/video.
- **Razonamiento simbólico estable**: GPT-1 y sus sucesores son todavía notoriamente débiles en matemática, lógica formal, y planning.

---

## 10. Conexión con la Clase 20 del curso (de Transformers a ChatGPT)

GPT-1 es la **primera pieza del arco narrativo de la Clase 20**. La clase recorre:

1. **GPT-1 (2018)**: pre-training + fine-tuning. Decoder-only nace.
2. **BERT (2018)**: encoder-only + MLM. Domina NLU.
3. **GPT-2 (2019)**: scaling + zero-shot. "Language Models are Unsupervised Multitask Learners".
4. **GPT-3 (2020)**: 175B params, few-shot in-context learning. "Language Models are Few-Shot Learners".
5. **InstructGPT/ChatGPT (2022)**: RLHF, alignment, chat interface.

La comparación más directa es **GPT-1 vs BERT** (mismo tamaño, mismo año):

| Aspecto | GPT-1 (Jun 2018) | BERT-base (Oct 2018) |
|---|---|---|
| Arquitectura | Transformer decoder (causal mask) | Transformer encoder (bidireccional) |
| Capas | 12 | 12 |
| Dim | 768 | 768 |
| Heads | 12 | 12 |
| Parámetros | 117M | 110M |
| Objetivo pre-training | Forward LM (next-token) | MLM + NSP |
| Datos | BookCorpus (800M words) | BookCorpus + Wikipedia (3.3B words) |
| Fine-tuning | End-to-end + auxiliar LM | End-to-end + classification head |
| GLUE | 72.8 | 78.3 (base), 80.5 (large) |
| Direccionalidad | Unidireccional | Bidireccional |
| Posicional | Aprendido | Aprendido |
| Activación | GELU | GELU |
| Tokenización | BPE (40k) | WordPiece (30k) |

**Lección clave**: con arquitecturas casi idénticas (mismo tamaño, mismo año, ambos basados en Vaswani 2017), la decisión de **direccionalidad y objetivo de pre-training** define el comportamiento downstream. BERT gana en understanding (clasificación, QA extractivo), GPT-1 sienta las bases para la generación.

La narrativa de la Clase 20 es que esta aparente derrota de GPT-1 en 2018 se revierte en 2020+ cuando la escala revela que **decoder-only generative pre-training escala mejor** que encoder MLM. El motivo (en retrospectiva): un decoder es naturalmente generativo, y la generación es la tarea más general (subsume clasificación, QA, traducción, summarization, etc.). Un encoder MLM es bueno para representaciones pero no genera secuencias coherentes largas sin modificaciones (ver: BART, T5).

GPT-1 es entonces, en el marco de la Clase 20, **el momento fundacional del paradigma decoder-only que culmina en ChatGPT**. Sin GPT-1, no hay GPT-2, no hay GPT-3, no hay ChatGPT. La línea genealógica es directa.

---

## 11. Notas para integrar al sitio Hugo

Para la versión condensada en `papers/gpt-1-radford-2018.md` del sitio del curso, los puntos que conviene destacar (en este orden de importancia):

1. **Three-sentence summary**: Pre-training generativo + fine-tuning discriminativo, decoder-only Transformer 117M sobre BookCorpus, SOTA en 9/12 datasets NLU. Establece el paradigma decoder-only que culminará en ChatGPT cuatro años después.

2. **La fórmula central**:
   $$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C})$$
   Auxiliar LM loss durante fine-tuning como regularizador.

3. **La figura clave**: zero-shot behaviors emergen con el pre-training (Figura 2 derecha). Antecedente directo de GPT-3.

4. **Tabla comparativa con BERT**: mismo tamaño (117M vs 110M), mismas dimensiones (12L × 768d), distinto objetivo (forward LM vs MLM), distinta direccionalidad.

5. **Input transformations**: el ancestro del prompting moderno.

6. **Las tres ablations** (sin pre-training: -14.8 pts; sin Transformer: -5.6 pts; sin aux LM: -0.3 pts) que resumen "qué importa más".

7. **Conexión cronológica al resto de la Clase 20**: GPT-1 (2018) → BERT (2018, derrota momentánea) → GPT-2 (2019, zero-shot validado) → GPT-3 (2020, few-shot) → ChatGPT (2022, alignment).

8. **Lectura recomendada complementaria**: Vaswani et al. 2017 (Transformer), Peters et al. 2018 (ELMo, contemporáneo), Devlin et al. 2018 (BERT, comparación directa), Liu et al. 2018 ICLR (decoder-only Transformer original).

9. **Snippets de código de referencia**: el repositorio original de OpenAI (`openai/finetune-transformer-lm` en GitHub) es bastante legible. Vale la pena para mostrar cómo se implementaba un Transformer decoder en TensorFlow 1.x pre-PyTorch-dominance.

10. **Una pregunta abierta para discusión en clase**: ¿por qué GPT-1, que fue eclipsado por BERT en 2018, terminó siendo la línea ganadora? Respuesta corta: porque la generación condicional es estrictamente más general que la representación bidireccional, y porque escala mejor (los detalles de por qué quedan para GPT-2 y GPT-3, pero la semilla está en GPT-1).

---

## Apéndice A: ecuaciones del paper en una sola página

**Pre-training (forward LM)**:
$$L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)$$

**Arquitectura decoder-only**:
$$
\begin{aligned}
h_0 &= U W_e + W_p \\
h_l &= \text{transformer\_block}(h_{l-1}), \quad l = 1, \dots, 12 \\
P(u) &= \text{softmax}(h_n W_e^T)
\end{aligned}
$$

**Cabeza de clasificación supervisada**:
$$P(y \mid x^1, \dots, x^m) = \text{softmax}(h_l^m W_y)$$

**Objetivo supervisado puro**:
$$L_2(\mathcal{C}) = \sum_{(x, y)} \log P(y \mid x^1, \dots, x^m)$$

**Objetivo de fine-tuning combinado**:
$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C}), \quad \lambda = 0.5$$

**Masked self-attention (causal)**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V$$
$$M_{ij} = \begin{cases} 0 & \text{si } j \leq i \\ -\infty & \text{si } j > i \end{cases}$$

---

## Apéndice B: cronología contextual 2017–2018

| Fecha | Hito |
|---|---|
| Jun 2017 | "Attention is All You Need" (Vaswani et al., NeurIPS 2017) |
| Ago 2017 | ULMFiT propuesto (publicado ACL 2018) |
| Nov 2017 | CoVe (McCann et al., NeurIPS 2017) |
| Feb 2018 | ELMo (Peters et al., NAACL 2018) |
| Mar 2018 | Liu et al. decoder-only Transformer para summarization (ICLR 2018) |
| Abr 2018 | GLUE benchmark introducido (Wang et al.) |
| **Jun 2018** | **GPT-1 (Radford et al., OpenAI technical report)** |
| Oct 2018 | BERT (Devlin et al., NAACL 2019) |
| Feb 2019 | GPT-2 (Radford et al.) |
| May 2020 | GPT-3 (Brown et al., NeurIPS 2020) |
| Mar 2022 | InstructGPT (Ouyang et al., NeurIPS 2022) |
| Nov 2022 | ChatGPT release |
| Mar 2023 | GPT-4 |

GPT-1 ocupa el lugar exacto entre el Transformer original (que era seq2seq para traducción) y BERT (que estableció el paradigma de pre-training masivo en la comunidad). Es la **prueba de concepto** de que el Transformer decoder solo, con LM autoregresivo, puede transferirse a tareas NLU diversas con cambios mínimos.

---

## Referencias clave dentro del paper (numeración del paper)

- [21] Howard & Ruder, ULMFiT, ACL 2018.
- [34] Liu et al., decoder-only Transformer para summarization, ICLR 2018.
- [38] McCann et al., CoVe, NeurIPS 2017.
- [42] Pennington et al., GloVe, EMNLP 2014.
- [44] Peters et al., ELMo, NAACL 2018.
- [52] Rocktäschel et al., reasoning about entailment con neural attention, 2015.
- [53] Sennrich et al., BPE, 2015.
- [62] Vaswani et al., Attention is All You Need, NeurIPS 2017.
- [64] Wang et al., GLUE, 2018.
- [71] Zhu et al., BookCorpus, ICCV 2015.
