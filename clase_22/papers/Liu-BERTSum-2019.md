---
title: "BERTSum — Fine-tune BERT for Extractive Summarization (Liu, 2019)"
slug: bertsum-liu-2019
date: 2019-09-05
authors:
  - Yang Liu
venue: "arXiv:1903.10318 / EMNLP 2019 (versión extendida con Lapata)"
year: 2019
tags:
  - summarization
  - extractive
  - bert
  - transformer
  - cnn-dailymail
  - rouge
course: "IA UC — Clase 22 (Modelos de generación: resumen automático)"
status: completed
---

# BERTSum — Fine-tune BERT for Extractive Summarization

> Yang Liu (University of Edinburgh). *Fine-tune BERT for Extractive Summarization.* arXiv:1903.10318v2, septiembre 2019. Versión extendida posterior con Mirella Lapata: *Text Summarization with Pretrained Encoders*, EMNLP 2019 (arXiv:1908.08345).

---

## 1. Resumen ejecutivo

BERTSum es la primera aplicación exitosa de BERT al problema de resumen automático extractivo a nivel de documento. El paper observa una tensión arquitectónica fundamental: BERT fue diseñado para producir representaciones contextuales **a nivel de token** sobre **una o dos** oraciones (sentence A / sentence B), pero el resumen extractivo requiere representaciones **a nivel de oración** sobre **m oraciones** (típicamente $m \approx 30$–$50$ en CNN/DM).

La solución de Liu es elegante por su simplicidad:

1. Insertar un token `[CLS]` **antes de cada oración** del documento (no solo al inicio).
2. Insertar `[SEP]` después de cada oración.
3. Introducir **interval segment embeddings** $E_A, E_B$ que alternan por paridad de la oración ($E_A$ para oraciones impares, $E_B$ para pares).
4. Tomar el vector de cada `[CLS]` como la representación de la oración correspondiente.
5. Apilar una **capa de resumen** sobre estos vectores `[CLS]` — la mejor variante es un **inter-sentence Transformer de 2 capas**.
6. Clasificación binaria por oración con loss BCE contra un **oracle** construido por algoritmo greedy que maximiza ROUGE-2 frente al resumen humano.
7. En inferencia, ranquear oraciones por score y aplicar **trigram blocking** para evitar redundancia.

Resultados sobre CNN/DailyMail: ROUGE-1 = 43.25, ROUGE-2 = 20.24, ROUGE-L = 39.63 — superando al estado del arte previo (NeuSum, REFRESH, Pointer-Generator, DCA) por ~1.65 puntos en ROUGE-L. En NYT50 BERTSum alcanza R-1 = 46.66, R-2 = 26.35, R-L = 42.62.

El paper es citado explícitamente como referencia del **Extractive Model** en el slide 57 del Appendix de la clase 22 del curso IA UC. Toda la sección de slides 17–32 que el curso dedica a extracción de oraciones implementa las ideas centrales de BERTSum.

---

## 2. Contexto histórico — la era pre-BERTSum

### 2.1 El problema del resumen automático

El resumen automático de un solo documento (single-document summarization, SDS) es una tarea con dos paradigmas clásicos:

- **Extractivo**: seleccionar y concatenar oraciones (o spans) del documento original. La salida es un subconjunto del input.
- **Abstractivo**: generar texto nuevo que puede incluir palabras, paráfrasis y reorganizaciones no presentes literalmente en la fuente.

Cada paradigma tiene sus virtudes. Extractivo es **fiel por construcción** (no inventa hechos, no alucina), simple de evaluar y robusto a errores de generación; pero es **rígido**, no puede reescribir y muchas veces produce resúmenes telegráficos o redundantes. Abstractivo es **fluido** y compacto, pero hereda los problemas de los modelos generativos: alucinaciones, repeticiones, dificultad para mantener coherencia factual.

### 2.2 El estado del arte antes de BERT

Para entender por qué BERTSum es una contribución, conviene mirar el panorama de 2018–2019:

**Lado extractivo** (etiquetar oraciones con $y_i \in \{0,1\}$):

- **SummaRuNNer** (Nallapati et al., 2017): RNN jerárquica (word-level + sentence-level) que produce un score por oración. Loss BCE.
- **REFRESH** (Narayan et al., 2018): entrenado con **reinforcement learning** maximizando ROUGE directamente. Aprende a rankear oraciones globalmente.
- **NeuSum** (Zhou et al., 2018): puntúa y selecciona conjuntamente — el modelo decide secuencialmente qué oración agregar dado lo ya seleccionado.
- **BanditSum** (Dong et al., 2018): formaliza la selección como un contextual bandit.
- **Latent extractive** (Zhang et al., 2018): variables latentes sobre la selección.

Estos modelos compartían dos limitaciones: usaban **embeddings estáticos** (GloVe o word2vec, no contextuales) y arquitecturas relativamente shallow (LSTMs bidireccionales). El techo de ROUGE-L en CNN/DM rondaba **37.7–38.0**.

**Lado abstractivo** (encoder-decoder):

- **Pointer-Generator Network** (See, Liu & Manning, 2017): encoder-decoder con copy mechanism (que aprende cuándo copiar palabras de la fuente vs. generar del vocabulario) y coverage loss (que penaliza repeticiones). ROUGE-L ~38.0 sobre CNN/DM. **Marcó la línea base abstractiva** durante años.
- **DCA — Deep Communicating Agents** (Celikyilmaz et al., 2018): múltiples agentes representando diferentes regiones del documento, con atención jerárquica.
- **Deep Reinforced** (Paulus, Xiong & Socher, 2018): combina cross-entropy con policy gradient sobre ROUGE.

El techo de los abstractivos rondaba ROUGE-L ~38.0 también — el paradigma extractivo y el abstractivo estaban prácticamente empatados.

### 2.3 Por qué BERT directo no funciona para summarization

Cuando Devlin et al. publicaron BERT (octubre 2018), la comunidad NLP rápidamente lo aplicó a un zoo de tareas:

- Sentence-pair classification (NLI, MRPC, QQP) — natural: input = `[CLS] sent_A [SEP] sent_B [SEP]`, output = clase del `[CLS]`.
- Single-sentence classification (SST-2, CoLA) — directo: `[CLS] sent [SEP]`, output del `[CLS]`.
- Token classification (NER) — directo: clasifica cada token.
- Question Answering (SQuAD) — span prediction: dos heads (start, end) sobre tokens.

Pero summarization es una tarea con **m oraciones** y necesita un vector por cada una. BERT vanilla tiene **dos problemas estructurales**:

1. **Output a nivel de token, no de oración.** Si concatenas todo el documento como `[CLS] doc [SEP]`, obtienes un único vector `[CLS]` que comprime el documento entero — no $m$ vectores que representen las $m$ oraciones individualmente. Y los vectores token a token no son embeddings de oración utilizables.

2. **Segment embeddings binarios.** BERT solo conoce dos segmentos: $E_A$ y $E_B$. Está diseñado para distinguir sentence-A de sentence-B (pairs de NLI, MRPC). No tiene mecanismo nativo para distinguir oración 1, 2, 3, ..., m.

3. **Límite de 512 tokens.** BERT-base tiene positional embeddings absolutos hasta posición 512. Documentos noticiosos de CNN/DM pueden tener 700–1500 tokens. Hay truncamiento forzoso.

Liu (2019) ataca los puntos (1) y (2) con dos cambios mínimos y un re-uso elegante del token `[CLS]`. El punto (3) se acepta como limitación.

---

## 3. Idea central — BERT modificado para sentence-level encoding

### 3.1 Insertar `[CLS]` antes de cada oración

La intuición clave: en BERT vanilla, el primer `[CLS]` aprende durante pretraining a **agregar features** de la secuencia que tiene a su derecha (vía la objective NSP — Next Sentence Prediction — y los gradientes que reciben los tokens iniciales). Si insertamos `[CLS]` **antes de cada oración**, cada uno de esos `[CLS]` puede agregar features de la oración que le sigue, especialmente después de fine-tuning con un objetivo apropiado.

La construcción del input es:

```
[CLS] sent_1_tokens [SEP] [CLS] sent_2_tokens [SEP] [CLS] sent_3_tokens [SEP] ... [CLS] sent_m_tokens [SEP]
```

Cada oración queda flanqueada por `[CLS]` (a la izquierda) y `[SEP]` (a la derecha). La presencia de múltiples `[CLS]` es una desviación importante respecto a BERT vanilla — en el preentrenamiento original, `[CLS]` aparece **una sola vez** por secuencia.

### 3.2 Interval segment embeddings

Para que el modelo pueda distinguir oraciones consecutivas dentro del documento (no solo "antes/después del `[SEP]`"), Liu introduce los **interval segment embeddings**: a la oración $i$ se le asigna

$$
\text{segment}(\text{sent}_i) = \begin{cases} E_A & \text{si } i \text{ es impar} \\ E_B & \text{si } i \text{ es par} \end{cases}
$$

De este modo, para un documento de cinco oraciones $[s_1, s_2, s_3, s_4, s_5]$ los embeddings de segmento aplicados a sus tokens son $[E_A, E_B, E_A, E_B, E_A]$. Cada token de la oración $s_i$ recibe el embedding $E_A$ o $E_B$ según la paridad de $i$.

Esto reusa la arquitectura existente de BERT (que solo tiene dos segmentos posibles) sin agregar parámetros. La señal de paridad no identifica unívocamente cada oración (no es un identificador), pero **alterna**: el modelo sabe que `[CLS]` con segmento $E_A$ y `[CLS]` con segmento $E_B$ corresponden a oraciones distintas y consecutivas.

Una alternativa más expresiva sería usar embeddings absolutos por oración ($E_1, E_2, ..., E_m$), pero requeriría inicializar parámetros nuevos y entrenarlos desde cero, perdiendo el beneficio del pretraining. La alternancia $E_A/E_B$ es un compromiso pragmático.

### 3.3 Token embeddings y position embeddings

El input total a BERT en BERTSum es la suma de tres embeddings (igual que BERT vanilla):

$$
\text{input}_j = \text{TokenEmb}(t_j) + \text{IntervalSegEmb}(t_j) + \text{PosEmb}(j)
$$

donde:

- $\text{TokenEmb}$: embedding del token WordPiece. Heredado de BERT, fine-tuned.
- $\text{IntervalSegEmb}$: $E_A$ o $E_B$ según paridad de la oración que contiene a $t_j$. Reusa los dos segment embeddings de BERT.
- $\text{PosEmb}(j)$: position embedding absoluto en la posición $j$. Hasta 512 en BERT-base.

Para documentos que exceden 512 tokens, BERTSum **trunca** — esta es una limitación reconocida que sucesores como Longformer (Beltagy 2020) y BigBird (Zaheer 2020) atacarán explícitamente.

### 3.4 El vector $T_i$ — embedding de oración

Una vez que el documento entero pasa por las 12 capas de BERT-base, se recupera el vector de la última capa correspondiente al token `[CLS]` que precede a la oración $i$:

$$
T_i = \text{BERT}(\text{document})[\text{position of i-th [CLS]}]
$$

$T_i \in \mathbb{R}^{768}$ (hidden size de BERT-base) es el **embedding de la oración $i$**. Es contextual — depende de todas las demás oraciones del documento por la atención global de BERT — y es lo que se feedea a la capa de resumen.

---

## 4. Arquitectura completa — Summarization Layers

Encima de los embeddings de oración $\{T_1, T_2, ..., T_m\}$, BERTSum apila una **summarization layer** cuya tarea es producir un score $\hat{Y}_i \in [0, 1]$ por oración. Liu explora tres variantes.

### 4.1 Variante A — Simple Classifier (linear + sigmoid)

La opción mínima:

$$
\hat{Y}_i = \sigma(W_o T_i + b_o)
$$

donde $W_o \in \mathbb{R}^{1 \times 768}$, $b_o \in \mathbb{R}$, $\sigma$ es sigmoid. Cada $T_i$ se proyecta independientemente — no hay interacción entre las predicciones de diferentes oraciones más allá de lo que BERT ya capturó internamente.

Resultado CNN/DM: R-1 = 43.23, R-2 = 20.22, R-L = 39.60. Sorprendentemente fuerte — el grueso del gain viene del pretraining de BERT, no de la capa de salida.

### 4.2 Variante B — Inter-sentence Transformer (la ganadora)

La intuición: aunque BERT ya tiene atención global, sus capas fueron entrenadas con objetivos a nivel de **token** (MLM y NSP). Una capa Transformer adicional, entrenada **a nivel de oración** durante el fine-tuning, puede refinar la representación para que sea más útil al clasificador.

La construcción:

$$
\begin{aligned}
h^0 &= \text{PosEmb}(T) \\
\tilde{h}^{\,l} &= \text{LN}(h^{l-1} + \text{MHAtt}(h^{l-1})) \\
h^l &= \text{LN}(\tilde{h}^{\,l} + \text{FFN}(\tilde{h}^{\,l})) \\
\hat{Y}_i &= \sigma(W_o h^L_i + b_o)
\end{aligned}
$$

Notación:

- $T = [T_1, ..., T_m]$ es la matriz de embeddings de oración output de BERT, en $\mathbb{R}^{m \times 768}$.
- $\text{PosEmb}$ es un **nuevo** conjunto de positional embeddings que indican la posición **de la oración** (no del token) dentro del documento. Inicializados aleatoriamente, entrenados from scratch.
- $\text{LN}$ = layer normalization. $\text{MHAtt}$ = multi-head attention (Vaswani 2017). $\text{FFN}$ = feed-forward de dos capas.
- $L$ es la profundidad del Inter-sentence Transformer. Liu prueba $L = 1, 2, 3$ y reporta que **$L = 2$ es óptimo**.
- $h^L_i$ es la representación final de la oración $i$ tras $L$ capas. El clasificador final es de nuevo lineal + sigmoid.

Resultado CNN/DM: R-1 = 43.25, R-2 = 20.24, R-L = 39.63 — la mejor variante.

Crucialmente, las capas del Inter-sentence Transformer trabajan sobre una secuencia de **m elementos** (típicamente 20–50 oraciones), no sobre los 500+ tokens del documento. El cómputo es barato — toda la maquinaria pesada vive en BERT.

### 4.3 Variante C — LSTM

Una alternativa que explora si las RNNs aún tienen lugar dado que la evidencia mixta (Chen et al., 2018):

$$
\begin{pmatrix} F_i \\ I_i \\ O_i \\ G_i \end{pmatrix} = \text{LN}_h(W_h h_{i-1}) + \text{LN}_x(W_x T_i)
$$

$$
C_i = \sigma(F_i) \odot C_{i-1} + \sigma(I_i) \odot \tanh(G_{i-1})
$$

$$
h_i = \sigma(O_i) \odot \tanh(\text{LN}_c(C_i))
$$

$$
\hat{Y}_i = \sigma(W_o h_i + b_o)
$$

Con **per-gate layer normalization** (un LN distinto por gate) para estabilizar el entrenamiento. Es esencialmente un LSTM unidireccional con triple LN.

Resultado CNN/DM: R-1 = 43.22, R-2 = 20.17, R-L = 39.59 — peor que Transformer, comparable al Simple Classifier.

### 4.4 Síntesis arquitectónica

| Capa | R-1 | R-2 | R-L |
|------|-----|-----|-----|
| Simple Classifier | 43.23 | 20.22 | 39.60 |
| Inter-sentence Transformer (L=2) | **43.25** | **20.24** | **39.63** |
| LSTM | 43.22 | 20.17 | 39.59 |

La diferencia entre las tres variantes es **marginal** (<0.1 ROUGE). La conclusión tácita es que el pretraining de BERT hace el trabajo pesado y la summarization layer es casi cosmética. Sin embargo el Inter-sentence Transformer es estable y elegante — se convierte en la elección de referencia.

---

## 5. Inter-sentence Transformer — análisis del variant ganador

### 5.1 Por qué 2 capas son óptimas

Liu prueba $L \in \{1, 2, 3\}$. Con $L = 1$ probablemente no hay suficiente capacidad para combinar señales entre oraciones distantes. Con $L = 3$ se sobreajusta sobre la cabeza de resumen y el modelo se vuelve menos generalizable. $L = 2$ es el punto dulce, consistente con lo observado en otras arquitecturas que apilan pocas capas sobre encoders preentrenados (BERT-on-top patterns).

### 5.2 Position embeddings sin clamp

Los $\text{PosEmb}$ del Inter-sentence Transformer codifican **posición absoluta de la oración** dentro del documento ($1, 2, ..., m$). No hay clipping: cada posición tiene su embedding. Esto importa porque en CNN/DM hay un sesgo posicional fuerte — las primeras oraciones de un artículo de noticias suelen ser las más informativas (el "lead" del periodismo inverted-pyramid). El modelo puede aprender este sesgo directamente vía las position embeddings.

LEAD-3 (un baseline trivial que extrae las primeras 3 oraciones) alcanza R-L = 36.67 — solo 3 puntos por debajo de BERTSum. Una porción del "gain" de cualquier modelo extractivo neuronal sobre CNN/DM es simplemente aprender que la lead matter.

### 5.3 Capacidad de re-ranking global

A diferencia del Simple Classifier (que clasifica cada $T_i$ independientemente), el Inter-sentence Transformer permite que **cada oración vea a las otras** vía multi-head attention. Esto enable:

- **Penalizar redundancia**: si dos oraciones contienen información similar, el modelo puede aprender a darle alto score solo a una.
- **Reforzar coherencia**: una oración con anáforas vagas puede recibir score más alto si su antecedente está en el documento.
- **Modelar discurso**: el flujo retórico del artículo (introducción → desarrollo → cierre) puede informar qué oraciones son centrales.

En la práctica, sin embargo, BERTSum **no implementa modelado explícito de redundancia** durante el scoring — para eso usa trigram blocking en inferencia (sección 7.2).

---

## 6. Oracle target — cómo construir labels para resumen extractivo

### 6.1 El problema del ground truth ausente

CNN/DailyMail y NYT son datasets pensados para summarization abstractiva: cada documento viene con un resumen humano (highlights / abstract). Pero el resumen humano **no es un subconjunto de oraciones del documento** — es texto nuevo, parafraseado.

Para entrenar un clasificador binario por oración, necesitamos un label $y_i \in \{0, 1\}$ que indique si la oración debe entrar al resumen. ¿Cómo se construye ese label a partir de un resumen abstractivo?

### 6.2 Algoritmo greedy maximizando ROUGE

La solución estándar (heredada de Nallapati et al., 2017 y refinada en BERTSum) es construir un **oracle**: el subconjunto de oraciones del documento que, concatenadas, **maximiza ROUGE** contra el resumen humano.

Encontrar el subset óptimo exacto es NP-hard (es una versión de set selection con función objetivo no-monótona). El algoritmo greedy estándar es:

```
oracle = []
best_rouge = 0
while True:
    best_candidate = None
    best_new_rouge = best_rouge
    for s in document_sentences:
        if s in oracle:
            continue
        candidate = oracle + [s]
        r = ROUGE(candidate, gold_summary)
        if r > best_new_rouge:
            best_new_rouge = r
            best_candidate = s
    if best_candidate is None:
        break  # no sentence improves ROUGE
    oracle.append(best_candidate)
    best_rouge = best_new_rouge
```

Se itera agregando una oración a la vez (la que más sube ROUGE-2 sobre el conjunto actual) hasta que ninguna nueva oración mejore. Las oraciones seleccionadas reciben label $y_i = 1$, el resto $y_i = 0$.

### 6.3 Por qué el oracle sigue siendo aproximado

El oracle establece un **techo** para cualquier sistema extractivo: ningún modelo puede superarlo, por construcción. En CNN/DM el oracle alcanza R-1 = 52.59, R-2 = 31.24, R-L = 48.87 — significativamente más alto que cualquier modelo (BERTSum alcanza R-L = 39.63, dejando ~9 puntos de gap).

Este gap del oracle es informativo:

- **No es un techo absoluto** de la tarea: es el mejor resumen alcanzable **dado que solo podemos copiar oraciones literales**. Un modelo abstractivo idealmente podría superarlo (parafraseando).
- **No es el "oracle real"**: el verdadero ground truth (el resumen humano) no es alcanzable por extracción porque contiene paráfrasis.
- **El gap modelo–oracle (~9 puntos)** indica que aún hay margen para mejorar la selección extractiva — el modelo no está saturando.

### 6.4 Ruido del oracle como upper bound

Una crítica conocida: el oracle greedy puede asignar label 1 a oraciones que **no son las "más importantes"** semánticamente, simplemente porque comparten n-gramas con el resumen humano. Esto introduce ruido en el entrenamiento. MatchSum (Zhong et al. 2020), un sucesor, intenta abordar esto reformulando la tarea como matching a nivel de resumen completo.

---

## 7. Entrenamiento e inferencia

### 7.1 Detalles de entrenamiento

**Loss**:

$$
\mathcal{L} = \sum_{i=1}^{m} \text{BCE}(\hat{Y}_i, y_i^{\text{oracle}}) = -\sum_i \left[ y_i \log \hat{Y}_i + (1 - y_i) \log(1 - \hat{Y}_i) \right]
$$

Binary cross-entropy promediada sobre las oraciones. El modelo entero (BERT + summarization layer) se **fine-tune jointly**.

**Optimizador**: Adam con $\beta_1 = 0.9$, $\beta_2 = 0.999$.

**Schedule de learning rate**: el schedule de Vaswani et al. (2017) con warm-up:

$$
\text{lr} = 2 \times 10^{-3} \cdot \min\left(\text{step}^{-0.5}, \; \text{step} \cdot \text{warmup}^{-1.5}\right)
$$

Con 10000 pasos de warmup. El pico se alcanza cuando $\text{step} = \text{warmup}$, y luego decae como $\text{step}^{-0.5}$.

**Steps**: 50000 sobre 3 GPUs GTX 1080 Ti con gradient accumulation cada 2 steps, batch size efectivo ≈ 36.

**Selección de checkpoint**: cada 1000 steps se evalúa loss en val set. Se eligen los **top-3 checkpoints por loss** y se reportan los **resultados promediados** sobre el test set (model averaging).

### 7.2 Inferencia con Trigram Blocking

Una vez entrenado, BERTSum produce scores $\hat{Y}_i$ para cada oración del documento. El procedimiento de selección es:

1. **Rank** las oraciones por $\hat{Y}_i$ descendente.
2. **Trigram blocking**: iterar la lista ranqueada, agregando una oración al resumen $S$ si y solo si no comparte **ningún trigrama** con $S$.
3. **Length cap**: detenerse cuando $|S| = 3$ (para CNN/DM, ajustable por dataset).

El pseudo-código:

```
S = []
for candidate in sentences_ranked_by_score:
    if len(S) >= 3:
        break
    trigrams_S = set of trigrams in S
    trigrams_c = set of trigrams in candidate
    if trigrams_S ∩ trigrams_c == ∅:
        S.append(candidate)
return S
```

**¿Por qué trigram blocking?**

Sin él, dos oraciones casi idénticas (por ejemplo: variantes parafraseadas de la misma fact) recibirían scores parecidos y ambas entrarían al top-3 — produciendo un resumen redundante. Trigram blocking es una versión simple de **Maximal Marginal Relevance** (MMR; Carbonell & Goldstein 1998): seleccionar la siguiente oración solo si añade información nueva respecto a las ya elegidas.

La ablación (Tabla 2 del paper) muestra que sin trigram blocking:

- R-1 cae de 43.23 → 42.57 (-0.66)
- R-2 cae de 20.22 → 19.96 (-0.26)
- R-L cae de 39.60 → 39.04 (-0.56)

Trigram blocking aporta más al ROUGE que el cambio Classifier → Transformer en la capa de resumen. Es **el truco con mayor return-on-effort** del paper.

### 7.3 Costo de inferencia

Para un documento de $m$ oraciones, BERTSum hace:

1. Un forward pass de BERT-base sobre el documento entero (≤512 tokens): $O(n^2 d)$ por capa donde $n$ es la longitud en tokens, $d = 768$. Total ≈ 110M parámetros.
2. Un forward pass del Inter-sentence Transformer sobre $m$ vectores (2 capas, $d = 768$): $O(m^2 d)$.
3. Sort de $m$ scores + chequeo de trigramas $O(m^2)$ en el peor caso.

Para $m \approx 30$, esto es muy barato. El bottleneck es BERT, no las capas adicionales.

---

## 8. Experimentos y resultados

### 8.1 CNN/DailyMail (Tabla 1)

| Model | R-1 | R-2 | R-L |
|---|---|---|---|
| PGN (See 2017) | 39.53 | 17.28 | 37.98 |
| DCA (Celikyilmaz 2018) | 41.69 | 19.47 | 37.92 |
| LEAD-3 | 40.42 | 17.62 | 36.67 |
| ORACLE (greedy) | 52.59 | 31.24 | 48.87 |
| REFRESH (Narayan 2018) | 41.0 | 18.8 | 37.7 |
| NeuSum (Zhou 2018) | 41.59 | 19.01 | 37.98 |
| Transformer (random init, no BERT) | 40.90 | 18.02 | 37.17 |
| **BERTSum + Classifier** | 43.23 | 20.22 | 39.60 |
| **BERTSum + Transformer** | **43.25** | **20.24** | **39.63** |
| **BERTSum + LSTM** | 43.22 | 20.17 | 39.59 |

Observaciones:

- **Gap BERT vs no-BERT**: el Transformer entrenado from-scratch sobre summarization (40.90 / 18.02 / 37.17) tiene resultados peores que el LEAD-3 baseline (40.42 / 17.62 / 36.67) en ROUGE-1 y -L. El pretraining de BERT es **lo que mueve la aguja** — el Transformer del Inter-sentence layer por sí solo no rinde.
- **+1.65 ROUGE-L** sobre el mejor previo (NeuSum: 37.98 → BERTSum: 39.63). En la era pre-GPT, esto era una mejora muy significativa.
- **Oracle gap**: BERTSum a 39.63, oracle a 48.87. ~9 puntos de gap todavía explotables.
- **Equivalencia entre summarization layers**: el techo no está en la capa de salida sino en cuánto puede extraer BERT del documento.

### 8.2 Ablation studies (Tabla 2)

| Configuración | R-1 | R-2 | R-L |
|---|---|---|---|
| BERTSum+Classifier (full) | 43.23 | 20.22 | 39.60 |
| − interval segments | 43.21 | 20.17 | 39.57 |
| − trigram blocking | 42.57 | 19.96 | 39.04 |

- **Interval segments aportan poco** (∆R-L ≈ 0.03). Sorpresivo dada la motivación del paper — sugiere que BERT puede igualmente segmentar oraciones por las posiciones de los `[CLS]` y `[SEP]`.
- **Trigram blocking aporta mucho** (∆R-L ≈ 0.56). La redundancia es un problema real en summaries top-k naïves.

### 8.3 NYT50 (Tabla 3)

Sobre el dataset New York Times Annotated Corpus, filtrado a resúmenes con ≥50 palabras (NYT50), con evaluación **limited-length recall** (predicciones truncadas a la longitud del gold):

| Model | R-1 | R-2 | R-L |
|---|---|---|---|
| First-$k$ words | 39.58 | 20.11 | 35.78 |
| Full (Durrett 2016) | 42.2 | 24.9 | — |
| Deep Reinforced (Paulus 2018) | 42.94 | 26.02 | — |
| **BERTSum+Classifier** | **46.66** | **26.35** | **42.62** |

BERTSum sostiene el lead — +3.7 puntos R-1 sobre Durrett, +6.8 sobre First-$k$ words.

### 8.4 XSum (en la versión EMNLP extendida)

En el paper extendido Liu & Lapata (EMNLP 2019, arXiv 1908.08345) se incluye XSum (Narayan et al. 2018), un dataset de resúmenes "extreme" (1 oración por documento). Aquí el paradigma extractivo pierde por construcción — las oraciones del documento rara vez sintetizan en una sola frase la idea central. BERTSumAbs (la versión abstractiva) supera a BERTSumExt en XSum por márgenes amplios. CNN/DM y NYT permanecen como territorio extractivo.

---

## 9. BERTSumAbs — la extensión abstractiva (Liu & Lapata, EMNLP 2019)

El paper original de 2019 (Liu solo) cubre solo el setup extractivo. La versión extendida con Mirella Lapata (publicada en EMNLP 2019, arXiv 1908.08345) introduce **BERTSumAbs**, la variante abstractiva.

### 9.1 Arquitectura

- **Encoder**: BERTSum (BERT + interval segments + inter-sentence transformer encoding), reusado.
- **Decoder**: Transformer de 6 capas entrenado **from scratch** sobre el dataset de summarization.

El decoder genera el resumen token por token con cross-attention al encoder. Es la arquitectura encoder-decoder estándar de Vaswani, pero con un encoder potentísimo (BERT pretrained) y un decoder modesto (Transformer aleatorio).

### 9.2 Two-stage fine-tuning

Aquí está la innovación principal de BERTSumAbs. Si entrenas encoder y decoder con el **mismo learning rate**, el encoder se desestabiliza (BERT requiere lrs muy bajos para fine-tuning) mientras el decoder no converge (necesita lrs altos para aprender from scratch).

La solución: **dos optimizadores Adam separados**, cada uno con su propio schedule:

$$
\text{lr}_E = 2 \times 10^{-3} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}_E^{-1.5}), \quad \text{warmup}_E = 20000
$$

$$
\text{lr}_D = 0.1 \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}_D^{-1.5}), \quad \text{warmup}_D = 10000
$$

El encoder usa lrs efectivos ~$2\text{e-}5$ (típico de fine-tuning BERT) mientras el decoder usa lrs efectivos ~$1\text{e-}3$ (típico de Transformer from-scratch). Esto permite que cada componente se entrene en su "régimen" óptimo.

### 9.3 Resultados extractivo vs abstractivo

Sobre CNN/DM:

| Variante | R-1 | R-2 | R-L |
|---|---|---|---|
| BERTSumExt | 43.25 | 20.24 | 39.63 |
| BERTSumAbs | 41.72 | 19.39 | 38.76 |
| BERTSumExtAbs (two-stage: pre-train Ext, luego Abs) | 42.13 | 19.60 | 39.18 |

Insight: el **extractivo gana en R-2** porque la rigidez del oracle (que solo permite oraciones literales del documento) preserva n-gramas exactos del resumen humano (que muchas veces son citas o frases hechas).

Sobre XSum la historia se invierte — el abstractivo gana porque el dataset requiere paráfrasis.

---

## 10. Limitaciones reconocibles

### 10.1 Límite de 512 tokens

BERT-base tiene positional embeddings absolutos hasta 512. Documentos largos (artículos de NYT, papers científicos, documentos legales) se truncan, perdiendo información potencialmente relevante. Las primeras 512 tokens pueden no contener el oracle.

Sucesores como **Longformer** (Beltagy et al. 2020) y **BigBird** (Zaheer et al. 2020) introducen atención sparse para manejar 4096+ tokens. **PRIMERA** (Xiao et al. 2022) preentrena específicamente sobre documentos largos.

### 10.2 Ruido del oracle

Como mencionamos en §6, el oracle greedy no es el verdadero ground truth. Esto introduce:

- **False positives**: oraciones que comparten n-gramas con el resumen humano por coincidencia léxica, no por relevancia semántica.
- **False negatives**: oraciones semánticamente equivalentes al resumen humano pero con vocabulario diferente.

El modelo aprende a aproximar el oracle, no la "verdad". Esto se nota en métricas como factualidad o semantic similarity que ROUGE no captura.

### 10.3 Rigidez del paradigma extractivo

BERTSum **no puede**:

- Parafrasear o resumir múltiples oraciones en una.
- Resolver anáforas en el output (si una oración seleccionada contiene "él" sin antecedente local, queda colgando).
- Producir resúmenes verdaderamente concisos cuando el documento no tiene oraciones cortas e informativas.
- Sintetizar — solo seleccionar.

Esto se ve en resúmenes BERTSum sobre artículos largos: pueden ser informativos pero su flujo es entrecortado, sin las transiciones suaves que un humano insertaría.

### 10.4 Bias hacia LEAD

CNN/DailyMail tiene un sesgo posicional fuerte (lead-bias del periodismo). BERTSum, como cualquier modelo neural extractivo entrenado sobre CNN/DM, aprende este sesgo. Esto puede no generalizar a dominios sin lead-bias (literatura, papers académicos, redes sociales).

### 10.5 Métricas ROUGE como objetivo

ROUGE es un proxy ruidoso. Maximizar ROUGE-2 contra un solo resumen humano puede sobre-ajustar a expresiones particulares del anotador. Métricas alternativas (BERTScore, MoverScore, factualidad medida con NLI) no fueron evaluadas en BERTSum original.

---

## 11. Sucesores

BERTSum es la línea base **canónica** del summarization extractivo neural moderno. Sus sucesores se dividen en familias:

### 11.1 MatchSum (Zhong et al. 2020)

Reformula la selección como **summary-level matching**: en vez de scorear cada oración independientemente y luego concatenarlas, MatchSum genera candidatos de resumen (subsets de oraciones) y entrena un modelo que dice cuál candidato matchea mejor con el documento (vía cosine similarity en espacio de embeddings BERT). Resuelve el problema de optimización global que BERTSum aproxima con trigram blocking.

ROUGE en CNN/DM: R-1 = 44.41, R-2 = 20.86, R-L = 40.55. +0.92 R-L sobre BERTSum.

### 11.2 PEGASUS (Zhang et al. 2020, Google)

Cambia el **pretraining objective**: en vez de MLM (BERT) o LM causal (GPT), preentrena con **Gap Sentence Generation** (GSG) — predecir oraciones enteras enmascaradas del documento. Esto es prácticamente "pretrain para summarization". Combinado con un encoder-decoder Transformer (estilo BART), establece un nuevo SOTA abstractivo.

ROUGE CNN/DM: R-1 = 44.17, R-2 = 21.47, R-L = 41.11. Supera a BERTSum y MatchSum en R-2 sin la rigidez extractiva.

### 11.3 BART / T5 fine-tuned

**BART** (Lewis et al. 2020, Facebook) y **T5** (Raffel et al. 2020, Google) son encoder-decoders preentrenados con denoising. Fine-tune sobre CNN/DM da:

- BART: R-1 = 44.16, R-2 = 21.28, R-L = 40.90
- T5-large: R-1 = 42.50, R-2 = 20.68, R-L = 39.75

Estos modelos generan resúmenes **abstractivos** que ya no se pueden derrotar fácilmente con extractive — la frontera de paradigmas se desdibuja.

### 11.4 HiBERT (Zhang et al. 2019)

Hierarchical BERT: en vez de aplanar el documento, codifica primero a nivel de oración con BERT, luego a nivel de documento con otro Transformer encima. Más explícitamente jerárquico que el inter-sentence Transformer de BERTSum.

### 11.5 BERTSum-MMR y variantes con MMR explícito

Algunos sucesores reemplazan trigram blocking con MMR completo durante la selección, incorporando el trade-off relevancia/diversidad como un parámetro $\lambda$ aprendible.

### 11.6 LLMs zero-shot/few-shot (era 2022+)

GPT-3, ChatGPT, Claude, Llama-2/3, Mistral. Estos modelos hacen summarization en zero-shot con prompts como "summarize this article". Los resultados ROUGE no siempre son mejores que BERTSum (porque los LLMs no están explícitamente entrenados para maximizar ROUGE), pero las evaluaciones humanas tienden a preferirlos por fluidez, coherencia y reducción de redundancia.

La summarization extractiva clásica (BERTSum-style) sigue siendo relevante en escenarios con:

- Restricciones de fidelidad estricta (legal, médico, científico) donde no se permite alucinación.
- Latencia/costo crítico — BERTSum es ~100× más barato que un LLM.
- Necesidad de interpretabilidad — sabes exactamente qué oraciones se seleccionaron.

---

## 12. Conexión con la clase 22 del curso IA UC

La clase 22 (Procesamiento de Lenguaje Natural: Generación) dedica una sección entera al resumen automático. El recorrido del PDF de la clase es:

**Slides 1–16: introducción y abstractive** — definiciones, paradigmas extractivo vs abstractivo, Pointer-Generator Network (See 2017), copy mechanism, coverage.

**Slides 17–32: Extractive Model** — esta sección implementa **directamente** las ideas de BERTSum:

- Slide 17–20: Definición del problema extractivo, $y_i \in \{0, 1\}$ por oración.
- Slide 21–24: Encoder BERT con multi-`[CLS]`.
- Slide 25–28: Interval segment embeddings $E_A, E_B$ alternantes.
- Slide 29–30: Inter-sentence Transformer como capa de resumen.
- Slide 31: Loss BCE contra oracle greedy.
- Slide 32: Inferencia con trigram blocking.

**Slides 33–56: ROUGE y evaluación** — métricas, oracle como techo, baselines (LEAD-3).

**Slide 57: Appendix con referencias** — cita explícita a Liu (2019) "Fine-tune BERT for Extractive Summarization" como **fuente del extractive model**.

Por lo tanto el paper de Liu es **la referencia primaria** del módulo de resumen extractivo del curso. Comprender BERTSum es comprender cómo se hace summarization extractiva moderna con encoders preentrenados, qué decisiones de diseño importan (interval segments, trigram blocking, inter-sentence transformer), y dónde están sus límites (512 tokens, oracle ruidoso, rigidez extractiva).

### 12.1 Conexiones con otras clases

- **Clase 19 (Transformers)**: BERTSum reusa la arquitectura Transformer (encoder, multi-head attention, layer norm, FFN). El Inter-sentence Transformer es un Vanilla Transformer encoder de 2 capas operando sobre embeddings de oración en vez de tokens.
- **Clase 20 (BERT y modelos preentrenados)**: BERTSum es la primera aplicación canónica de fine-tuning de BERT a una tarea estructural (no clasificación simple). Introduce el patrón de **modificar la entrada de BERT** (multi-`[CLS]`, interval segments) para adaptarlo a tareas no contempladas en el pretraining original.
- **Clase 21 (Generación abstractiva)**: BERTSum es el contrapunto extractivo a Pointer-Generator Network. La discusión "extractive vs abstractive" del curso se sostiene en estos dos papers.
- **Lab de la clase 22** (si aplica): la implementación práctica de un modelo de resumen extractivo seguiría el playbook de BERTSum: tokenizar con WordPiece, insertar múltiples `[CLS]`, fine-tune BERT con BCE, inferir con trigram blocking.

---

## 13. Síntesis final

BERTSum es un paper "simple" en el mejor sentido: una idea clara (multi-`[CLS]` + interval segments + inter-sentence Transformer), una implementación reproducible, y resultados que mueven el state-of-the-art por márgenes claros. Su simplicidad es precisamente lo que lo vuelve **didáctico** — es el ejemplo canónico de cómo adaptar un modelo preentrenado (BERT) a una tarea estructuralmente distinta (selección a nivel de oración) sin reinventar la arquitectura.

Las decisiones de diseño relevantes que se enseñan a través de este paper:

1. **Reusar pretraining**: no entrenar from scratch. El gain mayor (de ~38 a ~40 R-L) viene de BERT, no de las capas adicionales.
2. **Modificar la entrada, no la arquitectura**: multi-`[CLS]` y interval segments aprovechan la maquinaria existente.
3. **Construir labels aproximados**: el oracle greedy permite entrenar clasificadores supervisados sobre datasets diseñados para abstractive.
4. **Distinguir scoring de selection**: el modelo puntúa oraciones independientemente, pero la selección final aplica trigram blocking para mitigar redundancia.
5. **Aceptar limitaciones honestamente**: 512 tokens, rigidez extractiva, ROUGE como proxy.

La era post-BERTSum (PEGASUS, BART, T5, LLMs) ha movido la frontera hacia el paradigma abstractivo, pero los conceptos centrales de BERTSum siguen vigentes: cualquier sistema de selección de oraciones moderno —desde RAG retrievers hasta passage rerankers en QA— usa variantes del patrón `[CLS]` por oración + capa de scoring por encima de un encoder preentrenado.

Para el alumno de la clase 22 de IA UC, el take-away es: el resumen extractivo moderno **es BERTSum**, con perfeccionamientos. El paper de Liu es la base sobre la cual se construyen tanto los sistemas extractivos que aún se usan en producción (donde fidelidad y costo importan más que fluidez) como la pedagogía del módulo de generación del curso.

---

## 14. Referencias clave del paper

- **Devlin et al. 2018** — BERT: Pre-training of Deep Bidirectional Transformers.
- **Vaswani et al. 2017** — Attention Is All You Need (Transformer).
- **See, Liu & Manning 2017** — Get to the Point: Summarization with Pointer-Generator Networks (PGN baseline).
- **Narayan, Cohen & Lapata 2018** — Ranking sentences for extractive summarization with RL (REFRESH baseline).
- **Zhou et al. 2018** — Neural document summarization by jointly learning to score and select (NeuSum baseline).
- **Nallapati, Zhai & Zhou 2017** — SummaRuNNer (baseline RNN-based extractive).
- **Carbonell & Goldstein 1998** — Maximal Marginal Relevance (origen del trigram blocking).
- **Hermann et al. 2015** — CNN/DailyMail dataset.
- **Durrett, Berg-Kirkpatrick & Klein 2016** — NYT preprocessing y baselines.
- **Ba, Kiros & Hinton 2016** — Layer Normalization.
- **Paulus, Xiong & Socher 2018** — Deep Reinforced (RL abstractive baseline).

---

> **Cita sugerida (BibTeX):**
> ```bibtex
> @article{liu2019bertsum,
>   title={Fine-tune BERT for Extractive Summarization},
>   author={Liu, Yang},
>   journal={arXiv preprint arXiv:1903.10318},
>   year={2019}
> }
> ```
>
> Versión extendida (con Lapata, EMNLP 2019):
> ```bibtex
> @inproceedings{liu-lapata-2019-text,
>   title={Text Summarization with Pretrained Encoders},
>   author={Liu, Yang and Lapata, Mirella},
>   booktitle={EMNLP-IJCNLP},
>   year={2019},
>   url={https://arxiv.org/abs/1908.08345}
> }
> ```
