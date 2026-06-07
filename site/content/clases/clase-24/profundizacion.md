---
title: "Profundizacion - Question Answering"
weight: 20
math: true
---

> Tratamiento matemático de los modelos centrales de la clase. Cinco partes: (I) la red neuronal genérica para MRC, (II) el Stanford Attentive Reader y la atención bilineal, (III) la Attention Flow Layer de BiDAF, (IV) el span prediction de BERT con su decodificación óptima, y (V) las métricas de evaluación y el entrenamiento contrastivo de DPR. Cada parte deriva las fórmulas que la [teoría](/clases/clase-24/teoria) presenta de forma conceptual.

---

## Parte I — La red neuronal genérica para MRC

### La plantilla común

Casi todos los modelos de Machine Reading Comprehension instancian la misma plantilla probabilística. Dado un contexto $c = (c_1, \dots, c_n)$ y una pregunta $q = (q_1, \dots, q_m)$, se modela la distribución sobre la respuesta $a$ como:

$$p(a \mid c, q) = \operatorname{softmax}\big(W(a)\, g(c, q)\big), \qquad a \in V$$

donde:

- $g(c, q) \in \mathbb{R}^d$ es una **representación conjunta** del par contexto-pregunta,
- $W(a)$ es una proyección que puntúa cada respuesta candidata $a$ del espacio de respuestas $V$,
- la softmax normaliza esos puntajes en una distribución de probabilidad.

La forma del espacio $V$ define el **sub-tipo de MRC**:

| Formulación | Espacio de respuestas $V$ | Salida |
|---|---|---|
| **Cloze** | el vocabulario o las entidades del pasaje | una palabra/entidad |
| **Multiple choice** | el conjunto fijo de opciones | índice de la opción |
| **Span extraction** | todos los pares $(i, j)$ con $i \le j$ del pasaje | índices start/end |
| **Generative** | $V^*$ (secuencias) | secuencia generada autoregresivamente |

### Las tres operaciones

La función $g(c, q)$ se descompone en tres operaciones que todos los modelos comparten, aunque las implementen distinto:

1. **Encode.** Mapear $c$ y $q$ a representaciones contextuales:
$$\mathbf{H}^c = \text{Encoder}_c(c) \in \mathbb{R}^{n \times d}, \qquad \mathbf{H}^q = \text{Encoder}_q(q) \in \mathbb{R}^{m \times d}$$
El encoder fue un Bi-LSTM (Stanford AR, BiDAF) y luego un Transformer (BERT).

2. **Interact / combine.** Fusionar ambas representaciones, típicamente con **atención**, para obtener una representación del contexto **consciente de la pregunta** (*query-aware*):
$$\mathbf{G} = \text{Attention}(\mathbf{H}^c, \mathbf{H}^q)$$

3. **Predict.** Producir la respuesta con un clasificador, una atención de salida o un decoder generativo.

El resto de esta profundización detalla cómo tres modelos —Stanford AR, BiDAF y BERT— especializan los pasos 2 y 3.

---

## Parte II — Stanford Attentive Reader y la atención bilineal

### Encoding

La pregunta se codifica con un Bi-LSTM y se resume en un único vector $q$ concatenando los estados finales de ambas direcciones:

$$q = [\overrightarrow{h}_m \,;\, \overleftarrow{h}_1] \in \mathbb{R}^{2h}$$

El pasaje se codifica con otro Bi-LSTM, produciendo un vector contextual por token:

$$\tilde{p}_i = [\overrightarrow{h}_i \,;\, \overleftarrow{h}_i] \in \mathbb{R}^{2h}, \qquad i = 1, \dots, n$$

### La atención bilineal

El núcleo del modelo es la **atención bilineal** (Chen et al., 2016), que mide la relevancia de cada token del pasaje para la pregunta:

$$\alpha_i = \operatorname*{softmax}_i\big(q^{\top} W_s\, \tilde{p}_i\big)$$

La matriz aprendida $W_s \in \mathbb{R}^{2h \times 2h}$ es la pieza clave. Sin ella tendríamos un simple producto punto $q^\top \tilde{p}_i$, que asume que pregunta y pasaje viven en el mismo espacio y son directamente comparables. La forma bilineal $q^\top W_s \tilde{p}_i$ aprende una **transformación** que alinea ambos espacios antes de comparar — es estrictamente más expresiva.

Desarrollando, $q^\top W_s \tilde{p}_i = (W_s^\top q)^\top \tilde{p}_i$, es decir: se proyecta la pregunta a $W_s^\top q$ y luego se hace producto punto con cada token del pasaje. La softmax normaliza sobre los $n$ tokens, dando una distribución $\alpha \in \Delta^{n-1}$.

### Salida y predicción

El vector de salida es la combinación convexa de los tokens del pasaje según la atención:

$$o = \sum_{i=1}^{n} \alpha_i\, \tilde{p}_i \in \mathbb{R}^{2h}$$

Para la tarea cloze de CNN/Daily Mail, la respuesta es una **entidad** presente en el pasaje. Se predice eligiendo la entidad cuyo embedding mejor alinea con $o$:

$$a = \arg\max_{a \in p \cap E} W_a^{\top} o$$

donde $p \cap E$ es el conjunto de entidades candidatas que aparecen en el pasaje y $W_a$ es la matriz de embeddings de respuesta. Restringir el argmax a $p \cap E$ (y no a todo el vocabulario) incorpora la restricción dura de la tarea: la respuesta **debe** ser una entidad del pasaje.

### Por qué supera al Attentive Reader de Hermann

El Attentive Reader original (Hermann et al., 2015) usaba una atención aditiva (estilo Bahdanau) y combinaba pregunta y documento con una no linealidad adicional antes de predecir:

$$m_i = \tanh\big(W_1 \tilde{p}_i + W_2 q\big), \quad \alpha_i \propto \exp(w^\top m_i), \quad r = \sum_i \alpha_i \tilde{p}_i, \quad g = \tanh\big(W_3 [r; q]\big)$$

Chen et al. (2016) hicieron tres simplificaciones que, contra la intuición, **mejoraron** el desempeño:

1. Reemplazaron la atención aditiva por la **bilineal** $q^\top W_s \tilde{p}_i$ — más directa y expresiva para comparar dos vectores.
2. Usaron $o = \sum_i \alpha_i \tilde{p}_i$ **directamente** para predecir, sin la capa $g = \tanh(W_3[r;q])$ extra.
3. Predijeron sobre el conjunto de entidades del pasaje en lugar de todo el vocabulario.

La lección metodológica: en MRC, **más capas no equivale a mejor**; una atención bien diseñada captura la interacción pregunta-pasaje sin maquinaria adicional.

---

## Parte III — La Attention Flow Layer de BiDAF

### Notación

Sea $\mathbf{H} \in \mathbb{R}^{d \times T}$ la matriz de representaciones contextuales del **contexto** ($T$ tokens, columnas $h_t$) y $\mathbf{U} \in \mathbb{R}^{d \times J}$ la de la **pregunta** ($J$ tokens, columnas $u_j$), ambas producidas por el Bi-LSTM de la capa de contextualización.

### La matriz de similitud compartida

BiDAF computa una **matriz de similitud** $S \in \mathbb{R}^{T \times J}$ entre cada token del contexto y cada token de la pregunta:

$$S_{tj} = \alpha(h_t, u_j) = w^{\top}_{(S)}\,[\,h_t \,;\, u_j \,;\, h_t \circ u_j\,]$$

donde $\circ$ es el producto elemento a elemento (Hadamard), $[\,;\,]$ la concatenación por filas, y $w_{(S)} \in \mathbb{R}^{3d}$ un vector de pesos aprendido. La inclusión del término $h_t \circ u_j$ permite capturar interacciones multiplicativas (similitud por dimensión), no solo aditivas. Esta **única** matriz $S$ alimenta las dos direcciones de atención.

### Context-to-Query (C2Q)

¿Qué palabras de la pregunta son más relevantes para cada palabra del contexto? Para cada fila $t$ de $S$ se normaliza con softmax y se promedian los vectores de la pregunta:

$$a_t = \operatorname{softmax}(S_{t:}) \in \mathbb{R}^{J}, \qquad \tilde{U}_{:t} = \sum_{j=1}^{J} (a_t)_j\, u_j$$

El resultado $\tilde{U} \in \mathbb{R}^{d \times T}$ contiene, para cada token del contexto, un resumen *query-aware* de la pregunta.

### Query-to-Context (Q2C)

¿Qué palabras del contexto tienen la mayor similitud con **alguna** palabra de la pregunta? Es decir, ¿qué tokens del contexto son críticos para responder? Se toma el máximo por columna de $S$ y se normaliza sobre los tokens del contexto:

$$b = \operatorname{softmax}\big(\max_{\text{col}}(S)\big) \in \mathbb{R}^{T}, \qquad \tilde{h} = \sum_{t=1}^{T} b_t\, h_t \in \mathbb{R}^{d}$$

El vector $\tilde{h}$ (un único vector que resume el contexto relevante) se replica $T$ veces para formar $\tilde{H} \in \mathbb{R}^{d \times T}$.

### El vector combinado (megamerge)

Para cada token del contexto se forma un vector que combina su representación original, su contexto de pregunta (C2Q) y el contexto crítico (Q2C):

$$G_{:t} = \beta\big(h_t,\, \tilde{U}_{:t},\, \tilde{H}_{:t}\big) = \big[\, h_t \,;\, \tilde{U}_{:t} \,;\, h_t \circ \tilde{U}_{:t} \,;\, h_t \circ \tilde{H}_{:t} \,\big] \in \mathbb{R}^{4d}$$

Aquí está la idea de **attention-flow**: en vez de colapsar el contexto en un vector fijo, cada token $t$ obtiene su propio $G_{:t}$ de dimensión $4d$ que **fluye** a la Modeling Layer. No hay resumen prematuro; la información por token se preserva.

### Modeling y Output Layers

La **Modeling Layer** pasa $\mathbf{G}$ por dos Bi-LSTM, produciendo $\mathbf{M} \in \mathbb{R}^{2d \times T}$, que captura interacciones entre tokens del contexto condicionadas a la pregunta.

La **Output Layer** predice el span con dos distribuciones sobre las $T$ posiciones:

$$p^{\text{start}} = \operatorname{softmax}\big(w^{\top}_{(p^1)}[\mathbf{G}; \mathbf{M}]\big), \qquad p^{\text{end}} = \operatorname{softmax}\big(w^{\top}_{(p^2)}[\mathbf{G}; \mathbf{M}^2]\big)$$

donde $\mathbf{M}^2$ es la salida de un Bi-LSTM adicional aplicado a $\mathbf{M}$. La pérdida de entrenamiento es la suma de las log-verosimilitudes negativas de los índices verdaderos de start ($y^1$) y end ($y^2$):

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\Big[\log p^{\text{start}}_{y^1_i} + \log p^{\text{end}}_{y^2_i}\Big]$$

---

## Parte IV — Span prediction en BERT

### Input: pregunta y contexto juntos

BERT procesa la pregunta y el contexto en una **única secuencia**, aprovechando los **segment embeddings**:

$$\texttt{[CLS]}\; q_1 \dots q_m \;\texttt{[SEP]}\; c_1 \dots c_n \;\texttt{[SEP]}$$

Cada token recibe la suma de tres embeddings: token, posición y **segmento** ($E_A$ para la pregunta, $E_B$ para el contexto). El self-attention permite que cada token del contexto atienda a todos los tokens de la pregunta y viceversa, en cada capa — una interacción mucho más rica que la atención de una sola capa de los attentive readers.

BERT produce un vector contextual $T_i \in \mathbb{R}^{H}$ por token (con $H = 768$ en BERT-base, $1024$ en BERT-large).

### Output: dos vectores, una predicción por posición

Se introducen **dos** vectores aprendidos: un **start vector** $S \in \mathbb{R}^{H}$ y un **end vector** $E \in \mathbb{R}^{H}$. La probabilidad de que el token $i$ sea el inicio de la respuesta es:

$$P^{\text{start}}_i = \frac{\exp(S^{\top} T_i)}{\sum_{j} \exp(S^{\top} T_j)}$$

y análogamente para el end con $E$:

$$P^{\text{end}}_i = \frac{\exp(E^{\top} T_i)}{\sum_{j} \exp(E^{\top} T_j)}$$

{{< concept-alert type="clave" >}}
El detalle que la clase enfatiza: el **mismo** vector $S$ (de longitud $H$) se aplica a **todas** las posiciones para el start, y el **mismo** $E$ para el end. Es un clasificador compartido sobre tokens, no uno por posición — por eso BERT-for-QA añade apenas $2H$ parámetros sobre el modelo preentrenado.
{{< /concept-alert >}}

### Loss

El entrenamiento minimiza la suma de las log-verosimilitudes negativas de las posiciones verdaderas de start ($s^*$) y end ($e^*$):

$$\mathcal{L} = -\log P^{\text{start}}_{s^*} - \log P^{\text{end}}_{e^*}$$

### Decodificación del span óptimo

En inferencia, no basta tomar $\arg\max$ de start y de end por separado: podrían dar un span inválido ($\text{end} < \text{start}$). Se busca el par $(i, j)$ que maximiza el puntaje conjunto sujeto a la restricción de orden:

$$(\hat{i}, \hat{j}) = \arg\max_{i \le j \le i + L_{\max}} \big(S^{\top} T_i + E^{\top} T_j\big)$$

donde $L_{\max}$ acota la longitud máxima del span (p. ej. 30 tokens). El puntaje del span es la suma de logits (equivalente a maximizar $P^{\text{start}}_i \cdot P^{\text{end}}_j$). Para SQuAD 2.0 con preguntas sin respuesta, se compara el mejor span con el puntaje del span nulo $S^\top T_{\texttt{[CLS]}} + E^\top T_{\texttt{[CLS]}}$ más un umbral $\tau$; si el span nulo gana, el modelo **se abstiene**.

---

## Parte V — Métricas y entrenamiento contrastivo

### Token-level F1, paso a paso

El F1 de SQuAD trata la predicción y la respuesta dorada como **bolsas de tokens**. Sea $\text{pred}$ el conjunto de tokens de la predicción y $\text{gold}$ el de la respuesta dorada, y sea $|\text{pred} \cap \text{gold}|$ el número de tokens compartidos (con multiplicidad). Entonces:

$$\text{precision} = \frac{|\text{pred} \cap \text{gold}|}{|\text{pred}|}, \qquad \text{recall} = \frac{|\text{pred} \cap \text{gold}|}{|\text{gold}|}, \qquad F_1 = 2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}$$

**Ejemplo.** Gold = "through contact with Persian traders" (5 tokens), predicción = "contact with Persian traders" (4 tokens). Tokens compartidos: {contact, with, Persian, traders} = 4.

$$\text{precision} = \tfrac{4}{4} = 1.0, \quad \text{recall} = \tfrac{4}{5} = 0.8, \quad F_1 = 2\cdot\frac{1.0 \times 0.8}{1.0 + 0.8} = \frac{1.6}{1.8} \approx 0.889$$

EM aquí daría 0 (no es coincidencia exacta), pero F1 ≈ 0.889 reconoce que la respuesta es casi perfecta. Por eso ambas métricas se reportan juntas. Cuando hay varias respuestas doradas, se toma el **máximo** F1 (y EM) sobre las referencias y se promedia sobre el dataset.

### Normalización

Antes de comparar, SQuAD aplica una normalización estándar: minúsculas, eliminación de puntuación, eliminación de los artículos `a`/`an`/`the`, y colapso de espacios en blanco. Así "The Persian Traders" y "persian traders" se consideran equivalentes. Sin esta normalización, EM penalizaría diferencias triviales de superficie.

### Mean Reciprocal Rank

Para el ranking de candidatos (o de pasajes en retrieval), si $\text{rank}_i$ es la posición del primer ítem relevante para la consulta $i$:

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

**Ejemplo.** Tres consultas con el primer acierto en posiciones 1, 3 y 2 respectivamente: $\text{MRR} = \tfrac{1}{3}(1 + \tfrac{1}{3} + \tfrac{1}{2}) = \tfrac{1}{3}\cdot\tfrac{11}{6} \approx 0.611$. MRR penaliza fuertemente que la respuesta correcta aparezca tarde en el ranking.

### Entrenamiento contrastivo de DPR

El passage retrieval moderno (la etapa 2 del pipeline IR-based) usa un **bi-encoder**: dos encoders BERT $E_Q$ y $E_P$ que mapean pregunta y pasaje a vectores, con similitud por producto interno:

$$\text{sim}(q, p) = E_Q(q)^{\top} E_P(p)$$

DPR (Karpukhin et al., 2020) se entrena con una pérdida contrastiva: para cada pregunta $q_i$ con su pasaje positivo $p_i^+$ y un conjunto de negativos $\{p_{i,j}^-\}$, se maximiza la probabilidad del positivo:

$$\mathcal{L}(q_i, p_i^+, p_{i,1}^-, \dots, p_{i,k}^-) = -\log \frac{\exp\big(\text{sim}(q_i, p_i^+)\big)}{\exp\big(\text{sim}(q_i, p_i^+)\big) + \sum_{j=1}^{k} \exp\big(\text{sim}(q_i, p_{i,j}^-)\big)}$$

El truco de eficiencia son los **in-batch negatives**: dentro de un batch de $B$ pares $(q_i, p_i^+)$, el positivo de una pregunta sirve de negativo para las otras $B-1$. Con una matriz de similitud $B \times B$ se obtienen $B$ ejemplos de entrenamiento, cada uno con $B-1$ negativos "gratis". A esto se suma un **hard negative** de BM25 (un pasaje léxicamente similar pero incorrecto) que enseña al modelo a distinguir relevancia semántica de mero solapamiento de palabras. La recuperación se hace con **Maximum Inner Product Search** (MIPS) sobre el índice de vectores (FAISS). El desarrollo completo está en el [fundamento de dense retrieval](/fundamentos/dense-retrieval).

---

## Cierre

Las cinco partes muestran una progresión matemática coherente. La plantilla genérica (Parte I) se especializa en la atención bilineal del Stanford AR (Parte II), que BiDAF generaliza a atención bidireccional con flujo sin resumen (Parte III), que BERT reemplaza por self-attention preentrenada con un clasificador de span de apenas $2H$ parámetros (Parte IV). Las métricas (Parte V) cierran el ciclo: definen qué significa "responder bien" y, en el caso de DPR, cómo entrenar el retriever que alimenta todo el pipeline. La constante a través de todo es la **atención** como mecanismo de interacción pregunta-contexto — primero como capa explícita, después disuelta en las múltiples capas del Transformer.

Para el recorrido conceptual, ver la [Teoría](/clases/clase-24/teoria). Para los conceptos transversales: [Question Answering](/fundamentos/question-answering), [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension), [Métricas de QA](/fundamentos/qa-evaluation-metrics) y [Dense Retrieval](/fundamentos/dense-retrieval).
