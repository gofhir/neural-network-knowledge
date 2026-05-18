---
title: "Análisis exhaustivo — Deep Contextualized Word Representations (ELMo, Peters et al. 2018)"
authors: "Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer"
venue: "NAACL-HLT 2018 (Best Paper Award)"
arxiv: "1802.05365"
year: 2018
institution: "Allen Institute for AI (AI2) + Paul G. Allen School (UW)"
curso: "Diplomado IA UC — Clase 20 (Embeddings contextualizados, BERT, GPT)"
audiencia: "Roberto — análisis INTERNO de referencia (no la versión pública del site)"
---

# Deep Contextualized Word Representations — ELMo (Peters et al., 2018)

> "Each token is assigned a representation that is a function of the entire input sentence."

Esta es la frase que pasó a definir una era. Hasta inicios de 2018, los embeddings dominantes (word2vec, GloVe, FastText) eran **estáticos**: una palabra equivalía siempre al mismo vector independientemente del contexto. ELMo, junto a CoVe y ULMFiT, fue uno de los tres trabajos que rompieron ese paradigma; pero fue ELMo el que cristalizó la receta — **biLM pre-entrenado a gran escala + combinación lineal aprendida de todas las capas internas** — que se volvió el puente conceptual hacia BERT y la era Transformer.

Este documento es un análisis interno de referencia. La idea es que sirva como material de estudio profundo: cubre contexto histórico, derivaciones matemáticas, decisiones de arquitectura, resultados experimentales con análisis crítico, limitaciones, y conexión con el resto del curso.

---

## 1. Contexto histórico: dónde estaba NLP entre 2013 y 2017

### 1.1 El paradigma de embeddings estáticos

Entre 2013 y 2017 la lingüística computacional aplicada vivió la "era de los embeddings estáticos". Tres trabajos definieron el paisaje:

- **word2vec (Mikolov et al., 2013)** — Skip-gram y CBOW. Aprendía vectores densos de baja dimensión (típicamente 300) optimizando que palabras que coocurren en ventanas pequeñas tuvieran vectores cercanos. Negative sampling como aproximación al softmax sobre todo el vocabulario.
- **GloVe (Pennington et al., 2014)** — Factorización de la matriz de coocurrencias globales con ponderación logarítmica. Combinaba la información global de LSA con la geometría local de word2vec.
- **FastText (Bojanowski et al., 2017)** — Extendió word2vec a subword units (n-gramas de caracteres), permitiendo manejar palabras fuera de vocabulario (OOV) y morfología rica (turco, finés, etc.).

Estos vectores se cargaban como **lookup table congelado** (o se fine-tuneaban débilmente) al inicio de la red de la tarea. Las arquitecturas downstream típicas eran BiLSTM + CRF (para etiquetado), encoder-decoder con atención (Bahdanau 2015 para MT), BiDAF (Seo et al. 2017 para QA), ESIM (Chen et al. 2017 para entailment).

### 1.2 El problema fundamental: polisemia y dependencia de contexto

Un embedding estático asigna **un único vector** a la palabra "play":

- "Chico Ruiz made a spectacular **play** on Alusik's grounder" (sentido deportivo, sustantivo)
- "Olivia De Havilland signed to do a Broadway **play**" (sentido teatral, sustantivo)
- "I will **play** the piano" (verbo, acción musical)
- "Watch the kids **play**" (verbo, sentido lúdico)

GloVe colapsa todos esos sentidos en un solo punto del espacio. Sus vecinos más cercanos a "play" en el paper son: *playing, game, games, played, players, plays, player, Play, football, multiplayer* — una nube morfológica y deportiva, sin separación por sentido.

Esto importa porque tareas como Word Sense Disambiguation (WSD), Semantic Role Labeling (SRL), Coref y QA dependen críticamente de que la representación de una palabra capture **qué función cumple en esta oración específica**.

Antes de ELMo se intentaron varias soluciones, pero todas tenían debilidades:

| Enfoque | Año | Limitación |
|---|---|---|
| Sense embeddings (Neelakantan 2014) | 2014 | Requiere predefinir número de sentidos; pierde flexibilidad |
| context2vec (Melamud 2016) | 2016 | Solo contexto, no la palabra; un solo LSTM bidireccional |
| CoVe (McCann 2017) | 2017 | Encoder de NMT — limitado por corpus paralelo |
| TagLM (Peters 2017) | 2017 | Solo usa la capa top del biLM |

### 1.3 Los tres antecedentes directos de ELMo (2017)

**CoVe — Learned in Translation (McCann et al., 2017, NeurIPS)**. Tomaba el encoder de un modelo seq2seq de traducción inglés→alemán (Bi-LSTM de 2 capas, dim 300) y exportaba sus estados ocultos como features. Mostró ganancias sobre GloVe en SNLI, QA y clasificación. **Problema**: el corpus de entrenamiento (WMT) es chico (~7M pares paralelos) y caro de obtener; además solo se usaba la capa superior.

**TagLM (Peters et al., 2017, ACL)**. Mismos autores que ELMo, un año antes. Pre-entrenaron un biLM forward+backward sobre 1B Words y lo usaron como features adicionales para NER y chunking. **Limitación clave**: usaron solo la capa superior del biLM. ELMo es esencialmente TagLM + "usa TODAS las capas con pesos aprendidos por tarea". Esa pequeña modificación cambia el régimen de desempeño.

**ULMFiT (Howard & Ruder, 2018, ACL — concurrente)**. AWD-LSTM pre-entrenado como LM y fine-tuned end-to-end con discriminative learning rates y slanted triangular schedules. Filosofía distinta: **fine-tuning total** vs **feature extraction congelado** de ELMo. ULMFiT es, en retrospectiva, más cercano al espíritu BERT que el propio ELMo.

**Dai & Le 2015 ("Semi-supervised sequence learning")** ya había propuesto pre-entrenar LSTMs como autoencoders o LMs antes de fine-tune. Es el ancestro genético de toda esta familia.

---

## 2. La contribución central: "Deep + Contextualized + Word + Representations"

Cada palabra del título es deliberada:

- **Deep**: usa **todas las capas** internas del biLM, no solo la última. Esto es la innovación cuantitativa que diferencia ELMo de CoVe y TagLM.
- **Contextualized**: el vector de una palabra es función de toda la oración, no solo del token.
- **Word**: el output sigue siendo a nivel palabra (un vector por token), compatible con cualquier arquitectura downstream existente.
- **Representations**: features extraídas de un modelo congelado — no fine-tuning end-to-end.

La idea revolucionaria es la **combinación lineal aprendida por tarea**:

$$\mathbf{ELMo}_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task} \mathbf{h}_{k,j}^{LM}$$

donde $\mathbf{s}^{task}$ son pesos softmax-normalizados sobre las $L+1$ capas (token + $L$ capas LSTM), $\gamma^{task}$ es un escalar global que permite escalar la magnitud, y $\mathbf{h}_{k,j}^{LM}$ es la representación del token $k$ en la capa $j$ del biLM.

Por qué importa: el paper muestra mediante ablations (Tabla 2) que **cada capa codifica información distinta** — sintaxis en las capas bajas, semántica en las altas. Permitir que el modelo downstream aprenda qué mezcla quiere para su tarea concreta es lo que separa ELMo de sus predecesores.

---

## 3. Arquitectura detallada

### 3.1 Capa de entrada: Character-CNN + Highway

ELMo NO usa lookup table de palabras. Toda la información a nivel token se computa **desde caracteres**, siguiendo a Kim et al. (2015) y Józefowicz et al. (2016, "Exploring the limits of language modeling").

Pipeline por token:
1. Cada carácter se embebe en un vector de baja dimensión (typically 16-d).
2. Se aplican **2048 filtros convolucionales** con anchos $w \in \{1, 2, 3, 4, 5, 6, 7\}$ sobre la secuencia de chars.
3. **Max-pooling sobre el tiempo** por filtro → vector de dimensión $\sum_w n_w$.
4. Dos capas **Highway Network** (Srivastava et al. 2015): $\mathbf{y} = T \cdot g(W\mathbf{x}) + (1-T) \cdot \mathbf{x}$, donde $T = \sigma(W_T \mathbf{x} + b_T)$ es el "transform gate". Las Highway permiten redes muy profundas en input fijo (analogía con LSTM en el dominio feed-forward).
5. **Proyección lineal** a 512 dimensiones — el "token layer" $\mathbf{x}_k^{LM}$.

Beneficios cruciales:
- **No hay OOV**. Cualquier palabra inventada, misspell, número, símbolo o entidad rara se mapea a un vector razonable.
- **Sensibilidad morfológica**. "Walked", "walking", "walker" comparten estructura de caracteres.
- **Robustez a typos**. "machne" se mapea cerca de "machine".

Esta capa es **context-insensitive**: dos ocurrencias de "play" producen exactamente el mismo $\mathbf{x}_k^{LM}$.

### 3.2 BiLSTM apilado de 2 capas

Los autores reducen el modelo de Józefowicz et al. (CNN-BIG-LSTM) a la mitad para balancear cómputo:

- **L = 2** capas LSTM por dirección.
- **4096 unidades ocultas** por LSTM.
- **Proyección a 512** después de cada capa (siguiendo Sak et al. 2014, "projection LSTM").
- **Residual connection** desde capa 1 a capa 2.

Cada capa, en cada dirección, produce un estado oculto $\overrightarrow{\mathbf{h}}_{k,j}^{LM}$ o $\overleftarrow{\mathbf{h}}_{k,j}^{LM}$. Se concatenan por capa:

$$\mathbf{h}_{k,j}^{LM} = [\overrightarrow{\mathbf{h}}_{k,j}^{LM}; \overleftarrow{\mathbf{h}}_{k,j}^{LM}] \in \mathbb{R}^{1024}$$

Resultado: para cada token $k$, ELMo expone $2L+1 = 5$ vectores:
- $\mathbf{h}_{k,0}^{LM} = [\mathbf{x}_k^{LM}; \mathbf{x}_k^{LM}]$ (token layer duplicado para tener dimensión 1024)
- $\mathbf{h}_{k,1}^{LM}$ (BiLSTM capa 1)
- $\mathbf{h}_{k,2}^{LM}$ (BiLSTM capa 2)

### 3.3 Objetivo: Forward LM + Backward LM (no concatenados)

Aquí hay un matiz fundamental que distingue ELMo de un "BiLSTM tradicional":

Un BiLSTM clásico (e.g., Bi-LSTM tagger) concatena estados forward y backward en **cada paso** y los alimenta como entrada a la capa siguiente. Esto crea "data leakage" si se entrena como LM (la palabra a predecir está implícitamente disponible vía el backward).

ELMo entrena **dos LMs independientes** que comparten solo:
- La capa de embeddings de tokens ($\Theta_x$, los chars-CNN-Highway).
- La capa softmax de salida ($\Theta_s$).

Las LSTMs son **completamente separadas**. El objetivo es maximizar:

$$\sum_{k=1}^{N} \Big( \log p(t_k \mid t_1, \ldots, t_{k-1}; \Theta_x, \overrightarrow{\Theta}_{LSTM}, \Theta_s) + \log p(t_k \mid t_{k+1}, \ldots, t_N; \Theta_x, \overleftarrow{\Theta}_{LSTM}, \Theta_s) \Big)$$

El forward LM predice $t_k$ dado $t_{<k}$. El backward LM predice $t_k$ dado $t_{>k}$. Nunca se mezclan internamente durante el pre-training. Solo al exportar las representaciones se concatenan capa-por-capa.

Esta es lo que se llama **"shallow bidirectionality"** — y es una limitación que BERT corregirá con Masked LM, que sí permite atención bidireccional profunda.

### 3.4 La ecuación ELMo en detalle

Para cada token $k$, el biLM expone el conjunto:

$$R_k = \{\mathbf{x}_k^{LM}, \overrightarrow{\mathbf{h}}_{k,j}^{LM}, \overleftarrow{\mathbf{h}}_{k,j}^{LM} \mid j = 1, \ldots, L\} = \{\mathbf{h}_{k,j}^{LM} \mid j = 0, \ldots, L\}$$

donde por convención $\mathbf{h}_{k,0}^{LM}$ es el token layer y $\mathbf{h}_{k,j}^{LM}$ para $j \geq 1$ es la concatenación de las direcciones.

La forma más simple sería usar solo la capa superior: $E(R_k) = \mathbf{h}_{k,L}^{LM}$ (esto es lo que hacían TagLM y CoVe).

ELMo generaliza:

$$\mathbf{ELMo}_k^{task} = E(R_k; \Theta^{task}) = \gamma^{task} \sum_{j=0}^{L} s_j^{task} \mathbf{h}_{k,j}^{LM}$$

donde:
- $s_j^{task}$ son pesos **softmax-normalizados** sobre las capas: $s_j = \exp(w_j) / \sum_i \exp(w_i)$. Esto garantiza $\sum_j s_j = 1$.
- $\gamma^{task}$ es un **escalar libre** que permite reescalar la norma del vector resultante. Sin $\gamma$, el modelo no podía adaptarse a la diferencia de magnitudes entre activaciones del biLM y representaciones internas de la tarea.
- Opcionalmente, **layer normalization** (Ba et al. 2016) sobre cada $\mathbf{h}_{k,j}^{LM}$ antes de ponderar — útil cuando las distribuciones por capa son muy distintas.

Los autores notan en el supplemental (A.2) que sin $\gamma$, la opción "last-only" (que es matemáticamente equivalente a $\gamma=1, s_L=1, s_{j<L}=0$) **falla totalmente en SRL y se desempeña muy mal en SNLI**. La escala importa.

### 3.5 Cómo se usa downstream

Receta estándar:

1. **Congelar** los pesos del biLM (no se actualizan durante entrenamiento de la tarea).
2. Ejecutar el biLM una vez sobre cada oración de entrada y guardar los $2L+1$ vectores por token.
3. **Concatenar** $\mathbf{ELMo}_k^{task}$ con la representación context-independent del token de la tarea (e.g., GloVe + char-CNN del modelo base): $[\mathbf{x}_k; \mathbf{ELMo}_k^{task}]$.
4. Pasar a la arquitectura RNN/CNN de la tarea sin cambios.

Variantes que el paper explora:
- **Solo input**: ELMo concatenado en la capa de embeddings.
- **Solo output**: ELMo concatenado al output del biLSTM de la tarea (antes de la cabeza).
- **Input y output**: con pesos $\mathbf{s}^{task}$ separados para cada posición.

La Tabla 3 muestra que el lugar óptimo depende de la tarea:
- SQuAD, SNLI: mejor input+output (porque hay capas de atención posteriores que se benefician).
- SRL, Coref: mejor solo input.

Además agregan:
- **Dropout moderado** a ELMo.
- **Regularización L2** $\lambda \|\mathbf{w}\|_2^2$ sobre los pesos $\mathbf{s}$. $\lambda$ grande ($=1$) fuerza a promedio uniforme; $\lambda$ pequeño ($=0.001$) permite que los pesos diverjan. La Tabla 2 muestra que $\lambda=0.001$ suele ganar por margen pequeño.

---

## 4. Entrenamiento del biLM

- **Corpus**: One Billion Word Benchmark (Chelba et al., 2014). ~30M oraciones, ~800M tokens de noticias en inglés. Mezcla balanceada, vocabulario open.
- **Épocas**: 10.
- **Perplexity**: 39.7 (promedio forward/backward), comparado con 30.0 para el CNN-BIG-LSTM forward-only de Józefowicz (modelo el doble de grande). Es decir, ELMo sacrifica algo de PPL por eficiencia de cómputo.
- **Optimización**: Adam con clipping de gradiente, dropout 0.1 entre capas LSTM.
- **Hardware**: 3 GPUs durante 2 semanas (según el supplemental). Coste relativamente modesto comparado con BERT-Large (16 TPU × 4 días).

### 4.1 Fine-tuning del biLM en dominio

El paper introduce un detalle pragmático muy valioso: **fine-tune del biLM en el dominio target** antes de congelarlo. Una época sobre el train split de la tarea (ignorando las labels supervisadas) puede bajar la perplexity drásticamente. Tabla 7 del supplemental:

| Dataset | PPL antes | PPL después |
|---|---|---|
| SNLI | 72.1 | 16.8 |
| SQuAD context | 99.1 | 43.5 |
| SQuAD questions | 158.2 | 52.0 |
| CoNLL 2003 NER | 103.2 | 46.3 |
| SST | 131.5 | 78.6 |

Para SNLI, este fine-tune del biLM mejora la accuracy supervisada 0.6%. Para sentiment no ayuda. Es un truco específico de dominio.

---

## 5. Experimentos clave

Seis tareas, todas con caída del state-of-the-art simplemente agregando ELMo a la baseline:

| Tarea | Métrica | SOTA previo | Baseline | + ELMo | Δ absoluto | Δ relativo (err) |
|---|---|---|---|---|---|---|
| **SQuAD** (QA) | F1 | 84.4 (Liu 2017) | 81.1 | **85.8** | +4.7 | **24.9%** |
| **SNLI** (Entailment) | Acc | 88.6 (Chen 2017) | 88.0 | 88.7±0.17 | +0.7 | 5.8% |
| **SRL** (CoNLL 2012) | F1 | 81.7 (He 2017) | 81.4 | **84.6** | +3.2 | **17.2%** |
| **Coref** (CoNLL 2012) | avg F1 | 67.2 (Lee 2017) | 67.2 | 70.4 | +3.2 | 9.8% |
| **NER** (CoNLL 2003) | F1 | 91.93±0.19 | 90.15 | **92.22±0.10** | +2.06 | **21%** |
| **SST-5** (Sentiment) | Acc | 53.7 (McCann 2017) | 51.4 | 54.7±0.5 | +3.3 | 6.8% |

Reducciones relativas de error de **6-25%**. Lo notable es la consistencia: nunca empeora, siempre mejora con margen significativo.

### 5.1 Comparación directa con CoVe

ELMo > CoVe en todas las tareas comparables. Algunas direct comparisons:

- **SST-5**: BCN+CoVe = 53.7, BCN+ELMo = **54.7** (+1.0 absoluto).
- **SNLI**: ESIM+CoVe ~88, ESIM+ELMo = 88.7.
- **SQuAD**: el incremento de +1.8% que daba CoVe sobre baseline vs +4.7% de ELMo es 2.6x mejor.

Los autores atribuyen esto a tres factores: (1) corpus de pre-training 4x más grande (monolingual vs paralelo), (2) deep usage (todas las capas), (3) objetivo de LM (no MT).

### 5.2 Sample efficiency (Sección 5.4)

Resultado dramático y poco discutido: con ELMo, el modelo de SRL alcanza su accuracy máxima en **epoch 10**, comparado con epoch **486** sin ELMo. **98% de reducción en número de updates**.

Además, con ELMo el modelo logra con **1% del training set** el mismo F1 que la baseline con **10%** (Figura 1). Reducción de 10x en datos etiquetados. Esto es lo que después se llamará "few-shot transfer" y es el principal driver del valor económico del pre-training.

---

## 6. Análisis de la arquitectura: ¿qué aprende cada capa?

Esta sección (5.3 del paper) es la más teóricamente importante.

### 6.1 Word Sense Disambiguation (WSD)

Usan el biLM directamente (sin tarea supervisada) para WSD con 1-nearest neighbor sobre SemCor 3.0. Tabla 5:

| Modelo | F1 (WSD) |
|---|---|
| WordNet 1st Sense | 65.9 |
| Iacobacci 2016 (SOTA) | 70.1 |
| CoVe — capa 1 | 59.4 |
| CoVe — capa 2 | 64.7 |
| **biLM — capa 1** | 67.4 |
| **biLM — capa 2** | **69.0** |

La capa superior del biLM (69.0) supera a la capa inferior (67.4) para semántica. Casi alcanza SOTA específico **sin entrenamiento supervisado**.

### 6.2 POS tagging (sintaxis)

Tabla 6, mismo setup pero ahora prediciendo POS:

| Modelo | Acc |
|---|---|
| Ling 2015 (SOTA) | 97.8 |
| CoVe — capa 1 | 93.3 |
| CoVe — capa 2 | 92.8 |
| **biLM — capa 1** | **97.3** |
| **biLM — capa 2** | 96.8 |

Aquí se invierte: la capa **inferior** (97.3) supera la superior (96.8) para sintaxis básica.

### 6.3 Implicación teórica

Este patrón confirma una hipótesis recurrente en NLP neural:

- **Capas bajas** codifican información **léxica y sintáctica** (POS, morfología, dependencias locales).
- **Capas altas** codifican información **semántica** (sentido, rol, co-referencia).

Esto es coherente con:
- Belinkov et al. 2017 sobre encoders de NMT.
- Søgaard & Goldberg 2016 sobre multi-task learning donde supervisar POS en capas bajas ayuda al parsing en capas altas.
- Hashimoto et al. 2017 sobre joint many-task models con jerarquía explícita.

**Por eso importan los pesos task-specific**: una tarea de POS quiere $s_1 \gg s_2$, una tarea de WSD quiere $s_2 > s_1$. Forzar a usar solo $s_L$ (como CoVe) condena a la mitad de las tareas.

### 6.4 Visualización de pesos aprendidos (Figura 2 del paper)

Cuando ELMo se incluye **en el input**: las tareas dan más peso a la **capa 1 (sintáctica)** — especialmente Coref y SQuAD, donde necesitan saber qué tipo de palabra antes de razonar.

Cuando ELMo se incluye **en el output**: el peso es más balanceado, con leve preferencia por capas bajas.

---

## 7. Limitaciones (críticas honestas)

### 7.1 Velocidad

El biLM es un **LSTM secuencial de 2 capas con 4096 unidades**. Por construcción, no se puede paralelizar a través del eje temporal: cada token depende del anterior. Para una oración de longitud $N$:

- Forward: $N$ pasos secuenciales × 2 capas.
- Backward: $N$ pasos secuenciales × 2 capas.

Inferencia típica en GPU: ~100-200 tokens/segundo para batch=1. Comparado con un Transformer que paraleliza en $O(1)$ pasos sobre toda la secuencia, es **órdenes de magnitud más lento** en aceleradores modernos. Esta es una de las razones principales por las que BERT lo reemplazó tan rápido.

### 7.2 Shallow bidirectionality

Forward LM y Backward LM son **dos modelos independientes** que comparten solo tokens y softmax. En ningún momento del pre-training un parámetro de la dirección forward "ve" información de la backward y viceversa. La combinación bidireccional ocurre **solo en el output** vía concatenación.

Esto contrasta con BERT, donde Masked LM permite atención bidireccional verdadera en cada capa, en cada cabezal. Es la razón por la que BERT (8 meses después) supera a ELMo por márgenes amplios (~10% absoluto en SQuAD, similar en GLUE).

### 7.3 Tamaño del corpus de pre-training

1B Words es modesto. BERT usa BookCorpus + Wikipedia (~3.3B tokens); GPT-2, WebText (40GB); GPT-3, 300B tokens. ELMo nunca se reescaló masivamente — el AllenNLP team siguió otras direcciones.

### 7.4 Feature extraction vs fine-tuning

ELMo es **feature extraction**: los pesos del biLM se congelan. ULMFiT y BERT mostraron que **fine-tuning end-to-end del encoder** suele ganar más, porque adapta las representaciones internas a la tarea (no solo combina las existentes).

ELMo intentó solucionar esto con el "fine-tune del biLM en dominio" pero es solo LM-fine-tune, no task-fine-tune.

### 7.5 Tamaño del modelo en memoria

Los $2L+1$ vectores por token deben mantenerse en RAM/GPU. Para batch grande y secuencias largas, el footprint es no trivial. AllenNLP usaba HDF5 caching para mitigar.

### 7.6 Sin handle de sub-palabra a nivel de output

El char-CNN entra a nivel input, pero la salida sigue siendo word-level. Una palabra rara fuera del vocab del softmax de entrenamiento se predice mal. BERT con WordPiece resuelve esto cleanly.

---

## 8. Impacto y legado

### 8.1 La onda corta (2018)

ELMo se publica en NAACL en **junio 2018**. Wins Best Paper Award. El paper original (arXiv) es de **febrero 2018**.

En **junio 2018** Radford et al. publican GPT-1 (Transformer decoder unidireccional, fine-tuning end-to-end).

En **octubre 2018**, ocho meses después de ELMo, Devlin et al. publican BERT (Transformer encoder, Masked LM, fine-tuning end-to-end). BERT cita ELMo como antecedente directo y supera sus resultados en todos los benchmarks por márgenes amplios.

A pesar de ser eclipsado tan rápido, ELMo cumplió un rol histórico crítico:

- **Estableció el paradigma "pre-train un LM gigante, transfer a tareas downstream"** como el camino dominante. CoVe lo había sugerido con MT; ELMo lo confirmó con LM puro y escala.
- **Convenció a la comunidad NLP** de que pre-training es la palanca. Antes de ELMo, mucha gente seguía iterando sobre arquitecturas más complejas con embeddings estáticos.
- **Open-source en AllenNLP** desde el día 1, lo que aceleró la adopción.

### 8.2 La onda larga

Aunque BERT/RoBERTa/T5 dominan benchmarks académicos, ELMo sigue siendo relevante en:

- **Sistemas con restricciones de cómputo** donde un Transformer es prohibitivo.
- **Dominios con vocabulario altamente especializado** donde el char-CNN es ventaja (clinical NLP, código, multilingüe sin WordPiece).
- **Pipelines legacy** donde reemplazar embeddings por ELMo es un cambio mínimo (drop-in), versus reescribir todo para fine-tune un Transformer.
- **Investigación interpretativa**: el formato $L+1$ capas explícitas con pesos aprendidos es ideal para analizar qué codifica el modelo.

Variantes notables que extendieron ELMo:
- **ELMoForManyLangs (2019)**: multilingüe.
- **Flair embeddings (Akbik 2018)**: char-LM contextualizado, filosofía similar.
- **BioELMo, SciELMo**: pre-trained en dominios médico/científico.

---

## 9. Conexión con la Clase 20 IA UC

La Clase 20 del Diplomado IA UC compara los **tres paradigmas de embeddings contextualizados** que aparecieron en 1 año:

| Modelo | Año | Familia | Bidireccionalidad | Uso downstream | Tamaño típico |
|---|---|---|---|---|---|
| **ELMo** | feb 2018 | BiLSTM | Shallow (2 LMs) | Feature extraction (congelado) | ~94M params |
| **GPT-1** | jun 2018 | Transformer decoder | Unidireccional (L→R) | Fine-tuning end-to-end | 117M params |
| **BERT** | oct 2018 | Transformer encoder | Deep bidireccional (MLM) | Fine-tuning end-to-end | 110M (base) / 340M (large) |

La narrativa pedagógica de la clase:

1. **Embeddings estáticos (clases 16-18)**: word2vec/GloVe, TF-IDF, BM25. Limitación: polisemia.
2. **ELMo (clase 20, primera parte)**: rompe la barrera de la polisemia con biLM + combinación de capas. Pero arrastra ineficiencias del LSTM.
3. **Transformer (clase 19)**: self-attention paraleliza la secuencia, abre la escala.
4. **GPT y BERT (clase 20, segunda parte)**: combinan Transformer + objetivos de LM (causal o masked) y barren todos los benchmarks.

Cross-links del site:
- **Fundamentos** → `embeddings-distribuidos.md` (concepto base de vectores densos).
- **Fundamentos** → `embeddings-contextualizados.md` (que abarca CoVe, ELMo, BERT — este paper es la pieza central).
- **Clase 14** → `transformers/` (la arquitectura que reemplaza al biLSTM).
- **Clase 19** → atención self-attention (la operación clave).
- **Clase 20** → `papers/elmo-peters-2018.md` (versión pública condensada).
- **Clase 20** → `papers/bert-devlin-2018.md` (sucesor directo).

---

## 10. Referencias clave citadas por el paper

Listado de los antecedentes y dependencias críticas que vale la pena conocer:

**Embeddings estáticos**:
- Mikolov et al. 2013 — word2vec.
- Pennington et al. 2014 — GloVe.
- Bojanowski et al. 2017 — FastText.
- Turian et al. 2010 — embeddings como features semi-supervisados (clásico).

**Arquitectura de char-CNN**:
- Kim et al. 2015 — "Character-Aware Neural Language Models".
- Józefowicz et al. 2016 — "Exploring the Limits of Language Modeling".
- Srivastava et al. 2015 — Highway Networks.
- Ling et al. 2015 — Finding function in form.

**LSTM y normalización**:
- Hochreiter & Schmidhuber 1997 — LSTM original.
- Ba et al. 2016 — Layer Normalization.
- Srivastava et al. 2014 — Dropout.
- Gal & Ghahramani 2016 — Variational dropout en RNN.

**Antecedentes contextuales**:
- Dai & Le 2015 — Semi-supervised sequence learning (ancestro).
- McCann et al. 2017 — CoVe.
- Peters et al. 2017 — TagLM (predecesor directo, mismos autores).
- Melamud et al. 2016 — context2vec.
- Ramachandran et al. 2017 — Pretraining seq2seq.

**Tareas y baselines**:
- Rajpurkar et al. 2016 — SQuAD.
- Bowman et al. 2015 — SNLI.
- Pradhan et al. 2012 — OntoNotes (Coref, SRL).
- Sang & De Meulder 2003 — CoNLL 2003 NER.
- Socher et al. 2013 — SST.
- Seo et al. 2017 — BiDAF.
- Chen et al. 2017 — ESIM.
- He et al. 2017 — Deep SRL.
- Lee et al. 2017 — End-to-end Coref.

**Corpus de pre-training**:
- Chelba et al. 2014 — One Billion Word Benchmark.

---

## 11. Síntesis conceptual

Si tuviera que reducir el paper a tres ideas accionables:

1. **Pre-train un LM grande sobre texto plano monolingual, congélalo, exporta features**. Esta es la receta dominante de NLP moderno y ELMo la cristalizó.

2. **No tires capas intermedias**. Las capas bajas y altas de una red profunda codifican distintos niveles lingüísticos. Una combinación lineal aprendida por tarea casi siempre supera a "solo la última capa".

3. **El sesgo inductivo del char-input vale la pena**. Modelos sin OOV y robustos a morfología generalizan mejor en dominios técnicos y lenguas con flexión rica.

Y una idea contra-intuitiva: aunque ELMo fue rápidamente superado por BERT, su receta de "feature extraction de un modelo congelado" sigue siendo valiosa en producción cuando el coste de fine-tuning es prohibitivo o cuando se quieren features reusables entre tareas.

---

## Notas para integrar al site (versión pública condensada)

Esta es la lista de piezas que debería trasladarse a `papers/elmo-peters-2018.md` (versión condensada, ~1500-2000 palabras, audiencia más amplia del Diplomado):

### Estructura recomendada para el site

1. **Card de metadata** (autores, venue, año, link arXiv, palabras clave).
2. **TL;DR de 3 frases** — primer párrafo del bloque "Síntesis conceptual" (sección 11).
3. **¿Qué problema resolvió?** — Sección 1.2 condensada (polisemia, embeddings estáticos).
4. **La idea central** — Sección 2 + ecuación ELMo (math en LaTeX).
5. **Arquitectura en una figura mental** — Diagrama: chars → CNN → Highway → BiLSTM × 2 → combinación lineal. Mantener char-CNN como pieza distintiva.
6. **Resultados en una tabla** — Tabla resumen de 6 tareas con incrementos.
7. **¿Qué aprende cada capa?** — Sección 6.3 (sintaxis abajo, semántica arriba). Pieza muy didáctica.
8. **Limitaciones honestas** — Versión corta de sección 7 (foco en velocidad y shallow bidir).
9. **Por qué importa en 2026** — Conexión con el ecosistema actual + cuándo seguir usándolo.
10. **Cross-links bidireccionales**:
    - ← `fundamentos/embeddings-distribuidos.md` (qué venía antes)
    - → `fundamentos/embeddings-contextualizados.md` (el concepto que crearemos)
    - → `papers/bert-devlin-2018.md` (el sucesor)
    - ↔ `clase_20/teoria.md` (la clase principal)

### Piezas que NO van al site (quedan solo en este análisis interno)

- Detalles numéricos del Supplemental (Tabla 7, hiperparámetros exactos).
- Las 11 listas de referencias completas — al site solo van 4-5 papers ancla.
- Discusión de variantes posteriores (BioELMo, Flair) — quizás breve mención.
- Derivación matemática completa del softmax sobre $s_j$ — al site va solo la ecuación final.

### Tono del site

- Más narrativo, menos enumerativo.
- Diagramas Mermaid donde el flujo del modelo se entienda visualmente (chars → token vector → 2 LSTMs → combinación).
- Una cita textual del paper para abrir.
- Cerrar con la conexión a BERT y al GPT-1 que aparecen en la misma clase.

### Math que debe sobrevivir al site

Solo estas dos ecuaciones, etiquetadas:

$$\mathbf{ELMo}_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task} \mathbf{h}_{k,j}^{LM} \quad (\text{ecuación principal})$$

$$\sum_{k=1}^{N} \big( \log p(t_k \mid t_{<k}) + \log p(t_k \mid t_{>k}) \big) \quad (\text{objetivo de pre-training})$$

Todo lo demás se puede explicar en prosa.
