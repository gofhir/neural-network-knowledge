---
title: "ELMo (Deep Contextualized Word Representations)"
weight: 295
math: true
---

{{< paper-card
    title="Deep Contextualized Word Representations"
    authors="Peters, Neumann, Iyyer, Gardner, Clark, Lee, Zettlemoyer"
    year="2018"
    venue="NAACL 2018 (Best Paper Award)"
    pdf="/papers/elmo-peters-2018.pdf"
    arxiv="1802.05365" >}}
ELMo introduce **embeddings contextualizados** vía un biLM profundo: una capa de entrada char-CNN + Highway seguida de **dos BiLSTM apilados**, pre-entrenados como modelos de lenguaje sobre 1B Words. La contribución central es la **combinación lineal aprendida por tarea de todas las capas internas** (no solo la última). Drop-in feature sobre el embedding estático de cualquier arquitectura downstream, mejoras de 6 a 25% en reducción relativa de error sobre seis tareas (SQuAD, SNLI, SRL, Coref, NER, SST-5) y un salto cualitativo en sample efficiency. Es el puente conceptual entre la era word2vec/GloVe y la era BERT/GPT.
{{< /paper-card >}}

---

## Contexto

Entre 2013 y 2017 la lingüística computacional aplicada vivió la era de los **embeddings estáticos**: word2vec (Mikolov 2013), GloVe (Pennington 2014) y FastText (Bojanowski 2017) asignaban un único vector a cada palabra, cargado como lookup table al inicio de la red de la tarea. Las arquitecturas downstream típicas —BiLSTM+CRF para etiquetado, encoder-decoder con atención de Bahdanau para MT, BiDAF para QA, ESIM para entailment— heredaban esos vectores fijos y construían toda la lógica contextual desde cero. El problema fundamental era la polisemia: "play" en *"made a spectacular play"* (deporte), en *"signed to do a Broadway play"* (teatro) y en *"watch the kids play"* (verbo lúdico) colapsaban en un mismo punto del espacio. Los vecinos más cercanos a "play" en GloVe son una nube morfológica y deportiva (*playing, game, players, football*) sin separación por sentido. Para WSD, SRL, coreferencia y QA esto es una limitación dura: todas dependen críticamente de capturar **qué función cumple la palabra en esta oración**.

Antecedentes directos —CoVe (McCann 2017, encoder de un seq2seq de traducción inglés-alemán), TagLM (Peters 2017, biLM idéntico al de ELMo pero usando solo la capa superior), context2vec (Melamud 2016, un solo LSTM bidireccional)— habían mostrado que pre-entrenar un encoder y usarlo como feature mejora las tareas, pero todos se limitaban a la última capa o estaban acotados por corpus paralelos pequeños. Dai & Le (2015) habían propuesto pre-entrenar LSTMs como autoencoders o LMs antes de fine-tune; es el ancestro genético de toda la familia. ELMo cristalizó la receta moderna —pre-entrenar un LM grande sobre texto plano monolingüe, exportar features de todas las capas con una mezcla aprendida por tarea— y abrió el camino que GPT-1 (junio 2018) y BERT (octubre 2018) consolidarían ese mismo año.

## Ideas principales

### 1. Pre-entrenar un biLM profundo desde caracteres

ELMo no usa lookup table de palabras. La entrada se computa **desde caracteres** siguiendo a Kim et al. (2015) y Józefowicz et al. (2016):

1. Cada carácter se embebe en baja dimensión (~16-d).
2. **2048 filtros convolucionales** con anchos $w \in \{1, \ldots, 7\}$ sobre la secuencia de chars.
3. **Max-pooling por filtro** sobre la dimensión temporal.
4. Dos capas **Highway Network** (Srivastava 2015), análogas a un LSTM en el dominio feed-forward.
5. **Proyección lineal a 512** dimensiones — el token layer $\mathbf{x}_k^{LM}$.

Esto da tres ventajas estructurales: **no hay OOV** (cualquier palabra inventada o con typo se mapea a un vector razonable), **sensibilidad morfológica** ("walked"/"walking"/"walker" comparten estructura de chars) y **robustez a ruido superficial**. La capa es deliberadamente *context-insensitive*: dos ocurrencias de "play" producen exactamente el mismo $\mathbf{x}_k^{LM}$ — todo lo contextual sucede más arriba.

Sobre esta entrada se apilan **L = 2 capas BiLSTM** con 4096 unidades cada una, proyección a 512 después de cada capa (projection LSTM, Sak 2014) y conexión residual entre capa 1 y capa 2. El modelo es esencialmente un *CNN-BIG-LSTM* reducido a la mitad respecto del original de Józefowicz, para balancear cómputo.

El objetivo de pre-training es la log-likelihood conjunta de dos LMs independientes —forward y backward— que comparten solo el embedding de tokens y la capa softmax:

$$\sum_{k=1}^{N} \Big( \log p(t_k \mid t_{<k}; \Theta_x, \overrightarrow{\Theta}_{LSTM}, \Theta_s) + \log p(t_k \mid t_{>k}; \Theta_x, \overleftarrow{\Theta}_{LSTM}, \Theta_s) \Big)$$

Las dos LSTMs nunca se mezclan durante el pre-training: el forward predice $t_k$ a partir de su prefijo, el backward a partir de su sufijo. La concatenación ocurre solo al exportar las representaciones, capa por capa. Esto es lo que después se llamará **"shallow bidirectionality"** — y es la limitación que BERT corregirá con Masked LM.

Corpus: **One Billion Word Benchmark** (~800M tokens de noticias), 10 épocas, ~2 semanas en 3 GPUs. La perplexity final (39.7 promedio) es peor que el CNN-BIG-LSTM original (30.0), pero a la mitad del cómputo.

### 2. Combinación lineal aprendida por tarea (la innovación cuantitativa)

Para cada token $k$, el biLM expone $2L+1 = 5$ vectores: el token layer y los estados ocultos concatenados forward/backward de cada una de las 2 capas LSTM. Por convención se reorganizan en $L+1 = 3$ vectores por token:

$$R_k = \{\mathbf{h}_{k,j}^{LM} \mid j = 0, 1, 2\}$$

donde $\mathbf{h}_{k,0}^{LM}$ es el token layer y $\mathbf{h}_{k,j}^{LM}$ para $j \geq 1$ son las concatenaciones BiLSTM. CoVe y TagLM usaban solo la capa superior $\mathbf{h}_{k,L}^{LM}$. ELMo generaliza con la ecuación principal del paper:

$$\mathbf{ELMo}_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task} \, \mathbf{h}_{k,j}^{LM}$$

Tres piezas:

- $s_j^{task}$ son pesos **softmax-normalizados** sobre las capas ($\sum_j s_j = 1$), **aprendidos por tarea**.
- $\gamma^{task}$ es un **escalar global** que reescala la norma del vector resultante — sin él, la diferencia de magnitudes entre activaciones del biLM y representaciones internas de la tarea hace que el ajuste falle.
- Opcionalmente, layer normalization sobre cada $\mathbf{h}_{k,j}^{LM}$ antes de ponderar.

El paper demuestra mediante ablations (Tabla 2) que esta generalización no es cosmética: usar solo la última capa **falla totalmente en SRL** y degrada SNLI. La razón se discute en la siguiente idea principal.

### 3. Distintas capas codifican distintos niveles lingüísticos

La sección 5.3 del paper —probablemente la más teóricamente importante— usa el biLM **sin tarea supervisada** para sondar qué codifica cada capa:

| Tarea (probing) | Capa 1 biLM | Capa 2 biLM | SOTA supervisado |
|---|---|---|---|
| **POS tagging** (sintaxis) | **97.3** | 96.8 | 97.8 |
| **WSD** (semántica) | 67.4 | **69.0** | 70.1 |

Para POS la capa **inferior** gana; para WSD la capa **superior**. La conclusión: **capas bajas codifican información léxica y sintáctica** (POS, morfología, dependencias locales), **capas altas codifican semántica** (sentido, rol, coreferencia). Este patrón es consistente con Belinkov et al. 2017 sobre encoders NMT y con la familia de joint many-task models de Hashimoto et al. 2017.

Por eso los pesos $s_j^{task}$ importan: una tarea de tagging quiere $s_1 \gg s_2$, una tarea de QA o WSD quiere $s_2 > s_1$. Forzar el uso de solo $s_L$ —como CoVe— condena a la mitad de las tareas. La Figura 2 del paper confirma además que cuando ELMo se incluye **en la capa de input** las tareas dan más peso a la capa sintáctica (Coref y SQuAD necesitan saber qué tipo de palabra antes de razonar); cuando se incluye **en el output** el peso es más balanceado.

### 4. Receta de uso downstream: drop-in feature

ELMo está diseñado para integrarse sin reescribir la arquitectura de la tarea:

1. **Congelar** los pesos del biLM (no se actualizan).
2. Ejecutar el biLM una vez por oración y guardar los $2L+1$ vectores por token.
3. **Concatenar** $\mathbf{ELMo}_k^{task}$ con la representación context-independent del modelo base (GloVe + char-CNN de la tarea): $[\mathbf{x}_k; \mathbf{ELMo}_k^{task}]$.
4. Alimentar la arquitectura RNN/CNN existente sin cambios.

Tres variantes que el paper explora:

- **Solo input**: concatenar en la capa de embeddings de la tarea.
- **Solo output**: concatenar al estado oculto del biLSTM de la tarea antes de la cabeza.
- **Input y output**: con pesos $\mathbf{s}^{task}$ separados para cada posición.

El lugar óptimo depende de la tarea: SQuAD y SNLI prefieren input+output (la atención posterior se beneficia de ambos), SRL y Coref prefieren solo input. Como detalles prácticos, los autores agregan dropout sobre ELMo y regularización L2 sobre los pesos $\mathbf{s}$ (un $\lambda = 0.001$ pequeño suele ganar por margen estrecho sobre el promedio uniforme).

Un detalle pragmático muy valioso: **fine-tune del biLM en dominio** antes de congelarlo. Una época sobre el train split de la tarea (sin labels) baja la perplexity drásticamente (e.g., SNLI: 72.1 → 16.8) y mejora la accuracy supervisada en ~0.6% para entailment. No siempre ayuda —en sentiment es nulo— pero es un truco barato.

## Resultados experimentales

Seis tareas, todas con caída del estado del arte simplemente agregando ELMo a la arquitectura baseline correspondiente:

| Tarea | Métrica | SOTA previo | Baseline | + ELMo | Δ absoluto | Δ relativo (error) |
|---|---|---|---|---|---|---|
| **SQuAD** (QA) | F1 | 84.4 | 81.1 | **85.8** | +4.7 | **24.9%** |
| **SNLI** (Entailment) | Acc | 88.6 | 88.0 | 88.7 | +0.7 | 5.8% |
| **SRL** (CoNLL 2012) | F1 | 81.7 | 81.4 | **84.6** | +3.2 | **17.2%** |
| **Coref** (CoNLL 2012) | avg F1 | 67.2 | 67.2 | 70.4 | +3.2 | 9.8% |
| **NER** (CoNLL 2003) | F1 | 91.93 | 90.15 | **92.22** | +2.06 | **21%** |
| **SST-5** (Sentiment) | Acc | 53.7 | 51.4 | 54.7 | +3.3 | 6.8% |

Reducciones relativas de error de **6 a 25%**, consistentes en todas las tareas. Comparado directamente con CoVe (el competidor más fuerte), ELMo gana en todos los benchmarks: en SQuAD el incremento de +1.8% de CoVe contrasta con +4.7% de ELMo (2.6× mejor). Los autores atribuyen la diferencia a tres factores: (1) corpus de pre-training cuatro veces más grande y monolingüe (no paralelo), (2) deep usage (todas las capas), (3) objetivo de LM puro (no MT).

Más impactante aún es el resultado de **sample efficiency** (Sección 5.4). En SRL el modelo con ELMo alcanza su accuracy máxima en **epoch 10**, comparado con epoch **486** sin ELMo — 98% de reducción en número de updates. Y con ELMo el modelo logra con **1% del training set** el mismo F1 que la baseline obtiene con **10%**: reducción de 10× en datos etiquetados. Esto es lo que la literatura posterior llamará *few-shot transfer* y es el driver económico principal del paradigma de pre-training.

## Limitaciones reconocibles

- **Velocidad e inferencia secuencial**: el biLM es un LSTM de 2 capas con 4096 unidades; cada token depende del anterior y no se paraleliza en el eje temporal. Inferencia típica ~100-200 tokens/segundo en GPU (batch 1). Frente a un Transformer que paraleliza en $O(1)$ pasos sobre toda la secuencia, ELMo es órdenes de magnitud más lento. Es una de las razones principales por las que BERT lo reemplazó tan rápido.
- **Shallow bidirectionality**: forward LM y backward LM son **dos modelos independientes**. Ningún parámetro de una dirección ve información de la otra durante el pre-training; la bidireccionalidad solo emerge al concatenar en el output. BERT, ocho meses después, corrige esto con Masked LM y obtiene atención bidireccional verdadera en cada capa.
- **Corpus modesto**: 1B Words (~800M tokens) frente a BookCorpus+Wikipedia (3.3B) de BERT o los 300B tokens de GPT-3. ELMo nunca se reescaló masivamente.
- **Feature extraction vs fine-tuning**: ELMo congela el biLM. ULMFiT (concurrente) y BERT mostraron que fine-tunear el encoder end-to-end suele ganar más, porque adapta las representaciones internas en lugar de solo combinar las existentes.
- **Output word-level**: el char-CNN cubre la entrada, pero el softmax del LM sigue siendo word-level. Palabras raras fuera del vocab de pre-training se predicen mal. WordPiece (BERT) y BPE (GPT-2) resuelven esto cleanly.
- **Footprint en memoria**: los $2L+1$ vectores por token deben mantenerse disponibles. Para batches grandes y secuencias largas no es trivial; AllenNLP usaba HDF5 caching.

## Por qué importa hoy

ELMo cumplió un rol histórico crítico aunque su ventana de dominio en benchmarks fue corta. **Estableció el paradigma "pre-train un LM gigante sobre texto plano, transfer a tareas downstream"** como la dirección dominante de NLP. Antes de febrero de 2018 mucha gente seguía iterando sobre arquitecturas más complejas con embeddings estáticos; después de ELMo la conversación cambió, y en seis meses se publicaron GPT-1 (junio 2018, Transformer decoder unidireccional con fine-tuning) y BERT (octubre 2018, Transformer encoder bidireccional con MLM). BERT cita ELMo como antecedente directo, supera sus resultados por márgenes amplios (~10% en SQuAD), y consolida la receta que ELMo abrió.

La narrativa pedagógica que importa retener: ELMo es el momento en que NLP aprende que **no se tira nada del pre-entrenamiento**. La idea de "todas las capas con pesos aprendidos por tarea" sigue viva en variantes modernas como adapters, prompt tuning con representaciones intermedias, y los pipelines de probing/interpretabilidad. La distinción capas bajas = sintaxis / capas altas = semántica es uno de los hallazgos más replicados en la literatura de interpretabilidad de Transformers, y se enseña por primera vez de manera limpia en este paper.

En 2026 ELMo sigue siendo relevante en nichos específicos: sistemas con restricciones de cómputo donde un Transformer es prohibitivo, dominios con vocabulario altamente especializado donde el char-CNN da ventaja sin tokenizer dedicado (clinical NLP, código, lenguas con flexión rica como turco o finés, datasets con muchos typos o ruido superficial), pipelines legacy donde reemplazar embeddings es un cambio mínimo —drop-in sobre la arquitectura existente, sin reescribir el modelo de la tarea— y trabajo de investigación interpretativa donde el formato $L+1$ capas explícitas con pesos aprendidos por tarea es ideal para sondar qué codifica el modelo. Variantes notables que extendieron la idea: **ELMoForManyLangs** (multilingüe, 2019), **Flair embeddings** (Akbik 2018, char-LM contextualizado con filosofía similar pero más liviano), **BioELMo** y **SciELMo** (pre-trained en dominios médico y científico, ampliamente usados en pipelines clínicos previos a la adopción masiva de ClinicalBERT). El paradigma "feature extraction de un encoder congelado" sigue siendo valioso en producción cuando el coste de fine-tuning es prohibitivo o cuando se quieren features reusables entre múltiples tareas downstream.

La comparación con los dos sucesores inmediatos que aparecen en la misma clase ilustra la lógica de la transición de paradigma:

| Modelo | Año | Familia | Bidireccionalidad | Uso downstream |
|---|---|---|---|---|
| **ELMo** | feb 2018 | BiLSTM | Shallow (2 LMs independientes) | Feature extraction (congelado) |
| **GPT-1** | jun 2018 | Transformer decoder | Unidireccional (L→R) | Fine-tuning end-to-end |
| **BERT** | oct 2018 | Transformer encoder | Deep bidireccional (MLM) | Fine-tuning end-to-end |

Cada paso resuelve una limitación del anterior: GPT cambia el biLSTM secuencial por un Transformer paralelizable; BERT corrige la shallow bidirectionality vía Masked LM y pasa de feature extraction a fine-tuning end-to-end. La línea que conecta los tres es continua: pre-train un LM grande, transfer a tareas downstream. ELMo es donde esa línea se vuelve nítida.

La síntesis accionable del paper en tres ideas: (1) pre-train un LM grande sobre texto plano y exporta features de un modelo congelado, (2) no tires capas intermedias —cada nivel codifica algo distinto—, (3) el sesgo inductivo del char-input vale la pena en dominios técnicos. La primera idea sobrevive en GPT/BERT, la segunda en interpretabilidad mecanicista, la tercera en arquitecturas como ByT5 y CANINE que regresan a la representación de chars.

## Notas y enlaces

- El paper se lee en una sentada (~12 páginas + supplemental). Las secciones 3 (modelo), 5.3 (análisis por capa) y 5.4 (sample efficiency) son las que vale la pena estudiar a fondo.
- Código y checkpoints originales: [github.com/allenai/bilm-tf](https://github.com/allenai/bilm-tf) y la integración de AllenNLP. Implementaciones modernas en HuggingFace bajo `allenai/elmo`.
- Antecedente inmediato: **TagLM** (Peters et al., ACL 2017) — mismo equipo, biLM idéntico, pero usando solo la última capa. ELMo es esencialmente TagLM + "usa todas las capas con pesos aprendidos".
- Concurrente: **ULMFiT** (Howard & Ruder, ACL 2018) — propone fine-tuning total en lugar de feature extraction; en retrospectiva, más cercano al espíritu de BERT.
- Sucesor inmediato y dominante: ver [BERT (Devlin 2018)](/papers/bert-devlin-2018).

Ver fundamentos: [Embeddings distribuidos](/fundamentos/embeddings-distribuidos) - [Pre-training y BERT](/fundamentos/pretraining-bert) - [Transfer Learning](/fundamentos/transfer-learning) - [LSTM y GRU](/fundamentos/lstm-gru) - [Self-attention](/fundamentos/self-attention) - [Clase 20](/clases/clase-20).
