---
title: "Teoria - Question Answering"
weight: 10
math: true
---

> Recorrido conceptual de las 43 diapositivas de la clase **Question Answering Models** de Vladimir Araujo. Sigue las seis secciones del temario: Introduction, Motivation, Areas in QA, Deep Learning Approaches, Metrics y References. El foco no es copiar las slides sino reconstruir el porqué de cada idea y cómo se conecta con el resto del curso.

El **Question Answering (QA)** es el problema de construir sistemas que respondan preguntas formuladas por humanos en lenguaje natural. Es, a la vez, una de las tareas más antiguas y más difíciles del NLP: antigua porque nace con la propia idea de inteligencia artificial (el Turing Test de 1950 es, en el fondo, un test de QA conversacional), y difícil porque responder bien una pregunta arbitraria exige comprender el lenguaje, recuperar información relevante y razonar sobre ella. La clase organiza ese vasto territorio y se concentra en la sub-área donde el deep learning produjo los avances más nítidos: la **Reading Comprehension**.

---

## 1. Introduction — ¿Qué es Question Answering?

### La definición operativa

{{< concept-alert type="clave" >}}
Un sistema de Question Answering es un sistema que **responde preguntas formuladas por humanos en una consulta en lenguaje natural** (*"systems that answer questions posed by humans in natural language query"*).
{{< /concept-alert >}}

La definición parece trivial hasta que uno mira la variedad de preguntas que un humano puede hacer. La clase abre con cuatro ejemplos deliberadamente heterogéneos:

- *"What time is it?"* — respondible por un reloj, sin lenguaje.
- *"What is the weather outside?"* — requiere un sensor o un servicio externo.
- *"How many did Satchel Paige strike out last night?"* — requiere recuperar un dato factual de una base de datos deportiva.
- *"What are those guys doing across the street?"* — requiere percepción visual y razonamiento sobre una escena.

Cada pregunta exige una **fuente de información** distinta y un **tipo de razonamiento** distinto. Esa heterogeneidad es el primer mensaje de la clase: QA no es una tarea única, sino una familia de tareas que comparten la interfaz (pregunta en lenguaje natural) pero difieren en todo lo demás.

### Una de las tareas más antiguas del NLP

QA precede al deep learning por décadas:

- **Años 50 — el Turing Test.** El *Imitation Game* de Alan Turing (1950) propone evaluar la inteligencia de una máquina por su capacidad de sostener una conversación indistinguible de la humana. En esencia, un protocolo de QA conversacional.
- **Años 60 — sistemas de tarjetas perforadas.** El trabajo de Simmons et al. (1961), *"Indexing and Dependency Logic for Answering English Questions"*, ya intentaba responder preguntas en inglés analizando dependencias sintácticas: dada *"What do worms eat?"*, el sistema buscaba estructuras de dependencia compatibles (*worms eat grass*, *birds eat worms*, etc.) y descartaba las que no encajaban. Aparecen también los primeros sistemas de **interfaz en lenguaje natural a bases de datos**: BASEBALL (Green et al., 1961), LUNAR (Woods et al., 1973) para consultar análisis de rocas lunares, y más tarde los NLIDB (Androutsopoulos et al.).

Esta historia importa porque fija dos paradigmas que el resto de la clase contrasta: el **simbólico/basado en reglas** (parsing, lógica, bases de conocimiento) que dominó hasta los 2000, y el **neuronal/basado en datos** que lo desplazó.

---

## 2. Motivation — ¿Por qué QA es importante?

La clase justifica el estudio de QA con cuatro argumentos:

1. **QA es un problema "AI-complete".** Resolver QA en general implica resolver muchos otros problemas: comprensión de texto, recuperación de información, razonamiento, diálogo. Es un termómetro del progreso de la IA.
2. **Implica muchas aplicaciones.** Búsqueda (search), recuperación de información (IR), diálogo, reading comprehension — todas son instancias o consumidores de QA.
3. **Resultados útiles y cotidianos.** Siri, Google Search, Alexa. QA dejó de ser académico: está embebido en productos de uso masivo.
4. **Mucho por resolver todavía.** La diapositiva con la respuesta absurda de Siri a *"Do you believe in Santa Claus?"* ("Well, those cookies don't eat themselves") ilustra que la robustez, el sentido común y la veracidad siguen siendo problemas abiertos.

{{< concept-alert type="contexto" >}}
El argumento de "AI-complete" no es retórico: cada vez que un benchmark de QA se satura (los modelos alcanzan o superan el desempeño humano), la comunidad descubre que el benchmark medía menos de lo que creía, y diseña uno más difícil. Esa dinámica —saturar y re-dificultar— estructura toda la historia de los datasets que veremos en la sección 4.
{{< /concept-alert >}}

---

## 3. Areas in QA

### 3.1 Las tres clases de datos que define a un sistema QA

Antes de taxonomizar las áreas, la clase observa que todo sistema QA se caracteriza por tres componentes:

| Componente | Variantes |
|---|---|
| **Question** | factual/factoid; complex/narrative; information retrieval |
| **Context / Source** | corpus/corpora; knowledge base; fuentes no lingüísticas (sensores, imágenes) |
| **Answer** | un hecho único; una oración o párrafo (extraído o generado); un documento; otra pregunta; un objeto (e.g. imágenes) |

El ejemplo de la tabla *"How different are questions?"* lo hace concreto: *"What is a cell?"* se responde con una enciclopedia; *"What is the price of an iphone?"* con un sitio web; *"How was the movie?"* con una opinión personal. La fuente determina la arquitectura.

### 3.2 Las cuatro áreas

La clase divide el campo en cuatro áreas según el tipo de respuesta y de fuente:

```
                 ┌─ Information Retrieval ──→ respuesta = documento/párrafo/oración
                 │
   Question  ────┼─ Reading Comprehension ─→ respuesta basada en un documento
   Answering     │
                 ├─ Semantic Parsing ──────→ respuesta = forma lógica (para usar con una KB)
                 │
                 └─ Visual QA ─────────────→ respuesta basada en una imagen
```

La clase anuncia que profundizará en **Reading Comprehension**, pero antes detalla el pipeline clásico de IR-based QA. El área de **Visual QA** se trató en el [fundamento de VQA](/fundamentos/visual-question-answering); **Semantic Parsing** queda mencionada (traducir la pregunta a una consulta estructurada tipo SQL/SPARQL sobre una base de conocimiento).

### 3.3 IR-based Factoid QA — el pipeline clásico

Para preguntas factoid cuya respuesta es un segmento corto de texto que existe en la web o en una colección de documentos (Jurafsky & Martin, 2019), el pipeline tiene tres etapas:

```
                              ┌──────────── Passage Retrieval ────────────┐
  Question                    │                                            │
     │                        │  Document      Relevant     Passage        │
     ▼                        │  Retrieval ──→  Docs    ──→ Retrieval ──→ passages
 ┌─────────────────┐         │      ▲                                      │   │
 │ Question         │         │      │ Indexing                            │   │
 │ Processing       │─────────┘   [Documents]                              │   ▼
 │ - Query          │                                              ┌──────────────┐
 │   Formulation    │··············································──→│ Answer       │──→ Answer
 │ - Answer Type    │··············································──→│ Processing   │
 │   Detection      │                                              └──────────────┘
 └─────────────────┘
```

**Etapa 1 — Question processing.** Detecta el tipo de pregunta y el **tipo de respuesta esperado** (¿una persona? ¿una fecha? ¿un lugar?), detecta relaciones, y formula las queries que se envían al motor de búsqueda.

**Etapa 2 — Passage retrieval.** Recupera documentos ranqueados, selecciona pasajes adecuados y los re-ranquea. Esta etapa es la que la era moderna transformó: del BM25 léxico al **dense retrieval** con bi-encoders (ver [DPR](/papers/dpr-karpukhin-2020) y el [fundamento de dense retrieval](/fundamentos/dense-retrieval)).

**Etapa 3 — Answer processing.** Extrae respuestas candidatas de los pasajes, las ranquea y devuelve las N mejores.

Este pipeline retriever-reader sigue vivo: es la columna vertebral de los sistemas de **Retrieval-Augmented Generation (RAG)** que hoy conectan un retriever con un LLM.

### 3.4 Reading Comprehension (MRC)

{{< concept-alert type="clave" >}}
**Machine Reading Comprehension (MRC)** es una instancia de QA donde la respuesta se basa en un **pasaje corto de texto**. La formulación canónica es **P + Q → A**: dado un pasaje $P$ y una pregunta $Q$, producir la respuesta $A$. MRC pone el énfasis en la **comprensión del texto**: responder preguntas se concibe como una forma de *medir* la comprensión del lenguaje.
{{< /concept-alert >}}

El ejemplo de la clase: un pasaje narra que Alyssa viajó de Atlanta a Miami a visitar amigos. Pregunta: *"Why did Alyssa go to Miami?"* Respuesta: *"To visit some friends"*. La respuesta está en el texto, pero requiere resolver correferencias (Alyssa = she) y localizar la cláusula causal correcta. MRC es el terreno donde el deep learning brilló, y es el foco del resto de la clase. Ver el [fundamento de MRC](/fundamentos/machine-reading-comprehension).

---

## 4. Deep Learning Approaches

### 4.1 Del NLP clásico al deep learning

La clase contrasta dos pipelines:

- **Classical NLP:** detección de idioma → pre-procesamiento (tokenización, POS tagging, stopword removal) → modelado por idioma → inferencia. Cada etapa con **features hechas a mano** (*hand-crafted features*), específicas por idioma y por tarea.
- **Deep Learning NLP:** documentos → embeddings densos (word2vec, doc2vec, GloVe) → capas ocultas → unidades de salida. Los **extractores de características se aprenden automáticamente** (representaciones distribuidas).

El cambio de paradigma es el mismo que vimos en [Clase 16 (NLP clásico)](/clases/clase-16) y [Clase 18 (embeddings)](/clases/clase-18): pasar de ingeniería de features a representaciones aprendidas.

### 4.2 La red neuronal genérica para MRC

La clase propone una plantilla común a casi todos los modelos de MRC. Dado un contexto $c$ y una pregunta $q$, la probabilidad de la respuesta $a$ se modela como:

$$p(a \mid c, q) = \exp\big(W(a)\, g(c, q)\big), \qquad a \in V$$

con tres pasos:

1. **Encode** $c$ y $q$ con redes recurrentes (o, más tarde, Transformers).
2. **Combine** $c$ y $q$ con un MLP o con atención.
3. **Produce** la respuesta $a$ con un clasificador, atención o un setup generativo.

Casi todos los modelos que siguen son especializaciones de esta plantilla; lo que cambia es **cómo se codifica**, **cómo se combina** y **cómo se produce la respuesta**. La derivación formal está en la [profundización](/clases/clase-24/profundizacion).

### 4.3 Datasets — "We need data for our models"

Antes de los modelos, los datos. La clase lista los datasets que aparecieron alrededor de 2015, el momento en que el deep learning para QA se volvió viable:

| Dataset | Tipo | Aporte |
|---|---|---|
| **CNN/Daily Mail** | cloze | primer dataset masivo de RC; [Hermann 2015](/papers/cnn-dailymail-hermann-2015) |
| **bAbI** | sintético | 20 toy tasks, una habilidad de razonamiento cada una; [Weston 2015](/papers/babi-weston-2015) |
| **Children's Book Test** | cloze | cloze por tipo de palabra; [Hill 2016](/papers/childrens-book-test-hill-2016) |
| **WikiReading** | slot-filling | predecir propiedades de Wikidata desde texto de Wikipedia |
| **SQuAD** | extractive | 100k+ preguntas, respuesta = span; [Rajpurkar 2016](/papers/squad-rajpurkar-2016) |
| **LAMBADA** | cloze (last word) | requiere contexto de discurso amplio; [Paperno 2016](/papers/lambada-paperno-2016) |
| **MS MARCO** | generativo/real | preguntas reales de Bing, respuestas generadas; [Nguyen 2016](/papers/ms-marco-nguyen-2016) |

El [fundamento de QA](/fundamentos/question-answering) organiza estos datasets por formato (cloze / extractive / generative / synthetic).

### 4.4 El CNN Dataset como tarea cloze

El dataset de Hermann et al. (2015) se construye como una **tarea cloze**: se toma un artículo de noticias con sus *bullet-point summaries*, y se genera una query reemplazando una entidad del resumen por un placeholder `X`. El modelo debe predecir qué entidad va en `X` leyendo el artículo.

El truco crucial es la **anonimización de entidades**: cada entidad nombrada se reemplaza por un marcador abstracto (`ent381`, `ent212`, ...) y los marcadores se barajan por documento. Así el modelo no puede acertar usando conocimiento del mundo ni un modelo de lenguaje a priori ("el productor de la BBC suele ser X") — está **obligado a leer el pasaje** para resolver la correferencia entre marcadores. Es un diseño elegante para forzar comprensión y no memorización.

{{< concept-alert type="trampa" >}}
Chen et al. (2016) demostraron después que el dataset es **más fácil de lo que se creía**: cerca del 75% es resoluble con features simples y alrededor del 25% tiene ruido (errores de correferencia, ambigüedad). El dataset estaba esencialmente "resuelto", lo que motivó benchmarks más difíciles como SQuAD. Volveremos a esto en la sección del Stanford Attentive Reader.
{{< /concept-alert >}}

### 4.5 Stanford Attentive Reader

El **Stanford Attentive Reader** (Chen et al., 2016) demostró una arquitectura **mínima y altamente exitosa** para reading comprehension. Su flujo:

**Paso 1 — Representación de la pregunta $Q$.** Un **Bi-LSTM** codifica la pregunta; se concatena el último estado de cada dirección para obtener un único vector $q$.

**Paso 2 — Representación del pasaje $P$.** Otro **Bi-LSTM** codifica el pasaje, produciendo un vector contextual $\tilde{p}_i$ por cada token $i$.

**Paso 3 — Atención.** La atención indica los tokens importantes del pasaje dados por la pregunta. Con una forma **bilineal**:

$$\alpha_i = \operatorname*{softmax}_i\big(q^{\top} W_s\, \tilde{p}_i\big)$$

Cada $\alpha_i$ mide cuán relevante es el token $i$ del pasaje para la pregunta $q$. El término $W_s$ es una matriz aprendida que permite comparar pregunta y pasaje en un espacio común (no un simple producto punto).

**Paso 4 — Respuesta $A$.** Se forma el vector de salida como combinación ponderada de los tokens del pasaje, y se predice la entidad:

$$o = \sum_i \alpha_i\, \tilde{p}_i, \qquad a = \arg\max_{a \in p \cap E} W_a^{\top} o$$

donde $p \cap E$ son las entidades candidatas presentes en el pasaje. El modelo elige la entidad cuyo embedding mejor alinea con $o$.

La lección del Stanford Attentive Reader es doble: (1) una arquitectura **simple** —dos Bi-LSTM y una atención bilineal— supera a modelos más complejos de la época; (2) el mismo grupo, al analizar manualmente el dataset, mostró su techo realista. Por eso es la arquitectura **didáctica canónica** de MRC (aparece en CS224n). Ver el [paper](/papers/stanford-attentive-reader-chen-2016).

### 4.6 SQuAD — la respuesta es un span

El **Stanford Question Answering Dataset** (Rajpurkar et al., 2016) reformula MRC como **extractive QA**: la respuesta es un **span contiguo de texto dentro del pasaje** (de Wikipedia). El ejemplo de la clase: dado un párrafo sobre Marco Polo, la pregunta *"How did some suspect that Polo learned about China instead of by actually visiting it?"* tiene respuesta *"through contact with Persian traders"* — exactamente esos tokens del párrafo.

Esta formulación tiene dos virtudes: es **realista** (las preguntas las escriben humanos, no un generador cloze) y es **evaluable automáticamente** (basta comparar el span predicho con el span dorado). SQuAD se volvió el benchmark que la comunidad atacó entre 2016 y 2019, hasta que BERT superó el desempeño humano. Su secuela, [SQuAD 2.0](/papers/squad2-rajpurkar-2018), agregó **preguntas sin respuesta** para forzar la abstención. Ver el [paper de SQuAD](/papers/squad-rajpurkar-2016).

### 4.7 BiDAF — Bi-Directional Attention Flow

El **BiDAF** (Seo et al., 2017) parte de una crítica a los attentive readers previos: resumían el contexto en un vector de tamaño fijo (pérdida de información), usaban atención unidireccional (solo la pregunta atendía al contexto) y atención dinámica con memoria (los errores se propagaban).

{{< concept-alert type="clave" >}}
La idea central de BiDAF: **la atención debe fluir en ambas direcciones** — del contexto a la pregunta (Context2Query) y de la pregunta al contexto (Query2Context) — y **no debe resumir prematuramente**. Cada token del contexto conserva su propia representación atendida (*attention-flow*, no *attention-summarization*).
{{< /concept-alert >}}

La arquitectura tiene **seis capas**:

1. **Character Embedding Layer** — Char-CNN, maneja palabras fuera de vocabulario.
2. **Word Embedding Layer** — GloVe preentrenado.
3. **Phrase/Contextual Embedding Layer** — Bi-LSTM que contextualiza pregunta y contexto.
4. **Attention Flow Layer** — el corazón del modelo: computa una **matriz de similitud** $S$ entre cada token del contexto y cada token de la pregunta, y de ella deriva las dos atenciones (C2Q y Q2C).
5. **Modeling Layer** — Bi-LSTM que modela las interacciones entre las representaciones *query-aware*.
6. **Output Layer** — dos predictores que dan la distribución de **start** y **end** del span.

La matriz de similitud usa la forma $S_{tj} = w^{\top}_{(S)}[h_t; u_j; h_t \circ u_j]$ (concatenación del token de contexto, el token de pregunta y su producto elemento a elemento). La derivación completa está en la [profundización](/clases/clase-24/profundizacion).

Una propiedad valiosa: los **pesos de atención son interpretables**. La clase muestra el mapa de atención para *"Where did Super Bowl 50 take place?"*, donde se ve que "where" se alinea con tokens de lugar ("Stadium", "Santa Ana") y "Super Bowl" con sus menciones en el contexto. Ver el [paper de BiDAF](/papers/bidaf-seo-2017).

### 4.8 Transformer-based — "Introduction to the Sesame World"

La llegada de los **Transformers** (Vaswani et al., 2017) y de los modelos preentrenados (BERT, Devlin et al., 2018; y la fauna de "Sesame Street" — ELMo, BERT, ERNIE, Big Bird...) cambió el juego. La receta:

1. **Transformer-based:** arquitectura de self-attention, sin recurrencias (ver [Clase 14](/clases/clase-14)).
2. **Pre-trained con corpora enormes:** el modelo aprende representaciones de lenguaje de propósito general.
3. **Fine-tuned en una variedad de tareas, incluida QA.**

#### BERT for QA

BERT aprovecha el preentrenamiento y necesita **menos pasos de entrenamiento** para aprender la tarea de QA. Su uso para extractive QA:

- **Input layer:** se aprovecha el **segment embedding** de BERT. La pregunta y el contexto se pasan **juntos** al modelo, separados por `[SEP]`: `[CLS] question [SEP] reference/context`. El segment embedding $A$ marca la pregunta y el $B$ marca el contexto.
- **Output layer:** **dos predictores** que predicen el **start span** y el **end span**. Se aplica una capa de predicción sobre cada token (posición). Concretamente, hay un vector de pesos de start (de longitud igual a la dimensión oculta, p. ej. 768) que se aplica a **cada posición** y produce, vía softmax, la probabilidad de que cada token sea el inicio de la respuesta; un vector análogo para el end.

El ejemplo de la clase: pregunta *"How many parameters does BERT-large have?"* sobre un texto de referencia que dice "…it has 24 layers and an embedding size of 1,024, for a total of 340M parameters!" — el modelo debe marcar el span "340M". También aquí los **attention weights** sirven de interpretabilidad: para *"Macedonia was under the rule of which country?"*, la atención se concentra en "hellenistic greece".

La mecánica del span prediction (loss, decodificación del span óptimo) está en la [profundización](/clases/clase-24/profundizacion) y en el [fundamento de MRC](/fundamentos/machine-reading-comprehension).

### 4.9 Generative QA

Los Transformers también habilitan **QA generativa**, donde la respuesta se **genera** token a token en lugar de extraerse. La clase resume las tres familias de modelos de lenguaje según su objetivo de entrenamiento:

| Familia | Modelo & arquitectura | Objetivo de entrenamiento |
|---|---|---|
| **Autoregressive LM** | GPT, GPT-2/3 (decoder autoregresivo) | predecir la siguiente palabra dadas las previas |
| **Masked LM** | BERT, RoBERTa, XLM-R (encoder bidireccional) | predecir palabras enmascaradas dado el resto |
| **Encoder-Decoder** | BART, T5 | corromper una secuencia y predecir la original |

Tres caminos hacia la QA generativa:

- **GPT fine-tuned** (Radford et al., 2018): se entrega pregunta + contexto como *source tokens*, y el modelo genera la respuesta token a token como *target tokens* (atención mixta self/cross).
- **BART** (Lewis et al., 2019): encoder-decoder preentrenado como autoencoder de denoising (reconstruir la entrada corrompida). Para QA, el encoder procesa pregunta + contexto y el decoder **genera la respuesta** (`bos q1 ... qn eos bos d1 ... dm eos` → `bos s1 ... sn eos`).
- **T5** (Raffel et al., 2019): preentrenado en una mezcla de tareas con el framework text-to-text. Para QA: el encoder procesa pregunta + contexto, el decoder genera la respuesta. *"When was Franklin D. Roosevelt born?"* → *"1882"*.

Estos modelos encoder-decoder son los mismos que vimos en [Clase 22 (Summarization)](/clases/clase-22); QA y summarization comparten el backbone generativo. La diferencia es el formato de entrada/salida, no la arquitectura.

---

## 5. Metrics

La evaluación de QA depende del formato de la respuesta. La clase presenta cinco métricas. El [fundamento de métricas de QA](/fundamentos/qa-evaluation-metrics) las desarrolla con ejemplos numéricos.

### Accuracy

Para QA de respuesta cerrada (cloze, multiple choice): ¿la respuesta coincide con el target dorado? Es una medida binaria:

$$\text{accuracy} = \frac{\text{correct}}{\text{correct} + \text{incorrect}}$$

### Mean Reciprocal Rank (MRR)

Para QA con un conjunto ranqueado de respuestas candidatas: mide la posición del primer ítem relevante. Si la primera respuesta es correcta, aporta 1; si la segunda, $\tfrac{1}{2}$; si no hay respuesta correcta, 0:

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

MRR es la métrica natural del passage retrieval (la etapa 2 del pipeline IR-based factoid).

### Exact Match (EM)

1 o 0 según si la predicción coincide exactamente (tras normalización) con **una de las** (típicamente 3) respuestas doradas. Es la métrica estricta de SQuAD.

### F1-Score

Interpretado como un promedio ponderado de **precision** y **recall** a nivel de **tokens**: se tratan predicción y respuesta dorada como bolsas de tokens y se mide el solapamiento. Alcanza su mejor valor en 1 y el peor en 0:

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

F1 es más indulgente que EM: premia respuestas parcialmente correctas. Es la métrica principal de SQuAD junto a EM.

### BLEU

Medida de exactitud para traducción y, por extensión, para generación de texto (la respuesta generada en QA generativo, como en MS MARCO). Mide solapamiento de n-gramas con referencias.

{{< concept-alert type="contexto" >}}
La elección de métrica refleja el formato de la respuesta: **Accuracy** para cloze/multiple-choice, **EM/F1** para extractive QA (span), **MRR / top-k** para retrieval, **BLEU/ROUGE** para generative QA. Ninguna métrica léxica captura del todo la corrección semántica — un problema que la era de los LLMs reabrió con evaluadores tipo *LLM-as-judge*.
{{< /concept-alert >}}

---

## 6. References

La clase cierra con sus referencias. Material principal:

- **Notebook tutorial:** [vgaraujov/Question-Answering-Tutorial](https://github.com/vgaraujov/Question-Answering-Tutorial) (el laboratorio asociado).
- **Papers/libros:** Jurafsky & Martin, *Speech and Language Processing* (2019); Ojokoh & Adebisi, *A Review of Question Answering Systems* (2019).
- **Slides de referencia:** Christopher Manning (*NLP with Deep Learning*, CS224n); Karl Moritz Hermann (*Deep Learning for NLP: Question Answering*); Deepak Gupta (*Question Answering: Learning to Answer from Text*).

---

## Síntesis — el arco de la clase

La clase traza un arco de 60 años: del QA simbólico (parsing + lógica + bases de datos) al QA neuronal (representaciones aprendidas + atención). Dentro del paradigma neuronal, la evolución de las arquitecturas de MRC es nítida y acumulativa:

```
Attentive Reader (2015)     atención unidireccional, resumen en vector fijo
        │
Stanford Attentive Reader   bilinear attention, más simple, mejor
(2016)  │
        ▼
BiDAF (2017)                atención bidireccional, attention-flow, span prediction
        │
        ▼
BERT for QA (2018)          self-attention pretrained, span prediction sobre cada token
        │
        ▼
Generative QA (GPT/BART/T5) la respuesta se genera, no se extrae
        │
        ▼
Retrieval-Augmented (DPR/RAG) retriever denso + reader/generador, open-domain
```

Cada paso resuelve una limitación del anterior: el Stanford AR simplifica el Attentive Reader, BiDAF rompe el resumen prematuro, BERT reemplaza las RNN por self-attention preentrenada, la generación supera la rigidez del span, y el retrieval denso lleva todo esto al dominio abierto. Esa cadena —y las métricas que la miden— es lo que hay que llevarse de la clase.

Para la mecánica matemática de cada modelo, ver la [Profundización](/clases/clase-24/profundizacion). Para los conceptos transversales, los fundamentos de [Question Answering](/fundamentos/question-answering), [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension), [Métricas de QA](/fundamentos/qa-evaluation-metrics) y [Dense Retrieval](/fundamentos/dense-retrieval).
