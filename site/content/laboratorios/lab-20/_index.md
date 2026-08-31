---
title: "Lab 20 - Modelos de lenguaje: XLNet, RoBERTa, BETO y GPT-2"
weight: 200
math: true
sidebar:
  open: true
---

**Módulo:** NLP — modelos de lenguaje pre-entrenados
**Notebook del curso:** [lab20.ipynb](/notebooks/lab20.ipynb) · [HTML](/notebooks-html/lab20.html)

## Encuadre

La contraparte práctica de la [clase 20](/clases/clase-20). El notebook carga cinco modelos de HuggingFace, entrena un clasificador de noticias falsas y plantea **trece preguntas** — la mayoría de las cuales no se responden ejecutando código, sino leyendo con atención qué se cargó.

{{< concept-alert type="atencion" >}}
**Este laboratorio se analiza distinto a los demás del site.** En el lab [42](/laboratorios/lab-42), [43](/laboratorios/lab-43) o [44](/laboratorios/lab-44), lo que se documenta son **mediciones propias**: tiempos, métricas, ablaciones. Aquí no las hay — el notebook publicado es el del curso, con las demos del profesor ejecutadas y las celdas de actividad vacías.

Lo que sí hay es un recorrido por **el código y los papers**: qué carga cada línea, por qué las cabezas de tarea de XLNet tienen seis submódulos y no dos, qué falla cuando se cruzan tokenizador y modelo, y qué está mal en el pipeline de entrenamiento que el notebook propone. Las afirmaciones se apoyan en el código fuente y en los [seis papers](#papers-que-aparecen-en-el-laboratorio) que se analizaron para acompañarlo.
{{< /concept-alert >}}

## La tesis del laboratorio

Los cinco modelos que el notebook carga —XLNet, RoBERTa, BETO, BERT y GPT-2— tienen **la misma arquitectura de fondo**: un Transformer. Lo que los distingue son decisiones que quedan fuera del diagrama y que el código sí revela:

| Modelo | Tokenizador | Vocabulario | `[CLS]` | Dependencia |
|---|---|---:|---|---|
| BERT | WordPiece | 30.522 | ID 101, **al inicio** | — |
| RoBERTa | Byte-level BPE | 50.265 | `<s>`, ID **0** | — |
| BETO | WordPiece (de SentencePiece BPE) | 31.002 | ID **4** | — |
| XLNet | Unigram LM | 32.000 | `<cls>`, **al final** | `sentencepiece` |

Cinco modelos, cuatro convenciones de tokenización incompatibles y tres posiciones distintas para el vector de clasificación. **Casi todos los errores del laboratorio salen de esta tabla**, no de la arquitectura.

## Bloques del lab

{{< cards >}}
  {{< card link="01-las-tres-cabezas-de-xlnet" title="Las tres cabezas de XLNet" subtitle="Backbone, QA y opción múltiple sobre el mismo encoder. Por qué la cabeza de QA tiene seis submódulos contra los dos de BERT — el final condicionado en el inicio, estilo R-Net — y por qué el <cls> va al final de la secuencia" icon="cube-transparent" >}}
  {{< card link="02-el-tokenizador-cruzado" title="El tokenizador cruzado" subtitle="Un accidente durante la ejecución —tokenizador de XLNet decodificando salidas de RoBERTa— que produce texto basura sin lanzar ninguna excepción. El modo de falla silencioso, cuándo sí explota, y la práctica defensiva que lo evita" icon="exclamation" >}}
  {{< card link="03-las-trece-preguntas" title="Las trece preguntas" subtitle="Las respuestas del laboratorio, con lo que hay que mirar en cada caso: el modelo, el tokenizador, los identificadores de tokens especiales, y las tres preguntas conceptuales sobre límites de contexto e instruction tuning" icon="question-mark-circle" >}}
  {{< card link="04-fake-news-y-el-atajo-de-reuters" title="Fake news y el atajo de Reuters" subtitle="El clasificador alcanza accuracy alta aprendiendo el estilo de redacción de una agencia, no factualidad. Más ocho defectos del pipeline de entrenamiento: la truncación que no coincide, el attention_mask descartado y la loss declarada que nunca se usa" icon="exclamation" >}}
  {{< card link="05-gpt-2-y-los-limites-del-contexto" title="GPT-2 y los límites del contexto" subtitle="Por qué max_length=10000 no funciona sobre un modelo de 1024 posiciones, por qué un TL;DR sin instruction tuning responde a veces, y la asimetría del ecosistema en español: hay BETO, pero el equivalente decoder-only es mucho más escaso" icon="chat-alt" >}}
{{< /cards >}}

## Clase y fundamentos

{{< cards >}}
  {{< card link="/clases/clase-20" title="Clase 20 - ELMo, BERT, GPT y ChatGPT" subtitle="El marco: de los embeddings estáticos a los contextuales, y de ahí a los modelos instruction-tuned" icon="academic-cap" >}}
  {{< card link="/fundamentos/transformer" title="Transformer" subtitle="La arquitectura común a los cinco modelos del laboratorio" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Mecanismo de Atención" subtitle="Qué cambia entre atención bidireccional y causal, y por qué eso separa encoder de decoder" icon="book-open" >}}
{{< /cards >}}

## Papers que aparecen en el laboratorio

{{< cards >}}
  {{< card link="/papers/xlnet-yang-2019" title="XLNet (2019)" subtitle="Yang et al. — permutar el orden de factorización, no la secuencia. Los dos flujos de atención existen para resolver un problema que la propia idea crea" icon="document-text" >}}
  {{< card link="/papers/roberta-liu-2019" title="RoBERTa (2019)" subtitle="Liu et al. — ninguna arquitectura nueva, y aun así supera a XLNet. Con el corpus de BERT controlado, la ventaja del permutation LM se evapora" icon="document-text" >}}
  {{< card link="/papers/beto-canete-2020" title="BETO (2020)" subtitle="Cañete et al. — el BERT en español de la Universidad de Chile. Tres precisiones sobre el paper: la configuración declarada no cuadra con el config.json, el tokenizador es SentencePiece BPE y son seis autores" icon="document-text" >}}
  {{< card link="/papers/whole-word-masking-cui-2019" title="Whole Word Masking (2019)" subtitle="Cui et al. — el wwm del nombre de BETO. Un cambio de una línea que convierte una tarea morfológica en una semántica" icon="document-text" >}}
  {{< card link="/papers/sentencepiece-kudo-2018" title="SentencePiece (2018)" subtitle="Kudo y Richardson — tratar el espacio como un símbolo más. La dependencia que el laboratorio descubre cuando XLNetTokenizer falla al cargar" icon="document-text" >}}
  {{< card link="/papers/swag-zellers-2018" title="SWAG (2018)" subtitle="Zellers et al. — Adversarial Filtering, y el dataset construido para durar que BERT superó en tres meses cruzando el desempeño humano" icon="document-text" >}}
  {{< card link="/papers/bert-devlin-2018" title="BERT (2018)" subtitle="Devlin et al. — el punto de partida de los otros cinco" icon="document-text" >}}
  {{< card link="/papers/gpt-2-radford-2019" title="GPT-2 (2019)" subtitle="Radford et al. — el decoder del laboratorio, y de donde RoBERTa toma el byte-level BPE" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Lab 18 - Word Embeddings](/laboratorios/lab-18) (los embeddings estáticos que estos modelos reemplazan) · [Lab 21](/laboratorios/lab-21) · [Lab 22 - Summarization](/laboratorios/lab-22) (BertSum, el mismo encoder puesto a resumir) · [Lab 24 - Question Answering](/laboratorios/lab-24) (las cabezas de QA de esta página, entrenadas) · Dominio [Texto](/dominios/texto).
