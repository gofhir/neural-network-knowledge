---
title: "Teaching Machines to Read and Comprehend (CNN/Daily Mail)"
weight: 112
math: true
---

{{< paper-card
    title="Teaching Machines to Read and Comprehend"
    authors="Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, Phil Blunsom"
    year="2015"
    venue="NeurIPS 2015 (arXiv 1506.03340)"
    pdf="/papers/cnn-dailymail-hermann-2015.pdf"
    arxiv="1506.03340" >}}
Dos contribuciones inseparables de DeepMind que destrabaron la comprensión lectora supervisada a gran escala. La primera es un **dataset**: convertir los *bullet points* abstractivos de CNN y el Daily Mail en ~1M de tripletas contexto–query–respuesta tipo **Cloze**, anonimizando y permutando las entidades para forzar al modelo a leer el pasaje en vez de explotar el prior del lenguaje. La segunda es una **familia de arquitecturas con atención** (Deep LSTM, Attentive y Impatient Reader) que muestra, con una ablación limpia (Uniform Reader ~39% vs Attentive ~63% en CNN), que la atención query-aware sobre tokens del documento es el ingrediente clave. El paper detonó la oleada moderna de Machine Reading Comprehension (SQuAD, CBT, MS MARCO) y es el ancestro directo del Stanford Attentive Reader.
{{< /paper-card >}}

---

## El problema

Hacia 2015 la comprensión lectora automática (*machine reading comprehension*, MRC) estaba atrapada en un cuello de botella de datos. El paper lo plantea sin rodeos: el progreso "desde algoritmos superficiales de recuperación tipo bag-of-words hacia máquinas capaces de leer y comprender documentos ha sido lento". Las dos familias tradicionales eran las **gramáticas hechas a mano** (sistemas de reglas) y la **extracción de información** (detectar triples predicado-argumento que luego se consultan como una base relacional). Ninguna aprendía a explotar la estructura del documento de forma flexible.

El obstáculo era doble y se retroalimentaba: *(a)* ausencia de datasets supervisados grandes y *(b)* dificultad de estructurar modelos lo bastante flexibles. Sin datos, el aprendizaje supervisado simplemente no podía entrar al espacio. Los corpora existentes eran diminutos — MCTest, por ejemplo, tenía apenas unos cientos de historias, útil para *testing* pero inservible para entrenar una red profunda.

La alternativa que la comunidad había explorado era generar **narrativas sintéticas** (el linaje de Memory Networks y los bAbI tasks). Pero el paper advierte sobre la trampa histórica: muchos enfoques análogos **fracasaron al pasar de datos sintéticos a entornos reales**, porque esos "mundos cerrados" no capturan la riqueza ni el ruido del lenguaje natural. El mundo cerrado de bAbI no garantiza funcionamiento sobre prosa periodística real.

La tensión de fondo que recorre todo el paper es la distinción entre **comprensión real versus *pattern matching* y conocimiento del mundo a priori**. Un modelo puede acertar no porque "entendió", sino porque memorizó estadísticas de co-ocurrencia. El objetivo se formaliza como estimar la probabilidad condicional

$$p(a \mid c, q),$$

donde $c$ es el documento de contexto, $q$ la query y $a$ la respuesta. Y el paper subraya que para una evaluación enfocada se quiere **excluir el conocimiento del mundo** (co-ocurrencia léxica) y testear la capacidad central de **detectar y comprender las relaciones entre entidades del documento**. Esa es la semilla intelectual de la anonimización.

---

## Idea central -- de bullet points a queries Cloze

La innovación que destraba el cuello de botella es elegante y oportunista. Los autores observan que las **oraciones de resumen y sus documentos asociados pueden convertirse fácilmente en tripletas contexto–query–respuesta** con algoritmos simples de detección de entidades.

El insumo viene de dos sitios que estructuralmente regalan resúmenes: **CNN** y el **Daily Mail**. Ambos complementan cada artículo con una serie de **bullet points** que sintetizan la información. Se recolectaron 93k artículos de CNN y 220k del Daily Mail.

El punto crítico es que **estos bullets son abstractivos, no copian oraciones literales del documento**. Eso es lo que hace la tarea no trivial: si fueran extractos literales, responder sería localizar la oración copiada. Al ser paráfrasis, el modelo debe hacer generalización léxica y resolución de correferencia para conectar la query con el pasaje.

El pipeline de construcción del triple:

1. Tomar un artículo y sus bullet points.
2. Convertir cada bullet en una pregunta tipo **Cloze** — el "procedimiento Cloze" de Wilson Taylor (1953), originalmente una técnica de medición de legibilidad donde se borra una palabra y el lector la rellena.
3. La conversión Cloze **reemplaza una entidad a la vez por un placeholder** (`X`). Un bullet con $k$ entidades genera hasta $k$ queries distintas.

El resultado es un corpus de aproximadamente **1M de data points**. El documento es el artículo; la query es el bullet con una entidad enmascarada; la respuesta es la entidad enmascarada. Un ejemplo del propio paper (Daily Mail):

- **Query**: "Producer X will not press charges against Jeremy Clarkson, his lawyer says."
- **Answer**: "Oisin Tymon"

El modelo debe leer el artículo (un productor de la BBC golpeado por Clarkson no presentará cargos) para resolver `X`.

### Anonimización y permutación de entidades

Esta es la decisión de diseño más influyente — y la que más debate generaría después. El paper ilustra el problema con tres queries Cloze de titulares del Daily Mail:

- "The hi-tech bra that helps you beat breast X"
- "Could Saccharin help beat X?"
- "Can fish oils help fight prostate X?"

Un modelo de lenguaje de n-gramas predeciría trivialmente que **X = cancer** sin mirar el documento, simplemente porque "cancer" es muy frecuente en ese corpus. Esa es la **solución degenerada** vía priors del modelo de lenguaje que se quiere prohibir: acertar sin leer.

Para impedirlo, el procedimiento tiene tres pasos:

1. **Coreferencia**: un sistema de resolución de correferencia establece qué menciones se refieren a la misma entidad.
2. **Reemplazo abstracto**: todas las entidades se sustituyen por **marcadores abstractos** (`@entN`). Todas las menciones de "Jeremy Clarkson" pasan a ser `ent212`; "BBC" a `ent381`; etc.
3. **Permutación aleatoria**: estos marcadores se **permutan cada vez que se carga un data point**. El mismo documento, recargado, asigna identidades distintas a las mismas entidades.

El contraste, con texto original y anonimizado:

> "The BBC producer allegedly struck by Jeremy Clarkson will not press charges against the 'Top Gear' host..."
>
> "the ent381 producer allegedly struck by ent212 will not press charges against the 'ent153' host..."

Un humano contesta ambas versiones. Pero en la anonimizada **el contexto es obligatorio**, mientras que la original podría responderse con conocimiento de fondo (cualquiera que sepa quién es Clarkson). Tras el procedimiento, "la única estrategia restante es explotar el contexto presentado con cada pregunta".

Dos consecuencias técnicas finas:

- La **permutación** es lo que mata el prior del modelo de lenguaje: aunque el modelo quisiera memorizar "ent212 = persona famosa", la identidad de `ent212` cambia entre recargas. Lo obliga a tratar los marcadores como **variables ligadas localmente** al documento.
- Los modelos **no distinguen** entre marcadores de entidad y palabras regulares: el vocabulario incluye todo, y el modelo debe **aprender** a diferenciar entidades de variables a partir de la secuencia de entrada.

El paper reconoce honestamente que un sistema de producción se beneficiaría de usar todas las fuentes de información; la anonimización es una decisión de *evaluación científica*, no de *deployment*.

### Estadísticas

Partición temporal limpia: validación de marzo 2015, test de abril 2015, evitando fuga entre splits.

| Métrica | CNN train | CNN test | DM train | DM test |
|---|---|---|---|---|
| # documentos | 90,266 | 1,093 | 196,961 | 10,397 |
| # queries | 380,298 | 3,198 | 879,450 | 53,182 |
| Prom # entidades | 26.4 | 24.5 | 26.5 | 26.0 |
| Prom # tokens | 762 | 716 | 813 | 780 |

La escala (~380k + ~880k queries de entrenamiento) es un salto de tres a cuatro órdenes de magnitud sobre los "cientos de ejemplos" previos. El promedio de **~26 entidades por documento** es clave: el modelo elige entre ~26 candidatos, fijando el *majority baseline* muy por debajo de lo trivial. La respuesta correcta está en la entidad más frecuente solo el 30.5% (CNN) / 25.6% (DM) de las veces, confirmando que un baseline de frecuencia no resuelve la tarea.

---

## Arquitecturas

Los tres modelos comparten una capa de salida. Se estima

$$p(a \mid d, q) \propto \exp\big(W(a)\, g(d, q)\big), \quad a \in V,$$

donde $V$ es el vocabulario, $W(a)$ indexa la fila $a$, y $g(d, q)$ devuelve un **embedding conjunto** del par documento–query. El modelo no privilegia entidades: debe aprender a diferenciarlas. Toda la diferencia entre modelos está en cómo se computa $g(d, q)$.

### Deep LSTM Reader

Se alimenta el documento palabra por palabra a un Deep LSTM encoder; tras un delimitador `|||`, se alimenta la query (o al revés). El modelo procesa el par como **una sola secuencia larga**, con *skip connections* desde cada entrada a cada capa oculta y de cada capa a la salida. El Deep LSTM Reader se define como

$$g^{\text{LSTM}}(d, q) = y(|d| + |q|),$$

la salida del **último paso temporal** tras leer todo. El problema conceptual, que motiva los modelos siguientes, es que el vector oculto de ancho fijo forma un **cuello de botella** para propagar dependencias a larga distancia entre query y respuesta.

### Attentive Reader

Para sortear el cuello de botella se introduce atención, inspirada en la traducción neuronal (Bahdanau et al.) y la atención visual. Documento y query se codifican con **LSTMs bidireccionales separados de una sola capa**.

El **encoding de la query** concatena las salidas finales forward y backward:

$$u = \overrightarrow{y_q}(|q|) \,\|\, \overleftarrow{y_q}(1).$$

Para cada token del documento, $y_d(t) = \overrightarrow{y_d}(t) \,\|\, \overleftarrow{y_d}(t)$. La representación $r$ del documento es una **suma ponderada** de estos vectores, donde los pesos miden cuánto atiende la red a cada token al responder:

$$m(t) = \tanh\big(W_{ym}\, y_d(t) + W_{um}\, u\big),$$
$$s(t) \propto \exp\big(w_{ms}^{\top}\, m(t)\big),$$
$$r = y_d\, s,$$

con $s(t)$ la **atención normalizada** (softmax sobre los scores). El modelo se cierra con

$$g^{\text{AR}}(d, q) = \tanh\big(W_{rg}\, r + W_{ug}\, u\big).$$

El paper observa que el Attentive Reader es una **generalización de Memory Networks**: estas atienden a nivel de oración, mientras que el Attentive Reader atiende a nivel de **token**, cada uno embebido con su contexto pasado y futuro completo gracias al encoder bidireccional. Un grano de atención más fino.

### Impatient Reader

El Impatient Reader puede **releer el documento a medida que lee cada token de la query**. En cada token $i$ computa una representación $r(i)$:

$$m(i, t) = \tanh\big(W_{dm}\, y_d(t) + W_{rm}\, r(i-1) + W_{qm}\, y_q(i)\big),$$
$$s(i, t) \propto \exp\big(w_{ms}^{\top}\, m(i, t)\big),$$
$$r(i) = y_d^{\top}\, s(i) + \tanh\big(W_{rr}\, r(i-1)\big), \quad 1 \le i \le |q|.$$

El término clave es $r(i-1)$ dentro de $m(i, t)$: la atención sobre el documento en el paso $i$ depende del estado de relectura acumulado en el paso anterior. El "impaciente" no espera a leer toda la query para mirar el documento; reenfoca su atención token a token. La representación final es

$$g^{\text{IR}}(d, q) = \tanh\big(W_{rg}\, r(|q|) + W_{qg}\, u\big).$$

---

## Resultados

Métrica: **accuracy** (fracción de queries cuya entidad-respuesta se predice bien). Todos los hiperparámetros se afinaron en validación; optimizador *asynchronous RmsProp*, modelos de atención con hidden size 256 y una sola capa.

| Modelo | CNN valid | CNN test | DM valid | DM test |
|---|---|---|---|---|
| Maximum frequency | 30.5 | 33.2 | 25.6 | 25.5 |
| Exclusive frequency | 36.6 | 39.3 | 32.7 | 32.8 |
| Frame-semantic model | 36.3 | 40.2 | 35.5 | 35.5 |
| Word distance model | 50.5 | 50.9 | 56.4 | 55.5 |
| Deep LSTM Reader | 55.0 | 57.0 | 63.3 | 62.2 |
| Uniform Reader | 39.0 | 39.4 | 34.6 | 34.4 |
| **Attentive Reader** | **61.6** | **63.0** | **70.5** | **69.0** |
| **Impatient Reader** | **61.8** | **63.8** | **69.0** | **68.0** |

Lecturas clave:

- **El Uniform Reader es la ablación crítica.** Es idéntico al Attentive Reader salvo que **fija todos los $m(t)$ iguales** (atención uniforme). Su rendimiento se desploma a ~39% (CNN) / ~34% (DM), por debajo incluso del Deep LSTM. Esto aísla causalmente la contribución de la atención: la diferencia entre ~39% y ~63% en CNN es atención pura. Es uno de los primeros *ablation studies* limpios del valor de la atención en MRC.
- **Atención > LSTM puro.** Attentive (63.0) e Impatient (63.8) superan al Deep LSTM (57.0), pese a usar LSTMs de **una sola capa** frente al Deep LSTM multicapa.
- **El Word distance es sorprendentemente fuerte** (50.9 CNN, 55.5 DM), muy por encima del frame-semantic (40.2 / 35.5). El paper lo explica por el **solapamiento léxico** frecuente entre highlights y artículo del Daily Mail, y anticipa que con preguntas reales (no Cloze) este baseline rendiría mucho peor.
- **El frame-semantic falla** por cobertura pobre (relaciones que no encajan en la estructura predicado-argumento de PropBank) y porque no escala a respuestas que requieren **varias oraciones**.
- **Daily Mail es más fácil que CNN** para casi todos los modelos, probablemente por el mayor solapamiento léxico.

Los **heatmaps de atención** muestran que para acertar el modelo hace **generalización léxica** ('killed' → 'deceased') y **resolución de correferencia/anáfora**, pero también se apoya en **heurísticas crudas** como la proximidad de las palabras de la query al candidato. El apéndice documenta honestamente los **fallos**: queries ambiguas (típicamente ubicaciones precedidas por "in") y errores del clustering de correferencia ("Kate Middleton" y "The Duchess of Cambridge" no agrupados). Esa inclusión de negativos prefigura la crítica posterior.

---

## Limitaciones -- la crítica de Chen et al. (2016)

Los autores reconocen parcialmente las limitaciones (queries ambiguas, errores de correferencia), pero la crítica más célebre vino después. En **"A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task"** (Chen, Bolton & Manning, ACL 2016), Stanford analizó manualmente una muestra y construyó un clasificador con **features simples**:

- Un sistema basado en **8 entity-centric features superficiales** (si la entidad aparece en la query, frecuencia, posición de la primera aparición, n-gram match, etc.) más una capa neuronal mínima alcanzaba ~72.4% (CNN) y ~75.8% (DM), **superando** los ~63% / ~69% de los Readers originales.
- Por inspección de ~100 ejemplos, estimaron que **solo ~25%** de las preguntas requieren razonamiento genuino multi-oración; cerca de un tercio o más son resolubles por parafraseo de una sola oración.
- Identificaron que **~25% de los ejemplos son ruidosos o no resolubles** ni por un humano, por errores de correferencia, queries ambiguas o respuestas que necesitan conocimiento ausente tras la anonimización. Esto fija un **techo práctico de accuracy ~75%**.

La consecuencia es incómoda: el dataset diseñado para medir "comprensión real versus pattern matching" resultó, en buena parte, **resoluble por pattern matching superficial sobre entidades**. La anonimización eliminó el prior del modelo de lenguaje, pero no las pistas léxicas y posicionales locales (proximidad, solapamiento) — exactamente las "heurísticas crudas" que los propios heatmaps insinuaban. Y el ruido de la correferencia automática significó que una fracción sustancial de la supervisión era incorrecta.

Esto no invalida el paper — su metodología de generación a escala y sus arquitecturas atentivas fueron seminales — pero recalibró expectativas: **CNN/Daily Mail mide más "lectura local guiada por atención" que "comprensión profunda con inferencia multi-oración"**, y motivó datasets posteriores con anotación humana de mayor calidad.

---

## Por qué importa hoy

El impacto fue desproporcionado respecto de su tamaño. Detonó la **oleada moderna de datasets de MRC**:

- **SQuAD** (Rajpurkar et al., 2016) respondió directamente a las limitaciones: preguntas formuladas por humanos sobre Wikipedia, respuestas que son **spans arbitrarios** (no solo entidades), sin el ruido de la correferencia automática.
- **MS MARCO** (Microsoft, 2016): queries reales de búsqueda de Bing, multi-documento — atacando la limitación de "single document" que el propio Hermann señalaba.
- **Children's Book Test** (Hill et al., 2016), **Who-did-What**, **NewsQA**, **RACE**: una familia entera de benchmarks Cloze y QA que siguieron el molde.

En arquitecturas, el linaje es directo. El **Stanford Attentive Reader** (Chen et al., 2016) es una reformulación y simplificación del Attentive Reader de Hermann: con una atención **bilineal** $s(t) \propto \exp(q^{\top} W y_d(t))$ — más simple que el $\tanh$ original — y prediciendo directamente sobre entidades, obtenía mejor rendimiento. Esa versión es la que se enseña como modelo canónico de atención para MRC. De ahí salen **BiDAF**, **R-NET**, **DrQA** y, finalmente, la transición a representaciones pre-entrenadas (**BERT** y descendientes) que dominaron SQuAD desde 2018. El mecanismo de atención query-aware sobre tokens del documento que Hermann formalizó es el ancestro conceptual de todas ellas.

El paper también consolidó la práctica de **visualizar la atención como herramienta de interpretabilidad** e incluir análisis de errores cualitativos, un estándar luego común en NLP neuronal. Y el split abstractivo (artículo + highlights) del mismo corpus se reutilizó masivamente como dataset de **summarization abstractiva** (Pointer-Generator, PEGASUS, BART, T5): el mismo corpus sirvió a dos tareas según se usara el bullet como query Cloze (QA) o como target de resumen.

---

## Conexión con la clase 24

En la Clase 24 (Question Answering Models), el material del profesor presenta el **CNN Dataset como tarea Cloze** para motivar el **Stanford Attentive Reader**. El rol pedagógico de este paper es el de **puente histórico y conceptual**:

1. **Motiva por qué QA neuronal necesita atención.** La narrativa "Deep LSTM → cuello de botella de vector fijo → atención" es el arco que justifica por qué los modelos de QA atienden sobre el pasaje en lugar de comprimirlo. El Uniform Reader (~39%) versus Attentive (~63%) es el experimento que convence del valor de la atención sin invocar todavía la complejidad de los Transformers.

2. **Introduce el formato Cloze como puente hacia QA extractivo.** Predecir una entidad enmascarada leyendo el pasaje es la versión simplificada del span extraction de SQuAD que el Stanford Attentive Reader resuelve: primero Cloze (respuesta = una entidad de un conjunto pequeño), luego spans arbitrarios.

3. **Conecta con la línea del curso sobre atención y embeddings contextualizados.** El Attentive Reader y su sucesor de Stanford son el eslabón entre los embeddings estáticos / contextualizados y la aplicación a responder preguntas. La atención bidireccional sobre tokens del documento es la idea que BERT lleva al extremo con self-attention multicapa.

4. **Enseña la lección metodológica del benchmark.** La crítica de Chen 2016 muestra que un dataset puede medir algo distinto de lo que pretende, que los baselines simples son esenciales para calibrar la dificultad real, y que el progreso aparente debe interrogarse — un score alto no garantiza comprensión.

---

## Notas y enlaces

- **arXiv**: [1506.03340](https://arxiv.org/abs/1506.03340) (v3, 19 nov 2015). **Venue**: NeurIPS 2015 (*Advances in Neural Information Processing Systems 28*).
- **Generador de datos**: [github.com/deepmind/rc-data](https://github.com/deepmind/rc-data) — publica el *script* que reconstruye el corpus desde las URLs; el texto de los artículos no se distribuye directamente por licencia.
- **Crítica fundamental**: Chen, Bolton & Manning (2016), "A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task", ACL 2016 — establece el techo de ~75%, el ~25% de ruido e introduce el Stanford Attentive Reader.
- **Linaje de atención**: Bahdanau, Cho & Bengio (2014, traducción neuronal); Memory Networks de Weston et al. y End-to-End Memory Networks como antecesores del mecanismo token-level.
- **Procedimiento Cloze**: Wilson L. Taylor (1953), "Cloze procedure: a new tool for measuring readability", *Journalism Quarterly* — técnica de medición de legibilidad humana reutilizada como tarea de aprendizaje automático.

Ver fundamentos: [Question Answering](/fundamentos/question-answering) - [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension) - [Mecanismo de atención](/fundamentos/mecanismo-atencion) - [LSTM y GRU](/fundamentos/lstm-gru).

Ver papers: [Stanford Attentive Reader (Chen 2016)](/papers/stanford-attentive-reader-chen-2016) - [SQuAD (Rajpurkar 2016)](/papers/squad-rajpurkar-2016) - [Children's Book Test (Hill 2016)](/papers/childrens-book-test-hill-2016).

Ver clase: [Clase 24 -- Question Answering Models](/clases/clase-24).
