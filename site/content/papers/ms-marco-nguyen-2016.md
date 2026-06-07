---
title: "MS MARCO (A Human Generated MAchine Reading COmprehension Dataset)"
weight: 117
math: true
---

{{< paper-card
    title="MS MARCO: A Human Generated MAchine Reading COmprehension Dataset"
    authors="Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, Li Deng"
    year="2016"
    venue="NeurIPS 2016 Workshop (arXiv 1611.09268)"
    pdf="/papers/ms-marco-nguyen-2016.pdf"
    arxiv="1611.09268" >}}
Dataset de Question Answering construido por Microsoft a partir del motor de busqueda Bing, pensado para acercar el benchmark al problema real de un asistente de voz. A diferencia de SQuAD, las preguntas son **queries anonimizadas de usuarios reales** muestreadas de los logs de Bing, los passages son fragmentos de documentos web recuperados por el buscador, y las respuestas son **compuestas en lenguaje natural por editores humanos** (no spans extraidos). Por construccion incluye preguntas **sin respuesta** y mas de la mitad son de tipo descriptivo. Escalo hasta ~1M de preguntas y 8.8M de passages, y su componente de **passage ranking** se volvio el benchmark de facto de la recuperacion neuronal moderna (DPR, ColBERT, TREC Deep Learning Track).
{{< /paper-card >}}

---

## El problema (SQuAD artificial)

Para 2016 el campo de Machine Reading Comprehension (MRC) y Question Answering (QA) estaba dominado por **SQuAD** (Rajpurkar et al., 2016): ~100k pares pregunta-respuesta sobre 536 articulos de Wikipedia, con una metrica trivial de computar (Exact Match y F1 sobre spans de texto). El equipo de Microsoft argumenta que SQuAD --y la mayoria de los datasets MRC de la epoca-- sufrian de tres limitaciones que los alejaban del problema real que se quiere resolver con un asistente tipo Cortana, Siri o Alexa.

**Preguntas artificiales.** En SQuAD un crowd worker lee un parrafo y *luego* formula una pregunta cuya respuesta es un span dentro de ese parrafo. Esto introduce un sesgo: la pregunta reutiliza vocabulario del passage, comparte estructura sintactica con la oracion que contiene la respuesta y nunca tiene typos ni ambiguedad de intencion. La distribucion resultante no se parece a la distribucion real de necesidades de informacion de un usuario.

**Respuesta como span extraido.** SQuAD modela QA como *localizacion*: marcar inicio y fin del span correcto. Es atractivo computacionalmente, pero es una idealizacion. Un asistente de voz no puede leer en voz alta un span recortado de Wikipedia; necesita *sintetizar* una respuesta en lenguaje natural bien formada.

**Un solo passage.** SQuAD garantiza que la respuesta esta en el parrafo dado. En el mundo real la informacion puede estar repartida entre varios documentos, puede ser conflictiva, o puede directamente no existir. El paper insiste en que el texto real es ruidoso y que los sistemas deben ser robustos a entradas problematicas.

A esto se suma la escala. El paper recoge la regla de oro de la epoca --el deep learning necesita al menos 100.000 ejemplos-- y observa que los datasets MRC grandes solian ser **sinteticos** (generados automaticamente, como los Cloze de CNN/Daily Mail), mientras que los datasets de alta calidad eran pequenos. MS MARCO busca romper ese trade-off: grande *y* real. La tesis de fondo es "datasets over algorithms": gran parte del progreso atribuido a nuevas arquitecturas fue habilitado por la aparicion del dataset adecuado (ImageNet para vision, las bases de DARPA para reconocimiento de voz). MS MARCO se postula como ese dataset para MRC y neural IR.

---

## Idea central (preguntas reales de Bing, respuestas generadas)

MS MARCO ataca las tres limitaciones simultaneamente con un cambio de fuente de datos: **usar el motor de busqueda Bing como pipeline de generacion del dataset**.

1. **Preguntas reales.** Las preguntas son queries anonimizadas, muestreadas directamente de los logs de Bing. No son inventadas mirando un passage: son lo que usuarios reales tipearon buscando informacion, con toda su ambiguedad. Un ejemplo del paper es *"in what type of circulation does the oxygenated blood flow between the heart and the cells of the body?"*; otro, *"will I qualify for osap if i'm new in Canada"* (con typos y forma coloquial). Una query como "what is the age of barack obama" puede aparecer en los logs simplemente como `barack obama age`, sin estructura interrogativa explicita.

2. **Passages reales.** Para cada pregunta, el sistema de recuperacion de passages de Bing devuelve ~10 passages extraidos de documentos web reales del indice. Estos passages pueden o no contener la respuesta, reflejando el escenario real donde el recuperador a veces falla.

3. **Respuestas generadas por humanos.** Aqui esta el cambio de paradigma frente a SQuAD. Editores humanos leen la pregunta, inspeccionan los passages y **componen una respuesta en lenguaje natural** sintetizando la informacion (extraida estrictamente de los passages provistos). La respuesta no es un span: es texto libre, idealmente en oraciones completas. Esto convierte a MS MARCO en un benchmark de **QA generativo/abstractivo**, no solo extractivo.

El subproducto de este diseno es que MS MARCO incluye, por construccion, preguntas **sin respuesta**: si ningun passage contiene la informacion, el editor marca la pregunta como no respondible. El paper defiende esto explicitamente: reconocer informacion insuficiente o conflictiva es una capacidad importante de un modelo MRC. (SQuAD recien incorporo no-answer en su v2.0 de 2018, *despues* de MS MARCO.)

---

## Construccion

El pipeline tiene cinco etapas encadenadas, con auditoria continua de calidad:

1. **Query sampling.** Se muestrean queries de los logs de Bing. Un clasificador de ML, entrenado sobre datos anotados por humanos, filtra las que no son preguntas (queries navegacionales o de otra intencion).
2. **Recuperacion de documentos.** Bing recupera documentos relevantes desde su indice web a gran escala.
3. **Extraccion de passages.** Se extraen automaticamente ~10 passages por pregunta, presentados en orden de ranking al editor.
4. **Composicion de respuesta.** El editor sintetiza una respuesta en lenguaje natural y marca con `is_selected: 1` los passages que uso. Si ninguno sirve, todos quedan con `is_selected: 0` y la pregunta se marca como no respondible.
5. **Anotacion de tipo.** Un clasificador asigna a cada pregunta un *segment label*: NUMERIC, ENTITY, LOCATION, PERSON o DESCRIPTION.

Adicionalmente, un proceso **post-hoc de review-and-rewrite** genera las respuestas *well-formed*. Un segundo editor reescribe la respuesta original si tiene mala gramatica, si hay alto solapamiento con un passage (senal de copia literal), o si no se entiende sin el contexto de pregunta y passage. El ejemplo canonico: pregunta "tablespoon in cup", respuesta original "16", respuesta well-formed *"There are 16 tablespoons in a cup."*. Esta distincion es la base de la diferencia entre la tarea novice y la intermediate.

El paper advierte una consecuencia de usar un indice vivo: ~300.000 documentos no pudieron recuperarse en el post-procesamiento porque ya no estaban en el indice, y para el resto es probable que el contenido haya cambiado desde la extraccion original. Es importante notar que el paper es un documento vivo: la v3 del arXiv (2018) describe el dataset tal como evoluciono desde la v1.0 presentada en NIPS 2016 hasta la v2.1; las cifras finales (1M de preguntas, 8.8M de passages) corresponden a esa edicion acumulada.

### Comparacion con otros datasets

| Dataset | Segment | Fuente de preguntas | Respuesta | # Preguntas | # Documentos |
|---|---|---|---|---|---|
| NewsQA | No | Crowd-sourced | Span de palabras | 100k | 10k |
| DuReader | No | Crowd-sourced | Human generated | 200k | 1M |
| NarrativeQA | No | Crowd-sourced | Human generated | 46.765 | 1.572 historias |
| SearchQA | No | Generated | Span de palabras | 140k | 6.9M passages |
| RACE | No | Crowd-sourced | Multiple choice | 97k | 28k |
| SQuAD | No | Crowd-sourced | Span de palabras | 100k | 536 |
| **MS MARCO** | **Yes** | **User logs** | **Human generated** | **1M** | **8.8M passages, 3.2M docs** |

MS MARCO es el unico con preguntas provenientes de *user logs* y con anotacion de segmento, y combina respuestas human-generated con la mayor escala documental. La distribucion por tipo revela el dato mas significativo: **53,12% de las preguntas son DESCRIPTION** (requieren respuesta textual larga), 26,12% NUMERIC, 8,81% ENTITY, 6,17% LOCATION y 5,78% PERSON. Que mas de la mitad sea descriptiva justifica por si solo la decision de usar respuestas generadas en lugar de spans.

---

## Las tres tareas

El paper define tres tareas de dificultad creciente sobre el mismo dataset:

**(a) Novice -- es respondible + responder.** El sistema predice primero si la pregunta puede responderse usando *solo* los passages provistos. Si no, devuelve literalmente `"No Answer Present"`. Si si, genera la respuesta. Combina deteccion de no-answer con generacion.

**(b) Intermediate -- respuesta well-formed.** Igual que novice, pero la respuesta debe ser *well-formed*: leida en voz alta debe tener sentido aun sin el contexto de la pregunta ni los passages. Es la tarea pensada para asistentes de voz y es estrictamente mas dificil, porque exige vocabulario general (no solo el del passage) y oraciones autocontenidas.

**(c) Passage re-ranking -- recuperacion neuronal.** Es una tarea de Information Retrieval pura. Dada una pregunta y un conjunto de **1000 passages recuperados con BM25** (Robertson et al., 2009), el sistema debe rankearlos por probabilidad de contener informacion relevante. La coleccion de passages toma la union de todos los passages del dataset y usa las anotaciones `is_selected` como senal de relevancia. El paper advierte que esta senal es **incompleta**: los editores no anotaban exhaustivamente, asi que hay passages relevantes con `is_selected: 0` (falsos negativos en el etiquetado). Esta tarea alimento el **TREC Deep Learning Track 2019**.

---

## Diferencias con SQuAD

| Dimension | SQuAD | MS MARCO | Consecuencia |
|---|---|---|---|
| Escala | ~100k preguntas, 536 docs | ~1M preguntas, 8.8M passages | >10x mas grande; permite entrenar modelos deep grandes |
| Origen de la pregunta | Editorial, mirando el span | Muestreada de logs de Bing | Distribucion natural de necesidades; ruido y ambiguedad reales |
| Respuesta | Span contiguo del passage | Compuesta por editores | QA generativo/abstractivo, no extractivo |
| No-answer | No (hasta SQuAD 2.0, 2018) | Si, por construccion | El modelo debe reconocer informacion insuficiente |
| Passages | Un parrafo garantizado | ~10 passages, posiblemente sin respuesta | Multi-passage reasoning + robustez |

La consecuencia metodologica mas profunda es la cuarta y la tercera fila combinadas: al introducir no-answer y respuestas abstractivas, MS MARCO hace invalida la metrica EM/F1 de SQuAD y obliga a un cambio de evaluacion.

---

## Metricas (ROUGE-L, BLEU)

En SQuAD la respuesta es un span, asi que Exact Match (coincidencia caracter a caracter) y F1 (solapamiento de tokens) son apropiados. En MS MARCO la respuesta es texto generado libremente, posiblemente con vocabulario ajeno al passage. EM colapsaria a casi cero (dos humanos parafrasean distinto la misma respuesta correcta), de modo que se necesitan metricas de **generacion** tomadas de traduccion automatica y summarization.

El paper usa dos familias segun la categoria de pregunta. Para respuestas **numericas / Yes-No**, accuracy y curvas precision-recall (son cortas o binarias y admiten evaluacion exacta). Para respuestas **descriptivas largas**, **ROUGE-L** (Lin, 2004) y **BLEU** (Papineni et al., 2002), mas un *phrasing-aware evaluation framework* (Mitra et al., 2016).

ROUGE-L se basa en la **subsecuencia comun mas larga (LCS)**. Si $X$ es la referencia (longitud $m$) y $Y$ la respuesta generada (longitud $n$):

$$
R_{lcs} = \frac{\mathrm{LCS}(X,Y)}{m}, \qquad
P_{lcs} = \frac{\mathrm{LCS}(X,Y)}{n}, \qquad
F_{lcs} = \frac{(1+\beta^2)\,R_{lcs}\,P_{lcs}}{R_{lcs} + \beta^2 P_{lcs}}
$$

La ventaja de la LCS es que premia el orden relativo de los tokens sin exigir contiguidad, lo que la hace tolerante a parafrasis moderada. BLEU mide precision de n-gramas con penalizacion por brevedad:

$$
\mathrm{BLEU} = \mathrm{BP}\cdot\exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right), \qquad
\mathrm{BP} = \begin{cases} 1 & c > r \\ e^{1 - r/c} & c \le r \end{cases}
$$

donde $p_n$ es la precision de n-gramas, $c$ la longitud del candidato y $r$ la de la referencia. El aporte propio del paper es **pa-BLEU** (pairwise BLEU): como cada pregunta puede tener varias respuestas de referencia de editores distintos, esa diversidad estima cuan variadamente las personas frasean la misma respuesta. pa-BLEU incorpora el consenso entre multiples referencias y, segun el paper, correlaciona mejor con juicios humanos.

Como referencia de dificultad, los baselines generativos de 2016 eran modestos: un Seq2Seq vanilla alcanzaba apenas R-L = 0,089 y una Memory Network 0,119, mientras que el truco trivial de **devolver el mejor passage entero** llegaba a R-L = 0,351. La leccion que envejecio bien: el cuello de botella en QA generativo no es el decoder sino el **acceso y la fusion de la evidencia recuperada** --exactamente lo que el dense retrieval y RAG vendrian a atacar.

---

## Por que importa hoy (passage ranking, dense retrieval, RAG)

MS MARCO supero ampliamente su proposito original como benchmark de MRC generativo y se convirtio en **infraestructura central de la recuperacion neuronal moderna**. El componente de **passage ranking** --1000 passages BM25 por query, con relevancia de `is_selected`-- se volvio el benchmark de facto de neural IR, institucionalizado por el TREC Deep Learning Track 2019.

Sobre esa coleccion de passages se entrenaron las arquitecturas que definieron el *dense retrieval*: **DPR** (bi-encoder con negativos in-batch), **ColBERT** (late interaction con MaxSim sobre embeddings token-level) y la familia de cross-encoders re-rankers basados en BERT. El tamano de MS MARCO --cientos de miles de queries con relevancia-- fue precisamente lo que hizo viable entrenar estos modelos con aprendizaje supervisado a escala.

Por extension, MS MARCO es uno de los pilares del **RAG (Retrieval-Augmented Generation)**. El mapeo entre el diseno de 2016 y la practica actual es casi uno a uno:

- La instruccion a los editores --"sintetiza la respuesta extrayendo la informacion estrictamente de los passages"-- es literalmente la restriccion de *groundedness* / *faithfulness* que se exige hoy a un LLM en un pipeline RAG para evitar alucinaciones.
- La obligacion de devolver `"No Answer Present"` cuando los passages no alcanzan es la capacidad de **abstencion** que distingue a un sistema RAG robusto de uno que inventa.
- Las respuestas **well-formed** --autocontenidas, gramaticales, con vocabulario general-- son el estandar de salida que se espera de un asistente conversacional.
- La tarea de passage re-ranking con 1000 candidatos BM25 es, estructuralmente, la etapa de *retrieve-then-rerank* que precede a la generacion en casi cualquier arquitectura RAG de produccion.

Para un ingeniero que construye QA sobre datos propios (un corpus clinico o normativo, por ejemplo), MS MARCO deja tres lecciones transferibles: (1) la calidad del retriever domina el resultado final --invertir en un buen blocker/ranker rinde mas que en un decoder sofisticado--; (2) la abstencion es una capacidad de primera clase, no un caso borde; y (3) la evaluacion de generacion libre es intrinsecamente dificil y ROUGE/BLEU son solo proxies, lo que anticipa la necesidad actual de evaluacion con LLM-as-judge o metricas de faithfulness especificas.

---

## Conexion con la clase 24

La [Clase 24](/clases/clase-24) (Question Answering) situa MS MARCO en varios de sus ejes:

- **Datasets de QA:** el panorama de la clase lista MS MARCO entre los datasets de referencia, junto a SQuAD, NewsQA, NarrativeQA y RACE. MS MARCO es el ejemplar canonico del dataset con preguntas reales de usuario y respuestas generadas.
- **Generative QA:** mientras SQuAD modela QA como extraccion de span, MS MARCO exige *sintetizar* la respuesta en lenguaje natural --el tipo de QA que la clase cubre en su parte generativa. La distincion novice (responder) vs. intermediate (well-formed autocontenida) ilustra de forma concreta que significa "generar" una respuesta.
- **IR-based / factoid QA:** la tarea de passage re-ranking conecta con el QA basado en recuperacion. El pipeline de MS MARCO (query -> BM25 -> re-ranking neuronal -> lectura/generacion) es el esquema arquetipico del QA factoid sobre corpus abierto.
- **Dense retrieval:** MS MARCO es el dataset sobre el que se entrenan los retrievers densos (DPR, ColBERT). Es el puente entre el BM25 lexico y el retrieval semantico neuronal.

En sintesis: si SQuAD es el referente de QA extractivo single-passage de la clase, MS MARCO es el referente de QA generativo multi-passage con recuperacion --el dataset que conecta los tres bloques de la clase (factoid/IR, retrieval denso y generacion).

---

## Notas y enlaces

- **arXiv:** 1611.09268 -- <https://arxiv.org/abs/1611.09268> (v3, 31 oct 2018).
- **Sitio del dataset:** <https://microsoft.github.io/msmarco/> -- incluye el *datasheet* del dataset (inspirado en Gebru et al., 2018, "Datasheets for Datasets").
- **TREC Deep Learning Track:** <https://trec.nist.gov/> -- la tarea de passage/document ranking que nacio de MS MARCO.
- **Detalle historico:** v1.0 en NIPS 2016; v1.1 (ene 2017); v2.0 (mar 2018) y v2.1 (abr 2018). Los numeros "finales" (1M preguntas, 8.8M passages) corresponden a la edicion acumulada descrita en la v3 del arXiv.

Ver fundamentos: [Question Answering](/fundamentos/question-answering) - [Dense Retrieval](/fundamentos/dense-retrieval) - [Metricas de evaluacion de QA](/fundamentos/qa-evaluation-metrics).

Ver papers: [SQuAD (Rajpurkar 2016)](/papers/squad-rajpurkar-2016) - [DPR (Karpukhin 2020)](/papers/dpr-karpukhin-2020) - [ROUGE (Lin 2004)](/papers/rouge-lin-2004).

Ver clase: [Clase 24 -- Question Answering](/clases/clase-24).
