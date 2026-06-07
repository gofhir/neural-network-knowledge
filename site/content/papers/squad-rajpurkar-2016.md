---
title: "SQuAD (100,000+ Questions for Machine Comprehension)"
weight: 114
math: true
---

{{< paper-card
    title="SQuAD: 100,000+ Questions for Machine Comprehension of Text"
    authors="Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang"
    year="2016"
    venue="EMNLP 2016 (arXiv 1606.05250)"
    pdf="/papers/squad-rajpurkar-2016.pdf"
    arxiv="1606.05250" >}}
El benchmark fundacional del Machine Reading Comprehension moderno: 107,785 pares pregunta-respuesta sobre 536 articulos de Wikipedia, donde **la respuesta es siempre un span contiguo del pasaje** (extractive QA). La decision de diseno central -- responder seleccionando un fragmento literal del texto en vez de elegir entre opciones o generar texto libre -- combina realismo (el sistema enfrenta $O(L^2)$ candidatos) con evaluabilidad automatica (metricas **Exact Match** y **F1 token-level**). El baseline de regresion logistica alcanza 51.0% F1 frente a 86.8% F1 humano, dejando un gap de ~36 puntos que la comunidad cerro en dos anios y que **BERT (2018) llego a superar**. SQuAD fijo el patron metodologico -- dataset grande de alta calidad + metricas robustas + leaderboard con test oculto -- replicado por GLUE, Natural Questions, HotpotQA y casi todo el NLP de fines de los 2010.
{{< /paper-card >}}

---

## El problema -- datasets previos

La tesis motivacional del paper es una observacion historica: los datasets grandes y realistas mueven campos enteros. Los autores invocan ImageNet (reconocimiento de objetos, 2009) y el Penn Treebank (parsing sintactico, 1993). La pregunta implicita es por que la comprension lectora (Reading Comprehension, RC) no tenia su ImageNet.

La respuesta: los datasets de RC existentes sufrian de uno de dos defectos mutuamente excluyentes.

**(i) Alta calidad pero pequenios.** MCTest (Richardson 2013) tiene 660 historias creadas por crowdworkers, 4 preguntas por historia y 4 opciones por pregunta -- 2640 preguntas en total. Es dificil y requiere razonamiento de sentido comun, pero con miles de ejemplos no se puede entrenar una red profunda sin overfitting masivo. El problema no es la dificultad sino la escala.

**(ii) Grandes pero semi-sinteticos.** Aqui entran los datasets *cloze*. [CNN/Daily Mail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015) construyo 1.4M de preguntas borrando entidades de los resumenes abstractivos de articulos de noticias; la tarea es rellenar la entidad faltante. El Children's Book Test (Hill 2015, 688K) predice una palabra borrada dadas las 20 oraciones anteriores. Son enormes porque se generan automaticamente, pero esa misma automatizacion es su talon de Aquiles. Critico: Chen et al. (2016) demostraron que CNN/Daily Mail requeria mucho menos razonamiento del que se pensaba y estaba "casi saturado" -- el ruido y la simplicidad estructural del cloze hacian que el techo del benchmark fuera bajo y poco informativo.

A esto se suma una tercera familia: los datasets puramente sinteticos como bAbI (Weston 2015), utiles para diagnostico controlado pero cuyo lenguaje procedural no captura la riqueza del lenguaje natural.

Hay una distincion conceptual fina que el paper resalta: en las queries cloze la respuesta es meramente *sugerida* (suggested) por el pasaje, mientras que en SQuAD las respuestas estan *implicadas* (entailed, derivables) por el pasaje. Ademas, las respuestas cloze son palabras o entidades individuales; las de SQuAD a menudo incluyen no-entidades y frases largas.

SQuAD se posiciona en la interseccion vacia: grande (casi dos ordenes de magnitud mas que MCTest) y de alta calidad (preguntas escritas por humanos sobre pasajes reales).

---

## Idea central -- extractive QA, la respuesta es un span

El corazon de SQuAD es una decision de diseno elegante: **la respuesta a cada pregunta es un segmento contiguo de texto -- un *span* -- dentro del pasaje correspondiente.** No hay opciones multiples, no hay generacion libre, no hay banco de candidatos predefinido. El sistema debe seleccionar, entre todos los spans posibles del pasaje, aquel que responde la pregunta.

Esta formulacion, conocida hoy como *extractive question answering*, tiene dos virtudes que explican su adopcion masiva.

**Es realista.** A diferencia del multiple choice (que permite eliminar distractores) y del cloze (que reduce todo a predecir una entidad), responder con un span obliga a lidiar con un numero grande de candidatos. Si el pasaje tiene $L$ palabras, hay $O(L^2)$ spans posibles. El sistema no clasifica entre 4 alternativas: tiene que localizar las fronteras exactas de la respuesta.

**Es evaluable automaticamente.** Este es el punto pragmatico decisivo. Una respuesta de forma libre es dificil de evaluar: como decide una maquina si "la fuerza de gravedad" y "gravity" son la misma respuesta. Con spans, la respuesta correcta es un fragmento literal del texto, lo que permite metricas automaticas robustas. Esta evaluabilidad es lo que habilita un leaderboard cerrado y comparaciones reproducibles entre cientos de sistemas -- el motor social del benchmark.

El ejemplo canonico del paper (Figura 1) es un pasaje sobre precipitacion meteorologica, con preguntas como "What causes precipitation to fall?" $\to$ **gravity**, "What is another main form of precipitation besides drizzle, rain, snow, sleet and hail?" $\to$ **graupel**, y "Where do water droplets collide with ice crystals to form precipitation?" $\to$ **within a cloud**. Las tres respuestas son spans literales de longitudes distintas (una palabra, una palabra, una frase de tres palabras), y la primera exige razonar que "under" en "falls under gravity" denota causa, no localizacion.

---

## Construccion del dataset

La recoleccion fue en tres etapas: curacion de pasajes, crowdsourcing de pares pregunta-respuesta, y respuestas adicionales.

**Curacion de pasajes.** Se usaron los PageRanks internos de Wikipedia (Project Nayuki) para obtener los top 10,000 articulos en ingles, de los cuales se muestrearon **536 articulos** uniformemente al azar. De cada articulo se extrajeron parrafos individuales, eliminando imagenes, figuras y tablas, y descartando parrafos de menos de 500 caracteres: **23,215 parrafos**. La particion se hizo **a nivel de articulo** (no de parrafo, lo que evita fuga de informacion entre splits): 80% train, 10% dev, 10% test.

**Recoleccion de pares pregunta-respuesta.** Crowdworkers via Daemo sobre Amazon Mechanical Turk, con requisitos exigentes (tasa de aceptacion del 97%, minimo 1000 HITs, EE.UU. o Canada), 4 minutos por parrafo, USD 9 por hora. En cada parrafo el trabajador formulaba y respondia hasta 5 preguntas: escribia la pregunta y resaltaba (highlight) la respuesta en el parrafo. Detalle de diseno clave contra el sesgo de copiar literalmente: se animo a usar palabras propias y se **deshabilito el copy-paste** sobre el texto del parrafo. Esto induce variacion lexica y sintactica entre pregunta y pasaje -- justo lo que hace la tarea dificil.

**Respuestas adicionales.** Para estimar el rendimiento humano y robustecer la evaluacion, se obtuvieron **al menos 2 respuestas adicionales** por pregunta en dev y test (total $\geq 3$ por pregunta). En esta tarea secundaria se mostraban solo preguntas y parrafos, y se pedia seleccionar el span mas corto que respondiera. Un dato que anticipa la secuela: **el 2.6% de las preguntas fueron marcadas como no respondibles** por al menos un crowdworker adicional.

| Metrica de construccion | Valor |
|---|---|
| Articulos de Wikipedia | 536 (de top 10,000 por PageRank) |
| Parrafos extraidos | 23,215 |
| Pares pregunta-respuesta | 107,785 |
| Preguntas por parrafo | hasta 5 |
| Respuestas por pregunta (dev/test) | $\geq 3$ |
| Split | 80% train / 10% dev / 10% test (por articulo) |
| Pago a crowdworkers | USD 9 / hora |
| Plataforma | Daemo sobre Amazon Mechanical Turk |
| Preguntas no respondibles (dev/test) | 2.6% |

---

## Tipos de razonamiento

Los autores analizan el dev set en tres ejes: diversidad de respuestas, razonamiento requerido y divergencia sintactica.

**Diversidad en las respuestas.** Se categorizan automaticamente con constituency parses, POS tags y NER de Stanford CoreNLP. A diferencia de los datasets cloze, **casi la mitad de las respuestas no son entidades de nombre propio**:

| Tipo de respuesta | Porcentaje | Ejemplo |
|---|---|---|
| Date | 8.9% | 19 October 1512 |
| Other Numeric | 10.9% | 12 |
| Person | 12.9% | Thomas Coke |
| Location | 4.4% | Germany |
| Other Entity | 15.3% | ABC Sports |
| Common Noun Phrase | 31.8% | property damage |
| Adjective Phrase | 3.9% | second-largest |
| Verb Phrase | 5.5% | returned to Earth |
| Clause | 3.7% | to avoid trivialization |
| Other | 2.7% | quietly |

**Razonamiento requerido.** Sobre 192 ejemplos etiquetados manualmente (4 por cada uno de 48 articulos), un resultado central: **todos presentan alguna divergencia lexica o sintactica** entre pregunta y respuesta (un ejemplo puede caer en mas de una categoria):

| Tipo de razonamiento | % | Descripcion |
|---|---|---|
| Lexical variation (synonymy) | 33.3% | sinonimos ("sometimes called" $\leftrightarrow$ "referred to as") |
| Lexical variation (world knowledge) | 9.1% | requiere conocimiento del mundo |
| Syntactic variation | 64.1% | estructura de dependencias no coincide ni con modificaciones locales |
| Multiple sentence reasoning | 13.6% | anafora o fusion de multiples oraciones |
| Ambiguous | 6.1% | sin respuesta unica o desacuerdo con el crowdworker |

La **variacion sintactica domina (64.1%)**: es el desafio mas frecuente. Para cuantificarla, los autores definen una *divergencia sintactica* sobre paths de arboles de dependencias: se detectan *anchors* (pares palabra-lema comunes a pregunta y oracion de respuesta), se extraen paths no lexicalizados desde el anchor hasta la wh-word y hasta el span, y se mide la *edit distance* minima entre ellos. El humano resulta **insensible a la divergencia sintactica**, mientras que el baseline degrada -- senial de que el entendimiento profundo no se distrae con diferencias superficiales.

---

## Metricas EM y F1

SQuAD introdujo el par de metricas que se volvio estandar de facto para extractive QA. Ambas **ignoran puntuacion y articulos** (a, an, the) tras normalizar.

**Exact Match (EM).** Porcentaje de predicciones que coinciden *exactamente* con alguna respuesta ground truth. Binaria por pregunta: 1 si la prediccion normalizada es identica a alguna referencia, 0 si no. Estricta, siempre queda por debajo del F1.

**F1 token-level (macro-promediado).** Mide el solapamiento promedio. Se tratan prediccion y ground truth como *bags of tokens*. Sea $P$ el bag predicho y $G$ el gold; los tokens compartidos son $|P \cap G|$ (con multiplicidad):

$$\text{precision} = \frac{|P \cap G|}{|P|}, \qquad \text{recall} = \frac{|P \cap G|}{|G|}, \qquad F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}.$$

Como en dev/test cada pregunta tiene $\geq 3$ respuestas gold, el F1 de una pregunta es el **maximo** sobre todas las referencias, y luego se promedia sobre las $N$ preguntas:

$$F_1 = \frac{1}{N} \sum_{i=1}^{N} \max_{g \in \text{gold}_i} F_1(\text{pred}_i, g).$$

El operador max es deliberado: humanos razonables discrepan sobre los limites exactos de un span (incluir o no frases no esenciales). Tomar el maximo sobre las referencias hace la metrica robusta a esa variabilidad legitima sin penalizar al sistema por elegir una frontera defendible distinta de una referencia arbitraria.

**Ejemplo de computo.** Gold answers {"within a cloud", "a cloud"}, prediccion "in a cloud". Tras normalizar (quitar articulos): prediccion {in, cloud}, golds {within, cloud} y {cloud}. Contra la primera: compartidos {cloud} = 1, precision $1/2$, recall $1/2$, F1 = 0.5. Contra la segunda: {in, cloud} vs {cloud}, compartidos 1, precision $1/2$, recall $1/1$, F1 $= 2 \cdot 0.5 \cdot 1 / 1.5 \approx 0.667$. Se toma el maximo: **F1 = 0.667**. El EM seria 0 porque ninguna coincidencia es exacta.

---

## Baselines y gap humano

Una restriccion de ingenieria compartida: en vez de los $O(L^2)$ spans, solo se usan los que son *constituents* del constituency parse. Ignorando puntuacion y articulos, el 77.3% de las respuestas correctas del dev son constituents -- un **techo efectivo del 77.3%** sobre estos baselines especificos (no sobre el dataset).

**Sliding Window.** Computa solapamiento unigram/bigram entre la oracion candidata y la pregunta, con la extension por distancia de Richardson (2013).

**Logistic Regression** (el modelo fuerte). Extrae features por candidato, discretizando cada feature continua en 10 buckets, para un total de **180 millones de features**, la mayoria lexicalizadas o de dependency tree path. Grupos: matching word/bigram frequencies (TF-IDF), root match, lengths, span word frequencies, constituent label, span POS tags, lexicalized, y dependency tree paths. Loss de log-verosimilitud multiclase, AdaGrad (lr 0.1), L2, tres pasadas.

| Metodo | EM Dev | EM Test | F1 Dev | F1 Test |
|---|---|---|---|---|
| Random Guess | 1.1% | 1.3% | 4.1% | 4.3% |
| Sliding Window | 13.2% | 12.5% | 20.2% | 19.7% |
| Sliding Win. + Dist. | 13.3% | 13.0% | 20.2% | 20.0% |
| Logistic Regression | 40.0% | 40.4% | 51.0% | 51.0% |
| **Human** | 80.3% | **77.0%** | 90.5% | **86.8%** |

El rendimiento humano se estima tratando la *segunda* respuesta de cada pregunta como "prediccion humana" y las otras como ground truth: **86.8% F1 / 77.0% EM** en test. La regresion logistica (51.0% F1) supera ampliamente al sliding window (~20%) pero queda muy por debajo del humano. Diagnostico revelador: el modelo selecciona la oracion correcta con **79.3% de accuracy** -- el grueso de la dificultad esta en encontrar el span exacto *dentro* de la oracion.

**Ablation.** Quitar features lexicalizadas y de dependency tree paths es lo mas danino: F1 dev cae de 51.0% a 35.8% al quitar ambas. Consistente con que la variacion sintactica es el desafio dominante (64.1%), los dependency tree path features juegan un rol mucho mayor que en Chen et al. (2016).

El gap de ~36 puntos no es un fracaso sino una **invitacion**: los autores lo enmarcan como "un buen problema desafio para investigacion futura". El paper ya documenta la respuesta en tiempo real: en la version de octubre de 2016, Wang y Jiang con [Match-LSTM + Answer Pointer](/papers/bidaf-seo-2017) ya habian alcanzado 70.3% F1, "mas que reduciendo a la mitad" el gap.

---

## Limitaciones

**Toda respuesta es un span (no hay unanswerable).** En SQuAD v1.0 cada pregunta tiene por construccion una respuesta que es un span del pasaje. Un sistema puede asumir que *siempre* hay respuesta y nunca abstenerse. El 2.6% de preguntas marcadas como no respondibles anticipa el problema, que [SQuAD 2.0 (Rajpurkar, Jia, Liang, 2018)](/papers/squad2-rajpurkar-2018) vino a arreglar agregando 50,000+ preguntas no respondibles escritas adversarialmente, forzando a decidir *si* responder ademas de *que*.

**Sesgo de Wikipedia.** Los 536 articulos son texto enciclopedico formal, solo en ingles, sesgado hacia alto PageRank. No transfiere necesariamente a texto conversacional, tecnico, multilingue o de baja calidad.

**Preguntas formuladas mirando el pasaje.** Los crowdworkers escribieron las preguntas *con el parrafo a la vista*. Aunque se deshabilito el copy-paste, esto induce un sesgo de "pregunta retrospectiva": alto solapamiento de contenido y presuposicion de que la respuesta existe ahi -- distinto del open-domain QA real donde el usuario pregunta *sin* haber leido el documento.

**Span contiguo.** Excluye respuestas que requieran agregacion, conteo o sintesis de fragmentos no adyacentes. El techo del 77.3% por restringir a constituents es una limitacion de los baselines del paper, no del dataset.

---

## Por que importa hoy

SQuAD fue el benchmark central del Machine Reading Comprehension entre 2016 y 2019, y su leaderboard publico el motor competitivo del subcampo:

- **2016:** regresion logistica 51.0% F1 (este paper); Match-LSTM + Answer Pointer 70.3% F1.
- **2016-2018:** una sucesion de arquitecturas con atencion bidireccional y pointers ([BiDAF](/papers/bidaf-seo-2017), R-Net, DrQA, QANet) escalando el F1.
- **2018:** [BERT (Devlin et al.)](/papers/bert-devlin-2018) marca el punto de inflexion. El fine-tuning de un Transformer preentrenado **supero el rendimiento humano** (86.8% F1) en el leaderboard de v1.1 -- hito citado como evidencia del poder del preentrenamiento contextual.
- **2018+:** SQuAD 2.0 reintroduce el desafio con preguntas no respondibles.

Mas alla de los numeros, SQuAD establecio un *patron metodologico*: dataset grande de alta calidad + metricas automaticas robustas (EM/F1) + leaderboard publico con test oculto. Ese patron se replico en GLUE, SuperGLUE, Natural Questions, TriviaQA, HotpotQA y CoQA. La formulacion de extractive QA con span quedo como la interfaz estandar de las pipelines de QA en produccion, incluyendo las cabezas de "question answering" de Hugging Face Transformers, que exponen directamente los logits de inicio y fin de span sobre el pasaje.

---

## Conexion con la clase 24

En el material del curso, el PDF presenta SQuAD en el slide 28 como el ejemplo paradigmatico de **extractive QA**, con la consigna "the answer is a span" -- exactamente la idea central de este analisis. El ejemplo del profesor es el de **Marco Polo** (un pasaje sobre el explorador con respuesta-span), que cumple el mismo rol pedagogico que la precipitacion de la Figura 1: mostrar que la respuesta no se genera ni se elige de una lista, sino que se *localiza* dentro del pasaje. Los slides 41-42 introducen las metricas **Exact Match y F1**.

SQuAD es el puente natural de la clase entre la comprension lectora como tarea cognitiva y la maquinaria de NLP moderno que la resuelve (representaciones contextuales, atencion, fine-tuning de Transformers). El gap humano-maquina de 2016 y su cierre por BERT en 2018 es una narrativa didactica perfecta: ilustra en un solo benchmark por que el preentrenamiento contextual fue una revolucion, conectando esta clase con el material de embeddings contextualizados (ELMo/BERT/GPT) del modulo de NLP.

La leccion transferible a sistemas reales: la eleccion de la *formulacion de la tarea* y de la *metrica* es tan determinante como la arquitectura. SQuAD triunfo no por un truco de modelado sino por convertir comprension lectora en algo medible automaticamente y a escala. El operador max sobre multiples referencias gold es un patron directamente aplicable a cualquier evaluacion con multiples respuestas correctas defendibles.

---

## Notas y enlaces

- **Paper:** Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP 2016. arXiv:1606.05250.
- **Dataset:** https://stanford-qa.com (SQuAD v1.0; los resultados del paper son sobre v1.0; v1.1 corrige errores menores de tokenizacion y es la del leaderboard historico). Reproducibilidad completa en CodaLab.
- **Trabajo siguiente:** [SQuAD 2.0 (Rajpurkar, Jia, Liang, 2018)](/papers/squad2-rajpurkar-2018) -- agrega 50,000+ preguntas no respondibles adversariales.
- **Hito de superacion humana:** [BERT (Devlin et al., 2018)](/papers/bert-devlin-2018) -- primer modelo en superar el F1 humano (86.8%) en el leaderboard de v1.1.
- **Datasets de contexto:** [CNN/Daily Mail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015), MCTest (Richardson 2013), CBT (Hill 2015), bAbI (Weston 2015).
- **Herramientas del paper:** Stanford CoreNLP (constituency, POS, NER, dependencies); Daemo + Amazon Mechanical Turk; Project Nayuki Wikipedia PageRanks; AdaGrad.

Ver fundamentos: [Question Answering](/fundamentos/question-answering) - [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension) - [Metricas de evaluacion en QA](/fundamentos/qa-evaluation-metrics).

Ver papers: [CNN/Daily Mail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015) - [SQuAD 2.0 (Rajpurkar 2018)](/papers/squad2-rajpurkar-2018) - [BiDAF (Seo 2017)](/papers/bidaf-seo-2017) - [BERT (Devlin 2018)](/papers/bert-devlin-2018).

Ver clase: [Clase 24 -- Question Answering y Machine Reading Comprehension](/clases/clase-24).
