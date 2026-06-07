---
title: "Stanford Attentive Reader (A Thorough Examination of CNN/Daily Mail)"
weight: 113
math: true
---

{{< paper-card
    title="A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task"
    authors="Danqi Chen, Jason Bolton, Christopher D. Manning"
    year="2016"
    venue="ACL 2016 (arXiv 1606.02858)"
    pdf="/papers/stanford-attentive-reader-chen-2016.pdf"
    arxiv="1606.02858" >}}
Paper con doble alma: introduce el **Stanford Attentive Reader** -- un modelo de comprension lectora neuronal mas simple que el Attentive Reader original, basado en bi-GRU mas **atencion bilineal** -- y al mismo tiempo audita a mano el benchmark CNN/Daily Mail para mostrar que era mucho mas facil de lo que se creia. El modelo alcanza **73.6% (CNN) y 76.6% (Daily Mail)**, superando el estado del arte previo por 7-10%. El analisis manual de 100 ejemplos revela que **~25% es ruido** (errores de correferencia o casos ambiguos) y que solo 2 de 100 exigen razonar sobre multiples oraciones. Conclusion: el dataset estaba esencialmente resuelto, y la comunidad necesitaba benchmarks mas dificiles como SQuAD.
{{< /paper-card >}}

---

## El problema

Antes de 2015, la comprension lectora (machine reading comprehension, MRC) supervisada estaba estrangulada por la falta de datos. Datasets como MCTest (2013) tenian apenas cientos de documentos, porque anotar a mano requeria expertise y diseno cuidadoso. Con tan pocos ejemplos es imposible entrenar redes profundas, que es justo donde uno esperaria capturar razonamiento textual.

Hermann et al. (2015), de DeepMind, rompieron el cuello de botella con una idea barata y elegante: los articulos de CNN y Daily Mail vienen con *bullet points* que los resumen. Tomando un bullet point, reemplazando una entidad por `@placeholder`, y pidiendo al modelo recuperar esa entidad, se obtiene una tarea **cloze** (de completar) generable a escala masiva -- 380.298 ejemplos de entrenamiento para CNN y 879.450 para Daily Mail. Ver la [paper-card de CNN/Daily Mail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015).

El detalle de diseno mas importante es la **anonimizacion de entidades**: un pipeline de NER y correferencia reemplaza cada cadena de correferencia por un marcador abstracto `@entityn`. Hermann argumenta que esto es necesario: obliga al sistema a entender el passage delante de sus ojos en vez de adivinar con conocimiento del mundo (responder "Obama" a cualquier pregunta de politica de EE.UU. porque es estadisticamente probable). Pero la anonimizacion tiene un costo: cuando el NER o la correferencia fallan, el error queda "horneado" en los datos, y a veces vuelve la pregunta imposible incluso para un humano.

Quedaban dos preguntas abiertas que nadie habia respondido con rigor: que nivel de comprension lectora exige *realmente* esta tarea algo artificial, y que han aprendido de verdad los modelos que rinden bien en ella. Chen et al. se proponen exactamente eso. Ver fundamentos de [comprension lectora automatica](/fundamentos/machine-reading-comprehension) y [question answering](/fundamentos/question-answering).

---

## La doble contribucion

El paper entrelaza dos aportes que se refuerzan mutuamente:

**(a) Un modelo mas simple y mejor.** El *Stanford Attentive Reader* (lo llaman "Neural net" en sus tablas) es una variante simplificada del Attentive Reader de Hermann. Pese a tener menos componentes, obtiene **73.6% en CNN y 76.6% en Daily Mail**, superando el SOTA previo por 7-10%. La leccion de modelado: la **atencion bilineal** simple supera al mecanismo de atencion mas elaborado del original.

**(b) Un analisis manual que muestra que el dataset es facil.** Muestrean 100 ejemplos del dev de CNN y los clasifican a mano segun el tipo de razonamiento que exigen. El hallazgo demoledor: cerca del **25% son ruido** (errores de correferencia o casos ambiguos/imposibles), y de los ejemplos *respondibles*, la inmensa mayoria se resuelve identificando una sola oracion relevante. El techo realista esta alrededor de **75%**, y sus sistemas ya estan ahi.

Las dos contribuciones cierran como tenaza: el modelo establece un *lower bound* fuerte de lo alcanzable, y el analisis manual establece un *upper bound* del techo del dataset. Cuando ambos casi coinciden, la conclusion es ineludible -- el task esta esencialmente resuelto.

---

## Stanford Attentive Reader -- arquitectura

Dada la tripleta `(p, q, a)` con passage `p = {p_1,...,p_m}` y pregunta `q = {q_1,...,q_l}`, el objetivo es inferir la entidad correcta $a \in p \cap E$ que corresponde al placeholder, donde $E$ es el conjunto de marcadores de entidad. La restriccion dura es clave: **la respuesta correcta siempre aparece en el passage**.

### Paso 1 -- Encoding

Las palabras se mapean a vectores $d$-dimensionales con una matriz de embeddings (inicializada con GloVe). Una RNN bidireccional poco profunda codifica los embeddings contextuales de cada palabra del passage:

$$\overrightarrow{h}_i = \mathrm{RNN}(\overrightarrow{h}_{i-1}, p_i), \qquad \overleftarrow{h}_i = \mathrm{RNN}(\overleftarrow{h}_{i+1}, p_i)$$

$$\tilde{p}_i = \mathrm{concat}(\overrightarrow{h}_i, \overleftarrow{h}_i) \in \mathbb{R}^h, \qquad h = 2\tilde{h}$$

Una segunda RNN bidireccional comprime la pregunta en un unico vector $q \in \mathbb{R}^h$. La celda recurrente es la **GRU** (no LSTM): rinde parecido pero es mas barata. Ver fundamentos de [LSTM y GRU](/fundamentos/lstm-gru).

### Paso 2 -- Atencion bilineal

Se compara el embedding de la pregunta con cada embedding contextual del passage, produciendo una distribucion de atencion $\alpha$ y un vector de salida $o$:

$$\alpha_i = \mathrm{softmax}_i\, q^\top W_s \tilde{p}_i$$

$$o = \sum_i \alpha_i \tilde{p}_i$$

Aqui $W_s \in \mathbb{R}^{h \times h}$ es el **termino bilineal**, que mide similitud entre $q$ y $\tilde{p}_i$ de forma mas flexible que un producto punto. El producto punto $q^\top \tilde{p}_i$ obligaria a comparar las dimensiones una a una; el termino bilineal aprende una transformacion que alinea el espacio de la pregunta con el del passage antes de medir similitud. El softmax sobre $i$ convierte los scores en pesos que suman 1, y $o$ resume "que parte del passage importa para esta pregunta". Ver fundamentos del [mecanismo de atencion](/fundamentos/mecanismo-atencion).

### Paso 3 -- Prediccion

Usando $o$ se predice la respuesta mas probable, restringida a las entidades candidatas del passage:

$$a = \arg\max_{a \in p \cap E}\, W_a^\top o$$

Se aplica softmax sobre $W_a^\top o$ limitado a $p \cap E$ y se entrena con log-verosimilitud negativa. La restriccion $a \in p \cap E$ evita competir contra todo el vocabulario.

### Tres diferencias con el Attentive Reader original

El modelo "basicamente sigue" al de Hermann, pero introduce tres cambios -- y solo el primero importa:

| Cambio | Original (Hermann) | Stanford Attentive Reader |
|---|---|---|
| Score de atencion | capa $\tanh$ (MLP no lineal) | **termino bilineal** $q^\top W_s \tilde{p}_i$ |
| Prediccion | combina $o$ y $q$ con otra capa no lineal | usa $o$ directo |
| Espacio de respuesta | todo el vocabulario $V$ | solo entidades en el passage |

Los autores son explicitos: de los tres, solo el bilineal (tomado de Luong et al. 2015 para traduccion automatica) parece importante; los otros dos solo mantienen el modelo simple. Esta honestidad -- aislar que importa de verdad en lugar de inflar la contribucion -- es parte del valor del paper.

---

## El analisis manual (el dataset era mas facil de lo creido)

El corazon del trabajo. Chen et al. muestrean uniformemente 100 ejemplos del dev de CNN y los clasifican a mano por tipo de razonamiento (si un ejemplo cae en varias categorias, lo asignan a la mas facil):

| Categoria | (%) | Descripcion |
|---|---|---|
| Exact match | 13 | las palabras alrededor del placeholder aparecen tal cual junto a una entidad |
| Paraphrasing | 41 | exactamente una oracion del passage parafrasea la pregunta |
| Partial clue | 19 | sin match completo, se infiere por solapamiento parcial |
| Multiple sentences | 2 | requiere integrar varias oraciones |
| Coreference errors | 8 | error critico de correferencia: practicamente no respondible |
| Ambiguous / hard | 17 | ni un humano respondería con confianza |

Dos hallazgos sorprenden a los propios autores:

1. **Coreference errors + ambiguous/hard = 25%.** Una cuarta parte del sample es ruido no respondible (salvo por suerte). Esto pone una barrera dura: entrenar mucho por encima de **75%** de accuracy es practicamente imposible, porque el 25% restante esta corrupto en origen.

2. **Solo 2 de 100 requieren multiples oraciones.** Mucho menos de lo que sugeria Hermann. En la mayoria de los casos respondibles, la tarea se reduce a identificar la oracion unica mas relevante e inferir desde ahi. Es decir, CNN/Daily Mail se parece mas a **extraccion de relaciones de una sola oracion** que a comprension de discurso amplio.

### Que aporta de verdad el deep learning

Cruzando las categorias con el accuracy de cada sistema se ve donde esta la ganancia:

| Categoria | Classifier | Neural net |
|---|---|---|
| Exact match | 100.0% | 100.0% |
| Paraphrasing | 78.1% | 95.1% |
| Partial clue | 73.7% | 89.5% |
| Multiple sentences | 50.0% | 50.0% |
| Coreference errors | 50.0% | 37.5% |
| Ambiguous / hard | 11.8% | 5.9% |
| **All** | **66.0%** | **74.0%** |

La diferencia entre el clasificador de features y la red neuronal esta **casi enteramente** en paraphrasing (78.1% -> 95.1%) y partial clue (73.7% -> 89.5%). El aporte real de las representaciones distribuidas no es razonamiento complejo, sino **robustez ante reformulaciones y variacion lexica** entre dos oraciones. En exact match ambos aciertan el 100% (trivial); en los casos ruidosos ambos fracasan (no hay nada que aprender). La red ya logra performance casi optima en todos los casos de una oracion y no ambiguos: no queda headroom util.

---

## Resultados

Detalles de entrenamiento: vocabulario de 50k, embeddings $d = 100$ con GloVe preentrenado, tamano oculto $h = 128$ (CNN) y $256$ (Daily Mail), SGD con learning rate 0.1, dropout 0.2, gradient clipping, hasta 30 epochs. Cada modelo se corre 5 veces y se promedia; los ensembles promedian las 5 probabilidades. El **relabeling** (renumerar `@entityn` por orden de aparicion) acelera la convergencia y suma ganancias leves.

| Modelo | CNN Test | DM Test |
|---|---|---|
| Frame-semantic model (Hermann) | 40.2 | 35.5 |
| Word distance model (Hermann) | 50.9 | 55.5 |
| Deep LSTM Reader (Hermann) | 57.0 | 62.2 |
| Attentive Reader (Hermann) | 63.0 | 69.0 |
| MemNNs window + self-sup. (Hill) | 66.8 | N/A |
| **Ours: Classifier** | 67.9 | 68.3 |
| **Ours: Neural net** | 72.7 | 76.0 |
| **Ours: Neural net (relabeling)** | **73.6** | **76.6** |
| Ours: Neural net (relabeling, ensemble) | 77.6 | 79.2 |

Observaciones:

- El **clasificador convencional** (67.9% CNN test) supera todos los enfoques de Hermann, incluidos sus sistemas neuronales, y el mejor single-system de Hill et al. (2016). Que features superficiales venzan a las redes del paper original ya es senal de que el task no era dificil.
- El **modelo neuronal single** supera lo previo por mas de 5%; el relabeling agrega 0.6-0.9%, llevando el SOTA a 73.6% y 76.6%.
- Los ensembles de 5 modelos suman 2-4% adicionales.

### El clasificador de features: que importa

Antes de la red, Chen et al. construyen deliberadamente un clasificador entity-centric con 8 plantillas de features (aparicion en passage/pregunta, frecuencia, posicion, n-gram match, word distance, co-ocurrencia, dependency match) entrenado con LambdaMART. La ablacion -- accuracy *tras quitar* cada feature, donde un numero bajo indica feature importante -- confirma el diagnostico del analisis manual:

| Quitar feature | Accuracy |
|---|---|
| Full model | 67.1 |
| − **n-gram match** | **60.5** |
| − frequency of entity | 63.7 |
| − word distance | 65.4 |
| − dependency parse match | 65.6 |

Las dos features mas decisivas son **n-gram match** y **frecuencia de la entidad**: matching superficial puramente local. Cuadra perfecto con que el task se resuelve mayormente identificando una sola oracion.

---

## Limitaciones

El paper es honesto, pero conviene marcar sus limites:

- **El analisis manual es de 100 ejemplos, solo de CNN.** No se audito Daily Mail. Con $n=100$ los porcentajes tienen intervalos amplios (el 25% de ruido podria estar entre ~17% y ~34%). Los autores publican los indices de su muestra para reproducibilidad, pero sigue siendo pequena.
- **La taxonomia tiene juicio subjetivo.** Distinguir "paraphrasing" de "partial clue" o decidir que es "ambiguous/hard" depende del anotador; no se reporta acuerdo inter-anotador.
- **El techo de ~75% es estimacion, no cota dura.** Un modelo puede "adivinar con suerte" parte del 25% ruidoso y superar el 75% sin entender realmente.
- **Sin innovacion arquitectonica profunda.** El valor esta en la simplificacion y el rigor empirico, no en un mecanismo nuevo -- la atencion bilineal venia de Luong et al. (2015).
- **Conclusiones especificas del dataset.** "RC es extraccion de relaciones de una oracion" aplica a CNN/Daily Mail por como fue construido, no a RC en general (MCTest, por contraste, tiene >50% de preguntas multi-oracion).

---

## Por que importa hoy

El paper es un clasico por dos vias independientes.

**Catalizador de SQuAD y benchmarks mas dificiles.** Cuando el techo realista es ~75% y el mejor sistema ya llega a 73.6-76.6%, el dataset esta esencialmente resuelto -- no por razonamiento profundo, sino porque un cuarto es ruido inalcanzable y el resto se resuelve con matching de una oracion. La implicancia metodologica fue enorme: la comunidad necesitaba benchmarks con menos ruido, preguntas que genuinamente integren multiples oraciones, y respuestas que no se reduzcan a entidades anonimizadas. Este trabajo es uno de los catalizadores intelectuales detras de **SQuAD** (Rajpurkar et al. 2016, el mismo ano), con spans de texto arbitrarios y preguntas escritas por humanos. Ver la [paper-card de SQuAD](/papers/squad-rajpurkar-2016).

**Cultura de dataset auditing.** El paper es citado como ejemplo paradigmatico de no aceptar un benchmark al pie de la letra, sino auditar a mano que mide realmente. Esa cultura critica -- que despues produjo trabajos sobre artefactos de anotacion, atajos espurios y "Clever Hans" en NLP -- tiene aqui uno de sus antecedentes mas limpios.

**Danqi Chen y la arquitectura didactica canonica.** Danqi Chen paso de este trabajo (al inicio de su doctorado con Manning) a convertirse en una figura central de MRC y open-domain QA (DrQA, dense retrieval; hoy profesora en Princeton). Y el Stanford Attentive Reader se volvio la pieza pedagogica estandar para ensenar comprension lectora con atencion -- aparece en CS224n de Stanford -- porque tiene el balance ideal: simple de derivar a mano (bi-GRU + atencion bilineal + argmax) y conecta con sucesores como **BiDAF** y, eventualmente, BERT para QA. Ver la [paper-card de BiDAF (Seo 2017)](/papers/bidaf-seo-2017).

---

## Conexion con la clase 24

En la Clase 24 del curso, las slides 23-27 usan el Stanford Attentive Reader como modelo central para ensenar comprension lectora. El rol pedagogico es presentar una arquitectura de atencion minima pero completa, que el estudiante pueda seguir ecuacion por ecuacion.

La matematica de las slides 26-27 corresponde directamente al paper:

- **Slide 26 -- atencion bilineal:** $\alpha_i = \mathrm{softmax}_i\, q^\top W_s \tilde{p}_i$ y el vector de salida $o = \sum_i \alpha_i \tilde{p}_i$. La clase enfatiza por que el termino bilineal $W_s$ es mas expresivo que un producto punto.
- **Slide 27 -- prediccion:** $a = \arg\max_{a \in p \cap E}\, W_a^\top o$, con la respuesta restringida a entidades del passage.

El encadenamiento de la clase es: (1) plantear RC como inferir una entidad faltante; (2) codificar passage y pregunta con bi-GRU; (3) usar atencion para que la pregunta "consulte" el passage; (4) predecir restringiendose a candidatos validos. El paper provee tanto la formalizacion limpia de este pipeline como la leccion critica -- el dataset era facil -- que motiva la transicion historica hacia SQuAD y, mas adelante, hacia modelos preentrenados como BERT aplicados a span extraction. Es el puente entre la era pre-Transformer (bi-GRU + atencion task-specific) y la era de fine-tuning.

Ver clase: [Clase 24 -- Reading Comprehension](/clases/clase-24).

---

## Notas y enlaces

- Paper: Chen, D., Bolton, J., Manning, C. D. (2016). *A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task.* ACL 2016. arXiv:1606.02858.
- Codigo original: [danqi/rc-cnn-dailymail](https://github.com/danqi/rc-cnn-dailymail).
- Dataset CNN/Daily Mail: [deepmind/rc-data](https://github.com/deepmind/rc-data) (Hermann et al. 2015, "Teaching Machines to Read and Comprehend", NIPS 2015).
- Atencion bilineal: Luong, Pham, Manning (2015), EMNLP -- origen del termino bilineal.
- GRU: Cho et al. (2014). GloVe: Pennington, Socher, Manning (2014). LambdaMART: Wu et al. (2010).
- Competidor contemporaneo: Hill et al. (2016), "The Goldilocks Principle", ICLR.
- Sucesores: SQuAD (Rajpurkar et al. 2016) como respuesta a la necesidad de benchmarks mas dificiles; tesis doctoral de Danqi Chen, "Neural Reading Comprehension and Beyond" (Stanford, 2018). Stanford CS224n usa este modelo como ejemplo didactico canonico.

Ver fundamentos: [Comprension Lectora Automatica](/fundamentos/machine-reading-comprehension) - [Question Answering](/fundamentos/question-answering) - [Mecanismo de Atencion](/fundamentos/mecanismo-atencion) - [LSTM y GRU](/fundamentos/lstm-gru).

Ver papers: [CNN/Daily Mail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015) - [SQuAD (Rajpurkar 2016)](/papers/squad-rajpurkar-2016) - [BiDAF (Seo 2017)](/papers/bidaf-seo-2017).
