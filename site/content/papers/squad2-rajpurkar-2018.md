---
title: "SQuAD 2.0 (Know What You Don't Know)"
weight: 116
math: true
---

{{< paper-card
    title="Know What You Don't Know: Unanswerable Questions for SQuAD"
    authors="Pranav Rajpurkar, Robin Jia, Percy Liang"
    year="2018"
    venue="ACL 2018 (arXiv 1806.03822)"
    pdf="/papers/squad2-rajpurkar-2018.pdf"
    arxiv="1806.03822" >}}
SQuAD 1.1 garantizaba que la respuesta siempre existe en el contexto, lo que entrenaba a los modelos a **responder algo siempre** y, hacia 2018, ya estaba saturado (sistemas por encima del humano en exact match). SQuAD 2.0 agrega **53.775 preguntas no respondibles** escritas adversarialmente por crowdworkers: relevantes al parrafo y con una respuesta **plausible pero incorrecta** del tipo correcto presente en el texto. El objetivo deja de ser solo *span selection* y pasa a ser *answerability* + *span selection*: el sistema debe **abstenerse** cuando ninguna respuesta esta respaldada. El mejor modelo de la epoca (DocQA + ELMo) cae de 85.8 F1 en 1.1 a 66.3 F1 en 2.0, frente a 89.5 humano: una brecha que reabrio el benchmark. Best Short Paper Award, ACL 2018.
{{< /paper-card >}}

---

## El problema: SQuAD 1.1 estaba saturado

La comprension lectora automatica (*machine reading comprehension*) se habia vuelto tarea central del NLP hacia 2016-2018, impulsada por datasets grandes etiquetados: CNN/Daily Mail, NewsQA, TriviaQA y sobre todo SQuAD ([Rajpurkar et al. 2016](/papers/squad-rajpurkar-2016)). Para mediados de 2018, los mejores sistemas ya **superaban la exactitud humana en exact match** sobre SQuAD 1.1. El benchmark estaba saturado: las mejoras se median en decimas y dejaban de ser informativas sobre el progreso real.

El exito en 1.1 no implicaba comprension genuina. Weissenborn et al. (2017) mostraron que los modelos podian aprender **heuristicas de coincidencia de tipo** (*type-matching*): si la pregunta es "cuando", buscar la fecha mas cercana a las palabras de la pregunta; si es "quien", la persona. Jia y Liang (2017) mostraron que el exito en 1.1 no garantizaba robustez frente a oraciones distractoras.

La causa raiz es estructural: **SQuAD 1.1 garantiza que la respuesta correcta siempre existe en el contexto**. Bajo esa garantia el modelo nunca necesita verificar que la respuesta esta realmente *implicada* (*entailed*) por el texto; solo seleccionar el span que parece mas relacionado con la pregunta. Es un sesgo de diseno que premia adivinar por coincidencia superficial. En terminos practicos: 1.1 entrenaba modelos para **siempre responder algo**, que es exactamente el comportamiento que produce alucinaciones cuando la respuesta no esta en los documentos.

Las dos heuristicas combinadas (tipo + contexto lexico) resuelven una fraccion enorme de 1.1 sin razonamiento sobre el significado. SQuAD 2.0 esta disenado para que ese par de heuristicas falle, porque la respuesta plausible (el span del tipo correcto, cerca de las palabras de la pregunta) esta presente pero es incorrecta.

---

## Idea central: ensenar a abstenerse

SQuAD 2.0 combina los datos de SQuAD 1.1 con **53.775 preguntas nuevas no respondibles** (*unanswerable*), escritas adversarialmente sobre los mismos parrafos. La consigna tiene dos condiciones:

1. **Relevancia**: la pregunta debe ser relevante al tema del parrafo.
2. **Respuesta plausible**: debe existir en el contexto un span del **mismo tipo** que el que la pregunta pide, aunque sea incorrecto.

Para tener exito en SQuAD 2.0 un sistema ya no solo debe responder cuando es posible, sino tambien **determinar cuando ninguna respuesta esta respaldada por el parrafo y abstenerse**. La frase del titulo, *know what you don't know*, resume el cambio de objetivo: de *span selection* a *answerability* + *span selection*.

El paper ilustra el mecanismo con un parrafo de la *Endangered Species Act* que dice que ciertas leyes tuvieron "poca oposicion". Las preguntas no respondibles:

- *"Which laws faced significant opposition?"* &rarr; respuesta plausible (incorrecta): *later laws*. El parrafo dice lo contrario. Es una **trampa de negacion/exclusion**.
- *"What was the name of the 1937 treaty?"* &rarr; respuesta plausible (incorrecta): *Bald Eagle Protection Act* (que es de 1940). El tratado de 1937 se menciona pero no se nombra. Es un **entity swap / confusion de fechas**.

En ambos casos hay un span del tipo correcto justo ahi, listo para que un modelo superficial lo agarre. Esa es la esencia adversarial del dataset.

---

## Construccion del dataset

Un *negative example* es un par (passage, pregunta no respondible).

**Proceso.** Sobre la plataforma de crowdsourcing Daemo, cada tarea consistia en un articulo completo de SQuAD 1.1. Por cada parrafo los trabajadores debian formular **hasta cinco preguntas imposibles de responder** que (a) referenciaran entidades del parrafo y (b) tuvieran una respuesta plausible presente. Se les mostraban las preguntas reales de 1.1 como inspiracion, lo que reforzaba que las no respondibles se parecieran a las respondibles. Primero escribian la pregunta, luego resaltaban la respuesta plausible.

**Filtrado.** Se eliminaron preguntas de trabajadores con 25 o menos preguntas por articulo (senal de abandono), tanto en datos nuevos como en los heredados.

**Splits.** Se reuso la **misma particion de articulos** que 1.1. En dev y test quedo una proporcion **aproximadamente 1:1** de respondibles a no respondibles; en train hay **aproximadamente el doble** de respondibles. Esta diferencia importa para el calibrado del umbral.

| Split | Metrica | SQuAD 1.1 | SQuAD 2.0 |
|---|---|---:|---:|
| **Train** | Total de ejemplos | 87.599 | 130.319 |
| | Ejemplos negativos | 0 | 43.498 |
| **Development** | Total de ejemplos | 10.570 | 11.873 |
| | Ejemplos negativos | 0 | 5.945 |
| **Test** | Total de ejemplos | 9.533 | 8.862 |
| | Ejemplos negativos | 0 | 4.332 |

(Las 53.775 preguntas no respondibles del resumen son la suma antes del filtrado; las tablas reportan los negativos retenidos: 43.498 + 5.945 + 4.332.)

**Calidad humana.** Para confirmar limpieza, crowdworkers adicionales respondieron todas las preguntas de dev y test, con respondibles y no respondibles **mezcladas**. Por cada pregunta debian resaltar la respuesta o marcarla no respondible. Se recolectaron **multiples respuestas por pregunta** (promedio 4.8) y se eligio por **voto mayoritario**. A diferencia de 1.1 (que evaluo a un unico humano y probablemente subestimo la exactitud humana), aqui la estimacion es mas alta y robusta.

---

## Por que las negativas son dificiles

El nucleo conceptual es que **no cualquier pregunta no respondible sirve**. Una negativa trivial se detecta con heuristicas lexicas baratas, y entrenar contra ella no ensena nada sobre *answerability* real. El paper revisa enfoques previos para mostrar por que fallan:

- **Distant supervision** (Zero-shot RE, Levy et al. 2017): 65% de los negativos no tienen respuesta plausible; basta el *type-matching* para descartarlos.
- **TFIDF** (Clark y Gardner 2017): emparejan preguntas con otros parrafos del articulo por solapamiento TF-IDF, pero los parrafos recuperados suelen ser poco relevantes y un detector de solapamiento los separa facil.
- **NewsQA** (Trischler et al. 2017): solo 9.5% resultan no respondibles (no escala) y se excluyen del dataset final.
- **RuleBased** (Jia y Liang 2017): edicion por reglas (reemplazo de entidades/numeros por palabras similares, antonimos de WordNet). Poco diversas.

La diferencia clave de SQuAD 2.0: como las negativas se construyen para **compartir lexico y tipo con el passage** (relevancia + respuesta plausible), no pueden filtrarse por solapamiento de palabras ni por *type-matching*. El modelo esta obligado a razonar sobre si el texto realmente **implica** la respuesta. Esto conecta directo con *recognizing textual entailment* (RTE): decidir si una hipotesis esta implicada, contradicha o es neutral respecto de una premisa.

---

## Metricas con abstencion

Siguiendo a Rajpurkar et al. (2016), se reportan **Exact Match (EM)** y **F1**. La extension clave es el manejo de los negativos:

> Para ejemplos negativos, abstenerse recibe un puntaje de **1**, y cualquier otra respuesta recibe **0**, tanto para EM como para F1.

El espacio de salida incluye una opcion explicita de "no answer". La evaluacion es, conceptualmente:

$$
\text{score}(q) =
\begin{cases}
1 & \text{si } q \text{ es negativa y el modelo abstiene} \\
0 & \text{si } q \text{ es negativa y el modelo responde algo} \\
\text{EM/F1 estandar} & \text{si } q \text{ es positiva (vs gold spans)}
\end{cases}
$$

**Como decide abstenerse.** Los modelos predicen, ademas de la distribucion sobre spans, una **probabilidad de que la pregunta sea no respondible**, y abstienen cuando esa probabilidad supera un umbral $\tau$. El $\tau$ se ajusta por modelo sobre el dev set para **maximizar F1**. Esto funciona mejor que el simple argmax, debido a las **distintas proporciones de negativos en entrenamiento (2:1) y test (1:1)**: el umbral optimo depende del *prior* de la clase en produccion, que rara vez coincide con el de entrenamiento.

Un baseline que **siempre se abstiene** obtiene **48.9 F1 en test**: con proporcion ~1:1, abstenerse acierta en la mitad (las negativas) y falla en la otra mitad (las positivas). Que ese baseline mudo quede tan cerca del mejor sistema real (66.3 F1) es la metrica mas elocuente del paper: la mayor parte de la distancia entre "no hacer nada" y "comprender" sigue sin recorrerse.

La metrica **castiga simetricamente la sobre-confianza y la sub-confianza**: un modelo que nunca se abstiene paga en todas las negativas; uno que se abstiene de mas paga en todas las positivas que podia responder. El umbral $\tau$ equilibra esos dos errores, la misma decision que enfrenta cualquier sistema de QA en produccion al elegir un umbral de confianza.

---

## Resultados: la caida de los modelos

Se evaluaron tres arquitecturas con capacidad de "no answer": **BNA** (BiDAF-No-Answer, Levy et al. 2017), **DocQA** (Clark y Gardner 2017) y **DocQA + ELMo** (con representaciones contextualizadas ELMo, Peters et al. 2018).

| Sistema | 1.1 test F1 | 2.0 dev EM | 2.0 dev F1 | 2.0 test EM | 2.0 test F1 |
|---|---:|---:|---:|---:|---:|
| BNA | 77.3 | 59.8 | 62.6 | 59.2 | 62.1 |
| DocQA | 81.0 | 61.9 | 64.8 | 59.3 | 62.3 |
| DocQA + ELMo | 85.8 | 65.1 | 67.6 | 63.4 | 66.3 |
| **Human** | 91.2 | 86.3 | 89.0 | 86.9 | 89.5 |
| **Gap humano-maquina** | 5.4 | 21.2 | 21.4 | 23.5 | 23.2 |

Lecturas clave:

- El mejor modelo, **DocQA + ELMo, alcanza solo 66.3 F1 en test de 2.0** frente a 89.5 humano: una brecha de **23.2 puntos**. El mismo modelo en 1.1 obtiene 85.8 F1, a solo **5.4 puntos** del humano.
- Los modelos existentes estan **mas cerca del baseline trivial (48.9 F1) que de los humanos**. SQuAD 2.0 reabre el espacio para mejorar.

**Negativas automaticas vs manuales.** Entrenando sobre 1.1 aumentado con TFIDF o RuleBased, el mejor puntaje sobre SQuAD 2.0 (67.6 F1) queda **15.4 puntos por debajo** del mejor sobre cualquier dataset automatico (89.6 F1 con RuleBased). Las negativas automaticas son **mucho mas faciles de detectar**.

**Distractores como trampa.** Aislando los falsos positivos (responder a una pregunta no respondible), maquinas y humanos por igual coinciden con la respuesta plausible **aproximadamente la mitad** de las veces (DocQA + ELMo: 54.9 EM, 69.2 F1; humano: 46.4 EM, 60.6 F1). Los distractores cumplen su funcion: son spans creibles capaces de hacer caer incluso a un humano atento.

---

## Tipos de unanswerable

Inspeccionando 100 negativos del dev set, el **93% son efectivamente no respondibles** (7% es ruido del dataset):

| Tipo | Descripcion | % |
|---|---|---:|
| **Negation** | Palabra de negacion insertada o removida. | 9% |
| **Antonym** | Uso de un antonimo. | 20% |
| **Entity Swap** | Entidad, numero o fecha reemplazada por otra. | 21% |
| **Mutual Exclusion** | Frase mutuamente excluyente con algo que si tiene respuesta. | 15% |
| **Impossible Condition** | Pide una condicion que nada en el parrafo satisface. | 4% |
| **Other Neutral** | El parrafo no implica ninguna respuesta. | 24% |
| **Answerable** | La pregunta si es respondible (ruido). | 7% |

La distribucion es **mucho mas diversa** que RuleBased (que solo cubriria algo de Antonym y Entity Swap). La categoria mas frecuente, "Other Neutral" (24%), abarca razonamiento que no encaja en patrones lexicos simples: justo lo que un detector de superficie no captura.

---

## Por que importa hoy

SQuAD 2.0 se volvio el benchmark de referencia para reading comprehension extractivo durante toda la era de los modelos preentrenados. Pocos meses despues, **[BERT (Devlin et al. 2018)](/papers/bert-devlin-2018)** y sus sucesores (RoBERTa, ALBERT, XLNet, SpanBERT) lo usaron como tarea estrella, cerrando la brecha de 23.2 puntos hasta superar el nivel humano hacia 2019-2020. Esto lo convirtio en el termometro canonico del salto que trajo el preentrenamiento masivo.

El aporte mas duradero es operacional: **en sistemas reales, abstenerse cuando la respuesta no esta respaldada es preferible a alucinar una respuesta plausible**. El patron de falsos positivos (responder con un span del tipo correcto pero no implicado por el texto) es exactamente el modo de falla de un **RAG mal calibrado**: recupera un pasaje relacionado y el LLM completa una respuesta que el pasaje no respalda. SQuAD 2.0 formaliza la distincion *answerability* contra *plausibility* y muestra que solo se aprende contra **negativos adversariales bien construidos**, no automaticos. La leccion se traduce directo a como construir conjuntos de evaluacion de "no respondible" para un sistema corporativo: si las preguntas negativas no comparten lexico ni tipo con los documentos, el benchmark sobreestima la capacidad de abstencion.

**Limitaciones.** El 7% de negativos muestreados son en realidad respondibles (techo ~93% de pureza). La proporcion ~1:1 es una decision de diseno, no una distribucion natural; el calibrado del umbral es sensible. Hereda parrafos de Wikipedia en ingles (transferencia a otros dominios no evaluada). La abstencion es binaria con umbral, sin grados de confianza. Y **no mide recuperacion**: SQuAD 2.0 siempre entrega el parrafo correcto, mientras que en un RAG real una fuente extra de "no respondible" es que la recuperacion falle.

---

## Conexion con la clase 24

La [Clase 24](/clases/clase-24) cubre Question Answering. SQuAD 2.0 conecta en varios puntos:

- **Extiende SQuAD**: la clase introduce SQuAD 1.1 como dataset fundacional de QA extractivo. SQuAD 2.0 es su sucesor directo y corrige su debilidad de diseno (respuesta siempre garantizada). Entender por que se necesito la version 2.0 es entender por que 1.1 no media comprension real.
- **Answerability como concepto central**: la clase trata QA real, donde no toda pregunta tiene respuesta en el corpus. La capacidad de abstenerse, el *know what you don't know*, separa un QA de demo de uno de produccion. Este paper le da nombre, dataset y metrica.
- **QA extractivo basado en BERT**: la clase muestra el QA moderno como prediccion de span de inicio/fin sobre las representaciones de BERT. SQuAD 2.0 anade una salida de probabilidad de "no answer" (tipicamente asociada al token `[CLS]`) y un umbral $\tau$. La arquitectura BERT-QA que ensena la clase resuelve, en la practica, SQuAD 2.0, y la logica de abstencion por umbral es la que implementan las cabezas de QA de la era BERT.
- **Conexion con RTE/NLI**: el paper enlaza *answerability* con *recognizing textual entailment*; decidir si el parrafo implica que cierto span es la respuesta es un problema de inferencia (entailment / contradiction / neutral).

---

## Notas y enlaces

- Paper (arXiv): https://arxiv.org/abs/1806.03822
- Dataset y leaderboard: https://rajpurkar.github.io/SQuAD-explorer/ (SQuAD 2.0 es el benchmark primario)
- Licencia: CC BY-SA 4.0. Premio: Best Short Paper Award, ACL 2018.
- Predecesor: Rajpurkar, Zhang, Lopyrev, Liang (2016), "SQuAD: 100,000+ Questions for Machine Comprehension of Text", EMNLP.

Ver fundamentos: [Question Answering](/fundamentos/question-answering) - [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension) - [Metricas de evaluacion de QA](/fundamentos/qa-evaluation-metrics).

Ver papers: [SQuAD 1.1 (Rajpurkar 2016)](/papers/squad-rajpurkar-2016) - [BERT (Devlin 2018)](/papers/bert-devlin-2018).

Ver clase: [Clase 24 -- Question Answering](/clases/clase-24).
