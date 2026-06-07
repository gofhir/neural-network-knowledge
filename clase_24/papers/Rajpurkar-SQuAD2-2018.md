# Know What You Don't Know: Unanswerable Questions for SQuAD (SQuAD 2.0)

## Metadata

| Campo | Valor |
|---|---|
| Título | Know What You Don't Know: Unanswerable Questions for SQuAD |
| Autores | Pranav Rajpurkar\*, Robin Jia\*, Percy Liang (\* primeros dos autores con contribución igual) |
| Afiliación | Computer Science Department, Stanford University |
| Venue | ACL 2018 (Association for Computational Linguistics) — Best Short Paper Award |
| arXiv | 1806.03822v1 [cs.CL], 11 de junio de 2018 |
| Nombre del dataset | SQuAD 2.0 (en la versión ACL se llamó SQuADRUn) |
| Licencia | CC BY-SA 4.0 |
| Reproducibilidad | Código, datos y experimentos en CodaLab (https://bit.ly/2rDHBgY) |
| Financiamiento | Facebook; R. Jia con NSF Graduate Research Fellowship (DGE-114747) |

Este es uno de los papers más citados en la historia reciente del NLP: definió el benchmark estándar para *reading comprehension* extractivo durante toda la era BERT y dio nombre a una capacidad que hoy es central en QA en producción y en RAG: saber abstenerse cuando no se conoce la respuesta.

## Contexto: SQuAD 1.1 estaba saturado

La comprensión lectora automática (*machine reading comprehension*) se había vuelto una tarea central del NLP hacia 2016-2018, impulsada por la aparición de grandes datasets etiquetados: CNN/Daily Mail (Hermann et al., 2015), WikiReading (Hewlett et al., 2016), MS MARCO (Nguyen et al., 2016), NewsQA (Trischler et al., 2017), TriviaQA (Joshi et al., 2017) y, sobre todo, SQuAD (Rajpurkar et al., 2016), el más usado de todos. Estos datasets habían desencadenado una carrera de arquitecturas: BiDAF (Seo et al., 2016), Mnemonic Reader (Hu et al., 2017), R-Net / Gated Self-Matching (Wang et al., 2017), DocumentQA (Clark y Gardner, 2017), FusionNet (Huang et al., 2018).

El problema era de su propio éxito. Para mediados de 2018, los mejores sistemas ya **superaban la exactitud humana en *exact match*** sobre SQuAD 1.1. El benchmark estaba efectivamente saturado: las mejoras se medían en décimas de punto y dejaban de ser informativas sobre el progreso real en comprensión del lenguaje.

El paper articula la crítica de fondo con precisión. El éxito en SQuAD 1.1 no implicaba comprensión genuina:

- Weissenborn et al. (2017) mostraron que los modelos podían hacerlo bien aprendiendo **heurísticas de contexto y de coincidencia de tipo** (*type-matching*): buscar el span del tipo correcto (una fecha si la pregunta es "cuándo", una persona si es "quién") más cercano a las palabras de la pregunta.
- Jia y Liang (2017) demostraron que el éxito en SQuAD 1.1 no garantizaba robustez frente a oraciones distractoras insertadas en el passage.

La causa raíz, según los autores, es estructural: **SQuAD 1.1 garantiza que la respuesta correcta siempre existe en el documento de contexto**. Bajo esa garantía, el modelo nunca necesita verificar que la respuesta está realmente *implicada* (*entailed*) por el texto. Solo necesita seleccionar el span que parece más relacionado con la pregunta. Es un sesgo de diseño que premia el *guessing* por coincidencia superficial: dado que siempre hay una respuesta, adivinar la más plausible es una estrategia sin penalización.

En términos prácticos para alguien que construye sistemas de QA: SQuAD 1.1 entrenaba modelos para **siempre responder algo**, lo que es exactamente el comportamiento que produce alucinaciones cuando el sistema se enfrenta a una pregunta cuya respuesta no está en sus documentos.

Conviene precisar el contraste entre las dos heurísticas que los modelos de SQuAD 1.1 aprendían. La heurística de **coincidencia de tipo** (*type matching*) explota la regularidad de que el tipo de la respuesta está determinado por la palabra interrogativa: "quién" → persona, "cuándo" → fecha, "cuánto" → cantidad. Un modelo que aprende esta correlación restringe el espacio de búsqueda a los spans del tipo correcto sin verificar nada más. La heurística de **coincidencia de contexto** premia el span rodeado por las palabras que más se solapan con la pregunta. Combinadas, estas dos heurísticas resuelven una fracción enorme de SQuAD 1.1 sin ningún razonamiento sobre el significado: encontrar la entidad del tipo correcto más cercana a las palabras clave de la pregunta. SQuAD 2.0 está diseñado precisamente para que ese par de heurísticas falle, porque la respuesta plausible —el span del tipo correcto, cerca de las palabras de la pregunta— está presente pero es incorrecta.

## La idea central: enseñar a abstenerse

SQuAD 2.0 combina los datos de SQuAD 1.1 con **53,775 preguntas nuevas, no respondibles (*unanswerable*)**, escritas adversarialmente por crowdworkers sobre los mismos párrafos. La consigna de diseño tiene dos condiciones explícitas:

1. **Relevancia**: la pregunta debe ser relevante al tema del párrafo.
2. **Existencia de una respuesta plausible**: debe existir en el contexto algún span del **mismo tipo** que el que la pregunta pide, aunque sea incorrecto.

Para tener éxito en SQuAD 2.0, un sistema ya no solo debe responder cuando es posible, sino también **determinar cuándo ninguna respuesta está respaldada por el párrafo y abstenerse**. La frase que da título al paper —"know what you don't know"— resume el cambio de objetivo: de *span selection* a *answerability* + *span selection*.

La Figura 1 del paper ilustra el mecanismo. Sobre un párrafo de la *Endangered Species Act* que describe leyes que "tuvieron bajo costo para la sociedad ... y poca oposición se levantó", las preguntas no respondibles son:

- *"Which laws faced significant opposition?"* → respuesta plausible (incorrecta): *later laws*. El párrafo dice exactamente lo contrario: poca oposición. Es una **trampa de negación/exclusión**.
- *"What was the name of the 1937 treaty?"* → respuesta plausible (incorrecta): *Bald Eagle Protection Act*. El párrafo menciona un tratado de 1937 y menciona la Bald Eagle Protection Act (de 1940), pero no le da nombre al tratado de 1937. Es un **entity swap / confusión de fechas**.

En ambos casos hay un span del tipo correcto ("una ley", "el nombre de un tratado") justo ahí, listo para que un modelo superficial lo agarre. Esa es la esencia adversarial del dataset.

## Construcción del dataset

Los autores definen *negative example* como un par (passage, pregunta no respondible).

**Plataforma y proceso.** Se usó la plataforma de crowdsourcing Daemo (Gaikwad et al., 2015). Cada tarea consistía en un artículo completo de SQuAD 1.1. Para cada párrafo, los trabajadores debían formular **hasta cinco preguntas imposibles de responder** basándose solo en ese párrafo, pero que (a) referenciaran entidades del párrafo y (b) tuvieran una respuesta plausible presente. Como inspiración, se les mostraban las preguntas reales de SQuAD 1.1 para ese párrafo, lo que reforzaba que las no respondibles se parecieran a las respondibles. Tiempo objetivo: 7 minutos por párrafo; pago: USD 10.50 por hora. La interfaz les pedía primero escribir la pregunta no respondible y luego resaltar la respuesta plausible en el párrafo.

**Filtrado de ruido.** Se eliminaron las preguntas de trabajadores que escribieron 25 o menos preguntas por artículo (señal de que abandonaron por no entender la tarea). Este filtro se aplicó tanto a los datos nuevos como a las preguntas respondibles heredadas de SQuAD 1.1.

**Splits.** Se reusó la **misma partición de artículos** que SQuAD 1.1, combinando datos viejos y nuevos en cada split. En dev y test se removieron los artículos para los que no se recolectaron preguntas no respondibles. Esto dejó una proporción **aproximadamente 1:1** de respondibles a no respondibles en dev y test, mientras que en train hay **aproximadamente el doble** de respondibles que de no respondibles. Este detalle importa para el calibrado del umbral (ver más abajo): la distribución de negativos difiere entre entrenamiento y test.

### Estadísticas: SQuAD 2.0 vs SQuAD 1.1

| Split | Métrica | SQuAD 1.1 | SQuAD 2.0 |
|---|---|---:|---:|
| **Train** | Total de ejemplos | 87,599 | 130,319 |
| | Ejemplos negativos | 0 | 43,498 |
| | Artículos totales | 442 | 442 |
| | Artículos con negativos | 0 | 285 |
| **Development** | Total de ejemplos | 10,570 | 11,873 |
| | Ejemplos negativos | 0 | 5,945 |
| | Artículos totales | 48 | 35 |
| | Artículos con negativos | 0 | 35 |
| **Test** | Total de ejemplos | 9,533 | 8,862 |
| | Ejemplos negativos | 0 | 4,332 |
| | Artículos totales | 46 | 28 |
| | Artículos con negativos | 0 | 28 |

(El total de 53,775 preguntas no respondibles del resumen es la suma a través de los splits originales antes del filtrado; las tablas reportan los negativos retenidos por split: 43,498 + 5,945 + 4,332.)

**Calidad humana.** Para confirmar que el dataset es limpio, se contrataron crowdworkers adicionales para responder todas las preguntas de dev y test. Se les mostraba el artículo completo y, por cada párrafo, todas las preguntas asociadas con las respondibles y no respondibles **mezcladas**. Por cada pregunta debían o resaltar la respuesta o marcarla como no respondible, sabiendo que cada párrafo tendría de ambos tipos. Tiempo: un minuto por pregunta. Para reducir ruido se recolectaron **múltiples respuestas humanas por pregunta** (promedio 4.8) y se eligió la final por **voto mayoritario**, rompiendo empates a favor de responder y prefiriendo respuestas más cortas. Nota metodológica importante: en SQuAD 1.1, Rajpurkar et al. (2016) evaluaron a un **único** humano, por lo que probablemente subestimaron la exactitud humana; aquí la estimación es más alta y más robusta.

## Por qué las negativas son difíciles: contraste con negativas automáticas

El núcleo conceptual del paper es que **no cualquier pregunta no respondible sirve**. Una negativa trivial puede detectarse con heurísticas léxicas baratas, y entrenar contra ella no enseña nada sobre *answerability* real. El paper revisa los enfoques previos para mostrar por qué fallan:

- **Distant supervision (Zero-shot RE, Levy et al. 2017)**: el 65% de sus negativos **no tienen respuesta plausible**, lo que los hace fáciles de identificar — basta el *type-matching* para descartarlos.
- **TriviaQA (Joshi et al. 2017)**: genera negativos al recuperar documentos web que no contienen la respuesta, pero los **excluye** del dataset final.
- **TFIDF (Clark y Gardner, 2017)**: emparejan preguntas existentes de SQuAD con *otros* párrafos del mismo artículo según solapamiento TF-IDF. El problema: con un conjunto pequeño de contextos posibles, los párrafos recuperados suelen ser poco relevantes a la pregunta, por lo que un detector basado en solapamiento de palabras los separa fácil.
- **NewsQA (Trischler et al. 2017)**: produce no respondibles porque los crowdworkers escriben preguntas viendo solo un resumen. Pero solo el **9.5%** resultan no respondibles (no escala), algunas están mal anotadas y otras son fuera de alcance (preguntas de resumen). También las excluyen del dataset final.
- **RuleBased (Jia y Liang, 2017)**: edición por reglas de preguntas de SQuAD — reemplazo de entidades/números por palabras similares y de sustantivos/adjetivos por antónimos de WordNet. Poco diversas.
- **Datasets de selección de oraciones** (QASent, WikiQA): los baselines léxicos son altamente competitivos (Yih et al. 2013), señal de que las negativas se distinguen por superficie. WikiQA además es pequeño (3,047 preguntas, 1,473 respuestas).
- **Multiple choice** (MCTest, RACE): tienen opción "none of the above", pero las opciones no suelen estar disponibles en sistemas reales, y el estilo (fill-in-the-blank, interpretación) difiere del extractivo.

La diferencia clave de SQuAD 2.0: como las negativas se construyen para **compartir léxico y tipo con el passage** (relevancia + respuesta plausible), no pueden ser filtradas por solapamiento de palabras ni por *type-matching*. El modelo está obligado a razonar sobre si el texto realmente **implica** la respuesta. Esto conecta directamente con *recognizing textual entailment* (RTE): decidir si una hipótesis está implicada, contradicha o es neutral respecto de una premisa.

## Métricas ajustadas: EM y F1 con la opción "no answer"

Siguiendo a Rajpurkar et al. (2016), se reportan **Exact Match (EM)** y **F1** promedio. La extensión clave para SQuAD 2.0 es el manejo de los ejemplos negativos:

> Para ejemplos negativos, abstenerse recibe un puntaje de **1**, y cualquier otra respuesta recibe **0**, tanto para EM como para F1.

Es decir, el espacio de salida del modelo incluye una opción explícita de "no answer". La función de evaluación se vuelve, conceptualmente:

$$
\text{score}(q) =
\begin{cases}
1 & \text{si } q \text{ es negativa y el modelo abstiene} \\
0 & \text{si } q \text{ es negativa y el modelo responde algo} \\
\text{EM/F1 estándar} & \text{si } q \text{ es positiva (con respecto a los gold spans)}
\end{cases}
$$

Para una pregunta positiva, el F1 se calcula como el solapamiento de tokens entre la predicción y la respuesta de referencia (la mejor sobre las múltiples respuestas humanas), igual que en SQuAD 1.1.

**Cómo decide el modelo abstenerse.** Los modelos evaluados predicen, además de la distribución sobre spans, una **probabilidad de que la pregunta sea no respondible**. En test, el modelo **abstiene cuando esa probabilidad supera un umbral** $\tau$. El umbral se ajusta por separado para cada modelo sobre el dev set, eligiendo el $\tau$ que **maximiza F1 en dev**. Los autores observan que esto funciona algo mejor que tomar el simple argmax de la predicción, posiblemente debido a las **distintas proporciones de negativos en entrenamiento (2:1) y test (1:1)** — un punto fino de calibración que cualquiera que despliegue un clasificador de abstención debe tener presente: el umbral óptimo depende del *prior* de la clase en producción, que rara vez coincide con el de entrenamiento.

Una referencia útil de la tabla principal: un baseline que **siempre se abstiene** obtiene **48.9 F1 en test**. Eso fija el piso "trivial" del benchmark. El número no es casual: con una proporción ~1:1 de negativas, abstenerse siempre acierta en la mitad de los casos (las negativas, score 1) y falla en la otra mitad (las positivas, score 0), lo que da algo cercano al 50%. El que ese baseline mudo quede tan cerca de los modelos reales (66.3 F1 del mejor sistema) es la métrica más elocuente del paper: la mayor parte de la distancia entre "no hacer nada" y "comprender" sigue sin recorrerse. En contraste, en SQuAD 1.1 un baseline así no tiene sentido, porque todas las preguntas son respondibles y abstenerse siempre da 0.

Vale la pena notar que esta formulación de la métrica tiene una propiedad de diseño deliberada: **castiga simétricamente la sobre-confianza y la sub-confianza**. Un modelo que nunca se abstiene paga el costo completo en todas las negativas; uno que se abstiene de más paga en todas las positivas que sí podía responder. El umbral $\tau$ es el botón que equilibra esos dos tipos de error (falsos positivos contra falsos negativos de abstención), y maximizar F1 sobre dev equivale a encontrar el punto de operación que mejor los balancea para la proporción de clases del benchmark. Esta es la misma decisión que enfrenta cualquier sistema de QA en producción que deba elegir un umbral de confianza sobre un *prior* real de preguntas sin respuesta.

## Modelos baseline y resultados

Se evaluaron tres arquitecturas existentes, todas con capacidad de "no answer":

- **BNA (BiDAF-No-Answer)**, propuesto por Levy et al. (2017).
- **DocQA (DocumentQA No-Answer)** de Clark y Gardner (2017).
- **DocQA + ELMo**, la versión con representaciones contextualizadas ELMo (Peters et al., 2018).

### Resultados principales (Tabla 3)

| Sistema | SQuAD 1.1 test EM | SQuAD 1.1 test F1 | SQuAD 2.0 dev EM | SQuAD 2.0 dev F1 | SQuAD 2.0 test EM | SQuAD 2.0 test F1 |
|---|---:|---:|---:|---:|---:|---:|
| BNA | 68.0 | 77.3 | 59.8 | 62.6 | 59.2 | 62.1 |
| DocQA | 72.1 | 81.0 | 61.9 | 64.8 | 59.3 | 62.3 |
| DocQA + ELMo | 78.6 | 85.8 | 65.1 | 67.6 | 63.4 | 66.3 |
| **Human** | 82.3 | 91.2 | 86.3 | 89.0 | 86.9 | 89.5 |
| **Human–Machine Gap** | 3.7 | 5.4 | 21.2 | 21.4 | 23.5 | 23.2 |

Lecturas clave:

- El mejor modelo, **DocQA + ELMo, alcanza solo 66.3 F1 en test de SQuAD 2.0**, frente a una exactitud humana de **89.5 F1** — una brecha de **23.2 puntos**.
- El mismo modelo en SQuAD 1.1 obtiene **85.8 F1**, a solo **5.4 puntos** de los humanos (91.2).
- El dato del resumen ("un sistema neuronal fuerte que obtiene 86% F1 en SQuAD 1.1 solo logra 66% en SQuAD 2.0") corresponde precisamente a DocQA + ELMo: 85.8 → 66.3.
- Los modelos existentes están **más cerca del baseline trivial (48.9 F1) que de los humanos**. Esto es la evidencia central de que SQuAD 2.0 reabre el espacio para mejorar: la saturación de SQuAD 1.1 quedó atrás.

### Negativas automáticas vs manuales (Tabla 4)

Para verificar que la dificultad viene del diseño adversarial manual y no de cualquier negativa, entrenaron y testearon las tres arquitecturas sobre SQuAD 1.1 aumentado con TFIDF o RuleBased. Para que la comparación fuera justa, generaron las negativas automáticas solo sobre los 285 artículos para los que SQuAD 2.0 tiene no respondibles, y testearon agregando negativas en proporción ~1:1.

| Sistema | SQuAD 1.1 + TFIDF EM | + TFIDF F1 | + RuleBased EM | + RuleBased F1 | SQuAD 2.0 dev EM | SQuAD 2.0 dev F1 |
|---|---:|---:|---:|---:|---:|---:|
| BNA | 72.7 | 76.6 | 80.1 | 84.8 | 59.8 | 62.6 |
| DocQA | 75.6 | 79.2 | 80.8 | 84.8 | 61.9 | 64.8 |
| DocQA + ELMo | 79.4 | 83.0 | 85.7 | 89.6 | 65.1 | 67.6 |

El mejor puntaje sobre SQuAD 2.0 (67.6 F1) está **15.4 puntos por debajo** del mejor puntaje sobre cualquiera de los dos datasets automáticos (89.6 F1 con RuleBased). Conclusión: las negativas automáticas son **mucho más fáciles de detectar**. RuleBased, en particular, deja a los modelos casi al nivel de SQuAD 1.1 — apenas perturba la tarea.

### Las respuestas plausibles como distractores (Tabla 5)

Midieron con qué frecuencia los sistemas eran engañados a responder exactamente la respuesta plausible (incorrecta) provista por los crowdworkers. Aislaron los **falsos positivos** (casos donde el sistema respondió a una pregunta no respondible) y midieron EM/F1 entre la predicción y la respuesta plausible:

| Sistema | EM | F1 |
|---|---:|---:|
| BNA | 48.6 | 63.0 |
| DocQA | 55.0 | 69.9 |
| DocQA + ELMo | 54.9 | 69.2 |
| Human | 46.4 | 60.6 |

Para máquinas y humanos por igual, **aproximadamente la mitad** de las respuestas equivocadas a preguntas no respondibles coinciden exactamente con la respuesta plausible. Esto confirma que los distractores cumplen su función: son spans creíbles, del tipo correcto, capaces de hacer caer incluso a un humano atento.

## Análisis de tipos de unanswerable (Tabla 1)

Los autores inspeccionaron manualmente 100 ejemplos negativos al azar del dev set y definieron categorías de fenómenos. Notablemente, encontraron que el **93% de los negativos muestreados son efectivamente no respondibles** (el 7% restante es ruido del dataset: la categoría "Answerable").

| Tipo de razonamiento | Descripción | Ejemplo (S = oración, Q = pregunta) | % |
|---|---|---|---:|
| **Negation** | Palabra de negación insertada o removida. | S: "Several hospital pharmacies have decided to outsource high risk preparations..." Q: "What types of pharmacy functions have **never** been outsourced?" | 9% |
| **Antonym** | Uso de un antónimo. | S: "the extinction of the dinosaurs... allowed the tropical rainforest to **spread out**..." Q: "The extinction of what led to the **decline** of rainforests?" | 20% |
| **Entity Swap** | Entidad, número o fecha reemplazada por otra. | S: "...the 9–88 cm as projected... in its **Third** Assessment Report." Q: "What was the projection... in the **fourth** assessment report?" | 21% |
| **Mutual Exclusion** | Palabra/frase mutuamente excluyente con algo para lo que sí hay respuesta. | S: "BSkyB... waiv[ed] the charge for subscribers whose package included two or more premium channels." Q: "What service did BSkyB give away for free **unconditionally**?" | 15% |
| **Impossible Condition** | Pide una condición que nada en el párrafo satisface. | S: "Union forces left Jacksonville... Union forces then retreated to Jacksonville and held the city for the remainder of the war." Q: "After what battle did Union forces leave Jacksonville **for good**?" | 4% |
| **Other Neutral** | Otros casos donde el párrafo no implica ninguna respuesta. | (varios) | 24% |
| **Answerable** | La pregunta sí es respondible (ruido del dataset). | S: "Schuenemann et al. concluded in 2011 that the Black Death... was caused by a variant of Y. pestis..." Q: "Who **discovered** Y. pestis?" | 7% |

La distribución muestra **amplia diversidad**, mucho mayor que RuleBased (que solo cubriría algo de Antonym y Entity Swap). La categoría más frecuente, "Other Neutral" (24%), abarca razonamiento que no encaja en patrones léxicos simples — exactamente lo que un detector de superficie no captura. El conjunto Antonym (20%) + Entity Swap (21%) + Mutual Exclusion (15%) requiere razonamiento sobre el significado, no solo sobre el solapamiento de palabras.

## Limitaciones

El paper es honesto sobre sus bordes, aunque siendo un *short paper* no dedica una sección formal a limitaciones:

- **Ruido residual**: el 7% de los negativos muestreados son en realidad respondibles. Es un techo de ~93% de pureza en la clase negativa.
- **Dependencia del crowdsourcing**: la calidad y diversidad de las no respondibles depende de la consigna (referenciar entidades, dejar respuesta plausible) y del filtrado de trabajadores. El sesgo de los crowdworkers hacia ciertos patrones (negación, antónimos) es un artefacto posible.
- **Distribución de clases artificial**: la proporción ~1:1 en dev/test y 2:1 en train es una decisión de diseño, no un reflejo de ninguna distribución natural de preguntas no respondibles. El calibrado del umbral es sensible a esto, y un sistema en producción enfrentará proporciones distintas.
- **Dominio acotado**: hereda los párrafos de SQuAD 1.1 (Wikipedia en inglés). La transferencia a otros dominios (clínico, legal, conversacional) no está evaluada.
- **Abstención binaria**: la tarea reduce "answerability" a una decisión binaria con umbral. No modela grados de confianza ni la distinción entre "no está en el texto" y "el texto es ambiguo".
- **Distractores fáciles de igualar**: el hecho de que ~50% de los falsos positivos coincidan con la respuesta plausible sugiere que el dataset, en parte, mide la resistencia a un distractor específico construido por humanos, no la *answerability* en abstracto.
- **No mide recuperación**: SQuAD 2.0 entrega siempre el párrafo correcto junto a la pregunta. La tarea es de comprensión sobre un contexto dado, no de *open-domain QA* donde el sistema debe primero recuperar el pasaje. En un RAG real, una fuente adicional de "no respondible" es que la recuperación falle; ese eslabón no está en el benchmark, que asume el contexto ya provisto.

## Impacto

SQuAD 2.0 se volvió el benchmark de referencia para *reading comprehension* extractivo durante toda la era de los modelos preentrenados. Cronología relevante:

- ELMo (Peters et al., 2018) ya aparece como el componente que más ayuda entre los baselines del propio paper (DocQA + ELMo es el mejor modelo).
- Pocos meses después, **BERT (Devlin et al., 2018)** y sus sucesores (RoBERTa, ALBERT, XLNet, SpanBERT) usaron SQuAD 2.0 como tarea estrella de evaluación, cerrando la brecha de 23.2 puntos hasta superar el nivel humano hacia 2019-2020. Esto convirtió a SQuAD 2.0 en el termómetro canónico del salto de capacidad que trajo el preentrenamiento masivo.
- La **calibración y la abstención** dejaron de ser una curiosidad y pasaron a ser una capacidad de primera clase. El paper popularizó la idea operacional de que un sistema de QA debe poder responder "no sé".

Para un ingeniero que construye QA en producción o RAG, este es el aporte más duradero: **en sistemas reales, abstenerse cuando la respuesta no está respaldada es preferible a alucinar una respuesta plausible**. El patrón de falsos positivos (responder con un span del tipo correcto pero no implicado por el texto) es exactamente el modo de falla de un RAG mal calibrado: recupera un pasaje relacionado y el LLM "completa" una respuesta que el pasaje no respalda. SQuAD 2.0 formaliza precisamente esa distinción — *answerability* contra *plausibility* — y muestra que requiere entrenar explícitamente contra negativos adversariales, no automáticos. La lección de la Tabla 4 (negativas triviales no enseñan abstención real) se traduce directo a cómo construir conjuntos de evaluación de "no respondible" para un sistema corporativo: si las preguntas negativas no comparten léxico ni tipo con los documentos, el benchmark sobreestima la capacidad de abstención.

## Conexión con la Clase 24

La Clase 24 cubre Question Answering. SQuAD 2.0 conecta en varios puntos del material del profesor:

- **Extiende SQuAD (slide 28)**: la clase introduce SQuAD 1.1 como el dataset fundacional de QA extractivo. SQuAD 2.0 es su sucesor directo y corrige su debilidad de diseño (respuesta siempre garantizada). Entender por qué se necesitó la versión 2.0 es entender por qué SQuAD 1.1 no medía comprensión real.
- **Answerability como concepto central**: la clase trata QA real, donde no toda pregunta tiene respuesta en el corpus. La capacidad de abstenerse —el "know what you don't know"— es justamente lo que separa un QA de demo de un QA de producción. Este paper le da nombre, dataset y métrica a ese concepto.
- **QA extractivo basado en BERT (slides 33-36)**: la clase muestra el QA extractivo moderno como predicción de span de inicio/fin sobre las representaciones de BERT. SQuAD 2.0 añade una salida adicional (la probabilidad de "no answer", típicamente asociada al token `[CLS]`) y un umbral $\tau$ sobre ella. Es decir, la arquitectura BERT-QA que enseña la clase resuelve, en la práctica, SQuAD 2.0 — no solo SQuAD 1.1 — y la lógica de abstención por umbral descrita aquí es exactamente la que implementan las cabezas de QA de la era BERT.
- **Conexión con RTE/NLI**: el propio paper enlaza *answerability* con *recognizing textual entailment*: decidir si el párrafo *implica* que cierto span es la respuesta es un problema de inferencia (entailment / contradiction / neutral), lo que tiende un puente entre QA extractivo y las tareas de inferencia natural del lenguaje (SNLI, SICK) que aparecen en el currículo de NLP.

## Notas y enlaces

- **Paper (arXiv)**: https://arxiv.org/abs/1806.03822
- **Dataset y leaderboard**: https://rajpurkar.github.io/SQuAD-explorer/ (SQuAD 2.0 es el benchmark primario del leaderboard oficial)
- **Reproducibilidad (CodaLab)**: https://bit.ly/2rDHBgY
- **Licencia**: CC BY-SA 4.0
- **Predecesor**: Rajpurkar, Zhang, Lopyrev, Liang (2016), "SQuAD: 100,000+ Questions for Machine Comprehension of Text", EMNLP — define SQuAD 1.1 y las métricas EM/F1.
- **Trabajos relacionados citados como contraste de negativas**: Levy et al. (2017, Zero-shot RE / BNA), Clark y Gardner (2017, DocumentQA / TFIDF), Jia y Liang (2017, adversarial examples / RuleBased), Trischler et al. (2017, NewsQA), Joshi et al. (2017, TriviaQA).
- **Componente clave de los baselines**: Peters et al. (2018), ELMo — la mejora más grande entre los modelos evaluados.
- **Premio**: Best Short Paper Award, ACL 2018.

Resumen ejecutivo para Roberto: SQuAD 2.0 no agrega una arquitectura, agrega un **diseño de dataset** que captura la diferencia entre "hay un span del tipo correcto" (plausibilidad) y "el texto realmente implica esta respuesta" (answerability). La evidencia numérica —brecha humano-máquina de 5.4 → 23.2 puntos al pasar de 1.1 a 2.0, y 15.4 puntos de diferencia entre negativas manuales y automáticas— es el argumento de que abstenerse correctamente es difícil y que solo se aprende contra negativos adversariales bien construidos. Es la lección directa para construir y evaluar la capacidad de "no sé" en cualquier sistema de QA o RAG en producción.
