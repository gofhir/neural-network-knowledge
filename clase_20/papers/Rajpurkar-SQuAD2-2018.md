# Análisis interno — Rajpurkar, Jia & Liang (2018) "Know What You Don't Know: Unanswerable Questions for SQuAD"

> Documento complementario al material público del site sobre Question Answering y a la práctica del lab 20 (`XLNetForQuestionAnswering`). Aquí se profundiza en aspectos que el resumen ejecutivo del paper deja implícitos: la motivación adversarial heredada de Jia & Liang 2017, la taxonomía completa de preguntas no-respondibles, la métrica EM/F1 ajustada, los baselines BNA y DocQA del paper, cómo modelos posteriores (BERT, XLNet, RoBERTa, ALBERT, DeBERTa) atacan el problema, y la conexión directa con el head `answer_class` que aparece en el warning del lab 20.

- **Paper**: Rajpurkar, Jia, Liang. *Know What You Don't Know: Unanswerable Questions for SQuAD*. arXiv:1806.03822v1 (11 Jun 2018). ACL 2018 — **Best Short Paper Award**.
- **Autores**: Pranav Rajpurkar, Robin Jia, Percy Liang (Stanford NLP). Los primeros dos autores contribuyeron igualmente.
- **Datos y leaderboard**: `https://rajpurkar.github.io/SQuAD-explorer/`. Licencia CC BY-SA 4.0.
- **PDF local**: [Rajpurkar-SQuAD2-2018.pdf](Rajpurkar-SQuAD2-2018.pdf).

---

## 1. Tesis del paper: los modelos QA "trampean" porque siempre asumen que hay respuesta

La tesis central de SQuAD 2.0 es una crítica metodológica al diseño original de SQuAD 1.1 (Rajpurkar et al. 2016) y, por extensión, a casi todos los datasets de extractive reading comprehension previos a 2018. El argumento es simple y demoledor: si un dataset de QA garantiza que toda pregunta tiene respuesta dentro del contexto, entonces el modelo no necesita resolver el problema de comprensión que el dataset pretende medir. Solo necesita resolver un problema más fácil: dado un párrafo y una pregunta, encontrar el span de tokens que se ve más relacionado con la pregunta.

Esta crítica se sostiene en dos observaciones empíricas hechas en 2017 que el paper cita como motivación directa:

1. **Weissenborn, Wiese & Seiffe (CoNLL 2017)** demostraron que modelos QA aparentemente sofisticados sobre SQuAD 1.1 estaban explotando heurísticas de **context-matching** (palabras de la pregunta que aparecen cerca de la respuesta) y **type-matching** (la respuesta es del tipo correcto: una persona para `who`, un lugar para `where`, una fecha para `when`). Mostraron que un baseline neuronal extremadamente simple — sin atención bidireccional, sin gating, sin self-attention — alcanzaba performance comparable al estado del arte. La conclusión: los modelos no estaban aprendiendo comprensión, sino patrones superficiales.

2. **Jia & Liang (EMNLP 2017)** —el mismo Robin Jia que es segundo autor del paper de SQuAD 2.0— mostraron que insertar **oraciones adversariales** (gramaticalmente correctas, semánticamente irrelevantes, pero conteniendo entidades del tipo correcto) al final del párrafo hacía colapsar el F1 de modelos SOTA de ~75 F1 a ~36 F1. Esto era una prueba decisiva de que los modelos no estaban verificando que el span elegido **realmente respondiera** la pregunta; solo estaban eligiendo el span con mayor solapamiento léxico.

La pregunta de investigación que Rajpurkar, Jia y Liang formulan en este paper es la consecuencia natural de esas dos observaciones: **¿pueden los modelos QA decir "no hay respuesta" cuando efectivamente no la hay?** Si la respuesta es no, entonces todas las métricas de SQuAD 1.1 están sobreestimando lo que los modelos realmente comprenden. Si la respuesta es sí, hay que construir un benchmark donde esto se mida directamente.

El paper opta por el camino constructivo: construir un dataset donde aproximadamente el **50% de las preguntas no tengan respuesta**, donde esas preguntas estén escritas adversarialmente para parecer respondibles, y donde el modelo deba aprender a abstenerse. La hipótesis empírica: si un modelo SOTA en SQuAD 1.1 (~86 F1) cae a ~66 F1 en SQuAD 2.0, mientras el humano se mantiene cerca de 89.5 F1, entonces el gap de 23.2 puntos cuantifica exactamente cuánto de la performance previa era "trampa" — heurísticas de matching que se rompen cuando la pregunta no tiene respuesta.

El resultado experimental confirma la hipótesis con fuerza. El modelo DocQA + ELMo, que es la mejor variante del paper, alcanza 85.8 F1 en SQuAD 1.1 (a 5.4 puntos del humano, gap considerado "casi cerrado" en 2018) y solo 66.3 F1 en SQuAD 2.0 (a 23.2 puntos del humano, gap "enorme"). El benchmark se vuelve, de un día para otro, varias órdenes de magnitud más difícil. Para contexto: en SQuAD 1.1, modelos basados en BiDAF habían superado el baseline de la BiLSTM trivial por ~15 F1 a lo largo de 18 meses de progreso de la literatura. En SQuAD 2.0, **toda** esa mejora se evapora cuando se introduce la posibilidad de abstención.

La consecuencia metodológica es profunda. Antes de SQuAD 2.0, "saber" en QA era operacionalizado como "extraer el span correcto cuando existe". Después de SQuAD 2.0, "saber" se redefine como "extraer el span correcto cuando existe **y** abstenerse cuando no existe". La distinción no es semántica: cambia completamente la naturaleza de lo que se entrena y se evalúa. Un modelo que confunda "respondible" con "no respondible" es ahora penalizado simétricamente; un modelo que prefiera siempre dar una respuesta (porque hacerlo le sumaba puntos en SQuAD 1.1) ahora paga ese sesgo. La elección de un threshold de abstención se vuelve un hiperparámetro tan relevante como la arquitectura.

Esta reformulación anticipa, **cinco años antes**, las preocupaciones contemporáneas sobre **alucinación** en LLMs. Un LLM que responde con confianza una pregunta para la cual no tiene información correcta está cometiendo exactamente el error que SQuAD 2.0 castiga: producir un output plausible donde correspondería abstención. Aunque el paper se inscribe explícitamente en extractive QA (donde el espacio de outputs está restringido a spans del contexto), la noción de "knowing what you don't know" se vuelve, en retrospectiva, una de las articulaciones más tempranas del problema de calibración epistémica que dominaría la conversación de LLMs desde 2022 en adelante.

Es importante notar también qué **no** plantea el paper. No afirma que SQuAD 2.0 mide comprensión profunda, ni que un modelo que la resuelva entenderá lenguaje. Lo único que afirma es que SQuAD 1.1 sobrestima sistemáticamente, que esa sobreestimación es del orden de 20+ puntos F1 para una clase amplia de modelos, y que un benchmark con preguntas no-respondibles adversariales captura un componente importante de comprensión que SQuAD 1.1 ignora. La modestia de las afirmaciones es lo que permite que el paper envejezca bien: no se ha demostrado falso en ningún punto, y modelos posteriores que "superaron" SQuAD 2.0 (RoBERTa, ALBERT, DeBERTa) lo hicieron incorporando precisamente el componente que el paper exigía — un mecanismo de abstención calibrado, no más capacidad bruta de matching.

---

## 2. Limitaciones de SQuAD 1.1 y el contexto adversarial

### 2.1 SQuAD 1.1 — recap del problema

SQuAD 1.1 (Rajpurkar et al. 2016, EMNLP) fue el dataset que detonó la era moderna de extractive QA. 100K+ pares (pregunta, párrafo, respuesta-span) sobre 536 artículos de Wikipedia, anotados por crowdworkers. Cada respuesta es un span literal del párrafo. La métrica es Exact Match (EM, 1 si el string predicho es idéntico al gold, 0 si no) y F1 a nivel de token. Por convención, se promedia sobre múltiples respuestas humanas y se reporta el máximo F1.

En los dos años siguientes a su publicación (2016-2018), SQuAD 1.1 saturó. La progresión de modelos fue acelerada:

| Año | Modelo | F1 dev |
|---|---|---|
| 2016 | BiDAF (Seo et al.) | 77.3 |
| 2017 | DrQA (Chen et al.) | 78.8 |
| 2017 | R-Net (MSRA) | 79.5 |
| 2017 | DocumentQA (Clark & Gardner) | 81.0 |
| 2017 | Reinforced Mnemonic Reader (Hu et al.) | 81.8 |
| 2018 | QANet (Yu et al.) | 84.6 |
| 2018 | DocQA + ELMo (Peters et al.) | 85.8 |
| 2018 | BERT-large single | 90.9 |

El humano single-annotator estaba en ~82 F1 (Rajpurkar et al. 2016, que reportaron una sola respuesta humana por pregunta) y el humano agregado en ~91 F1 (estimado a posteriori con anotaciones múltiples). Para mediados de 2018, los modelos top estaban dentro de 5 F1 del humano, y BERT-large los superaría meses después. El consenso de la comunidad era que SQuAD 1.1 estaba virtualmente resuelto.

Pero esta narrativa de progreso se basaba en una suposición tácita que el dataset hacía explícita: **todas las preguntas tienen respuesta en el párrafo**. Esto es razonable para un benchmark inicial (simplifica anotación, define una métrica clara), pero introduce un sesgo sistemático: el modelo nunca tiene que decidir si la pregunta es respondible. Su único trabajo es elegir entre los spans del párrafo cuál se ve más relacionado con la pregunta. Como Weissenborn et al. (2017) demostraron, esto puede lograrse mayormente con context-matching + type-matching, sin verificación de entailment.

### 2.2 Adversarial SQuAD (Jia & Liang 2017) — el precedente directo

El paper de 2017 de Robin Jia y Percy Liang —los mismos autores de SQuAD 2.0— es el antecedente que da fuerza retórica al nuevo dataset. Lo que hicieron fue construir oraciones adversariales con dos requisitos:

1. **No contradicen** ninguna información del párrafo original (preservan la verdad).
2. **Comparten muchas palabras** con la pregunta (alta superposición léxica).

Ejemplo del paper original: el párrafo dice "Peyton Manning became the oldest quarterback ever to play in a Super Bowl at age 39". La pregunta es "What is the name of the quarterback who was 38 in Super Bowl XXXIII?". La respuesta correcta es "John Elway" (que aparece más adelante en el párrafo). El ataque inserta al final del párrafo: "Quarterback Jeff Dean had jersey number 37 in Champ Bowl XXXIV". Esta oración es irrelevante, no contradice nada, y comparte palabras con la pregunta. Modelos SOTA caen al span "Jeff Dean" porque tiene más solapamiento con la pregunta adversarial que el span "John Elway".

El resultado de Jia & Liang 2017: F1 cae de ~75 a ~36 en 16 modelos SOTA distintos. La generalidad del ataque demostraba que era un problema arquitectónico, no de un modelo particular. La crítica implícita: los modelos no están aprendiendo el sentido de la pregunta, están aprendiendo a buscar la oración del párrafo con más solapamiento léxico, y luego extraer una entidad del tipo correcto.

Sin embargo, el setup de Jia & Liang 2017 tenía una limitación: era un **test set adversarial** evaluado sobre modelos entrenados en SQuAD 1.1 estándar. Cuando los autores trataron de **entrenar** modelos con ejemplos adversariales similares, descubrieron que esos modelos sí aprendían a resistir el ataque. Es decir, el ataque era efectivo contra modelos que no lo habían visto, pero no era un problema fundamental imposible de resolver dentro del paradigma de SQuAD. Esto motivó la búsqueda de un benchmark donde los ejemplos adversariales fueran genuinamente difíciles incluso para modelos entrenados sobre ellos.

### 2.3 La idea: combinar preguntas respondibles con preguntas no-respondibles

SQuAD 2.0 da el salto conceptual. En vez de modificar preguntas existentes con reglas (la estrategia de Jia & Liang 2017, llamada `RULE BASED` en el paper actual) o emparejar preguntas con párrafos aleatorios (`T F I DF`, la estrategia de Clark & Gardner 2017), los autores piden a crowdworkers humanos que **escriban preguntas nuevas** que cumplan dos requisitos:

1. **Relevantes al párrafo**: usan entidades, eventos, conceptos mencionados explícitamente.
2. **No respondibles**: la respuesta no está en el párrafo, aunque el párrafo contenga un span "plausible" del tipo correcto.

Los anotadores ven la pregunta original de SQuAD 1.1 sobre el párrafo como inspiración, lo que las induce a escribir preguntas estructuralmente similares. El resultado son preguntas que un modelo basado en matching tiene casi garantizado fallar: usan vocabulario del párrafo, tienen estructura sintáctica plausible, y existe un span en el contexto que un type-matcher seleccionaría como respuesta. Solo entendiendo el contenido del párrafo se puede determinar que la pregunta no tiene respuesta.

La hipótesis empírica subyacente: las preguntas humanas adversariales son **cualitativamente más difíciles** que las generadas automáticamente. El paper la verifica en la Tabla 4 (Sección 5.3): mismo modelo, entrenado y testeado en SQuAD 1.1 + `T F I DF` da 83.0 F1; en SQuAD 1.1 + `RULE BASED` da 89.6 F1; en SQuAD 2.0 da 67.6 F1. La diferencia de 22 F1 entre SQuAD 2.0 y el dataset rule-based confirma que las preguntas escritas por humanos son sustancialmente más adversariales.

---

## 3. Construcción del dataset

### 3.1 Composición global

SQuAD 2.0 mantiene todas las preguntas respondibles de SQuAD 1.1 y añade 53,775 preguntas no-respondibles. Las nuevas preguntas se distribuyen sobre los mismos 442 artículos de Wikipedia que SQuAD 1.1, garantizando comparabilidad directa de los splits. Estadísticas oficiales (Tabla 2 del paper):

|   | SQuAD 1.1 | SQuAD 2.0 |
|---|---|---|
| **Train — total examples** | 87,599 | 130,319 |
| Train — negative examples | 0 | 43,498 |
| Train — total articles | 442 | 442 |
| Train — articles con negatives | 0 | 285 |
| **Dev — total examples** | 10,570 | 11,873 |
| Dev — negative examples | 0 | 5,945 |
| Dev — total articles | 48 | 35 |
| Dev — articles con negatives | 0 | 35 |
| **Test — total examples** | 9,533 | 8,862 |
| Test — negative examples | 0 | 4,332 |
| Test — total articles | 46 | 28 |
| Test — articles con negatives | 0 | 28 |

Observaciones críticas sobre el balance:

- En **train**, el ratio respondible:no-respondible es aproximadamente **2:1** (86,821 respondibles vs 43,498 no-respondibles). Los autores eligieron mantener este sesgo en train porque el costo de anotación de no-respondibles es mayor (requiere conocimiento profundo del párrafo para escribir preguntas plausibles pero falsas).
- En **dev** y **test**, el ratio es aproximadamente **1:1** (49.6% no-respondibles en test). Esta simetría es deliberada: hace que un baseline trivial "siempre responder" o "siempre abstener" obtenga ~50% (con resultados específicos que veremos en la sección de métricas).
- Solo los **285 artículos** (de 442) que tuvieron suficientes preguntas no-respondibles anotadas se incluyen en train con la nueva mezcla. Esto significa que algunas preguntas de SQuAD 1.1 sobre 157 artículos sin anotaciones no-respondibles **se mantienen** en train de SQuAD 2.0, pero sin contraparte negativa. Sutil pero relevante para entender por qué la distribución de train es 2:1 y no 1:1.
- **Dev y test se restringen** a articles con negative examples (35 y 28 respectivamente). Esto evita que un modelo gane puntos en evaluación gratis sobre artículos donde nunca se le pide abstener.

### 3.2 Proceso de anotación

Los autores usaron la plataforma de crowdsourcing **Daemo** (Gaikwad et al. 2015), un sistema de UC Stanford diseñado específicamente para experimentos académicos. La tarea se diseñó así:

1. Cada tarea consistía en **un artículo completo** de SQuAD 1.1 (no un párrafo aislado), lo que permitía a los workers contextualizarse en el tema.
2. Para cada párrafo del artículo, se pedía **hasta 5 preguntas no-respondibles**.
3. Como referencia visual, los workers veían las preguntas **respondibles** de SQuAD 1.1 para el mismo párrafo. Esto era estratégico: induce a los anotadores a escribir preguntas estructuralmente similares (mismas formas interrogativas, mismo vocabulario del dominio), lo que hace la tarea de discriminación más difícil para los modelos.
4. Para cada pregunta no-respondible escrita, el worker debía además **resaltar un span del párrafo** que sirviera como "respuesta plausible" — un span del tipo correcto que un modelo ingenuo elegiría.
5. **Tiempo asignado**: 7 minutos por párrafo. **Pago**: $10.50/hora.

El paper especifica un filtro de calidad: se eliminaron las preguntas de workers que escribieron **25 o menos preguntas** en un artículo dado. La lógica del filtro: anotadores que no completaron suficiente trabajo en un artículo probablemente no entendieron bien la tarea y abandonaron, lo que correlaciona con baja calidad. Este filtro se aplicó simétricamente a las preguntas respondibles existentes de SQuAD 1.1 para mantener consistencia.

### 3.3 Características de las preguntas adversariales

Las preguntas no-respondibles cumplen tres propiedades, formalizadas en la Sección 2 (Desiderata) del paper:

1. **Relevancia**: usan vocabulario y entidades del párrafo. Esto evita que una heurística basada en superposición léxica las identifique trivialmente como "fuera de tópico". Si un modelo simplemente midiera "qué fracción de las palabras de la pregunta aparece en el párrafo", las no-respondibles serían indistinguibles de las respondibles.

2. **Existencia de respuesta plausible**: el párrafo contiene un span del **tipo** correcto (una persona si la pregunta es `who`, una fecha si es `when`, etc.). Esto neutraliza las heurísticas de type-matching. Si la pregunta pide "what year" y el párrafo no tiene ninguna fecha, un modelo podría abstenerse sin entender realmente. Forzar que exista una fecha en el párrafo obliga al modelo a verificar **semánticamente** si esa fecha responde la pregunta.

3. **No-respondibilidad genuina**: ningún span del párrafo entail la pregunta. Esto es lo que los crowdworkers humanos garantizan al escribir las preguntas. El paper reporta que en una inspección manual de 100 ejemplos negativos del dev set, el **93%** son genuinamente no-respondibles (7% son ruido — preguntas que en realidad sí tienen respuesta y fueron mal-anotadas).

### 3.4 Validación humana

Para estimar el ceiling de performance humana, los autores hicieron una anotación adicional sobre todos los ejemplos de dev y test:

- Crowdworkers vieron artículos completos con preguntas mezcladas (respondibles y no-respondibles aleatoriamente).
- Para cada pregunta, debían **resaltar la respuesta en el párrafo** o **marcarla como no-respondible**.
- **4.8 anotadores por pregunta** en promedio. La respuesta final se eligió por **mayoría**, rompiendo empates a favor de responder (y prefiriendo respuestas más cortas en caso de empate adicional).

Esto resulta en métricas de humano más altas que las reportadas para SQuAD 1.1 original (donde Rajpurkar et al. 2016 evaluaron a un solo anotador, subestimando el techo humano). El humano en SQuAD 2.0 dev alcanza **86.3 EM / 89.0 F1**, y en test **86.9 EM / 89.5 F1**.

Una observación importante: el humano hace, recalculado con la métrica de SQuAD 2.0, **82.3 EM / 91.2 F1 en SQuAD 1.1 test** (los números clásicos). Esto significa que pasar de SQuAD 1.1 a SQuAD 2.0 cuesta al humano **apenas 1.7 F1** (de 91.2 a 89.5). En contraste, los mejores modelos caen ~19 F1 en la misma transición. Esta asimetría es exactamente el punto del paper: lo que SQuAD 2.0 añade es trivial para humanos (decidir si una pregunta tiene respuesta en un texto que ya leíste es algo natural) y catastrófico para modelos que no fueron diseñados con ese mecanismo en mente.

### 3.5 Splits y compatibilidad

Crucial para la comunidad: SQuAD 2.0 reusa el **mismo split de artículos** que SQuAD 1.1. Un modelo entrenado en SQuAD 2.0 puede evaluarse en SQuAD 1.1 sin contaminación: las preguntas son distintas, pero ningún artículo de test cruza dominios. Esto facilita ablations directas entre versiones, como las que el paper hace en la Tabla 3 (comparación SQuAD 1.1 vs SQuAD 2.0 con la misma arquitectura).

---

## 4. Tipos de preguntas no-respondibles (taxonomía del paper)

La Sección 4.3 del paper presenta una taxonomía construida a partir de una inspección manual de 100 ejemplos negativos del dev set. La distribución de fenómenos es notablemente más diversa que la de cualquier método automático previo. La Tabla 1 del paper categoriza así:

| Categoría | Descripción | Ejemplo | % |
|---|---|---|---|
| **Negation** | Palabra de negación insertada o eliminada respecto al enunciado del párrafo. | S: "*Several hospital pharmacies have decided to outsource high risk preparations…*"<br>Q: "*What types of pharmacy functions have **never** been outsourced?*" | 9% |
| **Antonym** | Antónimo usado en la pregunta. | S: "*…the extinction of the dinosaurs allowed the tropical rainforest to spread out across the continent.*"<br>Q: "*The extinction of what led to the **decline** of rainforests?*" | 20% |
| **Entity Swap** | Entidad, número o fecha reemplazada por otra. | S: "*These values are much greater than the 9–88 cm as projected in its Third Assessment Report.*"<br>Q: "*What was the projection of sea level increases in the **fourth** assessment report?*" | 21% |
| **Mutual Exclusion** | Pregunta sobre algo mutuamente exclusivo con algo presente en el párrafo. | S: "*BSkyB waived the charge for subscribers whose package included two or more premium channels.*"<br>Q: "*What service did BSkyB give away for free **unconditionally**?*" | 15% |
| **Impossible Condition** | Pregunta condicional cuya condición no se cumple en el párrafo. | S: "*Union forces left Jacksonville and confronted a Confederate Army at the Battle of Olustee… Union forces then retreated to Jacksonville and held the city for the remainder of the war.*"<br>Q: "*After what battle did Union forces leave Jacksonville **for good**?*" | 4% |
| **Other Neutral** | El párrafo simplemente no implica ninguna respuesta a la pregunta. | S: "*Schuenemann et al. concluded in 2011 that the Black Death was caused by a variant of Y. pestis…*"<br>Q: "*Who **discovered** Y. pestis?*" | 24% |
| **Answerable** | Pregunta de hecho sí respondible (ruido de anotación). | — | 7% |

Reflexiones sobre la taxonomía:

**Antonym y Entity Swap juntos son 41%** — son las dos categorías más frecuentes. Son las únicas dos categorías que `RULE BASED` de Jia & Liang 2017 cubría: ese método aplicaba reemplazos de entidades/números y sustituciones por antónimos de WordNet. Sin embargo, incluso aquí los humanos producen ejemplos más sutiles. Un reemplazo automático de "Third" → "Fourth" en el ejemplo de entity swap es similar a lo que `RULE BASED` haría, pero el humano puede producir variantes más creativas (intercambios entre entidades del mismo tipo dentro del párrafo, no solo del vocabulario general).

**Mutual Exclusion (15%)** y **Impossible Condition (4%)** son categorías genuinamente nuevas que ningún método automático produce. Requieren razonamiento sobre el contenido del párrafo, no solo modificación léxica. El ejemplo de BSkyB es muy revelador: el párrafo dice "waived the charge for subscribers whose package included two or more premium channels" (servicio gratis bajo condición). La pregunta cambia "gave away for free" + agrega "unconditionally". Para detectar que es no-respondible, el modelo debe entender que "gratis bajo condición" es mutuamente excluyente con "gratis incondicionalmente". Esto es razonamiento lógico, no matching léxico.

**Other Neutral (24%)** es la categoría más grande y la más heterogénea. Incluye casos como "Who discovered Y. pestis?" cuando el párrafo solo dice "Schuenemann et al. concluyeron en 2011 que la Peste Negra fue causada por una variante de Y. pestis". El párrafo habla de quien estableció la conexión Y. pestis–Peste Negra, no de quien descubrió la bacteria. La distinción requiere comprender la diferencia entre "concluir que X causó Y" y "descubrir X".

**Negation (9%)** es relativamente baja, lo que es sorprendente dado que es uno de los retos clásicos de NLP. Una hipótesis: los anotadores prefieren mecanismos más sutiles, y la negación explícita (`never`, `not`) suena demasiado "obvia" como adversarial. Vale notar que casi todos los sistemas QA pre-BERT manejan mal la negación, y esta categoría sigue siendo un benchmark interesante para modelos más recientes.

**Answerable (7%)** es ruido del dataset — preguntas que los anotadores marcaron como no-respondibles pero que en realidad sí lo son. Este ruido pone un techo natural a la performance humana: ningún sistema (humano incluido) puede alcanzar 100% F1 sin "tener razón" en estos casos ambiguos. La cifra 7% es importante para contextualizar el 89.5 humano: si el ruido es 7%, el humano está virtualmente en el techo del benchmark.

Comparación con el método rule-based de Jia & Liang 2017: el paper hace notar que `RULE BASED` solo genera variantes de los tipos Antonym (20%) y Entity Swap (21%) — 41% de las categorías de SQuAD 2.0. Esto significa que **59% de las preguntas adversariales de SQuAD 2.0 son tipos que ningún método automático produce**. La diversidad es el argumento principal por el cual el dataset es más difícil que las alternativas automatizadas.

---

## 5. Métrica EM/F1 ajustada para no-respondibles

### 5.1 El ajuste fundamental

Las métricas tradicionales de SQuAD 1.1 — Exact Match (EM) y F1 a nivel de token — están definidas asumiendo que existe una respuesta gold. SQuAD 2.0 las extiende con una regla simple pero específica:

- Para una pregunta **respondible**, EM y F1 se calculan exactamente como en SQuAD 1.1. Se toma el máximo sobre múltiples respuestas humanas gold.
- Para una pregunta **no-respondible**:
  - Si el modelo predice **no-answer** (la cadena vacía): EM = 1 y F1 = 1.
  - Si el modelo predice **cualquier span** (no vacío): EM = 0 y F1 = 0.

Es decir, en preguntas no-respondibles la métrica es binaria: o aciertas la abstención o sales con 0 puntos. No hay crédito parcial. El paper lo formaliza en la footnote 3 de la Sección 5.2:

> *For negative examples, abstaining receives a score of 1, and any other response gets 0, for both exact match and F1.*

El EM/F1 global se promedia sobre **todas** las preguntas del split, mezclando respondibles y no-respondibles. Esto significa que un modelo que sea perfecto en respondibles pero responda siempre en no-respondibles (porque eso era óptimo en SQuAD 1.1) obtendrá ~50 EM/F1 en SQuAD 2.0. Un modelo que abstenga siempre obtendrá también ~50 (acertando todas las no-respondibles, fallando todas las respondibles). El paper explicita que el baseline **always-abstain** alcanza **48.9 F1 en test** — la mitad inferior del dataset es casi exactamente 1:1.

### 5.2 Formalización

Sea $D$ el conjunto de preguntas del split, particionado en $D = D_+ \cup D_-$ donde $D_+$ son respondibles y $D_-$ no-respondibles. Sea $\hat{y}(q)$ la predicción del modelo para la pregunta $q$ (un span, posiblemente vacío). Sea $g(q)$ el conjunto de respuestas gold para $q$ (no vacío si $q \in D_+$, definido como $\{\emptyset\}$ si $q \in D_-$).

El EM por pregunta es:

$$
\text{EM}(q) =
\begin{cases}
\max_{a \in g(q)} \mathbb{1}[\hat{y}(q) = a] & \text{si } q \in D_+ \\
\mathbb{1}[\hat{y}(q) = \emptyset] & \text{si } q \in D_-
\end{cases}
$$

El F1 por pregunta usa la fórmula clásica de overlap token a token entre $\hat{y}(q)$ y $a$, normalizando por la mejor respuesta gold:

$$
\text{F1}(q) =
\begin{cases}
\max_{a \in g(q)} \text{F1}_{\text{token}}(\hat{y}(q), a) & \text{si } q \in D_+ \\
\mathbb{1}[\hat{y}(q) = \emptyset] & \text{si } q \in D_-
\end{cases}
$$

Y la métrica global:

$$
\text{EM} = \frac{1}{|D|}\sum_{q \in D} \text{EM}(q), \qquad \text{F1} = \frac{1}{|D|}\sum_{q \in D} \text{F1}(q)
$$

### 5.3 Threshold de no-answer como hiperparámetro

Los modelos QA típicamente producen, para cada par (pregunta, párrafo), una distribución de probabilidad sobre spans. Para soportar abstención, se introduce un **score de no-answer** $s_\text{null}$. Si $s_\text{null}$ excede un threshold $\tau$, el modelo abstiene; si no, predice el span de máxima probabilidad. Formalmente:

$$
\hat{y}(q) =
\begin{cases}
\emptyset & \text{si } s_\text{null}(q) > \tau \\
\arg\max_{(i,j)} \text{score}(i, j \mid q) & \text{en otro caso}
\end{cases}
$$

El threshold $\tau$ es un hiperparámetro escalar que se ajusta sobre el dev set para maximizar F1. El paper observa (Sección 5.1) que ajustar $\tau$ específicamente para F1 funciona mejor que simplemente usar el argmax de las probabilidades del modelo, "posiblemente debido a las diferentes proporciones de ejemplos negativos en train y test". Es decir: si train tiene 2:1 respondibles:no-respondibles y test tiene 1:1, el threshold óptimo no será el "neutro" sino uno que compense el cambio de distribución.

Esta práctica de calibración por threshold es estándar en la literatura post-SQuAD 2.0 (BERT, RoBERTa, ALBERT, DeBERTa) y se documenta como hiperparámetro estándar de evaluación en todos los papers que reportan resultados sobre el benchmark.

### 5.4 Implicaciones para el diseño de modelos

La introducción de la métrica tiene tres consecuencias arquitectónicas que dominan la literatura posterior:

1. **El modelo debe producir un score de no-answer**, no solo una distribución sobre spans. Esto requiere un head adicional o un mecanismo dentro de la arquitectura que represente "nada del párrafo responde la pregunta".
2. **La calibración importa tanto como la accuracy**. Un modelo que distingue bien respondibles de no-respondibles pero con scores mal calibrados (todos los no-answers tienen score apenas mayor que la mayoría de respondibles) será sensible al threshold y difícil de evaluar de forma estable.
3. **El threshold introduce un trade-off precision-recall sobre abstención**. Un $\tau$ alto hace al modelo más confiado en responder (recall alto en respondibles, accuracy baja en no-respondibles); un $\tau$ bajo lo hace más cauteloso. El F1 óptimo del dev set fija ese trade-off para test.

---

## 6. Baselines del paper

El paper evalúa tres modelos sobre SQuAD 2.0 (Tabla 3, Sección 5.2). Todos son arquitecturas pre-BERT modificadas para soportar abstención.

### 6.1 BNA — BiDAF + No-Answer pointer

BNA es la versión "no-answer aware" del BiDAF clásico (Seo et al. 2016), propuesta originalmente por Levy et al. (2017) en el contexto de Zero-Shot Relation Extraction. La modificación arquitectónica es minimalista: se introduce un **token especial de no-answer** al inicio del párrafo (similar a `[CLS]` en BERT, pero el paper es pre-BERT). El modelo produce una distribución sobre todos los token positions del párrafo, incluyendo este token especial. Si el token de no-answer recibe la máxima probabilidad, el modelo abstiene.

Resultados:
- SQuAD 1.1 test: 68.0 EM / 77.3 F1
- SQuAD 2.0 dev: 59.8 EM / 62.6 F1
- SQuAD 2.0 test: 59.2 EM / 62.1 F1

**Gap SQuAD 1.1 → 2.0 en F1**: 15.2 puntos de caída. El modelo es claramente peor en el dataset adversarial.

### 6.2 DocQA — DocumentQA No-Answer

DocQA (Clark & Gardner 2017) es una arquitectura más sofisticada que BiDAF, diseñada para reading comprehension multi-paragraph. Para SQuAD 2.0, los autores la modifican con una cabeza explícita que predice $P(\text{unanswerable})$ como un escalar adicional. El modelo abstiene cuando este escalar excede un threshold tuned en dev.

Resultados:
- SQuAD 1.1 test: 72.1 EM / 81.0 F1
- SQuAD 2.0 dev: 61.9 EM / 64.8 F1
- SQuAD 2.0 test: 59.3 EM / 62.3 F1

**Gap SQuAD 1.1 → 2.0 en F1**: 18.7 puntos de caída. Aún más drástico que BNA.

### 6.3 DocQA + ELMo

La misma arquitectura DocQA con embeddings de palabras enriquecidos con ELMo (Peters et al. 2018) — los contextual embeddings basados en BiLSTM que precedieron a BERT. ELMo aporta representaciones más ricas, lo que mejora ambos benchmarks.

Resultados:
- SQuAD 1.1 test: 78.6 EM / 85.8 F1
- SQuAD 2.0 dev: 65.1 EM / 67.6 F1
- SQuAD 2.0 test: 63.4 EM / 66.3 F1

**Gap SQuAD 1.1 → 2.0 en F1**: 19.5 puntos de caída.

### 6.4 Humano

Anotación múltiple, majority vote, breaking ties a favor de respuesta.

- SQuAD 1.1 test (re-evaluado): 82.3 EM / 91.2 F1
- SQuAD 2.0 dev: 86.3 EM / 89.0 F1
- SQuAD 2.0 test: 86.9 EM / 89.5 F1

**Gap SQuAD 1.1 → 2.0 en F1 humano**: -1.7 (mejora marginal). Esto confirma que el dataset es prácticamente igual de fácil para humanos. Las preguntas no-respondibles agregan poco trabajo cognitivo a un humano que ya está leyendo el párrafo cuidadosamente.

### 6.5 El gap humano-máquina como tesis empírica

Tabla resumen del paper (extraído de Tabla 3):

| Métrica | SQuAD 1.1 test (humano vs DocQA+ELMo) | SQuAD 2.0 test (humano vs DocQA+ELMo) |
|---|---|---|
| EM gap | 82.3 - 78.6 = **3.7** | 86.9 - 63.4 = **23.5** |
| F1 gap | 91.2 - 85.8 = **5.4** | 89.5 - 66.3 = **23.2** |

El gap se **cuadruplica**. En SQuAD 1.1, el gap de 5.4 F1 era percibido por la comunidad como "casi cerrado" — el benchmark estaba a meses de ser superhumano. En SQuAD 2.0, el gap de 23.2 F1 es enorme, comparable al gap en GLUE en 2017 (que tomó dos años de progreso intensivo para cerrar). El mensaje del paper: la métrica de "comprensión" en QA estaba mal calibrada, y SQuAD 2.0 la recalibra a la baja por más de 15 puntos.

### 6.6 Plausible answers como distractores

El paper incluye un análisis revelador en la Sección 5.4 (y Apéndice A.2). Para cada modelo, midieron qué fracción de los **falsos positivos** (preguntas no-respondibles donde el modelo predijo un span) coincidían con el "plausible answer" anotado por los crowdworkers como distractor.

Tabla 5 del paper:

| Sistema | EM | F1 |
|---|---|---|
| BNA | 48.6 | 63.0 |
| DocQA | 55.0 | 69.9 |
| DocQA + ELMo | 54.9 | 69.2 |
| Human | 46.4 | 60.6 |

Interpretación: aproximadamente **la mitad** de los falsos positivos coinciden exactamente con el span plausible que el anotador marcó como distractor. Esto confirma que los distractores efectivamente cumplen su función: no son artefactos arbitrarios, son las respuestas que el modelo se siente más tentado a dar. El humano también cae en el ~50%, lo que sugiere que los distractores son "naturalmente engañosos", no idiosincráticos del modelo.

Este análisis es metodológicamente importante porque valida la **calidad de la anotación** del dataset. Si los plausible answers no fueran realmente distractores, los modelos los predecirían en una fracción aleatoria (no la mitad). El hecho de que humanos también caigan en la misma trampa con frecuencia comparable indica que los distractores capturan un fenómeno real de comprensión, no un artefacto del anotador.

---

## 7. Cómo modelos modernos abordan SQuAD 2.0

### 7.1 BERT — el approach minimalista

BERT (Devlin et al. 2018) introduce el approach que se volvió canónico. La arquitectura de fine-tuning para SQuAD 2.0 reutiliza el head de SQuAD 1.1 con una modificación elegante: tratar el token `[CLS]` como la posición de "no-answer".

Recordemos el head de SQuAD 1.1 en BERT. Hay dos vectores aprendidos $S, E \in \mathbb{R}^H$. Para cada token $T_i$ del párrafo:

- Score de inicio: $s_i = S \cdot T_i$
- Score de fin: $e_i = E \cdot T_i$

El span score es $s_i + e_j$ para $j \ge i$. Se predice el par $(i, j)$ que maximiza este score.

Para SQuAD 2.0, BERT extiende la lógica así. Sea $C$ la representación final del token `[CLS]`. Definir el **null score**:

$$s_\text{null} = S \cdot C + E \cdot C$$

Y el **mejor span no-null**:

$$s_{i,j}^* = \max_{1 \le i \le j \le N} (S \cdot T_i + E \cdot T_j)$$

donde $N$ es la longitud del párrafo (los índices se restringen a tokens del párrafo, excluyendo `[CLS]` y la pregunta). La predicción final:

$$
\hat{y} =
\begin{cases}
\emptyset \text{ (no answer)} & \text{si } s_\text{null} > s_{i,j}^* + \tau \\
\text{span}(i^*, j^*) & \text{en otro caso}
\end{cases}
$$

con $\tau$ tuned en dev set. Esta formulación es brillante por su simplicidad: no requiere head adicional, reusa los vectores $S, E$ existentes, y trata "no answer" como un span degenerado de longitud cero en la posición `[CLS]`.

Resultado de BERT-large single en SQuAD 2.0: **81.9 F1 dev / 83.1 F1 test**. Versus el mejor baseline del paper original (DocQA + ELMo, 66.3 F1 test), BERT representa una mejora de **+16.8 F1** — recuperando casi exactamente los puntos que SQuAD 2.0 había "robado" al paradigma pre-2018. Sin embargo, BERT-large sigue ~6 puntos por debajo del humano (89.5 F1), dejando espacio para mejoras posteriores.

### 7.2 XLNet — head adicional `answer_class`

XLNet (Yang et al. 2019) refina la mecánica de SQuAD 2.0 con una decisión arquitectónica que aparecerá directamente en el lab 20 del curso. En vez de derivar el score de no-answer del mismo head de span, XLNet añade un **head explícito de clasificación binaria** llamado `answer_class`.

La formulación de XLNet para SQuAD 2.0 consta de tres componentes:

1. **Start head**: predice $p(\text{start}_i \mid q, p)$ — distribución sobre posiciones de inicio del span.
2. **End head**: predice $p(\text{end}_j \mid \text{start}_i, q, p)$ — distribución sobre posiciones de fin, **condicionada** en la posición de inicio. Esto es una diferencia respecto a BERT (donde start y end son independientes) y permite que el modelo capture la dependencia entre los dos endpoints del span.
3. **Answer class head**: predice $p(\text{answerable} \mid q, p) \in [0, 1]$ — clasificación binaria de si la pregunta es respondible.

Matemáticamente, el head de `answer_class` es una pequeña MLP de dos capas sobre una representación pooled del input. Concretamente, en `transformers` (la implementación de HuggingFace de XLNet), la arquitectura es:

$$
\text{answer\_class}(h_{\text{CLS}}, h_{\text{start}}) = w_2^\top \cdot \tanh\!\left(W_1 \cdot [h_{\text{CLS}}; h_{\text{start}}] + b_1\right)
$$

donde:
- $h_{\text{CLS}}$ es la representación del token `<cls>` (XLNet usa `<cls>` al final, no al inicio como BERT).
- $h_{\text{start}}$ es la representación esperada del token de inicio, ponderada por la distribución $p(\text{start})$.
- $W_1 \in \mathbb{R}^{H \times 2H}$ es la matriz de la primera capa (llamada `dense_0` en el código).
- $w_2 \in \mathbb{R}^{H}$ es el vector de la segunda capa (llamada `dense_1`, output binario).
- $b_1$ es el bias de la primera capa.

Los parámetros explícitos en la implementación son entonces:

| Parámetro | Shape | Rol |
|---|---|---|
| `answer_class.dense_0.weight` | $H \times 2H$ | Proyección de la concatenación CLS+start |
| `answer_class.dense_0.bias` | $H$ | Bias de la primera capa |
| `answer_class.dense_1.weight` | $H$ (sin bias) | Proyección al logit binario |

Estos son **exactamente** los nombres de parámetros que aparecen en el warning del lab 20 cuando se carga `XLNetForQuestionAnswering` con pesos pre-entrenados que no incluyen el head de SQuAD 2.0. El warning indica que esos pesos están inicializados aleatoriamente y requieren fine-tuning sobre SQuAD 2.0 para tener significado.

La razón por la cual XLNet eligió un head separado en vez del approach de BERT es empírica: factorizar `answer_class` separadamente del span head permite que el modelo aprenda la clasificación de respondibilidad con una señal de entrenamiento dedicada, sin contaminar la señal del span. Resultados de XLNet-large en SQuAD 2.0: **88.4 F1 dev / 89.1 F1 test**, superando ligeramente al humano por primera vez (89.5).

### 7.3 SpanBERT y RoBERTa — threshold sobre logits

SpanBERT (Joshi et al. 2020) y RoBERTa (Liu et al. 2019) adoptan el approach de BERT (reusar `[CLS]` como no-answer) sin agregar head específico. Sus mejoras vienen de pre-training más fuerte:

- **RoBERTa**: eliminar NSP, dynamic masking, batch grande, más data, más pasos. RoBERTa-large alcanza **89.4 F1 dev / 89.8 F1 test** en SQuAD 2.0.
- **SpanBERT**: cambiar masking de tokens individuales a spans contiguos, agregar objetivo de Span Boundary Objective (predecir tokens internos del span desde los endpoints). SpanBERT-large alcanza **88.7 F1 / 88.7 F1**.

Ambos son ligeramente mejores que XLNet con la mitad de la complejidad del head — sugiriendo que el bottleneck real es la calidad del pretraining, no la arquitectura del head.

### 7.4 ALBERT, DeBERTa — el régimen super-humano

| Modelo | Año | F1 dev | F1 test |
|---|---|---|---|
| Baselines del paper (DocQA + ELMo) | 2018 | 67.6 | 66.3 |
| BERT-large single | 2018 | 81.9 | 83.1 |
| XLNet-large | 2019 | 88.4 | 89.1 |
| RoBERTa-large | 2019 | 89.4 | 89.8 |
| SpanBERT-large | 2020 | 88.7 | 88.7 |
| ALBERT-xxlarge | 2019 | 88.1 | 90.9 |
| **Humano** | — | **89.0** | **89.5** |
| DeBERTa-large | 2020 | 90.7 | 91.1 |
| DeBERTa-v2-xxlarge | 2020 | 91.4 | 92.2 |
| DeBERTa-v3-large | 2021 | 91.5 | 92.3 |
| Ensemble top leaderboard | 2022 | 93.0+ | 93.1+ |

A partir de XLNet/RoBERTa (mediados de 2019), los modelos superan el humano. DeBERTa (He et al. 2020-2021) lleva el techo a ~92 F1, ~2.5 puntos sobre el humano. Esto **no** significa que el problema esté resuelto en el sentido de comprensión — los modelos siguen explotando patrones estadísticos y la generalización a dominios fuera de Wikipedia es pobre — pero sí significa que SQuAD 2.0 dejó de ser un benchmark discriminante.

La progresión confirma la dinámica clásica de benchmarks de NLP: un benchmark difícil se vuelve fácil en 2-3 años de progreso intensivo. SQuAD 1.1 saturó en 2 años; SQuAD 2.0 en aproximadamente lo mismo. La comunidad respondió creando benchmarks más difíciles: HotpotQA (multi-hop), DROP (numerical reasoning), Natural Questions (real Google queries), TriviaQA (no fácil), BoolQ (yes/no), y eventualmente benchmarks de razonamiento como BIG-Bench y MMLU.

---

## 8. Limitaciones del benchmark

SQuAD 2.0 fue una mejora sustancial sobre SQuAD 1.1, pero hereda y conserva varias limitaciones que la literatura subsiguiente abordó:

### 8.1 "Unanswerable" sigue siendo dentro del contexto

Una pregunta es "no-respondible" en SQuAD 2.0 si el **párrafo dado** no contiene la respuesta. Pero la respuesta podría existir en otra parte de Wikipedia, o en cualquier KB externo. El benchmark no aborda el problema de QA cuando el contexto no es relevante en absoluto. Para eso surgieron benchmarks como **Natural Questions** (Kwiatkowski et al. 2019), que opera sobre páginas completas de Wikipedia y obliga al modelo a navegar contenido potencialmente irrelevante, y **MS MARCO** (Nguyen et al. 2016), que usa queries reales de Bing donde la mayoría de documentos retrieved no contienen respuesta.

### 8.2 No aborda multi-hop reasoning

Toda pregunta de SQuAD 2.0 es respondible (o no) usando un único párrafo. Esto excluye el razonamiento que requiere conectar información de múltiples fuentes — por ejemplo, "¿En qué año nació el director de la película X?", que requiere primero encontrar el director y luego buscar su fecha de nacimiento. Este tipo de razonamiento fue capturado por **HotpotQA** (Yang et al. 2018), publicado el mismo año, que diseña preguntas que requieren obligatoriamente dos saltos a través de párrafos distintos.

### 8.3 No aborda razonamiento numérico ni procedural

SQuAD 2.0 es exclusivamente extractive: la respuesta es un span literal del texto. Esto excluye preguntas como "¿Cuántos años hay entre X e Y?" (resta entre fechas) o "¿Cuál de A, B, C es el más alto?" (comparación). **DROP** (Dua et al. 2019) abordó este vacío con preguntas que requieren aritmética, comparación y reasoning compositional. RACE (Lai et al. 2017) ya cubría multiple-choice y interpretación.

### 8.4 Dominio limitado: Wikipedia

Toda la diversidad de SQuAD 2.0 está dentro del estilo enciclopédico de Wikipedia. Modelos entrenados aquí no necesariamente generalizan bien a texto biomédico (PubMedQA), legal (CUAD), financiero (FiQA), o conversacional (CoQA, QuAC). La literatura posterior diversificó dominios; SQuAD permanece como benchmark "vainilla".

### 8.5 Anotación crowdsource y sesgo de tipo

Como en SQuAD 1.1, las preguntas son generadas por crowdworkers que **leyeron el párrafo primero**. Esto sesga el estilo de pregunta hacia formas que un anotador produce cuando ya conoce la respuesta — distinto de cómo un usuario real preguntaría sin contexto previo. **Natural Questions** atacó este sesgo usando queries reales de Google search; SQuAD 2.0 no lo aborda.

### 8.6 Métrica binaria sobre no-respondibles

La regla EM=F1=1 si abstiene, 0 si no, descarta información. Un modelo que abstiene con confianza muy baja recibe el mismo crédito que uno que abstiene con confianza alta. Métricas como **expected calibration error** o **Brier score** capturarían la calidad de la calibración de incertidumbre, pero no son parte del benchmark estándar.

Estas limitaciones no invalidan SQuAD 2.0; son las restricciones naturales de cualquier dataset. Lo notable es que ningún benchmark posterior absorbió completamente todas las dimensiones: HotpotQA cubrió multi-hop pero no abstención; Natural Questions cubrió queries reales pero no diversidad de no-respondibles adversariales; DROP cubrió numerical reasoning pero no calibración. SQuAD 2.0 quedó como el benchmark de referencia para **abstención en QA extractivo** y se mantiene como ablation estándar de cualquier modelo serio de comprensión.

---

## 9. Conexión con la clase 20 del Diplomado IA UC

La clase 20 cubre la transición ELMo → BERT → GPT → ChatGPT, articulando la era de contextual embeddings y pre-training masivo. SQuAD 2.0 es un benchmark transversal a casi todo el contenido de la clase porque encarna el problema que motiva las arquitecturas presentadas.

**Conexión con ELMo**: el paper de SQuAD 2.0 evalúa DocQA con y sin ELMo, mostrando que ELMo aporta ~3 F1 (62.3 → 66.3 en test). Esto es consistente con el rol de ELMo como mejora de representaciones, no como solución del problema de razonamiento.

**Conexión con BERT**: SQuAD 2.0 es uno de los benchmarks que BERT reportó en su versión v2 (NAACL 2019). La extensión del head de SQuAD 1.1 con `[CLS]` como no-answer es elegante y se volvió canónica.

**Conexión con la noción de "alucinación"**: SQuAD 2.0 introduce, en 2018, la idea de que un modelo de comprensión debe **saber lo que no sabe**. Esta es exactamente la preocupación contemporánea con LLMs que producen respuestas plausibles pero incorrectas a preguntas para las cuales no tienen información. La continuidad conceptual desde SQuAD 2.0 hasta la investigación de calibración en LLMs (Kadavath et al. 2022, "Language Models (Mostly) Know What They Know") es directa: el benchmark de 2018 anticipa, en formato extractive, el desafío central de calibración epistémica que dominaría la era post-ChatGPT.

**Conexión con RLHF**: aunque SQuAD 2.0 es pre-RLHF, su filosofía es compatible con el objetivo de entrenamiento de RLHF. Un modelo entrenado con feedback humano aprende, entre otras cosas, a **declinar** preguntas que no puede responder con confianza. Esta es la versión generativa de lo que SQuAD 2.0 evalúa en formato extractive. El benchmark se vuelve un test diagnóstico útil incluso para LLMs: si un LLM falla en SQuAD 2.0 (predice spans plausibles pero incorrectos en lugar de abstener), revela un problema de calibración que probablemente también ocurre en su comportamiento generativo.

---

## 10. Conexión con el lab 20

El lab 20 del curso usa `XLNetForQuestionAnswering` (no `XLNetForQuestionAnsweringSimple`) cargado desde HuggingFace. La diferencia entre ambas clases es exactamente el contenido de este paper.

### 10.1 Las dos variantes de XLNet QA en HuggingFace

HuggingFace transformers expone dos clases para QA con XLNet:

- **`XLNetForQuestionAnsweringSimple`**: head simple, solo `start_logits` y `end_logits` independientes. Equivalente al head de BERT para SQuAD 1.1. **No tiene mecanismo de no-answer**.
- **`XLNetForQuestionAnswering`**: head completo con tres componentes — `start_logits`, `end_logits` condicionado en start, y `answer_class` para predecir respondibilidad. **Diseñado específicamente para SQuAD 2.0**.

El lab 20 usa la segunda variante. Cuando se carga el checkpoint pre-entrenado (típicamente `xlnet-base-cased`), aparece un warning en la celda 13 que dice algo equivalente a:

```
Some weights of XLNetForQuestionAnswering were not initialized from the model
checkpoint at xlnet-base-cased and are newly initialized:
['answer_class.dense_0.bias',
 'answer_class.dense_0.weight',
 'answer_class.dense_1.weight',
 'start_logits.dense.bias',
 'start_logits.dense.weight',
 'end_logits.dense_0.bias',
 'end_logits.dense_0.weight',
 'end_logits.dense_1.bias',
 'end_logits.dense_1.weight',
 'end_logits.LayerNorm.bias',
 'end_logits.LayerNorm.weight']
You should probably TRAIN this model on a down-stream task to be able
to use it for predictions and inference.
```

Los parámetros que aparecen en este warning son **exactamente** los heads de QA que el checkpoint pre-entrenado no incluye:

- `start_logits.dense.*`: capa lineal del head de start.
- `end_logits.dense_0.*`, `end_logits.dense_1.*`, `end_logits.LayerNorm.*`: head de end, factorizado en MLP con LayerNorm intermedio (más sofisticado que BERT, captura la dependencia condicional end | start).
- `answer_class.dense_0.*`, `answer_class.dense_1.*`: **el head de SQuAD 2.0 que es el sujeto de este paper**. `dense_0` es la primera capa de la MLP de answer_class (input: concatenación CLS + expected start representation; output: vector de tamaño $H$). `dense_1` es la segunda capa (input: $H$; output: logit escalar binario "answerable vs not").

### 10.2 Interpretación pedagógica del warning

Para el alumno del lab 20, el warning tiene un significado conceptual profundo:

1. **El pre-training de XLNet** (objetivo de permutation language modeling sobre billions of tokens) produce un encoder potente, pero no produce un head de QA. Los heads de QA no son parte del checkpoint público.
2. **Para usar XLNet en QA**, hay que fine-tunear los heads sobre un dataset etiquetado. SQuAD 1.1 fine-tunea solo `start_logits` y `end_logits`; SQuAD 2.0 fine-tunea adicionalmente `answer_class`.
3. **El head `answer_class` es el aporte arquitectónico del paper que estamos analizando**. Sin SQuAD 2.0, esta cabeza no tendría razón de existir. La existencia de `answer_class.dense_0` en el código de HuggingFace es una huella directa del paper de Rajpurkar, Jia y Liang (2018) en la implementación moderna.
4. **Si el alumno fine-tuneara este modelo sobre SQuAD 2.0**, obtendría aproximadamente **89 F1** (cerca del humano), gracias al pre-training fuerte de XLNet combinado con los heads inicializados. Sin el fine-tuning, el modelo produce predicciones aleatorias en las posiciones de start/end y un logit binario aleatorio en answer_class.

### 10.3 El head answer_class en detalle

Para que la conexión sea completa, vale formalizar exactamente qué computa el head. Sea $H_{\text{enc}} \in \mathbb{R}^{N \times H}$ la salida del encoder XLNet sobre la secuencia `<sep> question <sep> paragraph <cls>` de longitud $N$. Sea $h_{\text{cls}} = H_{\text{enc}}[N-1, :] \in \mathbb{R}^H$ la representación del token `<cls>` (al final en XLNet, no al inicio como BERT).

Para predecir si la pregunta es respondible:

1. Calcular distribución de start: $p_{\text{start}} = \text{softmax}(\text{start\_logits}(H_{\text{enc}}))$.
2. Calcular **expected start representation**: $\bar{h}_{\text{start}} = \sum_{i=1}^{N} p_{\text{start}}[i] \cdot H_{\text{enc}}[i, :] \in \mathbb{R}^H$.
3. Concatenar: $z = [h_{\text{cls}}; \bar{h}_{\text{start}}] \in \mathbb{R}^{2H}$.
4. Aplicar MLP:
$$\hat{p}_{\text{answerable}} = \sigma\left(w_2^\top \cdot \tanh(W_1 z + b_1)\right)$$

donde $W_1 \in \mathbb{R}^{H \times 2H}$ (es `answer_class.dense_0.weight`), $b_1 \in \mathbb{R}^H$ (es `answer_class.dense_0.bias`), y $w_2 \in \mathbb{R}^H$ (es `answer_class.dense_1.weight`, sin bias en este componente).

El output $\hat{p}_{\text{answerable}}$ es un escalar en $[0, 1]$ interpretable como la probabilidad de que la pregunta tenga respuesta en el párrafo. Durante inferencia con threshold $\tau$, el modelo abstiene si $\hat{p}_{\text{answerable}} < \tau$.

Durante el fine-tuning sobre SQuAD 2.0, este head se entrena con una loss binary cross-entropy sobre las etiquetas respondible/no-respondible, sumada a la loss estándar de cross-entropy sobre start/end (solo para las preguntas respondibles). La loss total:

$$
\mathcal{L} = \underbrace{-\log p(\text{start}^* \mid q, p)}_{\text{solo si respondible}} - \underbrace{\log p(\text{end}^* \mid \text{start}^*, q, p)}_{\text{solo si respondible}} - \underbrace{\log p(y_{\text{answerable}})}_{\text{siempre}}
$$

El término de `answer_class` se aplica a todas las preguntas (respondibles y no-respondibles), mientras que los términos de start/end se aplican solo a las respondibles (no hay span gold cuando la respuesta no existe).

### 10.4 Llevándolo a la práctica del lab

Para el alumno que termina el lab 20 con curiosidad sobre cómo extender el ejercicio:

1. **Fine-tunear sobre SQuAD 2.0**: el script `run_qa.py` de HuggingFace soporta directamente SQuAD 2.0 con la flag `--version_2_with_negative=True`. Cambiar el dataset de `squad` a `squad_v2` y volver a entrenar.
2. **Inspeccionar el head `answer_class`** del modelo fine-tuned: leer los pesos `model.answer_class.dense_0.weight`, ver sus normas, visualizar sobre qué tokens activa más fuerte. Esto da intuición sobre qué "señales del contexto" el modelo usa para decidir respondibilidad.
3. **Variar el threshold $\tau$** en evaluación y graficar la curva de F1 contra threshold. Esto ilustra empíricamente el trade-off precision-recall sobre abstención, y por qué calibrar $\tau$ en dev es crítico.
4. **Comparar contra `XLNetForQuestionAnsweringSimple`** entrenado en SQuAD 1.1: el modelo sin `answer_class` no puede abstener, y al evaluarse en SQuAD 2.0 obtendrá aproximadamente la mitad del F1 que la versión con head completo. Esta comparación es la demostración empírica directa de la tesis del paper.

---

## 11. Notas para integrar al site

Cosas que el material público del site no menciona explícitamente y conviene incorporar:

1. **Categoría completa de tipos de no-respondibles** (Tabla 1 del paper): Negation 9%, Antonym 20%, Entity Swap 21%, Mutual Exclusion 15%, Impossible Condition 4%, Other Neutral 24%, Ruido 7%. Los porcentajes ilustran que las preguntas humanas son diversas, mientras que métodos rule-based solo cubren Antonym + Entity Swap (41%).
2. **Comparación SQuAD 1.1 vs RULE BASED vs SQuAD 2.0** (Tabla 4): mismo modelo, 89.6 → 89.6 → 67.6 F1. Muestra que el dataset adversarial humano es ~22 F1 más difícil que el rule-based.
3. **Métrica EM/F1 ajustada**: 1 si abstiene en no-respondible, 0 si responde. Baseline always-abstain = 48.9 F1.
4. **Plausible answers como distractores**: ~50% de los falsos positivos coinciden con el span anotado como distractor. Confirma calidad de anotación.
5. **Conexión `answer_class` ↔ lab 20**: el warning de XLNetForQuestionAnswering muestra exactamente el head que este paper introduce conceptualmente.
6. **Limitaciones**: contexto único, no multi-hop, no numerical, no fuera-de-Wikipedia. Útil para contextualizar HotpotQA, DROP, Natural Questions.

---

## 12. Lectura recomendada complementaria

- **Rajpurkar et al. (2016) — SQuAD 1.1** (EMNLP). El dataset original. Prerequisito conceptual.
- **Jia & Liang (2017) — Adversarial Examples for Evaluating Reading Comprehension Systems** (EMNLP). El paper de adversarial SQuAD que motiva directamente SQuAD 2.0.
- **Weissenborn, Wiese & Seiffe (2017) — Making Neural QA as Simple as Possible but not Simpler** (CoNLL). Crítica empírica a SQuAD 1.1, evidencia de que los modelos aprenden heurísticas.
- **Devlin et al. (2018) — BERT** (NAACL 2019). Sección 4.3 reporta SQuAD 2.0 con la formulación `[CLS]` como no-answer.
- **Yang et al. (2019) — XLNet**. Introduce el head `answer_class` explícito que aparece en el lab 20.
- **Joshi et al. (2020) — SpanBERT**. Mejora de pre-training para span tasks, incluyendo SQuAD 2.0.
- **He et al. (2020-2021) — DeBERTa**. Modelo que actualmente lidera SQuAD 2.0 leaderboard con disentangled attention.
- **Yang et al. (2018) — HotpotQA**. Multi-hop QA, captura la dimensión que SQuAD 2.0 no aborda.
- **Dua et al. (2019) — DROP**. Numerical reasoning sobre comprensión de texto.
- **Kwiatkowski et al. (2019) — Natural Questions**. Queries reales, escala de páginas Wikipedia completas.
- **Kadavath et al. (2022) — Language Models (Mostly) Know What They Know**. La continuación filosófica de la tesis de SQuAD 2.0 en la era de LLMs.
