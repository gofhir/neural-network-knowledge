---
title: "LAMBADA (Word prediction requiring a broad discourse context)"
weight: 121
math: true
---

{{< paper-card
    title="The LAMBADA dataset: Word prediction requiring a broad discourse context"
    authors="Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, Raquel Fernández"
    year="2016"
    venue="ACL 2016 (arXiv 1606.06031)"
    pdf="/papers/lambada-paperno-2016.pdf"
    arxiv="1606.06031" >}}
LAMBADA es un dataset de evaluacion, no un modelo: su contribucion es metodologica. Cada item es un pasaje narrativo de ~4.6 oraciones donde hay que predecir la **ultima palabra** de la oracion final. La propiedad que lo define se verifica con humanos: la palabra es adivinable **con el pasaje completo** pero **no con la ultima oracion sola**, aislando asi la comprension de contexto amplio que la perplejidad promedio esconde. Construido filtrando BookCorpus con cuatro LMs y un protocolo de tres pasos en CrowdFlower (~1 de cada 25 pasajes sobrevive). El resultado central es un abismo: humanos ~86% de accuracy, los mejores LMs de 2016 <1%. Anos despues GPT-2 y GPT-3 lo reportaron como metrica insignia de contexto largo, documentando el cierre del gap via escala.
{{< /paper-card >}}

---

## El problema

Hacia 2015-2016 la evaluacion estandar de un modelo de lenguaje (LM) era su **perplejidad** sobre un texto de prueba, el exponencial de la entropia cruzada media por token:

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(w_i \mid w_{1:i-1})\right)$$

El defecto que articula el paper: un texto natural promedio esta dominado por tokens que se predicen **localmente** (articulos, preposiciones, concordancias, frases hechas). Un modelo puede lograr una perplejidad excelente capturando esas regularidades de corto alcance sin entender nada del discurso que lo rodea. Los autores citan el sistema conversacional de Vinyals y Le (2015), que produce respuestas localmente plausibles pero globalmente incoherentes ("loro-like"): la *ilusion de comprension*.

La pregunta de investigacion: como construir un test que mida **solo** la integracion de contexto de discurso amplio, descontando lo que se resuelve con patrones locales. La perplejidad promedio mezcla ambas cosas; se necesita un benchmark que aisle quirurgicamente la dependencia de largo alcance.

---

## Idea central

LAMBADA plantea predicir una palabra (el marco clasico del LM) con una restriccion de diseno que cambia todo. Cada item es un pasaje narrativo donde se adivina la **ultima palabra de la oracion final** (*target word*), bajo una doble condicion verificada empiricamente:

- Un hablante de ingles **acierta** la palabra si ve el pasaje completo (~4.6 oraciones de contexto + la oracion objetivo).
- El mismo hablante **falla** si solo ve la ultima oracion aislada.

El ejemplo canonico: el contexto es una conversacion sobre perder un bebe y un embarazo no planeado; la oracion objetivo, *"Do you honestly think that I would want you to have a ___?"*, admite mil continuaciones por si sola, pero el discurso amplio fija sin ambiguedad la palabra *miscarriage*. La elegancia es que esta propiedad no se postula: **se verifica con humanos**. Un pasaje entra a LAMBADA solo si gente real exhibe el patron "acierta con contexto, falla sin el". La consecuencia para los modelos es contundente: para tener exito, un modelo **no puede apoyarse en el contexto local**, porque por construccion la informacion necesaria esta fuera de la oracion objetivo.

---

## Construccion

**Fuente.** Los pasajes vienen del BookCorpus (Zhu et al., 2015), novelas no publicadas. La eleccion es deliberada: al ser ineditas se **minimiza la utilidad del conocimiento de mundo externo** (Wikipedia, tramas conocidas), forzando al modelo a usar el texto presente. Tras limpieza queda un corpus de **5.325 novelas y 465 millones de palabras**, dividido por novela completa (no por pasaje) en una particion de entrenamiento (2.662 novelas, 203M palabras) y una particion dev+test de la que se construye LAMBADA. La division novela-disjunta garantiza que los pasajes sean **autocontenidos**.

**Contexto.** Numero minimo de oraciones completas antes de la oracion objetivo que acumulen al menos **50 tokens**. La tarea es adivinar la ultima palabra de la oracion objetivo (restriccion elegida por naturalidad para los sujetos, no por necesidad teorica).

**Filtrado automatico.** Para abaratar costos se descartaron primero los pasajes faciles para LMs estandar, usando **cuatro modelos** (un RNN preentrenado de Mikolov y tres entrenados sobre BookCorpus: 4-grama, RNN, feed-forward), distintos de los que luego se evaluan. Cualquier pasaje cuya palabra objetivo tuviera probabilidad $\geq 0{,}00175$ segun *cualquiera* de los cuatro fue excluido.

**Filtrado humano (CrowdFlower).** Sobre los sobrevivientes se aplica el protocolo que define el dataset:

1. Un sujeto adivina con el **pasaje completo**. Si acierta, continua.
2. Un **segundo** sujeto adivina con el pasaje completo. Si tambien acierta, continua.
3. Mas sujetos intentan con la **oracion objetivo sola** hasta acertar o acumular 10 fallos. Si nadie la adivina sin contexto, el pasaje **entra a LAMBADA**.

El paso 2 (dos sujetos consecutivos coincidentes) se anadio tras un piloto para subir la barra de "adivinable con contexto amplio". Las tasas de descarte muestran lo estricto del filtro:

| Etapa | Items descartados |
|---|---|
| Paso 1 (falla con contexto completo) | 84-86% |
| Paso 2 (segundo sujeto falla con contexto) | 6-7% adicional |
| Paso 3 (alguien acierta con la oracion sola) | 3-5% adicional |
| **Sobrevivientes** | **~1 de cada 25** |

El costo final fue **$1,24 promedio por item**. Los autores defienden el enfoque *perfect-match* (coincidencia exacta de palabra): aceptar continuaciones sinonimas o plausibles requeriria anotacion manual masiva, inviable a esta escala.

**Estadisticas finales:**

| Atributo | Valor |
|---|---|
| Total de pasajes | 10.022 |
| Dev | 4.869 (de 1.331 novelas) |
| Test | 5.153 (de 1.332 novelas) |
| Contexto promedio | 4,6 oraciones + 1 objetivo |
| Longitud total | ~75 tokens |
| Corpus de entrenamiento | 2.662 novelas, 203M palabras |

El corpus de entrenamiento es del **mismo dominio** que dev+test pero **sin filtrar**: LAMBADA quiere evaluar modelos de proposito general, no fomentar modelos ad-hoc que solo prediquen ultimas palabras tipo LAMBADA.

---

## Propiedades

**La palabra suele estar en el contexto.** Mas del **80%** de los pasajes incluyen la palabra objetivo (o su lema) en el contexto, frente a menos del 15% en datos sin filtrar. El filtrado humano selecciona pasajes donde la palabra esta anclada referencialmente en el discurso previo. Aun asi la tarea no es trivial: hay que identificar *cual* de las palabras del contexto es la continuacion correcta.

**Categorias gramaticales (POS):**

| Categoria | Proporcion |
|---|---|
| Nombres propios (PN) | 48% |
| Sustantivos comunes (CN) | 37% |
| Verbos | 7,7% |
| Pronombres | 0,3% |
| Adjetivos, adverbios, otros | resto |

Los **nombres propios estan masivamente sobrerrepresentados**: cuando el contexto exige una expresion referencial, la restriccion de "una sola palabra" excluye sintagmas con articulo, y la **co-referencia** (rastrear a que entidad apunta el texto) resulta mas facil que otros fenomenos. Los sustantivos comunes mezclan co-referencia directa, *bridging* (shutter → *camera*), cuasi-sinonimia ('lose the baby' → *miscarriage*) e inferencia de participantes prototipicos (un desayuno → *coffee*, aunque nunca se mencione). Verbos, adjetivos y adverbios son raros porque muchos se adivinan con contexto local y el paso 3 los elimina.

**Cuando la palabra NO esta en el contexto.** Cerca del 16% de LAMBADA tiene el lema ausente del contexto. En ~1/3 de esos casos la palabra esta "casi ahi" (misma raiz, otra categoria: death → *died*); en el resto se exige inferencia de discurso mas compleja (icy road → *driving*, propiedades cualitativas → *lonely*). Una revision manual confirmo que **todos** los nombres propios objetivo si estan en el contexto.

En suma, LAMBADA muestrea "casi cualquier aspecto de la comprension textual" —co-referencia, desambiguacion de sentido, entailment, pragmatica— bajo el paraguas unico de la prediccion de palabra.

---

## Modelos y resultados

Se evaluaron LMs entrenados de forma no supervisada sobre los 203M de palabras: RNN simple (Elman), LSTM, N-grama (SRILM) con y sin cache, y una Memory Network; mas baselines *tailored* (Sup-CBOW, Unsup-CBOW) y baselines aleatorios. Vocabulario de las 60.000 palabras mas frecuentes (cubre el 95% de las objetivo). El LSTM es similar al Deep LSTM Reader de Hermann et al. (2015) y la Memory Network a la mejor de CBT (Hill et al., 2016): no son hombres de paja.

La metrica de interes es **accuracy** (acierto exacto), porque a diferencia del modelado estandar sabemos que los humanos pueden predecir la palabra. Como la accuracy muestra efecto de piso, tambien se reportan perplejidad y rango mediano. Se construyo ademas un **control set** de 5.000 pasajes de las mismas novelas pero **sin filtrar**, un benchmark de LM estandar.

| Conjunto | Metodo | Accuracy | Perplejidad | Rango mediano |
|---|---|---|---|---|
| **LAMBADA** | Random word from passage | 1,6 | — | — |
| | Random capitalized word | 7,3 | — | — |
| | N-Gram | 0,1 | 3125 | 993 |
| | N-Gram w/cache | 0,1 | **768** | 87 |
| | RNN | 0 | 14725 | 7831 |
| | LSTM | 0 | 5357 | 324 |
| | Memory Network | 0 | 16318 | 846 |
| **Control** | N-Gram | 19,1 | 285 | 17 |
| | RNN | 15,4 | 277 | 24 |
| | LSTM | **21,9** | 149 | 12 |
| | Memory Network | 8,5 | 566 | 46 |

La lectura es demoledora. En el **control set** los LMs son excelentes (N-gramas y LSTM aciertan en ~1/5 de los casos, 19-22%): son modelos genuinamente buenos. En **LAMBADA** todos colapsan a accuracy esencialmente cero, sin superar siquiera la heuristica trivial de "palabra capitalizada aleatoria del pasaje" (7,3%, que explota el sesgo de nombres propios). El **gap es el resultado central**: humanos ~86% (implicito en el protocolo de dos sujetos consecutivos correctos) frente a modelos <1%. El mejor desempeno relativo es N-Gram w/cache (perplejidad 768), porque el cache incorpora estadisticas del pasaje, pero aun asi no acierta la palabra.

---

## Por que es dificil para los LMs

La dificultad tiene dos capas. La **intrinseca**: la informacion necesaria reside fuera de la oracion objetivo, distribuida en 4-5 oraciones. Un modelo que solo captura bien la distribucion condicional local —la "senal facil" del lenguaje— no tiene de donde sacar la respuesta; necesita mantener entidades en memoria, resolver co-referencia, hacer *bridging*, inferir participantes prototipicos e integrar pistas distantes. Son fenomenos de **integracion de informacion distante**, justo lo que la perplejidad promedio no estresa.

La capa **de diseno**, que los autores reconocen con honestidad: uno de los primeros filtros descarta los pasajes que LMs simples predicen bien, asi que por construccion los LMs estandar estan condenados a fallar. Pero esto no invalida el benchmark: los humanos *si* resuelven la tarea, de modo que un modelo que afirme buena comprension del lenguaje deberia poder hacerlo. La hipotesis de los autores sobre el camino: una memoria de mas largo plazo acoplada a razonamiento sobre lo almacenado, y posiblemente mecanismos de **atencion** (Bahdanau et al., 2014) —una observacion que, con el diario del lunes, anticipa el camino real hacia la solucion.

---

## Por que importa hoy

LAMBADA se volvio un **benchmark estandar** de comprension de contexto largo, y su trayectoria es uno de los ejemplos mas nitidos de como la escala cerro un gap que en 2016 parecia infranqueable:

- **GPT-2 (2019)** reporto LAMBADA explicitamente como una de sus metricas insignia de contexto largo, mostrando saltos grandes en zero-shot al escalar el modelo. La tarea paso de "imposible" a "indicador de progreso".
- **GPT-3 (2020)** lo incluyo como prueba de dependencias de largo alcance, alcanzando en *few-shot* niveles muy por encima del estado del arte previo y acercandose al rango humano.

La moraleja es doble. Por un lado valida la tesis de Paperno et al.: la integracion de contexto amplio era una capacidad ausente y real, no un artefacto, y medirla detecto algo que la perplejidad promedio escondia. Por otro, el camino a la solucion no fue el que el paper apostó (Memory Networks con razonamiento explicito) sino la **escala de Transformers autoregresivos**, donde la atencion provee de facto el acceso a contexto distante que las RNN/LSTM de 2016 no lograban explotar. El benchmark sobrevivio a la hipotesis arquitectonica de sus propios autores. Ademas consolido una metodologia: **filtrar con humanos para aislar una capacidad especifica**, no solo recolectar etiquetas.

---

## Conexion con la Clase 24

La Clase 24 lista LAMBADA entre los **datasets pre-2015** de la era de comprension lectora / modelado de lenguaje, junto a CNN/Daily Mail y Children's Book Test (CBT). El propio paper se compara con esos parientes de la familia **cloze**:

- **CNN/Daily Mail** (Hermann et al., 2015): articulos con resumenes donde se adivina una entidad nombrada removida. Como LAMBADA, exige contexto amplio, pero su genero es noticioso, sus items se limitan a entidades y la tarea es de tipo resumen, no continuacion narrativa. Ver [CNN/Daily Mail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015).
- **CBT** (Hill et al., 2016): excerpts con una palabra removida en una secuencia de 21 oraciones. La distincion clave: **CBT no fue filtrado para ser adivinable solo con contexto amplio**, asi que muchos items se resuelven con la ultima oracion sola y solo una porcion pequena prueba comprension de contexto amplio —justo el foco exclusivo de LAMBADA. Ver [Children's Book Test (Hill 2016)](/papers/childrens-book-test-hill-2016).

LAMBADA es el test mas puro de comprension de contexto amplio del grupo, y el puente donde se ve con numeros concretos por que los modelos de 2016 no "entendian" el discurso y como la generacion siguiente lo reporto para demostrar que si empezaban a hacerlo. Ver [GPT-2 (Radford 2019)](/papers/gpt-2-radford-2019), que lo reporta como metrica insignia.

Lección de evaluacion transferible: una metrica promedio agregada puede ocultar fallas sistematicas en el subconjunto de casos dificiles. LAMBADA es, conceptualmente, un *slice* de evaluacion deliberadamente adversarial —casos donde la senal facil no basta— para descubrir que *no* aprende el modelo. Filtrar el test set para aislar la capacidad que de verdad importa, en vez de promediar sobre la distribucion natural, es directamente aplicable a la validacion de scorers donde los casos triviales inflan las metricas.

---

## Notas y enlaces

- **Paper:** Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., Fernández, R. (2016). *The LAMBADA dataset: Word prediction requiring a broad discourse context*. ACL 2016. arXiv:1606.06031.
- **Acronimo:** LAnguage Modeling Broadened to Account for Discourse Aspects.
- **Recursos:** corpus de entrenamiento + dev set en `clic.cimec.unitn.it/lambada/` (el test set se reservo para una competencia publica).
- **Corpus fuente:** BookCorpus (Zhu et al., 2015), *Aligning books and movies*, ICCV 2015.
- **Cifras clave:** 10.022 pasajes (4.869 dev / 5.153 test); contexto ~4,6 oraciones / ~75 tokens; >80% con la palabra objetivo en el contexto; 48% nombres propios; ~1 de cada 25 pasajes sobrevive al filtro; gap humano (~86%) vs. modelos (<1%, mejor relativo N-Gram w/cache con perplejidad 768).

Ver fundamentos: [Question Answering](/fundamentos/question-answering) - [Modelos de Lenguaje](/fundamentos/modelos-de-lenguaje).

Ver papers: [CNN/Daily Mail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015) - [Children's Book Test (Hill 2016)](/papers/childrens-book-test-hill-2016) - [GPT-2 (Radford 2019)](/papers/gpt-2-radford-2019).

Ver clase: [Clase 24](/clases/clase-24).
