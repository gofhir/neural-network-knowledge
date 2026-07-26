# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*.
- **Autores:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou. Todos en **Google Research, Brain Team**.
- **Venue:** *36th Conference on Neural Information Processing Systems* (**NeurIPS 2022**).
- **Preprint:** arXiv:2201.11903 (versión v6, 10 de enero de 2023), [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903).
- **Una frase:** basta con **mostrarle al modelo unos pocos ejemplos que incluyan los pasos intermedios de razonamiento**, y un modelo de lenguaje suficientemente grande empieza a resolver problemas de razonamiento que antes fallaba, sin entrenar ni ajustar un solo parámetro.

El paper introduce el **chain-of-thought prompting** (prompting con cadena de pensamiento): una técnica de *few-shot* en la que cada ejemplo (exemplar) del prompt no es un par ⟨entrada, salida⟩ sino un **triple** ⟨entrada, cadena de pensamiento, salida⟩. La "cadena de pensamiento" es una serie de pasos de razonamiento intermedios en lenguaje natural que conducen a la respuesta final. Al ver unos pocos de estos triples, el modelo aprende a **generar su propia cadena de pensamiento** antes de responder, en vez de saltar directamente a la respuesta.

El hallazgo central y más citado es que esta capacidad es **emergente con la escala**: la cadena de pensamiento **no ayuda —incluso perjudica— en modelos pequeños**, y solo produce ganancias a partir de aproximadamente **100.000 millones (100B) de parámetros**. El resultado estrella: con solo **ocho exemplars** de cadena de pensamiento, **PaLM 540B alcanza el estado del arte en GSM8K** (problemas matemáticos de enunciado verbal), superando incluso a un GPT-3 *finetuneado* con un verificador. No se ajustó ningún modelo en todo el trabajo: todo se logra únicamente por prompting sobre modelos *off-the-shelf*.

Para la **Clase 34 (Razonamiento)** este es EL paper central. Es la bisagra que convierte el "scratchpad" (Nye et al., 2021, cómputo intermedio para tareas simbólicas) en una técnica general para elicitar razonamiento multi-paso, y es el ancestro directo del paradigma de **test-time compute** que hoy encarnan modelos como o1 y R1. Es el ejemplo de la slide 28 con Natalia y GSM8K.

## 2. Contexto: los LLMs fallan en aritmética simple pese a su escala

El panorama del NLP se transformó al escalar los modelos de lenguaje (Peters et al., 2018; Devlin et al., 2019; Brown et al., 2020). Escalar confiere beneficios predecibles —mejor desempeño y mejor eficiencia de muestras (Kaplan et al., 2020)— que siguen **leyes de escala suaves y monótonas**. El problema que motiva el paper es que **escalar el tamaño del modelo, por sí solo, no basta** para lograr buen desempeño en tareas difíciles como razonamiento aritmético, de sentido común y simbólico (Rae et al., 2021).

Esto es contraintuitivo y algo embarazoso: un modelo de 175B parámetros que escribe ensayos fluidos y traduce entre idiomas puede fallar en un problema aritmético que un niño resuelve. La razón es que la aritmética de varios pasos exige **encadenar operaciones**, y el *standard prompting* obliga al modelo a producir la respuesta en un solo paso hacia adelante (*one forward pass*), sin espacio para desplegar el cálculo. En esas tareas, la curva de escala del prompting estándar es **plana**: agrandar el modelo casi no mueve la aguja.

Antes de este trabajo existían dos líneas para inyectar pasos intermedios, ambas con limitaciones:

1. **Racionales entrenados o finetuneados.** Ling et al. (2017) fueron pioneros en generar racionales en lenguaje natural para resolver problemas matemáticos; Cobbe et al. (2021) extendieron la idea creando un dataset grande (GSM8K) y finetuneando un modelo preentrenado, además de un **verificador**. El costo: crear un conjunto grande de racionales de alta calidad es caro, mucho más que los simples pares entrada–salida.
2. **Few-shot prompting** al estilo Brown et al. (2020): en vez de finetunear un checkpoint por tarea, se "promptea" el modelo con unos pocos ejemplos entrada–salida. El costo: **funciona mal en tareas que requieren razonamiento** y no mejora sustancialmente al escalar (Rae et al., 2021).

La contribución del paper es **combinar las fortalezas de ambas ideas evitando sus limitaciones**: usar few-shot prompting (barato, sin dataset ni finetuning) pero con exemplars que incluyen la cadena de razonamiento (los pasos intermedios que hacen viable la tarea).

## 3. Contribución central

La contribución es a la vez **un método y un descubrimiento empírico**:

- **El método** es el chain-of-thought prompting: aumentar cada exemplar del few-shot con una cadena de pensamiento. Es un método puramente de prompting, lo que tiene dos virtudes prácticas que el paper subraya: no requiere un dataset de entrenamiento grande, y **un único checkpoint sirve para muchas tareas** sin pérdida de generalidad.
- **El descubrimiento** es que el razonamiento por cadena de pensamiento es una **habilidad emergente de la escala** (Wei et al., 2022b): aparece de golpe alrededor de los 100B parámetros y no puede predecirse extrapolando modelos pequeños. En modelos por debajo de ~10B parámetros, la cadena de pensamiento **empeora** el desempeño respecto del prompting estándar.

El paper enumera cuatro propiedades atractivas de la cadena de pensamiento:

1. **Cómputo adaptativo.** Permite descomponer problemas multi-paso en pasos intermedios, de modo que se asigna **más cómputo a los problemas que requieren más razonamiento** (más tokens intermedios). Esta es la semilla conceptual del *test-time compute*.
2. **Interpretabilidad.** Ofrece una ventana legible al comportamiento del modelo: sugiere cómo llegó a una respuesta y da oportunidad de depurar dónde se torció el razonamiento (aunque caracterizar del todo el cómputo del modelo sigue abierto).
3. **Generalidad.** Sirve para problemas matemáticos, de sentido común, de manipulación simbólica, y en principio para **cualquier tarea que los humanos resuelvan mediante lenguaje**.
4. **Facilidad.** Se elicita en modelos grandes *off-the-shelf* con solo incluir ejemplos de cadena de pensamiento en el prompt. Nada de finetuning.

## 4. Método

### 4.1. Qué es una cadena de pensamiento

El punto de partida es introspectivo: piensa en tu propio proceso al resolver un problema matemático de varios pasos. Es típico **descomponerlo en pasos intermedios** y resolver cada uno antes de dar la respuesta final: *"Después de que Jane le da 2 flores a su mamá le quedan 10... luego de darle 3 a su papá tendrá 7... así que la respuesta es 7."* El objetivo del paper es dotar al modelo de la capacidad de generar una cadena así —una serie coherente de pasos intermedios de razonamiento que conducen a la respuesta.

Formalmente, el prompt consiste en triples

$$\langle \text{input},\ \text{chain of thought},\ \text{output} \rangle$$

donde la *chain of thought* es la secuencia de pasos de razonamiento en lenguaje natural. Contrasta con el *standard prompting* de Brown et al. (2020), donde el prompt consiste en pares ⟨input, output⟩ y el modelo da la respuesta directamente.

### 4.2. El ejemplo canónico (Figura 1)

El ejemplo de la Figura 1 —el mismo de la slide 28 de la clase— ilustra el contraste:

**Standard prompting.** El exemplar termina en "A: The answer is 11." Ante la pregunta de test ("The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"), el modelo responde directamente:

> A: The answer is 27. ❌ (incorrecto)

**Chain-of-thought prompting.** El exemplar incluye el razonamiento: "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. $5 + 6 = 11$. The answer is 11." Ante la misma pregunta de test, el modelo ahora **despliega los pasos**:

> A: The cafeteria had 23 apples originally. They used 20 to make lunch. So they had $23 - 20 = 3$. They bought 6 more apples, so they have $3 + 6 = 9$. The answer is 9. ✓ (correcto)

La clave: **el problema no cambió; cambió el formato de la demostración**. Y con eso, un problema que el modelo habría fallado ahora lo acierta. Los autores optan por llamarlo "cadena de pensamiento" y no "solución" para capturar la idea de que **imita un proceso de pensamiento paso a paso** (además, las soluciones y explicaciones suelen venir *después* de la respuesta, no antes).

### 4.3. Detalles del setup

- **Exemplars.** Como la mayoría de los datasets solo tienen partición de evaluación, los autores **compusieron manualmente un conjunto de ocho exemplars** con cadenas de pensamiento, usados para todos los benchmarks (excepto AQuA, de opción múltiple, con cuatro exemplars). Estos exemplars **no pasaron por *prompt engineering***; su robustez se estudia después.
- **Decodificación.** Se muestrea por *greedy decoding* (trabajo posterior —Wang et al., 2022a, self-consistency— mejora esto tomando la respuesta mayoritaria sobre muchas generaciones muestreadas).
- **Sin finetuning.** No se ajustó ningún modelo. Todo es prompting sobre modelos preentrenados.

### 4.4. Los modelos evaluados

Se evalúan **cinco familias**, cubriendo un amplio rango de escalas —esencial para detectar la emergencia:

- **GPT-3** (Brown et al., 2020): text-ada/babbage/curie/davinci-001/002, ≈ 350M, 1.3B, 6.7B y 175B parámetros.
- **LaMDA** (Thoppilan et al., 2022): 422M, 2B, 8B, 68B y 137B.
- **PaLM**: 8B, 62B y 540B.
- **UL2 20B** (Tay et al., 2022).
- **Codex** (code-davinci-002).

## 5. Experimentos y resultados

Tres dominios: aritmético, de sentido común, y simbólico. Tres conclusiones transversales estructuran todo:

### 5.1. Razonamiento aritmético

Cinco benchmarks de problemas de enunciado verbal: **GSM8K** (Cobbe et al., 2021), **SVAMP** (problemas con estructuras variadas, Patel et al., 2021), **ASDiv** (problemas diversos, Miao et al., 2020), **AQuA** (algebraicos, opción múltiple) y **MAWPS** (Koncel-Kedziorski et al., 2016).

El resultado emblemático está en GSM8K (Figura 2). Con PaLM 540B:

| | GSM8K solve rate |
|---|---|
| PaLM 540B, standard prompting | ≈ 18% |
| Finetuned GPT-3 175B (Cobbe et al.) | 33% |
| Prior best (finetuned + verificador) | 55% |
| **PaLM 540B, chain-of-thought prompting** | **57%** |

La cadena de pensamiento **más que triplica** el desempeño del prompting estándar y **supera el estado del arte anterior**, que se lograba con finetuning y un verificador. Los números exactos de la Tabla 1: PaLM 540B pasa de **17.9% (standard) a 56.9% (CoT)**, un salto de **+39.0 puntos**; agregando una calculadora externa post-hoc para la aritmética sube a 58.6%. En los otros benchmarks con PaLM 540B: SVAMP 69.4 → 79.0, ASDiv 72.1 → 73.9, AQuA 25.2 → 35.8, MAWPS 79.2 → 93.3. PaLM con CoT logra nuevo SOTA en GSM8K, SVAMP y MAWPS, y queda a menos de 2% del SOTA en AQuA y ASDiv.

Las **tres conclusiones clave**:

1. **Es una habilidad emergente de la escala.** La cadena de pensamiento no impacta positivamente a los modelos pequeños; **solo produce ganancias con modelos de ~100B parámetros**. Los autores notan cualitativamente que los modelos pequeños producen cadenas *fluidas pero ilógicas*, lo que las lleva a un desempeño **peor** que el estándar. En la Tabla 2 se ve con nitidez: LaMDA 8B en GSM8K cae de 3.2% (standard) a 1.6% (CoT); GPT 6.7B cae de 4.0 a 2.4; recién LaMDA 137B (6.5 → 14.3), GPT-3 175B (15.6 → 46.9) y PaLM 540B (17.9 → 56.9) muestran el salto.
2. **Las ganancias son mayores en problemas más difíciles.** En GSM8K (el dataset con menor baseline) el desempeño **más que se duplicó** para los mayores GPT y PaLM. En cambio, en SingleOp (el subconjunto más fácil de MAWPS, de un solo paso), las mejoras fueron negativas o mínimas.
3. **Comparado con SOTA supervisado.** GPT-3 175B y PaLM 540B con CoT compiten favorablemente con métodos que finetunean un modelo específico por tarea sobre un dataset etiquetado —sin haber finetuneado nada.

### 5.2. Razonamiento de sentido común

Aunque la cadena de pensamiento parece hecha para matemáticas, su **naturaleza lingüística** la hace aplicable a razonamiento de sentido común (sobre interacciones físicas y humanas bajo conocimiento de fondo general). Cinco datasets: **CSQA** (Talmor et al., 2019), **StrategyQA** (estrategias multi-hop, Geva et al., 2021), **Date Understanding** y **Sports Understanding** (de BIG-bench) y **SayCan** (mapear instrucciones a acciones de robot, Ahn et al., 2022).

Con PaLM 540B y CoT: escalar mejora el standard prompting en todas las tareas, y la cadena de pensamiento añade ganancias adicionales, mayores para PaLM 540B. Resultados destacados: **supera el SOTA previo en StrategyQA (75.6% vs 69.4%)** y a un aficionado deportivo humano en Sports Understanding (**95.4% vs 84%**). La ganancia fue mínima en CSQA. Esto demuestra que la cadena de pensamiento no es solo una muleta aritmética.

### 5.3. Razonamiento simbólico y generalización de longitud

Dos tareas de juguete: **concatenación de últimas letras** (p.ej., "Amy Brown" → "yn") y **coin flip** (rastrear si una moneda sigue con la cara arriba tras una secuencia de personas que la voltean o no). Son triviales para humanos pero desafiantes para modelos.

Dos regímenes de evaluación:

- **In-domain:** los ejemplos de test tienen el mismo número de pasos que los exemplars. Con PaLM 540B, CoT logra **casi 100% de solve rate**. Nota fina: incluso aquí, donde la estructura de solución perfecta ya está en los exemplars y el modelo solo debe repetir los pasos con símbolos nuevos, **los modelos pequeños fallan** —la capacidad de manipulación abstracta sobre símbolos nuevos solo emerge a escala de 100B.
- **Out-of-domain (OOD):** los ejemplos de test tienen **más pasos** que los exemplars (nombres de 3–4 palabras vistos tras exemplars de 2 palabras). El standard prompting **falla por completo**. Con CoT, los modelos logran curvas de escala ascendentes: la cadena de pensamiento **facilita la generalización de longitud** más allá de las cadenas vistas, para modelos de escala suficiente.

## 6. Análisis y ablaciones: descartando explicaciones alternativas

Este es el corazón argumentativo del paper. Si la cadena de pensamiento simplemente diera "más cómputo" o "más tokens", el hallazgo sería menos interesante. Los autores diseñan **tres ablaciones** (Figura 5, con LaMDA 137B y PaLM 540B) para aislar qué es lo que realmente importa:

1. **Solo ecuación (*equation only*).** Se promptea al modelo a producir **solo la ecuación matemática** antes de la respuesta. En GSM8K **no ayuda mucho**: la semántica de los enunciados es demasiado compleja para traducirla directamente a una ecuación sin los pasos de razonamiento en lenguaje natural. (En datasets de uno o dos pasos, como SVAMP/ASDiv/MAWPS, sí ayuda, porque la ecuación se deriva fácil del enunciado.)
2. **Solo cómputo variable (*variable compute only*).** Para aislar la hipótesis de que la cadena solo sirve para "gastar más cómputo" en problemas difíciles, se promptea al modelo a emitir **una secuencia de puntos (`. . .`)** de longitud igual a los caracteres de la ecuación necesaria. Este variante **rinde igual que el baseline**: **el cómputo variable por sí solo NO explica el éxito** de la cadena de pensamiento. Hay utilidad específica en expresar los pasos vía lenguaje natural.
3. **Razonamiento después de la respuesta (*chain of thought after answer*).** Para descartar que la cadena solo sirva para "activar" conocimiento relevante del preentrenamiento, se coloca la cadena **después** de la respuesta. También **rinde igual que el baseline**: el razonamiento secuencial es útil por razones que van más allá de activar conocimiento; el modelo **depende de la cadena producida antes** para dar la respuesta.

Las tres ablaciones convergen en la misma conclusión: **lo que importa es el razonamiento secuencial en lenguaje natural desplegado antes de responder**, no el cómputo extra ni la mera activación de conocimiento.

**Robustez.** La sensibilidad a los exemplars es una preocupación central del prompting (variar la permutación puede llevar la accuracy de GPT-3 en SST-2 de 54.3% al 93.4%). Los autores prueban cadenas escritas por **tres anotadores independientes** (A, B, C), una versión más concisa, y **exemplars muestreados aleatoriamente del training set de GSM8K** (escritos por *crowdworkers sin background en ML*). Aunque hay varianza —en coin flip, el Anotador A logra 99.6% y el C 71.4%, ambos por encima del standard 50%—, **todas las variantes superan el baseline por amplio margen**. El éxito de la cadena de pensamiento **no depende de un estilo lingüístico particular** ni de un conjunto específico de anotadores. Los autores aclaran, honestos, que el *prompt engineering* **sí importa** en casos difíciles (hubo tareas, como revertir una lista de 5 ítems, que solo un coautor logró resolver con la cadena adecuada).

**¿Por qué la escala ayuda?** Un análisis de errores manual arroja luz. En LaMDA 137B sobre GSM8K: de 50 ejemplos con respuesta correcta, **todas las cadenas eran lógica y matemáticamente correctas excepto dos** que llegaron a la respuesta por coincidencia. De 50 con respuesta incorrecta, el **46% eran "casi correctas"** (error de calculadora, de mapeo de símbolo, o un paso faltante) y el 54% tenían errores mayores de comprensión semántica o coherencia. Al escalar PaLM de 62B a 540B, se **corrige una porción sustancial** de los errores: sobre 45 errores del 62B categorizados como comprensión semántica (20), un-paso-faltante (18) y otros (7), escalar a 540B arregla 6, 12 y 4 respectivamente. La emergencia parece involucrar múltiples habilidades que maduran con la escala: comprensión semántica, mapeo de símbolos, aritmética, mantenerse en tema, fidelidad.

## 7. Limitaciones

Los autores son explícitos y prudentes:

1. **No garantiza "razonamiento" real.** Aunque la cadena emula el proceso de un razonador humano, esto **no responde si la red neuronal está realmente "razonando"**; lo dejan como pregunta abierta.
2. **No hay garantía de caminos de razonamiento correctos.** Una cadena puede llevar tanto a respuestas correctas como incorrectas; mejorar la **factualidad** de las generaciones es dirección abierta. El propio checklist del paper advierte que la cadena de pensamiento generada **no siempre es factual** y no sugieren usarla de forma factual en escenarios del mundo real sin más cuidado.
3. **Emerge solo a gran escala.** Esto la hace **costosa de servir** en aplicaciones reales; inducir razonamiento en modelos pequeños queda como trabajo futuro.
4. **Costo de anotación.** En few-shot es mínimo, pero podría ser prohibitivo para finetuning (mitigable con generación sintética o generalización zero-shot).

## 8. Conexión con la Clase 34 (Razonamiento)

La Clase 34 recorre la **escalera de la causalidad** de Pearl, la abstracción y sistematicidad, y la distinción **System 1 vs System 2** (Kahneman): el pensamiento rápido, automático y asociativo frente al lento, deliberado y secuencial. En esa lente, el prompting estándar es **System 1 puro**: el modelo produce la respuesta en un único paso hacia adelante, por asociación. La cadena de pensamiento es el intento de dotar al modelo de algo parecido a **System 2**: desplegar un proceso secuencial y deliberado de pasos intermedios antes de comprometerse con una respuesta.

Es el punto exacto del **arco de razonamiento en LLMs** que traza la clase:

- **Scratchpad (Nye et al., 2021).** El antecedente directo, citado en el related work del paper: se le da al modelo un "borrador" para escribir resultados computacionales intermedios (p.ej., predecir línea a línea la ejecución de un programa Python), y la predicción paso a paso supera a predecir la salida final directamente. El scratchpad demostró el principio en tareas simbólicas/programáticas; **la cadena de pensamiento lo generaliza a razonamiento en lenguaje natural** y muestra que es una habilidad emergente.
- **Chain-of-Thought (este paper).** La bisagra: convierte "escribir pasos intermedios" en un método general de prompting y descubre la emergencia con la escala. Standard prompting solo da una **cota inferior** de las capacidades de un LLM.
- **Test-time compute (o1, R1).** La descendencia. La propiedad #1 del paper —*asignar más cómputo a problemas que requieren más pasos*— es precisamente la idea que hoy se lleva al extremo: en lugar de dar unos pocos exemplars, se **entrena al modelo (vía RL) para generar cadenas de pensamiento largas por sí mismo** y se le deja "pensar" más tiempo en inferencia. o1 de OpenAI y R1 de DeepSeek son la materialización de que **más cómputo en test-time, gastado en razonamiento explícito, mejora el desempeño en tareas difíciles**. La cadena de pensamiento de Wei et al. es el eslabón conceptual que hizo pensable ese salto: primero se demostró que el razonamiento explícito ayuda (prompting), luego que puede internalizarse y escalarse (entrenamiento + test-time compute).

Vale la pena que el estudiante internalice tres ideas:

1. **Formato > parámetros (a veces).** El mismo modelo, con el mismo peso, resuelve o falla según cómo se le pida razonar. El conocimiento estaba latente; el prompting lo desbloquea.
2. **Emergencia.** Ciertas capacidades **no se extrapolan** desde modelos pequeños: aparecen de golpe a cierta escala. Esto rompe la intuición de las leyes de escala suaves.
3. **Razonamiento explícito ≠ razonamiento correcto.** La cadena es interpretable y suele ser fiel, pero **no hay garantía** de que el camino sea válido —una advertencia crucial para cualquier uso de alto riesgo.

**Enlaces internos:**

- Clase: [/clases/clase-34](/clases/clase-34) — Razonamiento (Pearl, System 1/2, arco de razonamiento en LLMs).
- Antecedente scratchpad: Nye et al. (2021), *Show your work: Scratchpads for intermediate computation*.
- Descendencia test-time compute: o1 (OpenAI, 2024), R1 (DeepSeek, 2025).
- Mejora directa: Wang et al. (2022a), *Self-consistency* — respuesta mayoritaria sobre muchas cadenas muestreadas.

## 9. Nota final: relevancia para salud y registros clínicos

En el dominio clínico, la cadena de pensamiento tiene un valor que trasciende la accuracy: aporta **trazabilidad y auditabilidad** al razonamiento de un modelo. Un LLM que emite solo un veredicto ("estos dos registros corresponden al mismo paciente", o "este cuadro sugiere X") es una caja negra inaceptable para una decisión clínica o de gobernanza de datos. En cambio, un modelo que **despliega los pasos** —"el RUT coincide salvo el dígito verificador, la fecha de nacimiento es idéntica, el nombre difiere por una transposición de caracteres compatible con error de tipeo, por lo tanto es alta la probabilidad de que sea el mismo paciente"— produce un **rastro de razonamiento explícito y revisable** por un humano, exactamente lo que exige el *patient matching* auditable y el razonamiento clínico responsable. La misma limitación del paper aplica con más fuerza aquí: la cadena es interpretable pero **no está garantizada como correcta ni factual**, de modo que en salud debe tratarse como una **hipótesis argumentada que un revisor humano valida**, no como una prueba —el valor está en hacer *inspeccionable* el razonamiento, no en delegarle la decisión.
