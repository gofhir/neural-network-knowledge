# Large Language Monkeys: Scaling Inference Compute with Repeated Sampling — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*.
- **Autores:** Bradley Brown, Jordan Juravsky, Ryan Ehrlich (contribución igual), Ronald Clark, Quoc V. Le, Christopher Ré y Azalia Mirhoseini.
- **Afiliaciones:** Departamento de Ciencia de la Computación, **Universidad de Stanford**; **Universidad de Oxford**; **Google DeepMind**. Brown trabajó en el proyecto como investigador visitante en Stanford.
- **Publicación:** preprint **arXiv:2407.21787v3** (versión de 30 de diciembre de 2024). Código en `github.com/ScalingIntelligence/large_language_monkeys`; datos en el conjunto *monkey_business* de HuggingFace.
- **Título inspirado** en el *teorema del mono infinito*: un mono tecleando al azar durante tiempo infinito acabaría escribiendo cualquier texto. La analogía es deliberada: si un LLM genera suficientes intentos independientes, es probable que alguno acierte.

El paper explora una idea sencilla pero de consecuencias profundas: **el cómputo de inferencia como un eje de escalamiento independiente del cómputo de entrenamiento**. En lugar de dar al modelo un solo intento por problema —la práctica habitual—, se lo hace **muestrear repetidamente** muchas soluciones candidatas con temperatura positiva y luego se selecciona una respuesta final con un verificador. La métrica central es la **cobertura** (*coverage*): la fracción de problemas resueltos por **al menos una** de las muestras generadas. El hallazgo empírico es que la cobertura crece de forma suave y predecible con el número de muestras $k$ **a lo largo de cuatro órdenes de magnitud** (de una a diez mil muestras por problema), y que esa relación suele modelarse con una **ley de potencia exponenciada**, sugiriendo la existencia de *leyes de escala en tiempo de inferencia* análogas a las de entrenamiento.

La segunda tesis, igual de importante, es una **advertencia**: escalar la cobertura solo se traduce en desempeño real cuando existe un **verificador** capaz de identificar la muestra correcta dentro de la colección. En dominios verificables (código con tests, pruebas formales con *proof checker*) el aumento de cobertura se convierte directamente en aciertos. En dominios sin verificador automático (problemas matemáticos de respuesta abierta), los métodos comunes de selección —votación por mayoría, modelos de recompensa— **se estancan más allá de unas cien muestras** y no logran capturar las soluciones correctas raras. Este análisis es directamente relevante para la **Clase 34 (Razonamiento)**, donde el paper se cita bajo *Pass@k* para cuantificar la intuición de que "muestrear muchas respuestas hace probable que alguna llegue al resultado correcto".

## 2. Contexto: cómputo de entrenamiento vs. cómputo de inferencia

Durante la última década, la mejora dramática en las capacidades de los LLM se explicó casi enteramente por **escalar el cómputo de entrenamiento**: modelos más grandes, corridas de preentrenamiento más largas y conjuntos de datos mayores. Las *leyes de escala de entrenamiento* (Kaplan et al., Hoffmann et al./Chinchilla) formalizaron esta relación como una ley de potencia entre la pérdida y el cómputo, y dieron a los desarrolladores confianza en que grandes inversiones de entrenamiento rendirían frutos predecibles.

El cómputo de **inferencia**, en cambio, recibió una inversión comparativamente escasa. Los autores observan que, aunque técnicas como *chain-of-thought* elevan la calidad de la respuesta al costo de salidas más largas, en la práctica **usuarios y desarrolladores restringen el modelo a un solo intento por problema**. El paper propone tratar la inferencia como una segunda palanca de escalamiento, mediante la técnica más simple imaginable: **muestreo repetido** (Figura 1 del paper). El procedimiento tiene dos pasos: (1) generar muchas soluciones candidatas independientes muestreando el LLM con temperatura positiva; (2) usar un verificador específico del dominio (tests unitarios, *proof checker*, votación) para elegir una respuesta final.

### El vínculo con Pass@k

En el contexto de código, la cobertura del paper **es exactamente la métrica pass@k** de Chen et al. (2021), donde $k$ es el número de muestras por problema. Pass@k mide la probabilidad de que al menos una de $k$ muestras pase los tests. Para reducir la varianza, los autores adoptan el **estimador insesgado** de Chen et al.: se generan $N$ muestras por problema, se cuenta el número de muestras correctas $C_i$ para el problema $i$, y se calcula

$$\text{pass@}k = \frac{1}{\#\text{problemas}} \sum_{i=1}^{\#\text{problemas}} \left(1 - \frac{\binom{N-C_i}{k}}{\binom{N}{k}}\right).$$

El término $\binom{N-C_i}{k} / \binom{N}{k}$ es la probabilidad de que las $k$ muestras elegidas sean **todas** incorrectas; su complemento es la probabilidad de que al menos una sea correcta. Los autores usan la implementación numéricamente estable propuesta en el paper original. Para GSM8K y MATH, la cobertura corresponde a un **verificador oráculo** que revisa si alguna muestra produce la respuesta final correcta; para MiniF2F se usa el *proof checker* de Lean4; para CodeContests y SWE-bench Lite se usa pass@k directamente sobre los tests.

El precedente inspirador es **AlphaCode** (Li et al., *Science* 2022), sistema de programación competitiva de vanguardia que descubrió que el desempeño **sigue mejorando hasta un millón de muestras por problema**. El objetivo de este trabajo es caracterizar sistemáticamente ese beneficio a través de un rango amplio de tareas, modelos y presupuestos de muestreo.

## 3. Contribución central

El paper articula tres observaciones principales:

1. **Escalar el cómputo de inferencia mediante muestreo repetido produce grandes mejoras de cobertura** en una variedad de tareas y modelos. Esto hace posible —y a veces económicamente ventajoso— **amplificar un modelo débil con muchas muestras** hasta superar el desempeño de un solo intento de un modelo más capaz.
2. **La relación entre cobertura y número de muestras puede modelarse con una ley de potencia exponenciada**, sugiriendo una forma de leyes de escala para el cómputo en tiempo de inferencia.
3. **En dominios sin verificadores automáticos**, los métodos comunes de selección se estancan más allá de aproximadamente 100 muestras, generando una brecha creciente entre su desempeño y la cota superior que marca la cobertura.

La efectividad del muestreo repetido depende de **dos propiedades separables**:

- **Cobertura:** al aumentar $k$, ¿qué fracción de problemas podemos resolver usando *cualquiera* de las muestras generadas?
- **Precisión:** ¿con qué frecuencia podemos *identificar* las muestras correctas dentro de la colección?

Ambas son necesarias para un buen desempeño real. Con muestras ilimitadas, cualquier modelo que asigne probabilidad no nula a toda secuencia alcanza cobertura perfecta; pero el muestreo repetido solo es práctico si la cobertura mejora con un presupuesto factible **y** si las muestras correctas pueden ser identificadas. La dificultad del problema de precisión varía según la tarea: un *proof checker* o un conjunto de tests unitarios verifican automáticamente cada muestra, mientras que en problemas de texto abierto hace falta otro mecanismo de verificación.

## 4. Método y resultados de cobertura

### 4.1 Las cinco tareas

Los autores se centran en tareas de tipo *pasa/falla*, donde una solución candidata puede calificarse como correcta o incorrecta. Evalúan cinco:

1. **GSM8K:** problemas matemáticos de nivel escolar. Subconjunto aleatorio de 128 problemas de test. **Sin verificador automático.**
2. **MATH:** problemas matemáticos más difíciles. 128 problemas aleatorios de test. **Sin verificador automático.**
3. **MiniF2F-MATH:** problemas matemáticos formalizados en lenguajes de verificación de pruebas. Se usa **Lean4** y los 130 problemas de test derivados de MATH. **Verificador automático** (proof checker de Lean4).
4. **CodeContests:** problemas de programación competitiva con casos de prueba entrada-salida (ocultos al modelo). Soluciones en Python3. **Verificador automático** (casos de prueba).
5. **SWE-bench Lite:** *issues* reales de GitHub; el modelo debe editar archivos del repositorio (un solo archivo en el subconjunto Lite). **Verificador automático** (suite de tests unitarios del repositorio).

Para SWE-bench Lite, cada "muestra" es una **trayectoria multiturno completa** entre el LLM y el *framework* de agente (se usa la librería open-source **Moatless Tools**), con hasta 250 intentos independientes por *issue*.

### 4.2 El muestreo repetido es efectivo en todas las tareas

Evaluando Llama-3-8B-Instruct y Llama-3-70B-Instruct (y DeepSeek-Coder-V2-Instruct en SWE-bench, por la longitud de contexto requerida), generando hasta **10.000 muestras por problema**, los autores encuentran que la cobertura mejora suavemente con el presupuesto de muestreo en las cinco tareas. El resultado más llamativo:

> En **SWE-bench Lite**, DeepSeek-Coder-V2-Instruct resuelve **15,9% de los issues con una sola muestra**, pero **56% con 250 muestras** — superando el estado del arte de un solo intento (43%, logrado por CodeStory Aide, una mezcla de GPT-4o y Claude 3.5 Sonnet) por 13 puntos.

Cuando a todos los modelos se les da un solo intento, GPT-4o supera a los modelos Llama y DeepSeek en cada tarea. Pero al aumentar $k$, **los tres modelos más débiles superan el desempeño de un solo intento de GPT-4o**. Esta es la tesis de la *amplificación*: el muestreo repetido convierte un modelo más débil en uno que compite con —o supera a— uno más fuerte.

### 4.3 El efecto es robusto entre tamaños y familias

Ampliando la evaluación a Llama-3 (8B base, 8B-Instruct, 70B-Instruct), Gemma (2B, 7B) y Pythia (ocho modelos de 70M a 12B) sobre MATH y CodeContests, la cobertura crece en casi todos los modelos, con los **modelos más pequeños mostrando algunos de los aumentos más pronunciados**:

- **CodeContests con Gemma-2B:** la cobertura crece **más de 300×**, de un pass@1 de **0,02% a un pass@10k de 7,1%**.
- **MATH con Pythia-160M:** la cobertura crece de un pass@1 de **0,27% a un pass@10k de 57%**.

La única excepción es la familia Pythia en CodeContests: todos los modelos Pythia logran **cobertura cero** incluso con 10.000 muestras, lo que los autores atribuyen a que Pythia fue entrenada con menos datos de código que Llama y Gemma. La lección: el muestreo repetido amplifica una capacidad que el modelo **ya posee latentemente**; si la probabilidad de acierto es exactamente cero, ningún presupuesto de muestreo la rescata.

### 4.4 Leyes de escala en tiempo de inferencia

Inspirándose en el reporte técnico de GPT-4 —que modela la tasa media de log-pass-rate en problemas de código como ley de potencia del cómputo de entrenamiento—, los autores modelan el **logaritmo de la cobertura** $c$ como función del número de muestras $k$:

$$\log(c) \approx a\,k^{b},$$

y, exponenciando ambos lados, obtienen el modelo final:

$$c \approx \exp\!\left(a\,k^{b}\right),$$

donde $a, b \in \mathbb{R}$ son parámetros ajustados con `curve_fit` de SciPy. Este es un ajuste de **ley de potencia exponenciada**. Sobre un eje $x$ logarítmico, la relación se ve **casi log-lineal** en varios órdenes de magnitud. Ejemplos de parámetros ajustados: Llama-3-8B-Instruct en MATH da $a=-1{,}33,\ b=-0{,}43$ (error $0{,}003 \pm 0{,}0027$); Llama-3-8B-Instruct en CodeContests da $a=-3{,}88,\ b=-0{,}11$. Los ajustes no son tan exactos como las leyes de entrenamiento —MiniF2F-MATH es notoriamente ruidoso—, pero constituyen evidencia temprana alentadora de que los beneficios del escalamiento de inferencia pueden caracterizarse.

Una segunda regularidad: para una tarea dada, las curvas de cobertura de **modelos de la misma familia** tienen forma de **curva sigmoide (S) con pendientes similares pero desplazamientos horizontales distintos**. Superponiendo las curvas y desplazándolas en el eje log hasta hacerlas pasar por un punto de anclaje común $(1, c)$, se ve que coinciden en forma. Esto implica que, dentro de una familia, **el aumento multiplicativo del presupuesto de muestreo necesario para pasar la cobertura de $c$ a $c'$ es aproximadamente constante**.

### 4.5 La economía: muestrear mucho de un modelo barato

El aporte práctico más importante para un profesional es que el muestreo repetido **abre un nuevo grado de libertad para optimizar conjuntamente desempeño y costo**.

Midiendo el costo en **FLOPs de inferencia** (aproximados con una fórmula explícita para transformers densos), los autores re-grafican la cobertura como función del cómputo total en vez del número de muestras. El hallazgo es que **el modelo que maximiza la cobertura depende de la tarea y del presupuesto**: en MiniF2F, GSM8K y MATH, Llama-3-8B-Instruct obtiene siempre mayor cobertura que el modelo 70B (más caro) a igual presupuesto de FLOPs; pero en CodeContests el modelo 70B es casi siempre más eficiente. Es decir, a veces conviene **muestrear muchas veces un modelo pequeño**, y a veces **pocas veces uno grande**.

Midiendo el costo en **dólares de API** sobre SWE-bench Lite con el mismo *framework* de agente (Moatless Tools), la Tabla 1 del paper compara un solo intento de Claude 3.5 Sonnet y GPT-4o contra **cinco muestras** de DeepSeek-Coder-V2-Instruct:

| Modelo | Costo por intento (USD) | Nº intentos | Issues resueltos (%) | Costo total (USD) | Costo relativo |
|---|---|---|---|---|---|
| DeepSeek-Coder-V2-Instruct | 0,0072 | 5 | 29,62 | 10,8 | 1× |
| GPT-4o | 0,13 | 1 | 24,00 | 39 | 3,6× |
| Claude 3.5 Sonnet | 0,17 | 1 | 26,70 | 51 | 4,7× |

El modelo DeepSeek es más débil, pero **más de 10× más barato por intento**. Muestreándolo cinco veces resuelve **más issues** que un solo intento de Claude o GPT (29,62% vs. 26,70% y 24,00%) y a la vez cuesta **más de 3× menos**. Los autores además señalan que el muestreo repetido es una **carga de inferencia distinta** de servir un chatbot: como todas las muestras comparten el mismo prompt, se puede usar *batch* grande y optimizaciones de atención con prefijo compartido (Hydragen, SGLang), maximizando el *throughput* y abaratando el muestreo por debajo del costo de hacer muchas peticiones ingenuas.

## 5. El rol del verificador: la limitación central

Aquí está el corazón crítico del paper. Toda la sección anterior mide **cobertura**, que asume implícitamente un verificador oráculo perfecto. Pero la cobertura es solo una **cota superior** del desempeño real; convertir cobertura en aciertos requiere resolver el problema de **precisión**: encontrar "la aguja en el pajar" —las muestras correctas raras dentro de una colección mayoritariamente incorrecta.

De las cinco tareas, solo **GSM8K y MATH carecen de verificador automático**. Los autores prueban tres métodos comunes de selección sobre las colecciones de 10.000 muestras:

1. **Votación por mayoría:** elegir la respuesta final más común (esto es exactamente *self-consistency*).
2. **Modelo de recompensa + Best-of-N:** puntuar cada solución con un modelo de recompensa (ArmoRM-Llama3-8B) y elegir la de mayor puntaje.
3. **Modelo de recompensa + votación por mayoría:** votación ponderada por el puntaje del modelo de recompensa.

El resultado (Figura 7 del paper) es contundente. Con **Llama-3-8B-Instruct en MATH**, la **cobertura crece de 82,9% (100 muestras) a 98,44% (10.000 muestras)**. Pero el mayor aumento de desempeño de los métodos de selección en ese mismo rango es de apenas **40,50% a 41,41%**. Los tres métodos **se estancan alrededor de las 100 muestras** mientras la cobertura sigue creciendo por encima del 95%. La brecha entre cobertura (desempeño con verificador perfecto) y desempeño real **crece** con el número de muestras.

Para la votación por mayoría, esta saturación es intuitiva: la aparición de soluciones correctas raras **no cambia la respuesta más común**, así que agregar más muestras no ayuda. Lo preocupante es que el **modelo de recompensa** tampoco logra escalar: no distingue de manera confiable las soluciones correctas infrecuentes.

¿Es que verificar es tan difícil como resolver? Para responderlo, los autores evalúan manualmente **105 cadenas de razonamiento (chains-of-thought)** de muestras correctas de Llama-3-8B-Instruct en GSM8K. Encuentran que **más del 90% de las cadenas son fieles** (siguen pasos lógicos válidos), incluso en problemas donde el modelo acierta pocas veces. Esto indica que **hay señal que un verificador podría explotar**; el problema no es que las soluciones correctas sean espurias, sino que los verificadores actuales no la aprovechan. (De paso, el análisis descubrió un problema de GSM8K con *ground truth* incorrecto — el único que Llama-3-70B no "resolvió" en 10.000 intentos.)

### Verificadores imperfectos: dos cuentos con moraleja

Incluso los verificadores "automáticos" no son perfectos; el software ocupa un terreno intermedio:

- **Tests inestables (*flaky*) en SWE-bench Lite:** el **11,3% de los problemas** (34 en total) tienen suites de tests que dan resultados inconsistentes sobre la misma solución. En 30 de esos 34 casos la inestabilidad afecta incluso a las soluciones de referencia del dataset. Ejemplo: dos issues manipulan conjuntos de Python (no ordenados); soluciones correctas que no imponen un orden pasan los tests solo en corridas "afortunadas". Los autores repiten los tests 11 veces y usan votación por mayoría.
- **Falsos negativos en CodeContests:** cuando un problema admite múltiples salidas correctas pero los tests exigen una específica, o cuando casos de prueba generados por mutación violan la especificación del problema, soluciones correctas fallan los tests. De los 122 problemas de test con soluciones en Python3, **35 tienen soluciones "correctas" que fallan sus tests**. El muestreo repetido en estos casos incluye un elemento de "tirar los dados" para producir la salida exacta que pasa.

La conclusión del paper es clara: **la verificación escalable es necesaria para beneficiarse plenamente del muestreo repetido**. Equipar a los modelos con la capacidad de evaluar sus propias salidas —o diseñar conversores que hagan verificable una tarea no estructurada (por ejemplo, formalizar un enunciado matemático informal a Lean)— es la frontera abierta.

## 6. Conexión con la Clase 34 (Razonamiento)

Este paper es una de las piezas fundacionales del paradigma de **cómputo en tiempo de test** (*test-time compute*) que domina la generación actual de modelos de razonamiento. Su relevancia para la Clase 34 opera en tres niveles:

**Pass@k y la intuición del muestreo.** El paper cuantifica precisamente la afirmación citada en la clase (slide 33): si se muestrean muchas respuestas, es probable que alguna sea correcta. La cobertura *es* pass@k, y su crecimiento log-lineal sobre cuatro órdenes de magnitud le da forma matemática a esa intuición. Pero el paper también inyecta el matiz esencial: **pass@k es una cota superior optimista** que solo se materializa con un buen verificador.

**El puente hacia o1 y R1.** El muestreo repetido es una de las formas más simples de gastar más cómputo en inferencia. Los modelos de razonamiento posteriores —**o1 de OpenAI, DeepSeek-R1**— internalizan esta idea: en vez de muestrear muchas trayectorias cortas *en paralelo* y verificarlas externamente, aprenden por RL a producir **una sola trayectoria de razonamiento larga** que explora, verifica y corrige *dentro* de su propia cadena de pensamiento (búsqueda "en serie" en el espacio de tokens de deliberación). Este paper es el trabajo que estableció empíricamente que **el cómputo de inferencia es un eje de escala legítimo**, con sus propias leyes de potencia — la premisa que hace sensato entrenar modelos que "piensen más". La sección de trabajo relacionado del paper ya conecta con búsqueda en árbol (Tree-of-Thoughts, Graph-of-Thoughts), deliberación con tokens y auto-crítica (Self-Refine), todo el ecosistema del razonamiento en tiempo de inferencia.

**Self-Consistency como caso particular.** La votación por mayoría que el paper evalúa como método de precisión es exactamente **Self-Consistency** (Wang et al., 2023), la técnica que muestrea varias cadenas de razonamiento y elige la respuesta más votada. El aporte crítico de este paper es mostrar **por qué Self-Consistency se estanca**: la votación por mayoría es insensible a las soluciones correctas raras, así que satura alrededor de las 100 muestras aun cuando la cobertura sigue subiendo. Para tareas donde la respuesta correcta es minoritaria, Self-Consistency deja sobre la mesa casi toda la ganancia que un verificador perfecto capturaría. Esto motiva la investigación en **verificadores** —modelos de recompensa de proceso, *process supervision*— como complemento indispensable del muestreo.

**Enlaces internos:**

- Clase: [/clases/clase-34](/clases/clase-34) — Razonamiento (Pass@k, test-time compute).
- Paper hermano de selección: cualquier análisis de *Self-Consistency* (Wang et al., 2023) — la votación por mayoría cuya saturación este paper diagnostica.
- Fundamento transversal: escalamiento en tiempo de inferencia y leyes de escala.

## 7. Nota final: relevancia para salud

Para un sistema de IA clínica, este paper entrega una lección precisa y sobria: **el muestreo repetido solo mejora el desempeño real cuando existe un verificador confiable, y en medicina ese verificador casi nunca existe en el momento de la decisión**. La distinción del paper entre tareas verificables (código con tests, teoremas con *proof checker*) y no verificables se mapea directamente al problema clínico. Un diagnóstico diferencial, una interpretación de imagen o una recomendación de tratamiento no tienen un *ground truth* ejecutable contra el cual chequear cada muestra al instante; la "verdad" llega después —con la evolución del paciente, la biopsia o el seguimiento— o nunca llega de forma inequívoca. En ese régimen, aunque muestrear al modelo mil veces elevara la *cobertura* (la probabilidad de que **alguna** respuesta sea correcta), esa cobertura **no se traduce en acierto**, porque el sistema no puede identificar cuál de las mil es la buena — y el paper demuestra que la votación por mayoría y los modelos de recompensa se estancan justamente donde más se los necesita. Peor aún: presentar una respuesta "más votada" con falsa confianza puede enmascarar que la respuesta correcta estaba presente pero fue descartada. La conclusión operativa para salud es que el muestreo repetido es valioso solo en los eslabones **verificables** del flujo clínico —código de análisis, cálculos de dosis con validación de unidades, conciliación estructurada de datos contra reglas explícitas— y que invertir cómputo de inferencia sin invertir primero en un verificador confiable produce la ilusión de rigor sin el rigor.
