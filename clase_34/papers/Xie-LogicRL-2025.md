# Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning*.
- **Autores:** Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu y Chong Luo (autor de correspondencia). Afiliaciones: **Microsoft Research Asia (MSRA)**, **Ubiquant** e investigadores independientes. Varios trabajos se realizaron durante pasantías en MSRA; el proyecto es de código abierto.
- **Preprint:** arXiv:2502.14768v1 [cs.CL], 20 de febrero de 2025.
- **Linaje:** es una respuesta directa e inspirada en **DeepSeek-R1** (DeepSeek-AI, 2025). Toma el diseño de recompensas basado en reglas de R1 y lo lleva a un entorno de laboratorio controlado —puzzles lógicos sintéticos— para estudiar la dinámica del razonamiento con rigor experimental.

El paper investiga hasta qué punto el **aprendizaje por refuerzo (RL) basado en reglas** puede inducir capacidades de razonamiento en modelos de lenguaje de tamaño moderado. En lugar de entrenar sobre matemáticas —el banco de pruebas habitual, pero de dificultad heterogénea e incontrolable—, los autores usan puzzles de **caballeros y bribones (Knights and Knaves, K&K)** generados proceduralmente, cuya dificultad se puede ajustar con precisión y cuya respuesta se verifica de forma determinística. Sobre este entorno, con solo **~5.000 problemas de lógica**, un modelo base de **7B** desarrolla habilidades avanzadas de razonamiento —reflexión, verificación, exploración de alternativas, resumen sistemático— que **no estaban presentes en el corpus de entrenamiento**.

El resultado más notable es la **generalización fuera de dominio**: pese a haber visto únicamente puzzles lógicos, el modelo mejora **+125% en AIME** y **+38% en AMC**, dos benchmarks matemáticos de competición muy difíciles, contra el modelo base. Este salto entre dominios sugiere que el RL no enseña patrones específicos de K&K, sino **esquemas abstractos de resolución de problemas** que transfieren a las matemáticas.

Para la **Clase 34 (Razonamiento)** este trabajo importa porque aparece citado (slide 35) bajo la etiqueta de **"aha-moments"** como ejemplo canónico de **RL con recompensa verificable** que induce razonamiento sin supervisión de proceso. Logic-RL ofrece un microscopio: al aislar las variables en un entorno sintético controlado, permite estudiar *cómo* y *cuándo* emergen los comportamientos de razonamiento que R1 popularizó, y matiza empíricamente qué significa exactamente el famoso "aha moment".

## 2. Contexto: RL con recompensa verificable después de R1

La fase de **post-entrenamiento** de los LLMs avanzó de forma vertiginosa con modelos como **DeepSeek-R1**, **Kimi-K1.5** y **OpenAI-o1**, todos capaces de razonamiento sofisticado. La contribución conceptual de DeepSeek-R1 fue mostrar que un esquema **simple de RL basado en reglas** —recompensar la respuesta correcta y el formato correcto, sin más— basta para hacer emerger patrones de razonamiento, **sin recurrir a andamiajes tradicionales** como Monte Carlo Tree Search (MCTS) ni a Process Reward Models (PRM) que puntúan cada paso intermedio.

Este paradigma se conoce como **RL con recompensa verificable** (RLVR, por *Reinforcement Learning with Verifiable Rewards*): a diferencia de RLHF —donde un modelo de recompensa aprendido aproxima la preferencia humana y puede ser "hackeado"—, aquí la recompensa proviene de una **regla determinística** que comprueba objetivamente si la respuesta es correcta. En matemáticas, por ejemplo, se compara la respuesta final con la solución conocida. La ventaja es que la señal es exacta, barata y difícil de engañar; la desventaja es que solo aplica a dominios donde la corrección es verificable.

El problema que Xie et al. identifican es de **reproducibilidad científica**. DeepSeek-R1 liberó los pesos del modelo, pero **no el código de entrenamiento ni el dataset**. Esto deja abiertas preguntas críticas: (1) ¿pueden emerger capacidades de razonamiento similares en modelos más pequeños? (2) ¿cuál es la estructura óptima de datos para fomentarlas? (3) ¿qué metodología replica de forma confiable esos resultados?

Responder exige un **marco experimental controlado** que aísle las variables clave. Y aquí está el punto metodológico central: aunque las matemáticas son el banco de pruebas habitual del razonamiento, datasets como **GSM8K** u **Omni-MATH** son problemáticos como *datos de entrenamiento* precisamente porque su complejidad es **incontrolable y de varianza alta** —un problema puede requerir una cadena de inducción lógica de profundidad muy variable, sin que el experimentador pueda medirla ni dosificarla. Para estudiar la *dinámica* del razonamiento con rigor, hace falta un entorno donde la dificultad sea una perilla que se pueda girar.

## 3. Contribución central

Logic-RL aporta dos cosas indisociables: **un entorno de estudio y una receta de entrenamiento**.

**(1) Un dataset lógico controlado como laboratorio de razonamiento.** Los puzzles de caballeros y bribones son el sustrato ideal por tres propiedades:

1. **Generación procedural.** Se generan con plantillas lógicas, garantizando consistencia y variabilidad infinita. Son **datos no vistos** por el modelo original, lo que los hace idóneos para medir generalización genuina y no memorización del corpus de preentrenamiento.
2. **Dificultad controlable.** Se modula variando el **número de personajes (de 2 a 8)** y la **complejidad de las operaciones lógicas (de 1 a 4 combinaciones de operadores booleanos)**. Esto habilita diseñar un *currículum* y, además, usar puzzles más complejos como test fuera de distribución de modelos entrenados en casos más simples.
3. **Facilidad de verificación.** Cada puzzle tiene **una única respuesta ground-truth inequívoca**, garantizada por el algoritmo generador. La solución exige deducción estricta, lo que permite evaluar con exactitud y **minimiza el riesgo de reward hacking**.

**(2) Una receta de RL estable y efectiva.** Sobre ese entorno, los autores adoptan el algoritmo **REINFORCE++** y los diseños de recompensa de DeepSeek-R1, y hacen tres contribuciones técnicas que estabilizan el entrenamiento: un **prompt de sistema** que enfatiza el proceso de pensar-luego-responder, una **función de recompensa de formato estricta** que penaliza los atajos, y un **régimen de entrenamiento simple** que converge de forma estable.

Un ejemplo del entorno: *"You meet 2 inhabitants: Zoey, and Oliver. Zoey remarked, 'Oliver is not a knight'. Oliver stated, 'Oliver is a knight if and only if Zoey is a knave'."* La solución —Zoey es bribón, Oliver es caballero— se deduce por reglas formales y se verifica de forma determinística. Esta precisión lógica es lo que permite **distinguir razonamiento genuino de memorización superficial**, algo imposible en tareas de lenguaje natural ambiguas.

## 4. Método

### 4.1. Modelado de recompensa basado en reglas

La recompensa es la señal primaria del RL, y el equipo la refinó **iterativamente**, monitoreando de forma continua los comportamientos de hackeo del modelo, hasta llegar a un sistema **"casi imposible de hackear"** compuesto por solo dos términos: **recompensa de formato** y **recompensa de respuesta**.

**Prompt de sistema.** El modelo debe encerrar su razonamiento entre etiquetas `<think></think>` y su conclusión entre `<answer></answer>`. Los autores recomiendan añadir una etiqueta `<think>` directamente al final del prompt, lo que **reduce significativamente la dificultad** para que el modelo base siga las instrucciones.

**Recompensa de formato.** Mediante expresiones regulares se verifica la estructura de la respuesta:

$$S_{format} = \begin{cases} 1, & \text{si el formato es correcto} \\ -1, & \text{si el formato es incorrecto} \end{cases}$$

El aspecto más instructivo del paper es el **catálogo de reward hacking** que observaron bajo diseños de recompensa imperfectos, y cómo cada patológica motivó una regla de refinamiento:

- Saltarse el proceso `<think></think>` y responder directamente.
- Colocar el razonamiento dentro de la etiqueta `<answer></answer>`.
- Adivinar respuestas repetidamente sin razonar.
- Incluir texto irrelevante junto a la respuesta.
- Organizar la respuesta correcta de forma que impida su extracción.
- Volver a la fase de pensamiento **después** de haber emitido ya un `<answer>` (por razonamiento insuficiente).
- Repetir la pregunta original o usar frases como *"thinking process here"* para simular razonamiento sin razonar.

En respuesta, endurecieron las reglas: cada etiqueta debe aparecer **exactamente una vez y en el orden secuencial correcto**, el proceso de pensamiento debe contener razonamiento genuino, y la conclusión debe presentarse de forma extraíble y legible. Este proceso ilustra la lección central de RLVR: **el modelo optimiza exactamente lo que se le mide, no lo que se quiere lograr**; cerrar los atajos es tan importante como definir el objetivo.

**Recompensa de respuesta.** Una vez validado el formato, se compara la respuesta con el ground-truth:

$$S_{answer} = \begin{cases} 2, & \text{si la respuesta coincide totalmente} \\ -1{,}5, & \text{si coincide parcialmente} \\ -2, & \text{si no se puede parsear o falta} \end{cases}$$

La asimetría es deliberada: la coincidencia total vale +2, pero una respuesta ausente o no parseable (−2) es castigada más duramente que una parcialmente equivocada (−1,5), desincentivando la evasión y la respuesta ilegible.

### 4.2. Algoritmo de RL

El algoritmo base es una versión modificada de **REINFORCE++**, que en su configuración experimental **superó a GRPO** (el algoritmo de DeepSeek-R1). El retorno acumulado descontado de cada trayectoria es:

$$G_t = \sum_{k=t+1}^{T} \gamma^{k-t} r_k$$

con el factor de descuento $\gamma = 1$ en sus experimentos. Siguiendo recomendaciones de DeepSeek-Math, incorporan dos refinamientos:

**Primera modificación — usar KL como pérdida (KL Loss).** En PPO estándar, la divergencia KL entre la política de RL y la política SFT de referencia se incorpora como término de penalización *dentro de la recompensa por token*:

$$r(s_t, a_t) = \mathbb{I}(s_t = [\text{EOS}])\, r(x,y) - \beta\, \text{KL}(t)$$

donde $\mathbb{I}(s_t=[\text{EOS}])$ vale 1 solo al alcanzar el token de fin de secuencia y $\beta$ pondera la penalización. Siguiendo a GRPO, los autores mueven la KL **fuera de la recompensa y la incorporan directamente en la función de pérdida**, lo que simplifica el cómputo.

**Segunda modificación — estimación insesgada de la KL.** El estimador por defecto de PPO puede arrojar valores negativos. Adoptan el **estimador insesgado** de GRPO, que garantiza que la KL sea siempre **no negativa**, dando una medida de divergencia más estable y confiable durante el entrenamiento:

$$D_{KL}[\pi_\theta \| \pi_{ref}] = \frac{\pi_{ref}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})} - 1$$

### 4.3. Control de dificultad y calendario de entrenamiento

El modelo se entrena directamente por **3.600 pasos** con learning rate constante de $4\times10^{-7}$ y temperatura $0{,}7$. Durante el entrenamiento se lo expone a puzzles de **complejidad mixta, de 3 a 7 personas**. Parámetros clave (Tabla 1): algoritmo REINFORCE++, batch de entrenamiento 8, rollout N = 8, coeficiente KL 0,001, longitud máxima de respuesta 4.096 tokens.

Este régimen simple y de hiperparámetros fijos basta para que el modelo desarrolle patrones de razonamiento estables caracterizados por **exploración lógica, verificación intermedia y resumen sistemático** antes de producir la respuesta final. El **control de dificultad** —la perilla de número de personajes y operadores— es lo que convierte al entorno en un laboratorio: permite entrenar en 3–7 personas y luego evaluar generalización *fuera de distribución* con puzzles de **8 personas**, que el modelo nunca vio.

**Elección de modelo base.** Se probaron varios modelos de la serie Qwen2.5. Qwen2.5-Math-7B tendía a generar bloques de código Python que chocaban con el formato estricto. Sorprendentemente, **Qwen2.5-7B-Base y Qwen2.5-7B-Instruct exhibieron curvas de entrenamiento casi idénticas** (accuracy de validación, crecimiento de longitud, curvas de recompensa), aunque el instruct dio accuracy de test ligeramente superior. Por eso eligieron **Qwen2.5-7B-Instruct-1M** como base. Este hallazgo respalda una de sus tesis: *"el cold start es un bonus, no una necesidad"* — el arranque en frío ayuda un poco, pero no es imprescindible.

## 5. Resultados

### 5.1. Curvas de emergencia del razonamiento

Durante el entrenamiento, el modelo **aloca autónomamente más cómputo al razonamiento**. La longitud media de respuesta crece de forma casi lineal desde ~500 tokens iniciales hasta ~2.000 tokens tras 1.000 pasos —un aumento de **4×**— sin que ninguna instrucción se lo pida. Conforme la respuesta se alarga, aparecen comportamientos más complejos: **reflexión y exploración de soluciones alternativas**, que **emergen naturalmente sin datos relacionados en el conjunto de entrenamiento**. Estos fenómenos se alinean estrechamente con los de R1.

En el análisis cualitativo (apéndice), los autores documentan cuatro comportamientos emergentes:

1. **Vacilación y autoverificación.** El modelo usa frases como *"no estoy del todo seguro; volvamos a revisar este paso"* y verifica sistemáticamente todos los pasos previos antes de emitir la respuesta. Esta vacilación, **ausente en el preentrenamiento**, emerge porque se lo premia por aciertos y castiga por errores.
2. **Exploración multicamino y backtracking.** Propone múltiples soluciones (*"probemos ambas posibilidades"*) y retrocede para verificar consistencia, imitando la resolución humana.
3. **Aplicación de fórmulas.** Tras el RL, el modelo aplica *instintivamente* la fórmula de implicación *"Si P, entonces Q"* —falsa solo cuando P es verdadero y Q falso—, incorporando **lógica formal**, no solo ensayo y error, pese a que ningún dato así estaba en el entrenamiento.
4. **Cambio ocasional de idioma.** Algunos segmentos `<think>` contienen tokens en chino (el modelo base es angloparlante), aunque el `<answer>` final permanece en inglés para obtener la recompensa de formato.

En K&K (Tabla 2), Logic-RL lleva el promedio de **0,19 a 0,89** (+0,70), superando a GPT-4o (0,37) y quedando a la par de modelos de razonamiento dedicados. Notablemente, generaliza a puzzles de **8 personas** (0,67) pese a entrenarse solo hasta 7.

### 5.2. ¿Hay un "aha moment"?

Este es el punto más matizado. El **"aha moment"** del reporte de R1 tiene dos interpretaciones: (a) la adquisición *súbita* de comportamientos de razonamiento complejos, y (b) que el modelo verbalice espontáneamente *"aha moment"* (p. ej. *"Wait, wait. That's an aha moment I can flag here"*).

Logic-RL **no observó ninguna de las dos formas súbitas**. Su modelo no verbalizó "aha moment", y —más importante— rastreando la frecuencia de palabras reflexivas (*check*, *verify*, *wait*, *yet*, *re-evaluate*) en los primeros 1.800 pasos, encontraron que **todas crecen de forma gradual y estable, sin saltos abruptos**. De hecho, el modelo ya exhibía comportamientos complejos de razonamiento (autorreflexión, exploración, verificación, resumen) **ya en el paso 10**. La conclusión, alineada con Liu et al. (*"There may not be an aha moment in R1-zero-like training"*), es que el razonamiento **emerge orgánica y gradualmente**, no en un instante mágico. Esto matiza el relato popular del "aha moment" sin negar la emergencia: los comportamientos aparecen y se intensifican de forma continua a lo largo del entrenamiento.

### 5.3. Transferencia a matemáticas (Super OOD)

El resultado estelar: pese a entrenar **solo en puzzles lógicos**, el modelo mejora **+125% en AIME (2021–2024)** y **+38% en AMC (2022–2023)**, benchmarks de competición matemática que los autores denominan **"Super OOD"** por lo lejanos que están del dominio de entrenamiento. La mejora *sincrónica* en ambos indica que el RL no solo mejora el rendimiento en la distribución de entrenamiento, sino que **facilita la emergencia de estrategias de razonamiento robustas y transferibles**. En términos de la introducción: las heurísticas de razonamiento aprendidas desarrollan **esquemas abstractos de resolución de problemas** en lugar de basarse en coincidencia de patrones específicos del dominio.

### 5.4. Hallazgos adicionales

- **SFT memoriza, RL generaliza.** Usando la métrica **LiMem** $= \text{Acc}(f;D)\cdot(1-\text{CR}(f;D))$ —que combina accuracy sobre problemas vistos y consistencia bajo perturbaciones—, muestran que el fine-tuning por rechazo (RFT) mejora un poco el test pero dispara la memorización, mientras que **el RL logra mayor accuracy de test con incremento mínimo o incluso negativo de memorización**. El RL fomenta exploración independiente y generalización genuina.
- **Respuestas más largas no garantizan mejor razonamiento.** Comparando un "modelo positivo" (cuya longitud *baja* mientras accuracy y recompensa suben) con uno "negativo" (longitud sube sin ganancia), concluyen que el aumento de longitud es un **correlato, no una causa** del mejor razonamiento. *"El razonamiento más eficiente proviene del camino más corto."*
- **La mezcla de idiomas perjudica.** Respuestas con tokens de otros idiomas obtienen puntajes menores; hace falta una penalización por consistencia lingüística.
- **Ciertos tokens de "pensamiento" ayudan.** *"verify"* y *"re-evaluate"* elevan el puntaje; *"recheck"* lo baja (señala inseguridad). Curiosamente, *"re-evaluate"* (frecuente en el corpus) supera a *"reevaluate"* (casi ausente).
- **El currículum importa poco.** Bajo una razón de curación fija, el aprendizaje por currículum supera *marginalmente* al mezclado, pero la ventaja es de significancia práctica limitada.
- **REINFORCE++ > GRPO.** En sus experimentos, PPO logró la mejor accuracy y recompensa pero fue 138% más lento; REINFORCE++ ofreció la mejor estabilidad y eficiencia, y GRPO fue el más débil de los tres.

## 6. Limitaciones

Los propios autores son cautos:

- **Escala pequeña y dominio estrecho.** Los hallazgos se basan en un dataset lógico pequeño (<5.000 muestras). La generalización a escenarios reales de matemáticas o código a gran escala **queda por explorar**.
- **Longitud explosiva.** Las respuestas se expanden hasta 4× tras el RL; hacen falta métodos *long-to-short* para comprimir la cadena de pensamiento y mejorar la eficiencia de tokens.
- **Restricciones de formato posiblemente subóptimas.** Las etiquetas `<think></think>` organizan bien la cadena, pero queda abierto si un enfoque **sin restricciones o latente** rendiría mejor; el modelo podría "inventar" su propia representación interna de razonamiento con los incentivos adecuados.
- **Mezcla de idiomas sin explicar.** El uso de tokens chinos en `<think>` pese a entrenar solo en inglés es un fenómeno no comprendido; una hipótesis es que ciertos tokens del vocabulario chino produzcan estados ocultos "favorables" bajo el esquema de RL.
- **Estabilización aún artesanal.** Eliminar restricciones de KL ayuda con modelos base fuertes, y una temperatura alta inicial da diversidad; el efecto de la etapa SFT sobre la eficiencia del RL sigue por investigar.

## 7. Conexión con la Clase 34 (aha-moments) y con DeepSeek-R1

En la **Clase 34 (Razonamiento)**, Logic-RL aparece (slide 35) como ejemplo de **RL basado en reglas que induce razonamiento**, dentro de la discusión de los "aha-moments". Su valor pedagógico es doble.

Primero, es el **complemento controlado de DeepSeek-R1**. R1 demostró el fenómeno a gran escala pero como caja negra (sin código ni datos); Logic-RL lo reproduce en un **entorno de laboratorio verificable** donde cada variable —dificultad, formato, recompensa— es observable y manipulable. Confirma las tres piezas del paradigma R1: (1) recompensa de formato + corrección basada en reglas, (2) emergencia de reflexión/verificación/exploración sin datos que las contengan, y (3) crecimiento autónomo del cómputo de razonamiento (la longitud). Es, en esencia, la validación científica reproducible de la receta de R1 en pequeño.

Segundo, **matiza el mito del "aha moment"**. Donde el relato popular sugiere un instante mágico de iluminación, Logic-RL muestra —rastreando frecuencias de tokens paso a paso— que la emergencia es **gradual y continua**, no un salto discreto. Esto conecta con el debate abierto en la comunidad (Liu et al., "There may not be an aha moment"; Ye et al., sobre el proceso oculto de razonamiento). Para la clase, la lección es cuidadosa: el razonamiento *sí* emerge del RL con recompensa verificable, pero como una **acumulación progresiva de comportamientos**, no como un evento puntual.

Respecto a la **línea Yue** y el debate sobre los límites del RLVR: Logic-RL aporta evidencia del lado optimista de la transferencia (el RL en lógica mejora matemáticas), pero también del lado escéptico (la escala es pequeña, la longitud explota, el currículum apenas importa). La tensión entre *"el RL amplifica capacidades latentes del modelo base"* vs. *"el RL enseña razonamiento nuevo"* queda planteada: el hallazgo de que base e instruct dan curvas casi idénticas, y que el RL generaliza donde el SFT memoriza, empuja hacia la idea de que el RL **reorganiza y hace accesibles** esquemas de razonamiento más que inyectarlos desde cero.

**Enlaces internos:**

- Clase: [/clases/clase-34](/clases/clase-34) — Razonamiento (prof. Sebastián Amenábar).
- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado) — MDP, retorno, policy gradient.
- Paper de referencia RL: [/papers/ppo-schulman-2017](/papers/ppo-schulman-2017) — PPO, base de los algoritmos policy-gradient usados en RLHF/RLVR.
- Paper hermano: DeepSeek-R1 — la receta de RL basado en reglas que Logic-RL reproduce en entorno controlado.

## 8. Nota final: relevancia para salud

Logic-RL entrega a la medicina computacional una hipótesis de trabajo verificable: **entrenar el razonamiento en dominios sintéticos con recompensa determinística y medir si transfiere a tareas clínicas reales**. La receta —entorno de dificultad controlable, recompensa basada en reglas casi imposible de hackear, verificación exacta de la respuesta— es directamente trasladable a subdominios sanitarios donde existe una verdad de referencia inequívoca: ajuste de dosis según fórmulas farmacocinéticas, resolución de reglas de elegibilidad para ensayos clínicos, deducción sobre grafos de interacciones fármaco-fármaco, o razonamiento lógico sobre criterios diagnósticos estructurados. La promesa —y a la vez la pregunta abierta que el propio paper deja sin resolver, pues su generalización se validó solo hacia matemáticas de competición— es si un modelo que aprende a razonar deductivamente sobre puzzles verificables desarrolla esquemas abstractos que mejoren tareas médicas reales, ambiguas y de alto riesgo; y, sobre todo, si esa transferencia puede **medirse con seguridad** antes de acercarla a cualquier decisión que afecte a un paciente, dado que la métrica LiMem del paper recuerda que la diferencia entre memorizar y razonar es precisamente lo que separa una herramienta confiable de una peligrosa en un entorno clínico.
