# Tree of Thoughts: Deliberate Problem Solving with Large Language Models — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*.
- **Autores:** Shunyu Yao (Princeton), Dian Yu (Google DeepMind), Thomas L. Griffiths (Princeton), Jeffrey Zhao (Google DeepMind), Yuan Cao (Google DeepMind), Izhak Shafran (Google DeepMind), Karthik Narasimhan (Princeton).
- **Venue:** *37th Conference on Neural Information Processing Systems* (NeurIPS 2023).
- **Preprint:** arXiv:2305.10601v2 (3 dic 2023). Los experimentos se realizaron entre el 5 y el 16 de mayo de 2023.
- **Código y prompts:** [github.com/princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm).
- **Linaje:** proviene del mismo grupo de Princeton que ReAct (Yao et al., 2022). Se apoya explícitamente en Chain-of-Thought (Wei et al., 2022) y Self-Consistency (Wang et al., 2022), y bebe de una tradición mucho más antigua: la caracterización del problem solving como búsqueda en un espacio combinatorio propuesta por Newell, Shaw y Simon en los años 1950-1970.

El paper introduce **Tree of Thoughts (ToT)**, un marco de inferencia para modelos de lenguaje que **generaliza a Chain-of-Thought (CoT)**. Mientras CoT hace que el modelo produzca una cadena lineal de pasos intermedios de izquierda a derecha, ToT permite que el modelo **explore un árbol de "pensamientos"** —unidades coherentes de texto que sirven de pasos intermedios— generando varios candidatos por paso, **autoevaluando** cada estado parcial y usando **algoritmos de búsqueda clásicos (BFS/DFS con backtracking)** para decidir qué ramas expandir, cuáles podar y cuándo retroceder. La tesis central es que el mecanismo autorregresivo token a token de los LLM se parece al "Sistema 1" del pensamiento humano —rápido, asociativo, automático— y que puede beneficiarse de un "Sistema 2" deliberado que mantenga alternativas y planifique.

El salto empírico es grande. En **Game of 24**, GPT-4 con CoT resuelve apenas el **4 %** de los problemas, mientras que ToT alcanza el **74 %**. Los autores validan el marco en tres tareas nuevas diseñadas para desafiar a GPT-4: Game of 24, Creative Writing y Mini Crosswords 5×5, todas las cuales requieren planificación, búsqueda o lookahead no triviales.

Para la **Clase 34 (Razonamiento)** el paper es la pieza que conecta el razonamiento de LLMs con la búsqueda clásica de la IA. Es citado (slide 32) bajo el rótulo "Cómo potenciar LLMs: Tree-of-Thought y Agent-Debate", y prefigura conceptualmente la idea de *test-time compute* que más tarde harían explícita los modelos de razonamiento tipo o1/R1: gastar más cómputo en inferencia, deliberando, para resolver mejor.

## 2. Contexto: los límites del razonamiento lineal e izquierda-a-derecha

El punto de partida del paper es una observación incómoda. Los LLM escalados (GPT, PaLM) son cada vez más capaces de razonamiento matemático, simbólico y de sentido común, pero por debajo de todo ese progreso sigue operando el **mismo mecanismo autorregresivo original**: decisiones a nivel de token, una a una, de izquierda a derecha. La pregunta que se hacen los autores es si ese mecanismo tan simple basta para construir un solucionador general de problemas.

La respuesta se apoya en la psicología cognitiva. La teoría de los "procesos duales" (Sloman, Stanovich, Kahneman) distingue dos modos de decisión: un **Sistema 1** rápido, automático e inconsciente, y un **Sistema 2** lento, deliberado y consciente. Las elecciones asociativas token a token de un LLM recuerdan al Sistema 1. Lo que les falta, argumentan, es un Sistema 2 de planificación que haga dos cosas: (1) mantener y explorar diversas alternativas para las decisiones actuales en lugar de comprometerse con una sola, y (2) evaluar el estado presente y mirar hacia adelante o retroceder para tomar decisiones globales.

Este marco permite diagnosticar con precisión las carencias de los métodos existentes de prompting. El paper formaliza cuatro:

- **Input-output (IO) prompting.** La forma más básica: se mapea la entrada $x$ a la salida $y$ envolviendo $x$ con instrucciones o ejemplos few-shot, $y \sim p_\theta^{IO}(y|x)$. No hay pasos intermedios.
- **Chain-of-Thought (CoT).** Introduce una cadena de pensamientos $z_1, \dots, z_n$ para tender un puente entre $x$ y $y$, donde cada $z_i$ es una secuencia coherente de lenguaje (por ejemplo una ecuación intermedia). Cada pensamiento se muestrea secuencialmente, $z_i \sim p_\theta^{CoT}(z_i | x, z_{1 \cdots i-1})$, y en la práctica $[z_{1\cdots n}, y]$ se genera como un único flujo continuo de texto.
- **Self-Consistency con CoT (CoT-SC).** Un método de ensamble que muestrea $k$ cadenas de pensamiento i.i.d. y devuelve la salida más frecuente, $\arg\max_y \#\{i \mid y^{(i)} = y\}$. Mejora sobre CoT porque explora un conjunto más rico de procesos de razonamiento, pero tiene dos límites: **dentro de cada cadena no hay exploración local** de pasos alternativos, y la heurística de "voto mayoritario" solo aplica cuando el espacio de salidas es reducido (por ejemplo, respuesta múltiple).

El diagnóstico se resume en dos fallas, una local y una global. **Localmente**, estos métodos no exploran continuaciones distintas dentro de un proceso de pensamiento —no ramifican el árbol—. **Globalmente**, no incorporan planificación, lookahead ni backtracking para evaluar esas opciones —les falta la búsqueda guiada por heurísticas que caracteriza a la resolución humana de problemas—. El análisis de errores lo cuantifica de forma contundente: en Game of 24, cerca del **60 % de las muestras de CoT ya han fracasado tras generar el primer paso** (equivalentemente, las primeras tres palabras, por ejemplo "4 + 9"). Una decisión inicial mala condena toda la cadena, y el decoding lineal no tiene forma de deshacerla.

## 3. Contribución central: el marco Tree of Thoughts

La propuesta es enmarcar cualquier problema como una **búsqueda en un árbol**, donde cada nodo es un estado $s = [x, z_{1\cdots i}]$: la entrada más la secuencia de pensamientos parciales acumulados hasta ese punto. Las ramas corresponden a operadores que extienden un estado con un nuevo pensamiento. La navegación del árbol la gobiernan heurísticas, y la innovación fundamental es **de dónde salen esas heurísticas**: en vez de estar programadas (como en Deep Blue) o aprendidas con entrenamiento dedicado (como en AlphaGo), aquí las provee **el propio LLM razonando en lenguaje natural sobre los estados**. Esta implementación de heurísticas de búsqueda vía autoevaluación del modelo es lo genuinamente nuevo.

Una instancia concreta de ToT responde a cuatro preguntas de diseño: (1) cómo descomponer el proceso en pasos de pensamiento; (2) cómo generar pensamientos candidatos desde cada estado; (3) cómo evaluar heurísticamente los estados; (4) qué algoritmo de búsqueda usar. Los cuatro componentes son **modulares** e independientes.

El paper subraya cuatro virtudes conceptuales del marco:

1. **Generalidad.** IO, CoT, CoT-SC y self-refinement son casos particulares de ToT (árboles de profundidad o amplitud limitadas). ToT los subsume.
2. **Modularidad.** El LLM base, la descomposición, la generación, la evaluación y la búsqueda se pueden variar por separado.
3. **Adaptabilidad.** Distintas propiedades del problema, capacidades del modelo y restricciones de recursos se acomodan reconfigurando los componentes.
4. **Conveniencia.** No requiere entrenamiento adicional: basta un LLM preentrenado.

## 4. Método: los cuatro componentes en detalle

### 4.1. Descomposición del pensamiento

Mientras CoT muestrea pensamientos de forma continua sin descomposición explícita, ToT aprovecha las propiedades del problema para **diseñar el tamaño del paso intermedio**. Un pensamiento puede ser un par de palabras (Crosswords), una línea de ecuación (Game of 24) o un párrafo entero de plan de escritura (Creative Writing). La regla de oro es un compromiso: un pensamiento debe ser **suficientemente "pequeño"** para que el modelo genere candidatos diversos y prometedores (generar un libro entero es demasiado grande para ser coherente), pero **suficientemente "grande"** para que el modelo pueda evaluar su potencial (generar un solo token es demasiado pequeño para evaluar).

### 4.2. Generador de pensamientos $G(p_\theta, s, k)$

Dado un estado $s = [x, z_{1\cdots i}]$, se generan $k$ candidatos para el siguiente paso con una de dos estrategias:

- **(a) Muestreo i.i.d. desde un prompt CoT:** $z^{(j)} \sim p_\theta^{CoT}(z_{i+1} | s)$. Funciona mejor cuando el espacio de pensamientos es rico (por ejemplo, cada pensamiento es un párrafo), donde muestrear de forma independiente produce diversidad. Se usa en Creative Writing.
- **(b) Propuesta secuencial con un "propose prompt":** $[z^{(1)}, \dots, z^{(k)}] \sim p_\theta^{propose}(z_{i+1}^{(1\cdots k)} | s)$. Funciona mejor cuando el espacio es restringido (una palabra, una línea), porque proponer los candidatos en el mismo contexto **evita duplicados**. Se usa en Game of 24 y Crosswords.

### 4.3. Evaluador de estados $V(p_\theta, S)$

Dado un frente (frontier) de estados distintos, el evaluador estima el progreso de cada uno hacia la solución, funcionando como la **heurística** que le dice a la búsqueda qué estados conservar y en qué orden explorarlos. Hay dos estrategias:

- **(a) Valorar cada estado independientemente (value):** $V(p_\theta, S)(s) \sim p_\theta^{value}(v | s)$, donde un "value prompt" razona sobre $s$ y produce un escalar (por ejemplo 1-10) o una clasificación (`sure`/`likely`/`impossible`) traducible a valor. La base de esa valoración combina **simulaciones de lookahead** (por ejemplo, confirmar rápido que 5, 5, 14 alcanzan 24 vía $5+5+14$) con **sentido común** (por ejemplo, que 1, 2, 3 son demasiado pequeños para llegar a 24, o que ninguna palabra empieza con "tzxc"). Lo primero promueve estados buenos; lo segundo elimina estados malos. Las valoraciones no necesitan ser perfectas, solo aproximadamente útiles para decidir.
- **(b) Votar entre estados (vote):** $V(p_\theta, S)(s) = \mathbb{1}[s = s^*]$, donde un buen estado $s^* \sim p_\theta^{vote}(s^* | S)$ se elige comparando deliberadamente los estados de $S$ en un "vote prompt". Es natural cuando el éxito es difícil de valorar de forma absoluta (por ejemplo, la coherencia de un pasaje): en vez de puntuar cada uno, se comparan y se vota el más prometedor. Los autores lo describen como una self-consistency "paso a paso".

En ambas estrategias se puede promptear el modelo varias veces y agregar los resultados para intercambiar tiempo/cómputo por heurísticas más robustas (en Game of 24 se muestrea el valor 3 veces por candidato).

### 4.4. Algoritmo de búsqueda

Sobre esos componentes se puede enchufar cualquier algoritmo de búsqueda. El paper explora dos simples y deja A\* y MCTS para trabajo futuro:

- **Breadth-First Search (BFS), Algoritmo 1.** Mantiene los $b$ estados más prometedores por paso. En cada nivel $t$ se generan todos los sucesores $S'_t = \{[s, z] \mid s \in S_{t-1}, z \in G(p_\theta, s, k)\}$, se evalúan con $V_t$ y se conservan los $b$ mejores: $S_t = \arg\max_{S \subset S'_t, |S| = b} \sum_{s \in S} V_t(s)$. Se usa cuando el árbol es poco profundo ($T \le 3$) y las etapas iniciales se pueden podar a un conjunto pequeño ($b \le 5$). Aplica a Game of 24 y Creative Writing.
- **Depth-First Search (DFS), Algoritmo 2.** Explora primero el estado más prometedor hasta alcanzar la salida final ($t > T$) o hasta que el evaluador considere que desde $s$ es imposible resolver el problema ($V(p_\theta, \{s\})(s) \le v_{th}$, un umbral de valor). En ese caso **poda el subárbol** de $s (cambiando exploración por explotación) y **retrocede (backtrack) al estado padre** para continuar. Aplica a Crosswords.

## 5. Experimentos y resultados

Salvo indicación contraria, se usa GPT-4 en modo Chat Completion con temperatura 0.7.

### 5.1. Game of 24

Un desafío aritmético: usar 4 números y las operaciones básicas ($+ - \times /$) para obtener 24 (por ejemplo, de "4 9 10 13" sale "(10 − 4) × (13 − 9) = 24"). Se toman 100 juegos difíciles (índices 901-1000) del sitio 4nums.com, que ordena 1362 juegos por tiempo humano de resolución. El éxito exige una ecuación válida que dé 24 usando cada número una sola vez. La descomposición natural es en 3 pasos (una ecuación intermedia cada uno); se usa BFS con $b = 5$, un "propose prompt", y cada candidato se valora como `sure`/`maybe`/`impossible` muestreando 3 veces.

| Método | Éxito |
|---|---|
| IO prompt | 7.3 % |
| CoT prompt | 4.0 % |
| CoT-SC (k=100) | 9.0 % |
| ToT (b=1) | 45 % |
| ToT (b=5) | **74 %** |
| IO + Refine (k=10) | 27 % |
| IO (best of 100) | 33 % |
| CoT (best of 100) | 49 % |

El contraste es dramático: IO, CoT y CoT-SC quedan por debajo del 10 %, mientras que ToT con $b=1$ ya llega a 45 % y con $b=5$ a 74 %. Un dato clave para el análisis costo-beneficio: tratando IO/CoT "best of $k$" como si visitaran $k$ nodos de un bandit, incluso el mejor de 100 muestras de CoD (49 %) queda muy por debajo de ToT explorando más nodos. El escalamiento por muestreo ciego rinde mucho menos que la búsqueda deliberada.

### 5.2. Creative Writing

Se inventa una tarea abierta: dada una entrada de 4 oraciones aleatorias, producir un pasaje coherente de 4 párrafos que terminen respectivamente en esas 4 oraciones. Se construye un ToT de **profundidad 2** (un solo paso intermedio): el modelo genera $k=5$ planes y vota el mejor, luego genera $k=5$ pasajes basados en ese plan y vota el mejor ($b=1$). La coherencia se mide de dos formas: un puntaje escalar 1-10 dado por GPT-4 zero-shot (5 muestras promediadas, con desviación estándar de ~0.56) y comparaciones ciegas entre pares hechas por humanos.

- **Puntaje GPT-4:** ToT **7.56** vs. CoT 6.93 vs. IO 6.19.
- **Preferencia humana** sobre 100 pares: los humanos prefieren ToT sobre CoT en **41** casos, CoT sobre ToT en **21**, y encuentran ~38 pares similarmente coherentes.

Aquí el refinamiento iterativo resulta más eficaz que en Game of 24 —al ser una tarea de lenguaje natural— y mejora la coherencia de IO de 6.19 a 7.67 y la de ToT de 7.56 a 7.91. Los autores sugieren que el refinamiento puede verse como una **tercera vía de generación de pensamientos** dentro de ToT: nuevos pensamientos que nacen de refinar los viejos, en lugar de muestrearse i.i.d. o secuencialmente.

### 5.3. Mini Crosswords 5×5

El problema de búsqueda más profundo del paper: un crucigrama 5×5 con 10 pistas (5 horizontales, 5 verticales) cuya salida es un tablero de 25 letras. Se usan 20 juegos de prueba tomados de GooBix (156 juegos en total). Se evalúa a tres niveles: proporción de letras correctas (25 por juego), de palabras (10 por juego) y de juegos completos. Se usa **DFS**: se explora la palabra-pista más prometedora hasta que el estado deja de serlo, y entonces se retrocede al padre. Para hacer la búsqueda tratable, los pensamientos posteriores no pueden cambiar letras ya rellenadas, de modo que hay a lo sumo 10 pasos intermedios (con un límite de 100 pasos de búsqueda). El evaluador traduce cada estado en restricciones de letras y determina para cada pista si es posible rellenarla; si alguna pista restante se considera imposible, se poda el subárbol.

| Método | Letra | Palabra | Juego |
|---|---|---|---|
| IO | 38.7 | 14 | 0 |
| CoT | 40.6 | 15.6 | 1 |
| **ToT** | **78** | **60** | **20 %** (4/20) |
| ToT + best state (oráculo) | 82.4 | 67.5 | 35 % (7/20) |
| ToT − prune (ablación) | 65.4 | 41.5 | 5 |
| ToT − backtrack (ablación) | 54.6 | 20 | 5 |

ToT lleva el éxito a nivel de palabra del <16 % de IO/CoT al **60 %**, resolviendo 4 de 20 juegos. Las ablaciones son reveladoras. Emitir desde el **mejor estado del oráculo** (en vez del que la heurística determina) sube a 7/20 juegos, señal de que la heurística de salida es mejorable. **Quitar la poda** ("−prune") empeora el desempeño general, aunque a veces encuentra soluciones que ToT con poda no alcanza —el evaluador imperfecto poda estados que en realidad eran correctos, a menudo por palabras raras u obsoletas que GPT-4 no reconoce—. **Quitar el backtracking** ("−backtrack", equivalente a un BFS codicioso con $b=1$ que permite sobrescribir) desploma el éxito de palabra a 20 %. La lección es que **el backtracking y una buena heurística de poda son componentes críticos**, no adornos.

### 5.4. Experimentos adicionales (apéndice)

- **Tareas más fáciles (GSM8K, StrategyQA), zero-shot ToT-BFS:** ToT mejora sobre CoT solo marginalmente (GSM8K: IO 51 → CoT 86 → ToT 90; StrategyQA: 73 → 82 → 83), porque GPT-4 + CoT ya es muy bueno en ellas y el cuello de botella de StrategyQA es conocimiento externo, no razonamiento. ToT rinde donde CoT ya lucha.
- **Modelo más débil (GPT-3.5):** el orden "ToT > CoT > IO" se mantiene. En Creative Writing, GPT-3.5 + ToT supera a GPT-4 + IO. En Game of 24, GPT-3.5 + ToT alcanza solo 19 % frente al 74 % de GPT-4 + ToT. Al cruzar modelos (GPT-4 genera + GPT-3.5 evalúa = 64 %; GPT-3.5 genera + GPT-4 evalúa = 31 %), se identifica que **el cuello de botella de esta tarea es la generación de pensamientos, no la evaluación**.

## 6. Limitaciones

El propio paper es franco con sus límites:

- **Costo de inferencia.** ToT requiere significativamente más cómputo que IO o CoT. En Game of 24, resolver un problema con ToT consume ~5.5k tokens de completion (~$0.74 por caso), comparable a 100 intentos de CoT (~6.7k tokens, $0.47) —aunque ToT rinde mejor que el mejor de esos 100 intentos—. En Creative Writing, ToT usa unas **5× más tokens y dinero** ($0.32 vs. $0.06-0.07). En general, ToT puede requerir de 5 a 100 veces más tokens generados que CoT. La flexibilidad modular permite ajustar el compromiso costo-desempeño (tamaño de haz, número de votos, few-shot vs. zero-shot, GPT-3.5 vs. GPT-4).
- **Dependencia del evaluador.** La heurística es tan buena como la capacidad del modelo de autoevaluarse. En Crosswords, el evaluador podó estados correctos por no reconocer palabras raras; la ablación de la poda mostró que soluciones válidas se perdían por juicios imperfectos del propio LLM.
- **Necesidad selectiva.** La búsqueda deliberada no hace falta en muchas tareas que GPT-4 ya domina; el trabajo solo explora tres tareas relativamente simples diseñadas para desafiarlo, y su valor se materializa en problemas de planificación/búsqueda genuinos.
- **Solo inferencia, sin fine-tuning.** El trabajo usa un LLM off-the-shelf; los autores conjeturan que **entrenar el modelo con decisiones contrafactuales de alto nivel** (deliberar sobre el próximo párrafo en vez de predecir el próximo token) podría potenciar aún más estas capacidades —una intuición que anticipa la línea de los modelos de razonamiento entrenados con RL.

## 7. Conexión con la Clase 34 y con el test-time compute

La conclusión del paper enuncia con claridad la tesis que la Clase 34 recoge: el "Sistema 1" asociativo de los LLM **puede aumentarse provechosamente con un "Sistema 2"** basado en buscar en un árbol de caminos posibles hacia la solución. ToT es, literalmente, un puente entre dos tradiciones: aporta a los LLM las intuiciones clásicas del problem solving de Newell y Simon (búsqueda heurística en un espacio combinatorio), y a la vez los LLM aportan a esos métodos clásicos algo que estos no tenían —una forma de resolver problemas difíciles de formalizar, como la escritura creativa, donde no existe una función de evaluación programable.

Esta lectura ilumina el resto de la clase. **Agent-Debate** (el otro método citado en el slide 32) comparte el ADN de ToT: en vez de comprometerse con una sola línea de razonamiento, se generan y comparan alternativas, y la deliberación explícita mejora la robustez. Y en el eje temporal, ToT es un precursor conceptual del **test-time compute** que hoy definen modelos como o1 y R1: la idea de que **gastar más cómputo en la inferencia** —explorando, evaluando y descartando caminos— vale más que forzar una respuesta en una sola pasada. La diferencia de implementación es importante: ToT orquesta la búsqueda **externamente**, con un algoritmo (BFS/DFS) que llama al LLM como generador y como evaluador, mientras que la generación siguiente de modelos de razonamiento **internaliza** esa deliberación en una sola cadena larga aprendida vía RL. Pero la observación de que "más cómputo produce mayor inteligencia" —que el paper enuncia casi al pasar en su apéndice de costos— es exactamente la apuesta que esa línea de modelos convertiría en paradigma. Un beneficio adicional que la clase valora: al operar sobre lenguaje legible en vez de valores implícitos de token, ToT **mejora la interpretabilidad** de las decisiones del modelo y la posibilidad de alineamiento humano.

En contraste con CoT/Self-Consistency, la posición de ToT es precisa. CoT rompió el mapeo directo entrada-salida introduciendo pasos intermedios, pero mantuvo la linealidad. CoT-SC agregó diversidad **entre** cadenas completas, pero sin exploración **dentro** de cada cadena y con un voto que solo sirve para salidas discretas. ToT unifica ambas ideas y las generaliza: exploración local (ramificar en cada paso) más planificación global (evaluar, mirar adelante, retroceder), con IO/CoT/CoT-SC recuperados como casos degenerados de árboles de amplitud o profundidad 1.

---

**Nota sobre relevancia para salud.** El valor de ToT para la medicina no está en la exactitud marginal sino en el *modo* de razonar. Una decisión clínica de alto riesgo —un diagnóstico diferencial, la elección de un esquema terapéutico, la lectura de un caso con hallazgos contradictorios— es justamente el tipo de problema donde comprometerse de forma lineal con la primera hipótesis (el equivalente del "4 + 9" que condena la cadena en Game of 24) resulta peligroso. Un asistente clínico estructurado como ToT podría **enumerar hipótesis alternativas explícitamente**, evaluar cada una contra la evidencia disponible con un paso de valoración auditable, podar las inviables y **retroceder** cuando un nuevo dato invalida una rama, dejando una traza de razonamiento en lenguaje legible por el médico —lo que la sección de impacto del paper destaca como ganancia de interpretabilidad y oportunidad de alineamiento humano. El precio es real: la deliberación con backtracking cuesta 5 a 100 veces más cómputo que una respuesta directa y depende de la calidad de la autoevaluación, un evaluador imperfecto puede podar la hipótesis correcta (como ocurrió con las palabras raras en Crosswords). En un contexto de alto riesgo ese compromiso suele ser aceptable e incluso deseable: se paga más cómputo y más latencia a cambio de una deliberación explícita, revisable y capaz de corregirse, que es exactamente lo que exige una decisión donde el error tiene consecuencias graves.
