# Q-learning (Watkins & Dayan, 1992) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Q-learning* (Technical Note).
- **Autores:** Christopher J. C. H. Watkins (entonces en Highbury, Londres; el algoritmo proviene de su tesis doctoral de 1989 en Cambridge) y Peter Dayan (Centre for Cognitive Science, University of Edinburgh; al momento de publicación, dirección en el CNL del Salk Institute, San Diego).
- **Venue:** *Machine Learning*, vol. 8, pp. 279–292, 1992. Kluwer Academic Publishers.
- **Keywords (del paper):** Q-learning, reinforcement learning, temporal differences, asynchronous dynamic programming.
- **Antecedente clave:** Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD Thesis, University of Cambridge. Es ahí donde Q-learning se introduce por primera vez; este paper de 1992 **prueba en detalle** el teorema de convergencia que aquella tesis solo había esbozado.

Este no es el paper que *inventa* Q-learning —eso fue la tesis de Watkins de 1989— sino el paper que **lo demuestra correcto**. Su contribución central es un único resultado matemático: la prueba de que Q-learning converge a los valores-acción óptimos $Q^*(x,a)$ con probabilidad 1, bajo condiciones razonables sobre las tasas de aprendizaje y sobre el ambiente markoviano. En palabras de los propios autores en el cierre: "Tal garantía había eludido previamente a la mayoría de los métodos de aprendizaje por refuerzo". Ese es el peso histórico del documento: convierte una heurística atractiva en un algoritmo con fundamento teórico.

El abstract lo resume con precisión: Q-learning es "una manera simple para que los agentes aprendan a actuar óptimamente en dominios markovianos controlados"; equivale a "un método incremental de programación dinámica que impone demandas computacionales limitadas"; funciona "mejorando sucesivamente sus evaluaciones de la calidad de acciones particulares en estados particulares". El paper prueba que converge "siempre que todas las acciones sean muestreadas repetidamente en todos los estados y los valores-acción se representen de forma discreta", y esboza extensiones a (a) el caso no descontado pero con estados absorbentes y (b) el caso en que muchos valores $Q$ se actualizan por iteración en vez de uno solo.

Para la Clase 31 (Aprendizaje Reforzado) este paper es la piedra angular teórica: la sección 3 de la clase desarrolla Q-Learning con la ecuación de Bellman, la tabla $Q$, la política $\varepsilon$-greedy y el factor de descuento $\gamma$, y luego presenta Deep Q-Learning como la extensión que escala el algoritmo tabular de 1992 a espacios de estado masivos mediante redes neuronales. Entender este paper es entender *por qué* la regla de actualización de Q-Learning funciona —no solo *cómo* se escribe—.

## 2. Contexto histórico: RL, MDP, programación dinámica y TD

El paper se inscribe en una tradición que, hacia 1992, ya tenía varias décadas pero estaba apenas empezando a unificarse. Para situarlo hay que reconstruir cuatro hilos que confluyen en Q-learning.

**Procesos de decisión de Markov (MDP).** El escenario es un "agente computacional moviéndose por un mundo discreto y finito, eligiendo una acción de una colección finita en cada paso de tiempo". Ese mundo es un *proceso de Markov controlado*: en el paso $n$ el agente registra el estado $x_n \in X$, elige la acción $a_n$, recibe una recompensa probabilística $r_n$ cuyo valor medio $\mathcal{R}_x(a)$ depende solo del estado y la acción, y el mundo transita probabilísticamente al estado $y_n$ según una ley de transición $P_{xy}[a]$. La propiedad de Markov —que el futuro depende solo del estado y acción actuales, no de la historia— es lo que hace tratable todo el problema.

**Programación dinámica de Bellman.** La tarea del agente es hallar una *política óptima* que maximice la recompensa total descontada esperada. "Descontada" significa que una recompensa recibida $s$ pasos en el futuro vale menos que una recibida ahora, por un factor $\gamma^s$ con $0 \le \gamma < 1$. La teoría de programación dinámica (DP) —Bellman & Dreyfus, 1962; Ross, 1983— garantiza que existe al menos una política estacionaria óptima $\pi^*$ y provee métodos para calcular la función de valor óptima $V^*$ y una $\pi^*$, **suponiendo que $\mathcal{R}_x(a)$ y $P_{xy}[a]$ son conocidas**. El valor de un estado bajo una política satisface la ecuación de Bellman: el agente espera recibir la recompensa inmediata por la acción que la política recomienda y luego moverse a un estado que "vale" $V^*(y)$, ponderado por las probabilidades de transición. Aunque parece circular, está bien definido y es la base de toda la teoría.

**El problema del aprendizaje sin modelo.** La dificultad central: el agente de Q-learning enfrenta la tarea de determinar $\pi^*$ **sin conocer inicialmente** $\mathcal{R}_x(a)$ ni $P_{xy}[a]$. Existían métodos tradicionales (Sato, Abe & Takeda, 1988) para aprender el modelo del ambiente mientras se ejecuta DP concurrentemente, pero el paper señala su debilidad: cualquier supuesto de *equivalencia de certeza* —calcular acciones como si el modelo actual fuera exacto— "cuesta caro en las etapas tempranas del aprendizaje" (Barto & Singh, 1990). Watkins clasifica Q-learning como **programación dinámica incremental**, por la forma paso a paso en que determina la política óptima.

**Las diferencias temporales (TD) de Sutton.** El aprendizaje en Q-learning procede de manera similar al método de diferencias temporales (TD) de Sutton (1984; 1988): el agente prueba una acción en un estado y evalúa sus consecuencias en términos de la recompensa o castigo inmediato que recibe *y su estimación del valor del estado al que es llevado*. Probando todas las acciones en todos los estados repetidamente, aprende cuáles son mejores a la larga, juzgadas por la recompensa descontada de largo plazo. Esta idea —*bootstrapping*, estimar a partir de estimaciones— es la herencia directa de TD. El paper agradece de hecho a Rich Sutton "sus incansables esfuerzos" por la claridad del manuscrito.

Q-learning es, en palabras del propio Watkins (1989) citadas aquí, una forma "primitiva" de aprendizaje —pero precisamente por eso puede operar como base de dispositivos mucho más sofisticados. El paper lista usos contemporáneos: Barto & Singh (1990), Sutton (1990), Chapman & Kaelbling (1991), Mahadevan & Connell (1991) y Lin (1992, que lo desarrolló independientemente), más aplicaciones industriales.

## 3. Contribución central: aprender $Q^*$ directamente, sin modelo

La idea fundamental es desplazar el objeto de aprendizaje desde la función de valor de estado $V$ hacia una **función de valor-acción** $Q$. Para una política $\pi$, el valor $Q$ (o *action-value*) se define como la recompensa descontada esperada por ejecutar la acción $a$ en el estado $x$ y *luego* seguir la política $\pi$. Los valores óptimos se definen como $Q^*(x,a) = Q^{\pi^*}(x,a)$.

La conexión con DP es directa y elegante: $V^*(x) = \max_a Q^*(x,a)$, y si $a^*$ es una acción que alcanza ese máximo, entonces una política óptima se forma simplemente como $\pi^*(x) = a^*$. **Aquí reside la utilidad de los valores $Q$**: si un agente puede aprenderlos, puede decidir trivialmente qué es óptimo hacer —basta tomar el argmax sobre las acciones en el estado actual—. Aunque puede haber más de una política óptima o más de una $a^*$, los valores $Q^*$ son únicos.

Esto es lo que hace a Q-learning **model-free** (sin modelo): a diferencia de DP, el agente nunca necesita estimar ni almacenar $P_{xy}[a]$ ni $\mathcal{R}_x(a)$ explícitamente. Toda la información sobre dinámica y recompensas queda absorbida implícitamente en los valores $Q$, que se aprenden por experiencia directa de ensayo y error. Y es **off-policy**: el operador $\max_{a'}$ en la regla de actualización (ver §4) hace que el agente aprenda el valor de la política *greedy óptima* aunque, mientras explora, esté ejecutando una política distinta y subóptima. Esta separación entre la política de comportamiento (la que genera la experiencia) y la política objetivo (la que se evalúa y mejora) es lo que distingue a Q-learning de métodos on-policy como SARSA, y es central para su capacidad de explorar libremente sin sesgar lo que aprende.

## 4. Método: la regla de actualización

En Q-learning la experiencia del agente consiste en una secuencia de etapas o *episodios* distintos. En el $n$-ésimo episodio, el agente:

1. observa su estado actual $x_n$,
2. selecciona y ejecuta una acción $a_n$,
3. observa el estado subsiguiente $y_n$,
4. recibe un pago inmediato $r_n$, y
5. ajusta sus valores $Q_{n-1}$ usando un factor de aprendizaje $\alpha_n$.

La regla de actualización, en la notación moderna que usa la Clase 31, es:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\big]
$$

Las piezas, leídas desde el paper:

- **$Q(s,a)$**: la estimación actual del valor de tomar la acción $a$ en el estado $s$. Es lo que se está corrigiendo.
- **$r$**: la recompensa inmediata observada.
- **$\gamma \max_{a'} Q(s',a')$**: el término que el paper llama "lo mejor que el agente cree que puede hacer desde el estado $y$" (aquí $s'$). El máximo sobre las acciones del estado siguiente es el corazón del carácter off-policy y de la conexión con la ecuación de optimalidad de Bellman.
- **$r + \gamma \max_{a'} Q(s',a') - Q(s,a)$**: el **error de diferencia temporal** (TD error). Es la discrepancia entre el valor estimado actual y un objetivo mejor informado construido con la recompensa real más el valor descontado del mejor sucesor. Si es cero, la estimación ya es consistente con la ecuación de Bellman.
- **$\alpha$** (el $\alpha_n$ del paper): la tasa de aprendizaje, que pondera cuánto corregir hacia el objetivo. El paper la indexa por el número de veces que el par estado-acción ha sido visitado, lo que es esencial para la prueba.

Dos notas críticas del paper sobre el método. Primera: "esta descripción **supone una representación de tabla de búsqueda** (look-up table) para los $Q_n(x,a)$". Watkins (1989) ya había mostrado que Q-learning puede **no** converger correctamente con otras representaciones —una advertencia profética sobre la inestabilidad que reaparecería con aproximadores de función—. Segunda: la condición más importante, implícita en el teorema, es que la secuencia de episodios "debe incluir un número infinito de episodios para cada estado y acción de partida". El paper defiende que esta es una condición fuerte pero ineludible: bajo las condiciones estocásticas del teorema, "ningún método podría garantizar hallar una política óptima bajo condiciones más débiles". Notablemente, los episodios *no* necesitan formar una secuencia continua —el $y$ de un episodio no tiene por qué ser el $x$ del siguiente—, lo que da enorme flexibilidad operacional (incluido el replay de experiencias).

Esta exigencia de visitar infinitamente todos los pares estado-acción es exactamente lo que en la Clase 31 motiva la política **$\varepsilon$-greedy**: con probabilidad $1-\varepsilon$ el agente explota (toma el argmax de $Q$) y con probabilidad $\varepsilon$ explora (acción aleatoria). La exploración garantiza la cobertura que el teorema requiere; la explotación aprovecha lo aprendido. El balance exploración/explotación no es un detalle de implementación sino la condición que hace válida la garantía teórica.

## 5. Teoría: el teorema de convergencia y el action-replay process

### 5.1. El teorema

El enunciado formal: dadas recompensas acotadas $|r_n| \le \mathcal{R}$, tasas de aprendizaje $0 \le \alpha_n < 1$, y definiendo $n^i(x,a)$ como el índice de la $i$-ésima vez que la acción $a$ se prueba en el estado $x$, si se cumple

$$
\sum_{i=1}^{\infty} \alpha_{n^i(x,a)} = \infty, \qquad \sum_{i=1}^{\infty} \big[\alpha_{n^i(x,a)}\big]^2 < \infty
$$

para todo $x, a$, entonces $Q_n(x,a) \to Q^*(x,a)$ cuando $n \to \infty$, para todo $x, a$, con probabilidad 1.

Estas son las clásicas condiciones de **aproximación estocástica de Robbins-Monro**: la suma de las tasas debe diverger (para que el aprendizaje nunca se "congele" antes de tiempo y pueda alcanzar cualquier valor) pero la suma de sus cuadrados debe converger (para que el ruido se promedie a cero y las estimaciones se estabilicen). Una tasa como $\alpha_i = 1/i$ las satisface; una tasa constante $\alpha$ no satisface la segunda y por eso, en la práctica, solo da convergencia aproximada —un punto que la Clase 31 toca al discutir hiperparámetros—.

### 5.2. El action-replay process (ARP)

La clave de la prueba es un proceso de Markov controlado **artificial** llamado *action-replay process* (ARP), construido a partir de la secuencia de episodios y de la secuencia de tasas de aprendizaje. El paper ofrece una analogía memorable —un juego de cartas— para hacerlo intuitivo:

Imagínese cada episodio $(x_t, a_t, y_t, r_t, \alpha_t)$ escrito en una carta. Todas las cartas forman un mazo infinito, con el primer episodio cerca del fondo y extendiéndose infinitamente hacia arriba, en orden. La carta del fondo (numerada 0) lleva escritos los valores iniciales $Q_0(x,a)$. Un estado del ARP, $(x, n)$, consiste en un número de carta (o *nivel*) $n$ junto con un estado $x$ del proceso real. Las acciones permitidas son las mismas que en el proceso real.

La transición funciona así: dado el estado $(x,n)$ y la acción $a$, primero se eliminan todas las cartas de episodios posteriores a $n$, dejando un mazo finito. Se retiran cartas de arriba una a una hasta hallar una cuyo estado y acción de inicio coincidan con $x$ y $a$, digamos en el episodio $t$. Entonces se lanza una moneda sesgada con probabilidad $\alpha_t$ de salir cara. Si sale cara, se *reproduce* (replay) el episodio de esa carta: se emite la recompensa $r_t$ y se transita al estado $(y_t, t-1)$ —un nivel más abajo—. Si sale cruz, se descarta esa carta y se sigue buscando. Si se alcanza la carta del fondo, el juego termina en un estado absorbente especial y entrega la recompensa $Q_0(x,a)$.

El punto crucial: tomar una acción en el ARP **siempre** produce una transición a un nivel más bajo, de modo que el proceso termina tras un número finito de acciones (las cartas solo se quitan, nunca se reponen). El ARP es tan proceso de Markov controlado como el real, de modo que tiene sus propios valores $Q^*$ óptimos bien definidos.

### 5.3. Los dos lemas que articulan la prueba

La demostración descansa sobre dos lemas:

- **Lema A:** "Los $Q_n(x,a)$ son los valores-acción óptimos para los estados $(x,n)$ y las acciones $a$ del ARP." El ARP fue *construido* para tener esta propiedad; el lema se prueba por inducción hacia atrás, descendiendo por la pila de episodios pasados. La intuición: la regla de actualización de Q-learning es exactamente la ecuación de optimalidad de Bellman *dentro* del ARP.

- **Lema B:** "El ARP converge al proceso real." Se descompone en cuatro partes. **B.1**: por el factor de descuento $\gamma$, ignorar el valor del estado $(s+1)$-ésimo incurre un error que tiende a 0 cuando $s \to \infty$ (la cola descontada es despreciable). **B.2**: para cualquier nivel $l$ existe un nivel más alto $h$ tal que la probabilidad de "caer por debajo de $l$" tras $s$ acciones, partiendo de arriba de $h$, puede hacerse arbitrariamente pequeña. **B.3**: con probabilidad 1, las probabilidades de transición $P^{(n)}$ y las recompensas esperadas $\mathcal{R}^{(n)}$ del ARP convergen a las del proceso real cuando $n \to \infty$ —y aquí es donde las condiciones sobre las sumas de las tasas de aprendizaje (las de Robbins-Monro) garantizan la convergencia, vía un teorema estándar de aproximación estocástica (Kushner & Clark, 1978)—. **B.4**: si las transiciones y recompensas de dos procesos de Markov aproximadamente iguales están cerca, entonces los valores de las acciones también lo están (la discrepancia crece a lo sumo cuadráticamente en el número de pasos $s$).

Juntando todo: el ARP tiende al proceso real (Lema B), de modo que sus valores $Q$ óptimos tienden a los del proceso real; pero esos valores óptimos del nivel $n$ del ARP son exactamente $Q_n(x,a)$ (Lema A); por lo tanto $Q_n(x,a) \to Q^*(x,a)$ con probabilidad 1. La elegancia del argumento está en que reduce un problema de convergencia estocástica iterativa a un problema de convergencia de un proceso de Markov auxiliar hacia otro.

## 6. Extensiones (sección 4 del paper)

Por claridad, el teorema probado fue algo restringido. El paper esboza dos extensiones usadas en la práctica, para las cuales el resultado de convergencia se mantiene:

- **Caso no descontado ($\gamma = 1$) con estados absorbentes.** En un proceso con estados meta absorbentes —que terminan atrapando al agente—, esa certeza última de quedar atrapado cumple el rol que $\gamma < 1$ jugaba antes: asegura que el valor de cualquier estado bajo cualquier política está acotado y que sigue valiendo el Lema B.1. El paper construye una cota superior cruda para $V^*$ usando $u^* = \max_x u(x)$ (pasos tras los cuales hay probabilidad positiva de haber llegado a una meta) y $p^* = \min_x p(x) > 0$.

- **Actualizar muchos valores $Q$ por iteración.** En vez de cambiar un solo $Q$ por paso (Barto, Bradtke & Singh, 1991), el ARP se modifica menormente para permitir tomar más de una acción por nivel. Mientras se sigan cumpliendo las condiciones estocásticas, la prueba no requiere modificación no trivial; intuitivamente, solo se *acelera* la estimación de recompensas y transiciones.

El paper también señala un puente conceptual: si el agente recuerda los detalles de sus episodios, puede reusarlos más de una vez (equivale a reponer cartas descartadas más abajo en la pila del ARP), lo que en el límite de re-presentar cartas "viejas" infinitamente equivale al paso de equivalencia de certeza —calcular las acciones óptimas para la *muestra* observada del ambiente—. Hay, pues, un *continuo* entre Q-learning puro y los métodos basados en modelo, no una dicotomía. Esto prefigura tanto el *experience replay* (central en DQN) como los métodos model-based modernos.

## 7. Limitaciones reconocidas y heredadas

- **Tabular / look-up table.** La limitación más consecuente, explicitada por el propio paper: la prueba supone una representación tabular discreta de $Q_n(x,a)$, y Watkins (1989) ya advertía que con otras representaciones la convergencia puede fallar. Una tabla con una entrada por par $(s,a)$ no escala: es inviable para espacios de estado grandes o continuos (imágenes, control robótico de alta dimensión). Esta es exactamente la barrera que **Deep Q-Learning** (DQN, Mnih et al., 2013) rompió al sustituir la tabla por una red neuronal como aproximador de $Q$ —pagando el precio de perder las garantías de convergencia y teniendo que reintroducir estabilidad mediante experience replay y una red objetivo separada—.

- **Sin TD($\lambda$) / multi-step.** El teorema solo prueba una versión restringida del algoritmo comprensivo de Watkins (1989): no permite actualizaciones basadas en recompensas de más de una iteración. La generalización multi-paso —el TD($\lambda$) de Sutton, donde una recompensa de $r$ iteraciones atrás se pondera por $\lambda^r$— no se extiende trivialmente, y el paper sugiere que podrían requerirse métodos de prueba alternativos (Kushner & Clark, 1978).

- **Exigencia de exploración exhaustiva.** La condición de visitar infinitamente todo par estado-acción es teóricamente ineludible pero prácticamente costosa, y no dice *cómo* explorar eficientemente —el problema de la exploración eficiente queda abierto—.

## 8. Impacto

Q-learning es uno de los algoritmos más influyentes de toda la historia del aprendizaje por refuerzo, y este paper de 1992 es su acta de nacimiento teórica. Su importancia se mide en tres planos:

1. **Fundacional para el RL sin modelo.** Estableció que un agente puede aprender a actuar óptimamente *con garantía de convergencia* sin construir jamás un mapa de su ambiente, solo por ensayo y error. Combinado con el carácter off-policy, esto lo volvió el algoritmo de referencia para una generación de investigación en RL tabular.

2. **Base de Deep Q-Learning.** La regla de actualización de Watkins y Dayan es, literalmente, el núcleo de DQN: la red neuronal de DQN se entrena para minimizar el mismo error TD $r + \gamma \max_{a'} Q(s',a') - Q(s,a)$, ahora con $Q$ parametrizada por pesos $\theta$. Los logros de DQN en los juegos de Atari (2013/2015) y, por esa vía, la línea que conduce a AlphaGo y al RL profundo moderno, descienden directamente de este algoritmo.

3. **Garantía teórica como sello.** El paper logró lo que "había eludido a la mayoría de los métodos de RL": una prueba de convergencia con probabilidad 1 bajo condiciones razonables. Esa solidez teórica es parte de por qué Q-learning, y no otras heurísticas contemporáneas, se volvió canónico en libros de texto y cursos.

## 9. Conexión con la Clase 31 (Aprendizaje Reforzado)

La Clase 31 dedica su **sección 3** específicamente a Q-Learning, y este paper provee todo su andamiaje conceptual:

- **La ecuación de Bellman.** La clase introduce la ecuación de optimalidad de Bellman como el principio que la regla de actualización persigue: $Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s',a')]$. El error TD del paper es justamente la medida de cuánto viola la estimación actual esta ecuación; cada actualización empuja $Q$ hacia su punto fijo de Bellman.

- **La tabla Q.** La representación look-up table que el paper asume y cuya convergencia prueba es la "tabla $Q$" que la clase manipula: una matriz indexada por estado y acción que el agente rellena por experiencia. Entender el teorema de convergencia es entender *por qué* iterar la regla sobre esa tabla termina dando los valores correctos.

- **$\varepsilon$-greedy y el dilema exploración/explotación.** La condición del teorema de visitar infinitamente todo par $(s,a)$ es la justificación teórica directa de la política $\varepsilon$-greedy que la clase presenta: la exploración ($\varepsilon$) garantiza la cobertura que la convergencia exige, la explotación ($1-\varepsilon$) aprovecha lo aprendido.

- **El factor de descuento $\gamma$.** El mismo $\gamma$ que la clase usa para ponderar recompensas futuras es el que, en el paper, asegura que el problema está bien definido (Lema B.1) y que el agente prefiere recompensas próximas.

- **El salto a Deep Q-Learning.** La clase presenta DQN como la forma de *escalar* este algoritmo tabular a espacios grandes reemplazando la tabla por una red. La limitación tabular que el paper reconoce explícitamente es precisamente el problema que la segunda mitad de la sección de RL de la clase resuelve.

Enlaces internos del site:

- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado)
- Clase: [/clases/clase-31](/clases/clase-31)
- Paper de continuación (escalado profundo): [/papers/dqn-atari-mnih-2013](/papers/dqn-atari-mnih-2013)
