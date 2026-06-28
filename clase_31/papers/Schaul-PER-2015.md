# Prioritized Experience Replay — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Prioritized Experience Replay*.
- **Autores:** Tom Schaul, John Quan, Ioannis Antonoglou y David Silver (Google DeepMind).
- **Venue:** *International Conference on Learning Representations* (ICLR 2016), publicado como conference paper.
- **Año / preprint:** 2015 (primera versión en arXiv el 18 nov 2015); versión v4 del 25 feb 2016. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952).
- **Antecedente directo:** el *experience replay* del DQN de Mnih et al. (2013, 2015) y el Double DQN de van Hasselt et al. (2016), sobre el cual se montan los experimentos.

Este paper hace una sola cosa, y la hace muy bien: cambia **cómo se muestrean** las transiciones del *replay buffer* del DQN. La observación de partida es casi trivial una vez enunciada, pero nadie la había explotado a escala: el *experience replay* original samplea las transiciones almacenadas **uniformemente al azar**, es decir, repite cada experiencia con la misma frecuencia con que ocurrió, **sin importar cuán informativa sea**. Pero no todas las transiciones enseñan lo mismo. Algunas son sorprendentes, raras o cargadas de información sobre el error del modelo; otras son redundantes y ya están bien predichas. Replay uniforme las trata a todas igual, lo que desperdicia cómputo en lo que el agente ya domina.

La contribución es **Prioritized Experience Replay (PER)**: muestrear las transiciones con probabilidad creciente en su **error de diferencia temporal (TD-error)** $|\delta|$, de modo que las transiciones "sorprendentes" —aquellas donde la predicción $Q$ está más lejos de su objetivo bootstrap— se repiten más a menudo. Como el muestreo no uniforme introduce un **sesgo** en el estimador (las actualizaciones dejan de provenir de la distribución correcta), PER lo corrige con **pesos de importance sampling** $w_i$ que se *anelan* (annealing) a lo largo del entrenamiento. Todo esto se implementa de forma eficiente con una estructura de datos *sum-tree* (variante proporcional) o un *binary heap* (variante rank-based), de modo que muestrear y actualizar prioridades en un buffer de $10^6$ transiciones cuesta $O(\log N)$.

El resultado empírico, en el benchmark Atari 2600: PER acelera el aprendizaje **por un factor cercano a 2** y mejora el score final sobre DQN y Double DQN con replay uniforme, superando al DQN uniforme en **41 de 49 juegos** y estableciendo un nuevo estado del arte. Su importancia perdura porque PER se convirtió en uno de los seis componentes de **Rainbow** (Hessel et al., 2018), la combinación canónica de mejoras del DQN.

Para la Clase 31 esto importa porque PER es la **mejora más directa al ingrediente de *experience replay*** que la clase enseña como uno de los dos pilares de estabilidad del DQN (el otro es la *target network*). Donde Double DQN ataca el sesgo de sobreestimación del `max` y Dueling DQN reorganiza la arquitectura en $V + A$, PER ataca la *eficiencia de uso de los datos* ya guardados — y el laboratorio de la clase puede implementarlo como un cambio acotado sobre el buffer.

## 2. Contexto: el experience replay uniforme y su desperdicio

El aprendizaje por refuerzo *online* en su forma más simple descarta cada experiencia inmediatamente tras una sola actualización. Esto tiene dos problemas que el paper enumera: (a) las actualizaciones quedan **fuertemente correlacionadas** en el tiempo, rompiendo el supuesto i.i.d. de los algoritmos basados en gradiente estocástico, y (b) las experiencias raras pero útiles se **olvidan rápido**. El *experience replay* (Lin, 1992) resuelve ambos: al guardar las transiciones en una memoria y mezclarlas, rompe las correlaciones temporales y permite reusar cada experiencia más de una vez.

El DQN (Mnih et al., 2013, 2015) hizo de esto un componente central: usó una memoria deslizante grande (las últimas $10^6$ transiciones), de la cual sampleaba **uniformemente al azar**, revisitando cada transición unas ocho veces en promedio. El *experience replay* permite cambiar interacción con el entorno —cara— por cómputo y memoria —baratos.

El punto que el paper subraya es el límite de ese diseño: el replay uniforme "simplemente repite las transiciones a la misma frecuencia con que se experimentaron originalmente, sin importar su relevancia". Una transición puede ser más o menos **sorprendente, redundante o relevante para la tarea**; algunas no son útiles ahora pero lo serán cuando la competencia del agente aumente (Schmidhuber, 1991). El *experience replay* ya liberó al agente de procesar las transiciones en el orden exacto en que ocurren; PER lo libera del último vínculo, el de considerarlas con la misma frecuencia con que ocurrieron.

El paper apela además a dos pilares de motivación. **Neurociencia:** en el hipocampo de roedores se observa *replay* de experiencias, y las secuencias asociadas a recompensa —y las de alto TD-error— se reproducen con mayor frecuencia (Singer & Frank, 2009; McNamara et al., 2014). **Planificación:** el *prioritized sweeping* (Moore & Atkeson, 1993) ordena las actualizaciones de *value iteration* por su impacto esperado, usando el TD-error como medida de prioridad. PER traslada esa idea al RL *model-free* con aproximador de función, con la cautela añadida de una priorización estocástica que es más robusta al ruido.

### 2.1. El ejemplo motivador: Blind Cliffwalk

Para mostrar cuánto se puede ganar, el paper introduce un entorno artificial, el **Blind Cliffwalk**: $n$ estados, dos acciones (una "correcta" que avanza, una "incorrecta" que termina el episodio), y una única recompensa de $1$ al final de la secuencia correcta. La probabilidad de que una secuencia aleatoria de acciones llegue a la recompensa es $2^{-n}$: las transiciones relevantes (los raros éxitos) quedan ocultas en una masa de fracasos redundantes.

Comparando dos agentes que hacen Q-learning sobre la *misma* memoria —uno con replay uniforme, otro con un *oráculo* que elige greedy la transición que más reduce la pérdida global— el oráculo logra un *speed-up exponencial* (nótese la escala log-log de la Figura 1). El oráculo no es realizable (requiere conocer la pérdida tras la actualización), pero la brecha gigantesca justifica la búsqueda de una aproximación práctica. Esa aproximación es priorizar por TD-error.

## 3. Contribución central: priorizar por TD-error

El componente central de PER es el **criterio de importancia** de cada transición. El criterio ideal sería cuánto puede aprender el agente de una transición en su estado actual (el *progreso de aprendizaje esperado*), pero eso no es accesible directamente. Un *proxy* razonable es la **magnitud del TD-error** $\delta$, que indica cuán "sorprendente" es la transición: qué tan lejos está el valor estimado de su estimación bootstrap de un paso. Esto encaja de forma natural con algoritmos *online* como Q-learning o SARSA, que ya computan $\delta$ y actualizan los parámetros en proporción a él.

La versión más simple —**priorización greedy por TD-error**— guarda con cada transición su último $\delta$ y replaya siempre la de mayor $|\delta|$; las transiciones nuevas entran con prioridad máxima para garantizar que toda experiencia se vea al menos una vez. En el Blind Cliffwalk esto ya reduce sustancialmente el esfuerzo de aprendizaje. Pero la versión greedy tiene tres patologías que motivan la solución real del paper:

1. **Estancamiento.** Como los TD-errors solo se actualizan para las transiciones que se replayan (para evitar barridos costosos sobre todo el buffer), una transición con bajo error en su primera visita puede no replayarse durante muchísimo tiempo — con una memoria deslizante, efectivamente nunca.
2. **Sensibilidad al ruido.** Los picos de error por recompensas estocásticas, exacerbados por el *bootstrapping*, hacen que el sistema persiga ruido.
3. **Pérdida de diversidad.** Los errores se encogen lento (sobre todo con aproximación de función), de modo que las transiciones inicialmente de alto error se replayan una y otra vez. Esa falta de diversidad propicia el *overfitting*.

## 4. Método: priorización estocástica, $\alpha$, $\beta$ y pesos de importance sampling

### 4.1. Priorización estocástica

La solución es un muestreo estocástico que **interpola entre la priorización greedy pura y el muestreo uniforme**. Se garantiza que la probabilidad de ser muestreado sea monótona en la prioridad de la transición, pero con probabilidad no nula incluso para la transición de menor prioridad. La probabilidad de muestrear la transición $i$ es:

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}
$$

donde $p_i > 0$ es la prioridad de la transición y el exponente $\alpha$ controla **cuánta priorización se aplica**: $\alpha = 0$ recupera el caso uniforme, y $\alpha = 1$ es priorización plena por prioridad. El paper define dos formas de fijar $p_i$:

- **Proporcional:** $p_i = |\delta_i| + \epsilon$, donde $\epsilon$ es una constante positiva pequeña que evita el caso límite de que una transición con error cero nunca vuelva a visitarse.
- **Rank-based:** $p_i = 1/\text{rank}(i)$, donde $\text{rank}(i)$ es la posición de la transición al ordenar el buffer por $|\delta_i|$. En este caso $P$ se vuelve una distribución de ley de potencias con exponente $\alpha$.

Ambas son monótonas en $|\delta|$, pero la rank-based es más robusta porque es **insensible a outliers** (solo importa el orden, no la magnitud del error). En la práctica ambas dan resultados similares en Atari — el paper conjetura que se debe al uso intensivo de *clipping* de recompensas y TD-errors en el DQN, que ya elimina los outliers.

### 4.2. Corrección del sesgo con importance sampling

Estimar la esperanza con actualizaciones estocásticas exige que esas actualizaciones provengan de la **misma distribución** que la esperanza. PER cambia esa distribución de forma incontrolada, y por tanto introduce un **sesgo**: cambia la solución a la que convergen las estimaciones, aun con política y distribución de estados fijas. La corrección son **pesos de importance sampling (IS)**:

$$
w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^{\beta}
$$

que compensan completamente las probabilidades no uniformes $P(i)$ cuando $\beta = 1$. Estos pesos se incorporan al update de Q-learning usando $w_i \delta_i$ en lugar de $\delta_i$ (es decir, *weighted IS*, no IS ordinario). Por estabilidad, los pesos se normalizan siempre por $1/\max_i w_i$, de modo que solo escalan la actualización **hacia abajo**.

El exponente $\beta$ se **anela** (annealing) linealmente desde un valor inicial $\beta_0$ hasta $1$, alcanzando la corrección plena solo al final del entrenamiento. La justificación: la naturaleza insesgada de las actualizaciones importa más cerca de la convergencia; al inicio el proceso es altamente no estacionario de todos modos (cambian la política, la distribución de estados y los objetivos bootstrap), así que un pequeño sesgo temprano es tolerable. Hay una interacción deliberada entre $\alpha$ y $\beta$: subir ambos a la vez prioriza más agresivamente *y* corrige más fuerte.

Un beneficio extra del IS con aproximación no lineal: la priorización asegura que las transiciones de alto error se vean muchas veces, mientras que el peso IS **reduce la magnitud del gradiente** (y con ello el paso efectivo en el espacio de parámetros), permitiendo seguir la curvatura de paisajes de optimización muy no lineales sin pasos disruptivos.

### 4.3. El algoritmo completo y el sum-tree

PER se integra sobre el **Double DQN** afinado: la única modificación es reemplazar el muestreo uniforme por la priorización estocástica más la corrección IS (Algoritmo 1 del paper). Cada nueva transición entra con prioridad máxima; cada $K$ pasos se muestrea un minibatch según $P(j)$, se computan los pesos $w_j$, el TD-error $\delta_j$ (con el objetivo Double DQN), se actualiza la prioridad $p_j \leftarrow |\delta_j|$ y se acumula el cambio de pesos $\Delta \leftarrow \Delta + w_j \cdot \delta_j \cdot \nabla_\theta Q$.

La eficiencia es crítica con $N = 10^6$: la complejidad de muestrear **no puede depender de $N$**. Dos estructuras de datos lo resuelven:

- **Variante proporcional → *sum-tree*.** Un árbol binario donde cada nodo interno es la *suma* de sus hijos y las hojas guardan las prioridades; la raíz contiene $p_{\text{total}}$. Para muestrear un minibatch de tamaño $k$, se divide $[0, p_{\text{total}}]$ en $k$ rangos iguales, se muestrea un valor uniforme de cada rango y se recupera la hoja correspondiente recorriendo el árbol. Actualizar una prioridad y muestrear cuestan ambos $O(\log N)$.
- **Variante rank-based → *binary heap* + muestreo estratificado.** Se aproxima la función de densidad acumulada con una función lineal por tramos de $k$ segmentos de igual probabilidad; al muestrear se elige un segmento y luego uniformemente dentro de él. Tomar exactamente una transición por segmento es una forma de **muestreo estratificado** que balancea el minibatch (siempre habrá una transición de $|\delta|$ alto, una media, etc.).

El overhead total fue solo de **2 %–4 %** en tiempo de ejecución, con uso de memoria adicional despreciable — barato para la ganancia que entrega.

## 5. Experimentos: Atari 2600

El benchmark es la suite Atari (Bellemare et al., 2012) con su setup de RL end-to-end desde visión. Las baselines son DQN (Nature, Mnih et al. 2015) y Double DQN afinado, ambas con replay uniforme; PER mantiene **idéntica** arquitectura de red, algoritmo de aprendizaje, tamaño de memoria y protocolo de evaluación — la *única* diferencia es el mecanismo de muestreo.

Solo se ajustó un hiperparámetro respecto a la baseline: como PER elige transiciones de alto error más a menudo, las magnitudes de gradiente son mayores, así que se **redujo el step-size $\eta$ por un factor 4**. Para los nuevos $\alpha$ y $\beta_0$ una búsqueda en grilla gruesa (sobre 8 juegos) halló el punto dulce: $\alpha = 0.7,\ \beta_0 = 0.5$ para rank-based y $\alpha = 0.6,\ \beta_0 = 0.4$ para proporcional.

Resultados (Tabla 1 del paper):

- **PER sobre DQN:** mejora el score en **41 de 49 juegos**; la mediana del score normalizado sube de **48 % a 106 %**.
- **PER sobre Double DQN:** la ganancia es **complementaria** a la de Double Q-learning; la mediana sobre 57 juegos sube de **111 % a 128 %** y la media de **418 % a 551 %**, llevando juegos como River Raid, Seaquest y Surround a nivel humano por primera vez. (La media es poco fiable porque un solo juego, Video Pinball, la domina.)
- **Velocidad:** en agregado el aprendizaje es **el doble de rápido**; los puntos de equivalencia (cuando PER iguala el rendimiento final de Double DQN) se alcanzan al 38 %–47 % del tiempo total de entrenamiento. PER también reduce el retraso inicial en juegos que tardan en "despegar" (Battlezone, Zaxxon, Frostbite).

En la discusión, dos hallazgos secundarios elegantes: el muestreo uniforme está **implícitamente sesgado hacia transiciones desactualizadas** (generadas por políticas viejas), mientras que PER —al premiar las transiciones no vistas y las recientes, que tienden a tener mayor error— corrige ese sesgo; y la distribución empírica de TD-errors se vuelve *heavy-tailed* a medida que avanza el aprendizaje, validando la forma de la ecuación de priorización.

## 6. Limitaciones reconocidas

- **El TD-error es solo un *proxy*.** Captura la escala de mejora potencial pero ignora la estocasticidad inherente de recompensas y transiciones, la observabilidad parcial y los límites de capacidad del aproximador. Es problemático cuando hay **transiciones no aprendibles** (ruido irreducible): PER las perseguiría indefinidamente. El apéndice discute alternativas (la *derivada* del error, la norma del cambio de pesos, asimetría entre errores positivos y negativos), pero en los experimentos preliminares ninguna superó a $|\delta|$ — aunque eso quizá diga más sobre lo casi determinista de los entornos probados que sobre las medidas en sí.
- **Solo aborda qué replayar, no qué guardar ni cuándo borrar.** El paper asume que el contenido del buffer está fuera de su control; la gestión de memoria (qué transiciones conservar/erradicar) queda como extensión (Sección 6).
- **Errores desactualizados.** Como las prioridades solo se refrescan al replayar, el $|\delta|$ guardado puede quedar *stale* respecto al modelo actual; la priorización opera sobre una estimación posiblemente vieja del error.
- **Hiperparámetros nuevos.** Introduce $\alpha$, $\beta_0$ y su schedule de annealing, además de exigir reducir $\eta$ — más superficie de ajuste que el replay uniforme.

## 7. Impacto y conexión con la Clase 31

PER se volvió un componente estándar del deep RL basado en valor. Su huella más visible es **Rainbow** (Hessel et al., 2018), que combina seis mejoras del DQN —Double Q-learning, *prioritized replay*, *dueling networks*, *multi-step learning*, *distributional RL* (C51) y *noisy nets*— y muestra, vía estudios de ablación, que **PER es uno de los componentes cuya remoción más degrada el rendimiento**. La idea de muestrear datos por su "sorpresa" también reaparece en arquitecturas distribuidas posteriores (Ape-X, R2D2), donde el cálculo de prioridades se distribuye entre múltiples actores.

Para la Clase 31, PER ocupa un lugar preciso en la secuencia de mejoras del DQN que la clase recorre:

- **Mejora directa al *experience replay*.** La Clase 31 enseña la transición de Q-learning tabular ([Watkins 1992](/papers/q-learning-watkins-1992)) a **DQN** (ver [/papers/dqn-nature-mnih-2015](/papers/dqn-nature-mnih-2015)), donde la *experience replay* y la *target network* son los dos ingredientes que estabilizan el entrenamiento. PER es la mejora más natural sobre el *primer* ingrediente: en vez de samplear uniformemente del buffer, samplear por TD-error. Esto la hace complementaria —no competidora— de las otras mejoras que la clase ve.
- **Encaja en el patrón "mejoras quirúrgicas y combinables".** Donde [Double DQN](/papers/double-dqn-van-hasselt-2015) ataca el sesgo de sobreestimación del `max` y [Dueling DQN](/papers/dueling-dqn-wang-2015) reorganiza la arquitectura en $V + A$, PER ataca la *eficiencia de uso de datos*. Los experimentos del propio paper muestran que la ganancia de PER es complementaria a la de Double DQN; esa composabilidad es la lección de diseño que la clase transmite y que culmina en Rainbow.
- **Lo que el laboratorio implementa.** PER es un cambio acotado y muy implementable sobre el buffer del DQN del laboratorio: reemplazar la lista circular de muestreo uniforme por un *sum-tree*, almacenar prioridades $p_i = |\delta_i| + \epsilon$, muestrear $P(i) \propto p_i^\alpha$, y aplicar el peso $w_i$ normalizado al término de pérdida. El estudiante ve de primera mano por qué el muestreo no uniforme **necesita** la corrección IS — sin ella, el agente converge a una solución sesgada.
- **Fundamento transversal.** El TD-error como señal de "progreso de aprendizaje", la ecuación de Bellman que lo define, y el dilema exploración/explotación que la priorización roza forman parte del cuerpo conceptual de [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado), que la [/clases/clase-31](/clases/clase-31) referencia para anclar la teoría antes de llegar a las mejoras del DQN.

Recursos del curso vinculados: [Clase 31](/clases/clase-31) · [Fundamento: Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado) · [DQN Nature (Mnih 2015)](/papers/dqn-nature-mnih-2015) · [Dueling DQN (Wang 2015)](/papers/dueling-dqn-wang-2015) · [Double DQN (van Hasselt 2015)](/papers/double-dqn-van-hasselt-2015).
