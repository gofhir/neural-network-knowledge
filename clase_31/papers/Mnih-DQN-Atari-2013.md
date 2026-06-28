# Playing Atari with Deep Reinforcement Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Playing Atari with Deep Reinforcement Learning*.
- **Autores:** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller — todos de **DeepMind Technologies** (Londres).
- **Venue:** NeurIPS 2013 Deep Learning Workshop (NIPS DLW 2013). El paper circuló primero —y es citado casi siempre— como **preprint arXiv:1312.5602v1 (19 dic 2013)**, [arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602).
- **Antecesor directo de:** *Human-level control through deep reinforcement learning* (Mnih et al., **Nature 2015**), que añade la **red objetivo** (target network) y escala a 49 juegos. Esta versión de 2013 es la "primera mitad" de esa historia.

Este es el paper que **fundó el deep reinforcement learning** como subcampo viable. Su tesis es directa y se cumple: presenta "el primer modelo de deep learning que aprende exitosamente políticas de control directamente desde entrada sensorial de alta dimensión usando reinforcement learning". El modelo es una **red convolucional** entrenada con una variante de **Q-learning**, cuya entrada son **pixeles crudos** del juego y cuya salida es una función de valor que estima recompensas futuras. Lo aplican a **siete juegos de Atari 2600** del Arcade Learning Environment **sin ajustar la arquitectura ni el algoritmo de un juego a otro**, superando a todos los enfoques previos en seis de ellos y a un **jugador humano experto en tres**.

La contribución de ingeniería que hace todo esto posible —y la que más se enseña— es el **experience replay**: en lugar de aprender de transiciones consecutivas (fuertemente correlacionadas), el agente guarda sus experiencias en un buffer de memoria y muestrea minibatches **aleatorios** de él para cada actualización de pesos. Esto rompe las correlaciones temporales, reutiliza cada experiencia en muchas actualizaciones y suaviza la distribución de entrenamiento sobre comportamientos pasados, estabilizando lo que hasta entonces era una combinación notoriamente inestable: aproximadores no-lineales + aprendizaje off-policy + bootstrapping.

Para la **Clase 31 (Aprendizaje Reforzado)** este paper es la pieza central: la clase enseña Q-Learning tabular y luego da el salto a **Deep Q-Learning (DQN)**, y el Laboratorio 31 es un *Tutorial DQN*. Entender este paper es entender por qué el lab clona la idea de Watkins (Q-learning) y la envuelve en una CNN + buffer de replay.

## 2. Contexto histórico: por qué combinar deep learning con RL era difícil

Aprender a controlar agentes directamente desde entrada sensorial de alta dimensión (visión, habla) es, en palabras del paper, "uno de los desafíos de larga data del reinforcement learning". Hasta 2013, las aplicaciones exitosas de RL sobre estos dominios **dependían de características diseñadas a mano** (hand-crafted features) combinadas con funciones de valor o políticas **lineales**. El rendimiento de tales sistemas dependía críticamente de la calidad de esa representación manual — exactamente el cuello de botella que el deep learning acababa de eliminar en visión (Krizhevsky et al., 2012; AlexNet) y reconocimiento de habla.

La pregunta natural era: ¿se pueden usar las mismas técnicas de deep learning para RL con datos sensoriales? El paper enumera con precisión **por qué no es trivial**, y aquí está el corazón conceptual del problema:

1. **Las señales de recompensa son escasas, ruidosas y retardadas.** El deep learning supervisado se entrena con grandes cantidades de datos etiquetados a mano; RL debe aprender de una **señal escalar de recompensa** que llega tarde. El retardo entre una acción y la recompensa que provoca puede ser de **miles de pasos temporales** — abismal comparado con la asociación directa entrada-objetivo del aprendizaje supervisado (el problema de la *asignación de crédito temporal*).

2. **Las muestras no son i.i.d.** La mayoría de algoritmos de deep learning asumen que las muestras son **independientes**. En RL uno encuentra típicamente **secuencias de estados altamente correlacionados** (los frames consecutivos de un juego son casi idénticos).

3. **La distribución de datos no es estacionaria.** En RL la distribución de datos **cambia a medida que el algoritmo aprende nuevos comportamientos** — problemático para métodos de deep learning que asumen una distribución fija subyacente.

A esto se suma la advertencia teórica que el paper recoge de la sección de trabajo relacionado: ya se había demostrado que **combinar Q-learning (model-free) con aproximadores no-lineales** (Tsitsiklis & Van Roy, 1997), o incluso con **aprendizaje off-policy** (Baird, 1995), **puede hacer que la Q-network diverja**. Esta es la combinación que la comunidad posterior bautizaría como la *tríada mortal* (deadly triad): aproximación de función + bootstrapping + off-policy. Por eso "la mayoría del trabajo en RL se enfocó en aproximadores lineales con mejores garantías de convergencia". El gran logro empírico de DQN es **funcionar de todos modos**, sin garantías teóricas, gracias al experience replay.

El precedente inspirador es **TD-Gammon** (Tesauro, 1995): un programa de backgammon que aprendió por RL puro y self-play hasta nivel sobrehumano, usando un perceptrón multicapa. Pero los intentos de replicar el éxito de TD-Gammon en ajedrez, Go y damas **fracasaron**, creando la creencia generalizada de que era un caso especial que solo funcionaba en backgammon —quizás porque la estocasticidad de los dados ayuda a explorar y suaviza la función de valor—. DQN es, en cierto sentido, la reivindicación de la apuesta de Tesauro veinte años después, ahora con CNNs y hardware moderno.

El antecedente metodológicamente más cercano es **Neural Fitted Q-learning (NFQ)** de Riedmiller (2005) —coautor de este paper—. NFQ optimiza la misma secuencia de pérdidas, pero con una actualización **batch** cuyo costo por iteración escala con el tamaño del dataset; DQN usa **gradiente estocástico** con costo constante bajo por iteración, lo que escala a datasets grandes. Y mientras NFQ visual primero aprendía una representación de baja dimensión con autoencoders profundos, **DQN aplica RL end-to-end directamente desde los pixeles**, aprendiendo características relevantes para discriminar action-values.

## 3. Fundamento: del Q-learning tabular a la Q-network

El paper formaliza el problema como un agente que interactúa con un entorno $E$ (el emulador de Atari) en una secuencia de acciones, observaciones y recompensas. En cada paso el agente elige una acción $a_t$ del conjunto de acciones legales $A = \{1, \dots, K\}$, que modifica el estado interno del emulador y el puntaje. El agente **no observa el estado interno**; solo recibe una imagen $x_t \in \mathbb{R}^d$ (vector de pixeles de la pantalla actual) y una recompensa $r_t$ (el cambio en el puntaje).

Como una sola pantalla no basta para entender la situación (el entorno está **parcialmente observado** y muchos estados son *perceptualmente aliasados* — por ejemplo, no se puede saber la dirección de una pelota de un solo frame), el paper define el estado como la **secuencia** $s_t = x_1, a_1, x_2, \dots, a_{t-1}, x_t$. Esto convierte el problema en un **MDP grande pero finito**, donde cada secuencia es un estado distinto, y permite aplicar métodos estándar de RL.

El objetivo es maximizar la **recompensa futura descontada** $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$, con factor de descuento $\gamma$. Se define la **función de valor de acción óptima**:

$$Q^*(s,a) = \max_\pi \mathbb{E}[R_t \mid s_t=s, a_t=a, \pi]$$

que obedece la **ecuación de Bellman**, la identidad que sostiene todo: si el valor óptimo $Q^*(s',a')$ del siguiente estado fuese conocido para toda acción $a'$, la estrategia óptima es elegir la $a'$ que maximiza $r + \gamma Q^*(s',a')$:

$$Q^*(s,a) = \mathbb{E}_{s' \sim E}\left[ r + \gamma \max_{a'} Q^*(s',a') \,\middle|\, s,a \right]$$

La idea básica de muchos algoritmos de RL es estimar $Q$ usando la ecuación de Bellman como una actualización iterativa: $Q_{i+1}(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q_i(s',a') \mid s,a]$. Estas *value iteration* convergen a $Q^*$ cuando $i \to \infty$. **Pero esto es totalmente impráctico**: la función de valor se estima por separado para cada secuencia, **sin ninguna generalización**. Este es precisamente el muro contra el que choca el **Q-learning tabular** que la Clase 31 enseña primero — funciona en gridworlds con un puñado de estados, pero hay $256^{210 \times 160}$ pantallas posibles en Atari; una tabla es inconcebible.

La solución es usar un **aproximador de función** $Q(s,a;\theta) \approx Q^*(s,a)$. En RL solía ser lineal; aquí es una **red neuronal con pesos $\theta$**, que el paper llama **Q-network**. Se entrena minimizando una secuencia de funciones de pérdida $L_i(\theta_i)$ que cambian en cada iteración $i$:

$$L_i(\theta_i) = \mathbb{E}_{s,a \sim \rho(\cdot)}\left[ (y_i - Q(s,a;\theta_i))^2 \right]$$

donde el **objetivo** (target) es $y_i = \mathbb{E}_{s' \sim E}[r + \gamma \max_{a'} Q(s',a';\theta_{i-1}) \mid s,a]$ y $\rho(s,a)$ es la **distribución de comportamiento** (behaviour distribution) sobre secuencias y acciones. Un detalle crucial que el paper subraya: **los objetivos dependen de los pesos de la red** — en contraste con el aprendizaje supervisado, donde los targets son fijos antes de empezar. Al optimizar $L_i(\theta_i)$, los parámetros de la iteración anterior $\theta_{i-1}$ se mantienen fijos. (En la versión 2013 esto se hace de modo aproximado; la **Nature 2015** lo formaliza con una red objetivo separada — ver §7.)

Diferenciando la pérdida se obtiene el gradiente, y reemplazando las esperanzas por **muestras individuales** del entorno y la distribución de comportamiento se recupera el familiar algoritmo de **Q-learning**. El método es **model-free** (resuelve la tarea directamente con muestras de $E$, sin construir un modelo de $E$) y **off-policy** (aprende sobre la política greedy $a = \max_a Q(s,a;\theta)$ mientras *sigue* una distribución de comportamiento que asegura exploración — típicamente **$\epsilon$-greedy**: la acción greedy con probabilidad $1-\epsilon$ y una acción aleatoria con probabilidad $\epsilon$).

## 4. Contribución central: Deep Q-Learning con Experience Replay

La idea es conectar un algoritmo de RL a una red neuronal profunda que opera directamente sobre imágenes RGB y procesa los datos eficientemente con actualizaciones de gradiente estocástico. Frente al enfoque online de TD-Gammon —que actualiza desde muestras *on-policy* recién generadas—, DQN introduce el **experience replay** (idea original de Lin, 1993, que el paper reescala).

**El mecanismo.** En cada paso temporal se almacena la experiencia del agente $e_t = (s_t, a_t, r_t, s_{t+1})$ en un dataset $D = e_1, \dots, e_N$, una **memoria de replay** acumulada sobre muchos episodios. En el bucle interno del algoritmo se aplican actualizaciones de Q-learning (minibatch updates) a muestras de experiencia $e \sim D$ extraídas **al azar** del pool. Tras el replay, el agente ejecuta una acción según una política $\epsilon$-greedy.

El paper enumera con cuidado las **tres ventajas** del replay sobre el Q-learning online estándar, y vale la pena entender cada una porque son las que estabilizan el entrenamiento:

1. **Eficiencia de datos.** Cada paso de experiencia se usa potencialmente en **muchas actualizaciones de pesos**, no se descarta tras un único uso. En un régimen donde recolectar datos del emulador es caro, reutilizar cada transición es valioso.

2. **Rompe correlaciones, reduce varianza.** Aprender de muestras consecutivas es ineficiente por las **fuertes correlaciones** entre ellas; **aleatorizar las muestras rompe esas correlaciones y reduce la varianza** de las actualizaciones. Este es el punto que ataca directamente el problema #2 del §2.

3. **Evita lazos de retroalimentación divergentes.** Al aprender on-policy, los parámetros actuales determinan la siguiente muestra con la que se entrenan. El ejemplo del paper es nítido: si la acción maximizante es "mover a la izquierda", las muestras de entrenamiento serán dominadas por el lado izquierdo; si luego cambia a "derecha", la distribución de entrenamiento cambia también. Es fácil ver cómo surgen **lazos de retroalimentación indeseados** que pueden atascar los parámetros en un mínimo local pobre o hacerlos **divergir catastróficamente**. Con experience replay, la distribución de comportamiento se promedia sobre muchos estados previos, **suavizando el aprendizaje y evitando oscilaciones o divergencia**.

El paper nota una sutileza importante: al aprender con experience replay, es **necesario aprender off-policy** —porque los parámetros actuales difieren de los que generaron las muestras almacenadas—, lo que **motiva la elección de Q-learning** (que es off-policy por naturaleza). SARSA, al ser on-policy, no encajaría limpiamente con el replay.

El **Algoritmo 1 (Deep Q-learning with Experience Replay)** es el núcleo, y es exactamente el que el Lab 31 implementa:

```
Inicializar memoria de replay D con capacidad N
Inicializar Q con pesos aleatorios
para episodio = 1, M:
    inicializar secuencia s_1 = {x_1} y preprocesar φ_1 = φ(s_1)
    para t = 1, T:
        con probabilidad ε  →  acción aleatoria a_t
        en otro caso        →  a_t = max_a Q(φ(s_t), a; θ)
        ejecutar a_t en el emulador; observar r_t y la imagen x_{t+1}
        s_{t+1} = s_t, a_t, x_{t+1};  preprocesar φ_{t+1} = φ(s_{t+1})
        almacenar (φ_t, a_t, r_t, φ_{t+1}) en D
        muestrear minibatch aleatorio (φ_j, a_j, r_j, φ_{j+1}) de D
        y_j = r_j                                      si φ_{j+1} es terminal
        y_j = r_j + γ max_{a'} Q(φ_{j+1}, a'; θ)        si no es terminal
        paso de descenso de gradiente sobre (y_j − Q(φ_j, a_j; θ))²
```

Dado que usar historias de longitud arbitraria como entrada es difícil, la Q-función opera sobre una **representación de longitud fija** producida por una función $\phi$ (el preprocesamiento + stack de 4 frames; ver §5). En la práctica, $D$ almacena solo las **últimas $N$ tuplas** y muestrea **uniformemente** al azar. El paper reconoce que esto es limitado: el buffer no diferencia transiciones importantes y siempre sobreescribe las antiguas; un muestreo más sofisticado podría enfatizar las transiciones de las que más se aprende (anticipando el *prioritized experience replay* posterior, en la línea del *prioritized sweeping* de Moore & Atkeson, 1993).

## 5. Método: preprocesamiento, arquitectura y entrenamiento

### 5.1. Preprocesamiento y función $\phi$

Los frames crudos de Atari son imágenes de **210 × 160 pixeles con paleta de 128 colores** — costosos de procesar. El preprocesamiento (la función $\phi$) reduce la dimensionalidad en pasos:

1. Convertir RGB a **escala de grises**.
2. **Downsamplear** a 110 × 84.
3. **Recortar** una región de **84 × 84** que captura aproximadamente el área de juego (el crop cuadrado se necesita solo porque la implementación GPU de convoluciones 2D de Krizhevsky espera entradas cuadradas).
4. **Apilar (stack) los últimos 4 frames** de la historia para formar la entrada a la Q-función.

El stack de 4 frames es clave para la **observabilidad parcial**: un solo frame no revela velocidad ni dirección; cuatro frames consecutivos sí. La entrada final a la red es entonces un tensor **84 × 84 × 4**.

### 5.2. Arquitectura de la red (la misma para los 7 juegos)

Una decisión de diseño elegante: en vez de pasar el par (estado, acción) por la red —lo que exige un *forward pass* por acción, con costo lineal en el número de acciones—, DQN usa una arquitectura con **una unidad de salida por acción posible**, y solo el estado como entrada. Así, **un único forward pass calcula los Q-values de todas las acciones** del estado.

La arquitectura exacta, idéntica para los siete juegos:

- **Entrada:** imagen 84 × 84 × 4 (producida por $\phi$).
- **Primera capa oculta:** convolución de **16 filtros 8 × 8 con stride 4**, seguida de no-linealidad **ReLU** (rectifier).
- **Segunda capa oculta:** convolución de **32 filtros 4 × 4 con stride 2**, seguida de ReLU.
- **Capa oculta final:** totalmente conectada, **256 unidades ReLU**.
- **Salida:** capa lineal totalmente conectada, **una salida por acción válida** (entre 4 y 18 acciones según el juego).

A las CNNs entrenadas con este enfoque las llaman **Deep Q-Networks (DQN)** — el nombre que quedó en la historia.

### 5.3. Entrenamiento e hiperparámetros

- **Mismos** arquitectura, algoritmo e hiperparámetros en los siete juegos, demostrando robustez sin información específica del juego.
- **Reward clipping:** como la escala de puntajes varía mucho entre juegos, fijan toda recompensa positiva a **+1**, toda negativa a **−1**, y dejan el 0 sin cambios. Esto limita la escala de las derivadas del error y facilita usar la misma tasa de aprendizaje en todos los juegos. Tiene un costo reconocido: el agente **no puede distinguir recompensas de distinta magnitud**.
- **Optimizador:** **RMSProp** con minibatches de tamaño **32**.
- **Exploración:** política $\epsilon$-greedy con $\epsilon$ **annealado linealmente de 1 a 0.1** sobre el primer millón de frames, y fijo en 0.1 después.
- **Escala:** entrenaron **10 millones de frames** en total, con memoria de replay de **1 millón de frames recientes**.
- **Frame-skipping:** el agente ve y elige acciones cada $k$-ésimo frame (repitiendo la última acción en los saltados), lo que le permite jugar ~$k$ veces más partidas sin aumentar mucho el runtime, porque avanzar el emulador un paso cuesta mucho menos que elegir una acción. Usan **$k = 4$** en todos los juegos **excepto Space Invaders**, donde $k=4$ hacía invisibles los láseres (por su período de parpadeo), así que usaron $k=3$. Esta fue **la única diferencia de hiperparámetros entre juegos**.

## 6. Experimentos y resultados

Los siete juegos: **Beam Rider, Breakout, Enduro, Pong, Q\*bert, Seaquest, Space Invaders**.

### 6.1. Estabilidad del entrenamiento

Evaluar el progreso en RL es difícil. La recompensa total promedio por episodio es **muy ruidosa** porque pequeños cambios en los pesos cambian drásticamente la distribución de estados visitados. Una métrica más estable es el **Q estimado promedio** sobre un conjunto fijo de estados (recolectados con una política aleatoria antes de entrenar): el valor $Q$ predicho aumenta de forma **mucho más suave** que la recompensa total. Crucialmente, **no observaron ningún problema de divergencia en ninguno de sus experimentos** — evidencia empírica de que, pese a la falta de garantías teóricas, el método entrena redes grandes de forma estable con señal de RL y SGD. Esta observación es justamente la respuesta empírica a la "tríada mortal" del §2.

### 6.2. Visualización de la función de valor (Seaquest)

El paper muestra que la función de valor aprendida es interpretable: en Seaquest, el valor predicho **salta cuando aparece un enemigo** (punto A), **alcanza su pico cuando el torpedo está por impactar** (punto B), y **vuelve a su valor original cuando el enemigo desaparece** (punto C). DQN aprende cómo evoluciona el valor a lo largo de una secuencia compleja de eventos.

### 6.3. Evaluación principal

DQN se compara con los mejores métodos de la literatura: **Sarsa** (políticas lineales sobre conjuntos de características diseñadas a mano) y **Contingency** (que augmenta esas características con una representación aprendida de las partes de la pantalla bajo control del agente). Ambos baselines **incorporan conocimiento previo significativo** (sustracción de fondo, tratar cada uno de los 128 colores como un canal separado). **DQN solo recibe los pixeles RGB crudos y debe aprender a detectar objetos por sí mismo.** También se compara con un **humano experto** (recompensa mediana tras ~2 horas de juego) y una política aleatoria, y con la búsqueda evolutiva **HNeat**.

Tabla de resultados (recompensa total promedio, $\epsilon$-greedy con $\epsilon=0.05$):

| Método | B. Rider | Breakout | Enduro | Pong | Q\*bert | Seaquest | S. Invaders |
|---|---|---|---|---|---|---|---|
| Random | 354 | 1.2 | 0 | −20.4 | 157 | 110 | 179 |
| Sarsa | 996 | 5.2 | 129 | −19 | 614 | 665 | 271 |
| Contingency | 1743 | 6 | 159 | −17 | 960 | 723 | 268 |
| **DQN** | **4092** | **168** | **470** | **20** | **1952** | **1705** | **581** |
| Human | 7456 | 31 | 368 | −3 | 18900 | 28010 | 3690 |
| HNeat Best | 3616 | 52 | 106 | 19 | 1800 | 920 | 1720 |
| HNeat Pixel | 1332 | 4 | 91 | −16 | 1325 | 800 | 1145 |
| **DQN Best** | 5184 | 225 | 661 | 21 | 4500 | 1740 | 1075 |

Lectura: **DQN supera a Sarsa y Contingency por un margen sustancial en los siete juegos**, pese a no usar casi conocimiento previo. Frente a HNeat —que se evalúa solo en su mejor episodio único porque produce políticas determinísticas que explotan secuencias fijas y no generalizan—, DQN gana en todos menos Space Invaders, **incluso comparando el promedio de DQN contra el mejor episodio de HNeat**. Y lo más comentado: **DQN supera al experto humano en Breakout, Enduro y Pong**, y se acerca al humano en Beam Rider. En Q\*bert, Seaquest y Space Invaders queda muy lejos del humano, porque esos juegos exigen estrategias que se extienden sobre **escalas temporales largas** — el talón de Aquiles del horizonte de descuento y la asignación de crédito.

## 7. Limitaciones

- **No hay red objetivo (target network).** Esta es la limitación más importante para entender la evolución del campo. En esta versión 2013, los targets $y_i = r + \gamma \max_{a'} Q(s',a';\theta_{i-1})$ se calculan con la **misma red** que se está actualizando (con $\theta_{i-1}$ "congelados" solo de manera aproximada). Esto deja un objetivo móvil que persigue a la propia predicción, fuente de inestabilidad. La solución —una **red objetivo separada** $\hat{Q}$ con parámetros $\theta^-$ que se copian periódicamente desde la red online— llega en la **Nature 2015**, no aquí. El que DQN-2013 funcione bien *sin* target network se debe en gran medida al experience replay.
- **Reward clipping borra magnitudes.** Fijar todas las recompensas a $\pm 1$ impide distinguir una recompensa pequeña de una grande, limitando el comportamiento en juegos donde la magnitud importa.
- **Replay uniforme y de tamaño finito.** El buffer no prioriza transiciones informativas y siempre sobreescribe las antiguas; el muestreo uniforme da igual peso a todas. El paper mismo señala el camino hacia *prioritized replay*.
- **Memoria parcialmente observable acotada.** El stack de 4 frames captura dinámica de corto plazo, pero no dependencias de largo horizonte — de ahí el bajo rendimiento en juegos de estrategia larga (Q\*bert, Seaquest, Space Invaders).
- **Sin garantías de convergencia.** Es un resultado empírico; no hay prueba de convergencia para esta combinación no-lineal + off-policy + bootstrapping.

## 8. Impacto: el nacimiento del deep RL

Este paper **inauguró el deep reinforcement learning** como campo. Demostró por primera vez que una sola arquitectura, sin ingeniería específica por tarea, podía aprender control de nivel humano (y superarlo) directamente desde pixeles. La receta —CNN como aproximador de $Q$, experience replay para estabilizar, $\epsilon$-greedy para explorar, reward clipping para uniformar— se volvió el patrón base sobre el que se construyó casi todo lo posterior: la **Nature 2015** (target network, 49 juegos, nivel humano comparado sistemáticamente), **Double DQN** (corrige el sesgo de sobreestimación del operador max), **Dueling DQN** (separa valor de estado y ventaja de acción), **Prioritized Experience Replay**, **Rainbow** (combina todas las mejoras), y la línea que culmina en **AlphaGo / AlphaZero** del mismo grupo. El experience replay, en particular, sigue siendo un ingrediente estándar de los algoritmos off-policy modernos.

## 9. Conexión con la Clase 31 (Aprendizaje Reforzado)

La Clase 31 sigue exactamente el arco histórico de este paper. Mapeo pieza por pieza:

- **De Q-Learning tabular a DQN.** La clase enseña primero **Q-Learning** (Watkins & Dayan, 1992) en su forma tabular: una tabla $Q[s][a]$ actualizada con la regla $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$. Este paper muestra **por qué la tabla no escala** (un estado por pantalla de Atari es imposible) y **cuál es el salto**: reemplazar la tabla por una **red neuronal** $Q(s,a;\theta)$ que generaliza entre estados similares. El mismo target de TD, el mismo $\max$ de Bellman, la misma naturaleza off-policy — solo cambia el "almacén" de la Q de tabla a red.

- **El Laboratorio 31 (Tutorial DQN)** implementa directamente el **Algoritmo 1** de §4: un buffer de replay (`deque` o array circular de tuplas $(s,a,r,s',\text{done})$), una red que mapea estado → Q-values por acción, el bucle de interacción $\epsilon$-greedy, el muestreo de minibatches aleatorios, el cálculo del target $y_j$ (con el caso terminal vs. no-terminal del pseudocódigo) y el paso de descenso de gradiente sobre el error TD al cuadrado. Cada línea del Algoritmo 1 tiene su contraparte en el código del lab.

- **Por qué el lab usa replay.** Las tres ventajas del §4 son la justificación pedagógica directa de por qué el lab no entrena online sobre transiciones consecutivas: romper correlaciones, reutilizar datos y evitar divergencia. Si el lab usa una variante con **target network** (común en tutoriales modernos de DQN), conviene aclarar que **esa pieza viene de la Nature 2015, no de este paper de 2013** (ver §7).

### Enlaces internos del curso

- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado)
- Clase: [/clases/clase-31](/clases/clase-31)
- Laboratorio: [/laboratorios/lab-31](/laboratorios/lab-31)
- Paper antecesor (Q-learning tabular): [/papers/q-learning-watkins-1992](/papers/q-learning-watkins-1992)
- Paper sucesor (red objetivo, nivel humano): [/papers/dqn-nature-mnih-2015](/papers/dqn-nature-mnih-2015)
