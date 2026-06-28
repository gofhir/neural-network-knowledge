# Human-level control through deep reinforcement learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Human-level control through deep reinforcement learning*.
- **Autores:** Volodymyr Mnih, Koray Kavukcuoglu, David Silver (los tres con contribución equitativa), Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg y Demis Hassabis — todos de Google DeepMind (Londres).
- **Venue:** *Nature*, vol. 518, pp. 529–533, 26 de febrero de 2015. DOI: 10.1038/nature14236.
- **Recibido / aceptado:** recibido 10 de julio de 2014; aceptado 16 de enero de 2015.
- **Código:** disponible (solo usos no comerciales) en `https://sites.google.com/a/deepmind.com/dqn`.

Este es el paper que convirtió al aprendizaje reforzado profundo en un campo. Su tesis es directa y ambiciosa: construir **un único agente** —el *deep Q-network* (DQN)— que aprenda políticas de control exitosas **directamente desde píxeles crudos y el puntaje del juego**, usando *la misma* arquitectura de red, los *mismos* hiperparámetros y el *mismo* algoritmo de aprendizaje a través de un conjunto diverso de 49 juegos de Atari 2600, alcanzando un nivel comparable o superior al de un *tester* humano profesional. La frase clave del resumen lo dice sin rodeos: el trabajo "tiende un puente sobre la división entre las entradas sensoriales de alta dimensión y las acciones, dando como resultado el primer agente artificial capaz de aprender a sobresalir en un conjunto diverso de tareas desafiantes".

El problema técnico que el paper resuelve es la **inestabilidad** del aprendizaje por refuerzo cuando se usa un aproximador de función no lineal (una red neuronal) para representar la función de valor-acción $Q$. El propio paper enumera las tres causas de esa inestabilidad: (1) las correlaciones presentes en la secuencia de observaciones; (2) el hecho de que pequeñas actualizaciones a $Q$ pueden cambiar significativamente la política y por ende la distribución de los datos; y (3) las correlaciones entre los valores-acción $Q$ y los valores objetivo $r + \gamma \max_{a'} Q(s', a')$. DQN ataca estas tres con dos mecanismos centrales: **experience replay** y una **red objetivo (target network)** actualizada solo periódicamente. Esa segunda idea es la contribución que distingue esta versión madura de *Nature* del preprint de 2013.

Para la Clase 31 (Aprendizaje Reforzado) esto importa porque **este es el DQN canónico que la clase enseña en el módulo de Deep Q-Learning**: los dos "trucos" que el curso destaca —experience replay y red objetivo— provienen exactamente de aquí, y el laboratorio de la clase los implementa paso a paso. Entender este paper es entender por qué Q-learning con redes neuronales no funciona "de fábrica" y qué dos modificaciones de ingeniería lo vuelven estable.

## 2. Contexto histórico: del DQN 2013 a la portada de Nature

El aprendizaje reforzado tiene un fundamento normativo profundo, arraigado en la psicología (Thorndike, 1911) y la neurociencia (las notables analogías entre las señales fásicas de las neuronas dopaminérgicas y los algoritmos de *temporal difference* — Schultz, Dayan & Montague, 1997). El propio paper abre invocando esa herencia. Pero hasta 2013, los agentes de RL exitosos estaban confinados a dominios donde las características útiles podían *diseñarse a mano* (*handcrafted features*) o a espacios de estado de baja dimensión y completamente observables. TD-Gammon (Tesauro, 1995) había sido un éxito aislado en backgammon; el RL para fútbol robótico (Riedmiller et al., 2009) dependía de representaciones cuidadosamente elaboradas.

La versión **2013** de DQN (Mnih et al., "Playing Atari with Deep Reinforcement Learning", NIPS Deep Learning Workshop) introdujo la idea revolucionaria de combinar Q-learning con una red convolucional profunda y *experience replay*, evaluada sobre 7 juegos de Atari. Fue un resultado llamativo pero preliminar: sin la red objetivo, con menos juegos y sin la evaluación rigurosa contra humanos. Esta versión de **Nature 2015** es la forma madura y definitiva de ese trabajo. Las diferencias clave respecto al preprint de 2013 son:

- **La red objetivo (target network)**: el aporte estabilizador que el preprint 2013 *no* tenía. Es la novedad metodológica central de esta versión.
- **Recorte del error (error clipping)** además del recorte de recompensas.
- **Evaluación rigurosa y a gran escala**: 49 juegos (no 7), comparación cuantitativa contra un *tester* humano profesional y contra el mejor aproximador lineal de la literatura, normalización de desempeño, análisis t-SNE de las representaciones aprendidas y ablaciones que aíslan la contribución de cada componente.

El paper se publicó como artículo destacado en *Nature* —con repercusión de portada— precisamente porque por primera vez un único sistema de aprendizaje *de propósito general*, recibiendo solo los píxeles y el puntaje, dominaba un abanico amplio y variado de tareas. Era un avance hacia la meta central de la inteligencia artificial general (Legg & Hutter, 2007) que había eludido los esfuerzos previos.

El contraste explícito que el paper marca es contra métodos como *neural fitted Q-iteration* (Riedmiller, 2005): aunque existen otros métodos estables para entrenar redes en el contexto de RL, esos involucran el reentrenamiento repetido de redes *de novo* sobre cientos de iteraciones, lo que los vuelve demasiado ineficientes para redes grandes. DQN, en cambio, es eficiente porque aprende online de forma incremental.

## 3. Contribución central

La contribución de DQN-Nature es **un algoritmo único, estable y general** que aprende control a nivel humano desde percepción cruda. Se descompone en cuatro aportes concretos:

1. **La red objetivo (target network)** — la idea estabilizadora principal de esta versión. Una copia $\hat{Q}$ de la red $Q$ se mantiene *congelada* y se usa para generar los valores objetivo del *temporal-difference*; solo se sincroniza con la red $Q$ cada $C$ pasos. Esto rompe la retroalimentación inestable en la que una actualización de $Q(s_t, a_t)$ arrastra también al objetivo, provocando oscilaciones o divergencia.

2. **Experience replay** (heredado y refinado de 2013) — las transiciones $e_t = (s_t, a_t, r_t, s_{t+1})$ se almacenan en una memoria $D$ y el entrenamiento muestrea minibatches *uniformemente al azar* de ese reservorio, rompiendo las correlaciones temporales entre muestras consecutivas y promediando la distribución de comportamiento sobre muchos estados pasados.

3. **Recorte de recompensas (reward clipping)** — todas las recompensas positivas se fijan en $+1$ y las negativas en $-1$ (el $0$ se deja intacto). Esto limita la escala de las derivadas del error y permite usar *la misma* tasa de aprendizaje a través de juegos cuyos puntajes varían en órdenes de magnitud.

4. **Evaluación rigurosa a escala** — 49 juegos con una sola configuración, comparación contra humano experto y baseline lineal, análisis de representaciones (t-SNE) y ablaciones que demuestran que tanto el replay como la red objetivo son *cruciales*.

La idea de diseño que une todo: estabilizar Q-learning con redes profundas requiere **descorrelacionar los datos** (replay) y **descorrelacionar el objetivo de la predicción** (red objetivo). Las dos modificaciones atacan dos de las tres fuentes de inestabilidad que el paper diagnostica.

## 4. Método

### 4.1. Formalización: MDP y Q-learning

El agente interactúa con el emulador de Atari mediante una secuencia de acciones, observaciones y recompensas. En cada paso selecciona una acción $a_t$ del conjunto de acciones legales $A = \{1, \dots, K\}$. Como el estado interno del emulador no es observable y la pantalla actual $x_t$ no basta para entender la situación (estados *perceptualmente aliased*, problema parcialmente observable), el algoritmo trabaja sobre **secuencias** $s_t = x_1, a_1, x_2, \dots, a_{t-1}, x_t$. Esto da lugar a un *Markov decision process* (MDP) grande pero finito, donde cada secuencia es un estado distinto.

El objetivo es maximizar el retorno futuro descontado $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$, con factor de descuento $\gamma = 0.99$. La función de valor-acción óptima es:

$$Q^*(s,a) = \max_\pi \mathbb{E}\!\left[ R_t \mid s_t = s,\, a_t = a,\, \pi \right]$$

y obedece la ecuación de Bellman: si se conociera $Q^*(s', a')$ para todas las acciones siguientes, la estrategia óptima sería elegir $a'$ que maximice $\mathbb{E}[r + \gamma Q^*(s', a')]$. DQN aproxima esta función con una red convolucional, $Q(s, a; \theta_i) \approx Q^*(s,a)$, llamada *Q-network*.

### 4.2. La función de pérdida y la red objetivo

El entrenamiento minimiza una secuencia de funciones de pérdida que cambia en cada iteración $i$:

$$L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_i^{-}) - Q(s, a; \theta_i) \right)^2 \right]$$

Lo crucial está en los parámetros: $\theta_i$ son los pesos de la red $Q$ en la iteración $i$, mientras que $\theta_i^{-}$ son los pesos de la **red objetivo** usados para calcular el objetivo. Estos últimos **solo se actualizan con los de la red $Q$ cada $C$ pasos** y se mantienen fijos entre actualizaciones. El paper explica con precisión por qué esto estabiliza: en Q-learning online estándar, una actualización que aumenta $Q(s_t, a_t)$ frecuentemente *también* aumenta $Q(s_{t+1}, a)$ para todo $a$, y por ende aumenta el objetivo $y_j$, pudiendo provocar oscilaciones o divergencia de la política. Generar los objetivos con un conjunto de parámetros más antiguo introduce un **retraso (delay)** entre el momento en que se actualiza $Q$ y el momento en que esa actualización afecta a los objetivos, haciendo la divergencia mucho menos probable.

El gradiente de la pérdida es:

$$\nabla_{\theta_i} L(\theta_i) = \mathbb{E}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_i^{-}) - Q(s, a; \theta_i) \right) \nabla_{\theta_i} Q(s, a; \theta_i) \right]$$

El algoritmo es **model-free** (resuelve la tarea directamente con muestras del emulador, sin estimar la dinámica de transición $P(r, s' \mid s, a)$) y **off-policy** (aprende sobre la política voraz $a = \arg\max_{a'} Q(s, a'; \theta)$ mientras sigue una política de comportamiento $\epsilon$-greedy que garantiza exploración).

### 4.3. Experience replay en detalle

Las experiencias $e_t = (s_t, a_t, r_t, s_{t+1})$ se almacenan en la memoria $D_t = \{e_1, \dots, e_t\}$ (en la práctica solo las últimas $N = 1$ millón de transiciones). En el bucle interno se aplican actualizaciones de Q-learning sobre muestras $(s, a, r, s') \sim U(D)$ extraídas uniformemente al azar. El paper enumera tres ventajas sobre Q-learning online:

1. **Eficiencia de datos**: cada transición puede reutilizarse en muchas actualizaciones de peso.
2. **Rompe correlaciones**: aprender de muestras consecutivas es ineficiente por las fuertes correlaciones entre ellas; aleatorizar reduce la varianza de las actualizaciones.
3. **Evita bucles de retroalimentación nocivos**: aprendiendo *on-policy*, los parámetros actuales determinan la siguiente muestra (si la acción maximizante es "ir a la izquierda", las muestras se dominan por el lado izquierdo). El replay promedia la distribución de comportamiento sobre muchos estados previos, suavizando el aprendizaje y evitando oscilaciones. El replay obliga a aprender *off-policy*, lo que motiva la elección de Q-learning.

El paper reconoce una limitación del replay tal como está implementado: el muestreo *uniforme* da igual importancia a todas las transiciones, y el buffer siempre sobrescribe con las más recientes. Una estrategia más sofisticada enfatizaría las transiciones de las que más se puede aprender —análogo al *prioritized sweeping* (Moore & Atkeson, 1993)— sembrando explícitamente la idea que más tarde fructificaría en *prioritized experience replay*.

### 4.4. Arquitectura de la red convolucional

La arquitectura usa una decisión de diseño importante: en vez de pasar el par (estado, acción) y obtener un escalar (lo que exigiría un *forward pass* por acción), la red recibe **solo el estado** y produce un valor $Q$ separado por cada acción en la capa de salida — así calcula los valores de todas las acciones con un único *forward pass*. La arquitectura completa:

- **Entrada**: imagen de $84 \times 84 \times 4$ producida por el mapa de preprocesamiento $\phi$ (los 4 frames apilados).
- **Capa convolucional 1**: 32 filtros de $8 \times 8$, stride 4, seguidos de rectificador (ReLU).
- **Capa convolucional 2**: 64 filtros de $4 \times 4$, stride 2, + ReLU.
- **Capa convolucional 3**: 64 filtros de $3 \times 3$, stride 1, + ReLU.
- **Capa totalmente conectada**: 512 unidades rectificadoras.
- **Capa de salida**: lineal, totalmente conectada, con una salida por acción válida (entre 4 y 18 según el juego).

La elección de la red convolucional (LeCun et al., 1998) se inspira en el trabajo de Hubel y Wiesel sobre los campos receptivos en la corteza visual temprana: capas jerárquicas de filtros convolucionales que explotan las correlaciones espaciales locales de las imágenes y aportan robustez ante transformaciones naturales (cambios de escala o punto de vista).

### 4.5. Preprocesamiento

Trabajar con frames crudos de $210 \times 160$ píxeles con paleta de 128 colores es costoso. El preprocesamiento $\phi$ hace: (1) por cada píxel toma el **máximo** entre el frame actual y el anterior, para eliminar el parpadeo (*flickering*) que aparece cuando ciertos sprites de Atari solo se dibujan en frames pares o impares; (2) extrae el canal Y (luminancia) del RGB y lo reescala a $84 \times 84$; (3) apila los $m = 4$ frames más recientes para formar la entrada (el algoritmo es robusto a $m = 3$ o $5$).

Adicionalmente se usa **frame-skipping** con $k = 4$: el agente ve y selecciona acciones solo cada 4-º frame, repitiendo la última acción en los frames omitidos. Como avanzar el emulador es mucho más barato que evaluar la red, esto permite jugar ~4× más partidas sin aumentar significativamente el cómputo.

### 4.6. Detalles de entrenamiento y el algoritmo completo

- **Optimizador**: RMSProp con minibatches de tamaño 32.
- **Política de comportamiento**: $\epsilon$-greedy con $\epsilon$ recocido linealmente de 1.0 a 0.1 sobre el primer millón de frames, fijo en 0.1 después.
- **Escala de entrenamiento**: 50 millones de frames por juego (~38 días de experiencia de juego en total), memoria de replay de 1 millón de frames.
- **Recorte de error**: además del recorte de recompensas, el término de error $r + \gamma \max_{a'} \hat{Q} - Q$ se recorta al intervalo $[-1, 1]$, lo que equivale a usar una pérdida de valor absoluto fuera de ese intervalo — una forma de pérdida de Huber que mejora aún más la estabilidad.

El **Algoritmo 1 (deep Q-learning con experience replay)** integra todo: inicializa la memoria $D$ a capacidad $N$, la red $Q$ con pesos aleatorios $\theta$ y la red objetivo $\hat{Q}$ con $\theta^{-} = \theta$. Por cada episodio y cada paso $t$: con probabilidad $\epsilon$ elige acción aleatoria, si no $a_t = \arg\max_a Q(\phi(s_t), a; \theta)$; ejecuta, observa $r_t$ y $x_{t+1}$; almacena la transición en $D$; muestrea un minibatch al azar; fija $y_j = r_j$ si el episodio termina en $j+1$, o $y_j = r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^{-})$ en caso contrario; da un paso de descenso de gradiente sobre $(y_j - Q(\phi_j, a_j; \theta))^2$; y **cada $C$ pasos resetea $\hat{Q} = Q$**.

## 5. Experimentos y resultados

### 5.1. Configuración y comparación

Se entrenó una red distinta por juego pero con *idéntica* arquitectura, algoritmo y configuración de hiperparámetros a través de los 49 juegos. El único conocimiento previo inyectado fue mínimo: que la entrada son imágenes visuales (motivando la CNN), el puntaje específico del juego, el número de acciones (no sus correspondencias) y el contador de vidas. Los hiperparámetros se eligieron mediante una búsqueda *informal* (no grid search, por el alto costo computacional) sobre cinco juegos de validación —Pong, Breakout, Seaquest, Space Invaders y Beam Rider— y luego se fijaron para los 44 restantes.

La evaluación fue cuidadosa para evitar sobreajuste: cada agente jugó cada juego 30 veces, hasta 5 minutos por partida, con condiciones iniciales aleatorias ("no-op") y política $\epsilon$-greedy con $\epsilon = 0.05$. El *tester* humano profesional usó el mismo motor de emulación bajo condiciones controladas (sin pausar, guardar ni recargar; audio deshabilitado para igualar la entrada sensorial), promediando ~20 episodios de hasta 5 minutos tras ~2 horas de práctica por juego.

### 5.2. Resultados principales

- DQN **supera a los mejores métodos de RL existentes** (incluido el mejor aproximador lineal de Bellemare et al.) en **43 de los 49 juegos**, sin incorporar el conocimiento previo de Atari que usaban esos métodos.
- DQN alcanzó un nivel **comparable al del *tester* humano profesional** a través del conjunto de 49 juegos, logrando **más del 75 % del puntaje humano en más de la mitad de ellos (29 juegos)**.
- El desempeño se normaliza como $100 \times \frac{\text{DQN} - \text{aleatorio}}{\text{humano} - \text{aleatorio}}$, de modo que 100 % = nivel humano y 0 % = juego aleatorio. En la Figura 3, DQN va desde superar masivamente al humano (Video Pinball, Boxing, Breakout, Star Gunner — algunos por encima del 1000 %) hasta quedar muy por debajo en los juegos del fondo de la lista.
- Las **curvas de entrenamiento** (Figura 2) muestran un aprendizaje estable: tanto el puntaje promedio por episodio como el valor $Q$ promedio predicho sobre un conjunto de estados *held-out* crecen suavemente con las épocas en Space Invaders y Seaquest — evidencia de que entrenar redes grandes con señal de RL y SGD se logró de forma estable.

### 5.3. Análisis de representaciones (t-SNE)

Se aplicó t-SNE (van der Maaten & Hinton, 2008) a las representaciones de la última capa oculta sobre estados de Space Invaders. Como era de esperar, t-SNE mapea estados perceptualmente similares a puntos cercanos; pero —y esto es lo interesante— también genera *embeddings* similares para estados perceptualmente **distintos** pero cercanos en términos de recompensa esperada. Por ejemplo, DQN asigna valores de estado altos tanto a pantallas llenas como casi completas, porque aprendió que completar una pantalla lleva a una nueva pantalla llena de naves enemigas. Esto demuestra que la red aprende representaciones que sostienen comportamiento adaptativo a partir de entradas sensoriales de alta dimensión. Más aún, la Figura Extendida 1 muestra que las representaciones **generalizan a datos generados por políticas distintas a la propia** (estados de juego humano y de agente caen en clústeres superpuestos en el *embedding*).

### 5.4. Ablaciones: replay y red objetivo son cruciales

La Tabla Extendida 3 contiene el experimento que valida el diseño. Se entrenaron agentes por 10 millones de frames con **todas las combinaciones** de replay encendido/apagado y red objetivo separada sí/no, con tres tasas de aprendizaje. El resultado demuestra que **ambos componentes son críticos**: desactivar el replay o la red objetivo degrada drásticamente el desempeño. La Tabla Extendida 4 muestra además que reemplazar la red convolucional por un aproximador lineal (manteniendo replay y red objetivo) también colapsa el rendimiento — los tres componentes centrales (replay, red objetivo y CNN profunda) son indispensables.

## 6. Limitaciones reconocidas

- **Planificación temporalmente extendida.** La limitación más célebre: los juegos que demandan estrategias de planificación a largo plazo siguen siendo un desafío mayor para *todos* los agentes existentes, incluido DQN. El ejemplo emblemático es **Montezuma's Revenge**, donde DQN queda en el fondo absoluto de la Figura 3 (esencialmente nivel aleatorio), porque las recompensas son escasas y diferidas y requieren secuencias largas de subobjetivos. Esto contrasta con casos donde DQN *sí* descubre estrategias de largo plazo, como Breakout (aprende a cavar un túnel por el costado del muro para enviar la pelota detrás y destruir muchos bloques de una vez).
- **El recorte de recompensas pierde información de magnitud.** Fijar todas las recompensas a $\pm 1$ permite una sola tasa de aprendizaje a través de juegos, pero el propio paper admite que esto puede afectar el desempeño porque el agente **no puede diferenciar entre recompensas de distinta magnitud**.
- **Replay uniforme y no priorizado.** Como ya se señaló, el muestreo uniforme da igual peso a todas las transiciones y el buffer sobrescribe ciegamente las más antiguas; el paper anticipa que sesgar el contenido del replay hacia eventos salientes (como el *replay* hipocampal priorizado observado empíricamente) sería una mejora futura.
- **Observabilidad parcial.** La aproximación de apilar 4 frames maneja la observabilidad parcial de forma heurística; estados que requieren memoria más allá de esa ventana quedan fuera del alcance del estado representado.

## 7. Impacto y conexión neurocientífica

DQN-Nature es un **hito histórico del aprendizaje reforzado profundo**: el primer agente artificial que aprende a sobresalir en un abanico diverso de tareas desafiantes desde percepción cruda con una sola configuración. Su publicación en *Nature* —con la repercusión que conlleva— marcó el inicio de la era moderna del deep RL y abrió el camino a una cascada de trabajos: Double DQN, Dueling DQN, prioritized experience replay, Rainbow, y eventualmente AlphaGo y AlphaZero del mismo grupo de DeepMind.

El paper enmarca su aporte en términos neurocientíficos, no solo de ingeniería. El *experience replay* tiene un correlato biológico plausible: la reactivación comprimida en el tiempo de trayectorias recientemente experimentadas durante periodos *offline* (por ejemplo, reposo en vigilia) ofrece un mecanismo putativo por el cual las funciones de valor podrían actualizarse eficientemente mediante interacciones con los ganglios basales. El *end-to-end* RL —usar la recompensa para moldear continuamente las representaciones de la CNN hacia las características salientes del ambiente— se apoya en evidencia de que las señales de recompensa durante el aprendizaje perceptual influyen en las características de las representaciones de la corteza visual primate (Law & Gold, 2009; Sigala & Logothetis, 2002). El paper cierra con su tesis: el poder de combinar técnicas de machine learning de punta con mecanismos inspirados en la biología para crear agentes capaces de dominar tareas diversas.

## 8. Conexión con la Clase 31 (Aprendizaje Reforzado)

Este paper *es* el material de la Clase 31 en su segmento de Deep Q-Learning, no una referencia tangencial. Mapeo pieza por pieza:

- **Los "dos trucos clave" que la clase enseña** —experience replay y target network— provienen literalmente de aquí. La clase explica por qué Q-learning con redes neuronales diverge "de fábrica" (las tres fuentes de inestabilidad de §1) y presenta replay + red objetivo como las dos modificaciones que lo arreglan. El estudiante debe entender que el replay descorrelaciona los *datos* y la red objetivo descorrelaciona el *objetivo de la predicción*.

- **El laboratorio de la clase los implementa.** El [lab DQN](/laboratorios/lab-31) construye exactamente el Algoritmo 1 de este paper: una red $Q$ y una red objetivo $\hat{Q}$, un buffer de replay del que se muestrean minibatches, la pérdida de TD con el objetivo $y_j = r_j + \gamma \max_{a'} \hat{Q}(s', a'; \theta^{-})$ calculado con la red congelada, y el reseteo $\hat{Q} \leftarrow Q$ cada $C$ pasos. La política $\epsilon$-greedy con recocido, el factor de descuento $\gamma$ y el optimizador sobre el error TD son todos del paper. El lab típicamente usa un entorno más liviano que Atari (como CartPole o LunarLander de Gym), pero la maquinaria conceptual es idéntica.

- **La red objetivo es el detalle que separa el lab "que funciona" del "que diverge".** Igual que el paper demuestra en la ablación de la Tabla Extendida 3, si el estudiante implementa Q-learning con red neuronal *sin* red objetivo, observará oscilaciones o divergencia; agregar la copia congelada estabiliza el entrenamiento. Esto convierte la ablación del paper en una lección práctica reproducible en el lab.

- **Fundamentos transversales.** Los conceptos base —MDP, ecuación de Bellman, función de valor-acción $Q$, on-policy vs off-policy, model-free, exploración/explotación vía $\epsilon$-greedy— están desarrollados en el fundamento de [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado), que la Clase 31 referencia para anclar la teoría antes de llegar a DQN.

- **Posición en la genealogía del deep RL.** Este paper se lee junto a sus vecinos: el [DQN 2013 de Mnih et al.](/papers/dqn-atari-mnih-2013) es el preludio (la versión sin red objetivo, sobre 7 juegos), y el [Double DQN de van Hasselt et al. 2015](/papers/double-dqn-van-hasselt-2015) es la secuela inmediata que corrige el *sesgo de sobreestimación* del $\max$ en el objetivo de DQN. Entender la transición 2013 → Nature 2015 → Double DQN es entender cómo el campo refinó iterativamente un mismo núcleo algorítmico.

Recursos del curso vinculados: [Clase 31](/clases/clase-31) · [Laboratorio 31](/laboratorios/lab-31) · [Fundamento: Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado) · [DQN Atari (Mnih 2013)](/papers/dqn-atari-mnih-2013) · [Double DQN (van Hasselt 2015)](/papers/double-dqn-van-hasselt-2015).
