# Asynchronous Methods for Deep Reinforcement Learning (A3C) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Asynchronous Methods for Deep Reinforcement Learning*.
- **Autores:** Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Tim Harley, Timothy P. Lillicrap, David Silver, Koray Kavukcuoglu. Todos en **Google DeepMind** (Mirza además afiliado a MILA, Universidad de Montreal).
- **Venue:** *Proceedings of the 33rd International Conference on Machine Learning (ICML 2016)*, Nueva York. JMLR W&CP volumen 48.
- **Año:** 2016. **Preprint:** arXiv:1602.01783v2 (16 jun 2016), [arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783).
- **Linaje:** sale del mismo grupo que DQN (Mnih et al., 2013; 2015 *Nature*). Es, en buena medida, la respuesta de DeepMind a las limitaciones prácticas de su propio DQN.

El paper propone un marco "conceptualmente simple y liviano" para aprendizaje reforzado profundo basado en **descenso de gradiente asíncrono**. La idea central es ejecutar **múltiples agentes (workers) en paralelo, cada uno con su propia copia del ambiente**, y actualizar de forma asíncrona un conjunto de parámetros compartidos. El paper presenta variantes asíncronas de cuatro algoritmos estándar de RL —one-step Q-learning, one-step Sarsa, n-step Q-learning y advantage actor-critic— y muestra que el paralelismo tiene un **efecto estabilizador** que permite entrenar redes neuronales con todos ellos de forma confiable, sin recurrir a experience replay.

La estrella del trabajo es la variante actor-critic, **A3C (Asynchronous Advantage Actor-Critic)**. A3C supera el estado del arte en el dominio Atari 2600 entrenando **la mitad del tiempo en una sola CPU multi-core, sin GPU**, y además resuelve tareas de control motor continuo (MuJoCo), conducción 3D (TORCS) y navegación en laberintos 3D generados aleatoriamente (Labyrinth) a partir de entrada puramente visual. Los autores la describen como "el agente de RL más general y exitoso hasta la fecha" porque funciona en 2D y 3D, espacios de acción discretos y continuos, y con redes feedforward y recurrentes.

Para la **Clase 31 (Aprendizaje Reforzado)** este paper importa porque representa la **otra gran familia** del campo: mientras DQN y Q-learning son métodos *value-based* (aprenden una función de valor y derivan la política de ella), A3C es un método *actor-critic* que **parametriza y optimiza la política directamente**. Entender A3C es entender por qué el RL moderno —A2C, PPO, la base de RLHF en LLMs— se construyó sobre el paradigma policy-gradient en lugar de quedarse en value-learning puro.

## 2. Contexto: por qué DQN necesitaba el experience replay y cómo prescindir de él

Para situar A3C hay que entender primero el problema que resuelve. Cuando un agente de RL online recorre el ambiente, la secuencia de observaciones que encuentra es **no estacionaria** y los updates online consecutivos están **fuertemente correlacionados** (los estados sucesivos se parecen mucho entre sí). Entrenar una red neuronal sobre datos así correlacionados desestabiliza el aprendizaje hasta el punto de divergencia, razón por la cual "se pensaba que la combinación de RL online simple con redes neuronales profundas era fundamentalmente inestable".

La solución de **DQN** (Mnih et al., 2015) fue el **experience replay**: guardar las transiciones del agente en una memoria grande y muestrear minibatches aleatorios de distintos pasos de tiempo. Esto descorrelaciona los updates y reduce la no estacionariedad. Pero el replay tiene tres costos que el paper enumera explícitamente:

1. **Memoria y cómputo:** usa más memoria y más cómputo por cada interacción real con el ambiente (hay que almacenar y volver a procesar transiciones viejas).
2. **Obliga a off-policy:** muestrear datos generados por una política antigua exige que el algoritmo de aprendizaje sea **off-policy** (capaz de aprender de datos producidos por una política distinta de la actual). Esto descarta de raíz a toda la familia *on-policy* —Sarsa, actor-critic, métodos policy-gradient— que es justamente donde viven los algoritmos más fundamentales del campo.
3. **Hardware especializado:** los enfoques previos dependían fuertemente de GPUs (DQN, Double DQN, Prioritized DQN) o de arquitecturas masivamente distribuidas (Gorila: 130 máquinas, 100 actor-learners + 30 parameter servers).

El paper propone un **paradigma distinto**: en vez de descorrelacionar con memoria, descorrelacionar con **diversidad espacial**. Si en cualquier instante hay 16 workers explorando partes diferentes del ambiente con políticas de exploración distintas, el flujo agregado de updates que llega a los parámetros compartidos es "más estacionario" y menos correlacionado en el tiempo que el de un solo agente online. Los workers paralelos **cumplen el rol estabilizador que en DQN cumplía el experience replay** — pero sin memoria, y sin la restricción de off-policy. Esto reabre la puerta a aplicar robustamente la familia on-policy completa (Sarsa, n-step, actor-critic) con redes profundas.

El **antecedente directo** es Gorila (Nair et al., 2015), que ya hacía entrenamiento asíncrono distribuido de DQN sobre 130 máquinas con un parameter server central. La innovación de A3C es hacer lo mismo **en una sola máquina con hilos de CPU**: mantener los learners en un solo nodo elimina los costos de comunicación de gradientes/parámetros y habilita updates estilo **Hogwild!** (Recht et al., 2011) — escrituras sin locks sobre los parámetros compartidos.

## 3. Contribución central

La contribución de A3C es un **marco asíncrono multi-hilo** que descansa en dos ideas de diseño:

1. **Actor-learners asíncronos en un solo nodo.** Como Gorila, pero usando múltiples hilos de CPU sobre una misma máquina en lugar de máquinas separadas con parameter server. Esto elimina el costo de comunicación y permite updates lock-free (Hogwild!).
2. **Múltiples actores descorrelacionan los datos.** Como en cualquier instante los workers exploran estados diferentes —y se les puede dar deliberadamente políticas de exploración distintas—, los cambios agregados a los parámetros están menos correlacionados en el tiempo. **Esto reemplaza al experience replay.**

De estas dos ideas se desprenden los aportes concretos:

- Cuatro algoritmos clásicos vueltos asíncronos y estables con redes profundas (1-step Q, 1-step Sarsa, n-step Q, advantage actor-critic), demostrando que el efecto estabilizador del paralelismo aplica tanto a métodos value-based/off-policy como policy-based/on-policy.
- **A3C** en particular: actor-critic con **función de ventaja** $A = Q - V$, **n-step returns** para propagación rápida de recompensa, y **regularización por entropía** para mejorar exploración.
- Un resultado de eficiencia sorprendente: speedup roughly lineal en el número de workers, mejores resultados en Atari **en mucho menos tiempo de pared y sin GPU**, abaratando radicalmente el RL profundo.

## 4. Método

### 4.1. Background: value-based vs. policy-based

El paper recuerda el marco estándar: en cada paso $t$ el agente recibe un estado $s_t$, elige una acción $a_t$ según su política $\pi$, recibe recompensa $r_t$ y pasa a $s_{t+1}$. El objetivo es maximizar el retorno esperado $R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ con factor de descuento $\gamma \in (0,1]$.

- **Value-based** (p.ej. Q-learning, DQN): se aproxima la función de valor de acción $Q(s,a;\theta)$ y la política se deriva de ella (típicamente $\varepsilon$-greedy). El update de one-step Q-learning empuja $Q(s,a)$ hacia el target $r + \gamma \max_{a'} Q(s',a';\theta^-)$.
- **Policy-based** (p.ej. REINFORCE, Williams 1992): se parametriza la política $\pi(a|s;\theta)$ **directamente** y se hace ascenso de gradiente sobre $\mathbb{E}[R_t]$. El gradiente REINFORCE es $\nabla_\theta \log \pi(a_t|s_t;\theta)\, R_t$, un estimador no sesgado de $\nabla_\theta \mathbb{E}[R_t]$.

### 4.2. Actor-critic y la función de ventaja

El gradiente REINFORCE puro tiene **varianza alta**. Se la reduce —sin introducir sesgo— restando una **baseline** $b_t(s_t)$ del retorno: $\nabla_\theta \log \pi(a_t|s_t;\theta)\,(R_t - b_t(s_t))$. La baseline que se usa habitualmente es una estimación aprendida de la función de valor de estado, $b_t(s_t) \approx V^\pi(s_t)$.

Cuando la baseline es $V$, la cantidad $R_t - b_t$ que escala el gradiente de la política es una estimación de la **función de ventaja**:

$$A(a_t, s_t) = Q(a_t, s_t) - V(s_t)$$

porque $R_t$ estima $Q^\pi(a_t,s_t)$ y $b_t$ estima $V^\pi(s_t)$. La ventaja mide *cuánto mejor que el promedio* es tomar la acción $a_t$ en el estado $s_t$. Esto da la arquitectura **actor-critic**: la política $\pi$ es el **actor** (decide qué hacer) y la baseline/función de valor $V$ es el **crítico** (juzga cuán buena fue la decisión). De ahí el nombre **Advantage Actor-Critic**.

### 4.3. n-step returns

Tanto el n-step Q como A3C operan en *forward view*: en lugar de actualizar con el retorno de un solo paso, se computan **n-step returns** explícitos. Una sola recompensa $r$ afecta directamente el valor de los $n$ estados-acción precedentes, en vez de propagarse lentamente paso a paso. El estimador de ventaja de A3C es:

$$A(s_t, a_t) = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(s_{t+k};\theta_v) - V(s_t;\theta_v)$$

donde $k$ varía según el estado y está acotado por $t_{max}$. Los autores prefieren el forward view (calcular n-step returns directamente como targets) sobre el backward view de las eligibility traces porque resulta más fácil entrenar redes con métodos de momento y backprop.

### 4.4. El algoritmo asíncrono A3C

Cada hilo actor-learner ejecuta (Algoritmo S3 del paper):

1. Sincroniza sus parámetros específicos del hilo $\theta', \theta_v'$ con los compartidos $\theta, \theta_v$.
2. Actúa según $\pi(a_t|s_t;\theta')$ durante hasta $t_{max}$ pasos (o hasta estado terminal), recolectando recompensas.
3. Calcula el retorno bootstrappeado $R$ (0 si terminal, $V(s_t;\theta_v')$ si no — "bootstrap desde el último estado").
4. Recorre hacia atrás los pasos acumulando gradientes:
   - **Actor:** $d\theta \leftarrow d\theta + \nabla_{\theta'} \log \pi(a_i|s_i;\theta')\,(R - V(s_i;\theta_v'))$.
   - **Crítico:** $d\theta_v \leftarrow d\theta_v + \partial (R - V(s_i;\theta_v'))^2 / \partial \theta_v'$ (regresión del valor al retorno).
5. Aplica un **update asíncrono** de $\theta$ y $\theta_v$ con los gradientes acumulados.

Detalles clave:

- **Parámetros compartidos.** Aunque actor y crítico se muestran como redes separadas por generalidad, en la práctica **comparten casi todas las capas**: una CNN con una salida softmax para $\pi(a_t|s_t;\theta)$ y una salida lineal para $V(s_t;\theta_v)$, con todas las capas no-output compartidas.
- **Acumulación de gradientes.** Acumular updates sobre varios pasos (similar a usar minibatches) reduce la probabilidad de que múltiples actor-learners se sobrescriban mutuamente y permite cambiar eficiencia de cómputo por eficiencia de datos.
- **Diversidad de exploración.** Dar a cada hilo una política de exploración distinta (p.ej. $\varepsilon$-greedy con $\varepsilon$ muestreado periódicamente de una distribución) mejora robustez y desempeño, y maximiza la descorrelación.

### 4.5. Regularización por entropía

A3C añade la **entropía de la política** a la función objetivo para "desincentivar la convergencia prematura a políticas determinísticas subóptimas" (técnica de Williams & Peng, 1991). El gradiente del objetivo completo es:

$$\nabla_{\theta'} \log \pi(a_t|s_t;\theta')\,(R_t - V(s_t;\theta_v)) + \beta\, \nabla_{\theta'} H(\pi(s_t;\theta'))$$

donde $H$ es la entropía y $\beta$ controla la fuerza del término. Una política con alta entropía sigue explorando; el bonus de entropía evita que el actor colapse demasiado rápido a una acción única. En los experimentos de Atari/TORCS se usó $\beta = 0.01$. (En el caso continuo de MuJoCo se usó la entropía diferencial de la normal de salida, $-\tfrac12(\log(2\pi\sigma^2)+1)$, con multiplicador $10^{-4}$.)

### 4.6. Optimización y configuración

Se probaron SGD con momento, RMSProp con estadísticas por hilo y **RMSProp con estadísticas compartidas**; esta última resultó la más robusta y es la usada. Los updates son lock-free (estilo Hogwild!). Configuración típica de Atari: **16 hilos actor-learner, sin GPU**, updates cada 5 acciones ($t_{max}=5$), target network compartida actualizada cada 40000 frames (en los métodos value-based), $\gamma=0.99$, learning rate inicial muestreado de $\text{LogUniform}(10^{-4}, 10^{-2})$ y recocido a 0.

## 5. Experimentos

Cuatro plataformas:

- **Atari 2600 (Arcade Learning Environment).** Los cuatro métodos asíncronos entrenan exitosamente con solo 16 cores de CPU; los métodos asíncronos aprenden **más rápido que DQN** (entrenado en una GPU Nvidia K40), con A3C —el único policy-based— superando significativamente a los tres value-based. Sobre 57 juegos, A3C **mejora el estado del arte en la mitad del tiempo de entrenamiento** usando solo CPU: A3C LSTM alcanza 623.0% de score humano-normalizado medio (4 días en CPU) frente a 121.9% de DQN (8 días en GPU) y 463.6% de Prioritized DQN. Tras **un solo día**, A3C ya iguala el score medio de Dueling Double DQN. Se entrenaron un agente feedforward y uno recurrente (256 celdas LSTM tras la última capa oculta).
- **TORCS (simulador de carreras 3D).** Exige aprender la dinámica del auto, no solo reaccionar a píxeles. A3C fue el mejor de los cuatro métodos, alcanzando entre 75% y 90% del score de un tester humano en ~12 horas de entrenamiento.
- **MuJoCo (control motor continuo).** Solo se evaluó A3C, porque —a diferencia de los métodos value-based— se extiende fácilmente a acciones continuas (la política emite media $\mu$ y varianza $\sigma^2$ de una normal de la que se muestrea la acción). Encontró buenas soluciones en menos de 24 horas, normalmente en pocas horas, tanto desde estado físico como desde píxeles.
- **Labyrinth (navegación en laberintos 3D aleatorios).** Tarea nueva y más difícil: cada episodio presenta un laberinto generado al azar, de modo que el agente debe aprender una **estrategia general de exploración**, no memorizar un mapa. El agente A3C LSTM, con solo imágenes RGB 84×84 de entrada, aprendió una estrategia razonable (encontrar el portal y volver a él tras cada respawn).

**Escalabilidad.** El speedup de entrenamiento es roughly lineal en el número de actor-learners; con 16 hilos se logra al menos un orden de magnitud de aceleración. Sorprendentemente, one-step Q y Sarsa muestran speedups **superlineales** que no se explican solo por cómputo: con más workers requieren *menos datos* para un score dado, porque el paralelismo reduce el sesgo de los métodos one-step. **Robustez:** sobre 50 learning rates e inicializaciones aleatorias, hay un amplio rango de learning rates que rinde bien, y prácticamente no hay corridas con score 0 en regiones buenas — los métodos son estables y no divergen una vez que empiezan a aprender.

## 6. Limitaciones

- **Forward view sin eligibility traces.** Los autores reconocen que sus métodos n-step usan corrected n-step returns directamente como targets (forward view), mientras que combinar distintos retornos vía eligibility traces (backward view) podría mejorarlos. También señalan que A3C podría beneficiarse de mejores estimadores de ventaja, como **Generalized Advantage Estimation** (Schulman et al., 2015b) — anticipando explícitamente lo que vendría después.
- **No se descarta el replay.** El paper es cuidadoso: mostrar que el Q-learning online estable es posible sin experience replay **no significa que el replay sea inútil**. Incorporar replay al marco asíncrono podría mejorar la eficiencia de datos reusando experiencias viejas, lo que ayudaría en dominios donde interactuar con el ambiente es caro (como TORCS).
- **Sesgo de los métodos value-based.** Los métodos value-based investigados podrían beneficiarse de técnicas de reducción de sobreestimación de Q (Double DQN, etc.), no integradas aquí.
- **Sensibilidad a hiperparámetros y ruido de scheduling.** Aunque robusto dentro de un rango, el desempeño depende del learning rate y del recocido de la exploración; la naturaleza lock-free (Hogwild!) introduce no determinismo que hace difícil reproducir corridas exactas.

## 7. Impacto

A3C fue uno de los papers de RL más influyentes de su época y **popularizó el actor-critic y los métodos on-policy** en el deep RL, que hasta entonces estaba dominado por DQN y sus variantes value-based. Su demostración de que se podía superar el estado del arte **en CPU, sin GPU ni clusters**, democratizó la experimentación y reorientó al campo hacia policy gradients.

Sus descendientes directos definieron el RL moderno:

- **A2C** (Advantage Actor-Critic), la versión **síncrona** de A3C: se descubrió que sincronizar los workers (esperar a que todos terminen su rollout antes de un único update batched) iguala o supera a A3C, es más simple y aprovecha mejor las GPUs. A2C es esencialmente "A3C sin la primera A".
- **PPO** (Schulman et al., 2017), hoy el algoritmo policy-gradient de referencia, hereda el esquema actor-critic con ventaja y rollouts paralelos de A3C y le añade el clipped surrogate objective para updates estables. PPO es además el algoritmo que está **en el corazón de RLHF** para alinear modelos de lenguaje grandes.

La línea actor-critic con ventaja iniciada/popularizada por A3C es, por tanto, la columna vertebral del RL que conecta los juegos Atari con el fine-tuning de los LLMs actuales.

## 8. Conexión con la Clase 31 (Aprendizaje Reforzado)

La Clase 31 cubre Q-Learning y su versión profunda DQN —la familia **value-based**. A3C es el complemento natural: introduce la **otra gran familia, policy-based / actor-critic**, y conviene presentarlo contrastándolo punto por punto con DQN.

| Eje | DQN (value-based) | A3C (actor-critic / policy-based) |
|---|---|---|
| Qué aprende | Función de valor $Q(s,a)$; política derivada ($\varepsilon$-greedy) | Política $\pi(a\|s)$ directa + valor $V(s)$ como crítico |
| On/off-policy | Off-policy | On-policy |
| Descorrelación | Experience replay (memoria grande) | Workers paralelos diversos (sin memoria) |
| Acciones | Solo discretas (argmax sobre $Q$) | Discretas **y continuas** (MuJoCo) |
| Hardware | GPU / clusters | CPU multi-core, barato |
| Exploración | $\varepsilon$-greedy | $\varepsilon$ por hilo + bonus de entropía |

Las tres ideas de A3C que vale la pena que el estudiante de la clase internalice:

1. **Descorrelacionar sin memoria:** la diversidad de workers paralelos reemplaza al experience replay, lo que reabre la puerta a los métodos on-policy.
2. **Ventaja $A = Q - V$:** restar la baseline $V$ reduce la varianza del gradiente de política sin sesgo; el crítico le dice al actor cuánto mejor-que-el-promedio fue su acción.
3. **Entropía para explorar:** el bonus de entropía evita el colapso prematuro de la política, un patrón que reaparece en todo el RL moderno.

**Enlaces internos:**

- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado) — marco MDP, retorno, value vs. policy.
- Clase: [/clases/clase-31](/clases/clase-31) — Aprendizaje Reforzado (Q-Learning, DQN).
- Paper hermano value-based: [/papers/dqn-nature-mnih-2015](/papers/dqn-nature-mnih-2015) — el DQN que A3C busca superar y abaratar.
- Descendiente policy-gradient: [/papers/ppo-schulman-2017](/papers/ppo-schulman-2017) — el sucesor de A3C/A2C que domina el RL actual y RLHF.
