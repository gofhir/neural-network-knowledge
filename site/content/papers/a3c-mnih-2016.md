---
title: "A3C: Asynchronous Advantage Actor-Critic (2016)"
weight: 353
math: true
---

{{< paper-card
    title="Asynchronous Methods for Deep Reinforcement Learning"
    authors="Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Tim Harley, Timothy Lillicrap, David Silver, Koray Kavukcuoglu"
    year="2016"
    venue="ICML 2016"
    pdf="/papers/a3c-mnih-2016.pdf"
    arxiv="1602.01783" >}}
El paper de DeepMind que abrió la **otra gran familia** del deep RL frente a [DQN](/papers/dqn-nature-mnih-2015): los métodos *policy-based* / actor-critic. Su idea central es ejecutar **múltiples workers en paralelo**, cada uno con su propia copia del ambiente, que actualizan de forma asíncrona unos parámetros compartidos. Esa diversidad espacial **decorrelaciona los datos y reemplaza al experience replay** sin necesidad de memoria ni de off-policy. La estrella es **A3C (Asynchronous Advantage Actor-Critic)**: combina función de ventaja $A = Q - V$, *n-step returns* y regularización por entropía, corre en una sola **CPU multi-core sin GPU**, y supera el estado del arte en Atari en la mitad del tiempo de entrenamiento. Es el antecedente directo de A2C y [PPO](/papers/ppo-schulman-2017).
{{< /paper-card >}}

---

## Contexto: por qué DQN necesitaba el replay

Cuando un agente de RL recorre el ambiente online, las observaciones que encuentra son **no estacionarias** y los updates consecutivos están **fuertemente correlacionados** (los estados sucesivos se parecen mucho). Entrenar una red neuronal sobre datos así desestabiliza el aprendizaje hasta divergir; por eso se creía que "RL online simple + redes profundas era fundamentalmente inestable".

La solución de [DQN](/papers/dqn-nature-mnih-2015) (Mnih et al., 2015) fue el **experience replay**: guardar transiciones en una memoria grande y muestrear minibatches aleatorios de distintos pasos de tiempo, lo que decorrelaciona los updates. Pero el replay tiene tres costos:

1. **Memoria y cómputo:** almacenar y reprocesar transiciones viejas en cada interacción real.
2. **Obliga a off-policy:** muestrear datos de una política antigua exige un algoritmo capaz de aprender de una política distinta de la actual. Esto descarta de raíz toda la familia *on-policy* —Sarsa, actor-critic, policy-gradient—, justo donde viven los algoritmos más fundamentales del campo.
3. **Hardware especializado:** los enfoques previos dependían de GPUs (DQN, Double DQN, Prioritized DQN) o de clusters masivos (Gorila: 130 máquinas).

A3C propone un paradigma distinto: en vez de decorrelacionar con **memoria**, decorrelacionar con **diversidad espacial**. Si en cualquier instante 16 workers exploran partes diferentes del ambiente con políticas de exploración distintas, el flujo agregado de updates que llega a los parámetros compartidos es más estacionario y menos correlacionado que el de un solo agente. **Los workers paralelos cumplen el rol estabilizador que en DQN cumplía el replay** —pero sin memoria y sin la restricción de off-policy—, lo que reabre la puerta a la familia on-policy completa con redes profundas.

El antecedente directo es **Gorila** (Nair et al., 2015), que ya hacía DQN asíncrono sobre 130 máquinas con un parameter server. La innovación de A3C es lograr lo mismo **en una sola máquina con hilos de CPU**: eso elimina el costo de comunicar gradientes y habilita updates estilo **Hogwild!** (escrituras sin locks sobre los parámetros compartidos).

## Contribución central

A3C es un **marco asíncrono multi-hilo** que descansa en dos ideas de diseño:

1. **Actor-learners asíncronos en un solo nodo.** Múltiples hilos de CPU sobre una misma máquina, en lugar de máquinas separadas con parameter server. Elimina el costo de comunicación y permite updates lock-free.
2. **Múltiples actores decorrelacionan los datos.** Como cada worker explora estados diferentes —y se les puede dar deliberadamente exploraciones distintas—, los cambios agregados a los parámetros están menos correlacionados en el tiempo. **Esto reemplaza al experience replay.**

De ahí se desprenden los aportes concretos: cuatro algoritmos clásicos vueltos asíncronos y estables (one-step Q-learning, one-step Sarsa, n-step Q-learning y advantage actor-critic), demostrando que el efecto estabilizador del paralelismo aplica tanto a métodos value-based/off-policy como policy-based/on-policy; y **A3C** en particular, con un resultado de eficiencia sorprendente: speedup casi lineal en el número de workers y mejores resultados en Atari en mucho menos tiempo de pared **y sin GPU**.

## Método

### Value-based vs. policy-based

En el marco estándar, en cada paso $t$ el agente recibe un estado $s_t$, elige una acción $a_t$ según su política $\pi$, recibe recompensa $r_t$ y pasa a $s_{t+1}$. El objetivo es maximizar el retorno esperado $R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$.

- **Value-based** (Q-learning, DQN): se aproxima la función de valor $Q(s,a;\theta)$ y la política se deriva de ella ($\varepsilon$-greedy).
- **Policy-based** (REINFORCE, Williams 1992): se parametriza la política $\pi(a|s;\theta)$ **directamente** y se hace ascenso de gradiente sobre $\mathbb{E}[R_t]$, con gradiente $\nabla_\theta \log \pi(a_t|s_t;\theta)\, R_t$.

### Actor-critic y la función de ventaja

El gradiente REINFORCE puro tiene **varianza alta**. Se la reduce —sin sesgo— restando una **baseline** $b_t(s_t)$ del retorno: $\nabla_\theta \log \pi(a_t|s_t;\theta)\,(R_t - b_t(s_t))$. La baseline habitual es una estimación aprendida del valor de estado, $b_t(s_t) \approx V^\pi(s_t)$. Cuando la baseline es $V$, la cantidad que escala el gradiente es la **función de ventaja**:

$$A(a_t, s_t) = Q(a_t, s_t) - V(s_t)$$

La ventaja mide *cuánto mejor que el promedio* es tomar la acción $a_t$ en el estado $s_t$. Esto da la arquitectura **actor-critic**: la política $\pi$ es el **actor** (decide qué hacer) y la función de valor $V$ es el **crítico** (juzga cuán buena fue la decisión). De ahí el nombre **Advantage Actor-Critic**.

### n-step returns

A3C opera en *forward view*: en lugar de actualizar con el retorno de un solo paso, computa **n-step returns** explícitos, de modo que una sola recompensa afecta directamente el valor de los $n$ estados-acción precedentes en vez de propagarse lentamente. El estimador de ventaja es:

$$A(s_t, a_t) = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(s_{t+k};\theta_v) - V(s_t;\theta_v)$$

donde $k$ está acotado por $t_{max}$. El forward view se prefiere sobre las eligibility traces (backward view) porque resulta más fácil entrenar redes con momento y backprop.

### El algoritmo asíncrono

Cada hilo actor-learner: (1) sincroniza sus parámetros con los compartidos; (2) actúa según $\pi$ hasta $t_{max}$ pasos (o estado terminal), recolectando recompensas; (3) calcula el retorno bootstrappeado $R$ (0 si terminal, $V(s_t)$ si no); (4) recorre los pasos hacia atrás acumulando gradientes de **actor** ($\nabla \log \pi \cdot (R - V)$) y de **crítico** (regresión de $V$ al retorno, $\partial(R-V)^2$); (5) aplica un **update asíncrono lock-free** de los parámetros compartidos.

Detalles clave: actor y crítico **comparten casi todas las capas** (una CNN con salida softmax para $\pi$ y salida lineal para $V$); la **acumulación de gradientes** sobre varios pasos —como un minibatch— reduce el riesgo de que workers se sobrescriban; y dar a cada hilo una **exploración distinta** ($\varepsilon$ muestreado por hilo) maximiza la decorrelación.

### Regularización por entropía

A3C añade la **entropía de la política** al objetivo para "desincentivar la convergencia prematura a políticas determinísticas subóptimas":

$$\nabla_{\theta'} \log \pi(a_t|s_t;\theta')\,(R_t - V(s_t;\theta_v)) + \beta\, \nabla_{\theta'} H(\pi(s_t;\theta'))$$

donde $H$ es la entropía y $\beta$ (= 0.01 en Atari) controla su fuerza. Una política con alta entropía sigue explorando; el bonus evita que el actor colapse demasiado rápido a una acción única.

La configuración típica de Atari: **16 hilos actor-learner, sin GPU**, updates cada 5 acciones ($t_{max}=5$), $\gamma=0.99$, **RMSProp con estadísticas compartidas** (la variante más robusta) y learning rate muestreado de $\text{LogUniform}(10^{-4}, 10^{-2})$.

## Experimentos

- **Atari 2600.** Los cuatro métodos asíncronos entrenan con solo 16 cores de CPU y aprenden **más rápido que DQN** (sobre GPU K40), con A3C —el único policy-based— superando a los tres value-based. A3C LSTM alcanza **623% de score humano-normalizado medio en 4 días de CPU** frente a 121.9% de DQN (8 días de GPU); tras un solo día ya iguala a Dueling Double DQN.
- **TORCS (carreras 3D).** A3C fue el mejor de los cuatro, alcanzando 75-90% del score de un tester humano en ~12 horas.
- **MuJoCo (control continuo).** Solo A3C, porque —a diferencia de los value-based— se extiende fácil a acciones continuas (la política emite media $\mu$ y varianza $\sigma^2$ de una normal). Buenas soluciones en menos de 24 horas.
- **Labyrinth (laberintos 3D aleatorios).** Cada episodio genera un laberinto distinto, así que el agente debe aprender una **estrategia general de exploración**, no memorizar un mapa. A3C LSTM lo logró con solo imágenes RGB 84×84.

**Escalabilidad y robustez:** el speedup es casi lineal en el número de workers (one-step Q y Sarsa muestran incluso speedups *superlineales*, porque más workers reducen el sesgo de los métodos one-step). Sobre 50 learning rates e inicializaciones aleatorias, hay un amplio rango que rinde bien y casi no hay corridas que divergan.

## Limitaciones

- **Forward view sin eligibility traces.** Los autores reconocen que combinar retornos vía eligibility traces o usar mejores estimadores como **GAE** (Schulman et al., 2015) podría mejorar A3C —anticipando lo que vendría.
- **No descartan el replay.** Mostrar que el Q-learning online es estable sin replay no significa que el replay sea inútil; sumarlo mejoraría la eficiencia de datos donde interactuar es caro.
- **Sensibilidad y no determinismo.** El desempeño depende del learning rate y del recocido de la exploración; la naturaleza lock-free (Hogwild!) hace difícil reproducir corridas exactas.

## Impacto

A3C **popularizó el actor-critic y los métodos on-policy** en el deep RL, hasta entonces dominado por DQN y sus variantes value-based. Demostrar que se podía superar el estado del arte **en CPU, sin GPU ni clusters** democratizó la experimentación y reorientó al campo hacia los policy gradients. Sus descendientes definieron el RL moderno:

- **A2C** (Advantage Actor-Critic), la versión **síncrona** de A3C: sincronizar los workers (esperar a que todos terminen su rollout antes de un único update batched) iguala o supera a A3C, es más simple y aprovecha mejor las GPUs. Es "A3C sin la primera A".
- **[PPO](/papers/ppo-schulman-2017)** (Schulman et al., 2017), hoy el algoritmo policy-gradient de referencia, hereda el esquema actor-critic con ventaja y rollouts paralelos de A3C y le añade el clipped surrogate objective. PPO es además el algoritmo que está **en el corazón de RLHF** para alinear LLMs.

La línea actor-critic con ventaja iniciada por A3C es la columna vertebral del RL que conecta los juegos Atari con el fine-tuning de los modelos de lenguaje actuales.

## Por qué importa para la Clase 31

La [Clase 31](/clases/clase-31) cubre Q-Learning y DQN —la familia **value-based**. A3C es el complemento natural: introduce la **otra gran familia, policy-based / actor-critic**, y conviene presentarlo contrastándolo punto por punto con DQN.

| Eje | DQN (value-based) | A3C (actor-critic / policy-based) |
|---|---|---|
| Qué aprende | Función de valor $Q(s,a)$; política derivada ($\varepsilon$-greedy) | Política $\pi(a\|s)$ directa + valor $V(s)$ como crítico |
| On/off-policy | Off-policy | On-policy |
| Decorrelación | Experience replay (memoria grande) | Workers paralelos diversos (sin memoria) |
| Acciones | Solo discretas (argmax sobre $Q$) | Discretas **y continuas** (MuJoCo) |
| Hardware | GPU / clusters | CPU multi-core, barato |
| Exploración | $\varepsilon$-greedy | $\varepsilon$ por hilo + bonus de entropía |

Tres ideas que vale la pena internalizar: (1) **decorrelacionar sin memoria** —la diversidad de workers reemplaza al replay y reabre la puerta a los métodos on-policy; (2) **ventaja $A = Q - V$** —restar la baseline $V$ reduce la varianza del gradiente sin sesgo; (3) **entropía para explorar** —evita el colapso prematuro de la política, patrón que reaparece en todo el RL moderno.

## Notas y enlaces

- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado) — marco MDP, retorno, value vs. policy.
- Clase: [/clases/clase-31](/clases/clase-31) — Aprendizaje Reforzado (Q-Learning, DQN).
- Paper hermano value-based: [/papers/dqn-nature-mnih-2015](/papers/dqn-nature-mnih-2015) — el DQN que A3C busca superar y abaratar.
- Descendiente policy-gradient: [/papers/ppo-schulman-2017](/papers/ppo-schulman-2017) — el sucesor que domina el RL actual y RLHF.
- arXiv: [1602.01783](https://arxiv.org/abs/1602.01783) · ICML 2016, JMLR W&CP vol. 48. Afiliación: Google DeepMind.
