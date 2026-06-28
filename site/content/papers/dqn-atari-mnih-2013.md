---
title: "DQN: Playing Atari with Deep RL (2013)"
weight: 348
math: true
---

{{< paper-card
    title="Playing Atari with Deep Reinforcement Learning"
    authors="Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller"
    year="2013"
    venue="NeurIPS Deep Learning Workshop 2013"
    pdf="/papers/dqn-atari-mnih-2013.pdf"
    arxiv="1312.5602" >}}
El paper de DeepMind que **fundó el deep reinforcement learning**. Presenta el primer modelo de deep learning que aprende políticas de control directamente desde **pixeles crudos** usando RL: una **red convolucional** entrenada con una variante de **Q-learning** que aproxima la función de valor de acción. Se aplica a **siete juegos de Atari 2600** con la **misma arquitectura, algoritmo e hiperparámetros**, superando a todos los métodos previos en seis y a un **experto humano en tres** (Breakout, Enduro, Pong). La pieza de ingeniería que lo hace estable es el **experience replay**: un buffer de transiciones del que se muestrea al azar para romper las correlaciones temporales. Esta versión 2013 **aún no tiene red objetivo** —eso llega en la [Nature 2015](/papers/dqn-nature-mnih-2015).
{{< /paper-card >}}

---

## Contexto: por qué deep learning + RL era difícil

Aprender control directamente desde entrada sensorial de alta dimensión (visión, audio) era, según el paper, "uno de los desafíos de larga data del reinforcement learning". Hasta 2013 las aplicaciones exitosas de RL sobre estos dominios **dependían de características diseñadas a mano** combinadas con funciones de valor o políticas **lineales** — el mismo cuello de botella que el deep learning acababa de eliminar en visión (AlexNet, 2012). La pregunta natural era si las técnicas de deep learning servían también para RL sobre datos sensoriales. El paper enumera con precisión por qué no era trivial:

1. **Recompensas escasas, ruidosas y retardadas.** El deep learning supervisado se entrena con datos etiquetados; RL aprende de una señal escalar de recompensa que puede llegar **miles de pasos** después de la acción que la provocó (el problema de la *asignación de crédito temporal*).
2. **Muestras no i.i.d.** Los algoritmos de deep learning asumen muestras independientes; en RL los frames consecutivos de un juego están **fuertemente correlacionados**.
3. **Distribución no estacionaria.** En RL la distribución de datos **cambia a medida que el agente aprende** nuevos comportamientos, mientras el deep learning asume una distribución fija.

A esto se suma una advertencia teórica: ya se había demostrado que combinar **Q-learning** (model-free) con **aproximadores no-lineales** y **aprendizaje off-policy** puede hacer divergir la red. Esa combinación —aproximación de función + bootstrapping + off-policy— es lo que la comunidad posterior bautizó como la **tríada mortal** (*deadly triad*). El gran logro empírico de DQN es **funcionar de todos modos**, sin garantías de convergencia, gracias al experience replay. El precedente inspirador es **TD-Gammon** (Tesauro, 1995), cuyo éxito en backgammon no se había logrado replicar en otros juegos; DQN es su reivindicación veinte años después, ahora con CNNs.

## Del Q-learning tabular a la Q-network

El agente interactúa con el emulador de Atari recibiendo en cada paso una imagen $x_t$ (pixeles de la pantalla) y una recompensa $r_t$ (cambio en el puntaje). Como una sola pantalla no basta para entender la situación (no se ve velocidad ni dirección), el estado se define como la **secuencia** de imágenes y acciones recientes. El objetivo es maximizar la **recompensa futura descontada** $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$. La función de valor de acción óptima obedece la **ecuación de Bellman**:

$$Q^*(s,a) = \mathbb{E}_{s' \sim E}\left[ r + \gamma \max_{a'} Q^*(s',a') \,\middle|\, s,a \right]$$

Muchos algoritmos de RL estiman $Q$ usando esta ecuación como actualización iterativa. **Pero estimar un valor por secuencia, sin generalización, es impráctico**: hay $256^{210 \times 160}$ pantallas posibles en Atari y una tabla es inconcebible. Este es exactamente el muro contra el que choca el **Q-learning tabular** que la [Clase 31](/clases/clase-31) enseña primero, que funciona en gridworlds de pocos estados. La solución es un **aproximador de función** $Q(s,a;\theta) \approx Q^*(s,a)$ — aquí una **red neuronal** que el paper llama **Q-network**, entrenada minimizando el error cuadrático contra el target $y_i = r + \gamma \max_{a'} Q(s',a';\theta_{i-1})$:

$$L_i(\theta_i) = \mathbb{E}_{s,a \sim \rho(\cdot)}\left[ (y_i - Q(s,a;\theta_i))^2 \right]$$

Un detalle crucial: **los targets dependen de los propios pesos de la red**, en contraste con el aprendizaje supervisado donde son fijos. El método es **model-free** (aprende directo de muestras, sin modelar el entorno) y **off-policy** (aprende sobre la política greedy mientras sigue una política exploratoria $\epsilon$-greedy: acción greedy con probabilidad $1-\epsilon$, aleatoria con probabilidad $\epsilon$).

## Contribución central: Experience Replay

En lugar de aprender de transiciones consecutivas, el agente almacena cada experiencia $e_t = (s_t, a_t, r_t, s_{t+1})$ en una **memoria de replay** $D$ y, en cada paso, muestrea un **minibatch aleatorio** de ese pool para actualizar los pesos. El paper enumera las tres ventajas que estabilizan el entrenamiento:

1. **Eficiencia de datos.** Cada transición se reutiliza en **muchas actualizaciones**, no se descarta tras un único uso — valioso cuando recolectar datos del emulador es caro.
2. **Rompe correlaciones, reduce varianza.** Aprender de muestras consecutivas es ineficiente por sus fuertes correlaciones; **aleatorizar las muestras las rompe y reduce la varianza** de las actualizaciones. Ataca directamente el problema #2 del contexto.
3. **Evita lazos de retroalimentación divergentes.** Al aprender on-policy, los parámetros actuales determinan la siguiente muestra: si la acción maximizante es "mover a la izquierda", el entrenamiento se domina por ese lado; al cambiar a "derecha", la distribución cambia con él, generando oscilaciones o **divergencia catastrófica**. El replay promedia el comportamiento sobre muchos estados previos, suavizando el aprendizaje.

El paper nota una sutileza: aprender con replay **obliga a ser off-policy** (los parámetros actuales difieren de los que generaron las muestras almacenadas), lo que **motiva elegir Q-learning** en vez de SARSA (on-policy). El **Algoritmo 1 (Deep Q-learning with Experience Replay)** —el que el [Lab 31](/laboratorios/lab-31) implementa— es:

```
Inicializar memoria de replay D con capacidad N
Inicializar Q con pesos aleatorios
para episodio = 1, M:
    inicializar secuencia s_1 y preprocesar φ_1 = φ(s_1)
    para t = 1, T:
        con probabilidad ε  →  acción aleatoria a_t
        en otro caso        →  a_t = max_a Q(φ(s_t), a; θ)
        ejecutar a_t; observar r_t y la imagen x_{t+1}
        almacenar (φ_t, a_t, r_t, φ_{t+1}) en D
        muestrear minibatch aleatorio (φ_j, a_j, r_j, φ_{j+1}) de D
        y_j = r_j                                  si φ_{j+1} es terminal
        y_j = r_j + γ max_{a'} Q(φ_{j+1}, a'; θ)    si no es terminal
        paso de descenso de gradiente sobre (y_j − Q(φ_j, a_j; θ))²
```

En la práctica $D$ guarda solo las últimas $N$ tuplas y muestrea uniformemente. El paper reconoce que esto es limitado —el buffer no prioriza transiciones informativas— anticipando el *prioritized experience replay* posterior.

## Método: preprocesamiento y arquitectura

Los frames crudos (210×160 pixeles, 128 colores) se reducen con la función $\phi$: conversión a **escala de grises**, downsampling a 110×84, recorte de **84×84** sobre el área de juego, y **apilado de los últimos 4 frames**. El stack de 4 frames resuelve la **observabilidad parcial** (un frame no revela velocidad; cuatro sí). La entrada final es un tensor **84×84×4**.

Una decisión elegante: en vez de pasar el par (estado, acción) por la red, DQN usa **una salida por acción** y solo el estado como entrada, de modo que **un único forward pass calcula los Q-values de todas las acciones**. La arquitectura, idéntica para los siete juegos:

- **Conv 1:** 16 filtros 8×8, stride 4 + ReLU.
- **Conv 2:** 32 filtros 4×4, stride 2 + ReLU.
- **FC oculta:** 256 unidades ReLU.
- **Salida:** capa lineal con una salida por acción válida (4 a 18 según el juego).

A estas CNNs las llaman **Deep Q-Networks (DQN)**. Hiperparámetros compartidos: **reward clipping** (toda recompensa positiva a +1, negativa a −1, para usar la misma tasa de aprendizaje en todos los juegos), **RMSProp** con minibatch 32, $\epsilon$ annealado de 1 a 0.1 sobre el primer millón de frames, **10 millones de frames** de entrenamiento, replay de **1 millón** de frames recientes y **frame-skipping** de $k=4$ ($k=3$ solo en Space Invaders, donde $k=4$ hacía invisibles los láseres — la única diferencia entre juegos).

## Resultados

Los siete juegos: **Beam Rider, Breakout, Enduro, Pong, Q\*bert, Seaquest, Space Invaders**. Evaluar el progreso en RL es difícil porque la recompensa por episodio es muy ruidosa; una métrica más estable es el **Q estimado promedio** sobre un conjunto fijo de estados, que sube de forma suave. Crucialmente, **no observaron divergencia en ningún experimento** — la respuesta empírica a la tríada mortal.

DQN se compara con **Sarsa** y **Contingency** (políticas lineales sobre features diseñadas a mano, con conocimiento previo como sustracción de fondo), un **humano experto** y la búsqueda evolutiva **HNeat**. DQN solo recibe pixeles crudos.

| Método | B. Rider | Breakout | Enduro | Pong | Q\*bert | Seaquest | S. Invaders |
|---|---|---|---|---|---|---|---|
| Random | 354 | 1.2 | 0 | −20.4 | 157 | 110 | 179 |
| Sarsa | 996 | 5.2 | 129 | −19 | 614 | 665 | 271 |
| Contingency | 1743 | 6 | 159 | −17 | 960 | 723 | 268 |
| **DQN** | **4092** | **168** | **470** | **20** | **1952** | **1705** | **581** |
| Human | 7456 | 31 | 368 | −3 | 18900 | 28010 | 3690 |
| **DQN Best** | 5184 | 225 | 661 | 21 | 4500 | 1740 | 1075 |

**DQN supera a Sarsa y Contingency en los siete juegos** pese a no usar casi conocimiento previo, y **supera al experto humano en Breakout, Enduro y Pong**. En Q\*bert, Seaquest y Space Invaders queda lejos del humano porque exigen estrategias sobre **escalas temporales largas** — el talón de Aquiles del horizonte de descuento y la asignación de crédito. La función de valor aprendida es interpretable: en Seaquest el valor predicho salta cuando aparece un enemigo y baja cuando desaparece.

## Limitaciones

- **No hay red objetivo (target network).** En esta versión 2013, los targets se calculan con la **misma red** que se actualiza, dejando un objetivo móvil que persigue a su propia predicción. La solución —una red objetivo separada copiada periódicamente— llega en la **[Nature 2015](/papers/dqn-nature-mnih-2015)**, no aquí. Que DQN-2013 funcione sin ella se debe en gran medida al experience replay.
- **Reward clipping borra magnitudes.** Fijar todo a ±1 impide distinguir recompensas pequeñas de grandes.
- **Replay uniforme y finito.** El buffer no prioriza transiciones informativas; el paper mismo señala el camino al *prioritized replay*.
- **Memoria de corto plazo.** El stack de 4 frames captura dinámica reciente pero no dependencias de largo horizonte, de ahí el bajo rendimiento en juegos de estrategia larga.
- **Sin garantías de convergencia.** Es un resultado puramente empírico.

## Impacto: el nacimiento del deep RL

Este paper **inauguró el deep reinforcement learning**. Demostró por primera vez que una sola arquitectura, sin ingeniería específica por tarea, podía aprender control de nivel humano (y superarlo) directamente desde pixeles. La receta —CNN como aproximador de $Q$, experience replay para estabilizar, $\epsilon$-greedy para explorar, reward clipping para uniformar— se volvió el patrón base de casi todo lo posterior: la **[Nature 2015](/papers/dqn-nature-mnih-2015)** (target network, 49 juegos), **Double DQN** (corrige el sesgo del operador max), **Dueling DQN**, **Prioritized Experience Replay**, **Rainbow** y la línea que culmina en **AlphaGo / AlphaZero**. El experience replay sigue siendo ingrediente estándar de los algoritmos off-policy modernos.

## Por qué importa para la Clase 31

La [Clase 31 (Aprendizaje Reforzado)](/clases/clase-31) sigue exactamente el arco histórico de este paper:

- **De Q-Learning tabular a DQN.** La clase enseña primero el [Q-Learning](/papers/q-learning-watkins-1992) tabular de Watkins & Dayan (1992): una tabla $Q[s][a]$ actualizada con la regla TD. Este paper muestra **por qué la tabla no escala** y **cuál es el salto**: reemplazarla por una red neuronal que generaliza entre estados similares. El mismo target de TD, el mismo $\max$ de Bellman, la misma naturaleza off-policy — solo cambia el "almacén" de la $Q$.
- **El [Laboratorio 31](/laboratorios/lab-31)** (Tutorial DQN) implementa directamente el Algoritmo 1: buffer de replay, red estado → Q-values por acción, bucle $\epsilon$-greedy, muestreo de minibatches, cálculo del target con su caso terminal y descenso de gradiente sobre el error TD. Cada línea del pseudocódigo tiene su contraparte en el código.
- **Por qué el lab usa replay.** Las tres ventajas (romper correlaciones, reutilizar datos, evitar divergencia) son la justificación pedagógica de no entrenar online. Si el lab usa una variante con **target network** (común en tutoriales modernos), conviene aclarar que esa pieza viene de la **Nature 2015**, no de este paper de 2013.

## Notas y enlaces

- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado)
- Clase: [/clases/clase-31](/clases/clase-31) · Laboratorio: [/laboratorios/lab-31](/laboratorios/lab-31)
- Paper antecesor (Q-learning tabular): [/papers/q-learning-watkins-1992](/papers/q-learning-watkins-1992)
- Paper sucesor (red objetivo, nivel humano): [/papers/dqn-nature-mnih-2015](/papers/dqn-nature-mnih-2015)
- arXiv: [1312.5602](https://arxiv.org/abs/1312.5602) · DeepMind Technologies, Londres
