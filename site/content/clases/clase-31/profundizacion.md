---
title: "Profundizacion - Aprendizaje Reforzado"
weight: 20
math: true
---

> Esta pagina complementa la [teoria de la clase 31](/clases/clase-31/teoria) con las derivaciones formales del aprendizaje reforzado. Seis bloques: **Parte I** formaliza el problema de control como un proceso de decision de Markov y justifica el descuento. **Parte II** deriva las ecuaciones de Bellman y prueba por que la iteracion de valor converge (el operador de Bellman es una contraccion). **Parte III** obtiene la regla de Q-Learning como aproximacion estocastica del punto fijo de Bellman y discute su convergencia. **Parte IV** desarrolla la matematica de DQN —la perdida, la red objetivo, el experience replay, la triada mortal— y sus tres mejoras canonicas (Double DQN, Dueling, PER). **Parte V** deriva el teorema del gradiente de politica, REINFORCE, la funcion de ventaja (actor-critic) y el objetivo recortado de PPO. **Parte VI** cierra con la integracion de redes y busqueda en AlphaGo.

---

## Parte I — El proceso de decision de Markov y el retorno

### I.1 Definicion formal del MDP

El aprendizaje reforzado modela la interaccion agente-ambiente como un **proceso de decision de Markov** (MDP), una tupla $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ donde:

- $\mathcal{S}$ es el conjunto de **estados**;
- $\mathcal{A}$ es el conjunto de **acciones** (eventualmente $\mathcal{A}(s)$ por estado);
- $P(s' \mid s, a)$ es la **dinamica de transicion**: la probabilidad de pasar al estado $s'$ tras ejecutar $a$ en $s$;
- $R(s, a)$ es la **recompensa esperada** inmediata, $R(s,a) = \mathbb{E}[r_{t+1} \mid s_t = s, a_t = a]$;
- $\gamma \in [0, 1]$ es el **factor de descuento**.

El adjetivo *de Markov* nombra el supuesto central: la dinamica depende **solo** del estado y la accion actuales, no de la historia completa,

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t).
$$

Esta propiedad es lo que vuelve tratable todo el problema: basta con conocer el estado presente para predecir el futuro. En DQN, donde un frame aislado de Atari no determina la velocidad de la pelota, el "estado" se construye apilando cuatro frames consecutivos precisamente para *recuperar* la propiedad de Markov (ver [DQN](/papers/dqn-nature-mnih-2015)).

Una **politica** $\pi(a \mid s)$ es una distribucion sobre acciones dado el estado. Si es determinista escribimos $a = \pi(s)$. El objetivo del agente es hallar una politica que maximice la recompensa acumulada.

### I.2 El retorno descontado

La cantidad que el agente maximiza no es la recompensa inmediata sino el **retorno**, la suma descontada de recompensas futuras desde el paso $t$:

$$
G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k+1}.
$$

El retorno admite una forma **recursiva** que sera la semilla de todas las ecuaciones de Bellman:

$$
G_t = r_{t+1} + \gamma\, G_{t+1}.
$$

Es decir, el retorno de hoy es la recompensa de manana mas el retorno descontado de pasado-manana en adelante. Esta autosimilitud es lo que permite el *bootstrapping*: estimar un valor a partir de otra estimacion.

### I.3 Por que descontar: el papel de $\gamma$

El factor $\gamma$ cumple tres funciones simultaneas, no una sola.

**(1) Convergencia matematica.** Si las recompensas estan acotadas, $|r| \le R_{\max}$, entonces con $\gamma < 1$ la serie geometrica converge:

$$
|G_t| \le \sum_{k=0}^{\infty} \gamma^k R_{\max} = \frac{R_{\max}}{1-\gamma} < \infty.
$$

Sin descuento ($\gamma = 1$) en una tarea continua (sin estados terminales), el retorno puede divergir y el problema de optimizacion deja de estar bien definido. El paper de [Watkins y Dayan](/papers/q-learning-watkins-1992) hace de esto un lema explicito: el descuento garantiza que la "cola" de recompensas lejanas sea despreciable, lo que sostiene la prueba de convergencia (su Lema B.1).

**(2) Preferencia temporal.** Una recompensa que llega $k$ pasos en el futuro vale $\gamma^k$ de lo que valdria ahora. Con $\gamma = 0.99$, una recompensa a 100 pasos vale $0.99^{100} \approx 0.37$; a 500 pasos, $\approx 0.007$. El agente prefiere lo proximo a lo lejano, lo que codifica incertidumbre sobre el futuro distante.

**(3) Horizonte efectivo.** El descuento induce un horizonte temporal caracteristico $\approx 1/(1-\gamma)$. Con $\gamma = 0.99$ el agente "ve" unos 100 pasos hacia adelante. Subir $\gamma$ alarga el horizonte pero endurece el problema de credito (atribuir una recompensa lejana a las acciones que la causaron). En tareas con recompensa dispersa y diferida —como Montezuma's Revenge, donde DQN fracasa— el horizonte efectivo es demasiado corto para conectar acciones tempranas con su pago lejano.

El caso $\gamma = 1$ **si** es valido cuando existen **estados absorbentes** que terminan el episodio con certeza: la terminacion garantizada juega el papel acotante que jugaba $\gamma < 1$.

---

## Parte II — Las ecuaciones de Bellman

### II.1 Funciones de valor

Dadas una politica $\pi$, definimos dos funciones de valor. La **funcion de valor de estado** es el retorno esperado partiendo de $s$ y siguiendo $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[ G_t \mid s_t = s \right].
$$

La **funcion de valor-accion** (o *Q-value*) es el retorno esperado partiendo de $s$, tomando $a$, y siguiendo $\pi$ despues:

$$
Q^\pi(s, a) = \mathbb{E}_\pi\!\left[ G_t \mid s_t = s,\, a_t = a \right].
$$

Ambas se relacionan: $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s,a)]$, y la **ventaja** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ mide cuanto mejor que el promedio es la accion $a$ en $s$ (con $\mathbb{E}_{a\sim\pi}[A^\pi(s,a)] = 0$). Esta descomposicion sera el corazon de Dueling DQN (Parte IV) y de actor-critic (Parte V).

### II.2 Bellman expectation: la consistencia de $V^\pi$ y $Q^\pi$

Sustituyendo la recursion $G_t = r_{t+1} + \gamma G_{t+1}$ dentro de la esperanza obtenemos la **ecuacion de expectativa de Bellman**, que expresa el valor de un estado en terminos del valor de sus sucesores:

$$
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\Big[ R(s,a) + \gamma\, V^\pi(s') \Big].
$$

La version para la funcion-accion:

$$
Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a)\Big[ R(s,a) + \gamma \sum_{a'} \pi(a' \mid s')\, Q^\pi(s', a') \Big].
$$

Aunque parezca circular —el valor se define en terminos de si mismo— el sistema esta bien definido: es un conjunto de $|\mathcal{S}|$ ecuaciones lineales en las $|\mathcal{S}|$ incognitas $V^\pi(s)$, con solucion unica cuando $\gamma < 1$.

### II.3 Bellman optimality: $V^*$ y $Q^*$

La politica optima $\pi^*$ es la que maximiza el valor en todo estado. Sus funciones de valor optimas $V^*$ y $Q^*$ satisfacen las **ecuaciones de optimalidad de Bellman**, donde la suma sobre la politica se reemplaza por un $\max$:

$$
V^*(s) = \max_{a} \sum_{s'} P(s' \mid s, a)\Big[ R(s,a) + \gamma\, V^*(s') \Big],
$$

$$
\boxed{\;Q^*(s, a) = \sum_{s'} P(s' \mid s, a)\Big[ R(s,a) + \gamma \max_{a'} Q^*(s', a') \Big]\;}
$$

La utilidad de $Q^*$ es directa: una vez conocida, la **politica optima es greedy** sobre ella, $\pi^*(s) = \arg\max_a Q^*(s,a)$. No hace falta el modelo $P$ ni $R$ para actuar: basta el argmax. Aqui reside todo el atractivo de aprender $Q$ en vez de $V$ —y es la razon por la que Q-Learning puede ser *model-free*. La relacion entre ambas optimas es $V^*(s) = \max_a Q^*(s,a)$.

### II.4 El operador de Bellman y la contraccion

Definimos el **operador de optimalidad de Bellman** $\mathcal{T}$, que toma una funcion de valor-accion $Q$ y devuelve otra:

$$
(\mathcal{T}Q)(s,a) = \sum_{s'} P(s' \mid s, a)\Big[ R(s,a) + \gamma \max_{a'} Q(s', a') \Big].
$$

Por construccion, $Q^*$ es el **punto fijo** de $\mathcal{T}$: $\mathcal{T}Q^* = Q^*$. La **iteracion de valor** simplemente aplica $\mathcal{T}$ repetidamente, $Q_{k+1} = \mathcal{T}Q_k$, partiendo de cualquier $Q_0$. Que esto converja a $Q^*$ no es obvio; lo garantiza el siguiente resultado.

**Teorema (contraccion).** El operador $\mathcal{T}$ es una **contraccion** de modulo $\gamma$ en la norma del supremo $\|\cdot\|_\infty$:

$$
\|\mathcal{T}Q_1 - \mathcal{T}Q_2\|_\infty \le \gamma\, \|Q_1 - Q_2\|_\infty.
$$

*Demostracion.* Para cualquier par $(s,a)$,

$$
\big|(\mathcal{T}Q_1)(s,a) - (\mathcal{T}Q_2)(s,a)\big|
= \gamma \left| \sum_{s'} P(s'\mid s,a)\Big[ \max_{a'} Q_1(s',a') - \max_{a'} Q_2(s',a') \Big] \right|.
$$

Usamos que el operador $\max$ es no-expansivo, $|\max_x f(x) - \max_x g(x)| \le \max_x |f(x) - g(x)|$, y que las probabilidades suman 1:

$$
\le \gamma \sum_{s'} P(s'\mid s,a)\, \max_{a'} \big| Q_1(s',a') - Q_2(s',a') \big|
\le \gamma\, \|Q_1 - Q_2\|_\infty.
$$

Como vale para todo $(s,a)$, tomando el supremo a la izquierda se obtiene la cota. $\blacksquare$

Por el **teorema del punto fijo de Banach**, una contraccion en un espacio metrico completo tiene un punto fijo unico al que toda iteracion converge geometricamente. Aqui ese punto fijo es $Q^*$, y la tasa de convergencia es:

$$
\|Q_k - Q^*\|_\infty \le \gamma^k\, \|Q_0 - Q^*\|_\infty \to 0.
$$

Cada barrido de iteracion de valor reduce el error por al menos un factor $\gamma$. **Este es el fundamento que justifica que value iteration converge** — y por que $\gamma < 1$ (modulo de contraccion estrictamente menor que 1) es esencial. La misma estructura de contraccion reaparece debilitada en el caso muestral de Q-Learning, donde el operador exacto $\mathcal{T}$ se reemplaza por una version estocastica.

---

## Parte III — Q-Learning como aproximacion estocastica

### III.1 De la iteracion de valor a la regla muestral

La iteracion de valor exacta exige conocer $P$ y $R$ para evaluar $\mathcal{T}Q$. El agente sin modelo no los tiene; solo observa transiciones $(s, a, r, s')$ muestreadas del ambiente. La idea de [Q-Learning](/papers/q-learning-watkins-1992) es reemplazar la esperanza sobre $s'$ por una **muestra unica** y actualizar incrementalmente.

Definamos el **objetivo muestral** de una transicion como un estimador insesgado de $(\mathcal{T}Q)(s,a)$:

$$
y = r + \gamma \max_{a'} Q(s', a'),
$$

donde $\mathbb{E}_{s' \sim P}[y] = (\mathcal{T}Q)(s,a)$. Queremos mover $Q(s,a)$ hacia su objetivo de Bellman $(\mathcal{T}Q)(s,a)$, pero solo disponemos de la muestra ruidosa $y$. La **aproximacion estocastica de Robbins-Monro** resuelve exactamente este problema: para hallar el punto fijo de un operador cuando solo se observan evaluaciones ruidosas, se itera

$$
Q(s,a) \leftarrow (1-\alpha)\, Q(s,a) + \alpha\, y,
$$

un promedio movil exponencial entre la estimacion vieja y el objetivo muestral. Reordenando se obtiene la forma canonica de la **regla de Q-Learning**:

$$
\boxed{\;Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\, \underbrace{r + \gamma \max_{a'} Q(s',a') - Q(s,a)}_{\text{error TD } \delta}\,\big]\;}
$$

El termino entre corchetes es el **error de diferencia temporal** (TD error) $\delta$: la discrepancia entre la estimacion actual y un objetivo mejor informado construido con la recompensa real mas el valor del mejor sucesor. Si $\delta = 0$ para toda transicion, $Q$ satisface la ecuacion de optimalidad de Bellman; cada actualizacion empuja $Q$ hacia su punto fijo $Q^*$.

### III.2 Off-policy: aprender lo optimo mientras se explora

El operador $\max_{a'}$ es lo que hace a Q-Learning **off-policy**. El objetivo $y$ usa $\max_{a'} Q(s', a')$ —el valor de la *politica greedy*— con independencia de la accion que realmente se tomara en $s'$ o de la politica de comportamiento que genero la transicion. Por eso el agente puede aprender la politica optima mientras se comporta de forma exploratoria y suboptima. Esto contrasta con SARSA (on-policy), que usa $Q(s', a')$ con la $a'$ efectivamente tomada,

$$
\text{SARSA:}\quad Q(s,a) \leftarrow Q(s,a) + \alpha\,[\, r + \gamma\, Q(s', a') - Q(s,a) \,].
$$

La separacion entre **politica de comportamiento** (la que genera datos) y **politica objetivo** (la que se evalua) es lo que permite, mas tarde, reutilizar transiciones viejas en el experience replay de DQN.

### III.3 Convergencia (Watkins y Dayan, 1992)

La garantia teorica de Q-Learning, probada por [Watkins y Dayan](/papers/q-learning-watkins-1992), es que en el caso **tabular** (una entrada por par $(s,a)$) las estimaciones convergen a los valores optimos con probabilidad 1:

$$
Q_n(s,a) \to Q^*(s,a) \quad \text{c.p. 1, para todo } (s,a),
$$

bajo dos condiciones. La primera, de **cobertura**: cada par $(s,a)$ debe visitarse infinitas veces. La segunda, sobre las **tasas de aprendizaje** $\alpha_n$, son las condiciones clasicas de Robbins-Monro:

$$
\sum_{n=1}^{\infty} \alpha_n = \infty, \qquad \sum_{n=1}^{\infty} \alpha_n^2 < \infty.
$$

La primera suma debe **diverger** para que el aprendizaje nunca se congele antes de tiempo y pueda alcanzar cualquier valor; la segunda debe **converger** para que el ruido de muestreo se promedie a cero y las estimaciones se estabilicen. Una tasa como $\alpha_n = 1/n$ las satisface; una tasa **constante** viola la segunda condicion y por eso, en la practica, solo da convergencia aproximada (que es justamente lo que se usa con redes neuronales).

La prueba original construye un proceso de Markov auxiliar (el *action-replay process*) y demuestra que sus valores optimos coinciden con los $Q_n$ y que converge al proceso real; es, en esencia, una version muestral del argumento de contraccion de la Parte II.

### III.4 Exploracion: la politica $\varepsilon$-greedy

La condicion de cobertura (visitar todo par $(s,a)$ infinitas veces) no es un detalle de implementacion: es lo que hace **valida la garantia de convergencia**. La forma estandar de cumplirla es la politica **$\varepsilon$-greedy**:

$$
\pi(a \mid s) =
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|} & \text{si } a = \arg\max_{a'} Q(s, a'), \\[2mm]
\dfrac{\varepsilon}{|\mathcal{A}|} & \text{en otro caso.}
\end{cases}
$$

Con probabilidad $1-\varepsilon$ el agente **explota** (toma el argmax de $Q$) y con probabilidad $\varepsilon$ **explora** (accion aleatoria uniforme). El balance exploracion/explotacion no es opcional: la exploracion garantiza la cobertura que el teorema exige, la explotacion aprovecha lo aprendido. En la practica $\varepsilon$ se **anela** (annealing) desde 1.0 hacia un valor pequeno (0.1 en DQN) a medida que las estimaciones maduran.

---

## Parte IV — Deep Q-Network (DQN)

### IV.1 La perdida de DQN

En espacios de estado masivos (imagenes) la tabla $Q$ es inviable. [DQN](/papers/dqn-nature-mnih-2015) la reemplaza por una red neuronal $Q(s,a;\theta)$ que aproxima $Q^*$. Entrenarla es una regresion del valor estimado hacia el objetivo de Bellman, minimizando el error TD cuadratico:

$$
\boxed{\;L(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)}\!\left[ \Big( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \Big)^2 \right]\;}
$$

cuyo gradiente, tratando el objetivo como constante respecto de $\theta$, es

$$
\nabla_\theta L(\theta) = \mathbb{E}\!\left[ \Big( r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta) \Big)\, \big(-\nabla_\theta Q(s,a;\theta)\big) \right].
$$

Dos detalles cruciales viven en esta ecuacion: la **red objetivo** $\theta^-$ (no $\theta$) en el target, y la **distribucion de muestreo** $U(D)$ uniforme sobre un buffer $D$. Cada uno resuelve una fuente de inestabilidad.

### IV.2 La red objetivo: un blanco no estacionario

El problema que ataca $\theta^-$ es que el objetivo de regresion **se mueve junto con lo que se predice**. En Q-Learning online ingenuo, una actualizacion que sube $Q(s_t, a_t)$ frecuentemente sube tambien $Q(s_{t+1}, a)$ para todo $a$, lo que sube el objetivo $y$ —el agente persigue un blanco que el mismo desplaza, generando oscilaciones o divergencia. La solucion de DQN es congelar una copia $\theta^-$ de la red y usarla **solo para calcular el objetivo**, sincronizandola con la red online cada $C$ pasos:

$$
\theta^- \leftarrow \theta \quad \text{cada } C \text{ pasos}.
$$

Esto introduce un **retraso** entre el momento en que se actualiza $Q$ y el momento en que esa actualizacion afecta los objetivos, haciendo la divergencia mucho menos probable. El target se vuelve aproximadamente estacionario entre sincronizaciones, recuperando algo parecido a la regresion supervisada con etiquetas fijas.

### IV.3 Experience replay: romper correlaciones

El gradiente estocastico supone muestras **i.i.d.**, pero las transiciones consecutivas de un episodio estan fuertemente correlacionadas (estados sucesivos se parecen) y su distribucion es no estacionaria (cambia con la politica). El **experience replay** almacena cada transicion $e_t = (s_t, a_t, r_t, s_{t+1})$ en un buffer $D$ (las ultimas $\sim 10^6$) y entrena con minibatches **muestreados uniformemente al azar** de $D$. Esto logra tres cosas:

1. **Rompe las correlaciones temporales**, aproximando el supuesto i.i.d. y reduciendo la varianza de las actualizaciones.
2. **Reutiliza cada transicion** muchas veces (eficiencia de datos: cada experiencia se revisa unas 8 veces en promedio).
3. **Promedia la distribucion de comportamiento** sobre muchos estados pasados, evitando bucles de retroalimentacion donde la politica actual sesga las muestras futuras.

El replay **exige** off-policy (las transiciones fueron generadas por politicas viejas), lo que motiva usar Q-Learning y no un metodo on-policy.

### IV.4 La triada mortal

DQN combina tres ingredientes que, juntos, pueden hacer divergir el aprendizaje —la llamada **triada mortal** (*deadly triad*):

| Ingrediente | Que aporta | Riesgo |
| --- | --- | --- |
| **Aproximacion de funcion** | Generalizar a estados no vistos | Errores se propagan a estados vecinos |
| **Bootstrapping** | Estimar a partir de estimaciones ($\max_{a'} Q$) | Errores se amplifican al realimentarse |
| **Off-policy** | Aprender de datos de otra politica (replay) | Distribucion de muestreo no coincide con la objetivo |

Cualquier par de los tres es manejable; los tres juntos rompen la garantia de contraccion de la Parte II (la version muestral con aproximacion ya no es necesariamente una contraccion) y pueden divergir. DQN no *elimina* la triada —vive con ella— sino que la **estabiliza empiricamente** con la red objetivo (atenua el bootstrapping inestable) y el replay (atenua la no estacionariedad off-policy). Las ablaciones del paper confirman que quitar cualquiera de los dos degrada drasticamente el desempeno.

### IV.5 Double DQN: corregir el sesgo de sobreestimacion

El operador $\max_{a'}$ del target tiene un sesgo sistematico **hacia arriba**. Reescribamoslo separando seleccion y evaluacion de la accion:

$$
\max_{a'} Q(s', a'; \theta^-) = Q\Big(s',\, \arg\max_{a'} Q(s', a'; \theta^-);\, \theta^-\Big).
$$

Los **mismos** pesos $\theta^-$ eligen la accion (argmax) **y** reportan su valor. Cuando las estimaciones tienen ruido —y siempre lo tienen durante el aprendizaje— el $\max$ se inclina hacia las acciones cuyo ruido las hace lucir mejores de lo que son, sobreestimando. El [Teorema 1 de van Hasselt et al.](/papers/double-dqn-van-hasselt-2015) cuantifica el sesgo: en un estado con $m$ acciones de igual valor verdadero $V_*$ y error cuadratico medio $C > 0$,

$$
\max_a Q_t(s,a) \ge V_*(s) + \sqrt{\frac{C}{m-1}},
$$

cota que se anula con el estimador desacoplado. **Double DQN** rompe el acoplamiento: **selecciona** la accion con la red online $\theta$ pero la **evalua** con la red objetivo $\theta^-$:

$$
\boxed{\;Y^{\text{DoubleDQN}} = r + \gamma\, Q\Big(s',\, \arg\max_{a'} Q(s', a'; \theta);\, \theta^-\Big)\;}
$$

Es, literalmente, mover el argmax de una red a la otra —un cambio de una linea, sin redes ni hiperparametros nuevos— que reduce la sobreestimacion y estabiliza el aprendizaje.

### IV.6 Dueling DQN: factorizar $V$ y $A$

[Dueling DQN](/papers/dueling-dqn-wang-2015) no cambia el algoritmo sino la **arquitectura**: divide la red en dos flujos tras el tronco convolucional, uno que estima el valor de estado $V(s;\theta,\beta)$ (un escalar) y otro la ventaja $A(s,a;\theta,\alpha)$ (un vector). La motivacion: en muchos estados la eleccion de accion es irrelevante (carretera despejada en Enduro), y lo que importa es $V(s)$, que ademas se propaga en cada actualizacion por bootstrapping.

La combinacion ingenua $Q = V + A$ es **no identificable**: sumar una constante a $V$ y restarla de $A$ deja $Q$ intacto, asi que $V$ y $A$ no convergen a valores con sentido. Se fuerza identificabilidad restando un agregado de la ventaja. La variante que realmente se usa resta la **media**:

$$
\boxed{\;Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + \Big( A(s,a;\theta,\alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a';\theta,\alpha) \Big)\;}
$$

Restar la media (en lugar del $\max$) sacrifica la semantica estricta de $V$ y $A$ —quedan desplazados por una constante— pero **mejora la estabilidad**: las ventajas solo necesitan cambiar tan rapido como su propia media. Como restar una constante no altera el ranking relativo de los $Q$, la politica greedy se preserva exactamente. La factorizacion deja a la red aprender *que estados valen* sin tener que aprender el efecto de cada accion en cada estado.

### IV.7 Prioritized Experience Replay (PER)

El replay uniforme repite cada transicion con la frecuencia con que ocurrio, sin importar cuanto ensena. [PER](/papers/per-schaul-2015) muestrea con probabilidad creciente en la **magnitud del error TD** $|\delta|$ —las transiciones "sorprendentes" se repiten mas. La probabilidad de muestrear la transicion $i$ es:

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}, \qquad p_i = |\delta_i| + \epsilon,
$$

donde $\alpha$ controla cuanta priorizacion se aplica ($\alpha=0$ recupera el caso uniforme, $\alpha=1$ priorizacion plena) y $\epsilon > 0$ evita que una transicion de error cero nunca vuelva a visitarse. Pero el muestreo no uniforme **introduce un sesgo**: las actualizaciones ya no provienen de la distribucion correcta, asi que la esperanza estimada cambia. Se corrige con **pesos de importance sampling**:

$$
w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^{\beta},
$$

que compensan completamente las probabilidades no uniformes cuando $\beta = 1$. El TD error de cada muestra se reemplaza por $w_i\, \delta_i$ en la actualizacion. El exponente $\beta$ se **anela** de $\beta_0$ hacia 1: la correccion insesgada importa mas cerca de la convergencia, mientras que al inicio el proceso ya es muy no estacionario y un pequeno sesgo es tolerable. Implementado con una *sum-tree*, muestrear y actualizar prioridades cuestan $O(\log N)$, con overhead de solo 2-4%.

Double DQN, Dueling y PER son **ortogonales** —corrigen el sesgo del max, reorganizan la representacion y mejoran el uso de datos, respectivamente— y se combinan; juntas con otras tres mejoras forman **Rainbow**.

---

## Parte V — Metodos de gradiente de politica

### V.1 El teorema del gradiente de politica

Los metodos *value-based* (Q-Learning, DQN) aprenden $Q$ y derivan la politica con un argmax —inviable en espacios de accion **continuos**. Los metodos *policy-based* parametrizan la politica $\pi_\theta(a \mid s)$ directamente y ascienden por el gradiente del retorno esperado:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ G_0 \right], \qquad \theta \leftarrow \theta + \eta\, \nabla_\theta J(\theta).
$$

El problema es que la distribucion de trayectorias $\tau$ depende de $\theta$, asi que no se puede derivar la esperanza directamente. El **teorema del gradiente de politica** resuelve esto con el *log-derivative trick*. Para una trayectoria, $\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)$, y como $\log p_\theta(\tau) = \sum_t \log \pi_\theta(a_t \mid s_t) + \text{(terminos de dinamica sin }\theta)$, la dinamica $P$ desaparece al derivar. El resultado es:

$$
\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(a \mid s)\, Q^{\pi_\theta}(s, a) \right]\;}
$$

La interpretacion es elegante: sube la probabilidad logaritmica de las acciones ponderada por **cuan buenas fueron** ($Q^{\pi_\theta}$). No requiere conocer ni derivar la dinamica del ambiente.

### V.2 REINFORCE

[REINFORCE](/papers/a3c-mnih-2016) (Williams, 1992) es la version Monte Carlo: reemplaza $Q^{\pi_\theta}(s_t, a_t)$ por el retorno muestral $G_t$ de la trayectoria,

$$
\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t.
$$

Es un estimador **insesgado** del gradiente, pero de **varianza alta**: $G_t$ acumula la aleatoriedad de toda la trayectoria. AlphaGo usa exactamente esta regla en su refinamiento por self-play: $\Delta\rho \propto \nabla_\rho \log p_\rho(a_t \mid s_t)\, z_t$, con $z_t = \pm 1$ el resultado terminal de la partida.

### V.3 Baseline y la funcion de ventaja

La varianza se reduce —**sin introducir sesgo**— restando una *baseline* $b(s)$ que no depende de la accion:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(a \mid s)\, \big( Q^{\pi_\theta}(s,a) - b(s) \big) \right].
$$

El termino extra se anula en esperanza porque $\mathbb{E}_{a}[\nabla_\theta \log \pi_\theta(a\mid s)\, b(s)] = b(s)\, \nabla_\theta \sum_a \pi_\theta(a\mid s) = b(s)\, \nabla_\theta 1 = 0$. La baseline optima en varianza es aproximadamente $V^{\pi_\theta}(s)$, y al usarla la cantidad que pondera el gradiente se vuelve la **funcion de ventaja**:

$$
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).
$$

Esto da la arquitectura **actor-critic**: la politica $\pi_\theta$ es el **actor** (decide), y una funcion de valor aprendida $V$ es el **critico** (juzga cuanto mejor que el promedio fue la accion). [A3C](/papers/a3c-mnih-2016) estima la ventaja con *n-step returns*,

$$
A(s_t, a_t) = \sum_{i=0}^{k-1} \gamma^i\, r_{t+i} + \gamma^k V(s_{t+k}; \theta_v) - V(s_t; \theta_v),
$$

ejecuta **multiples workers en paralelo** (cuya diversidad descorrelaciona los datos, reemplazando al experience replay y reabriendo la puerta a metodos on-policy) y agrega un **bono de entropia** $\beta\, H(\pi_\theta(\cdot \mid s))$ al objetivo para desincentivar la convergencia prematura a politicas deterministas:

$$
\nabla_{\theta} \log \pi_\theta(a_t \mid s_t)\, A(s_t, a_t) + \beta\, \nabla_\theta H\big(\pi_\theta(\cdot \mid s_t)\big).
$$

### V.4 PPO: el objetivo recortado

El gradiente de politica solo es valido **localmente**, alrededor de la politica que genero los datos; insistir con varios pasos de SGD sobre el mismo lote empuja la politica fuera de esa region y la colapsa. TRPO controla esto con una restriccion dura de divergencia KL, pero exige optimizacion de segundo orden. [PPO](/papers/ppo-schulman-2017) logra lo mismo con primer orden y un **recorte**. Sea el **ratio de probabilidades** entre la politica nueva y la vieja:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}, \qquad r_t(\theta_{old}) = 1.
$$

El objetivo *surrogate* sin proteccion seria $L^{CPI} = \hat{\mathbb{E}}_t[r_t(\theta)\, \hat{A}_t]$, que maximizado lleva a pasos excesivos. PPO propone:

$$
\boxed{\;L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\!\left[ \min\!\Big( r_t(\theta)\, \hat{A}_t,\;\; \text{clip}\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\, \hat{A}_t \Big) \right]\;}
$$

con $\epsilon \approx 0.2$. La mecanica: el segundo argumento del $\min$ recorta el ratio a la banda $[1-\epsilon, 1+\epsilon]$, eliminando el incentivo a moverlo fuera de ella; tomar el $\min$ vuelve el objetivo una **cota inferior pesimista** del objetivo sin recortar. Para ventaja positiva ($\hat{A}_t > 0$, accion buena) el objetivo crece con $r_t$ pero **se aplana** cuando $r_t > 1+\epsilon$ —ya no premia subir mas la probabilidad; para ventaja negativa se aplana cuando $r_t < 1-\epsilon$. En ambos casos el aplanamiento le quita al optimizador el incentivo de dar pasos enormes, **evitando las actualizaciones destructivas** del gradiente vanilla. Esto habilita varias epocas de SGD sobre el mismo lote, mejorando la eficiencia de muestras. El nombre *proximal* viene de que la nueva politica se mantiene "proxima" a la vieja, no por una restriccion explicita, sino porque el objetivo deja de premiar el alejamiento.

La misma idea de "no alejarse demasiado de la politica previa" reaparece, una decada despues, como el termino KL central de [RLHF](/fundamentos/rlhf): PPO ajusta el modelo de lenguaje para maximizar una recompensa aprendida de preferencias humanas, manteniendolo proximo al modelo de referencia para que no "se rompa".

---

## Parte VI — AlphaGo: redes guiando la busqueda

[AlphaGo](/papers/alphago-silver-2016) integra los tres conceptos de las partes anteriores —politica, valor y gradiente de politica— en un sistema que vencio a un profesional humano de Go, un espacio de busqueda de $\sim b^d \approx 250^{150}$ intratable por fuerza bruta. La clave es **reducir profundidad y amplitud** con aprendizaje profundo.

**Pipeline de entrenamiento.** (1) Una *policy network* supervisada $p_\sigma(a\mid s)$ aprende a imitar jugadas humanas por ascenso de verosimilitud, $\Delta\sigma \propto \nabla_\sigma \log p_\sigma(a\mid s)$. (2) Una *policy network* por RL $p_\rho$, inicializada con $p_\sigma$, se refina por **self-play** contra versiones pasadas de si misma usando REINFORCE con recompensa terminal $z_t = \pm 1$: $\Delta\rho \propto \nabla_\rho \log p_\rho(a_t\mid s_t)\, z_t$. (3) Una *value network* $v_\theta(s)$ aprende por regresion a predecir el resultado de partidas de $p_\rho$, minimizando $(z - v_\theta(s))^2$ —con la advertencia de que entrenar con posiciones correlacionadas de una misma partida provoca sobreajuste (se resuelve usando una posicion por partida distinta).

**Busqueda (MCTS guiado).** Cada simulacion del *Monte Carlo Tree Search* desciende eligiendo la accion que maximiza valor mas un bono de exploracion proporcional al prior de la policy network:

$$
a_t = \arg\max_a \Big( Q(s_t, a) + u(s_t, a) \Big), \qquad u(s,a) \propto \frac{P(s,a)}{1 + N(s,a)},
$$

donde $P(s,a) = p_\sigma(a\mid s)$ y $N$ es el conteo de visitas. El bono favorece jugadas de alto prior y pocas visitas al inicio, y asintoticamente las de alto valor (variante PUCT). Las hojas se evaluan combinando la value network con un rollout rapido $z_L$:

$$
V(s_L) = (1-\lambda)\, v_\theta(s_L) + \lambda\, z_L,
$$

resultando $\lambda = 0.5$ la mezcla optima (value network y rollouts son complementarios). La policy network reduce la **amplitud** (no considera todas las jugadas), la value network reduce la **profundidad** (no juega hasta el final), y MCTS las orquesta. AlphaGo evaluo miles de veces menos posiciones que Deep Blue, compensando con posiciones elegidas mas inteligentemente y evaluadas con mas precision —un enfoque mas cercano a como juega un humano.

---

## Sintesis matematica

| Concepto | Ecuacion central |
| --- | --- |
| Retorno descontado | $G_t = \sum_k \gamma^k r_{t+k+1} = r_{t+1} + \gamma G_{t+1}$ |
| Bellman optimality | $Q^*(s,a) = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q^*(s',a')]$ |
| Contraccion | $\|\mathcal{T}Q_1 - \mathcal{T}Q_2\|_\infty \le \gamma\, \|Q_1 - Q_2\|_\infty$ |
| Q-Learning | $Q(s,a) \leftarrow Q(s,a) + \alpha\,[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| Perdida DQN | $L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2]$ |
| Double DQN | $Y = r + \gamma\, Q(s', \arg\max_{a'} Q(s',a';\theta);\, \theta^-)$ |
| Dueling | $Q = V + (A - \tfrac{1}{|\mathcal{A}|}\sum_{a'} A)$ |
| PER | $P(i) \propto p_i^\alpha$, correccion $w_i = (N\,P(i))^{-\beta}$ |
| Gradiente de politica | $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a\mid s)\, A^\pi(s,a)]$ |
| PPO clipped | $L^{CLIP} = \hat{\mathbb{E}}_t[\min(r_t \hat{A}_t,\, \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t)]$ |

El hilo conductor: la **ecuacion de optimalidad de Bellman** define el objetivo; su **operador de contraccion** garantiza que la iteracion converge; **Q-Learning** la aproxima por muestreo; **DQN** la escala con redes pagando el precio de la triada mortal (estabilizada con red objetivo y replay); el **gradiente de politica** ataca el problema desde el otro lado, optimizando la politica directamente con la ventaja como senal; y **AlphaGo** integra politica, valor y busqueda. Una ecuacion de punto fijo, dos grandes familias de algoritmos.

---

**Ver tambien:** [Teoria de la clase 31](/clases/clase-31/teoria) · [Practica desde 0](/clases/clase-31/practica) · Fundamentos: [Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado) · [RLHF](/fundamentos/rlhf) · Papers: [Q-Learning](/papers/q-learning-watkins-1992) · [DQN](/papers/dqn-nature-mnih-2015) · [Double DQN](/papers/double-dqn-van-hasselt-2015) · [Dueling DQN](/papers/dueling-dqn-wang-2015) · [PER](/papers/per-schaul-2015) · [A3C](/papers/a3c-mnih-2016) · [PPO](/papers/ppo-schulman-2017) · [AlphaGo](/papers/alphago-silver-2016).
