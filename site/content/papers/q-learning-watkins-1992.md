---
title: "Q-Learning (1992)"
weight: 347
math: true
---

{{< paper-card
    title="Q-learning"
    authors="Christopher J.C.H. Watkins, Peter Dayan"
    year="1992"
    venue="Machine Learning 8"
    pdf="/papers/q-learning-watkins-1992.pdf" >}}
Technical Note que **demuestra correcto** uno de los algoritmos más influyentes del aprendizaje por refuerzo. Q-learning no se inventa aquí —eso fue la tesis de Watkins de 1989— sino que este paper aporta su contribución decisiva: la **prueba de que converge a los valores-acción óptimos $Q^*$ con probabilidad 1**, una garantía que "había eludido a la mayoría de los métodos de RL". El algoritmo es *model-free* (no necesita conocer recompensas ni transiciones) y *off-policy* (aprende la política óptima mientras explora con otra), aprendiendo solo por ensayo y error mediante una regla de actualización incremental. Fundamento de la [Clase 31](/clases/clase-31) y núcleo directo de [Deep Q-Learning](/papers/dqn-atari-mnih-2013).
{{< /paper-card >}}

---

## Contexto

Hacia 1992, el aprendizaje por refuerzo (RL) reunía varias tradiciones que apenas empezaban a unificarse. Q-learning las anuda en cuatro hilos.

**Procesos de decisión de Markov (MDP).** Un agente se mueve por un mundo discreto y finito: en cada paso registra el estado $x_n$, elige una acción $a_n$ de un conjunto finito, recibe una recompensa probabilística $r_n$ con media $\mathcal{R}_x(a)$, y el mundo transita al estado $y_n$ según una ley $P_{xy}[a]$. La **propiedad de Markov** —el futuro depende solo del estado y acción actuales, no de la historia— es lo que hace tratable todo el problema.

**Programación dinámica de Bellman.** La meta del agente es una *política óptima* que maximice la recompensa total **descontada** esperada: una recompensa recibida $s$ pasos en el futuro vale $\gamma^s$ veces menos, con $0 \le \gamma < 1$. La teoría de programación dinámica (DP) garantiza que existe una política estacionaria óptima $\pi^*$ y da métodos para calcular el valor óptimo $V^*$ —pero **suponiendo que $\mathcal{R}_x(a)$ y $P_{xy}[a]$ son conocidas**.

**El problema sin modelo.** La dificultad central de Q-learning es hallar $\pi^*$ **sin conocer** las recompensas ni las transiciones. Métodos previos aprendían el modelo del ambiente y corrían DP en paralelo, pero los supuestos de "equivalencia de certeza" salen caros en las etapas tempranas del aprendizaje. Watkins clasifica Q-learning como **programación dinámica incremental**.

**Diferencias temporales (TD).** El aprendizaje procede como el método TD de Sutton: el agente prueba una acción, evalúa la recompensa inmediata *y su estimación del valor del estado al que llega*. Estimar a partir de estimaciones —el *bootstrapping*— es la herencia directa de TD.

## Contribución: aprender $Q^*$ directamente

La idea clave es desplazar el objeto de aprendizaje desde la función de valor de estado $V$ hacia una **función de valor-acción** $Q$. El valor $Q^\pi(x,a)$ es la recompensa descontada esperada por ejecutar la acción $a$ en el estado $x$ y *luego* seguir la política $\pi$. Los valores óptimos son $Q^*(x,a) = Q^{\pi^*}(x,a)$.

La conexión con DP es elegante: $V^*(x) = \max_a Q^*(x,a)$, y si $a^*$ alcanza ese máximo, una política óptima se forma como $\pi^*(x) = a^*$. **Aquí reside la utilidad de los valores $Q$**: si el agente puede aprenderlos, decidir qué es óptimo es trivial —basta tomar el argmax sobre las acciones del estado actual—. Los valores $Q^*$ son únicos, aunque puede haber varias políticas óptimas.

Esto hace a Q-learning **model-free**: el agente nunca estima ni almacena $P_{xy}[a]$ ni $\mathcal{R}_x(a)$; toda la dinámica queda absorbida implícitamente en los valores $Q$, aprendidos por experiencia directa. Y es **off-policy**: el operador $\max_{a'}$ de la regla hace que el agente aprenda el valor de la política *greedy óptima* aunque, mientras explora, ejecute una política distinta y subóptima. Esa separación entre **política de comportamiento** (la que genera experiencia) y **política objetivo** (la que se evalúa) distingue a Q-learning de métodos on-policy como SARSA y le permite explorar libremente sin sesgar lo que aprende.

## La regla de actualización

La experiencia del agente es una secuencia de *episodios*. En cada uno: observa el estado $x_n$, ejecuta la acción $a_n$, observa el estado siguiente $y_n$, recibe el pago $r_n$ y ajusta sus valores con un factor de aprendizaje $\alpha_n$. En la notación moderna de la Clase 31:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\big]
$$

Las piezas:

- **$Q(s,a)$**: la estimación actual que se está corrigiendo.
- **$r$**: la recompensa inmediata observada.
- **$\gamma \max_{a'} Q(s',a')$**: "lo mejor que el agente cree que puede hacer desde el estado siguiente". El máximo es el corazón del carácter off-policy y de la conexión con la ecuación de optimalidad de Bellman.
- **$r + \gamma \max_{a'} Q(s',a') - Q(s,a)$**: el **error de diferencia temporal (TD error)**, la discrepancia entre la estimación actual y un objetivo mejor informado. Si es cero, la estimación ya es consistente con Bellman.
- **$\alpha$**: la tasa de aprendizaje, indexada por el número de visitas al par $(s,a)$ —lo que resulta esencial para la prueba—.

Dos advertencias del paper. Primera: la descripción **supone una tabla de búsqueda** (look-up table) para los $Q_n(x,a)$; con otras representaciones la convergencia puede fallar —profecía de la inestabilidad de los aproximadores—. Segunda: la condición ineludible es que la secuencia incluya **infinitos episodios para cada par estado-acción**; bajo las condiciones estocásticas del teorema, ningún método podría garantizar optimalidad con condiciones más débiles. Notablemente, los episodios *no* tienen que formar una secuencia continua —el $y$ de uno no necesita ser el $x$ del siguiente—, lo que habilita el *replay* de experiencias.

Esa exigencia de cobertura es exactamente lo que motiva la política **$\varepsilon$-greedy**: con probabilidad $1-\varepsilon$ el agente explota (argmax de $Q$) y con $\varepsilon$ explora (acción aleatoria). El balance exploración/explotación no es un detalle de implementación, sino la condición que vuelve válida la garantía teórica.

## La prueba de convergencia (la contribución de 1992)

### El teorema

Dadas recompensas acotadas $|r_n| \le \mathcal{R}$, tasas $0 \le \alpha_n < 1$, y siendo $n^i(x,a)$ el índice de la $i$-ésima vez que la acción $a$ se prueba en el estado $x$, si

$$
\sum_{i=1}^{\infty} \alpha_{n^i(x,a)} = \infty, \qquad \sum_{i=1}^{\infty} \big[\alpha_{n^i(x,a)}\big]^2 < \infty
$$

para todo $x, a$, entonces $Q_n(x,a) \to Q^*(x,a)$ cuando $n \to \infty$, con probabilidad 1.

Son las clásicas condiciones de **aproximación estocástica de Robbins-Monro**: la suma de las tasas debe diverger (para que el aprendizaje nunca se congele antes de tiempo y pueda alcanzar cualquier valor), pero la suma de sus cuadrados debe converger (para que el ruido se promedie a cero y las estimaciones se estabilicen). Una tasa $\alpha_i = 1/i$ las satisface; una tasa **constante** no satisface la segunda, y por eso en la práctica solo da convergencia aproximada.

### El action-replay process (ARP)

La clave de la prueba es un proceso de Markov controlado **artificial**, el *action-replay process* (ARP), construido a partir de la secuencia de episodios y tasas. La analogía del paper es un juego de cartas: cada episodio $(x_t, a_t, y_t, r_t, \alpha_t)$ es una carta; todas forman un mazo infinito, con el primer episodio cerca del fondo (la carta 0 lleva los valores iniciales $Q_0$). Un estado del ARP, $(x, n)$, combina un nivel $n$ (número de carta) con un estado real $x$.

Dado $(x,n)$ y la acción $a$: se descartan las cartas posteriores a $n$ y se retiran cartas de arriba hasta hallar una cuyo estado-acción inicial coincida con $(x,a)$, digamos el episodio $t$. Se lanza una moneda sesgada con probabilidad $\alpha_t$ de cara. Si sale cara, se **reproduce** ese episodio: se emite $r_t$ y se transita a $(y_t, t-1)$ —un nivel más abajo—. Si sale cruz, se descarta la carta y se sigue. Si se llega al fondo, el juego termina entregando $Q_0(x,a)$. Como toda acción baja de nivel y las cartas nunca se reponen, el ARP **siempre termina** tras un número finito de pasos, y como es un MDP legítimo tiene sus propios valores $Q^*$ bien definidos.

### Los dos lemas

- **Lema A:** los $Q_n(x,a)$ *son* los valores-acción óptimos del estado $(x,n)$ del ARP. El ARP fue construido para tener esta propiedad; se prueba por inducción hacia atrás. La intuición: la regla de Q-learning es exactamente la ecuación de optimalidad de Bellman *dentro* del ARP.
- **Lema B:** el ARP **converge al proceso real**. Sus cuatro partes muestran que la cola descontada es despreciable (vía $\gamma$), que es improbable caer muy por debajo de un nivel dado, que las transiciones y recompensas del ARP convergen a las reales —y aquí entran las condiciones de Robbins-Monro sobre las tasas—, y que dos MDP aproximadamente iguales tienen valores de acción cercanos.

Juntando ambos: el ARP tiende al proceso real (Lema B), luego sus $Q^*$ tienden a los reales; pero esos $Q^*$ del nivel $n$ son exactamente $Q_n(x,a)$ (Lema A); por lo tanto $Q_n(x,a) \to Q^*(x,a)$ con probabilidad 1. La elegancia está en reducir una convergencia estocástica iterativa a la convergencia de un MDP auxiliar hacia otro.

El paper esboza además dos extensiones que preservan la garantía: el **caso no descontado** ($\gamma=1$) con estados absorbentes (la certeza de quedar atrapado cumple el rol que jugaba $\gamma<1$) y la actualización de **muchos valores $Q$ por iteración**. Observa también un *continuo* —no una dicotomía— entre Q-learning puro y los métodos basados en modelo, prefigurando tanto el *experience replay* como el RL model-based moderno.

## Por qué no escala: de la tabla a DQN

La limitación más consecuente es explícita en el propio paper: la prueba supone una **tabla discreta** con una entrada por par $(s,a)$, y eso **no escala** a espacios de estado grandes o continuos (imágenes, control robótico de alta dimensión). Esta es exactamente la barrera que **Deep Q-Learning** ([DQN, Mnih et al. 2013](/papers/dqn-atari-mnih-2013)) rompió: sustituyó la tabla por una **red neuronal** $Q(s,a;\theta)$ entrenada para minimizar el mismo error TD $r + \gamma \max_{a'} Q(s',a') - Q(s,a)$. El precio fue perder las garantías de convergencia y tener que reintroducir estabilidad mediante *experience replay* y una *red objetivo* separada.

Otras limitaciones reconocidas: el teorema no cubre las actualizaciones **multi-paso** TD($\lambda$), y la **exploración exhaustiva** es teóricamente ineludible pero costosa, sin que el paper diga *cómo* explorar de forma eficiente.

## Impacto

Q-learning es el acta de nacimiento teórica de toda una rama del RL, y su importancia se mide en tres planos:

1. **Fundacional para el RL sin modelo.** Estableció que un agente puede aprender a actuar óptimamente *con garantía de convergencia* sin construir jamás un mapa de su ambiente, solo por ensayo y error. Combinado con su carácter off-policy, fue el algoritmo de referencia de una generación de investigación en RL tabular.
2. **Base de Deep Q-Learning.** La regla de Watkins y Dayan es, literalmente, el núcleo de DQN. La línea que conduce a los juegos de Atari, AlphaGo y el RL profundo moderno desciende directamente de este algoritmo.
3. **Garantía teórica como sello.** Logró lo que había eludido a la mayoría de las heurísticas contemporáneas: una prueba de convergencia con probabilidad 1 bajo condiciones razonables. Esa solidez es parte de por qué Q-learning, y no otras heurísticas, se volvió canónico en libros de texto y cursos.

## Por qué importa para la Clase 31

La [Clase 31](/clases/clase-31) ("Aprendizaje Reforzado") dedica su sección 3 a Q-Learning, y este paper provee todo su andamiaje:

- **La ecuación de Bellman** que la regla persigue: $Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s',a')]$. El error TD mide cuánto la viola la estimación actual; cada actualización empuja $Q$ hacia su punto fijo.
- **La tabla Q** que la clase manipula es la look-up table cuya convergencia el paper prueba.
- **$\varepsilon$-greedy** se justifica directamente en la condición de visitar infinitamente todo par $(s,a)$.
- **El factor de descuento $\gamma$** asegura que el problema esté bien definido y que el agente prefiera recompensas próximas.
- **El salto a Deep Q-Learning** resuelve la barrera tabular que el paper reconoce.

## Notas y enlaces

- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado)
- Clase: [/clases/clase-31](/clases/clase-31)
- Escalado profundo: [/papers/dqn-atari-mnih-2013](/papers/dqn-atari-mnih-2013)
- Origen del algoritmo: Watkins, C. J. C. H. (1989), *Learning from Delayed Rewards*, PhD Thesis, University of Cambridge.
- Venue: *Machine Learning*, vol. 8, pp. 279-292, 1992. Kluwer Academic Publishers.
