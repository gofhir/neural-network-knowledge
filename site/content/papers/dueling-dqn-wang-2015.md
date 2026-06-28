---
title: "Dueling DQN (2015)"
weight: 351
math: true
---

{{< paper-card
    title="Dueling Network Architectures for Deep Reinforcement Learning"
    authors="Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas"
    year="2015"
    venue="ICML 2016"
    pdf="/papers/dueling-dqn-wang-2015.pdf"
    arxiv="1511.06581" >}}
Paper de Google DeepMind (Best Paper Award de ICML 2016) que no propone un algoritmo nuevo de aprendizaje reforzado, sino una **arquitectura de red**. La idea: partir la red en **dos flujos** (streams) que comparten el tronco convolucional —uno estima el valor del estado $V(s)$, otro la ventaja $A(s,a)$ de cada acción— y recombinarlos en $Q(s,a)$. Así la red aprende **qué estados son valiosos sin tener que evaluar el efecto de cada acción**. Es una mejora *drop-in* del [DQN](/papers/dqn-nature-mnih-2015): misma entrada/salida, mismo bucle de Q-learning, pero una red que generaliza mejor a través de las acciones. Componente canónico de Rainbow.
{{< /paper-card >}}

---

## Contexto: por qué separar $V$ y $A$

El [DQN](/papers/dqn-nature-mnih-2015) (Mnih et al., 2015) estima $Q(s,a)$ con una sola red: capas convolucionales sobre los píxeles y, al final, un vector de Q-valores, uno por acción. Funciona, pero arrastra una ineficiencia estructural.

**La observación clave:** para muchos estados *no hace falta* estimar el valor de cada acción. El ejemplo recurrente del paper es Enduro (conducir esquivando autos): saber si conviene ir a izquierda o derecha solo importa cuando la colisión es inminente. En la mayoría de los frames —carretera despejada— la acción no cambia nada de lo que viene; lo que de verdad determina el retorno es el **valor del estado**. Y como el Q-learning se apoya en *bootstrapping*, estimar bien $V(s)$ importa para **todos** los estados, porque ese valor se propaga hacia atrás en cada actualización temporal.

El DQN mezcla ambas cosas: cada Q-valor es el valor del estado más la ventaja de la acción, y la red debe aprender esa suma para cada par $(s,a)$ por separado. El paper lo cuantifica en Seaquest: tras entrenar, el *action gap* promedio (diferencia entre la mejor y la segunda mejor acción) ronda **0.04**, mientras el valor de estado promedio ronda **15** —la señal que distingue acciones es ~375 veces más pequeña que el valor que comparten. En un solo stream, un poco de ruido reordena las acciones y desestabiliza la política casi-greedy. Separar $V$ de $A$ vuelve la arquitectura robusta a ese efecto de escala. El antecedente teórico viene de Baird (1993) y el *advantage updating* (Harmon et al., 1995), pero ahí representación y algoritmo estaban acoplados; aquí están **desacoplados por construcción**.

## La idea: dos streams y el truco de la media

La ventaja relaciona las tres funciones:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

con $\mathbb{E}_{a\sim\pi}[A^\pi(s,a)] = 0$. Intuitivamente, $V$ mide cuán bueno es estar en el estado; $A$ resta ese valor para dar una medida **relativa** de cada acción.

La dueling architecture representa **ambas funciones con un solo modelo**: un stream emite un **escalar** $V(s;\theta,\beta)$ y el otro un **vector** $A(s,a;\theta,\alpha)$ de dimensión $|A|$, donde $\theta$ es el tronco convolucional compartido y $\alpha,\beta$ los dos streams.

**El problema de identificabilidad.** La combinación ingenua

$$Q(s,a) = V(s;\theta,\beta) + A(s,a;\theta,\alpha)$$

es **no identificable**: dado $Q$, no se pueden recuperar $V$ y $A$ de forma única. Si se suma una constante a $V$ y se resta la misma a $A$, el $Q$ no cambia. Esa libertad sobrante impide que $V$ y $A$ converjan a algo con sentido y degrada el desempeño. Una opción es anclar restando el **máximo** de la ventaja, de modo que para la acción óptima $Q(s,a^*) = V(s)$. Pero la variante que **realmente se usa** resta la **media**:

$$Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + \Big(A(s,a;\theta,\alpha) - \frac{1}{|A|}\sum_{a'} A(s,a';\theta,\alpha)\Big)$$

El compromiso es elegante: restar la media pierde la semántica estricta de $V$ y $A$ (quedan desplazados por una constante), pero **estabiliza la optimización**, porque las ventajas solo deben cambiar tan rápido como su propia media, en lugar de compensar cada cambio en la acción óptima. Punto crucial: restar la media **no altera el ranking** de los $Q$, así que la política greedy o ε-greedy se preserva exactamente, y al actuar basta evaluar el stream de ventaja. Esta agregación **es parte de la red, no un paso algorítmico aparte**: se implementa como una capa y se entrena con retropropagación pura, sin supervisión extra ni tocar la ecuación de Bellman.

## Arquitectura

La red usada en Atari:

- **Tronco convolucional compartido** (idéntico a DQN): 3 capas conv —32 filtros 8×8 stride 4, 64 filtros 4×4 stride 2, 64 filtros 3×3 stride 1.
- **Bifurcación en dos streams**, cada uno con una capa densa de **512 unidades**. El stream de valor termina en **una sola salida** (el escalar $V$); el de ventaja en **$|A|$ salidas** (una por acción; en Atari, de 3 a 18). ReLU entre capas.
- **Módulo de agregación** que recombina ambos streams con la ecuación de la media. La salida tiene la misma forma que un DQN estándar, por eso la red es *drop-in*: se reciclan DDQN, SARSA, prioritized replay sin cambios.

**Detalles que importan:** como ambos streams retropropagan hacia la última capa conv compartida, el gradiente combinado se **reescala por $1/\sqrt{2}$** para estabilizar; y se aplica **gradient clipping** (norma ≤ 10), que el paper incluye también en sus líneas base para que la comparación sea justa.

**Evidencia visual.** Los *saliency maps* (Jacobianos de $V$ y $A$ respecto a los píxeles) en Enduro muestran roles distintos: el stream de **valor** atiende al horizonte de la carretera y al marcador; el de **ventaja** casi no atiende a la imagen sin autos enfrente, pero se activa apenas hay un auto en curso de colisión —justo cuando la acción importa.

## Experimentos

**Evaluación de política (entorno "corredor").** Antes de Atari, el paper aísla el efecto comparando un MLP de un stream contra la versión dueling con **5, 10 y 20 acciones** (acciones extra son no-ops). Con 5 acciones convergen parejos; pero **a más acciones, la dueling architecture gana cada vez más**, porque el stream $V$ aprende un valor compartido entre muchas acciones similares. Esto predice exactamente lo que pasará en Atari.

**Atari (57 juegos),** mismos hiperparámetros, solo píxeles. Se entrena la dueling network con DDQN y se compara contra un DDQN de un solo stream con igual número de parámetros (*Single Clip*) y contra el original (*Single*). Resultados (% de desempeño humano, métrica 30 no-ops):

| Agente | Media / Mediana |
|---|---|
| Nature DQN | 227.9% / 79.1% |
| Single | 307.3% / 117.8% |
| Single Clip | 341.2% / 132.6% |
| **Duel Clip** | **373.1% / 151.5%** |
| Prior. Single | 434.6% / 123.7% |
| **Prior. Duel Clip** | **591.9% / 172.1%** |

Hallazgos clave:

- **Duel Clip** supera a Single Clip (capacidad equivalente) en **75.4%** de los juegos (43/57) y a Single en **80.7%** (46/57); nivel humano en 42 de 57.
- **La ventaja crece con el número de acciones.** En juegos de 18 acciones, Duel Clip gana el **86.6%** de las veces (26/30) —confirmación directa del experimento del corredor.
- **Robustez.** Bajo la métrica más exigente (*human starts*, que castiga la memorización), sigue ganando: 70.2% sobre Single, y 83.3% en juegos de 18 acciones.
- **Ortogonal al prioritized replay.** Como [PER](/papers/per-schaul-2015) y la dueling architecture atacan aspectos distintos, su combinación —**Prior. Duel Clip**— fijó el nuevo estado del arte en ALE (media 591.9%).

## Limitaciones

- **Mejora arquitectónica, no algorítmica.** Hereda las patologías del Q-learning profundo (sobreestimación, dependencia del target network, sensibilidad a hiperparámetros) y no resuelve la exploración: Montezuma's Revenge sigue en 0.0.
- **Beneficio condicionado al número de acciones.** Con pocas acciones (3-4, o 5 en el corredor) la ventaja sobre el single-stream es marginal; el costo extra de los dos streams casi no se paga.
- **Sin garantía de qué representa cada stream.** Al restar la media, $V$ y $A$ quedan desplazados por una constante y pierden su interpretación estricta —el "valor" aprendido no es exactamente $V^\pi$.
- **Interacciones sutiles.** Combinar con prioritized replay exigió reajustar hiperparámetros a mano.
- **No es universalmente superior:** en algunos juegos (Freeway, Breakout) rinde por debajo del single-stream.

## Impacto

La dueling architecture se volvió uno de los **componentes canónicos del DQN moderno**. Su mayor legado es ser una de las seis mejoras integradas en **Rainbow** (Hessel et al., 2018), junto a Double Q-learning, [prioritized replay](/papers/per-schaul-2015), multi-step learning, distributional RL y noisy nets —demostrando que estas mejoras son en gran medida complementarias y aditivas. Con Double DQN y PER, forma el trío de mejoras "de bajo costo y alto retorno" que casi toda implementación seria de RL basado en valor incorpora por defecto.

Su valor pedagógico es notable: enseña que **el sesgo inductivo de la arquitectura puede sustituir parte del trabajo del algoritmo**. La factorización $Q = V + (A - \text{media}\,A)$ impone estructura conocida del problema —que valor de estado y ventaja son cantidades de naturaleza y escala distintas— directamente en la red, en vez de esperar que un único stream las descubra desde cero. Ganó el Best Paper Award de ICML 2016 por esa elegancia: una idea simple, barata, combinable con todo, y con mejoras dramáticas y bien aisladas.

## Por qué importa para la Clase 31

La [Clase 31](/clases/clase-31) introduce el DQN como puente entre el Q-learning tabular y el RL profundo. La dueling architecture es la **siguiente pieza natural** de esa historia:

- **No cambia el algoritmo de la clase.** El bucle de DQN/DDQN —muestrear del replay buffer, calcular el target $y = r + \gamma\,Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$, dar un paso de gradiente— queda intacto. Solo se reemplaza la red interna: donde había un MLP de un stream, ahora hay dos streams y una capa de agregación.
- **Hace tangible la distinción $V$ vs $Q$ vs $A$.** Las tres cantidades que define el [fundamento de aprendizaje reforzado](/fundamentos/aprendizaje-reforzado) dejan de ser notación y se vuelven *dos cabezas físicas* de una red, con la identidad $\mathbb{E}_a[A]=0$ implementada literalmente como la resta de la media.
- **Ilustra el rol del sesgo inductivo.** Conecta con la lección transversal del curso: meter conocimiento del problema en la arquitectura (como las convnets con la invarianza traslacional) acelera el aprendizaje. Aquí el conocimiento es "en muchos estados la acción no importa, el valor sí".

## Notas y enlaces

- Preprint: [arxiv.org/abs/1511.06581](https://arxiv.org/abs/1511.06581) (arXiv:1511.06581v3, abr 2016).
- Venue: ICML 2016 — Best Paper Award. Afiliación: Google DeepMind, Londres.
- Paper base: [/papers/dqn-nature-mnih-2015](/papers/dqn-nature-mnih-2015). Extensión complementaria: [/papers/per-schaul-2015](/papers/per-schaul-2015).
- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado). Clase: [/clases/clase-31](/clases/clase-31).
