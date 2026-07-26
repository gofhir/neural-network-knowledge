# Apprenticeship Learning via Inverse Reinforcement Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Apprenticeship Learning via Inverse Reinforcement Learning*.
- **Autores:** Pieter Abbeel y Andrew Y. Ng, Computer Science Department, Stanford University.
- **Venue:** *Proceedings of the 21st International Conference on Machine Learning (ICML 2004)*, Banff, Canadá.
- **Año:** 2004. Existe una versión extendida ("full paper") citada por los propios autores con las demostraciones completas: Abbeel & Ng (2004), disponible en su época en `cs.stanford.edu/~pabbeel/irl/`.
- **Linaje:** es la continuación directa de Ng & Russell (2000), *Algorithms for Inverse Reinforcement Learning*, el trabajo que fundó el aprendizaje reforzado inverso (IRL). Este paper toma la maquinaria de IRL y la reorienta hacia un objetivo distinto: no recuperar la recompensa "verdadera", sino igualar el desempeño de un experto.

El problema que aborda el paper es el siguiente. En el formalismo estándar de un proceso de decisión de Markov (MDP), se asume que la **función de recompensa está dada** y, a partir de ella, algoritmos estándar encuentran la política óptima. Pero los autores argumentan que en muchas tareas reales —el ejemplo recurrente es **conducir en autopista**— especificar la recompensa a mano es sorprendentemente difícil. Conducir bien implica compensar (trade off) muchos deseos simultáneos: mantener distancia de seguridad, no acercarse al borde, evitar peatones, mantener una velocidad razonable, cierta preferencia por el carril central, no cambiar de carril demasiado seguido, etcétera. Asignar los pesos exactos que balancean todos estos factores es tan difícil que, en palabras de los autores, "aunque son capaces de conducir competentemente, no creen poder especificar con confianza una función de recompensa específica para la tarea de conducir bien".

La observación clave es que **es mucho más fácil demostrar la tarea que especificar su recompensa**. Del mismo modo que a un joven se le enseña a conducir mostrándole cómo se hace, en lugar de dictarle una fórmula de recompensa, aquí se aprende de las **demostraciones de un experto**. Esto se llama *apprenticeship learning* (aprendizaje por aprendizaje de oficio, o aprendizaje por imitación/demostración).

La contribución central es un algoritmo que, dado un MDP sin recompensa (`MDP\R`), un conjunto de *features* conocidas y las demostraciones de un experto, produce una política cuyo desempeño se acerca al del experto. El resultado teórico notable es que **aunque el algoritmo nunca recupere la verdadera función de recompensa del experto**, la política que devuelve alcanza un desempeño comparable al del experto, medido justamente respecto de esa recompensa desconocida. El algoritmo termina en un número pequeño de iteraciones y requiere sorprendentemente pocas demostraciones.

## 2. Contexto: de recuperar la recompensa a igualar el desempeño

Para situar este paper hay que entender qué lo distingue tanto de su predecesor (Ng & Russell, 2000) como de los enfoques de imitación directa que dominaban entonces.

**Enfoques de imitación directa.** La mayoría de los métodos previos de aprendizaje por demostración intentaban **imitar directamente al demostrador**: aplicar aprendizaje supervisado para aprender un mapeo directo de estados a acciones (Pomerleau 1989 con ALVINN, Sammut et al. 1992 "learning to fly", entre otros). Esto es esencialmente lo que hoy llamamos *behavioral cloning*. El paper señala una limitación crucial de estos métodos con el ejemplo de la conducción: **seguir ciegamente la trayectoria del experto no funciona**, porque el patrón de tráfico que se encuentra es distinto cada vez. Atkeson & Schaal (1997) son citados como excepción notable —hacen que un brazo robótico siga una trayectoria demostrada penalizando cuadráticamente la desviación— pero eso solo aplica cuando la tarea *es* imitar la trayectoria; no generaliza a situaciones nuevas.

**El giro conceptual.** El campo entero del aprendizaje reforzado se funda en la premisa de que **la función de recompensa —no la política ni la función de valor— es la definición más sucinta, robusta y transferible de una tarea**. Si eso es cierto, entonces lo natural para el aprendizaje por demostración es *aprender la recompensa*, no la política. De ahí que el paper recurra al **aprendizaje reforzado inverso** (IRL): el problema de derivar una función de recompensa a partir de comportamiento observado, planteado por Ng & Russell (2000).

**La diferencia con Ng & Russell (2000).** Aquí está la sutileza que hay que destacar. El problema del IRL puro está **mal condicionado (ill-posed)**: muchas funciones de recompensa distintas —incluida la trivial $R=0$, bajo la cual toda política es óptima— explican el comportamiento del experto. Ng & Russell (2000) atacaban directamente el problema de recuperar una recompensa "buena". Abbeel & Ng dan un paso al costado: **su objetivo NO es recuperar la recompensa verdadera del experto**. El objetivo es más modesto y a la vez más útil: encontrar una política cuyo **desempeño** se acerque al del experto. El paso de IRL aparece dentro del algoritmo (adivinar pesos de recompensa), pero las garantías de desempeño **no dependen de que ese paso recupere la recompensa correcta** — solo dependen de igualar las *feature expectations*. Este reencuadre es lo que hace posible dar garantías formales pese a la ambigüedad inherente del IRL.

## 3. Formalización: recompensa lineal y feature expectations

El escenario es un MDP definido como la tupla $(S, A, T, \gamma, D, R)$: estados $S$ finitos, acciones $A$, probabilidades de transición $T=\{P_{sa}\}$, factor de descuento $\gamma \in [0,1)$, distribución inicial de estados $D$ (de la que se muestrea $s_0$) y recompensa $R$ acotada en valor absoluto por 1. Cuando falta la recompensa, se escribe `MDP\R` $=(S, A, T, \gamma, D)$.

El supuesto estructural central es que existe un vector de **features** $\phi: S \to [0,1]^k$ sobre los estados, y que la recompensa "verdadera" es una **combinación lineal** de esas features:

$$R^*(s) = w^* \cdot \phi(s), \qquad w^* \in \mathbb{R}^k.$$

Para garantizar que la recompensa quede acotada por 1 se supone $\|w^*\|_1 \le 1$. En el dominio de la conducción, $\phi$ sería un vector de indicadores de los distintos deseos —si acabamos de chocar con otro auto, si vamos en el carril central, etc.— y el vector desconocido $w^*$ codifica cuánto pesa cada uno.

Una política $\pi$ es un mapeo de estados a distribuciones de probabilidad sobre acciones. Su valor es el retorno descontado esperado, y aquí ocurre la manipulación algebraica clave del paper:

$$
E_{s_0 \sim D}[V^\pi(s_0)] = E\left[\sum_{t=0}^{\infty} \gamma^t R(s_t) \,\middle|\, \pi\right]
= E\left[\sum_{t=0}^{\infty} \gamma^t\, w \cdot \phi(s_t) \,\middle|\, \pi\right]
= w \cdot E\left[\sum_{t=0}^{\infty} \gamma^t \phi(s_t) \,\middle|\, \pi\right].
$$

El último paso solo saca $w$ fuera de la esperanza (linealidad). Eso motiva definir el objeto central del paper, las **feature expectations** (expectativas de features): el vector de valores de feature acumulados y descontados que induce una política,

$$\mu(\pi) = E\left[\sum_{t=0}^{\infty} \gamma^t \phi(s_t) \,\middle|\, \pi\right] \in \mathbb{R}^k.$$

Con esta notación, el valor de cualquier política se escribe de forma compacta como un producto interno:

$$E_{s_0 \sim D}[V^\pi(s_0)] = w \cdot \mu(\pi).$$

Esta es la idea que lo hace todo posible: **dado que la recompensa es lineal en las features, las feature expectations de una política determinan completamente su valor esperado** para *cualquier* $w$. Es decir, si dos políticas tienen las mismas $\mu$, tienen el mismo desempeño bajo toda recompensa lineal en $\phi$ — sin importar cuál sea el $w$ verdadero.

**Mezcla de políticas (mixing).** Un truco técnico importante: dadas dos políticas $\pi_1, \pi_2$, se puede construir una política mixta $\pi_3$ que al inicio de la trayectoria lanza una moneda con sesgo $\lambda$ y, según el resultado, actúa siempre según $\pi_1$ (con probabilidad $\lambda$) o siempre según $\pi_2$ (con probabilidad $1-\lambda$). Por linealidad de la esperanza, $\mu(\pi_3) = \lambda\mu(\pi_1) + (1-\lambda)\mu(\pi_2)$. El aleatorio se elige **una sola vez al comienzo**, no en cada paso. Generalizando, cualquier combinación convexa de las feature expectations de un conjunto de políticas es alcanzable mezclándolas. Esto será clave para el paso final del algoritmo.

**Estimación de $\mu_E$.** Del experto $\pi_E$ solo observamos trayectorias. Su feature expectation $\mu_E = \mu(\pi_E)$ se estima empíricamente por Monte Carlo a partir de $m$ trayectorias:

$$\hat\mu_E = \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{\infty} \gamma^t \phi\big(s_t^{(i)}\big).$$

En la práctica las trayectorias se truncan tras $H$ pasos; si $H$ es el $\epsilon$-horizonte $H_\epsilon = \log_\gamma(\epsilon(1-\gamma))$, el truncamiento introduce a lo sumo $\epsilon$ de error.

## 4. Método: por qué basta igualar las feature expectations

El problema se reduce a: dado `MDP\R`, el feature mapping $\phi$ y $\mu_E$, encontrar una política $\tilde\pi$ cuyo desempeño se acerque al del experto bajo la recompensa desconocida $R^* = (w^*)^\top \phi$. La estrategia es hallar $\tilde\pi$ tal que $\|\mu(\tilde\pi) - \mu_E\|_2 \le \epsilon$.

**La garantía fundamental.** Supongamos que logramos esa cercanía. Entonces, para *cualquier* $w$ con $\|w\|_1 \le 1$:

$$
\left| E\Big[\sum_t \gamma^t R(s_t)\,\big|\,\pi_E\Big] - E\Big[\sum_t \gamma^t R(s_t)\,\big|\,\tilde\pi\Big] \right|
= \left| w^\top \mu(\tilde\pi) - w^\top \mu_E \right|
\le \|w\|_2\, \|\mu(\tilde\pi) - \mu_E\|_2
\le 1 \cdot \epsilon = \epsilon.
$$

El primer paso usa la representación del valor como producto interno; el segundo es Cauchy-Schwarz ($|x^\top y| \le \|x\|_2\|y\|_2$); el tercero usa $\|w\|_2 \le \|w\|_1 \le 1$. La conclusión es profunda: **si igualamos las feature expectations dentro de $\epsilon$, entonces la diferencia de desempeño respecto del experto es a lo sumo $\epsilon$, para toda recompensa lineal posible**. No necesitamos conocer $w^*$; la cota vale uniformemente sobre todos los $w$ admisibles. Por eso el objetivo de "igualar $\mu_E$" es suficiente aunque nunca identifiquemos la recompensa verdadera.

**El bucle del algoritmo (versión max-margin).** El problema queda reducido a hallar una política con $\mu(\tilde\pi)$ cercana a $\mu_E$. El algoritmo lo hace iterativamente:

1. Elegir aleatoriamente una política inicial $\pi^{(0)}$, calcular (o estimar por Monte Carlo) $\mu^{(0)} = \mu(\pi^{(0)})$, fijar $i=1$.
2. **Paso IRL / max-margin.** Resolver
$$t^{(i)} = \max_{w:\|w\|_2 \le 1} \; \min_{j \in \{0,\dots,i-1\}} \; w^\top\big(\mu_E - \mu^{(j)}\big),$$
y sea $w^{(i)}$ el $w$ que alcanza el máximo.
3. Si $t^{(i)} \le \epsilon$, **terminar**.
4. **Paso RL forward.** Usando el algoritmo de RL, calcular la política óptima $\pi^{(i)}$ para el MDP con recompensa $R = (w^{(i)})^\top \phi$.
5. Calcular (o estimar) $\mu^{(i)} = \mu(\pi^{(i)})$.
6. $i \leftarrow i+1$ y volver al paso 2.

**Interpretación geométrica y conexión con SVM.** El paso 2 es un problema de máximo margen. Se puede reescribir como el programa:

$$
\max_{t,\,w} \; t \quad \text{s.a.} \quad w^\top \mu_E \ge w^\top \mu^{(j)} + t \;\; (j=0,\dots,i-1), \quad \|w\|_2 \le 1.
$$

La restricción $w^\top\mu_E \ge w^\top\mu^{(j)} + t$ dice que el algoritmo busca una recompensa $R=w^{(i)}\cdot\phi$ bajo la cual **el experto lo hace mejor, por un margen $t$, que todas las políticas encontradas hasta ahora**. Esto es exactamente el problema de hallar el **hiperplano de máximo margen** que separa dos conjuntos de puntos, en el sentido de las máquinas de vectores de soporte (SVM, Vapnik 1998): se asocia la etiqueta $+1$ a $\mu_E$ y la etiqueta $-1$ a las $\{\mu(\pi^{(j)})\}$, y $w^{(i)}$ es el vector unitario ortogonal al hiperplano separador. Por la restricción $\|w\|_2 \le 1$ (norma 2, a diferencia de la norma 1 de Ng & Russell 2000) **no es un programa lineal sino uno cuadrático (QP)**; cualquier solver de SVM o de QP genérico sirve. Cada iteración empuja el conjunto de políticas encontradas hacia $\mu_E$, y $w^{(i)}$ es la dirección en que la mejor política actual queda más lejos del experto.

**El método de proyección (sin QP).** El paper ofrece una variante que evita el solver de QP, llamada *projection method*. Reemplaza el paso 2 por una **proyección ortogonal** de $\mu_E$ sobre la recta que une las dos últimas iteraciones. Concretamente, se calcula

$$\bar\mu^{(i-1)} = \bar\mu^{(i-2)} + \frac{(\mu^{(i-1)}-\bar\mu^{(i-2)})^\top(\mu_E-\bar\mu^{(i-2)})}{(\mu^{(i-1)}-\bar\mu^{(i-2)})^\top(\mu^{(i-1)}-\bar\mu^{(i-2)})}\,(\mu^{(i-1)}-\bar\mu^{(i-2)}),$$

y luego se toma $w^{(i)} = \mu_E - \bar\mu^{(i-1)}$ como la nueva dirección de recompensa y $t^{(i)} = \|\mu_E - \bar\mu^{(i-1)}\|_2$ como la distancia. En la primera iteración se fija $w^{(1)} = \mu_E - \mu^{(0)}$ y $\bar\mu^{(0)} = \mu^{(0)}$. Es más simple de implementar y, empíricamente, converge algo más rápido que el max-margin.

**Selección de la política final.** Al terminar, el algoritmo devuelve el **conjunto** $\{\pi^{(i)} : i=0,\dots,n\}$. La garantía (derivada de las condiciones de terminación) es que existe al menos una política del conjunto cuyo desempeño bajo $R^*$ es al menos tan bueno como el del experto menos $\epsilon$. Hay dos formas de elegir:

- **Con ayuda humana:** pedir al diseñador que inspeccione las políticas y elija una con desempeño aceptable. Por el teorema de Carathéodory basta examinar a lo sumo $k+1$ políticas (no las $n+1$).
- **Sin ayuda humana:** resolver un QP que halla el punto más cercano a $\mu_E$ dentro de la **envolvente convexa (convex closure)** de $\{\mu^{(0)},\dots,\mu^{(n)}\}$:
$$\min \|\mu_E - \mu\|_2 \quad \text{s.a.} \quad \mu = \sum_i \lambda_i \mu^{(i)}, \;\; \lambda_i \ge 0, \;\; \sum_i \lambda_i = 1.$$
Como $\mu_E$ está separado de los puntos por un margen de a lo sumo $\epsilon$, la solución cumple $\|\mu_E - \mu\|_2 \le \epsilon$. Y **mezclando** las políticas $\pi^{(i)}$ con los pesos $\lambda_i$ (según el truco de mixing) se obtiene una política real cuya feature expectation es exactamente ese $\mu$, y por lo tanto con desempeño cercano al del experto.

El paper subraya explícitamente que, aunque un paso se llame "IRL", el algoritmo **no necesariamente recupera la recompensa subyacente correcta**: las garantías dependen solo de igualar (aproximadamente) las feature expectations.

## 5. Resultados teóricos: terminación y complejidad de muestra

Dos teoremas dan las garantías formales.

**Teorema 1 (terminación).** Para todo `MDP\R`, features $\phi:S\to[0,1]^k$ y todo $\epsilon>0$, ambas versiones del algoritmo (max-margin y proyección) terminan con $t^{(i)}\le\epsilon$ tras a lo sumo

$$n = O\!\left(\frac{k}{(1-\gamma)^2\epsilon^2}\,\log\frac{k}{(1-\gamma)\epsilon}\right)$$

iteraciones. Es decir, el número de iteraciones es **polinómico en $k$ y en $1/\epsilon$ e independiente del tamaño del espacio de estados $|S|$**. La demostración (Lema 3) muestra que en cada iteración la distancia a $\mu_E$ se reduce por un factor fijo, $t^{(i+1)}/t^{(i)} \le \sqrt{k / (k+(1-\gamma)^2\epsilon^2)} < 1$, lo que da convergencia geométrica.

**Teorema 2 (complejidad de muestra).** Como $\mu_E$ debe estimarse por Monte Carlo, ¿cuántas trayectorias $m$ del experto se necesitan? El teorema garantiza que, con probabilidad al menos $1-\delta$, si

$$m \ge \frac{2k}{(\epsilon(1-\gamma))^2}\,\log\frac{2k}{\delta},$$

el algoritmo termina en el número de iteraciones del Teorema 1 y devuelve una política $\tilde\pi$ tal que, para *cualquier* recompensa verdadera $R^*(s)=(w^*)^\top\phi(s)$ con $\|w^*\|_1\le 1$,

$$E\Big[\sum_{t}\gamma^t R^*(s_t)\,\big|\,\tilde\pi\Big] \ge E\Big[\sum_{t}\gamma^t R^*(s_t)\,\big|\,\pi_E\Big] - \epsilon.$$

La prueba combina la desigualdad de Hoeffding (con union bound sobre las $k$ componentes) para acotar $\|\mu_E - \hat\mu_E\|_\infty$, y luego $\|\cdot\|_2 \le \sqrt{k}\|\cdot\|_\infty$. Lo crucial: **el número de demostraciones depende de $k$ (dimensión de las features), no del tamaño del espacio de estados**. Esto explica por qué, empíricamente, bastan pocas demostraciones.

**Degradación elegante.** Si la recompensa verdadera **no** está exactamente en el espacio generado por las features —es decir $R^*(s)=w^*\cdot\phi(s)+\varepsilon(s)$ con un residuo $\varepsilon(s)$— el algoritmo sigue funcionando: su desempeño es peor que el del experto por no más de $O(\|\varepsilon\|_\infty)$. La aproximación lineal se degrada suavemente, no catastróficamente.

## 6. Experimentos

**Gridworld.** Grillas de 128×128 con recompensas múltiples y esparcidas, divididas en 64 "macroceldas" de 16×16. Cada macrocelda $i$ tiene una feature indicadora $\phi_i(s)$; solo unas pocas tienen recompensa positiva. El agente tiene cuatro acciones (direcciones cardinales) con 30% de probabilidad de fallo (movimiento aleatorio), y $\gamma=0.99$. Dos hallazgos:

- **Convergencia (Figura 3):** con $\mu_E$ conocido exactamente, max-margin y proyección convergen a tasas similares, con la proyección algo mejor. La distancia a las feature expectations del experto cae rápidamente en pocas decenas de iteraciones.
- **Eficiencia de muestra (Figura 4):** al variar el número $m$ de trayectorias del experto, el algoritmo de IRL alcanza un desempeño cercano al del experto con **muchas menos** demostraciones que tres baselines de imitación directa: "mimic the expert" (copiar la acción del experto en estados ya vistos, aleatoria en el resto), "parameterized policy stochastic" (frecuencias empíricas por macrocelda) y "parameterized policy majority vote" (acción más frecuente por macrocelda). Notablemente, las políticas parametrizadas **nunca alcanzan** el desempeño del experto porque su clase de políticas no es suficientemente rica. Además, cuando al algoritmo se le indican de antemano qué macroceldas tienen peso no nulo (reduciendo la dimensión de $\phi$), aprende con aún menos trayectorias. El eje x está en escala logarítmica base 10: la ventaja en cantidad de datos es de órdenes de magnitud.

**Simulador de conducción (el ejemplo estrella).** Se implementó un simulador de conducción en autopista 3D. El auto propio va a 25 m/s (56 mph), más rápido que el resto del tráfico, por lo que a veces es necesario salirse de la carretera para evitar colisiones. El MDP tiene cinco acciones (tres para virar suavemente a cada carril, dos para salirse paralelo a la ruta por izquierda o derecha). Las features indican el carril actual (cinco, incluyendo off-road izquierdo y derecho) y la distancia al auto más cercano en el carril (discretizada, 10 valores; distancia 0 = colisión), 15 features en total. La feature expectation del experto se estimó de **una sola trayectoria de 1200 muestras (2 minutos de conducción)**.

Los autores demostraron **cinco estilos de conducción** distintos, uno de los autores conduciendo 2 minutos por estilo:

1. **Nice:** máxima prioridad evitar colisiones; preferencia carril derecho > central > izquierdo > off-road.
2. **Nasty:** chocar tantos autos como sea posible.
3. **Right lane nice:** ir por el carril derecho, pero salirse para evitar chocar autos en ese carril.
4. **Right lane nasty:** ir off-road a la derecha, pero volver a la ruta para chocar autos del carril derecho.
5. **Middle lane:** ir por el carril central ignorando todo (chocando todos los autos del central).

En **cada** caso el algoritmo logró imitar cualitativamente el estilo demostrado. Como nunca se especificó una recompensa "verdadera", no se puede reportar desempeño respecto de $R^*$; pero la Tabla 1 muestra, para cada estilo, las feature expectations del experto $\hat\mu_E$ y las de la política aprendida $\mu(\tilde\pi)$ (muy cercanas entre sí), junto con los pesos $\tilde w$ hallados. Aunque la teoría no garantiza nada sobre los pesos, estos "generalmente tienen sentido intuitivo": por ejemplo, en el estilo "Nice" hay recompensas negativas para colisiones y off-road, y positivas mayores para el carril derecho. Este es precisamente el experimento que el profesor cita en la Clase 33 como el "aprender a conducir usando IRL".

## 7. Limitaciones

- **Ambigüedad de la recompensa.** El algoritmo no recupera la recompensa verdadera; el paso IRL solo adivina *algún* $w$ que separa. Esto es una elección de diseño consciente (el objetivo es igualar desempeño), pero significa que los pesos hallados no son interpretables como "la" recompensa del experto — la Tabla 1 los muestra como intuitivos, no como identificados.
- **Supuesto de recompensa lineal.** Todo descansa en $R^*(s)=w^*\cdot\phi(s)$. Los autores reconocen que extender a recompensas **no lineales** en las features, e incorporar construcción y selección automática de features, es un problema abierto importante. Notan que en el caso extremo de una feature por par estado-acción se pueden aprender recompensas totalmente generales, pero eso no escala.
- **Requiere resolver muchos MDPs.** Cada iteración invoca un solver de RL completo (paso 4) sobre el `MDP\R` con la recompensa candidata. Aunque el número de iteraciones es pequeño (Teorema 1), **cada una implica resolver un MDP entero**, lo que es caro en dominios grandes; en el simulador de conducción el paso de RL se resolvió sobre una versión discretizada. Además se supone conocido el modelo de transiciones (o al menos un simulador): el algoritmo no aprende la dinámica de las demostraciones.
- **Selección final no automática por defecto.** La versión más limpia devuelve un *conjunto* de políticas; obtener una sola requiere o inspección humana o el QP de proyección sobre la envolvente convexa (que produce una política *mixta* estocástica, no una determinística única).

## 8. Legado

Este paper es un pilar fundacional del IRL moderno y del aprendizaje por imitación basado en recompensa. Su influencia se ve en dos líneas principales:

- **Maximum Entropy IRL** (Ziebart et al., 2008). El *matching* de feature expectations de Abbeel & Ng deja abierta la ambigüedad de qué política elegir entre las muchas que igualan $\mu_E$. MaxEnt IRL resuelve esa ambigüedad con un principio de máxima entropía: entre todas las distribuciones sobre trayectorias que igualan las feature expectations del experto, elegir la de máxima entropía. Esto da un modelo probabilístico bien definido y se volvió el estándar del IRL clásico.
- **Generative Adversarial Imitation Learning (GAIL)** (Ho & Ermon, 2016). Aquí el linaje es directo: el paso de max-margin de Abbeel & Ng —separar las feature expectations del experto de las del aprendiz con un clasificador tipo SVM— es reemplazado por un **discriminador de red neuronal** que distingue trayectorias del experto de las del aprendiz, mientras la política se entrena para "engañarlo" (esquema GAN). GAIL puede verse como la versión adversarial y no lineal del matching de ocupación/feature expectations que este paper inauguró, evitando el costoso bucle de resolver un MDP por iteración.

Más ampliamente, la idea de **matching de feature/occupancy expectations** como objetivo del aprendizaje por imitación —en lugar de clonar acciones o recuperar la recompensa exacta— es una de las contribuciones conceptuales duraderas de este trabajo.

## 9. Conexión con la Clase 33 (Aprendizaje por Imitación y Aprendizaje Reforzado Inverso)

La Clase 33 del profesor Rodrigo Toro Icarte cubre el espectro del aprendizaje por imitación y el IRL: generalización en RL, clonación conductual (behavioral cloning, BC), DAgger, IRL, y la comparación RL vs. imitación. Este paper es citado **explícitamente en la slide 24** como el ejemplo canónico de "aprender a conducir usando IRL" — y en efecto el experimento del simulador de conducción con cinco estilos es esa demostración.

Ubicado en el mapa de la clase, el paper permite contrastar tres enfoques:

| Enfoque | Qué aprende | Debilidad principal |
|---|---|---|
| **Behavioral cloning** | Mapeo directo estado→acción (supervisado) | Se rompe fuera de la distribución de demostraciones (compounding errors); no generaliza a situaciones nuevas |
| **DAgger** | Igual, pero con corrección iterativa del experto | Requiere consultar al experto durante el entrenamiento |
| **IRL / apprenticeship (este paper)** | La recompensa (implícita) que explica al experto, vía matching de feature expectations | Requiere resolver muchos MDPs; supone recompensa lineal |

La lección central que este paper aporta a la clase es la distinción entre **imitar la política** e **inferir el objetivo**. El BC clona *qué hizo* el experto; el IRL infiere *por qué* lo hizo (la recompensa), lo que da una representación más transferible y robusta ante cambios de situación —exactamente el argumento del ejemplo de conducción, donde el patrón de tráfico cambia en cada episodio y copiar la trayectoria fracasa. La idea de que **basta igualar el comportamiento agregado (feature expectations) para igualar el desempeño, sin recuperar la recompensa verdadera**, es la contribución teórica que hace del IRL una herramienta práctica y no solo una curiosidad mal condicionada.

**Enlaces internos:**

- Clase: [/clases/clase-33](/clases/clase-33) — Aprendizaje por Imitación y Aprendizaje Reforzado Inverso.
- Fundamento transversal: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado) — marco MDP, retorno, políticas, funciones de valor.
- Clase precedente de RL: [/clases/clase-31](/clases/clase-31) — Aprendizaje Reforzado (Q-Learning, DQN, actor-critic), donde se define el RL "forward" que este algoritmo invoca en cada iteración.

## 10. Nota final: conexión con salud, MDM y record linkage

Para el lector experto en FHIR y *master data management*, la idea central de este paper —**igualar las feature expectations del experto en lugar de replicar cada una de sus decisiones**— tiene una traducción casi literal al problema de calibrar un *scorer* de record linkage. Un equipo de expertos en gestión de identidad de pacientes resuelve a diario miles de decisiones de *match/no-match* sobre pares de registros, pero rara vez puede escribir la "función de recompensa" exacta que pondera cada evidencia (coincidencia de nombre, fecha de nacimiento, RUT, dirección, teléfono). Igual que Abbeel & Ng con la conducción, el conocimiento experto es más fácil de *demostrar* que de *especificar*. El análogo del *feature matching* sería calibrar los pesos de un scorer de matching (por ejemplo, un modelo Fellegi-Sunter o un GBM) de modo que **el comportamiento agregado del sistema reproduzca las estadísticas de features de las decisiones expertas** —tasas de aceptación por rango de similitud de nombre, distribución de umbrales efectivos, proporción de casos derivados a revisión manual— sin exigir que el sistema replique cada dictamen individual, que muchas veces es ruidoso o inconsistente entre revisores. La garantía del paper —que igualar las expectativas agregadas dentro de $\epsilon$ acota la diferencia de "desempeño" para *cualquier* ponderación verdadera de la evidencia— es tranquilizadora en este contexto: significa que un scorer calibrado sobre el comportamiento agregado del panel de expertos se desempeñará de forma cercana a ese panel bajo el verdadero criterio de calidad de matching, aunque nunca lleguemos a escribir explícitamente cuál es ese criterio.
