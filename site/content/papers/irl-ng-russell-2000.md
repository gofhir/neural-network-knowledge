---
title: "IRL: Algorithms for Inverse Reinforcement Learning (2000)"
weight: 368
math: true
---

{{< paper-card
    title="Algorithms for Inverse Reinforcement Learning"
    authors="Andrew Y. Ng, Stuart Russell (UC Berkeley)"
    year="2000"
    venue="ICML 2000"
    pdf="/papers/irl-ng-russell-2000.pdf" >}}
Este es el **paper fundacional del aprendizaje reforzado inverso (IRL)**. Mientras el RL estándar toma una recompensa y produce una política óptima, el IRL invierte la flecha: dado el comportamiento óptimo observado de un agente en un MDP, **recupera la función de recompensa que ese agente está optimizando**. Ng y Russell hacen tres cosas. Primero, caracterizan en forma cerrada *todas* las recompensas que hacen óptima a una política dada en un MDP finito. Segundo, identifican el problema central: esa caracterización es **degenerada** —incontables recompensas, incluida $R = 0$, explican cualquier conducta—, de modo que el IRL está **mal planteado (ill-posed)**. Tercero, introducen **heurísticas de margen** que seleccionan una recompensa "significativa" y se resuelven como **programas lineales**. De aquí nacen el [apprenticeship learning](/papers/apprenticeship-abbeel-ng-2004), el MaxEnt IRL, GAIL y, conceptualmente, el RLHF que alinea los LLM de hoy. Es la piedra angular teórica de la [Clase 33](/clases/clase-33).
{{< /paper-card >}}

---

## Contexto: por qué el problema está mal planteado

El [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado) presupone que la recompensa está dada. Pero Ng y Russell parten de dos observaciones. La científica: el RL se usa como modelo del aprendizaje animal y humano, y esa literatura *asume* la recompensa conocida. Al examinar conducta natural —una abeja que pondera néctar contra distancia, tiempo y riesgo de depredadores— la recompensa **multiatributo debería tratarse como incógnita a determinar empíricamente**, porque nadie articula sus pesos relativos a priori. La de ingeniería: para tareas como "conducir bien" es más fácil observar a un experto que escribir su recompensa; conviene entonces **recuperar la recompensa del experto** en vez de imitar su política, porque la recompensa es una descripción más parsimoniosa, robusta y transferible de la tarea.

El obstáculo es intrínseco. Muchísimas recompensas distintas explican la misma conducta. El caso extremo es $R = 0$ (o cualquier vector constante): si la recompensa es idéntica sin importar la acción, *toda* política es óptima, incluida la observada. Esta **degeneración** no se resuelve dentro del enunciado original del IRL; hay que añadir criterios externos que rompan la ambigüedad. Ese es el hilo conductor del [aprendizaje reforzado inverso](/fundamentos/aprendizaje-reforzado-inverso).

## Contribución: caracterización, degeneración y programas lineales

Un MDP finito es la tupla $(S, A, \{P_{sa}\}, \gamma, R)$. Renombrando acciones, se asume que la política observada es $\pi(s) \equiv a_1$. El resultado central es el **Teorema 3**: $\pi \equiv a_1$ es óptima **si y solo si**, para toda acción $a \neq a_1$, la recompensa satisface

$$\left(\boldsymbol{P}_{a_1} - \boldsymbol{P}_a\right)\left(\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}\right)^{-1} \boldsymbol{R} \succeq 0.$$

La derivación es corta: de la ecuación de Bellman vectorial $\boldsymbol{V}^\pi = \left(\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}\right)^{-1} \boldsymbol{R}$ (la matriz $\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}$ es siempre invertible porque $\gamma < 1$ mantiene los autovalores dentro del círculo unitario) y la condición de optimalidad de Bellman. Reemplazando por desigualdades estrictas ($\succ 0$) se obtiene la condición para que $\pi$ sea la política óptima **única**.

Este teorema describe todo el conjunto solución, y revela de inmediato la degeneración: $\boldsymbol{R} = 0$ siempre lo satisface, y para casi todo MDP hay infinitas soluciones. La respuesta de Ng y Russell es una **heurística de margen**: entre todas las $\boldsymbol{R}$ válidas, elegir la que hace que desviarse un solo paso de $\pi$ sea lo más costoso posible. Formalmente, maximizar la suma sobre estados del margen entre la acción observada y la mejor alternativa,

$$\sum_{s \in S} \left( Q^\pi(s, a_1) - \max_{a \neq a_1} Q^\pi(s,a) \right),$$

restando una **penalización $\ell_1$**, $-\lambda \lVert \boldsymbol{R} \rVert_1$, que fuerza recompensas dispersas ("simples"). Todo esto se formula como un **programa lineal** eficientemente resoluble. Aparece un fenómeno de transición de fase: existe un umbral $\lambda_0$ tal que la $\boldsymbol{R}$ óptima está acotada lejos de $0$ para $\lambda < \lambda_0$ y colapsa a $\boldsymbol{R} = 0$ para $\lambda > \lambda_0$; la elección natural es $\lambda = \lambda_0^{-}$, la recompensa más simple que aún explica la conducta.

## Las tres formulaciones

**Caso 1 — MDP finito.** Política completa conocida y espacio de estados tabular: el LP anterior con margen y penalización $\ell_1$.

**Caso 2 — Aproximación lineal de funciones.** Para espacios grandes o infinitos ($S = \mathbb{R}^n$), la recompensa se aproxima como combinación lineal de $d$ funciones base fijas, $R(s) = \alpha_1 \phi_1(s) + \cdots + \alpha_d \phi_d(s)$. Por linealidad, la función de valor hereda la misma combinación, y la condición de optimalidad se vuelve un conjunto de restricciones lineales en los $\alpha_i$. Como hay infinitas restricciones, se **muestrea un subconjunto finito $S_0$** de estados; y como puede no existir ninguna recompensa no trivial en esa clase que haga a $\pi$ exactamente óptima, se **relajan** las restricciones penalizando las violaciones (peso $1$ si se cumple, peso $2$ si se viola).

**Caso 3 — Trayectorias muestreadas.** El escenario realista: la política solo se conoce a través de un conjunto finito de **trayectorias**, sin modelo explícito del MDP. Se estima el valor por **Monte Carlo** —los retornos empíricos descontados bajo cada función base son, en esencia, las *feature expectations*— y se corre un algoritmo **iterativo con generación incremental de restricciones**: en cada paso se busca una recompensa que haga a la política experta al menos tan buena como todas las candidatas halladas hasta ahora, se calcula la política óptima bajo esa recompensa, se agrega al conjunto y se repite. Cada nueva política óptima se vuelve una restricción que empuja la recompensa a distinguir al experto de alternativas cada vez más competitivas.

## Resultados

Los tres algoritmos se validan sobre gridworlds y mountain-car. En el **gridworld discreto $5 \times 5$** (acciones ruidosas, 30 % de movimiento aleatorio), correr sin penalización ($\lambda = 0$) recupera la mayor parte de la estructura de la recompensa pero con un resultado "abultado"; con $\lambda = 1.05$ —justo bajo la transición de fase— la recompensa recuperada queda muy cercana a la verdadera. En **mountain-car** (espacio continuo, 26 funciones base gaussianas sobre la posición) la solución captura casi perfectamente la estructura $R = -c$ constante, con la variante de recompensa positiva en el fondo de la colina recuperada salvo un artefacto atribuido al efecto de "salirse disparado". En el **gridworld continuo desde trayectorias** ($m = 5000$ trayectorias de 30 pasos), la solución era razonable ya tras **1 iteración** y se estabilizaba hacia las **15**; la fracción de estados con acción distinta a la óptima verdadera fue típicamente entre **3 % y 10 %**, y hacia las 15 iteraciones no hubo diferencia estadísticamente significativa entre el valor de la política óptima verdadera ($\approx 6.65$) y el de la política óptima de la recompensa ajustada.

## Limitaciones

- **La ambigüedad se maneja, no se elimina.** Las heurísticas de margen escogen *una* recompensa, pero esa elección es una preferencia inductiva (margen máximo, dispersión $\ell_1$), no una consecuencia del problema.
- **Optimalidad del experto.** El método asume que la conducta observada es óptima o casi; ruido del observador, subóptimalidad y múltiples políticas óptimas quedan como problemas abiertos.
- **Observabilidad parcial.** El enfoque es para MDPs completamente observables; la extensión a POMDPs queda explícitamente abierta.
- **Dependencia de las features y escala.** En los casos 2 y 3 la calidad está limitada por la expresividad de la base $\phi$; los autores concluyen que el IRL es soluble "al menos para dominios discretos y continuos de tamaño moderado".

## Por qué importa para la Clase 33

La [Clase 33](/clases/clase-33) gira en torno a **cómo se aprende y se especifica el objetivo** de un agente, contrastando clonación conductual, DAgger e IRL. Ng y Russell aportan la pieza teórica fundacional: demuestran que inferir el objetivo desde la conducta es tratable, caracterizable en forma cerrada y reducible a programación lineal, pero **intrínsecamente ambiguo**, lo que obliga a introducir preferencias inductivas explícitas. Entender este paper es entender por qué el RL moderno se preocupa tanto del *reward design* y el *reward modeling*.

El linaje que abre es directo. El [apprenticeship learning de Abbeel y Ng (2004)](/papers/apprenticeship-abbeel-ng-2004) retoma la idea de igualar *feature expectations* entre experto y aprendiz. El **MaxEnt IRL** (Ziebart, 2008) reemplaza la arbitrariedad del margen por un principio de máxima entropía. **GAIL** (Ho & Ermon, 2016) conecta el IRL con las GANs. Y el **RLHF** que alinea los LLM es, conceptualmente, IRL: en vez de programar la recompensa "buena respuesta", se infiere un modelo de recompensa desde preferencias humanas y se optimiza la política contra él. Todos comparten la premisa de Ng y Russell: **la recompensa, no la política, es la descripción más parsimoniosa, robusta y transferible de una tarea** —el eje del [aprendizaje reforzado inverso](/fundamentos/aprendizaje-reforzado-inverso).
