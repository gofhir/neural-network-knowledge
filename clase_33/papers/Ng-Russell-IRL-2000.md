# Algorithms for Inverse Reinforcement Learning (Ng & Russell) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Algorithms for Inverse Reinforcement Learning*.
- **Autores:** Andrew Y. Ng y Stuart Russell. Ambos en la **CS Division, U.C. Berkeley** (Berkeley, CA 94720). Correos: `ang@cs.berkeley.edu`, `russell@cs.berkeley.edu`.
- **Venue:** *Proceedings of the Seventeenth International Conference on Machine Learning (ICML 2000)*.
- **Año:** 2000.
- **Linaje:** el trabajo formaliza una idea planteada por Russell (1998, *Learning agents for uncertain environments*) y conecta con la estimación estructural de MDPs en econometría (Rust, 1994; Sargent, 1978) y con el problema inverso de Kalman en teoría de control. Es el **paper fundacional del Inverse Reinforcement Learning (IRL)**.

El paper define y ataca por primera vez de forma algorítmica el problema del **aprendizaje reforzado inverso**: dado el comportamiento óptimo observado de un agente en un proceso de decisión de Markov (MDP), **recuperar la función de recompensa que ese agente está optimizando**. Es el problema inverso del RL estándar: mientras el RL toma una recompensa y produce una política óptima, el IRL toma una política (u observaciones de ella) y produce la recompensa que la hace óptima.

El aporte técnico tiene tres piezas. Primero, una **caracterización completa y cerrada** del conjunto de todas las recompensas que hacen óptima a una política $\pi$ dada en un MDP finito, expresada como un sistema de desigualdades matriciales lineales. Segundo, la identificación del **problema central del IRL: la degeneración**. El conjunto de soluciones es enorme y contiene soluciones triviales sin contenido —notablemente $R = 0$, que hace óptima a *cualquier* política—, de modo que el problema está **mal planteado (ill-posed)**. Tercero, una batería de **heurísticas de margen** que seleccionan, entre las infinitas soluciones, una recompensa "significativa", y que se traducen en **programas lineales (LP)** eficientemente resolubles.

Ng y Russell desarrollan la solución en tres escenarios de complejidad creciente: (1) MDP finito con política completa conocida; (2) espacios de estado grandes o infinitos, con la recompensa aproximada como combinación lineal de funciones base; y (3) el caso realista en que la política solo se conoce a través de un conjunto finito de **trayectorias muestreadas**, sin modelo explícito del ambiente. Los tres casos se validan sobre gridworlds discretos y continuos y sobre el problema clásico del *mountain-car*.

Para la **Clase 33** este paper es la piedra angular: es el punto de partida de toda la línea que va desde el *apprenticeship learning* de Abbeel & Ng (2004), pasando por el *Maximum Entropy IRL* de Ziebart (2008), hasta GAIL (2016) y, conceptualmente, el **RLHF** que alinea los grandes modelos de lenguaje actuales. Todos comparten la premisa de Ng y Russell: **la recompensa, no la política, es la descripción más parsimoniosa, robusta y transferible de una tarea**.

## 2. Contexto y motivación: por qué el problema está mal planteado

El paper abre distinguiendo dos fuentes de motivación. La primera es científica: el RL se usa como **modelo computacional del aprendizaje animal y humano** (Watkins, 1989; Schmajuk & Zanutto, 1997; y evidencia neurofisiológica en forrajeo de abejas —Montague et al., 1995— y vocalización de aves cantoras —Doya & Sejnowski, 1995). Pero esa literatura *asume la recompensa fija y conocida*: por ejemplo, los modelos de forrajeo de abejas suponen que la recompensa en cada flor es una función saturante simple del contenido de néctar. Ng y Russell argumentan que, al examinar comportamiento natural, **la función de recompensa debe tratarse como una incógnita a determinar empíricamente**, sobre todo cuando es *multiatributo*: una abeja que pondera ingesta de néctar contra distancia de vuelo, tiempo y riesgo de viento y depredadores no revela sus pesos relativos *a priori*. El vínculo con la econometría es directo: la evaluación de utilidad multiatributo (Keeney & Raiffa, 1976) estudia decisiones *one-shot*; el caso secuencial lo abordaron primero Sargent (1978) —al estimar el costo efectivo de contratación de una firma asumiendo conducta racional— y luego la estimación estructural de MDPs (Rust, 1994).

La segunda motivación es de ingeniería: el **aprendizaje por aprendizaje (apprenticeship learning)**. Un diseñador de agentes puede tener solo una idea muy vaga de la recompensa cuya optimización generaría el comportamiento "deseable" (piénsese en la tarea de "conducir bien"). En vez de imitar directamente la política de un experto —el enfoque del *imitation learning*, que aprende un mapeo directo de estados a acciones—, Ng y Russell proponen **recuperar la recompensa del experto y usarla para generar comportamiento**. El argumento clave: la recompensa es una descripción **más parsimoniosa, robusta y transferible** de la tarea que la política. Todo el campo del RL se funda en la presuposición de que la recompensa, no la política, es la definición más sucinta de la tarea; luego el IRL puede, en algunos dominios, ser una forma efectiva de apprenticeship learning.

El problema, sin embargo, es intrínsecamente **mal planteado**. Muchísimas recompensas distintas explican la misma conducta observada. El caso extremo es $R = 0$ (y cualquier vector constante): si la recompensa es la misma sin importar la acción, *toda* política es óptima, incluida la observada. Esta degeneración no se resuelve dentro del enunciado original del IRL; hace falta introducir criterios adicionales —las heurísticas de margen— que rompan la ambigüedad seleccionando la recompensa que "mejor explica" por qué la conducta observada es óptima y no otra.

## 3. Contribución central

Las contribuciones concretas del paper son:

1. **Caracterización del conjunto solución** (Teorema 3): condición necesaria y suficiente, en forma de desigualdad matricial, para que una política sea óptima bajo una recompensa dada en MDPs finitos.
2. **Diagnóstico de la degeneración** y propuesta de **heurísticas de margen** que la eliminan, favoreciendo la recompensa que hace la conducta observada máximamente distinguible de las alternativas subóptimas.
3. **Formulación como programa lineal** en los tres escenarios, con un término de penalización $\ell_1$ (parámetro $\lambda$) que además fuerza recompensas "simples" (dispersas).
4. **Aproximación lineal de funciones** para escalar a espacios de estado grandes/infinitos.
5. **Algoritmo iterativo desde trayectorias muestreadas**, que no requiere modelo explícito del MDP y usa estimación Monte Carlo con generación incremental de restricciones.
6. **Validación experimental** en gridworlds (discreto y continuo) y mountain-car, mostrando recuperación fiel de la estructura de recompensa.

## 4. Método formal

### 4.1. Notación y fundamentos de MDP

Un MDP finito es una tupla $(S, A, \{P_{sa}\}, \gamma, R)$, donde $S$ es un conjunto finito de $N$ estados, $A = \{a_1, \dots, a_k\}$ un conjunto de $k$ acciones, $P_{sa}(\cdot)$ las probabilidades de transición al tomar la acción $a$ en el estado $s$, $\gamma \in [0,1)$ el factor de descuento y $R : S \mapsto \mathbb{R}$ la recompensa, acotada en valor absoluto por $R_{\max}$. Por simplicidad se escribe $R(s)$ en vez de $R(s,a)$ (la extensión es trivial). Una política es un mapa $\pi : S \mapsto A$, y su función de valor evaluada en un estado $s_1$ es

$$V^\pi(s_1) = \mathbb{E}\left[R(s_1) + \gamma R(s_2) + \gamma^2 R(s_3) + \cdots \mid \pi\right],$$

donde la esperanza es sobre la distribución de la secuencia de estados que se atraviesa al ejecutar $\pi$ desde $s_1$. La función Q correspondiente es

$$Q^\pi(s,a) = R(s) + \gamma\, \mathbb{E}_{s' \sim P_{sa}(\cdot)}\left[V^\pi(s')\right].$$

El paper recuerda dos resultados clásicos que sostienen todo lo demás. Las **ecuaciones de Bellman** (Teorema 1):

$$V^\pi(s) = R(s) + \gamma \sum_{s'} P_{s\pi(s)}(s')\, V^\pi(s'), \qquad Q^\pi(s,a) = R(s) + \gamma \sum_{s'} P_{sa}(s')\, V^\pi(s').$$

Y la **optimalidad de Bellman** (Teorema 2): $\pi$ es óptima si y solo si, para todo $s$, $\pi(s) \in \arg\max_{a \in A} Q^\pi(s,a)$.

Para el tratamiento vectorial, se fija una enumeración de los estados de $1$ a $N$. Los vectores $\boldsymbol{R}$ y $\boldsymbol{V}^\pi$ tienen dimensión $N$ (recompensa y valor por estado); $\boldsymbol{P}_a$ es la matriz $N \times N$ cuyo elemento $(i,j)$ es la probabilidad de transitar al estado $j$ al tomar la acción $a$ en el estado $i$. Los símbolos $\prec$ y $\preceq$ denotan desigualdad vectorial (estricta y no estricta, componente a componente). Sin pérdida de generalidad, renombrando acciones, se asume que la política observada es $\pi(s) \equiv a_1$ (constante); esto es solo para simplificar notación.

### 4.2. Caso 1 — IRL en MDPs finitos: caracterización del conjunto solución

El resultado central es el **Teorema 3**: dado un espacio de estados finito $S$, acciones $A = \{a_1, \dots, a_k\}$, matrices de transición $\{\boldsymbol{P}_a\}$ y descuento $\gamma \in (0,1)$, la política $\pi(s) \equiv a_1$ es óptima **si y solo si**, para todas las acciones $a = a_2, \dots, a_k$, la recompensa $\boldsymbol{R}$ satisface

$$\left(\boldsymbol{P}_{a_1} - \boldsymbol{P}_a\right)\left(\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}\right)^{-1} \boldsymbol{R} \succeq 0. \tag{4}$$

La demostración es elegante. Como $\pi(s) \equiv a_1$, la ecuación de Bellman en forma vectorial es $\boldsymbol{V}^\pi = \boldsymbol{R} + \gamma \boldsymbol{P}_{a_1} \boldsymbol{V}^\pi$, de donde

$$\boldsymbol{V}^\pi = \left(\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}\right)^{-1} \boldsymbol{R}. \tag{5}$$

La matriz $\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}$ es **siempre invertible**: como $\boldsymbol{P}_{a_1}$ es una matriz de transición, sus autovalores están dentro del círculo unitario del plano complejo; con $\gamma < 1$, los autovalores de $\gamma \boldsymbol{P}_{a_1}$ quedan en el interior del círculo (en particular, $1$ no es autovalor), de modo que $\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}$ no tiene autovalores nulos y no es singular. Sustituyendo la condición de optimalidad de Bellman, $\pi \equiv a_1$ es óptima si y solo si, para todo $s$, $\sum_{s'} P_{sa_1}(s') V^\pi(s') \geq \sum_{s'} P_{sa}(s') V^\pi(s')$ para toda $a$, lo que en forma vectorial es $\boldsymbol{P}_{a_1} \boldsymbol{V}^\pi \succeq \boldsymbol{P}_a \boldsymbol{V}^\pi$; y usando (5) se obtiene exactamente (4).

Un **corolario** importante (Remark): reemplazando todas las desigualdades de la demostración por estrictas, la condición $\left(\boldsymbol{P}_{a_1} - \boldsymbol{P}_a\right)\left(\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}\right)^{-1} \boldsymbol{R} \succ 0$ es necesaria y suficiente para que $\pi \equiv a_1$ sea la **única** política óptima.

**La degeneración.** El Teorema 3 caracteriza *todo* el conjunto de recompensas válidas, pero revela de inmediato dos problemas. Primero, $\boldsymbol{R} = 0$ (y cualquier vector constante) siempre satisface (4): si la recompensa es la misma sin importar la acción, cualquier política —incluida $\pi \equiv a_1$— es óptima. Segundo, para la mayoría de los MDPs hay muchísimas $\boldsymbol{R}$ que cumplen (4). Exigir unicidad de la política óptima alivia lo primero pero no es del todo satisfactorio, porque vectores de recompensa arbitrariamente cercanos a $0$ siguen siendo soluciones. La respuesta no está en el enunciado original del IRL, sino en criterios adicionales.

### 4.3. La heurística de margen y el LP con penalización $\ell_1$

La idea natural: entre todas las $\boldsymbol{R}$ que satisfacen (4), elegir la que hace que **desviarse un solo paso de $\pi$ sea lo más costoso posible**. Formalmente, de todas las funciones $R$ que satisfacen (4) con $|R(s)| \leq R_{\max}$, se busca maximizar la suma de las diferencias entre la calidad de la acción óptima y la de la siguiente mejor:

$$\sum_{s \in S} \left( Q^\pi(s, a_1) - \max_{a \in A \setminus a_1} Q^\pi(s,a) \right). \tag{6}$$

Es decir, se maximiza el **margen** entre la acción observada y la mejor alternativa. Adicionalmente, si se cree que —en igualdad de condiciones— soluciones con recompensas mayoritariamente pequeñas son "más simples" y preferibles, se agrega opcionalmente una penalización tipo *weight-decay* $-\lambda \lVert \boldsymbol{R} \rVert_1$, donde $\lambda$ es un coeficiente ajustable que balancea entre tener recompensas pequeñas y maximizar (6). El uso del término $\ell_1$ tiene un efecto notable: para $\lambda$ suficientemente grande, $\boldsymbol{R}$ será distinta de cero en solo unos pocos estados (dispersión), consistente con la idea de recompensa "simple".

El paper observa un fenómeno de **transición de fase**: asumiendo que la solución no es ya degenerada en $\lambda = 0$, al aumentar $\lambda$ existe un umbral $\lambda_0$ tal que la $\boldsymbol{R}$ óptima está acotada lejos de $0$ para $\lambda < \lambda_0$ y colapsa a $\boldsymbol{R} = 0$ para $\lambda > \lambda_0$. La elección automática natural es $\lambda = \lambda_0^{-}$ (un valor justo antes de la transición, hallable por búsqueda binaria sobre $\lambda$), pues da la $\boldsymbol{R}$ "más simple" (mayor coeficiente de penalización) que todavía no es cero y por tanto sí explica, al menos parcialmente, por qué $\pi$ es óptima.

Reuniendo todo, el problema de optimización es:

$$
\begin{aligned}
\text{maximizar} \quad & \sum_{i=1}^{N} \min_{a \in \{a_2, \dots, a_k\}} \left\{ \left(\boldsymbol{P}_{a_1}(i) - \boldsymbol{P}_a(i)\right)\left(\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}\right)^{-1} \boldsymbol{R} \right\} - \lambda \lVert \boldsymbol{R} \rVert_1 \\
\text{s.a.} \quad & \left(\boldsymbol{P}_{a_1} - \boldsymbol{P}_a\right)\left(\boldsymbol{I} - \gamma \boldsymbol{P}_{a_1}\right)^{-1} \boldsymbol{R} \succeq 0, \quad \forall a \in A \setminus a_1 \\
& |\boldsymbol{R}_i| \leq R_{\max}, \quad i = 1, \dots, N
\end{aligned}
$$

donde $\boldsymbol{P}_a(i)$ denota la $i$-ésima fila de $\boldsymbol{P}_a$. Esto se formula fácilmente como **programa lineal** y se resuelve eficientemente. El $\min$ sobre acciones dentro de la suma captura, para cada estado, el peor margen (la alternativa más competitiva), que es lo que se busca maximizar.

### 4.4. Caso 2 — Aproximación lineal de funciones en espacios grandes/infinitos

Para espacios infinitos (el paper se concentra en $S = \mathbb{R}^n$), una representación tabular de $\boldsymbol{R}$ es inviable. Ng y Russell aproximan la recompensa como combinación lineal de $d$ funciones base fijas, conocidas y acotadas:

$$R(s) = \alpha_1 \phi_1(s) + \alpha_2 \phi_2(s) + \cdots + \alpha_d \phi_d(s), \tag{8}$$

donde $\phi_1, \dots, \phi_d$ son las *features* (mapeos de $S$ a $\mathbb{R}$) y los $\alpha_i$ los parámetros a ajustar. Por linealidad de la esperanza, si $V_i^\pi$ es la función de valor de $\pi$ en el MDP cuya recompensa es $R = \phi_i$, entonces la función de valor bajo $R$ de (8) es

$$V^\pi = \alpha_1 V_1^\pi + \cdots + \alpha_d V_d^\pi. \tag{9}$$

Aplicando el mismo argumento del Teorema 3, la generalización de la condición (4) para que $\pi(s) \equiv a_1$ sea óptima es

$$\mathbb{E}_{s' \sim P_{sa_1}}\left[V^\pi(s')\right] \geq \mathbb{E}_{s' \sim P_{sa}}\left[V^\pi(s')\right] \tag{10}$$

para todos los estados $s$ y todas las acciones $a \in A \setminus a_1$. Como por (9) $V^\pi$ es lineal en los $\alpha_i$, esto es un conjunto de restricciones lineales en los $\alpha_i$.

Aparecen dos dificultades. Primera: en espacios infinitos hay **infinitas restricciones** de la forma (10), imposibles de verificar todas; se **muestrea un subconjunto finito grande $S_0$** de estados y se imponen las restricciones solo en $s \in S_0$. Segunda —más sutil—: al restringir $R$ a la clase de aproximación lineal (8), puede que ya **no exista ninguna recompensa no trivial** en esa clase que haga a $\pi$ exactamente óptima (más allá de $R = 0$). Como compromiso, se **relajan** algunas restricciones (10) pagando una penalización cuando se violan. El LP final es:

$$
\begin{aligned}
\text{maximizar} \quad & \sum_{s \in S_0} \min_{a \in \{a_2, \dots, a_k\}} \left\{ p\left( \mathbb{E}_{s' \sim P_{sa_1}}\left[V^\pi(s')\right] - \mathbb{E}_{s' \sim P_{sa}}\left[V^\pi(s')\right] \right) \right\} \\
\text{s.a.} \quad & |\alpha_i| \leq 1, \quad i = 1, \dots, d
\end{aligned}
$$

donde $p(x) = x$ si $x \geq 0$ y $p(x) = 2x$ en caso contrario. Es decir, se premia con peso $1$ la satisfacción de la restricción y se penaliza con peso $2$ su violación (el factor $2$ es un peso de penalización elegido heurísticamente, al que los resultados no fueron muy sensibles; valores moderadamente mayores dieron resultados similares).

### 4.5. Caso 3 — IRL desde trayectorias muestreadas

El escenario más realista: la política $\pi$ se conoce **solo a través de un conjunto de trayectorias** en el espacio de estados, y **no se dispone de un modelo explícito del MDP** (aunque sí se asume la capacidad de hallar una política óptima bajo cualquier recompensa que se proponga). Se fija una distribución inicial $D$ y, por simplicidad, un único estado de inicio $s_0$ (esto es sin pérdida de generalidad, pues $s_0$ puede ser un estado "ficticio" cuya distribución de siguiente-estado bajo cualquier acción es $D$). La recompensa se expresa de nuevo con la clase de aproximación lineal. El objetivo es hallar $R$ tal que $\pi$ maximice $\mathbb{E}_{s_0 \sim D}[V^\pi(s_0)]$.

El corazón del método es estimar $V^\pi(s_0)$ por **Monte Carlo**. Se ejecutan $m$ trayectorias bajo $\pi$; para cada $i = 1, \dots, d$ se define $\hat{V}_i^\pi(s_0)$ como el retorno empírico promedio que se habría obtenido si la recompensa hubiese sido $R = \phi_i$. Con $m = 1$ trayectoria que visita la secuencia $(s_0, s_1, \dots)$:

$$\hat{V}_i^\pi(s_0) = \phi_i(s_0) + \gamma \phi_i(s_1) + \gamma^2 \phi_i(s_2) + \cdots$$

(en general se promedia sobre las $m$ trayectorias). Estas son, en esencia, las **feature expectations** estimadas empíricamente. Entonces, para cualquier valor de los $\alpha_i$, el estimador natural de $V^\pi(s_0)$ es

$$\hat{V}^\pi(s_0) = \alpha_1 \hat{V}_1^\pi(s_0) + \cdots + \alpha_d \hat{V}_d^\pi(s_0), \tag{11}$$

justificado por $V^\pi(s_0) = \alpha_1 V_1^\pi(s_0) + \cdots + \alpha_d V_d^\pi(s_0)$. En la práctica, las trayectorias se truncan tras $H$ pasos; con $H = H_\epsilon = \log_\gamma\!\left(\epsilon(1-\gamma)/R_{\max}\right)$ (el *tiempo de horizonte-$\epsilon$*), el descuento hace que el truncamiento introduzca a lo sumo un error $\epsilon$ en las estimaciones. El paper también menciona la alternativa de Kearns et al. (1999) para obtener un estimador insesgado del retorno de horizonte infinito ejecutando trayectorias de largo esperado $O(H_\epsilon)$.

El algoritmo es **iterativo, con generación incremental de restricciones**. Se estiman los valores para $\pi^*$ (la política experta) y para una política base $\pi_1$ elegida al azar. En el paso inductivo, se tiene un conjunto $\{\pi_1, \dots, \pi_k\}$ y se busca un ajuste de los $\alpha_i$ tal que la recompensa resultante satisfaga (idealmente)

$$V^{\pi^*}(s_0) \geq V^{\pi_i}(s_0), \quad i = 1, \dots, k, \tag{12}$$

es decir, que la recompensa haga a la política experta al menos tan buena como todas las políticas candidatas encontradas hasta ahora. Con la misma relajación anterior, la optimización es:

$$
\begin{aligned}
\text{maximizar} \quad & \sum_{i=1}^{k} p\left( \hat{V}^{\pi^*}(s_0) - \hat{V}^{\pi_i}(s_0) \right) \\
\text{s.a.} \quad & |\alpha_i| \leq 1, \quad i = 1, \dots, d
\end{aligned}
$$

con el mismo $p(\cdot)$ (peso $1$ / peso $2$); tanto $\hat{V}^{\pi^*}(s_0)$ como $\hat{V}^{\pi_i}(s_0)$ son funciones lineales (implícitas) de los $\alpha_i$ vía (11), por lo que el problema es un LP. Resuelto el LP, se obtiene una nueva recompensa $R = \alpha_1 \phi_1 + \cdots + \alpha_d \phi_d$; se calcula la política $\pi_{k+1}$ que **maximiza $V^\pi(s_0)$ bajo esa $R$**, se agrega al conjunto y se repite (por un número grande de iteraciones, hasta hallar una $R$ satisfactoria). La generación incremental de restricciones es la clave: cada nueva política óptima bajo la recompensa candidata se convierte en una nueva restricción que empuja la recompensa a distinguir al experto de las alternativas cada vez más competitivas.

## 5. Experimentos

Ng y Russell validan los tres algoritmos sobre tres dominios; reporto solo las cifras visibles en las figuras y el texto de las imágenes.

**Gridworld discreto $5 \times 5$.** El agente parte de la casilla inferior-izquierda y debe llegar a la casilla superior-derecha (absorbente), donde recibe recompensa $1$. Las acciones corresponden a moverse en las cuatro direcciones cardinales, pero son ruidosas: hay un **30 % de probabilidad** de moverse en una dirección aleatoria. La Figura 1 muestra (arriba) la política óptima y (abajo) la recompensa verdadera —un pico de $1$ en la esquina objetivo, $0$ en el resto. Corriendo el algoritmo de la Sección 3.2 **sin término de penalización** ($\lambda = 0$), la Figura 2 (arriba) muestra que se recupera la mayor parte de la estructura de la recompensa, aunque el resultado es "abultado" (*bumpy*), en parte por rupturas de simetría arbitrarias en la política elegida. Con $\lambda = 1.05$ (justo por debajo de la transición de fase), la Figura 2 (abajo) da una recompensa muy cercana a la verdadera. El paper anota en nota al pie que valores intermedios como $\lambda = 0.5$ **no** dieron funciones "suaves": $\lambda$ pequeño produce muchos valores cerca de $\pm 1$; $\lambda$ grande, muchos cerca de $0$; e intermedio, una mezcla de ambos.

**Mountain-car.** La recompensa verdadera, no descontada, es $-1$ por paso hasta alcanzar la meta en lo alto de la colina; el estado es la posición $x$ del auto y su velocidad. Como el espacio es continuo, se usó la versión de la Sección 4. La clase de aproximación fueron todas las combinaciones lineales de **26 funciones base gaussianas** uniformemente espaciadas, solo sobre la posición $x$ del auto. Dada la política óptima (determinada con una discretización fina de $120 \times 120$ del espacio de estados; se usó una muestra de $|S_0| = 5000$ estados, sin contar los que no daban restricciones no triviales), la Figura 4a muestra que la solución captura casi perfectamente la estructura $R = -c$ (constante negativa) —nótese la escala del eje $y$, en el rango $\approx -2.5063$ a $-2.5067$. El eje de posición del auto abarca el rango $[-1.2, 0.6]$. En una variante más exigente, la recompensa verdadera se cambió a $1$ en el intervalo $[-0.72, -0.32]$ (centrado en el fondo de la colina) y $0$ en el resto, con $\gamma = 0.99$; la política óptima es ir tan rápido como se pueda al fondo y estacionar ahí (no siempre posible: cerca de la cima derecha y con demasiada velocidad, el auto puede dispararse por el extremo derecho y entrar al estado absorbente aunque frene). La Figura 4b muestra que se recupera la estructura principal (recompensa grande y positiva alrededor del intervalo), con un artefacto en el lado derecho atribuido justamente al efecto de "salirse disparado".

**Gridworld continuo $5 \times 5$ (desde trayectorias).** El estado es $[0,1] \times [0,1]$; cada una de las cuatro acciones cardinales mueve al agente $0.2$ en la dirección deseada, tras lo cual se agrega ruido uniforme en $[-0.1, 0.1]$ a cada coordenada y el estado se trunca al cuadrado unitario. La recompensa verdadera es $1$ en el cuadrado (no absorbente) $[0.8, 1] \times [0.8, 1]$ y $0$ en el resto, con $\gamma = 0.9$. La clase de aproximación fue un arreglo $15 \times 15$ de funciones base gaussianas bidimensionales. La distribución inicial $D$ fue uniforme; el algoritmo corrió con $m = 5000$ trayectorias de $30$ pasos cada una para evaluar cada política, y cuando se necesitaba la política "óptima" para comparación, el MDP se resolvió sobre una discretización de $50 \times 50$. La Figura 5 muestra los resultados en 5 corridas: la solución era razonable ya tras **1 iteración** y solía estabilizarse hacia las **15 iteraciones**. Comparando la política óptima de la recompensa ajustada con la verdadera, la fracción de estados con acción distinta fue típicamente entre **3 % y 10 %** (esperable dado que hay muchas políticas casi óptimas distintas). Midiendo la *calidad* (usando la recompensa verdadera, con **50000 trials Monte Carlo de 50 pasos**), hacia las 15 iteraciones no se detectó diferencia estadísticamente significativa entre el valor de la política óptima verdadera (**≈ 6.65**) y el de la política óptima de la recompensa ajustada.

## 6. Limitaciones

El propio paper y la perspectiva posterior señalan varias limitaciones:

- **Ambigüedad no resuelta del todo.** Las heurísticas de margen escogen *una* recompensa, pero la elección es en última instancia una preferencia inductiva (margen máximo, dispersión $\ell_1$), no una consecuencia del problema. La degeneración se maneja, no se elimina.
- **Supuesto de optimalidad del experto.** El método asume que la conducta observada es óptima (o casi). El paper anota como problema abierto: si la conducta es fuertemente inconsistente con la optimalidad, ¿se pueden identificar recompensas "localmente consistentes" por regiones del espacio de estados?
- **Ruido del observador.** En aplicaciones reales puede haber ruido sustancial en las mediciones de las entradas sensoriales y acciones del agente, y la propia selección de acción del agente puede ser ruidosa o subóptima; además, puede haber muchas políticas óptimas de las que solo se observan unas pocas. ¿Cuáles son las métricas adecuadas para ajustar tales datos?
- **Identificabilidad y diseño de experimentos.** ¿Cómo diseñar experimentos que maximicen la identificabilidad de la recompensa?
- **Observabilidad parcial.** El enfoque se plantea para MDPs completamente observables; su extensión a entornos parcialmente observables (POMDPs) queda como pregunta abierta explícita.
- **Recompensas "fáciles" de optimizar.** El paper conecta con el *reward shaping* basado en potenciales (Ng, Harada & Russell, 1999) y pregunta si se pueden diseñar algoritmos de IRL que recuperen recompensas "fáciles" de optimizar.
- **Dependencia de las features.** En los casos 2 y 3, la calidad de la recompensa recuperada está limitada por la expresividad de la base $\phi$ elegida; una base pobre puede impedir representar cualquier recompensa no trivial que explique la política.
- **Escala.** Los propios autores concluyen que el IRL es "soluble, al menos para dominios discretos y continuos de tamaño moderado"; la escalabilidad a problemas grandes no está garantizada.

## 7. Legado

Este paper abrió un campo entero. Su descendencia directa:

- **Apprenticeship learning (Abbeel & Ng, 2004).** Retoma la idea de igualar *feature expectations* entre experto y aprendiz. En vez de recuperar explícitamente la recompensa, garantiza que la política aprendida tenga expectativas de features cercanas a las del experto, con garantías de desempeño. Formaliza y hace práctico el caso 3 de Ng y Russell.
- **Maximum Entropy IRL (Ziebart et al., 2008).** Ataca la ambigüedad de frente: entre todas las distribuciones sobre trayectorias que igualan las feature expectations del experto, elige la de **máxima entropía**. Esto reemplaza la arbitrariedad de las heurísticas de margen por un principio probabilístico y da un modelo generativo de la conducta, útil justamente cuando el experto es subóptimo o ruidoso —la limitación que Ng y Russell dejaron abierta.
- **GAIL (Ho & Ermon, 2016).** *Generative Adversarial Imitation Learning* conecta el IRL con las GANs: un discriminador aprende a distinguir trayectorias del experto de las del aprendiz (haciendo implícitamente el rol de la recompensa), mientras el aprendiz aprende a engañarlo. Salta la recuperación explícita de la recompensa y escala a problemas de alta dimensión.
- **RLHF como IRL.** El *Reinforcement Learning from Human Feedback* que alinea los grandes modelos de lenguaje es, conceptualmente, IRL: en vez de programar la recompensa "buena respuesta", se **infiere un modelo de recompensa a partir de preferencias humanas** y luego se optimiza una política (el LLM) contra esa recompensa aprendida. La premisa de Ng y Russell —la recompensa es la descripción más parsimoniosa y transferible de la tarea— es exactamente la justificación del RLHF frente a la imitación directa (SFT).

## 8. Conexión con la Clase 33

La Clase 33 gira en torno a **cómo se aprende y se especifica el objetivo** de un agente. Ng & Russell aportan la pieza teórica fundacional: demuestran que el problema inverso —inferir el objetivo desde la conducta— es tratable, caracterizable en forma cerrada y reducible a programación lineal, pero **intrínsecamente ambiguo**, lo que obliga a introducir preferencias inductivas explícitas (margen, dispersión, más tarde máxima entropía). Entender este paper es entender por qué el RL moderno se preocupa tanto del *reward design* y del *reward modeling*: la recompensa es la especificación de la tarea, y recuperarla bien —desde demostraciones o desde preferencias— es lo que hace posible tanto el apprenticeship learning clásico como la alineación de LLMs contemporánea.

## Nota final: relevancia para salud, MDM y record linkage (lector experto FHIR)

Para un ingeniero que construye sistemas de *Master Data Management* y *record linkage* sobre FHIR, el IRL ofrece un marco conceptual directo: **inferir la función de match a partir de las decisiones de los stewards**. Hoy los pesos de los comparadores (Jaro-Winkler sobre nombres, coincidencia de fechas de nacimiento, distancia fonética, concordancia de identificadores) suelen fijarse a mano o calibrarse con Fellegi-Sunter. Pero cada vez que un steward humano resuelve una cola de casos dudosos —confirma un merge, rechaza un enlace, escala a revisión— actúa como el "experto óptimo" de Ng y Russell: sus decisiones revelan una función de recompensa latente sobre el espacio de features del par de registros. Tratar la secuencia de decisiones del steward como trayectorias observadas y aplicar IRL (o su descendiente práctico, el aprendizaje desde preferencias al estilo RLHF) permitiría **recuperar los pesos implícitos del match** —incluida la ponderación multiatributo entre precisión y recall que ningún steward articula explícitamente, tal como la abeja no articula su ponderación entre néctar y riesgo— y transferirlos a nuevas poblaciones o cambios de dinámica (una nueva fuente HL7, un giro demográfico) reoptimizando la política de bloqueo/matching, en vez de copiar decisiones caso a caso.
