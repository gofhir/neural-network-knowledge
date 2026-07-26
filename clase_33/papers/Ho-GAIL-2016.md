# Generative Adversarial Imitation Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Generative Adversarial Imitation Learning* (GAIL).
- **Autores:** Jonathan Ho y Stefano Ermon (**Stanford University**).
- **Venue:** *Advances in Neural Information Processing Systems* (**NeurIPS/NIPS 2016**).
- **Preprint:** arXiv:1606.03476v1 (10 jun 2016), [arxiv.org/abs/1606.03476](https://arxiv.org/abs/1606.03476).
- **Una línea:** deriva un marco general que muestra que "ejecutar RL sobre la recompensa recuperada por IRL" equivale a hacer *occupancy measure matching* (igualar la distribución de ocupación estado-acción entre aprendiz y experto), y que una instancia particular de ese marco produce un objetivo tipo GAN; de ahí se obtiene un algoritmo *model-free* de imitación que supera con amplitud a los métodos previos en control continuo de alta dimensión.

GAIL responde a una pregunta incómoda que arrastraba el aprendizaje por imitación hacia 2016. Cuando se quiere aprender una tarea a partir de demostraciones de un experto —sin poder consultar al experto durante el entrenamiento y sin señal de recompensa— hay dos caminos clásicos. El primero, *behavioral cloning* (clonación conductual), trata la imitación como un problema de aprendizaje supervisado sobre pares estado-acción; es engañosamente simple, pero sufre de **error acumulado** (*compounding error*) por *covariate shift*. El segundo, *inverse reinforcement learning* (IRL), recupera una función de costo bajo la cual el experto es óptimo y luego extrae una política de ese costo mediante RL; evita el error acumulado porque razona sobre trayectorias completas, pero es **indirecto y lento**: muchos algoritmos de IRL requieren resolver un problema de RL en un bucle interno por cada actualización del costo.

La observación central de Ho y Ermon es que ese rodeo por la función de costo es, en muchos casos, innecesario. Si el objetivo último del aprendiz es *actuar* como el experto, ¿por qué pagar el costo de aprender una función de costo que luego habrá que "resolver" con RL? El paper caracteriza formalmente la política que se obtendría de correr RL sobre el costo aprendido por IRL de máxima entropía causal, y muestra que esa política es, exactamente, la que iguala la **medida de ocupación** del experto. Esto convierte la imitación en un problema de *matching* de distribuciones, y una elección específica del regularizador del costo lo transforma en un juego adversario idéntico en forma al de las GAN. El discriminador cumple el papel de "recompensa aprendida" y la política el de "generador".

Para la **Clase 33 (Aprendizaje por Imitación y Aprendizaje Reforzado Inverso)**, GAIL es el puente moderno que unifica los tres hilos del curso: IRL (recuperar recompensas), imitación (reproducir conducta) y las GAN (aprendizaje adversario de distribuciones). Aunque no aparezca citado literalmente en las diapositivas, es el resultado canónico que explica *por qué* IRL e imitación son, en el fondo, la misma cosa vista desde ángulos duales.

## 2. Contexto: las dos familias y sus límites

### 2.1. Behavioral cloning: simple pero con error acumulado

La clonación conductual ajusta una política $\pi$ como un clasificador (o regresor) supervisado sobre los pares $(s, a)$ de las trayectorias del experto. Es atractiva por su simplicidad —es aprendizaje supervisado puro, sin interacción con el entorno—, pero, como señalan los autores, "solo tiende a tener éxito con grandes cantidades de datos, debido al error acumulado causado por *covariate shift*".

El mecanismo del fallo es sutil pero fatal. El clasificador se entrena bajo la distribución de estados que *visita el experto*, pero se ejecuta bajo la distribución de estados que visita *el propio aprendiz*. Cualquier error de un solo paso lleva al aprendiz a un estado ligeramente fuera de la distribución de entrenamiento, donde su predicción es aún peor, lo que lo empuja a estados todavía más ajenos. Los errores se realimentan y crecen a lo largo del horizonte —de ahí el término *compounding error*. Esta es exactamente la patología que **DAgger** (Ross et al., 2011) ataca al permitir consultar al experto en los estados que el aprendiz visita realmente; pero DAgger requiere un experto interactivo consultable durante el entrenamiento, supuesto que aquí se descarta explícitamente.

### 2.2. IRL clásico: correcto pero costoso

IRL toma el camino opuesto: en lugar de ajustar decisiones de un solo paso, "aprende una función de costo que prioriza trayectorias enteras sobre otras", de modo que el error acumulado deja de ser un problema. Por eso IRL ha tenido éxito en tareas que van desde predecir el comportamiento de taxistas hasta planificar el paso de robots cuadrúpedos. El precio es computacional: "muchos algoritmos de IRL son extremadamente caros de ejecutar, porque requieren aprendizaje por refuerzo en un bucle interno". Escalar IRL a entornos grandes ha sido, por ello, foco de mucha investigación.

Hay además una crítica más profunda, casi filosófica, que motiva todo el paper: IRL aprende una función de costo que *explica* la conducta experta, pero no le dice directamente al aprendiz *cómo actuar*. Dado que el objetivo real del aprendiz suele ser tomar acciones que imiten al experto —de hecho, muchos algoritmos de IRL se evalúan por la calidad de las acciones óptimas de los costos que aprenden—, los autores preguntan: ¿por qué debemos aprender una función de costo, si hacerlo posiblemente incurre en un gasto computacional considerable y aun así no produce acciones directamente?

## 3. Contribución central: IRL ↔ occupancy matching ↔ GAN

### 3.1. El marco de IRL con regularizador $\psi$

El punto de partida es el IRL de **máxima entropía causal** (Ziebart et al.). Dado un experto $\pi_E$, este ajusta un costo resolviendo

$$\underset{c \in \mathcal{C}}{\text{maximizar}}\; \left( \min_{\pi \in \Pi} -H(\pi) + \mathbb{E}_\pi[c(s,a)] \right) - \mathbb{E}_{\pi_E}[c(s,a)],$$

donde $H(\pi) = \mathbb{E}_\pi[-\log \pi(a \mid s)]$ es la **entropía causal descontada** de la política. Intuitivamente, IRL busca un costo que asigne costo bajo al experto y costo alto a las demás políticas. El bloque interno,

$$\mathrm{RL}(c) = \arg\min_{\pi \in \Pi} -H(\pi) + \mathbb{E}_\pi[c(s,a)],$$

es el procedimiento de RL (regularizado por entropía) que recupera una política de alto entropía que minimiza el costo esperado.

Para estudiar el poder expresivo máximo de IRL, los autores consideran la clase de costos más grande posible, *todas* las funciones $\mathcal{C} = \mathbb{R}^{S \times A}$. Con una clase tan amplia, IRL sobreajustaría con datos finitos, así que incorporan un **regularizador de costo convexo** $\psi : \mathbb{R}^{S \times A} \to \bar{\mathbb{R}}$, definiendo la primitiva

$$\mathrm{IRL}_\psi(\pi_E) = \arg\max_{c \in \mathbb{R}^{S \times A}} -\psi(c) + \left( \min_{\pi \in \Pi} -H(\pi) + \mathbb{E}_\pi[c(s,a)] \right) - \mathbb{E}_{\pi_E}[c(s,a)].$$

Lo notable es que $\psi$ no es una molestia técnica: resulta ser la pieza que gobierna qué algoritmo de imitación se obtiene. Elegir $\psi$ es elegir el algoritmo.

### 3.2. La medida de ocupación como cambio de variable

La herramienta clave para volver todo tratable es la **medida de ocupación** (*occupancy measure*). Para una política $\pi$, se define

$$\rho_\pi(s,a) = \pi(a \mid s) \sum_{t=0}^{\infty} \gamma^t\, P(s_t = s \mid \pi).$$

$\rho_\pi$ es la distribución (no normalizada, descontada) de pares estado-acción que un agente encuentra al navegar el entorno con la política $\pi$. Su virtud es que linealiza los costos esperados: $\mathbb{E}_\pi[c(s,a)] = \sum_{s,a} \rho_\pi(s,a)\, c(s,a)$. El conjunto de medidas de ocupación válidas $\mathcal{D} = \{\rho_\pi : \pi \in \Pi\}$ resulta ser un conjunto convexo definido por restricciones afines (las ecuaciones de flujo de Bellman). Y por la Proposición 3.1 (Syed et al.) hay una **correspondencia biunívoca** entre políticas y medidas de ocupación: dada $\rho \in \mathcal{D}$, la única política que la genera es $\pi_\rho(a \mid s) = \rho(s,a) / \sum_{a'} \rho(s,a')$. Esto permite pasar libremente entre optimizar sobre políticas y optimizar sobre distribuciones —un problema convexo.

### 3.3. El resultado bisagra: RL∘IRL es occupancy matching

Con esta maquinaria, el paper prueba su resultado central (Proposición 3.2):

$$\mathrm{RL} \circ \mathrm{IRL}_\psi(\pi_E) = \arg\min_{\pi \in \Pi} -H(\pi) + \psi^*(\rho_\pi - \rho_{\pi_E}),$$

donde $\psi^*$ es la **conjugada convexa** de $\psi$. La prueba descansa en que el costo y la política óptimos forman un **punto de silla** de cierta función: IRL encuentra una coordenada de ese punto de silla (el costo) y correr RL sobre la salida de IRL revela la otra (la política).

La lectura es profunda: **el IRL regularizado por $\psi$, implícitamente, busca una política cuya medida de ocupación esté cerca de la del experto, medida por la función convexa $\psi^*$.** IRL no es "realmente" sobre recuperar costos; es el *dual* de un problema de igualación de distribuciones de ocupación, y el costo recuperado no es más que la variable dual óptima.

El caso límite lo deja transparente. Si $\psi$ es una función constante (sin regularización), el Corolario 3.2.1 establece que $\rho_{\tilde\pi} = \rho_{\pi_E}$: la política recuperada iguala **exactamente** la medida de ocupación del experto. La demostración lo enmarca como dualidad de Lagrange: minimizar $-\bar H(\rho)$ sujeto a $\rho(s,a) = \rho_E(s,a)$ para todo $(s,a)$ es el problema primal, los costos $c(s,a)$ son las variables duales de las restricciones de igualdad, y como $-\bar H$ es estrictamente convexa se cumple dualidad fuerte, de modo que el óptimo primal (la política) se recupera unívocamente del óptimo dual (el costo). En palabras del propio paper: los algoritmos clásicos de IRL que resuelven RL repetidamente en un bucle interno pueden interpretarse como una forma de **ascenso dual**, en el que uno resuelve una y otra vez el problema primal (que es RL) con las variables duales fijas. Y ahí está el costo del método clásico: el ascenso dual es eficiente solo si el primal es barato, pero en IRL el primal *es* RL.

## 4. Método: de la teoría al algoritmo GAIL

### 4.1. Por qué el matching exacto no basta

El Corolario 3.2.1 es elegante pero inútil en la práctica. La distribución del experto se entrega solo como un conjunto finito de muestras, así que en entornos grandes la mayoría de los valores de $\rho_{\pi_E}$ serán exactamente cero; igualar la ocupación de forma exacta forzaría a la política aprendida a *nunca* visitar pares estado-acción no vistos, simplemente por falta de datos. Además, con aproximación de funciones (una política parametrizada $\pi_\theta$), el número de restricciones sería igual al número de puntos en $S \times A$: intratable.

La solución es relajar la igualdad exacta a una penalización suave, motivada por la Proposición 3.2:

$$\min_\pi\; d_\psi(\rho_\pi, \rho_{\pi_E}) - H(\pi), \qquad d_\psi(\rho_\pi, \rho_{\pi_E}) \triangleq \psi^*(\rho_\pi - \rho_{\pi_E}).$$

Distintos $\psi$ dan distintos algoritmos. Los autores muestran que los regularizadores de tipo **indicador** $\delta_{\mathcal{C}}$ (que fuerzan al costo a vivir en un subespacio lineal de funciones base) recuperan exactamente los algoritmos previos de *apprenticeship learning* —*feature expectation matching* de Abbeel-Ng, MWAL, LPAL. Estos escalan a entornos grandes con políticas parametrizadas, pero tienen un defecto conocido: si la clase $\mathcal{C}$ es demasiado restrictiva —lo habitual en subespacios lineales, salvo que las funciones base estén cuidadosamente diseñadas— no recuperan políticas verdaderamente parecidas al experto. La razón, ahora clara desde la Proposición 3.2, es que *apprenticeship learning* equivale a forzar a que $\pi_E$ se codifique como un elemento de $\mathcal{C}$; si $\mathcal{C}$ no contiene un costo que explique bien la conducta experta, la recuperación falla.

### 4.2. El regularizador $\psi_{GA}$ y el objetivo tipo GAN

La contribución técnica fina es proponer un regularizador que combine lo mejor de ambos mundos: la expresividad del caso constante (matching exacto) y la tratabilidad de los indicadores lineales. Los autores definen

$$\psi_{GA}(c) \triangleq \begin{cases} \mathbb{E}_{\pi_E}[g(c(s,a))] & \text{si } c < 0 \\ +\infty & \text{en otro caso} \end{cases}, \qquad g(x) = \begin{cases} -x - \log(1 - e^{x}) & \text{si } x < 0 \\ +\infty & \text{en otro caso} \end{cases}.$$

Este $\psi_{GA}$ penaliza poco los costos que asignan costo negativo a pares estado-acción del experto, pero penaliza fuertemente los costos que asignan costo alto (cercano a cero) al experto. A diferencia de los indicadores $\delta_{\mathcal{C}}$, que son fijos, $\psi_{GA}$ es un promedio sobre los datos del experto y por tanto **se adapta al conjunto de demostraciones**; y crucialmente, permite *cualquier* función de costo (mientras sea negativa en todas partes), no solo las de un subespacio de dimensión finita.

El motivo de esta elección aparentemente arcana es el resultado del apéndice (Corolario A.1.1), que se deriva usando la **pérdida logística** $\phi(x) = \log(1 + e^{-x})$ dentro de una correspondencia general entre riesgos de clasificación binaria y $f$-divergencias (Nguyen et al.):

$$\psi_{GA}^*(\rho_\pi - \rho_{\pi_E}) = \max_{D \in (0,1)^{S \times A}} \mathbb{E}_\pi[\log D(s,a)] + \mathbb{E}_{\pi_E}[\log(1 - D(s,a))].$$

El lado derecho es exactamente la **pérdida logarítmica negativa óptima** del problema de clasificación binaria que distingue los pares $(s,a)$ generados por $\pi$ de los generados por $\pi_E$. Y esa pérdida óptima es, salvo una constante, la **divergencia de Jensen-Shannon** entre las medidas de ocupación:

$$D_{JS}(\rho_\pi, \rho_{\pi_E}) = D_{KL}\!\left(\rho_\pi \,\Big\|\, \tfrac{\rho_\pi + \rho_{\pi_E}}{2}\right) + D_{KL}\!\left(\rho_{\pi_E} \,\Big\|\, \tfrac{\rho_\pi + \rho_{\pi_E}}{2}\right).$$

Sustituyendo, el objetivo de imitación (con la entropía causal $H$ tratada como regularizador de la política, controlado por $\lambda \ge 0$) queda:

$$\min_\pi\; \psi_{GA}^*(\rho_\pi - \rho_{\pi_E}) - \lambda H(\pi) = D_{JS}(\rho_\pi, \rho_{\pi_E}) - \lambda H(\pi).$$

En otras palabras: GAIL busca la política cuya medida de ocupación **minimiza la divergencia de Jensen-Shannon** respecto a la del experto. Como $D_{JS}$ es una métrica genuina entre distribuciones (una raíz cuadrada de la JS lo es), a diferencia del *apprenticeship learning* lineal, GAIL *puede* imitar al experto de forma exacta.

### 4.3. La analogía con las GAN y el algoritmo

Expandiendo la conjugada, el objetivo se vuelve un juego de silla explícito:

$$\min_\pi \max_D\; \mathbb{E}_\pi[\log D(s,a)] + \mathbb{E}_{\pi_E}[\log(1 - D(s,a))] - \lambda H(\pi).$$

La correspondencia con las GAN es directa. En una GAN, un generador $G$ trata de confundir a un discriminador $D$ cuyo trabajo es distinguir los datos generados de los datos reales; cuando $D$ ya no puede distinguirlos, $G$ ha igualado la distribución verdadera. Aquí, la medida de ocupación del aprendiz $\rho_\pi$ hace de "distribución generada" y la del experto $\rho_{\pi_E}$ de "distribución verdadera". El **discriminador $D$ es la recompensa aprendida** y la **política $\pi$ es el generador**.

El algoritmo (Algorithm 1) alterna dos pasos que introducen aproximación de funciones —una política $\pi_\theta$ y un discriminador $D_w : S \times A \to (0,1)$:

1. **Paso del discriminador.** Se muestrean trayectorias $\tau_i \sim \pi_{\theta_i}$ y se da un paso de gradiente **Adam** sobre $w$ para *aumentar* el objetivo respecto de $D$: $\hat{\mathbb{E}}_{\tau_i}[\nabla_w \log D_w(s,a)] + \hat{\mathbb{E}}_{\tau_E}[\nabla_w \log(1 - D_w(s,a))]$. Es decir, $D$ aprende a asignar valores altos a los pares del aprendiz y bajos a los del experto.
2. **Paso de la política.** Se da un paso **TRPO** sobre $\theta$ para *disminuir* el objetivo respecto de $\pi$, usando como función de costo $\log D_{w_{i+1}}(s,a)$. El gradiente es $\hat{\mathbb{E}}_{\tau_i}[\nabla_\theta \log \pi_\theta(a \mid s)\, Q(s,a)] - \lambda \nabla_\theta H(\pi_\theta)$, con $Q(\bar s, \bar a) = \hat{\mathbb{E}}_{\tau_i}[\log D_{w_{i+1}}(s,a) \mid s_0 = \bar s, a_0 = \bar a]$.

El discriminador se interpreta como una **función de costo local** que da señal de aprendizaje a la política: dar un paso que reduce el costo esperado respecto de $c(s,a) = \log D(s,a)$ mueve la política hacia regiones del espacio estado-acción que el discriminador clasifica como "de experto". El paso TRPO cumple el mismo papel que en el *apprenticeship learning* de Ho et al.: es un paso de gradiente natural restringido por KL que impide que la política cambie demasiado por el ruido en la estimación del gradiente de política.

### 4.4. Por qué GAIL evita el bucle costoso de IRL clásico

Aquí está la ganancia arquitectónica. El IRL clásico, visto como ascenso dual, resuelve un problema de RL *completo* en el bucle interno por cada actualización del costo —recordemos que el primal *es* RL, así que cada iteración es cara. GAIL, en cambio, **no resuelve un RL hasta convergencia por cada actualización del discriminador**: entrelaza un solo paso de gradiente del discriminador (la "recompensa") con un solo paso de TRPO de la política. Ambas coordenadas del punto de silla avanzan simultáneamente, en vez de resolver una por completo antes de tocar la otra. Es la misma diferencia que separa entrenar una GAN (pasos alternados de $G$ y $D$) de un esquema anidado donde se optimizaría $G$ por completo por cada paso de $D$. Así GAIL hereda la corrección de IRL (razona sobre ocupación, no sobre decisiones de un paso, evitando el error acumulado de la clonación) sin pagar el bucle interno de RL.

## 5. Experimentos: control continuo en MuJoCo

Los autores evalúan Algorithm 1 en **9 tareas** de control basado en física, desde tareas clásicas de baja dimensión —**cartpole**, **acrobot**, **mountain car**— hasta tareas difíciles de alta dimensión simuladas con **MuJoCo**: **Reacher**, **HalfCheetah**, **Hopper**, **Walker**, **Ant** y un **humanoide 3D** (locomoción resuelta solo recientemente por RL *model-free*). El humanoide tiene un espacio de observación de 376 dimensiones continuas y 17 acciones continuas.

El protocolo es cuidadoso. Cada tarea trae una función de costo verdadera (de OpenAI Gym); primero se generan políticas experto corriendo **TRPO** sobre esos costos verdaderos. Luego, para medir la eficiencia respecto al *número de demostraciones*, se muestrean conjuntos de datos de distinto tamaño (cada trayectoria consta de unos 50 pares estado-acción). Los baselines son tres: **behavioral cloning**, **feature expectation matching (FEM)** y **game-theoretic apprenticeship learning (GTAL)**. Todos los métodos usan la misma arquitectura de red (dos capas ocultas de 100 unidades con `tanh`), y a FEM, GTAL y GAIL se les da *exactamente la misma* cantidad de interacción con el entorno.

Los hallazgos:

- **Tareas clásicas de control** (cartpole, acrobot, mountain car): la clonación conductual sufrió en eficiencia de datos experto frente a FEM y GTAL, que en su mayoría alcanzaban rendimiento casi experto con un rango amplio de tamaños de dataset. GAIL siempre produjo políticas mejores que las tres líneas base.
- **Reacher:** la única excepción notable —la clonación conductual fue *más* eficiente en muestras que GAIL. Los autores lograron mejorar levemente a GAIL en Reacher con regularización de entropía causal; en el caso de 4 trayectorias, la mejora de $\lambda = 0$ a $\lambda = 10^{-3}$ fue estadísticamente significativa (test de Wilcoxon de suma de rangos, una cola, $p = 0.05$). En todas las demás tareas no se usó regularización de entropía.
- **Demás entornos MuJoCo:** GAIL mostró una **gran ventaja** sobre los baselines. Alcanzó al menos el 70 % del rendimiento experto para todos los tamaños de dataset probados, dominando casi siempre a las líneas base. FEM y GTAL rindieron mal en **Ant**, produciendo políticas consistentemente peores que una política aleatoria. La clonación conductual llegó a rendimiento satisfactorio con suficientes datos en HalfCheetah, Hopper, Walker y Ant, pero **no superó el 60 % en el humanoide**, tarea en la que GAIL alcanzó rendimiento experto exacto para todos los tamaños de dataset. El mensaje empírico: GAIL es muy eficiente en el *número de demostraciones expertas* que necesita.

## 6. Limitaciones

Los propios autores son explícitos sobre las fronteras del método.

- **Eficiencia en interacción con el entorno.** Aunque GAIL es eficiente en datos del experto, "no es particularmente eficiente en muestras en cuanto a la interacción con el entorno durante el entrenamiento". El número de muestras necesarias para estimar el gradiente del objetivo de imitación fue comparable al número que TRPO necesita para entrenar las políticas experto desde señales de refuerzo. Los autores sugieren que inicializar los parámetros de la política con clonación conductual —que no requiere interacción alguna— podría acelerar considerablemente el aprendizaje.
- **Carácter *model-free*.** Al ser libre de modelo, GAIL necesita generalmente más interacción con el entorno que los métodos basados en modelo. El *guided cost learning* (Finn et al.), por ejemplo, hereda la eficiencia de muestras del *guided policy search*, pero también su exigencia de que la dinámica sea bien aproximada por dinámicas lineales variantes en el tiempo ajustadas iterativamente. (Curiosamente, tanto GAIL como *guided cost learning* alternan optimización de política y ajuste del costo —"ajuste del discriminador" en GAIL—, pese a derivarse de forma completamente distinta.)
- **Sin interacción con el experto.** Igual que IRL, GAIL no interactúa con el experto durante el entrenamiento: explora aleatoriamente para descubrir qué acciones acercan la ocupación de la política a la del experto, mientras que métodos como DAgger simplemente se lo preguntan al experto. Los autores anticipan que un método que combine buenos modelos del entorno con interacción experta ganaría en complejidad de muestras tanto de datos del experto como de interacción.
- **Inestabilidad adversaria (implícita).** El objetivo es un juego min-max resuelto por pasos alternados de gradiente sin garantía de convergencia global a un equilibrio, herencia directa de la dinámica de entrenamiento de las GAN. El paso TRPO existe precisamente para amortiguar el ruido y evitar que la política diverja por gradientes mal estimados —una salvaguarda necesaria contra la fragilidad del esquema adversario.

## 7. Legado y conexiones

GAIL abrió la línea de la **imitación adversaria** y se convirtió en la referencia obligada del campo. Su descendiente conceptual más directo es **AIRL** (*Adversarial Inverse Reinforcement Learning*, Fu et al., 2018), que refina el discriminador para que la recompensa recuperada sea *transferible* —GAIL entrega una gran política pero no una función de recompensa reutilizable e independiente de la dinámica, precisamente porque su discriminador enreda recompensa y dinámica; AIRL restructura el discriminador para desacoplarlas y recuperar una recompensa que sobreviva a cambios del entorno.

La conexión conceptual con **RLHF** (aprendizaje por refuerzo a partir de retroalimentación humana), la técnica que afinó ChatGPT e InstructGPT, es más sutil pero iluminadora. En RLHF, un *modelo de recompensa* aprendido de preferencias humanas hace de señal que una política de lenguaje optimiza con PPO. GAIL comparte el esqueleto: un discriminador/recompensa aprendido de datos humanos (demostraciones, en vez de preferencias) que guía una política optimizada con un método de región de confianza (TRPO, primo de PPO). En ambos casos, la "recompensa" no es dada por el entorno sino *aprendida* de conducta humana, y la política se optimiza contra esa recompensa cambiante. GAIL es, en este sentido, un ancestro estructural del aprendizaje de recompensa que domina el alineamiento moderno de modelos de lenguaje.

## 8. Conexión con la Clase 33

La Clase 33 —*Aprendizaje por Imitación y Aprendizaje Reforzado Inverso*, del prof. Rodrigo Toro Icarte— cubre exactamente las tres piezas que GAIL unifica. IRL (recuperar la recompensa que hace óptimo al experto, con Ng-Russell y Ziebart como fundacionales), la imitación directa (clonación conductual, con DAgger como corrección al error acumulado) y la *combinación* de ambas. GAIL es el resultado que hace explícita esa combinación: demuestra formalmente que **IRL es el dual de un problema de igualación de ocupación**, y que por tanto "correr RL sobre la recompensa de IRL" e "imitar igualando la distribución de ocupación" son la misma operación vista desde el lado dual y el primal, respectivamente. Además tiende el puente hacia las GAN —una técnica que en el curso aparece en el módulo de modelos generativos— al mostrar que la elección correcta del regularizador convierte la imitación en un juego adversario que minimiza la divergencia de Jensen-Shannon. Para el estudiante, GAIL es la pieza que ordena el mapa de la clase: no son tres temas separados, sino tres caras de un mismo problema de *matching* de distribuciones.

---

**Nota para el lector experto en salud y FHIR.** El *occupancy measure matching* tiene una lectura natural en la resolución de identidades de pacientes (MDM/record linkage). Piénsese en la política de resolución de duplicados como un agente que "navega" un flujo de decisiones —fusionar, no fusionar, escalar a revisión humana— sobre pares de recursos `Patient`. La *medida de ocupación experta* $\rho_{\pi_E}$ sería la distribución global de decisiones que toma el equipo de *data stewards* humanos a lo largo de todos los casos que enfrentan; hacer *matching* de ocupación equivale a exigir que la política automática reproduzca esa distribución de decisiones **globalmente**, no que acierte caso a caso de forma aislada (que es lo que haría un clasificador entrenado por clonación, vulnerable al mismo error acumulado cuando el sistema empieza a visitar distribuciones de pares que los humanos rara vez vieron). El discriminador $D$ sería, entonces, un modelo que aprende a distinguir una traza de decisiones tomada por el sistema de una tomada por un humano experto; cuando el discriminador ya no puede diferenciarlas, la política de MDM ha internalizado el criterio humano agregado —incluidos los casos límite que ninguna regla determinista de *blocking* o umbral de similitud captura bien. La advertencia de eficiencia de GAIL también se traslada: se necesitan pocas demostraciones expertas, pero mucha "interacción con el entorno" (muchas corridas del pipeline de resolución), lo que en un contexto clínico obliga a simular contra un histórico curado en lugar de aprender en producción.
