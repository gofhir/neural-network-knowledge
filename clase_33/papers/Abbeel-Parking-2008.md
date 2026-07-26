# Apprenticeship Learning for Motion Planning with Application to Parking Lot Navigation — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Apprenticeship Learning for Motion Planning with Application to Parking Lot Navigation*.
- **Autores:** Pieter Abbeel, Dmitri Dolgov, Andrew Y. Ng, Sebastian Thrun. Abbeel, Ng y Thrun pertenecían al Computer Science Department de Stanford University; Dolgov estaba en el Toyota Research Institute (USA).
- **Venue:** *IEEE/RSJ International Conference on Intelligent Robots and Systems* (IROS 2008).
- **Vehículo experimental:** *Junior*, el auto robótico del Stanford Racing Team (RADAR, LIDAR, cámaras, GPS+IMU de alta precisión).

Este trabajo aborda un problema práctico y aparentemente prosaico —hacer que un auto planifique cómo moverse dentro de un estacionamiento— y lo convierte en un caso de estudio limpio de **aprendizaje inverso**. La observación de fondo es que los algoritmos de planificación de movimiento usan **funciones de costo (o potenciales) complejas** con muchos términos que compiten entre sí: suavidad de la trayectoria, distancia a los obstáculos, curvatura máxima, mantenerse en el carril, evitar la marcha atrás, etc. Ajustar a mano el balance entre estos términos —cuánto pesa cada uno— es tedioso y poco intuitivo. La tesis del paper es que **es mucho más fácil demostrar unas pocas trayectorias buenas que especificar a mano los pesos**, y que esas demostraciones contienen implícitamente la información sobre el compromiso deseado.

El aporte concreto es un algoritmo eficiente que, dadas unas pocas demostraciones de trayectorias hechas por un conductor humano, **infiere automáticamente los pesos de la función de costo** de modo que el planificador reproduzca el estilo de conducción demostrado. Los autores adaptan la maquinaria de *apprenticeship learning via inverse reinforcement learning* de Abbeel & Ng (2004) —originalmente formulada para procesos de decisión de Markov (MDP)— al escenario de **campos de potencial para planificación de movimiento**. El planificador subyacente es el mismo que usó el equipo de Stanford en el DARPA Urban Challenge, extendido con nuevos términos de costo para capturar una gama amplia de estilos naturales de conducción.

Para la Clase 33 ("Aprendizaje por Imitación y Aprendizaje Reforzado Inverso", prof. Rodrigo Toro Icarte) este paper es el ejemplo canónico de **IRL aplicado a la robótica del mundo real**: el profesor lo cita explícitamente (slide 25) como el caso de "aprender a estacionar usando IRL". No se aprende una política de control desde recompensas dadas, sino que se **recupera la función de costo (la recompensa negada)** a partir del comportamiento experto —el corazón conceptual del aprendizaje reforzado inverso.

## 2. Contexto: de Abbeel & Ng 2004 a la conducción autónoma

Para ubicar este trabajo hay que recordar la distinción central de la Clase 33. En el **aprendizaje reforzado (RL) clásico** se supone conocida una función de recompensa $R(s)$ y el objetivo es hallar la política $\pi$ que maximiza la recompensa acumulada esperada. En el **aprendizaje reforzado inverso (IRL)** el problema se da vuelta: se observan trayectorias de un experto que se presume actúa de forma (aproximadamente) óptima, y se busca **inferir la función de recompensa que explica ese comportamiento**. La motivación es que, en muchos problemas reales, la recompensa es justamente lo difícil de especificar —conducir "como un buen conductor humano" es fácil de reconocer pero difícil de escribir como una fórmula.

El marco de Abbeel & Ng (2004), *Apprenticeship learning via inverse reinforcement learning* (referencia [1] del paper), introdujo la idea que este trabajo hereda directamente: modelar la recompensa como una **combinación lineal de features** (características) del estado,

$$R(s) = w^\top \phi(s) = \sum_{k=1}^{p} w_k\, \phi_k(s),$$

y observar que, para esa clase de recompensas, el valor esperado de una política depende únicamente de las **expectativas de features** (feature expectations) —el valor acumulado (o esperado y descontado) de cada característica bajo esa política. Si dos políticas igualan sus expectativas de features, obtienen el mismo valor bajo *cualquier* $w$. Por lo tanto, para imitar al experto basta con **hacer coincidir las expectativas de features del aprendiz con las del experto**, sin necesidad de recuperar exactamente el $w$ "verdadero" (que, de hecho, está sub-determinado). El algoritmo alterna entre proponer pesos $w$ (que separan al experto de las políticas ya encontradas, en un sentido de máximo margen) y resolver el problema de RL bajo esos pesos.

El puente hacia la conducción autónoma es natural y, en 2008, muy oportuno. El grupo de Stanford venía de ganar el **DARPA Grand Challenge (2005)** con Stanley y de competir con **Junior** en el **DARPA Urban Challenge (2007)**. El algoritmo de planificación de este paper "se basa en el algoritmo usado por el Stanford racing team en el DARPA Urban Challenge" (referencia [7], Dolgov et al.), esto es, el planificador de dos fases con búsqueda global tipo A\* en el estado cinemático del vehículo más suavizado local continuo —lo que hoy conocemos como **hybrid A\***. El paper toma ese planificador probado en competencia y le añade la capa de aprendizaje: en vez de sintonizar los pesos a mano para cada estilo de maniobra, los aprende de demostraciones.

## 3. Contribución central

La contribución puede resumirse en tres puntos:

1. **Adaptación de apprenticeship learning al setting de campos de potencial.** El algoritmo de Abbeel & Ng estaba formulado sobre MDP discretos; aquí se lo reformula sobre el problema de **minimización de un potencial sobre trayectorias continuas**, donde "resolver el MDP" se reemplaza por "correr el planificador de movimiento". El paper muestra que la equivalencia entre valor y expectativas de features se traslada de forma directa: la analogía es política $\leftrightarrow$ trayectoria, recompensa $\leftrightarrow$ potencial (costo), y expectativas de features $\leftrightarrow$ valores acumulados de los términos del potencial.

2. **Incorporación de conocimiento previo mediante restricciones convexas.** El algoritmo admite un conjunto convexo $W$ que codifica información previa sobre los pesos. En los experimentos, por ejemplo, se impone que el peso de la marcha atrás sea **al menos tan alto** como el de la marcha adelante —capturando la idea de que ir en reversa nunca debe ser preferible por defecto, solo un atajo ocasional.

3. **Una variante de máximo margen que permite superar al experto.** A diferencia de la formulación original, que aprende a "igualar" el comportamiento del experto, la versión de este paper (afín a Syed & Schapire [13]) explota que se conoce el **signo** de cada término del costo —todos los términos son penalizaciones, luego $w \ge 0$— para aprender pesos que permiten desempeñarse **igual o mejor** que el experto, no solamente imitarlo.

Un matiz importante que el paper enfatiza y que la Clase 33 debe subrayar: la similitud se aprende **a nivel de los términos de costo, no a nivel geográfico**. Cuando el planificador aprende a "cortar camino" a través de un espacio abierto, puede hacerlo en un lugar físico distinto al del experto, porque la ubicación geográfica del atajo no contribuye a la función de costo. Se imita el *estilo* codificado en las features, no la trayectoria literal.

## 4. Método

### 4.1. Planificación como optimización de un potencial

Sea $S$ el espacio de estados del robot y una trayectoria $s$ una secuencia de estados. El planificador minimiza un potencial total que es lineal en sus términos:

$$\Phi(s) = \sum_{k=1}^{p} w_k\, \phi_k(s), \qquad \min_{s \in S} \Phi(s),$$

sujeto a que $s$ empiece en el estado inicial $s_0$ y termine en la meta $s_G$ (más restricciones como distancia máxima entre estados sucesivos). Cada $\phi_k(\cdot)$ es un **término de potencial** (un feature de la trayectoria completa) y $w \in \mathbb{R}^p$ es el vector de pesos que se quiere aprender. Un problema de planificación se denota $M = (S, s_0, s_G, \{\phi_k\}, w)$, y $\bar M$ es el mismo problema **sin** el vector de pesos. Los términos, en general, definen un potencial complejo con un paisaje de optimización no lineal y multimodal —de ahí que el planificador real sea de dos fases.

Nótese que, a diferencia del RL clásico, muchos términos **no se descomponen** como suma de contribuciones que dependen de un solo instante $t$: la suavidad, por ejemplo, depende de estados consecutivos. El algoritmo no requiere esa descomposición, lo cual es una de las razones por las que se necesita adaptarlo del setting MDP puro.

### 4.2. Features de la trayectoria

El corazón "de ingeniería" del método es el diseño de los términos $\phi_k$. Para generar trayectorias que se parezcan a las de un humano en un estacionamiento, el potencial (Eqn. 3 del paper) combina siete términos:

1. **Longitud recorrida hacia adelante:** $\sum_{i:\,\delta_i=0} \lVert x_i - x_{i-1}\rVert$, con peso $w_{fwd}$.
2. **Longitud recorrida en reversa:** $\sum_{i:\,\delta_i=1} \lVert x_i - x_{i-1}\rVert$, con peso $w_{rev}$.
3. **Número de cambios de dirección** (forward $\leftrightarrow$ reverse): $\sum_{i:\,\delta_i \neq \delta_{i-1}} 1$, con peso $w_{sw}$.
4. **Longitud recorrida fuera del camino** (off-road): $\sum_{i:\,R(s_i)=0} \lVert x_i - x_{i-1}\rVert$, con peso $w_{road}$.
5. **Distancia agregada al grafo de carriles:** $\sum_i D(x_i, \theta_i, G)$, con peso $w_{lane}$.
6. **Desalineación con las direcciones principales** del estacionamiento: $\sum_i \sin^2\!\big(2(\theta_i - \alpha_i)\big)$, con peso $w_{dir}$.
7. **Suavidad** (curvatura agregada): $\sum_{i} (\Delta x_{i+1} - \Delta x_i)^2$, con peso $w_{curv}$, donde $\Delta x_i = x_i - x_{i-1}$.

El estado cinemático del vehículo es $\langle x, \theta, \delta\rangle$, con $x = \langle x, y\rangle$ la posición, $\theta$ la orientación y $\delta \in \{0,1\}$ la dirección de marcha (adelante o atrás). La red de carriles se modela como un grafo dirigido $G = \langle V, E\rangle$, y se definen distancias punto-a-grafo $D(x, G) = \min_E D(E, x)$ y su versión orientada $D(x, \theta, G)$, que solo considera aristas cuyo ángulo $\alpha_E$ está cerca —dentro de $\alpha_{\min}$— de la orientación $\theta$ del auto. Un indicador $R(s)=1$ señala que el auto está "en el camino" ($D(x,G) < D_{road}$), y $L(s)=1$ que está en el carril correcto ($D(x,\theta,G) < D_{lane}$).

Dos features merecen comentario. El término de **distancia a los carriles** distingue a los conductores que cortan camino por el espacio abierto de los que se mantienen en el carril hasta llegar a su meta. El término de **alineación con las direcciones principales** distingue a quienes recortan las curvas para minimizar curvatura de quienes toman curvas más amplias siguiendo las direcciones dominantes del estacionamiento. Aparte de estos, hay términos de **restricciones duras** (evitar colisiones, radio de giro mínimo) que se fijan con pesos enormes —órdenes de magnitud mayores— y **no se incluyen en el aprendizaje**, porque deben satisfacerse siempre, independientemente del estilo.

### 4.3. Aprendizaje de los pesos: el bucle IRL/planner

Sea $\mu_k(\{s^{(i)}\}) = \sum_{i} \phi_k(s^{(i)})$ el valor acumulado del término $k$ sobre un conjunto de trayectorias —el equivalente de la **expectativa de features** en este setting. Dadas $m$ demostraciones expertas $\{s_E^{(i)}\}$ para $m$ problemas $\{\bar M^{(i)}\}$, el algoritmo procede así:

1. Elegir un vector de pesos inicial $w^{(0)}$ al azar. Fijar $j=0$.
2. **Resolver la planificación** con los pesos actuales: para cada problema $i$, hallar $s^{(i)} = \arg\min_s \Phi_{w^{(j)}}(s)$. (Este es el paso "planner": correr hybrid A\* + suavizado con el $w$ vigente.)
3. Calcular los valores acumulados $\mu^{(j)}_k = \sum_i \phi_k(s^{(i)})$.
4. **Actualizar los pesos** resolviendo un programa convexo (paso "IRL"):
$$\min_{w,\,x} \; \lVert w\rVert_2^2 \quad \text{s.a.} \quad \mu = \sum_j x_j\, \mu^{(j)},\; x \ge 0,\; \sum_j x_j = 1,\; w \ge 0,\; w \ge \mu - \mu_E,\; w \in W.$$

Si $\lVert w\rVert \le \epsilon$ para la precisión deseada, terminar; de lo contrario, normalizar $w^{(j)} = w/\lVert w\rVert$, incrementar $j$ y volver al paso 2.

Intuitivamente, el algoritmo **alterna entre "adivinar" inteligentemente un nuevo vector de pesos y resolver la planificación con él**. Adivinar los pesos es solo un problema de optimización convexa, resoluble de forma eficiente. Las restricciones $w \ge 0$ y $w \ge \mu - \mu_E$ codifican que los pesos son positivos y que un término contribuye a la "distancia" respecto del experto solo cuando el experto está superando a la mejor trayectoria actual $\mu^{(j)}$ —esto es lo que permite igualar o superar al experto en vez de meramente imitarlo. La restricción $w \in W$ inyecta el conocimiento previo (por ejemplo, reversa $\ge$ adelante).

Al terminar se garantiza $\lVert \mu - \mu_E\rVert \le \lVert w\rVert \le \epsilon$: eligiendo estocásticamente (según $x$) entre las trayectorias/pesos encontrados a lo largo de las iteraciones, se puede rendir tan bien como el experto hasta la precisión $\epsilon$. Como mezclar estocásticamente es indeseable en la práctica, se inspeccionan los pesos $w^{(j)}$ con $x^{(j)} > 0$ y se escoge uno; el análisis convexo garantiza una solución con a lo sumo $p+1$ entradas no nulas y que al menos una rinde como el experto. En los experimentos, típicamente bastó con **elegir el conjunto de pesos cuyos conteos acumulados de costo quedaban más cerca de los del experto** entre todas las iteraciones.

### 4.4. Integración con el planificador de dos fases

El planificador es el mismo esquema de dos fases de Dolgov et al. [7] —lo que se conoce como **hybrid A\***:

- **Fase I — búsqueda global.** Una variante de **A\*** con un conjunto discreto de acciones de control, aplicada al estado cinemático 4-dimensional del vehículo. Por la discretización gruesa, esta fase no puede acomodar limpiamente los términos que dependen de propiedades locales (suavidad, alineación), así que solo optimiza el subconjunto de features **globales**: $\langle w_{fwd}, w_{rev}, w_{sw}, w_{road}, w_{lane}\rangle$. Además, por costo computacional, la atracción continua hacia el carril se reemplaza por una versión discreta análoga al indicador on-road: penaliza la longitud recorrida fuera del carril, $w'_{lane}\sum_{i:\,L(s_i)=0}\lVert x_i - x_{i-1}\rVert$.
- **Fase II — suavizado local.** La trayectoria de A\* (subóptima por la discretización) se afina con **gradiente conjugado**, una técnica de optimización numérica continua muy eficiente. Como el comportamiento global ya está fijado, esta fase solo usa los términos **locales**: $\langle w_{dir}, w_{curv}, w_{lane}\rangle$, cuyo gradiente puede calcularse analíticamente.

La consecuencia metodológica es elegante: como las features de las dos fases **no se intersectan**, el aprendizaje también se corre en dos fases —primero se aprenden los pesos del planificador global, y luego, con esos pesos fijos, se aprenden los pesos del suavizador. Típicamente el algoritmo converge a una buena solución en **5 a 10 iteraciones**.

## 5. Experimentos y resultados

Los datos se recolectaron con **Junior**, pero sin usar su capacidad de conducción autónoma: un humano manejaba mientras se registraban los mensajes del sistema de pose GPS+IMU y del LIDAR 3D, lo que permitió reconstruir después los mapas de obstáculos exactos y las trayectorias precisas.

Se pidió al conductor humano navegar un estacionamiento con **tres estilos muy distintos**:

- **"Nice"** (prolijo): manejar en el carril derecho siempre que sea seguro.
- **"Sloppy"** (descuidado): permitido desviarse de los carriles estándar; solo marcha adelante.
- **"Backward"** (con reversa): permitido ir en reversa, pero solo cuando acorta el camino a la meta.

Para cada estilo se recolectaron **cinco demostraciones** y se corrió el algoritmo cinco veces con **validación cruzada leave-one-out**: se aprende de cuatro demostraciones y se evalúa en la quinta, dejada fuera. En total, **15 experimentos** de aprendizaje/prueba (3 estilos × 5 folds).

Los resultados cualitativos (Figs. 4–9) muestran que los estilos aprendidos son **muy similares** a los del experto:

- Aprendiendo de cuatro demostraciones "nice", el planificador aprende a **mantenerse en el carril derecho** siempre que puede.
- Aprendiendo de demostraciones "backward", aprende que la reversa está permitida **para hacer un atajo**, y ejecuta exitosamente ese atajo en la tarea de prueba no vista.
- Aprendiendo del conductor "sloppy", aprende a **cortar camino a través de espacios de estacionamiento** cuando corresponde.

Cuantitativamente (Tabla I del paper), el algoritmo encuentra pesos tales que, **tanto en entrenamiento como en prueba**, los valores acumulados de las funciones de costo son cercanos a los del experto. Y los pesos aprendidos coinciden con la intuición sobre cada estilo:

- Para el conductor **"nice"**, la penalización por ir en reversa, salir del carril o salir del camino es **mucho más alta** que para los otros dos estilos (por ejemplo, pesos de off-road del orden de $\sim 4$–$20$ frente a $\sim 2$ en "sloppy").
- Para el estilo **"backward"**, el costo de ir en reversa es **tan bajo como el de ir hacia adelante** —consistente con la restricción $w \in W$ que fuerza reversa $\ge$ adelante, capturando que la reversa solo se aprende como recurso de atajo y no como estilo por defecto.

Las entradas del vector de pesos resultaron además **razonablemente consistentes entre distintas corridas de entrenamiento**. El paper también reporta una salvedad honesta: como el planificador **no es óptimo** (por la discretización), la convergencia del aprendizaje solo puede garantizarse hasta la precisión del propio planificador; aun así, en la práctica el algoritmo encontró buenos óptimos de forma consistente, y de los 15 experimentos solo en uno la heurística de "elegir los pesos más cercanos al experto" devolvió un mal conjunto (que se corrigió inspeccionando manualmente).

## 6. Limitaciones

- **El planificador no es óptimo.** Como Fase I usa un conjunto muy discretizado de acciones de control, puede no encontrar la trayectoria óptima para un $w$ dado. Esto rompe el supuesto teórico de "resolver el problema" en cada iteración y acota las garantías de convergencia a la precisión del planificador.
- **Features hechas a mano.** El método aprende *pesos*, pero la elección de los siete términos $\phi_k$ (longitud, reversa, cambios de dirección, off-road, distancia a carriles, alineación, suavidad) sigue siendo ingeniería manual. IRL aquí ahorra el ajuste del *balance*, no el diseño de las características.
- **Grafo de carriles y direcciones dadas de antemano.** El planificador supone que la red de carriles $G$ se le entrega como entrada; las direcciones principales se derivan de esa red (aunque los autores citan [6] para computarlas de sensores).
- **Restricciones duras fuera del aprendizaje.** Colisiones y radio de giro se fijan con pesos enormes a mano y no se aprenden, lo cual es sensato pero implica que parte del comportamiento crítico sigue siendo especificado manualmente.
- **Escala del experimento.** El estudio es una prueba de concepto: un estacionamiento, un vehículo, tres estilos, cinco demostraciones por estilo. No es una evaluación a gran escala ni una demostración de conducción autónoma cerrada de extremo a extremo (los datos se recolectaron con manejo manual).

## 7. Conexión con la Clase 33 (Aprendizaje por Imitación e IRL)

Este paper es el ejemplo que el profesor Toro Icarte usa (slide 25) para ilustrar **IRL aplicado**, y funciona tan bien como ejemplo por varias razones que conviene explicitar:

- **Es IRL genuino, no clonación de comportamiento.** La clonación de comportamiento (behavioral cloning) aprendería un mapa directo estado $\to$ acción imitando las decisiones del experto. Aquí, en cambio, se **recupera la función de costo** que el experto parece estar minimizando, y luego se planifica con ella. La diferencia práctica es la **generalización**: el planificador reproduce el *estilo* en tareas nuevas (start/goal no vistos), no la trayectoria memorizada —de ahí que pueda cortar camino "en otro lugar" que el experto.

- **Materializa las expectativas de features.** El concepto abstracto de "hacer coincidir feature expectations" de Abbeel & Ng 2004 se vuelve tangible: los $\mu_k$ son literalmente los conteos acumulados de longitud en reversa, cambios de dirección, distancia a carriles, etc. El estudiante ve que "igualar el comportamiento experto" se reduce a **igualar estadísticas interpretables de la trayectoria**.

- **Ilustra la diferencia entre RL y IRL.** En RL clásico, el costo (o recompensa) está dado y se busca la trayectoria óptima; aquí el costo es **precisamente lo desconocido** y se lo infiere de demostraciones. El paso "resolver la planificación" hace de subrutina de RL (encontrar la mejor trayectoria dado $w$), envuelto en un bucle externo que ajusta $w$ —el bucle IRL/planner. Es el esqueleto del IRL basado en máximo margen.

- **Demuestra la ventaja de motivación del IRL.** El paper abre justificando IRL de la forma que la clase enfatiza: es difícil escribir la recompensa/costo a mano, pero fácil demostrar buen comportamiento. Estacionar es el caso perfecto —nadie sabe decir "el peso de la suavidad debe ser 100 y el de la reversa 96,8", pero cualquiera sabe manejar prolijamente y dejar que el algoritmo lea los pesos de esa demostración.

Este trabajo también conecta linealmente con la genealogía del laboratorio de Stanford hacia la conducción autónoma moderna (Stanley, Junior, hybrid A\*), mostrando que IRL no es un ejercicio de pizarra sino una herramienta desplegada sobre un vehículo real de investigación.

## 8. Nota final: conexión con salud y MDM (lector experto FHIR)

Para un lector que construye sistemas de *Master Data Management* de pacientes sobre FHIR, este paper ofrece una analogía sorprendentemente directa. Un motor de deduplicación (record linkage) puntúa pares de recursos `Patient` con una **función de scoring que es, en el fondo, una combinación de features**: coincidencia de nombre, de fecha de nacimiento, de RUT/identificador, distancia fonética, similitud de dirección, etc. Sintonizar a mano los **pesos** de esas reglas —cuánto vale una coincidencia exacta de nombre frente a una de fecha de nacimiento— es exactamente el problema tedioso y poco intuitivo que Abbeel et al. describen para los términos del potencial de conducción. La lección transferible es que, en vez de calibrar esos pesos a mano, se pueden **aprender de las decisiones de los expertos**: las resoluciones históricas de un data steward (los pares que un humano confirmó como "mismo paciente" o "distintos") son las "demostraciones de trayectorias", y el bucle IRL/scorer ajusta los pesos de las reglas hasta que las estadísticas agregadas del sistema (las "feature expectations" de match/no-match) coincidan con las del experto. Igual que aquí se aprende a estacionar "como un humano" recuperando el costo implícito, un MDM puede aprender a deduplicar "como el mejor steward" recuperando el scoring implícito en sus decisiones —conservando además la interpretabilidad de features clínicas explícitas, algo esencial cuando una fusión de pacientes errónea tiene consecuencias de seguridad del paciente.
