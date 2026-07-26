---
title: "Apprenticeship Learning para Motion Planning: Parking (2008)"
weight: 370
math: true
---

{{< paper-card
    title="Apprenticeship Learning for Motion Planning with Application to Parking Lot Navigation"
    authors="Pieter Abbeel, Dmitri Dolgov, Andrew Y. Ng, Sebastian Thrun (Stanford)"
    year="2008"
    venue="IROS 2008"
    pdf="/papers/apprenticeship-parking-abbeel-2008.pdf" >}}
Este trabajo toma un problema aparentemente prosaico —hacer que un auto planifique cómo moverse dentro de un estacionamiento— y lo convierte en un caso de estudio limpio de [aprendizaje reforzado inverso](/fundamentos/aprendizaje-reforzado-inverso). Los planificadores de movimiento usan **funciones de costo complejas** con muchos términos que compiten (suavidad, distancia a obstáculos, curvatura, mantenerse en el carril, evitar la marcha atrás). Ajustar a mano el peso de cada término es tedioso y poco intuitivo. La tesis del paper: **es mucho más fácil demostrar unas pocas trayectorias buenas que especificar a mano los pesos**, porque esas demostraciones ya contienen el compromiso deseado. El aporte es un algoritmo que, dadas unas pocas demostraciones de un conductor humano, **infiere automáticamente los pesos** de modo que el planificador (**hybrid A\***) reproduzca el estilo demostrado. Es el ejemplo canónico de "aprender a estacionar por IRL" que la [Clase 33](/clases/clase-33) cita en su slide 25. Adaptan la maquinaria de [apprenticeship learning de Abbeel & Ng (2004)](/papers/apprenticeship-abbeel-ng-2004), originalmente para MDPs, al escenario de campos de potencial sobre trayectorias continuas.
{{< /paper-card >}}

---

## Contexto: de Abbeel & Ng (2004) a la conducción autónoma

La distinción central de la [Clase 33](/clases/clase-33) es la que separa el RL del IRL. En el **aprendizaje reforzado clásico** se supone conocida la recompensa $R(s)$ y se busca la política óptima. En el **aprendizaje reforzado inverso** el problema se invierte: se observan trayectorias de un experto (aproximadamente) óptimo y se busca **inferir la función de recompensa que explica ese comportamiento**. La motivación es que en problemas reales la recompensa es justamente lo difícil de especificar: conducir "como un buen conductor humano" es fácil de reconocer pero difícil de escribir como fórmula.

El marco de [Abbeel & Ng (2004)](/papers/apprenticeship-abbeel-ng-2004) —del que este trabajo hereda directamente— modela la recompensa como combinación lineal de features del estado, $R(s) = w^\top \phi(s)$, y observa que el valor esperado de una política depende únicamente de sus **expectativas de features**. Por lo tanto, para imitar al experto basta con **hacer coincidir las expectativas de features del aprendiz con las del experto**, sin recuperar exactamente el $w$ "verdadero" (que está sub-determinado). El puente hacia la conducción autónoma era natural en 2008: el grupo de Stanford venía de ganar el DARPA Grand Challenge (2005) con *Stanley* y de competir con *Junior* en el DARPA Urban Challenge (2007). El planificador de este paper es el mismo de esa competencia —**hybrid A\***— al que se le añade la capa de aprendizaje.

## Método: aprender la función de costo del planner

### Planificación como optimización de un potencial

El planificador minimiza un potencial total lineal en sus términos:

$$\Phi(s) = \sum_{k=1}^{p} w_k\, \phi_k(s), \qquad \min_{s \in S} \Phi(s),$$

donde cada $\phi_k(\cdot)$ es un **feature de la trayectoria completa** y $w \in \mathbb{R}^p$ es el vector de pesos a aprender. A diferencia del RL clásico, muchos términos **no se descomponen** por instante (la suavidad depende de estados consecutivos), razón por la que hay que adaptar el algoritmo desde el setting MDP puro.

El potencial combina **siete términos** de la trayectoria: longitud hacia adelante, longitud en reversa, número de cambios de dirección, longitud fuera del camino (*off-road*), distancia agregada al grafo de carriles, desalineación con las direcciones principales del estacionamiento y suavidad (curvatura agregada). El estado cinemático del vehículo es $\langle x, \theta, \delta\rangle$: posición, orientación y dirección de marcha $\delta \in \{0,1\}$. Aparte están los términos de **restricciones duras** (colisiones, radio de giro mínimo) que se fijan con pesos enormes y **no se incluyen en el aprendizaje**, porque deben satisfacerse siempre, independientemente del estilo.

### El bucle IRL/planner

Sea $\mu_k(\{s^{(i)}\}) = \sum_{i} \phi_k(s^{(i)})$ el valor acumulado del término $k$ —el equivalente de la **expectativa de features** en este setting. El algoritmo alterna entre "adivinar" inteligentemente un vector de pesos y resolver la planificación con él:

1. Elegir $w^{(0)}$ al azar.
2. **Resolver la planificación** con los pesos actuales (correr hybrid A\* + suavizado con el $w$ vigente).
3. Calcular los valores acumulados $\mu^{(j)}_k$.
4. **Actualizar los pesos** resolviendo un programa convexo que minimiza $\lVert w\rVert_2^2$ sujeto a $w \ge 0$, $w \ge \mu - \mu_E$ y $w \in W$.

Las restricciones $w \ge 0$ (los pesos son positivos, todos son penalizaciones) y $w \ge \mu - \mu_E$ implementan una **variante de máximo margen** que —a diferencia de la formulación original que solo "iguala" al experto— permite desempeñarse **igual o mejor** que él. El conjunto convexo $W$ inyecta conocimiento previo: por ejemplo, forzar que el peso de la marcha atrás sea al menos tan alto como el de la marcha adelante, capturando que ir en reversa nunca debe ser preferible por defecto. Al terminar se garantiza $\lVert \mu - \mu_E\rVert \le \lVert w\rVert \le \epsilon$.

### Integración con el planificador de dos fases

El **hybrid A\*** tiene dos fases y el aprendizaje sigue la misma estructura:

- **Fase I — búsqueda global.** Una variante de A\* sobre el estado cinemático discretizado del vehículo; por la discretización gruesa solo optimiza los features **globales** (longitud adelante/reversa, cambios de dirección, off-road, carril).
- **Fase II — suavizado local.** La trayectoria de A\* se afina con **gradiente conjugado**, usando solo los términos **locales** (alineación, curvatura, carril), cuyo gradiente es analítico.

Como las features de las dos fases **no se intersectan**, se aprenden primero los pesos del planificador global y luego los del suavizador. El algoritmo converge típicamente en **5 a 10 iteraciones**.

## Resultados

Los datos se recolectaron con **Junior**, el auto robótico de Stanford, pero sin su modo autónomo: un humano manejaba mientras se registraban pose GPS+IMU y LIDAR 3D para reconstruir después mapas y trayectorias exactas. Se pidió al conductor navegar un estacionamiento con **tres estilos**: *nice* (prolijo, siempre en el carril derecho), *sloppy* (descuidado, permite desviarse del carril, solo adelante) y *backward* (permite reversa, pero solo cuando acorta el camino). Por estilo se recolectaron **cinco demostraciones** y se corrió el algoritmo con **validación cruzada leave-one-out**: 15 experimentos en total (3 estilos × 5 folds).

Los estilos aprendidos resultaron **muy similares** a los del experto. Aprendiendo del conductor *nice*, el planificador se mantiene en el carril derecho siempre que puede; del *backward*, aprende que la reversa está permitida para hacer un atajo y lo ejecuta en la tarea no vista; del *sloppy*, aprende a cortar camino a través de espacios de estacionamiento. Cuantitativamente, los valores acumulados de costo quedaron cercanos a los del experto **tanto en entrenamiento como en prueba**, y los pesos aprendidos coincidieron con la intuición: para *nice*, las penalizaciones por reversa, salir del carril o del camino son mucho más altas (off-road del orden de $\sim 4$–$20$ frente a $\sim 2$ en *sloppy*); para *backward*, el costo de reversa quedó tan bajo como el de ir adelante. Un matiz clave: la similitud se aprende **a nivel de los términos de costo, no geográfico** —el planificador puede cortar camino en un lugar físico distinto al del experto, porque imita el *estilo* codificado en las features, no la trayectoria literal.

## Limitaciones

- **El planificador no es óptimo.** La Fase I usa un conjunto discretizado de acciones, así que puede no hallar la trayectoria óptima para un $w$ dado; esto rompe el supuesto teórico de "resolver el problema" en cada iteración y acota las garantías de convergencia a la precisión del planificador.
- **Features hechas a mano.** El método aprende *pesos*, pero la elección de los siete términos $\phi_k$ sigue siendo ingeniería manual. IRL ahorra el ajuste del *balance*, no el diseño de las características.
- **Grafo de carriles dado de antemano.** El planificador supone que la red de carriles $G$ se le entrega como entrada.
- **Restricciones duras fuera del aprendizaje.** Colisiones y radio de giro se fijan a mano con pesos enormes y no se aprenden.
- **Escala del experimento.** Es una prueba de concepto: un estacionamiento, un vehículo, tres estilos, cinco demostraciones por estilo, con datos recolectados por manejo manual.

## Por qué importa para la Clase 33

Este es el ejemplo que el profesor Toro Icarte usa (slide 25) para ilustrar **IRL aplicado a la robótica del mundo real**, y funciona bien como caso de estudio por varias razones:

- **Es IRL genuino, no clonación de comportamiento.** La clonación aprendería un mapa directo estado $\to$ acción; aquí se **recupera la función de costo** que el experto minimiza y luego se planifica con ella. La ventaja práctica es la **generalización**: el planificador reproduce el estilo en tareas nuevas (start/goal no vistos), no la trayectoria memorizada.
- **Materializa las expectativas de features.** El concepto abstracto de "hacer coincidir feature expectations" de [Abbeel & Ng (2004)](/papers/apprenticeship-abbeel-ng-2004) se vuelve tangible: los $\mu_k$ son literalmente los conteos acumulados de longitud en reversa, cambios de dirección, distancia a carriles, etc.
- **Ilustra la diferencia entre RL e IRL.** El paso "resolver la planificación" hace de subrutina de RL (mejor trayectoria dado $w$), envuelto en un bucle externo que ajusta $w$ —el esqueleto del IRL de máximo margen.
- **Demuestra la motivación del IRL.** Nadie sabe decir "el peso de la suavidad debe ser 100 y el de la reversa 96,8", pero cualquiera sabe manejar prolijamente y dejar que el algoritmo lea los pesos de esa demostración.

El trabajo conecta además con la genealogía del laboratorio de Stanford hacia la conducción autónoma moderna (Stanley, Junior, hybrid A\*), mostrando que el IRL no es un ejercicio de pizarra sino una herramienta desplegada sobre un vehículo real de investigación —una pieza central del trabajo en [robótica](/dominios/robotica).
