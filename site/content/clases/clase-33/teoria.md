---
title: "Teoría - Imitación e IRL"
weight: 10
math: true
---

> **Recorrido de la Clase 33** del Diplomado IA UC (Rodrigo Toro Icarte). Esta clase es la **continuación directa** de la [Clase 31 (Aprendizaje Reforzado)](/clases/clase-31): si allí el agente aprendía por ensayo y error a partir de una recompensa dada, aquí se atacan tres preguntas que el RL puro deja abiertas. ¿Las políticas aprendidas por RL **generalizan** a situaciones nuevas? ¿Qué hacemos cuando **no sabemos definir la recompensa**? ¿Y si en vez de aprender por ensayo y error simplemente **imitamos a un experto**? La clase discute cuatro ideas: generalización en RL, aprendizaje reforzado inverso (IRL), aprendizaje por imitación, y la comparación entre refuerzo e imitación.

---

## 1. Recordatorio: supervisado y reforzado

La clase abre repasando los dos paradigmas ya vistos, porque toda la sesión vive en la tensión entre ambos.

En el **aprendizaje supervisado** tenemos un conjunto de entrenamiento $T$ de pares $(x, y)$ y buscamos parámetros $\theta$ tales que $f_\theta(x) \approx y$ para todo $(x, y) \in T$. Dos hechos lo caracterizan: (1) el problema se resuelve por **descenso de gradiente**, y (2) esperamos que $f_\theta(x)$ **generalice** a instancias fuera del conjunto de entrenamiento. Esta noción de generalización —y su vigilancia contra el overfitting— será central hoy.

En el **aprendizaje reforzado**, en cambio, un **agente** interactúa con un **ambiente** en un bucle: ejecuta una acción $a$, y el ambiente le devuelve un nuevo estado $s$ y una recompensa $r$. El agente sigue una **política** $\pi(a\mid s)$ y su objetivo es **ajustar esa política para obtener la mayor recompensa acumulada** posible. La [Clase 31](/clases/clase-31) desarrolló en detalle cómo el agente resuelve este problema (MDP, Q-Learning, DQN).

{{< concept-alert type="recordar" >}}
La clase de hoy discutirá **cuatro ideas**: (1) generalización en aprendizaje reforzado, (2) aprendizaje reforzado inverso, (3) aprendizaje por imitación, y (4) aprendizaje reforzado *versus* aprendizaje por imitación.
{{< /concept-alert >}}

---

## 2. Generalización en aprendizaje reforzado

### 2.1 La disciplina del supervisado

En aprendizaje supervisado la práctica es rigurosa: se **entrena** (fase lenta) y se **testea** (fase rápida) en conjuntos **separados**, precisamente para detectar el **overfitting** —cuando el modelo memoriza el entrenamiento en vez de aprender un patrón transferible. La regla de oro es inviolable: *nunca uses tus datos de test para entrenar tu modelo*.

### 2.2 ¿El RL generaliza?

Aquí viene la pregunta incómoda. En aprendizaje reforzado, la práctica habitual durante años fue **entrenar y testear en el mismo dominio**: entrenar un agente en un juego y reportar su puntaje en ese mismo juego, con el mismo layout y la misma semilla. Eso **oculta por completo** si el agente aprendió a jugar o solo memorizó una secuencia de acciones.

La clase apoya la discusión en tres trabajos:

- **Witty et al. — *Measuring and Characterizing Generalization in Deep RL*.** Abre el problema: no basta con un número agregado; hay que **caracterizar dónde y cómo falla** un agente aparentemente competente al enfrentar estados legítimos pero fuera de su distribución de entrenamiento. → [paper](/papers/generalization-rl-witty-2018)
- **Zhang et al. — *A Study on Overfitting in Deep Reinforcement Learning*.** Diseña un **gridworld** con un MDP $M = \langle S, A, r, p, \mu\rangle$: recompensa de **diamante** $(+1)$, **rayo** $(-1)$, **bomba** $(-1)$ y **llave** $(+0.1$, termina el episodio$)$. La pieza clave es $\mu$, un **generador de mapas iniciales**, con el que se produce un conjunto $\bar S_0$ de configuraciones (posiciones del agente, murallas y objetos) que se **divide en train y test**. El hallazgo: los agentes **memorizan** y fallan en configuraciones nuevas. → [paper](/papers/overfitting-rl-zhang-2018)
- **Cobbe et al. — *Quantifying Generalization in Reinforcement Learning*.** Replica el experimento en un dominio más complejo, **CoinRun**, con niveles generados proceduralmente, y **cuantifica** la brecha: se necesitan **miles de niveles** de entrenamiento para cerrarla. → [paper](/papers/quantifying-generalization-cobbe-2019)

### 2.3 La respuesta

{{< concept-alert type="clave" >}}
**¿Podemos aprender políticas que generalicen usando aprendizaje reforzado?** Sí, **pero debemos entrenar en ambientes que varíen en las dimensiones en que queremos generalizar.** Igual que en supervisado, la generalización emerge de la **diversidad del entrenamiento**, no aparece gratis. Profundización en el fundamento [Generalización en RL](/fundamentos/generalizacion-en-rl).
{{< /concept-alert >}}

---

## 3. Aprendizaje reforzado inverso (IRL)

### 3.1 El problema de definir la recompensa

El objetivo del agente es maximizar recompensa. El problema: un agente **está dispuesto a hacer cualquier cosa con tal de conseguirla**, así que definir una buena recompensa es sorprendentemente difícil. El ejemplo de la clase es **conducir**: ¿cuál es una buena recompensa?

- ¿Llegar al punto de destino?
- ¿No cruzar en luz roja?
- ¿No matar a nadie?
- ...

Ninguna lista manual captura del todo lo que significa "conducir bien". De ahí la pregunta: **¿será posible aprender la función de recompensa?**

### 3.2 La observación clave y su formalización

{{< concept-alert type="clave" >}}
**Si observamos el comportamiento de un agente experto, es posible inferir la recompensa que está optimizando.** Esa es toda la idea del aprendizaje reforzado inverso.
{{< /concept-alert >}}

Matemáticamente, buscamos una función $R : S \to \mathbb{R}$ tal que la política del experto $\pi^*$ sea al menos tan buena como cualquier otra:

$$
\mathbb{E}_{\pi^*}\!\left[\sum_{t=0}^{\infty} \gamma^t R(s_t)\right] > \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^t R(s_t)\right] \quad \forall\, \pi \neq \pi^*.
$$

Como esta condición es ambigua (la recompensa trivial $R=0$ la satisface), se reformula **maximizando el margen** $m$ con el que el experto supera a las demás políticas:

$$
\max_{R:S\to\mathbb{R}} m \quad \text{s.a.}\quad \mathbb{E}_{\pi^*}\!\left[\sum_t \gamma^t R(s_t)\right] \ge \mathbb{E}_{\pi}\!\left[\sum_t \gamma^t R(s_t)\right] + m \quad \forall\, \pi \neq \pi^*.
$$

### 3.3 La estrategia incremental

Enumerar todas las políticas $\pi \neq \pi^*$ es imposible, así que se usa un bucle incremental: se elige una política inicial $\pi_0$, se resuelve para $R_1$ usándola como único rival, se computa la política óptima $\pi_1$ bajo $R_1$, y si $\pi_1$ resulta mejor que el experto se la agrega al modelo y se resuelve $R_2$, y así hasta que ninguna política nueva supere a $\pi^*$. Cada paso requiere resolver un RL completo por dentro —lo que hace al IRL clásico costoso.

### 3.4 Ejemplos: conducir y estacionar

La clase muestra dos aplicaciones históricas de Pieter Abbeel y Andrew Ng:

- **Aprender a conducir** con distintos estilos, vía *apprenticeship learning* (igualar las **expectativas de features** del experto). → [Abbeel & Ng, 2004](/papers/apprenticeship-abbeel-ng-2004)
- **Aprender a estacionar** un auto robótico, aprendiendo la **función de costo** del planificador de movimiento a partir de demostraciones. → [Abbeel et al., 2008](/papers/apprenticeship-parking-abbeel-2008)

{{< concept-alert type="recordar" >}}
**¿Qué pasa si no sé cómo definir una función de recompensa para mi problema?** No importa: **la puedes aprender de demostraciones expertas**. Profundización en el fundamento [Aprendizaje Reforzado Inverso](/fundamentos/aprendizaje-reforzado-inverso), que cubre además MaxEnt IRL ([Ziebart et al., 2008](/papers/maxent-irl-ziebart-2008)) y GAIL ([Ho & Ermon, 2016](/papers/gail-ho-ermon-2016)).
{{< /concept-alert >}}

---

## 4. Aprendizaje por imitación

### 4.1 La idea directa

El IRL usa demostraciones en **dos pasos**: primero aprende una recompensa $R$, y luego una política $\pi^*(a\mid s)$ mediante RL sobre $R$. La clase plantea el atajo natural:

> *[wait] ¿no sería más fácil aprender $\pi^*(a\mid s)$ directo, imitando a los expertos?*

Esa es la idea del **aprendizaje por imitación** (o *behavioral cloning*). Dejamos que expertos resuelvan la tarea y **registramos qué acción ejecutan en cada estado**. Con esa información construimos un **conjunto de entrenamiento supervisado** (estado → acción del experto) y entrenamos $\pi_\theta(s)$ como un clasificador estándar. Una vez entrenada, $\pi_\theta(a\mid s)$ **debería generalizar** a situaciones nuevas.

### 4.2 Funciona bastante bien... pero no excelente

En general, el aprendizaje por imitación **funciona bastante bien**. El ejemplo estelar de la clase es **Gato** (Reed et al., 2022): un único agente que resuelve gran cantidad de tareas —Atari, subtitulado, chat, control de un brazo robótico real— entrenado como imitación masiva de demostraciones. → [paper](/papers/gato-reed-2022)

Pero *tampoco funciona excelente*. El aprendizaje por imitación tiende a **fallar por dos motivos principales**, ambos derivados del mismo defecto: el control **no es** un problema i.i.d. La política, al actuar, visita estados que **ella misma genera**; en cuanto comete un pequeño error, aterriza en un estado poco parecido a los del experto, donde predice peor, y los errores **se acumulan** (*compounding error*).

### 4.3 DAgger: dataset aggregation

La solución que discute la clase es **DAgger** (Ross et al., 2011): recolectar etiquetas expertas precisamente sobre los **estados que la política visita**. Se rueda la política actual, se **consulta al experto** qué acción tomaría en esos estados, se **agregan** los nuevos pares al dataset y se **reentrena**, iterando. Su análisis como aprendizaje online sin arrepentimiento garantiza que el error crece **linealmente** (no cuadráticamente) en el horizonte. → [paper](/papers/dagger-ross-2011). Este es el algoritmo que implementa el [laboratorio de la clase](/laboratorios/lab-33) sobre Atari Breakout.

{{< concept-alert type="advertencia" >}}
**¿Podemos aprender una buena política imitando a expertos?** Sí, **pero procura que $\pi_\theta(a\mid s)$ se mantenga dentro de su zona de entrenamiento.** Apenas la política se sale de la distribución de estados que vio el experto, no hay garantía sobre su comportamiento. Profundización en el fundamento [Aprendizaje por Imitación](/fundamentos/aprendizaje-por-imitacion).
{{< /concept-alert >}}

---

## 5. Refuerzos vs. imitación

### 5.1 Caso de estudio: AlphaGo Zero

El contraste se ilustra con **AlphaGo Zero** (Silver et al., Nature 2017): un sistema que aprende Go **por auto-juego puro, sin usar ninguna partida humana**, y que **supera** a todas las versiones anteriores de AlphaGo —incluida la de 2016, que sí partía de imitación de partidas humanas. → [paper](/papers/alphago-zero-silver-2017). Es la evidencia más nítida de que el RL puro puede alcanzar desempeño **sobrehumano**, mientras que la imitación rara vez supera a su maestro.

### 5.2 La tabla comparativa

| | **Aprendizaje reforzado** | **Aprendizaje por imitación** |
|---|---|---|
| Demostraciones de expertos | No las necesita ✓ | Las requiere ✗ |
| Techo de desempeño | Puede superar a los humanos ✓ | No suele superar a su maestro ✗ |
| Velocidad de aprendizaje | Aprende muy lento ✗ | Aprende una buena política rápido ✓ |
| Seguridad en entrenamiento | Puede explorar acciones inseguras ✗ | No requiere exploración peligrosa ✓ |

### 5.3 Refuerzos + imitación

Por eso muchos sistemas reales **combinan ambos**: arrancar con imitación (rápido y seguro, para partir de una política razonable) y refinar con RL (para superar el techo humano). AlphaGo 2016 hizo justamente esto —aprendizaje supervisado de partidas humanas, seguido de RL por auto-juego— antes de que AlphaGo Zero mostrara que, en un dominio con modelo perfecto del mundo, el RL puro basta.

---

## 6. Resumen

La clase discutió cuatro ideas:

1. **Generalización en RL.** ¿Podemos aprender políticas que generalicen? Sí, pero entrenando en ambientes que varíen en las dimensiones donde queremos generalizar.
2. **Aprendizaje reforzado inverso.** ¿Qué hago si no sé definir la recompensa? La puedes aprender de demostraciones expertas.
3. **Aprendizaje por imitación.** ¿Puedo aprender una buena política imitando expertos? Sí, pero cuida que la política se mantenga en su zona de entrenamiento.
4. **Refuerzo vs. imitación.** El refuerzo aprende de forma autónoma a resolver problemas pero es lento; la imitación aprende rápido pero requiere demostraciones y no suele superar al maestro.

---

**Ver también:** [Clase 33 - Profundización](/clases/clase-33/profundizacion) · [Clase 33 - Práctica](/clases/clase-33/practica) · [Laboratorio: DAgger sobre Breakout](/laboratorios/lab-33) · Fundamentos: [IRL](/fundamentos/aprendizaje-reforzado-inverso) · [Imitación](/fundamentos/aprendizaje-por-imitacion) · [Generalización en RL](/fundamentos/generalizacion-en-rl).
