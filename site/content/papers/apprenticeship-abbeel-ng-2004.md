---
title: "Apprenticeship Learning via Inverse RL (2004)"
weight: 369
math: true
---

{{< paper-card
    title="Apprenticeship Learning via Inverse Reinforcement Learning"
    authors="Pieter Abbeel, Andrew Y. Ng (Stanford)"
    year="2004"
    venue="ICML 2004"
    pdf="/papers/apprenticeship-abbeel-ng-2004.pdf" >}}
Continuación directa del [IRL fundacional de Ng y Russell (2000)](/papers/irl-ng-russell-2000), este paper reorienta la maquinaria del aprendizaje reforzado inverso hacia un objetivo distinto: **no recuperar la recompensa "verdadera" del experto, sino igualar su desempeño**. La motivación es el ejemplo estrella de la [Clase 33](/clases/clase-33): conducir bien implica compensar muchos deseos a la vez (distancia de seguridad, no salirse, evitar peatones, velocidad razonable, preferencia de carril) y asignar los pesos exactos es tan difícil que —dicen los autores— "aunque son capaces de conducir competentemente, no creen poder especificar con confianza una función de recompensa". La clave: es mucho más fácil *demostrar* la tarea que *especificarla*. El resultado teórico notable es que, **aunque el algoritmo nunca recupere la recompensa verdadera**, la política que devuelve alcanza un desempeño comparable al del experto, con pocas demostraciones y en pocas iteraciones. Este es el "aprender a conducir usando IRL" de la slide 24.
{{< /paper-card >}}

---

## Contexto: de recuperar la recompensa a igualar el desempeño

Los métodos previos de aprendizaje por demostración intentaban **imitar directamente al demostrador** —aprender por supervisión un mapeo estado → acción, lo que hoy llamamos *behavioral cloning* (ALVINN de Pomerleau, 1989). El paper señala su falla con el ejemplo de la conducción: **seguir ciegamente la trayectoria del experto no funciona**, porque el patrón de tráfico es distinto cada vez. Como todo el [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado) se funda en que la recompensa —no la política— es la definición más sucinta y transferible de una tarea, lo natural es *aprender la recompensa*, y de ahí el recurso al [aprendizaje reforzado inverso](/fundamentos/aprendizaje-reforzado-inverso).

La sutileza que separa este paper de su predecesor: el IRL puro está **mal condicionado** (muchas recompensas, incluida $R = 0$, explican al experto), y [Ng y Russell (2000)](/papers/irl-ng-russell-2000) atacaban de frente el problema de recuperar una recompensa "buena". Abbeel y Ng dan un paso al costado: **su objetivo no es recuperar la recompensa verdadera**, sino hallar una política cuyo desempeño se acerque al del experto. El paso de IRL aparece dentro del algoritmo, pero las garantías **no dependen de que ese paso acierte la recompensa** —solo de igualar las *feature expectations*. Ese reencuadre es lo que permite dar garantías formales pese a la ambigüedad inherente del IRL.

## Método: feature expectations y recompensa lineal

El supuesto estructural es que la recompensa verdadera es una **combinación lineal de features** conocidas $\phi: S \to [0,1]^k$:

$$R^*(s) = w^* \cdot \phi(s), \qquad \|w^*\|_1 \le 1.$$

En la conducción, $\phi$ es un vector de indicadores (chocamos, vamos en carril central, etc.) y $w^*$ codifica cuánto pesa cada deseo. Sacando $w$ fuera de la esperanza por linealidad, el valor de una política se escribe como un producto interno con las **feature expectations** —el vector de features acumuladas y descontadas que induce la política:

$$\mu(\pi) = E\left[\sum_{t=0}^{\infty} \gamma^t \phi(s_t) \,\middle|\, \pi\right], \qquad E_{s_0 \sim D}[V^\pi(s_0)] = w \cdot \mu(\pi).$$

Esta es la idea que lo habilita todo: **como la recompensa es lineal en las features, las feature expectations determinan completamente el valor de una política para cualquier $w$**. Si dos políticas tienen las mismas $\mu$, tienen el mismo desempeño bajo toda recompensa lineal, sea cual sea el $w$ verdadero. La feature expectation del experto $\mu_E$ se estima por Monte Carlo a partir de sus trayectorias.

**La garantía fundamental.** Si se halla una política $\tilde\pi$ con $\|\mu(\tilde\pi) - \mu_E\|_2 \le \epsilon$, entonces para *cualquier* $w$ con $\|w\|_1 \le 1$:

$$\left| w^\top \mu(\tilde\pi) - w^\top \mu_E \right| \le \|w\|_2\, \|\mu(\tilde\pi) - \mu_E\|_2 \le 1 \cdot \epsilon = \epsilon,$$

por Cauchy-Schwarz y $\|w\|_2 \le \|w\|_1 \le 1$. La diferencia de desempeño respecto del experto es a lo sumo $\epsilon$ **uniformemente sobre todas las recompensas admisibles** —sin conocer nunca $w^*$. Por eso igualar $\mu_E$ es suficiente aunque jamás se identifique la recompensa verdadera.

## El bucle max-margin y su conexión con SVM

El algoritmo halla iterativamente una política con $\mu$ cercana a $\mu_E$. Partiendo de una política aleatoria, en cada iteración resuelve un problema de **máximo margen**:

$$\max_{t,\,w} \; t \quad \text{s.a.} \quad w^\top \mu_E \ge w^\top \mu^{(j)} + t \;\; (j < i), \quad \|w\|_2 \le 1,$$

es decir, busca una recompensa $R = w \cdot \phi$ bajo la cual **el experto lo hace mejor, por un margen $t$, que todas las políticas encontradas hasta ahora**. Esto es exactamente el hiperplano de máximo margen de una **SVM** (Vapnik, 1998): etiqueta $+1$ para $\mu_E$, $-1$ para las $\mu(\pi^{(j)})$, y $w^{(i)}$ es el vector unitario ortogonal al separador. Por la restricción $\|w\|_2 \le 1$ (norma 2, a diferencia de la norma 1 de Ng y Russell) **no es un LP sino un QP**. Luego se resuelve el MDP con esa recompensa candidata (paso RL "forward"), se calcula la nueva $\mu^{(i)}$ y se repite hasta $t^{(i)} \le \epsilon$. El paper también ofrece un *projection method* que evita el QP proyectando ortogonalmente $\mu_E$ sobre la recta entre las dos últimas iteraciones —más simple y algo más rápido.

Al terminar, el algoritmo devuelve un **conjunto** de políticas; existe al menos una cuyo desempeño iguala al del experto salvo $\epsilon$. Para obtener una sola se inspecciona a mano (basta examinar $k+1$ políticas por Carathéodory) o se resuelve un QP que halla el punto más cercano a $\mu_E$ en la **envolvente convexa** de las $\mu^{(i)}$ y se **mezclan** las políticas con esos pesos. El paper subraya que, aunque un paso se llame "IRL", **no recupera necesariamente la recompensa correcta**: todo descansa en igualar feature expectations.

## Resultados: terminación, complejidad de muestra y los cinco estilos

Dos teoremas dan las garantías. **Terminación:** ambas versiones terminan tras a lo sumo $n = O\!\left(\frac{k}{(1-\gamma)^2\epsilon^2}\log\frac{k}{(1-\gamma)\epsilon}\right)$ iteraciones —polinómico en $k$ y $1/\epsilon$ e **independiente del tamaño del espacio de estados $|S|$**, por convergencia geométrica. **Complejidad de muestra:** bastan $m \ge \frac{2k}{(\epsilon(1-\gamma))^2}\log\frac{2k}{\delta}$ trayectorias del experto (vía Hoeffding + union bound) para garantizar, con probabilidad $1-\delta$, desempeño dentro de $\epsilon$ del experto. Otra vez, el número de demostraciones depende de $k$, no de $|S|$ —por eso empíricamente bastan pocas. Si la recompensa verdadera no está exactamente en el span de las features, el desempeño se degrada suavemente, en $O(\|\varepsilon\|_\infty)$.

En el **gridworld** (128×128, 64 macroceldas como features) max-margin y proyección convergen en pocas decenas de iteraciones, y el algoritmo alcanza desempeño cercano al experto con **órdenes de magnitud menos demostraciones** que las líneas base de imitación directa —que además nunca alcanzan al experto porque su clase de políticas es demasiado pobre.

El **simulador de conducción en autopista 3D** es el ejemplo estrella. El auto propio va a 25 m/s (más rápido que el tráfico), con cinco acciones y 15 features (carril actual y distancia discretizada al auto más cercano). La feature expectation del experto se estimó de **una sola trayectoria de 1200 muestras (2 minutos de conducción)**. Los autores demostraron **cinco estilos** —*Nice* (evitar toda colisión, preferir carril derecho), *Nasty* (chocar cuantos autos se pueda), *Right lane nice*, *Right lane nasty* y *Middle lane*— y en **cada** caso el algoritmo imitó cualitativamente el estilo demostrado, con feature expectations aprendidas muy cercanas a las del experto. Aunque la teoría no garantiza nada sobre los pesos, estos "generalmente tienen sentido intuitivo" (recompensa negativa para colisiones y off-road, positiva para el carril preferido). Este es el experimento que la [Clase 33](/clases/clase-33) cita como "aprender a conducir usando IRL".

## Limitaciones

- **La recompensa no se identifica.** El paso IRL solo adivina *algún* $w$ que separa; los pesos hallados no son interpretables como "la" recompensa del experto (son intuitivos, no identificados).
- **Recompensa lineal.** Todo descansa en $R^* = w^* \cdot \phi$; extender a recompensas no lineales y a selección automática de features queda como problema abierto.
- **Resuelve muchos MDPs.** Cada iteración invoca un solver de RL completo sobre el `MDP\R` con la recompensa candidata, caro en dominios grandes; además supone conocida la dinámica de transiciones (o un simulador).
- **Selección final.** La versión limpia devuelve un *conjunto*; obtener una sola política exige inspección humana o un QP que produce una política *mixta* estocástica.

## Por qué importa para la Clase 33

La [Clase 33](/clases/clase-33) cubre el espectro del aprendizaje por imitación e IRL. Este paper permite contrastar tres enfoques: **behavioral cloning** (clona el mapeo estado → acción y se rompe fuera de la distribución de demostraciones), **DAgger** (corrige iterativamente pero requiere consultar al experto durante el entrenamiento) e **IRL / apprenticeship** (infiere la recompensa implícita vía matching de feature expectations, a costa de resolver muchos MDPs y suponer recompensa lineal).

La lección central es la distinción entre **imitar la política** e **inferir el objetivo**: el BC clona *qué hizo* el experto, el IRL infiere *por qué* lo hizo —una representación más transferible y robusta ante cambios de situación, exactamente el argumento del ejemplo de conducción donde el tráfico cambia en cada episodio. La contribución teórica duradera —**basta igualar el comportamiento agregado (feature expectations) para igualar el desempeño, sin recuperar la recompensa verdadera**— convierte al [aprendizaje reforzado inverso](/fundamentos/aprendizaje-reforzado-inverso) en herramienta práctica y no en curiosidad mal condicionada. De aquí salen el MaxEnt IRL (Ziebart, 2008), que resuelve la ambigüedad restante con máxima entropía, y GAIL (Ho & Ermon, 2016), que reemplaza el paso max-margin tipo SVM por un discriminador de red neuronal en esquema GAN.
