---
title: "GAIL: Generative Adversarial Imitation Learning (2016)"
weight: 373
math: true
---

{{< paper-card
    title="Generative Adversarial Imitation Learning"
    authors="Jonathan Ho, Stefano Ermon (Stanford)"
    year="2016"
    venue="NeurIPS 2016"
    pdf="/papers/gail-ho-ermon-2016.pdf" >}}
GAIL deriva un marco general que muestra que "ejecutar RL sobre la recompensa recuperada por [IRL](/fundamentos/aprendizaje-reforzado-inverso)" equivale a hacer *occupancy measure matching* —igualar la distribución de ocupación estado-acción entre aprendiz y experto— y que una instancia particular de ese marco produce un objetivo tipo **GAN**. De ahí obtiene un algoritmo *model-free* de imitación que supera con amplitud a los métodos previos en control continuo de alta dimensión. El discriminador cumple el papel de "recompensa aprendida" y la política el de "generador", evitando el bucle costoso del IRL clásico (resolver un problema de RL completo por cada actualización del costo). Es el puente moderno que unifica los tres hilos de la [Clase 33](/clases/clase-33): IRL, imitación y aprendizaje adversario de distribuciones.
{{< /paper-card >}}

---

## Contexto: las dos familias y sus límites

Hacia 2016, aprender una tarea a partir de demostraciones expertas —sin poder consultar al experto durante el entrenamiento y sin señal de recompensa— tenía dos caminos clásicos, ambos insatisfactorios. El **behavioral cloning** trata la imitación como aprendizaje supervisado sobre pares estado-acción; es engañosamente simple pero sufre de **error acumulado** (*compounding error*) por *covariate shift*: el clasificador se entrena bajo la distribución de estados del experto pero se ejecuta bajo la del aprendiz, y cualquier error de un paso lo lleva a estados cada vez más ajenos donde su predicción empeora. Esta es la patología que [DAgger](/papers/dagger-ross-2011) ataca permitiendo consultar al experto en los estados visitados —pero DAgger exige un experto interactivo consultable, supuesto que GAIL descarta explícitamente.

El **IRL clásico** toma el camino opuesto: aprende una función de costo que prioriza trayectorias enteras, de modo que el error acumulado deja de ser problema. El precio es computacional: muchos algoritmos de IRL "requieren aprendizaje por refuerzo en un bucle interno", resolviendo un RL completo por cada actualización del costo. Hay además una crítica casi filosófica que motiva el paper: si el objetivo real del aprendiz es *actuar* como el experto, ¿por qué pagar por aprender una función de costo que luego habrá que "resolver" con RL y que aun así no produce acciones directamente?

## Método: IRL ↔ occupancy matching ↔ GAN

El punto de partida es el IRL de **máxima entropía causal** (Ziebart et al.), con un **regularizador de costo convexo** $\psi$ que gobierna qué algoritmo de imitación se obtiene —elegir $\psi$ *es* elegir el algoritmo. La herramienta clave es la **medida de ocupación**:

$$\rho_\pi(s,a) = \pi(a \mid s) \sum_{t=0}^{\infty} \gamma^t\, P(s_t = s \mid \pi),$$

la distribución (descontada, no normalizada) de pares estado-acción que un agente encuentra al navegar con $\pi$. Linealiza los costos esperados, $\mathbb{E}_\pi[c(s,a)] = \sum_{s,a}\rho_\pi(s,a)\,c(s,a)$, y hay una **correspondencia biunívoca** entre políticas y medidas de ocupación. Con esta maquinaria el paper prueba su resultado bisagra (Proposición 3.2):

$$\mathrm{RL} \circ \mathrm{IRL}_\psi(\pi_E) = \arg\min_{\pi \in \Pi} -H(\pi) + \psi^*(\rho_\pi - \rho_{\pi_E}),$$

donde $\psi^*$ es la conjugada convexa de $\psi$. La lectura es profunda: **el IRL regularizado busca implícitamente una política cuya medida de ocupación esté cerca de la del experto.** IRL no es "realmente" sobre recuperar costos; es el **dual** de un problema de igualación de distribuciones de ocupación, y el costo recuperado es solo la variable dual óptima. En el caso sin regularización, $\rho_{\tilde\pi} = \rho_{\pi_E}$: la ocupación se iguala exactamente. Y ahí queda expuesto el costo del método clásico: el IRL clásico es un **ascenso dual** en el que se resuelve una y otra vez el primal —que *es* RL.

El matching exacto es inútil en la práctica (con muestras finitas la mayoría de $\rho_{\pi_E}$ son cero, forzando a la política a nunca visitar pares no vistos). GAIL relaja la igualdad a una penalización suave eligiendo un regularizador $\psi_{GA}$ que combina la expresividad del matching exacto con la tratabilidad de los indicadores lineales, y que **se adapta a las demostraciones** en vez de fijar un subespacio de costos. Usando la pérdida logística dentro de la correspondencia entre riesgos de clasificación binaria y $f$-divergencias, la conjugada resulta ser la **divergencia de Jensen-Shannon** entre las medidas de ocupación:

$$\min_\pi\; \psi_{GA}^*(\rho_\pi - \rho_{\pi_E}) - \lambda H(\pi) = D_{JS}(\rho_\pi, \rho_{\pi_E}) - \lambda H(\pi).$$

GAIL busca la política cuya ocupación **minimiza la divergencia de Jensen-Shannon** respecto a la del experto. Expandiendo la conjugada, el objetivo se vuelve un juego de silla idéntico en forma al de las [GAN](/papers/gan-goodfellow-2014):

$$\min_\pi \max_D\; \mathbb{E}_\pi[\log D(s,a)] + \mathbb{E}_{\pi_E}[\log(1 - D(s,a))] - \lambda H(\pi).$$

La ocupación del aprendiz $\rho_\pi$ hace de "distribución generada" y la del experto $\rho_{\pi_E}$ de "distribución verdadera"; el **discriminador $D$ es la recompensa aprendida** y la **política $\pi$ es el generador**. El algoritmo alterna un paso de gradiente **Adam** sobre el discriminador (aprende a distinguir aprendiz de experto) y un paso **TRPO** sobre la política usando $\log D(s,a)$ como función de costo local. La ganancia arquitectónica: GAIL **no resuelve un RL hasta convergencia por cada actualización del discriminador** —entrelaza un solo paso de cada coordenada del punto de silla, igual que se entrena una GAN—, heredando la corrección de IRL sin pagar el bucle interno de RL.

## Resultados

GAIL se evalúa en **9 tareas** de control físico, desde clásicas de baja dimensión (cartpole, acrobot, mountain car) hasta MuJoCo de alta dimensión (Reacher, HalfCheetah, Hopper, Walker, Ant y un **humanoide 3D** de 376 observaciones y 17 acciones continuas). Los baselines son behavioral cloning, feature expectation matching (FEM) y game-theoretic apprenticeship learning (GTAL), todos con la misma arquitectura y la misma cantidad de interacción con el entorno. GAIL **siempre produjo políticas mejores que las tres líneas base** en las tareas clásicas. La única excepción fue Reacher, donde el behavioral cloning fue más eficiente en muestras (y la regularización de entropía causal, de $\lambda=0$ a $\lambda=10^{-3}$, mejoró a GAIL de forma estadísticamente significativa, $p=0.05$). En el resto de MuJoCo la ventaja fue **grande**: GAIL alcanzó al menos el 70% del rendimiento experto para todos los tamaños de dataset, mientras FEM y GTAL rindieron peor que una política aleatoria en Ant y el behavioral cloning **no superó el 60% en el humanoide**, tarea donde GAIL alcanzó rendimiento experto exacto. El mensaje empírico: GAIL es muy eficiente en el *número de demostraciones expertas* que necesita.

## Limitaciones

- **Eficiencia en interacción con el entorno.** GAIL es eficiente en datos del experto pero "no es particularmente eficiente en muestras en cuanto a la interacción con el entorno": el número de muestras para estimar el gradiente fue comparable al que TRPO necesita para entrenar las políticas experto desde cero. Los autores sugieren inicializar con behavioral cloning para acelerar.
- **Carácter *model-free*:** necesita más interacción que los métodos basados en modelo (como *guided cost learning*).
- **Sin interacción con el experto:** igual que IRL, explora aleatoriamente para descubrir qué acciones acercan su ocupación a la del experto, en vez de preguntárselo como hace [DAgger](/papers/dagger-ross-2011).
- **Inestabilidad adversaria:** el juego min-max se resuelve por pasos alternados sin garantía de convergencia global, herencia directa de la dinámica de las GAN; el paso TRPO existe precisamente para amortiguar el ruido y evitar que la política diverja.

## Por qué importa para la Clase 33

La [Clase 33](/clases/clase-33) cubre exactamente las tres piezas que GAIL unifica: [IRL](/fundamentos/aprendizaje-reforzado-inverso) (recuperar la recompensa que hace óptimo al experto), la [imitación directa](/fundamentos/aprendizaje-por-imitacion) (behavioral cloning, con DAgger como corrección al error acumulado) y su combinación. GAIL demuestra formalmente que **IRL es el dual de un problema de igualación de ocupación**: "correr RL sobre la recompensa de IRL" e "imitar igualando la distribución de ocupación" son la misma operación vista desde el lado dual y el primal. Además tiende el puente hacia las [GAN](/papers/gan-goodfellow-2014) —del módulo de modelos generativos— al mostrar que la elección correcta del regularizador convierte la imitación en un juego adversario que minimiza la divergencia de Jensen-Shannon.

GAIL también es un **ancestro estructural del RLHF** que domina el alineamiento moderno de modelos de lenguaje: en ambos casos la "recompensa" no la da el entorno sino que se *aprende* de conducta humana (demostraciones aquí, preferencias en RLHF), y la política se optimiza contra esa recompensa cambiante con un método de región de confianza (TRPO, primo de PPO). Para el estudiante, GAIL es la pieza que ordena el mapa de la clase: IRL, imitación y GAN no son tres temas separados, sino tres caras de un mismo problema de *matching* de distribuciones.
