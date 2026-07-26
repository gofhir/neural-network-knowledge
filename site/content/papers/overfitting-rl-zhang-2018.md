---
title: "A Study on Overfitting in Deep RL (2018)"
weight: 376
math: true
---

{{< paper-card
    title="A Study on Overfitting in Deep Reinforcement Learning"
    authors="Chiyuan Zhang, Oriol Vinyals, Rémi Munos, Samy Bengio (Google Brain)"
    year="2018"
    venue="arXiv:1804.06893"
    pdf="/papers/overfitting-rl-zhang-2018.pdf" >}}
El paper hace una pregunta incómoda y la responde con un experimento limpio: cuando un agente de RL profundo alcanza recompensa óptima **durante el entrenamiento**, ¿eso dice algo sobre su desempeño en situaciones nuevas? La respuesta es **no**. La tesis central es que la práctica habitual del RL profundo —entrenar y evaluar en el **mismo** entorno, sin partición train/test— **oculta sistemáticamente el sobreajuste**. Para demostrarlo, los autores construyen un *gridworld* configurable generado por un procedimiento aleatorio (el generador $\mu$), separan un *pool* de configuraciones iniciales de entrenamiento de uno disjunto de prueba, y muestran que los agentes A3C **memorizan** grandes colecciones de laberintos —incluso laberintos con **recompensas completamente aleatorias**, donde por diseño no hay nada generalizable que aprender—. Es el experimento del gridworld de recompensas (diamante $+1$, rayo $-1$, bomba $-1$, llave $+0.1$) de las diapositivas 15-16 de la [Clase 33](/clases/clase-33), y el punto de partida de la [generalización en RL](/fundamentos/generalizacion-en-rl).
{{< /paper-card >}}

---

## Contexto: por qué el RL "no generaliza" por defecto

El origen del problema es la **formulación misma** del aprendizaje reforzado. El RL se plantea normalmente en un escenario de *continual learning*, sin etapas de entrenamiento y prueba explícitamente separadas: el objetivo es maximizar la recompensa acumulada a lo largo del tiempo. A diferencia del aprendizaje supervisado, donde el *train/test split* es dogma desde hace décadas, en RL —incluso en tareas tan populares como Atari— no había en 2018 protocolos experimentales estandarizados. La consecuencia es directa: **el desempeño sobre el conjunto de entrenamiento se reporta como si fuera el desempeño verdadero.** Si un agente juega bien el mismo Breakout con el que se entrenó, se declara que "aprendió a jugar Breakout", pero ese número no dice nada sobre datos no vistos.

El paper formaliza una tarea de RL como un par $(M, P_0)$, donde $M = (S, A, P, r)$ es un MDP y $P_0$ es una distribución sobre los estados iniciales. La propuesta —trivial en supervisado, ausente en RL— es partir una muestra i.i.d. $\hat{S}_0$ de $P_0$ en conjuntos disjuntos de entrenamiento y prueba, y medir la generalización como la diferencia:

$$
\text{Generalización} = \text{Recompensa}_{\text{test}} - \text{Recompensa}_{\text{train}}
$$

Para paliar la ausencia de *split*, la comunidad recurría a un parche: **inyectar estocasticidad** (política estocástica, *random starts*, *sticky actions*, *frame skipping*) para evitar que trucos degenerados dieran puntajes artificialmente altos. El motivo era concreto: en un entorno determinista, un algoritmo tan tonto como *Brute* puede optimizar una **secuencia de acciones de lazo abierto** memorizada sin mirar los estados —de hecho superó al mejor método de aprendizaje en **45 de 55** juegos de Atari—. La estocasticidad quiebra esos trucos, pero casi no existían estudios sobre si esas técnicas realmente previenen o detectan el sobreajuste en general.

## Método: el gridworld de la clase, los *splits* y el *test worker*

El entorno es un *gridworld* 2D con 5 objetos y un agente. Hay que recolectar los objetos de recompensa positiva evitando los negativos, y luego tomar el objeto terminante antes del *timeout*. Las recompensas —las que menciona textualmente la clase— son: los dos **diamantes** valen $+1$ cada uno; la **bomba** y el **rayo** valen $-1$ cada uno; y la **llave** es el objeto terminante con recompensa $+0.1$ que cierra el episodio de inmediato. Si el agente no toma la llave en 200 pasos, el episodio termina con penalización de *timeout* de $-1$. La aritmética es fija: un agente oráculo logra exactamente $2.1$ (dos diamantes $+2$ más la llave $+0.1$), techo que se mantiene constante en las tres variantes de dificultad —**BASIC** (sala $9\times9$), **BLOCKS** ($13\times13$ con 8 obstáculos) y **TUNNEL** ($13\times13$ con corredores)—. El generador $\mu$ garantiza que cada laberinto sea **resoluble**, y el algoritmo de aprendizaje es **A3C**.

La maquinaria clave es la noción de **niveles**: el *id* de nivel es una semilla del generador que determina el estado inicial (paredes, ubicaciones y recompensas de los objetos). Se reserva un *pool* finito de ids para entrenamiento y uno disjunto para prueba. La modificación decisiva a A3C es un ***test worker***: durante el entrenamiento jala continuamente los pesos más recientes para interactuar con los niveles de prueba aislados, pero **no computa ni envía gradientes**. Así se obtienen **curvas de aprendizaje y de prueba simultáneas**, como las que uno da por sentadas en supervisado. Finalmente, inspirados en el paper de los *random labels* (Zhang et al., 2017), los autores extienden el entorno para medir la memorización pura: con probabilidad $p$, la recompensa de cada objeto se **invierte de signo** (determinada por el *id* de nivel), solo en los niveles de entrenamiento. Cuando $p=0.5$, el buen desempeño de prueba es **imposible por diseño**, y un buen desempeño de entrenamiento es por definición sobreajuste serio.

## Resultados

- **Recompensa óptima de entrenamiento ≠ generalización.** Entrenando con 10, 100, 1.000 y 10.000 niveles, la recompensa de entrenamiento alcanza el óptimo en todos los casos, pero la de prueba varía enormemente: aumenta con más niveles y baja en laberintos más difíciles. Con solo 10 niveles el agente rinde mal en todos los entornos.
- **Los agentes memorizan ruido puro.** Con conjuntos pequeños, los agentes alcanzan desempeño casi óptimo de entrenamiento **incluso con $p=0.5$** —memorizan asignaciones de recompensa completamente aleatorias—. Incluso en TUNNEL con 10.000 niveles logran recompensas no triviales bajo ruido pesado, abriendo una enorme brecha train/test. Es el paralelo directo de los *random labels* en visión: la red tiene capacidad de sobra para memorizar.
- **La estocasticidad no salva ni detecta.** Estudiando política estocástica, *random starts* (RAND-SPAWN) y *sticky actions* ($\zeta=0.25$), como regularizadores apenas mejoran un poco la prueba; como detectores el fracaso es total: al evaluar sobre el conjunto de entrenamiento con estocasticidad añadida, los puntajes de un agente que generaliza fatal (10 niveles) **apenas se distinguen** de uno que generaliza bien (10.000 niveles). Un agente sobreajustado construye una **tabla de búsqueda "blanda"** robusta a pequeñas perturbaciones; solo un cambio grande de estado lo delata a veces, y cuando la memorización ocurre sobre muchos niveles, ni los *random starts* lo detectan.
- **Sesgo inductivo: cuándo sí generaliza.** En laberintos regulares el valor de un objeto es **espacialmente invariante** (una bomba vale $-1$ en cualquier posición), lo que casa con el sesgo de una **ConvNet**. Comparando ConvNets contra MLPs, los MLPs **ajustan mejor el entrenamiento**, pero las **ConvNets generalizan consistentemente mejor** cuando el juego es regular. La buena generalización requiere que el sesgo inductivo del modelo sea **compatible** con la regularidad del problema.

## Limitaciones

Es un estudio **empírico en un entorno de juguete**: un *gridworld* controlado, no un dominio real ni Atari completo; la extensión a control continuo o gran escala se argumenta por analogía. El análisis de sesgo inductivo es **cualitativo y abierto**: no ofrece una caracterización matemática de qué tareas producen modelos simples. Se estudia un solo algoritmo (A3C con ConvNet/MLP), sin métodos basados en valor ni políticas recurrentes. Y el aporte es **diagnóstico y metodológico** —dice cómo medir, no cómo curar—: su recomendación (aislar train/test) es necesaria pero no suficiente para garantizar generalización.

## Por qué importa para la Clase 33

En la [Clase 33](/clases/clase-33), este paper aparece justo donde se plantea la pregunta "¿el aprendizaje reforzado generaliza?". La respuesta es matizada y precisa: **el RL profundo no generaliza por defecto**, porque la práctica habitual entrena y evalúa en el mismo ambiente y así confunde memorización con aprendizaje. La demostración con recompensas aleatorias cierra la puerta a cualquier defensa: si una red memoriza ruido puro con recompensa óptima de entrenamiento, entonces la recompensa de entrenamiento **nunca fue evidencia de comprensión**.

Pero el trabajo también entrega la receta constructiva, que es la lección central de la clase sobre [generalización en RL](/fundamentos/generalizacion-en-rl): **sí se puede generalizar, pero hay que entrenar variando explícitamente las dimensiones a lo largo de las cuales se desea generalizar.** El agente entrenado con 10 niveles no generaliza; el de 10.000 sí —misma arquitectura, mismo algoritmo, distinta **diversidad** de configuraciones iniciales—. La generalización no es un regalo del modelo: se **construye** eligiendo qué variar en el generador $\mu$ y midiéndola con un *split* honesto. Es la premisa que [Cobbe et al. (CoinRun, 2019)](/papers/quantifying-generalization-cobbe-2019) convertiría en un *benchmark* explícito, mostrando que se necesitan **miles** de niveles para cerrar la brecha y que la regularización supervisada (dropout, L2, *data augmentation*, *batch norm*) también ayuda en RL.

Para un lector que construye sistemas de *record linkage* / MDM sobre datos FHIR, el paper describe una trampa cotidiana: evaluar un modelo de *patient matching* sobre la misma distribución —el mismo hospital, la misma fuente, el mismo período— en la que se entrenó equivale a reportar la recompensa de entrenamiento como si fuera generalización. Un *scorer* con F1 altísimo sobre pares del hospital A puede haber memorizado sus convenciones idiosincráticas (formatos de RUT, patrones de digitación, apodos locales) sin aprender a resolver identidad de pacientes en general —el análogo del agente que memoriza 10 laberintos—. La corrección que exige el paper se traduce en la necesidad de **splits por sitio, por fuente y por período**: entrenar en los hospitales A y B y medir en el C es la única forma de saber si el modelo generaliza a la fuente nueva a la que se desplegará.
