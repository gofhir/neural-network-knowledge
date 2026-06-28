---
title: "Prioritized Experience Replay (2015)"
weight: 352
math: true
---

{{< paper-card
    title="Prioritized Experience Replay"
    authors="Tom Schaul, John Quan, Ioannis Antonoglou, David Silver"
    year="2015"
    venue="ICLR 2016"
    pdf="/papers/per-schaul-2015.pdf"
    arxiv="1511.05952" >}}
Paper de Google DeepMind que mejora el **experience replay** del DQN sin tocar la red ni el algoritmo: en lugar de muestrear las transiciones del buffer **uniformemente al azar**, las muestrea con probabilidad creciente en su **error de diferencia temporal (TD-error)** $|\delta|$, de modo que las transiciones "sorprendentes" se repiten más. El muestreo no uniforme introduce un sesgo, que se corrige con **pesos de importance sampling**, y se implementa eficientemente con un **sum-tree** ($O(\log N)$). En Atari 2600 acelera el aprendizaje **cerca de 2×** y supera al DQN uniforme en **41 de 49 juegos**. Es uno de los seis componentes de **Rainbow**.
{{< /paper-card >}}

---

## Contexto: el desperdicio del replay uniforme

El RL *online* en su forma más simple descarta cada experiencia tras una sola actualización, lo que deja las actualizaciones fuertemente correlacionadas en el tiempo (rompiendo el supuesto i.i.d. del gradiente estocástico) y olvida rápido las experiencias raras. El *experience replay* (Lin, 1992) resuelve ambos problemas: guarda las transiciones en una memoria y las mezcla, rompiendo las correlaciones temporales y reusando cada experiencia varias veces. El DQN ([Mnih et al. 2015](/papers/dqn-nature-mnih-2015)) lo hizo central, con una memoria deslizante de $10^6$ transiciones de la cual sampleaba **uniformemente**, revisitando cada transición unas ocho veces.

El límite de ese diseño es el punto de partida del paper: el replay uniforme "repite las transiciones a la misma frecuencia con que se experimentaron, sin importar su relevancia". Pero no todas enseñan lo mismo. Algunas son sorprendentes, raras o cargadas de información sobre el error del modelo; otras son redundantes y ya están bien predichas. Replay uniforme las trata a todas igual, desperdiciando cómputo en lo que el agente ya domina.

El paper apela a dos motivaciones. **Neurociencia:** en el hipocampo de roedores se observa *replay* de experiencias, y las secuencias de alto TD-error y asociadas a recompensa se reproducen más a menudo. **Planificación:** el *prioritized sweeping* (Moore & Atkeson, 1993) ya ordena las actualizaciones de *value iteration* por su impacto esperado usando el TD-error como prioridad. PER traslada esa idea al RL *model-free* con aproximador de función.

### El ejemplo motivador: Blind Cliffwalk

Para cuantificar la ganancia posible, el paper introduce un entorno artificial, el **Blind Cliffwalk**: $n$ estados, dos acciones (una "correcta" que avanza, una "incorrecta" que termina el episodio) y una única recompensa al final de la secuencia correcta. La probabilidad de que una secuencia aleatoria llegue a la recompensa es $2^{-n}$: las transiciones útiles quedan ocultas en una masa de fracasos redundantes. Comparando dos agentes sobre la *misma* memoria —uno con replay uniforme, otro con un *oráculo* que elige la transición que más reduce la pérdida global— el oráculo logra un *speed-up exponencial*. El oráculo no es realizable, pero la brecha justifica buscar una aproximación práctica: priorizar por TD-error.

## Contribución central: priorizar por TD-error

El criterio ideal sería cuánto puede aprender el agente de una transición (el *progreso de aprendizaje esperado*), pero eso no es accesible directamente. Un *proxy* razonable es la **magnitud del TD-error** $\delta$: indica cuán "sorprendente" es la transición, qué tan lejos está el valor estimado $Q$ de su estimación bootstrap. Esto encaja de forma natural con Q-learning o SARSA, que ya computan $\delta$.

La versión más simple —**priorización greedy**— replaya siempre la transición de mayor $|\delta|$, pero tiene tres patologías: (1) **estancamiento** —como los TD-errors solo se actualizan al replayar, una transición de bajo error inicial puede no volver a verse nunca—; (2) **sensibilidad al ruido** —el sistema persigue picos de error por recompensas estocásticas—; y (3) **pérdida de diversidad** —las transiciones de alto error se replayan una y otra vez, propiciando *overfitting*—.

## Método: priorización estocástica, $\alpha$, $\beta$ e importance sampling

### Priorización estocástica

La solución interpola entre la priorización greedy pura y el muestreo uniforme: la probabilidad de muestrear una transición es monótona en su prioridad, pero no nula ni siquiera para la de menor prioridad. La probabilidad de muestrear la transición $i$ es:

$$P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}$$

donde el exponente $\alpha$ controla **cuánta priorización se aplica**: $\alpha = 0$ recupera el caso uniforme, $\alpha = 1$ es priorización plena. El paper define dos formas de fijar $p_i$:

- **Proporcional:** $p_i = |\delta_i| + \epsilon$, donde $\epsilon$ es una constante positiva pequeña que evita que una transición de error cero nunca vuelva a visitarse.
- **Rank-based:** $p_i = 1/\text{rank}(i)$, según la posición de la transición al ordenar el buffer por $|\delta_i|$. Es más robusta porque solo importa el orden, no la magnitud, lo que la vuelve insensible a *outliers*.

Ambas dan resultados similares en Atari, probablemente porque el DQN ya aplica *clipping* de recompensas y TD-errors, que elimina los outliers.

### Corrección del sesgo con importance sampling

Estimar la esperanza con actualizaciones estocásticas exige que provengan de la **misma distribución** que la esperanza. PER cambia esa distribución de forma incontrolada e introduce un **sesgo**: cambia la solución a la que convergen las estimaciones. La corrección son **pesos de importance sampling (IS)**:

$$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^{\beta}$$

que compensan completamente las probabilidades no uniformes cuando $\beta = 1$. Se incorporan al update usando $w_i \delta_i$ en lugar de $\delta_i$, y se normalizan por $1/\max_i w_i$, de modo que solo escalan la actualización **hacia abajo** (por estabilidad). El exponente $\beta$ se **anela** (annealing) linealmente desde $\beta_0$ hasta $1$: la corrección plena solo importa cerca de la convergencia, mientras que al inicio el proceso ya es altamente no estacionario y un pequeño sesgo temprano es tolerable. Un beneficio extra con aproximación no lineal: el peso IS reduce la magnitud del gradiente, permitiendo seguir paisajes de optimización muy curvados sin pasos disruptivos.

### El algoritmo y el sum-tree

PER se integra sobre el **Double DQN**: la única modificación es reemplazar el muestreo uniforme por la priorización estocástica más la corrección IS. Cada nueva transición entra con prioridad máxima (para garantizar que se vea al menos una vez); cada $K$ pasos se muestrea un minibatch según $P(j)$, se computan los $w_j$, el TD-error $\delta_j$ (con objetivo Double DQN), se refresca $p_j \leftarrow |\delta_j|$ y se acumula $\Delta \leftarrow \Delta + w_j \cdot \delta_j \cdot \nabla_\theta Q$.

Con $N = 10^6$, muestrear **no puede depender de $N$**. La variante proporcional usa un **sum-tree**: un árbol binario donde cada nodo interno es la *suma* de sus hijos y las hojas guardan las prioridades; la raíz contiene $p_{\text{total}}$. Para muestrear un minibatch de tamaño $k$ se divide $[0, p_{\text{total}}]$ en $k$ rangos iguales, se muestrea un valor uniforme de cada rango y se recupera la hoja recorriendo el árbol. Muestrear y actualizar cuestan ambos $O(\log N)$. La variante rank-based usa un *binary heap* con muestreo estratificado. El overhead total fue de solo **2 %–4 %** en tiempo de ejecución.

## Experimentos: Atari 2600

Las baselines son DQN (Nature) y Double DQN afinado, ambas con replay uniforme; PER mantiene **idéntica** arquitectura, algoritmo, tamaño de memoria y protocolo de evaluación —la única diferencia es el muestreo—. Solo se ajustó un hiperparámetro: como PER elige transiciones de alto error más a menudo, las magnitudes de gradiente son mayores, así que se **redujo el step-size por un factor 4**. Los valores dulces fueron $\alpha = 0.7,\ \beta_0 = 0.5$ (rank-based) y $\alpha = 0.6,\ \beta_0 = 0.4$ (proporcional).

| Comparación | Métrica | Baseline | PER |
|---|---|---|---|
| PER sobre DQN | mediana score normalizado | 48 % | **106 %** |
| PER sobre DQN | juegos mejorados | — | **41 de 49** |
| PER sobre Double DQN | mediana (57 juegos) | 111 % | **128 %** |
| PER sobre Double DQN | media (57 juegos) | 418 % | **551 %** |

Lecturas principales: (1) la ganancia de PER es **complementaria** a la de Double Q-learning, no competidora; (2) en agregado el aprendizaje es **el doble de rápido** —los puntos de equivalencia se alcanzan al 38 %–47 % del tiempo total—; (3) PER reduce el retraso inicial en juegos que tardan en "despegar". Dos hallazgos secundarios: el muestreo uniforme está **implícitamente sesgado hacia transiciones desactualizadas** (de políticas viejas), y PER lo corrige al premiar las recientes y no vistas; y la distribución empírica de TD-errors se vuelve *heavy-tailed* a medida que avanza el aprendizaje.

## Limitaciones reconocidas

- **El TD-error es solo un *proxy*:** ignora la estocasticidad de recompensas y transiciones, la observabilidad parcial y los límites del aproximador. Problemático con **transiciones no aprendibles** (ruido irreducible), que PER perseguiría indefinidamente.
- **Solo aborda qué replayar,** no qué guardar ni cuándo borrar; la gestión de memoria queda como extensión.
- **Errores desactualizados:** como las prioridades solo se refrescan al replayar, el $|\delta|$ guardado puede quedar *stale* respecto al modelo actual.
- **Hiperparámetros nuevos:** $\alpha$, $\beta_0$ y su schedule de annealing, más la necesidad de reducir el step-size.

## Por qué importa para la Clase 31

La [Clase 31](/clases/clase-31) recorre la secuencia de mejoras del DQN, y PER ocupa un lugar preciso en ella:

- **Mejora directa al *experience replay*.** El [DQN](/papers/dqn-nature-mnih-2015) estabiliza el entrenamiento con dos ingredientes —*experience replay* y *target network*—. PER es la mejora más natural sobre el primero: en vez de samplear uniformemente, samplear por TD-error.
- **Encaja en el patrón "mejoras quirúrgicas y combinables".** Donde el Double DQN ataca el sesgo de sobreestimación del `max` y el [Dueling DQN](/papers/dueling-dqn-wang-2015) reorganiza la arquitectura en $V + A$, PER ataca la *eficiencia de uso de datos* ya guardados. Su ganancia es complementaria a la de Double DQN; esa composabilidad culmina en Rainbow (Hessel et al., 2018), donde las ablaciones muestran que **PER es uno de los componentes cuya remoción más degrada el rendimiento**.
- **Lo que el laboratorio implementa.** PER es un cambio acotado sobre el buffer: reemplazar la lista circular de muestreo uniforme por un *sum-tree*, almacenar $p_i = |\delta_i| + \epsilon$, muestrear $P(i) \propto p_i^\alpha$ y aplicar el peso $w_i$ normalizado a la pérdida. El estudiante ve de primera mano por qué el muestreo no uniforme **necesita** la corrección IS —sin ella, el agente converge a una solución sesgada—.
- **Fundamento transversal.** El TD-error como señal de "progreso de aprendizaje", la ecuación de Bellman que lo define y el dilema exploración/explotación forman parte del cuerpo conceptual de [Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado).

## Notas y enlaces

- arXiv: https://arxiv.org/abs/1511.05952 (v1 del 18 nov 2015; v4 del 25 feb 2016).
- Venue: International Conference on Learning Representations (ICLR 2016), conference paper.
- Afiliación: Google DeepMind.
- Recursos del curso: [Clase 31](/clases/clase-31) · [Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado) · [DQN Nature (Mnih 2015)](/papers/dqn-nature-mnih-2015) · [Dueling DQN (Wang 2015)](/papers/dueling-dqn-wang-2015).
