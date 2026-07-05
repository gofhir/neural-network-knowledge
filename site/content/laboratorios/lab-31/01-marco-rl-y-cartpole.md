---
title: "El marco RL y el ambiente CartPole"
weight: 1
---

Antes de tocar la red de DQN, el tutorial planta las dos bases: el **formalismo de aprendizaje reforzado** (qué optimiza un agente y por qué) y el **ambiente CartPole** (dónde lo demuestra). Esta página recorre esas celdas con el detalle matemático que las hace precisas.

## El bucle agente ↔ ambiente

Todo RL vive en un mismo bucle. En cada paso de tiempo $t$:

1. El ambiente presenta un **estado** $s_t$.
2. El agente elige una **acción** $a_t$ según su **política** $\pi(a\mid s)$ (una distribución sobre acciones dado el estado).
3. El ambiente responde con el **siguiente estado** $s_{t+1}$ y una **recompensa escalar** $r_{t+1}$.

Se formaliza como un **Proceso de Decisión de Markov (MDP)** $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma\rangle$. La palabra clave es **Markov**: $s_t$ debe contener *toda* la información necesaria para predecir el futuro — dado el presente, el pasado no aporta nada. Esta propiedad es exactamente el corazón de una de las preguntas de la [tarea](04-actividades) (por qué el estado necesita las velocidades).

## El objetivo: retorno esperado descontado

El agente no maximiza la recompensa inmediata, sino el **retorno** — la suma descontada de recompensas futuras. La política óptima es:

$$
\pi^* = \arg\max_{\pi \in \Pi}\ \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty}\gamma^{t}R_{t+1}\right]
$$

**Por qué el descuento $\gamma \in [0,1)$** (en el lab $\gamma = 0.95$):

- **Convergencia.** Sin él, en un episodio infinito la suma $\sum R_{t+1}$ diverge. Con $\gamma < 1$ y recompensas acotadas por $R_{\max}$, el retorno queda acotado por la serie geométrica $\sum_t \gamma^t R_{\max} = \frac{R_{\max}}{1-\gamma}$. Con $R_{\max}=1$ y $\gamma=0.95$ eso da un techo de $20$.
- **Preferencia temporal.** Valora más lo cercano. Un $\gamma=0.95$ implica un **horizonte efectivo** de $\approx \frac{1}{1-\gamma}=20$ pasos: el agente "ve" unos 20 pasos hacia adelante.

## Q-values óptimos: el objeto que DQN estima

En vez de aprender $\pi^*$ directamente, los métodos de Q-learning aprenden la **función de valor de acción óptima**:

$$
q_*(s,a) = \max_{\pi}\ \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty}\gamma^{t}R_{t+1}\,\middle|\, S_0=s,\, A_0=a\right]
$$

Interpretación operativa: *"si ejecuto $a$ ahora en $s$ y de ahí en adelante juego óptimo, ¿cuánta recompensa acumulo?"*. La ganancia de conocer $q_*$ es que **la política óptima se lee sin planificar**, con un simple $\arg\max$:

$$
\pi^*(a\mid s) = \arg\max_{a\in\mathcal{A}}\ q_*(s,a)
$$

Esto convierte un problema de control (buscar sobre el espacio de políticas $\Pi$, gigantesco) en un problema de **regresión**: estimar una función $q_*: \mathcal{S}\times\mathcal{A}\to\mathbb{R}$. Ese pivote es lo que hace posible usar una red neuronal.

## La ecuación de Bellman

Cuando el ambiente es un MDP, los Q-values óptimos cumplen la **ecuación de optimalidad de Bellman** — el corazón de todo DQN:

$$
q_*(s,a) = \mathbb{E}_{s'\sim P}\!\left[\, r + \gamma \max_{a'} q_*(s',a')\ \middle|\ s,a\right]
$$

Es una definición **recursiva y de punto fijo**: $q_*$ es el único punto fijo del operador de Bellman $\mathcal{T}$, que es una **contracción** de factor $\gamma$ en norma sup — por el teorema de Banach, iterar $\mathcal{T}$ converge a $q_*$ desde cualquier inicialización. En DQN, la red aproxima $q_*$ y el lado derecho $r + \gamma\max_{a'}Q(s',a')$ se vuelve el **target** de entrenamiento (lo veremos literal en la [implementación](02-dqn-implementacion)).

{{< callout type="warning" >}}
**Gotcha profundo.** Al usar una red no lineal como aproximador se pierde la garantía teórica de contracción (el operador "Bellman + proyección sobre la red" ya no es contracción). Por eso DQN necesita dos estabilizadores — *experience replay* y *target network* — que aparecen en [Mnih et al. 2015](/papers/dqn-nature-mnih-2015). Es la **tríada mortal**: aproximación de funciones + bootstrapping + off-policy.
{{< /callout >}}

## OpenAI Gym / Gymnasium

Gym (hoy **Gymnasium**, el fork mantenido) define un **contrato uniforme** para ambientes de RL. Toda la interacción se reduce a cinco métodos:

```python
env = gym.make('CartPole-v1')          # instanciar
obs, info = env.reset(seed=42)          # estado inicial
obs, reward, terminated, truncated, info = env.step(action)   # transición
env.render()                            # visualizar
env.close()                             # liberar
```

{{< callout type="warning" >}}
**Gotcha de versión (crítico para que el notebook corra hoy).** La API cambió entre `gym` clásico y `gymnasium`. El `step()` viejo devolvía **4** valores `(obs, reward, done, info)`; el nuevo devuelve **5**, partiendo `done` en dos señales distintas:

- **`terminated`**: el episodio terminó por la **dinámica del MDP** (el bastón se cayó). Es un estado terminal genuino → en el target de Bellman, aquí $q_* = r$ (sin bootstrap del futuro).
- **`truncated`**: el episodio se cortó por un **límite artificial** (timeout de 500 pasos en `CartPole-v1`), no porque el MDP terminara. El estado **no** es terminal.

El notebook colapsa ambas con `done = terminated or truncated`. Es un shortcut pedagógico: cuando `truncated=True` trata el estado como terminal (target $=r$), lo que introduce un pequeño sesgo de bootstrapping. En CartPole no muerde porque el propio código pone su tope en 210 < 500 pasos, así que `truncated` casi nunca se dispara por el límite de Gym.
{{< /callout >}}

## El ambiente CartPole

El clásico de control desde Barto, Sutton & Anderson (1983). Un bastón (péndulo invertido) montado sobre un carro que se mueve en 1D. Física inherentemente **inestable**: el equilibrio vertical es un punto de equilibrio inestable, cualquier desviación se amplifica.

**Espacio de estados** — vector $s \in \mathbb{R}^4$:

| Índice | Variable | Símbolo | Rango |
|--------|----------|---------|-------|
| 0 | Posición del carro | $x$ | $[-4.8,\ 4.8]$ |
| 1 | Velocidad del carro | $\dot{x}$ | $(-\infty,\ \infty)$ |
| 2 | Ángulo del bastón | $\theta$ | $[-0.418,\ 0.418]$ rad ($\pm 24°$) |
| 3 | Velocidad angular | $\dot{\theta}$ | $(-\infty,\ \infty)$ |

**Espacio de acciones** — discreto, $|\mathcal{A}|=2$: empujar izquierda (0) o derecha (1). No hay "no hacer nada": el carro *siempre* recibe fuerza, lo que hace la tarea más difícil.

**Recompensa**: $+1$ por cada paso vivo. **Termina** cuando $|\theta| > 12°$ o $|x| > 2.4$. Maximizar recompensa ≡ sobrevivir más pasos.

## El baseline aleatorio

Una política uniforme $\pi(a\mid s) = 0.5$ que ignora el estado, medida en esta ejecución, obtiene:

```
Episodios: 37, 22, 12, 16, 12, 14, 10, 39, 21, 27  →  media 21.0
```

**21 pasos antes de caer.** Este es el piso contra el cual DQN mostrará su valor (**210**, exactamente **10×**). El baseline también revela algo del MDP: aún actuando al azar sobrevives ~21 pasos, porque desde el estado inicial (casi vertical, velocidades ~0) toma varias sacudidas acumular suficiente ángulo para caer.

---

**Siguiente:** [Implementación de DQN](02-dqn-implementacion) — la clase `DeepQNetwork`, el replay buffer y el loop de entrenamiento, contrastados con datos reales.
