---
title: "DQN desde cero"
weight: 2
math: true
---

En la [teoría de la clase 31](/clases/clase-31) vimos cómo Q-Learning aprende una tabla $Q(s,a)$ que estima el retorno esperado de tomar la acción $a$ en el estado $s$ y luego actuar óptimamente. Funciona perfecto cuando el espacio de estados es pequeño y discreto — una grilla con galleta —, porque cabe en una tabla. Pero el mundo real no cabe en tablas: la pantalla de un Atari tiene $256^{84\times84\times4}$ estados posibles, y un estado continuo como la posición de un péndulo es directamente *infinito*. La idea que destrabó el RL profundo, y que Mnih et al. publicaron en *Nature* en 2015, es reemplazar la tabla por una **red neuronal** $Q(s,a;\theta)$ que aproxima la función de valor. Eso es **Deep Q-Network (DQN)**.

Cambiar la tabla por una red suena trivial — "es solo regresión sobre el target de Bellman" —, pero en la práctica el entrenamiento *diverge* casi siempre si uno lo hace ingenuamente. La contribución real de DQN no es la red: son los **dos trucos de estabilización** que hacen que el entrenamiento converja. **Experience replay** rompe la correlación temporal entre muestras consecutivas, y una **target network** congelada rompe el lazo de retroalimentación entre el predictor y su propio objetivo. Vamos a implementar DQN desde cero en los tres frameworks sobre el entorno de control clásico más simple — CartPole — y a entender por qué esos dos trucos no son opcionales sino *cruciales*. La matemática de fondo (MDP, ecuación de Bellman, Q-Learning tabular) está en el [fundamento de aprendizaje reforzado](/fundamentos/aprendizaje-reforzado); el paper original, en el [análisis de DQN Nature 2015](/papers/dqn-nature-mnih-2015).

---

## 1. El entorno: CartPole en una imagen

CartPole es el "hola mundo" del control por RL. Un carro se desliza sobre un riel horizontal y sostiene un poste articulado por su base. El poste tiende a caer; el agente debe empujar el carro a izquierda o derecha para mantenerlo erguido el mayor tiempo posible.

| Componente | Valor | Detalle |
|---|---|---|
| Estado $s$ | vector de **4 dimensiones** | $[x,\ \dot{x},\ \vartheta,\ \dot{\vartheta}]$: posición del carro, velocidad del carro, ángulo del poste, velocidad angular del poste |
| Acciones $a$ | **2 discretas** | $0$ = empujar a la izquierda, $1$ = empujar a la derecha |
| Recompensa $r$ | **+1 por cada paso** | mientras el poste siga arriba y el carro dentro del riel |
| Terminación | $\lvert\vartheta\rvert > 12°$, $\lvert x\rvert > 2.4$, o 500 pasos | el episodio acaba al caer el poste o salirse del riel |
| Retorno máximo | **500** (en `CartPole-v1`) | mantener el equilibrio los 500 pasos |

La recompensa es engañosamente simple: $+1$ por paso. No hay ninguna señal que diga "ese empujón fue bueno". El agente solo sabe que **sobrevivir más pasos = más recompensa total**, y debe inferir, de esa señal global, qué acciones mantienen el equilibrio. Ese es el problema de **asignación de crédito** que hace difícil al RL: la acción que tomé hace 30 pasos puede ser la culpable de que el poste se caiga ahora.

Usamos `gymnasium` solo para el *entorno* (la física del carro y el poste). El DQN completo — red, replay, target network, ε-greedy, pérdida — lo escribimos a mano.

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
state, info = env.reset(seed=0)
print(state.shape)        # (4,)  -> [x, x_dot, theta, theta_dot]
print(env.action_space.n) # 2     -> {0: izquierda, 1: derecha}

# Un paso del entorno: el agente da una acción, el entorno responde.
next_state, reward, terminated, truncated, info = env.step(action=1)
# reward == 1.0 mientras el poste no caiga; terminated=True al fallar.
done = terminated or truncated
```

{{< concept-alert type="clave" >}}
En CartPole la recompensa no distingue "buena" de "mala" acción: siempre es $+1$ mientras no caigas. Todo el aprendizaje proviene de las **terminaciones**: cuando el poste cae, el flujo de $+1$ se corta, y el target de Bellman propaga ese "fin de las recompensas" hacia atrás en el tiempo. Por eso el agente aprende a *evitar* los estados que preceden a una caída, sin que nadie se los marque explícitamente.
{{< /concept-alert >}}

---

## 2. Aproximar $Q(s,a;\theta)$ con una MLP

En Q-Learning tabular, $Q$ es una tabla: una celda por cada par $(s,a)$. Con estados continuos eso es imposible, así que aproximamos $Q$ con una red neuronal de parámetros $\theta$. La arquitectura idiomática para estados vectoriales (no imágenes) es una **MLP** pequeña:

$$
s \in \mathbb{R}^4 \;\longrightarrow\; \text{Linear}(4,128)\to\text{ReLU}\to\text{Linear}(128,128)\to\text{ReLU}\to\text{Linear}(128,2)
$$

Una decisión de diseño importante: la red **no** recibe la acción como entrada. En su lugar, emite un **vector de $Q$-valores, uno por acción** — aquí 2 salidas. Es decir, $Q(s,\cdot;\theta): \mathbb{R}^4 \to \mathbb{R}^2$. Esto es mucho más eficiente que la alternativa $Q(s,a)$ con la acción concatenada, porque obtenemos los valores de *todas* las acciones en un solo forward pass — lo cual necesitamos tanto para elegir $\arg\max_a Q(s,a)$ como para calcular $\max_{a'} Q(s',a')$ del target.

```text
            ┌──────────────────────────────┐
   s (4)──▶ │  Linear 4→128  ReLU           │
            │  Linear 128→128  ReLU         │──▶  Q(s, ·)  =  [ Q(s,izq), Q(s,der) ]  ∈ ℝ²
            │  Linear 128→2                 │
            └──────────────────────────────┘
                              índice argmax = acción greedy
```

La función objetivo que queremos aproximar es el **target de Bellman**. Para una transición $(s,a,r,s',\text{done})$, el valor verdadero de $Q(s,a)$ debería satisfacer:

$$
Q(s,a) \;=\; r + \gamma \max_{a'} Q(s',a') \cdot (1 - \text{done})
$$

donde $\gamma \in [0,1)$ es el factor de descuento (usamos $0.99$) y el factor $(1-\text{done})$ anula el bootstrap cuando $s'$ es terminal — no hay futuro después de caer. La pérdida es el error cuadrático entre la predicción de la red y ese target:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\big(\,\underbrace{r + \gamma \max_{a'} Q(s',a';\theta^{-})}_{\text{target (red congelada }\theta^-)} - \underbrace{Q(s,a;\theta)}_{\text{predicción}}\,\big)^2\Big]
$$

Los dos detalles cruciales — que $\theta^{-}$ sea una red *distinta* y congelada, y que las muestras vengan de un *buffer* $\mathcal{D}$ y no de la trayectoria reciente — son las secciones 4 y 5. Primero, la política de exploración.

---

## 3. ε-greedy con decay

Si el agente siempre eligiera $\arg\max_a Q(s,a)$ (la acción que *cree* mejor), nunca probaría alternativas y podría quedar atrapado en una estrategia mediocre — el clásico dilema **exploración vs explotación**. La solución estándar es **ε-greedy**: con probabilidad $\varepsilon$ se toma una acción **aleatoria** (explorar), y con probabilidad $1-\varepsilon$ la acción greedy (explotar).

Al principio del entrenamiento la red no sabe nada, así que conviene $\varepsilon \approx 1$ (casi puro azar). A medida que aprende, bajamos $\varepsilon$ para explotar lo aprendido. Usamos un **decay exponencial** desde $\varepsilon_{\text{inicio}}=1.0$ hasta un piso $\varepsilon_{\text{fin}}=0.05$:

$$
\varepsilon_t = \varepsilon_{\text{fin}} + (\varepsilon_{\text{inicio}} - \varepsilon_{\text{fin}})\, e^{-t/\tau}
$$

donde $t$ es el número de pasos vividos y $\tau$ controla la velocidad de decay. El piso $\varepsilon_{\text{fin}}>0$ es deliberado: nunca dejamos de explorar del todo, porque el entorno puede tener regiones que la red aún no visitó.

---

## 4. Experience replay: romper la correlación temporal

Aquí está el primer truco crucial. Si entrenáramos la red con las transiciones *en el orden en que ocurren*, los minibatches estarían formados por estados consecutivos de una misma trayectoria — altamente correlacionados ($s_t$ y $s_{t+1}$ se parecen muchísimo). El descenso de gradiente estocástico **asume muestras i.i.d.**; alimentarlo con datos correlacionados produce gradientes sesgados, oscilaciones y olvido catastrófico de lo aprendido en estados que dejaron de visitarse.

**Experience replay** lo resuelve con un buffer circular $\mathcal{D}$ que guarda las últimas $N$ transiciones $(s,a,r,s',\text{done})$. En cada paso de entrenamiento **sampleamos un minibatch aleatorio** del buffer en vez de usar la transición recién vivida:

```text
  juego:  ...─▶ (s,a,r,s')  ─push─▶  ┌──────────────────────────────┐
                                     │  Replay Buffer (deque, N=10k) │
                                     │  [..., t-3, t-2, t-1, t]      │
                                     └──────────────────────────────┘
  train:  sample minibatch aleatorio de 64 ◀──────────┘  (rompe correlación temporal)
```

Dos beneficios: (1) los minibatches mezclan transiciones de momentos muy distintos, **descorrelacionándolas** y acercándolas a i.i.d.; (2) cada transición se **reutiliza** en muchos updates (eficiencia de datos), no se descarta tras un solo gradiente.

```python
from collections import deque
import random

class ReplayBuffer:
    """Buffer circular de transiciones. deque con maxlen descarta las más viejas."""
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)   # muestreo SIN reemplazo
        s, a, r, s_next, done = zip(*batch)               # desempaqueta a 5 tuplas
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buffer)
```

{{< concept-alert type="recordar" >}}
El `deque(maxlen=N)` implementa el buffer circular gratis: al superar la capacidad, descarta automáticamente la transición más antigua. Empezamos a entrenar solo cuando el buffer tiene suficientes muestras (p. ej. al menos un batch, o un mínimo de calentamiento como 1000), para que el primer minibatch ya sea variado.
{{< /concept-alert >}}

---

## 5. Target network: romper el lazo de retroalimentación

El segundo truco crucial. Mirá de nuevo el target de Bellman:

$$
y = r + \gamma \max_{a'} Q(s',a';\theta)
$$

Si usáramos los **mismos** parámetros $\theta$ para la predicción $Q(s,a;\theta)$ *y* para el target $y$, estaríamos persiguiendo un objetivo móvil: cada update de $\theta$ cambia la predicción **y** el target simultáneamente. Es como intentar acercarte a tu propia sombra — el suelo se mueve bajo tus pies. Esto genera oscilaciones y divergencia.

La solución es mantener una **red target** con parámetros $\theta^{-}$ que son una **copia congelada** de $\theta$. El target se calcula con $\theta^{-}$ (fijo), de modo que durante muchos pasos el objetivo es **estable**:

$$
y = r + \gamma \max_{a'} Q(s',a';\theta^{-}), \qquad \theta^{-} \leftarrow \theta \;\text{ cada } C \text{ pasos}
$$

Cada $C$ pasos (p. ej. $C=500$) **sincronizamos** $\theta^{-} \leftarrow \theta$, copiando los pesos aprendidos a la red congelada. Entre sincronizaciones, la red online persigue un blanco fijo — regresión estable —, y al sincronizar el blanco da un salto y vuelve a quedar fijo. Es el equivalente RL de separar "el examen" de "el libro de respuestas".

{{< concept-alert type="advertencia" >}}
El gradiente **nunca** debe fluir hacia la red target. Al calcular $y$ con $\theta^{-}$ hay que cortar el grafo: `.detach()` en PyTorch, `tf.stop_gradient` en TensorFlow, `jax.lax.stop_gradient` en JAX. Si te olvidas, el backward propaga al target y reintroduces exactamente el lazo de retroalimentación que la red congelada vino a romper.
{{< /concept-alert >}}

---

## 6. Implementación PyTorch

Juntamos las cuatro piezas: `QNetwork` (la MLP), `ReplayBuffer` (sección 4), `select_action` (ε-greedy), `optimize_model` (samplear, target con red congelada + `detach`, Smooth L1, backprop), sincronización cada $C$ pasos, y el loop sobre episodios.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque

# ---------- Red Q: MLP que mapea estado (4) -> Q-valores (2) ----------
class QNetwork(nn.Module):
    def __init__(self, n_obs=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),     # una salida por acción
        )

    def forward(self, x):
        return self.net(x)                    # (B, 4) -> (B, 2)

# ---------- ε-greedy ----------
def select_action(state, q_net, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)            # explorar
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)  # (1,4)
        q = q_net(s)                                  # (1,2)
        return int(q.argmax(dim=1).item())            # explotar: argmax_a Q(s,a)

# ---------- Un paso de optimización del DQN ----------
def optimize_model(q_net, target_net, buffer, optimizer, batch_size=64, gamma=0.99):
    if len(buffer) < batch_size:
        return None
    batch = random.sample(buffer, batch_size)
    s, a, r, s_next, done = zip(*batch)

    s      = torch.as_tensor(np.array(s),      dtype=torch.float32)   # (B,4)
    a      = torch.as_tensor(a,                dtype=torch.int64).unsqueeze(1)  # (B,1)
    r      = torch.as_tensor(r,                dtype=torch.float32).unsqueeze(1)  # (B,1)
    s_next = torch.as_tensor(np.array(s_next), dtype=torch.float32)   # (B,4)
    done   = torch.as_tensor(done,             dtype=torch.float32).unsqueeze(1)  # (B,1)

    # Q(s,a;θ): tomamos la columna de la acción efectivamente tomada -> gather
    q_sa = q_net(s).gather(1, a)                       # (B,1)

    # target: r + γ max_a' Q(s',a';θ⁻) · (1-done), con la red CONGELADA y detach
    with torch.no_grad():
        q_next_max = target_net(s_next).max(dim=1, keepdim=True)[0]   # (B,1)
        target = r + gamma * q_next_max * (1.0 - done)                # (B,1)

    loss = F.smooth_l1_loss(q_sa, target)              # Huber: robusta a outliers
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(q_net.parameters(), 100.0)  # estabiliza
    optimizer.step()
    return loss.item()

# ---------- Loop de entrenamiento sobre episodios ----------
def train_dqn(n_episodes=600, C=500, eps_start=1.0, eps_end=0.05, eps_decay=2000):
    env = gym.make("CartPole-v1")
    n_obs = env.observation_space.shape[0]   # 4
    n_act = env.action_space.n               # 2

    q_net      = QNetwork(n_obs, n_act)
    target_net = QNetwork(n_obs, n_act)
    target_net.load_state_dict(q_net.state_dict())   # θ⁻ ← θ inicial
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = deque(maxlen=10_000)

    step_count = 0
    recent = deque(maxlen=50)
    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            eps = eps_end + (eps_start - eps_end) * np.exp(-step_count / eps_decay)
            action = select_action(state, q_net, eps, n_act)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.append((state, action, reward, next_state, float(terminated)))
            state = next_state
            ep_reward += reward
            step_count += 1

            optimize_model(q_net, target_net, buffer, optimizer)

            # sincronizar la target network cada C pasos
            if step_count % C == 0:
                target_net.load_state_dict(q_net.state_dict())

        recent.append(ep_reward)
        if ep % 20 == 0:
            print(f"ep {ep:4d} | reward {ep_reward:5.0f} | "
                  f"media50 {np.mean(recent):6.1f} | eps {eps:.3f}")
    return q_net

# q_net = train_dqn()
```

Dos sutilezas de implementación: usamos `terminated` (no `truncated`) en el flag `done` del target — un episodio que se corta por llegar al límite de 500 pasos *no* es un fallo, así que ahí sí hay futuro y no debemos anular el bootstrap. Y `gather(1, a)` selecciona, de las 2 columnas de salida, exactamente la $Q$ de la acción que tomamos: ese es el único valor que aparece en la pérdida.

La curva esperada: las primeras decenas de episodios el reward ronda 10-30 (el poste cae casi de inmediato, casi puro azar). Hacia el episodio 150-300 la media empieza a trepar, y con suerte la media móvil supera 195-475 — CartPole se considera "resuelto" en media 195 sobre 100 episodios en la versión v0, y la v1 permite llegar a 500.

---

## 7. Implementación TensorFlow

Equivalente con `tf.keras` y `GradientTape`. La red target es un segundo `Sequential` cuyos pesos copiamos con `set_weights`.

```python
import tensorflow as tf
import numpy as np
import gymnasium as gym
import random
from collections import deque

def build_q_network(n_obs=4, n_actions=2, hidden=128):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_obs,)),
        tf.keras.layers.Dense(hidden, activation="relu"),
        tf.keras.layers.Dense(hidden, activation="relu"),
        tf.keras.layers.Dense(n_actions),          # Q-valores, uno por acción
    ])

def select_action(state, q_net, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)
    s = tf.convert_to_tensor(state[None, :], dtype=tf.float32)   # (1,4)
    q = q_net(s, training=False)                                  # (1,2)
    return int(tf.argmax(q, axis=1)[0].numpy())

@tf.function
def train_step(q_net, target_net, optimizer, s, a, r, s_next, done, gamma=0.99):
    # target con la red CONGELADA: stop_gradient implícito (no está en el tape)
    q_next_max = tf.reduce_max(target_net(s_next, training=False), axis=1)  # (B,)
    target = r + gamma * q_next_max * (1.0 - done)                          # (B,)
    target = tf.stop_gradient(target)                                       # explícito

    with tf.GradientTape() as tape:
        q_all = q_net(s, training=True)                       # (B,2)
        # Q(s,a): one-hot de la acción y suma -> selecciona la columna correcta
        a_onehot = tf.one_hot(a, depth=tf.shape(q_all)[1])    # (B,2)
        q_sa = tf.reduce_sum(q_all * a_onehot, axis=1)        # (B,)
        loss = tf.reduce_mean(tf.keras.losses.huber(target, q_sa))

    grads = tape.gradient(loss, q_net.trainable_variables)
    grads = [tf.clip_by_value(g, -100.0, 100.0) for g in grads]
    optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    return loss

def train_dqn_tf(n_episodes=600, C=500, batch_size=64,
                 eps_start=1.0, eps_end=0.05, eps_decay=2000):
    env = gym.make("CartPole-v1")
    n_obs, n_act = env.observation_space.shape[0], env.action_space.n

    q_net      = build_q_network(n_obs, n_act)
    target_net = build_q_network(n_obs, n_act)
    target_net.set_weights(q_net.get_weights())          # θ⁻ ← θ
    optimizer  = tf.keras.optimizers.Adam(learning_rate=1e-3)
    buffer     = deque(maxlen=10_000)

    step_count = 0
    recent = deque(maxlen=50)
    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            eps = eps_end + (eps_start - eps_end) * np.exp(-step_count / eps_decay)
            action = select_action(state, q_net, eps, n_act)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.append((state, action, reward, next_state, float(terminated)))
            state = next_state; ep_reward += reward; step_count += 1

            if len(buffer) >= batch_size:
                b = random.sample(buffer, batch_size)
                s, a, r, s_next, dn = zip(*b)
                train_step(
                    q_net, target_net, optimizer,
                    tf.constant(np.array(s),      tf.float32),
                    tf.constant(a,                tf.int32),
                    tf.constant(r,                tf.float32),
                    tf.constant(np.array(s_next), tf.float32),
                    tf.constant(dn,               tf.float32),
                )
            if step_count % C == 0:
                target_net.set_weights(q_net.get_weights())  # sincronizar θ⁻

        recent.append(ep_reward)
        if ep % 20 == 0:
            print(f"ep {ep:4d} | reward {ep_reward:5.0f} | "
                  f"media50 {np.mean(recent):6.1f} | eps {eps:.3f}")
    return q_net

# q_net = train_dqn_tf()
```

La diferencia idiomática frente a PyTorch: en vez de `gather`, TF selecciona $Q(s,a)$ multiplicando por un **one-hot** de la acción y sumando. Y `target_net(s_next, training=False)` ya está fuera del `GradientTape`, así que no recibe gradiente; el `tf.stop_gradient` explícito es defensivo y deja la intención clara.

---

## 8. Implementación JAX

En JAX el modelo no tiene estado: los pesos son un pytree de parámetros que pasamos explícitamente. Usamos `flax.linen` para la MLP, `optax` para Adam, y `jax.lax.stop_gradient` sobre el target. El replay buffer vive en NumPy puro — no necesita ser diferenciable ni estar en el device.

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import gymnasium as gym
import random
from collections import deque
from functools import partial

# ---------- Red Q en Flax ----------
class QNetwork(nn.Module):
    n_actions: int = 2
    hidden: int = 128
    @nn.compact
    def __call__(self, x):                   # x: (B, 4)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.n_actions)(x)   # (B, 2)

model = QNetwork(n_actions=2)

def select_action(params, state, eps, n_actions, key):
    if random.random() < eps:
        return random.randrange(n_actions)
    q = model.apply(params, state[None, :])          # (1,2)
    return int(jnp.argmax(q, axis=1)[0])

# ---------- Pérdida y paso de gradiente ----------
def dqn_loss(params, target_params, s, a, r, s_next, done, gamma=0.99):
    q_all = model.apply(params, s)                    # (B,2)
    # Q(s,a): toma la columna de la acción tomada
    q_sa = jnp.take_along_axis(q_all, a[:, None], axis=1).squeeze(1)  # (B,)

    # target con red CONGELADA -> stop_gradient
    q_next = model.apply(target_params, s_next)       # (B,2)
    q_next_max = jnp.max(q_next, axis=1)              # (B,)
    target = r + gamma * q_next_max * (1.0 - done)    # (B,)
    target = jax.lax.stop_gradient(target)

    # Huber (smooth L1)
    return jnp.mean(optax.huber_loss(q_sa, target))

@partial(jax.jit, static_argnames=())
def train_step(params, target_params, opt_state, s, a, r, s_next, done):
    loss, grads = jax.value_and_grad(dqn_loss)(
        params, target_params, s, a, r, s_next, done)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -100.0, 100.0), grads)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ---------- Loop ----------
def train_dqn_jax(n_episodes=600, C=500, batch_size=64,
                  eps_start=1.0, eps_end=0.05, eps_decay=2000):
    global optimizer
    env = gym.make("CartPole-v1")
    n_obs, n_act = env.observation_space.shape[0], env.action_space.n

    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, n_obs)))    # inicializa θ
    target_params = jax.tree_util.tree_map(lambda x: x, params)   # θ⁻ ← θ (copia)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    buffer = deque(maxlen=10_000)

    step_count = 0
    recent = deque(maxlen=50)
    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            eps = eps_end + (eps_start - eps_end) * np.exp(-step_count / eps_decay)
            key, sub = jax.random.split(key)
            action = select_action(params, jnp.asarray(state), eps, n_act, sub)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.append((state, action, reward, next_state, float(terminated)))
            state = next_state; ep_reward += reward; step_count += 1

            if len(buffer) >= batch_size:
                b = random.sample(buffer, batch_size)
                s, a, r, s_next, dn = zip(*b)
                params, opt_state, _ = train_step(
                    params, target_params, opt_state,
                    jnp.asarray(np.array(s),      jnp.float32),
                    jnp.asarray(a,                jnp.int32),
                    jnp.asarray(r,                jnp.float32),
                    jnp.asarray(np.array(s_next), jnp.float32),
                    jnp.asarray(dn,               jnp.float32),
                )
            if step_count % C == 0:
                target_params = jax.tree_util.tree_map(lambda x: x, params)  # θ⁻ ← θ

        recent.append(ep_reward)
        if ep % 20 == 0:
            print(f"ep {ep:4d} | reward {ep_reward:5.0f} | "
                  f"media50 {np.mean(recent):6.1f} | eps {eps:.3f}")
    return params

# params = train_dqn_jax()
```

Donde JAX brilla: la pérdida es una **función pura** de `(params, target_params, batch)`, y `jax.lax.stop_gradient(target)` expresa el congelamiento de la red target de forma explícita y local. La copia $\theta^{-}\leftarrow\theta$ es simplemente reasignar el pytree `target_params`. El `@jax.jit` compila todo el paso de gradiente a XLA: el primer step compila, los siguientes vuelan. El replay buffer queda en NumPy/`deque` y solo convertimos a `jnp.array` el minibatch que entra al `train_step`.

{{< concept-alert type="clave" >}}
Verificación de dimensiones, idéntica en los tres frameworks. Con batch $B=64$ y CartPole: el minibatch es $s\in\mathbb{R}^{B\times4}$, acciones $a\in\mathbb{Z}^{B}$, recompensas $r\in\mathbb{R}^{B}$. La red da $Q(s,\cdot)\in\mathbb{R}^{B\times2}$; seleccionamos la columna de la acción (`gather`/one-hot/`take_along_axis`) para obtener $Q(s,a)\in\mathbb{R}^{B}$. El target usa $\max_{a'}Q(s',a';\theta^-)\in\mathbb{R}^{B}$. Predicción y target tienen la **misma forma** $(B,)$ antes de la pérdida — si no, hay un broadcasting silencioso que arruina el entrenamiento.
{{< /concept-alert >}}

---

## 9. Por qué replay y target network son cruciales: la tríada mortal

DQN no es "Q-Learning con una red en vez de tabla y ya". Ese DQN ingenuo **diverge**. La razón tiene nombre — la **tríada mortal** (Sutton & Barto): la combinación de tres ingredientes que, juntos, hacen inestable o divergente al aprendizaje por diferencias temporales.

| Ingrediente de la tríada | Qué aporta | Presente en DQN |
|---|---|---|
| **Aproximación de funciones** | la MLP $Q(s,a;\theta)$ generaliza entre estados | sí (es el punto de DQN) |
| **Bootstrapping** | el target usa la propia estimación $\max Q(s',a')$, no el retorno real | sí (target de Bellman) |
| **Entrenamiento off-policy** | aprendemos de datos generados por otra política (el buffer, una mezcla histórica) | sí (experience replay) |

Cuando los tres coinciden, el error puede **amplificarse en cada update** en vez de decrecer: una sobreestimación de $Q(s',a')$ infla el target, que infla la predicción, que infla el siguiente target... un lazo de retroalimentación divergente. Los dos trucos de DQN atacan directamente dos de los mecanismos por los que ese lazo se descontrola:

- **Experience replay** rompe la **correlación temporal**. Sin él, los minibatches son trayectorias casi idénticas; la red sobreajusta el tramo de estados que está visitando *ahora* y olvida el resto. Con un buffer que mezcla épocas, el gradiente se acerca a i.i.d. y el entrenamiento se vuelve estadísticamente sano. Es lo que vuelve *manejable* el componente off-policy.
- **Target network** rompe el **objetivo móvil**. Sin ella, predicción y target comparten $\theta$, así que cada paso de gradiente persigue un blanco que se mueve con él — oscilación o divergencia. Congelando $\theta^{-}$ por $C$ pasos, convertimos el RL en una secuencia de problemas de regresión supervisada estables. Es lo que domestica al **bootstrapping**.

{{< concept-alert type="advertencia" >}}
La prueba empírica es contundente: si en cualquiera de las tres implementaciones de arriba (a) usás la transición recién vivida en vez de samplear del buffer, o (b) calculás el target con `q_net` en vez de `target_net`, el reward **no sube** o colapsa tras un arranque prometedor. No son "mejoras de rendimiento": son **condiciones de convergencia**. Quitá uno y mirá la curva caer — es el mejor experimento didáctico de toda esta práctica.
{{< /concept-alert >}}

---

## 10. Las mejoras: Double DQN, Dueling, PER

DQN abrió una familia entera de refinamientos. Los tres más importantes, en orden de impacto:

**Double DQN** (van Hasselt et al., 2015). El $\max_{a'}$ del target de DQN tiende a **sobreestimar** los $Q$-valores: como la misma red elige la acción *y* evalúa su valor, cualquier ruido positivo en la estimación se selecciona sistemáticamente (sesgo de maximización). Double DQN **desacopla** las dos operaciones: la **red online** elige el $\arg\max$, y la **red target** lo evalúa.

$$
y^{\text{DQN}} = r + \gamma\, Q(s', \textstyle\arg\max_{a'} Q(s',a';\theta);\, \theta^{-})
$$

El cambio en código es de una línea: en vez de tomar `target_net(s_next).max()`, se hace `a* = q_net(s_next).argmax()` (red online) y luego `target_net(s_next)[a*]` (red target). Reduce el sesgo de sobreestimación y mejora la estabilidad casi gratis. Detalles en el [análisis de Double DQN](/papers/double-dqn-van-hasselt-2015).

**Dueling DQN** (Wang et al., 2015). Reescribe la arquitectura de la red para estimar por separado el **valor del estado** $V(s)$ y la **ventaja** de cada acción $A(s,a)$, recombinándolos como $Q(s,a)=V(s)+\big(A(s,a)-\tfrac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')\big)$. La intuición: en muchos estados *da igual* qué acción tomes (el poste está perfectamente vertical), y separar "cuán bueno es este estado" de "cuánto importa la acción" permite aprender $V(s)$ con todas las muestras del estado, sin diluirlo entre acciones.

**Prioritized Experience Replay (PER)** (Schaul et al., 2015). El replay uniforme samplea todas las transiciones con igual probabilidad, pero algunas son **más informativas** que otras — aquellas con mayor error de TD $\lvert y - Q(s,a)\rvert$ (las "sorpresas"). PER samplea con probabilidad proporcional a ese error, acelerando el aprendizaje, y corrige el sesgo introducido con *importance sampling*. En código: el buffer pasa de un `deque` uniforme a una estructura con prioridades (típicamente un *sum-tree* para muestreo $O(\log N)$).

| Mejora | Qué corrige | Costo de implementación |
|---|---|---|
| **Double DQN** | sobreestimación del $\max$ (sesgo de maximización) | trivial: 1-2 líneas en el target |
| **Dueling DQN** | comparte aprendizaje de $V(s)$ entre acciones | medio: cambia la cabeza de la red |
| **PER** | samplea las transiciones más informativas | alto: sum-tree + corrección IS |

Estas tres, combinadas con otras (multi-step, distributional, noisy nets), forman el agente **Rainbow** (Hessel et al., 2017), que las integra todas y fue durante años el estado del arte en Atari.

---

## 11. Comparación lado a lado de los tres frameworks

| Concepto | PyTorch | TensorFlow | JAX |
|---|---|---|---|
| Red Q | `nn.Module` + `nn.Sequential` | `tf.keras.Sequential` | `flax.linen.Module` (pytree de params) |
| Estado de los pesos | mutable, en el módulo | mutable, en el modelo | inmutable, pasado explícito |
| Seleccionar $Q(s,a)$ | `q.gather(1, a)` | `sum(q * one_hot(a))` | `jnp.take_along_axis(q, a)` |
| Target congelado | `target_net(s_next)` + `.detach()` | red fuera del tape + `tf.stop_gradient` | `model.apply(target_params,…)` + `jax.lax.stop_gradient` |
| Sincronizar $\theta^- \leftarrow \theta$ | `target.load_state_dict(q.state_dict())` | `target.set_weights(q.get_weights())` | reasignar el pytree `target_params` |
| Backprop | `loss.backward()` + `optim.step()` | `tape.gradient` + `apply_gradients` | `jax.value_and_grad` + `optax` |
| Replay buffer | `deque` (NumPy/Python) | `deque` | `deque` (NumPy puro, fuera del device) |
| Compilación | `torch.compile` (opcional) | `@tf.function` | `@jax.jit` (esencial) |

La lectura: para **entender** DQN, PyTorch es el más directo (el `.detach()` hace visible el congelamiento del target). TF compila bien el `train_step` con `@tf.function` para throughput. JAX separa limpiamente la lógica pura (pérdida, gradiente) del estado mutable (buffer, contadores), y el `stop_gradient` deja la tríada explícita — pero exige cuidado con el manejo de claves PRNG y el JIT.

---

## 12. Cómo seguir

1. **Apaga la target network** (usa `q_net` en el target) y grafica la curva de reward. Vas a ver oscilar o colapsar. Repite apagando el replay (entrena con la última transición). Es el experimento que vuelve tangible la tríada mortal.
2. **Implementa Double DQN** cambiando solo el cálculo del target (argmax con la red online, evaluación con la target). Compara las curvas: Double DQN suele ser más estable.
3. **Mide la sensibilidad a hiperparámetros**: barre $C \in \{1, 100, 500, 2000\}$ y $\gamma \in \{0.9, 0.99, 0.999\}$. DQN es notoriamente sensible.
4. **Pasa a un entorno más difícil** (`LunarLander-v2`, 8 dims de estado, 4 acciones) reutilizando *exactamente* la misma maquinaria — solo cambian `n_obs` y `n_actions`.
5. **Escala a píxeles**: reemplaza la MLP por una CNN sobre frames de Atari (apilando 4 frames para capturar movimiento), tal como el DQN original de *Nature*.

---

## 13. Cross-links

- [Teoría - Clase 31: Aprendizaje Reforzado](/clases/clase-31): Q-Learning tabular, la ecuación de Bellman y la transición a DQN que esta práctica implementa.
- [Fundamento: Aprendizaje reforzado](/fundamentos/aprendizaje-reforzado): MDP, política, función de valor, Bellman y Q-Learning — la base formal de todo lo de arriba.
- [Paper DQN Nature (Mnih et al., 2015)](/papers/dqn-nature-mnih-2015): el paper canónico que introdujo experience replay y target network sobre 49 juegos de Atari.
- [Paper Double DQN (van Hasselt et al., 2015)](/papers/double-dqn-van-hasselt-2015): la corrección del sesgo de sobreestimación del $\max$.
- [Laboratorio - Clase 31](/laboratorios/lab-31): la versión guiada en notebook, con el entorno ejecutado y la curva de aprendizaje medida.

---

**Ver también:** [Profundización - Clase 31](/clases/clase-31/profundizacion) · [Teoría - Clase 31](/clases/clase-31/teoria).
