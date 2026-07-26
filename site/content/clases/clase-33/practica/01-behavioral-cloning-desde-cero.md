---
title: "Behavioral Cloning desde cero"
weight: 1
math: true
---

La [teoría de la Clase 33](/clases/clase-33/teoria) plantea el atajo más natural del aprendizaje por imitación: *si ya tengo demostraciones de un experto, ¿por qué no aprender la política directamente, como un problema supervisado?* Eso es el **Behavioral Cloning** (BC): registrar los pares (estado, acción) del experto y entrenar un clasificador $\pi_\theta(a\mid s)$ que los reproduzca. En este capítulo lo construimos **desde cero** sobre un gridworld resbaladizo de juguete, y —más importante— **reproducimos el modo de fallo** que la clase advierte: el *distribution shift* y su **compounding error**. El [camino 02 (DAgger)](/clases/clase-33/practica/02-dagger-desde-cero) lo arregla.

> **Lecturas de apoyo:** el [fundamento de Aprendizaje por Imitación](/fundamentos/aprendizaje-por-imitacion) explica el porqué; el [paper de Ross et al. (2011)](/papers/dagger-ross-2011) prueba la cota $\mathcal{O}(T^2\epsilon)$ que veremos aparecer empíricamente.

---

## 1. El montaje: un experto que sí podemos consultar

Para estudiar imitación necesitamos dos cosas: un **experto** cuya política conozcamos (para generar demostraciones y —en el camino 02— para consultarlo sobre estados nuevos) y un **ambiente** donde la política del aprendiz pueda *desviarse* de la del experto. Usamos un **gridworld $N\times N$ resbaladizo**: el agente parte arriba-izquierda y debe llegar a la meta abajo-derecha; el experto es la política de **camino más corto** (Manhattan), analítica y consultable en cualquier estado. La clave didáctica es que el ambiente es **resbaladizo**: con probabilidad $p$ la acción ejecutada se perturba. Ese resbalón es lo que empuja al aprendiz fuera de la banda de estados que visitó el experto.

```python
import numpy as np

class SlipperyGrid:
    """Gridworld NxN. Estado = (fila, col). 4 acciones: 0=arriba,1=abajo,2=izq,3=der.
    Meta en (N-1, N-1). Con prob. `slip` la acción se reemplaza por una aleatoria."""
    def __init__(self, n=8, slip=0.15, seed=0):
        self.n, self.slip = n, slip
        self.rng = np.random.default_rng(seed)
        self.goal = (n - 1, n - 1)

    def reset(self, start=(0, 0)):
        self.s = start
        return self.s

    def step(self, a):
        if self.rng.random() < self.slip:          # resbalón: acción aleatoria
            a = self.rng.integers(4)
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
        r, c = self.s
        r = min(max(r + dr, 0), self.n - 1)         # rebota contra los bordes
        c = min(max(c + dc, 0), self.n - 1)
        self.s = (r, c)
        done = (self.s == self.goal)
        return self.s, (1.0 if done else -0.01), done

def expert_action(s, goal):
    """Política experta: reduce primero la distancia vertical, luego la horizontal."""
    r, c = s; gr, gc = goal
    if r < gr:  return 1     # abajo
    if r > gr:  return 0     # arriba
    if c < gc:  return 3     # derecha
    if c > gc:  return 2     # izquierda
    return 1                 # ya en la meta (irrelevante)
```

Este experto es **óptimo y determinista**, y podemos evaluarlo en *cualquier* estado —incluso en estados que un camino óptimo nunca visitaría (por ejemplo, si el aprendiz resbaló hacia una esquina equivocada). Esa propiedad es justamente la que DAgger explotará.

{{< concept-alert type="clave" >}}
El experto es una **función consultable** $s \mapsto a^*$, no solo un conjunto fijo de trayectorias. El Behavioral Cloning **solo usa** trayectorias del experto (estados que el experto visita). DAgger usará también la capacidad de **consultar** al experto sobre estados que visita el *aprendiz*. Toda la diferencia entre ambos vive en esa distinción.
{{< /concept-alert >}}

---

## 2. Generar el dataset de demostraciones

Rodamos el experto muchas veces y guardamos cada par (estado, acción). Codificamos el estado como un vector one-hot de dimensión $N^2$ (entrada de la red) y la acción como una etiqueta $\{0,1,2,3\}$.

```python
def one_hot(s, n):
    v = np.zeros(n * n, dtype=np.float32)
    v[s[0] * n + s[1]] = 1.0
    return v

def collect_expert_data(env, n_episodes=200, max_steps=100):
    X, Y = [], []
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = expert_action(s, env.goal)     # el experto decide
            X.append(one_hot(s, env.n)); Y.append(a)
            s, _, done = env.step(a)           # el ambiente resbala
            if done: break
    return np.array(X), np.array(Y, dtype=np.int64)

env = SlipperyGrid(n=8, slip=0.15, seed=0)
X, Y = collect_expert_data(env)
print(X.shape, Y.shape)   # p.ej. (~1600, 64) (~1600,)
```

Nota sutil pero decisiva: como el ambiente **resbala** incluso durante las demostraciones, el experto sí visita ocasionalmente estados fuera del camino ideal —pero **muy pocas veces**, y siempre corrige de inmediato. El aprendiz verá esos estados raros con frecuencia ínfima en su dataset. Esa escasez es la raíz del problema.

---

## 3. La política del aprendiz: un clasificador (triple framework)

El Behavioral Cloning es, literalmente, **clasificación multiclase**: entrada = estado one-hot, salida = distribución sobre 4 acciones, pérdida = entropía cruzada. Aquí está el corazón —una MLP de una capa oculta— en los tres frameworks.

### PyTorch

```python
import torch, torch.nn as nn

def make_policy_torch(n_states, n_actions=4, hidden=64):
    return nn.Sequential(
        nn.Linear(n_states, hidden), nn.ReLU(),
        nn.Linear(hidden, n_actions),      # logits (sin softmax: lo aplica la pérdida)
    )

def train_bc_torch(X, Y, n_states, epochs=30, lr=1e-2):
    policy = make_policy_torch(n_states)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(policy(Xt), Yt)     # entropía cruzada estado→acción experta
        loss.backward(); opt.step()
    return policy

@torch.no_grad()
def act_torch(policy, s, n):
    logits = policy(torch.tensor(one_hot(s, n)))
    return int(logits.argmax())
```

### TensorFlow

```python
import tensorflow as tf

def make_policy_tf(n_states, n_actions=4, hidden=64):
    return tf.keras.Sequential([
        tf.keras.layers.Input((n_states,)),
        tf.keras.layers.Dense(hidden, activation="relu"),
        tf.keras.layers.Dense(n_actions),          # logits
    ])

def train_bc_tf(X, Y, n_states, epochs=30, lr=1e-2):
    policy = make_policy_tf(n_states)
    policy.compile(optimizer=tf.keras.optimizers.Adam(lr),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    policy.fit(X, Y, epochs=epochs, batch_size=256, verbose=0)
    return policy

def act_tf(policy, s, n):
    logits = policy(one_hot(s, n)[None, :])
    return int(tf.argmax(logits, axis=1)[0])
```

### JAX (con Flax + Optax)

```python
import jax, jax.numpy as jnp, optax
from flax import linen as fnn

class PolicyJAX(fnn.Module):
    hidden: int = 64
    n_actions: int = 4
    @fnn.compact
    def __call__(self, x):
        x = fnn.relu(fnn.Dense(self.hidden)(x))
        return fnn.Dense(self.n_actions)(x)         # logits

def train_bc_jax(X, Y, n_states, epochs=30, lr=1e-2):
    model = PolicyJAX()
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, n_states)))
    opt = optax.adam(lr); opt_state = opt.init(params)

    def loss_fn(params, x, y):
        logits = model.apply(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    Xj, Yj = jnp.array(X), jnp.array(Y)
    for _ in range(epochs):
        params, opt_state, _ = step(params, opt_state, Xj, Yj)
    return model, params

def act_jax(model, params, s, n):
    logits = model.apply(params, one_hot(s, n)[None, :])
    return int(jnp.argmax(logits, axis=1)[0])
```

Las tres implementaciones son el **mismo** clasificador: MLP → logits → entropía cruzada con etiquetas enteras. No hay nada específico de RL aquí; es aprendizaje supervisado puro. Ese es precisamente el punto —y también la trampa.

---

## 4. El modo de fallo: compounding error

Entrenemos la política y evaluémosla dejándola **conducir sola** (sin experto). Medimos la tasa de éxito en llegar a la meta.

```python
def evaluate(act_fn, env, n_episodes=200, max_steps=200):
    wins = 0
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = act_fn(s)                  # ahora decide el APRENDIZ
            s, _, done = env.step(a)
            if done: wins += 1; break
    return wins / n_episodes

policy = train_bc_torch(X, Y, n_states=64)
print("BC éxito:", evaluate(lambda s: act_torch(policy, s, 8), env))
print("Experto :", evaluate(lambda s: expert_action(s, env.goal), env))
```

El experto llega casi siempre. El Behavioral Cloning, en cambio, **falla mucho más de lo que su baja pérdida de entrenamiento sugeriría**. La razón es exactamente la de la clase: la red aprendió a actuar bien en la **banda estrecha de estados que visita el experto** (la diagonal hacia la meta). Cuando un resbalón la empuja fuera de esa banda —a una esquina, a un borde lejano— aterriza en estados **casi ausentes del dataset**, predice mal, y ese error la lleva a estados **aún más raros**. Los errores se **componen**: un pequeño $\epsilon$ por paso se amplifica a $\mathcal{O}(T^2\epsilon)$ sobre el episodio.

{{< concept-alert type="advertencia" >}}
La pérdida de entrenamiento del BC puede ser **excelente** y aun así la política **fallar al desplegarse**. No es overfitting clásico: es que train y test provienen de **distribuciones de estados distintas** —el experto genera train, la propia política genera test. El aprendizaje supervisado asume que ambas coinciden; en control, **no coinciden**. Esta es la lección central del capítulo.
{{< /concept-alert >}}

### 4.1 Comprobar el diagnóstico

Si el problema es realmente el distribution shift, entonces al **subir el resbalón** (más desviaciones → más estados fuera de banda) el BC debería empeorar más rápido que el experto. Y así ocurre:

```python
for slip in [0.0, 0.1, 0.2, 0.3]:
    e = SlipperyGrid(n=8, slip=slip, seed=1)
    Xs, Ys = collect_expert_data(e)
    pol = train_bc_torch(Xs, Ys, 64)
    bc  = evaluate(lambda s: act_torch(pol, s, 8), e)
    exp = evaluate(lambda s: expert_action(s, e.goal), e)
    print(f"slip={slip:.1f}  BC={bc:.2f}  experto={exp:.2f}  brecha={exp-bc:.2f}")
```

Con `slip=0.0` (sin resbalón, un solo camino) el BC casi iguala al experto —memoriza la diagonal y listo. A medida que sube el resbalón, la **brecha se abre**: el ambiente arrastra a la política a territorio no visto. Ese es el fenómeno que DAgger elimina.

---

## 5. Qué nos llevamos

- El **Behavioral Cloning** convierte la imitación en clasificación supervisada: simple, rápido, agnóstico al framework (idéntico en PyTorch, TensorFlow y JAX).
- Su talón de Aquiles es el **distribution shift**: entrena sobre estados del experto, pero al actuar visita los suyos propios; los errores se **acumulan** ($\mathcal{O}(T^2\epsilon)$).
- El diagnóstico se confirma subiendo el resbalón: la brecha con el experto crece con la frecuencia de desvíos.

En el [camino 02](/clases/clase-33/practica/02-dagger-desde-cero) cerramos el bucle con **DAgger**: consultamos al experto sobre los estados que visita el aprendiz, y la garantía pasa de cuadrática a **lineal**.

---

**Ver también:** [Clase 33 - Teoría](/clases/clase-33/teoria) · [Clase 33 - Profundización](/clases/clase-33/profundizacion) · [Camino 02: DAgger](/clases/clase-33/practica/02-dagger-desde-cero) · [Laboratorio: DAgger sobre Breakout](/laboratorios/lab-33).
