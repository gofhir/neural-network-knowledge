---
title: "DAgger desde cero"
weight: 2
math: true
---

El [camino 01](/clases/clase-33/practica/01-behavioral-cloning-desde-cero) mostró cómo el **Behavioral Cloning** falla por *distribution shift*: entrenado sobre los estados del experto, el aprendiz visita los suyos propios y los errores se **acumulan** ($\mathcal{O}(T^2\epsilon)$). Este capítulo implementa **DAgger** (Dataset Aggregation, [Ross et al. 2011](/papers/dagger-ross-2011)) sobre **el mismo gridworld resbaladizo**, y verifica empíricamente que **cierra la brecha**. La idea, en una frase: *deja que el aprendiz conduzca, y pídele al experto que etiquete los estados donde el aprendiz efectivamente se mete.*

> **Lecturas de apoyo:** el [fundamento de Aprendizaje por Imitación](/fundamentos/aprendizaje-por-imitacion) sitúa DAgger en el mapa; la [profundización de la clase](/clases/clase-33/profundizacion) desarrolla su garantía $\mathcal{O}(T\epsilon)$ vía no-regret. Reutilizamos el entorno `SlipperyGrid`, `expert_action`, `one_hot` y las redes del [camino 01](/clases/clase-33/practica/01-behavioral-cloning-desde-cero).

---

## 1. La idea: recolectar donde el aprendiz visita, no donde el experto visita

El Behavioral Cloning entrena sobre $d_{\pi^*}$ (la distribución de estados del **experto**) pero se evalúa sobre $d_{\hat\pi}$ (la del **aprendiz**). DAgger elimina esa discrepancia con un bucle:

1. Entrena una política inicial $\hat\pi_1$ por BC sobre las demostraciones del experto.
2. **Rueda la política actual** en el ambiente y guarda los estados que visita.
3. **Consulta al experto** qué acción tomaría *en esos estados visitados* (aunque el aprendiz ya no lo esté imitando bien).
4. **Agrega** esos pares $(s, a^*)$ al dataset acumulado $\mathcal{D}$.
5. **Reentrena** $\hat\pi_{i+1}$ sobre todo $\mathcal{D}$. Repite.

Al iterar, $\mathcal{D}$ va cubriendo justamente los estados donde el aprendiz se equivoca —los estados "de recuperación" que el experto casi nunca visita por sí solo. La política aprende no solo a seguir el camino, sino a **volver a él** cuando se desvía.

{{< concept-alert type="clave" >}}
DAgger es **active learning dirigido al control**: en vez de etiquetar estados al azar, etiqueta exactamente los estados que la política alcanza —los más informativos para corregir su trayectoria. El costo es requerir un **experto consultable** durante el entrenamiento (aquí, la función `expert_action`; en el [laboratorio](/laboratorios/lab-33), un DQN pre-entrenado).
{{< /concept-alert >}}

---

## 2. El bucle DAgger (agnóstico al framework)

El corazón de DAgger es puro control de datos: rodar, consultar, agregar, reentrenar. La red interna es la misma MLP del camino 01, así que el bucle es idéntico en los tres frameworks —solo cambian las funciones `train_bc_*` y `act_*` que ya definimos. Aquí el orquestador, escrito de forma neutral:

```python
import numpy as np

def run_learner(act_fn, env, max_steps=200):
    """Rueda la política del APRENDIZ y devuelve los estados que visita."""
    visited = []
    s = env.reset()
    for _ in range(max_steps):
        visited.append(s)
        a = act_fn(s)
        s, _, done = env.step(a)
        if done: break
    return visited

def dagger(env, train_fn, act_factory, n_iters=10, rollouts_per_iter=20):
    """train_fn(X, Y) -> modelo ;  act_factory(modelo) -> (estado -> acción)."""
    n = env.n
    # Iteración 1: dataset inicial = demostraciones del experto (BC puro)
    X, Y = collect_expert_data(env, n_episodes=rollouts_per_iter)
    history = []
    for it in range(n_iters):
        model = train_fn(X, Y)                     # (re)entrena sobre TODO D
        act_fn = act_factory(model)                # política actual del aprendiz
        # Evalúa la política actual conduciendo sola
        win = evaluate(act_fn, env)
        history.append(win)
        # --- Paso DAgger: rodar aprendiz, consultar experto sobre lo visitado ---
        new_X, new_Y = [], []
        for _ in range(rollouts_per_iter):
            for s in run_learner(act_fn, env):     # estados que visita el APRENDIZ
                new_X.append(one_hot(s, n))
                new_Y.append(expert_action(s, env.goal))   # etiqueta del EXPERTO
        X = np.concatenate([X, np.array(new_X, np.float32)])
        Y = np.concatenate([Y, np.array(new_Y, np.int64)])
    return history, X, Y
```

Obsérvese la línea decisiva: `expert_action(s, env.goal)` se llama sobre `s` proveniente de `run_learner` —**estados del aprendiz**, no del experto. Ahí está toda la diferencia con el camino 01, donde `expert_action` solo se aplicaba a estados generados por el propio experto.

---

## 3. Conectar la red (los tres frameworks)

`dagger` es genérico: recibe una función de entrenamiento y una fábrica de política. Cada framework provee ese par reutilizando las piezas del [camino 01](/clases/clase-33/practica/01-behavioral-cloning-desde-cero).

### PyTorch

```python
hist_torch, X_t, Y_t = dagger(
    SlipperyGrid(n=8, slip=0.15, seed=0),
    train_fn=lambda X, Y: train_bc_torch(X, Y, n_states=64, epochs=40),
    act_factory=lambda pol: (lambda s: act_torch(pol, s, 8)),
)
print("éxito por iteración (PyTorch):", [f"{h:.2f}" for h in hist_torch])
```

### TensorFlow

```python
hist_tf, _, _ = dagger(
    SlipperyGrid(n=8, slip=0.15, seed=0),
    train_fn=lambda X, Y: train_bc_tf(X, Y, n_states=64, epochs=40),
    act_factory=lambda pol: (lambda s: act_tf(pol, s, 8)),
)
```

### JAX (Flax + Optax)

```python
def train_fn_jax(X, Y):
    model, params = train_bc_jax(X, Y, n_states=64, epochs=40)
    return (model, params)

hist_jax, _, _ = dagger(
    SlipperyGrid(n=8, slip=0.15, seed=0),
    train_fn=train_fn_jax,
    act_factory=lambda mp: (lambda s: act_jax(mp[0], mp[1], s, 8)),
)
```

En los tres casos, la iteración 0 (`history[0]`) es exactamente el Behavioral Cloning del camino 01 —el dataset aún es solo del experto. A partir de la iteración 1, el dataset empieza a incluir estados de recuperación, y la tasa de éxito **sube**.

---

## 4. El resultado: la brecha se cierra

Al graficar `history` se observa el patrón característico de DAgger: la primera iteración (BC puro) rinde bajo, y las siguientes **suben monótonamente** hasta acercarse al desempeño del experto, para luego estabilizarse.

```python
import matplotlib.pyplot as plt
exp = evaluate(lambda s: expert_action(s, SlipperyGrid(8, 0.15).goal),
               SlipperyGrid(8, 0.15))
plt.plot(range(len(hist_torch)), hist_torch, "o-", label="DAgger")
plt.axhline(hist_torch[0], ls="--", c="gray", label="BC (iter 0)")
plt.axhline(exp, ls=":", c="green", label="experto")
plt.xlabel("iteración DAgger"); plt.ylabel("tasa de éxito"); plt.legend()
```

La lectura conecta con la teoría: cada iteración añade a $\mathcal{D}$ los estados donde el aprendiz **fallaba**, de modo que la distribución de entrenamiento converge a la distribución que el aprendiz realmente induce, $d_{\hat\pi}$. Con la deriva por iteración acotada por $\lVert d_{\pi_i}-d_{\hat\pi_i}\rVert_1 \le 2T\beta_i$ y $\beta_i \to 0$, el error total pasa de $\mathcal{O}(T^2\epsilon)$ (BC) a $\mathcal{O}(T\epsilon)$ (DAgger) —lineal en el horizonte.

{{< concept-alert type="recordar" >}}
La mejora **no** viene de una red mejor ni de más épocas: es la **misma** MLP del camino 01. Viene de entrenar sobre los **datos correctos** —los estados que el aprendiz visita, etiquetados por el experto. En imitación, *qué* estados etiquetas importa tanto como *cuánto* entrenas.
{{< /concept-alert >}}

---

## 5. Variante: mezcla experto-aprendiz ($\beta$-schedule)

El DAgger original rueda, en la iteración $i$, una **política mezcla** $\pi_i = \beta_i\,\pi^* + (1-\beta_i)\,\hat\pi_i$, con $\beta_i \to 0$. Al principio ($\beta$ alto) el experto "sujeta el volante" y evita que el aprendiz recolecte basura de estados absurdos; al final ($\beta \to 0$) recolecta puramente sobre su propia distribución. Es un cambio de una línea en `run_learner`:

```python
def run_learner_mixed(act_fn, env, beta, max_steps=200):
    visited = []; s = env.reset()
    for _ in range(max_steps):
        visited.append(s)
        a = expert_action(s, env.goal) if np.random.random() < beta else act_fn(s)
        s, _, done = env.step(a)
        if done: break
    return visited
# en dagger: beta = max(0.0, 1.0 - it / 3)  # decae en las primeras iteraciones
```

En este gridworld, tan pequeño y con experto perfecto, la variante sin mezcla ($\beta=0$) ya converge bien; el $\beta$-schedule importa más en tareas largas o con expertos costosos/humanos, donde recolectar sobre estados catastróficos temprano es peligroso o caro.

---

## 6. Qué nos llevamos

- **DAgger** arregla el distribution shift del Behavioral Cloning **sin cambiar la red**: solo cambia *sobre qué estados* se recolectan etiquetas expertas.
- El bucle **rodar → consultar experto → agregar → reentrenar** es agnóstico al framework; PyTorch, TensorFlow y JAX comparten el mismo orquestador.
- La garantía teórica ($\mathcal{O}(T\epsilon)$ lineal vs. $\mathcal{O}(T^2\epsilon)$ cuadrática) se manifiesta como una curva de éxito que **sube por iteración** hasta alcanzar al experto.
- El precio es un **experto consultable** en el bucle —el supuesto que el [laboratorio](/laboratorios/lab-33) satisface con un DQN pre-entrenado sobre Atari Breakout.

---

**Ver también:** [Clase 33 - Teoría](/clases/clase-33/teoria) · [Clase 33 - Profundización](/clases/clase-33/profundizacion) · [Camino 01: Behavioral Cloning](/clases/clase-33/practica/01-behavioral-cloning-desde-cero) · [Laboratorio: DAgger sobre Breakout](/laboratorios/lab-33).
