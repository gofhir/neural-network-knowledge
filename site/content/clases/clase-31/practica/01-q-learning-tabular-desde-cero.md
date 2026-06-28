---
title: "Q-Learning tabular desde cero"
weight: 1
math: true
---

La [teoria de la clase 31](/clases/clase-31) introdujo un paradigma distinto al aprendizaje supervisado: un agente que aprende **por ensayo y error**, interactuando con un ambiente y guiandose por recompensas. La pieza central de esa clase es **Q-Learning** (Watkins, 1989; Watkins y Dayan, 1992), el algoritmo que aprende a actuar de forma optima **sin conocer el modelo del ambiente** —sin saber de antemano que pasa al ejecutar cada accion ni cuanto se va a recompensar. Solo prueba, observa y ajusta.

En este capitulo lo construimos **desde cero** sobre un gridworld de juguete definido a mano —sin `gym`, sin librerias de RL, para que todo el ciclo quepa en la cabeza y sea reproducible con copiar y pegar. El nucleo del algoritmo es una **tabla** de numeros (la tabla Q) y una **regla de actualizacion de una linea**. No hay red neuronal, no hay autograd, no hay gradientes que retropropagar. Por eso el corazon del codigo es **NumPy puro**, y es **identico** en cualquier framework.

Justamente por eso lo presentamos distinto al resto de la practica del curso. Primero lo implementamos completo en NumPy: el ambiente, la tabla, el loop, la politica aprendida. Despues mostramos como se ve esa **misma** tabla Q representada como tensor en **PyTorch, TensorFlow y JAX** —no porque haga falta, sino para dejar clarisimo donde **no** entra el deep learning todavia, y por que el salto a **DQN** ([camino 02](/clases/clase-31/practica/02-dqn-desde-cero)) si lo necesita. El contraste es la leccion: cuando la tabla deja de caber, una red neuronal la reemplaza, y recien ahi vuelven PyTorch, TensorFlow y JAX a hacer lo suyo de verdad.

> **Lecturas de apoyo:** el [fundamento de Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado) cubre el MDP, el retorno descontado y la ecuacion de Bellman con calma; el [paper de Watkins y Dayan (1992)](/papers/q-learning-watkins-1992) es la prueba de convergencia original. Aqui nos enfocamos en el codigo.

---

## 1. El MDP: que es exactamente lo que aprendemos

Antes de tocar codigo, fijemos el objeto matematico. Q-Learning resuelve un **Proceso de Decision de Markov** (MDP), definido por la tupla $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

| Simbolo | Nombre | En nuestro gridworld |
|---|---|---|
| $\mathcal{S}$ | espacio de estados | las 16 celdas del grid $4\times4$ |
| $\mathcal{A}$ | espacio de acciones | $\{\text{arriba}, \text{abajo}, \text{izquierda}, \text{derecha}\}$ |
| $P(s' \mid s, a)$ | dinamica de transicion | a que celda llego al moverme (la **desconocemos** desde el agente) |
| $R(s, a, s')$ | funcion de recompensa | $+1$ en la meta, $-1$ en trampas, $-0.01$ por paso |
| $\gamma$ | factor de descuento | $0.95$ (cuanto valen las recompensas futuras) |

La propiedad de **Markov** es la que hace tratable todo esto: el futuro depende solo del estado actual, no de como llegamos ahi. La celda 6 "vale" lo mismo sin importar el camino que tomamos para llegar.

El objetivo del agente es encontrar una **politica** $\pi(s) \to a$ —una receta de que accion tomar en cada estado— que maximice el **retorno descontado** esperado:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

El descuento $\gamma < 1$ cumple dos roles: matematicamente garantiza que la suma converge aunque el episodio sea largo, y conceptualmente codifica "una recompensa hoy vale mas que la misma recompensa en 10 pasos". Con $\gamma = 0.95$, una recompensa a 10 pasos de distancia vale $0.95^{10} \approx 0.60$ de su valor nominal; a 50 pasos, apenas $0.077$. Eso empuja al agente a llegar **rapido** a la meta, no solo a llegar.

### 1.1 La tabla Q y la ecuacion de Bellman

La **funcion de valor-accion** $Q^\pi(s, a)$ responde la pregunta: *"si estoy en el estado $s$, tomo la accion $a$, y desde ahi sigo la politica $\pi$, ¿cuanto retorno espero?"*. La politica optima sale directo de la $Q$ optima $Q^*$: en cada estado, elige la accion de mayor $Q$.

$$
\pi^*(s) = \arg\max_a Q^*(s, a).
$$

La $Q^*$ satisface la **ecuacion de optimalidad de Bellman**, que es la columna vertebral de todo:

$$
Q^*(s, a) = \mathbb{E}_{s'}\Big[\, r + \gamma \max_{a'} Q^*(s', a') \,\Big]. \tag{1}
$$

En palabras: el valor de tomar $a$ en $s$ es la recompensa inmediata **mas** el mejor valor posible desde el estado siguiente, descontado. Es recursiva —$Q^*$ aparece en ambos lados— y por eso se resuelve iterando.

Como tenemos $|\mathcal{S}| = 16$ estados y $|\mathcal{A}| = 4$ acciones, $Q$ es literalmente una **tabla de $16 \times 4 = 64$ numeros**. Eso es "tabular": una entrada por cada par $(s, a)$. Esta finitud es la que se rompe en problemas reales y motiva DQN.

### 1.2 La regla de actualizacion: el corazon del algoritmo

No conocemos $P$ ni $R$, asi que no podemos evaluar la esperanza de la Ecuacion (1) directamente. La idea genial de Watkins es **aproximarla con muestras**: cada vez que el agente ejecuta $a$ en $s$, observa una transicion real $(s, a, r, s')$ y la usa para corregir su estimacion actual de $Q$ un poquito hacia el objetivo de Bellman:

$$
\boxed{\;Q(s, a) \leftarrow Q(s, a) + \alpha\Big[\,\underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{objetivo (TD target)}} - \underbrace{Q(s, a)}_{\text{estimacion actual}}\,\Big]\;} \tag{2}
$$

Desarmemos cada pieza, porque toda la practica gira en torno a esta linea:

- **$r + \gamma \max_{a'} Q(s', a')$** es el **TD target** (objetivo de diferencia temporal): una estimacion "mejorada" del valor, porque incorpora la recompensa **real** $r$ que acabamos de observar mas nuestra mejor estimacion del futuro.
- **$Q(s, a)$** es lo que creiamos antes de movernos.
- **El corchete** es el **TD error** $\delta$: cuanto nos equivocamos. Si es positivo, esta accion resulto mejor de lo esperado; subimos su valor. Si es negativo, la bajamos.
- **$\alpha \in (0, 1]$** es la **tasa de aprendizaje**: cuanto de ese error incorporamos. $\alpha = 1$ reemplaza por completo; $\alpha \to 0$ no aprende. Tipico: $0.1$.

El $\max_{a'}$ es lo que hace a Q-Learning **off-policy** (lo discutimos al final): el target asume que en $s'$ actuaremos **optimamente**, aunque para explorar hayamos tomado una accion sub-optima. Es decir, aprende sobre la politica *greedy* mientras se comporta con una politica *exploratoria*.

---

## 2. NumPy puro (base): el algoritmo completo

Aqui esta toda la sustancia. Cuatro bloques: el ambiente, la tabla, el loop con $\varepsilon$-greedy, y la visualizacion de lo aprendido.

### 2.1 El gridworld a mano

Definimos un grid $4\times4$. El agente parte arriba-izquierda (celda `S`), debe llegar a la meta abajo-derecha (`G`, $+1$) y evitar dos trampas (`H`, $-1$, que terminan el episodio). Cada paso "normal" cuesta $-0.01$ para incentivar caminos cortos.

```text
indices de celda          mapa
  0  1  2  3              S  .  .  .
  4  5  6  7              .  H  .  H
  8  9 10 11              .  .  .  .
 12 13 14 15              .  .  .  G
```

```python
import numpy as np

# --- Definicion del ambiente (todo a mano, sin gym) ---
GRID_ROWS, GRID_COLS = 4, 4
N_STATES = GRID_ROWS * GRID_COLS          # 16 celdas -> 16 estados
N_ACTIONS = 4                              # 0:arriba 1:abajo 2:izq 3:der
ACTION_NAMES = ["↑", "↓", "←", "→"]

START = 0          # celda S (fila 0, col 0)
GOAL = 15          # celda G (fila 3, col 3)
HOLES = {5, 7}     # trampas H

# Cada accion como (delta_fila, delta_col)
MOVES = {
    0: (-1, 0),   # arriba
    1: (1, 0),    # abajo
    2: (0, -1),   # izquierda
    3: (0, 1),    # derecha
}


def state_to_rc(s):
    """Convierte indice de estado 0..15 a (fila, columna)."""
    return divmod(s, GRID_COLS)


def rc_to_state(r, c):
    """Convierte (fila, columna) a indice de estado."""
    return r * GRID_COLS + c


def reset():
    """Inicia un episodio: el agente vuelve a START."""
    return START


def step(state, action):
    """Dinamica del ambiente: dado (estado, accion) devuelve (estado', recompensa, terminado).

    Esta funcion ES el ambiente. El agente NO la conoce por dentro;
    solo la llama y observa lo que sale. Eso es aprendizaje 'model-free'.
    """
    r, c = state_to_rc(state)
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc

    # Si choca contra un muro (sale del grid), se queda donde estaba.
    if not (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS):
        nr, nc = r, c

    next_state = rc_to_state(nr, nc)

    # Recompensa y bandera de termino segun la celda destino.
    if next_state == GOAL:
        return next_state, 1.0, True
    if next_state in HOLES:
        return next_state, -1.0, True
    return next_state, -0.01, False   # costo por paso: incentiva rapidez
```

Tres decisiones de diseño que vale la pena notar:

- **El "choque con muro" deja al agente en su lugar.** Es la convencion mas comun en gridworlds; alternativamente podriamos prohibir la accion. Dejarlo quieto es mas simple y el agente aprende solo a no malgastar pasos contra los bordes.
- **El ambiente es deterministico.** La misma accion en el mismo estado siempre lleva al mismo lugar. En FrozenLake "resbaladizo" hay azar (`P(s'|s,a)` no es degenerada); aqui lo evitamos para que el resultado sea inspeccionable a ojo. Q-Learning maneja ambos casos sin cambiar una linea.
- **El costo por paso $-0.01$** es lo que diferencia "llegar" de "llegar rapido". Sin el, cualquier camino a la meta tendria el mismo retorno y la politica podria dar vueltas.

### 2.2 La tabla Q

El objeto que aprendemos. Una matriz `(N_STATES, N_ACTIONS)` inicializada en cero. La entrada `Q[s, a]` es nuestra estimacion actual de $Q(s, a)$.

```python
# La tabla Q: 16 estados x 4 acciones = 64 numeros. ESTO es lo que se aprende.
Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
```

Inicializar en cero es una eleccion neutra. Inicializar en valores **altos** ("optimismo frente a la incertidumbre") es un truco clasico que fomenta la exploracion temprana: el agente prueba acciones no vistas porque cree que valen mucho, y se desilusiona solo tras probarlas. Lo dejamos en cero por claridad.

### 2.3 La politica $\varepsilon$-greedy: exploracion vs explotacion

El dilema central de RL: ¿el agente **explota** lo que ya sabe (toma la mejor accion segun $Q$) o **explora** algo nuevo (prueba una accion al azar, por si esconde algo mejor)? Si solo explota, se queda atrapado en la primera ruta decente que encuentra. Si solo explora, nunca aprovecha lo aprendido.

La solucion clasica es **$\varepsilon$-greedy**: con probabilidad $\varepsilon$ elige al azar (explora), con probabilidad $1 - \varepsilon$ elige la mejor accion conocida (explota).

```python
def epsilon_greedy(Q, state, epsilon, rng):
    """Elige accion: explora (azar) con prob. epsilon, explota (greedy) si no."""
    if rng.random() < epsilon:
        return rng.integers(N_ACTIONS)           # EXPLORA: accion al azar
    return int(np.argmax(Q[state]))              # EXPLOTA: mejor accion conocida
```

> **Gotcha del `argmax`.** Con la tabla inicializada en cero, `np.argmax` rompe los empates devolviendo siempre el **primer** indice (accion "arriba"). Al inicio, mientras todo vale 0, el agente sesgaria hacia arriba. El `epsilon` alto al comienzo (ver decay abajo) lo compensa; si quieres ser estricto, rompe empates al azar con `np.flatnonzero(Q[s] == Q[s].max())` y elige uno aleatorio de ahi.

### 2.4 El loop de Q-Learning con decay de $\varepsilon$

El programa completo. Por cada episodio: reseteamos, y mientras no termine elegimos accion $\varepsilon$-greedy, damos el paso, **aplicamos la regla de actualizacion (Ecuacion 2)** y avanzamos. Entre episodios, **decaemos $\varepsilon$**: empezamos explorando mucho y terminamos explotando casi siempre.

```python
# --- Hiperparametros ---
ALPHA = 0.1          # tasa de aprendizaje (alpha)
GAMMA = 0.95         # factor de descuento (gamma)
N_EPISODES = 2000
MAX_STEPS = 100      # corta episodios que no terminan (evita loops infinitos)

EPS_START = 1.0      # 100% exploracion al inicio
EPS_END = 0.05       # 5% exploracion al final (nunca deja de explorar del todo)
EPS_DECAY = 0.995    # multiplicador por episodio

rng = np.random.default_rng(42)
Q = np.zeros((N_STATES, N_ACTIONS))
epsilon = EPS_START
returns_history = []      # retorno por episodio, para graficar el aprendizaje

for ep in range(N_EPISODES):
    state = reset()
    total_return = 0.0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        # 1) Elegir accion (exploracion / explotacion)
        action = epsilon_greedy(Q, state, epsilon, rng)

        # 2) Ejecutar en el ambiente: observar (s', r, done)
        next_state, reward, done = step(state, action)

        # 3) ====== REGLA DE Q-LEARNING (Ecuacion 2) ======
        best_next = 0.0 if done else np.max(Q[next_state])   # max_a' Q(s', a')
        td_target = reward + GAMMA * best_next               # objetivo de Bellman
        td_error = td_target - Q[state, action]              # delta
        Q[state, action] += ALPHA * td_error                 # actualizacion
        # =================================================

        state = next_state
        total_return += reward
        steps += 1

    # 4) Decay de epsilon: explorar menos a medida que aprendemos
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    returns_history.append(total_return)

    if (ep + 1) % 200 == 0:
        avg = np.mean(returns_history[-200:])
        print(f"ep {ep+1:4d}  eps={epsilon:.3f}  retorno_medio(200)={avg:+.3f}")
```

Tres sutilezas del loop que suelen morder:

- **`best_next = 0` cuando `done`.** En un estado terminal no hay futuro: el retorno se acaba. Si no lo forzamos a 0, el agente "alucina" valor despues de la meta y la tabla diverge. Es el bug numero uno al implementar Q-Learning a mano.
- **El `max` es sobre el estado siguiente `next_state`, no el actual.** Lee la Ecuacion (2) con cuidado: $\max_{a'} Q(s', a')$.
- **`MAX_STEPS` corta episodios.** Mientras la politica es mala, el agente puede vagar sin llegar a meta ni trampa. El tope evita loops infinitos sin afectar lo aprendido.

### 2.5 Verificacion: la politica y los valores aprendidos

Tras entrenar, extraemos la politica *greedy* (la mejor accion por estado) y el valor de cada estado $V(s) = \max_a Q(s, a)$. Si todo fue bien, las flechas deben dibujar un camino de `S` a `G` esquivando las trampas.

```python
def render_policy(Q):
    """Imprime el grid con la flecha de la mejor accion en cada celda."""
    print("\nPolitica aprendida (mejor accion por celda):")
    for r in range(GRID_ROWS):
        row = []
        for c in range(GRID_COLS):
            s = rc_to_state(r, c)
            if s == GOAL:
                row.append(" G ")
            elif s in HOLES:
                row.append(" H ")
            else:
                row.append(f" {ACTION_NAMES[np.argmax(Q[s])]} ")
        print("".join(row))


def render_values(Q):
    """Imprime V(s) = max_a Q(s,a) por celda."""
    V = Q.max(axis=1).reshape(GRID_ROWS, GRID_COLS)
    print("\nValores de estado V(s) = max_a Q(s,a):")
    for r in range(GRID_ROWS):
        print("  ".join(f"{V[r, c]:+.2f}" for c in range(GRID_COLS)))


render_policy(Q)
render_values(Q)
```

Salida esperada (las flechas exactas pueden variar en celdas equivalentes, pero el flujo S→G es estable):

```text
Politica aprendida (mejor accion por celda):
 ↓  ←  ↓  ↑
 ↓  H  ↓  H
 →  →  ↓  ↓
 →  →  →  G

Valores de estado V(s) = max_a Q(s,a):
+0.73  +0.66  +0.37  -0.01
+0.78  +0.00  +0.88  +0.00
+0.83  +0.88  +0.94  +0.99
+0.85  +0.94  +1.00  +0.00
```

Leyendo el resultado: a lo largo de la **ruta que el agente realmente recorre** los valores **crecen al acercarse a la meta** —de $\approx 0.73$ en `S` hasta $1.00$ junto a `G`—, exactamente como predice el descuento $\gamma$: cada paso mas cerca de la recompensa vale $1/\gamma \approx 1.05$ veces mas. Las celdas-trampa (`H`) marcan $V = 0.00$ porque son terminales y su fila de la tabla nunca se actualiza (no se actua desde ahi); la meta tambien marca $0.00$ por la misma razon. Las celdas alejadas del camino optimo (como la esquina superior derecha) tienen valores bajos o negativos porque el agente las visito poco. Las flechas trazan una ruta que **rodea** las trampas. Esto es la ecuacion de Bellman propagando el $+1$ de la meta hacia atras, paso a paso, episodio a episodio.

```python
# Verificacion automatica: ¿la politica greedy llega a la meta sin caer en trampa?
def rollout_greedy(Q, max_steps=50):
    s, path = reset(), [reset()]
    for _ in range(max_steps):
        s, _, done = step(s, int(np.argmax(Q[s])))
        path.append(s)
        if done:
            break
    return path

path = rollout_greedy(Q)
print("\nTrayectoria greedy:", path)
assert path[-1] == GOAL, "La politica no llega a la meta"
assert not (set(path) & HOLES), "La politica cae en una trampa"
print("OK: la politica llega a la meta esquivando trampas.")
```

Eso es Q-Learning tabular **completo**. Todo lo demas en esta pagina es contexto. El algoritmo entero son las cinco lineas del bloque marcado en 2.4.

---

## 3. La misma tabla Q en PyTorch, TensorFlow y JAX

Aqui esta el punto pedagogico de esta pagina. El algoritmo de arriba **no necesita ningun framework de deep learning**: no hay funcion a derivar, no hay grafo de computo, no hay parametros que un optimizador deba ajustar por gradiente. La actualizacion de la Ecuacion (2) es una **asignacion aritmetica directa** a una celda de una matriz.

Aun asi, vale la pena ver como se representa la tabla Q como **tensor** en cada framework y como luce la actualizacion vectorizada. El ejercicio deja en evidencia que, sin una red de por medio, los tres se reducen a "NumPy con otro nombre" —y prepara el contraste con DQN, donde la tabla se reemplaza por una red y **ahi si** entra el autograd.

En los tres casos hacemos lo mismo: representar `Q` de shape `(16, 4)`, y aplicar la actualizacion para una transicion $(s, a, r, s')$.

### 3.1 PyTorch — `torch.zeros` y asignacion in-place

```python
import torch

# La tabla Q es un tensor. NO requires_grad: no hay nada que derivar.
Q = torch.zeros(N_STATES, N_ACTIONS, dtype=torch.float64)

def q_update_torch(Q, s, a, r, s_next, done, alpha=0.1, gamma=0.95):
    """Misma Ecuacion (2), con tensores PyTorch. Asignacion in-place, sin autograd."""
    best_next = torch.tensor(0.0) if done else Q[s_next].max()
    td_target = r + gamma * best_next
    Q[s, a] += alpha * (td_target - Q[s, a])      # update in-place
    return Q

# Ejemplo: transicion (s=14, a=3 'derecha', r=+1, s'=15 meta, terminal)
Q = q_update_torch(Q, s=14, a=3, r=1.0, s_next=15, done=True)
print(Q[14])   # tensor([... 0.1000])  -> Q[14,3] subio de 0 a alpha*1 = 0.1
```

La indexacion `Q[s_next].max()` y la asignacion `Q[s, a] += ...` son **identicas** a NumPy. No envolvemos nada en `torch.no_grad()` porque ni siquiera activamos el grafo (`requires_grad=False` por defecto). PyTorch aqui es un contenedor de arrays, nada mas.

### 3.2 TensorFlow — `tf.Variable` y `.assign`

```python
import tensorflow as tf

# tf.Variable porque queremos mutar la tabla in-place entre pasos.
Q = tf.Variable(tf.zeros((N_STATES, N_ACTIONS), dtype=tf.float64))

def q_update_tf(Q, s, a, r, s_next, done, alpha=0.1, gamma=0.95):
    """Misma Ecuacion (2), con TensorFlow. Sin GradientTape: no hay gradientes."""
    best_next = tf.constant(0.0, dtype=tf.float64) if done else tf.reduce_max(Q[s_next])
    td_target = r + gamma * best_next
    nuevo = Q[s, a] + alpha * (td_target - Q[s, a])
    Q[s, a].assign(nuevo)                          # mutacion explicita de la celda
    return Q

Q = q_update_tf(Q, s=14, a=3, r=1.0, s_next=15, done=True)
print(Q[14].numpy())   # [... 0.1]
```

La diferencia idiomatica: TensorFlow no permite `Q[s, a] += x` sobre una `Variable`; hay que usar `.assign` (o `scatter_nd_update` para lotes). Y, crucial: **no hay `tf.GradientTape`**. En el [camino 02](/clases/clase-31/practica/02-dqn-desde-cero), cuando `Q` pase a ser una red, el `GradientTape` reaparece para derivar la perdida TD respecto a los pesos. Aqui no hace falta porque no hay pesos.

### 3.3 JAX — arrays inmutables y `.at[].set()`

```python
import jax
import jax.numpy as jnp

# En JAX los arrays son INMUTABLES: no existe Q[s,a] += x.
Q = jnp.zeros((N_STATES, N_ACTIONS), dtype=jnp.float64)

def q_update_jax(Q, s, a, r, s_next, done, alpha=0.1, gamma=0.95):
    """Misma Ecuacion (2), estilo funcional JAX: devuelve una tabla NUEVA."""
    best_next = jnp.where(done, 0.0, jnp.max(Q[s_next]))
    td_target = r + gamma * best_next
    nuevo = Q[s, a] + alpha * (td_target - Q[s, a])
    return Q.at[s, a].set(nuevo)                   # update funcional: Q nuevo, no muta

Q = q_update_jax(Q, s=14, a=3, r=1.0, s_next=15, done=True)
print(Q[14])   # [... 0.1]
```

JAX es el mas distinto, y por la razon mas instructiva: sus arrays son **inmutables** (como exige la programacion funcional pura que habilita `jit`, `grad` y `vmap`). No puedes mutar una celda; `Q.at[s, a].set(valor)` devuelve **una tabla nueva** con esa celda cambiada. Usamos `jnp.where(done, ...)` en vez de un `if` de Python porque, si quisieramos compilar con `@jax.jit`, el `if` sobre un valor trazado fallaria. Esa misma maquinaria funcional es la que hace a JAX brillar en DQN y en MAML, donde sirve diferenciar a traves de la actualizacion.

### 3.4 La tabla comparativa

| Concepto | NumPy | PyTorch | TensorFlow | JAX |
|---|---|---|---|---|
| Crear la tabla Q | `np.zeros((16,4))` | `torch.zeros(16,4)` | `tf.Variable(tf.zeros(...))` | `jnp.zeros((16,4))` |
| `max_a' Q(s',a')` | `Q[s2].max()` | `Q[s2].max()` | `tf.reduce_max(Q[s2])` | `jnp.max(Q[s2])` |
| Actualizar celda | `Q[s,a] += x` | `Q[s,a] += x` | `Q[s,a].assign(...)` | `Q.at[s,a].set(...)` |
| Mutabilidad | mutable | mutable | mutable (`Variable`) | **inmutable** (funcional) |
| ¿Autograd usado? | no existe | **no** (`requires_grad=False`) | **no** (sin `GradientTape`) | **no** (sin `grad`) |

La conclusion es deliberada: en Q-Learning tabular, **los cuatro hacen lo mismo y ninguno usa su capacidad estrella** (autodiferenciacion). El framework es decorado. Eso cambia de raiz en el camino siguiente.

---

## 4. Por que tabular no escala: la maldicion de la dimensionalidad

La tabla Q tiene una entrada por cada par $(s, a)$. Funciona de maravillas con 16 estados. Pero la cantidad de estados crece **exponencialmente** con la cantidad de variables que describen el mundo —la **maldicion de la dimensionalidad**:

| Problema | Estados | ¿Tabla viable? |
|---|---:|---|
| Nuestro grid $4\times4$ | $16$ | si, trivial |
| Tic-tac-toe | $\sim 5\,000$ | si |
| Backgammon | $\sim 10^{20}$ | no |
| Ajedrez | $\sim 10^{47}$ | imposible |
| Go ($19\times19$) | $\sim 10^{170}$ | imposible |
| **Atari desde pixeles** ($210\times160$ RGB) | $256^{100\,800}$ | absurdo |

Dos problemas, no uno:

1. **Memoria.** No cabe una tabla de $10^{47}$ filas en ninguna maquina concebible.
2. **Generalizacion.** Aunque cupiera, una tabla trata cada estado como **aislado**: aprender el valor de un estado no dice **nada** sobre estados parecidos. En Atari, dos frames que difieren en un pixel son entradas completamente distintas de la tabla, y el agente tendria que visitar cada uno por separado. Imposible.

La salida es **aproximacion de funciones**: en vez de almacenar $Q(s, a)$ en una tabla, **aprendemos una funcion parametrica** $Q_\theta(s, a)$ —una red neuronal con parametros $\theta$— que *generaliza* a estados nunca vistos. Eso es **Deep Q-Networks (DQN)**, el [camino 02](/clases/clase-31/practica/02-dqn-desde-cero). La regla de actualizacion sigue siendo la de Bellman, pero ahora se convierte en una **perdida** que minimizamos por gradiente descendente:

$$
\mathcal{L}(\theta) = \mathbb{E}\Big[\big(\,r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\,\big)^2\Big].
$$

El TD error pasa de ser una **asignacion** a ser un **residuo a minimizar**. Y para minimizarlo necesitamos derivar respecto a $\theta$ —**ahi** entran de verdad PyTorch, TensorFlow y JAX, con su autograd, sus optimizadores y sus grafos. Lo que en esta pagina era decorado, en DQN es esencial.

---

## 5. Tres ideas para llevarse

### 5.1 Exploracion vs explotacion

Es el dilema fundacional de RL y no tiene solucion perfecta. **Explotar** (tomar la mejor accion conocida) maximiza la recompensa **a corto plazo** pero arriesga quedar atrapado en un optimo local —la primera ruta "suficientemente buena". **Explorar** (probar acciones nuevas) sacrifica recompensa inmediata para descubrir si hay algo mejor. $\varepsilon$-greedy con decay es la respuesta pragmatica: explora agresivamente al principio (cuando no sabes nada) y explota cada vez mas a medida que aprendes. Un ejemplo cotidiano: elegir restaurante. Explotar es volver a tu favorito; explorar es probar el nuevo de la esquina. Si solo explotas, nunca descubres que el de la esquina es mejor; si solo exploras, nunca disfrutas tu favorito. Otras estrategias (Boltzmann/softmax, UCB, inicializacion optimista) refinan el *como* explorar, pero el dilema permanece.

### 5.2 On-policy vs off-policy

Q-Learning es **off-policy**, y el detalle vive en el $\max_{a'}$ de la Ecuacion (2). El target asume que en el estado siguiente actuaremos de forma **greedy** (la *politica objetivo*), aunque para llegar ahi hayamos tomado una accion exploratoria al azar (la *politica de comportamiento*). Comportamiento y aprendizaje estan **desacoplados**: el agente puede explorar como quiera —incluso con datos de otro agente o de experiencia pasada almacenada— y aun asi aprende la politica optima.

Su contraparte es **SARSA**, **on-policy**, que usa $Q(s', a')$ con la accion $a'$ **realmente tomada** por la politica $\varepsilon$-greedy, no la del $\max$:

$$
\underbrace{Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]}_{\text{Q-Learning: OFF-policy, usa el max}}
$$
$$
\underbrace{Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\, Q(s',a') - Q(s,a)]}_{\text{SARSA: ON-policy, usa la accion realmente tomada } a'}
$$

La diferencia practica es de **caracter**: SARSA aprende una politica que **tiene en cuenta su propia exploracion**, asi que tiende a ser mas "cautelosa" (evita estados donde un paso aleatorio seria catastrofico). Q-Learning aprende la politica optima **ignorando** que explora, asi que puede ser mas "agresiva". El ejemplo canonico es el *Cliff Walking* de Sutton y Barto: SARSA toma el camino seguro lejos del acantilado (porque sabe que a veces dara un paso al azar y caer cuesta caro), mientras Q-Learning aprende el camino optimo pegado al borde (y cae mas seguido **durante el entrenamiento**, pero su politica greedy final es mas corta).

Que Q-Learning sea off-policy es justo lo que habilita el **replay buffer** de DQN: reutilizar transiciones viejas, generadas por una politica antigua, para seguir aprendiendo la optima. Una politica on-policy como SARSA no puede hacerlo limpiamente, porque sus datos quedan "vencidos" en cuanto la politica cambia.

### 5.3 El camino hacia adelante

La tabla Q es donde **empieza** RL, no donde termina. El proximo paso reemplaza la tabla por una red neuronal —Deep Q-Networks— y con ello hereda dos problemas nuevos (la inestabilidad de entrenar con targets moviles, y la correlacion de muestras secuenciales) que DQN resuelve con dos trucos famosos: el **target network** y el **replay buffer**. Eso es exactamente el [camino 02](/clases/clase-31/practica/02-dqn-desde-cero), donde la matematica de Bellman que acabas de ver se mantiene **intacta** y solo cambia la forma de representar $Q$.

---

**Ver tambien:** [Teoria de la clase 31](/clases/clase-31) · [Camino 02 - DQN desde cero](/clases/clase-31/practica/02-dqn-desde-cero) · [Fundamento de Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado) · [Paper Q-Learning (Watkins y Dayan, 1992)](/papers/q-learning-watkins-1992).
