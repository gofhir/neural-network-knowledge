---
title: "Implementación de DQN"
weight: 2
---

Aquí Q-learning se vuelve código. Esta página desglosa la clase `DeepQNetwork` y el loop de entrenamiento, y contrasta cada decisión con los **datos reales** de la ejecución en Colab (DQN resolvió CartPole en el **episodio 85**, test perfecto de **210/210**).

## De la tabla a la red: por qué DQN

Q-learning tabular guarda un número $Q(s,a)$ por cada par estado-acción. Funciona con estados discretos y pocos. **CartPole lo rompe**: el estado es un vector continuo en $\mathbb{R}^4$ → hay infinitos estados, no hay tabla posible.

La idea de DQN ([Mnih et al. 2013](/papers/dqn-atari-mnih-2013) / [2015](/papers/dqn-nature-mnih-2015)) es reemplazar la tabla por un **aproximador de funciones**: una red neuronal $Q_\theta(s,a)$.

$$
\underbrace{s\in\mathbb{R}^4}_{\text{entrada}} \;\longrightarrow\; \text{red } Q_\theta \;\longrightarrow\; \underbrace{[\,Q_\theta(s,\text{izq}),\; Q_\theta(s,\text{der})\,]\in\mathbb{R}^2}_{\text{un Q-value por acción}}
$$

{{< callout type="warning" >}}
**Gotcha de diseño.** La red **no** toma $(s,a)$ y devuelve un escalar. Toma solo $s$ y devuelve **todos** los Q-values de golpe (un output por acción). Es deliberado: el $\arg\max_a Q(s,a)$ y el $\max_{a'} Q(s',a')$ del target se calculan con **un solo forward pass**, no $|\mathcal{A}|$ pases. Solo funciona con **acciones discretas y finitas** — por eso DQN no aplica directo a control continuo (para eso están DDPG, SAC, etc.).
{{< /callout >}}

## Hiperparámetros

```python
discount_rate() = 0.95    # γ  — horizonte efectivo ~20 pasos
learning_rate() = 0.001   # α  — default clásico de Adam
batch_size()    = 24      # mini-batch del replay
```

Envolver constantes en funciones (`def discount_rate(): return 0.95`) no aporta nada aquí — es un vestigio del material original de [Evan Hennis](https://github.com/ehennis/ReinforcementLearning). En código real serían constantes o config.

## La red: `build_model()`

```python
Input(shape=(4,)) → Dense(24, relu) → Dense(24, relu) → Dense(2, linear)
```

{{< callout type="error" >}}
**El error #1 de quien viene de clasificación.** La capa de salida es **`linear`, no `softmax`**. La red hace **regresión** de Q-values (valores reales, aquí hasta ~$\frac{1}{1-\gamma}=20$), *no* clasifica acciones. Por eso la pérdida es **MSE** (`mean_squared_error`), no cross-entropy. Softmax + cross-entropy aquí sería conceptualmente incorrecto.
{{< /callout >}}

Es una red diminuta: dos capas ocultas de 24 neuronas, ~700 parámetros. Suficiente para CartPole y, de paso, la razón por la que **la GPU no ayuda** en este lab (el cuello de botella son los miles de forward pass minúsculos y secuenciales, no el álgebra; el overhead de despacho CPU→GPU domina).

## La política: `get_action_epsilon_greedy()`

```python
if np.random.rand() <= self.epsilon:
    return random.randrange(self.nA)        # EXPLORAR
action_vals = self.model.predict(state)
return np.argmax(action_vals[0])            # EXPLOTAR
```

Implementa el dilema **exploración vs explotación**. Con probabilidad $\varepsilon$ tira una acción al azar (descubre estados nuevos); si no, explota su mejor estimación actual. Sin exploración, el agente nunca visitaría estados nuevos y se estancaría.

## El replay buffer: `store()`

```python
self.memory = deque([], maxlen=2500)
def store(self, s, a, r, ns, done):
    self.memory.append((s, a, r, ns, done))
```

El agente **genera su propio dataset**: cada paso produce una transición $(s, a, r, s', \text{done})$ que se guarda en un buffer circular. Ese buffer **es** el conjunto de entrenamiento de la red (esta es la respuesta literal a la [Tarea #1](04-actividades)).

{{< callout type="info" >}}
**Por qué el replay buffer es esencial** (la contribución #1 de Mnih 2015). Entrenar con transiciones en el orden que ocurren violaría el supuesto i.i.d. del SGD: (1) **correlación temporal** entre pasos consecutivos → gradientes correlacionados → la red oscila; (2) **distribución no estacionaria** → sobreajuste a la región reciente y olvido. El **muestreo aleatorio** de mini-batches des-correlaciona las transiciones y reutiliza cada experiencia muchas veces.
{{< /callout >}}

## El corazón: `learn()`

Aquí Bellman se vuelve un `fit`:

```python
minibatch = random.sample(self.memory, batch_size)   # 1. muestreo aleatorio
st_predict  = self.model.predict(st)                 # 2. Q(s)  batcheado
nst_predict = self.model.predict(nst)                #    Q(s') batcheado
for state, action, reward, nstate, done in minibatch:
    if done:  target = reward                        # 3. estado terminal → sin futuro
    else:     target = reward + self.gamma * np.amax(nst_predict[index])   # bootstrapping
    target_f = st_predict[index]
    target_f[action] = target                        # 4. sobrescribe SOLO la acción tomada
    ...
self.model.fit(x_reshape, y_reshape, epochs=1)       # 5. un paso de SGD
if self.epsilon > self.epsilon_min:
    self.epsilon *= self.epsilon_decay               # 6. decae epsilon
```

El **target de Bellman** por transición:

$$
y = \begin{cases} r & \text{si done} \\ r + \gamma \max_{a'} Q_\theta(s',a') & \text{si no} \end{cases}
$$

y la pérdida que se minimiza:

$$
\mathcal{L}(\theta) = \sum_{(s,a,r,s')}\big( y - Q_\theta(s,a) \big)^2
$$

{{< callout type="info" >}}
**El truco del paso 4.** La red predice 2 Q-values, pero solo *uno* (el de la acción tomada) debe cambiar. Se copia la predicción actual `target_f = [Q(s,izq), Q(s,der)]` y se sobrescribe **solo** `target_f[action] = target`. Así el gradiente por las acciones no tomadas es cero (predicción = etiqueta) y solo se ajusta $Q(s,a)$. Es cómo se hace un update de Q-learning con una red de salida múltiple.
{{< /callout >}}

{{< callout type="error" >}}
**El gotcha más importante del notebook: no hay target network.** El target $y = r + \gamma\max_{a'}Q(s',a')$ se calcula con la **misma** red que se está actualizando (`self.model`). Estás persiguiendo un blanco móvil → inestabilidad. Mnih 2015 lo arregla con una **copia congelada** $Q_{\theta^-}$ (target network) actualizada cada $C$ pasos. El propio notebook admite ser "una implementación lejos de ser eficiente, con fines educativos" — esta es la simplificación. Funciona en CartPole por ser fácil. En [Experimentos](03-experimentos-y-analisis) lo agregamos y medimos su efecto.

Relacionado: el $\max_{a'}$ sobre estimaciones ruidosas tiene **sesgo de sobreestimación** ($\mathbb{E}[\max]\ge\max\mathbb{E}$, por Jensen). Es lo que corrige [Double DQN](/papers/double-dqn-van-hasselt-2015).
{{< /callout >}}

## El loop de entrenamiento — contra los datos reales

El loop hace lo canónico: por episodio resetea, y por paso elige acción ε-greedy → ejecuta → guarda transición → `learn()`. Para cuando `avg(últimos 10) > 195`.

![Curva de recompensa de DQN en CartPole](/laboratorios/lab-31/reward-curve.png)

**Lo que midió esta ejecución real:**

| Métrica | Valor medido |
|---------|--------------|
| Episodios hasta resolver | **85** |
| Media reward primeros 50 eps | **11.7** (≈ nivel aleatorio) |
| Primer episodio con 210 | **76** |
| Media últimos 10 eps de train | **204.6** |
| **Test (100 eps, ε=0)** | **210.0** (perfecto) |

{{< callout type="warning" >}}
**Gotcha revelado por los datos: epsilon decae por paso, no por episodio.** `epsilon *= 0.995` se aplica **dentro de `learn()`**, o sea una vez por **paso de gradiente**. Como `learn()` corre cada paso, epsilon cae de 1.0 al piso 0.001 en ~1400 pasos → **hacia el episodio ~65 la exploración prácticamente terminó** (epsilon en ep 50 = 0.073; en ep 66 ya < 0.01). Lectura profunda: **el tramo final de mejora (episodios ~65→85) ocurre con $\varepsilon\approx 0$**, es decir con explotación pura. El agente no sigue mejorando porque explore más, sino porque el **replay buffer** le permite re-aprender de experiencias pasadas. Evidencia directa del valor del experience replay.
{{< /callout >}}

Otro detalle: el tope de 210 pasos (`range(210)`) es **más bajo** que el límite de 500 de `CartPole-v1`. Por eso el test alcanza exactamente 210 (no 500): el código corta antes. El agente en realidad podría equilibrar mucho más.

## La curva de pérdida

![Curva de pérdida de DQN](/laboratorios/lab-31/loss-curve.png)

La loss **no baja monótonamente** como en aprendizaje supervisado: sube y baja porque el **target se mueve** (bootstrapping — el objetivo depende de la propia red, que cambia). Que la política mejore mientras la loss oscila es normal en RL y refuerza por qué la estabilización (target network) importa. *(El notebook no registra la loss; esta curva proviene de una reproducción local que la expone.)*

## Fase de test

El test fija **ε=0** (explotación pura, `get_best_action`) y **no entrena** (sin `learn()`). Evalúa la política "en frío": **210/210 en los 100 episodios** → la política es robusta, no un golpe de suerte. En la curva es la meseta plana perfecta tras la línea `training end`.

---

**Siguiente:** [Experimentos propios y análisis](03-experimentos-y-analisis) — cuatro ablations que validan con datos las afirmaciones de esta página.
